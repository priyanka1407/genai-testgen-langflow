"""
Stage 1 + 2: requirements extraction and requirement-aligned test case generation,
with judge verification and a coverage-driven gap-closing loop.

Per requirement:
    generate -> dedupe -> judge (verify) -> drop rejected
      -> gaps measured on VERIFIED coverage -> re-prompt only for the gaps (max N rounds)

Generation is done PER REQUIREMENT (not one giant prompt) so that each call is
small and never truncated, gaps are re-prompted precisely, and a failure in one
requirement is retried and isolated instead of failing the whole run.
"""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from . import coverage, prompts
from .ingest import format_documents
from .llm import LLM, QuotaExhausted
from .models import CaseVerdict, Requirement, RequirementSet, TestCase, TestCaseSet
from .verify import verify_cases

log = logging.getLogger("testgen.generator")


@dataclass
class ReqResult:
    req_id: str
    cases: list[TestCase] = field(default_factory=list)
    verdicts: dict[str, CaseVerdict] = field(default_factory=dict)
    generated: int = 0           # before dedupe / judge
    duplicates: int = 0
    rejected_by_judge: list[str] = field(default_factory=list)
    gap_rounds: int = 0
    error: str | None = None


def extract_requirements(llm: LLM, documents: list[tuple[str, str]],
                         previous: list[Requirement] | None = None) -> list[Requirement]:
    prev = json.dumps([{k: r.model_dump()[k] for k in ("module", "title", "acceptance_criteria", "business_rules")}
                       for r in previous or []], indent=2)
    prompt = prompts.EXTRACT_REQUIREMENTS.format(documents=format_documents(documents), previous=prev or "[]")
    result = llm.generate(prompts.SYSTEM_QA_ARCHITECT, prompt, RequirementSet, {"documents": documents})
    log.info("extracted %d requirements", len(result.requirements))
    return result.requirements


def _generate_once(llm: LLM, req: Requirement, data_profile: dict, start: int,
                   gap: dict | None, existing: list[TestCase]) -> list[TestCase]:
    gap_text = ""
    if gap:
        gap_text = prompts.GAP_INSTRUCTIONS.format(
            uncovered=gap["uncovered"], missing_types=gap["missing_types"],
            existing_titles=[c.title for c in existing],
        )
    prompt = prompts.GENERATE_TEST_CASES.format(
        requirement_json=req.model_dump_json(indent=2),
        data_profile=json.dumps(data_profile, indent=2),
        gap_instructions=gap_text, start_index=start,
    )
    ctx = {"requirement": req, "start_index": start, "gap": gap, "data_profile": data_profile}
    cases = llm.generate(prompts.SYSTEM_QA_ARCHITECT, prompt, TestCaseSet, ctx).test_cases
    # Temporary run-local ids; State.reconcile() assigns the stable ones
    for i, c in enumerate(cases):
        c.req_id = req.req_id
        c.tc_id = f"TMP-{req.req_id}-{start + i:03d}"
    return cases


def generate_for_requirement(llm: LLM, req: Requirement, data_profile: dict,
                             judge: LLM | None = None, max_gap_rounds: int = 2) -> ReqResult:
    res = ReqResult(req_id=req.req_id)
    cases: list[TestCase] = []
    verdicts: dict[str, CaseVerdict] = {}
    gap = None
    for round_no in range(max_gap_rounds + 1):
        new = _generate_once(llm, req, data_profile, res.generated + 1, gap, cases)
        res.generated += len(new)
        merged, dropped = coverage.dedupe(cases + new)
        res.duplicates += len(dropped)
        known = {c.tc_id for c in cases}
        new = [c for c in merged if c.tc_id not in known]
        if judge:
            v = verify_cases(judge, req, new)
            rejected = [tid for tid, x in v.items() if x.verdict == "reject"]
            res.rejected_by_judge += rejected
            new = [c for c in new if c.tc_id not in rejected]
            verdicts |= {tid: x for tid, x in v.items() if tid not in rejected}
        cases += new
        row = coverage.analyse_requirement(req, cases, verdicts if judge else None)
        if not row.uncovered_criteria and not row.missing_types:
            break
        if round_no == max_gap_rounds:
            log.warning("%s still has gaps after %d rounds: uncovered=%s missing=%s",
                        req.req_id, max_gap_rounds, row.uncovered_criteria, row.missing_types)
            break
        res.gap_rounds += 1
        gap = {"uncovered": row.uncovered_criteria, "missing_types": row.missing_types}
        log.info("%s gap round %d: %s", req.req_id, round_no + 1, gap)
    res.cases, res.verdicts = cases, verdicts
    return res


def _safe(llm, req, profile, judge, rounds, retries=2) -> ReqResult:
    for attempt in range(retries + 1):
        try:
            return generate_for_requirement(llm, req, profile, judge, rounds)
        except QuotaExhausted as e:   # retrying cannot help - fail this requirement immediately
            log.warning("%s: %s", req.req_id, e)
            return ReqResult(req_id=req.req_id, error=str(e))
        except Exception as e:  # noqa: BLE001 - isolate one requirement's failure
            log.warning("%s attempt %d failed: %s", req.req_id, attempt + 1, e)
            if attempt == retries:
                return ReqResult(req_id=req.req_id, error=f"{type(e).__name__}: {e}")
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def generate_test_cases(llm: LLM, reqs: list[Requirement], data_profile: dict, judge: LLM | None = None,
                        workers: int = 4, max_gap_rounds: int = 2) -> dict[str, ReqResult]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = pool.map(lambda r: _safe(llm, r, data_profile, judge, max_gap_rounds), reqs)
        return {r.req_id: r for r in results}


@dataclass
class IncrementalResult:
    cases: list[TestCase]
    verdicts: dict[str, CaseVerdict]
    obsolete: dict[str, str]            # tc_id -> reason (changed / removed requirement)
    results: dict[str, ReqResult]       # only requirements that were (re)generated this run
    statuses: dict[str, str]            # req_id -> new | changed | unchanged
    removed: list[str]

    @property
    def failed(self) -> set[str]:
        return {rid for rid, r in self.results.items() if r.error}


def incremental_generate(llm: LLM, reqs: list[Requirement], statuses: dict[str, str], removed: list[str],
                         data_profile: dict, st, judge: LLM | None = None, workers: int = 4,
                         max_gap_rounds: int = 2, full: bool = False) -> IncrementalResult:
    """Regenerate only new/changed requirements; reuse everything else from state `st`.
    Shared by the CLI pipeline and the Langflow Test Case Generator node."""
    if full:
        statuses = {k: "changed" if v == "unchanged" else v for k, v in statuses.items()}
    todo = [r for r in reqs if statuses.get(r.req_id) != "unchanged"]
    log.info("requirements: %s; removed: %s",
             {k: sum(1 for v in statuses.values() if v == k) for k in ("new", "changed", "unchanged")}, removed)
    results = generate_test_cases(llm, todo, data_profile, judge, workers, max_gap_rounds) if todo else {}
    cases: list[TestCase] = []
    verdicts: dict[str, CaseVerdict] = {}
    obsolete: dict[str, str] = {}
    for r in reqs:
        res = results.get(r.req_id)
        if res is None or res.error:  # unchanged, or failed -> keep the previous suite for it
            cases += st.cases_for(r.req_id)
            verdicts |= {tid: CaseVerdict(**v) for tid, v in st.verdicts.items() if tid.startswith(f"TC-{r.req_id}-")}
            continue
        new_cases, new_verdicts, gone = st.reconcile(r.req_id, res.cases, res.verdicts)
        for tid in [c.tc_id for c in new_cases]:   # requirement changed -> old reviewer edits are stale
            st.overrides.pop(tid, None)
        cases += new_cases
        verdicts |= new_verdicts
        obsolete |= {tid: f"{r.req_id} changed" for tid in gone}
    for rid in removed:
        obsolete |= {c.tc_id: f"{rid} removed" for c in st.cases_for(rid)}
    return IncrementalResult(cases, verdicts, obsolete, results, statuses, removed)
