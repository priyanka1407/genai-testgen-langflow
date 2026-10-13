"""
End-to-end orchestrator (also what the Langflow components call).

    python -m testgen.pipeline --docs samples/docs --data samples/data
    python -m testgen.pipeline ... --offline            # no API key needed
    python -m testgen.pipeline ... --model gemini-3.8-flash   # Gemini (GEMINI_API_KEY)
    python -m testgen.pipeline ... --jira               # real JIRA (default: local dry-run JIRA)
    python -m testgen.pipeline ... --approved-only      # runner data only for approved, unchanged cases
    python -m testgen.pipeline ... --full               # ignore state, regenerate everything
    python -m testgen.pipeline --epic PAY-100 --data samples/data              # epic + stories (local export)
    python -m testgen.pipeline --epic PAY-100 --epic-source jira ...          # epic + stories from live JIRA

Run it on every release: only new/changed requirements hit the LLM, ids stay stable,
JIRA issues are updated instead of duplicated, and reviewer edits are preserved.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from . import coverage, evaluate, exporters, generator, ingest, jira_publisher, jira_reader, prompts, testdata
from .env import load_env_file
from .llm import default_judge_model, get_llm
from .models import CaseVerdict, CoverageRow, Requirement, TestCase
from .state import State

log = logging.getLogger("testgen.pipeline")


@dataclass
class PipelineResult:
    requirements: list[Requirement]
    test_cases: list[TestCase]
    coverage: list[CoverageRow]
    summary: dict
    metrics: dict
    statuses: dict[str, str]
    jira: list[dict] = field(default_factory=list)
    reviews: dict = field(default_factory=dict)


def run(docs: list[str], data: list[str], out: str, state_dir: str = "state", offline: bool | None = None,
        rows_per_case: int = 3, jira: bool = False, approved_only: bool = False, max_gap_rounds: int = 2,
        workers: int = 4, verify: bool = True, full: bool = False, golden: str | None = "evals/golden.json",
        api_key: str | None = None, model: str | None = None, judge_model: str | None = None,
        epic: str | None = None, epic_source: str = "file", epic_folder: str = "samples/jira") -> PipelineResult:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    st = State.load(state_dir)
    llm = get_llm(offline, api_key=api_key, model=model)
    judge = get_llm(offline, api_key=api_key, model=judge_model or model or default_judge_model()) if verify else None

    # 1. data sources (consolidated) - only their profile ever reaches the LLM
    raw = ingest.load_data_sources(data) if data else {}
    sources, relations = ingest.consolidate(raw)
    profile = ingest.profile_sources(sources)

    # 2. requirements with stable ids + change detection
    if epic:
        # JIRA epic: every story is a requirement whose id is the story key -> per-story change detection
        reqs, epic_info = jira_reader.read_epic(epic, jira_reader.get_source(epic_source, epic_folder), llm=llm)
        if not reqs:
            raise SystemExit(f"Epic {epic} has no stories with content")
        statuses, removed = st.diff_requirements(reqs)
        exporters.audit(out_dir, "ingest", model=llm.name, judge=judge.name if judge else None,
                        prompt_version=prompts.PROMPT_VERSION, epic=epic_info,
                        datasets={k: len(v) for k, v in raw.items()}, joins=relations)
    else:
        documents = ingest.load_documents(docs or [])
        if not documents:
            raise SystemExit(f"No documents found in {docs}")
        exporters.audit(out_dir, "ingest", model=llm.name, judge=judge.name if judge else None,
                        prompt_version=prompts.PROMPT_VERSION, documents=exporters.doc_hashes(documents),
                        datasets={k: len(v) for k, v in raw.items()}, joins=relations)
        reqs = generator.extract_requirements(llm, documents, previous=st.previous_requirements())
        statuses, removed = st.assign_ids(reqs)

    # 3. generate + verify only what changed
    inc = generator.incremental_generate(llm, reqs, statuses, removed, profile, st, judge, workers,
                                         max_gap_rounds, full)
    results, cases, verdicts, obsolete, statuses = inc.results, inc.cases, inc.verdicts, inc.obsolete, inc.statuses

    # 4. JIRA: pull reviews (reviewer edits win), then push create/update/obsolete
    mode = "jira" if jira else "dry_run"
    client = jira_publisher.get_client(jira, state_dir)
    by_id = {c.tc_id: c for c in cases}
    reviews = jira_publisher.pull_reviews(client, st.jira[mode], by_id)
    for tid, rv in reviews.items():
        if rv["edited_case"] is not None and tid in by_id and results.get(by_id[tid].req_id) is None:
            st.overrides[tid] = rv["edited_case"].model_dump()
            by_id[tid] = rv["edited_case"]
    cases = [by_id[c.tc_id] for c in cases]
    flags = {tid: [f"judge: {v.verdict} - {v.reason}"] for tid, v in verdicts.items() if v.verdict != "accept"}
    sync_results = jira_publisher.sync(cases, list(obsolete), client, st.jira[mode], flags, skip_ids=set(st.overrides))
    reviews = jira_publisher.pull_reviews(client, st.jira[mode], {c.tc_id: c for c in cases})  # post-push status
    review_view = {tid: {k: v for k, v in rv.items() if k != "edited_case"} for tid, rv in reviews.items()}
    st.reviews = review_view
    exporters.audit(out_dir, "jira_sync", mode=mode,
                    ops={op: sum(1 for x in sync_results if x["op"] == op) for op in {x["op"] for x in sync_results}})

    # 5. approval gate -> synthetic data for the runner (rejected cases never run)
    runnable = [c for c in cases if review_view.get(c.tc_id, {}).get("status") != "rejected"]
    if approved_only:
        runnable = [c for c in runnable if review_view.get(c.tc_id, {}).get("status") == "approved"]
        exporters.audit(out_dir, "approval_gate", approved=len(runnable), total=len(cases))
    dataset = testdata.build_dataset(runnable, sources, rows_per_case)
    n_rows = exporters.write_testdata(dataset, runnable, out_dir)

    # 6. outputs, coverage (judge-verified), eval metrics, state
    exporters.write_requirements(reqs, out_dir)
    exporters.write_test_cases(cases, out_dir)
    exporters.write_json(out_dir, "verification.json", {k: v.model_dump() for k, v in verdicts.items()})
    run_stats = {
        "statuses": statuses, "removed": removed, "obsolete": obsolete,
        "requirements": {rid: {"generated": r.generated, "duplicates": r.duplicates,
                               "rejected_by_judge": r.rejected_by_judge, "gap_rounds": r.gap_rounds,
                               "error": r.error} for rid, r in results.items()},
    }
    exporters.write_json(out_dir, "run_stats.json", run_stats)
    exporters.write_json(out_dir, "reviews.json", review_view)
    exporters.write_json(out_dir, "jira_sync.json", sync_results)
    rows = coverage.analyse(reqs, cases, verdicts if verify else None)
    summary = exporters.write_coverage(rows, coverage.summary(rows), len(cases), n_rows, out_dir)
    gold = json.loads(Path(golden).read_text(encoding="utf-8")) if golden and Path(golden).exists() else None
    metrics = evaluate.compute(reqs, cases, verdicts, run_stats, review_view, gold)
    exporters.write_json(out_dir, "eval_metrics.json", metrics)
    st.commit(reqs, cases, verdicts, obsolete, failed=inc.failed)
    st.save()
    exporters.audit(out_dir, "complete", statuses=statuses, test_cases=len(cases), data_rows=n_rows,
                    verified_coverage=summary["verified_coverage_pct"], obsolete=len(obsolete))
    return PipelineResult(reqs, cases, rows, summary, metrics, statuses, sync_results, review_view)


def main() -> None:
    ap = argparse.ArgumentParser(description="GenAI test case + synthetic test data generator")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--docs", nargs="+", help="Release notes / requirement spec files or folders")
    src.add_argument("--epic", help="JIRA epic key, e.g. PAY-100 - reads the epic and its user stories")
    ap.add_argument("--epic-source", choices=["file", "jira"], default="file",
                    help="file = samples/jira/<EPIC>.json export (demo), jira = live JIRA via JIRA_URL / JIRA_API_TOKEN")
    ap.add_argument("--epic-folder", default="samples/jira")
    ap.add_argument("--data", nargs="*", default=[], help="CSV/JSON data source files or folders")
    ap.add_argument("--out", default="output")
    ap.add_argument("--state", help="Folder holding ids, history and JIRA mapping (default: state, or state/<EPIC>)")
    ap.add_argument("--offline", action="store_true", help="Use the heuristic backend (no API key)")
    ap.add_argument("--rows-per-case", type=int, default=3)
    ap.add_argument("--jira", action="store_true", help="Use real JIRA (default: local dry-run JIRA)")
    ap.add_argument("--approved-only", action="store_true", help="Only build data for JIRA-approved cases")
    ap.add_argument("--no-verify", action="store_true", help="Skip the judge (coverage becomes self-reported)")
    ap.add_argument("--full", action="store_true", help="Regenerate every requirement, not just changed ones")
    ap.add_argument("--gap-rounds", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--golden", default="evals/golden.json")
    ap.add_argument("--model", help="e.g. claude-opus-5-5 or gemini-3.8-flash (default: TESTGEN_MODEL or claude-opus-5-5)")
    ap.add_argument("--judge-model", help="model for the judge (default: same as --model)")
    a = ap.parse_args()
    load_env_file()   # GEMINI_API_KEY, TESTGEN_MODEL, JIRA_* from .env (shell variables win)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    state_dir = a.state or (f"state/{a.epic}" if a.epic else "state")
    golden = a.golden if not a.epic or a.golden != "evals/golden.json" else None   # golden set is for samples/docs
    res = run(a.docs, a.data, a.out, state_dir, offline=a.offline or None, rows_per_case=a.rows_per_case,
              jira=a.jira, approved_only=a.approved_only, max_gap_rounds=a.gap_rounds, workers=a.workers,
              verify=not a.no_verify, full=a.full, golden=golden, model=a.model, judge_model=a.judge_model,
              epic=a.epic, epic_source=a.epic_source, epic_folder=a.epic_folder)
    s, m = res.summary, res.metrics
    counts = {k: sum(1 for v in res.statuses.values() if v == k) for k in ("new", "changed", "unchanged")}
    ops = {op: sum(1 for x in res.jira if x["op"] == op) for op in {x["op"] for x in res.jira}}
    print(f"\nRequirements: {counts}   Test cases: {s['test_cases']}   Data rows: {s['test_data_rows']}")
    print(f"Coverage verified by judge: {s['verified_coverage_pct']}%  "
          f"(claimed {m['claimed_coverage_pct']}%, over-claim rate {m['overclaim_rate']})  "
          f"gaps: {s['requirements_with_gaps'] or 'none'}")
    if m.get("golden_recall") is not None:
        print(f"Golden must-have recall: {m['golden_recall']}  missed: {m.get('golden_missed') or 'none'}")
    print(f"JIRA: {ops or 'no changes'}")
    print(f"Estimated manual effort saved: {s['estimated_manual_hours_saved']} h  -> outputs in {a.out}/")


if __name__ == "__main__":
    main()
