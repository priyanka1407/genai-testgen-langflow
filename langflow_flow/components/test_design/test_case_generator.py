"""Langflow node 2/4 - requirement-aligned test cases + coverage report."""
import json

# Langflow loads custom nodes by running only plain top-level imports,
# so they must not sit inside try/except. Written for Langflow >= 1.6 (lfx);
# Component is imported first to avoid a circular import inside lfx.
from lfx.custom.custom_component.component import Component
from lfx.io import BoolInput, DataInput, DropdownInput, IntInput, Output, SecretStrInput, StrInput
from lfx.schema.data import Data
from lfx.schema.message import Message

from testgen import coverage, generator, ingest
from testgen.lf_support import API_KEY_INFO, MODELS, as_dicts, llm_from
from testgen.models import Requirement
from testgen.state import State


class TestCaseGeneratorComponent(Component):
    display_name = "Test Case Generator"
    description = "Requirement-aligned test cases with coverage gap closing (positive / negative / boundary)."
    icon = "list-checks"
    name = "TestgenTestCaseGenerator"

    inputs = [
        DataInput(name="requirements", display_name="Requirements", is_list=True),
        StrInput(name="data_path", display_name="Data sources folder", value="samples/data",
                 info="CSV/JSON datasets - only their schema profile is sent to the LLM"),
        IntInput(name="gap_rounds", display_name="Coverage gap rounds", value=2),
        StrInput(name="state_dir", display_name="State folder", value="state",
                 info="Memory between runs: only new/changed requirements are regenerated. "
                      "Use one folder per epic or document set, e.g. state/PAY-100"),
        BoolInput(name="full", display_name="Regenerate everything", value=False, advanced=True,
                  info="Ignore the saved state and regenerate every requirement"),
        BoolInput(name="verify", display_name="Verify with judge", value=True,
                  info="Independent LLM review; coverage counts only judge-verified criteria"),
        SecretStrInput(name="api_key", display_name="API Key (Claude or Gemini)", required=False, info=API_KEY_INFO),
        DropdownInput(name="model", display_name="Model", options=MODELS, value=MODELS[0], combobox=True,
                      info="gemini-* uses Gemini, claude-* uses Claude. You can also type a model id."),
        BoolInput(name="offline", display_name="Offline (no LLM)", value=False, advanced=True),
    ]
    outputs = [
        # group_outputs=True shows both outputs as separate dots (default is one dot + dropdown)
        Output(display_name="Test Cases", name="test_cases", method="generate", group_outputs=True),
        Output(display_name="Coverage Report", name="coverage_report", method="coverage_report",
               group_outputs=True),
    ]

    def _run(self):
        # Both outputs share one generation run
        if getattr(self, "_cache", None) is None:
            reqs = [Requirement(**d) for d in as_dicts(self.requirements)]
            sources, _ = ingest.consolidate(ingest.load_data_sources([self.data_path]) if self.data_path else {})
            profile = ingest.profile_sources(sources)
            llm = llm_from(self)
            judge = llm_from(self) if self.verify else None
            self._models = (llm.name, judge.name if judge else "off")   # e.g. gemini-3.8-flash / offline-heuristic
            st = State.load(self.state_dir)
            statuses, removed = st.diff_requirements(reqs)               # new / changed / unchanged per requirement
            inc = generator.incremental_generate(llm, reqs, statuses, removed, profile, st, judge,
                                                 max_gap_rounds=self.gap_rounds, full=self.full)
            failed = {rid: r.error for rid, r in inc.results.items() if r.error}
            if inc.results and len(failed) == len(inc.results):        # nothing generated -> show the real cause
                raise RuntimeError(f"Test case generation failed for every requirement: {next(iter(failed.values()))}")
            st.commit(reqs, inc.cases, inc.verdicts, inc.obsolete, failed=set(failed))
            st.save()                                                  # remember this run for the next one
            self._rejected = [t for r in inc.results.values() for t in r.rejected_by_judge]
            self._failed, self._changes = failed, (inc.statuses, removed, inc.obsolete)
            self._cache = (reqs, inc.cases, inc.verdicts if self.verify else None)
        return self._cache

    def generate(self) -> list[Data]:
        _, cases, _ = self._run()
        failed = getattr(self, "_failed", {})
        gen_model, judge_model = self._models
        self.status = (f"{len(cases)} test cases | {self._change_summary()} | generator: {gen_model} | judge: {judge_model}"
                       + (f" - FAILED for {sorted(failed)}: {next(iter(failed.values()))}" if failed else ""))
        return [Data(data=c.model_dump()) for c in cases]

    def coverage_report(self) -> Message:
        reqs, cases, verdicts = self._run()
        rows = coverage.analyse(reqs, cases, verdicts)
        lines = [f"| {r.req_id} | {r.risk} | {r.percent}% | {r.claimed_but_unverified or '-'} | "
                 f"{', '.join(r.test_types)} | {', '.join(r.missing_types) or '-'} |" for r in rows]
        table = ("| Requirement | Risk | Verified coverage | Claimed, not verified | Types | Missing |\n"
                 "|---|---|---|---|---|---|\n" + "\n".join(lines))
        gen_model, judge_model = self._models
        header = (f"**Generator:** `{gen_model}`  |  **Judge:** `{judge_model}`\n\n"
                  f"**This run:** {self._change_summary()}\n\n")
        return Message(text=header + table + "\n\n" + self._verdict_section(cases, verdicts)
                       + f"\n\n```json\n{json.dumps(coverage.summary(rows), indent=2)}\n```")

    def _change_summary(self) -> str:
        statuses, removed, obsolete = self._changes
        by = {k: sorted(r for r, v in statuses.items() if v == k) for k in ("new", "changed", "unchanged")}
        parts = [f"{k} {len(v)}" + (f" {v}" if v and k != "unchanged" else "") for k, v in by.items()]
        if removed:
            parts.append(f"removed {removed} ({len(obsolete)} test cases obsolete)")
        return ", ".join(parts)

    def _verdict_section(self, cases, verdicts) -> str:
        """What the judge decided: counts, plus every case it did not fully accept and why."""
        if verdicts is None:
            return "**Judge:** off - coverage above is what the generator CLAIMS, not verified."
        counts = {k: sum(1 for v in verdicts.values() if v.verdict == k) for k in ("accept", "revise")}
        rejected = getattr(self, "_rejected", [])
        text = (f"**Judge verdicts:** {counts['accept']} accepted, {counts['revise']} need revision, "
                f"{len(rejected)} rejected and removed")
        flagged = [(c, verdicts[c.tc_id]) for c in cases
                   if c.tc_id in verdicts and verdicts[c.tc_id].verdict != "accept"]
        if flagged:
            text += ("\n\n| Test case | Verdict | Claimed criteria | Verified criteria | Reason |\n"
                     "|---|---|---|---|---|\n"
                     + "\n".join(f"| {c.tc_id} | {v.verdict} | {c.covers_criteria} | {v.verified_criteria} "
                                 f"| {v.reason} |" for c, v in flagged))
        return text
