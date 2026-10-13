"""Langflow node 4/4 - publish to JIRA for human review (idempotent create-or-update)."""
from pathlib import Path

# Langflow loads custom nodes by running only plain top-level imports,
# so they must not sit inside try/except. Written for Langflow >= 1.6 (lfx);
# Component is imported first to avoid a circular import inside lfx.
from lfx.custom.custom_component.component import Component
from lfx.io import BoolInput, DataInput, Output, StrInput
from lfx.schema.message import Message

from testgen import exporters, jira_publisher
from testgen.lf_support import as_dicts
from testgen.models import TestCase
from testgen.state import State


class JiraReviewPublisherComponent(Component):
    display_name = "JIRA Review Publisher"
    description = "Publishes generated test cases to JIRA as 'pending-review' (human-in-the-loop)."
    icon = "user-check"
    name = "TestgenJiraPublisher"

    inputs = [
        DataInput(name="test_cases", display_name="Test Cases", is_list=True),
        BoolInput(name="dry_run", display_name="Dry run", value=True,
                  info="Use the local JSON JIRA in <state>/fake_jira.json instead of real JIRA"),
        StrInput(name="state_dir", display_name="State folder", value="state",
                 info="Holds the test case -> issue mapping so reruns update instead of duplicating"),
        StrInput(name="out_dir", display_name="Output folder", value="output"),
    ]
    outputs = [Output(display_name="Publish Result", name="result", method="publish")]

    def publish(self) -> Message:
        cases = [TestCase(**d) for d in as_dicts(self.test_cases)]
        out = Path(self.out_dir)
        exporters.write_test_cases(cases, out)
        st = State.load(self.state_dir)
        mode = "dry_run" if self.dry_run else "jira"
        client = jira_publisher.get_client(not self.dry_run, self.state_dir)
        # test cases of removed / changed requirements that are not yet marked obsolete in JIRA
        obsolete = [t for t in st.obsolete if t in st.jira[mode] and not st.jira[mode][t].get("obsolete")]
        results = jira_publisher.sync(cases, obsolete, client, st.jira[mode])
        st.save()
        exporters.audit(out, "jira_sync", source="langflow", mode=mode, issues=len(results))
        ops = {op: sum(1 for r in results if r["op"] == op) for op in {r["op"] for r in results}}
        self.status = f"JIRA: {ops}"
        return Message(text="\n".join(f"{r['tc_id']}: {r['issue_key']} {r['op']}" for r in results))
