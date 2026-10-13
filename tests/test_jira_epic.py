"""JIRA epic reader: criteria parsing, ADF, risk, REST paths (fake HTTP), and per-story incremental runs."""
import json
import shutil
from pathlib import Path

import pytest

from testgen import jira_reader, pipeline
from testgen.models import Requirement
from testgen.state import State

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ parsing
@pytest.mark.parametrize("description, expected", [
    ("Story text\n\nh3. Acceptance Criteria\n* First rule\n* Second rule\n\nh3. Notes\nignored",
     ["First rule", "Second rule"]),
    ("Story\n*Acceptance Criteria:*\n# Minimum is 1.00\n# At most 2 decimals", ["Minimum is 1.00", "At most 2 decimals"]),
    ("## Acceptance Criteria\n- Given A\n- When B\n- Then C\n- Given D\n- Then E",
     ["Given A When B Then C", "Given D Then E"]),
    ("Acceptance Criteria\n1. One\n2) Two", ["One", "Two"]),
])
def test_acceptance_criteria_styles(description, expected):
    assert jira_reader.split_description(description)[1] == expected


def test_adf_description_from_jira_cloud():
    adf = {"type": "doc", "content": [
        {"type": "paragraph", "content": [{"type": "text", "text": "As a user I want limits."}]},
        {"type": "heading", "attrs": {"level": 3}, "content": [{"type": "text", "text": "Acceptance Criteria"}]},
        {"type": "bulletList", "content": [
            {"type": "listItem", "content": [{"type": "paragraph", "content": [{"type": "text", "text": "Limit 500"}]}]},
            {"type": "listItem", "content": [{"type": "paragraph", "content": [{"type": "text", "text": "Reset daily"}]}]}]}]}
    assert jira_reader.split_description(jira_reader.field_text(adf))[1] == ["Limit 500", "Reset daily"]


def test_sample_epic_story_keys_criteria_field_and_risk():
    reqs, info = jira_reader.read_epic("PAY-100", jira_reader.FileJiraReader(ROOT / "samples/jira"))
    by = {r.req_id: r for r in reqs}
    assert list(by) == ["PAY-101", "PAY-102", "PAY-103", "PAY-104", "PAY-105"]
    assert info["ac_field"] == "customfield_10045"                       # auto-detected by display name
    assert by["PAY-104"].acceptance_criteria == ["Transfers above 5,000 require a one-time password",
                                                 "OTP expires after 300 seconds"]
    assert len(by["PAY-103"].acceptance_criteria) == 2                   # Given/When/Then grouped
    assert {k: r.risk for k, r in by.items()} == {"PAY-101": "high", "PAY-102": "high", "PAY-103": "medium",
                                                  "PAY-104": "high", "PAY-105": "low"}
    assert by["PAY-101"].module == "Domestic Payments v4.2"


def test_story_without_criteria_uses_the_llm():
    class FakeLLM:
        name = "fake"

        def generate(self, system, prompt, schema, context):
            return schema(requirements=[Requirement(req_id="x", module="m", title="t", description="d",
                                                    acceptance_criteria=["Extracted rule"], business_rules=[],
                                                    risk="low", source="s")])

    story = {"key": "PAY-9", "fields": {"summary": "Recent payees", "description": "Show the recent payees list."}}
    r = jira_reader.issue_to_requirement(story, {"key": "PAY-1", "fields": {"summary": "Epic"}}, None, FakeLLM())
    assert r.acceptance_criteria == ["Extracted rule"] and "extracted by LLM" in r.source


# ------------------------------------------------------------------ REST (fake HTTP session)
class FakeResponse:
    def __init__(self, status, body=None):
        self.status_code, self._body = status, body or {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"{self.status_code}")


class FakeSession:
    """Behaves like JIRA Cloud: /rest/api/2/search is retired (410), stories linked via 'Epic Link'."""

    def __init__(self):
        self.auth, self.headers, self.calls = None, {}, []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        if url.endswith("/rest/api/2/search"):
            return FakeResponse(410)
        if url.endswith("/rest/api/3/search/jql"):
            if params["jql"].startswith("parent"):
                return FakeResponse(200, {"issues": [], "isLast": True})
            if "nextPageToken" not in params:
                return FakeResponse(200, {"issues": [{"key": "PAY-201"}], "nextPageToken": "p2", "isLast": False})
            return FakeResponse(200, {"issues": [{"key": "PAY-202"}], "isLast": True})
        key = url.rsplit("/", 1)[1]
        desc = "Epic overview only" if key == "PAY-200" else "h3. Acceptance Criteria\n* Amount up to 500"
        fields = {"summary": f"Story {key}", "priority": {"name": "High"}, "description": desc}
        return FakeResponse(200, {"key": key, "fields": fields, "names": {"customfield_1": "Story Points"}})


def test_rest_reader_cloud_fallbacks_and_pagination():
    session = FakeSession()
    reader = jira_reader.RestJiraReader("https://x.atlassian.net", "me@x.com", "token", session=session)
    reqs, info = jira_reader.read_epic("PAY-200", reader)
    assert info["stories"] == ["PAY-201", "PAY-202"]                    # Epic Link fallback + 2 pages
    assert session.auth == ("me@x.com", "token")                        # Cloud basic auth
    assert [r.req_id for r in reqs] == ["PAY-201", "PAY-202"] and reqs[0].risk == "high"


def test_rest_reader_server_uses_bearer_token():
    session = FakeSession()
    jira_reader.RestJiraReader("https://jira.bank.local", None, "pat-123", session=session)
    assert session.headers["Authorization"] == "Bearer pat-123" and session.auth is None


# ------------------------------------------------------------------ per-story incremental runs (CLI)
@pytest.fixture
def epic_ws(tmp_path):
    shutil.copytree(ROOT / "samples/jira", tmp_path / "jira")
    return tmp_path


def _run(ws):
    return pipeline.run(None, [str(ROOT / "tests/fixtures/data")], str(ws / "output"), state_dir=str(ws / "state"),
                        offline=True, golden=None, epic="PAY-100", epic_source="file", epic_folder=str(ws / "jira"))


def test_editing_one_story_regenerates_only_that_story(epic_ws):
    first = _run(epic_ws)
    assert set(first.statuses.values()) == {"new"}
    assert all(c.tc_id.startswith(f"TC-{c.req_id}-") for c in first.test_cases)
    assert set(_run(epic_ws).statuses.values()) == {"unchanged"}

    path = epic_ws / "jira/PAY-100.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["stories"][3]["fields"]["customfield_10045"] = data["stories"][3]["fields"]["customfield_10045"].replace(
        "300 seconds", "120 seconds")                                    # PAY-104: OTP expiry changed
    path.write_text(json.dumps(data), encoding="utf-8")
    third = _run(epic_ws)
    assert {k for k, v in third.statuses.items() if v == "changed"} == {"PAY-104"}
    assert [k for k, v in third.statuses.items() if v == "unchanged"] == ["PAY-101", "PAY-102", "PAY-103", "PAY-105"]
    assert any(f.fixed_value == "120" for c in third.test_cases for f in c.data_fields)


def test_failed_regeneration_is_retried_next_run(tmp_path):
    st = State(path=tmp_path / "state.json")
    old = Requirement(req_id="PAY-1", module="m", title="t", description="d", acceptance_criteria=["limit 10"],
                      business_rules=[], risk="low", source="s")
    st.commit([old], [], {}, {})
    changed = old.model_copy(update={"acceptance_criteria": ["limit 20"]})
    st.commit([changed], [], {}, {}, failed={"PAY-1"})                 # generation failed this run
    assert st.diff_requirements([changed])[0]["PAY-1"] == "changed"    # so it is still pending


def test_epic_with_criteria_in_the_ac_field_becomes_a_requirement():
    class Src:
        def read(self, key):
            epic = {"key": "E-1", "fields": {"summary": "Epic", "description": "Overview only",
                                             "customfield_1": "* Epic-wide rule: audit every transfer"}}
            return epic, [], {"customfield_1": "Acceptance Criteria"}
    reqs, _ = jira_reader.read_epic("E-1", Src())
    assert [r.req_id for r in reqs] == ["E-1"]
