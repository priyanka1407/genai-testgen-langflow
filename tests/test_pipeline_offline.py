import json
import shutil
from pathlib import Path

import pytest

from testgen import coverage, evaluate, ingest, jira_publisher, pipeline, testdata
from testgen.models import CrossFieldConstraint, DataField, Requirement, SetupRecord, TestCase, TestStep
from testgen.offline import _Designer, verify
from testgen.state import State

ROOT = Path(__file__).resolve().parents[1]


def _field(**kw):
    base = dict(name="x", kind="decimal", valid=True, rule="", min_value=1, max_value=100,
                allowed_values=None, fixed_value=None, reference=None)
    return DataField(**(base | kw))


def _tc(fields, tc_id="TC-REQ-PAY-001-01", constraints=(), setup=(), outcome="success", title="t", covers=(0,)):
    return TestCase(tc_id=tc_id, req_id="REQ-PAY-001", title=title, test_type="positive",
                    technique="happy_path", priority="P1", layer="api", preconditions=[],
                    steps=[TestStep(step=1, action="a", expected="b")], expected_result="ok",
                    expected_outcome=outcome, covers_criteria=list(covers), data_fields=list(fields),
                    constraints=list(constraints), setup=list(setup), automation_candidate=True)


REFS = {"accounts": [
    {"account_id": "A1", "customer_id": "C1", "status": "ACTIVE", "balance": "500.00"},
    {"account_id": "A2", "customer_id": "C2", "status": "ACTIVE", "balance": "20000.00"},
    {"account_id": "A3", "customer_id": "C1", "status": "FROZEN", "balance": "900.00"},
], "customers": [
    {"customer_id": "C1", "kyc_status": "VERIFIED", "segment": "RETAIL"},
    {"customer_id": "C2", "kyc_status": "PENDING", "segment": "PREMIER"},
]}


# ------------------------------------------------------------------ 3. realistic data
def test_valid_and_invalid_ranges():
    ok = testdata.build_rows(_tc([_field()]), {}, 5)
    assert all(1 <= r["x"] <= 100 and r["_expect"] == "success" for r in ok)
    bad = testdata.build_rows(_tc([_field(valid=False)], outcome="rejection"), {}, 5)
    assert all(r["x"] < 1 or r["x"] > 100 for r in bad)


def test_join_enriches_records_and_filters_on_joined_columns():
    joined, relations = ingest.consolidate(REFS)
    assert {"from": "accounts", "to": "customers", "on": "customer_id", "matched": 3, "rows": 3} in relations
    f = _field(name="acc", kind="reference", reference="accounts.account_id[status=ACTIVE,customers.segment=RETAIL]")
    rows = testdata.build_rows(_tc([f]), joined, 5)
    assert {r["acc"] for r in rows} == {"A1"}
    assert rows[0]["_acc_record"]["customers.kyc_status"] == "VERIFIED"


def test_constraint_picks_a_record_that_can_fund_the_amount():
    acc = _field(name="account_id", kind="reference", reference="accounts.account_id[status=ACTIVE]")
    amt = _field(name="amount", fixed_value="10000.00")
    cons = [CrossFieldConstraint(left="amount", op="<=", right="account_id.balance", must_hold=True)]
    rows = testdata.build_rows(_tc([acc, amt], constraints=cons), REFS, 5)
    assert {r["account_id"] for r in rows} == {"A2"} and not any("_warnings" in r for r in rows)


def test_constraint_repairs_a_free_field_and_can_be_deliberately_violated():
    acc = _field(name="account_id", kind="reference", reference="accounts.account_id[status=ACTIVE]")
    amt = _field(name="amount", min_value=1, max_value=50000)
    hold = [CrossFieldConstraint(left="amount", op="<=", right="account_id.balance", must_hold=True)]
    for r in testdata.build_rows(_tc([acc, amt], constraints=hold), REFS, 5):
        assert float(r["amount"]) <= float(r["_account_id_record"]["balance"])
    broken = [CrossFieldConstraint(left="amount", op="<=", right="account_id.balance", must_hold=False)]
    for r in testdata.build_rows(_tc([acc, amt], constraints=broken, outcome="rejection"), REFS, 5):
        assert float(r["amount"]) > float(r["_account_id_record"]["balance"])  # insufficient funds case


def test_same_account_equality_between_reference_fields():
    src = _field(name="account_id", kind="reference", reference="accounts.account_id[status=ACTIVE]")
    dst = _field(name="destination_account_id", kind="reference", reference="accounts.account_id[status=ACTIVE]")
    cons = [CrossFieldConstraint(left="destination_account_id", op="==", right="account_id", must_hold=True)]
    for r in testdata.build_rows(_tc([src, dst], constraints=cons, outcome="rejection"), REFS, 5):
        assert r["account_id"] == r["destination_account_id"]


def test_unsatisfiable_constraint_is_reported_not_hidden():
    acc = _field(name="account_id", kind="reference", reference="accounts.account_id[status=FROZEN]")
    amt = _field(name="amount", fixed_value="5000.00")
    cons = [CrossFieldConstraint(left="amount", op="<=", right="account_id.balance", must_hold=True)]
    rows = testdata.build_rows(_tc([acc, amt], constraints=cons), REFS, 1)
    assert "not satisfiable" in rows[0]["_warnings"][0]


def test_setup_records_hit_the_exact_total():
    setup = [SetupRecord(entity="prior_transfer", description="earlier today", count=3,
                         fields=[_field(name="amount")], total_field="amount", total_value=9500.0)]
    row = testdata.build_rows(_tc([_field(name="amount", fixed_value="500.01")], setup=setup), {}, 1)[0]
    prior = row["_setup"]["prior_transfer"]
    assert len(prior) == 3 and round(sum(p["amount"] for p in prior), 2) == 9500.0


def test_data_is_deterministic():
    tc = _tc([_field(kind="iban"), _field()])
    assert testdata.build_rows(tc, {}, 3) == testdata.build_rows(tc, {}, 3)


# ------------------------------------------------------------------ 1. trustworthy results
def test_dedupe_keeps_boundary_siblings_but_drops_true_duplicates():
    a = _tc([_field(fixed_value="1.00")], "TC-1", title="amount at minimum")
    b = _tc([_field(fixed_value="1.01")], "TC-2", title="amount at minimum")   # different value -> keep
    c = _tc([_field(fixed_value="1.00")], "TC-3", title="amount at minimum")   # same data + outcome -> dup
    kept, dropped = coverage.dedupe([a, b, c])
    assert [k.tc_id for k in kept] == ["TC-1", "TC-2"] and dropped == ["TC-3"]


def _req(criteria, title="OTP for high-value transfers", risk="high"):
    return Requirement(req_id="REQ-AUT-002", module="Auth", title=title, description=title,
                       acceptance_criteria=criteria, business_rules=criteria, risk=risk, source="t")


def test_threshold_is_not_treated_as_a_rejection():
    req = _req(["Transfers above 5,000 require a one-time password"])
    cases = _Designer(req, {}, 1, None).design().test_cases
    by_value = {f.fixed_value: c.expected_outcome for c in cases for f in c.data_fields if f.name == "amount"}
    assert by_value == {"5000.00": "success", "5000.01": "other"}   # OTP challenge, not a rejection


def test_limit_and_minimum_boundaries_on_both_sides():
    req = _req(["Minimum transfer amount is 1.00", "Account is locked after 5 consecutive failed login attempts"])
    cases = _Designer(req, {}, 1, None).design().test_cases
    vals = {(f.name, f.fixed_value): c.expected_outcome for c in cases for f in c.data_fields if f.fixed_value}
    assert vals[("amount", "0.99")] == "rejection" and vals[("amount", "1.00")] == "success"
    assert vals[("failed_attempts", "4")] == "success" and vals[("failed_attempts", "5")] == "rejection"


def test_judge_does_not_confirm_an_overclaimed_criterion():
    req = _req(["OTP expires after 300 seconds", "Transfers above 5,000 require a one-time password"])
    honest = _tc([_field(name="elapsed_seconds", fixed_value="300")], "TC-1",
                 title="OTP expires after 300 seconds", covers=[0])
    liar = _tc([_field(name="elapsed_seconds", fixed_value="300")], "TC-2",
               title="OTP expires after 300 seconds", covers=[0, 1])   # claims the 5,000 rule too
    honest.req_id = liar.req_id = req.req_id
    verdicts = {v.tc_id: v for v in verify(req, [honest, liar]).verdicts}
    assert verdicts["TC-2"].verified_criteria == [0] and verdicts["TC-2"].verdict == "revise"
    row = coverage.analyse_requirement(req, [honest, liar], verdicts)
    assert row.uncovered_criteria == [1] and row.claimed_but_unverified == [1]


def test_ungrounded_expected_value_is_rejected():
    req = _req(["OTP expires after 300 seconds"])
    tc = _tc([], "TC-1", title="OTP expires after 300 seconds", covers=[0])
    tc.req_id, tc.expected_result = req.req_id, "OTP expires after 600 seconds"   # invented number
    v = verify(req, [tc]).verdicts[0]
    assert not v.grounded and v.verdict == "reject"


def test_eval_flags_regressions_against_baseline():
    base = {"verified_coverage_pct": 95.0, "overclaim_rate": 0.02, "golden_recall": 0.9}
    assert evaluate.compare({"verified_coverage_pct": 94.0, "overclaim_rate": 0.03, "golden_recall": 0.9}, base) == []
    regs = evaluate.compare({"verified_coverage_pct": 80.0, "overclaim_rate": 0.2, "golden_recall": 0.5}, base)
    assert len(regs) == 3


# ------------------------------------------------------------------ 2. safe on every release
@pytest.fixture
def workspace(tmp_path):
    # fixed copy of the sample inputs, so edits to samples/ never change test expectations
    shutil.copytree(ROOT / "tests/fixtures", tmp_path / "samples")
    return tmp_path


def _run(ws, **kw):
    return pipeline.run([str(ws / "samples/docs")], [str(ws / "samples/data")], str(ws / "output"),
                        state_dir=str(ws / "state"), offline=True, golden=str(ROOT / "evals/golden.json"), **kw)


def _ops(res):
    out = {}
    for x in res.jira:
        out[x["op"]] = out.get(x["op"], 0) + 1
    return out


def test_first_run_is_complete_and_correct(workspace):
    res = _run(workspace)
    assert res.summary["verified_coverage_pct"] == 100.0 and not res.summary["requirements_with_gaps"]
    assert res.metrics["golden_recall"] == 1.0
    assert _ops(res) == {"created": len(res.test_cases)}
    for entry in json.loads((workspace / "output/testdata/manifest.json").read_text()):
        for row in json.loads((workspace / "output/testdata" / entry["file"]).read_text()):
            assert "_warnings" not in row, (entry["tc_id"], row)


def test_rerun_without_changes_calls_no_llm_and_touches_nothing(workspace):
    first = _run(workspace)
    second = _run(workspace)
    assert set(second.statuses.values()) == {"unchanged"}
    assert json.loads((workspace / "output/run_stats.json").read_text())["requirements"] == {}  # no generation
    assert _ops(second) == {"unchanged": len(first.test_cases)}
    assert [c.tc_id for c in second.test_cases] == [c.tc_id for c in first.test_cases]


def test_changed_requirement_updates_in_place_and_keeps_other_ids(workspace):
    first = _run(workspace)
    notes = workspace / "samples/docs/release_notes_v4.2.md"
    notes.write_text(notes.read_text().replace("up to 10,000 USD", "up to 15,000 USD"))
    second = _run(workspace)
    assert second.statuses["REQ-PAY-001"] == "changed"
    assert [s for r, s in second.statuses.items() if r != "REQ-PAY-001"] == ["unchanged"] * 4
    untouched = lambda res: sorted(c.tc_id for c in res.test_cases if c.req_id != "REQ-PAY-001")  # noqa: E731
    assert untouched(first) == untouched(second)
    ops = _ops(second)
    assert ops.get("updated", 0) > 0 and ops.get("created", 0) <= 2    # no duplicate issues
    assert any(f.fixed_value == "15000.01" for c in second.test_cases for f in c.data_fields)


def test_reworded_title_keeps_its_requirement_id(tmp_path):
    st = State(path=tmp_path / "s.json")
    r1 = _req(["OTP expires after 300 seconds"], title="OTP for high-value transfers")
    st.assign_ids([r1])
    r2 = _req(["OTP expires after 300 seconds"], title="OTP for high value transfers")
    status, removed = st.assign_ids([r2])
    assert r2.req_id == r1.req_id and status[r2.req_id] == "unchanged" and not removed


def test_removed_requirement_obsoletes_its_issues_and_ids_are_not_reused(workspace):
    first = _run(workspace)
    n_auth = sum(1 for c in first.test_cases if c.req_id.startswith("REQ-AUT"))
    (workspace / "samples/docs/auth_spec.md").unlink()
    res = _run(workspace)
    assert _ops(res).get("obsoleted") == n_auth and n_auth > 0
    assert not any(c.req_id.startswith("REQ-AUT") for c in res.test_cases)
    fake = json.loads((workspace / "state/fake_jira.json").read_text())
    assert any("obsolete" in i["fields"]["labels"] for i in fake["issues"].values())
    st = State.load(workspace / "state")
    new = Requirement(req_id="x", module="Auth", title="Brand new auth rule", description="",
                      acceptance_criteria=["a"], business_rules=[], risk="high", source="t")
    st.assign_ids(st.previous_requirements() + [new])
    assert new.req_id == "REQ-AUT-003"   # 001/002 are retired, never reused


def test_approval_is_void_when_content_changes(workspace):
    _run(workspace)
    st = State.load(workspace / "state")
    client = jira_publisher.FileJiraClient(workspace / "state/fake_jira.json")
    for e in st.jira["dry_run"].values():
        client.set_status(e["key"], jira_publisher.APPROVED)
    approved = _run(workspace, approved_only=True)
    assert all(r["status"] == "approved" for r in approved.reviews.values())

    notes = workspace / "samples/docs/release_notes_v4.2.md"
    notes.write_text(notes.read_text().replace("up to 10,000 USD", "up to 15,000 USD"))
    res = _run(workspace, approved_only=True)
    manifest = {e["tc_id"] for e in json.loads((workspace / "output/testdata/manifest.json").read_text())}
    updated = {x["tc_id"] for x in res.jira if x["op"] in ("updated", "created")}
    assert updated and not (updated & manifest)                       # changed cases need re-approval
    assert any(t.startswith("TC-REQ-AUT") for t in manifest)          # untouched approvals still valid


def test_reviewer_edit_is_pulled_back_and_not_overwritten(workspace):
    _run(workspace)
    st = State.load(workspace / "state")
    tid = "TC-REQ-AUT-002-01"
    key = st.jira["dry_run"][tid]["key"]
    client = jira_publisher.FileJiraClient(workspace / "state/fake_jira.json")
    desc = client.data["issues"][key]["fields"]["description"]
    start = desc.index("h3. Expected result\n") + len("h3. Expected result\n")
    end = desc.index("\n", start)
    client.edit_description(key, desc[:start] + "Transfer of exactly 5,000 completes without an OTP prompt" + desc[end:])

    res = _run(workspace)
    edited = next(c for c in res.test_cases if c.tc_id == tid)
    assert edited.expected_result == "Transfer of exactly 5,000 completes without an OTP prompt"
    assert next(x["op"] for x in res.jira if x["tc_id"] == tid) == "kept-reviewer-edit"
    again = _run(workspace)                                            # stays edited on later runs
    assert next(c for c in again.test_cases if c.tc_id == tid).expected_result.startswith("Transfer of exactly")


def test_rejected_cases_never_reach_the_runner(workspace):
    _run(workspace)
    st = State.load(workspace / "state")
    client = jira_publisher.FileJiraClient(workspace / "state/fake_jira.json")
    client.set_status(st.jira["dry_run"]["TC-REQ-PAY-002-01"]["key"], jira_publisher.REJECTED)
    res = _run(workspace)
    manifest = {e["tc_id"] for e in json.loads((workspace / "output/testdata/manifest.json").read_text())}
    assert "TC-REQ-PAY-002-01" not in manifest
    assert res.metrics["reviewer_rejection_rate"] == 1.0


def test_profile_never_exposes_identifiers_or_per_row_values():
    joined, _ = ingest.consolidate(REFS)
    cols = ingest.profile_sources(joined)["accounts"]["columns"]
    assert cols["account_id"] == {"type": "identifier", "nulls": 0, "distinct": 3}
    assert cols["customer_id"]["type"] == "identifier" and "values" not in cols["customer_id"]
    assert cols["status"]["values"] == ["ACTIVE", "FROZEN"]          # repeating category: listed
    text = json.dumps(cols)
    assert "A1" not in text and "C1" not in text
