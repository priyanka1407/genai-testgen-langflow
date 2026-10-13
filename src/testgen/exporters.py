"""
Outputs: everything a reviewer, auditor and the automated test runner needs.

  test_cases.json          full structured test cases
  test_cases.csv           flat view for Excel / test-management import
  traceability.csv         requirement -> test cases -> criteria coverage
  coverage_summary.json    headline metrics (+ effort estimate)
  verification.json        judge verdict per test case
  run_stats.json           per-requirement status (new/changed/unchanged), generated/dupes/rejected
  reviews.json             JIRA review status per test case
  eval_metrics.json        quality metrics (see evaluate.py)
  testdata/<TC-ID>.json    data-driven rows, consumed directly by the runner
  testdata/manifest.json   tc_id -> layer/priority/file, used to select what to run
  audit_log.jsonl          who/what/when/which model+prompt version, input hashes
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .models import CoverageRow, Requirement, TestCase

# Baseline used for the "manual effort saved" estimate - tune to your team's numbers
MANUAL_MIN_PER_TEST_CASE = 20
MANUAL_MIN_PER_DATA_ROW = 5


def write_test_cases(cases: list[TestCase], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "test_cases.json").write_text(
        json.dumps([c.model_dump() for c in cases], indent=2), encoding="utf-8")
    with (out / "test_cases.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["tc_id", "req_id", "title", "type", "technique", "priority", "layer",
                    "preconditions", "steps", "expected_result", "expected_outcome",
                    "claimed_criteria", "automation_candidate"])
        for c in cases:
            w.writerow([c.tc_id, c.req_id, c.title, c.test_type, c.technique, c.priority, c.layer,
                        " | ".join(c.preconditions),
                        " | ".join(f"{s.step}. {s.action} => {s.expected}" for s in c.steps),
                        c.expected_result, c.expected_outcome, c.covers_criteria, c.automation_candidate])


def write_requirements(reqs: list[Requirement], out: Path) -> None:
    (out / "requirements.json").write_text(
        json.dumps([r.model_dump() for r in reqs], indent=2), encoding="utf-8")


def write_coverage(rows: list[CoverageRow], summary: dict, n_cases: int, n_rows: int, out: Path) -> dict:
    with (out / "traceability.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["req_id", "risk", "verified_coverage_pct", "uncovered_criteria",
                    "claimed_but_unverified", "test_types", "missing_types", "test_cases"])
        for r in rows:
            w.writerow([r.req_id, r.risk, r.percent, r.uncovered_criteria, r.claimed_but_unverified,
                        ";".join(r.test_types), ";".join(r.missing_types), ";".join(r.test_case_ids)])
    summary = summary | {
        "test_cases": n_cases,
        "test_data_rows": n_rows,
        "estimated_manual_hours_saved": round(
            (n_cases * MANUAL_MIN_PER_TEST_CASE + n_rows * MANUAL_MIN_PER_DATA_ROW) / 60, 1),
    }
    (out / "coverage_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_testdata(dataset: dict[str, list[dict]], cases: list[TestCase], out: Path) -> int:
    td = out / "testdata"
    td.mkdir(parents=True, exist_ok=True)
    manifest = []
    by_id = {c.tc_id: c for c in cases}
    total = 0
    for tc_id, rows in dataset.items():
        if not rows:
            continue
        (td / f"{tc_id}.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
        tc = by_id[tc_id]
        manifest.append({"tc_id": tc_id, "req_id": tc.req_id, "layer": tc.layer, "priority": tc.priority,
                         "type": tc.test_type, "file": f"{tc_id}.json", "rows": len(rows)})
        total += len(rows)
    (td / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return total


def write_json(out: Path, name: str, data) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / name).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _sha(path_or_text: str) -> str:
    return hashlib.sha256(path_or_text.encode("utf-8")).hexdigest()[:16]


def audit(out: Path, event: str, **fields) -> None:
    record = {"ts": datetime.now(timezone.utc).isoformat(), "event": event} | fields
    with (out / "audit_log.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def doc_hashes(docs: list[tuple[str, str]]) -> dict[str, str]:
    return {name: _sha(text) for name, text in docs}
