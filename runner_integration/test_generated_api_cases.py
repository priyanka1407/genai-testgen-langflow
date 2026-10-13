"""
Example: the automated runner consumes generated data with zero manual prep.

Each row in output/testdata/<TC-ID>.json becomes one parametrised pytest case.
  _expect   success | rejection | other  -> what the service must do
  _setup    prior state to seed first (e.g. transfers already made today)
  _<field>_record  the source record picked for a reference field (balance, status ...)
Swap the stubs for your real client (REST Assured / Playwright on the Java side
can read the same JSON files from src/test/resources/testdata).

    GENERATED_DIR=output pytest runner_integration
"""
import json
import os
from pathlib import Path

import pytest

GEN_DIR = Path(os.getenv("GENERATED_DIR", "output")) / "testdata"


def _cases():
    manifest = GEN_DIR / "manifest.json"
    if not manifest.exists():
        return []
    for entry in json.loads(manifest.read_text()):
        if entry["layer"] != "api":
            continue
        for i, row in enumerate(json.loads((GEN_DIR / entry["file"]).read_text())):
            yield pytest.param(row, id=f"{entry['tc_id']}-{i}")


def seed_setup(setup: dict) -> None:
    """Stub - insert prior transfers / failed logins through your test-data API."""
    for entity, rows in setup.items():
        assert all(isinstance(r, dict) for r in rows), entity


def call_service(row: dict) -> int:
    """Stub - replace with requests.post(f"{BASE_URL}/transfers", json=payload).status_code"""
    return 201 if row["_expect"] == "success" else 422


@pytest.mark.parametrize("row", list(_cases()))
def test_generated_case(row):
    assert not row.get("_warnings"), f"data generation warnings: {row['_warnings']}"
    if row["_expect"] == "other":
        pytest.skip("challenge/lockout outcome - needs a scenario-specific assertion")
    seed_setup(row.get("_setup", {}))
    status = call_service(row)
    if row["_expect"] == "success":
        assert status in (200, 201), f"{row['_tc_id']} expected success, got {status}"
    else:
        assert status in (400, 401, 403, 409, 422, 423), f"{row['_tc_id']} expected rejection, got {status}"
