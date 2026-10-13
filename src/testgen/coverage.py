"""
Coverage + quality gate.

- dedupe(): drop near-duplicate test cases (same requirement, ~same title/steps).
- analyse(): requirement -> test case traceability, uncovered acceptance criteria
  and missing test types. When judge verdicts are supplied, a criterion only
  counts as covered if the judge VERIFIED it (claims alone are reported separately
  as claimed_but_unverified).
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

from .models import CaseVerdict, CoverageRow, Requirement, TestCase

NUMERIC = re.compile(r"\d")


def required_types(req: Requirement) -> set[str]:
    types = {"positive", "negative"}
    if any(NUMERIC.search(r) for r in req.business_rules):
        types.add("boundary")
    if req.risk == "high":
        types.add("security_or_edge")
    return types


def signature(tc: TestCase) -> str:
    steps = " ".join(s.action for s in tc.steps)
    data = " ".join(f"{f.name}={f.fixed_value or f.reference or f.rule}" for f in tc.data_fields)
    setup = " ".join(f"{s.entity}:{s.total_value}" for s in tc.setup)
    return re.sub(r"\W+", " ", f"{tc.title} {steps} {data} {setup}".lower()).strip()


def similarity(a: TestCase, b: TestCase) -> float:
    return SequenceMatcher(None, signature(a), signature(b)).ratio()


def data_key(tc: TestCase) -> tuple:
    """What the test actually exercises: exact values, record filters, prior state, expected outcome."""
    return (
        tc.expected_outcome,
        tuple(sorted((f.name, f.fixed_value, f.reference, f.valid) for f in tc.data_fields)),
        tuple(sorted((c.left, c.op, c.right, c.must_hold) for c in tc.constraints)),
        tuple(sorted((s.entity, s.count, s.total_value) for s in tc.setup)),
    )


def dedupe(cases: list[TestCase], threshold: float = 0.92) -> tuple[list[TestCase], list[str]]:
    """A duplicate exercises the SAME data with the SAME expected outcome and reads almost the same.
    Boundary siblings (1.00 vs 1.01) differ in data, so they are never merged."""
    kept: list[TestCase] = []
    dropped: list[str] = []
    for tc in cases:
        key = data_key(tc)
        dup = any(k.req_id == tc.req_id and k.test_type == tc.test_type and data_key(k) == key
                  and similarity(k, tc) >= threshold for k in kept)
        (dropped.append(tc.tc_id) if dup else kept.append(tc))
    return kept, dropped


def _positive_like(c: TestCase) -> bool:
    return c.test_type == "positive" or (c.expected_outcome == "success" and c.test_type in ("boundary", "edge"))


def analyse_requirement(req: Requirement, cases: list[TestCase],
                        verdicts: dict[str, CaseVerdict] | None = None) -> CoverageRow:
    mine = [c for c in cases if c.req_id == req.req_id]
    total = len(req.acceptance_criteria)
    claimed = sorted({i for c in mine for i in c.covers_criteria if 0 <= i < total})
    if verdicts is None:
        covered = claimed
    else:
        covered = sorted({i for c in mine if c.tc_id in verdicts for i in verdicts[c.tc_id].verified_criteria})
    types = {c.test_type for c in mine}
    missing = []
    for t in sorted(required_types(req)):
        if t == "security_or_edge":
            if not types & {"security", "edge"} and not any(c.technique == "error_guessing" for c in mine):
                missing.append("security")
        elif t == "positive":
            if not any(_positive_like(c) for c in mine):
                missing.append("positive")
        elif t not in types:
            missing.append(t)
    return CoverageRow(
        req_id=req.req_id, risk=req.risk, total_criteria=total, covered_criteria=covered,
        uncovered_criteria=[i for i in range(total) if i not in covered],
        claimed_but_unverified=[i for i in claimed if i not in covered],
        test_types=sorted(types), test_case_ids=[c.tc_id for c in mine], missing_types=missing,
    )


def analyse(reqs: list[Requirement], cases: list[TestCase],
            verdicts: dict[str, CaseVerdict] | None = None) -> list[CoverageRow]:
    return [analyse_requirement(r, cases, verdicts) for r in reqs]


def summary(rows: list[CoverageRow]) -> dict:
    total = sum(r.total_criteria for r in rows)
    covered = sum(len(r.covered_criteria) for r in rows)
    return {
        "requirements": len(rows),
        "acceptance_criteria": total,
        "criteria_verified": covered,
        "verified_coverage_pct": round(100 * covered / total, 1) if total else 100.0,
        "criteria_claimed_but_unverified": sum(len(r.claimed_but_unverified) for r in rows),
        "requirements_fully_covered": sum(1 for r in rows if not r.uncovered_criteria and not r.missing_types),
        "requirements_with_gaps": [r.req_id for r in rows if r.uncovered_criteria or r.missing_types],
    }
