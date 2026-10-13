"""
Independent verification ("LLM-as-judge") of generated test cases.

Why: the generator fills covers_criteria itself, so coverage computed from it is
self-reported. The judge re-reads each test case against the requirement and
returns which criteria it GENUINELY verifies and whether every expected value is
grounded in the requirement text. Coverage, the gap loop and the eval all use
the judge's verified_criteria, never the generator's claim.

Verdicts:
  accept -> kept
  revise -> kept, flagged 'needs-revision' in JIRA so the reviewer looks closely
  reject -> dropped (and counted in the eval)
"""
from __future__ import annotations

import json

from . import prompts
from .llm import LLM
from .models import CaseVerdict, Requirement, TestCase, VerificationSet


def verify_cases(judge: LLM, req: Requirement, cases: list[TestCase]) -> dict[str, CaseVerdict]:
    if not cases:
        return {}
    prompt = prompts.VERIFY_TEST_CASES.format(
        requirement_json=req.model_dump_json(indent=2),
        test_cases_json=json.dumps([c.model_dump() for c in cases], indent=2),
    )
    result = judge.generate(prompts.SYSTEM_REVIEWER, prompt, VerificationSet,
                            {"requirement": req, "test_cases": cases})
    n = len(req.acceptance_criteria)
    by_id: dict[str, CaseVerdict] = {}
    for v in result.verdicts:
        v.verified_criteria = sorted({i for i in v.verified_criteria if 0 <= i < n})
        by_id[v.tc_id] = v
    # A case the judge skipped is treated conservatively: nothing verified, needs review
    for c in cases:
        by_id.setdefault(c.tc_id, CaseVerdict(
            tc_id=c.tc_id, verified_criteria=[], grounded=False,
            unsupported_claims=["no verdict returned by judge"], verdict="revise", reason="not reviewed"))
    return {c.tc_id: by_id[c.tc_id] for c in cases}
