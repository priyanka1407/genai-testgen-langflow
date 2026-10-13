"""
Evaluation: is the generated suite actually good, and did a change make it worse?

Metrics (written to output/eval_metrics.json on every pipeline run)
  verified_coverage_pct     criteria the judge confirmed are genuinely tested
  claimed_coverage_pct      criteria the generator CLAIMED - the gap to verified is the over-claim
  overclaim_rate            share of claimed (case, criterion) links the judge did not confirm
  grounded_rate             share of cases whose expected values are backed by the requirement
  judge_reject_rate         share of freshly generated cases the judge rejected (this run)
  duplicate_rate            share of freshly generated cases removed as near-duplicates (this run)
  reviewer_rejection_rate   share of reviewed cases rejected by humans in JIRA
  golden_recall             share of must-have scenarios (evals/golden.json) that some case covers

CLI - score a run against the golden set and the stored baseline (exit 1 on regression, for CI):
    python -m testgen.evaluate --out output --golden evals/golden.json --baseline evals/baseline.json
    python -m testgen.evaluate ... --update-baseline
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from pathlib import Path

from .env import load_env_file
from .models import CaseVerdict, Requirement, TestCase

HIGHER_IS_BETTER = ["verified_coverage_pct", "grounded_rate", "golden_recall"]
LOWER_IS_BETTER = ["overclaim_rate", "judge_reject_rate", "duplicate_rate", "reviewer_rejection_rate"]
TOLERANCE = {"verified_coverage_pct": 2.0, "default": 0.03}


def _canon(v) -> str:
    try:
        s = format(Decimal(str(v)), "f")
    except InvalidOperation:
        return str(v)
    return s.rstrip("0").rstrip(".") if "." in s else s


def case_text(c: TestCase) -> str:
    """Searchable text of a case; fixed values rendered canonically as name=value."""
    parts = [c.title, c.expected_result, *(f"{s.action} {s.expected}" for s in c.steps)]
    parts += [f"{f.name}={_canon(f.fixed_value)}" for f in c.data_fields if f.fixed_value is not None]
    parts += [f"{f.name} {f.reference}" for f in c.data_fields if f.reference]
    parts += [f"{s.entity} {s.description} {s.total_field}={_canon(s.total_value)}" for s in c.setup]
    text = " ".join(parts).lower()
    return re.sub(r"(?<=\d),(?=\d{3})", "", text)


def _hit(keywords: list[str], text: str) -> bool:
    return all(re.search(rf"(?<![\w.]){re.escape(k.lower())}(?![\w.])", text) for k in keywords)


def golden_recall(reqs: list[Requirement], cases: list[TestCase], golden: dict) -> dict:
    results = []
    for g in golden.get("requirements", []):
        req = max(reqs, key=lambda r: SequenceMatcher(None, r.title.lower(), g["match_title"].lower()).ratio(),
                  default=None)
        ok_req = req and SequenceMatcher(None, req.title.lower(), g["match_title"].lower()).ratio() >= 0.6
        mine = [c for c in cases if ok_req and c.req_id == req.req_id]
        for m in g["must_have"]:
            hit = next((c.tc_id for c in mine if _hit(m["keywords"], case_text(c))
                        and m.get("outcome") in (None, c.expected_outcome)), None)
            results.append({"id": m["id"], "requirement": g["match_title"], "covered_by": hit})
    found = sum(1 for r in results if r["covered_by"])
    return {"recall": round(found / len(results), 3) if results else None,
            "missed": [r["id"] for r in results if not r["covered_by"]], "details": results}


def compute(reqs: list[Requirement], cases: list[TestCase], verdicts: dict[str, CaseVerdict],
            run_stats: dict, reviews: dict, golden: dict | None = None) -> dict:
    total = sum(len(r.acceptance_criteria) for r in reqs) or 1
    claimed = {(c.req_id, i) for c in cases for i in c.covers_criteria}
    verified = {(c.req_id, i) for c in cases if c.tc_id in verdicts for i in verdicts[c.tc_id].verified_criteria}
    links = [(c.tc_id, i) for c in cases for i in c.covers_criteria]
    unconfirmed = [(t, i) for t, i in links if t in verdicts and i not in verdicts[t].verified_criteria]
    judged = [verdicts[c.tc_id] for c in cases if c.tc_id in verdicts]
    gen = sum(s["generated"] for s in run_stats.get("requirements", {}).values())
    dup = sum(s["duplicates"] for s in run_stats.get("requirements", {}).values())
    rej = sum(len(s["rejected_by_judge"]) for s in run_stats.get("requirements", {}).values())
    decided = [r for r in reviews.values() if r["status"] in ("approved", "rejected")]
    m = {
        "requirements": len(reqs),
        "test_cases": len(cases),
        "verified_coverage_pct": round(100 * len(verified) / total, 1),
        "claimed_coverage_pct": round(100 * len(claimed) / total, 1),
        "overclaim_rate": round(len(unconfirmed) / len(links), 3) if links else 0.0,
        "grounded_rate": round(sum(v.grounded for v in judged) / len(judged), 3) if judged else None,
        "judge_reject_rate": round(rej / gen, 3) if gen else None,
        "duplicate_rate": round(dup / gen, 3) if gen else None,
        "reviewer_rejection_rate": round(sum(r["status"] == "rejected" for r in decided) / len(decided), 3)
        if decided else None,
        "failed_requirements": [k for k, s in run_stats.get("requirements", {}).items() if s.get("error")],
    }
    if golden:
        g = golden_recall(reqs, cases, golden)
        m["golden_recall"], m["golden_missed"] = g["recall"], g["missed"]
    return m


def compare(metrics: dict, baseline: dict) -> list[str]:
    regressions = []
    for k in HIGHER_IS_BETTER + LOWER_IS_BETTER:
        new, old = metrics.get(k), baseline.get(k)
        if new is None or old is None:
            continue
        tol = TOLERANCE.get(k, TOLERANCE["default"])
        worse = (old - new) if k in HIGHER_IS_BETTER else (new - old)
        if worse > tol:
            regressions.append(f"{k}: {old} -> {new}")
    return regressions


def _load_run(out: Path):
    reqs = [Requirement(**d) for d in json.loads((out / "requirements.json").read_text(encoding="utf-8"))]
    cases = [TestCase(**d) for d in json.loads((out / "test_cases.json").read_text(encoding="utf-8"))]
    verdicts = {k: CaseVerdict(**v) for k, v in
                json.loads((out / "verification.json").read_text(encoding="utf-8")).items()}
    stats = json.loads((out / "run_stats.json").read_text(encoding="utf-8"))
    reviews = json.loads((out / "reviews.json").read_text(encoding="utf-8"))
    return reqs, cases, verdicts, stats, reviews


def main() -> None:
    ap = argparse.ArgumentParser(description="Score a generated suite")
    ap.add_argument("--out", default="output")
    ap.add_argument("--golden", default="evals/golden.json")
    ap.add_argument("--baseline", default="evals/baseline.json")
    ap.add_argument("--update-baseline", action="store_true")
    a = ap.parse_args()
    load_env_file()
    out = Path(a.out)
    golden = json.loads(Path(a.golden).read_text(encoding="utf-8")) if Path(a.golden).exists() else None
    metrics = compute(*_load_run(out), golden=golden)
    (out / "eval_report.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "golden_missed"}, indent=2))
    if metrics.get("golden_missed"):
        print("Missed must-have scenarios:", ", ".join(metrics["golden_missed"]))
    base = Path(a.baseline)
    if a.update_baseline:
        base.parent.mkdir(parents=True, exist_ok=True)
        base.write_text(json.dumps({k: metrics[k] for k in HIGHER_IS_BETTER + LOWER_IS_BETTER if k in metrics},
                                   indent=2), encoding="utf-8")
        print(f"Baseline updated: {base}")
    elif base.exists():
        regressions = compare(metrics, json.loads(base.read_text(encoding="utf-8")))
        if regressions:
            print("REGRESSIONS vs baseline:\n  " + "\n  ".join(regressions))
            sys.exit(1)
        print("No regressions vs baseline.")


if __name__ == "__main__":
    main()
