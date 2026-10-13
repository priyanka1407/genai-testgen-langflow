"""
Offline heuristic backend - no API key, fully deterministic.

Used for demos, CI and unit tests. It mimics the three LLM tasks:
  * requirement extraction  - markdown convention "## Module" / "### Feature" / "- criterion"
  * test case design        - limit-aware boundary analysis, status partitions,
                              same-account rules, daily cumulative limits
  * verification (judge)    - lexical/numeric support check of each claimed criterion

The limit semantics matter: "up to N" is a maximum (N+step is rejected), but
"above N require OTP" is a THRESHOLD (N+step triggers a challenge, it is not a rejection).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import BaseModel

from .models import (
    CaseVerdict, CrossFieldConstraint, DataField, Requirement, RequirementSet, SetupRecord,
    TestCase, TestCaseSet, TestStep, VerificationSet,
)

NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
WORD = re.compile(r"[a-z]{4,}")
STOP = {"must", "with", "that", "from", "have", "then", "when", "this", "into", "will", "should",
        "each", "across", "they", "their", "there", "being", "verify", "value", "case"}
REJECT_WORDS = re.compile(r"reject|lock|expire|denied|declin|blocked|cannot|must be reactivated")
DAILY = re.compile(r"per (calendar )?day|daily")
SAME_ACCOUNT = re.compile(r"cannot be the same|must not be the same|must (be )?differ")


def _f(name, kind="decimal", valid=True, rule="", fixed=None, reference=None, mn=None, mx=None):
    return DataField(name=name, kind=kind, valid=valid, rule=rule, min_value=mn, max_value=mx,
                     allowed_values=None, fixed_value=fixed, reference=reference)


def _floats(text: str) -> set[float]:
    out = set()
    for m in NUM.finditer(text):
        try:
            out.add(round(float(m.group().replace(",", "")), 2))
        except ValueError:
            pass
    return out


# ---------------------------------------------------------------- limits
@dataclass
class Limit:
    kind: str   # max | min | precision | above (exclusive threshold) | at (inclusive threshold)
    value: float
    step: float
    field: str


def _field_for(low: str, after: str, raw: str) -> tuple[str, float]:
    if "attempt" in after:
        return "failed_attempts", 1
    if "second" in after:
        return "elapsed_seconds", 1
    if "minute" in after:
        return "elapsed_minutes", 1
    if "decimal place" in after or any(w in low for w in ("amount", "transfer", "payment", "usd", "balance")):
        return "amount", 0.01
    return "value", 0.01 if "." in raw else 1


def parse_limits(text: str) -> list[Limit]:
    low = text.lower()
    out = []
    for m in NUM.finditer(text):
        s, e = m.start(), m.end()
        prev = text[s - 1] if s else " "
        nxt = text[e] if e < len(text) else " "
        if prev in "-:" or prev.isalpha() or nxt == ":":   # error codes, times
            continue
        raw = m.group()
        before, after = low[max(0, s - 45):s], low[e:e + 45]
        field, step = _field_for(low, after, raw)
        if "decimal place" in after:
            kind = "precision"
        elif any(w in before for w in ("up to", "maximum", "at most", "not exceed", "no more than", "limit of")):
            kind = "max"
        elif any(w in before for w in ("minimum", "at least", "no less than")):
            kind = "min"
        elif any(w in before for w in ("above", "more than", "over ", "greater than", "exceeding")):
            kind = "above"
        elif "after" in before:
            kind = "at"
        else:
            continue
        out.append(Limit(kind, float(raw.replace(",", "")), step, field))
    return out


def _fmt(v: float, step: float) -> str:
    return str(int(round(v))) if step == 1 else f"{v:.2f}"


# ---------------------------------------------------------------- designer
class _Designer:
    def __init__(self, req: Requirement, profile: dict, start: int, gap: dict | None):
        self.req, self.profile, self.idx, self.gap = req, profile, start, gap
        self.cases: list[TestCase] = []
        accounts = profile.get("accounts", {}).get("columns", {})
        self.has_accounts = "account_id" in accounts
        self.has_balance = "balance" in accounts
        self._active_done = False

    def _context_filters(self) -> str:
        """Values the requirement names that exist in the (joined) account data, e.g.
        'retail customer ... USD' -> customers.segment=RETAIL,currency=USD."""
        text = " ".join([self.req.title, *self.req.acceptance_criteria])
        found = []
        for col, ci in self.profile.get("accounts", {}).get("columns", {}).items():
            if "status" in col:
                continue
            for v in ci.get("values", []):
                if len(v) > 2 and re.search(rf"\b{re.escape(v)}\b", text, re.IGNORECASE):
                    found.append(f"{col}={v}")
        return ",".join(found)

    def account(self, filt="status=ACTIVE") -> DataField:
        extra = self._context_filters()
        filt = f"{filt},{extra}" if extra else filt
        if self.has_accounts:
            return _f("account_id", "reference", rule=f"existing account [{filt}]",
                      reference=f"accounts.account_id[{filt}]")
        return _f("account_id", "account_number", rule="existing account")

    def add(self, title, ttype, tech, outcome, criteria, fields, expected, constraints=(), setup=(), prio=None):
        self.cases.append(TestCase(
            tc_id=f"TC-{self.req.req_id}-{self.idx:02d}", req_id=self.req.req_id, title=title,
            test_type=ttype, technique=tech, priority=prio or ("P1" if self.req.risk == "high" else "P2"),
            layer="api", preconditions=["User is authenticated"],
            steps=[TestStep(step=1, action=f"Prepare request for '{self.req.title}' using the test data row",
                            expected="Request is well-formed"),
                   TestStep(step=2, action="Submit the request", expected=expected)],
            expected_result=expected, expected_outcome=outcome, covers_criteria=list(criteria),
            data_fields=list(fields), constraints=list(constraints), setup=list(setup),
            automation_candidate=True,
        ))
        self.idx += 1

    @staticmethod
    def _expected(outcome: str, crit: str) -> str:
        return {"success": f"Accepted - {crit}",
                "rejection": f"Rejected as specified - {crit}",
                "other": f"Additional step required as specified - {crit}"}[outcome]

    def _value_fields(self, lim: Limit, value: str, ok: bool, label: str, crit: str):
        if lim.field == "amount":
            fields = [self.account(), _f("amount", "decimal", ok, f"{label}: {crit}", fixed=value)]
            cons = [CrossFieldConstraint(left="amount", op="<=", right="account_id.balance", must_hold=True)] \
                if self.has_balance else []
            return fields, cons
        return [self.account(), _f(lim.field, "integer", ok, f"{label}: {crit}", fixed=value)], []

    def limits(self, i: int, crit: str):
        made = False
        trig = "rejection" if REJECT_WORDS.search(crit.lower()) else "other"
        for lim in parse_limits(crit):
            L, st = lim.value, lim.step
            if lim.kind == "max":
                points = [(L - st, "success", "just below limit"), (L, "success", "at limit"),
                          (L + st, "rejection", "just above limit")]
            elif lim.kind == "min":
                points = [(L - st, "rejection", "just below minimum"), (L, "success", "at minimum"),
                          (L + st, "success", "just above minimum")]
            elif lim.kind == "above":
                points = [(L, "success", "at threshold - not triggered"), (L + st, trig, "just above threshold - triggered")]
            elif lim.kind == "at":
                points = [(L - st, "success", "one below threshold - not triggered"), (L, trig, "at threshold - triggered")]
            else:  # precision: N decimals ok, N+1 rejected
                n = int(L)
                for val, outcome, label in ((f"10.{'2' * n}", "success", f"{n} decimals"),
                                            (f"10.{'2' * (n + 1)}", "rejection", f"{n + 1} decimals")):
                    fields = [self.account(), _f("amount", "decimal", outcome == "success", f"{label}: {crit}", fixed=val)]
                    self.add(f"{self.req.title} - amount with {label} ({val})", "boundary", "boundary_value",
                             outcome, [i], fields, self._expected(outcome, crit))
                made = True
                continue
            for v, outcome, label in points:
                val = _fmt(v, st)
                fields, cons = self._value_fields(lim, val, outcome == "success", label, crit)
                self.add(f"{self.req.title} - {lim.field} {label} ({val})", "boundary", "boundary_value",
                         outcome, [i], fields, self._expected(outcome, crit), cons)
            if lim.kind == "max" and lim.field == "amount" and DAILY.search(crit.lower()):
                self.daily(i, crit, L)
            made = True
        return made

    def daily(self, i: int, crit: str, L: float):
        prior = L - 500
        setup = [SetupRecord(entity="prior_transfer", description="Transfers already made today from the same account",
                             count=2, fields=[_f("amount", "decimal", rule="earlier transfer today", mn=1, mx=L)],
                             total_field="amount", total_value=prior)]
        cons = [CrossFieldConstraint(left="amount", op="<=", right="account_id.balance", must_hold=True)] \
            if self.has_balance else []
        for amt, outcome, label in ((500.0, "success", "reaches"), (500.01, "rejection", "exceeds")):
            fields = [self.account(), _f("amount", "decimal", outcome == "success",
                                         f"cumulative with {prior:.2f} already sent today {label} limit",
                                         fixed=f"{amt:.2f}")]
            self.add(f"{self.req.title} - cumulative daily total {label} the limit ({prior:.0f} + {amt:.2f})",
                     "boundary", "state_transition", outcome, [i], fields, self._expected(outcome, crit), cons, setup)

    def statuses(self, i: int, crit: str) -> bool:
        made = False
        for ds, info in self.profile.items():
            cols = info.get("columns", {})
            id_col = next((c for c in cols if c.endswith("_id") and "." not in c), None)
            for col, ci in cols.items():
                if "." in col or "status" not in col or not id_col:
                    continue
                for v in ci.get("values", []):
                    if not re.search(rf"\b{re.escape(v)}\b", crit):
                        continue
                    outcome = "rejection" if REJECT_WORDS.search(crit.lower()) else "other"
                    ref = _f(id_col, "reference", rule=f"{ds} in status {v}", reference=f"{ds}.{id_col}[{col}={v}]")
                    self.add(f"{self.req.title} - {col} {v} is handled ({outcome})", "negative",
                             "equivalence_partitioning", outcome, [i],
                             [ref, _f("amount", "decimal", rule="small valid amount", fixed="50.00")],
                             self._expected(outcome, crit))
                    made = True
                # the valid partition: one ACTIVE record must still be accepted
                if made and "ACTIVE" in ci.get("values", []) and not self._active_done:
                    self._active_done = True
                    ref = _f(id_col, "reference", rule=f"{ds} in status ACTIVE (valid partition)",
                             reference=f"{ds}.{id_col}[{col}=ACTIVE]")
                    self.add(f"{self.req.title} - {col} ACTIVE is accepted (valid partition)", "positive",
                             "equivalence_partitioning", "success", [i],
                             [ref, _f("amount", "decimal", rule="small valid amount", fixed="50.00")],
                             self._expected("success", f"{ds} in status ACTIVE is processed normally"))
        return made

    def same_account(self, i: int, crit: str) -> bool:
        if not SAME_ACCOUNT.search(crit.lower()):
            return False
        dest = _f("destination_account_id", "reference", rule="same as source account",
                  reference="accounts.account_id[status=ACTIVE]" if self.has_accounts else None)
        self.add(f"{self.req.title} - source and destination are the same account", "negative",
                 "error_guessing", "rejection", [i],
                 [self.account(), dest, _f("amount", "decimal", rule="small valid amount", fixed="50.00")],
                 self._expected("rejection", crit),
                 [CrossFieldConstraint(left="destination_account_id", op="==", right="account_id", must_hold=True)])
        return True

    def design(self) -> TestCaseSet:
        gap = self.gap
        crit_idx = gap["uncovered"] if gap else range(len(self.req.acceptance_criteria))
        wanted = set(gap["missing_types"]) if gap else None
        for i in crit_idx:
            crit = self.req.acceptance_criteria[i]
            specific = [self.limits(i, crit), self.statuses(i, crit), self.same_account(i, crit)]
            if not any(specific):
                outcome = "rejection" if REJECT_WORDS.search(crit.lower()) else "success"
                self.add(f"Verify: {crit}", "negative" if outcome == "rejection" else "positive",
                         "error_guessing" if outcome == "rejection" else "happy_path",
                         outcome, [i], [self.account()], self._expected(outcome, crit))
        if wanted is None or "negative" in wanted:
            bad = _f("account_id", "account_number", valid=False, rule="account does not exist")
            self.add(f"{self.req.title} - unknown account is rejected", "negative", "error_guessing",
                     "rejection", [], [bad], "Rejected with a validation error", prio="P2")
        if self.req.risk == "high" and (wanted is None or "security" in wanted):
            self.add(f"{self.req.title} - unauthenticated request is rejected", "security", "error_guessing",
                     "rejection", [], [self.account()], "Rejected - the request is not authenticated")
        if wanted and "boundary" in wanted:
            for rule in self.req.business_rules:
                self.limits(self.req.acceptance_criteria.index(rule) if rule in self.req.acceptance_criteria else 0, rule)
        return TestCaseSet(test_cases=self.cases)


# ---------------------------------------------------------------- judge
def _tokens(s: str) -> set[str]:
    return set(WORD.findall(s.lower())) - STOP


def _case_text(c: TestCase) -> str:
    parts = [c.title, c.expected_result, *(s.action + " " + s.expected for s in c.steps),
             *(f"{f.rule} {f.fixed_value or ''}" for f in c.data_fields)]
    return " ".join(parts)


def supports(criterion: str, text: str) -> bool:
    cw, tw = _tokens(criterion), _tokens(text)
    if _floats(criterion) & _floats(text):
        return True
    return bool(cw) and len(cw & tw) / len(cw) >= 0.5


def verify(req: Requirement, cases: list[TestCase]) -> VerificationSet:
    req_nums = _floats(" ".join(req.acceptance_criteria + req.business_rules))
    allowed = {round(n + d, 2) for n in req_nums for d in (0, 0.01, -0.01, 1, -1)}
    out = []
    for c in cases:
        text = _case_text(c)
        n = len(req.acceptance_criteria)
        verified = [i for i in c.covers_criteria if 0 <= i < n and supports(req.acceptance_criteria[i], text)]
        unsupported = [f"value {v:g} in expected result" for v in _floats(c.expected_result) if v not in allowed]
        grounded = not unsupported
        overclaim = sorted(set(c.covers_criteria) - set(verified))
        verdict = "reject" if not grounded else ("revise" if overclaim else "accept")
        out.append(CaseVerdict(
            tc_id=c.tc_id, verified_criteria=verified, grounded=grounded, unsupported_claims=unsupported,
            verdict=verdict,
            reason="ok" if verdict == "accept" else f"unverified criteria {overclaim}" if grounded else "ungrounded values"))
    return VerificationSet(verdicts=out)


# ---------------------------------------------------------------- extraction
def extract(documents: list[tuple[str, str]]) -> RequirementSet:
    reqs: list[Requirement] = []
    n = 0
    for doc_name, text in documents:
        module, feature, bullets = "General", None, []

        def flush():
            nonlocal n
            if feature and bullets:
                n += 1
                reqs.append(Requirement(
                    req_id=f"TMP-{n:03d}", module=module, title=feature, description=feature,
                    acceptance_criteria=list(bullets),
                    business_rules=[b for b in bullets if NUM.search(b)],
                    risk="high" if module.lower() in {"payments", "auth", "transactions"} else "medium",
                    source=f"{doc_name} > {feature}",
                ))

        for line in text.splitlines():
            s = line.strip()
            if s.startswith("### "):
                flush(); feature, bullets = s[4:].strip(), []
            elif s.startswith("## "):
                flush(); module, feature, bullets = s[3:].strip(), None, []
            elif s.startswith(("- ", "* ")) and feature:
                bullets.append(s[2:].strip())
        flush()
    return RequirementSet(requirements=reqs)


class OfflineLLM:
    name = "offline-heuristic"

    def generate(self, system: str, prompt: str, schema: type[BaseModel], context: dict):
        if schema is RequirementSet:
            return extract(context["documents"])
        if schema is TestCaseSet:
            return _Designer(context["requirement"], context.get("data_profile", {}),
                             context.get("start_index", 1), context.get("gap")).design()
        if schema is VerificationSet:
            return verify(context["requirement"], context["test_cases"])
        raise NotImplementedError(schema)
