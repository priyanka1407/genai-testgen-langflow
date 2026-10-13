"""
Stage 3: synthetic, scenario-aligned test data.

The LLM produced a *recipe* per test case. This module turns it into concrete rows:

  data_fields   reference -> record sampled from the consolidated (joined) sources,
                             filtered by state, e.g. accounts.account_id[status=FROZEN,
                             customers.kyc_status=VERIFIED]
                valid     -> value inside the declared range / enum / format
                invalid   -> value deliberately outside it
                fixed     -> used verbatim (boundary values)
  constraints   cross-field rules (amount <= account_id.balance, dest == source ...).
                Reference records are chosen to satisfy them; free fields are repaired;
                anything that cannot be satisfied is reported in _warnings.
  setup         prior state rows (e.g. transfers already made today) with exact totals.

Deterministic: seeded per test case id -> identical data on every rerun.
No PII leaves this process; names/emails are synthetic.
"""
from __future__ import annotations

import hashlib
import random
import string
import uuid
from datetime import date, timedelta

from .models import CrossFieldConstraint, DataField, SetupRecord, TestCase

FIRST = ["Aarav", "Diya", "Liam", "Maya", "Noah", "Priya", "Omar", "Sofia", "Kenji", "Zara"]
LAST = ["Sharma", "Patel", "Smith", "Garcia", "Chen", "Okafor", "Silva", "Kumar", "Brown", "Ito"]
CURRENCIES = ["USD", "EUR", "GBP", "INR", "SGD"]
NEGATE = {"<=": ">", "<": ">=", ">=": "<", ">": "<=", "==": "!=", "!=": "=="}


def _rng(seed_text: str) -> random.Random:
    return random.Random(int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16))


def _iban(r: random.Random, valid: bool) -> str:
    bban = "".join(r.choices(string.digits, k=18))
    country = "DE"
    # ISO 13616 mod-97 check digits
    num = int("".join(str(int(c, 36)) for c in bban + country + "00"))
    check = 98 - num % 97
    if not valid:
        check = (check + 1) % 97 or 1
    return f"{country}{check:02d}{bban}"


def parse_reference(ref: str) -> tuple[str, str, dict[str, str]]:
    """'accounts.account_id[status=FROZEN,customers.kyc_status=VERIFIED]'
    -> ('accounts', 'account_id', {'status': 'FROZEN', 'customers.kyc_status': 'VERIFIED'})"""
    base, _, flt = ref.partition("[")
    ds, _, col = base.partition(".")
    filters = dict(p.split("=", 1) for p in flt.rstrip("]").split(",") if "=" in p)
    return ds.strip(), col.strip(), {k.strip(): v.strip() for k, v in filters.items()}


def reference_candidates(f: DataField, refs: dict[str, list[dict]]) -> list[dict]:
    ds, col, filters = parse_reference(f.reference or "")
    return [row for row in refs.get(ds, []) if row.get(col) not in (None, "")
            and all(str(row.get(k)) == v for k, v in filters.items())]


def _value(f: DataField, r: random.Random):
    """Value for a non-reference field (or a reference field with no matching record)."""
    if f.fixed_value is not None:
        return f.fixed_value
    lo = f.min_value if f.min_value is not None else 1
    hi = f.max_value if f.max_value is not None else max(lo * 100, 1000)
    k = f.kind
    if k in ("integer", "decimal"):
        v = r.uniform(lo, hi) if f.valid else r.choice([lo - r.uniform(0.01, 10), hi + r.uniform(0.01, 10)])
        return int(v) if k == "integer" else round(v, 2)
    if k == "enum" and f.allowed_values:
        return r.choice(f.allowed_values) if f.valid else "INVALID_" + r.choice(f.allowed_values)
    if k == "currency":
        return r.choice(f.allowed_values or CURRENCIES) if f.valid else "XXX1"
    if k == "date":
        d = date.today() + timedelta(days=r.randint(0, 30))
        return d.isoformat() if f.valid else "2024-02-30"
    if k == "boolean":
        return r.choice([True, False])
    if k == "name":
        return f"{r.choice(FIRST)} {r.choice(LAST)}" if f.valid else ""
    if k == "email":
        user = r.choice(FIRST).lower() + str(r.randint(1, 999))
        return f"{user}@example.test" if f.valid else f"{user}@@invalid"
    if k == "phone":
        return "+1555" + "".join(r.choices(string.digits, k=7)) if f.valid else "12ab"
    if k == "uuid":
        return str(uuid.UUID(int=r.getrandbits(128))) if f.valid else "not-a-uuid"
    if k in ("account_number", "reference"):
        return "".join(r.choices(string.digits, k=12)) if f.valid else "999999999999"
    if k == "iban":
        return _iban(r, f.valid)
    length = int(hi if f.max_value is not None else 12)
    if not f.valid and f.max_value is not None:
        length = int(f.max_value) + 1
    return "".join(r.choices(string.ascii_letters, k=max(1, length)))


# ---------------------------------------------------------------- constraints
_MISSING = object()


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def resolve(token: str, row: dict, ctx: dict[str, dict], names: frozenset = frozenset()):
    """Field name -> its value; '<ref_field>.<column>' -> column of the record picked
    for that reference field; anything else -> literal. _MISSING if not resolvable yet."""
    if token in row:
        return row[token]
    if token in names:
        return _MISSING  # a field of this test case that is not filled yet
    if _num(token) is not None:
        return token
    head, _, rest = token.partition(".")
    if rest and (head in ctx or head in names):
        return ctx[head].get(rest, _MISSING) if head in ctx else _MISSING
    return token


def holds(c: CrossFieldConstraint, row: dict, ctx: dict, names: frozenset = frozenset()) -> bool | None:
    lv, rv = resolve(c.left, row, ctx, names), resolve(c.right, row, ctx, names)
    if lv is _MISSING or rv is _MISSING or c.left not in row:
        return None
    ln, rn = _num(lv), _num(rv)
    a, b = (ln, rn) if ln is not None and rn is not None else (str(lv), str(rv))
    return {"<=": a <= b, "<": a < b, ">=": a >= b, ">": a > b, "==": a == b, "!=": a != b}[c.op]


def _satisfied(cons: list[CrossFieldConstraint], row: dict, ctx: dict, names: frozenset) -> bool:
    return all(holds(c, row, ctx, names) in (None, c.must_hold) for c in cons)


def _repair(c: CrossFieldConstraint, f: DataField, row: dict, ctx: dict, r: random.Random,
            names: frozenset) -> None:
    """Change the free field c.left so the constraint ends up as c.must_hold requires."""
    target = c.op if c.must_hold else NEGATE[c.op]
    rv = resolve(c.right, row, ctx, names)
    if rv is _MISSING:
        return
    b = _num(rv)
    if target == "==":
        row[c.left] = rv
        return
    if target == "!=":
        row[c.left] = (b + 1) if b is not None else f"{rv}_X"
        return
    if b is None:
        return
    lo = f.min_value if f.min_value is not None else 0
    if target in ("<=", "<"):
        v = r.uniform(min(lo, b), b)
        v = min(v, b - 0.01) if target == "<" else v
    else:
        v = r.uniform(b, b + max(1.0, abs(b) * 0.1))
        v = max(v, b + 0.01) if target == ">" else v
    row[c.left] = int(v) if f.kind == "integer" else round(v, 2)


# ---------------------------------------------------------------- setup
def _setup_rows(s: SetupRecord, refs: dict, r: random.Random) -> list[dict]:
    rows = []
    for _ in range(max(0, s.count)):
        rec = {}
        for f in s.fields:
            cands = reference_candidates(f, refs) if f.reference else []
            _, col, _ = parse_reference(f.reference or "")
            rec[f.name] = r.choice(cands)[col] if cands else _value(f, r)
        rows.append(rec)
    if rows and s.total_field and s.total_value is not None:
        weights = [r.uniform(0.5, 1.5) for _ in rows]
        parts = [round(s.total_value * w / sum(weights), 2) for w in weights]
        parts[-1] = round(s.total_value - sum(parts[:-1]), 2)  # exact total
        for rec, p in zip(rows, parts):
            rec[s.total_field] = p
    return rows


# ---------------------------------------------------------------- rows
def build_row(tc: TestCase, refs: dict[str, list[dict]], r: random.Random) -> dict:
    row: dict = {}
    ctx: dict[str, dict] = {}
    warnings: list[str] = []
    fields = {f.name: f for f in tc.data_fields}
    names = frozenset(fields)

    # 1. free + fixed fields first, so references can be chosen to fit them
    for f in tc.data_fields:
        if not f.reference:
            row[f.name] = _value(f, r)

    # 2. reference fields: pick a record that satisfies every constraint it takes part in
    for f in tc.data_fields:
        if not f.reference:
            continue
        _, col, _ = parse_reference(f.reference)
        cands = reference_candidates(f, refs)
        if not cands:
            row[f.name] = _value(f, r)
            warnings.append(f"no source record matches {f.reference}; generated value used")
            continue
        r.shuffle(cands)
        pick = next((c for c in cands if _satisfied(tc.constraints, row | {f.name: c[col]}, ctx | {f.name: c}, names)), None)
        if pick is None:
            pick = cands[0]
        row[f.name], ctx[f.name] = pick[col], pick

    # 3. repair free fields for anything still violated
    for c in tc.constraints:
        if holds(c, row, ctx, names) not in (None, c.must_hold):
            f = fields.get(c.left)
            if f and not f.reference and f.fixed_value is None:
                _repair(c, f, row, ctx, r, names)
        if holds(c, row, ctx, names) not in (None, c.must_hold):
            warnings.append(f"constraint {c.left} {c.op} {c.right} (must_hold={c.must_hold}) not satisfiable")

    # expose the useful columns of picked records (e.g. balance) for the runner / debugging
    for name, rec in ctx.items():
        row[f"_{name}_record"] = rec
    if tc.setup:
        row["_setup"] = {s.entity: _setup_rows(s, refs, r) for s in tc.setup}
    row["_tc_id"] = tc.tc_id
    row["_expect"] = tc.expected_outcome
    if warnings:
        row["_warnings"] = warnings
    return row


def build_rows(tc: TestCase, refs: dict[str, list[dict]], rows_per_case: int = 3) -> list[dict]:
    """rows_per_case variants for data-driven execution of one test case."""
    if not tc.data_fields and not tc.setup:
        return []
    deterministic = all(f.fixed_value is not None or f.reference for f in tc.data_fields)
    n = 1 if deterministic and not any(f.reference for f in tc.data_fields) else rows_per_case
    out, seen = [], set()
    for i in range(n):
        row = build_row(tc, refs, _rng(f"{tc.tc_id}:{i}"))
        key = repr(sorted((k, str(v)) for k, v in row.items()))
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


def build_dataset(cases: list[TestCase], refs: dict[str, list[dict]], rows_per_case: int = 3) -> dict[str, list[dict]]:
    return {tc.tc_id: build_rows(tc, refs, rows_per_case) for tc in cases}
