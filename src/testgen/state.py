"""
Persistent state that makes the pipeline safe to run on every release.

Stable requirement ids
    Identity = module + normalised title (fuzzy-matched, so a reworded title keeps its id).
    A content hash of acceptance criteria + business rules detects CHANGED requirements.
    Ids are never reused: a removed REQ-PAY-003 stays retired.

Incremental (diff) mode
    new / changed  -> regenerated
    unchanged      -> test cases, verdicts and reviewer edits reused as-is (no LLM call)
    removed        -> their test cases are marked obsolete (and flagged in JIRA)

Stable test case ids
    When a changed requirement is regenerated, each new case is matched to the most
    similar old case of the same type and inherits its tc_id (so the JIRA issue is
    updated, not duplicated). Unmatched new cases get fresh numbers; unmatched old
    cases become obsolete.

Everything lives in <state_dir>/state.json - commit it, or keep it on a shared volume.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from .coverage import similarity
from .models import CaseVerdict, Requirement, TestCase

TITLE_MATCH = 0.85      # reworded title still the same requirement
CONTENT_SAME = 0.97     # tiny wording drift in criteria counts as unchanged (numbers must match exactly)
TC_MATCH = 0.75         # regenerated case inherits the old id above this similarity


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _module_code(module: str) -> str:
    return (re.sub(r"[^A-Za-z0-9]", "", module).upper() + "XXX")[:3]


def content_text(r: Requirement) -> str:
    return _norm(" | ".join(r.acceptance_criteria) + " || " + " | ".join(r.business_rules))


def _numbers(text: str) -> list[str]:
    return sorted(re.findall(r"\d+(?:\.\d+)?", text.replace(",", "")))


def content_hash(r: Requirement) -> str:
    return hashlib.sha256(content_text(r).encode()).hexdigest()[:16]


@dataclass
class State:
    path: Path
    registry: dict = field(default_factory=dict)       # identity key -> {req_id, hash, text, module}
    counters: dict = field(default_factory=dict)       # module code -> last number
    requirements: dict = field(default_factory=dict)   # req_id -> requirement dict
    test_cases: dict = field(default_factory=dict)     # tc_id -> test case dict (current, non-obsolete)
    verdicts: dict = field(default_factory=dict)       # tc_id -> verdict dict
    tc_counters: dict = field(default_factory=dict)    # req_id -> last case number
    overrides: dict = field(default_factory=dict)      # tc_id -> reviewer-edited test case dict
    reviews: dict = field(default_factory=dict)        # tc_id -> {status, jira_status}
    obsolete: dict = field(default_factory=dict)       # tc_id -> reason
    jira: dict = field(default_factory=lambda: {"dry_run": {}, "jira": {}})  # mode -> tc_id -> issue info

    # ------------------------------------------------------------ persistence
    @classmethod
    def load(cls, state_dir: str | Path) -> "State":
        path = Path(state_dir) / "state.json"
        st = cls(path=path)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            for k, v in data.items():
                setattr(st, k, v)
        return st

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: v for k, v in self.__dict__.items() if k != "path"}
        self.path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    def previous_requirements(self) -> list[Requirement]:
        return [Requirement(**d) for d in self.requirements.values()]

    # ------------------------------------------------------------ requirement ids
    def assign_ids(self, reqs: list[Requirement]) -> tuple[dict[str, str], list[str]]:
        """Rewrite req.req_id in place. Returns ({req_id: new|changed|unchanged}, removed_ids)."""
        status: dict[str, str] = {}
        used: set[str] = set()
        new_registry: dict = {}
        for r in reqs:
            key = f"{_norm(r.module)}|{_norm(r.title)}"
            entry = self.registry.get(key)
            if entry is None or entry["req_id"] in used:
                entry = self._fuzzy(r, used)
            text = content_text(r)
            if entry:
                r.req_id = entry["req_id"]
                # wording drift is tolerated, but ANY changed number (limit, threshold) is a change
                same = entry["hash"] == content_hash(r) or (
                    SequenceMatcher(None, entry["text"], text).ratio() >= CONTENT_SAME
                    and _numbers(entry["text"]) == _numbers(text))
                status[r.req_id] = "unchanged" if same else "changed"
            else:
                code = _module_code(r.module)
                self.counters[code] = self.counters.get(code, 0) + 1
                r.req_id = f"REQ-{code}-{self.counters[code]:03d}"
                status[r.req_id] = "new"
            used.add(r.req_id)
            new_registry[key] = {"req_id": r.req_id, "hash": content_hash(r), "text": text, "module": r.module}
        removed = [e["req_id"] for e in self.registry.values() if e["req_id"] not in used]
        # keep retired entries so their ids are never handed out again
        for k, e in self.registry.items():
            if e["req_id"] in removed:
                new_registry.setdefault(k, e | {"retired": True})
        self.registry = new_registry
        return status, removed

    def diff_requirements(self, reqs: list[Requirement]) -> tuple[dict[str, str], list[str]]:
        """Compare requirements that ALREADY have stable ids (e.g. JIRA story keys, or ids assigned by
        assign_ids) with what was generated last time. Returns ({req_id: new|changed|unchanged}, removed)."""
        status: dict[str, str] = {}
        for r in reqs:
            old = self.requirements.get(r.req_id)
            if old is None:
                status[r.req_id] = "new"
                continue
            before, after = content_text(Requirement(**old)), content_text(r)
            same = before == after or (SequenceMatcher(None, before, after).ratio() >= CONTENT_SAME
                                       and _numbers(before) == _numbers(after))
            status[r.req_id] = "unchanged" if same else "changed"
        ids = {r.req_id for r in reqs}
        return status, [rid for rid in self.requirements if rid not in ids]

    def _fuzzy(self, r: Requirement, used: set[str]) -> dict | None:
        title = _norm(r.title)
        best, score = None, 0.0
        for key, e in self.registry.items():
            if e.get("retired") or e["req_id"] in used or _norm(e["module"]) != _norm(r.module):
                continue
            s = SequenceMatcher(None, key.split("|", 1)[1], title).ratio()
            if s > score:
                best, score = e, s
        return best if score >= TITLE_MATCH else None

    # ------------------------------------------------------------ test case ids
    def cases_for(self, req_id: str, with_overrides: bool = True) -> list[TestCase]:
        out = []
        for tc_id, d in self.test_cases.items():
            if d["req_id"] == req_id:
                out.append(TestCase(**(self.overrides.get(tc_id, d) if with_overrides else d)))
        return sorted(out, key=lambda c: c.tc_id)

    def reconcile(self, req_id: str, new_cases: list[TestCase],
                  verdicts: dict[str, CaseVerdict]) -> tuple[list[TestCase], dict[str, CaseVerdict], list[str]]:
        """Give regenerated cases stable ids. Returns (cases, verdicts re-keyed, obsolete old ids)."""
        old = self.cases_for(req_id, with_overrides=False)
        temp_ids = [c.tc_id for c in new_cases]
        taken: set[str] = set()
        final: list[str | None] = [None] * len(new_cases)
        # greedy best-match over all (new, old) pairs of the same test type
        pairs = sorted(((similarity(n, o), i, o.tc_id) for i, n in enumerate(new_cases) for o in old
                        if n.test_type == o.test_type), reverse=True)
        for score, i, old_id in pairs:
            if score < TC_MATCH:
                break
            if final[i] is None and old_id not in taken:
                final[i], _ = old_id, taken.add(old_id)
        last = max([self.tc_counters.get(req_id, 0)] + [int(o.tc_id.rsplit("-", 1)[1]) for o in old])
        for i in range(len(new_cases)):
            if final[i] is None:
                last += 1
                final[i] = f"TC-{req_id}-{last:02d}"
        self.tc_counters[req_id] = last
        new_verdicts = {}
        for c, tmp, fid in zip(new_cases, temp_ids, final):
            c.tc_id, c.req_id = fid, req_id
            if tmp in verdicts:
                new_verdicts[fid] = verdicts[tmp].model_copy(update={"tc_id": fid})
        obsolete = [o.tc_id for o in old if o.tc_id not in taken]
        return new_cases, new_verdicts, obsolete

    # ------------------------------------------------------------ commit
    def commit(self, reqs: list[Requirement], cases: list[TestCase], verdicts: dict[str, CaseVerdict],
               obsolete: dict[str, str], failed: set[str] | frozenset = frozenset()) -> None:
        # A requirement whose regeneration FAILED keeps its previous version, so the next run
        # still sees it as changed (or new) and retries it instead of silently skipping it.
        stored = {}
        for r in reqs:
            if r.req_id in failed:
                if r.req_id in self.requirements:
                    stored[r.req_id] = self.requirements[r.req_id]
            else:
                stored[r.req_id] = r.model_dump()
        self.requirements = stored
        live = {c.tc_id for c in cases}
        # store the generated version as the base; reviewer edits stay in self.overrides
        self.test_cases = {
            c.tc_id: self.test_cases[c.tc_id] if c.tc_id in self.overrides and c.tc_id in self.test_cases
            else c.model_dump()
            for c in cases
        }
        self.verdicts = {tid: v.model_dump() for tid, v in verdicts.items() if tid in live}
        self.obsolete |= obsolete
        for tid in obsolete:
            self.overrides.pop(tid, None)
