"""
Read a JIRA epic and its user stories and turn them into Requirements.

Each STORY becomes one requirement whose id is the story key (PAY-101), so test cases are
traceable as TC-PAY-101-01. The epic itself becomes a requirement too if it has its own
acceptance criteria. Change detection then works per story: edit one story's acceptance
criteria and only that story's test cases are regenerated.

Where acceptance criteria are found (first match wins):
  1. a dedicated field - JIRA_AC_FIELD (e.g. customfield_10045) or auto-detected by its
     display name containing "acceptance criteria"
  2. an "Acceptance Criteria" section in the description (h3. / ## / *bold* heading,
     bullets, numbered items or Given/When/Then)
  3. no criteria at all -> the LLM extracts them from the story text (or, offline,
     each sentence of the description becomes a criterion)

Sources
  RestJiraReader  - JIRA Cloud (email + API token) or Server/Data Center (personal access token).
                    Children found via `parent = EPIC`, falling back to `"Epic Link" = EPIC`.
  FileJiraReader  - local JSON export (samples/jira/<EPIC>.json) for demos without JIRA access.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Protocol

from .models import Requirement

HIGH_PRIORITIES = {"highest", "high", "critical", "blocker"}
HIGH_RISK_WORDS = re.compile(r"payment|transfer|money|fund|auth|login|password|otp|security|fraud|limit", re.I)
AC_HEADING = re.compile(r"^\s*(?:h[1-6]\.\s*|#{1,6}\s*)?[*_]*\s*acceptance criteria\s*:?\s*[*_]*\s*:?\s*$", re.I)
# "# item" is a numbered list item in JIRA wiki markup, so only ##..###### count as headings
ANY_HEADING = re.compile(r"^\s*(?:h[1-6]\.\s+|#{2,6}\s+)|^\s*[*_]{1,2}[^*_]+[*_]{1,2}\s*:?\s*$")
BULLET = re.compile(r"^(?:[*#-]+|\d+[.)])\s+(.*)$")
CONTINUATION = re.compile(r"^(when|then|and|but)\b", re.I)


# ---------------------------------------------------------------- text helpers
def adf_to_text(node) -> str:
    """Atlassian Document Format (JIRA Cloud rich text) -> wiki-like plain text."""
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return "".join(adf_to_text(n) for n in node)
    kind, content = node.get("type"), node.get("content", [])
    if kind == "text":
        return node.get("text", "")
    if kind == "hardBreak":
        return "\n"
    if kind == "heading":
        return f"h{node.get('attrs', {}).get('level', 3)}. {adf_to_text(content)}\n"
    if kind in ("bulletList", "orderedList"):
        return "".join(f"* {adf_to_text(item).strip()}\n" for item in content)
    if kind == "paragraph":
        return adf_to_text(content) + "\n"
    return adf_to_text(content)


def field_text(value) -> str:
    if isinstance(value, dict) and value.get("type") == "doc":
        return adf_to_text(value)
    if isinstance(value, list):
        return "\n".join(f"* {field_text(v)}" for v in value)
    return "" if value is None else str(value)


def parse_criteria(lines: list[str]) -> list[str]:
    """Bullets / numbered items / Given-When-Then blocks -> one string per criterion."""
    out: list[str] = []
    for raw in lines:
        s = raw.strip()
        if not s:
            continue
        m = BULLET.match(s)
        body = (m.group(1) if m else s).strip()
        if CONTINUATION.match(body) and out:
            out[-1] = f"{out[-1]} {body}"          # When/Then/And continue the previous Given
        elif body:
            out.append(body)
    return out


def split_description(text: str) -> tuple[str, list[str]]:
    """Returns (narrative, acceptance criteria found in an 'Acceptance Criteria' section)."""
    narrative, ac_lines, in_ac = [], [], False
    for line in text.replace("\r\n", "\n").split("\n"):
        if AC_HEADING.match(line):
            in_ac = True
            continue
        if in_ac and ANY_HEADING.match(line):
            in_ac = False
        (ac_lines if in_ac else narrative).append(line)
    return "\n".join(narrative).strip(), parse_criteria(ac_lines)


def _sentences(text: str) -> list[str]:
    clean = re.sub(r"(?m)^\s*(?:h[1-6]\.|#+|[*-])\s*", "", text)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", clean) if len(s.strip()) > 15]


# ---------------------------------------------------------------- sources
class EpicSource(Protocol):
    def read(self, epic_key: str) -> tuple[dict, list[dict], dict]: ...   # (epic, stories, field names)


class FileJiraReader:
    """Reads <folder>/<EPIC>.json shaped like JIRA REST issues: {"epic": {...}, "stories": [...], "names": {...}}."""

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)

    def read(self, epic_key: str):
        path = self.folder / f"{epic_key}.json"
        if not path.exists():
            raise FileNotFoundError(f"No local export for {epic_key}: expected {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return data["epic"], data.get("stories", []), data.get("names", {})


class RestJiraReader:
    """JIRA Cloud: user = account email, token = API token.  Server/DC: token = personal access token (no user)."""

    def __init__(self, url: str | None = None, user: str | None = None, token: str | None = None, session=None):
        import requests

        self.base = (url or os.environ["JIRA_URL"]).rstrip("/")
        user = user or os.getenv("JIRA_USER")
        token = token or os.getenv("JIRA_API_TOKEN") or os.getenv("JIRA_PAT")
        if not token:
            raise RuntimeError("No JIRA token: set JIRA_API_TOKEN (Cloud) or JIRA_PAT (Server/DC), or fill the node field")
        self.s = session or requests.Session()
        if user:
            self.s.auth = (user, token)                              # Cloud: email + API token
        else:
            self.s.headers["Authorization"] = f"Bearer {token}"     # Server/DC: personal access token

    def _get(self, path: str, **params):
        r = self.s.get(f"{self.base}{path}", params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def search_keys(self, jql: str) -> list[str]:
        """All issue keys for a JQL query. Uses /rest/api/2/search; on JIRA Cloud, where that endpoint
        has been retired, falls back to /rest/api/3/search/jql (token pagination)."""
        keys: list[str] = []
        r = self.s.get(f"{self.base}/rest/api/2/search", params={"jql": jql, "fields": "summary", "maxResults": 100},
                       timeout=30)
        if r.status_code in (404, 410):
            token = None
            while True:
                params = {"jql": jql, "fields": "summary", "maxResults": 100}
                if token:
                    params["nextPageToken"] = token
                page = self._get("/rest/api/3/search/jql", **params)
                keys += [i["key"] for i in page.get("issues", [])]
                token = page.get("nextPageToken")
                if page.get("isLast", True) or not token:
                    return keys
        r.raise_for_status()
        start, page = 0, r.json()
        while True:
            keys += [i["key"] for i in page.get("issues", [])]
            start += len(page.get("issues", []))
            if not page.get("issues") or start >= page.get("total", 0):
                return keys
            page = self._get("/rest/api/2/search", jql=jql, fields="summary", maxResults=100, startAt=start)

    def issue(self, key: str) -> tuple[dict, dict]:
        data = self._get(f"/rest/api/2/issue/{key}", expand="names")   # v2 returns description as wiki text
        return data, data.get("names", {})

    def read(self, epic_key: str):
        import requests

        epic, names = self.issue(epic_key)
        keys: list[str] = []
        for jql in (f"parent = {epic_key}", f'"Epic Link" = {epic_key}'):
            try:
                keys = self.search_keys(f"{jql} ORDER BY key")
            except requests.HTTPError:
                continue    # e.g. "Epic Link" does not exist on this instance
            if keys:
                break
        stories = []
        for k in keys:
            issue, n = self.issue(k)
            names |= n
            stories.append(issue)
        return epic, stories, names


# ---------------------------------------------------------------- issue -> requirement
def find_ac_field(names: dict, configured: str | None = None) -> str | None:
    if configured:
        return configured
    return next((fid for fid, name in names.items() if "acceptance criteria" in str(name).lower()), None)


def _risk(fields: dict, text: str) -> str:
    """JIRA priority decides when it is set; keywords only fill in when there is no priority."""
    prio = str((fields.get("priority") or {}).get("name", "")).lower()
    if prio in HIGH_PRIORITIES:
        return "high"
    if prio in ("low", "lowest", "minor", "trivial"):
        return "low"
    if prio:
        return "medium"
    return "high" if HIGH_RISK_WORDS.search(text) else "medium"


def issue_to_requirement(issue: dict, epic: dict, ac_field: str | None, llm=None) -> Requirement | None:
    f = issue.get("fields", {})
    summary = f.get("summary") or issue["key"]
    narrative, criteria = split_description(field_text(f.get("description")))
    if ac_field and f.get(ac_field):
        criteria = parse_criteria(field_text(f.get(ac_field)).split("\n")) or criteria
    used_llm = False
    if not criteria and llm is not None and narrative:
        from .generator import extract_requirements   # local import: generator imports models only
        extracted = extract_requirements(llm, [(issue["key"], f"### {summary}\n{narrative}")])
        criteria = [c for r in extracted for c in r.acceptance_criteria]
        used_llm = bool(criteria)
    if not criteria:
        criteria = _sentences(narrative)
    if not criteria:
        return None
    epic_name = (epic.get("fields") or {}).get("summary") or epic.get("key", "")
    return Requirement(
        req_id=issue["key"],
        module=epic_name,
        title=summary,
        description=narrative or summary,
        acceptance_criteria=criteria,
        business_rules=[c for c in criteria if re.search(r"\d", c)],
        risk=_risk(f, f"{summary} {narrative} {' '.join(criteria)}"),
        source=f"JIRA {issue['key']} (epic {epic.get('key')})" + (" - criteria extracted by LLM" if used_llm else ""),
    )


def read_epic(epic_key: str, source: EpicSource, ac_field: str | None = None, llm=None) -> tuple[list[Requirement], dict]:
    """Returns (requirements, info). info: epic summary, story keys, which stories needed the LLM / were skipped."""
    epic, stories, names = source.read(epic_key)
    field = find_ac_field(names, ac_field or os.getenv("JIRA_AC_FIELD"))
    reqs: list[Requirement] = []
    skipped: list[str] = []
    # The epic itself counts only if it has EXPLICIT criteria (AC field or AC section), not just an overview
    epic_fields = epic.get("fields", {})
    if (field and epic_fields.get(field)) or split_description(field_text(epic_fields.get("description")))[1]:
        epic_req = issue_to_requirement(epic, epic, field, llm=None)
        if epic_req:
            reqs.append(epic_req)
    for story in stories:
        r = issue_to_requirement(story, epic, field, llm)
        (reqs.append(r) if r else skipped.append(story["key"]))
    info = {
        "epic": epic_key, "epic_summary": (epic.get("fields") or {}).get("summary"), "ac_field": field,
        "stories": [s["key"] for s in stories], "skipped_no_content": skipped,
        "llm_extracted": [r.req_id for r in reqs if "extracted by LLM" in r.source],
    }
    return reqs, info


def get_source(kind: str, folder: str = "samples/jira", url: str | None = None, user: str | None = None,
               token: str | None = None) -> EpicSource:
    if kind == "file":
        return FileJiraReader(folder)
    return RestJiraReader(url or None, user or None, token or None)
