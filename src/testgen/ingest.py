"""
Input ingestion.

- load_documents(): release notes + requirement specs (.md/.txt, optional .docx/.pdf)
- load_data_sources(): CSV / JSON datasets from multiple systems, consolidated into
  one reference store that the synthetic-data engine draws from.
- consolidate(): joins the datasets on shared keys (accounts <-> customers ...).
- profile_sources(): a *schema profile* (columns, types, ranges, enums) - this is what
  the LLM sees, never the raw rows, so no production data is sent to the model.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

TEXT_EXT = {".md", ".txt"}


def _read_doc(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in TEXT_EXT:
        return path.read_text(encoding="utf-8")
    if ext == ".docx":
        import docx  # pip install python-docx
        return "\n".join(p.text for p in docx.Document(str(path)).paragraphs)
    if ext == ".pdf":
        from pypdf import PdfReader  # pip install pypdf
        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    raise ValueError(f"Unsupported document type: {path}")


def load_documents(paths: list[str | Path]) -> list[tuple[str, str]]:
    """Return [(doc_name, text)]. Directories are expanded."""
    docs: list[tuple[str, str]] = []
    for p in map(Path, paths):
        files = sorted(f for f in p.rglob("*") if f.is_file()) if p.is_dir() else [p]
        for f in files:
            if f.suffix.lower() in TEXT_EXT | {".docx", ".pdf"}:
                docs.append((f.name, _read_doc(f)))
    return docs


def format_documents(docs: list[tuple[str, str]]) -> str:
    return "\n\n".join(f'<document name="{n}">\n{t}\n</document>' for n, t in docs)


# ---------------------------------------------------------------- data sources
def load_data_sources(paths: list[str | Path]) -> dict[str, list[dict]]:
    """{dataset_name: rows}. Dataset name = file stem (accounts.csv -> 'accounts')."""
    store: dict[str, list[dict]] = {}
    for p in map(Path, paths):
        files = sorted(p.glob("*")) if p.is_dir() else [p]
        for f in files:
            if f.suffix.lower() == ".csv":
                with f.open(newline="", encoding="utf-8") as fh:
                    store[f.stem] = list(csv.DictReader(fh))
            elif f.suffix.lower() == ".json":
                data = json.loads(f.read_text(encoding="utf-8"))
                store[f.stem] = data if isinstance(data, list) else [data]
    return store


def consolidate(store: dict[str, list[dict]]) -> tuple[dict[str, list[dict]], list[dict]]:
    """Join datasets on shared key columns (*_id / *_key / *_code) so each record carries its
    related data, e.g. every account row gets 'customers.kyc_status', 'customers.daily_limit'.

    A join A -> B on key k is made when k is unique in B (many-to-one lookup).
    Returns (enriched_store, relations). Original datasets are not modified.
    """
    keyish = lambda c: c.endswith(("_id", "_key", "_code"))  # noqa: E731
    relations = []
    enriched = {name: [dict(r) for r in rows] for name, rows in store.items()}
    for a, a_rows in store.items():
        if not a_rows:
            continue
        for b, b_rows in store.items():
            if a == b or not b_rows:
                continue
            for k in (c for c in a_rows[0] if keyish(c) and c in b_rows[0]):
                lookup = {}
                for r in b_rows:
                    lookup.setdefault(r[k], []).append(r)
                if any(len(v) > 1 for v in lookup.values()):
                    continue  # not unique in B -> would fan out, skip
                matched = 0
                for row in enriched[a]:
                    hit = lookup.get(row.get(k))
                    if hit:
                        matched += 1
                        row.update({f"{b}.{c}": v for c, v in hit[0].items() if c != k})
                relations.append({"from": a, "to": b, "on": k, "matched": matched, "rows": len(a_rows)})
    return enriched, relations


def _infer(values: list) -> str:
    def is_num(v):
        try:
            float(v); return True
        except (TypeError, ValueError):
            return False
    if values and all(is_num(v) for v in values):
        return "number"
    if values and all(str(v).lower() in {"true", "false"} for v in values):
        return "boolean"
    return "string"


def _is_identifier(col: str, vals: list) -> bool:
    """Key-like name, or TEXT that differs on every row (emails, names, timezones of single customers...).
    Numeric measures like balance stay visible as a min/max range - an aggregate, not an identifier."""
    base = col.rsplit(".", 1)[-1]
    unique_text = len(vals) > 1 and _infer(vals) == "string" and len({str(v) for v in vals}) == len(vals)
    return base.endswith(("_id", "_key", "_code")) or unique_text


def profile_sources(store: dict[str, list[dict]], max_enum: int = 12) -> dict:
    """Column-level profile: type, numeric range or small category set, null count.

    Privacy: identifier-like columns are reported only as {"type": "identifier", "distinct": n};
    their values and ranges never reach the LLM. Values are listed only for repeating categories
    (status, currency, segment...), which the LLM needs to write filters like [status=FROZEN].
    """
    profile: dict = {}
    for name, rows in store.items():
        cols: dict = {}
        columns = list(dict.fromkeys(c for r in rows for c in r))  # union, joined rows may differ
        for col in columns:
            vals = [r.get(col) for r in rows if r.get(col) not in (None, "")]
            nulls = len(rows) - len(vals)
            if _is_identifier(col, vals):
                cols[col] = {"type": "identifier", "nulls": nulls, "distinct": len({str(v) for v in vals})}
                continue
            kind = _infer(vals)
            info: dict = {"type": kind, "nulls": nulls}
            if kind == "number":
                nums = [float(v) for v in vals]
                info |= {"min": min(nums), "max": max(nums)}
            distinct = sorted({str(v) for v in vals})
            if len(distinct) <= max_enum and kind != "number":
                info["values"] = distinct
            cols[col] = info
        profile[name] = {"rows": len(rows), "columns": cols}
    return profile
