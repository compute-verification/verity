"""`research data vocab-check`: what in a store's labels/ is off the vocabulary (README §8.1), read-only.

The audit reads the assertion FILES, not the catalog: the catalog stores every value as text (`index.scalar_text`), so `true` the
boolean and `"true"` the word look the same there, and the point of the check is to tell them apart.  Findings, each with counts,
the asserters and a few example targets:

    unknown_keys      keys no vocabulary entry describes
    alias_keys        old spellings of a key (`notes` / `verify_note` -> `note`, `batch` -> `B`, `checker` -> `verifier`)
    alias_values      old spellings of a value (`candidate=A` -> `A-GKR`)
    string_typed      bool / int / number keys holding the word (`zk="true"`) instead of the value
    off_enum          enum keys holding a value outside the closed set
    ref_keys          `path` / `result_artifact` / `constructed_from_attempt` written as labels (they are refs)
    measurement_keys  measurements / run parameters written as labels (`vocab.MEASUREMENT_KEYS`: `l`, `subbatches`, ...)
    unreadable        files that do not parse as an assertion

Nothing is written; correcting is the coordinator's call (`labels` are append-only: a correction is a new assertion, never an edit).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, TYPE_CHECKING

from . import vocab
from .model import Label

if TYPE_CHECKING:
    from .local import LocalStore

CATEGORIES = ("unknown_keys", "alias_keys", "alias_values", "string_typed", "off_enum", "ref_keys", "measurement_keys", "unreadable")
EXAMPLES = 3


def classify(label: Label) -> tuple[str, str] | None:
    """(category, detail) for one assertion, or None when it is on-vocabulary."""
    k, v = label.key, label.value
    if k in vocab.REF_KEYS:
        return "ref_keys", k
    if k in vocab.MEASUREMENT_KEYS:
        return "measurement_keys", k
    if k in vocab.KEY_ALIASES:
        return "alias_keys", f"{k} -> {vocab.KEY_ALIASES[k]}"
    entry = vocab.key(k)
    if entry is None:
        return "unknown_keys", k
    if isinstance(v, str) and (k, v) in vocab.VALUE_ALIASES:
        return "alias_values", f"{k}={v} -> {vocab.VALUE_ALIASES[(k, v)]}"
    if entry.form in (vocab.BOOL, vocab.INT, vocab.NUMBER) and isinstance(v, str):
        return "string_typed", f"{k} ({entry.form} held as a string" + (f": {v!r}" if entry.form == vocab.BOOL else "") + ")"
    if entry.check(v):
        return "off_enum", f"{k}={vocab.scalar_text(v)}"
    return None


def audit(store: LocalStore) -> dict[str, Any]:
    """The report: `{"labels": n, "findings": n, "by_category": {cat: {detail: {"count", "by": {...}, "examples": [target, ...]}}}}`."""
    cats: dict[str, dict[str, dict[str, Any]]] = {c: {} for c in CATEGORIES}
    n = 0
    for p in sorted((store.root / "labels").glob("*/*.json")):
        n += 1
        try:
            lab = Label.from_doc(json.loads(p.read_bytes()))
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            _hit(cats["unreadable"], f"{type(e).__name__}", p.parent.name, str(p.relative_to(store.root)))
            continue
        hit = classify(lab)
        if hit:
            cat, detail = hit
            _hit(cats[cat], detail, lab.by, lab.target)
    for c in cats.values():
        for d in c.values():
            d["by"] = dict(sorted(d["by"].items(), key=lambda kv: (-kv[1], kv[0])))
    return {"labels": n, "findings": sum(d["count"] for c in cats.values() for d in c.values()),
            "by_category": {c: dict(sorted(v.items(), key=lambda kv: (-kv[1]["count"], kv[0]))) for c, v in cats.items() if v}}


def _hit(bucket: dict[str, dict[str, Any]], detail: str, by: str, target: str) -> None:
    d = bucket.setdefault(detail, {"count": 0, "by": defaultdict(int), "examples": []})
    d["count"] += 1
    d["by"][by] += 1
    if len(d["examples"]) < EXAMPLES and target not in d["examples"]:
        d["examples"].append(target)


def render(rep: dict[str, Any]) -> str:
    lines = [f"{rep['labels']} assertion file(s); {rep['findings']} off the vocabulary"]
    for cat in CATEGORIES:
        rows = rep["by_category"].get(cat)
        if not rows:
            continue
        lines.append(f"\n{cat} ({sum(d['count'] for d in rows.values())}):")
        for detail, d in rows.items():
            who = ", ".join(f"{b} x{n}" if n > 1 else b for b, n in list(d["by"].items())[:4]) + (" ..." if len(d["by"]) > 4 else "")
            lines.append(f"  {d['count']:>5}  {detail:<60} by {who}")
            lines.append(f"         e.g. {', '.join(x[:40] for x in d['examples'])}")
    return "\n".join(lines)
