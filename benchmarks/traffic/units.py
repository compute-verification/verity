"""The one record every source reduces to: per LLM call, its session, its context, what of it is registered, what is new
to its session, and its output (all in tokens).

- ``ctx``: input tokens of the call (the whole context; rule 1 counts all of it).
- ``reg``: the leading tokens a registered prefix covers (shared system prompt, tool definitions; rule 2 counts ctx - reg).
- ``new``: tokens new to the session at this call, user and tool tokens only (rule 3 sums them per session; the session's
  first call has new = ctx - reg). The model's own earlier outputs are not new: they were generated inside the session.
- ``out``: generated tokens.
- ``rules``: which of the three rules the source can speak to (a length-only trace has no prefix or session structure).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FIELDS = ("sess", "turn", "ctx", "reg", "new", "out")


def save(path: Path, *, src: str, cls: str, rules: list[str], note: str, **arrays) -> None:
    missing = [f for f in FIELDS if f not in arrays]
    assert not missing, f"missing fields {missing}"
    n = len(arrays["ctx"])
    data = {f: np.asarray(arrays[f], dtype=np.int64) for f in FIELDS}
    assert all(len(v) == n for v in data.values())
    assert (data["reg"] <= data["ctx"]).all() and (data["new"] <= data["ctx"]).all() and (data["new"] >= 0).all()
    meta = {"src": src, "cls": cls, "rules": rules, "note": note, "calls": n,
            "sessions": int(len(np.unique(data["sess"])))}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, meta=json.dumps(meta), **data)


def load(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    return meta, {f: z[f] for f in FIELDS}
