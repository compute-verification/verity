"""The claim ids: the stable id of each security claim a protocol's output cites, with the Lean names it names.

`claim_ids.json`, beside this file, is written by `tools/verity/lean/claim_ids.py` from the locks: an id a premise entry of
a `lean-audit.json` carries is ``{"kind": "assumption", "lean": [the Lean names carrying it]}``; an id no lock carries,
which a reader here needs, is ``{"kind": "assumption" | "model" | "guarantee", "lean": [], "name": what it means}``.  An
id is kebab-case (`trusted-device`) or `<property>/<instance>` (`cr/sha-512`), and once cited it is never renamed."""
from __future__ import annotations

import json
from pathlib import Path

IDS: dict[str, dict] = json.loads(Path(__file__).with_name("claim_ids.json").read_text(encoding="utf-8"))


def get(cid: str) -> dict:
    """The row of claim id ``cid``; `KeyError` when the table has none."""
    try:
        return IDS[cid]
    except KeyError:
        raise KeyError(f"{cid!r} is no claim id of verity/commitments/claim_ids.json") from None
