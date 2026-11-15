"""Every premise a deployed guarantee rests on is on the trust list (`tools/verity/lean/trust-list.json`), in its
`trust_gap` entry, or pending a ruling there; the list's entries are each one line and scoped only to deployed
guarantees, and a pending entry holds only while it is still off the list and out of the gap."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _posture():
    spec = importlib.util.spec_from_file_location("lean_posture", ROOT / "tools" / "verity" / "lean" / "posture.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _marks(postures: list[dict]) -> dict[tuple[str, str], str]:
    return {(p["name"], n): m for p in postures for n, m in
            [*p["assumptions"].items(), *((n, e["mark"]) for n, e in p["open"].items())]}


def test_a_deployed_guarantee_rests_on_nothing_off_the_trust_list_but_its_gap() -> None:
    posture = _posture()
    trust = json.loads(posture.TRUST_LIST.read_text())
    marks = _marks(posture.postures(posture.locks(ROOT), trust, posture._check()))
    off = sorted(f"{g}: {p}" for (g, p), m in marks.items() if m == "OFF")
    assert not off, (f"deployed guarantees rest on premises off the trust list and out of their `trust_gap`: {off}; a "
                     "premise goes on the list (tools/verity/lean/trust-list.json) only by Daniel's ruling, else into "
                     "the guarantee's `trust_gap` entry in its lock")
    stale = sorted(f"{g}: {p}" for g, ps in trust["pending"].items() for p in ps if marks.get((g, p)) != "pending")
    assert not stale, f"trust-list.json's `pending` lists {stale}, no longer pending: remove each entry"


def test_the_trust_list_entries_are_one_line_and_scoped_to_deployed_guarantees() -> None:
    posture = _posture()
    trust = json.loads(posture.TRUST_LIST.read_text())
    deployed = {g for lock in posture.locks(ROOT).values() for g in lock.get("deployed") or []}
    assert set(trust) == {"list", "pending"}
    for p, e in trust["list"].items():
        assert set(e) <= {"reason", "only"} and isinstance(e["reason"], str) and e["reason"].strip(), p
        assert "\n" not in e["reason"], p
        assert set(e.get("only", [])) <= deployed, f"{p}'s `only` names {sorted(set(e['only']) - deployed)}, no deployed guarantee"
    for g, ps in trust["pending"].items():
        assert g in deployed and all(isinstance(w, str) and w.strip() and "\n" not in w for w in ps.values()), g
