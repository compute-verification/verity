"""`posture.py`: each deployed guarantee's premises, marked against the trust list."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import check as A  # noqa: E402

spec = importlib.util.spec_from_file_location("lean_posture", HERE / "posture.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


def rec(assumptions=(), open_=()):
    return {"premises": {"assumptions": list(assumptions), "domain": [], "open": list(open_), "reads": []}}


LOCK = {
    "assumptions": {"H.CR": {"owner": "@lean"}, "H.Prf": {"owner": "@lean"}},
    "open": {"O.Src": {"owner": "@proofs", "issue": "LEAN-1", "due": "2026-11-30"},
             "O.Gap": {"owner": "@proofs", "issue": "LEAN-2", "due": "2026-12-01"},
             "O.Fix": {"owner": "@proofs", "issue": "LEAN-3", "due": "2026-12-02"}},
    "deployed": ["G.zk", "G.ks"], "trust_gap": {"G.zk": ["O.Gap"]},
    "witnesses": {"G.zk": {"honest": ["G.zk_honest"], "nontrivial": "G.zk_nontrivial"}},
    "guarantees": {"G.zk": rec(["H.CR", "H.Prf"], ["O.Gap", "O.Src"]), "G.ks": rec(["H.CR", "H.Prf"]),
                   "G.zk_honest": rec(["H.CR"], ["O.Fix"]), "G.zk_nontrivial": rec()},
}
TRUST = {"list": {"H.CR": {"reason": "collision resistance"}, "H.Prf": {"reason": "a PRF", "only": ["G.zk"]}},
         "pending": {"G.zk": {"O.Src": "a ruling decides"}}}


def test_each_premise_is_marked_on_the_list_in_the_gap_pending_or_off_and_open_ones_carry_their_ticket():
    zk, ks = P.postures({"L/lean-audit.json": LOCK}, TRUST, A)
    assert zk["assumptions"] == {"H.CR": "on", "H.Prf": "on"} and ks["assumptions"] == {"H.CR": "on", "H.Prf": "OFF"}
    assert zk["open"] == {"O.Gap": {"mark": "gap", "owner": "@proofs", "issue": "LEAN-2", "due": "2026-12-01"},
                          "O.Src": {"mark": "pending", "owner": "@proofs", "issue": "LEAN-1", "due": "2026-11-30"}}
    assert zk["witnesses"] == ["O.Fix"] and ks["witnesses"] == []
    assert P.line(zk) == ("G.zk (L/lean-audit.json): assumptions: H.CR [on], H.Prf [on]; open: O.Gap [gap] @proofs "
                          "LEAN-2 due 2026-12-01, O.Src [pending] @proofs LEAN-1 due 2026-11-30; witnesses take: O.Fix")
    assert P.line(ks).endswith("assumptions: H.CR [on], H.Prf [OFF]; open: none; witnesses take: none")


def test_a_listed_finder_its_proof_reads_shows_as_an_open_premise_with_the_lists_ticket():
    def rec_reading(finders, cites=()):
        r = rec(["H.CR"])
        r["premises"] |= {"domain": [f"D.{c}" for c in cites], P.FINDERS: list(finders)}
        return r

    lock = {
        "assumptions": {"H.CR": {"owner": "@lean"}},
        P.FINDERS: {"F.byChoice": {"owner": "@proofs", "issue": "PRF-1", "due": "2026-11-20"},
                    "F.cited": {"owner": "@soundness", "issue": "PRF-2", "due": "2026-11-21"}},
        "domain": {"D.Sub": {"owner": "@proofs", "cites": "G.sub"}},
        "deployed": ["G.reads", "G.cites", "G.search"],
        "guarantees": {"G.reads": rec_reading(["F.byChoice", "F.search"]), "G.cites": rec_reading([], ["Sub"]),
                       "G.sub": rec_reading(["F.cited"]), "G.search": rec_reading(["F.search"])},
    }
    reads, cites, search = P.postures({"L/lean-audit.json": lock}, TRUST, A)
    assert reads["open"] == {"F.byChoice": {"mark": "finder", "owner": "@proofs", "issue": "PRF-1", "due": "2026-11-20"}}
    assert cites["open"] == {"F.cited": {"mark": "finder", "owner": "@soundness", "issue": "PRF-2", "due": "2026-11-21"}}
    assert search["open"] == {} and reads["assumptions"] == {"H.CR": "on"}
    assert P.line(reads).endswith("open: F.byChoice [finder] @proofs PRF-1 due 2026-11-20; witnesses take: none")
