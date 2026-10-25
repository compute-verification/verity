"""scan_zk's host side: the run it reads (each link-window's rows the warden's committed ones, each egress bucket
accepting the view its ingress bucket wrote, request 5 in request 1's slot past its egress) and each round's host read
(`scan_zk.member`, as Lean's registered reads make it, and the references' outputs) as the round's: honest accepted
with every output 1, the ingress units' view leaves the egress units' state leaves; chain's swapped buckets' ok bits 0
on reads that hold; forged's egress row refused ("does not open"). Nothing here lowers or proves (`scan_zk.prove` does,
on a node)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def _load():
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("network_traces_scan_zk", HERE / "scan_zk.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sz = _load()


def test_each_rounds_host_read_is_the_rounds(tmp_path):
    h = sz.host(tmp_path)
    at, T = h.at, sz.PARAMS.window
    assert at.admitted == {1: 0, 2: 2, 3: 2, 4: 12, 5: 22}
    assert [s.q for s in at.chain.scan[23].slots] == [5, 2, 3, 4]
    honest, chain, forged = (h.rounds[k] for k in ("honest", "chain", "forged"))
    ing, eg = honest
    assert ing.outputs == [1] * T and eg.outputs == [sz.EG.ACCEPT] * T
    assert ing.bcs["view"] == eg.bcs["state"] and ing.bcs["sout"][:-1] == ing.bcs["sin"][1:]
    assert ing.ports["view"]["root"] == eg.ports["state"]["root"] and ing.ports["sin"]["reads"] == list(range(T))
    assert ing.ports["sout"]["reads"] == list(range(1, T + 1)) and eg.ports["state"][sz.REG.READS_FROM] == T
    assert chain[0].refused is None and [t for t, o in enumerate(chain[0].outputs) if not o] == [sz.A, sz.A + 1, sz.A + 2]
    assert chain[0].ports["sin"]["root"] != ing.ports["sin"]["root"] and chain[1].outputs == eg.outputs
    (f,) = forged
    t = f.changed
    assert "does not open" in f.refused and f.outputs == [int(i != t) for i in range(T)] and f.rows[t] != eg.rows[t]
