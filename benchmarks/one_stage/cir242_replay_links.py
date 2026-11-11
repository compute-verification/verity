"""CIR-242 scratch: what the two-stage design (`spot.verifier.second_stage`, `Spot.SecondStage`) derives from a replay
block whose body is a sparse-MLA row's chain of scores and values units, at small statics (DR = DV = 16, BN = 16,
G = 8, T = 32 keys in two key blocks, NH = 2 heads): its links (`link_runs`) and wires (`wire_runs`), for three
layouts of the row's keys among the replay block's parameters.

* R1 ``kv``: one parameter of whole key rows, the values units reading their columns of it;
* R2 ``kv`` and ``kvc``: the keys whole for the scores units and again in column blocks for the values units;
* R3 ``kvcols``: the keys only in column blocks, the scores units a variant reading one port a column block.

``python cir242_replay_links.py [T]`` prints one JSON line a layout, at T keys (32 by default)."""
from __future__ import annotations

import json

from verity.circuits.codec import descriptor_id
from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.refs import Coll, concat
from verity.circuits.types import Array
from verity.definitions.registry import sparse_mla as M
from verity.definitions.registry.prims import BF16, F32
from verity.protocols.spot.verifier import second_stage as SS
from verity.protocols.spot.verifier.one_stage import partition as P

GLM = M.SPARSE_MLA_GLM53
DR, DV, BN, G, T, NH = 16, 16, 16, 8, 32, 2
KW, NV, NC = DR + DV, DV // G, (DR + DV) // G
SM = dict(DR=DR, DV=DV, BN=BN, DOT=GLM["DOT"], SCALE=0.0625)


def _sep_type(S):
    state = () if S.FIRST else (("m_prev", F32), ("l_prev", Array(4, F32)))
    return ((("q_nope", Array(DV, BF16)), ("q_rope", Array(DR, BF16)))
            + tuple((f"kv{c}", Array(S.NVIS, Array(G, BF16))) for c in range(NC)) + state,
            bind(M.SparseMlaHeadScores, NVIS=S.NVIS, FIRST=S.FIRST, **SM).signature()[1])


def _sep(B, S, q_nope, q_rope, *rest):
    kv, state = rest[:NC], rest[NC:]
    rows = Coll(Array(S.NVIS, Array(KW, BF16)), concat([kv[c][k].refs for k in range(S.NVIS) for c in range(NC)]))
    return B.call(bind(M.SparseMlaHeadScores, NVIS=S.NVIS, FIRST=S.FIRST, **SM), q_nope, q_rope, rows, *state)


ScoresColsSep = CompositeDefinition("Cir242ScoresColsSep", 0, ("NVIS", "FIRST"), _sep_type, _sep, register=False)


def _chain(B, layout, q_nope, q_rope, keys):
    outs = []
    for h in range(NH):
        st, o = None, [None] * NV
        for lo, hi in M.visits(T, BN):
            nvis, first = hi - lo, st is None
            if layout == "R3":
                kvs = [keys["kvcols"][c][lo:hi] for c in range(NC)]
                sc = B.call(bind(ScoresColsSep, NVIS=nvis, FIRST=first), q_nope[h], q_rope[h], *kvs,
                            *(() if first else (st[0], st[1])))
            else:
                sc = B.call(bind(M.SparseMlaHeadScores, NVIS=nvis, FIRST=first, **SM), q_nope[h], q_rope[h],
                            keys["kv"][lo:hi], *(() if first else (st[0], st[1])))
            for c in range(NV):
                if layout == "R1":
                    kvc = Coll(Array(nvis, Array(G, BF16)),
                               concat([keys["kv"][k][c * G:(c + 1) * G].refs for k in range(lo, hi)]))
                else:
                    kvc = keys["kvc" if layout == "R2" else "kvcols"][c][lo:hi]
                vu = bind(M.SparseMlaHeadValues, NVIS=nvis, FIRST=first, BN=BN, DOT=GLM["DOT"], G=G)
                o[c] = B.call(vu, sc[2], kvc, *(() if first else (sc[3], o[c])))
            st = sc
        end = bind(M.SparseMlaHeadEnd, DV=G, INV=GLM["INV"])
        outs += [B.call(end, st[1], o[c]) for c in range(NV)]
    return Coll(Array(NH, Array(DV, BF16)), concat([x.refs for x in outs]))


def replay(layout: str):
    keys = {"R1": (("kv", Array(T, Array(KW, BF16))),),
            "R2": (("kv", Array(T, Array(KW, BF16))), ("kvc", Array(NV, Array(T, Array(G, BF16))))),
            "R3": (("kvcols", Array(NC, Array(T, Array(G, BF16)))),)}[layout]

    def sig(_S):
        return (("q_nope", Array(NH, Array(DV, BF16))), ("q_rope", Array(NH, Array(DR, BF16)))) + keys, \
            Array(NH, Array(DV, BF16))

    def body(B, _S, q_nope, q_rope, *ks):
        return _chain(B, layout, q_nope, q_rope, dict(zip((k for k, _ in keys), ks)))

    return bind(CompositeDefinition(f"Cir242Replay{layout}", 0, (), sig, body, register=False))


def derive(layout: str) -> dict:
    fn = replay(layout)
    defs, key = SS.definitions(P.population_program(fn, 2)), descriptor_id(fn)
    links, wires = SS.link_runs(defs, key), SS.wire_runs(defs, key)
    short = lambda t: t.split("{")[0]
    return {"layout": layout,
            "links": None if links is None else [[short(x.template), x.port, x.value, per, [[r.unit, r.count, r.row, r.step] for r in rs]]
                                                 for x, per, rs in links],
            "wires": None if wires is None else len(wires)}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        T = int(sys.argv[1])
    for lay in ("R1", "R2", "R3"):
        print(json.dumps({"T": T, **derive(lay)}), flush=True)
