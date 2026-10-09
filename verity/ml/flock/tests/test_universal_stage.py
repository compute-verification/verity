"""`verity_flock.universal_stage`, the universal unit's class statement from arrays (``pod/rec_inner.py universal --fast``):
its gate circuit is `universal.traced`'s, and the files it stages are `class_statement.stage`'s, byte for byte."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
POD = HERE.parent / "pod"
ROOT = HERE.parents[3]

#: (G, N_IN, N_OUT): odd input counts, outputs that leave the output group's last word part empty (8,5,3; 24,9,5), and
#: 87-rec-private.sh's default class; each netlist fills 3/4 of the gates, the rest padding
CLASSES = [(8, 5, 3), (16, 16, 8), (24, 9, 5)]


@pytest.mark.parametrize("cls", [(4, 3, 2), *CLASSES, (40, 7, 6)])
def test_trace_is_universal_traced(cls):
    from verity_flock import universal_stage as US

    from verity.ml.private_circuits import universal as U

    want = U.traced(*cls)
    got = US.trace(*cls).traced()
    assert (got.n_inputs, got.outputs, got.counts) == (want.n_inputs, want.outputs, want.counts)
    assert got.gates == [(k, tuple(ops)) for k, ops in want.gates]


#: `UniversalLower.uuType`'s digests (its ``#guard digestIs``): `universal.build` over the ports (op, src, out, x), one committed
#: bit each, which Lean proves computes `Desc.eval`
UU_TYPE = {
    (1, 1, 1): "7893b854a063f857204b130c83412de9bab8b4343c2a310082a6ebf92a2463e1a47c3b8ef8d89e12e23a0eed1ce55bad57b116d5408ec7ca04f33e25adb94dca",
    (2, 2, 1): "328621bfd9f046f61da710617fd3d3c05b5610b264756be3c537d31a06bc3a82f72642f3b1e07841418e4b25ec1c82b57db902d39c47f032b57de0ea9ec9f474",
    (3, 2, 2): "af5181c36f9cdf73afa01bad1b1483534c7bc447dc7b95aade613d3e9451f5411853fc2d4439ecb279b0b26d0c0eccb510116dd0eb227b6e03421ae26237679a",
    (5, 3, 4): "7a9f3ed9861b21b68c5b8965f0911c01b97427c874628e97f3ca655237f98ed4a7239c8af3917a98fd84221b0505cfc65db031474cd992864fae8c61a010ce97",
    (7, 9, 3): "91e4515de7ea597ca9b8693a3ecdd3be4a2da6a94fe1d9f1fa653bb6a450ca0abfe8eda5e612b7f2b94ed393f90d4e8c56838f130c9becea5da7d996053a7f8e",
    (16, 16, 8): "4dc00ba99ad07dd1fd8d4433b9ce86acbe43fc079535fd77bfab60cde2732c1cd52b2a033b355e8f5ff2cae416ef6baed086c72d1dbda118fd9b34591516d0d3",
}


@pytest.mark.parametrize("cls", sorted(UU_TYPE))
def test_the_staged_unit_is_the_lean_builders_type(cls):
    """`rec_inner.universal` stages the unit in argument order: its class lowering (`universal_stage.build_of` over
    `program_unit`'s rows ``prog``, ``x``), read at the row bits that hold the description and x, is a type whose digest is
    the one Lean pins for `uuType`, with exactly `and_count` ANDs. The rows hold description leaf j at bit j mod 16 of the
    ``prog`` row's word j // 16, and x's leaf j likewise in the ``x`` row (`class_statement.pack` and `universal_stage.rows_of`
    alike), and the lowering computes `universal.interpret` on them."""
    import random

    sys.path.insert(0, str(POD))
    import rec_inner as RI
    from verity_flock import circuit_types as CT
    from verity_flock import class_statement as CS
    from verity_flock import partition_units as PU
    from verity_flock import universal_stage as US

    from verity.ml.circuits import library
    from verity.ml.circuits import partition_object as PO
    from verity.ml.circuits.defs import bind
    from verity.ml.private_circuits import universal as U

    G, N_IN, N_OUT = cls
    K = U.index_bits(G, N_IN)
    unit = RI.program_unit(bind(U.UniversalUnit, G=G, N_IN=N_IN, N_OUT=N_OUT))
    units = PU.PartitionUnits(PU.one_call(unit), query=PO.template_instance_query(unit))
    [(_shape, cl)] = units.classes().items()
    act, uc = units.activation(cl.definition), cl.representative
    groups, _ = CS.argument_order(uc, CS.sources(act, uc, CS.owners_of(units, cl.definition.id)))
    low = CS.class_lowering(act, uc, "unit-class/test", groups, True, build=US.build_of(G, N_IN, N_OUT))
    C, (p_prog, p_x) = low.circuit, low.definition.port_words
    words = [C.inputs[f"in{j}"] for j in range(low.definition.w_in)]
    n_desc = 2 * G + 2 * G * K + N_OUT * K
    ins = [words[j // 16][j % 16] for j in range(n_desc)] + [words[p_prog + j // 16][j % 16] for j in range(N_IN)]
    outs = [C.outputs[f"out{j}"][0] for j in range(N_OUT)]
    t = CT.from_gf2(C, ins, outs, [(1, 2 * G), (1, 2 * G * K), (1, N_OUT * K), (1, N_IN)], [(1, N_OUT)], library.PROGRAM, {})
    assert len(t.items) == U.and_count(G, N_IN, N_OUT)
    assert t.digest() == UU_TYPE[cls]

    rng = random.Random(f"argument-order/{cls}")
    net = U.pad(U.Netlist(N_IN, tuple((rng.randrange(4), rng.randrange(N_IN + g), rng.randrange(N_IN + g)) for g in range(G)),
                          tuple(rng.randrange(N_IN + G) for _ in range(N_OUT))), G, N_IN)
    xs = [[rng.randrange(2) for _ in range(N_IN)] for _ in range(3)]
    leaves = [[b for part in U.arguments(net, x) for b in part] for x in xs]
    lane = [[lv[-1 - k] for lv in leaves] for k in uc.inputs]
    want = [[U.interpret(net, x)[j] for x in xs] for j in range(N_OUT)]
    rows, _outs, per_port = CS.pack(uc, lane, want, low.definition.w_in, low.definition.w_out, groups, low.definition.port_words, True)
    for i, lv in enumerate(leaves):
        assert [int(per_port["p0"][i, j // 16]) >> (j % 16) & 1 for j in range(n_desc)] == lv[:n_desc]
        assert [int(per_port["p1"][i, j // 16]) >> (j % 16) & 1 for j in range(N_IN)] == lv[n_desc:]
        assert not (per_port["p0"][i, -(-n_desc // 16):].any() or per_port["p1"][i, -(-N_IN // 16):].any())
    got, _cut, _ok = low.evaluate(rows.astype(np.uint64))
    assert got[:, :N_OUT].tolist() == [list(r) for r in zip(*want)]

    u = US.unit(US.trace(G, N_IN, N_OUT), program_row=True, argument_order=True)
    assert US._port_layout(u)[0] == [p_prog, p_x]
    assert np.array_equal(US.rows_of(u, net, xs, [p_prog, p_x]), rows)


_STAGE_BOTH = r"""
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import rec_inner as RI
RI._benchmark()
from verity_flock import class_statement as CS
CS.cache_slot_circuits()
res = {}
for cls, n in json.loads(sys.argv[3]):
    for fast in (False, True):
        d = Path(sys.argv[2]) / f"{cls}-{fast}"
        rec = RI.universal(cls, n, d, 0, fast=fast, salt_key=bytes(range(32)))
        files = {f"{s}/{f.name}": hashlib.sha256(f.read_bytes()).hexdigest() for s in ("stage", "other") for f in sorted((d / s).iterdir())}
        res[f"{cls}/{fast}"] = {"record": rec, "files": files}
print(json.dumps(res))
"""


@pytest.mark.slow
def test_fast_staging_writes_class_statement_stage_files(tmp_path):
    """Both statements `rec_inner.universal` stages (the registered netlist's and the other's) through today's path and through
    `universal_stage`, under one salt key: every file (``circuit.txt``, ``inst-n.bin``, ``pub-n.bin``, ``partition.json``) has
    the same SHA-256, and the records agree. One process for all classes, the slot circuits composed once (about a minute):
    the layout, the composed circuit and the salted rows are only checked whole by staging them."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(p) for p in (ROOT, ROOT / "catalog", ROOT / "kernels",
                                                                        ROOT / "verity" / "ml" / "flock" / "python")))
    todo = [[",".join(map(str, c)), 1 + c[2] % 4] for c in CLASSES]
    p = subprocess.run([sys.executable, "-c", _STAGE_BOTH, str(POD), str(tmp_path), json.dumps(todo)], env=env, capture_output=True,
                       text=True, check=False)
    assert p.returncode == 0, p.stderr[-3000:]
    res = json.loads(p.stdout)
    for cls, n in todo:
        today, fast = res[f"{cls}/False"], res[f"{cls}/True"]
        names = ("circuit.txt", f"inst-{n}.bin", "partition.json", f"pub-{n}.bin")
        assert sorted(today["files"]) == [f"{s}/{f}" for s in ("other", "stage") for f in names]
        assert fast["files"] == today["files"], cls
        assert fast["record"] == today["record"], cls
        assert today["record"]["netlist"]["padded_gates"] > 0
