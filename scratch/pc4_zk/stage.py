"""stage.py ROLE K N OUT: one Pearl-C4 row unit class (`Pc4RowA_v1{K}` for row_a, `Pc4RowB_v1{K}` for row_b) over N rows,
staged for C-Flock with its anchors as public input regions (row_a: E_A, F_A, F_B; row_b: E_B, F_B, F_A, G).

row_a's units come from the call Program at N rows of A and one of B, row_b's from the weight Program at N rows of B, each
partitioned by its own Q_template_instances query (`pc4.query`). The class unit is class_statement's (`sources`, `pack`, the
activation's own evaluation as the reference), except its leaf order: the public ports first, unpadded and each in its
parameter's leaf order (one u16 per E2M1 code, two per FP32 word, as `TileAnchors.regions` lays them out), then zero leaves
up to an aligned power-of-two span of 128-bit words (flock-public-ports' `public_span` rule), then the x row, padded to whole
64-word SHA-512 blocks as class_statement pads a row. The anchors are the verifier's own (`TileAnchors`); x is a seeded
N(0, 1) row in BF16. Writes circuit.txt, inst-N.bin, pub-N.bin, public-inputs-N.bin (`TileAnchors.public_inputs`), the C-Flock
writer's flock-public-inputs.bin, partition.json (the verifier's own verity/partition/v1 object), pin.txt and stage.json
under OUT."""
from __future__ import annotations

import hashlib
import json
import resource
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from verity.ir import partition_object as PO
from verity.ir.program import Program
from verity_flock import circuit as FC
from verity_flock import class_statement as CS
from verity_flock import gf2
from verity_flock import ir_lower as IL
from verity_flock import partition_units as PU
from verity_pouw.circuit import pc4
from verity_pouw.circuit.anchors import TileAnchors, public_ports
from verity_pouw.protocol import Context, Matmul, Shape
from verity_pouw.schemes import pearl_c4 as F

ROLE, K, N, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
SIDE = {"row_a": ("call", "Pc4RowA"), "row_b": ("weight", "Pc4RowB")}
FLIP = {"row_a": ("fa", 8), "row_b": ("g", 0)}


def part(name: str) -> bytes:
    return hashlib.sha256(f"verity/pouw/pc4rowa-public-inputs/v1\0{name}".encode()).digest()


def lowering(act, uc, name, groups, names, public):
    W = CS._words
    gw = [sum(W(uc.in_widths[t]) for t in g) for g in groups]
    port_words = tuple(w if names[i] in public else CS._pad(w, CS.BLOCK_WORDS) for i, w in enumerate(gw))
    off = [sum(port_words[:i]) for i in range(len(port_words))]
    src = []
    for i in range(len(groups)):
        if names[i] in public:
            src += [("leaf", off[i] + j) for j in range(port_words[i])]
    words128 = -(-len(src) // 8)
    span = 8 * (1 << (words128 - 1).bit_length())
    src += [("zero", 0)] * (span - len(src))
    for i in range(len(groups)):
        if names[i] not in public:
            src += [("leaf", off[i] + j) for j in range(port_words[i])]
    pos = {p: j for j, (k, p) in enumerate(src) if k == "leaf"}
    C = gf2.Circuit()
    leaves = [C.input(f"in{j}", 16) for j in range(len(src))]
    wires = {}
    for i, g in enumerate(groups):
        at = off[i]
        for t in g:
            w = uc.in_widths[t]
            wires[act.absolute(uc.inputs[t])] = [b for q in range(at, at + W(w)) for b in leaves[pos[q]]][:w]
            at += W(w)
    IL.lower_gates(C, [act.gate(j) for j in uc.gates], wires)
    out_bits = []
    for j, w in zip(uc.outputs, uc.out_widths):
        out_bits += list(wires[act.absolute(j)]) + [gf2.ZERO] * (16 * W(w) - w)
    w_out = max(1, len(out_bits) // 16)
    out_bits += [gf2.ZERO] * (16 * w_out - len(out_bits))
    for j in range(w_out):
        C.output(f"out{j}", out_bits[16 * j:16 * (j + 1)])
    units = IL.Units(target=None, in_widths=[16] * len(src), out_widths=[16] * w_out, in_src=[src],
                     out_src=[[("ret", p) for p in range(w_out)]], signature=(uc.shape,), gates=[], arg_gates=[], ret_gates=[])
    lay = IL.layout(C, [f"in{j}" for j in range(len(src))], [f"out{j}" for j in range(w_out)])
    return IL.Lowering(name, CS.ClassDefinition(name, port_words, w_out), units, C, lay), port_words, span


def bf16(v: np.ndarray) -> np.ndarray:
    u = v.astype(np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def gb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)


def main() -> None:
    t0 = time.time()
    side, prefix = SIDE[ROLE]
    shape = pc4.Shape(N, K, 1, tile=(1, 1)) if ROLE == "row_a" else pc4.Shape(1, K, N, tile=(1, 1))
    name = f"pc4-zk-{ROLE}"
    PU.PARTITIONS[name] = lambda arg: {"query": pc4.query(shape, side)}
    ctx = Context(part("salt"), Matmul(3, Shape(shape.m, shape.k, shape.n), 9), part("root_a"), part("root_b"))
    ta = TileAnchors.of(F.PearlC4("nvfp4"), ctx, shape)
    ports = public_ports(shape, ROLE)
    units = PU.partition_units(Program(pc4.call(shape) if side == "call" else pc4.weight(shape)), name)
    t_units = time.time() - t0
    classes = units.classes()
    (sh, cl), = [(s, c) for s, c in classes.items() if str(c.representative.definition).startswith(prefix)]
    act, uc = units.activation(cl.definition), cl.representative
    params = act.fn.signature()[0]
    leaf_param, start = [], {}
    for i, (nm, t) in enumerate(params):
        start[nm] = len(leaf_param)
        leaf_param += [i] * t.leaves
    # a public port's words are its parameter's leaves in order (the verifier's derivation), not the unit's first-read order
    groups = [sorted(g, key=lambda t: -1 - uc.inputs[t]) for g in CS.sources(act, uc, CS.owners_of(units, cl.definition.id))]
    names = []
    for g in groups:
        ks = {leaf_param[-1 - uc.inputs[t]] for t in g if uc.inputs[t] < 0}
        assert len(ks) == 1 and all(uc.inputs[t] < 0 for t in g), "each group one parameter's leaves"
        names.append(params[ks.pop()][0])
    low, port_words, span = lowering(act, uc, f"unit-class/{sh[:16]}", groups, names, set(ports))
    t_lower = time.time() - t0
    first = ta.regions(ROLE, 0)
    anchor = shape.ports[ROLE]["regions"]
    per_row = {p for p, a in anchor.items() if shape.derivations[a].per_row}
    regs = [{p: ta._region(ROLE, p, i) if p in per_row else first[p] for p in first} for i in range(N)]
    # the unit's leaves are the anchors' values (G's FP32 words), not their u16 region words
    shared = {p: ta._derive(anchor[p], ()) for p in first if p not in per_row}
    leaf_vals = [{p: ta._derive(anchor[p], (ta._ids(anchor[p])[i],)) if p in per_row else shared[p] for p in first}
                 for i in range(N)]
    rng = np.random.default_rng(20261002)
    widths = [act.width(-1 - p) for p in range(act.params)]
    ins, want = [[] for _ in uc.inputs], [[] for _ in uc.outputs]
    for i in range(N):
        vals = {"x": [int(w) for w in bf16(rng.standard_normal(K))], **leaf_vals[i]}
        words = [0] * act.params
        for nm, _t in params:
            for j, v in enumerate(vals[nm]):
                words[start[nm] + j] = v
        assert all(0 <= w < 1 << b for w, b in zip(words, widths))
        run = CS._evaluation(act, ("pc4-zk", ROLE, i), words)
        for port, x in zip(ins, uc.inputs):
            port.append(run[x])
        for port, j in zip(want, uc.outputs):
            port.append(run[j])
    rows, outs, per_port = CS.pack(uc, ins, want, low.definition.w_in, low.definition.w_out, groups, port_words)
    per_port = {names[int(k[1:])]: v for k, v in per_port.items()}
    for nm in ports:
        assert all(list(per_port[nm][i]) == regs[i][nm] for i in range(N)), f"{nm}: not the verifier's words"
    got, _cut, ok = low.evaluate(rows.astype(np.uint64))
    assert int(ok) == (1 << N) - 1 and np.array_equal(got.astype(np.uint16), outs), "the laid-out unit disagrees with the reference"
    t_eval = time.time() - t0
    partition, program = CS.partition_binding(units, name)
    CS.cache_slot_circuits()
    port = lambda nm: SimpleNamespace(name=nm, dtype="u16")  # noqa: E731
    cname = sh[:16]
    content = uc.form.encode()
    sub = SimpleNamespace(id=f"unit-class/{cname}", template=SimpleNamespace(id="unit-class"), inputs=[port(n) for n in names],
                          outputs=[port("out")],
                          program_digests=lambda statics: {f"unit-class/{cname}": hashlib.sha256(content).hexdigest()})
    iset = SimpleNamespace(subcircuit=sub, name=f"unit-class-{cname}", content_digest=hashlib.sha256(content).hexdigest())
    text = FC.compose(low, sub, partition=partition, program=program, public_inputs=ports)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "circuit.txt").write_text(text)
    meta = json.loads(text.split("\n", 2)[1][5:])
    obj = PO.build(units.program, units.query)
    assert PO.digest(obj) == meta["partition"]["digest"], "the partition object is not the one META binds"
    (OUT / "partition.json").write_text(json.dumps(obj, sort_keys=True))
    salt_key = hashlib.sha256(b"verity/flock/class-statement/stress-salts\0" + b"0").digest()
    rows_in = {nm: per_port[nm] for nm in names if nm not in ports}
    FC.write(text, iset, 0, N, OUT / f"inst-{N}.bin", OUT / f"pub-{N}.bin", salt_key, rows=rows_in, outs={"out": outs},
             public_words={nm: per_port[nm] for nm in ports})
    own = ta.public_inputs(ROLE, range(N))
    ta.check_public_inputs(ROLE, range(N), own)
    (OUT / f"public-inputs-{N}.bin").write_bytes(own)
    # one anchor bit flipped: bit 0 of word FLIP[1] of region FLIP[0], instance 0
    head = json.loads(own.split(b"\n", 1)[0])
    flip = FLIP[ROLE]
    at = len(own.split(b"\n", 1)[0]) + 1 + 2 * (sum(p["words"] for p in head["ports"][:[p["port"] for p in head["ports"]].index(flip[0])]) + flip[1])
    flipped = bytearray(own)
    flipped[at] ^= 1
    (OUT / "public-inputs-flipped.bin").write_bytes(bytes(flipped))
    try:
        ta.check_public_inputs(ROLE, range(N), bytes(flipped))
        python_refusal = None
    except Exception as e:  # noqa: BLE001
        python_refusal = str(e)
    assert python_refusal, "check_public_inputs accepted a flipped anchor bit"
    FC.write_public_inputs(text, {nm: per_port[nm] for nm in ports}, N, OUT / "flock-public-inputs.bin")
    pub = (OUT / f"pub-{N}.bin").read_bytes()
    body = own.split(b"\n", 1)[1]
    rec = {"role": ROLE, "k": K, "definition": str(uc.definition), "shape": sh, "ands": low.layout.counts["ands"], "n": N,
           "k_log": meta["k_log"], "vus_per_block": meta["vus_per_block"], "ports": [[p[0], p[1]] for p in meta["ports"]],
           "input_ports": meta["input_ports"], "public_inputs": meta["public_inputs"], "port_words": dict(zip(names, port_words)),
           "unit_leaves": len(low.units.in_src[0]), "public_span_leaves": span, "circuit_bytes": len(text),
           "circuit_sha512": FC.digest(text), "class_sha512": FC.class_digest(text), "partition_digest": meta["partition"]["digest"],
           "own_equals_c_flock_writer": own == (OUT / "flock-public-inputs.bin").read_bytes(),
           "pub_ends_with_own_body": pub.endswith(body), "own_sha256": hashlib.sha256(own).hexdigest(), "own_bytes": len(own),
           "seed_a": ta.seed_a.hex(), "seed_b": ta.seed_b.hex(),
           "flipped": {"port": flip[0], "word": flip[1], "instance": 0, "bit": 0, "python_refusal": python_refusal},
           "seconds": {"units": round(t_units, 1), "lower": round(t_lower, 1), "evaluate": round(t_eval, 1),
                       "total": round(time.time() - t0, 1)},
           "maxrss_gb": gb()}
    (OUT / "stage.json").write_text(json.dumps(rec, indent=1))
    (OUT / "pin.txt").write_text(FC.digest(text))
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
