"""Pearl+'s credited block through the door: `window_cut.credited` at one size, the tile the verifier's own draw picks
(`scheme.drawn_blocks` over a one-call run), its one block cut by Python's `restrict`, evaluated on bits on the honest
tile and on that tile with its y word's sign flipped, lowered by Lean's `flock-lower-check compose` (`lean_lower`), and
staged for `flock-circuit` with every row registered (`registered_value.Registered`, `FC.write(registered=...)`), so no
row is committed afresh: ``OUT/honest`` holds the honest rows and outputs, ``OUT/dishonest`` the flipped y (registered as
the prover would) with the honest outputs (ok 1), which no proof may carry. Each holds the verifier's ``--registered``
file (`registered.reads_file`), whose roots of ``unit``, ``rows`` and ``col`` the verifier computes from its draw, and
``OUT/honest/registered-wrong.json`` names the root of another unit's ``unit`` row. `credited_door.sh` proves both and
verifies them by Lean directly and through the door (``door``: `proofs.verify`, as it is and with `lean_lower` as
`one_circuit.lower`); run it on a node."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import random
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SCHEME = "pearl-c-h100-v1-h3"
LAYOUT = "iv/hybrid/4"
#: the core's parameters the verifier's draw fixes (`scheme.CreditedBlock`), registered under their public salts
DRAWN = ("unit", "rows", "col")
RUN_DRAW = "verity/pouw/pearl-plus-run-draw/v1"


def _peak_gib(who: int = resource.RUSAGE_SELF) -> float:
    return round(resource.getrusage(who).ru_maxrss / 2**20, 3)


def _flat(x) -> list[int]:
    if isinstance(x, (bytes, bytearray)):
        return list(x)
    if isinstance(x, int):
        return [x]
    return [w for v in x for w in _flat(v)]


def run(k: int, tm: int, tn: int, n: int, unit: int = 7) -> tuple[object, list]:
    """The scheme at the size and a run of one call that is one credited block: `drawn_blocks`' scheme and passes."""
    from verity.protocols.resource_accounting.compute.pouw.prover import schemes as pouw_schemes
    from verity.protocols.resource_accounting.compute.pouw.prover.schemes.pearl_c_work import UnitRecord
    from verity.protocols.resource_accounting.compute.pouw.verifier.protocol import Matmul, Shape
    return (pouw_schemes.get(SCHEME, k_min=k, tile=(tm, tn)),
            [[UnitRecord(Matmul(unit, Shape(tm, k, n), 0), tm, k)]])


def draw(k: int, tm: int, tn: int, n: int, coin: bytes) -> tuple[object, int]:
    """The verifier's draw of one credited block on ``coin`` (`scheme.drawn_blocks`), and its index among the run's
    credited blocks."""
    from verity.commitments.randomness import derive
    from verity.protocols.resource_accounting.compute.pouw import scheme as PS
    scheme, passes = run(k, tm, tn, n)
    (block,) = PS.drawn_blocks(scheme, passes, derive(coin, RUN_DRAW, {"run": bytes(64)}), 1)
    assert block.size == (k, tm, tn, n), block
    return block, PS.credited_blocks(scheme, passes).index(block)


def tile(k: int, tm: int, tn: int, n: int, seed: str = "credited", block=None) -> tuple[list, tuple]:
    """An honest -h3 tile on h100 (`test_window_cut_credited._tile`'s) at ``block``'s unit, rows and col (unit 7, rows
    and cols from 0 without one): the core's arguments and `reference`'s."""
    from verity.protocols.resource_accounting.compute.pouw.prover.circuit import inert_bound, pc8
    from verity.protocols.resource_accounting.compute.pouw.prover.schemes import pearl_c as C
    from verity.protocols.resource_accounting.compute.pouw.prover.schemes import pearl_c_u as CU
    from verity.protocols.resource_accounting.compute.pouw.prover.schemes.pearl_c_device import DEVICES
    from verity_vllm.program.predict import window_cut as WK
    core = WK.credited_product(SCHEME, k, tm, tn, n)
    rho, rhor = core.bindings["RHO"], core.bindings["RHOR"]
    rng, dev = random.Random(seed), "h100"

    def f32() -> list[int]:
        return [int(np.float32(rng.uniform(-2, 2)).view(np.uint32)) for _ in range(k)]

    a, b = [f32() for _ in range(tm)], [f32() for _ in range(tn)]
    salt, root = rng.randbytes(32), rng.randbytes(64)
    unit, rows, cols, col = ((7, list(range(tm)), list(range(tn)), 0) if block is None
                             else (block.unit, list(block.rows), list(block.cols), block.col))
    scheme = C.PearlC(k_min=k, tile=(tm, tn), forming="v1", device=DEVICES[dev], hashing="h3")
    ctx = C.Context(salt, C.Matmul(unit, C.Shape(tm, k, tn), 0), bytes(64), root)
    ops = scheme.encode(ctx, scheme.noise(ctx), [C.f32_row(x) for x in a], [C.f32_row(x) for x in b],
                        C.Tile(tuple(rows), tuple(cols)))
    checked, seed_b = scheme.checked(ctx, ops), scheme.noise(ctx).seed_b
    shape = pc8.Shape(tm, k, tn, tile=(tm, tn), dev=dev)
    _d, ng = inert_bound._promoted(dev, k)
    cert = [[[list(inert_bound.certificates(ops.a.codes[i], ops.b.codes[j], dev))[2 * z:2 * z + 2] for z in range(ng - 1)]
             for j in range(tn)] for i in range(tm)]
    y = [[CU.y(checked.u[i][j], ops.a.alpha[i], ops.b.alpha[j]) << 16 for j in range(tn)] for i in range(tm)]
    args = [a, b, y, cert, salt, root, unit, rows, col, [list(ops.e_b[j]) for j in range(tn)],
            [pc8.basis_fa(shape, seed_b)[t * 32:t * 32 + 32] for t in range(k)],
            [pc8.basis_fb(shape, seed_b)[t * 32:t * 32 + 32] for t in range(k)],
            [pc8.gram_g(shape, seed_b)[p * 32:p * 32 + 32] for p in range(32)]]
    return args, (a, b, y, cert, salt, root, unit, rows, cols, rho, rhor)


def flipped(args: list, ref: tuple, bit: int = 31) -> tuple[list, tuple]:
    """``args`` with y's first word's bit ``bit`` flipped: one y word changed."""
    y = [list(r) for r in args[2]]
    y[0][0] ^= 1 << bit
    return [*args[:2], y, *args[3:]], (*ref[:2], y, *ref[3:])


def value_bits(t, value) -> list[int]:
    """A value of type ``t``'s bits, low bit of each element first."""
    from verity.circuits import boolean as BL
    words, widths = _flat(value), BL.elements(t)
    assert len(words) == len(widths), (len(words), len(widths))
    return [(w >> j) & 1 for w, wd in zip(words, widths) for j in range(wd)]


def port_bits(core, args: list) -> list[tuple[str, list[int]]]:
    """Each core parameter's bits, low bit of each element first: the block's input bits, ports concatenated."""
    return [(p, value_bits(t, a)) for a, (p, t) in zip(args, core.signature()[0])]


def drawn_bits(core, block) -> dict[str, list[int]]:
    """The bits of the parameters the draw fixes, from the drawn block alone."""
    types = dict(core.signature()[0])
    return {"unit": value_bits(types["unit"], block.unit), "rows": value_bits(types["rows"], list(block.rows)),
            "col": value_bits(types["col"], block.col)}


def words16(bits: list[int]) -> list[int]:
    bits = bits + [0] * (-len(bits) % 16)
    return [sum(bits[16 * w + j] << j for j in range(16)) for w in range(len(bits) // 16)]


def reference(k: int, n: int, ref: tuple) -> list[int]:
    from verity.protocols.resource_accounting.compute.pouw.prover.circuit import pearl_plus_tile as T
    a, b, y, cert, salt, root, unit, rows, cols, rho, rhor = ref
    digest, ok = T.reference("h100", k, n, a, b, y, cert, salt, root, unit, rows, cols, 0, rho, rhor)
    return [*(digest or bytes(64)), ok]


def register(program, bits: dict[str, list[int]], supplied: tuple[str, ...] = DRAWN) -> dict:
    """Each value's one registered row (`Registered.commit`): its bits as `registered.width` leaves in the circuit's
    ``sha512/row/v2`` port row; the ``supplied`` values under their public salts (`supplied_salts`), so their roots are
    a function of the value the verifier computes itself."""
    from verity.circuits.codec import program_digest
    from verity.commitments import registered_value as RV
    from verity.protocols.proofs.spot.verifier.one_stage import registered as REG
    prog = program_digest(program)
    out = {}
    for p, v in bits.items():
        assert REG.width(program, p) == 16, p
        out[p] = RV.Registered.commit(prog, p, [words16(v)], RV.supplied_salts(prog, p, 1) if p in supplied else None,
                                      word_bits=16, schema=RV.SCHEMA_HM96_SHA512_ROW_V2)
    return out


def reads(program, partition, bits: dict[str, list[int]], roots: dict[str, str]) -> dict:
    """The verifier's ``--registered`` file (`registered.reads_file`) over its program and the partition's query."""
    from verity.commitments import registered_value as RV
    from verity.protocols.proofs.spot.verifier.one_stage import registered as REG
    return REG.reads_file(program, partition["query"], {p: len(v) // 16 for p, v in bits.items()}, roots,
                          schemas={p: RV.SCHEMA_HM96_SHA512_ROW_V2 for p in bits})


def lean_lower(lower_bin: str, partition, fn) -> "callable":
    """`one_circuit.lower` as Lean's `flock-lower-check compose` gives it, for a partition object of one template
    (`Q_template_instance`) and ``fn``, that template's Definition as the verifier builds it (a decoded one is no
    `Program`'s root): C_S's template instance lowered to the templated block circuit, its input ports the template's
    parameters as bit rows, its output the template's return bits as one u16 port ``o0``. The function the door's
    `one_circuit.lower` would need, taking the partition and the template beside C_S."""
    from verity.circuits import boolean as BL
    from verity.circuits import partition_object as PO
    from verity.circuits.codec import descriptor_id, program_digest
    from verity.circuits.program import Program

    key = partition["query"]["params"]["template"]
    if descriptor_id(fn) != key:
        raise ValueError(f"the template {descriptor_id(fn)} is not the partition's {key}")

    def lower(c_s: bytes) -> bytes:
        params, ret = fn.signature()
        ins = ",".join(f"{p}:{sum(BL.elements(t))}" for p, t in params)
        outs = f"o0:{-(-sum(BL.elements(ret)) // 16)}"
        with tempfile.TemporaryDirectory(prefix="credited-lower-") as d:
            src, dst = Path(d, "c_s.json"), Path(d, "circuit.txt")
            src.write_bytes(c_s)
            p = subprocess.run([lower_bin, "compose", str(src), key, f"unit/{fn.id}", ins, outs, LAYOUT,
                                partition["program"], PO.digest(partition), json.dumps({key: program_digest(Program(fn))}),
                                str(dst)], capture_output=True, text=True)
            if p.returncode != 0:
                raise RuntimeError(f"flock-lower-check compose ({p.returncode}): {(p.stderr or p.stdout)[-2000:]}")
            return dst.read_bytes()
    return lower


def stage(a) -> int:
    k, tm, tn, n = map(int, a.size.split(","))
    from verity.circuits import partition_object as PO
    from verity.circuits import restrict as R
    from verity.circuits.codec import descriptor_chunks, descriptor_id
    from verity.circuits.evaluation import evaluate_bits
    from verity.protocols.proofs.spot.verifier.one_stage import one_circuit as OC
    from verity_flock import circuit as FC
    from verity_vllm.program.predict import window_cut as WK

    rec: dict = {"size": [k, tm, tn, n], "scheme": SCHEME, "phases": {}}
    t = time.perf_counter()

    def phase(name: str, **extra) -> None:
        nonlocal t
        now = time.perf_counter()
        rec["phases"][name] = {"wall_s": round(now - t, 2), "peak_gib": _peak_gib(), **extra}
        print(f"PHASE {name} {json.dumps(rec['phases'][name])}", flush=True)
        t = now

    coin = os.urandom(64)
    block, index = draw(k, tm, tn, n, coin)
    rec["draw"] = {"coin": coin.hex(), "block": {"index": index, "unit": block.unit, "rows": list(block.rows),
                                                 "col": block.col, "size": list(block.size), "real": block.real}}
    print("DRAW " + json.dumps(rec["draw"]), flush=True)
    program, part = WK.credited(SCHEME, k, tm, tn, n)
    c = b"".join(descriptor_chunks(program))
    core = WK.credited_product(SCHEME, k, tm, tn, n)
    key = descriptor_id(core)
    door_draw = {"units": [index]}
    units = OC.block(c, part, index)
    phase("program", descriptor_bytes=len(c), block_units=units)
    b = R.restrict(c, part, units)
    phase("restrict", block_bytes=len(b.descriptor), bit_gates=int(b.bits.size))

    honest, href = tile(k, tm, tn, n, block=block)
    dishonest, dref = flipped(honest, href)
    want = [reference(k, n, href), reference(k, n, dref)]
    hb, db = dict(port_bits(core, honest)), dict(port_bits(core, dishonest))
    mine = drawn_bits(core, block)
    assert all(hb[p] == mine[p] for p in DRAWN)

    def words(g: list[int]) -> list[int]:
        return [sum(g[8 * i + j] << j for j in range(8)) for i in range(64)] + [g[512]]

    got = [evaluate_bits(core, *(np.asarray([v], dtype=np.uint8) for v in pb.values())).reshape(-1).tolist()
           for pb in (hb, db)]
    as_words = [words(g) for g in got]
    rec["bits"] = {"of": "the core (the lowered unit) on the staged bits", "honest_ok": as_words[0][-1],
                   "dishonest_ok": as_words[1][-1], "honest_digest": bytes(as_words[0][:64]).hex(),
                   "reference_agrees": as_words == want}
    phase("bits")
    print("BITS " + json.dumps(rec["bits"]), flush=True)
    assert as_words == want, ("the core on bits is not the scheme's reference", as_words, want)
    assert as_words[0][-1] == 1 and as_words[1][-1] == 0

    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    text = lean_lower(a.lower, part, core)(b.descriptor).decode()
    rec["lower"] = {"by": "lean flock-lower-check compose (lean_lower)", "wall_s": round(time.perf_counter() - t0, 2),
                    "peak_gib_children": _peak_gib(resource.RUSAGE_CHILDREN)}
    meta = FC.meta_of(text)
    rec["circuit"] = {"bytes": len(text), "head": text.split("\n", 1)[0], "sha512": hashlib.sha512(text.encode()).hexdigest(),
                      "ports": [q[0] for q in meta.get("ports", [])]}
    print("LOWER " + json.dumps(rec["lower"] | {"circuit": rec["circuit"]}), flush=True)
    phase("lower")

    outs_h = {"o0": np.asarray([words16(got[0])], dtype=np.uint16)}
    verifier_roots = {p: r.root.hex() for p, r in register(program, mine).items()}
    wrong = {"unit": drawn_bits(core, SimpleNamespace(unit=block.unit + 1, rows=block.rows, col=block.col))["unit"]}
    rec["registered"] = {}
    for tag, bits in (("honest", hb), ("dishonest", db)):
        d = a.out / tag
        (d / "tables").mkdir(parents=True, exist_ok=True)
        (d / "circuit.txt").write_text(text)
        regs = register(program, bits)
        assert all(regs[p].root.hex() == verifier_roots[p] for p in DRAWN)
        roots = {p: r.root.hex() for p, r in regs.items()} | verifier_roots
        (d / "registered.json").write_text(json.dumps(reads(program, part, bits, roots)))
        if tag == "honest":
            bad = roots | {p: r.root.hex() for p, r in register(program, wrong).items()}
            (d / "registered-wrong.json").write_text(json.dumps(reads(program, part, bits, bad)))
        tables = {p: np.asarray([words16(v)], dtype=np.uint16) for p, v in bits.items()}
        salts = {p: [r.salts[0]] for p, r in regs.items()} | {"o0": [os.urandom(192)]}
        iset = SimpleNamespace(name=f"credited-{tag}", subcircuit=SimpleNamespace(id=core.id, inputs=[], outputs=[]))
        FC.write(text, iset, 0, 1, d / "inst-1.bin", d / "pub-1.bin", indices=[units[0][0]], tables=tables,
                 refs={p: [0] for p in bits}, share_rows=True, salts=salts,
                 registered={p: r.statement([0])[2] for p, r in regs.items()}, outs=outs_h)
        (d / "unit.tmpl").write_text(FC.template_of(text))
        (d / "partition.json").write_text(json.dumps(part))
        (d / "program.json").write_bytes(c)
        (d / "draw.json").write_text(json.dumps(door_draw))
        rec["registered"][tag] = {p: r.root.hex()[:16] for p, r in regs.items()}
    phase("stage")
    rec["peak_gib"] = _peak_gib()
    (a.out / "stage.json").write_text(json.dumps(rec, indent=1, default=str))
    print("STAGE " + json.dumps({x: rec[x] for x in ("size", "draw", "circuit", "bits", "registered")}, default=str),
          flush=True)
    return 0


def door(a) -> int:
    """`proofs.verify` on each staged proof, with the verifier's own program, partition, draw and registered file: as it
    is, then with `lean_lower` as `one_circuit.lower`. The Lean executable is the one the job built from the same
    source (`proofs.executable` builds it in the tree), and `one_circuit.restrict`, a function of its arguments, is
    computed once for the calls."""
    from verity.protocols.proofs.spot.verifier.one_stage import one_circuit as OC
    from verity.protocols.proofs.spot.verifier.one_stage import proofs as PF
    os.environ[PF.TMPL_SWITCH] = "1"
    PF.executable = lambda: Path(a.lean)
    restrict, memo = OC.restrict, {}

    def once(c: bytes, partition, units) -> bytes:
        key = (hashlib.sha512(c).hexdigest(), json.dumps(partition, sort_keys=True), json.dumps([list(u) for u in units]))
        if key not in memo:
            memo[key] = restrict(c, partition, units)
        return memo[key]

    OC.restrict = once
    h = a.out / "honest"
    program, partition = (h / "program.json").read_bytes(), json.loads((h / "partition.json").read_text())
    draw_ = json.loads((h / "draw.json").read_text())

    def call(label: str, tag: str, reads_name: str) -> None:
        d = a.out / tag
        t0 = time.perf_counter()
        found, refusal = PF.verify(PF.Proof(d / "circuit.txt", d / "pub-1.bin", tuple(map(Path, a.session[tag]))),
                                   program=program, partition=partition, draw=draw_,
                                   registered=json.loads((d / reads_name).read_text()))
        print("DOOR " + json.dumps({"call": label, "proof": tag, "registered": reads_name,
                                    "wall_s": round(time.perf_counter() - t0, 2), "peak_gib": _peak_gib(),
                                    "refusal": refusal, "findings": [dataclasses.asdict(f) for f in found or []]},
                                   default=str), flush=True)

    call("as-is", "honest", "registered.json")
    from verity_vllm.program.predict import window_cut as WK
    OC.lower = lean_lower(a.lower, partition, WK.credited_product(SCHEME, *map(int, a.size.split(","))))
    call("lean_lower", "honest", "registered.json")
    call("lean_lower", "dishonest", "registered.json")
    call("lean_lower", "honest", "registered-wrong.json")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    st = sub.add_parser("stage")
    st.add_argument("--size", default="256,1,1,1", help="K,TM,TN,N")
    st.add_argument("--lower", required=True, help="flock-lower-check, Lean's lowering")
    st.add_argument("--out", type=Path, required=True)
    dr = sub.add_parser("door")
    dr.add_argument("--size", default="256,1,1,1", help="K,TM,TN,N")
    dr.add_argument("--lower", required=True)
    dr.add_argument("--lean", required=True, help="verity-verify-proof")
    dr.add_argument("--out", type=Path, required=True)
    dr.add_argument("--session", nargs=2, action="append", metavar=("PROOF", "DIR"), required=True)
    a = ap.parse_args()
    if a.cmd == "door":
        a.session = {t: [s for u, s in a.session if u == t] for t, _ in a.session}
        return door(a)
    return stage(a)


if __name__ == "__main__":
    sys.exit(main())
