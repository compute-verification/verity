"""Pearl+'s credited block through the door's lowering: `window_cut.credited` at one size, its one block cut by Python's
`restrict`, evaluated on bits on an honest test-instance tile and on that tile with its y word's sign flipped, lowered by
Lean's `flock-lower-check compose` (or, with ``--python-lower`` or when Lean refuses, `one_circuit.lower`), and staged for
`flock-circuit`: ``OUT/honest`` holds the honest rows and outputs, ``OUT/dishonest`` the flipped rows with the honest
outputs (ok 1), which no proof may carry. `credited_door.sh` proves and verifies both; run it on a node."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import resource
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SCHEME = "pearl-c-h100-v1-h3"
LAYOUT = "iv/hybrid/4"


def _peak_gib(who: int = resource.RUSAGE_SELF) -> float:
    return round(resource.getrusage(who).ru_maxrss / 2**20, 3)


def _flat(x) -> list[int]:
    if isinstance(x, (bytes, bytearray)):
        return list(x)
    if isinstance(x, int):
        return [x]
    return [w for v in x for w in _flat(v)]


def tile(k: int, tm: int, tn: int, n: int, seed: str = "credited") -> tuple[list, tuple]:
    """An honest -h3 tile on h100 (`test_window_cut_credited._tile`'s): the core's arguments and `reference`'s."""
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
    salt, root, unit, rows, cols = rng.randbytes(32), rng.randbytes(64), 7, list(range(tm)), list(range(tn))
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
    args = [a, b, y, cert, salt, root, unit, rows, 0, [list(ops.e_b[j]) for j in range(tn)],
            [pc8.basis_fa(shape, seed_b)[t * 32:t * 32 + 32] for t in range(k)],
            [pc8.basis_fb(shape, seed_b)[t * 32:t * 32 + 32] for t in range(k)],
            [pc8.gram_g(shape, seed_b)[p * 32:p * 32 + 32] for p in range(32)]]
    return args, (a, b, y, cert, salt, root, unit, rows, cols, rho, rhor)


def flipped(args: list, ref: tuple, bit: int = 31) -> tuple[list, tuple]:
    """``args`` with y's first word's bit ``bit`` flipped: one y word changed."""
    y = [list(r) for r in args[2]]
    y[0][0] ^= 1 << bit
    return [*args[:2], y, *args[3:]], (*ref[:2], y, *ref[3:])


def port_bits(core, args: list) -> list[tuple[str, list[int]]]:
    """Each core parameter's bits, low bit of each element first: the block's input bits, ports concatenated."""
    from verity.circuits import boolean as BL
    out = []
    for a, (p, t) in zip(args, core.signature()[0]):
        words, widths = _flat(a), BL.elements(t)
        assert len(words) == len(widths), (p, len(words), len(widths))
        out.append((p, [(w >> j) & 1 for w, wd in zip(words, widths) for j in range(wd)]))
    return out


def words16(bits: list[int]) -> list[int]:
    bits = bits + [0] * (-len(bits) % 16)
    return [sum(bits[16 * w + j] << j for j in range(16)) for w in range(len(bits) // 16)]


def reference(k: int, n: int, ref: tuple) -> list[int]:
    from verity.protocols.resource_accounting.compute.pouw.prover.circuit import pearl_plus_tile as T
    a, b, y, cert, salt, root, unit, rows, cols, rho, rhor = ref
    digest, ok = T.reference("h100", k, n, a, b, y, cert, salt, root, unit, rows, cols, 0, rho, rhor)
    return [*(digest or bytes(64)), ok]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="256,1,1,1", help="K,TM,TN,N")
    ap.add_argument("--lower", required=True, help="flock-lower-check, Lean's lowering")
    ap.add_argument("--python-lower", action="store_true", help="lower by one_circuit.lower, not Lean's compose")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    k, tm, tn, n = map(int, a.size.split(","))

    from verity.circuits import partition_object as PO
    from verity.circuits import restrict as R
    from verity.circuits.codec import descriptor_chunks, descriptor_id, program_digest
    from verity.circuits.evaluation import evaluate_bits
    from verity.circuits.ops.boolean.versions import block_circuit, boolean_table
    from verity.circuits.program import Program
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

    program, part = WK.credited(SCHEME, k, tm, tn, n)
    c = b"".join(descriptor_chunks(program))
    core = WK.credited_product(SCHEME, k, tm, tn, n)
    key = descriptor_id(core)
    phase("program", descriptor_bytes=len(c))
    b = R.restrict(c, part, [(0, 1)])
    phase("restrict", block_bytes=len(b.descriptor), bit_gates=int(b.bits.size))

    honest, href = tile(k, tm, tn, n)
    dishonest, dref = flipped(honest, href)
    want = [reference(k, n, href), reference(k, n, dref)]
    hb, db = port_bits(core, honest), port_bits(core, dishonest)

    def words(g: list[int]) -> list[int]:
        return [sum(g[8 * i + j] << j for j in range(8)) for i in range(64)] + [g[512]]

    got = [evaluate_bits(core, *(np.asarray([v], dtype=np.uint8) for _p, v in pb)).reshape(-1).tolist() for pb in (hb, db)]
    as_words = [words(g) for g in got]
    rec["bits"] = {"of": "the core (the lowered unit) on the staged bits", "honest_ok": as_words[0][-1],
                   "dishonest_ok": as_words[1][-1], "honest_digest": bytes(as_words[0][:64]).hex(),
                   "reference_digest": bytes(want[0][:64]).hex(), "reference_agrees": as_words == want}
    phase("bits")
    print("BITS " + json.dumps(rec["bits"]), flush=True)
    fn = block_circuit(b.program.fn, boolean_table(b.program))
    root = evaluate_bits(fn, np.asarray([sum((v for _p, v in hb), []), sum((v for _p, v in db), [])], dtype=np.uint8))
    root = [words(g) for g in root.reshape(2, -1).tolist()]
    rec["block_root"] = {"of": "restrict's block root on the same bits, one port", "honest_ok": root[0][-1],
                         "dishonest_ok": root[1][-1], "honest_digest": bytes(root[0][:64]).hex(), "agrees": root == as_words}
    phase("block-root-bits")
    print("BLOCK-ROOT " + json.dumps(rec["block_root"]), flush=True)
    assert as_words == want, ("the core on bits is not the scheme's reference", as_words, want)
    assert as_words[0][-1] == 1 and as_words[1][-1] == 0

    part3 = {x: part[x] for x in ("format", "program", "query")}
    ins = ",".join(f"{p}:{len(v)}" for p, v in hb)
    outs_h = {"o0": np.asarray([words16(got[0])], dtype=np.uint16)}
    rec |= {"key": key, "ins_bits": sum(len(v) for _p, v in hb), "outs": f"o0:{outs_h['o0'].shape[1]}"}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "c_s.json").write_bytes(b.descriptor)
    text = None
    if not a.python_lower:
        argv = [a.lower, "compose", str(a.out / "c_s.json"), key, f"unit/{core.id}", ins, rec["outs"], LAYOUT,
                part3["program"], PO.digest(part3), json.dumps({key: program_digest(Program(core))}),
                str(a.out / "circuit.txt")]
        t0 = time.perf_counter()
        p = subprocess.run(argv, capture_output=True, text=True)
        rec["lower"] = {"by": "lean flock-lower-check compose", "rc": p.returncode, "wall_s": round(time.perf_counter() - t0, 2),
                        "peak_gib_children": _peak_gib(resource.RUSAGE_CHILDREN), "stdout": p.stdout.strip()[-600:],
                        "stderr": p.stderr.strip()[-3000:]}
        print("LOWER " + json.dumps(rec["lower"]), flush=True)
        if p.returncode == 0:
            text = (a.out / "circuit.txt").read_text()
    if text is None:
        from verity.protocols.proofs.spot.verifier.one_stage import one_circuit as OC
        t0 = time.perf_counter()
        text = OC.lower(b.descriptor).decode()
        rec["lower_python"] = {"by": "one_circuit.lower", "wall_s": round(time.perf_counter() - t0, 2), "peak_gib": _peak_gib()}
        print("LOWER-PYTHON " + json.dumps(rec["lower_python"]), flush=True)
        (a.out / "circuit.txt").write_text(text)
    meta = FC.meta_of(text)
    rec["circuit"] = {"bytes": len(text), "head": text.split("\n", 1)[0], "sha512": hashlib.sha512(text.encode()).hexdigest(),
                      "ports": [q[0] for q in meta.get("ports", [])]}
    phase("lower")
    for tag, bits in (("honest", hb), ("dishonest", db)):
        d = a.out / tag
        (d / "tables").mkdir(parents=True, exist_ok=True)
        (d / "circuit.txt").write_text(text)
        rows = {p: np.asarray([words16(v)], dtype=np.uint16) for p, v in bits}
        iset = SimpleNamespace(name=f"credited-{tag}", subcircuit=SimpleNamespace(id=core.id, inputs=[], outputs=[]))
        FC.write(text, iset, 0, 1, d / "inst-1.bin", d / "pub-1.bin", indices=[0], rows=rows, outs=outs_h)
        (d / "unit.tmpl").write_text(FC.template_of(text))
        (d / "partition.json").write_text(json.dumps(part3))
        (d / "program.json").write_bytes(c)
    phase("stage")
    rec["peak_gib"] = _peak_gib()
    (a.out / "stage.json").write_text(json.dumps(rec, indent=1, default=str))
    print("STAGE " + json.dumps({x: rec[x] for x in ("size", "key", "ins_bits", "outs", "circuit", "bits")}, default=str),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
