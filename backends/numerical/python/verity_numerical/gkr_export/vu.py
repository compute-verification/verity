"""The full Verity VU for Candidate A: the epilogue circuit and the chained-unit witness of ``bench-instances/v1``.

A VU of ``K = 1536`` is 96 word-chained transition units from ``c_0 = 0`` plus the epilogue (``Epilogue.check``:
the RN-even cast of the last FP32 word to the BF16 output word).  Candidate A proves it as two data-parallel
circuits over one commitment (PROTOCOL.md section 12): the unit circuit (``build_unit_circuit``) over ``B * 96``
unit rows laid out VU-major (row ``v * 96 + i`` is unit ``i`` of VU ``v``), and the epilogue circuit
(:func:`build_epilogue_circuit`) over ``B`` rows.  The chain linkage (unit ``i``'s ``y`` = unit ``i+1``'s ``x.c``,
``x.c = 0`` at ``i = 0``, unit 95's ``y`` = the epilogue's ``y32``) and the public output words (``y16`` = the
recorded word) are linear constraints on the committed vector, checked by the Rust prover/verifier; this module
only lays the witness out.

    python -m verity_numerical.gkr_export.vu export --root DIR --tier vu-k1536 --lo 0 --hi 4096 --out OUT [--procs N]
    python -m verity_numerical.gkr_export.vu export --root DIR --tier vu-k1536-neg --out OUT           # one dir per id
    python -m verity_numerical.gkr_export.vu synth --out OUT --vus 2 --steps 3 [--seed S]            # tests

``--field babybear`` (export, synth) uses the BabyBear embedding of :mod:`.babybear` (limb columns, the six
rules, values mod 2^31 - 2^27 + 1) for the ``--features babybear`` build of the crate; the default is Goldilocks.

``export`` writes ``circuit.txt`` (unit), ``epilogue.txt``, ``units.bin`` (``B*steps`` rows), ``epilogue.bin``
(``B`` rows), ``public.bin`` (``B`` rows of one word: the claimed ``y16``) and ``manifest.json``.  Every unit's hints
come from WP2 ``gen_step`` on the recorded incoming accumulator; its ``y`` is the recorded outgoing accumulator
(``vu-k1536.acc.u32``), so a chain the checker would compute differently is rejected by the circuit, not repaired.
For the negatives, a ``Reject`` raised while generating hints is recorded in ``manifest.json`` (``rejected_at_witness``)
and no witness is written for that id.
"""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import random
import struct
import sys
from collections.abc import Sequence
from pathlib import Path

from verity_numerical.checker import REAL, Params, Reject, check_step, gen_step
from verity_numerical.checker.ctx import IntCtx
from verity_numerical.checker.gadgets import Epilogue
from verity_numerical.checker.step import check_step_expr, step_output

from .circuit import (
    GOLDILOCKS,
    CircuitCtx,
    _get,
    _leaves,
    _set,
    build_unit_circuit,
    export_circuit,
    unit_witness,
    verify_unit,
)

__all__ = ["FieldApi", "build_epilogue_circuit", "epilogue_witness", "field_api", "vu_rows", "write_rows"]

STEP = 16


def build_epilogue_circuit(p: Params = REAL) -> tuple[CircuitCtx, list[str]]:
    """``Epilogue.check`` as a circuit: columns ``one, y32, y16, hints...``; ``y16`` is asserted equal to the cast."""
    ctx = CircuitCtx()
    y32 = ctx.column("y32")
    y16 = ctx.column("y16")
    sample = Epilogue.gen(p, 0)
    paths: list[str] = []
    _leaves(sample, "h", paths)
    h_sym = copy.deepcopy(sample)
    for path in paths:
        _set(h_sym, path, ctx.column(path))
    got = Epilogue.check(ctx, p, y32, h_sym)
    ctx.eq("y16", got, y16)
    return ctx, paths


def epilogue_witness(ctx: CircuitCtx, paths: list[str], y32: int, y16: int, h: dict) -> list[int]:
    vals: list[int | None] = [None] * len(ctx.cols)
    vals[0] = 1
    vals[1] = y32
    vals[2] = y16
    for i, path in enumerate(paths):
        vals[3 + i] = int(_get(h, path))
    return [int(v) for v in vals]  # type: ignore[arg-type]


class FieldApi:
    """The per-field export functions (``goldilocks``: :mod:`.circuit`; ``babybear``: :mod:`.babybear`); with
    ``variant="v2"`` the checker-v2 circuits of :mod:`.v2` (Goldilocks only; PROTOCOL.md 14)."""

    def __init__(self, name: str, variant: str = "v1") -> None:
        self.name = name
        self.variant = variant
        if variant not in ("v1", "v2"):
            raise ValueError(f"unknown checker variant {variant!r}")
        if variant == "v2":
            if name != "goldilocks":
                raise ValueError("checker v2 is exported over Goldilocks only (the BabyBear limb split of its "
                                 "state_decode / epilogue word arithmetic is not built)")
            from . import v2

            self.modulus = GOLDILOCKS
            self.build_unit, self.build_epilogue = v2.build_unit_circuit_v2, v2.build_epilogue_circuit_v2
            self.unit_witness, self.epilogue_witness, self.verify_unit = None, None, v2.verify_row
            self._v2 = v2
        elif name == "goldilocks":
            self.modulus = GOLDILOCKS
            self.build_unit, self.build_epilogue = build_unit_circuit, build_epilogue_circuit
            self.unit_witness, self.epilogue_witness, self.verify_unit = unit_witness, epilogue_witness, verify_unit
        elif name == "babybear":
            from . import babybear as bb

            self.modulus = bb.BABYBEAR
            self.build_unit, self.build_epilogue = bb.build_unit_circuit_bb, bb.build_epilogue_circuit_bb
            self.unit_witness, self.epilogue_witness, self.verify_unit = bb.unit_witness_bb, bb.epilogue_witness_bb, bb.verify_unit_bb
        else:
            raise ValueError(f"unknown field {name!r}")

    def export_circuit(self, ctx: CircuitCtx, path: Path) -> None:
        if self.variant == "v2":
            self._v2.export_circuit_v2(ctx, path, modulus=self.modulus)
        else:
            export_circuit(ctx, path, modulus=self.modulus)

    def export_extra(self, unit: CircuitCtx, epi: CircuitCtx, out: Path) -> None:
        """Files beyond the two circuits: v2 writes ``chain.txt`` (the linkage, read by ``main.rs::load_vu``)."""
        if self.variant == "v2":
            self._v2.write_chain(self._v2.chain_spec(unit, epi), out / "chain.txt", self.modulus)

    def vu_rows(self, unit, epi, a, b, y16, accs=None, p: Params = REAL, cross_check: bool = True):
        if self.variant == "v2":
            return self._v2.vu_rows_v2(unit, epi, a, b, y16, accs, p, cross_check, self.modulus)
        return vu_rows(unit, epi, a, b, y16, accs, p, cross_check, api=self)


def field_api(name: str = "goldilocks", variant: str = "v1") -> FieldApi:
    return FieldApi(name, variant)


def vu_rows(unit: tuple[CircuitCtx, list[str]], epi: tuple[CircuitCtx, list[str]], a: Sequence[int], b: Sequence[int],
            y16: int, accs: Sequence[int] | None = None, p: Params = REAL, cross_check: bool = True,
            api: FieldApi | None = None) -> tuple[list[list[int]], list[int]]:
    """The unit rows (one per 16-product step, chained from ``c_0 = 0``) and the epilogue row of one VU.

    ``accs[i]`` is the recorded accumulator after step ``i``; when absent the checker's own ``step_output`` is used.
    With ``cross_check`` every unit is also run through WP2 ``check_step`` (raises ``Reject`` on a mismatch)."""
    api = api or field_api()
    uctx, upaths = unit
    ectx, epaths = epi
    if len(a) != len(b) or len(a) % STEP:
        raise Reject("operand count must be a multiple of 16")
    steps = len(a) // STEP
    if accs is not None and len(accs) != steps:
        raise Reject("one recorded accumulator per step expected")
    rows = []
    c = 0
    for i in range(steps):
        ai, bi = list(a[i * STEP:(i + 1) * STEP]), list(b[i * STEP:(i + 1) * STEP])
        h = gen_step(p, c, ai, bi)
        y = int(accs[i]) if accs is not None else step_output(p, c, ai, bi)
        if cross_check:
            check_step(p, c, ai, bi, y, h)
        rows.append(api.unit_witness(uctx, upaths, c, ai, bi, y, h))
        c = y
    he = Epilogue.gen(p, c)
    if cross_check:
        ctx = IntCtx()
        ctx.eq("y16", Epilogue.check(ctx, p, c, he), y16)
    return rows, api.epilogue_witness(ectx, epaths, c, y16, he)


def write_rows(rows: list[list[int]], path: Path, ncols: int | None = None, modulus: int = GOLDILOCKS) -> None:
    """Flat little-endian u64: ``n_rows n_cols`` then the rows, reduced mod ``modulus`` (``circuit.rs::read_witness``)."""
    ncols = len(rows[0]) if rows else (ncols or 0)
    with open(path, "wb") as f:
        f.write(struct.pack("<QQ", len(rows), ncols))
        f.writelines(struct.pack(f"<{ncols}Q", *(v % modulus for v in r)) for r in rows)


# ---------------------------------------------------------------------------------------------------------------------
# dataset export (multiprocessing: the circuits are rebuilt once per worker)
# ---------------------------------------------------------------------------------------------------------------------

_UNIT: tuple | None = None
_EPI: tuple | None = None
_API: FieldApi | None = None


def _init_worker(field: str = "goldilocks", variant: str = "v1") -> None:
    global _UNIT, _EPI, _API
    _API = field_api(field, variant)
    _UNIT = _API.build_unit()
    _EPI = _API.build_epilogue()


def _one(task: tuple[int, list[int], list[int], int, list[int] | None]) -> tuple[int, bytes | None, bytes | None, str | None]:
    vid, a, b, y16, accs = task
    assert _UNIT is not None and _EPI is not None and _API is not None
    try:
        rows, erow = _API.vu_rows(_UNIT, _EPI, a, b, y16, accs, cross_check=accs is not None)
    except Reject as e:
        return vid, None, None, str(e)
    m = _API.modulus
    ncols = len(rows[0])
    ub = b"".join(struct.pack(f"<{ncols}Q", *(v % m for v in r)) for r in rows)
    eb = struct.pack(f"<{len(erow)}Q", *(v % m for v in erow))
    return vid, ub, eb, None


def _write_batch(out: Path, results: list[tuple[int, bytes, bytes]], y16s: list[int], steps: int, ucols: int, ecols: int,
                 modulus: int = GOLDILOCKS) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "units.bin", "wb") as f:
        f.write(struct.pack("<QQ", len(results) * steps, ucols))
        f.writelines(ub for _, ub, _ in results)
    with open(out / "epilogue.bin", "wb") as f:
        f.write(struct.pack("<QQ", len(results), ecols))
        f.writelines(eb for _, _, eb in results)
    write_rows([[y] for y in y16s], out / "public.bin", modulus=modulus)


def export(root: Path, tier: str, lo: int, hi: int | None, out: Path, procs: int, field: str = "goldilocks",
           variant: str = "v1") -> dict:
    from verity_numerical.bench import instances as I

    api = field_api(field, variant)
    tierdata = I.load(tier, root)
    hi = tierdata.n if hi is None else min(hi, tierdata.n)
    unit_ctx, _ = api.build_unit()
    epi_ctx, _ = api.build_epilogue()
    out.mkdir(parents=True, exist_ok=True)
    api.export_circuit(unit_ctx, out / "circuit.txt")
    api.export_circuit(epi_ctx, out / "epilogue.txt")
    api.export_extra(unit_ctx, epi_ctx, out)
    steps = tierdata.K // STEP
    tasks = []
    for vid in range(lo, hi):
        x, w, y = tierdata.instance(vid)
        accs = tierdata.accs(vid) if tierdata.acc is not None else None
        tasks.append((vid, x, w, y, accs))
    results: dict[int, tuple[bytes | None, bytes | None, str | None]] = {}
    if procs > 1:
        with mp.get_context("fork").Pool(procs, initializer=_init_worker, initargs=(field, variant)) as pool:
            for vid, ub, eb, err in pool.imap_unordered(_one, tasks, chunksize=4):
                results[vid] = (ub, eb, err)
    else:
        _init_worker(field, variant)
        for t in tasks:
            vid, ub, eb, err = _one(t)
            results[vid] = (ub, eb, err)
    # the dataset's identity is the committed manifest (a pod rebuild rewrites the `generated` block, not the arrays)
    committed = I.DEFAULT_ROOT / "manifest.json"
    manifest: dict = {"tier": tier, "range": [lo, hi], "steps": steps, "unit_columns": len(unit_ctx.cols),
                      "epilogue_columns": len(epi_ctx.cols), "field": field, "variant": variant, "modulus": api.modulus,
                      "manifest_sha256": I.manifest_sha256(I.DEFAULT_ROOT) if committed.is_file() else I.manifest_sha256(root),
                      "root_manifest_sha256": I.manifest_sha256(root), "root": str(root),
                      "rejected_at_witness": {}, "written": []}
    if tier == "vu-k1536-neg":
        for vid in range(lo, hi):
            ub, eb, err = results[vid]
            if err is not None:
                manifest["rejected_at_witness"][vid] = err
                continue
            _write_batch(out / "neg" / str(vid), [(vid, ub, eb)], [int(tierdata.y[vid])], steps, len(unit_ctx.cols), len(epi_ctx.cols),
                         api.modulus)
            api.export_extra(unit_ctx, epi_ctx, out / "neg" / str(vid))
            manifest["written"].append(vid)
        rows = {r[0]: r for r in tierdata.index}
        manifest["index"] = {vid: dict(zip(tierdata.columns, rows[vid])) for vid in range(lo, hi)}
    else:
        ok = []
        for vid in range(lo, hi):
            ub, eb, err = results[vid]
            if err is not None:
                manifest["rejected_at_witness"][vid] = err
            else:
                ok.append((vid, ub, eb))
        if manifest["rejected_at_witness"]:
            print(f"{len(manifest['rejected_at_witness'])} positives rejected at witness generation", file=sys.stderr)
        _write_batch(out, ok, [int(tierdata.y[vid]) for vid, _, _ in ok], steps, len(unit_ctx.cols), len(epi_ctx.cols), api.modulus)
        manifest["written"] = [vid for vid, _, _ in ok]
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str) + "\n")
    return manifest


# ---------------------------------------------------------------------------------------------------------------------
# synthetic VUs for the Rust tests (random finite operands, the checker's own chain)
# ---------------------------------------------------------------------------------------------------------------------

def synth(out: Path, vus: int, steps: int, seed: int, p: Params = REAL, field: str = "goldilocks", variant: str = "v1") -> dict:
    api = field_api(field, variant)
    if variant == "v2":
        return api._v2.synth_v2(out, vus, steps, seed, p, api)
    m = api.modulus
    rng = random.Random(seed)
    unit = api.build_unit()
    epi = api.build_epilogue()
    uctx, _ = unit
    ectx, _ = epi

    def bf16() -> int:
        return (rng.randrange(2) << 15) | (rng.randrange(100, 150) << 7) | rng.randrange(2**7)

    out.mkdir(parents=True, exist_ok=True)
    api.export_circuit(uctx, out / "circuit.txt")
    api.export_circuit(ectx, out / "epilogue.txt")
    all_rows: list[list[int]] = []
    erows: list[list[int]] = []
    y16s: list[int] = []
    for _ in range(vus):
        a = [bf16() for _ in range(steps * STEP)]
        b = [bf16() for _ in range(steps * STEP)]
        c = 0
        for i in range(steps):
            c = step_output(p, c, a[i * STEP:(i + 1) * STEP], b[i * STEP:(i + 1) * STEP])
        y16 = Epilogue.check(IntCtx(), p, c, Epilogue.gen(p, c))
        rows, erow = vu_rows(unit, epi, a, b, y16, api=api)
        for r in rows:
            assert api.verify_unit(uctx, r) is None
        assert api.verify_unit(ectx, erow) is None
        all_rows += rows
        erows.append(erow)
        y16s.append(y16)
    write_rows(all_rows, out / "units.bin", modulus=m)
    write_rows(erows, out / "epilogue.bin", modulus=m)
    write_rows([[y] for y in y16s], out / "public.bin", modulus=m)
    # negatives: a wrong public word; a broken chain (unit 1's incoming word replaced by a valid but different one)
    neg = out / "negatives"
    neg.mkdir(exist_ok=True)
    write_rows([[y16s[0] ^ 1], *[[y] for y in y16s[1:]]], neg / "public_wrong_word.bin", modulus=m)
    if steps >= 2:
        # re-generate VU 0's unit 1 from a different incoming word (its own hints consistent) -> linkage must fail
        limbed = field == "babybear"
        c_in = (all_rows[1][1] + (all_rows[1][2] << 16)) if limbed else all_rows[1][1]
        c_alt = c_in ^ (1 << 20)  # flip a fraction bit of x.c
        off = 3 if limbed else 2
        a_ops = [all_rows[1][off + i] for i in range(STEP)]
        b_ops = [all_rows[1][off + STEP + i] for i in range(STEP)]
        h = gen_step(p, c_alt, a_ops, b_ops)
        y_alt = check_step_expr(IntCtx(), p, c_alt, a_ops, b_ops, h)
        alt_row = api.unit_witness(uctx, unit[1], c_alt, a_ops, b_ops, y_alt, h)
        assert api.verify_unit(uctx, alt_row) is None
        broken = list(all_rows)
        broken[1] = alt_row
        write_rows(broken, neg / "units_broken_chain.bin", modulus=m)
    return {"vus": vus, "steps": steps, "unit_columns": len(uctx.cols), "epilogue_columns": len(ectx.cols), "y16": y16s,
            "field": field}


def finalize(result: Path, export_manifest: Path, run_id: str | None = None) -> list[str]:
    """Complete the Rust prover's ``result.json``: the ``instances`` reference of the export, ``overhead.vs_native_peak``
    for VU-scope runs, and the contract's validation; returns the contract's problem list (empty when conforming)."""
    from verity_numerical.bench import contract

    doc = json.loads(result.read_text())
    man = json.loads(export_manifest.read_text())
    fp = doc["workload_fingerprint"]
    lo, _hi = man["range"]
    if run_id:
        doc["run_id"] = run_id
    fp["instances"] = contract.instances_ref(man["tier"], lo, lo + fp["B"], man["manifest_sha256"])
    values = {m["name"]: m["value"] for m in doc["measurements"]}
    if fp.get("unit") == "vu" and "vu.count" in values:
        per_vu = values["t.total"] / values["vu.count"]
        doc["measurements"].append(contract.measurement("overhead.vs_native_peak", contract.overhead(per_vu, fp["K"])))
    problems = contract.validate(doc)
    doc["contract_problems"] = problems
    result.write_text(json.dumps(doc, indent=1) + "\n")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m verity_numerical.gkr_export.vu", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--root", required=True)
    e.add_argument("--tier", default="vu-k1536")
    e.add_argument("--lo", type=int, default=0)
    e.add_argument("--hi", type=int, default=None)
    e.add_argument("--out", required=True)
    e.add_argument("--procs", type=int, default=1)
    e.add_argument("--field", default="goldilocks", choices=("goldilocks", "babybear"))
    e.add_argument("--variant", default="v1", choices=("v1", "v2"), help="checker arithmetisation (v2: PROTOCOL.md 14)")
    s = sub.add_parser("synth")
    s.add_argument("--out", required=True)
    s.add_argument("--vus", type=int, default=2)
    s.add_argument("--steps", type=int, default=3)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--field", default="goldilocks", choices=("goldilocks", "babybear"))
    s.add_argument("--variant", default="v1", choices=("v1", "v2"), help="checker arithmetisation (v2: PROTOCOL.md 14)")
    f = sub.add_parser("finalize")
    f.add_argument("--result", required=True)
    f.add_argument("--export-manifest", required=True)
    f.add_argument("--run-id", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "export":
        m = export(Path(args.root), args.tier, args.lo, args.hi, Path(args.out), args.procs, args.field, args.variant)
        print(json.dumps({k: v for k, v in m.items() if k not in ("written", "index")}, default=str))
    elif args.cmd == "finalize":
        problems = finalize(Path(args.result), Path(args.export_manifest), args.run_id)
        print(json.dumps({"contract_problems": problems}))
        return 1 if problems else 0
    else:
        print(json.dumps(synth(Path(args.out), args.vus, args.steps, args.seed, field=args.field, variant=args.variant)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
