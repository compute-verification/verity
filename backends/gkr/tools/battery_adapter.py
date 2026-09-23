"""Run the BabyBear forgery battery (``fixtures/redteam/babybear-battery/v1``, 83 must-reject cases) through the Rust
prover + verifier built with ``--features babybear``.

Every case becomes the prover's own input format (a circuit from ``verity_numerical.gkr_export`` in BabyBear mode and a
witness file of the case's mutated hints, reduced mod p as a forger would commit them) and is proved and verified
end to end (``verity-gkr run`` / ``run-vu``, PCS mode); ``prover_accepts`` is "the verifier accepted".  The
verifier's constraint check on the same witness without proving (``check-witness``) is recorded alongside.

* ``step`` cases (wrap, fractional, zero_swap, hint_swap): one row of the unit circuit -- ``c``, the operands, the
  claimed ``y`` as limbs, the mutated hint tree (limb hints completed canonically), the derived range chunks.
* ``vu`` cases (chain, epilogue): a 2-unit VU through the unit + epilogue circuits with the chain linkage.  The
  limbed circuit links unit ``i+1``'s incoming limbs to unit ``i``'s outgoing limbs, so the forger's only choice is
  the honest linkage: unit ``i``'s outgoing limbs are what the unit's (mutated) hints make the circuit produce,
  evaluated without checking (a lax mod-p pass); the "+p accumulator" of the chain cases has no limb form at all.
* ``gadget`` cases (z3 models with one rule dropped): the gadget's own contract as a circuit -- the premise ranges
  on its inputs, the embedded gadget on input/hint columns, the claimed outputs as ``eq`` -- and the model as the row.

    uv run --package verity-numerical python backends/gkr/tools/battery_adapter.py [--bin verity-gkr] [--out report.json]

The exit status is 0 iff every case is rejected.  The report is ``run_battery``'s (``BatteryReport``), plus the
per-case verdicts of the prover and of ``check-witness``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from verity_numerical.checker import REAL, Params
from verity_numerical.checker.ctx import Table
from verity_numerical.gkr_export.babybear import (
    BABYBEAR,
    BabyBearCircuitCtx,
    _derive,
    _get,
    build_epilogue_circuit_bb,
    build_unit_circuit_bb,
    epilogue_witness_bb,
    export_circuit_bb,
    limb_hints,
    unit_witness_bb,
    write_witness_bb,
)
from verity_numerical.redteam.babybear import DEFAULT, BabyBearCtx, check_step_bb, limbs
from verity_numerical.redteam.battery_babybear import Case, load_battery, run_battery

HERE = Path(__file__).resolve().parent
GKR = HERE.parent
LIMB = 16


class LaxBB(BabyBearCtx):
    """Checks nothing; forms the unit's output limbs as the circuit's wires would (a forger's linkage values)."""

    def rng(self, name, x, lo, hi):
        pass

    def eq(self, name, a, b):
        pass

    def boolean(self, name, b):
        pass

    def lookup(self, name, table: Table, key):
        k = (key - table.lo) % self.p + table.lo
        try:
            return table.row(k)
        except KeyError:
            return table.fn(table.lo)

    def lt_pow(self, name, x, ipow, bits):
        pass


def unit_row_limbs(ctx, paths, c: tuple[int, int], a, b, y: tuple[int, int], h: dict) -> list[int]:
    """`unit_witness_bb` with the incoming / outgoing words given as limb pairs (the chain's linkage values)."""
    k = len(a)
    h = limb_hints(h)
    vals: list[int | None] = [None] * len(ctx.cols)
    vals[0] = 1
    vals[1], vals[2] = int(c[0]), int(c[1])
    for i in range(k):
        vals[3 + i] = int(a[i])
        vals[3 + k + i] = int(b[i])
    vals[3 + 2 * k], vals[4 + 2 * k] = int(y[0]), int(y[1])
    for i, path in enumerate(paths):
        vals[5 + 2 * k + i] = int(_get(h, path))
    return _derive(ctx, vals)


class Adapter:
    def __init__(self, binary: Path, work: Path, p: Params = REAL) -> None:
        self.bin = binary
        self.work = work
        self.p = p
        self.unit = build_unit_circuit_bb(p)
        self.epi = build_epilogue_circuit_bb(p)
        self.vu_dir = work / "vu"
        self.vu_dir.mkdir(parents=True, exist_ok=True)
        export_circuit_bb(self.unit[0], self.vu_dir / "circuit.txt")
        export_circuit_bb(self.epi[0], self.vu_dir / "epilogue.txt")
        self.gadget_circuits: dict[str, tuple[Path, list[str], list[str], int]] = {}
        self.verdicts: dict[str, dict] = {}
        self.not_exercised: list[str] = []

    # -- the Rust side ---------------------------------------------------------------------------------------------
    def _run(self, *args: str) -> tuple[bool, str]:
        r = subprocess.run([str(self.bin), *args], capture_output=True, text=True, check=False)
        text = r.stderr + r.stdout
        m = re.search(r'verify=Err\("([^"]*)"\)', text)
        if m:
            return r.returncode == 0, m.group(1)
        tail = (text.strip().splitlines() or [""])[-1]
        return r.returncode == 0, tail[-200:]

    def prove_unit(self, circuit: Path, witness: Path) -> tuple[bool, str]:
        return self._run("run", "--circuit", str(circuit), "--witness", str(witness), "--units", "1")

    def check_witness(self, circuit: Path, witness: Path) -> tuple[bool, str]:
        return self._run("check-witness", "--circuit", str(circuit), "--witness", str(witness))

    def honest_self_check(self) -> dict[str, str]:
        """An honest unit and an honest 2-unit VU through the same paths: both must be accepted."""
        import random

        from verity_numerical.checker.ctx import IntCtx
        from verity_numerical.checker.step import check_vu_expr
        from verity_numerical.gkr_export.circuit import random_finite_unit
        from verity_numerical.redteam.babybear import gen_vu_bb

        rng = random.Random(7)
        c, a, b, y, h = random_finite_unit(rng)
        ok_s, _ = self.step_case(Case("honest-step", "honest", "step", "REAL", {"c": c, "a": a, "b": b}, h, {"y": y}, {"y": y}, "none"))
        _c2, a2, b2, _, _ = random_finite_unit(rng)
        aa, bb = list(a) + list(a2), list(b) + list(b2)
        hv = gen_vu_bb(self.p, aa, bb)
        y16 = int(check_vu_expr(IntCtx(), self.p, aa, bb, hv))
        ok_v, _ = self.vu_case(Case("honest-vu", "honest", "vu", "REAL", {"a": aa, "b": bb}, hv, {"y16": y16}, {"y16": y16}, "none"))
        return {"step": "accept" if ok_s else "reject", "vu": "accept" if ok_v else "reject"}

    # -- cases -----------------------------------------------------------------------------------------------------
    def step_case(self, case: Case) -> tuple[bool, dict]:
        ctx, paths = self.unit
        x, h = case.inputs, case.hints
        vals = unit_witness_bb(ctx, paths, x["c"], x["a"], x["b"], case.claimed["y"], h)
        d = self.work / f"{case.id}.bin"
        write_witness_bb([vals], d)
        ok, msg = self.prove_unit(self.vu_dir / "circuit.txt", d)
        cw, cmsg = self.check_witness(self.vu_dir / "circuit.txt", d)
        return ok, {"prover": "accept" if ok else f"reject: {msg}", "check_witness": "accept" if cw else f"reject: {cmsg}"}

    def vu_case(self, case: Case) -> tuple[bool, dict]:
        uctx, upaths = self.unit
        ectx, epaths = self.epi
        a, b, h = list(case.inputs["a"]), list(case.inputs["b"]), case.hints
        k = len(a) // len(h["steps"])
        c = (0, 0)
        rows = []
        for s, start in enumerate(range(0, len(a), k)):
            hs = limb_hints(h["steps"][s])
            lo, hi = check_step_bb(LaxBB(), self.p, c, a[start : start + k], b[start : start + k], hs, DEFAULT)
            y = (int(lo) % BABYBEAR, int(hi) % BABYBEAR)
            rows.append(unit_row_limbs(uctx, upaths, c, a[start : start + k], b[start : start + k], y, hs))
            c = y
        erow = epilogue_witness_bb(ectx, epaths, 0, case.claimed["y16"], h["epilogue"])
        erow[1], erow[2] = c  # the epilogue's y32 limbs are the last unit's outgoing limbs (linkage)
        erow = _derive(ectx, list(erow))
        d = self.work / case.id
        d.mkdir(exist_ok=True)
        write_witness_bb(rows, d / "units.bin")
        write_witness_bb([erow], d / "epilogue.bin")
        write_witness_bb([[case.claimed["y16"]]], d / "public.bin")
        for f in ("circuit.txt", "epilogue.txt"):
            if not (d / f).exists():
                os.symlink(self.vu_dir / f, d / f)
        ok, msg = self._run("run-vu", "--dir", str(d))
        cw_u, cmu = self.check_witness(self.vu_dir / "circuit.txt", d / "units.bin")
        cw_e, cme = self.check_witness(self.vu_dir / "epilogue.txt", d / "epilogue.bin")
        return ok, {
            "prover": "accept" if ok else f"reject: {msg}",
            "check_witness": "accept" if cw_u and cw_e else f"reject: {cmu if not cw_u else cme}",
        }

    @staticmethod
    def gadget_hints(case: Case) -> dict:
        """The model's hints plus the limb hints the embedding requires (a z3 model with LIMB16 dropped has none):
        the canonical split of the value mod p, the only pair of in-range limbs congruent to it."""
        h = dict(case.hints)
        for whole, lo, hi in (("f", "f_lo", "f_hi"), ("low", "low_lo", "low_hi")):
            if whole in h and lo not in h:
                h[lo], h[hi] = limbs(int(h[whole]) % BABYBEAR)
        return h

    def gadget_circuit(self, case: Case) -> tuple[Path, list[str], list[str], int]:
        from verity_numerical.redteam.z3_babybear import SPECS_BB

        key = f"{case.gadget}:{','.join(sorted(case.inputs))}:{','.join(sorted(self.gadget_hints(case)))}"
        if key in self.gadget_circuits:
            return self.gadget_circuits[key]
        spec = SPECS_BB[case.gadget]
        ctx = BabyBearCircuitCtx()
        xin = sorted(case.inputs)
        hin = sorted(self.gadget_hints(case))
        xcols = {n: ctx.column(f"x.{n}") for n in xin}
        hcols = {n: ctx.column(f"h.{n}") for n in hin}
        n_out = len(case.claimed["outputs"])
        ocols = [ctx.column(f"out[{i}]") for i in range(n_out)]
        contract = spec.contract(DEFAULT, self.p)
        for name, (lo, hi) in contract.inputs.items():  # the premises: what upstream gadgets guarantee
            if name in xcols:
                ctx.rng(f"premise.{name}", xcols[name], lo, hi)
        outs = spec.call(DEFAULT, ctx, self.p, dict(xcols), dict(hcols))
        for i, got in enumerate(outs):
            ctx.eq(f"out[{i}]", got, ocols[i])
        path = self.work / f"gadget_{case.gadget}_{len(self.gadget_circuits)}.txt"
        export_circuit_bb(ctx, path)
        self.gadget_circuits[key] = (path, xin, hin, n_out)
        self._gadget_ctx = getattr(self, "_gadget_ctx", {})
        self._gadget_ctx[key] = ctx
        return self.gadget_circuits[key]

    def gadget_case(self, case: Case) -> tuple[bool, dict]:
        path, xin, hin, _n_out = self.gadget_circuit(case)
        key = f"{case.gadget}:{','.join(sorted(case.inputs))}:{','.join(sorted(self.gadget_hints(case)))}"
        ctx = self._gadget_ctx[key]
        hints = self.gadget_hints(case)
        vals: list[int | None] = [None] * len(ctx.cols)
        vals[0] = 1
        i = 1
        for n in xin:
            vals[i] = int(case.inputs[n])
            i += 1
        for n in hin:
            vals[i] = int(hints[n])
            i += 1
        for o in case.claimed["outputs"]:
            vals[i] = int(o)
            i += 1
        row = _derive(ctx, vals)
        d = self.work / f"{case.id}.bin"
        write_witness_bb([row], d)
        ok, msg = self.prove_unit(path, d)
        cw, cmsg = self.check_witness(path, d)
        return ok, {"prover": "accept" if ok else f"reject: {msg}", "check_witness": "accept" if cw else f"reject: {cmsg}"}

    def accepts(self, case: Case) -> bool:
        t0 = time.time()
        try:
            fn = {"step": self.step_case, "vu": self.vu_case, "gadget": self.gadget_case}[case.level]
            ok, v = fn(case)
        except Exception as e:  # noqa: BLE001 -- a witness the adapter cannot even form: counted as *not exercised*
            ok, v = False, {"prover": f"not exercised: adapter raised {e!r}", "check_witness": "n/a"}
            self.not_exercised.append(case.id)
        v["seconds"] = round(time.time() - t0, 3)
        self.verdicts[case.id] = v
        print(f"{case.id:<24} {case.cls:<11} {case.level:<7} {'ACCEPT (BUG)' if ok else 'reject':<13} {v['prover'][:90]}", flush=True)
        return ok


def build_binary(target: Path) -> Path:
    env = dict(os.environ, CARGO_TARGET_DIR=str(target))
    subprocess.run(["cargo", "build", "--release", "--features", "babybear"], cwd=GKR, env=env, check=True)
    return target / "release" / "verity-gkr"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", type=Path, default=None, help="verity-gkr built with --features babybear (default: build it)")
    ap.add_argument("--target", type=Path, default=Path(os.environ.get("CARGO_TARGET_DIR", GKR / "target")) / "babybear")
    ap.add_argument("--out", type=Path, default=None, help="report JSON")
    ap.add_argument("--work", type=Path, default=None, help="keep the generated circuits/witnesses here")
    ap.add_argument("--only", default=None, help="comma-separated case ids or classes")
    args = ap.parse_args(argv)
    binary = args.bin or build_binary(args.target)
    cases = load_battery()
    if args.only:
        sel = set(args.only.split(","))
        cases = [c for c in cases if c.id in sel or c.cls in sel or c.level in sel]
    work = args.work or Path(tempfile.mkdtemp(prefix="bb-battery-"))
    work.mkdir(parents=True, exist_ok=True)
    ad = Adapter(binary, work)
    honest = ad.honest_self_check()
    print(f"honest self-check (the adapter's format is accepted by the prover): step={honest['step']} vu={honest['vu']}")
    report = run_battery(ad.accepts, cases)
    print(report.render())
    cw_accepted = [cid for cid, v in ad.verdicts.items() if v["check_witness"] == "accept"]
    print(f"check-witness (constraint check without proving) accepted: {len(cw_accepted)}/{len(cases)}")
    print(f"not exercised (adapter could not form the witness): {len(ad.not_exercised)}")
    ok = report.ok and not ad.not_exercised and honest["step"] == "accept" and honest["vu"] == "accept"
    if args.out:
        from dataclasses import asdict

        args.out.write_text(json.dumps({
            **asdict(report), "ok": ok, "honest_self_check": honest, "not_exercised": ad.not_exercised,
            "check_witness_accepted": cw_accepted, "verdicts": ad.verdicts, "binary": str(binary),
        }, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
