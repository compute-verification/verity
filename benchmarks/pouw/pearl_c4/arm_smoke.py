#!/usr/bin/env python3
"""Pearl-C4's arms through the harness's arm contract without the harness (no cuBLASLt library): `check_arm`, then per
shape and arm, each flag set prepared on BF16 operands of the fixture's kind, launched, gated (`check_gate`), its first
``all`` output's middle byte flipped and the gate rerun (it must refuse), and `known_bad` gated (it must refuse).  With
--run-dir, the hash call dumps its transcript as the harness would (`_Call.dump`: its ``transcript_buffers`` poisoned,
the weight side and the call relaunched) into RUN_DIR/transcripts/<arm>/<shape>, its no-write negative control beside it
(<shape>-nowrite: the same buffers poisoned and read back without a launch), and each arm's ``verifier`` (its reference
replay; arm interface v0.5, `check_verifier` when the harness has it) goes to RUN_DIR/verifiers.json with the shapes and
controls dumped.  --verify-run RUN_DIR, on the CPU alone, runs them per shape as the harness's verify.py does: the commands
in order, a transcript replay-verified (development shortcut; the protocol's check is a ZK proof, PROTOCOL.md) only by a
line starting ACCEPT that names its sha256, with no REJECT naming it and every command exiting 0, and a control passing
only by a REJECT naming its sha256.  One JSON object per line; exit 4 when any gate, control, check or replay fails.

  python3 arm_smoke.py --harness DIR [--shapes 128x1024x256 ...] [--arms PearlC4Nv PearlC4Mx] [--roles hash ...] [--run-dir D]
  python3 arm_smoke.py --verify-run D --run-id ID --commit SHA

DIR holds the harness's arm.py (benchmarks/pouw/harness).  Shapes are MxKxN, as the harness's `Shape` orders them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from verity_config.settings import of

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
settings = of(__file__)


def verify(v: dict, run_dir: Path, run_id: str, commit: str, arm: str, label: str, control: bool = False) -> dict:
    """The reference replay's commands on one shape's dump, judged as verify.py judges them; a negative control passes only
    by a REJECT line naming its transcript's sha256 (and no ACCEPT)."""
    path = run_dir / v["transcript"].format(arm=arm, shape=label)
    values = {"python": sys.executable, "run_dir": str(run_dir), "run_id": run_id, "commit": commit, "arm": arm,
              "shape": label, "transcript": str(path), "dir": str(path.parent)}
    env = dict(settings.environ(), PYTHONPATH=os.pathsep.join(
        [str(ROOT / "verity" / "core" / "protocols" / "pouw"), str(ROOT), *settings.PYTHONPATH]))
    exits, lines = [], []
    for cmd in v["commands"]:
        p = subprocess.run([x.format(**values) for x in cmd], capture_output=True, text=True, env=env)
        exits.append(p.returncode)
        lines += p.stdout.splitlines()
        if p.returncode:
            lines += p.stderr.splitlines()[-5:]
            break
    if not path.is_file():
        return {"ok": False, "exits": exits, "why": f"no transcript at {path}", "tail": lines[-5:]}
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    named = [ln for ln in lines if f"sha256:{sha}" in ln]
    accept = [ln for ln in named if re.match(r"ACCEPT\b", ln)]
    reject = [ln for ln in named if re.match(r"REJECT\b", ln)]
    ok = bool(reject) and not accept if control else bool(accept) and not reject and not any(exits)
    return {"ok": ok, "control": control, "exits": exits, "sha256": sha, "accept_line": accept[0] if accept else None,
            "reject_line": reject[0] if reject else None, "tail": [] if ok else lines[-5:]}


def operands(d, shape, seed: int):
    import arm as A

    rng = np.random.default_rng(seed)
    bufs = []
    for rows in (shape.m, shape.n):
        raw = rng.integers(0, 256, (rows, shape.k, 2), dtype=np.uint16)
        x = (raw[..., 0] >> 7) << 15 | (123 + raw[..., 1] % 6) << 7 | (raw[..., 0] & 0x7F)
        p = d.alloc(x.nbytes)
        d.write(p, x.astype("<u2").tobytes())
        bufs.append(A.Buffer(p, x.nbytes))
    return A.OperandSet(0, 0, seed, bufs[0], bufs[1])


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", type=Path)
    ap.add_argument("--shapes", nargs="*", default=["128x1024x256", "32x2048x384"])
    ap.add_argument("--arms", nargs="*", default=["PearlC4Nv", "PearlC4Mx"])
    ap.add_argument("--roles", nargs="*", default=["hash", "nohash", "plain"])
    ap.add_argument("--run-dir", type=Path, help="dump each shape's transcript (and its no-write control) under "
                    "RUN_DIR/transcripts and record each arm's reference replay in RUN_DIR/verifiers.json")
    ap.add_argument("--verify-run", type=Path, metavar="RUN_DIR",
                    help="no GPU: run the reference replays a --run-dir pass recorded; an ACCEPT is replay-verified "
                         "(development shortcut; the protocol's check is a ZK proof, PROTOCOL.md)")
    ap.add_argument("--run-id", default="arm-smoke")
    ap.add_argument("--commit", default=settings.RESEARCH_SOURCE_SHA or "unknown")
    a = ap.parse_args(argv)
    ok = True

    def emit(row: dict) -> None:
        nonlocal ok
        ok &= row.get("ok", True) is not False
        print(json.dumps(row, default=str), flush=True)

    if a.verify_run:
        recorded = json.loads((a.verify_run / "verifiers.json").read_text())
        for name, rec in recorded.items():
            for label, control in [(s, False) for s in rec["shapes"]] + [(s, True) for s in rec.get("controls", [])]:
                r = verify(rec["verifier"], a.verify_run.resolve(), a.run_id, a.commit, name, label, control)
                emit({"arm": name, "shape": label, "ok": r["ok"], "verification": r})
        emit({"verify": "pass" if ok and recorded else "FAIL"})
        return 0 if ok and recorded else 4
    if a.harness is None:
        ap.error("--harness is required without --verify-run")
    sys.path[:0] = [str(a.harness.resolve()), str(HERE)]
    import arm as A
    import pearl_c4_arm as PA

    recorded: dict = {}
    for name in a.arms:
        arm = getattr(PA, name)()
        A.check_arm(arm, {"nvfp4", "mxfp4"})
        ctx = {"transcripts_dir": str(a.run_dir / "transcripts")} if a.run_dir else {}
        build = arm.build(ctx)
        v = A.check_verifier(arm.name, build) if hasattr(A, "check_verifier") else build["verifier"]
        emit({"arm": arm.name, "build": build, "verifier_checked": hasattr(A, "check_verifier")})
        recorded[arm.name] = {"verifier": v, "shapes": [], "controls": []}
        d = arm.d
        for spec in a.shapes:
            m, k, n = (int(v) for v in spec.split("x"))
            shape = A.Shape(f"m{m}-n{n}-k{k}", m, k, n)
            why = arm.domain(shape)
            if why:
                emit({"arm": arm.name, "shape": shape.label, "outside": why})
                continue
            ops = operands(d, shape, int.from_bytes(hashlib.sha256(spec.encode()).digest()[:4], "little"))
            w = arm.work(shape)
            emit({"arm": arm.name, "shape": shape.label, "work": w.__dict__})
            for role, flags in arm.flag_sets.items():
                if role not in a.roles:
                    continue
                call = arm.prepare(None, shape, flags, ops)
                call.launch(0)
                d.sync()
                g = arm.gate(None, call, shape, flags)
                A.check_gate(g, call)
                emit({"arm": arm.name, "shape": shape.label, "role": role, "ok": g.ok, "checked": g.checked, "detail": g.detail})
                out = next(nm for nm, v in g.checked.items() if v == "all")
                buf = call.outputs[out]
                call.launch(0)
                d.sync()
                mid = bytearray(d.read(buf.ptr + buf.nbytes // 2, 1))
                d.write(buf.ptr + buf.nbytes // 2, bytes([mid[0] ^ 1]))
                flipped = arm.gate(None, call, shape, flags)
                emit({"arm": arm.name, "shape": shape.label, "role": role, "control": f"flip {out}", "ok": not flipped.ok})
                if a.run_dir and role == "hash":
                    dumps = Path(ctx["transcripts_dir"]) / arm.name
                    call.dump(dumps / shape.label)
                    call.dump(dumps / f"{shape.label}-nowrite", write=False)
                    recorded[arm.name]["controls"].append(f"{shape.label}-nowrite")
                call.free()
            bad = arm.known_bad(None, shape, ops)
            bad.launch(0)
            d.sync()
            g = arm.gate(None, bad, shape, arm.flag_sets["hash"])
            emit({"arm": arm.name, "shape": shape.label, "control": "known_bad", "ok": not g.ok,
                  "relaunch": g.detail["relaunch"]})
            bad.free()
            d.free(ops.a.ptr)
            d.free(ops.b.ptr)
            recorded[arm.name]["shapes"].append(shape.label)
    if a.run_dir:
        (a.run_dir / "verifiers.json").write_text(json.dumps(recorded, indent=1) + "\n")
    emit({"smoke": "pass" if ok else "FAIL"})
    return 0 if ok else 4


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
