"""The e2e's two panel rows (research store ``internal/pouw/panel/add-a-row.md``), from the window's ``e2e.json`` and the
verify job's ``prefill-verify.json``, ``decode-verify.json`` and ``control-verify.json``, as ``panel.py append`` commands.

  python benchmarks/pouw/pearl_c_vllm/panel_rows.py E2E_JSON VERIFY_DIR [--panel PANEL_PY] [--append]

The rows go under ``pearl-c-sm120`` at the scheme's version (v1-h1), with the coordinator's e2e shapes (11:50Z item 5).  Each
row's slowdown is ``pearlc`` over stock vLLM ``fp8`` in the same window, interleaved rep by rep, or in ABAB blocks when the two
engines could not stay resident, which the row's description then says; ``--hash-free`` is ``pearlc-nohash`` over ``fp8``.  The clocks
are every rep's SM clock, the arm's and the ``fp8`` baseline's.  Refused, with the reason, unless every gate held, both verify
passes repeated their timed commitments and were accepted, and the negative control was rejected.  Without ``--append`` it
prints the commands.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

PANEL = "/cursor/stores/bc-b729c175-2ef6-418e-98fe-10896709028b/internal/pouw/panel/panel.py"
PHASES = {"prefill_8192": ("prefill", "prefill_8192", "prefill", "s"), "decode_b32_step": ("decode", "decode_b32", "decode", "step_s")}


def interleaving(summary: dict) -> str:
    """How the baseline was timed against the arm, as the row says it (the root's ruling, 12:11Z)."""
    if summary["interleave"] == "rep":
        return ("the stock FP8 baseline interleaved rep by rep: both engines resident in their own processes on the same GPU, "
                "each idle while the other times")
    return ("the stock FP8 baseline in ABAB blocks of reps, not rep by rep: both engines could not stay resident on the GPU ("
            + (summary.get("resident_failure") or "blocks requested") + ")")


def rows(e2e: dict, verdicts: dict[str, dict], panel: str) -> list[list[str]]:
    res, summary, run = e2e["results"], e2e["summary"], e2e.get("run")
    bad = [f"gates of {m}" for m, ok in summary["gates"].items() if not ok]
    bad += [f"{p}: its commitment did not repeat" for p in ("prefill", "decode")
            if not summary["verify_pass"].get(p, {}).get("same_commitment_as_timed")]
    bad += [f"{p}: not accepted" for p in ("prefill", "decode") if not verdicts[p].get("accepted")]
    if verdicts["control"].get("accepted", True):
        bad.append("the negative control was not rejected")
    if not run or any(run not in verdicts[p].get("transcript", "") for p in verdicts):
        bad.append(f"the verdicts don't name the window's run ({run})")
    if bad:
        raise SystemExit("no panel rows: " + "; ".join(bad))
    version = e2e["scheme"].removeprefix("pearl-c-sm120-")
    out = []
    for key, (phase, reps, pass_, _) in PHASES.items():
        s = summary["split"][key]
        clocks = {m: ",".join(str(r["clock"]["sm_mhz"]) for r in res[m]["reps"][reps]) for m in ("pearlc", "fp8")}
        desc = (f"PoUW MVP e2e: Llama-3.1-8B-Instruct in vLLM (BF16, TP 1, eager) with every linear committed by "
                f"pouw_pearl_c_device; {phase} {'one 8,192-token prompt' if phase == 'prefill' else '32 sequences, per step'}; "
                f"over stock vLLM FP8 {s['slowdown_vs_fp8']:.3f}x (GEMM {s['gemm_x_fp8']:.3f}, hashing and records "
                f"{s['hashing_x_fp8']:.3f}, else {s['else_x_fp8']:.3f}), over BF16 {s['slowdown_vs_bf16']:.3f}x; "
                + interleaving(summary))
        cmd = [sys.executable, panel, "append", "--line", "pearl-c-sm120", "--version", version, "--precision", "fp8",
               "--phase", phase, "--shape", s["shape"], "--kind", "measured", "--slowdown", f"{s['slowdown_vs_fp8']:.4f}",
               "--hash-free", f"{s['nohash_s'] / s['fp8_s']:.4f}", "--change", "kernel", "--description", desc,
               "--source", run, "--run", run, "--verifier-commit", verdicts[pass_]["verifier_commit"],
               "--verifier-accept", verdicts[pass_]["line"], "--transcript", verdicts[pass_]["transcript"],
               "--negative-control", verdicts["control"]["line"], "--sm-clock-arm", clocks["pearlc"],
               "--sm-clock-base", clocks["fp8"], "--by", "bc-dd22acf8"]
        if phase == "decode":
            cmd += ["--decode-method", "dependent-chain"]
        out.append(cmd)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("e2e", type=Path)
    p.add_argument("verify_dir", type=Path)
    p.add_argument("--panel", default=PANEL)
    p.add_argument("--append", action="store_true")
    a = p.parse_args(argv)
    verdicts = {x: json.loads((a.verify_dir / f"{x}-verify.json").read_text()) for x in ("prefill", "decode", "control")}
    for cmd in rows(json.loads(a.e2e.read_text()), verdicts, a.panel):
        if a.append:
            subprocess.run(cmd, check=True)
        else:
            print(shlex.join(cmd))
    return 0


if __name__ == "__main__":
    sys.exit(main())
