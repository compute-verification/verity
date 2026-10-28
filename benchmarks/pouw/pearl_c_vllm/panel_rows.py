"""The e2e's panel rows (research store ``internal/pouw/panel/add-a-row.md``), from the window's ``e2e.json`` and the verify
job's ``prefill-verify.json``, ``decode-verify.json`` and ``control-verify.json`` (and ``control-leaves-verify.json`` where the
window retained it), as ``panel.py append`` commands.

  python benchmarks/pouw/pearl_c_vllm/panel_rows.py E2E_JSON VERIFY_DIR [--panel PANEL_PY] [--append]

The rows go under ``pearl-c-sm120`` at the scheme's version (v1-h3), at the panel's e2e shapes.  Each
row's slowdown is ``pearlc`` over stock vLLM ``fp8`` in the same window, interleaved rep by rep, or in ABAB blocks when the two
engines could not stay resident, which the row's description then says; ``--hash-free`` is ``pearlc-nohash`` over ``fp8``.  The clocks
are every rep's SM clock, the arm's and the ``fp8`` baseline's.  Refused, with the reason, unless every gate held, both verify
passes repeated their timed commitments and were replay-verified (development shortcut; the protocol's check is a ZK
proof), and the negative control was rejected.  The decode row is the
prefill row's attempt, another phase of it (``--attempt``).  Without ``--append`` it prints the commands.

When the window also timed stock FP8 as vLLM serves it (``--fp8-graphs``), decode's headline row is over that graphed
baseline (``slowdown_vs_fp8_graphs``, its reps' clocks as the base's) at the decode shape, and the eager row goes beside it
at the same shape with `EAGER_SUFFIX`, in the same attempt, as the panel's rule says.  Both pass the same gate.
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

from verity_config.settings import of

settings = of(__file__)
PANEL = "/cursor/stores/bc-b729c175-2ef6-418e-98fe-10896709028b/internal/pouw/panel/panel.py"
#: the decode shape's suffix for the row over eager stock FP8, beside the headline over graphed FP8
EAGER_SUFFIX = "-eager"
PHASES = {"prefill_8192": ("prefill", "prefill_8192", "prefill", "s"), "decode_b32_step": ("decode", "decode_b32", "decode", "step_s")}


def interleaving(summary: dict) -> str:
    """How the baseline was timed against the arm, as the row says it."""
    if summary["interleave"] == "rep":
        return ("the stock FP8 baseline interleaved rep by rep: both engines resident in their own processes on the same GPU, "
                "each idle while the other times")
    return ("the stock FP8 baseline in ABAB blocks of reps, not rep by rep: both engines could not stay resident on the GPU ("
            + (summary.get("resident_failure") or "blocks requested") + ")")


#: the lane that appends the rows
BY = settings.PANEL_BY


def rows(e2e: dict, verdicts: dict[str, dict], panel: str, kernel_ratio: tuple[float, str] | None = None) -> list[list[str]]:
    res, summary, run = e2e["results"], e2e["summary"], e2e.get("run")
    bad = [f"gates of {m}" for m, ok in summary["gates"].items() if not ok]
    bad += [f"{p}: its commitment did not repeat" for p in ("prefill", "decode")
            if not summary["verify_pass"].get(p, {}).get("same_commitment_as_timed")]
    bad += [f"{p}: not accepted" for p in ("prefill", "decode") if not verdicts[p].get("accepted")]
    bad += [f"{c}: the negative control was not rejected" for c in verdicts if c.startswith("control")
            and verdicts[c].get("accepted", True)]
    bad += [f"{p}: its eager replay did not repeat the commitment" for p in ("prefill", "decode")
            if summary["verify_pass"].get(p, {}).get("same_commitment_eager") is False]
    if not run or any(run not in verdicts[p].get("transcript", "") for p in verdicts):
        bad.append(f"the verdicts don't name the window's run ({run})")
    if bad:
        raise SystemExit("no panel rows: " + "; ".join(bad))
    version = e2e["scheme"].removeprefix("pearl-c-sm120-")
    out = []
    for key, (phase, reps, pass_, _) in PHASES.items():
        s = summary["split"][key]
        clocks = {m: ",".join(str(r["clock"]["sm_mhz"]) for r in res[m]["reps"][reps]) for m in ("pearlc", "fp8")}
        graphs = bool(e2e.get("graphs")) and phase == "decode"
        split = (f"hashing and records {s['hashing_x_fp8']:.3f}, the rest {s['gemm_x_fp8'] + s['else_x_fp8']:.3f}; no GEMM "
                 "split: the GEMM runs inside the call's graph" if graphs else
                 f"GEMM {s['gemm_x_fp8']:.3f}, hashing and records {s['hashing_x_fp8']:.3f}, else {s['else_x_fp8']:.3f}")
        desc = (f"PoUW MVP e2e: Llama-3.1-8B-Instruct in vLLM (BF16, TP 1, eager) with every linear committed by "
                f"pouw_pearl_c_device{' (--graphs: each decode call one CUDA graph)' if e2e.get('graphs') else ''}"
                "; "
                f"{phase} {'one 8,192-token prompt' if phase == 'prefill' else '32 sequences, per step'}; "
                f"over stock vLLM FP8 {s['slowdown_vs_fp8']:.3f}x ({split}), over BF16 {s['slowdown_vs_bf16']:.3f}x; "
                + (f"over stock FP8 as vLLM serves it (compiled, CUDA graphs) {s['slowdown_vs_fp8_graphs']:.3f}x; "
                   if "slowdown_vs_fp8_graphs" in s else "")
                + (f"kernel-time ratio {kernel_ratio[0]:.2f}x (Pearl-C's GPU ms a step over stock FP8's, {kernel_ratio[1]}); "
                   if kernel_ratio and phase == "decode" else "")
                + interleaving(summary))
        cmd = [sys.executable, panel, "append", "--line", "pearl-c-sm120", "--version", version, "--precision", "fp8",
               "--phase", phase, "--shape", s["shape"], "--kind", "measured", "--slowdown", f"{s['slowdown_vs_fp8']:.4f}",
               "--hash-free", f"{s['nohash_s'] / s['fp8_s']:.4f}", "--change", "kernel", "--description", desc,
               "--source", run, "--run", run, "--verifier-commit", verdicts[pass_]["verifier_commit"],
               "--verifier-accept", verdicts[pass_]["line"], "--transcript", verdicts[pass_]["transcript"],
               "--negative-control", verdicts["control"]["line"], "--sm-clock-arm", clocks["pearlc"],
               "--sm-clock-base", clocks["fp8"], "--by", BY]
        if phase == "decode":
            cmd += ["--decode-method", "dependent-chain"]
            if "slowdown_vs_fp8_graphs" in s and "fp8-graphs" in res:
                out.append(headline_over_graphed(cmd, s, res, reps, desc))
                cmd[cmd.index("--shape") + 1] = s["shape"] + EAGER_SUFFIX
        out.append(cmd)
    return out


def headline_over_graphed(eager: list[str], s: dict, res: dict, reps: str, desc: str) -> list[str]:
    """Decode's headline row: the eager row's command over the graphed baseline instead, its slowdown, hash-free ratio and
    base clocks the ``fp8-graphs`` reps'."""
    cmd = list(eager)
    for flag, value in (("--slowdown", f"{s['slowdown_vs_fp8_graphs']:.4f}"),
                        ("--hash-free", f"{s['nohash_s'] / s['fp8_graphs_s']:.4f}"),
                        ("--sm-clock-base", ",".join(str(r["clock"]["sm_mhz"]) for r in res["fp8-graphs"]["reps"][reps])),
                        ("--description", "Headline, over stock FP8 as vLLM serves it (compiled, CUDA graphs), both arms under "
                                          f"CUDA graphs; the row over eager stock FP8 is beside it at {s['shape']}{EAGER_SUFFIX}. "
                                          + desc)):
        cmd[cmd.index(flag) + 1] = value
    return cmd


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("e2e", type=Path)
    p.add_argument("verify_dir", type=Path)
    p.add_argument("--panel", default=PANEL)
    p.add_argument("--append", action="store_true")
    p.add_argument("--kernel-ratio", type=float, help="decode's kernel-time ratio (served_profile.py's GPU ms a step, Pearl-C over "
                   "stock FP8), published beside the eager ratio until both arms run with CUDA graphs")
    p.add_argument("--kernel-ratio-run", help="the profile run the kernel-time ratio comes from")
    a = p.parse_args(argv)
    if (a.kernel_ratio is None) != (a.kernel_ratio_run is None):
        raise SystemExit("--kernel-ratio and --kernel-ratio-run go together")
    verdicts = {x: json.loads((a.verify_dir / f"{x}-verify.json").read_text()) for x in ("prefill", "decode", "control", "control-leaves")
                if x != "control-leaves" or (a.verify_dir / f"{x}-verify.json").exists()}
    first, *rest = rows(json.loads(a.e2e.read_text()), verdicts, a.panel,
                        (a.kernel_ratio, a.kernel_ratio_run) if a.kernel_ratio is not None else None)
    if not a.append:
        print(shlex.join(first))
        for cmd in rest:
            print(shlex.join([*cmd, "--attempt", "<the prefill row's attempt>"]))
        return 0
    out = subprocess.run(first, check=True, capture_output=True, text=True).stdout
    print(out, end="")
    attempt = re.search(r"appended attempt (\d+)", out)
    if not attempt:
        raise SystemExit("panel.py did not say which attempt it appended; append the decode rows with --attempt by hand")
    for cmd in rest:
        subprocess.run([*cmd, "--attempt", attempt.group(1)], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
