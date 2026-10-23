"""Pearl-C's served device path on one H100 or H200 at GLM-5.3's in-domain linears (`glm53.linears`, one TP8 rank): the gate and
three retained passes, then the reference replay of each.

  python benchmarks/pouw/pearl_c_sm90/glm53_passes.py capture --out DIR [--cubin C]       # on the card
  python benchmarks/pouw/pearl_c_sm90/glm53_passes.py verify --out DIR [--jobs J] [--fast]  # on CPUs, after

``capture`` installs `pouw_pearl_c_device.PearlCDeviceRun` (``pearl-c-h100-v1-h3`` on this directory's kernels, the serial
schedule) on synthetic BF16 weights, one a distinct in-domain (n, k), under an epoch whose salt coin a challenger opened after
the weights' root (its coin commitment is published as ``coins.json``, where `verify_run` reads it), then:

- the gate (`bench_arm.PearlCArm.gate`): at each weight and m in 16 and 64, three fresh passes on poisoned outputs: y within
  `bench_arm.ERR` rel RMS of x·Wᵀ, the commitment twice bit for bit, and the tile leaves omitted changing the transcript root;
- ``prefill``: one call a weight at `PREFILL` rows, retained from outputs poisoned 0xA5;
- ``decode``: one call a weight at `DECODE` rows (40 of them, so the m₁₆ filler and the 64-row tiles run), retained the same way;
- ``control``: decode's calls again, retained, with every commitment write omitted (`e2e.CONTROL_OMITS`), so the transcript is
  whatever the poison left.

``--shapes`` names the weights' (n, k) in place of the in-domain ones; one shape alone makes every tile verify draws that
shape's (a pass calls every weight the epoch commits, `audit.Epoch`'s rule).

``verify`` runs `verify_run.verify` on each pass ``capture.json`` lists (grouped_check.py's too) under ``coins.json``: every
pass but control must be accepted (replay-verified, development shortcut; the protocol's check is a ZK proof) and control
rejected.  Each mode writes its JSON in DIR
(``capture.json``, ``verify.json``) and exits 0 only when everything it checks holds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VLLM_DIR = HERE.parent / "pearl_c_vllm"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(VLLM_DIR))

import glm53  # noqa: E402

PREFILL, DECODE = 1024, 40
PASSES = ("prefill", "decode", "control")
EPOCH = 20261023


def capture(a) -> int:
    import bench_arm as BA
    import e2e
    import torch
    from verity.core.protocols.pouw.audit import weights_root
    from verity_vllm.protocol_options import pouw_pearl_c_device as PPD

    out, dev = Path(a.out), torch.device("cuda")
    shapes = [tuple(map(int, s.split(":"))) for s in a.shapes] if a.shapes else sorted(glm53.in_domain(a.scheme))
    gen = torch.Generator(device=dev).manual_seed(20261023)
    weights = {w: (0.05 * torch.randn(n, k, device=dev, generator=gen)).to(torch.bfloat16) for w, (n, k) in enumerate(shapes)}
    kernel = PPD.Kernel.load(str(HERE), a.cubin)
    salt_key = PPD.new_salt_key(a.scheme)
    roots = PPD.weight_roots(kernel, weights, a.scheme, salt_key=salt_key)
    coins = BA._Coins(EPOCH)
    opening = coins.open("salt", weights_root(roots))
    (out / "coins.json").write_text(json.dumps(coins.commitment.to_json()) + "\n")
    run = PPD.PearlCDeviceRun(PPD.scheme_of(a.scheme), f"pouw:{a.scheme}", kernel, weights, roots, coins.commitment, opening,
                              salt_key=salt_key)
    arm = BA.PearlCArm()
    arm.run, arm.weights = run, weights
    log = {"scheme": a.scheme, "cubin_sha256": kernel.cubin_sha256, "kernel_source_sha256": kernel.source_sha256,
           "device": torch.cuda.get_device_name(dev), "shapes": shapes, "prefill": PREFILL, "decode": DECODE,
           "gate": arm.gate(None), "passes": {}}
    print(f"gate: {'PASS' if log['gate']['passed'] else 'FAIL'}", flush=True)
    xs = {m: {w: torch.randn(m, W.shape[1], device=dev, generator=gen).to(torch.bfloat16) for w, W in weights.items()}
          for m in (PREFILL, DECODE)}
    for name, m, omit in (("prefill", PREFILL, ()), ("decode", DECODE, ()), ("control", DECODE, e2e.CONTROL_OMITS)):
        run.begin(retain=str(out / name), run=os.environ.get("RESEARCH_RUN_ID", "glm53-passes"))
        run.omit = frozenset(omit)
        rms = {}
        try:
            for w, W in weights.items():
                y = run.gemm(w, xs[m][w], W, None)
                ref = xs[m][w].float() @ W.float().T
                rms[w] = float(((y.float() - ref).pow(2).sum() / ref.pow(2).sum()).sqrt())
            log["passes"][name] = {"m": m, "omit": sorted(omit), "rel_rms": rms,
                                   **{k: v for k, v in run.end().items() if k != "manifest"}}
        finally:
            run.omit = frozenset()
        print(f"{name}: {log['passes'][name]['tiles']} tiles, root {log['passes'][name]['transcript_root'][:16]}", flush=True)
    honest = [log["passes"][p] for p in ("prefill", "decode")]
    log["ok"] = log["gate"]["passed"] and all(max(p["rel_rms"].values()) < BA.ERR for p in honest)
    (out / "capture.json").write_text(json.dumps(log, indent=1) + "\n")
    print(f"capture {'PASS' if log['ok'] else 'FAIL'}", flush=True)
    return 0 if log["ok"] else 1


def verify(a) -> int:
    import verify_run as VR
    from verity.core.protocols.pouw.coins import CoinCommitment

    out = Path(a.out)
    coins = CoinCommitment.from_json(json.loads((out / "coins.json").read_text()))
    names = list(json.loads((out / "capture.json").read_text())["passes"]) if (out / "capture.json").exists() else PASSES
    res = {}
    for name in names:
        v = VR.verify(out / name / "manifest.json", VR.DRAWS, a.jobs, os.environ.get("RESEARCH_SOURCE_SHA", "unknown"),
                      fast=a.fast, cross_check=a.cross_check if a.fast else 0, coins=coins)
        res[name] = {k: v.get(k) for k in ("accepted", "line", "verdicts", "error")}
        print(f"{name}: {v['line']}", flush=True)
    res["ok"] = bool(names) and all(res[x]["accepted"] == (x != "control") for x in names)
    (out / "verify.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(f"verify {'PASS' if res['ok'] else 'FAIL'}", flush=True)
    return 0 if res["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("mode", choices=("capture", "verify"))
    p.add_argument("--out", required=True)
    p.add_argument("--scheme", default=glm53.SCHEME)
    p.add_argument("--cubin", help="the build's cubin (default this directory's)")
    p.add_argument("--shapes", nargs="*", help="capture: N:K, a weight each (default every in-domain linear)")
    p.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    p.add_argument("--fast", action="store_true", help="verify_run's --fast")
    p.add_argument("--cross-check", type=int, default=64, help="with --fast: every N-th tile on the scalar reference too")
    a = p.parse_args(argv)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    return capture(a) if a.mode == "capture" else verify(a)


if __name__ == "__main__":
    sys.exit(main())
