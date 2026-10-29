"""Pearl-C's served device path on one H100 or H200 at GLM-5.3's in-domain linears (`glm53.linears`, one TP8 rank): the gate and
three retained passes, then the reference replay of each.

  python benchmarks/pouw/pearl_c_sm90/glm53_passes.py capture --out DIR [--cubin C] [--schedule S] [--graphs]  # on the card
  python benchmarks/pouw/pearl_c_sm90/glm53_passes.py verify --out DIR [--jobs J] [--fast]  # on CPUs, after
  python benchmarks/pouw/pearl_c_sm90/glm53_passes.py full --out DIR [--jobs J] [--without-y] [--passes P ...]  # on CPUs, after

``capture`` installs `pouw_pearl_c_device.PearlCDeviceRun` (``pearl-c-h100-v1-h3`` on this directory's kernels, the serial
schedule or ``--schedule``, its decode calls replayed as CUDA graphs with ``--graphs``) on synthetic BF16 weights, one a distinct in-domain (n, k), under an epoch whose salt coin a challenger opened after
the weights' root (its coin commitment is published as ``coins.json``, where `verify_run` reads it), then:

- the gate (`bench_arm.PearlCArm.gate`): at each weight and m in 16 and 64, three fresh passes on poisoned outputs: y within
  `bench_arm.ERR` rel RMS of x·Wᵀ, the commitment twice bit for bit, and the tile leaves omitted changing the transcript root;
- ``prefill``: one call a weight at `PREFILL` rows, retained from outputs poisoned 0xA5, each call's served y beside it
  (``y/<call>.bf16``);
- ``decode``: one call a weight at `DECODE` rows (40 of them, so the m₁₆ filler and the 64-row tiles run), retained the same way;
- ``control``: decode's calls again, retained, with every commitment write omitted (`e2e.CONTROL_OMITS`), so the transcript is
  whatever the poison left.

``--shapes`` names the weights' (n, k) in place of the in-domain ones; one shape alone makes every tile verify draws that
shape's (a pass calls every weight the epoch commits, `audit.Epoch`'s rule).

``verify`` runs `verify_run.verify` on each pass ``capture.json`` lists (grouped_check.py's too) under ``coins.json``: every
pass but control must be accepted (replay-verified, development shortcut; the protocol's check is a ZK proof) and control
rejected.  ``full`` checks prefill's and decode's outputs (``--passes``) in full, every tile of every call rather than verify's samples: its
openings and its leaf against C~ and U recomputed from the opened rows on the device record (`clear_audit.Verifier.check_tile`,
the verifier's own), then every real row's served y against y_unit of those U and α (RN_f32(RN_f32(U·RN_f32(1/α_A))·RN_f32(1/α_B)),
then BF16 nearest-even, as `pearl_c.PearlC.useful` states it and fixture.py's y is), on `fast_tile`'s batch forms with every
``--cross-check``-th tile on the scalar reference too.  Each mode writes its JSON in DIR (``capture.json``, ``verify.json``,
``full.json``) and exits 0 only when everything it checks holds.
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
from verity_config.settings import of  # noqa: E402

settings = of(__file__)
PREFILL, DECODE = 1024, 40
PASSES = ("prefill", "decode", "control")
FULL = ("prefill", "decode")
EPOCH = 20261023


def capture(a) -> int:
    import bench_arm as BA
    import e2e
    import torch
    from verity.core.protocols.pouw.audit import weights_root
    from verity_vllm.protocols.pouw import pearl_c as PPD

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
                              salt_key=salt_key, schedule=a.schedule, graphs=a.graphs)
    arm = BA.PearlCArm()
    arm.run, arm.weights = run, weights
    log = {"scheme": a.scheme, "schedule": a.schedule, "graphs": run.graphs, "cubin_sha256": kernel.cubin_sha256, "kernel_source_sha256": kernel.source_sha256,
           "device": torch.cuda.get_device_name(dev), "shapes": shapes, "prefill": PREFILL, "decode": DECODE,
           "gate": arm.gate(None), "passes": {}}
    print(f"gate: {'PASS' if log['gate']['passed'] else 'FAIL'}", flush=True)
    xs = {m: {w: torch.randn(m, W.shape[1], device=dev, generator=gen).to(torch.bfloat16) for w, W in weights.items()}
          for m in (PREFILL, DECODE)}
    for name, m, omit in (("prefill", PREFILL, ()), ("decode", DECODE, ()), ("control", DECODE, e2e.CONTROL_OMITS)):
        run.begin(retain=str(out / name), run=settings.RESEARCH_RUN_ID or "glm53-passes")
        run.omit = frozenset(omit)
        rms = {}
        try:
            for i, (w, W) in enumerate(weights.items()):
                y = run.gemm(w, xs[m][w], W, None)
                if name in FULL:
                    (out / name / "y").mkdir(parents=True, exist_ok=True)
                    (out / name / "y" / f"{i}.bf16").write_bytes(y.contiguous().view(torch.int16).cpu().numpy().tobytes())
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
        v = VR.verify(out / name / "manifest.json", VR.DRAWS, a.jobs, settings.RESEARCH_SOURCE_SHA or "unknown",
                      fast=a.fast, cross_check=a.cross_check if a.fast else 0, coins=coins)
        res[name] = {k: v.get(k) for k in ("accepted", "line", "verdicts", "error")}
        print(f"{name}: {v['line']}", flush=True)
    res["ok"] = bool(names) and all(res[x]["accepted"] == (x != "control") for x in names)
    (out / "verify.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(f"verify {'PASS' if res['ok'] else 'FAIL'}", flush=True)
    return 0 if res["ok"] else 1


_FULL: dict = {}


def y_unit(u, alpha_a, alpha_b):
    """The served y of U's FP32 words under α_A (a row's) and α_B (a column's): RN_f32(RN_f32(U·RN_f32(1/α_A))·RN_f32(1/α_B)),
    rounded to BF16 nearest-even (numpy's float32 is IEEE's, as `fast_tile.install` holds the process to)."""
    import numpy as np

    def inv(alpha):
        return np.float32(1) / alpha.view(np.float32)
    w = ((u.view(np.float32) * inv(alpha_a)[:, None]) * inv(alpha_b)[None, :]).view(np.uint32).astype(np.uint64)
    return ((w + 0x7FFF + ((w >> 16) & 1)) >> 16).astype(np.uint16)


def full_tile(tid: int) -> tuple[int, str | None, int, bool | None]:
    """Tile `tid` of the pass in `_FULL`, and on every `cross_check`-th tile id whether the scalar reference gives the same:
    (tile, its failure or None, y words compared, the same or None where it was replayed once)."""
    got = _full_tile(tid)
    j = _FULL
    if j["undo"] is None or not j["every"] or tid % j["every"]:
        return (*got, None)
    j["undo"]()
    try:
        return (*got, _full_tile(tid) == got)
    finally:
        j["undo"] = j["install"](j["scheme"])


def _full_tile(tid: int) -> tuple[int, str | None, int]:
    """The verifier's `check_tile` (the openings, then C~ and U recomputed from the opened rows on the device record, against
    the tile's leaf), then each real row's served y against y_unit of those U and α (`pearl_c`'s `PearlC.useful`)."""
    import numpy as np

    j = _FULL
    verifier, view = j["verifier"], j["view"]
    ref = verifier.refs[tid]
    v = verifier.check_tile(j["transcript"], j["prover"].open(j["epoch"], tid))
    if not v.ok:
        return tid, v.reason, 0
    ops, checked = view.seen
    real = [p for p, i in enumerate(ref.tile.rows) if i < j["real"][ref.matmul]]
    if not real or j["y"] is None:
        return tid, None, 0
    want = y_unit(np.array(checked.u, np.uint32)[real], np.array(ops.a.alpha, np.uint32)[real], np.array(ops.b.alpha, np.uint32))
    got = j["y"][ref.matmul][np.ix_([ref.tile.rows[p] for p in real], list(ref.tile.cols))]
    if not np.array_equal(got, want):
        p, q = map(int, np.argwhere(got != want)[0])
        return tid, (f"served y differs at {int((got != want).sum())} of {got.size} words, first row {ref.tile.rows[real[p]]} "
                     f"column {ref.tile.cols[q]}: {int(got[p, q]):#06x}, y_unit {int(want[p, q]):#06x}"), got.size
    return tid, None, got.size


def full_pass(d: Path, coins, jobs: int, every: int, without_y: bool = False) -> dict:
    """Every tile of the retained pass in `d` through `full_tile`, its epoch re-started under `coins` as `verify_run` starts it;
    `without_y`, C~ and U alone (a pass captured before its served y was retained)."""
    import multiprocessing

    import verify_run as VR
    import clear_work as CW
    import fast_tile as FT
    import numpy as np
    from clear_audit import Transcript, Verifier
    from verity.ml.commitments import Commitment
    from verity_vllm.protocols.pouw import pearl_c as PD

    class Seen(CW._TileRuleView):
        """`verify_run`'s view of the scheme, keeping the operands and checked words of the digest it last took."""

        def checked(self, ctx, ops):
            self.seen = (ops, self._scheme.checked(ctx, ops))
            return self.seen[1]

    out: dict = {"pass": str(d)}
    try:
        man = json.loads((d / "manifest.json").read_text())
        PD.manifest_schema(man, False)
        scheme = PD.scheme_of(man["scheme"], **{k: tuple(v) if isinstance(v, list) else v for k, v in man.get("instance", {}).items()})
        epoch, _ = PD.manifest_epoch(man, coins, False)
        c = man["commitment"]
        transcript = Transcript(tuple(bytes.fromhex(u["root_a"]) for u in man["matmuls"]), Commitment(bytes.fromhex(c["root"]), c["count"]),
                                ())
        view = Seen(scheme)
        verifier = Verifier(view, epoch)
        if len(verifier.refs) != c["count"]:
            raise ValueError(f"the commitment counts {c['count']} tiles; the workload has {len(verifier.refs)}")
        ys = None if without_y else {}
        for pos, u in enumerate(man["matmuls"] if ys is not None else ()):
            f = d / "y" / f"{pos}.bf16"
            if not f.exists() or f.stat().st_size != 2 * u["real_rows"] * u["n"]:
                raise ValueError(f"call {pos}: no served y of {u['real_rows']} x {u['n']} BF16 words at {f}")
            ys[pos] = np.memmap(f, "<u2", "r", shape=(u["real_rows"], u["n"]))
        fast = scheme.forming == "v1" and scheme.device.g is not None
        _FULL.clear()
        _FULL.update(scheme=scheme, epoch=epoch, transcript=transcript, verifier=verifier, view=view, y=ys,
                     prover=VR.Opened(man, VR.Files(d), verifier.refs), real={pos: u["real_rows"] for pos, u in enumerate(man["matmuls"])},
                     every=every, install=FT.install, undo=FT.install(scheme) if fast else None)
        try:
            todo = range(c["count"])
            if jobs > 1 and len(todo) > 1:
                # fork: the workers unpickle `full_tile` from this module, which callers load from its path
                with multiprocessing.get_context("fork").Pool(min(jobs, len(todo))) as pool:
                    verdicts = pool.map(full_tile, todo, chunksize=max(1, len(todo) // (8 * jobs)))
            else:
                verdicts = [full_tile(t) for t in todo]
        finally:
            if _FULL.get("undo") is not None:
                _FULL.pop("undo")()
        failed = [{"tile": t, "reason": r} for t, r, _, _ in verdicts if r is not None]
        differ = [t for t, _, _, same in verdicts if same is False]
        shapes: dict[str, int] = {}
        for ref in verifier.refs:
            s = epoch.workload.matmuls[ref.matmul].shape
            shapes[f"{s.n}x{s.k}"] = shapes.get(f"{s.n}x{s.k}", 0) + 1
        words = sum(len(ref.tile.rows) * len(ref.tile.cols) for ref in verifier.refs)
        out.update(ok=not failed and not differ, tiles=len(verdicts), matmuls=len(man["matmuls"]), shapes=shapes,
                   words=words, y_words=sum(y for _, _, y, _ in verdicts), replay="batch" if fast else "scalar",
                   cross_checked=sum(same is not None for *_, same in verdicts), cross_check_failed=differ,
                   failed=len(failed), failures=failed[:20], transcript_root=c["root"])
        first = "; ".join(f"tile {x['tile']}: {x['reason']}" for x in failed[:3])
        y = "served y not retained" if ys is None else f"served y ({out['y_words']} words) as y_unit"
        out["line"] = (f"{'PASS' if out['ok'] else 'FAIL'}: every tile ({out['tiles']}) of {out['matmuls']} matmuls, (N, K) "
                       f"{', '.join(f'{s} ({t})' for s, t in shapes.items())}: C~ and U ({words} words each) by their leaves, "
                       f"{y}; {len(failed)} failed{f' ({first})' if failed else ''}; "
                       f"{out['cross_checked']} tiles cross-checked on the scalar reference"
                       f"{f', differing on {differ}' if differ else ''}; transcript root {c['root'][:16]}")
    except Exception as ex:  # noqa: BLE001 - a pass that can't be checked fails, with the reason
        out.update(ok=False, error=f"{type(ex).__name__}: {ex}", line=f"FAIL: {type(ex).__name__}: {ex}")
    return out


def full(a) -> int:
    from verity.core.protocols.pouw.coins import CoinCommitment

    out = Path(a.out)
    coins = CoinCommitment.from_json(json.loads((out / "coins.json").read_text()))
    res = {}
    for name in a.passes:
        res[name] = full_pass(out / name, coins, a.jobs, a.cross_check, a.without_y)
        print(f"{name}: {res[name]['line']}", flush=True)
    res["ok"] = bool(a.passes) and all(res[x]["ok"] for x in a.passes)
    (out / "full.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(f"full {'PASS' if res['ok'] else 'FAIL'}", flush=True)
    return 0 if res["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("mode", choices=("capture", "verify", "full"))
    p.add_argument("--out", required=True)
    p.add_argument("--scheme", default=glm53.SCHEME)
    p.add_argument("--cubin", help="the build's cubin (default this directory's)")
    p.add_argument("--shapes", nargs="*", help="capture: N:K, a weight each (default every in-domain linear)")
    p.add_argument("--schedule", default="serial", help="capture: the run's schedule (verity.core.protocols.pouw.serving.SCHEDULES)")
    p.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    p.add_argument("--fast", action="store_true", help="verify_run's --fast")
    p.add_argument("--cross-check", type=int, default=64, help="with --fast, and full: every N-th tile on the scalar reference too")
    p.add_argument("--without-y", action="store_true", help="full: C~ and U alone (passes captured before served y was retained)")
    p.add_argument("--passes", nargs="+", choices=FULL, default=list(FULL), help="full: the passes to check (default both)")
    p.add_argument("--graphs", action="store_true", help="capture: calls of up to 256 rows replayed as CUDA graphs")
    a = p.parse_args(argv)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    return {"capture": capture, "verify": verify, "full": full}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
