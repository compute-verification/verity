"""-h3's device screen (``pearl-c-sm120-v1-h3`` unless ``--scheme``; untimed, one GPU): the served device path's calls against
the reference.

  python benchmarks/pouw/pearl_c_vllm/h3_screen.py capture --kernel SHIP --out DIR [--scheme S]    # on the GPU
  python benchmarks/pouw/pearl_c_vllm/h3_screen.py check --out DIR [--jobs J]                        # on CPUs, after

``capture`` runs `PearlCDeviceRun` (the serial schedule) on synthetic BF16 weights, (256, 1024) and (128, 2048), at m = 20,
100 and 200, each pass retained from outputs poisoned 0xA5: eagerly (each call's ``rowdig_a``, ``seeds_a``, ``seed_a`` and
``ea_codes`` dumped beside the pass), under CUDA graphs (the unit index read from the call's key slot), eagerly with
``seeds_a`` omitted (the negative control: E_A's lines read the poison), and eagerly with ``rowdig_a`` omitted (the seeds read
poisoned digests).  ``check`` holds them to the reference (`pearl_c.PearlC`):

- each eager call's row digests are `row_digest`'s and its row seeds `row_seeds` over its m₁₆ committed rows, E_A's line of
  row i is ``sample_line(seed_i, 0, 0, 0)``, and each padding row r's is ``sample_line(seed_A, 0, 0, r)``;
- every tile of each pass is `Verifier.check_tile`'s: openings, then the leaf recomputed by the reference (E_A per row from
  the row's own seed), so the honest passes' leaves are the reference's bit for bit and the control's are not;
- and `verify_run`'s audit of each pass, recorded (its tile rule includes -h3's row cap, which an honest row can exceed).

Writes ``screen.json`` in DIR, and exits 0 only when every eager call matches, every honest tile checks and each control
fails some tile.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
from pathlib import Path

from verity_config.settings import of

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "clear"))
settings = of(__file__)

SCHEME = "pearl-c-sm120-v1-h3"
WEIGHTS = {0: (256, 1024), 1: (128, 2048)}
CALLS = ((0, 20), (1, 100), (0, 200))
PASSES = {"eager": (False, ()), "graphs": (True, ()), "omit-seeds": (False, ("seeds_a",)), "omit-rowdig": (False, ("rowdig_a",))}


def capture(args) -> int:
    import torch
    from verity.core.protocols.pouw.audit import weights_root
    from verity.core.protocols.pouw import verifier
    from verity.core.protocols.pouw.coins import Challenger
    from verity_vllm.protocols.pouw import pearl_c as PPD

    out, dev = Path(args.out), torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(20261006)
    weights = {w: (0.05 * torch.randn(n, k, device=dev, generator=gen)).to(torch.bfloat16) for w, (n, k) in WEIGHTS.items()}
    bad = PPD.refusals(args.scheme, {w: tuple(t.shape) for w, t in weights.items()})
    if bad:
        raise SystemExit(f"{args.scheme} refuses the screen's weights: " + "; ".join(bad))
    kernel = PPD.Kernel.load(args.kernel, args.cubin)
    salt_key = PPD.new_salt_key(args.scheme)
    roots = PPD.weight_roots(kernel, weights, args.scheme, salt_key=salt_key)
    challenger = Challenger(0, verifier.coins)
    opening = challenger.open("salt", weights_root(roots))
    xs = [torch.randn(m, weights[w].shape[1], device=dev, generator=gen).to(torch.bfloat16) for w, m in CALLS]
    s, log = PPD.scheme_of(args.scheme), {}
    dumps = ("seeds_a", "seed_a", "ea_codes", "rowdig_a")
    for name, (graphs, omit) in PASSES.items():
        run = PPD.PearlCDeviceRun(s, f"pouw:{args.scheme}", kernel, weights, roots, challenger.commitment, opening, graphs=graphs,
                                  salt_key=salt_key)
        d = out / name
        try:
            run.begin(retain=str(d), run=settings.RESEARCH_RUN_ID or "h3-screen")
            run.omit = frozenset(omit)
            for index, ((w, m), x) in enumerate(zip(CALLS, xs)):
                y = run.gemm(w, x, weights[w], None)
                torch.cuda.synchronize()
                if not graphs:
                    W = weights[w]
                    sh = run._shape(-(-m // PPD.ROW_ALIGN) * PPD.ROW_ALIGN, W.shape[0], W.shape[1])
                    for b in dumps:
                        (d / f"call{index}.{b}").write_bytes(PPD._bytes(sh.b[b])[:sh.size[b]])
                ref = x.float() @ weights[w].float().T
                log.setdefault(name, []).append({"call": index, "rel_rms": float(((y.float() - ref).pow(2).sum() / ref.pow(2).sum()).sqrt())})
            log[name + ":end"] = {k: v for k, v in run.end().items() if k != "manifest"}
        except Exception as ex:  # noqa: BLE001 - a pass that can't run is recorded, and check fails on its missing manifest
            log[name + ":error"] = f"{type(ex).__name__}: {ex}"
        finally:
            run.omit = frozenset()
    (out / "capture.json").write_text(json.dumps({"scheme": args.scheme, "cubin_sha256": kernel.cubin_sha256,
                                                  "calls": CALLS, "weights": WEIGHTS, "passes": log}, indent=1) + "\n")
    print(json.dumps(log), flush=True)
    return 0 if not any(k.endswith(":error") for k in log) else 1


def _scheme(man: dict):
    from verity_vllm.protocols.pouw import pearl_c as PPD
    return PPD.scheme_of(man["scheme"], **{k: tuple(v) if isinstance(v, list) else v for k, v in man.get("instance", {}).items()})


def _calls(d: Path) -> dict:
    """An eager pass's calls held to the reference: the row seeds, E_A's committed rows and its padding rows."""
    import verify_run as VR
    from verity.core.protocols.pouw.protocol import Context, Matmul, Shape
    from verity.ml.protocols.pouw.schemes import pearl_c as C

    man = json.loads((d / "manifest.json").read_text())
    s, files, out = _scheme(man), VR.Files(d), []
    for u in man["matmuls"]:
        wt = man["weights"][str(u["weight"])]
        ctx = Context(bytes.fromhex(man["salt"]), Matmul(u["index"], Shape(u["m"], u["k"], u["n"]), u["weight"]),
                      bytes.fromhex(u["root_a"]), bytes.fromhex(wt["root"]))
        data = files(u["rows"])
        rows = [VR.f32_row(data, u["k"], i) for i in range(u["m"])]
        seeds = s.row_seeds(ctx, rows, range(u["m"]))
        got = (d / f"call{u['index']}.seeds_a").read_bytes()
        seed_a, ea = (d / f"call{u['index']}.seed_a").read_bytes()[:32], (d / f"call{u['index']}.ea_codes").read_bytes()
        lines = [ea[32 * r:32 * r + 32] for r in range(len(ea) // 32)]
        seed_a = C.S512Seed(seed_a)
        rows_ok = [lines[i] == bytes(C.sample_line(x, 0, 0, 0)) for i, x in enumerate(seeds)]
        pad_ok = [lines[r] == bytes(C.sample_line(seed_a, 0, 0, r)) for r in range(u["m"], len(lines))]
        dig = d / f"call{u['index']}.rowdig_a"
        digests = dig.read_bytes()[:64 * u["m"]] == b"".join(C.row_digest(x) for x in rows)
        out.append({"call": u["index"], "m16": u["m"], "padded": len(lines), "seed_a": seed_a == s.noise(ctx).seed_a,
                    "row_digests": digests, "row_seeds": got[:32 * u["m"]] == b"".join(seeds), "ea_rows": all(rows_ok),
                    "ea_rows_wrong": [i for i, ok in enumerate(rows_ok) if not ok][:8], "ea_padding": all(pad_ok)})
    return {"calls": out, "ok": all(c["seed_a"] and c["row_digests"] and c["row_seeds"] and c["ea_rows"] and c["ea_padding"]
                                    for c in out)}


_JOB: dict = {}


def _tile(tid: int) -> tuple[int, bool, str | None]:
    j = _JOB
    v = j["verifier"].check_tile(j["transcript"], j["prover"].open(j["epoch"], tid))
    return tid, v.ok, v.reason


def _tiles(d: Path, jobs: int) -> dict:
    """Every tile of a pass under `Verifier.check_tile`: its openings, then its leaf recomputed by the reference."""
    import verify_run as VR
    from verity.ml.commitments import Commitment
    from clear_audit import Transcript, Verifier
    from verity.core.protocols.pouw.coins import CoinCommitment
    import clear_work as CW
    from verity_vllm.protocols.pouw.pearl_c import manifest as PM

    man = json.loads((d / "manifest.json").read_text())
    s = _scheme(man)
    epoch, _ = PM.manifest_epoch(man, CoinCommitment.from_json(man["coins"]))
    c = man["commitment"]
    transcript = Transcript(tuple(bytes.fromhex(u["root_a"]) for u in man["matmuls"]), Commitment(bytes.fromhex(c["root"]), c["count"]), ())
    verifier = Verifier(CW._TileRuleView(s), epoch)
    _JOB.update(verifier=verifier, transcript=transcript, epoch=epoch, prover=VR.Opened(man, VR.Files(d), verifier.refs))
    ids = range(c["count"])
    if jobs > 1:
        with multiprocessing.get_context("fork").Pool(min(jobs, c["count"])) as pool:
            got = pool.map(_tile, ids, chunksize=1)
    else:
        got = [_tile(t) for t in ids]
    bad = [{"tile": t, "reason": r} for t, ok, r in got if not ok]
    return {"tiles": c["count"], "failed": len(bad), "failures": bad[:8]}


def check(args) -> int:
    import verify_run as VR
    out = Path(args.out)
    res: dict = {"capture": json.loads((out / "capture.json").read_text()) if (out / "capture.json").exists() else None}
    for name, (graphs, omit) in PASSES.items():
        d = out / name
        if not (d / "manifest.json").exists():
            res[name] = {"error": "no manifest"}
            continue
        r = {"tiles": _tiles(d, args.jobs)}
        if not graphs and not omit:
            r["calls"] = _calls(d)
        v = VR.verify(d / "manifest.json", VR.DRAWS, args.jobs, settings.RESEARCH_SOURCE_SHA or "unknown")
        r["verify_run"] = {k: v.get(k) for k in ("accepted", "line", "verdicts", "error")}
        res[name] = r
        print(f"{name}: {r['tiles']['tiles']} tiles, {r['tiles']['failed']} failed check_tile; "
              f"{'calls ' + str(r['calls']['ok']) + '; ' if 'calls' in r else ''}{v['line']}", flush=True)
    honest = [res.get(n, {}) for n in ("eager", "graphs")]
    controls = [res.get(n, {}) for n, (_, omit) in PASSES.items() if omit]
    res["ok"] = (all("tiles" in r and r["tiles"]["failed"] == 0 for r in honest) and res.get("eager", {}).get("calls", {}).get("ok", False)
                 and all("tiles" in c and c["tiles"]["failed"] > 0 for c in controls))
    (out / "screen.json").write_text(json.dumps(res, indent=1) + "\n")
    print(f"h3 screen {'PASS' if res['ok'] else 'FAIL'}", flush=True)
    return 0 if res["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("mode", choices=("capture", "check"))
    p.add_argument("--out", required=True)
    p.add_argument("--scheme", default=SCHEME)
    p.add_argument("--kernel", default=str(HERE.parent / "pearl_c_sm120"), help="the ship (run.py, fixture.py, the cubin)")
    p.add_argument("--cubin")
    p.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    a = p.parse_args(argv)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    return capture(a) if a.mode == "capture" else check(a)


if __name__ == "__main__":
    sys.exit(main())
