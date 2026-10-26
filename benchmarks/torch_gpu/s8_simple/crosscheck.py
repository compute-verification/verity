"""On one CUDA device: KP2-S8's `wgmma.linear` (HASH-10, `benchmarks/torch_gpu/s8_linear/`) against `s8_simple.linear`,
word for word, a second witness that both compute `ScaledMmS8Coordinate_v1{K, GEMM_SM90_S8}`. JSON goes to
``--out``/crosscheck.json and a summary to ``--out``/summary.txt.

The inputs are `check.py weights` in ``--scratch``: the served set is layer 0 of the demo's weights and the head. At each
linear and each of ``--ms``, x and its scales are `check.py gpu`'s for that (set, linear, m), special rows included.
Every word of y [m, n] from `s8_simple.linear` (cuBLASLt's s32 sums, the fused epilogue) is compared with two wgmma arms:

* ``default``: `wgmma.linear`, the API, at `wgmma.default_cfg`;
* ``tuned``: `wgmma.scaled_mm` at the config `configs.CONFIGS` picks for (n, k, m) (`run.table_cfg`'s rule), the table
  the skeleton's timings ran.

Each arm's mismatches are counted, with the reference words' classes, and each is timed with CUDA events. The cubin is
$S8_WGMMA_CUBIN, and `Lib.identity` holds it to the SASS gate and MANIFEST beside it; the exit is 1 unless every word
matches and the identity is ok.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parent / "s8_linear"))  # after sha512_rows': both directories have a bench.py

import check as C  # noqa: E402
import s8_simple as S  # noqa: E402

MS = (4096, 16384)
ARMS = ("default", "tuned")


def tuned_cfg(m: int, n: int, k: int):
    """`run.table_cfg` for the wgmma kernel on this card: the first (m_max, config) of (n, k) with m <= m_max, else the
    last; None where the table has no row."""
    import torch
    import wgmma as W
    from configs import CONFIGS
    major, minor = torch.cuda.get_device_capability()
    rows = CONFIGS.get(f"wgmma:sm_{major}{minor}", {}).get((n, k))
    if not rows:
        return None
    return W.Cfg.parse(next((text for m_max, text in rows if m <= m_max), rows[-1][1]))


def arm(name: str):
    """A wgmma arm as a linear: (x, sx, w, sw) -> y bf16, and its config for (m, n, k)."""
    import torch
    import wgmma as W

    def cfg(m: int, n: int, k: int):
        return (tuned_cfg(m, n, k) if name == "tuned" else None) or W.default_cfg(m, n, k, W.lib().sms)

    if name == "default":
        return W.linear, cfg

    def linear(x, sx, w, sw):
        y = torch.empty((x.shape[0], w.shape[0]), dtype=torch.bfloat16, device=x.device)
        W.scaled_mm(x, sx, w, sw, y, cfg(x.shape[0], w.shape[0], x.shape[1]))
        return y
    return linear, cfg


def word_classes(torch, y) -> dict[str, int]:
    v = y.view(torch.int16)
    e, mant = v & 0x7F80, v & 0x007F
    return {"nan": int(((e == 0x7F80) & (mant != 0)).sum()), "inf": int(((e == 0x7F80) & (mant == 0)).sum()),
            "zero": int(((v & 0x7FFF) == 0).sum()), "subnormal": int(((e == 0) & (mant != 0)).sum())}


def compare(torch, ref, got, limit: int = 8) -> dict[str, Any]:
    a, b = ref.view(torch.int16), got.view(torch.int16)
    bad = a != b
    n = int(bad.sum())
    out: dict[str, Any] = {"mismatches": n}
    if n:
        idx = bad.nonzero()[:limit].tolist()
        out["first"] = [{"i": i, "j": j, "ref": int(a[i, j]) & 0xFFFF, "got": int(b[i, j]) & 0xFFFF} for i, j in idx]
        out["rows_bad"] = int(bad.any(1).sum())
    return out


def run(a: argparse.Namespace) -> int:
    import torch
    import wgmma as W
    dev = torch.device("cuda")
    scratch, out = Path(a.scratch), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    made = json.loads((scratch / "weights.json").read_text())
    ident = W.lib().identity()
    arms = {name: arm(name) for name in ARMS}
    env = C._env(torch, dev) | {"triton": __import__("triton").__version__, "cubin": ident}
    runs = []
    for name, (n, k) in a.shapes.items():
        meta = made[f"{a.set}/{name}"]
        w = torch.from_numpy(np.load(scratch / meta["w"])).to(dev)
        sw = torch.from_numpy(np.load(scratch / meta["sw"])).to(dev)
        for m in a.ms:
            rng = np.random.default_rng(C.seed_of("x", a.set, name, m))
            g = torch.Generator(device=dev)
            g.manual_seed(int(rng.integers(1 << 62)))
            x = torch.randint(0, 256, (m, k), dtype=torch.uint8, device=dev, generator=g).view(torch.int8)
            sx_np = C.scales(rng, m, C.TOKEN_SPREAD)
            t = min(m, len(C.SPECIALS))
            codes, sc = C.specials(t, k, rng, C.TOKEN_SPREAD)
            x[:t], sx_np[:t] = torch.from_numpy(codes).to(dev), sc
            sx = torch.from_numpy(sx_np).to(dev)
            ref = S.linear(x, sx, w, sw)
            r: dict[str, Any] = {"set": a.set, "shape": name, "n": n, "k": k, "m": m, "w_sha256": meta["w_sha256"],
                                 "words": m * n, "classes": word_classes(torch, ref)}
            r["simple_ms"] = round(C._timed(torch, lambda: S.linear(x, sx, w, sw), a.warm, a.reps)[0], 4)
            for label, (fn, cfg) in arms.items():
                got = fn(x, sx, w, sw)
                r[label] = {"cfg": str(cfg(m, n, k)), **compare(torch, ref, got)}
                ms, _, last = C._timed(torch, lambda: fn(x, sx, w, sw), a.warm, a.reps)
                r[label] |= {"ms": round(ms, 4), "repeat_equal": bool(torch.equal(last.view(torch.int16), got.view(torch.int16)))}
                del got, last
            runs.append(r)
            print(json.dumps(r), flush=True)
            del x, sx, ref
        del w, sw
        torch.cuda.empty_cache()
    ok = ident.get("ok") is True and all(r[arm_]["mismatches"] == 0 and r[arm_]["repeat_equal"] for r in runs for arm_ in ARMS)
    doc = {"env": env, "runs": runs, "layers": layers(runs), "pass": ok}
    (out / "crosscheck.json").write_text(json.dumps(doc, indent=1) + "\n")
    (out / "summary.txt").write_text(summary(doc))
    return 0 if ok else 1


def layers(runs: list[dict]) -> list[dict]:
    """Each m's times summed over `check.LAYER`'s linears, where every one of them ran."""
    out = []
    for m in sorted({r["m"] for r in runs}):
        rs = {r["shape"]: r for r in runs if r["m"] == m}
        if all(s in rs for s in C.LAYER):
            out.append({"m": m, "simple_ms": sum(rs[s]["simple_ms"] for s in C.LAYER),
                        **{f"{arm_}_ms": sum(rs[s][arm_]["ms"] for s in C.LAYER) for arm_ in ARMS}})
    return out


def summary(doc: dict) -> str:
    c = doc["env"]["cubin"]
    lines = [f"{doc['env'].get('name')} cubin {c.get('sha256', '?')[:16]} identity ok {c.get('ok')} "
             f"(SASS problems {c.get('sass_problems')}, manifest {c.get('manifest')})"]
    for r in doc["runs"]:
        lines.append(f"{r['set']}/{r['shape']:5s} ({r['n']}, {r['k']}) m {r['m']:6d}: {r['words']:,} words "
                     f"(nan {r['classes']['nan']:,}, inf {r['classes']['inf']:,}, zero {r['classes']['zero']:,}, "
                     f"subnormal {r['classes']['subnormal']:,}); simple {r['simple_ms']:.3f} ms; "
                     + "; ".join(f"{arm_} {r[arm_]['cfg']} {r[arm_]['mismatches']} mismatches, {r[arm_]['ms']:.3f} ms"
                                 for arm_ in ARMS))
    for v in doc["layers"]:
        lines.append(f"layer m {v['m']}: simple {v['simple_ms']:.3f} ms, wgmma default {v['default_ms']:.3f}, "
                     f"tuned {v['tuned_ms']:.3f}")
    lines.append(f"pass: {doc['pass']}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="served", choices=C.SETS)
    ap.add_argument("--shapes", type=C._shapes, default=dict(C.SHAPES), help="names of check.SHAPES, or NAME=N:K")
    ap.add_argument("--ms", type=lambda s: [int(v) for v in s.split(",")], default=list(MS))
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--reps", type=int, default=5)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
