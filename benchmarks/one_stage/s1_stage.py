"""S1's torch-loop chain staged under C-Flock's `UnitStager` (CIR-100): each W8A8 linear one `ScaledMmS8_v1{K=128,N=128}` unit,
each row quantization one `QuantRowS8_v1{N=128,RELU}` unit (#4433), one hm96-sha512/row/v2 row a port and one output row a return
field. Prints each stager's ANDs, rows, k_log and any refusal, and with ``chain`` stages one token of the torch loop's model
(seed 0) through layer 0 -> QuantRowS8{RELU} -> layer 1 -> QuantRowS8{no RELU}, every port reading a whole output row of its writer
word for word, checked against the loop's own torch values. A run input, not a tool:

    PYTHONPATH=.:verity/ml/flock/python:integrations/torch_loop python benchmarks/one_stage/s1_stage.py UNIT... --out DIR

UNIT: mm (one row per port), mm-wrows (w as 128 rows), mm-wswrows (w and sw as 128 rows each), coord (main's per-channel
`ScaledMmS8Coordinate_v1{K=128}`), q-relu, q-norelu, chain.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path

import numpy as np

H = 128


def say(phase: str, **kw) -> None:
    print("S1 " + json.dumps({"phase": phase, "rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2), **kw},
                             default=str), flush=True)


def defs():
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops import gemm as G
    from verity.ml.circuits.ops import quant as Q

    return {"mm": bind(G.ScaledMmS8, K=H, N=H, **G.GEMM_SM120_S8), "coord": bind(G.ScaledMmS8Coordinate, K=H, **G.GEMM_SM120_S8),
            "q-relu": bind(Q.QuantRowS8, N=H, RELU=True), "q-norelu": bind(Q.QuantRowS8, N=H, RELU=False)}


def rows_of(fn, split: tuple[str, ...] = ()):
    """One V2 row per parameter, or per row of each parameter in ``split``."""
    from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2

    params, _ret = fn.signature()
    out = []
    for nm, t in params:
        if nm in split:
            out += [(f"{nm}{r}", V2, sum(t.elem.leaf_widths()), [f"{nm}[{r}]"]) for r in range(t.n)]
        else:
            out.append((nm, V2, sum(t.leaf_widths()), [nm]))
    return out


def build(unit: str, fn, rows, out: Path):
    from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2
    from verity_flock import class_statement as CS

    ret = fn.signature()[1]
    n_out = len(ret.fields) if hasattr(ret, "fields") else 1
    out_rows = [(V2, (j,)) for j in range(n_out)]
    say("bound", unit=unit, definition=fn.id, rows=len(rows), row_bytes={r: b // 8 for r, _s, b, _p in rows[:4]},
        row_bytes_rest=sorted({b // 8 for _r, _s, b, _p in rows[4:]}), outputs=n_out)
    t = time.time()
    was = gc.isenabled()
    gc.disable()
    try:
        st = CS.UnitStager(fn, f"s1/{fn.id.split('{')[0]}", rows, out_rows, circuit=out / unit / "circuit.txt")
    except Exception as e:  # noqa: BLE001
        say("refused", unit=unit, seconds=round(time.time() - t, 1), error=f"{type(e).__name__}: {e}",
            where=traceback.format_exc().strip().splitlines()[-3:])
        return None
    finally:
        if was:
            gc.enable()
    m, c = st.meta, getattr(st.low, "counts", None) or st.low.layout.counts
    say("built", unit=unit, seconds=round(time.time() - t, 1), ands=c["ands"], unit_rows=c.get("rows"),
        nnz=c.get("nnz_a", 0) + c.get("nnz_b", 0), k_log=m["k_log"], unit_log=m["unit_log"], vus_per_block=m["vus_per_block"],
        lookups=m.get("per_vu", {}), out_words=st.out_words, ports=len(st.row_ports), pin=st.pin,
        circuit_bytes=(out / unit / "circuit.txt").stat().st_size)
    return st


def packed(a: np.ndarray, bits: int) -> np.ndarray:
    """Elements of ``bits`` as a row holds them: their little-endian bytes in u16 words."""
    return np.ascontiguousarray(np.asarray(a).reshape(-1).astype(np.dtype(f"<u{bits // 8}"))).view("<u2")


def stage(unit: str, st, vals: dict, out: Path) -> dict:
    t = time.time()
    got = st.stage({k: np.asarray(v, dtype=np.uint16).reshape(1, -1) for k, v in vals.items()}, out / unit / "round", os.urandom(32))
    say("staged", unit=unit, seconds=round(time.time() - t, 1), ands=got["ands"],
        out_words={j: int(np.asarray(o).shape[-1]) for j, o in got["outs"].items()})
    return {j: np.asarray(o, dtype=np.uint16).reshape(-1) for j, o in got["outs"].items()}


def random_vals(rows, rng) -> dict:
    return {r: rng.integers(0, 1 << 16, b // 16, dtype=np.uint16) for r, _s, b, _p in rows}


def chain(D: dict, out: Path, tok: int) -> bool:
    """One token of the torch loop's model through both linears and both quantizations as units, each port the writer's row."""
    import torch
    from verity_torch_loop import loop as TL

    model = TL.Model()
    x0 = model.embed[tok][None, :]
    xq, sx = TL.quantize(x0)
    L0, L1 = model.layers
    y0 = TL.linear(xq, sx, L0.wq, L0.sw)
    q1, s1 = TL.quantize(torch.relu(y0.to(torch.float32)))
    y1 = TL.linear(q1, s1, L1.wq, L1.sw)
    q2, s2 = TL.quantize(y1.to(torch.float32))
    w = lambda L: {"w": packed(TL.bits(L.wq), 8), "sw": packed(TL.bits(L.sw), 32)}  # noqa: E731

    mm = build("mm", D["mm"], rows_of(D["mm"]), out)
    if mm is None:
        return False
    a = stage("chain-mm0", mm, {"x": packed(TL.bits(xq), 8), "sx": packed(TL.bits(sx), 32), **w(L0)}, out)
    qr = build("q-relu", D["q-relu"], rows_of(D["q-relu"]), out)
    if qr is None:
        return False
    b = stage("chain-q-relu", qr, {"x": a["o0"]}, out)
    c = stage("chain-mm1", mm, {"x": b["o0"], "sx": b["o1"], **w(L1)}, out)
    del mm, qr
    gc.collect()
    qn = build("q-norelu", D["q-norelu"], rows_of(D["q-norelu"]), out)
    if qn is None:
        return False
    d = stage("chain-q-norelu", qn, {"x": c["o0"]}, out)
    checks = {"mm0_is_loop_linear": np.array_equal(a["o0"], packed(TL.bits(y0), 16)),
              "q_relu_codes_are_loop_quantize": np.array_equal(b["o0"], packed(TL.bits(q1), 8)),
              "q_relu_scale_is_loop_quantize": np.array_equal(b["o1"], packed(TL.bits(s1), 32)),
              "mm1_is_loop_linear": np.array_equal(c["o0"], packed(TL.bits(y1), 16)),
              "q_norelu_codes_are_loop_quantize": np.array_equal(d["o0"], packed(TL.bits(q2), 8)),
              "q_norelu_scale_is_loop_quantize": np.array_equal(d["o1"], packed(TL.bits(s2), 32))}
    say("chain", token=tok, ports={"q-relu.x": "mm0.o0 (128 words)", "mm1.x": "q-relu.o0 (64 words)", "mm1.sx": "q-relu.o1 (2 words)",
                                   "q-norelu.x": "mm1.o0 (128 words)"}, **checks)
    return all(checks.values())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("units", nargs="+", choices=("mm", "mm-wrows", "mm-wswrows", "coord", "q-relu", "q-norelu", "chain"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--token", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out)
    D = defs()
    ok = True
    rng = np.random.default_rng(1)
    for u in a.units:
        if u == "chain":
            ok &= chain(D, out, a.token)
            continue
        fn = D["mm" if u.startswith("mm") else u]
        rows = rows_of(fn, {"mm-wrows": ("w",), "mm-wswrows": ("w", "sw")}.get(u, ()))
        st = build(u, fn, rows, out)
        if st is None:
            ok = False
            continue
        try:
            stage(u, st, random_vals(rows, rng), out)
        except Exception as e:  # noqa: BLE001
            ok = False
            say("stage-refused", unit=u, error=f"{type(e).__name__}: {e}", where=traceback.format_exc().strip().splitlines()[-3:])
        del st
        gc.collect()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
