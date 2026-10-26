"""One drawn unit of a k_log 27 subcircuit of M3's draw (PRF-172) through C-Flock ``--zk``, for its peak memory: the unit staged
as FAMILIES' and WINDOWS' size maps staged it (`UnitStager`, flat, reads inline, an hm96-sha512/row/v2 row per parameter or per
cut part of one, `K.out_rows` per returned field), on synthetic well-formed values, proved by M0 under ``--zk`` (`a0.session`, the
verifier's subset draw of the one unit) and verified by the Lean verifier of record (`a0.lean_verify`), then verified again on the
same session with one bit of its largest proof file flipped. Every flock-circuit and flock-verify process runs under
``timed.py``, which records its wall seconds and peak RSS.

A key is FAMILY:SIZE[:...][:cutK[+J]]: ``chunk:N:RELU`` (`QuantChunkS8_v1`), ``absmax:N:RELU`` (`RowAbsMax_v1`),
``s8rows:K:R`` (`ScaledMmS8CoordinateRows_v1` at sm_90's s8 step) and ``gather:WORLD:N`` (`AllGather_v1`); ``cutK`` reads the
first parameter as K rows, ``cutK+J`` as K rows of its first elements and J rows of one element each. A quantizer's values are
cut from one ReLU row of ``ROW`` N(0, 1) bf16 values: a chunk is that row's chunk ``CHUNK`` under the row's amax, an absmax over
16,384 its first part, and the combine at 16,384 + 3 the last part's values and the other three parts' amaxes. An s8 coordinate
reads codes in [-127, 127] under a positive f32 scale each; a gather reads N(0, 1) bf16 parts.

    PYTHONPATH=<tree>:<tree>/verity/ml/flock/python:<tree>/integrations/vllm \\
      python benchmarks/one_stage/k27_peaks.py --key chunk:8192:1:cut64 --out DIR --bin flock-circuit --lean flock-verify \\
      --timed benchmarks/pouw/hidden_zk/timed.py

Prints ``K27 {json}`` per phase and writes DIR/k27.json: the build's and the stage's seconds and this process's peak RSS, the
unit's ANDs, rows, k_log and circuit bytes, each prover and verifier process's peak RSS and wall seconds, and both verdicts.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.setrecursionlimit(100000)

ROW, CHUNK = 65536, 3


def say(rec: dict, phase: str, **kw) -> None:
    rec[phase] = {"rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2), **kw}
    print("K27 " + json.dumps({"phase": phase, **rec[phase]}, default=str), flush=True)
    (Path(rec["out"]) / "k27.json").write_text(json.dumps(rec, indent=1, default=str))


def free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def definition(key: str):
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops import gemm as G
    from verity.ml.circuits.ops import quant as Q
    from verity_vllm.program.registry import b1_tp2

    fam, *a = key.split(":")
    cut = next((c for c in a if c.startswith("cut")), None)
    a = [int(x) for x in a if not x.startswith("cut")]
    k, _, j = (cut or "")[3:].partition("+")
    if fam in ("chunk", "absmax"):
        fn = bind({"chunk": Q.QuantChunkS8, "absmax": Q.RowAbsMax}[fam], N=a[0], RELU=bool(a[1]))
        return fn, ({"x": (int(k), int(j or 0))} if cut else {})
    if fam == "s8rows":
        return bind(G.ScaledMmS8CoordinateRows, K=a[0], R=a[1], **G.GEMM_SM90_S8), {}
    if fam == "gather":
        return bind(b1_tp2.AllGather, WORLD=a[0], N=a[1]), ({"parts": (int(k), int(j or 0))} if cut else {})
    raise SystemExit(f"no unit {key}")


def rows(fn, cuts: dict):
    """FAMILIES' and WINDOWS' rows (k_map.py, q_sizes3.py): a row per parameter, or K rows of a cut one and J of one element."""
    import a11_kinds as K
    from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2

    params, ret = fn.signature()
    out = []
    for p, t in params:
        if p not in cuts:
            out.append((p, V2, K.leaf_bits(t), [p]))
            continue
        (k, ones), w = cuts[p], sum(t.elem.leaf_widths())
        n = t.n - ones
        out += [(f"{p}#{i}", V2, (n // k) * w, [f"{p}[{r}]" for r in range(i * n // k, (i + 1) * n // k)]) for i in range(k)]
        out += [(f"{p}#{k + i}", V2, w, [f"{p}[{n + i}]"]) for i in range(ones)]
    return out, K.out_rows(len(ret.fields) if hasattr(ret, "fields") else 1)


def bf16(rng: np.random.Generator, n: int) -> np.ndarray:
    """``n`` N(0, 1) values rounded to bf16 by truncation (m1_unit's `bf16_rows`)."""
    return (rng.standard_normal(n).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def amax(w: np.ndarray, relu: bool) -> int:
    """`RowAbsMax_v1`'s word on finite bf16 words: the integer max of |x|'s encodings, a negative x read as 0 under RELU."""
    return int((np.where(w & 0x8000, 0, w) if relu else w & 0x7FFF).max())


def values(key: str, fn, cuts: dict, rng: np.random.Generator) -> tuple[dict, dict]:
    """Each parameter's elements (bf16 words, s8 codes as u8, f32 words) and what its output must be where a twin is cheap."""
    fam = key.split(":")[0]
    S = fn.bindings
    if fam in ("chunk", "absmax"):
        N, relu = int(S["N"]), bool(S["RELU"])
        row = bf16(rng, max(ROW, (CHUNK + 1) * N))
        if fam == "chunk":
            return {"x": row[CHUNK * N:(CHUNK + 1) * N], "amax": np.array([amax(row, relu)], np.uint16)}, {}
        j = cuts.get("x", (0, 0))[1]
        m = N - j
        parts = np.array([amax(row[i * m:(i + 1) * m], relu) for i in range(j)], np.uint16)
        x = np.concatenate([row[j * m:(j + 1) * m], parts])
        return {"x": x}, {"o0": np.array([amax(x, relu)], np.uint16), "o0_is_row_amax": amax(row[:(j + 1) * m], relu)}
    if fam == "s8rows":
        Kk, R = int(S["K"]), int(S["R"])
        codes = lambda: rng.integers(-127, 128, Kk // R).astype(np.int8).view(np.uint8)  # noqa: E731
        scale = lambda: np.array([abs(rng.standard_normal()) * 3 / 127 + 1e-3], np.float32).view(np.uint32)  # noqa: E731
        return {**{f"x{r}": codes() for r in range(R)}, "sx": scale(), **{f"wrow{r}": codes() for r in range(R)}, "sw": scale()}, {}
    if fam == "gather":
        W, N = int(S["WORLD"]), int(S["N"])
        parts = bf16(rng, W * N)
        return {"parts": parts.reshape(W, N)}, {"o0": parts}
    raise SystemExit(f"no values for {key}")


def elem_bits(t) -> int:
    while hasattr(t, "elem"):
        t = t.elem
    return sum(t.leaf_widths())


def packed(a: np.ndarray, bits: int) -> np.ndarray:
    """A row's elements as its words: their little-endian bytes in u16 words (m1_unit's `_packed`)."""
    return np.ascontiguousarray(np.asarray(a).reshape(-1).astype(np.dtype(f"<u{bits // 8}"))).view("<u2")


def row_words(fn, in_rows, vals: dict) -> dict:
    params = dict(fn.signature()[0])
    out = {}
    for name, _s, bits, paths in in_rows:
        p = paths[0].split("[")[0]
        el = [vals[p] if q == p else vals[p][int(q[len(p) + 1:-1])] for q in paths]
        w = packed(np.concatenate([np.asarray(e).reshape(-1) for e in el]), elem_bits(params[p]))
        if w.size * 16 != bits:
            raise SystemExit(f"row {name}: {w.size} words for {bits} bits")
        out[name] = w.reshape(1, -1)
    return out


def timed_bin(real: str, py: str, timed: str, tdir: Path, name: str) -> str:
    """A wrapper of ``real`` that runs each call under timed.py, its record ``tdir/<subcommand>-<ns>.time``."""
    tdir.mkdir(parents=True, exist_ok=True)
    w = tdir / name
    w.write_text(f'#!/usr/bin/env bash\nexec "{py}" "{timed}" "{tdir}/$1-$(date +%s%N).time" "{real}" "$@"\n')
    w.chmod(0o755)
    return str(w)


def times(tdir: Path, sub: str) -> list[dict]:
    return [{**json.loads(f.read_text()), "file": f.name} for f in sorted(tdir.glob(f"{sub}-*.time"))]


def peak(ts: list[dict]) -> dict:
    t = max(ts, key=lambda x: x["maxrss_kb"], default=None)
    return {"maxrss_gb": round(t["maxrss_kb"] / 2**20, 2), "wall_s": t["wall_s"], "exit": t["exit"], "n": len(ts)} if t else {}


def cgroup_peak_gb() -> float | None:
    try:
        path = Path("/proc/self/cgroup").read_text().strip().split("::", 1)[1]
        return round(int(Path(f"/sys/fs/cgroup{path}/memory.peak").read_text()) / 2**30, 2)
    except (OSError, IndexError, ValueError):
        return None


def flipped(src: Path, out: Path) -> dict:
    """The session under ``src`` copied to ``out`` with one bit of its largest proof file flipped, at the file's middle byte."""
    shutil.rmtree(out, ignore_errors=True)
    (sess,) = sorted((src / "sessions").glob("*/session.json"))
    dst = out / "sessions" / sess.parent.name
    shutil.copytree(sess.parent, dst, copy_function=os.link)
    big = max(dst.glob("*.bin"), key=lambda f: f.stat().st_size)
    b = bytearray(big.read_bytes())
    at = len(b) // 2
    b[at] ^= 1
    big.unlink()
    big.write_bytes(bytes(b))
    return {"file": big.name, "bytes": len(b), "byte": at, "bit": 0}


def verify(fv: str, d: Path, out: Path) -> dict:
    import a0

    try:
        return a0.lean_verify(fv, {"dir": d}, 1, out, zk=True)
    except Exception as ex:  # a timeout or a crash of the verifier is a verdict this record keeps
        return {"accepted": False, "why": f"{type(ex).__name__}: {ex}"[:600]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bin", default=None, help="M0's flock-circuit: prove the unit")
    ap.add_argument("--lean", default=None, help="the Lean verifier of record (flock-verify)")
    ap.add_argument("--timed", default=str(HERE.parent / "pouw" / "hidden_zk" / "timed.py"))
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    import a0
    import a11_kinds as K
    from m1_unit import release
    from verity_flock import class_statement as CS
    from verity_flock import ir_lower as IL

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec: dict = {"key": a.key, "out": str(out)}
    fn, cuts = definition(a.key)
    in_rows, out_rows = rows(fn, cuts)
    _ps, ret = fn.signature()
    say(rec, "bound", id=fn.id, cuts=cuts, in_rows=len(in_rows), out_rows=len(out_rows), out_bits=K.leaf_bits(ret),
        row_bytes=sorted({b // 8 for _r, _s, b, _p in in_rows}), cpus=len(os.sched_getaffinity(0)))

    d = out / "stage"
    t = time.time()
    with K.no_cyclic_gc():
        st = CS.UnitStager(fn, f"m1/{K.family(fn.id)}", in_rows, out_rows, circuit=d / "circuit.txt", reads="inline", stage_ands="auto")
    m, c = st.meta, getattr(st.low, "counts", None) or st.low.layout.counts
    say(rec, "built", seconds=round(time.time() - t, 1), ands=c["ands"], unit_rows=c.get("rows"), k_log=m["k_log"], unit_log=m["unit_log"],
        vus_per_block=m["vus_per_block"], pin=st.pin, circuit_bytes=(d / "circuit.txt").stat().st_size)
    if not a.bin:
        return 0

    vals, want = values(a.key, fn, cuts, np.random.default_rng(a.seed))
    words = row_words(fn, in_rows, vals)
    t = time.time()
    with K.no_cyclic_gc():
        got = st.stage(words, d, os.urandom(32))
    for f in ("inst", "pub"):
        (d / f"{f}.bin").rename(d / f"{f}-1.bin")
    IL.write_tables(d / "tables")
    o0 = np.asarray(got["outs"]["o0"]).reshape(-1)
    twin = {k: bool(np.array_equal(o0, v)) for k, v in want.items() if k == "o0"}
    if "o0_is_row_amax" in want:
        twin["o0_is_row_amax"] = int(o0[0]) == want["o0_is_row_amax"]
    say(rec, "staged", seconds=round(time.time() - t, 1), ands=got["ands"], twin=twin,
        inst_bytes=(d / "inst-1.bin").stat().st_size, pub_bytes=(d / "pub-1.bin").stat().st_size)
    if not all(twin.values()):
        return 1
    pin = st.pin
    del st, got
    say(rec, "released", rss_now_gb=release())

    tdir = out / "times"
    fc = timed_bin(a.bin, sys.executable, a.timed, tdir, "flock-circuit")
    fv = timed_bin(a.lean, sys.executable, a.timed, tdir, "flock-verify")
    t = time.time()
    record, live = a0.session(fc, {"dir": d, "pin": pin}, 1, "subset:1", free_port(), out / "session", zk=True,
                              prove_timeout=4 * 3600)
    live = live or {}
    say(rec, "proved", seconds=round(time.time() - t, 1), accepted=live.get("accepted"), why=live.get("why"),
        **{k: live.get(k) for k in ("k_log", "nbl", "dense_m", "prove_total_s", "witness_s", "proof_bytes", "units")},
        verdict=(record or {}).get("verdict"), prove=peak(times(tdir, "prove")), serve=peak(times(tdir, "serve")))
    v = verify(fv, d, out / "session")
    say(rec, "verified", **v, lean=peak(times(tdir, "verify")))
    ok = bool(live.get("accepted")) and v["accepted"]
    if ok:
        for f in tdir.glob("verify-*.time"):
            f.rename(f.with_name("honest-" + f.name))
        flip = flipped(out / "session", out / "flipped")
        nv = verify(fv, d, out / "flipped")
        say(rec, "flipped", **flip, accepted=nv["accepted"], why=nv.get("why"), verify_s=nv.get("verify_s"), lean=peak(times(tdir, "verify")))
        ok = not nv["accepted"]
    say(rec, "done", ok=ok, cgroup_peak_gb=cgroup_peak_gb(),
        children_maxrss_gb=round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20, 2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
