"""``tools/tc_probe_fp4/fp8_gpucheck.py``, the FP8 step check on the GPU, without a GPU.

The hash and the code tables are pinned (a fill unit's plan depends on them) and the hash equals a plain-integer restatement; the
decode table packs exactly ``fp8.products``; a numpy emulation of the fused launches (the primary model standing in for the device)
passes the CPU half, and each way a launch could lie (a dropped sample, a dropped flag, a wrong count, a wrong dumped operand, an
unflagged negative control, a broken recorded chain) fails it; ``fill-verify`` passes an emulated unit and fails a corrupted one.
With nvcc on the PATH, a real build passes the SASS gate and its host build equals the numpy port and ``fp8.groupsum``.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import fp8_gpucheck as G  # noqa: E402

SEED = 20261001
SPECS = [G.CONTROLS[c] for c in ("primary", "(32,) w25 f-133", "ada (16,16) w14 f-139")]


def _mix(x: int) -> int:
    x = (x * 0x9E3779B1 + 0x7F4A7C15) & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x85EBCA6B) & 0xFFFFFFFF
    x ^= x >> 13
    x = (x * 0xC2B2AE35) & 0xFFFFFFFF
    return x ^ (x >> 16)


def test_hash_and_tables_pinned():
    assert G.family_key(SEED, "e4m3", 0) == 2245046515
    assert G.family_key(SEED, "chain", 5) == 2015000481
    assert G.family_key((1 << 40) + 7, "e5m2", 3) == 768526580
    lo, hi = [0, 1, 0xFFFFFFFF], [0, 7, 0xFFFFFFFF]
    assert [int(v) for v in G.tile_key(0x12345678, lo, hi)] == [1621931501, 163870525, 908275395]
    assert [int(v) for v in G.tile_key(0x12345678, lo, hi)] == [_mix(_mix(0x12345678 ^ a) ^ b) for a, b in zip(lo, hi)]
    assert G.tables_digest(G.family("e4m3", "g_cancellation", SEED)[1]) == "df68ace8fca1135a6ccba3d5aa541e39423c090193cc6e57f88e583900b9a71f"
    assert G.tables_digest(G.family("chain", "c_spikes", SEED)[1]) == "2c1ec783616164eca6ac68c5d72a8bc498fd02f7179bd8853ca142c1086ba52f"
    assert G.family("e4m3", "g_cancellation", SEED + 1)[0]["key"] != G.family("e4m3", "g_cancellation", SEED)[0]["key"]
    assert G.tables_digest(G.family("e4m3", "g_cancellation", SEED + 1)[1]) == G.tables_digest(G.family("e4m3", "g_cancellation", SEED)[1])


@pytest.mark.parametrize("kind", ["e4m3", "e5m2", "e4m3e5m2"])
def test_decode_table_packs_fp8_products(kind):
    """fp8_check.cu's product() on the decode table, restated in numpy, equals fp8.products on every code pair."""
    _, fa, fb, _ = G.KINDS[kind]
    dec, pshift = G.decode_table(fa, fb)
    a, b = np.repeat(np.arange(256), 256).reshape(-1, 1), np.tile(np.arange(256), 256).reshape(-1, 1)
    da, db = dec[a].astype(np.int64), dec[256 + b].astype(np.int64)
    ga, gb, na, nb, ia, ib = da & 31, db & 31, (da >> 17) & 1 == 1, (db >> 17) & 1 == 1, (da >> 18) & 1 == 1, (db >> 18) & 1 == 1
    nan = na | nb | (ia & (gb == 0) & ~nb) | (ib & (ga == 0) & ~na)
    inf = (ia | ib) & ~nan
    mag = np.where(nan | inf, 0, (ga * gb) << pshift)
    exp = ((da >> 8) & 255) + ((db >> 8) & 255) - 128
    neg = ((da ^ db) >> 16) & 1 == 1
    want = G.F8.products(fa, a, fb, b)
    for got, w in zip((mag, exp, neg, nan, inf), want):
        live = want[0] != 0 if got is exp else np.ones_like(got, bool)
        assert np.array_equal(got[live], np.asarray(w)[live])


def _emulate(kind: str, name: str, first: int, count: int, smask: int, ctl, steps: int = 0, rng=None) -> tuple[dict, dict, dict, np.ndarray]:
    """A fused launch in numpy with the primary model as the device: the kept tiles (in a shuffled order, as atomics leave them),
    flags, counts and (chains) the first chain's record, as GpuCheck.fused_step / fused_chain return them."""
    fam, tabs = G.family(kind, name, SEED)
    _, fa, fb, _ = G.KINDS[kind]
    rng = rng or np.random.default_rng(0)
    lo, hi = (np.concatenate(x) for x in zip(*G.launch_tiles(kind, first, count, steps)))
    A, Bt, C = G.gen_tiles(fam, tabs, lo, hi)
    n = lo.size
    rec = None
    if kind == "chain":
        C = np.zeros((n, 16, 8), np.uint32)
        W = np.zeros((len(SPECS), n, 16, 8), np.uint32)
        D = np.zeros((n, 16, 8), np.uint32)
        c0 = G.gen_chain_c0(fam, tabs, np.arange(first, first + count, dtype=np.uint32))
        for c in range(count):
            prev = c0[c]
            for s in range(steps):
                t = c * steps + s
                C[t] = prev
                W[:, t] = G.cpu_words(SPECS, fa, fb, A[t:t + 1], Bt[t:t + 1], prev[None])[:, 0]
                D[t] = W[0, t]
                if ctl and (lo[t], hi[t]) == tuple(ctl):
                    D[t, 0, 0] ^= 1
                prev = D[t]
        rec = np.concatenate([c0[:1], D[:steps]])
    else:
        W = G.cpu_words(SPECS, fa, fb, A, Bt, C)
        D = W[0].copy()
        if ctl:
            D[np.flatnonzero((lo == ctl[0]) & (hi == ctl[1]))[0], 0, 0] ^= 1
    gate = G.gated_mask(fa, fb, A, Bt, C, W[0]).reshape(n, 128)
    bad = (W != D[None]).reshape(len(SPECS), n, 128)
    t_i, e_i = np.nonzero(bad[0])
    flags = np.array([[lo[t], hi[t], e, D[t].flat[e], W[0][t].flat[e], gate[t, e]] for t, e in zip(t_i, e_i)], np.uint32).reshape(-1, G.FLAG_WORDS)
    why = np.where((G.hs(G.tile_key(fam["key"], lo, hi), G.SLOT_SAMPLE) & np.uint32(smask)) == 0, G.KEPT_SAMPLED, 0) | \
        np.where(bad[0].any(1), G.KEPT_FLAGGED, 0)
    kept = rng.permutation(np.flatnonzero(why))
    tm = np.zeros((kept.size, G.MAXM), np.uint32)
    tm[:, :len(SPECS)] = bad[:, kept].sum(-1).T
    meta = np.concatenate([np.stack([lo[kept], hi[kept], why[kept]], 1).astype(np.uint32), tm], 1)
    words = np.concatenate([C[kept].reshape(-1, 128), D[kept].reshape(-1, 128)], 1) if kind == "chain" else D[kept].reshape(-1, 128)
    nd = min(4, kept.size)
    ops = np.concatenate([A[kept[:nd]].reshape(nd, -1), Bt[kept[:nd]].reshape(nd, -1), C[kept[:nd]].reshape(nd, -1).view(np.uint8)], 1)
    r = {"meta": meta, "words": np.ascontiguousarray(words), "ops": np.ascontiguousarray(ops), "flags": flags}
    if rec is not None:
        r["rec"] = rec
    summ = {"tiles": n, "elements": 128 * n, "gated": int(gate.sum()), "mismatches": [int((bad[m] & gate).sum()) for m in range(len(SPECS))],
            "record_only": [int((bad[m] & ~gate).sum()) for m in range(len(SPECS))], "nflag": len(flags), "nsamp": kept.size,
            "flag_overflow": False, "sample_overflow": False, "device_ms": 1.0}
    return r, summ, fam, tabs


def _check(kind, r, summ, fam, tabs, first, count, smask, ctl, steps=0):
    return G.check_launch(kind, fam, tabs, SPECS, r, summ, smask, G.launch_tiles(kind, first, count, steps), ctl)


@pytest.mark.parametrize("kind,name", [("e4m3", "g_randn_random_acc"), ("e5m2", "g_near_overflow"), ("e4m3e5m2", "g_wide_spread"),
                                       ("e4m3", "g_subnormal_isolated")])
def test_emulated_step_launch_passes_and_lies_fail(kind, name):
    first, count, smask, ctl = (1 << 33) - 40, 96, 7, ((1 << 33) - 23 & 0xFFFFFFFF, 1)
    r, summ, fam, tabs = _emulate(kind, name, first, count, smask, ctl)
    ok = _check(kind, r, summ, fam, tabs, first, count, smask, ctl)
    assert ok["failures"] == [] and ok["negative_control_flagged"] and ok["primary_gated_beyond_control"] == 0
    assert 0 < ok["sampled_tiles"] < count and ok["flags"] == 1
    lies = {"a sample dropped": lambda r, s: {**r, "meta": r["meta"][1:], "words": r["words"][1:]},
            "a flag dropped": lambda r, s: {**r, "flags": r["flags"][1:]},
            "a count changed": lambda r, s: {**r, "meta": r["meta"] + np.eye(1, r["meta"].shape[1], 3 + len(SPECS) - 1, np.uint32)},
            "a dumped byte changed": lambda r, s: {**r, "ops": r["ops"] ^ np.eye(1, r["ops"].shape[1], 700, np.uint8)},
            "a device word changed": lambda r, s: {**r, "words": r["words"] ^ np.eye(1, r["words"].shape[1], 77, np.uint32)}}
    for lie, f in lies.items():
        assert _check(kind, f(r, summ), summ, fam, tabs, first, count, smask, ctl)["failures"], lie
    assert _check(kind, r, summ, fam, tabs, first, count, smask, (ctl[0] + 1, ctl[1]))["failures"], "the negative control moved"
    assert _check(kind, r, {**summ, "mismatches": [summ["mismatches"][0] + 1, *summ["mismatches"][1:]]}, fam, tabs, first, count, smask, ctl)["failures"]


def test_emulated_chain_launch_and_record():
    first, count, steps, smask, ctl = 5, 3, 24, 3, (7, 11)
    r, summ, fam, tabs = _emulate("chain", "c_gauss_acc", first, count, smask, ctl, steps)
    ok = _check("chain", r, summ, fam, tabs, first, count, smask, ctl, steps)
    assert ok["failures"] == [] and ok["negative_control_flagged"]
    rec = G.check_recorded_chain(fam, tabs, first, r["rec"], r)
    assert rec["failures"] == [] and rec["steps"] == steps
    broken = r["rec"].copy()
    broken[9, 3, 3] ^= 4
    assert G.check_recorded_chain(fam, tabs, first, broken, r)["failures"]
    assert G.check_recorded_chain(fam, tabs, first + 1, r["rec"])["failures"]


def test_fill_verify_passes_an_emulated_unit_and_fails_a_lie(tmp_path):
    unit = tmp_path / "u"
    plan = G.make_plan(unit, "e4m3", ["g_randn_random_acc", "g_acc_floor"], SEED, 2, ["primary", "(32,) w25 f-133", "ada (16,16) w14 f-139"], 3,
                       blocks=1, per_warp=4, samp_cap=64, ndump=4)
    for name in plan["families"]:
        for i in range(plan["launches"]):
            first, count = G.launch_span(plan, i)
            r, summ, _, _ = _emulate("e4m3", name, first, count, plan["smask"], G.launch_control(plan, i), rng=np.random.default_rng(i))
            with open(G.launch_path(unit, name, i), "wb") as f:
                np.savez(f, summary=json.dumps(summ | {"family": name, "launch": i, "gpu_uuid": "GPU-emulated"}), **r)
    args = G.parser().parse_args(["--phase", "fill-verify", "--unit", str(unit), "--workers", "1"])
    assert G.fill_verify(args) == 0
    v = json.loads((unit / "unit_verify.json").read_text())
    assert v["validation"] == "passed" and v["negative_controls"] == v["negative_controls_flagged"] == 2
    assert v["totals"]["tiles"] == 4 * plan["blocks"] * G.WPB * plan["per_warp"]
    p = G.launch_path(unit, "g_acc_floor", 1)
    with np.load(p) as z:
        r = {k: z[k] for k in z.files}
    r["words"] = r["words"] ^ np.eye(1, 128, 5, np.uint32)
    np.savez(p, **r)
    (unit / f"V_{p.stem}.json").unlink()
    assert G.fill_verify(args) == 1


def test_launch_span_control_and_tiles():
    plan = {"kind": "e4m3", "blocks": 2, "per_warp": 3}
    L = 2 * G.WPB * 3
    assert G.launch_span(plan, 5) == (5 * L, L) and G.launch_control(plan, 0) == (17, 0) and G.launch_control(plan, 1) is None
    lo, hi = next(G.launch_tiles("e4m3", (1 << 32) - 2, 4))
    assert lo.tolist() == [0xFFFFFFFE, 0xFFFFFFFF, 0, 1] and hi.tolist() == [0, 0, 1, 1]
    chain = {"kind": "chain", "chains": 10, "steps": 4}
    assert G.launch_span(chain, 2) == (20, 10) and G.launch_control(chain, 0) == (3, 11)
    lo, hi = (np.concatenate(x) for x in zip(*G.launch_tiles("chain", 20, 3, 4)))
    assert lo.tolist() == [20] * 4 + [21] * 4 + [22] * 4 and hi.tolist() == [0, 1, 2, 3] * 3


def test_tool_declaration_parses_the_gate():
    from tool_fp8 import TC_PROBE_FP8_GPUCHECK, parse_gpucheck

    argv = ["uv", "run", "python", "tools/tc_probe_fp4/fp8_gpucheck.py", "--phase", "gate", "--out", "/x", "--lease-via-fill", "me", "--seed", "5"]
    _, k = TC_PROBE_FP8_GPUCHECK.resolve_params(argv)
    assert k["phase"] == "gate" and k["seed"] == 5 and "_unknown" not in parse_gpucheck(argv)


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("nvcc") is None or shutil.which("cuobjdump") is None, reason="needs nvcc and cuobjdump (CUDA 13)")
def test_real_build_passes_sass_and_host_port(tmp_path):
    """The only check of fp8_check.cu's host build (the generator and the model as compiled) against the numpy port short of a GPU."""
    so, _ = G.build(tmp_path)
    sass = G.sass_gate(so)
    assert sass["ok"], sass
    assert G.host_port_check(G.GpuCheck(so), SEED)["failures"] == []
