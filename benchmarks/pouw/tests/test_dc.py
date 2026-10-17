"""dc: wgmma.cuh is gen_wgmma.py's output, the receipt twin (dgemm_ref.py) against a per-word loop and against
dgemm.cu's host reference (pinned), and analyze.py's slot arithmetic and pass lines on synthetic rows. The kernels
themselves need an H200 or a B200 (dgemm gate, run.sh)."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

DC = Path(__file__).resolve().parents[1] / "dc"


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, DC / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_wgmma_header_is_current():
    gen = _load("gen_wgmma")
    assert (DC / "wgmma.cuh").read_text() == gen.render(), "run python benchmarks/pouw/dc/gen_wgmma.py"


def _loop_receipts(X, W, seed, bn, L=32):
    """The receipt word by word, as dgemm.cu's header states it (pgemm.cu's: a boundary every L)."""
    ref = _load("dgemm_ref")
    m, k = X.shape
    n = W.shape[0]
    tiles_n = n // bn
    M = 0xFFFFFFFF
    R = np.zeros((m, tiles_n, 16), dtype=np.uint32)
    T = np.zeros((m, n), dtype=np.int64)
    for i in range(m):
        st = {}
        for tn in range(tiles_n):
            for c in range(16):
                y = int(ref.mix32(np.uint32(seed) ^ ref.mix32(np.uint32((i * tiles_n + tn) * 16 + c))))
                st[tn, c] = [y, int(ref.mix32(np.uint32((y + 0x9E3779B9) & M)))]
        for s in range(k // L):
            T[i] += X[i, L * s:L * s + L].astype(np.int64) @ W[:, L * s:L * s + L].T.astype(np.int64)
            for tn in range(tiles_n):
                for j in range(bn):
                    y, z = st[tn, j % 16]
                    t = (j // 16) % 16
                    p = int(T[i, tn * bn + j]) & M
                    u = y ^ p ^ int(ref.KEY_XOR[t])
                    y = (u * int(ref.KEY_MUL[t]) + z) & M
                    rr = ref.ROT[t]
                    z = ((((u << rr) | (u >> (32 - rr))) & M) + y) & M
                    st[tn, j % 16] = [y, z]
        for (tn, c), (y, z) in st.items():
            R[i, tn, c] = y ^ (((z << 16) | (z >> 16)) & M)
    return T, R


def test_receipt_twin_matches_word_loop():
    ref = _load("dgemm_ref")
    rng = np.random.default_rng(3)
    X = rng.integers(-127, 128, size=(3, 96), dtype=np.int8)
    W = rng.integers(-127, 128, size=(64, 96), dtype=np.int8)
    T, R = ref.receipts(X, W, 11, 32)
    T2, R2 = _loop_receipts(X, W, 11, 32)
    assert (T == T2).all() and (R == R2).all()
    X = rng.integers(-127, 128, size=(3, 128), dtype=np.int8)
    W = rng.integers(-127, 128, size=(64, 128), dtype=np.int8)
    T, R = ref.receipts(X, W, 11, 32, L=64)
    T2, R2 = _loop_receipts(X, W, 11, 32, L=64)
    assert (T == T2).all() and (R == R2).all() and not (R == ref.receipts(X, W, 11, 32)[1]).all()


def _pg_sass(kernels: dict[tuple[int, int, int], list[str]]) -> str:
    """cuobjdump's form of pgemm's kernels: a Function line, then one opcode a line."""
    out = []
    for (nb, lw, mode), ops in kernels.items():
        out.append(f"\t\tFunction : _Z5pg100ILi{nb}ELi{lw}ELi{mode}EEvPKaS1_iiiijPiPjPx")
        out += [f"        /*{16 * i:04x}*/                   {op} R0, R1 ;  /* 0x0 */" for i, op in enumerate(ops)]
    return "\n".join(out) + "\n"


def test_pipe_fold_profile_and_qstar(tmp_path):
    """pgemm's honest loader at NB = 4, LW = 4 (4 loads a group): duo's LOP3 + IMAD + LEA.HI on every word of the first
    three loads' range is 3 instructions a word (ALU 2, IMAD 1, LDTM 1/32); q* at L = 32 m is m c_M / 1024 + c_X / 32."""
    pipe = _load("pipe")
    fold = ["LOP3.LUT", "IMAD", "LEA.HI"] * 32
    ops = ["S2R", "SYNCS.PHASECHK.TRANS64.TRYWAIT"] + (["LDTM.x32"] + fold) * 4 + ["BRA"]
    ks = pipe.kernels(_pg_sass({(4, 4, 2): ops, (4, 4, 0): ["UTCIMMA", "EXIT"]}))
    assert set(ks) == {(4, 4, 2), (4, 4, 0)} and ks[(4, 4, 2)] == ops
    p = pipe.fold_profile(ks[(4, 4, 2)], 4, 4)
    assert p["words"] == 96 and abs(p["instructions"] - (3 * 96 + 3) / 96) < 1e-12
    assert p["ALU"] == 2 and p["IMAD"] == 1 and p["LDTM"] == 3 / 96
    mach = {"c_M": 1.0, "c_X": 1.0}
    q32, q64 = pipe.qstar(mach, 32, 1 / 32), pipe.qstar(mach, 64, 1 / 32)
    assert q32["mma_only"] == 1 / 1024 and q64["mma_only"] == 2 / 1024 and q64["with_loads"] == 2 / 1024 + 1 / 32
    rates = [{"arm": 2, "L": 32, "slots_per_clk": 30.0, "nb": 2, "lw": 1},
             {"arm": 2, "L": 32, "slots_per_clk": 31.0, "nb": 4, "lw": 1},
             {"arm": 2, "L": 64, "slots_per_clk": 62.0, "nb": 4, "lw": 2}]
    assert pipe.best(rates, 2, 32)["nb"] == 4 and pipe.best(rates, 2, 64)["lw"] == 2 and pipe.best(rates, 1, 32) is None


def _code(seed: int, which: int, n: int) -> np.ndarray:
    """dgemm.cu's code(): the s8 inputs every mode makes."""
    ref = _load("dgemm_ref")
    base = ref.mix32(np.uint32(seed ^ (which << 28)))
    with np.errstate(over="ignore"):
        x = ref.mix32(base + np.arange(n, dtype=np.uint32))
    return ((x % 255).astype(np.int64) - 127).astype(np.int8)


def test_receipt_twin_matches_host_reference():
    """`dgemm cpuref DIR 128 256 256 7 90` (dgemm.cu's host_ref, CUDA 13.0) wrote these R and C."""
    ref = _load("dgemm_ref")
    m, n, k = 128, 256, 256
    X = _code(7, 1, m * k).reshape(m, k)
    W = _code(7, 2, n * k).reshape(n, k)
    T, R = ref.receipts(X, W, 7, 128)
    assert hashlib.sha256(T.astype(np.int32).tobytes()).hexdigest() == \
        "25fc61f6afb0f6eee278ae91b6a50e1db948e15d9cbbf27e1d7f424071fa198e"
    assert hashlib.sha256(R.tobytes()).hexdigest() == "427f9cfb6776df8a476f3b230fac55b47e2ddf0a08605a1829ea6f746f0f74ba"


def _rows(d: Path, rows: dict[str, dict[str, float]]):
    d.mkdir(parents=True, exist_ok=True)
    (d / "rows.jsonl").write_text("".join(json.dumps({"probe": p, "rates": r, "sm_mhz": 1755.0, "tags": []}) + "\n"
                                          for p, r in rows.items()))


def test_analyze_sm100_pass_line(tmp_path):
    an = _load("analyze")
    t5 = {"0:UTCIMMA*": 1 / 128}   # 256 slots a SM a clock at 128 x 256
    base = {"tc_i8_m128_n256": t5, "ld_32x32b_x32": {"0:LDTM.x32": 0.25}, "imma_s8_k16": {"0:IMMA.16816.S8.S8": 0.5},
            "s_duo": {"0:LOP3.LUT": 2.0}}
    _rows(tmp_path / "a", {**base, "ldmma_w8": {**t5, "1:LDTM.x32": 0.25}})
    out = an.analyze("sm100", tmp_path / "a")
    assert out["M4"]["rho_T5"] == 256 and out["M4"]["rho_ld"] == 256 and out["M4"]["rho_sync"] == 64
    assert out["M4"]["pass"] and out["fold_headroom"] == 64 / 256
    _rows(tmp_path / "b", {**base, "ldmma_w8": {**t5, "1:LDTM.x32": 0.2}})
    assert not an.analyze("sm100", tmp_path / "b")["M4"]["pass"]


def test_analyze_sm100_tmem_model(tmp_path):
    """An exact (OP_UNTIL) point at tensor 0.5 loading 0.6 of ρ_T5, ld alone 0.8: honest at L = 32 is where the edge to
    the MMA alone crosses t = l (6/11), and the adversary adds mma.sync's share of the pipe the MMA leaves (5/11)."""
    an = _load("analyze")
    assert an.upper([(0.5, 0.6)], 0.25) == 0.6 and an.upper([(0.5, 0.6)], 0.75) == 0.0
    t5 = {"0:UTCIMMA*": 1 / 128}
    _rows(tmp_path, {"tc_i8_m128_n256": t5, "ld_32x32b_x32": {"0:LDTM.x32": 0.8 / 4},
                     "imma_s8_k16": {"0:IMMA.16816.S8.S8": 0.5}, "s_duo": {"0:LOP3.LUT": 2.0},
                     "ldU_w8": {"0:UTCIMMA*": 0.5 / 128, "1:LDTM.x32": 0.6 / 4}})
    m4 = an.analyze("sm100", tmp_path)["M4"]
    assert abs(m4["honest_fraction_L32"] - 6 / 11) < 1e-9 and not m4["pass"]
    assert abs(m4["eps_tensor_L32"] - (5 / 11 * 64) / (6 / 11 * 256 + 5 / 11 * 64)) < 1e-9
    assert m4["L_saturated"] == 32 * 41   # 0.024 loaded at tensor 0.98: one slot in 41


def _gate(d: Path, loops: dict[str, list[tuple[int, dict[str, int]]]]):
    """gate.json beside rows.jsonl: each probe's timed loops in SASS order, as (instructions, {pattern: count})."""
    vs = []
    for probe, ls in loops.items():
        out, at = [], 0
        for n, counts in ls:
            out.append({"head": hex(at), "tail": hex(at + 16 * (n - 1)), "counts": counts})
            at += 16 * (n + 4)
        vs.append({"probe": probe, "verdict": {"loops": out}})
    (d / "gate.json").write_text(json.dumps({"verdicts": vs}))


WGI = "IGMMA.64x256x32.S8.S8*"


def _sm90_run(d: Path, sync_rate: float) -> Path:
    """A synthetic sm90 run: the tensor alone at 128 slots/SM/clk (a loop of 8 with 4 IGMMA), scalar issue 4 a SM-clock,
    mix lanes beside the saturated tensor issuing 2, IMAD lanes beside it at half their alone rate, mma.sync beside it
    at sync_rate warp instructions/SM/clk with the tensor at 0.9."""
    t = 128 / 4096
    _rows(d, {"wg_s8_n256_a1": {f"0:{WGI}": t}, "imma_s8_k16": {"0:IMMA.16816.S8.S8": 0.5},
              "s_lop3": {"0:LOP3.LUT": 2.0}, "s_imad": {"0:IMAD": 2.0}, "s_shfl": {"0:SHFL.BFLY": 1.0},
              "s_ffma": {"0:FFMA": 3.8}, "p_ffma_lop3": {"0:FFMA": 2.0, "0:LOP3.LUT": 2.0},
              "wgU_mix": {f"0:{WGI}": t, "1:FFMA": 1.0, "1:I*ADD*": 1.0},
              "wgU_imad": {f"0:{WGI}": t, "1:IMAD": 1.0},
              "wgU_sync_k16": {f"0:{WGI}": 0.9 * t, "1:IMMA.16816.S8.S8": sync_rate}})
    _gate(d, {"wg_s8_n256_a1": [(8, {WGI: 4})], "p_ffma_lop3": [(8, {"FFMA": 4, "LOP3.LUT": 4})],
              "wgU_mix": [(8, {WGI: 4}), (64, {"FFMA": 32, "I*ADD*": 32})]})
    return d


def _census_tables(census, lines, rows):
    by_row: dict[str, dict[str, float]] = {}
    for ln in lines:
        assert (ln["kind"] not in by_row.get(ln["row"], {})), f"two lines for {ln['kind']} on {ln['row']}"
        by_row.setdefault(ln["row"], {})[ln["kind"]] = census.rowsmod.u_of(
            census.rowsmod.CYCLE_U, ln["rate"], ln["noise"], ln["cycles"], ln["u"])
    assert set(by_row) == {r["row"] for r in rows}
    return by_row


def test_census_sm90_rows(tmp_path):
    """The issue facet through scalar issue (0, 1) and the mix point (1/128, 0.5078) a SMSP-cycle is IGMMA 63 + other 1
    (x 0.98); IMAD at half beside the tensor gives its pipe facet IGMMA 64 + IMAD 2 (x 0.98); mma.sync inside the
    shared line joins M, outside it gets MS."""
    census = _load("census")
    run = _sm90_run(tmp_path / "r20261017-000000-abcd" / "out", 0.05)
    dev = census.Device("sm90", [run])
    lines, rows = census.census(dev)
    cu = census.rowsmod.CYCLE_U
    by_row = _census_tables(census, lines, rows)
    assert abs(by_row["JI90_0"]["IGMMA_S8_N256"] / cu - 0.98 * 63) < 1e-6
    assert abs(by_row["JI90_0"]["FFMA"] / cu - 0.98) < 1e-9 and abs(by_row["JI90_0"]["IMMA16816_S8"] / cu - 0.98) < 1e-9
    assert abs(by_row["JPI90_0"]["IGMMA_S8_N256"] / cu - 0.98 * 64) < 1e-6
    assert abs(by_row["JPI90_0"]["IMAD"] / cu - 0.98 * 2) < 1e-9
    assert {"IGMMA_S8_N256", "IMMA16816_S8"} <= set(by_row["M"]) and "MS" not in by_row
    assert set(by_row["issue"]) == {"IGMMA_S8_N256", "IMMA16816_S8", "ALU", "IMAD", "SHFL", "FFMA"}
    assert "FFMA" not in {k for r, ks in by_row.items() if r in ("A", "I", "SHFL") for k in ks}
    assert all(ln["runs"] == "r20261017-000000-abcd" for ln in lines)
    text = census.tables(dev)
    assert "q*: c_M 61.740 SMSP-cycles over 128 words a lane" in text
    run2 = _sm90_run(tmp_path / "r20261017-000001-abce" / "out", 0.2)
    by_row = _census_tables(census, *census.census(census.Device("sm90", [run2])))
    assert "IMMA16816_S8" in by_row["MS"] and "IMMA16816_S8" not in by_row["M"]


def test_device_census_is_well_formed():
    """Every device directory's lines are lines machine_model reads (one of rate, cycles and u), each on a row rows.csv
    states, once a row, naming its runs."""
    census = _load("census")
    devices = DC.parent / "machine_model" / "data" / "devices"
    dirs = sorted(p for p in devices.iterdir() if p.is_dir()) if devices.exists() else []
    for d in dirs:
        with open(d / "census.csv", newline="") as f:
            lines = list(csv.DictReader(f))
        with open(d / "rows.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        assert tuple(lines[0]) == census.rowsmod.CENSUS_COLUMNS
        by_row = _census_tables(census, lines, rows)
        assert {"issue", "M", "A", "I", "SHFL"} <= set(by_row), d
        for ln in lines + rows:
            assert ln["runs"] and all(census.rowsmod.RUN_ID.fullmatch(r) for r in ln["runs"].split()), (d, ln)


def test_analyze_sm90_pass_line(tmp_path):
    an = _load("analyze")
    wg = {"0:IGMMA.64x256x32.S8.S8*": 128 / 4096}   # 128 slots a SM a clock
    _rows(tmp_path / "a", {"wg_s8_n256_a1": wg, "wg_e4m3_n256_a1": {"0:QGMMA.64x256x32.F32.E4M3.E4M3*": 128 / 4096},
                           "imma_s8_k16": {"0:IMMA.16816.S8.S8": 1.0},
                           "wgmix_imad_r64": {**wg, "0:IMAD": 0.5}, "s_imad": {"0:IMAD": 2.0}})
    out = an.analyze("sm90", tmp_path / "a")
    assert out["M3"]["honest_slots"] == 128 and out["M3"]["pass"]
    j = out["joints"]["wgmix_imad_r64"]
    assert j["lane_fraction"] == 0.25 and j["a"] == (2.0 - 0.5) / (128 / 4096) and j["tensor_fraction"] == 1.0
    _rows(tmp_path / "b", {"wg_s8_n256_a1": wg, "imma_s8_k16": {"0:IMMA.16816.S8.S8": 1.1}})
    assert not an.analyze("sm90", tmp_path / "b")["M3"]["pass"]
