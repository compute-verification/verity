"""dc: wgmma.cuh is gen_wgmma.py's output, the receipt twin (dgemm_ref.py) against a per-word loop and against
dgemm.cu's host reference (pinned), and analyze.py's slot arithmetic and pass lines on synthetic rows. The kernels
themselves need an H200 or a B200 (dgemm gate, run.sh)."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import re
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


def _pg_sass(kernels: dict[tuple[int, int, int, int], list[str]]) -> str:
    """cuobjdump's form of pgemm's kernels: a Function line, then one opcode a line."""
    out = []
    for (nb, lw, mode, mi), ops in kernels.items():
        out.append(f"\t\tFunction : _Z5pg100ILi{nb}ELi{lw}ELi{mode}ELb{mi}EEvPKaS1_iiiijPiPjPx")
        out += [f"        /*{16 * i:04x}*/                   {op} R0, R1 ;  /* 0x0 */" for i, op in enumerate(ops)]
    return "\n".join(out) + "\n"


def test_pipe_fold_profile_and_qstar(tmp_path):
    """pgemm's honest loader at NB = 4, LW = 4 (4 loads a group): duo's LOP3 + IMAD + LEA.HI on every word of the first
    three loads' range is 3 instructions a word (ALU 2, IMAD 1, LDTM 1/32); q* at L = 32 m is m c_M / 1024 + c_X / 32."""
    pipe = _load("pipe")
    fold = ["LOP3.LUT", "IMAD", "LEA.HI"] * 32
    ops = ["S2R", "SYNCS.PHASECHK.TRANS64.TRYWAIT"] + (["LDTM.x32"] + fold) * 4 + ["BRA"]
    ks = pipe.kernels(_pg_sass({(4, 4, 2, 0): ops, (4, 4, 0, 1): ["UTCIMMA", "EXIT"]}))
    assert set(ks) == {(4, 4, 2, 0), (4, 4, 0, 1)} and ks[(4, 4, 2, 0)] == ops
    p = pipe.fold_profile(ks[(4, 4, 2, 0)], 4, 4)
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


def test_analyze_sm100_li_joint(tmp_path):
    """tcliU: the MMA at 0.98 beside lanes folding 1/16 x128 load a SM a clock (256 words) keeps up with L = 64's half of
    0.98 x 256 slots; tclaneU_li is a lane joint against the LOP3 + IMAD pair alone; neither enters the TMEM region."""
    an = _load("analyze")
    t5 = {"0:UTCIMMA*": 1 / 128}
    _rows(tmp_path, {"tc_i8_m128_n256": t5, "ld_32x32b_x32": {"0:LDTM.x32": 0.25}, "s_duo": {"0:LOP3.LUT": 2.0},
                     "p_imad_lop3": {"0:IMAD": 2.0, "0:LOP3.LUT": 2.0},
                     "tcliU_x128": {"0:UTCIMMA*": 0.98 / 128, "1:LDTM.x128": 1 / 16, "1:LOP3.LUT": 4.0, "1:IMAD": 4.0},
                     "tclaneU_li": {"0:UTCIMMA*": 1 / 128, "1:LOP3.LUT": 1.5, "1:IMAD": 1.5}})
    out = an.analyze("sm100", tmp_path)
    li = out["joints"]["tcliU_x128"]
    assert li["li_words"] == 256 and abs(li["tensor_fraction"] - 0.98) < 1e-12
    assert abs(li["keeps_up_l64"] - 256 / (0.98 * 128)) < 1e-12 and "ld_words" not in li
    assert out["joints"]["tclaneU_li"]["lane_fraction"] == 0.75
    assert all(p[0] != 0.98 for p in out["tmem_model"]["tmem_points"])


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


def _census_tables(census, lines, rows, epochs=(), every_row=True):
    """A row's table of u by kind, from the lines open at the last epoch. A kind has several lines on a row only in
    disjoint epochs (`since` through `until`, in `epochs`' order, as machine_model's `holds` reads them). Every line is
    on a stated row; with `every_row`, every stated row has a line (a measured facet no class loads, like TMAB, has
    none)."""
    by_row: dict[str, dict[str, float]] = {}
    spans: dict[tuple[str, str], list[tuple[int, int]]] = {}
    last = max(len(epochs) - 1, 0)
    for ln in lines:
        since, until = ln.get("since", ""), ln.get("until", "")
        span = (epochs.index(since) if since else 0, epochs.index(until) if until else last)
        key = (ln["kind"], ln["row"])
        assert all(span[1] < s[0] or s[1] < span[0] for s in spans.get(key, [])), \
            f"two lines for {ln['kind']} on {ln['row']}"
        spans.setdefault(key, []).append(span)
        if until:
            continue
        by_row.setdefault(ln["row"], {})[ln["kind"]] = census.rowsmod.u_of(
            census.rowsmod.CYCLE_U, ln["rate"], ln["noise"], ln["cycles"], ln["u"])
    stated = {r["row"] for r in rows}
    assert {ln["row"] for ln in lines} <= stated
    assert set(by_row) == stated if every_row else set(by_row) <= stated
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


def test_census_tables_take_one_line_an_epoch():
    """A refit closes a line at one epoch and opens its successor at the next; two lines open in one epoch are
    refused."""
    census = _load("census")
    rows = [{"row": "JI100_0"}]

    def line(u, since="", until=""):
        return {"kind": "UNI", "row": "JI100_0", "rate": "", "noise": "", "cycles": "", "u": u, "since": since,
                "until": until}

    epochs = ["DC-100", "DC-100-ri"]
    refit = [line("0.906", until="DC-100"), line("0.0008624", since="DC-100-ri")]
    assert _census_tables(census, refit, rows, epochs)["JI100_0"]["UNI"] == census.rowsmod.u_of(
        census.rowsmod.CYCLE_U, "", "", "", "0.0008624")
    for bad in ([line("0.906"), line("0.0008624", since="DC-100-ri")], [line("0.906"), line("0.0008624")]):
        try:
            _census_tables(census, bad, rows, epochs)
        except AssertionError as e:
            assert "two lines for UNI on JI100_0" in str(e)
        else:
            raise AssertionError("overlapping lines accepted")


def test_device_census_is_well_formed():
    """Every device directory's lines are lines machine_model reads (one of rate, cycles and u), each on a row rows.csv
    states, once a row in each epoch of model.toml, naming its runs; a `datasheet` row and its lines name the note that
    cites the vendor document instead (directly, or as `@NAME` from machine_model's data/citations.toml), since no run
    measured them."""
    import tomllib
    census = _load("census")
    note = re.compile(r"note:\d{8}T\d{4}Z-[a-z0-9-]+")
    named = tomllib.loads((DC.parent / "machine_model" / "data" / "citations.toml").read_text())
    devices = DC.parent / "machine_model" / "data" / "devices"
    dirs = sorted(p for p in devices.iterdir() if p.is_dir()) if devices.exists() else []
    for d in dirs:
        with open(d / "census.csv", newline="") as f:
            lines = list(csv.DictReader(f))
        with open(d / "rows.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        assert tuple(lines[0]) == census.rowsmod.CENSUS_COLUMNS
        epochs = tomllib.loads((d / "model.toml").read_text())["epochs"]
        by_row = _census_tables(census, lines, rows, epochs, every_row=False)
        assert {"issue", "M", "A", "I", "SHFL"} <= set(by_row), d
        sheet = {r["row"] for r in rows if r["status"] == "datasheet"}
        for ln in lines + rows:
            cite = note if ln["row"] in sheet else census.rowsmod.RUN_ID
            runs = [named.get(r[1:], r) if r.startswith("@") else r for r in ln["runs"].split()]
            assert runs and all(cite.fullmatch(r) for r in runs), (d, ln)


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


def test_uniform_tables_pin_every_loop():
    """Each family's tables pin every probe's loop: the alone chains count a uniform opcode, every other uniform probe
    the MMA beside them, and each IMAD-form probe an opcode of its loop."""
    un = _load("uniform")
    for fam in ("sm90", "sm100"):
        tab = un.tables(fam)
        assert len(tab) == 115, fam
        for name, t in tab.items():
            assert t["counted"] in t["counts"], (fam, name)
            if name.startswith("uz_"):
                assert un.kind_of(t["counted"]) == "UNI" and set(map(un.kind_of, t["counts"])) <= {"UNI", "ALU"}
            elif t["binary"].endswith(("uniform", "issue", "sat")):
                assert t["counted"].startswith("IMMA.16832"), (fam, name)


def test_uniform_fit_and_check():
    """UP at its alone rate, an issue slot on the issue rows, 0 on a pipe row with its limit beside it; a point UP's line
    exceeds is reported."""
    un = _load("uniform")
    coef = {"ALU": {"issue": 1.0, "A": 2.0, "JI9_0": 0.98}, "IMAD": {"issue": 1.0, "I": 2.0}}
    pts = [{"probe": "uz", "T": 200.0, "n": {"UNI": 100}},
           {"probe": "uh", "T": 300.0, "n": {"ALU": 64, "IMAD": 32, "UNI": 64, "other": 4}}]
    f = un.fit(pts, coef, ["issue", "A", "I", "JI9_0", "UP"], 1.96)
    assert f["UP"] == {"c": 1.96} and f["issue"]["c"] == 1.0 and f["JI9_0"]["c"] == 0.98
    assert f["A"]["c"] == 0.0 and abs(f["A"]["limit"] - min((306 - 128) / 64, 204 / 100)) < 1e-12
    assert abs(f["issue"]["limit"] - 2.04) < 1e-12 and f["issue"]["set_by"] == "uz"
    pts[0]["T"] = 400.0
    assert un.fit(pts, coef, ["A"], 1.96)["A"] == {"c": 0.0, "limit": (306 - 128) / 64, "set_by": "uh"}
    assert un.check(pts, coef, f) == []
    assert un.check(pts + [{"probe": "bad", "T": 100.0, "n": {"UNI": 60}}], coef, f) == \
        ["bad: UP 117.60 > T 100.00 (1 + 0.02)"]
    g = un.fit(pts, coef, ["issue", "JI9_0"], 1.96, price=0.9)
    assert g["issue"]["c"] == 0.9 and abs(g["JI9_0"]["c"] - 0.98 * 0.9) < 1e-12


def test_uniform_issue_price_reads_only_saturated_streams(tmp_path):
    """The issue price is the least slowdown a uniform instruction adds, counted in full from the SASS (a loop-control
    uniform instruction the tables drop is still subtracted), over the streams whose baseline issues ≥ 0.95, less 2%."""
    un = _load("uniform")
    out = tmp_path / "out"
    (out / "probes").mkdir(parents=True)
    (out / "sass").mkdir()
    loops = {"uv_0": (98, 0), "uv_32x": (98, 48), "uv_64x": (98, 96), "uw_0": (98, 0), "uw_64x": (98, 96)}
    T = {"uv_0": 100.0, "uv_32x": 140.0, "uv_64x": 196.0, "uw_0": 110.0, "uw_64x": 120.0}
    sass, verdicts, rows = [], [], []
    for name, (vec, uni) in loops.items():
        ops = ["FFMA R1, R1, 1, R1"] * (vec - 3) + ["UIADD3 UR4, UR4, 1, URZ", "UISETP.NE.AND UP0, PT, UR4, 8, PT"] + \
              ["ULOP3.LUT UR5, UR5, UR6, 3, 0x96, !UPT"] * uni + ["@P0 BRA 0x10"]
        sass.append(f"\t\tFunction : {name}\n" + "\n".join(f"        /*{16 * (i + 1):04x}*/   {op} ;   /* 0x0 */"
                                                           for i, op in enumerate(ops)))
        verdicts.append({"binary": "sm90_sat", "probe": name,
                         "verdict": {"loops": [{"head": "0x0010", "tail": hex(16 * len(ops))}]}})
        rows.append(json.dumps({"probe": name, "run": "r1", "rates": {"0:IMMA.16832.S8.S8": 4 * 8 / T[name]}}))
    (out / "sass" / "sm90_sat.sass").write_text("\n".join(sass) + "\n")
    (out / "gate-build.json").write_text(json.dumps({"verdicts": verdicts}))
    (out / "probes" / "rows.jsonl").write_text("\n".join(rows) + "\n")
    assert un.loop_counts(out)["uv_32x"] == {"FFMA": 95, "UIADD3": 1, "UISETP.NE.AND": 1, "ULOP3.LUT": 48, "BRA": 1}
    s = un.saturated([out], "sm90")
    uv = next(v for k, v in s["streams"].items() if k.startswith("uv "))
    uw = next(v for k, v in s["streams"].items() if k.startswith("uw "))
    assert abs(uv["issue0"] - 0.98) < 1e-12 and uw["issue0"] < 0.95
    assert uv["points"]["uv_32x"]["added"] == 48 and abs(uv["points"]["uv_32x"]["dT"] - 40 / 48) < 1e-12
    assert abs(uw["points"]["uw_64x"]["dT"] - 10 / 96) < 1e-12
    assert s["set_by"] == "uv_32x" and abs(s["price"] - 40 / 48 / 1.02) < 1e-12


def test_ldtm_price_drops_the_single_load_restructure():
    """The LDTM price is the least isolated load cost over the priced points of a read stream: at least two loads, the
    point still issue-saturated. The single-load point (the loop restructures) and the load-path-bound points above it
    (issue < 0.95) are dropped, and streams below 0.95 aren't read. Repeats are the price computed from each run."""
    ld = _load("ldtm")

    def fake(out):
        return {
            "mw": {"T0": 1329.0, "issue0": 0.965, "read": True, "tc0": 100.0, "points": {
                "mw_1": {"K": 1, "T": 1323.0, "loads": 1, "others": 5, "added": {}, "issue": 0.974, "all": -6.0, "ldtm": -11.0},
                "mw_2": {"K": 2, "T": 1333.0, "loads": 2, "others": 6, "added": {}, "issue": 0.968, "all": 2.0, "ldtm": -1.0},
                "mw_8": {"K": 8, "T": 1467.0, "loads": 8, "others": 24, "added": {}, "issue": 0.896, "all": 17.0, "ldtm": 14.0}}},
            "nx": {"T0": 1112.0, "issue0": 0.981, "read": True, "points": {
                "nx_2": {"K": 2, "T": 1120.0, "loads": 2, "others": 6, "added": {}, "issue": 0.981, "all": 4.0, "ldtm": 1.0}}},
            "lv": {"T0": 790.0, "issue0": 0.944, "read": False, "points": {
                "lv_2": {"K": 2, "T": 800.0, "loads": 2, "others": 6, "added": {}, "issue": 0.924, "all": 2.0, "ldtm": -3.0}}},
        }

    ld.streams = fake
    assert {n for n, _ in ld.priced(fake(None)["mw"])} == {"mw_2"}          # K=1 and the desaturated K=8 dropped
    assert {n for n, _ in ld.priced(fake(None)["lv"])} == set()             # below 0.95, not read
    res = ld.analyze([Path("r1/out"), Path("r2/out")])
    assert res["set_by"] == "mw_2" and abs(res["least"] + 1.0) < 1e-12      # lv_2 lower but not read; mw_1 excluded
    assert abs(res["price"] - (-1.0 / 1.02)) < 1e-12 and res["pass"] is False
    assert res["price_by_run"] == [-1.0 / 1.02, -1.0 / 1.02] and res["price_spread"] == 0.0 and res["repeat_spread"] == 0.0


def test_ldtm_loop_counts_takes_the_larger_loop(tmp_path):
    """An m probe times two loops (the lane stream and the MMA warp's); loop_counts counts the larger, the stream."""
    ld = _load("ldtm")
    out = tmp_path / "out"
    (out / "sass").mkdir(parents=True)
    body = ["FFMA R1, R1, 1, R1", "LDTM.x128 R4, tmem[UR6]", "@P0 BRA 0x10",      # the stream loop, 0x10..0x30
            "UTCIMMA gdesc[UR6], gdesc[UR8], tmem[UR5], tmem[UR10], idesc, UPT", "@P1 BRA 0x40"]  # the MMA loop, 0x40..0x50
    sass = "\t\tFunction : mw_2\n" + "\n".join(f"        /*{16 * (i + 1):04x}*/   {op} ;   /* 0x0 */"
                                               for i, op in enumerate(body))
    (out / "sass" / "sm100_ldtm.sass").write_text(sass + "\n")
    (out / "gate-build.json").write_text(json.dumps({"verdicts": [{"binary": "sm100_ldtm", "probe": "mw_2", "verdict":
        {"loops": [{"head": "0x0010", "tail": "0x0030", "top": {"FFMA": 1, "LDTM.x128": 1, "BRA": 1}},
                   {"head": "0x0040", "tail": "0x0050", "top": {"UTCIMMA": 1}}]}}]}))
    assert ld.loop_counts(out)["mw_2"] == {"FFMA": 1, "LDTM.x128": 1, "BRA": 1}


def test_emp_prices_every_loop_on_ds90_facets(tmp_path):
    """emp.py charges every opcode between the gated loop's head and tail at the trip rate: a wgmma s8 n256 loop at
    1/32 a SM-clock fills the tensor facet exactly, a 2:4 sparse form at its dense equivalent, BGMMA only issue, and a
    loop 1% faster is reported above the sheet. The event clock is the longest block's cycles over the time a call."""
    emp = _load("emp")
    run, sass = tmp_path / "run" / "sm90_emp", tmp_path / "sass"
    run.mkdir(parents=True)
    sass.mkdir()
    bodies = {"wg": ["IGMMA.64x256x32.S8.S8 R24, gdesc[UR4], R24"] * 4 + ["FFMA R1, R1, 1, R1"] * 8 +
                    ["WARPGROUP.ARRIVE", "UIADD3 UR4, UR4, 1, URZ", "ISETP.LE.AND P0, PT, R2, 8, PT", "@P0 BRA 0x10"],
              "sp": ["IGMMA.SP.64x256x64.S8.S8 R24, gdesc[UR4], R24, R2, 0x0"] * 4 + ["@P0 BRA 0x10"],
              "b1": ["BGMMA.64x256x256.AND.POPC R24, gdesc[UR4], R24"] * 4 + ["@P0 BRA 0x10"]}
    rates = {"wg": 1 / 32, "sp": 1 / 32, "b1": 1 / 32, "fast": 1.01 / 32}
    text, verdicts, rows = [], [], []
    for probe, rate in rates.items():
        kern = "wg" if probe == "fast" else probe
        if probe != "fast":
            body = bodies[kern]
            text.append(f"\t\tFunction : {kern}\n" + "\n".join(f"        /*{16 * (i + 1):04x}*/   {op} ;   /* 0x0 */"
                                                              for i, op in enumerate(body)))
        op = bodies[kern][0].split()[0]
        verdicts.append({"probe": probe, "verdict": {"kernel": kern, "loops": [
            {"head": "0x0010", "tail": hex(16 * len(bodies[kern])), "counts": {op: 4}}]}})
        rows.append(json.dumps({"binary": "sm90_emp", "probe": probe, "run": "r1", "sm_mhz": 1965, "spread": 0.0,
                                "tags": [], "counted": op, "rate": rate, "rates": {f"0:{op}": rate}}))
    (sass / "sm90_emp.sass").write_text("\n".join(text) + "\n")
    (run / "gate.json").write_text(json.dumps({"verdicts": verdicts}))
    (run / "rows.jsonl").write_text("\n".join(rows) + "\n")
    (run / "times.jsonl").write_text(json.dumps({"op": "time", "probe": "wg", "calls": 10, "ms": 20.0,
                                                 "cyc": [3_900_000, 3_960_000]}) + "\n")
    res = {r["probe"]: r for r in emp.analyze(tmp_path / "run", emp.read_sass(sass))}
    wg = res["wg"]
    assert abs(wg["occupancy"]["tensor"] - 1.0) < 1e-12 and abs(wg["macs_sm_clk"] - 4096) < 1e-9
    assert abs(wg["occupancy"]["issue"] - 16 / 128 / 4) < 1e-12 and abs(wg["uniform_issue"] - 1 / 512) < 1e-12
    assert abs(wg["occupancy"]["alu"] - 2 / 512) < 1e-12 and wg["event_mhz"] == 1980.0
    assert abs(res["sp"]["occupancy"]["tensor"] - 1.0) < 1e-12 and abs(res["sp"]["macs_sm_clk"] - 4096) < 1e-9
    assert "tensor" not in res["b1"]["occupancy"] and res["b1"]["worst"] == "issue" and res["b1"]["event_mhz"] is None
    assert res["fast"]["worst"] == "tensor" and abs(res["fast"]["worst_occ"] - 1.01) < 1e-12
    assert emp.main([str(tmp_path / "run"), str(sass), "--json", str(tmp_path / "out.json")]) == 1
    assert json.loads((tmp_path / "out.json").read_text())["over"] == ["fast"]


def test_uniform_write_census_is_idempotent(tmp_path):
    """--write replaces UNI's measured lines in place and the UP row, puts UP in the current vector's rows once, and
    census.py --write keeps both. The data sheet's UNI,DS_issue (A2u), in census.csv or in its own census/DS-90.csv, and
    DC-90-ri's refit of JI90_0 aren't uniform.py's and stay; JI90_0's new line keeps its `until`, so the refit still
    replaces it on DC-90-ri."""
    import shutil
    un, census = _load("uniform"), _load("census")
    dev = tmp_path / "sm_90"
    shutil.copytree(DC.parent / "machine_model" / "data" / "devices" / "sm_90", dev)
    res = {"up": {"rate": 0.5, "by": "uz_64i", "cycles": 1.96}, "uniform_alone": {"uz_64i": 0.5}, "violations": [],
           "rows": {"UP": {"c": 1.96}, "issue": {"c": 1.0, "limit": 1.2, "set_by": "uh_64i"},
                    "JI90_0": {"c": 0.5}, "A": {"c": 0.0, "limit": 0.9, "set_by": "uh_64i"}}}
    before = (dev / "census.csv").read_text().splitlines()
    old = [ln.split(",") for ln in before if ln.startswith("UNI,")]
    until = next(f[7] for f in old if f[1] == "JI90_0" and not f[6])
    others = [",".join(f) for f in old if f[6] or f[1].startswith("DS_")]
    assert until and "UNI,JI90_0,,,0.0001862,,DC-90-ri,,r20261017-204843-4803" in others
    for _ in range(2):
        un.write_census(dev, res, "r20261017-000000-abcd")
    cen = (dev / "census.csv").read_text().splitlines()
    uni = [ln for ln in cen if ln.startswith("UNI,")]
    assert uni == ["UNI,UP,,,1.96,,,,r20261017-000000-abcd", "UNI,issue,,,1,,,,r20261017-000000-abcd",
                   f"UNI,JI90_0,,,0.5,,,{until},r20261017-000000-abcd", *others]
    sheet = [dev / "census.csv", dev / "census" / "DS-90.csv"]
    assert any(ln.startswith("UNI,DS_issue,") for p in sheet if p.exists() for ln in p.read_text().splitlines())
    first = next(i for i, ln in enumerate(before) if ln.startswith("UNI,"))
    assert cen[:first] == before[:first] and cen[first].startswith("UNI,UP,")
    assert sum(ln.startswith("UP,") for ln in (dev / "rows.csv").read_text().splitlines()) == 1
    import tomllib
    assert tomllib.loads((dev / "vectors" / "DC-90.toml").read_text())["rows"].count("UP") == 1
    assert sum(ln["kind"] == "UNI" for ln in census.kept(dev, "census.csv")) == len(uni)
    assert sum(r["row"] == "UP" for r in census.kept(dev, "rows.csv")) == 1


def test_census_write_keeps_what_it_does_not_write(tmp_path, capsys):
    """census.py --write replaces its own rows and drops a joint row its hull no longer has (JPA90_7 here), and keeps
    every other line and row in the directory: a data-sheet line in census.csv (written here where the data sheet has
    its own census/DS-90.csv), TMAB, and uniform.py's UNI and UP. A row whose old description extends the generated one
    keeps its note."""
    import shutil
    census = _load("census")
    dev = tmp_path / "sm_90"
    shutil.copytree(DC.parent / "machine_model" / "data" / "devices" / "sm_90", dev)
    run = _sm90_run(tmp_path / "r20261017-000000-abcd" / "out", 0.05)
    with open(dev / "census.csv", newline="") as f:
        before = list(csv.DictReader(f))
    sheet = {**before[0], "kind": "ALU", "row": "DS_alu", "rate": "", "noise": "", "cycles": "2", "u": "", "since": "",
             "until": "", "runs": "note:20261017T1818Z-draft-axioms"}
    if sheet not in before:
        with open(dev / "census.csv", "a", newline="") as f:
            csv.DictWriter(f, fieldnames=list(before[0]), lineterminator="\n").writerow(sheet)
        before.append(sheet)
    with open(dev / "rows.csv", newline="") as f:
        rows_before = list(csv.DictReader(f))
    foreign = [ln for ln in before if ln["kind"] == "UNI" or not census.OWN_ROW.fullmatch(ln["row"])]
    assert sheet in foreign and any(ln["kind"] == "UNI" for ln in foreign)
    gen_rows = census.census(census.Device("sm90", [run]))[1]
    gen_m = next(r for r in gen_rows if r["row"] == "M")["description"]
    stale = {**before[0], "row": "JPA90_7", "cycles": "1", "rate": "", "noise": "", "u": ""}
    with open(dev / "census.csv", "a", newline="") as f:
        csv.DictWriter(f, fieldnames=list(before[0]), lineterminator="\n").writerow(stale)
    for r in rows_before:
        if r["row"] == "M":
            r["description"] = gen_m + ". A note added by hand"
    (dev / "rows.csv").write_text(census.rows_csv(rows_before + [{"row": "JPA90_7", "status": "joint", "runs":
                                                                   "r20261017-000000-abcd", "description": "stale"}]))
    assert census.main(["sm90", str(run), "--write", str(dev)]) == 0
    with open(dev / "census.csv", newline="") as f:
        after = list(csv.DictReader(f))
    with open(dev / "rows.csv", newline="") as f:
        rows_after = {r["row"]: r for r in csv.DictReader(f)}
    assert all(ln in after for ln in foreign)
    assert "JPA90_7" not in {ln["row"] for ln in after} and "JPA90_7" not in rows_after
    assert {r["row"] for r in rows_before if not census.OWN_ROW.fullmatch(r["row"])} <= set(rows_after)
    assert {"DS_issue", "TMAB", "UP"} <= set(rows_after)
    assert rows_after["M"]["description"] == gen_m + ". A note added by hand"
    assert "M's description is regenerated" not in capsys.readouterr().err


def test_census_prices_a_memory_lane_and_moves_every_scalar_coefficient_class(tmp_path):
    """A memory lane timed alone and beside the tensor puts its class on the issue row and every joint issue row at the
    scalar coefficient (LD.E's is UNLISTED, the measured readings' catch-all); --write gives the classes that carry the
    scalar coefficient without a lane in this run (NOROW, ATOMS32 from r20261021-070616-d06c, and STS32 and STG32 from
    r20261021-161119-62ac) the new hull's joint issue lines, and keeps their issue lines."""
    import shutil
    census = _load("census")
    dev = tmp_path / "sm_90"
    shutil.copytree(DC.parent / "machine_model" / "data" / "devices" / "sm_90", dev)

    def lines():
        with open(dev / "census.csv", newline="") as f:
            return {(ln["kind"], ln["row"]): ln["cycles"] for ln in csv.DictReader(f) if ln["kind"] != "UNI"}

    before = lines()
    run = _sm90_run(tmp_path / "r20261017-000000-abcd" / "out", 0.05)
    _rows(run / "mem", {"s_lde": {"0:LD.E*": 0.5}, "wgU_lde": {f"0:{WGI}": 128 / 4096, "1:LD.E*": 0.25}})
    _gate(run / "mem", {"wgU_lde": [(8, {WGI: 4}), (20, {"LD.E*": 16})]})
    assert census.main(["sm90", str(run), "--write", str(dev)]) == 0
    after = lines()
    ji = [r for k, r in after if k == "FFMA" and census.JI_ROW.fullmatch(r)]
    assert ji and all(after["UNLISTED", r] == after["FFMA", r] == after["NOROW", r] == after["ATOMS32", r]
                      == after["STS32", r] == after["STG32", r] for r in ji)
    assert all(after[k, "issue"] == before[k, "issue"] for k in ("UNLISTED", "ATOMS32", "STS32", "STG32", "NOROW"))
    assert census.analyze.lane_of("wgU_lds128_r32") == "lds128" and census.analyze.lane_of("wgmix_atome_r16") == "atome"


def _a2u_out(root: Path, i64_trips: int) -> Path:
    """sm100_a2u's output for a2_tc_i0 and a2_tc_i64: each a fixed loop (4 UTCIMMA and its uniform chain) and an until loop
    (the lanes and the stop poll, an LDS), with an instruction outside either, at a 2^20-cycle stop that holds the tensor
    pipe exactly (2048 trips of 4 MMAs at 128 clocks). a2_tc_i64's lanes run `i64_trips` trips summed over the 31 until
    warps; a2_tc_i0's run 32703, so its block issues 4 a cycle with one trip a warp dropped."""
    until = {"IADD3": 64, "FFMA": 63, "LDS": 1, "NOP": 1}
    fixed = {"a2_tc_i0": {"UTCIMMA": 4, "UIADD3": 1, "UISETP.NE.AND": 1, "NOP": 2},
             "a2_tc_i64": {"UTCIMMA": 4, "UIMAD": 32, "ULOP3.LUT": 32, "UIADD3": 1, "UISETP.NE.AND": 1}}
    trips = {"a2_tc_i0": 32703, "a2_tc_i64": i64_trips}
    out = root / "out"
    (out / "sass").mkdir(parents=True)
    (out / "probes").mkdir()
    sass, verdicts, times = [], [], []
    for name, fx in fixed.items():
        order = [until, fx] if name == "a2_tc_i64" else [fx, until]
        ops, loops = ["S2R R0, SR_TID.X"], []
        for counts in order:
            head = 16 * (len(ops) + 1)
            ops += [f"{op} R1" for op, n in counts.items() for _ in range(n)]
            loops.append({"head": hex(head), "tail": hex(16 * len(ops))})
        ops.append("EXIT")
        sass.append(f"\t\tFunction : {name}\n" + "\n".join(f"        /*{16 * (i + 1):04x}*/   {op} ;   /* 0x0 */"
                                                           for i, op in enumerate(ops)))
        verdicts.append({"binary": "sm100_a2u", "probe": name, "verdict": {"loops": loops}})
        times += [json.dumps({"op": "time", "probe": name, "iters": 2048, "trips": [trips[name]] * 2,
                              "stop": [2 ** 20] * 2, "sm_mhz": {"during": 1965, "after": 1965}, "throttle": "0x0"})] * 3
    (out / "sass" / "sm100_a2u.sass").write_text("\n".join(sass) + "\n")
    (out / "gate-build.json").write_text(json.dumps({"verdicts": verdicts}))
    (out / "probes" / "times.jsonl").write_text("\n".join(times) + "\n")
    return out


def test_a2u_issue_bound_and_uniform_price(tmp_path):
    """A2u's test on synthetic timings. Where each added uniform instruction takes an issue slot from the saturated lanes,
    the block issues one warp instruction an SMSP-cycle and the uniform price is 1; where the uniform datapath issues
    beside the lanes for free, the block issues 1 + 1/32 and A2u fails at price 0. One trip over 1 at the stop's boundary
    still holds. NOP is left out of `issue_low` and the instruction outside the loops out of every count; the until loop
    is told by its stop poll in either order."""
    a2 = _load("a2u")
    tight = a2.analyze(_a2u_out(tmp_path / "tight", 31679))
    base, i64 = tight["probes"]["a2_tc_i0"], tight["probes"]["a2_tc_i64"]
    assert base["until"] == {"IADD3": 64, "FFMA": 63, "LDS": 1, "NOP": 1} and i64["fixed_uniform_a_trip"] == 66
    assert tight["a2u_holds"] and abs(tight["max_issue_low"] - 1) < 1e-12 and abs(base["tensor"] - 1) < 1e-12
    assert abs(i64["uniform_price"] - 1) < 1e-12 and base["issue"] > base["issue_nonop"]
    free = a2.analyze(_a2u_out(tmp_path / "free", 32703))
    assert not free["a2u_holds"] and abs(free["max_issue_low"] - 1.03125) < 1e-12
    assert abs(free["probes"]["a2_tc_i64"]["uniform_price"]) < 1e-12
    edge = a2.analyze(_a2u_out(tmp_path / "edge", 31680))
    assert edge["a2u_holds"] and 1 < edge["max_issue_low"] < 1 + a2.TOL
