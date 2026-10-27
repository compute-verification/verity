"""ACCEL v3.1's CPU reference (benchmarks/pouw/accel_v31/v31.py): the binding identity, the row layout, and the stub
challenger's verdicts on an honest call and on calls that skip a noise tile or forge committed values; and Dither's
no-fold call (`Fill`) against dither.py's known answers, with its verdicts, sample check and table."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "accel_v31"))
import v31


def _call(sh: v31.Shape, kind: str = "llm", seed: int = 3):
    A, B = v31.operands(sh, np.random.default_rng(seed), kind)
    seeds = {s: bytes([seed, i]) * 32 for i, s in enumerate(v31.STREAMS)}
    c = v31.Coins.expand(sh, seeds)
    return A, B, seeds, c, v31.form(sh, A, B, c)


@pytest.mark.parametrize("kind", ["llm", "uniform", "zero", "max"])
@pytest.mark.parametrize("lp,t,k", [(256, 3, 1536), (192, 3, 1536), (256, 3, 1280), (512, 2, 2048)])
def test_binding_identity_gives_o(kind, lp, t, k):
    sh = v31.Shape(128, 256, k, lp, t)
    _, _, _, c, f = _call(sh, kind)
    v = v31.values(sh, f)
    assert (v31.binding(sh, f, c, v.T_full) == v.O).all()
    assert (f.X + 256 * f.wX == f.Ad + v31.v_full(sh, c)).all() and np.abs(f.X).max() <= 128


def test_max_operands_pass_256_wraps_in_a_noise_tile():
    """g-max512's point: wrap3 and wrap4 hold a column's sums in 16-bit lanes, so they flush within a noise tile."""
    sh = v31.Shape(128, 256, 2048, 512, 2)
    _, _, _, _, f = _call(sh, "max")
    per_tile = (f.wW != 0).reshape(sh.n, sh.k // sh.lp, sh.lp).sum(axis=2)
    assert (per_tile > 256).mean() > 0.25


def test_coins_have_their_ranges():
    sh = v31.Shape(128, 128, 768)
    _, _, _, c, _ = _call(sh)
    assert c.U.min() == -80 and c.U.max() == 80 and c.V.min() == -80 and c.V.max() == 80
    assert set(np.unique(c.dA)) == {0, 1}
    assert all(sorted(p) == list(range(sh.lp)) for p in c.perm) and len({tuple(p) for p in c.perm}) == len(c.perm)


def test_rows_hold_each_cells_words_where_word_leaf_says():
    sh = v31.Shape(256, 384, 1536, row=2048)
    _, _, _, _, f = _call(sh)
    v = v31.values(sh, f)
    words, obuf = v31.words_buffer(sh, v.T), v31.o_buffer(v.O)
    assert len(words) == sh.tiles * sh.S * 32768 and len(obuf) == 4 * sh.m * sh.n
    rng = np.random.default_rng(0)
    for _ in range(200):
        i, j, s = int(rng.integers(sh.m)), int(rng.integers(sh.n)), int(rng.integers(1, sh.S + 1))
        p, o = sh.word_leaf(i, j, s)
        assert int.from_bytes(v31.row_bytes(sh, words, obuf, p)[o:o + 2], "little") == v.T[s - 1, i, j]
        p, o = sh.o_leaf(i, j)
        assert int.from_bytes(v31.row_bytes(sh, words, obuf, p)[o:o + 4], "little") == v.O[i, j]
    assert all(len(v31.row_bytes(sh, words, obuf, p)) == sh.row for p in range(sh.leaves))


@pytest.mark.parametrize("cheat", [None, "skip:4", "forge:1:0.5", "forge:2:0.5", "forgeO:0.5"])
def test_verdicts(cheat):
    sh = v31.Shape(256, 256, 1536)
    A, B, _, _, _ = _call(sh)
    ch = v31.StubChallenger(b"\x07" * 64)
    seeds = ch.seeds(0, v31.operand_roots(A, B))
    prover, _ = v31.prove(sh, A, B, seeds, b"\x09" * 64, 0, cheat)
    ok, _, line = v31.verdict(sh, ch, 0, prover, 48)
    assert ok == (cheat is None), line
    assert line.startswith("ACCEPT" if cheat is None else "REFUSE")


@pytest.mark.parametrize("lp", [256, 192])
def test_a_tiles_coins_are_the_calls_rows(lp):
    sh = v31.Shape(384, 256, 1536, lp)
    seeds = {s: bytes([5, i]) * 32 for i, s in enumerate(v31.STREAMS)}
    c = v31.Coins.expand(sh, seeds)
    r = v31.Coins.rows(sh, seeds, range(256, 384), range(128, 256))
    assert (r.dA == c.dA[256:]).all() and (r.V == c.V[256:]).all()
    assert (r.dB == c.dB[128:]).all() and (r.U == c.U[128:]).all() and (r.perm == c.perm).all()


def _gpu_dir(tmp_path, m=256, n=256, k=1536, cheat=None):
    """`gpu-prep`'s directory with an exact GPU's outputs (the reference's, in `v31 gate`'s files) and its control."""
    d = tmp_path / "gate"
    args = ["v31", "gpu-prep", str(d), str(m), str(n), str(k)] + (["--cheat", cheat] if cheat else [])
    assert v31.main(args) == 0
    sh = v31.Shape(m, n, k)
    A = np.frombuffer((d / "A.bin").read_bytes(), np.int8).reshape(m, k)
    B = np.frombuffer((d / "B.bin").read_bytes(), np.int8).reshape(n, k)
    seeds = v31.seeds_from((d / "seeds.bin").read_bytes())
    ks = (d / "params.bin").read_bytes()[4 + 3 * 128 + 64:4 + 3 * 128 + 128]
    prover, _ = v31.prove(sh, A, B, seeds, ks, 0, cheat)
    (d / "words.bin").write_bytes(prover.cm.words)
    (d / "O.bin").write_bytes(prover.cm.obuf)
    (d / "levels.bin").write_bytes(v31.levels_bytes(prover.cm.levels))
    (d / "control").mkdir()
    for name in ("words", "O", "levels"):
        (d / "control" / f"{name}.bin").write_bytes(b"\xa5" * len((d / f"{name}.bin").read_bytes()))
    return d, sh, A, B, seeds, ks, prover


def test_gpu_check_passes_an_exact_gpu_and_refuses_a_wrong_word(tmp_path, capsys):
    d, *_ = _gpu_dir(tmp_path)
    assert v31.main(["v31", "gpu-check", str(d), "--q", "16"]) == 0
    assert capsys.readouterr().out.splitlines()[-1].startswith("ACCEPT")
    w = bytearray((d / "words.bin").read_bytes())
    w[2 * 1000] ^= 1
    (d / "words.bin").write_bytes(bytes(w))
    assert v31.main(["v31", "gpu-check", str(d), "--q", "16"]) == 1


def test_gpu_check_wants_a_cheat_refused(tmp_path, capsys):
    d, *_ = _gpu_dir(tmp_path, cheat="forgeO:1")
    assert v31.main(["v31", "gpu-check", str(d), "--q", "48"]) == 0
    assert capsys.readouterr().out.splitlines()[-1].startswith("REFUSE")


def test_gpu_sample_rebuilds_a_tile_from_its_rows(tmp_path):
    """`v31 bench`'s sample files (two tiles of the checked call) from the whole call's reference: the tile alone, from
    its A and B rows and the coins of those rows, gives the same words, O and leaves."""
    _, sh, A, B, seeds, ks, prover = _gpu_dir(tmp_path, 384, 256, 1280)
    s = tmp_path / "sample"
    s.mkdir()
    (s / "seeds.bin").write_bytes(v31.seeds_bytes(seeds))
    (s / "params.bin").write_bytes(v31.params_file(sh, ks))
    tiles = [0, 5]
    (s / "sample.json").write_text(json.dumps({"m": sh.m, "n": sh.n, "k": sh.k, "lp": sh.lp, "t": sh.t, "row": sh.row,
                                               "tiles": tiles, "root": prover.cm.root.hex()}))
    O = np.frombuffer(prover.cm.obuf, "<u4").reshape(sh.m, sh.n)
    for i, tile in enumerate(tiles):
        tm, tn = divmod(tile, sh.n // v31.TILE)
        (s / f"t{i}_A.bin").write_bytes(A[tm * 128:(tm + 1) * 128].tobytes())
        (s / f"t{i}_B.bin").write_bytes(B[tn * 128:(tn + 1) * 128].tobytes())
        (s / f"t{i}_words.bin").write_bytes(prover.cm.words[tile * sh.S * 32768:(tile + 1) * sh.S * 32768])
        (s / f"t{i}_O.bin").write_bytes(np.ascontiguousarray(O[tm * 128:(tm + 1) * 128, tn * 128:(tn + 1) * 128]).tobytes())
        (s / f"t{i}_leaves.bin").write_bytes(b"".join(prover.cm.levels[0][tile * sh.lpt:(tile + 1) * sh.lpt]))
    assert v31.main(["v31", "gpu-sample", str(s)]) == 0
    lv = bytearray((s / "t1_leaves.bin").read_bytes())
    lv[64 * 7] ^= 1
    (s / "t1_leaves.bin").write_bytes(bytes(lv))
    assert v31.main(["v31", "gpu-sample", str(s)]) == 1


def test_table_sets_gamma_beside_d6(tmp_path, capsys):
    arms = {a: {"ms": ms, "mhz": 2000} for a, ms in (("lt", 2.0), ("ovl", 3.4), ("call", 4.0), ("gemm", 2.6))}
    line = {"shape": "8b-qkv", "m": 8192, "n": 6144, "k": 4096, "row": 8192, "cprio": "lo", "carve": 100, "dyn": 1,
            "arms": arms, "gamma_ovl": 1.7, "gamma_call": 2.0, "gamma_gemm": 1.3, "wrap_rate_A": 0.001, "wrap_rate_B": 0.02}
    new = dict(line, wrap=4, wcw=256, ccta=1, maxnreg=208, pf=2, arms=dict(arms, cutlass={"ms": 1.7, "mhz": 2000}),
               gamma_ovl_cutlass=2.0)
    (tmp_path / "bench.jsonl").write_text("not json\n" + json.dumps(line) + "\n" + json.dumps(new) + "\n")
    assert v31.main(["v31", "table", str(tmp_path / "bench.jsonl")]) == 0
    old, row = capsys.readouterr().out.splitlines()[-2:]
    assert old.startswith("| 8b-qkv | 8B | 8192 x 6144 x 4096 | r8192 lo c100 d1 w1 k0 n208 | 2.000 | - | 3.400 | 1.70 | - |")
    assert row.startswith("| 8b-qkv | 8B | 8192 x 6144 x 4096 | r8192 lo c100 d1 w4/256 k1 n208 p2 | 2.000 | 1.700 | 3.400 | 1.70 "
                          "| 2.00 |") and "| 1.70 | 1.00 |" in row
    assert "| -, -, 2.600, - | -, -, - |" in row


def test_timeline_splits_calls_and_measures_the_overlap(tmp_path, capsys):
    ms = 1_000_000
    spans = [   # (start, end, kernel) in ms: two serial calls, then four overlapped, call 5's compute 5 ms late
        (0, 10, "w"), (10, 30, "g"), (30, 60, "l"),
        (60, 70, "w"), (70, 90, "g"), (90, 120, "l"),
        (120, 130, "w"), (130, 150, "g"), (150, 190, "l"),
        (150, 165, "w"), (165, 190, "g"), (190, 230, "l"),
        (190, 205, "w"), (205, 230, "g"), (230, 270, "l"),
        (235, 245, "w"), (245, 270, "g"), (270, 300, "l"),
    ]
    names = {"w": "void wrap4_kernel<(int)128>(WrapArgs)", "g": "void gemm_kernel<(bool)1>(GArgs)",
             "l": "void leaf_persist<V31Row>(Persist, Domain, V31Row)"}
    rows = ['"Start (ns)","Duration (ns)","Strm","Name"', f'{5 * ms},{ms},7,"[CUDA memset]"']
    rows += [f'{s * ms},{(e - s) * ms},{13 if n == "l" else 14},"{names[n]}"' for s, e, n in spans]
    (tmp_path / "t.csv").write_text("\n".join(rows) + "\n")
    assert v31.main(["v31", "timeline", str(tmp_path / "t.csv")]) == 0
    out = capsys.readouterr().out
    assert "| wrap4_kernel<128> | 1 | 10.000 | 15.000 | 1.50 | 100% |" in out
    assert "| gemm_kernel<true> | 1 | 20.000 | 25.000 | 1.25 |" in out
    assert "| leaf_persist<V31Row> | 1 | 30.000 | 40.000 | 1.33 | 68% |" in out
    assert "steady period (GEMM end to GEMM end): median 40.000 ms, 40.000, 40.000, 40.000" in out
    assert "steady window 120.000 ms: compute alone 0.0%, commit alone 4.2%, both 95.8%, neither 0.0%" in out


def test_challenger_keeps_the_order_and_never_redraws():
    sh = v31.Shape(128, 128, 768)
    ch = v31.StubChallenger(os.urandom(64))
    with pytest.raises(RuntimeError):
        ch.check(0, b"r" * 64, sh, 4)
    ch.seeds(0, b"o" * 128)
    with pytest.raises(RuntimeError):
        ch.seeds(0, b"o" * 128)
    ch.check(0, b"r" * 64, sh, 4)
    with pytest.raises(RuntimeError):
        ch.check(0, b"r" * 64, sh, 4)


# ---------------------------------------------------------------------------------------------------- Dither's no-fold call
# Known answers of dither.py (#4974 at d848cc79d7: `words`, `commit` under prf_salt(call_key(Ks, 3), p), `pad_rows`,
# `Shape.domain`) on salt bytes 0..63, coin 64..127, Ks 128..191 and B (and the served A) from default_rng(11)
DITHER = {
    "pad": ((128, 128, 768, 256, 512, 1024), False, 64,
            "ca33cc7f8f39b599e894626cb0315b32399636eb01065485b117da6dd27b41f6cb2882fdab22f1be6c9187947a3ef3482378d8ddd41f45c74aad749c3a1a8625",
            "87e1ecd5e70b7752c7fa17f4b612e2ac7aa5d8ce706cd35b11a40f81b6bee445f68510c850903fecc072bd25e0f0681b0e122c1e7311a0cb09d37bbc101ee6b1",
            "82bacbfb81da17626b1f693ee3b1d705c30d29c72fc0c18e4b97167747a3ca86d44765f37af617d96a8c01b5414f4bf7c3e0993cf8344052ea453bde1e73f050",
            "b6e4f4aafc1f927063fb2ebc3dddddebae81b983de66e896d61d72d4ce2e4c08f09faaa18b95b2a61e3ca43396259f5a7ca5250a13f9adc951edeb8c8d474406"),
    "served": ((128, 256, 1024, 128, 1024, 2048), True, 32,
               "04e8ad032050e4cc8cd8499402bab20d2910cfd6ecf46858bb4518d2918fcb9635595bd9b4dd2de3683bbe93109a6b1884c2ba7f986ce9e48c433a19700fb911",
               "6190dc3a8eb662afbfab9ed0f971aff78c7aaba9951c613acbe78b1e980b9565c8bebca831384589b00ec65b2e2c594b629e4293da4d778fef87fc4d08e0124c",
               "8d9e34924189d8061168a0d39b53b5fc6de70863711c4e0a160438d4c0331707b15131d29b63abdc96f741e3b630c9a1909de9c3557494d7a218d05fe690fd8e",
               "f634c088c0c0bace599978a0ccb01996b373dc5f24476f52f75bac97d6cc2a8ca437974e9e725408d86b0a33e18565981c928d4f0842cbb7e896bdf91fa29939"),
}


@pytest.mark.parametrize("case", sorted(DITHER))
def test_fill_is_dithers_no_fold_call(case):
    """The GPU is checked against v31's Fill; Fill against dither.py's own answers (pad, words, domain, root)."""
    (m, n, k, lp, L, row), served, leaves, words_h, root_h, pad_h, dom_h = DITHER[case]
    f = v31.Fill(m, n, k, lp, L, row)
    rng = np.random.default_rng(11)
    B = rng.integers(-128, 128, size=(n, k)).astype(np.int8)
    A = rng.integers(-128, 128, size=(m, k)).astype(np.int8) if served else None
    seeds = v31.call_seeds(bytes(range(64)), bytes(range(64, 128)))
    pad = v31.pad_rows(seeds, range(m), k)
    assert hashlib.sha512(pad.tobytes()).hexdigest() == pad_h
    assert np.array_equal(v31.pad_rows(seeds, range(5, 9), k), pad[5:9])
    words = v31.words_buffer(f, v31.fill_words(f, pad if A is None else A, B, v31.Coins.expand(f, seeds)))
    assert hashlib.sha512(words).hexdigest() == words_h
    assert f.leaves == leaves and f.domain().domain_id.hex() == dom_h
    assert v31.fill_commit(f, words, v31.call_key(bytes(range(128, 192)), 3)).root.hex() == root_h
    assert v31.fill_seeds_from(v31.fill_seeds_bytes(seeds)) == seeds


def _fill_answer(f: v31.Fill, cm, B: np.ndarray, A: np.ndarray | None):
    def answer(i: int, j: int, s: int) -> dict:
        pos = v31.fill_cells_positions(f, i, j, s)
        return {"rows": {p: (cm.row(p), cm.salt(p), cm.path(p)) for p in pos}, "B": B[j], "A": None if A is None else A[i]}
    return answer


@pytest.mark.parametrize("cheat", [None, "skip", "forge"])
@pytest.mark.parametrize("served", [False, True])
def test_fill_verdicts(cheat, served):
    """Honest calls pass every proof block; a call that leaves a noise tile out of X W'^T, or overwrites T_2, is refused."""
    f = v31.Fill(256, 128, 2048, 256, 512)
    rng = np.random.default_rng(5)
    B = rng.integers(-128, 128, size=(f.n, f.k)).astype(np.int8)
    seeds = v31.call_seeds(rng.bytes(64), rng.bytes(64))
    A = rng.integers(-128, 128, size=(f.m, f.k)).astype(np.int8) if served else None
    T = v31.fill_words(f, v31.pad_rows(seeds, range(f.m), f.k) if A is None else A, B, v31.Coins.expand(f, seeds),
                       skip=5 if cheat == "skip" else None)
    if cheat == "forge":
        T[1] ^= 0x0F0F
    cm = v31.fill_commit(f, v31.words_buffer(f, T), v31.call_key(rng.bytes(64), 0))
    picks = v31.fill_picks(f, rng.bytes(64), 48)
    assert {s for _, _, s in picks} == set(range(1, f.S + 1))
    ok, fails, line = v31.fill_verdict(f, seeds, cm.root, picks, _fill_answer(f, cm, B, A), "t")
    assert ok == (cheat is None), line
    assert line.startswith("ACCEPT" if cheat is None else "REFUSE")
    if cheat:
        assert all("mod 2^16" in x for x in fails)
    bad = dict(_fill_answer(f, cm, B, A)(*picks[0]))
    p0 = next(iter(bad["rows"]))
    row, _, path = bad["rows"][p0]
    bad["rows"] = {**bad["rows"], p0: (row, bytes(64 * 3), path)}
    assert "don't open the root" in v31.fill_check(f, seeds, cm.root, *picks[0], bad)


def test_fill_sample_rebuilds_a_tile_and_the_params(tmp_path, capsys):
    """`v31 fill-bench`'s sample files: a tile's words and leaves from its A and B rows and those rows' coins alone, and the
    params file as fill-params writes it."""
    f = v31.Fill(256, 384, 1024, 256, 512, 2048)
    rng = np.random.default_rng(2)
    B = rng.integers(-128, 128, size=(f.n, f.k)).astype(np.int8)
    seeds = v31.call_seeds(rng.bytes(64), rng.bytes(64))
    ks, key = rng.bytes(64), rng.bytes(64)
    A = v31.pad_rows(seeds, range(f.m), f.k)
    cm = v31.fill_commit(f, v31.words_buffer(f, v31.fill_words(f, A, B, v31.Coins.expand(f, seeds))), key)
    assert v31.main(["v31", "fill-params", str(tmp_path / "p"), "--L", "512", "--row", "2048", "x:256:384:1024"]) == 0
    assert (tmp_path / "p" / v31.fill_params_name(256, 384, 1024, 256, 512, 2048)).stat().st_size == len(v31.fill_params(f, ks))
    s = tmp_path / "s"
    s.mkdir()
    (s / "seeds.bin").write_bytes(v31.fill_seeds_bytes(seeds))
    (s / "ks.bin").write_bytes(key)
    (s / "params.bin").write_bytes(v31.fill_params(f, ks))
    tiles = [0, 4]
    (s / "sample.json").write_text(json.dumps({"m": f.m, "n": f.n, "k": f.k, "lp": f.lp, "L": f.L, "row": f.row,
                                               "tiles": tiles, "root": cm.root.hex()}))
    for i, tile in enumerate(tiles):
        tm, tn = divmod(tile, f.n // v31.TILE)
        (s / f"t{i}_A.bin").write_bytes(A[tm * 128:(tm + 1) * 128].tobytes())
        (s / f"t{i}_B.bin").write_bytes(B[tn * 128:(tn + 1) * 128].tobytes())
        (s / f"t{i}_words.bin").write_bytes(cm.words[tile * f.S * 32768:(tile + 1) * f.S * 32768])
        (s / f"t{i}_leaves.bin").write_bytes(b"".join(cm.levels[0][tile * f.lpt:(tile + 1) * f.lpt]))
    assert v31.main(["v31", "fill-sample", str(s)]) == 0
    lv = bytearray((s / "t1_leaves.bin").read_bytes())
    lv[64 * 3] ^= 1
    (s / "t1_leaves.bin").write_bytes(bytes(lv))
    assert v31.main(["v31", "fill-sample", str(s)]) == 1
    capsys.readouterr()


def test_fill_table(tmp_path, capsys):
    names = ("lt", "cutlass", "coins", "coins_w", "pad", "form_a", "form_b", "gemm", "commit", "ovl")
    arms = {a: {"ms": ms, "mhz": 2000} for a, ms in zip(names, (2.0, 1.9, 0.01, 0.02, 0.03, 0.1, 0.2, 2.05, 1.2, 2.1))}
    line = {"fill": True, "shape": "8b-up", "m": 16384, "n": 28672, "k": 4096, "L": 4096, "S": 1, "row": 1024, "cprio": "hi",
            "ccta": 1, "sched": 1, "arms": arms, "gamma_ovl": 1.05, "gamma_ovl_cutlass": 2.1 / 1.9, "gamma_ovl_w": 1.06,
            "gamma_ovl_pad": 1.07, "gamma_call": 1.7, "gamma_gemm": 1.025}
    (tmp_path / "b.jsonl").write_text("x\n" + json.dumps(line) + "\n")
    assert v31.main(["v31", "fill-table", str(tmp_path / "b.jsonl")]) == 0
    row = capsys.readouterr().out.splitlines()[-1]
    assert row.startswith("| 8b-up | 16384 x 28672 x 4096 | 4096 | 1 | r1024 hi k1 s1 | 2.000 | 1.900 | 2.100 | 1.050 | 1.105 |")
    assert "| 95.2% | 0.010, 0.020, 0.030, 0.100, 0.200, 2.050, 1.200 |" in row
