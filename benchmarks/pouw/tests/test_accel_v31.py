"""Dither v3.1's CPU reference (benchmarks/pouw/accel_v31/v31.py): the binding identity, the row layout, and the stub
challenger's verdicts on an honest call and on calls that skip a noise tile or forge committed values; and Dither's
no-fold call (`Fill`) against dither.py's known answers, with its verdicts, sample check and table; and both shapes' bound
on t = min(L, K) / L'. perm_kernel launches within its launch bounds, and with VERITY_POUW_V31_LIB naming a libv31.so built
for the GPU in hand, the GPU forms W' at L' 1,024 and 2,048 under the permutation perm_kernel writes."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import settings

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "accel_v31"))
import fill_dither
import v31
import v31lib


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
    its A and B rows and the coins of those rows, gives the same prefix sums, O and leaves."""
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
    assert "| wrap4_kernel<128> | ? x ?, ? regs, 0 KB | 1 | 10.000 | 15.000 | 1.50 | 100% |" in out
    assert "| gemm_kernel<true> | ? x ?, ? regs, 0 KB | 1 | 20.000 | 25.000 | 1.25 |" in out
    assert "| leaf_persist<V31Row> | ? x ?, ? regs, 0 KB | 1 | 30.000 | 40.000 | 1.33 | 68% |" in out
    assert "steady period (GEMM end to GEMM end): median 40.000 ms, 40.000, 40.000, 40.000" in out
    assert "steady calls 120.000 ms: compute alone 0.0%, commit alone 4.2%, both 95.8%, neither 0.0%" in out


def test_timeline_of_fill_once_under_ltw(tmp_path, capsys):
    ms = 1_000_000
    spans = [   # (start, end, kernel) in ms: cuBLASLt's autotuning and the operands, two serial calls (the weight side in
        # the first only), then four overlapped calls, each commit beside the next call's GEMM (sched 1)
        (0, 2, "g"), (2, 4, "g"), (5, 6, "gen"),
        (10, 11, "b"), (11, 12, "n"), (12, 13, "b"), (13, 14, "n"), (14, 15, "p"), (15, 16, "a"), (16, 17, "fb"),
        (17, 27, "g"), (27, 28, "s"), (28, 33, "l"),
        (40, 41, "b"), (41, 42, "n"), (42, 43, "a"), (43, 53, "g"), (53, 54, "s"), (54, 59, "l"),
        (60, 61, "b"), (61, 62, "n"), (62, 63, "a"), (63, 73, "g"),
        (73, 74, "b"), (74, 75, "n"), (75, 76, "a"), (76, 87, "g"), (76, 77, "s"), (77, 83, "l"),
        (87, 88, "b"), (88, 89, "n"), (89, 90, "a"), (90, 101, "g"), (90, 91, "s"), (91, 97, "l"),
        (101, 102, "b"), (102, 103, "n"), (103, 104, "a"), (104, 115, "g"), (104, 105, "s"), (105, 111, "l"),
        (115, 116, "s"), (116, 121, "l"),
    ]
    names = {"g": "void cutlass::Kernel2<cutlass_80_tensorop_i16832gemm_s8_256x128_64x3_tn_align16>(T1::Params)",
             "gen": "void gen_kernel(signed char *, int, unsigned int, bool)", "b": "void bits_kernel(Seeds, int, unsigned long, unsigned char *)",
             "n": "void noise_kernel(Seeds, int, unsigned long, signed char *)", "p": "void perm_kernel(Seeds, int, unsigned short *)",
             "a": "void fill_a_kernel(const signed char *, const unsigned char *, const signed char *, signed char *, unsigned long, int, int, int)",
             "fb": "void fill_b_kernel(const signed char *, const unsigned char *, const signed char *, const unsigned short *, signed char *, unsigned long, int, int, int)",
             "s": "void salt_persist(Persist, int, const uint4 *, unsigned long, unsigned long, unsigned long *, unsigned long *, unsigned long *)",
             "l": "void leaf_persist<LtRow>(Persist, int, LtRow, const unsigned long *, const unsigned long *, unsigned long, unsigned long, unsigned long *)"}
    launch = {"g": (448, 256, 232, 0.098), "l": (376, 128, 96, 0.0)}
    rows = ['"Start (ns)","Duration (ns)","GrdX","BlkX","Reg/Trd","StcSMem (MB)","DymSMem (MB)","Strm","Name"']
    for s, e, n in spans:
        g, b, r, sm = launch.get(n, (1, 128, 32, 0.0))
        rows.append(f'{s * ms},{(e - s) * ms},{g},{b},{r},0.000,{sm},{13 if n in "sl" else 14},"{names[n]}"')
    (tmp_path / "t.csv").write_text("\n".join(rows) + "\n")
    assert v31.main(["v31", "timeline", str(tmp_path / "t.csv"), "--gemm", "gemm", "--from", "bits_kernel"]) == 0
    out = capsys.readouterr().out
    assert "6 calls (2 serial), steady overlapped calls 3..5" in out
    gemm = "cutlass::Kernel2<cutlass_80_tensorop_i16832gemm_s8_256x128_64x3_tn_align16>"
    assert f"| {gemm} | 448 x 256, 232 regs, 100 KB | 1 | 10.000 | 11.000 | 1.10 | 64% |" in out
    assert "| leaf_persist<LtRow> | 376 x 128, 96 regs, 0 KB | 1 | 5.000 | 6.000 | 1.20 | 71% |" in out
    assert "| bits_kernel | 1 x 128, 32 regs, 0 KB | 1 | 1.000 | 1.000 | 1.00 |" in out
    assert "fill_b_kernel" not in out
    assert "serial call: 19.000 ms; steady period (GEMM end to GEMM end): median 14.000 ms, 14.000, 14.000, 14.000" in out
    assert "steady calls 42.000 ms: compute alone 50.0%, commit alone 0.0%, both 50.0%, neither 0.0%" in out
    assert v31.main(["v31", "timeline", str(tmp_path / "t.csv"), "--gemm", "gemm", "--from", "wrap4_kernel"]) == 2
    capsys.readouterr()


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
# Known answers of verity/core/protocols/pouw/circuit/dither.py (`values_buffer`, `commit` under salts(Ks, 3 leaves),
# `pad_rows`, `Shape.domain`) on salt bytes 0..63, coin 64..127, operand-root receipt 128..191, Ks 192..255, call 3, and B
# (and the served A) from default_rng(11)
DITHER = {
    "pad": ((128, 128, 768, 256, 512, 1024), False, 64,
            "ca33cc7f8f39b599e894626cb0315b32399636eb01065485b117da6dd27b41f6cb2882fdab22f1be6c9187947a3ef3482378d8ddd41f45c74aad749c3a1a8625",
            "1e47bddbd672cda9e491849a4d36601324c6f53bbaf1846421d0fdc0cf661feaca9a94e4d551d983bd10d3a8b2d0880c87d3feefefeb3408400a435c42394358",
            "82bacbfb81da17626b1f693ee3b1d705c30d29c72fc0c18e4b97167747a3ca86d44765f37af617d96a8c01b5414f4bf7c3e0993cf8344052ea453bde1e73f050",
            "6421cee42b8709156a8a1cd0f3e635d30ffb46e24c9dc02bf237e7b945cafa5fded2781bbf7889c099204de7ab94bd73f1f3899bd1a2935097222d8dc95241f5"),
    "served-s1": ((128, 256, 1024, 128, 1024, 16384), True, 8,
                  "a74b8f1974578405b71e32f67a5a82e64270344f1d2e92b4eff7a91d8daefca6d52142d7efa0e6e861bd1acdce45ee0e3118d5b8ddd207333fc7c23f4e05b052",
                  "16d2aa6c56d874d28a29497562987e9e79aaddc63606eb6a185115f7bb1b580db0012fae9441a383041000e1e4d4165de20b7b526a48842ee4a6fd50b26794a2",
                  "",
                  "4adf24d3810b2fc8265abc628048a008b8953dfbc8c8d2ae3e903010d59ca57a9352da0982a04a65f8c378587ddd64a8acc4171ce0f5d975393764d4f5bd4a27"),
    "served-s4": ((128, 128, 2048, 128, 512, 8192), True, 20,
                  "1874fd6268c6e4b9cff84849612586c42971a64a7cdb7538e27b974237662d05f1c72ee420eb622d0e126593e7de6b57eca7307e960e008ca5a85788e5f7259a",
                  "8f6ab51000cd9b0886323c15a35bcb4004afb77ff1ae1e9e97b79c53c5dd6899b191e2d7f093f3513cb41e5c1dc9eb29da0d7a029d89126874ff758aa9f675fb",
                  "",
                  "a35c4d0d56e43d4cbcba147e3df21668b90691967dfd29b8e7a22871eabcc30532275b7a24e571c039a5479c9114c751374c231fd794f072578a55c2808f0b93"),
}


@pytest.mark.parametrize("case", sorted(DITHER))
def test_fill_is_dithers_no_fold_call(case):
    """The GPU is checked against v31's Fill; Fill against dither.py's own answers (pad, values, domain, root)."""
    (m, n, k, lp, L, row), served, leaves, values_h, root_h, pad_h, dom_h = DITHER[case]
    f = v31.Fill(m, n, k, lp, L, row, served)
    rng = np.random.default_rng(11)
    B = rng.integers(-128, 128, size=(n, k)).astype(np.int8)
    if served:
        A = rng.integers(-128, 128, size=(m, k)).astype(np.int8)
        seeds = v31.served_seeds(bytes(range(64, 128)), bytes(range(128, 192)), 3)
        assert set(seeds) == set(v31.STREAMS)
    else:
        seeds = v31.fill_seeds(bytes(range(64)), bytes(range(64, 128)))
        A = v31.pad_rows(seeds, range(m), k)
        assert hashlib.sha512(A.tobytes()).hexdigest() == pad_h
        assert np.array_equal(v31.pad_rows(seeds, range(5, 9), k), A[5:9])
    values = v31.fill_buffer(f, *v31.fill_values(f, A, B, v31.Coins.expand(f, seeds)))
    assert hashlib.sha512(values).hexdigest() == values_h
    assert f.leaves == leaves and f.domain().domain_id.hex() == dom_h
    assert v31.fill_commit(f, values, bytes(range(192, 256)), 3 * leaves).root.hex() == root_h
    assert v31.fill_seeds_from(v31.fill_seeds_bytes(seeds), served) == seeds


def _fill_answer(f: v31.Fill, cm, B: np.ndarray, A: np.ndarray | None):
    def answer(i: int, j: int, s: int) -> dict:
        pos = v31.fill_cells_positions(f, i, j, s)
        return {"rows": {p: (cm.row(p), cm.salt(p), cm.path(p)) for p in pos}, "B": B[j], "A": None if A is None else A[i]}
    return answer


@pytest.mark.parametrize("cheat", [None, "skip", "forge", "forgeO", "nodither"])
@pytest.mark.parametrize("served,L", [(False, 512), (True, 512), (True, 2048)])
def test_fill_verdicts(cheat, served, L):
    """Honest calls pass every proof block. Refused: a call that leaves a noise tile out of X W'^T, overwrites T_2 (or, at
    S = 1, O), overwrites O, or (served) leaves the dithers out of O."""
    f = v31.Fill(256, 128, 2048, 256, L, 1024, served)
    if not served and cheat in ("forgeO", "nodither"):
        pytest.skip("a filler call commits no O")
    rng = np.random.default_rng(5)
    B = rng.integers(-128, 128, size=(f.n, f.k)).astype(np.int8)
    if served:
        seeds = v31.served_seeds(rng.bytes(64), rng.bytes(64), 0)
        A = rng.integers(-128, 128, size=(f.m, f.k)).astype(np.int8)
    else:
        seeds = v31.fill_seeds(rng.bytes(64), rng.bytes(64))
        A = None
    Aop = v31.pad_rows(seeds, range(f.m), f.k) if A is None else A
    co = v31.Coins.expand(f, seeds)
    T, O = v31.fill_values(f, Aop, B, co, skip=5 if cheat == "skip" else None)
    if cheat == "forge" and f.W >= 2:
        T[1] ^= 0x0F0F
    if cheat == "forgeO" or (cheat == "forge" and f.W < 2):
        O[::3] ^= 0x10000
    if cheat == "nodither":
        O = v31.imatmul(Aop, B) & 0xFFFFFFFF
    cm = v31.fill_commit(f, v31.fill_buffer(f, T, O), rng.bytes(64), 7 * f.leaves)
    picks = v31.fill_picks(f, rng.bytes(64), 48)
    assert {s for _, _, s in picks} == set(range(1, f.S + 1))
    ok, fails, line = v31.fill_verdict(f, seeds, cm.root, picks, _fill_answer(f, cm, B, A), "t")
    refused = cheat is not None and not (cheat == "skip" and f.S == 1)   # at S = 1 the noise never enters O
    assert ok == (not refused), line
    assert line.startswith("REFUSE" if refused else "ACCEPT")
    if refused:
        assert all(("mod 2^32" if cheat in ("forgeO", "nodither") or f.S == 1 else "mod 2^16") in x for x in fails), fails
    bad = dict(_fill_answer(f, cm, B, A)(*picks[0]))
    p0 = next(iter(bad["rows"]))
    row, _, path = bad["rows"][p0]
    bad["rows"] = {**bad["rows"], p0: (row, bytes(64 * 3), path)}
    assert "don't open the root" in v31.fill_check(f, seeds, cm.root, *picks[0], bad)


def test_both_shapes_refuse_t_past_8():
    """t = min(L, K) / L' is at most 8 (dither/PROTOCOL.md, "The segment and the noise period"), in v3.1's Shape and in
    Fill: L 8,192 at L' 256 is refused at K 16,384 and L 4,096 at L' 512 passes, as do the defaults, L >= K at
    L' >= K / 8, and every call the GPU's gates (v31lib.GATES and PIPELINE) and fill_dither.py's cases build."""
    with pytest.raises(ValueError, match="at most 8"):
        v31.Shape(128, 128, 16384, 256, 32)
    with pytest.raises(ValueError, match="at most 8"):
        v31.Fill(128, 128, 16384, 256, 8192)
    assert v31.Shape(128, 128, 16384, 512, 8).L == v31.Fill(128, 128, 16384, 512, 4096).L == 4096
    assert v31.Shape(128, 128, 16384).t == 3 and v31.Fill(128, 128, 16384).L == 8 * v31.Fill(128, 128, 16384).lp
    assert v31.Fill(128, 128, 4096, 512, 8192).S == 1
    for line in v31lib.GATES.strip().splitlines():
        _, kind, *dims = line.split()[:8]
        v31.Fill(*map(int, dims), kind == "served")
    for line in fill_dither.CASES.strip().splitlines():
        v31.Fill(*map(int, line.split()[1:7]), line.split()[7] == "served")
    for dims, served in v31lib.PIPELINE:
        v31.Fill(*dims, served)


@pytest.mark.parametrize("served", [False, True])
def test_fill_sample_rebuilds_a_tile_and_the_params(tmp_path, capsys, served):
    """`v31 fill-bench`'s sample files: a tile's values and leaves from its A and B rows and those rows' coins alone, and
    the params file as fill-params writes it."""
    f = v31.Fill(256, 384, 1024, 256, 512, 2048, served)
    rng = np.random.default_rng(2)
    B = rng.integers(-128, 128, size=(f.n, f.k)).astype(np.int8)
    if served:
        seeds = v31.served_seeds(rng.bytes(64), rng.bytes(64), 5)
        A = rng.integers(-127, 128, size=(f.m, f.k)).astype(np.int8)
    else:
        seeds = v31.fill_seeds(rng.bytes(64), rng.bytes(64))
        A = v31.pad_rows(seeds, range(f.m), f.k)
    ks, off = rng.bytes(64), 5 * f.leaves
    cm = v31.fill_commit(f, v31.fill_buffer(f, *v31.fill_values(f, A, B, v31.Coins.expand(f, seeds))), ks, off)
    args = ["--served"] if served else []
    assert v31.main(["v31", "fill-params", str(tmp_path / "p"), "--L", "512", "--row", "2048", "x:256:384:1024", *args]) == 0
    name = v31.fill_params_name(256, 384, 1024, 256, 512, 2048, served)
    assert (tmp_path / "p" / name).stat().st_size == len(v31.fill_params(f, ks))
    s = tmp_path / "s"
    s.mkdir()
    (s / "seeds.bin").write_bytes(v31.fill_seeds_bytes(seeds))
    (s / "params.bin").write_bytes(v31.fill_params(f, ks))
    tiles = [0, 4]
    (s / "sample.json").write_text(json.dumps({"m": f.m, "n": f.n, "k": f.k, "lp": f.lp, "L": f.L, "row": f.row, "o": served,
                                               "off": off, "tiles": tiles, "root": cm.root.hex()}))
    tb = f.lpt * f.row
    for i, tile in enumerate(tiles):
        tm, tn = divmod(tile, f.n // v31.TILE)
        (s / f"t{i}_A.bin").write_bytes(A[tm * 128:(tm + 1) * 128].tobytes())
        (s / f"t{i}_B.bin").write_bytes(B[tn * 128:(tn + 1) * 128].tobytes())
        (s / f"t{i}_words.bin").write_bytes(cm.values[tile * tb:(tile + 1) * tb])
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
    later = dict(line, ltw=1, pf=2, gamma_ovl_nomem=1.02)
    served_arms = {a: v for a, v in arms.items() if a != "pad"}
    served = dict(line, o=True, row=16384, arms=served_arms, gamma_ovl_pad=None, gamma_ovl_w=1.25, gamma_ovl_w_cutlass=1.3)
    fused = dict(served, arms=dict(served_arms, wgl={"ms": 2.5, "mhz": 1980}), gamma_ovl_w_wgl=1.25 * 2.0 / 2.5, wg=1, rp=4,
                 wbn=256, fused_leaves=3 * 28672, o_leaves=4 * 28672, fused_blocks=(3 * 129 + 40) * 28672, row_blocks=129)
    no_row = dict(fused, fused_leaves=0, fused_blocks=129 * 28672, lminb=4, sahead=1)
    words = dict(fused, rp=5, S=4, W=3, wgw=1, r5w=0, leaves=10 * 28672)
    tst = dict(words, r5w=1, tst=1)
    lines = (line, later, served, fused, no_row, words, tst)
    (tmp_path / "b.jsonl").write_text("x\n" + "\n".join(json.dumps(x) for x in lines) + "\n")
    assert v31.main(["v31", "fill-table", str(tmp_path / "b.jsonl")]) == 0
    row, row2, row3, row4, row5, row6, row7 = capsys.readouterr().out.splitlines()[-7:]
    assert "| served r16384 hi k1 s1 wg256 rp5 r5w0 |" in row6 and row6.endswith("| 30.0% / 33.1% |")
    assert "| served r16384 hi k1 s1 wg256 rp5 tst |" in row7
    assert row.endswith("| - | - | - |") and row3.endswith("| - | - | - |")
    assert "| served r16384 hi k1 s1 wg256 rp4 |" in row4 and row4.endswith("| 2.500 | 1.000 | 75.0% / 82.8% |")
    assert row5.endswith("| 0.0% / 25.0% |") and "| served r16384 hi k1 s1 wg256 rp4 m4 sa |" in row5
    assert row.startswith("| 8b-up | 16384 x 28672 x 4096 | 4096 | 1 | r1024 hi k1 s1 | 2.000 | 1.900 | 2.100 | 1.050 | 1.105 |"
                          " 1.060 | - |")
    assert "| 1.070 | - | 1.700 |" in row
    assert "| 95.2% | 0.010, 0.020, 0.030, 0.100, 0.200, 2.050, 1.200 |" in row
    assert "| r1024 hi k1 s1 p2 ltw |" in row2 and "| 1.070 | 1.020 | 1.700 |" in row2
    assert "| served r16384 hi k1 s1 |" in row3 and "| 1.250 | 1.300 | - | - | 1.700 |" in row3
    assert "| 80.0% | 0.010, 0.020, -, 0.100, 0.200, 2.050, 1.200 |" in row3


def test_fill_table_sets_the_commits_parts_against_the_ruler_and_the_bar(tmp_path, capsys):
    """A line that carries the commit's parts gets a second table: each part's compressions a SM-cycle, over the build's
    ruler (of_sha) and, given --bar, over the card's bar; its variant names the leaf path (nofull: rows hashed through the
    padding checks; mN and smN: the leaf and salt kernels at MINB N; cf: each group's levels folded by its own CTA rather
    than by level kernels; mkN: the salt kernel's mask N). A line that timed only some arms (V31_ARMS) shows "-" for the
    rest."""
    arms = {a: {"ms": ms, "mhz": 2000} for a, ms in (("lt", 2.0), ("ovl", 2.1), ("commit", 1.2), ("leaf", 1.0),
                                                      ("leaf_nomem", 0.9), ("salt", 0.2), ("sha", 0.5))}
    line = {"fill": True, "shape": "8b-up", "m": 16384, "n": 28672, "k": 4096, "L": 4096, "S": 1, "row": 16384, "o": True,
            "cprio": "hi", "ccta": 4, "sched": 1, "arms": arms, "gamma_ovl": 1.05, "gamma_ovl_w": 1.06, "gamma_call": 1.7,
            "gamma_gemm": 1.025, "full": 1, "lminb": 6, "lfold": 0, "comp": {"sha": 1e8, "leaf": 3e8, "salt": 6e7},
            "per_sm_cycle": {"sha": 0.018, "leaf": 0.0171, "leaf_nomem": 0.0175, "salt": 0.0144, "commit": None},
            "of_sha": {"leaf": 0.95, "leaf_nomem": 0.9722, "salt": 0.8, "commit": None}}
    before = dict(line, full=0, lminb=5, sminb=5, lfold=1)
    only = dict(line, mask=2, arms={x: arms[x] for x in ("salt", "sha")}, gamma_ovl=None, gamma_ovl_w=None, gamma_call=None,
                gamma_gemm=None)
    (tmp_path / "b.jsonl").write_text("\n".join(json.dumps(x) for x in (line, before, only)) + "\n")
    assert v31.main(["v31", "fill-table", str(tmp_path / "b.jsonl"), "--bar", "0.018"]) == 0
    out = capsys.readouterr().out.splitlines()
    parts = out[out.index("| shape | variant | sha / SM-cycle | part | ms | MHz | compressions | / SM-cycle | of sha | of bar |"):]
    assert out[4].startswith("| 8b-up | 16384 x 28672 x 4096 | 4096 | 1 | served r16384 hi k4 s1 m6 mk2 | - | - | - | - |")
    assert len(parts) == 2 + 3 * 4
    assert parts[12].endswith(" s1 m6 mk2 | 0.01800 | salt | 0.200 | 2000 | 60000000 | 0.01440 | 0.800 | 0.800 |")
    leaf, nomem, salt, commit = parts[2:6]
    assert leaf == "| 8b-up | served r16384 hi k4 s1 m6 | 0.01800 | leaf | 1.000 | 2000 | 300000000 | 0.01710 | 0.950 | 0.950 |"
    assert nomem.endswith("| leaf_nomem | 0.900 | 2000 | 300000000 | 0.01750 | 0.972 | 0.972 |")
    assert salt.endswith("| salt | 0.200 | 2000 | 60000000 | 0.01440 | 0.800 | 0.800 |")
    assert commit.endswith("| commit | 1.200 | 2000 | 360000000 | - | - | - |")
    assert "| served r16384 hi k4 s1 sm5 cf nofull | 0.01800 | leaf |" in parts[6]
    v31.main(["v31", "fill-table", str(tmp_path / "b.jsonl")])
    assert capsys.readouterr().out.splitlines()[-12].endswith("| 0.01710 | 0.950 | - |")


def test_perm_kernel_launches_within_its_bounds():
    """perm_kernel's SHA-512 takes 90 registers a thread, so a CTA of 1,024 never launches and pi stays unwritten: every
    launch takes PERM_THREADS(L') threads, at most the kernel's __launch_bounds__ of 512."""
    src = (Path(v31.__file__).parent / "v31.cu").read_text()
    bound = int(re.search(r"__launch_bounds__\((\d+)\) perm_kernel\(", src).group(1))
    cap = int(re.search(r"PERM_THREADS\(int lp\) \{ return lp < (\d+) \? lp : \1; \}", src).group(1))
    assert cap <= bound <= 512
    launches = re.findall(r"perm_kernel<<<[^,]+, ([^,]+),", src)
    assert launches and set(launches) == {"PERM_THREADS(lp)"}


@pytest.mark.skipif(not settings.V31_LIB, reason="VERITY_POUW_V31_LIB names a libv31.so (v31lib.build_lib) for the GPU in hand "
                    "(about 5 s)")
@pytest.mark.parametrize("name", ["f-lp1024", "c-skip-lp1024", "s-lp1024", "f-lp2048"])
def test_gpu_forms_w_prime_under_the_written_permutation(name):
    """At L' >= 1,024 the GPU forms W' under the pi_tau perm_kernel writes: its values and every tree level are the
    reference's, an honest call is accepted and one that skips a noise tile refused. Had perm_kernel's CTA not launched,
    v31_prove would refuse the call (a launch failed) rather than commit W' under an unwritten pi."""
    _, kind, *dims, a_kind, cheat = next(g.split() for g in v31lib.GATES.strip().splitlines() if g.split()[0] == name)
    rep = v31lib.gate_case(v31lib.Lib(settings.V31_LIB), name, kind == "served", *map(int, dims), a_kind, cheat, 64,
                           np.random.default_rng(7))
    assert rep["exact"] and rep["good"], rep
