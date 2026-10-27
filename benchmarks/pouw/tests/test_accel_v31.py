"""ACCEL v3.1's CPU reference (benchmarks/pouw/accel_v31/v31.py): the binding identity, the row layout, and the stub
challenger's verdicts on an honest call and on calls that skip a noise tile or forge committed values."""
from __future__ import annotations

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


@pytest.mark.parametrize("kind", ["llm", "uniform", "zero"])
@pytest.mark.parametrize("lp,t,k", [(256, 3, 1536), (192, 3, 1536), (256, 3, 1280), (512, 2, 2048)])
def test_binding_identity_gives_o(kind, lp, t, k):
    sh = v31.Shape(128, 256, k, lp, t)
    _, _, _, c, f = _call(sh, kind)
    v = v31.values(sh, f)
    assert (v31.binding(sh, f, c, v.T_full) == v.O).all()
    assert (f.X + 256 * f.wX == f.Ad + v31.v_full(sh, c)).all() and np.abs(f.X).max() <= 128


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
    new = dict(line, wrap=2, wcw=256, ccta=1, maxnreg=208, pf=2)
    (tmp_path / "bench.jsonl").write_text("not json\n" + json.dumps(line) + "\n" + json.dumps(new) + "\n")
    assert v31.main(["v31", "table", str(tmp_path / "bench.jsonl")]) == 0
    old, row = capsys.readouterr().out.splitlines()[-2:]
    assert old.startswith("| 8b-qkv | 8B | 8192 x 6144 x 4096 | r8192 lo c100 d1 w1 k0 n208 |")
    assert row.startswith("| 8b-qkv | 8B | 8192 x 6144 x 4096 | r8192 lo c100 d1 w2/256 k1 n208 p2 |") and "| 1.70 | 1.00 |" in row
    assert "| -, -, 2.600, - | -, -, - |" in row


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
