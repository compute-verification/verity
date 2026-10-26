"""`s8_simple.linear` and its check's CPU twin (`check.twin`) against `ScaledMmS8Coordinate_v1{K, GEMM_SM90_S8}`'s reference
evaluator, on the check's special codes and scales, and the check itself run end to end on the CPU at tiny shapes."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from verity.ml.circuits import evaluation as E
from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops import gemm as G

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import check as C  # noqa: E402


def _definition(x, sx, w, sw, pairs=None) -> np.ndarray:
    """The Definition's word at every (row of x, row of w), uint16 [r, n], or at the given (i, j) pairs."""
    i, j = pairs if pairs is not None else (np.repeat(np.arange(len(x)), len(w)), np.tile(np.arange(len(w)), len(x)))
    coord = bind(G.ScaledMmS8Coordinate, K=x.shape[1], **G.GEMM_SM90_S8)
    got = E.evaluate_batch(coord, x[i].view(np.uint8), sx[i].view(np.uint32), w[j].view(np.uint8), sw[j].view(np.uint32),
                           kernel="reference")
    got = np.asarray(got).astype(np.uint16)
    return got if pairs is not None else got.reshape(len(x), len(w))


def _specials(k: int, seed: int, keep=None):
    rng = np.random.default_rng(seed)
    x, sx = C.specials(len(C.SPECIALS), k, rng, C.TOKEN_SPREAD)
    w, sw = C.specials(len(C.SPECIALS), k, rng, C.CHANNEL_SPREAD)
    rows = [i for i, (p, _) in enumerate(C.SPECIALS) if keep is None or p in keep]
    return x[rows], sx[rows], w[rows], sw[rows]


def test_the_twin_is_the_definition_on_every_pair_of_special_rows() -> None:
    """Every special token against every special channel at K = 128: each code pattern under each special scale on both
    sides, so inf times zero, NaN scales, overflow to inf and subnormal products all occur."""
    x, sx, w, sw = _specials(128, 1)
    want = _definition(x, sx, w, sw)
    assert np.array_equal(C.twin(x, sx, w, sw), want)
    assert all(C.classes(want)[c] > 0 for c in ("nan", "inf", "zero", "subnormal")), C.classes(want)


def test_the_twin_is_the_definition_on_sums_past_2_24() -> None:
    """The max-magnitude and [100, 127] patterns at K = 2048, whose sums pass 2^24, so `cvt.rn.f32.s32` rounds."""
    x, sx, w, sw = _specials(2048, 2, keep={"min", "max", "big+", "big-", "alt"})
    assert (np.abs(C.exact_sums(x, w)) >= 1 << 24).sum() >= 8
    assert np.array_equal(C.twin(x, sx, w, sw), _definition(x, sx, w, sw))


def test_the_twin_takes_the_definitions_order_where_the_other_orders_round_differently() -> None:
    """Scales searched so that sw first (CUTLASS c2x's order) or sx * sw first rounds to another bf16 word: the twin is the
    Definition there and both other orders differ from it, so a check through the twin fails a reordered epilogue. Random
    scales rarely tell the orders apart."""
    rng = np.random.default_rng(7)
    r, k = 64, 64
    x, w = rng.integers(-128, 128, (r, k)).astype(np.int8), rng.integers(-128, 128, (r, k)).astype(np.int8)
    acc = (x.astype(np.int64) * w).sum(1)
    cx = (np.abs(rng.standard_normal((r, 1 << 16))) * C.TOKEN_SPREAD).astype(np.float32)
    cw = (np.abs(rng.standard_normal((r, 1 << 16))) * C.CHANNEL_SPREAD).astype(np.float32)
    f = acc.astype(np.int32).astype(np.float32)[:, None]
    want = C.bf16_rn(((f * cx) * cw).view(np.uint32))
    apart = (want != C.bf16_rn(((f * cw) * cx).view(np.uint32))) & (want != C.bf16_rn((f * (cx * cw)).view(np.uint32)))
    rows = np.flatnonzero(apart.any(1))
    assert len(rows) >= 4
    pick = apart[rows].argmax(1)
    sx, sw = cx[rows, pick], cw[rows, pick]
    pairs = (np.arange(len(rows)), np.arange(len(rows)))
    d = _definition(x[rows], sx, w[rows], sw, pairs)
    acc_d = np.diag(C.exact_sums(x[rows], w[rows]))[:, None]
    assert np.array_equal(np.diag(C.twin(x[rows], sx, w[rows], sw)), d)
    for order in ("sw-first", "scales-first"):
        assert (C.epilogue(acc_d, sx, sw, order)[:, 0] != d).all()


@pytest.mark.parametrize("m", [1, 7, 30])
def test_linear_on_the_cpu_is_the_twin(m: int) -> None:
    """`linear` on the CPU (`torch._int_mm`, `epilogue="eager"`) is the twin at every word but a NaN, where torch's CPU
    cast writes another NaN word for the Definition's 0x7FFF (s8_simple's docstring)."""
    torch = pytest.importorskip("torch")
    import s8_simple as S

    rng = np.random.default_rng(m)
    x, sx, w, sw = _specials(128, 3)
    x, sx = x[:m], sx[:m]
    w = np.concatenate([w, rng.integers(-128, 128, (10, 128)).astype(np.int8)])
    sw = np.concatenate([sw, C.scales(rng, 10, C.CHANNEL_SPREAD)])
    y = S.linear(*(torch.from_numpy(a) for a in (x, sx, w, sw)), epilogue="eager")
    got, want = y.view(torch.int16).numpy().view(np.uint16), C.twin(x, sx, w, sw)
    nan = want == C.NAN_BF16
    assert np.array_equal(got[~nan], want[~nan])
    assert (((got[nan] & 0x7F80) == 0x7F80) & ((got[nan] & 0x7F) != 0)).all() and nan.any()


def test_linear_refuses_what_the_definition_does_not_bind() -> None:
    torch = pytest.importorskip("torch")
    import s8_simple as S

    assert S.K_LIMIT == G.S8_K_LIMIT

    def ops(m, k, n, dtype=torch.int8):
        return torch.zeros(m, k, dtype=dtype), torch.ones(m), torch.zeros(n, k, dtype=torch.int8), torch.ones(n)

    for args, why in ((ops(2, 48, 8), "K = 48"), (ops(1, 1 << 17, 8), "K = 131072"), (ops(2, 64, 12), "N = 12"),
                      (ops(2, 64, 8, torch.uint8), "dtypes"), (ops(2, 64, 8), "epilogue='eager'")):
        with pytest.raises(ValueError, match=why):
            S.linear(*args)
    with pytest.raises(ValueError, match="pass one of"):
        S.linear(*ops(2, 64, 8), epilogue="fast")
    for k in (48, 1 << 17):
        with pytest.raises(ValueError):
            bind(G.ScaledMmS8Coordinate, K=k, **G.GEMM_SM90_S8)


def test_the_ptx_audit_takes_only_the_definitions_instructions() -> None:
    """`ptx_audit` passes PTX made of `FUSED_PTX` blocks, and fails one with an fma, a flushed multiply, a missing multiply
    or other f32 arithmetic: what a compiler pass that touched the epilogue would leave."""
    import s8_simple as S

    def block(i: int, src: str = S.FUSED_PTX) -> str:
        return src.replace("$0", f"%rs{i}").replace("$1", f"%r{i}").replace("$2", f"%f{2 * i}").replace("$3", f"%f{2 * i + 1}")

    body = "ld.global.v4.b32 {%r1, %r2, %r3, %r4}, [%rd1];\nmov.f32 %f9, 0f00000000;\n"
    good = body + "\n".join(block(i) for i in range(4))
    audit = C.ptx_audit(good)
    assert audit["ok"] and audit["cvt.rn.f32.s32"] == audit["blocks"] == 4 and audit["mul.rn.f32"] == 8
    for bad in (good.replace("mul.rn.f32 t, t, %f1;", "fma.rn.f32 t, t, %f1, %f9;"),
                good.replace("mul.rn.f32 t, t, %f1;", "mul.rn.ftz.f32 t, t, %f1;"),
                good.replace("mul.rn.f32 t, t, %f1;", ""),
                good + "\nadd.f32 %f9, %f9, %f9;"):
        assert not C.ptx_audit(bad)["ok"]


def test_the_check_passes_on_the_cpu_and_fails_a_wrong_word(tmp_path: Path) -> None:
    """`weights`, `gpu` and `verify` on the CPU at two tiny shapes, m = 1, 7 and 300 (256 rows sampled), with the eager
    epilogue: they pass, the only differences being torch's CPU NaN word. Then the run files hold the twin's words and the
    record claims a GPU: it passes only with the fused epilogue's fields (every word equal to the eager one's, PTX that
    passed its audit), and one flipped finite word in either epilogue's words fails it."""
    pytest.importorskip("torch")
    s, o = tmp_path / "scratch", tmp_path / "out"
    s.mkdir()
    common = ["--scratch", str(s), "--sets", "edge", "--shapes", "a=64:96,b=40:256", "--ms", "1,7,300", "--device", "cpu"]
    verify = ["verify", *common, "--out", str(o)]
    assert C.main(["weights", *common]) == 0
    assert C.main(["gpu", *common, "--out", str(o)]) == 0
    assert C.main(verify) == 0
    v = json.loads((o / "verify.json").read_text())
    assert v["pass"] and len(v["results"]) == 6 and {r["epilogue"] for r in v["results"]} == {"eager"}
    assert sorted({r["rows_checked"] for r in v["results"]}) == [1, 7, 256]
    assert sum(r["nan_word_mismatches"] for r in v["results"]) == sum(r["mismatches"] for r in v["results"]) > 0

    g = o / "gpu.json"
    rec = json.loads(g.read_text())
    for run in rec["runs"]:
        d = dict(np.load(s / run["file"]))
        d["y"] = d["y_eager"] = C.twin(d["x"], d["sx"][d["rows"]], np.load(s / run["w"]), np.load(s / run["sw"]))
        np.savez(s / run["file"], **d)
    gpu = rec | {"env": rec["env"] | {"device": "cuda"}}
    fused = [run | {"epilogue": "fused", "fused_ptx": "p", "fused_vs_eager_mismatches": 0} for run in rec["runs"]]
    for record, code in ((gpu, 1), (gpu | {"runs": fused}, 1), (gpu | {"runs": fused, "ptx": {"p": {"ok": False}}}, 1),
                         (gpu | {"runs": fused[:-1] + [fused[-1] | {"fused_vs_eager_mismatches": 1}],
                                 "ptx": {"p": {"ok": True}}}, 1),
                         (gpu | {"runs": fused, "ptx": {"p": {"ok": True}}}, 0)):
        g.write_text(json.dumps(record))
        assert C.main(verify) == code
    f = s / "run-edge-a-m7.npz"
    clean = dict(np.load(f))
    for key in ("y", "y_eager"):
        d = {name: a.copy() for name, a in clean.items()}
        d[key][0, 0] ^= 1
        np.savez(f, **d)
        assert C.main(verify) == 1
