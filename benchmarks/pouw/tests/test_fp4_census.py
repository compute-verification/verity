"""fp4_census.py's vectorised atom is bit-exact against verity.ml.tc, and its quantizers match fp4_quality.py's."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fp4_census as C  # noqa: E402


def test_atom_matches_the_pinned_models():
    rep = C.selftest(cases=600, seed=11)
    for name, r in rep.items():
        assert r["step_mismatches"] == 0 and r["chain_mismatches"] == 0, (name, r)


def test_e2m1_rounds_to_nearest_even():
    ties = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.75, 7.0, -100.0])
    got = C.E2M1_VALUES[C.e2m1_codes(ties) & 7] * np.where(C.e2m1_codes(ties) & 8, -1, 1)
    assert got.tolist() == [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, -1.0, 6.0, -6.0]


@pytest.mark.parametrize("fmt", ["nvf4", "mxf4"])
def test_quantizer_is_lossless_on_its_own_grid(fmt):
    rng = np.random.default_rng(3)
    x = rng.standard_t(3, (16, 256))
    codes, sc, g = C.quantize(x, fmt)
    d = C.dequantize(codes, sc, g, fmt)
    codes2, sc2, _ = C.quantize(d, fmt)
    assert np.array_equal(C.dequantize(codes2, sc2, g, fmt), d)


@pytest.mark.parametrize("fmt", ["nvf4", "mxf4"])
def test_quality_quantizers_agree(fmt):
    torch = pytest.importorskip("torch")
    import fp4_quality as Q
    rng = np.random.default_rng(4)
    x = rng.standard_t(3, (32, 512))
    codes, sc, g = C.quantize(x, fmt)
    want = C.dequantize(codes, sc, g, fmt)
    got = Q.QUANT[fmt](torch.tensor(x, dtype=torch.float32)).double().numpy()
    assert np.allclose(got, want, rtol=0, atol=1e-6 * np.abs(x).max())


def test_uniform_scale_chains_are_exact():
    """Equal scales everywhere: every step sums integers far below 2^24, so the whole chain is exact."""
    from verity.ml.tc.models import BLACKWELL_SM120_MXF4
    atom = C.Atom(BLACKWELL_SM120_MXF4)
    rng = np.random.default_rng(5)
    A, B = C.family("grid-aligned", 16, 2048, rng)
    r = C.census_matrix(atom, "mxf4", A, B, "none", 1.0, rng, pairs=16)
    assert r["words"]["exact_chains"] == 1.0


def test_breakeven_has_no_paying_dense_rewrite():
    doc = C.breakeven(C.PRICES)
    dense = [r for r in doc["strassen"] if "sparse" not in r["sub_products"] and "hypothetical" not in r["sub_products"]]
    assert dense and not any(r["pays"] for r in dense)


def test_attack_kernel_tables_and_fragments_are_consistent():
    """fp4_attack's opcode/flop/word tables cover the three routes and the fragment layout matches the kernel's stride."""
    from kernels import fp4_attack as A

    assert set(A.KINDS) == set(A.OPCODES) == set(A.FLOP_PER_MMA) == set(A.WORDS_PER_SET) == set(A.PUBLIC_PRICE)
    assert "SP" in A.OPCODES["sparse"] and "168128" in A.OPCODES["sparse"]      # the 2:4 m16n8k128 route
    assert A.FLOP_PER_MMA == {"dense": 2 * 16 * 8 * 64, "sparse": 2 * 16 * 8 * 128, "int8": 2 * 16 * 8 * 32}
    for kind in A.KINDS:
        frag = A.fragments(kind, 2)
        assert frag.shape[0] == 2 * 32 * A.NSETS * A.WORDS_PER_SET[kind]
        assert frag.dtype == np.uint32
    sp = A.fragments("sparse", 1).reshape(32, A.NSETS, A.WORDS_PER_SET["sparse"])
    assert (sp[:, :, 8] == 0x44444444).all()                                   # valid 2:4 ordered metadata everywhere


def test_identity_gate_reads_only_the_leased_device(monkeypatch):
    """nvidia-smi lists every GPU on a node whatever CUDA_VISIBLE_DEVICES says (node 2, server.md 06:35Z), so the gate asks
    for the visible device with -i and checks it against gpu-lease's GPU_LEASE_UUID; anything else fails closed."""
    from types import SimpleNamespace

    from kernels import fp4_attack as A

    node = [f"GPU-{i}" for i in range(8)]
    calls = []

    def smi(argv, **_):
        calls.append(argv)
        out = [node[int(argv[argv.index("-i") + 1])]] if "-i" in argv else node
        return SimpleNamespace(stdout="\n".join(out) + "\n")

    monkeypatch.setattr(A.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(A.subprocess, "run", smi)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    monkeypatch.setenv("GPU_LEASE_UUID", "GPU-4")
    ok = A._check_identity(None)
    assert ok["ok"] and ok["device_uuid"] == "GPU-4" and calls[-1][1:3] == ["-i", "4"]
    assert A._check_identity("GPU-4")["ok"] and not A._check_identity("GPU-5")["ok"]
    monkeypatch.setenv("GPU_LEASE_UUID", "GPU-5")                              # the lease names another GPU
    assert not A._check_identity(None)["ok"]
    monkeypatch.delenv("GPU_LEASE_UUID")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")                                  # unrestricted: all 8 visible
    assert A._check_identity(None)["device_uuid"] == "8 devices visible" and not A._check_identity(None)["ok"]
    monkeypatch.setattr(A.shutil, "which", lambda _: None)                      # no nvidia-smi at all
    assert not A._check_identity(None)["ok"]


def test_base_split_dense_noise_gives_no_saving():
    """Dense row-relative noise (delta=1/4) changes enough codes that the salt-free base split ties the honest MMA."""
    rng = np.random.default_rng(9)
    A, _ = C.family("gaussian", 128, 4096, rng)
    r = C.base_split("nvf4", A, "row", 0.25, rng)
    assert r["code_change_density"] > 0.4
    assert r["cheater_A_cost_per_dense_mac"] == 1.0 and not r["beats_honest"]


def test_base_split_sparse_delta_is_cheaper():
    """Tiny block-relative noise on an outlier-heavy family leaves Delta 2:4-coverable, so the cheater saves."""
    rng = np.random.default_rng(9)
    A, _ = C.family("outlier-in-block", 128, 4096, rng)
    r = C.base_split("nvf4", A, "block", 0.0625, rng)
    assert r["cheater_A_cost_per_dense_mac"] < 1.0 and r["beats_honest"]


def test_salt_dead_share_tracks_iid_and_falls_with_noise():
    """The share of 4-groups with >= 2 salt-dead codes (theory §5's debit statistic) matches the iid Binomial of the
    measured change density, and shrinks as the noise floor rises (more codes change with the salt)."""
    shares = []
    for delta in (0.125, 0.25, 1.0):
        rng = np.random.default_rng(9)
        A, _ = C.family("gaussian", 128, 4096, rng)
        r = C.base_split("nvf4", A, "row", delta, rng)
        assert 0.0 <= r["share_4groups_ge2_salt_dead"] <= 1.0
        assert abs(r["share_4groups_ge2_salt_dead"] - r["share_4groups_ge2_salt_dead_iid"]) < 0.05
        assert r["debit_2of4_per_mac"] == 0.5 * r["share_4groups_ge2_salt_dead"]
        shares.append(r["share_4groups_ge2_salt_dead"])
    assert shares[0] > shares[1] > shares[2]


def test_base_split_reads_dumped_activations(tmp_path):
    """basesplit --acts runs the split on fp4_quality.py's dumped activation rows, one result per (file, linear, format)."""
    rng = np.random.default_rng(5)
    np.savez(tmp_path / "layer00.npz", **{"mlp.up_proj/x": rng.standard_normal((32, 256)).astype(np.float32),
                                          "mlp.up_proj/w": rng.standard_normal((32, 256)).astype(np.float32)})
    args = C.parse_args(["basesplit", "--acts", str(tmp_path), "--noise", "none,row", "--delta", "0.25"])
    res = C.cmd_basesplit(args)
    assert {(r["source"], r["linear"], r["k"], r["noise"]) for r in res["results"]} == {("layer00", "mlp.up_proj", 256, "row")}
    assert len(res["results"]) == len(C.atoms_from(args))
    assert all(r["cheater_A_cost_per_dense_mac"] == 1.0 for r in res["results"])


@pytest.mark.parametrize("fmt", ["nvf4", "mxf4"])
def test_fp4_representable_by_hand(fmt):
    """A block is q/2 * s.  7 = 2 * 3.5 fits under a UE4M3 scale but not under a power of two (any scale fits one value);
    {6, 1} and {1.5, 0.5} are already on the grid; {6, 5} is off every grid (6:5 is no ratio of E2M1 values); zero fits."""
    blocks = {"seven": [14], "six_one": [12, 2], "six_five": [12, 10], "zero": [0, 0], "three_halves": [3, 1]}
    N = np.array([(v + [0] * 16)[:16] for v in blocks.values()], dtype=np.int64)
    strict, anyscale = C.fp4_representable(N, fmt)
    got = dict(zip(blocks, zip(strict.tolist(), anyscale.tolist())))
    assert got["seven"] == ((fmt == "nvf4"), True)                       # 3.5 is UE4M3, not 2^e; any scale 7/6 works
    assert got["six_one"] == (True, True) and got["zero"] == (True, True) and got["three_halves"] == (True, True)
    assert got["six_five"] == (False, False)


def test_merge_rate_sees_duplicates_and_is_small_on_salted_gaussians():
    """Duplicate rows cancel under '-' and merge under '+' (2v fits, a scale doubling); salted Gaussian blocks almost never
    sum onto one FP4 value, so epsilon_R is small there."""
    rng = np.random.default_rng(4)
    x = rng.standard_normal((32, 512))
    dup = C.merge_rate("nvf4", np.repeat(x[::2], 2, axis=0), "none", 0.25, rng)["patterns"]["adjacent-row"]
    assert dup["dup"] == 1.0 and dup["-"]["cancel"] == 1.0 and dup["+"]["eps_any_scale"] == 1.0
    g = C.merge_rate("nvf4", x, "row", 0.25, rng)
    assert 0.0 < g["zero_codes"] < 0.2 and g["zero_blocks"] == 0.0
    assert g["eps_R_strict_max"] <= g["eps_R_any_scale_max"] < 0.01


def test_merge_partners_find_every_duplicate_row():
    """With every row duplicated, each block has a partner in its k-block column (its twin); a lone Gaussian block rarely does."""
    rng = np.random.default_rng(6)
    x = rng.standard_normal((16, 256))
    codes, sc, _ = C.quantize(np.repeat(x, 2, axis=0), "nvf4")
    N, E = C.block_ints(codes, sc, "nvf4")
    p = C.merge_partners(N, E, np.abs(N).max(-1) > 0, "nvf4")
    assert p["same_kblock_any_row"] == 1.0 and p["rows"] == 32 and p["kblocks_searched"] == 16
    codes, sc, _ = C.quantize(x, "nvf4")
    N, E = C.block_ints(codes, sc, "nvf4")
    assert C.merge_partners(N, E, np.abs(N).max(-1) > 0, "nvf4")["same_kblock_any_row"] < 0.05
