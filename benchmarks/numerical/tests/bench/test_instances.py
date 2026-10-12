"""``bench-instances/v1``: layout, prefixes, seeds, silicon agreement on a fixed subset, negatives, pinned digests.

The fast suite reads only the committed files (indices, ``y``/``out`` words, negatives, seeds, manifest) and the
generator; the tests that need the 27 MB built arrays skip when they are absent.  ``VERITY_BENCH_FULL=1`` rechecks
every instance and rebuilds the corner tier (minutes)."""

from __future__ import annotations

import hashlib
import os
import json
from pathlib import Path

import pytest
from verity.ml.circuits.silicon import silicon as S
from verity.ml.circuits.silicon.errors import InvalidArtifact
from verity_numerical.bench import contract
from verity_numerical.bench import instances as I
from verity_numerical.bench import instances_gen as G

ROOT = I.DEFAULT_ROOT
MANIFEST = json.loads((ROOT / "manifest.json").read_text())
BUILT = all((ROOT / n).is_file() for t in I.TIERS for n in I.BUILT_FILES[t])
needs_built = pytest.mark.skipif(not BUILT, reason="the built operand arrays are absent (python -m verity_numerical.bench.instances build)")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- manifest and digests ------------------------------------------------------------------------------------------------------------


def test_manifest_identity() -> None:
    assert MANIFEST["schema"] == MANIFEST["dataset"] == contract.INSTANCES_DATASET == "bench-instances/v1"
    assert MANIFEST["target"]["profile"] == contract.PROFILE and MANIFEST["target"]["pipeline"] == G.PIPELINE.name
    assert MANIFEST["hf"]["revision"] == G.HF_REVISION and MANIFEST["seeds"]["experiment_id"] == G.EXPERIMENT_ID
    assert MANIFEST["tiers"]["vu-k1536"]["n"] == I.N_VU == 4096 and MANIFEST["tiers"]["tu-k16"]["n"] == I.N_TU == 4096


def test_committed_digests_are_pinned() -> None:
    for tier in I.TIERS:
        for name in I.COMMITTED_FILES[tier]:
            rec = MANIFEST["tiers"][tier]["files"][name]
            assert rec["committed"] is True
            assert sha256(ROOT / name) == rec["sha256"], name
            assert (ROOT / name).stat().st_size == rec["bytes"], name


@needs_built
def test_built_digests_are_pinned() -> None:
    for tier in I.TIERS:
        for name in I.BUILT_FILES[tier]:
            rec = MANIFEST["tiers"][tier]["files"][name]
            assert rec["committed"] is False
            assert sha256(ROOT / name) == rec["sha256"], name


def test_committed_footprint_is_small() -> None:
    total = sum(rec["bytes"] for tier in I.TIERS for name, rec in MANIFEST["tiers"][tier]["files"].items() if rec["committed"])
    total += (ROOT / "manifest.json").stat().st_size + sum(p.stat().st_size for p in (ROOT / "seeds").rglob("*") if p.is_file())
    assert total < 1_100_000, total


def test_cross_checks_recorded_zero_disagreements() -> None:
    vu = MANIFEST["tiers"]["vu-k1536"]["cross_check"]["hawkeye_close"]
    tu = MANIFEST["tiers"]["tu-k16"]["cross_check"]["hawkeye_close"]
    assert vu["disagreements"] == 0 and vu["steps_compared"] + vu["skipped_saturation_steps_silicon_only"] == 4096 * 96
    assert tu["disagreements"] == 0 and tu["compared"] + tu["skipped_saturation_silicon_only"] == 4096
    checks = MANIFEST["seeds"]["checks"]
    assert checks == {"sha256_ok": 24, "silicon_vs_out_row_j": 24, "weights_download_vs_w_row": 24}


# --- layout ------------------------------------------------------------------------------------------------------------------------------


def test_u16_u32_layout_round_trip(tmp_path: Path) -> None:
    words16 = [0, 1, 0x7F80, 0xFFFF, 0xBED1]
    words32 = [0, 0x80000000, 0x7F800000, 0xFFFFFFFF, 0x3F800000]
    I.write_u16(tmp_path / "a.u16", words16)
    I.write_u32(tmp_path / "b.u32", words32)
    assert (tmp_path / "a.u16").read_bytes() == b"".join(w.to_bytes(2, "little") for w in words16)
    assert (tmp_path / "b.u32").read_bytes() == b"".join(w.to_bytes(4, "little") for w in words32)
    assert list(I.read_u16(tmp_path / "a.u16")) == words16 and list(I.read_u32(tmp_path / "b.u32")) == words32


def test_index_files_are_positional_rows() -> None:
    for tier in I.TIERS:
        doc = json.loads((ROOT / f"{tier}.index.json").read_text())
        assert doc["schema"] == "bench-instances/v1/index" and doc["tier"] == tier
        assert len(doc["rows"]) == MANIFEST["tiers"][tier]["n"]
        assert all(len(row) == len(doc["columns"]) for row in doc["rows"])
        assert [row[0] for row in doc["rows"]] == list(range(len(doc["rows"])))


def test_load_refuses_unknown_tier_and_names_the_build_when_arrays_are_absent(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        I.load("vu-k32", ROOT)
    (tmp_path / "vu-k1536.index.json").write_text(json.dumps({"columns": [], "rows": []}))
    with pytest.raises(FileNotFoundError, match="instances build"):
        I.load("vu-k1536", tmp_path)


@needs_built
def test_load_shapes() -> None:
    vu = I.load("vu-k1536")
    assert vu.n == 4096 and vu.K == 1536 and len(vu.x) == len(vu.w) == 4096 * 1536 and len(vu.y) == 4096 and len(vu.acc) == 4096 * 96
    x, w, y = vu.instance(0)
    assert len(x) == len(w) == 1536 and isinstance(y, int)
    tu = I.load("tu-k16")
    assert tu.n == 4096 and len(tu.a) == len(tu.b) == 4096 * 16 and len(tu.c) == len(tu.out) == 4096
    c, a, b, out = tu.instance(0)
    assert len(a) == len(b) == 16 and isinstance(c, int) and isinstance(out, int)


# --- composition and prefixes ---------------------------------------------------------------------------------------------------------------


def vu_index() -> dict:
    return json.loads((ROOT / "vu-k1536.index.json").read_text())


def test_every_prefix_is_representative() -> None:
    rows = vu_index()["rows"]
    kinds = [row[1][0] for row in rows]
    assert kinds[0] == "real" and "seed" in rows[0][1]
    for lo in range(0, 4096, 64):
        block = kinds[lo : lo + 64]
        assert block.count("real") == 54 and block.count("corner") == 10, lo
        fams = {rows[i][1][1] for i in range(lo, lo + 64) if kinds[i] == "corner"}
        assert len(fams) >= 8, f"block {lo} holds corners from only {fams}"
    assert kinds.count("corner") == 640 and kinds.count("real") == 3456
    first64 = rows[:64]
    assert sum(1 for r in first64 if "seed" in r[1]) >= 24, "B=64 holds every hardware-confirmed seed"
    assert {r[1][1] for r in first64 if r[1][0] == "corner"} >= {"sat-final", "near-sat", "cancel", "floor-flush"}


def test_id_zero_is_the_k1536_conformance_vector() -> None:
    doc = vu_index()
    row0 = doc["rows"][0]
    assert doc["seeds"][row0[2]] == "qkv_proj_decode_0" and row0[3] == 1014
    y = I.read_u16(ROOT / "vu-k1536.y.u16")
    fx = json.loads((ROOT / "seeds" / "qkv_proj_decode_0" / "fixture.json").read_text())
    assert y[0] == fx["committed_word"] == 0xBED1


def test_all_24_seeds_are_present_with_their_committed_words() -> None:
    doc = vu_index()
    y = I.read_u16(ROOT / "vu-k1536.y.u16")
    seeds_doc = json.loads((ROOT / "seeds" / "seeds.json").read_text())
    seen = {}
    for row in doc["rows"]:
        if "seed" in row[1]:
            seen[doc["seeds"][row[2]]] = row
    assert set(seen) == set(G.SEED_NAMES) and len(seen) == 24
    for name, row in seen.items():
        rec = seeds_doc["seeds"][name]
        assert row[3] == rec["j"] and y[row[0]] == rec["out_row_word_j"], name
    # the seed files themselves are the recorded bytes
    for name in G.SEED_NAMES:
        for f in ("x.u16", "w_row.u16", "fixture.json"):
            assert sha256(ROOT / "seeds" / name / f) == seeds_doc["seeds"][name]["sha256"][f], f"{name}/{f}"


def test_real_rows_share_activation_rows() -> None:
    doc = vu_index()
    per_seed: dict[int, set[int]] = {}
    for row in doc["rows"]:
        if row[1][0] == "real":
            per_seed.setdefault(row[2], set()).add(row[3])
    assert len(per_seed) == 24 and all(len(js) == 1 + G.ROWS_PER_X for js in per_seed.values())


# --- silicon agreement on a fixed subset -------------------------------------------------------------------------------------------------------


def test_seeds_reproduce_the_captured_output_words() -> None:
    seeds = G.load_seeds(ROOT / "seeds")
    for s in seeds:
        acc = S.tc_dot_chain(G.PIPELINE, 0, s.x, s.w)
        assert G.epilogue(acc) == s.committed_word, s.name


@needs_built
def test_silicon_agrees_on_a_fixed_subset() -> None:
    vu = I.load("vu-k1536")
    ids = [*range(64), *range(64, 4096, 257)]
    for i in ids:
        x, w, y = vu.instance(i)
        accs = G.chain_accs(x, w)
        assert accs == vu.accs(i) and G.epilogue(accs[-1]) == y, i
    tu = I.load("tu-k16")
    for i in range(0, 4096, 61):
        c, a, b, out = tu.instance(i)
        assert S.tc_dot(G.PIPELINE, c, a, b) == out, i


@needs_built
def test_tu_chain_states_come_from_vu_chains() -> None:
    vu, tu = I.load("vu-k1536"), I.load("tu-k16")
    n = 0
    for row in tu.index[:512]:
        if row[2] is None:
            continue
        vid, s = row[2]
        c, a, b, out = tu.instance(row[0])
        x, w, _ = vu.instance(vid)
        accs = vu.accs(vid)
        assert c == (accs[s - 1] if s else 0) and a == x[16 * s : 16 * s + 16] and b == w[16 * s : 16 * s + 16] and out == accs[s]
        n += 1
    assert n > 400


# --- corners: deterministic, valid, and what they claim ----------------------------------------------------------------------------------


def test_corner_generation_is_deterministic_and_valid() -> None:
    seeds = G.load_seeds(ROOT / "seeds")
    y = I.read_u16(ROOT / "vu-k1536.y.u16")
    rows = vu_index()["rows"]
    picked = {}
    for row in rows:
        if row[1][0] == "corner" and row[4]["family"] not in picked:
            picked[row[4]["family"]] = row
    assert set(picked) == set(G.CORNER_FAMILIES)
    for fam, row in picked.items():
        c1 = G.generate_corner(fam, row[4]["i"], seeds)
        c2 = G.generate_corner(fam, row[4]["i"], seeds)
        assert (c1.x, c1.w) == (c2.x, c2.w) and c1.tags == row[1]
        accs = G.chain_accs(c1.x, c1.w)  # inside the domain: no InvalidArtifact
        assert G.epilogue(accs[-1]) == y[row[0]], fam
    sat = next(row for row in rows if row[1][:2] == ["corner", "sat-final"])
    assert y[sat[0]] in (0x7F80, 0xFF80)


def test_bf16_subnormal_operands_are_exact_in_silicon() -> None:
    """What silicon.bf16_product does with BF16 subnormals (recorded for the manifest): an exponent field of 0 is read as 1
    (the 2^-126 scale) with the unnormalised significand and no implicit bit, so the product is exact in value but is
    carried at the *unnormalised* exponent -- 2^-133 = 0x0001 x 1.0 is Term(exponent=-126, magnitude=2^16), not
    (-133, 2^23) -- and that unnormalised exponent is what enters the alignment maximum.  No flush to zero: a subnormal x
    normal product survives into the FP32 subnormal range; a subnormal x subnormal product (below 2^-252) is always lost to
    the -132 alignment floor."""

    tiny = G.bf16(0, 0, 1)  # 2^-133
    t = S.bf16_product(tiny, G.bf16(0, 127, 0))  # x 1.0
    assert (t.negative, t.exponent, t.magnitude) == (False, -126, 1 << 16)
    assert t.magnitude * 2 ** (t.exponent - 23) == 2**-133
    assert S.tc_dot(G.PIPELINE, 0, [tiny] + [0] * 15, [G.bf16(0, 127, 0)] + [0] * 15) == 0x00010000  # 2^-133 = 0x10000 x 2^-149
    assert S.tc_dot(G.PIPELINE, 0, [tiny] * 16, [tiny] * 16) == 0  # 16 x 2^-266: below the floor, flushed
    big = G.bf16(0, 254, 0)  # 2^127
    assert S.tc_dot(G.PIPELINE, 0, [tiny] + [0] * 15, [big] + [0] * 15) == 0x3C800000  # 2^-133 x 2^127 = 2^-6 exactly


# --- negatives -------------------------------------------------------------------------------------------------------------------------------


def test_negatives_differ_and_are_tagged() -> None:
    neg = I.load("vu-k1536-neg")
    assert neg.columns == ["id", "mistake", "base_id", "claimed_y", "correct_y", "verdict", "note"]
    mistakes = {m.name for m in G.MISTAKES} | set(G.REJECT_CLASSES)
    seen = set()
    for i in range(neg.n):
        x, w, claimed = neg.instance(i)
        _, mistake, base, claimed_y, correct, verdict, _ = neg.index[i]
        seen.add(mistake)
        assert mistake in mistakes and 0 <= base < 4096 and claimed == claimed_y
        if verdict == "wrong":
            accs = G.chain_accs(x, w)
            assert G.epilogue(accs[-1]) == correct != claimed, (i, mistake)
        else:
            assert verdict == "reject" and correct is None and mistake in G.REJECT_CLASSES
            with pytest.raises(InvalidArtifact):
                G.chain_accs(x, w)
    assert seen == mistakes, mistakes - seen


def test_nonfinite_negative_is_rejected_not_computed() -> None:
    neg = I.load("vu-k1536-neg")
    rows = [row for row in neg.index if row[1] == "nonfinite-operand"]
    assert len(rows) == 4
    for row in rows:
        x, w, _ = neg.instance(row[0])
        assert any((v >> 7) & 0xFF == 0xFF for v in x + w)
        with pytest.raises(InvalidArtifact):
            S.tc_dot_chain(G.PIPELINE, 0, x, w)


def test_sensitivity_table_matches_the_negatives_and_hides_what_real_data_hides() -> None:
    sens = MANIFEST["sensitivity"]
    assert set(sens) == {m.name for m in G.MISTAKES} | set(G.REJECT_CLASSES)
    for m in G.MISTAKES:
        s = sens[m.name]
        assert 0 <= s["y_differs"] <= s["applicable"] <= 4096 and s["y_differs"] == s["y_differs_real"] + s["y_differs_corner"]
        if m.level == "chain":
            assert s["y_differs"] <= s["acc_differs"] <= s["acc_applicable"], m.name
    assert sens["sign-flip"]["y_differs"] == 4096
    # the real distribution hides the accumulation-order mistakes at the output word while the accumulators see them
    for name in ("ieee-fp32-fma", "single-group-16", "adder-24-bit", "reversed-k"):
        assert sens[name]["y_differs_real"] < 10 < sens[name]["acc_differs"], name
    assert sens["no-exponent-floor"]["y_differs_real"] == 0 and sens["no-exponent-floor"]["y_differs_corner"] > 0
    assert sens["non-sticky-saturation"]["y_differs_real"] == 0 and sens["non-sticky-saturation"]["y_differs_corner"] > 0


# --- generator pieces ------------------------------------------------------------------------------------------------------------------------


def test_rng_is_a_documented_sha256_counter_stream() -> None:
    key = hashlib.sha256(b"bench-instances/v1|corner|cancel|3").digest()
    block0 = hashlib.sha256(key + (0).to_bytes(8, "little")).digest()
    rng = G.Rng("corner", "cancel", 3)
    assert rng.bits(8) == block0[0]
    assert rng.bits(16) == int.from_bytes(block0[1:3], "little")


def test_ieee_fma_chain_is_binary32_sequential() -> None:
    one, two = 0x3F80, 0x4000
    assert G.ieee_fma_chain([one] * 3, [two] * 3) == 0x40C00000  # 6.0
    # 2^24 + 1 is not representable: the sequential sum rounds (ties to even) where the tensor core's 25-bit adder would not
    big = G.bf16(0, 127 + 24, 0)
    assert G.ieee_fma_chain([big, one, one], [one, one, one]) == 0x4B800000  # 2^24 + 1 + 1 -> 2^24 after each RNE
    assert G.ieee_fma_chain([big, one, one], [one, one, one], 0) != S.tc_dot(G.PIPELINE, 0, [big, one, one] + [0] * 13, [one] * 3 + [0] * 13)


def test_fused_row_mapping() -> None:
    assert G.tensor_of("qkv_proj", "5", 1014) == ("model.layers.5.self_attn.q_proj.weight", 1014)
    assert G.tensor_of("qkv_proj", "5", 1536 + 3) == ("model.layers.5.self_attn.k_proj.weight", 3)
    assert G.tensor_of("qkv_proj", "5", 1536 + 256 + 7) == ("model.layers.5.self_attn.v_proj.weight", 7)
    assert G.tensor_of("gate_up_proj", "11", 8960 + 1) == ("model.layers.11.mlp.up_proj.weight", 1)
    assert G.tensor_of("lm_head", "27", 5) == ("model.embed_tokens.weight", 5)


# --- slow: everything ------------------------------------------------------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("VERITY_BENCH_FULL"), reason="minutes: set VERITY_BENCH_FULL=1")
@needs_built
def test_full_recheck() -> None:
    rep = I.verify(ROOT, full=True, log=lambda *_: None)
    assert rep["ok"], rep["problems"]
