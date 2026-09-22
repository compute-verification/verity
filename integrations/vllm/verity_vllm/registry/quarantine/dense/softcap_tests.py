"""Structure tests for `AttentionSoftcap_v1` (softcap.py) and the `MufuTanh` primitive's refusal discipline (prims.py).

    /path/to/.venv/bin/python -m pytest verity_vllm/registry/quarantine/dense/softcap_tests.py -q      (laptop, torch-free, no tables)

What is checked here (structure + the in-tree MufuTanh table; the Definition's differential record is the difftest adapter's run,
out/gen/lanes/gemma/softcap_difftest_n25.quarantine.json, 25/25 bitwise vs FA2 softcap):
  * gate population: `AttnBlockSoftcap_v1` = `AttnBlock_v3` + NVIS x {MufuTanh, F32MulFtz} + one constant, for several (NVIS, FIRST);
    `AttentionSoftcap_v1` = `Attention_v3` + NH x T x {MufuTanh, F32MulFtz} + the per-block constants at small T;
  * the two FA2 constants: `softmax_scale / CAP` as an RN32 division and `CAP * log2e` as double-then-float (Gemma-2-2B values);
  * the tanh stage precedes the row max: in body order every `MufuTanh` node comes before the first `F32Max`;
  * `MufuTanh.evaluate` raises `MufuTanhUnmeasured` (never approximates) for words outside the loaded shards / established rules.
    The shard-file round trip (writer -> loader, odd-symmetry gating, corrupt-file rejection) is exercised by
    `verity_capture/bench/measure_tanh.py --selftest`, which owns the on-disk format (this package may not do file I/O).
"""
from __future__ import annotations

import math
import struct
from collections import Counter

import numpy as np
import pytest

from verity.ir.defs import bind
from verity_vllm.registry import prims as P
from verity_vllm.registry.b1 import AttentionV3, AttnBlockV3, fa2_kblock_n

from verity_vllm.registry.quarantine.dense.softcap import AttentionSoftcap, AttnBlockSoftcap, softcap_mul_bits, softcap_scale_log2_bits


def kind_counts(sp) -> Counter:
    """Primitive ids of every gate of a specialization (batch members counted n times)."""
    out: Counter = Counter()

    def walk(fn, mult):
        if isinstance(fn, P.PrimitiveDefinition):  # a batched primitive
            out[fn.id] += mult
            return
        for node in fn.body.nodes:
            if node.form == "primitive":
                out[node.fn.id] += mult
            elif node.form == "call":
                walk(node.fn, mult)
            else:
                walk(node.fn, mult * node.n)

    walk(sp, 1)
    return out


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


@pytest.mark.parametrize("nvis,first", [(128, True), (128, False), (37, True), (1, False), (64, False)])
def test_block_is_v3_plus_softcap_stage(nvis, first):
    D, BN, CAP = 64, 128, 50.0
    base = kind_counts(bind(AttnBlockV3, D=D, NVIS=nvis, FIRST=first, BN=BN))
    soft = kind_counts(bind(AttnBlockSoftcap, D=D, NVIS=nvis, FIRST=first, BN=BN, CAP=CAP))
    diff = Counter(soft)
    diff.subtract(base)
    diff = +diff  # positive part
    # added: one tanh + one ftz multiply per visible score, the softcap multiplier constant, the CAP*log2e exp2 constant
    assert dict(diff) == {P.MufuTanh.id: nvis, P.F32MulFtz.id: nvis,
                          f"Const32[0x{softcap_mul_bits(D, CAP):08x}]_v1": 1, f"Const32[0x{softcap_scale_log2_bits(CAP):08x}]_v1": 1}, diff
    # removed: only v3's D^-0.5 * log2e constant; everything else of AttnBlock_v3 is present with the same multiplicity
    neg = Counter(base)
    neg.subtract(soft)
    neg = +neg
    assert dict(neg) == {f"Const32[0x{P.bits_of(np.float32(float(np.float32(D ** -0.5)) * 1.4426950408889634)):08x}]_v1": 1}, neg
    assert soft.total() == base.total() + 2 * nvis + 1


def test_tanh_stage_precedes_row_max():
    sp = bind(AttnBlockSoftcap, D=64, NVIS=40, FIRST=True, BN=128, CAP=50.0)
    ids = [n.fn.id for n in sp.body.nodes if n.form == "primitive"]
    last_tanh = max(i for i, k in enumerate(ids) if k == P.MufuTanh.id)
    first_max = min(i for i, k in enumerate(ids) if k == P.F32Max.id)
    first_ex2 = min(i for i, k in enumerate(ids) if k == P.MufuEx2Ftz.id)
    assert last_tanh < first_max < first_ex2
    assert ids.count(P.MufuTanh.id) == 40 and ids.count(P.F32MulFtz.id) >= 40


def test_attention_is_v3_plus_per_score_tanh():
    T, NH, KVH, D = 5, 2, 1, 64
    BN = fa2_kblock_n(D)
    base = kind_counts(bind(AttentionV3, T=T, NH=NH, KVH=KVH, D=D, BN=BN))
    soft = kind_counts(bind(AttentionSoftcap, T=T, NH=NH, KVH=KVH, D=D, BN=BN, CAP=50.0))
    diff = Counter(soft)
    diff.subtract(base)
    diff = +diff
    assert diff[P.MufuTanh.id] == NH * T and diff[P.F32MulFtz.id] == NH * T
    # a two-block head (T > BN): tanh count is still one per visible score
    T2 = BN + 3
    soft2 = kind_counts(bind(AttentionSoftcap, T=T2, NH=1, KVH=1, D=D, BN=BN, CAP=50.0))
    assert soft2[P.MufuTanh.id] == T2


def test_fa2_constants_gemma2():
    # Gemma-2-2B: D = 256 -> softmax_scale = 1/16 exactly; params.softcap = f32(0.0625 / 50) ; scale_softmax_log2 = f32(50 * M_LOG2E)
    assert softcap_mul_bits(256, 50.0) == f32_bits(0.0625 / 50.0) == 0x3AA3D70A
    assert softcap_scale_log2_bits(50.0) == f32_bits(50.0 * 1.4426950408889634) == f32_bits(72.13475204444817)
    # D = 128 (Qwen-like geometry with a cap): the division is on the f32-narrowed scale
    s = struct.unpack("<f", struct.pack("<f", 128 ** -0.5))[0]
    assert softcap_mul_bits(128, 30.0) == f32_bits(s / 30.0)
    # CAP must be positive: FA2 takes the Is_softcap path iff softcap > 0
    with pytest.raises(ValueError):
        bind(AttnBlockSoftcap, D=64, NVIS=4, FIRST=True, BN=128, CAP=0.0).body


# ---------------------------------------------------------------------------------------------------------------------
# MufuTanh refusal discipline (torch-free, no file I/O: tables switched off through the override hook)
# ---------------------------------------------------------------------------------------------------------------------


@pytest.fixture
def no_tables():
    saved = dict(P.MUFU_TANH_RULES)
    saved_override = list(P._TANH_TABLES_OVERRIDE)
    P.mufu_tanh_use_tables(None)
    yield
    P.MUFU_TANH_RULES.clear()
    P.MUFU_TANH_RULES.update(saved)
    P._TANH_TABLES_OVERRIDE[:] = saved_override
    P._tanh_shards.cache_clear()


def test_mufu_tanh_refuses_without_tables(no_tables):
    for k in P.MUFU_TANH_RULES:
        P.MUFU_TANH_RULES[k] = None
    assert P.mufu_tanh_coverage() == {"shards": [], "rules": {"identity_below": None, "saturate_from": None, "nan_out": None}}
    for w in (0x3F000000, 0xBF000000, 0x00000001, 0x41000000, 0x7F800000, 0x7FC00000, 0x00000000):
        with pytest.raises(P.MufuTanhUnmeasured):
            P.MufuTanh.evaluate(w)
    assert P.MufuTanh.conformance.startswith("quarantine")
    assert P.MufuTanh.id == "MufuTanh_v1"


def test_mufu_tanh_rules_apply_only_when_established(no_tables):
    P.MUFU_TANH_RULES.update(identity_below=None, saturate_from=None, nan_out=None)
    with pytest.raises(P.MufuTanhUnmeasured):
        P.MufuTanh.evaluate(0x41000000)
    P.MUFU_TANH_RULES.update(identity_below=0x31000000, saturate_from=0x41000000, nan_out=0x7FFFFFFF)
    assert P.MufuTanh.evaluate(0x30FFFFFF) == 0x30FFFFFF and P.MufuTanh.evaluate(0xB0FFFFFF) == 0xB0FFFFFF
    assert P.MufuTanh.evaluate(0x00000000) == 0x00000000 and P.MufuTanh.evaluate(0x80000000) == 0x80000000
    assert P.MufuTanh.evaluate(0x41000000) == 0x3F800000 and P.MufuTanh.evaluate(0xC1000000) == 0xBF800000
    assert P.MufuTanh.evaluate(0x7F800000) == 0x3F800000 and P.MufuTanh.evaluate(0xFF800000) == 0xBF800000
    assert P.MufuTanh.evaluate(0x7FC00001) == 0x7FFFFFFF and P.MufuTanh.evaluate(0xFFC00000) == 0x7FFFFFFF
    with pytest.raises(P.MufuTanhUnmeasured):
        P.MufuTanh.evaluate(0x40000000)  # the gap between the rules: no shard loaded
    assert math.isnan(struct.unpack("<f", struct.pack("<I", 0x7FFFFFFF))[0])


# ---------------------------------------------------------------------------------------------------------------------
# the in-tree default table (quarantine/dense/tables/mufu_tanh_sm89.{json,xzblocks})
# ---------------------------------------------------------------------------------------------------------------------

MUFU_TANH_TABLE_SHA256 = "5e261b88720e17cf8d868f2f9612bf6663600c40b7bc97e706d34855831ff247"   # 449,932 B, job 0917T214646-3b1c9a (cp-cov-d)
MUFU_TANH_WORDS_SHA256 = "139fa3f9787e3e158044a909cb7645b2e5feeaae76407078c5c5f8421ddd4392"   # the 2^28 measured words, le u32


@pytest.fixture
def default_table(monkeypatch):
    saved_override = list(P._TANH_TABLES_OVERRIDE)
    P._TANH_TABLES_OVERRIDE.clear()
    monkeypatch.delenv(P.MUFU_TANH_TABLES_ENV, raising=False)
    P._tanh_shards.cache_clear()
    yield P.mufu_tanh_table_manifest()
    P._TANH_TABLES_OVERRIDE[:] = saved_override
    P._tanh_shards.cache_clear()


def test_default_table_manifest_and_coverage(default_table):
    man = default_table
    assert man["format"] == "verity/mufu-tanh-table/v1" and man["primitive"] == "MufuTanh" and man["ptx"] == "tanh.approx.f32 $0, $1;"
    assert man["table"]["sha256_file"] == MUFU_TANH_TABLE_SHA256 and man["table"]["bytes"] == 449932
    assert man["table"]["sha256_words_le_u32"] == MUFU_TANH_WORDS_SHA256
    assert man["table"]["odd_symmetry"] == {"checked": 1 << 28, "violations": 0, "definition": man["table"]["odd_symmetry"]["definition"]}
    assert man["rules"] == {k: f"0x{v:08X}" for k, v in P.MUFU_TANH_RULES.items()}
    assert man["hardware"]["capability"] == [8, 9] and "580.178.04" in man["hardware"]["nvidia_smi"][0] and man["hardware"]["triton"] == "3.7.1"
    assert man["evidence"]["conformance_job"] == {"id": "0917T215619-1af45f", "worker": "cp-cov-d", "checked": 400077, "mismatches": 0,
                                                  "note": man["evidence"]["conformance_job"]["note"]}
    cov = P.mufu_tanh_coverage()
    assert cov["rules"] == man["rules"] and len(cov["shards"]) == 1
    (sh,) = cov["shards"]
    assert (sh["lo"], sh["hi"], sh["n_words"], sh["odd_symmetry_verified"], sh["sha256_words"]) == \
           ("0x31000000", "0x41000000", 1 << 28, True, MUFU_TANH_WORDS_SHA256)
    # the table closes the table-only region [identity_below, saturate_from) with margin on both sides
    assert int(sh["lo"], 16) <= P.MUFU_TANH_RULES["identity_below"] and int(sh["hi"], 16) >= P.MUFU_TANH_RULES["saturate_from"]
    assert P.MufuTanh.conformance.startswith("quarantine") and MUFU_TANH_TABLE_SHA256[:8] in P.MufuTanh.conformance


def test_default_table_matches_recorded_raw_sample_bitwise(default_table):
    """512 (x, hw) pairs read from the raw measured .npy on the worker (independent of the compact encoding), plus their sign flips."""
    pairs = default_table["sample"]["pairs"]
    assert len(pairs) == 512 == default_table["sample"]["n"]
    in_table_region = 0
    for xh, yh in pairs:
        x, y = int(xh, 16), int(yh, 16)
        assert P.MufuTanh.evaluate(x) == y, (xh, yh, f"0x{P.MufuTanh.evaluate(x):08X}")
        assert P.MufuTanh.evaluate(x ^ 0x80000000) == y ^ 0x80000000, (xh, yh)
        in_table_region += P.MUFU_TANH_RULES["identity_below"] <= x < P.MUFU_TANH_RULES["saturate_from"]
    assert in_table_region >= 100  # the sample is uniform over [2^-29, 8) by word; most of it is in the table-only region
    # the shard object itself agrees on every sample word (table path, no rules involved)
    (shard,) = P._tanh_shards()
    assert all(shard.lookup(int(xh, 16)) == int(yh, 16) for xh, yh in pairs)


def test_default_table_recorded_specials(default_table):
    """Words recorded in the shard summaries (out/gen/lanes/gemma): identity edge, first hw != x word, x = 1, max-ulp word, 8 - ulp, 8."""
    E = P.MufuTanh.evaluate
    assert E(0x3B800000) == 0x3B800000 and E(0x3B800001) == 0x3B800000         # 2^-8 -> identity; first non-identity word drops one ulp
    assert E(0x3F800000) == 0x3F42F848 and E(0xBF800000) == 0xBF42F848         # tanh.approx(1.0) = 0.76160097 (RN32 would be 0x3F42F7D6)
    assert E(0x3F9BFFF8) == 0x3F56DBE4                                          # the 134-ulp word (RN32 0x3F56DB5E)
    assert E(0x3E9FFFFF) == 0x3E9AFD0A and E(0x3EA00000) == 0x3E9AFC98            # the 114-ulp monotonicity drop at the 0x3EA00000 boundary
    assert 0x3E9AFD0A - 0x3E9AFC98 == 114
    assert E(0x40FFFFFF) == 0x3F7FFFFC and E(0x41000000) == 0x3F800000          # last non-saturated word; saturation onset
    assert E(0x7F800000) == 0x3F800000 and E(0xFF800000) == 0xBF800000 and E(0x7FC00000) == 0x7FFFFFFF and E(0xFFC00001) == 0x7FFFFFFF


def test_default_table_covers_every_word_class(default_table):
    rng = np.random.default_rng(917)
    words = np.concatenate([rng.integers(0, 1 << 32, 2000, dtype=np.uint64), np.array([0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FFFFFFF,
                            0xFFFFFFFF, 0x00000001, 0x3B800001, 0x40FFFFFF, 0x41000000], dtype=np.uint64)])
    for w in words.tolist():
        y = P.MufuTanh.evaluate(int(w))
        a = w & 0x7FFFFFFF
        if a > 0x7F800000:
            assert y == 0x7FFFFFFF
        elif a >= 0x41000000:
            assert y == (0x3F800000 | (w & 0x80000000))
        elif a < 0x3B800001:
            assert y == w
        else:
            assert (y & 0x80000000) == (w & 0x80000000) and 0x3B7FFFFF <= (y & 0x7FFFFFFF) <= 0x3F800000   # |tanh| <= 1, sign kept


def test_default_table_sha256_is_verified_at_load(default_table):
    t = dict(default_table["table"])
    path = P.MUFU_TANH_TABLE_DIR + "/" + t["file"]
    P._TanhShard(path, t, t["odd_symmetry"])                                    # the real header loads
    with pytest.raises(P.MufuTanhUnmeasured, match="sha256"):
        P._TanhShard(path, {**t, "sha256_file": "00" * 32}, t["odd_symmetry"])  # any mismatch is a refusal, not a warning
    with pytest.raises(P.MufuTanhUnmeasured, match="missing"):
        P._TanhShard(path + ".absent", t, t["odd_symmetry"])
    # without a verified odd symmetry the shard answers positive words only
    s = P._TanhShard(path, t, {"checked": 0, "violations": 0})
    assert s.sym_ok is False and s.lookup(0x3F800000) == 0x3F42F848 and s.lookup(0xBF800000) is None
