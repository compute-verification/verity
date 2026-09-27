"""FA-tap exactness without a GPU: the sealed record's digest and verdict, the words a launch must write, and that the cases reach
every geometry the hidden sources declare (the GPU run is `tests/properties/fa_tap_exactness_gpu.py --so ... --out ...` on a pod)."""
from __future__ import annotations

import copy
import json

import pytest

from verity_vllm.acquire.sources.hidden_source import fa3_tile
from verity_vllm.properties import fa_tap_exactness as X
from verity_vllm.properties.record import digest_of
from verity_vllm.commit.hidden_stream import SRC_BITS, StreamLayout
from tests.properties import fa_tap_exactness_gpu as G


def _record(**fields):
    case = {"case": "prefill_short", "D": 64, "skipped": 0, "out_lse_eq_installed": True, "out_lse_eq_untapped": True,
            "spill_outside_layout": 0, "expected_but_unwritten": 0}
    neg = {"case": "prefill_short", "D": 64, "why": "wrong kBlockN", "skipped": 1, "buffer_untouched": True, "out_eq_installed": True}
    case["ok"], neg["ok"] = X.case_ok(case), X.negative_ok(neg)
    return X.seal({"schema": X.SCHEMA, "backend": "fa2", "cases": [case], "negatives": [neg], **fields})


def test_sealed_record_verifies_and_every_edit_is_named():
    doc = _record()
    assert doc["ok"] is True and X.verify_record(doc) == []
    edited = copy.deepcopy(doc)
    edited["cases"][0]["out_lse_eq_installed"] = False
    stale_ok = ["cases prefill_short hd64: ok=True but its fields say False", "ok=True but the entries say False"]
    assert X.verify_record(edited) == ["digest does not match the record", *stale_ok]
    assert X.verify_record({**edited, "digest": digest_of(edited)}) == stale_ok
    assert X.seal(edited)["ok"] is False


@pytest.mark.parametrize("entries", [{"negatives": []}, {"cases": []}, {"cases": [{"case": "decode_gqa", "D": 128, "error": "boom", "ok": False}]}])
def test_no_negatives_no_cases_or_an_error_is_not_ok(entries):
    doc = _record()
    doc = X.seal({**doc, **entries})
    assert doc["ok"] is False and X.verify_record(doc) == []


def test_verify_cli(tmp_path, capsys):
    path = tmp_path / X.RECORD
    path.write_text(json.dumps(_record()))
    assert G.main(["--verify", str(path)]) == 0
    doc = _record()
    doc["cases"][0]["skipped"] = 1
    path.write_text(json.dumps(doc))
    assert G.main(["--verify", str(path)]) == 1 and "digest does not match" in capsys.readouterr().out


def test_fa2_causal_rows_visit_their_m_tiles_key_blocks():
    """hd 128 (kBlockN 64), sq = sk = 129: rows of M tile 0 see one key block, tile 1 two, row 128 (tile 2) all three."""
    c = X._case("t", 1, 1, [129], [129])
    geo = X.geometry(2, c, 128)
    assert {k: geo[k] for k in ("HB", "M", "NB", "BN", "kBM", "swapped")} == {"HB": 1, "M": 129, "NB": 3, "BN": 64, "kBM": 64, "swapped": False}
    rows = list(X.visited_rows(c, geo))
    assert [nbm for _, r, nbm in rows] == [1] * 64 + [2] * 64 + [3]
    layout = StreamLayout(1, 129, 3, 64, 128)
    mask = X.expected_written(layout, SRC_BITS["S"] | SRC_BITS["ROW"] | SRC_BITS["FIN"], iter(rows))
    o = layout.offsets
    blocks = 64 * 1 + 64 * 2 + 3
    assert mask[o["S"]:o["P"]].sum() == blocks * 64 and mask[o["ROW"]:o["O"]].sum() == blocks * 7 and mask[o["FIN"]:o["total"]].sum() == 129 * 4
    assert mask.sum() == blocks * (64 + 7) + 129 * 4 and not mask[o["total"]:].any()


def test_chunked_rows_and_padding_rows():
    """Chunked prefill (sq 17 < sk 300): the tile's causal limit reaches every key block; rows at or past a sequence's own length are
    never written (M is the batch maximum)."""
    c = X._case("t", 2, 2, [17, 5], [300, 5])
    geo = X.geometry(2, c, 128)
    rows = list(X.visited_rows(c, geo))
    assert geo["M"] == 17 and geo["HB"] == 4
    assert {(slab, nbm) for slab, _, nbm in rows} == {(0, 5), (1, 5), (2, 1), (3, 1)}
    assert sorted({r for slab, r, _ in rows if slab >= 2}) == [0, 1, 2, 3, 4]


def test_fa3_packed_rows_follow_the_packgqa_divide():
    """FA3 hd 128, 2 query rows per sequence, g = 4 heads per KV head: 8 packed rows per slab, one M tile of 64, n_block_max from the
    tile's last query row ((64 - 1) // 4 + 1 = 16) -- every key block of these short sequences."""
    c = X._case("t", 8, 2, [2, 2], [130, 2])
    geo = X.geometry(3, c, 128)
    assert (geo["HB"], geo["M"], geo["g"], (geo["kBM"], geo["BN"])) == (4, 8, 4, fa3_tile(128, 2, 4, 1))
    rows = list(X.visited_rows(c, geo))
    assert len(rows) == 4 * 8 and {(slab // 2, nbm) for slab, _, nbm in rows} == {(0, 2), (1, 1)}


def test_fa2_cases_reach_the_swapped_and_unswapped_decodes():
    by = {c["name"]: c for c in X.cases(2)}
    for D in (64, 96, 128, 256):
        assert X.geometry(2, by["decode_gqa"], D)["swapped"] is True
        assert X.geometry(2, by["g2_decode_gqa_softcap_window"], D)["swapped"] is False
        assert X.geometry(2, by["decode_mha"], D)["swapped"] is False
    assert all(c["window"][0] > max(c["seqlens_k"]) for c in X.cases(2) if c["window"][0] >= 0)


def test_fa3_cases_reach_both_tiles_of_every_head_dim_and_no_disarmed_launch():
    for D in (64, 128):
        assert {(X.geometry(3, c, D)["kBM"], X.geometry(3, c, D)["BN"]) for c in X.cases(3)} == {fa3_tile(D, 1, 1, 1), fa3_tile(D, 1000, 1, 1)}
    assert all(c["softcap"] == 0.0 and c["window"] == [-1, -1] for c in X.cases(3))


def test_quick_is_a_subset():
    for v in (2, 3):
        names = [c["name"] for c in X.cases(v, quick=True)]
        assert names and set(names) <= set(X.QUICK) and set(names) <= {c["name"] for c in X.cases(v)}


def _guarded_stream(rows, row_max):
    """A stream whose ROW entries carry `row_max` (per visited block, kernel visit order) and the guarded max in word 3."""
    import numpy as np

    from verity_vllm.program.kernels.derived_rows import guard_neg_inf_zero

    layout = StreamLayout(1, len(rows), 2, 128, 64)
    w = np.full(layout.padded_words(), X.FILL, dtype=np.uint32)
    R = X._row_plane(w, layout)
    it = iter(row_max)
    for slab, r, nbm in rows:
        for b in range(nbm - 1, -1, -1):
            m = next(it)
            R[slab, r, b, 0] = m
            R[slab, r, b, 3] = guard_neg_inf_zero(m) if b < nbm - 1 else 0
    return w, layout


def test_check_row3_is_the_guard_after_the_first_block_and_names_a_wrong_word():
    rows = [(0, 0, 1), (0, 1, 2), (0, 2, 2)]
    # visit order: row 0's only block, then rows 1 and 2 first (block 1) and later (block 0): 1.0 | +inf, -inf | -0, NaN
    w, layout = _guarded_stream(rows, [0x3F800000, 0x7F800000, 0xFF800000, 0x80000000, 0x7FFFFFFF])
    r = X.check_row3(w, layout, iter(rows))
    assert r == {"guard_words": 2, "guard_mismatch": 0, "first_block_nonzero": 0, "ir_sampled": 2, "ir_mismatch": 0}
    assert X._row_plane(w, layout)[0, 1, 0, 3] == 0                                  # -inf is guarded to +0
    bad = w.copy()
    X._row_plane(bad, layout)[0, 2, 0, 3] = 0x7F800000
    X._row_plane(bad, layout)[0, 0, 0, 3] = 1
    r = X.check_row3(bad, layout, iter(rows))
    assert r["guard_mismatch"] == 1 and r["first_mismatch"] == {"slab": 0, "row": 2, "block": 0, "row_max": 0x7FFFFFFF, "word3": 0x7F800000}
    assert r["first_block_nonzero"] == 1 and r["ir_mismatch"] == 1


def test_the_guarded_build_writes_word_3_at_every_visited_block_and_the_masked_digest_ignores_only_it():
    rows = [(0, 0, 1), (0, 1, 2), (0, 2, 2)]
    w, layout = _guarded_stream(rows, [0x3F800000] * 5)
    plain = X.expected_written(layout, SRC_BITS["ROW"], iter(rows))
    guarded = X.expected_written(layout, SRC_BITS["ROW"], iter(rows), row3=1)
    assert guarded.sum() - plain.sum() == 5 and not (plain & ~guarded).any()
    base = w.copy()
    X._row_plane(base, layout)[..., 3] = X.FILL
    assert X.masked_digest(w, layout) == X.masked_digest(base, layout)
    X._row_plane(base, layout)[0, 1, 1, 1] = 7
    assert X.masked_digest(w, layout) != X.masked_digest(base, layout)


def _guarded_record(**row3):
    doc = _record()
    case = dict(doc["cases"][0], row3={"guard_words": 3, "guard_mismatch": 0, "first_block_nonzero": 0, "ir_sampled": 3, "ir_mismatch": 0,
                                       "stream_eq_baseline": True, **row3})
    case["ok"] = X.case_ok(case)
    return X.seal({**doc, "cases": [case], "tap": {"row3": 1}, "baseline": {"digest": "b" * 64}})


def test_a_guarded_record_needs_its_baseline_and_word_3_checked_in_every_case():
    assert _guarded_record()["ok"] is True and X.verify_record(_guarded_record()) == []
    assert _guarded_record(guard_words=0, ir_sampled=0)["ok"] is True                   # every row visits one key block: nothing to sample
    for bad in ({"stream_eq_baseline": None}, {"stream_eq_baseline": False}, {"guard_mismatch": 1}, {"ir_sampled": 0}):
        assert _guarded_record(**bad)["ok"] is False
    assert X.seal({**_guarded_record(), "baseline": {}})["ok"] is False
    assert X.seal({**_record(), "tap": {"row3": 1}, "baseline": {"digest": "b" * 64}})["ok"] is False    # a case without word 3 checked


def _ms_stream(rows, row_max, scale_log2):
    """A stream with the MS class: ROW word 0 = `row_max` per visited block (kernel visit order), MS = the IR's max_scaled of it."""
    import numpy as np

    from verity_vllm.program.kernels.derived_rows import guard_neg_inf_zero, mul_ftz

    layout = StreamLayout(1, len(rows), 2, 128, 64, ms=True)
    w = np.full(layout.padded_words(), X.FILL, dtype=np.uint32)
    R, o = X._row_plane(w, layout), layout.offsets
    MS = w[o["MS"]:o["total"]].reshape(1, len(rows), 2)
    it = iter(row_max)
    for slab, r, nbm in rows:
        for b in range(nbm - 1, -1, -1):
            R[slab, r, b, 0] = m = next(it)
            MS[slab, r, b] = mul_ftz(guard_neg_inf_zero(np.uint32(m)), np.uint32(scale_log2))[0]
    return w, layout, MS


def test_check_ms_is_the_irs_max_scaled_of_every_visited_block_and_names_a_wrong_word():
    rows = [(0, 0, 1), (0, 1, 2), (0, 2, 2)]
    sl2 = X.scale_log2_bits({"softcap": 0.0}, 64)
    assert sl2 == X.scale_log2_bits({"softcap": 0.0}, 64) != X.scale_log2_bits({"softcap": 50.0}, 64)
    w, layout, MS = _ms_stream(rows, [0x3F800000, 0x7F800000, 0xFF800000, 0x80000000, 0x7FFFFFFF], sl2)
    assert MS[0, 1, 0] == 0                                                            # max -inf: the guard's 0, times the scale
    r = X.check_ms(w, layout, iter(rows), sl2)
    assert r == {"ms_words": 5, "ms_mismatch": 0, "ms_mismatch_neg_inf_max": 0, "ir_sampled": 5, "ir_mismatch": 0} and X.ms_ok({"ms": r})
    bad = w.copy()
    bad[layout.word_index("MS", 0, 1, block=0)] = 0xFF800000                           # -inf * scale, the unguarded product
    r = X.check_ms(bad, layout, iter(rows), sl2)
    assert (r["ms_mismatch"], r["ms_mismatch_neg_inf_max"], r["ir_mismatch"]) == (1, 1, 1) and not X.ms_ok({"ms": r})
    assert r["first_mismatch"] == {"slab": 0, "row": 1, "block": 0, "row_max": 0xFF800000, "ms": 0xFF800000}


def test_the_ms_build_writes_its_class_at_every_visited_block_and_the_masked_digest_is_the_six_classes():
    rows = [(0, 0, 1), (0, 1, 2), (0, 2, 2)]
    w, layout, _ = _ms_stream(rows, [0x3F800000] * 5, X.scale_log2_bits({"softcap": 0.0}, 64))
    six = StreamLayout(1, 3, 2, 128, 64)
    assert X.expected_written(layout, 123, iter(rows)).sum() - X.expected_written(six, 59, iter(rows)).sum() == 5
    assert X.expected_written(layout, 123, iter(rows)).sum() - X.expected_written(layout, 59, iter(rows)).sum() == 5
    assert not X.expected_written(layout, 123, iter(rows))[layout.total_words:].any()
    import numpy as np

    base = np.full(six.padded_words(), X.FILL, dtype=np.uint32)
    base[:six.total_words] = w[:six.total_words]
    assert X.masked_digest(w, layout) == X.masked_digest(base, six)
    base[six.offsets["FIN"]] = 3
    assert X.masked_digest(w, layout) != X.masked_digest(base, six)


def test_a_record_of_a_build_with_the_ms_class_checks_it_in_every_case():
    ms = {"ms_words": 4, "ms_mismatch": 0, "ms_mismatch_neg_inf_max": 0, "ir_sampled": 4, "ir_mismatch": 0}
    doc = _guarded_record()

    def sealed(**m):
        case = dict(doc["cases"][0], ms=dict(ms, **m))
        case["ok"] = X.case_ok(case)
        return X.seal({**doc, "cases": [case], "tap": {"row3": 1, "src": 123}})

    assert sealed()["ok"] is True and X.verify_record(sealed()) == []
    assert sealed(ms_words=0, ir_sampled=0)["ok"] is True
    for bad in ({"ms_mismatch": 1}, {"ir_mismatch": 1}, {"ir_sampled": 0}):
        assert sealed(**bad)["ok"] is False
    assert X.seal({**doc, "tap": {"row3": 1, "src": 123}})["ok"] is False             # a case without the MS class checked


def test_fa3_masked_from_follows_the_packed_tile():
    """FA3 hd 128, 130 prefill rows over 130 keys, g = 2: 260 packed rows in 128-row tiles; the tile starting at packed row 256 (query row
    128) guards key blocks from 1 up, the earlier tiles every block; a decode over 257 keys guards only its first block (2)."""
    c = X._case("t", 4, 2, [130, 1], [130, 257])
    geo = X.geometry(3, c, 128)
    mf = X.masked_from(c, geo)
    assert (geo["kBM"], geo["BN"], geo["g"]) == (128, 128, 2)
    assert {mf[(0, r)] for r in range(256)} == {0} and {mf[(0, r)] for r in (256, 259)} == {1} and mf[(2, 0)] == mf[(3, 1)] == 2


def test_check_ms_guards_only_the_check_inf_blocks_under_the_per_iteration_construction():
    import numpy as np

    rows = [(0, 0, 1), (0, 1, 2), (0, 2, 2)]
    sl2 = X.scale_log2_bits({"softcap": 0.0}, 64)
    # row 1 visits block 1 with max +inf, then block 0 with -inf; row 2 -0 then NaN
    w, layout, MS = _ms_stream(rows, [0x3F800000, 0x7F800000, 0xFF800000, 0x80000000, 0x7FFFFFFF], sl2)
    unguarded = w.copy()
    unguarded[layout.word_index("MS", 0, 1, block=0)] = 0xFF800000                    # FA3's unmasked iteration: -inf * scale
    mf = {(0, 0): 0, (0, 1): 1, (0, 2): 1}                                             # block 0 of rows 1 and 2 unguarded
    r = X.check_ms(unguarded, layout, iter(rows), sl2, masked_from=mf)
    assert (r["ms_mismatch"], r["ir_mismatch"], r["unguarded_words"]) == (0, 0, 2) and X.ms_ok({"ms": r})
    assert X.check_ms(w, layout, iter(rows), sl2, masked_from=mf)["ms_mismatch"] == 1   # the guarded value is wrong there
    assert X.check_ms(unguarded, layout, iter(rows), sl2)["ms_mismatch"] == 1           # and the default construction says so


def test_a_per_iteration_record_checks_every_output_row():
    doc = _guarded_record()
    ms = {"ms_words": 4, "ms_mismatch": 0, "ms_mismatch_neg_inf_max": 0, "ir_sampled": 4, "ir_mismatch": 0}

    def sealed(**out):
        case = dict(doc["cases"][0], ms=ms, **({"out_vs_ir": out} if out else {}))
        case["ok"] = X.case_ok(case)
        return X.seal({**doc, "cases": [case], "tap": {"row3": 1, "src": 123}, "construction": "check-inf-per-iteration"})

    good = {"rows": 3, "heads": 12, "mismatch": 0, "reference_heads": 2, "unevaluated": 0}
    assert sealed(**good)["ok"] is True and X.verify_record(sealed(**good)) == []
    for bad in ({"mismatch": 1}, {"unevaluated": 1}, {"rows": 0}):
        assert sealed(**dict(good, **bad))["ok"] is False
    assert sealed()["ok"] is False                                                     # a case without the output check


def test_the_edge_cases_name_query_tokens_of_their_own_launch():
    for v in (2, 3):
        edge = [c for c in X.cases(v) if c.get("special")]
        assert {c["name"] for c in edge} == ({"edge_rows", "g2_edge_rows_softcap"} if v == 2 else {"edge_rows"})
        for c in edge:
            toks = [t for ts in c["special"].values() for t in ts]
            assert len(set(toks)) == len(toks) and max(toks) < sum(c["seqlens_q"])


@pytest.mark.parametrize("fa_version,D", [(2, 64), (2, 96), (2, 128), (2, 256), (3, 64), (3, 128)])
def test_every_negative_declares_a_geometry_the_kernel_did_not_launch(fa_version, D):
    negs = X.negatives(fa_version, D)
    assert len(negs) == (3 if fa_version == 2 else 5)
    for c, decl, why in negs:
        geo = X.geometry(fa_version, c, D)
        assert decl != {k: geo[k] for k in ("HB", "M", "NB", "BN", "D")}, why
