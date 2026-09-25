"""FA-tap exactness without a GPU: the sealed record's digest and verdict, the words a launch must write, and that the cases reach
every geometry the hidden sources declare (the GPU run is `tests/check/fa_tap_exactness_gpu.py --so ... --out ...` on a pod)."""
from __future__ import annotations

import copy
import json

import pytest

from verity_vllm.acquire.sources.hidden_source import fa3_tile
from verity_vllm.properties import fa_tap_exactness as X
from verity_vllm.commit.hidden_stream import SRC_BITS, StreamLayout
from tests.check import fa_tap_exactness_gpu as G


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
    assert X.verify_record({**edited, "digest": X._digest(edited)}) == stale_ok
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


@pytest.mark.parametrize("fa_version,D", [(2, 64), (2, 96), (2, 128), (2, 256), (3, 64), (3, 128)])
def test_every_negative_declares_a_geometry_the_kernel_did_not_launch(fa_version, D):
    negs = X.negatives(fa_version, D)
    assert len(negs) == (3 if fa_version == 2 else 5)
    for c, decl, why in negs:
        geo = X.geometry(fa_version, c, D)
        assert decl != {k: geo[k] for k in ("HB", "M", "NB", "BN", "D")}, why
