"""The router-softmax tap's sources without a GPU: `build_ext.edit_block` makes exactly its eight edits, each at one whole line of the
block (refusing a missing or repeated anchor), renames the namespace, and the words the edits store land at the offsets the router-softmax
policy commits them at.  The pinned file itself is checked by sha256 when the pod builds (`cut_block`, `pinned_sources`)."""
from __future__ import annotations

import re

import pytest

from verity_vllm.acquire.router_tap_src import build_ext as BE
from verity_vllm.query import router_softmax as RS


def block(anchors=None) -> str:
    """A stand-in for the pinned namespace: every anchor line once, between filler lines."""
    lines = [a for a, _how, _new in BE.EDITS] if anchors is None else anchors
    body = "\n".join(f"  // filler {i}\n{a}" for i, a in enumerate(lines))
    return f"{BE.BLOCK_START}{body}\n{BE.BLOCK_END}"


def test_every_edit_lands_once_and_the_namespace_is_renamed():
    out = BE.edit_block(block())
    assert out.startswith("namespace vllm {\nnamespace verity_router_tap {\n") and out.endswith("} // namespace verity_router_tap\n} // namespace vllm\n")
    assert "namespace moe" not in out
    for anchor, how, new in BE.EDITS:
        assert out.count(new) == 1
        assert (anchor in out) == (how != "replace")
        if how == "before":
            assert new + "\n" + anchor in out
        elif how == "after":
            assert anchor + "\n" + new in out
    assert out.count("if constexpr (Tap") == 6 and "bool Tap = false>" in out and "float* __restrict__ tap_out = nullptr" in out


@pytest.mark.parametrize("fault", ["missing", "twice"])
def test_a_missing_or_repeated_anchor_is_refused(fault):
    anchors = [a for a, _h, _n in BE.EDITS]
    anchors = anchors[1:] if fault == "missing" else anchors + [anchors[3]]
    with pytest.raises(ValueError, match="occurs"):
        BE.edit_block(block(anchors))


def test_an_indented_copy_of_an_anchor_is_not_an_anchor():
    """`scale /= denom;` also sits, more indented, in moeTopK: only the whole line at topkGating's indentation is an anchor."""
    anchors = [a for a, _h, _n in BE.EDITS] + ["            scale /= denom;"]
    assert BE.edit_block(block(anchors)).count("scale;") == 1


def test_cut_block_refuses_text_that_is_not_the_pinned_namespace():
    with pytest.raises(ValueError, match="not the pinned one"):
        BE.cut_block(block())
    with pytest.raises(ValueError, match="opens 0"):
        BE.cut_block("int x;\n")


def test_the_stores_write_the_policys_layout():
    """Each store's offset is where `query.router_softmax.offsets` puts the value: max 0, ex 1 + column, rcp 1 + E, p 2 + E + column,
    the selected probabilities 2 + 2E + round and the scale 2 + 2E + TOPK."""
    stores = {}
    for _a, _how, new in BE.EDITS:
        m = re.search(r"tap_out\[static_cast<int64_t>\(thread_row\) \* tap_stride( \+ [^\]]+)?\] =\s*(\w+)", new)
        if m:
            stores[m.group(2)] = (m.group(1) or "").strip(" +")
    assert stores == {"thread_max": "", "row_chunk": "2 + NUM_EXPERTS + first_elt_read_by_thread + ldg * ELTS_PER_LDG * THREADS_PER_ROW + ii",
                      "reciprocal_row_sum": "1 + NUM_EXPERTS", "max_val": "2 + 2 * NUM_EXPERTS + k_idx", "scale": "2 + 2 * NUM_EXPERTS + k"}
    exps = [new for _a, _h, new in BE.EDITS if "tap_stride + 1 + first_elt_read_by_thread" in new]
    assert len(exps) == 1
    for E, K, renorm in ((64, 8, False), (128, 8, True)):
        o = RS.offsets(E, K, renorm)
        assert (o["max"], o["ex"], o["rcp"], o["p"]) == (0, 1, 1 + E, 2 + E)
        if renorm:
            assert (o["sel"], o["scale"], RS.words(E, K, renorm)) == (2 + 2 * E, 2 + 2 * E + K, 3 + 2 * E + K)
        else:
            assert RS.words(E, K, renorm) == 2 + 2 * E


def test_the_pinned_sources_and_the_launcher():
    assert set(BE.PINNED_SOURCES) == {BE.KERNEL_FILE, "csrc/cuda_compat.h", "csrc/libtorch_stable/cub_helpers.h"}
    assert all(re.fullmatch(r"[0-9a-f]{64}", v) for v in BE.PINNED_SOURCES.values()) and re.fullmatch(r"[0-9a-f]{64}", BE.BLOCK_SHA256)
    cu = (BE.HERE / "router_tap.cu").read_text()
    for op in ("topk_softmax(", "topk_softmax_tap(", "tap_version()"):
        assert op in cu
    assert "--use_fast_math" not in " ".join(BE.nvcc_flags()) and f"-DVERITY_ROUTER_TAP_VERSION={BE.TAP_VERSION}" in BE.nvcc_flags()
