"""The engine's host twin, and on a CUDA device every variant, against the CPU reference (check.py's checks)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from verity.commitments.cuda.commit_slots import slots as S
from verity.commitments.cuda.sha512_rows import check
from verity.commitments.cuda.sha512_rows import engine as E
from verity.device import compute_capability, gpu_card_reporting

#: the engine's build cache for the tests, the one the benchmarks' settings default to
BUILD = Path.home() / ".cache" / "verity" / "sha512_rows"

#: the GPU card files (`verity.device`), whose ``arch`` is the target the device library is built for
CARD_FILES = Path(__file__).resolve().parents[5] / "calibration/hardware/cards"


def arch(name: str) -> str:
    """The target a library is built for on the device reporting ``name`` (torch's ``get_device_name``): its card file's
    ``arch`` (`verity.device.gpu_card_reporting`) without ``sm_``, such as ``90a``; a device no card file names is refused."""
    card = gpu_card_reporting(name, directory=CARD_FILES)
    compute_capability(card)
    return card["arch"].removeprefix("sm_")


def _cuda() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


class TorchEngine(E.Engine):
    """The engine on torch, which verity doesn't import."""

    def runtime(self) -> Any:
        import torch
        return torch


def _engine(device: str) -> E.Engine:
    if device == "host":
        return E.Engine(device, build=BUILD)
    import torch
    return TorchEngine(device, arch(torch.cuda.get_device_name(device)), build=BUILD)


#: one layer of commit_slots' frozen H200 layout, the benchmarks' too (bench.h200_rows)
H200_LAYER = check.layout_rows(S.h200_layout(layers=1).groups[0])

ENGINES = ["host", pytest.param("cuda", marks=pytest.mark.skipif(not _cuda(), reason="no CUDA device"))]


@pytest.fixture(scope="module", params=ENGINES)
def engine(request: pytest.FixtureRequest) -> E.Engine:
    return _engine(request.param)


def test_a_cuda_device_needs_a_runtime_verity_does_not_import() -> None:
    with pytest.raises(NotImplementedError, match="kernels/gpu-code-out-of-benchmarks"):
        E.Engine("cuda", build=BUILD)


def test_a_cuda_device_needs_the_target_its_caller_names() -> None:
    class Runtime(E.Engine):
        def runtime(self) -> Any:
            return SimpleNamespace()

    with pytest.raises(ValueError, match="gpu_card_reporting"):
        Runtime("cuda", build=BUILD)


@pytest.mark.parametrize(("name", "target"), [
    ("NVIDIA H100 80GB HBM3", "90a"), ("NVIDIA H200", "90a"), ("NVIDIA B200", "100a"),
    ("NVIDIA RTX PRO 6000 Blackwell Server Edition", "120a")])
def test_the_target_is_the_card_files(name: str, target: str) -> None:
    assert arch(name) == target


def test_a_device_no_card_file_names_is_refused_by_name() -> None:
    with pytest.raises(KeyError, match="NVIDIA Tesla T4"):
        arch("NVIDIA Tesla T4")


def test_row_digests_match_the_reference_at_edge_sizes(engine: E.Engine) -> None:
    r = check.check_rows(engine)
    assert set(r["mismatches"]) == set(engine.variants)
    assert all(v == 0 for v in r["mismatches"].values()), r


def test_hm96_leaves_match_hm96s_under_the_same_salt_bytes_in_place_and_fused(engine: E.Engine) -> None:
    r = check.check_leaves(engine)
    assert r["mismatches"] == {"salts": 0, "strided": 0, "fused": 0}, r


def test_a_leaf_takes_its_salt_as_data_and_nothing_derives_one() -> None:
    """The engine has no salt of its own: ``leaves`` refuses a call with no buffer, a key in its place, or a buffer that is
    not a salt a leaf; and a leaf moves with its own salt's bytes alone."""
    e = E.Engine("host", build=BUILD)
    G, S = 3, 2
    x = e.row_digests(e.rows([[(0, 0, n)] for n in (1, 64, 128)], [np.zeros((S, 128), dtype=np.uint8)]))
    lt = e.leaf_table([bytes(64)], [0] * G, list(range(G)))
    assert not hasattr(E, "SaltKey") and not hasattr(e, "rekey")
    for bad in (None, bytes(32), np.zeros((G, S, 191), dtype=np.uint8), np.zeros((S, G, 192), dtype=np.uint8),
                np.zeros((G, S, 192), dtype=np.uint16), np.zeros((G, S, 384), dtype=np.uint8)[:, :, ::2]):
        with pytest.raises(ValueError, match="salt"):
            e.leaves(lt, x, bad)
    salts = np.random.default_rng(5).integers(0, 256, (G, S, 192), dtype=np.uint8)
    base = E.digests(e.leaves(lt, x, salts))
    for g, s, n in ((0, 0, 0), (1, 1, 100), (2, 0, 191)):
        moved = salts.copy()
        moved[g, s, n] ^= 1
        got = E.digests(e.leaves(lt, x, moved))
        assert [i for i in range(G * S) if got[i] != base[i]] == [g * S + s]


def test_a_graph_is_pointed_only_at_salts_leaves_would_take() -> None:
    """``resalt`` hands a captured graph's leaf launches the next buffer's address, so it holds that buffer to what
    ``leaves`` holds its own to before the address goes anywhere: a row too few, a row too many, a short salt, another
    dtype, salts not contiguous and strides other than the captured ones are each refused, with the library untouched."""
    e = E.Engine("host", build=BUILD)
    G, S = 3, 2
    good = np.zeros((G, S, 192), dtype=np.uint8)
    strides = tuple(int(v) for v in good.strides[:2])
    e.lib = None   # a refusal comes before any call into the library
    for bad in (None, np.zeros((G - 1, S, 192), dtype=np.uint8), np.zeros((G + 1, S, 192), dtype=np.uint8),
                np.zeros((G, S, 191), dtype=np.uint8), np.zeros((G, S, 192), dtype=np.uint16),
                np.zeros((G, S, 384), dtype=np.uint8)[:, :, ::2]):
        with pytest.raises(ValueError, match="salt"):
            e.resalt(0, None, 1, 0, bad, G, S, strides)
    with pytest.raises(ValueError, match="captured reading its salts at strides"):
        e.resalt(0, None, 1, 0, np.zeros((G, 2 * S, 192), dtype=np.uint8)[:, ::2], G, S, strides)
    with pytest.raises(AttributeError):   # salts ``leaves`` would take reach the library
        e.resalt(0, None, 1, 0, good, G, S, strides)


def test_a_devices_salts_are_its_own_uint8_tensor_at_16_byte_addresses_and_strides() -> None:
    """On a device a salt is read by 16-byte loads, so ``leaves`` and ``resalt`` refuse (``_salts``, here on a CPU tensor
    in a device engine's place) a buffer at an address or at strides that are not multiples of 16 bytes, one of another
    dtype and one on another device."""
    torch = pytest.importorskip("torch")
    G, S = 3, 2
    dev = SimpleNamespace(host=False, torch=torch, device=torch.device("cpu"), _ptr=lambda t: t.data_ptr())
    flat = torch.zeros(G * S * 192 + 16 * G * S + 16, dtype=torch.uint8)
    at = -flat.data_ptr() % 16
    good = flat[at:at + G * S * 192].reshape(G, S, 192)
    assert E.Engine._salts(dev, good, G, S) == (good.data_ptr(), S * 192, 192)
    odd = flat[at + 1:at + 1 + G * S * 192].reshape(G, S, 192)
    padded = flat[at:at + G * S * 200].reshape(G, S, 200)[:, :, :192]
    for bad, why in ((odd, "multiples of 16 bytes"), (padded, "multiples of 16 bytes"),
                     (good.to(torch.int8), "uint8 tensor on cpu"), (good.numpy(), "uint8 tensor on cpu")):
        with pytest.raises(ValueError, match=why):
            E.Engine._salts(dev, bad, G, S)
    dev.device = torch.device("cuda:0")
    with pytest.raises(ValueError, match="uint8 tensor on cuda:0"):
        E.Engine._salts(dev, good, G, S)


def test_frame_leaves_levels_pads_and_roots_match_tree_levels(engine: E.Engine) -> None:
    r = check.check_trees(engine)
    assert r["mismatches"] and not any(r["mismatches"].values()), r


@pytest.mark.skipif(not _cuda(), reason="no CUDA device")
def test_the_h200_layer_matches_on_sampled_slots() -> None:
    r = check.check_layout(_engine("cuda"), *H200_LAYER, 257, [0, 31, 32, 256])
    assert r["mismatches"] == {"x": 0, "leaves": 0}, r


def test_the_h200_layer_is_commit_slots_frozen_layout() -> None:
    sources, rows = H200_LAYER
    sizes = [sum(n for _, _, n in r) for r in rows]
    assert len(rows) == 25 and sum(sizes) == 663_824 and max(sizes) == 32768
    assert sorted(set(sizes)) == [4, 256, 8192, 16384, 16388, 32768]
    assert all(off + n <= sources[t] for r in rows for t, off, n in r)


def _cost(e: E.Engine, fn: Any) -> dict[str, int]:
    e.counts(reset=True)
    fn()
    return e.counts(reset=True)


def test_each_call_costs_its_compressions_on_the_host_twin() -> None:
    """Counted, not timed: a row of at most 111 bytes is one compression, the empty row's over a constant schedule (its
    block is padding alone), and a row whose last block is padding alone ends with one, as do 112 and 128 bytes' second;
    an hm96 leaf five (c two, h two of which the padding block's schedule is a constant, the frame leaf one from the row's
    midstate), its salt read and not hashed for, and fused with its unit's slot-tree leaf six; a unit's slot-tree leaf one from its
    frame table's midstate (two from rank 2^24), against frame_leaves' two from the IV; a node two."""
    e = E.Engine("host", build=BUILD)
    G, S = 6, 4
    t = (np.arange(S * 128) % 251).astype(np.uint8).reshape(S, 128)
    table = e.rows([[(0, 0, n)] for n in (0, 1, 16, 64, 100, 111)], [t])
    assert _cost(e, lambda: e.row_digests(table)) == {"compressions": G * S, "kw": S}
    pads = e.rows([[(0, 0, n)] for n in (110, 111, 112, 127, 128)], [t])
    assert _cost(e, lambda: e.row_digests(pads)) == {"compressions": (1 + 1 + 2 + 2 + 2) * S, "kw": 3 * S}
    x = e.row_digests(table)
    lt = e.leaf_table([bytes(64)], [0] * G, list(range(G)))
    salts = np.zeros((G, S, 192), dtype=np.uint8)
    assert _cost(e, lambda: e.leaves(lt, x, salts)) == {"compressions": 5 * G * S, "kw": G * S}
    ft = e.frame_table([bytes(range(64))], 100, G)
    assert _cost(e, lambda: e.frame_table([bytes(range(64))], 100, G)) == {"compressions": G, "kw": 0}
    dest = e.u32s(np.arange(G))
    assert _cost(e, lambda: e.leaves(lt, x, salts, None, 0, ft, dest, e._words(S, G))) == {"compressions": 6 * G * S, "kw": G * S}
    v = e._words(G, S)
    assert _cost(e, lambda: e.frame_units(ft, np.arange(G), v, e._words(S, G)))["compressions"] == G * S
    assert _cost(e, lambda: e.frame_leaves(v, [bytes(64)], 100))["compressions"] == 2 * G * S
    far = e.frame_table([bytes(64)], 1 << 24, G)
    assert _cost(e, lambda: e.frame_units(far, np.arange(G), v, e._words(S, G)))["compressions"] == 2 * G * S
    mids = e.node_mids([bytes(64)], 3)
    assert _cost(e, lambda: e.level(e._words(S, 2 * G), [bytes(64)], 3, (1 << 64) - G, mids))["compressions"] == 2 * G * S


def test_words_and_digests_round_trip() -> None:
    d = [bytes(range(i, i + 64)) for i in range(3)]
    assert E.digests(E.words(d)) == d
    assert E.digests(np.stack([E.words(d)])) == d


def test_a_variant_name_is_four_modes_and_an_optional_straight_line_mark() -> None:
    assert E.variant_flags("2211") == ("-DSX_V=90", "-DSX_ROLL=1")
    assert E.variant_flags("0000s") == ("-DSX_V=0", "-DSX_ROLL=0")
    assert E.variant_flags("0033") == ("-DSX_V=240", "-DSX_ROLL=1")
    with pytest.raises(ValueError):
        E.variant_flags("4000")


def test_a_row_outside_its_tensor_is_refused() -> None:
    e = E.Engine("host", build=BUILD)
    with pytest.raises(ValueError, match="outside"):
        e.rows([[(0, 8, 9)]], [np.zeros((2, 16), dtype=np.uint8)])
    with pytest.raises(ValueError, match="one or two parts"):
        e.rows([[(0, 0, 1)] * 3], [np.zeros((2, 16), dtype=np.uint8)])
