"""``hints_device`` (torch, the prover's device) == ``witness.hints_fp4`` / ``public_vectors_fp4`` (numpy), bit for bit:
every edge family (>= 1e5 random units in total with ``FP4_HINTS_UNITS``; 2048 per family by default), the bench
instances in their chain layout (pads included), and the graph-captured path on CUDA.  Torch-gated (the laptop venv
has no torch): ``pytest archive/direct/ligero/fp4/hints_device_test.py``."""
from __future__ import annotations

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from .chain import _unit_arrays, instances_fp4  # noqa: E402
from .hints_device import PIN_NAMES, HintsDevice, hints_device_for, public_rows_torch, unit_hints_torch  # noqa: E402
from .relation import STEPS, compile_fp4_unit  # noqa: E402
from .witness import FAMILIES, hints_fp4, oracle_words, public_vectors_fp4, random_units, unit_hints  # noqa: E402

UNITS = int(os.environ.get("FP4_HINTS_UNITS", "2048"))
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(scope="module")
def system():
    return compile_fp4_unit()


def _same_as_numpy(sys, hd: HintsDevice, c, a, b) -> None:
    pub, hints, y = hd(c, a, b)
    ref_h = torch.from_numpy(hints_fp4(sys, c, a, b)).to(hints.device)
    if not torch.equal(hints, ref_h):
        bad = [n for i, (_, n) in enumerate(sys.hints) if not torch.equal(hints[i], ref_h[i])]
        raise AssertionError(f"device hints differ from numpy on rows {bad[:12]}")
    ref_p = public_vectors_fp4(a, b)
    for i, name in enumerate(PIN_NAMES):
        assert torch.equal(pub[i], torch.from_numpy(ref_p[name]).to(pub.device)), f"public row {name} differs"
    _, ref_y = unit_hints(c, a, b)
    assert torch.equal(y, torch.from_numpy(ref_y).to(y.device))


def test_pin_order_is_the_compiled_order(system):
    assert tuple(n for _, n in system.pins) == PIN_NAMES


@pytest.mark.parametrize("device", DEVICES)
def test_every_family_matches_numpy(system, device):
    rng = np.random.default_rng(20260923)
    hd = HintsDevice(system, UNITS, device, graph=False)
    for fam in FAMILIES:
        c, a, b = random_units(rng, UNITS, fam)
        _same_as_numpy(system, hd, c, a, b)


def test_words_match_the_model(system):
    """The words the device hints encode are the model's (a spot check through the numpy oracle, 64 units per family)."""
    rng = np.random.default_rng(7)
    hd = HintsDevice(system, 64, "cpu", graph=False)
    for fam in FAMILIES:
        c, a, b = random_units(rng, 64, fam)
        _, _, y = hd(c, a, b)
        assert torch.equal(y, torch.from_numpy(oracle_words(c, a, b)))


@pytest.mark.parametrize("device", DEVICES)
def test_bench_instances_chain_layout(system, device):
    """The bench instances as ``prove_vus`` lays them out (170 VUs x 24 steps + 16 pad columns at l = 4096)."""
    data = instances_fp4(170)
    a, b, c = _unit_arrays(data, 4096)
    hd = hints_device_for(system, 4096, device)
    _same_as_numpy(system, hd, c, a, b)
    _, _, y = hd(c, a, b)
    ends = y[STEPS - 1:STEPS * 170:STEPS].cpu().numpy()
    assert (ends == np.asarray([d[3] for d in data])).all(), "the chain ends are not the recorded final words"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph capture")
def test_cuda_graph_captures_and_replays(system):
    hd = hints_device_for(system, 4096, "cuda")
    assert hd.captured, "the hint graph did not capture (an eager fallback would be ~200 launches per sub-batch)"
    rng = np.random.default_rng(3)
    for fam in ("random", "span-scales", "cancel", "sparse-subnormal-acc"):
        c, a, b = random_units(rng, 4096, fam)
        _same_as_numpy(system, hd, c, a, b)          # replay after replay: the static buffers are overwritten


def test_off_domain_operands_are_refused(system):
    hd = HintsDevice(system, 32, "cpu", graph=False)
    rng = np.random.default_rng(5)
    c, a, b = random_units(rng, 32, "random")
    for pos, code in (((3, 65), 0x7F), ((7, 64), 0x80), ((5, 9), 16)):
        a_bad = a.copy()
        a_bad[pos] = code
        with pytest.raises(ValueError):
            hd(c, a_bad, b)
    with pytest.raises(ValueError):
        hd(c, np.where(a == 0, 300, a), b)          # not a byte


def test_eager_functions_agree_with_the_class(system):
    rng = np.random.default_rng(9)
    c, a, b = random_units(rng, 256, "random")
    ct, at, bt = (torch.from_numpy(np.asarray(v, dtype=np.int64)) for v in (c, a, b))
    pub = public_rows_torch(at, bt)
    hints, bad, y = unit_hints_torch(system, ct, at, bt, pub)
    assert not bool(bad.any())
    ref = HintsDevice(system, 256, "cpu", graph=False)(c, a, b)
    assert torch.equal(pub, ref[0]) and torch.equal(hints, ref[1]) and torch.equal(y, ref[2])
