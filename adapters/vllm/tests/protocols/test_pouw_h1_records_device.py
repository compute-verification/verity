"""`pouw_h1_records.holds_on` reads a call's records on their own device and holds only what the host's read (`reason`)
accepts: honest calls hold there, and every forged one is left to the host's read, which gives the reason. On CUDA, dq's
bits are the host's."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
E = pytest.importorskip("verity_vllm.engine.int8_exact")
ZR = pytest.importorskip("verity_vllm.protocols.pouw.h1.records")

K, N, ROWS = 64, 96, 128
HALF = 4 * 32 * 32
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _call(m: int, device: str, seed: int = 0, s_w=None):
    """(zy, y [m, N] bf16, s_a [m], s_w [N]) of an honest call of m rows over ROWS slots, on `device`."""
    g = torch.Generator().manual_seed(seed)
    xq = torch.randint(-127, 128, (m, K), dtype=torch.int8, generator=g)
    w = torch.randint(-127, 128, (N, K), dtype=torch.int8, generator=g)
    s_a = torch.rand(m, generator=g) * 0.02
    s_w = torch.rand(N, generator=g) * 0.01 if s_w is None else torch.as_tensor(s_w, dtype=torch.float32)
    y, zy = E.scaled_mm_zy(xq, s_a, w, s_w, ROWS)
    return tuple(t.to(device) for t in (zy, y, s_a, s_w))


def _read(zy, y, s_a, s_w) -> tuple[bool, str | None]:
    """(the device's answer, the host's reason), held to: the device holds only records the host accepts."""
    on = ZR.holds_on(zy, ROWS, N, y, s_a, s_w)
    host = ZR.reason(zy.cpu().numpy(), ROWS, N, y.cpu().view(torch.int16).numpy(), s_a.cpu().numpy(), s_w.cpu().numpy())
    assert not on or host is None
    return on, host


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("m", [0, 1, 2, 31, 32, 33, 64, 100, ROWS])
def test_honest_records_hold_on_their_device(device, m):
    assert _read(*_call(m, device)) == (True, None)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("m", [0, 1, 33, 100])
@pytest.mark.parametrize("half", ["z", "y"])
def test_a_flipped_bit_is_left_to_the_hosts_read(device, m, half):
    """A flipped y bit never holds; a flipped z bit holds only where the host's read accepts it too (a served slot's z
    whose dq rounds to the same y, which no value check sees)."""
    rng = np.random.default_rng(m * 7 + len(half))
    for _ in range(8):
        zy, y, s_a, s_w = _call(m, device, seed=int(rng.integers(1 << 30)))
        u = int(rng.integers(zy.shape[0]))
        b = int(rng.integers(HALF)) if half == "z" else HALF + int(rng.integers(ZR.ZY_BYTES - HALF))
        zy[u, b] ^= 1 << int(rng.integers(8))
        on, host = _read(zy, y, s_a, s_w)
        assert on == (host is None)
        if half == "y":
            assert host is not None


@pytest.mark.parametrize("device", DEVICES)
def test_an_empty_slots_z_with_its_y_redone_is_left_to_the_hosts_read(device):
    zy, y, s_a, s_w = _call(1, device, seed=3)
    z, _ = ZR.split(zy.cpu().numpy(), ROWS, N)
    z[40, 5] = 9
    redone = E.zy_records(torch.from_numpy(z), E.dequantize(torch.from_numpy(z), torch.zeros(ROWS).index_copy(
        0, torch.arange(1), s_a.cpu()), s_w.cpu()))
    on, host = _read(redone.to(device), y, s_a, s_w)
    assert not on and host is not None and "empty slot" in host


@pytest.mark.parametrize("device", DEVICES)
def test_a_served_y_that_isnt_the_records_is_left_to_the_hosts_read(device):
    zy, y, s_a, s_w = _call(5, device, seed=4)
    y = y.clone()
    y[2, 7] = -y[2, 7] if float(y[2, 7]) else 1.0
    on, host = _read(zy, y, s_a, s_w)
    assert not on and "served y" in host


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("bad", [float("inf"), float("nan")])
def test_a_non_finite_weight_scale_is_left_to_the_hosts_read(device, bad):
    zy, y, s_a, s_w = _call(4, device, seed=5)
    s_w = s_w.clone()
    s_w[3] = bad
    assert not ZR.holds_on(zy, ROWS, N, y, s_a, s_w)


@pytest.mark.parametrize("device", DEVICES)
def test_negative_and_zero_weight_scales_hold(device):
    s_w = torch.rand(N, generator=torch.Generator().manual_seed(6)) * 0.01
    s_w[::3], s_w[1::7] = -s_w[::3], 0.0
    assert _read(*_call(40, device, seed=6, s_w=s_w)) == (True, None)


@pytest.mark.parametrize("device", DEVICES)
def test_what_isnt_a_calls_tensors_is_left_to_the_hosts_read(device):
    zy, y, s_a, s_w = _call(3, device, seed=7)
    assert not ZR.holds_on(zy.cpu().numpy(), ROWS, N, y, s_a, s_w)
    assert not ZR.holds_on(zy[:-1], ROWS, N, y, s_a, s_w)
    assert not ZR.holds_on(zy, ROWS, N, y.view(torch.int16), s_a, s_w)
    assert not ZR.holds_on(zy, ROWS, N, y[:2], s_a, s_w)
    assert not ZR.holds_on(zy, ROWS + 32, N, y, s_a, s_w)
    if device == "cuda":
        assert not ZR.holds_on(zy, ROWS, N, y, s_a.cpu(), s_w)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_dq_bits_on_cuda_are_the_hosts():
    """`engine.int8_exact.dequantize`'s bf16 bits on CUDA are the host's for every finite input, from the denormals'
    scales to the largest products, all but a NaN's."""
    g = torch.Generator().manual_seed(8)
    z = torch.randint(-(1 << 31), (1 << 31) - 1, (4096, 256), dtype=torch.int32, generator=g)
    z[::5] = torch.randint(-300, 300, (z[::5].shape), dtype=torch.int32, generator=g)
    e = torch.randint(-149, 100, (4096,), generator=g).to(torch.float32)
    s_a = torch.rand(4096, generator=g) * torch.exp2(e)
    s_w = (torch.rand(256, generator=g) - 0.5) * torch.exp2(torch.randint(-140, 20, (256,), generator=g).to(torch.float32))
    s_w[::17] = 0.0
    host = E.dequantize(z, s_a, s_w).view(torch.int16)
    dev = E.dequantize(z.cuda(), s_a.cuda(), s_w.cuda()).view(torch.int16).cpu()
    nan = (host & 0x7FFF) > 0x7F80
    assert torch.equal(host[~nan], dev[~nan])
    assert torch.equal(nan, (dev & 0x7FFF) > 0x7F80)
