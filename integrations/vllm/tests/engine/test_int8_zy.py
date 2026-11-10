"""`verity_int8::scaled_mm_zy`: scaled_mm's y and, from the same product, the call's z/y records (`int8_exact.zy_records`):
one record a 32 x 32 unit over the call's T slots, its exact s32 z over the whole K, then its
y = dq(z) in bf16, both row-major and little-endian, units in row-block-major order, empty slots zero bytes."""
from __future__ import annotations

import functools

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from verity_vllm.engine import int8_exact as E
from verity_vllm.engine import int8_ops as OPS
from verity_vllm.protocols.pouw import credit_gate as CG

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
UR, UC = E.UNIT
CELLS = UR * UC


class _Ops(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.seen: list[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.seen.append(str(func))
        return func(*args, **(kwargs or {}))


def _args(m, k, n, seed=0, device="cpu"):
    """A call's (xq, s_a) from row_quant of a random x, and a random weight w [n, k] with its scales."""
    g = torch.Generator(device).manual_seed(seed)
    x = (torch.randn(m, k, generator=g, device=device) * 3).bfloat16()
    w = torch.randint(-127, 128, (n, k), generator=g, dtype=torch.int8, device=device)
    sw = torch.rand(n, generator=g, device=device) * 0.01 + 1e-4
    xq, sa = OPS.row_quant(x)
    return xq, sa, w, sw


def _slots(xq, sa, rows):
    """The call's T slots: row i < m is xq[i] under sa[i], every later one zero codes under scale 0."""
    slots, scales = xq.new_zeros((rows, xq.shape[1])), sa.new_zeros(rows)
    slots[:xq.shape[0]], scales[:xq.shape[0]] = xq, sa.reshape(-1)
    return slots, scales


def _unit(zy, u):
    """Unit u's record read as the layout states it: (z [UR, UC] '<i4', y bits [UR, UC] '<i2')."""
    rec = zy[u].cpu().numpy().tobytes()
    assert len(rec) == E.ZY_BYTES == 6 * CELLS
    return (np.frombuffer(rec, "<i4", CELLS).reshape(UR, UC),
            np.frombuffer(rec, "<i2", CELLS, offset=4 * CELLS).reshape(UR, UC))


def _differing(zy, ref):
    """The units whose record bytes differ from the reference's."""
    return sorted(torch.nonzero((zy != ref).any(dim=1)).flatten().tolist())


def _not_dq(zy, scales, sw, rows, n):
    """The units whose y is not dq(z, s_a, s_w) of their own z: a check of the records alone."""
    z, y = E.zy_split(zy, rows, n)
    bad = (y.view(torch.int16) != E.dequantize(z, scales, sw).view(torch.int16))
    return sorted(torch.nonzero(bad.reshape(rows // UR, UR, n // UC, UC).any(dim=3).any(dim=1).flatten()).flatten().tolist())


@pytest.mark.parametrize("m,rows,n,k", [(5, 32, 96, 64), (33, 64, 64, 96), (64, 64, 32, 64), (1, 96, 64, 40)])
def test_each_record_is_its_units_exact_dots_then_their_dequant(m, rows, n, k):
    """Unit (b, c) is record b (N / 32) + c; its first 4096 bytes are the s32 dots of slot rows 32b.. by weight rows 32c..
    over the whole K, row-major and little-endian (against an int64 product), its last 2048 the bf16 bits of dq(z), which
    are the served y's block on live rows and zero on empty slots."""
    xq, sa, w, sw = _args(m, k, n, seed=m + rows)
    y, zy = OPS.scaled_mm_zy(xq, sa, w, sw, rows)
    assert torch.equal(y.view(torch.int16), OPS.scaled_mm(xq, sa, w, sw).view(torch.int16))
    assert zy.dtype == torch.uint8 and tuple(zy.shape) == ((rows // UR) * (n // UC), E.ZY_BYTES)
    slots, scales = _slots(xq, sa, rows)
    dots = (slots.long() @ w.long().T).to(torch.int32)
    dq = E.dequantize(dots, scales, sw).view(torch.int16)
    for u in range(zy.shape[0]):
        b, c = divmod(u, n // UC)
        z_u, y_u = _unit(zy, u)
        assert np.array_equal(z_u, dots[b * UR:(b + 1) * UR, c * UC:(c + 1) * UC].numpy()), u
        assert np.array_equal(y_u, dq[b * UR:(b + 1) * UR, c * UC:(c + 1) * UC].numpy()), u
    yr = E.zy_split(zy, rows, n)[1]
    assert torch.equal(yr[:m].view(torch.int16), y.view(torch.int16))
    assert not yr[m:].view(torch.int16).any() and not E.zy_split(zy, rows, n)[0][m:].any()


@pytest.mark.slow
def test_each_records_z_is_the_compiled_units_z_in_its_cells_order():
    """Record u's z words are `pouw_h1.Call.z(Y, u)`: the compiled Program's unit u, its cells in its tile's pair order, so
    a leaf of the record is the unit's z as H1's statement reads it.  The only test that holds the layout to the compiled
    Program rather than to the module doc; about 30 s, compiling the smallest H1 call (K = 64, two by two units)."""
    from verity_vllm.protocols.pouw import h1 as PH
    m, rows, n, k = 40, 64, 64, 64
    xq, sa, w, sw = _args(m, k, n, seed=11)
    zy = OPS.scaled_mm_zy(xq, sa, w, sw, rows)[1]
    call = PH.Call(0, 0, 0, "q", m, k, n, 0, rows)
    assert len(call.compiled().units) == zy.shape[0] == call.units
    dots = E.int_mm(xq, w).numpy()
    for u in range(zy.shape[0]):
        assert np.array_equal(_unit(zy, u)[0].reshape(-1), call.z(dots, u)), u


def test_the_records_split_back_into_z_and_y():
    g = torch.Generator().manual_seed(1)
    z = torch.randint(-2 ** 31, 2 ** 31 - 1, (64, 96), generator=g, dtype=torch.int32)
    y = torch.randn(64, 96, generator=g).bfloat16()
    got = E.zy_split(E.zy_records(z, y), 64, 96)
    assert torch.equal(got[0], z) and torch.equal(got[1].view(torch.int16), y.view(torch.int16))


def test_a_tampered_unit_is_found_by_the_record_compare():
    """A flipped z word, a flipped y bit or two swapped records: the compare against the reference records names exactly
    the units touched, and the records' own check (y = dq(z)) names a unit whose z or y alone changed."""
    m, rows, n, k = 40, 64, 96, 64
    xq, sa, w, sw = _args(m, k, n, seed=7)
    ref = OPS.scaled_mm_zy(xq, sa, w, sw, rows)[1]
    scales = _slots(xq, sa, rows)[1]
    assert _differing(ref, E.scaled_mm_zy(xq, sa, w, sw, rows)[1]) == [] and _not_dq(ref, scales, sw, rows, n) == []
    z_word = ref.clone()
    z_word[4, 4 * 37 + 3] ^= 0x40
    assert _differing(z_word, ref) == [4] and _not_dq(z_word, scales, sw, rows, n) == [4]
    y_bit = ref.clone()
    y_bit[2, 4 * CELLS + 2 * 100] ^= 1
    assert _differing(y_bit, ref) == [2] and _not_dq(y_bit, scales, sw, rows, n) == [2]
    swapped = ref.clone()
    swapped[[0, 5]] = ref[[5, 0]]
    assert _differing(swapped, ref) == [0, 5]


def test_the_op_is_one_credited_gemm_that_wrote_y_and_the_records():
    xq, sa, w, sw = _args(3, 64, 32)
    with CG.Trace("q") as trace:
        y, zy = OPS.scaled_mm_zy(xq, sa, w, sw, 32)
    v = CG.verdict(trace, y, zy)
    assert v["credit"] and v["kernel"] == "verity_int8::scaled_mm_zy" and v["gemms"] == ["verity_int8::scaled_mm_zy"], v
    assert CG.verdict(trace, y, zy[1:3])["credit"]
    with _Ops() as mode:
        OPS.scaled_mm_zy(xq, sa, w, sw, 32)
    assert mode.seen == ["verity_int8.scaled_mm_zy.default"]
    torch.library.opcheck(OPS.scaled_mm_zy, (xq, sa, w, sw, 64))


def test_partial_units_and_rows_past_the_slots_are_refused():
    xq, sa, w, sw = _args(5, 64, 48)
    with pytest.raises(ValueError, match="whole 32 x 32 units"):
        OPS.scaled_mm_zy(xq, sa, w, sw, 32)
    xq, sa, w, sw = _args(40, 64, 32)
    with pytest.raises(ValueError, match="whole 32 x 32 units"):
        OPS.scaled_mm_zy(xq, sa, w, sw, 48)
    with pytest.raises(ValueError, match="past the call's 32 slots"):
        OPS.scaled_mm_zy(xq, sa, w, sw, 32)


def test_a_call_of_no_rows_has_only_zero_records():
    xq, sa, w, sw = _args(0, 64, 64)
    y, zy = OPS.scaled_mm_zy(xq, sa, w, sw, 32)
    assert tuple(y.shape) == (0, 64) and tuple(zy.shape) == (2, E.ZY_BYTES) and not zy.any()


# Llama-3.1-8B's linears as (K, N): qkv, o, gate_up, down
LLAMA = {"qkv": (4096, 6144), "o": (4096, 4096), "gate_up": (4096, 28672), "down": (14336, 4096)}


def _rows(m: int) -> int:
    return max(64, -(-m // UR) * UR)


@functools.cache
def _weight(k: int, n: int):
    g = torch.Generator("cuda").manual_seed(k * 7 + n)
    return (torch.randint(-127, 128, (n, k), generator=g, dtype=torch.int8, device="cuda"),
            torch.rand(n, generator=g, device="cuda") * 0.01 + 1e-4)


def _cuda_call(m, k, n, seed=0):
    g = torch.Generator("cuda").manual_seed(seed)
    xq, sa = OPS.row_quant((torch.randn(m, k, generator=g, device="cuda") * 3).bfloat16())
    return (xq, sa, *_weight(k, n))


@cuda
@pytest.mark.parametrize("m", [1, 17, 32, 64, 256])
@pytest.mark.parametrize("linear", list(LLAMA))
def test_cuda_records_are_int_mm_blocks_and_the_served_y(linear, m):
    """On CUDA (Triton, `int8_fused.product_zy`) at Llama-3.1-8B's linears and the picked config: y bit for bit
    `verity_int8::scaled_mm`'s; each unit's z the 32 x 32 block of ``torch._int_mm`` over the whole K, its y dq of it and the
    served y's block (zero past M); the records the op's torch body's; a flipped byte found by the compare."""
    k, n = LLAMA[linear]
    rows = _rows(m)
    xq, sa, w, sw = _cuda_call(m, k, n, seed=m)
    y, zy = OPS.scaled_mm_zy(xq, sa, w, sw, rows)
    served = OPS.scaled_mm(xq, sa, w, sw)
    assert torch.equal(y.view(torch.int16), served.view(torch.int16))
    slots, scales = _slots(xq, sa, rows)
    dots = torch._int_mm(slots, w.t())
    z, yr = E.zy_split(zy, rows, n)
    assert torch.equal(z, dots)
    assert torch.equal(yr.view(torch.int16), E.dequantize(dots, scales, sw).view(torch.int16))
    assert torch.equal(yr[:m].view(torch.int16), served.view(torch.int16)) and not yr[m:].view(torch.int16).any()
    ref = E.scaled_mm_zy(xq, sa, w, sw, rows)[1]
    assert torch.equal(zy, ref)
    u = (m * 31 + 5) % zy.shape[0]
    zy[u, 4 * (m % CELLS)] ^= 0x10
    assert _differing(zy, ref) == [u]


def _configs():
    from verity_vllm.engine import int8_fused as F
    cfgs = ["16x64x128:w4:s4:k1", "32x32x256:w4:s3:k2", "16x32x256:w4:s3:k4", "64x64x128:w4:s3:k3",
            "128x128x64:w4:s3:k1", "128x64x128:w4:s3:k1:t1"]
    if F.arch() in ("sm_90", "sm_100"):
        cfgs.append("64x64x128:w4:s3:k1:t2")
    return cfgs


@cuda
@pytest.mark.parametrize("m,k,n", [(17, 4096, 6144), (1, 14336, 4096), (100, 4096, 4096)])
def test_cuda_every_config_writes_the_same_records(m, k, n):
    """Pointer tiles narrower and wider than a unit, split-K 2, 3 and 4 (split equal to no split), row tiles past T and
    the TMA kernel: into 0xA5-poisoned y and records, twice through one workspace, the same y as `scaled_mm` at that config
    and the same records as the unsplit torch body, the workspace zero again after, and every kernel's PTX passes
    `int8_fused.ptx_check`."""
    from verity_vllm.engine import int8_fused as F
    rows = _rows(m)
    xq, sa, w, sw = _cuda_call(m, k, n, seed=k + m)
    ref = E.scaled_mm_zy(xq, sa, w, sw, rows)[1]
    work = F.Workspace("cuda", words=m * n)
    for text in _configs():
        cfg = F.Cfg.parse(text)
        served = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
        F.scaled_mm(xq, sa, w, sw, served, cfg, work)
        y = torch.full((m, 2 * n), 0xA5, dtype=torch.uint8, device="cuda").view(torch.bfloat16)
        zy = torch.full((E.zy_units(m, n, rows), E.ZY_BYTES), 0xA5, dtype=torch.uint8, device="cuda")
        for _ in range(2):
            kernel = F.scaled_mm_zy(xq, sa, w, sw, y, zy, rows, cfg, work)
            assert torch.equal(y.view(torch.int16), served.view(torch.int16)), text
            assert _differing(zy, ref) == [], text
        assert not work.ws.any() and not work.cnt.any(), text
        ptx = F.ptx_check(kernel)
        assert ptx["ok"], (text, ptx)


@cuda
def test_cuda_op_is_one_credited_gemm_and_replays_in_a_graph():
    """On CUDA the op is still one credited GEMM writing y and the records, and a CUDA graph of it (as vLLM captures the
    decode step) writes the records of the inputs it is replayed on."""
    m, rows = 32, 64
    k, n = LLAMA["qkv"]
    xq, sa, w, sw = _cuda_call(m, k, n, seed=3)
    with CG.Trace("qkv") as trace:
        y, zy = OPS.scaled_mm_zy(xq, sa, w, sw, rows)
    v = CG.verdict(trace, y, zy)
    assert v["credit"] and v["gemms"] == ["verity_int8::scaled_mm_zy"], v
    sx, ss = xq.clone(), sa.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gy, gzy = OPS.scaled_mm_zy(sx, ss, w, sw, rows)
    for seed in (4, 5):
        nx, ns = _cuda_call(m, k, n, seed=seed)[:2]
        sx.copy_(nx)
        ss.copy_(ns)
        graph.replay()
        ey, ezy = OPS.scaled_mm_zy(nx, ns, w, sw, rows)
        assert torch.equal(gy.view(torch.int16), ey.view(torch.int16)) and torch.equal(gzy, ezy)
    y0, zy0 = OPS.scaled_mm_zy(xq[:0], sa[:0], w, sw, rows)
    assert tuple(y0.shape) == (0, n) and not zy0.any()
