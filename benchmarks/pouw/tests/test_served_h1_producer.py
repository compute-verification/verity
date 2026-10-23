"""Kernels' producer of vllm's `Served.receipts` (`served_h1/producer.py`, BRIDGE2) against `h1tile.reference`, word for
word: a call's receipts are (units, 2 (KS + m)) u32, unit u of the call row block u // (N / 32) and column block
u % (N / 32) of its M slot rows (the served codes, then zero rows) and its weight, at position u0 + u, seeded from the
run's salt and the call's registered roots, root(a/<K>@<s>) and root(b/<K>). A row block past the served rows has v1's
noise-derived receipt, not zero. Through `Served` (slow: the window Program's real tile unit takes half a minute to
declare) every call's ``receipt@<c>`` is those receipts' words; and on a GPU with h1u.cu's -hbc build named by
`VERITY_POUW_H1U_SO` the units kernel gives the same words at Llama-3.1-8B's four linear shapes."""
from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from verity.core.protocols.pouw.circuit import h1tile as HT  # noqa: E402

SERVED_H1 = Path(__file__).resolve().parents[1] / "served_h1"
SALT, RA, RB = hashlib.sha512(b"salt").digest(), hashlib.sha512(b"A").digest(), hashlib.sha512(b"B").digest()
#: Llama-3.1-8B's linears (n, K): qkv, o, gate_up, down
LLAMA = [(6144, 4096), (4096, 4096), (28672, 4096), (4096, 14336)]
SO = "VERITY_POUW_H1U_SO"


def _producer():
    spec = importlib.util.spec_from_file_location("served_h1_producer", SERVED_H1 / "producer.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


PR = _producer()


def _ref(a: np.ndarray, b: np.ndarray, base: int, u: int, roots=(RA, RB)) -> list[int]:
    """`h1tile.reference`'s receipt of unit u of A [M, K] and B [N, K] s8 at position base + u."""
    i, j = divmod(u, len(b) // 32)
    return HT.reference((a[32 * i:32 * i + 32].view(np.uint8)).tolist(), (b[32 * j:32 * j + 32].view(np.uint8)).tolist(),
                        SALT, base + u, roots, pairs=[(0, 0)])[1]


class _At(PR.Receipts):
    """The producer at a given first unit, without a window Program."""

    def __init__(self, u0: int, *args, **kw):
        super().__init__(*args, **kw)
        self.u0 = u0

    def start(self, call):
        return self.u0, (self.served.m // 32) * (call.n // 32)


def _call(index=3, n=64, k=64, m=20, layer=1):
    return SimpleNamespace(index=index, step=0, layer=layer, module="o", m=m, k=k, n=n)


def test_a_calls_receipts_are_h1tile_reference_in_tile_order_at_its_positions():
    """(units, 2 (KS + m)) u32, unit u row block u // (N / 32) and column block u % (N / 32) of the padded A and B at
    position u0 + u; the dead row block's receipts (zero codes) are v1's, its noise's, not zero."""
    rng = np.random.default_rng(20261023)
    n, k, m, u0 = 64, 64, 20, 10
    codes, w = rng.integers(-127, 128, (m, k), dtype=np.int8), rng.integers(-127, 128, (n, k), dtype=np.int8)
    made = _At(u0, SimpleNamespace(m=64), SimpleNamespace(salt=SALT), roots=lambda c: (RA, RB))
    got = made(_call(n=n, k=k, m=m), codes, np.ones(m, np.float32), w, np.ones(n, np.float32))
    assert got.dtype == np.uint32 and got.shape == (4, 2 * (k // 32 + 2))
    a = np.zeros((64, k), np.int8)
    a[:m] = codes
    for u in range(4):
        assert got[u].tolist() == _ref(a, w, u0, u), f"unit {u}"
    assert got[2:].any(), "a dead row block's v1 receipt is its noise's"
    assert _ref(a, w, u0, 2) != _ref(a, w, u0 + 1, 2), "a unit's receipt is at its position"


def test_a_calls_seeds_take_its_registered_roots_and_refuse_before_they_are():
    """By default the roots are the registered ``a/<K>@<s>`` and ``b/<K>``'s (the Registered row/v2 trees `Served`
    committed); a call asked for before either is registered is refused by name."""
    reg = {}
    served = SimpleNamespace(m=64, program=SimpleNamespace(fn=SimpleNamespace(bindings={"PREFIX": "h1/"})),
                             a_table=lambda c: "h1/a/64@0", registered=reg)
    made = _At(0, served, SimpleNamespace(salt=SALT))
    with pytest.raises(ValueError, match=r"\['h1/a/64@0', 'h1/b/64'\] isn't registered yet"):
        made.roots(_call())
    reg["h1/a/64@0"], reg["h1/b/64"] = SimpleNamespace(root=RA), SimpleNamespace(root=RB)
    assert made.roots(_call()) == (RA, RB)


def test_a_call_past_the_slot_or_unbound_is_refused():
    rng = np.random.default_rng(1)
    w = rng.integers(-127, 128, (64, 64), dtype=np.int8)
    with pytest.raises(ValueError, match="bound to no run"):
        _At(0, SimpleNamespace(m=64), roots=lambda c: (RA, RB))(_call(), np.zeros((1, 64), np.int8), None, w, None)
    made = _At(0, SimpleNamespace(m=64), SimpleNamespace(salt=SALT), roots=lambda c: (RA, RB))
    with pytest.raises(ValueError, match="not at most 64 rows of 64 int8 codes"):
        made(_call(), np.zeros((65, 64), np.int8), None, w, None)
    with pytest.raises(ValueError, match="no weight of linear 1"):
        made(_call(), np.zeros((1, 64), np.int8), None, None, None)


def _window(producer, shapes, modules, blocks, rows: list[int], steps: int, seed: int, records):
    """An H1 window over ``shapes`` whose step s serves rows[s] rows (later steps empty) with each call's receipts from
    ``producer(served)``: the run, each call's served codes by index, each linear's weight codes."""
    from verity_vllm.protocol_options import pouw_h1 as PH
    from verity_vllm.protocol_options import pouw_h1_served as SV
    served = SV.Served(blocks=blocks, window=hashlib.sha512(b"window").digest())
    made = served.receipts = producer(served)
    run = PH.H1Run("t", SALT, modules, shapes, 64, steps, key=bytes([seed + 1]) * 32, served=served)
    made.bind(run)
    rng, g = np.random.default_rng(seed), torch.Generator().manual_seed(seed)
    ws = [(torch.from_numpy(rng.integers(-127, 128, (n, k), dtype=np.int8)), torch.rand(n, 1, generator=g) + 0.5)
          for n, k in shapes]
    codes = {}
    for m in rows:
        xs = {}
        for w, ((n, k), (wt, sw)) in enumerate(zip(shapes, ws)):
            x = xs.setdefault(blocks[w], torch.from_numpy(rng.integers(-127, 128, (m, k), dtype=np.int8)))
            s_a = torch.rand(m, generator=g) + 0.5
            call = run.record(w, x, s_a, wt, sw, z=records(x, s_a, wt, sw))
            codes[call.index] = x.numpy()
    run.close()
    return run, codes, [wt.numpy() for wt, _sw in ws]


def _held_to_reference(run, codes, weights, units_of):
    """Every call's ``receipt@<c>`` is `receipt_words` of h1tile's receipts (of ``units_of(call)``) on its padded codes and
    weight at its Program positions under the registered roots; returns the units checked, live and dead."""
    from verity.core.protocols.pouw.circuit import h1window as HW
    from verity_vllm.protocol_options import pouw_h1_served as SV
    served, pre, seen = run.served, run.served.program.fn.bindings["PREFIX"], {"live": 0, "dead": 0}
    assert not served.refused, served.refused
    for c in run.calls:
        u0, n = HW.call_units(served.program)[c.index]
        roots = (served.registered[served.a_table(c.index)].root, served.registered[pre + f"b/{c.k}"].root)
        rows = np.asarray(served.registered[pre + f"receipt@{c.index}"].rows, np.uint16)
        assert rows.shape == served.receipt_shape(c.index) == (n, 2 * HT.Layout(32, 32, c.k).sets * 2)
        a = np.zeros((served.m, c.k), np.int8)
        a[:len(codes.get(c.index, ()))] = codes.get(c.index, np.zeros((0, c.k), np.int8))
        for u in units_of(c, n):
            want = SV.receipt_words(np.array([_ref(a, weights[c.layer], u0, u, roots)], np.uint32))[0]
            assert np.array_equal(rows[u], want), f"call {c.index} unit {u}"
            seen["live" if 32 * (u // (c.n // 32)) < c.m else "dead"] += 1
    return seen


@pytest.mark.slow
def test_each_calls_receipt_table_is_h1tile_reference_through_served():
    """Through `Served` with the window Program's real tile unit (`h1window.tile_unit`) and the torch chain: every unit of
    every call (a step of 20 served rows, its second row block dead, then an empty step), word for word."""
    from verity_vllm.engine import int8_ops as OPS
    shapes, blocks = [(64, 64), (32, 64), (32, 128)], (0, 0, 1)
    run, codes, ws = _window(lambda s: PR.Receipts(s), shapes, {0: "q", 1: "k", 2: "o"}, blocks, [20], 2, 3,
                             lambda x, s_a, wt, sw: OPS.scaled_mm_zy(x, s_a, wt, sw, 64)[1])
    seen = _held_to_reference(run, codes, ws, lambda c, n: range(n))
    assert seen == {"live": 4, "dead": 12}


def _kernel():
    so = os.environ.get(SO)
    if not so or not torch.cuda.is_available():
        pytest.skip(f"no GPU, or {SO} names no h1u.cu -hbc build (benchmarks/hash/dither/sbench.sh's h1u_kc.so): its run "
                    "is a recorded run on vy-nebius-2, about 5 minutes")
    return PR.Units(so)


def test_the_units_kernel_is_h1tile_reference_at_llama_8b_shapes_through_served():
    """On the GPU through `Served` at one Llama-3.1-8B layer (qkv, o and gate_up at K = 4096, down at K = 14336, each on
    its own input) over three steps of 64, 20 and 0 served rows (both row blocks live, the second dead, both dead): every
    unit of every call equal to receipts.py's torch chain, and the first and last unit of each row block equal to
    `h1tile.reference`, word for word. The z/y records are zeros: no receipt reads them."""
    units = _kernel()
    from verity.core.protocols.pouw.circuit import h1window as HW
    from verity_vllm.engine import int8_exact as E
    from verity_vllm.protocol_options import pouw_h1 as PH
    from verity_vllm.protocol_options import pouw_h1_served as SV
    RC = PR.RC

    def zeros(x, s_a, wt, sw):
        return torch.zeros(PH.units(x.shape[1], wt.shape[0], 64), E.ZY_BYTES, dtype=torch.uint8)

    run, codes, ws = _window(lambda s: PR.Receipts(s, units=units), LLAMA, {0: "qkv", 1: "o", 2: "gate_up", 3: "down"},
                             (0, 1, 2, 3), [64, 20], 3, 5, zeros)
    served, pre = run.served, run.served.program.fn.bindings["PREFIX"]
    assert not served.refused, served.refused
    for c in run.calls:
        u0, _n = HW.call_units(served.program)[c.index]
        roots = (served.registered[served.a_table(c.index)].root, served.registered[pre + f"b/{c.k}"].root)
        a = torch.zeros(served.m, c.k, dtype=torch.int8, device="cuda")
        a[:c.m] = torch.from_numpy(codes.get(c.index, np.zeros((0, c.k), np.int8))).cuda()
        want = SV.receipt_words(RC.call_receipts(RC.Call(a, torch.from_numpy(ws[c.layer]).cuda()), run.salt, u0, *roots))
        assert np.array_equal(np.asarray(served.registered[pre + f"receipt@{c.index}"].rows, np.uint16), want), c.index

    def ends(c, n):
        cb = c.n // 32
        return sorted({u for i in range(n // cb) for u in (i * cb, (i + 1) * cb - 1)})

    assert _held_to_reference(run, codes, ws, ends) == {"live": 24, "dead": 24}
