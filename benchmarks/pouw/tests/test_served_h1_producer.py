"""Kernels' producer of vllm's `Served.receipts` (`served_h1/producer.py`, BRIDGE2) against `h1tile.reference`, word for
word: a call's receipts are (units, 2 (KS + m)) u32, unit u of the call row block u // (N / 32) and column block
u % (N / 32) of its M slot rows (the served codes, then zero rows) and its weight, at position u0 + u, seeded from the
run's salt and the call's registered roots, root(a/<K>@<s>) and root(b/<K>): the unit's public row as the verifier
builds it (`credit.public_table`), which the references are seeded from. A row block past the served rows has v1's
noise-derived receipt, not zero, unless the producer skips dead tiles (``skip_dead``, ``H1_SKIP_DEAD=1``): then it has v4's
zero receipt and a row block holding a served row its full one. Through `Served` (slow: the window Program's real tile
unit takes half a minute to declare) every call's ``receipt@<c>`` is those receipts' words; and on a GPU with h1u.cu's
-hbc build named by `VERITY_POUW_H1U_SO` the units kernel gives the same words at Llama-3.1-8B's four linear shapes, a
call at a time or a step's calls of one K in one launch (`Receipts.prepare`).

Under J10-R (a window Program over `h1tile.PouwH1Tile5`) each unit's receipt is `h1tile.j10r_reference`'s, its noise
gathered from its call's one stream (seed_c at the call's first position), and its public row carries the call and the
unit's blocks: on the CPU through the torch chain (and through `Served`, slow), and on a GPU through noise.cu's
`h1_noise_calls` (`VERITY_POUW_NOISE_CUBIN`) and h1u.cu's J10-R units."""
from __future__ import annotations

import hashlib
import importlib.util
import os
import re
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


def _ref(a: np.ndarray, b: np.ndarray, u: int, position: int, roots=(RA, RB), salt=SALT, served=None) -> list[int]:
    """`h1tile.reference`'s receipt of unit u of A [M, K] and B [N, K] s8 at ``position``; with ``served``, v4's under the
    admit bits of A's first ``served`` rows."""
    i, j = divmod(u, len(b) // 32)
    gated = None if served is None else [int(32 * i + r < served) for r in range(32)]
    return HT.reference((a[32 * i:32 * i + 32].view(np.uint8)).tolist(), (b[32 * j:32 * j + 32].view(np.uint8)).tolist(),
                        salt, position, roots, pairs=[(0, 0)], gated=gated)[1]


def _ref_j10r(a: np.ndarray, b: np.ndarray, u: int, position: int, call: int, roots=(RA, RB), salt=SALT) -> list[int]:
    """`h1tile.j10r_reference`'s receipt of unit u of A [M, K] (M the slot) and B [N, K] s8 at ``position``, its call's
    first position ``call`` and its tile's `h1tile.j10r_blocks`."""
    i, j = divmod(u, len(b) // 32)
    blocks = HT.j10r_blocks(a.shape[0], a.shape[1], (32, 32), i, j)
    return HT.j10r_reference((a[32 * i:32 * i + 32].view(np.uint8)).tolist(), (b[32 * j:32 * j + 32].view(np.uint8)).tolist(),
                             salt, position, roots, call, blocks, pairs=[(0, 0)])[1]


class _At(PR.Receipts):
    """The producer at a given first unit, without a window Program (v1's unit unless ``j10r``): unit u0 + t's public row
    is `core_check`'s row of the run's salt, u0 + t and the call's roots, and under J10-R u0 and the tile's blocks."""

    def __init__(self, u0: int, *args, **kw):
        kw.setdefault("j10r", False)
        super().__init__(*args, **kw)
        self.u0 = u0

    def start(self, call):
        return self.u0, (self.served.m // 32) * (call.n // 32)

    def public(self, call):
        from verity.core.protocols.pouw import core_check as CC
        (u0, n), ncb = self.start(call), call.n // 32
        return [CC.public_words(self.run.salt, u0 + t, self.roots(call),
                                (u0, CC.j10r_blocks(self.served.m, call.k, (32, 32), t // ncb, t % ncb)) if self.j10r
                                else None) for t in range(n)]


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
        assert got[u].tolist() == _ref(a, w, u, u0 + u), f"unit {u}"
    assert got[2:].any(), "a dead row block's v1 receipt is its noise's"
    assert _ref(a, w, 2, u0 + 2) != _ref(a, w, 2, u0 + 3), "a unit's receipt is at its position"


@pytest.mark.parametrize("m", [64, 20, 0])
def test_skipping_dead_tiles_gives_h1tile_reference_v4_on_a_step_of_64_20_or_0_served_rows(m):
    """With ``skip_dead`` at a 64-row slot, every unit's receipt is `h1tile.reference`'s v4 under the served rows' admit
    bits, one row a unit in tile order: a row block holding a served row (the straddling one at 20) computed in full, the
    ungated producer's words, and a row block past them zero words. `check` holds every kept unit to that reference, so a
    dead tile's v1 receipt in its place is refused."""
    rng = np.random.default_rng(20261024 + m)
    n, k, u0 = 64, 64, 10
    codes, w = rng.integers(-127, 128, (m, k), dtype=np.int8), rng.integers(-127, 128, (n, k), dtype=np.int8)
    call, args = _call(n=n, k=k, m=m), (codes, np.ones(m, np.float32), w, np.ones(n, np.float32))
    v1 = _At(u0, SimpleNamespace(m=64), SimpleNamespace(salt=SALT), roots=lambda c: (RA, RB), skip_dead=False)(call, *args)
    made = _At(u0, SimpleNamespace(m=64), SimpleNamespace(salt=SALT), roots=lambda c: (RA, RB), keep=lambda c: True,
               skip_dead=True)
    got = made(call, *args)
    assert got.dtype == np.uint32 and got.shape == v1.shape == (4, 2 * (k // 32 + 2))
    a = np.zeros((64, k), np.int8)
    a[:m] = codes
    live = [32 * (u // 2) < m for u in range(4)]
    for u in range(4):
        assert got[u].tolist() == _ref(a, w, u, u0 + u, served=m), f"unit {u}"
        assert got[u].tolist() == (v1[u].tolist() if live[u] else [0] * got.shape[1]), f"unit {u}"
    assert all(v1[u].any() for u in range(4)), "v1's dead tile has its noise's receipt"
    seen = PR.check(made.kept[call.index], list(range(4)))
    assert all(v["ok"] and v["pub"] for v in seen.values()) and [v["live"] for v in seen.values()] == live
    unskipped = PR.check(dict(made.kept[call.index], receipts=v1), list(range(4)))
    assert [v["ok"] for v in unskipped.values()] == live


def test_skipping_dead_tiles_is_off_unless_h1_skip_dead_is_1(monkeypatch):
    monkeypatch.delenv("H1_SKIP_DEAD", raising=False)
    assert not PR.Receipts(None).skip_dead and PR.Receipts(None, skip_dead=True).skip_dead
    monkeypatch.setenv("H1_SKIP_DEAD", "1")
    assert PR.Receipts(None).skip_dead and not PR.Receipts(None, skip_dead=False).skip_dead


def test_a_public_row_gives_back_the_salt_position_and_roots_it_carries():
    """`public_fields` inverts `core_check.public_words`, the verifier's public row; a row of another width is refused."""
    from verity.core.protocols.pouw import core_check as CC
    assert PR.public_fields(CC.public_words(SALT, 2**40 + 7, (RA, RB))) == (SALT, 2**40 + 7, (RA, RB))
    with pytest.raises(ValueError, match="public row is"):
        PR.public_fields(CC.public_words(SALT, 7, (RA, RB))[:-1])


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


def _window(producer, shapes, modules, blocks, rows: list[int], steps: int, seed: int, records, unit=None):
    """An H1 window over ``shapes`` whose step s serves rows[s] rows (later steps empty) with each call's receipts from
    ``producer(served)``: the run, each call's served codes by index, each linear's weight codes, each call's receipts as
    the producer returned them, by index (the native commit keeps no rows), and the producer. ``unit``: the window
    Program's tile unit (`Served.unit`), v1's by default."""
    from verity_vllm.protocols.pouw import h1 as PH
    from verity_vllm.protocols.pouw.h1 import served as SV
    served = SV.Served(blocks=blocks, window=hashlib.sha512(b"window").digest(), unit=unit)
    made, made_by = producer(served), {}

    def recorded(call, *args):
        out = made_by[call.index] = made(call, *args)
        return out

    served.receipts = recorded
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
    return run, codes, [wt.numpy() for wt, _sw in ws], made_by, made


def _registered_receipts(run, made_by, c) -> np.ndarray:
    """Call c's receipt table's words: `receipt_words` of the receipts the producer returned for it, which the registered
    ``receipt@<c>`` is the native commit of (its root, under its nonce and the window's open root)."""
    from verity.core.protocols.pouw.circuit import h1window as HW
    from verity_vllm.protocols.pouw.h1 import registered as NR
    from verity_vllm.protocols.pouw.h1 import served as SV
    served, rc = run.served, run.served.receipt_table(c.index)
    words, n = SV.receipt_words(made_by[c.index]), HW.call_units(served.program)[c.index][1]
    assert words.shape == served.receipt_shape(c.index) == (n, 2 * HT.TileShape(32, 32, c.k).sets * 2)
    tree = NR.commit(served.program_digest, rc, words, run._key, SV.RECEIPT_NONCE + c.index, window=served.root)
    assert tree.root == served.registered[rc].root, f"call {c.index}'s {rc} is not its receipts' words"
    return words


def _held_to_reference(run, codes, weights, made_by, units_of):
    """Every call's ``receipt@<c>`` is `receipt_words` of h1tile's receipts (of ``units_of(call)``) on its padded codes and
    weight, each seeded from its unit's public row as the verifier builds it (`credit.public_table`: the run's salt, its
    Program position and the registered roots), whose table is the ``pub@<c>`` `Served` registered; returns the units
    checked, live and dead."""
    from verity.core.protocols.pouw import credit as CR
    from verity.core.protocols.pouw.circuit import h1window as HW
    from verity_vllm.protocols.pouw.h1 import served as SV
    served, pre, seen = run.served, run.served.program.fn.bindings["PREFIX"], {"live": 0, "dead": 0}
    assert not served.refused, served.refused
    j10r = HW.is_j10r(served.program)
    for c in run.calls:
        u0, n = HW.call_units(served.program)[c.index]
        roots = (served.registered[served.a_table(c.index)].root, served.registered[pre + f"b/{c.k}"].root)
        pv = served.pub_table(c.index)
        blocks = HW.call_blocks(served.program, c.index) if j10r else None
        pub, _salts, verifiers = CR.public_table(served.program_digest, pv, run.salt.hex(), served.root, u0, n,
                                                 (roots[0].hex(), roots[1].hex()), blocks)
        assert served.registered[pv].root == verifiers.root, f"call {c.index}'s {pv} is not the verifier's"
        rows = _registered_receipts(run, made_by, c)
        a = np.zeros((served.m, c.k), np.int8)
        a[:len(codes.get(c.index, ()))] = codes.get(c.index, np.zeros((0, c.k), np.int8))
        for u in units_of(c, n):
            salt, position, rs = PR.public_fields(pub[u])
            assert (salt, position, rs) == (run.salt, u0 + u, roots), f"call {c.index} unit {u}'s public row"
            assert PR.j10r_fields(pub[u]) == (None if blocks is None else (u0, blocks[u])), f"call {c.index} unit {u}"
            ref = (_ref(a, weights[c.layer], u, position, rs, salt) if blocks is None
                   else _ref_j10r(a, weights[c.layer], u, position, u0, rs, salt))
            want = SV.receipt_words(np.array([ref], np.uint32))[0]
            assert np.array_equal(rows[u], want), f"call {c.index} unit {u}"
            seen["live" if 32 * (u // (c.n // 32)) < c.m else "dead"] += 1
    return seen


@pytest.mark.slow
def test_each_calls_receipt_table_is_h1tile_reference_through_served():
    """Through `Served` with the window Program's real tile unit (`h1window.tile_unit`) and the torch chain: every unit of
    every call (a step of 20 served rows, its second row block dead, then an empty step), word for word; and `check` of
    every kept call, its reference seeded from the verifier's public rows, finds every unit's row the producer's."""
    from verity_vllm.engine import int8_ops as OPS
    shapes, blocks = [(64, 64), (32, 64), (32, 128)], (0, 0, 1)
    run, codes, ws, made_by, made = _window(lambda s: PR.Receipts(s, keep=lambda c: True), shapes,
                                            {0: "q", 1: "k", 2: "o"}, blocks, [20], 2, 3,
                                            lambda x, s_a, wt, sw: OPS.scaled_mm_zy(x, s_a, wt, sw, 64)[1])
    seen = _held_to_reference(run, codes, ws, made_by, lambda c, n: range(n))
    assert seen == {"live": 4, "dead": 12}
    got = [v for c in sorted(made.kept) for v in PR.check(made.kept[c], list(range(len(made_by[c])))).values()]
    assert sorted(made.kept) == [c.index for c in run.calls] and len(got) == 16
    assert all(v["ok"] and v["pub"] for v in got) and sum(v["live"] for v in got) == 4


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
    from verity_vllm.protocols.pouw import h1 as PH
    from verity_vllm.protocols.pouw.h1 import served as SV
    RC = PR.RC

    def zeros(x, s_a, wt, sw):
        return torch.zeros(PH.units(x.shape[1], wt.shape[0], 64), E.ZY_BYTES, dtype=torch.uint8)

    run, codes, ws, made_by, _made = _window(lambda s: PR.Receipts(s, units=units), LLAMA,
                                             {0: "qkv", 1: "o", 2: "gate_up", 3: "down"}, (0, 1, 2, 3), [64, 20], 3, 5,
                                             zeros)
    served, pre = run.served, run.served.program.fn.bindings["PREFIX"]
    assert not served.refused, served.refused
    for c in run.calls:
        u0, _n = HW.call_units(served.program)[c.index]
        roots = (served.registered[served.a_table(c.index)].root, served.registered[pre + f"b/{c.k}"].root)
        a = torch.zeros(served.m, c.k, dtype=torch.int8, device="cuda")
        a[:c.m] = torch.from_numpy(codes.get(c.index, np.zeros((0, c.k), np.int8))).cuda()
        want = SV.receipt_words(RC.call_receipts(RC.Call(a, torch.from_numpy(ws[c.layer]).cuda()), run.salt, u0, *roots))
        assert np.array_equal(_registered_receipts(run, made_by, c), want), c.index

    def ends(c, n):
        cb = c.n // 32
        return sorted({u for i in range(n // cb) for u in (i * cb, (i + 1) * cb - 1)})

    assert _held_to_reference(run, codes, ws, made_by, ends) == {"live": 24, "dead": 24}


def test_skipping_dead_tiles_the_units_kernel_is_h1tile_reference_v4_at_llama_8b_shapes():
    """On the GPU at Llama-3.1-8B's four linear shapes, a 64-row slot, over steps of 64, 20 and 0 served rows: with
    ``skip_dead`` (`kh_unit`'s live mask) every unit of a row block holding a served row is the ungated kernel's words and
    every unit of a dead one zero words, and `check` holds the first and last unit of each row block to
    `h1tile.reference`'s v4 under the served rows' admit bits, word for word."""
    units = _kernel()
    rng = np.random.default_rng(20261024)
    seen = {"live": 0, "dead": 0}
    for n, k in LLAMA:
        b = torch.from_numpy(rng.integers(-127, 128, (n, k), dtype=np.int8)).cuda()
        ncb = n // 32
        for m in (64, 20, 0):
            codes, call = rng.integers(-127, 128, (m, k), dtype=np.int8), _call(n=n, k=k, m=m)

            def made(skip, keep=None, b=b):
                return _At(1000, SimpleNamespace(m=64), SimpleNamespace(salt=SALT), units=units, weight=lambda c: b,
                           roots=lambda c: (RA, RB), keep=keep, skip_dead=skip)

            v1, on = made(False)(call, codes, None, None, None), made(True, keep=lambda c: True)
            got = on(call, codes, None, None, None)
            live = np.repeat(np.arange(2) * 32 < m, ncb)
            assert np.array_equal(got[live], v1[live]) and not got[~live].any(), (n, k, m)
            checked = PR.check(on.kept[call.index], [0, ncb - 1, ncb, 2 * ncb - 1])
            assert all(v["ok"] and v["pub"] for v in checked.values()), (n, k, m, checked)
            for v in checked.values():
                seen["live" if v["live"] else "dead"] += 1
    assert seen == {"live": 24, "dead": 24}


class _AtEach(_At):
    """The producer with each call's first unit given, without a window Program."""

    def __init__(self, starts, *args, **kw):
        super().__init__(0, *args, **kw)
        self.starts = starts

    def start(self, call):
        return self.starts[call.index]


def test_a_steps_calls_of_one_k_in_one_units_kernel_launch_are_each_calls_own_receipts_at_llama_8b_shapes():
    """On the GPU, `prepare` of a step's calls of one K (qkv, o, gate_up and qkv again at K = 4096, down twice at
    K = 14336, each on its own codes, roots and first unit) and then each call's ask give, over steps of 64, 20 and 0
    served rows with dead tiles computed and skipped, in one launch, a launch a call or two launches, every call's
    receipts as its ask alone gives them, word for word, each ready before its ask and gone after it; and at 20 rows
    `check` holds the first and last unit of each row block of every call to `h1tile.reference` (v1's, and v4's with the
    skip)."""
    units = _kernel()
    rng = np.random.default_rng(20261025)
    ws = [torch.from_numpy(rng.integers(-127, 128, (n, k), dtype=np.int8)).cuda() for n, k in LLAMA]
    checked = {"live": 0, "dead": 0}
    for k, linears in ((4096, [0, 1, 2, 0]), (14336, [3, 3])):
        calls, starts, u0 = [], {}, 1000
        for i, w in enumerate(linears):
            n = LLAMA[w][0]
            calls.append(SimpleNamespace(index=10 + i, step=0, layer=w, module=str(w), m=0, k=k, n=n))
            starts[10 + i] = (u0, 2 * (n // 32))
            u0 += 2 * (n // 32) + 7
        for m in (64, 20, 0):
            batch = [(c, rng.integers(-127, 128, (m, k), dtype=np.int8), None, None, None) for c in calls]
            for skip in (False, True):
                def made(keep=None, launches=1, skip=skip, starts=starts):
                    return _AtEach(starts, SimpleNamespace(m=64), SimpleNamespace(salt=SALT), units=units,
                                   weight=lambda x: ws[x.layer], keep=keep, skip_dead=skip, launches=launches,
                                   roots=lambda x: (hashlib.sha512(bytes([x.index])).digest(), RB))

                want = [made()(*args) for args in batch]
                for launch in (1, len(calls), 2):
                    together = made(keep=lambda c: True, launches=launch)
                    together.prepare(batch)
                    assert sorted(together._ready) == [c.index for c in calls], (k, m, skip, launch)
                    for args, w in zip(batch, want):
                        assert np.array_equal(together(*args), w), (k, m, skip, launch, args[0].index)
                        assert args[0].index not in together._ready
                if m == 20:
                    for c in calls:
                        cb = c.n // 32
                        got = PR.check(together.kept[c.index], [0, cb - 1, cb, 2 * cb - 1])
                        assert all(v["ok"] and v["pub"] for v in got.values()), (k, skip, c.index, got)
                        for v in got.values():
                            checked["live" if v["live"] else "dead"] += 1
    assert checked == {"live": 24, "dead": 24}


#: noise.cu built for the device (``nvcc -cubin``, `receipts.Cubin`), for the GPU tests of its J10-R streams
CUBIN = "VERITY_POUW_NOISE_CUBIN"


@pytest.mark.parametrize("m,n,k", [(32, 32, 64), (64, 96, 128), (128, 64, 4096)])
def test_a_j10r_units_counters_are_its_rows_and_columns_blocks_of_its_calls_one_stream(m, n, k):
    """`receipts.j10r_counters` of every unit (I, J) of an M × N call at K is `h1tile.j10r_unit_blocks` at its
    `h1tile.j10r_blocks`, the counters its gates read, all inside the call's stream (`receipts.stream_blocks`): units of
    one row block share their rows' counters and units of one column block their columns'."""
    RC, lay = PR.RC, HT.TileShape(32, 32, k)
    us = RC.units(m, n)
    got = RC.j10r_counters(torch.tensor([i for i, _j in us]), torch.tensor([j for _i, j in us]), m, k)
    want = [HT.j10r_unit_blocks(lay, HT.j10r_blocks(m, k, (32, 32), i, j)) for i, j in us]
    assert got.tolist() == want and int(got.max()) < RC.stream_blocks(m, n, k)
    rows = 32 * lay.rb
    assert all(got[a, :rows].tolist() == got[b, :rows].tolist() for a, (i, _j) in enumerate(us)
               for b, (i2, _j2) in enumerate(us) if i == i2)


def test_a_j10r_public_row_gives_back_its_call_and_blocks():
    """`j10r_fields` inverts the J10-R part of `core_check.public_words` (the call's first position and the unit's
    blocks), `public_fields` still reads its salt, position and roots, and a v1 row has no J10-R fields."""
    from verity.core.protocols.pouw import core_check as CC
    row = CC.public_words(SALT, 2**40 + 7, (RA, RB), (2**40, (9 * 64, 64 * 9 + 9 * 32)))
    assert PR.public_fields(row) == (SALT, 2**40 + 7, (RA, RB))
    assert PR.j10r_fields(row) == (2**40, (9 * 64, 64 * 9 + 9 * 32))
    assert PR.j10r_fields(CC.public_words(SALT, 7, (RA, RB))) is None


@pytest.mark.parametrize("n,k,m", [(64, 64, 20), (96, 128, 64), (64, 64, 0)])
def test_j10r_receipts_are_h1tile_j10r_reference_and_check_refuses_v1s_in_their_place(n, k, m):
    """Under J10-R (`h1tile.PouwH1Tile_v5`, ungated) each unit's receipt at a 64-row slot is `h1tile.j10r_reference`'s,
    seeded by the call's seed_c at its first position u0 and σ_R at u0 + u, a dead row block's its noise's, not zero, and
    not v1's; `check` holds each kept unit to that reference from its J10-R public row, so v1's receipts, or a v1 public
    row, in their place are refused."""
    rng = np.random.default_rng(20261025 + n + m)
    u0 = 10
    codes, w = rng.integers(-127, 128, (m, k), dtype=np.int8), rng.integers(-127, 128, (n, k), dtype=np.int8)
    call, args = _call(n=n, k=k, m=m), (codes, np.ones(m, np.float32), w, np.ones(n, np.float32))
    made = _At(u0, SimpleNamespace(m=64), SimpleNamespace(salt=SALT), roots=lambda c: (RA, RB), keep=lambda c: True,
               j10r=True)
    got = made(call, *args)
    v1 = _At(u0, SimpleNamespace(m=64), SimpleNamespace(salt=SALT), roots=lambda c: (RA, RB))(call, *args)
    units = 2 * (n // 32)
    assert got.dtype == np.uint32 and got.shape == v1.shape == (units, 2 * HT.TileShape(32, 32, k).sets)
    a = np.zeros((64, k), np.int8)
    a[:m] = codes
    for u in range(units):
        assert got[u].tolist() == _ref_j10r(a, w, u, u0 + u, u0), f"unit {u}"
        assert got[u].any() and got[u].tolist() != v1[u].tolist(), f"unit {u}"
    kept = made.kept[call.index]
    seen = PR.check(kept, list(range(units)))
    assert all(v["ok"] and v["pub"] for v in seen.values())
    assert [v["live"] for v in seen.values()] == [32 * (u // (n // 32)) < m for u in range(units)]
    assert not any(v["ok"] for v in PR.check(dict(kept, receipts=v1), list(range(units))).values())
    v1_pub = _At(u0, SimpleNamespace(m=64), SimpleNamespace(salt=SALT), roots=lambda c: (RA, RB)).public(call)
    assert not any(v["pub"] for v in PR.check(dict(kept, pub=v1_pub), list(range(units))).values())
    other = _At(u0 + 1, SimpleNamespace(m=64), SimpleNamespace(salt=SALT), roots=lambda c: (RA, RB), j10r=True)
    assert not any(v["pub"] for v in PR.check(dict(kept, pub=other.public(call)), list(range(units))).values())


def test_j10r_refuses_skipping_dead_tiles():
    """J10-R's unit has no dead tile (a row block past the served rows has its noise's receipt), so ``skip_dead`` is
    refused there rather than giving zero rows the verifier would refuse."""
    rng = np.random.default_rng(3)
    w = rng.integers(-127, 128, (64, 64), dtype=np.int8)
    made = _At(0, SimpleNamespace(m=64), SimpleNamespace(salt=SALT), roots=lambda c: (RA, RB), j10r=True, skip_dead=True)
    with pytest.raises(ValueError, match="skip_dead under J10-R"):
        made(_call(m=1), np.zeros((1, 64), np.int8), None, w, None)


@pytest.mark.slow
def test_each_calls_j10r_receipt_table_is_h1tile_j10r_reference_through_served():
    """Through `Served` with a window Program over J10-R's tile unit (`h1tile.PouwH1Tile5` under `h1zy`) and the torch
    chain, the producer taking J10-R from the Program (`h1window.is_j10r`): every call's ``pub@<c>`` is the verifier's
    J10-R public table (`credit.public_table` with `h1window.call_blocks`), and every unit of every call's
    ``receipt@<c>`` is `h1tile.j10r_reference` seeded from its row, word for word; `check` of every kept call agrees."""
    from verity.core.protocols.pouw.circuit import h1window as HW
    from verity_vllm.engine import int8_ops as OPS
    shapes, blocks = [(64, 64), (32, 64), (32, 128)], (0, 0, 1)
    run, codes, ws, made_by, made = _window(lambda s: PR.Receipts(s, keep=lambda c: True), shapes,
                                            {0: "q", 1: "k", 2: "o"}, blocks, [20], 2, 3,
                                            lambda x, s_a, wt, sw: OPS.scaled_mm_zy(x, s_a, wt, sw, 64)[1],
                                            unit=lambda k: HW.tile_unit(k, (32, 32), HT.PouwH1Tile5))
    assert HW.is_j10r(run.served.program) and made.j10r
    seen = _held_to_reference(run, codes, ws, made_by, lambda c, n: range(n))
    assert seen == {"live": 4, "dead": 12}
    got = [v for c in sorted(made.kept) for v in PR.check(made.kept[c], list(range(len(made_by[c])))).values()]
    assert sorted(made.kept) == [c.index for c in run.calls] and len(got) == 16
    assert all(v["ok"] and v["pub"] for v in got) and sum(v["live"] for v in got) == 4


def _cubin():
    path = os.environ.get(CUBIN)
    if not path or not torch.cuda.is_available():
        pytest.skip(f"no GPU, or {CUBIN} names no noise.cu cubin (nvcc -cubin -gencode arch=compute_120a,code=sm_120a "
                    "-I benchmarks/codec_tensor): its run is a recorded run on vy-nebius-2 (producer_gpu.sh)")
    return PR.RC.Cubin(path)


def test_noise_cus_j10r_call_streams_are_seed_cs_noise_blocks():
    """On the GPU, noise.cu's `h1_noise_calls` over calls of 1, 37, 0 and 600 blocks and Llama-3.1-8B's gate_up stream
    (64 rows, N = 28672, K = 4096: 258,624 blocks) in one launch gives each call's stream block for block as hashlib does
    (`seeds.noise_block` of its seed_c), and its J10-R receipts are the torch chain's over hashlib's stream."""
    from verity.core.protocols.pouw.circuit import seeds as SD
    cubin, RC = _cubin(), PR.RC
    seeds = [SD.call_seed(SALT, c, RA, RB) for c in (0, 1, 2, 3, 2**33)]
    counts = [1, 37, 0, 600, RC.stream_blocks(64, 28672, 4096)]
    got = RC.call_streams(seeds, counts, torch.device("cuda"), cubin)
    want = RC.call_streams(seeds, counts, torch.device("cpu"), None)
    assert [tuple(g.shape) for g in got] == [(c, 16) for c in counts]
    for c, (g, w) in enumerate(zip(got, want)):
        assert torch.equal(g.cpu(), w), f"call {c}"
    rng = np.random.default_rng(7)
    a = torch.from_numpy(rng.integers(-127, 128, (64, 4096), dtype=np.int8)).cuda()
    b = torch.from_numpy(rng.integers(-127, 128, (4096, 4096), dtype=np.int8)).cuda()
    a[20:] = 0
    call = RC.Call(a, b)
    assert np.array_equal(RC.call_receipts(call, SALT, 1000, RA, RB, cubin, j10r=True),
                          RC.call_receipts(call, SALT, 1000, RA, RB, None, j10r=True))


def _j10r_kernel():
    units = _kernel()
    if not units.j10r:
        pytest.skip("this h1u.cu build has no J10-R units (kh_units_multi_j10r)")
    return units


def test_the_producer_sizes_each_j10r_workspace_through_dither_j10r_ws():
    """`Units.multi` takes each J10-R call's workspace from dither's `j10r_ws.workspace_bytes` at the call's N, which
    asks the loaded build (an H1U_JW build's column blocks and W' past its rows) and refuses a size its flags
    contradict: no served_h1 module calls a build's kh_j10r_*_bytes itself, so an H1U_JW build loaded here isn't sized
    by its rows."""
    call = re.compile(r"\bkh_j10r_(?:rows|jw|ws|workspace)_bytes\s*\(")
    hits = [f"{p.name}:{i}" for p in sorted(SERVED_H1.glob("*.py"))
            for i, line in enumerate(p.read_text().splitlines(), 1) if call.search(line)]
    assert hits == []
    assert Path(PR.WS.__file__) == SERVED_H1.parents[1] / "hash" / "dither" / "j10r_ws.py"


def test_the_units_kernels_j10r_receipts_are_h1tile_j10r_reference_at_llama_8b_shapes():
    """On the GPU at Llama-3.1-8B's four linear shapes, a 64-row slot, over steps of 64, 20 and 0 served rows: h1u.cu's
    J10-R units (`kh_seeds_j10r`, `kh_units_multi_j10r`) give every unit the torch chain's J10-R receipt (noise.cu's
    stream when `VERITY_POUW_NOISE_CUBIN` is set, else hashlib's), word for word, and `check` holds the first and last unit
    of each row block to `h1tile.j10r_reference` from its J10-R public row."""
    units = _j10r_kernel()
    cubin = _cubin() if os.environ.get(CUBIN) else None
    rng = np.random.default_rng(20261026)
    seen = {"live": 0, "dead": 0}
    for n, k in LLAMA:
        b = torch.from_numpy(rng.integers(-127, 128, (n, k), dtype=np.int8)).cuda()
        ncb = n // 32
        for m in (64, 20, 0):
            codes, call = rng.integers(-127, 128, (m, k), dtype=np.int8), _call(n=n, k=k, m=m)

            def made(keep=None, b=b, **kw):
                return _At(1000, SimpleNamespace(m=64), SimpleNamespace(salt=SALT), weight=lambda c: b,
                           roots=lambda c: (RA, RB), keep=keep, j10r=True, **kw)

            on = made(keep=lambda c: True, units=units)
            got = on(call, codes, None, None, None)
            torch_chain = made(device="cuda", cubin=cubin)(call, codes, None, None, None)
            assert np.array_equal(got, torch_chain), (n, k, m)
            checked = PR.check(on.kept[call.index], [0, ncb - 1, ncb, 2 * ncb - 1])
            assert all(v["ok"] and v["pub"] for v in checked.values()), (n, k, m, checked)
            for v in checked.values():
                seen["live" if v["live"] else "dead"] += 1
    assert seen == {"live": 24, "dead": 24}


def test_a_steps_calls_of_one_k_in_one_j10r_units_kernel_launch_are_each_calls_own_receipts():
    """On the GPU under J10-R, `prepare` of a step's calls of one K (qkv, o, gate_up and qkv again at K = 4096, down twice
    at K = 14336, each on its own codes, roots and first unit, so each its own seed_c and row workspace) and then each
    call's ask give, at 20 served rows, in one launch, a launch a call or two launches, every call's receipts as its ask
    alone gives them, word for word."""
    units = _j10r_kernel()
    rng = np.random.default_rng(20261027)
    ws = [torch.from_numpy(rng.integers(-127, 128, (n, k), dtype=np.int8)).cuda() for n, k in LLAMA]
    for k, linears in ((4096, [0, 1, 2, 0]), (14336, [3, 3])):
        calls, starts, u0 = [], {}, 1000
        for i, w in enumerate(linears):
            n = LLAMA[w][0]
            calls.append(SimpleNamespace(index=10 + i, step=0, layer=w, module=str(w), m=0, k=k, n=n))
            starts[10 + i] = (u0, 2 * (n // 32))
            u0 += 2 * (n // 32) + 7
        batch = [(c, rng.integers(-127, 128, (20, k), dtype=np.int8), None, None, None) for c in calls]

        def made(launches=1, starts=starts):
            return _AtEach(starts, SimpleNamespace(m=64), SimpleNamespace(salt=SALT), units=units,
                           weight=lambda x: ws[x.layer], launches=launches, j10r=True,
                           roots=lambda x: (hashlib.sha512(bytes([x.index])).digest(), RB))

        want = [made()(*args) for args in batch]
        for launch in (1, len(calls), 2):
            together = made(launches=launch)
            together.prepare(batch)
            assert sorted(together._ready) == [c.index for c in calls], (k, launch)
            for args, w in zip(batch, want):
                assert np.array_equal(together(*args), w), (k, launch, args[0].index)
