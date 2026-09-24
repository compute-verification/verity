"""``fp4-nvf4+poseidon2`` (fp4/hashed.py): the in-circuit decode layer, torch-free -- the composed system's constraints
evaluated with the numpy witness (``witness.build_witness_np``) on chains of the edge families, the committer's row
digests against the in-circuit sponge, and the decode's negatives (a NaN scale, the padding bit, a forged numerator, a
wrong anchor, a wrong scale) each violating a named constraint.  The pod runs the same system through the prover
(``run.py --relation fp4-nvf4 --auth included-hash gate-vu``)."""
from __future__ import annotations

import numpy as np
import pytest

from .. import hashchain
from .. import poseidon2 as h2
from ..compile import P
from .hashed import FP4_HASHED, FP4Format, NIBBLES, PIN_NAMES, chain_sponge_np, hashed_hints_np, numerator_table_check, sponge_iv
from .relation import K, NUMERATOR, STEPS, WORDS
from .witness import FAMILIES, MODEL, build_witness_np, check_constraints_np, hints_fp4, random_units

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def hr():
    return hashchain.compose(FP4_HASHED)


def _chains(rng, n_vus: int, steps: int, family: str = "random"):
    """``n_vus`` chains of ``steps`` units of ``family``: (a, b) (l, 68) with l = n_vus * steps, c_in (l,), y (l,)."""
    _, A, B = random_units(rng, n_vus * steps, family)
    A, B = A.reshape(n_vus, steps, WORDS), B.reshape(n_vus, steps, WORDS)
    c_in, y = np.zeros((n_vus, steps), dtype=np.int64), np.zeros((n_vus, steps), dtype=np.int64)
    for v in range(n_vus):
        c = 0
        for s in range(steps):
            c_in[v, s] = c
            c = MODEL.step(c, [int(x) for x in A[v, s]], [int(x) for x in B[v, s]])
            y[v, s] = c
    return A.reshape(-1, WORDS), B.reshape(-1, WORDS), c_in.reshape(-1), y.reshape(-1)


def _base(hr, c_in, a, b):
    """The relation's own hints (the prefix of ``sys.hints`` its generator supplies)."""
    import dataclasses
    return hints_fp4(dataclasses.replace(hr.sys, hints=hr.sys.hints[: hr.n_base_hints]), c_in, a, b)


def _witness(hr, a, b, c_in, n_vus, unchecked=False, mutate_hints=None):
    base = _base(hr, c_in, a, b)
    extra, digests, pub = hashed_hints_np(hr, a, b, n_vus, unchecked=unchecked)
    hints = np.concatenate([base, extra], 0)
    if mutate_hints is not None:
        mutate_hints(hints, {n: i for i, (_, n) in enumerate(hr.sys.hints)})
    W = build_witness_np(hr.sys, pub, hints)
    return W, pub, digests


def test_numerator_algebra():
    assert numerator_table_check()
    for code in range(16):
        m, e0, e1, s = code & 1, (code >> 1) & 1, (code >> 2) & 1, code >> 3
        p = e0 * e1
        mag = 2 * e0 + 4 * e1 + 2 * p + m * (1 + e1 + 2 * p)
        assert (1 - 2 * s) * mag == (-1 if s else 1) * NUMERATOR[code & 7]


def test_compose_census(hr):
    c = hr.census
    assert c["rows_base"] == 1583 and c["k"] == WORDS and c["word_bits"] == 4
    assert c["lane_bits"] == 24 and c["rate_lanes_used"] == 12 and c["elements_per_unit"] == NIBBLES == 72
    assert c["permutations_per_unit"] == 2 and c["linked_rows"] == 3 + 2 * h2.CAPACITY
    assert c["rows_private_operands"] < 1200, c            # the decode layer: ~1.1k rows per unit (design: ~900 + 576 bits)
    assert hr.sys.pins == [(i, n) for i, n in hr.sys.pins] and len(hr.sys.pins) == 1 + hashchain.N_DIGEST
    names = hr.extra_hint_names
    assert names[: len(PIN_NAMES)] == list(PIN_NAMES)      # the converted pins first, in the device generator's order
    assert names[len(PIN_NAMES):len(PIN_NAMES) + 4] == ["a[0].b0", "a[0].b1", "a[0].b2", "a[0].b3"]
    assert names[len(PIN_NAMES) + 4 * K:len(PIN_NAMES) + 4 * K + 8] == [f"a[{K}].b{j}" for j in range(8)]
    assert names[-h2.CAPACITY:] == [f"b.cap[{x}]" for x in range(h2.CAPACITY)]
    assert hashchain._block_layout(hr, len(PIN_NAMES))
    hook = hr.sys.witness_hooks[0] if hr.sys.witness_hooks else None
    if hook is not None:
        assert hook.n_lanes == 12 and hook.per == 6 and hook.nb == 4
        assert all(len(s[0]) == 288 for s in hook.sponges)


def test_lane_packing_matches_reference():
    fmt = FP4Format()
    rng = np.random.default_rng(3)
    _, A, _ = random_units(rng, 5, "random")
    lanes = fmt.lanes_np(A)
    assert lanes.shape == (5, h2.RATE) and (lanes[:, 12:] == 0).all() and (lanes < 2**24).all()
    nib = fmt.nibbles_np(A)
    for r in range(5):
        for j in range(12):
            assert lanes[r, j] == sum(int(nib[r, 6 * j + q]) << (4 * q) for q in range(6))
        # the scale bytes' nibbles are the byte's low then high nibble (the bit rows' order)
        for g in range(4):
            assert nib[r, K + 2 * g] == A[r, K + g] & 15 and nib[r, K + 2 * g + 1] == A[r, K + g] >> 4
    with pytest.raises(ValueError):
        fmt.nibbles_np(np.concatenate([A[:, :K] + 16, A[:, K:]], 1))


@pytest.mark.parametrize("family", ["random", "zero-scales", "subnormal-scales", "top-scales", "span-scales", "sparse",
                                    "all-zero-codes", "max-codes", "cancel", "half-codes"])
def test_honest_chains_close(hr, family):
    """Two 4-step chains of the family: every constraint of the composed system holds, the digests published at the chain
    ends equal the committer's row digests, the capacity links chain."""
    rng = np.random.default_rng(abs(hash(family)) % 2**32)
    steps, n_vus = 4, 2
    a, b, c_in, y = _chains(rng, n_vus, steps, family)
    # the composed system is per unit: a 4-step chain runs 4 permutations from the same IV (the constraints are the same;
    # the 24-step row's digest is test_full_rows_digests_match_committer)
    W, pub, digests = _witness_short(hr, a, b, c_in, n_vus, steps)
    bad = check_constraints_np(hr.sys, W, pub)
    assert bad == [], bad[:10]
    # capacity links: chain["y"] at column j == chain["c"] rows at column j + 1 inside a VU, 0 at a start
    from .witness import eval_expr_np
    c_rows, y_exprs = hr.sys.chain["c"], hr.sys.chain["y"]
    for v in range(n_vus):
        for x in range(3, len(c_rows)):
            assert W[c_rows[x], v * steps] == 0
            for s in range(1, steps):
                assert W[c_rows[x], v * steps + s] == eval_expr_np(y_exprs[x], W)[v * steps + s - 1]


def _witness_short(hr, a, b, c_in, n_vus, steps):
    """The numpy witness of ``steps``-unit chains (the test's short chains): the sponge runs ``steps`` permutations."""
    from . import witness as w
    base = _base(hr, c_in, a, b)
    vals = w.public_vectors_fp4(a, b)
    fmt = hr.fmt
    digests = []
    for _, nm, role in hashchain.ROLES:
        x = a if nm == "a" else b
        for i in range(WORDS):
            for j in range(fmt.bit_layout[i]):
                vals[f"{nm}[{i}].b{j}"] = (x[:, i] >> j) & 1
        cap, dg = chain_sponge_np(fmt.lanes_np(x), sponge_iv(role), n_vus, steps)
        for xx in range(h2.CAPACITY):
            vals[f"{nm}.cap[{xx}]"] = cap[xx]
        digests.append(dg)
    digests = np.concatenate(digests, 1)
    extra = np.stack([vals[n] for n in hr.extra_hint_names])
    table = hashchain.public_table(digests, n_vus, steps, a.shape[0])
    pub = {hashchain.HASH_PIN_END: table[:, 0]}
    for x in range(hashchain.N_DIGEST):
        pub[hashchain.hash_pin(x)] = table[:, 1 + x]
    W = build_witness_np(hr.sys, pub, np.concatenate([base, extra], 0))
    return W, pub, digests


def test_full_rows_digests_match_committer(hr):
    """A 24-step VU: ``hashed_hints_np`` (the prover's sponge along the chain) publishes the digests
    the leaf's ``native`` (the committer, ``Poseidon2Leaf.with_lanes(FP4Format)``) computes on the flat 1632-word row."""
    rng = np.random.default_rng(11)
    a, b, c_in, y = _chains(rng, 1, STEPS, "random")
    W, pub, digests = _witness(hr, a, b, c_in, 1)
    assert check_constraints_np(hr.sys, W, pub) == []
    da = hr.leaf.native(a.reshape(1, -1), word_bits=4, role=h2.ROLE_X)
    db = hr.leaf.native(b.reshape(1, -1), word_bits=4, role=h2.ROLE_W)
    assert np.array_equal(digests[:, :8], da) and np.array_equal(digests[:, 8:], db)
    # the IV is its own domain: the 16-bit-lane hash of the same bytes (as 8-bit words) is a different digest
    other = h2.hash_rows_np(a.reshape(1, -1), 8, h2.ROLE_X)
    assert not np.array_equal(other, da)


def _violations(hr, a, b, c_in, n_vus, steps, **kw):
    from . import witness as w
    chk = w._check_domain
    if kw.pop("unchecked", False):
        w._check_domain = lambda *args: None
    try:
        base = _base(hr, c_in, a, b)
        vals = w.public_vectors_fp4(a, b)
    finally:
        w._check_domain = chk
    fmt = hr.fmt
    digests = []
    for _, nm, role in hashchain.ROLES:
        x = a if nm == "a" else b
        for i in range(WORDS):
            for j in range(fmt.bit_layout[i]):
                vals[f"{nm}[{i}].b{j}"] = (x[:, i] >> j) & 1
        cap, dg = chain_sponge_np(fmt.lanes_np(x), sponge_iv(role), n_vus, steps)
        for xx in range(h2.CAPACITY):
            vals[f"{nm}.cap[{xx}]"] = cap[xx]
        digests.append(dg)
    digests = np.concatenate(digests, 1)
    extra = np.stack([vals[n] for n in hr.extra_hint_names])
    hints = np.concatenate([base, extra], 0)
    if kw.get("mutate") is not None:
        kw["mutate"](hints, {n: i for i, (_, n) in enumerate(hr.sys.hints)})
    table = hashchain.public_table(digests, n_vus, steps, a.shape[0])
    pub = {hashchain.HASH_PIN_END: table[:, 0]}
    for x in range(hashchain.N_DIGEST):
        pub[hashchain.hash_pin(x)] = table[:, 1 + x]
    W = build_witness_np(hr.sys, pub, hints)
    return check_constraints_np(hr.sys, W, pub)


def test_negative_nan_scale(hr):
    """A UE4M3 NaN scale byte (0x7F) computed on: the prover's decode is bypassed (``unchecked``), the in-circuit
    finiteness constraint rejects."""
    rng = np.random.default_rng(21)
    a, b, c_in, y = _chains(rng, 1, 2, "random")
    a = a.copy()
    a[1, K + 2] = 0x7F
    bad = _violations(hr, a, b, c_in, 1, 2, unchecked=True)
    assert any("finite" in n for n in bad), bad[:5]


def test_negative_padding_bit(hr):
    rng = np.random.default_rng(22)
    a, b, c_in, y = _chains(rng, 1, 2, "random")
    b = b.copy()
    b[0, K] |= 0x80
    bad = _violations(hr, a, b, c_in, 1, 2, unchecked=True)
    assert any(".pad" in n for n in bad), bad[:5]


def test_negative_forged_numerator(hr):
    """The converted pin ``a[5].n`` is a hint now: changing it without changing the committed bits breaks its constraint."""
    rng = np.random.default_rng(23)
    a, b, c_in, y = _chains(rng, 1, 2, "random")

    def forge(hints, pos):
        hints[pos["a[5].n"], 1] += 1
    bad = _violations(hr, a, b, c_in, 1, 2, mutate=forge)
    assert any("a[5].n" in n for n in bad), bad[:5]


def test_negative_wrong_anchor(hr):
    """``anc.G`` one below / above the max: the range check or the product constraint rejects."""
    rng = np.random.default_rng(24)
    a, b, c_in, y = _chains(rng, 1, 2, "random")
    for delta in (-1, 1):
        def forge(hints, pos, d=delta):
            hints[pos["anc.G"], 0] += d
        bad = _violations(hr, a, b, c_in, 1, 2, mutate=forge)
        assert any("anc" in n for n in bad), (delta, bad[:5])


def test_negative_wrong_scale_and_code_bits(hr):
    """The hashed bits differ from the words the computation used (a scale byte's bit / a code's bit): the decode
    constraints tie the pins to the bits, so a changed bit contradicts the pin (or the boolean / lane)."""
    rng = np.random.default_rng(25)
    a, b, c_in, y = _chains(rng, 1, 2, "top-scales")

    def flip_scale(hints, pos):
        hints[pos[f"a[{K + 1}].b3"], 1] ^= 1
    bad = _violations(hr, a, b, c_in, 1, 2, mutate=flip_scale)
    assert bad, "a flipped scale bit must violate a constraint"

    def flip_code(hints, pos):
        hints[pos["b[17].b2"], 0] ^= 1
    bad = _violations(hr, a, b, c_in, 1, 2, mutate=flip_code)
    assert any("b[17]" in n for n in bad), bad[:5]

    def nonbool(hints, pos):
        hints[pos["a[3].b0"], 0] = 2
    bad = _violations(hr, a, b, c_in, 1, 2, mutate=nonbool)
    assert any("a[3].bool0" in n for n in bad), bad[:5]


def test_negative_part_and_sm_forged(hr):
    rng = np.random.default_rng(26)
    a, b, c_in, y = _chains(rng, 1, 2, "random")
    for pin in ("g[1].part", "g[2].sm", "g[0].X"):
        def forge(hints, pos, p=pin):
            hints[pos[p], 1] += 1
        bad = _violations(hr, a, b, c_in, 1, 2, mutate=forge)
        assert any(pin in n for n in bad), (pin, bad[:5])


def test_float_formats_unchanged():
    """The refactor keeps the BF16 / E4M3 compositions byte-identical (the Rust ``hashed_sys_id`` pins): the system id of
    ``compose(fp8-ada)`` is the one pinned in ligero-verify's relation table."""
    from pathlib import Path
    import re
    from ..relations import relation
    from .hashed import system_id_np as system_id
    src = (Path(__file__).resolve().parents[3] / "ligero-verify" / "src" / "relation.rs").read_text()
    for name in ("fp8-ada", "bf16-hopper"):
        hs = hashchain.compose(relation(name)).sys
        sid = system_id(hs).hex()
        block = src[src.index(f'name: "{name}"'):]
        m = re.search(r'hashed_sys_id:\s*"([0-9a-f]{64})"', block)
        assert m is not None and m.group(1) == sid, (name, sid)
