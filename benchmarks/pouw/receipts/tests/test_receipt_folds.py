"""The reference against its own vectors, against the XOR-in-13 values and the r9 twin (both written apart
from it), and the key rules' stated equivalences and rates."""
from __future__ import annotations

import hashlib
import importlib.util
import random
import shutil
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import receipt_conform as C  # noqa: E402
import receipt_folds as F  # noqa: E402

#: the sha256 of the r9 folds' reference twin, receipt_folds_r9.py
P2FF_SHA256 = "c5a3424f90850ca818eff182f43132ae72cf21aadcaacb99758ca454932f5bb4"


def test_the_published_vectors_are_what_the_reference_makes():
    assert C.build() == C.load(), "vectors.json is stale: python receipt_conform.py --write, and say in the PR why they moved"


def test_the_reference_meets_every_vector():
    assert C.check_reference() > 500


def test_xin13_matches_fill_sec_r5s_drop_in_values():
    """The chains that checked pfill's -DPFILL_XIN13 build against the r9 twin with."""
    want = {0: (0x0F64F41C, 0xD07F70C5, 0x965BAAEA), 5: (0x961AD57B, 0xA920F01A, 0x713A93F6),
            1023: (0xB6EE0819, 0x4D1BAFF2, 0x8751D122)}
    words = [tuple(w & F.M32 for w in d) for d in C.R5_WORDS]
    for y0, chain in want.items():
        assert tuple(F.chain("xin13", y0, words, C.R5_KEYS)) == chain


def test_p2ff_r9s_twin_meets_the_vectors():
    path = HERE / "fixtures" / "receipt_folds_r9.py"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == P2FF_SHA256, f"{path} is not P2-FF r9's: restore it from art:91c156a1…"
    spec = importlib.util.spec_from_file_location("receipt_folds_r9", path)
    rf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rf)
    assert C.p2ff_r9(rf) > 100


@pytest.mark.parametrize("compiler", ["g++", "nvcc"])
def test_the_header_meets_every_vector(compiler):
    if not shutil.which(compiler):
        pytest.skip(f"no {compiler}")
    assert C.check_header(compiler=compiler) > 300


def _words() -> list[int]:
    rng = random.Random(19)
    return [rng.getrandbits(32) for _ in range(4000)] + [int(c["a"], 16) for c in C.load()["key_rules"]]


def test_lambda8_is_p2ffs_int8_ratio_on_both_signs():
    for a in _words()[:600]:
        assert F.lambda8_ok(a) == (bool(a & 1) and not F.int8_ratio(a) and not F.int8_ratio(-a & F.M32))


def test_kj19_on_odd_keys_is_no_run_of_twelve_equal_bits():
    for a in _words():
        a |= 1
        run = any(((a >> lo) & 0xFFF) in (0, 0xFFF) for lo in range(1, 20))
        assert F.kj19(a) == (not run), hex(a)


def test_kj19s_exhaustive_count_is_the_combinatorial_one():
    """Odd a fails KJ19 iff bits 1..30 hold a run of 12 equal bits (bit 31 free): a run-length count over 30 bits."""
    count = {(1, 0): 1, (1, 1): 1}   # (run length, last bit) -> strings without a run of 12 yet
    for _ in range(29):
        nxt: dict = {}
        for (r, b), n in count.items():
            for bit in (0, 1):
                k = (r + 1, b) if bit == b else (1, bit)
                if k[0] < 12:
                    nxt[k] = nxt.get(k, 0) + n
        count = nxt
    fails = 2 * ((1 << 30) - sum(count.values()))
    rates = C.load()["key_rates"]
    assert fails == rates["kj19_fails"] == 10481280
    assert abs(fails / rates["odd"] - 4.88e-3) < 1e-5
    assert rates["key_ok_r9"] == rates["key_ok_r11"] + rates["key_ok_r9_and_kj19_fails"]


def test_the_soundness_status_is_the_ruling():
    """XOR-in-13 and LI need (RT) and (G3); ram binds pairs only. Duo is proved for runs of one, two and three words,
    and four at the deployed keys, and needs (D) past them."""
    status = {f: (d["status"], d["needs"]) for f, d in F.FOLDS.items()}
    assert status == {"duo": ("open", ("D",)), "xin13": ("open", ("RT", "G3")), "li": ("open", ("RT", "G3", "HO")),
                      "ram": ("pairs-only", ()), "xin7": ("retired", ())}


def _li_half(k: int, a: int, y: int, d0: int, d1: int) -> int:
    """Lean's `Pouw.Receipt.liHalf` at n = 32."""
    return ((y ^ d0 ^ k) * a + d1) & F.M32


def test_li_is_leans_half_steps_and_a_checkpoint_moves_at_the_changes_lowest_bit():
    """li_step is two of Lean's half steps, and on random chains a word's XOR change moves the checkpoint (the XOR of
    the chains' states) at its lowest set bit and nowhere below it (`LiCheckpointLowBit`)."""
    rng = random.Random(23)
    r = lambda: rng.getrandbits(32)  # noqa: E731
    for _ in range(200):
        y, a0, a1, k0, k1, *d = r(), r() | 1, r() | 1, r(), r(), r(), r(), r(), r()
        assert F.li_step(y, *d, a0, a1, k0, k1) == _li_half(k1, a1, _li_half(k0, a0, y, d[0], d[1]), d[2], d[3])
    for _ in range(200):
        chains = [([(r(), r() | 1, r(), r()) for _ in range(16)], r()) for _ in range(8)]

        def checkpoint(cs):
            x = 0
            for steps, y in cs:
                for k, a, d0, d1 in steps:
                    y = _li_half(k, a, y, d0, d1)
                x ^= y
            return x

        c, i, slot, x = rng.randrange(8), rng.randrange(16), rng.randrange(2), r() or 1
        changed = [(list(s), y) for s, y in chains]
        k, a, d0, d1 = changed[c][0][i]
        changed[c][0][i] = (k, a, d0 ^ x, d1) if slot == 0 else (k, a, d0, d1 ^ x)
        diff = checkpoint(chains) ^ checkpoint(changed)
        assert diff & -diff == x & -x


def test_the_e2e_key_rule_is_weaker_than_key_ok_r9():
    """E2E's xin keys (segment.XA0, XA1) are drawn under Λ8 alone; one of them fails key_ok and another KJ19."""
    xa0, xa1 = F.e2e_xin_keys()
    assert not F.key_ok_r9(xa0[1]) and F.lambda8_ok(xa0[1])
    assert F.key_ok_r9(xa1[0]) and not F.kj19(xa1[0])
