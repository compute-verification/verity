"""Registered weights (`registration`): one rule for Pearl-C4 and served FP8's Pearl-C, and the keyed transforms' key,
drawn from the epoch's salt coin after the weights' root is recorded and refused for any other salt."""

from __future__ import annotations

import itertools

import pytest

import epoch_coins as E
from verity.core.protocols.pouw import registration as R
from verity.core.protocols.pouw.audit import Epoch, Workload, weights_root
from verity.core.protocols.pouw.protocol import DomainError, Matmul, Shape
from verity.core.protocols.pouw.schemes import get
from verity.core.protocols.pouw.schemes import pearl_c as C
from verity.core.protocols.pouw.schemes import pearl_c4 as F
from verity.ml.randomness import derive

FP8, FP4 = "pearl-c-sm120-v1-h3", "pearl-c-nvfp4-v0"
VO = (R.VO_ROTATION, R.HEAD_INTERLEAVE)
STACKS = (*R.REGISTRATION_STACKS, *R.UNROTATED_STACKS, VO, (R.GAMMA_FOLD, *VO), (R.ROTATION_8, R.GAMMA_FOLD),
          (R.ROTATION_8, R.VO_ROTATION), ("an-unnamed-transform",))
WORK = Workload((Matmul(0, Shape(64, 1024, 128), 0), Matmul(1, Shape(64, 1024, 256), 1)))
ROOTS = {0: b"\x01" * 64, 1: b"\x02" * 64}


def _outcome(scheme, transforms, curated, needs_gamma_fold):
    try:
        scheme.check_registration(transforms, curated, needs_gamma_fold)
    except DomainError as e:
        return str(e)
    return None


def test_both_constructions_take_the_one_shared_rule():
    """Pearl-C4 and served FP8 read the same function, so every declared stack meets the same verdict under both."""
    assert C.check_registration is F.check_registration is R.check_registration
    assert (F.ROTATION_8, F.GAMMA_FOLD, F.REGISTRATION_STACKS) == (R.ROTATION_8, R.GAMMA_FOLD, R.REGISTRATION_STACKS)
    fp8, fp4 = get(FP8), get(FP4)
    assert isinstance(fp8, C.PearlC) and not isinstance(fp8, F.PearlC4)
    for t, curated, fold in itertools.product(STACKS, (False, True), (False, True)):
        assert _outcome(fp8, t, curated, fold) == _outcome(fp4, t, curated, fold), (t, curated, fold)


def test_served_fp8_registers_the_rotated_stacks_and_a_curated_unrotated_checkpoint():
    s = get(FP8)
    for stack in R.REGISTRATION_STACKS:
        for curated, fold in itertools.product((False, True), (False, True)):
            s.check_registration(stack, curated, fold)
    s.check_registration((), curated=True)
    with pytest.raises(DomainError, match=r"curated list; apply 'keyed-8-block-rotation'"):
        s.check_registration(())
    with pytest.raises(DomainError, match="never on an unrotated checkpoint"):
        s.check_registration(VO, curated=True)
    with pytest.raises(DomainError, match="not an adopted stack"):
        s.check_registration((R.ROTATION_8, R.GAMMA_FOLD), curated=True)


def test_served_fp8_folds_gamma_where_the_checkpoint_needs_it():
    s = get(FP8)
    s.check_registration((R.GAMMA_FOLD,), curated=True, needs_gamma_fold=True)
    with pytest.raises(DomainError, match="only with γ folded"):
        s.check_registration((), curated=True, needs_gamma_fold=True)
    with pytest.raises(DomainError, match="curated list"):
        s.check_registration((R.GAMMA_FOLD,), needs_gamma_fold=True)


def _started(seed=E.SEED, roots=ROOTS):
    ch = E.challenger(seed)
    opening = ch.open("salt", weights_root(roots))
    return Epoch.start(WORK, roots, ch.commitment, opening), opening


def test_the_rotation_key_is_the_salt_coins_and_moves_with_the_coin_and_the_weights():
    epoch, opening = _started()
    key = R.rotation_key(epoch, opening)
    assert key == derive(epoch.salt, R.ROTATION_KEY_DOMAIN, {"weights": epoch.weights_root}).stream("key", 32)
    assert len(key) == 32 and key != epoch.salt[:32]
    assert R.rotation_key(*_started()) == key                       # the same coin over the same weights: the same key
    assert R.rotation_key(*_started(b"another challenger")) != key
    assert R.rotation_key(*_started(roots={0: b"\x01" * 64, 1: b"\x03" * 64})) != key


def test_no_rotation_key_exists_before_the_weights_root_is_recorded():
    """A coin opened after anything but this epoch's weights' root keys nothing: the prover can't have the key while it
    still chooses the weights."""
    epoch, _ = _started()
    early = E.challenger().open("salt", weights_root({0: b"\x09" * 64, 1: b"\x02" * 64}))
    with pytest.raises(DomainError, match="not opened after what the verifier recorded"):
        Epoch.start(WORK, ROOTS, E.COINS, early)
    with pytest.raises(DomainError, match="rotation key: the salt's coin was not opened after"):
        R.rotation_key(epoch, early)
    draw = E.challenger().open("draw", weights_root(ROOTS))
    with pytest.raises(DomainError, match="not its 'salt' coin"):
        R.rotation_key(epoch, draw)


def test_a_salt_without_the_live_coin_keys_nothing():
    """An `Epoch.unverified` salt (a beacon's, a fixed seed's) has no key, with or without a coin commitment beside it,
    and a forged salt beside a good opening has none either."""
    epoch, opening = _started()
    with pytest.raises(DomainError, match="no coin commitment"):
        R.rotation_key(Epoch.unverified(WORK, ROOTS, b"a fixed seed"), opening)
    with pytest.raises(DomainError, match="not its salt coin's"):
        R.rotation_key(Epoch.unverified(WORK, ROOTS, opening.coin, coins=E.COINS), opening)
    with pytest.raises(DomainError, match="not its salt coin's"):
        R.rotation_key(Epoch(WORK, ROOTS, b"\x00" * 32, epoch.coins), opening)
