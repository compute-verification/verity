"""Registered weights, one rule for every construction that serves a checkpoint (Pearl-C4, and served FP8's Pearl-C).

The keyed 8-block rotation (rung 3) is the rule, keyed after the weights' root is recorded; an unrotated checkpoint
registers only from the curated list, with γ folded where it needs it, for Pearl-C4 and for served FP8.
`check_registration` reads what a registration declares,
`check_linear` which of its linears the stack's rotations reach, and `rotation_key` is the only key its keyed transforms take:
one drawn from the epoch's live salt coin (`Epoch.start`).
No Lean statement reads a registration.
"""

from __future__ import annotations

from collections.abc import Sequence

from verity.ml.randomness import derive

from .audit import Epoch, salt_of
from .coins import CoinOpening, coin_reason
from .protocol import DomainError

#: The keyed 8-block rotation is the rule, and the keyed V/O rotation (each KV head's v_proj rows and o_proj columns)
#: plus the keyed head interleave (o_proj's heads) are adopted inside it, after it.  The keyed latent rotation (MLA's q
#: and kv latents) is adopted after either: each latent a keyed 8-block rotation of its
#: own, on the rows of the linear that writes it (q_a; kv_a's first kv_lora_rank rows), with its RMSNorm (q_a_layernorm,
#: kv_a_layernorm) folded into the columns of the linears that read it (q_b and the DSA indexer's wq_b; kv_b, and with it
#: the W_UK_T and W_UV the absorbed attention derives from kv_b at load).
ROTATION_8 = "keyed-8-block-rotation"
VO_ROTATION = "keyed-vo-rotation"
HEAD_INTERLEAVE = "keyed-head-interleave"
LATENT_ROTATION = "keyed-latent-rotation"
#: Every RMSNorm scale γ folded into the linears it feeds, with no rotation (the `fold` rung).  The rotation folds γ
#: itself (rung 3), so a rotated stack never declares it.
GAMMA_FOLD = "gamma-fold"
#: The transform stacks a registration may apply, in their order
REGISTRATION_STACKS = ((ROTATION_8,), (ROTATION_8, LATENT_ROTATION), (ROTATION_8, VO_ROTATION, HEAD_INTERLEAVE),
                       (ROTATION_8, VO_ROTATION, HEAD_INTERLEAVE, LATENT_ROTATION))
#: The stacks of an unrotated checkpoint, which must be curated
UNROTATED_STACKS = ((), (GAMMA_FOLD,))
ROTATION_KEY_DOMAIN = "verity/pouw/rotation-key/v1"
#: Where a linear sits against the streams the rotations mix: it reads the residual stream (its columns take the keyed
#: 8-block rotation), writes it (its rows take it), reads an MLA latent behind that latent's own RMSNorm (MLA's q_b and
#: kv_b, the DSA indexer's wq_b: only the latent rotation reaches them), or none of these.
READS_RESIDUAL, WRITES_RESIDUAL, READS_LATENT, OFF_RESIDUAL = "reads-residual", "writes-residual", "reads-latent", "off-residual"
RESIDUAL_SIDES = (READS_RESIDUAL, WRITES_RESIDUAL, READS_LATENT, OFF_RESIDUAL)


def check_registration(transforms: Sequence[str], curated: bool = False, needs_gamma_fold: bool = False) -> None:
    """Refuse a registration of a checkpoint's weights outside the registered-weights rule.  `transforms` are the
    transforms it applies, in order; `curated` says the checkpoint is on the curated list, the fallback for an unrotated
    one; `needs_gamma_fold` says it registers only with γ folded, as Llama-3.1-70B does under Pearl-C4: the cast moves its
    layer 0's update by 158% unfolded, and by under 4% with γ folded.

    The keyed 8-block rotation is the rule, alone or followed by the V/O rotation and the head interleave together, and
    either followed by the latent rotation (the stacks measured and adopted).  V/O, the interleave or the latent rotation
    on an unrotated checkpoint is refused, curated or not.  An
    unrotated checkpoint that needs γ folded declares `GAMMA_FOLD`.  This reads what the registration declares; applying
    the transforms is the registration's own, and keying them is `rotation_key`'s."""
    t = tuple(transforms)
    if t in REGISTRATION_STACKS:
        return
    if ROTATION_8 not in t and {VO_ROTATION, HEAD_INTERLEAVE, LATENT_ROTATION} & set(t):
        raise DomainError(f"registration {t}: the V/O rotation, the head interleave and the latent rotation apply inside "
                          "the keyed 8-block rotation only, never on an unrotated checkpoint")
    if t not in UNROTATED_STACKS:
        raise DomainError(f"registration {t}: not an adopted stack, one of {REGISTRATION_STACKS + UNROTATED_STACKS}")
    if not curated:
        raise DomainError(f"registration: an unrotated checkpoint must be on the curated list; apply {ROTATION_8!r} "
                          "(keyed by `rotation_key`) or register a curated checkpoint")
    if needs_gamma_fold and GAMMA_FOLD not in t:
        raise DomainError(f"registration {t}: this checkpoint registers only with γ folded, by the keyed 8-block "
                          f"rotation or by {GAMMA_FOLD!r}")


def check_linear(transforms: Sequence[str], side: str, curated: bool = False, needs_gamma_fold: bool = False) -> None:
    """Refuse crediting one linear of a registration: the registration's own rule (`check_registration`), then, under a
    rotated stack, a linear its rotations don't reach: one that reads an MLA latent where the stack has no latent
    rotation, or one `side` says is off every stream they mix.  The stack leaves such a linear's weights as the
    checkpoint has them, unrotated and not curated, so it is not registered and earns no credit; it is still served and
    bound.  A curated unrotated checkpoint registers each of its linears."""
    if side not in RESIDUAL_SIDES:
        raise ValueError(f"side {side!r}: one of {RESIDUAL_SIDES}")
    check_registration(transforms, curated, needs_gamma_fold)
    if ROTATION_8 not in transforms:
        return
    if side == READS_LATENT and LATENT_ROTATION not in transforms:
        raise DomainError(f"registration {tuple(transforms)}: a linear that reads an MLA latent (q_b, kv_b, the DSA "
                          f"indexer's wq_b) is reached by {LATENT_ROTATION!r} only, so without it it is unregistered "
                          "and uncredited")
    if side == OFF_RESIDUAL:
        raise DomainError(f"registration {tuple(transforms)}: a linear off every stream the stack rotates is not reached, "
                          "so it is unregistered and uncredited")


def rotation_key(epoch: Epoch, opening: CoinOpening) -> bytes:
    """The 32-byte key of a registration's keyed transforms:
    ``derive(salt, "verity/pouw/rotation-key/v1", {"weights": weights_root}).stream("key", 32)`` over the salt of an epoch
    started from `opening`, the challenger's salt coin (`Epoch.start`).  The challenger opens that coin only after it
    recorded the weights' root, so no key exists before the weights are committed, and the prover chooses none.  Refuses
    (`DomainError`) an epoch whose salt is not that coin's: an `Epoch.unverified` one, a beacon's or a fixed seed's."""
    if epoch.coins is None:
        raise DomainError("rotation key: the epoch has no coin commitment; start it from the challenger's salt coin "
                          "(Epoch.start), not from a seed")
    reason = coin_reason(epoch.coins, opening, "salt", epoch.weights_root)
    if reason is not None:
        raise DomainError(f"rotation key: the salt's {reason}")
    if salt_of(epoch.coins, opening.coin, epoch.weights_root) != epoch.salt:
        raise DomainError("rotation key: the epoch's salt is not its salt coin's; start it from that coin (Epoch.start)")
    return derive(epoch.salt, ROTATION_KEY_DOMAIN, {"weights": epoch.weights_root}).stream("key", 32)


__all__ = ["GAMMA_FOLD", "HEAD_INTERLEAVE", "LATENT_ROTATION", "OFF_RESIDUAL", "READS_LATENT", "READS_RESIDUAL",
           "REGISTRATION_STACKS", "RESIDUAL_SIDES", "ROTATION_8", "ROTATION_KEY_DOMAIN", "UNROTATED_STACKS", "VO_ROTATION", "WRITES_RESIDUAL", "check_linear",
           "check_registration", "rotation_key"]
