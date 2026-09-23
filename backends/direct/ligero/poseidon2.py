"""Poseidon2 over BabyBear (width 24, Plonky3 parameters): the Ligero gadget's view of the ``verity.commitments`` reference.

The permutation, its constants (pinned by ``CONSTANTS_SHA256``), the known-answer vector, the sponge and the operand
packing live in :mod:`verity.commitments.poseidon2_babybear` so the committer (``leaf/v2h``, torch-free) and the
in-relation gadget (``hashchain.py``) are literally the same code; ``permute_ring`` evaluates the permutation over any
ring (Python ints, numpy / torch arrays, or the compiler's affine ``Expr`` with the S-box supplied as product rows).
"""
from verity.commitments.poseidon2_babybear import *  # noqa: F401,F403
from verity.commitments.poseidon2_babybear import __all__  # noqa: F401
