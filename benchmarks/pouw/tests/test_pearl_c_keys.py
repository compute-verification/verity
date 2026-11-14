"""The scheme's unit keys are the kernels': `pearl_c.keys_of` and `pearl_c.unit_block` give what each kernel's `fixture.py`
gives on a one-unit epoch of its gate's challenger, for every device variant its gate checks, and the bytes `PINNED` holds,
which the two fixtures gave while they derived the keys themselves.
"""
from __future__ import annotations

import hashlib

import pytest
from conftest import load
from verity.protocols.accounting.pouw.prover.schemes import pearl_c as C
from verity.protocols.accounting.pouw.verifier.protocol import Matmul, Shape

KERNELS = {kernel: load(f"{kernel}/fixture.py") for kernel in ("pearl_c_sm120", "pearl_c_sm90")}
M, N, K = 128, 128, 1024
#: sha256 over the keys' names and bytes, in name order, per kernel and device variant
PINNED = {"pearl_c_sm120/sm120": "3a4f0e81075690f0b90087bc68b1bac53ee91d291a1e3efef1458bc3c548afed",
          "pearl_c_sm120/sm120-unpromoted": "3a4f0e81075690f0b90087bc68b1bac53ee91d291a1e3efef1458bc3c548afed",
          "pearl_c_sm90/h100": "8c9b7772eaffc6fc737e9271b65089a314863a21b7537235440cb4eff879f022"}


@pytest.mark.parametrize("kernel,device", [(kernel, device) for kernel, F in KERNELS.items() for device in F.VARIANTS],
                         ids=lambda x: getattr(x, "name", x))
def test_the_schemes_keys_are_the_kernels(kernel: str, device) -> None:
    F = KERNELS[kernel]
    scheme, matmul = C.PearlC(forming="v1", device=device, hashing="h3"), Matmul(F.INDEX, Shape(M, K, N), F.WEIGHT)
    epoch, _, _, root_b = F.epoch_of(matmul, F.CV.rows_f32(F.CASE, "B", N, K), F.test_challenger(), F.SALT_KEY)
    seed_b = C.seed_b(epoch.salt, root_b, scheme.hashing)
    keys = C.keys_of(scheme, epoch, matmul, seed_b)
    assert keys == F.keys_of(scheme, epoch, matmul, seed_b)
    assert C.unit_block(epoch.salt, root_b, matmul.index) == F.unit_block(epoch.salt, root_b, matmul.index) == keys["unit"]
    assert hashlib.sha256(b"".join(n.encode() + keys[n] for n in sorted(keys))).hexdigest() == PINNED[f"{kernel}/{device.name}"]
