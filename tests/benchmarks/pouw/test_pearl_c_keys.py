"""The scheme's unit keys are the kernels': `pearl_c.keys_of` and `pearl_c.unit_block` give what each kernel's `fixture.py`
gives on a one-unit epoch of its gate's challenger over B's tree under the fixture's salts, for every device variant its
gate checks, and the bytes `PINNED` holds.
"""
from __future__ import annotations

import hashlib

import pytest
from conftest import load
from verity.protocols.accounting.compute.pouw.prover.schemes import pearl_c as C
from verity.protocols.accounting.compute.pouw.verifier.protocol import Matmul, Shape

KERNELS = {kernel: load(f"{kernel}/fixture.py") for kernel in ("pearl_c_sm120", "pearl_c_sm90")}
M, N, K = 128, 128, 1024
#: sha256 over the keys' names and bytes, in name order, per kernel and device variant
PINNED = {"pearl_c_sm120/sm120": "cf402602d733d653e315601a0504f46c6acddc1b535e384e6dfa1c863c14fcd1",
          "pearl_c_sm120/sm120-unpromoted": "cf402602d733d653e315601a0504f46c6acddc1b535e384e6dfa1c863c14fcd1",
          "pearl_c_sm90/h100": "4a8f41b446a2872320df1d0df2a4692f4f71932f9e723be69678d6cfcb37f631"}


@pytest.mark.parametrize("kernel,device", [(kernel, device) for kernel, F in KERNELS.items() for device in F.VARIANTS],
                         ids=lambda x: getattr(x, "name", x))
def test_the_schemes_keys_are_the_kernels(kernel: str, device) -> None:
    F = KERNELS[kernel]
    scheme, matmul = C.PearlC(forming="v1", device=device, hashing="h3"), Matmul(F.INDEX, Shape(M, K, N), F.WEIGHT)
    epoch, _, _, root_b = F.epoch_of(matmul, F.CV.rows_f32(F.CASE, "B", N, K), F.test_challenger(), F.salts_of("b", N))
    seed_b = C.seed_b(epoch.salt, root_b, scheme.hashing)
    keys = C.keys_of(scheme, epoch, matmul, seed_b)
    assert keys == F.keys_of(scheme, epoch, matmul, seed_b)
    assert C.unit_block(epoch.salt, root_b, matmul.index) == F.unit_block(epoch.salt, root_b, matmul.index) == keys["unit"]
    assert hashlib.sha256(b"".join(n.encode() + keys[n] for n in sorted(keys))).hexdigest() == PINNED[f"{kernel}/{device.name}"]
