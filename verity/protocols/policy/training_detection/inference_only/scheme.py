"""The inference-only check's declaration in the one table (`verity.protocols.run.schemes`): a component that
charges nothing, run beside the sampled proofs, whose copy of its party is the verifier's manifest
(`verity.protocols.policy.training_detection.inference_only.verifier.parties.MANIFEST`)."""
from __future__ import annotations

from verity.protocol import Role
from verity.protocols.accounting import scheme as S

CHARGES_NOTHING = S.Lacks("charges nothing: the inference-only check credits no resource")

S.register(S.Scheme(
    name="inference-only",
    protocol="verity/inference-only/v0",
    resource=CHARGES_NOTHING,
    blocks=CHARGES_NOTHING,
    assumptions=(),
    log=(),
    charges=(),
    parties=(Role("verifier", "auditor", needs=("component",)),),
    channels=(),
    channel_schemas=(),
    colocated=(),
    needs=frozenset(),
    depends=("sampled-proofs",),
    record="inference-only",
    symbol=CHARGES_NOTHING,
    published=CHARGES_NOTHING,
    check=S.Lacks("the inference-only check's verifier of record is its Lean check, which the one table does not load"),
    reader=CHARGES_NOTHING,
    premises=CHARGES_NOTHING,
    tables=CHARGES_NOTHING,
    output_format=CHARGES_NOTHING,
    product_family=CHARGES_NOTHING,
    y_committed=CHARGES_NOTHING,
    salt_open_root=CHARGES_NOTHING,
    kernels=CHARGES_NOTHING,
    answers=S.Lacks("has no piece that answers outside an exchange"),
))
