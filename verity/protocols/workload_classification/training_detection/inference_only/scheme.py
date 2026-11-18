"""The inference-only check's declaration in the one table (`verity.protocols.compose.schemes`): a component that
charges nothing, run beside the sampled proofs, whose copy of its party is the verifier's manifest
(`verity.protocols.workload_classification.training_detection.inference_only.verifier.parties.MANIFEST`)."""
from __future__ import annotations

from verity.protocol import Role
from verity.protocols.resource_accounting import scheme as S

CHARGES_NOTHING = S.Lacks("charges nothing: the inference-only check credits no resource")

S.register(S.Scheme(
    name="inference-only",
    protocol="verity/inference-only/v0",
    resource=CHARGES_NOTHING,
    blocks=(),
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
    record_owner="proofs",
))
