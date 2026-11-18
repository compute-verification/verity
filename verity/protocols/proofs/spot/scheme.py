"""The sampled proofs' declaration in the one table (`verity.protocols.compose.schemes`): a component that charges
nothing, under the composed id the accepted applications list. The party's manifest and its registration records run
under ``verity/sampled-proofs/one-stage/v1`` (`verity.protocols.proofs.spot.verifier.parties.MANIFEST`), its copy of
these parties and channels."""
from __future__ import annotations

from verity.protocol import Channel, Role
from verity.protocols.resource_accounting import scheme as S

CHARGES_NOTHING = S.Lacks("charges nothing: the sampled proofs credit no resource")

S.register(S.Scheme(
    name="sampled-proofs",
    protocol="verity/sampled-proofs/v1",
    resource=CHARGES_NOTHING,
    blocks=(),
    assumptions=(),
    log=(),
    charges=(),
    parties=(Role("challenger", "auditor", needs=("host",)), Role("firewall", "developer", needs=("pin",)),
             Role("workers", "developer", needs=("serve",)),
             Role("verifier", "auditor", needs=("audit", "component", "product"))),
    channels=(Channel("challenger", "verifier", ("Challenger",), carries_coins=True),
              Channel("verifier", "challenger", ("Done",)),
              Channel("firewall", "verifier", ("Firewall",)),
              Channel("workers", "verifier", ("Registration",))),
    channel_schemas=S.Lacks("its messages' schemas are the verifier's manifest's"),
    colocated=(),
    needs=frozenset({"windows"}),
    depends=(),
    record="integrity",
    symbol=CHARGES_NOTHING,
    published=CHARGES_NOTHING,
    record_owner="proofs",
))
