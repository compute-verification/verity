"""The sampled proofs' declaration in the one table (`verity.protocols.run.schemes`): a component that charges
nothing, under the composed id the accepted applications list. The party's manifest and its registration records run
under ``verity/sampled-proofs/one-stage/v1`` (`verity.protocols.proof.spot.verifier.parties.MANIFEST`), its copy of
these parties and channels.

The built executables a sampled-proofs verdict runs (`EXECUTABLES`):

- ``spot-verifier``: the records audit's verdict of record on the registration, the sample and the proved blocks.
- ``spot-flock-caller``: the caller's check, which gates that verdict (`one_stage.verifier.check`).
- ``verity-verify-proof``: each sampled proof block's session, Flock's base verifier.
- ``vstar-recursive``: the composer's sampled-proofs row (``vstar-recursive sampled check``), from the proofs and
  public files.

Flock's ZK server (``flock-zk-server``) runs in the prover's environment and the table-read template's
(``flock-read-tmpl``) in the prover's compose: no verdict runs either."""
from __future__ import annotations

from typing import Any

from verity.protocol import Channel, Role
from verity.protocols.accounting import scheme as S

CHARGES_NOTHING = S.Lacks("charges nothing: the sampled proofs credit no resource")
EXECUTABLES = (S.Executable("verity/protocols/proof/spot/verifier/lean", "spot-verifier"),
               S.Executable("verity/protocols/proof/spot_flock/verifier/lean", "spot-flock-caller"),
               S.Executable("verity/protocols/proof/flock/verifier", "verity-verify-proof"),
               S.Executable("verity/protocols/proof/spot_flock_recursion/verifier/lean", "vstar-recursive"))
#: served_verdicts.sh's variable for each optional binding of SPOT's live challenger
_PP = {"gate": "PP_CHALLENGER", "peer": "PP_PEER", "key": "PP_CHALLENGER_KEY", "gate-key": "PP_GATE_KEY",
       "main-file": "MAIN_FILE"}


def _spot_hooks(cfg, params, bound, pinned):
    """The window's values committed under hiding leaves and sealed, its registration before the sample, and the sampled
    proof blocks proved in Flock --zk; the sample on the live challenger when its gate is bound, else on the challenger
    party's loopback host's coins where ``loopback`` is bound, else on a local development session's, simulated either
    way, which served_verdicts.sh takes as a development run under VERITY_HOLD_PLACEMENT=1 (`PP_DEVELOPMENT_UNDER_HOLD`,
    placement.sh's ``development_under_hold``)."""
    hooks = {v: f"gate:{bound[k]}" if k == "gate" else bound[k] for k, v in _PP.items() if k in bound}
    if "gate" in bound:
        return hooks
    hooks |= {"PP_CHALLENGER": f"loopback:{bound['loopback']}"} if "loopback" in bound else {}
    return {**hooks, "PP_DEVELOPMENT_UNDER_HOLD": "1"}


def launch_spot(declaration: S.Scheme) -> tuple[Any, tuple[str, ...]]:
    """The sampled proofs' workers piece (its `Launch.entry`): its hooks, which run no script of their own. It launches
    and collects, and decides no verdict."""
    return _spot_hooks, ()

S.register(S.Scheme(
    name="sampled-proofs",
    protocol="verity/sampled-proofs/v1",
    resource=CHARGES_NOTHING,
    blocks=CHARGES_NOTHING,
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
    pieces={"workers": S.Launch(optional=("gate", "peer", "key", "gate-key", "loopback", "main-file"), reaches="challenger",
                                starts=("loopback", "main-file"), starts_optional=("loopback", "main-file"),
                                entry="verity.protocols.proof.spot.scheme:launch_spot", executables=EXECUTABLES)},
    answers=S.Lacks("has no piece that answers outside an exchange"),
    record="integrity",
    symbol=CHARGES_NOTHING,
    published=CHARGES_NOTHING,
    check=S.Lacks("the sampled proofs' verifier of record is their party's, which the one table does not load"),
    reader=CHARGES_NOTHING,
    premises=CHARGES_NOTHING,
    tables=CHARGES_NOTHING,
    output_format=CHARGES_NOTHING,
    product_family=CHARGES_NOTHING,
    y_committed=CHARGES_NOTHING,
    salt_open_root=CHARGES_NOTHING,
    kernels=CHARGES_NOTHING,
))
