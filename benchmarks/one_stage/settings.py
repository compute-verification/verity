"""The one-stage audit commands' own environment variables, declared once in a `verity_config.settings.Registry`;
the shared ones are `verity_config.settings.SHARED`'s."""
from __future__ import annotations

from verity_config.settings import VALUE, Registry, Setting

REGISTRY = Registry(__file__, (),
    Setting("SPOT_LEVELS", str, None, VALUE, "the levels of blocks a served command commits and audits (SPOT: 1, or 2: "
                                             "outer blocks, then inner blocks inside each sampled outer block): what the "
                                             "prover served, never what the verifier checks; a command that can't run it "
                                             "refuses it by name (`verity.protocols.proofs.spot.verifier.levels`)"),
)
