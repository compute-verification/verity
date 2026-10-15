"""Sampled proofs as a protocol option: the Commit's own commitment and replay, declared here, with no hooks of its own.

Its one scheme, ``vllm-v1``, is the Commit path as it runs today: the commitment scheme `verity_vllm.commit.scheme`
(`verity.ml.commitments.vllm_v1`) over the required values of the row's Program, and the replay's draw: up to
`--replay-per-stratum` units of every stratum, uniformly (`commit.challenge.stratum_picks`).  Its hooks
are the Commit's committer and taps, which the Commit installs exactly as before; this option adds none.  A row whose
target declares no protocols never reaches this module: `protocol_options.commit_context` returns None first.

The Commit binds what it commits to once its roots are final (`Composition.bind`): the run roots, the Program digest and
the required-value manifest's digest.
"""
from __future__ import annotations

from verity_vllm.commit import challenge
from verity_vllm.commit import scheme as commitment_scheme
from verity_vllm.protocol_options.interface import Context, Option, Target

NAME = "sampled-proofs"
DEFAULT_SCHEME = commitment_scheme.NAME
LAW = ("every replay stratum drawn, and a uniform min(k, size)-subset of its units with k = the Commit's "
       "--replay-per-stratum (commit.challenge.stratum_picks)")


def schemes() -> tuple[str, ...]:
    return (commitment_scheme.NAME,)


def refusals(scheme: str, ctx: Context) -> tuple[str, ...]:
    if ctx.stage != "commit":
        return ("sampled proofs runs in the Commit stage of a row (`verity-vllm row`, declared by the target's `protocols`); "
                "verity_vllm.LLM commits nothing",)
    return ()


def requires_build(scheme: str, ctx: Context) -> bool:
    """Sampled proofs verifies the Program emitted by the row's Build."""

    return True


def install(scheme: str, target: Target) -> Option:
    entry = f"{NAME}:{scheme}"
    return Option(name=entry, hooks=target.hooks(entry),
                  details={"commitment_scheme": commitment_scheme.NAME, "law": LAW,
                           "challenge": "legacy (the run root)" if challenge.LEGACY else "verity.ml.randomness.derive from an auditor key"})
