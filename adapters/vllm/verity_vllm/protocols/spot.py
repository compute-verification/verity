"""SPOT as a protocol option: the Commit's own commitment and replay, declared here, with no hooks of its own.

Its scheme, ``rows-sha512``, names what the Commit commits under: hm96-sha512 rows in frame-v3-sha512 trees, 64-byte run
roots (`verity_vllm.commit.step_rows`), over the required values of the row's Program, and the replay's sample: up to
`--replay-per-stratum` units of every stratum, uniformly (`commit.challenge.stratum_picks`).  Its hooks are the Commit's
committer and taps, which the Commit installs exactly as before; this option adds none, and the Commit refuses an entry
whose scheme is not its committer's (`protocols.commit_context`).  ``vllm-v1`` (SHA-256,
32-byte roots) is registered only so a recorded run's target still parses and digests as it did; a Commit refuses it by
name.  A row whose target declares no protocols never reaches this module: `protocol_options.commit_context` returns None
first.

The Commit binds what it commits to once its roots are final (`Composition.bind`): the run roots, the Program digest and
the required-value manifest's digest.  It takes no settings.
"""
from __future__ import annotations

from verity_vllm.commit import step_rows
from verity import hooks as H
from verity_vllm.protocols.interface import BUILD_BOUND, Context, Option, Target

NAME = "sampled-proofs"
DEFAULT_SCHEME = step_rows.NAME
RECORDED = "vllm-v1"
DESIGN = ("every replay stratum drawn, and a uniform min(k, size)-subset of its units with k = the Commit's "
       "--replay-per-stratum (commit.challenge.stratum_picks)")


def schemes() -> tuple[str, ...]:
    return (DEFAULT_SCHEME, RECORDED)


def refusals(scheme: str, ctx: Context) -> tuple[str, ...]:
    if ctx.stage != "commit":
        return (("SPOT runs in the Commit stage: a row's (`verity-vllm row`, declared by the target's `protocols`) "
                 "or serve-openai's Commit windows; verity_vllm.LLM commits nothing"),)
    if scheme == RECORDED:
        return ((f"{RECORDED} (SHA-256, 32-byte roots) is kept only to read recorded runs: a Commit commits under "
                 f"{DEFAULT_SCHEME}, so declare {NAME}:{DEFAULT_SCHEME}"),)
    if own := sorted(ctx.settings_of(NAME)):
        return (f"the {NAME} settings {own}: it takes none",)
    return ()


def requests(scheme: str, ctx: Context) -> tuple[H.Request, ...]:
    """SPOT verifies the Program emitted by the row's Build."""
    return (BUILD_BOUND,)


def install(scheme: str, target: Target) -> Option:
    entry = f"{NAME}:{scheme}"
    return Option(name=entry, hooks=target.hooks(entry),
                  details={"commitment_scheme": scheme, "law": DESIGN,
                           "challenge": "verity.commitments.randomness.derive from an auditor key"})


__all__ = ["DEFAULT_SCHEME", "NAME"]
