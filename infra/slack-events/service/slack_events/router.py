"""The model step's seam: who should see a post that names no one.  Off until it is built and measured."""
from __future__ import annotations

import os


def route_unaddressed(post: dict, thread: list[dict], registry: dict) -> list[str] | None:
    """The handles a person's unaddressed post should ring, or None for no-model mode, where the consumer rings @top and
    says so in the thread.  None unless ROUTER_MODEL_ENABLED=1.

    The planned call (not built; nothing here calls a model):

    - Vercel AI Gateway's OpenAI-compatible endpoint (`https://ai-gateway.vercel.sh/v1`), authenticated by the deployment's
      OIDC token (`VERCEL_OIDC_TOKEN`), so there is no provider key to hold; a route with zero data retention.
    - A fast model, its cross-model fallbacks carried in each request (`providerOptions.gateway.models`), the IDs taken from
      Gateway's live model list when it is built.
    - Input: the post, its thread, and each handle's remit, owned paths and `examples` from `registry`.  Output: JSON with 1
      to 3 handles and a one-line reason (or `answer`, for the chat agent).
    - The consumer validates every name against the registry's handles and never accepts `top` from the model; no valid
      name, an error or a timeout sends the post to @top with a line saying why.
    - It turns on only after the replay of Daniel's addressed posts, names stripped, scores 8 in 10 or better."""
    if os.environ.get("ROUTER_MODEL_ENABLED") != "1":
        return None
    raise NotImplementedError("ROUTER_MODEL_ENABLED=1, but the model step is not built: unset it")
