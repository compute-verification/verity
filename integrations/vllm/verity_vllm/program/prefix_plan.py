"""The cached prefix of every request of a prefix-caching workload, from its declared arrival chronology and prompt ids alone.

vLLM v1 (`KVCacheManager.get_computed_blocks`) hashes a prompt in full blocks of `block_size` tokens, each hash chained on the block
before it, and starts a new request on the longest run of leading blocks already cached, never all of its prompt (at most
`num_tokens - 1` rows: the last prompt row is always computed, for its logits).  A block is cached when the scheduler allocates it,
so a request scheduled earlier in the same step has already cached it.  The scheduler takes requests in arrival order (arrive_step,
then index).  For request r, over the requests q scheduled before it:

    C_r            = bs * min((L_r - 1) // bs, max_q min(common(q, r), L_q) // bs)
    producer(r, b) = the first q with L_q >= (b + 1) * bs whose first (b + 1) * bs prompt ids are r's        (block b < C_r / bs)

r's prefill then computes rows C_r..L_r-1 at positions C_r.. (its launch rows), and its attention reads rows 0..C_r-1 of every layer's
key/value cache as values the producers wrote: carried over, never recomputed.  A request whose prompt repeats tokens an earlier one
generated could also hit the blocks that request filled while decoding; the plan does not predict those, and Match's engine check
(the scheduler's num_computed_tokens at the request's first step) shows it.

A shape is (LP, T) for a request served whole and (LP, T, C) for one with C cached rows, so every shape without a cached prefix keeps
its key, its directory name and its Program digest.
"""
from __future__ import annotations

from typing import Any

#: vLLM's block size when the row does not set `engine.args.block_size` (the FlashAttention backends).  A backend that supports other
#: sizes only (the sparse MLA backends: 64) gets its own from vLLM unless the row declares one, so its prefix row declares it; GM-01 (G5,
#: `check.match.prefix_reads.check_hits`) holds the block size the engine served at to the plan's.
DEFAULT_BLOCK_SIZE = 16


def prefix_caching(workload: dict[str, Any]) -> bool:
    """Whether the row serves with vLLM's prefix cache on (`engine.args.enable_prefix_caching`)."""
    return ((workload.get("engine") or {}).get("args") or {}).get("enable_prefix_caching") is True


def block_size(workload: dict[str, Any]) -> int:
    return int(((workload.get("engine") or {}).get("args") or {}).get("block_size") or DEFAULT_BLOCK_SIZE)


def _common(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def plan(workload: dict[str, Any]) -> list[dict[str, Any]]:
    """Per request of `workload["requests"]` (in that order): {"cached": C, "producers": [request position per cached block]}.
    Every C is 0 when the row serves without the prefix cache."""
    reqs = workload.get("requests") or []
    if not prefix_caching(workload):
        return [{"cached": 0, "producers": []} for _ in reqs]
    missing = [i for i, r in enumerate(reqs) if not isinstance(r.get("prompt_token_ids"), list)]
    if missing:
        raise ValueError(f"the row serves with the prefix cache, so its cached prefixes come from the prompts, but requests {missing} "
                         f"declare no prompt_token_ids")
    bs = block_size(workload)
    order = sorted(range(len(reqs)), key=lambda i: (int(reqs[i].get("arrive_step", 0) or 0), i))
    out: list[dict[str, Any]] = [{"cached": 0, "producers": []} for _ in reqs]
    for k, r in enumerate(order):
        ids = reqs[r]["prompt_token_ids"]
        cap = (len(ids) - 1) // bs
        producers: list[int] = []
        for b in range(cap):
            hi = (b + 1) * bs
            q = next((q for q in order[:k] if len(reqs[q]["prompt_token_ids"]) >= hi
                      and _common(reqs[q]["prompt_token_ids"][:hi], ids[:hi]) == hi), None)
            if q is None:
                break
            producers.append(q)
        out[r] = {"cached": len(producers) * bs, "producers": producers}
    return out


def cached_of(workload: dict[str, Any]) -> list[int]:
    """C per request of `workload["requests"]`."""
    return [p["cached"] for p in plan(workload)]


def cached_by_id(workload: dict[str, Any]) -> dict[str, int]:
    """C per request id of `workload["requests"]`."""
    return {str(r.get("request_id")): c for r, c in zip(workload.get("requests") or [], cached_of(workload))}


def with_cached(workload: dict[str, Any]) -> list[dict[str, Any]]:
    """`workload["requests"]`, each with its `cached` rows: what a request's Program lookup (`request_shape`) reads."""
    return [dict(r, cached=c) for r, c in zip(workload.get("requests") or [], cached_of(workload))]


def request_shape(r: dict[str, Any]) -> tuple[int, ...]:
    """The shape of a declared request ({"prompt_len", "max_tokens"[, "cached"]}): its Program's (LP, T[, C])."""
    return shape(int(r["prompt_len"]), int(r["max_tokens"]) - 1, int(r.get("cached") or 0))


def declares(result: dict[str, Any], s: tuple[int, ...]) -> bool:
    """Whether a derive's result.json declares shape `s`: its LP, T and cached rows C (0 when it records none)."""
    return (int(result.get("LP", -1)), int(result.get("T", -1)), int(result.get("C") or 0)) == (s[0], s[1], cached_rows(s))


def shape(LP: int, T: int, C: int = 0) -> tuple[int, ...]:
    return (int(LP), int(T)) if not C else (int(LP), int(T), int(C))


def shape_name(s: tuple[int, ...]) -> str:
    """`LP{lp}_T{t}`, and `_C{c}` for a cached prefix: the request derive's directory suffix (`build_request_<name>`)."""
    return f"LP{s[0]}_T{s[1]}" + (f"_C{s[2]}" if len(s) > 2 and s[2] else "")


def shape_key(s: tuple[int, ...]) -> str:
    """`"LP,T"`, or `"LP,T,C"`: the key GP-01 and the Build's records file a shape under."""
    return ",".join(str(int(x)) for x in shape(*s))


def parse_shape_key(k: str) -> tuple[int, ...]:
    return shape(*(int(x) for x in k.split(",")))


def cached_rows(s: tuple[int, ...]) -> int:
    return int(s[2]) if len(s) > 2 else 0
