"""Decode rows with draft positions: what the step compiler (`steps.compile_steps`) makes of a launch that decodes speculatively,
at the bf16 KV cache.

Under speculative decoding a decode row is its request's last known token and then up to K draft tokens, K the launch's
``num_speculative_tokens``. The step verifies the drafts: the model runs the row's 1 + d positions, the accept keeps the first
``kept`` of the d drafts, and the request's outputs grow by ``kept`` + 1 tokens, the kept drafts and one more. The request's
next row starts at the first position past them, so the positions of the rejected drafts are computed again, on kept tokens.

A request's Program follows its rows. Its draft schedule (`steps.RequestUnit.drafts`) is each of its decode passes' (d, kept),
read off the advice (`window_advice.drafts`): a row's kept count is its request's next row's start less its own, less one, and
its last row keeps what its outputs still lack, so no field of the advice says it. One pass of the Program (`drive`):

- calls the draft layer on the pass's first token and feeds the first d of its K proposals after that token;
- runs the decoder over the 1 + d rows as one sequence's rows of one launch, on the history of the kept positions;
- selects a token at each of the 1 + d rows, and calls the accept over the d drafts and the 1 + d selected tokens;
- returns the first ``kept`` + 1 selected tokens as the request's next outputs;
- returns the check that the accept's count is ``kept`` (`KeptCheck_v1`), which the verifier requires to be 1; in the request's
  last pass, that it is at least ``kept``, since there the request's length rule cuts the pass's tokens;
- cuts every layer's history back to the kept positions.

The kept counts are advice, charged in the rows that say them, and they shape the Program before the step that outputs the
tokens they count: a schedule the model didn't keep makes a check 0. A launch without drafts compiles as it did: its units
hold no schedule, and neither `build.build_program` nor a unit's key reads this module.

**Roles.** The accept and the draft layer are reached by role (`ROLES`), each with the Definition family it binds and that
family's ports, so naming another Definition is that entry's change:

- ``draft_accept``: a greedy request's accept, the registered `Accept_v1{K}` at K = d;
- ``sampled_accept``: a sampled request's accept. It is declared with its ports and bound to nothing: its ``logits`` port
  takes the model's rows after temperature and top-p, which no registered unit writes, so a sampled request with a draft row
  is refused by name;
- ``draft_layer``: the model's draft layer. It is bound to a stand-in (`StandIn`: a registered table from a pass's first token
  to K drafts) until its Definition is registered. A stand-in computes what no served engine ran, so a compile under one is
  the registration's own choice (`steps.Registered.draft_layer`), its Programs' SRC configs name it, and a served engine's φ
  (`phi.served_phi`) refuses a launch with drafts.

Refused by name (`Unexpressible`): a KV cache format a request Program doesn't state, the launch's or one its checkpoint
turns on (`domain.cache_refusal`); a draft row in a launch that registers no draft, or more drafts than it registers; a
launch with drafts and no draft layer; drafts off a batch-invariant target, off one rank or off the dense decoder; and a
request with a draft row that is sampled, hits the prefix cache, is preempted, is aborted, or is carried into or out of its
window.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.refs import Coll, array_of, concat
from verity.circuits.types import Array
from verity.definitions.ops.prims import const
from verity.definitions.ops.scalar import BIT, I32, I32Eq, I32Le, SelectI32

from .common import Unexpressible
from .domain import cache_refusal

__all__ = ["ENGINE_ARG", "ROLES", "KeptCheck", "Role", "StandIn", "accept", "drive", "launch", "schedule", "src_config"]

#: the engine arg that states a launch's drafts: the most a decode row carries
ENGINE_ARG = "num_speculative_tokens"


@dataclass(frozen=True)
class Role:
    """A declared role: the Definition family it names, that family's ports, whether a Definition is bound to it here, and
    whether what is bound is a stand-in, which no served launch compiles under."""

    definition: str
    ports: str
    bound: bool
    stand_in: bool
    reads: str


ROLES: dict[str, Role] = {
    "draft_accept": Role(
        "Accept_v1{K}",
        "(cand: Array<K, i32>, argmax: Array<K+1, i32>) -> (verdict: Array<K, i32>, accepted: i32, appended: Array<K+1, i32>)",
        True, False, "a greedy request's drafts and the tokens the model selects at the pass's rows"),
    "sampled_accept": Role(
        "SampledAccept_v1{K,V,TILE}",
        "(logits: Array<K+1, Array<V, f32>>, draft: Array<K, i32>, seed: i64, pos: Array<K+1, i64>, temp: f32) -> "
        "(verdict: Array<K, i32>, accepted: i32, appended: Array<K+1, i32>)",
        False, False, "a sampled request's drafts and the pass's logit rows after temperature and top-p, which no registered "
                      "unit writes"),
    "draft_layer": Role(
        "DraftLayerStandIn_v1{K,TABLE}", "(token: i32) -> Array<K, i32>", True, True,
        "the pass's first token alone; the model's draft layer reads the decoder's hidden row too, and its ports are its "
        "Definition's to name"),
}


def _layer_refusal(S: Mapping[str, Any]) -> str | None:
    k, table = S.get("K", 1), S.get("TABLE")
    ok = type(k) is int and 1 <= k < 1 << 8
    if ok and table is not None:
        ok = isinstance(table, tuple) and 1 <= len(table) < 1 << 16 and all(
            isinstance(row, tuple) and len(row) == 2 and type(row[0]) is int and isinstance(row[1], tuple)
            and len(row[1]) == (k if "K" in S else len(row[1])) and all(type(x) is int and 0 <= x < 1 << 31 for x in (row[0], *row[1]))
            for row in table) and [row[0] for row in table] == sorted({row[0] for row in table})
    if not ok:
        return (f"K must be in [1, 2^8) and TABLE at least one (token, its K drafts), token ids in [0, 2^31), the tokens "
                f"increasing (got K={k!r}, TABLE={table!r:.200})")
    return None


def _layer_body(B: Any, S: Any, token: Coll) -> Coll:
    """Each draft the token's own, replaced by its table row's where the token is a row's (the rows' tokens differ)."""
    hits = [(B.call(I32Eq, token, B.call(const(32, t))), drafts) for t, drafts in S.TABLE]
    slots = []
    for j in range(S.K):
        slot = token
        for hit, drafts in hits:
            slot = B.call(SelectI32, hit, B.call(const(32, drafts[j])), slot)
        slots.append(slot)
    return array_of(slots)


DraftLayerStandIn = CompositeDefinition(
    "DraftLayerStandIn", 1, ("K", "TABLE"), lambda S: ((("token", I32),), Array(S.K, I32)), _layer_body, register=False,
    refuse=_layer_refusal, doc="a stand-in for a model's draft layer: the K drafts TABLE holds for the token, the token itself K "
                               "times for a token it doesn't hold",
    conformance="not-yet-established: a stand-in, which no served engine computes")


def _kept_refusal(S: Mapping[str, Any]) -> str | None:
    kept, last = S.get("KEPT", 0), S.get("LAST", 0)
    if not (type(kept) is int and 0 <= kept < 1 << 8 and last in (0, 1) and type(last) is int):
        return f"KEPT must be in [0, 2^8) and LAST 0 or 1 (got KEPT={kept!r}, LAST={last!r})"
    return None


def _kept_body(B: Any, S: Any, accepted: Coll) -> Coll:
    want = B.call(const(32, S.KEPT))
    return B.call(I32Le, want, accepted) if S.LAST else B.call(I32Eq, accepted, want)


KeptCheck = CompositeDefinition(
    "KeptCheck", 1, ("KEPT", "LAST"), lambda S: ((("accepted", I32),), BIT), _kept_body, register=False, refuse=_kept_refusal,
    doc="1 exactly when the accept's count `accepted` is KEPT, the drafts the advice says the pass kept; at LAST = 1, the "
        "request's last pass, when it is at least KEPT (signed)",
    conformance="not-yet-established: one comparison of the accept's count with the schedule's")


@dataclass(frozen=True)
class StandIn:
    """The draft layer's stand-in, as a registration names it: ``table``, each (token, the drafts after it), the tokens
    increasing and every row K drafts long at the launch's K. `propose` is its function."""

    table: tuple[tuple[int, tuple[int, ...]], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "table", tuple((t, tuple(drafts)) for t, drafts in self.table))

    def definition(self, K: int) -> Any:
        """`DraftLayerStandIn_v1` at the launch's K and this table, or `Unexpressible` naming why it has none."""
        try:
            return bind(DraftLayerStandIn, K=K, TABLE=self.table)
        except ValueError as e:
            raise Unexpressible(f"the draft layer's stand-in has no Definition at {K} drafts: {e}") from None

    def propose(self, token: int) -> tuple[int, ...]:
        """The drafts after ``token``: its table row's, else the token itself, as many as the table's rows hold."""
        return dict(self.table).get(token, (token,) * len(self.table[0][1]))

    def config(self) -> dict:
        return {"stand_in": ROLES["draft_layer"].definition, "table": [[t, list(drafts)] for t, drafts in self.table]}


def accept(policy_kind: str, d: int, who: str = "") -> Any:
    """The accept a pass of d drafts calls under a request's sampling: role ``draft_accept``'s `Accept_v1{K=d}` for a greedy
    one; `Unexpressible` for a sampled one (``who`` names it), whose role ``sampled_accept`` is bound to nothing."""
    if policy_kind == "greedy":
        from verity.definitions.registry.spec import Accept
        return bind(Accept, K=d)
    role = ROLES["sampled_accept"]
    raise Unexpressible(f"{who}a {policy_kind} request with a draft row: its accept is the role sampled_accept's, {role.definition} "
                        f"{role.ports}, which reads {role.reads}; no Definition is bound to it")


def launch(registered: Any) -> int:
    """K, the most drafts a decode row of the registered launch carries; 0 for a launch without drafts. `Unexpressible` for a
    KV cache format no request Program states, for drafts the compiler doesn't hold (an arg that is no positive count, a
    target without batch invariance, more than one rank), and for a launch with drafts and no draft layer or one without
    and a draft layer."""
    phi = registered.phi
    if why := cache_refusal(phi):
        raise Unexpressible(why)
    K, layer = (phi.engine or {}).get(ENGINE_ARG), registered.draft_layer
    if K is None or (type(K) is int and K == 0):
        if layer is not None:
            raise Unexpressible(f"the registration names a draft layer, and its launch drafts no token (engine arg {ENGINE_ARG} "
                                f"{K!r})")
        return 0
    if type(K) is not int or K < 1:
        raise Unexpressible(f"engine arg {ENGINE_ARG} {K!r}: the drafts a decode row carries at most, a positive count")
    facts = [] if phi.target_profile().batch_invariant else ["a target without batch invariance, whose launches a draft row changes"]
    if (world := int((phi.tp or {}).get("world", 1))) != 1:
        facts.append(f"tensor-parallel world {world}")
    if facts:
        raise Unexpressible(f"a launch that drafts {K} tokens a decode row on {' and '.join(facts)}: draft rows compile on one "
                            f"rank of a batch-invariant target")
    if not isinstance(layer, StandIn):
        role = ROLES["draft_layer"]
        raise Unexpressible(f"a launch that drafts {K} tokens a decode row, and the registration names no draft layer"
                            f"{'' if layer is None else f' this compiler knows ({type(layer).__name__})'}: no Definition of the "
                            f"model's draft layer is registered, and its role holds a stand-in ({role.definition}), which a "
                            f"registration names itself (`Registered.draft_layer`)")
    layer.definition(K)
    return K


def schedule(i: int, q: Any, rows: Sequence[tuple[int, int, Any]], counts: Mapping[tuple[int, int], tuple[int, int]], K: int, *,
             C: int = 0, carried: bool = False, left: bool = False) -> tuple[tuple[int, int], ...]:
    """Request i's draft schedule: each decode pass's (drafts, kept) in order, from its rows past its prompt (``q``: its
    advice record; ``rows``: (step, index in the step, row) each; ``counts``: `window_advice.drafts`); () for a request none
    of whose rows carries a draft. A row past its last output is in no pass. `Unexpressible` for a draft row where the launch
    registers none, or one of more drafts than it registers, and for a request with one whose Program this module doesn't
    write: one a row resumes, one with a prefix-cache hit (``C``), one its client aborted, and one its window carries in
    (``carried``) or leaves with outputs to keep (``left``)."""
    mine = [(s, counts[s, j]) for s, j, _ in rows if (s, j) in counts]
    if not mine:
        return ()
    if not K:
        raise Unexpressible(f"request {i}: step {mine[0][0]} runs a draft row of {mine[0][1][0]} drafts, and the registered launch "
                            f"drafts no token (engine arg {ENGINE_ARG})")
    if (over := next(((s, d) for s, (d, _) in mine if d > K), None)) is not None:
        raise Unexpressible(f"request {i}: step {over[0]} runs a draft row of {over[1]} drafts, over the registered {ENGINE_ARG} "
                            f"{K}")
    if (resumed := next((s for s, _, row in rows if row.preempted), None)) is not None:
        raise Unexpressible(f"request {i}: step {resumed} resumes it, and it has a draft row: a preempted request's draft rows "
                            f"aren't compiled")
    why = (f"a prefix-cache hit of {C} tokens" if C else "a client abort, where its last pass's count isn't its length rule's cut"
           if q.finish == "abort" else "a Commit window that carries it in" if carried else
           "a Commit window that leaves it with outputs to keep" if left else None)
    if why:
        raise Unexpressible(f"request {i}: draft rows on a request with {why}, which its Program doesn't hold")
    passes: list[tuple[int, int]] = []
    lacks = q.output_len - 1
    for s, j, row in rows:
        if row.start < q.prompt_len or lacks <= 0:
            continue
        d, c = counts.get((s, j), (0, 0))
        passes.append((d, c))
        lacks -= c + 1
    return tuple(passes)


def src_config(phi: Any, drafts: tuple, layer: Any) -> dict:
    """What a request Program's SRC config says of its draft schedule: the launch's K, each pass's (drafts, kept), and the
    draft layer it calls; nothing without a schedule."""
    if not drafts:
        return {}
    return {"drafts": {"K": phi.engine[ENGINE_ARG], "passes": [list(p) for p in drafts], "layer": layer.config()}}


def drive(model: Any, phi: Any, rows: list[Coll], *, LP: int, T: int, drafts: tuple, layer: Any, plain: bool) -> list[Coll]:
    """A greedy request's forwards under its draft schedule -> its T + 1 samples (the module's text): the prefill of its LP
    rows and the sample after it, then each pass of ``drafts``. The checks of its passes with a draft are the model's
    (`common.with_checks`). ``plain``: the request is one the passes are written for, a greedy request wrapper on one rank
    of the dense decoder with no history, chunk, edge or launch batch; `Unexpressible` otherwise, and for a schedule
    that doesn't make the request's T tokens after its first."""
    K = (phi.engine or {}).get(ENGINE_ARG)
    if not plain or model.ctx is not None or not isinstance(layer, StandIn) or type(K) is not int:
        raise Unexpressible("vocabulary cannot express: a draft schedule but on a greedy request wrapper of a registered launch "
                            "with drafts and a draft layer, on one rank of the dense decoder, with no cached prefix, prefill "
                            "chunks, edge or launch batch")
    if sum(c + 1 for _, c in drafts) != T or any(not 0 <= c <= d <= K for d, c in drafts):
        raise Unexpressible(f"vocabulary cannot express: a draft schedule {list(drafts)} for {T} decode tokens at {K} drafts a row "
                            f"(each pass keeps at most its drafts and makes one token more, {T} in all)")
    hid = model.forward(rows, 0)
    toks = [model.sample(hid[-1], 0, LP - 1, None)]
    checks, p = [], LP
    for k, (d, c) in enumerate(drafts):
        first = Coll(I32, toks[-1].refs)
        cand: list[Coll] = []
        if d:
            proposed = model.call(layer.definition(K), first)
            cand = [Coll(I32, proposed.refs.slice(j, j + 1)) for j in range(d)]
        hid = model.forward([first, *cand], p)
        picked = [model.sample(hid[j], len(toks) + j, p + j, None) for j in range(d + 1)]
        if d:
            last = int(k == len(drafts) - 1)
            verdict = model.call(accept("greedy", d), Coll(Array(d, I32), concat([x.refs for x in cand])),
                                 Coll(Array(d + 1, I32), concat([x.refs for x in picked])))
            ok = model.call(bind(KeptCheck, KEPT=c, LAST=last), Coll(I32, verdict.refs.slice(d, d + 1)))
            checks.append((ok, f"the kept count (KeptCheck_v1 of Accept_v1): pass {k}'s accept keeps "
                               f"{'at least ' if last else ''}{c} of its {d} drafts"))
        toks += picked[:c + 1]
        p += c + 1
        for kh, vh in model.hist:
            del kh[p:], vh[p:]
    model.checks = checks
    return toks
