"""What every decoder of `build.decoder` writes with: the refusal (`Unexpressible`), the vocabulary's kinds (`kind`), the served
class, the circuit types of a shape, the tensor-parallel rank's view of the other ranks' partials (`_Peers`), the
batch a request's calls run in (`launch_batch`, `_Launch`) or a batch-invariant row's launch context (`launch_rows`, `launch_seqlen_q`) and the
token select over a row's whole-vocabulary logits (`select_token`), a wrapper's forwards over a decoder (`drive`), and a compile's
collector (`cyclic_gc_off`)."""

from __future__ import annotations

import functools
import gc
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from verity.circuits.refs import Coll, Refs, concat
from verity.circuits.types import Array, Type, Value


class Unexpressible(Exception):
    """φ asks for a structure the vocabulary cannot express.  The message names the fact precisely."""


def kind(role: str, **statics):
    """The vocabulary's kind for `role` at `statics` (the object the rules' `ctx.kind` returns), or `Unexpressible` naming why not."""
    from verity_vllm.program.frontend import targets as TG
    from verity_vllm.program.frontend.rules.vocab import PROFILE_B1_EAGER_V3 as VOC
    try:
        fn = VOC.kind(role, **statics)
    except TG.UnregisteredTarget as e:
        raise Unexpressible(f"numerics_unregistered: {e}") from None
    if fn is None:
        raise Unexpressible(f"vocabulary {VOC.name!r} has no kind for role {role!r} with {statics}")
    return fn


def served_class(vllm_class: str) -> str:
    """The class vLLM instantiates for a family (`FAMILY_FACTS[...]["vllm_class"]`): the one its registry maps the architecture to
    ("X (= Y)" is Y), else the named class."""
    alias = re.search(r"\(= (\w+)\)", vllm_class)
    return alias.group(1) if alias else re.match(r"\w+", vllm_class).group(0)


def _arr(shape: tuple[int, ...], w: int) -> Type:
    t: Type = Value(w)
    for n in reversed(shape):
        t = Array(int(n), t)
    return t


def _arr_shape(t: Type) -> tuple[int, ...]:
    shape = []
    while isinstance(t, Array):
        shape.append(t.n)
        t = t.elem
    return tuple(shape)


def _ptypes(spec) -> tuple:
    params, _ = spec.signature()
    return tuple(t for _, t in params)


def with_checks(model, out: Coll, B=None, stop: dict | None = None, sampled: Sequence | None = None) -> tuple[Coll, dict | None]:
    """`out`, then `derive`'s checks output where the model made any (`model.checks`: (one bit, what it checks) each, which the
    verifier requires to be 1), a stochastic request's seed rule follows them (`sampled`: its sampler's seed, then its
    `seed_rule.PARAMS` Inputs; `seed_rule.check`'s Calls next on the Builder `B`) and its length rule last (`stop`:
    `frontend.stop_rule.check` over `out`), and what `derive`'s provenance says of them: the Program returns the tuple (out,
    checks). Each is refused unless a declared relation computes it (`program.inputs.check`)."""
    from verity.circuits.types import Tuple
    from verity_vllm.program import inputs
    from verity_vllm.program.frontend import stop_rule as SR
    from verity_vllm.program.frontend import seed_rule as SEED
    checks = [*(getattr(model, "checks", None) or ()), *([] if sampled is None else [SEED.check(B, sampled[0], sampled[1:])]),
              *([] if stop is None else [SR.check(B, out, stop)])]
    if not checks:
        return out, None
    for c, what in checks:
        inputs.check(model.B, c, what)
    ret = Coll(Tuple(out.type, Array(len(checks), Value(1))), concat([out.refs, *[c.refs for c, _ in checks]]))
    return ret, {"output": 1, "accept": 1, "of": [w for _, w in checks]}


def edge_config(edge: tuple[str, ...]) -> dict:
    """What a request Program's SRC config, and its unit's key, say of its `edge` (`with_edge`): nothing where it has none, True
    where it returns the latent rows alone, else its groups."""
    return {"edge": True if tuple(edge) == ("prefix",) else list(edge)} if edge else {}


def with_edge(model, out: Coll, edge: tuple[str, ...]) -> Coll:
    """`out` (the samples, then any checks: `with_checks`), then for a request a Commit window leaves, every layer's rows so far of
    each history group of `edge` (`dsa._Model.edge_rows`), so a window's edge reads them at the fields after its samples and checks."""
    if not edge:
        return out
    from verity.circuits.types import Tuple, field_types
    rows = model.edge_rows(tuple(edge))
    head = field_types(out.type) if isinstance(out.type, Tuple) else (out.type,)
    return Coll(Tuple(*head, *(r.type for r in rows)), concat([out.refs, *[r.refs for r in rows]]))


@dataclass(frozen=True)
class _Stochastic:
    """The row's stochastic policy as the select reads it: the seed (and, for several requests, the `splits`) Inputs, the f32 bits of
    temperature and top_p, and S: "derived" (an Input entry per event) or the single request's constant split count."""
    seed: Coll
    splits: Coll | None
    S: Any
    temperature_bits: int
    top_p_bits: int


def request_passes(wrapper: str, LP: int | None, T: int | None, M: int, C: int, chunks: tuple[int, ...] | None,
                   dense: bool, edge: bool = False) -> tuple[bool, tuple[int, ...], list[int], list[bool]]:
    """A wrapper's forward passes, as `drive` runs them -> (resumed, its prefill chunks, each pass's rows, whether each samples): a
    `step`'s M rows; a `request`'s prefill of rows C..LP-1 after a history of C (in its `chunks` when it ran in several launches,
    only the last sampling) and its T decodes; or, off the `dense` decoder, one resumed at decode C >= LP, its decodes C..LP+T-1,
    and one whose rows are a Commit window's edge (`edge`, `build.build_program`'s), which may stop inside its prompt at LP + T
    (T < 0), sampling nothing."""
    E = int(LP) + int(T) if wrapper == "request" else 0
    if C and not 0 < C < (int(LP) if dense else E):
        raise Unexpressible(f"vocabulary cannot express: a history of {C} rows for a prompt of {LP}" +
                            (" (0 < C < LP)" if dense else f" and {T} decodes (0 < C < LP + T)"))
    if edge and (wrapper != "request" or dense) or wrapper == "request" and not (int(T) >= 0 or edge and E > max(C, 0)):
        raise Unexpressible(f"vocabulary cannot express: a {wrapper} to position {E} of a {LP}-token prompt" + (
            " with its rows a window's edge (a request's, on the sparse-attention decoder)" if edge else
            " (one that stops inside its prompt samples nothing: only a request a window leaves there has a Program, its rows the edge's)"))
    resumed = wrapper == "request" and C >= int(LP)
    chunks = tuple(int(c) for c in chunks or ())
    if chunks and (wrapper != "request" or resumed or min(chunks) < 1 or sum(chunks) != min(int(LP), E) - C):
        raise Unexpressible(f"vocabulary cannot express: prefill chunks {list(chunks)} of a {wrapper} (a request's chunks are its "
                            f"{max(min(int(LP or 0), E) - C, 0)} prefill rows in turn, each at least one)")
    chunks = chunks if len(chunks) > 1 else ()
    passes = [M] if wrapper == "step" else [1] * (E - C) if resumed else [*(chunks or [min(int(LP), E) - C]), *[1] * max(int(T), 0)]
    if wrapper == "request" and E < int(LP):
        return resumed, chunks, passes, [False] * len(passes)
    return resumed, chunks, passes, [False] * max(len(chunks) - 1, 0) + [True] * (len(passes) - max(len(chunks) - 1, 0))


def drive(model, wrapper: str, rows: list[Coll], *, M: int, pos0: int, LP: int | None, T: int | None, policy: _Stochastic | None,
          batch: list | None, C: int = 0, chunks: tuple[int, ...] = (), last: Coll | None = None) -> list[Coll]:
    """The wrapper's forwards over `model` -> its samples: a `step`'s M rows at pos0.. and its last row's sample, or a `request`'s
    positions C..LP+T-1 after a history of C: its prefill of rows C..LP-1 (in `chunks`, one forward a chunk, each its own launch)
    and the sample after it, then a decode at each position from LP on, each fed the previous sample; a history reaching LP or
    past it (C >= LP) resumes at decode C, fed `last`, the token there; a request that stops inside its prompt (LP + T < LP)
    samples nothing. The request's launch step k is its k-th forward."""
    if wrapper == "step":
        hid = model.forward(rows, pos0)
        return [model.sample(hid[-1], 0, pos0 + M - 1, policy)]
    LP, T = int(LP), int(T)
    toks: list[Coll] = []
    step = 0
    if C < LP:
        a = C
        for n in chunks or (min(LP, LP + T) - C,):
            if batch is not None:
                model.launch.step = step
            hid = model.forward(rows[a:a + n], a)
            a, step = a + n, step + 1
        if a == LP:
            toks.append(model.sample(hid[-1], 0, LP - 1, policy))
    for p in range(max(C, LP), LP + T):
        if batch is not None:
            model.launch.step = step
        hid = model.forward([Coll(Value(32), (toks[-1] if toks else last).refs)], p)
        toks.append(model.sample(hid[-1], len(toks), p, policy))
        step += 1
    return toks


def select_token(call, k, logits: Refs, V: int, target, event: int, pos: int, policy: _Stochastic | None) -> Coll:
    """The token of sampling event `event` from a row's V logits at position `pos` (`call(fn, *refs)` emits, `k(role, **statics)` is
    the kind): greedy `TokenSelect`, or `GumbelTopPTokenSelect[vllm]` over the derived `splits` entry or at the single request's
    constant S, its per-row constants in operand order (top_p, pos, temperature[, S])."""
    if policy is None:
        return call(k("token_select", V=V), logits)
    fn = (k("token_select_stochastic", V=V, CONSTRUCTION=target.effective_sampler_construction()) if policy.S == "derived"
          else k("token_select_stochastic_split", V=V, S=int(policy.S)))
    top_p = call(k("const", W=32, BITS=policy.top_p_bits))
    pos_c = call(k("const", W=64, BITS=int(pos) & 0xFFFFFFFFFFFFFFFF))
    temp = call(k("const", W=32, BITS=policy.temperature_bits))
    splits = policy.splits.refs.slice(event, event + 1) if policy.S == "derived" else call(k("const", W=32, BITS=int(policy.S))).refs
    return call(fn, logits, top_p.refs, policy.seed.refs, pos_c.refs, temp.refs, splits)


class _Peers:
    """A tensor-parallel rank's view of the other ranks' partials (`vllm_meta.WithPeers`): world - 1 peer Inputs per collective, in
    call order, the other ranks in rank order; `n` counts the Inputs the collectives have read."""

    def __init__(self, inputs: list[Coll], world: int, rank: int) -> None:
        self.inputs, self.world, self.rank, self.n = list(inputs), world, rank, 0

    def take(self, own: list) -> list[list]:
        """One collective's rows per rank, in rank order: this rank's `own` rows (refs) and each of the next world - 1 peer Inputs'."""
        peers = self.inputs[self.n:self.n + self.world - 1]
        self.n += self.world - 1
        N = own[0].count
        others = [[p.refs.slice(i * N, (i + 1) * N) for i in range(len(own))] for p in peers]
        return others[:self.rank] + [list(own)] + others[self.rank:]

    def rows(self, own: list) -> list:
        """Each row's operand of a per-row collective: that row of every rank, rank order."""
        ranks = self.take(own)
        return [concat([r[i] for r in ranks]) for i in range(len(own))]

    def whole(self, own: list):
        """The operand of one collective over all rows: every rank's rows, rank-major (`collectives.AllReduceRule`'s parts)."""
        return concat([concat(r) for r in self.take(own)])


class _Launch:
    """The batch a request's calls run in (`pipeline.launch_context.request_launch_batch`, parsed as `targets.parse_launch_context`
    parses the Build's `--launch-context`): per request step, every sequence the engine step schedules as (rows, start).  `step` is
    the request step being written; without a batch a call is its own launch.  `capture_sizes`: the breakable CUDA graphs'
    (`graph_capture_sizes`), whose captured kernels run at the step's padded tokens; `max_num_seqs` the engine's, up to which a uniform
    decode step replays a FULL graph (`targets.graph_launch`)."""

    def __init__(self, batch: list | None = None, capture_sizes: tuple[int, ...] = (), max_num_seqs: int = 0) -> None:
        from verity_vllm.program.frontend import targets as TG
        self.ctx, self.step = TG.parse_launch_context({"batch": batch}) if batch else None, 0
        self.capture_sizes, self.max_num_seqs = tuple(capture_sizes), int(max_num_seqs)

    def with_batch(self, batch: list) -> _Launch:
        """This launch's graphs over the request's batch."""
        return _Launch(batch, self.capture_sizes, self.max_num_seqs)

    def graph(self, own: int) -> tuple[str, int]:
        """The CUDA-graph mode (FULL, PIECEWISE or NONE) of the step `own` rows run in, and the tokens its forward runs at
        (`targets.launch_graph`)."""
        from verity_vllm.program.frontend import targets as TG
        try:
            return TG.launch_graph(own, self.step, self.ctx, self.capture_sizes, self.max_num_seqs)
        except ValueError as e:
            raise Unexpressible(f"request step {self.step}: {e}") from None

    def tokens(self, own: int, sampled: bool = False) -> int:
        """The rows of the launch `own` rows run in (`targets.launch_tokens`); after the logits selection (`sampled`), its sequences."""
        from verity_vllm.program.frontend import targets as TG
        try:
            return TG.launch_tokens(own, self.step, self.ctx, sampled=sampled, capture_sizes=self.capture_sizes)
        except ValueError as e:
            raise Unexpressible(f"request step {self.step}: {e}") from None

    def keys(self, own: list[int], without: list[tuple[int, int]] = ()) -> list[int]:
        """Every query row's key count in the attention launch (`targets.launch_keys`), `own`'s without a batch; `without`: sequences
        (rows, start) of the step whose rows another kernel runs (`prefills`' dense-MHA prefills), left out.  A FULL graph (a uniform
        decode, so no prefills) captures the attention at the padded batch, its padding rows without keys (their requests' seq_lens
        are 0, so the indexer's top-k leaves them -1); PIECEWISE graphs and an uncaptured step run it on the real batch."""
        from collections import Counter
        from verity_vllm.program.frontend import targets as TG
        seqs = TG.launch_sequences(self.step, self.ctx)
        if seqs is None:
            return list(own)
        drop, keys = Counter((int(n), int(p)) for n, p in without), []
        for n, p in seqs:
            if drop[(n, p)]:
                drop[(n, p)] -= 1
            else:
                keys += [p + i + 1 for i in range(n)]
        if Counter(own) - Counter(keys):
            raise Unexpressible(f"request step {self.step}: the launch batch's key counts {keys} do not hold this request's {own}")
        mode, padded = self.graph(len(own))
        return keys + [0] * (padded - len(keys)) if mode == "FULL" else keys

    def prefills(self, own: tuple[int, int], threshold: int) -> list[tuple[int, int]]:
        """The step's prefills, (rows, start) each, as vLLM's MLA metadata builder counts them: every sequence from the first of
        more than `threshold` rows (`split_decodes_and_prefills`) in the order `reorder_batch_to_split_decodes_and_prefills` gives
        the batch, the extends (start > 0) of more than `threshold` rows and then the first chunks (start 0).  So with such an extend
        every first chunk is a prefill, and without one the first chunks of more than `threshold` rows are; a first chunk of at most
        `threshold` rows beside a longer one is refused, as the persistent batch's order, which the batch does not record, decides
        whether it is one.  `own`: this request's (rows, start), the step's one sequence without a batch."""
        from verity_vllm.program.frontend import targets as TG
        try:
            seqs = TG.launch_sequences(self.step, self.ctx)
        except ValueError as e:
            raise Unexpressible(f"request step {self.step}: {e}") from None
        seqs = [own] if seqs is None else [(int(n), int(p)) for n, p in seqs]
        if own not in seqs:
            raise Unexpressible(f"request step {self.step}: the launch batch's sequences {seqs} do not hold this request's {own} "
                                f"(rows, start)")
        extends = [s for s in seqs if s[1] > 0 and s[0] > threshold]
        first = [s for s in seqs if s[1] == 0]
        if extends:
            return extends + first
        short, long = [s for s in first if s[0] <= threshold], [s for s in first if s[0] > threshold]
        if short and long:
            raise Unexpressible(f"vocabulary cannot express: request step {self.step} schedules first chunks of {[n for n, _ in short]} "
                                f"rows beside ones of {[n for n, _ in long]} rows, above the decode threshold {threshold}: whether vLLM's "
                                f"metadata builder counts the short ones prefills follows the persistent batch's order")
        return long


def launch_batch(phi, LP: int, T: int) -> list | None:
    """The batch each call of φ's (LP, T) request runs in where its kernel choices read the launch (`launch_context.batch_dispatch`:
    a target without batch invariance co-scheduling several requests), as the Build is given it (`launch_context.shape_contexts`);
    None for a request that is its own launch.  Requests of the shape co-scheduled differently have no one Program."""
    from verity_vllm.pipeline import launch_context as LC
    w = {"requests": phi.requests, "engine": {"target": {"batch_invariant": phi.target_profile().batch_invariant}}}
    if not LC.batch_dispatch(w):
        return None
    batch = LC.shape_contexts(w).get((int(LP), int(T)))
    if batch is None or batch == "conflict":
        raise Unexpressible(f"vocabulary cannot express: the launch batch of request shape LP {LP}, T {T}: "
                            f"{'its requests are co-scheduled differently' if batch else 'no request of the workload has it'}")
    return batch


def in_launch(model, batch: list | None, model_class: str, chunks: tuple[int, ...] = (), T: int = 0, note: str = "") -> None:
    """`model`'s calls run in `batch` (`_Launch.with_batch`) where its request has one; `Unexpressible` for a decoder whose kernel
    choices read its own rows only (`reads_launch`), or for prefill `chunks` that aren't one a step of `batch` before its T decodes
    (none for a request a window leaves mid-prefill, T < 0)."""
    if batch is not None:
        if not getattr(model, "reads_launch", False):
            raise Unexpressible(f"vocabulary cannot express: a request run in a launch batch (no batch invariance{note}), whose kernel "
                                f"choices the {model_class} decoder reads at its own rows")
        model.launch = model.launch.with_batch(batch)
    steps = len(chunks) + max(int(T or 0), 0)
    if chunks and (batch is None or len(batch) != steps):
        raise Unexpressible(f"vocabulary cannot express: a prefill in {len(chunks)} chunks " + (
            f"at a launch batch of {len(batch)} steps, not one a chunk and one a decode ({steps})" if batch is not None else
            "without a launch batch (on a batch-invariant target a chunked prefill is the request wrapper's Program)"))


def graph_capture_sizes(phi) -> tuple[int, ...]:
    """The capture sizes of φ's breakable CUDA graphs (`TargetProfile.graph_capture_sizes`); () for an execution without them."""
    return phi.target_profile().graph_capture_sizes()


def launch_rows(phi, LP: int, T: int, C: int = 0) -> dict | None:
    """The launch context a batch-invariant row's (LP, T[, C]) request runs under, as the Build is given it (`launch_context.spec_for`:
    per request step, the launch's max_seqlen_q, every co-scheduled sequence's rows counted), parsed as `targets.parse_launch_context`
    parses `--launch-context`; None without the workload's requests, or for a shape none of them has (the Build's own rows)."""
    if not phi.requests:
        return None
    from verity_vllm.pipeline import launch_context as LC
    from verity_vllm.program.frontend import targets as TG
    from verity_vllm.program.prefix_plan import shape
    v = LC.shape_contexts({"requests": phi.requests, "engine": {"target": {"batch_invariant": True}}}).get(shape(LP, T, C))
    return None if v is None else TG.parse_launch_context(v)


def launch_seqlen_q(ctx: dict, own_rows: int, step: int, tp, accepted: bool) -> int:
    """The max_seqlen_q a request's attention at request `step` reads under its launch context (`targets.launch_seqlen_q`, as the
    Build's attention binding reads it): FA3's tile has no one value for a shape whose requests run in different launches ('conflict');
    FA2's reads the request's own rows."""
    from verity_vllm.program.frontend import targets as TG
    try:
        return TG.launch_seqlen_q(own_rows, step, ctx)[0]
    except TG.LaunchContextConflict as e:
        if tp.effective_flash_attn_version() == 3 and not accepted:
            raise Unexpressible(f"vocabulary cannot express: request step {step}: FA3's tile (kBlockN) is not one value -- {e}") from None
        return own_rows
    except TG.LaunchContextInvalid as e:
        raise Unexpressible(f"vocabulary cannot express: request step {step}: {e}") from None


def cyclic_gc_off(f):
    """`f` with the cyclic collector off while it runs: a compile frees its intermediates by reference counting, and the collector's
    passes over the bodies it builds cost a quarter of the compile."""
    @functools.wraps(f)
    def run(*args, **kwargs):
        on = gc.isenabled()
        gc.disable()
        try:
            return f(*args, **kwargs)
        finally:
            if on:
                gc.enable()
    return run
