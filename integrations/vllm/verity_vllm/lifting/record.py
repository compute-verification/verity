"""The record half of stage DENSE B1 on what a record directory holds ON DISK — **lifted-padded**, lane `pad` (R17 BRIEF §7; R16 BRIEF §7B).

R17 (LIFTING-SPEC v0.5): R0 is R17-1's rule — the Build's DECLARED lag (`workload_program.json`) == the record's ATTESTED lag (RunHeader),
through GP-01's one function; R4c reports the TRUE lifted widths (R17-6(b)); every report carries `rules_version = lifted-v1` (A5).

Given the Build's derived request descriptor (`build_request/descriptor.json.gz`), the Commit's binding map (`binding_map_p<k>.json`), the
run rows (`runs.jsonl`) and the committed transcript of the request (the sampled token ids, in event order, with its provenance named), this
module checks the SPEC §5–§7 statements a fixed padded Program makes about a real early-EOS record, each as a named PASS/FAIL, and writes
the ⊥ assignment of the declared-but-unexecuted population (the values a prover commits for steps `≥ a_r`, before any challenge):

  R1  population    the derived body has exactly `cap = max_tokens_r` sampling events (a served-shape derivation `build_request_LP<lp>_T<served>` is
                    refused by name, F-pad-15); `lift_request_component` builds `LiftedRequest_v2{…, EOS, GUARD=in, LAG}` (v2 = padded-v1);
                    the guard adds exactly `2·(cap−1) + |E| + 1` gates; PAD-03's deletion rule recovers the unlifted rows.
  R2  activity      `a_r` from the committed transcript by SPEC §5 (`activity_prefix`); the record served exactly its Active prefix (`served_count`,
                    `check_served_prefix`): stopped at the first EOS, or ran to the cap with no EOS.
  R3  executed set  the binding map commits identities for the request at EXACTLY the engine steps the declared guard candidate predicts:
                    INPUT guard (a, implemented): `arrive + [0, a_r)`; OUTPUT guard (b): `arrive + [0, a_r]` (the engine runs the step after the
                    EOS on the EOS token and discards its sample — measured on vLLM bi-eager, exec M-0140: 12 events for an EOS at event 10).
                    Both predictions are reported with the observed set (PAD-05); a committed step outside the declared set is "declared
                    inactivity executed", a missing one "declared activity not committed".
  R4  assignment    the node-level activity closure of the lifted body under the transcript (guards by their spec functions, strict elsewhere):
                    every node of events `< a_r` is Active; from `a_r` on every node in the token cone is ⊥; the closure's Active event set is
                    exactly `[0, a_r)` (SPEC §5's formula and the Program's own semantics agree on this record); the per-event ⊥ leaf / word counts
                    are written out (the population Commit binds as `enc(⊥)`).  Node-level is exact under the DENSE rule (an output is ⊥ iff
                    any operand leaf is ⊥); a node with MIXED operand tags outside a ⊥ step is a named FAIL ("gate-level closure required"),
                    never counted.
  R5  interface     lifted `out[s] = 0‖y_s` for `s < a_r` and `enc(⊥)` for `s ≥ a_r` == `assign_inactive(record)`.
  R6  values        row-level bit-exactness of the Active strata needs the retained committed store (in-process on the pod): NOT EVALUATED
                    here — reported as such, never inferred from the active-reference C2 verdict (which is named separately, as context).

Nothing here reads a served length into the Program: the Program is the descriptor's; the transcript only drives the guards.
"""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from verity.ir.codec import _spec_id
from verity.ir.defs import REGISTRY
from verity.ir.refs import Affine, Concat, Explicit, Refs, Strided
from verity_vllm.registry import lifted as L

# rev S-r17-29 n2: where the EOS set the tool checks against CAME FROM.  Of record = the Program's pinned constant as the Build declared it
# (`workload_program.json stop_rule.eos_token_ids`); a check-time argument (`--eos`) is NOT of record and makes R9 `None` by name.
EOS_SOURCE_OF_RECORD = "workload_program.json stop_rule.eos_token_ids (the Program's pinned constant, of record)"
EOS_SOURCE_GIVEN = "--eos (a check-time argument: NOT of record -- R9 is None by name; rev S-r17-29 n2 / F-pm-24)"
# The derived body's vocabulary (frontend/rules/vocab.py: b1 + fp8 + moe + ref_prims) registers by IMPORT; a record tool that decodes a
# descriptor must have the whole vocabulary registered or the decode fails by name on the first primitive of a family it never imported
# (found on the OLMoE row: `primitive F32Sub_v1 is not in the registry` -- the router softmax's RN subtract, registry/ref_prims).
from verity_vllm.registry import b1 as _b1, dense as _dense, fp8 as _fp8, moe as _moe, ref_prims as _ref_prims  # noqa: F401
try:  # the padded MoE vocabulary (moe's registry/moe_pad.py: MoeFfnFixedL_v1 + its hand lift; lane/r17-moe-pad, int-8 candidate) -- by import,
    from verity_vllm.registry import moe_pad as _moe_pad  # noqa: F401  # like the others; absent on a tree without moe's module (this lane), present on int-8
except ImportError:  # pragma: no cover
    _moe_pad = None
from verity.ir.types import Value
from verity_vllm.ir import workload as W

LABEL = L.LABEL
SCHEMA = "verity/pad/lifted-record-check/v1"
A, B = "A", "⊥"  # node tags: Active / bot


def node_reads(r: Refs) -> set[int]:
    """Body node indices a refs object reads — at RUN level (no per-leaf iteration: a weights parameter run is 10^8 leaves)."""
    if isinstance(r, (Affine, Strided)):
        return {r.idx} if r.space == "n" else set()
    if isinstance(r, Concat):
        out: set[int] = set()
        for p in r.parts:
            out |= node_reads(p)
        return out
    if isinstance(r, Explicit):
        return {idx for space, idx, _ in r if space == "n"}
    raise TypeError(type(r))


def _leaf_widths(t) -> list[int]:
    from verity.ir.types import Array, Record, Tuple
    if isinstance(t, Value):
        return [t.w]
    if isinstance(t, Array):
        return _leaf_widths(t.elem) * t.n
    if isinstance(t, Tuple):
        return [w for f in t.fields for w in _leaf_widths(f)]
    if isinstance(t, Record):
        return [w for _, f in t.fields for w in _leaf_widths(f)]
    raise TypeError(t)


@dataclass
class Closure:
    tags: list[str]                       # per lifted body node
    event_tag: list[str]                  # per event s: tag of its boundary (token) node
    event_token: list[int | None]         # per event: the committed token (Active) or None (⊥)
    guard_live: list[int]                 # live_s for s in 1..cap-1 (index s-1)
    step_of_node: list[int]               # lifted node → event segment it belongs to
    mixed_nodes: list[tuple[int, str, int]] = field(default_factory=list)  # (node, spec id, step) with mixed operand tags
    bot_leaves_by_event: dict[int, int] = field(default_factory=dict)
    bot_words_by_event: dict[int, int] = field(default_factory=dict)
    active_nodes_by_event: dict[int, int] = field(default_factory=dict)
    bot_nodes_by_event: dict[int, int] = field(default_factory=dict)
    node_rows: list[dict] = field(default_factory=list)  # the assignment listing: (node, spec, event, tag, leaves)
    guard_bot_leaves: int = 0  # the guard's own bot leaves (tok_in_s for s >= a_r, bot32): not part of any step's population
    leaves_by_event: dict[int, int] = field(default_factory=dict)       # EVERY leaf of the event's nodes (Active or bot)
    words_by_event: dict[int, int] = field(default_factory=dict)        # Σ leaf_words(w) over them: the size a lifted binding map must carry (F-pad-06(b))
    unlifted_words_by_event: dict[int, int] = field(default_factory=dict)  # one word per leaf: what a legacy one-word reader would expect
    lifted_bits_by_event: dict[int, int] = field(default_factory=dict)  # R17-6(b): Σ (w+1) over the event's leaves — the TRUE lifted width, in bits
    payload_bits_by_event: dict[int, int] = field(default_factory=dict)  # Σ w over them (information: what the unlifted Program's width was)
    node_width_max: tuple[int, int, str] = (0, 0, "")                   # (lifted bits, payload bits, spec id) of the widest single node output
    # Continuation_v1 (R17-32): the controller bits of the Program under this transcript, per slot t = 1..cap (index t-1)
    emit: list[int] = field(default_factory=list)                       # e_t from `slot=<t>/emit` (t=1: the constant 1)
    is_eos: list[int] = field(default_factory=list)                     # b_t from `slot=<t>/is_eos`
    designated_out: list[int | None] = field(default_factory=list)      # o_t from `slot=<t>/out`: the token when e_t = 1, None (bot) otherwise


def _slot_of_name(name: str) -> int | None:
    """`slot=<t>/…` -> the slot's event index `t - 1`; `s=<s>/…` -> `s`; anything else None."""
    if name.startswith("slot="):
        return int(name[5:].split("/")[0]) - 1
    if name.startswith("s="):
        return int(name[2:].split("/")[0])
    return None


def activity_closure(lifted, transcript: Sequence[int], eos: Sequence[int]) -> Closure:
    """Node-level tags of the lifted component body driven by the committed transcript.  Guards evaluate by their spec functions on the
    boundary tokens (`LStepLive`: Active non-EOS → 1, EOS → 0, ⊥ → 0; `LSelect`: live ? token : ⊥); the boundary node of an Active event
    `s` carries `transcript[s]` (which must exist — an Active event with no committed token is a record gap); every other node is strict
    under the DENSE rule (⊥ iff any operand node is ⊥)."""
    body = lifted.body
    nidx, bounds, guards = list(lifted.node_map), list(lifted.boundaries), set(lifted.guard_nodes)
    cap = len(bounds)
    eos_set = {int(e) for e in eos}
    lb = [nidx[b] for b in bounds]                     # lifted node of event s's boundary
    event_of_boundary = {k: s for s, k in enumerate(lb)}
    # event segment of each lifted node: original node k belongs to the segment ending at the first boundary >= k; guard nodes
    # `s=<s>/…` belong to segment s (they precede its first reader); head guards (eos consts, bot32) to segment 0
    step_of_node = [0] * len(body.nodes)
    seg = 0
    for k in range(len(body.nodes)):
        n = body.nodes[k]
        if k in guards:
            name = n.name or ""
            step_of_node[k] = _slot_of_name(name) or 0
            continue
        step_of_node[k] = seg
        if k in event_of_boundary:
            seg += 1
    tags: list[str] = [A] * len(body.nodes)
    event_tag: list[str] = [A] * cap
    event_token: list[int | None] = [None] * cap
    guard_live: list[int] = []
    emit: list[int] = []
    is_eos: list[int] = []
    designated: list[int | None] = []
    mixed: list[tuple[int, str, int]] = []
    tok_val: dict[int, int | None] = {}  # boundary/tok_in node → its Active token value (None = ⊥); controller nodes → their bit
    for k, n in enumerate(body.nodes):
        sid = _spec_id(n.fn)
        fam = getattr(n.fn, "family", None)
        reads = set()
        for a in n.args:
            reads |= node_reads(a.refs)
        in_tags = {tags[j] for j in reads}
        if k in guards:
            name = n.name or ""
            if sid.startswith("LBot"):
                tags[k] = B  # the named inactive constant (family Const, value enc(bot))
            elif name == "slot=1/emit":
                tags[k] = A  # e_1 = 1: the constant Active(1)
                tok_val[k] = 1
                emit.append(1)
            elif fam == "Const":
                tags[k] = A
            # Continuation_v1 (R17-32) controller nodes, evaluated by their spec functions' meaning (the registered evaluators replay them in R9):
            elif name.endswith("/is_eos"):
                (tok,) = [j for j in reads if j in event_of_boundary]
                v = tok_val[tok]
                b = 1 if (v is not None and v in eos_set) else 0  # IsEOS total: a bot token is not EOS
                tags[k] = A
                tok_val[k] = b
                is_eos.append(b)
            elif name.endswith("/emit"):
                prev_e = [j for j in reads if (body.nodes[j].name or "").endswith("/emit")]
                prev_b = [j for j in reads if (body.nodes[j].name or "").endswith("/is_eos")]
                e = 1 if (len(prev_e) == 1 and len(prev_b) == 1 and tok_val[prev_e[0]] == 1 and tok_val[prev_b[0]] == 0) else 0
                tags[k] = A
                tok_val[k] = e
                emit.append(e)
            elif name.endswith("/out"):
                (tok,) = [j for j in reads if j in event_of_boundary]
                (e_k,) = [j for j in reads if (body.nodes[j].name or "").endswith("/emit")]
                if tok_val[e_k] == 1 and tok_val[tok] is not None:
                    tags[k] = A
                    tok_val[k] = tok_val[tok]
                    designated.append(tok_val[tok])
                else:
                    tags[k] = B
                    tok_val[k] = None
                    designated.append(None)
            elif name.endswith("/live"):
                (prev,) = [j for j in reads if j in event_of_boundary or (j in guards and (body.nodes[j].name or "").endswith("/tok_in"))]
                v = tok_val[prev]
                live = 0 if v is None else (0 if v in eos_set else 1)
                guard_live.append(live)
                tags[k] = A
                tok_val[k] = live
            elif name.endswith("/tok_in"):
                live_k = k - 1
                prev = [j for j in reads if j in event_of_boundary][0]
                if tok_val[live_k] == 1:
                    tags[k] = A
                    tok_val[k] = tok_val[prev]
                else:
                    tags[k] = B
                    tok_val[k] = None
            else:
                raise ValueError(f"unexpected guard node {k} {sid} {name!r}")
            continue
        # an original node: dense strict rule
        if len(in_tags) > 1:
            mixed.append((k, sid, step_of_node[k]))
        tags[k] = B if B in in_tags else A
        if k in event_of_boundary:
            s = event_of_boundary[k]
            event_tag[s] = tags[k]
            if tags[k] == A:
                if s >= len(transcript):
                    raise L.ServedPrefixFailure("closure", len(transcript), s + 1, f"event {s} is Active in the Program but the record has only {len(transcript)} committed tokens")
                event_token[s] = int(transcript[s])
                tok_val[k] = event_token[s]
            else:
                tok_val[k] = None
    cl = Closure(tags, event_tag, event_token, guard_live, step_of_node, mixed, emit=emit, is_eos=is_eos, designated_out=designated)
    for k, n in enumerate(body.nodes):
        s = step_of_node[k]
        widths = _leaf_widths(n.out)  # lifted widths (w+1)
        sid = _spec_id(n.fn)
        if k not in guards:
            cl.leaves_by_event[s] = cl.leaves_by_event.get(s, 0) + len(widths)
            cl.words_by_event[s] = cl.words_by_event.get(s, 0) + sum(L.leaf_words(w - 1) for w in widths)
            cl.unlifted_words_by_event[s] = cl.unlifted_words_by_event.get(s, 0) + len(widths)
            cl.lifted_bits_by_event[s] = cl.lifted_bits_by_event.get(s, 0) + sum(widths)
            cl.payload_bits_by_event[s] = cl.payload_bits_by_event.get(s, 0) + sum(w - 1 for w in widths)
            if sum(widths) > cl.node_width_max[0]:
                cl.node_width_max = (sum(widths), sum(w - 1 for w in widths), sid)
        if k in guards:  # the guard's own gates are reported apart from the population (they are the Program's, not a step's values)
            if tags[k] == B:
                cl.guard_bot_leaves += len(widths)
        elif tags[k] == B:
            cl.bot_nodes_by_event[s] = cl.bot_nodes_by_event.get(s, 0) + 1
            cl.bot_leaves_by_event[s] = cl.bot_leaves_by_event.get(s, 0) + len(widths)
            cl.bot_words_by_event[s] = cl.bot_words_by_event.get(s, 0) + sum(L.leaf_words(w - 1) for w in widths)
        else:
            cl.active_nodes_by_event[s] = cl.active_nodes_by_event.get(s, 0) + 1
        cl.node_rows.append({"node": k, "spec": sid, "event": s, "tag": tags[k], "leaves": len(widths), "guard": k in guards})
    return cl


def enable_closure(lifted, cl: Closure, *, lag: int) -> dict:
    """R4' (LIFTING-SPEC v0.6 B9 / A9.3, the record-side meaning under `padded-v1`): a node's word is ⊥ iff the call that OWNS it is disabled.
    For the Continuation_v1 request body the owner of every original node of event segment `s >= 1` is slot s's step call, whose enable is
    `run_s` (= the guard's `live` of the previous event; `run_1 = 1`, the prefill segment 0 is always enabled; under LAG the unguarded steps
    `1..lag` are enabled by construction).  Compared node for node with the PROPAGATION closure (`activity_closure`, v0.5.2 A3 -- the
    `lifted-v1` meaning): (i) `enabled_but_bot` -- a node the enable says is Active that propagation makes ⊥ -- must be EMPTY (a ⊥ word reached
    an enabled gate along a taken path: a fault, A9.4's second theorem; the enables are monotone so this cannot happen honestly); (ii)
    `disabled_but_active` -- nodes the enable makes ⊥ that propagation left Active: the class of constants and weights-only gates INSIDE a
    disabled step (per-position constants, a norm's eps broadcast) -- reported by spec id with counts: exactly the words that change meaning
    between the two rules versions, and the extra ⊥ words a padded Commit materializes beyond the propagation count (A14.3 (g), projected)."""
    body = lifted.body
    guards = set(lifted.guard_nodes)
    cap = len(lifted.boundaries)
    n_unguarded = 1 + L.check_lag(lag)  # segments 0..lag: enabled by construction (r_1 = 1; the lag steps have no guard)
    run = [1] * cap
    for s in range(n_unguarded, cap):
        run[s] = int(cl.guard_live[s - n_unguarded]) if s - n_unguarded < len(cl.guard_live) else 0
    enabled_but_bot: list[dict] = []
    disabled_but_active: dict[str, int] = {}
    extra_bot_words_by_event: dict[int, int] = {}
    extra_bot_leaves_by_event: dict[int, int] = {}
    bot_words_by_event: dict[int, int] = {}
    tags_e: list[str] = list(cl.tags)
    for k, n in enumerate(body.nodes):
        if k in guards:
            continue
        s_ = cl.step_of_node[k]
        te = A if run[s_] == 1 else B
        tags_e[k] = te
        widths = _leaf_widths(n.out)
        words = sum(L.leaf_words(w - 1) for w in widths)
        if te == B:
            bot_words_by_event[s_] = bot_words_by_event.get(s_, 0) + words
        if te != cl.tags[k]:
            sid = _spec_id(n.fn)
            if te == A:
                enabled_but_bot.append({"node": k, "spec": sid[:80], "event": s_})
            else:
                disabled_but_active[sid.split("{")[0]] = disabled_but_active.get(sid.split("{")[0], 0) + 1
                extra_bot_words_by_event[s_] = extra_bot_words_by_event.get(s_, 0) + words
                extra_bot_leaves_by_event[s_] = extra_bot_leaves_by_event.get(s_, 0) + len(widths)
    monotone = all(run[s] >= run[s + 1] for s in range(cap - 1))
    return {"run": run, "tags": tags_e, "enabled_but_bot": enabled_but_bot, "disabled_but_active_by_spec": disabled_but_active,
            "disabled_but_active_nodes": sum(disabled_but_active.values()), "extra_bot_words_by_event": extra_bot_words_by_event,
            "extra_bot_leaves_by_event": extra_bot_leaves_by_event, "bot_words_by_event": bot_words_by_event, "enables_monotone": monotone,
            "ok": not enabled_but_bot and monotone}


def _bits(w_lifted: int, x: int) -> str:
    """The raw committed bit string of a lifted `w_lifted`-bit word, tag first (`0‖v` / `1‖0^w`), as SPEC A3.4 commits it."""
    return format(int(x), f"0{int(w_lifted)}b")


def three_vus(lifted, cl: Closure, transcript: Sequence[int], eos: Sequence[int], a_r: int, cap: int, lag: int) -> dict:
    """SPEC v0.5.1 A7 (c): one VU of each kind from THIS record, its words LISTED (not counted): an active step (every word `0‖v`), the
    transition `s = a_r + LAG − 1 → a_r + LAG` (`tok_in_{a_r} = Active(EOS)` fed to the placeholder step; `live_{a_r+LAG} = LStepLive(EOS) =
    Active(0)`; `tok_in_{a_r+LAG} = LSelect(Active(0), ·, ⊥) = ⊥`; every consumer `⊥`), and an inactive step (every word `enc(⊥)`, every
    relation `⊥ → ⊥`, the guard `Active(0)`).

    The guard chain is REPLAYED here word for word by the registered lifted evaluators (`LStepLive<E>_v1.evaluate`, `LSelect32_v1.evaluate`)
    on the record's committed transcript -- an evaluation independent of `activity_closure`'s tag propagation, and checked against it
    (`guard_replay_agrees`).  Body words: an inactive node's leaves are the CONSTANT `enc(⊥) = 1‖0^w` (A3.4: listed per node, one literal
    per distinct width, since the word depends on `w` alone); an active node's leaves are `0‖v` with `v` the committed raw payload -- the
    Commit's leaf, opened at the original root on the pod (values are not on disk here; the FORM is listed, the value is the Commit's).
    A length stop (`a_r = cap`) has no transition and no inactive step: those entries are None, `ok` is the active VU's alone."""
    body = lifted.body
    n_exec = L.expected_served(a_r, cap, lag)
    eos_ids = [int(e) for e in eos]
    step_live, sel = L.l_step_live(len(eos_ids)), L.l_select(32)
    eos_words = [L.enc_active(32, e) for e in eos_ids]
    bot32 = L.enc_bot(32)
    # --- the interface and guard chain, replayed by the evaluators (LServe's letter: LAG=0 gates on out[s-1], LAG=1 on tok_in_{s-1}) ---
    out: list[int] = []
    tok_in: list[int | None] = [None] * cap
    live: list[int | None] = [None] * cap
    for s in range(cap):
        if s == 0:
            out.append(L.enc_active(32, int(transcript[0])) if cl.event_tag[0] == A else bot32)
            continue
        if s <= lag:
            tok_in[s] = out[s - 1]
        else:
            g = out[s - 1] if lag == 0 else tok_in[s - 1]
            live[s] = int(step_live.evaluate(g, *eos_words))
            tok_in[s] = int(sel.evaluate(live[s], out[s - 1], bot32))
        # the step's output word: the committed sample when the step ran (its input Active), ⊥ otherwise (strictness through the body)
        ran = tok_in[s] is not None and L.is_active(32, tok_in[s])
        if ran and s < len(transcript):
            out.append(L.enc_active(32, int(transcript[s])))
        elif ran:
            out.append(None)  # an Active step with no committed token: a record gap (R2 reports it; listed as None here)
        else:
            out.append(bot32)
    replay_live = [live[s] for s in range(1 + lag, cap)]
    guard_replay_agrees = (replay_live == list(cl.guard_live)) and all(
        (cl.event_tag[s] == A) == (out[s] is not None and L.is_active(32, out[s])) for s in range(cap))
    # --- the Continuation_v1 controller (R17-32), replayed by ITS registered evaluators on the same words: slot t = s + 1; b_t = LIsEOS(out[s]);
    # o_t = LSelect32(e_t, out[s], bot); e_{t+1} = LEmitNext[H=1](e_t, b_t).  Checked against the closure's tags (independent of its propagation).
    is_eos_p, emit_next_p = L.l_is_eos(len(eos_ids)), L.l_emit_next(1)
    c_emit: list[int | None] = [1] + [None] * (cap - 1)
    c_is_eos: list[int | None] = [None] * cap
    c_out: list[int | None] = [None] * cap
    for s in range(cap):
        y = out[s] if out[s] is not None else None
        if y is None:  # a record gap (an Active step with no committed token): the controller words of this slot are unknown, listed as None
            if s + 1 < cap:
                c_emit[s + 1] = None
            continue
        c_is_eos[s] = int(is_eos_p.evaluate(y, *eos_words))
        c_out[s] = None if c_emit[s] is None else int(sel.evaluate(c_emit[s], y, bot32))
        if s + 1 < cap:
            c_emit[s + 1] = None if c_emit[s] is None else int(emit_next_p.evaluate(c_emit[s], c_is_eos[s]))
    controller_replay_agrees = (
        [None if e is None else int(e) for e in c_emit] == [None if e is None else int(e) for e in cl.emit]
        and [None if b is None else int(b) for b in c_is_eos] == [None if b is None else int(b) for b in cl.is_eos]
        and [None if o is None or L.is_bot(32, o) else L.payload(32, o) for o in c_out] == list(cl.designated_out))

    def w_(s: int, x: int | None) -> str | None:
        return None if x is None else _bits(33, x)

    def controller_words(s: int) -> dict:
        t = s + 1
        e, b, o = c_emit[s], c_is_eos[s], c_out[s]
        r = 1 if s <= lag else live[s]
        return {"slot": t, "emit": (None if e is None else _bits(2, e)), "run": (None if r is None else _bits(2, r)), "is_eos": (None if b is None else _bits(2, b)),
                "pair_emit_run": (None if e is None or r is None else f"({e}, {r})"),
                "kind": (None if e is None or r is None else ("active" if e and r else "lag" if r else "inactive")),
                "out": w_(s, o), "out_is": (None if o is None else ("⊥" if L.is_bot(32, o) else f"Active({L.payload(32, o)})")),
                "relation": (f"is_eos_{t} = LIsEOS{len(eos_ids)}_v1(out[{s}], eos_set); out_{t} = LSelect32_v1(emit_{t}, out[{s}], LBot32)"
                             + (f"; emit_{t + 1} = LEmitNext[H=1]_v1(emit_{t}, is_eos_{t})" if t < cap else "; no successor slot (cap)"))}

    def guard_words(s: int) -> dict:
        if s == 0:
            return {"step": s, "guarded": False, "note": "event 0 (prefill): unguarded, unconditionally live", "out": w_(s, out[s])}
        if s <= lag:
            return {"step": s, "guarded": False, "note": f"head step 1..LAG: unguarded, tok_in_{s} = out[{s - 1}]", "tok_in": w_(s, tok_in[s]), "out": w_(s, out[s])}
        g = out[s - 1] if lag == 0 else tok_in[s - 1]
        return {"step": s, "guarded": True, "gating_token": (f"out[{s - 1}]" if lag == 0 else f"tok_in[{s - 1}]"), "g": w_(s, g),
                "live": _bits(2, live[s]), "live_is": ("Active(1)" if live[s] == 1 else "Active(0)" if live[s] == 0 else "INV_1"),
                "tok_in": w_(s, tok_in[s]), "tok_in_is": ("⊥" if L.is_bot(32, tok_in[s]) else f"Active({L.payload(32, tok_in[s])})" if L.is_active(32, tok_in[s]) else "INV"),
                "out": w_(s, out[s]),
                "relation": f"live_{s} = LStepLive{len(eos_ids)}_v1({'out' if lag == 0 else 'tok_in'}[{s - 1}], eos_set); tok_in_{s} = LSelect32_v1(live_{s}, out[{s - 1}], LBot32)"}

    def body_words(s: int) -> dict:
        rows = []
        n_leaves = 0
        for k, n in enumerate(body.nodes):
            if cl.step_of_node[k] != s or k in lifted.guard_nodes:
                continue
            widths = sorted(set(_leaf_widths(n.out)))  # lifted widths (w+1)
            nl = len(_leaf_widths(n.out))
            n_leaves += nl
            row = {"node": k, "name": n.name, "spec": _spec_id(n.fn), "leaves": nl, "w_lifted": widths, "tag": cl.tags[k]}
            if cl.tags[k] == B:
                row["words"] = {str(w): "1" + "0" * (w - 1) for w in widths}  # enc(⊥) per distinct width: one literal covers every leaf
                row["form"] = "enc(⊥) = 1‖0^w, every leaf; relation ⊥ → ⊥ (dense strict rule)"
            else:
                row["form"] = "0‖v, every leaf; v = the committed raw payload (the Commit's leaf, opened at the original root pod-side)"
            rows.append(row)
        tags = {r["tag"] for r in rows}
        return {"step": s, "nodes": len(rows), "leaves": n_leaves, "tags": sorted(tags), "rows": rows}

    def vu(s: int, kind: str) -> dict:
        d = {"kind": kind, "step": s, "guard": guard_words(s), "controller": controller_words(s), "body": body_words(s)}
        if kind == "active":
            d["ok"] = d["body"]["tags"] == [A] and (out[s] is not None and L.is_active(32, out[s])) and (s == 0 or L.is_active(32, tok_in[s]))
        elif kind == "inactive":
            d["ok"] = (d["body"]["tags"] == [B] and live[s] == 0 and L.is_bot(32, tok_in[s]) and L.is_bot(32, out[s])
                       and c_emit[s] == 0 and c_is_eos[s] == 0 and L.is_bot(32, c_out[s]))  # (e, r) = (0, 0); IsEOS(bot) = 0 by the gate; o = bot
        return d

    res: dict[str, Any] = {"a_r": a_r, "cap": cap, "lag": lag, "n_exec": n_exec, "eos": eos_ids, "guard_replay_agrees": guard_replay_agrees,
                           "controller_replay_agrees": controller_replay_agrees,
                           "replayed_by": [step_live.id, sel.id, "LBot32_v1 (const enc(⊥))", is_eos_p.id, emit_next_p.id],
                           "interface_out": [w_(s, out[s]) for s in range(cap)],
                           "designated_out": [w_(s, c_out[s]) for s in range(cap)],
                           "controller_pairs": [(None if c_emit[s] is None else f"({c_emit[s]}, {1 if s <= lag else live[s]})") for s in range(cap)]}
    s_act = max(0, min(a_r - 1, n_exec - 1))  # the EOS-sampling step on an EOS stop (its out is Active(EOS)); the last executed step otherwise
    res["active"] = vu(s_act, "active")
    if n_exec < cap:
        s_last, s_first_bot = a_r + lag - 1, a_r + lag
        tr = {"kind": "transition", "from_step": s_last, "to_step": s_first_bot,
              "last_active": vu(s_last, "active"), "first_inactive": vu(s_first_bot, "inactive")}
        # the letter of A7 (c): tok_in_{a_r} = Active(EOS) under LAG=1 (the placeholder step runs on the EOS); live_{a_r+LAG} = Active(0); tok_in_{a_r+LAG} = ⊥
        fed = tok_in[a_r] if lag == 1 and a_r < cap else (out[a_r - 1] if a_r >= 1 else None)
        tr["tok_in_a_r_is_active_eos"] = (fed is not None and L.is_active(32, fed) and L.payload(32, fed) in set(eos_ids))
        tr["live_first_inactive_is_active_0"] = live[s_first_bot] == 0
        tr["tok_in_first_inactive_is_bot"] = L.is_bot(32, tok_in[s_first_bot])
        tr["every_consumer_bot"] = tr["first_inactive"]["body"]["tags"] == [B]
        # Continuation_v1 across the transition (R17-32): the stop slot a_r has (e, r) = (1, 1), is_eos 1, o = Active(EOS); under LAG = 1 the
        # lag slot a_r + 1 has (0, 1) and o = bot (its real token is committed, never designated); the first inactive slot has (0, 0), o = bot
        s_stop = a_r - 1
        tr["controller"] = {"stop_slot": {"slot": a_r, **controller_words(s_stop)},
                            "lag_slot": ({"slot": a_r + 1, **controller_words(a_r)} if lag == 1 and a_r < cap else None),
                            "first_inactive_slot": {"slot": s_first_bot + 1, **controller_words(s_first_bot)}}
        tr["stop_slot_is_1_1_eos_emitted"] = (c_emit[s_stop] == 1 and c_is_eos[s_stop] == 1 and c_out[s_stop] is not None
                                              and L.is_active(32, c_out[s_stop]) and L.payload(32, c_out[s_stop]) in set(eos_ids))
        tr["lag_slot_is_0_1_out_bot"] = (True if lag == 0 else (c_emit[a_r] == 0 and live[a_r] == 1 and c_out[a_r] is not None and L.is_bot(32, c_out[a_r])))
        tr["first_inactive_is_0_0_out_bot"] = (c_emit[s_first_bot] == 0 and live[s_first_bot] == 0 and c_out[s_first_bot] is not None and L.is_bot(32, c_out[s_first_bot]))
        tr["ok"] = (tr["last_active"]["ok"] and tr["first_inactive"]["ok"] and (tr["tok_in_a_r_is_active_eos"] if lag == 1 else True)
                    and tr["live_first_inactive_is_active_0"] and tr["tok_in_first_inactive_is_bot"] and tr["every_consumer_bot"]
                    and tr["stop_slot_is_1_1_eos_emitted"] and tr["lag_slot_is_0_1_out_bot"] and tr["first_inactive_is_0_0_out_bot"])
        res["transition"] = tr
        res["inactive"] = vu(cap - 1, "inactive")
        res["ok"] = bool(guard_replay_agrees and controller_replay_agrees and res["active"]["ok"] and tr["ok"] and res["inactive"]["ok"])
    else:
        res["transition"] = None
        res["inactive"] = None
        res["note"] = "length stop (a_r = cap): no transition and no inactive step in this record; the active VU alone is listed"
        res["ok"] = bool(guard_replay_agrees and controller_replay_agrees and res["active"]["ok"])
    return res


def load_descriptor(build_request_dir: str | Path) -> dict:
    p = Path(build_request_dir) / "descriptor.json.gz"
    with gzip.open(p, "rt") as f:
        return json.load(f)


def executed_steps(bmap: dict, request_id: str) -> set[int]:
    """Engine steps at which the binding map commits at least one identity of `request_id`."""
    out: set[int] = set()
    for e in bmap.get("entries") or []:
        ident = e.get("identity") or {}
        if "step" in ident and str(ident.get("request_id", "r0")) == request_id:
            out.add(int(ident["step"]))
    return out


def read_capture(capture_dir: str | Path) -> dict:
    """What a Match capture directory (`match/capture/`) attests about the engine of record, read without interpretation:
    the `RunHeader` (the first event of `log.jsonl.gz`), the `StepStart` rows (which request occupied which row at which engine step, and
    the step's `input_token_ids`), the `SampleEnd` sampled ids per step (EVERY step, a LAG=1 placeholder's discarded sample included),
    and `tokens.json`'s `timing.output_lag_measured` (the harness's own measurement of the sample→input lag).  Nothing here is a LAG source
    (RULING 21: the header field, or nothing); the measured lag and the step rows are what the declared LAG is checked AGAINST."""
    c = Path(capture_dir)
    header = None
    steps: dict[int, dict] = {}
    with gzip.open(c / "log.jsonl.gz", "rt") as f:
        for line in f:
            e = json.loads(line)
            k = e.get("$e")
            if k == "RunHeader" and header is None:
                header = e
            elif k == "StepStart":
                st = steps.setdefault(int(e["step"]), {})
                st["rows"] = [(str(r[0]), int(r[1])) for r in e.get("rows") or []]
                st["input_token_ids"] = list(e.get("input_token_ids") or [])
            elif k == "SampleEnd":
                steps.setdefault(int(e["step"]), {})["sampled"] = list(e.get("sampled_token_ids") or [])
    tk = json.loads((c / "tokens.json").read_text()) if (c / "tokens.json").exists() else {}
    timing = tk.get("timing") or {}
    spec = json.loads((c / "spec_tokens.json").read_text()) if (c / "spec_tokens.json").exists() else {}
    run = json.loads((c.parent / "run.json").read_text()) if (c.parent / "run.json").exists() else {}
    realized = ((run.get("realized") or {}).get("requests") if isinstance(run.get("realized"), dict) else None) or []
    return {"header": header, "steps": steps, "tokens": tk,
            "spec_requests": {str(r.get("request_id")): r for r in (spec.get("requests") or [])},  # the engine uuid -> declared (LP, cap, arrive)
            "realized_requests": [dict(r) for r in realized],                                    # Match run.json realized.requests[k], manifest order
            "row_uuids": sorted({rid for st in steps.values() for rid, _ in (st.get("rows") or [])}),
            "lag_measured": timing.get("output_lag_measured"), "lag_measured_source": timing.get("output_lag_source"),
            "engine_steps": timing.get("engine_steps"), "step_iterations": timing.get("step_iterations")}


def capture_request(cap_doc: dict, request_id: str) -> dict:
    """One request's view of the capture: the engine steps at which it occupied a row, its sampled id at each (the last row of a prefill
    step is the sampling row), and the input token fed at each decode step."""
    executed: list[int] = []
    sampled: dict[int, int] = {}
    inputs: dict[int, int] = {}
    prompt_rows: int | None = None  # the request's row count at its FIRST step (its prefill): the prompt length the engine saw
    for step in sorted(cap_doc["steps"]):
        st = cap_doc["steps"][step]
        rows = [i for i, (rid, _) in enumerate(st.get("rows") or []) if rid == request_id]
        if not rows:
            continue
        if prompt_rows is None:
            prompt_rows = len(rows)
        executed.append(step)
        # the request's sampling row is its LAST row of the step (the prefill's final position; a decode step's single row); the
        # sampled array is one id per REQUEST in row order of first appearance (vLLM's sampler output), so index by request order
        req_order: list[str] = []
        for rid, _ in st.get("rows") or []:
            if rid not in req_order:
                req_order.append(rid)
        smp = st.get("sampled") or []
        j = req_order.index(request_id)
        if j < len(smp):
            sampled[step] = int(smp[j])
        ins = st.get("input_token_ids") or []
        if len(rows) == 1 and rows[0] < len(ins):
            inputs[step] = int(ins[rows[0]])
    return {"executed": executed, "sampled": sampled, "inputs": inputs, "prompt_rows": prompt_rows}


def lifted_program_digest(lifted) -> str:
    """sha256 of the lifted Program's id -- the digest `lifted_instances_emit` writes as `result.json program.digest` and the one a Commit /
    required manifest AGAINST Lift(P) would carry (F-pad-30).  One function, so the tool and the emitter cannot drift."""
    import hashlib
    return hashlib.sha256(lifted.id.encode()).hexdigest()


def required_population(commit_verdict: dict, *, program_digests: set[str] | dict[str, str], events: int, bot_steps: list[int], act_steps: list[int],
                        n_bot_events: int, want_requests: list[str]) -> tuple[bool | None, dict]:
    """R7 / W4: the Commit verdict's `required_manifest` (program_digest, populations, coverage[]) against the lifted population.
    `program_digests`: the fixed Program(s) the manifest may traverse (the component's, or the workload's); `events`: the population's
    event count (cap, or Σ cap_r); `bot_steps` / `act_steps`: the ENGINE steps with a ⊥ / an Active event; `n_bot_events`: Σ over requests
    of their ⊥ events; `want_requests`: the Commit ids with ⊥ events.  Returns (ok, fields) -- see check_record's R7 note."""
    rm = commit_verdict.get("required_manifest") or {}
    pops = rm.get("populations") or {}
    cov = [c for c in (rm.get("coverage") or []) if isinstance(c, dict)]
    mpd = rm.get("program_digest")
    names = program_digests if isinstance(program_digests, dict) else {d: "unlifted" for d in program_digests}
    same = (mpd in names) if mpd is not None else None
    # F-pad-30: WHICH fixed Program the manifest traverses -- the unlifted P (today's cap-shape Build) or Lift(P) (the same population under the
    # lift, reqval M-0591 / the emitter's result.json digest): named, so the first Commit against Lift(P) reads 'lifted', never 'another Program'
    manifest_of = names.get(mpd) if same else None
    msteps = pops.get("steps")
    by_fam = pops.get("by_family") or {}
    base = {"manifest_program_digest": mpd, "manifest_of_fixed_program": same, "manifest_of": manifest_of, "manifest_digest": rm.get("manifest_digest"),
            "manifest_steps": msteps, "events": events, "identities": pops.get("identities"), "coverage_runs": len(cov)}
    if not cov or not pops:
        return None, {**base, "why": "no required-manifest coverage record in the Commit verdict"}
    if same is False:
        return None, {**base, "why": "the Commit's required manifest traverses ANOTHER Program (a served-shape derivation, ruling 35): not the fixed "
                                     "population D_MAX x requests -- its coverage says nothing about the bot population"}
    # F-pad-29 (padrev M-0608): EVERY coverage run is read; the runs must agree on what they left unbound (steps, families, requests, ok) --
    # a row whose runs disagree is refused by name, never read from run 0 alone
    def _shape(c: dict) -> tuple:
        return (sorted(int(s) for s in (c.get("steps_with_missing") or [])), {str(k): int(v) for k, v in (c.get("missing_by_family") or {}).items()},
                sorted(str(x) for x in (c.get("requests_with_missing") or [])), c.get("ok"), c.get("missing_n"))
    shapes = [_shape(c) for c in cov]
    disagree = [i for i, sh in enumerate(shapes) if sh != shapes[0]]
    if disagree:
        return False, {**base, "coverage_runs_agree": False, "runs_disagreeing_with_run_0": disagree,
                       "run_shapes": [{"steps_with_missing": sh[0][:8], "missing_by_family": sh[1], "requests_with_missing": sh[2], "ok": sh[3], "missing_n": sh[4]} for sh in shapes],
                       "why": f"the Commit's {len(cov)} coverage runs DISAGREE on what is unbound (runs {disagree} differ from run 0): nothing is read from one run alone"}
    c0 = cov[0]  # every run of the row checks the same manifest and the runs agree (checked above)
    swm = sorted(int(s) for s in (c0.get("steps_with_missing") or []))
    miss_by = {str(k): int(v) for k, v in (c0.get("missing_by_family") or {}).items()}
    chk_by = {str(k): int(v) for k, v in (c0.get("checked_by_family") or {}).items()}
    fully_bound = c0.get("ok") is True and not swm and not miss_by
    bad: list[str] = []
    if msteps is not None and int(msteps) != events:
        bad.append(f"manifest steps {msteps} != population events {events}")
    active_missing = sorted(set(swm) & set(act_steps) - set(bot_steps))
    if active_missing:
        bad.append(f"ACTIVE steps with unbound identities {active_missing[:8]} (a coverage hole of the executed population)")
    # per-event families: population / events identities per event -> x n_bot_events unbound when the Commit binds the executed steps only
    # (today's Commit); 0 unbound everywhere when the whole fixed population is bound (a lifted Commit, or no bot event at all)
    fam: dict[str, dict[str, Any]] = {}
    for f, pop in by_fam.items():
        pop = int(pop)
        per_event = pop // events if events and pop % events == 0 else None
        got = miss_by.get(f, 0)
        want = 0 if fully_bound else (per_event * n_bot_events if per_event is not None else 0)
        fam[f] = {"population": pop, "per_event": per_event, "unbound": got, "unbound_expected": want, "checked": chk_by.get(f)}
        if got != want:
            bad.append(f"{f}: {got} unbound != {want}" + ("" if fully_bound or per_event is None else f" (= {per_event}/event x {n_bot_events} bot events)"))
    if not fully_bound:
        if swm != sorted(set(bot_steps)) and not active_missing:
            bad.append(f"steps_with_missing {swm[:8]}.. != bot steps {sorted(set(bot_steps))[:8]}..")
        rws = c0.get("requests_with_missing")
        if rws is not None and swm and sorted(str(x) for x in rws) != sorted(want_requests):
            bad.append(f"requests_with_missing {rws} != {sorted(want_requests)}")
    return not bad, {**base, "coverage_runs_agree": True, "coverage_ok": c0.get("ok"), "checked": c0.get("checked"), "missing_n": c0.get("missing_n"), "steps_with_missing": swm,
                     "bot_steps": sorted(set(bot_steps)), "fixed_population_fully_bound": fully_bound,
                     # F-pad-31 (padrev M-0698): None when there is no bot step (the property is not exercised: the empty set equals the empty set is
                     # not a finding), True/False only when the record has a bot population to compare against
                     "unbound_is_exactly_bot_population": (swm == sorted(set(bot_steps))) if bot_steps else None, "families": fam, "why": "; ".join(bad) or None,
                     "note": "the required manifest of the FIXED Program traverses D_MAX x requests (no post-hoc rebuild); a Commit of the executed "
                             "steps only leaves unbound exactly the bot population -- what the lifted record half binds as enc(bot) (A3)"}


def check_record(desc: dict, transcript: Sequence[int], eos: Sequence[int], *, cap: int, bmap: dict | None = None, request_id: str = "r0",
                 arrive: int = 0, run_row: dict | None = None, c2_value_check: dict | None = None, transcript_source: str = "unspecified",
                 boundary_prefix: str = L.BOUNDARY_PREFIX, guard: str = L.GUARD_IN, lag: int, lag_source: str = "unspecified",
                 capture: dict | None = None, commit_request_id: str | None = None, workload_digest: str | None = None, lifted_workload_digest: str | None = None,
                 params: list[tuple[str, str]] | None = None, commit_verdict: dict | None = None,
                 eos_source: str = EOS_SOURCE_GIVEN,
                 declared: tuple[int | None, str | None] | None = None) -> dict:
    """The named checks R1–R6 on one request.  Returns the JSON-able report; `ok` is the conjunction of R1–R5 (R6 is never PASS here).
    `capture` (from `read_capture`) adds the capture-side checks R0b/R3c/R3d — the ENGINE's executed set, its placeholder input and the
    harness's measured lag against the declared LAG; they are evidence beside R3 (the Commit's executed set), never a substitute for it
    in the `ok` rule (F-pad-17).  `declared` (R17-1, SPEC v0.5 A2.2): the Build's DECLARED lag `(lag, field)` from the row's
    `workload_program.json` (`L.declared_lag_of_row`); R0 then requires declared == attested, and `(None, None)` — a Build that declared
    no lag — is R0 False by name.  `declared=None` (no workload program available to the caller, e.g. a synthetic record) leaves the
    declaration unchecked and SAYS so in `record_verdict.lag_declared_checked`."""
    if guard not in L.GUARDS_IMPLEMENTED:
        raise ValueError(f"guard={guard!r}: only {L.GUARDS_IMPLEMENTED} is implemented (DP-pad-01 option (a))")
    lag = L.check_lag(lag)
    # two id spaces, named: the ENGINE's request id (the capture's StepStart rows, the workload uuid) and the COMMIT's positional id
    # (`identity.request_id = "r<k>"` in binding-map/1); the caller says which is which, nothing is inferred from a population of one
    commit_rid = request_id if commit_request_id is None else commit_request_id
    attested = lag_source == L.LAG_FIELD  # RULING 21: the ONE attested field, by name; any other provenance string is an unattested lag
    rep: dict[str, Any] = {"schema": SCHEMA, "label": LABEL, "rules_version": L.LIFTED_RULES_VERSION, "program_rules_version": L.PADDED_RULES_VERSION,
                           "record_side_meaning": "R4e_enable_closure (LIFTING-SPEC v0.6 B9); R4_assignment is the lifted-v1 propagation closure beside it", "candidate": L.DP_PAD_02,
                           "request_id": request_id, "commit_request_id": commit_rid,
                           "arrive_step": arrive, "cap": cap,
                           "eos": sorted(int(e) for e in eos), "transcript_source": transcript_source, "guard": guard, "lag": lag, "lag_source": lag_source,
                           "scope": "R1-R5 on disk (the Program's own closure under the committed tokens + the binding map's executed set); "
                                    "R6 (Active strata bit-exact vs the retained committed store) NOT evaluated here — this is not a record-half PASS",
                           "checks": {}}
    ck = rep["checks"]

    def put(name: str, ok: bool | None, **kw):
        ck[name] = {"ok": ok, **kw}

    # R0 — R17-1 (SPEC v0.5 A2): LAG is a DECLARED specialization parameter the record must ATTEST.  Two sides, both by name: the
    # attestation (`lag_source` names the ONE header field, RULING 21 / F-pad-17(b) / padrev M-0168 — anything else is unattested) and the
    # declaration (the Build's workload_program.json, read by `declared_lag_of_row` through the same rule GP-01 / GM-01 apply).  R0 is True
    # only when attested AND (declaration checked → declared present and equal); a Build that declared no lag is never ok.
    decl_lag, decl_field = (declared if declared is not None else (None, None))
    decl_checked = declared is not None
    decl_ok = (decl_lag is not None and decl_lag == lag) if decl_checked else None
    r0 = attested and (decl_ok is not False)
    why0 = []
    if not attested:
        why0.append(f"the LAG={lag} names no attested header field ({L.LAG_FIELD}); a lag-0 record without the flag is indistinguishable from a "
                    f"lag-1 record with a deleted step")
    if decl_checked and decl_lag is None:
        why0.append("the Build declared NO lag (workload_program.json has none of " + ", ".join(L.LAG_DECLARED_FIELDS) +
                    "): a pre-R17-1 Build is not a lifted Program of record (moe M-0015 (3): G6 FAILS it by name on an EOS stop)")
    elif decl_checked and decl_lag != lag:
        why0.append(f"the Build declared LAG={decl_lag} ({decl_field}) but the record attests LAG={lag} ({lag_source}): the Program was built for "
                    f"another profile — refused by name (G1 fails it too)")
    put("R0_lag_attestation", r0, lag=lag, lag_source=lag_source, expected_field=L.LAG_FIELD,
        declared_lag=decl_lag, declared_field=decl_field, declared_checked=decl_checked, declared_equals_attested=decl_ok,
        rule="R17-1: LAG declared by the Build (workload_program.json lag.value / stop_rule.lag / stop_rule.eos_lag_rows) == attested by the record "
             "(RunHeader.profile.engine_kwargs.async_scheduling), one function (GP-01 eos_lag_rows_declared) on both; never inferred from a trace",
        why="; ".join(why0) or None)
    # The Program of record is the BUILD's: when the declaration is checked, Lift(P) and every Program-side prediction below (R1-R3c, R8,
    # R9's `A = LAG`) use the DECLARED lag -- R17-1 / CONTINUATION-SPEC s1: `A` is fixed at Build, never re-fitted to the record.  The
    # attested lag is what R0 compared it with.  On a declared != attested record R0 has refused by name already; the checks below then
    # read the record against the Program it claims to be an execution of, so the disagreement is NAMED where it shows (a deleted lag step /
    # execution past the stop in R3c / R9) instead of being absorbed by re-lifting at the record's own profile.  Without a declaration
    # (synthetic record, `declared=None`) the attested lag is all there is and is used, said so in `lag_program_source`.
    lag_attested = lag
    if decl_checked and decl_lag is not None:
        lag, lag_program_source = int(decl_lag), f"declared ({decl_field})"
    else:
        lag_program_source = f"attested ({lag_source}); no declaration checked"
    rep["lag_attested"], rep["lag_program"], rep["lag_program_source"] = lag_attested, lag, lag_program_source

    # R1 population.  `params` = the wrapper's own [(name, type repr)] (derive_step instances.json.gz): `request_component` CACHES the
    # specialization by (program_digest, program_fn) on FIRST decode, so a param-less decode here would fix the parameter names to p<j> for
    # the whole process and the workload recompose (W1, F-pad-26(a)) would digest differently from the Build's -- found by the pin on the real
    # B8 row (same gates, another digest); the component must be decoded with the Build's names the first time
    comp = W.request_component(desc, REGISTRY, params)
    bounds = L.request_boundaries(comp.fn.body, boundary_prefix)
    if len(bounds) != cap:
        put("R1_population", False, why=f"the derived body has {len(bounds)} sampling events, the manifest declares cap={cap}", events=len(bounds))
        rep["ok"] = False
        return rep
    lifted = L.lift_request_component(comp.fn, eos, max_tokens=cap, lag=lag, boundary_prefix=boundary_prefix)  # F-pad-15: the declared cap, cross-checked
    # F-pad-30 (padrev M-0638): the LIFTED Program's digest -- sha256 of its id, what lifted_instances_emit writes as result.json program.digest
    # -- so that a Commit / a required manifest AGAINST Lift(P) is read as binding the fixed Program (named 'lifted'), not as 'another Program'
    lifted_digest = lifted_program_digest(lifted)
    rows_u = [(_spec_id(n.fn), comp.fn.body.offsets[k], comp.fn.body.offsets[k + 1]) for k, n in enumerate(comp.fn.body.nodes)]
    rows_l = L.address_map_modulo_guards(lifted)
    ranges_ok = [(lo, hi) for _, lo, hi in rows_l] == [(lo, hi) for _, lo, hi in rows_u]
    # PAD-03 modulo the wrapper AND the class vector (v0.7 A11 / G1: `Lifted[X]_v2{..., ORD=m}` / `Lifted[F_v1]_v2{ORD=m}` unlift to X / F_v1)
    specs_ok = [L.unlifted_id(s) for s, _, _ in rows_l] == [_spec_id(n.fn) for n in comp.fn.body.nodes]
    guard_gates = lifted.body.gates - comp.fn.body.gates
    gg = L.continuation_gate_count(cap, len(set(eos)), lag)  # R17-32: the R17-1 guard + the Continuation_v1 controller (3 per slot)
    put("R1_population", ranges_ok and specs_ok and guard_gates == gg,
        events=cap, lifted_id=lifted.id[:120], program_digest=comp.program_digest, lifted_program_digest=lifted_digest,
        unlifted_gates=comp.fn.body.gates, lifted_gates=lifted.body.gates,
        guard_gates=guard_gates, guard_gates_expected=gg, guard_gates_r17_1=L.guard_gate_count(cap, len(set(eos)), lag),
        controller_gates_continuation_v1=3 * cap, continuation=L.CONTINUATION_VERSION,
        deletion_rule_ranges=ranges_ok, deletion_rule_specs=specs_ok)
    # R2 activity
    y = [int(t) for t in transcript]
    try:
        for i, t in enumerate(y):
            L.check_boundary(32, t, what=f"committed token {i}", prescribed=True)
        n = len(y)
        if n > cap:
            raise L.ServedPrefixFailure(request_id, n, cap, f"{n} committed tokens exceed the cap {cap}")
        a_r = L.activity_prefix(y + [L.enc_bot(32)] * (cap - n), eos, cap)
        L.check_served_prefix(request_id, n, a_r, lag=lag, n_events=cap)
        put("R2_activity", True, served=n, a_r=a_r, executed_expected=L.expected_served(a_r, cap, lag), stopped_by_eos=a_r < cap,
            first_eos_index=(a_r - 1) if a_r < cap else None, placeholder_sample=(y[a_r] if lag == 1 and a_r < cap and a_r < n else None))
    except Exception as e:  # noqa: BLE001
        put("R2_activity", False, why=f"{type(e).__name__}: {e}", served=len(y))
        a_r = None
    # R3 executed set — the two lag candidates (DP-pad-02) predict DIFFERENT executed sets on an early stop: LAG=0: the engine never runs
    # step a_r (steps [0, a_r) committed); LAG=1 (vLLM async scheduling): the placeholder step a_r IS executed on the EOS token and its
    # sample committed (steps [0, a_r]).  Both predictions are reported; the verdict is against the DECLARED lag, never a silent pick
    # (padrev M-0168: the same attested value GP-01 reads as eos_lag_rows — `lag_source` names the field).
    if bmap is not None and a_r is not None:
        ex = executed_steps(bmap, commit_rid)
        pred = {lg: {arrive + s for s in range(L.expected_served(a_r, cap, lg))} for lg in L.LAGS_IMPLEMENTED}
        want = pred[lag]
        extra, missing = sorted(ex - want), sorted(want - ex)
        ids_seen = sorted({str((e.get("identity") or {}).get("request_id")) for e in bmap.get("entries") or []})
        put("R3_executed_set", not extra and not missing, declared_lag=lag, commit_request_id=commit_rid, binding_map_request_ids=ids_seen[:16],
            binding_map_program_digest=bmap.get("program_digest"), entries=len(bmap.get("entries") or []),
            committed_steps=sorted(ex)[:40], committed_steps_n=len(ex),
            expected_by_lag={str(lg): sorted(v) for lg, v in pred.items()}, matches={str(lg): ex == v for lg, v in pred.items()},
            declared_inactivity_executed=extra[:8], declared_activity_not_committed=missing[:8])
        # R3a which Program the Commit BOUND: the fixed (declared-cap) Program of record, or another one -- today's early-stop rows Match and
        # Commit against the SERVED-SHAPE derivation (LP<lp>_T<a_r>, the practice the lifting replaces), whose identities address a different
        # Program: then R3 (the executed SET, by step) is the only Commit-side statement here and no identity address is comparable.  Named,
        # never inferred; not a failure of the record (it is the scope of what a Commit of today can attest about the fixed Program).
        bd = bmap.get("program_digest")
        bound = ("component" if bd == comp.program_digest else "lifted_component" if bd == lifted_digest else
                 "workload" if workload_digest is not None and bd == workload_digest else
                 "lifted_workload" if lifted_workload_digest is not None and bd == lifted_workload_digest else "other")
        same = bound != "other"  # a B>=2 Commit binds the WORKLOAD Program (GP-01's compose of the fixed components): that is the fixed Program too
        put("R3a_commit_program", True if same else None, binding_map_program_digest=bd, fixed_program_digest=comp.program_digest,
            lifted_program_digest=lifted_digest, workload_digest=workload_digest, lifted_workload_digest=lifted_workload_digest, bound=bound,
            binds_fixed_program=same, binds_lifted=bound.startswith("lifted_") if same else None,
            why=None if same else "the Commit bound ANOTHER Program (a served-shape derivation, by today's Match/Commit practice): only the executed "
                                  "SET (R3, by engine step) is comparable; its identities address that Program, not the fixed one")
        if not same:
            rep["scope"] += f"; the Commit binds Program {str(bd)[:16]}.. not the fixed Program {comp.program_digest[:16]}.. (R3a): executed-set evidence only"
        # R3e the bot POPULATION in the Commit: the lifted record half binds enc(bot) for every value of every bot event (A3; F-pad-06(b) Sigma
        # leaf_words) -- today's Commit binds only the executed steps (sys's int-3 earlystop row: 7,900 required identities of steps 12..31
        # unbound, verdict 'identity coverage').  Reported by name; `None` (scope), never a pass: a record with bot events and no bot
        # identities is NOT a lifted record, and `record_verdict.bot_population_bound` says so beside `ok`.
        n_exec = L.expected_served(a_r, cap, lag)
        bot_steps = [arrive + s for s in range(n_exec, cap)]
        bound_bot = sorted(ex & set(bot_steps))
        put("R3e_bot_population", None if bot_steps else True, bot_steps_n=len(bot_steps), bot_steps_with_identities=bound_bot[:8],
            bot_population_bound=(bool(bot_steps) and len(bound_bot) == len(bot_steps)) if bot_steps else None,
            why=None if not bot_steps else ("no identity of any bot step is committed: today's Commit binds the executed steps only -- the lifted record "
                                            "half (enc(bot) per bot value, Sigma leaf_words per event) is not in this record" if not bound_bot else
                                            "identities at bot steps are committed: whether they carry enc(bot) is R6's question (values)"))
    else:
        put("R3_executed_set", None, why="no binding map given" if bmap is None else "R2 failed")
        put("R3a_commit_program", None, why="no binding map given" if bmap is None else "R2 failed")
        put("R3e_bot_population", None, why="no binding map given" if bmap is None else "R2 failed")
    # R7 the REQUIRED POPULATION (reqval coordination; BRIEF s7B 'required logical values remain assigned before sampling'): the Commit's
    # verdict.json records the required manifest it checked coverage against -- the Program it traverses (`program_digest`), its step count and
    # per-family identity populations, and per run the identities left UNBOUND by step / family.  When that manifest is the FIXED Program's
    # (the cap-shape derivation, today's Build), it already traverses D_MAX x requests with no post-hoc rebuild, and a Commit of the executed
    # steps only leaves unbound EXACTLY the bot population: steps_with_missing == the bot steps, and every per-step family's missing count ==
    # (its population / cap) x |bot steps|.  Measured on sys's int-3 earlystop row: 7,900 of 12,943 unbound = 20 bot steps x 395 per step
    # (instance_outputs 363 + fa2 stream 30 + input 1 + sampled 1); weights (303) are not per step and never missing.  `ok` True when the
    # unbound set is exactly the bot population (the required manifest IS the fixed population and the Active population is fully bound);
    # False when an ACTIVE step has an unbound identity (a coverage hole of the executed population), when the manifest's step count is not
    # the cap, or when the missing counts are not the bot stratum's; None with the scope named when there is no verdict / no coverage record,
    # or when the manifest traverses ANOTHER Program (a served-shape Commit, ruling 35: not the fixed population -- nothing is inferred).
    if commit_verdict is not None and a_r is not None:
        n_exec = L.expected_served(a_r, cap, lag)
        ok7, f7 = required_population(commit_verdict, program_digests={comp.program_digest: "unlifted", lifted_digest: "lifted"}, events=cap,
                                      bot_steps=[arrive + s for s in range(n_exec, cap)], act_steps=[arrive + s for s in range(0, n_exec)],
                                      n_bot_events=cap - n_exec, want_requests=[commit_request_id or request_id] if n_exec < cap else [])
        put("R7_required_population", ok7, fixed_program_digest=comp.program_digest, lifted_program_digest=lifted_digest, cap=cap, **f7)
    else:
        put("R7_required_population", None, why="no Commit verdict given" if commit_verdict is None else "no a_r")
    if capture is not None and a_r is not None:
        cr = capture_request(capture, request_id)
        ex_c = set(cr["executed"])
        pred = {lg: {arrive + s for s in range(L.expected_served(a_r, cap, lg))} for lg in L.LAGS_IMPLEMENTED}
        put("R3c_capture_executed_set", ex_c == pred[lag], declared_lag=lag, engine_steps=sorted(ex_c), engine_steps_n=len(ex_c),
            expected_by_lag={str(lg): sorted(v) for lg, v in pred.items()}, matches={str(lg): ex_c == v for lg, v in pred.items()},
            source="match/capture/log.jsonl.gz StepStart rows (the engine of record's own step log, Match side; NOT the Commit's binding map)")
        # R0b: the harness measured the sample->input lag itself (tokens.json timing.output_lag_measured); it must equal the declared LAG.
        # A cross-check only: it is not the attested header field and is never a LAG source (RULING 21)
        lm = capture.get("lag_measured")
        put("R0b_lag_measured_consistency", (int(lm) == lag) if lm is not None else None, declared_lag=lag, measured=lm,
            measured_source=capture.get("lag_measured_source"), engine_steps=capture.get("engine_steps"), step_iterations=capture.get("step_iterations"),
            note="cross-check of the declared LAG against the harness's measurement; never a source (an absent header field stays a refusal)")
        # R3d: under LAG=1 the placeholder step a_r is fed the committed EOS (DP-pad-02's model of the engine: the guard's operand is the
        # previous sample), and the sample it produced is the one the transcript carries at index a_r
        if lag == 1 and a_r < cap:
            st = arrive + a_r
            fed = cr["inputs"].get(st)
            smp = cr["sampled"].get(st)
            y_ = [int(t) for t in transcript]
            put("R3d_placeholder_step", (fed is not None and fed == y_[a_r - 1] and int(y_[a_r - 1]) in {int(e) for e in eos}
                                         and (smp is None or (a_r < len(y_) and smp == y_[a_r]))),
                step=st, input_token=fed, committed_eos=y_[a_r - 1], placeholder_sample=smp, transcript_at_a_r=(y_[a_r] if a_r < len(y_) else None))
        # the capture's own sample array vs the transcript given (both are the engine's SampleEnd ids when --tokens is omitted)
        cs = [cr["sampled"][st] for st in cr["executed"] if st in cr["sampled"]]
        put("R2b_capture_samples", cs == [int(t) for t in transcript], capture_samples_n=len(cs), transcript_n=len(transcript))
    if run_row is not None and a_r is not None:
        # the Commit run row (runs.jsonl, schema verity/commit-integ/delta/v1): `generated_tokens` = the tokens the DRIVER returned = a_r (the EOS
        # included, a LAG-1 placeholder's discarded sample excluded); `engine_steps` = `iterations` = the engine LOOP count = a_r + LAG for a B1
        # request WHATEVER the stop reason -- under async scheduling the loop runs one extra iteration after the last token in both cases; on an
        # EOS stop that iteration schedules the request (the placeholder step: a StepStart row, an executed step of the Program), on a length
        # stop it schedules nothing (no StepStart row).  So the run row's `engine_steps` is NOT the executed-step count (that is R3c's StepStart
        # rows = expected_served(a_r, cap, LAG) = min(a_r + LAG, cap)); the two coincide on an early stop and differ by one on a length stop.
        # Measured: sys's int-3 earlystop row 11 / 12 (a_r 11, LAG 1); the OLMoE B1 length-stop row 32 / 33 (a_r = cap = 32, LAG 1) -- the
        # first model of this check (`engine_steps == expected_served`) was REFUTED by the length-stop record and is replaced by the
        # stop-reason-independent count, which discriminates LAG on a length stop too (LAG 0 would read 32 / 32).
        gt, es, it = run_row.get("generated_tokens"), run_row.get("engine_steps"), run_row.get("iterations")
        n_exec = L.expected_served(a_r, cap, lag)
        n_iter = a_r + lag
        ok_gt = (gt == a_r) if gt is not None else None
        ok_es = (es == n_iter) if es is not None else None
        ok_it = (it == n_iter) if it is not None else None
        put("R3b_run_row", None if (ok_gt is None and ok_es is None and ok_it is None) else (ok_gt is not False and ok_es is not False and ok_it is not False),
            generated_tokens=gt, a_r=a_r, engine_steps=es, iterations=it, expected_iterations=n_iter, expected_executed_steps=n_exec,
            iterations_exceed_executed_steps=(n_iter > n_exec), transcript_len=len(y), declared_lag=lag,
            note="run-row engine_steps == iterations == a_r + LAG (loop count, stop-reason independent); the executed-step count is R3c's StepStart rows")
    elif run_row is not None:
        put("R3b_run_row", None, why="R2 failed")
    else:  # F-pad-28a: at B >= 2 the loop count is the batch's (max_r(arrive_r + a_r) + LAG) -- the workload path passes no run row, by design
        put("R3b_run_row", None, why="no run row given (at B >= 2 the run row's loop count is a workload fact, not a request's: unevaluated)")
    # R4 assignment
    if a_r is not None:
        try:
            cl = activity_closure(lifted, y, eos)
            active_events = [s for s, t in enumerate(cl.event_tag) if t == A]
            bad_mixed = [m for m in cl.mixed_nodes if m[2] < L.expected_served(a_r, cap, lag)]
            n_exec = L.expected_served(a_r, cap, lag)
            live_ok = cl.guard_live == [1 if s < n_exec else 0 for s in range(1 + lag, cap)]
            pre_ok = all(cl.bot_nodes_by_event.get(s, 0) == 0 for s in range(n_exec))
            post_ok = all(cl.event_tag[s] == B for s in range(n_exec, cap))
            events_ok = active_events == list(range(n_exec))
            put("R4_assignment", events_ok and not bad_mixed and pre_ok and post_ok and live_ok,
                active_events=len(active_events), a_r=a_r, guards_live_ok=live_ok, active_prefix_all_active=pre_ok, inactive_boundaries_bot=post_ok,
                mixed_operand_nodes_in_active_steps=[{"node": k, "spec": s[:60], "event": e} for k, s, e in bad_mixed[:8]],
                mixed_operand_nodes_in_bot_steps=len(cl.mixed_nodes) - len(bad_mixed),
                bot_leaves_by_event={str(s): v for s, v in sorted(cl.bot_leaves_by_event.items())},
                bot_words_by_event={str(s): v for s, v in sorted(cl.bot_words_by_event.items())},
                bot_leaves_total=sum(cl.bot_leaves_by_event.values()), bot_words_total=sum(cl.bot_words_by_event.values()),
                bot_nodes_total=sum(cl.bot_nodes_by_event.values()), active_nodes_total=sum(cl.active_nodes_by_event.values()),
                guard_bot_leaves=cl.guard_bot_leaves,
                # bot32 + tok_in_s for s in [n_exec, cap) + the Continuation_v1 designated outputs slot=<t>/out with emit_t = 0 (t in (a_r, cap]: cap - a_r)
                guard_bot_leaves_expected=1 + (cap - n_exec) + (cap - a_r))
            # F-pad-06(b) (padrev): the size a LIFTED binding map / committed store must record per event -- Σ leaf_words(w) over every leaf
            # of the event's nodes (33-bit leaves take 2 words) -- so a one-word-per-leaf reader is refused by SIZE, not trusted.  Reported
            # beside the legacy one-word count; the difference is the number of 32-bit-payload leaves in the event
            put("R4b_lifted_layout_words", True,
                leaves_by_event={str(s): v for s, v in sorted(cl.leaves_by_event.items())},
                lifted_words_by_event={str(s): v for s, v in sorted(cl.words_by_event.items())},
                one_word_per_leaf_by_event={str(s): v for s, v in sorted(cl.unlifted_words_by_event.items())},
                leaves_total=sum(cl.leaves_by_event.values()), lifted_words_total=sum(cl.words_by_event.values()),
                one_word_per_leaf_total=sum(cl.unlifted_words_by_event.values()),
                note="the required `Σ leaf_words` field of a lifted binding map (F-pad-06(b)); a store of one_word_per_leaf_total words is NOT a lifted store")
            # R17-6(b) (SPEC v0.5 A4.4): the TRUE lifted widths of the population, by event and in total -- Σ (w+1) over every leaf; the
            # payload total beside it for information (it is the unlifted Program's width, NOT a lifted quantity); widest single node output
            put("R4c_true_lifted_widths", True, rules_version=L.LIFTED_RULES_VERSION, kind=L.USE_KIND_LIFTED,
                lifted_bits_by_event={str(s): v for s, v in sorted(cl.lifted_bits_by_event.items())},
                lifted_bits_total=sum(cl.lifted_bits_by_event.values()), payload_bits_total=sum(cl.payload_bits_by_event.values()),
                grown_by_bits=sum(cl.leaves_by_event.values()),  # one tag bit per leaf: lifted_bits_total - payload_bits_total
                widest_node_output={"lifted_bits": cl.node_width_max[0], "payload_bits": cl.node_width_max[1], "spec": cl.node_width_max[2][:80]},
                w_max_active_reference=L.W_MAX, w_max_lifted=L.W_MAX_LIFTED, w_max_lifted_status=L.W_MAX_LIFTED_STATUS,
                note="widths are the lifted Σ(w+1): nothing is split into an activity bit and a payload to stay under 32 bits (R17-6(b)); "
                     "VU admissibility of Lift(P) is partition.validate_width at w_max_lifted on a lifted query (lifted_width_admissibility), not here")
            rep["assignment"] = cl.node_rows
            # R4' (v0.6 B9): the ENABLE closure -- the padded-v1 meaning -- beside the propagation closure, compared node for node
            ec = enable_closure(lifted, cl, lag=lag)
            ext_w, ext_l = sum(ec["extra_bot_words_by_event"].values()), sum(ec["extra_bot_leaves_by_event"].values())
            put("R4e_enable_closure", ec["ok"], rules_version=L.PADDED_RULES_VERSION, run_by_event=ec["run"], enables_monotone=ec["enables_monotone"],
                enabled_but_bot=ec["enabled_but_bot"][:8], enabled_but_bot_n=len(ec["enabled_but_bot"]),
                disabled_but_active_by_spec=dict(sorted(ec["disabled_but_active_by_spec"].items(), key=lambda kv: -kv[1])[:12]),
                disabled_but_active_nodes=ec["disabled_but_active_nodes"],
                bot_words_by_event_enable={str(k): v for k, v in sorted(ec["bot_words_by_event"].items())},
                bot_words_total_enable=sum(ec["bot_words_by_event"].values()), bot_words_total_propagation=sum(cl.bot_words_by_event.values()),
                extra_bot_words_under_enable=ext_w, extra_bot_leaves_under_enable=ext_l,
                materialization_projection={"bot_words_enable": sum(ec["bot_words_by_event"].values()),
                                            "bytes_u32_per_word": 4 * sum(ec["bot_words_by_event"].values()), "projected_not_measured": True,
                                            "why": "an active-reference Commit binds no bot word; A14.3 (g) is MEASURED only on a padded Commit"},
                root_owned_class_exempt=sorted({_spec_id(lifted.body.nodes[k].fn).split("{")[0] for k in lifted.guard_nodes})[:12],
                root_owned_judged_by="R8_guard_replay / R9_controller_replay: the registered relation evaluated on the record's own operands (v0.7 F3)",
                note="padded-v1 (A9.3): an ORIGINAL node is bot iff its owning step's enable run_s = 0; enabled_but_bot must be empty (a bot on a taken "
                     "path is a fault); disabled_but_active = constants / weights-only gates inside disabled steps -- the words that differ from "
                     "lifted-v1. The root-owned guard/controller class (LStepLive, LIsEOS, LSelect tok_in/out, LEmitNext, LBot) is enabled by the "
                     "root's constant 1 and its bot words are the VALUE of its relation (v0.7 A9.4/B9 F3): judged by replay, not by this closure")
        except Exception as e:  # noqa: BLE001
            put("R4_assignment", False, why=f"{type(e).__name__}: {e}")
            cl = None
    else:
        put("R4_assignment", None, why="R2 failed")
        cl = None
    # R5 interface
    if a_r is not None and cl is not None:
        want = L.assign_inactive(request_id, y, eos, cap, lag=lag) if ck["R2_activity"]["ok"] else None
        got = [L.enc_active(32, cl.event_token[s]) if cl.event_tag[s] == A else L.enc_bot(32) for s in range(cap)]
        put("R5_interface", want == got, bot_events=sum(1 for x in got if L.is_bot(32, x)), expected_bot_events=cap - L.expected_served(a_r, cap, lag))
    else:
        put("R5_interface", None, why="R2/R4 failed")
    # R8 three VUs (SPEC v0.5.1 A7 (c)): one VU of each kind from this record with its words LISTED; the guard chain replayed by the
    # registered lifted evaluators on the committed transcript, independently of R4's tag propagation and checked against it
    if a_r is not None and cl is not None and ck["R2_activity"]["ok"]:
        try:
            tv = three_vus(lifted, cl, y, eos, a_r, cap, lag)
            put("R8_three_vus", tv["ok"], guard_replay_agrees=tv["guard_replay_agrees"], controller_replay_agrees=tv["controller_replay_agrees"],
                active_step=tv["active"]["step"], active_controller=tv["active"]["controller"]["pair_emit_run"],
                transition=(None if tv["transition"] is None else {"from_step": tv["transition"]["from_step"], "to_step": tv["transition"]["to_step"],
                                                                     **{k: tv["transition"][k] for k in ("tok_in_a_r_is_active_eos", "live_first_inactive_is_active_0",
                                                                                                          "tok_in_first_inactive_is_bot", "every_consumer_bot",
                                                                                                          "stop_slot_is_1_1_eos_emitted", "lag_slot_is_0_1_out_bot",
                                                                                                          "first_inactive_is_0_0_out_bot", "ok")}}),
                inactive_controller=(None if tv["inactive"] is None else tv["inactive"]["controller"]["pair_emit_run"]),
                inactive_step=(None if tv["inactive"] is None else tv["inactive"]["step"]), note=tv.get("note"),
                replayed_by=tv["replayed_by"], words="rep['three_vus'] (guard words in full per step; body words per node, one enc(⊥) literal per width)")
            rep["three_vus"] = tv
        except Exception as e:  # noqa: BLE001
            put("R8_three_vus", False, why=f"{type(e).__name__}: {e}")
    else:
        put("R8_three_vus", None, why="R2/R4 failed")
    # R9 Continuation_v1 (COORD R17-32; CONTINUATION-SPEC v0.1; rev M-0445; revb S-r17b-16): the Program's controller bits under this
    # transcript (`slot=<t>/emit`, `slot=<t>/is_eos`, `slot=<t>/out`, the guard's `live` = run) EQUAL the rule's recurrence over the same raw tokens;
    # against the capture: rev's (3a)-(3e) computed from the rule + the RAW SampleEnd ids (lag slot included), the slot <-> engine-step map a
    # stated fact (revb R1), and the WRONG profile tried on the same transcript (spec s7: it must fail by name when the lag was exercised)
    if a_r is not None and cl is not None and ck["R2_activity"]["ok"]:
        try:
            from verity_vllm.lifting import continuation as K
            A_ = lag  # A = LAG of the PROGRAM (the Build's declared lag when checked; revb R6: one field, one rule) -- never re-fitted to the record
            slots = K.recurrence(y, cap=cap, A=A_, eos=eos)
            run_prog = [1] * (1 + lag) + list(cl.guard_live)  # run_1 = 1; steps 1..LAG unguarded (run 1); run_{s+1} = live_s
            emit_rule = [int(s.emit) for s in slots]
            run_rule = [int(s.run) for s in slots]
            out_rule = [s.out for s in slots]
            prog = {"emit": cl.emit, "is_eos": cl.is_eos, "run": run_prog, "out": cl.designated_out}
            rule = {"emit": emit_rule, "is_eos": [int(s.is_eos) for s in slots], "run": run_rule, "out": out_rule}
            agree = {k: prog[k] == rule[k] for k in prog}
            inv = K.invariants(slots, cap=cap, A=A_)
            r9: dict[str, Any] = {"version": K.CONTINUATION_VERSION, "A": A_, "A_source": lag_program_source, "A_attested": lag_attested,
                                  "profile": K.PROFILES[A_], "cap": cap, "program_bits_equal_rule": agree,
                                  "bits": {"emit": emit_rule, "run": run_rule}, "kinds": [s.kind for s in slots], "lag_slots": K.lag_slots(slots),
                                  "served_by_emit": K.served_output(slots), "executed_by_run": K.executed_slots(slots), "invariants": inv,
                                  "designated_output_served_count": sum(1 for o in cl.designated_out if o is not None),
                                  "raw_committed_active_count": sum(1 for t in cl.event_tag if t == A),
                                  "families": {"is_eos": "LIsEOS<E>_v1 slot=<t>/is_eos", "emit": "Const2 slot=1/emit; LEmitNext[H=1]_v1 slot=<t>/emit", "run": "LStepLive<E>_v1 s=<s>/live (R17-1 unit = run, exhaustively equal: continuation.chain_bits)",
                                               "out": "LSelect32_v1 slot=<t>/out (the return)", "in": "LSelect32_v1 s=<s>/tok_in"}}
            prompt_rows = capture_request(capture, request_id)["prompt_rows"] if capture is not None else None  # the engine's own prefill row count
            r9["rng"] = K.rng_accounting(slots, prompt_len=prompt_rows)  # revb R5: the Definition's statement, named on every record
            r9["rng"]["prompt_len_source"] = None if prompt_rows is None else "capture StepStart rows of the request at its first step (the prefill)"
            ok9 = all(agree.values()) and all(inv.values())
            if capture is not None:
                cr = capture_request(capture, request_id)
                tkreq = next((r for r in ((capture.get("tokens") or {}).get("requests") or []) if str(r.get("request_id")) == str(request_id)), None)
                has_served = tkreq is not None and isinstance(tkreq.get("generated_token_ids"), list)
                served_rec = [int(t) for t in tkreq["generated_token_ids"]] if has_served else [int(t) for t in y[:a_r]]
                timing_req = (((capture.get("tokens") or {}).get("timing") or {}).get("requests") or {}).get(str(request_id)) or {}
                chunk = (capture.get("tokens") or {}).get("timing", {}).get("chunk_attribution") or {}
                smap = K.slot_map(cr["executed"], arrive_step=arrive, prefill_steps=timing_req.get("prefill_steps"),
                                  chunked_prefill_observed=chunk.get("chunked_prefill_observed"))
                r9["slot_map"] = smap
                r9["slot_map_vacuous_by_absence"] = smap["vacuous_by_absence"]  # revb S-r17b-20 n1: named, never counted as a pass
                r9["served_source"] = ("tokens.json requests[].generated_token_ids (the driver's served list, of record)" if has_served else
                                       "the transcript's first a_r tokens (no generated_token_ids in the capture's tokens.json: the served list is "
                                       "DERIVED here, so (b) is the transcript against itself and only (a), (c), (d), (e), (f) are of record)")
                if smap["ok"]:
                    raw_by_slot = {st - arrive + 1: int(v) for st, v in cr["sampled"].items()}
                    ct = K.check_transcript(executed_slots_of_record=smap["slots"], raw_by_slot=raw_by_slot, served_of_record=served_rec, cap=cap, A=A_, eos=eos)
                    r9["match_side"] = {k: v for k, v in ct.items() if k != "slots"}
                    r9["wrong_profile"] = K.wrong_profile_negative(executed_slots_of_record=smap["slots"], raw_by_slot=raw_by_slot, served_of_record=served_rec,
                                                                   cap=cap, A_program=A_, eos=eos)
                    ok9 = ok9 and ct["ok"] and (r9["wrong_profile"]["fails_by_name"] is not False)
                else:
                    ok9 = False
            else:
                has_served = False  # no capture: nothing of record to check the rule against
            # rev S-r17-29 n1 / n2: R9 is OF RECORD only when BOTH the served list (the driver's generated_token_ids) and the EOS set (the
            # Build's pinned stop_rule) are the record's own; a DERIVED served list or a check-time --eos makes the verdict None, never True
            # (a False stays False: a named failure against derived inputs is still a failure of the rule on the raw transcript)
            r9["eos_source"] = eos_source
            not_of_record: list[str] = []
            if not has_served:
                not_of_record.append("served list DERIVED from the transcript (no generated_token_ids in tokens.json): (b) is the transcript against itself (n1)")
            if eos_source != EOS_SOURCE_OF_RECORD:
                not_of_record.append(f"EOS set from {eos_source} (n2)")
            # rev M-0677 reading rule, in code: no LAG declaration checked -> `A` is the record's attested lag, not the Build's declaration
            # (a synthetic record, or a pre-R17-1 Program that declares no LAG) -> R9 is NOT of record for this request
            if not (decl_checked and decl_lag is not None):
                not_of_record.append(f"A = LAG taken from the attestation, no Build declaration checked (lag_program_source = {lag_program_source!r}; rev M-0677)")
            r9["of_record"] = not not_of_record
            r9["not_of_record_because"] = not_of_record
            if ok9 is True and not_of_record:
                ok9 = None
            put("R9_continuation_v1", ok9, **{k: r9[k] for k in ("version", "A", "A_source", "A_attested", "profile", "program_bits_equal_rule", "lag_slots", "executed_by_run",
                                                                  "designated_output_served_count", "raw_committed_active_count", "of_record", "eos_source")},
                invariants_all=all(inv.values()), match_side_ok=(r9.get("match_side") or {}).get("ok"), slot_map_ok=(r9.get("slot_map") or {}).get("ok"),
                slot_map_vacuous_by_absence=r9.get("slot_map_vacuous_by_absence"), chunked_prefill_attested=(r9.get("slot_map") or {}).get("chunked_prefill_attested"),
                wrong_profile_fails_by_name=(r9.get("wrong_profile") or {}).get("fails_by_name"),
                rng_advances_iff_run={k: r9["rng"][k] for k in ("draws", "draws_after_served_eos", "lag_draw_slot", "lag_draw_pos")},
                why=(None if not not_of_record else "not of record (rev S-r17-29): " + "; ".join(not_of_record)),
                note="the controller bits are computed from the Program's rule + the raw transcript, never supplied or inferred from where the rows stop; "
                     "full table in rep['continuation']")
            rep["continuation"] = r9
        except Exception as e:  # noqa: BLE001
            put("R9_continuation_v1", False, why=f"{type(e).__name__}: {e}")
    else:
        put("R9_continuation_v1", None, why="R2/R4 failed")
    # R6 values: not evaluated here
    put("R6_values", None, why="row-level bit-exactness of the Active strata needs the retained committed store (in-process on the pod); not evaluated on disk",
        active_reference_c2=(None if c2_value_check is None else {k: c2_value_check.get(k) for k in ("ok", "verdict", "strata", "evaluated", "pass") if k in c2_value_check}),
        note="the active-reference C2 verdict is context, not this lane's evidence")
    # F-pad-17(a): `ok` is a verdict only when the record's executed set (R3, binding map) was evaluated; Program-only runs report None.
    # F-pad-22 (padrev M-0352): `ok` is False whenever ANY evaluated check is False -- the capture checks (R0b, R2b, R3c, R3d) included: a
    # Commit whose executed set the engine's own step log refutes does not read True ("beside R3" also means "never overruled by R3").
    core = all(ck[k]["ok"] is True for k in ("R1_population", "R2_activity", "R4_assignment", "R5_interface"))
    any_false = [k for k, c in ck.items() if c["ok"] is False]
    rep["failed_checks"] = any_false
    if not r0 or any_false:
        rep["ok"] = False
    elif ck["R3_executed_set"]["ok"] is None:
        rep["ok"] = None if core else False
        rep["scope"] += "; R3 not evaluated (no binding map): Program-side closure only, NO record verdict"
    else:
        rep["ok"] = core and ck["R3_executed_set"]["ok"] is True
    # F-pad-23 (padrev M-0398): what `ok` is a statement ABOUT, as fields beside it -- a finalizer must not over-read `ok = True`
    a_r = ck["R2_activity"].get("a_r")
    padding_exercised = a_r is not None and L.expected_served(a_r, cap, lag) < cap           # at least one bot event in the population
    # F-pad-28b (padrev M-0581): LAG is discriminated by TWO independent record facts, named apart -- (i) the EXECUTED SET (R3 / R3c: the two
    # lag predictions of expected_served differ, i.e. an early stop with a_r + 1 <= cap; a length stop says nothing here) and (ii) the Commit
    # RUN ROW (R3b: engine_steps == iterations == a_r + LAG, the loop count -- the two lag predictions ALWAYS differ, so an evaluated R3b
    # discriminates LAG on every row, the length stop included: 32 / 33 refuses LAG 0).  `lag_discriminated` = either.
    by_exec = a_r is not None and L.expected_served(a_r, cap, 0) != L.expected_served(a_r, cap, 1)
    r3b = ck.get("R3b_run_row", {})
    by_run_row = r3b.get("ok") is not None and (r3b.get("engine_steps") is not None or r3b.get("iterations") is not None)
    lag_discriminated = bool(by_exec or by_run_row)
    rep["record_verdict"] = {
        "ok": rep["ok"],
        "padding_exercised": padding_exercised,          # False: 0 bot -- the population rule was exercised only trivially (a length stop)
        "lag_discriminated": lag_discriminated,          # False: NO record fact separates LAG 0 from LAG 1 -- this record says nothing about DP-pad-02
        "lag_discriminated_by_executed_set": by_exec,    # R3/R3c: the two lag predictions of the executed set differ (early stop only)
        "lag_discriminated_by_run_row": by_run_row,      # R3b: the loop count a_r + LAG was evaluated (every stop reason)
        "executed_set_evaluated": ck["R3_executed_set"]["ok"] is not None,
        "commit_binds_fixed_program": ck.get("R3a_commit_program", {}).get("binds_fixed_program"),
        "bot_population_bound": ck.get("R3e_bot_population", {}).get("bot_population_bound"),  # False: the Commit carries NO identity of any bot step
        "fixed_population_traversed": ck.get("R7_required_population", {}).get("ok"),  # True: the required manifest IS D_MAX x requests and what is unbound is exactly the bot population
        "values_evaluated": False,                       # R6 never on disk
        "attested": attested,
        "lag_declared_checked": decl_checked,            # R17-1: False -> no workload_program.json was given; the declaration is unverified
        "lag_declared_equals_attested": decl_ok,
        "rules_version": L.LIFTED_RULES_VERSION,
        "active_reference_commit": active_reference_commit_context(commit_verdict),
        "statement": ("structure + executed set of the fixed Program under the committed tokens" + ("" if padding_exercised else "; 0 bot: padding trivial")
                      + ("" if lag_discriminated else "; lag not discriminated")
                      + ("" if decl_checked else "; lag declaration NOT checked (no workload_program.json)")
                      + ("; the bot population is NOT bound in this Commit (executed steps only)" if ck.get("R3e_bot_population", {}).get("bot_population_bound") is False else "")
                      + "; values (R6) not evaluated"),
    }
    return rep


# --- the MIXED BATCH record half: a population of requests (GP-01's workload), on top of the per-request checks -------------------------

def live_schedule(a_r: int, cap: int, arrive: int, lag: int) -> dict[int, int]:
    """`live_{r,t}` for the engine steps `t` request `r` is programmed for (`arrive <= t < arrive + cap`): 1 for its executed events
    (`expected_served(a_r, cap, lag)` of them), 0 after — the value `live_count_hook` recomputes from the committed tokens, as a plain
    function of the transcript-derived `a_r` (the Program-side hook and this agree by construction; T-M5 tests the hook)."""
    n = L.expected_served(a_r, cap, lag)
    return {arrive + s: (1 if s < n else 0) for s in range(cap)}


def check_workload_record(requests: Sequence[dict], eos: Sequence[int], *, lag: int, lag_source: str, bmap: dict | None = None,
                          capture: dict | None = None, boundary_prefix: str = L.BOUNDARY_PREFIX, row_facts: dict | None = None,
                          commit_verdict: dict | None = None, declared: tuple[int | None, str | None] | None = None,
                          eos_source: str = EOS_SOURCE_GIVEN) -> dict:
    """The MIXED BATCH stage's record half — **lifted-padded**.  `requests`: one dict per declared request with the manifest
    facts and its record: `{"request_id" (engine/capture id), "commit_request_id" (binding-map id), "desc" (build_request descriptor),
    "cap", "arrive", "transcript" (optional when a capture is given), "params" (the wrapper parameters `[(name, type repr)]`, naming the
    shared `weights` group for GP-01's compose)}`.  Per request the R-checks of `check_record` run unchanged (each
    request's guard on its own tokens); across requests:

      W1 population   the lifted WORKLOAD (`lifted_workload`, GP-01's compose over the lifted components) has `Σ_r cap_r` events and
                      `body_gates = unlifted + Σ_r guard_gate_count(cap_r)`; its digest depends on (manifest, EOS, LAG) only.  With
                      `row_facts` (the row's own `workload_program.json`: `workload_digest`, `gates`, `component_gates`,
                      `component_program_digests {r<k>: digest}`) the unlifted recompose is PINNED to the Build's record — a mismatch is
                      W1 False by name (F-pad-26(a)); without it the report says `pinned_to_row: False`.
      W2 chronology   for every engine step `t` of the horizon: the set of requests the ENGINE ran (capture StepStart rows), the set the COMMIT
                      committed (binding map identities at `t`), and the set the PROGRAM declares live (`live_schedule` from each request's
                      own committed tokens) are the same set — `n_live_t` (the S-migration operand) is a value of the committed tokens that the
                      record's two observations must agree with; a request committed/run at a step its own guard has ⊥ is "declared
                      inactivity executed" AT THE WORKLOAD LEVEL (a cross-request effect would show here even when each request's own
                      R3 passes — e.g. a request kept alive by another's tokens).
      W3 no crossing  a request's `a_r` is a function of ITS transcript only (recomputed here from each transcript alone) and equals the
                      per-request R2 value — no request's activity depends on another's tokens (the Program has no cross-request edge;
                      this is the record-side statement of it).
      W4 values       R6 per request: not evaluated on disk.
    `ok` follows F-pad-17: False when unattested or any per-request/workload check fails; None when no binding map was given (Program +
    capture only); True only with the Commit's executed sets evaluated."""
    lag = L.check_lag(lag)
    attested = lag_source == L.LAG_FIELD
    rep: dict[str, Any] = {"schema": SCHEMA + "/workload", "label": LABEL, "rules_version": L.LIFTED_RULES_VERSION, "program_rules_version": L.PADDED_RULES_VERSION, "candidate": L.DP_PAD_02,
                           "lag": lag, "lag_source": lag_source,
                           "lag_declared": (declared[0] if declared is not None else None), "lag_declared_field": (declared[1] if declared is not None else None),
                           "lag_declared_checked": declared is not None, "requests": [], "checks": {}}
    per: list[dict] = []
    comps = []
    for r in requests:
        rid, crid = str(r["request_id"]), str(r.get("commit_request_id") or r["request_id"])
        transcript = r.get("transcript")
        if transcript is None:
            if capture is None:
                raise ValueError(f"request {rid}: no transcript and no capture")
            cr = capture_request(capture, rid)
            transcript = [cr["sampled"][st] for st in cr["executed"] if st in cr["sampled"]]
        sub = check_record(r["desc"], transcript, eos, cap=int(r["cap"]), bmap=bmap, request_id=rid, arrive=int(r.get("arrive", 0)),
                           transcript_source=str(r.get("transcript_source", "capture" if r.get("transcript") is None else "given")),
                           boundary_prefix=boundary_prefix, lag=lag, lag_source=lag_source, capture=capture, commit_request_id=crid,
                           workload_digest=r.get("workload_digest"), params=r.get("params"), declared=declared, eos_source=eos_source)
        sub.pop("assignment", None)
        sub["committed_tokens"] = [int(t) for t in transcript]  # the executed transcript (served + LAG row's sample): what R2 read; A7 (d)'s rule of record reads it
        per.append(sub)
        rep["requests"].append(sub)
        comp = W.request_component(r["desc"], REGISTRY, r.get("params"))  # `params`: the wrapper's [(name, type repr)] (derive_step instances.json.gz), names the shared group
        # the recompose is the BUILD's workload: its request ids are the manifest's r<k> (COORD R15 ruling 2), a static of the workload digest --
        # the first F-pad-26(a) pin caught the tool composing under the ENGINE uuids (same gates, another digest)
        comps.append(W.WorkloadRequest(crid, len(comps), int(r.get("prompt_len") or 0), int(r["cap"]), int(r.get("arrive", 0)), comp))
    ck = rep["checks"]

    def put(name, ok, **kw):
        ck[name] = {"ok": ok, **kw}

    # W0 pairing (F-pad-24, padrev M-0398): the engine-uuid <-> Commit r<k> pairing is READ from the record and checked, never taken by hand:
    # a bijection (distinct uuids, distinct r<k>), every uuid the engine ran declared, and per pair the record's own facts about the uuid
    # (spec_tokens.json / tokens.json / run.json realized.requests[k]: prompt_len, max_tokens, arrive_step) equal the manifest's for r<k>.
    # LIMIT (named): two requests with the same (prompt_len, cap, arrive) are indistinguishable by pairing facts -- only R6 sees their swap.
    pairs = [(str(r.get("commit_request_id") or r["request_id"]), str(r["request_id"]), r) for r in requests]
    p_bad: list[str] = []
    if len({u for _, u, _ in pairs}) != len(pairs):
        p_bad.append("engine ids not distinct (not injective)")
    if len({c for c, _, _ in pairs}) != len(pairs):
        p_bad.append("commit ids not distinct (not injective)")
    declared_u = {u for _, u, _ in pairs}
    undeclared_ran = [u for u in (capture or {}).get("row_uuids", []) if u not in declared_u]
    if undeclared_ran:
        p_bad.append(f"engine ran undeclared request ids {undeclared_ran[:4]}")
    declared_absent = sorted(declared_u - set((capture or {}).get("row_uuids", []))) if capture is not None else []
    fact_mismatch: dict[str, dict] = {}
    if capture is not None:
        spec = capture.get("spec_requests") or {}
        tokreq = {str(q.get("request_id")): q for q in (capture.get("tokens") or {}).get("requests") or []}
        realized = capture.get("realized_requests") or []
        for k, (crid, u, r) in enumerate(pairs):
            want = {"prompt_len": r.get("prompt_len"), "max_tokens": int(r["cap"]), "arrive_step": int(r.get("arrive", 0))}
            for src_name, src in (("spec_tokens.json", spec.get(u)), ("tokens.json", tokreq.get(u))):
                if src is None:
                    continue
                got = {"prompt_len": src.get("prompt_len", len(src.get("prompt_token_ids") or []) or None), "max_tokens": src.get("max_tokens"),
                       "arrive_step": src.get("arrive_step")}
                diff = {f: (want[f], got[f]) for f in want if want[f] is not None and got[f] is not None and int(want[f]) != int(got[f])}
                if diff:
                    fact_mismatch.setdefault(crid, {})[src_name] = diff
            if realized:  # run.json realized.requests[k] is in manifest order: its request_id must be the uuid declared for r<k>
                kk = int(crid[1:]) if crid.startswith("r") and crid[1:].isdigit() else k
                if kk < len(realized) and str(realized[kk].get("request_id")) != u:
                    fact_mismatch.setdefault(crid, {})["run.json realized.requests[k].request_id"] = (u, str(realized[kk].get("request_id")))
    # IDENTITY facts (prompt_len, max_tokens, run.json order) decide WHICH uuid is r<k>: a mismatch is a wrong pairing -- refused.  The
    # record's arrive_step is a CHRONOLOGY fact (when the engine admitted the request): a uuid whose identity facts match but whose arrive_step
    # differs from the declaration IS r<k>, admitted at another step -- W0 False by name, and the report continues so W2 names the step.
    identity_mismatch = {c: {sname: ({f: v for f, v in d.items() if f != "arrive_step"} if isinstance(d, dict) else d)  # run.json order: a tuple
                             for sname, d in m.items() if not (isinstance(d, dict) and set(d) == {"arrive_step"})}
                         for c, m in fact_mismatch.items()}
    identity_mismatch = {c: m for c, m in identity_mismatch.items() if any(bool(v) for v in m.values())}
    arrive_disagreement = {c: {sname: d["arrive_step"] for sname, d in m.items() if isinstance(d, dict) and "arrive_step" in d}
                           for c, m in fact_mismatch.items()}
    arrive_disagreement = {c: m for c, m in arrive_disagreement.items() if m}
    if identity_mismatch:
        p_bad.append("the record's own facts about a uuid do not match the manifest facts for its r<k>")
    elif arrive_disagreement:
        p_bad.append("the record's arrive_step for a uuid differs from its r<k>'s declared arrival (a chronology fact: see W2 at that step)")
    same_shape = {}
    for c1, u1, r1 in pairs:
        key = (r1.get("prompt_len"), int(r1["cap"]), int(r1.get("arrive", 0)))
        same_shape.setdefault(str(key), []).append(c1)
    indistinguishable = {k: v for k, v in same_shape.items() if len(v) > 1}
    sources_evaluated = [n for n, present in (
        ("spec_tokens.json", bool((capture or {}).get("spec_requests"))), ("tokens.json", bool(((capture or {}).get("tokens") or {}).get("requests"))),
        ("run.json realized.requests", bool((capture or {}).get("realized_requests")))) if present]
    # F-pad-25 (padrev M-0452): a pairing checked against NO record fact is a pairing taken by hand -- `ok` None (F-pad-17 class), never True;
    # the bijection/declaration refusals above still apply (they need no fact source)
    facts_verified = bool(sources_evaluated)
    put("W0_pairing", False if p_bad else (True if facts_verified else None), pairs={c: u for c, u, _ in pairs}, why="; ".join(p_bad) or
        (None if facts_verified else "no pairing fact source in the record (spec_tokens.json / tokens.json requests / run.json realized.requests): "
                                     "the uuid <-> r<k> pairing is unverified"),
        declared_but_never_ran=declared_absent, undeclared_ran=undeclared_ran, fact_mismatch=fact_mismatch, identity_mismatch=identity_mismatch,
        arrive_step_disagreement=arrive_disagreement, sources_evaluated=sources_evaluated,
        facts_verified=facts_verified, indistinguishable_by_pairing_facts=indistinguishable,
        limit="requests with equal (prompt_len, cap, arrive) cannot be told apart by pairing facts; a swap of their identities is seen only by R6 (values)")
    # A pairing failure (no bijection, or identity facts that refute the declared uuid <-> r<k>) stops the report: there is no pairing the
    # later checks could be about.  An arrive_step disagreement alone reads W0 False by name (the report is False, F-pad-22) and the checks
    # CONTINUE under the (identity-verified) pairing, so that W2 names the step at which the engine disagreed (an early admission is a
    # chronology fact, named at its step -- padrev's early-admission test) and R3c names the request's executed set.
    hard = [b for b in p_bad if not b.startswith("the record's arrive_step")]
    if hard:
        raise ValueError(f"W0 pairing refused by name: {'; '.join(hard)} (pairs {dict((c, u) for c, u, _ in pairs)})")

    # W1 population: GP-01's compose over the lifted components -- PINNED to the row's own Build facts when given (F-pad-26(a), padrev
    # M-0452): the unlifted recompose must reproduce the recorded workload_digest, total gates, component gates and every component's
    # program_digest, else the tool is checking a Program the Build never emitted -> W1 False by name (never a fallback to the recompose)
    try:
        wl_u = L.workload_program(comps, lag)  # R17-1: on moe's tree the lag is a static of Workload_v1 (the Build's digest carries it)
        wl_l = L.lifted_workload(comps, eos, lag=lag, boundary_prefix=boundary_prefix)
        events = sum(int(r["cap"]) for r in requests)
        guard = sum(L.continuation_gate_count(int(r["cap"]), len(set(eos)), lag) for r in requests)  # R17-32: guard + controller
        gu, gl = wl_u.fn.body.gates, wl_l.fn.body.gates
        pin_bad: list[str] = []
        pin: dict[str, Any] = {"pinned_to_row": row_facts is not None}
        if row_facts is not None:
            want_wd, want_g, want_cg = row_facts.get("workload_digest"), row_facts.get("gates"), row_facts.get("component_gates")
            pin.update(recomposed_workload_digest=wl_u.digest, recorded_workload_digest=want_wd, recomposed_gates=wl_u.program.gates, recorded_gates=want_g,
                               recomposed_program_digest=getattr(wl_u.program, "digest", None),
                       recomposed_component_gates=gu, recorded_component_gates=want_cg)
            if want_wd is not None and wl_u.digest != want_wd:
                pin_bad.append(f"recomposed workload_digest {wl_u.digest[:16]} != recorded {str(want_wd)[:16]}")
            if want_g is not None and int(wl_u.program.gates) != int(want_g):
                pin_bad.append(f"recomposed gates {wl_u.program.gates} != recorded {want_g}")
            if want_cg is not None and int(gu) != int(want_cg):
                pin_bad.append(f"recomposed component gates {gu} != recorded {want_cg}")
            want_pd = row_facts.get("component_program_digests") or {}
            got_pd = {str(r.get("commit_request_id") or r["request_id"]): c.program_digest for r, c in zip(requests, (c.component for c in comps))}
            pin["component_program_digests"] = {k: {"recomposed": got_pd.get(k), "recorded": want_pd.get(k)} for k in sorted(set(got_pd) | set(want_pd))}
            for k in sorted(set(got_pd) | set(want_pd)):
                if want_pd.get(k) is not None and got_pd.get(k) != want_pd.get(k):
                    pin_bad.append(f"{k}: component program_digest {str(got_pd.get(k))[:16]} != recorded {str(want_pd.get(k))[:16]}")
        put("W1_population", not pin_bad and gl == gu + guard and len(wl_l.requests) == len(requests), events=events, requests=len(requests),
            unlifted_gates=gu, lifted_gates=gl, guard_gates=gl - gu, guard_gates_expected=guard, why="; ".join(pin_bad) or None,
            lifted_workload_digest=getattr(wl_l, "digest", None), **pin)
    except Exception as e:  # noqa: BLE001
        put("W1_population", False, why=f"{type(e).__name__}: {e}")
    # W2 chronology: engine rows vs committed identities vs the Program's live schedule, per engine step
    a_rs = [p["checks"]["R2_activity"].get("a_r") for p in per]
    if all(a is not None for a in a_rs):
        prog_live: dict[int, set[str]] = {}
        for r, a_r in zip(requests, a_rs):
            for t, lv in live_schedule(a_r, int(r["cap"]), int(r.get("arrive", 0)), lag).items():
                prog_live.setdefault(t, set())
                if lv:
                    prog_live[t].add(str(r["request_id"]))
        horizon = sorted(prog_live)
        rows = {}
        if capture is not None:
            for r in requests:
                cr = capture_request(capture, str(r["request_id"]))
                for t in cr["executed"]:
                    rows.setdefault(t, set()).add(str(r["request_id"]))
        committed = {}
        if bmap is not None:
            for r in requests:
                for t in executed_steps(bmap, str(r.get("commit_request_id") or r["request_id"])):
                    committed.setdefault(t, set()).add(str(r["request_id"]))
        n_live = {str(t): len(prog_live[t]) for t in horizon}
        bad_rows = {str(t): sorted(rows.get(t, set()) ^ prog_live[t]) for t in horizon if capture is not None and rows.get(t, set()) != prog_live[t]}
        extra_rows = {str(t): sorted(v) for t, v in rows.items() if t not in prog_live}  # the engine ran a step no request is programmed for
        bad_commit = {str(t): sorted(committed.get(t, set()) ^ prog_live[t]) for t in horizon if bmap is not None and committed.get(t, set()) != prog_live[t]}
        extra_commit = {str(t): sorted(v) for t, v in committed.items() if t not in prog_live}
        ok2 = not bad_rows and not extra_rows and not bad_commit and not extra_commit
        put("W2_chronology", ok2 if (capture is not None or bmap is not None) else None, horizon=[horizon[0], horizon[-1]] if horizon else [],
            n_live_by_step=n_live, engine_rows_evaluated=capture is not None, commit_evaluated=bmap is not None,
            engine_vs_program_mismatch=bad_rows, engine_steps_outside_population=extra_rows,
            commit_vs_program_mismatch=bad_commit, commit_steps_outside_population=extra_commit,
            why=None if (capture is not None or bmap is not None) else "neither a capture nor a binding map: the Program's own schedule only")
        # W3: a_r from each transcript alone == the per-request value (no request's activity depends on another's tokens)
        recomputed = []
        for r, p in zip(requests, per):
            y = [int(t) for t in (r.get("transcript") or [])] if r.get("transcript") is not None else None
            if y is None and capture is not None:
                cr = capture_request(capture, str(r["request_id"]))
                y = [cr["sampled"][st] for st in cr["executed"] if st in cr["sampled"]]
            cap = int(r["cap"])
            recomputed.append(L.activity_prefix(y + [L.enc_bot(32)] * max(cap - len(y), 0), eos, cap) if y is not None and len(y) <= cap else None)
        put("W3_no_crossing", recomputed == a_rs, a_r_per_request={str(r["request_id"]): a for r, a in zip(requests, a_rs)}, recomputed=recomputed)
    else:
        put("W2_chronology", None, why="a request failed R2")
        put("W3_no_crossing", None, why="a request failed R2")
    put("W4_values", None, why="R6 per request: not evaluated on disk")
    # W5 the required population of the WORKLOAD (R7 at the workload level): the Commit's required manifest traverses the fixed workload
    # (Σ cap_r events, GP-01's compose) and a Commit of the executed steps only leaves unbound exactly the union of the requests' ⊥ events
    if commit_verdict is not None and all(a is not None for a in a_rs):
        bot_u: list[int] = []
        act_u: list[int] = []
        want_r: list[str] = []
        n_bot = 0
        for r, a_r in zip(requests, a_rs):
            cap_r, arr = int(r["cap"]), int(r.get("arrive", 0))
            n_exec = L.expected_served(a_r, cap_r, lag)
            bot_u += [arr + s for s in range(n_exec, cap_r)]
            act_u += [arr + s for s in range(0, n_exec)]
            n_bot += cap_r - n_exec
            if n_exec < cap_r:
                want_r.append(str(r.get("commit_request_id") or r["request_id"]))
        digests: dict[str, str] = {c.component.program_digest: "unlifted" for c in comps}
        digests.update({d: "lifted" for d in ((p.get("checks") or {}).get("R1_population", {}).get("lifted_program_digest") for p in per) if d})
        w1 = ck.get("W1_population") or {}
        digests.update({d: "unlifted" for d in (w1.get("recomposed_workload_digest"), w1.get("recorded_workload_digest"), w1.get("recomposed_program_digest")) if d})
        if w1.get("lifted_workload_digest"):
            digests[w1["lifted_workload_digest"]] = "lifted"
        ok5, f5 = required_population(commit_verdict, program_digests=digests, events=sum(int(r["cap"]) for r in requests), bot_steps=bot_u,
                                      act_steps=act_u, n_bot_events=n_bot, want_requests=want_r)
        put("W5_required_population", ok5, requests_with_bot_events=want_r, **f5)
    else:
        put("W5_required_population", None, why="no Commit verdict given" if commit_verdict is None else "a request failed R2")
    per_ok = [p["ok"] for p in per]
    core = all(v is True for k, v in ((k, c["ok"]) for k, c in ck.items()) if k in ("W1_population", "W3_no_crossing")) and \
        (ck["W2_chronology"]["ok"] is not False)
    rep["failed_checks"] = [k for k, c in ck.items() if c["ok"] is False] + \
        [f"{p['request_id']}:{k}" for p in per for k in p.get("failed_checks", [])]  # F-pad-22 at both levels: any False anywhere -> False
    decl_ok = (declared[0] is not None and declared[0] == lag) if declared is not None else None  # R17-1: the per-request R0 already carries the reason
    if not attested or decl_ok is False or any(v is False for v in per_ok) or not core or rep["failed_checks"]:
        rep["ok"] = False
    elif bmap is None or any(v is None for v in per_ok):
        rep["ok"] = None
        rep["scope"] = "Program + capture closure only (no binding map): NO record verdict"
    elif ck["W0_pairing"]["ok"] is None:  # F-pad-25: the uuid <-> r<k> pairing was taken by hand (no fact source in the record)
        rep["ok"] = None
        rep["scope"] = "pairing unverified (W0: no pairing fact source in the record): NO record verdict"
    else:
        rep["ok"] = ck["W2_chronology"]["ok"] is True
    # F-pad-23 at the workload level: what the verdict is about.  `ok` is a statement about SETS of request ids per engine step (padrev's
    # measured limit: exchanged identities of two co-live requests commit the same set), never about which values were committed under which id.
    rv = [p.get("record_verdict") or {} for p in per]
    rep["record_verdict"] = {
        "ok": rep["ok"],
        "fixed_population_traversed": ck.get("W5_required_population", {}).get("ok"),
        "padding_exercised": any(v.get("padding_exercised") for v in rv), "requests_padded": [p["request_id"] for p, v in zip(per, rv) if v.get("padding_exercised")],
        "lag_discriminated": any(v.get("lag_discriminated") for v in rv),
        "lag_discriminated_by_executed_set": any(v.get("lag_discriminated_by_executed_set") for v in rv),
        # F-pad-28a (padrev M-0581): the Commit run row's loop count is a WORKLOAD fact at B >= 2 (max_r(arrive_r + a_r) + LAG), never a
        # per-request one -- R3b is deliberately UNEVALUATED per request here (no run_row is passed) and no workload-level R3b exists yet
        "lag_discriminated_by_run_row": False, "run_row_evaluated": False,
        "run_row_note": "R3b is not evaluated at B >= 2: the run row's engine_steps/iterations count the whole batch's loop (max_r(arrive_r + a_r) + LAG), not a request's",
        "executed_set_evaluated": bmap is not None, "values_evaluated": False, "attested": attested,
        "lag_declared_checked": declared is not None, "lag_declared_equals_attested": decl_ok, "rules_version": L.LIFTED_RULES_VERSION,
        "active_reference_commit": active_reference_commit_context(commit_verdict),
        "statement": "per-step SETS of live request ids (engine rows / Commit identities / the Program's live schedule) and each request's structure + "
                     "executed set; NOT which values were committed under which id (a label swap of co-live requests is seen only by R6)"
                     + ("" if declared is not None else "; lag declaration NOT checked (no workload_program.json)"),
    }
    return rep


def _commit_manifest_digest(commit_verdict: dict | None) -> str | None:
    return ((commit_verdict or {}).get("required_manifest") or {}).get("manifest_digest")


def _verdict_executed_prefix(commit_verdict: dict | None) -> dict | None:
    """The Commit's OWN executed-prefix statement, if its verdict carries one (moe aad4bd3ea writes `executed_prefix_of_record` -- per request
    served_output_len / executed_prefix_len / cap / eos_stop -- and `rows_past_the_stop` into its coverage block(s)); None on a pre-F-moe-r17-03
    verdict.  Searched where commit_delta puts it: top level, `required_manifest`, and each `required_manifest.coverage[*]` record."""
    if not isinstance(commit_verdict, dict):
        return None
    cands: list[Any] = [commit_verdict, commit_verdict.get("required_manifest")]
    rm = commit_verdict.get("required_manifest")
    if isinstance(rm, dict):
        cov = rm.get("coverage")
        cands += list(cov) if isinstance(cov, list) else [cov]
    for c in cands:
        if isinstance(c, dict) and isinstance(c.get("executed_prefix_of_record"), dict):
            ep = c["executed_prefix_of_record"]
            return {"requests": ep.get("requests") or {k: v for k, v in ep.items() if isinstance(v, dict) and "executed_prefix_len" in v},
                    "request_steps": ep.get("request_steps"), "ok": ep.get("ok"), "lag": ep.get("lag"), "rows_past_the_stop": c.get("rows_past_the_stop"),
                    "programmed_not_executed_n": c.get("programmed_not_executed_n") or (c.get("manifest") or {}).get("programmed_not_executed_n")}
    return None


def active_reference_commit_context(commit_verdict: dict | None) -> dict | None:
    """The ACTIVE-REFERENCE Commit's own verdict (`commit/verdict.json`: `commit_pass`, `first_fail_reason`), carried BESIDE the lifted-padded
    verdict as named context -- never folded into `ok` (the two are statements about different populations: the Commit's sampled replay over
    the active-reference strata vs this tool's fixed lifted population), never hidden (a reader of a lifted `ok True` must see a Commit FAIL
    on the same record: SmolLM2 B2 mixedstop r17_2153 -- C2 partial on r0's post-stop 'recycled-window' strata, R17-26/27's population
    question, while the lifted population is complete by construction)."""
    if not isinstance(commit_verdict, dict):
        return None
    return {"commit_pass": commit_verdict.get("commit_pass"), "first_fail_reason": commit_verdict.get("first_fail_reason"),
            "runs_instrumented": commit_verdict.get("runs_instrumented"), "runs_failed": commit_verdict.get("runs_failed"),
            "label": "active-reference Commit verdict of the same record: context beside the lifted-padded verdict, never a lifted fact"}


def w5_from_binding_level(rep: dict, w5b: dict | None) -> None:
    """W5 at B >= 2 is decided by the Commit's own BINDING rule per (request, engine step) (a7_correspondence_workload.w5_binding_level), not
    by the verdict's name-level manifest coverage alone: `op/member@step` names cannot attribute a hole to one of two co-scheduled requests
    (moe's r17_1845 mixedstop: name-level checked 19,264 / missing 0 with 40 composition mismatches at the bot steps; binding-level missing
    6,180 = r0's bot population exactly).  The name-level reading is kept beside it as `verdict_manifest_coverage`; the record's `ok`,
    `failed_checks` and `record_verdict.fixed_population_traversed` are recomputed from the merged W5.  In place; no-op without a binding-level verdict."""
    ck = rep.get("checks") or {}
    old = ck.get("W5_required_population") or {}
    if not w5b or w5b.get("ok") is None:
        # no binding-level verdict (no binding map with the Commit's verdict, as in a pull that leaves binding_map_p*.json out): at B >= 2
        # the name-level manifest coverage CANNOT attribute a hole to a request (co-scheduled `op/member@step` names), so a False read from
        # it alone is not a measurement -- W5 becomes None BY NAME, the name-level numbers kept beside; a True stays (nothing missing anywhere)
        if old.get("ok") is False and len(rep.get("requests") or []) >= 2 and "no Commit verdict" not in str(old.get("why")):
            ck["W5_required_population"] = {**old, "ok": None,
                                            "decided_by": "undecidable here: no binding map given, so the Commit's own binding rule could not be run per (request, "
                                                          "engine step); the name-level verdict cannot attribute a hole to a request at B >= 2 (kept beside)",
                                            "verdict_manifest_coverage": {k: old.get(k) for k in ("ok", "coverage_ok", "checked", "missing_n", "steps_with_missing", "families", "why")
                                                                          if k in old} | {"label": "name-level: cannot attribute a hole to a request at B >= 2"},
                                            "why": "no binding map: the binding-level rule (a7_correspondence_workload.w5_binding_level) was not run; " + str(old.get("why"))}
            per = rep.get("requests") or []
            rep["failed_checks"] = [k for k, c in ck.items() if c.get("ok") is False] + [f"{p['request_id']}:{k}" for p in per for k in p.get("failed_checks", [])]
            if rep.get("ok") is False and not rep["failed_checks"]:
                rep["ok"] = None  # nothing False remains, nothing decides: not a pass either
                rv = rep.get("record_verdict") or {}
                rv["ok"] = None
                rv["fixed_population_traversed"] = None
        return
    if old.get("ok") is None and "why" in old and "no Commit verdict" in str(old.get("why")):
        return  # nothing to reconcile: no verdict at all
    name_level = {k: old.get(k) for k in ("ok", "coverage_ok", "checked", "missing_n", "steps_with_missing", "families", "why", "unbound_is_exactly_bot_population",
                                          "fixed_population_fully_bound") if k in old}
    agree = (old.get("ok") == w5b["ok"]) if old.get("ok") is not None else None
    cle = (w5b.get("commit_lengths_equal_match_lengths") or {})
    merged = {**old, "ok": w5b["ok"], "decided_by": "binding-level (the Commit's own load_required_manifest -> coverage_check per (request, engine step)) under "
                                                     + str(w5b.get("commit_rule_used") or "the Commit's rule"),
              "binding_level": w5b, "verdict_manifest_coverage": {**name_level, "label": "name-level: cannot attribute a hole to a request at B >= 2"}, "name_level_agrees": agree,
              "commit_rule_used": w5b.get("commit_rule_used"),
              "unbound_is_exactly_bot_population": w5b.get("unbound_is_exactly_bot_population"), "fixed_population_fully_bound": w5b.get("fixed_population_fully_bound"),
              "why": (None if w5b["ok"] else "binding-level: " + "; ".join(f"{k}={v}" for k, v in w5b.items()
                                                                          if k in ("bot_cells_not_fully_unbound", "active_cells_with_holes", "holes_outside_the_population",
                                                                                   "bot_cells_in_commit_not_executed_class", "active_cells_bound_under_rule_of_record") and v not in (None, {}, True))
                      + ("" if cle.get("ok") is not False else f"; F-r17b-21: the Commit's lengths differ from the Match record's {cle.get('by_request')}")),
              "note": old.get("note", "") + (" | at B >= 2 the verdict's name-level coverage cannot attribute a per-request hole (its composition_mismatch is the "
                                             "corroborating signal): W5 is the binding-level verdict" if agree is False else "")}
    ck["W5_required_population"] = merged
    # recompute the record verdict's dependents
    per = rep.get("requests") or []
    rep["failed_checks"] = [k for k, c in ck.items() if c.get("ok") is False] + [f"{p['request_id']}:{k}" for p in per for k in p.get("failed_checks", [])]
    if rep.get("ok") is False and old.get("ok") is False and w5b["ok"] is True and not rep["failed_checks"]:
        # W5 was the only False: the record's ok is the W2 chronology verdict again, under the same guards as check_workload_record
        rv_ok = ck.get("W2_chronology", {}).get("ok") is True and all(p.get("ok") is True for p in per) and ck.get("W0_pairing", {}).get("ok") is True \
            and (rep.get("record_verdict") or {}).get("attested") is not False and (rep.get("record_verdict") or {}).get("lag_declared_equals_attested") is not False
        rep["ok"] = True if rv_ok else rep["ok"]
    elif w5b["ok"] is False:
        rep["ok"] = False
    if isinstance(rep.get("record_verdict"), dict):
        rep["record_verdict"]["ok"] = rep["ok"]
        rep["record_verdict"]["fixed_population_traversed"] = w5b["ok"]


def a7_correspondence_request(build_request: str | Path, desc: dict, params, rep: dict, bmap: dict | None, commit_verdict: dict | None, *,
                              lag: int, cap: int, arrive: int, commit_rid: str, path: str) -> dict:
    """SPEC v0.5.1 A7 (revb F2) for ONE request: the identity-level table active-reference <-> lifted over `RequiredValues(P)` (the Commit's
    own generator, digest-compared with the Commit verdict's), written to `path`; the summary is returned for the report.  Needs R2's
    `a_r` (the served prefix from the committed tokens); without it nothing is written and the reason is named."""
    from verity_vllm.lifting import correspondence as C
    a_r = (rep.get("checks") or {}).get("R2_activity", {}).get("a_r")
    if a_r is None:
        return {"ok": None, "why": "R2 failed: no served prefix, no assignment, no table"}
    try:
        man = C.required_identities(build_request, request_id=commit_rid, arrive=arrive)
        wbs = C.spec_leaf_widths(desc, params)
        # the binding column is THE Commit's verdict per identity (binding.load_required_manifest + coverage_check on this manifest and map)
        ccov = C.commit_coverage(man, bmap, requests=[commit_rid], caps={commit_rid: int(cap)}, arrivals={commit_rid: int(arrive)}) if bmap is not None else None
        rows, summ = C.correspondence(man, bmap, commit_rid=commit_rid, a_r=int(a_r), cap=cap, lag=lag, arrive=arrive, widths_by_spec=wbs, commit_cov=ccov)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "why": f"{type(e).__name__}: {e}"}
    cmd = _commit_manifest_digest(commit_verdict)
    summ["commit_manifest_digest"] = cmd
    summ["manifest_is_the_commits"] = (cmd == summ["manifest_digest"]) if cmd else None
    summ["table"] = C.write_table(rows, path, summ)
    summ["ok"] = _a7_ok(summ) and summ["manifest_is_the_commits"] is not False
    if summ["manifest_is_the_commits"] is False:
        # by name: a B1 Commit of today's practice binds a SERVED-SHAPE derivation (build_request_LP<p>_T<a_r>, MOE-05 / ruling 35), whose
        # required manifest is another Program's -- the fixed population (this table) is Lift(P)'s; the binding column is read by identity
        # NAME against that map (R3a / R7 say the same), so `ok` is False for the Commit half only, never for the table
        summ["why"] = ("the Commit's required manifest is another Program's (served-shape derivation): manifest_digest of the fixed population "
                       f"{summ['manifest_digest'][:16]} != the Commit's {cmd[:16]}; the table's binding column is read by identity name against "
                       "that map; the population and assignment columns are Lift(P)'s and stand")
    return summ


def _a7_ok(summ: dict) -> bool:
    """The table's own verdict: every row widened, no executed identity unbound, and -- when the record HAS a stop -- the bot suffix present.
    A length stop (a_r == cap) has no bot row: the suffix check is None, `padding_exercised` False, and the verdict is the structural one;
    A7 (a) evidence additionally needs `padding_exercised` (reported, not folded into ok).  `bot_population_bound` is reported, not judged:
    False is the active-reference Commit (the executed set), True would be a Commit of Lift(P) itself."""
    ck = summ["checks"]
    return ck["every_row_widened"] is True and ck["no_active_identity_unbound"] is not False and ck["inactive_suffix_present"] is not False \
        and ck.get("every_row_one_identity") is not False  # F-r17b-28: a table with two rows under one identity key is not identity-level


RULE_EXECUTED_PREFIX = "executed-prefix of record (F-moe-r17-03 aad4bd3ea: the Commit requires each request's executed prefix; the bot rows are its NOT-EXECUTED class)"
RULE_CAP_SHAPE = "cap shape (pre-F-moe-r17-03 Commit: the requirement over every request's cap; the bot rows read unbound = the lifted reading R3e)"


def w5_under_commit_rule(w5b: dict, epr: dict, commit_verdict: dict | None, *, has_cells: bool) -> dict:
    """W5 read against the Commit's OWN population statement (revb M-0342), in place on `w5b` (a7_correspondence_workload.w5_binding_level).
    Which rule the record's Commit applied is read from its verdict: a Commit on moe's aad4bd3ea writes `executed_prefix_of_record` (+
    `rows_past_the_stop`) into its coverage block -- then W5 is: every identity of a bot cell in the Commit's NOT-EXECUTED class
    (`programmed_not_executed_n` == the bot population, cap shape minus rule of record) and never bound, every identity of an active cell
    bound under the rule of record (coverage ok), no committed row past a stop, AND (revb F-r17b-21) the Commit's own per-request
    served_output_len / executed_prefix_len EQUAL this record's Match-derived a_r / n_exec (a Commit that stops elsewhere than the record
    shrinks its own population: False by name).  A pre-F-moe-r17-03 verdict keeps the cap-shaped reading (bot rows unbound, R3e).  The rule
    used is recorded; the name-level reading stays beside it labelled 'cannot attribute'."""
    vcov = _verdict_executed_prefix(commit_verdict)
    cro = w5b.get("commit_rule_of_record") or {}
    lengths = None
    if vcov is not None:
        lengths = {}
        for rid, rec in (vcov.get("requests") or {}).items():
            ours = (epr.get("requests") or {}).get(rid) or {}
            lengths[rid] = {"commit_served": rec.get("served_output_len"), "commit_executed": rec.get("executed_prefix_len"),
                            "match_a_r": ours.get("served_output_len"), "match_n_exec": ours.get("r2_r3_expected_served"),
                            "equal": ((rec.get("served_output_len") == ours.get("served_output_len") and rec.get("executed_prefix_len") == ours.get("r2_r3_expected_served"))
                                      if ours else None)}
    lengths_equal = (all(v["equal"] is True for v in lengths.values()) if lengths else None)
    if vcov is not None:
        ok_r = (cro.get("available") is True and cro.get("coverage_ok") is True and w5b.get("unbound_is_exactly_bot_population") in (True, None)
                and cro.get("programmed_not_executed_equals_bot_population") in (True, None) and (cro.get("rows_past_the_stop") or {}).get("ok") is not False
                and lengths_equal is not False)
        w5b["ok"] = bool(ok_r) if has_cells else None
        w5b["bot_cells_in_commit_not_executed_class"] = cro.get("programmed_not_executed_equals_bot_population")
        w5b["bot_identities_never_bound"] = (w5b.get("bot_identities_unbound") == w5b.get("bot_identities"))
        w5b["active_cells_bound_under_rule_of_record"] = cro.get("coverage_ok")
        w5b["commit_rule_used"] = RULE_EXECUTED_PREFIX
    else:
        w5b["commit_rule_used"] = RULE_CAP_SHAPE
    w5b["commit_verdict_executed_prefix"] = vcov
    w5b["commit_lengths_equal_match_lengths"] = {"ok": lengths_equal, "by_request": lengths,
                                                 "rule": "revb F-r17b-21: the Commit's own served_output_len / executed_prefix_len per request must equal the Match "
                                                         "record's (this tool's a_r from match/capture tokens, n_exec = min(a_r + LAG, cap)); a Commit that stops "
                                                         "elsewhere than the record shrinks its own population -- False by name here",
                                                 "gates": vcov is not None}
    w5b["name_level_reading"] = "kept beside as W5_required_population.verdict_manifest_coverage, labelled 'cannot attribute' at B >= 2"
    return w5b


def executed_prefix_of_record(reqs: Sequence[dict], subs: dict[str, dict], *, lag: int, caps: dict[str, int], arrivals: dict[str, int]) -> dict:
    """Each request's EXECUTED PREFIX OF RECORD by the Commit's own rule -- moe's `verity_capture.commit.executed_prefix.executed_prefix_of_record`
    (R17-1 / F-moe-r17-03 aad4bd3ea: one rule, two consumers): the served tokens are this record's committed transcript up to and including
    the first declared EOS (R2's `a_r`; the LAG row's sample after it is executed, not served), the caps and LAG the declaration's.  Returns
    moe's document plus `request_steps` (the kwarg `binding.load_required_manifest` takes) and `agrees_with_r2_r3`: this tool's
    `expected_served(a_r, cap, lag)` per request equals moe's `executed_prefix_len` -- the two consumers read the one rule alike.  Without
    moe's module in the tree: `available` False, named, and `request_steps` None (the cap-shaped reading stands alone)."""
    try:
        from verity_capture.commit import executed_prefix as EP
    except ImportError as e:
        return {"available": False, "why": f"verity_capture.commit.executed_prefix not in this tree ({e}): moe's F-moe-r17-03 aad4bd3ea", "request_steps": None}
    served: dict[str, list[int]] = {}
    eos_seen: list[int] = []
    ours: dict[str, int | None] = {}
    for r in reqs:
        crid = str(r["commit_request_id"])
        sub = subs.get(crid) or {}
        toks = sub.get("committed_tokens")
        a_r = ((sub.get("checks") or {}).get("R2_activity") or {}).get("a_r")
        eos_seen = [int(x) for x in (sub.get("eos") or eos_seen)]
        if toks is None or a_r is None:
            ours[crid] = None
            continue
        served[crid] = [int(t) for t in toks[: int(a_r)]]          # the served output: through the first declared EOS inclusive (R2), never the LAG row's sample
        ours[crid] = L.expected_served(int(a_r), int(caps[crid]), lag)
    epr = EP.executed_prefix_of_record(list(caps), served or None, caps=caps, eos_of_record=eos_seen or None, lag=lag)
    steps = EP.request_steps_of_record(epr, arrivals)
    per = {rid: {"executed_prefix_len": rec.get("executed_prefix_len"), "served_output_len": rec.get("served_output_len"), "eos_stop": rec.get("eos_stop"),
                 "cap_exhausted": rec.get("cap_exhausted"), "last_served_token": rec.get("last_served_token"), "why": rec.get("why"),
                 "r2_r3_expected_served": ours.get(rid), "agrees": (ours.get(rid) == rec.get("executed_prefix_len")) if ours.get(rid) is not None else None}
           for rid, rec in (epr.get("requests") or {}).items()}
    return {"available": True, "ok": epr.get("ok"), "rule": epr.get("rule") or getattr(EP, "RULE", None), "module": "verity_capture.commit.executed_prefix",
            "lag": epr.get("lag"), "eos_of_record": epr.get("eos_of_record"), "served_known": epr.get("served_known"), "any_eos_stop": epr.get("any_eos_stop"),
            "problems": epr.get("problems"), "requests": per, "request_steps_of_record": {k: ([min(v), max(v)] if v else []) for k, v in steps.items()},
            "agrees_with_r2_r3": all(p["agrees"] is True for p in per.values()) if per else None, "request_steps": steps}


def a7_correspondence_workload(wp: dict, reqs: Sequence[dict], rep: dict, bmap: dict | None, commit_verdict: dict | None, *, lag: int, prefix: str) -> dict:
    """A7 at B >= 2: the Commit's GLOBAL required manifest (`build_global_manifest` over the Build's own declaration) split per `r<k>`; one
    table per request (`<prefix>.<rk>.jsonl.gz`) with the mixed batch visible across them (a request's bot rows at engine steps where the
    other's are active — BRIEF §7 (d))."""
    from verity_vllm.lifting import correspondence as C
    subs = {str(s.get("commit_request_id")): s for s in rep.get("requests") or []}
    try:
        man = C.workload_identities(wp, {str(r["commit_request_id"]): r["component_dir"] for r in reqs})
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "why": f"{type(e).__name__}: {e}"}
    cmd = _commit_manifest_digest(commit_verdict)
    out: dict[str, Any] = {"schema": C.SCHEMA + "/workload", "label": LABEL, "rules_version": L.LIFTED_RULES_VERSION, "manifest_digest": man.get("manifest_digest"),
                           "program_digest": man.get("program_digest"), "commit_manifest_digest": cmd,
                           "manifest_is_the_commits": (cmd == man.get("manifest_digest")) if cmd else None, "requests": {}}
    parts = C.split_by_request(man)
    # F-r17b-15 (1): ONE run of the Commit's own binding rule over the FULL global manifest and the map; every request's table reads its
    # rows' verdicts from it (a batch-level hidden plane is bound per co-scheduled request at its engine step, member `*`, as commit_delta does)
    ccov = None
    caps = {str(r["commit_request_id"]): int(r["cap"]) for r in reqs}
    arrivals = {str(r["commit_request_id"]): int(r.get("arrive", 0)) for r in reqs}
    # the EXECUTED PREFIX OF RECORD by moe's one rule (R17-1 / F-moe-r17-03 `verity_capture.commit.executed_prefix`: cap on a length stop, served
    # + declared LAG on an EOS stop, the last served token a declared EOS id, LAG never from the trace) -- the same function the Commit runs;
    # this tool's own R2/R3 (a_r, expected_served) is checked AGAINST it below, never substituted for it
    epr = executed_prefix_of_record(reqs, subs, lag=lag, caps=caps, arrivals=arrivals)
    out["executed_prefix_of_record"] = {k: v for k, v in epr.items() if k != "request_steps"}
    if bmap is not None:
        try:
            ccov = C.commit_coverage(man, bmap, requests=list(caps), caps=caps, arrivals=arrivals, request_steps=epr.get("request_steps"))
            out["commit_coverage"] = {k: ccov.get(k) for k in ("coverage", "expansion", "horizon", "arrivals", "rule", "shape", "of_record")}
        except Exception as e:  # noqa: BLE001
            return {**out, "ok": False, "why": f"the Commit's binding rule could not be run on this manifest and map: {type(e).__name__}: {e}"}
    if ccov is not None:
        # W5 at the BINDING level (F-pad-W5b, on moe's r17_1845 mixedstop record): the Commit verdict's manifest coverage is NAME-level
        # (`op/member@step`) and at B >= 2 cannot attribute a hole to a request -- on that record it reads checked 19,264 / missing 0 with
        # 40 composition mismatches at exactly the bot steps, while the Commit's own binding rule (this ccov) reads missing 6,180 = r0's bot
        # population.  So the per-(request, engine step) verdict of the binding rule is the W5 fact; the verdict's name-level reading is kept
        # beside it (its composition_mismatch is the corroborating signal), never read alone at B >= 2.
        tot: dict[tuple, int] = {}
        mis: dict[tuple, int] = {}
        for (rid, st, *_rest), b in ccov["bound"].items():
            k = (str(rid), int(st) if isinstance(st, int) or str(st).lstrip("-").isdigit() else -1)
            tot[k] = tot.get(k, 0) + 1
            if not b:
                mis[k] = mis.get(k, 0) + 1
        bot_cells: set[tuple] = set()
        act_cells: set[tuple] = set()
        for r in reqs:
            crid = str(r["commit_request_id"])
            a_r = ((subs.get(crid) or {}).get("checks") or {}).get("R2_activity", {}).get("a_r")
            if a_r is None:
                continue
            cap_r, arr = int(r["cap"]), int(r.get("arrive", 0))
            n_exec = L.expected_served(int(a_r), cap_r, lag)
            bot_cells |= {(crid, arr + s) for s in range(n_exec, cap_r)}
            act_cells |= {(crid, arr + s) for s in range(0, n_exec)}
        bot_all_unbound = {f"{k[0]}@{k[1]}": (mis.get(k, 0), tot.get(k, 0)) for k in sorted(bot_cells) if mis.get(k, 0) != tot.get(k, 0) or tot.get(k, 0) == 0}
        active_holes = {f"{k[0]}@{k[1]}": (mis.get(k, 0), tot.get(k, 0)) for k in sorted(act_cells) if mis.get(k, 0)}
        other_holes = {f"{k[0]}@{k[1]}": (mis[k], tot.get(k, 0)) for k in sorted(mis) if k not in bot_cells and k not in act_cells}
        by_req_missing: dict[str, int] = {}
        for (rid, _st), m in mis.items():
            by_req_missing[rid] = by_req_missing.get(rid, 0) + m
        n_bot_ids = sum(tot.get(k, 0) for k in bot_cells)
        out["w5_binding_level"] = {
            "ok": (not bot_all_unbound and not active_holes and not other_holes) if (bot_cells or act_cells) else None,
            "rule": ccov["rule"], "expected_n": ccov["coverage"]["expected_n"], "bound_n": ccov["coverage"]["bound_n"], "missing_n": ccov["coverage"]["missing_n"],
            "bot_cells": len(bot_cells), "bot_identities": n_bot_ids, "bot_identities_unbound": sum(mis.get(k, 0) for k in bot_cells),
            "active_cells": len(act_cells), "active_identities_unbound": sum(mis.get(k, 0) for k in act_cells),
            "missing_by_request": by_req_missing, "requests_with_bot_events": sorted({k[0] for k in bot_cells}),
            "bot_cells_not_fully_unbound": dict(list(bot_all_unbound.items())[:8]), "active_cells_with_holes": dict(list(active_holes.items())[:8]),
            "holes_outside_the_population": dict(list(other_holes.items())[:8]),
            "unbound_is_exactly_bot_population": (not bot_all_unbound and not active_holes and not other_holes) if bot_cells else None,
            "fixed_population_fully_bound": ccov["coverage"]["missing_n"] == 0,
            "note": "per (request, engine step): every identity of a bot cell unbound, none of an active cell -- the Commit's own binding rule "
                    "(load_required_manifest -> coverage_check); the verdict's name-level manifest coverage cannot attribute a hole at B >= 2"}
        # the Commit's RULE OF RECORD (R17-1 / F-moe-r17-03) beside the cap-shaped reading: the requirement placed over each request's executed
        # prefix must be fully bound (coverage ok) and the rows it does NOT require (programmed_not_executed_n) must be EXACTLY this table's bot
        # population -- the two consumers of the one rule name the same 20 x 309 identities of r0 on moe's r17_1845; and no committed leaf of a
        # request lies past its stop (moe's rows_past_the_stop on the map: 'no cross-request poisoning' read on the binding map itself)
        ofr = ccov.get("of_record")
        if ofr and ofr.get("available"):
            pne = ofr.get("programmed_not_executed_n")
            past = None
            try:
                from verity_capture.commit import executed_prefix as EP
                past = EP.rows_past_the_stop(bmap, epr["request_steps"]) if epr.get("request_steps") else None
            except Exception as e:  # noqa: BLE001
                past = {"ok": None, "why": f"{type(e).__name__}: {e}"}
            out["w5_binding_level"]["commit_rule_of_record"] = {
                "available": True, "coverage_ok": (ofr.get("coverage") or {}).get("ok"), "expected_n": (ofr.get("coverage") or {}).get("expected_n"),
                "bound_n": (ofr.get("coverage") or {}).get("bound_n"), "missing_n": (ofr.get("coverage") or {}).get("missing_n"),
                "missing_by_request": ofr.get("missing_by_request"), "programmed_not_executed_n": pne, "request_steps_of_record": ofr.get("request_steps_of_record"),
                "programmed_not_executed_equals_bot_population": (int(pne) == int(n_bot_ids)) if pne is not None else None,
                "cap_shape_minus_of_record_equals_bot_population": ((int(ccov["coverage"]["expected_n"]) - int((ofr.get("coverage") or {}).get("expected_n") or 0)) == int(n_bot_ids)),
                "rows_past_the_stop": past,
                "verdict_records_programmed_not_executed_n": (((commit_verdict or {}).get("required_manifest") or {}).get("programmed_not_executed_n")
                                                             if isinstance((commit_verdict or {}).get("required_manifest"), dict) else None),
                "note": "the Commit of record requires the executed prefix (F-moe-r17-03 aad4bd3ea) and leaves the bot rows unrequired; the lifted table keeps "
                        "them in the population as bot / unbound (SPEC A3.6 N8, R3e) -- the same identities, read by the two consumers of the one rule"}
        else:
            out["w5_binding_level"]["commit_rule_of_record"] = {"available": False, "why": (ofr or {}).get("why") or epr.get("why") or "no executed prefix of record"}
        # WHICH Commit rule this record's Commit applied (revb M-0342: W5 must read against the Commit's population statement and record the rule):
        # a Commit on moe's aad4bd3ea writes `executed_prefix_of_record` (+ rows_past_the_stop) into its coverage block; the pre-F-moe-r17-03 Commit
        # placed the requirement over the cap.  Under the executed-prefix rule W5 is: every identity of a bot cell in the Commit's NOT-EXECUTED class
        # (programmed_not_executed == the bot population) and never bound, every identity of an active cell bound (coverage of record ok), no row past
        # the stop -- AND (revb F-r17b-21) the Commit's own per-request lengths EQUAL this record's Match-derived a_r / n_exec, else False by name
        w5_under_commit_rule(out["w5_binding_level"], epr, commit_verdict, has_cells=bool(bot_cells or act_cells))
    oks = []
    slice_view: dict[int, dict[str, dict]] = {}
    for r in reqs:
        crid = str(r["commit_request_id"])
        sub = subs.get(crid) or {}
        a_r = (sub.get("checks") or {}).get("R2_activity", {}).get("a_r")
        if a_r is None or crid not in parts:
            out["requests"][crid] = {"ok": None, "why": "R2 failed" if a_r is None else "no identities of this request in the global manifest"}
            oks.append(None)
            continue
        try:
            wbs = C.spec_leaf_widths(r["desc"], r.get("params"))
            part = dict(parts[crid], engine_steps=man.get("engine_steps"))  # the declared composition travels with the split (plane slice steps)
            rows, summ = C.correspondence(part, bmap, commit_rid=crid, a_r=int(a_r), cap=int(r["cap"]), lag=lag, arrive=int(r.get("arrive", 0)), widths_by_spec=wbs,
                                          commit_cov=ccov)
        except Exception as e:  # noqa: BLE001
            out["requests"][crid] = {"ok": False, "why": f"{type(e).__name__}: {e}"}
            oks.append(False)
            continue
        summ["table"] = C.write_table(rows, f"{prefix}.{crid}.jsonl.gz", summ)
        summ["ok"] = _a7_ok(summ)
        out["requests"][crid] = summ
        oks.append(summ["ok"])
        # F-r17b-23: one representative plane per engine step (the first hidden op's `…/L0`) with BOTH placements, for the mixed-step statement
        for row in rows:
            if row.get("packed_plane") and str(row.get("op_path", "")).endswith("/L0"):
                slice_view.setdefault(int(row["engine_step"]), {}).setdefault(crid, {
                    "assignment": row["assignment"], "op_path": row["op_path"], "lifted_word": row["lifted_word"],
                    "lifted_slot": {k: row["lifted_slot"].get(k) for k in ("seq_index", "row_range", "own_rows", "element_range", "elements", "slot_rows", "n_seq")},
                    "committed": ({k: row["committed_slice"].get(k) for k in ("seq_index", "row_range", "rows_total", "n_seq", "carrier", "members", "elements", "element_range_in_carrier", "M_geometry_committed")}
                                  if row["committed_slice"].get("present") else {"present": row["committed_slice"].get("present"), "why": row["committed_slice"].get("why"),
                                                                                  "bot_slot_elements": row["committed_slice"].get("bot_slot_elements")})})
    # the MIXED view (BRIEF §7 (d)): per ENGINE step, which requests are active / bot -- a step where one request is bot and another active
    mixed: dict[int, dict[str, str]] = {}
    for crid, summ in out["requests"].items():
        for s, ps in (summ.get("per_step") or {}).items():
            e = int(s) + int(summ.get("arrive") or 0)
            mixed.setdefault(e, {})[crid] = ps["assignment"]
    mixed_steps = sorted(e for e, m in mixed.items() if len(set(m.values())) > 1)
    out["mixed_engine_steps"] = {"n": len(mixed_steps), "first": mixed_steps[:8], "last": mixed_steps[-8:],
                                 "by_step_first": {str(e): mixed[e] for e in mixed_steps[:4]},
                                 # F-r17b-23: 'r1's Active slice beside r0's enc(bot) slice' with BOTH offsets -- the slot in the lifted fixed plane
                                 # (manifest geometry + declared composition) and the rows of the committed plane (the record's composition), or
                                 # 'absent (committed n_seq = k)' for the request the record did not run at that step
                                 "plane_slices_by_step": {str(e): slice_view.get(e) for e in (mixed_steps[:3] + mixed_steps[-1:]) if e in slice_view},
                                 "plane_slices_transition": {str(e): slice_view.get(e) for e in sorted(slice_view) if mixed_steps and e in (mixed_steps[0] - 1, mixed_steps[0])}}
    out["padding_exercised"] = any(s.get("padding_exercised") is True for s in out["requests"].values())
    out["requests_padded"] = sorted(k for k, s in out["requests"].items() if s.get("padding_exercised") is True)
    # F-r17b-18 (revb): the packed per-step planes are the WORKLOAD's, each counted ONCE here (a plane row of the global manifest = one plane
    # at its whole-launch extent); the per-request tables keep them out of their own widths (`widths.excludes_planes`) and report only their view
    out["plane_widths"] = C.plane_widths(man, {str(r["commit_request_id"]): C.spec_leaf_widths(r["desc"], r.get("params")) for r in reqs})
    # the workload's lifted bits = Σ per-request (planes excluded) + the planes once
    req_bits = sum(int(((s.get("widths") or {}).get("lifted_bits_total")) or 0) for s in out["requests"].values() if isinstance(s.get("widths"), dict))
    req_payload = sum(int(((s.get("widths") or {}).get("payload_bits_total")) or 0) for s in out["requests"].values() if isinstance(s.get("widths"), dict))
    # F-r17b-23: the requests' SLOTS of the planes tile them exactly -- Σ over requests of their slot bits == the planes once
    slot_bits = sum(int((((s.get("widths") or {}).get("planes_seen") or {}).get("slice_lifted_bits_total")) or 0) for s in out["requests"].values() if isinstance(s.get("widths"), dict))
    out["widths"] = {"requests_lifted_bits_total": req_bits, "requests_payload_bits_total": req_payload,
                     "planes_lifted_bits_total": out["plane_widths"]["lifted_bits_total"], "planes_payload_bits_total": out["plane_widths"]["payload_bits_total"],
                     "workload_lifted_bits_total": req_bits + out["plane_widths"]["lifted_bits_total"],
                     "workload_payload_bits_total": req_payload + out["plane_widths"]["payload_bits_total"],
                     "plane_slots_lifted_bits_total": slot_bits,
                     "plane_slots_tile_the_planes": (slot_bits == out["plane_widths"]["lifted_bits_total"]) if slot_bits else None,
                     "rule": "F-r17b-18: per-request totals exclude the packed planes; each plane is counted once here at its whole-launch extent; "
                             "F-r17b-23: the requests' slots of a plane tile it (Σ slot bits == planes once)"}
    out["ok"] = all(o is True for o in oks) and out["manifest_is_the_commits"] is not False if oks else None
    return out


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build-request", default=None, help="dir with descriptor.json.gz (derive_step --wrapper request) -- ONE request (B1)")
    ap.add_argument("--build-workload", default=None, help="a ROW dir with build_workload/workload_program.json (GP-01 B>=2): every declared request is "
                    "checked (check_workload_record) -- its component from inputs.derived['LP,T'] rebased onto <row>/build_request_LP<lp>_T<t>, "
                    "params from instances.json.gz, cap/arrive/EOS from the manifest facts, the engine uuid from request_ids.workload_request_ids")
    ap.add_argument("--commit", default=None, help="commit dir with binding_map_p<k>.json / runs.jsonl / value_check.json")
    ap.add_argument("--pair", type=int, default=0)
    ap.add_argument("--capture", default=None, help="the Match capture dir (match/capture/: log.jsonl.gz + tokens.json): the RunHeader is the lag's "
                    "attestation (refused by name when the field is absent — then an explicit --lag runs the other checks with R0 FAILED), the "
                    "StepStart rows are the engine's executed set (R3c), SampleEnd the committed sample array (the transcript when --tokens is omitted)")
    ap.add_argument("--tokens", default=None, help="JSON file: {\"request_id\": ..., \"tokens\": [...], \"source\": \"...\"} or a bare list — the COMMITTED "
                    "sampled-token array (every SampleEnd incl. a LAG=1 placeholder's discarded sample), never the driver's generated_token_ids")
    ap.add_argument("--eos", default=None, help="comma-separated declared EOS ids (required with --build-request; --build-workload reads stop_rule)")
    ap.add_argument("--cap", type=int, default=None, help="declared max_tokens (required with --build-request)")
    ap.add_argument("--request-id", default="r0", help="the ENGINE's request id (the capture's StepStart rows / tokens.json)")
    ap.add_argument("--commit-request-id", default=None, help="the COMMIT's positional request id in the binding map (binding-map/1 identity.request_id, "
                    "e.g. r0) when it differs from --request-id; nothing is inferred")
    ap.add_argument("--arrive", type=int, default=0)
    ap.add_argument("--boundary-prefix", default=L.BOUNDARY_PREFIX, help="spec-id prefix of the sampling-event node (derived bodies: TokenSelect_v1)")
    ap.add_argument("--header", default=None, help="the run header JSON (RunHeader): LAG is read from profile.engine_kwargs.async_scheduling (RULING 21), "
                    "refused by name when absent")
    ap.add_argument("--lag", type=int, default=None, help="only WITHOUT --header (synthetic records): the declared post-EOS lag (DP-pad-02): 0 or 1")
    ap.add_argument("--lag-source", default=None, help="only WITH --lag: the attested field it was read from; an unattested lag FAILS R0 by name")
    ap.add_argument("--workload-program", default=None, help="R17-1: the row's build_workload/workload_program.json — the Build's DECLARED lag "
                    "(lag.value / stop_rule.lag / stop_rule.eos_lag_rows) R0 cross-checks against the attestation; with --build-request it is "
                    "looked for at <build-request>/../build_workload/workload_program.json when omitted; --build-workload reads the row's own")
    ap.add_argument("--out", required=True)
    ap.add_argument("--correspondence", default=None, help="SPEC v0.5.1 A7 (revb F2): write the IDENTITY-LEVEL correspondence table active-reference <-> "
                    "lifted (verity_vllm.lifting.correspondence) to this path (B1) / this prefix + '.<r<k>>.jsonl.gz' (B>=2); default: beside --out as "
                    "<out>.correspondence[.<rk>].jsonl.gz; its summary goes into the report as A7_correspondence")
    ap.add_argument("--no-correspondence", action="store_true", help="skip the A7 table (structure-only runs)")
    a = ap.parse_args(argv)
    declared: tuple[int | None, str | None] | None = None
    wp_decl = None
    if a.workload_program is not None:
        wp_decl = json.loads(Path(a.workload_program).read_text())
    elif a.build_request is not None and (Path(a.build_request).parent / "build_workload" / "workload_program.json").exists():
        wp_decl = json.loads((Path(a.build_request).parent / "build_workload" / "workload_program.json").read_text())
    if wp_decl is not None:
        declared = L.declared_lag_of_row(wp_decl)
    capture = read_capture(a.capture) if a.capture else None
    if a.header is not None or capture is not None:
        hdr = json.loads(Path(a.header).read_text()) if a.header is not None else capture["header"]
        if hdr is None:
            ap.error("the capture log has no RunHeader")
        try:
            lag, lag_source = L.lag_from_header(hdr)
            if a.lag is not None and a.lag != lag:
                ap.error(f"--lag {a.lag} contradicts the attested field ({L.LAG_FIELD} -> LAG={lag}); the header rules")
        except ValueError as e:
            if a.lag is None:
                ap.error(f"{e}; pass --lag N to run the OTHER checks anyway (R0 FAILS by name; the report is never ok)")
            lag, lag_source = a.lag, f"UNATTESTED ({L.LAG_FIELD} absent from the RunHeader; --lag {a.lag} declared on the command line)"
    elif a.lag is None or a.lag_source is None:
        ap.error("either --header/--capture, or both --lag and --lag-source (an undeclared lag is refused; nothing is defaulted)")
    else:
        lag, lag_source = a.lag, a.lag_source
    if (a.build_request is None) == (a.build_workload is None):
        ap.error("exactly one of --build-request (B1) / --build-workload (B>=2)")
    bmap = run_row = vc = commit_verdict = None
    if a.commit:
        c = Path(a.commit)
        bp = c / f"binding_map_p{a.pair}.json"
        if bp.exists():
            bmap = json.loads(bp.read_text())
        elif bp.with_suffix(".json.gz").exists():
            with gzip.open(bp.with_suffix(".json.gz"), "rt") as f:
                bmap = json.load(f)
        else:
            bmap = None
        rp = c / "runs.jsonl"
        if rp.exists():
            rows = [json.loads(l) for l in rp.read_text().splitlines() if l.strip()]
            run_row = next((r for r in rows if r.get("pair") == a.pair and r.get("arm") == "instrumented"), None)
        vp = c / "value_check.json"
        vc = json.loads(vp.read_text()) if vp.exists() else None
        cv = c / "verdict.json"  # the Commit's verdict: required_manifest.{program_digest, populations, coverage[]} -> R7
        commit_verdict = json.loads(cv.read_text()) if cv.exists() else None
    if a.build_workload is not None:
        if capture is None:
            ap.error("--build-workload needs --capture (the transcripts and the engine rows of every request)")
        row = Path(a.build_workload)
        wp = json.loads((row / "build_workload" / "workload_program.json").read_text())
        if declared is None:  # the row's own declaration (R17-1): always checked at B >= 2
            declared = L.declared_lag_of_row(wp)
        declared_eos = [int(x) for x in ((wp.get("stop_rule") or {}).get("eos_token_ids") or [])]
        if a.eos and declared_eos:  # rev S-r17-29 n2: the EOS set of record is the Program's pinned constant, never a check-time argument
            ap.error(f"--eos {a.eos} refused: workload_program.json stop_rule.eos_token_ids = {declared_eos} declares the EOS set of record; a "
                     "check-time EOS set is F-pm-24's 'extend the set to fit' at the checker (rev S-r17-29 n2)")
        eos = declared_eos or [int(x) for x in (a.eos.split(",") if a.eos else [])]
        eos_source = EOS_SOURCE_OF_RECORD if declared_eos else EOS_SOURCE_GIVEN
        if not eos:
            ap.error("no declared EOS set: neither --eos nor workload_program.json stop_rule.eos_token_ids (nothing is defaulted)")
        uuids = ((wp.get("request_ids") or {}).get("workload_request_ids") or {})
        derived = (wp.get("inputs") or {}).get("derived") or {}
        reqs = []
        for r in wp["requests"]:
            rid = str(r["request_id"])
            if rid not in uuids:
                ap.error(f"{rid}: no engine request id in request_ids.workload_request_ids (the Build's own declared mapping; nothing is inferred)")
            lp, cap = int(r["prompt_len"]), int(r["cap"])
            src_dir = derived.get(f"{lp},{cap - 1}")
            if not src_dir:  # F-pad-26(b): a component the Build never declared is not read by a naming convention -- refused by name
                ap.error(f"{rid}: inputs.derived has no entry '{lp},{cap - 1}' (the Build's own record of the component it composed; the "
                         f"dir-name convention build_request_LP{lp}_T{cap - 1} is NOT a source; nothing is inferred)")
            comp_dir = row / Path(src_dir).name
            if not (comp_dir / "descriptor.json.gz").exists():
                ap.error(f"{rid}: component dir {comp_dir} has no descriptor.json.gz (inputs.derived['{lp},{cap - 1}'] = {src_dir})")
            params = None
            ip = comp_dir / "instances.json.gz"
            if ip.exists():
                with gzip.open(ip, "rt") as f:
                    params = [(str(n), str(t)) for n, t in (json.load(f).get("params") or [])]
            reqs.append({"request_id": uuids[rid], "commit_request_id": rid, "desc": load_descriptor(comp_dir), "cap": cap, "arrive": int(r.get("arrive_step", 0)),
                         "prompt_len": lp, "params": params, "component_dir": str(comp_dir), "workload_digest": wp.get("workload_digest")})
        # F-pad-26(a): the row's own Build facts the W1 recompose must reproduce (workload_program.json is the Build's record, not the tool's)
        row_facts = {"workload_digest": wp.get("workload_digest"), "gates": wp.get("gates"), "component_gates": (wp.get("population") or {}).get("component_gates"),
                     "component_program_digests": {str(r["request_id"]): ((r.get("component") or {}).get("program_digest")) for r in wp["requests"]}}
        rep = check_workload_record(reqs, eos, lag=lag, lag_source=lag_source, bmap=bmap, capture=capture, boundary_prefix=a.boundary_prefix, commit_verdict=commit_verdict,
                                    row_facts=row_facts, declared=declared, eos_source=eos_source)
        rep["eos_source"] = eos_source
        rep["row"] = {"build_workload": str(row), "workload_digest": wp.get("workload_digest"), "manifest_digest": wp.get("manifest_digest"),
                      "requests": [{k: q[k] for k in ("request_id", "commit_request_id", "cap", "arrive", "prompt_len", "component_dir")} for q in reqs],
                      "stop_rule": wp.get("stop_rule"), "padding_scope_of_record": wp.get("padding_scope")}
        if not a.no_correspondence:
            rep["A7_correspondence"] = a7_correspondence_workload(wp, reqs, rep, bmap, commit_verdict, lag=lag,
                                                                  prefix=(a.correspondence or (str(a.out) + ".correspondence")))
            w5_from_binding_level(rep, rep["A7_correspondence"].get("w5_binding_level"))
        Path(a.out).write_text(json.dumps(rep, indent=1, default=str))
        print(json.dumps({k: v.get("ok") for k, v in rep["checks"].items()} | {"requests_ok": [p["ok"] for p in rep["requests"]], "ok": rep["ok"]}))
        return 0 if rep["ok"] else 1
    if a.eos is None or a.cap is None:
        ap.error("--eos and --cap are required with --build-request")
    desc = load_descriptor(a.build_request)
    params = None
    ip = Path(a.build_request) / "instances.json.gz"
    if ip.exists():  # the wrapper's parameter names (the Build's); see check_record's R1 note
        with gzip.open(ip, "rt") as f:
            params = [(str(n), str(t)) for n, t in (json.load(f).get("params") or [])] or None
    if a.tokens is not None:
        tk = json.loads(Path(a.tokens).read_text())
        tokens, src = (tk, "list") if isinstance(tk, list) else (tk["tokens"], str(tk.get("source", "file")))
    elif capture is not None:
        seen: list[str] = []
        for st in capture["steps"].values():
            for rid, _ in st.get("rows") or []:
                if rid not in seen:
                    seen.append(rid)
        if a.request_id not in seen:
            # the default `r0` is the COMMIT's positional id, not an engine uuid: on a one-request capture the request is unambiguous and is
            # taken (named in the report); on several, refuse by name -- never pair by position
            if len(seen) == 1:
                a.commit_request_id = a.commit_request_id or a.request_id
                a.request_id = seen[0]
            else:
                ap.error(f"--request-id {a.request_id!r} occupies no row of the capture; its requests are {seen} -- name one (a B >= 2 capture is never paired by position)")
        cr = capture_request(capture, a.request_id)
        tokens = [cr["sampled"][st] for st in cr["executed"] if st in cr["sampled"]]
        src = "match/capture/log.jsonl.gz SampleEnd sampled_token_ids at the request's steps (every executed step, placeholder included)"
    else:
        ap.error("--tokens is required without --capture")
    eos_b1 = [int(x) for x in a.eos.split(",")]
    declared_eos_b1 = [int(x) for x in (((wp_decl or {}).get("stop_rule") or {}).get("eos_token_ids") or [])]
    if declared_eos_b1 and declared_eos_b1 != eos_b1:  # rev S-r17-29 n2: the row's pinned set rules; a differing --eos is refused by name
        ap.error(f"--eos {a.eos} contradicts the row's workload_program.json stop_rule.eos_token_ids = {declared_eos_b1} (the EOS set of record)")
    eos_source_b1 = EOS_SOURCE_OF_RECORD if declared_eos_b1 else EOS_SOURCE_GIVEN
    rep = check_record(desc, tokens, eos_b1, cap=a.cap, bmap=bmap, request_id=a.request_id, arrive=a.arrive,
                       run_row=run_row, c2_value_check=vc, transcript_source=src, boundary_prefix=a.boundary_prefix, lag=lag, params=params,
                       lag_source=lag_source, capture=capture, commit_request_id=a.commit_request_id, commit_verdict=commit_verdict, declared=declared,
                       eos_source=eos_source_b1)
    rep["eos_source"] = eos_source_b1
    if not a.no_correspondence:
        rep["A7_correspondence"] = a7_correspondence_request(a.build_request, desc, params, rep, bmap, commit_verdict, lag=lag, cap=a.cap, arrive=a.arrive,
                                                             commit_rid=(a.commit_request_id or a.request_id),
                                                             path=(a.correspondence or (str(a.out) + ".correspondence.jsonl.gz")))
    Path(a.out).write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: v.get("ok") for k, v in rep["checks"].items()} | {"ok": rep["ok"]}))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
