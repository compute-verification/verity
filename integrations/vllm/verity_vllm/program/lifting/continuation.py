"""`Continuation_v1` -- the continuation logic of the fixed padded Program.

Two computed Boolean circuit values per TOKEN-GENERATION SLOT `t = 1..C` (`C = max_tokens`; slot 1 = the token produced from the prompt;
slots are NOT engine iterations: chunked prefill advances the state once, an unscheduled request is pending, not inactive):

    e_t = emit_t   -- this slot's token belongs to the request's output
    r_t = run_t    -- this slot performs an active model computation

    e_1 = r_1 = 1
    e_{t+1} = h_t AND e_t AND NOT b_t                    h_t = [t < C]  (construction-time cap term)
    r_{t+1} = e_{t+1}            (A = 0, synchronous)
            = h_t AND e_t        (A = 1, the admitted one-step-ahead async profile)

    b_t    = IsEOS(y_t)                 total over the lifted encoding: [tag = 0] AND [payload in EOS_pinned]; IsEOS(bot) = 0 by the tag GATE
    stop_t = e_t AND b_t                the actual stop; an EOS in the discarded lag slot never restarts anything
    o_t    = Select(e_t, y_t, bot)      the DESIGNATED output (emit, never run: the real-but-discarded async token never leaks out)
    u_{t+1} = Select(r_{t+1}, y_t, bot) the next decode input, from the RAW token
    S_{t+1} = Select(r_t, S_cand, S_t)  persistent state over the existing structured machinery (no K/V copy)

`A` is fixed at BUILD from the RESOLVED profile (the same field the declared LAG reads, through the ONE lag rule -- `A = LAG`), never
from where the rows stop; `n_executed = n (A = 0) / min(n + 1, C) (A = 1)` with `n` served slots is a DERIVED TEST INVARIANT, never an input
that shortens the Program.  This module is the SEMANTIC core: the reference evaluator (`next_flags`), an
independently written three-state reference machine (SERVING / LAG / INACTIVE), the checker's recurrence over a RAW transcript
(`recurrence`), the named invariants (`invariants`), the Match-side checks (`check_transcript`), the slot <-> engine-iteration map
(`slot_map` -- a STATED fact, contiguity + unchunked prefill or FAIL by name), and the derived invariant.  The registered gates
(`LIsEOS<E>_v1`, `LEmitNext_v1`, `LRunNext<A>_v1`) and the construction live in `verity/ml/definitions/registry/lifted.py` (`continuation_chain`); the
lag unit (`LStepLive`/`LSelect` guard chain) is proven to compute exactly `r` / `u` of this rule (`chain_bits`, tested exhaustively).

Indexing: slot `t` <-> the Build's sampling EVENT / step `s = t - 1` (`out[s]`; the lifted tool's `a_r` is the first event with `r = 0`).
Everything here is labelled `lifted-padded` and carries `CONTINUATION_VERSION` beside `LIFTED_RULES_VERSION`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Sequence

from verity.ml.definitions.registry import lifted as L
from verity_vllm.program.lifting import lifted_records

CONTINUATION_VERSION = "continuation-v1"  # every Continuation_v1 result carries it (recorder, builder and Match share one version)
PROFILES = {0: "synchronous (zero lag: run moves with emit)", 1: "admitted one-step-ahead async (run trails emit by exactly one slot after early EOS)"}
PROFILE_FIELD = lifted_records.LAG_FIELD  # RunHeader.profile.engine_kwargs.async_scheduling -- the ONE resolved field A and LAG both read
FAMILIES = ("is_eos", "emit", "run", "out", "in")  # b_t, e_t, r_t, o_t, u_t: Q(P) identities with named families
#: the Definition's statement about randomness.  The sampler of slot `t` EXECUTES iff `r_t = 1` (it is the
#: slot's sampling event; `u_t = bot` makes the whole step bot by strictness, sampler included), so a stochastic Program draws exactly once
#: per executed slot: the lag slot (e, r) = (0, 1) consumes ONE sample after the served EOS, an inactive slot none.  With the counter RNG of
#: record (Philox keyed by (seed, pos), no sequential state; `pos` = the QUERY position whose logits are sampled, the
#: record's own `pos` fact: slot t draws at pos = prompt_len - 1 + (t - 1), the ledger's "draw k at prompt_len - 1 + k") "advances" means "the
#: draw at pos exists": the served draws at pos < prompt_len - 1 + n are unchanged whether or not the lag slot ran.  stoch_recompute carries the
#: accounting; POSITIONS HERE ARE THE LEDGER'S (one convention on the record).
RNG_STATEMENT = ("RNG advances iff r_t: the sampler of slot t executes iff r_t = 1; the lag slot (0, 1) consumes exactly one draw after the served "
                 "EOS at pos = prompt_len - 1 + n (n served slots; pos = the query position of record, slot t at prompt_len - 1 + (t - 1)); an "
                 "inactive slot (0, 0) draws nothing; served draws are unchanged by the lag slot (counter RNG keyed by (seed, pos), no "
                 "sequential state)")


def rng_accounting(slots: Sequence["Slot"], *, prompt_len: int | None = None) -> dict[str, Any]:
    """The named RNG field: how many draws the rule says the Program makes, and where the post-EOS one is (position relative to the prompt, and
    absolute when `prompt_len` is known).  A statement about the Definition; the record-side count is stoch_recompute's."""
    exe = executed_slots(slots)
    lag = lag_slots(slots)
    return {"statement": RNG_STATEMENT, "draws": len(exe), "draw_slots": exe, "draws_after_served_eos": len(lag),
            "lag_draw_slot": (lag[0] if lag else None),
            "lag_draw_pos": (None if not lag or prompt_len is None else int(prompt_len) - 1 + (lag[0] - 1)),  # the ledger's pos
            "inactive_slots_draw_nothing": [s.t for s in slots if not s.run]}


def check_profile(A: int) -> int:
    """`A in {0, 1}` and nothing else: the same refusal as the lag rule's `check_lag` (one rule).  `A` is the resolved profile, never a fit."""
    return L.check_lag(A)


def resolve_profile(header: dict) -> tuple[int, str]:
    """`A` from the record's RESOLVED profile (`RunHeader.profile.engine_kwargs.async_scheduling`, True -> 1, False -> 0), through the lag rule's
    `lag_from_header` -- the one rule; an absent / non-boolean attestation is refused by name, never defaulted."""
    return lifted_records.lag_from_header(header)


def declared_profile(workload_program: dict | None) -> tuple[int | None, str | None]:
    """`A` as the Build DECLARED it (the Program's digest-bound `lag=<n>` static and the sidecar fields; the lag rule's `declared_lag_of_row`)."""
    return lifted_records.declared_lag_of_row(workload_program)


# ---------------------------------------------------------------------------------------------------------
# the reference evaluator and the independent three-state machine
# ---------------------------------------------------------------------------------------------------------


def next_flags(*, emit: bool, token_is_eos: bool, has_next_slot: bool, async_one_step: bool) -> tuple[bool, bool]:
    """The reference evaluator for the Boolean transition (not the Definition): `(e_{t+1}, r_{t+1})` from `(e_t, b_t, h_t, A)`."""
    next_emit = has_next_slot and emit and not token_is_eos
    next_run = (has_next_slot and emit) if async_one_step else next_emit
    return next_emit, next_run


class State(Enum):
    """The independently written three-state reference: SERVING -> SERVING | INACTIVE (sync) | LAG (async); LAG -> INACTIVE;
    INACTIVE -> INACTIVE; the cap overrides everything (no slot after C)."""
    SERVING = "serving"
    LAG = "lag"
    INACTIVE = "inactive"


STATE_BITS = {State.SERVING: (True, True), State.LAG: (False, True), State.INACTIVE: (False, False)}  # (emit, run); (1, 0) has no state


def three_state_step(state: State, *, token_is_eos: bool, has_next_slot: bool, async_one_step: bool) -> State:
    """One transition of the reference machine, written from the states' meaning rather than from the equations."""
    if not has_next_slot:
        return State.INACTIVE  # the cap: nothing after slot C in either profile
    if state is State.SERVING:
        if not token_is_eos:
            return State.SERVING
        return State.LAG if async_one_step else State.INACTIVE
    return State.INACTIVE  # LAG -> INACTIVE (the lag token, EOS or not, changes nothing); INACTIVE absorbing


def three_state_run(is_eos: Sequence[bool], *, cap: int, A: int) -> list[tuple[bool, bool]]:
    """The reference machine over a full pattern `is_eos[t-1] = IsEOS(y_t)` for `t = 1..cap` (a slot whose token is never produced reads
    False -- its bit is irrelevant, the machine is already INACTIVE); returns `(e_t, r_t)` for `t = 1..cap`."""
    check_profile(A)
    st, out = State.SERVING, []
    for t in range(1, cap + 1):
        out.append(STATE_BITS[st])
        st = three_state_step(st, token_is_eos=bool(is_eos[t - 1]) if t - 1 < len(is_eos) else False, has_next_slot=t < cap, async_one_step=A == 1)
    return out


# ---------------------------------------------------------------------------------------------------------
# IsEOS, total over the lifted encoding
# ---------------------------------------------------------------------------------------------------------


def is_eos_total(w: int, x: int, eos_set: Iterable[int]) -> bool:
    """`IsEOS` as a TOTAL Definition over the lifted `w+1`-bit word: `[tag(x) = 0] AND [payload(x) in EOS_pinned]` -- the tag test is a gate,
    so `IsEOS(bot) = 0` is not a convention; a noncanonical `1||nonzero` word is not EOS either (never normalized, never matched)."""
    return L.is_active(w, x) and L.payload(w, x) in {int(e) for e in eos_set}


# ---------------------------------------------------------------------------------------------------------
# the checker's recurrence over a RAW transcript and the derived invariant
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Slot:
    """One token-generation slot of one request: the two bits, the raw token (None = the slot produced none), `is_eos`, `stop`, the designated
    output (`None` = bot), the next slot's input (`None` = bot; absent for the last slot)."""
    t: int
    emit: bool
    run: bool
    raw: int | None
    is_eos: bool
    stop: bool
    out: int | None
    next_emit: bool
    next_run: bool
    next_in: int | None

    @property
    def kind(self) -> str:
        return "active" if self.emit and self.run else ("lag" if self.run else ("inactive" if not self.emit else "UNREACHABLE(1,0)"))


def recurrence(raw: Sequence[int | None], *, cap: int, A: int, eos: Iterable[int], w: int = 32, lifted_words: bool = False) -> list[Slot]:
    """The bits COMPUTED from the Program's rule and the record's RAW transcript `raw[t-1] = y_t` (the sampled token of every EXECUTED slot,
    the lag slot's discarded sample included -- the fold has it, tokens.json has only the served ones; `None` where the record produced no
    token).  `lifted_words`: `raw` are `w+1`-bit words (bot allowed) rather than plain ids.  Nothing is judged here (see `check_transcript`):
    a slot with `r_t = 1` and no raw token reads `b_t = 0` and is named there as a deleted step; a token at `r_t = 0` is named there as
    execution past the stop.  Returns exactly `cap` slots (the population is fixed; the inactive suffix is present as slots with (0, 0))."""
    check_profile(A)
    if cap < 1:
        raise ValueError(f"C = {cap}: a request declares at least one token-generation slot")
    if len(raw) > cap:
        raise ValueError(f"{len(raw)} raw tokens for a cap of {cap}: more slots than the Program has (named, never absorbed)")
    eos_set = {int(e) for e in eos}
    e, r, out = True, True, []
    for t in range(1, cap + 1):
        y = raw[t - 1] if t - 1 < len(raw) else None
        if y is not None and not lifted_words:
            y = L.enc_active(w, int(y))
        b = (y is not None) and r and is_eos_total(w, int(y), eos_set)  # an inactive token is not EOS (IsEOS total)
        ne, nr = next_flags(emit=e, token_is_eos=b, has_next_slot=t < cap, async_one_step=A == 1)
        yv = (L.payload(w, int(y)) if L.is_active(w, int(y)) else None) if y is not None else None
        out.append(Slot(t=t, emit=e, run=r, raw=yv, is_eos=bool(b), stop=bool(e and b), out=(yv if e else None),
                        next_emit=ne, next_run=nr, next_in=((yv if nr else None) if t < cap else None)))
        e, r = ne, nr
    return out


def bits(slots: Sequence[Slot]) -> list[tuple[bool, bool]]:
    return [(s.emit, s.run) for s in slots]


def executed_slots(slots: Sequence[Slot]) -> list[int]:
    """`{t : r_t = 1}` -- the executed set the Program's rule PREDICTS (Match compares the record's rows against it)."""
    return [s.t for s in slots if s.run]


def served_output(slots: Sequence[Slot]) -> list[int | None]:
    """`(y_t : e_t = 1)` as a LIST, EOS included, position by position."""
    return [s.raw for s in slots if s.emit]


def lag_slots(slots: Sequence[Slot]) -> list[int]:
    """The slots with `(e, r) = (0, 1)`: the one permitted extra executed row under A = 1, none under A = 0."""
    return [s.t for s in slots if s.run and not s.emit]


def n_executed_derived(n_served: int, *, cap: int, A: int) -> int:
    """The DERIVED test invariant: `n_executed = n` (A = 0) / `min(n + 1, C)` (A = 1) for `n` served slots including the EOS.
    Equal to the Commit's executed-prefix rule of record and to the lag rule's `expected_served(a_r, cap, lag)` with `a_r = n`; never an input."""
    check_profile(A)
    if not (1 <= n_served <= cap):
        raise ValueError(f"n_served = {n_served} outside [1, C = {cap}]")
    return n_served if A == 0 else min(n_served + 1, cap)


def invariants(slots: Sequence[Slot], *, cap: int, A: int) -> dict[str, bool]:
    """The named invariants, each a fact about one slot table."""
    check_profile(A)
    bs = bits(slots)
    e = [x for x, _ in bs]
    r = [x for _, x in bs]
    last = slots[-1]
    return {
        "length_is_cap": len(slots) == cap,
        "initial_bits_one": bool(e[0] and r[0]),
        "emit_implies_run": all((not x) or y for x, y in bs),
        "no_unreachable_1_0": not any(x and not y for x, y in bs),
        "emit_never_returns": all(not (e[i] and not e[i - 1]) for i in range(1, len(e))),
        "run_never_restarts": all(not (r[i] and not r[i - 1]) for i in range(1, len(r))),
        "at_most_one_lag_slot": len(lag_slots(slots)) <= 1,
        "no_lag_slot_when_synchronous": A == 1 or not lag_slots(slots),
        "final_next_bits_false": not last.next_emit and not last.next_run,
        "executed_is_derived_invariant": len(executed_slots(slots)) == n_executed_derived(max(1, len(served_output(slots))), cap=cap, A=A),
        "stop_only_when_emitting": all((not s.stop) or s.emit for s in slots),
        "at_most_one_stop": sum(1 for s in slots if s.stop) <= 1,
        "designated_output_is_emit_prefix": all((s.out is not None) == (s.emit and s.raw is not None) for s in slots),
    }


# ---------------------------------------------------------------------------------------------------------
# one mechanism: the executed prefix the Commit's consumers count on, READ FROM THE RECURRENCE (first EOS, not last)
# ---------------------------------------------------------------------------------------------------------


def executed_prefix_by_rule(served: Sequence[int], *, cap: int, A: int, eos: Iterable[int]) -> dict[str, Any]:
    """The per-request block of the Commit's `executed_prefix_of_record` (`verity_vllm/commit/executed_prefix.py`) computed FROM the rule: the
    recurrence over the SERVED list as the raw prefix.  `executed_prefix_len = |{t : r_t = 1}|`, `eos_step` = the FIRST EOS's request-relative
    row, `lag_steps` = the (0, 1) slot(s) as request-relative rows (`t - 1`).  A served token AFTER the first EOS is `served_past_first_eos`
    (the rule stops at the first EOS; a last-token test would accept `[EOS, x, EOS]`); a last served token outside the pinned
    set on an early stop and a served list over the cap are named too -- each a problem, `ok` their absence.  No LAG default: `A` is the
    declared / attested profile (`check_profile`).  The Commit's one code path; nothing here reads a trace."""
    check_profile(A)
    eos_set = {int(e) for e in eos}
    a = len(served)
    problems: list[str] = []
    if a == 0:
        return {"ok": False, "served_output_len": 0, "executed_prefix_len": cap, "why": "served no token at all: not a stop of the Program's rule", "problems": ["served no token at all"]}
    if a > cap:
        problems.append(f"served {a} tokens over the declared cap {cap}")
    first_eos = next((i for i, t in enumerate(served) if int(t) in eos_set), None)
    past = [i for i in range((first_eos + 1) if first_eos is not None else a, a)]
    if past:
        problems.append(f"served past an EOS at index {first_eos}: positions {past} (the rule stops at the FIRST EOS)")
    early = a < cap
    if early and first_eos is None:
        problems.append(f"served {a} of its cap {cap} but no served token is a declared EOS id {sorted(eos_set)[:8]}")
    slots = recurrence([int(t) for t in served[:cap]], cap=cap, A=A, eos=eos_set) if a <= cap else None
    exe = executed_slots(slots) if slots is not None else list(range(1, cap + 1))
    lag = lag_slots(slots) if slots is not None else []
    stop = (first_eos is not None) and not past and a <= cap
    return {"ok": not problems, "rule": "executed_prefix_len = |{t : r_t = 1}| of Continuation_v1's recurrence over the served list (first EOS stops; A = LAG)",
            "A": A, "cap": cap, "served_output_len": a, "executed_prefix_len": len(exe), "cap_exhausted": (a == cap and first_eos is None) if not problems else None,
            "eos_stop": (stop if not problems else None), "first_eos_index": first_eos, "served_past_first_eos": past,
            "last_served_token": int(served[-1]), "eos_step": (first_eos if stop else None), "lag_steps": [t - 1 for t in lag],
            "derived_invariant_n_executed": (n_executed_derived(a, cap=cap, A=A) if (1 <= a <= cap and not past) else None),
            "problems": problems, "why": ("; ".join(problems) or None)}


# ---------------------------------------------------------------------------------------------------------
# slots <-> engine iterations: a STATED fact, or FAIL by name
# ---------------------------------------------------------------------------------------------------------


def slot_map(executed_engine_steps: Sequence[int], *, arrive_step: int, prefill_steps: Sequence[int] | None = None,
             chunked_prefill_observed: bool | None = None) -> dict[str, Any]:
    """The map from a request's executed ENGINE steps to its token-generation SLOTS: one iteration per slot, contiguous from `arrive_step`
    (`slot t <-> engine step arrive + t - 1`).  Chunked prefill ("advances the state once") and an unscheduled iteration ("pending, not
    inactive") break the identity map and the two bits cannot express them: both are NAMED here (`ok False`), never absorbed -- the workload's
    applicability attests contiguity + unchunked prefill on the pinned profile; a preempted request is a Match FAIL by name."""
    steps = [int(s) for s in executed_engine_steps]
    problems: list[str] = []
    if steps and steps[0] != arrive_step:
        problems.append(f"first executed engine step {steps[0]} != arrive_step {arrive_step}: the request's slot 1 is not at its arrival")
    gaps = [(a, b) for a, b in zip(steps, steps[1:]) if b != a + 1]
    if gaps:
        problems.append(f"executed engine steps are not contiguous ({len(gaps)} gap(s), first {gaps[0]}): an unscheduled / preempted slot is "
                        "pending, not inactive -- the two bits cannot express it; FAIL by name")
    if chunked_prefill_observed:
        problems.append("chunked prefill observed: several iterations advance slot 1 once; the one-iteration-per-slot map does not hold")
    if prefill_steps is not None and len(prefill_steps) > 1:
        problems.append(f"prefill spans {len(prefill_steps)} engine steps {list(prefill_steps)}: not one iteration per slot")
    # the two prefill sub-checks read the capture's OWN attestation (tokens.json timing) and are VACUOUS when it is absent --
    # said so by name, never counted as a pass; the POSITIVE attestation (unchunked prefill on the pinned profile) is the workload's applicability
    vacuous = ([] if chunked_prefill_observed is not None else ["chunked_prefill: no attestation in the capture (tokens.json timing.chunk_attribution absent)"]) + \
              ([] if prefill_steps is not None else ["prefill_steps: no attestation in the capture (tokens.json timing.requests[].prefill_steps absent)"])
    return {"ok": not problems, "rule": "slot t <-> engine step arrive_step + t - 1 (one iteration per slot, contiguous, unchunked prefill)",
            "arrive_step": arrive_step, "executed_engine_steps": steps, "slots": [s - arrive_step + 1 for s in steps] if not problems else None,
            "problems": problems,
            "chunked_prefill_attested": ("absent" if chunked_prefill_observed is None else ("observed" if chunked_prefill_observed else "not-observed")),
            "prefill_steps_attested": ("absent" if prefill_steps is None else [int(x) for x in prefill_steps]),
            "vacuous_by_absence": vacuous,
            "positive_attestation": "the workload's applicability (contiguity + unchunked prefill on the pinned profile) -- not this map's to give"}


def expected_draw_positions(served: Sequence[int], *, cap: int, A: int, eos: Iterable[int], prompt_len: int) -> dict[str, Any]:
    """The sample positions the RULE prescribes, `{L - 1 + (t - 1) : r_t = 1}` with `L = prompt_len` (the record's
    `pos` fact = the query position whose logits are sampled: the sample ledger's 'draw k at pos = prompt_len - 1 + k', slot t = k + 1;
    the counter RNG is keyed by `(seed, pos)`), read from the recurrence over the SERVED list (the one mechanism).
    An ORDERED list: the served slots' draws, then the one lag draw (A = 1, EOS before the cap), nothing after -- so an over-cap draw without
    an EOS, a second post-EOS draw, or a sample at an inactive slot is an EXTRA position by name, a missing served draw a MISSING one
    (`compare_draw_positions`).  `A in {0, 1}` (check_profile) -- a lag >= 2 has no admitted profile."""
    check_profile(A)
    if prompt_len < 1:
        raise ValueError(f"prompt_len = {prompt_len}: a request has at least one prompt token")
    slots = recurrence([int(t) for t in served][:cap], cap=cap, A=A, eos=eos)
    run = [s.t for s in slots if s.run]
    lag = lag_slots(slots)
    L = int(prompt_len)
    pos = lambda t: L - 1 + (t - 1)  # noqa: E731  -- slot t's query position, the ledger's convention
    return {"rule": "expected draw positions = {L - 1 + (t - 1) : r_t = 1}, L = prompt_len, pos = the query position of record (Continuation_v1 "
                    "recurrence over the served list; A = LAG; the draw ledger's convention)",
            "prompt_len": int(prompt_len), "A": A, "cap": cap, "n_served": len(served),
            # the recurrence runs on served[:cap]; a served list OVER the cap is named here (and its samples past the cap
            # are EXTRA positions in the compare), never truncated silently
            "n_served_over_cap": max(0, len(served) - cap),
            "positions": [pos(t) for t in run], "slots": run, "n": len(run),
            "lag_positions": [pos(t) for t in lag],
            "served_positions": [pos(s.t) for s in slots if s.emit],
            "first_forbidden_position": (pos(run[-1]) + 1) if run else pos(1)}


def compare_draw_positions(expected: Sequence[int], drawn: Sequence[int]) -> dict[str, Any]:
    """`drawn` = the positions the record's ledger shows a sample at (stoch_recompute's `draw_accounting`), in ledger order.  Every extra / missing / repeated /
    out-of-order position is a problem BY NAME; `ok` is their absence -- 'RNG advances iff r_t' measured literally."""
    exp = [int(p) for p in expected]
    got = [int(p) for p in drawn]
    problems: list[str] = []
    extra = sorted(set(got) - set(exp))
    missing = sorted(set(exp) - set(got))
    dup = sorted({p for p in got if got.count(p) > 1})
    if extra:
        problems.append(f"draws at positions {extra[:8]} where r_t = 0 (over the cap / past the lag slot / at an inactive slot): the RNG advanced without r_t")
    if missing:
        problems.append(f"no draw at positions {missing[:8]} where r_t = 1: an executed slot without its sampling event")
    if dup:
        problems.append(f"positions {dup[:8]} drawn more than once: two draws in one slot")
    if not extra and not missing and not dup and got != exp:
        problems.append(f"draws out of order: ledger {got[:8]} vs the rule's {exp[:8]}")
    return {"ok": not problems, "expected": exp, "drawn": got, "extra": extra, "missing": missing, "repeated": dup, "problems": problems}


# ---------------------------------------------------------------------------------------------------------
# what Match CHECKS per request, computed from the Program's rule + the raw transcript
# ---------------------------------------------------------------------------------------------------------


def check_transcript(*, executed_slots_of_record: Sequence[int], raw_by_slot: dict[int, int], served_of_record: Sequence[int], cap: int,
                     A: int, eos: Iterable[int], w: int = 32) -> dict[str, Any]:
    """(a) executed slots of record == `{t : r_t = 1}` exactly -- a row at `r_t = 0` is "execution past the stop", a missing
    row at `r_t = 1` "a deleted step"; (b) served output == `(y_t : e_t = 1)` as a list, EOS included, position by position; (c) the lag
    slot's token is NOT in the served output; (d) the inactive suffix has no rows (its identities stay in the population); (e)
    `stop_t = e_t AND IsEOS(y_t)` under the PINNED set: a served early stop on a token outside the set is FAIL, never extend the
    set; (5) the derived invariant `n_exec` checked AFTER (a).  Bits are computed from the rule + the raw transcript; nothing is supplied by
    the prover, nothing inferred from where the rows stop.  Every failure is named; `ok` is the conjunction."""
    check_profile(A)
    exe = sorted(int(t) for t in executed_slots_of_record)
    raw: list[int | None] = [raw_by_slot.get(t) for t in range(1, cap + 1)]
    slots = recurrence(raw, cap=cap, A=A, eos=eos, w=w)
    want = executed_slots(slots)
    past = [t for t in exe if t not in want]
    deleted = [t for t in want if t not in exe]
    served_pred = served_output(slots)
    served_rec = [int(x) for x in served_of_record]
    first_diff = next((i for i, (a, b) in enumerate(zip(served_pred, served_rec)) if a != b), None)
    if first_diff is None and len(served_pred) != len(served_rec):
        first_diff = min(len(served_pred), len(served_rec))
    lag = lag_slots(slots)
    lag_tok = [raw_by_slot.get(t) for t in lag]
    eos_set = {int(e) for e in eos}
    # (e): a served stop is the LAST served token being in the pinned set (or the cap exhausted); a stop before the cap on a token outside it fails
    stopped_before_cap = len(served_rec) < cap
    last_in_set = bool(served_rec) and served_rec[-1] in eos_set
    checks = {
        "a_executed_set_equals_run_slots": {"ok": not past and not deleted, "predicted": want, "of_record": exe,
                                            "execution_past_the_stop": past, "deleted_steps": deleted,
                                            "why": (None if not past and not deleted else
                                                    (f"rows at slots {past} where r_t = 0: execution past the stop; " if past else "") +
                                                    (f"no rows at slots {deleted} where r_t = 1: a deleted step" if deleted else ""))},
        "b_served_equals_emit_tokens": {"ok": first_diff is None, "predicted": served_pred, "of_record": served_rec, "first_differing_position": first_diff,
                                        "why": None if first_diff is None else f"served output differs from (y_t : e_t = 1) at position {first_diff}"},
        # (c): the served list is positional (slot t <-> position t-1 over the emit prefix); a lag token served shows as an entry past the prefix
        "c_lag_token_not_served": {"ok": len(served_rec) <= len(served_pred), "lag_slots": lag, "lag_tokens": lag_tok,
                                   "served_entries_past_emit_prefix": max(0, len(served_rec) - len(served_pred))},
        "d_inactive_suffix_has_no_rows": {"ok": not any(t in exe for t in range(1, cap + 1) if not slots[t - 1].run),
                                          "inactive_slots": [s.t for s in slots if not s.run], "note": "identities of the inactive suffix stay in the population"},
        "e_stop_under_pinned_set": {"ok": (not stopped_before_cap) or last_in_set, "stopped_before_cap": stopped_before_cap, "last_served_in_eos_set": last_in_set,
                                    "eos_set": sorted(eos_set), "why": None if (not stopped_before_cap) or last_in_set else
                                    "served early stop on a token outside the pinned EOS set: FAIL, never extend the set to fit"},
    }
    n_srv = len(served_rec)
    derived = n_executed_derived(n_srv, cap=cap, A=A) if 1 <= n_srv <= cap else None
    checks["f_n_executed_derived_invariant"] = {"ok": derived is not None and derived == len(exe) == len(want), "n_served": n_srv, "n_executed_of_record": len(exe),
                                                "n_executed_predicted": len(want), "derived": derived, "rule": "n (A = 0) / min(n + 1, C) (A = 1); checked AFTER (a), never an input"}
    inv = invariants(slots, cap=cap, A=A)
    return {"version": CONTINUATION_VERSION, "rules_version": L.LIFTED_RULES_VERSION, "label": L.LABEL, "A": A, "profile": PROFILES[A], "cap": cap,
            "ok": all(c["ok"] for c in checks.values()) and all(inv.values()), "checks": checks, "invariants": inv,
            "slots": [{"t": s.t, "emit": int(s.emit), "run": int(s.run), "kind": s.kind, "raw": s.raw, "is_eos": int(s.is_eos), "stop": int(s.stop),
                       "out": s.out, "next_in": s.next_in} for s in slots],
            "failed": [k for k, c in checks.items() if not c["ok"]] + [f"invariant:{k}" for k, v in inv.items() if not v]}


def wrong_profile_negative(*, executed_slots_of_record: Sequence[int], raw_by_slot: dict[int, int], served_of_record: Sequence[int], cap: int,
                           A_program: int | None = None, eos: Iterable[int], A_attested: int | None = None) -> dict[str, Any]:
    """The other profile against the same transcript (a wrong profile is an applicability or Match failure, never a fitted explanation):
    the rule under `1 - A` must FAIL by name on any record that exercised the lag (an EOS stop before the cap); on a cap-exhausting record both
    profiles agree and the negative is 'not exercised' (stated, not claimed).  `A_program` is the PROGRAM's fixed profile (the Build's declared
    lag; the keyword `A_attested` is accepted as its older name, the report carries only `A_program`).  When the other profile does NOT fail on a lag-exercising record, the record fits the
    other Program -- the GM-01 G1+G6 situation (a record run under another scheduler than the one declared), named here as
    `record_fits_other_profile`."""
    if A_program is None:
        A_program = A_attested
    if A_program is None:
        raise TypeError("wrong_profile_negative: A_program (the Program's fixed profile) is required")
    other = 1 - check_profile(A_program)
    res = check_transcript(executed_slots_of_record=executed_slots_of_record, raw_by_slot=raw_by_slot, served_of_record=served_of_record, cap=cap, A=other, eos=eos)
    exercised = len(served_of_record) < cap
    fails = (not res["ok"]) if exercised else None
    return {"A_program": A_program, "A_tried": other, "exercised": exercised, "fails_by_name": fails,
            "record_fits_other_profile": (res["ok"] if exercised else None),
            "failed": res["failed"], "note": (("the wrong profile is refused by the executed-set check, not fitted" if fails else
                                               f"the record PASSES under A = {other}, the profile the Program was NOT built for: it is an execution of "
                                               f"another scheduler than the declared one (refused by name at GM-01's G1 and G6, never re-fitted)") if exercised else
                                              "cap exhausted: both profiles predict C executed slots; the lag is not exercised on this record")}


# ---------------------------------------------------------------------------------------------------------
# the lag unit computes exactly r / u of this rule
# ---------------------------------------------------------------------------------------------------------


def chain_bits(raw_words: Sequence[int], *, cap: int, lag: int, eos: Sequence[int], w: int = 32) -> dict[str, list]:
    """The lag guard chain of `LServe_v1` / `lift_request_component` SIMULATED with the registered evaluators (`LStepLive<E>_v1`,
    `LSelect<w>_v1`, `LBot<w>_v1`): steps `1..LAG` unguarded; `live_s = LStepLive(g_s, eos_set)` with `g_s = out[s-1]` (LAG = 0) or
    `tok_in_{s-1}` (LAG = 1); `tok_in_s = LSelect(live_s, out[s-1], bot)`; `out[s]` bot by strictness when `tok_in_s` is bot.  `raw_words[s]`
    is the word the sampler WOULD produce at event `s` if it runs (the transcript the record would commit).  Returns `run[t]` for slots
    `t = 1..cap` (`run_1 = 1`, `run_{s+1} = live_s`) and `in[t]` (`in_1 = prompt`, `in_{s+1} = tok_in_s`) -- to be compared with `recurrence`."""
    E = len(eos)
    step_live, sel, bot = L.l_step_live(E), L.l_select(w), L.enc_bot(w)
    eos_words = [L.enc_active(w, int(e)) for e in eos]
    L.check_lag(lag)
    out = [raw_words[0]]
    run, ins = [True], [None]
    gate = out[0]
    for s in range(1, cap):
        prev = out[s - 1]
        if s <= lag:
            live, tok_in = 1, prev
        else:
            live = step_live.evaluate(gate, *eos_words)
            tok_in = sel.evaluate(live, prev, bot)
        gate = tok_in if lag else None  # LAG = 1 gates on the fed token; LAG = 0 on out[s] (set below)
        y = raw_words[s] if (tok_in != bot and s < len(raw_words)) else bot  # strictness: a bot input yields a bot sample
        out.append(y)
        if not lag:
            gate = y
        run.append(bool(live == 1))
        ins.append(None if tok_in == bot else tok_in)
    return {"run": run, "in": ins, "out": out}
