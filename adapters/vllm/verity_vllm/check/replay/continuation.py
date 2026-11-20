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
that shortens the Program.  This module holds the reference evaluator (`next_flags`), the checker's recurrence over a RAW
transcript (`recurrence`), the executed prefix the Commit's consumers count on (`executed_prefix_by_rule`) and the sample
positions the rule prescribes (`expected_sample_positions`).

Indexing: slot `t` <-> the Build's sampling EVENT / step `s = t - 1` (`out[s]`).
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

LAGS_IMPLEMENTED = (0, 1)  # the engine's post-EOS placeholder rows (vLLM async scheduling: 1); a DECLARED static, no default


def _is_active(w: int, x: int) -> bool:
    """`x` is a `w+1`-bit word whose tag bit `w` is 0."""
    return 0 <= x < (1 << (w + 1)) and (x >> w) & 1 == 0


def _payload(w: int, x: int) -> int:
    return x & ((1 << w) - 1)


def _enc_active(w: int, v: int) -> int:
    if not (0 <= v < (1 << w)):
        raise ValueError(f"payload {v:#x} is not a {w}-bit word")
    return v


def check_profile(A: int) -> int:
    """`A in {0, 1}` and nothing else: the serving profile's post-EOS lag is a declared static -- `0` (the step after the EOS is
    never executed) or `1` (vLLM async scheduling: the EOS-stopped request is fed forward once more, its sample discarded).  Neither
    is a default; anything else is refused by name."""
    if A not in LAGS_IMPLEMENTED:
        raise ValueError(f"LAG={A!r}: the profile's post-EOS lag must be declared as one of {LAGS_IMPLEMENTED}; nothing is assumed")
    return int(A)


# ---------------------------------------------------------------------------------------------------------
# the reference evaluator
# ---------------------------------------------------------------------------------------------------------


def next_flags(*, emit: bool, token_is_eos: bool, has_next_slot: bool, async_one_step: bool) -> tuple[bool, bool]:
    """The reference evaluator for the Boolean transition (not the Definition): `(e_{t+1}, r_{t+1})` from `(e_t, b_t, h_t, A)`."""
    next_emit = has_next_slot and emit and not token_is_eos
    next_run = (has_next_slot and emit) if async_one_step else next_emit
    return next_emit, next_run


# ---------------------------------------------------------------------------------------------------------
# IsEOS, total over the lifted encoding
# ---------------------------------------------------------------------------------------------------------


def is_eos_total(w: int, x: int, eos_set: Iterable[int]) -> bool:
    """`IsEOS` as a TOTAL Definition over the lifted `w+1`-bit word: `[tag(x) = 0] AND [payload(x) in EOS_pinned]` -- the tag test is a gate,
    so `IsEOS(bot) = 0` is not a convention; a noncanonical `1||nonzero` word is not EOS either (never normalized, never matched)."""
    return _is_active(w, x) and _payload(w, x) in {int(e) for e in eos_set}


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


def recurrence(raw: Sequence[int | None], *, cap: int, A: int, eos: Iterable[int], w: int = 32) -> list[Slot]:
    """The bits COMPUTED from the Program's rule and the record's RAW transcript `raw[t-1] = y_t` (the sampled token of every EXECUTED slot,
    the lag slot's discarded sample included -- the fold has it, tokens.json has only the served ones; `None` where the record produced no
    token).  Nothing is judged here: a slot with `r_t = 1` and no raw token reads `b_t = 0` (a deleted step), and a token at
    `r_t = 0` is execution past the stop; Match's `chronology` and the replay's `stoch_recompute.sample_accounting` refuse both.
    Returns exactly `cap` slots (the population is fixed; the inactive suffix is present as slots with (0, 0))."""
    check_profile(A)
    if cap < 1:
        raise ValueError(f"C = {cap}: a request declares at least one token-generation slot")
    if len(raw) > cap:
        raise ValueError(f"{len(raw)} raw tokens for a cap of {cap}: more slots than the Program has (named, never absorbed)")
    eos_set = {int(e) for e in eos}
    e, r, out = True, True, []
    for t in range(1, cap + 1):
        y = raw[t - 1] if t - 1 < len(raw) else None
        if y is not None:
            y = _enc_active(w, int(y))
        b = (y is not None) and r and is_eos_total(w, int(y), eos_set)  # an inactive token is not EOS (IsEOS total)
        ne, nr = next_flags(emit=e, token_is_eos=b, has_next_slot=t < cap, async_one_step=A == 1)
        yv = (_payload(w, int(y)) if _is_active(w, int(y)) else None) if y is not None else None
        out.append(Slot(t=t, emit=e, run=r, raw=yv, is_eos=bool(b), stop=bool(e and b), out=(yv if e else None),
                        next_emit=ne, next_run=nr, next_in=((yv if nr else None) if t < cap else None)))
        e, r = ne, nr
    return out


def executed_slots(slots: Sequence[Slot]) -> list[int]:
    """`{t : r_t = 1}` -- the executed set the Program's rule PREDICTS (Match compares the record's rows against it)."""
    return [s.t for s in slots if s.run]


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
# the sample positions the rule prescribes, and the ledger's compared with them
# ---------------------------------------------------------------------------------------------------------


def expected_sample_positions(served: Sequence[int], *, cap: int, A: int, eos: Iterable[int], prompt_len: int) -> dict[str, Any]:
    """The sample positions the RULE prescribes, `{L - 1 + (t - 1) : r_t = 1}` with `L = prompt_len` (the record's
    `pos` fact = the query position whose logits are sampled: the sample ledger's 'sample k at pos = prompt_len - 1 + k', slot t = k + 1;
    the counter RNG is keyed by `(seed, pos)`), read from the recurrence over the SERVED list (the one mechanism).
    An ORDERED list: the served slots' samples, then the one lag sample (A = 1, EOS before the cap), nothing after -- so a sample past the cap without
    an EOS, a second post-EOS sample, or a sample at an inactive slot is an EXTRA position by name, a missing served sample a MISSING one
    (`compare_sample_positions`).  `A in {0, 1}` (check_profile) -- a lag >= 2 has no admitted profile."""
    check_profile(A)
    if prompt_len < 1:
        raise ValueError(f"prompt_len = {prompt_len}: a request has at least one prompt token")
    slots = recurrence([int(t) for t in served][:cap], cap=cap, A=A, eos=eos)
    run = [s.t for s in slots if s.run]
    lag = lag_slots(slots)
    L = int(prompt_len)
    pos = lambda t: L - 1 + (t - 1)  # noqa: E731  -- slot t's query position, the ledger's convention
    return {"rule": "expected sample positions = {L - 1 + (t - 1) : r_t = 1}, L = prompt_len, pos = the query position of record (Continuation_v1 "
                    "recurrence over the served list; A = LAG; the sample ledger's convention)",
            "prompt_len": int(prompt_len), "A": A, "cap": cap, "n_served": len(served),
            # the recurrence runs on served[:cap]; a served list OVER the cap is named here (and its samples past the cap
            # are EXTRA positions in the compare), never truncated silently
            "n_served_past_cap": max(0, len(served) - cap),
            "positions": [pos(t) for t in run], "slots": run, "n": len(run),
            "lag_positions": [pos(t) for t in lag],
            "served_positions": [pos(s.t) for s in slots if s.emit],
            "first_forbidden_position": (pos(run[-1]) + 1) if run else pos(1)}


def compare_sample_positions(expected: Sequence[int], sampled: Sequence[int]) -> dict[str, Any]:
    """`sampled` = the positions the record's ledger shows a sample at (stoch_recompute's `sample_accounting`), in ledger order.  Every extra / missing / repeated /
    out-of-order position is a problem BY NAME; `ok` is their absence -- 'RNG advances iff r_t' measured literally."""
    exp = [int(p) for p in expected]
    got = [int(p) for p in sampled]
    problems: list[str] = []
    extra = sorted(set(got) - set(exp))
    missing = sorted(set(exp) - set(got))
    dup = sorted({p for p in got if got.count(p) > 1})
    if extra:
        problems.append(f"samples at positions {extra[:8]} where r_t = 0 (over the cap / past the lag slot / at an inactive slot): the RNG advanced without r_t")
    if missing:
        problems.append(f"no sample at positions {missing[:8]} where r_t = 1: an executed slot without its sampling event")
    if dup:
        problems.append(f"positions {dup[:8]} sampled more than once: two samples in one slot")
    if not extra and not missing and not dup and got != exp:
        problems.append(f"samples out of order: ledger {got[:8]} vs the rule's {exp[:8]}")
    return {"ok": not problems, "expected": exp, "sampled": got, "extra": extra, "missing": missing, "repeated": dup, "problems": problems}
