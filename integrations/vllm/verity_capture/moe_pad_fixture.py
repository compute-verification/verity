"""Stage 3c fixture engine for the padded MoE block (R17-42) under the declared async lag (R17-1).

`B` requests x `S` decode rows of `LMoeBlock_v1` in ONE Program whose statics are the envelope AND the declared `LAG`
(`PaddedDecodeFixture_v1{B,S,LAG,E,TOPK,VPT,H,I}`): the route of every (request, row) and the stop of every request are
ASSIGNMENTS to that one circuit, never its shape.  Two kinds of inactivity meet here and are kept apart (revb M-1037 t1), both
EXPLICIT ENABLEMENT under R17-43 / LIFTING-SPEC v0.6 (pad's B7; rev M-1064 taken):

  * continuation padding (early termination): rows `s >= executed_prefix_len[t]` are DISABLED rows -- the block's enable `a[t][s]`
    (the row's `Continuation_v1` run bit in a decode Program; here the Program's per-row enable input, which the checker sets FROM THE
    DECLARATION: `a[t][s] = [s < executed_prefix_len[t]]`) is Active(0), and every value the row produces is bot by A9.3 whatever the row
    reads (the row has no token; `x[t][s] = bot` is what the preceding disabled layer would hand it, and the words are the same for any x);
    every slot enable is Active(0) (LEnable(0, bot) = 0); nothing is painted or invented (Stage 2 (c2));
  * unselected experts: at an ENABLED row the `E - TOPK` slots the router did not pick are DISABLED slots (`a_e = LEnable(1, s_e) = 0`)
    INSIDE an enabled block: their routed weight and output words are bot, their input `x` wired in unchanged (A12).

The row enable is the ONE fact rev M-1064 names: a row whose `a[t][s]` disagrees with the declared row set is caught here by the row-set
comparison BEFORE evaluation (the same declaration sets both), and an enabled row fed no token is INV by name (Stage 2 (c2): LEnable(1, bot)
= INV_1) -- never a silent inactive row.

The declared lengths (R17-1) are the record's two facts, never one: `served_output_len[t]` = the tokens served (the EOS token is the
last served one); `executed_prefix_len[t] = served + LAG` when the request stopped at EOS (the post-EOS row is active internal
computation whose sampled token is NOT a served output), `= served` when it exhausted the cap (no row follows the cap).  The checker
takes BOTH from the declaration and compares the trace's executed rows against them: a row executed beyond the declared prefix is
`FAIL:R17-1/LAG-ROW-BEYOND-DECLARED` (a LAG=0 header on a LAG=1 trace), a declared row that was not executed is
`FAIL:R17-1/MISSING-EXECUTED-ROW` (a LAG=1 header on a LAG=0 trace, or an EOS "widened" to absorb the lag row); EOS-stopped requests
whose observed post-EOS rows differ surface as `ALERT:VARIABLE-LAG` beside the failure -- never absorbed, never inferred into the header.
The engine below is a fixture: it EXECUTES `served + lag_engine` rows per EOS request so the checker can be shown catching each
mutation by name.  (v1 of this fixture, before pad's B7, made the continuation rows inactive by `x = bot` falling through the strict
relations; the v2 re-cut makes them DISABLED rows through the enable, as LIFTING-SPEC v0.6 A13 reads the step -- the row-level structure
and the four negatives are unchanged by that, the digest and every disabled row's `a_e` words are new.)

LIMIT (revb S-r17b-45): the checker knows only the two DECLARED lengths, so the PAIR mutation -- served widened to `served + 1` AND `LAG`
re-declared 0 -- yields the same `executed_prefix_len` and PASSES here (a different digest, but a passing one).  Each single-field
negative is caught because the other field is held fixed.  At record level the two fields have two independent sources -- `served`
from tokens.json tied to the fold (F-r16-24), `LAG` from the attested `async_scheduling` header (G6 declared == attested) -- and the
WORKLOAD digest moves with `LAG` (the request Program digest is shared across LAG 0/1: S-r17b-44's two int-5 records, 0da657b7dac17478
both, workload 89437ee368282230 vs 640ba14c873f9b96); that defeats the pair, not the row-set comparison.  This fixture's one Program
plays the workload's role.  Torch-free; the reference evaluator (`serve3_reference.materialize`) evaluates the Program."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np

from verity.ir import Program, bind
from verity.ir.codec import encode_program, program_digest
from verity.ir.defs import composite
from verity.ir.refs import tuple_of
from verity_vllm.registry import lifted as L
from verity_vllm.registry import moe_pad as MP
from verity_vllm.registry import serve3_reference as ref
from verity_vllm.registry.prims import BF16
from verity.ir.types import Array, Tuple

EOS, CAP = "eos", "cap"


# ---------------------------------------------------------------------------------------------------------
# the one Program of (B, S, LAG, envelope)
# ---------------------------------------------------------------------------------------------------------


def _fixture_signature(S):
    ins, rec = MP.block_signature(S)
    return ((("xs", Array(S.B, Array(S.S, Array(S.H, MP.V(16))))),) + ins[1:-1] + (("run", Array(S.B, Array(S.S, L.ENABLE_T))),),
            Tuple(*([rec] * (S.B * S.S))))


@composite("PaddedDecodeFixture", 1, ["B", "S", "LAG", "E", "TOPK", "VPT", "H", "I"], _fixture_signature, conformance="protocol-defined")
def PaddedDecodeFixture(B, S, xs, wg, w13, w2, run):
    """`B` requests x `S` rows, one padded block per (t, s) on shared weights, each under ITS ROW'S enable `run[t][s]` (the place of
    `Continuation_v1`'s run bit; a fixture input here, set from the declaration by the checker); `LAG` is a STATIC (the digest carries the
    declaration; R17-1: Build and GM-01 consume the same recorded fact) -- the cap-shape circuit has `S` rows whatever the lag, the lag
    decides which rows the declaration makes executed = enabled."""
    if S.LAG not in (0, 1):
        raise ValueError("LAG must be 0 or 1 (BRIEF s4: the async-scheduling profile executes at most one post-EOS step)")
    blk = MP.block(S.E, S.TOPK, S.VPT, S.H, S.I)
    return tuple_of(*[B.call(blk, xs[t][s], wg, w13, w2, run[t][s], name=f"req{t}/row{s}") for t in range(S.B) for s in range(S.S)])


def fixture(B: int, S: int, LAG: int, E: int, TOPK: int, VPT: int, H: int, I: int):
    return bind(PaddedDecodeFixture, B=B, S=S, LAG=LAG, E=E, TOPK=TOPK, VPT=VPT, H=H, I=I)


# ---------------------------------------------------------------------------------------------------------
# the declaration and the engine
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Declared:
    """One request's declared facts (tokens.json + the attested profile): how many tokens were served and how it stopped."""
    served_output_len: int
    stop: str  # EOS | CAP


def executed_prefix_len(d: Declared, S: int, LAG: int) -> int:
    """R17-1: `served + LAG` for an EOS stop (the post-EOS row is executed), `served` for cap exhaustion (specified separately);
    the cap-shape Program has `S` rows and must hold the declared prefix -- internal capacity allocates for the declared lag."""
    if d.stop == CAP:
        if d.served_output_len != S:
            raise ValueError(f"cap exhaustion means served_output_len == cap rows ({S}); got {d.served_output_len}")
        return S
    if d.stop != EOS:
        raise ValueError(f"unknown stop {d.stop!r}")
    n = d.served_output_len + LAG
    if n > S:
        raise ValueError(f"declared executed_prefix_len {n} = served {d.served_output_len} + LAG {LAG} exceeds the {S} rows allocated: "
                         "the Program's internal capacity must allocate for the declared lag")
    return n


@dataclass
class Trace:
    """What the engine executed: the Active token row `x[t][s]` (bf16 words) for every executed (request, row); nothing else."""
    rows: dict = field(default_factory=dict)      # (t, s) -> list[int] bf16 words
    engine_lag: dict = field(default_factory=dict)  # t -> the post-EOS rows this engine ran for request t (the fixture's knob)

    def executed_rows(self, t: int) -> list[int]:
        return sorted(s for (tt, s) in self.rows if tt == t)


def bf16(v) -> list[int]:
    b = np.asarray(v, dtype=np.float32).reshape(-1).view(np.uint32).astype(np.uint64)
    r = ((b >> 16) & 1) + 0x7FFF
    return [int(w) for w in ((b + r) >> 16)]


def token_for_route(route: tuple[int, ...], E: int, H: int, seed: int) -> list[int]:
    """An Active token whose one-hot router picks exactly `route` in order (as the Stage 2 fixture)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, H).astype(np.float32)
    x[:E] = -1.0 - np.arange(E) * 0.25
    for j, e in enumerate(route):
        x[e] = 3.0 - j
    return bf16(x)


def one_hot_weights(E: int, H: int, I: int, seed: int = 1729) -> dict:
    rng = np.random.default_rng(seed)
    wg = np.zeros((E, H), dtype=np.float32)
    for e in range(E):
        wg[e, e] = 1.0
    return {"wg": bf16(wg), "w13": bf16(rng.normal(0, 0.3, (E, 2 * I, H))), "w2": bf16(rng.normal(0, 0.3, (E, H, I)))}


def run_engine(declared: list[Declared], S: int, E: int, TOPK: int, H: int, engine_lag: int | list[int], routes=None, seed: int = 0) -> Trace:
    """The fixture engine: request `t` executes `served + engine_lag[t]` rows (EOS) or `S` rows (cap), each with a chosen route.
    `engine_lag` is the ENGINE's behaviour (what the async profile actually did), deliberately separate from the header's `LAG`."""
    lags = [engine_lag] * len(declared) if isinstance(engine_lag, int) else list(engine_lag)
    tr = Trace()
    rng = np.random.default_rng(seed)
    for t, d in enumerate(declared):
        n = S if d.stop == CAP else d.served_output_len + lags[t]
        if n > S:
            raise ValueError(f"engine cannot execute {n} rows of {S} for request {t}")
        tr.engine_lag[t] = lags[t] if d.stop == EOS else 0
        for s in range(n):
            route = tuple(routes[t][s]) if routes is not None else tuple(int(v) for v in rng.permutation(E)[:TOPK])
            tr.rows[(t, s)] = token_for_route(route, E, H, seed=seed * 1000 + t * 100 + s)
    return tr


# ---------------------------------------------------------------------------------------------------------
# the checker (GM-01's role at the block level): declared lengths in, the trace compared, the assignment classified
# ---------------------------------------------------------------------------------------------------------


@dataclass
class Result:
    verdict: str
    LAG: int
    served_output_len: list
    executed_prefix_len: list
    failures: list
    alerts: list
    population: dict
    digest: str
    rows: dict = field(default_factory=dict)  # "t/s" -> class


def _classify(rec: list[int], E: int, TOPK: int, H: int) -> dict:
    o = iter(rec)
    take = lambda n: [next(o) for _ in range(n)]  # noqa: E731
    y, ids, alpha, s, g = take(H), take(TOPK), take(TOPK), take(E), take(E)
    z = [take(H) for _ in range(E)]
    u = [take(H) for _ in range(TOPK)]
    a_e = take(E)                                                     # the slot enables (B7 / A12): Active(0) / Active(1), never bot
    assert next(o, None) is None
    b16, b32, b1 = L.enc_bot(16), L.enc_bot(32), L.enc_bot(1)
    inv = (any(v == L.inv(16) for v in y) or any(v == L.inv(32) for v in ids + alpha + g) or any(v == L.inv(16) for zz in z for v in zz)
           or any(v == L.inv(16) for uu in u for v in uu) or any(v == L.inv(1) for v in s + a_e))    # every member the claim covers (revb S-r17b-45)
    return {"y_bot": all(v == b16 for v in y), "ids_bot": all(v == b32 for v in ids), "ids": [v & 0xFFFFFFFF if v >> 32 == 0 else None for v in ids],
            "s_active_count": sum(1 for v in s if v == 1), "s_bot": all(v == b1 for v in s), "g_bot_count": sum(1 for v in g if v == b32),
            "z_bot_slots": sum(1 for zz in z if all(v == b16 for v in zz)), "z_active_slots": sum(1 for zz in z if all(v >> 16 == 0 for v in zz)),
            "u_bot": all(all(v == b16 for v in uu) for uu in u), "inv": inv,
            "a_e": a_e, "enabled_slots": sum(1 for v in a_e if v == 1), "disabled_slots": sum(1 for v in a_e if v == 0),
            "a_e_matches_s": all((ae == se) for ae, se in zip(a_e, s)) if all(v in (0, 1) for v in s) else all(v == 0 for v in a_e)}


def check(trace: Trace, declared: list[Declared], S: int, LAG: int, weights: dict, E: int, TOPK: int, VPT: int, H: int, I: int) -> Result:
    B = len(declared)
    served = [d.served_output_len for d in declared]
    executed = [executed_prefix_len(d, S, LAG) for d in declared]          # from the declaration ONLY
    failures, alerts = [], []
    for t in range(B):
        have = set(trace.executed_rows(t))
        want = set(range(executed[t]))
        for s in sorted(have - want):
            failures.append(f"R17-1/LAG-ROW-BEYOND-DECLARED: request {t} row {s} executed; declared executed_prefix_len {executed[t]} "
                            f"(served_output_len {served[t]}, stop {declared[t].stop}, LAG {LAG})")
        for s in sorted(want - have):
            failures.append(f"R17-1/MISSING-EXECUTED-ROW: request {t} row {s} declared executed (executed_prefix_len {executed[t]} = "
                            f"served {served[t]} + LAG {LAG}) but absent from the trace")
    observed = {t: (max(trace.executed_rows(t)) + 1 - served[t]) for t in range(B) if declared[t].stop == EOS and trace.executed_rows(t)}
    if len(set(observed.values())) > 1:
        alerts.append(f"VARIABLE-LAG: observed post-EOS rows differ across EOS-stopped requests {observed} -- ALERT + DECISIONS-PENDING; "
                      f"not absorbed into LAG {LAG}")
    sp = fixture(B, S, LAG, E, TOPK, VPT, H, I)
    prog = Program(sp)
    digest = str(program_digest(encode_program(prog)))
    pop = {"expanded_slot_bodies": B * S * E, "compact_pairs": sum(executed) * TOPK, "executed_rows": sum(executed),
           "enabled_rows": sum(executed), "disabled_rows": sum(S - n for n in executed),
           "continuation_disabled_rows": sum(S - n for n in executed), "continuation_disabled_slots": sum(S - n for n in executed) * E,
           "unselected_disabled_slots": sum(executed) * (E - TOPK),
           "lag_rows": sum(1 for t in range(B) if declared[t].stop == EOS) * LAG,
           "lag_row_enabled_slots": sum(1 for t in range(B) if declared[t].stop == EOS) * LAG * TOPK,
           "cap_exhausted_requests": sum(1 for d in declared if d.stop == CAP)}
    if failures:
        return Result("FAIL", LAG, served, executed, failures, alerts, pop, digest)
    # the assignment: ENABLED rows where the declaration executes them (run = 1), DISABLED rows beyond (run = 0) -- evaluated on the ONE Program
    X_T = Array(H, BF16)                                                  # the UNLIFTED types: lift_words / bot_words encode from them
    xs, run = [], []
    for t in range(B):
        for s in range(S):
            xs.extend(L.lift_words(X_T, trace.rows[(t, s)]) if (t, s) in trace.rows else L.bot_words(X_T))
            run.append(L.enable_word(1 if s < executed[t] else 0))    # the row enable FROM THE DECLARATION (rev M-1064: one fact)
    inputs = [xs, L.lift_words(Array(E, Array(H, BF16)), weights["wg"]),
              L.lift_words(Array(E, Array(2 * I, Array(H, BF16))), weights["w13"]),
              L.lift_words(Array(E, Array(H, Array(I, BF16))), weights["w2"]), run]
    tr = ref.materialize(prog, inputs)
    outs = [tr[g] for g in prog.circuit.root_outputs()]
    per = len(outs) // (B * S)
    rows = {}
    for t in range(B):
        for s in range(S):
            c = _classify(outs[(t * S + s) * per:(t * S + s + 1) * per], E, TOPK, H)
            executed_row = s < executed[t]
            if c["inv"]:
                failures.append(f"ASSIGNMENT: request {t} row {s} carries INV")
            if executed_row:
                ok = (not c["y_bot"] and not c["ids_bot"] and c["s_active_count"] == TOPK == c["enabled_slots"] and c["a_e_matches_s"]
                      and c["z_bot_slots"] == E - TOPK == c["disabled_slots"] and c["z_active_slots"] == TOPK and c["g_bot_count"] == E - TOPK
                      and len(set(c["ids"])) == TOPK and all(v is not None and 0 <= v < E for v in c["ids"]))
                kind = "lag-row" if (declared[t].stop == EOS and s >= served[t]) else "served-row"
                if not ok:
                    failures.append(f"ASSIGNMENT: request {t} row {s} ({kind}) is not an enabled block with {TOPK} enabled / {E - TOPK} disabled slots: {c}")
            else:
                ok = (c["y_bot"] and c["ids_bot"] and c["s_bot"] and c["z_bot_slots"] == E and c["g_bot_count"] == E and c["u_bot"]
                      and c["disabled_slots"] == E)                       # every slot enable Active(0) = LEnable(0, bot); nothing Active
                kind = "continuation-disabled-row"
                if not ok:
                    failures.append(f"ASSIGNMENT: request {t} row {s} beyond executed_prefix_len is not a disabled all-bot row: {c}")
            rows[f"{t}/{s}"] = kind
    return Result("FAIL" if failures else "PASS", LAG, served, executed, failures, alerts, pop, digest, rows)


# ---------------------------------------------------------------------------------------------------------
# CLI: the two-request case + the three named negatives as evidence
# ---------------------------------------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--E", type=int, default=4)
    ap.add_argument("--TOPK", type=int, default=2)
    ap.add_argument("--VPT", type=int, default=2)
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--I", type=int, default=16)
    ap.add_argument("--S", type=int, default=5)
    a = ap.parse_args(argv)
    E, TOPK, VPT, H, I, S = a.E, a.TOPK, a.VPT, a.H, a.I, a.S
    w = one_hot_weights(E, H, I)
    decl = [Declared(2, EOS), Declared(S, CAP)]
    cases = {}
    tr = run_engine(decl, S, E, TOPK, H, engine_lag=1)
    cases["two_requests_early_eos_plus_declared_lag1"] = asdict(check(tr, decl, S, 1, w, E, TOPK, VPT, H, I))
    cases["neg_lag0_header_on_lag1_trace"] = asdict(check(tr, decl, S, 0, w, E, TOPK, VPT, H, I))
    cases["neg_lag1_header_on_lag0_trace"] = asdict(check(run_engine(decl, S, E, TOPK, H, engine_lag=0), decl, S, 1, w, E, TOPK, VPT, H, I))
    decl3 = [Declared(1, EOS), Declared(2, EOS), Declared(S, CAP)]
    cases["neg_variable_lag_surfaces"] = asdict(check(run_engine(decl3, S, E, TOPK, H, engine_lag=[1, 2, 0]), decl3, S, 1, w, E, TOPK, VPT, H, I))
    cases["neg_eos_widened_to_absorb_the_lag_row"] = asdict(check(tr, [Declared(3, EOS), Declared(S, CAP)], S, 1, w, E, TOPK, VPT, H, I))
    out = {"schema": "moe_pad_fixture/stage3c/v2-enablement", "envelope": {"E": E, "TOPK": TOPK, "VPT": VPT, "H": H, "I": I, "S": S},
           "fixed_select_member": MP.FIXED_SELECT_MEMBER, "cases": cases}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    for k, v in cases.items():
        print(f"[3c] {k}: {v['verdict']} served {v['served_output_len']} executed {v['executed_prefix_len']} "
              f"failures {len(v['failures'])} alerts {len(v['alerts'])}" + (f" :: {v['failures'][0][:110]}" if v["failures"] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
