"""The credit difftest's cases: the arguments of `credit.window` and `credit.period_credit`, the entries, and of the
steps they run or will, each held to the Lean on its own: `credit._credited`, `credit.drawn`, `credit._outcome`,
`credit._a_link` and `credit._weights`.

A `credited` case is an honest served window read through its challenger's session (`pouw_rounds_cases.Session`): its
manifest rows (the serving process's call rows, each with its credit bit, a credited one with its z table's root), the
round log (each round's root the challenger received, over its inner root and its A segments, `_a_log`, and its coin),
the window's registered entries (each K's A and B values, their domains one-stage's registration's under the window's
open root and its circuit, `registered.domain`), and the draw: `credit.draw_root` over them, recorded by the
challenger, then its draw coin opened after it; the auditor's ledger opens and ends the session for the window, its
open pinned to a commit the host's pins log (``pins``) records on the verifier's copy of main's first-parent history
(``main``, `pouw_rounds_cases.PINS` and `MAIN`). Then one mutation (`CREDIT_NAMES`) aimed at one refusal: a credit bit
flipped after the draw coin with the draw root recomputed (``changed``, a verdict withdrawn), a row changed
likewise, a missing or malformed bit, a credited row with no z root, the draw root left stale, no draw coin, a coin of
another family, a session pinned to a commit the log doesn't record (``unpinned``) or to a logged one off main's
first-parent line (``branch-pin``), and the forms.

A `drawn` case is a draw coin, the draw's stream, registration and receipt, a window's layout
(`circuit.h1window.program`'s arguments, `layout`): linears at one or two K, called step by step in a shuffled order, so
each K's stratum has ranges apart, and the committed K the draw is sized at (one to the window's units). Its mutations
break each rule `credit.strata` mirrors from `h1window.program`, the coin's, the arguments' and K's forms, give a K past
the window's units, and ``budget`` draws at K = `PLAN_K` from a population larger than it (each stratum's k_s rounded
up, so a unit or two past K).

An `outcome` case is a `drawn` case's arguments and one accepted proof outcome for each unit the coin draws; its
mutations drop one (``missing``), reject one (``failed``), repeat one, add an undrawn unit, prove the units of another
coin the prover chose (``chosen``), and break the forms.

A `window-drawn` case is what `credit._window_drawn` reads of a window, the draw `Credit.descriptorDrawn` gives the
J10-R period's statement (`Properties.Pouw.PeriodJ10R.creditDrawn`): the window's open root, its draw (its coin's live
opening, its root and its receipts), its circuit's descriptor (today's units or J10-R's) and a committed K. Its
mutations break the coin's opening, the root and receipts the draw is keyed by, the descriptor's hex and K's form, and
give a K past the window's units.

A `window` case is a `credited` case with its circuit's descriptor (`window_desc`: `h1window.program`'s encoding as far
as `credit.layout_of` reads it, each call's tiles a call node each or one batch, under a ``PREFIX``), its weights (each
K's ``<PREFIX>b/<K>``, its linears' B rows under the registered domain of its name and the circuit, binding no window,
`weights_of`), its open root `salt.window_open_root` of their `weights.weights_root`, round 0's ``requests`` and its
session, the auditor's ledger opening it inside a period whose open signs that weights' root (`credit.WEIGHTS_FIELD`)
and names a model credit pins (`credit.MODEL_FIELD`, `MODEL`),
a committed K, one call per manifest row, and an accepted proof of each unit its draw coin draws at K under the served window's arguments, as the verifier of
record gives it for the unit's session (`_window_proofs`: verified against the descriptor, its registered finding its
tile's A and B ports at its K's values and their registered roots); its mutations add
a call the manifest lacks, prove units the prover chose under its own stream, fail or drop a drawn unit, break K's form
or give one past the window's units, pass a `credited` refusal through, and break the descriptor (`DESC_NAMES`): its
form, its program function's record, a template, a K no linear has, the root's nodes, the record's layout, a record
naming another layout over the same nodes (``alt-layout``), or an H1 window's whole descriptor with a linear, so a
tile, at a K the tile's proof doesn't cover (``tile-k``: 2^17 or past, or no multiple of 32). Its mutations of the
composed check present a round log the challenger never received (``never-received``, its coins carried as if opened
after its roots and its draw root over it, so only `core_check.rounds` refuses it), a round coin opened after another
root, an A tree the round roots didn't commit, and no A value. Its mutations of the registration (`REGISTRATION_NAMES`) register the A values under another
window's open root or another circuit, their trees under those domains so `_a_link` accepts them
(``other-registration``), give one proof's session another registration's root for a value it opened
(``proof-other-registration``), change an opened B value's registered root (``entry-root``: the entry is not the root
the sessions opened, which the weights step refuses before any proof is read) or register it twice, verify one proof
against another circuit, drop a proof's A ports or every proof's B ports (``proof-no-b``), open an unregistered value,
and break an opened A entry's form. A window registers main's served form: each round's A table of each K
``…/a/<K>@<s>`` (M rows for each of the round's calls of K, call after call, `credit.step_tables`), each call's z/y
``…/zy@<c>`` (`h1window.table`) and each round's input-ids table ``…/ids@<s>``, every value but the weights' binding
the window's open root, each drawn unit's proof opening its own call's round's table. Its mutations register the A
tables with no window root (``a-call-unbound``) and point a proof at another round's or K's table
(``proof-other-call``). Its Commit record (`commit_record`, ROOTS1) registers each round's A tables and each ids table
over its true rows under the Commit's names; its mutations (`AROW_NAMES`, made before the draw) register one A table
with every row other (``a-zero``), one row other (``a-row-off``) or its calls' rows offset by one call
(``a-call-off``), one with a call missing that the Commit registers so too (``a-call-missing``), one ids table with a
row other (``ids-off``) or none (``ids-missing``), drop one of the Commit's values or register one twice, drop a MoE
window's (`expert_layout`) round's A table of its routed experts' downs (``commit-missing-expert``), present
another Commit record than the one its draw root
binds (``commit-unbound``) and break the record's form (``credit-commit-form``). Round 0's ids table ``…/ids@0`` binds
the window's session, not its open root, which contains it: round 0's ``requests`` is `credit.requests_root` of its
``request_table`` and ids@0's root. Its mutations (`IDS0_NAMES`, made before the window opens) open the window over the
bare request table's root or over another ids@0 root, and bind ids@0's domain to another session.
Each manifest row states its call's served rows m_s (`served_rows`: one a step, the period's T table rows, part of them
or none, so an empty step's calls are no credit, T one whose `h1window.slot` is M, `table_rows`); its mutations
(`ROWS_NAMES`, made before the draw) give a call m_s past T, one call of a step another m_s than its step's, or an m_s
that is no natural or absent, and ``t-other`` and ``t-form`` give the window a T whose slot is not M or that is no
positive integer.
Its mutations of the weights (`WINDOW_WEIGHTS_NAMES`) sign another weights' root in the period's open, none or a
malformed one, name no model credit pins in it (``weights-model``), open the window in no period, change or break round 0's ``requests``, and give a ``b/<K>`` other rows
under the domain of those rows (so `credit._registration` accepts it).

A `period` case is one to three such windows of one circuit in one period of the auditor's ledger (`period-open`
committing the SHA-512 of the circuit's descriptor, the K every window is drawn at, `credit.K_FIELD`, the tables'
rows T, `credit.ROWS_FIELD`, and the weights' root every window opens, `credit.WEIGHTS_FIELD`, with the pinned model it is of, `credit.MODEL_FIELD`, each window's session opened and ended, then the `period-close` listing them), presented in its close's order, each with that
descriptor, a window of no period beside it in the ledger. Its mutations leave the period unclosed or a
window incomplete, drop, reorder or add a window, present the window of no period among the period's, name a period
never opened or no period, forge the ledger, refuse one window, make one window wholly another circuit's (its
descriptor, registration and proofs, ``other-circuit``, so `credit.window` alone credits it), give one the descriptor
of another layout (``alt-layout``), open the period committing no circuit, one that is no H1 window's (``bad-circuit``)
or one with a tile at a K its proof doesn't cover (``tile-k``), open it committing no K or a malformed one, commit a
K past a window's units, commit no T, a malformed one or one whose slot is not its windows' (``no-rows``, ``t-form``,
``t-other``), prove one window's draw at another K (``other-k``), present one window's round log the challenger never
received (``never-received``), give one window's proof another registration's root (``proof-other-registration``),
pin one window's session to a logged branch commit off main's first-parent line (``branch-pin``, refused by
`challenger.credited_session`'s pin check before its circuit's), open the period signing no weights' root or a
malformed one, or naming no model credit pins (`MODEL_NAMES`: none, as a bare weights' root, a malformed one, another
checkpoint, an unpinned role), give one of two or three windows a ``b/<K>`` with one row's s_w changed (``other-sw``)
or every code zero (``zero-codes``), each registered as the served path registers it (`registered_b`) and that window's
open root over its own weights' root, and break the forms. The period's open records the commit it opened under
(`credit.PERIOD_PIN_FIELD`, MAIN[-1], older than every window's) and a freshness floor (`credit.STALE_FIELD`); a window
counts only when its pin is at most the floor of first-parent commits behind that pin, counted from it and never from
main's head (`credit._pin_hygiene`, after the per-window pin check). ``stale-pin`` records a nearer pin (MAIN[2]) and a
floor of one and pins one window to MAIN[4], the oldest the pins log records, two first-parent commits before the open
pin and so past the floor, refusing the period. Every window of a period shares the weights its open signs; the window of
another circuit registers them under its own.

An `a_link` case is a window's registered entries, among them one or two A values (``…/a/<K>``) whose trees are frame-v3
SHA-512 trees (`verity.ml.commitments.merkle.tree_levels`) of `steps` equal blocks, and its round log: each round's
`credit.cover` segments of every A value under `credit.round_root` of a random inner root. Its mutations register an A
tree other than the one the rounds committed, rewrite a segment with or without its round root, leave a gap,
add a segment, misalign one, list an unregistered value, register none or one twice, and break the forms. It may also
register rounds' A tables ``…/a/<K>@<s>``, each committed whole by its round; its mutations list one in another
round (``table-other-round``) and name a round past the window's (``table-past``).

A `weights` case is a window's layout, its circuit (its descriptor's SHA-512 and PREFIX), its registered entries (each
K's ``<PREFIX>b/<K>`` under its registered domain, among its A values, z tables and names that are no weights: a
non-canonical K, ``b/032`` or one past Python's ``int`` digit limit), its round-0 request table's root and its session,
and the window's open root over `weights.weights_root` of the ``…/b/<K>`` roots, that root and the session
(`salt.window_open_root`). Its mutations register a B value other than the one the open root binds (another code or
s_w), open the window on another request table's root or session or another root, register a K's B value twice, never,
of a K the layout lacks or under another prefix, give its entry other rows, words, domain (another circuit's, a
window's, other rows') or keys, and break the forms.

`tests/test_difftest.py` draws them for the harness, and `verity/core/protocols/pouw/tests/test_pouw_credit.py` holds each
mutation to the refusal it aims at.
"""
from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import random
import struct
from collections.abc import Callable

import pouw_rounds_cases as rc
from verity.core.protocols.pouw import credit as Cr
from verity.core.protocols.pouw import salt as PS
from verity.core.protocols.pouw import weights as Wt
from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.pouw.circuit import live_reads as LR
from verity.core.protocols.sampled_proofs import challenger as Ch
from verity.core.protocols.sampled_proofs.one_stage import registered as G
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain, tree_levels

MODULES = ["q_proj", "k_proj", "o_proj", "down_proj", "attn·out"]


def _hex(rng: random.Random) -> str:
    # a digest of one short draw: 64 drawn bytes a root made a period's examples overrun Hypothesis's entropy budget
    return hashlib.sha512(rng.randbytes(8)).hexdigest()


def _row(rng: random.Random, i: int) -> dict:
    """A manifest row as the serving process writes it (`Call.summary()` with its tables and bound), its credit bit
    among it."""
    bit = rng.random() < 0.7
    tables = {"a": _hex(rng), **({"z": _hex(rng)} if bit or rng.random() < 0.5 else {})}
    return {"index": i, "step": rng.randrange(4), "layer": rng.randrange(32), "module": rng.choice(MODULES),
            Cr.SERVED_ROWS: rng.randint(0, 64), "rows": rng.randint(1, 64), "k": rng.choice([64, 4096]), "n": 4096,
            "base": rng.randrange(1 << 20), "units": rng.randint(1, 8), "leaves": [rng.randrange(-3, 1 << 40), None],
            "tables": tables, "bound": {"m": rng.randint(1, 1 << 16), "ok": True}, Cr.CREDIT: bit}


def _draw_root(d: dict) -> str:
    salt, served, commit = bytes.fromhex(d["salt"]), d["served"], Cr.commit_of(d["commit"])[0]
    calls = Cr.calls_root(salt, d["calls"], None if served is None else bytes.fromhex(served),
                          None if commit is None else commit[0])
    return Cr.draw_root(calls, bytes.fromhex(d["draw"]["receipts"]), d["rounds"]).hex()


#: each way a window's A tables or input-ids tables are not the served Commit's, made before its draw: one round's A
#: table of a K every row other (``a-zero``), one row (``a-row-off``) or its calls' rows offset by one call
#: (``a-call-off``), one missing a call (``a-call-missing``, the Commit's too), one round's ids table one row other
#: (``ids-off``) or unregistered (``ids-missing``), and a Commit record lacking a table's value or registering it twice;
#: on a MoE window (`expert_layout`), a Commit record lacking a round's A table of the experts' downs
#: (``commit-missing-expert``: their K is the experts' alone, so its calls are expert calls only)
AROW_NAMES = ["a-zero", "a-row-off", "a-call-off", "a-call-missing", "ids-off", "ids-missing", "commit-missing",
              "commit-twice", "commit-missing-expert"]
#: the `AROW_NAMES` that need a round's A table of two calls or more
TWO_CALLS = ("a-call-off", "a-call-missing")
#: each way round 0's input-ids table is not inside round 0's request table under the window's session, made before the
#: window opens: round 0's request root the bare request table's (``ids0-outside``) or over another ids@0 root
#: (``ids0-root-off``), and ids@0's domain binding another session (``ids0-other-session``, the Commit registering it so)
IDS0_NAMES = ["ids0-outside", "ids0-root-off", "ids0-other-session"]
#: each way the window's request table ``…/requests@0`` (`credit.is_requests`, the live unit's ``req``) is not round
#: 0's under the window's session: at another root than ``request_table`` (``req0-root-off``), registered twice
#: (``req0-twice``) or under another PREFIX (``req0-other-prefix``), and its domain binding another session
#: (``req0-other-session``) or the window's open root, which contains it (``req0-window``)
REQ0_NAMES = ["req0-root-off", "req0-twice", "req0-other-prefix", "req0-other-session", "req0-window"]
#: the served Commit's words of a step's input token id (an int32)
IDS_WORDS = 2
#: a request table's words a row (`circuit.live.HEADER_WORDS`: five u32 headers as u16 words)
REQUEST_WORDS = 10


def _a_log(rng: random.Random, steps: int, ks: list[int], program: str, window: str | Callable[[str | None], str],
           other: str | None = None, calls: list[int] | None = None, prefix: str | None = None,
           weights: list[dict] | None = None, off: str | None = None, sid: str = "s-1", m: int = 1,
           requests: str | None = None, t: int | None = None) -> tuple[list[dict], list[dict], dict[str, dict]]:
    """A window's registered entries under its registration (the circuit whose descriptor's SHA-512 is `program`, and the
    window's open root `window`, or `window` of round 0's registered input-ids table's root when it is a function, since
    that root is inside the open root), each value under `prefix` (a random ``h1/w-<i>/`` when None), its round log over
    its A values and input-ids tables (`_log`), its coins not yet opened, and the entries of its A and ids tables over
    their true rows by name, the ones the served Commit registers. Each K of `ks` registers its B value
    (``…/b/<K>``, held across windows: `weights`' entry, or one at random 32-bit when None) and either its A value
    (``…/a/<K>``, `steps` equal blocks of K/2 words, its domain binding the window's open root, its tree under that
    domain) or, with `calls` (each call's K, in call order, `steps` dividing their count), the tables main's served form
    has (`h1window.table`): each round s's A table of each K ``…/a/<K>@<s>``, M = `m` rows for each of the round's calls
    of K, committed whole by round s, and each call c's z table ``…/zy@<c>``, both binding the window's open root, and
    each round s its input-ids table
    ``…/ids@<s>`` of as many slots, committed whole by round s, ids@0's domain binding the window's session `sid` in
    its open root's place (`credit.domain_of`); and at random a z value. ``other`` registers the A values and call
    tables under another window's open root and ids@0 under another session (``window``), another circuit
    (``program``) or no window root nor session (``unbound``, main's h1_prove form). `off` (`AROW_NAMES`') registers an
    A table or an ids table over other rows than its true ones, an A table one call short (its true rows too), or no ids
    table of a round, and ``ids0-other-session`` binds ids@0's domain to another session. With `calls` and round 0's
    request table's root `requests`, it registers (at random, always under `REQ0_NAMES` or with `t`) the request table
    ``…/requests@0`` at that root, its domain binding the session as ids@0's does, which `REQ0_NAMES` break. With `t`
    (the period's T) its request table has T rows, and each round s registers its slot table ``…/slots@<s>`` (T leaves
    of `live_reads.SLOT_WORDS`, committed whole by round s, binding the open root), the tables the live units read,
    which `LIVE_TABLE_NAMES` break."""
    prefix = f"h1/w-{rng.randrange(100)}/" if prefix is None else prefix
    values, entries, true = {}, [], {}
    prog = _hex(rng) if other == "program" else program
    session = (None if other == "unbound" else f"{sid}-other" if other == "window" or off == "ids0-other-session"
               else sid)
    target, own, tables = None, None, []
    for st in range(steps if calls else 0):
        mine: dict[int, int] = {}
        for k in calls[st * (len(calls) // steps):(st + 1) * (len(calls) // steps)]:
            mine[k] = mine.get(k, 0) + 1
        tables += [(f"{prefix}a/{k}@{st}", Wt.leaf_words(k), n, st, Wt.chunks(k)) for k, n in mine.items()]
    if calls and off in ("a-zero", "a-row-off"):
        target = rng.choice(tables)[0]
    elif calls and off in TWO_CALLS:
        target = rng.choice([v for v, _w, n, _s, _x in tables if n > 1])
    elif calls and off in ("ids-off", "ids-missing"):
        target = f"{prefix}ids@{rng.randrange(steps)}"

    def a(v: str, words: int, leaves: int, rnd: int | None) -> None:
        if v == target and off == "ids-missing":
            return
        first = Cr.ids_step(v) == 0
        dom = G.domain(prog, v, words, leaves, window=None if first else own, session=session if first else None)
        # rows from one short draw: 64 drawn bytes a row made a window's smallest example too large for Hypothesis
        seed = rng.randbytes(8) + v.encode()
        rows = [hashlib.sha512(seed + i.to_bytes(4, "big")).digest() for i in range(leaves)]
        true[v] = {"value": v, "root": tree_levels(dom, rows)[-1][0].hex(), "leaves": leaves, "words": words,
                   "domain": dom.domain_id.hex()}
        if v == target and off == "a-zero":
            rows = [hashlib.sha512(b"zero row" + i.to_bytes(4, "big")).digest() for i in range(leaves)]
        elif v == target and off == "a-call-off":
            rows = rows[m:] + rows[:m]
        elif v == target and off != "a-call-missing":
            i = rng.randrange(leaves)
            rows = rows[:i] + [rng.randbytes(64)] + rows[i + 1:]
        levels = tree_levels(dom, rows)
        values[v] = (leaves, levels, rnd)
        entries.append({**true[v], "root": levels[-1][0].hex()})

    leaves = rng.choice([1, 2, 4, 8])
    if calls:
        a(f"{prefix}ids@0", IDS_WORDS, leaves, 0)
    if callable(window):
        window = window(next((e["root"] for e in entries if e["value"] == f"{prefix}ids@0"), None))
    own = window if other is None else _hex(rng) if other == "window" else None if other == "unbound" else window
    for k in ks:
        if calls is None:
            a(f"{prefix}a/{k}", k // 2, steps * rng.choice([1, 2, 3, 4]), None)
        if weights is None:
            b = {"value": f"{prefix}b/{k}", "root": _hex(rng), "leaves": rng.randint(1, 64), "words": k // 2,
                 **({G.WIDTH_KEY: 32} if rng.random() < 0.3 else {})}
            entries.append({**b, "domain": Cr.domain_of(program, window, b)})
    entries += [dict(e) for e in weights or ()]
    for v, words, n, st, x in tables:
        a(v, words, (n - (v == target and off == "a-call-missing")) * m * x, st)
    for c, k in enumerate(calls or []):
        zy = {"value": f"{prefix}zy@{c}", "root": _hex(rng), "leaves": rng.randint(1, 8), "words": 3}
        entries.append({**zy, "domain": G.domain(prog, zy["value"], 3, zy["leaves"], window=own).domain_id.hex()})
    for s in range(1, steps if calls else 0):
        a(f"{prefix}ids@{s}", IDS_WORDS, leaves, s)
    if calls and t is not None:
        odd = rng.randrange(steps)
        for s in range(steps):
            if not (s == odd and off == "live-no-slots"):
                a(LR.table(prefix, LR.SLOTS, s), LR.SLOT_WORDS,
                  t + rng.choice([-1, 1]) if s == odd and off == "live-slots-rows" and t > 1 else
                  t + 1 if s == odd and off == "live-slots-rows" else t, s)
    if (calls and requests is not None and off != "live-no-requests"
            and (off in REQ0_NAMES or t is not None or rng.random() < 0.5)):
        n = leaves if t is None else t + 1 if off == "live-requests-rows" else t
        v = f"x/{prefix}requests@0" if off == "req0-other-prefix" else f"{prefix}requests@0"
        dom = (G.domain(prog, v, REQUEST_WORDS, n, window=own) if off == "req0-window" else
               G.domain(prog, v, REQUEST_WORDS, n, session=f"{sid}-other" if off == "req0-other-session"
                        else sid if off == "ids0-other-session" else session))
        e = {"value": v, "root": _hex(rng) if off == "req0-root-off" else requests, "leaves": n,
             "words": REQUEST_WORDS, "domain": dom.domain_id.hex()}
        entries += [e, dict(e)] if off == "req0-twice" else [e]
    if rng.random() < 0.5:
        entries.append({"value": f"{prefix}z/{rng.choice([64, 4096])}", "root": _hex(rng), "leaves": 3,
                        "domain": _hex(rng)})
    rng.shuffle(entries)
    return entries, _log(steps, values, [_hex(rng) for _ in range(steps)]), true


def commit_record(rng: random.Random, true: dict[str, dict], steps: int, off: str | None = None) -> str:
    """The served Commit's record (`credit.COMMIT_FORMAT`) in hex of its canonical JSON: its registered values `true`'s
    A tables and ids tables (`_a_log`'s) under the Commit's names (`credit.commit_rows` of each A table's round and K,
    and `credit.commit_ids`), among values of its own; ``commit-missing`` drops one of them and ``commit-twice``
    registers one twice."""
    names = {}
    for v in true:
        s, a = Cr.ids_step(v), Cr.a_value(v)
        if s is not None:
            names[v] = Cr.commit_ids(s)
        elif a is not None and a[1] is not None:
            names[v] = Cr.commit_rows(int(a[1]), int(a[0]))
    reg = [{**e, "value": names[v]} for v, e in sorted(true.items()) if v in names]
    if reg and off == "commit-missing":
        del reg[rng.randrange(len(reg))]
    elif reg and off == "commit-missing-expert":
        gone = rng.choice(sorted(n for n in names.values() if n.endswith(f"/k{EXPERT_K}")))
        reg = [e for e in reg if e["value"] != gone]
    elif reg and off == "commit-twice":
        reg.append(dict(rng.choice(reg)))
    reg += [{"value": f"{name}/{s}", "root": _hex(rng), "leaves": rng.randint(1, 64), "words": rng.choice([1, 8]),
             "domain": _hex(rng)} for s in range(steps) for name in ("input-embeds", "sampled-ids")
            if rng.random() < 0.5]
    rng.shuffle(reg)
    rec = {"format": Cr.COMMIT_FORMAT, "protocol": "verity/sampled-proofs/one-stage/v1", "program": _hex(rng),
           "window": {"kind": "served", "steps": steps}, "registered": reg}
    return Cr._canonical(rec).encode().hex()


#: a manifest row with no served rows m_s at all
MISSING = object()
#: each way a window's calls' served rows m_s are not one natural at most its A slot M a step, made before its draw: a
#: call's past M (``rows-past``), one call of a step other than its step's (``rows-other``), or not a natural or absent
#: (``rows-form``)
ROWS_NAMES = ["rows-past", "rows-other", "rows-form"]


def _once(lins: list[str]) -> list[int]:
    """The round counts S whose C/S-call blocks of `lins` call each linear once."""
    return [s for s in range(1, len(lins) + 1) if len(lins) % s == 0 and all(
        len(set(lins[i:i + len(lins) // s])) == len(lins) // s for i in range(0, len(lins), len(lins) // s))]


def table_rows(rng: random.Random, lay: dict) -> int:
    """The tables' leaf count T a period of windows of layout `lay` commits (`credit.ROWS_FIELD`): one whose
    `h1window.slot` is the layout's A slot M, M itself or some T in (M/2, M] (any T in [1, M] when M is the tile's R)."""
    m, r = lay["m"], lay["tile"][0]
    return rng.choice([m, rng.randint(1 if m == r else m // 2 + 1, m)])


def other_rows(rng: random.Random, lay: dict) -> int:
    """A T no window of layout `lay` has: a positive integer whose `h1window.slot` is not its A slot M."""
    m, r = lay["m"], lay["tile"][0]
    return rng.choice([m + rng.randint(1, m), *([rng.randint(1, m // 2)] if m > r else [])])


def bad_rows(rng: random.Random, t: int) -> object:
    """A table row count that is no positive integer."""
    return rng.choice([0, -1, True, None, str(t), [t]])


def served_rows(rng: random.Random, t: int, calls: int, steps: int, name: str = "honest",
                each: list[int] | None = None) -> list:
    """Each call's served rows m_s, one a step of `steps` over `calls` calls (call c at step ``c // (C/S)``): `each`'s
    (a fixed length's schedule, `length_rows`) or, when None, the tables' T rows, some of them or none (an empty step,
    whose calls are no credit), then mutation `name` (`ROWS_NAMES`)."""
    m = t
    if each is None:
        each = [rng.choice([m, m, rng.randint(1, m), rng.randint(0, max(1, m // 8)), 0]) for _ in range(steps)]
    per = calls // steps
    out: list = [each[c // per] for c in range(calls)]
    match name:
        case "rows-past":
            out[rng.randrange(calls)] = m + rng.randint(1, 3)
        case "rows-other":
            i = rng.choice([c for c in range(calls) if c % per])
            out[i] = rng.choice([v for v in range(m + 1) if v != out[i]])
        case "rows-form":
            out[rng.randrange(calls)] = rng.choice([True, False, -1, str(m), None, [m], MISSING])
    return out


def honest_credit(rng: random.Random, name: str = "honest", rows: int | None = None, sid: str = "s-1",
                  leaf: int = rc.RECORD_LEAF, ks: list[int] | None = None, program: str | None = None,
                  other: str | None = None, calls: list[int] | None = None, prefix: str | None = None,
                  weights: list[dict] | None = None, off: str | None = None, rounds: int | None = None, m: int = 1,
                  served: list | None = None, t: int | None = None) -> tuple[dict, rc.Session]:
    """An honest served window of `rows` manifest rows (one to five when None), each call's served rows ``served[i]``
    (`credit.SERVED_ROWS`; `MISSING` drops it) when given, whose draw coin opened after its draw root
    (none for ``no-draw``), its session `sid` signing its record at `leaf`. Its round log and entries are `_a_log`'s at
    the contractions `ks` (and `calls`, a round count dividing theirs) under the circuit `program` (random ones when
    None), `other`, `prefix`, `weights`, `off` and the A slot `m`, and its Commit record `commit_record`'s over their
    true tables, bound under its draw root, over `rounds` rounds (one to three dividing the calls
    when None); with `weights`, its open root is `salt.window_open_root` of their
    `weights_root` (`fold`), round 0's ``requests`` and `sid`, else a random one. Round 0's ``requests`` is
    `credit.requests_root` of its ``request_table`` (a random root) and its input-ids table's (`IDS0_NAMES` break
    it), or the bare table's when it registers none; with `t` it registers the live units' tables (`_a_log`). Each
    round's root is the one the
    challenger received and its coin the one opened after it; under ``never-received`` the challenger received other
    roots, and the log carries their coins as if opened after its own, its draw root over it (the composed check's
    counterexample: `_credited` and `_a_link` accept it, `core_check.rounds` doesn't)."""
    if rounds is None:
        rounds = rng.choice([s for s in (1, 2, 3) if calls is None or len(calls) % s == 0])
    ch = Wd.live_challenger(rng.randrange(4), rounds, rng.randbytes)
    table, opened = _hex(rng), {}

    def open_root(ids0: str | None) -> str:
        if ids0 is not None and off == "ids0-root-off":
            ids0 = _hex(rng)
        opened["requests"] = (table if ids0 is None or off == "ids0-outside"
                              else Cr.requests_root(bytes.fromhex(table), bytes.fromhex(ids0)).hex())
        opened["window"] = rc._window(rng) if weights is None else {
            "root": PS.window_open_root(bytes.fromhex(fold(weights)), bytes.fromhex(opened["requests"]), sid).hex()}
        return opened["window"]["root"]

    ks = rng.sample([64, 4096, 14336], rng.randint(1, 2)) if ks is None else ks
    entries, log, true = _a_log(rng, rounds, ks, _hex(rng) if program is None else program, open_root, other,
                                calls, prefix, weights, off, sid, m, table, t)
    window = opened["window"]
    log[0]["requests"], log[0]["request_table"] = opened["requests"], table
    d = {"salt": _hex(rng), "calls": [_row(rng, i) for i in range(rng.randint(1, 5) if rows is None else rows)],
         "served": rng.choice([None, _hex(rng)]), "commit": commit_record(rng, true, rounds, off),
         "draw": {"receipts": _hex(rng)}, "entries": entries}
    for row, ms in zip(d["calls"], served or []):
        row[Cr.SERVED_ROWS] = ms
        if ms is MISSING:
            del row[Cr.SERVED_ROWS]
    received = [bytes.fromhex(x["root"]) for x in log]
    if name == "never-received":
        received = [rng.randbytes(64) for _ in log]

    def draw_after(rec) -> bytes:
        opened = rec.openings if name != "never-received" else [
            dataclasses.replace(o, after=bytes.fromhex(x["root"])) for o, x in zip(rec.openings, log)]
        d["rounds"] = [{**x, "coin": o.to_json()} for x, o in zip(log, opened)]
        d["draw"]["root"] = _draw_root(d)
        return bytes.fromhex(d["draw"]["root"])

    s = rc.Session(rng, ch, received, salt=None, sid=sid, leaf=leaf, draw=None if name == "no-draw" else draw_after)
    if name == "no-draw":
        d["draw"]["coin"] = ch.open(Wd.DRAW, 0, draw_after(s.rounds)).to_json()
    else:
        d["draw"]["coin"] = s.draw[0].to_json()
    return {"key": s.key(), "pins": list(rc.PINS), "main": list(rc.MAIN), "window": window, "session": s.files(), **d}, s


def length_rows(rng: random.Random, max_tokens: list[int], steps: int, t: int) -> list[int]:
    """Each step's served rows m_s in a window of `steps` steps and T = `t` table rows that serves the fixed length
    `max_tokens`: step 0 each request's prompt (one row or more, at most `t` in all), step s ≥ 1 one decode row for each
    request whose max_tokens is past s."""
    return [rng.randint(len(max_tokens), t) if s == 0 else sum(1 for n in max_tokens if n > s) for s in range(steps)]


def fixed_length(rng: random.Random, t: int, rounds: int, aim: str | None = None) -> list[int]:
    """Each request's public max_tokens in a window of T = `t` table rows and `rounds` rounds: one to `t` requests,
    often four or fewer, each served to its cap inside the window; aimed at ``length-past`` one runs two rounds or more,
    at ``length-early`` one ends before the window's last round (`rounds` ≥ 2 for both)."""
    out = [rng.randint(1, rounds) for _ in range(rng.choice([rng.randint(1, min(t, 4)), rng.randint(1, t)]))]
    if aim == "length-past":
        out[rng.randrange(len(out))] = rng.randint(2, rounds)
    elif aim == "length-early":
        out[rng.randrange(len(out))] = rng.randint(1, rounds - 1)
    return out


def _flip_bit(d: dict, rng: random.Random, to: bool | None = None) -> None:
    rows = [r for r in d["calls"] if to is None or r[Cr.CREDIT] != to] or d["calls"]
    row = rng.choice(rows)
    row[Cr.CREDIT] = not row[Cr.CREDIT] if to is None else to
    if row[Cr.CREDIT]:
        row["tables"].setdefault("z", _hex(rng))


def _mutate_credit(d: dict, s: rc.Session, name: str, rng: random.Random) -> None:
    if rc.pin_mutation(d, name, rng, lambda opened: rc.ledger_lines(rc.ledger_events(rng, d["window"], s, opened=opened),
                                                                     rc._key(0))):
        return
    calls, j = d["calls"], rng.randrange(len(d["calls"]))
    match name:
        case "changed":
            _flip_bit(d, rng)
            d["draw"]["root"] = _draw_root(d)
        case "withdrawn":
            _flip_bit(d, rng, to=False)
            d["draw"]["root"] = _draw_root(d)
        case "row-changed":
            calls[j]["tables"]["a"] = _hex(rng)
            d["draw"]["root"] = _draw_root(d)
        case "stale-root":
            _flip_bit(d, rng)
        case "no-bit":
            del calls[j][Cr.CREDIT]
        case "bit-form":
            calls[j][Cr.CREDIT] = rng.choice([1, 0, "true", None, [True]])
        case "no-z":
            calls[j][Cr.CREDIT] = True
            calls[j]["tables"] = rng.choice([{"a": _hex(rng)}, {"a": _hex(rng), "z": _hex(rng)[:-2]}, None,
                                             {"a": _hex(rng), "z": _hex(rng).upper()}])
        case "index":
            calls[j]["index"] = rng.choice([j + 1, str(j), None, True if j == 1 else -1])
        case "row-form":
            deep = 0
            for _ in range(Cr.DEPTH):
                deep = [deep]
            calls[j] = rng.choice([[1], "row", None, {**calls[j], "bound": deep}])
        case "calls-form":
            d["calls"] = rng.choice([None, {"0": calls}, "x"])
        case "salt-form":
            d["salt"] = rng.choice([None, d["salt"][:-2], d["salt"].upper(), 5])
        case "served-form":
            d["served"] = rng.choice(["", "ab", 5, _hex(rng).upper()])
        case "commit-form":
            rec, h = json.loads(bytes.fromhex(d["commit"])), d["commit"]
            d["commit"] = rng.choice([
                None, 5, "", "zz", h.upper(), h[:-2], b"\xff\xfe".hex(), json.dumps(rec).encode().hex(),
                *(Cr._canonical(x).encode().hex() for x in (
                    [rec], {**rec, "format": "verity/registration/v1"}, {k: v for k, v in rec.items() if k != "format"},
                    {**rec, "registered": {}}, {k: v for k, v in rec.items() if k != "registered"},
                    {**rec, "registered": rec["registered"] + [5]}))])
        case "rounds-form":
            k = rng.randrange(len(d["rounds"]))
            d["rounds"] = rng.choice([None, "x", d["rounds"][:k] + [5] + d["rounds"][k + 1:],
                                      d["rounds"][:k] + [{**d["rounds"][k], "root": "zz"}] + d["rounds"][k + 1:],
                                      d["rounds"][:k] + [{**d["rounds"][k], "coin": None}] + d["rounds"][k + 1:]])
        case "draw-form":
            d["draw"] = rng.choice([None, {**d["draw"], "receipts": "zz"}, {k: v for k, v in d["draw"].items()
                                                                            if k != "receipts"}])
        case "no-draw":
            pass
        case "draw-coin":
            coin = d["draw"]["coin"]
            other = Wd.live_challenger(s.ch.commitment.epoch, len(d["rounds"]), rng.randbytes)
            d["draw"]["coin"] = rng.choice([{**coin, "salt": rc._flip(coin["salt"])}, None, {**coin, "index": 1},
                                            {**coin, "after": rc._flip(coin["after"])},
                                            other.open(Wd.DRAW, 0, bytes.fromhex(d["draw"]["root"])).to_json()])
        case "key-form":
            d["key"] = rng.choice([None, {**d["key"], "public": "00"}, {**d["key"], "H": 0}])
        case "forged-ledger":
            d["ledger"] = rc.ledger_lines(rc.ledger_events(rng, d["window"], s), rc._key(1))
        case _:
            raise ValueError(name)


CREDIT_NAMES = ["changed", "withdrawn", "row-changed", "stale-root", "no-bit", "bit-form", "no-z", "index", "row-form",
                "calls-form", "salt-form", "served-form", "commit-form", "rounds-form", "draw-form", "no-draw",
                "draw-coin", "key-form", "forged-ledger", "unpinned", "branch-pin"]
CREDIT_KEYS = ("key", "ledger", "pins", "main", "window", "session", "salt", "calls", "served", "commit", "rounds",
               "draw")


def credit_instance(rng: random.Random, name: str = "honest") -> dict:
    """One `credited` case: an honest window, then mutation `name`."""
    d, s = honest_credit(rng, name)
    if name != "honest":
        _mutate_credit(d, s, name, rng)
    if "ledger" not in d:
        d["ledger"] = rc.ledger_lines(rc.ledger_events(rng, d["window"], s), rc._key(0))
    return {k: d[k] for k in CREDIT_KEYS}


def credit_args(d: dict) -> tuple:
    """`credit._credited`'s arguments for case `d`, the challenger's signatures checked with `lms_verify`."""
    return (*(d[k] for k in CREDIT_KEYS), rc.lms_verify)


LINEARS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def layout(rng: random.Random, big: bool = False) -> dict:
    """A served window's layout, `h1window.program`'s arguments: one to four linears at one or two K, each n whole blocks
    of the tile's C, called once a step over one to three steps (A block s at step s) in a shuffled order; ``big``, a
    population past `credit.RECORD_K`."""
    r, c = rng.choice([(1, 1), (2, 4), (4, 8), (8, 8), (16, 16)])
    ks = rng.sample([32, 64, 4096, 6144, 14336], rng.randint(1, 2))
    names = rng.sample(LINEARS, rng.randint(len(ks), 4))
    linears = [[name, ks[i] if i < len(ks) else rng.choice(ks), c * (rng.randint(1, 4) if not big else 64)]
               for i, name in enumerate(names)]
    calls = [[name, step] for step in range(rng.randint(1, 3)) for name in names]
    rng.shuffle(calls)
    m = r << (rng.randint(0, 2) if not big else 9)
    return {"calls": calls, "linears": linears, "m": m, "tile": [r, c]}


#: a MoE layer's routed experts, as H1 names each projection (vllm's `pouw_h1_hooks.expert_names`:
#: ``prefix.expert.projection``), and their downs' K, the intermediate width, which no other linear of `expert_layout` has
EXPERTS_PREFIX, EXPERT_K = "model.layers.1.mlp.experts", 32


def expert_layout(rng: random.Random) -> dict:
    """A served MoE window's layout, as vllm's H1 lays one out: a dense linear at K = H, then a MoE layer's two to four
    routed experts, each its gate_up at H then its down at `EXPERT_K`, every one called once a step in that static
    order (an idle expert's call is its empty slots) over one to three steps."""
    r, c = rng.choice([(1, 1), (2, 4), (4, 8)])
    h = rng.choice([64, 4096])
    linears = [["model.layers.0.self_attn.qkv_proj", h, c * rng.randint(1, 4)]]
    for e in range(rng.randint(2, 4)):
        linears += [[f"{EXPERTS_PREFIX}.{e}.gate_up", h, c * rng.randint(1, 4)],
                    [f"{EXPERTS_PREFIX}.{e}.down", EXPERT_K, c * rng.randint(1, 4)]]
    calls = [[name, step] for step in range(rng.randint(1, 3)) for name, _k, _n in linears]
    return {"calls": calls, "linears": linears, "m": r << rng.randint(0, 2), "tile": [r, c]}


def _canon(x: object) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"))


def _template_id(k: int, unit: dict) -> str:
    return f"{Cr.TILE_FN}{{UNIT={_canon(unit)},K={k}}}"


def row_type(k: int) -> list:
    """`h1zy.ROW` (K) as the codec writes it: K s8 codes, then the row's f32 scale, each its bits."""
    return ["Record", [["codes", ["Array", k, ["Array", 8, ["Value", 1]]]], ["scale", ["Array", 32, ["Value", 1]]]]]


def _template(k: int, r: int, c: int, unit: dict, receipt: object = None, j10r: bool = False) -> dict:
    """K's template as `h1window.template` encodes it: a composite over the tile unit `unit`, its A and B rows
    ``a_k<K>`` and ``b_k<K>`` (n·R and n·C chunk rows, `row_type` of K/n, n = `weights.chunks` (K): the zy unit's, or at
    n > 1 `h1zy.PouwH1TileZyChunks`'), then the unit's z/y record ``zy``, the salt, its position and the roots (and,
    `j10r`, a J10-R unit's call and blocks); returning ``{receipt, ok}`` with ``receipt`` the type `receipt` when one
    is given, else a word."""
    word, n = ["Value", 16], _n(k)
    params = [[f"a_k{k}", ["Array", n * r, row_type(k // n)]], [f"b_k{k}", ["Array", n * c, row_type(k // n)]],
              ["zy", ["Array", 3 * r * c, word]], ["salt", ["Array", 4, word]], ["position", word],
              ["roots", ["Array", 2, ["Array", 4, word]]]]
    if j10r:
        params += [["call", ["Array", 4, word]], ["blocks", ["Array", 2, ["Array", 4, word]]]]
    ret = word if receipt is None else ["Record", [["receipt", receipt], ["ok", ["Value", 1]]]]
    return {"form": "composite", "id": Cr.TILE_FN, "bindings": {"K": k, "UNIT": unit},
            "body": {"params": params, "ret_type": ret, "nodes": [{"form": "call", "fn": 0, "args": [], "out": 0}],
                     "ret": {"affine": ["n", 0, 0, 1]}}}


def record(tiles: list, lay: dict, prefix: str = "") -> str:
    """`h1window.program`'s program function: `credit.WINDOW_FN` and its record."""
    vals = {"TILES": tiles, "LINEARS": lay["linears"], "CALLS": lay["calls"], "M": lay["m"], "PREFIX": prefix}
    return Cr.WINDOW_FN + "{" + ",".join(f"{k}={_canon(vals[k])}" for k in Cr.RECORD) + "}"


def _n(k: int) -> int:
    """`weights.chunks` (K), or 1 at a K it refuses: a ``tile-k`` case's, which `credit.layout_of` refuses before it
    reads K's template."""
    try:
        return Wt.chunks(k)
    except ValueError:
        return 1


def _unit_fn(k: int, r: int, c: int, j10r: bool = False) -> str:
    """K's unit id in the descriptor: the tile's (J10-R's v5 when `j10r`), wrapped at a chunked K in
    `h1zy.PouwH1TileZyChunks_v1`, or its v2 over J10-R (`h1zy.unit_chunks`)."""
    tile, n = f"H1Tile_v{5 if j10r else 1}{{K={k},R={r},C={c}}}", _n(k)
    return tile if n == 1 else f"PouwH1TileZyChunks_v{2 if j10r else 1}{{UNIT={tile},CHUNKS={n}}}"


def window_desc(rng: random.Random, lay: dict, prefix: str = "", receipt: object = None, j10r: bool = False) -> dict:
    """A served window's circuit as `h1window.program(linears, calls, m, tile)` encodes it (`codec.encode_program`), as
    far as `credit.layout_of` reads it: its program function's record, each K's template, and the root's input batches,
    then each call's (m/R)·(n/C) tiles, each a call of K's template or, at random, a run of them one batch; each
    template returning a receipt of type `receipt` (`_template`) when one is given, and over J10-R units when `j10r`.
    At a chunked K (`weights.chunks` n > 1) the unit is `h1zy.PouwH1TileZyChunks_v1` {UNIT, CHUNKS = n} over the tile
    (v2 over a J10-R one), as `h1window.program` wraps it."""
    (r, c), ks = lay["tile"], sorted({k for _name, k, _n in lay["linears"]})
    units = {k: {"fn": _unit_fn(k, r, c, j10r)} for k in ks}
    tids = {k: _template_id(k, units[k]) for k in ks}
    callees = ["Input16_v1", *(u["fn"] for u in units.values()), *tids.values()]
    defs = {u["fn"]: {"form": "primitive", "id": u["fn"], "params": [], "ret": ["Value", 16]} for u in units.values()}
    defs |= {tids[k]: _template(k, r, c, units[k], receipt, j10r) for k in ks}
    defs["Input16_v1"] = {"form": "primitive", "id": "Input16_v1", "params": [], "ret": ["Value", 16]}
    nodes = [{"form": "batch", "fn": 0, "args": [], "out": 0, "axes": [], "n": rng.randint(1, 64)}
             for _ in range(2 * len(ks) + 3)]
    n_of = {name: (k, n) for name, k, n in lay["linears"]}
    for lin, _blk in lay["calls"]:
        k, n = n_of[lin]
        size, fn = (lay["m"] // r) * (n // c), callees.index(tids[k])
        if rng.random() < 0.3:
            nodes.append({"form": "batch", "fn": fn, "args": [], "out": 1, "axes": [0], "n": size})
        else:
            nodes += [{"form": "call", "fn": fn, "args": [], "out": 1} for _ in range(size)]
    fn = record([[k, {"fn": tids[k]}] for k in ks], lay, prefix)
    root = f"Root[{fn}]_v1"
    defs[root] = {"form": "composite", "id": root, "bindings": {},
                  "body": {"params": [], "ret_type": ["Value", 16], "nodes": nodes, "ret": {"concat": []}}}
    return {"callees": callees, "definitions": defs, "program_fn": fn, "root": root, "schema": Cr.SCHEMA,
            "types": [["Value", 16]]}


def desc_bytes(desc: dict) -> bytes:
    """A descriptor's bytes as the verifier holds them: canonical JSON without annotations."""
    return _canon(desc).encode()


def descriptor(rng: random.Random, lay: dict, prefix: str = "", receipt: object = None, j10r: bool = False) -> str:
    """`window_desc`'s bytes in hex, as a window's evidence carries them."""
    return desc_bytes(window_desc(rng, lay, prefix, receipt, j10r)).hex()


def j10r_misformed(rng: random.Random, lay: dict, prefix: str = "") -> str:
    """A J10-R window's descriptor (`descriptor`'s, ``j10r``) whose template at some K is in no one form: its ``call``
    or ``blocks`` dropped, the two swapped, or (a window of two K) that K's template today's while the other's is
    J10-R's; in hex."""
    desc = window_desc(rng, lay, prefix, j10r=True)
    tiles = json.loads(dict(Cr._record(desc["program_fn"]))["TILES"])
    _k, tid = rng.choice(tiles)
    params = desc["definitions"][tid["fn"]]["body"]["params"]
    how = rng.choice(["drop", "swap", *(["mixed"] if len(tiles) > 1 else [])])
    if how == "drop":
        del params[rng.choice([-1, -2])]
    elif how == "swap":
        params[-2:] = params[-2:][::-1]
    else:
        del params[-2:]
    return desc_bytes(desc).hex()


def _alt_layout(rng: random.Random, lay: dict) -> dict:
    """Another layout of `lay`'s linears: another slot, another n, or the calls in another order, with other units or
    other strata than `lay`'s."""
    (r, c), out = lay["tile"], json.loads(json.dumps(lay))
    while True:
        match rng.choice(["m", "n", "calls", "drop"]):
            case "m":
                out["m"] = lay["m"] * 2
            case "n":
                out["linears"][0][2] = lay["linears"][0][2] + c
            case "calls" if len({k for _n, k, _ in lay["linears"]}) > 1:
                rng.shuffle(out["calls"])
            case "drop" if len(lay["calls"]) > len(lay["linears"]):
                del out["calls"][next(i for i, (lin, _b) in enumerate(lay["calls"])
                                      if sum(x == lin for x, _ in lay["calls"]) > 1
                                      and max(b for x, b in lay["calls"] if x == lin) == lay["calls"][i][1])]
            case _:
                continue
        if (Cr.strata(out)[0] or None) != Cr.strata(lay)[0] and Cr.strata(out)[1] is None:
            return out
        out = json.loads(json.dumps(lay))


def _break_desc(rng: random.Random, lay: dict, name: str) -> str:
    """A descriptor of `lay`'s window broken as `name` says, in hex."""
    desc = window_desc(rng, lay)
    defs, fn = desc["definitions"], desc["program_fn"]
    root = defs[desc["root"]]["body"]["nodes"]
    tiles = json.loads(dict(Cr._record(fn))["TILES"])
    tid = rng.choice(tiles)[1]["fn"]
    match name:
        case "descriptor-form":
            raw = desc_bytes(desc)
            return rng.choice([raw + b" ", raw.replace(b",", b", ", 1), b"\xff" + raw, b"[]",
                               desc_bytes({**desc, "schema": "verity-ir/descriptor/v2"}),
                               desc_bytes({k: v for k, v in desc.items() if k != "callees"}),
                               desc_bytes({**desc, "callees": [*desc["callees"], 5]}),
                               desc_bytes({**desc, "program_fn": None}), b'{"a":NaN}']).hex()
        case "not-window":
            pairs = Cr._record(fn)
            other = rng.choice([
                "PouwH1Window_v2" + fn[len(Cr.WINDOW_FN):],
                Cr.WINDOW_FN + "{" + ",".join(f"{k}={v}" for k, v in pairs[:-1]) + "}",
                Cr.WINDOW_FN + "{" + ",".join(f"{k}={v}" for k, v in [pairs[1], pairs[0], *pairs[2:]]) + "}",
                fn.replace(f"M={lay['m']}", f"M= {lay['m']}"), fn.replace(f"M={lay['m']}", f"M=0{lay['m']}"),
                fn.replace('PREFIX=""', "PREFIX=5"), record(tiles[::-1] if len(tiles) > 1 else [], lay),
                record([[t[0], t[1]["fn"]] for t in tiles], lay), record([[0, tiles[0][1]]] + tiles[1:], lay)])
            desc["program_fn"] = other
        case "template":
            t = defs[tid]
            k = next(k for k, x in tiles if x["fn"] == tid)
            body = t["body"]
            match rng.choice(["k", "id", "form", "params", "rows", "unit", "missing"] + ["rc"] * (len(tiles) > 1)):
                case "k":
                    t["bindings"]["K"] = k + 1
                case "id":
                    t["id"] = "PouwH1WindowTile_v2"
                case "form":
                    t["form"] = "primitive"
                case "params":
                    body["params"] = body["params"][:4]
                case "rows":
                    n, x = Wt.chunks(k), rng.randrange(2)
                    rows = body["params"][x][1][1]
                    body["params"][x][1] = rng.choice([
                        ["Array", 0, ["Value", 16]], ["Value", 16], ["Array", rows, ["Array", 2, ["Value", 16]]],
                        ["Array", rows, row_type(k // n + 1)], ["Array", rows, row_type(k)] if n > 1 else ["Value", 16],
                        ["Array", rows // n, row_type(k)] if n > 1 else ["Array", rows, row_type(2 * k)],
                        ["Array", rows + 1, row_type(k // n)] if n > 1 else ["Array", rows, ["Value", 16]]])
                case "unit":
                    t["bindings"]["UNIT"] = {"fn": "Other_v1"}
                case "missing":
                    del defs[tid]
                case "rc":
                    body["params"][1][1][1] += 1
        case "tile-ks":
            k = max(k for k, _t in tiles) * 2
            unit = {"fn": f"H1Tile_v1{{K={k}}}"}
            extra = _template_id(k, unit)
            defs[extra] = _template(k, *lay["tile"], unit)
            desc["program_fn"] = record(tiles + [[k, {"fn": extra}]], lay)
        case "body":
            calls = [i for i, x in enumerate(root) if not desc["callees"][x["fn"]].startswith("Input")]
            i = rng.choice(calls)
            match rng.choice(["drop", "add", "zero", "scan", "foreign", "fn"]):
                case "drop":
                    del root[i]
                case "add":
                    root.insert(i, dict(root[i]))
                case "zero":
                    root[i] = {**root[i], "form": "batch", "n": 0}
                case "scan":
                    root[i] = {**root[i], "form": "scan", "n": 1}
                case "foreign":
                    root[i] = {**root[i], "fn": 1}
                case "fn":
                    root[i] = {**root[i], "fn": rng.choice([len(desc["callees"]), -1, "2", None])}
        case "layout-form":
            lin = lay["linears"]
            desc["program_fn"] = record(tiles, rng.choice([
                {**lay, "m": -1}, {**lay, "m": True}, {**lay, "m": None}, {**lay, "calls": []}, {**lay, "linears": []},
                {**lay, "linears": lin + [["x", 32]]}, {**lay, "calls": lay["calls"] + [[lin[0][0], -1]]},
                {**lay, "calls": [[5, 0]]}]))
        case "record-layout":
            desc["program_fn"] = record(tiles, {**lay, "calls": lay["calls"] + [["lm_head", 0]]})
        case "alt-layout":
            desc["program_fn"] = record(tiles, _alt_layout(rng, lay))
        case "tile-k":
            desc = window_desc(rng, _off_k(lay, rng))
        case _:
            raise ValueError(name)
    return desc_bytes(desc).hex()


#: each way a window's committed descriptor is no H1 window's, or not the one its record says
LAYOUT_DESC_NAMES = ["descriptor-form", "not-window", "template", "tile-ks", "body", "layout-form", "record-layout",
                     "alt-layout"]
#: and an H1 window's, its record and its nodes alike, with a tile at a K its proof doesn't cover
DESC_NAMES = [*LAYOUT_DESC_NAMES, "tile-k"]


def units_of(lay: dict) -> int:
    """The window's units N under `lay`."""
    ss, why = Cr.strata(lay)
    assert why is None, why
    return sum(hi - lo for s in ss for lo, hi in s["units"])


def _k(rng: random.Random, lay: dict) -> int:
    """A committed K the window `lay` can be drawn at: one to its N units."""
    n = units_of(lay)
    return rng.choice([1, n, rng.randint(1, n), rng.randint(1, n)])


def _bad_k(rng: random.Random) -> object:
    return rng.choice([0, -1, True, "3", None, [2], Cr.MAX_K + 1])


def _draw_args(rng: random.Random) -> dict:
    stream = rng.choice([_hex(rng), "h1/w-3", ""])
    receipt = rng.choice([{"receipts": _hex(rng)}, {}, {"a": "b", "pass": "x y~"}])
    return {"coin": _hex(rng), "stream": stream, "registration": _hex(rng), "receipt": receipt}


def _break_layout(lay: dict, rng: random.Random, name: str) -> dict:
    lin, (r, c) = lay["linears"], lay["tile"]
    match name:
        case "layout-form":
            return rng.choice([None, [], {**lay, "extra": 1}, {k: v for k, v in lay.items() if k != "m"},
                               {**lay, "tile": [r]}, {**lay, "tile": [0, c]}, {**lay, "m": -1}, {**lay, "m": True},
                               {**lay, "calls": []}, {**lay, "linears": []}, {**lay, "linears": lin + [["x", 32]]},
                               {**lay, "calls": lay["calls"] + [[lin[0][0], -1]]}, {**lay, "calls": [[5, 0]]}])
        case "named-twice":
            return {**lay, "linears": lin + [[lin[0][0], lin[0][1], lin[0][2]]]}
        case "unknown-linear":
            return {**lay, "calls": lay["calls"] + [["lm_head", 0]]}
        case "uncalled":
            return {**lay, "linears": lin + [["lm_head", lin[0][1], c]]}
        case "slot":
            return {**lay, "m": rng.choice([r * 3, r // 2 if r > 1 else 0, r * 2 + 1 if r > 1 else 3])}
        case "kn":
            bad = rng.choice([[lin[0][0], 0, lin[0][2]], [lin[0][0], lin[0][1], lin[0][2] + 1 if c > 1 else 0],
                              [lin[0][0], lin[0][1], 0]])
            return {**lay, "linears": [bad] + lin[1:]}
        case "blocks":
            return {**lay, "calls": lay["calls"] + [[lin[0][0], 1 + max(b for _l, b in lay["calls"]) + 1]]}
    raise ValueError(name)


def _off_k(lay: dict, rng: random.Random) -> dict:
    """`lay` with its first linear at a K its tile's proof doesn't cover: 2^17 or past, or no multiple of 32."""
    lin = lay["linears"]
    k = rng.choice([Cr.TILE_K_LIMIT, Cr.TILE_K_LIMIT + Cr.TILE_K_STEP, 2 * Cr.TILE_K_LIMIT, Cr.TILE_K_LIMIT - 22,
                    lin[0][1] + rng.choice([1, 16, 31]), 1, 33])
    return {**lay, "linears": [[lin[0][0], k, lin[0][2]]] + lin[1:]}


def drawn_instance(rng: random.Random, name: str = "honest") -> dict:
    """One `drawn` case: a draw coin's arguments and a window's layout, then mutation `name`."""
    d = {**_draw_args(rng), "layout": layout(rng, big=name == "budget")}
    d["k"] = PLAN_K if name == "budget" else _k(rng, d["layout"])
    match name:
        case "honest" | "budget":
            pass
        case "k-form":
            d["k"] = _bad_k(rng)
        case "k-past-units":
            d["k"] = units_of(d["layout"]) + rng.randint(1, 5)
        case "coin-form":
            d["coin"] = rng.choice([None, d["coin"][:-2], d["coin"].upper(), 5])
        case "args-form":
            k, bad = rng.choice([("stream", None), ("stream", 5), ("registration", "zz"),
                                 ("registration", d["registration"].upper()), ("receipt", None), ("receipt", ["a"]),
                                 ("receipt", {"a": 1}), ("receipt", {"a": 'quote"d'}), ("receipt", {"\\": "x"}),
                                 ("receipt", {"a": "é"})])
            d[k] = bad
        case _:
            d["layout"], d["k"] = _break_layout(d["layout"], rng, name), 1
    return d


#: `circuit.plan.RECORD_K`, the K a period commits for ε = 0.1% at δ = 2⁻⁴⁰
PLAN_K = 27_713
DRAWN_NAMES = ["budget", "coin-form", "args-form", "layout-form", "named-twice", "unknown-linear", "uncalled", "slot", "kn",
               "blocks", "k-form", "k-past-units"]
DRAWN_KEYS = ("coin", "stream", "registration", "receipt", "layout", "k")


def _draw_coin(rng: random.Random) -> dict:
    """A live opening of a window's draw coin as the challenger gives it (`challenger.LiveOpening`), at random bytes:
    `credit._window_drawn` reads only its coin."""
    return Ch.LiveOpening(rng.randrange(1 << 20), Wd.DRAW, 0, bytes.fromhex(_hex(rng)), bytes.fromhex(_hex(rng)),
                          bytes.fromhex(_hex(rng))[:32],
                          tuple(bytes.fromhex(_hex(rng)) for _ in range(rng.randint(0, 3)))).to_json()


def window_drawn_instance(rng: random.Random, name: str = "honest") -> dict:
    """One `window-drawn` case: a window's open root, its draw (its coin's live opening, its root and its receipts), its
    circuit's descriptor (`descriptor`, today's units or J10-R's) and a committed K, then mutation `name`."""
    lay, j10r = layout(rng), rng.random() < 0.5
    if j10r:
        lay["m"] = 2 * lay["tile"][0]
    d = {"window": {"root": _hex(rng)}, "draw": {"coin": _draw_coin(rng), "root": _hex(rng), "receipts": _hex(rng)},
         "descriptor": descriptor(rng, lay, j10r=j10r), "k": _k(rng, lay)}
    match name:
        case "honest":
            pass
        case "coin-form":
            o = d["draw"]["coin"]
            d["draw"]["coin"] = rng.choice([None, 5, {}, {k: v for k, v in o.items() if k != "salt"}, {**o, "extra": 1},
                                            {**o, "epoch": True}, {**o, "label": 5}, {**o, "coin": "zz"},
                                            {**o, "path": ["ab"]}, {**o, "path": "x"}])
        case "args-form":
            where, bad = rng.choice([("window", {}), ("window", {"root": 5}), ("root", None), ("root", "zz"),
                                     ("root", d["draw"]["root"].upper()), ("receipts", None), ("receipts", 5),
                                     ("receipts", 'quote"d')])
            if where == "window":
                d["window"] = bad
            elif bad is None:
                del d["draw"][where]
            else:
                d["draw"][where] = bad
        case "descriptor-hex":
            h = d["descriptor"]
            d["descriptor"] = rng.choice([h.upper(), h[:-1], "", "zz", 5, None])
        case "k-form":
            d["k"] = _bad_k(rng)
        case "k-past-units":
            d["k"] = units_of(lay) + rng.randint(1, 5)
        case _:
            raise ValueError(name)
    return d


WINDOW_DRAWN_NAMES = ["coin-form", "args-form", "descriptor-hex", "k-form", "k-past-units"]
WINDOW_DRAWN_KEYS = ("window", "draw", "descriptor", "k")


def _proofs(rng: random.Random, units: list[int]) -> list[dict]:
    """An accepted outcome of each of `units`, as the verifier of record gives it (`credit.PROOF_KEYS`), of a random
    circuit and registration: `_outcome` reads only its unit and verdict."""
    program = _hex(rng)
    proofs = [{"unit": u, "verdict": "accept", "program": program,
               "registered": {"a0": {"value": "h1/a/64", "root": _hex(rng)}}} for u in units]
    rng.shuffle(proofs)
    return proofs


def _window_proofs(rng: random.Random, d: dict, units: list[int]) -> list[dict]:
    """Each of `units`' outcome as the verifier of record gives it for its session, verified against the window's
    descriptor: accepted, its program the descriptor's SHA-512, and its session's registered finding each of its tile's
    A ports (``a<i>``, up to two) at its K's A value (or its call's round's A table of K, and its ``zy`` port at its
    call's z table, when the window registers its rounds' tables) and each B port (``b<j>``) at its K's B value, each at
    the root the window's entry registers."""
    ss, _ = Cr.strata(d["layout"])
    r, c = d["layout"]["tile"]
    program = hashlib.sha512(bytes.fromhex(d["descriptor"])).hexdigest()
    roots = {e["value"]: e["root"] for e in d["entries"]}
    starts = list(itertools.accumulate((size for size, _w in Cr.call_work(d["layout"])), initial=0))
    per = max(1, len(d["layout"]["calls"]) // len(d["rounds"]))
    out = []
    for u in units:
        k = next(s["template"][1:] for s in ss if any(lo <= u < hi for lo, hi in s["units"]))
        call = next(i for i in range(len(starts) - 1) if starts[i] <= u < starts[i + 1])
        b = next(v for v in roots if v.split("/")[-2:] == ["b", k])
        a = next((v for v in roots if v.endswith(f"/a/{k}@{call // per}")), None)
        zy = {"zy": {"value": a.rsplit("/", 2)[0] + f"/zy@{call}"}} if a is not None else {}
        a = a or next(v for v in roots if v.endswith(f"/a/{k}"))
        reg = {**{f"a{i}": {"value": a, "root": roots[a]} for i in range(min(r, 2))},
               **{f"b{j}": {"value": b, "root": roots[b]} for j in range(min(c, 2))},
               **{port: {**x, "root": roots[x["value"]]} for port, x in zy.items()}}
        out.append({"unit": u, "verdict": "accept", "program": program, "registered": reg})
    rng.shuffle(out)
    return out


def _opened(rng: random.Random, d: dict, kind: str) -> str:
    """A value of `kind` (``a`` or ``b``) some drawn unit's proof opened."""
    return rng.choice(sorted({x["value"] for p in d["proofs"] for x in p["registered"].values()
                              if x["value"].split("/")[-2] == kind}))


def _other_root(rng: random.Random, proofs: list[dict]) -> None:
    """One proof's session opened one of its values against another root than the window's: another registration's."""
    p = rng.choice(proofs)
    v = rng.choice(sorted({x["value"] for x in p["registered"].values()}))
    root = _hex(rng)
    for x in p["registered"].values():
        if x["value"] == v:
            x["root"] = root


def _mutate_proofs(proofs: list[dict], rng: random.Random, name: str, units: list[int]) -> list[dict]:
    j = rng.randrange(len(proofs))
    match name:
        case "missing":
            del proofs[j]
        case "failed":
            proofs[j]["verdict"] = "reject"
        case "twice":
            proofs.insert(rng.randrange(len(proofs) + 1), dict(proofs[j]))
        case "undrawn":
            picked = set(units)
            u = next(u for u in itertools.count(rng.randrange(1 << 20)) if u not in picked)
            proofs.insert(rng.randrange(len(proofs) + 1), {**proofs[j], "unit": u})
        case "proofs-form":
            p, x = proofs[j], next(iter(proofs[j]["registered"].values()))
            bad = rng.choice([{**p, "verdict": "maybe"}, {**p, "why": "x"}, {**p, "unit": -1}, {**p, "program": "zz"},
                              {**p, "program": None}, {k: v for k, v in p.items() if k != "registered"},
                              {**p, "registered": [x]}, {**p, "registered": {"a0": {**x, "root": "zz"}}},
                              {**p, "registered": {"a0": {**x, "extra": 1}}}, {**p, "registered": {"a0": {**x, "value": 5}}},
                              {**p, "registered": {"a\n": x}}, {**p, "registered": {"a0": {**x, "value": "é"}}}])
            proofs = rng.choice([None, "x", proofs + [5], proofs[:j] + [bad] + proofs[j + 1:]])
        case _:
            raise ValueError(name)
    return proofs


def _partial(rng: random.Random) -> tuple[dict, list[int]]:
    """A draw whose coin leaves some of the window's units undrawn, and its units."""
    while True:
        d = {**_draw_args(rng), "layout": layout(rng)}
        d["k"] = _k(rng, d["layout"])
        units, _ = Cr.drawn(*(d[k] for k in DRAWN_KEYS))
        if len(units) < sum(hi - lo for s in Cr.strata(d["layout"])[0] for lo, hi in s["units"]):
            return d, units


def outcome_instance(rng: random.Random, name: str = "honest") -> dict:
    """One `outcome` case: a draw's arguments and an accepted proof of each unit its coin draws, then mutation `name`."""
    if name == "chosen":
        d, units = _partial(rng)
        other = {**d, "coin": _hex(rng)}
        chosen, _ = Cr.drawn(*(other[k] for k in DRAWN_KEYS))
        return {**d, "proofs": _proofs(rng, chosen if chosen != units else units[1:] + [max(units) + 1])}
    d = {**_draw_args(rng), "layout": layout(rng)}
    d["k"] = _k(rng, d["layout"])
    units, _ = Cr.drawn(*(d[k] for k in DRAWN_KEYS))
    proofs = _proofs(rng, units)
    return {**d, "proofs": proofs if name == "honest" else _mutate_proofs(proofs, rng, name, units)}


OUTCOME_NAMES = ["missing", "failed", "twice", "undrawn", "chosen", "proofs-form"]
OUTCOME_KEYS = (*DRAWN_KEYS, "proofs")


def _served_units(d: dict, stream: str | None = None, k: int | None = None) -> list[int]:
    """The units `d`'s draw coin draws at its K (or `k`) under the served window's arguments (`credit.window`'s), or
    under `stream`."""
    coin = d["draw"]["coin"]["coin"]
    got, why = Cr.drawn(coin, d["window"]["root"] if stream is None else stream, d["draw"]["root"],
                        {"receipts": d["draw"]["receipts"]}, d["layout"], d["k"] if k is None else k)
    assert why is None, why
    return got


def b_counts(lay: dict) -> dict[int, int]:
    """Each K of layout `lay` and its B rows, its linears' n in turn."""
    rows: dict[int, int] = {}
    for _name, k, n in lay["linears"]:
        rows[k] = rows.get(k, 0) + n
    return rows


def weights_of(rng: random.Random, lay: dict, program: str, prefix: str) -> list[dict]:
    """Each K's ``<prefix>b/<K>`` of layout `lay` as one-stage registers it under the circuit `program`, binding no
    window (`_b_entry`), at a random root."""
    return [_b_entry(program, f"{prefix}b/{k}", k, n, rng) for k, n in sorted(b_counts(lay).items())]


def fold(weights: list[dict]) -> str:
    """`weights.weights_root` of `weights`' roots, in hex: the root a period's open signs."""
    return Wt.weights_root({int(e["value"].rsplit("/", 1)[1]): bytes.fromhex(e["root"]) for e in weights}).hex()


#: the model a period's open names (`credit.MODEL_FIELD`): a role credit pins, at its pinned checkpoint
MODEL = {"name": min(Cr.PINNED_MODELS), "checkpoint_sha512": Cr.PINNED_MODELS[min(Cr.PINNED_MODELS)]}
#: each way a period's open names no model credit pins
MODEL_NAMES = ["no-model", "model-form", "other-checkpoint", "other-model"]


def signs(weights: list[dict]) -> dict:
    """What a period's open signs of its weights: their `fold` and the model they are of (`MODEL`)."""
    return {Cr.WEIGHTS_FIELD: fold(weights), Cr.MODEL_FIELD: dict(MODEL)}


def _unmodel(rng: random.Random, opened: dict, kind: str) -> None:
    """A period-open's fields `opened` with its model broken as `kind`: none (a bare weights' root, as the gate signed
    before its model registry), malformed, another checkpoint than the pinned one, or a role credit doesn't pin."""
    match kind:
        case "no-model":
            del opened[Cr.MODEL_FIELD]
        case "model-form":
            opened[Cr.MODEL_FIELD] = rng.choice([
                None, 5, MODEL["name"], opened[Cr.WEIGHTS_FIELD], [dict(MODEL)], {"name": MODEL["name"]},
                {**MODEL, "extra": 1}, {**MODEL, "name": 5}, {**MODEL, "name": None},
                {"name": MODEL["name"], "checkpoint": MODEL["checkpoint_sha512"]}])
        case "other-checkpoint":
            c = MODEL["checkpoint_sha512"]
            opened[Cr.MODEL_FIELD] = {**MODEL, "checkpoint_sha512": rng.choice([_hex(rng), c.upper(), c[:-2], None, 5])}
        case "other-model":
            opened[Cr.MODEL_FIELD] = {**MODEL, "name": rng.choice([n for n in (
                "LLAMA31_70B_INT8", "QWEN3_06B_INT8", MODEL["name"].lower(), "", MODEL["name"] + " ")
                if n not in Cr.PINNED_MODELS])}
        case _:
            raise ValueError(kind)


def b_rows(rng: random.Random, k: int, n: int) -> list[tuple[bytes, float]]:
    """`n` B rows of K, each its K int8 codes and their f32 s_w."""
    return [(rng.randbytes(k), rng.choice([2.0 ** -9, 0.0123, 3.5])) for _ in range(n)]


def b_row_words(codes: bytes, sw: float) -> list[int]:
    """A B row's u16 words as the prover and the auditor's registry commit them (`weights.row_words` of
    `weights.row_bytes`): its K int8 codes then its f32 s_w (`h1zy.ROW`), little-endian in pairs."""
    return Wt.row_words(Wt.row_bytes(bytes(codes), struct.pack("<f", sw)))


def b_leaf_words(codes: bytes, sw: float) -> list[list[int]]:
    """A B row of K's leaves' u16 words: x = `weights.chunks` (K) leaves, leaf c chunk c's K/x codes then the row's s_w
    (`b_row_words` of each)."""
    x = Wt.chunks(len(codes))
    kc = len(codes) // x
    return [b_row_words(codes[c * kc:(c + 1) * kc], sw) for c in range(x)]


def registered_b(program: str, v: str, rows: list[tuple[bytes, float]]) -> dict:
    """The entry of `rows` registered as the served path registers each K's B rows (`registered.Registered`, each row
    its `b_leaf_words`, salts drawn once, binding no window)."""
    words = [w for codes, sw in rows for w in b_leaf_words(codes, sw)]
    return G.Registered.commit(program, v, words, G.supplied_salts(program, v, len(words)), schema=G.SCHEMA_V2).entry()


def _rehome(weights: list[dict], program: str, prefix: str) -> list[dict]:
    """`weights`' roots registered under another circuit `program` and `prefix`."""
    out = []
    for e in weights:
        v = prefix + "b/" + e["value"].rsplit("/", 1)[1]
        out.append({**e, "value": v, "domain": G.domain(program, v, e["words"], e["leaves"]).domain_id.hex()})
    return out


def round_choices(lay: dict, name: str = "honest") -> list[int]:
    """The round counts an honest window of layout `lay` may have under mutation `name`: its calls the rounds' equal
    blocks, each calling every linear at most once (`_once`; two calls a round or more under ``rows-other``), at most
    three when one is."""
    lins = [lin for lin, _blk in lay["calls"]]
    once = [s for s in _once(lins) if name != "rows-other" or len(lins) // s > 1]
    return [s for s in once if s <= 3] or once[:1]


#: each way the tables a window's live units read are not its registration's, made before its draw: one round's slot
#: table unregistered (``live-no-slots``) or of other than T rows (``live-slots-rows``), and round 0's request table
#: unregistered (``live-no-requests``) or of other than T rows (``live-requests-rows``)
LIVE_TABLE_NAMES = ["live-no-slots", "live-slots-rows", "live-no-requests", "live-requests-rows"]
#: each way a window's live evidence does not prove each step's served rows: its form, a descriptor that is no canonical
#: descriptor (``live-descriptor-form``), no live Program or one whose units are not S calls of its unit
#: (``live-body``), a live Program at another T, S or PREFIX, and its rounds' proof outcomes: one missing, twice, past the
#: rounds, rejected, verified against another circuit, opening ``m@<s>`` at another m than the manifest's m_s
#: (``live-m-other``), ``round@<s>`` at another round than s (``live-round-other``), another round's slot table
#: (``live-slots-other``), public caps other than the max_tokens its period's open committed (``live-caps-swapped``: the
#: requests' max_tokens in reverse order, so a hidden header could take another request's cap and its decode rows;
#: ``live-caps-other``: one changed, one more or one fewer, or another value) or other ok rows than ``live-ok``
#: (``live-ok-off``, a unit that computed 0), and their form; and an honest v1 live Program (``live-v1``), whose units
#: don't bind each request's cap to its public max_tokens
LIVE_NAMES = [*LIVE_TABLE_NAMES, "live-form", "live-descriptor-form", "live-not-live", "live-body", "live-t-other",
              "live-s-other", "live-prefix", "live-missing", "live-twice", "live-past", "live-failed",
              "live-other-circuit", "live-m-other", "live-round-other", "live-slots-other", "live-caps-swapped", "live-caps-other",
              "live-ok-off", "live-proofs-form", "live-v1"]


def live_program(t: int, steps: int, prefix: str, version: int = 2) -> dict:
    """A live Program's descriptor (`circuit.live.program_v2`'s encoding as far as `credit.live_of` reads it, or
    `circuit.live.program`'s at `version` 1): its program function's record T, S and PREFIX, its unit
    ``PouwLiveSlots_v2{T=<T>}`` (a composite bound to T alone) and its root's input batches (S·T slot words, T headers,
    S m's, S rounds and, at v2, T caps, as bits) then S calls of the unit, round s's unit s."""
    fn_id, unit_id = f"PouwLiveWindow_v{version}", f"PouwLiveSlots_v{version}"
    fn = f"{fn_id}{{T={t},S={steps},PREFIX={Cr._canonical(prefix)}}}"
    unit, root = f"{unit_id}{{T={t}}}", f"Root[{fn_id}]_v1"
    batches = (steps * t * 32, t * 160, steps * 32, steps * 32) + ((t * 32,) if version == 2 else ())
    nodes = [{"form": "batch", "fn": 0, "n": n} for n in batches]
    nodes += [{"form": "call", "fn": 1} for _s in range(steps)]
    return {"schema": Cr.SCHEMA, "callees": ["Input1_v1", unit], "program_fn": fn, "root": root,
            "definitions": {"Input1_v1": {"form": "primitive", "id": "Input1_v1"},
                            root: {"form": "composite", "id": root, "bindings": {},
                                   "body": {"nodes": nodes, "params": []}},
                            unit: {"form": "composite", "id": unit_id, "bindings": {"T": t},
                                   "body": {"nodes": [], "params": []}}}}


def live_hex(p: dict) -> str:
    return Cr._canonical(p).encode().hex()


def _live_ms(d: dict, steps: int) -> list[int]:
    """Each step's served rows m_s as the manifest states them (0 for a malformed one, which an earlier step refuses)."""
    per = len(d["calls"]) // steps
    ms = [d["calls"][s * per].get(Cr.SERVED_ROWS) for s in range(steps)]
    return [m if type(m) is int and 0 <= m < 1 << 32 else 0 for m in ms]


def live_proof(rng: random.Random, d: dict, desc: str, s: int, m: int, prefix: str | None = None,
               max_tokens: list[int] | None = None) -> dict:
    """Round s's accepted live@<s> proof outcome, as the verifier of record gives it on the ``--registered`` file
    `live_reads.reads_file` builds for the window `d` of PREFIX `prefix` (its descriptor's when None) at served rows
    `m` and the public caps of `max_tokens` (its period's fixed length when None): each port at its value and root (a
    random root for a table the window doesn't register)."""
    steps, t = len(d["rounds"]), d["rows"]
    prefix = Cr._prefix(bytes.fromhex(d["descriptor"])) if prefix is None else prefix
    program = hashlib.sha512(bytes.fromhex(d["descriptor"])).hexdigest()
    roots = {e["value"]: e["root"] for e in d["entries"] if isinstance(e, dict) and "value" in e}
    for v in [LR.table(prefix, LR.SLOTS, x) for x in range(steps)] + [LR.table(prefix, LR.REQUESTS)]:
        if v not in roots:
            roots[v] = _hex(rng)
    mt = d["length"]["max_tokens"] if max_tokens is None else max_tokens
    reg = LR.reads_file(program, t, steps, s, roots, d["window"]["root"], d["sid"], m, prefix, mt)
    return {"unit": s, "verdict": "accept", "program": hashlib.sha512(bytes.fromhex(desc)).hexdigest(),
            "registered": {port: {"value": x["value"], "root": x["root"]} for port, x in reg["ports"].items()}}


def live_evidence(rng: random.Random, d: dict, prefix: str) -> dict:
    """The window `d`'s live evidence (`credit.LIVE_KEYS`): its live Program at its period's T, its S rounds and its
    circuit's PREFIX `prefix` (given, since a broken descriptor's names none), and each round's accepted proof outcome
    (`live_proof`) at its step's m_s, in any order."""
    steps = len(d["rounds"])
    desc = live_hex(live_program(d["rows"], steps, prefix))
    proofs = [live_proof(rng, d, desc, s, m, prefix) for s, m in enumerate(_live_ms(d, steps))]
    rng.shuffle(proofs)
    return {"descriptor": desc, "proofs": proofs}


def _break_live_desc(rng: random.Random, d: dict, name: str) -> str:
    t, steps, prefix = d["rows"], len(d["rounds"]), Cr._prefix(bytes.fromhex(d["descriptor"]))
    p = live_program(t, steps, prefix)
    unit, root = p["callees"][1], p["root"]
    match name:
        case "live-descriptor-form":
            return rng.choice([json.dumps(p).encode().hex(), live_hex({**p, "schema": "verity-ir/descriptor/v0"}),
                               live_hex({k: v for k, v in p.items() if k != "callees"}), b"\xff\xfe".hex(),
                               live_hex([p])])
        case "live-not-live":
            pre = Cr._canonical(prefix)
            p["program_fn"] = rng.choice([f"{Cr.LIVE_FN}{{T=0,S={steps},PREFIX={pre}}}",
                                          f"{Cr.LIVE_FN}{{T={t},S={steps}}}",
                                          f"{Cr.LIVE_FN}{{T={t},S={steps},PREFIX=5}}",
                                          f"{Cr.LIVE_FN}{{S={steps},T={t},PREFIX={pre}}}",
                                          f"{Cr.LIVE_FN}{{T={t},S=0{steps},PREFIX={pre}}}",
                                          f"PouwLiveWindow_v1{{T={t},S={steps},PREFIX={pre}}}"])
            if rng.random() < 0.3:
                return d["descriptor"]
        case "live-body":
            nodes = p["definitions"][root]["body"]["nodes"]
            match rng.randrange(6):
                case 0:
                    p["definitions"][unit]["bindings"] = {"T": t, "N": 1}
                case 1:
                    p["definitions"][unit]["bindings"] = {"T": t + 1}
                case 2:
                    p["definitions"][unit]["id"] = "PouwLiveSlots_v1"
                case 3:
                    nodes.append({"form": "call", "fn": 1}) if rng.random() < 0.5 else nodes.pop()
                case 4:
                    p["callees"].append("And2_v1")
                    p["definitions"]["And2_v1"] = {"form": "primitive", "id": "And2_v1"}
                    nodes.append({"form": "call", "fn": 2})
                case _:
                    del p["definitions"][unit]
        case "live-t-other":
            p = live_program(rng.choice([x for x in (t - 1, t + 1, 2 * t) if x >= 1 and x != t]), steps, prefix)
        case "live-s-other":
            p = live_program(t, rng.choice([x for x in (steps - 1, steps + 1) if x >= 1]), prefix)
        case "live-prefix":
            p = live_program(t, steps, rng.choice([f"other-{rng.randrange(100)}/", prefix + "x/", ""]))
    return live_hex(p)


def _mutate_live(rng: random.Random, d: dict, name: str) -> None:
    live, steps = d["live"], len(d["rounds"])
    proofs = live["proofs"]
    match name:
        case "live-form":
            h = live["descriptor"]
            d["live"] = rng.choice([None, [live], {**live, "statements": []}, {"descriptor": h},
                                    {**live, "descriptor": h.upper()}, {**live, "descriptor": h[:-1]},
                                    {**live, "descriptor": 5}])
        case "live-descriptor-form" | "live-not-live" | "live-body" | "live-t-other" | "live-s-other" | "live-prefix":
            live["descriptor"] = _break_live_desc(rng, d, name)
        case "live-missing":
            del proofs[rng.randrange(len(proofs))]
        case "live-twice":
            proofs.insert(rng.randrange(len(proofs) + 1), json.loads(json.dumps(rng.choice(proofs))))
        case "live-past":
            proofs.append({**rng.choice(proofs), "unit": steps + rng.randrange(3)})
        case "live-failed":
            rng.choice(proofs)["verdict"] = "reject"
        case "live-other-circuit":
            rng.choice(proofs)["program"] = _hex(rng)
        case "live-m-other":
            i = rng.randrange(len(proofs))
            s = proofs[i]["unit"]
            m = _live_ms(d, steps)[s]
            other = rng.choice([x for x in (m - 1, m + 1, rng.randint(0, d["rows"])) if x >= 0 and x != m])
            proofs[i] = live_proof(rng, d, live["descriptor"], s, other)
        case "live-round-other":
            p = rng.choice(proofs)
            raw = bytes.fromhex(d["descriptor"])
            x = rng.choice([y for y in (p["unit"] + 1, p["unit"] - 1, rng.randrange(1 << 16)) if y >= 0 and y != p["unit"]])
            reg = LR.round_table(hashlib.sha512(raw).hexdigest(), Cr._prefix(raw), d["window"]["root"], x)[2]
            p["registered"]["s"] = {"value": rng.choice([reg.value, LR.table(Cr._prefix(raw), LR.ROUND_ROWS, p["unit"])]),
                                    "root": reg.root.hex()}
        case "live-slots-other":
            p = rng.choice(proofs)
            ports = [port for port in p["registered"] if port.startswith("slot_")]
            prefix = Cr._prefix(bytes.fromhex(d["descriptor"]))
            others = [e for e in d["entries"] if e["value"].startswith(f"{prefix}{LR.SLOTS}@")
                      and e["value"] != p["registered"][ports[0]]["value"]]
            for port in rng.sample(ports, rng.randint(1, len(ports))):
                p["registered"][port] = ({"value": others[0]["value"], "root": others[0]["root"]} if others
                                         else {**p["registered"][port], "root": _hex(rng)})
        case "live-caps-swapped":
            mt, i = list(d["length"]["max_tokens"]), rng.randrange(len(proofs))
            other = mt[::-1] if mt[::-1] != mt else [mt[0] + 1, *mt[1:]]
            proofs[i] = live_proof(rng, d, live["descriptor"], proofs[i]["unit"], _live_ms(d, steps)[proofs[i]["unit"]],
                                   max_tokens=other)
        case "live-caps-other":
            mt, i = list(d["length"]["max_tokens"]), rng.randrange(len(proofs))
            q = rng.randrange(len(mt))
            others = [mt[:q] + [mt[q] + 1] + mt[q + 1:]] + ([mt + [1]] if len(mt) < d["rows"] else [])
            others += [mt[:q] + [mt[q] - 1] + mt[q + 1:]] if mt[q] > 1 else []
            others += [mt[:-1]] if len(mt) > 1 else []
            others += [mt[::-1]] if mt[::-1] != mt else []
            proofs[i] = live_proof(rng, d, live["descriptor"], proofs[i]["unit"], _live_ms(d, steps)[proofs[i]["unit"]],
                                   max_tokens=rng.choice(others))
            if rng.random() < 0.3:
                p = proofs[i]
                port = rng.choice([x for x in p["registered"] if x.startswith("cap_")])
                p["registered"][port] = rng.choice([{**p["registered"][port], "root": _hex(rng)},
                                                    {**p["registered"][port], "value": p["registered"]["m"]["value"]}])
        case "live-v1":
            prefix = Cr._prefix(bytes.fromhex(d["descriptor"]))
            desc = live_hex(live_program(d["rows"], steps, prefix, version=1))
            program = hashlib.sha512(bytes.fromhex(d["descriptor"])).hexdigest()
            roots = {e["value"]: e["root"] for e in d["entries"] if isinstance(e, dict) and "value" in e}
            ps = []
            for s, m in enumerate(_live_ms(d, steps)):
                reg = LR.reads_file(program, d["rows"], steps, s, roots, d["window"]["root"], d["sid"], m, prefix)
                ps.append({"unit": s, "verdict": "accept", "program": hashlib.sha512(bytes.fromhex(desc)).hexdigest(),
                           "registered": {port: {"value": x["value"], "root": x["root"]} for port, x in reg["ports"].items()}})
            d["live"] = {"descriptor": desc, "proofs": ps}
        case "live-ok-off":
            rng.choice(proofs)["registered"][LR.OK_PORT]["root"] = _hex(rng)
        case "live-proofs-form":
            p = rng.choice(proofs)
            match rng.randrange(5):
                case 0:
                    live["proofs"] = rng.choice([None, {"0": p}, 5])
                case 1:
                    del p["registered"]
                case 2:
                    p["verdict"] = "maybe"
                case 3:
                    p["unit"] = rng.choice([-1, True, "0", None])
                case _:
                    p["registered"][LR.OK_PORT]["root"] = "zz"
        case _:
            raise ValueError(name)


def honest_window(rng: random.Random, name: str = "honest", sid: str = "s-1", leaf: int = rc.RECORD_LEAF,
                  lay: dict | None = None, desc: str | None = None, prefix: str | None = None,
                  weights: list[dict] | None = None, t: int | None = None, length: list[int] | None = None,
                  j10r: bool = False) -> tuple[dict, rc.Session]:
    """An honest served window (`honest_credit`) of the circuit `desc` (`descriptor` of its layout `lay`, a random one
    when None), its weights `weights` (`weights_of` when None, its open root over them, ``d["weights"]``), a K it can be
    drawn at, its period's table rows T `t` (``rows``, `table_rows` when None) and each step's served rows at most T,
    one call per manifest row, and an accepted proof of each unit its draw coin
    draws at that K, and its live evidence (``live``, `live_evidence`): its live Program and each round's accepted
    live@<s> proof outcome at its step's m_s. Its A rows are its rounds' A tables of each K, as main's served form
    registers them
    (`h1window.table`), beside each round's input-ids table, over rounds that call each linear once (one round under
    `TWO_CALLS`), and its Commit record registers each of them (`commit_record`); under `AROW_NAMES`
    one of them is not the Commit's. Its calls' served rows are the schedule of the fixed length `length`
    (`fixed_length` when None, aimed at ``length-past`` and ``length-early``; `length_rows`), then `ROWS_NAMES`'
    mutation, ``d["length"]`` the ``{max_tokens, stop}`` its period's open commits. ``layout`` is the layout, which the verifier
    reads from the descriptor. Its values are named `prefix` (a random ``h1/w-<n>/`` when None) and their roles, and so
    are its circuit's parameters (``PREFIX``) when `desc` is None, as a served path names them, over J10-R units when
    `j10r`."""
    lay = layout(rng) if lay is None else lay
    prefix = f"h1/w-{rng.randrange(100)}/" if prefix is None else prefix
    desc = descriptor(rng, lay, prefix, j10r=j10r) if desc is None else desc
    program = hashlib.sha512(bytes.fromhex(desc)).hexdigest()
    weights = weights_of(rng, lay, program, prefix) if weights is None else weights
    other = {"other-registration": rng.choice(["window", "program"]), "a-call-unbound": "unbound"}.get(name)
    k_of = {lin: k for lin, k, _n in lay["linears"]}
    t = table_rows(rng, lay) if t is None else t
    aim = name if name in ("length-past", "length-early") else None
    choices = [s for s in round_choices(lay, name) if s >= 2] if aim else round_choices(lay, name)
    steps = 1 if name in TWO_CALLS else rng.choice(choices)
    if length is None:
        length = fixed_length(rng, t, steps, aim)
    d, s = honest_credit(rng, name if name in [*CREDIT_NAMES, "never-received"] else "honest", rows=len(lay["calls"]),
                         sid=sid, leaf=leaf, ks=sorted(set(k_of.values())), program=program, other=other,
                         calls=[k_of[lin] for lin, _blk in lay["calls"]], prefix=prefix, weights=weights,
                         off=name if name in [*AROW_NAMES, *IDS0_NAMES, *REQ0_NAMES, *LIVE_TABLE_NAMES] else None,
                         m=lay["m"], rounds=steps, t=t,
                         served=served_rows(rng, t, len(lay["calls"]), steps, name, length_rows(rng, length, steps, t)))
    d["length"] = {"max_tokens": length, "stop": Wd.STOP_FIXED}
    d["layout"], d["k"], d["descriptor"], d["weights"], d["rows"] = lay, _k(rng, lay), desc, weights, t
    d["proofs"] = _window_proofs(rng, d, _served_units(d))
    d["sid"] = sid
    d["live"] = live_evidence(rng, d, prefix)
    return d, s


def window_instance(rng: random.Random, name: str = "honest", lay: dict | None = None) -> dict:
    """One `window` case: an honest window (of layout `lay` when given), then mutation `name` (``j10r``: an honest window
    over J10-R units, each call of at least two row tiles)."""
    credit = name.removeprefix("credit-")

    def call_ks(lay: dict) -> list[int]:
        ks = {lin: k for lin, k, _n in lay["linears"]}
        return [ks[lin] for lin, _b in lay["calls"]]

    if name == "proof-other-call":
        lay = layout(rng)
        while len(set(call_ks(lay))) < 2:
            lay = layout(rng)
    elif name in TWO_CALLS:
        lay = layout(rng)
        while len(set(call_ks(lay))) == len(lay["calls"]):
            lay = layout(rng)
    elif name == "rows-other":
        lay = layout(rng)
        while not any(len(lay["calls"]) // s > 1 for s in _once([lin for lin, _blk in lay["calls"]])):
            lay = layout(rng)
    elif name == "commit-missing-expert":
        lay = expert_layout(rng)
    elif name in ("length-past", "length-early"):
        lay = layout(rng)
        while max(round_choices(lay)) < 2:
            lay = layout(rng)
    elif name == "j10r":
        lay = layout(rng)
        lay["m"] = 2 * lay["tile"][0]
    d, s = honest_window(rng, credit, lay=lay, j10r=name == "j10r")
    signed: dict | None = {}
    committed: object = "same"
    mt = list(d["length"]["max_tokens"])
    match name:
        case "honest" | "j10r":
            pass
        case "length-none":
            committed = rng.choice([None, "absent"])
        case "length-eos":
            committed = {**d["length"], "stop": rng.choice(["eos-token", "eos", "stop", ""])}
        case "length-form":
            committed = rng.choice([{"max_tokens": []}, {"max_tokens": [], "stop": Wd.STOP_FIXED},
                                    {"max_tokens": [0], "stop": Wd.STOP_FIXED},
                                    {"max_tokens": [True], "stop": Wd.STOP_FIXED},
                                    {"max_tokens": [-1], "stop": Wd.STOP_FIXED},
                                    {"max_tokens": "4", "stop": Wd.STOP_FIXED},
                                    {"max_tokens": mt, "stop": 5}, {"max_tokens": mt, "stop": None},
                                    {**d["length"], "served": mt}, {"stop": Wd.STOP_FIXED}, 5, [mt], "fixed"])
        case "length-past":
            q = next(q for q, t in enumerate(mt) if t >= 2)
            committed = {**d["length"], "max_tokens": mt[:q] + [mt[q] - 1] + mt[q + 1:]}
        case "length-early":
            q = mt.index(min(mt))
            committed = {**d["length"], "max_tokens": mt[:q] + [mt[q] + 1] + mt[q + 1:]}
        case "length-slots":
            committed = {**d["length"], "max_tokens": mt + [1] * (d["layout"]["m"] + 1 - len(mt))}
        case _ if (name in AROW_NAMES or name in IDS0_NAMES or name in REQ0_NAMES or name in ROWS_NAMES
                   or name in LIVE_TABLE_NAMES):
            pass
        case _ if name in LIVE_NAMES:
            _mutate_live(rng, d, name)
        case "commit-unbound":
            rec = json.loads(bytes.fromhex(d["commit"]))
            rec["registered"].append({"value": "sampled-ids/0", "root": _hex(rng), "leaves": 1, "words": 2,
                                      "domain": _hex(rng)})
            d["commit"] = Cr._canonical(rec).encode().hex()
        case "manifest-count":
            lin = d["layout"]["linears"][0][0]
            d["layout"]["calls"].insert(rng.randrange(len(d["layout"]["calls"]) + 1), [lin, 0])
            prefix = Cr._prefix(bytes.fromhex(d["descriptor"]))
            d["descriptor"] = descriptor(rng, d["layout"], prefix)
            program = hashlib.sha512(bytes.fromhex(d["descriptor"])).hexdigest()
            b = {e["value"] for e in d["weights"]}
            d["entries"] = [e for e in d["entries"] if e["value"] not in b] + _rehome(d["weights"], program, prefix)
        case "chosen":
            d, s = honest_window(rng)
            while True:
                chosen = _served_units(d, stream=_hex(rng))
                if chosen != sorted(p["unit"] for p in d["proofs"]):
                    break
                d, s = honest_window(rng)
            d["proofs"] = _window_proofs(rng, d, chosen)
        case "failed" | "missing":
            d["proofs"] = _mutate_proofs(d["proofs"], rng, name, [])
        case "descriptor-hex":
            h = d["descriptor"]
            d["descriptor"] = rng.choice([h.upper(), h[:-1], "", "zz", 5, None])
        case _ if name in DESC_NAMES:
            d["descriptor"] = _break_desc(rng, d["layout"], name)
        case "k-form":
            d["k"] = _bad_k(rng)
        case "k-past-units":
            d["k"] = units_of(d["layout"]) + rng.randint(1, 5)
        case "t-other":
            d["rows"] = other_rows(rng, d["layout"])
        case "t-form":
            d["rows"] = bad_rows(rng, d["rows"])
        case "never-received":
            pass
        case "round-coin":
            coin = d["rounds"][rng.randrange(len(d["rounds"]))]["coin"]
            coin["after"] = rc._flip(coin["after"])
        case "a-untied":
            e = rng.choice([e for e in d["entries"] if Cr.is_a(e["value"])])
            e["root"] = _tree(rng, e["leaves"])[1][-1][0].hex()
        case "no-a":
            d["entries"] = [e for e in d["entries"] if not Cr.is_a(e["value"])]
        case "other-registration" | "a-call-unbound":
            pass
        case "proof-other-registration":
            _other_root(rng, d["proofs"])
        case "entry-root":
            v = _opened(rng, d, "b")
            next(e for e in d["entries"] if e["value"] == v)["root"] = _hex(rng)
        case "proof-other-circuit":
            rng.choice(d["proofs"])["program"] = _hex(rng)
        case "proof-other-call":
            p = rng.choice(d["proofs"])
            mine = next(x["value"] for port, x in p["registered"].items() if port.startswith("a"))
            other = rng.choice([e for e in d["entries"] if Cr.a_value(e["value"]) and e["value"] != mine])
            for port in [port for port in p["registered"] if port.startswith("a")]:
                p["registered"][port] = {"value": other["value"], "root": other["root"]}
        case "proof-no-a":
            p = rng.choice(d["proofs"])
            p["registered"] = {port: x for port, x in p["registered"].items() if not port.startswith("a")}
        case "proof-no-b":
            for p in d["proofs"]:
                p["registered"] = {port: x for port, x in p["registered"].items() if not port.startswith("b")}
        case "proof-unregistered":
            p = rng.choice(d["proofs"])
            p["registered"]["c0"] = {"value": "h1/c/" + str(rng.randrange(100)), "root": _hex(rng)}
        case "entry-twice":
            v = _opened(rng, d, "b")
            d["entries"].append(dict(next(e for e in d["entries"] if e["value"] == v)))
        case "entry-form":
            v = _opened(rng, d, "a")
            e = next(e for e in d["entries"] if e["value"] == v)
            k, bad = rng.choice([("words", 0), ("words", None), ("words", True), ("words", Cr.MAX_ROWS + 1),
                                 (G.WIDTH_KEY, 16), (G.WIDTH_KEY, 8), (G.WIDTH_KEY, "32"), ("extra", 1)])
            e[k] = bad
        case "weights-period":
            e = rng.choice(d["weights"])
            signed = signs([{**x, "root": _hex(rng)} if x is e else x for x in d["weights"]])
        case "weights-unsigned":
            w = fold(d["weights"])
            signed = rng.choice([{"cap": 4}, {**signs(d["weights"]),
                                              Cr.WEIGHTS_FIELD: rng.choice([None, 5, w.upper(), w[:-2], [w]])}])
        case "weights-model":
            signed = signs(d["weights"])
            _unmodel(rng, signed, rng.choice(MODEL_NAMES))
        case "weights-no-period":
            signed = None
        case "weights-requests":
            d["rounds"][0]["requests"] = _hex(rng)
        case "weights-no-requests":
            d["rounds"][0]["requests"] = rng.choice([None, "zz", d["rounds"][0]["requests"].upper()])
            if rng.random() < 0.5:
                del d["rounds"][0]["requests"]
        case "weights-leaves":
            e = rng.choice([x for x in d["entries"] if Cr.is_held(x["value"])])
            e["leaves"] += rng.choice([-1, 1]) if e["leaves"] > 1 else 1
            e["domain"] = G.domain(hashlib.sha512(bytes.fromhex(d["descriptor"])).hexdigest(), e["value"], e["words"],
                                   e["leaves"]).domain_id.hex()
        case _ if credit in CREDIT_NAMES:
            _mutate_credit(d, s, credit, rng)
        case _:
            raise ValueError(name)
    if "ledger" not in d:
        if signed == {}:
            signed = signs(d["weights"])
        if signed is not None and committed != "absent":
            signed = {**signed, Cr.LENGTH_FIELD: dict(d["length"]) if committed == "same" else committed}
        events = rc.ledger_events(rng, d["window"], s, period=signed or {})
        if signed is None:
            events = [x for x in events if not x[0].startswith("period-")]
        d["ledger"] = rc.ledger_lines(events, rc._key(0))
    return {k: d[k] for k in WINDOW_KEYS}


#: each way a window's registered weights are not the ones its period's open signed and its open root binds
WINDOW_WEIGHTS_NAMES = ["weights-period", "weights-unsigned", "weights-model", "weights-no-period", "weights-requests",
                        "weights-no-requests", "weights-leaves"]
#: each way a window does not serve its period's fixed length: its period's open commits none (absent or null),
#: an EOS-keyed stop mode or a malformed one; a request ran past its public max_tokens (``length-past``: the open's
#: max_tokens of a request served two rounds or more is one less) or stopped before it (``length-early``: one more);
#: and more requests than the window's slots
LENGTH_NAMES = ["length-none", "length-eos", "length-form", "length-past", "length-early", "length-slots"]
#: each way a window's entries or its drawn units' proofs are not its registration's
REGISTRATION_NAMES = ["other-registration", "a-call-unbound", "proof-other-registration", "entry-root",
                      "proof-other-circuit", "proof-other-call", "proof-no-a", "proof-no-b", "proof-unregistered",
                      "entry-twice", "entry-form"]
WINDOW_NAMES = ["manifest-count", "chosen", "failed", "missing", "descriptor-hex", *DESC_NAMES,
                "credit-changed", "credit-no-draw", "credit-forged-ledger", "credit-branch-pin", "k-form", "k-past-units",
                "never-received", "round-coin", "a-untied", "no-a", *REGISTRATION_NAMES, *WINDOW_WEIGHTS_NAMES,
                *AROW_NAMES, *IDS0_NAMES, *REQ0_NAMES, "commit-unbound", "credit-commit-form", *ROWS_NAMES, "t-other",
                "t-form", *LIVE_NAMES, "j10r", *LENGTH_NAMES]
WINDOW_KEYS = ("key", "ledger", "pins", "main", *Cr.PROVED_KEYS, "k", "rows")
#: `credit.window`'s arguments, in order
WINDOW_ARGS = ("key", "ledger", "pins", "main", "window", "session", "salt", "calls", "served", "commit", "rounds",
               "draw", "descriptor", "entries", "proofs", "live", "k", "rows")


def window_args(d: dict) -> tuple:
    """`credit.window`'s arguments for case `d`, the challenger's signatures checked with `lms_verify`."""
    return (*(d[k] for k in WINDOW_ARGS), rc.lms_verify)


def _evidence(d: dict) -> dict:
    return {k: d[k] for k in Cr.PROVED_KEYS}


def period_instance(rng: random.Random, name: str = "honest", committed_too: dict | None = None, *,
                    period_pin: str | None = None, pin_floor: int | None = None,
                    stale_window: int | None = None) -> dict:
    """One `period` case: one to three honest windows in one period of the auditor's ledger, each proving the circuit
    whose SHA-512 the period's open committed (one window layout's descriptor, `descriptor`), presented in its close's
    order, and a window of no period opened after the close; then mutation `name`. The period's open commits
    `committed_too` too.  `period_pin`, `pin_floor` and `stale_window` override the period's open pin, its freshness floor
    and which window runs the oldest logged pin (MAIN[4]), for the test that pushes main past the floor."""
    n = rng.randint(2 if name in ("order", "other-sw", "zero-codes") else 1, 3)
    pname = rng.choice(rc.PERIODS)
    lay = layout(rng)
    t = table_rows(rng, lay)
    prefix = f"h1/w-{rng.randrange(100)}/"
    desc = (_break_desc(rng, lay, rng.choice(LAYOUT_DESC_NAMES)) if name == "bad-circuit" else
            _break_desc(rng, lay, "tile-k") if name == "tile-k" else descriptor(rng, lay, prefix))
    program = hashlib.sha512(bytes.fromhex(desc)).hexdigest()
    weights = weights_of(rng, lay, program, prefix)
    bad = rng.randrange(n) if name in ("never-received", "other-circuit", "other-sw", "zero-codes") else None
    odd, odd_prefix, odd_weights = desc, prefix, weights
    if name == "other-circuit":
        odd_prefix = f"other-{rng.randrange(1 << 20)}/"
        odd = descriptor(rng, lay, prefix=odd_prefix)
        odd_weights = _rehome(weights, hashlib.sha512(bytes.fromhex(odd)).hexdigest(), odd_prefix)
    elif name in ("other-sw", "zero-codes"):
        k, rows = rng.choice(sorted(b_counts(lay).items()))
        v, b = f"{prefix}b/{k}", b_rows(rng, k, rows)
        j = rng.randrange(rows)
        other = (b[:j] + [(b[j][0], b[j][1] * 2)] + b[j + 1:] if name == "other-sw"
                 else [(bytes(k), sw) for _codes, sw in b])
        weights = [registered_b(program, v, b) if e["value"] == v else e for e in weights]
        odd_weights = [registered_b(program, v, other) if e["value"] == v else e for e in weights]
    length = fixed_length(rng, t, min(round_choices(lay)))
    wins = [honest_window(rng, "never-received" if i == bad and name == "never-received" else "honest", sid=f"s-{i + 1}",
                          leaf=4 * i + 2, lay=lay, desc=odd if i == bad else desc,
                          prefix=odd_prefix if i == bad else prefix, weights=odd_weights if i == bad else weights, t=t,
                          length=length)
            for i in range(n)]
    foreign = honest_window(rng, sid="o-1", leaf=4 * n + 3, lay=lay, desc=desc, prefix=prefix, weights=weights, t=t,
                            length=length)
    pub = wins[0][1].pub
    descriptor_ = bytes.fromhex(desc)
    least = units_of(lay)
    big_k = rng.choice([1, least, rng.randint(1, least)])
    for d, _s in wins + [foreign]:
        d["proofs"] = _window_proofs(rng, d, _served_units(d, k=big_k))
    ends = [{} for _ in wins]
    if name == "incomplete":
        ends[rng.randrange(n)] = {"complete": False}
    committed = {"cap": 4, "circuit": program, Cr.K_FIELD: big_k, Cr.ROWS_FIELD: t, **signs(weights),
                 Cr.LENGTH_FIELD: {"max_tokens": length, "stop": Wd.STOP_FIXED}, **(committed_too or {})}
    match name:
        case "no-circuit":
            committed = {Cr.K_FIELD: big_k, Cr.ROWS_FIELD: t, **signs(weights), Cr.LENGTH_FIELD: committed[Cr.LENGTH_FIELD]}
        case "length-none":
            del committed[Cr.LENGTH_FIELD]
        case "no-weights":
            del committed[Cr.WEIGHTS_FIELD]
        case "weights-form":
            w = committed[Cr.WEIGHTS_FIELD]
            committed[Cr.WEIGHTS_FIELD] = rng.choice([None, 5, w.upper(), w[:-2], [w]])
        case _ if name in MODEL_NAMES:
            _unmodel(rng, committed, name)
        case "no-k":
            del committed[Cr.K_FIELD]
        case "no-rows":
            del committed[Cr.ROWS_FIELD]
        case "t-form":
            committed[Cr.ROWS_FIELD] = bad_rows(rng, t)
        case "t-other":
            committed[Cr.ROWS_FIELD] = other_rows(rng, lay)
        case "k-form":
            committed[Cr.K_FIELD] = _bad_k(rng)
        case "k-past-units":
            committed[Cr.K_FIELD] = past = min(units_of(d["layout"]) for d, _s in wins) + rng.randint(1, 5)
            for d, _s in wins:
                if units_of(d["layout"]) >= past:
                    d["proofs"] = _window_proofs(rng, d, _served_units(d, k=past))
    # the period's open pin P (`credit.committed_pin`) and freshness floor f (`credit.committed_floor`): a window counts
    # only if its pin is at most f first-parent commits behind P on main's line, counted from P and never from main's
    # head.  Honest P is MAIN[-1], older than every logged session (PINS pin MAIN[4], [2], [0]), and no floor is recorded
    # (default 0), so every honest session is at or after P; `stale-pin` records a nearer P (MAIN[2]) and floor 1 and pins
    # one window to MAIN[4], two first-parent commits before P and so past the floor.
    if period_pin is None:
        period_pin = rc.MAIN[2] if name == "stale-pin" else rc.MAIN[-1]
    if pin_floor is None and name == "stale-pin":
        pin_floor = 1
    open_body = {"period": pname, "lane": rc.LANE, "by": rc.LANE, **committed, Cr.PERIOD_PIN_FIELD: period_pin,
                 "key_sha512": hashlib.sha512(pub).hexdigest()}
    if pin_floor is not None:
        open_body[Cr.STALE_FIELD] = pin_floor
    events = [("period-open", None, open_body)]
    pins = list(rc.PINS)
    branch = rng.randrange(n) if name == "branch-pin" else None
    # the window pinned to MAIN[4], the oldest the pins log records: the per-window pin check passes (it is logged and on
    # main) but the period hygiene refuses it when MAIN[4] is past P + f (`credit._pin_hygiene`).
    stale = stale_window if stale_window is not None else (rng.randrange(n) if name == "stale-pin" else None)
    if branch is not None:
        pins.append(Ch.pin_line(rc.BRANCH, rc.MAIN[0], "2026-10-23T03:05:00Z"))
    for i, ((d, s), end) in enumerate(zip(wins, ends)):
        o = rc._open(rng, d["window"], pub)
        if i == branch:
            o.update(pinned=rc.BRANCH, pins=Ch.pins_seal(pins))
        if i == stale:
            o.update(pinned=rc.MAIN[4], pins=Ch.pins_seal(pins))
        events += [("open", s.sid, o), ("end", s.sid, rc._end(s, **end))]
    outcomes = [{"root": d["window"]["root"], "session": s.sid, "complete": end.get("complete", True)}
                for (d, s), end in zip(wins, ends)]
    events.append(("period-close", None, {"period": pname, "lane": rc.LANE, "by": rc.LANE, "windows": outcomes,
                                          "complete": all(w["complete"] for w in outcomes)}))
    fd, fs = foreign
    events += [("open", fs.sid, rc._open(rng, fd["window"], pub)), ("end", fs.sid, rc._end(fs))]
    leaves = [0] + [x for i in range(n) for x in (4 * i + 1, 4 * i + 4)] + [4 * n + 1, 4 * n + 2, 4 * n + 5]
    key = rc._key(1) if name == "forged-ledger" else rc._key(0)
    ledger = rc.ledger_lines(events if name != "no-close" else [e for e in events if e[0] != "period-close"], key,
                             leaves if name != "no-close" else [q for q, e in zip(leaves, events) if e[0] != "period-close"])
    windows = [_evidence(d) for d, _s in wins]
    out = {"key": wins[0][0]["key"], "ledger": ledger, "pins": pins, "main": list(rc.MAIN), "name": pname}
    i = rng.randrange(n)
    match name:
        case ("honest" | "incomplete" | "no-close" | "forged-ledger" | "no-circuit" | "no-k" | "k-form" | "k-past-units"
              | "no-rows" | "t-form" | "t-other" | "bad-circuit" | "never-received" | "other-circuit" | "branch-pin" | "no-weights" | "weights-form"
              | "other-sw" | "zero-codes" | "stale-pin" | "length-none"):
            pass
        case _ if name in MODEL_NAMES:
            pass
        case "tile-k":
            pass
        case "other-k":
            d = wins[i][0]
            n = units_of(d["layout"])
            mine = _served_units(d, k=big_k)
            other = rng.choice([k for k in (big_k - 1, big_k + 1, rng.randint(1, n)) if 1 <= k <= n and k != big_k
                                and _served_units(d, k=k) != mine] or [None])
            if other is None:
                windows[i]["proofs"] = windows[i]["proofs"][1:]
            else:
                windows[i]["proofs"] = _window_proofs(rng, d, _served_units(d, k=other))
        case "missing":
            del windows[i]
        case "order":
            i, j = rng.sample(range(n), 2)
            windows[i], windows[j] = windows[j], windows[i]
        case "extra":
            windows.append(_evidence(fd) if rng.random() < 0.5 else dict(windows[i]))
        case "foreign":
            windows.insert(rng.randrange(n), _evidence(fd))
        case "alt-layout":
            windows[i]["descriptor"] = descriptor(rng, _alt_layout(rng, lay))
        case "descriptor-form":
            h = windows[i]["descriptor"]
            windows[i]["descriptor"] = rng.choice([h.upper(), h[:-1], "", "zz", 5, None, list(descriptor_)])
        case "unopened":
            out["name"] = next(p for p in rc.PERIODS if p != pname)
        case "name":
            out["name"] = rng.choice([None, 5, [pname]])
        case "window-refused":
            windows[i]["proofs"] = _mutate_proofs(windows[i]["proofs"], rng, rng.choice(["failed", "missing"]), [])
        case "proof-other-registration":
            _other_root(rng, windows[i]["proofs"])
        case "live-refused":
            proofs = windows[i]["live"]["proofs"]
            if rng.random() < 0.5:
                del proofs[rng.randrange(len(proofs))]
            else:
                rng.choice(proofs)["verdict"] = "reject"
        case "windows-form":
            drop = rng.choice(["proofs", "descriptor", "live"])
            windows = rng.choice([None, {"0": windows[0]}, windows[:i] + [5] + windows[i + 1:],
                                  windows[:i] + [{k: v for k, v in windows[i].items() if k != drop}]
                                  + windows[i + 1:]])
        case "key-form":
            out["key"] = rng.choice([None, {**out["key"], "public": "00"}])
        case _:
            raise ValueError(name)
    return {**out, "windows": windows}


PERIOD_NAMES = ["no-close", "incomplete", "missing", "order", "extra", "foreign", "unopened", "name", "window-refused",
                "windows-form", "forged-ledger", "key-form", "other-circuit", "no-circuit", "descriptor-form", "no-k",
                "k-form", "k-past-units", "other-k", "no-rows", "t-form", "t-other", "alt-layout", "bad-circuit", "tile-k",
                "never-received",
                "proof-other-registration", "branch-pin", "no-weights", "weights-form", *MODEL_NAMES, "other-sw",
                "zero-codes", "live-refused", "stale-pin", "length-none"]
PERIOD_KEYS = ("key", "ledger", "pins", "main", "name", "windows")


def registered_file_instance(rng: random.Random, name: str = "honest") -> dict:
    """A `registered-file` case: a window's circuit (`descriptor` of a random layout, its parameters named a prefix or
    none), one of its calls, the window's open root, its rounds (one to three dividing its calls), and the roots the
    window registers (``{value: [root, ...]}``) for each value a statement of that call reads (`h1window.table`'s A
    table of its round and K, its z table and the weights' ``b/<K>``), some other rounds' and K's A tables among them,
    as `credit.registered_file` takes them, and at random (always under the ``receipt-`` mutations) the call's receipt
    table ``receipt@<call>``, its templates returning a receipt of whole 16-bit words (bits, of 1-bit leaves or wider); ``missing`` leaves one of the
    call's values out, ``twice`` registers one twice, ``receipt-twice`` registers the receipt table twice and
    ``receipt-width`` gives the templates a receipt of no whole 16-bit words, or none, ``pub-roots`` registers the
    call's A table or ``b/<K>``, whose roots its units' public rows hold, at a root that is no SHA-512 digest, and
    ``ok-registered`` registers the window's ok rows ``<prefix>ok``, which are the verifier's own. Its templates are
    J10-R's at random (always under ``j10r``), whose public rows the verifier derives each unit's call and blocks for,
    and ``j10r-form`` gives a J10-R window a template in no one form (`j10r_misformed`)."""
    lay = layout(rng)
    j10r = name == "j10r" or (name == "honest" and rng.random() < 0.5)
    prefix = rng.choice(["", f"h1/w-{rng.randrange(100)}/"])
    call = rng.randrange(len(lay["calls"]))
    steps = rng.choice([s for s in (1, 2, 3) if len(lay["calls"]) % s == 0])
    k_of = {lin: k for lin, k, _n in lay["linears"]}
    a = f"{prefix}a/{k_of[lay['calls'][call][0]]}@{Cr.step_row(lay, steps, call)[0]}"
    vals = [a, f"{prefix}zy@{call}", f"{prefix}b/{k_of[lay['calls'][call][0]]}"]
    receipted = name.startswith("receipt-") or rng.random() < 0.5
    bit = ["Value", 1]
    w = rng.randint(1, 8)
    receipt = rng.choice([["Array", 16 * w, bit], ["Array", w, ["Value", 16]], ["Value", 16 * w],
                          ["Tuple", [["Value", 16], ["Array", 2 * w, ["Value", 8]]]]]) if receipted else None
    if receipted:
        vals.append(f"{prefix}receipt@{call}")
    roots = {v: [_hex(rng)] for v in vals}
    for st, k, _cs in Cr.step_tables(lay, steps):
        if f"{prefix}a/{k}@{st}" != a and rng.random() < 0.5:
            roots[f"{prefix}a/{k}@{st}"] = [_hex(rng)]
    v = rng.choice(vals[:3])
    match name:
        case "honest" | "j10r":
            pass
        case "j10r-form":
            return {"descriptor": j10r_misformed(rng, lay, prefix), "call": call, "roots": roots, "window": _hex(rng),
                    "steps": steps, "salt": _hex(rng)}
        case "missing":
            if rng.random() < 0.5:
                del roots[v]
            else:
                roots[v] = []
        case "twice":
            roots[v].append(_hex(rng))
        case "receipt-twice":
            roots[vals[-1]].append(_hex(rng))
        case "pub-roots":
            roots[rng.choice(vals[0:3:2])] = [rng.choice(["", _hex(rng)[:-2], _hex(rng).upper(), _hex(rng) + "00"])]
        case "ok-registered":
            roots[f"{prefix}{Cr.OK_VALUE}"] = [_hex(rng)]
        case "receipt-width":
            receipt = rng.choice([None, bit, ["Array", 16 * rng.randint(0, 4) + rng.randint(1, 15), bit],
                                  ["Array", 2, ["Array", 4, bit]], ["Array", rng.randint(1, 3), ["Value", 5]],
                                  ["Value", 16 * rng.randint(0, 4) + rng.randint(1, 15)], ["Array", 3, ["Value", "16"]]])
        case _:
            raise ValueError(name)
    return {"descriptor": descriptor(rng, lay, prefix, receipt, j10r), "call": call, "roots": roots, "window": _hex(rng),
            "steps": steps, "salt": _hex(rng)}


REGISTERED_FILE_NAMES = ["missing", "twice", "receipt-twice", "receipt-width", "pub-roots", "ok-registered", "j10r",
                         "j10r-form"]


def _tree(rng: random.Random, count: int, dom: CommitmentDomain | None = None) -> tuple[str, list[tuple[bytes, ...]]]:
    """A frame-v3 SHA-512 tree over `count` leaf digests under `dom` (a random domain when None): its domain id and
    levels."""
    if dom is None:
        dom = CommitmentDomain(rng.randbytes(64), rng.randrange(-2, 4), RangeIndexedDomain(0, count), "sha512")
    return dom.domain_id.hex(), tree_levels(dom, [rng.randbytes(64) for _ in range(count)])


def _log(steps: int, values: dict, inners: list[str]) -> list[dict]:
    """The round log: each round's root over its inner root and its `cover` segments of every A value, a window's A
    value's equal blocks round by round and a round's A table whole in its round (`values`' third member)."""
    out = []
    for r in range(steps):
        segs = {}
        for v, (count, levels, rnd) in values.items():
            per = count // steps
            if rnd is None or rnd == r:
                lo, hi = (r * per, (r + 1) * per) if rnd is None else (0, count)
                segs[v] = [[lv, i, levels[lv][i].hex()] for lv, i in Cr.cover(lo, hi)]
        out.append({"inner": inners[r], "a": segs})
    return _reroot(out)


def _reroot(log: list[dict]) -> list[dict]:
    for r, x in enumerate(log):
        x["root"] = Cr.round_root(r, bytes.fromhex(x["inner"]), {v: [(lv, i, bytes.fromhex(h)) for lv, i, h in s]
                                                                  for v, s in x["a"].items()}).hex()
    return log


def a_link_instance(rng: random.Random, name: str = "honest") -> dict:
    """One `a_link` case: a window's registered entries and its round log, then mutation `name`. Its A values are each
    K's ``…/a/<K>`` and, at random (always under the ``table-`` mutations), one or two rounds' A tables
    ``…/a/<K>@<s>``, each committed whole by its round; and at random (always under the ``ids-`` mutations) some
    rounds' input-ids tables ``…/ids@<s>``, each committed whole by its round; and at random (always under the
    ``slots-`` mutations) some rounds' slot tables ``…/slots@<s>``, alike; and at random (always under the
    ``receipt-`` mutations) some calls' receipt tables ``…/receipt@<c>``, each committed whole by its call's round, of
    the window's ``calls`` (the rounds' equal blocks). At random (always under ``layout-rows``) its ``rows`` give each
    A value's and receipt table's leaf count as a layout would (`credit.table_rows`); ``leaves-cap`` gives an entry no
    ``rows`` name a count past `credit.MAX_ROWS`, and ``layout-rows`` one that ``rows`` names another count than its
    own, at random 2^40, which the verifier refuses before it hashes any pad."""
    calling, tying, receipts = name.startswith("table-"), name.startswith("ids-"), name.startswith("receipt-")
    slotting = name.startswith("slots-")
    steps = rng.choice([2, 3, 4] if name in ("blocks", "swapped", "table-other-round", "ids-other-round",
                                             "slots-other-round", "receipt-other-round") else [1, 2, 3, 4])
    calls = steps * rng.randint(1, 3)
    window = f"h1/w-{rng.randrange(100)}"
    values, entries = {}, []

    def a(v: str, count: int, rnd: int | None) -> None:
        domain, levels = _tree(rng, count)
        values[v] = (count, levels, rnd)
        entries.append({"value": v, "root": levels[-1][0].hex(), "leaves": count, "words": 96, "domain": domain})

    sizes = [2, 3, 4, 5] if name == "swapped" else [1, 2, 3, 4, 5, 8]
    ks = rng.sample([64, 4096, 6144, 14336], rng.randint(1, 2))
    for k in ks:
        a(f"{window}/a/{k}", steps * rng.choice(sizes), None)
    if calling or rng.random() < 0.5:
        for st, k in rng.sample([(st, k) for st in range(steps) for k in ks], rng.randint(1, min(2, steps * len(ks)))):
            a(f"{window}/a/{k}@{st}", rng.choice(sizes), st)
    if tying or rng.random() < 0.5:
        for s in rng.sample(range(steps), rng.randint(1, steps)):
            a(f"{window}/ids@{s}", rng.choice(sizes), s)
    if slotting or rng.random() < 0.5:
        for s in rng.sample(range(steps), rng.randint(1, steps)):
            a(f"{window}/slots@{s}", rng.choice(sizes), s)
    if receipts or rng.random() < 0.5:
        for c in rng.sample(range(calls), rng.randint(1, calls)):
            a(f"{window}/receipt@{c}", rng.choice(sizes), c // (calls // steps))
    entries += [{"value": f"{window}/z/{rng.choice([64, 4096])}", "root": _hex(rng), "leaves": 3, "domain": _hex(rng)},
                {"value": f"{window}/a/x", "root": _hex(rng)}][:rng.randint(0, 2)]
    rng.shuffle(entries)
    log = _log(steps, values, [_hex(rng) for _ in range(steps)])
    tied = [e for e in entries if Cr.is_tied(e["value"])]
    sized = [x for x in tied if Cr.ids_step(x["value"]) is None and Cr.slots_step(x["value"]) is None]
    rows = ({x["value"]: x["leaves"] for x in sized}
            if name == "layout-rows" or (name not in ("leaves-cap", "blocks") and rng.random() < 0.5) else None)
    e = rng.choice(sized if name == "layout-rows" else [x for x in tied if Cr.ids_step(x["value"]) is not None] if tying
                   else [x for x in tied if Cr.slots_step(x["value"]) is not None] if slotting
                   else [x for x in tied if Cr.receipt_call(x["value"]) is not None] if receipts
                   else [x for x in tied if Cr.is_a(x["value"]) and (values[x["value"]][2] is not None) == calling]
                   if calling or name == "blocks" else tied)
    v = e["value"]
    own = values[v][2]
    r = rng.randrange(steps) if own is None else own
    segs = log[r]["a"][v]
    match name:
        case "honest":
            pass
        case "untied":
            e["root"] = _tree(rng, e["leaves"])[1][-1][0].hex()
        case "late-rows":
            domain, other = _tree(rng, values[v][0])
            e["root"], e["domain"] = other[-1][0].hex(), domain
        case "rewritten":
            segs[rng.randrange(len(segs))][2] = _hex(rng)
            _reroot(log)
        case "round-root":
            segs[rng.randrange(len(segs))][2] = _hex(rng)
        case "gap":
            del segs[rng.randrange(len(segs))]
            _reroot(log)
        case "extra":
            segs.append([0, values[v][0], _hex(rng)])
            _reroot(log)
        case "swapped":
            if len(segs) > 1:
                segs[0], segs[1] = segs[1], segs[0]
            else:
                segs[0] = [segs[0][0], segs[0][1] + 1, segs[0][2]]
            _reroot(log)
        case "misaligned":
            lv, i, h = segs[0]
            segs[0] = [lv + 1, i // 2, h] if lv + 1 < Cr.LEVELS else [lv, i + 1, h]
            _reroot(log)
        case "unknown":
            log[r]["a"][f"{window}/a/{rng.choice([1, 2, 3])}"] = [[0, 0, _hex(rng)]]
            _reroot(log)
        case "no-a":
            entries = [x for x in entries if not Cr.is_a(x["value"])]
        case "twice":
            entries.append(dict(e))
        case "entry-form":
            k, bad = rng.choice([("root", None), ("root", "zz"), ("domain", None), ("domain", e["domain"].upper()),
                                 ("leaves", -1), ("leaves", True), ("leaves", None), ("leaves", "4")])
            e[k] = bad
        case "leaves-cap":
            e["leaves"] = rng.choice([0, Cr.MAX_ROWS + 1, 1 << 40])
        case "layout-rows":
            e["leaves"] = rng.choice([e["leaves"] + 1, e["leaves"] - 1 or 2, 1 << 40])
        case "blocks":
            e["leaves"] += 1 if (e["leaves"] + 1) % steps else 2 if (e["leaves"] + 2) % steps else 3
        case "log-form":
            log = rng.choice([[], None, "x"])
        case "round-form":
            log[r] = rng.choice([5, {**log[r], "inner": "zz"}, {k: x for k, x in log[r].items() if k != "root"}])
        case "segments-form":
            bad = rng.choice([None, [[Cr.LEVELS, 0, _hex(rng)]], [[0, -1, _hex(rng)]], [[0, 0, "zz"]], [[0, 0]],
                              [[True, 0, _hex(rng)]]])
            log[r]["a"] = rng.choice([{**log[r]["a"], v: bad}, [segs]])
        case "entries-form":
            entries = rng.choice([None, entries + [5], {"0": entries}])
        case "table-other-round":
            log[rng.choice([x for x in range(steps) if x != own])]["a"][v] = [list(x) for x in segs]
            _reroot(log)
        case "table-past":
            k = Cr.a_value(v)[0]
            e["value"] = f"{window}/a/{k}@{steps + rng.randrange(3)}"
            log[own]["a"][e["value"]] = log[own]["a"].pop(v)
            _reroot(log)
        case "ids-other-round":
            log[rng.choice([x for x in range(steps) if x != own])]["a"][v] = [list(x) for x in segs]
            _reroot(log)
        case "ids-past":
            e["value"] = f"{window}/ids@{steps + rng.randrange(3)}"
            log[own]["a"][e["value"]] = log[own]["a"].pop(v)
            _reroot(log)
        case "slots-other-round":
            log[rng.choice([x for x in range(steps) if x != own])]["a"][v] = [list(x) for x in segs]
            _reroot(log)
        case "slots-past":
            e["value"] = f"{window}/slots@{steps + rng.randrange(3)}"
            log[own]["a"][e["value"]] = log[own]["a"].pop(v)
            _reroot(log)
        case "receipt-other-round":
            log[rng.choice([x for x in range(steps) if x != own])]["a"][v] = [list(x) for x in segs]
            _reroot(log)
        case "receipt-past":
            e["value"] = f"{window}/receipt@{calls + rng.randrange(3)}"
            log[own]["a"][e["value"]] = log[own]["a"].pop(v)
            _reroot(log)
        case "receipt-calls":
            calls = rng.choice([0, calls + rng.randrange(1, steps)] if steps > 1 else [0])
        case "receipt-unknown":
            free = [c for c in range(calls) if f"{window}/receipt@{c}" not in values]
            log[own]["a"][rng.choice([f"{window}/zy@{Cr.receipt_call(v)}", f"{window}/receipt@0{Cr.receipt_call(v)}",
                                      f"{window}/receipts@{Cr.receipt_call(v)}",
                                      *(f"{window}/receipt@{c}" for c in free)])] = [[0, 0, _hex(rng)]]
            _reroot(log)
        case _:
            raise ValueError(name)
    return {"rounds": log, "entries": entries, "calls": calls, "rows": rows}


A_LINK_NAMES = ["untied", "late-rows", "rewritten", "round-root", "gap", "extra", "swapped", "misaligned", "unknown",
                "no-a", "twice", "entry-form", "leaves-cap", "layout-rows", "blocks", "log-form", "round-form",
                "segments-form", "entries-form", "table-other-round", "table-past", "ids-other-round", "ids-past",
                "slots-other-round", "slots-past", "receipt-other-round", "receipt-past",
                "receipt-calls", "receipt-unknown"]


def _b_entry(program: str, v: str, k: int, n: int, rng: random.Random) -> dict:
    """K's ``b/<K>`` of `n` rows: `weights.chunks` (K) leaves a row of `weights.leaf_words` (K) words."""
    x, w = Wt.chunks(k), Wt.leaf_words(k)
    return {"value": v, "root": _hex(rng), "leaves": x * n, "words": w,
            "domain": G.domain(program, v, w, x * n).domain_id.hex()}


def weights_instance(rng: random.Random, name: str = "honest") -> dict:
    """One `weights` case: a window's registered weights under its open root, then mutation `name`."""
    lay = layout(rng)
    program, prefix = _hex(rng), rng.choice(["", "h1/w-3/"])
    rows: dict[int, int] = {}
    for _name, k, n in lay["linears"]:
        rows[k] = rows.get(k, 0) + n
    entries = [_b_entry(program, f"{prefix}b/{k}", k, n, rng) for k, n in sorted(rows.items())]
    b = list(entries)
    lone = ["a/32@0", "zy@0", "b/x", f"b/0{min(rows)}", "b/" + "0" * 5000 + "1"]
    entries += [{"value": f"{prefix}{v}", "root": _hex(rng)}
                for v in [*(f"a/{k}" for k in rows), *lone][:rng.randint(0, len(rows) + len(lone))]]
    rng.shuffle(entries)
    requests, session = _hex(rng), rng.choice(["s-1", "a.b_c:d-9", "7"])
    roots = {k: bytes.fromhex(e["root"]) for k, e in zip(sorted(rows), b)}
    window = {"root": PS.window_open_root(Wt.weights_root(roots), bytes.fromhex(requests), session).hex()}
    e = rng.choice(b)
    k = next(kk for kk in rows if e["value"] == f"{prefix}b/{kk}")
    match name:
        case "honest":
            pass
        case "other-root":
            e["root"] = _hex(rng)
        case "other-requests":
            requests = _hex(rng)
        case "other-session":
            session = session + "x"
        case "other-window":
            window = {"root": _hex(rng)}
        case "missing":
            entries.remove(e)
        case "twice":
            entries.append({**e, "root": _hex(rng)})
        case "foreign":
            entries.append({"value": f"{prefix}b/{rng.choice([1, 2, 31, 128])}", "root": _hex(rng)})
        case "other-prefix":
            entries.append({**e, "value": rng.choice(["x/", "h1/w-4/", "h1/w-3/h1/w-3/"]) + e["value"]})
        case "other-leaves":
            e["leaves"] = rng.choice([e["leaves"] + 1, e["leaves"] - 1, str(e["leaves"]), True, None])
        case "other-words":
            e["words"] = rng.choice([e["words"] + 1, k // 2, k, str(e["words"])])
        case "other-domain":
            e["domain"] = rng.choice([_hex(rng)[:64], G.domain(_hex(rng), e["value"], e["words"], e["leaves"]).domain_id.hex(),
                                      G.domain(program, e["value"], e["words"], e["leaves"], window=_hex(rng)).domain_id.hex(),
                                      G.domain(program, e["value"], e["words"], e["leaves"] + 1).domain_id.hex(), None])
        case "entry-form":
            got = rng.choice(["extra", "drop", "bits"])
            if got == "extra":
                e["row_schema"] = "x"
            elif got == "bits":
                e["word_bits"] = 16
            else:
                del e[rng.choice(["leaves", "words", "domain"])]
        case "root-form":
            e["root"] = rng.choice([None, "zz", e["root"].upper(), 5])
        case "window-form":
            window = rng.choice([None, {}, {"root": "zz"}, [window]])
        case "requests-form":
            requests = rng.choice([None, "zz", requests.upper(), 5])
        case "session-form":
            session = rng.choice([None, "", "-s", "s/1", "x" * 65, 5, "\u015b"])
        case "circuit-form":
            if rng.random() < 0.5:
                program = rng.choice([None, "zz", program.upper(), program[:64], 5])
            else:
                prefix = rng.choice([None, 5, "a\"b/", "a\\b/", "\u015b/", "tab\t/"])
        case "layout-form":
            lay = rng.choice([None, {**lay, "m": 0}, {k: v for k, v in lay.items() if k != "tile"}])
        case "entries-form":
            entries = rng.choice([None, entries + [5], {"0": entries}])
        case _:
            raise ValueError(name)
    return {"window": window, "layout": lay, "entries": entries, "requests": requests, "session": session,
            "program": program, "prefix": prefix}


WEIGHTS_NAMES = ["other-root", "other-requests", "other-session", "other-window", "missing", "twice", "foreign",
                 "other-prefix", "other-leaves", "other-words", "other-domain", "entry-form", "root-form", "window-form",
                 "requests-form", "session-form", "circuit-form", "layout-form", "entries-form"]
WEIGHTS_KEYS = ("window", "layout", "entries", "requests", "session", "program", "prefix")
