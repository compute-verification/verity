"""The docket (Lean `Spot.Docket.check`): what a window is committed to, fixed before the verifier samples.

    {"format": "verity/registration/v1", "protocol": "verity/sampled-proofs/one-stage/v1",
     "program": "<SHA-512 of the program>", "query": "<SHA-512 of the query>", "partition": "<partition digest>",
     "population": <proof blocks the query derives>, "law": {"law": "subset", "k": K} | {"law": "stratified" | "work", "k": K, "strata": [...]},
     "scheme": "<leaf and tree scheme>",
     "roots": [{"port": ..., "root": ..., "leaves": ..., "domain": ...}, ...],
     "leaf_layer": "<SHA-512 of the verifier's public file>" | null,
     "anchors": [], "window": {"kind": "served" | "stand-in", ...provenance}, "previous": null | "<digest>"}

* **Digest.** `SHA-512("verity/registration/v1\\0" || canon(record))`.
* **Rule.** Everything a sampled proof block's statement can depend on is fixed by the record, or by objects the verifier already
  holds, before the sample. The verifier checks R1-R8 (`check`) against its own copies, then, where they pass, the
  caller's check (`caller.check`: Declared day, Binding, Seed salt, Registered compiler, and the pool's marks and chain,
  which `check` reads only to parse), writes a `receipt` with its own clock, and only then samples.
* **The checks** (`check`): the record has exactly these keys, and
  - R1: the format and protocol are these, the sampling design is the verifier's own expanded over its program and query
    (`draw.derive`, with its own work table for `work`) as canonical bytes, and `window.kind` is `served` or `stand-in`;
  - R2: the program, query and partition digests and the population are the verifier's own (an explicit partition
    object must be over the verifier's program);
  - R3: the scheme is the verifier's, and the roots are exactly the ports its layout derives, each exactly
    `{port, root, leaves, domain}` with a hex root and the verifier's leaf count and domain;
  - R4: the leaf layer is the digest of the verifier's copy, so a null one is refused when the verifier holds one;
  - R5: the anchors are the verifier's;
  - R6: `previous` is the digest of this prover's last record (null for the first), and no earlier record has the same
    window;
  - R7 refuses a registered value, a supplied one included (a table opening's root and domain), whose parameter's
    gates hold a run Input node (`run_input_named`, `R7-run-input`): a run Input's rows are committed once, in its one
    tree, which the members' check holds (`members.run_input_roots_at`), and are never a registered value; and a
    supplied anchor (`registered.root/v1`, `registered.domain/v1`) of a value V that neither the verifier's pinned
    roots nor the weights' registration of record pins (`anchor_unpinned`, `R7-anchor-unpinned`);
  - R8, in the variant with no members (no ``members``), under a template query: each run Input node's leaves, other
    than a parameter's the verifier expects registered or a tied one's, are read by one proof block alone
    (`partition.shared_inputs`, `R8-shared-input`); the per-member variant holds each reader to the node's one tree
    instead;
  - R8, under a template query: the verdict publishes the query's fan-out F on the verifier's program (`partition.fanout`, `Verdict.fanout`), the N at which the capped two-stage form charges a
    replay block (`plancheck.two`'s `fanout`). A run Input a parameter the verifier names `tied` holds is let through:
    the network period check ties its rows to the dockets that computed them, which this check doesn't, so the
    verdict publishes each tied name it let through (`Verdict.rests_on_period_check`), and with any its acceptance is
    none on its own; it refuses a tied name that isn't a run Input of the program's own, or that it expects registered
    or holds (`R8-tied`).
* **Receipt**: `{"format": "verity/registration-receipt/v1", "registration": <digest>, "verifier", "received": "<UTC,
  %Y-%m-%dT%H:%M:%SZ>"}`, and the sample names it.
* **The public file** (`matches_public`): the verifier holds its own (scheme, roots, their domains, its instance count)
  to the record. That they are the proved rows' is `verity-verify-proof`'s check of the public file (`checkPublic`), stated by
  the Lean guarantee `FailClosed.flock_verify_public_ties`; `verity-verify-proof` reads no `scheme` from the header, only META's.
* **`window`** is provenance, apart from the binding and the seed salt it names (Binding, Seed salt). `kind` says whether the roots are what the integration
  committed as it ran (`served`) or a fresh commitment to values captured from its run (`stand-in`, whose link to the run's
  root is provenance, not proof).
* **Window domains** (`window_domain`, `verity.commitments.window`'s): each port's frame domain binds the program, the
  partition, the port, its schema, its leaf count and the run, so the roots can't be read as anything else.
* **Registered values** (`verity/registration/v2`): the record adds `"registered": [{"value", "root", "leaves", "words",
  "domain"}, ...]` (with `"word_bits"` for a value whose leaves are not 16 bits), the program's weights, constants and
  tables committed once under hiding leaves (`registered`), and R7 holds them to the verifier's own list and to the roots
  the log's earlier records registered under the same domain, or to a root the verifier held before the record
  (`pinned`: weights recorded before the salt). A record registers values exactly when it is v2; v1's bytes are
  unchanged.
* **Per-window values** (`window_root`, `held`): a window-independent program registers its window's values under the
  same names every window. The verifier takes the window's open root R from the challenger's signed open of the
  window, never from the record, and every registered value's domain binds R but the ones it holds across windows
  (`held`, the weights, and its pinned, supplied and output values), so a window's values are held to no earlier
  window's roots, and a root registered in another window opens under no domain R7 takes. A verifier that names no R
  takes every domain as before, which is sound only while each window's values take fresh names: it refuses its own
  inputs (ValueError, naming the fix) when a value it doesn't hold across windows was registered by an earlier record
  under the domain it takes now (`unwindowed`), a window-independent program's per-window value from its second
  window on.
* **Supplied values**: a registered value the verifier fills itself (`supplied`, `registered.supplied_root`: rows of the
  record's roots, positions after the log's, its coin bound to its binding and the record's window and its own bytes,
  under public salts). R7 recomputes its root and refuses a record that names another (`R7-supplied`); it is the
  record's own, so no earlier record holds it. The window is otherwise provenance: R6 takes one record a window, so a
  salt bound to it is one salt a window.
* **Declared day** (`day`, `declarations`): the prover declares one circuit a day, before the day's first coin, in the
  registration log (`verity/registration/declaration/v1`,
  `verity.protocols.registration.verifier.registration`). The verifier passes
  the window's day and the declarations it holds when it samples (its view's `declarations`: `{"day", "program",
  "blocks"}` each): of record, the UTC day of the window open's `started_utc` and the declarations its lane signed
  before that open, both from the challenger's ledger checked under the auditor's key (`challenger.declared_day`), so a
  window that opens before UTC midnight and closes after is a window of its open's day. The caller's check (`caller.check`) refuses a day with no
  declaration (`R2-undeclared`), a day declared twice (`R2-declared-twice`, which the log's own check refuses as
  `duplicate-declaration`), and a record whose program is not the day's declared one (`R2-declared`), as its Lean does
  (`VStar.Recursive.Caller.declaredCode`).
* **Binding** (`binding`: `{"stream", "prover"}`, the verifier's stream digest and the identity it registered the prover
  under): the caller's check holds the record's window to name both (`R6-stream`, `R6-prover`), and a window salt is bound to them,
  so a tile seeded for one prover's stream is refused in any other's, in the same window or not.
* **Seed salt** (`seed_salt`): the window's seed salt, the one coin of the challenger's seed-salt family released after
  the window's open root, before the window serves (`coins.draw_seed_salt`), which the verifier reads from the
  challenger's log (`coins.seed_salt_of`), never from the record: lowercase hex of 64 bytes, read on the window's
  session, so only beside `window_root`. The record's window names it (`SEED_SALT`), a certified input the window's
  circuits read, and the caller's check refuses a window that names another or none, or names one where the verifier read none
  (`R6-seed-salt`), so a salt the server chose is refused by name.
* **Registered compiler** (`compiler`): the registration log's registered compiler as its view publishes it
  (`verity.protocols.registration.verifier.registration`): its item, with its digest and the public length ℓ of the advice it reads, or None
  when the log registers none. The caller's check holds the record's window to name that compiler by its digest
  (`window.compiler.digest`; `R6-compiler`) and advice of at most ℓ bits (`window.advice.bits`; `R6-advice-length`),
  and, where the log registers none, to name neither a compiler nor advice (`R6-compiler`), so advice no registered
  compiler reads is refused by name. A verifier that reads no log's compiler passes `UNREAD`, the default, and the
  caller's check holds the window to nothing here.
* **Output values**: a registered value that is an output of an earlier run (`outputs`: `{value: {"record": D, "port":
  P}}`; never the inference-only check's θ, which holds registered public data only) reuses that output's commitment instead of
  committing it afresh: its entry takes port P's domain and root from the `roots` of the log's record with digest D, and
  P's leaf count must be the value's rows (`R7-output` when the log holds no such record or port, or the counts differ).
  Its rows stay under the earlier run's hiding leaves, so nothing about the value is opened, and the reads open against
  that root under that domain (`registered.reads_file`'s `domains`).
* **Reused values**: an output value may name the previous record's registered value instead of a port (`{value:
  {"record": D, "value": V}}`): its entry takes the domain and root of V's entry in the `registered` of the log's record
  with digest D (its last naming V), whose leaves, words and word bits must be the value's own (`R7-output` when D is
  not the log's last record, `last_digest`, which R6 holds `previous` to, when the log holds no such value, or when they
  differ). D is the previous record's so that the record whose own check held V's domain is the one the verifier's
  caller ties this record to (Pearl+'s window record W, chained right before it). Nothing is registered afresh: a
  record whose entry names another domain or root is `R7-domain` or `R7-held`, as for a port.
* **Window history**: an output value the verifier gives only a port (`{value: {"port": P}}`) is port P of the previous
  record, the log's last, which R6 holds `previous` to (`last_digest`): the verifier names the record itself, so a
  window's history, the earlier windows' outputs it reads, is held to the previous window's root and domain for P, and
  with no previous record it is `R7-output`.
* **Pool chain** (`pool`: `{"pin", "pout", "genesis", "lag", "reset", "verdicts", "week"}`): a program whose window
  opens a pool from a registered value `pin` and leaves it in a registered value `pout` (PoolWindow's rows, each the
  pool's root). A window carries the previous window's pool (`pool_chains`) unless a public rule or serving restarts
  it: `pin` is then a reused value, the previous record's `pout` in `pout`'s row format (`pool_values`), so its entry
  takes that entry's domain and root, whatever the verifier made of that window. The chain restarts from the empty
  pool, `pin` a supplied value filled from `genesis`, the empty pool's root, so its root is fixed:
  - by the public rule, which takes no advice: at a docket's first window, at a docket boundary (the log's
    last record registered another program), and after the verifier refused the window `lag` windows of record before
    this one (`pool_refused`), `lag` the fixed public lag at which serving learns a verdict. Whether it did is read
    from the verifier's own verdict log (`verdicts`: each earlier record's digest and whether its run accepted), so
    where every run accepted it never restarts the chain;
  - by serving's own reset (`reset`), which an engine crash needs, at most one a week: the record's window names it
    (`POOL_RESET`, a bool; `R6-pool-reset` where it names another or none). Each window also names its week
    (`POOL_WEEK`, its index among the monitoring period's weeks on the verifier's clock, `week`; `R6-pool-week`) and
    whether the week's reset is spent at it (`POOL_SPENT`; `R6-pool-spent`): that it resets, or that the previous
    record, of the same week, named it spent (`pool_spent_before`), so a week's windows from its reset on all name it
    spent, and the next week starts unspent. The caller's check refuses a reset where the week's is spent before it
    (`R6-pool-reset-week`). Serving chooses the resets, so they are advice: the records name, of a week's n windows,
    the one serving reset at or none, log2(n + 1) bits a week (log2 1009, 9.98 bits, for 600-second windows), which
    leakage charges. The rest of the chain, the spent marks and the weeks included, is a function of the resets, the
    verdicts and the verifier's clock, and public.
  The caller's check refuses a v2 record whose entry for `pin` is not the pool its window opens (`pool_open`) by name
  (`R7-pool-chain`),
  beside the R7 codes the reuse or the fill gives it. The pool's root is all the chain carries. What a window takes up
  that the one before left open at its edge is part of its certified input, which the verifier pins at the root the
  network attester signed for the window, so a root the network attester didn't sign is `R7-held`, whether the window
  carries the pool or restarts it.
* **Verdicts** (`verdicts`, the pool's form): a verifier may give its verdict log, and the caller's check
  (`caller.check`) then refuses a record with an output value that reads a port of an earlier record the log names
  refused (`R7-output-refused`, after `R7-pool-chain`), so a window carries no start past a window whose run the
  verifier refused. A record the log doesn't name passes, as the pool's lag rule reads it, and a reused registered
  value (`value`, the pool's `pin` among them) is not held to it.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any, Iterable, Mapping, Sequence

from verity.commitments import docket_digest as DD
from verity.commitments import domains as DM
from verity.commitments import history as HT
from verity.commitments.window import WINDOW_DOMAIN, window_domain
from verity.circuits import partition_object as PO
from verity.circuits.codec import canonical_json, descriptor_chunks
from verity.circuits.program import REGISTRATION, Program
from verity.circuits.supplied import SUPPLIED, registered_values

from . import draw as D, link_points as LP, partition as P, registered as G
from verity.commitments import registered_value as RV
from .verdict import Verdict, _Collect

RECEIPT = DM.REGISTRATION_RECEIPT
#: the window's field naming its seed salt (Seed salt)
SEED_SALT = "seed_salt"
#: the window's field naming the reset serving chose for its pool (Pool chain)
POOL_RESET = "pool_reset"
#: the window's field naming its week on the verifier's clock (Pool chain)
POOL_WEEK = "pool_week"
#: the window's field naming whether its week's one reset is spent at it (Pool chain)
POOL_SPENT = "pool_spent"
#: `compiler` when the verifier reads no registration log's compiler (Registered compiler): the caller's check holds the
#: window to nothing
UNREAD: Any = object()
TIED_WHY = ("a tied value is a run Input of the program whose rows a later check ties to the registrations that computed "
            "them: a parameter of the program, neither registered nor held, whose gates are one interval of run Input nodes")
SHARED_WHY = ("a run Input node whose leaves two proof blocks read, or a batch broadcasts, could hold rows one of them "
              "computed, which the fan-out can't see, and the variant with no members holds no one tree of it: give each "
              "proof block its own leaves, or bind the value at registration")
RUN_INPUT_WHY = ("a run Input's rows are committed once, in its one tree, before any coin: a registered value, a supplied "
                 "one included, is no run Input node")
#: the supplied functions whose value is a registered value's root or domain (`anchor_unpinned`)
ANCHOR_FUNCTIONS = ("registered.root/v1", "registered.domain/v1")
PIN_WHY = ("a value the weights' registration of record pins is registered in the record at that pin, in root, leaf count, "
           "words and domain, and a supplied anchor's value, which a supplied function reads from the record's entry, "
           "is registered there")
SUPPLIED_SOURCE_WHY = ("a supplied input's rows are the verifier's own: the parameters' supplied names its source, C's "
                       "mark its function, or it is an earlier record's output; with none, nothing fixes its root")
ANCHOR_UNPINNED_WHY = ("a supplied anchor reads a registered value's root or domain, and the verifier holds that value "
                       "to no pin: neither its own pinned roots nor a pin of the weights' registration of record")
#: R8's refusals of the program's registered inputs against the weights' registration of record (`weights_held`)
WEIGHTS_WHY = {
    "R8-weights-digest": "the weights' registration is not the one of record: its SHA-512 is not the digest the "
                         "auditor's side holds for it",
    "R8-weights-form": "the weights' registration is not {program, values: {value: {value, root, leaves, words, domain}}}, "
                       "one object or a list of them, with no other key",
    "R8-weights-domain": "a pin names another value, a root that is not 64 bytes of hex, or a domain other than the one "
                         "its own program, value and shape make",
    "R8-weights-shape": "a registered input's leaves, words or width are not its pin's",
    "R8-unpinned": "a registered input of the program that the weights' registration of record does not pin and whose "
                   "root the verifier does not hold itself, whatever it expects by name: nothing fixes its root before "
                   "the coin",
    "R8-weights-query": "the program has a registered input, or the verifier holds a weights' registration of record, "
                        "under a query that is no template query that applies: R8 can't read which proof blocks read "
                        "the input",
}
_KEYS = {"format", "protocol", "program", "query", "partition", "population", "law", "scheme", "roots", "leaf_layer",
         "anchors", "window", "previous"}
_ROOT_KEYS = {"port", "root", "leaves", "domain"}
_HEX = re.compile(r"[0-9a-f]+")


def query_digest(query: Mapping[str, Any]) -> str:
    return hashlib.sha512(canonical_json(dict(query))).hexdigest()


def _of_record(of_record: tuple[bytes, str]) -> tuple[list[tuple[str, Any]], dict[str, dict[str, Any]]]:
    """The weights' registration of record's pins, by value, after its digest, its form and each pin's domain, which
    the pin's own program, value and shape make (`registered_value.domain`); the refusals and the pins that pass."""
    body, digest = of_record
    if not (isinstance(body, bytes) and isinstance(digest, str) and hashlib.sha512(body).hexdigest() == digest):
        got = hashlib.sha512(body).hexdigest() if isinstance(body, bytes) else None
        return [("R8-weights-digest", {"registration": got, "of_record": digest})], {}
    try:
        doc = json.loads(body.decode("utf-8"))
        docs = doc if isinstance(doc, list) else [doc]
        if not all(isinstance(d, dict) and set(d) == {"program", "values"}
                   and isinstance(d["program"], str) and isinstance(d["values"], dict) and all(
                isinstance(e, dict) and set(RV.ENTRY_KEYS) <= set(e) <= {*RV.ENTRY_KEYS, RV.WIDTH_KEY}
                and all(type(e[k]) is int and e[k] >= 0 for k in ("leaves", "words") + ((RV.WIDTH_KEY,) if RV.WIDTH_KEY in e else ()))
                for e in d["values"].values()) for d in docs):
            raise TypeError("an entry")
        if len(names := [v for d in docs for v in d["values"]]) != len(set(names)):
            raise ValueError("a value pinned twice")
    except (ValueError, KeyError, TypeError) as e:
        return [("R8-weights-form", f"{type(e).__name__}: {e}")], {}
    bad, pins = [], {}
    for program, v, e in sorted((d["program"], v, e) for d in docs for v, e in d["values"].items()):
        try:
            want = RV.domain(program, v, e["words"], e["leaves"], e.get(RV.WIDTH_KEY, RV.WORD_BITS)).domain_id.hex()
        except ValueError:
            want = None
        if e["value"] != v or e["domain"] != want or not (isinstance(e["root"], str) and len(e["root"]) == 128
                                                          and _HEX.fullmatch(e["root"])):
            bad.append(v)
        else:
            pins[v] = e
    return ([("R8-weights-domain", {"values": bad})] if bad else []), pins


def marked_inputs(program: Program) -> list[str]:
    """The program's registered inputs: the parameters it binds at registration that hold gates."""
    bindings = getattr(program, "bindings", ())
    return [n for (n, _), b in zip(program.fn.signature()[0], bindings)
            if b == REGISTRATION and G.held_gates(program, [n])] if bindings else []


def weights_held(program: Program, query: Mapping[str, Any], registered: Iterable[str],
                 of_record: tuple[bytes, str] | None, held: Iterable[str] = (), linked: Iterable[str] = ()
                 ) -> tuple[list[tuple[str, Any]], dict[str, dict[str, Any]]]:
    """R8's hold on the program's registered inputs (the parameters it binds at registration): each must be pinned by
    the weights' registration of record or one whose root the verifier itself holds (``held``, `held_entries`, or
    round 1's record for a value a link names, ``linked``, `second_stage.round_one_held`); that
    the verifier expects it registered by name excuses nothing, since R7 holds such a value's root only where it is held. `of_record`: the registration's bytes and the SHA-512 the auditor's side holds for it, None
    where it hands none; a pin's domain is its own program's (`_of_record`). A registered input neither pins nor expects
    that any template parameter reads is refused by name, at one reader as at several (`R8-unpinned`): nothing fixes
    its root before the coin, at one expected by name as at one not. A program with
    a registered input, or a verifier holding a registration, under a query that is no template query that applies is
    refused (`R8-weights-query`); a pinned input whose leaves, words or width are not the program's is refused
    (`R8-weights-shape`). A pin of no registered input excuses nothing. The refusals, in `WEIGHTS_WHY`'s order, and the accepted pins, `{value: {"root",
    "leaves", "words", "word_bits", "domain"}}` of each pinned input, which the members' check holds each member's
    registered read of it to (`members.pins_held`)."""
    marked = marked_inputs(program)
    if not marked and of_record is None:
        return [], {}
    out, pins = _of_record(of_record) if of_record is not None else ([], {})
    if out and out[0][0] in ("R8-weights-digest", "R8-weights-form"):
        return out, {}
    held = set(held) | set(linked)
    shape = []
    for v in marked:
        if v not in pins:
            continue
        e = pins[v]
        try:
            ok = G.shape(program, v, e["words"]) == e["leaves"] and G.width(program, v) == e.get(RV.WIDTH_KEY, RV.WORD_BITS)
        except ValueError:
            ok = False
        if not ok:
            shape.append(v)
    if shape:
        out.append(("R8-weights-shape", {"values": shape}))
    try:
        templated = P.is_template_query(query) and P.units(program, query) >= 0
    except P.QueryRefused:
        templated = False
    if not templated:
        if of_record is not None or marked:
            out.append(("R8-weights-query", {"query": dict(query)}))
        return out, {}
    loose = {v: g for v in marked if v not in pins and v not in held for g in G.held_gates(program, [v])}
    read = {g for _, _, g in P.input_reads(program, query, loose.values())} if loose else set()
    if unpinned := [v for v, g in loose.items() if g in read]:
        out.append(("R8-unpinned", {"values": unpinned}))
    excused = [v for v in marked if v in pins and v not in shape]
    accepted = {v: {"root": pins[v]["root"], "leaves": pins[v]["leaves"], "words": pins[v]["words"],
                    "word_bits": pins[v].get(RV.WIDTH_KEY, RV.WORD_BITS), "domain": pins[v]["domain"]}
                for v in excused}
    return out, accepted


WEIGHTS_UNREGISTERED_WHY = ("registered: the program marks registered inputs the verifier does not expect, and the parameters "
                    "carry no weights' registration of record R8 accepts: a marked input is read only at a pin R8 accepted")


WEIGHTS_QUERY_WHY = ("registered: the program marks registered inputs the verifier does not expect, and the partition is "
                     "no template query that applies to the program, whose ranges its members' reads of them are read "
                     "off: a marked input is read only at a pin R8 accepted")


def read_pins_why(program: Program, query: Mapping[str, Any]) -> str:
    """`Docket.readPinsWhy`: why `read_pins` is None, no template query applying (`WEIGHTS_QUERY_WHY`), else no
    registration R8 accepts (`WEIGHTS_UNREGISTERED_WHY`)."""
    templated = P.is_template_query(query) and isinstance(P.evaluate(program, query), PO.TemplatePartition)
    return WEIGHTS_UNREGISTERED_WHY if templated else WEIGHTS_QUERY_WHY


def read_pins(program: Program, query: Mapping[str, Any], registered: Iterable[str], of_record: tuple[bytes, str] | None,
              roots: Mapping[str, Mapping[str, Any]] | None = None
              ) -> tuple[list[str], list[tuple[str, range]], list[tuple[int, str, int, int]], dict[str, dict[str, Any]],
                         Mapping[str, Mapping[str, Any]]] | None:
    """`Docket.readPins`: what the members' check holds each registered read to, on R8's own derivation
    (`weights_held`): the marked inputs the verifier does not expect, their gates as ``(value, gates)``, the template
    ranges some member of which reads one (`partition.member_reads`), the accepted pins, and the roots the verifier
    itself holds (``roots``, `held_entries`). None where there is such an input and no registration of record, or R8
    refuses it; the first three empty where every marked input is expected."""
    roots = roots or {}
    expected = set(registered)
    loose = [v for v in marked_inputs(program) if v not in expected]
    if not loose:
        return [], [], [], weights_held(program, query, registered, of_record, roots)[1], roots
    if of_record is None:
        return None
    codes, pins = weights_held(program, query, registered, of_record, roots)
    if codes:
        return None
    part = P.evaluate(program, query)
    if not isinstance(part, PO.TemplatePartition):
        return None
    held = [(v, g) for v in loose for g in G.held_gates(program, [v])]
    ranges = [r for r in part.ranges if P.member_reads(program, held, r[0], 0, r[3])]
    return loose, held, ranges, pins, roots


def held_entries(program: Program, registered: Mapping[str, int], schemas: Mapping[str, str] | None,
                 window_root: str | None, held: Sequence[str] | None, supplied: Mapping[str, Any],
                 pinned: Mapping[str, str], outputs: Mapping[str, Any], log: Iterable[Mapping[str, Any]]
                 ) -> dict[str, dict[str, Any]]:
    """`Docket.heldEntries`: ``{value: {"root", "leaves", "domain"}}`` for each value the verifier expects registered
    whose root it holds itself, its ``pinned`` root, else the one an earlier record of the log registered under the
    domain it takes now (`held_roots`), as R7 holds it; none where its entries don't compute."""
    try:
        window, fixed = _window(window_root, held, registered, supplied, pinned, outputs)
        mine = G.registered_entries(program, registered, schemas, window, fixed)
    except ValueError:
        return {}
    logged = held_roots(log)
    out = {}
    for v, m in mine.items():
        r = pinned.get(v) or logged.get((v, m["domain"]))
        if r is not None:
            out[v] = {"root": r, "leaves": m["leaves"], "domain": m["domain"]}
    return out


#: `OneStage.PARTITION_QUERY`
PARTITION_QUERY = ("the partition object's query is not the parameters' query: the sample is drawn over the query's "
                   "population and every session's proof block is the partition's")


def partition_query_refused(partition: Mapping[str, Any], query: Mapping[str, Any]) -> bool:
    """`OneStage.paramsParsed`'s refusal: a partition object in any form but the pre-canonical `{program, query}` whose
    query is not `query`."""
    return set(partition) != {"program", "query"} and query_digest(partition.get("query") or {}) != query_digest(query)


def held_roots(log: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], str]:
    """`{(value, domain): root}` for every value an earlier record of the log registered, the root of its first
    docket under that domain: R7 holds a value to the root registered under the domain it takes now, so a
    per-window value, whose domain binds its window, is held to none of another window's."""
    held: dict[tuple[str, str], str] = {}
    for rec in log:
        for e in rec.get("registered") or ():
            held.setdefault((e["value"], e["domain"]), e["root"])
    return held


def receipt(rec: Mapping[str, Any], verifier: str, received: str | None = None) -> dict[str, Any]:
    """The verifier's own log entry for a record it accepted, with its own clock; the sample names it."""
    return {"format": RECEIPT, "registration": DD.digest(rec), "verifier": verifier,
            "received": received or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


HISTORY = "R7-history"
"""The verifier's code for a registration whose history tables both its verifiers refuse (`verifier.check`)."""


class HistoryRefused(ValueError):
    """History tables `check` can't read, or whose roots it can't hold to the ones the window registers."""


def resolved(supplied: Any, history: Mapping[str, Any] | None) -> Any:
    """``supplied`` as `check` read it under the history tables ``history``, for a docket `check` accepted: each
    source ``{"history": "H/root"}`` or ``{"history": "H/domain"}`` the bytes of table ``H``'s root or domain id
    (`history.resolve`), as the Lean's params hold them. It computes no table's root: `check` held each registered
    root to its table (`history.roots_of`), so the root is the registered one and the domain id the table's own."""
    if history is None or supplied is None:
        return supplied
    roots = {t["name"]: (bytes.fromhex(history["registered"][t["name"]]),
                         HT.domain(t["program"], t["name"], t["words"], t["capacity"]).domain_id)
             for t in history["tables"]}
    return HT.resolve(roots, supplied)


def with_history(supplied: Any, history: Mapping[str, Any] | None, leaf_layer: str | None) -> Any:
    """``supplied`` with each ``history`` source the bytes of the root or domain id `history.roots_of` computes for its
    table from the members' public bodies `leaf_layer` binds; `HistoryRefused` where it can't."""
    if history is None:
        return supplied
    if not isinstance(history, Mapping) or set(history) != {"digests", "held", "registered", "tables"}:
        raise HistoryRefused("the keys of the history tables are not digests, held, registered, tables")
    if leaf_layer is None:
        raise HistoryRefused("history tables are read from members' public bodies the leaf layer binds: the verifier "
                             "names no leaf_layer")
    try:
        return HT.resolve(HT.roots_of(leaf_layer, history), supplied)
    except ValueError as e:
        raise HistoryRefused(str(e)) from e


def check(rec: Mapping[str, Any], program: Program, query: Mapping[str, Any], sampling_design: Mapping[str, Any], scheme: str,
          ports: Mapping[str, Mapping[str, Any]], leaf_layer: str | None = None, anchors: Sequence[Any] = (),
          log: Iterable[Mapping[str, Any]] = (), partition: Mapping[str, Any] | None = None,
          work_table: Mapping[str, Any] | None = None, registered: Mapping[str, int] | None = None,
          supplied: Mapping[str, Any] | None = None, pinned: Mapping[str, str] | None = None,
          binding: Mapping[str, str] | None = None,
          outputs: Mapping[str, Mapping[str, str]] | None = None,
          schemas: Mapping[str, str] | None = None, window_root: str | None = None,
          held: Sequence[str] | None = None, members: Sequence[LP.Member] = (), day: int | None = None,
          declarations: Sequence[Mapping[str, Any]] | None = None, seed_salt: str | None = None,
          pool: Mapping[str, Any] | None = None, tied: Sequence[str] | None = None,
          history: Mapping[str, Any] | None = None, compiler: Any = UNREAD,
          certified: Mapping[str, Mapping[str, Any]] | None = None,
          verdicts: Mapping[str, bool] | None = None,
          coins: Mapping[str, str] | None = None, released: Mapping[str, int] | None = None,
          commitments: Mapping[str, str] | None = None, attester_keys: Sequence[str] = (),
          weights_of_record: tuple[bytes, str] | None = None,
          linked: Sequence[Mapping[str, Any]] = ()) -> Verdict:
    """R1-R8 against the verifier's own copies: its program and query, its sampling design and scheme, the ports its layout derives
    (`{port: {"leaves", "domain"}}`), the digest of its copy of the leaf layer, its anchors, its log of this prover's
    earlier records, the parameters it expects registered (`registered`: `{parameter: words}`), of those the ones it
    fills itself (`supplied`: `{parameter: [[source, ...], ...]}`, `registered.sources`; and C's supplied inputs, by their
    marks under its `coins` and `commitments` (`{function: hex}`: what it holds of each function as a coin and as a released
    commitment), `registered.marked_rows`, held to the order rule at `released`, its place of each coin,
    `registered.order`, the storage session's key read only when its SHA-512 is one of `attester_keys`, hex, the
    configuration's pins of the memory attester) and the roots it held before the
    record for others (`pinned`: `{parameter: root}`, the weights' root recorded before the salt, say), and of the rest the
    ones that are outputs of an earlier run or reuse an earlier record's registered value (`outputs`: `{parameter:
    {"record", "port" | "value"}}`, or `{parameter: {"port"}}`, that port of the log's last record); and
    those whose rows are not `row/v1` rows (`schemas`: `{parameter: schema}`, `registered_value.SCHEMAS`), whose rows R7 counts in
    that format (`registered_entries`), and a supplied one's it commits in it (`registered.supplied_root`); ValueError without `registered`, or in another form.  `binding`: the stream and prover the verifier registers this record for
    (`{"stream", "prover"}`, `_binding`), which the caller's check holds the window to name and a window salt binds.  `partition`: the
    verifier's own partition object when it isn't the canonical one (a record made before the canonical envelope); its
    program must still be `program`'s, and the population is `query`'s; a ValueError (`PARTITION_QUERY`) where it is in any
    form but the pre-canonical `{program, query}` and its query is not `query`.  R1 holds the registered sampling design to the verifier's own
    sampling design expanded over its program and query (`draw.derive`): for `stratified`, the strata by template descriptor id and
    each stratum's k by the rule; for `work`, also its own `work_table`.  R7 holds the registered values to the verifier's
    own (`registered_entries`), a pinned value to its pinned root, and a value an earlier record of the log registered
    under the domain it takes now to that record's root (`held_roots`): a value is committed once a domain; a supplied
    value's root is the one it recomputes from this record and the log (`registered.supplied_root`) instead, and an output
    value's domain and root are its earlier record's port's, or its earlier record's registered value's (`R7-output`, in
    the values' sorted order before the other R7 codes).  `window_root`: the window's open root R, which the verifier reads from the challenger's signed open of this
    record's window (never from the record), and which every registered value's domain binds but the ones it holds
    across windows: `held` (registered values, the weights, say), and its pinned, supplied and output values
    (`registered_value.domain`).  So a per-window value registered in one window is held to nothing in the next, and a root of
    another window's opens under no domain R7 takes.  Without `window_root`, a value it doesn't hold across windows that
    the log registered under the domain it takes now is a ValueError (`unwindowed`).  `day` and `declarations`: the
    window's day and the registration log's declarations (Declared day), both or neither, which the caller's check reads (`caller.check`).  `seed_salt`: the window's seed salt the verifier read from its
    challenger (Seed salt), which the caller's check holds the window to name; ValueError in another form or without `window_root`.
    `compiler`: the registration log's registered compiler, its item or None when it registers none (Registered
    compiler, `_registered_compiler`), whose digest the caller's check holds the window's compiler to and whose length ℓ its advice's
    bits; `UNREAD` holds the window to nothing; ValueError in another form.
    `pool`: the values a window opens and leaves its pool in, the empty pool's root, whether the verifier refused the
    window the public lag back, the window's own reset, which the caller's check holds it to name, and whether its week's one reset is
    spent (Pool chain, `_pool`), whose `pin` it adds to `outputs` or `supplied` (`pool_values`) and the caller's check holds to
    the chain (`R7-pool-chain`).  `history`: the history tables the program reads (``{"digests", "held", "registered",
    "tables"}``, `history.roots_of`), each table's root computed from its writers' public bodies the leaf layer binds
    and held to the root the window registers for it, which needs `leaf_layer`; a supplied value's source
    ``{"history": "H/root"}`` or ``{"history": "H/domain"}`` is then the bytes of the root or domain id it computed for
    table ``H`` (`history.resolve`).  History tables it can't read or hold to their roots raise `HistoryRefused`, which
    the verifier reports as `HISTORY` where its Lean refuses them too.  `certified`: the entries the verifier built from
    dockets it read under the deployment's key, of values that are none of its program's parameters (the
    certified rows of a link), ``{value: {"root", "leaves", "words", "domain"}}``, with `word_bits` when its leaves
    are not 16 bits (`_certified`).  With any, it expects a v2 record though it expects no parameter registered, R7
    holds the record's registered values to be the ones it expects registered and the certified ones, each once, a
    certified value's entry to the verifier's leaves, words, word bits and domain as its own values', and its root to
    the verifier's (`R7-certified`): a certified value is never supplied, pinned, held or an output, and a record that
    names one the verifier holds no entry for is refused as before (`R7-values`).  `verdicts`: the verifier's verdict
    log, in the pool's form (`_verdicts`), which the caller's check holds each output value's earlier port to
    (`caller.check`, `R7-output-refused`).  R3
    reads `ports` by name, so its codes come in the ports' sorted order whatever order the mapping has, as the Lean port
    reads them.  R7 refuses a registered value, a supplied one included, whose gates hold a run Input node
    (`run_input_named`, `R7-run-input`), and a supplied anchor whose value no pin holds (`anchor_unpinned`,
    `R7-anchor-unpinned`).  Under a template query the verdict carries the
    fan-out F of `query` on `program` (`partition.fanout`), which it publishes whatever its codes; under any other query it
    carries none.  `tied`: parameters, run Inputs, whose rows a later check ties to the dockets that computed them
    (the scan's slot entries, which the network period check ties to the blocks that registered them), which R8 lets
    through though several proof blocks read them, on that later check alone: the verdict publishes the ones it let
    through (`Verdict.rests_on_period_check`), and refuses a name that isn't a parameter of `program`, one `registered`
    or `held` names, or one whose gates aren't one interval of run Input nodes (`R8-tied`).  `weights_of_record`:
    the weights' registration of record, which R8 holds the program's registered inputs to (`weights_held`); the members'
    check holds each member's read of a pinned input to the pin it accepts (`members.pins_held`), and R7 reads none.  `members`: the per-member
    form's members, each its units (`[lo, hi)` ranges) and its circuit's units a proof block; after R8, `link-points`
    refuses the record when a member's fewest and most sampled proof blocks give different link points
    (`link_points.refused`), no sample under the sampling design ends (`link_points.NEVER_ENDS`), or the members or the
    sampling design are not well formed over the population."""
    c = _Collect()
    _declarations(day, declarations)
    log = list(log)
    pool = _pool(pool, registered or {}, supplied, pinned, outputs, schemas)
    if verdicts is not None:
        _verdicts(verdicts)
    supplied, outputs, schemas = pool_values(pool, program, supplied, outputs, schemas, log)
    supplied = with_history(supplied, history, leaf_layer)
    sup = G.sources(supplied if supplied is not None else {})
    sup = {**sup, **G.marked_rows(program, registered or {}, sup, coins, commitments, released, len(log),
                                  [bytes.fromhex(k) for k in attester_keys])}
    if supplied is not None and not registered:
        raise ValueError("supplied values are registered values: the verifier expects none registered")
    cert = _certified(certified, registered or {}, sup)
    v2 = bool(registered) or bool(cert)
    pins = _pins(pinned, registered or {}, sup)
    outs = _outputs(outputs, registered or {}, sup, pins, log)
    formats = _schemas(schemas, registered or {})
    bound = _binding(binding)
    window, fixed = _window(window_root, held, registered or {}, sup, pins, outs)
    _seed_salt(seed_salt, window)
    _registered_compiler(compiler)
    refuse_unwindowed(unwindowed(program, registered or {}, formats, window, fixed, log))
    fan, shared, through, bad = _readers(program, query, registered or {}, held or (), tied or ())
    weights, _accepted = weights_held(program, query, registered or {}, weights_of_record,
                                      held_entries(program, registered or {}, formats, window_root, held, sup, pins,
                                                   outs, log), [x["value"] for x in linked])
    keys = _KEYS | {"registered"} if v2 else _KEYS
    if not isinstance(rec, Mapping) or set(rec) != keys:
        c.fail("keys", sorted(set(rec) ^ keys) if isinstance(rec, Mapping) else rec)
        return c.verdict(fanout=fan, rests_on_period_check=through)
    # R1
    if (rec["format"], rec["protocol"]) != (DM.REGISTRATION if v2 else DM.REGISTRATION_V1, DM.ONE_STAGE_PROTOCOL):
        c.fail("R1-protocol", [rec["format"], rec["protocol"]])
    try:
        mine = D.derive(program, query, sampling_design, work_table)
    except (P.QueryRefused, ValueError) as e:
        c.fail("R1-law", {"registered": rec["law"], "verifier": f"{dict(sampling_design)} does not apply: {e}"})
    else:
        if not D.same_design(rec["law"], mine):
            c.fail("R1-law", {"registered": rec["law"], "verifier": mine})
    if not isinstance(rec["window"], Mapping) or rec["window"].get("kind") not in DD.KINDS:
        c.fail("window", rec["window"])
    # R2
    if partition is not None and partition_query_refused(partition, query):
        raise ValueError(PARTITION_QUERY)
    part = dict(partition) if partition is not None else P.partition(program, query)
    if part["program"] != P.program_digest(program):
        c.fail("R2-program", {"partition": part["program"], "verifier": P.program_digest(program)})
    try:
        part_digest = P.digest(part)
    except ValueError as e:
        c.fail("R2-partition", {"registered": rec["partition"], "verifier": str(e)})
        part_digest = None
    for field, mine in (("program", part["program"]), ("query", query_digest(part["query"])), ("partition", part_digest)):
        if mine is not None and rec[field] != mine:
            c.fail(f"R2-{field}", {"registered": rec[field], "verifier": mine})
    try:
        population = P.units(program, query)
    except P.QueryRefused as e:
        c.fail("R2-query", str(e))
    else:
        if rec["population"] != population:
            c.fail("R2-population", {"registered": rec["population"], "verifier": population})
    # R3
    if rec["scheme"] != scheme:
        c.fail("R3-scheme", {"registered": rec["scheme"], "verifier": scheme})
    roots = rec["roots"] if isinstance(rec["roots"], list) else []
    if not roots or not all(isinstance(r, Mapping) and set(r) == _ROOT_KEYS for r in roots):
        c.fail("R3-roots", rec["roots"])
    else:
        got = {r["port"]: r for r in roots}
        if len(got) != len(roots) or set(got) != set(ports):
            c.fail("R3-ports", {"registered": [r["port"] for r in roots], "verifier": sorted(ports)})
        for p, want in sorted(ports.items()):
            r = got.get(p)
            if r is None:
                continue
            if not (isinstance(r["root"], str) and _HEX.fullmatch(r["root"])):
                c.fail("R3-root", p)
            if r["leaves"] != want["leaves"]:
                c.fail("R3-leaves", {"port": p, "registered": r["leaves"], "verifier": want["leaves"]})
            if r["domain"] != want["domain"]:
                c.fail("R3-domain", {"port": p, "registered": r["domain"], "verifier": want["domain"]})
    # R4: when the verifier holds the leaf layer, the record must name it (a null leaf layer is refused)
    if rec["leaf_layer"] != leaf_layer:
        c.fail("R4-leaf-layer", {"registered": rec["leaf_layer"], "verifier": leaf_layer})
    # R5
    if rec["anchors"] != list(anchors):
        c.fail("R5-anchors", rec["anchors"])
    # R6
    earlier = list(log)
    if rec["previous"] != (DD.digest(earlier[-1]) if earlier else None):
        c.fail("R6-previous", rec["previous"])
    if any(e.get("window") == rec["window"] for e in earlier):
        c.fail("R6-window", rec["window"])
    # R7
    marks = G.marks(json.loads(b"".join(descriptor_chunks(program))))
    if v2:
        _check_registered(c, rec["registered"], program, registered or {}, pins, rec, sup, earlier, bound, outs, formats,
                          window, fixed, cert, table_entries(program, registered or {}, weights_of_record),
                          held_tables(weights_of_record))
        _certified_held(c, rec, cert)
        if named := run_input_named(program, registered or {}):
            c.fail("R7-run-input", {"values": named, "why": RUN_INPUT_WHY})
        if loose := anchor_unpinned(marks, pinned, weights_of_record):
            c.fail("R7-anchor-unpinned", {"values": loose, "why": ANCHOR_UNPINNED_WHY})
        if off := off_pin(marks, rec["registered"], weights_of_record):
            c.fail("R7-pin", {"values": off, "why": PIN_WHY})
    # a supplied input's rows have a source whatever the record registers
    if unsourced := unsourced_supplied(program, sup, outputs, marks, weights_of_record):
        c.fail("R7-supplied-source", {"values": unsourced, "why": SUPPLIED_SOURCE_WHY})
    # R8
    if bad:
        c.fail("R8-tied", {"tied": bad, "why": TIED_WHY})
    if shared and not members:
        c.fail("R8-shared-input", {"inputs": shared, "why": SHARED_WHY})
    for code, d in weights:
        c.fail(code, {"refused": d, "why": WEIGHTS_WHY[code]})
    if members:
        try:
            refused = LP.refused(*LP.sampling_design_of(program, query, sampling_design, work_table), members)
        except (P.QueryRefused, ValueError) as e:
            c.fail("link-points", str(e))
        else:
            if refused:
                c.fail("link-points", {"members": refused})
    return c.verdict(fanout=fan, rests_on_period_check=through)


def _readers(program: Program, query: Mapping[str, Any], registered: Iterable[str], held: Iterable[str],
             tied: Sequence[str]) -> tuple[int | None, list[int], tuple[str, ...], list[str]]:
    """R8's reading of `program`: under a template query that applies, its fan-out (`partition.fanout`), else None, and
    the run Input nodes its proof blocks don't each read alone outside the gates of the parameters the verifier expects
    `registered` and of the tied ones (`partition.shared_inputs`), else none; the tied names it lets through (`_tied`, the Lean's `Docket.tiedThrough`); and the tied names it can't let through. A
    function of the program, query and verifier alone, so a verdict publishes F and the tied names it let through
    whatever the record."""
    registered = list(registered)
    gates = _tied(program, tied, registered, held)
    bad = [t for t, g in zip(tied, gates) if g is None]
    through = tuple(t for t, g in zip(tied, gates) if g is not None)
    if not P.is_template_query(query):
        return None, [], through, bad
    try:
        fan, own = P.fanout(program, query), P.shared_inputs(program, query, G.held_gates(program, registered))
    except P.QueryRefused:
        return None, [], through, bad
    return fan, [k for k in own if not any(g is not None and P.within(program, g, k) for g in gates)], through, bad


def run_input_named(program: Program, registered: Iterable[str]) -> list[str]:
    """The registered names, supplied ones included, whose parameter's gates hold a run Input node
    (`partition.any_run_input`), which R7 refuses (`R7-run-input`). The Lean's `Docket.runInputNamed` is this."""
    return [v for v in registered if any(P.any_run_input(program, g) for g in G.held_gates(program, [v]))]


def anchor_unpinned(marks: Mapping[str, Mapping[str, Any]], pinned: Mapping[str, str] | None,
                    of_record: tuple[bytes, str] | None) -> list[str]:
    """The values V of C's supplied anchors (`marks`, `registered.marks`: a mark of a function of `ANCHOR_FUNCTIONS`)
    that are neither in the verifier's `pinned` nor pinned by the weights' registration of record (`_of_record`'s
    pins), each once, in the marks' order; R7 refuses them (`R7-anchor-unpinned`). C's marks come from a descriptor's
    `inputs`. The Lean's `Docket.anchorUnpinned` is this."""
    held = set(pinned or {}) | (set(_of_record(of_record)[1]) if of_record is not None else set())
    out: list[str] = []
    for m in marks.values():
        x = m["args"].get("value") if m["supplied"] in ANCHOR_FUNCTIONS else None
        if x is not None and x not in held and x not in out:
            out.append(x)
    return out


def off_pin(marks: Mapping[str, Mapping[str, Any]], got: Any, of_record: tuple[bytes, str] | None) -> list[str]:
    """The values the weights' registration of record pins (`_of_record`'s pins, in its order) whose record entry (the
    record's last ``registered`` entry of the value) is not that pin in root, leaf count, words and domain, or that a
    supplied anchor of C names (`marks`, a mark of `ANCHOR_FUNCTIONS`) and the record registers no entry of; R7 refuses
    them (`R7-pin`). A value the verifier's `pinned` names is held by `R7-held` and R7's own entry. The Lean's
    `Docket.offPin` is this."""
    pins = _of_record(of_record)[1] if of_record is not None else {}
    entries = [e for e in got if isinstance(e, Mapping)] if isinstance(got, list) else []
    anchors = {m["args"].get("value") for m in marks.values() if m["supplied"] in ANCHOR_FUNCTIONS}
    out: list[str] = []
    for x, pin in pins.items():
        e = next((e for e in reversed(entries) if e.get("value") == x), None)
        if (e is None and x in anchors) or (e is not None and any(
                e.get(k) != pin[k] or type(e.get(k)) is bool for k in ("root", "leaves", "words", "domain"))):
            out.append(x)
    return out


def held_tables(of_record: tuple[bytes, str] | None) -> dict[str, dict[str, str]]:
    """Each table the weights' registration of record pins (`_of_record`'s pins), its ``root`` and ``domain``: the only
    source of a table input's rows (`registered.HELD_ROOT`, `registered.HELD_DOMAIN`), never the record's entry. The
    Lean's `Docket.heldTables` is this."""
    pins = _of_record(of_record)[1] if of_record is not None else {}
    return {v: {"root": e["root"], "domain": e["domain"]} for v, e in pins.items()}


def table_entries(program: Program, registered: Mapping[str, int],
                  of_record: tuple[bytes, str] | None) -> dict[str, dict[str, Any]]:
    """R7's expected entry of each table a table input of C names (`supplied.registered_values` past the values the
    verifier expects registered) that the weights' registration of record pins, built from that pin: its leaves, words,
    word bits and domain (which `_of_record` holds to the derived one). A pin no mark names adds nothing, so the record
    registering it is refused `R7-values`. The Lean's `Docket.tableEntries` is this."""
    pins = _of_record(of_record)[1] if of_record is not None else {}
    names = registered_values(program, list(registered))[len(registered):]
    return {x: {"leaves": pins[x]["leaves"], "words": pins[x]["words"],
                "word_bits": pins[x].get(RV.WIDTH_KEY, RV.WORD_BITS), "domain": pins[x]["domain"]}
            for x in names if x in pins}


def unsourced_supplied(program: Program, supplied: Mapping[str, Any], outputs: Mapping[str, Any] | None = None,
                       marks: Mapping[str, Mapping[str, Any]] | None = None,
                       of_record: tuple[bytes, str] | None = None) -> list[str]:
    """The program's parameters bound as supplied inputs (`SuppliedInput<w>_v1`) that hold gates, that the parameters'
    ``supplied`` (C's marked rows among them) gives no source, or that are a table input (``marks``, a mark of
    `ANCHOR_FUNCTIONS`) whose table the registration of record pins not (`held_tables`: a mark is no source), and that
    are no output of an earlier record (``outputs``, a pool's reused ``pin`` among them); R7 refuses them
    (`R7-supplied-source`). The Lean's `Docket.unsourced` is this."""
    bindings, outputs, held = getattr(program, "bindings", ()), outputs or {}, held_tables(of_record)
    unheld = {n for n, m in (marks or {}).items()
              if m["supplied"] in ANCHOR_FUNCTIONS and m["args"].get("value") not in held}
    return [n for (n, _), b in zip(program.fn.signature()[0], bindings)
            if b == SUPPLIED and G.held_gates(program, [n]) and (n not in supplied or n in unheld)
            and n not in outputs] if bindings else []


def marked_gates(program: Program) -> list[tuple[str, range]]:
    """Every marked parameter's gates, ``(value, gates)`` each (`marked_inputs`): the members' check holds each port
    reading one to its entry of record. The Lean's `Docket.markedGates` is this."""
    return [(v, g) for v in marked_inputs(program) for g in G.held_gates(program, [v])]


def _tied(program: Program, tied: Sequence[str], registered: Sequence[str], held: Iterable[str]) -> list[range | None]:
    """Each of `tied`'s Input gates where R8 may let it through, else None: a parameter of the program that the verifier
    neither expects registered nor holds, whose gates are one non-empty interval (`registered.held_gates`) of run Input
    nodes (`partition.run_inputs`). A tied Input is never a registered value: R7 holds nothing of it. The Lean's
    `Docket.tiedGate` is this."""
    fixed = {*registered, *held}
    out: list[range | None] = []
    for t in tied:
        g = None if t in fixed else next(iter(G.held_gates(program, [t])), None)
        out.append(g if g is not None and P.run_inputs(program, g) else None)
    return out


def _window(window_root: str | None, held: Sequence[str] | None, registered: Mapping[str, int],
            supplied: Mapping[str, Any], pinned: Mapping[str, str],
            outputs: Mapping[str, tuple[str, str, str]]) -> tuple[str | None, tuple[str, ...]]:
    """The window's open root (`registered_value.window_root`) and the values whose domain leaves it out: `held`, a list of
    values it expects registered, then its pinned, supplied and output values; ValueError otherwise, which the Lean
    verifier refuses too (`OneStage.paramsOfJson`)."""
    if held is not None and not (isinstance(held, (list, tuple)) and all(isinstance(v, str) and v in registered
                                                                          for v in held)):
        raise ValueError(f"held values are a list of registered values: {held!r}")
    return RV.window_root(window_root), (*(held or ()), *pinned, *supplied, *outputs)


def unwindowed(program: Program, values: Mapping[str, int], schemas: Mapping[str, str] | None, window: str | None,
               fixed: Sequence[str], log: Iterable[Mapping[str, Any]]) -> list[str]:
    """The values a verifier naming no window root can't take: each it expects registered and doesn't hold across
    windows (`fixed`) that an earlier record of the log registered under the domain it takes now. A window-independent
    program that registers its per-window values under fixed names reaches them from its second window on, and without
    R every window's domain is the same, so an earlier window's roots would hold. None with a window root, and none for a
    program whose per-window values take a fresh name each window."""
    if window is not None or not values:
        return []
    try:
        mine = G.registered_entries(program, values, schemas, fixed=fixed)
    except ValueError:
        return []
    logged = held_roots(log)
    return sorted(v for v, m in mine.items() if v not in fixed and (v, m["domain"]) in logged)


def refuse_unwindowed(again: Sequence[str]) -> None:
    """ValueError naming the fix when `unwindowed` found values; the Lean verifier refuses too (`OneStage.paramsOfJson`)."""
    if again:
        raise ValueError(f"registered values {list(again)} were registered by an earlier record of the log under the domains "
                         "they take now, and the verifier names no window root: a value registered afresh each window under "
                         "a fixed name takes the window's open root (`window_root`, from the challenger's signed open of "
                         "the window), and a value held across windows is named in `held` (or pinned)")


def last_digest(log: Sequence[Mapping[str, Any]]) -> str:
    """The previous record's digest, the one R6 holds `previous` to: the log's last record's, and empty, which names no
    record, when the log is empty (`Docket.lastDigest`)."""
    if not log:
        return ""
    try:
        return DD.digest(log[-1])
    except (TypeError, ValueError):
        return ""


def _outputs(outputs: Mapping[str, Mapping[str, str]] | None, registered: Mapping[str, int], supplied: Mapping[str, Any],
             pinned: Mapping[str, str], log: Sequence[Mapping[str, Any]] = ()) -> dict[str, tuple[str, str, str]]:
    """The verifier's output values, `{value: (record digest, "port" | "value", name)}`, each of a value it expects
    registered and neither fills nor pinned, an object of exactly a `record` (lowercase hex of 64 bytes) and either a `port`
    or a `value` (a string: the earlier record's port, or its registered value), or of a `port` alone, a window's
    history: that port of the previous record, whose digest the verifier takes from `log` (`last_digest`); ValueError
    otherwise, which the Lean verifier refuses too (`OneStage.paramsOfJson`)."""
    if outputs is None:
        return {}
    if not isinstance(outputs, Mapping) or not registered:
        raise ValueError(f"output values are an object of registered values' earlier ports or values: {outputs!r}")
    out = {}
    for v, o in outputs.items():
        if (v in registered and v not in supplied and v not in pinned and isinstance(o, Mapping) and set(o) == {"port"}
                and isinstance(o["port"], str)):
            out[v] = (last_digest(log), "port", o["port"])
            continue
        kind = next((k for k in ("port", "value") if isinstance(o, Mapping) and set(o) == {k, "record"}), None)
        if (v not in registered or v in supplied or v in pinned or kind is None or not isinstance(o[kind], str)
                or not (isinstance(o["record"], str) and len(o["record"]) == 128 and _HEX.fullmatch(o["record"]))):
            raise ValueError(f"output value {v!r}: not a registered value it neither fills nor pinned, with an earlier "
                             f"record's digest and a port or a value, or a port of the previous record: {o!r}")
        out[v] = (o["record"], kind, o[kind])
    return out


def _schemas(schemas: Mapping[str, str] | None, registered: Mapping[str, int]) -> dict[str, str]:
    """The verifier's row formats, each of a value it expects registered, one of `registered_value.SCHEMAS`: a supplied value's
    is the format R7 commits its rows in (`registered.supplied_root`); ValueError otherwise, which the Lean verifier
    refuses too (`OneStage.paramsOfJson`)."""
    if schemas is None:
        return {}
    if not isinstance(schemas, Mapping) or not registered:
        raise ValueError(f"row formats are an object of registered values' formats: {schemas!r}")
    for v, s in schemas.items():
        if v not in registered or not isinstance(s, str) or s not in RV.SCHEMAS:
            raise ValueError(f"row format of {v!r}: not a registered value's, in one of {RV.SCHEMAS}: {s!r}")
    return dict(schemas)


def _output_port(log: Sequence[Mapping[str, Any]], record: str, port: str, key: str = "roots",
                 name: str = "port") -> Mapping[str, Any] | None:
    """The `roots` entry for `port` of the log's record with digest `record` (its last, as a dict reads it), or None; with
    `key="registered", name="value"`, its `registered` entry for the value `port`."""
    for r in log:
        try:
            d = DD.digest(r)
        except (TypeError, ValueError):
            continue
        if d != record:
            continue
        roots = r.get(key)
        hits = [e for e in roots if isinstance(e, Mapping) and e.get(name) == port] if isinstance(roots, list) else []
        return hits[-1] if hits else None
    return None


def _earlier(log: Sequence[Mapping[str, Any]], record: str, kind: str, name: str,
             want: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """An output value's earlier entry: the port `name` of the log's record with digest `record`, with the value's leaf
    count, or (`kind == "value"`) its registered value `name`, of the log's last record alone (`last_digest`), with the
    value's leaves, words and word bits; each with a string domain and root, else None."""
    if kind == "value" and record != last_digest(log):
        return None
    if kind == "port":
        e = _output_port(log, record, name)
        ok = e is not None and type(e.get("leaves")) is int and e["leaves"] == want["leaves"]
    else:
        e = _output_port(log, record, name, "registered", "value")
        width = e.get(RV.WIDTH_KEY, RV.WORD_BITS) if e is not None else None
        ok = (e is not None and type(e.get("leaves")) is int and e["leaves"] == want["leaves"]
              and type(e.get("words")) is int and e["words"] == want["words"]
              and type(width) is int and width == want["word_bits"])
    return e if ok and isinstance(e.get("domain"), str) and isinstance(e.get("root"), str) else None


def _pins(pinned: Mapping[str, str] | None, registered: Mapping[str, int], supplied: Mapping[str, Any]) -> dict[str, str]:
    """The verifier's pinned roots, each of a value it expects registered and doesn't fill itself, as lowercase hex of
    64 bytes; ValueError otherwise, which the Lean verifier refuses too (`OneStage.paramsOfJson`)."""
    if pinned is None:
        return {}
    if not isinstance(pinned, Mapping) or not registered:
        raise ValueError(f"pinned roots are an object of registered values' roots: {pinned!r}")
    for v, r in pinned.items():
        if v not in registered or v in supplied or not (isinstance(r, str) and len(r) == 128 and _HEX.fullmatch(r)):
            raise ValueError(f"pinned value {v!r}: not a registered value it doesn't fill, pinned to 64 bytes: {r!r}")
    return dict(pinned)


CERTIFIED_KEYS = ("root", "leaves", "words", "domain")
"""A certified entry's keys, with `registered_value.WIDTH_KEY` for leaves that are not 16 bits (`_certified`)."""


def _certified(certified: Mapping[str, Mapping[str, Any]] | None, registered: Mapping[str, int],
               supplied: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The verifier's certified entries, `{value: {"root", "leaves", "words", "word_bits", "domain"}}` in the values'
    sorted order: each of a value it neither expects a parameter registered under (`registered`) nor fills itself
    (`supplied`), an object of exactly `CERTIFIED_KEYS` (`root` lowercase hex of 64 bytes, `leaves` and `words`
    naturals, `domain` a string) and, optionally, `word_bits`, one of `registered_value.WIDTHS`, 16 when left out; ValueError
    otherwise, which the Lean verifier refuses too (`OneStage.paramsParsed`)."""
    if certified is None:
        return {}
    if not isinstance(certified, Mapping) or not all(isinstance(v, str) for v in certified):
        raise ValueError(f"certified entries are an object of values' entries: {certified!r}")
    keys, out = set(CERTIFIED_KEYS), {}
    for v, e in sorted(certified.items()):
        width = e.get(RV.WIDTH_KEY, RV.WORD_BITS) if isinstance(e, Mapping) else None
        if (v in registered or v in supplied or not isinstance(e, Mapping) or not keys <= set(e) <= keys | {RV.WIDTH_KEY}
                or not (isinstance(e["root"], str) and len(e["root"]) == 128 and _HEX.fullmatch(e["root"]))
                or not all(type(e[k]) is int and e[k] >= 0 for k in ("leaves", "words"))
                or not isinstance(e["domain"], str) or type(width) is not int or width not in RV.WIDTHS):
            raise ValueError(f"certified value {v!r}: one it expects a parameter registered under or fills, or an entry "
                             f"that is not a root of 64 bytes, its leaves, words and domain, and word_bits when not "
                             f"{RV.WORD_BITS}: {e!r}")
        out[v] = {"root": e["root"], "leaves": e["leaves"], "words": e["words"], "word_bits": width,
                  "domain": e["domain"]}
    return out


def _declarations(day: int | None, declarations: Sequence[Mapping[str, Any]] | None
                  ) -> tuple[int, list[Mapping[str, Any]]] | None:
    """The window's day and the log's declarations (Declared day): both or neither, the day a natural number and each
    declaration an object of exactly `blocks`, `day` and `program`, as the registration log's view writes it, with a
    natural day and a string program; ValueError otherwise."""
    if day is None and declarations is None:
        return None
    if type(day) is not int or day < 0 or not isinstance(declarations, (list, tuple)):
        raise ValueError(f"a declared day is a natural day and the log's declarations, both or neither: {day!r}, "
                         f"{declarations!r}")
    for d in declarations:
        if not (isinstance(d, Mapping) and set(d) == {"blocks", "day", "program"} and type(d["day"]) is int
                and d["day"] >= 0 and isinstance(d["program"], str)):
            raise ValueError(f"a declaration is the registration log view's {{blocks, day, program}}: {d!r}")
    return day, list(declarations)


def _binding(binding: Mapping[str, str] | None) -> dict[str, str] | None:
    """The verifier's binding: exactly `stream`, its stream's digest (lowercase hex of 64 bytes), and `prover`, the identity
    it registered the prover under (lowercase hex of whole bytes); ValueError otherwise, which the Lean verifier refuses
    too (`OneStage.paramsOfJson`)."""
    if binding is None:
        return None
    if not (isinstance(binding, Mapping) and set(binding) == {"stream", "prover"}):
        raise ValueError(f"a binding is exactly a stream and a prover: {binding!r}")
    s, p = binding["stream"], binding["prover"]
    if not (isinstance(s, str) and len(s) == 128 and _HEX.fullmatch(s)):
        raise ValueError(f"the binding's stream is not lowercase hex of 64 bytes: {s!r}")
    if not (isinstance(p, str) and _HEX.fullmatch(p) and len(p) % 2 == 0):
        raise ValueError(f"the binding's prover is not lowercase hex bytes: {p!r}")
    return {"stream": s, "prover": p}


def _seed_salt(seed_salt: str | None, window: str | None) -> str | None:
    """The window's seed salt the verifier read (Seed salt): None, or lowercase hex of 64 bytes beside the window's open
    root `window`; ValueError otherwise, which the Lean verifier refuses too (`OneStage.paramsOfJson`)."""
    if seed_salt is None:
        return None
    if not (isinstance(seed_salt, str) and len(seed_salt) == 128 and _HEX.fullmatch(seed_salt)):
        raise ValueError(f"a seed salt is lowercase hex of 64 bytes: {seed_salt!r}")
    if window is None:
        raise ValueError("a seed salt is read on the window's session: it needs the window's open root (window_root)")
    return seed_salt


def _registered_compiler(item: Any) -> Any:
    """The registered compiler the verifier read (Registered compiler): `UNREAD`; None, a log that registers none; or the
    log's item as its view publishes it (`verity.protocols.registration.verifier.registration`), its digest lowercase hex of 64 bytes, its
    advice format a string and its length ℓ a natural, as `(digest, ℓ)`; ValueError otherwise, which the Lean verifier
    refuses too (`OneStage.paramsOfJson`)."""
    if item is UNREAD or item is None:
        return item
    cap_bits = item.get("cap") if isinstance(item, Mapping) else None
    if not (isinstance(item, Mapping) and set(item) == {"advice", "cap", "digest", "kind"} and item["kind"] == "builder"
            and isinstance(item["digest"], str) and isinstance(item["advice"], str)
            and type(cap_bits) is int and cap_bits >= 0):
        raise ValueError("the registered compiler is the registration log's item: its kind, a digest, an advice format and "
                         f"a natural length, not {item!r}")
    if not (len(item["digest"]) == 128 and _HEX.fullmatch(item["digest"])):
        raise ValueError(f"the registered compiler's digest is not lowercase hex of 64 bytes: {item['digest']!r}")
    return item["digest"], cap_bits


#: a pool's fields (Pool chain)
POOL_KEYS = ("genesis", "lag", "pin", "pout", "reset", "verdicts", "week")


def _names(x: Any, want: int | None) -> bool:
    """Whether a window field names `want`, an integer that is not a bool, or none (`jeq`)."""
    return x is None if want is None else type(x) is int and x == want


def _pool(pool: Mapping[str, Any] | None, registered: Mapping[str, int], supplied: Any, pinned: Any,
          outputs: Any, schemas: Any) -> dict[str, Any] | None:
    """The verifier's pool (Pool chain): None, or exactly `pin` and `pout`, two registered values it neither fills,
    pinned nor gives as outputs, `pin` with no row format of its own (it takes `pout`'s, `pool_values`), `genesis`,
    lowercase hex of 64 bytes, `lag`, the public lag at which serving learns a verdict, a positive integer, `verdicts`,
    the verifier's verdict log, an object of record digests (lowercase hex of 64 bytes) to whether its run on that
    record accepted, `reset`, the reset the record's window names, a bool R6 holds it to, and `week`, the window's week
    on the verifier's clock, a natural R6 holds it to. ValueError otherwise, which the Lean verifier refuses too
    (`OneStage.paramsOfJson`)."""
    if pool is None:
        return None
    if not (isinstance(pool, Mapping) and set(pool) == set(POOL_KEYS)):
        raise ValueError(f"a pool is exactly {list(POOL_KEYS)}: {pool!r}")
    given = [m for m in (supplied, pinned, outputs) if isinstance(m, Mapping)]
    for v in (pool["pin"], pool["pout"]):
        if not (isinstance(v, str) and v in registered and not any(v in m for m in given)):
            raise ValueError(f"the pool's {v!r}: not a registered value it neither fills, pinned nor gives as an output")
    if pool["pin"] == pool["pout"]:
        raise ValueError(f"a pool opens and closes in two values, not {pool['pin']!r} alone")
    if isinstance(schemas, Mapping) and pool["pin"] in schemas:
        raise ValueError(f"the pool's {pool['pin']!r} takes the row format of {pool['pout']!r}, which it reuses")
    g = pool["genesis"]
    if not (isinstance(g, str) and len(g) == 128 and _HEX.fullmatch(g)):
        raise ValueError(f"the pool's genesis is lowercase hex of 64 bytes: {g!r}")
    if type(pool["reset"]) is not bool:
        raise ValueError(f"the pool's reset is a bool: {pool['reset']!r}")
    if not (type(pool["lag"]) is int and pool["lag"] >= 1):
        raise ValueError(f"the pool's lag is a positive integer: {pool['lag']!r}")
    if not (type(pool["week"]) is int and pool["week"] >= 0):
        raise ValueError(f"the pool's week is a natural: {pool['week']!r}")
    _verdicts(pool["verdicts"], "the pool's verdicts")
    return dict(pool)


def _verdicts(verdicts: Any, what: str = "the verdicts") -> dict[str, bool]:
    """A verdict log: an object of record digests (lowercase hex of 64 bytes) to whether the verifier's run on that record
    accepted; ValueError otherwise, which the Lean verifier refuses too (`OneStage.verdictsOf`)."""
    if not (isinstance(verdicts, Mapping) and all(isinstance(d, str) and len(d) == 128 and _HEX.fullmatch(d)
                                                  and type(a) is bool for d, a in verdicts.items())):
        raise ValueError(f"{what} are an object of record digests (lowercase hex of 64 bytes) to bools: {verdicts!r}")
    return dict(verdicts)


def refused(verdicts: Mapping[str, bool], d: str) -> bool:
    """Whether the verdict log names the verifier's run on record `d` refused; a record it holds no verdict for is not
    refused (`Docket.refusedIn`)."""
    return verdicts.get(d) is False


def pool_refused(pool: Mapping[str, Any], log: Sequence[Mapping[str, Any]]) -> bool:
    """Whether the verifier refused the window `lag` windows of record before this one (Pool chain): its verdict log
    (`verdicts`) names its run on that record of the log refused. A log shorter than the lag, or a record it holds no
    verdict for, is not refused, so where every run accepted the chain never restarts (`Docket.Pool.refused`)."""
    lag = pool["lag"]
    if len(log) < lag:
        return False
    try:
        d = DD.digest(log[len(log) - lag])
    except (TypeError, ValueError):
        return False
    return refused(pool["verdicts"], d)


def pool_spent_before(pool: Mapping[str, Any], log: Sequence[Mapping[str, Any]]) -> bool:
    """Whether the week's one reset is spent before this window (Pool chain): the log's last record, the previous window
    of record, names it spent (`POOL_SPENT`) in the verifier's week (`POOL_WEEK`), so a week starts unspent
    (`Docket.Pool.spentBefore`)."""
    last = log[-1] if log else None
    w = last.get("window") if isinstance(last, Mapping) else None
    return isinstance(w, Mapping) and w.get(POOL_SPENT) is True and _names(w.get(POOL_WEEK), pool["week"])


def pool_chains(pool: Mapping[str, Any], log: Sequence[Mapping[str, Any]], program: Program) -> bool:
    """Whether the window carries the previous window's pool (Pool chain): neither the verifier refused the window the
    public lag back (`pool_refused`) nor serving reset it (`reset`), and the log's last record, the previous window of
    record, registered its program (`Docket.Pool.chains`). Otherwise it opens the empty pool."""
    last = log[-1] if log else None
    return (not pool["reset"] and not pool_refused(pool, log) and isinstance(last, Mapping)
            and last.get("program") == P.program_digest(program))


def pool_values(pool: Mapping[str, Any] | None, program: Program, supplied: Any, outputs: Any, schemas: Any,
                log: Sequence[Mapping[str, Any]]) -> tuple[Any, Any, Any]:
    """The verifier's supplied values, output values and row formats with its pool's `pin` among them (Pool chain): the
    previous record's `pout`, reused (`{"record": last_digest(log), "value": pout}`) in `pout`'s row format, when the
    window carries the previous window's pool (`pool_chains`), else filled from the empty pool's root (`[[{"bytes":
    genesis}]]`) in the default one. What reads the verifier's values (the reads file's domains, formats and fixed
    values) takes these. A malformed object is left as it is, for its own check to refuse."""
    if pool is None:
        return supplied, outputs, schemas
    pin, pout = pool["pin"], pool["pout"]
    if not pool_chains(pool, log, program):
        if supplied is not None and not isinstance(supplied, Mapping):
            return supplied, outputs, schemas
        return {**(supplied or {}), pin: [[{"bytes": pool["genesis"]}]]}, outputs, schemas
    if isinstance(schemas, Mapping) and pout in schemas:
        schemas = {**schemas, pin: schemas[pout]}
    if outputs is not None and not isinstance(outputs, Mapping):
        return supplied, outputs, schemas
    return supplied, {**(outputs or {}), pin: {"record": last_digest(log), "value": pout}}, schemas


def pool_open(pool: Mapping[str, Any], rec: Mapping[str, Any], program: Program, values: Mapping[str, int],
              schemas: Mapping[str, str] | None, window: str | None, fixed: Sequence[str],
              supplied: Mapping[str, Any], log: Sequence[Mapping[str, Any]],
              binding: Mapping[str, str] | None) -> tuple[str, str] | str:
    """The domain and root of the pool the record's window opens (Pool chain): the previous record's entry for `pout` (its
    last naming it) when the window carries the previous window's pool (`pool_chains`), else the empty pool's: the
    domain the verifier gives `pin` (`registered_entries`) and the root R7 recomputes for it filled from `genesis`
    (`registered.supplied_root`); where there is none, why (`Docket.poolOpen`, none there)."""
    if pool_chains(pool, log, program):
        es = log[-1].get("registered")
        hits = [e for e in es if isinstance(e, Mapping) and e.get("value") == pool["pout"]] if isinstance(es, list) else []
        d, r = (hits[-1].get("domain"), hits[-1].get("root")) if hits else (None, None)
        if isinstance(d, str) and isinstance(r, str):
            return d, r
        return f"the previous record registers no {pool['pout']!r} with a domain and root"
    try:
        want = G.registered_entries(program, values, schemas, window, fixed).get(pool["pin"])
        if want is None:
            return f"the verifier gives {pool['pin']!r} no entry"
        return want["domain"], G.supplied_root(program, pool["pin"], [[("bytes", pool["genesis"])]], rec, want,
                                               sorted(supplied), log, None, binding)
    except ValueError as e:
        return f"the empty pool fills no {pool['pin']!r}: {e}"


def _certified_held(c: _Collect, rec: Mapping[str, Any], certified: Mapping[str, Mapping[str, Any]]) -> None:
    """R7's certified roots: the record's entry for each certified value (its last naming it) has the root the verifier
    holds for it (`_certified`), else `R7-certified`."""
    got = rec["registered"]
    for v, want in certified.items():
        hits = [e for e in got if isinstance(e, Mapping) and e.get("value") == v] if isinstance(got, list) else []
        mine = hits[-1].get("root") if hits else None
        if not hits or mine != want["root"]:
            c.fail("R7-certified", {"value": v, "registered": mine, "verifier": want["root"]})


def _check_registered(c: _Collect, got: Any, program: Program, values: Mapping[str, int], pinned: Mapping[str, str],
                      rec: Mapping[str, Any], supplied: Mapping[str, Any], log: Sequence[Mapping[str, Any]],
                      binding: Mapping[str, str] | None = None,
                      outputs: Mapping[str, tuple[str, str, str]] | None = None,
                      schemas: Mapping[str, str] | None = None, window: str | None = None,
                      fixed: Sequence[str] = (), certified: Mapping[str, Mapping[str, Any]] | None = None,
                      tables: Mapping[str, Mapping[str, Any]] | None = None,
                      held_tables: Mapping[str, Mapping[str, str]] | None = None) -> None:
    try:
        mine = G.registered_entries(program, values, schemas, window, fixed)
    except ValueError as e:
        c.fail("R7-registered", {"verifier": f"{dict(values)} does not apply: {e}"})
        return
    logged = held_roots(log)
    held = {v: logged[(v, m["domain"])] for v, m in mine.items() if (v, m["domain"]) in logged}
    held.update(pinned)
    # the tables C's table inputs name, built from their pins: R7's expected entries after the verifier's own
    mine.update({v: dict(e) for v, e in (tables or {}).items()})
    for v, (d, kind, name) in sorted((outputs or {}).items()):
        if v not in mine:
            continue
        e = _earlier(log, d, kind, name, mine[v])
        if e is None:
            c.fail("R7-output", {"value": v, "record": d, kind: name, "verifier": mine[v]})
            continue
        mine[v] = {**mine[v], "domain": e["domain"]}
        held[v] = e["root"]
    # the certified entries, after the verifier's own: none is an output, and `_certified_held` holds its root
    for v, e in (certified or {}).items():
        mine[v] = {k: e[k] for k in ("leaves", "words", "word_bits", "domain")}
    keys = set(RV.ENTRY_KEYS)
    if not isinstance(got, list) or not all(isinstance(e, Mapping) and keys <= set(e) <= keys | {RV.WIDTH_KEY} for e in got):
        c.fail("R7-registered", got)
        return
    by_value = {e["value"]: e for e in got}
    if len(by_value) != len(got) or set(by_value) != set(mine):
        c.fail("R7-values", {"registered": [e["value"] for e in got], "verifier": sorted(mine)})
    for v, want in mine.items():
        e = by_value.get(v)
        if e is None:
            continue
        if not (isinstance(e["root"], str) and len(e["root"]) == 128 and _HEX.fullmatch(e["root"])):
            c.fail("R7-root", v)
        # a count is an int, never a float or bool equal to it, and 16 is named by leaving word_bits out: one record
        # digest per docket
        for k in ("leaves", "words", "domain"):
            if e[k] != want[k] or (k != "domain" and type(e[k]) is not int):
                c.fail(f"R7-{k}", {"value": v, "registered": e[k], "verifier": want[k]})
        width = e.get(RV.WIDTH_KEY, RV.WORD_BITS)
        if type(width) is not int or width != want["word_bits"] or e.get(RV.WIDTH_KEY) == RV.WORD_BITS:
            c.fail("R7-word_bits", {"value": v, "registered": width, "verifier": want["word_bits"]})
        if v in held and v not in supplied and e["root"] != held[v]:
            c.fail("R7-held", {"value": v, "registered": e["root"], "verifier": held[v]})
    prog = P.program_digest(program) if supplied else None
    for v in sorted(supplied):
        want, e = mine.get(v), by_value.get(v)
        if want is None:
            c.fail("R7-supplied", {"value": v, "verifier": "not a value it expects registered"})
            continue
        if e is None:
            continue
        try:
            root = G.supplied_root(program, v, supplied[v], rec, want, sorted(supplied), log, prog, binding,
                                   (schemas or {}).get(v, RV.SCHEMA), held_tables)
        except ValueError as err:
            c.fail("R7-supplied", {"value": v, "verifier": str(err)})
            continue
        if e["root"] != root:
            c.fail("R7-supplied", {"value": v, "registered": e["root"], "verifier": root})


#: `OneStageZk.TABLE_PORT`, the port a docket names a window's row tables' root at
TABLE_PORT = "row-table"
#: `OneStageZk.ROW_TABLES`
ROW_TABLES = ("roots: the registration names a row tables' root (row-table), the window's and no member's: the per-member "
              "form is refused on a window with row tables until its guarantee covers a root every member reads")
#: `OneStageZk.NOT_ONE_MEMBER`
NOT_ONE_MEMBER = ("public file: the registration's roots are not one member's: the one statement's public file is held to "
                  "a registration whose ports are bare, or are all one window member's, <member>/<port>")


def name_template(name: str) -> str:
    """`OneStageZk.NAME_TEMPLATE`."""
    return (f"roots: the member {name}'s name is another member's template: a root {name}/<field> would read both as the "
            "member's own port and as the window's tree of the template's field")


def own_field(name: str, field: str) -> str:
    """`OneStageZk.OWN_FIELD`."""
    return (f"roots: the member {name} holds a root of its own at {field}, a field its template has a window's tree of: a "
            "member reads its template's field at the window's one tree, which every wire into the field reads")


def public_ports(rec: Mapping[str, Any]) -> dict[str, Any]:
    """`OneStageZk.publicPorts`: the record's roots by port as v2's one statement's public file names them (bare): its
    own where no root's port holds a ``/``; else, where every root's port but the window's trees of a template's field
    (``<template>/<field>``, its template another port's name's template) is ``<member>/<port>`` of one member, whose
    own port names hold no ``/``, that member's roots with its name and the ``/`` taken off, with its template's fields'
    trees at its fields; a run input node's tree ``input/n<w>`` is the window's, no public file's. Raises `ValueError`
    with v2's refusal where the roots are two members' or more, or bare and a
    member's (`NOT_ONE_MEMBER`), or one is a row tables' (`ROW_TABLES`) or names no port, or the member holds a root of
    its own at a field its template has a tree of (`own_field`)."""
    roots = [r for r in rec["roots"] if not G.is_fresh_value(r.get("port"))]
    ports = [r["port"] for r in roots if isinstance(r.get("port"), str)]
    if all("/" not in p for p in ports):
        return {r["port"]: r for r in roots if isinstance(r.get("port"), str)}
    ts = {n.rsplit("/", 1)[0] for p in ports if "/" in p and "/" in (n := p.rsplit("/", 1)[0])}
    names = {p.rsplit("/", 1)[0] if "/" in p else None for p in ports if "/" not in p or p.rsplit("/", 1)[0] not in ts}
    if None in names or len(names) != 1:
        raise ValueError(NOT_ONE_MEMBER)
    name, = names
    out: dict[str, Any] = {}
    fields: dict[str, Any] = {}
    for r in roots:
        port = r.get("port")
        if not isinstance(port, str):
            raise ValueError("roots: a root of the registration names no port")
        if port == TABLE_PORT:
            raise ValueError(ROW_TABLES)
        if not port.startswith(name + "/") and port != name and "/" in port and port.rsplit("/", 1)[0] in ts:
            t, f = port.rsplit("/", 1)
            if name.rsplit("/", 1)[0] == t and f not in fields:
                fields[f] = {"domain": r.get("domain"), "leaves": r.get("leaves"), "port": f, "root": r.get("root"),
                             "tree": port}
            continue
        bare = port[len(name) + 1:]
        out.pop(bare, None)
        out[bare] = {"domain": r.get("domain"), "leaves": r.get("leaves"), "port": bare, "root": r.get("root")}
    if (f := next((f for f in fields if f in out), None)) is not None:
        raise ValueError(own_field(name, f))
    return {**out, **fields}


def matches_public(rec: Mapping[str, Any], scheme: str, roots: Mapping[str, str], domains: Mapping[str, str],
                   population: int, reg: Mapping[str, Any] | None = None) -> Verdict:
    """The verifier's own public file (its scheme, roots, their domains and its instance count) against the record, at
    the record's roots by port ``reg`` (`Record.matchesPublicBy`; the record's own by default, `Record.matchesPublic`,
    or v2's one statement's, `public_ports`)."""
    c = _Collect()
    reg = {r["port"]: r for r in rec["roots"]} if reg is None else reg
    if scheme != rec["scheme"]:
        c.fail("scheme", {"registered": rec["scheme"], "public": scheme})
    if {p: r["root"] for p, r in reg.items()} != dict(roots):
        c.fail("roots", {"registered": {p: r["root"] for p, r in reg.items()}, "public": dict(roots)})
    if {p: domains.get(p) for p in reg} != {p: r["domain"] for p, r in reg.items()}:
        c.fail("domains", {"registered": {p: r["domain"] for p, r in reg.items()}, "public": dict(domains)})
    if population != rec["population"]:
        c.fail("population", {"registered": rec["population"], "public": population})
    return c.verdict()


#: `OneStageZkStatement.COMMITTED`
COMMITTED = "committed/"


def field_tree_name(v: str) -> bool:
    """`OneStageZkStatement.fieldTreeName`."""
    parts = v.split("/")
    return len(parts) >= 2 and parts[-1].startswith("o") and len(parts[-1]) > 1 and parts[-1][1:].isdigit()


def registered_trees_held(rec: Mapping[str, Any], who: str, registered: Mapping[str, Any] | None,
                          tmpl: str | None = None, linked: Sequence[Mapping[str, Any]] = ()) -> str | None:
    """`OneStageZkStatement.registeredTreesHeld`'s refusal, none where it holds: each port of the registered file of
    ``who`` that opens a committed tree (`portTreeHeld`'s `treePortOf`) held to the record's root, domain and leaf
    count at that tree's port, at the member's template ``tmpl`` (none for the one statement); a port of ``tmpl`` a
    link names (``linked``, `second_stage.round_one_held`) opens the link's value alone, held to round 1's entry of it
    (`LINKED_TREE_WHY`)."""
    by = {r["port"]: r for r in rec.get("roots", []) if isinstance(r.get("port"), str)}
    for name, e in sorted(((registered or {}).get("ports") or {}).items()):
        v = e.get("value", "")
        if (link := next((x for x in linked if tmpl is not None and x["template"] == tmpl and x["port"] == name),
                         None)) is not None:
            if v != link["value"]:
                return linked_tree_why(who, name, link["value"])
            if (why := _held_to(who, name, link["value"], e, link["entry"])) is not None:
                return why
            continue
        if not v.startswith(COMMITTED) and field_tree_name(v) and (tmpl is None or f"{tmpl}/{name}" != v):
            return (f"registered: {who}'s port {name}: the field tree it opens is neither its own output's nor one C "
                    "wires into it at this row")
        t = v[len(COMMITTED):] if v.startswith(COMMITTED) else v if G.is_fresh_value(v) or field_tree_name(v) else None
        if t is None:
            continue
        if (r := by.get(t)) is None:
            return (f"registered: {who}'s port {name} opens the committed tree {t}, and the registration holds no root "
                    f"{t}: a committed tree a statement opens is one its window registered before any coin")
        if (why := _held_to(who, name, t, e, r)) is not None:
            return why
    return None


def linked_tree_why(who: str, port: str, value: str) -> str:
    """`OneStageZkStatement.LINKED_TREE_WHY`."""
    return (f"registered: {who}'s port {port}: a link names it to read round 1's {value}, and it opens another tree: a "
            "linked port is read only at round 1's tree, and the second stage commits no tree of its own for it")


def _held_to(who: str, name: str, t: str, e: Mapping[str, Any], r: Mapping[str, Any]) -> str | None:
    """`OneStageZkStatement.heldTo`: port ``name``'s entry ``e``, opening tree ``t``, at ``r``'s root, domain and leaf
    count, else `TREE_OTHER`'s refusal."""
    for what, k in (("root", "root"), ("domain", "domain"), ("leaf count", "leaves")):
        mine = e.get(k).lower() if k != "leaves" else e.get(k)
        if r.get(k) != mine:
            return (f"registered: {who}'s port {name} opens the committed tree {t} at another {what} than the "
                    "registration's: a statement opens its window's registered tree, never another window's or "
                    "another member's")
    return None


def fresh_reads(registered: Mapping[str, Any] | None) -> list[str]:
    """`OneStageZkStatement.freshReads`: the ports the verifier's registered file reads at a run input node's tree."""
    return [p for p, e in ((registered or {}).get("ports") or {}).items() if G.is_fresh_value(e.get("value"))]


def public_held(rec: Mapping[str, Any], header: Mapping[str, Any], registered: Mapping[str, Any] | None = None) -> Verdict:
    """`OneStageZkStatement.publicHeld`'s comparison: the one statement's public file's ``header`` against the record at
    `public_ports`, each port the registration holds no root of that the verifier's ``registered`` file reads at a run
    input node's tree (`fresh_reads`) left out of its roots and domains. Raises `ValueError` where `public_ports` does."""
    reg = public_ports(rec)
    fv = header.get("frame_v3") or {}
    drop = {p for p in fresh_reads(registered) if p not in reg}
    roots = {p: r for p, r in (fv.get("roots") or {}).items() if p not in drop}
    domains = {p: d for p, d in (fv.get("domain_ids") or {}).items() if p not in drop}
    return matches_public(rec, header.get("scheme"), roots, domains, header.get("instances"), reg=reg)


registered_entries = G.registered_entries

__all__ = ["NOT_ONE_MEMBER", "PARTITION_QUERY", "POOL_KEYS", "RECEIPT", "ROW_TABLES", "TABLE_PORT", "WEIGHTS_UNREGISTERED_WHY", "read_pins",
           "WINDOW_DOMAIN", "check", "fresh_reads", "held_roots", "marked_inputs", "matches_public", "partition_query_refused", "pool_chains", "pool_open", "pool_refused",
           "pool_spent_before", "pool_values", "public_held", "public_ports", "query_digest", "receipt", "registered_trees_held",
           "refuse_unwindowed", "refused", "registered_entries", "unwindowed", "weights_held", "window_domain"]
