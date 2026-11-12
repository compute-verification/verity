"""The registration record (README "The registration record"): what a window is committed to, fixed before the verifier samples.

    {"format": "verity/registration/v1", "protocol": "verity/sampled-proofs/one-stage/v1",
     "program": "<SHA-512 of the program>", "query": "<SHA-512 of the query>", "partition": "<partition digest>",
     "population": <proof blocks the query derives>, "law": {"law": "subset", "k": K} | {"law": "stratified" | "work", "k": K, "strata": [...]},
     "scheme": "<leaf and tree scheme>",
     "roots": [{"port": ..., "root": ..., "leaves": ..., "domain": ...}, ...],
     "leaf_layer": "<SHA-512 of the verifier's public file>" | null,
     "anchors": [], "window": {"kind": "served" | "stand-in", ...provenance}, "previous": null | "<digest>"}

* **Digest.** `SHA-512("verity/registration/v1\\0" || canon(record))`.
* **Rule.** Everything a sampled proof block's statement can depend on is fixed by the record, or by objects the verifier already
  holds, before the sample. The verifier checks R1-R8 (`check`) against its own copies, writes a `receipt` with its own
  clock, and only then samples.
* **The checks** (`check`): the record has exactly these keys, and
  - R1: the format and protocol are these, the sampling design is the verifier's own expanded over its program and query
    (`draw.derive`, with its own work table for `work`) as canonical bytes, and `window.kind` is `served` or `stand-in`;
  - R2: the program, query and partition digests and the population are the verifier's own (an explicit partition
    object must be over the verifier's program), and for a verifier that reads the registration log's declarations
    (Declared day), the program is the one circuit the log declares for the window's day;
  - R3: the scheme is the verifier's, and the roots are exactly the ports its layout derives, each exactly
    `{port, root, leaves, domain}` with a hex root and the verifier's leaf count and domain;
  - R4: the leaf layer is the digest of the verifier's copy, so a null one is refused when the verifier holds one;
  - R5: the anchors are the verifier's;
  - R6: `previous` is the digest of this prover's last record (null for the first), no earlier record has the same
    window, for a bound verifier the window names its stream and prover (Binding), the window names the seed salt
    the verifier read from its challenger, or none when it read none (Seed salt), and it names the reset serving
    chose for its pool, its week and whether the week's reset is spent, as the verifier holds them, at most one reset
    a week, or none of them for a verifier with no pool (Pool chain);
  - R8, under a template query: each run Input node's leaves, other than a parameter's the verifier expects registered,
    are read by one proof block alone, never broadcast to a batch or read by two (`partition.shared_inputs`,
    `R8-shared-input`), so a served port reaches a proof block only as
    its own leaves and a value several read comes as a registered value; and the verdict publishes the query's fan-out F
    on the verifier's program (`partition.fanout`, `Verdict.fanout`), the N at which the capped two-stage form charges a
    replay block (`plancheck.two`'s `fanout`). A run Input a parameter the verifier names `tied` holds is let through:
    the network period check ties its rows to the registrations that computed them, which this check doesn't, so the
    verdict publishes each tied name it let through (`Verdict.rests_on_period_check`), and with any its acceptance is
    none on its own; it refuses a tied name that isn't a run Input of the program's own, or that it expects registered
    or holds (`R8-tied`).
* **Receipt**: `{"format": "verity/registration-receipt/v1", "registration": <digest>, "verifier", "received": "<UTC,
  %Y-%m-%dT%H:%M:%SZ>"}`, and the sample names it.
* **The public file** (`matches_public`): the verifier holds its own (scheme, roots, their domains, its instance count)
  to the record. That they are the proved rows' is `flock-verify`'s check of the public file (`checkPublic`), stated by
  the Lean guarantee `FailClosed.flock_verify_public_ties`; `flock-verify` reads no `scheme` from the header, only META's.
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
  registration log (`verity/registration/declaration/v1`, `verity.hardware.registration`). The verifier passes
  the window's day and the declarations it holds when it samples (its view's `declarations`: `{"day", "program",
  "blocks"}` each): of record, the UTC day of the window open's `started_utc` and the declarations its lane signed
  before that open, both from the challenger's ledger checked under the auditor's key (`challenger.declared_day`), so a
  window that opens before UTC midnight and closes after is a window of its open's day. R2 refuses a day with no
  declaration (`R2-undeclared`), a day declared twice (`R2-declared-twice`, which the log's own check refuses as
  `duplicate-declaration`), and a record whose program is not the day's declared one (`R2-declared`), as R2's Lean does
  (`Spot.Registration.declaredCode`), which `verifier.check` runs beside it.
* **Binding** (`binding`: `{"stream", "prover"}`, the verifier's stream digest and the identity it registered the prover
  under): R6 also holds the record's window to name both (`R6-stream`, `R6-prover`), and a window salt is bound to them,
  so a tile seeded for one prover's stream is refused in any other's, in the same window or not.
* **Seed salt** (`seed_salt`): the window's seed salt, the one coin of the challenger's seed-salt family released after
  the window's open root, before the window serves (`coins.draw_seed_salt`), which the verifier reads from the
  challenger's log (`coins.seed_salt_of`), never from the record: lowercase hex of 64 bytes, read on the window's
  session, so only beside `window_root`. The record's window names it (`SEED_SALT`), a certified input the window's
  circuits read, and R6 refuses a window that names another or none, or names one where the verifier read none
  (`R6-seed-salt`), so a salt the server chose is refused by name.
* **Output values**: a registered value that is an output of an earlier run (`outputs`: `{value: {"record": D, "port":
  P}}`; never the inference-only check's θ, which holds registered public data only) reuses that output's commitment instead of
  committing it afresh: its entry takes port P's domain and root from the `roots` of the log's record with digest D, and
  P's leaf count must be the value's rows (`R7-output` when the log holds no such record or port, or the counts differ).
  Its rows stay under the earlier run's hiding leaves, so nothing about the value is opened, and the reads open against
  that root under that domain (`registered.reads_file`'s `domains`).
* **Reused values**: an output value may name an earlier record's registered value instead of a port (`{value:
  {"record": D, "value": V}}`): its entry takes the domain and root of V's entry in the `registered` of the log's record
  with digest D (its last naming V), whose leaves, words and word bits must be the value's own (`R7-output` when the log
  holds no such record or value, or they differ). Nothing is registered afresh: a record whose entry names another domain
  or root is `R7-domain` or `R7-held`, as for a port.
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
  - by the public rule, which takes no advice: at a registration's first window, at a registration boundary (the log's
    last record registered another program), and after the verifier refused the window `lag` windows of record before
    this one (`pool_refused`), `lag` the fixed public lag at which serving learns a verdict. Whether it did is read
    from the verifier's own verdict log (`verdicts`: each earlier record's digest and whether its run accepted), so
    where every run accepted it never restarts the chain;
  - by serving's own reset (`reset`), which an engine crash needs, at most one a week: the record's window names it
    (`POOL_RESET`, a bool; `R6-pool-reset` where it names another or none). Each window also names its week
    (`POOL_WEEK`, its index among the monitoring period's weeks on the verifier's clock, `week`; `R6-pool-week`) and
    whether the week's reset is spent at it (`POOL_SPENT`; `R6-pool-spent`): that it resets, or that the previous
    record, of the same week, named it spent (`pool_spent_before`), so a week's windows from its reset on all name it
    spent, and the next week starts unspent. R6 refuses a reset where the week's is spent before it
    (`R6-pool-reset-week`). Serving chooses the resets, so they are advice: the records name, of a week's n windows,
    the one serving reset at or none, log2(n + 1) bits a week (log2 1009, 9.98 bits, for 600-second windows), which
    leakage charges. The rest of the chain, the spent marks and the weeks included, is a function of the resets, the
    verdicts and the verifier's clock, and public.
  R7 refuses a record whose entry for `pin` is not the pool its window opens (`pool_open`) by name (`R7-pool-chain`),
  beside the R7 codes the reuse or the fill gives it. The pool's root is all the chain carries. What a window takes up
  that the one before left open at its edge is part of its certified input, which the verifier pins at the root the
  network certifier signed for the window, so a root the certifier didn't sign is `R7-held`, whether the window
  carries the pool or restarts it.
"""
from __future__ import annotations

import hashlib
import re
import time
from typing import Any, Iterable, Mapping, Sequence

from verity.commitments import history as HT
from verity.commitments.window import WINDOW_DOMAIN, window_domain
from verity.circuits.codec import canonical_json
from verity.circuits.program import Program

from . import draw as D, link_points as LP, partition as P, registered as G
from .verdict import Verdict, _Collect

FORMAT = "verity/registration/v1"
FORMAT_V2 = "verity/registration/v2"
PROTOCOL = "verity/sampled-proofs/one-stage/v1"
RECEIPT = "verity/registration-receipt/v1"
KINDS = ("served", "stand-in")
#: the window's field naming its seed salt (Seed salt)
SEED_SALT = "seed_salt"
#: the window's field naming the reset serving chose for its pool (Pool chain)
POOL_RESET = "pool_reset"
#: the window's field naming its week on the verifier's clock (Pool chain)
POOL_WEEK = "pool_week"
#: the window's field naming whether its week's one reset is spent at it (Pool chain)
POOL_SPENT = "pool_spent"
TIED_WHY = ("a tied value is a run Input of the program whose rows a later check ties to the registrations that computed "
            "them: a parameter of the program, neither registered nor held, whose gates are one interval of run Input nodes")
SHARED_WHY = ("a run Input node whose leaves two proof blocks read, or a batch broadcasts, could hold rows one of them "
              "computed, which the fan-out can't see: give each proof block its own leaves, or register the value")
_KEYS = {"format", "protocol", "program", "query", "partition", "population", "law", "scheme", "roots", "leaf_layer",
         "anchors", "window", "previous"}
_ROOT_KEYS = {"port", "root", "leaves", "domain"}
_HEX = re.compile(r"[0-9a-f]+")


def query_digest(query: Mapping[str, Any]) -> str:
    return hashlib.sha512(canonical_json(dict(query))).hexdigest()


def record(program: Program, query: Mapping[str, Any], law: Mapping[str, Any], scheme: str, roots: Sequence[Mapping[str, Any]],
           window: Mapping[str, Any], leaf_layer: str | None = None, anchors: Sequence[Any] = (),
           previous: str | None = None, partition: Mapping[str, Any] | None = None,
           work_table: Mapping[str, Any] | None = None, registered: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """`partition`: an explicit partition object over `program` (default: the canonical `P.partition(program, query)`).
    `law` is written expanded over the program and query (`draw.derive`: a `stratified` rule becomes its strata, and a
    `work` rule its strata sized by `work_table`). `registered`: the registered values' entries (`Registered.entry`),
    which make the record v2."""
    part = dict(partition) if partition is not None else P.partition(program, query)
    rec = {"format": FORMAT_V2 if registered else FORMAT, "protocol": PROTOCOL, "program": part["program"],
           "query": query_digest(part["query"]), "partition": P.digest(part), "population": P.units(program, query),
           "law": D.derive(program, query, law, work_table), "scheme": scheme,
           "roots": [{"port": r["port"], "root": r["root"], "leaves": int(r["leaves"]), "domain": r["domain"]} for r in roots],
           "leaf_layer": leaf_layer, "anchors": list(anchors), "window": dict(window), "previous": previous}
    if registered:
        rec["registered"] = [{**{k: (int(e[k]) if k in ("leaves", "words") else e[k]) for k in G.ENTRY_KEYS},
                              **G.width_entry(int(e.get(G.WIDTH_KEY, G.WORD_BITS)))} for e in registered]
    return rec


def record_digest(tag: str, canon: bytes) -> bytes:
    """The one registration digest, `SHA-512(tag || 0 || canon)`, over a record's canonical bytes: `digest`'s, and PoUW's
    over the bytes it was handed (`pouw.core_check.digest`)."""
    return hashlib.sha512(tag.encode() + b"\0" + canon).digest()


def digest(rec: Mapping[str, Any]) -> str:
    """`SHA-512(format || 0 || canon(record))`, the record's own format (v1 or v2) as the tag."""
    tag = FORMAT_V2 if rec.get("format") == FORMAT_V2 else FORMAT
    return record_digest(tag, canonical_json(dict(rec))).hex()


def held_roots(log: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], str]:
    """`{(value, domain): root}` for every value an earlier record of the log registered, the root of its first
    registration under that domain: R7 holds a value to the root registered under the domain it takes now, so a
    per-window value, whose domain binds its window, is held to none of another window's."""
    held: dict[tuple[str, str], str] = {}
    for rec in log:
        for e in rec.get("registered") or ():
            held.setdefault((e["value"], e["domain"]), e["root"])
    return held


def receipt(rec: Mapping[str, Any], verifier: str, received: str | None = None) -> dict[str, Any]:
    """The verifier's own log entry for a record it accepted, with its own clock; the sample names it."""
    return {"format": RECEIPT, "registration": digest(rec), "verifier": verifier,
            "received": received or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


HISTORY = "R7-history"
"""The verifier's code for a registration whose history tables both its verifiers refuse (`verifier.check`)."""


class HistoryRefused(ValueError):
    """History tables `check` can't read, or whose roots it can't hold to the ones the window registers."""


def check(rec: Mapping[str, Any], program: Program, query: Mapping[str, Any], law: Mapping[str, Any], scheme: str,
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
          history: Mapping[str, Any] | None = None,
          certified: Mapping[str, Mapping[str, Any]] | None = None) -> Verdict:
    """R1-R8 against the verifier's own copies: its program and query, its sampling design and scheme, the ports its layout derives
    (`{port: {"leaves", "domain"}}`), the digest of its copy of the leaf layer, its anchors, its log of this prover's
    earlier records, the parameters it expects registered (`registered`: `{parameter: words}`), of those the ones it
    fills itself (`supplied`: `{parameter: [[source, ...], ...]}`, `registered.sources`) and the roots it held before the
    record for others (`pinned`: `{parameter: root}`, the weights' root recorded before the salt, say), and of the rest the
    ones that are outputs of an earlier run or reuse an earlier record's registered value (`outputs`: `{parameter:
    {"record", "port" | "value"}}`, or `{parameter: {"port"}}`, that port of the log's last record); and
    those whose rows are not `row/v1` rows (`schemas`: `{parameter: schema}`, `registered.SCHEMAS`), whose rows R7 counts in
    that format (`registered_entries`), and a supplied one's it commits in it (`registered.supplied_root`); ValueError without `registered`, or in another form.  `binding`: the stream and prover the verifier registers this record for
    (`{"stream", "prover"}`, `_binding`), which R6 holds the window to name and a window salt binds.  `partition`: the
    verifier's own partition object when it isn't the canonical one (a record made before the canonical envelope); its
    program must still be `program`'s, and the population is `query`'s.  R1 holds the registered sampling design to the verifier's own
    sampling design expanded over its program and query (`draw.derive`): for `stratified`, the strata by template descriptor id and
    each stratum's k by the rule; for `work`, also its own `work_table`.  R7 holds the registered values to the verifier's
    own (`registered_entries`), a pinned value to its pinned root, and a value an earlier record of the log registered
    under the domain it takes now to that record's root (`held_roots`): a value is committed once a domain; a supplied
    value's root is the one it recomputes from this record and the log (`registered.supplied_root`) instead, and an output
    value's domain and root are its earlier record's port's, or its earlier record's registered value's (`R7-output`, in
    the values' sorted order before the other R7 codes).  `window_root`: the window's open root R, which the verifier reads from the challenger's signed open of this
    record's window (never from the record), and which every registered value's domain binds but the ones it holds
    across windows: `held` (registered values, the weights, say), and its pinned, supplied and output values
    (`registered.domain`).  So a per-window value registered in one window is held to nothing in the next, and a root of
    another window's opens under no domain R7 takes.  Without `window_root`, a value it doesn't hold across windows that
    the log registered under the domain it takes now is a ValueError (`unwindowed`).  `day` and `declarations`: the
    window's day and the registration log's declarations (Declared day), both or neither; R2 holds the record's program
    to the day's one declaration (`_declared`).  `seed_salt`: the window's seed salt the verifier read from its
    challenger (Seed salt), which R6 holds the window to name; ValueError in another form or without `window_root`.
    `pool`: the values a window opens and leaves its pool in, the empty pool's root, whether the verifier refused the
    window the public lag back, the window's own reset, which R6 holds it to name, and whether its week's one reset is
    spent (Pool chain, `_pool`), whose `pin` it adds to `outputs` or `supplied` (`pool_values`) and R7 holds to the
    chain (`R7-pool-chain`).  `history`: the history tables the program reads (``{"digests", "held", "registered",
    "tables"}``, `history.roots_of`), each table's root computed from its writers' public bodies the leaf layer binds
    and held to the root the window registers for it, which needs `leaf_layer`; a supplied value's source
    ``{"history": "H/root"}`` or ``{"history": "H/domain"}`` is then the bytes of the root or domain id it computed for
    table ``H`` (`history.resolve`).  History tables it can't read or hold to their roots raise `HistoryRefused`, which
    the verifier reports as `HISTORY` where its Lean refuses them too.  `certified`: the entries the verifier built from
    registrations it read under the deployment's key, of values that are none of its program's parameters (the
    certified rows of a link), ``{value: {"root", "leaves", "words", "domain"}}``, with `word_bits` when its leaves
    are not 16 bits (`_certified`).  With any, it expects a v2 record though it expects no parameter registered, R7
    holds the record's registered values to be the ones it expects registered and the certified ones, each once, a
    certified value's entry to the verifier's leaves, words, word bits and domain as its own values', and its root to
    the verifier's (`R7-certified`): a certified value is never supplied, pinned, held or an output, and a record that
    names one the verifier holds no entry for is refused as before (`R7-values`).  R3
    reads `ports` by name, so its codes come in the ports' sorted order whatever order the mapping has, as the Lean port
    reads them.  Under a template query R8 refuses a run Input node the proof blocks don't each read alone, unless it
    is a parameter's that `registered` names (`partition.shared_inputs`) or one of `tied`'s, and the verdict carries the
    fan-out F of `query` on `program` (`partition.fanout`), which it publishes whatever its codes; under any other query it
    carries none.  `tied`: parameters, run Inputs, whose rows a later check ties to the registrations that computed them
    (the scan's slot entries, which the network period check ties to the blocks that registered them), which R8 lets
    through though several proof blocks read them, on that later check alone: the verdict publishes the ones it let
    through (`Verdict.rests_on_period_check`), and refuses a name that isn't a parameter of `program`, one `registered`
    or `held` names, or one whose gates aren't one interval of run Input nodes (`R8-tied`).  `members`: the per-member
    form's members, each its units (`[lo, hi)` ranges) and its circuit's units a proof block; after R8, `link-points`
    refuses the record when a member's fewest and most sampled units give different link points
    (`link_points.refused`), no sample under the sampling design ends (`link_points.NEVER_ENDS`), or the members or the
    sampling design are not well formed over the population."""
    c = _Collect()
    days = _declarations(day, declarations)
    log = list(log)
    pool = _pool(pool, registered or {}, supplied, pinned, outputs, schemas)
    supplied, outputs, schemas = pool_values(pool, program, supplied, outputs, schemas, log)
    if history is not None:
        if not isinstance(history, Mapping) or set(history) != {"digests", "held", "registered", "tables"}:
            raise HistoryRefused("the keys of the history tables are not digests, held, registered, tables")
        if leaf_layer is None:
            raise HistoryRefused("history tables are read from members' public bodies the leaf layer binds: the verifier "
                                 "names no leaf_layer")
        try:
            supplied = HT.resolve(HT.roots_of(leaf_layer, history), supplied)
        except ValueError as e:
            raise HistoryRefused(str(e)) from e
    sup = G.sources(supplied if supplied is not None else {})
    if supplied is not None and not registered:
        raise ValueError("supplied values are registered values: the verifier expects none registered")
    cert = _certified(certified, registered or {}, sup)
    v2 = bool(registered) or bool(cert)
    pins = _pins(pinned, registered or {}, sup)
    outs = _outputs(outputs, registered or {}, sup, pins, log)
    formats = _schemas(schemas, registered or {})
    bound = _binding(binding)
    window, fixed = _window(window_root, held, registered or {}, sup, pins, outs)
    salt = _seed_salt(seed_salt, window)
    refuse_unwindowed(unwindowed(program, registered or {}, formats, window, fixed, log))
    fan, shared, through, bad = _readers(program, query, registered or {}, held or (), tied or ())
    keys = _KEYS | {"registered"} if v2 else _KEYS
    if not isinstance(rec, Mapping) or set(rec) != keys:
        c.fail("keys", sorted(set(rec) ^ keys) if isinstance(rec, Mapping) else rec)
        return c.verdict(fanout=fan, rests_on_period_check=through)
    # R1
    if (rec["format"], rec["protocol"]) != (FORMAT_V2 if v2 else FORMAT, PROTOCOL):
        c.fail("R1-protocol", [rec["format"], rec["protocol"]])
    try:
        mine = D.derive(program, query, law, work_table)
    except (P.QueryRefused, ValueError) as e:
        c.fail("R1-law", {"registered": rec["law"], "verifier": f"{dict(law)} does not apply: {e}"})
    else:
        if not D.same_design(rec["law"], mine):
            c.fail("R1-law", {"registered": rec["law"], "verifier": mine})
    if not isinstance(rec["window"], Mapping) or rec["window"].get("kind") not in KINDS:
        c.fail("window", rec["window"])
    # R2
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
    if days is not None:
        _declared(c, rec["program"], *days)
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
    if rec["previous"] != (digest(earlier[-1]) if earlier else None):
        c.fail("R6-previous", rec["previous"])
    if any(e.get("window") == rec["window"] for e in earlier):
        c.fail("R6-window", rec["window"])
    w = rec["window"] if isinstance(rec["window"], Mapping) else {}
    if bound is not None:
        for k in ("stream", "prover"):
            if w.get(k) != bound[k]:
                c.fail(f"R6-{k}", {"registered": w.get(k), "verifier": bound[k]})
    if w.get(SEED_SALT) != salt:
        c.fail("R6-seed-salt", {"registered": w.get(SEED_SALT), "verifier": salt})
    reset = pool["reset"] if pool is not None else None
    if w.get(POOL_RESET) is not reset:
        c.fail("R6-pool-reset", {"registered": w.get(POOL_RESET), "verifier": reset})
    week = pool["week"] if pool is not None else None
    if not _names(w.get(POOL_WEEK), week):
        c.fail("R6-pool-week", {"registered": w.get(POOL_WEEK), "verifier": week})
    before = pool is not None and pool_spent_before(pool, earlier)
    spent = (pool["reset"] or before) if pool is not None else None
    if w.get(POOL_SPENT) is not spent:
        c.fail("R6-pool-spent", {"registered": w.get(POOL_SPENT), "verifier": spent})
    if pool is not None and pool["reset"] and before:
        c.fail("R6-pool-reset-week", {"registered": True, "verifier": "the week's reset is spent before this window"})
    # R7
    if v2:
        _check_registered(c, rec["registered"], program, registered or {}, pins, rec, sup, earlier, bound, outs, formats,
                          window, fixed, cert)
        if pool is not None:
            _pool_chain(c, rec, pool, pool_open(pool, rec, program, registered, formats, window, fixed, sup, earlier,
                                                bound))
        _certified_held(c, rec, cert)
    # R8
    if bad:
        c.fail("R8-tied", {"tied": bad, "why": TIED_WHY})
    if shared:
        c.fail("R8-shared-input", {"inputs": shared, "why": SHARED_WHY})
    if members:
        try:
            refused = LP.refused(*LP.law_of(program, query, law, work_table), members)
        except (P.QueryRefused, ValueError) as e:
            c.fail("link-points", str(e))
        else:
            if refused:
                c.fail("link-points", {"members": refused})
    return c.verdict(fanout=fan, rests_on_period_check=through)


def _readers(program: Program, query: Mapping[str, Any], registered: Iterable[str], held: Iterable[str],
             tied: Sequence[str]) -> tuple[int | None, list[int], tuple[str, ...], list[str]]:
    """R8's reading of `program`: under a template query, its fan-out, the run Input nodes its proof blocks don't each
    read alone (`partition.fanout`, `partition.shared_inputs`) outside the gates of the parameters the verifier expects
    `registered` (registered values) and of the `tied` ones it may let through (`_tied`), and the tied names whose Input
    holds such a node, which it let through; and, under any query, the tied names it can't let through. `(None, [], ())`
    and those for any other query, or one that doesn't apply. A function of the program, query and verifier alone, so a
    verdict publishes F and the tied names it let through whatever the record."""
    registered = list(registered)
    gates = _tied(program, tied, registered, held)
    bad = [t for t, g in zip(tied, gates) if g is None]
    if not P.is_template_query(query):
        return None, [], (), bad
    try:
        fan, own = P.fanout(program, query), P.shared_inputs(program, query, G.held_gates(program, registered))
    except P.QueryRefused:
        return None, [], (), bad
    through = tuple(t for t, g in zip(tied, gates) if g is not None and any(P.within(program, g, k) for k in own))
    return fan, [k for k in own if not any(g is not None and P.within(program, g, k) for g in gates)], through, bad


def _tied(program: Program, tied: Sequence[str], registered: Sequence[str], held: Iterable[str]) -> list[range | None]:
    """Each of `tied`'s Input gates where R8 may let it through, else None: a parameter of the program that the verifier
    neither expects registered nor holds, whose gates are one non-empty interval (`registered.held_gates`) of run Input
    nodes (`partition.run_inputs`). A tied Input is never a registered value: R7 holds nothing of it. The Lean's
    `Registration.tiedGate` is this."""
    fixed = {*registered, *held}
    out: list[range | None] = []
    for t in tied:
        g = None if t in fixed else next(iter(G.held_gates(program, [t])), None)
        out.append(g if g is not None and P.run_inputs(program, g) else None)
    return out


def _window(window_root: str | None, held: Sequence[str] | None, registered: Mapping[str, int],
            supplied: Mapping[str, Any], pinned: Mapping[str, str],
            outputs: Mapping[str, tuple[str, str, str]]) -> tuple[str | None, tuple[str, ...]]:
    """The window's open root (`registered.window_root`) and the values whose domain leaves it out: `held`, a list of
    values it expects registered, then its pinned, supplied and output values; ValueError otherwise, which the Lean
    verifier refuses too (`OneStage.paramsOfJson`)."""
    if held is not None and not (isinstance(held, (list, tuple)) and all(isinstance(v, str) and v in registered
                                                                          for v in held)):
        raise ValueError(f"held values are a list of registered values: {held!r}")
    return G.window_root(window_root), (*(held or ()), *pinned, *supplied, *outputs)


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
    record, when the log is empty (`Registration.lastDigest`)."""
    if not log:
        return ""
    try:
        return digest(log[-1])
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
    """The verifier's row formats, each of a value it expects registered, one of `registered.SCHEMAS`: a supplied value's
    is the format R7 commits its rows in (`registered.supplied_root`); ValueError otherwise, which the Lean verifier
    refuses too (`OneStage.paramsOfJson`)."""
    if schemas is None:
        return {}
    if not isinstance(schemas, Mapping) or not registered:
        raise ValueError(f"row formats are an object of registered values' formats: {schemas!r}")
    for v, s in schemas.items():
        if v not in registered or not isinstance(s, str) or s not in G.SCHEMAS:
            raise ValueError(f"row format of {v!r}: not a registered value's, in one of {G.SCHEMAS}: {s!r}")
    return dict(schemas)


def _output_port(log: Sequence[Mapping[str, Any]], record: str, port: str, key: str = "roots",
                 name: str = "port") -> Mapping[str, Any] | None:
    """The `roots` entry for `port` of the log's record with digest `record` (its last, as a dict reads it), or None; with
    `key="registered", name="value"`, its `registered` entry for the value `port`."""
    for r in log:
        try:
            d = digest(r)
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
    count, or (`kind == "value"`) its registered value `name`, with the value's leaves, words and word bits; each with a
    string domain and root, else None."""
    if kind == "port":
        e = _output_port(log, record, name)
        ok = e is not None and type(e.get("leaves")) is int and e["leaves"] == want["leaves"]
    else:
        e = _output_port(log, record, name, "registered", "value")
        width = e.get(G.WIDTH_KEY, G.WORD_BITS) if e is not None else None
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
"""A certified entry's keys, with `registered.WIDTH_KEY` for leaves that are not 16 bits (`_certified`)."""


def _certified(certified: Mapping[str, Mapping[str, Any]] | None, registered: Mapping[str, int],
               supplied: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The verifier's certified entries, `{value: {"root", "leaves", "words", "word_bits", "domain"}}` in the values'
    sorted order: each of a value it neither expects a parameter registered under (`registered`) nor fills itself
    (`supplied`), an object of exactly `CERTIFIED_KEYS` (`root` lowercase hex of 64 bytes, `leaves` and `words`
    naturals, `domain` a string) and, optionally, `word_bits`, one of `registered.WIDTHS`, 16 when left out; ValueError
    otherwise, which the Lean verifier refuses too (`OneStage.paramsOfJson`)."""
    if certified is None:
        return {}
    if not isinstance(certified, Mapping) or not all(isinstance(v, str) for v in certified):
        raise ValueError(f"certified entries are an object of values' entries: {certified!r}")
    keys, out = set(CERTIFIED_KEYS), {}
    for v, e in sorted(certified.items()):
        width = e.get(G.WIDTH_KEY, G.WORD_BITS) if isinstance(e, Mapping) else None
        if (v in registered or v in supplied or not isinstance(e, Mapping) or not keys <= set(e) <= keys | {G.WIDTH_KEY}
                or not (isinstance(e["root"], str) and len(e["root"]) == 128 and _HEX.fullmatch(e["root"]))
                or not all(type(e[k]) is int and e[k] >= 0 for k in ("leaves", "words"))
                or not isinstance(e["domain"], str) or type(width) is not int or width not in G.WIDTHS):
            raise ValueError(f"certified value {v!r}: one it expects a parameter registered under or fills, or an entry "
                             f"that is not a root of 64 bytes, its leaves, words and domain, and word_bits when not "
                             f"{G.WORD_BITS}: {e!r}")
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


def _declared(c: _Collect, program: Any, day: int, declarations: Sequence[Mapping[str, Any]]) -> None:
    """R2's declared day: the window's day has exactly one declaration, and the record's program is its program."""
    mine = [d for d in declarations if d["day"] == day]
    if not mine:
        c.fail("R2-undeclared", {"day": day})
    elif len(mine) > 1:
        c.fail("R2-declared-twice", {"day": day, "programs": [d["program"] for d in mine]})
    elif program != mine[0]["program"]:
        c.fail("R2-declared", {"day": day, "registered": program, "declared": mine[0]["program"]})


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
    verdicts = pool["verdicts"]
    if not (isinstance(verdicts, Mapping) and all(isinstance(d, str) and len(d) == 128 and _HEX.fullmatch(d)
                                                  and type(a) is bool for d, a in verdicts.items())):
        raise ValueError(f"the pool's verdicts are an object of record digests (lowercase hex of 64 bytes) to bools: "
                         f"{verdicts!r}")
    return dict(pool)


def pool_refused(pool: Mapping[str, Any], log: Sequence[Mapping[str, Any]]) -> bool:
    """Whether the verifier refused the window `lag` windows of record before this one (Pool chain): its verdict log
    (`verdicts`) names its run on that record of the log refused. A log shorter than the lag, or a record it holds no
    verdict for, is not refused, so where every run accepted the chain never restarts (`Registration.Pool.refused`)."""
    lag = pool["lag"]
    if len(log) < lag:
        return False
    try:
        d = digest(log[len(log) - lag])
    except (TypeError, ValueError):
        return False
    return pool["verdicts"].get(d) is False


def pool_spent_before(pool: Mapping[str, Any], log: Sequence[Mapping[str, Any]]) -> bool:
    """Whether the week's one reset is spent before this window (Pool chain): the log's last record, the previous window
    of record, names it spent (`POOL_SPENT`) in the verifier's week (`POOL_WEEK`), so a week starts unspent
    (`Registration.Pool.spentBefore`)."""
    last = log[-1] if log else None
    w = last.get("window") if isinstance(last, Mapping) else None
    return isinstance(w, Mapping) and w.get(POOL_SPENT) is True and _names(w.get(POOL_WEEK), pool["week"])


def pool_chains(pool: Mapping[str, Any], log: Sequence[Mapping[str, Any]], program: Program) -> bool:
    """Whether the window carries the previous window's pool (Pool chain): neither the verifier refused the window the
    public lag back (`pool_refused`) nor serving reset it (`reset`), and the log's last record, the previous window of
    record, registered its program (`Registration.Pool.chains`). Otherwise it opens the empty pool."""
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
    (`registered.supplied_root`); where there is none, why (`Registration.poolOpen`, none there)."""
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


def _pool_chain(c: _Collect, rec: Mapping[str, Any], pool: Mapping[str, Any], want: tuple[str, str] | str) -> None:
    """R7's pool chain: the record's entry for `pin` (its last naming it) has the domain and root of the pool its window
    opens (`want`, `pool_open`, or why there is none), else `R7-pool-chain`."""
    got = rec["registered"]
    hits = [e for e in got if isinstance(e, Mapping) and e.get("value") == pool["pin"]] if isinstance(got, list) else []
    mine = (hits[-1].get("domain"), hits[-1].get("root")) if hits else None
    if mine != want:
        c.fail("R7-pool-chain", {"value": pool["pin"], "registered": mine, "verifier": want})


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
                      fixed: Sequence[str] = (), certified: Mapping[str, Mapping[str, Any]] | None = None) -> None:
    try:
        mine = G.registered_entries(program, values, schemas, window, fixed)
    except ValueError as e:
        c.fail("R7-registered", {"verifier": f"{dict(values)} does not apply: {e}"})
        return
    logged = held_roots(log)
    held = {v: logged[(v, m["domain"])] for v, m in mine.items() if (v, m["domain"]) in logged}
    held.update(pinned)
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
    keys = set(G.ENTRY_KEYS)
    if not isinstance(got, list) or not all(isinstance(e, Mapping) and keys <= set(e) <= keys | {G.WIDTH_KEY} for e in got):
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
        # digest per registration
        for k in ("leaves", "words", "domain"):
            if e[k] != want[k] or (k != "domain" and type(e[k]) is not int):
                c.fail(f"R7-{k}", {"value": v, "registered": e[k], "verifier": want[k]})
        width = e.get(G.WIDTH_KEY, G.WORD_BITS)
        if type(width) is not int or width != want["word_bits"] or e.get(G.WIDTH_KEY) == G.WORD_BITS:
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
                                   (schemas or {}).get(v, G.SCHEMA))
        except ValueError as err:
            c.fail("R7-supplied", {"value": v, "verifier": str(err)})
            continue
        if e["root"] != root:
            c.fail("R7-supplied", {"value": v, "registered": e["root"], "verifier": root})


def matches_public(rec: Mapping[str, Any], scheme: str, roots: Mapping[str, str], domains: Mapping[str, str],
                   population: int) -> Verdict:
    """The verifier's own public file (its scheme, roots, their domains and its instance count) against the record."""
    c = _Collect()
    reg = {r["port"]: r for r in rec["roots"]}
    if scheme != rec["scheme"]:
        c.fail("scheme", {"registered": rec["scheme"], "public": scheme})
    if {p: r["root"] for p, r in reg.items()} != dict(roots):
        c.fail("roots", {"registered": {p: r["root"] for p, r in reg.items()}, "public": dict(roots)})
    if {p: domains.get(p) for p in reg} != {p: r["domain"] for p, r in reg.items()}:
        c.fail("domains", {"registered": {p: r["domain"] for p, r in reg.items()}, "public": dict(domains)})
    if population != rec["population"]:
        c.fail("population", {"registered": rec["population"], "public": population})
    return c.verdict()


registered_entries = G.registered_entries

__all__ = ["FORMAT", "FORMAT_V2", "KINDS", "POOL_KEYS", "PROTOCOL", "RECEIPT", "WINDOW_DOMAIN", "check", "digest",
           "held_roots", "matches_public", "pool_chains", "pool_open", "pool_refused", "pool_spent_before",
           "pool_values", "query_digest", "receipt", "record", "record_digest", "refuse_unwindowed",
           "registered_entries", "unwindowed", "window_domain"]
