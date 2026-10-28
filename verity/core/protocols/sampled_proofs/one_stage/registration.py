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
  holds, before the sample. The verifier checks R1-R6 (`check`) against its own copies, writes a `receipt` with its own
  clock, and only then draws.
* **The checks** (`check`): the record has exactly these keys, and
  - R1: the format and protocol are these, the sampling design is the verifier's own expanded over its program and query
    (`draw.derive`, with its own work table for `work`) as canonical bytes, and `window.kind` is `served` or `stand-in`;
  - R2: the program, query and partition digests and the population are the verifier's own (an explicit partition
    object must be over the verifier's program);
  - R3: the scheme is the verifier's, and the roots are exactly the ports its layout derives, each exactly
    `{port, root, leaves, domain}` with a hex root and the verifier's leaf count and domain;
  - R4: the leaf layer is the digest of the verifier's copy, so a null one is refused when the verifier holds one;
  - R5: the anchors are the verifier's;
  - R6: `previous` is the digest of this prover's last record (null for the first), no earlier record has the same
    window, and for a bound verifier the window names its stream and prover (Binding).
* **Receipt**: `{"format": "verity/registration-receipt/v1", "registration": <digest>, "verifier", "received": "<UTC,
  %Y-%m-%dT%H:%M:%SZ>"}`, and the sample names it.
* **The public file** (`matches_public`): the verifier holds its own (scheme, roots, their domains, its instance count)
  to the record. That they are the proved rows' is flock-verify's check of the public file (`checkPublic`), stated by
  the Lean guarantee `FailClosed.flock_verify_public_ties`; flock-verify reads no `scheme` from the header, only META's.
* **`window`** is provenance, apart from the binding it names (Binding). `kind` says whether the roots are what the integration
  committed as it ran (`served`) or a fresh commitment to values captured from its run (`stand-in`, whose link to the run's
  root is provenance, not proof).
* **Window domains** (`window_domain`, `verity.ml.commitments.window`'s): each port's frame domain binds the program, the
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
* **Binding** (`binding`: `{"stream", "prover"}`, the verifier's stream digest and the identity it registered the prover
  under): R6 also holds the record's window to name both (`R6-stream`, `R6-prover`), and a window salt is bound to them,
  so a tile seeded for one prover's stream is refused in any other's, in the same window or not.
* **Output values**: a registered value that is an output of an earlier run (`outputs`: `{value: {"record": D, "port":
  P}}`; never NCI's baseline θ, which holds registered public data only) reuses that output's commitment instead of
  committing it afresh: its entry takes port P's domain and root from the `roots` of the log's record with digest D, and
  P's leaf count must be the value's rows (`R7-output` when the log holds no such record or port, or the counts differ).
  Its rows stay under the earlier run's hiding leaves, so nothing about the value is opened, and the reads open against
  that root under that domain (`registered.reads_file`'s `domains`).
* **Reused values**: an output value may name an earlier record's registered value instead of a port (`{value:
  {"record": D, "value": V}}`): its entry takes the domain and root of V's entry in the `registered` of the log's record
  with digest D (its last naming V), whose leaves, words and word bits must be the value's own (`R7-output` when the log
  holds no such record or value, or they differ). Nothing is registered afresh: a record whose entry names another domain
  or root is `R7-domain` or `R7-held`, as for a port.
"""
from __future__ import annotations

import hashlib
import re
import time
from typing import Any, Iterable, Mapping, Sequence

from verity.ml.commitments.window import WINDOW_DOMAIN, window_domain
from verity.ml.circuits.codec import canonical_json
from verity.ml.circuits.program import Program

from . import draw as D, partition as P, registered as G
from .verdict import Verdict, _Collect

FORMAT = "verity/registration/v1"
FORMAT_V2 = "verity/registration/v2"
PROTOCOL = "verity/sampled-proofs/one-stage/v1"
RECEIPT = "verity/registration-receipt/v1"
KINDS = ("served", "stand-in")
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


def digest(rec: Mapping[str, Any]) -> str:
    """`SHA-512(format || 0 || canon(record))`, the record's own format (v1 or v2) as the tag."""
    tag = FORMAT_V2 if rec.get("format") == FORMAT_V2 else FORMAT
    return hashlib.sha512(tag.encode() + b"\0" + canonical_json(dict(rec))).hexdigest()


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


def check(rec: Mapping[str, Any], program: Program, query: Mapping[str, Any], law: Mapping[str, Any], scheme: str,
          ports: Mapping[str, Mapping[str, Any]], leaf_layer: str | None = None, anchors: Sequence[Any] = (),
          log: Iterable[Mapping[str, Any]] = (), partition: Mapping[str, Any] | None = None,
          work_table: Mapping[str, Any] | None = None, registered: Mapping[str, int] | None = None,
          supplied: Mapping[str, Any] | None = None, pinned: Mapping[str, str] | None = None,
          binding: Mapping[str, str] | None = None,
          outputs: Mapping[str, Mapping[str, str]] | None = None,
          schemas: Mapping[str, str] | None = None, window_root: str | None = None,
          held: Sequence[str] | None = None) -> Verdict:
    """R1-R7 against the verifier's own copies: its program and query, its sampling design and scheme, the ports its layout derives
    (`{port: {"leaves", "domain"}}`), the digest of its copy of the leaf layer, its anchors, its log of this prover's
    earlier records, the parameters it expects registered (`registered`: `{parameter: words}`), of those the ones it
    fills itself (`supplied`: `{parameter: [[source, ...], ...]}`, `registered.sources`) and the roots it held before the
    record for others (`pinned`: `{parameter: root}`, the weights' root recorded before the salt, say), and of the rest the
    ones that are outputs of an earlier run or reuse an earlier record's registered value (`outputs`: `{parameter:
    {"record", "port" | "value"}}`); and of the ones it doesn't fill,
    those whose rows are not `row/v1` rows (`schemas`: `{parameter: schema}`, `registered.SCHEMAS`), whose rows R7 counts in
    that format (`registered_entries`); ValueError without `registered`, or in another form.  `binding`: the stream and prover the verifier registers this record for
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
    the log registered under the domain it takes now is a ValueError (`unwindowed`).  R3 reads `ports` by name, so its codes come in the ports' sorted
    order whatever order the mapping has, as the Lean port reads them."""
    c = _Collect()
    v2 = bool(registered)
    sup = G.sources(supplied if supplied is not None else {})
    if supplied is not None and not v2:
        raise ValueError("supplied values are registered values: the verifier expects none registered")
    pins = _pins(pinned, registered or {}, sup)
    outs = _outputs(outputs, registered or {}, sup, pins)
    formats = _schemas(schemas, registered or {}, sup)
    bound = _binding(binding)
    window, fixed = _window(window_root, held, registered or {}, sup, pins, outs)
    log = list(log)
    refuse_unwindowed(unwindowed(program, registered or {}, formats, window, fixed, log))
    keys = _KEYS | {"registered"} if v2 else _KEYS
    if not isinstance(rec, Mapping) or set(rec) != keys:
        c.fail("keys", sorted(set(rec) ^ keys) if isinstance(rec, Mapping) else rec)
        return c.verdict()
    # R1
    if (rec["format"], rec["protocol"]) != (FORMAT_V2 if v2 else FORMAT, PROTOCOL):
        c.fail("R1-protocol", [rec["format"], rec["protocol"]])
    try:
        mine = D.derive(program, query, law, work_table)
    except (P.QueryRefused, ValueError) as e:
        c.fail("R1-law", {"registered": rec["law"], "verifier": f"{dict(law)} does not apply: {e}"})
    else:
        if not D.same_law(rec["law"], mine):
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
    if bound is not None:
        w = rec["window"] if isinstance(rec["window"], Mapping) else {}
        for k in ("stream", "prover"):
            if w.get(k) != bound[k]:
                c.fail(f"R6-{k}", {"registered": w.get(k), "verifier": bound[k]})
    # R7
    if v2:
        _check_registered(c, rec["registered"], program, registered, pins, rec, sup, earlier, bound, outs, formats, window,
                          fixed)
    return c.verdict()


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


def _outputs(outputs: Mapping[str, Mapping[str, str]] | None, registered: Mapping[str, int], supplied: Mapping[str, Any],
             pinned: Mapping[str, str]) -> dict[str, tuple[str, str, str]]:
    """The verifier's output values, `{value: (record digest, "port" | "value", name)}`, each of a value it expects
    registered and neither fills nor pinned, an object of exactly a `record` (lowercase hex of 64 bytes) and either a `port`
    or a `value` (a string: the earlier record's port, or its registered value); ValueError otherwise, which the Lean
    verifier refuses too (`OneStage.paramsOfJson`)."""
    if outputs is None:
        return {}
    if not isinstance(outputs, Mapping) or not registered:
        raise ValueError(f"output values are an object of registered values' earlier ports or values: {outputs!r}")
    out = {}
    for v, o in outputs.items():
        kind = next((k for k in ("port", "value") if isinstance(o, Mapping) and set(o) == {k, "record"}), None)
        if (v not in registered or v in supplied or v in pinned or kind is None or not isinstance(o[kind], str)
                or not (isinstance(o["record"], str) and len(o["record"]) == 128 and _HEX.fullmatch(o["record"]))):
            raise ValueError(f"output value {v!r}: not a registered value it neither fills nor pinned, with an earlier "
                             f"record's digest and a port or a value: {o!r}")
        out[v] = (o["record"], kind, o[kind])
    return out


def _schemas(schemas: Mapping[str, str] | None, registered: Mapping[str, int], supplied: Mapping[str, Any]) -> dict[str, str]:
    """The verifier's row formats, each of a value it expects registered and doesn't fill, one of `registered.SCHEMAS`;
    ValueError otherwise, which the Lean verifier refuses too (`OneStage.paramsOfJson`)."""
    if schemas is None:
        return {}
    if not isinstance(schemas, Mapping) or not registered:
        raise ValueError(f"row formats are an object of registered values' formats: {schemas!r}")
    for v, s in schemas.items():
        if v not in registered or v in supplied or not isinstance(s, str) or s not in G.SCHEMAS:
            raise ValueError(f"row format of {v!r}: not a registered value it doesn't fill, in one of {G.SCHEMAS}: {s!r}")
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


def _check_registered(c: _Collect, got: Any, program: Program, values: Mapping[str, int], pinned: Mapping[str, str],
                      rec: Mapping[str, Any], supplied: Mapping[str, Any], log: Sequence[Mapping[str, Any]],
                      binding: Mapping[str, str] | None = None,
                      outputs: Mapping[str, tuple[str, str, str]] | None = None,
                      schemas: Mapping[str, str] | None = None, window: str | None = None,
                      fixed: Sequence[str] = ()) -> None:
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
            root = G.supplied_root(program, v, supplied[v], rec, want, sorted(supplied), log, prog, binding)
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

__all__ = ["FORMAT", "FORMAT_V2", "KINDS", "PROTOCOL", "RECEIPT", "WINDOW_DOMAIN", "check", "digest", "held_roots",
           "matches_public", "query_digest", "receipt", "record", "refuse_unwindowed", "registered_entries", "unwindowed",
           "window_domain"]
