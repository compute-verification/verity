"""ROOTS1 on vllm's side: the Commit in H1's serving engine, whose record registers each step's A table of K
``a/<K>@<s>`` as its AROW of the step's calls of K and each round's input-ids table ``ids@<s>`` as its step's
input token ids, layer 0's input. Core's credit holds each to the Commit's value of the same rows (`credit._arow`).

* **The Commit** (`acquire.h1_commit.attach`): the windowed collector under rows-sha512 (`SCHEME`: hm96-sha512 rows in
  frame-v3-sha512 trees, 64-byte roots; with no Build, so not a Commit of record) with the int8 source
  (`acquire.h1_commit.H1Source`) at each linear H1 records, at the linear's Int8RowQuant_v1 site `site(module)` at every
  engine step (`Sites`: no Build names its fx node), a routed expert's by its H1 name (under its call, an idle
  expert's zero rows too), and the step-input port (`served_record.StepInputPort`): the step's input token ids, taken
  at the model's forward, so also on a model that gathers its embedding fused and never calls it (GLM-5.3 and
  DeepSeek-V3.2 at TP1). `CommitTap` wraps
  the committer's acquire and keeps a host copy of what the Commit acquired at each engine step: each site's x_q
  (``/0``) and s_a (``/1``), and the ids. `committed` is what a run records of the Commit, refused unless its root is
  rows-sha512's.
* **The server's Commit** (`acquire.h1_commit.tap`): on serve-openai's rows-sha512 collector H1 builds no committer.
  The tap reads each linear's x_q and s_a where the bench row's int8 source acquires them, under the manifest's op path
  ``<module>/<fx node>`` (``ops``, read as `site(module)`), and a step's ids at its step-input port.
* **The ids** (``port``): the port's T ids, zero past the step's scheduled tokens, read up to those tokens, so they
  carry no CUDA graph's padding.
* **Its values** (`CommitTap.arow`, `CommitTap.input_ids`): call c = (s, w)'s rows are the Commit's x_q and s_a of
  linear w's site at the engine step H1's step s maps to, in H1's table form (`pouw_h1_served.Served.a_words`), and its
  AROW of step s and K is those of step s's calls of K, call after call in call order (`Served.step_calls`) whatever
  order the forward ran them in, registered as `credit.commit_rows` of (s, K) once they are all in; round s's input ids
  are the Commit's port ids at that engine step (`pouw_h1_served.ids_words`), registered as `credit.commit_ids`
  of s. Step s's first call to come, whichever linear its forward runs first, maps s to the next engine step the
  Commit acquired (an empty call's to none). When the rows are the window's, the Commit registers the window's own
  tree, one commitment; rows off them make a tree of the Commit's own under the same domain and salts, at another root.
* **The record** (`CommitTap.record`, the commitment's ``commit``, `CommitTap.hex`): a ``verity/registration/v2`` record
  (one-stage's `registration`) of those values, with what the tap refused, its canonical JSON in hex as
  `credit.commit_of` reads it. `pouw_h1.calls_root` binds its digest (`credit.calls_root`), so the sampling coin follows it.
* **A named record** (`pouw_h1.H1Run`'s ``commit_record``): in place of the tap's, the window names a record that
  reuses its tables, one-stage's registration of the window's audited circuit. At the close the window's registration
  W (`registration`), a v2 record of its own entries, goes into the prover's log, and the named record is chained right
  after W and registers each of the window's tables under the Commit's name with W's entry of it (`reused`), so ROOTS1
  holds it as it holds the tap's. The audited circuit's side (`reuse`): its tables are W's (`shape`: one K, the same
  calls each round), each the rows its proof blocks make salted as H1 salted W's (`Reused`), refused unless W's root.
* **The serving process's own check** (`roots1`, at the window's close): each step's A table of each K and each round's
  input-ids table is the Commit's value of the same rows (`credit._arow`'s rule), the tap refused nothing, and each
  prefill row's input id is its request's prompt token (`ids_reason`, until Served.lean's layer-0 tie holds it in core).
  A window that fails credits nothing (`UNTIED`, `pouw_h1.H1Run.gated`).
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from verity.circuits.codec import canonical_json
from verity.protocols.pouw.verifier import credit as CR
from verity.protocols.pouw.verifier import weights as Wt
from verity.protocols.spot.verifier.one_stage import registered as G
from verity.protocols.spot.verifier.one_stage import registration as REG

from verity_vllm.commit import step_rows
from verity_vllm.protocols.pouw.h1 import names as NM
from verity_vllm.protocols.pouw.h1 import registered as NR
from verity_vllm.protocols.pouw.h1.words import ids_words

#: the scheme H1's own Commit commits under, and its run root's bytes
SCHEME, ROOT_BYTES = step_rows.NAME, 64

#: the credit refusal of every call of a window whose tables aren't the Commit's values (``details.roots1``)
UNTIED = ("credit: a-not-the-commits: a call's A table or a round's input-ids table is not the Commit's value of its "
          "rows, or a step's input ids are not its requests' prompt tokens (details.roots1)")
#: ``commit_record(run)``: at the window's close, once its tables are fixed, the window's registration (`registration`,
#: chained into the prover's log) and the record reusing its tables that the window names as its Commit record (`named`);
#: it raises when it has none
CommitRecord = Callable[[Any], tuple[Mapping[str, Any], Mapping[str, Any]]]


def site(module: str) -> str:
    """Linear ``module``'s Int8RowQuant_v1 site: the op path its x_q and s_a are committed under (``<site>/0``, ``/1``)."""
    return f"{module}/row_quant"


class Sites(dict):
    """`Int8Source`'s quant sites, (engine step, module) -> `site(module)` for each of `modules` at every engine step: the
    source reads them with ``get``."""

    def __init__(self, modules: Sequence[str]) -> None:
        super().__init__({(0, m): site(m) for m in modules})
        self.modules = frozenset(modules)

    def get(self, key: Any, default: Any = None) -> Any:
        return site(key[1]) if key[1] in self.modules else default


def _host(t: Any) -> np.ndarray:
    return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)


def _same(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    return all(a.get(k) == b.get(k) for k in ("root", "domain", "leaves"))


@dataclass
class CommitTap:
    """What the Commit acquired, kept for H1 (`tap` on a committer, `attach`), and the Commit's values built from it.
    ``port`` is the name the Commit's step-input port acquires the step's token ids under, T ids zero past the step's
    scheduled tokens; ``ops`` the Commit's op path of a site, ``<module>/<fx node>``, by which it acquires the site's x_q
    and s_a, -> the site (`site`)."""

    port: str | None = None
    ops: dict[str, str] = field(default_factory=dict)
    refused: list[str] = field(default_factory=list)
    #: the Commit's registered entries by its name of them (`credit.commit_rows`, `credit.commit_ids`), and where it
    #: acquired each; how many are trees of the Commit's own (rows off the window's)
    values: dict[str, dict[str, Any]] = field(default_factory=dict)
    where: dict[str, dict[str, Any]] = field(default_factory=dict)
    own: int = 0
    #: each round's input ids as the Commit acquired them
    round_ids: dict[int, list[int]] = field(default_factory=dict)
    #: (engine step, site, ``0`` x_q or ``1`` s_a) -> the member name the Commit acquired it under, which its store keys
    acquired: dict[tuple[int, str, str], str] = field(default_factory=dict)
    #: site -> [engine step, x_q, s_a] the Commit acquired last; engine step -> its ids, the port's cut to the step's
    #: scheduled tokens; H1 step -> engine step
    _x: dict[str, list[Any]] = field(default_factory=dict)
    _ids: dict[int, np.ndarray] = field(default_factory=dict)
    _steps: dict[int, int | None] = field(default_factory=dict)
    #: each open step A table's rows as the Commit acquired them so far (one block a call, by call) and where
    _rows: dict[str, dict[int, np.ndarray]] = field(default_factory=dict)
    _at: dict[str, dict[int, tuple[int, int | None, str, int]]] = field(default_factory=dict)
    _last: int = -1
    _com: Any = None

    def tap(self, com: Any) -> None:
        """Wrap ``com``'s acquire (the committer's ``_acquire(name, tensor, job)``, which every source calls) so each
        tensor it acquires is also `take`'n, with the step's scheduled tokens where the committer has them (its
        ``_plan_shape``, set before the step's forward)."""
        orig = com._acquire

        def acquire(name: str, t: Any, job: Any) -> Any:
            n = int((getattr(com, "_plan_shape", None) or (0,))[0])
            self.take(name, t, int(job.step), n or None)
            return orig(name, t, job)
        com._acquire, self._com = acquire, com

    def untap(self) -> None:
        if self._com is not None:
            vars(self._com).pop("_acquire", None)
            self._com = None

    def take(self, name: str, t: Any, step: int, n: int | None = None) -> None:
        """One tensor the Commit acquired at engine step ``step``: a site's x_q or s_a, or the step's input ids, the
        port's first ``n``, the step's scheduled tokens."""
        if self.port is not None and name == self.port:
            if n is None:
                self.refused.append(f"engine step {step}: the port's input ids with no count of the step's scheduled tokens")
                return
            self._ids[step] = _host(t).reshape(-1)[:n].astype(np.int64)
            return
        op, _, port = name.rpartition("/")
        op = self.ops.get(op, op)
        if not op.endswith("/row_quant") or port not in ("0", "1"):
            return
        self.acquired[(step, op, port)] = name
        if port == "0":
            x = _host(t)
            self._x[op] = [step, x.reshape(len(x), int(np.prod(x.shape[1:]))).astype(np.int8), None]
        elif op in self._x and self._x[op][0] == step:
            self._x[op][2] = _host(t).reshape(-1).astype("<f4")
        else:
            self.refused.append(f"{op}: an s_a at engine step {step} with no x_q")

    def _engine_step(self, call: Any) -> int | None:
        if call.step not in self._steps and call.m:
            got = self._x.get(site(call.module))
            j = got[0] if got is not None and got[0] > self._last else None
            self._steps[call.step] = j
            if j is not None:
                self._last = j
        return self._steps.get(call.step)

    def _put(self, name: str, reg: Any, same: bool, where: Mapping[str, Any]) -> None:
        if name in self.values:
            self.refused.append(f"the Commit registers {name} twice")
        self.own += not same
        self.values[name], self.where[name] = {**reg.entry(), "value": name}, dict(where)

    def arow(self, run: Any, served: Any, call: Any, value: str, words: np.ndarray | None) -> None:
        """Call ``call``'s rows of the Commit's AROW of its step and K: the Commit's rows of its site at its step. Once the
        step's calls of K are all in, in whichever order they came (``words``, the window's step A table ``value``
        whole, else None), its value `credit.commit_rows` of (step, K) is the step's calls' rows call after call in call
        order: the window's tree of ``value`` when they are its ``words``, else the Commit's own."""
        j, s = self._engine_step(call), site(call.module)
        got = self._x.get(s)
        codes, scales = np.zeros((0, call.k), np.int8), np.zeros(0, "<f4")
        if j is not None and got is not None and got[0] == j:
            del self._x[s]
            if got[2] is None or got[1].shape != (len(got[2]), call.k):
                self.refused.append(f"call {call.index}: the Commit's x_q and s_a at {s} are not one row a scale of K "
                                    f"{call.k}")
            else:
                codes, scales = got[1], got[2]
        if len(codes) > served.m:
            self.refused.append(f"call {call.index}: the Commit's {len(codes)} rows are past its {served.m} slots")
            codes, scales = codes[:served.m], scales[:served.m]
        rows, cs = self._rows.setdefault(value, {}), served.step_calls(call.index)
        rows[call.index] = served.a_words(call, codes, scales)
        self._at.setdefault(value, {})[call.index] = (call.index, j, s, len(codes))
        if len(rows) < len(cs):
            return
        rows, places = self._rows.pop(value), self._at.pop(value)
        mine, at = np.concatenate([rows[c] for c in cs]), [places[c] for c in cs]
        same = words is not None and np.array_equal(mine, words) and value in served.registered
        reg = served.registered[value] if same else served.a_tree(run, call, mine)
        self._put(CR.commit_rows(call.step, call.k), reg, same,
                  {"round": call.step, **{f: [x[i] for x in at] for i, f in enumerate(("calls", "steps", "sites", "rows"))}})

    def ids_at(self, r: int) -> list[int]:
        """Round r's input ids: the Commit's ids of the engine step H1's step r maps to (none for a step it acquired
        nothing at), its scheduled tokens, so none of the positions a CUDA graph padded the step with
        (`pouw_h1.Call.pad`)."""
        j = self._steps.get(r)
        got = self._ids.get(j) if j is not None else None
        if j is not None and got is None:
            self.refused.append(f"step {r}: the Commit acquired no input ids at engine step {j}")
        self.round_ids[r] = [] if got is None else [int(t) for t in got]
        return self.round_ids[r]

    def input_ids(self, run: Any, served: Any, r: int, ids: Sequence[int]) -> None:
        """Round r's input ids ``ids`` as the Commit's value `credit.commit_ids` of r: the window's input-ids table when
        its words are the same, else the Commit's own tree."""
        try:
            mine = served.id_words(ids)
        except ValueError as e:
            self.refused.append(f"step {r}: {e}")
            mine = served.id_words([])
        theirs = served.ids_of(r)
        same = theirs is not None and np.array_equal(mine, theirs)
        reg = served.ids[r] if same else served.ids_tree(run, r, mine)
        self._put(CR.commit_ids(r), reg, same, {"round": r, "step": self._steps.get(r), "rows": len(ids)})

    def record(self, served: Any) -> dict[str, Any]:
        """The Commit's ``verity/registration/v2`` record of its values, under the window's Program."""
        return {"format": REG.FORMAT_V2, "protocol": REG.PROTOCOL, "program": served.program_digest,
                "window": {"kind": "served", "refused": list(self.refused)},
                "registered": [self.values[n] for n in sorted(self.values)]}

    def hex(self, served: Any) -> str:
        """`record`'s canonical JSON in hex, the window evidence's ``commit`` (`credit.commit_of`)."""
        return canonical_json(self.record(served)).hex()


def digest(commit: str) -> bytes:
    """The digest `credit.calls_root` binds of a Commit record in hex (`CommitTap.hex`). ValueError on one
    `credit.commit_of` refuses."""
    got, why = CR.commit_of(commit)
    if why is not None:
        raise ValueError(why[1])
    return got[0]


def _tables(run: Any) -> Iterator[tuple[int, Any, str, list[int] | None]]:
    """The window's tables credit holds to the Commit's values (`credit._arow`), each round's A table of each K and then
    each round's input-ids table: its round, its registered tree (None when it has none), the Commit's name of it and,
    for an A table, the round's calls of its K."""
    served = run.served
    layout = {"calls": served.calls, "linears": served.linears, "m": served.m}
    for s, k, calls in CR.step_tables(layout, served.steps):
        yield s, served.registered.get(served.a_table(calls[0])), CR.commit_rows(s, k), calls
    for s in range(run.steps):
        yield s, served.ids.get(s), CR.commit_ids(s), None


def untied(run: Any) -> str | None:
    """Why the window's tables aren't the Commit's values (`credit._arow`'s rule, on the serving process's own trees): a
    tap that refused, a step's A table of K that isn't the Commit's `credit.commit_rows` of its round and K, or a round's
    input-ids table that isn't its `credit.commit_ids`, in root, domain and leaf count. None when each is."""
    served, tap = run.served, run.served.tap
    if tap.refused:
        return f"roots1: the Commit's tap refused: {'; '.join(tap.refused[:4])}"
    if served.refused or served.program is None:
        return "roots1: the window's served trees were refused, so no table is the Commit's"
    for s, mine, name, calls in _tables(run):
        got = tap.values.get(name)
        if mine is not None and got is not None and _same(mine.entry(), got):
            continue
        if calls is None:
            return (f"roots1: round {s}'s input-ids table is not the Commit's {name}: its step's input ids "
                    "are not the ones the Commit acquired" + (", round 0's its prompts' tokens" if s == 0 else ""))
        at = tap.where.get(name, {})
        return (f"roots1: round {s}'s A table {served.a_table(calls[0])} of calls {calls} is not the Commit's {name} "
                f"({at.get('sites')} at engine steps {at.get('steps')}): its rows are not the ones the Commit acquired")
    return None


def registration(run: Any, previous: str | None = None) -> dict[str, Any]:
    """The window's registration W at its close: a ``verity/registration/v2`` record of its served trees' entries
    (`pouw_h1_served.Served.entries`: its tables' and weights' roots, each domain binding the window Program and its
    open root) under the window Program, with what the served trees refused, chained after ``previous``, the digest of
    the prover's last record (`registration.digest`; None for its first). It is the one-stage log's record whose
    tables a named Commit record reuses (`reused`)."""
    served = run.served
    return {"format": REG.FORMAT_V2, "protocol": REG.PROTOCOL, "program": served.program_digest,
            "window": {"kind": "served", "entry": run.entry, "open_root": served.root, "refused": list(served.refused)},
            "registered": served.entries(), "previous": previous}


def reused(run: Any, window: Any, record: Any) -> str | None:
    """Why ``record`` can't be the window's Commit record in place of its tap's, its tables reused from ``window``: the
    window's served trees were refused, ``window`` isn't the window's own registration (`registration`), ``record``
    isn't a record credit reads as a Commit record (`credit.commit_of`) chained right after ``window`` (its
    ``previous`` the digest of ``window``), or it doesn't register each of the window's A tables and input-ids tables
    once under the Commit's name (`credit.commit_rows`, `credit.commit_ids`) with the root, domain and leaf count of
    ``window``'s entry of it, ROOTS1's rule (`credit._arow`). None when it can be."""
    served = run.served
    if served.refused or served.program is None:
        return "the window's served trees were refused, so it has no tables a Commit record can reuse"
    if not isinstance(window, Mapping) or dict(window) != registration(run, window.get("previous")):
        return ("the window's registration is not its own: a verity/registration/v2 record of the window's served "
                "trees' entries under its Program and open root (pouw_h1_commit.registration)")
    if not isinstance(record, Mapping):
        return "the named record is not a JSON object"
    try:
        got, why = CR.commit_of(canonical_json(dict(record)).hex())
    except (TypeError, ValueError) as e:
        return f"the named record has no canonical JSON: {e}"
    if why is not None:
        return f"the named record is not a Commit record credit reads: {why[1]}"
    previous = REG.digest(window)
    if record.get("previous") != previous:
        return (f"the named record's previous is not the window registration's digest {previous[:16]}…: it is chained "
                "right after the registration whose tables it reuses")
    theirs: dict[Any, list[Mapping[str, Any]]] = {}
    for entry in got[1]:
        theirs.setdefault(entry.get("value"), []).append(entry)
    for s, mine, name, calls in _tables(run):
        table = f"A table {served.a_table(calls[0])}" if calls is not None else "input-ids table"
        if mine is None:
            return f"the window registers no round {s}'s {table}, so no Commit record reuses it as {name}"
        entries = theirs.get(name, [])
        if len(entries) == 1 and _same(mine.entry(), entries[0]):
            continue
        how = "twice" if len(entries) > 1 else "not" if not entries else "at another root, domain or leaf count"
        return (f"the named record registers {name} {how}: credit holds it to round {s}'s {table} as the window's "
                "registration commits it, registered once with that entry's root, domain and leaf count (ROOTS1)")
    return None


@dataclass(frozen=True)
class Reused(G.Registered):
    """A table of the window's registration W as a registered value of the audited circuit (`reuse`): W's rows, salts and
    tree of it, under W's entry's domain ``domain_id``, by the circuit's name of it, so its entry is W's entry but in name.
    W's entry names its domain by id alone (round 0's input ids bind the window's session, which W doesn't hold), so it
    has no `domain` to recompute."""

    domain_id: bytes = b""

    @property
    def domain(self) -> Any:
        raise ValueError(f"{self.value}: a table reused from the window's registration has W's domain id "
                         f"{self.domain_id.hex()[:16]}… and no binding to recompute it from (`entry`)")

    def entry(self) -> dict[str, Any]:
        return {"value": self.value, "root": self.root.hex(), "leaves": self.leaves, "words": self.words,
                **G.width_entry(self.word_bits), "domain": self.domain_id.hex()}


def shape(layout: Mapping[str, Any], steps: int) -> tuple[int, int, int]:
    """The audited circuit's tables (`window_class.Tables`) of H1's window Program of ``layout`` (its meta's, `served.
    write_program`) over its ``steps`` rounds: ``(K, M, calls)``, its one K, its M rows a call and its calls of K a round.
    ValueError unless every round's A tables are one K's, the same calls a round, a row one leaf (`weights.chunks`)."""
    got = CR.step_tables(dict(layout), steps)
    if not got or len(got) != steps or len({(k, len(cs)) for _s, k, cs in got}) != 1:
        raise ValueError(f"H1's window Program's A tables {[(s, k, len(cs)) for s, k, cs in got]} (round, K, calls) over its "
                         f"{steps} rounds are not one K's with the same calls each round, the tables the audited circuit reads")
    k, calls = got[0][1], len(got[0][2])
    if Wt.chunks(k) != 1:
        raise ValueError(f"K = {k}: an A row of {Wt.chunks(k)} leaves, and the audited circuit reads a quantizer's row as one")
    return k, int(layout["m"]), calls


def reuse(window: Mapping[str, Any], layout: Mapping[str, Any], steps: int, key: bytes,
          rows_of: Callable[[str, int, np.ndarray], np.ndarray]) -> dict[str, Reused]:
    """The audited circuit's tables (`shape`) as its registered values, each W's table of it (``window``, the window's
    registration, `registration`): round s's A table ``<prefix>a/<K>@<s>`` as `credit.commit_rows` of (s, K), and its
    input-ids table ``<prefix>ids@<s>`` as `credit.commit_ids` of s. Each holds the rows ``rows_of(name, rows, empty)``
    gives (the rows the circuit's proof blocks make of it, every other row ``empty``: zero codes of scale 0, an id
    `words.EMPTY_ID`), salted as H1 salts W's (H1's ``key``, at the table's `names.nonce` or `names.IDS_NONCE` + s),
    under W's entry's domain. ValueError unless each is W's entry of it in root, leaves and words: the proof blocks don't
    make the table H1 committed, so the window's registration can't reuse it."""
    k, m, _calls = shape(layout, steps)
    theirs = {e.get("value"): e for e in window.get("registered") or () if isinstance(e, Mapping)}
    tables = [(CR.commit_rows(s, k), f"{NM.PREFIX}a/{k}@{s}", NM.nonce(cs[0], "a"), m * len(cs),
               np.zeros(k // 2 + 2, np.uint16)) for s, _k, cs in CR.step_tables(dict(layout), steps)]
    tables += [(CR.commit_ids(s), f"{NM.PREFIX}ids@{s}", NM.IDS_NONCE + s, m, ids_words([], 1)[0]) for s in range(steps)]
    out: dict[str, Reused] = {}
    for name, h1, nonce, n, empty in tables:
        e = theirs.get(h1)
        if e is None or not isinstance(e.get("domain"), str):
            raise ValueError(f"the window's registration registers no {h1}, which the audited circuit reads as {name}")
        words = np.ascontiguousarray(rows_of(name, n, empty), dtype=np.uint16)
        dom = bytes.fromhex(e["domain"])
        bcs, levels = NR._numpy(words, key, nonce, None, dom)
        got = out[name] = Reused(program=str(window.get("program")), value=name, words=int(words.shape[1]),
                                 word_bits=G.WORD_BITS, rows=NR.Rows(words, n), salts=NR.Salts(key, nonce, n), bcs=bcs,
                                 levels=levels, schema=G.SCHEMA_V2, domain_id=dom)
        if (got.root.hex(), got.leaves, got.words, G.WORD_BITS) != (e.get("root"), e.get("leaves"), e.get("words"),
                                                                     e.get(G.WIDTH_KEY, G.WORD_BITS)):
            raise ValueError(f"{name}: the rows the audited circuit's proof blocks make of it are not the window's "
                             f"registration's {h1} (root {got.root.hex()[:16]}… for {str(e.get('root'))[:16]}…, "
                             f"{got.leaves} rows of {got.words} words for {e.get('leaves')} of {e.get('words')}): the model "
                             "didn't make the table H1 committed, or H1's key is another")
    return out


def named(run: Any) -> tuple[str | None, dict[str, Any]]:
    """The Commit record ``run``'s ``commit_record`` names at the window's close (`pouw_h1.H1Run.finish`), its canonical
    JSON in hex as `credit.commit_of` reads it, and what the run records of it (``details.commit_record``): the digests
    of the window's registration and of the record; else None and why it can't be the window's (`reused`), or how
    ``commit_record`` failed. A failure of any other type is recorded in ``run`` as that refusal, as `bind` records one,
    and raised."""
    try:
        window, record = run.commit_record(run)
        why = reused(run, window, record)
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as e:   # the window's refusal, not the server's
        why = f"{type(e).__name__}: {e}"
    except BaseException as e:
        run.commits["commit"], run.details["commit_record"] = None, _unnamed(f"{type(e).__name__}: {e}")
        raise
    if why is not None:
        return None, _unnamed(why)
    return canonical_json(dict(record)).hex(), {"named": True, "registration": REG.digest(window),
                                                "record": REG.digest(record)}


def _unnamed(why: str) -> dict[str, Any]:
    return {"named": False, "reason": why[:500]}


def ids_reason(run: Any, tap: CommitTap) -> str | None:
    """Why some round's ids aren't its slots' prompt tokens: a round whose Commit ids aren't its slots, one per served row,
    or a slot whose id isn't the next prompt token of the request it names (counted from its admission, again from 0
    once it is resumed after a preemption). None when every prefill row's input is its prompt's token. Decode rows are
    generated tokens, and it doesn't check them."""
    pos: dict[int, int] = {}
    readmitted: dict[int, set[int]] = {}
    for x in run.readmitted:
        readmitted.setdefault(int(x["round"]), set()).add(int(x["index"]))
    for r in range(run.steps):
        ids, owners = tap.round_ids.get(r, []), run.owners.get(r, [])
        if len(ids) != len(owners):
            return f"roots1: step {r}'s {len(ids)} input ids in the Commit are not its {len(owners)} slots of the batch"
        for q in readmitted.get(r, ()):
            pos[q] = 0
        for i, (t, q) in enumerate(zip(ids, owners)):
            if q is None:
                continue
            p, prompt = pos.get(q, 0), run.requests[q].prompt
            pos[q] = p + 1
            if p < len(prompt) and t != prompt[p]:
                return (f"roots1: step {r}'s slot {i} has input id {t} in the Commit, and request {q}'s prompt token {p} "
                        f"is {prompt[p]}: layer 0's input is not its round's request table")
    return None


def record_of(run: Any) -> str | None:
    """The window's Commit record in hex: with ``commit_record``, the one `bind` named (None before it, or when it named
    none); else, with the Commit in ``run``'s engine (its ``served``'s tap), the tap's (`CommitTap.hex`)."""
    if run.commit_record is not None:
        return run.commits.get("commit")
    if run.served is None or run.served.tap is None:
        return None
    return run.served.tap.hex(run.served)


def bind(run: Any) -> str | None:
    """Bind the window's Commit record into ``run``'s commitment at its close (`pouw_h1.H1Run.finish`, before its
    `calls_root`): ``commit_record``'s (`named`, what the run records of it in ``details.commit_record``), else its
    tap's. When ``commit_record`` names none, the window's refusal: ``finish`` binds no record, opens no draw coin and,
    once its transcript is written, raises it as ValueError. Else None."""
    if run.commit_record is not None:
        run.commits["commit"], run.details["commit_record"] = named(run)
        if run.commits["commit"] is None:
            return f"{run.entry}: the window names no Commit record: {run.details['commit_record']['reason']}"
    elif (commit := record_of(run)) is not None:
        run.commits["commit"] = commit
    return None


def roots1(run: Any) -> dict[str, Any]:
    """The window's ROOTS1 outcome at its close (``details.roots1``): `untied`, then `ids_reason`, and what the tap
    built."""
    tap = run.served.tap
    why = untied(run) or ids_reason(run, tap)
    return {"tied": why is None, "reason": why, "values": len(tap.values), "own_trees": tap.own,
            "steps": {str(s): j for s, j in sorted(tap._steps.items())}}


def committed(com: Any, rc: Any, extra: Sequence[str] = ()) -> dict[str, Any]:
    """What a run records of H1's own Commit ``com``, finalized as ``rc`` (its `RunCommitment`): the run root, the scheme,
    the committer's name and the finalize's ``extra`` keys. ValueError unless it committed under `SCHEME` to a
    `ROOT_BYTES` root: a SHA-256 run root (vllm-v1's, 32 bytes) is refused."""
    scheme, root = getattr(com, "commit_scheme", None), bytes(rc.run_root)
    if scheme != SCHEME or len(root) != ROOT_BYTES:
        raise ValueError(f"H1's Commit {getattr(com, 'name', com)} committed under {scheme} to a {len(root)}-byte run root, not "
                         f"under {SCHEME} to a {ROOT_BYTES}-byte one: build it with acquire.h1_commit.attach")
    return {"run_root": root.hex(), "commit_scheme": scheme, "name": com.name, **{k: rc.extra.get(k) for k in extra}}


def calls_root(salt: bytes, manifest: Sequence[Mapping[str, Any]], served: bytes | None = None,
               commit: str | None = None) -> bytes:
    """`pouw_h1.H1Run.calls_root` over the manifest rows `manifest`, the served trees' digest `served` when the window has them,
    and the digest of its Commit record `commit` (hex, `pouw_h1_commit.digest`) when it has one: core's
    `credit.calls_root`, the one place the record is bound. ValueError on a record `credit.commit_of` refuses."""
    return CR.calls_root(salt, manifest, served, None if commit is None else digest(commit))


__all__ = ["ROOT_BYTES", "SCHEME", "CommitRecord", "CommitTap", "Reused", "Sites", "UNTIED", "bind", "calls_root",
           "committed", "digest", "ids_reason", "named", "record_of", "registration", "reuse", "reused", "roots1", "shape", "site",
           "untied"]
