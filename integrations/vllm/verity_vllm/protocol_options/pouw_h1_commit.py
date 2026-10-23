"""ROOTS1 (VLLM-73) on vllm's side: the Commit in H1's serving engine, each call's A table ``a/<K>@<c>`` the Commit's
AROW of its (step, linear), and each step's input token ids, layer 0's input, as round s's ``ids@<s>``.

* **The Commit** (`attach`): the committer (`commit.committer.native_host.NativeHostCommitter`, with no Build, so not a
  Commit of record) with the int8 source (`acquire.sources.int8_source.Int8Source`) at each linear H1 records, at the
  linear's Int8RowQuant_v1 site `site(module)` at every engine step (`Sites`: no Build names its fx node), and the
  embedding's input, which the committer acquires as ``<embed>/in``, the step's input token ids. `CommitTap` wraps the
  committer's acquire and keeps a host copy of what the Commit acquired at each engine step: each site's x_q (``/0``)
  and s_a (``/1``), and the ids.
* **AROW** (`CommitTap.arow`, as `pouw_h1_served.Served` registers call c = (s, w)): the Commit's x_q and s_a of linear w's
  site at the engine step H1's step s maps to, as H1's A table of c. Layer 0's call maps s to the next engine step the
  Commit acquired (an empty call's to none); a step it acquired nothing at is empty. The table is H1's value ``a/<K>@<c>`` over its M slots, each
  slot the codes then the f32 s_a, with zero codes of scale 0 past them (`Served.a_words`), under H1's registered
  domain and salts (`Served.a_tree`). Equal rows make one tree, so an honest window's AROW is H1's registered entry, one
  commitment (the tap takes H1's tree). Rows off H1's make a tree of the Commit's own, at another root.
* **ids** (`CommitTap.ids`, at round s's close): `pouw_h1_requests.ids_table` of the Commit's ids of step s, whose root
  goes under round s's inner root (`pouw_h1_requests.round_root`), before its coin.
* **The record** (`CommitTap.record`, the commitment's ``served.commit``): each call's AROW entry (its registered entry,
  with its call, engine step and site), each round's ids root, the Commit's run root once it is given
  (`CommitTap.run_root`), and what the tap refused. `bind` folds its digest into the served trees' digest
  (`pouw_h1_served.served_digest`), the one calls_root binds, so the draw coin follows it.
* **The checks** stand in for core's until it holds the ties (compute accounting's `credit._arow` and Served.lean's
  layer-0 tie, VLLM-73).
  - `arow_reason`, on the record: each call's registered A entry is the Commit's AROW of the call (value, root, domain,
    rows and words), and each round's ids root is the Commit's. `pouw_h1_challenger.manifest_reason` applies it to a
    record that carries the Commit's.
  - `ids_reason`, on the serving process's own values: in each round, a slot's input id is the prompt token of the
    request its slot table names (`pouw_h1.H1Run.owners`, `pouw_h1_requests.RequestEntry.prompt`), position by
    position up to the prompt's length. Decode rows are generated tokens, and it doesn't check them.
  - A window that fails either credits nothing (`UNTIED`, `pouw_h1.H1Run.gated`).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from verity.core.protocols.pouw import credit as CR
from verity.ml.circuits.codec import canonical_json

from verity_vllm.protocol_options import pouw_h1_requests as RQ

SCHEMA = "verity-vllm/pouw-h1-commit/v1"
DIGEST_TAG = b"verity/pouw/h1/commit/v1\0"
BIND_TAG = b"verity/pouw/h1/served-commit/v1\0"
#: the credit refusal of every call of a window whose A tables or input ids aren't the Commit's (``details.roots1``)
UNTIED = ("credit: a-not-the-commits: a call's A table is not the Commit's AROW of its (step, linear), or a step's input "
          "ids are not its requests' prompt tokens (details.roots1)")
#: the record fields `arow_reason` holds a call's registered A entry and its AROW to
SAME = ("value", "root", "domain", "leaves", "words")


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


class _EveryStep(dict):
    """`Int8Source`'s names at each engine step (its ``_at``, X-03's targets): step 0's at every step."""

    def get(self, key: Any, default: Any = None) -> Any:
        return self[0] if 0 in self else default


def _host(t: Any) -> np.ndarray:
    return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)


@dataclass
class CommitTap:
    """What the Commit acquired, kept for H1 (`tap` on a committer, `attach`), and each call's AROW and each round's ids
    table built from it. ``embed`` is the module whose input the committer acquires as the step's token ids."""

    embed: str | None = None
    run_root: bytes | None = None
    refused: list[str] = field(default_factory=list)
    #: each call's AROW entry, by call index; how many are trees of the Commit's own (rows off H1's)
    arows: dict[int, dict[str, Any]] = field(default_factory=dict)
    own: int = 0
    #: each round's ids record and table, and the ids it holds
    ids_roots: dict[int, dict[str, Any]] = field(default_factory=dict)
    tables: dict[int, Any] = field(default_factory=dict)
    round_ids: dict[int, list[int]] = field(default_factory=dict)
    #: site -> [engine step, x_q, s_a] the Commit acquired last; engine step -> its ids; H1 step -> engine step
    _x: dict[str, list[Any]] = field(default_factory=dict)
    _ids: dict[int, np.ndarray] = field(default_factory=dict)
    _steps: dict[int, int | None] = field(default_factory=dict)
    _last: int = -1
    _com: Any = None

    def tap(self, com: Any) -> None:
        """Wrap ``com``'s acquire (the committer's ``_acquire(name, tensor, job)``, which every source calls) so each
        tensor it acquires is also `take`'n."""
        orig = com._acquire

        def acquire(name: str, t: Any, job: Any) -> Any:
            self.take(name, t, int(job.step))
            return orig(name, t, job)
        com._acquire, self._com = acquire, com

    def untap(self) -> None:
        if self._com is not None:
            vars(self._com).pop("_acquire", None)
            self._com = None

    def take(self, name: str, t: Any, step: int) -> None:
        """One tensor the Commit acquired at engine step ``step``: a site's x_q or s_a, or the embedding's input ids."""
        if self.embed is not None and name == f"{self.embed}/in":
            self._ids[step] = _host(t).reshape(-1).astype(np.int64)
            return
        op, _, port = name.rpartition("/")
        if not op.endswith("/row_quant") or port not in ("0", "1"):
            return
        if port == "0":
            x = _host(t)
            self._x[op] = [step, x.reshape(len(x), -1).astype(np.int8), None]
        elif op in self._x and self._x[op][0] == step:
            self._x[op][2] = _host(t).reshape(-1).astype("<f4")
        else:
            self.refused.append(f"{op}: an s_a at engine step {step} with no x_q")

    def _engine_step(self, call: Any) -> int | None:
        if call.layer == 0 and call.step not in self._steps and call.m:
            got = self._x.get(site(call.module))
            j = got[0] if got is not None and got[0] > self._last else None
            self._steps[call.step] = j
            if j is not None:
                self._last = j
        return self._steps.get(call.step)

    def arow(self, run: Any, served: Any, call: Any, value: str, words: np.ndarray | None) -> None:
        """Call ``call``'s AROW: the Commit's rows of its site at its step as H1's A table ``value``; H1's registered tree
        when its rows (``words``) are the same, else the Commit's own."""
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
        mine = served.a_words(call, codes, scales)
        same = words is not None and np.array_equal(mine, words) and value in served.registered
        reg = served.registered[value] if same else served.a_tree(run, call, mine)
        self.own += not same
        self.arows[call.index] = {**reg.entry(), "call": call.index, "step": j, "site": s, "rows": len(codes)}

    def ids(self, run: Any, r: int) -> bytes:
        """Round r's ids table's root: the Commit's input ids of the engine step H1's step r maps to (none for a step it
        acquired nothing at)."""
        j = self._steps.get(r)
        got = self._ids.get(j) if j is not None else None
        if j is not None and got is None:
            self.refused.append(f"step {r}: the Commit acquired no input ids at engine step {j}")
        ids = [] if got is None else [int(t) for t in got]
        try:
            table = RQ.ids_table(run.salt, r, run.rows, ids, run._key)
        except ValueError as e:
            self.refused.append(str(e))
            ids, table = [], RQ.ids_table(run.salt, r, run.rows, [], run._key)
        self.round_ids[r], self.tables[r] = ids, table
        self.ids_roots[r] = {"round": r, "step": j, "root": table.commitment.root.hex(), "rows": len(ids)}
        return table.commitment.root

    def record(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "arow": [self.arows[c] for c in sorted(self.arows)],
                "ids": [self.ids_roots[r] for r in sorted(self.ids_roots)],
                "run_root": None if self.run_root is None else self.run_root.hex(), "refused": list(self.refused)}


def digest_of(record: Mapping[str, Any]) -> bytes:
    """The Commit record's digest (`CommitTap.record`, canonical JSON)."""
    return hashlib.sha512(DIGEST_TAG + canonical_json(dict(record))).digest()


def bind(served: bytes, commit: Mapping[str, Any] | None) -> bytes:
    """The digest calls_root binds for a window's served trees' digest ``served`` (`pouw_h1_served.digest_of`) and its
    Commit record ``commit``: ``served`` itself with no Commit, else SHA-512 of both. Compute accounting may move the
    bind to the session's close; this is the one place it is made."""
    return served if commit is None else hashlib.sha512(BIND_TAG + served + digest_of(commit)).digest()


def arow_reason(record: Mapping[str, Any]) -> str | None:
    """Why the window `record` (`pouw_h1.CALLS_FILE`: its commitment and manifest rows) doesn't hold each call's A table
    to the Commit's AROW: a call whose manifest row registers no single A table, or whose registered entry isn't its
    AROW in the Commit's record, a Commit record that refused or lists another call, or a round whose ids root isn't the
    Commit's. None when it does, and for a record that carries no Commit record (core's `credit._arow` will need one)."""
    c = record.get("commitment") or {}
    served = c.get("served") or {}
    commit = served.get("commit")
    if commit is None:
        return None
    try:
        if commit.get("refused"):
            return f"roots1: the Commit's tap refused: {'; '.join(commit['refused'][:4])}"
        arow = {int(e["call"]): e for e in commit["arow"]}
        entries = {e["value"]: e for e in served["entries"]}
        rows = list(record.get("calls") or ())
        for row in rows:
            i = int(row["index"])
            registered = (row.get("tables") or {}).get("registered") or {}
            mine = [v for v in registered if (CR.a_value(v) or (None, None))[1] == str(i)]
            if len(mine) != 1:
                return f"roots1: call {i}'s manifest row registers {len(mine)} A tables, not its one a/<K>@{i}"
            v, got = mine[0], arow.get(i)
            if got is None:
                return f"roots1: call {i} has no AROW in the Commit's record"
            e = entries.get(v)
            if e is None or registered[v] != e["root"] or any(got.get(k) != e.get(k) for k in SAME):
                return (f"roots1: call {i}'s A table {v} is not the Commit's AROW of its step ({got.get('site')} at "
                        f"engine step {got.get('step')}): its rows are not the ones the Commit acquired")
        if sorted(arow) != [int(row["index"]) for row in rows]:
            return "roots1: the Commit's record lists AROWs of calls the window doesn't have"
        ids = {int(x["round"]): x["root"] for x in commit["ids"]}
        for x in c.get("rounds") or ():
            if x.get("ids") is None or x.get("ids") != ids.get(int(x["round"])):
                return (f"roots1: round {x['round']}'s root is not over the Commit's ids table of its step (layer 0's "
                        "input)")
    except (AttributeError, KeyError, TypeError, ValueError) as e:
        return f"roots1: the Commit's record is malformed ({e})"
    return None


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


def roots1(run: Any) -> dict[str, Any]:
    """The window's ROOTS1 outcome at its close: `arow_reason` on its record so far and `ids_reason` on its values, and
    what the tap built (``details.roots1``)."""
    tap = run.served.tap
    record = {"commitment": {"served": run.served.document(), "rounds": run.round_log}, "calls": run._rows(run.calls)}
    why = arow_reason(record) or ids_reason(run, tap)
    return {"tied": why is None, "reason": why, "arows": len(tap.arows), "own_trees": tap.own,
            "rounds": len(tap.ids_roots), "steps": {str(s): j for s, j in sorted(tap._steps.items())}}


def source(com: Any, model: Any, layers: Sequence[Any]) -> Any:
    """The int8 source (`Int8Source`) on committer ``com``, at the Int8RowQuant_v1 site of each of ``layers`` (H1's
    linears, `pouw_h1_hooks.layers`, by their names in ``model``, as `pouw.install` names them) at every engine step."""
    from verity_vllm.acquire.sources.int8_source import Int8Source

    ids = {id(m) for m in layers}
    sites = Sites([name for name, m in model.named_modules() if id(m) in ids])
    if len(sites.modules) != len(ids):
        raise ValueError(f"{len(ids) - len(sites.modules)} of H1's linears are not named modules of the model")
    src = Int8Source(com, model, sites, [])
    src.quant, src._at = sites, _EveryStep(src._at)
    return src


def attach(engine: Any, layers: Sequence[Any], **committer: Any) -> tuple[Any, Any, CommitTap]:
    """The Commit on ``engine`` (`engine.build.build_engine`'s), installed: a `NativeHostCommitter` (``committer`` its
    keywords), the int8 source (`source`) and its `CommitTap`. Returns the committer, the source and the tap;
    ``committer.close()`` and ``tap.untap()`` remove them."""
    from verity_vllm.commit.committer.native_host import NativeHostCommitter

    com = NativeHostCommitter(engine, **committer)
    src = source(com, com.model, layers)
    com.attach_source(src)
    tap = CommitTap(next((n for n, m in com._modules if com._acquires_input(n, m)), None))
    if tap.embed is None:
        raise ValueError("the committer acquires no module's input: no embedding gives the step's token ids")
    tap.tap(com)
    com.install()
    return com, src, tap


__all__ = ["BIND_TAG", "CommitTap", "DIGEST_TAG", "SAME", "SCHEMA", "Sites", "UNTIED", "arow_reason", "attach", "bind",
           "digest_of", "ids_reason", "roots1", "site", "source"]
