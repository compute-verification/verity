"""The checkpoint tie of the committer's weights registration: a zero-knowledge proof, once per model and key, that the rows
committed under each registered weight root (`window_weights.weights_registration`: hm96 rows under the committer's secret
salts, a hiding commitment) are checkpoint X's, which the verifier checks before it pins those roots (`checked_pins`). It
never commits a weight: what it computes from X is X's public digests.

The tie is a window of the one proof service (`interface.Window`), one SHA-512 chain per safetensors shard of X in the
shard's byte order (`gates.sha512_stream`). The shard's header is public, so the chain starts at the verifier's own state
after it (a supplied value: its chaining value and its pending bytes). Each tensor after it is rows of one registered weight
parameter, and each row is a `Sha512StreamRow_v1` unit whose ``data`` port is a registered read of that row (opened against
the registered root at its public position, in-circuit, by the Lean verifier's registered reads) and whose state is wired
from the unit before. The shard's last row is a `Sha512StreamLast_v1` unit, which also ends the stream: its digest is a
registered read of the verifier's own SHA-512 of the shard file, whose SHA-256 the checkpoints manifest pins. A field read
at several widths (the tied embedding's table, LM head and ``x0`` rows) is one tensor of the shard: its first parameter is
the shard's (by name, an embedding's table after every other), and each other one is a chain of its own, from SHA-512's IV
over the field's bytes to the verifier's SHA-512 of them. An embedding's table (`window_builder.table_param`) is a declared
table (`interface.Table`), its rows ``hm96-sha512/row-seg/v1`` leaves its reads open as such (`interface.seg_ports`); a
template's port has one row schema, so a table's own chain hashes `TABLE_TAG` before the field's bytes, and its rows pend an
odd count of bytes where a shard's (after a header safetensors aligns to 8 bytes) and every other chain's pend an even
count: they are templates of their own. Under a ``work`` law with every unit in the output class (`Tie.outputs`), every unit is proved.

A field the weights record holds to an analytic table (the rotary table, `window_weights`) is no checkpoint bytes: the tie
reads none of it and names its parameters unrooted, as a registration of no served table does. A registration that roots
it is refused at the pins, since the tie window holds no root of it.

Public: the layout (the shards, their headers and tensor offsets, each parameter's rows and width), the digests and the
registered roots. Private: the rows, their salts and every state but the first. Refused by name: a cast or a composition
whose rows aren't its source tensors' bytes in order (1:1, fused along dim 0 and tied are), a row straddling two tensors, a
tensor two parameters split, a row of fewer than 56 words (C-Flock proves a unit only with a row of at least 112 bytes, one
that compresses before its last block), a shard of no tensor, bytes between or after the tensors, a tensor no registered
parameter holds, a checkpoint not of record or of another model, and a template whose port would read a table's rows and
another parameter's.

    verity-vllm weights-tie commit --phi PHI --checkpoint-dir X [--manifest M] --key-file K --registration-out REG \\
        --out DIR [--law work:1] [--cap 16] [--run-id ID] [--grain 512,9,64] [--served-dir Y --served-manifest MY [--forge-digest]]
    verity-vllm weights-tie verify --phi PHI --checkpoint-dir X [--manifest M] --registration REG --out DIR [--law ...] \\
        [--cap ...] [--run-id ...] [--grain ...]
    verity-vllm weights-tie pins --phi PHI --checkpoint-dir X [--manifest M] --registration REG --summary SUMMARY --out PINS \\
        [--law ...] [--cap ...] [--run-id ...] [--grain ...]

``commit``: the committer's registration (its key in ``--key-file``, made 0600 when absent and never printed) and its tie
window, written by core's registrar. ``--served-dir``: a committer serving another checkpoint's words as X's (the refused
arm), its registration committing them; ``--forge-digest``: its last units' digests forged to X's. ``verify``: the verifier's
tie of X (``DIR/tie.json``: its circuit's SHA-512, run root and units) and the records audit's ``--registered`` file
(``DIR/registered.json``: the registration's entries, held to the verifier's weights record, and the supplied values) and
``--outputs`` file (``DIR/outputs.json``: every unit). ``pins``: the registration's roots, written only once that audit's
``summary.json`` accepted the tie (`checked_pins`).
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from verity.core.protocols.sampled_proofs.one_stage import draw as D, registered as G, registration as R
from verity.ml.commitments.gates import sha512_stream as SS
from verity.ml.windows import interface as IF
from verity_vllm.check import weights_of_record as WR
from verity_vllm.check.checkpoint_index import read_safetensors_header
from verity_vllm.config import UsageError, group, option
from verity_vllm.records.weights_registration import WeightsRegistration

from . import window_weights as WW
from .window_builder import Grain

#: the domain of the run root's hash and the tie window's provenance
TIE_TAG = "verity-vllm/weights-tie/v1"
NAME = "verity-vllm/weights-tie"
#: the composition rules (`check.weights_of_record.compose`) whose field is its source tensors' bytes in order, besides a
#: fused one along dim 0 (``fused … (dim 0)``)
WHOLE = ("1:1", "tied lm_head <- model.embed_tokens.weight")
#: what a table's own chain hashes before the field's bytes (module docstring)
TABLE_TAG = b"t"
_GRAIN = Grain()
_CHUNK = 1 << 24


def _field(param: str) -> str:
    return WW._param(param)[0]


def _table(param: str) -> bool:
    return WW._param(param)[2]


def _whole(rule: Any) -> bool:
    return rule in WHOLE or (isinstance(rule, str) and rule.startswith("fused ") and rule.endswith(" (dim 0)"))


def _digests(path: str) -> tuple[str, bytes]:
    """The file's SHA-256 (hex), which the manifest pins, and its SHA-512, of one read."""
    a, b = hashlib.sha256(), hashlib.sha512()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            a.update(chunk)
            b.update(chunk)
    return a.hexdigest(), b.digest()


@dataclass(frozen=True)
class Chain:
    """One SHA-512 chain of the tie: ``name`` its supplied values' prefix, ``cv`` and ``pending`` the state it starts at
    (the verifier's own), ``rows`` its units' ``(parameter, row)`` in stream order, ``length`` the stream's bytes and
    ``digest`` the verifier's SHA-512 of them."""
    name: str
    cv: tuple[int, ...]
    pending: bytes
    rows: tuple[tuple[str, int], ...]
    length: int
    digest: bytes

    def supplied(self) -> dict[str, list[int]]:
        """Its supplied values' one row each: the start's chaining value and pending bytes (none when nothing pends) and the
        digest it ends at, each as its port's words (`interface.digest_words`, `sha512_stream.carry_words`)."""
        return {f"{self.name}/start": IF.digest_words(b"".join(v.to_bytes(8, "big") for v in self.cv)),
                **({f"{self.name}/start-carry": SS.carry_words(self.pending)} if self.pending else {}),
                f"{self.name}/sha512": IF.digest_words(self.digest)}


@dataclass(frozen=True)
class Tie:
    """X's tie at a grain, from its public bytes and the verifier's weights record alone (`tie`): the record's ``program``
    digest, the registered ``model``, the ``grain``, each shard's pinned SHA-256, each rooted weight parameter's shape entry
    (``entries``: value, leaves, words, domain; no root) and the parameters with no root (an analytic table's among them),
    and its ``chains``; ``run_id`` names the run."""
    program: str
    model: dict[str, str]
    grain: dict[str, int]
    shards: dict[str, str]
    entries: dict[str, dict[str, Any]]
    unrooted: dict[str, str]
    chains: tuple[Chain, ...]
    run_id: str = ""

    @property
    def run(self) -> str:
        """The run's root: SHA-512 of the tie's identity (the record's program, the model, the grain, the shards' pins) and
        its id."""
        return hashlib.sha512(json.dumps({"tie": TIE_TAG, "program": self.program, "model": self.model, "grain": self.grain,
                                          "shards": self.shards, "id": self.run_id}, sort_keys=True).encode()).hexdigest()

    def _plan(self) -> tuple[list[tuple[Any, int]], list[list[tuple[Any, str, int, int, str, int]]]]:
        """The templates ``(Definition, units)`` in first use, and per chain its units ``(Definition, template id, index,
        pending bytes, parameter, row)`` in stream order, its last a `Sha512StreamLast_v1`."""
        fns: dict[str, Any] = {}
        count: dict[str, int] = {}
        plan = []
        for c in self.chains:
            h, steps = len(c.pending), []
            for k, (p, j) in enumerate(c.rows):
                w = self.entries[p]["words"]
                fn = SS.last(h, w, c.length) if k == len(c.rows) - 1 else SS.row(h, w)
                tid = IF.descriptor_id(fn)
                fns.setdefault(tid, fn)
                count[tid] = count.get(tid, 0) + 1
                steps.append((fn, tid, count[tid] - 1, h, p, j))
                h = (h + 2 * w) % SS.BLOCK
            plan.append(steps)
        return [(fns[t], count[t]) for t in fns], plan

    def circuit(self) -> IF.Circuit:
        """The tie's circuit: each chain's first unit reads its start state, every other unit's state is wired from the unit
        before (``cv`` from ``o0``, ``carry`` from ``o1``), each unit's ``data`` reads its row of its parameter, and the last
        unit's ``o0`` reads the chain's digest; the patterns coalesced to their runs. An embedding's table is declared a table
        of its parameter's every row, so its reads are row-seg rows (`interface.seg_ports`)."""
        templates, plan = self._plan()
        wires, reads = [], []
        for c, steps in zip(self.chains, plan):
            prev = None
            for _fn, tid, i, h, p, j in steps:
                if prev is None:
                    reads.append(IF.Read(tid, "cv", f"{c.name}/start", i, (1,), 0, (0,)))
                    if h:
                        reads.append(IF.Read(tid, "carry", f"{c.name}/start-carry", i, (1,), 0, (0,)))
                else:
                    wires.append(IF.Wire(tid, "cv", prev[0], "o0", i, (1,), prev[1], (0,)))
                    if h:
                        wires.append(IF.Wire(tid, "carry", prev[0], "o1", i, (1,), prev[1], (0,)))
                reads.append(IF.Read(tid, "data", p, i, (1,), int(j) * self.entries[p]["words"], (0,)))
                prev = (tid, i)
            reads.append(IF.Read(prev[0], "o0", f"{c.name}/sha512", prev[1], (1,), 0, (0,)))
        params = {**{p: e["leaves"] * e["words"] for p, e in self.entries.items()},
                  **{n: len(row) for c in self.chains for n, row in c.supplied().items()}}
        tables = tuple(IF.Table(p, p, 0, e["words"], e["leaves"]) for p, e in self.entries.items() if _table(p))
        return IF.Circuit(NAME, tuple(templates), wires=tuple(IF.coalesce(wires)),
                          registered=IF.Registered(params, tables, reads=tuple(IF.coalesce(reads))))

    def window(self, law: dict[str, Any], cap: int, values: dict[str, G.Registered] | None = None) -> IF.Window:
        """The tie's window under ``law`` in members of at most ``cap`` units, ``values`` the committed weights (the
        committer's; none for the verifier's empty window), and its supplied values (`supplied_values`). Under a ``work``
        law every template is credited alike (work 1)."""
        c = self.circuit()
        table = {tid: 1 for tid in c.ids} if law.get("law") == "work" else None
        win = IF.Window(c, self.run, law, cap, registered=values, work_table=table)
        win.registered = {**(values or {}), **self.supplied_values(win.window.partition_digest)}
        return win

    def supplied_values(self, program: str) -> dict[str, G.Registered]:
        """Each chain's supplied values committed as core's ``registered.Registered`` under ``program`` (the window's
        partition digest) at their public salts, so the committer and the verifier make the same value."""
        return {n: G.Registered.commit(program, n, [row], G.supplied_salts(program, n, 1), schema=G.SCHEMA_V2)
                for c in self.chains for n, row in c.supplied().items()}

    def digest(self, law: dict[str, Any], cap: int) -> str:
        """The SHA-512 of the tie's circuit record, the one the verifier holds the window's registration to."""
        return hashlib.sha512(IF.circuit_bytes(self.window(law, cap).circuit_record())).hexdigest()

    def units(self) -> int:
        return sum(len(c.rows) for c in self.chains)

    def outputs(self) -> list[list[int]]:
        """The output class the audit proves whole: every unit, since a unit left unproved breaks its chain."""
        return [[0, self.units()]]

    def record(self, win: IF.Window, rows: dict[str, np.ndarray], forge_digest: bool = False) -> None:
        """Record every unit into ``win``, each row unit's data its parameter's row of ``rows`` (by parameter, ``(leaves,
        words)`` u16) and its states as `sha512_stream` computes them, each template's units at once. ``forge_digest``: each
        last unit's digest the chain's, whatever its rows hash to (a forger's, which that unit's proof refuses)."""
        templates, plan = self._plan()
        got: dict[str, tuple[list, list]] = {IF.descriptor_id(fn): ([], []) for fn, _ in templates}
        for c, steps in zip(self.chains, plan):
            cv, pend = list(c.cv), c.pending
            for k, (_fn, tid, _i, h, p, j) in enumerate(steps):
                data = np.asarray(rows[p][j], np.uint16)
                ins = [np.array(cv, np.uint64), *([np.array(SS.carry_words(pend), np.uint16)] if h else []), data]
                cv, pend = SS.row_reference(cv, pend, data.tolist())
                if k == len(steps) - 1:
                    d = c.digest if forge_digest else SS.end_reference(cv, pend, c.length)
                    outs = [np.frombuffer(d, ">u8").astype(np.uint64)]
                else:
                    outs = [np.array(cv, np.uint64), *([np.array(SS.carry_words(pend), np.uint16)] if pend else [])]
                got[tid][0].append(ins)
                got[tid][1].append(outs)
        for fn, _n in templates:
            ins, outs = got[IF.descriptor_id(fn)]
            win.record(fn, [np.stack(x) for x in zip(*ins)], [np.stack(x) for x in zip(*outs)])


def tie(registered: Any, checkpoint_dir: str, checkpoints_manifest: str | None = None, *, run_id: str = "",
        grain: Grain = _GRAIN, kv_cache_dtype: str = "auto", record: dict[str, Any] | None = None) -> Tie:
    """X's tie (`Tie`, module docstring) from its shards and the verifier's weights record of them (``record``, default
    `window_weights.weights_record`'s, at ``registered``'s model and revision): refused (ValueError) by name unless X is of
    record and the registered model's, every rooted parameter's field is its source tensors' bytes in order, uncast and
    unsharded, in whole rows, every byte of every shard after its header is a row of a registered parameter, and each
    shard's SHA-256 is its pinned one. An analytic table's parameters it leaves unrooted. It commits no weight."""
    rec = record or WW.weights_record(registered, checkpoint_dir, checkpoints_manifest, kv_cache_dtype, grain)
    if rec.get("of_record") is not True:
        raise ValueError(f"the checkpoint is not of record ({rec.get('why') or rec['checkpoint'].get('all_shards_pinned')}): "
                         "a tie is of the pinned shards")
    model, entry = registered.phi.model, rec["checkpoint"]["entry"]
    if (entry["repo"], entry["revision"]) != (model["repo"], model["revision"]):
        raise ValueError(f"the checkpoint's manifest entry is {entry['repo']}@{entry['revision']}, not the registered model "
                         f"{model['repo']}@{model['revision']}")
    ck = WR.CheckpointIndex(checkpoint_dir, verify_shards=False)
    fields = {f["field"]: f for f in rec["fields"]}
    entries = {p: e for p, e in rec["registered_entries"].items() if fields[_field(p)].get("pinned") != "analytic"}
    of: dict[str, list[str]] = {}
    for p in sorted({*rec["registered_entries"], *rec["unrooted"]}):
        of.setdefault(_field(p), []).append(p)
    claimed: dict[str, tuple[str, int]] = {}
    alone: list[tuple[str, list[str]]] = []
    for p in sorted(entries, key=lambda p: (_table(p), p)):
        e, f = entries[p], fields[_field(p)]
        srcs, row, at, span = list(f.get("sources") or ()), 2 * e["words"], 0, []
        if f.get("cast") or f.get("shard") or not srcs or not _whole(f.get("rule")):
            raise ValueError(f"{p}: its field is {f.get('rule')!r} (cast {f.get('cast')!r}, shard {f.get('shard')!r}); a tie "
                             "reads a field whose rows are its source tensors' bytes in order (1:1, fused along dim 0 or "
                             "tied), uncast and unsharded")
        if row < 112:
            raise ValueError(f"{p}: its rows are {row} bytes; C-Flock proves a unit only with a row of at least 112 bytes "
                             "(one that compresses before its last block), so a tie reads rows of at least 56 words")
        for s in srcs:
            n = ck.tensors[s][4] - ck.tensors[s][3]
            if at % row or n % row:
                raise ValueError(f"{p}: tensor {s} is the field's bytes [{at}, {at + n}), which straddle its rows of {row} bytes")
            span.append((s, at // row))
            at += n
        if at != row * e["leaves"]:
            raise ValueError(f"{p}: its sources are {at} bytes, its {e['leaves']} rows {row * e['leaves']}")
        if not any(s in claimed for s in srcs):
            claimed |= {s: (p, j) for s, j in span}
        elif all(s in claimed for s in srcs):
            alone.append((p, srcs))
        else:
            raise ValueError(f"{p}: its tensors {srcs} are partly another parameter's ({[s for s in srcs if s in claimed]})")
    chains, shards = [], {}
    for fn in ck.shards:
        path = os.path.join(checkpoint_dir, fn)
        base = read_safetensors_header(path)[1]
        with open(path, "rb") as fh:
            cv, pend = SS.state(fh.read(base))
        rows, at = [], base
        for name, (_sh, _dt, _shape, lo, hi) in sorted(((n, t) for n, t in ck.tensors.items() if t[0] == fn), key=lambda x: x[1][3]):
            if lo != at:
                raise ValueError(f"{fn}: bytes [{at}, {lo}) between its tensors; a tie hashes a shard's tensors' rows alone")
            if name not in claimed:
                raise ValueError(f"{fn}: tensor {name} is no registered weight parameter's rows")
            p, j0 = claimed[name]
            rows += [(p, j0 + k) for k in range((hi - lo) // (2 * entries[p]["words"]))]
            at = hi
        if not rows:
            raise ValueError(f"{fn}: no tensor; a shard's chain ends on its last row")
        sha256, sha512 = _digests(path)
        if os.path.getsize(path) != at:
            raise ValueError(f"{fn}: bytes [{at}, {os.path.getsize(path)}) after its last tensor")
        if sha256 != rec["checkpoint"]["shards"][fn]["pinned_sha256"]:
            raise ValueError(f"{fn}: its SHA-256 is not the one the manifest pins")
        shards[fn] = sha256
        chains.append(Chain(f"tie/{fn}", tuple(cv), pend, tuple(rows), at, sha512))
    for p, srcs in alone:
        tag = TABLE_TAG if _table(p) else b""
        data = tag + b"".join(ck.array(s).tobytes() for s in srcs)
        cv, pend = SS.state(tag)
        chains.append(Chain(f"tie/{p}", tuple(cv), pend, tuple((p, j) for j in range(entries[p]["leaves"])), len(data),
                            hashlib.sha512(data).digest()))
    return Tie(rec["program_digest"], {"repo": model["repo"], "revision": model["revision"]}, dict(rec["grain"]), shards,
               {p: dict(e) for p, e in sorted(entries.items())}, WW._unrooted(rec, of, entries), tuple(chains), run_id)


def commit(registered: Any, key: bytes, checkpoint_dir: str, checkpoints_manifest: str | None = None, *, law: dict[str, Any],
           cap: int, run_id: str = "", served: dict[str, np.ndarray] | None = None, grain: Grain = _GRAIN,
           kv_cache_dtype: str = "auto", forge_digest: bool = False) -> tuple[dict[str, Any], Tie, IF.Window]:
    """The committer's weights registration of X under its 32-byte ``key`` (`records.weights_registration`: each rooted
    weight parameter's `window_weights.weight_value`, the roots `weights_registration` registers) and its tie window of X,
    every unit recorded: -> (the registration, the tie, the window, to close and write). ``served``: the words it serves
    by field (`window_weights.record_words`' form; default X's own), which the registration commits, so a committer serving
    another checkpoint's words as X's (the refused arm) registers their roots; ``forge_digest``: its last units' digests X's."""
    t = tie(registered, checkpoint_dir, checkpoints_manifest, run_id=run_id, grain=grain, kv_cache_dtype=kv_cache_dtype)
    words = served if served is not None else WW.record_words(registered, checkpoint_dir, checkpoints_manifest, kv_cache_dtype)
    values = {p: v for p, v in WW.weight_values(registered, words, key, grain, keep_rows=True).items() if p in t.entries}
    if missing := sorted(set(t.entries) - set(values)):
        raise ValueError(f"no words served for {missing}")
    registration = WeightsRegistration(program=t.program, repo=t.model["repo"], revision=t.model["revision"], grain=t.grain,
                                       values={p: v.entry() for p, v in values.items()}, unrooted=t.unrooted).dump()
    win = t.window(law, cap, values)
    t.record(win, {p: WW._rows(p, words[_field(p)]) for p in t.entries}, forge_digest)
    return registration, t, win


def audit_inputs(registration: Any, record: dict[str, Any], t: Tie, law: dict[str, Any], cap: int) -> dict[str, Any]:
    """What the verifier hands the records audit of the committer's tie window: ``registered``, its ``--registered`` file
    (the registration's entries, held to the verifier's weights record's shapes by `window_weights.registration_pins`,
    and the tie's supplied values under the window's partition digest); ``outputs``, every unit; ``circuit`` and ``run``,
    the tie's circuit record's SHA-512 and run root; ``units``. Refused (ValueError) where the registration isn't the
    record's."""
    WW.registration_pins(registration, record)
    reg = WeightsRegistration.of(registration, "the weights registration")
    win = t.window(law, cap)
    supplied = {n: v.entry() for n, v in t.supplied_values(win.window.partition_digest).items()}
    return {"circuit": hashlib.sha512(IF.circuit_bytes(win.circuit_record())).hexdigest(), "run": t.run,
            "units": t.units(), "outputs": t.outputs(),
            "registered": [{"program": reg.program, "values": {p: dict(e) for p, e in reg.values.items()}},
                           {"program": win.window.partition_digest, "values": supplied}]}


def checked_pins(registration: Any, record: dict[str, Any], t: Tie, summary: dict[str, Any], law: dict[str, Any],
                 cap: int) -> dict[str, str]:
    """The verifier's pins of the registration's roots (`window_weights.registration_pins`), only once the records audit
    of its tie window accepted it: refused (ValueError) unless ``summary`` (``records_audit.py audit``'s ``summary.json``)
    is an accepting audit, not a forged arm's, of the verifier's own tie circuit and run root, that proved every unit and
    held the registration's root for every weight, as the pin and as the window's root."""
    pins, want = WW.registration_pins(registration, record), audit_inputs(registration, record, t, law, cap)
    if summary.get("ok") is not True or "forged" in summary:
        raise ValueError("the records audit of the tie didn't accept it: no root is pinned")
    if (summary.get("circuit"), summary.get("run_root")) != (want["circuit"], want["run"]):
        raise ValueError("the records audit was of another circuit or run than the verifier's tie of the checkpoint")
    outs, held = summary.get("outputs"), summary.get("registered")
    if not isinstance(outs, dict) or outs.get("units") != want["units"] or outs.get("proved") is not True:
        raise ValueError(f"the records audit proved {outs} of the tie's {want['units']} units, not every one")
    for what in ("pins", "roots"):
        got = held.get(what) if isinstance(held, dict) else None
        if not isinstance(got, dict):
            raise ValueError(f"the records audit held no registered {what}: no root is pinned")
        if bad := sorted(p for p, r in pins.items() if str(got.get(p, "")).lower() != r.lower()):
            raise ValueError(f"the records audit's {what} for {bad} aren't the registration's roots")
    return pins


def _registered(phi: str, default_seed: int) -> Any:
    from .phi import Phi
    from .steps import Registered

    return Registered.of(Phi.from_json(json.loads(Path(phi).read_text())), default_seed)


def _key(path: str) -> bytes:
    """The committer's 32-byte key from ``path``, drawn fresh and written 0600 when the file is absent."""
    p = Path(path)
    if not p.exists():
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(os.urandom(32))
    return p.read_bytes()


@dataclass(frozen=True, kw_only=True)
class Checkpoint:
    """X, the registered φ and the tie's window."""
    phi: str = option("--phi", help="the registered φ's JSON (`Phi.to_json`)")
    default_seed: int = option("--default-seed", type=int, default=7)
    checkpoint_dir: str = option("--checkpoint-dir", help="checkpoint X's safetensors shards and config.json")
    manifest: str | None = option("--manifest", default=None, help="the checkpoints manifest pinning X (default the repository's)")
    law: str = option("--law", default="work:1")
    cap: int = option("--cap", type=int, default=16)
    run_id: str = option("--run-id", default="")
    grain: str | None = option("--grain", default=None, help="gemm_words,rows,select (default the window grain's)")


@dataclass(frozen=True, kw_only=True)
class CommitOptions:
    """The committer's weights registration of X and its tie window, every unit recorded."""
    x: Checkpoint = group(Checkpoint)
    key_file: str = option("--key-file", help="the committer's 32-byte key, drawn and written 0600 when absent")
    registration_out: str = option("--registration-out")
    out: str = option("--out", help="the tie window's directory")
    served_dir: str | None = option("--served-dir", default=None, help="serve this checkpoint's words as X's (the refused arm)")
    served_manifest: str | None = option("--served-manifest", default=None)
    forge_digest: bool = option("--forge-digest", default=False, action="store_true",
                                help="the last units' digests forged to X's (with --served-dir)")


@dataclass(frozen=True, kw_only=True)
class VerifyOptions:
    """The verifier's tie of X and the records audit's --registered and --outputs files for the committer's window."""
    x: Checkpoint = group(Checkpoint)
    registration: str = option("--registration")
    out: str = option("--out")


@dataclass(frozen=True, kw_only=True)
class PinsOptions:
    """The registration's roots, written only once the records audit accepted the tie (exit 1 where it didn't)."""
    x: Checkpoint = group(Checkpoint)
    registration: str = option("--registration")
    out: str = option("--out")
    summary: str = option("--summary", help="records_audit.py audit's summary.json of the tie window")


COMMANDS = {"commit": CommitOptions, "verify": VerifyOptions, "pins": PinsOptions}


def main(a: CommitOptions | VerifyOptions | PinsOptions) -> int:
    from verity_vllm.config import CHECKPOINTS

    if isinstance(a, CommitOptions) and a.forge_digest and not a.served_dir:
        raise UsageError("--forge-digest forges a committer's that serves other words: give --served-dir")
    x = a.x
    manifest = x.manifest or str(CHECKPOINTS)
    grain = Grain(*(int(g) for g in x.grain.split(","))) if x.grain else _GRAIN
    registered, law = _registered(x.phi, x.default_seed), D.parse_law(x.law)
    if isinstance(a, CommitOptions):
        served = WW.record_words(registered, a.served_dir, a.served_manifest) if a.served_dir else None
        registration, t, win = commit(registered, _key(a.key_file), x.checkpoint_dir, manifest, law=law, cap=x.cap,
                                      run_id=x.run_id, served=served, grain=grain, forge_digest=a.forge_digest)
        bad = win.mismatches(kernel="auto")
        Path(a.registration_out).write_text(json.dumps(registration, indent=1, sort_keys=True) + "\n")
        win.write(win.close(), a.out, {"tie": TIE_TAG, "program": t.program, "model": t.model, "shards": t.shards}, registrar=R)
        rec = json.loads((Path(a.out) / "registration.json").read_text())
        print(json.dumps({"circuit": rec["window"]["circuit"], "run": t.run, "units": t.units(), "members": len(win.window.members),
                          "mismatched": {k: len(v) for k, v in bad.items()}, "registration": R.digest(rec),
                          "roots": WeightsRegistration.of(registration, a.registration_out).roots}))
        return 0
    record = WW.weights_record(registered, x.checkpoint_dir, manifest, grain=grain)
    t = tie(registered, x.checkpoint_dir, manifest, run_id=x.run_id, grain=grain, record=record)
    registration = json.loads(Path(a.registration).read_text())
    out = Path(a.out)
    if isinstance(a, VerifyOptions):
        got = audit_inputs(registration, record, t, law, x.cap)
        out.mkdir(parents=True, exist_ok=True)
        for f in ("registered", "outputs"):
            (out / f"{f}.json").write_text(json.dumps(got[f], indent=1, sort_keys=True) + "\n")
        shown = {k: got[k] for k in ("circuit", "run", "units")}
        (out / "tie.json").write_text(json.dumps(shown, indent=1, sort_keys=True) + "\n")
        print(json.dumps(shown))
        return 0
    try:
        pins = checked_pins(registration, record, t, json.loads(Path(a.summary).read_text()), law, x.cap)
    except ValueError as e:
        print(json.dumps({"pinned": False, "why": str(e)}))
        return 1
    out.write_text(json.dumps(pins, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"pinned": True, "weights": len(pins)}))
    return 0


__all__ = ["NAME", "TIE_TAG", "Chain", "Tie", "audit_inputs", "checked_pins", "commit", "tie"]
