"""verity_vllm.observe.contract -- ONE value-observation contract over today's on-disk capture formats (v2 Phase 4b).

A *runtime occurrence* is WHERE a value was seen: engine step, invocation (the k-th time the site ran inside the step), tensor-parallel
rank, the site (an implementation module path such as `model.layers.3.self_attn.qkv_proj`, or a native tap id such as `fa2.m1/L3`), the
slot at that site (`out`, tuple slot `0`/`1`, `in` = the consumer-side acquisition of a promoted interior value, a hidden plane name), the
buffer view the collector observed, and -- when the runtime said so -- the request and the token position of the row.

A *value observation* is an occurrence together with what was observed (bytes, or their digest; a fold record carries neither -- it is the
observation of a STRUCTURE, `payload_or_digest` is None) and an optional `hint_value_id`: the COLLECTOR'S index for the value (the fold's
root node id, a committed leaf name).  The hint is a bookkeeping handle, never an identity: `resolve.resolve` derives the identity from
the occurrence alone and `resolve.validate` compares the two, reporting a disagreement as a named fault.

`iter_observations(record_dir)` adapts the v1 Match record (`<row>/match/` = program.json + instances.jsonl + capture/log.jsonl.gz, or that
directory itself) into a STREAM of ValueObservation -- one per instance row, in node order -- without loading the record: the header line
is read for the request table and the operand form, and each instances.jsonl line is decoded, adapted and released.  An observation states
only what the runtime stated (a row's stamped request / position: the meta channel); the row-count rule over the engine step's `StepStart`
rows (`read_steps`: `BD.read_steps` over the capture log, O(steps)) is the rows channel of `resolve`, not the adapter's.  Nothing here
writes, and nothing changes what Match writes (v1 stays the producer of record; this is a reader beside it).

Dependency-free (json / gzip / dataclasses).
"""
from __future__ import annotations

import gzip
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterator

SCHEMA = "verity-vllm/value-observation/v1"

#: the fold's collective meta (TP rank folds) carries the rank; a TP1 record has none
_TAG_RE = re.compile(r'^\{"\$e":\s*"(\w+)"')
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$|/)")


@dataclass(frozen=True)
class RuntimeOccurrence:
    """WHERE a value was observed.  `site` is an implementation module path or a native tap id; `slot` the member at that site
    (`out` | `0` | `1` | `in` | plane).  `row` / `rows` are the runtime's batch coordinates of the observation (row `row` of a launch
    over `rows` rows -- the fold's own coordinates; they let the row-count rule attribute an unstamped row).  `request` / `position`
    are set only when the runtime STATED them (the fold's meta channel); None means "not stated", never "none"."""

    step: int
    invocation: int
    rank: int
    site: str
    slot: str
    view: tuple | None = None
    request: str | None = None
    position: int | None = None
    row: int | None = None
    rows: int | None = None

    def layer(self) -> int | None:
        m = _LAYER_RE.search(self.site)
        return int(m.group(1)) if m else None


@dataclass(frozen=True)
class ValueObservation:
    """An occurrence + WHAT was observed.  `payload_or_digest`: bytes, a hex digest, or None (a structural record).  `hint_value_id` is the
    collector's own index for the value (the fold node id / a committed leaf name) -- a hint for `resolve.validate`, not an identity.
    `spec` (the Definition the fold assigned), `reads` (root node ids this value's producer read -- the dataflow channel's input) and
    `param_reads` (root parameter indices read) are the fold's STRUCTURAL statements about the value, carried for resolution."""

    occurrence: RuntimeOccurrence
    payload_or_digest: bytes | str | None = None
    hint_value_id: int | str | None = None
    spec: str | None = None
    name: str | None = None
    reads: tuple[int, ...] = ()
    param_reads: tuple[int, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)


@dataclass
class RecordHeader:
    """The instances.jsonl header line (the fold's own statement about the record) + program.json's parameter layout."""

    schema: str | None
    operand_refs_form: str
    n_instances: int | None
    requests: list[dict]                 # [{index, request_id, prompt_len, served_samples, max_tokens}] in prompt-parameter order
    n_input_nodes: int | None
    params: list[str]                    # program.json `params` names when stated (weights LAST by the fold contract)
    tp: dict | None                      # program.json `tp` ({"rank", "world"}) for a rank fold
    digest: str | None

    @property
    def weights_param(self) -> int | None:
        return len(self.params) - 1 if self.params else (self.n_input_nodes - 1 if self.n_input_nodes else None)


# ------------------------------------------------------------------------------------------------------------------ the record on disk


def match_dir(record_dir: str) -> str:
    """`<row>/match` when `record_dir` is a row dir, else `record_dir` itself (a fold dir: program.json + instances.jsonl)."""
    if os.path.isfile(os.path.join(record_dir, "instances.jsonl")):
        return record_dir
    m = os.path.join(record_dir, "match")
    if os.path.isfile(os.path.join(m, "instances.jsonl")):
        return m
    raise FileNotFoundError(f"{record_dir}: neither it nor its match/ holds an instances.jsonl")


def read_header(record_dir: str) -> RecordHeader:
    d = match_dir(record_dir)
    hdr: dict = {}
    with open(os.path.join(d, "instances.jsonl"), "rb") as f:
        first = f.readline().strip()
    if first:
        o = json.loads(first)
        if "schema" in o and "node" not in o:
            hdr = o
    n_in, params, tp, digest = None, [], None, hdr.get("digest")
    pj_path = os.path.join(d, "program.json")
    if os.path.isfile(pj_path):
        # program.json can be GBs (the descriptor); read only the small top-level scalars with a bounded scan of its head
        n_in, params, tp, digest = _program_json_layout(pj_path, digest)
    return RecordHeader(schema=hdr.get("schema"), operand_refs_form=hdr.get("operand_refs_form") or "runs", n_instances=hdr.get("n_instances"),
                        requests=list(hdr.get("requests") or []), n_input_nodes=n_in, params=params, tp=tp, digest=digest)


def _program_json_layout(path: str, digest: str | None) -> tuple[int | None, list[str], dict | None, str | None]:
    """`n_input_nodes`, `params` names, `tp`, `digest` of program.json WITHOUT decoding the (possibly multi-GB) `program` descriptor.
    The writer emits the small keys around the descriptor; the head and tail of the file are scanned for them with a bounded window,
    falling back to a full parse only when the file is small."""
    size = os.path.getsize(path)
    if size <= 64 << 20:
        pj = json.load(open(path))
        return pj.get("n_input_nodes"), [n for n, _t in (pj.get("params") or [])], pj.get("tp"), pj.get("digest") or digest
    n_in, params, tp = None, [], None
    with open(path, "rb") as f:
        head = f.read(1 << 20).decode("utf-8", "replace")
        f.seek(max(0, size - (1 << 20)))
        tail = f.read().decode("utf-8", "replace")
    for text in (head, tail):
        m = re.search(r'"n_input_nodes":\s*(\d+)', text)
        if m and n_in is None:
            n_in = int(m.group(1))
        m = re.search(r'"digest":\s*"([0-9a-f]{128})"', text)
        if m and digest is None:
            digest = m.group(1)
        m = re.search(r'"params":\s*(\[\[.*?\]\])', text, re.S)
        if m and not params:
            try:
                params = [n for n, _t in json.loads(m.group(1))]
            except Exception:  # noqa: BLE001 -- a window cut through the list: stated as unknown
                params = []
        m = re.search(r'"tp":\s*(\{[^{}]*\})', text)
        if m and tp is None:
            try:
                tp = json.loads(m.group(1))
            except Exception:  # noqa: BLE001
                tp = None
    return n_in, params, tp, digest


def read_steps(record_dir: str) -> tuple[dict, dict[int, dict]]:
    """(RunHeader, {engine step: {rows, logits_rows, logits_indices}}) of the record's capture log -- `batch_decomp.read_steps`, the
    fold's own chronology (the row-count rule reads it)."""
    from verity_vllm.check.match import batch_decomp as BD
    d = match_dir(record_dir)
    log = os.path.join(d, "capture", "log.jsonl.gz")
    if not os.path.isfile(log):
        log = os.path.join(d, "capture", "log.jsonl")
    if not os.path.isfile(log):
        raise FileNotFoundError(f"{d}: no capture/log.jsonl(.gz)")
    return BD.read_steps(log)


# ------------------------------------------------------------------------------------------------------------------ adapting one row


def _view(v: Any) -> tuple | None:
    """A recorded view (`operand_views` / `output_views` entries: dict or list) as a hashable tuple (shape, strides, offset, dtype) --
    None when the row carries none."""
    if v is None:
        return None
    if isinstance(v, dict):
        return (tuple(v.get("shape") or ()), tuple(v.get("strides") or ()), v.get("offset"), v.get("dtype"))
    if isinstance(v, (list, tuple)):
        return tuple(tuple(x) if isinstance(x, list) else x for x in v)
    return (str(v),)


def view_layout(view: tuple | None) -> tuple | None:
    """The LAYOUT of a view -- (shape, strides, dtype) -- without its storage offset: the offset is the row's address inside the launch's
    buffer (row k of an H-wide output sits at k*H) and moves with every row of a launch; a mutated view is a changed shape / stride / dtype."""
    if view is None:
        return None
    if len(view) == 4:
        return (view[0], view[1], view[3])
    return tuple(view)


def site_and_slot(hint: str | None, name: str | None, spec: str | None) -> tuple[str, str]:
    """(site, slot) of a fold row.  The fold's `hint` is `<module path>/<memory root>` (`model.layers.0.self_attn.qkv_proj/weight`,
    `.../rotary_emb/rope.q`) -- the module path is the site; the value observed is the instance's OUTPUT (`out`) unless the name's tail
    says a tuple slot.  A row without a module hint (the sampler, a constant) is sited at its name's tail."""
    h = hint or ""
    if ":" in h and "/" not in h.split(":", 1)[0]:
        h = h.split(":", 1)[0]          # a Const row's hint is `<compute hint>: <tail>`
    mod, _, tail = h.rpartition("/")
    if not mod:
        mod = tail or (name or "").split("/")[-1] or (spec or "").split("{")[0] or "?"
    slot = "out"
    if name:
        last = name.rsplit("/", 1)[-1]
        if last in ("0", "1", "in", "sampled_token_ids"):
            slot = last
    return mod, slot


def _reads(operand_refs: Any, form: str, n_input_nodes: int | None) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """(root node ids read, root parameter indices read) of a row's `operand_refs` in either stored form: runs `[space, idx, lo, hi]` or
    [R17 mfast F2] node-space progressions `["N", idx, lo, hi, dI, dL, dH, count]` (only the DISTINCT node ids matter to resolution)."""
    nodes: set[int] = set()
    params: set[int] = set()
    for a in operand_refs or ():
        for r in a or ():
            if not r:
                continue
            sp = r[0]
            if sp == "p":
                params.add(int(r[1]))
            elif sp == "n":
                nodes.add(int(r[1]))
            elif sp == "N":
                idx, dI, c = int(r[1]), int(r[4]), int(r[7])
                if dI == 0 or c <= 1:
                    nodes.add(idx)
                else:
                    nodes.update(idx + k * dI for k in range(c))
    return tuple(sorted(nodes)), tuple(sorted(params))


def adapt_row(o: dict, hdr: RecordHeader, *, invocation: int = 0) -> ValueObservation:
    """One instances.jsonl row -> ValueObservation.  `request` / `position` are the row's META channel -- what the runtime STAMPED on the
    row (the fold's `_attribution` at emission) -- and nothing else: an unstamped row is observed with request None, and the row-count
    rule over the engine step's StepStart rows is `resolve`'s rows channel (the Match stage's channel order), never the adapter's guess."""
    step = int(o["step"]) if o.get("step") is not None else -1
    site, slot = site_and_slot(o.get("hint"), o.get("name"), o.get("spec"))
    req, pos = o.get("request"), o.get("position")
    row, rows = o.get("row"), o.get("rows")
    coll = o.get("collective") or {}
    rank = int(coll.get("rank")) if coll.get("rank") is not None else int((hdr.tp or {}).get("rank", 0) or 0)
    outs = o.get("output_views") or []
    occ = RuntimeOccurrence(step=step, invocation=int(invocation), rank=rank, site=site, slot=slot, view=_view(outs[0]) if outs else None,
                            request=(str(req) if req is not None else None), position=(int(pos) if pos is not None else None),
                            row=(int(row) if row is not None else None), rows=(int(rows) if rows is not None else None))
    reads, preads = _reads(o.get("operand_refs"), hdr.operand_refs_form, hdr.n_input_nodes)
    extra = {k: o[k] for k in ("event", "resolution", "pattern", "operand_roots", "collective", "const") if k in o}
    return ValueObservation(occurrence=occ, payload_or_digest=None, hint_value_id=o.get("node"), spec=o.get("spec"), name=o.get("name"),
                            reads=reads, param_reads=preads, extra=extra)


# ------------------------------------------------------------------------------------------------------------------ the stream


def iter_rows(record_dir: str) -> Iterator[dict]:
    """The instances.jsonl rows (header line skipped), decoded one at a time."""
    d = match_dir(record_dir)
    with open(os.path.join(d, "instances.jsonl"), "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if "node" not in o:
                continue      # the header
            yield o


def iter_observations(record_dir: str, *, header: RecordHeader | None = None) -> Iterator[ValueObservation]:
    """Stream the record's value observations in node order (one per instances.jsonl row; the record is never loaded).  `invocation` =
    the ordinal of the row's site among the LAUNCHES of that site seen so far in the same engine step (the collector's hook order = the
    runtime's call order inside the step; a launch over `rows` rows emits `rows` fold rows and row 0 opens a new invocation)."""
    hdr = header if header is not None else read_header(record_dir)
    cur_step: int | None = None
    seen: dict[str, int] = {}
    for o in iter_rows(record_dir):
        s = o.get("step")
        if s != cur_step:
            cur_step, seen = s, {}
        site, _slot = site_and_slot(o.get("hint"), o.get("name"), o.get("spec"))
        if (o.get("row") or 0) == 0:
            seen[site] = seen.get(site, -1) + 1
        yield adapt_row(o, hdr, invocation=seen.get(site, 0))


def count_observations(record_dir: str) -> int:
    return sum(1 for _ in iter_rows(record_dir))
