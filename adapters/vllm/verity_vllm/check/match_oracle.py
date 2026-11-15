"""The Match record's values, read one identity at a time.

`MatchOracle` indexes the Match stage's record of one attempt (the capture's snapshots, log and tokens, and the fold's
instances) and reads, lazily per step and file, the bytes the Program produced for an identity: `member_bytes` for a member
of an op at a step, `ids_bytes` and `sampled_id` for a request's token ids.  `oracle_tie_facts` gives the digests that tie
an oracle to its Match record.  The comparison against the committed bytes is `oracle_compare`'s, whose docstring says why
this record is the reference.  CPU only.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

_ITEMSIZE = {"bfloat16": 2, "float16": 2, "float32": 4, "float64": 8, "int32": 4, "int64": 8, "int16": 2, "int8": 1, "uint8": 1, "bool": 1,
             "float8_e4m3fn": 1, "float8_e5m2": 1}
_ROPE_MEMBER = {"query": "rope.q", "key": "rope.k"}                # RoPE's members (the Build's hints) -> the Match fold's hint suffixes


def _prod(xs) -> int:
    n = 1
    for x in xs:
        n *= int(x)
    return n


def _contiguous(shape, strides) -> bool:
    exp = 1
    for d, s in zip(reversed(list(shape)), reversed(list(strides))):
        if int(d) != 1 and int(s) != exp:
            return False
        exp *= int(d)
    return True


def oracle_tie_facts(match_dir: str) -> dict[str, Any]:
    """The facts that tie an oracle to a Match record: sha256 of match/fold_summary.json, of match/instances.jsonl, of the
    RunHeader line (line 1 of capture/log.jsonl.gz, the created_utc carrier); a missing file is named None (never a placeholder digest),
    and a log that is there but can't be read is None with `run_header_unreadable` saying why."""
    out: dict[str, Any] = {}
    for key, rel in (("fold_summary_sha256", "fold_summary.json"), ("instances_sha256", "instances.jsonl")):
        pth = os.path.join(match_dir, rel)
        out[key] = hashlib.sha256(Path(pth).read_bytes()).hexdigest() if os.path.exists(pth) else None
    lg = os.path.join(match_dir, "capture", "log.jsonl.gz")
    out["run_header_sha256"] = None
    if os.path.exists(lg):
        try:
            with gzip.open(lg, "rb") as f:
                out["run_header_sha256"] = hashlib.sha256(f.readline()).hexdigest()
        except (OSError, EOFError) as e:
            out["run_header_unreadable"] = f"unreadable: capture/log.jsonl.gz ({type(e).__name__}: {e})"[:240]
    return out


class MatchOracle:
    """The Match stage's value record of one attempt, indexed for identity lookups.  CPU only; reads the values lazily per (step, file)."""

    def __init__(self, match_dir: str) -> None:
        self.dir = match_dir
        capture_dir = os.path.join(match_dir, "capture")
        self.snap_path = os.path.join(capture_dir, "snapshots.json")
        self.inst_path = os.path.join(match_dir, "instances.jsonl")
        self.log_path = os.path.join(capture_dir, "log.jsonl.gz")
        for p in (self.snap_path, self.inst_path, self.log_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"match oracle: {p} missing (Match stage of this attempt did not write it)")
        snap = json.loads(Path(self.snap_path).read_text())
        self.snapshot_steps: set[int] = {int(s) for s in snap.get("steps") or []}
        # (event, alloc) -> [entries]; an entry = spanned storage bytes [byte_offset, byte_offset+nbytes) of one alloc, copied right after the call
        self._snap: dict[tuple[int, int], list[dict]] = defaultdict(list)
        for e in snap.get("entries") or []:
            if "sha256" not in e or "file" not in e:
                continue
            v = e["view"]["$v"] if isinstance(e.get("view"), dict) and "$v" in e["view"] else e.get("view")
            alloc = int(v[0]) if isinstance(v, (list, tuple)) else int(v["alloc"])
            self._snap[(int(e["event"]), alloc)].append(e)
        # instances: per (step, op_path) -> list of (request uuid, row, hint, event, operand_views, output_views)
        self.requests: dict[str, str] = {}                      # workload uuid -> r<k>
        self.request_index: dict[str, int] = {}
        self._inst: dict[tuple[int, str], list[dict]] = defaultdict(list)
        self.fold_digest: str | None = None
        with open(self.inst_path) as f:
            hdr = json.loads(f.readline())
            self.fold_digest = hdr.get("digest")
            for r in hdr.get("requests") or []:
                self.requests[str(r["request_id"])] = f"r{int(r['index'])}"
                self.request_index[str(r["request_id"])] = int(r["index"])
            for line in f:
                m = json.loads(line)
                hint = m.get("hint") or ""
                op = hint.split("/", 1)[0] if hint else (m.get("name") or "")
                self._inst[(int(m["step"]), op)].append({"request": m.get("request"), "row": m.get("row"), "position": m.get("position"),
                                                          "hint": hint, "event": int(m["event"]) if m.get("event") is not None else None, "spec": m.get("spec"),
                                                          "operand_views": m.get("operand_views") or [], "output_views": m.get("output_views") or []})
        # capture log: request row order + sampled ids per step (every step), header provenance
        self.step_requests: dict[int, list[str]] = {}
        self.step_rows: dict[int, list[tuple[str, int]]] = {}       # step -> [(request uuid, token position)] in the engine's row order
        self.sampled: dict[int, list[int]] = {}
        self.header: dict[str, Any] = {}
        self.n_steps = 0
        with gzip.open(self.log_path, "rt") as f:
            for line in f:
                k = line[6:22]
                if '"StepStart"' in k or '"SampleEnd"' in k or '"RunHeader"' in k or '"RunEnd"' in k:
                    r = json.loads(line)
                    e = r.get("$e")
                    if e == "StepStart":
                        order: list[str] = []
                        rows = [(str(u), int(p)) for u, p in (r.get("rows") or [])]
                        for rid, _pos in rows:
                            if rid not in order:
                                order.append(rid)
                        self.step_requests[int(r["step"])] = order
                        self.step_rows[int(r["step"])] = rows
                        self.n_steps = max(self.n_steps, int(r["step"]) + 1)
                    elif e == "SampleEnd":
                        self.sampled[int(r["step"])] = [int(x) for x in r.get("sampled_token_ids") or []]
                    elif e == "RunHeader":
                        self.header = {"created_utc": r.get("created_utc"), "profile_id": r.get("profile_id"), "schema": r.get("schema"),
                                       "vllm": (r.get("profile") or {}).get("vllm", {}).get("version") if isinstance(r.get("profile"), dict) else None}
        self._files: dict[str, bytes] = {}
        # per step, every instance's OUTPUT views by storage (alloc -> [(byte lo, byte hi, instance)]), built on
        # demand for the promoted members' dataflow resolution (`_promoted_instances`); a small cache, the compare walks the map by step
        self._outs_by_step: dict[int, dict[int, list[tuple[int, int, dict]]]] = {}
        self._families_at: dict[tuple[int, str], frozenset[str]] = {}
        tok = os.path.join(capture_dir, "tokens.json")
        self.tokens_sha256 = hashlib.sha256(Path(tok).read_bytes()).hexdigest() if os.path.exists(tok) else None
        # the ids of every row of every step: prompt ids + generated ids per request (tokens.json) at the StepStart rows' (request, position)
        # -- the runner's ids buffer is a Root of the capture (prompt_token roots), so it is never in a snapshot; this IS its value record
        self.tokens: dict[str, tuple[list[int], list[int]]] = {}
        if os.path.exists(tok):
            for r in (json.loads(Path(tok).read_text()).get("requests") or []):
                self.tokens[str(r.get("request_id"))] = ([int(x) for x in r.get("prompt_token_ids") or []], [int(x) for x in r.get("generated_token_ids") or []])

    # ------------------------------------------------------------------ provenance
    def provenance(self) -> dict[str, Any]:
        # `tie`: file digests of the Match record this oracle READ (fold_summary.json, instances.jsonl, the
        # RunHeader line of capture/log.jsonl.gz) so the finalizer / a reviewer ties the oracle to the Match stage OF RECORD by fact, without
        # reading the fold: the same attempt's match/ (or the chain-reuse Match dir) must carry equal digests, else the oracle is another Match
        return {"match_dir": self.dir, "capture_created_utc": self.header.get("created_utc"), "capture_profile_id": self.header.get("profile_id"),
                "vllm": self.header.get("vllm"), "fold_digest": self.fold_digest, "snapshot_steps": sorted(self.snapshot_steps), "steps": self.n_steps,
                "tokens_sha256": self.tokens_sha256, "requests": dict(self.requests), "tie": oracle_tie_facts(self.dir)}


    # ------------------------------------------------------------------ workload identity
    def workload_check(self, workload: dict[str, Any] | None) -> dict[str, Any]:
        """Is this Match record THIS workload's?  Compared BY NAME before any value compare: the run's declared requests (workload uuid ->
        r<index>, prompt_len, max_tokens) against the fold header's request table and tokens.json (prompt ids per request; generated <= max_tokens).
        A Match record of another workload (another row's --match-dir) is refused named -- never a 'value mismatch' that happens to point
        the right way.  Returns {"ok", "problems", "compared"}; ok is None when the run passed no workload (not checked, said so)."""
        if not workload:
            return {"ok": None, "problems": ["no workload given: oracle identity not checked"], "compared": 0}
        problems: list[str] = []
        declared = {}
        for i, r in enumerate(workload.get("requests") or []):
            rid = str(r.get("request_id"))
            declared[rid] = {"label": f"r{int(r.get('index', i))}", "prompt_len": r.get("prompt_len"), "max_tokens": r.get("max_tokens")}
        if not declared:
            return {"ok": None, "problems": ["workload declares no requests: oracle identity not checked"], "compared": 0}
        oracle_ids = set(self.requests) | set(self.tokens)
        if set(declared) != oracle_ids:
            only_run, only_oracle = sorted(set(declared) - oracle_ids), sorted(oracle_ids - set(declared))
            problems.append(f"request ids differ: run-only {[d[:8] for d in only_run]}, oracle-only {[d[:8] for d in only_oracle]}")
        for rid, d in declared.items():
            if rid in self.requests and self.requests[rid] != d["label"]:
                problems.append(f"request {rid[:8]}: label {self.requests[rid]} in the oracle != {d['label']} declared")
            if rid in self.tokens:
                prompt, gen = self.tokens[rid]
                if d["prompt_len"] is not None and len(prompt) != int(d["prompt_len"]):
                    problems.append(f"request {d['label']}: oracle prompt has {len(prompt)} ids != declared prompt_len {d['prompt_len']}")
                if d["max_tokens"] is not None and len(gen) > int(d["max_tokens"]):
                    problems.append(f"request {d['label']}: oracle generated {len(gen)} ids > declared max_tokens {d['max_tokens']}")
        return {"ok": not problems, "problems": problems, "compared": len(declared),
                "oracle_requests": len(oracle_ids), "run_requests": len(declared)}

    # ------------------------------------------------------------------ values
    def _file(self, rel: str) -> bytes:
        b = self._files.get(rel)
        if b is None:
            b = Path(os.path.join(self.dir, "capture", rel)).read_bytes()
            if len(self._files) > 64:
                self._files.clear()
            self._files[rel] = b
        return b

    def _view_bytes(self, event: int, view: dict) -> tuple[bytes | None, str | None]:
        """The logical (contiguous) bytes of one storage view at the snapshot of `event`, or (None, why)."""
        alloc, off, shape, strides, dtype = int(view["alloc"]), int(view["offset"]), list(view["shape"]), list(view.get("strides") or []), str(view["dtype"])
        isz = _ITEMSIZE.get(dtype)
        if isz is None:
            return None, f"dtype {dtype} unknown itemsize"
        numel = _prod(shape) if shape else 1
        if not strides or _contiguous(shape, strides):
            lo, hi = off * isz, (off + numel) * isz
        else:
            span = 1 + sum((int(d) - 1) * int(s) for d, s in zip(shape, strides))
            lo, hi = off * isz, (off + span) * isz
        cands = self._snap.get((event, alloc)) or []
        ent = next((e for e in sorted(cands, key=lambda e: (0 if str(e.get("path", "")).startswith("outs") else 1, e.get("path", "")))
                    if int(e["byte_offset"]) <= lo and hi <= int(e["byte_offset"]) + int(e["nbytes"])), None)
        if ent is None:
            return None, f"no snapshot of alloc {alloc} bytes [{lo},{hi}) at event {event} ({len(cands)} entries of that alloc)"
        data = self._file(ent["file"])
        base = int(ent["byte_offset"])
        raw = data[lo - base:hi - base]
        if len(raw) != hi - lo:
            return None, f"snapshot file {ent['file']} short: {len(raw)} of {hi - lo} bytes"
        if strides and not _contiguous(shape, strides):
            import numpy as np
            u8 = np.frombuffer(raw, dtype=np.uint8)
            a = np.lib.stride_tricks.as_strided(u8, shape=tuple(int(d) for d in shape) + (isz,), strides=tuple(int(s) * isz for s in strides) + (1,))
            raw = np.ascontiguousarray(a).tobytes()
        return raw, None

    # ------------------------------------------------------------------ promoted members: the Value's producer, not the hint's prefix
    @staticmethod
    def _view_bytes_range(view: Any) -> tuple[int, int, int] | None:
        """(alloc, byte lo, byte hi) the storage range a fold view spans (contiguous: offset..offset+numel; strided: the span), or None
        when there is no view, its allocation, offset, shape or strides cannot be read, or its dtype has no itemsize: such a view is skipped."""
        try:
            alloc, off = int(view["alloc"]), int(view["offset"])
            shape, strides = list(view.get("shape") or []), list(view.get("strides") or [])
            isz = _ITEMSIZE.get(str(view.get("dtype")))
        except (KeyError, TypeError, ValueError):
            return None
        if isz is None:
            return None
        numel = _prod(shape) if shape else 1
        span = numel if (not strides or _contiguous(shape, strides)) else 1 + sum((int(d) - 1) * int(s) for d, s in zip(shape, strides))
        return alloc, off * isz, (off + span) * isz

    def _step_outputs(self, step: int) -> dict[int, list[tuple[int, int, dict]]]:
        """alloc -> [(byte lo, byte hi, instance)] over the OUTPUT views of every fold instance at `step` (all op paths)."""
        idx = self._outs_by_step.get(step)
        if idx is None:
            idx = defaultdict(list)
            for (s, _op), insts in self._inst.items():
                if s != step:
                    continue
                for m in insts:
                    for ov in m.get("output_views") or []:
                        r = self._view_bytes_range(ov)
                        if r is not None:
                            idx[r[0]].append((r[1], r[2], m))
            if len(self._outs_by_step) >= 2:
                self._outs_by_step.clear()
            self._outs_by_step[step] = idx
        return idx

    def _promoted_instances(self, step: int, producer: dict[str, Any], request_uuid: str | None) -> tuple[list[dict] | None, str | None]:
        """The fold instances that PRODUCE a promoted member, resolved through the Program's own
        dataflow rather than the hint's module prefix.

        A promoted manifest row names an INTERIOR Program row whose output another module reads: identity (producer module, `out`), with
        `spec` = the producing family and `consumers` = the reading modules in Program order.  The fold hints such an instance by the
        vLLM module path the value was OBSERVED under (the producing Call's own path -- for the final norm's narrowing that is `<module>.norm`,
        for the embedding scale it is the scale's site), which is not the Program row's module: `_inst[(step, op_path)]` finds nothing, and
        row #57's 432 `model`/`out` identities were "no fold instance of this op at this step" over a Match that snapshotted every step.
        Same Value, two names -- an Occurrence-resolution question, answered here by the Value itself:
        the instance whose OUTPUT storage range contains an OPERAND view of a consumer module's instance at this step, of the producer's
        family.  The first consumer in Program order that resolves wins (the collector observes the promoted value at exactly that
        consumer's input, PROMOTED_OBSERVATION_RULE); no consumer resolving = (None, why) -- not compared, named, never complete.
        Non-promoted rows never enter here (their key is `_inst[(step, op_path)]`, unchanged) -- except a member whose producer family the
        fold has NO instance of under the identity's path at this step while the Program names its consumers (`_names_producer`): a v2
        (`Q_module_body_v1`) manifest has no `promoted` flag, and #57's 432 `model/out` are that Value (fold hint `model.normalizer`)."""
        spec = str(producer.get("spec") or "").split("{", 1)[0].strip()
        consumers = [str(c) for c in (producer.get("consumers") or [])]
        if not spec or not consumers:
            return None, "promoted member without producer spec / consumers in the manifest row: not resolvable"
        outs = self._step_outputs(step)
        n_consumer_insts = 0
        for cop in consumers:
            cins = self._inst.get((step, cop)) or []
            if request_uuid is not None:
                cins = [c for c in cins if c.get("request") == request_uuid]
            n_consumer_insts += len(cins)
            found: dict[tuple, dict] = {}
            for c in cins:
                for opv in c.get("operand_views") or []:
                    r = self._view_bytes_range(opv)
                    if r is None:
                        continue
                    alloc, blo, bhi = r
                    for plo, phi, p in outs.get(alloc, ()):
                        if plo <= blo and bhi <= phi and str(p.get("spec") or "").split("{", 1)[0] == spec:
                            ov0 = (p.get("output_views") or [{}])[0]
                            found[(p.get("event"), p.get("request"), p.get("position"), p.get("row"), int(ov0.get("alloc", -1)), int(ov0.get("offset", -1)))] = p
            if found:
                return list(found.values()), None
        if n_consumer_insts == 0:
            return None, f"no fold instance of the promoted member's consumers {consumers[:3]} at this step"
        return None, f"no fold instance of producer {spec} whose output the consumers {consumers[:3]} read at this step (promoted member)"

    def _names_producer(self, step: int, op_path: str, producer: dict[str, Any] | None) -> bool:
        """Does the fold hold an instance of the producer's family under `op_path` at `step`?  False with no producer spec."""
        spec = str((producer or {}).get("spec") or "").split("{", 1)[0].strip()
        if not spec:
            return False
        fams = self._families_at.get((step, op_path))
        if fams is None:
            fams = self._families_at[(step, op_path)] = frozenset(str(m.get("spec") or "").split("{", 1)[0] for m in self._inst.get((step, op_path)) or ())
        return spec in fams

    def member_bytes(self, step: int, op_path: str, member: str, request_label: str | None,
                     producer: dict[str, Any] | None = None, nbytes: int | None = None) -> tuple[bytes | None, str | None, int]:
        """Oracle bytes of (step, op_path, member) for request r<k> (None: all requests, row order): concatenation over the instances'
        output views (operand view 0 for the ids member 'in'); returns (bytes | None, why, n_instances).

        `producer` (from the required manifest's row: {"spec": <family base>, "ordinal": k | None}) names the Program op that PRODUCES the
        member.  A module whose forward is exploded into several fold instances per row under the same hint (dense's Gemma composites:
        AddScalar / Square / Mean / Rsqrt / ScaleRow / Narrow ... all hinted `model.layers.N.input_layernorm`) is read at the producer's
        instances only; when the producer family occurs k > 1 times per row (the fused norm's two NarrowF32ToBf16: residual, normed) the
        manifest must say which by `ordinal` (index in Program/fold order per row), else the member is 'ambiguous producer' -- partial,
        never guessed from which candidate happens to equal the committed bytes (that would hide a swapped residual / normed pair).

        An `interior` producer (a `call_boundaries` row, op path `<module>/<fx node>`) is read at its module's instances of the producer
        family.  When every row's group holds the same k > 1 producer instances and their member views together are the identity's
        `nbytes`, the member is all of them (one Call per instance: an attention row per head, a KV quantization per KV head), in row
        then storage order."""
        if step not in self.snapshot_steps:
            return None, "step not snapshotted by the Match", 0
        if (producer or {}).get("interior"):
            op_path = op_path.split("/", 1)[0]
        uuid: str | None = None
        if request_label is not None and request_label != "*":
            uuid = next((u for u, lab in self.requests.items() if lab == request_label), None)
            if uuid is None:
                return None, f"request {request_label} not in the Match record's request table", 0
        promoted = bool((producer or {}).get("promoted")) or (bool((producer or {}).get("consumers"))
                                                              and not self._names_producer(step, op_path, producer))
        if promoted:
            # a promoted member is the output of an interior Program row: its fold instance sits under the
            # producing Call's path, not the identity's module -- resolved through the Value (consumer operand -> producer output), see
            # `_promoted_instances`; the hint-prefix key below is never consulted for it
            insts, why = self._promoted_instances(step, producer or {}, uuid)
            if insts is None:
                return None, why, 0
            want_hint = None
        else:
            insts = self._inst.get((step, op_path)) or []
            if not insts:
                return None, "no fold instance of this op at this step", 0
            want_hint = _ROPE_MEMBER.get(member) if op_path.endswith("rotary_emb") else None
            if want_hint:
                insts = [m for m in insts if m["hint"].endswith(want_hint)]
            if uuid is not None:
                insts = [m for m in insts if m["request"] == uuid]
        if not insts:
            return None, "no fold instance for this request/member at this step", 0
        producer_view0 = promoted                     # the promoted member IS the producer row's single output (`PROMOTED_MEMBER`)
        spec = "" if promoted else str((producer or {}).get("spec") or "").split("{", 1)[0].strip()
        if spec:
            by_spec = [m for m in insts if str(m.get("spec") or "").split("{", 1)[0] == spec]
            if not by_spec:
                seen = sorted({str(m.get("spec") or "").split("{", 1)[0] for m in insts})
                return None, f"no fold instance of producer {spec} under {op_path} at this step (fold has {seen[:6]})", 0
            # per row (position): k producer instances -> ONE is picked.  Preferred selector = the manifest's
            # `producer_operand`: the family of the instance whose OUTPUT the producer consumes (its first operand view), resolved by
            # storage identity (alloc, offset) among the module's own instances at this step -- independent of emission order, which a
            # dag-criterion fold may permute for two independent producers.  Fallback = `producer_ordinal` in EVENT order (the manifest
            # confirms it against Program order; a permuted fold would make the two disagree, so the ordinal is only trusted alone when
            # no operand is declared).  Neither: 'ambiguous producer' -- partial, never guessed.
            per_pos: dict[Any, list[dict]] = defaultdict(list)
            for m in by_spec:
                per_pos[(m.get("request"), m.get("position"), m.get("row"))].append(m)
            k = max(len(v) for v in per_pos.values())
            if k > 1 and nbytes is not None and len({len(v) for v in per_pos.values()}) == 1 and self._member_span(by_spec, member, producer) == nbytes:
                pass
            elif k > 1:
                ordinal = (producer or {}).get("ordinal")
                operand = str((producer or {}).get("operand") or "").split("{", 1)[0].strip()
                picked = []
                if operand:
                    out_fam: dict[tuple[int, int], set[str]] = defaultdict(set)      # (alloc, offset) of an output view -> producing families
                    for m in insts:
                        for ov in m.get("output_views") or []:
                            out_fam[(int(ov["alloc"]), int(ov["offset"]))].add(str(m.get("spec") or "").split("{", 1)[0])
                    for v in per_pos.values():
                        fed = []
                        for m in v:
                            opv = (m.get("operand_views") or [None])[0]
                            fams = out_fam.get((int(opv["alloc"]), int(opv["offset"]))) if opv else None
                            if fams and operand in fams:
                                fed.append(m)
                        if len(fed) != 1:
                            return None, (f"producer_operand {operand}: {len(fed)} of {len(v)} {spec} instances in a row's group under {op_path} "
                                          f"consume a {operand} output (member {member!r}; want exactly 1)"), len(by_spec)
                        picked.append(fed[0])
                elif ordinal is not None:
                    for v in per_pos.values():
                        v = sorted(v, key=lambda m: int(m.get("event") or 0))
                        if int(ordinal) >= len(v):
                            return None, f"producer_ordinal {ordinal} >= {len(v)} instances of {spec} in a row's group under {op_path}", len(by_spec)
                        picked.append(v[int(ordinal)])
                else:
                    return None, (f"ambiguous producer: {k} instances of {spec} per row under {op_path} for member {member!r} -- the manifest row "
                                  "needs producer_operand (family feeding the producer) or producer_ordinal (index among the module's producer "
                                  "instances in Program order)"), len(by_spec)
                by_spec = picked
            insts = by_spec
            producer_view0 = len(insts[0].get("output_views") or []) == 1       # the producer op's single output IS the member
        # ORDER = the request's rows in the packed member = token POSITION order (a request's tokens are contiguous and increasing in the
        # engine's row order), tie-broken by the view's storage offset.  NOT the fold's `row`: for attention instances `row` is the sequence's
        # index inside the varlen launch (and not even constant over a 256-token prefill: rows {0,1} for one request) -- sorting by it
        # permuted r0's 256 attention rows and reported a false C2 mismatch (at B2: 16 identities, all *.self_attn.attn).
        def _order(m: dict) -> tuple[int, int]:
            pos = m.get("position")
            first = m["output_views"][0] if m.get("output_views") else (m["operand_views"][0] if m.get("operand_views") else {})
            return (int(pos) if pos is not None else (int(m["row"]) if m.get("row") is not None else 0), int(first.get("offset") or 0))
        insts = sorted(insts, key=_order)
        port = _port_view(member, producer)
        parts: list[bytes] = []
        for m in insts:
            if member == "in":
                views = m["operand_views"][:1]
            elif member in ("out", "0") or want_hint or producer_view0:
                views = m["output_views"][:1]
            elif port is not None and len(m["output_views"]) > 1:
                views = m["output_views"][port:port + 1]
            else:
                views = m["output_views"][:1]
            if not views:
                return None, f"instance has no view for member {member!r}", len(insts)
            b, why = self._view_bytes(int(m["event"]), views[0])
            if b is None:
                return None, why, len(insts)
            parts.append(b)
        return b"".join(parts), None, len(insts)

    def _member_span(self, insts: list[dict], member: str, producer: dict[str, Any] | None = None) -> int | None:
        """The bytes of the views `member` reads over `insts` (an instance with one output view: that view; else the member's port view,
        `_port_view`, else view 0); None when one has no such view."""
        port = _port_view(member, producer)
        n = 0
        for m in insts:
            ovs = m.get("output_views") or []
            j = port if port is not None and len(ovs) > 1 else 0
            r = self._view_bytes_range(ovs[j]) if j < len(ovs) else None
            if r is None:
                return None
            n += r[2] - r[1]
        return n

    def ids_bytes(self, step: int, request_label: str, itemsize: int) -> tuple[bytes | None, str | None]:
        """The input token ids of request r<k>'s rows at `step` (row order), from tokens.json (prompt + generated ids) at the StepStart
        rows' positions -- available for EVERY step (the ids buffer is a capture Root, never snapshotted)."""
        rows = self.step_rows.get(step)
        if rows is None:
            return None, "no StepStart in the Match record for this step"
        uuid = next((u for u, lab in self.requests.items() if lab == request_label), None)
        if uuid is None or uuid not in self.tokens:
            return None, f"request {request_label} not in the Match record's tokens.json"
        prompt, gen = self.tokens[uuid]
        out = bytearray()
        for u, pos in rows:
            if u != uuid:
                continue
            if pos < len(prompt):
                tid = prompt[pos]
            elif pos - len(prompt) < len(gen):
                tid = gen[pos - len(prompt)]
            else:
                return None, f"position {pos} beyond prompt {len(prompt)} + generated {len(gen)} of {request_label}"
            out += int(tid).to_bytes(itemsize, "little", signed=True)
        if not out:
            return None, f"request {request_label} has no rows at step {step} in the Match record"
        return bytes(out), None

    def sampled_id(self, step: int, request_label: str) -> tuple[int | None, str | None]:
        ids = self.sampled.get(step)
        order = self.step_requests.get(step)
        if ids is None or order is None:
            return None, "no SampleEnd/StepStart in the Match record for this step"
        labs = [self.requests.get(u, u) for u in order]
        if request_label not in labs:
            return None, f"request {request_label} not scheduled at step {step} in the Match record ({labs})"
        i = labs.index(request_label)
        if i >= len(ids):
            return None, f"SampleEnd carries {len(ids)} ids for {len(labs)} requests"
        return ids[i], None


def _port_view(member: str, producer: dict[str, Any] | None) -> int | None:
    """The output view of a several-output Call that `member` is: its manifest row's `port`, else the port `query.required.member_name`
    spells into it (`<port>`, or `<hint>.<port>` when the Build recorded a hint), when that port is a digit; else None."""
    port = (producer or {}).get("port")
    port = str(port) if port is not None else member.rsplit(".", 1)[-1]
    return int(port) if port.isdigit() else None
