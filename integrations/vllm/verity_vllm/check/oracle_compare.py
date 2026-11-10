"""Live C2 -- value correspondence by ORACLE COMPARE.

The Commit's C1 (coverage) and C3 (openings) cannot see a relabel, a same-name byte swap or a late read (every
one of them got COMMIT PASS with coverage 12943/12943 and 79/79 openings verified).  C2 asks: are the COMMITTED bytes of
each identity the value the Program produced for that identity?  The reference here is the Match stage of the SAME attempt: the m1
capture (`verity_vllm.pipeline.m1_capture`, a separate process and a separate acquisition path -- torch dispatcher observer + D2H
snapshots -- sharing nothing with the native collector) recorded, for every snapshotted engine step, the bytes of every dispatcher
call's arguments and outputs (`capture/snapshots.json` + `capture/values/<step>/<sha>.bin`, spanned storage bytes, content-addressed),
and the fold's runtime-correspondence record (`instances.jsonl`) says, per Program instance, WHICH event and WHICH storage view (alloc,
offset, shape, dtype) is its output, attributed to a request and a token position.  So the oracle value of identity
(request r, step s, op_path, output_member) is: the fold's instances of op_path at step s attributed to r, in row order, each read at
its output view from the snapshot of its producing event.  That is the Program-evaluated value of record for the Match verdict, read
from the record -- the collector's own bytes never enter the expected side.

    compared identity  : committed bytes (opened from the collector's retained copy of the member and verified against the run root,
                         `OpenedReader`, sliced at the binding-map entry's element offset + the identity's request-local range) == oracle
                         bytes                                                                    -> equal | MISMATCH (named; an opening
                         that does not verify is a mismatch)
    sampled_token_ids  : the committed id == the capture's SampleEnd id for that request at that step (every step, no snapshot needed)
    not compared       : identities of a step the Match did not snapshot (row_pod MATCH_SNAP_STEPS, default 0,1), of a family the
                         Match has no value for (the FA2/FA3 M1 kernel-interior stream: its C2 is the tap's own bit-exact xcheck,
                         capacity FA2-01), registered inputs (weights: digest-pinned by the Build), or an op the fold did not instance
                         -> counted per reason, never silently dropped

Result: `result` is False on ANY mismatch (the verdict fails closed naming the identity), True when every comparable identity of every
snapshotted step was compared equal AND every step was snapshotted (complete live C2), else None with the coverage stated (partial:
the finalizer grades C2 'not checked' / partial).  Provenance of the oracle (Match dir, capture created_utc,
profile id, fold digest, snapshot steps, tokens.json sha256) is recorded so a reviewer can tell the reference is this attempt's Match.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from collections import Counter, defaultdict
from typing import Any, Callable

# re-exported: the engine-layer tp worker may import this module but not commit/ (tests/lint P9)
from verity_vllm.commit.opened import ORDINAL_SEP, OPENED_METHOD, OpenedReader, OpeningNotVerified, meta_by_name  # noqa: F401  -- re-exported

SCHEMA = "verity-sweep/oracle-compare/v1"
METHOD = "oracle-compare (Match capture of the same attempt: snapshots at the fold's instance output views + SampleEnd ids)"
NO_ORACLE_FAMILY_PREFIXES = ("fa2.m1/", "fa3.m1/", "moe/")          # kernel-interior streams: no dispatcher-level value in the Match record
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
        out[key] = hashlib.sha256(open(pth, "rb").read()).hexdigest() if os.path.exists(pth) else None
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
        cap = os.path.join(match_dir, "capture")
        self.snap_path = os.path.join(cap, "snapshots.json")
        self.inst_path = os.path.join(match_dir, "instances.jsonl")
        self.log_path = os.path.join(cap, "log.jsonl.gz")
        for p in (self.snap_path, self.inst_path, self.log_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"match oracle: {p} missing (Match stage of this attempt did not write it)")
        snap = json.load(open(self.snap_path))
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
        tok = os.path.join(cap, "tokens.json")
        self.tokens_sha256 = hashlib.sha256(open(tok, "rb").read()).hexdigest() if os.path.exists(tok) else None
        # the ids of every row of every step: prompt ids + generated ids per request (tokens.json) at the StepStart rows' (request, position)
        # -- the runner's ids buffer is a Root of the capture (prompt_token roots), so it is never in a snapshot; this IS its value record
        self.tokens: dict[str, tuple[list[int], list[int]]] = {}
        if os.path.exists(tok):
            for r in (json.load(open(tok)).get("requests") or []):
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
        r<index>, prompt_len, max_tokens) against the fold header's request table and tokens.json (prompt ids per request; generated <= cap).
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
            b = open(os.path.join(self.dir, "capture", rel), "rb").read()
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
    def _view_bytes_range(view: dict) -> tuple[int, int, int] | None:
        """(alloc, byte lo, byte hi) the storage range a fold view spans (contiguous: offset..offset+numel; strided: the span), or None
        when the view has no itemsize for its dtype."""
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


def _family_of(entry: dict) -> str:
    ident = entry["identity"]
    if entry.get("kind") == "registered-input" or ident.get("op_path") == "weights":
        return "weights"
    name = str(entry.get("name") or ident.get("op_path") or "")
    if name.startswith(NO_ORACLE_FAMILY_PREFIXES) or str(ident.get("op_path", "")).startswith(NO_ORACLE_FAMILY_PREFIXES) or (m := ident.get("output_member")) == "norm_scale" or m == "router_softmax" or m == "vocab_range" or m == "scale_products":
        return "hidden_stream"                                     # ... and a tap's Call-interior value (query.norm_scales / router_softmax / vocab_range / scale_products MEMBER): no Match value
    if ident.get("output_member") == "sampled_token_ids":
        return "sampled_token_ids"
    if ident.get("output_member") == "in":
        return "input_token_ids"
    return "instance_outputs"


def _query_family_of(entry: dict) -> str:
    """The Q(P) family (required-manifest vocabulary) a binding-map entry belongs to -- `fa2.m1/` -> fa2_hidden_m1_stream,
    `fa3.m1/` -> fa3_hidden_m1_stream, `moe/` -> moe_block_stream, registered inputs -> weights; the oracle's own coarse classes
    (`_family_of`: weights / hidden_stream / ids / instance_outputs) stay what the verdict reads, this is the label the POPULATION uses."""
    ident = entry["identity"]
    if entry.get("kind") == "registered-input" or ident.get("op_path") == "weights":
        return "weights"
    name = str(entry.get("name") or ident.get("op_path") or "")
    op = str(ident.get("op_path", ""))
    for pref, qf in (("fa2.m1/", "fa2_hidden_m1_stream"), ("fa3.m1/", "fa3_hidden_m1_stream"), ("moe/", "moe_block_stream")):
        if name.startswith(pref) or op.startswith(pref):
            return qf
    return _family_of(entry)


def query_families_of(manifest: dict[str, Any] | None) -> dict[str, int] | None:
    """Identities per Q(P) family from the required manifest's rows (the manifest's own identity rule -- an M1 stream identity is one per
    (layer, ENGINE step), a moe plane one per (plane, request, step); binding ENTRIES are more numerous than either).  None when no
    manifest was supplied (the statement then says so rather than counting entries as identities)."""
    if not isinstance(manifest, dict) or not isinstance(manifest.get("identities"), list):
        return None
    return dict(Counter(str(r.get("family")) for r in manifest["identities"]))


def runtime_rows_of(manifest: dict[str, Any] | None) -> dict[tuple, tuple[int, int]]:
    """(step, request, op_path, member, invocation) -> (r0, rows) of each identity whose Values are the runtime's rows r0.. (`runtime_rows`)."""
    return {(int(r["step"]), str(r["request_id"]), str(r["op_path"]), str(r.get("output_member", "out")), int(r.get("invocation", 0))):
            (int(r["runtime_rows"][0]), int(r["runtime_rows"][1]) - int(r["runtime_rows"][0])) for r in (manifest or {}).get("identities") or () if r.get("runtime_rows")}


def producers_of(manifest: dict[str, Any] | None) -> dict[tuple[str, str], dict[str, Any]]:
    """(op_path, output_member) -> {"spec": producer family base, "ordinal": producer_ordinal | None, "operand": producer_operand | None}
    from the required manifest's rows (`spec` is the Program op family that produces the member; `producer_ordinal` the optional index
    among same-family producers per row; `producer_operand` the family whose output the producer consumes -- the order-free selector).
    A PROMOTED row additionally carries `promoted: True` and `consumers` (the reading modules, Program order, unioned over the
    member's rows): the oracle resolves it through the Value's dataflow, not the hint prefix (MatchOracle._promoted_instances)."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    consumers: dict[tuple[str, str], list[str]] = {}
    for r in (manifest or {}).get("identities") or []:
        if r.get("family") not in ("instance_outputs", "call_boundaries") or not r.get("spec"):
            continue
        key = (str(r.get("op_path")), str(r.get("output_member")))
        cur = out.get(key)
        # a PROMOTED row (`promoted: true`, `consumers` = the reading modules, Program order) is resolved by the
        # oracle through the Value's dataflow (MatchOracle._promoted_instances); `promoted` is part of the producer's identity here (a
        # promoted and a vocabulary row on one member is a conflict), the consumer lists of the rows are unioned in first-seen order
        ent = {"spec": str(r["spec"]).split("{", 1)[0], "ordinal": r.get("producer_ordinal"), "operand": r.get("producer_operand")}
        if r.get("port") is not None:
            ent["port"] = str(r["port"])
        if r.get("promoted"):
            ent["promoted"] = True                    # a vocabulary row's entry is byte-for-byte what it was (no key added)
        if r.get("family") == "call_boundaries":
            ent["interior"] = True
        if cur is None:
            out[key] = ent
        elif cur != ent:
            out[key] = {"spec": None, "ordinal": None, "operand": None, "conflict": True}          # two producers claim one member: no filter, fail visibly
            consumers.pop(key, None)
            continue
        if ent.get("promoted"):
            cl = consumers.setdefault(key, [])
            for c in r.get("consumers") or []:
                if str(c) not in cl:
                    cl.append(str(c))
    for key, cl in consumers.items():
        if out[key].get("promoted"):
            out[key]["consumers"] = cl
    return out


def producers_of_programs(programs: dict[str, dict[str, Any]], aliases: dict[str, Any] | None = None, query: str | None = None) -> dict[tuple[str, str], dict[str, Any]]:
    """(op_path, output_member) -> {"spec", "operand", "consumers"} read from the Programs of record by dataflow, under `query`, the one
    the query path built the manifest with (`required.population`, the Correspondence `query.cli` reads; None: the query of record) --
    such a manifest carries no producer annotations.  Per required `instance_outputs` Value of every producing Call: `spec` = the Call's family; `operand` = the
    family of the Call whose output its FIRST operand reads (the order-free selector among same-family instances of one module: a fused
    norm's two narrowings read AddWidenedBf16 -> `1` and MulVecF32 -> `0`); `consumers` = the modules of the Calls that read the Value
    outside its own module, Program order (the dataflow resolution when the fold observed the producer under another path).  A fact the
    member's rows disagree on is None (the oracle then names the ambiguity; never guessed); a family disagreement is a `conflict`.
    `programs`: request label -> `manifest_format.load_program(dir)`; `aliases`: the manifest's `op_path_aliases` (alias -> [canon, k])."""
    from verity_vllm.correspondence.reader_for_query import Correspondence
    from verity_vllm.query.program_view import from_instances

    out: dict[tuple[str, str], dict[str, Any]] = {}
    seen: set[int] = set()
    for prog in programs.values():
        inst = prog["instances"]
        if id(inst) in seen:
            continue
        seen.add(id(inst))
        res = prog.get("result") or {}
        P = from_instances(inst)
        corr = Correspondence.of(P, prog.get("dir"), implementation_paths=Correspondence.implementation_paths_of(res))
        tpinfo = res.get("tp") if isinstance(res.get("tp"), dict) else {}
        world = int(tpinfo.get("world") or 1)
        program_producer_facts(P, corr, tp=({"world": world, "rank": int(tpinfo.get("rank") or 0)} if world > 1 else None), aliases=aliases, into=out, query=query)
        del P, corr
    return out


def program_producer_facts(P: Any, corr: Any, *, tp: dict[str, Any] | None = None, aliases: dict[str, Any] | None = None,
                           into: dict[tuple[str, str], dict[str, Any]] | None = None, query: str | None = None) -> dict[tuple[str, str], dict[str, Any]]:
    """`producers_of_programs` for one ProgramView + Correspondence, accumulated into `into` (the member's facts over every Program)."""
    from verity_vllm.query.required import QUERY_OF_RECORD, population

    canon = {str(k): str(v[0]) for k, v in (aliases or {}).items() if isinstance(v, (list, tuple)) and v}
    out = {} if into is None else into
    pop = population(P, corr, tp=tp, query=query or QUERY_OF_RECORD)
    members = {cid: [cm.member for cm in pop.members_of(cid) if cm.family == "instance_outputs"] for cid in pop.producers()}
    members = {cid: ms for cid, ms in members.items() if ms}
    readers: dict[int, list[int]] = defaultdict(list)
    for c in P.calls:
        for r in c.reads:
            if r.call >= 0 and r.call in members and (not readers[r.call] or readers[r.call][-1] != c.id):
                readers[r.call].append(c.id)
    for cid, ms in members.items():
        c = P.call(cid)
        own = corr.module_of(cid)
        path = corr.implementation_path(pop.program_path(cid))
        path = canon.get(path, path)
        first = next(iter(c.reads), None)
        operand = P.call(first.call).family if first is not None and first.call >= 0 else None
        cons: list[str] = []
        for rc in readers.get(cid, ()):
            m = corr.module_of(rc)
            if m != own:
                m = corr.implementation_path(m)
                if m not in cons:
                    cons.append(m)
        for mem in ms:
            key = (path, mem)
            cur = out.get(key)
            if cur is None:
                out[key] = {"spec": c.family, "operand": operand, "consumers": list(cons)}
                continue
            if cur.get("conflict"):
                continue
            if cur["spec"] != c.family:
                out[key] = {"spec": None, "operand": None, "consumers": [], "conflict": True}
                continue
            if cur["operand"] != operand:
                cur["operand"] = None
            for m in cons:
                if m not in cur["consumers"]:
                    cur["consumers"].append(m)
    return out


def merge_producers(declared: dict[tuple[str, str], dict[str, Any]], derived: dict[tuple[str, str], dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """The manifest's producer entries (`producers_of`) completed by the Program-derived facts (`producers_of_programs`) where the
    manifest row declares no selector of its own: `operand` (when the manifest gives neither `producer_operand` nor `producer_ordinal`)
    and `consumers` (when it is not a promoted row).  Only for members both name with the SAME family; a manifest entry is never
    overridden, a member the manifest does not name is never added."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for key, ent in declared.items():
        d = derived.get(key)
        if not d or d.get("conflict") or ent.get("conflict") or not ent.get("spec") or d.get("spec") != ent.get("spec"):
            out[key] = ent
            continue
        ent = dict(ent)
        if d.get("operand") and ent.get("operand") is None and ent.get("ordinal") is None:
            ent["operand"] = d["operand"]
        if d.get("consumers") and not ent.get("promoted"):
            ent["consumers"] = list(d["consumers"])
        out[key] = ent
    return out


def promote_handed_down(manifest: dict[str, Any] | None, derived: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    """Mark, in place, the `instance_outputs` rows of a `v2-query` manifest whose Value the producer module's body hands DOWN: every module
    the Programs name as reading it (`derived[...]["consumers"]`, `producers_of_programs`) is inside the producer's module subtree
    (`required.under`), so it is no return of that module.  Such a row gets `promoted: true` + those `consumers` (Program order) and the
    acquisition observes it at its first consumer's input (`manifest_format.PROMOTED_OBSERVATION_RULE`) instead of hooking the module's
    forward output: row #57's `model/out` is Gemma2's embed scale (the model body's `hidden *= normalizer`, read by `model.layers.0`)
    while `Gemma2Model` returns the final norm's output -- the two differ from element 0 (form (B): 432
    mismatches, 159,408 equal).  A v2 manifest carries no `promoted` flag (`required`: such a Value simply crosses the body boundary).
    Only rows whose family the Programs agree on (`spec` equal to the row's, no conflict) and that are not already promoted; a member
    read by any module outside the subtree is a return and stays a module output.  Returns {"members": {"op/member": consumers}, "rows": n}."""
    from verity_vllm.query.required import under

    picked: dict[tuple[str, str], list[str]] = {}
    for key, d in derived.items():
        cons = [str(c) for c in (d.get("consumers") or [])]
        if cons and not d.get("conflict") and all(under(c, key[0]) for c in cons):
            picked[key] = cons
    rows = 0
    members: dict[str, list[str]] = {}
    for r in (manifest or {}).get("identities") or []:
        if r.get("family") != "instance_outputs" or r.get("promoted"):
            continue
        key = (str(r.get("op_path")), str(r.get("output_member")))
        cons = picked.get(key)
        if cons is None or str(derived[key].get("spec") or "") != str(r.get("spec") or "").split("{", 1)[0].strip():
            continue
        r["promoted"] = True
        r["consumers"] = list(cons)
        members[f"{key[0]}/{key[1]}"] = list(cons)
        rows += 1
    return {"members": members, "rows": rows}


def _foreign_oracle(oracle: MatchOracle, bmap: dict[str, Any], wcheck: dict[str, Any]) -> dict[str, Any]:
    """The result of a Match record whose request table is not this workload's: False, nothing compared."""
    why = "foreign oracle: the Match record is not this workload's -- " + "; ".join(wcheck["problems"][:4])
    return {"schema": SCHEMA, "method": "oracle-compare (Match capture of the same attempt: snapshots + fold + capture log + tokens.json)",
            "result": False, "grade": "foreign-oracle", "why": why, "complete": False, "compared": 0, "equal": 0, "mismatch_n": 0, "mismatch": [],
            "compared_by_family": {}, "not_compared": {"foreign oracle": len(bmap.get("entries") or [])}, "steps_compared": [], "steps_not_snapshotted": [],
            "attribution": {"complete": False, "pairs": 0, "unverified_n": 0, "ambiguous_steps": []}, "workload_check": wcheck, "oracle": oracle.provenance()}


def _ambiguous_steps(pairs_seen: set[tuple[int, str]], attrib: dict[tuple[int, str], dict[str, Any]], snapshot_steps: Any) -> list[int]:
    """Steps not snapshotted where two requests carry the same ids AND the same sampled id: a swap there is not attributable."""
    ambiguous: list[int] = []
    for s in sorted({s for s, _ in pairs_seen}):
        if s in snapshot_steps:
            continue
        sig = Counter((attrib.get((s, r), {}).get("ids_sha"), attrib.get((s, r), {}).get("sampled_value")) for ss, r in pairs_seen if ss == s)
        if any(n > 1 for n in sig.values()):
            ambiguous.append(s)
    return ambiguous


def oracle_compare(oracle: MatchOracle, bmap: dict[str, Any], committed: Callable[[int, str, int, int], bytes | None], *,
                   max_report: int = 12, producers: dict[tuple[str, str], dict[str, Any]] | None = None,
                   workload: dict[str, Any] | None = None, aliases: dict[str, Any] | None = None,
                   query_families: dict[str, int] | None = None, own: frozenset[str] = frozenset(),
                   runtime_rows: dict[tuple, tuple[int, int]] | None = None) -> dict[str, Any]:
    """Compare every binding-map entry's committed bytes with the Match oracle.  `committed(step, name, byte_lo, byte_hi)` returns the
    bytes of member `name` at `step` in [byte_lo, byte_hi) (None if not retained) -- in the Commit an `OpenedReader`, whose
    `OpeningNotVerified` makes the entry a mismatch.  `own` names what the run's sources committed that no module produced (their
    `own_names`: the KV slots of commit --memory-map), so no Match value: not compared, counted as hidden_stream.  `runtime_rows`
    (`runtime_rows_of(manifest)`): an identity that is the runtime's rows r0.. is compared with the fold's rows from r0.  `producers` (producers_of(manifest))
    names each member's producing Program op so exploded-module folds are read at the producer's instances.  `workload` (the run's declared
    requests) is checked against the oracle's request table BY NAME first: a foreign Match record is result False 'foreign oracle' with
    nothing compared.

    On the ORACLE side, a shared module's invocation k >= 1 (the binding map's `invocation`) is a Program op of its own:
    the fold's instances are under the per-layer Program paths, so the oracle is read at the path the manifest's `op_path_aliases`
    (`aliases`, the table the replay resolves through) gives for (canonical op, k) -- `model.layers.k.self_attn.rotary_emb`.  Before this,
    every invocation was read at the CANONICAL op (= layer 0's fold instance) while the committed side reads invocation k's
    own bytes: layers 1..N would compare as FALSE MISMATCHES (and before the reader was fixed they compared EQUAL vacuously, layer 0 against layer 0).
    Without the table, invocation k >= 1 is NOT COMPARED, by name (partial) -- never equal by collapse, never a mismatch by mis-addressing."""
    inv_alias = {(str(v[0]), int(v[1])): str(k) for k, v in (aliases or {}).items()}       # (canonical op, invocation) -> Program op_path
    wcheck = oracle.workload_check(workload)
    if wcheck.get("ok") is False:
        return _foreign_oracle(oracle, bmap, wcheck)
    compared = equal = 0
    mism: list[dict] = []
    not_cmp: Counter = Counter()
    not_cmp_fam: Counter = Counter()
    by_fam_cmp: Counter = Counter()
    not_cmp_qfam: Counter = Counter()                  # no-oracle entries per Q(P) family (fa2_hidden_m1_stream / moe_block_stream / weights)
    shared_inv: Counter = Counter()                    # invocations >= 1 of a shared name: compared at their Program op / not compared
    steps_cmp: set[int] = set()
    steps_seen: set[int] = set()
    seen: set[tuple] = set()
    # REQUEST ATTRIBUTION: per (step, request) the two per-request id families the ids oracle covers at EVERY
    # step (tokens.json + StepStart rows; SampleEnd) -- a relabel of equal-row-count requests on one engine step (b2swapeq) moves r0's
    # rows to r1's slots: the bound `in` row / sampled id of r0 then carries r1's token, which THIS request's oracle value refutes.
    attrib: dict[tuple[int, str], dict[str, Any]] = {}
    pairs_seen: set[tuple[int, str]] = set()
    padding_not_cmp: Counter = Counter()               # bot-by-enablement entries per Q(P) family: no execution, no oracle value
    padding_pairs: set[tuple[int, str]] = set()
    name_inv = same_name_invocations(bmap)                                       # (step, member name) -> invocations bound
    for e in bmap.get("entries") or []:
        ident = e.get("identity") or {}
        step = int(ident.get("step", -1))
        if step < 0:
            continue
        if e.get("activity") == "bot-by-enablement":
            # a PADDING entry (a padded record's inactive stratum, LIFTING-SPEC v0.7 B9.1) materializes an
            # identity the executed prefix of record left INACTIVE: nothing executed, so the Match oracle holds no value for it and its
            # (step, request) pair is not one request attribution can or must verify -- its word is checked against bot_word by the
            # padded Commit's inactive-stratum check, not here.  Counted apart, per Q(P) family, never as 'identity without a byte
            # range' nor as an unverified attribution pair (the first #3 pod run graded 'partial' on exactly these 7,665 + 21 pairs).
            padding_not_cmp[str(e.get("family") or _query_family_of(e))] += 1   # a padding entry carries its Q(P) family (padding_entries)
            padding_pairs.add((step, str(ident.get("request_id", "r0"))))
            continue
        fam = "hidden_stream" if e.get("name") in own else _family_of(e)
        steps_seen.add(step)
        if fam in ("weights", "hidden_stream"):
            not_cmp["no oracle family: " + fam] += 1
            not_cmp_fam[fam] += 1
            not_cmp_qfam[_query_family_of(e)] += 1                   # per Q(P) family, in binding entries (identities below)
            continue
        rid = str(ident.get("request_id", "r0"))
        key = (step, rid, ident.get("op_path"), ident.get("output_member"), ident.get("invocation"), tuple(ident.get("element_range") or ()))
        if key in seen:
            continue
        seen.add(key)
        if rid != "*":
            pairs_seen.add((step, rid))
        enc = e.get("encoding") or {}
        dtype = str(enc.get("dtype") or "")
        isz = _ITEMSIZE.get(dtype)
        lo, hi = ident.get("element_range") or [0, None]
        if isz is None or hi is None or int(hi) <= int(lo):
            not_cmp["identity without a byte range (dtype/range)"] += 1
            not_cmp_fam[fam] += 1
            continue
        eo = int(e.get("element_offset") or 0)
        blo, bhi = (eo + int(lo)) * isz, (eo + int(hi)) * isz
        try:
            got = committed(step, entry_read_name(e, name_inv.get((step, str(e.get("name"))), 1)), blo, bhi)   # exact invocation
        except OpeningNotVerified as exc:
            compared += 1
            by_fam_cmp[fam] += 1
            steps_cmp.add(step)
            mism.append({"identity": ident, "family": fam, "name": e.get("name"), "why": f"opened value not verified: {exc}"})
            continue
        if got is None:
            not_cmp["committed bytes not retained on host (retain=none, or an excluded family under retain=host + exclude)"] += 1
            not_cmp_fam[fam] += 1
            continue
        # the Program op of THIS invocation: 0 = the op_path itself; k >= 1 = the alias table's path, else not compared
        inv_k = int(ident.get("invocation") or 0)
        op_k = str(ident.get("op_path"))
        if inv_k > 0:
            op_k = inv_alias.get((op_k, inv_k))
            if op_k is None:
                not_cmp[f"shared-module invocation >= 1 with no Program path in the alias table (op {ident.get('op_path')}): not compared"] += 1
                not_cmp_fam[fam] += 1
                shared_inv["not_compared"] += 1
                continue
            shared_inv["compared_at_program_op"] += 1
        if fam == "sampled_token_ids":
            oid, why = oracle.sampled_id(step, rid)
            if oid is None:
                not_cmp[why] += 1
                not_cmp_fam[fam] += 1
                continue
            cid = int.from_bytes(got[:isz], "little", signed=True)
            compared += 1
            by_fam_cmp[fam] += 1
            steps_cmp.add(step)
            a = attrib.setdefault((step, rid), {})
            a["sampled"] = cid == oid
            a["sampled_value"] = oid
            if cid == oid:
                equal += 1
            else:
                mism.append({"identity": ident, "family": fam, "name": e.get("name"), "committed_id": cid, "oracle_id": oid})
            continue
        if fam == "input_token_ids":
            ob, why = oracle.ids_bytes(step, rid, isz)
            n_inst = len(ob) // isz if ob else 0
        else:
            prod = (producers or {}).get((str(ident.get("op_path")), str(ident.get("output_member"))))
            ob, why, n_inst = oracle.member_bytes(step, op_k, str(ident.get("output_member")), rid,
                                                  producer=(prod if prod and prod.get("spec") else None), nbytes=bhi - blo)
        if ob is None:
            not_cmp[why] += 1
            not_cmp_fam[fam] += 1
            continue
        r0, n = (runtime_rows or {}).get((step, rid, str(ident.get("op_path")), str(ident.get("output_member")), inv_k), (0, 0))
        ob = ob[r0 * (len(got) // n):(r0 + n) * (len(got) // n)] if n and len(got) % n == 0 and len(ob) >= (r0 + n) * (len(got) // n) else ob
        compared += 1
        by_fam_cmp[fam] += 1
        steps_cmp.add(step)
        if fam == "input_token_ids":
            a = attrib.setdefault((step, rid), {})
            a["ids"] = ob == got
            a["ids_sha"] = hashlib.sha256(ob).hexdigest()[:16]
        if len(ob) != len(got):
            mism.append({"identity": ident, "family": fam, "name": e.get("name"), "why": f"size: committed {len(got)} B, oracle {len(ob)} B over {n_inst} instances"})
        elif ob == got:
            equal += 1
        else:
            first = next((i for i in range(0, len(ob), isz) if ob[i:i + isz] != got[i:i + isz]), None)
            mism.append({"identity": ident, "family": fam, "name": e.get("name"), "why": f"bytes differ (first element {first // isz if first is not None else None} of {len(ob) // isz})",
                         "committed_sha256": hashlib.sha256(got).hexdigest()[:16], "oracle_sha256": hashlib.sha256(ob).hexdigest()[:16]})
    steps_not_snap = sorted(s for s in steps_seen if s not in oracle.snapshot_steps)
    complete = compared > 0 and not mism and not steps_not_snap and sum(v for k, v in not_cmp.items() if not k.startswith("no oracle family")) == 0
    result: bool | None = False if mism else (True if complete else None)
    # attribution: every (step, request) the map binds has BOTH id families compared and equal; steps where two requests carry the same
    # ids AND the same sampled id are ambiguous for a swap unless the step's values were snapshotted (then instance_outputs decide)
    attrib_missing = sorted(k for k in pairs_seen if not (attrib.get(k, {}).get("ids") and attrib.get(k, {}).get("sampled")))
    ambiguous = _ambiguous_steps(pairs_seen, attrib, oracle.snapshot_steps)
    attribution_complete = compared > 0 and not mism and not attrib_missing and not ambiguous
    grade = ("mismatch" if mism else "complete" if complete else "attribution-complete" if attribution_complete else "partial")
    why = None
    if mism:
        m = mism[0]
        i = m["identity"]
        why = (f"C2 oracle mismatch: {len(mism)} identities differ from the Match oracle (first: step {i.get('step')} {i.get('op_path')}/{i.get('output_member')} "
               f"inv {i.get('invocation')} req {i.get('request_id')} range {i.get('element_range')} -- {m.get('why') or ('id ' + str(m.get('committed_id')) + ' vs ' + str(m.get('oracle_id')))})")
    elif result is None:
        why = (f"{grade}: " + (f"steps {steps_not_snap[:8]}{'...' if len(steps_not_snap) > 8 else ''} not snapshotted by the Match (row_pod MATCH_SNAP_STEPS=all for a complete C2); " if steps_not_snap else "")
               + (f"request attribution unverified at {len(attrib_missing)} (step, request) pairs (first {attrib_missing[:4]}); " if attrib_missing else "")
               + (f"attribution ambiguous at steps {ambiguous[:8]} (equal ids + sampled id across requests, values not snapshotted); " if ambiguous else "")
               + "; ".join(f"{k} x{v}" for k, v in not_cmp.items() if not k.startswith("no oracle family")))
    return {"schema": SCHEMA, "method": METHOD, "result": result, "why": why, "grade": grade, "compared": compared, "equal": equal, "mismatch_n": len(mism), "mismatch": mism[:max_report],
            "compared_by_family": dict(by_fam_cmp), "not_compared": dict(not_cmp), "not_compared_by_family": dict(not_cmp_fam),
            # the no-oracle families said PER Q(P) FAMILY and IN IDENTITIES (the manifest's rule), with the binding-entry count
            # beside it -- `hidden_stream` is the oracle's class, not a Q(P) family, and entries are not identities (an M1 stream identity is
            # one per (layer, engine step); its entries are per member)
            "not_compared_by_query_family": {qf: {"identities": (int(query_families[qf]) if isinstance(query_families, dict) and qf in query_families else None),
                                                  "binding_entries": int(n),
                                                  # on a PADDED record the manifest count is the FULL cap (executed + inactive); the
                                                  # live (executed) identities are the manifest's minus the padding entries stated apart in `padding` for
                                                  # this family (a per-request stream identity is one entry; a shared plane's cells are per member)
                                                  **({"identities_basis": "required manifest at full cap (executed + inactive by enablement)",
                                                      "identities_live": (int(query_families[qf]) - int(padding_not_cmp.get(qf, 0))
                                                                          if isinstance(query_families, dict) and qf in query_families else None),
                                                      "identities_inactive_stated_apart": int(padding_not_cmp.get(qf, 0))} if padding_not_cmp else {}),
                                                  "why": ("no oracle family: kernel-interior stream (no dispatcher-level value in the Match record)"
                                                          if qf.endswith("_stream") else "no oracle family: registered input (pinned, not a Match value)")}
                                             for qf, n in sorted(not_cmp_qfam.items())},
            "not_compared_identities_source": ("required manifest rows" if isinstance(query_families, dict) else "no manifest supplied: identities not stated (entries only)"),
            # the corrected reach: invocations >= 1 of shared names compared at THEIR op (pre-fix: layer 0 vs layer 0)
            "shared_invocations": {"compared_at_program_op": int(shared_inv["compared_at_program_op"]), "not_compared": int(shared_inv["not_compared"]),
                                   "note": "pre-fix records compared these entries as the first tensor's bytes vs the canonical op's oracle value (vacuous equal)"},
            # the padded record's inactive stratum, stated apart: not compared (no execution), not attribution pairs
            **({"padding": {"entries_not_compared": int(sum(padding_not_cmp.values())), "by_query_family": dict(sorted(padding_not_cmp.items())),
                            "pairs": len(padding_pairs), "steps": sorted({s for s, _ in padding_pairs}),
                            "pair_list": [[s, r] for s, r in sorted(padding_pairs)],          # the pairs themselves, for the cross-document check
                            "rule": "bot-by-enablement entries materialize identities the executed prefix left inactive: no oracle value exists; "
                                    "their words are checked == bot_word by the inactive-stratum check, not by the oracle"}}
               if padding_not_cmp else {}),
            "steps_compared": sorted(steps_cmp), "steps_not_snapshotted": steps_not_snap, "complete": complete,
            "attribution": {"complete": attribution_complete, "pairs": len(pairs_seen), "unverified": attrib_missing[:max_report], "unverified_n": len(attrib_missing),
                            "ambiguous_steps": ambiguous, "requests": sorted({r for _, r in pairs_seen}, key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)},
            "workload_check": wcheck, "oracle": oracle.provenance()}


def same_name_invocations(bmap: dict[str, Any]) -> dict[tuple[int, str], int]:
    """(step, member name) -> the number of DISTINCT invocations bound under that name at that step (> 1 = a shared module's metas)."""
    inv: dict[tuple[int, str], set[int]] = {}
    for e in bmap.get("entries") or []:
        ident = e.get("identity") or {}
        if "step" not in ident or e.get("name") is None:
            continue
        inv.setdefault((int(ident["step"]), str(e["name"])), set()).add(int(ident.get("invocation", 0) or 0))
    return {k: len(v) for k, v in inv.items()}


def entry_read_name(e: dict[str, Any], same_name_invocations: int) -> str:
    """The name a committed reader is asked for: the entry's member name, suffixed `#ord<tensor_ordinal>` when MORE THAN ONE invocation of that
    member is bound at the step (a shared module) -- a bare name would return the first meta's bytes for every invocation.  Non-shared
    members keep the bare name (every reader of record and every test double understands it)."""
    name = str(e.get("name"))
    if same_name_invocations > 1 and e.get("tensor_ordinal") is not None and ORDINAL_SEP not in name:
        return f"{name}{ORDINAL_SEP}{int(e['tensor_ordinal'])}"
    return name
