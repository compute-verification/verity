"""Score the predictor against the traced Builds: per row and granularity, is the predicted Program the traced one?

For every row of an inventory (`inventory.py`), φ is read from the row's DECLARED inputs only (`phi_of`: the row id, its workload
manifest, the declared target `target_family.declared_profile`, the pinned checkpoint's `config.json`); the Program is predicted at
the granularity each Build dir stores (`build_step`: one engine step; `build_request*`: one request shape (LP, T); `build_workload`:
the GP-01 composition of the request components over the manifest; a tensor-parallel row's `rank<r>/build_*`, each at its rank) and
compared with what the Build traced:

    exact        the predicted Program digest equals the traced one
    equivalence  `program.descriptor_equivalence.compare` on the two Programs (only run when the digests differ and the traced
                 descriptor is at hand; equal digests are equal descriptors)
    divergence   the first place the descriptors differ: program_fn, a definition only one side has or whose body differs, or the
                 first root node whose callee or reference differs

A unit the vocabulary cannot express is scored `unexpressible` with the fact that stops it; a unit over `max_rows` (LP + T) is
`over_budget`; a Build dir without a Program (the Build refused or failed) is `no_trace`.  Outputs go wherever `--out` says and
`score` refuses an output path inside the rows root: a score is never written into a Build row (a switched row's AOT Build is
`aot.emit`'s, run by its own Build phase).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass

from verity_vllm.config import UsageError, option

from . import inventory as INV
from .build import Unexpressible, build_program, decoder
from .phi import Phi

SCHEMA = "verity-predict/score/v1"


# ------------------------------------------------------------------------------------------------------------------- φ of a row


def _checkpoint_revision(repo: str, checkpoints: dict | None) -> str | None:
    for c in (checkpoints or {}).get("checkpoints") or []:
        if c.get("repo") == repo and c.get("revision"):
            return c["revision"]
    return None


def hf_config_path(hf_home: str, repo: str, revision: str) -> str:
    return os.path.join(hf_home, "models--" + repo.replace("/", "--"), "snapshots", revision, "config.json")


def phi_of(rec: dict, *, hf_home: str | None = None, checkpoints: dict | None = None, config: str | None = None) -> Phi:
    """φ from a row's declared inputs: never a Build output (the traced Programs are what φ's prediction is scored against).  The
    checkpoint's `config.json` is `config`, else the pinned snapshot's under `hf_home`."""
    from verity_vllm.config import DTYPES, RowEngine, RowId
    from verity_vllm.engine.profiles.devices import device_name_of_row
    from verity_vllm.target_family import declared_profile
    rid = RowId.parse(rec["row"])
    w = rec.get("workload")
    if not w:
        raise Unexpressible(f"row {rec['row']}: no declared workload manifest at {rec.get('workload_path')}")
    repo = w.get("model")
    rev = _checkpoint_revision(repo, checkpoints) or RowEngine.of(w).revision
    cp = config or hf_config_path(str(hf_home), repo, rev)
    if not os.path.exists(cp):
        raise Unexpressible(f"no config.json for {repo}@{rev} under {hf_home}")
    hf = json.load(open(cp))
    reqs = [{"request_id": q.get("request_id"), "index": q.get("index", i), "prompt_len": q.get("prompt_len") or q.get("prompt_token_ids"),
             "max_tokens": q.get("max_tokens"), "arrive_step": q.get("arrive_step", 0), "n": q.get("n"), "seed": q.get("seed")}
            for i, q in enumerate(w.get("requests") or [])]
    target = declared_profile(rec["row"], w)
    if (device := device_name_of_row(rec["row"])) is not None:
        target["device_name"] = device
    eng = RowEngine.of(w)
    phi = Phi(model={"repo": repo, "revision": rev, "hf_config": hf, "dtype": DTYPES.get(rid.dtype, rid.dtype)}, target=target,
              tp={"world": rid.world, "rank": 0}, engine={**eng.args, **({"env": dict(eng.env)} if eng.env else {})},
              workload={"requests": reqs}, execution=rid.execution, profile=target.get("numerics_profile") or "b1-eager-v3")
    phi.sampling = sampling_of(w, phi)
    return phi


#: the policy fields a component Program reads (`vllm_meta.make_wrapper`); the rest of the Build's policy record is provenance
_POLICY_FIELDS = ("kind", "temperature", "top_p", "top_k", "seeded", "splits")


def sampling_of(w: dict, phi: Phi) -> dict:
    """The row's sampling policy as the Build declares it (`pipeline.build.sampling_policy` over the workload manifest, then
    `sampler_geometry` on the declared target: the split count, the single request's constant or "derived")."""
    import tempfile
    from verity_vllm.pipeline.build import sampler_geometry, sampling_policy
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "workload.json")
        with open(path, "w") as f:
            json.dump(w, f)
        pol = sampling_policy(None, path)
        refusal = sampler_geometry(pol, phi.target_profile()) if pol.get("kind") == "stochastic" else None
    if refusal is not None:
        raise Unexpressible(f"sampling: the Build refuses the policy ({refusal.get('hint')})")
    return {k: pol[k] for k in _POLICY_FIELDS if k in pol}


def phi_key(phi: Phi) -> str:
    """What the predicted step / request Program depends on (the workload's requests do not enter a component)."""
    k = {"model": {k: v for k, v in phi.model.items()}, "target": phi.target, "tp": phi.tp, "sampling": phi.sampling,
         "engine": phi.engine, "execution": phi.execution, "profile": phi.profile}
    return hashlib.sha256(json.dumps(k, sort_keys=True).encode()).hexdigest()[:16]


def _of_rank(phi: Phi | None, pk: str | None, rank: int | None) -> tuple:
    """(φ, its key) of a unit traced at `rank` (a tensor-parallel row's `rank<r>/` dirs); the row's own for any other unit."""
    if phi is None or rank is None or rank == int(phi.tp.get("rank", 0)):
        return phi, pk
    p = Phi.from_json({**phi.to_json(), "tp": {**phi.tp, "rank": rank}})
    return p, phi_key(p)


# ------------------------------------------------------------------------------------------------------------------- units


def unit_shape(b: dict) -> tuple:
    if b.get("wrapper") == "step":
        return ("step", b.get("M"), b.get("pos0"), None, None)
    return (b.get("wrapper"), None, None, b.get("LP"), b.get("T"))


def unit_rows(shape: tuple) -> int:
    wrapper, M, _pos0, LP, T = shape
    return int(M or 1) if wrapper == "step" else int(LP or 0) + int(T or 0)


def _build(phi: Phi, shape: tuple, with_desc: bool = True):
    wrapper, M, pos0, LP, T = shape
    if wrapper == "step":
        return build_program(phi, "step", M=int(M), pos0=int(pos0), with_desc=with_desc)
    return build_program(phi, wrapper, LP=int(LP), T=int(T), with_desc=with_desc)


def _predict_task(task: dict) -> dict:
    """One distinct (φ, shape): the predicted digest (run in a pool worker)."""
    t0 = time.time()
    out = {"key": task["key"]}
    try:
        r = _build(Phi.from_json(task["phi"]), tuple(task["shape"]))
        out.update(status="predicted", digest=r.digest, function=r.function, n_root_nodes=r.n_root_nodes, gates=getattr(r.program, "gates", None))
    except Unexpressible as e:
        out.update(status="unexpressible", reason=str(e))
    except Exception as e:  # noqa: BLE001 -- a predictor bug is a scored outcome, named
        out.update(status="predict_error", reason=f"{type(e).__name__}: {e}"[:600])
    out["secs"] = round(time.time() - t0, 2)
    return out


def _workload_task(task: dict) -> dict:
    t0 = time.time()
    out = {"key": task["key"]}
    try:
        out.update(status="predicted", **predict_workload(Phi.from_json(task["phi"])))
    except Unexpressible as e:
        out.update(status="unexpressible", reason=str(e))
    except Exception as e:  # noqa: BLE001
        out.update(status="predict_error", reason=f"{type(e).__name__}: {e}"[:600])
    out["secs"] = round(time.time() - t0, 2)
    return out


def request_component(r, canon: dict):
    """A predicted request Program as GP-01's component (`program.workload.request_component`), its definitions shared through
    `canon` (one dict a composition)."""
    from verity.ml.circuits.defs import REGISTRY
    from verity_vllm.program import workload as W
    from verity.ml.definitions.registry.catalog import load as load_registry
    load_registry()
    params = [(name, repr(t)) for name, t in r.program.fn.signature()[0]]
    return W.request_component(r.desc, REGISTRY, params, canon=canon, digest=r.digest)


def predict_workload(phi: Phi, comps: dict | None = None, canon: dict | None = None) -> dict:
    """GP-01's composition (`program.workload.WorkloadProgram`) over the predicted request components, manifest order, with the
    lag the declared target resolves (`TargetProfile.lag()`); `comps` holds components already built under `canon`
    (`request_component`), by (LP, T), so a caller that has each request's Program need not hold them all."""
    from verity_vllm.pipeline.global_program import request_id_of
    from verity_vllm.program import workload as W
    reqs = phi.requests
    if any(int(r.get("n") or 1) != 1 for r in reqs):
        raise Unexpressible("a request with n > 1 (parallel sampling) in the workload")
    canon = {} if canon is None else canon
    comps = dict(comps or {})
    for LP, T in phi.request_shapes():
        if (LP, T) not in comps:
            comps[(LP, T)] = request_component(build_program(phi, "request", LP=LP, T=T), canon)
    wreqs = []
    for i, q in enumerate(reqs):
        lp, cap = int(q["prompt_len"]), int(q["max_tokens"])
        wreqs.append(W.WorkloadRequest(request_id_of(i), i, lp, cap, int(q.get("arrive_step") or 0), comps[(lp, cap - 1)], T=cap - 1))
    wl = W.WorkloadProgram(wreqs, manifest_digest=None, lag=int(phi.target_profile().lag()))
    return {"digest": wl.digest, "function": wl.fn.id[:300], "n_root_nodes": len(wl.program.root.body.nodes), "lag": wl.lag}


# ------------------------------------------------------------------------------------------------------------------- divergence


def load_descriptor(path: str) -> dict:
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as f:
        return json.load(f)


def _refs_view(a: dict) -> str:
    return json.dumps(a.get("refs"), sort_keys=True)[:240]


def first_divergence(pred: dict, traced: dict) -> dict:
    """The first place two descriptors differ, in the order a reader would look: program_fn, the definition sets, a shared
    definition's content, then the root body node by node (callee, form, argument types and references)."""
    if pred.get("program_fn") != traced.get("program_fn"):
        return {"where": "program_fn", "predicted": str(pred.get("program_fn"))[:300], "traced": str(traced.get("program_fn"))[:300]}
    dp, dt = pred["definitions"], traced["definitions"]
    root = traced.get("root")
    only_p, only_t = sorted(set(dp) - set(dt)), sorted(set(dt) - set(dp))
    if only_p or only_t:
        return {"where": "definitions", "only_predicted": only_p[:6], "only_traced": only_t[:6], "n_only_predicted": len(only_p),
                "n_only_traced": len(only_t)}
    differing = [k for k in sorted(dp) if k != root and dp[k] != dt[k]]
    if differing:
        k = differing[0]
        fields = sorted(f for f in set(dp[k]) | set(dt[k]) if dp[k].get(f) != dt[k].get(f))
        return {"where": "definition", "definition": k, "fields": fields, "n_differing": len(differing)}
    for key in ("types", "callees", "schema"):
        if pred.get(key) != traced.get(key):
            a, b = pred.get(key) or [], traced.get(key) or []
            i = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            return {"where": key, "index": i, "predicted": str(a[i] if i < len(a) else None)[:300], "traced": str(b[i] if i < len(b) else None)[:300],
                    "n_predicted": len(a), "n_traced": len(b)}
    bp, bt = dp[root]["body"], dt[root]["body"]
    cp, ct = pred["callees"], traced["callees"]
    tp, tt = pred["types"], traced["types"]
    for k, (x, y) in enumerate(zip(bp["nodes"], bt["nodes"])):
        if x == y:
            continue
        fx, fy = cp[x["fn"]], ct[y["fn"]]
        d = {"where": "root node", "node": k, "callee_predicted": fx[:200], "callee_traced": fy[:200]}
        if fx == fy:
            for ai, (ax, ay) in enumerate(zip(x.get("args") or [], y.get("args") or [])):
                if ax != ay:
                    d.update(arg=ai, type_predicted=str(tp[ax["type"]])[:160], type_traced=str(tt[ay["type"]])[:160], refs_predicted=_refs_view(ax),
                             refs_traced=_refs_view(ay))
                    break
            else:
                d["fields"] = sorted(f for f in set(x) | set(y) if x.get(f) != y.get(f))
        return d
    if len(bp["nodes"]) != len(bt["nodes"]):
        return {"where": "root length", "n_predicted": len(bp["nodes"]), "n_traced": len(bt["nodes"])}
    if bp.get("ret") != bt.get("ret"):
        return {"where": "root ret", "predicted": json.dumps(bp.get("ret"))[:300], "traced": json.dumps(bt.get("ret"))[:300]}
    return {"where": "annotations or encoding only"}


def equivalence(pred_program, traced_desc: dict) -> dict:
    from verity.ml.circuits.codec import decode_program
    from verity.ml.circuits.defs import REGISTRY
    from verity_vllm.program import descriptor_equivalence as DE
    from verity.ml.definitions.registry.catalog import load as load_registry
    load_registry()
    t0 = time.time()
    try:
        B = decode_program(traced_desc, REGISTRY)
        rep = DE.compare(pred_program, B, gate_samples=500, call_samples=12, eval_samples=2)
        return {"verdict": rep.get("verdict"), "secs": round(time.time() - t0, 1)}
    except DE.Mismatch as e:
        return {"verdict": "not equivalent", "mismatch": str(e)[:600], "secs": round(time.time() - t0, 1)}
    except Exception as e:  # noqa: BLE001 -- e.g. a traced descriptor naming a Definition this tree does not register
        return {"verdict": "not compared", "error": f"{type(e).__name__}: {e}"[:600], "secs": round(time.time() - t0, 1)}


def _diverge_task(task: dict) -> dict:
    t0 = time.time()
    out = {"key": task["key"]}
    try:
        r = _build(Phi.from_json(task["phi"]), tuple(task["shape"]))
        traced = load_descriptor(task["descriptor"])
        out["divergence"] = first_divergence(r.desc, traced)
        out["equivalence"] = equivalence(r.program, traced) if task.get("equivalence", True) else None
    except Exception as e:  # noqa: BLE001
        out["divergence"] = {"where": "error", "error": f"{type(e).__name__}: {e}"[:600]}
    out["secs"] = round(time.time() - t0, 2)
    return out


# ------------------------------------------------------------------------------------------------------------------- score


def _map_path(p: str | None, path_map: list[tuple[str, str]]) -> str | None:
    if not p:
        return p
    for old, new in path_map:
        if p.startswith(old):
            return new + p[len(old):]
    return p


def mem_estimate_mb(C, request_rows: list[int]) -> int:
    """A unit's peak resident memory, over-estimated: the root's nodes (the decoder's `nodes_per_row_layer` per row and layer) at
    6 KB each, and the attention reads of the earlier rows' keys (rows²/2 per layer) at 0.4 KB each.  Measured: a Qwen3-4B request
    of LP+T = 79 peaks at 814 MB (estimated 1.2 GB)."""
    rows = sum(request_rows)
    return int(250 + 0.006 * rows * C.NL * C.nodes_per_row_layer + 0.0002 * C.NL * sum(r * r for r in request_rows))


def _child(fn, task: dict, conn) -> None:
    import resource
    r = fn(task)
    r["max_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    conn.send(r)
    conn.close()


def _pool_map(fn, tasks: list[dict], workers: int, log, cache: "Cache | None" = None, mem_mb: int | None = None, run: bool = True) -> dict:
    """fn over tasks, each in its own spawned process: at most `workers` at a time, and, past the first, only while the running
    tasks' `mem_mb` estimates (`mem_estimate_mb`) sum to at most `mem_mb`. A process that dies without a result (the OOM killer)
    is recorded as `killed`, never waited on; results already in `cache` are reused and every new one is appended to it as it
    arrives. `run=False` returns the cached results only."""
    from multiprocessing.connection import wait
    out: dict = {}
    todo = []
    for t in tasks:
        hit = cache.get(t["key"]) if cache is not None else None
        if hit is not None:
            out[t["key"]] = hit
        elif run:
            todo.append(t)
    if workers <= 0:  # in this process (tests, debugging): no isolation from a unit that exhausts memory
        for t in todo:
            r = fn(t)
            out[r["key"]] = r
            if cache is not None:
                cache.put(r)
            log(r)
        return out
    ctx = mp.get_context("spawn")
    running: dict = {}

    def finish(r: dict) -> None:
        out[r["key"]] = r
        if cache is not None:
            cache.put(r)
        log(r)

    def admissible() -> int | None:
        if not running:
            return 0
        used = sum(t.get("mem_mb", 0) for _, _, t in running.values())
        return next((i for i, t in enumerate(todo) if mem_mb is None or used + t.get("mem_mb", 0) <= mem_mb), None)

    while todo or running:
        while todo and len(running) < workers and (i := admissible()) is not None:
            t = todo.pop(i)
            rd, wr = ctx.Pipe(duplex=False)
            p = ctx.Process(target=_child, args=(fn, t, wr))
            p.start()
            wr.close()
            running[p.sentinel] = (p, rd, t)
        ready = wait([s for s in running])
        for s in ready:
            p, rd, t = running.pop(s)
            r = None
            if rd.poll():
                try:
                    r = rd.recv()
                except EOFError:
                    r = None
            p.join()
            rd.close()
            if r is None:
                r = {"key": t["key"], "status": "killed", "reason": f"worker died without a result (exit code {p.exitcode}; "
                                                                    f"SIGKILL is the OOM killer) at rows {t.get('rows')}"}
            finish(r)
    return out


#: the code version of a cached result written before results carried one
UNSTAMPED = "unstamped"


def code_version() -> str:
    """The predictor's code: a digest of the files git tracks in the two packages a predicted Program is built from (`verity`,
    `verity_vllm`), as they are on disk.  Only tracked files count (`git ls-files`), so a file a test, a build or a run writes into
    a package's tree (a `.lake`, a cache, a report) never moves it; an edit to a tracked file does.  A namespace package's portions
    are hashed as one tree, by path under the package, and only those holding a module, so neither sys.path order nor a directory
    that joined the namespace from the cwd moves it.  A portion outside a git checkout is refused: nothing says which of its files
    are source."""
    import pkgutil
    import subprocess
    from pathlib import Path, PurePosixPath

    import verity
    import verity_vllm
    h = hashlib.sha256()
    for pkg in (verity, verity_vllm):
        files = []
        for root in (Path(r).resolve() for r in pkg.__path__ if any(pkgutil.iter_modules([r]))):
            ls = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True)
            if ls.returncode:
                raise RuntimeError(f"code_version: {root} is not in a git checkout, so its tracked files aren't known; run from a "
                                   f"checkout (research ships one) ({ls.stderr.decode(errors='replace').strip()})")
            files += [(PurePosixPath(rel), root / rel) for rel in ls.stdout.decode().split("\0") if rel]
        for rel, p in sorted(files):
            if p.is_file():
                h.update(f"{pkg.__name__}/{rel}\0".encode() + hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()[:16]


class Cache:
    """Task results by (code version, key), persisted as JSONL (a scoring run resumes where a suspended one stopped).  A result
    is stamped with the `code` that wrote it, and only results of `code`, or of a version `reuse` names (`UNSTAMPED` for results
    written before they carried one), are reused: an older predictor's refusal or Program never stands in for this one's.  Of
    several results for a key, `code`'s wins, then each `reuse` version's in the order named, then the last written."""

    def __init__(self, path: str | None, code: str, reuse: tuple[str, ...] = ()) -> None:
        self.path, self.code, self.d = path, code, {}
        rank = {v: i for i, v in reversed(list(enumerate((code, *reuse))))}
        if path and os.path.exists(path):
            for line in open(path):
                if line.strip():
                    r = json.loads(line)
                    v = r.get("code", UNSTAMPED)
                    held = self.d.get(r["key"])
                    if r.get("status") != "killed" and v in rank and (held is None or rank[v] <= rank[held.get("code", UNSTAMPED)]):
                        self.d[r["key"]] = r

    def get(self, key: str) -> dict | None:
        return self.d.get(key)

    def put(self, r: dict) -> None:
        r["code"] = self.code
        self.d[r["key"]] = r
        if self.path:
            with open(self.path, "a") as f:
                f.write(json.dumps(r, sort_keys=True) + "\n")


def plan(inv: dict, *, hf_home: str, checkpoints: dict | None, max_rows: int, rows_filter: str | None = None,
         max_mem_mb: int | None = None) -> dict:
    """The rows' φ, their units, and the distinct prediction tasks (one per (φ, shape)); a unit over `max_rows` or with an
    estimated peak (`mem_estimate_mb`) over `max_mem_mb` is `over_budget`."""
    import re
    rows, tasks, wtasks = [], {}, {}
    rx = re.compile(rows_filter) if rows_filter else None
    for rec in inv["rows"]:
        if rx and not rx.search(rec["row"]):
            continue
        row = {"row": rec["row"], "job": rec["job"], "dir": rec["dir"], "source_root": rec.get("source_root"), "units": []}
        try:
            phi = phi_of(rec, hf_home=hf_home, checkpoints=checkpoints)
            row["context_digest"] = phi.digest()
            pk = phi_key(phi)
            try:
                C = decoder(phi)
                pre = None
            except Unexpressible as e:
                pre = str(e)
        except Unexpressible as e:
            phi, pk, pre = None, None, str(e)
        row["context_error"] = pre if phi is None else None
        traced_req = set()
        for name, b in sorted(rec["builds"].items()):
            if INV.unit_kind(name) == "build_workload":
                continue
            uphi, upk = _of_rank(phi, pk, INV.unit_rank(name))
            shape = unit_shape(b)
            u = {"unit": name, "shape": list(shape), "rows": unit_rows(shape) if shape[0] else None, "traced_digest": (b.get("program") or {}).get("digest"),
                 "traced_function": (b.get("program") or {}).get("function"), "traced_n_root_nodes": (b.get("program") or {}).get("n_root_nodes"),
                 "descriptor": b.get("descriptor"), "descriptor_gz_bytes": b.get("descriptor_gz_bytes")}
            if shape[0] == "request":
                traced_req.add((shape[3], shape[4]))
            if b.get("ok") is not True or not u["traced_digest"]:
                u.update(status="no_trace", reason=str(b.get("message") or b.get("exception") or b.get("unsupported") or b.get("_error") or "Build dir has no Program")[:300])
            elif pre is not None:
                u.update(status="unexpressible", reason=pre)
            elif u["rows"] > max_rows:
                u.update(status="over_budget", reason=f"LP+T = {u['rows']} > max_rows {max_rows}")
            elif max_mem_mb is not None and (mem := mem_estimate_mb(C, [u["rows"]])) > max_mem_mb:
                u.update(status="over_budget", reason=f"estimated peak {mem} MB > max_mem_mb {max_mem_mb}")
            else:
                key = f"{upk}:{':'.join(str(x) for x in shape)}"
                u["key"] = key
                tasks.setdefault(key, {"key": key, "phi": uphi.to_json(), "shape": list(shape), "rows": u["rows"],
                                       "mem_mb": mem_estimate_mb(C, [u["rows"]])})
            row["units"].append(u)
        if phi is not None:
            pred_req = set(phi.request_shapes())
            row["request_shapes"] = {"traced": sorted(traced_req), "predicted": sorted(pred_req), "equal": traced_req >= pred_req,
                                     "traced_only": sorted(traced_req - pred_req), "predicted_only": sorted(pred_req - traced_req)}
        for name, wb in sorted(rec["builds"].items()):
            if INV.unit_kind(name) != "build_workload":
                continue
            uphi, _ = _of_rank(phi, pk, INV.unit_rank(name))
            u = {"unit": name, "traced_digest": wb.get("workload_digest"), "traced_lag": wb.get("lag"), "n_requests": wb.get("n_requests")}
            total = sum(int(lp) + int(t) for lp, t in phi.request_shapes()) if phi is not None else None
            u["rows"] = total
            if wb.get("ok") is not True or not wb.get("workload_digest"):
                u.update(status="no_trace", reason=str(wb.get("message") or wb.get("reason") or "no workload Program")[:300])
            elif pre is not None:
                u.update(status="unexpressible", reason=pre)
            elif total > max_rows:
                u.update(status="over_budget", reason=f"sum of request LP+T = {total} > max_rows {max_rows}")
            elif max_mem_mb is not None and (mem := mem_estimate_mb(C, [int(lp) + int(t) for lp, t in phi.request_shapes()])) > max_mem_mb:
                u.update(status="over_budget", reason=f"estimated peak {mem} MB > max_mem_mb {max_mem_mb}")
            else:
                key = f"W:{uphi.digest()[:16]}"
                u["key"] = key
                wtasks.setdefault(key, {"key": key, "phi": uphi.to_json(), "rows": total,
                                        "mem_mb": mem_estimate_mb(C, [int(lp) + int(t) for lp, t in phi.request_shapes()])})
            row["units"].append(u)
        rows.append(row)
    return {"rows": rows, "tasks": tasks, "wtasks": wtasks}


def score(inv: dict, *, hf_home: str, checkpoints: dict | None = None, max_rows: int = 600, max_mem_mb: int | None = None, workers: int = 2,
          mem_mb: int | None = 8000,
          path_map: list[tuple[str, str]] | None = None, diverge: bool = True, max_diverge: int = 40, equivalence_on: bool = True,
          rows_filter: str | None = None, workload: bool = True, cache: str | None = None, cached_only: bool = False,
          reuse_code: tuple[str, ...] = (), log=None) -> dict:
    """`max_rows`: request units over LP+T = max_rows (and workloads over that many summed rows) are `over_budget`; the units
    running at once have estimated peaks summing to at most `mem_mb` (`mem_estimate_mb`). `cached_only`: no prediction runs, the
    units without one in `cache` are `not_run` (the cached ones' first divergences are still located). `cache` reuses this
    code's results (`code_version`) and those of the versions `reuse_code` names; each unit records the version that predicted it."""
    t0 = time.time()
    log = log or (lambda r: print(json.dumps({k: v for k, v in r.items() if k != "phi"})[:400], file=sys.stderr, flush=True))
    code = code_version()
    C = Cache(cache, code, tuple(reuse_code))
    P = plan(inv, hf_home=hf_home, checkpoints=checkpoints, max_rows=max_rows, rows_filter=rows_filter, max_mem_mb=max_mem_mb)
    tasks = sorted(P["tasks"].values(), key=lambda t: t["rows"])
    pred = _pool_map(_predict_task, tasks, workers, log, C, mem_mb, run=not cached_only)
    wpred = _pool_map(_workload_task, sorted(P["wtasks"].values(), key=lambda t: t["rows"]), workers, log, C, mem_mb,
                      run=not cached_only) if workload else {}
    # divergence: one task per (prediction, traced digest), smallest first
    dtasks = {}
    for row in P["rows"]:
        for u in row["units"]:
            p = (wpred if INV.unit_kind(u["unit"]) == "build_workload" else pred).get(u.get("key")) if u.get("key") else None
            if p is None:
                if u.get("key"):
                    u.update(status="not_run", reason="not in the cache (cached_only)" if cached_only else
                             "workload prediction skipped" if INV.unit_kind(u["unit"]) == "build_workload" else "no prediction result")
                continue
            u["predict_secs"] = p.get("secs")
            u["predicted_by"] = p.get("code", UNSTAMPED)
            if p["status"] != "predicted":
                u.update(status=p["status"], reason=p.get("reason"))
                continue
            u["predicted_digest"] = p["digest"]
            u["predicted_function"] = p.get("function")
            u["predicted_n_root_nodes"] = p.get("n_root_nodes")
            u["exact"] = p["digest"] == u["traced_digest"]
            u["status"] = "exact" if u["exact"] else "differs"
            if u["exact"]:
                u["equivalence"] = {"verdict": "equivalent", "by": "equal digest"}
            elif INV.unit_kind(u["unit"]) != "build_workload":
                desc = _map_path(u.get("descriptor"), path_map or [])
                u["divergence_key"] = f"D:{u['key']}:{u['traced_digest'][:16]}"
                if diverge and desc and os.path.exists(desc):
                    dtasks.setdefault(u["divergence_key"], {"key": u["divergence_key"], "phi": P["tasks"][u["key"]]["phi"], "shape": u["shape"],
                                                            "descriptor": desc, "rows": u["rows"], "equivalence": equivalence_on,
                                                            "mem_mb": 3 * P["tasks"][u["key"]]["mem_mb"]})
    dt = sorted(dtasks.values(), key=lambda t: (C.get(t["key"]) is None, t["rows"]))[:len([k for k in dtasks if C.get(k)]) + max_diverge] \
        if diverge else []
    div = _pool_map(_diverge_task, dt, max(1, min(workers, 2)), log, C, mem_mb)
    for row in P["rows"]:
        for u in row["units"]:
            d = div.get(u.get("divergence_key")) if u.get("status") == "differs" else None
            if d is not None:
                u["first_divergence"] = d.get("divergence")
                u["equivalence"] = d.get("equivalence")
            elif u.get("status") == "differs" and INV.unit_kind(u["unit"]) != "build_workload":
                u["first_divergence"] = {"where": "not located", "reason": "traced descriptor not at hand or over the divergence budget"}
    res = {"schema": SCHEMA, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "inventory_root": inv.get("root"), "max_rows": max_rows,
           "code": code, "reuse_code": sorted(reuse_code), "rows": P["rows"], "n_prediction_tasks": len(tasks), "n_workload_tasks": len(P["wtasks"]), "n_divergence_tasks": len(dt),
           "secs": round(time.time() - t0, 1)}
    res["summary"] = summarize(res)
    return res


def _cause(u: dict) -> str:
    d = u.get("first_divergence") or {}
    w = d.get("where")
    if w == "definitions":
        return "definitions: only predicted " + ",".join(x.split("{")[0] for x in d.get("only_predicted", [])[:2]) + " / only traced " + \
            ",".join(x.split("{")[0] for x in d.get("only_traced", [])[:2])
    if w == "definition":
        return f"definition {str(d.get('definition')).split('{')[0]} ({','.join(d.get('fields') or [])})"
    if w == "root node":
        return f"root node: {str(d.get('callee_predicted')).split('{')[0]} vs {str(d.get('callee_traced')).split('{')[0]}" + \
            (f" arg {d['arg']}" if "arg" in d else "")
    if w == "program_fn":
        return "program_fn"
    return str(w)


def summarize(res: dict) -> dict:
    by = Counter()
    by_gran = Counter()
    by_tree = Counter()
    by_code = Counter()
    causes = Counter()
    unexpressible = Counter()
    rows_exact = rows_scored = 0
    for row in res["rows"]:
        sts = []
        for u in row["units"]:
            k = INV.unit_kind(u["unit"])
            g = "workload" if k == "build_workload" else ("step" if k == "build_step" else "request")
            by[u.get("status")] += 1
            by_gran[(g, u.get("status"))] += 1
            if u.get("status") in ("exact", "differs"):
                by_tree[(os.path.basename(str(row.get("source_root"))), u["status"])] += 1
                by_code[u.get("predicted_by", UNSTAMPED)] += 1
            if u.get("status") == "differs":
                causes[_cause(u)] += 1
            if u.get("status") == "unexpressible":
                unexpressible[str(u.get("reason"))[:160]] += 1
            sts.append(u.get("status"))
        compared = [s for s in sts if s in ("exact", "differs")]
        if compared:
            rows_scored += 1
            rows_exact += all(s == "exact" for s in compared)
    eq = sum(1 for row in res["rows"] for u in row["units"] if (u.get("equivalence") or {}).get("verdict") == "equivalent")
    return {"units_by_status": dict(by), "units_by_granularity": {f"{g}:{s}": n for (g, s), n in sorted(by_gran.items(), key=str)},
            "units_compared": by["exact"] + by["differs"], "units_exact": by["exact"], "units_equivalent": eq,
            "rows": len(res["rows"]), "rows_with_a_compared_unit": rows_scored, "rows_all_compared_exact": rows_exact,
            "top_divergence_causes": causes.most_common(10), "top_unexpressible": unexpressible.most_common(10),
            "compared_by_code": dict(sorted(by_code.items())),
            "compared_by_source_tree": {t: {"exact": by_tree[(t, "exact")], "differs": by_tree[(t, "differs")]}
                                        for t in sorted({t for t, _ in by_tree})}}


def table_md(res: dict) -> str:
    s = res["summary"]
    lines = [f"# predictor score ({res['utc']}, max_rows {res['max_rows']})", "",
             f"units compared {s['units_compared']}: exact {s['units_exact']}, equivalent {s['units_equivalent']}; rows {s['rows']}, "
             f"rows with a compared unit {s['rows_with_a_compared_unit']}, all-exact rows {s['rows_all_compared_exact']}", "",
             f"predictor code {res.get('code')}; compared units by the code that predicted them: "
             + (", ".join(f"{c} {n}" for c, n in s.get("compared_by_code", {}).items()) or "none"), "",
             "| granularity:status | units |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in s["units_by_granularity"].items()]
    lines += ["", "| first divergence | units |", "|---|---|"] + [f"| {c} | {n} |" for c, n in s["top_divergence_causes"]]
    lines += ["", "| unexpressible | units |", "|---|---|"] + [f"| {c} | {n} |" for c, n in s["top_unexpressible"]]
    lines += ["", "| traced from source tree | exact | differs |", "|---|---|---|"] + \
        [f"| {t} | {v['exact']} | {v['differs']} |" for t, v in s.get("compared_by_source_tree", {}).items()]
    lines += ["", "| row | step | request (exact/compared) | workload |", "|---|---|---|---|"]
    for row in res["rows"]:
        us = row["units"]
        st = "/".join(str(u.get("status")) for u in us if INV.unit_kind(u["unit"]) == "build_step") or "-"
        rq = [u for u in us if INV.unit_kind(u["unit"]).startswith("build_request")]
        rc = [u for u in rq if u.get("status") in ("exact", "differs")]
        wk = "/".join(str(u.get("status")) for u in us if INV.unit_kind(u["unit"]) == "build_workload") or "-"
        rs = f"{sum(u.get('status') == 'exact' for u in rc)}/{len(rc)} of {len(rq)}" if rc else (rq[0].get("status") if rq else "-")
        lines.append(f"| {row['job']}/{row['row']} | {st} | {rs} | {wk} |")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------------------------------- CLI


def _refuse_inside(out: str | None, roots: list[str]) -> None:
    if not out:
        return
    o = os.path.realpath(out)
    for r in roots:
        if r and (o == os.path.realpath(r) or o.startswith(os.path.realpath(r).rstrip("/") + "/")):
            raise UsageError(f"--out {out} is inside the rows root {r}: a score is never written into a Build row")


@dataclass(frozen=True, kw_only=True)
class Options:
    inventory: str | None = option("--inventory", default=None, help="inventory JSON(.gz) (`predict inventory`); or --rows")
    rows: str | None = option("--rows", default=None, help="rows root (<root>/<job>/<row>/build_summary.json), inventoried in place")
    hf_home: str = option("--hf-home", help="HF hub dir holding models--<org>--<name>/snapshots/<rev>/config.json")
    checkpoints: str | None = option("--checkpoints", default=None, help="manifests/checkpoints.json (the pinned revisions)")
    path_map: list[str] | None = option("--path-map", default=None, nargs="*", help="OLD=NEW prefixes for the inventory's descriptor paths")
    max_rows: int = option("--max-rows", type=int, default=600, help="skip request units with LP+T above this (over_budget)")
    mem_mb: int = option("--mem-mb", type=int, default=8000, help="the units running at once have estimated peaks summing to at most this")
    max_mem_mb: int | None = option("--max-mem-mb", type=int, default=None, help="skip units whose estimated peak is above this (over_budget)")
    workers: int = option("--workers", type=int, default=2)
    cache: str | None = option("--cache", default=None, help="JSONL of task results: reused, and appended to as results arrive")
    max_diverge: int = option("--max-diverge", type=int, default=40, help="locate the first divergence of at most this many differing units not in --cache")
    no_equivalence: bool = option("--no-equivalence", action="store_true", default=False)
    no_workload: bool = option("--no-workload", action="store_true", default=False)
    cached_only: bool = option("--cached-only", action="store_true", default=False, help="predict nothing: score the predictions already in --cache")
    reuse_code: list[str] | None = option("--reuse-code", default=None, nargs="*",
                                          help="also reuse --cache results of these predictor code versions, after this code's and "
                                               "in the order named (`unstamped`: written before results carried one); by default "
                                               "only this code's are")
    rows_filter: str | None = option("--rows-filter", default=None, help="regex on the row id")
    out: str = option("--out", help="score JSON")
    md: str | None = option("--md", default=None, help="markdown table")


def main(a: Options) -> int:
    if not (a.inventory or a.rows):
        raise UsageError("--inventory or --rows")
    inv = INV.load(a.inventory) if a.inventory else INV.inventory(a.rows)
    _refuse_inside(a.out, [inv.get("root"), a.rows])
    _refuse_inside(a.md, [inv.get("root"), a.rows])
    _refuse_inside(a.cache, [inv.get("root"), a.rows])
    checkpoints = json.load(open(a.checkpoints)) if a.checkpoints else None
    pm = [tuple(x.split("=", 1)) for x in (a.path_map or [])]
    res = score(inv, hf_home=a.hf_home, checkpoints=checkpoints, max_rows=a.max_rows, max_mem_mb=a.max_mem_mb, mem_mb=a.mem_mb, workers=a.workers, path_map=pm,
                max_diverge=a.max_diverge, equivalence_on=not a.no_equivalence, rows_filter=a.rows_filter, workload=not a.no_workload,
                cache=a.cache, cached_only=a.cached_only, reuse_code=tuple(a.reuse_code or ()))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    if a.md:
        with open(a.md, "w") as f:
            f.write(table_md(res))
    print(json.dumps(res["summary"], indent=1), file=sys.stderr)
    return 0
