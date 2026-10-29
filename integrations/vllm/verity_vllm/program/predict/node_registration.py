"""θ of a tensor-parallel node's Program (`node`) registered from its TP Commit of record: the registration log the inference-only
policy's full check reads (`pipeline.registration_log`), each rank's weights opened against the rank's committed weights root.

    verity-vllm predict node-registration --row ROW --config CONFIG [--wrapper request --LP N --T N | --wrapper step --M N --pos0 N]
        --program NODE_DESCRIPTOR --commit-out OUT --checkpoint-dir CK [--checkpoints-manifest M --model REPO --revision REV]
        --window JSON --out DIR --salts DIR [--workers N]

The parameters are the node's (`node.node_parameters` of the row's φ), held to the descriptor before anything is opened: they lower
onto its root's input nodes in order (width and members), and its θ is exactly their weights' input nodes. θ parameter
`rank<r>.weights[i]` is field i of rank r's weights record, and its value is the tensor rank r registered under that field's name
in the Commit of record:

* the Commit's instrumented run passed (`runs.jsonl`), and rank r's bundle (`replay_bundle_p<pair>_rank<r>/`) opens against its
  manifest (`replay_bundle.open_bundle`), whose sha256 `replay_deferred.json` names for the rank;
* rank r's Build of record (the bundle's `replay.programs`, or `--build-rank`) holds Programs whose digests are of record for the
  rank (`replay.of_record`), and its trace names field i `<prefix><name>` (`derive-report.json` `provenance.weights`), one prefix
  for every field: so field i of the node's rank r is the registered tensor field i of the rank's own Program reads;
* the tensor is composed from the checkpoint at the rank's (world, rank), as the Build's `weight_roles` say vLLM rewrites it, or
  read from the bundle, and opened against its tensor root under the rank's committed weights root (`cpu_replay.opener`, as the
  Commit's replay opens it), in the field's width and leaf count.

The tensor roots decide each value. With --checkpoints-manifest the checkpoint is also its pinned one, every shard's sha256 the
manifest's (`checkpoint_index.shard_pin`), else refused; without it no shard is hashed.

One spawned process per rank (`--workers` at a time) opens and commits its tensors one at a time, so it holds a tensor, not the
rank's weights (95 GB a GLM-5.3 B1 rank), and another reads the descriptor before them. Under --out: `record.json` (the
registration record), `trusted.json` (nci-check's trusted input), `provenance.json` (the instance's θ provenance) and
`registration.json` (digests, the checkpoint's pin, per-rank counts and timings).
Under --salts: each value's row salts, `<value>.salts`, the prover's, which nothing public carries, so --salts is refused inside
--out. Exit 0 written, 2 refused.
"""
from __future__ import annotations

import gzip
import json
import multiprocessing
import os
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verity_vllm.config import UsageError, option


@dataclass(frozen=True, kw_only=True)
class Options:
    row: str = option("--row", help="the TP row (`verity_vllm/workloads/<row>.json`)")
    config: str = option("--config", help="the pinned checkpoint's config.json")
    wrapper: str = option("--wrapper", default="request", choices=["step", "request"])
    M: int = option("--M", type=int, default=1)
    pos0: int = option("--pos0", type=int, default=8)
    LP: int | None = option("--LP", type=int, default=None)
    T: int | None = option("--T", type=int, default=None)
    program: str = option("--program", help="the node Program's descriptor (`predict node`'s node_descriptor.json.gz)")
    commit_out: str = option("--commit-out", help="the TP Commit of record's --out (runs.jsonl, replay_bundle_p<pair>_rank<r>/)")
    pair: int | None = option("--pair", type=int, default=None, help="the instrumented pair (default: the Commit's only one)")
    build_rank: list[str] = option("--build-rank", action="append", default=[],
                                   help="rank r's Build dir on this host, repeated in rank order (default: the bundle's)")
    checkpoint_dir: str = option("--checkpoint-dir", help="the checkpoint of record (safetensors shards + config.json)")
    checkpoints_manifest: str | None = option("--checkpoints-manifest", default=None, help="manifests/checkpoints.json (shard sha256 pins)")
    model: str | None = option("--model", default=None, help="the checkpoint's repo (its checkpoints.json entry)")
    revision: str | None = option("--revision", default=None, help="the checkpoint's revision")
    window: str = option("--window", help="the registration's window, a JSON object (the record's `window`)")
    out: str = option("--out", help="where the record, trusted input, provenance and summary go")
    salts: str = option("--salts", help="where each value's row salts go (private: never under --out)")
    workers: int = option("--workers", type=int, default=8, help="ranks registered at once, a spawned process each (1: in this process)")
    cmd: str = field(default="node-registration", init=False)


class Refused(Exception):
    """The node's parameters are not the descriptor's, or a rank's θ is not what its Commit of record registered."""


def load_descriptor(path: str | os.PathLike) -> dict[str, Any]:
    with (gzip.open(path, "rb") if str(path).endswith(".gz") else open(path, "rb")) as f:
        return json.loads(f.read())


def root_inputs(desc: dict[str, Any]) -> list[tuple[int, int, int]]:
    """(root node, width, members) of each input node of the descriptor's root, in order, read off its JSON (no decode)."""
    from verity.ml.circuits.program import input_id

    callees = desc.get("callees") or []
    out = []
    for k, n in enumerate(desc["definitions"][desc["root"]]["body"]["nodes"]):
        got = input_id(callees[n["fn"]] if isinstance(n["fn"], int) else n["fn"])
        if got is not None:
            out.append((k, got[1], int(n["n"]) if n["form"] in ("batch", "scan") else 1))
    return out


def theta_nodes(desc: dict[str, Any], params, theta: tuple[str, ...]) -> list[int]:
    """The descriptor's θ, once `params` are shown to lower onto its root's input nodes and θ to be exactly theirs in `theta`."""
    from verity_vllm.pipeline import registration_log as RL

    lowered = [(name, w, n) for name, t in params for w, n in RL.lower(t)]
    nodes = root_inputs(desc)
    if [(w, n) for _, w, n in lowered] != [(w, n) for _, w, n in nodes]:
        at = next((i for i, (a, b) in enumerate(zip(lowered, nodes)) if a[1:] != b[1:]), min(len(lowered), len(nodes)))
        raise Refused(f"the node's parameters lower to {len(lowered)} input nodes, the descriptor's root has {len(nodes)}, "
                      f"first apart at input node {at}: these are not its parameters")
    held = set(theta)
    want = [k for (name, _w, _n), (k, _w2, _n2) in zip(lowered, nodes) if name in held]
    got = list((desc.get("annotations") or {}).get("theta") or [])
    if got != want:
        raise Refused(f"the descriptor's θ is {len(got)} root nodes, the node's weights parameters lower to {len(want)} others")
    return got


def descriptor_theta(path: str, params: list[tuple[str, Any]], theta: tuple[str, ...]) -> tuple[str, list[int]]:
    """The descriptor's Program digest and θ (`theta_nodes`), `params` as `[(name, type JSON)]`."""
    from verity.ml.circuits import codec
    from verity.ml.circuits.types import type_from_json

    desc = load_descriptor(path)
    return codec.program_digest(desc), theta_nodes(desc, [(n, type_from_json(t)) for n, t in params], theta)


def commit_of_record(out: Path, pair: int | None) -> tuple[int, dict[str, Any], dict[str, Any]]:
    """(pair, its instrumented run's commit block, replay_deferred.json) of a TP Commit whose run passed; else Refused."""
    rows = [json.loads(ln) for ln in (out / "runs.jsonl").read_text().splitlines() if ln.strip()]
    inst = [r for r in rows if r.get("arm") != "control" and "commit" in r and (pair is None or int(r["pair"]) == pair)]
    if len(inst) != 1:
        raise Refused(f"{out / 'runs.jsonl'}: {len(inst)} instrumented runs" + (f" of pair {pair}" if pair is not None else "; name one with --pair"))
    c = inst[0]["commit"]
    if c.get("commit_pass") is not True:
        raise Refused(f"pair {inst[0]['pair']}: the Commit of record did not pass ({c.get('commit_fail_reason')})")
    return int(inst[0]["pair"]), c, json.loads((out / "replay_deferred.json").read_text())


def program_digest_of(d: str) -> str:
    res = json.loads(Path(d, "result.json").read_text())
    return str((res.get("program") or {}).get("digest") or res.get("program_digest") or "")


def trace_fields(d: str) -> list[str]:
    """The weights record's field names, in field order, as the Build's trace gave them (`aot.traced_names`)."""
    rep = json.loads(Path(d, "derive-report.json").read_text())
    return [w["name"] for w in sorted(rep["provenance"]["weights"], key=lambda w: w["field"])]


def rank_job(o: Options, out: Path, pair: int, c: dict[str, Any], dd: dict[str, Any], P: Any, rank: int, pin: dict[str, Any] | None,
             digest: str) -> dict[str, Any]:
    """What rank `rank`'s process registers, once its bundle, Build and trace are shown to be the record's (module docstring)."""
    from verity_vllm.pipeline import cpu_replay as CPU
    from verity_vllm.pipeline import replay_bundle as RB
    from verity_vllm.pipeline.tp.rank_replay import request_program_dirs

    world = len(P.phis)
    try:
        b = RB.open_bundle(RB.path(out, pair, rank=rank))
    except (OSError, ValueError) as e:
        raise Refused(f"rank {rank}: {e}") from e
    ent = next((x for x in dd.get("bundles") or [] if (int(x["pair"]), int(x["rank"])) == (pair, rank)), None)
    ctx = b["context"]
    if ent is None or ent.get("manifest_sha256") != b["manifest_sha256"]:
        raise Refused(f"rank {rank}: replay_deferred.json names no bundle of pair {pair} with manifest sha256 {b['manifest_sha256'][:16]}")
    if (int(ctx["pair"]), int(ctx["tp"]["rank"]), int(ctx["tp"]["world"])) != (pair, rank, world):
        raise Refused(f"rank {rank}: its bundle is pair {ctx['pair']} tp {ctx['tp']}, not pair {pair} rank {rank} of world {world}")
    cfg = ctx["replay"]
    root = o.build_rank[rank] if o.build_rank else cfg["programs"][str(rank)]
    try:
        dirs = sorted(set(request_program_dirs(str(root), list(cfg.get("requests") or []), rank).values()))
    except RuntimeError as e:
        raise Refused(str(e)) from e
    of_record = set((cfg.get("of_record") or {}).get(str(rank)) or [])
    bad = [d for d in dirs if program_digest_of(d) not in of_record]
    if not dirs or bad:
        raise Refused(f"rank {rank}: {bad or root} holds no Program of record for the rank ({sorted(x[:16] for x in of_record)})")
    names = [n for n, _t in P.leaves[rank]]
    traced = {d: trace_fields(d) for d in dirs}
    first = traced[dirs[0]]
    prefix = first[0][:len(first[0]) - len(names[0])] if first and first[0].endswith(names[0]) else None
    if prefix is None or any(t != [prefix + n for n in names] for t in traced.values()):
        raise Refused(f"rank {rank}: its Build's trace names its {len(first)} weights fields otherwise than the node's {len(names)} "
                      f"(first {first[:1]} vs {names[:1]}): the fields are not one weights record")
    weights_root = ((c.get("per_rank") or {}).get(str(rank)) or {}).get("weight_registration", {}).get("root")
    return {"rank": rank, "world": world, "bundle": b["path"], "manifest_sha256": b["manifest_sha256"], "root": weights_root,
            "roles": CPU.registered_roles(dirs, prefix), "prefix": prefix, "program": digest, "salts": o.salts,
            "fields": [(f"rank{rank}.weights[{i}]", n, t.to_json()) for i, (n, t) in enumerate(P.leaves[rank])],
            "checkpoint_dir": o.checkpoint_dir, "checkpoints_manifest": o.checkpoints_manifest, "pin": pin}


def register_rank(job: dict[str, Any]) -> dict[str, Any]:
    """Rank `job["rank"]`'s θ parameters committed (`registered_rows.commit`) one tensor at a time, each opened against the rank's
    committed weights root, its salts written to `<salts>/<value>.salts`: the entries, without salts."""
    from verity.ml.circuits.types import type_from_json
    from verity_vllm.check import weights_of_record as WR
    from verity_vllm.commit import registered_rows as RR
    from verity_vllm.pipeline import cpu_replay as CPU
    from verity_vllm.pipeline import registration_log as RL

    t0, r, bundle = time.perf_counter(), int(job["rank"]), Path(job["bundle"])
    tree = json.loads((bundle / "weights" / "tree.json").read_text())
    _folded, why = CPU.tree_refusal(tree, job["root"])
    if why:
        raise Refused(f"rank {r}: {why}")
    ck = WR.CheckpointIndex(job["checkpoint_dir"], checkpoints_manifest=job["checkpoints_manifest"], pin=job["pin"], verify_shards=False)
    open_one = CPU.opener(bundle, tree, ck, tp={"world": int(job["world"]), "rank": r}, roles=job["roles"])
    at = {name: i for i, (name, *_x) in enumerate(tree["names"])}
    entries, by = {}, {"checkpoint": 0, "bundle": 0}
    for param, name, tj in job["fields"]:
        t, i = type_from_json(tj), at.get(name)
        if i is None:
            raise Refused(f"rank {r}: {param} is {name!r}, which the rank's committed weights tree does not register")
        dt = tree["names"][i][1]
        if CPU.WORDS.get(dt, (None,))[0] != next(iter(t.width_set())):
            raise Refused(f"rank {r}: {param} ({name}) is {t}, the rank registered it as {dt}")
        src, raw, why, _read = open_one(i)
        if raw is None:
            raise Refused(f"rank {r}: {name} does not open against the committed weights root ({why})")
        words = raw.view(CPU.WORDS[dt][1])
        if words.size != t.leaves:
            raise Refused(f"rank {r}: {name} is {words.size} words, {param} holds {t.leaves} leaves")
        c = RR.commit(job["program"], param, words, RL.word_bits(t))
        Path(job["salts"], f"{param}.salts").write_bytes(c.salts)
        entries[param] = c.entry
        by[src] += 1
    return {"rank": r, "entries": entries, "opened_by_source": by, "manifest_sha256": job["manifest_sha256"], "weights_root": job["root"],
            "prefix": job["prefix"], "wall_s": round(time.perf_counter() - t0, 3)}


def each_rank(jobs: list[dict[str, Any]], workers: int) -> Iterator[dict[str, Any]]:
    """`register_rank` of each job as it finishes: in this process when `workers` is 1, else in spawned processes, every other one
    terminated once one is refused."""
    if workers <= 1:
        yield from map(register_rank, jobs)
        return
    with multiprocessing.get_context("spawn").Pool(min(workers, len(jobs)), maxtasksperchild=1) as pool:
        yield from pool.imap_unordered(register_rank, jobs)


def register(o: Options) -> dict[str, Any]:
    from verity_vllm.check import weights_of_record as WR
    from verity_vllm.pipeline import registration_log as RL
    from verity_vllm.program.predict import aot as A
    from verity_vllm.program.predict.node import node_parameters

    t0 = time.perf_counter()
    out, salts = Path(o.out).resolve(), Path(o.salts).resolve()
    if salts == out or salts.is_relative_to(out):
        raise UsageError(f"--salts {o.salts} is under --out {o.out}: the salts are the prover's, and --out is what is published")
    window = json.loads(o.window)
    if not isinstance(window, dict):
        raise UsageError("--window is a JSON object")
    if o.wrapper == "request" and (o.LP is None or o.T is None):
        raise UsageError("--wrapper request needs --LP and --T")
    shape = {"M": o.M, "pos0": o.pos0} if o.wrapper == "step" else {"LP": o.LP, "T": o.T}
    P = node_parameters(A.phi_at(o.row, 0, o.config), o.wrapper, **shape)
    args = (o.program, [(n, t.to_json()) for n, t in P.params], P.theta)
    if o.workers <= 1:
        digest, theta = descriptor_theta(*args)
    else:
        with multiprocessing.get_context("spawn").Pool(1) as pool:  # B1's decoded descriptor is ~30 GB: never this process's
            digest, theta = pool.apply(descriptor_theta, args)
    t_desc = time.perf_counter()
    commit_out = Path(o.commit_out)
    pair, c, dd = commit_of_record(commit_out, o.pair)
    ck = WR.CheckpointIndex(o.checkpoint_dir, checkpoints_manifest=o.checkpoints_manifest, model=o.model, revision=o.revision,
                            verify_shards=bool(o.checkpoints_manifest), workers=o.workers)
    unpinned = [fn for fn, s in ck.pin["shards"].items() if s["pinned"] is not True]
    if o.checkpoints_manifest and unpinned:
        raise Refused(f"{o.checkpoint_dir}: shards {unpinned[:4]} ({len(unpinned)} of {len(ck.shards)}) are not the ones "
                      f"{o.checkpoints_manifest} pins for {ck.pin['entry']}")
    pin = ck.pin if o.checkpoints_manifest else None
    checkpoint = {"dir": o.checkpoint_dir, "manifest": o.checkpoints_manifest, "entry": ck.pin["entry"],
                  "pinned_shards": len(ck.shards) if pin else 0}
    t_ck = time.perf_counter()
    jobs = [rank_job(o, commit_out, pair, c, dd, P, r, pin, digest) for r in range(len(P.phis))]
    del ck
    t_jobs = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    salts.mkdir(parents=True, exist_ok=True)
    ranks = []
    for rk in each_rank(jobs, o.workers):
        ranks.append(rk)
        print(f"[node-registration] rank {rk['rank']}: {len(rk['entries'])} values committed ({rk['opened_by_source']}) in "
              f"{rk['wall_s']} s", flush=True)
    ranks.sort(key=lambda rk: rk["rank"])
    entries = {k: v for rk in ranks for k, v in rk.pop("entries").items()}
    reg = RL.record(digest, P.params, theta, entries, window)
    for name in ("record", "trusted", "provenance"):
        (out / f"{name}.json").write_text(json.dumps(reg[name], separators=(",", ":")) + "\n")
    doc = {"schema": "verity-vllm/node-registration/v1", "program_digest": digest, "record_digest": reg["digest"], "row": o.row,
           "wrapper": o.wrapper, "shape": shape, "commit_out": str(commit_out), "pair": pair, "window": window,
           "theta_parameters": len(P.theta), "theta_nodes": len(theta), "rows": sum(e["leaves"] for e in entries.values()),
           "checkpoint": checkpoint, "ranks": ranks, "salts": str(salts),
           "timings": {"descriptor_s": round(t_desc - t0, 3), "checkpoint_index_s": round(t_ck - t_desc, 3),
                       "ranks_checked_s": round(t_jobs - t_ck, 3), "commit_s": round(time.perf_counter() - t_jobs, 3),
                       "total_s": round(time.perf_counter() - t0, 3)}}
    (out / "registration.json").write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def main(o: Options) -> int:
    try:
        doc = register(o)
    except Refused as e:
        print(f"[node-registration] REFUSED: {e}", flush=True)
        return 2
    print(f"[node-registration] record {doc['record_digest'][:16]} registers {doc['theta_parameters']} θ parameters "
          f"({doc['rows']} rows) of Program {doc['program_digest'][:16]} in {doc['timings']['total_s']} s", flush=True)
    sys.stdout.flush()
    return 0
