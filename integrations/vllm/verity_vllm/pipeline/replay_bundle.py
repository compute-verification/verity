"""The replay bundle: what a Commit with `--replay-deferred` leaves for its C2 replay, which then runs on CPU without the engine.

The GPU Commit runs the committed run, the openings, the weights pin and the commitment, and instead of the sampled replay + boundary
linkage writes one bundle per instrumented run beside its outputs, `<out>/replay_bundle_p<pair>/`:

    store/            the committed store (`commit.store_dump`): per step the metas, the retained bytes and the tree levels; the run root
    weights/tree.json the committed weights root and the tree under it (`register_weights`): geo digest, chunk, the registered tensors
                      [name, dtype, shape, nbytes] in order and their tensor roots; `extra` names the registered tensors whose bytes
                      are in the bundle (buffers and engine constants, not checkpoint tensors)
    weights/extra/    those tensors' bytes, `<index>.bin` by their position in `names`
    binding_map.json  the run's binding map (`binding_map_p<pair>.json`)
    context.json      everything else the replay reads: the requests, sampling, served tokens, stop rule of record, the replay's
                      arguments and Program refs, the required manifest's path and sha256, the retention, the Commit's row as recorded
                      (weights pin, coverage, openings), verdict inputs (declaration, required classes) and the engine's
                      configuration of record (`engine_facts_of_record`, read in the Commit's process)
    MANIFEST.json     {schema, files: {path: {sha256, bytes}}}, written last

It is built under `replay_bundle_p<pair>.partial/` and renamed into place once sealed, so a bundle directory that exists is complete.
A full store is written from the moment the run root is fixed, beside the binding and manifest coverage (`start`), so the GPU is not
held for the two in turn; the bundle's device-resident tensors and the engine's facts are read before `start` returns, so nothing
after it reads the device and a leased Commit can give its GPU back there (`gpu_lease.release`); a slim one is written at the C2
site, after the challenge its reads come from.
`open_bundle` checks every file against the manifest before anything is read from it.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from verity_vllm.commit import store_dump as SD
from verity_vllm.pipeline import c2_replay as C2

SCHEMA = "verity-vllm/replay-bundle/v1"
MANIFEST = "MANIFEST.json"
#: the context's (unwritten) key for the sha256 of the files `write_store` hashed as it wrote them
HASHED = "_hashed"
ARGS = ("program_dir", "programs_root", "program_digest", "required_manifest", "replay_per_stratum", "replay_max", "replay_target", "replay_k",
        "replay_draw", "replay_workers", "replay_seed", "replay_challenge", "window_mb", "window_slots", "retain", "label", "serving_rows", "openings",
        "checkpoints", "workload", "weights_of_record", "checkpoint_dir")


def path(out: str | os.PathLike, pair: int, *, rank: int | None = None) -> Path:
    """The bundle of instrumented run ``pair``; a TP Commit's has one per rank (`pipeline.tp.commit_replay`)."""
    return Path(out) / (f"replay_bundle_p{pair}" + (f"_rank{rank}" if rank is not None else ""))


def partial(out: str | os.PathLike, pair: int, *, rank: int | None = None) -> Path:
    p = path(out, pair, rank=rank)
    return p.with_name(p.name + ".partial")


def deferred_records(bundle: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """The Commit's sampled-replay and boundary-linkage records while their replay is deferred: not evaluated, named, with the bundle."""
    why = f"deferred: the C2 replay runs on CPU from the replay bundle {bundle} (`verity-vllm row stage replay`)"
    return ({"schema": "verity-sweep/sampled-replay/v1", "method": "sampled-exact-replay", "result": None, "grade": "deferred", "why": why,
             "deferred": {"bundle": str(bundle)}},
            {"schema": "verity-sweep/boundary-linkage/v1", "result": None, "why": why, "pairs": 0, "linked": 0, "deferred": {"bundle": str(bundle)}})


def refuse(args: Any) -> None:
    """Exits by name when `--replay-deferred` cannot run with these Commit flags."""
    if not args.replay_deferred:
        return
    why = ("no --program-dir / --programs-root: there is no C2 replay to defer" if not (args.program_dir or args.programs_root) else
           "one instrumented committer only (no --arms, no control-only process)" if args.arms or args.only_arm == "control" else
           "--padded-finalize: padding steps are not in the replay bundle's store" if args.padded_finalize else
           f"--compiled-taps {args.compiled_taps}: the compiled check is its own C2" if args.compiled_taps != "off" else
           "--vu-export-dir exports from the replay, which does not run here" if args.vu_export_dir else
           "--serving-rows commits in the in-process replay; `verity-vllm serving-rows BUNDLE` writes a deferred Commit's"
           if args.serving_rows else None)
    if why:
        raise SystemExit(f"--replay-deferred: {why}")


@dataclass
class Started:
    """A full store's write running beside the checks before the C2 site (`start`), and the engine's facts read when it began."""
    future: Future
    hashed: dict[str, str]
    engine: dict[str, Any]


def start(out: Path, pair: int, com: Any, rc: Any, args: Any, protocols: Any) -> Started | None:
    """The full store's write, started once the run root is fixed, in a thread beside the binding and manifest coverage that precede the
    C2 site: the store is final at finalize and every one of those checks only reads it, so `defer` takes the same files and digests
    without waiting for both in turn.  The registered tensors the bundle carries (`weights_extra`, read from the device) and the engine's
    facts of record are read here, before it returns.  None when `defer` would not write the full store from here: no deferral, a slim
    one (its reads come from the challenge), a live-only replay, a retention other than host, or a store over the cap."""
    if (not args.replay_deferred or args.replay_deferred == "slim" or not getattr(args, "replay_bundle_overlap", 1)
            or C2.live_only(SimpleNamespace(protocols=protocols)) or C2.retention_of(com, args) != "host"
            or SD.nbytes(com) > float(getattr(args, "replay_bundle_cap_gb", 300)) * 1e9):
        return None
    from verity_vllm.check.replay.driver import engine_facts_of_record

    hashed: dict[str, str] = {}
    try:
        extras: Any = weights_extra(com, rc)
    except ValueError as e:                          # raised where the write is taken (`defer`), as before
        extras = e
    engine = engine_facts_of_record()
    pool = ThreadPoolExecutor(1, thread_name_prefix="replay-bundle")

    def write() -> float:
        t0 = time.perf_counter()
        if isinstance(extras, ValueError):
            raise extras
        write_store(out, pair, com, rc, digests=hashed, extras=extras)
        return time.perf_counter() - t0

    fut = pool.submit(write)
    pool.shutdown(wait=False)
    return Started(fut, hashed, engine)


def prepare(args: Any) -> Any:
    """`refuse`, then the slim bundle's planner (`replay_plan.Planner`), started now so it prepares while the engine runs."""
    refuse(args)
    if args.replay_deferred != "slim":
        return None
    from verity_vllm.pipeline.replay_plan import Planner

    return Planner(args)


def defer(out: Path, pair: int, com: Any, rc: Any, args: Any, *, tl: Any, arm: str, versions: dict, reqs: list[dict], sampling: dict,
          tokens: dict, stop_of_record: Any, ids_full: Any, declaration: Any, required_classes: Any, vc_result: Any = None,
          plan: Callable[[], dict[str, Any]] | None = None, row: dict[str, Any], deferred: dict[int, dict],
          started: Started | None = None) -> bool:
    """At the C2 site of a deferred run: writes the bundle's store and weights tree (or takes the write `start` began), puts the run's
    deferred sampled-replay and linkage records on `row` and the bundle's context in `deferred[pair]` (sealed once the run's record is
    final, after the weights pin and the openings).  ``plan`` makes the replay's committed reads (`c2_replay.plan`); with it the store is
    SLIM: only the ranges and tree nodes they opened.  False, with nothing written, when the store would exceed
    `args.replay_bundle_cap_gb`: the replay then runs in process."""
    from verity_vllm.check.replay.driver import engine_facts_of_record

    tl.start("validate.replay_bundle", pair=pair)
    slim, t = None, None
    if started is not None:
        t0 = time.perf_counter()
        write_s, hashed = started.future.result(), started.hashed
        print(f"[delta] REPLAY BUNDLE (pair {pair}): store written in {write_s:.1f} s beside the checks before the C2 site "
              f"(waited {time.perf_counter() - t0:.1f} s here)", flush=True)
    else:
        if plan is not None:
            t0 = time.perf_counter()
            try:
                with SD.touched(com) as t:
                    planned = plan()
            except Exception as e:  # noqa: BLE001  -- no plan, no slim bundle: the replay runs in this process
                tl.end("validate.replay_bundle", pair=pair, ok=False)
                print(f"[delta] REPLAY BUNDLE (pair {pair}): no plan ({e!r:.300}); the C2 replay runs in this process", flush=True)
                return False
            sr_p = planned.get("replay") or {}
            slim = {"bytes": t.nbytes(), "picked": sr_p.get("picked"), "unresolved_n": sr_p.get("unresolved_n"), "unresolved": sr_p.get("unresolved"),
                    "linkage": planned.get("linkage"), "plan_s": round(time.perf_counter() - t0, 3)}
            print(f"[delta] REPLAY BUNDLE (pair {pair}): slim -- {slim['picked']} picks + linkage read {slim['bytes'] >> 20} MiB of the store "
                  f"in {slim['plan_s']} s (unresolved {slim['unresolved_n']})", flush=True)
        size, cap = t.nbytes() if t is not None else SD.nbytes(com), float(getattr(args, "replay_bundle_cap_gb", 300)) * 1e9
        if size > cap:
            tl.end("validate.replay_bundle", pair=pair, ok=False)
            print(f"[delta] REPLAY BUNDLE (pair {pair}): REFUSED -- {size / 1e9:.1f} GB is over the {cap / 1e9:.0f} GB cap (--replay-bundle-cap-gb); "
                  "the C2 replay runs in this process", flush=True)
            return False
        hashed = {}
        write_store(out, pair, com, rc, digests=hashed, touched=t)
    caps = {str(q.get("request_id")): int(q.get("max_tokens") or sampling.get("max_tokens") or 1) for q in reqs}
    ctx = {"pair": pair, "arm": arm, "committer": com.name, "run_root": rc.run_root.hex(), "code": versions.get("git_head"),
           "args": {k: getattr(args, k, None) for k in ARGS}, "requests": reqs, "sampling": sampling, "tokens": tokens,
           "stop_of_record": stop_of_record,
           # the reconciliation of the not-executed class reads the placed identities only when a request stopped early
           "ids_full": ids_full if any(len(v) < caps.get(str(k), 0) for k, v in tokens.items()) else None,
           "retention": C2.retention_of(com, args),
           "required_manifest_sha256": hashlib.sha256(Path(args.required_manifest).read_bytes()).hexdigest() if args.required_manifest else None,
           "declaration": declaration, "required_classes": list(required_classes), "min_openings": min(64, args.openings),
           "vc_result": vc_result, "engine_of_record": started.engine if started is not None else engine_facts_of_record(), "slim": slim,
           HASHED: hashed}
    row["commit"]["sampled_replay"], row["commit"]["boundary_linkage"] = deferred_records(path(out, pair))
    deferred[pair] = ctx
    tl.end("validate.replay_bundle", pair=pair)
    print(f"[delta] C2 SAMPLED REPLAY + LINKAGE (pair {pair}): DEFERRED -- store and weights tree written to {partial(out, pair)}", flush=True)
    return True


PENDING = "PENDING: the C2 sampled replay + boundary linkage are deferred to the replay bundle"


def pending_verdict(commit: dict, tokens_equal: bool | None, *, min_openings: int = 64, declaration: dict | None = None,
                    required_classes: Any = None) -> tuple[bool | None, str]:
    """The verdict of an instrumented run whose C2 replay is deferred: (False, reason) when a check the Commit ran itself fails -- C1, a
    failing value-correspondence component (oracle compare, weights of record), the openings after release -- else (None, PENDING):
    `check.verdict.commit_verdict` decides once the replay is on the record."""
    from verity_vllm.check.verdict import _after_release_verdict, _c1_verdict

    wp, vc = commit.get("weights_pin"), commit.get("value_correspondence")
    fail = (_c1_verdict(commit, tokens_equal, min_openings, declaration, required_classes)
            or ((False, f"weights of record: {wp.get('why') or 'served weights differ from the checkpoint of record'}")
                if isinstance(wp, dict) and wp.get("result") is False else None)
            or ((False, vc.get("why") or f"value correspondence FAIL ({vc.get('method')})") if isinstance(vc, dict) and vc.get("result") is False else None)
            or _after_release_verdict(commit, min_openings))
    return fail if fail is not None else (None, PENDING)


def awaits_replay(c: dict) -> bool:
    return c.get("commit_pass") is None and (c.get("sampled_replay") or {}).get("grade") == "deferred"


def pass_of(rows: list[dict]) -> bool | None:
    """The Commit's X-03 pass over its instrumented runs: True when every one passed, None when none failed and some await their
    deferred replay, else False."""
    inst = [r.get("commit") or {} for r in rows if r["arm"] != "control"]
    if inst and all(bool(c.get("commit_pass")) for c in inst):
        return True
    if inst and all(bool(c.get("commit_pass")) or awaits_replay(c) for c in inst):
        return None
    return False


def seal_all(out: Path, deferred: dict[int, dict], rows: list[dict], verdict: dict, pending: list[int], *, tl: Any) -> None:
    """Seals the bundle of every deferred run that awaits its replay (the others failed a check of their own: nothing to replay) and
    names them in the Commit's verdict (`replay_deferred`)."""
    if not deferred:
        return
    bundles = []
    for p, ctx in deferred.items():
        row = next((rw for rw in rows if rw["pair"] == p and rw["arm"] != "control"), None)
        if row is None or not awaits_replay(row.get("commit") or {}):
            discard(out, p)
            continue
        hashed = ctx.pop(HASHED, None)
        with tl.span("closure.replay_bundle", pair=p):
            m = seal(out, p, ctx | {"row": row, "verdict": verdict}, Path(out) / f"binding_map_p{p}.json", hashed=hashed)
        bundles.append({"pair": p, "path": m["path"], "manifest_sha256": m["sha256"], "bytes": m["bytes"], "files": len(m["files"])})
        print(f"[delta] REPLAY BUNDLE (pair {p}): {m['path']} sealed, {len(m['files'])} files, {m['bytes'] >> 20} MiB, manifest sha256 {m['sha256'][:16]}", flush=True)
    verdict["replay_deferred"] = {"pending_pairs": pending, "bundles": bundles}


def _registered(model: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, t in list(model.named_parameters()) + list(model.named_buffers()):
        out.setdefault(name, t)
    return out


def weights_extra(com: Any, rc: Any, extra_names: set[str] | None = None) -> tuple[dict[str, Any], list[tuple[int, str, Any]]]:
    """The committed weights tree of the run and the bytes of the registered tensors the bundle carries, `(index in the tree's names,
    name, bytes)`, read from where they live (the device).  `extra_names`: those tensors (default: the registered buffers and engine
    constants, which no checkpoint holds)."""
    import torch

    from verity_vllm.check.weights_of_record import ENGINE_CONSTANT_RE

    tree = getattr(com, "weights_tree", None)
    if not tree or (rc.weight_registration or {}).get("root") != tree.get("root"):
        raise ValueError("replay bundle: the committer kept no weights tree for the run's committed weights root (register_weights)")
    buffers = {n for n, b in com.model.named_buffers() if isinstance(b, torch.Tensor) and b.numel()}
    want = (buffers | {n for n, *_ in tree["names"] if ENGINE_CONSTANT_RE.search(n)}) if extra_names is None else set(extra_names)
    reg = _registered(com.model)
    extra = []
    for i, (name, _dt, _shape, nb) in enumerate(tree["names"]):
        if name not in want:
            continue
        t = reg[name].detach().contiguous().reshape(-1)
        raw = SD._as_bytes(t.view(torch.uint8) if t.dtype != torch.uint8 else t)
        if len(raw) != int(nb):
            raise ValueError(f"replay bundle: {name} holds {len(raw)} B, registered {nb} B")
        extra.append((i, name, raw))
    return tree, extra


def write_store(out: str | os.PathLike, pair: int, com: Any, rc: Any, *, extra_names: set[str] | None = None,
                digests: dict[str, str] | None = None, touched: SD.Touched | None = None,
                extras: tuple[dict[str, Any], list[tuple[int, str, Any]]] | None = None, rank: int | None = None) -> Path:
    """The bundle's store and weights, written into its partial dir (the rest follows at `seal`).  `extras`: `weights_extra`'s
    answer, read before (default: read here, with `extra_names`).  ``digests`` receives the sha256 of the store's step files as they
    were written (by their path in the bundle)."""
    p = partial(out, pair, rank=rank)
    if p.exists():
        shutil.rmtree(p)
    dumped: dict[str, str] = {}
    SD.dump(com, rc, p / "store", digests=dumped, touched=touched)
    if digests is not None:
        digests.update({f"store/{k}": v for k, v in dumped.items()})
    tree, extra = extras if extras is not None else weights_extra(com, rc, extra_names)
    (p / "weights" / "extra").mkdir(parents=True, exist_ok=True)
    for i, _name, raw in extra:
        (p / "weights" / "extra" / f"{i}.bin").write_bytes(raw)
    doc = tree | {"extra": {name: i for i, name, _raw in extra}}
    (p / "weights" / "tree.json").write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    return p


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def seal(out: str | os.PathLike, pair: int, context: dict[str, Any], binding_map: str | os.PathLike | None, *,
         hashed: dict[str, str] | None = None, rank: int | None = None) -> dict[str, Any]:
    """Writes the context and the binding map into the partial bundle, then its manifest, and renames it into place.  ``hashed``: the
    sha256 of files taken as they were written (`write_store`); every other file is read and hashed here.  ``binding_map`` None: the
    partial holds it already (a TP rank writes its own)."""
    p, final = partial(out, pair, rank=rank), path(out, pair, rank=rank)
    if binding_map is not None:
        shutil.copyfile(binding_map, p / "binding_map.json")
    (p / "context.json").write_text(json.dumps({"schema": SCHEMA, **context}, default=str, separators=(",", ":")) + "\n")
    rels = sorted(x.relative_to(p).as_posix() for x in p.rglob("*") if x.is_file())
    with ThreadPoolExecutor(8) as pool:
        sums = dict(zip(rels, pool.map(lambda r: (hashed or {}).get(r) or _sha256(p / r), rels)))
    files = {r: {"sha256": sums[r], "bytes": (p / r).stat().st_size} for r in rels}
    man = {"schema": SCHEMA, "files": files, "bytes": sum(v["bytes"] for v in files.values()),
           "sealed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (p / MANIFEST).write_text(json.dumps(man, indent=1) + "\n")
    if final.exists():
        shutil.rmtree(final)
    os.replace(p, final)
    return man | {"path": str(final), "sha256": _sha256(final / MANIFEST)}


def discard(out: str | os.PathLike, pair: int, *, rank: int | None = None) -> None:
    shutil.rmtree(partial(out, pair, rank=rank), ignore_errors=True)


def open_bundle(bundle: str | os.PathLike) -> dict[str, Any]:
    """The bundle's manifest and context, every file checked against the manifest (a missing, extra or changed file is refused by name)."""
    b = Path(bundle)
    man = json.loads((b / MANIFEST).read_text())
    if man.get("schema") != SCHEMA:
        raise ValueError(f"{b}: manifest schema {man.get('schema')!r} is not {SCHEMA}")
    have = {f.relative_to(b).as_posix() for f in b.rglob("*") if f.is_file()} - {MANIFEST}
    want = set(man["files"])
    if have != want:
        raise ValueError(f"{b}: files differ from the manifest: missing {sorted(want - have)[:4]}, extra {sorted(have - want)[:4]}")
    rels = sorted(man["files"])
    with ThreadPoolExecutor(8) as pool:
        bad = [rel for rel, got in zip(rels, pool.map(lambda r: _sha256(b / r), rels)) if got != man["files"][rel]["sha256"]]
    if bad:
        raise ValueError(f"{b}: files whose sha256 is not the manifest's: {bad[:4]}")
    ctx = json.loads((b / "context.json").read_text())
    return {"path": str(b), "manifest": man, "manifest_sha256": _sha256(b / MANIFEST), "context": ctx}
