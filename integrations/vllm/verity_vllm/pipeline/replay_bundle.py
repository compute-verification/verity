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
                      (weights pin, coverage, openings) and verdict inputs (declaration, required classes)
    MANIFEST.json     {schema, files: {path: {sha256, bytes}}}, written last

It is built under `replay_bundle_p<pair>.partial/` and renamed into place once sealed, so a bundle directory that exists is complete.
`open_bundle` checks every file against the manifest before anything is read from it.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from verity_vllm.commit import store_dump as SD

SCHEMA = "verity-vllm/replay-bundle/v1"
MANIFEST = "MANIFEST.json"
#: the context's (unwritten) key for the sha256 of the files `write_store` hashed as it wrote them
HASHED = "_hashed"
ARGS = ("program_dir", "programs_root", "program_digest", "required_manifest", "replay_per_stratum", "replay_max", "replay_target", "replay_k",
        "replay_workers", "replay_seed", "replay_challenge", "window_mb", "window_slots", "retain", "label", "serving_rows", "openings",
        "checkpoints", "workload", "weights_of_record", "checkpoint_dir")


def path(out: str | os.PathLike, pair: int) -> Path:
    return Path(out) / f"replay_bundle_p{pair}"


def partial(out: str | os.PathLike, pair: int) -> Path:
    return Path(out) / f"replay_bundle_p{pair}.partial"


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
           "--vu-export-dir exports from the replay, which does not run here" if args.vu_export_dir else None)
    if why:
        raise SystemExit(f"--replay-deferred: {why}")


def defer(out: Path, pair: int, com: Any, rc: Any, args: Any, *, tl: Any, arm: str, versions: dict, reqs: list[dict], sampling: dict,
          tokens: dict, stop_of_record: Any, ids_full: Any, declaration: Any, required_classes: Any) -> tuple[dict, dict, dict]:
    """At the C2 site of a deferred run: writes the bundle's store and weights tree, and returns the run's deferred sampled-replay and
    linkage records and the bundle's context (sealed once the run's record is final, after the weights pin and the openings)."""
    tl.start("validate.replay_bundle", pair=pair)
    hashed: dict[str, str] = {}
    write_store(out, pair, com, rc, digests=hashed)
    caps = {str(q.get("request_id")): int(q.get("max_tokens") or sampling.get("max_tokens") or 1) for q in reqs}
    blocks = getattr(com, "_gpu_blocks", None) or {}
    ctx = {"pair": pair, "arm": arm, "committer": com.name, "run_root": rc.run_root.hex(), "code": versions.get("git_head"),
           "args": {k: getattr(args, k, None) for k in ARGS}, "requests": reqs, "sampling": sampling, "tokens": tokens,
           "stop_of_record": stop_of_record,
           # the reconciliation of the not-executed class reads the placed identities only when a request stopped early
           "ids_full": ids_full if any(len(v) < caps.get(str(k), 0) for k, v in tokens.items()) else None,
           "retention": ("device" if any(b is not None and bool(getattr(b, "is_cuda", False)) for b, _u in blocks.values())
                         else (str(args.retain) if args.retain else ("host" if blocks else None))),
           "required_manifest_sha256": hashlib.sha256(Path(args.required_manifest).read_bytes()).hexdigest() if args.required_manifest else None,
           "declaration": declaration, "required_classes": list(required_classes), "min_openings": min(64, args.openings),
           HASHED: hashed}
    sr, lk = deferred_records(path(out, pair))
    tl.end("validate.replay_bundle", pair=pair)
    print(f"[delta] C2 SAMPLED REPLAY + LINKAGE (pair {pair}): DEFERRED -- store and weights tree written to {partial(out, pair)}", flush=True)
    return sr, lk, ctx


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
    return fail if fail is not None else (None, "PENDING: the C2 sampled replay + boundary linkage are deferred to the replay bundle")


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


def write_store(out: str | os.PathLike, pair: int, com: Any, rc: Any, *, extra_names: set[str] | None = None,
                digests: dict[str, str] | None = None) -> Path:
    """The bundle's store and weights, written into its partial dir (the rest follows at `seal`).  `extra_names`: the registered
    tensors whose bytes go into the bundle (default: the registered buffers and engine constants, which no checkpoint holds).
    ``digests`` receives the sha256 of the store's step files as they were written (by their path in the bundle)."""
    import torch

    from verity_vllm.check.weights_of_record import ENGINE_CONSTANT_RE

    p = partial(out, pair)
    if p.exists():
        shutil.rmtree(p)
    dumped: dict[str, str] = {}
    SD.dump(com, rc, p / "store", digests=dumped)
    if digests is not None:
        digests.update({f"store/{k}": v for k, v in dumped.items()})
    tree = getattr(com, "weights_tree", None)
    if not tree or (rc.weight_registration or {}).get("root") != tree.get("root"):
        raise ValueError("replay bundle: the committer kept no weights tree for the run's committed weights root (register_weights)")
    buffers = {n for n, b in com.model.named_buffers() if isinstance(b, torch.Tensor) and b.numel()}
    want = (buffers | {n for n, *_ in tree["names"] if ENGINE_CONSTANT_RE.search(n)}) if extra_names is None else set(extra_names)
    reg = _registered(com.model)
    extra = {}
    (p / "weights" / "extra").mkdir(parents=True, exist_ok=True)
    for i, (name, _dt, _shape, nb) in enumerate(tree["names"]):
        if name not in want:
            continue
        t = reg[name].detach().contiguous().reshape(-1)
        raw = SD._as_bytes(t.view(torch.uint8) if t.dtype != torch.uint8 else t)
        if len(raw) != int(nb):
            raise ValueError(f"replay bundle: {name} holds {len(raw)} B, registered {nb} B")
        (p / "weights" / "extra" / f"{i}.bin").write_bytes(raw)
        extra[name] = i
    doc = tree | {"extra": extra}
    (p / "weights" / "tree.json").write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    return p


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def seal(out: str | os.PathLike, pair: int, context: dict[str, Any], binding_map: str | os.PathLike, *,
         hashed: dict[str, str] | None = None) -> dict[str, Any]:
    """Writes the context and the binding map into the partial bundle, then its manifest, and renames it into place.  ``hashed``: the
    sha256 of files taken as they were written (`write_store`); every other file is read and hashed here."""
    p, final = partial(out, pair), path(out, pair)
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


def discard(out: str | os.PathLike, pair: int) -> None:
    shutil.rmtree(partial(out, pair), ignore_errors=True)


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
