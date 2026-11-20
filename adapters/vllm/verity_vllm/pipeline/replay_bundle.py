"""The replay bundle a sealed window leaves, one per instrumented run, `<out>/replay_bundle_p<pair>/`:

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
    private.json      where the bundle's private part is (`PRIVATE_DOC`): its root and place, and the files there
    MANIFEST.json     {schema, files: {path: {sha256, bytes}}}, written last, the private part's files by their path in the bundle
                      too; with the values committed and not in the store (`NOT_RETAINED`), those too

The private part holds what opens every committed leaf and is never in the bundle or a run's outputs: the store's salt keys
(``store/keys.bin``) and the weights registration's salt file (``weights/tree.salts/``, `verity.commitments.salt_file`).  It goes
under the private directory (`private_root`), at ``replay/<run root>/<digest of the bundle's path>``.  A bundle sealed with them
inside (no ``private.json``) is read as it was stored.

It is built under `replay_bundle_p<pair>.partial/` and renamed into place once sealed, so a bundle directory that exists is complete.
`open_bundle` checks every file against the manifest before anything is read from it.

"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from verity.commitments import salt_file as SF

from verity_vllm.commit import step_rows
from verity_vllm.commit import store_dump as SD
from verity_vllm.log import init_logger
from verity_vllm.records.schema_ids import (
    BOUNDARY_LINKAGE_SCHEMA,
    REPLAY_BUNDLE_SCHEMA,
    SAMPLED_REPLAY_SCHEMA,
)

# The replay driver, the verdict's two checks, the weights of record's pattern and the planner are imported inside the functions
# that read them, and the two deferred records take their schema ids from `records.schema_ids`: the driver's import puts the
# catalog's Definitions into the Definition table, and the Commit that defers its replay to a bundle never loads the driver.

__all__ = ["ARGS", "HASHED", "MANIFEST", "NOT_RETAINED", "PENDING", "PRIVATE", "PRIVATE_DOC", "deferred_records", "discard",
           "load_store", "open_bundle", "partial", "path", "private_file", "private_refusal", "private_root", "seal", "seal_all",
           "weights_extra", "write_store", "write_store_files"]

logger = init_logger(__name__)

SCHEMA = REPLAY_BUNDLE_SCHEMA
MANIFEST = "MANIFEST.json"
#: the context's (unwritten) key for the sha256 of the files `write_store` hashed as it wrote them
HASHED = "_hashed"
#: the context's and manifest's key for the values committed and not retained, so not in the store (`store_dump.not_retained`)
NOT_RETAINED = "committed_not_retained"
#: the bundle's files that go to its private part, by their path in the bundle
PRIVATE = ("store/keys.bin", "weights/tree.salts")
PRIVATE_DOC = "private.json"
ARGS = ("program_dir", "programs_root", "program_digest", "required_manifest", "replay_per_stratum", "replay_max", "replay_target", "replay_k",
        "replay_draw", "replay_workers", "replay_seed", "replay_challenge", "window_mb", "window_slots", "retain", "label", "serving_rows", "openings",
        "checkpoints", "workload", "weights_of_record", "checkpoint_dir")


def path(out: str | os.PathLike, pair: int, *, rank: int | None = None) -> Path:
    """The bundle of instrumented run ``pair``."""
    return Path(out) / (f"replay_bundle_p{pair}" + (f"_rank{rank}" if rank is not None else ""))


def partial(out: str | os.PathLike, pair: int, *, rank: int | None = None) -> Path:
    p = path(out, pair, rank=rank)
    return p.with_name(p.name + ".partial")


def private_refusal(environ: Mapping[str, str] | None = None) -> str | None:
    """Why no bundle can be written here: VERITY_PROTOCOLS_CONFIG (the schemes') names no ``private`` directory, or one inside
    the run's directory (`serve_guard.inside_run`); None when `private_root` is its private."""
    from verity_vllm import settings
    from verity_vllm.pipeline.serve_guard import inside_run

    d = settings.get("VERITY_PROTOCOLS_CONFIG", environ).get("private")
    if not d:
        return ("a replay bundle's salt keys and salts go to VERITY_PROTOCOLS_CONFIG's private, which names none: set it, outside "
                "the run's directory")
    return inside_run("VERITY_PROTOCOLS_CONFIG's private", d, environ)


def private_root() -> Path:
    """The private directory a bundle's private part goes under: VERITY_PROTOCOLS_CONFIG's ``private``; refused as
    `private_refusal` says."""
    from verity_vllm import settings

    if why := private_refusal():
        raise ValueError(f"replay bundle: {why}")
    return Path(settings.get("VERITY_PROTOCOLS_CONFIG")["private"])


def load_store(bundle: str | os.PathLike) -> tuple[Any, Any]:
    """The bundle's store (`store_dump.load`), its salt keys from its private part or, sealed with them inside, from the store."""
    k, inside = private_file(bundle, PRIVATE[0]), Path(bundle) / PRIVATE[0]
    return SD.load(Path(bundle) / "store", keys=None if k is None or k == inside else k)


def private_file(bundle: str | os.PathLike, rel: str) -> Path | None:
    """Where the bundle's private file ``rel`` (one of `PRIVATE`) is: in its private part (`PRIVATE_DOC`), or in the bundle for one
    sealed with it inside; None when the bundle has none.  Refused by name when both hold it, or the private part lacks it."""
    b = Path(bundle)
    inside, there = b / rel, None
    if (b / PRIVATE_DOC).is_file():
        doc = json.loads((b / PRIVATE_DOC).read_text())
        there = Path(doc["root"]) / doc["place"] / rel if rel in doc["files"] else None
    if there is not None and inside.exists():
        raise ValueError(f"{b}: {rel} is both in the bundle and in its private directory {there}")
    if there is not None and not there.exists():
        raise ValueError(f"{b}: {rel} is not in its private directory {there}")
    return there if there is not None else inside if inside.exists() else None


def deferred_records(bundle: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """The Commit's sampled-replay and boundary-linkage records while their replay is deferred: not evaluated, named, with the bundle."""
    why = f"deferred: the C2 replay runs on CPU from the replay bundle {bundle} (`verity-vllm row stage replay`)"
    return ({"schema": SAMPLED_REPLAY_SCHEMA, "method": "sampled-exact-replay", "result": None, "grade": "deferred", "why": why,
             "deferred": {"bundle": str(bundle)}},
            {"schema": BOUNDARY_LINKAGE_SCHEMA, "result": None, "why": why, "pairs": 0, "linked": 0, "deferred": {"bundle": str(bundle)}})


PENDING = "PENDING: the C2 sampled replay + boundary linkage are deferred to the replay bundle"


def awaits_replay(c: dict) -> bool:
    """Whether a run's Commit record has no pass yet and its sampled replay is the deferred one (`deferred_records`)."""
    sr = c.get("sampled_replay")
    return c.get("commit_pass") is None and (sr.get("grade") if sr else None) == "deferred"


def commit_of(row: dict) -> dict:
    """A run's Commit record; an empty one for a row that holds none, which neither passed nor awaits a replay."""
    commit = row.get("commit")
    return commit if commit else {}


def seal_all(out: Path, deferred: dict[int, dict], rows: list[dict], verdict: dict, pending: list[int], *, tl: Any) -> None:
    """Seals the bundle of every deferred run that awaits its replay (the others failed a check of their own: nothing to replay) and
    names them in the Commit's verdict (`replay_deferred`)."""
    if not deferred:
        return
    bundles = []
    for p, ctx in deferred.items():
        row = next((rw for rw in rows if rw["pair"] == p and rw["arm"] != "control"), None)
        if row is None or not awaits_replay(commit_of(row)):
            discard(out, p)
            continue
        hashed = ctx.pop(HASHED, None)
        with tl.span("closure.replay_bundle", pair=p):
            m = seal(out, p, ctx | {"row": row, "verdict": verdict}, Path(out) / f"binding_map_p{p}.json", hashed=hashed)
        bundles.append({"pair": p, "path": m["path"], "manifest_sha256": m["sha256"], "bytes": m["bytes"], "files": len(m["files"])})
        logger.info("[delta] REPLAY BUNDLE (pair %s): %s sealed, %s files, %s MiB, manifest sha256 %s", p, m["path"], len(m["files"]),
                    m["bytes"] >> 20, m["sha256"][:16])
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
        raw = SD.as_bytes(t.view(torch.uint8) if t.dtype != torch.uint8 else t)
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
    p, doc = write_store_files(out, pair, com, rc, extra_names=extra_names, digests=digests, touched=touched, extras=extras, rank=rank)
    SD.write_doc(p / "store", doc, com._layouts)
    return p


def write_store_files(out: str | os.PathLike, pair: int, com: Any, rc: Any, *, extra_names: set[str] | None = None,
                      digests: dict[str, str] | None = None, touched: SD.Touched | None = None,
                      extras: tuple[dict[str, Any], list[tuple[int, str, Any]]] | None = None,
                      rank: int | None = None) -> tuple[Path, dict[str, Any]]:
    """`write_store` but for the store's ``store.json``: the partial dir and the document `store_dump.write_doc` writes there.  The
    keys and salts go to the bundle's private part under `private_root`."""
    p = partial(out, pair, rank=rank)
    at = hashlib.sha256(str(path(out, pair, rank=rank).resolve()).encode()).hexdigest()[:16]
    root, place = private_root(), f"replay/{rc.run_root.hex()}/{at}"
    if p.exists():
        shutil.rmtree(p)
    shutil.rmtree(root / place, ignore_errors=True)
    dumped: dict[str, str] = {}
    doc = SD.dump_files(com, rc, p / "store", digests=dumped, touched=touched, keys_to=root / place / PRIVATE[0])
    if digests is not None:
        digests.update({f"store/{k}": v for k, v in dumped.items()})
    tree, extra = extras if extras is not None else weights_extra(com, rc, extra_names)
    (p / "weights" / "extra").mkdir(parents=True, exist_ok=True)
    salts = {n: ys for n, ys in (getattr(com, "weights_salts", None) or {}).items() if ys}
    if tree.get("scheme") == step_rows.NAME and any(int(nb) for *_e, nb in tree["names"]) and not salts:
        raise ValueError("replay bundle: the committer holds no salts for its rows-sha512 weights registration (register_weights)")
    if salts:
        SF.write(root / place / PRIVATE[1], salts)
    held = [rel for rel in PRIVATE if (root / place / rel).exists()]
    (p / PRIVATE_DOC).write_text(json.dumps({"root": str(root), "place": place, "files": held}) + "\n")
    for i, _name, raw in extra:
        (p / "weights" / "extra" / f"{i}.bin").write_bytes(raw)
    wdoc = tree | {"extra": {name: i for i, name, _raw in extra}}
    (p / "weights" / "tree.json").write_text(json.dumps(wdoc, separators=(",", ":")) + "\n")
    return p, doc


def _files(b: Path) -> dict[str, Path]:
    """The bundle's files by their path in it, its private part's included; not its manifest."""
    files = {f.relative_to(b).as_posix(): f for f in b.rglob("*") if f.is_file()}
    files.pop(MANIFEST, None)
    for rel in PRIVATE:
        if (there := private_file(b, rel)) is not None and there.is_dir():
            files |= {f"{rel}/{f.relative_to(there).as_posix()}": f for f in there.rglob("*") if f.is_file()}
        elif there is not None:
            files[rel] = there
    return files


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
    partial holds it already (a TP rank writes its own).  The context's `NOT_RETAINED` record is the manifest's too."""
    p, final = partial(out, pair, rank=rank), path(out, pair, rank=rank)
    if binding_map is not None:
        shutil.copyfile(binding_map, p / "binding_map.json")
    (p / "context.json").write_text(json.dumps({"schema": SCHEMA, **context}, default=str, separators=(",", ":")) + "\n")
    at = _files(p)
    rels = sorted(at)
    with ThreadPoolExecutor(8) as pool:
        sums = dict(zip(rels, pool.map(lambda r: (hashed or {}).get(r) or _sha256(at[r]), rels)))
    files: dict[str, dict[str, Any]] = {r: {"sha256": sums[r], "bytes": at[r].stat().st_size} for r in rels}
    man = {"schema": SCHEMA, "files": files, "bytes": sum(v["bytes"] for v in files.values()),
           "sealed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           **({NOT_RETAINED: context[NOT_RETAINED]} if context.get(NOT_RETAINED) else {})}
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
    at = _files(b)
    have, want = set(at), set(man["files"])
    if have != want:
        raise ValueError(f"{b}: files differ from the manifest: missing {sorted(want - have)[:4]}, extra {sorted(have - want)[:4]}")
    rels = sorted(man["files"])
    with ThreadPoolExecutor(8) as pool:
        bad = [rel for rel, got in zip(rels, pool.map(lambda r: _sha256(at[r]), rels)) if got != man["files"][rel]["sha256"]]
    if bad:
        raise ValueError(f"{b}: files whose sha256 is not the manifest's: {bad[:4]}")
    ctx = json.loads((b / "context.json").read_text())
    return {"path": str(b), "manifest": man, "manifest_sha256": _sha256(b / MANIFEST), "context": ctx}
