"""The serving rows of a deferred Commit (``verity-vllm serving-rows BUNDLE --row-dir D --partition F --out O``): the sealed replay
bundle opened against the Commit's record as the CPU replay opens it (manifest sha256, run root, binding map, the Build's Programs of
record), the weights opened against the committed weights root, and the window `pipeline.serving_rows.after_commit` writes, the same
window the in-Commit hook writes, into ``O``.  A slim bundle holds the replay's picks only, not the population: refused.

The window is a ``stand-in`` (core's ``registration.KINDS``), not ``served``: serving's commitment is the record's roots, under the
scheme the record was committed in (``commitment``: the store's, whose run root and folded step roots `cpu_replay.of_record` holds
to the run root of record; ``rows-sha512`` by default, ``vllm-v1`` an older record's), and these rows are committed afterwards from
the bundle in trees of their own (`ml.windows.rows`: a ``frame-v3-sha512`` tree per port, fresh salts), so their roots are a fresh
commitment whose link to serving's is provenance (`stand_in`: ``of`` the record's commitment and run root, the bundle's manifest
digest and this command, and ``link``).  A SHA-512 record does not make it ``served``: these roots are not the record's, whatever its
hash.  A window is ``served`` when its roots are serving's own, made before sampling (the in-Commit hook's, `serving_rows.readers`,
which a deferred Commit with ``--serving-rows`` runs at its C2 site before its bundle is sealed, `c2_replay.serving_rows`); a
window from the bundle becomes ``served`` only once a circuit opens the record's own leaves (rows-sha512's ``hm96-sha512/row/v2``
stream rows, vllm-v1's SHA-256 leaves)."""
from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from verity_vllm.check.replay.index import ProgramIndex
from verity_vllm.config import option
from verity_vllm.pipeline import c2_replay as C2
from verity_vllm.pipeline import cpu_replay as CPU
from verity_vllm.pipeline import replay_bundle as RB
from verity_vllm.pipeline import serving_rows as SRW

ROWS = "verity-vllm serving-rows: replay bundle -> rows"


def link(commitment: str) -> str:
    """A stand-in's ``link`` to the record's ``commitment`` roots."""
    return (f"provenance, not proof: verity-vllm serving-rows opened the sealed replay bundle against the record's {commitment} roots "
            f"and committed these rows from it; no circuit opens the {commitment} leaves")


def stand_in(run_root: str, bundle_manifest_sha256: str, commitment: str) -> dict[str, Any]:
    """The window's link to serving's commitment: ``of`` the record's ``commitment`` (`of_record`'s) and run root, the bundle's
    manifest digest and the command that turned the bundle into rows, and ``link``."""
    return {"of": {"commitment": commitment, "run_root": run_root, "bundle_manifest_sha256": bundle_manifest_sha256, "rows": ROWS},
            "link": link(commitment)}


def of_record(bundle: str | os.PathLike, row_dir: str, required_manifest: str | None = None) -> SimpleNamespace:
    """A deferred Commit's sealed replay bundle opened against the Commit's record beside it as the CPU replay opens it
    (`cpu_replay.of_record`, decided or not), with the Build's Programs of record under ``row_dir`` (`c2_replay.programs_of_record`)
    and the replay's arrivals, and ``commitment``, the scheme the record was committed in (the store's: its run root and its step
    roots folded are the run root of record, 32 bytes under vllm-v1 and 64 under rows-sha512).  A slim bundle holds the replay's
    picks only: refused.  The Definitions the Build bound are not held to this tree's: no row is evaluated here, and the one-stage
    proof is over the verifier's own template.  A served window's bundle (`serve_seal.seal_window`) names no Programs: its
    Programs are the registered compiler's for the window's advice, so it opens with none (``programs`` empty, each request's
    arrival its first step in the window), and ``manifest`` is the alias table it sealed (the bench row's ``op_path_aliases``,
    which `opening.CommittedStore` resolves a shared module's layer-N path through), or None when it sealed none."""
    b = RB.open_bundle(bundle)
    ctx, bdir = b["context"], Path(b["path"])
    if ctx.get("slim"):
        raise CPU.Refused(f"{bdir} is a slim bundle: it holds the replay's picks, not every unit of the population")
    _rows, row, _verdict, store, run = CPU.of_record(b, bdir.parent, awaiting=False)
    if (sw := ctx.get("served_window")):
        aliases = sw.get("op_path_aliases")
        return SimpleNamespace(bundle=b, ctx=ctx, bdir=bdir, out=bdir.parent, pair=int(ctx["pair"]), row=row, store=store, run=run,
                               commitment=store.commit_scheme, manifest=None if aliases is None else {"op_path_aliases": aliases},
                               loaded={}, digests={}, programs={}, program_dirs={},
                               arrivals={str(q["request_id"]): int(q["arrive_step"]) for q in ctx["requests"]})
    o = SimpleNamespace(row_dir=row_dir, replay_workers=None, required_manifest=required_manifest)
    args = CPU.replay_args(o, ctx)
    _path, man = CPU.manifest_of(o, ctx, args)
    site = SimpleNamespace(args=args, manifest=man, reqs=ctx["requests"], sampling=ctx["sampling"], row=row)
    rl = C2.request_labels(ctx["requests"])
    loaded, digs = C2.programs_of_record(site, rl, definitions=False)
    arrivals, _moved = C2.arrivals(site, rl)
    return SimpleNamespace(bundle=b, ctx=ctx, bdir=bdir, out=bdir.parent, pair=int(ctx["pair"]), row=row, store=store, run=run,
                           commitment=store.commit_scheme, manifest=man, loaded=loaded, digests=digs,
                           programs={k: v["instances"] for k, v in loaded.items()},
                           program_dirs={k: v.get("dir") for k, v in loaded.items()}, arrivals=arrivals)


def window_of(r: SimpleNamespace, partition_file: str) -> Any:
    """The verifier's own window of a deferred Commit opened by `of_record` (`serving_rows.population`: its population program,
    partition and law over the Build's Programs of record, the run root of record), from the record alone: no committed value is read."""
    window, _rows = SRW.population(r.programs, json.loads(Path(partition_file).read_text()), bytes(r.run.run_root).hex(), r.program_dirs,
                                   arrivals=r.arrivals)
    return window


def window_of_record(bundle: str | os.PathLike, row_dir: str, partition_file: str, required_manifest: str | None = None) -> Any:
    """`window_of` the bundle opened against its record (`of_record`)."""
    return window_of(of_record(bundle, row_dir, required_manifest), partition_file)


#: the bytes of the only run root a ``served`` window is taken over: the SHA-512 record's (``frame-v3-sha512`` / ``hm96-sha512``)
SERVED_ROOT = 64


def source(window: dict[str, Any], run_root: str, bundle_manifest_sha256: str, commitment: str,
           weights_root: str | None = None) -> dict[str, bool]:
    """A registration's ``window`` held to the verifier's record, its run root ``run_root`` (hex) and ``commitment`` (`of_record`'s),
    each check by name; the one-stage audit's (A5's) source check.  Either kind names the record's commitment as the scheme its values
    were read under.  A ``stand-in``'s run, ``of`` and ``link`` are `stand_in`'s over the run root, the digest of the bundle's
    manifest and the commitment.  A ``served`` window (the in-Commit hook's, `serving_rows.readers`: serving's own commitment, made
    before sampling) has the run root as its run, which is a SHA-512 root (``sha512``: a ``vllm-v1`` root's is refused), and carries
    no stand-in link; given the record's committed ``weights_root`` (hex), the weights its rows read were registered under SHA-512
    too (``weights_sha512``: a vllm-v1 registration's 32-byte root is refused)."""
    want = stand_in(run_root, bundle_manifest_sha256, commitment)
    named = {"run": window.get("run") == run_root, "commitment": window.get("commitment") == commitment}
    if window.get("kind") == "served":
        return {"kind": True, **named, "sha512": len(run_root) == 2 * SERVED_ROOT,
                **({"weights_sha512": len(weights_root) == 2 * SERVED_ROOT} if weights_root is not None else {}),
                "link": "of" not in window and "link" not in window}
    return {"kind": window.get("kind") == "stand-in", **named, "of": window.get("of") == want["of"], "link": window.get("link") == want["link"]}


def opened_weights(r: SimpleNamespace, checkpoint: Any = None, threads: int = 8) -> Callable:
    """The replay's weights provider over the bundle's tensors (and ``checkpoint``'s, as vLLM rewrites them at load by the Build's
    `weight_roles`), opened against the committed weights root."""
    k0 = next(iter(r.programs))
    prefix = ProgramIndex(r.programs[k0], program_dir=r.program_dirs[k0]).prefix
    tree = json.loads((r.bdir / "weights" / "tree.json").read_text())
    root = (r.row["commit"].get("weight_registration") or {}).get("root")
    words, wc = CPU.open_weights(r.bdir, tree, root, checkpoint, threads=threads, roles=CPU.registered_roles(r.program_dirs.values(), prefix))
    if not wc["equal"]:
        raise CPU.Refused(f"the bundle's weights: {wc['why']}")
    return CPU.provider(words, prefix, set())


def from_bundle(bundle: str | os.PathLike, row_dir: str, partition_file: str, out_dir: str | os.PathLike, *, checkpoint: Any = None,
                weights: Callable | None = None, required_manifest: str | None = None, threads: int = 8,
                workers: int | None = None) -> dict[str, Any]:
    """The serving rows of a deferred Commit, from its sealed replay bundle opened against its record (`of_record`), and the weights
    opened against the committed weights root from the bundle's tensors and ``checkpoint`` (a `weights_of_record.CheckpointIndex`;
    without one, only the tensors the bundle carries).  ``weights`` (tests) replaces the opened weights."""
    from verity_vllm.check.oracle_compare import OpenedReader

    r = of_record(bundle, row_dir, required_manifest)
    if weights is None:
        weights = opened_weights(r, checkpoint, threads)
    bmap = json.loads((r.out / f"binding_map_p{r.pair}.json").read_text())
    committer = {"from": "replay bundle", "pair": r.pair, "source": r.ctx.get("code")}
    return SRW.after_commit(str(out_dir), partition_file=partition_file, programs=r.programs, bmap=bmap,
                            committed=OpenedReader(r.store, r.run), weights=weights,
                            manifest=r.manifest if isinstance(r.manifest, dict) else None, run_root=r.run.run_root,
                            row=(r.ctx.get("args") or {}).get("label"), arrivals=r.arrivals, program_dirs=r.program_dirs,
                            digests=r.digests, committer=committer, workers=workers,
                            stand_in=stand_in(bytes(r.run.run_root).hex(), r.bundle["manifest_sha256"], r.commitment))


@dataclasses.dataclass(frozen=True, kw_only=True)
class Options:
    """verity-vllm serving-rows BUNDLE --row-dir D --partition F --out O [--checkpoint-dir CK ...]: a deferred Commit's serving rows
    and one-stage registration, from its sealed replay bundle (`from_bundle`)."""
    bundle: str = option("bundle", help="the sealed replay bundle (<commit out>/replay_bundle_p<pair>)")
    row_dir: str = option("--row-dir", help="the row dir holding the Build's Programs of record (and manifest.json)")
    partition: str = option("--partition", help="the verifier's partition file: {templates, law, scope?, partition_digest?}")
    out: str = option("--out", help="the window's directory: registration.json, partition.json, index.json, window.json, M0's bodies")
    checkpoint_dir: str | None = option("--checkpoint-dir", default=None,
                                        help="the checkpoint of record (default: none, so only the tensors the bundle carries open)")
    checkpoints_manifest: str | None = option("--checkpoints-manifest", default=None, help="manifests/checkpoints.json (shard sha256 pins)")
    model: str | None = option("--model", default=None, help="the checkpoint's repo (its checkpoints.json entry)")
    revision: str | None = option("--revision", default=None, help="the checkpoint's revision")
    required_manifest: str | None = option("--required-manifest", default=None,
                                           help="the Commit's required manifest (default: its recorded path, else <row-dir>/manifest.json)")
    threads: int = option("--threads", type=int, default=8, help="threads opening the weights")
    workers: int | None = option("--workers", type=int, default=None, help="processes hashing the rows (default: min(16, CPUs))")


def main(o: Options) -> int:
    ck = None
    if o.checkpoint_dir:
        from verity_vllm.check import weights_of_record as WR
        ck = WR.CheckpointIndex(o.checkpoint_dir, checkpoints_manifest=o.checkpoints_manifest, model=o.model, revision=o.revision)
    try:
        from_bundle(o.bundle, o.row_dir, o.partition, o.out, checkpoint=ck, required_manifest=o.required_manifest, threads=o.threads,
                    workers=o.workers)
    except CPU.Refused as e:
        print(f"[serving-rows] REFUSED: {e}", flush=True)
        return 2
    return 0
