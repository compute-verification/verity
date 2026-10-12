"""The serving rows of a deferred Commit (``verity-vllm serving-rows BUNDLE --row-dir D --partition F --out O``): the sealed replay
bundle opened against the Commit's record as the CPU replay opens it (manifest sha256, run root, binding map, the Build's Programs of
record), the weights opened against the committed weights root, and the window `pipeline.serving_rows.after_commit` writes, the same
window the in-Commit hook writes, into ``O``.  A slim bundle holds the replay's picks only, not the population: refused.

The window is a ``stand-in`` (core's ``registration.KINDS``), not ``served``: serving's commitment is the vllm-v1 roots, and these rows
are committed afterwards from the bundle, so their roots are a fresh commitment whose link to serving's is provenance (`stand_in`:
``of`` the record's vllm-v1 run root, the bundle's manifest digest and this command, and ``link``).  It becomes ``served`` once
serving's record is itself ``frame-v3-sha512`` / ``hm96-sha512``, or once the circuit opens the vllm-v1 SHA-256 leaves."""
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
LINK = ("provenance, not proof: verity-vllm serving-rows opened the sealed replay bundle against the record's vllm-v1 roots and "
        "committed these rows from it; no circuit opens the vllm-v1 leaves")


def stand_in(run_root: str, bundle_manifest_sha256: str) -> dict[str, Any]:
    """The window's link to serving's commitment: ``of`` the vllm-v1 run root, the bundle's manifest digest and the command that
    turned the bundle into rows, and ``link``."""
    return {"of": {"commitment": "vllm-v1", "run_root": run_root, "bundle_manifest_sha256": bundle_manifest_sha256, "rows": ROWS},
            "link": LINK}


def of_record(bundle: str | os.PathLike, row_dir: str, required_manifest: str | None = None) -> SimpleNamespace:
    """A deferred Commit's sealed replay bundle opened against the Commit's record beside it as the CPU replay opens it
    (`cpu_replay.of_record`, decided or not), with the Build's Programs of record under ``row_dir`` (`c2_replay.programs_of_record`)
    and the replay's arrivals.  A slim bundle holds the replay's picks only: refused."""
    b = RB.open_bundle(bundle)
    ctx, bdir = b["context"], Path(b["path"])
    if ctx.get("slim"):
        raise CPU.Refused(f"{bdir} is a slim bundle: it holds the replay's picks, not every unit of the population")
    _rows, row, _verdict, store, run = CPU.of_record(b, bdir.parent, awaiting=False)
    o = SimpleNamespace(row_dir=row_dir, replay_workers=None, required_manifest=required_manifest)
    args = CPU.replay_args(o, ctx)
    _path, man = CPU.manifest_of(o, ctx, args)
    site = SimpleNamespace(args=args, manifest=man, reqs=ctx["requests"], sampling=ctx["sampling"], row=row)
    rl = C2.request_labels(ctx["requests"])
    loaded, digs = C2.programs_of_record(site, rl)
    arrivals, _moved = C2.arrivals(site, rl)
    return SimpleNamespace(bundle=b, ctx=ctx, bdir=bdir, out=bdir.parent, pair=int(ctx["pair"]), row=row, store=store, run=run,
                           manifest=man, loaded=loaded, digests=digs, programs={k: v["instances"] for k, v in loaded.items()},
                           program_dirs={k: v.get("dir") for k, v in loaded.items()}, arrivals=arrivals)


def window_of_record(bundle: str | os.PathLike, row_dir: str, partition_file: str, required_manifest: str | None = None) -> Any:
    """The verifier's own window of a deferred Commit (`serving_rows.population`: its population program, partition and law over the
    Build's Programs of record, the run root of record), from the record alone: no committed value is read."""
    r = of_record(bundle, row_dir, required_manifest)
    window, _rows = SRW.population(r.programs, json.loads(Path(partition_file).read_text()), bytes(r.run.run_root).hex(), r.program_dirs)
    return window


def opened_weights(r: SimpleNamespace, checkpoint: Any = None, threads: int = 8) -> Callable:
    """The replay's weights provider over the bundle's tensors (and ``checkpoint``'s), opened against the committed weights root."""
    k0 = next(iter(r.programs))
    prefix = ProgramIndex(r.programs[k0], program_dir=r.program_dirs[k0]).prefix
    tree = json.loads((r.bdir / "weights" / "tree.json").read_text())
    root = (r.row["commit"].get("weight_registration") or {}).get("root")
    words, wc = CPU.open_weights(r.bdir, tree, root, checkpoint, threads=threads)
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
                            stand_in=stand_in(bytes(r.run.run_root).hex(), r.bundle["manifest_sha256"]))


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
