"""The serving rows of a deferred Commit (``verity-vllm serving-rows BUNDLE --row-dir D --partition F --out O``): the sealed replay
bundle opened against the Commit's record as the CPU replay opens it (manifest sha256, run root, binding map, the Build's Programs of
record), the weights opened against the committed weights root, and the window `pipeline.serving_rows.after_commit` writes, the same
window the in-Commit hook writes, into ``O``.  A slim bundle holds the replay's picks only, not the population: refused."""
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


def from_bundle(bundle: str | os.PathLike, row_dir: str, partition_file: str, out_dir: str | os.PathLike, *, checkpoint: Any = None,
                weights: Callable | None = None, required_manifest: str | None = None, threads: int = 8,
                workers: int | None = None) -> dict[str, Any]:
    """The serving rows of a deferred Commit, from its sealed replay bundle: opened against the Commit's record beside it as the CPU
    replay opens it (`cpu_replay.of_record`, decided or not), its Programs the Build's of record under ``row_dir``, the arrivals the
    replay's, and the weights opened against the committed weights root from the bundle's tensors and ``checkpoint`` (a
    `weights_of_record.CheckpointIndex`; without one, only the tensors the bundle carries).  ``weights`` (tests) replaces the opened
    weights."""
    from verity_vllm.check.oracle_compare import OpenedReader

    b = RB.open_bundle(bundle)
    ctx, bdir = b["context"], Path(b["path"])
    out, pair = bdir.parent, int(ctx["pair"])
    if ctx.get("slim"):
        raise CPU.Refused(f"{bdir} is a slim bundle: it holds the replay's picks, not every unit of the population")
    _rows, row, _verdict, store, run = CPU.of_record(b, out, awaiting=False)
    o = SimpleNamespace(row_dir=row_dir, replay_workers=None, required_manifest=required_manifest)
    args = CPU.replay_args(o, ctx)
    _path, man = CPU.manifest_of(o, ctx, args)
    site = SimpleNamespace(args=args, manifest=man, reqs=ctx["requests"], sampling=ctx["sampling"], row=row)
    rl = C2.request_labels(ctx["requests"])
    loaded, digs = C2.programs_of_record(site, rl)
    progs = {k: v["instances"] for k, v in loaded.items()}
    arrivals, _moved = C2.arrivals(site, rl)
    if weights is None:
        k0 = next(iter(progs))
        prefix = ProgramIndex(progs[k0], program_dir=loaded[k0].get("dir")).prefix
        tree = json.loads((bdir / "weights" / "tree.json").read_text())
        words, wc = CPU.open_weights(bdir, tree, (row["commit"].get("weight_registration") or {}).get("root"), checkpoint, threads=threads)
        if not wc["equal"]:
            raise CPU.Refused(f"the bundle's weights: {wc['why']}")
        weights = CPU.provider(words, prefix, set())
    bmap = json.loads((out / f"binding_map_p{pair}.json").read_text())
    committer = {"from": "replay bundle", "bundle_manifest_sha256": b["manifest_sha256"], "pair": pair, "source": ctx.get("code")}
    return SRW.after_commit(str(out_dir), partition_file=partition_file, programs=progs, bmap=bmap, committed=OpenedReader(store, run),
                            weights=weights, manifest=man if isinstance(man, dict) else None, run_root=run.run_root,
                            row=(ctx.get("args") or {}).get("label"), arrivals=arrivals,
                            program_dirs={k: v.get("dir") for k, v in loaded.items()}, digests=digs, committer=committer, workers=workers)


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
