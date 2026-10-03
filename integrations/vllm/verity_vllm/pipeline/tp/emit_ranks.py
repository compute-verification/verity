"""The template rank's side of a relabeled TP Build (`build --tp W --rank t --emit-ranks R --emit-root DIR`).

After rank t has derived and written its own files, every rank r in R gets its files from t's derivation
(`program.frontend.relabel.rank_report`), written by the writer that writes t's (`build._write_program`), into
`DIR/rank<r>/<name of --out>`.  A rank's directory appears by one rename of a complete directory; a rank whose relabel refused or
raised has none, and its error is in `<--out>.relabel.json` (beside the template's directory, which stays a plain derive's), so the
row derives that rank for real.  A relabeled rank's `result.json` and `construction_manifest.json` carry `relabeled_from` and the
template's telemetry (the rank spent no wall time or memory of its own), and its `inputs.json` is the template's trace: the process
that wrote these bytes read what that trace lists."""
from __future__ import annotations

import copy
import json
import os
import resource
import shutil
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

SCHEMA = "verity-vllm/build-relabel/v1"


@dataclass
class Emit:
    """The ranks to write from the template's derivation, where, and what happened to each."""
    ranks: tuple[int, ...]
    root: str
    pending: list[tuple[int, str, str, dict]] = field(default_factory=list)   # (rank, its .tmp dir, its dir, its res), written
    report: list[dict] = field(default_factory=list)

    def dir_of(self, out: str, r: int) -> str:
        return os.path.join(self.root, f"rank{r}", os.path.basename(os.path.normpath(out)))


def parse_ranks(arg: str, world: int, rank: int, root: str | None) -> Emit:
    """`--emit-ranks 0,2,3` of rank `rank` of a world of `world`, into `--emit-root root`."""
    ranks = tuple(int(x) for x in arg.split(",") if x.strip())
    if world < 2 or not root or not ranks or len(set(ranks)) != len(ranks) or any(not 0 <= r < world or r == rank for r in ranks):
        raise ValueError(f"--emit-ranks {arg!r} --emit-root {root!r}: other ranks of the world of {world} than --rank {rank}, and a root")
    return Emit(ranks, root)


def tmp_of(dst: str) -> str:
    """A rank's directory while it is written: hidden, so no `build_request_LP*_T*` glob of a reader sees it."""
    return os.path.join(os.path.dirname(dst), f".{os.path.basename(dst)}.relabel-tmp")


def _rss_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


def emit(e: Emit, out: str, pre: dict, session, *, shards: Callable[[int], list[dict]], context: Callable[[], Any],
         post: Callable | None, write: Callable[..., None]) -> None:
    """Every rank of `e`: its report from the template's `session` (re-emitted in `context()`, the derive's contexts), then `post`
    (the Boolean lowering when the target's IR is boolean), written by `write(dir, res, rep, rank=r)` with `res` = the template's
    `pre` (its `res` before its own writer ran) at rank r: `tp.rank`, `model_pin.rank` and `tp.embedding_shards` (`shards(r)`)."""
    from verity_vllm.program.frontend.relabel import RelabelRefused, rank_report, shard_map
    for r in e.ranks:
        t0, dst = time.time(), e.dir_of(out, r)
        tmp = tmp_of(dst)
        rec: dict[str, Any] = {"rank": r, "dir": dst, "ok": False, "error": None}
        try:
            if session is None:
                raise RelabelRefused("the template's derivation kept no session")
            shutil.rmtree(tmp, ignore_errors=True)
            os.makedirs(tmp)
            res = copy.deepcopy(pre)
            res["tp"]["rank"] = res["model_pin"]["rank"] = r
            res["tp"]["embedding_shards"] = shards(r)
            with context():
                rep = rank_report(session, r, shards=shard_map(pre["tp"]["embedding_shards"], res["tp"]["embedding_shards"]))
            rep = post(rep) if post is not None else rep
            if not rep.ok:
                raise RelabelRefused(f"rank {r}'s report is a refusal: {rep.unsupported}")
            write(tmp, res, rep, rank=r)
            e.pending.append((r, tmp, dst, res))
            rec["ok"] = True
        except Exception as ex:  # noqa: BLE001 -- any failure: this rank is derived for real
            shutil.rmtree(tmp, ignore_errors=True)
            rec["error"] = f"{type(ex).__name__}: {ex}"[:4000]
            rec["traceback_tail"] = traceback.format_exc()[-3000:]
        rec["secs"], rec["max_rss_mb"] = round(time.time() - t0, 2), _rss_mb()
        e.report.append(rec)


def finish(e: Emit, out: str, res: dict, write_manifest: Callable[[str, dict], None]) -> None:
    """After the template's own `result.json`: each written rank's `inputs.json`, `result.json` and `construction_manifest.json`, then
    its directory renamed into place; `<out>.relabel.json` last."""
    by_rank = {rec["rank"]: rec for rec in e.report}
    for r, tmp, dst, res_r in e.pending:
        try:
            shutil.copyfile(os.path.join(out, "inputs.json"), os.path.join(tmp, "inputs.json"))
            res_r["wall_secs_total"], res_r["max_rss_mb"] = res.get("wall_secs_total"), res.get("max_rss_mb")
            res_r["relabeled_from"] = {"rank": res["tp"]["rank"], "dir": out}
            json.dump(res_r, open(os.path.join(tmp, "result.json"), "w"), indent=1, default=str)
            write_manifest(tmp, res_r)
            shutil.rmtree(dst, ignore_errors=True)
            os.replace(tmp, dst)
        except Exception as ex:  # noqa: BLE001
            shutil.rmtree(tmp, ignore_errors=True)
            by_rank[r].update(ok=False, error=f"{type(ex).__name__}: {ex}"[:4000])
    for r in e.ranks:
        by_rank.setdefault(r, {"rank": r, "dir": e.dir_of(out, r), "ok": False, "error": "the template's derive failed: no relabel ran"})
    doc = {"schema": SCHEMA, "template": {"dir": out, "world": res.get("tp", {}).get("world"), "rank": res.get("tp", {}).get("rank"),
                                          "ok": res.get("ok") is True}, "root": e.root, "ranks": [by_rank[r] for r in e.ranks]}
    json.dump(doc, open(f"{os.path.normpath(out)}.relabel.json", "w"), indent=1)


def read_report(out: str) -> dict:
    """`<out>.relabel.json`, or a report in which every rank failed (the template's process died before writing it)."""
    try:
        return json.load(open(f"{os.path.normpath(out)}.relabel.json"))
    except (OSError, ValueError):
        return {"schema": SCHEMA, "ranks": [], "missing": True}


__all__ = ["Emit", "SCHEMA", "emit", "finish", "parse_ranks", "read_report", "tmp_of"]
