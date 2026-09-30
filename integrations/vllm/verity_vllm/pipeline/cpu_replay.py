"""The C2 replay of a deferred Commit, on CPU, from its sealed replay bundle (`pipeline.replay_bundle`).

    verity-vllm commit-replay BUNDLE --row-dir D --checkpoint-dir CK [--checkpoints-manifest M --model REPO --revision REV]

The bundle is opened against its manifest and against the Commit's record beside it: the manifest sha256 the Commit's verdict names,
the run root of the instrumented run's row (`runs.jsonl`), the dumped store's step roots folding to it, the binding map, the
required manifest and the Build's Programs of record (under `D`).  The weights the relations consume are read from the checkpoint and
OPENED against the committed weights root: every registered tensor the checkpoint composes (`weights_of_record.compose`, in its
registered dtype) must hash to its tensor root in the committed tree, the tree must fold to the root of record, and a tensor that
does not open fails the replay by name (`weights_check.equal` False).  The registered buffers and engine constants no checkpoint holds
are the bundle's bytes, opened against the same root.  A tensor the checkpoint cannot compose is not served (a relation that reads it
is not evaluated, named).

Then the same C2 as the in-process Commit (`pipeline.c2_replay.run`), and the run's verdict (`check.verdict.commit_verdict`): the
row in `runs.jsonl`, `summary.json` and `verdict.json` are rewritten with the replay on the record and `commit.weights_attest`.
Exit 0 PASS, 3 FAIL, 2 refused (a bundle that is not this record's).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from verity_vllm.commit import scheme
from verity_vllm.commit import store_dump as SD
from verity_vllm.config import option
from verity_vllm.pipeline import c2_replay as C2
from verity_vllm.pipeline import replay_bundle as RB

ATTEST = "verity-vllm/weights-attest/v1"
#: registered dtype -> (the Program's value bits, the words the replay reads): the widths the live provider serves
WORDS = {"bfloat16": (16, np.uint16), "float32": (32, np.uint32), "float8_e4m3fn": (8, np.uint8), "float8_e4m3fnuz": (8, np.uint8),
         "uint8": (8, np.uint8), "int8": (8, np.uint8)}


@dataclass(frozen=True, kw_only=True)
class Options:
    bundle: str = option("bundle", help="the sealed replay bundle (<commit out>/replay_bundle_p<pair>)")
    row_dir: str = option("--row-dir", help="the row dir holding the Build's Programs of record (and manifest.json)")
    checkpoint_dir: str = option("--checkpoint-dir", help="the checkpoint of record (safetensors shards + config.json)")
    checkpoints_manifest: str | None = option("--checkpoints-manifest", default=None, help="manifests/checkpoints.json (shard sha256 pins)")
    model: str | None = option("--model", default=None, help="the checkpoint's repo (its checkpoints.json entry)")
    revision: str | None = option("--revision", default=None, help="the checkpoint's revision")
    required_manifest: str | None = option("--required-manifest", default=None,
                                           help="the Commit's required manifest (default: its recorded path, else <row-dir>/manifest.json)")
    threads: int = option("--threads", type=int, default=8, help="threads opening the weights")


class Refused(Exception):
    """The bundle is not the Commit record's, or the record has no deferred replay to decide."""


def tensor_root(raw: np.ndarray, chunk: int) -> bytes:
    mv = memoryview(raw).cast("B")
    return scheme.fold([scheme.pos_leaf(mv[o:o + chunk]) for o in range(0, len(mv), chunk)])


def _open_one(bundle: Path, tree: dict[str, Any], i: int, ck: Any) -> tuple[str, np.ndarray | None, str | None, bool]:
    """(source, the tensor's bytes when they open against its tensor root, why not, whether the bytes were read at all)."""
    from verity_vllm.check import weights_of_record as WR

    name, dt, shape, nb = tree["names"][i]
    if name in (tree.get("extra") or {}):
        src, raw = "bundle", np.fromfile(bundle / "weights" / "extra" / f"{i}.bin", dtype=np.uint8)
    else:
        src = "checkpoint"
        try:
            words, _note = WR.compose(name, list(shape), WORDS[dt][0], ck)
        except (KeyError, ValueError) as e:
            return src, None, f"not composed from the checkpoint: {e}"[:200], False
        raw = np.ascontiguousarray(words).reshape(-1).view(np.uint8)
    if raw.size != int(nb):
        return src, None, f"{raw.size} B, registered {nb} B", True
    if tensor_root(raw, int(tree["chunk"])) != bytes.fromhex(tree["tensor_roots"][i]):
        return src, None, "tensor root != the committed tensor root", True
    return src, raw, None, True


def open_weights(bundle: Path, tree: dict[str, Any], root_of_record: str | None, ck: Any, *, threads: int = 8
                 ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """The registered tensors in the widths the replay reads, each opened against the committed weights root, and the weights check."""
    t0 = time.perf_counter()
    names = tree["names"]
    roots = [bytes.fromhex(r) for r in tree["tensor_roots"]]
    folded = scheme.weights_root(bytes.fromhex(tree["geo_digest"]), scheme.names_digest(names), roots).hex()
    servable = [i for i, (_n, dt, _s, _b) in enumerate(names) if dt in WORDS]
    with ThreadPoolExecutor(max(1, threads)) as pool:
        opened = list(pool.map(lambda i: (i, *_open_one(bundle, tree, i, ck)), servable))
    words: dict[str, np.ndarray] = {}
    mismatched, unopened, by_src = [], [], {"checkpoint": 0, "bundle": 0}
    for i, src, raw, why, read in opened:
        name = names[i][0]
        if raw is not None:
            words[name] = raw.view(WORDS[names[i][1]][1])
            by_src[src] += 1
        elif not read:
            unopened.append({"name": name, "why": why})
        else:
            mismatched.append({"name": name, "source": src, "why": why})
    why = ("the bundle's weights tree does not fold to its root" if folded != tree["root"] else
           f"the bundle's weights root {tree['root'][:16]} is not the committed weights root {str(root_of_record)[:16]}" if tree["root"] != root_of_record else
           "slices that do not open against the committed weights root: " + ", ".join(f"{m['name']} ({m['why']})" for m in mismatched[:4])
           + (f" (+{len(mismatched) - 4} more)" if len(mismatched) > 4 else "") if mismatched else None)
    check = {"equal": why is None, "committed_root": root_of_record, "replay_root": folded, "why": why, "source": "checkpoint",
             "method": "checkpoint tensors opened against the committed weights root (tensor roots under the root of record)",
             "tensors": len(names), "servable": len(servable), "opened": len(words), "opened_by_source": by_src,
             "mismatched_n": len(mismatched), "mismatched": mismatched[:16], "unopened_n": len(unopened), "unopened": unopened[:16],
             "shards": getattr(ck, "pin", None), "wall_s": round(time.perf_counter() - t0, 3)}
    return words, check


def provider(words: dict[str, np.ndarray], prefix: str) -> Callable[[str, int, int], np.ndarray | None]:
    """The replay's weights reader over the opened tensors (Program field -> registered name, as `opening.model_weights_provider`)."""
    def read(field: str, lo: int, hi: int) -> np.ndarray | None:
        name = field[len(prefix):] if prefix and field.startswith(prefix) else field
        a = words[name] if name in words else words.get(field)
        return None if a is None or hi > a.size else a[lo:hi]

    return read


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _write(p: Path, text: str) -> None:
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, p)


def of_record(b: dict[str, Any], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], Any, Any]:
    """(runs.jsonl rows, the deferred run's row, verdict.json, store, run) once the bundle is shown to be this record's; else Refused."""
    ctx, bdir, pair = b["context"], Path(b["path"]), int(b["context"]["pair"])
    verdict = json.loads((out / "verdict.json").read_text())
    ent = next((x for x in ((verdict.get("replay_deferred") or {}).get("bundles") or []) if int(x["pair"]) == pair), None)
    if ent is None or ent.get("manifest_sha256") != b["manifest_sha256"]:
        raise Refused(f"the Commit's verdict names no bundle of pair {pair} with manifest sha256 {b['manifest_sha256'][:16]}")
    rows = [json.loads(ln) for ln in (out / "runs.jsonl").read_text().splitlines() if ln.strip()]
    row = next((r for r in rows if r["pair"] == pair and r["arm"] != "control"), None)
    if row is None or not RB.awaits_replay(row.get("commit") or {}):
        raise Refused(f"pair {pair}: no instrumented run awaiting its deferred replay in {out / 'runs.jsonl'}")
    store, run = SD.load(bdir / "store")
    c = row["commit"]
    folded = store._run_root(run)
    for what, got in (("the bundle's run root", ctx["run_root"]), ("the store's run root", run.run_root.hex()),
                      ("the store's step roots folded", folded.hex() if folded else None)):
        if got != c.get("run_root"):
            raise Refused(f"{what} {str(got)[:16]} is not the run root of record {str(c.get('run_root'))[:16]}")
    bm = out / f"binding_map_p{pair}.json"
    if not bm.is_file() or _sha256(bm) != b["manifest"]["files"]["binding_map.json"]["sha256"]:
        raise Refused(f"{bm} is not the bundle's binding map")
    run.weight_registration = c.get("weight_registration")
    return rows, row, verdict, store, run


def manifest_of(o: Options, ctx: dict[str, Any], args: SimpleNamespace) -> tuple[str | None, Any]:
    from verity_vllm.commit.partition import require_partition

    want = ctx.get("required_manifest_sha256")
    if not want:
        return None, None
    recorded = ctx["args"].get("required_manifest")
    p = o.required_manifest or (recorded if recorded and os.path.isfile(recorded) else os.path.join(o.row_dir, "manifest.json"))
    if hashlib.sha256(Path(p).read_bytes()).hexdigest() != want:
        raise Refused(f"{p} is not the Commit's required manifest (sha256 {want[:16]})")
    return p, require_partition(json.loads(Path(p).read_text()), args.programs_root, args.program_dir)


def replay_args(o: Options, ctx: dict[str, Any]) -> SimpleNamespace:
    """The Commit's replay arguments, with its Programs read from this row dir (their digests are checked against the record)."""
    a = dict(ctx["args"])
    a["programs_root"] = o.row_dir if a.get("programs_root") else None
    a["program_dir"] = os.path.join(o.row_dir, "build_request") if a.get("program_dir") else None
    return SimpleNamespace(**a, replay_cache=False, vu_export_dir=None)


def engine_of_record(row: dict[str, Any], ctx: dict[str, Any], art: Path) -> None:
    """The replay's `engine_of_record` is the Commit engine's (read in its process, in the bundle); this process's beside it."""
    eng = ctx.get("engine_of_record")
    if eng is None:
        return
    doc = json.loads(art.read_text())
    for sr in (row["commit"]["sampled_replay"], doc["sampled_replay"]):
        sr["engine_of_replay"], sr["engine_of_record"] = sr.get("engine_of_record"), eng
    _write(art, json.dumps(doc, indent=1, default=str) + "\n")


def decide(row: dict[str, Any], ctx: dict[str, Any], wc: dict[str, Any], b: dict[str, Any]) -> None:
    """The run's value correspondence and verdict with the replay on the record (in the in-process Commit's component order)."""
    from verity_vllm.check.verdict import commit_verdict

    c = row["commit"]
    vc = c["value_correspondence"]
    comps = vc["components"]
    for k in ("weights_pin", "replay"):
        if k in comps:
            comps[k] = comps.pop(k)
    fails = [(k, x) for k, x in comps.items() if x.get("result") is False]
    vc["why"] = "; ".join(f"{k}: {x.get('why')}" for k, x in fails) if fails else None
    c["weights_attest"] = {"schema": ATTEST, "result": wc["equal"], "pin": (c.get("weights_pin") or {}).get("result"),
                           "root": wc["committed_root"], "root_openings": f"{wc['opened']}/{wc['servable']}",
                           "opened_by_source": wc["opened_by_source"], "mismatched": [m["name"] for m in wc["mismatched"]],
                           "unopened": [u["name"] for u in wc["unopened"]], "source": "checkpoint", "why": wc["why"],
                           "replay": {"where": "cpu", "bundle": b["path"], "manifest_sha256": b["manifest_sha256"], "wall_s": wc["wall_s"]}}
    c["commit_pass"], c["commit_fail_reason"] = commit_verdict(c, row["tokens_equal_ref"], min_openings=ctx["min_openings"],
                                                               declaration=ctx["declaration"], required_classes=ctx["required_classes"])


def record(out: Path, rows: list[dict[str, Any]], verdict: dict[str, Any], b: dict[str, Any], wall_s: float) -> None:
    """runs.jsonl, summary.json and verdict.json with the decided run, each replaced whole."""
    _write(out / "runs.jsonl", "".join(json.dumps(r, default=str) + "\n" for r in rows))
    inst = [r for r in rows if r["arm"] != "control"]
    fails = [(r["pair"], r["arm"], r["commit"].get("commit_fail_reason")) for r in inst
             if not r["commit"].get("commit_pass") and not RB.awaits_replay(r["commit"])]
    pending = [r["pair"] for r in inst if RB.awaits_replay(r["commit"])]
    verdict.update(commit_pass=RB.pass_of(rows), runs_failed=len(fails), first_fail_reason=(fails[0][2] if fails else None))
    verdict["replay_deferred"]["pending_pairs"] = pending
    verdict["replay_deferred"].setdefault("replayed", []).append(
        {"pair": int(b["context"]["pair"]), "bundle": b["path"], "manifest_sha256": b["manifest_sha256"], "where": "cpu", "wall_s": round(wall_s, 3),
         "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    _write(out / "verdict.json", json.dumps(verdict, indent=1) + "\n")
    sp = out / "summary.json"
    if sp.is_file():
        summary = json.loads(sp.read_text())
        summary["commit_pass"] = verdict["commit_pass"]
        _write(sp, json.dumps(summary, indent=1, default=str) + "\n")


def replay(o: Options) -> bool | None:
    """The bundle's C2 replay and the run's verdict on the record; the verdict's commit_pass."""
    from verity_vllm.check import weights_of_record as WR
    from verity_vllm.pipeline.timeline import Timeline

    t0 = time.perf_counter()
    tl = Timeline.from_env("commit_replay")
    b = RB.open_bundle(o.bundle)
    ctx, bdir = b["context"], Path(b["path"])
    out, pair = bdir.parent, int(ctx["pair"])
    rows, row, verdict, store, run = of_record(b, out)
    tree = json.loads((bdir / "weights" / "tree.json").read_text())
    args = replay_args(o, ctx)
    man_path, man = manifest_of(o, ctx, args)
    args.required_manifest = man_path
    with tl.span("replay.open_weights", pair=pair) as sp:
        ck = WR.CheckpointIndex(o.checkpoint_dir, checkpoints_manifest=o.checkpoints_manifest, model=o.model, revision=o.revision)
        words, wc = open_weights(bdir, tree, (row["commit"].get("weight_registration") or {}).get("root"), ck, threads=o.threads)
        sp.update(equal=wc["equal"], opened=wc["opened"], servable=wc["servable"])
    print(f"[replay] WEIGHTS (pair {pair}): {'OPENED' if wc['equal'] else 'FAIL'} {wc['opened']}/{wc['servable']} registered tensors against the committed "
          f"weights root {str(wc['committed_root'])[:16]} ({wc['opened_by_source']}; unopened {wc['unopened_n']}) in {wc['wall_s']} s"
          + (f" -- {wc['why']}" if wc["why"] else ""), flush=True)
    vc = row["commit"]["value_correspondence"]
    C2.run(C2.Site(args=args, out=out, pair=pair, com=store, rc=run, row=row, reqs=ctx["requests"], sampling=ctx["sampling"], tokens=ctx["tokens"],
                   manifest=man, stop_of_record=ctx["stop_of_record"], ids_full=ctx["ids_full"], versions={"git_head": ctx.get("code")},
                   components=vc.setdefault("components", {}), vc_result=ctx.get("vc_result"), weights_of=lambda p: provider(words, p),
                   weights_check=lambda: dict(wc), tl=tl), {})
    engine_of_record(row, ctx, out / f"sampled_replay_p{pair}.json")
    decide(row, ctx, wc, b)
    record(out, rows, verdict, b, time.perf_counter() - t0)
    c = row["commit"]
    print(f"[replay] COMMIT {'PASS' if c['commit_pass'] else 'FAIL'} (pair {pair}): {c.get('commit_fail_reason') or 'every check on the record'}; "
          f"verdict commit_pass {verdict['commit_pass']}", flush=True)
    return verdict["commit_pass"]


def main(o: Options) -> int:
    try:
        ok = replay(o)
    except Refused as e:
        print(f"[replay] REFUSED: {e}", flush=True)
        return 2
    sys.stdout.flush()
    return 0 if ok else 3
