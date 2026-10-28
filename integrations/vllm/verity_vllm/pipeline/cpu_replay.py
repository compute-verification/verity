"""The C2 replay of a deferred Commit, on CPU, from its sealed replay bundle (`pipeline.replay_bundle`).

    verity-vllm commit-replay BUNDLE --row-dir D --checkpoint-dir CK [--checkpoints-manifest M --model REPO --revision REV]

The bundle is opened against its manifest and against the Commit's record beside it: the manifest sha256 the Commit's verdict names,
the run root of the instrumented run's row (`runs.jsonl`), the dumped store's step roots folding to it, the binding map, the
required manifest and the Build's Programs of record (under `D`), with the Definitions they bound.  The weights the relations
consume are read from the checkpoint and OPENED against the committed weights root: every registered tensor the checkpoint composes
(`weights_of_record.compose`, in its registered dtype) must hash to its tensor root in the committed tree, the tree must fold to the
root of record, and a tensor that does not open fails the replay by name (`weights_check.equal` False).  The registered buffers and
engine constants no checkpoint holds are the bundle's bytes, opened against the same root.  A tensor the checkpoint cannot compose is
not served (a relation that reads it is not evaluated, named).  The tree is rows-sha512's, its tensors SHA-512 rows
(`step_rows.tensor_root`) under a 64-byte root of record; a tree of another scheme, or under another root, is refused by name
(`tree_refusal`).

Then the same C2 as the in-process Commit (`pipeline.c2_replay.run`), and the run's verdict (`check.verdict.commit_verdict`): the
row in `runs.jsonl`, `summary.json` and `verdict.json` are rewritten with the replay on the record and `commit.weights_attest`.
Exit 0 PASS, 3 FAIL, 2 refused (a bundle that is not this record's, or a pick too large for this host's memory: `driver.PoolDoesNotFit`).

Before the record is rewritten, the decided replay leaves its SLIM KEEP beside the bundle, `<out>/replay_slim_p<pair>/` (`keep`): what a
later replay of the same picks reads, so the bundle can go and the Commit can still be replayed on CPU, by other row kernels, without
serving again (the sample is seeded from the run root, so it draws the same picks).  Not under other Definition bodies: the replay is
refused unless the Definitions its tree evaluates the picks by are the ones the Build of record bound (`check.replay.definitions`).

    store/            the bundle's store, slim (`commit.store_dump`, `touched=`): only the byte ranges and Merkle nodes the replay and the
                      linkage opened, folding to the run root
    weights/          the bundle's weights tree and the tensors it carries, slim (`slim_tree`): under rows-sha512 the salt keys and
                      bundle bytes of the tensors the replay consumed alone, never the registration's key;
    binding_map.json, context.json  the bundle's
    slim.json         {schema, pair, run_root, program_digest, programs, binding_map_sha256, bundle_manifest_sha256, commit_pass, picked,
                       picks_key, picks, touched_bytes, files: {path: {sha256, bytes}}, bytes, dump_sha256}, written last

`dump_sha256` is the sha256 of the canonical JSON of `files`.  It is built under `replay_slim_p<pair>.partial/` and renamed into place.

BUNDLE may be that keep (`open_keep`): its files are checked against `slim.json`, and the record beside it, decided by a replay of the
bundle the keep was cut from (the verdict's `replayed` names that bundle's manifest sha256), is read back in its deferred state
(`undecided`), so the replay decides the run again from its own draw.  It is refused before anything is evaluated unless this code's
draw is the keep's picks in order (`kept_draw`).  A replay from a keep writes no keep.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from verity_vllm.commit import step_rows
from verity_vllm.commit import store_dump as SD
from verity_vllm.config import option
from verity_vllm.pipeline import c2_replay as C2
from verity_vllm.pipeline import replay_bundle as RB
from verity_vllm.program.compiled_relations import warm_vml

ATTESTS = ("every weight field the replay consumed, composed from the checkpoint (or carried in the bundle), has the committed tensor root, "
           "and the tensor roots fold to the committed weights root")
#: registered dtype -> (the Program's value bits, the words the replay reads): the widths the live provider serves
WORDS = {"bfloat16": (16, np.uint16), "float32": (32, np.uint32), "float8_e4m3fn": (8, np.uint8), "float8_e4m3fnuz": (8, np.uint8),
         "uint8": (8, np.uint8), "int8": (8, np.uint8), "int32": (32, np.uint32)}
SLIM_SCHEMA = "verity-vllm/replay-slim/v1"
SLIM_DOC = "slim.json"
PICKS_KEY = "[request, request-relative step, Program row, op_path] of each pick, in draw order"


@dataclass(frozen=True, kw_only=True)
class Options:
    bundle: str = option("bundle", help="the sealed replay bundle (<commit out>/replay_bundle_p<pair>), or its slim keep (replay_slim_p<pair>)")
    row_dir: str = option("--row-dir", help="the row dir holding the Build's Programs of record (and manifest.json)")
    checkpoint_dir: str = option("--checkpoint-dir", help="the checkpoint of record (safetensors shards + config.json)")
    checkpoints_manifest: str | None = option("--checkpoints-manifest", default=None, help="manifests/checkpoints.json (shard sha256 pins)")
    model: str | None = option("--model", default=None, help="the checkpoint's repo (its checkpoints.json entry)")
    revision: str | None = option("--revision", default=None, help="the checkpoint's revision")
    required_manifest: str | None = option("--required-manifest", default=None,
                                           help="the Commit's required manifest (default: its recorded path, else <row-dir>/manifest.json)")
    threads: int = option("--threads", type=int, default=8, help="threads opening the weights")
    replay_workers: int | None = option("--replay-workers", type=int, default=None,
                                        help="the forked pool's workers (default: the Commit's own; the draw does not depend on it)")
    worker_peak_gb: float | None = option("--worker-peak-gb", type=float, default=None,
                                          help="a forked worker's measured peak memory beyond what the row evaluator states for its pick, "
                                               "GiB (default: the word replay's, admission commit.c2.replay_workers.small_vu): after the "
                                               "prewarm the workers are fitted to the cgroup's memory.max less its use (`driver.fit_pool`)")


class Refused(Exception):
    """The bundle is not the Commit record's, or the record has no deferred replay to decide."""


def open_keep(keep: str | os.PathLike) -> dict[str, Any]:
    """A slim keep opened as `replay_bundle.open_bundle` opens a bundle: every file checked against `slim.json` (a missing, extra or
    changed file is refused by name), the manifest sha256 and binding map those of the bundle it was cut from."""
    k = Path(keep)
    doc = json.loads((k / SLIM_DOC).read_text())
    if doc.get("schema") != SLIM_SCHEMA:
        raise ValueError(f"{k}: keep schema {doc.get('schema')!r} is not {SLIM_SCHEMA}")
    files = doc["files"]
    if hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest() != doc.get("dump_sha256"):
        raise ValueError(f"{k}: its file list is not the keep's dump sha256 {str(doc.get('dump_sha256'))[:16]}")
    have = {f.relative_to(k).as_posix() for f in k.rglob("*") if f.is_file()} - {SLIM_DOC}
    if have != set(files):
        raise ValueError(f"{k}: files differ from slim.json: missing {sorted(set(files) - have)[:4]}, extra {sorted(have - set(files))[:4]}")
    rels = sorted(files)
    with ThreadPoolExecutor(8) as pool:
        bad = [rel for rel, got in zip(rels, pool.map(lambda r: RB._sha256(k / r), rels)) if got != files[rel]["sha256"]]
    if bad:
        raise ValueError(f"{k}: files whose sha256 is not slim.json's: {bad[:4]}")
    ctx = json.loads((k / "context.json").read_text())
    if (int(ctx["pair"]), ctx["run_root"], files["binding_map.json"]["sha256"]) != (int(doc["pair"]), doc["run_root"], doc["binding_map_sha256"]):
        raise ValueError(f"{k}: its context and binding map are not slim.json's (pair, run root, binding map sha256)")
    return {"path": str(k), "manifest": {"files": {"binding_map.json": dict(files["binding_map.json"])}},
            "manifest_sha256": doc["bundle_manifest_sha256"], "context": ctx, "keep": doc}


def open_source(path: str | os.PathLike) -> dict[str, Any]:
    """The replay's source: a slim keep (it holds `slim.json`), else a sealed bundle."""
    return open_keep(path) if (Path(path) / SLIM_DOC).is_file() else RB.open_bundle(path)


def undecided(c: dict[str, Any], bundle: str) -> None:
    """A decided run's commit record back in its deferred state, in place: what the decided replay wrote (`c2_replay.record`, the
    linkage's opened values, `decide`) taken off, and the deferred records of ``bundle`` (its path of record) put back."""
    sr, lk = RB.deferred_records(Path(bundle))
    c.update(commit_pass=None, commit_fail_reason=RB.PENDING, sampled_replay=sr, boundary_linkage=lk)
    for k in ("weights_attest", "inactive_by_enablement", "not_executed_reconciliation"):
        c.pop(k, None)
    ov = c.get("opened_values")
    if isinstance(ov, dict):
        ov.pop("replay_linkage", None)
        if set(ov) <= {"method"}:
            c.pop("opened_values")
    vc = c.get("value_correspondence")
    if isinstance(vc, dict):
        for k in ("sampled_replay", "boundary_linkage", "negatives"):
            (vc.get("components") or {}).pop(k, None)
        vc["method"] = re.sub(r"^sampled-exact-replay\([^)]*\) \+ boundary-linkage \+ ", "", vc.get("method") or "")
        vc["result"] = None


def tree_scheme(tree: dict[str, Any]) -> str | None:
    """The scheme a weights tree was registered under: its ``scheme``, None for a tree recorded before trees named one."""
    return tree.get("scheme")


def root_scheme(root_of_record: str | None) -> str | None:
    """The scheme a committed weights root of record was registered under, by its size: rows-sha512 for 64 bytes, else None."""
    return step_rows.NAME if len(root_of_record or "") // 2 == step_rows.ROOT_BYTES else None


def tensor_key_of(tree: dict[str, Any], i: int) -> bytes | None:
    """A rows-sha512 tree's salt key of tensor ``i``: derived from the registration's ``Kw`` (a bundle's tree), else the slim keep's own
    key for it (`slim_tree`); None when the keep carries none."""
    if "key" in tree:
        return step_rows.tensor_key(bytes.fromhex(tree["key"]), i)
    ks = tree["tensor_keys"].get(str(i))
    return bytes.fromhex(ks) if ks else None


def tensor_root_of(tree: dict[str, Any], i: int, raw: np.ndarray) -> bytes:
    """Tensor ``i``'s root (`step_rows.keyed_tensor_root` under `tensor_key_of`)."""
    ks = tensor_key_of(tree, i)
    if ks is None:
        raise KeyError(f"registered tensor {i} ({tree['names'][i][0]}): the slim keep carries no salt key for it")
    return step_rows.keyed_tensor_root(bytes.fromhex(tree["geo_digest"]), i, tree["names"][i], memoryview(raw).cast("B"), ks)


def slim_tree(tree: dict[str, Any], opened: Iterable[str]) -> dict[str, Any]:
    """The weights tree a slim keep carries, which is published: never the registration's ``Kw`` (every weights leaf's salt derives
    from it), only ``step_rows.tensor_key(Kw, t)`` of each tensor ``opened`` names (those the replay consumed and opened), and the
    bundle's bytes of those alone."""
    at, keep_ = {e[0]: i for i, e in enumerate(tree["names"])}, set(opened)
    keys = {str(at[n]): tensor_key_of(tree, at[n]).hex() for n in sorted(keep_ & at.keys(), key=at.get)}
    return {k: v for k, v in tree.items() if k not in ("key", "tensor_keys", "extra")} | {
        "tensor_keys": keys, "extra": {n: i for n, i in tree["extra"].items() if str(i) in keys}}


def folded_root(tree: dict[str, Any]) -> str:
    """The root the tree's tensor roots fold to."""
    roots = [bytes.fromhex(r) for r in tree["tensor_roots"]]
    return step_rows.weights_root(bytes.fromhex(tree["geo_digest"]), step_rows.names_digest(tree["names"]), roots).hex()


def _open_one(bundle: Path, tree: dict[str, Any], i: int, ck: Any, tp: dict[str, int] | None = None, rw: Any = None,
              roles: dict[str, dict[str, Any]] | None = None) -> tuple[str, np.ndarray | None, str | None, bool]:
    """(source, the tensor's bytes when they open against its tensor root, why not, whether the bytes were read at all).  ``tp``
    ({"world", "rank"}): a TP rank's registered tensors are its shards, composed by `weights_of_record.compose`'s shard rule.  ``rw``
    (`fp8_ue8m0.Rewrites` over the registered names): the fields vLLM rewrites at load, composed as it rewrites them.  ``roles`` (the
    Build's, by registered name): a field vLLM makes at load by more than a copy, composed by its role's rule (`derived_weights`), as
    the weights of record are."""
    from verity_vllm.check import derived_weights
    from verity_vllm.check import weights_of_record as WR

    name, dt, shape, nb = tree["names"][i]
    src = "bundle" if name in (tree.get("extra") or {}) else "checkpoint"
    if tensor_key_of(tree, i) is None:
        return src, None, "the slim keep carries no salt key for it", False
    if src == "bundle":
        raw = np.fromfile(bundle / "weights" / "extra" / f"{i}.bin", dtype=np.uint8)
    else:
        if ck is None:
            return src, None, "no checkpoint given", False
        try:
            world, rank = (int(tp["world"]), int(tp["rank"])) if tp else (1, 0)
            if rw is not None and rw.covers(name):
                words, _note = rw.compose(name, list(shape))
            else:
                words, _note = (derived_weights.compose(name, list(shape), WORDS[dt][0], (roles or {}).get(name), ck, world, rank)
                                or WR.compose(name, list(shape), WORDS[dt][0], ck, world=world, rank=rank))
        except (KeyError, ValueError) as e:
            return src, None, f"not composed from the checkpoint: {e}"[:200], False
        raw = np.ascontiguousarray(words).reshape(-1).view(np.uint8)
    if raw.size != int(nb):
        return src, None, f"{raw.size} B, registered {nb} B", True
    if tensor_root_of(tree, i, raw) != bytes.fromhex(tree["tensor_roots"][i]):
        return src, None, "tensor root != the committed tensor root", True
    return src, raw, None, True


def registered_roles(program_dirs: Iterable[str | None], prefix: str = "") -> dict[str, dict[str, Any]]:
    """The Build's `weight_roles` (`fp8_ue8m0.build_roles` of each Program dir) by registered name: the Program's field less the
    wrapper prefix its index strips (`ProgramIndex.prefix`, as the replay's weights provider does)."""
    from verity_vllm.check import fp8_ue8m0

    roles: dict[str, dict[str, Any]] = {}
    for d in program_dirs:
        roles.update(fp8_ue8m0.build_roles(d))
    return {(f[len(prefix):] if prefix and f.startswith(prefix) else f): r for f, r in roles.items()}


def opener(bundle: Path, tree: dict[str, Any], ck: Any, *, tp: dict[str, int] | None = None,
           roles: dict[str, dict[str, Any]] | None = None) -> Callable[[int], tuple[str, np.ndarray | None, str | None, bool]]:
    """`_open_one` of the tree's tensor i, as `open_weights` opens it: one tensor at a time, the rewrites of ``roles`` made once."""
    from verity_vllm.check import fp8_ue8m0
    from verity_vllm.check import weights_of_record as WR

    rw = None
    if roles and ck is not None:
        dims = {n: (list(s), WORDS[dt][0]) for n, dt, s, _b in tree["names"] if dt in WORDS}
        at = {"world": int(tp["world"]), "rank": int(tp["rank"])} if tp else {}
        rw = fp8_ue8m0.Rewrites(list(dims), roles, lambda n: WR.compose(n, *dims[n], ck, **at), ck.config)
    return lambda i: _open_one(bundle, tree, i, ck, tp, rw, roles)


def tree_refusal(tree: dict[str, Any], root_of_record: str | None) -> tuple[str, str | None]:
    """(the root the tree's tensor roots fold to, why the tree is not the committed weights root's tree: None when it is).  Only a
    rows-sha512 tree under a rows-sha512 root of record (`root_scheme`) is read: a tree of another scheme, or one recorded before trees
    named theirs, is refused by name before anything opens, and so is a root of record of another size."""
    have, want = tree_scheme(tree), root_scheme(root_of_record)
    if have != step_rows.NAME:
        named = "names no scheme" if have is None else f"is {have}'s"
        return "", f"the bundle's weights tree {named}: only a {step_rows.NAME} tree is read"
    if root_of_record is not None and want is None:
        return "", (f"the committed weights root is {len(root_of_record) // 2} bytes, not a {step_rows.NAME} root: only a "
                    f"{step_rows.NAME} registration is read")
    folded = folded_root(tree)
    return folded, ("the bundle's weights tree does not fold to its root" if folded != tree["root"] else
                    f"the bundle's weights root {tree['root'][:16]} is not the committed weights root {str(root_of_record)[:16]}"
                    if tree["root"] != root_of_record else None)


def open_weights(bundle: Path, tree: dict[str, Any], root_of_record: str | None, ck: Any, *, threads: int = 8,
                 tp: dict[str, int] | None = None, roles: dict[str, dict[str, Any]] | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """The registered tensors in the widths the replay reads, each opened against the committed weights root, and the weights check;
    ``tp``: those of a TP rank (`_open_one`).  ``roles``: the Build's `weight_roles` by registered name (`fp8_ue8m0.build_roles`): the
    DeepGEMM linears and MLA's absorbed kv_b they name are composed as vLLM rewrites them at load (`fp8_ue8m0.Rewrites`), and the
    fields it makes by more than a copy (DeepSeek-V4's padded attn_sink, layer 0's hc_attn_fn_broadcast) by their rule
    (`derived_weights`), as the weights of record are, since the checkpoint's own words are not the served ones."""
    t0 = time.perf_counter()
    names = tree["names"]
    folded, refused = tree_refusal(tree, root_of_record)
    servable = [i for i, (_n, dt, _s, _b) in enumerate(names) if dt in WORDS]
    open_one = opener(bundle, tree, ck, tp=tp, roles=roles)
    with ThreadPoolExecutor(max(1, threads)) as pool:
        opened = list(pool.map(lambda i: (i, *open_one(i)), servable)) if tree_scheme(tree) == step_rows.NAME else []
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
    why = refused or ("slices that do not open against the committed weights root: " + ", ".join(f"{m['name']} ({m['why']})" for m in mismatched[:4])
                      + (f" (+{len(mismatched) - 4} more)" if len(mismatched) > 4 else "") if mismatched else None)
    check = {"equal": why is None, "committed_root": root_of_record, "replay_root": folded, "why": why, "source": "checkpoint",
             "method": "checkpoint tensors opened against the committed weights root (tensor roots under the root of record)",
             "tensors": len(names), "servable": len(servable), "opened": len(words), "opened_by_source": by_src,
             "mismatched_n": len(mismatched), "mismatched": mismatched[:16], "unopened_n": len(unopened), "unopened": unopened[:16],
             "shards": getattr(ck, "pin", None), "wall_s": round(time.perf_counter() - t0, 3)}
    return words, check


class OnRead(Mapping):
    """`open_weights`' tensors, each opened on its first read and kept: a reader of a few registered tensors (a rank's serving
    rows window) opens those alone, not the rank's every shard.  The tree is held to the committed weights root first (it folds to
    its root, and that root is the root of record, else ``ValueError``); each tensor is held to its tensor root as `open_weights`
    holds it, and one that does not open is refused by name on its read (``LookupError``, which ``.get`` does not swallow)."""

    def __init__(self, bundle: Path, tree: dict[str, Any], root_of_record: str | None, ck: Any, *, tp: dict[str, int] | None = None,
                 roles: dict[str, dict[str, Any]] | None = None):
        from verity_vllm.check import fp8_ue8m0
        from verity_vllm.check import weights_of_record as WR

        names = tree["names"]
        _folded, refused = tree_refusal(tree, root_of_record)
        if refused:
            raise ValueError(refused)
        rw = None
        if roles and ck is not None:
            dims = {n: (list(s), WORDS[dt][0]) for n, dt, s, _b in names if dt in WORDS}
            at = {"world": int(tp["world"]), "rank": int(tp["rank"])} if tp else {}
            rw = fp8_ue8m0.Rewrites(list(dims), roles, lambda n: WR.compose(n, *dims[n], ck, **at), ck.config)
        self._at = {n: i for i, (n, dt, _s, _b) in enumerate(names) if dt in WORDS}
        self._open = lambda i: _open_one(bundle, tree, i, ck, tp, rw, roles)
        self._dtype = {n: WORDS[dt][1] for n, dt, _s, _b in names if dt in WORDS}
        self.words: dict[str, np.ndarray] = {}

    def __contains__(self, name: object) -> bool:
        return name in self._at

    def __iter__(self):
        return iter(self._at)

    def __len__(self) -> int:
        return len(self._at)

    def __getitem__(self, name: str) -> np.ndarray:
        if name not in self.words:
            src, raw, why, _read = self._open(self._at[name])
            if raw is None:
                raise LookupError(f"registered tensor {name} ({src}) does not open against the committed weights root: {why}")
            self.words[name] = raw.view(self._dtype[name])
        return self.words[name]


def provider(words: dict[str, np.ndarray], prefix: str, consumed: set[str]) -> Callable[[str, int, int], np.ndarray | None]:
    """The replay's weights reader over the opened tensors (Program field -> registered name, as `opening.model_weights_provider`);
    every field it is asked for lands in ``consumed``."""
    def read(field: str, lo: int, hi: int) -> np.ndarray | None:
        name = field[len(prefix):] if prefix and field.startswith(prefix) else field
        name = name if name in words or field not in words else field
        consumed.add(name)
        a = words.get(name)
        return None if a is None or hi > a.size else a[lo:hi]

    return read


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _write(p: Path, text: str) -> None:
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, p)


def of_record(b: dict[str, Any], out: Path, *, awaiting: bool = True) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], Any, Any]:
    """(runs.jsonl rows, the deferred run's row, verdict.json, store, run) once the bundle is shown to be this record's; else Refused.
    ``awaiting=False``: the run's record may be decided already (a reader of the bundle that decides nothing)."""
    ctx, bdir, pair = b["context"], Path(b["path"]), int(b["context"]["pair"])
    verdict = json.loads((out / "verdict.json").read_text())
    ent = next((x for x in ((verdict.get("replay_deferred") or {}).get("bundles") or []) if int(x["pair"]) == pair), None)
    if ent is None or ent.get("manifest_sha256") != b["manifest_sha256"]:
        raise Refused(f"the Commit's verdict names no bundle of pair {pair} with manifest sha256 {b['manifest_sha256'][:16]}")
    rows = [json.loads(ln) for ln in (out / "runs.jsonl").read_text().splitlines() if ln.strip()]
    row = next((r for r in rows if r["pair"] == pair and r["arm"] != "control"), None)
    if not awaiting:
        if row is None:
            raise Refused(f"pair {pair}: no instrumented run in {out / 'runs.jsonl'}")
    elif "keep" in b and row is not None and not RB.awaits_replay(row.get("commit") or {}):
        done = next((x for x in ((verdict.get("replay_deferred") or {}).get("replayed") or [])
                     if int(x["pair"]) == pair and x.get("manifest_sha256") == b["manifest_sha256"]), None)
        if done is None:
            raise Refused(f"pair {pair}: the record was not decided by a replay of the bundle {b['manifest_sha256'][:16]} this keep was cut from")
        undecided(row["commit"], done["bundle"])
    if awaiting and (row is None or not RB.awaits_replay(row.get("commit") or {})):
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
    """The Commit's replay arguments, with its Programs read from this row dir (their digests are checked against the record).  No
    serving rows: a Commit's window is committed at its C2 site (`c2_replay.serving_rows`), before any draw, never by its replay."""
    a = dict(ctx["args"])
    a["programs_root"] = o.row_dir if a.get("programs_root") else None
    a["program_dir"] = os.path.join(o.row_dir, "build_request") if a.get("program_dir") else None
    a["serving_rows"] = None
    if o.replay_workers is not None:
        a["replay_workers"] = o.replay_workers
    return SimpleNamespace(**a, replay_cache=False, vu_export_dir=None)


def worker_peak(o: Options) -> int:
    from verity_vllm.pipeline.telemetry.admission import CALIBRATION
    return int(o.worker_peak_gb * 2**30) if o.worker_peak_gb else int(CALIBRATION["commit.c2.replay_workers.small_vu"]["value"])


def engine_of_record(row: dict[str, Any], ctx: dict[str, Any], art: Path) -> None:
    """The replay's `engine_of_record` is the Commit engine's (read in its process, in the bundle); this process's beside it."""
    eng = ctx.get("engine_of_record")
    if eng is None:
        return
    doc = json.loads(art.read_text())
    for sr in (row["commit"]["sampled_replay"], doc["sampled_replay"]):
        sr["engine_of_replay"], sr["engine_of_record"] = sr.get("engine_of_record"), eng
    _write(art, json.dumps(doc, indent=1, default=str) + "\n")


def weights_attest(c: dict[str, Any], wc: dict[str, Any], consumed: set[str], words: dict[str, np.ndarray], extra: Any,
                   b: dict[str, Any]) -> dict[str, Any]:
    """What the replay's weights openings attest: ``attests`` is None unless every consumed field opened and the tree folds to the root."""
    pin = c.get("weights_pin") or {}
    whole = wc["equal"] and consumed <= words.keys()
    by = {s: 0 for s in ("checkpoint", "bundle")}
    for name in consumed & words.keys():
        by["bundle" if name in extra else "checkpoint"] += 1
    return {"pin": {"equal": pin.get("result"), "digest": pin.get("root_of_record")},
            "root_openings": f"{len(consumed & words.keys())}/{len(consumed)}", "sources": by,
            "attests": ATTESTS if whole else None, "replay": f"cpu, from {'the slim keep of ' if 'keep' in b else ''}bundle {b['manifest_sha256']}"}


def decide(row: dict[str, Any], ctx: dict[str, Any], wc: dict[str, Any], b: dict[str, Any], consumed: set[str],
           words: dict[str, np.ndarray], extra: Any) -> None:
    """The run's value correspondence and verdict with the replay on the record (in the in-process Commit's component order)."""
    from verity_vllm.check.cublas_route_pin import stamp
    from verity_vllm.check.verdict import commit_verdict

    c = row["commit"]
    stamp(c)
    vc = c["value_correspondence"]
    comps = vc["components"]
    for k in ("weights_pin", "replay"):
        if k in comps:
            comps[k] = comps.pop(k)
    fails = [(k, x) for k, x in comps.items() if x.get("result") is False]
    vc["why"] = "; ".join(f"{k}: {x.get('why')}" for k, x in fails) if fails else None
    c["weights_attest"] = weights_attest(c, wc, consumed, words, extra, b)
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


def slim_path(out: str | os.PathLike, pair: int) -> Path:
    return Path(out) / f"replay_slim_p{pair}"


def keep(out: Path, b: dict[str, Any], store: Any, run: Any, touched: SD.Touched, plan: dict[str, Any], c: dict[str, Any], *,
         opened: Iterable[str]) -> dict[str, Any] | None:
    """Writes the decided replay's slim keep (module docstring) and returns its `slim.json`; None, with nothing written, when the draw's
    reads were not all made in this process (`plan` is not a plan of as many picks as the replay picked).  ``opened``: the registered
    tensors the replay consumed and opened, the only ones whose salt keys and bundle bytes the keep's weights tree carries (`slim_tree`)."""
    ctx, bdir, pair = b["context"], Path(b["path"]), int(b["context"]["pair"])
    sr = c.get("sampled_replay") or {}
    picked = (sr.get("sample") or {}).get("picked")
    if not plan.get("plan") or picked is None or plan.get("picked") != picked or len(plan.get("picks") or ()) != picked:
        print(f"[replay] SLIM KEEP (pair {pair}): NOT WRITTEN -- the replay picked {picked}, its reads were re-made for {plan.get('picked')}",
              flush=True)
        return None
    final = slim_path(out, pair)
    p = final.with_name(final.name + ".partial")
    shutil.rmtree(p, ignore_errors=True)
    hashed: dict[str, str] = {}
    SD.dump(store, run, p / "store", digests=hashed, touched=touched)
    tree = slim_tree(json.loads((bdir / "weights" / "tree.json").read_text()), opened)
    for rel in ["binding_map.json", "context.json", *(f"weights/extra/{i}.bin" for i in sorted(tree["extra"].values()))]:
        (p / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bdir / rel, p / rel)
    (p / "weights").mkdir(exist_ok=True)
    (p / "weights" / "tree.json").write_text(json.dumps(tree, separators=(",", ":")) + "\n")
    names = sorted(x.relative_to(p).as_posix() for x in p.rglob("*") if x.is_file())
    with ThreadPoolExecutor(8) as pool:
        sums = pool.map(lambda r: hashed.get(r.removeprefix("store/")) or RB._sha256(p / r), names)
        files = {r: {"sha256": h, "bytes": (p / r).stat().st_size} for r, h in zip(names, sums)}
    doc = {"schema": SLIM_SCHEMA, "pair": pair, "run_root": run.run_root.hex(), "program_digest": (ctx.get("args") or {}).get("program_digest"),
           "programs": sr.get("programs"), "binding_map_sha256": b["manifest"]["files"]["binding_map.json"]["sha256"],
           "bundle_manifest_sha256": b["manifest_sha256"], "commit_pass": c.get("commit_pass"), "picked": picked, "picks_key": PICKS_KEY,
           "picks": plan["picks"], "touched_bytes": touched.nbytes(), "files": files, "bytes": sum(f["bytes"] for f in files.values()),
           "dump_sha256": hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    (p / SLIM_DOC).write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    if final.exists():
        shutil.rmtree(final)
    os.replace(p, final)
    print(f"[replay] SLIM KEEP (pair {pair}): {final} -- {picked} picks, {doc['bytes'] >> 20} MiB in {len(files)} files, dump sha256 "
          f"{doc['dump_sha256'][:16]}, run root {doc['run_root'][:16]}", flush=True)
    return doc


def kept_draw(s: C2.Site, slim: dict[str, Any]) -> None:
    """Refused unless this code draws the keep's picks from it, in order (`C2.plan`, read only).  A keep holds only its picks' reads,
    so another draw (a sampler that changed, or a record's `replay_draw` this code does not make) would stop on a node it lacks."""
    try:
        got, why = json.loads(json.dumps(C2.plan(s)["replay"].get("picks"))), None
    except KeyError as e:
        got, why = None, repr(e)[:200]
    want = slim["picks"]
    if got == want:
        return
    draw = getattr(s.args, "replay_draw", None)
    seen = (f"reads what the keep does not hold ({why})" if why else
            f"shares {len({tuple(p) for p in want} & {tuple(p) for p in got or ()})} of the keep's {len(want)} picks ({len(got or ())} drawn)")
    raise Refused(f"this code's draw from the keep {seen}" + (f"; the record draws by {draw!r}" if draw else ""))


def replay(o: Options) -> bool | None:
    """The bundle's C2 replay and the run's verdict on the record; the verdict's commit_pass."""
    from verity_vllm.check import weights_of_record as WR
    from verity_vllm.pipeline.timeline import Timeline

    t0 = time.perf_counter()
    tl = Timeline.from_env("commit_replay")
    b = open_source(o.bundle)
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
    args.replay_worker_peak = worker_peak(o)
    vc, consumed, reads = row["commit"]["value_correspondence"], set(), {}
    site = C2.Site(args=args, out=out, pair=pair, com=store, rc=run, row=row, reqs=ctx["requests"], sampling=ctx["sampling"], tokens=ctx["tokens"],
                   manifest=man, stop_of_record=ctx["stop_of_record"], ids_full=ctx["ids_full"], versions={"git_head": ctx.get("code")},
                   components=vc.setdefault("components", {}), vc_result=ctx.get("vc_result"), weights_of=lambda p: provider(words, p, consumed),
                   weights_check=lambda: dict(wc), tl=tl, reads=reads)
    if "keep" in b:
        kept_draw(site, b["keep"])
    with SD.touched(store) as touched:
        C2.run(site, {})
    engine_of_record(row, ctx, out / f"sampled_replay_p{pair}.json")
    decide(row, ctx, wc, b, consumed, words, tree.get("extra") or {})
    if "keep" not in b:
        with tl.span("replay.slim_keep", pair=pair):
            keep(out, b, store, run, touched, reads, row["commit"], opened=consumed & words.keys())
    record(out, rows, verdict, b, time.perf_counter() - t0)
    c = row["commit"]
    print(f"[replay] COMMIT {'PASS' if c['commit_pass'] else 'FAIL'} (pair {pair}): {c.get('commit_fail_reason') or 'every check on the record'}; "
          f"verdict commit_pass {verdict['commit_pass']}", flush=True)
    return verdict["commit_pass"]


def main(o: Options) -> int:
    from verity_vllm.check.replay.driver import PoolDoesNotFit

    warm_vml()
    try:
        ok = replay(o)
    except (Refused, PoolDoesNotFit) as e:
        print(f"[replay] REFUSED: {e}", flush=True)
        return 2
    sys.stdout.flush()
    return 0 if ok else 3
