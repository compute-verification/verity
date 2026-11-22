"""A sealed replay bundle's record (`of_record`) and its weights, opened against the committed weights root
(`open_weights`, `registered_roles`, `OnRead`).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from verity.commitments import salt_file as SF

from verity_vllm.commit import step_rows
from verity_vllm.config import option
from verity_vllm.errors import Refused
from verity_vllm.log import init_logger
from verity_vllm.pipeline import replay_bundle as RB

# The checks this command reads (the weights of record and their rewrites, the route pin, the verdict, the driver's refusal), the
# partition, the admission's calibration and the timeline are imported inside the functions that read them: the route pin's and the
# driver's imports put the catalog's Definitions into the Definition table, which this module's import is to leave as it found it.

__all__ = ["WORDS", "OnRead", "Options", "Refused", "manifest_of", "of_record", "open_weights", "opener", "provider",
           "registered_roles", "replay_args", "salts_path", "tensor_root_of", "tensor_salts_of", "tree_refusal"]

logger = init_logger(__name__)

#: registered dtype -> (the Program's value bits, the words the replay reads): the widths the live provider serves
WORDS = {"bfloat16": (16, np.uint16), "float32": (32, np.uint32), "float8_e4m3fn": (8, np.uint8), "float8_e4m3fnuz": (8, np.uint8),
         "uint8": (8, np.uint8), "int8": (8, np.uint8), "int32": (32, np.uint32)}


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
                                        help="the forked pool's workers (default: the Commit's own; the sample does not depend on it)")


def undecided(c: dict[str, Any], bundle: str) -> None:
    """A decided run's commit record back in its deferred state, in place: what the decided replay wrote (its record, the
    linkage's opened values, its decision) taken off, and the deferred records of ``bundle`` (its path of record) put back."""
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
        components = vc.get("components")
        if components:  # a correspondence that holds no component has none of the replay's to take off
            for k in ("sampled_replay", "boundary_linkage", "negatives"):
                components.pop(k, None)
        vc["method"] = re.sub(r"^sampled-exact-replay\([^)]*\) \+ boundary-linkage \+ ", "", vc.get("method") or "")
        vc["result"] = None


def bundle_tensors(tree: dict[str, Any]) -> dict[str, int]:
    """The registered tensors whose bytes the bundle holds, each by its index (the tree's ``extra``); none for a tree that lists none:
    every tensor of such a tree is read from the checkpoint."""
    extra = tree.get("extra")
    return extra if extra else {}


def deferred_entries(verdict: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    """The entries of ``kind`` a Commit's verdict lists under ``replay_deferred`` (``bundles``: each sealed bundle; ``replayed``: each
    replay that decided a run); none for a verdict that deferred no replay or lists none of the kind."""
    deferred = verdict.get("replay_deferred")
    entries = deferred.get(kind) if deferred else None
    return entries if entries else []


def tree_scheme(tree: dict[str, Any]) -> str | None:
    """The scheme a weights tree was registered under: its ``scheme``, None for a tree recorded before trees named one."""
    return tree.get("scheme")


def root_scheme(root_of_record: str | None) -> str | None:
    """The scheme a committed weights root of record was registered under, by its size: rows-sha512 for 64 bytes, else None."""
    return step_rows.NAME if len(root_of_record or "") // 2 == step_rows.ROOT_BYTES else None


def salts_path(bundle: Path) -> Path:
    """The salt file of a bundle's weights registration, in its private part (`replay_bundle.private_file`), or of a slim keep's:
    ``weights/tree.salts`` beside ``weights/tree.json``."""
    return RB.private_file(bundle, RB.PRIVATE[1]) or SF.beside(Path(bundle) / "weights" / "tree.json")


def tensor_salts_of(bundle: Path, tree: dict[str, Any], i: int) -> bytes | None:
    """Tensor ``i``'s row salts from the registration's salt file (`salts_path`); None when it holds none for the tensor (a slim keep
    holds those of the tensors its replay opened alone)."""
    p, name = salts_path(bundle), tree["names"][i][0]
    if not int(tree["names"][i][3]):
        return b""
    if not (p / SF.INDEX).is_file() or name not in SF.params(p):
        return None
    return SF.read(p, name)


def tensor_root_of(bundle: Path, tree: dict[str, Any], i: int, raw: np.ndarray) -> bytes:
    """Tensor ``i``'s root under its salts (`tensor_salts_of`); KeyError where the salt file holds none for it."""
    ys = tensor_salts_of(bundle, tree, i)
    if ys is None:
        raise KeyError(f"registered tensor {i} ({tree['names'][i][0]}): the salt file holds no salts for it")
    return step_rows.tensor_root(bytes.fromhex(tree["geo_digest"]), i, tree["names"][i], memoryview(raw).cast("B"), ys)


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
    src = "bundle" if name in bundle_tensors(tree) else "checkpoint"
    if tensor_salts_of(bundle, tree, i) is None:
        return src, None, "the salt file holds no salts for it", False
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
    if tensor_root_of(bundle, tree, i, raw) != bytes.fromhex(tree["tensor_roots"][i]):
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


def rewrites(tree: dict[str, Any], roles: dict[str, dict[str, Any]] | None, ck: Any, tp: dict[str, int] | None) -> Any:
    """`fp8_ue8m0.Rewrites` over the tree's servable tensors: the fields of ``roles`` that vLLM rewrites at load, each read from
    the checkpoint as `weights_of_record.compose` composes it (a TP rank's shard under ``tp``).  None without roles or a checkpoint."""
    from verity_vllm.check import fp8_ue8m0
    from verity_vllm.check import weights_of_record as WR

    if not roles or ck is None:
        return None
    dims = {n: (list(s), WORDS[dt][0]) for n, dt, s, _b in tree["names"] if dt in WORDS}
    at = {"world": int(tp["world"]), "rank": int(tp["rank"])} if tp else {}

    def composed(name: str) -> tuple[np.ndarray, dict[str, Any]]:
        shape, bits = dims[name]
        return WR.compose(name, shape, bits, ck, **at)

    return fp8_ue8m0.Rewrites(list(dims), roles, composed, ck.config)


def opener(bundle: Path, tree: dict[str, Any], ck: Any, *, tp: dict[str, int] | None = None,
           roles: dict[str, dict[str, Any]] | None = None) -> Callable[[int], tuple[str, np.ndarray | None, str | None, bool]]:
    """`_open_one` of the tree's tensor i, as `open_weights` opens it: one tensor at a time, the rewrites of ``roles`` made once."""
    rw = rewrites(tree, roles, ck, tp)
    return lambda i: _open_one(bundle, tree, i, ck, tp, rw, roles)


def tree_refusal(tree: dict[str, Any], root_of_record: str | None) -> tuple[str, str | None]:
    """(the root the tree's tensor roots fold to, why the tree is not the committed weights root's tree: None when it is).  Only a
    rows-sha512 tree under a rows-sha512 root of record (`root_scheme`) is read: a tree of another scheme, or one recorded before trees
    named theirs, is refused by name before anything opens, and so is a root of record of another size."""
    have, want = tree_scheme(tree), root_scheme(root_of_record)
    if "key" in tree or "tensor_keys" in tree:
        return "", "the bundle's weights tree was registered under a weights key: only a registration with a salt file is read"
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
                 roles: dict[str, dict[str, Any]] | None = None) -> None:
        names = tree["names"]
        _folded, refused = tree_refusal(tree, root_of_record)
        if refused:
            raise ValueError(refused)
        rw = rewrites(tree, roles, ck, tp)
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


def of_record(b: dict[str, Any], out: Path, *, awaiting: bool = True) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], Any, Any]:
    """(runs.jsonl rows, the deferred run's row, verdict.json, store, run) once the bundle is shown to be this record's; else Refused.
    ``awaiting=False``: the run's record may be decided already (a reader of the bundle that decides nothing)."""
    ctx, bdir, pair = b["context"], Path(b["path"]), int(b["context"]["pair"])
    verdict = json.loads((out / "verdict.json").read_text())
    ent = next((x for x in deferred_entries(verdict, "bundles") if int(x["pair"]) == pair), None)
    if ent is None or ent.get("manifest_sha256") != b["manifest_sha256"]:
        raise Refused(f"the Commit's verdict names no bundle of pair {pair} with manifest sha256 {b['manifest_sha256'][:16]}")
    rows = [json.loads(ln) for ln in (out / "runs.jsonl").read_text().splitlines() if ln.strip()]
    row = next((r for r in rows if r["pair"] == pair and r["arm"] != "control"), None)
    if not awaiting:
        if row is None:
            raise Refused(f"pair {pair}: no instrumented run in {out / 'runs.jsonl'}")
    elif "keep" in b and row is not None and not RB.awaits_replay(RB.commit_of(row)):
        done = next((x for x in deferred_entries(verdict, "replayed")
                     if int(x["pair"]) == pair and x.get("manifest_sha256") == b["manifest_sha256"]), None)
        if done is None:
            raise Refused(f"pair {pair}: the record was not decided by a replay of the bundle {b['manifest_sha256'][:16]} this keep was cut from")
        undecided(row["commit"], done["bundle"])
    if row is None or (awaiting and not RB.awaits_replay(RB.commit_of(row))):
        raise Refused(f"pair {pair}: no instrumented run awaiting its deferred replay in {out / 'runs.jsonl'}")
    store, run = RB.load_store(bdir)
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
    serving rows: a Commit's window is committed before sampling, never by its replay."""
    a = dict(ctx["args"])
    a["programs_root"] = o.row_dir if a.get("programs_root") else None
    a["program_dir"] = os.path.join(o.row_dir, "build_request") if a.get("program_dir") else None
    a["serving_rows"] = None
    if o.replay_workers is not None:
        a["replay_workers"] = o.replay_workers
    return SimpleNamespace(**a, replay_cache=False, vu_export_dir=None)
