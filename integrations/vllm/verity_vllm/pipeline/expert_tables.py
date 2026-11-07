"""The committer's registration of the per-expert cut's tables (`expert_rows`), one a TP rank (``verity-vllm tp-register-tables``):
each table of each MoE layer's expert weights and scales on the rank (`expert_rows.table_param`, its rows `expert_rows.table_words`
of the weight as `check.weights_of_record` composes it from the checkpoint's shards for the rank) committed as core's
``registered.Registered`` of ``hm96-sha512/row-seg/v1`` rows under the committer's secret key, at the rank's program (`program`)
and the table's nonce (`nonces`), written before the first window's coins.  The record is ``{"format", "program", "repo",
"revision", "world", "rank", "prefix", "blocks", "values": {param: entry}}``, each entry core's ``Registered.entry()``: the
``{program, values}`` the verifier pins (``records_audit.py audit --registered``), so a window's table read and opening count
registered exactly where the window's root for the table is this one.  The key never leaves the committer; a window's committer
commits the tables it serves under the same key at the same program and nonces (`committer`, `serving_rows.after_commit`'s
``tables``), so a table served as registered has its registered root and any other's reads are free.

A projection's tables are its weight's blocks of ``blocks[w13|w2]`` rows (n, each a power of two dividing G and N) and its scales'
G-blocks, as the partition's read templates read them: a block of rows of ``H = n K / 2`` words, a scale row of ``H = 2 K / G``.
That the registered rows are the checkpoint's is the committer's own check, as for its weights (`window_weights`)."""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from verity_vllm.check import weights_of_record as WR
from verity_vllm.config import UsageError, option
from verity_vllm.pipeline.expert_rows import table_param, table_words

__all__ = ["FORMAT", "TABLES_NONCE", "committer", "entries", "nonces", "of_request", "program", "register", "served_fields",
           "tables_of"]

FORMAT = "verity-vllm/expert-tables/v1"
#: the salt nonces of the tables under the committer's key start here, a rank's 2^32 apart (`nonces`): above every weight
#: parameter's and the state's (`program.predict.window_weights.WEIGHTS_NONCE` and on)
TABLES_NONCE = 3 << 61
LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
PARAM_RE = re.compile(r"^weights/(.+)@\d+/b(\d+)/table$")
BITS = {"float8_e4m3fn": 8, "uint8": 8, "bfloat16": 16, "float16": 16, "float32": 32}
KINDS = {("w13", "weight"): 0, ("w13", "weight_scale_inv"): 1, ("w2", "weight"): 2, ("w2", "weight_scale_inv"): 3}


def program(repo: str, revision: str, world: int, rank: int, prefix: str) -> str:
    """The program a rank's tables are registered under (their domains', core's ``registered.domain``): the SHA-512 of the model,
    its revision, the TP world and rank and the Programs' wrapper prefix, so two ranks' tables of one name are distinct values."""
    doc = {"format": FORMAT, "repo": repo, "revision": revision, "world": int(world), "rank": int(rank), "prefix": prefix}
    return hashlib.sha512(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def tables_of(fields: Sequence[tuple[str, Sequence[int], int]], blocks: Mapping[str, int]) -> dict[str, list[tuple[int, str, str, int]]]:
    """The tables of the expert weights among a Program's weight ``fields`` (``(label, shape, bits)``, `weights_of_record.
    program_weight_fields`), by field: each ``(block, param, "w" | "ws", n)``, a weight ``(E, N, K)`` of e4m3 bytes in N / n blocks of
    ``blocks[w13|w2]`` rows, its scales ``(E, NB, K / G)`` of f32 words in NB G-blocks.  Refused for a block that is not a power of
    two dividing N."""
    out: dict[str, list[tuple[int, str, str, int]]] = {}
    for label, shape, bits in fields:
        m = WR.EXPERTS_RE.match(label)
        if m is None or m.group(4) not in ("weight", "weight_scale_inv") or m.group(3) not in blocks:
            continue
        E, N, K = (int(x) for x in shape)
        if m.group(4) == "weight":
            n = int(blocks[m.group(3)])
            if bits != 8 or n < 1 or n & (n - 1) or N % n:
                raise ValueError(f"{label}: {bits}-bit weights {list(shape)} in blocks of {n} rows; a block is a power of two "
                                 "dividing N rows of e4m3 bytes")
            out[label] = [(b, table_param(label, n * K // 2, b), "w", n) for b in range(N // n)]
        else:
            if bits != 32:
                raise ValueError(f"{label}: {bits}-bit scales, not f32 words")
            out[label] = [(b, table_param(label, 2 * K, b), "ws", 0) for b in range(N)]
    return out


def nonces(params: Sequence[str], rank: int) -> dict[str, int]:
    """Each table's salt nonce under the committer's key: `TABLES_NONCE` plus the rank's 2^32, its layer's 2^20, its projection's
    weight or scales' 2^18 (`KINDS`) and its block.  A table's nonce is its own whatever is registered beside it, so no two tables,
    ranks or weights take one salt, and a registration of some layers holds the whole registration's entries for them."""
    out: dict[str, int] = {}
    for p in params:
        m = PARAM_RE.match(p)
        e, L = (WR.EXPERTS_RE.match(m.group(1)), LAYER_RE.search(m.group(1))) if m else (None, None)
        kind = KINDS.get((e.group(3), e.group(4))) if e else None
        if m is None or L is None or kind is None or int(L.group(1)) >> 12 or int(m.group(2)) >> 18:
            raise ValueError(f"{p}: not a table of a layer's expert weights or scales (layer below 2^12, block below 2^18)")
        out[p] = TABLES_NONCE + (int(rank) << 32) + (int(L.group(1)) << 20) + (kind << 18) + int(m.group(2))
    if len(set(out.values())) != len(out):
        raise ValueError("two registered tables of one layer, projection and block")
    return out


def committer(doc: Mapping[str, Any], key: bytes) -> dict[str, Any]:
    """A window's committer's tables (`serving_rows.after_commit`'s ``tables``) from its registration ``doc``: its ``key``, the
    rank's program and each table's nonce (`nonces`)."""
    if doc.get("format") != FORMAT:
        raise ValueError(f"an expert tables registration is {FORMAT}, not {doc.get('format')!r}")
    if not isinstance(key, bytes) or len(key) != 32:
        raise ValueError("a table's salts are drawn under the committer's 32-byte key")
    return {"key": key, "program": doc["program"], "nonces": nonces(list(doc["values"]), int(doc["rank"]))}


def of_request(spec: Mapping[str, Any] | None, rank: int = 0) -> dict[str, Any] | None:
    """`committer` of a serving-rows request's ``tables``, ``{"key": FILE, "registration": FILE}`` (``{rank}`` in the registration's
    path its rank's), or None without one."""
    if not spec:
        return None
    doc = json.loads(Path(str(spec["registration"]).format(rank=rank)).read_text())
    if int(doc.get("rank", -1)) != int(rank):
        raise ValueError(f"{spec['registration']}: the tables registration of rank {doc.get('rank')}, not of rank {rank}")
    return committer(doc, Path(spec["key"]).read_bytes())


def _prefix(prog: dict[str, Any], ck: Any) -> str:
    """The Programs' wrapper prefix, as `weights_of_record` finds it over all the Program's weight fields."""
    from verity_vllm.query.manifest import format as MF

    inst, res = prog["instances"], prog.get("result") or {}
    fields = WR.program_weight_fields(inst)
    named = MF.wrapper_prefix_of(inst.get("rows", []), res.get("attention_impls_observed") or [])
    cands = sorted({".".join(f.split(".")[:k]) + "." for f, _s, _b in fields for k in (1, 2) if f.count(".") >= k} | {"", named})
    return max(cands, key=lambda p: (sum(1 for f, _s, _b in fields if ck.has(WR.strip_prefix(f, p))), p == named, -len(p)))


def entries(digest: str, words: np.ndarray, shape: Sequence[int], tables: Sequence[tuple[int, str, str, int]], key: bytes,
            nonce: Mapping[str, int]) -> dict[str, dict[str, Any]]:
    """The registered entries of one field's ``tables`` (`tables_of`), its words ``words`` of ``shape`` as composed for the rank,
    each table committed at the rank's program ``digest`` and its ``nonce`` under ``key``."""
    from verity.core.protocols.sampled_proofs.one_stage import registered as G
    from verity_vllm.protocols.pouw.h1 import registered as H1R

    E, N, K = (int(x) for x in shape)
    return {param: H1R.commit(digest, param, table_words(words, what, E, N, K, 1, n, b), key, nonce[param], schema=G.SCHEMA_SEG).entry()
            for b, param, what, n in tables}


def served_fields(path: str) -> tuple[list[tuple[str, list[int], int]], dict[str, str]]:
    """A capture's served parameters, ``params.json``'s ``{name: {shape, dtype, sha512}}`` (or a ``commit.json``'s ``opened``), as
    weight fields named as vLLM names them, ``(name, shape, bits)``, and each one's SHA-512 over its served bytes."""
    doc = json.loads(Path(path).read_text())
    fields, digests = [], {}
    for name, e in sorted(doc.get("opened", doc).items()):
        if (bits := BITS.get(str(e.get("dtype")).removeprefix("torch."))) is not None:
            fields.append((name, [int(x) for x in e["shape"]], bits))
            digests[name] = str(e["sha512"])
    return fields, digests


def _commit(job: tuple[dict[str, Any], dict[str, Any], str, list[tuple[str, list[int], int]], dict[str, list], str, bytes,
                       dict[str, int], dict[str, str]]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """One worker's tables: its fields composed as `weights_of_record` composes them for the rank, each held to its served SHA-512
    where one is given and each table committed; and why each field it left unresolved was."""
    prog, ckargs, prefix, fields, tables, digest, key, nonce, served = job
    shapes, out = {f: s for f, s, _b in fields}, {}

    def on_field(label: str, words: np.ndarray) -> None:
        want = served.get(label)
        if want and hashlib.sha512(np.ascontiguousarray(words).reshape(-1).view(np.uint8)).hexdigest() != want:
            raise ValueError(f"{label}: the checkpoint composed for the rank is not the served tensor (SHA-512 {want[:16]}..)")
        out.update(entries(digest, words, shapes[label], tables[label], key, nonce))

    rec = WR.weights_of_record(prog, WR.CheckpointIndex(**ckargs), prefix=prefix, fields=fields, on_field=on_field)
    return out, [f"{f['field']}: {f.get('why')}" for f in rec["fields"] if not f.get("digest")]


def register(program_dir: str | None, checkpoint_dir: str, key: bytes, blocks: Mapping[str, int], *, served: str | None = None,
             world: int | None = None, rank: int | None = None, layers: Sequence[int] | None = None,
             checkpoints_manifest: str | None = None, model: str | None = None, revision: str | None = None,
             verify_shards: bool = True, workers: int = 1) -> dict[str, Any]:
    """The rank's registration (module docstring) of the tables of the Program at ``program_dir`` (a rank Program of the Build:
    its weights struct and its TP world and rank), its expert weights composed from ``checkpoint_dir`` (pinned by
    ``checkpoints_manifest``) under ``key``, each projection's weight in blocks of ``blocks`` rows; ``layers``: only those
    layers' tables; ``workers``: processes, a layer's fields each.  Without a Program, ``served``: a capture's served
    parameters at TP ``world`` and ``rank`` (`served_fields`), the tables named as vLLM names the weights and each composed
    weight refused unless its SHA-512 is the served one."""
    from verity_vllm.check.checkpoint_index import manifest_entry
    from verity_vllm.check.component_programs import load_program_head

    if (program_dir is None) == (served is None) or (served is not None and (world is None or rank is None)):
        raise ValueError("a rank Program dir, or a capture's served parameters with their TP world and rank")
    sha: dict[str, str] = {}
    if served is not None:
        prog = {"instances": {"params": [], "rows": []}, "result": {"tp": {"world": int(world), "rank": int(rank)}}, "dir": None}
        all_fields, sha = served_fields(served)
    else:
        prog = load_program_head(program_dir)
        all_fields = WR.program_weight_fields(prog["instances"])
    tp = prog["result"].get("tp") or {}
    world, rank = int(tp.get("world") or 1), int(tp.get("rank") or 0)
    manifest = checkpoints_manifest or WR._default_manifest()
    ckargs = {"checkpoints_manifest": manifest, "model": model, "revision": revision}
    ck = WR.CheckpointIndex(checkpoint_dir, **ckargs, verify_shards=verify_shards)
    entry = (manifest_entry(manifest, checkpoint_dir, model, revision) if manifest else None) or {}
    repo, rev = model or entry.get("repo"), revision or entry.get("revision")
    if not repo or not rev:
        raise ValueError(f"{checkpoint_dir}: no checkpoints.json entry names its repo and revision; give --model and --revision")
    prefix = "" if served is not None else _prefix(prog, ck)
    fields = [f for f in all_fields if layers is None or ((m := LAYER_RE.search(f[0])) is not None and int(m.group(1)) in set(layers))]
    tables = tables_of(fields, blocks)
    if not tables:
        raise ValueError(f"no expert weights of {sorted(blocks)} among the weights" + (f" of layers {list(layers)}" if layers else ""))
    digest = program(repo, rev, world, rank, prefix)
    nonce = nonces([p for ts in tables.values() for _b, p, _w, _n in ts], rank)
    by_layer: dict[str, list[tuple[str, list[int], int]]] = {}
    for f in fields:
        if f[0] in tables:
            by_layer.setdefault((m := LAYER_RE.search(f[0])) and m.group(1) or "", []).append(f)
    jobs = [(prog, {"ckpt_dir": checkpoint_dir, **ckargs, "verify_shards": False}, prefix, fs, {f[0]: tables[f[0]] for f in fs}, digest,
             key, nonce, {f[0]: sha[f[0]] for f in fs if f[0] in sha}) for fs in by_layer.values()]
    values: dict[str, dict[str, Any]] = {}
    why: list[str] = []
    with ProcessPoolExecutor(workers) if workers > 1 else nullcontext() as ex:
        for got, unresolved in (ex.map if ex else map)(_commit, jobs):
            values, why = values | got, why + unresolved
    if missing := sorted(set(nonce) - set(values)):
        raise ValueError(f"{len(missing)} tables not composed from the checkpoint, e.g. {missing[0]}; weights_of_record left "
                         f"{len(why)} fields unresolved: {why[:2]}")
    return {"format": FORMAT, "program": digest, "repo": repo, "revision": rev, "world": world, "rank": rank, "prefix": prefix,
            "blocks": dict(sorted(blocks.items())), "values": dict(sorted(values.items()))}


@dataclass(frozen=True, kw_only=True)
class Options:
    """verity-vllm tp-register-tables --program-dir D --checkpoint-dir C --key K --block w13=1 --block w2=32 --out F [--layers 3,40]:
    the committer's registration of a TP rank's per-expert tables (`register`), written to ``F`` for the verifier to pin.  Without
    a Build: ``--served params.json --world W --rank R`` in place of ``--program-dir``."""
    program_dir: str | None = option("--program-dir", default=None, help="a rank Program of the Build (its weights struct, TP world and rank)")
    served: str | None = option("--served", default=None, help="a capture's params.json for the rank, in place of a Program")
    world: int | None = option("--world", type=int, default=None, help="with --served: the TP world")
    rank: int | None = option("--rank", type=int, default=None, help="with --served: the TP rank")
    checkpoint_dir: str = option("--checkpoint-dir", help="the checkpoint of record")
    key: str = option("--key", help="the committer's 32-byte secret key file (the one its windows commit the tables under)")
    block: list[str] = option("--block", action="append", default=[], help="PROJ=n: w13's or w2's weight in blocks of n rows")
    out: str = option("--out", help="the registration's file")
    layers: str | None = option("--layers", default=None, help="comma-separated layers (default: every MoE layer)")
    checkpoints_manifest: str | None = option("--checkpoints-manifest", default=None, help="manifests/checkpoints.json (shard pins)")
    model: str | None = option("--model", default=None, help="the checkpoint's repo (its checkpoints.json entry)")
    revision: str | None = option("--revision", default=None, help="the checkpoint's revision")
    no_verify_shards: bool = option("--no-verify-shards", default=False, action="store_true", help="skip the shard sha256")
    new_key: bool = option("--new-key", default=False, action="store_true", help="write 32 fresh OS bytes to --key first (mode 0600)")
    workers: int = option("--workers", type=int, default=1, help="processes, one a layer at a time")


def main(o: Options) -> int:
    blocks = {}
    for b in o.block:
        proj, _, n = b.partition("=")
        if proj not in ("w13", "w2") or not n.isdigit():
            raise UsageError(f"--block {b!r}: PROJ=n, PROJ w13 or w2")
        blocks[proj] = int(n)
    if not blocks:
        raise UsageError("give each projection's block rows: --block w13=n --block w2=n")
    if (o.program_dir is None) == (o.served is None) or (o.served is not None and (o.world is None or o.rank is None)):
        raise UsageError("give --program-dir, or --served with --world and --rank")
    if o.new_key:
        if Path(o.key).exists():
            raise UsageError(f"--new-key: {o.key} exists; a committer's key is written once")
        fd = os.open(o.key, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(os.urandom(32))
    doc = register(o.program_dir, o.checkpoint_dir, Path(o.key).read_bytes(), blocks, served=o.served, world=o.world, rank=o.rank,
                   layers=[int(x) for x in o.layers.split(",")] if o.layers else None, checkpoints_manifest=o.checkpoints_manifest,
                   model=o.model, revision=o.revision, verify_shards=not o.no_verify_shards, workers=o.workers)
    Path(o.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(f"[tp-register-tables] rank {doc['rank']}/{doc['world']}: {len(doc['values'])} tables under program {doc['program'][:16]} "
          f"-> {o.out}", flush=True)
    return 0
