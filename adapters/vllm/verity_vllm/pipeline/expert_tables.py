"""The committer's registration of the per-expert cut's tables (`expert_rows`), one a TP rank (``verity-vllm tp-register-tables``):
each table of each expert stack's weights and scales on the rank (`expert_rows.table_param`, its rows `expert_rows.table_words`
of the weight as `check.weights_of_record` composes it from the checkpoint's shards for the rank) committed as core's
``registered.Registered`` of ``hm96-sha512/row-seg/v1`` rows under the committer's secret key, at the rank's program (`program`)
and the table's nonce (`nonces`), written before the first window's coins as the record `records.expert_tables.ExpertTables`:
the ``{program, values}`` the verifier pins (``records_audit.py audit --registered``), so a window's table read and opening count
registered exactly where the window's root for the table is this one.  The key is minted for these nonces (`mint_key`, whose
sidecar `key_of` requires) and never leaves the committer; a window's committer commits the tables it serves under the same key at
the program and nonces the record names (`expert_rows.committer`, `serving_rows.after_commit`'s ``tables``), so a table served as
registered has its registered root and any other's reads are free.

A stack's tables are its e4m3 weight's blocks of ``blocks[proj]`` rows (n, a power of two dividing N) and its f32 scales' rows, as
the partition's read templates read them: a block of rows of ``H = n K / 2`` words, a scale row of ``H = 2 K / G``.  Without a Build,
a capture's served parameters name the weights (`served_fields`) and each composed weight is held to its served SHA-512, so the
registered rows are the served bytes; with one, that the registered rows are the checkpoint's is the committer's own check, as for
its weights (`window_weights`).

Outside the integration, this module is the per-expert cut's tables: with the registration, it exports the table layout
(`table_param`, `table_words`), the record (`ExpertTables`) and a window committer's registration (`committer`; `of_request`
reads it as a serving-rows request names it, by its file and its key file)."""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from verity_vllm.check import weights_of_record as WR
from verity_vllm.config import UsageError, option
from verity_vllm.pipeline.expert_rows import committer, of_request, table_param, table_words
from verity_vllm.records import expert_tables as XT
from verity_vllm.records.expert_tables import SCHEMA, ExpertTables

__all__ = ["NONCES", "SCHEMA", "TABLES_NONCE", "ExpertTables", "committer", "entries", "key_of", "mint_key", "nonces", "of_request",
           "program", "register", "served_fields", "table_param", "table_words", "tables_of"]

#: the salt nonces of the tables under the committer's key start here, a rank's 2^32 apart (`nonces`): above every weight
#: parameter's and the state's (`program.predict.window_weights.WEIGHTS_NONCE` and on)
TABLES_NONCE = 3 << 61
#: the nonce scheme `nonces` draws (a table's field's place in the source's weight-field order), which a key's sidecar names
#: (`mint_key`, `key_of`): a key whose tables were drawn at another scheme's nonces is refused
NONCES = "field-order"
BITS = {"float8_e4m3fn": 8, "uint8": 8, "bfloat16": 16, "float16": 16, "float32": 32}


def program(repo: str, revision: str, world: int, rank: int, prefix: str) -> str:
    """The program a rank's tables are registered under (their domains', core's ``registered.domain``): the SHA-512 of the model,
    its revision, the TP world and rank and the Programs' wrapper prefix, so two ranks' tables of one name are distinct values."""
    doc = {"format": SCHEMA, "repo": repo, "revision": revision, "world": int(world), "rank": int(rank), "prefix": prefix}
    return hashlib.sha512(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def tables_of(fields: Sequence[tuple[str, Sequence[int], int]], blocks: Mapping[str, int]) -> dict[str, list[tuple[int, str, str, int]]]:
    """The tables of the expert stacks among the weight ``fields`` (``(label, shape, bits)``, `weights_of_record.
    program_weight_fields`), by field, each ``(block, param, "w" | "ws", n)``: a stack's ``(E, N, K)`` e4m3 weight in N / n blocks of
    ``blocks[proj]`` rows, its ``(E, NB, K / G)`` f32 scales a row each.  Refused for a block that is not a power of two dividing
    N."""
    out: dict[str, list[tuple[int, str, str, int]]] = {}
    for label, shape, bits in fields:
        m = WR.EXPERTS_RE.match(label)
        if m is None or m.group(3) not in blocks or len(shape) != 3:
            continue
        E, N, K = (int(x) for x in shape)
        if bits == 8:
            n = int(blocks[m.group(3)])
            if n < 1 or n & (n - 1) or N % n:
                raise ValueError(f"{label}: weights {list(shape)} in blocks of {n} rows; a block is a power of two dividing N")
            out[label] = [(b, table_param(label, n * K // 2, b), "w", n) for b in range(N // n)]
        elif bits == 32:
            out[label] = [(b, table_param(label, 2 * K, b), "ws", 0) for b in range(N)]
    return out


def nonces(tables: Mapping[str, Sequence[tuple[int, str, str, int]]], order: Sequence[str], rank: int) -> dict[str, int]:
    """Each table's salt nonce under the committer's key: `TABLES_NONCE`, plus the rank's 2^32, its field's place in ``order``
    (every weight field of the rank, as its source lists them) times 2^18, plus its block.  A table's nonce is its own whatever is
    registered beside it, so no two tables, ranks or weights take one salt, and a registration of some stacks holds the whole
    registration's entries for them."""
    at = {f: i for i, f in enumerate(order)}
    out: dict[str, int] = {}
    for label, ts in tables.items():
        for b, param, _w, _n in ts:
            if at[label] >> 14 or b >> 18:
                raise ValueError(f"{param}: field {at[label]} or block {b} past the nonce's 14 and 18 bits")
            out[param] = TABLES_NONCE + (int(rank) << 32) + (at[label] << 18) + b
    return out


def _minted(key: bytes) -> dict[str, str]:
    """What a key's sidecar (``KEY.nonces``) holds: the nonce scheme its tables are drawn at (`NONCES`) and the key it was minted
    with, as SHA-512 under a tag, so a sidecar copied beside another key names none."""
    return {"nonces": NONCES, "key": hashlib.sha512(b"verity-vllm expert tables committer key\0" + key).hexdigest()}


def mint_key(path: str) -> bytes:
    """A fresh committer's key at ``path``, 32 OS bytes (mode 0600), with its sidecar ``path.nonces`` (`_minted`, mode 0600):
    the one key `key_of` reads.  Refused where either file exists: a key is minted once."""
    side = Path(f"{path}.nonces")
    if Path(path).exists() or side.exists():
        raise UsageError(f"--new-key: {path} or {side.name} exists; a committer's key is minted once, so give --key a new path")
    key = os.urandom(32)
    for p, data in ((path, key), (side, (json.dumps(_minted(key)) + "\n").encode())):
        with os.fdopen(os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as f:
            f.write(data)
    return key


def key_of(path: str) -> bytes:
    """The committer's key at ``path``, refused unless `mint_key` minted it for this nonce scheme, as its sidecar says.  Tables
    committed under the key at another scheme's nonces (by hand, or before a nonce was its field's place) can hold a nonce this
    scheme draws for another table, and then two tables draw one salt."""
    key, side = Path(path).read_bytes(), Path(f"{path}.nonces")
    if not side.is_file() or json.loads(side.read_text()) != _minted(key):
        raise UsageError(f"--key {path}: not minted for {NONCES!r} nonces (no {side.name} beside it naming them and this key): its "
                         "tables may repeat a nonce these tables draw, two tables under one salt; run with --new-key and a new "
                         "--key path")
    return key


def _prefix(prog: dict[str, Any], ck: Any) -> str:
    """The Programs' wrapper prefix, as `weights_of_record` finds it over all the Program's weight fields."""
    from verity_vllm.query.manifest import format as MF

    inst, res = prog["instances"], prog["result"]
    fields = WR.program_weight_fields(inst)
    named = MF.wrapper_prefix_of(inst.get("rows", []), res.get("attention_impls_observed", []))
    cands = sorted({".".join(f.split(".")[:k]) + "." for f, _s, _b in fields for k in (1, 2) if f.count(".") >= k} | {"", named})
    return max(cands, key=lambda p: (sum(1 for f, _s, _b in fields if ck.has(WR.strip_prefix(f, p))), p == named, -len(p)))


def entries(digest: str, words: np.ndarray, shape: Sequence[int], tables: Sequence[tuple[int, str, str, int]], key: bytes,
            nonce: Mapping[str, int]) -> dict[str, dict[str, Any]]:
    """The registered entries of one field's ``tables`` (`tables_of`), its words ``words`` of ``shape`` as composed for the rank,
    each table committed at the rank's program ``digest`` and its ``nonce`` under ``key``."""
    from verity.protocols.spot.verifier.one_stage import registered as G
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
             world: int | None = None, rank: int | None = None, stacks: Sequence[str] | None = None,
             checkpoints_manifest: str | None = None, model: str | None = None, revision: str | None = None,
             verify_shards: bool = True, workers: int = 1) -> dict[str, Any]:
    """The rank's registration (module docstring) of the tables of the Program at ``program_dir`` (a rank Program of the Build:
    its weights struct and its TP world and rank), its expert weights composed from ``checkpoint_dir`` (pinned by
    ``checkpoints_manifest``) under ``key``, each projection's weight in blocks of ``blocks`` rows; ``stacks``: only those expert
    stacks' tables (`weights_of_record.EXPERTS_RE`'s container, ``model.layers.5.mlp.experts``); ``workers``: processes, a stack's
    fields each.  Without a Program, ``served``: a capture's served parameters at TP ``world`` and ``rank`` (`served_fields`), the
    tables named as vLLM names the weights and each composed weight refused unless its SHA-512 is the served one."""
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
    tp = prog["result"].get("tp", {})
    world, rank = int(tp.get("world", 1)), int(tp.get("rank", 0))
    stack_of = {f: m.group(1) for f, _s, _b in all_fields if (m := WR.EXPERTS_RE.match(f)) is not None}
    if unknown := sorted(set(blocks) - {WR.EXPERTS_RE.match(f).group(3) for f in stack_of}):
        raise ValueError(f"--block {unknown}: no expert stack has that projection")
    if missing := sorted(set(stacks or ()) - set(stack_of.values())):
        raise ValueError(f"no expert stack {missing[:3]} among the weights")
    manifest = checkpoints_manifest or WR._default_manifest()
    ckargs = {"checkpoints_manifest": manifest, "model": model, "revision": revision}
    ck = WR.CheckpointIndex(checkpoint_dir, **ckargs, verify_shards=verify_shards)
    if verify_shards and (bad := sorted(fn for fn, s in ck.pin["shards"].items() if s.get("pinned") is not True)):
        raise ValueError(f"{checkpoint_dir}: {len(bad)} of {len(ck.shards)} shards not pinned by the checkpoints.json entry, e.g. {bad[0]}")
    entry = manifest_entry(manifest, checkpoint_dir, model, revision) if manifest else None
    repo = model or (entry["repo"] if entry is not None else None)
    rev = revision or (entry["revision"] if entry is not None else None)
    if not repo or not rev:
        raise ValueError(f"{checkpoint_dir}: no checkpoints.json entry names its repo and revision; give --model and --revision")
    prefix = "" if served is not None else _prefix(prog, ck)
    fields = [f for f in all_fields if f[0] in stack_of and (stacks is None or stack_of[f[0]] in stacks)]
    tables = tables_of(fields, blocks)
    if not tables:
        raise ValueError(f"no e4m3 expert weights of {sorted(blocks)} among the weights")
    digest = program(repo, rev, world, rank, prefix)
    nonce = nonces(tables, [f for f, _s, _b in all_fields], rank)
    by_stack: dict[str, list[tuple[str, list[int], int]]] = {}
    for f in fields:
        if f[0] in tables:
            by_stack.setdefault(stack_of[f[0]], []).append(f)
    jobs = [(prog, {"ckpt_dir": checkpoint_dir, **ckargs, "verify_shards": False}, prefix, fs, {f[0]: tables[f[0]] for f in fs}, digest,
             key, nonce, {f[0]: sha[f[0]] for f in fs if f[0] in sha}) for fs in by_stack.values()]
    values: dict[str, dict[str, Any]] = {}
    why: list[str] = []
    with ProcessPoolExecutor(workers) if workers > 1 else nullcontext() as ex:
        for got, unresolved in (ex.map if ex else map)(_commit, jobs):
            values, why = values | got, why + unresolved
    if missing := sorted(set(nonce) - set(values)):
        raise ValueError(f"{len(missing)} tables not composed from the checkpoint, e.g. {missing[0]}; weights_of_record left "
                         f"{len(why)} fields unresolved: {why[:2]}")
    return XT.ExpertTables(program=digest, repo=repo, revision=rev, world=world, rank=rank, prefix=prefix, blocks=blocks, values=values,
                           nonces=nonce).dump()


@dataclass(frozen=True, kw_only=True)
class Options:
    """verity-vllm tp-register-tables --program-dir D --checkpoint-dir C --key K --block w13=1 --block w2=32 --out F [--stacks S,..]:
    the committer's registration of a TP rank's per-expert tables (`register`), written to ``F`` for the verifier to pin.  Without
    a Build: ``--served params.json --world W --rank R`` in place of ``--program-dir``."""
    program_dir: str | None = option("--program-dir", default=None, help="a rank Program of the Build (its weights struct, TP world and rank)")
    served: str | None = option("--served", default=None, help="a capture's params.json for the rank, in place of a Program")
    world: int | None = option("--world", type=int, default=None, help="with --served: the TP world")
    rank: int | None = option("--rank", type=int, default=None, help="with --served: the TP rank")
    checkpoint_dir: str = option("--checkpoint-dir", help="the checkpoint of record")
    key: str = option("--key", help="the committer's 32-byte secret key file (the one its windows commit the tables under), minted "
                      "with --new-key: refused without its sidecar KEY.nonces")
    block: list[str] = option("--block", action="append", default=[], help="PROJ=n: the stacks' PROJ weights in blocks of n rows")
    out: str = option("--out", help="the registration's file")
    stacks: str | None = option("--stacks", default=None, help="comma-separated expert stacks, e.g. model.layers.5.mlp.experts "
                                "(default: every one)")
    checkpoints_manifest: str | None = option("--checkpoints-manifest", default=None, help="manifests/checkpoints.json (shard pins)")
    model: str | None = option("--model", default=None, help="the checkpoint's repo (its checkpoints.json entry)")
    revision: str | None = option("--revision", default=None, help="the checkpoint's revision")
    no_verify_shards: bool = option("--no-verify-shards", default=False, action="store_true", help="skip the shard sha256")
    new_key: bool = option("--new-key", default=False, action="store_true",
                           help="mint --key first: 32 fresh OS bytes and its sidecar KEY.nonces (mode 0600; mint_key)")
    workers: int = option("--workers", type=int, default=1, help="processes, one a stack at a time")


def main(o: Options) -> int:
    blocks = {}
    for b in o.block:
        proj, _, n = b.partition("=")
        if not proj or not n.isdigit():
            raise UsageError(f"--block {b!r}: PROJ=n")
        blocks[proj] = int(n)
    if not blocks:
        raise UsageError("give each projection's block rows: --block w13=n --block w2=n")
    if (o.program_dir is None) == (o.served is None) or (o.served is not None and (o.world is None or o.rank is None)):
        raise UsageError("give --program-dir, or --served with --world and --rank")
    key = mint_key(o.key) if o.new_key else key_of(o.key)
    doc = register(o.program_dir, o.checkpoint_dir, key, blocks, served=o.served, world=o.world, rank=o.rank,
                   stacks=o.stacks.split(",") if o.stacks else None, checkpoints_manifest=o.checkpoints_manifest,
                   model=o.model, revision=o.revision, verify_shards=not o.no_verify_shards, workers=o.workers)
    Path(o.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(f"[tp-register-tables] rank {doc['rank']}/{doc['world']}: {len(doc['values'])} tables under program {doc['program'][:16]} "
          f"-> {o.out}", flush=True)
    return 0
