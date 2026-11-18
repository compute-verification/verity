"""The committer's registration of the per-expert cut's tables (`expert_rows`), one a TP rank (``verity-vllm tp-register-tables``):
each table of each expert stack's weights and scales on the rank (`expert_rows.table_param`, its rows `expert_rows.table_words`
of the weight as `check.weights_of_record` composes it from the checkpoint's shards for the rank) committed as core's
``registered.Registered`` of ``hm96-sha512/row-seg/v1`` rows under fresh salts at the rank's program (`program`), written before
the first window's coins as the record `records.expert_tables.ExpertTables`: the ``{program, values}`` the verifier pins
(``records_audit.py audit --registered``), so a window's table read and opening count registered exactly where the window's root
for the table is this one.  The salts go to the salt file beside the record (`verity.commitments.salt_file`) and never leave the
committer; a window's committer commits the tables it serves under those salts at the program the record names
(`expert_rows.committer`, `serving_rows.after_commit`'s ``tables``), so a table served as registered has its registered root, and a
window serving any other is refused, publishing none of its tables.

A stack's tables are its e4m3 weight's blocks of ``blocks[proj]`` rows (n, a power of two dividing N) and its f32 scales' rows, as
the partition's read templates read them: a block of rows of ``H = n K / 2`` words, a scale row of ``H = 2 K / G``.  The rank's
embedding shard's tables (``shards``, ``--shard FIELD``) are its column blocks at its one width (`serving_templates.shard_block`,
which no argument sets), each the whole vocabulary's rows (`expert_rows.shard_words`).  A ``(B, K, N)`` weight a GEMM reads as
[B][N][K] (``bnk``, ``--bnk FIELD``) is one value in that order, ``hm96-sha512/row/v2`` rows of K words a read reads by
reference (`expert_rows.bnk_words`).  Without a Build,
a capture's served parameters name the weights (`served_fields`) and each composed weight is held to its served SHA-512, so the
registered rows are the served bytes; with one, that the registered rows are the checkpoint's is the committer's own check, as for
its weights (`window_weights`).

Outside the integration, this module is the per-expert cut's tables: with the registration, it exports the table layout
(`table_param`, `table_words`), the record (`ExpertTables`) and a window committer's registration (`committer`; `of_request`
reads it as a serving-rows request names it, by its file)."""
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
from verity.commitments import salt_file as SF

from verity_vllm.check import weights_of_record as WR
from verity_vllm.config import UsageError, option
from verity_vllm.pipeline.expert_rows import (
    bnk_param,
    bnk_words,
    committer,
    of_request,
    shard_param,
    shard_words,
    table_param,
    table_words,
)
from verity_vllm.pipeline.serving_templates import shard_block
from verity_vllm.records import expert_tables as XT
from verity_vllm.records.expert_tables import SCHEMA, ExpertTables

__all__ = ["SCHEMA", "ExpertTables", "bnk_fields", "bnk_param", "bnk_words", "committer", "entries", "of_request", "program",
           "register", "served_fields", "shard_param", "shard_words", "table_param", "table_words", "tables_of"]

BITS = {"float8_e4m3fn": 8, "uint8": 8, "bfloat16": 16, "float16": 16, "float32": 32}


def program(repo: str, revision: str, world: int, rank: int, prefix: str) -> str:
    """The program a rank's tables are registered under (their domains', core's ``registered.domain``): the SHA-512 of the model,
    its revision, the TP world and rank and the Programs' wrapper prefix, so two ranks' tables of one name are distinct values."""
    doc = {"format": SCHEMA, "repo": repo, "revision": revision, "world": int(world), "rank": int(rank), "prefix": prefix}
    return hashlib.sha512(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def tables_of(fields: Sequence[tuple[str, Sequence[int], int]], blocks: Mapping[str, int],
              shards: Sequence[str] = (), bnk: Sequence[str] = ()) -> dict[str, list[tuple[int, str, str, int]]]:
    """The tables of the expert stacks among the weight ``fields`` (``(label, shape, bits)``, `weights_of_record.
    program_weight_fields`), by field, each ``(block, param, "w" | "ws", n)``: a stack's ``(E, N, K)`` e4m3 weight in N / n blocks of
    ``blocks[proj]`` rows, its ``(E, NB, K / G)`` f32 scales a row each; and of each embedding shard ``shards`` names, each
    ``(block, param, "vocab", W)``: its ``(VS, H)`` bf16 words in H / W column blocks of its one width W (`shard_block`); and of
    each weight ``bnk`` names, ``(0, param, "bnk", K)``: its ``(B, K, N)`` bf16 words in (B, N, K) order.  Refused for a block
    that is not a power of two dividing N, a shard that is not two dimensions of 16-bit words and a ``bnk`` weight that is not
    three."""
    out: dict[str, list[tuple[int, str, str, int]]] = {}
    for label, shape, bits in fields:
        if label in bnk:
            if bits != 16 or len(shape) != 3:
                raise ValueError(f"{label}: a weight registered in (B, N, K) order is (B, K, N) 16-bit words, not {list(shape)} "
                                 f"of {bits} bits")
            out[label] = [(0, bnk_param(label, int(shape[1])), "bnk", int(shape[1]))]
            continue
        if label in shards:
            if bits != 16 or len(shape) != 2:
                raise ValueError(f"{label}: an embedding shard is (VS, H) 16-bit words, not {list(shape)} of {bits} bits")
            W = shard_block(int(shape[1]))
            out[label] = [(b, shard_param(label, W, b), "vocab", W) for b in range(int(shape[1]) // W)]
            continue
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


def _prefix(prog: dict[str, Any], ck: Any) -> str:
    """The Programs' wrapper prefix, as `weights_of_record` finds it over all the Program's weight fields."""
    from verity_vllm.query.manifest import format as MF

    inst, res = prog["instances"], prog["result"]
    fields = WR.program_weight_fields(inst)
    named = MF.wrapper_prefix_of(inst.get("rows", []), res.get("attention_impls_observed", []))
    cands = sorted({".".join(f.split(".")[:k]) + "." for f, _s, _b in fields for k in (1, 2) if f.count(".") >= k} | {"", named})
    return max(cands, key=lambda p: (sum(1 for f, _s, _b in fields if ck.has(WR.strip_prefix(f, p))), p == named, -len(p)))


def entries(digest: str, words: np.ndarray, shape: Sequence[int], tables: Sequence[tuple[int, str, str, int]],
            vocab: tuple[int, int] = (0, 0)) -> tuple[dict[str, dict[str, Any]], dict[str, bytes]]:
    """The registered entries of one field's ``tables`` (`tables_of`), its words ``words`` of ``shape`` as composed for the rank,
    each table committed at the rank's program ``digest`` under fresh salts, and those salts (192 bytes a row, in row order); an
    embedding shard's over ``vocab``, the whole vocabulary's rows and the rank's first token."""
    from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_SEG, SCHEMA_HM96_SHA512_ROW_V2
    from verity.protocols.resource_accounting.compute.pouw.prover.serving import (
        registered_commit,
    )

    def rows(b: int, what: str, n: int) -> np.ndarray:
        if what == "vocab":
            return shard_words(np.asarray(words).reshape([int(x) for x in shape]), vocab[0], vocab[1], b)
        if what == "bnk":
            return bnk_words(np.asarray(words).reshape([int(x) for x in shape]))
        E, N, K = (int(x) for x in shape)
        return table_words(words, what, E, N, K, 1, n, b)

    out, salts = {}, {}
    for b, param, what, n in tables:
        words_of = rows(b, what, n)
        ys = os.urandom(len(words_of) * SF.SALT_BYTES)
        v = registered_commit(digest, param, words_of, bytes(32), 0, row_salts=ys,
                              schema=SCHEMA_HM96_SHA512_ROW_V2 if what == "bnk" else SCHEMA_HM96_SHA512_ROW_SEG)
        out[param], salts[param] = v.entry(), ys
    return out, salts


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


def _commit(job: tuple[dict[str, Any], dict[str, Any], str, list[tuple[str, list[int], int]], dict[str, list], str,
                       dict[str, str], dict[str, tuple[int, int]]]) -> tuple[dict[str, dict[str, Any]], dict[str, bytes], list[str]]:
    """One worker's tables: its fields composed as `weights_of_record` composes them for the rank, each held to its served SHA-512
    where one is given and each table committed; their salts; and why each field it left unresolved was."""
    prog, ckargs, prefix, fields, tables, digest, served, vocab = job
    shapes, out, salts = {f: s for f, s, _b in fields}, {}, {}

    def on_field(label: str, words: np.ndarray) -> None:
        want = served.get(label)
        if want and hashlib.sha512(np.ascontiguousarray(words).reshape(-1).view(np.uint8)).hexdigest() != want:
            raise ValueError(f"{label}: the checkpoint composed for the rank is not the served tensor (SHA-512 {want[:16]}..)")
        got, ys = entries(digest, words, shapes[label], tables[label], vocab.get(label, (0, 0)))
        out.update(got)
        salts.update(ys)

    rec = WR.weights_of_record(prog, WR.CheckpointIndex(**ckargs), prefix=prefix, fields=fields, on_field=on_field)
    return out, salts, [f"{f['field']}: {f.get('why')}" for f in rec["fields"] if not f.get("digest")]


def register(program_dir: str | None, checkpoint_dir: str, blocks: Mapping[str, int], *, salts_to: str | Path,
             served: str | None = None,
             world: int | None = None, rank: int | None = None, stacks: Sequence[str] | None = None,
             checkpoints_manifest: str | None = None, model: str | None = None, revision: str | None = None,
             verify_shards: bool = True, workers: int = 1, shards: Sequence[str] = (), bnk: Sequence[str] = ()) -> dict[str, Any]:
    """The rank's registration (module docstring) of the tables of the Program at ``program_dir`` (a rank Program of the Build:
    its weights struct and its TP world and rank), its expert weights composed from ``checkpoint_dir`` (pinned by
    ``checkpoints_manifest``) under fresh salts, written to the salt file ``salts_to``, each projection's weight in blocks of ``blocks`` rows; ``stacks``: only those expert
    stacks' tables (`weights_of_record.EXPERTS_RE`'s container, ``model.layers.5.mlp.experts``); ``shards``: the rank's embedding
    shards' tables too, each field the rank's even share of the checkpoint's rows; ``bnk``: those weights in (B, N, K) order
    too; ``workers``: processes, a stack's
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
    if missing := sorted({*shards, *bnk} - {f for f, _s, _b in all_fields}):
        raise ValueError(f"--shard or --bnk {missing[:3]}: no weight of that name")
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
    fields = [f for f in all_fields
              if f[0] in shards or f[0] in bnk or f[0] in stack_of and (stacks is None or stack_of[f[0]] in stacks)]
    tables = tables_of(fields, blocks, shards, bnk)
    if not tables:
        raise ValueError(f"no e4m3 expert weights of {sorted(blocks)}, no embedding shard and no (B, N, K) weight among the weights")
    vocab: dict[str, tuple[int, int]] = {}
    for name, shape, _b in fields:
        if name in shards:
            src = WR.strip_prefix(name, prefix)
            V = int(ck.tensors[src][2][0]) if ck.has(src) else 0
            if int(shape[0]) * world != V:
                raise ValueError(f"--shard {name}: its {shape[0]} rows are not rank {rank}/{world}'s even share of the checkpoint's "
                                 f"{V or 'missing'} rows of {src}")
            vocab[name] = (V, rank * int(shape[0]))
    digest = program(repo, rev, world, rank, prefix)
    by_stack: dict[str, list[tuple[str, list[int], int]]] = {}
    for f in fields:
        if f[0] in tables:
            by_stack.setdefault(stack_of.get(f[0], f[0]), []).append(f)
    jobs = [(prog, {"ckpt_dir": checkpoint_dir, **ckargs, "verify_shards": False}, prefix, fs, {f[0]: tables[f[0]] for f in fs}, digest,
             {f[0]: sha[f[0]] for f in fs if f[0] in sha}, {f[0]: vocab[f[0]] for f in fs if f[0] in vocab})
            for fs in by_stack.values()]
    values: dict[str, dict[str, Any]] = {}
    salts: dict[str, bytes] = {}
    why: list[str] = []
    with ProcessPoolExecutor(workers) if workers > 1 else nullcontext() as ex:
        for got, ys, unresolved in (ex.map if ex else map)(_commit, jobs):
            values, salts, why = values | got, salts | ys, why + unresolved
    if missing := sorted({param for ts in tables.values() for _b, param, _w, _n in ts} - set(values)):
        raise ValueError(f"{len(missing)} tables not composed from the checkpoint, e.g. {missing[0]}; weights_of_record left "
                         f"{len(why)} fields unresolved: {why[:2]}")
    SF.write(salts_to, salts)
    return XT.ExpertTables(program=digest, repo=repo, revision=rev, world=world, rank=rank, prefix=prefix, blocks=blocks,
                           values=values).dump()


def bnk_fields(program_dir: Path, rank: int) -> tuple[str, ...]:
    """The rank's ``W_UK_T`` and ``W_UV`` fields, as ``weights/<label>@<K>``, for ``tp-register-tables --bnk``."""
    raise NotImplementedError("vllm/compiler-wires-period-pieces: no /bnk value is registered on the served path yet")


@dataclass(frozen=True, kw_only=True)
class Options:
    """verity-vllm tp-register-tables --program-dir D --checkpoint-dir C --block w13=1 --block w2=32 --out F [--stacks S,..]:
    the committer's registration of a TP rank's per-expert tables (`register`), written to ``F`` for the verifier to pin, its
    salts to the salt file beside it, which never leaves the committer.  Without
    a Build: ``--served params.json --world W --rank R`` in place of ``--program-dir``."""
    program_dir: str | None = option("--program-dir", default=None, help="a rank Program of the Build (its weights struct, TP world and rank)")
    served: str | None = option("--served", default=None, help="a capture's params.json for the rank, in place of a Program")
    world: int | None = option("--world", type=int, default=None, help="with --served: the TP world")
    rank: int | None = option("--rank", type=int, default=None, help="with --served: the TP rank")
    checkpoint_dir: str = option("--checkpoint-dir", help="the checkpoint of record")
    block: list[str] = option("--block", action="append", default=[], help="PROJ=n: the stacks' PROJ weights in blocks of n rows")
    shard: list[str] = option("--shard", action="append", default=[], help="FIELD: the rank's embedding shard FIELD's tables too, a "
                              "column block each at the shard's one width (no option sets it)")
    bnk: list[str] = option("--bnk", action="append", default=[], help="FIELD: the (B, K, N) weight FIELD in (B, N, K) order too")
    out: str = option("--out", help="the registration's file")
    stacks: str | None = option("--stacks", default=None, help="comma-separated expert stacks, e.g. model.layers.5.mlp.experts "
                                "(default: every one)")
    checkpoints_manifest: str | None = option("--checkpoints-manifest", default=None, help="manifests/checkpoints.json (shard pins)")
    model: str | None = option("--model", default=None, help="the checkpoint's repo (its checkpoints.json entry)")
    revision: str | None = option("--revision", default=None, help="the checkpoint's revision")
    no_verify_shards: bool = option("--no-verify-shards", default=False, action="store_true", help="skip the shard sha256")
    workers: int = option("--workers", type=int, default=1, help="processes, one a stack at a time")


def main(o: Options) -> int:
    blocks = {}
    for b in o.block:
        proj, _, n = b.partition("=")
        if not proj or not n.isdecimal():
            raise UsageError(f"--block {b!r}: PROJ=n")
        blocks[proj] = int(n)
    if not blocks and not o.shard and not o.bnk:
        raise UsageError("give each projection's block rows (--block w13=n --block w2=n), an embedding shard (--shard FIELD) or a "
                         "weight in (B, N, K) order (--bnk FIELD)")
    if (o.program_dir is None) == (o.served is None) or (o.served is not None and (o.world is None or o.rank is None)):
        raise UsageError("give --program-dir, or --served with --world and --rank")
    doc = register(o.program_dir, o.checkpoint_dir, blocks, salts_to=SF.beside(o.out), served=o.served, world=o.world, rank=o.rank,
                   stacks=o.stacks.split(",") if o.stacks else None, checkpoints_manifest=o.checkpoints_manifest,
                   model=o.model, revision=o.revision, verify_shards=not o.no_verify_shards, workers=o.workers, shards=o.shard,
                   bnk=o.bnk)
    Path(o.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(f"[tp-register-tables] rank {doc['rank']}/{doc['world']}: {len(doc['values'])} tables under program {doc['program'][:16]} "
          f"-> {o.out}", flush=True)
    return 0
