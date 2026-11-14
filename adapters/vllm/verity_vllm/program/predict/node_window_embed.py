"""The vocabulary shard's embedding (`EmbeddingShard_v1{VS, H, START}`) lowered into the node window (`node_window`) as reads
of padded tables: per column block b of `WIDTH` words, one `EmbeddingRead_v1{V, H=WIDTH}` unit a member, keyed by the token as
it is, over block b's table, a registered parameter of its own of V rows of `WIDTH` words, the shard's rows at its START and
zero rows elsewhere (`shard_param`). A token the shard owns reads its row; any other token below V reads the zero row, as the
shard gives zeros (`EmbeddingRead_v1` gives 0x7FFF only for a token past its table, which is why the table is padded to the
whole vocabulary). The all-reduce reads block b's ``o0`` by wire as its column block b of the embedding's partial.

The width is one, fixed at registration: a table's salts are its field's place and its block (`expert_tables.nonces`), so two
widths of one field would take the same salts, and `shard_param` and the lowering refuse any width but `WIDTH`. The window
declares the tables as parameters of the rank's registration and commits nothing: the prover commits them once a process.

The tables are not weights: `window_weights.weight_nonces` numbers every name it is handed, so a table among them moves the
plain weights' nonces and their roots stop being the registration's. `split` keeps the window's tables
(``/b<b>/vocab``, ``/b<b>/table``) and the absorption weights' entry (``/bnk``) apart from its plain weights; they go to
the tables' registration (`expert_tables`, its nonces from `expert_tables.TABLES_NONCE`), the rest to `window_weights`.
A batched GEMM's ``w`` reads its ``/bnk`` entry by a registered read (`node_window_sparse.entry`), so none of its rows is free.

`SHARD_VOCAB`, `shard_vocab`, `shard_param` and `unit_templates` belong in `serving_templates` and live here until it holds
them. Refused by name (`Unexpressible`): a shard of no registered vocabulary, one past its vocabulary, an H that isn't whole
blocks of `WIDTH`, a window whose norm grain cuts H at another width, and a table of its weight already declared at another
width.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from verity.circuits.codec import descriptor_id
from verity.commitments.gates.embedding import read
from verity.commitments.windows import interface as IF
from verity_vllm.check.weights_of_record import weight_param
from verity_vllm.pipeline import serving_templates as ST
from verity_vllm.query.program_view import spec_base, spec_statics

from .build import Unexpressible
from .node_window import Lowering, NodeWindow, Piece, _cols, _family, _Free
from .window_builder import WEIGHTS, Row

SHARD = "EmbeddingShard_v1"
#: an ``EmbeddingShard_v1`` row's whole vocabulary V, by its shard's VS: its statics carry VS and START, not V
SHARD_VOCAB: dict[int, int] = {19360: 154880}
#: the one row width of a shard's padded tables, the norms' apply width (`node_window._cols` of H = 6144 within 256)
WIDTH = 192
_TABLE = re.compile(rf"^{re.escape(WEIGHTS)}/(?P<label>.+)@(?P<words>\d+)/b(?P<block>\d+)/(?P<kind>vocab|table)$")
#: the absorption weights' entry of the rank's registration beside the tables: B·N rows of K words, ``weights/<label>@<K>/bnk``
_ENTRY = re.compile(rf"^{re.escape(WEIGHTS)}/(?P<label>.+)@(?P<words>\d+)/bnk$")


def shard_vocab(spec: str) -> int:
    """The vocabulary V of the ``EmbeddingShard_v1`` row ``spec``: what its padded tables' rows (and the read's key) range over."""
    VS = int(spec_statics(spec)["VS"])
    if VS not in SHARD_VOCAB:
        raise Unexpressible(f"{spec}: a shard of {VS} rows of no registered vocabulary ({sorted(SHARD_VOCAB)})")
    return SHARD_VOCAB[VS]


def shard_param(label: str, words: int, block: int) -> str:
    """The registered parameter of column block ``block`` of the embedding shard ``label`` as a table of V rows of ``words`` u16
    words, the shard's rows at its START and zero rows elsewhere: ``weights/<label>@<words>/b<block>/vocab``. Refused at any
    width but `WIDTH`."""
    if int(words) != WIDTH:
        raise Unexpressible(f"{label}: a shard's table at {words} words a row; its one width is {WIDTH}")
    return f"{weight_param(label, words)}/b{int(block)}/vocab"


def unit_templates(spec: str) -> tuple[tuple[str, int], ...]:
    """`serving_templates.unit_templates` of ``spec``, and for an ``EmbeddingShard_v1`` row the read of a block of its padded
    tables, ``EmbeddingRead_v1{V, H=WIDTH}``, H / `WIDTH` of them."""
    hit = ST.unit_templates(spec)
    if spec_base(spec) == SHARD:
        H = int(spec_statics(spec)["H"])
        hit += ((descriptor_id(read(shard_vocab(spec), WIDTH)), H // WIDTH),)
    return hit


def table_words(columns: np.ndarray, start: int, V: int) -> np.ndarray:
    """A block's padded table, ``(V, WIDTH)`` u16 words: ``columns`` (the shard's VS rows of the block's `WIDTH` columns) at
    rows [start, start + VS), zero rows elsewhere. What the prover commits for `shard_param` (`expert_tables` registers it)."""
    a = np.ascontiguousarray(columns, np.uint16)
    if a.ndim != 2 or a.shape[1] != WIDTH or not 0 <= start <= V - a.shape[0]:
        raise ValueError(f"a shard's block of {a.shape} words at row {start}: {WIDTH} columns within {V} rows")
    out: np.ndarray = np.zeros((V, WIDTH), np.uint16)
    out[start:start + a.shape[0]] = a
    return out


def split(params: Iterable[str]) -> tuple[list[str], list[str]]:
    """The weight parameters among ``params`` apart from the tables, sorted: ``(plain, tables)``, the tables an embedding
    shard's padded ones (``/b<b>/vocab``), the expert blocks' (``/b<b>/table``) and the absorption weights' entry
    (``/bnk``, `node_window_sparse.entry`'s). Only ``plain`` goes to `window_weights`."""
    ws = sorted(p for p in params if p.startswith(f"{WEIGHTS}/"))
    kept = [bool(_TABLE.match(p) or _ENTRY.match(p)) for p in ws]
    return [p for p, k in zip(ws, kept) if not k], [p for p, k in zip(ws, kept) if k]


def tables_of(params: Iterable[str]) -> dict[str, list[tuple[int, str, str, int]]]:
    """The shards' padded tables among ``params``, by weight, each ``(block, param, "vocab", 0)``: the form the tables'
    registration numbers them in (`expert_tables.nonces`). Only the ``/vocab`` tables: not the expert blocks' or a ``/bnk``
    entry, which `split` gives beside them."""
    out: dict[str, list[tuple[int, str, str, int]]] = {}
    for p in split(params)[1]:
        m = _TABLE.match(p)
        if m is not None and m["kind"] == "vocab":
            out.setdefault(m["label"], []).append((int(m["block"]), p, "vocab", 0))
    return {k: sorted(v) for k, v in out.items()}


def _one_width(params: Mapping[str, int], label: str) -> None:
    """Refuse a table of ``label`` already declared at a width other than `WIDTH`."""
    if other := sorted({int(m["words"]) for p in params if (m := _TABLE.match(p)) and m["kind"] == "vocab"
                        and m["label"] == label} - {WIDTH}):
        raise Unexpressible(f"{label}: its tables declared at {other} words a row and at {WIDTH}; a field has one width")


def _reads(w: NodeWindow, tid: str, at: int, units: int, lo: int, label: str, V: int, block: int) -> None:
    """Units ``[at, at + units)`` of the read ``tid`` read column block ``block`` of the shard ``label`` (from word ``lo`` of the
    weights record) as its padded table (`shard_param`): its root and domain id table reads, and its row, salt and path an
    opening at the key ``tok`` that the verdict ``o1`` holds."""
    param = shard_param(label, WIDTH, block)
    w.b.params[param] = V * WIDTH
    w.b.tables.setdefault((lo, WIDTH, block), IF.Table(param, param, 0, WIDTH, V))
    w.b.roots += [IF.TableRead(tid, "root", param, "root", at, units), IF.TableRead(tid, "domain", param, "domain", at, units)]
    w.b.openings.append(IF.TableOpening(tid, param, "tok", ("row", "salt", "path"), "o1", at, units))


def embedding_shard(w: NodeWindow, j: int, node: Any) -> None:
    """`EmbeddingShard_v1{VS, H, START}` as H / `WIDTH` column blocks of `EmbeddingRead_v1{V, WIDTH}` (module docstring), block
    b of member t at unit ``a + b M + t``, each keyed by member t's token over block b's padded table; the node's output is
    block b's ``o0`` at its columns."""
    spec, fam, M = descriptor_id(node.fn), _family(node), w.members(node)
    VS, H, START = (int(node.fn.bindings[x]) for x in ("VS", "H", "START"))
    V = shard_vocab(spec)
    if START < 0 or START + VS > V:
        raise Unexpressible(f"{spec}: rows [{START}, {START + VS}) past its vocabulary of {V}")
    if H % WIDTH or _cols(H, w.b.grain.norm) != WIDTH:
        raise Unexpressible(f"{spec}: H = {H} in blocks of {_cols(H, w.b.grain.norm)}; a shard's tables are {WIDTH} words a row")
    A, lo, label = H // WIDTH, w.weight(node.args[1].refs, VS * H), w.label(node.args[1].refs, VS * H)
    _one_width(w.b.params, label)
    fn = ST.template(next(t for t, _n in unit_templates(spec) if spec_base(t) == ST.READ)).fn
    t, a = w.unit(fam, fn, A * M)
    tok = w.take(node, 0)
    w.feed(t, "tok", a, tok if isinstance(tok, _Free) else [tok[i % M] for i in range(A * M)], (A, M))
    for b in range(A):
        _reads(w, t, a + b * M, M, lo, label, V, b)
    w.out(j, node, [Piece(0, H, Row(t, a, "o0", WIDTH), M, 1)])


#: the embedding shard's lowering by family (the node's Definition's name), beside `node_window.FAMILIES`
FAMILIES: dict[str, Lowering] = {"EmbeddingShard": embedding_shard}

__all__ = ["FAMILIES", "SHARD_VOCAB", "WIDTH", "embedding_shard", "shard_param", "shard_vocab", "split", "table_words",
           "tables_of", "unit_templates"]
