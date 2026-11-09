"""The committer's history tables: DSA's long rows in the table-read form read each of their TOPK history rows from a table of
the request's latent rows at the layer (`verity.ml.commitments.history`), one table per (request, layer), under the
registration's switch `SWITCH`.

Table ``(request, layer)``'s leaf i is the ``hm96-sha512/row-seg/v1`` string (``b || c``) of the latent row the request's own
``DsaLatentRow_v1`` unit outputs at position i at that layer, under that row's own salt; every leaf past its length is the
placeholder's. Its domain is `history.domain` at the serving run's program (``program``), its own name (`name`, the
request's and the layer's, which a window's `interface.History` declares it by), `WORDS` words and the capacity
`history.CAPACITY`, so its tree's shape says nothing of the request's length. A window
appends its own rows in position order from the table's length (`Histories.append`): a request open at a window's edge keeps
its tables, and the next window appends from that edge. The window's registration carries each table's root after the
window's appends (`Histories.entries`), fixed before the window's coins; the verifier recomputes it from the writers'
public strings and never takes the prover's. The window commits each appended row under the table's salt and string
(`Histories.salts`, `Histories.strings`; `ml.windows.rows.commit`'s ``salts_of`` and ``strings_of``), so the leaf is the
string the window commits and no row is hashed twice.

Retention is per request: a table keeps every row and salt it holds, `ROW_BYTES` a row a layer beside its tree, until its
request ends (`Histories.end`), since any later window's read may open any earlier position; a Commit's hooks release
their window's requests once it is committed, but those its request names ``open`` (`release`). Rows and salts are private:
they stay in this process on the node and in the windows' private files, never under a run's output.

The switch is refused with prefix caching on (`refusal`): a cache hit's leaf would be another request's row, bound only by a
memory-checked pool, and no hit is derived by trusted Python.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from verity.ml.commitments import history as HT
from verity.ml.commitments import hm96
from verity_vllm.program.predict.reads import PREFIX_CACHE

#: the registration's engine switch the table-read form is under, off by default
SWITCH = "dsa_table_reads"
#: a latent row's words: ``DsaLatentRow_v1{KVL=512, R=64}``'s [kv_c | k_pe], bf16
WORDS = 576
#: what a table retains a row: its words' bytes and its salt
ROW_BYTES = 2 * WORDS + hm96.SHA512.salt_bytes


def on(config: Mapping[str, Any] | None) -> bool:
    """Whether the registration's engine config ``config`` turns the table-read form on."""
    return bool((config or {}).get(SWITCH, False))


def refusal(config: Mapping[str, Any] | None) -> str | None:
    """Why ``config`` can't turn `SWITCH` on, or None (off, or on without prefix caching)."""
    if on(config) and (config or {}).get(PREFIX_CACHE):
        return (f"{SWITCH} with {PREFIX_CACHE} on: a cache hit's history leaf would be another request's latent row, which only "
                f"a memory-checked pool binds, and no hit is derived by trusted Python; serve with {PREFIX_CACHE} off")
    return None


def name(request: str, layer: int) -> str:
    """Request ``request``'s history table at layer ``layer``: the table that layer's long rows read, a registered parameter
    of its own."""
    return f"history/{request}/model.layers.{int(layer)}.self_attn.latent"


class Histories:
    """The committer's open history tables across windows, by (request, layer): each a `history.HistoryTable` of `WORDS`-word
    rows at the serving run's ``program`` under its `name`, appended window by window from its edge and kept until its request
    ends."""

    def __init__(self, program: str, words: int = WORDS, capacity: int = HT.CAPACITY) -> None:
        if not isinstance(program, str) or not program:
            raise ValueError("a history table's domain binds the serving run's program: give it")
        self.program, self.words, self.capacity = program, int(words), int(capacity)
        self.tables: dict[tuple[str, int], HT.HistoryTable] = {}

    def table(self, request: str, layer: int) -> HT.HistoryTable:
        """Request ``request``'s table at layer ``layer``, empty until its first append."""
        key = (str(request), int(layer))
        if key not in self.tables:
            self.tables[key] = HT.HistoryTable(self.program, name(*key), self.words, self.capacity)
        return self.tables[key]

    def edge(self, request: str, layer: int) -> int:
        """The position the next append of (``request``, ``layer``) starts at: its table's length."""
        t = self.tables.get((str(request), int(layer)))
        return 0 if t is None else len(t.rows)

    def append(self, request: str, layer: int, start: int, rows: Any, salts: Sequence[bytes] | None = None) -> range:
        """Rows ``rows`` (``(k, words)`` u16) at positions ``start`` to ``start + k`` of (``request``, ``layer``)'s table, each
        under its salt (fresh by default), refused unless ``start`` is the table's edge: positions are appended in order, once
        each. -> their positions."""
        t = self.table(request, layer)
        if int(start) != len(t.rows):
            raise ValueError(f"history {t.value} of request {request}: appends at its edge {len(t.rows)}, not at {start} (a table "
                             f"is append-only in position order)")
        return t.append(rows, salts)

    def strings(self, request: str, layer: int, positions: Iterable[int]) -> bytes:
        """The table's leaves' strings (``b || c``) at ``positions``, concatenated: the window's committed strings of those rows."""
        t = self.tables[(str(request), int(layer))]
        return b"".join(bytes(t.bcs[p]) for p in positions)

    def salts(self, request: str, layer: int, positions: Iterable[int]) -> bytes:
        """The rows' salts at ``positions``, concatenated."""
        t = self.tables[(str(request), int(layer))]
        return b"".join(bytes(t.salts[p]) for p in positions)

    def entries(self, keys: Iterable[tuple[str, int]] | None = None) -> dict[tuple[str, int], dict[str, Any]]:
        """Each table's registration entry (`history.HistoryTable.entry`: its root after the appends so far), by (request,
        layer); ``keys`` the window's, every open table's by default."""
        return {k: self.tables[k].entry() for k in (self.tables if keys is None else keys)}

    def requests(self) -> list[str]:
        """The requests with an open table."""
        return sorted({r for r, _l in self.tables})

    def retained(self, request: str | None = None) -> int:
        """The rows' and salts' bytes held, of ``request``'s tables or of every one."""
        return sum(len(t.rows) for (r, _l), t in self.tables.items() if request is None or r == request) * ROW_BYTES

    def end(self, request: str) -> int:
        """Drop ``request``'s tables once it has ended, so their rows and salts are no longer held. -> the bytes released."""
        held = self.retained(request)
        for k in [k for k in self.tables if k[0] == str(request)]:
            del self.tables[k]
        return held


#: the serving process's open tables, by the run's committer program, kept across the run's windows
_OPEN: dict[str, Histories] = {}


def of_request(committer: Mapping[str, Any] | None, on: bool) -> Histories | None:
    """The serving process's tables of the run whose committer is ``committer`` (a window request's: its table registration's
    ``program``), kept across the run's windows, or None for a window of no history tables (``on`` off)."""
    if not on:
        return None
    program = (committer or {}).get("program")
    if not isinstance(program, str) or not program:
        raise ValueError("a window of history tables takes its tables' program from its request's committer: give it")
    return _OPEN.setdefault(program, Histories(program))


def release(hs: Histories | None, open_requests: Iterable[str] = ()) -> int:
    """End every request of ``hs`` but ``open_requests``, once a window is committed or refused: a Commit serves its requests to
    their finish before its window commits, so its window request names none open, and a window that cuts a request names it,
    whose tables the next window appends to. -> the bytes released."""
    if hs is None:
        return 0
    keep = {str(r) for r in open_requests}
    return sum(hs.end(r) for r in hs.requests() if r not in keep)


def check(t: HT.HistoryTable, strings: Sequence[bytes]) -> bool:
    """Whether table ``t``'s root is the root `history.check` recomputes over ``strings``, the committed strings of its rows in
    position order (a window's public bodies, then the next's)."""
    try:
        HT.check(t.program, t.value, t.words, strings, t.root, t.leaves)
    except ValueError:
        return False
    return True
