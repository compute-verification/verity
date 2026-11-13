"""The values a two-stage interior registers for its replay block's parameters, and the history tables its reads open:
what `interior_rows.interior` takes as ``links`` and ``tables``, and the challenger's copies of them.

* `Supply`, `Operand`: a replay-block parameter a template's port reads, a link of round 2's
  (`second_stage.replay_links`), which reads such a port only at a value the interior registers under the link's name.
  The interior commits no tree of its own at the port: its rows are rows of that value, ``per`` a sampled span, each unit
  reading the one its place in the block gives it. The stream takes each as a per-span value (`span_values`: its rows a
  sampled span and a row's leaf type), so its interior program sizes it by the replay sample; its row format is its
  reading port's (`link_schemas`). `links` reads a block's off its Definition and round 1's record as round 2 does.
* `Operand`: a parameter the replay gives (a picking unit's scores and count, a table read's row, salt and path). The
  verifier holds none of its bytes, so the value is the interior's own: its rows the replayed units' at their places,
  each under a fresh salt, its root registered before the second draw's coins, and R7 holds its name, rows, words and
  domain to the verifier's own entry (`link_words`).
* `Supply`, `supplied`, `sources`, `history`: a parameter the verifier supplies. `supplied` makes its value at public
  salts and the verifier fills it from its own sources (`sources`), reading each source's bytes by the row's leaf type
  (`DIGEST` or `INDEX`), so no row there is one the prover picks after the coins. Two kinds:
  a table read's ``root`` and ``domain`` (`EmbeddingRead_v1{V, H}`), each sampled span's the table round 1 served for it
  (`boundary_tables`: committed at serve, its root a port of the class's round-1 record, its rows filled to its class's
  C, `boundary_tables.class_rows`), which the challenger names from its own sample and copies round 1's root for
  (`history`), the tables' strings published (`interior_rows.Interior.published`), so the verifier recomputes the served
  root from them; and an index, row r of a sampled span the I32 the parameter holds there, which the verifier fills
  from its own bytes. Every read must open its span's table at its key (`gates.embedding.reference`) and every unit's
  supplied row must be the value's at its place before anything is committed; `table_reads` makes such reads.
* `Input`, `inputs`: a parameter round 1's record commits as a port (a token select's ``logits``, where the serve
  commits each token's row before the first stage's sample). It is no `Operand`, and `links` leaves it out: the
  interior registers no value under its name, so no input of the block is one the prover gives after the sample.
  Round 2 reads the template's port only at round 1's port of the link's name, so the interior commits the port's rows
  under the serve's salts at the rows each unit reads and its header opens them at round 1's root
  (`interior_rows.interior`'s ``inputs``).
* `Return`, `returns`: the block's return a template's output port writes, the other end of a replay block. The
  interior registers no value for it: where round 1's record commits the return as a port, round 2 reads the
  template's output only at that port, so the interior commits those output rows under the serve's salts at the rows
  each unit writes and its header opens them at round 1's root (`interior_rows.interior`'s ``returns``).
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from verity.protocols.spot.verifier import second_stage as SS
from verity.protocols.spot.verifier.one_stage import registered as RG
from verity.circuits.codec import definitions
from verity.commitments import hm96, history as HT
from verity.commitments.gates import embedding as EMB
from verity.commitments.windows import rows as CS
from verity.commitments.windows.interface import digest_words
from verity_vllm.pipeline import boundary_tables as BT, serving_rows as SRW

#: a table read's parameters (`gates.embedding`'s `EmbeddingRead_v1`): its key, its table's root and domain id, its opening
READ = ("tok", "root", "domain", "row", "salt", "path")
_HM = hm96.SHA512


#: the held value a published table's strings are in `history`
HELD = "strings"
#: a table read's parameters the verifier supplies, each a `Digest` a row
TABLE = ("root", "domain")
#: a supplied row's leaf type (`population.leaf_type`'s shape): a `Digest`'s, and an I32 index's two u16 words
DIGEST, INDEX = (8, 64, 1), (2, 16)


def _places(value: str, per: int, at: Sequence[int]) -> None:
    if per < 1 or any(not 0 <= a < per for a in at):
        raise ValueError(f"{value}: units reading rows {sorted(set(at))[:4]}... of its {per} a sampled span")


@dataclass(frozen=True)
class Operand:
    """A replay-block parameter the replay gives, and the port that reads it: template ``template``'s parameter ``port``
    reads the registered value ``value``, the block parameter under the name round 2 gives its link
    (`second_stage.replay_links`). The value holds ``per`` rows a sampled span, each of ``words`` u16 words, and unit j
    of a span's n_t reads its row ``at[j]``: row ``s·per + at[j]`` for sampled span s, where round 2 holds a linked
    port's read of a value the interior registers. The verifier holds none of its bytes: the interior commits the rows
    its units replayed there, each under a fresh salt, a row no unit reads the zero row. Several ports, of one template
    or of several, may read one value (a block parameter each of several templates reads its rows of): one `Operand` a
    port, alike in ``per`` and ``words``."""
    template: str
    port: str
    value: str
    per: int
    at: tuple[int, ...]
    words: int

    def __post_init__(self) -> None:
        _places(self.value, self.per, self.at)
        if self.words < 1:
            raise ValueError(f"{self.value}: rows of {self.words} words")

    @property
    def leaf(self) -> tuple[int, ...]:
        """A row's leaf type: its u16 words, as its port lays them out."""
        return (self.words, 16)


@dataclass(frozen=True)
class Supply:
    """A replay-block parameter the verifier supplies, and the port that reads it: template ``template``'s parameter
    ``port`` reads the registered value ``value``, named and placed as an `Operand` is (``per`` rows a sampled span, unit
    j reading its row ``at[j]``). ``index``: the rows are indices, row r of a span the I32 ``index[r]`` (two u16 words, the low one first),
    from a ``bytes`` source; None for a table read's ``root`` or ``domain`` (``port``, one of `TABLE`), every row of
    sampled span s its table's, from a ``history`` source."""
    template: str
    port: str
    value: str
    per: int
    at: tuple[int, ...]
    index: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        _places(self.value, self.per, self.at)
        if self.index is None and self.port not in TABLE:
            raise ValueError(f"{self.value}: {self.template}'s {self.port!r} is no table read's {' or '.join(TABLE)}, and "
                             "it holds no indices")
        if self.index is not None and (len(self.index) != self.per or any(not 0 <= x < 1 << 32 for x in self.index)):
            raise ValueError(f"{self.value}: {len(self.index)} indices for its {self.per} rows a sampled span, each an I32")

    @property
    def words(self) -> int:
        """A row's u16 words: a `Digest`'s 32, or an I32's two."""
        return 32 if self.index is None else 2

    @property
    def leaf(self) -> tuple[int, ...]:
        """A row's leaf type: `DIGEST`, or `INDEX`."""
        return DIGEST if self.index is None else INDEX


Link = Supply | Operand


@dataclass(frozen=True)
class Input:
    """A replay-block parameter round 1's record commits as a port, and the port that reads it: template ``template``'s
    parameter ``port`` reads the port ``value`` of round 1's record, the block parameter under the name round 2 gives
    its link (`second_stage.replay_links`). The port holds ``per`` rows a replay block, each of ``words`` u16 words, and
    unit j of a block's n_t reads its row ``at[j]``: row ``sample[i]·per + at[j]`` of the port for the i-th sampled
    block, where round 2 holds a link's read of a port of round 1's record (`second_stage`'s linked row). The serve
    committed those rows before the first stage's sample, so the interior registers no value for it."""
    template: str
    port: str
    value: str
    per: int
    at: tuple[int, ...]
    words: int

    def __post_init__(self) -> None:
        _places(self.value, self.per, self.at)
        if self.words < 1:
            raise ValueError(f"{self.value}: rows of {self.words} words")

    def rows(self, sample: Sequence[int]) -> list[int]:
        """The row of the port each of the template's Q·n_t units reads, in slot order, for the replay sample
        ``sample`` (the spans' slots in round 1's population)."""
        return [int(s) * self.per + a for s in sample for a in self.at]


@dataclass(frozen=True)
class Return:
    """A return of the replay block, and the output port that writes it: template ``template``'s output ``port`` is the
    block's return ``value``, under the name round 2 gives its link (`second_stage.replay_links`). The return holds
    ``per`` rows a replay block, and unit j of a block's n_t writes its row ``at[j]``: row ``sample[i]·per + at[j]`` of
    the value for the i-th sampled block, where round 2 holds a return link's read (`second_stage`'s linked row).
    ``held``: round 1's record commits the value as a port, so the read is of round 1's rows; without it round 2 has
    no port to read the return at."""
    template: str
    port: str
    value: str
    per: int
    at: tuple[int, ...]
    held: bool = False

    def __post_init__(self) -> None:
        _places(self.value, self.per, self.at)

    def rows(self, sample: Sequence[int]) -> list[int]:
        """The row of the value each of the template's Q·n_t units writes, in slot order, for the replay sample
        ``sample`` (the spans' slots in round 1's population)."""
        return [int(s) * self.per + a for s in sample for a in self.at]


def values(links: Sequence[Link]) -> dict[str, Link]:
    """Each value ``links`` name, in their order, by the first link that reads it. Refused by name where two links read
    one value unlike each other: one the verifier's and one the replay's, or at other rows a sampled span, a row's
    leaf type, indices or (a table's) parameter."""
    def kind(x: Link) -> tuple[Any, ...]:
        mine = (x.index, x.port if x.index is None else None) if isinstance(x, Supply) else ()
        return (type(x), x.per, x.leaf, *mine)
    out: dict[str, Link] = {}
    for x in links:
        f = out.setdefault(x.value, x)
        if kind(f) != kind(x):
            raise ValueError(f"{x.value}: {f.template}'s {f.port!r} and {x.template}'s {x.port!r} read it as two values: "
                             "one value is the verifier's or the replay's, of one count of rows a sampled span, each of "
                             "one type")
    return out


def span_values(links: Sequence[Link]) -> tuple[tuple[str, int, tuple[int, ...]], ...]:
    """``links`` as a two-stage window stream takes them (`Stream`'s ``span_values``): each value's name, its rows a
    sampled span and a row's leaf type, which the stream's interior program sizes by its replay sample, a value several
    ports read once."""
    return tuple((v, x.per, x.leaf) for v, x in values(links).items())


def link_values(links: Sequence[Link], q: int) -> tuple[tuple[str, int, tuple[int, ...]], ...]:
    """The values of an interior of ``q`` sampled spans, as its program takes them (`mixed_population_program`'s
    ``(name, leaves, width)``, `Stream.interior_program`'s of `span_values`), so R7 registers them: each of
    ``links``', ``q·per`` rows of its leaf type."""
    return tuple((v, per * int(q), leaf) for v, per, leaf in span_values(links))


def link_words(links: Sequence[Link]) -> dict[str, int]:
    """The challenger's copy of what the interior registers (`Stream.register_interiors`' ``registered``): each of
    ``links``' values' words a row."""
    return {v: x.words for v, x in values(links).items()}


def link_schemas(members: CS.Member | Sequence[CS.Member], links: Sequence[Link]) -> dict[str, str]:
    """The challenger's copy of each of ``links``' values' row format (`Stream.register_interiors`' ``schemas``): its
    reading port's, from the templates' layouts (``members``, one or one a template)."""
    ms = {m.template: m for m in ([members] if isinstance(members, CS.Member) else members)}
    out: dict[str, str] = {}
    for x in links:
        s = ms[x.template].schema(x.port)
        if out.setdefault(x.value, s) != s:
            raise ValueError(f"{x.value}: {x.template}'s {x.port!r} reads it in {s}, another of its ports in "
                             f"{out[x.value]}: a value is committed in one row format")
    return out


Members = CS.Member | Sequence[CS.Member]
Cells = int | Sequence[int]


def cells(members: Members, n_rho: Cells) -> tuple[tuple[CS.Member, ...], tuple[int, ...]]:
    """A window's templates and each one's units a span: ``members``, one or one a template, and ``n_rho``, its n_t
    each."""
    ms = (members,) if isinstance(members, CS.Member) else tuple(members)
    ns = (int(n_rho),) if isinstance(n_rho, int) else tuple(int(n) for n in n_rho)
    if not ms or len(ns) != len(ms):
        raise ValueError(f"{len(ms)} templates and cells {list(ns)}: a replay block holds n_t units of each template")
    return ms, ns


def links(program: Any, key: str, record: Mapping[str, Any], members: Members, n_rho: Cells,
          supplied: Mapping[str, Sequence[int] | None] | None = None) -> tuple[Link, ...]:
    """The links of the replay block ``program`` defines under descriptor id ``key``, as the interior registers their
    values: each parameter of the block a template's port reads, under the name round 2 gives it where round 1's record
    ``record`` names its ports (`second_stage.replay_links`), with its rows a block and the row each of the template's n_t
    units reads as round 2 derives them (`second_stage.replay_rows`). ``members`` and ``n_rho``: the window's templates
    and their units a span, as `interior_rows.interior_window` takes them. ``supplied``: the block's parameters the
    verifier supplies, by name, each its indices a row or None for a table read's root or domain (a `Supply`); every
    other is the replay's (an `Operand` of its port's words). A return link names an output port and is none of them,
    and neither is a parameter ``record`` commits as a port (`inputs`): the interior registers no value for it.
    ValueError where the Definition makes no links, a link names a template the window has none of or a unit no run
    places, or the verifier is to supply a parameter ``record`` commits."""
    by = {m.template: (m, n) for m, n in zip(*cells(members, n_rho), strict=True)}
    bare, named = SS.links_of(definitions(program), key), SS.replay_links(program, key, record)
    if bare is None or named is None:
        raise ValueError(f"replay block {key} makes no links round 2 derives (`second_stage.replay_links`)")
    rows = {(r.template, r.port): r for r in SS.replay_rows(program, key)}
    held = _committed(record)
    out: list[Link] = []
    for b, x in zip(bare, named, strict=True):
        if x.template not in by:
            raise ValueError(f"{x.value} is read by {x.template}, a template the window has none of: {list(by)}")
        m, n = by[x.template]
        if x.port not in dict(m.rows):
            continue
        if x.value in held:
            if b.value in (supplied or {}):
                raise ValueError(f"{x.value}: round 1's record commits it as a port, so the verifier supplies no value "
                                 f"of that name for {x.template}'s {x.port!r}")
            continue
        r = rows[(x.template, x.port)]
        at = tuple(SS.row_at(r.runs, j) for j in range(n))
        if any(a is None for a in at):
            raise ValueError(f"{x.template}: its {x.port!r} rows place other than its {n} units a span")
        if b.value in (supplied or {}):
            out.append(Supply(x.template, x.port, x.value, r.per, at, supplied[b.value]))
        else:
            out.append(Operand(x.template, x.port, x.value, r.per, at, dict(m.rows)[x.port]))
    return tuple(out)


def _committed(record: Mapping[str, Any]) -> set[Any]:
    """The names of the ports round 1's record ``record`` commits."""
    return {r.get("port") for r in record.get("roots", ())}


def inputs(program: Any, key: str, record: Mapping[str, Any], members: Members, n_rho: Cells) -> tuple[Input, ...]:
    """The parameters of the replay block ``program`` defines under descriptor id ``key`` that round 1's record
    ``record`` commits as ports: each link round 2 derives (`second_stage.replay_links`, named as ``record`` names its
    ports) at a template's parameter whose value is one of ``record``'s ports, with its rows a block and the row each of
    the template's n_t units reads (`second_stage.replay_rows`). ``members`` and ``n_rho``: as `links` takes them,
    which gives every other parameter's link. ValueError where the Definition makes no links, or such a link names a
    template the window has none of or a unit no run places."""
    by = {m.template: (m, n) for m, n in zip(*cells(members, n_rho), strict=True)}
    named = SS.replay_links(program, key, record)
    if named is None:
        raise ValueError(f"replay block {key} makes no links round 2 derives (`second_stage.replay_links`)")
    rows = {(r.template, r.port): r for r in SS.replay_rows(program, key)}
    held = _committed(record)
    out: list[Input] = []
    for x in named:
        if x.template not in by:
            raise ValueError(f"{x.value} is read by {x.template}, a template the window has none of: {list(by)}")
        m, n = by[x.template]
        if x.port not in dict(m.rows) or x.value not in held:
            continue
        r = rows[(x.template, x.port)]
        at = tuple(SS.row_at(r.runs, j) for j in range(n))
        if any(a is None for a in at):
            raise ValueError(f"{x.template}: its {x.port!r} rows place other than its {n} units a span")
        out.append(Input(x.template, x.port, x.value, r.per, at, dict(m.rows)[x.port]))
    return tuple(out)


def returns(program: Any, key: str, record: Mapping[str, Any], members: Members, n_rho: Cells) -> tuple[Return, ...]:
    """The returns of the replay block ``program`` defines under descriptor id ``key``: each link round 2 derives
    (`second_stage.replay_links`, named as round 1's record ``record`` names its ports) at a template's output port, with
    its rows a block and the row each of the template's n_t units writes (`second_stage.replay_rows`), and whether
    ``record`` commits the value as a port. ``members`` and ``n_rho``: as `links` takes them, which gives the links at
    the templates' parameters. ValueError where the Definition makes no links, or a return names a template the window
    has none of, a port that is no output of it or a unit no run places."""
    by = {m.template: (m, n) for m, n in zip(*cells(members, n_rho), strict=True)}
    named = SS.replay_links(program, key, record)
    if named is None:
        raise ValueError(f"replay block {key} makes no links round 2 derives (`second_stage.replay_links`)")
    rows = {(r.template, r.port): r for r in SS.replay_rows(program, key)}
    held = _committed(record)
    out: list[Return] = []
    for x in named:
        if x.template not in by:
            raise ValueError(f"{x.value} is written by {x.template}, a template the window has none of: {list(by)}")
        m, n = by[x.template]
        if x.port in dict(m.rows):
            continue
        if x.port not in dict(m.outs):
            raise ValueError(f"{x.template}: {x.value} is linked at its {x.port!r}, none of its rows "
                             f"{[p for p, _w in m.rows]} or outputs {[p for p, _w in m.outs]}")
        r = rows[(x.template, x.port)]
        at = tuple(SS.row_at(r.runs, j) for j in range(n))
        if any(a is None for a in at):
            raise ValueError(f"{x.template}: its {x.port!r} rows place other than its {n} units a span")
        out.append(Return(x.template, x.port, x.value, r.per, at, x.value in held))
    return tuple(out)


def wired(rows: Sequence[Any], template: str, port: str, n: int, per: int) -> tuple[int, ...]:
    """The writer unit each of ``template``'s ``n`` units a block reads at its wired port ``port``, among the writer's
    ``per`` a block, as round 2 reads a wired port (`second_stage.wiredRow`): unit j's row under the port's runs in
    ``rows`` (the replay block's, `second_stage.replay_rows`). Without rows for the port, the writer's one unit; a
    writer of several units a block is refused, since a wire alone does not say which each unit reads."""
    r = next((x for x in rows if (x.template, x.port) == (template, port)), None)
    if r is None:
        if per != 1:
            raise ValueError(f"{template}: its wired {port!r} reads a writer of {per} units a block, and nothing says which "
                             "each of its units reads: give the replay block's rows (`second_stage.replay_rows`)")
        return (0,) * n
    at = tuple(SS.row_at(r.runs, j) for j in range(n))
    if r.per != per or any(a is None or a >= per for a in at):
        raise ValueError(f"{template}: its wired {port!r} rows place other than its {n} units a block on its writer's "
                         f"{per}")
    return tuple(int(a) for a in at)


def _supplies(links: Sequence[Link]) -> tuple[Supply, ...]:
    return tuple(x for x in values(links).values() if isinstance(x, Supply))


def supplied(program: str, links: Sequence[Link], q: int, entries: Sequence[Mapping[str, Any]] = (),
             schemas: Mapping[str, str] | None = None) -> tuple[Any, ...]:
    """The values the `Supply`s of ``links`` name for ``q`` sampled spans, each committed as core's `Registered` under
    ``program`` at its public salts (`registered.supplied_salts`), as `interface.supplied` fills a served window's table reads: an
    index's row ``s·per + r`` the I32 ``index[r]``'s two u16 words, the low one first; a table read's, its span's table's root or domain id as a
    `Digest` port's 32 words (`interface.digest_words`), from the tables' entries alone (``entries[s]`` sampled span s's,
    `history.HistoryTable.entry`). ``schemas``: each value's row format, its reading port's (`registered.SCHEMA_V2`
    where none is given)."""
    out = []
    for s in _supplies(links):
        if s.index is None:
            if len(entries) != q:
                raise ValueError(f"{s.value}: {len(entries)} tables for {q} sampled spans")
            rows = [digest_words(bytes.fromhex(e[s.port])) for e in entries for _ in range(s.per)]
        else:
            rows = [(x & 0xFFFF, x >> 16) for _ in range(q) for x in s.index]
        out.append(RG.Registered.commit(program, s.value, rows, RG.supplied_salts(program, s.value, len(rows)),
                                        schema=(schemas or {}).get(s.value, RG.SCHEMA_V2)))
    return tuple(out)


def sources(links: Sequence[Link], q: int, names: Sequence[str] = ()) -> dict[str, list[list[dict[str, str]]]]:
    """The challenger's copy of `supplied`'s values for ``q`` sampled spans (`Stream.register_interiors`' ``supplied``):
    each row one source; an `Operand` has none. An index's row ``s·per + r`` is ``{"bytes": ...}``, ``index[r]`` as four big-endian bytes, which
    the verifier reads as the row's two u16 leaves, the low one first (`registered.bytes_leaves`). A table read's is
    ``{"history": "<names[s]>/root"}`` or ``/domain``, which the verifier resolves to the root or domain id it recomputed
    for table ``names[s]``, the one the challenger names for sampled span s."""
    out = {}
    for s in _supplies(links):
        if s.index is None:
            if len(names) != q:
                raise ValueError(f"{s.value}: {len(names)} tables' names for {q} sampled spans")
            out[s.value] = [[{"history": f"{n}/{s.port}"}] for n in names for _ in range(s.per)]
        else:
            out[s.value] = [[{"bytes": f"{x:08x}"}] for _ in range(q) for x in s.index]
    return out


def placed(ms: Sequence[CS.Member], cells: Sequence[int], q: int,
            links: Sequence[Link]) -> list[tuple[Link, int, tuple[int, ...]]]:
    """Each of ``links`` with its member and the row of its value each of the member's Q·n_t units reads: unit j of
    sampled span s, row ``s·per + at[j]``. Refused by name for a template the window lacks, a port that is none of its
    rows or is wired, a port linked twice, and a placement of other than its n_t units."""
    at = {m.template: k for k, m in enumerate(ms)}
    out, seen = [], set()
    for s in links:
        k = at.get(s.template)
        if k is None:
            raise ValueError(f"{s.value} is read by {s.template}, a template the window has none of: {list(at)}")
        m = ms[k]
        if s.port not in dict(m.rows) or s.port in {p for p, _o, _ws in m.wired}:
            raise ValueError(f"{s.template}: {s.value} is read at its {s.port!r}, which is none of its rows "
                             f"{[p for p, _w in m.rows]} or is wired: an interior member's rows are its parameters (`interior_rows.named`)")
        if dict(m.rows)[s.port] != s.words:
            raise ValueError(f"{s.template}: its {s.port!r} row is {dict(m.rows)[s.port]} words, not {s.value}'s {s.words}")
        if len(s.at) != cells[k]:
            raise ValueError(f"{s.template}: {s.value} places {len(s.at)} units, not its {cells[k]} a sampled span")
        if (s.template, s.port) in seen:
            raise ValueError(f"{s.template}: its {s.port!r} reads two values")
        seen.add((s.template, s.port))
        out.append((s, k, tuple(i * s.per + a for i in range(q) for a in s.at)))
    return out


def operand(program: str, x: Operand, reads: Sequence[tuple[np.ndarray, Sequence[int]]], q: int, schema: str,
             salt_key: bytes | None, window: str | None = None) -> Any:
    """`Operand` ``x``'s value for ``q`` sampled spans, committed as core's `Registered` under ``program`` in its ports'
    row format ``schema``: row ``at[u]`` the row unit u replayed at a port that reads it (``reads``: each port's rows
    and ``at``), a row no unit reads the zero row, each under a fresh salt (``salt_key``: tests only, the salts derived
    from it and the value's name). ``window``: the interior window's open root, which its domain binds where the
    challenger names one (`Stream.register_interiors`' ``window_root``: a value registered afresh each window)."""
    own = np.zeros((q * x.per, x.words), dtype=np.uint16)
    for rows, at in reads:
        own[list(at)] = rows
    n = len(own)
    salts = None
    if salt_key is not None:
        ys = CS.salts(hashlib.sha512(bytes(salt_key) + x.value.encode()).digest()[:32], 0, n)
        salts = [ys[i * CS.SALT_BYTES:(i + 1) * CS.SALT_BYTES] for i in range(n)]
    return RG.Registered.commit(program, x.value, own.tolist(), salts, schema=schema, window=window)


def history(boundary: Mapping[str, Any], names: Sequence[str], words: int, positions: int,
            published: Mapping[str, Sequence[str]], bodies: Mapping[str, bytes]) -> dict[str, Any]:
    """The challenger's `history` for an interior registration (`Stream.register_interiors`' ``history``): sampled span
    s's table ``names[s]``, which the challenger names from the class and its own sample, as round 1's record
    ``boundary`` registered it as a port, under round 1's program, its leaves and its domain, of ``words`` words a row.
    Its rows are one held run of the class's ``positions`` strings (C, `boundary_tables.class_rows`), the ones the prover
    published for it (``published``, by name: `interior_rows.Interior.published`), and its registered root is round 1's, so the
    verifier holds what the reads open to the table served before the coins. ``bodies``: the interior's released public
    bodies by member key, which its leaf layer binds. Raises ValueError where round 1 registered no such table, or one
    under another domain."""
    roots = {r["port"]: r for r in boundary["roots"]}
    tables, held, registered = [], {}, {}
    for n in names:
        r = roots.get(n)
        if r is None:
            raise ValueError(f"round 1's record registers no table {n}")
        if HT.domain(boundary["program"], n, words, r["leaves"]).domain_id.hex() != r["domain"]:
            raise ValueError(f"round 1 registers {n} under another domain than a table of {words} words at {r['leaves']} "
                             "leaves")
        tables.append({"name": n, "program": boundary["program"], "writer": "none", "out": "none", "words": words,
                       "capacity": r["leaves"], "first": 0, "units": 0, "members": [],
                       "runs": [{"start": 0, "count": int(positions), "held": HELD, "row": 0}]})
        held[n] = {HELD: list(published.get(n, ()))}
        registered[n] = r["root"]
    return {"digests": {k: hashlib.sha512(b).hexdigest() for k, b in bodies.items()}, "tables": tables, "held": held,
            "registered": registered}


def reader(m: CS.Member) -> tuple[int, int] | None:
    """(V, H) of a member of table reads (`EmbeddingRead_v1{V, H}`), None for any other."""
    fn = SRW.template(m.template).fn
    if tuple(p for p, _t in fn.signature()[0]) != READ:
        return None
    b = getattr(fn, "bindings", {})
    return int(b["V"]), int(b["H"])


def _digests(words: np.ndarray) -> bytes:
    """`interface.digest_words`' inverse over consecutive 32-word digests: their bytes."""
    return np.asarray(words, dtype="<u2").view("<u8").astype(">u8").tobytes()


def _digest_words(b: bytes) -> np.ndarray:
    """`_digests`' inverse: consecutive 64-byte digests as `Digest` words (`interface.digest_words`), concatenated."""
    return np.concatenate([digest_words(b[i:i + 64]) for i in range(0, len(b), 64)])


def kept_table(kept: BT.Served) -> HT.HistoryTable:
    """The table `interior_rows.interior` opens for a drawn span, from what rank 0 kept of it (`boundary_tables.load`): its C rows
    committed again under their kept salts, round 1's program, name, words and leaves, refused unless that is the entry
    round 1 listed. On the node only, as the rows are."""
    t = HT.HistoryTable(kept.program, kept.value, kept.words, kept.leaves)
    n = len(kept.salts) // max(1, len(kept.rows))
    t.append(kept.rows, [kept.salts[i * n:(i + 1) * n] for i in range(len(kept.rows))])
    if t.entry() != kept.entry():
        raise ValueError(f"{kept.value}: its kept rows and salts commit to another table than the one round 1 listed")
    return t


def table_reads(tc: HT.HistoryTable, toks: Sequence[int]) -> dict[str, dict[str, np.ndarray]]:
    """Reads of table ``tc`` (`history.HistoryTable`: as the serve built it, `boundary_tables.Kept.table`) at keys
    ``toks``, each in [0, its leaves), as `interior_rows.interior` takes a read member's rows and outputs: each key, the table's root and
    domain id, the key's row and salt (the zero row under the zero salt past the table's rows) and its path; h the row,
    ok 1."""
    n, H = len(tc.rows), tc.words
    cols: dict[str, list[np.ndarray]] = {p: [] for p in (*READ, "o0", "o1")}
    for k in (int(x) for x in toks):
        if not 0 <= k < tc.leaves:
            raise ValueError(f"a read of {tc.value} at key {k}, outside its {tc.leaves} leaves")
        row = np.asarray(tc.rows[k]) if k < n else np.zeros(H, np.uint16)
        salt = bytes(tc.salts[k]) if k < n else bytes(_HM.salt_bytes)
        for p, v in (("tok", np.asarray([k & 0xFFFF, k >> 16])), ("root", digest_words(tc.root)),
                     ("domain", digest_words(tc.domain.domain_id)), ("row", row), ("salt", _digest_words(salt)),
                     ("path", _digest_words(tc.path(k))), ("o0", row), ("o1", np.ones(1))):
            cols[p].append(np.asarray(v, dtype=np.uint16))
    rows = {p: np.stack(cols[p]) for p in READ}
    return {"rows": rows, "outs": {"o0": np.stack(cols["o0"]), "o1": np.stack(cols["o1"])}}


def opened(m: CS.Member, d: Mapping[str, Mapping[str, np.ndarray]], cell: int, tcs: Sequence[HT.HistoryTable],
            V: int) -> None:
    """Every read of member ``m`` (``d``, its Q·n_t units, unit u of sampled span u // n_t) opens its span's table at its
    key: its root and domain id are the table's, and `gates.embedding.reference` gives its h with ok = 1."""
    r, o = d["rows"], d["outs"]
    for u in range(len(r["tok"])):
        tc = tcs[u // cell]
        tok = int(r["tok"][u][0]) | int(r["tok"][u][1]) << 16
        what = f"{m.template}: read {u} (sampled span {u // cell}, key {tok})"
        root, dom = _digests(r["root"][u]), _digests(r["domain"][u])
        if root != tc.root or dom != tc.domain.domain_id:
            raise ValueError(f"{what} takes a root or domain id other than its table's, {tc.value}'s")
        path = _digests(r["path"][u])
        h, ok = EMB.reference(V, tok, root, dom, [int(w) for w in r["row"][u]], _digests(r["salt"][u]),
                              [path[i:i + 64] for i in range(0, len(path), 64)])
        if ok != 1 or int(o["o1"][u][0]) != 1 or [int(w) for w in o["o0"][u]] != list(h):
            raise ValueError(f"{what} does not open its table there: its row, salt and path are no opening of {tc.value}'s "
                             "root at its key, or its h and ok are not the read's")


__all__ = ["DIGEST", "HELD", "INDEX", "READ", "TABLE", "Cells", "Link", "Members", "Operand", "Return", "Supply", "cells",
           "history", "kept_table", "link_schemas", "link_values", "link_words", "links", "opened", "operand", "placed",
           "reader", "returns", "sources", "span_values", "supplied", "table_reads", "values", "wired"]
