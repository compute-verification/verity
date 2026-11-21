"""Registered values: a program's weights, constants and tables, committed once at docket under hiding leaves, and
every read of one at a public position opened against that commitment.

A *registered value* is a root parameter of the program, read row by row: row `r` is its `words` leaves from
`r * words`, in the parameter's leaf order, each leaf `word_bits` bits (16 or 32: the parameter's leaf width rounded up
to whole u16 words, `width`). Each row is committed as the circuit statement commits that port's row
(`hm96-sha512/row/v1`): its u16 words (`registered_value.row_words`: each leaf `word_bits / 16` words, low first, zero-padded to whole
SHA-512 chunks of 64 words, as a Flock port row and its `values` row are laid out), `x = SHA-512(sha512_row_prefix(ROLE_X,
16, n) || row)` over those `n` words u16 LE, a 192-byte salt `y` sampled from the operating system once, at docket,
`b || c = hm96.SHA512.commit_string(default_key, x, y)`, and the frame-v3-sha512 leaf `row_leaf(D, r,
hm96.SHA512.tree_leaf(default_key, b || c), "hm96-sha512/row/v1")`. The value's root is the frame-v3-sha512 root over its
rows, under the domain `D` (`registered_value.domain`): `CommitmentDomain(B, -2, RangeIndexedDomain(0, rows), hash="sha512")`, `B` the 64
bytes of `identity_digest_sha512("verity/one-stage/registered-domain/v1", {"program", "value", "schema", "words", "leaves"})`, with
`"word_bits"` beside them when it is not 16 (16 is the default, so a 16-bit value's domain, entry and root are what they
were before `word_bits`). The prover keeps the rows and salts; the docket names the root.

A *per-window value* (a window's token rows, say, which a window-independent Program registers under the same name every
window) adds `"window": R` to the binding, R the window's open root (lowercase hex of 64 bytes) that the challenger's
signed open names: its domain, and so every leaf of its tree, is that window's alone, so a root registered in one window
opens under no other's domain. A value the verifier holds across windows (the weights) leaves `window` out, as every
value did before it: the verifier names which (`registration.check`'s `held`, with its pinned, supplied and output
values), and its domain is the same in every window.

A value read by a bit-row port is committed in that port's row format instead (`schema`, one of `registered_value.SCHEMAS`): under
`hm96-sha512/row/v2` a row is the bit row of its leaves (`bit_row((row, word_bits))`: its `registered_value.row_words` with no padding,
little-endian), `x = sha512_row_v2_digest` over it, and its leaf `row_leaf(D, r, ..., "hm96-sha512/row/v2")`; under
`hm96-sha512/row-seg/v1` the same bit row under `sha512_row_seg_digest`, its leaf under `"hm96-sha512/row-seg/v1"`. The
domain is the same in every format; the leaf frame binds the format. The verifier's reads file names a value's format
under `row_schema` when it is not `hm96-sha512/row/v1` (`schema_entry`), and the Lean verifier checks each port in its
own format: a port's rows must be of the format its docket names, and a bit row exactly `words * word_bits` bits.
A bit row's leaves pack the parameter's narrower leaves (`row_leaves`): H1's A, K int8 codes of one-bit leaves, is a
`row/v2` value of `words = K/2` leaves of `word_bits = 16`, a row 8K of the parameter's leaves.

A read is at a *public position* when the verifier derives it from its own program (`reads`): a proof block's parameter whose
leaves are exactly row `r` of a registered value reads row `r`, for every proof block, before anything is served; a parameter
that is an array of whole rows (H1's tile reads R rows of A) reads each element's row at its own key `name[i]`, the
circuit's port of that row, and the Lean verifier holds each such port as its own registered port. A statement
that reads row `r` commits it with the docket's salt, so its `b || c` is the registered one, and opens it: header
`registered: {port: {"value", "positions", "paths"}}`, each of the port's table rows its position and its frame-v3 path
(the siblings, leaf level first, 64 bytes each, hex). The verifier checks the port's row width, each path to the
registered root and each instance's position against its own read map (`reads_file`, the Lean verifier's `--registered`,
all from its own program but the root); a statement that commits a registered row under fresh salts has no path and is
refused.

A *supplied value* is a registered value the verifier fills itself, from what it holds before the sample: each row is the
bytes of its sources in order, each source a root the record names (`{"port": P}`, the `roots` entry of port P;
`{"value": V}`, the `registered` entry of value V, never a supplied value's; `{"domain": V}`, that entry's domain id),
bytes of its own (`{"bytes": hex}`), a
position (`{"position": "K"}`: K, a decimal below the record's population, after every proof block of the log's earlier
records, `log_base`, as a u64 big-endian, so no two records of one log share a position) or its coin bound to its
stream, its prover and the record's window (`{"window": hex}`: `window_salt`, SHA-512 of `WINDOW_SALT`, the coin, the
verifier's binding, `{"stream", "prover"}`, and the window's canonical JSON: a salt per prover, stream and window, as R6
takes one record a window and the window names the binding, so a tile seeded in one stream is no other's). A row's bytes
are read as the parameter's leaves of `w` bits
(`bytes_leaves`: each big-endian 32-bit half, its `32 / w` leaves least significant first, so a digest is 16 leaves of 32
bits or 512 of one; a row of two bytes is one 16-bit half, its `16 / w` leaves: `half_leaves`), except where the parameter's leaves are whole `Digest`s (`digests`: `DIGEST` or an array of them),
read as a circuit's `Digest` holds them (`digest_leaves`: each big-endian 64-bit word's 64 bits least significant
first). Its rows are
committed as any registered value's, in the row format the verifier names for it (`schema`, as the port that reads it:
under a bit-row format each row's parameter leaves packed into its `word_bits`-bit leaves, `bit_words`, so a `Digest`
row of `row/v2` is 32 u16 words), under public salts (`registered_value.supplied_salts`, row i's from the program's digest, the value's
name and i alone), so its root is a function of the record and the verifier's bytes: R7 recomputes it
(`supplied_root`) and refuses a record whose entry names another (`R7-supplied`). PoUW's seed inputs are such values: a
proof block's seed then reads the roots its operands are committed under, which the prover can't choose.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Collection, Iterable, Mapping, Sequence

from verity.commitments import hm96
from verity.commitments import session_coins as SC
from verity.commitments.crypto import lms
from verity.commitments.merkle import CommitmentDomain, merkle_depth
from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_SEG, SCHEMA_HM96_SHA512_ROW_V2
from verity.circuits import supplied as SU
from verity.circuits.codec import canonical_json, descriptor_chunks, jcs_domain_error
from verity.circuits.partition_object import TemplatePartition
from verity.circuits.program import Program
from verity.circuits.refs import runs as ref_runs
from verity.circuits.types import Array, Value
from verity.commitments import registered_value as RV

from . import partition as P
from verity.commitments.domains import (ONE_STAGE_PERIOD_SALT, REGISTERED_READS_V0, REGISTERED_READS_V1,
                                        REGISTERED_READS_V2)

READS = REGISTERED_READS_V0
#: the format whose port may carry `registered_value.READS_FROM`: proof block `reads_from + i` reads `reads[i]`, every other proof block nothing; each
#: of its objects holds only its own keys (Flock's `Registered.FORMAT`, `FILE_KEYS`, `PORT_KEYS`)
READS_V1 = REGISTERED_READS_V1
READS_V2 = REGISTERED_READS_V2
#: the `READS_V1` key naming the ports bound to the window's committed values, each `{"root", "domain"}` (`Flock.RegBound`)
BOUND = "bound"
SCHEMA_V2 = SCHEMA_HM96_SHA512_ROW_V2
SCHEMA_SEG = SCHEMA_HM96_SHA512_ROW_SEG
#: a SHA-512 digest as a circuit's `Digest` holds it: eight words of 64 one-bit leaves (`digest_leaves`)
DIGEST = Array(8, Array(64, Value(1)))
#: the reads-file key of a port not named after the template parameter it reads: that parameter's `reads` key, which the
#: Lean verifier holds its program's registered reads to (`Flock.Registered.PARAM_KEY`, `checkCover`)
PARAM_KEY = "param"
WINDOW_SALT = ONE_STAGE_PERIOD_SALT
#: a supplied row's source kinds: a served port's root, a registered value's root, the verifier's own bytes, a position
#: after the log's, the verifier's coin bound to the record's window
SOURCES = ("port", "value", "domain", "bytes", "position", "window")
#: a row of a supplied input whose layout is `le` leads with `(LE, "")` (`of_mark`); no scheme names it
LE = "le"
#: a supplied value is short: at most this many bits, whatever the population (`Supplied.SHORT_BITS`)
SHORT_BITS = 65536
_INST = hm96.SHA512
_HEX = re.compile(r"[0-9a-f]+")
_EVEN_HEX = re.compile(r"(?:[0-9a-f]{2})*")
_DEC = re.compile(r"0|[1-9][0-9]*")
#: `Flock.Registered.freshValue`'s prefix: ``input/n<w>`` is run input node ``w``'s one tree
FRESH = "input/n"


def fresh_value(w: int) -> str:
    """`Flock.Registered.freshValue`."""
    return f"{FRESH}{w}"


def is_fresh_value(v: Any) -> bool:
    """`Flock.Registered.isFreshValue`: ``v`` is ``fresh_value(w)`` for some node ``w``."""
    return type(v) is str and v.startswith(FRESH) and _DEC.fullmatch(v[len(FRESH):]) is not None


def width(program: Program, value: str) -> int:
    """The registered parameter `value`'s `word_bits`: its leaf width rounded up to whole u16 words, one width, at most
    32 bits."""
    t = dict(program.fn.signature()[0]).get(value)
    if t is None:
        raise ValueError(f"{value!r} is not a parameter of the program")
    w = t.homogeneous_width()
    if w is None or not 0 < w <= max(RV.WIDTHS):
        raise ValueError(f"parameter {value!r}: its leaves are not one width of at most {max(RV.WIDTHS)} bits "
                         f"({sorted(t.width_set())})")
    return 16 * -(-w // 16)


def parameter_gates(program: Program, value: str) -> range:
    """The root parameter `value`'s Input gates, which are one contiguous interval. Read run by run, never gate by gate: a
    weight tensor's parameter is ~1e9 gates."""
    names = [name for name, _ in program.fn.signature()[0]]
    if value not in names:
        raise ValueError(f"{value!r} is not a parameter of the program ({names})")
    if (gates := _gates(program, names.index(value))) is None:
        raise ValueError(f"parameter {value!r}: its gates are not one interval")
    return gates


def held_gates(program: Program, values: Iterable[str]) -> list[range]:
    """The Input gates of each of `values` that is a parameter of the program whose gates are one non-empty interval
    (`parameter_gates`, Lean's `Registered.Param.gates`), in `values`' order. Without `values` it reads nothing of the program,
    which may be a decoded one (`codec.DecodedProgram`, an old record's) without a signature."""
    if not (values := list(values)):
        return []
    names = [name for name, _ in program.fn.signature()[0]]
    return [g for v in values if v in names and (g := _gates(program, names.index(v))) is not None]


def _gates(program: Program, pi: int) -> range | None:
    first = end = None
    for _space, _idx, lo, hi in ref_runs(program.input_coll(pi).refs):
        if end is not None and lo != end:
            return None
        first, end = lo if first is None else first, hi
    return None if first is None or end == first else range(first, end)


def row_leaves(program: Program, value: str, words: int, schema: str = RV.SCHEMA) -> int:
    """The parameter's leaves in one registered row of `words` leaves: `words` under `SCHEMA`, each parameter leaf its own
    registered leaf; under a bit-row schema the row's `words * word_bits` bits, as many parameter leaves of the parameter's
    own width as they hold (H1's A: K int8 codes of one-bit leaves, K/2 u16 words)."""
    if RV.known_schema(schema) == RV.SCHEMA or words < 1:
        return words
    word_bits = width(program, value)
    w = dict(program.fn.signature()[0])[value].homogeneous_width()
    if words * word_bits % w:
        raise ValueError(f"parameter {value!r}: a {schema} row of {words} {word_bits}-bit leaves is no whole number of its "
                         f"{w}-bit leaves")
    return words * word_bits // w


def shape(program: Program, value: str, words: int, schema: str = RV.SCHEMA) -> int:
    """The rows of `value` read `words` leaves at a time, in `schema`'s rows (`row_leaves`)."""
    n, k = len(parameter_gates(program, value)), row_leaves(program, value, words, schema)
    if k < 1 or n % k:
        raise ValueError(f"parameter {value!r}: {n} leaves are not rows of {words}" +
                         (f" ({k} of its leaves in a {schema} row)" if k != words else ""))
    return n // k


def table(reads: Sequence[int]) -> tuple[list[int], list[int]]:
    """Each instance's position as a shared-row table: the distinct positions in first-read order, and each instance's ref.
    The positions are public, so the layout reveals nothing the program does not."""
    order: dict[int, int] = {}
    refs = [order.setdefault(int(p), len(order)) for p in reads]
    return list(order), refs


def verify_read(dom: CommitmentDomain, root: bytes, position: int, bc: bytes, path: bytes, schema: str = RV.SCHEMA) -> bool:
    """`bc`, as the registered value's row at `position`, opens `root` under the verifier's own domain `dom` through
    `path`, its leaf framed under `schema`: the check the Lean verifier makes for each table row of a registered port."""
    depth = merkle_depth(dom.count)
    if not 0 <= position < dom.count or len(bc) != _INST.proven_bytes or len(path) != 64 * depth:
        return False
    digest, i = RV.leaf(dom, position, bc, schema), position
    for level in range(depth):
        sib = path[64 * level:64 * level + 64]
        digest = dom.node(level, i >> 1, *((digest, sib) if i % 2 == 0 else (sib, digest)))
        i >>= 1
    return digest == root


def _row(hits: Sequence[tuple[str, int] | None], value: str, k: int) -> int | None:
    """The row of `value` that `hits` (each leaf's owner) is, whole and in order, `k` leaves to a row; None if it is not."""
    first = hits[0][1] if hits and hits[0] is not None and hits[0][0] == value else -1
    if first < 0 or first % k or len(hits) != k or list(hits) != [(value, first + j) for j in range(k)]:
        return None
    return first // k


def _source(template: str, parameter: str, element: int | None) -> str:
    return (f"parameter {parameter!r}" if element is None else f"row {element} of parameter {parameter!r}") + \
        f" of template {template}"


def reads(program: Program, query: Mapping[str, Any], values: Mapping[str, int],
          schemas: Mapping[str, str] | None = None, template: str | None = None) -> dict[str, dict[str, Any]]:
    """Every proof block's reads of the registered values, from the verifier's own program: `values` is `{parameter: words}`, and
    the result `{template parameter: {"value", "rows"}}`, `rows` one entry per proof block of the population (the row it reads,
    None for a proof block of another template). A proof block's parameter that holds a registered leaf must be exactly one row, read
    under its own name, or an array whose every element is one whole row of the value (H1's A: a tile's R rows), each
    element `i` read under `name[i]`, as the circuit's port of that row; and a template's parameter reads the same value
    in every proof block or in none: a read at a computed position is no public read. `schemas` (`{value: schema}`) gives a
    value's row format when it is not `SCHEMA`, which sets how many leaves a row holds (`row_leaves`). A key is one
    parameter's or one row's of it (in every template that has it): two reads under one key, a parameter `a[0]` beside
    row 0 of a parameter `a`, are refused, never merged. With `template`, the reads are that one template's: only its
    proof blocks read (every other's row is None), so the keys are its parameters and a parameter of one name reads its
    own value in each template; a template with no instance among the query's units is refused. The proof blocks are a
    template query's instances; a `Q_word` or `Q_call` proof block reads no registered value (`ValueError`)."""
    tp = P.evaluate(program, query)
    if not isinstance(tp, TemplatePartition):
        raise ValueError("a registered value is read by a template query's instances, not by a Q_word or Q_call unit")
    per_row = {v: row_leaves(program, v, w, (schemas or {}).get(v, RV.SCHEMA)) for v, w in values.items()}
    spans = []
    for v, w in values.items():
        shape(program, v, w, (schemas or {}).get(v, RV.SCHEMA))
        spans.append((parameter_gates(program, v), v))

    def owner(gate: int) -> tuple[str, int] | None:
        """The registered value whose leaf `gate` is, and which leaf: parameters' gates are disjoint intervals."""
        return next(((v, gate - r.start) for r, v in spans if gate in r), None)

    tids = {k: tid for k, tid, _a, _n in tp.ranges}
    if template is not None and template not in tids.values():
        raise ValueError(f"template {template} has no instance among the query's units "
                         f"({sorted(set(tids.values()))}): a read is a parameter of a template the query draws from")
    nodes = program.root.body.nodes
    out: dict[str, dict[str, Any]] = {}
    value_of: dict[tuple[str, str], str | None] = {}
    source: dict[str, tuple[str, str, int | None]] = {}
    for u in range(tp.population):
        k, _m = tp.locate(u)
        if template is not None and tids[k] != template:
            continue
        gates, at = tp.inputs(u), 0
        for name, t in nodes[k].fn.signature()[0]:
            s, at = gates[at:at + t.leaves], at + t.leaves
            hits = [owner(g) for g in s]
            value, rows = None, {}
            if any(hits):
                value = hits[0][0] if hits[0] is not None else None
                w = per_row.get(value, 0)
                if value is not None and len(s) == w and (row := _row(hits, value, w)) is not None:
                    rows = {name: row}
                elif value is not None and isinstance(t, Array) and t.n > 1 and t.elem.leaves == w:
                    rows = {f"{name}[{i}]": _row(hits[i * w:(i + 1) * w], value, w) for i in range(t.n)}
                if not rows or None in rows.values():
                    raise ValueError(f"unit {u}: parameter {name!r} is not one whole row of a registered value, nor an "
                                     "array of whole rows of one")
            if value_of.setdefault((tids[k], name), value) != value:
                raise ValueError(f"unit {u}: parameter {name!r} of {tids[k]} reads {value!r}, and in another unit "
                                 f"{value_of[(tids[k], name)]!r}")
            for i, (key, row) in enumerate(rows.items()):
                src = (name, i if key != name else None)
                if (was := source.setdefault(key, (tids[k], *src)))[1:] != src:
                    raise ValueError(f"port {key!r} names two reads: {_source(*was)} and {_source(tids[k], *src)}: "
                                     "a port reads one parameter's row")
                entry = out.setdefault(key, {"value": value, "rows": [None] * tp.population})
                if entry["value"] != value:
                    raise ValueError(f"parameter {key!r} reads {entry['value']!r} in one template and {value!r} in another")
                entry["rows"][u] = row
    return out


def registered_entries(program: Program, values: Mapping[str, int], schemas: Mapping[str, str] | None = None,
                       window: str | None = None, fixed: Sequence[str] = ()) -> dict[str, dict[str, Any]]:
    """The verifier's own `{value: {"leaves", "words", "word_bits", "domain"}}` for the parameters it expects registered
    (`values`: `{parameter: words}`), from its own program (`word_bits`: `width`; `leaves`: `shape`, in the value's row
    format from `schemas`, default `SCHEMA`). `window`: the window's open root, which every value's domain binds but the
    ones in `fixed`, the values it holds across windows (`registered_value.domain`)."""
    prog = P.program_digest(program)
    out = {}
    for v, w in values.items():
        n, b = shape(program, v, w, (schemas or {}).get(v, RV.SCHEMA)), width(program, v)
        out[v] = {"leaves": n, "words": int(w), "word_bits": b,
                  "domain": RV.domain(prog, v, w, n, b, None if v in fixed else window).domain_id.hex()}
    return out


def sources(supplied: Any) -> dict[str, list[list[tuple[str, str]]]]:
    """The verifier's supplied values, `{value: [[source, ...], ...]}` (a row a list of sources, each one of `SOURCES`
    naming a string), as `{value: [[(kind, name), ...], ...]}`; ValueError on any other form, which the Lean verifier
    refuses too (`OneStage.paramsOfJson`)."""
    if not isinstance(supplied, Mapping):
        raise ValueError(f"supplied values are an object of rows, not {supplied!r}")
    out: dict[str, list[list[tuple[str, str]]]] = {}
    for v, rows in supplied.items():
        if not (isinstance(v, str) and isinstance(rows, (list, tuple)) and all(isinstance(r, (list, tuple)) for r in rows)):
            raise ValueError(f"supplied value {v!r}: its rows are not lists of sources")
        out[v] = []
        for r in rows:
            if not all(isinstance(s, Mapping) and len(s) == 1 and set(s) <= set(SOURCES) and isinstance(*s.values(), str)
                       for s in r):
                raise ValueError(f"supplied value {v!r}: a source is not one of {SOURCES} naming a string: {r!r}")
            out[v].append([next(iter(s.items())) for s in r])
    return out


def bytes_leaves(data: bytes, bits: int) -> list[int]:
    """`data` as leaves of `bits` bits: each big-endian 32-bit half, its `32 / bits` leaves least significant first (a
    digest is 16 leaves of 32 bits, or 512 of one, as PoUW's `seeds.leaves` reads it)."""
    if not (0 < bits <= 32 and 32 % bits == 0):
        raise ValueError(f"leaves of {bits} bits hold no whole 32-bit halves")
    if len(data) % 4:
        raise ValueError(f"{len(data)} bytes are no whole 32-bit halves")
    mask = (1 << bits) - 1
    return [(h >> (bits * j)) & mask for i in range(0, len(data), 4) for h in (int.from_bytes(data[i:i + 4], "big"),)
            for j in range(32 // bits)]


def half_leaves(data: bytes, bits: int) -> list[int]:
    """Two bytes, one big-endian 16-bit half, as its ``16 / bits`` leaves of `bits` bits least significant first: the
    row Lean's `SecondStage.VERDICT_ONE_LEAF` fills, one 16-bit leaf (`Supplied.halfLeaves`)."""
    if not (0 < bits <= 16 and 16 % bits == 0):
        raise ValueError(f"leaves of {bits} bits hold no whole 16-bit half")
    if len(data) != 2:
        raise ValueError(f"{len(data)} bytes are not one 16-bit half")
    h = int.from_bytes(data, "big")
    return [(h >> (bits * j)) & ((1 << bits) - 1) for j in range(16 // bits)]


def le_leaves(data: bytes, bits: int) -> list[int]:
    """§5.4's `le` (`Supplied.leLeaves`, `verity.circuits.supplied.leaves`)."""
    return SU.leaves("le", data, bits)


def leaves_of(data: bytes, bits: int) -> list[int]:
    """A row's bytes as the parameter's leaves of `bits` bits: one 16-bit half's (`half_leaves`) where it is two bytes,
    else its 32-bit halves' (`bytes_leaves`; `Supplied.leavesOf`)."""
    return half_leaves(data, bits) if len(data) == 2 else bytes_leaves(data, bits)


def digests(t: Any) -> bool:
    """A parameter whose leaves are whole `Digest`s, eight SHA-512 words of 64 one-bit leaves each (`DIGEST`), or an array
    of them."""
    return t == DIGEST or (isinstance(t, Array) and t.elem == DIGEST)


def digest_leaves(data: bytes) -> list[int]:
    """`data` as whole `Digest`s' leaves: each big-endian 64-bit word's 64 bits, least significant first
    (`gates.rowleaf.digest_bits`)."""
    if len(data) % 64:
        raise ValueError(f"{len(data)} bytes are no whole digests")
    return [(h >> j) & 1 for i in range(0, len(data), 8) for h in (int.from_bytes(data[i:i + 8], "big"),)
            for j in range(64)]


def bit_words(row: Sequence[int], bits: int, word_bits: int) -> list[int]:
    """A row of `bits`-bit leaves as a bit row's `word_bits`-bit leaves: its bits in order, each leaf's least significant
    first (`row_leaves`' packing), so a `Digest`'s 512 leaves are 32 u16 words."""
    if len(row) * bits % word_bits:
        raise ValueError(f"{len(row)} leaves of {bits} bits are no whole {word_bits}-bit words")
    acc = 0
    for i, v in enumerate(row):
        if not 0 <= int(v) < 1 << bits:
            raise ValueError(f"leaf {v} does not fit {bits} bits")
        acc |= int(v) << (bits * i)
    return [(acc >> (word_bits * j)) & ((1 << word_bits) - 1) for j in range(len(row) * bits // word_bits)]


def log_base(log: Sequence[Any]) -> int:
    """The first position after the log's records: the sum of their populations, every proof block of each one position."""
    base = 0
    for e in log:
        n = e.get("population") if isinstance(e, Mapping) else None
        if type(n) is not int or n < 0:
            raise ValueError(f"a record of the log has no population: {n!r}")
        base += n
    return base


def window_salt(coin: bytes, window: Any, binding: Mapping[str, str] | None) -> bytes:
    """The salt of `window` (a record's) under the verifier's coin and binding (`{"stream", "prover"}`, each lowercase
    hex): SHA-512 of `WINDOW_SALT`, the coin, the stream, the prover and the window's canonical JSON, NUL between them
    (only the coin, first, may hold a NUL). ValueError without a binding, or for a window that is no object of the
    canonical domain."""
    if binding is None:
        raise ValueError("a window salt is bound to a stream and a prover: the verifier holds no binding")
    if not isinstance(window, Mapping) or jcs_domain_error(dict(window)) is not None:
        raise ValueError(f"the record's window is no canonical JSON object: {window!r}")
    return hashlib.sha512(WINDOW_SALT.encode() + b"\0" + coin + b"\0" + binding["stream"].encode() + b"\0"
                          + binding["prover"].encode() + b"\0" + canonical_json(dict(window))).digest()


def _hex_bytes(value: str, kind: str, name: str, text: Any) -> bytes:
    if not (isinstance(text, str) and _HEX.fullmatch(text) and len(text) % 2 == 0):
        raise ValueError(f"supplied value {value!r}: {kind} {name!r} is not lowercase hex bytes")
    return bytes.fromhex(text)


HELD_ROOT, HELD_DOMAIN = "held-root", "held-domain"


def _source_bytes(rec: Any, value: str, kind: str, name: str, supplied: Sequence[str], log: Sequence[Any],
                  binding: Mapping[str, str] | None, held: Mapping[str, Mapping[str, str]] | None = None) -> bytes:
    """A source's bytes: the root the record names for it (its last entry, as R3 and R7 read a name twice), the root or
    domain of a table the verifier holds (`held`: value to its `root` and `domain`, never the record's entry), the
    verifier's own, a position after the log's, or the verifier's coin bound to its binding and the record's window;
    ValueError where there is none."""
    rec = rec if isinstance(rec, Mapping) else {}
    if kind in (HELD_ROOT, HELD_DOMAIN):
        h = (held or {}).get(name)
        if not isinstance(h, Mapping):
            raise ValueError(f"supplied value {value!r}: the verifier holds no value {name!r}")
        return _hex_bytes(value, kind, name, h.get("root" if kind == HELD_ROOT else "domain"))
    if kind == "bytes":
        return _hex_bytes(value, kind, name, name)
    if kind == "window":
        return window_salt(_hex_bytes(value, kind, name, name), rec.get("window"), binding)
    if kind == "position":
        n = rec.get("population")
        if not (_DEC.fullmatch(name) and type(n) is int and int(name) < n):
            raise ValueError(f"supplied value {value!r}: position {name!r} is no unit of a population of {n!r}")
        p = log_base(log) + int(name)
        if p >= 1 << 64:
            raise ValueError(f"supplied value {value!r}: position {p} is no u64")
        return p.to_bytes(8, "big")
    field, key = ("roots", "port") if kind == "port" else ("registered", "value")
    if kind in ("value", "domain") and name in supplied:
        raise ValueError(f"supplied value {value!r} reads supplied value {name!r}'s root")
    es = rec.get(field)
    found = [e for e in (es if isinstance(es, list) else ()) if isinstance(e, Mapping) and e.get(key) == name]
    if not found:
        raise ValueError(f"supplied value {value!r}: the record names no {key} {name!r}")
    return _hex_bytes(value, kind, name, found[-1].get("domain" if kind == "domain" else "root"))


def supplied_rows(program: Program, value: str, rows: Sequence[Sequence[tuple[str, str]]], rec: Any,
                  supplied: Sequence[str] = (), log: Sequence[Any] = (),
                  binding: Mapping[str, str] | None = None,
                  held: Mapping[str, Mapping[str, str]] | None = None) -> list[list[int]]:
    """The rows the verifier fills a supplied value with, from `rec`, the log of this prover's earlier records, its
    binding (`window_salt`) and its own bytes: each row its sources' bytes, read as the parameter's leaves (`leaves_of`
    at its leaf width, or `digest_leaves` where its leaves are whole digests, `digests`). `supplied`: every supplied
    value's name, whose roots no row reads. ValueError where a source names nothing the record or the verifier holds, or
    the leaves are not one width dividing 32."""
    t = dict(program.fn.signature()[0]).get(value)
    w = None if t is None else t.homogeneous_width()
    if w is None:
        raise ValueError(f"{value!r} is not a parameter of the program whose leaves are one width")
    return [row_leaves_of(row, lambda k, n: _source_bytes(rec, value, k, n, supplied, log, binding, held), w, digests(t))
            for row in rows]


def row_leaves_of(row: Sequence[tuple[str, str]], source: Any, w: int, digest: bool) -> list[int]:
    """A supplied row's leaves: its sources' bytes (``source(kind, name)``) read by `le_leaves` where it leads with
    ``(LE, "")``, else as whole digests where ``digest``, else by `leaves_of`."""
    le = bool(row) and row[0][0] == LE
    data = b"".join(source(k, n) for k, n in (row[1:] if le else row))
    return le_leaves(data, w) if le else digest_leaves(data) if digest else leaves_of(data, w)


def _nat(x: Any) -> int | None:
    return x if type(x) is int and x >= 0 else None


def marks(desc: Any) -> dict[str, dict[str, Any]]:
    """The marks of a program descriptor's top-level `inputs` by input, each marked input's root nodes all
    `SuppliedInput<w>_v1` and no other input's one (`Supplied.marks`); none where it has no `inputs` list; ValueError
    otherwise, and on a function the registry lacks or a value that is not short."""
    def get(j: Any, k: str) -> Any:
        return j.get(k) if isinstance(j, Mapping) else None
    xs = get(desc, "inputs")
    if not isinstance(xs, list):
        return {}
    root = get(desc, "root")
    if not isinstance(root, str):
        raise ValueError("the program names no root")
    nodes = get(get(get(get(desc, "definitions"), root), "body"), "nodes")
    if not isinstance(nodes, list):
        raise ValueError("the program's root has no nodes")
    callees = get(desc, "callees")
    if not isinstance(callees, list):
        raise ValueError("the program has no callees")

    def callee(nd: Any) -> str:
        i = _nat(get(nd, "fn"))
        return callees[i] if i is not None and i < len(callees) and isinstance(callees[i], str) else ""
    out: dict[str, dict[str, Any]] = {}
    k = 0
    for e in xs:
        if not (isinstance(e, list) and e and isinstance(e[0], str)):
            raise ValueError("an input is not [name, nodes] or [name, nodes, mark]")
        name = e[0]
        n = _nat(e[1]) if len(e) > 1 else None
        if n is None:
            raise ValueError(f"input {name}: its nodes are not a natural number")
        ids = [callee(nd) for nd in nodes[k:k + n]]
        if len(ids) != n:
            raise ValueError(f"input {name}: the root has no nodes {k}..{k + n - 1}")
        supplied = [i.startswith("SuppliedInput") for i in ids]
        if len(e) == 2:
            if any(supplied):
                raise ValueError(f"input {name}: supplied input nodes and no mark naming their function")
        elif len(e) == 3:
            if not all(supplied):
                raise ValueError(f"input {name}: marked supplied, and its nodes are not all supplied inputs")
            out[name] = mark_of(name, e[2])
        else:
            raise ValueError(f"input {name}: an input is [name, nodes] or [name, nodes, mark]")
        k += n
    return out


def mark_of(name: str, m: Any) -> dict[str, Any]:
    """A top-level input's mark (`Supplied.markOf`): `{"supplied", "args", "layout", "bits"}`, its function one of the
    registry's, its value short."""
    if not (isinstance(m, Mapping) and set(m) == {"supplied", "args", "layout", "bits"}):
        raise ValueError(f"supplied input {name}: its mark is {{supplied, args, layout, bits}}, not {m!r}")
    if not (isinstance(m["supplied"], str) and isinstance(m["args"], Mapping) and isinstance(m["layout"], str)
            and _nat(m["bits"]) is not None):
        raise ValueError(f"supplied input {name}: a mark's function and layout are strings, its args an object and its "
                         "bits a natural number")
    if m["supplied"] not in SU.FUNCTIONS:
        raise ValueError(f"supplied input {name}: {m['supplied']} is no supplied function")
    if m["layout"] not in SU.LAYOUTS:
        raise ValueError(f"supplied input {name}: {m['layout']} is no layout")
    if not 0 < m["bits"] <= SHORT_BITS:
        raise ValueError(f"supplied input {name}: {m['bits']} bits is no short value (at most {SHORT_BITS}): a value that "
                         "grows with the population is computed in C")
    return dict(m)


def _arg(name: str, m: Mapping[str, Any], a: str, kind: type) -> Any:
    if set(m["args"]) != {a}:
        raise ValueError(f"supplied input {name}: {m['supplied']} takes the args [{a}]")
    x = m["args"][a]
    if not (type(x) is kind and (x >= 0 if kind is int else x != "")):
        raise ValueError(f"supplied input {name}: {m['supplied']}'s {a} is no {'natural' if kind is int else 'string'}")
    return x


def _coin(coins: Mapping[str, bytes], name: str, m: Mapping[str, Any]) -> bytes:
    if m["args"]:
        raise ValueError(f"supplied input {name}: {m['supplied']} takes no args")
    if m["supplied"] not in coins:
        raise ValueError(f"supplied input {name}: no statement: the verifier holds no coin for {m['supplied']}")
    return bytes(coins[m["supplied"]])


def _no_statement(name: str, f: str, why: str) -> ValueError:
    return ValueError(f"supplied input {name}: no statement: {f}'s {why}")


def _canonical_numbers(b: bytes) -> bool:
    """Every number in the JSON text ``b`` (read as JSON already) is ``0`` or digits with no leading zero (`Supplied.lean`'s
    `canonicalNumbers`): no sign, fraction or exponent. A number is a run of ``0-9 + - . e E`` that starts outside a string at
    one of ``0-9 + - .``."""
    def ok(t: bytes) -> bool:
        return t == b"0" or (49 <= t[0] <= 57 and t.isdigit())
    in_str = esc = False
    tok = bytearray()
    for c in b:
        if in_str:
            if esc:
                esc = False
            elif c == 92:
                esc = True
            elif c == 34:
                in_str = False
            continue
        start = 48 <= c <= 57 or c in (43, 45, 46)
        if tok and (start or c in (101, 69)):
            tok.append(c)
            continue
        if tok:
            if not ok(bytes(tok)):
                return False
            tok.clear()
        if c == 34:
            in_str = True
        elif start:
            tok.append(c)
    return not tok or ok(bytes(tok))


def _held_object(tbl: Mapping[str, bytes], kind: str, name: str, f: str) -> Mapping[str, Any]:
    """What the verifier holds under function ``f`` in ``tbl`` (``kind``, "commitment" or "coin"), as a JSON object of
    canonical numbers; "no statement" for what it holds none of, or that is no such object."""
    if f not in tbl:
        raise ValueError(f"supplied input {name}: no statement: the verifier holds no {kind} for {f}")
    raw = bytes(tbl[f])

    def not_json(c: str) -> Any:
        raise ValueError(c)
    try:
        doc = json.loads(raw.decode("utf-8"), parse_constant=not_json)
    except ValueError:
        raise _no_statement(name, f, f"{kind} is no JSON") from None
    if not _canonical_numbers(raw):
        raise _no_statement(name, f, f"{kind} holds a number that is no canonical natural")
    if not isinstance(doc, dict):
        raise _no_statement(name, f, f"{kind} is no object")
    return doc


def _session_fields(name: str, f: str, rel: Mapping[str, Any], hexes: Sequence[str],
                    nats: Sequence[tuple[str, int]]) -> dict[str, Any]:
    """The held commitment's fields ``hexes`` (lowercase hex of whole bytes, the session id 32 bytes) and ``nats`` (naturals
    below ``2 ** bits``)."""
    c = rel.get("commitment")
    if not isinstance(c, dict):
        raise _no_statement(name, f, "commitment holds no commitment object")
    out: dict[str, Any] = {}
    for k in hexes:
        v = c.get(k)
        if not (isinstance(v, str) and _EVEN_HEX.fullmatch(v)):
            raise _no_statement(name, f, f"commitment's {k} is no lowercase hex")
        out[k] = bytes.fromhex(v)
        if k == "session" and len(out[k]) != 32:
            raise _no_statement(name, f, f"commitment's session is {len(out[k])} bytes, not 32")
    for k, bits in nats:
        v = c.get(k)
        if type(v) is not int or not 0 <= v < 1 << bits:
            raise _no_statement(name, f, f"commitment's {k} is no u{bits}")
        out[k] = v
    return out


def _round_coin(coins: Mapping[str, bytes], commitments: Mapping[str, bytes], name: str,
                m: Mapping[str, Any]) -> tuple[list[tuple[str, str]], int]:
    """Round ``round``'s index, the last opening of the round that the coin holds, opened under the coin root of the
    commitment the verifier holds (never one the coin carries) with the coin key the coin releases, which hashes to the
    commitment's ``coin_key_sha512``."""
    f = m["supplied"]
    j = _arg(name, m, "round", int)
    rel = _held_object(commitments, "commitment", name, f)
    c = _session_fields(name, f, rel, ("session", "coin_root", "coin_key_sha512"), (("blocks", 64), ("k", 32)))
    if j >= 1 << 32:
        raise _no_statement(name, f, f"round {j} is no u32")
    if j >= c["k"]:
        raise _no_statement(name, f, f"round {j} is no round of the session's {c['k']}")
    coin = _held_object(coins, "coin", name, f)
    if set(coin) - {"openings", "coin_key"}:
        raise _no_statement(name, f, "coin holds more than its openings and its key: the coin root is the verifier's own")
    key = coin.get("coin_key")
    if not (isinstance(key, str) and _EVEN_HEX.fullmatch(key)):
        raise _no_statement(name, f, "coin's coin_key is no lowercase hex")
    key = bytes.fromhex(key)
    if hashlib.sha512(key).digest() != c["coin_key_sha512"]:
        raise _no_statement(name, f, "coin's key is not the one the commitment holds the hash of")
    ops = coin.get("openings")
    if not isinstance(ops, list):
        raise _no_statement(name, f, "coin holds no openings list")
    o = next((o for o in reversed(ops) if isinstance(o, dict) and type(o.get("j")) is int and o["j"] == j), None)
    if o is None:
        raise _no_statement(name, f, f"coin holds no opening of round {j}")
    index, salt, path = o.get("index"), o.get("salt"), o.get("path")
    if not (type(index) is int and index >= 0 and isinstance(salt, str) and _EVEN_HEX.fullmatch(salt)
            and isinstance(path, list) and all(isinstance(p, str) and _EVEN_HEX.fullmatch(p) for p in path)):
        raise _no_statement(name, f, f"opening of round {j} is malformed")
    if index >= 1 << 64:
        raise _no_statement(name, f, f"round {j}'s index {index} is no u64")
    if index >= c["blocks"]:
        raise _no_statement(name, f, f"round {j}'s index {index} is no block of the session's {c['blocks']}")
    if not SC.verify(c["session"], c["coin_root"], key, c["blocks"], c["k"], j, index, bytes.fromhex(salt),
                     [bytes.fromhex(p) for p in path]):
        raise _no_statement(name, f, f"round {j}'s index {index} is not its committed coin")
    return [("bytes", (j.to_bytes(4, "big") + index.to_bytes(8, "big")).hex())], 12


def _header(coins: Mapping[str, bytes], commitments: Mapping[str, bytes], name: str,
            m: Mapping[str, Any]) -> tuple[list[tuple[str, str]], int]:
    """The held commitment's signed header (the coin key by its hash) with ``"ended": "complete"`` and ``"rounds": k``, which
    the ``ended`` and ``rounds`` the session's release holds (the coin held under the function) are compared with, so a late
    or short session has no statement."""
    f = m["supplied"]
    want = _arg(name, m, "head", int)
    rel = _held_object(commitments, "commitment", name, f)
    c = _session_fields(name, f, rel, ("session", "coin_root", "coin_key_sha512", "binding"),
                        (("blocks", 64), ("k", 32), ("limit_ns", 64), ("block_bytes", 64)))
    released = _held_object(coins, "coin", name, f)
    ended, rounds = released.get("ended"), released.get("rounds")
    if not isinstance(ended, str):
        raise _no_statement(name, f, "coin holds no ended string")
    if type(rounds) is not int or not 0 <= rounds < 1 << 32:
        raise _no_statement(name, f, "coin's rounds is no u32")
    if ended != SC.COMPLETE:
        raise _no_statement(name, f, "session did not end complete")
    if rounds != c["k"]:
        raise _no_statement(name, f, f"session has {rounds} rounds, not its {c['k']}")
    head = SC.header(c["session"], c["coin_root"], c["coin_key_sha512"], c["binding"], c["blocks"], c["k"], c["limit_ns"],
                     c["block_bytes"], SC.COMPLETE, c["k"])
    if len(head) != want:
        raise _no_statement(name, f, f"header is {len(head)} bytes, not {want}")
    row = head + bytes(len(head) % 2)
    return [("bytes", row.hex())], len(row)


UNPINNED = "key is not one the configuration pins for the memory attester"
BAD_SIGNATURE = "signature does not verify over the message under the pinned key"


def _pinned_key(rel: Mapping[str, Any], name: str, f: str, pinned: Collection[bytes]) -> bytes:
    """The held commitment's LMS public key, refused unless its SHA-512 is one of ``pinned``, the configuration's pins of the
    memory attester's keys: the session's own object never names the key it is checked under."""
    key = rel.get("key")
    if not (isinstance(key, str) and _EVEN_HEX.fullmatch(key)):
        raise _no_statement(name, f, "commitment's key is no lowercase hex")
    k = bytes.fromhex(key)
    if hashlib.sha512(k).digest() not in pinned:
        raise _no_statement(name, f, UNPINNED)
    return k


def _context(commitments: Mapping[str, bytes], name: str, m: Mapping[str, Any],
             pinned: Collection[bytes]) -> tuple[list[tuple[str, str]], int]:
    """The held commitment's session id and the first 16 bytes of the pinned key, as the signature's message hashes them."""
    f = m["supplied"]
    if m["args"]:
        raise ValueError(f"supplied input {name}: {f} takes no args")
    rel = _held_object(commitments, "commitment", name, f)
    session = _session_fields(name, f, rel, ("session",), ())["session"]
    ident = _pinned_key(rel, name, f, pinned)[:16]
    if len(ident) != 16:
        raise _no_statement(name, f, f"key is {len(ident)} bytes, not an LMS identifier's 16")
    ctx = SC.signed_context(ident, session)
    return [("bytes", ctx.hex())], len(ctx)


def _message(coins: Mapping[str, bytes], commitments: Mapping[str, bytes], name: str, m: Mapping[str, Any],
             pinned: Collection[bytes]) -> tuple[list[tuple[str, str]], int]:
    """The ``message`` the session releases (the coin held under the function, `{message, signature}`), the 64-byte salted
    commitment the memory attester signed, released in the clear with its ``signature``, which verifies over it under the
    pinned key the commitment holds, at the deployed parameters (`lms.DEPLOYED`)."""
    f = m["supplied"]
    if m["args"]:
        raise ValueError(f"supplied input {name}: {f} takes no args")
    rel = _held_object(commitments, "commitment", name, f)
    session = _session_fields(name, f, rel, ("session",), ())["session"]
    key = _pinned_key(rel, name, f, pinned)
    released = _held_object(coins, "coin", name, f)
    msg = released.get("message")
    if not (isinstance(msg, str) and _EVEN_HEX.fullmatch(msg)):
        raise _no_statement(name, f, "coin's message is no lowercase hex")
    if len(msg) != 128:
        raise _no_statement(name, f, f"message is {len(msg) // 2} bytes, not 64")
    sig = released.get("signature")
    if not (isinstance(sig, str) and _EVEN_HEX.fullmatch(sig)):
        raise _no_statement(name, f, "coin's signature is no lowercase hex")
    try:
        ok = len(key) == 16 + lms.DEPLOYED.N and lms.verify(
            lms.DEPLOYED, key, lms.memory_attester_message(key[:16], session, bytes.fromhex(msg)),
            lms.Signature.decode(lms.DEPLOYED, bytes.fromhex(sig)))
    except ValueError:
        raise _no_statement(name, f, BAD_SIGNATURE) from None
    if not ok:
        raise _no_statement(name, f, BAD_SIGNATURE)
    return [("bytes", msg)], 64


def body(coins: Mapping[str, bytes], commitments: Mapping[str, bytes], name: str, m: Mapping[str, Any],
         pinned: Collection[bytes] = ()) -> tuple[list[tuple[str, str]], int]:
    """A supplied function's body (`Supplied.body`): its value's sources and its length in bytes; ValueError naming "no
    statement" where the verifier can't compute it. ``coins`` and ``commitments`` are what the verifier holds of each function
    as a coin and as a commitment, by function name. The storage session's four functions read the session in two entries by
    when each of its parts exists. The commitment, held before the first coin, is a JSON object `{commitment, key, place}`:
    the header's fields (`SessionCommitment.header`: the session id, the coin root, the hash of the coin key, the blocks, `k`,
    the limit, the answer's size, the binding), the LMS public key's hex and the place it was registered at. The coin,
    released after the session, is `{openings, coin_key}` for `pous.session.round_coin/v1` (`CoinOpening.to_json` and the
    coin key itself, which hashes to the commitment's), `{ended, rounds}` for `pous.session.header/v1` and
    `{message, signature}` for `pous.session.message/v1`.
    `pous.session.round_coin/v1` is `u32be(round) ‖ u64be(index)` of the round's last opening, which opens its committed coin
    below the commitment's blocks; `pous.session.header/v1` the header of a session that ended complete with all its rounds in
    `head` bytes, padded to whole u16 words; `pous.session.context/v1` the 106 bytes the signature's message hashes before the
    record; `pous.session.message/v1` the 64-byte message m the signature signs. The context and the message read the
    commitment's key only once its SHA-512 is one of ``pinned``, the configuration's pins of the memory attester's keys
    (`UNPINNED`): no body takes the key the session's own object names. The message is read only once the coin's
    ``signature`` verifies over it under that key (`BAD_SIGNATURE`): m and the signature are public, and the signature's
    leaf q shows how many signatures the attester has made. Each number they read is canonical (`_canonical_numbers`) and
    within its field's width."""
    f = m["supplied"]
    if f == "pous.session.round_coin/v1":
        return _round_coin(coins, commitments, name, m)
    if f == "pous.session.header/v1":
        return _header(coins, commitments, name, m)
    if f == "pous.session.context/v1":
        return _context(commitments, name, m, pinned)
    if f == "pous.session.message/v1":
        return _message(coins, commitments, name, m, pinned)
    if f == "const.u16/v1":
        x = _arg(name, m, "value", int)
        if x >= 1 << 16:
            raise ValueError(f"supplied input {name}: no statement: {x} is no u16")
        return [("bytes", x.to_bytes(2, "little").hex())], 2
    if f == "pous.setup.salt/v1":
        c = _coin(coins, name, m)
        if len(c) != 24:
            raise ValueError(f"supplied input {name}: no statement: {f}'s coin is {len(c)} bytes, not 24")
        return [("bytes", c.hex())], 24
    if f == "pouw.pearlplus.salt/v1":
        c = _coin(coins, name, m)
        if len(c) != 32:
            raise ValueError(f"supplied input {name}: no statement: {f}'s coin is {len(c)} bytes, not 32")
        return [("bytes", c.hex())], 32
    if f == "window.salt/v1":
        c = _coin(coins, name, m)
        if not c:
            raise ValueError(f"supplied input {name}: no statement: {f}'s coin is no bytes")
        return [("window", c.hex())], 64
    if f == "registered.root/v1":
        return [(HELD_ROOT, _arg(name, m, "value", str))], 64
    if f == "registered.domain/v1":
        return [(HELD_DOMAIN, _arg(name, m, "value", str))], 64
    if f == "record.root/v1":
        return [("port", _arg(name, m, "port", str))], 64
    if f == "log.base/v1":
        if m["args"]:
            raise ValueError(f"supplied input {name}: {f} takes no args")
        return [("position", "0")], 8
    raise ValueError(f"supplied input {name}: no statement: {f}'s body is not the verifier's yet")


def of_mark(coins: Mapping[str, bytes], commitments: Mapping[str, bytes], name: str, m: Mapping[str, Any],
            pinned: Collection[bytes] = ()) -> list[list[tuple[str, str]]]:
    """A supplied input's one row of sources (`Supplied.ofMark`): its body's, led by ``(LE, "")`` where its layout is
    `le`."""
    sources, n = body(coins, commitments, name, m, pinned)
    if 8 * n != m["bits"]:
        raise ValueError(f"supplied input {name}: {m['supplied']}'s value is {8 * n} bits, and the mark says {m['bits']}")
    return [([(LE, "")] if m["layout"] == "le" else []) + sources]


def marked_rows(program: Program, registered: Mapping[str, int], supplied: Mapping[str, Any],
                coins: Mapping[str, str] | None, commitments: Mapping[str, str] | None = None,
                released: Mapping[str, int] | None = None, records: int = 0,
                pinned: Collection[bytes] = ()) -> dict[str, list[list[tuple[str, str]]]]:
    """The rows of `program`'s supplied inputs (`marks`), each its function's value under the verifier's `coins` and
    `commitments` (`{function: hex}` each: what it holds of a function as a coin and as a released commitment, which the
    order rule counts the first of alone; `OneStage.markedRows`): ValueError where the verifier registers no read of the input, where
    `supplied` (the scheme's) has rows for it too, or where its layout is `digest` and its leaves aren't whole digests or
    the other way round; and refused by the order rule (`order`) at `released`, the verifier's place of each coin, after
    `records` records of the log; ``pinned``, the SHA-512s of the memory attester's keys the configuration pins."""
    got = marks(json.loads(b"".join(descriptor_chunks(program))))
    if not isinstance(coins or {}, Mapping):
        raise ValueError("coins are an object of hex by supplied function")
    held = {}
    for f, h in (coins or {}).items():
        if f not in SU.FUNCTIONS:
            raise ValueError(f"coins: {f} is no supplied function")
        if not (isinstance(h, str) and re.fullmatch(r"(?:[0-9a-f]{2})*", h)):
            raise ValueError(f"coins: {f}'s coin is not hex")
        held[f] = bytes.fromhex(h)
    if not isinstance(commitments or {}, Mapping):
        raise ValueError("commitments are an object of hex by supplied function")
    commit = {}
    for f, h in (commitments or {}).items():
        if f not in SU.FUNCTIONS:
            raise ValueError(f"commitments: {f} is no supplied function")
        if not (isinstance(h, str) and re.fullmatch(r"(?:[0-9a-f]{2})*", h)):
            raise ValueError(f"commitments: {f}'s commitment is not hex")
        commit[f] = bytes.fromhex(h)
    # A decoded Program has no `fn`, only the lowered root, whose parameters are the same names.
    params = dict(getattr(program, "fn", program.root).signature()[0]) if got else {}
    out = {}
    for v, m in got.items():
        if v not in registered:
            raise ValueError(f"supplied input {v}: the verifier registers no read of it: a supplied input is read only "
                             "through a port of the verifier's own registered reads")
        if v in supplied:
            raise ValueError(f"supplied input {v}: C's mark names its function, and the scheme supplies rows for it too")
        if v not in params:
            raise ValueError(f"supplied input {v} is not a parameter of the program")
        if (m["layout"] == "digest") != digests(params[v]):
            raise ValueError(f"supplied input {v}: layout {m['layout']} reads whole digests only into a parameter of "
                             "whole digests")
        out[v] = of_mark(held, commit, v, m, pinned)
    order(got, out, held, commit, released, records)
    return out


def _commitment_place(c: str, b: bytes) -> int:
    """The place `c`'s held commitment `b` was registered at: its JSON's ``place``, a natural; ValueError for a text that is
    no JSON object of one (`Supplied.commitmentPlace`)."""
    why = f"released: the verifier names no place of {c}'s commitment, which a coin it holds follows"
    try:
        doc = json.loads(b.decode("utf-8"))
    except ValueError as e:
        raise ValueError(why) from e
    if (p := _nat(doc.get("place")) if isinstance(doc, dict) else None) is None:
        raise ValueError(why)
    return p


def order(marked: Mapping[str, Mapping[str, Any]], rows: Mapping[str, list[list[tuple[str, str]]]],
          held: Mapping[str, bytes], commit: Mapping[str, bytes], released: Mapping[str, int] | None, records: int) -> None:
    """§5.4's order rule at registration (`Supplied.order`): every commitment C's marks read is registered before every
    coin they read, `E_SUPPLIED_ORDER` naming the late ones. A mark reads a coin where its function's value is one the
    verifier holds (`held`), at its place in `released` (`{function: n}`: the coin was released after n records, this
    one the `records`-th, its commitments before any coin it reads); its body's sources read the commitments: a
    ``value``'s or ``port``'s root this record's, at place `records`, and ``position`` the log's, at `records` - 1.
    ValueError where `released` is not an object of a place by supplied function, or names no place of a coin read
    beside a commitment. A held commitment (`commit`) is no coin, but a coin follows it: the ``place`` its JSON carries
    (where the verifier registered it) is before the place of every coin a mark reads, `E_SUPPLIED_ORDER` naming the
    commitments at or after it, and refused where a commitment has no place or the coin has none in `released`."""
    if released is not None and not (isinstance(released, Mapping) and all(
            f in SU.FUNCTIONS and _nat(n) is not None for f, n in released.items())):
        raise ValueError("released is an object of a place by supplied function")
    reads: dict[str, int] = {}
    for rs in rows.values():
        for k, n in (s for r in rs for s in r):
            at = records if k in ("value", "port") else records - 1 if k == "position" and records else None
            if at is not None:
                name = "log" if k == "position" else n
                reads[name] = max(reads.get(name, at), at)
    cms = sorted({m["supplied"] for m in marked.values() if m["supplied"] in commit})
    cos = sorted({m["supplied"] for m in marked.values() if m["supplied"] in held})
    if cms and cos:
        placed = {}
        for c in cms:
            placed[f"commitment {c}"] = _commitment_place(c, commit[c])
        for f in cos:
            if f not in (released or {}):
                raise ValueError(f"released: the verifier names no place of {f}'s coin, which follows a commitment it holds")
            SU.check_order(placed, released[f], f)
    for f in cos:
        if not reads:
            return
        if f not in (released or {}):
            raise ValueError(f"released: the verifier names no place of {f}'s coin, which a supplied value reads beside "
                             "commitments")
        SU.check_order(reads, released[f], f)


def cut(got: list[list[int]], n: int, k: int) -> list[list[int]]:
    """`got`'s one row of `n · k` leaves as `n` rows of `k` in order (`Supplied.cut`), where `n > 1`; else `got`."""
    if len(got) == 1 and n > 1 and len(got[0]) == n * k:
        return [got[0][i * k:(i + 1) * k] for i in range(n)]
    return got


def _in_format(program: Program, value: str, got: list[list[int]], word_bits: int, schema: str) -> list[list[int]]:
    """Supplied rows of the parameter's leaves as rows of `schema`: themselves under `SCHEMA`, else each packed into
    `word_bits`-bit leaves (`bit_words`)."""
    if RV.known_schema(schema) == RV.SCHEMA:
        return got
    w = dict(program.fn.signature()[0])[value].homogeneous_width()
    return [bit_words(r, w, word_bits) for r in got]


def supplied_root(program: Program, value: str, rows: Sequence[Sequence[tuple[str, str]]], rec: Any,
                  want: Mapping[str, Any], supplied: Sequence[str] = (), log: Sequence[Any] = (),
                  prog: str | None = None, binding: Mapping[str, str] | None = None, schema: str = RV.SCHEMA,
                  held: Mapping[str, Mapping[str, str]] | None = None) -> str:
    """The root R7 holds a supplied value's entry to: its rows (`supplied_rows`), as many as the verifier's own entry's
    `leaves` (`registered_entries`), each the parameter's leaves of one row of its `words` leaves in `schema`
    (`row_leaves`), committed in `schema` under `registered_value.supplied_salts` at its `word_bits`; a value given as
    one row of all those leaves (a mark's, `of_mark`) is cut into them in order (`cut`). `prog`: `program`'s digest
    (`P.program_digest`), when the caller holds it."""
    given = supplied_rows(program, value, rows, rec, supplied, log, binding, held)
    k = row_leaves(program, value, want["words"], schema)
    got = cut(given, want["leaves"], k)
    if len(got) != want["leaves"] or any(len(r) != k for r in got):
        raise ValueError(f"supplied value {value!r}: rows of {[len(r) for r in got]} leaves, not {want['leaves']} of "
                         f"{k}")
    prog = prog or P.program_digest(program)
    return RV.Registered.commit(prog, value, _in_format(program, value, got, want["word_bits"], schema),
                             RV.supplied_salts(prog, value, len(got)), want["word_bits"], schema).root.hex()


def supplied_value(program: Program, value: str, rows: Sequence[Sequence[tuple[str, str]]], rec: Any,
                   supplied: Sequence[str] = (), log: Sequence[Any] = (), prog: str | None = None,
                   binding: Mapping[str, str] | None = None, schema: str = RV.SCHEMA,
                   held: Mapping[str, Mapping[str, str]] | None = None) -> RV.Registered:
    """The prover's side of a supplied value: the rows the verifier fills from `rec` (a record without this value's own
    entry will do: no row reads it), the log and the binding under the public salts, in `schema` (the row format the
    verifier names for it), whose `entry()` the record registers and whose `statement` opens a proof block's reads like
    any registered value's."""
    got = supplied_rows(program, value, rows, rec, supplied, log, binding, held)
    prog, b = prog or P.program_digest(program), width(program, value)
    return RV.Registered.commit(prog, value, _in_format(program, value, got, b, schema),
                             RV.supplied_salts(prog, value, len(got)), b, schema)


def reads_file(program: Program, query: Mapping[str, Any], values: Mapping[str, int], roots: Mapping[str, str],
               ports: Mapping[str, str] | None = None, domains: Mapping[str, str] | None = None,
               schemas: Mapping[str, str] | None = None, window: str | None = None,
               fixed: Sequence[str] = (), positions: bool = True, template: str | None = None) -> dict[str, Any]:
    """The Lean verifier's `--registered` file, from the verifier's own program and query: per registered port, the value
    it reads, the value's leaf count, row width (`words` leaves of `word_bits`) and domain (`registered_entries`), each
    proof block's position (`reads`; without `positions`, none, in `READS_V2`, Flock's `Registered.FORMAT`, whose
    verifier derives every position from C), and the value's root from `roots` (`{value: root}`, of a docket the verifier
    accepted). `ports` names a template parameter's circuit port when they differ, and such a port names its parameter
    (`PARAM_KEY`); two reads it names one port are refused,
    as the Lean verifier refuses a file naming a port twice. `domains` (`{value: domain}`) gives
    an output value's domain, its earlier record's port's, which the docket's R7 held its entry to
    (`registration.check`'s `outputs`). `schemas` (`{value: schema}`) gives a value's row format when it is not
    `SCHEMA` (`row_schema`, `schema_entry`): the format of the circuit port that reads it, which the Lean verifier holds
    the port to. `window` and `fixed`: the window's open root and the values held across windows, as R7 took them
    (`registered_entries`). `template` makes it the file of that template's statements (`reads`' `template`), the one
    Lean holds a statement of it to: a port of a parameter its template does not have is refused (`OneCircuit.regRow`),
    and so is a `ports` entry naming a parameter at which `template` reads no registered value."""
    mine = registered_entries(program, values, schemas, window, fixed)
    out, read_as = {}, {}
    got = reads(program, query, values, schemas, template)
    if template is not None and (stray := sorted(set(ports or {}) - set(got))):
        raise ValueError(f"ports names parameters {stray}, and {template} reads no registered value at a parameter of "
                         "those names: a port reads its own template's parameter")
    for name, r in got.items():
        root = roots.get(r["value"])
        if root is None:
            raise ValueError(f"parameter {name!r} reads {r['value']!r}, whose registered root the verifier does not hold")
        e = mine[r["value"]]
        port = (ports or {}).get(name, name)
        if port in out:
            raise ValueError(f"port {port!r} names two reads: {read_as[port]!r} of {out[port]['value']!r} and {name!r} of "
                             f"{r['value']!r}: each read is its own port")
        read_as[port] = name
        out[port] = {"value": r["value"], "root": root, "leaves": e["leaves"], "words": e["words"],
                     "word_bits": e["word_bits"], "domain": (domains or {}).get(r["value"], e["domain"]),
                     **RV.schema_entry((schemas or {}).get(r["value"], RV.SCHEMA)),
                     **({"reads": list(r["rows"])} if positions else {}), **({PARAM_KEY: name} if port != name else {})}
    return {"format": READS if positions else READS_V2, "ports": out}


def bind_ports(file: Mapping[str, Any], ports: Iterable[str], roots: Mapping[str, Mapping[str, str]]) -> dict[str, Any]:
    """``file``, a `READS_V1` ``--registered`` file, with each of the statement's ``ports`` that it names no registered port
    for bound to the window's committed values (`BOUND`, `Flock.RegBound`): the root and domain of its rows in ``roots``
    (``{port: {"root", "domain"}}``, the window's docket, which the verifier accepted before any coin), which the
    Lean verifier holds the public header's to. Lean refuses a bound port the window's registered circuit record declares
    registered, and a file that binds ports without that record (`--interface`). A file of another format, or one with no
    other port, is returned as it is: Lean holds it to name every port. ValueError for a port ``roots`` holds no root and
    domain for."""
    free = sorted(p for p in ports if p not in file["ports"])
    if file.get("format") != READS_V1 or not free:
        return dict(file)
    missing = [p for p in free if not isinstance(roots.get(p), Mapping) or not {"root", "domain"} <= set(roots[p])]
    if missing:
        raise ValueError(f"ports {missing} are neither registered ports of the file nor ports the window's registration "
                         "holds a root and domain for")
    return {**file, BOUND: {p: {"root": roots[p]["root"], "domain": roots[p]["domain"]} for p in free}}


__all__ = ["BOUND", "DIGEST", "PARAM_KEY", "READS", "SCHEMA_SEG", "SCHEMA_V2",
           "SOURCES", "WINDOW_SALT", "bind_ports", "bit_words", "bytes_leaves", "cut", "digest_leaves", "digests",
           "half_leaves", "held_gates", "leaves_of", "log_base", "parameter_gates", "reads", "reads_file",
           "registered_entries", "row_leaves", "shape", "sources", "supplied_root", "supplied_rows",
           "supplied_value", "table", "verify_read", "width", "window_salt"]
