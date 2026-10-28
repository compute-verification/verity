"""Served trees as registered ports: the one form in which a served path registers what it served and has the Lean verifier
open it (`register_served`).

A *served tree* is a frame-v3-sha512 tree of `hm96-sha512/row/v1` rows that serving committed and that a proof block reads:
a template's operands, Dither's z/y leaves or receipt roots, H1's per-call A, y and z tables. Its root is a hiding
commitment, which binds nothing on its own: `EndToEndRegisteredRows` links a proof's value layer to a committed row only at
a registered read. So the service registers each served tree as a port of the verifier's own `--registered` file before
the session's statement and coin (`RowsSession.ServedRegistered`), and verifies with `flock-verify --zk --registered`;
`Flock.Guarantees.ServedRowsLinked` then gives the committed tree's row at every served read.

**The port form.** A served tree is the registered value of the program parameter whose rows it commits (`value`): its
`leaves` rows, each `words` leaves of `word_bits` bits (16 or 32), under `root` and `domain`. It enters the registration
record as one `registered` entry (`registered.ENTRY_KEYS`, plus `word_bits` when it is not 16), which R7 holds to the
verifier's own (`registration.check(..., registered=values, outputs=outputs, schemas=schemas, window_root=, held=)`), and
the verifier's
`--registered` file as one port per circuit port that reads it (`registered.reads_file`: the value, root, leaves, words,
word_bits, domain, and the row each proof block of the population reads, from the verifier's own program and query). Its domain
is one of three, and no other:

* the registered-value domain of `value` in the verifier's own program (`registered.registered_entries`), when serving
  committed the tree as a registered value (`Registered.commit`): under the window's open root R (`window_root`, from
  the challenger's signed open of the window) for a per-window value, and without it for a value held across windows
  (`held`, the weights), so a window-independent program registers each window's values under the same names;
* an earlier record's port's domain, when the tree is that port (`output`: `(record digest, port)`):
  the log's record with that digest names the port, its root and its leaf count, and the tree is that port's;
* an earlier record's registered value's domain, when the tree is that value (`reused`: `(record digest, value)`): the
  log's record with that digest registered the value with its root, domain, rows, words and word bits, and the tree is
  that entry, registered under no domain of its own (H1's `a/<K>@<s>` as a Commit's AROW, say).

Each tree names its row format (`row_schema`, the `--registered` port's key for it, one of `SCHEMAS`): the
frame-v3-sha512 leaf schema its leaves were hashed under, which must be the format of the circuit port that reads it
(`Flock.Registered.checkPort`). Absent means `hm96-sha512/row/v1` (`registered.SCHEMA`), so a v1 tree's port names none.
H1's and PoUW's served rows are `row/v2` and `row-seg/v1` bit rows: a row is exactly its `words` leaves of `word_bits`
bits, as many of the parameter's own leaves as those bits hold (`registered.row_leaves`; H1's A, K int8 codes, is K/2
u16 words), which sets the tree's rows and its domain. The record's `registered` entry names no format: R7 counts a
row's leaves in the format the verifier holds for the value (`registration.check(..., schemas=)`, the returned
`schemas`), and without it as `row/v1`, so it refuses a bit-row tree of leaves narrower than its words, such as H1's A.

**The program and the partition.** The positions are the verifier's: `reads` derives each proof block's row from the program and
the query it is given, the path's own public Program and its `Q_template_instance` query, never the universal unit's. The
Lean verifier derives the same proof blocks only from the same objects, so the session is verified with `--program` the
program's canonical encoding (`codec.encode_program`), `--partition` the returned `partition` (`partition.partition(program,
query)`), and `--unit-indices` the staged instances' proof blocks. Rows are fixed slots: a tree pads its unused rows with zeros,
so the positions are a function of the proof block index alone, and the layout is protocol structure.

`register_served` refuses (ValueError) a served tree whose value is not a parameter the program reads in whole rows, a
value twice, a root or domain that is not lowercase hex of 64 bytes, no rows or no leaves to a row, a width other than 16
or 32, a row format not in `SCHEMAS`, a bit row whose `words * word_bits` bits are no whole number of the parameter's
leaves or whose rows are not the parameter's, a domain that is none of the three above, an output whose record or port the log does
not hold or whose root, domain or rows are not the port's, a reused tree whose record or value the log does not hold or
whose root, domain, rows, words or word bits are not that entry's, a tree that is both an output and reused, a root other than the one an earlier record of the log
registered under the same domain (R7-held), without `window_root` a tree it doesn't hold that the log registered under
the same domain (`registration.unwindowed`: a fixed name served again needs the window's root), and a tree no proof block of the
population reads, which no proof would open.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from verity.ml.circuits.codec import canonical_json
from verity.ml.circuits.program import Program

from . import partition as P, registered as G, registration as R

_HEX64 = re.compile(r"[0-9a-f]{128}")
#: the row formats a served tree may name, each one the registered form reads
SCHEMAS = G.SCHEMAS


@dataclass(frozen=True)
class ServedTree:
    """One served tree: the program parameter whose rows it commits (`value`), its root and domain (lowercase hex, 64 bytes
    each), its rows (`leaves`), each `words` leaves of `word_bits` bits, its rows' format (`row_schema`, one of `SCHEMAS`), and,
    for an earlier run's output, `(record digest, port)`, or for an earlier record's registered value it reuses, `reused`:
    `(record digest, value)`."""

    value: str
    root: str
    leaves: int
    words: int
    domain: str
    word_bits: int = G.WORD_BITS
    output: tuple[str, str] | None = None
    row_schema: str = G.SCHEMA
    reused: tuple[str, str] | None = None

    @classmethod
    def of(cls, reg: G.Registered) -> "ServedTree":
        """A tree serving committed as a registered value (`Registered.commit`), under its registered-value domain."""
        e = reg.entry()
        return cls(e["value"], e["root"], e["leaves"], e["words"], e["domain"], reg.word_bits, row_schema=reg.schema)

    def entry(self) -> dict[str, Any]:
        """The registration record's `registered` entry."""
        return {"value": self.value, "root": self.root, "leaves": self.leaves, "words": self.words,
                **G.width_entry(self.word_bits), "domain": self.domain}


@dataclass(frozen=True)
class ServedRegistration:
    """What `register_served` returns. `registered`: the record's `registered` entries (`registration.record(...,
    registered=)`); `values`, `outputs`, `window_root` and `held`: R7's own copies (`registration.check(...,
    registered=values, outputs=outputs, window_root=window_root, held=held)`, `outputs` None when no tree is an earlier
    output); `schemas`: R7's row formats, `{value: schema}` for each tree whose rows are not `row/v1` rows
    (`registration.check(..., schemas=)`); `roots`: `{value: root}`; `file`: the `--registered` document, whose `ports` are
    the registered ports by circuit port; `partition`: the verifier's partition object, its `--partition` file."""

    registered: list[dict[str, Any]]
    values: dict[str, int]
    outputs: dict[str, dict[str, str]] | None
    roots: dict[str, str]
    file: dict[str, Any]
    partition: dict[str, Any]
    schemas: dict[str, str]
    window_root: str | None = None
    held: list[str] | None = None

    @property
    def ports(self) -> dict[str, dict[str, Any]]:
        return self.file["ports"]

    def file_bytes(self) -> bytes:
        """The `--registered` file as `flock-verify` reads it: canonical JSON."""
        return canonical_json(self.file)

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_bytes(self.file_bytes())
        return path


def _int(x: Any) -> bool:
    return type(x) is int


def _check_form(t: Any) -> None:
    if not isinstance(t, ServedTree):
        raise ValueError(f"a served tree is a ServedTree, not {t!r}")
    if not isinstance(t.value, str) or not t.value:
        raise ValueError(f"a served tree names the parameter whose rows it commits: {t.value!r}")
    for k in ("root", "domain"):
        if not (isinstance(getattr(t, k), str) and _HEX64.fullmatch(getattr(t, k))):
            raise ValueError(f"served tree {t.value!r}: its {k} is not lowercase hex of 64 bytes: {getattr(t, k)!r}")
    if not (_int(t.leaves) and t.leaves > 0):
        raise ValueError(f"served tree {t.value!r}: {t.leaves!r} leaves, and a tree has at least one")
    if not (_int(t.words) and t.words > 0):
        raise ValueError(f"served tree {t.value!r}: {t.words!r} words to a leaf, and a leaf has at least one")
    if not (_int(t.word_bits) and t.word_bits in G.WIDTHS):
        raise ValueError(f"served tree {t.value!r}: words of {t.word_bits!r} bits, not "
                         f"{' or '.join(map(str, G.WIDTHS))}")
    if t.row_schema not in SCHEMAS:
        raise ValueError(f"served tree {t.value!r}: rows of {t.row_schema!r}, not one of {SCHEMAS}")
    for o, what in ((t.output, "an output is (record digest, port)"), (t.reused, "reused is (record digest, value)")):
        if o is not None and not (isinstance(o, tuple) and len(o) == 2 and isinstance(o[0], str)
                                  and _HEX64.fullmatch(o[0]) and isinstance(o[1], str) and o[1]):
            raise ValueError(f"served tree {t.value!r}: {what}, not {o!r}")
    if t.output is not None and t.reused is not None:
        raise ValueError(f"served tree {t.value!r}: an earlier port or an earlier registered value, not both")


def register_served(program: Program, query: Mapping[str, Any], served: Sequence[ServedTree],
                    log: Sequence[Mapping[str, Any]] = (), ports: Mapping[str, str] | None = None,
                    window_root: str | None = None, held: Sequence[str] | None = None) -> ServedRegistration:
    """Register each served tree as a port of the verifier's own `--registered` file, in the module's port form: the record's
    `registered` entries, R7's copies, and the file, from the verifier's own program and query. `log`: this prover's earlier
    records, which an output tree's `(record digest, port)` and a reused tree's `(record digest, value)` name. `ports`: a template parameter's circuit port, where
    they differ (`registered.reads_file`). `window_root`: the window's open root, which every tree's registered-value
    domain binds but `held`'s, the values held across windows, and the outputs' and reused trees'. ValueError on anything the module's
    docstring refuses."""
    if not served:
        raise ValueError("no served tree: a registration with registered values registers at least one")
    for t in served:
        _check_form(t)
    names = [t.value for t in served]
    if len(set(names)) != len(names):
        raise ValueError(f"a value twice among the served trees: {sorted(v for v in set(names) if names.count(v) > 1)}")
    if held is not None and not all(isinstance(v, str) and v in names for v in held):
        raise ValueError(f"held values are served trees' values: {held!r}")
    values = {t.value: t.words for t in served}
    schemas = {t.value: t.row_schema for t in served if t.row_schema != G.SCHEMA}
    fixed = [*(held or ()), *(t.value for t in served if t.output is not None or t.reused is not None)]
    mine = G.registered_entries(program, values, schemas, G.window_root(window_root), fixed)
    log = list(log)
    again = R.unwindowed(program, values, schemas, window_root, fixed, log)
    logged = R.held_roots(log)
    outputs: dict[str, dict[str, str]] = {}
    domains: dict[str, str] = {}
    for t in served:
        want = mine[t.value]
        if t.leaves != want["leaves"] or t.word_bits != want["word_bits"]:
            raise ValueError(f"served tree {t.value!r}: {t.leaves} rows of {t.words} {t.word_bits}-bit leaves, and the "
                             f"program's parameter is {want['leaves']} rows of {want['words']} {want['word_bits']}-bit leaves")
        if t.reused is not None:
            record, value = t.reused
            got = R._earlier(log, record, "value", value, want)
            if got is None:
                raise ValueError(f"served tree {t.value!r}: the log holds no record {record[:16]}... registering {value!r} "
                                 f"with {t.leaves} rows of {t.words} {t.word_bits}-bit leaves")
            if (got["root"], got["domain"]) != (t.root, t.domain):
                raise ValueError(f"served tree {t.value!r}: root and domain are not value {value!r}'s of record "
                                 f"{record[:16]}...: {got!r}")
            outputs[t.value] = {"record": record, "value": value}
            domains[t.value] = t.domain
            continue
        if t.output is None:
            if t.domain != want["domain"]:
                raise ValueError(f"served tree {t.value!r}: domain {t.domain} is not {t.value!r}'s registered-value domain "
                                 f"{want['domain']}, and the tree names no earlier output whose domain it is")
            if t.value in again:
                R.refuse_unwindowed(again)
            if logged.get((t.value, t.domain), t.root) != t.root:
                raise ValueError(f"served tree {t.value!r}: an earlier record of the log registered it under this domain "
                                 f"with root {logged[(t.value, t.domain)][:16]}..., and a value is committed once a domain "
                                 f"(R7-held); a value served afresh each window is one the verifier doesn't hold "
                                 f"(`held`), under the window's root (`window_root`)")
            continue
        record, port = t.output
        got = R._output_port(log, record, port)
        if got is None:
            raise ValueError(f"served tree {t.value!r}: the log holds no record {record[:16]}... with a port {port!r}")
        if (got.get("root"), got.get("domain"), got.get("leaves")) != (t.root, t.domain, t.leaves):
            raise ValueError(f"served tree {t.value!r}: root, domain and leaves are not port {port!r}'s of record "
                             f"{record[:16]}...: {got!r}")
        outputs[t.value] = {"record": record, "port": port}
        domains[t.value] = t.domain
    roots = {t.value: t.root for t in served}
    file = G.reads_file(program, query, values, roots, ports=ports, domains=domains, schemas=schemas, window=window_root,
                        fixed=fixed)
    unread = sorted(set(values) - {e["value"] for e in file["ports"].values()})
    if unread:
        raise ValueError(f"served trees {unread}: no unit of the population reads them, so no proof would open them")
    return ServedRegistration([t.entry() for t in served], values, outputs or None, roots, file, P.partition(program, query),
                              schemas, window_root, None if held is None else list(held))


__all__ = ["SCHEMAS", "ServedRegistration", "ServedTree", "register_served"]
