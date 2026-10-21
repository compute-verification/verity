"""Served trees as registered ports: the one form in which a served path registers what it served and has the Lean verifier
open it (`register_served`).

A *served tree* is a frame-v3-sha512 tree of `hm96-sha512/row/v1` rows that serving committed and that a proof unit reads:
a template's operands, Dither's z/y leaves or receipt roots, H1's per-call A, y and z tables. Its root is a hiding
commitment, which binds nothing on its own: `EndToEndRegisteredRows` links a proof's value layer to a committed row only at
a registered read. So the service registers each served tree as a port of the verifier's own `--registered` file before
the session's statement and coin (`RowsSession.ServedRegistered`), and verifies with `flock-verify --zk --registered`;
`Flock.Guarantees.ServedRowsLinked` then gives the committed tree's row at every served read.

**The port form.** A served tree is the registered value of the program parameter whose rows it commits (`value`): its
`leaves` rows, each `words` leaves of `word_bits` bits (16 or 32), under `root` and `domain`. It enters the registration
record as one `registered` entry (`registered.ENTRY_KEYS`, plus `word_bits` when it is not 16), which R7 holds to the
verifier's own (`registration.check(..., registered=values, outputs=outputs)`), and the verifier's `--registered` file as
one port per circuit port that reads it (`registered.reads_file`: the value, root, leaves, words, word_bits, domain, and
the row each unit of the population reads, from the verifier's own program and query). Its domain is one of two, and no
other:

* the registered-value domain of `value` in the verifier's own program (`registered.registered_entries`), when serving
  committed the tree as a registered value (`Registered.commit`);
* an earlier record's port's domain, when the tree is that port (`output`: `(record digest, port)`, #2929's output values):
  the log's record with that digest names the port, its root and its leaf count, and the tree is that port's.

Each tree names its row format (`row_schema`, the `--registered` port's key for it, one of `SCHEMAS`): the
frame-v3-sha512 leaf schema its leaves were hashed under. Absent means `hm96-sha512/row/v1` (`registered.SCHEMA`), the one
format the registered form reads today, so a v1 tree's entry and port name none, and its bytes are the ones
`Registered.entry` and `reads_file` write. H1's and PoUW's served rows are `row/v2` and `row-seg/v1`: those are refused
until R7 and the Lean form (`Flock.Registered.checkPort`) read them.

**The program and the partition.** The positions are the verifier's: `reads` derives each unit's row from the program and
the query it is given, the path's own public Program and its `Q_template_instance` query, never the universal unit's. The
Lean verifier derives the same units only from the same objects, so the session is verified with `--program` the
program's canonical encoding (`codec.encode_program`), `--partition` the returned `partition` (`partition.partition(program,
query)`), and `--unit-indices` the staged instances' units. Rows are fixed slots: a tree pads its unused rows with zeros,
so the positions are a function of the unit index alone, and the layout is protocol structure.

`register_served` refuses (ValueError) a served tree whose value is not a parameter the program reads in whole rows, a
value twice, a root or domain that is not lowercase hex of 64 bytes, no rows or no leaves to a row, a width other than 16
or 32, a row format other than v1, a domain that is neither of the two above, an output whose record or port the log does
not hold or whose root, domain or rows are not the port's, a root an earlier record of the log registered otherwise
(R7-held), and a tree no unit of the population reads, which no proof would open.
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
#: the row formats a served tree may name; the registered form reads `READ` alone
SCHEMAS = (G.SCHEMA, "hm96-sha512/row/v2", "hm96-sha512/row-seg/v1")
READ = (G.SCHEMA,)


@dataclass(frozen=True)
class ServedTree:
    """One served tree: the program parameter whose rows it commits (`value`), its root and domain (lowercase hex, 64 bytes
    each), its rows (`leaves`), each `words` leaves of `word_bits` bits, its rows' format (`row_schema`, one of `SCHEMAS`), and,
    for an earlier run's output, `(record digest, port)`."""

    value: str
    root: str
    leaves: int
    words: int
    domain: str
    word_bits: int = G.WORD_BITS
    output: tuple[str, str] | None = None
    row_schema: str = G.SCHEMA

    @classmethod
    def of(cls, reg: G.Registered) -> "ServedTree":
        """A tree serving committed as a registered value (`Registered.commit`), under its registered-value domain."""
        e = reg.entry()
        return cls(e["value"], e["root"], e["leaves"], e["words"], e["domain"], reg.word_bits)

    def entry(self) -> dict[str, Any]:
        """The registration record's `registered` entry."""
        return {"value": self.value, "root": self.root, "leaves": self.leaves, "words": self.words,
                **G.width_entry(self.word_bits), "domain": self.domain}


@dataclass(frozen=True)
class ServedRegistration:
    """What `register_served` returns. `registered`: the record's `registered` entries (`registration.record(...,
    registered=)`); `values` and `outputs`: R7's own copies (`registration.check(..., registered=values, outputs=outputs)`,
    `outputs` None when no tree is an earlier output); `roots`: `{value: root}`; `file`: the `--registered` document,
    whose `ports` are the registered ports by circuit port; `partition`: the verifier's partition object, its `--partition`
    file."""

    registered: list[dict[str, Any]]
    values: dict[str, int]
    outputs: dict[str, dict[str, str]] | None
    roots: dict[str, str]
    file: dict[str, Any]
    partition: dict[str, Any]

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
    if t.row_schema not in READ:
        raise ValueError(f"served tree {t.value!r}: {t.row_schema} rows, and the registered form reads {' or '.join(READ)} rows "
                         f"alone (R7 and Flock.Registered.checkPort)")
    o = t.output
    if o is not None and not (isinstance(o, tuple) and len(o) == 2 and isinstance(o[0], str) and _HEX64.fullmatch(o[0])
                              and isinstance(o[1], str) and o[1]):
        raise ValueError(f"served tree {t.value!r}: an output is (record digest, port), not {o!r}")


def register_served(program: Program, query: Mapping[str, Any], served: Sequence[ServedTree],
                    log: Sequence[Mapping[str, Any]] = (), ports: Mapping[str, str] | None = None) -> ServedRegistration:
    """Register each served tree as a port of the verifier's own `--registered` file, in the module's port form: the record's
    `registered` entries, R7's copies, and the file, from the verifier's own program and query. `log`: this prover's earlier
    records, which an output tree's `(record digest, port)` names. `ports`: a template parameter's circuit port, where
    they differ (`registered.reads_file`). ValueError on anything the module's docstring refuses."""
    if not served:
        raise ValueError("no served tree: a registration with registered values registers at least one")
    for t in served:
        _check_form(t)
    names = [t.value for t in served]
    if len(set(names)) != len(names):
        raise ValueError(f"a value twice among the served trees: {sorted(v for v in set(names) if names.count(v) > 1)}")
    values = {t.value: t.words for t in served}
    mine = G.registered_entries(program, values)
    log = list(log)
    held = R.held_roots(log)
    outputs: dict[str, dict[str, str]] = {}
    domains: dict[str, str] = {}
    for t in served:
        want = mine[t.value]
        if t.leaves != want["leaves"] or t.word_bits != want["word_bits"]:
            raise ValueError(f"served tree {t.value!r}: {t.leaves} rows of {t.words} {t.word_bits}-bit leaves, and the "
                             f"program's parameter is {want['leaves']} rows of {want['words']} {want['word_bits']}-bit leaves")
        if t.output is None:
            if t.domain != want["domain"]:
                raise ValueError(f"served tree {t.value!r}: domain {t.domain} is not {t.value!r}'s registered-value domain "
                                 f"{want['domain']}, and the tree names no earlier output whose domain it is")
            if held.get(t.value, t.root) != t.root:
                raise ValueError(f"served tree {t.value!r}: an earlier record of the log registered it under root "
                                 f"{held[t.value][:16]}..., and a value is committed once (R7-held)")
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
    file = G.reads_file(program, query, values, roots, ports=ports, domains=domains)
    unread = sorted(set(values) - {e["value"] for e in file["ports"].values()})
    if unread:
        raise ValueError(f"served trees {unread}: no unit of the population reads them, so no proof would open them")
    return ServedRegistration([t.entry() for t in served], values, outputs or None, roots, file, P.partition(program, query))


__all__ = ["READ", "SCHEMAS", "ServedRegistration", "ServedTree", "register_served"]
