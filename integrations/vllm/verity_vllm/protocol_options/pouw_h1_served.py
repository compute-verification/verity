"""H1's served trees (VLLM-64): each call's A, y and z tables, the hiding tables under its round's root (`pouw_h1.H1Run`),
committed as registered values of the verifier's window Program, so a served path registers them as ports of the
verifier's own ``--registered`` file (`one_stage.served.register_served`) and verifies with ``flock-verify --zk
--registered``.

* **Values.** A table is the registered value ``h1/<window>/<call>/<table>`` (`value`): ``<window>`` the window's key
  (`Served.key`: the window-open root, `salt.window_open_root` of the weights' root, round 0's request table and the session,
  or the key the run is given), so each window's values are its own and a log's R7-held rule never meets one twice.
* **Rows** are fixed slots: a table is the call's T rows (`pouw_h1.slots`), row i the table leaf of slot i
  (`pouw_h1.table_leaves`: A's k s8 bytes then its f32 scale, y's n bf16, z's n s32) as u16 words, little-endian
  (`words`), and every empty slot zero words. The layout is protocol structure: no row count or tile shape becomes public.
* **The commitment.** With ``served`` (`Served`), `H1Run` commits each table as a registered value of the verifier's
  window Program (``program``, given the window's key): `registered.Registered.commit`'s ``hm96-sha512/row/v1`` rows
  (`SCHEMA`) under its registered-value domain, row i's salt `audit.hm96_salt` of the run's key at the table's nonce, and
  records its root in the call's ``tables``, so round s's root commits it before round s's coin. The program is the
  verifier's (compute accounting's H1 statement Program): it names a parameter for each table, T rows of its words.
* `trees` gives the window's `ServedTree`s and `register` registers them and writes ``--registered`` (`FILE`) in the
  window's directory. Refused (ValueError): a table off its fixed slots, a table whose root isn't the one its round's root
  commits, a round root that isn't over its calls' tables, and what `register_served` refuses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from verity.core.protocols.pouw import salt as PS
from verity.core.protocols.pouw.audit import hm96_salt
from verity.core.protocols.sampled_proofs.one_stage import partition as P, registered as G
from verity.core.protocols.sampled_proofs.one_stage.served import ServedRegistration, ServedTree, register_served
from verity.ml.circuits.program import Program

from verity_vllm.protocol_options import pouw_h1_requests as RQ

#: the tables in their nonce order (`pouw_h1.TABLE_TAGS`)
TABLES = ("a", "y", "z")
#: the rows' format: registered rows of u16 words, which the registered form reads today
SCHEMA = G.SCHEMA
FILE = "pouw-h1-registered.json"
VALUE = "h1/{window}/{call}/{table}"


def value(window: bytes, call: int, table: str) -> str:
    """Call `call`'s `table` in the window keyed `window`: its first 16 bytes, hex."""
    if table not in TABLES:
        raise ValueError(f"a call's tables are {TABLES}, not {table!r}")
    return VALUE.format(window=window[:16].hex(), call=int(call), table=table)


def width(table: str, k: int, n: int) -> int:
    """A slot's u16 words in `table` of a call of contraction k and n columns."""
    return {"a": -(-(k + 4) // 2), "y": n, "z": 2 * n}[table]


def words(leaves: Sequence[bytes], n: int) -> list[list[int]]:
    """Each leaf as `n` u16 words, little-endian, zero-padded."""
    out = []
    for leaf in leaves:
        if len(leaf) > 2 * n:
            raise ValueError(f"a {len(leaf)}-byte slot is past its {n} words")
        b = leaf + bytes(2 * n - len(leaf))
        out.append([int.from_bytes(b[2 * i:2 * i + 2], "little") for i in range(n)])
    return out


@dataclass
class Served:
    """How an `H1Run` commits its tables: as registered values of ``program(window)``, the verifier's window Program for the
    window's key (``window``, else the window-open root its commitment names). ``entries``: each committed table's
    registered entry, by (call, table)."""

    program: Callable[[bytes], Program]
    window: bytes | None = None
    entries: dict[tuple[int, str], dict[str, Any]] = field(default_factory=dict)
    _held: tuple[bytes, Program, str] | None = None

    def key(self, commits: Mapping[str, Any]) -> bytes:
        if self.window is not None:
            return self.window
        session = commits.get("session") or {}
        need = (commits.get("weights_root"), commits.get("requests_root"), session.get("session"))
        if None in need:
            raise ValueError("the window has no key: neither one given nor a window-open root (its weights' root, round 0's "
                             "request table and the session)")
        return PS.window_open_root(bytes.fromhex(need[0]), bytes.fromhex(need[1]), need[2])

    def held(self, commits: Mapping[str, Any]) -> tuple[bytes, Program, str]:
        """The window's key, its verifier's Program and that Program's digest, fixed at the first table."""
        if self._held is None:
            key = self.key(commits)
            prog = self.program(key)
            self._held = (key, prog, P.program_digest(prog))
        return self._held

    def commit(self, run: Any, call: Any, table: str, leaves: Sequence[bytes]) -> str:
        """`table` of `call` (its `table_leaves`) committed as its registered value; its root, hex."""
        key, _prog, digest = self.held(run.commits)
        nonce = RQ.TABLES_NONCE + 3 * call.index + TABLES.index(table)
        rows = words(leaves, width(table, call.k, call.n))
        reg = G.Registered.commit(digest, value(key, call.index, table), rows,
                                  [hm96_salt(run._key, nonce, i) for i in range(len(rows))])
        self.entries[(call.index, table)] = reg.entry()
        return reg.root.hex()


def _check_rounds(run: Any) -> None:
    for r, log in enumerate(run.round_log):
        got = RQ.step_digest(run.salt, r, run._rows([c for c in run.calls if c.step == r])).hex()
        if got != log["calls"]:
            raise ValueError(f"round {r}'s root is over calls {log['calls'][:16]}..., not its calls' tables ({got[:16]}...)")


def trees(run: Any) -> list[ServedTree]:
    """The window's served trees, each call's committed tables in call order: refused unless `run` committed them as
    `Served` values, every table is its call's T slots of its fixed width, and its root is the one its round's root
    commits (the call's ``tables``, in the step digest under ``round_log``'s root)."""
    served = getattr(run, "served", None)
    if not isinstance(served, Served):
        raise ValueError(f"{run.entry}: the window's tables aren't registered values (the run has no `Served`)")
    if len(run.round_log) != run.steps:
        raise ValueError(f"{run.entry}: {len(run.round_log)} of {run.steps} rounds closed, so a table may be under no root")
    _check_rounds(run)
    key, _prog, _digest = served.held(run.commits)
    out = []
    for c in run.calls:
        for t in TABLES:
            root = run.tables.get(c.index, {}).get(t)
            e = served.entries.get((c.index, t))
            if root is None and e is None:
                continue
            if e is None or e["value"] != value(key, c.index, t):
                raise ValueError(f"call {c.index}'s {t} table isn't the window's value {value(key, c.index, t)!r}")
            if (e["leaves"], e["words"]) != (run.rows, width(t, c.k, c.n)):
                raise ValueError(f"call {c.index}'s {t} table is {e['leaves']} rows of {e['words']} words, off its fixed "
                                 f"slots: {run.rows} rows of {width(t, c.k, c.n)}")
            if e["root"] != root:
                raise ValueError(f"call {c.index}'s {t} table has root {e['root'][:16]}..., not {str(root)[:16]}..., the one "
                                 f"round {c.step}'s root commits")
            out.append(ServedTree(e["value"], e["root"], e["leaves"], e["words"], e["domain"], row_schema=SCHEMA))
    return out


def register(run: Any, query: Mapping[str, Any], log: Sequence[Mapping[str, Any]] = (), ports: Mapping[str, str] | None = None,
             directory: str | Path | None = None) -> ServedRegistration:
    """The window's tables registered on its verifier's Program and `query` (`register_served`, after this prover's
    earlier records `log`), and the ``--registered`` file written as `FILE` in `directory` (default the run's
    ``transcript_dir``), the file ``flock-verify --zk --registered`` reads."""
    got = trees(run)
    _key, prog, _digest = run.served.held(run.commits)
    reg = register_served(prog, query, got, log=log, ports=ports)
    d = directory if directory is not None else run.transcript_dir
    if d is not None:
        Path(d).mkdir(parents=True, exist_ok=True)
        reg.write(Path(d) / FILE)
    return reg


__all__ = ["FILE", "SCHEMA", "Served", "TABLES", "VALUE", "register", "trees", "value", "width", "words"]
