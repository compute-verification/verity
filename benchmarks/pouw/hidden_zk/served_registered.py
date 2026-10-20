"""Whether a served statement's Lean check is the one `SessionSoundAt` reads, and whether its verdict may cite it
(RT4-LOGIC-25).

`Accounting.Guarantees.TS.SessionSoundAt` (#2928) takes `AcceptedLive`, flock-verify ``verify --zk --registered F``'s
acceptance, and `ServedRootsLinked`, which holds at δs = 0 where every served tree a unit's relation reads is a port of
the verifier's registered reads `F` (`Flock.SoundAtBridge.RowsSession.ServedRegistered`, #2909). `F` is a
``verity/registered-reads/v0`` document, the one `registered.reads_file` writes (and proofs' `served.register_served`,
its `file`): one entry per circuit port, ``{value, root, leaves, words, word_bits, domain, reads}``. `plan` takes a served
statement's circuit META, its public header and that document, and reads off what flock-verify refuses before it reads a
registered row (`Registered.check`, `checkPort`):

- per-port ``hm96-sha512/row/v2`` or ``row-seg/v1`` rows (META ``row_schemas``): a registered port's rows are
  ``hm96-sha512/row/v1`` rows (`checkPort`), and a statement of per-port rows has none (`HmRow.rowPorts`);
- a served port `F` names no tree for: its rows are committed afresh and open no registered root;
- units taken as stated: the statement binds no ``verity/partition/v1`` partition of its program (a composed circuit's
  default is the unit cover, whose units the verifier takes as stated);
- a public file without shared rows: a registered port is a table of a shared-row file.

With nothing in the way `plan` writes `F`, and `argv` is what the Lean check adds to ``verify --zk``. `claim` decides the
citation from that check's VERDICT lines: cited only when the plan was clear, and every session was accepted with its
units derived and each served port opened against its tree's root. Anything else is a refusal that says why, so a path
that can't register a tree never cites SessionSoundAt, and a plain ``--zk`` acceptance never stands in for it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from verity.ml.circuits.partition_object import FORMAT as PARTITION
from verity.core.protocols.sampled_proofs.one_stage import registered as G

CLAIM = "Accounting.Guarantees.TS.SessionSoundAt"
#: the only rows a registered port holds
ROW_V1 = G.SCHEMA


def header(pub: Path) -> dict[str, Any]:
    """A public file's header (its first line)."""
    with open(pub, "rb") as f:
        return json.loads(f.readline())


def blockers(meta: Mapping[str, Any], head: Mapping[str, Any], served: Sequence[str],
             reads: Mapping[str, Any] | None) -> list[str]:
    """What flock-verify ``--zk --registered`` refuses about reading the `served` ports of this statement (its circuit
    META and public header `head`) as the ports of `reads`, a ``--registered`` document; empty when nothing does."""
    out = []
    if reads is not None and reads.get("format") != G.READS:
        out.append(f"the registered reads are not a {G.READS} document: {reads.get('format')!r}")
    ports = [p[0] for p in meta.get("ports", ())]
    if "row_schemas" in meta or meta.get("row_schema") != ROW_V1:
        kinds = sorted({s for s, _bits in meta.get("row_schemas", ())}) or [meta.get("row_schema")]
        out.append(f"the statement's rows are {', '.join(map(str, kinds))}: a registered port's rows are {ROW_V1} rows "
                   "(Registered.checkPort), and a statement of per-port rows has none")
    by = dict((reads or {}).get("ports") or {})
    for p in served:
        if p not in ports:
            out.append(f"{p} is no port of the statement")
        elif p not in by:
            out.append(f"port {p}: its served tree is not registered")
    for p in by:
        if p not in served:
            out.append(f"port {p}: a registered tree no served port reads")
    if (meta.get("partition") or {}).get("rule") != PARTITION or not meta.get("program_sha512"):
        out.append(f"the statement binds no {PARTITION} partition of its program: its units are taken as stated, and a "
                   "registered read binds only at units the verifier derives (--partition, --program)")
    if "shared_rows" not in head:
        out.append("the public file has no shared rows: a registered port is a table of a shared-row file")
    return out


def plan(meta: Mapping[str, Any], head: Mapping[str, Any], reads: Mapping[str, Any] | None = None,
         served: Sequence[str] | None = None, out: Path | None = None) -> dict[str, Any]:
    """A served statement's registration plan: the served ports (`served`, default META's input row ports), what blocks
    reading them as the ports of `reads` (`blockers`), and, when nothing does, `reads` written to `out` as the verifier's
    ``--registered`` file."""
    if served is None:
        ports = list(meta.get("ports", ()))
        served = [p[0] for p in ports[:meta.get("input_ports", len(ports))]]
    served = list(served)
    why = blockers(meta, head, served, reads)
    file = None
    if not why and out is not None:
        Path(out).write_text(json.dumps(reads, sort_keys=True, separators=(",", ":")))
        file = str(out)
    roots = {p: {"value": e["value"], "root": e["root"]} for p, e in ((reads or {}).get("ports") or {}).items()}
    return {"claim": CLAIM, "served": served, "blockers": why, "registered": file, "roots": roots}


def argv(p: Mapping[str, Any] | None) -> list[str]:
    """What the Lean check adds to ``verify --zk`` for plan `p`: ``--registered F`` when it wrote `F`, else nothing."""
    return ["--registered", p["registered"]] if p and p.get("registered") and not p.get("blockers") else []


def verdicts(path: Path) -> list[dict[str, Any]]:
    """The VERDICT lines of a flock-verify output."""
    try:
        return [json.loads(x[8:]) for x in Path(path).read_text().splitlines() if x.startswith("VERDICT ")]
    except (OSError, ValueError):
        return []


def claim(p: Mapping[str, Any] | None, lines: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Whether a served statement's ``verify --zk`` check (`lines`, its VERDICT lines, run with `argv(p)`) may cite
    SessionSoundAt: only under a clear plan, with every session accepted, its units derived and its registered reads
    each served port's tree."""
    def no(why: str) -> dict[str, Any]:
        return {"claim": CLAIM, "cited": False, "why": why}

    if not p:
        return no("the path states no registration plan for its served trees")
    if p.get("blockers"):
        return no("its served trees are not registered: " + "; ".join(p["blockers"]))
    if not lines:
        return no("flock-verify --zk --registered printed no verdict")
    for v in lines:
        if not v.get("accepted"):
            return no(f"flock-verify --zk --registered refused a session: {v.get('why', '')}")
        if (v.get("partition") or {}).get("units") != "derived":
            return no("a session's units were not derived from the verifier's program")
        if v.get("registered") != p["roots"]:
            return no(f"a session's registered reads {v.get('registered')} are not the served trees' {p['roots']}")
    return {"claim": CLAIM, "cited": True, "why": ""}


def window(claims: Sequence[Mapping[str, Any] | None]) -> dict[str, Any]:
    """A window's citation from its drawn units' (`claim`'s): cited when every unit's is, else the first refusal."""
    if not claims:
        return {"claim": CLAIM, "cited": False, "why": "no unit's claim"}
    bad = next((c for c in claims if not (c and c.get("cited"))), None)
    if bad is None:
        return {"claim": CLAIM, "cited": True, "why": ""}
    return {"claim": CLAIM, "cited": False, "why": (bad or {}).get("why") or "a unit states no claim"}


__all__ = ["CLAIM", "ROW_V1", "argv", "blockers", "claim", "header", "plan", "verdicts", "window"]
