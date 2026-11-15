"""H1's live lowering: a served H1 window's live units (``live@<s>``, `circuit.live`) as Flock statements, and the
verifier's own files of each (`live.sh` and ``h1_prove/live.py`` prove and check them; the vLLM adapter's
``pouw_h1_circuits`` commits their circuit file).

The window's live form (`form`) is v2's up to `R` = `live.PIECE_ROWS` rows a round (unit s round s's ``live@<s>``), else
cut by rows (`live.PouwLiveRows_v1`): P = T / R pieces a round, unit u = s·P + j round s's piece j. Its Program and
partition query are `program_of`'s, the verifier's ``--registered`` file of a unit `files`', its own public file
`verifier_pub`'s, built from the opened ``b || c`` and its own known rows (`known_rows`), and the statement's stager
`live_stager`'s, whose staged ok `live_reference` holds to `live.live_ok_v2` (or `live.rows_ok`).
"""
from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from verity.protocols.resource_accounting.compute.pouw.prover.circuit import live as L
from verity.protocols.resource_accounting.compute.pouw.prover.lowering import h1unit as U
from verity.protocols.resource_accounting.compute.pouw.verifier import core_check as CC
from verity.protocols.resource_accounting.compute.pouw.verifier import live_reads as LR
from verity.protocols.proofs.spot.verifier.one_stage import partition as P
from verity.protocols.proofs.spot.verifier.one_stage import registered as G

if TYPE_CHECKING:
    from verity.circuits.program import Program
    from verity_flock.class_statement import UnitStager

NAME = "unit/pouw-live"
#: the rows of a piece, past which a round's live check is cut by rows
R = L.PIECE_ROWS


def form(pub: dict) -> tuple[int, int]:
    """The window's live form, (rows a unit, units a round): v2's (T, 1) up to `R` rows, else (R, T / R), pieces of R rows
    (`live.pieces`)."""
    T = pub["T"]
    return (T, 1) if T <= R else (R, L.pieces(T, R))


def label(pub: dict, u: int) -> str:
    """Unit u's name: ``live@<s>``, or ``live@<s> piece <j>`` when the round is cut by rows."""
    r, p = form(pub)
    return f"live@{u}" if p == 1 else f"live@{u // p} piece {u % p}"


def files(pub: dict, u: int, m: int) -> dict:
    """The verifier's ``--registered`` file of unit u at m and the public max_tokens (`live_reads.reads_file`, or
    `reads_file_rows` when the round is cut by rows)."""
    r, p = form(pub)
    if p == 1:
        return LR.reads_file(pub["program"], pub["T"], pub["S"], u, pub["roots"], pub["window"], pub["session"], m,
                             pub["prefix"], pub["max_tokens"])
    return LR.reads_file_rows(pub["program"], pub["T"], pub["S"], r, u, pub["roots"], pub["window"], pub["session"], m,
                              pub["max_tokens"], pub["prefix"])


def known_rows(pub: dict, u: int, m: int) -> dict[str, tuple[G.Registered, int]]:
    """The verifier's own registered rows of unit u, round s's piece j: ``m@<s>`` at m (row 0), ``round@<s>`` (row 0),
    ``caps`` (row i for ``cap_<i>``), ``live-ok`` (row u), and when the round is cut by rows ``base@<j>`` (row 0)."""
    prog, prefix, window = pub["program"], pub["prefix"], pub["window"]
    r, p = form(pub)
    s, j = divmod(u, p)
    cap = LR.caps_table(prog, prefix, window, pub["T"], pub["max_tokens"])[2]
    out = {f"cap_{i}": (cap, i) for i in range(r)} | {
        "m": (LR.m_table(prog, prefix, window, s, m)[2], 0), "s": (LR.round_table(prog, prefix, window, s)[2], 0),
        LR.OK_PORT: (LR.ok_table(prog, prefix, window, pub["S"] * p)[2], u)}
    return out if p == 1 else out | {"base": (LR.base_table(prog, prefix, window, j, r)[2], 0)}


def piece_opening(pub: dict, u: int, opening: dict) -> dict:
    """Unit u's opened rows from its round's ``opening.json``: piece j's ``slot_<k>`` is the round's ``slot_<jR+k>`` and its
    ``req_<k>`` the round's ``req_<k>``; a round not cut by rows is its own."""
    r, p = form(pub)
    if p == 1:
        return opening
    j = u % p
    return {f"slot_{k}": opening[f"slot_{j * r + k}"] for k in range(r)} | {f"req_{k}": opening[f"req_{k}"] for k in range(r)}


def piece_rows(pub: dict, u: int, z: dict) -> dict:
    """Unit u's prover rows (`window.py open`'s form) from its round's ``rows.npz`` ``z``: as `piece_opening` takes them,
    the caps, m and s as they are, and ``base`` and ``o0`` at the verifier's own known rows; a round not cut by rows is its
    own."""
    r, p = form(pub)
    if p == 1:
        return z
    j = u % p
    out: dict = {"units": np.array([u])}
    pairs = [(f"slot_{k}", f"slot_{j * r + k}") for k in range(r)] + [(f"req_{k}", f"req_{k}") for k in range(r)]
    pairs += [(f"cap_{k}", f"cap_{k}") for k in range(r)] + [("m", "m"), ("s", "s")]
    for dst, src in pairs:
        out |= {f"{x}_{dst}": z[f"{x}_{src}"] for x in "wsvrp"}
    known = known_rows(pub, u, 0)
    for port, words in (("base", LR.slot_words([j * r])), (LR.OK_PORT, [list(CC.OK_ROW)])):
        reg, i = known[port]
        out |= {f"w_{port}": np.asarray(words, np.uint16), f"s_{port}": np.frombuffer(reg.salts[i], np.uint8).reshape(1, -1),
                f"v_{port}": np.array([reg.value]), f"r_{port}": np.array([i]), f"p_{port}": np.array([reg.path(i).hex()])}
    return out


def check_openings(pub: dict, u: int, opening: dict, reg: dict) -> list[str]:
    """Why unit u's opened slot and header rows don't open their registered roots where the verifier's file reads them."""
    bad = []
    for port, e in reg["ports"].items():
        if "known" in e:
            continue
        op = opening.get(port)
        dom = G.domain(pub["program"], e["value"], e["words"], e["leaves"],
                       **({"session": pub["session"]} if e["value"].endswith(f"{LR.REQUESTS}@0") else {"window": pub["window"]}))
        if op is None or op["value"] != e["value"] or op["index"] != e["reads"][u] or dom.domain_id.hex() != e["domain"] or \
                not G.verify_read(dom, bytes.fromhex(e["root"]), op["index"], bytes.fromhex(op["bc"]),
                                  bytes.fromhex(op["path"]), e.get(G.SCHEMA_KEY, G.SCHEMA)):
            bad.append(f"{label(pub, u)}: port {port}'s opened row is not row {e['reads'][u]} of {e['value']}'s registered root")
    return bad


def verifier_pub(head: dict, pub: dict, u: int, m: int, opening: dict, reg: dict) -> bytes:
    """The verifier's own public file of unit u's statement at m, from the opened ``b || c`` and its own known rows."""
    bcs, reads = {}, {}
    for port, e in reg["ports"].items():
        if "known" in e:
            continue
        bcs[port], reads[port] = bytes.fromhex(opening[port]["bc"]), (e["value"], e["reads"][u], opening[port]["path"])
    for port, (r, i) in known_rows(pub, u, m).items():
        bcs[port], reads[port] = r.bcs[i], (r.value, i, r.path(i).hex())
    return U.verifier_pub(head, f"live-{u}", NAME, [bcs], [u], [reads])


def claimed(z: dict, pub: dict, u: int, m: int) -> dict:
    """Unit u's prover rows with m replaced by ``m``, at the path its claimed ``m@<s>`` registers."""
    r, i = known_rows(pub, u, m)["m"]
    return {**z, "w_m": np.asarray(LR.slot_words([m]), np.uint16), "v_m": np.array([r.value]),
            "p_m": np.array([r.path(i).hex()])}


def live_reference(T: int, r: int | None = None) -> Callable[[dict, dict], None]:
    """`UnitStager.reference_check`: the staged ok is `live.live_ok_v2` of the instance's slot leaves, headers, m, round s
    and caps, or with `r` rows a piece `live.rows_ok` of its r rows, headers and caps, m, s and base."""
    def check(rows: dict, outs: dict) -> None:
        u32 = lambda w: [int(w[2 * j]) | int(w[2 * j + 1]) << 16 for j in range(len(w) // 2)]  # noqa: E731
        n = T if r is None else r
        slot = [u32(rows[f"slot_{i}"][0])[0] for i in range(n)]
        req = [u32(rows[f"req_{i}"][0]) for i in range(n)]
        cap = [u32(rows[f"cap_{i}"][0])[0] for i in range(n)]
        m, s = u32(rows["m"][0])[0], u32(rows["s"][0])[0]
        if r is None:
            fn, want = "live_ok_v2", L.live_ok_v2(slot, req, m, s, cap)
        else:
            fn, want = "rows_ok", L.rows_ok(slot, req, m, s, u32(rows["base"][0])[0], cap, T)
        if int(outs["o0"][0][0]) != want:
            raise AssertionError(f"{NAME}: the lowered circuit's ok is {int(outs['o0'][0][0])}, {fn}'s {want}")
    return check


def stage(stager: UnitStager, z: dict, out: Path, u: int) -> dict:
    """One unit's statement on the prover's rows ``z`` (`window.py open`'s form), instance unit u."""
    st = U.statement_tables(z, stager.row_ports, [p.name for p in stager.ports if p.name not in stager.row_ports])
    out.mkdir(parents=True, exist_ok=True)
    got = stager.stage({p: z[f"w_{p}"] for p in stager.row_ports}, out, os.urandom(32), salts=st["salts"], indices=[u],
                       set_name=f"live-{u}", tables=st["tables"], refs=st["refs"], registered=st["registered"])
    return {"ok": int(got["outs"]["o0"][0][0]), "ands": got["ands"]}


def program_of(pub: dict) -> tuple[Program, dict[str, Any]]:
    """The window's live Program and its partition query: v2's (`live.program_v2`), or the pieced one past `R` rows."""
    T, S = pub["T"], pub["S"]
    r, p = form(pub)
    if p == 1:
        return L.program_v2(T, S, pub["prefix"]), L.query_v2(T)
    return L.program_rows(T, S, pub["prefix"], r), L.query_rows(T, r)


def live_stager(T: int, circuit: Path, part: dict, r: int | None = None) -> UnitStager:
    """Flock's stager of `live.template_v2(T)`, or with `r` rows a piece `live.template_rows(T, r)`, bound to the live
    Program's partition, its circuit written once."""
    from verity_flock import class_statement as CS
    tmpl, rows = (L.template_v2(T), LR.statement_rows(T, v2=True)) if r is None else \
        (L.template_rows(T, r), LR.statement_rows_pieces(r))
    stager = CS.UnitStager(tmpl, NAME, rows, LR.STATEMENT_OUT_ROWS, circuit=circuit,
                           bind={"partition": P.digest(part), "program": part["program"]})
    stager.reference_check = live_reference(T, r)
    return stager


__all__ = ["NAME", "R", "check_openings", "claimed", "files", "form", "known_rows", "label", "live_reference",
           "live_stager", "piece_opening", "piece_rows", "program_of", "stage", "verifier_pub"]
