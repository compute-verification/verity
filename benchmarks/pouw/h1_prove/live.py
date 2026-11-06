"""live.py: a served H1 window's live units (``live@<s>``, `circuit.live`) through sampled proofs, the
verifier's side and the prover's staging; `live.sh` proves them by C-Flock --zk and has Lean verify them.

    python live.py prepare --live DIR --work W
    python live.py summarize --work W --out OUT
    python live.py evidence --work W --out LIVE.json

DIR is the serving process's dump (`pouw_h1_live.dump`, `pouw_h1_gpu --live-dump`'s DIR/<window>): DIR/window.json is public (T, S, the
prefix, the window Program's digest, the open root, the session, the live tables' roots, each round's served rows m_s and
round 0's requests' max_tokens), and DIR/<s>/ the prover's rows and openings of round s.

``prepare``:

1. The verifier, from DIR/window.json alone, builds the live Program of T and S (`live.program_v2`, credit's) and its
   template-instance partition (`live.query_v2`): W/verifier/program.json and partition.json, Lean's ``--program`` and
   ``--partition``. Unit s is round s's ``live@<s>``. Each round's ``--registered`` file is `live_reads.reads_file` at the
   round's served rows m_s and the public max_tokens.
2. The verifier checks each opened row of DIR/<s>/opening.json against its registered root, at the position its file
   reads (`registered.verify_read`): the slot rows ``slots@<s>``'s and the header rows ``requests@0``'s. It then builds its
   own public file of the round (`h1unit.verifier_pub`) from the opened ``b || c`` and its own rows of ``m@<s>``,
   ``round@<s>``, ``caps`` and ``live-ok`` (`live_reads.m_table`, `round_table`, `caps_table`, `ok_table`, all public).
3. The prover stages each round as a one-instance shared-row statement on its rows (`UnitStager` of `live.template_v2(T)`,
   bound to the live Program and partition, the instance unit s). Its pub.bin must be the verifier's byte for byte, and the
   staged ok must be `live.live_ok_v2`'s on the round's words.

The checks of W/plan.tsv (``C`` lines; ``S`` lines are the statements to prove):

* ``honest-<s>``: round s's statement against the verifier's own files. ACCEPT, every round.
* ``m-other``: round 0's honest session against the verifier's files at m_0 + 1, a window whose ``live@0`` claims rows it
  didn't serve. REJECT.
* ``m-claimed``: a prover that claims m_0 + 1, staged on its rows with that m (the unit computes ok = 0), against the
  files at m_0 + 1. Its ok row opens no row of ``live-ok``. REJECT.
* ``other-round`` (S > 1): round 0's honest session against round 1's files. REJECT.

``summarize`` reads Lean's verdicts into OUT/live-summary.json, each against its expectation.

``evidence`` writes the window's live evidence as core's credit reads it (`credit.LIVE_STATEMENTS`: the live descriptor
and each round's honest statement's circuit, public file and sessions; `evidence`), which ``served_verdicts.py pouw
--live-evidence`` hands credit, whose verifier of record verifies each session itself.

**Past T = `live.PIECE_ROWS`** (`R`) a round's live check is cut by rows (`live.PouwLiveRows_v1`, `form`): the Program is
`live.program_rows` of P = T / R pieces a round, unit u = s·P + j round s's piece j, each statement one piece on its R
rows of round s's slot table, the first R headers and caps and the known ``base@<j>`` (`live_reads.reads_file_rows`),
its ok row u of S·P. ``prepare --units`` stages only the units it lists (each a statement of its own, the size of a
T 64 unit, whatever T is). The checks are the same, each at its unit (``honest-<s>-<j>``), ``m-claimed`` at the piece
holding row m_0 (or piece 0 when m_0 = T, which every piece refuses), and ``other-piece`` (P > 1): unit 0's honest
session against unit 1's files, another base. REJECT. Credit's form is the v2 one: it doesn't read the pieced Program yet.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import h1unit as U  # noqa: E402
from verity.ml.circuits.codec import canonical_json, encode_program  # noqa: E402
from verity.core.protocols.pouw import core_check as CC  # noqa: E402
from verity.core.protocols.pouw.circuit import live as L  # noqa: E402
from verity.core.protocols.pouw.circuit import live_reads as LR  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import partition as P  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import registered as G  # noqa: E402

NAME = "unit/pouw-live"
#: the rows of a piece, past which a round's live check is cut by rows
R = L.PIECE_ROWS
EXPECT = {"honest": "ACCEPT", "m-other": "REJECT", "m-claimed": "REJECT", "other-round": "REJECT", "other-piece": "REJECT"}


def log(msg: str) -> None:
    print(msg, flush=True)


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


def live_reference(T: int, r: int | None = None):
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


def stage(stager, z: dict, out: Path, u: int) -> dict:
    """One unit's statement on the prover's rows ``z`` (`window.py open`'s form), instance unit u."""
    st = U.statement_tables(z, stager.row_ports, [p.name for p in stager.ports if p.name not in stager.row_ports])
    out.mkdir(parents=True, exist_ok=True)
    got = stager.stage({p: z[f"w_{p}"] for p in stager.row_ports}, out, os.urandom(32), salts=st["salts"], indices=[u],
                       set_name=f"live-{u}", tables=st["tables"], refs=st["refs"], registered=st["registered"])
    return {"ok": int(got["outs"]["o0"][0][0]), "ands": got["ands"]}


def program_of(pub: dict):
    """The window's live Program and its partition query: v2's (`live.program_v2`), or the pieced one past `R` rows."""
    T, S = pub["T"], pub["S"]
    r, p = form(pub)
    if p == 1:
        return L.program_v2(T, S, pub["prefix"]), L.query_v2(T)
    return L.program_rows(T, S, pub["prefix"], r), L.query_rows(T, r)


def cmd_prepare(a) -> int:
    live, work = Path(a.live), Path(a.work)
    pub = json.loads((live / "window.json").read_text())
    T, S, served = pub["T"], pub["S"], pub["served"]
    r, p = form(pub)
    vp, sp = work / "verifier", work / "st"
    vp.mkdir(parents=True, exist_ok=True)
    program, query = program_of(pub)
    part = P.partition(program, query)
    (vp / "program.json").write_bytes(canonical_json(encode_program(program)))
    (vp / "partition.json").write_text(json.dumps(part))
    units = [u for u in range(len(served) * p) if a.units is None or u in a.units]
    log(f"live Program: T={T} S={S}, {P.evaluate(program, query).population} units ({p} a round of {r} rows), rounds "
        f"served {served}, staging {len(units)}")
    stager = live_stager(T, work / "circuit.txt", part, None if p == 1 else r)
    head = {"meta": stager.meta, "pin": stager.pin, "class": stager.class_sha512}
    (work / "pin.txt").write_text(stager.pin)
    record = {"T": T, "S": S, "R": r, "pieces": p, "served": served, "program_sha512": part["program"],
              "partition": P.digest(part), "ands": stager.low.layout.counts["ands"], "circuit_sha512": stager.pin,
              "rounds": {}}
    plan, problems = [], []
    rounds: dict[int, tuple[dict, dict]] = {}

    def round_rows(s: int) -> tuple[dict, dict]:
        if s not in rounds:
            rounds[s] = (dict(np.load(live / str(s) / "rows.npz")), json.loads((live / str(s) / "opening.json").read_text()))
        return rounds[s]

    def unit(u: int) -> tuple[dict, dict]:
        z, opening = round_rows(u // p)
        return piece_rows(pub, u, z), piece_opening(pub, u, opening)

    def tag_of(u: int) -> str:
        return f"honest-{u}" if p == 1 else f"honest-{u // p}-{u % p}"

    def statement(tag: str, z: dict, u: int, m: int, opening: dict) -> Path:
        d = sp / tag
        got = stage(stager, z, d, u)
        reg = files(pub, u, m)
        (vp / f"{tag}.registered.json").write_text(json.dumps(reg))
        (vp / f"{tag}.pub.bin").write_bytes(verifier_pub(head, pub, u, m, opening, reg))
        (d / "pin.txt").write_text(stager.pin)
        record["rounds"][tag] = {"round": u // p, "m": m, "staged_ok": got["ok"]} | ({} if p == 1 else {"piece": u % p})
        plan.append(f"S\t{tag}\t{d}\t1")
        return d

    def other(tag: str, of: str, u: int, m: int, opening: dict) -> None:
        o = vp / tag
        reg = files(pub, u, m)
        o.with_suffix(".registered.json").write_text(json.dumps(reg))
        o.with_suffix(".pub.bin").write_bytes(verifier_pub(head, pub, u, m, opening, reg))
        plan.append(f"C\t{tag}\t{sp / of}\t{o.with_suffix('.pub.bin')}\t{o.with_suffix('.registered.json')}\t"
                    f"{EXPECT[tag]}")

    for u in units:
        m = served[u // p]
        z, opening = unit(u)
        problems += check_openings(pub, u, opening, files(pub, u, m))
        tag = tag_of(u)
        d = statement(tag, z, u, m, opening)
        if (d / "pub.bin").read_bytes() != (vp / f"{tag}.pub.bin").read_bytes():
            problems.append(f"{tag}: the prover's pub.bin is not the verifier's own file")
        if record["rounds"][tag]["staged_ok"] != 1:
            problems.append(f"{tag}: the unit's ok is 0 at the served rows m = {m}")
        plan.append(f"C\t{tag}\t{d}\t{vp / f'{tag}.pub.bin'}\t{vp / f'{tag}.registered.json'}\t{EXPECT['honest']}")
        if u == 0:
            other("m-other", tag, 0, m + 1, opening)
            c_u = min(m // r, p - 1)
            cz, copening = unit(c_u)
            c = statement("m-claimed", claimed(cz, pub, c_u, m + 1), c_u, m + 1, copening)
            if record["rounds"]["m-claimed"]["staged_ok"] != 0:
                problems.append(f"m-claimed: the unit's ok is 1 at m = {m + 1}, past the served rows")
            plan.append(f"C\tm-claimed\t{c}\t{c / 'pub.bin'}\t{vp / 'm-claimed.registered.json'}\t{EXPECT['m-claimed']}")
            if p > 1:
                other("other-piece", tag, 1, m, unit(1)[1])
    if S > 1 and len(served) > 1 and 0 in units and p in units:
        d1 = vp / tag_of(p)
        plan.append(f"C\tother-round\t{sp / tag_of(0)}\t{d1}.pub.bin\t{d1}.registered.json\t{EXPECT['other-round']}")
    record["problems"] = problems
    (work / "record.json").write_text(json.dumps(record, indent=1))
    (work / "plan.tsv").write_text("\n".join(plan) + "\n")
    log("PREPARED " + json.dumps({k: record[k] for k in ("T", "S", "R", "pieces", "served", "ands", "problems")}))
    return 1 if problems else 0


def live_stager(T: int, circuit: Path, part: dict, r: int | None = None):
    """C-Flock's stager of `live.template_v2(T)`, or with `r` rows a piece `live.template_rows(T, r)`, bound to the live
    Program's partition, its circuit written once."""
    from verity_flock import class_statement as CS
    tmpl, rows = (L.template_v2(T), LR.statement_rows(T, v2=True)) if r is None else \
        (L.template_rows(T, r), LR.statement_rows_pieces(r))
    stager = CS.UnitStager(tmpl, NAME, rows, LR.STATEMENT_OUT_ROWS, circuit=circuit,
                           bind={"partition": P.digest(part), "program": part["program"]})
    stager.reference_check = live_reference(T, r)
    return stager


def _verdict(p: Path) -> dict | None:
    if not p.exists():
        return None
    v = [json.loads(x[8:]) for x in p.read_text().splitlines() if x.startswith("VERDICT ")]
    return {**v[0], "accepted": all(x.get("accepted") for x in v), "sessions": len(v),
            "why": next((x.get("why", "") for x in v if not x.get("accepted")), "")} if v else None


def cmd_summarize(a) -> int:
    work, out = Path(a.work), Path(a.out)
    record = json.loads((work / "record.json").read_text())
    checks = []
    for line in (work / "plan.tsv").read_text().splitlines():
        kind, tag, *rest = line.split("\t")
        if kind != "C":
            continue
        v = _verdict(out / "lean" / f"{tag}.out")
        err = out / "lean" / f"{tag}.err"
        word = ("ACCEPT" if v["accepted"] else "REJECT") if v else "NONE"
        checks.append({"tag": tag, "expect": rest[-1], "verdict": word, "pass": word == rest[-1],
                       "why": (v or {}).get("why") or (err.read_text(errors="replace").strip().splitlines()[-1:] if err.exists() else []),
                       "units": (v or {}).get("units")})
    ok = not record["problems"] and bool(checks) and all(c["pass"] for c in checks)
    accepted = [c for c in checks if c["tag"].startswith("honest-") and c["verdict"] == "ACCEPT"]
    doc = {"ok": ok, "T": record["T"], "S": record["S"], "R": record.get("R", record["T"]),
           "pieces": record.get("pieces", 1), "served": record["served"], "ands": record["ands"],
           "live_accepted": f"{len(accepted)}/{sum(t.startswith('honest-') for t in record['rounds'])}", "problems": record["problems"], "checks": checks,
           "circuit_sha512": record["circuit_sha512"], "program_sha512": record["program_sha512"],
           "partition": record["partition"], "inputs_sha512": hashlib.sha512((work / "plan.tsv").read_bytes()).hexdigest()}
    (out / "live-summary.json").write_text(json.dumps(doc, indent=1))
    log("LIVE " + json.dumps({k: doc[k] for k in ("ok", "live_accepted", "problems")}
                             | {"checks": {c["tag"]: [c["verdict"], c["expect"]] for c in checks}}))
    return 0 if ok else 1


def evidence(work: Path) -> dict:
    """The window's live evidence as core's credit reads it (`credit.LIVE_STATEMENTS`): the live Program's descriptor (its
    canonical JSON without annotations, whose SHA-512 is the program its sessions were proved under) in hex, and each
    round's statement, ``{circuit, public, round, sessions}``: the live circuit file, the verifier's own public file and
    the honest statement's sessions, which credit verifies itself; a round cut by rows has one a staged piece, with its
    ``piece``. ValueError on a round with no session."""
    record = json.loads((work / "record.json").read_text())
    desc = json.loads((work / "verifier" / "program.json").read_bytes())
    raw = canonical_json({k: v for k, v in desc.items() if k != "annotations"})
    if hashlib.sha512(raw).hexdigest() != record["program_sha512"]:
        raise ValueError(f"{work}'s live descriptor is not the program its statements were staged under")
    out = []
    for tag, r in sorted(record["rounds"].items(), key=lambda kv: (kv[1]["round"], kv[1].get("piece", 0))):
        if not tag.startswith("honest-"):
            continue
        sessions = sorted(str(p) for p in (work / "st" / tag / "sessions").glob("*") if p.is_dir())
        if not sessions:
            raise ValueError(f"round {r['round']}'s statement {tag} has no session under {work / 'st' / tag}: prove it first")
        out.append({"circuit": str(work / "circuit.txt"), "public": str(work / "verifier" / f"{tag}.pub.bin"),
                    "round": r["round"], "sessions": sessions} | ({"piece": r["piece"]} if "piece" in r else {}))
    return {"descriptor": raw.hex(), "statements": out}


def cmd_evidence(a) -> int:
    try:
        doc = evidence(Path(a.work))
    except (OSError, ValueError, KeyError) as e:
        log(f"LIVE-EVIDENCE refused: {type(e).__name__}: {e}")
        return 1
    Path(a.out).write_text(json.dumps(doc, indent=1))
    log(f"LIVE-EVIDENCE {len(doc['statements'])} rounds, descriptor {hashlib.sha512(bytes.fromhex(doc['descriptor'])).hexdigest()[:16]}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("evidence")
    p.add_argument("--work", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--live", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--units", type=lambda v: {int(x) for x in v.split(",")}, default=None,
                   help="stage only these units (u = s·P + j; default every served round's)")
    p = sub.add_parser("summarize")
    p.add_argument("--work", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"prepare": cmd_prepare, "summarize": cmd_summarize, "evidence": cmd_evidence}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
