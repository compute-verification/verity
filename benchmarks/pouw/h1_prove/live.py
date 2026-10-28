"""live.py: a served H1 window's live units (VLLM-85's ``live@<s>``, `circuit.live`) through sampled proofs, the
verifier's side and the prover's staging; `live.sh` proves them by Flock --zk and has Lean verify them.

    python live.py prepare --live DIR --work W
    python live.py summarize --work W --out OUT

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
from verity.core.protocols.pouw.circuit import live as L  # noqa: E402
from verity.core.protocols.pouw.circuit import live_reads as LR  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import partition as P  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import registered as G  # noqa: E402

NAME = "unit/pouw-live"
EXPECT = {"honest": "ACCEPT", "m-other": "REJECT", "m-claimed": "REJECT", "other-round": "REJECT"}


def log(msg: str) -> None:
    print(msg, flush=True)


def files(pub: dict, s: int, m: int) -> dict:
    """The verifier's ``--registered`` file of round s at m and the public max_tokens (`live_reads.reads_file`)."""
    return LR.reads_file(pub["program"], pub["T"], pub["S"], s, pub["roots"], pub["window"], pub["session"], m, pub["prefix"],
                         pub["max_tokens"])


def known_rows(pub: dict, s: int, m: int) -> dict[str, tuple[G.Registered, int]]:
    """The verifier's own registered rows of round s: ``m@<s>`` at m (row 0), ``round@<s>`` (row 0), ``caps`` (row i for
    ``cap_<i>``) and ``live-ok`` (row s)."""
    prog, prefix, window = pub["program"], pub["prefix"], pub["window"]
    cap = LR.caps_table(prog, prefix, window, pub["T"], pub["max_tokens"])[2]
    return {f"cap_{i}": (cap, i) for i in range(pub["T"])} | {
        "m": (LR.m_table(prog, prefix, window, s, m)[2], 0), "s": (LR.round_table(prog, prefix, window, s)[2], 0),
        LR.OK_PORT: (LR.ok_table(prog, prefix, window, pub["S"])[2], s)}


def check_openings(pub: dict, s: int, opening: dict, reg: dict) -> list[str]:
    """Why round s's opened slot and header rows don't open their registered roots where the verifier's file reads them."""
    bad = []
    for port, e in reg["ports"].items():
        if "known" in e:
            continue
        op = opening.get(port)
        dom = G.domain(pub["program"], e["value"], e["words"], e["leaves"],
                       **({"session": pub["session"]} if e["value"].endswith(f"{LR.REQUESTS}@0") else {"window": pub["window"]}))
        if op is None or op["value"] != e["value"] or op["index"] != e["reads"][s] or dom.domain_id.hex() != e["domain"] or \
                not G.verify_read(dom, bytes.fromhex(e["root"]), op["index"], bytes.fromhex(op["bc"]),
                                  bytes.fromhex(op["path"]), e.get(G.SCHEMA_KEY, G.SCHEMA)):
            bad.append(f"live@{s}: port {port}'s opened row is not row {e['reads'][s]} of {e['value']}'s registered root")
    return bad


def verifier_pub(head: dict, pub: dict, s: int, m: int, opening: dict, reg: dict) -> bytes:
    """The verifier's own public file of round s's statement at m, from the opened ``b || c`` and its own known rows."""
    bcs, reads = {}, {}
    for port, e in reg["ports"].items():
        if "known" in e:
            continue
        bcs[port], reads[port] = bytes.fromhex(opening[port]["bc"]), (e["value"], e["reads"][s], opening[port]["path"])
    for port, (r, i) in known_rows(pub, s, m).items():
        bcs[port], reads[port] = r.bcs[i], (r.value, i, r.path(i).hex())
    return U.verifier_pub(head, f"live-{s}", NAME, [bcs], [s], [reads])


def claimed(z: dict, pub: dict, s: int, m: int) -> dict:
    """Round s's prover rows with m replaced by ``m``, at the path its claimed ``m@<s>`` registers."""
    r, i = known_rows(pub, s, m)["m"]
    return {**z, "w_m": np.asarray(LR.slot_words([m]), np.uint16), "v_m": np.array([r.value]),
            "p_m": np.array([r.path(i).hex()])}


def live_reference(T: int):
    """`UnitStager.reference_check`: the staged ok is `live.live_ok_v2` of the instance's slot leaves, headers, m, round s
    and caps."""
    def check(rows: dict, outs: dict) -> None:
        u32 = lambda w: [int(w[2 * j]) | int(w[2 * j + 1]) << 16 for j in range(len(w) // 2)]  # noqa: E731
        slot = [u32(rows[f"slot_{i}"][0])[0] for i in range(T)]
        req = [u32(rows[f"req_{i}"][0]) for i in range(T)]
        cap = [u32(rows[f"cap_{i}"][0])[0] for i in range(T)]
        want = L.live_ok_v2(slot, req, u32(rows["m"][0])[0], u32(rows["s"][0])[0], cap)
        if int(outs["o0"][0][0]) != want:
            raise AssertionError(f"{NAME}: the lowered circuit's ok is {int(outs['o0'][0][0])}, live_ok_v2's {want}")
    return check


def stage(stager, z: dict, out: Path, s: int) -> dict:
    """One round's statement on the prover's rows ``z`` (`window.py open`'s form), instance unit s."""
    st = U.statement_tables(z, stager.row_ports, [p.name for p in stager.ports if p.name not in stager.row_ports])
    out.mkdir(parents=True, exist_ok=True)
    got = stager.stage({p: z[f"w_{p}"] for p in stager.row_ports}, out, os.urandom(32), salts=st["salts"], indices=[s],
                       set_name=f"live-{s}", tables=st["tables"], refs=st["refs"], registered=st["registered"])
    return {"ok": int(got["outs"]["o0"][0][0]), "ands": got["ands"]}


def cmd_prepare(a) -> int:
    live, work = Path(a.live), Path(a.work)
    pub = json.loads((live / "window.json").read_text())
    T, S, served = pub["T"], pub["S"], pub["served"]
    vp, sp = work / "verifier", work / "st"
    vp.mkdir(parents=True, exist_ok=True)
    program = L.program_v2(T, S, pub["prefix"])
    part = P.partition(program, L.query_v2(T))
    (vp / "program.json").write_bytes(canonical_json(encode_program(program)))
    (vp / "partition.json").write_text(json.dumps(part))
    log(f"live Program: T={T} S={S}, {P.evaluate(program, L.query_v2(T)).population} units, rounds served {served}")
    stager = live_stager(T, work / "circuit.txt", part)
    head = {"meta": stager.meta, "pin": stager.pin, "class": stager.class_sha512}
    (work / "pin.txt").write_text(stager.pin)
    record = {"T": T, "S": S, "served": served, "program_sha512": part["program"], "partition": P.digest(part),
              "ands": stager.low.layout.counts["ands"], "circuit_sha512": stager.pin, "rounds": {}}
    plan, problems = [], []

    def statement(tag: str, z: dict, s: int, m: int, opening: dict) -> Path:
        d = sp / tag
        got = stage(stager, z, d, s)
        reg = files(pub, s, m)
        (vp / f"{tag}.registered.json").write_text(json.dumps(reg))
        (vp / f"{tag}.pub.bin").write_bytes(verifier_pub(head, pub, s, m, opening, reg))
        (d / "pin.txt").write_text(stager.pin)
        record["rounds"][tag] = {"round": s, "m": m, "staged_ok": got["ok"]}
        plan.append(f"S\t{tag}\t{d}\t1")
        return d

    for s, m in enumerate(served):
        z, opening = dict(np.load(live / str(s) / "rows.npz")), json.loads((live / str(s) / "opening.json").read_text())
        problems += check_openings(pub, s, opening, files(pub, s, m))
        tag = f"honest-{s}"
        d = statement(tag, z, s, m, opening)
        if (d / "pub.bin").read_bytes() != (vp / f"{tag}.pub.bin").read_bytes():
            problems.append(f"{tag}: the prover's pub.bin is not the verifier's own file")
        if record["rounds"][tag]["staged_ok"] != 1:
            problems.append(f"{tag}: the unit's ok is 0 at the served rows m = {m}")
        plan.append(f"C\t{tag}\t{d}\t{vp / f'{tag}.pub.bin'}\t{vp / f'{tag}.registered.json'}\t{EXPECT['honest']}")
        if s == 0:
            o = vp / "m-other"
            reg = files(pub, 0, m + 1)
            (o.with_suffix(".registered.json")).write_text(json.dumps(reg))
            o.with_suffix(".pub.bin").write_bytes(verifier_pub(head, pub, 0, m + 1, opening, reg))
            plan.append(f"C\tm-other\t{d}\t{o.with_suffix('.pub.bin')}\t{o.with_suffix('.registered.json')}\t{EXPECT['m-other']}")
            c = statement("m-claimed", claimed(z, pub, 0, m + 1), 0, m + 1, opening)
            if record["rounds"]["m-claimed"]["staged_ok"] != 0:
                problems.append(f"m-claimed: the unit's ok is 1 at m = {m + 1}, past the served rows")
            plan.append(f"C\tm-claimed\t{c}\t{c / 'pub.bin'}\t{vp / 'm-claimed.registered.json'}\t{EXPECT['m-claimed']}")
    if S > 1 and len(served) > 1:
        d1 = vp / "honest-1"
        plan.append(f"C\tother-round\t{sp / 'honest-0'}\t{d1}.pub.bin\t{d1}.registered.json\t{EXPECT['other-round']}")
    record["problems"] = problems
    (work / "record.json").write_text(json.dumps(record, indent=1))
    (work / "plan.tsv").write_text("\n".join(plan) + "\n")
    log("PREPARED " + json.dumps({k: record[k] for k in ("T", "S", "served", "ands", "problems")}))
    return 1 if problems else 0


def live_stager(T: int, circuit: Path, part: dict):
    """Flock's stager of `live.template_v2(T)`, bound to the live Program's partition, its circuit written once."""
    from verity_flock import class_statement as CS
    stager = CS.UnitStager(L.template_v2(T), NAME, LR.statement_rows(T, v2=True), LR.STATEMENT_OUT_ROWS, circuit=circuit,
                           bind={"partition": P.digest(part), "program": part["program"]})
    stager.reference_check = live_reference(T)
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
    doc = {"ok": ok, "T": record["T"], "S": record["S"], "served": record["served"], "ands": record["ands"],
           "live_accepted": f"{len(accepted)}/{len(record['served'])}", "problems": record["problems"], "checks": checks,
           "circuit_sha512": record["circuit_sha512"], "program_sha512": record["program_sha512"],
           "partition": record["partition"], "inputs_sha512": hashlib.sha512((work / "plan.tsv").read_bytes()).hexdigest()}
    (out / "live-summary.json").write_text(json.dumps(doc, indent=1))
    log("LIVE " + json.dumps({k: doc[k] for k in ("ok", "live_accepted", "problems")}
                             | {"checks": {c["tag"]: [c["verdict"], c["expect"]] for c in checks}}))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--live", required=True)
    p.add_argument("--work", required=True)
    p = sub.add_parser("summarize")
    p.add_argument("--work", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return cmd_prepare(a) if a.cmd == "prepare" else cmd_summarize(a)


if __name__ == "__main__":
    raise SystemExit(main())
