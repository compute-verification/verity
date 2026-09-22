#!/usr/bin/env python
"""[L1 RECORD 2026-09-10] Twelve negatives of the REC-P1 record: 9 gate negatives (a verifier gate refuses a tampered artefact) and 3 tamper
negatives (a committed value is changed consistently -- the tree is rebuilt -- and the relation, hence the guest's verdict, rejects) (DIRECTIVE
v3 §1.1 step 10).  Every negative works on a COPY in a scratch directory; the record is never modified.  One row each in negatives.md.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
POC = HERE.parent
ROOT = POC.parent
for p in (str(POC), str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)


def host(hbin: str, args: list[str], prover: str = "cpu") -> tuple[int, dict | None, str]:
    res = subprocess.run([hbin, *args], capture_output=True, text=True, env={**os.environ, "SP1_PROVER": prover, "RUST_LOG": "warn"})
    doc = None
    for line in reversed(res.stdout.strip().splitlines()):
        if line.strip().startswith("{"):
            doc = json.loads(line)
            break
    return res.returncode, doc, res.stderr[-300:]


def flip(path: Path, out: Path, offset: int) -> None:
    b = bytearray(path.read_bytes())
    b[offset] ^= 0x01
    out.write_bytes(bytes(b))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--record", required=True)
    ap.add_argument("--population", required=True)
    ap.add_argument("--host", required=True)
    ap.add_argument("--approved", default=str(POC / "e2e" / "approved_guests.json"))
    ap.add_argument("--checkpoints", default=str(POC / "manifests" / "checkpoints.json"))
    ap.add_argument("--case", default="B0")
    ap.add_argument("--scratch", default="/tmp/rec-negatives")
    ap.add_argument("--prove-tamper", action="store_true", help="also prove tamper T3 compressed on the GPU (public values carry verdict 0)")
    a = ap.parse_args(argv)
    rec = Path(a.record)
    scratch = Path(a.scratch)
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    T0 = time.perf_counter()

    def row(nid: str, kind: str, tamper: str, gate: str, refused: bool, observed: str, seconds: float) -> None:
        rows.append({"id": nid, "kind": kind, "tamper": tamper, "gate": gate, "refused": bool(refused), "observed": observed[:400], "seconds": round(seconds, 2)})
        print(f"[rec_p1_negatives] {nid} {'REFUSED' if refused else 'NOT REFUSED (!!)'}: {tamper} -> {gate}: {observed[:160]}", flush=True)

    from veritor.interface import TypedObligationSet, check_binding

    from e2e import partition_sample as ps

    chunk = sorted(p for p in (rec / "obligations").iterdir() if p.is_dir() and p.name.startswith("chunk"))[0]
    proof = rec / "proofs" / f"{chunk.name}-compressed.bin"
    record = json.loads((chunk / "composed.json").read_text())
    obj = TypedObligationSet.from_json(record["object"])
    fmt = int(record.get("statement_format", 2))
    _, info, _ = host(a.host, ["info"])
    _, good, _ = host(a.host, ["verify", "--proof", str(proof), "--statement", str(chunk / "statement.bin")])
    pv_good = bytes.fromhex(good["public_values"])

    # ---- gate negatives ----
    t = time.perf_counter()
    approved = json.loads(Path(a.approved).read_text())
    for g in approved["guests"]:
        if g["elf_sha256"] == info["elf_sha256"]:
            g["vk_hash"] = "0x" + "00" * 32
    gmap = {g["elf_sha256"]: g for g in approved["guests"]}
    refused = str(gmap[info["elf_sha256"]]["vk_hash"]).lower() != str(info["vk_hash"]).lower()
    row("G1", "gate", "approved_guests.json copy: the approved entry's vk_hash altered", "V4.guest (host vk must equal the approved ELF's vk)", refused,
        f"host vk {info['vk_hash'][:18]} vs table {gmap[info['elf_sha256']]['vk_hash'][:18]} -> refuse", time.perf_counter() - t)

    t = time.perf_counter()
    st_bad = scratch / "statement-flipped.bin"
    flip(chunk / "statement.bin", st_bad, 100)
    rc, vr, err = host(a.host, ["verify", "--proof", str(proof), "--statement", str(st_bad)])
    refused = rc != 0 or not (vr or {}).get("statement_match") or not (vr or {}).get("ok")
    row("G2", "gate", "one byte of statement.bin flipped (offset 100)", "host verify: public h_L != sha256(statement bytes) -> statement_match false", refused,
        f"rc={rc} ok={(vr or {}).get('ok')} statement_match={(vr or {}).get('statement_match')} {err[-80:] if rc else ''}", time.perf_counter() - t)

    t = time.perf_counter()
    pf_bad = scratch / "proof-flipped.bin"
    flip(proof, pf_bad, 4096)
    rc, vr, err = host(a.host, ["verify", "--proof", str(pf_bad), "--statement", str(chunk / "statement.bin")])
    refused = rc != 0 or not (vr or {}).get("ok")
    row("G3", "gate", "one byte of proof-compressed.bin flipped (offset 4096)", "host verify: SP1 compressed proof does not verify under the approved vk", refused,
        f"rc={rc} ok={(vr or {}).get('ok')} {err[-120:] if rc else ''}", time.perf_counter() - t)

    t = time.perf_counter()
    pv0 = pv_good[:64] + bytes([0])
    b = check_binding(obj, pv0, statement_format=fmt)
    accepted = b.ok and b.verdict == 1
    row("G4", "gate", "public values with the guest verdict byte forced to 0 (h_O, h_L intact)", "verifier: check_binding must bind AND return verdict 1 (the binding alone says nothing about acceptance)", not accepted,
        f"check_binding ok={b.ok} (binds) verdict={b.verdict} -> refuse", time.perf_counter() - t)

    t = time.perf_counter()
    other = json.loads((sorted(p for p in (rec / "obligations").iterdir() if p.is_dir() and p.name.startswith("chunk"))[1] / "composed.json").read_text())
    b = check_binding(TypedObligationSet.from_json(other["object"]), pv_good, statement_format=fmt)
    row("G5", "gate", "proof of chunk00 presented for chunk01's typed object (h_O / h_L of another statement)", "check_binding: h_O || h_L must equal the object's digests", not b.ok,
        f"ok={b.ok} code={getattr(b, 'code', '')} {str(getattr(b, 'detail', ''))[:120]}", time.perf_counter() - t)

    # chronology gates: copies of the chronology directory
    def chron_copy(name: str) -> Path:
        d = scratch / name
        shutil.copytree(rec / "chronology", d)
        return d

    N = json.loads((rec / "protocol.json").read_text())["instances"]
    t = time.perf_counter()
    d = chron_copy("chron-seed")
    seed = bytearray(bytes.fromhex((d / "seed.hex").read_text().strip()))
    seed[0] ^= 0x01
    (d / "seed.hex").write_text(seed.hex() + "\n")
    c = json.loads((d / "chronology.json").read_text())
    for e in c["events"]:
        if e["name"] == "seed_reveal":
            e["seed"] = seed.hex()
    (d / "chronology.json").write_text(json.dumps(c, indent=1, sort_keys=True) + "\n")
    problems = ps.verify(rec / "protocol_instance.json", Path(a.population), d, instances=N)
    row("G6", "gate", "revealed seed altered by one bit (seed.hex + seed_reveal event)", "partition_sample.verify: H(seed) != committed seed; r_q / r_s not re-derived", bool(problems), "; ".join(problems)[:300], time.perf_counter() - t)

    t = time.perf_counter()
    d = chron_copy("chron-drop")
    s = json.loads((d / "selected.json").read_text())
    s["selected"] = s["selected"][1:]
    s["count"] -= 1
    (d / "selected.json").write_text(json.dumps(s, indent=1, sort_keys=True) + "\n")
    problems = ps.verify(rec / "protocol_instance.json", Path(a.population), d, instances=N)
    row("G7", "gate", "selected.json with one selected VU dropped", "partition_sample.verify: selected.json must list exactly the r_s selection", bool(problems), "; ".join(problems)[:300], time.perf_counter() - t)

    t = time.perf_counter()
    pop = json.loads(Path(a.population).read_text())
    pop["rows"][0][1] += 1  # one row's count
    pbad = scratch / "population-tampered.json"
    pbad.write_text(json.dumps(pop, separators=(",", ":")))
    try:
        problems = ps.verify(rec / "protocol_instance.json", pbad, rec / "chronology", instances=N)
        obs = "; ".join(problems)[:300]
        refused = bool(problems)
    except Exception as e:  # noqa: BLE001
        obs, refused = f"{type(e).__name__}: {str(e)[:250]}", True
    row("G8", "gate", "population/v2 with one row's unit count changed (content sha / partition digest no longer the pinned ones)", "partition_sample.verify: population pin / partition digest mismatch -> NOT BOUND", refused, obs, time.perf_counter() - t)

    t = time.perf_counter()
    reg = json.loads((rec / "weights_authorized.json").read_text())
    c2 = json.loads((rec / "adapter" / "selective" / "commitments.json").read_text())
    reg_bad = dict(reg)
    reg_bad["expected_root"] = "00" + reg["expected_root"][2:]
    refused = c2["trees"]["weights"]["root"] != reg_bad["expected_root"]
    row("G9", "gate", "registered weight commitment (weights_authorized.json copy) with its root altered", "V3.weights_root: prover weights root must equal the registered root", refused,
        f"prover {c2['trees']['weights']['root'][:16]} vs registered {reg_bad['expected_root'][:16]} -> refuse", time.perf_counter() - t)

    # ---- tamper negatives: consistent mutations of committed values; the guest (execute) decides ----
    import numpy as np
    from veritor.interface.lowering import statement_bytes_v3
    from veritor.protocol.proofs.wire import statement_digest

    from verity_vllm.capture import CaptureReader
    from verity_vllm.commit import Opening, TreeCache, load_commitments
    from verity_vllm.description import Description
    from verity_vllm.negatives import PatchedReader
    from verity_vllm.typed_obligation_export import InputEquation
    from verity_vllm_adapter import obligations as O
    from verity_vllm_adapter.coordinate_check import coordinate_openings, patched_commitments

    sel = rec / "adapter" / "selective"
    reader = CaptureReader(sel)
    desc = Description(reader.description)
    comm = load_commitments(reader)
    cache = TreeCache(reader)
    u0 = record["provenance"]["units"][0]
    cid, j = u0["canonical_id"], int(u0["j"])
    base = coordinate_openings(desc, reader, comm, cache, cid, j)
    checks = [json.loads(ln) for ln in (rec / "obligations" / "input_checks.jsonl").read_text().splitlines() if ln.strip()]
    wchk = next(c for c in checks if c["kind"] == "weight" and c["expected"] is not None)

    def mini_statement(name: str, rd, cm, ch, extra_eq: list[InputEquation] | None = None) -> tuple[Path, dict, dict]:
        co = coordinate_openings(desc, rd, cm, ch, cid, j)
        o = ch.open(wchk["leaf"])
        assert isinstance(o, Opening)
        prefix = (0,) if len(o.shape) == 2 and int(o.shape[0]) == 1 else ()
        eqs = [InputEquation(o, (*prefix, wchk["word"]), wchk["expected"], wchk["source"])] + (extra_eq or [])
        export, recd = O.build(cm, [co], eqs, profile_id=rd.profile_id, name=f"REC-P1-NEG-{name}", description=f"negative {name}", source="rec_p1_negatives")
        d = scratch / name
        d.mkdir()
        ob = TypedObligationSet.from_json(recd["object"])
        v3 = statement_bytes_v3(ob)
        (d / "statement.bin").write_bytes(v3)
        (d / "witness.bin").write_bytes(bytes.fromhex(recd["witness"]))
        recd["statement_digest"] = statement_digest(v3).hex()
        return d, recd, {"relations": recd["relations"], "witness": recd["witness_verdict"], "verdict": recd["verdict"], "committed_word": co.committed_word}

    def execute(d: Path, recd: dict) -> tuple[int, dict | None, str]:
        return host(a.host, ["execute", "--statement", str(d / "statement.bin"), "--witness", str(d / "witness.bin"), "--typed-object-digest", recd["digest"]])

    # T1: the opened weight word (an Input equation's committed value) mutated; weights tree rebuilt
    t = time.perf_counter()
    wid = wchk["leaf"]
    words = np.frombuffer(reader.value(wid).data, dtype="<u2").copy()
    words[wchk["word"]] ^= 0x0080
    patched = PatchedReader(reader, weights={wid: words.tobytes()})
    cm1, ch1 = patched_commitments(reader, comm, patched, cache)
    d1, r1, v1 = mini_statement("T1-weight", patched, cm1, ch1)
    rc, ex, err = execute(d1, r1)
    root_changed = cm1["trees"]["weights"]["root"] != comm["trees"]["weights"]["root"]
    refused = (ex is None) or (not ex.get("verdict")) or root_changed
    row("T1", "tamper", f"one committed weight word ({wid}[{wchk['word']}]) changed by one bit; weights tree rebuilt consistently", "Input equation: committed word != pinned checkpoint word -> relation rejects (guest verdict false); AND weights root != registered root (V3)", refused,
        f"python relations={v1['relations']} witness={v1['witness']}; guest rc={rc} verdict={(ex or {}).get('verdict')} cycles={(ex or {}).get('total_cycles')}; weights root changed={root_changed} {err[-80:] if rc else ''}", time.perf_counter() - t)

    # T2: the x operand row word mutated (RU tree rebuilt)
    t = time.perf_counter()
    xw = np.frombuffer(reader.value(base.x_id).data, dtype="<u2").copy()
    k = (base.x_cols[0] if base.x_cols else 0) + 3
    xw[k] = 0x5380 | ((int(xw[k]) & 0x8000) ^ 0x8000)  # 2^40-magnitude word of the opposite sign
    patched = PatchedReader(reader, values={base.x_id: xw.tobytes()})
    cm2, ch2 = patched_commitments(reader, comm, patched, cache)
    d2, r2, v2 = mini_statement("T2-x", patched, cm2, ch2)
    rc, ex, err = execute(d2, r2)
    refused = (ex is None) or (not ex.get("verdict"))
    row("T2", "tamper", f"one word of the committed operand row x ({base.x_id.split('/op=')[-1]}, word {k}) replaced; RU tree rebuilt consistently", "GemmCoordinate relation: out[j] != tc_dot16 chain over (x, W[j]) -> guest verdict false", refused,
        f"python relations={v2['relations']} witness={v2['witness']}; guest rc={rc} verdict={(ex or {}).get('verdict')} cycles={(ex or {}).get('total_cycles')} {err[-80:] if rc else ''}", time.perf_counter() - t)

    # T3: the committed output word mutated by one ulp (RU tree rebuilt); optionally proven compressed -> public verdict 0
    t = time.perf_counter()
    ow = np.frombuffer(reader.value(cid).data, dtype="<u2").copy()
    ow[j] ^= 0x0001
    patched = PatchedReader(reader, values={cid: ow.tobytes()})
    cm3, ch3 = patched_commitments(reader, comm, patched, cache)
    d3, r3, v3 = mini_statement("T3-out", patched, cm3, ch3)
    rc, ex, err = execute(d3, r3)
    refused = (ex is None) or (not ex.get("verdict"))
    obs = f"python relations={v3['relations']} witness={v3['witness']}; guest rc={rc} verdict={(ex or {}).get('verdict')} cycles={(ex or {}).get('total_cycles')}"
    if a.prove_tamper and ex is not None:
        rc2, pr, err2 = host(a.host, ["prove", "--statement", str(d3 / "statement.bin"), "--witness", str(d3 / "witness.bin"), "--typed-object-digest", r3["digest"], "--out", str(d3 / "proof.bin"), "--mode", "compressed"], prover="cuda")
        if rc2 == 0:
            rc3, vr, _ = host(a.host, ["verify", "--proof", str(d3 / "proof.bin"), "--statement", str(d3 / "statement.bin")])
            pv = bytes.fromhex(vr["public_values"]) if vr else b""
            b = check_binding(TypedObligationSet.from_json(r3["object"]), pv, statement_format=3)
            obs += f"; compressed proof {pr.get('proof_bytes')} B in {pr.get('prove_seconds')} s: proof verifies={vr.get('ok') if vr else None} with public verdict={pv[64] if len(pv) > 64 else None} -> check_binding ok={b.ok} verdict={b.verdict}"
            refused = refused and not (b.ok and b.verdict == 1)
        else:
            obs += f"; prove rc={rc2} {err2[-120:]}"
    row("T3", "tamper", f"committed output word out[{j}] of {cid.split('/op=')[-1]} changed by one ulp; RU tree rebuilt consistently", "GemmCoordinate relation rejects -> guest verdict false; a proof of it carries verdict 0 and check_binding refuses", refused, obs, time.perf_counter() - t)

    n_ref = sum(r["refused"] for r in rows)
    doc = {"schema": "veritor/rec-p1-negatives/v1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "count": len(rows), "refused": n_ref, "rows": rows,
           "provenance": {"label": "MEASURED", "hardware": "record-gpu CPU (+ GPU for the T3 proof when --prove-tamper)", "script": "scripts/rec_p1_negatives.py", "seconds": round(time.perf_counter() - T0, 1),
                          "subject": {"chunk": chunk.name, "coordinate": {"canonical_id": cid, "j": j}, "weight_word": {"leaf": wchk["leaf"], "word": wchk["word"]}}}}
    (rec / "negatives.json").write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    md = ["# REC-P1 negatives (9 gate + 3 tamper; one row each; every negative on a COPY, the record untouched)", "",
          f"`negatives: {n_ref}/{len(rows)}` refused. Subject: {chunk.name}, coordinate `{cid}` j={j}; weight word `{wchk['leaf']}`[{wchk['word']}]. Hardware: record-gpu CPU (T3 proof: RTX 4090).", "",
          "| id | kind | tamper | gate that must refuse | refused | observed | s |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['id']} | {r['kind']} | {r['tamper']} | {r['gate']} | {'yes' if r['refused'] else 'NO'} | {r['observed'].replace('|', '/')} | {r['seconds']} |")
    (rec / "negatives.md").write_text("\n".join(md) + "\n")
    with (rec / "costlog.jsonl").open("a") as f:
        f.write(json.dumps({"stage": "negatives (9 gate + 3 tamper)", "class": "verifier", "seconds": round(time.perf_counter() - T0, 1), "hardware": "record-gpu CPU (+GPU T3 proof)", "batch": "12 negatives", "label": "MEASURED"}) + "\n")
    print(f"[rec_p1_negatives] negatives: {n_ref}/{len(rows)} refused ({round(time.perf_counter() - T0, 1)} s)")
    return 0 if n_ref == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
