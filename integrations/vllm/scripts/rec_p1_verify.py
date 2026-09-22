#!/usr/bin/env python
# ============================================================================================================================
# LEGACY (DIRECTIVE v4) — NOT THE CONTRACT VERIFIER.  Kept only to reproduce the REC-P1 historical record.
#   * B_ex here is the count of SELECTED VUs of class "exempt (charged)" (line ~348) — the contract's B_ex is a count of
#     exempted RUs per audit (docs/PROTOCOL.md; docs/PROTOCOL-CONFORMANCE.md R8/R11).
#   * Input kinds are sampled VUs of the KindTable (lines ~105-143) — the contract has no input VUs (R2).
#   * V5 matches obligations against prover-written provenance["units"] (R4, A2's M1).
#   * The certificate emits allowance_B_ex / exempt_B_ex / "exempt (charged)", which e2e/certificate_v2.py refuses.
# Refuses to run unless --legacy-directive-v4 is passed (R14: legacy unselectable by accident).  With the flag, behaviour is
# unchanged.  The contract verifier is e2e/verifier_v3 (steps 1-4 live: driver.py; step 5: reverify.py); the
# contract runner is record_v5/ (A9).  [A10 2026-09-10]
# ============================================================================================================================
"""[L1 RECORD 2026-09-10] The verifier's checks, the certified Bound and the certificate of the REC-P1 record (DIRECTIVE v3 §1.1 steps 8-9).

LEGACY (DIRECTIVE v4): see the banner above; requires ``--legacy-directive-v4``.  Contract path: ``e2e.verifier_v3`` / ``record_v5``.

Everything here is re-derived from the record's files by the verifier role (nothing is copied from prover-side summaries without a check):

 V1  instance / population / partition digests: the population file's sha256 is the instance's pin; its partition digest is the validated
     P1/cu=whole digest the expressibility table and the selection carry; program digest as pinned.
 V2  chronology re-derivation (``e2e.partition_sample.verify``): seed commitment, r_q (with the logged redraw chain), r_s, selected.json.
 V3  commitments: sha256(commitments_phase0.json) == the chronology's boundary digest; sha256(commitments.json) == its interior digest;
     the weights tree root == the registered weight commitment's root (ROT-INPUT); weights / request trees unchanged by the replay.
 V4  guest: every proof's vk == the approved ELF's vk (e2e/approved_guests.json); host ``verify`` of every compressed proof against its
     statement bytes; ``check_binding`` of the public values (h_O || h_L || verdict = 1) against the chunk's typed object.
 V5  expected statements: every selected VU of a provable kind is exactly one GemmCoordinate obligation (its Gemm / t / j as located in the
     pinned program), every selected input unit is an Input equation whose expected word is the pinned checkpoint's (or the served prompt's),
     every root a statement references is the registered weights root or a committed tree of the selected RUs; no extra obligation.
 V6  input checks (``obligations/input_checks.jsonl``): every selected input unit's committed word == the prescribed value.

The Bound: ``bound_certified(KindTable(P1 as validated, x N_REQ instances), VerificationPolicy(q, s), delta, allowance = B_ex)`` with B_ex = the
realized number of selected VUs whose kind is `exempt (charged)`; if the certified result is the cap the certificate says (BOUND, capped) and the
pre-cap value is reported as DERIVED beside it.  Any failed check -> `NOT BOUND: <reason>`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
POC = HERE.parent
ROOT = POC.parent
for p in (str(POC), str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

P1_DIGEST = "56ebcdf53f302040870776981860ffcfafd161b1b962c759df8833057aa75b9a"

# [A10 2026-09-10] R14 legacy guard: this script is DIRECTIVE-v4 legacy and refuses to run unless the flag is passed explicitly.
LEGACY_FLAG = "--legacy-directive-v4"
LEGACY_POINTER = (
    f"{Path(__file__).name}: LEGACY (DIRECTIVE v4) — not the contract verifier; refusing to run without {LEGACY_FLAG}.\n"
    "  B_ex here = selected 'exempt (charged)' VUs and Input kinds are sampled VUs; the contract (docs/PROTOCOL.md) has neither.\n"
    "  Contract verifier: e2e/verifier_v3 (driver.py steps 1-4, reverify.py step 5); contract runner: record_v5/ (A9).\n"
    f"  Historical reproduction of the REC-P1 record: re-run with {LEGACY_FLAG} (behaviour is otherwise unchanged)."
)


def legacy_gate(argv: list[str] | None) -> list[str] | None:
    """Return ``argv`` without the legacy flag, or ``None`` (after printing the pointer) when it is absent.  ``-h/--help`` pass."""
    args = list(sys.argv[1:] if argv is None else argv)
    if LEGACY_FLAG in args:
        return [x for x in args if x != LEGACY_FLAG]
    if any(x in ("-h", "--help") for x in args):
        return args
    print(LEGACY_POINTER, file=sys.stderr)
    return None


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Log:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lines: list[str] = []
        self.fails: list[str] = []
        self.n = 0

    def check(self, code: str, ok: bool, what: str, **kw) -> bool:
        self.n += 1
        tail = " ".join(f"{k}={str(v)[:96]}" for k, v in kw.items())
        line = f"{utc()} {code} {'PASS' if ok else 'FAIL'} {what} {tail}".rstrip()
        self.lines.append(line)
        print("[rec_p1_verify]", line[:220], flush=True)
        if not ok:
            self.fails.append(f"{code}: {what}")
        return ok

    def note(self, text: str) -> None:
        self.lines.append(f"{utc()} NOTE {text}")

    def write(self) -> None:
        self.path.write_text("\n".join(self.lines) + "\n")


def host(hbin: str, args: list[str]) -> dict:
    res = subprocess.run([hbin, *args], capture_output=True, text=True, env={**os.environ, "SP1_PROVER": "cpu", "RUST_LOG": "warn"})
    if res.returncode != 0:
        raise RuntimeError(res.stderr[-400:])
    for line in reversed(res.stdout.strip().splitlines()):
        if line.strip().startswith("{"):
            return json.loads(line)
    raise RuntimeError("no json")


def unit_matches(u: dict, pu: dict) -> bool:
    """The mapping's located unit (request_id, t, layer, op_suffix, j) is the obligation's (canonical_id, j)."""
    cid = str(pu.get("canonical_id", ""))
    f = dict(part.split("=", 1) for part in cid.split("/") if "=" in part)
    loc = u.get("locate") or {}
    if loc.get("role") == "final":  # the lm_head GEMM: no layer, canonical op = lm_head.gemm
        want_l, want_op = "lm_head", "lm_head.gemm"
    else:
        want_l, want_op = str(loc.get("layer")), f"layers.{loc.get('layer')}.{loc.get('op_suffix')}"
    return (f.get("r") == u.get("request_id") and str(f.get("t")) == str(loc.get("t")) and str(f.get("l")) == want_l
            and f.get("op") == want_op and int(pu.get("j", -1)) == int(loc.get("j", -2)))


def kind_table(pop: dict[str, Any], instances: int, out_bits_per_ru: int):
    """A ``veritor/kind-table/v1`` of the validated P1/cu=whole partition replicated over ``instances`` requests: one REPLAY kind (the
    request, one CU; exit cut = its designated outputs), one VERIFICATION kind per population kind (copies = count x instances; cut =
    min(unit out bits, root out bits)); reach/ancestor bits set to the root's out bits (the sound, conservative direction)."""
    from veritor.plans.summary import kind_table_from_json

    counts: Counter = Counter()
    for r in pop["rows"]:  # population/v2 row: [vu_start, count, ..., kind_id, ...]
        counts[int(r[3])] += int(r[1])
    kinds = [{**k, "count": counts.get(int(k["id"]), 0)} for k in pop["kinds"]["entries"]]
    assert sum(counts.values()) == int(pop["vu_count"]), (sum(counts.values()), pop["vu_count"])
    rows: list[dict[str, Any]] = []
    root_bits = out_bits_per_ru * instances
    vk: list[list[Any]] = []
    total_units = 0
    for k in kinds:
        name = f"{k['fn']}|{k['form']}|{k['size']}"
        c = int(k.get("count", 0))
        if c == 0:
            continue
        total_units += c
        is_input = k["fn"] == "Input"
        ob = int(k["out_bits"])
        rows.append({"kind": name, "role": "verification", "copies": c * instances, "size": int(k["size"]), "replay_cost": 0, "proof_cost": 0,
                     "input_count": 0 if is_input else 1, "out_count": 1, "out_bits": ob, "interior_count": 0, "reach_bits": root_bits, "ancestor_bits": root_bits,
                     "source_inputs": 0, "source_weights": int(k["size"]) if is_input else 0, "min_depth": 2, "max_depth": 2, "children": [],
                     "verification_units": 1, "verification_kinds": [[name, 1]], "closed": True})
        vk.append([name, c])
    ru = {"kind": "RU:request", "role": "replay", "copies": instances, "size": int(pop["gates"]), "replay_cost": 0, "proof_cost": 0, "input_count": 0, "out_count": out_bits_per_ru // 32,
          "out_bits": out_bits_per_ru, "interior_count": int(sum(int(k.get("count", 0)) for k in kinds)), "reach_bits": root_bits, "ancestor_bits": root_bits,
          "source_inputs": 0, "source_weights": int(sum(int(k["size"]) * int(k.get("count", 0)) for k in kinds if k["fn"] == "Input")), "min_depth": 1, "max_depth": 1,
          "children": [[r["kind"], int(r["copies"]) // instances] for r in rows], "verification_units": total_units, "verification_kinds": vk, "closed": True}
    root = {"kind": "Root", "role": None, "copies": 1, "size": int(pop["gates"]) * instances, "replay_cost": 0, "proof_cost": 0, "input_count": 0, "out_count": (out_bits_per_ru // 32) * instances,
            "out_bits": root_bits, "interior_count": 0, "reach_bits": root_bits, "ancestor_bits": root_bits, "source_inputs": 0, "source_weights": ru["source_weights"] * instances,
            "min_depth": 0, "max_depth": 0, "children": [["RU:request", instances]], "verification_units": total_units * instances, "verification_kinds": [[n, c * instances] for n, c in vk], "closed": True}
    doc = {"schema": "veritor/kind-table/v1", "rows": [root, ru, *rows], "root": "Root", "n": int(pop["gates"]) * instances, "input_count": 0,
           "weight_count": root["source_weights"], "replay_unit_count": instances, "digest": ""}
    doc["digest"] = "sha256:" + hashlib.sha256(json.dumps({k: v for k, v in doc.items() if k != "digest"}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return kind_table_from_json(doc), doc


def main(argv: list[str] | None = None) -> int:
    argv = legacy_gate(argv)
    if argv is None:
        return 2
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(LEGACY_FLAG, action="store_true", help="REQUIRED: acknowledge this is the DIRECTIVE-v4 legacy verifier (see banner)")
    ap.add_argument("--record", required=True)
    ap.add_argument("--population", required=True)
    ap.add_argument("--expressibility", required=True)
    ap.add_argument("--host", required=True)
    ap.add_argument("--approved", default=str(POC / "e2e" / "approved_guests.json"))
    ap.add_argument("--out-bits-per-ru", type=int, default=256, help="the request CU's declared output domain: 16 token slots x 16 declared bits (vocabulary 49152 < 2^16; wire width 32 bits per slot; enforced by V7.output_domain)")
    ap.add_argument("--resolution", type=int, default=16)
    ap.add_argument("--verify-workers", type=int, default=7)
    ap.add_argument("--reuse-host-verify", action="store_true", help="reuse verify/host_verify.jsonl written by an earlier run of this verifier on this record (logged)")
    ap.add_argument("--dry-run-no-host-verify", action="store_true", help="DEV ONLY: skip host verify (reuse proofs.json); writes to verify-dry/, never a record")
    a = ap.parse_args(argv)
    rec = Path(a.record)
    vdir = rec / ("verify-dry" if a.dry_run_no_host_verify else "verify")
    vdir.mkdir(exist_ok=True)
    log = Log(vdir / "verifier.log")
    T0 = time.perf_counter()

    from veritor.interface import TypedObligationSet, check_binding

    from e2e import partition_sample as ps

    inst = ps.load_instance(rec / "protocol_instance.json")
    inst_doc = json.loads((rec / "protocol_instance.json").read_text())
    idig = ps.instance_digest(inst)
    proto = json.loads((rec / "protocol.json").read_text())
    chron = json.loads((rec / "chronology" / "chronology.json").read_text())
    selected = json.loads((rec / "chronology" / "selected.json").read_text())
    expr = json.loads(Path(a.expressibility).read_text())
    N = int(proto["instances"])

    # V1 digests
    pop_sha = sha256_file(Path(a.population))
    pop_doc = json.loads(Path(a.population).read_text())
    log.check("V1.instance", idig == proto["instance_digest"] == chron["instance_digest"] == selected["instance_digest"], "instance digest pinned by protocol, chronology and selection", digest=idig[:16])
    pin_sha = chron["population"]["content_sha256"]
    log.check("V1.population", pop_sha == proto["population_file_sha256"] and pin_sha == pop_doc["content_sha256"] == selected["population"]["content_sha256"] and chron["population"]["instances"] == N,
              "population file sha256 == protocol pin; content sha256 == chronology/selection pin; instances == N_REQ", sha=pop_sha[:16], content=pin_sha[:16], instances=N)
    log.check("V1.partition", pop_doc["partition_digest"] == P1_DIGEST == selected["partition_digest"] == expr["partition_digest"], "partition digest == validated P1/cu=whole (COMPACT-P1)", digest=P1_DIGEST[:16])
    log.check("V1.program", pop_doc["program_digest"] == expr["program_digest"] == (json.loads((rec / "maps.json").read_text())["program"]["digest"]), "program digest pinned", digest=pop_doc["program_digest"][:16])
    ob = json.loads((rec / "obligations" / "obligations.json").read_text())
    log.check("V1.program_rebuilt", ob["provenance"]["program"].get("digest") == pop_doc["program_digest"], "program rebuilt for the mapping has the pinned digest", digest=str(ob["provenance"]["program"].get("digest"))[:16])
    # V2 chronology
    t = time.perf_counter()
    problems = ps.verify(rec / "protocol_instance.json", Path(a.population), rec / "chronology", instances=N)
    log.check("V2.chronology", not problems, "chronology re-derived (seed, r_q incl. redraws, r_s, selected.json)", problems=problems, seconds=round(time.perf_counter() - t, 1))
    rq = next(e for e in chron["events"] if e["name"] == "r_q")
    conditioned = bool(rq.get("redraws", 0)) or bool(rq.get("conditioned"))
    # V3 commitments
    sel = rec / "adapter" / "selective"
    b_dig, i_dig = sha256_file(sel / "commitments_phase0.json"), sha256_file(sel / "commitments.json")
    b_ev = next((e for e in chron["events"] if e["name"] in ("boundary_commitment", "boundary")), None)
    i_ev = next((e for e in chron["events"] if e["name"] in ("interior_commitment", "interior")), None)
    b_logged = (b_ev or {}).get("digest") or (b_ev or {}).get("boundary_digest")
    i_logged = (i_ev or {}).get("digest") or (i_ev or {}).get("interior_digest")
    log.check("V3.boundary", b_dig == proto["boundary_digest"] and (b_logged in (None, b_dig)), "sha256(commitments_phase0.json) == chronology boundary digest", digest=b_dig[:16], logged=str(b_logged)[:16])
    log.check("V3.interior", i_dig == proto["interior_digest"] and (i_logged in (None, i_dig)), "sha256(commitments.json) == chronology interior digest", digest=i_dig[:16], logged=str(i_logged)[:16])
    c0 = json.loads((sel / "commitments_phase0.json").read_text())
    c2 = json.loads((sel / "commitments.json").read_text())
    reg = json.loads((rec / "weights_authorized.json").read_text())
    w_root = c2["trees"]["weights"]["root"]
    log.check("V3.weights_root", w_root == c0["trees"]["weights"]["root"] == reg["expected_root"], "prover weights root == registered weight commitment (ROT-INPUT), unchanged by replay", root=w_root[:16])
    unchanged = all(c2["trees"].get(k, {}).get("root") == v.get("root") for k, v in c0["trees"].items())
    log.check("V3.no_recommit", unchanged, "every phase-0 tree (weights, request boundaries) has the same root after the replay", trees=len(c0["trees"]))
    J = proto["J"]
    sel_rids = {proto["ru_map"]["rus"][r]["request_id"] for r in J}
    allowed_roots = {"weights": w_root}
    for tid, tr in c2["trees"].items():
        if tid.startswith("ru/") and tid.split("/")[1] in sel_rids:
            allowed_roots[tid] = tr["root"]
        if tid.startswith("boundary/") and tid.split("/")[1] in sel_rids:
            allowed_roots[tid] = tr["root"]
    # V4 guest + proofs
    approved = json.loads(Path(a.approved).read_text())
    g = {x["elf_sha256"]: x for x in approved["guests"]}
    info = host(a.host, ["info"])
    guest = g.get(info["elf_sha256"])
    log.check("V4.guest", guest is not None and str(guest["vk_hash"]).lower() == str(info["vk_hash"]).lower() and info["elf_sha256"].startswith("9fd00e60"),
              "host guest is the approved ELF 9fd00e60 (vk matches approved_guests.json)", elf=info["elf_sha256"][:16], vk=info["vk_hash"][:18], approval=(guest or {}).get("approval"))
    pdir = rec / "proofs"
    proofs = json.loads((pdir / "proofs.json").read_text()) if (pdir / "proofs.json").exists() else {}
    log.check("V4.prover_summary", bool(proofs), "proofs.json written by the prover (all chunks attempted)", present=bool(proofs))
    chunk_dirs = sorted(p for p in (rec / "obligations").iterdir() if p.is_dir() and p.name.startswith("chunk"))
    proven: dict[str, dict] = {}
    from concurrent.futures import ThreadPoolExecutor

    def _verify(cdir: Path):
        pf = pdir / f"{cdir.name}-compressed.bin"
        if not ((pdir / f"{cdir.name}.json").exists() and pf.exists()):
            return cdir, None, None
        try:
            return cdir, host(a.host, ["verify", "--proof", str(pf), "--statement", str(cdir / "statement.bin")]), None
        except RuntimeError as e:
            return cdir, None, str(e)[-160:]

    if a.reuse_host_verify:
        # the host verify results of THIS record's earlier run on this machine (verify/host_verify.jsonl); the log says so
        rows = [json.loads(ln) for ln in (vdir / "host_verify.jsonl").read_text().splitlines() if ln.strip()]
        by = {r["chunk"]: r for r in rows}
        verified = [(c, by[c.name]["host_verify"], by[c.name]["error"]) if c.name in by else (c, None, "no host_verify row") for c in chunk_dirs]
        log.note(f"host verify results reused from {vdir / 'host_verify.jsonl'} (written by this verifier's run of {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime((vdir / 'host_verify.jsonl').stat().st_mtime))} on this machine)")
    elif a.dry_run_no_host_verify:
        # DEV ONLY (never a record): reuse the prover-side host verify result recorded in proofs.json to exercise V5/V6/Bound without 50 CPU-min
        by_chunk = {r["chunk"]: r for r in proofs.get("results", [])}
        verified = [(c, {"ok": True, "statement_match": True, "verdict": True, "unsound": False, "vk_hash": info["vk_hash"], "public_values": by_chunk[c.name]["h_O"] + by_chunk[c.name]["h_L"] + "01"}, None)
                    if c.name in by_chunk else (c, None, "no prover-side record") for c in chunk_dirs]
        log.note("DRY RUN: V4.proof lines below reuse the prover-side host verify result (NOT an independent verification)")
    else:
        with ThreadPoolExecutor(max_workers=a.verify_workers) as ex:
            verified = list(ex.map(_verify, chunk_dirs))
        with (vdir / "host_verify.jsonl").open("w") as f:
            for cdir, vr, err in verified:
                f.write(json.dumps({"chunk": cdir.name, "host_verify": vr, "error": err}) + "\n")
    for cdir, vr, err in verified:
        if vr is None and err is None:
            log.check(f"V4.proof.{cdir.name}", False, "no compressed proof for this chunk", chunk=cdir.name)
            continue
        if vr is None:
            log.check(f"V4.proof.{cdir.name}", False, "host verify failed", err=err)
            continue
        record = json.loads((cdir / "composed.json").read_text())
        st = cdir / "statement.bin"
        h_l = hashlib.sha256(st.read_bytes()).hexdigest()
        pv = bytes.fromhex(vr["public_values"])
        obj = TypedObligationSet.from_json(record["object"])
        b = check_binding(obj, pv, statement_format=int(record.get("statement_format", 2)))
        ok = bool(vr.get("ok")) and bool(vr.get("statement_match")) and bool(vr.get("verdict")) and not vr.get("unsound") and str(vr.get("vk_hash")).lower() == str(info["vk_hash"]).lower() and b.ok and b.verdict == 1 \
            and h_l == record["statement_digest"] and pv[32:64].hex() == h_l and pv[:32].hex() == record["digest"] and pv[64] == 1
        log.check(f"V4.proof.{cdir.name}", ok, "compressed proof verifies; vk approved; public values = h_O||h_L||1 bound to the typed object", obligations=len(obj.obligations), h_L=h_l[:12], binding=b.ok)
        if ok:
            proven[cdir.name] = record
    # V5 expected statements
    units = [json.loads(ln) for ln in (rec / "mapping.units.jsonl").read_text().splitlines() if ln.strip()]
    by_vu = {u["vu_index"]: u for u in units}
    want_coord = {u["vu_index"] for u in units if str(u["class"]).startswith("provable")}
    want_input = {u["vu_index"] for u in units if str(u["class"]).startswith("input")}
    exempt = [u for u in units if str(u["class"]).startswith("exempt")]
    log.check("V5.selection", set(by_vu) == {r["vu_index"] for r in selected["selected"]} and len(units) == selected["count"], "mapping covers exactly the selected VUs", selected=selected["count"])
    checks = [json.loads(ln) for ln in (rec / "obligations" / "input_checks.jsonl").read_text().splitlines() if ln.strip()]
    input_eq_vus = {c["vu_index"] for c in checks if c["kind"] == "weight" and c.get("expected") is not None}
    prompt_vus = {c["vu_index"] for c in checks if c["kind"] == "prompt_token"}
    derived_only_vus = {c["vu_index"] for c in checks if c["kind"] == "weight" and c.get("expected") is None}
    got_coord: Counter = Counter()
    got_input: Counter = Counter()
    coord_ok = True
    extra = 0
    roots_ok = True
    for name, record in proven.items():
        sv = record.get("selected_vus") or {}
        prov_units = record["provenance"]["units"]
        if len(prov_units) != len(sv.get("coordinates", [])):
            coord_ok = False
        for vu, pu in zip(sv.get("coordinates", []), prov_units, strict=False):
            u = by_vu.get(vu)
            if u is None or not str(u["class"]).startswith("provable") or not unit_matches(u, pu):
                coord_ok = False
                log.note(f"statement {name}: VU {vu} located {json.dumps(u and u.get('locate'))[:120]} != obligation {pu.get('canonical_id')} j={pu.get('j')}")
            got_coord[vu] += 1
        for vu in sv.get("inputs", []):
            u = by_vu.get(vu)
            if u is None or not str(u["class"]).startswith("input"):
                coord_ok = False
            got_input[vu] += 1
        n_ob = len(record["object"]["obligations"])
        n_eq = sum(1 for v in sv.get("inputs", []) if v in input_eq_vus) + 2 * sum(1 for v in sv.get("inputs", []) if v in prompt_vus)
        if n_ob != len(prov_units) + n_eq:
            extra += n_ob - len(prov_units) - n_eq
        for r in record["object"]["roots"]:
            if allowed_roots.get(r["id"]) != r["root"]:
                roots_ok = False
                log.note(f"statement {name} references root {r['id']} = {r['root'][:16]} not among the committed trees of J / the registered weights root")
    missing_c = want_coord - set(got_coord)
    missing_i = want_input - set(got_input)
    dup = [v for v, n in list(got_coord.items()) + list(got_input.items()) if n > 1]
    log.check("V5.coordinates", coord_ok and not missing_c and not dup, "every selected provable VU is exactly one GemmCoordinate obligation at its located (request, t, Gemm, j)", want=len(want_coord), got=len(got_coord), missing=len(missing_c))
    log.check("V5.inputs", not missing_i and not dup and (input_eq_vus | prompt_vus | derived_only_vus) == want_input, "every selected input unit is listed by exactly one proven statement; those with a pinned checkpoint word are Input equations of that statement", want=len(want_input), got=len(got_input), missing=len(missing_i), input_equations=len(input_eq_vus) + 2 * len(prompt_vus))
    log.check("V5.inputs_derived_only", all(str(by_vu[v]["locate"].get("field", "")).endswith("cos_sin") for v in derived_only_vus), "input units of a derived table (rope cos/sin cache: registered with the weight commitment, not a checkpoint tensor) are checked by the verifier's opening + registered leaf table only (V6), not by an Input equation", count=len(derived_only_vus))
    log.check("V5.no_extra", extra == 0, "no obligation beyond the selected units' (coordinates + input equations)", extra=extra)
    log.check("V5.roots", roots_ok, "every referenced root is the registered weights root or a committed tree (RU interior / request boundary) of a selected RU", allowed=len(allowed_roots))
    # V6 input checks
    bad = [c for c in checks if not c["ok"]]
    log.check("V6.inputs", not bad and {c["vu_index"] for c in checks} == want_input, "every selected input unit's committed word == prescribed value (pinned checkpoint / served prompt); leaf sha == registered leaf table", checks=len(checks), failed=len(bad))
    if guest is not None:
        log.check("V4.elf_archived", sha256_file(POC / "results" / "GUEST-2026-09-09" / "build-1" / "veritor-zk-guest.elf") == info["elf_sha256"], "archived ELF sha256 == host's embedded guest") if (POC / "results" / "GUEST-2026-09-09" / "build-1" / "veritor-zk-guest.elf").exists() else log.note("archived ELF not on this host; sha verified at setup (rec_pod_setup.sh)")

    # V7 declared output domain: every served output token of every request is < 2^16 (16 slots per request; W_out = N x 16 x 16 bits)
    toks = []
    for ln in (rec / "adapter" / "production" / "production.jsonl").read_text().splitlines():
        if ln.strip():
            d = json.loads(ln)
            toks.append((d["request_id"], list(d["output_token_ids"])))
    log.check("V7.output_domain", len(toks) == N and all(len(t) == 16 and all(0 <= x < 2**16 for x in t) for _, t in toks) and a.out_bits_per_ru == 16 * 16,
              "every request has 16 output slots and every output token id is in the declared 16-bit domain (check_output_domains); W_out = N x 16 x 16 bits", requests=len(toks), max_token=max((max(t) for _, t in toks), default=None), W_out=N * a.out_bits_per_ru)

    # --- the Bound ---
    from veritor.analysis.bound import bound_certified
    from veritor.core.policy import VerificationPolicy

    law = inst_doc["sampler"]["law"]
    q, s, delta = Fraction(str(law["q"])), Fraction(str(law["s"])), Fraction(str(inst_doc["security"]["delta"]))
    log.check("V1.law", chron["law"]["q"] == law["q"] and chron["law"]["s"] == law["s"] and law["kind"] == "bernoulli", "chronology law == instance law (declared before any commitment)", q=law["q"], s=law["s"], delta=inst_doc["security"]["delta"])
    B_ex = len(exempt)
    table, table_doc = kind_table(pop_doc, N, a.out_bits_per_ru)
    (vdir / "kind_table.json").write_text(json.dumps(table_doc, sort_keys=True) + "\n")
    t = time.perf_counter()
    r = bound_certified(table, VerificationPolicy(q=q, s=s), delta, allowance=B_ex, resolution=a.resolution)
    bsec = round(time.perf_counter() - t, 2)
    st = r.search_status
    r0 = bound_certified(table, VerificationPolicy(q=q, s=s), delta, allowance=0, resolution=a.resolution)  # DERIVED, for the report: the same law without exemptions
    certified_source = st.source in ("appendix-a", "cap")
    all_ok = not log.fails
    bound_doc = {"B_ex_0_bits": r0.bits, "B_ex_0_capped": r0.capped, "B_ex_0_precap_bits": r0.search_status.precap_bits if r0.search_status.precap_bits is not None else r0.precap_bits, "bits": r.bits, "capped": r.capped, "out_bits": r.out_bits, "precap_bits": r.precap_bits, "certified_bits": st.certified_bits, "certified_precap_bits": st.precap_bits,
                 "source": st.source, "source_is_certified": certified_source, "case": st.case, "threshold": st.threshold, "multiplier": st.multiplier, "fallback_won": st.fallback_won,
                 "legacy_fold_bits": st.legacy_fold_bits, "candidates_certified": st.candidates_certified, "certified_status": str(getattr(r.certified, "status", "")), "seconds": bsec}
    log.check("B.certified", certified_source, "bound_certified source in {appendix-a, cap}", **{k: bound_doc[k] for k in ("bits", "capped", "out_bits", "certified_bits", "precap_bits", "source")})
    status = "BOUND" if (all_ok and certified_source and not log.fails) else "NOT BOUND"
    if status == "BOUND" and r.capped:
        status_line = "(BOUND, capped)"
    elif status == "BOUND":
        status_line = "BOUND"
    else:
        status_line = "NOT BOUND: " + "; ".join(log.fails)[:400]
    precap = st.precap_bits if st.precap_bits is not None else r.precap_bits
    precap_txt = f"pre-cap value {precap:g} bits" if precap is not None else f"no candidate below the cap with B_ex={B_ex}; with B_ex=0 the pre-cap value is {(r0.search_status.precap_bits if r0.search_status.precap_bits is not None else r0.precap_bits):g} bits"
    cert_line = (f"{status_line}: admissible outputs <= 2^{r.bits:g} bits [{'cap = declared output domain W_out = ' + str(r.out_bits) + ' bits (' + str(N) + ' requests x 16 slots x 16 declared bits; wire 32 bits/slot)' if r.capped else 'certified'}; "
                 f"{precap_txt} DERIVED] — program {pop_doc['program_digest'][:12]}…, partition P1/cu=whole {P1_DIGEST[:12]}… (validated: COMPACT-P1, "
                 f"scope cited in the instance), population {pop_sha[:12]}…, workload SMALL B0 (SmolLM2-135M, LP=32, STEPS=15), batch {N} requests, law Bernoulli q={q} s={s} δ={delta}, "
                 f"RU draw {'conditioned on ≥ 1 selected (demonstration)' if conditioned else 'unconditioned (no redraw needed)'}, J={J}, |Q|={selected['count']} "
                 f"({len(want_coord)} proven GemmCoordinate, {len(want_input)} input units prescribed; per-RU-copy formulation (SMALL, tonight); enforced by statement validation, "
                 f"B_ex={B_ex} exempt (charged)), auth_mode: per-word, guest: approved 9fd00e60, proofs {len(proven)}/{len(chunk_dirs)} compressed")
    cert = {"schema": "veritor/rec-p1-certificate/v1", "utc": utc(), "status": status, "status_line": status_line, "certificate_line": cert_line,
            "record": {"instance_digest": idig, "program_digest": pop_doc["program_digest"], "partition_digest": P1_DIGEST, "population_sha256": pop_sha, "plan": "P1/cu=whole",
                       "workload": "SMALL B0 SmolLM2-135M LP=32 STEPS=15", "batch": N, "auth_mode": "per-word", "guest": {"elf_sha256": info["elf_sha256"], "vk_hash": info["vk_hash"], "approval": (guest or {}).get("approval")}},
            "law": {"kind": "bernoulli", "q": str(q), "s": str(s), "delta": str(delta), "ru_draw": "conditioned on >= 1 selected (demonstration)" if conditioned else "unconditioned",
                    "redraws": rq.get("redraws", 0), "bound_law": "unconditioned (the Bound is for the stated law; the conditioning only affected which RUs were replayed)"},
            "selection": {"J": J, "selected_requests": sorted(sel_rids), "Q": selected["count"], "candidates": selected.get("candidates"), "kinds": selected["kinds"],
                          "provable": len(want_coord), "input_units": len(want_input), "exempt_B_ex": B_ex, "exempt_by_kind": dict(Counter(u["kind"] for u in exempt))},
            "input_units": "prescribed; per-RU-copy formulation (SMALL, tonight); enforced by statement validation",
            "proofs": {"chunks": len(chunk_dirs), "proven_verified": len(proven), "total_cycles": proofs.get("total_cycles"), "prove_wall_seconds": proofs.get("prove_wall_seconds")},
            "bound": {**bound_doc, "allowance_B_ex": B_ex, "label": "DERIVED (bound_certified over the kind table of the validated partition x N_REQ; cap = declared output domain)",
                      "kind_table": {"sha256": sha256_file(vdir / "kind_table.json"), "rows": len(table_doc["rows"]), "construction": kind_table.__doc__.strip()}},
            "checks": {"total": log.n, "failed": log.fails, "log": str(vdir / "verifier.log")},
            "validation_scope": inst_doc.get("discrepancies") or (inst_doc.get("partition") or {}).get("validation"),
            "provenance": {"label": "MEASURED (checks) / DERIVED (bound)", "hardware": "record-gpu CPU (same machine as the prover: NOT an independent verifier host)", "script": "scripts/rec_p1_verify.py",
                           "seconds": round(time.perf_counter() - T0, 1)}}
    (vdir / "certificate.json").write_text(json.dumps(cert, indent=1, sort_keys=True, default=str) + "\n")
    log.note(f"BOUND: {cert_line}")
    log.write()
    with (rec / "costlog.jsonl").open("a") as f:
        f.write(json.dumps({"stage": "verifier checks V1-V6 + certified bound", "class": "verifier", "seconds": round(time.perf_counter() - T0, 1), "hardware": "record-gpu CPU",
                            "batch": f"{len(chunk_dirs)} proofs, {selected['count']} selected VUs", "label": "MEASURED"}) + "\n")
    print(f"[rec_p1_verify] {log.n} checks, {len(log.fails)} failed; {status_line}")
    print("[rec_p1_verify] CERTIFICATE:", cert_line)
    return 0 if status == "BOUND" else 1


if __name__ == "__main__":
    sys.exit(main())
