"""``out/eod/e8/PACKAGE.md`` generator: one row per archive under ``out/eod/e8/{runs,live}`` -- interior rule (pre-D1 ``ru/`` spelling | E2 v1 |
E2 v2), driver (runner-interim | LIVE shipped driver | SIMULATION), verdict + refusing step, U, proof objects, timings (prove / verify), the
independent re-verification command, sha256 of the archive tarball (written to ``out/eod/e8/tarballs/<id>.tar.gz``; nothing inside an
archive is ever edited).  Generic protocol demonstration (DemoCore_v1); NOT vLLM inference.

    python -m e2e.demo_core.package_index --root <repo root> [--no-tar]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

V1_DIGEST = "2c56e6a0c9a41a65f0caf904d4f9b588813c9653b1144796a3af4fe857c6c465"


# the SHIPPED reverify.py's verdict per archive as RUN by E9 / E8 (by name; the shipped tool has no E8 hook, so this is a hand-kept ledger with the source)
SHIPPED_REVERIFY = {
    "CORE-live-exhaustive-02": "REFUSED by the shipped reverify.py (E9, D24/D25 registered path): (a) reverify.py:608 composed.json#statement_digest was E7's v2 digest while statement.bin is the proved v3 "
                               "statement (E7 fix 9ec1a95f); (b) E3 GenericLocator located 0/128 selected (E3 fix dbc463ad+ccab5da1). Record KEPT AS IS (accepted by the live driver; refused by the re-verifier on a since-fixed record defect). "
                               "E8's own run on the pod before D24: refused at reverify.py:295 (--expressibility) / :331 (maps.json) -- superseded by E9's registered path.",
    "CORE-live-sampled-B1-03a": "REFUSED at V3.weights_root registered=ABSENT (+ V5.* cascade): E8 DEFECT in derive_instance.py -- the derived instance dir lacked weights_authorized.json (fixed; rerun = -03). "
                                "The live driver ran it with weights_registry=False and accepted (finding for E9: a phase-0 weights tree the instance does not pin is worth a refusal).",
    "CORE-live-exempt-X1-03a": "REFUSED at V3.weights_root registered=ABSENT (+ V5.* cascade): same E8 defect as B1-03a (fixed; rerun = -03).",
    "CORE-live-sampled-B1-03": "REFUSED at V5.expected: CoverageError NO_REGION gate 32 (RU0's output, read by VU 16 of RU1; committed phase-0 in boundary/core-0-ru0) -- E3 expected.py:703-705 keeps only RETAINED requests' trees, "
                               "dropping the phase-0 boundary tree of an unselected RU (cross-RU dataflow). SIMULATION with boundary/* of all requests admitted: 43 checks, 0 failed; VERIFIED (out/eod/e8/live/SIM-e3-boundary-trees/).",
    "CORE-live-exempt-X1-03": "REFUSED at V5.expected: same E3 line as B1-03 (F={2,5}: exempted RUs' boundary trees dropped). SIMULATION: 43 checks, 0 failed; VERIFIED.",
    "CORE-live-neg-forged-03": "REFUSED at V4.proof.chunk00 + V5.coverage -- THE INTENDED OUTCOME (forged interior word gate 74; both verifiers agree).",
}


def shipped_reverify_verdict(root: Path, rid: str) -> str | None:
    """The SHIPPED reverify.py's own verdict line for an archive, read from the pulled stdout (out/eod/e8/live/reverify-shipped/<id>.stdout)."""
    f = root / "out" / "eod" / "e8" / "live" / "reverify-shipped" / f"{rid}.stdout"
    if not f.exists():
        return None
    m = re.search(r"\[reverify_v3\] (\d+ checks, \d+ failed; (?:VERIFIED|REFUSED)[^\n]{0,160})", f.read_text())
    return m.group(1) if m else "ran (no verdict line: see .stdout)"


def _j(p: Path) -> dict[str, Any]:
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return {}


def rule_of(rec: Path) -> str:
    c2 = _j(rec / "adapter" / "selective" / "commitments.json") or _j(rec / "adapter" / "selective" / "commitments_phase0.json")
    il = c2.get("interior_layout") or {}
    if il.get("rule"):
        return il["rule"]
    if il.get("digest") == V1_DIGEST:
        return "interior-layout/chunks-4096/v1"
    if il.get("tree", "").startswith("ru/"):
        return "pre-D1 ru/<req>/0 spelling (E8 interim)"
    return "n/a (no interior commitment reached)"


def row_of(rec: Path, tarballs: Path | None) -> dict[str, Any]:
    proto = _j(rec / "protocol.json")
    s5 = _j(rec / "protocol.step5.json")
    ex = _j(rec / "verifier_exit.json")
    rc = _j(rec / "run_config.json")
    sim = rec.name.startswith("SIM")
    live = (rec.parent.name == "live") and not sim
    driver = "SIMULATION (labelled; not a record)" if sim else ("LIVE shipped verifier_v3/driver.py" + (" + demo_core.step5_live" if s5 else "") if live else "runner-interim (e2e.demo_core.runner; shipped step functions)")
    if live or sim:
        if ex and ex.get("reason_code") not in (None, "OK", "DONE"):
            verdict, step, reason = "REFUSED", ex.get("step"), (ex.get("problems") or [ex.get("message")])[:1]
        elif s5:
            verdict, step, reason = s5.get("verdict_step5"), ("step5" if s5.get("checks_failed") else "done"), s5.get("checks_failed") or []
        elif proto:
            verdict, step, reason = "DONE (steps 1-4; no step 5 run)", "r_s", []
        else:
            log = rec / "driver.log"
            text = log.read_text() if log.exists() else ""
            if "] DONE:" in text:
                verdict, step, reason = "DONE (steps 1-4 through r_s)", "r_s", []
            else:
                tail = [ln for ln in text.splitlines() if "Error" in ln]
                verdict, step, reason = "CRASH (uncaught exception: no verifier_exit.json)", "stage1" if "ChronologyError" in text else "?", tail[-1:] if tail else []
        U = (s5.get("U") or {})
        proofs = (s5.get("step5") or {}).get("proof_objects")
        Q = (s5.get("step5") or {}).get("Q", proto.get("selected_vus"))
        prove_s = [x for x in (s5.get("prove_seconds") or []) if x is not None]
        law = None
        inst = _j(rec / "protocol_instance.json") or _j(rec / "instance" / "protocol_instance.json")
        if inst:
            law = {"q": inst["sampler"]["law"]["q"], "s": inst["sampler"]["law"]["s"], "B_ex": inst["epoch"]["B_ex"], "instance_id": inst.get("instance_id")}
    else:
        verdict, step, reason = proto.get("verdict"), ("step5" if proto.get("checks_failed") else "done"), proto.get("checks_failed") or []
        U = proto.get("U") or {}
        proofs = (proto.get("step5") or {}).get("proof_objects")
        Q = (proto.get("step5") or {}).get("Q")
        prove_s = [c.get("prove_s") for c in (proto.get("step5") or {}).get("per_chunk", []) if c.get("prove_s") is not None]
        law = {"q": proto.get("law", {}).get("q"), "s": proto.get("law", {}).get("s"), "B_ex": (proto.get("law") or {}).get("B_ex", rc.get("B_ex"))} if proto else None
    hv = [json.loads(l) for l in (rec / "verify" / "host_verify.jsonl").read_text().splitlines()] if (rec / "verify" / "host_verify.jsonl").exists() else []
    verify_s = [round(float(r["verify_seconds"]), 3) for r in hv if r.get("verify_seconds") is not None]
    pj = _j(rec / "proofs" / "proofs.json")
    wall_prove = pj.get("prove_wall_seconds")
    sha = size = None
    if tarballs is not None:
        tarballs.mkdir(parents=True, exist_ok=True)
        tb = tarballs / f"{rec.name}.tar.gz"
        subprocess.run(["tar", "-czf", str(tb), "-C", str(rec.parent), rec.name], check=True)
        sha, size = hashlib.sha256(tb.read_bytes()).hexdigest(), tb.stat().st_size
    accepted_by = []
    if live and proto and verdict == "ACCEPTED":
        accepted_by.append("driver.py (LIVE, shipped) steps 1-4 + step5_live V4/V5")
    elif live and proto and not s5:
        accepted_by.append("driver.py (LIVE, shipped) steps 1-4 DONE; step 5 pending")
    elif not live and not sim and verdict == "ACCEPTED":
        accepted_by.append("runner-interim (shipped step functions; NOT the shipped driver)")
    rv = _j(rec.parent / "reverify" / rec.name / "reverify.json")
    if rv:
        accepted_by.append(f"reverify_registered.py (PROPOSED): {rv.get('verdict')}")
    if rec.name == "CORE-live-exhaustive-03":
        accepted_by.append("shipped reverify.py on the LAPTOP (E9, INDEPENDENT machine, tree e0ad1912, host veritor-zk-host-laptop-9fd00e60 sha256 c1f96f17...): 46 checks, 0 failed, VERIFIED, 20.3 s, "
                           "record_v3.json equal to the pod run except path fields (/private/tmp/eod-e9/out/eod/e9/reverify-core-live-3/VERDICT.md)")
    sv = shipped_reverify_verdict(rec.parents[4] if rec.parent.name == 'live' else rec.parents[3], rec.name)
    if sv:
        accepted_by.append(f"reverify.py (shipped, MEASURED on the pod): `{sv}`" + (" -- " + SHIPPED_REVERIFY[rec.name] if rec.name in SHIPPED_REVERIFY else ""))
    elif rec.name in SHIPPED_REVERIFY:
        accepted_by.append("reverify.py (shipped): " + SHIPPED_REVERIFY[rec.name])
    return {"id": rec.name, "path": str(rec), "rule": rule_of(rec), "accepted_by": accepted_by, "driver": driver, "verdict": verdict, "step": step, "reason": reason, "law": law,
            "U": {k: U.get(k) for k in ("U_bits", "W_bits", "label")} if U else None, "Q": Q, "proofs": proofs, "prove_s": [round(float(x), 3) for x in prove_s], "prove_wall_s": wall_prove,
            "verify_s": verify_s, "sha256": sha, "tar_bytes": size, "total_s": (proto.get("timing") or {}).get("total_s"), "step5_s": (s5.get("timing") or {}).get("step5_s") or (proto.get("timing") or {}).get("step5_s")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--no-tar", action="store_true")
    a = ap.parse_args(argv)
    root = Path(a.root)
    e8 = root / "out" / "eod" / "e8"
    recs = sorted([p for d in ("runs", "live") for p in (e8 / d).iterdir() if p.is_dir() and (p / "protocol.json").exists() or (p / "verifier_exit.json").exists() or (p / "driver.log").exists()], key=lambda p: (p.parent.name, p.name))
    rows = [row_of(r, None if a.no_tar else e8 / "tarballs") for r in recs]
    (e8 / "PACKAGE.json").write_text(json.dumps({"schema": "veritor/demo-core/package-index/v1", "rows": rows}, indent=1, default=str) + "\n")
    utc = subprocess.run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], capture_output=True, text=True).stdout.strip()
    out = [f"# E8 CORE demonstrator — package index ({utc})", "",
           "Generic protocol demonstration (DemoCore_v1{N=8,R=8}: 8 RUs = stage calls, 128 VUs of two approved elem kinds; NOT vLLM inference). One row per archive; nothing inside an archive is edited after its run.",
           "**Rule** = the interior-tree spelling the archive carries (`pre-D1 ru/` = E8's interim spelling before E2's layout; `…/v1` = E2's first rule, digest `2c56e6a0…`; `…/v2` = E2's current rule, digest `6e1020a8…`, D13).",
           "**Driver** = who ran the five steps: `runner-interim` = `e2e.demo_core.runner` (E8's driver mirror calling the shipped step functions; the shipped `driver.py` could not drive a registered program before D1/D17); `LIVE` = the shipped `e2e.verifier_v3.driver` (+ `e2e.demo_core.step5_live` for step 5).",
           "Every timing MEASURED (`prove s` = SP1 cuda `prove_seconds` per proof object on the 4090, excl. setup; `verify s` = the verifier's own host `verify` per object). U from the shipped calculator (MODELED/ANALYTICAL; `U=0/128` means every covered VU is proved, U = W is never hidden).", "",
           "| archive | rule | driver | verdict | step | law (q, s, B_ex) | U | Q | proofs | prove s | verify s | tarball sha256 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    verified_by = ["", "## Which verifier accepted what (by name)", "",
                   "Three verifiers exist today: the SHIPPED live `e2e.verifier_v3.driver` (steps 1-4; step 5 through `e2e.demo_core.step5_live`, which calls the shipped V4/V5 functions), the SHIPPED post-hoc `e2e.verifier_v3.reverify.py` (registered path since D24/D25, E9), and E8's PROPOSED `e2e.demo_core.reverify_registered` (D20 fallback; shipped step functions by name; not the shipped tool). An archive is only as accepted as the verifiers named on its row.", ""]
    for r in rows:
        law = r["law"] or {}
        U = r["U"] or {}
        out.append(f"| `{r['id']}` | {r['rule']} | {r['driver']} | **{r['verdict']}** | {r['step']}{(' — ' + str(r['reason'][0])[:110]) if r['reason'] else ''} | {law.get('q')}, {law.get('s')}, {law.get('B_ex')} | "
                   f"{U.get('U_bits')}/{U.get('W_bits')} | {r['Q']} | {r['proofs']} | {', '.join(str(x) for x in r['prove_s']) or '—'} | {', '.join(str(x) for x in r['verify_s']) or '—'} | {(r['sha256'] or '—')[:16]}… |")
    for r in rows:
        neg = "-neg-" in r["id"] or r["id"].endswith("-REJECTED-b0ec3a94") or r["id"] == "CORE-exempt-X2"
        verified_by.append(f"- `{r['id']}`: " + ("; ".join(r["accepted_by"]) if r["accepted_by"] else
                           (f"{'NEGATIVE / predeclared refusal test -- ' if neg else ''}no verifier accepted it ({r['verdict']} at {r['step']})" + (" -- the intended outcome" if neg else ""))))
    out += verified_by
    out += ["", "## Reading the live rows", "",
            "- `CORE-live-exhaustive-03` — **D6 coverage line:** `sampled VUs = 128; covered by a verified expected statement = 128 (identities: 0 … 127); proof objects = 2; uncovered = 0`. **Record digests:** `protocol.json` sha256 `2ee475fbec6a625dd4889fb8375c931fb92febbcd3a07fc97144a2399d5e2fb6`; `proofs/chunk00-compressed.bin` `ec2882191d49abf3…`; `proofs/chunk01-compressed.bin` `97c98fa2073d0ad2…` (E9's list agrees). **Archive completion (02:25Z, not an edit):** the four driver step 2-4 outputs `verify/{population_derived,ru_fields,seed_custody,transcript}.json` (committed 17bed9c8 after the driver run) had been deleted from the working tree by an `rm -rf <rec>/verify` between step-5 attempts (01:27Z/01:29Z) and were missing from commits 8423d490..4017d981 — the shipped `reverify.py` over that tree gives `V8.ru_fields FAIL (rus=0)` (E9's run A). Restored byte-for-byte from E8's own history (`git checkout 2cd58c01 -- …`; sha256 bd3df57f… 8f006e17… af6a4305… 191c2222… = E9's `restored-from-e8-HEAD/`). The pod-side copy (rsynced 01:03Z, never deleted there) always had them, so the pod verdict stands; E9's laptop verdict is over the completed archive. `step5_live` now fails closed when they are absent.",
            "- `CORE-live-exhaustive-03` — the merged-tree rerun (`vllm-poc` @ `923d9d88` ∪ E8; D25: E7 `statement_digest`, E3 `locate_selected`, E9 `input_check` in the driver + `V2.partition_digest` by name): live driver steps 1-4 DONE 01:01:30Z with `STEP2_INPUT_CHECK source=e2e.demo_core.adapter:input_check supplied=True`; step 5 on pod `eod-e8-core2` (NVIDIA L4; host rebuilt from the approved guest's tree: elf `9fd00e60…`, vk `0x0018b8e4…`) ACCEPTED 128/128, 2 proofs (prove 3.28 + 3.23 s GPU, host verify 30.6 + 30.7 s CPU on the L4 box); the SHIPPED `reverify.py` on the pod: `46 checks, 0 failed; VERIFIED` (`out/eod/e8/live/reverify-shipped/CORE-live-exhaustive-03/`). The first CORE record accepted by BOTH shipped verifiers.",
            "- `CORE-live-sampled-B1-03` / `CORE-live-exempt-X1-03` — merged-tree reruns with the weights pin present: live driver + step 5 ACCEPTED (Q = 47 / 45, 1 proof each); the SHIPPED `reverify.py` REFUSES both at `V5.expected` (`NO_REGION: gate 32 lies in no declared RU region of request 'core-0-ru1'`): E3's `expected.py:703-705` admits only RETAINED requests' trees, dropping the phase-0 `boundary/core-0-ru0` tree whose word (RU0's output) VU 16 of RU1 reads — the cross-RU value. SIMULATION with `boundary/*` of all requests admitted (`SIM-e3-boundary-trees/`): `43 checks, 0 failed; VERIFIED` for both. Not a record defect; a driver-vs-reverify disagreement (the live step 5 already admits every phase-0 boundary tree, `runner.py:296`).",
            "- `CORE-live-sampled-B1-03a` / `CORE-live-exempt-X1-03a` — the first `-03` sampled attempts, kept as archived outcomes: live ACCEPTED, shipped `reverify.py` REFUSED at `V3.weights_root registered=ABSENT` — an E8 defect (`derive_instance.py` did not carry `weights_authorized.json`; fixed) that the live driver did not catch (`weights_registry=False`; finding for E9).",
            "- `CORE-live-neg-forged-03` — forged interior word (gate 74 xor 1) on the merged tree: live step 5 REJECTED (`V4.proofs_binding`: SP1 `PROOF DOES NOT VERIFY` on chunk00; `V5.coverage` 64 uncovered) AND the shipped `reverify.py` REFUSED (`45 checks, 2 failed; REFUSED: V4.proof.chunk00 …; V5.coverage`). Both verifiers agree on the negative.",
            "- `CORE-live-exhaustive-02` — THE 18:00 deliverable: E6's frozen v2 instance `E6-CORE-N8R8-bc1a7268`, exhaustive law q = s = 1, B_ex = 0, input seed 1; every step's verdict line is in `driver.log` / `protocol.json`; step 5 in `step5.log` / `protocol.step5.json`; 128/128 VUs covered by 2 SP1 proof objects.",
            "- `CORE-live-exhaustive-01` / `CORE-live-neg-early-rs-01` — the same runs before E9's `steps.py:311` fix (`bd954c36`): refused at `V3.interior_layout` by the verifier, kept as evidence of the refusal (not edited).",
            "- `CORE-live-sampled-B1` — predeclared sampled audit q = s = 1/2 (fresh verifier seed): J = {0,3,4,5,7}, |Q| = 46 of 80 candidates, 1 proof object, ACCEPTED; U = W published (128/128).",
            "- `CORE-live-exempt-X1` — controlled exemption TEST (labelled in the request): B_ex = 2, prover withholds RUs {2,5} ∩ J → F = {2,5} legal, retained RUs proved, |Q| = 46 of 96, ACCEPTED.",
            "- `CORE-live-neg-forged-01` — NEGATIVE: the prover commits gate 74 (VU 50's output, interior) xor 1; steps 1-4 pass (a well-formed dishonest record), the SP1 guest's verdict is 0 for chunk00 (`PROOF DOES NOT VERIFY`; the verifier's own `verify` reports `verdict: false`) → `V4.proofs_binding FAIL`, `V5.coverage FAIL` (64 uncovered) → REJECTED at the proof step.",
            "- `CORE-live-neg-early-rs-02` — NEGATIVE: premature `REQUEST_r_s` before READY_PHASE2 → `handshake/REFUSED_r_s` = `PREMATURE_CHALLENGE` (state `r_q_released`, required `step4_complete`, nothing drawn), the honest run then completes and is ACCEPTED (the refusal is the evidence, the verdict shows a refusal is not a rejection).",
            "- `CORE-live-neg-mispin-02` — NEGATIVE: E6's instance next to the population file of `DemoCore_v1{N=8,R=4}` → `V2.population_derived disagrees` → `STEP2_REFUSED` (exit 20) before r_q. `CORE-live-neg-mispin-01` (instance `partition_digest` last byte flipped): `V2` matches, then an UNCAUGHT `ChronologyError` in `partition_sample.stage1` (exit 1, no `verifier_exit.json`) — a refusal not by name; reported to E9/E6.",
            "- `SIM-live1-e9-311` — SIMULATION (the driver with `steps.py:311` patched in memory, before the fix landed; used only to identify the line). Not a record.",
            "", "## Re-verify independently", "",
            "Shipped `reverify.py` (E9, registered path since D24/D25): `PYTHONPATH=<tree>:<tree>/vllm-poc:<tree>/src SP1_PROVER=cpu python e2e/verifier_v3/reverify.py --record <archive> --population <archive>/instance/population.compact.json --host <veritor-zk-host with verify-many, guest 9fd00e60> --out <dir outside the archive> --instance-format contract` — RUN on the pod for every `-03` archive (`out/eod/e8/live/reverify-shipped/<id>.stdout`, `<id>/reverify.log`, `record_v3.json`, `coverage.json`, `host_verify.jsonl`). The host must embed the instance's pinned guest (`epoch.approved_elf_sha256` = `9fd00e60…`) AND have `verify-many`: built from `git archive ee6f186a zk/sp1` (= approved tree `1d4d4868` + the D6 `verify-many` host commit; guest/common unchanged) → host sha256 `e790b523…`, the same bytes as the first pod's (`host-build3.log`).",
            "History (D25): before 00:36Z the shipped tool had no registered path (`reverify.py:295` `--expressibility`, `:331` `maps.json`); E9's registered path over `CORE-live-exhaustive-02` then REFUSED on E7's `statement_digest` (fixed `9ec1a95f`) and E3's `locate_selected` (fixed `dbc463ad`+`ccab5da1`); `-02` is kept as is.",
            "PROPOSED fallback (D20; E8's package, shipped step functions by name): `PYTHONPATH=. python -m e2e.demo_core.reverify_registered --record <archive> --out <dir outside> --pod-ssh <E10 wrapper> --pod-dir <remote dir>` — prints the `reverify.py` checks it does NOT execute.",
            "Runner-interim archives (v1 / pre-D1 spelling): `verify/verifier.log` is the check-by-check log; the same `reverify_registered` command re-checks V4/V5 (proof objects, coverage) over them; steps 2-4 are re-derivable from `chronology/` with `e2e.partition_sample.verify`.", "",
            "## Tarballs", "", "`out/eod/e8/tarballs/<id>.tar.gz` (all < 5 MB; nothing left on the pod except scratch copies). Full sha256 in `PACKAGE.json`.", ""]
    for r in rows:
        if r["sha256"]:
            out.append(f"- `{r['id']}.tar.gz` {r['tar_bytes']} B sha256 `{r['sha256']}`")
    (e8 / "PACKAGE.md").write_text("\n".join(out) + "\n")
    print(f"[package_index] {len(rows)} archives -> {e8 / 'PACKAGE.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
