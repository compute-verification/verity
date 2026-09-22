#!/usr/bin/env python3
"""[PAD-INT, stage 2] Negatives for the padded path over a real early-terminating record — each one a claim that MUST be
rejected, with the layer that rejects it named:

  N1  live relation claimed on an inactive slot
      (a) the LIVE output pattern `(tok_out, live_out) = (y, 1)` asserted at a sampling event whose committed liveness is
          `live_in = 0`: the relation is false → rejected by the transparent checker AND by the SP1 host (`execute`
          verdict false; nothing to prove);
      (b) the slot relabelled live (`live_in = 1`, a token where production sampled nothing): the relation itself holds,
          so the rejection is the LINKAGE check — the claimed liveness contradicts the flag chain derived from the
          committed tokens (`n_present`, `finish_reason == stop`); `padint_linkage_check` rejects.
  N2  missing required inactive proof: the attached proof of the batch that carries the inactive units is removed from
      a copy of the record → `e2e.certificate --require-coverage` and `padint_linkage_check` both reject.
  N3  altered padded output
      (a) an inactive slot's designated output changed from the canonical 0 to a token: the guard relation
          `live_in = 0 ⟹ tok_out = 0` is false → transparent checker + SP1 host reject;
      (b) the record's served-token transcript extended past the EOS (one more token on the early-terminated request):
          the committed liveness no longer matches the statements' census → `padint_linkage_check` rejects, and the
          record-level pin (`served_tokens.json` sha / correspondence) is re-checked by `e2e.certificate`.

Usage: padint_negatives.py --record REC --bundle BROOT --batches PADBATCHES --host COVHOST --out OUT
Every negative writes its own directory under OUT and `negatives.json` summarises `expected_reject` vs `rejected`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "zk" / "bench"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vllm-poc"))

from pad_relations import PadBatcher, PadTranslated, make_statement  # noqa: E402
from pb_common import write_batch  # noqa: E402
from veritor.protocol.proofs.statement import Witness  # noqa: E402
from veritor.protocol.proofs.transparent import TransparentBackend  # noqa: E402

sys.path.insert(0, str(HERE.parent))
from padint_pad_statements import flag_chain  # noqa: E402


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def host_json(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {}


class UncheckedBatcher(PadBatcher):
    """A PadBatcher whose `batch()` does NOT run the transparent checker — so a forged claim reaches the SP1 host."""

    def batch_unchecked(self):
        chosen = self.prepared
        statement = make_statement(self.t.gate_set.id, bytes.fromhex(str(self.t.gate_set.digest)), 32,
                                   [self.t.program], [o for o, _ in chosen])
        witness = Witness(tuple(w for _, w in chosen))
        checker = TransparentBackend(self.t.gate_set, witness_format=2)
        try:
            proof = checker.prove(statement, witness)
            transparent_ok = bool(checker.verify(statement, proof))
        except Exception as e:  # noqa: BLE001 — a rejected witness may raise instead of returning False
            transparent_ok = False
            self.transparent_error = f"{type(e).__name__}: {str(e)[:200]}"
        return statement, witness, transparent_ok


def host_execute(host: str, directory: Path) -> dict[str, Any]:
    env = dict(os.environ)
    env["SP1_PROVER"] = "cpu"
    r = subprocess.run([host, "execute", "--statement", str(directory / "statement.bin"), "--witness", str(directory / "witness-fmt2.bin")],
                       capture_output=True, text=True, env=env, timeout=600)
    doc = host_json(r.stdout) if r.stdout else {}
    return {"rc": r.returncode, "verdict": doc.get("verdict"), "total_cycles": doc.get("total_cycles"),
            "stderr_tail": r.stderr[-300:] if r.returncode else ""}


def forged_guard_batch(out: Path, EOS: int, live_in: int, y: int, tok_out: int, live_out: int, tag: str) -> tuple[Path, bool]:
    from verity.ir.defs import bind
    from verity_vllm.registry import serve3

    translated = PadTranslated(bind(serve3.PadGuard, EOS=EOS), f"PadGuard<EOS={EOS}>", [1, 32])
    b = UncheckedBatcher(translated, f"padint-negative-{tag}", [32, 1])
    b.add(tag, [live_in, y], [tok_out, live_out])
    statement, witness, transparent_ok = b.batch_unchecked()
    entry = write_batch(out, statement, witness, {"relation": "pad-guard/negative", "batch": 1, "family": "pad-serve",
                                                    "negative": tag, "claim": {"live_in": live_in, "y": y, "tok_out": tok_out, "live_out": live_out},
                                                    "transparent_check_ok": transparent_ok})
    return Path(entry["dir"]), transparent_ok


def _padded(statements: dict[str, Any]) -> dict[str, Any]:
    """The padded program block of `padint_statements.json` (`program.padded`: function, digest, LP, D_MAX, EOS_ID)."""
    prog = statements["program"]
    return prog.get("padded", prog)


def padint_linkage_check(record: Path, statements: dict[str, Any], census_override: dict | None = None) -> list[str]:
    """The PAD-INT linkage check a padded record must pass: (1) every pad batch attached to the record names the padded
    program digest of the statements census AND the certified digest the record's manifest names; (2) an attached,
    CPU-verified batch carries inactive units (> 0) — an all-live record is not inactive evidence; (3) the census's
    liveness equals the flag chain of the record's served tokens; (4) attached proof files re-hash."""
    problems: list[str] = []
    manifest = json.loads((record / "manifest.json").read_text())
    cert_digest = ((manifest.get("program") or {}).get("digest")) or (manifest.get("inputs") or {}).get("program_digest")
    cov_file = record / "proofs" / "proof-cov" / "cov_summary.json"
    if not cov_file.exists():
        return ["no proofs/proof-cov/cov_summary.json — the pad statements are not attached to the record"]
    cs = json.loads(cov_file.read_text())
    prog = _padded(statements)
    inactive_attached = 0
    pad_batches = [b for b in cs.get("batches") or [] if "pad-" in b.get("name", "")]
    if not pad_batches:
        problems.append("no pad-serve batch attached")
    for b in pad_batches:
        # the attached batch's census.json (write_batch: census + the builder's meta) carries the program digests
        meta: dict[str, Any] = dict(b)
        if b.get("attached_dir") and (record / b["attached_dir"] / "census.json").exists():
            meta.update(json.loads((record / b["attached_dir"] / "census.json").read_text()))
        if b.get("attached"):
            if meta.get("program_digest") != prog["digest"]:
                problems.append(f"{b['name']}: attached census program_digest {str(meta.get('program_digest'))[:12]} != padded program {prog['digest'][:12]}")
            if cert_digest and meta.get("certified_program_digest") != cert_digest:
                problems.append(f"{b['name']}: attached census certified_program_digest != the record manifest's program digest")
        ok_modes = 0
        for mode, docm in (b.get("modes") or {}).items():
            att = docm.get("attached")
            vc = docm.get("verified_cpu")
            if att:
                f = record / att["file"]
                if not f.exists() or sha256_file(f) != att["sha256"]:
                    problems.append(f"{b['name']} {mode}: attached proof missing or sha mismatch")
                    continue
            if att and vc and vc.get("ok"):
                ok_modes += 1
        if ok_modes and (meta.get("inactive_units") or 0) > 0:
            inactive_attached += int(meta["inactive_units"])
    if inactive_attached <= 0:
        problems.append("no attached+verified pad batch carries inactive units (all-live evidence is not inactive evidence)")
    # liveness vs served tokens
    served = {r["request_id"]: r for r in json.loads((record / "served_tokens.json").read_text())["requests"]}
    census = census_override if census_override is not None else statements["census"]
    slots = int(prog["D_MAX"]) + 1
    by_req: dict[str, list] = {}
    for c in census["guard"]:
        by_req.setdefault(c["request_id"], []).append(c)
    for rid, items in by_req.items():
        s = served.get(rid)
        if s is None:
            problems.append(f"{rid[:8]}: census request not in served_tokens.json")
            continue
        live_in, _ = flag_chain([int(x) for x in s["served_token_ids"]], slots)
        claimed = [None] * slots
        for c in items:
            claimed[c["event"]] = c["live_in"]
        if claimed != live_in:
            bad = [g for g in range(slots) if claimed[g] != live_in[g]]
            problems.append(f"{rid[:8]}: claimed liveness != flag chain of the served tokens at events {bad} (n_present={len(s['served_token_ids'])}, finish={s.get('finish_reason')})")
        n_served, n_live = len(s["served_token_ids"]), sum(1 for c in items if c["live_in"])
        if n_served != n_live:
            # the chain absorbs at EOS, so a token served AFTER the EOS is a token at an inactive slot whose padded
            # output must be 0 — the transcript and the census disagree about what the request produced
            problems.append(f"{rid[:8]}: served transcript has {n_served} tokens but the census has {n_live} live events "
                            f"(a token at an inactive slot — padded output must be 0 — or a live slot without a token)")
    return problems


def run_certificate(record: Path) -> dict[str, Any]:
    cmd = [sys.executable, "-m", "e2e.certificate", "--record", str(record), "--require-coverage", "--out", str(record / "certificate-negative.json")]
    if not (record / "protocol_instance.json").exists():
        cmd.append("--historical-v2")  # [PROTO-A v3] pre-instance record: each negative must be refused on its own check, not on the absent instance
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT / "vllm-poc"), timeout=900)
    doc = {}
    p = record / "certificate-negative.json"
    if p.exists():
        doc = json.loads(p.read_text())
    return {"rc": r.returncode, "status": doc.get("status") or doc.get("verdict"), "problems": (doc.get("problems") or [])[:6], "tail": (r.stdout + r.stderr)[-400:]}


def copy_record(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--batches", required=True, help="padint_pad_statements output dir (padint_statements.json)")
    ap.add_argument("--host", required=True, help="the cov host binary (pad-serve guest)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    record, out = Path(args.record), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    statements = json.loads((Path(args.batches) / "padint_statements.json").read_text())
    EOS = int(_padded(statements).get("EOS_ID", 0))
    census = statements["census"]
    inactive = [c for c in census["guard"] if c["kind"] == "inactive"]
    live = [c for c in census["guard"] if c["kind"] == "live"]
    if not inactive:
        raise SystemExit("no inactive guard instance in the census — negatives over the inactive path need an early-terminating record")
    target = inactive[0]
    results: list[dict[str, Any]] = []

    def add(name, layer, rejected, detail, expected=True):
        results.append({"negative": name, "rejecting_layer": layer, "expected_reject": expected, "rejected": bool(rejected), "ok": bool(rejected) == expected, "detail": detail})
        print(f"  {name}: {'REJECTED' if rejected else 'ACCEPTED'} by {layer} -> {'ok' if bool(rejected) == expected else 'FAIL'}", flush=True)

    # ---- positive control: the honest inactive instance is accepted --------------------------------------------
    d0, t_ok = forged_guard_batch(out / "control-inactive", EOS, 0, 0, 0, 0, "control-inactive")
    ex0 = host_execute(args.host, d0)
    add("control: honest inactive instance", "transparent checker + SP1 host execute", not (t_ok and ex0["verdict"] is True), {"transparent_ok": t_ok, "execute": ex0}, expected=False)

    # ---- N1a: live output pattern on the inactive slot ---------------------------------------------------------
    y_live = live[0]["y"] if live else 42
    d1, t1 = forged_guard_batch(out / "n1a-live-pattern-on-inactive", EOS, 0, 0, 0, 1, "n1a")
    ex1 = host_execute(args.host, d1)
    add("N1a live relation (live_out=1) claimed on an inactive slot", "transparent checker + SP1 host execute",
        (not t1) and ex1["verdict"] is not True,
        {"slot": {"request_id": target["request_id"][:8], "event": target["event"], "t": target["t"], "address": target["address"]},
         "claim": {"live_in": 0, "y": 0, "tok_out": 0, "live_out": 1}, "transparent_ok": t1, "execute": ex1})

    # ---- N1b: the slot relabelled live -------------------------------------------------------------------------
    forged = json.loads(json.dumps(census))
    for c in forged["guard"]:
        if c["request_id"] == target["request_id"] and c["event"] == target["event"]:
            c["live_in"], c["y"], c["tok_out"], c["live_out"], c["kind"] = 1, int(y_live), int(y_live), 1, "live"
    p1b = padint_linkage_check(record, statements, census_override=forged)
    add("N1b inactive slot relabelled live (relation holds; liveness contradicts the committed tokens)", "padint_linkage_check (flag chain of served tokens)",
        any("claimed liveness" in p for p in p1b), {"problems": p1b})

    # ---- N2: the inactive proof removed from a record copy -----------------------------------------------------
    rec2 = out / "n2-record-missing-inactive-proof"
    copy_record(record, rec2)
    cs = json.loads((rec2 / "proofs" / "proof-cov" / "cov_summary.json").read_text())
    removed = []
    for b in cs.get("batches") or []:
        meta = b.get("meta") or b
        if "pad-guard" in b.get("name", "") and (meta.get("inactive_units") or 0) > 0:
            for mode, docm in (b.get("modes") or {}).items():
                att = docm.get("attached")
                if att and (rec2 / att["file"]).exists():
                    (rec2 / att["file"]).unlink()
                    removed.append(att["file"])
    p2 = padint_linkage_check(rec2, statements)
    c2 = run_certificate(rec2)
    add("N2 missing required inactive proof", "padint_linkage_check + e2e.certificate --require-coverage",
        bool(removed) and any("missing" in p for p in p2) and (c2["rc"] != 0 or c2["problems"]),
        {"removed": removed, "linkage_problems": p2, "certificate": c2})

    # ---- N3a: an inactive slot's designated output altered to a token -----------------------------------------
    d3, t3 = forged_guard_batch(out / "n3a-altered-padded-output", EOS, 0, 0, 17, 0, "n3a")
    ex3 = host_execute(args.host, d3)
    add("N3a altered padded output (inactive slot tok_out 0 -> 17)", "transparent checker + SP1 host execute",
        (not t3) and ex3["verdict"] is not True,
        {"slot": {"request_id": target["request_id"][:8], "event": target["event"], "t": target["t"]}, "claim": {"live_in": 0, "y": 0, "tok_out": 17, "live_out": 0},
         "transparent_ok": t3, "execute": ex3})

    # ---- N3b: the served transcript extended past EOS in a record copy ----------------------------------------
    rec3 = out / "n3b-record-altered-served-tokens"
    copy_record(record, rec3)
    st = json.loads((rec3 / "served_tokens.json").read_text())
    for r in st["requests"]:
        if r["request_id"] == target["request_id"]:
            r["served_token_ids"] = list(r["served_token_ids"]) + [int(y_live)]
    (rec3 / "served_tokens.json").write_text(json.dumps(st, indent=1) + "\n")
    p3 = padint_linkage_check(rec3, statements)
    c3 = run_certificate(rec3)
    add("N3b served transcript extended past EOS (one more token on the early-terminated request)", "padint_linkage_check + e2e.certificate",
        any(("served transcript" in p) or ("claimed liveness" in p) for p in p3) and (c3["rc"] != 0 or c3["problems"]),
        {"linkage_problems": p3, "certificate": c3})

    # ---- positive control on the real record ------------------------------------------------------------------
    p_pos = padint_linkage_check(record, statements)
    add("control: the real record passes the linkage check", "padint_linkage_check", bool(p_pos), {"problems": p_pos}, expected=False)

    for d in (rec2, rec3):
        shutil.rmtree(d, ignore_errors=True)  # the copies are large; the verdicts are recorded above
    doc = {"schema": "veritor/padint-negatives/v1", "record": record.name, "program": statements["program"], "evidence_label": statements.get("evidence_label"),
           "label": "HAND-BUILT FALLBACK — not the converged path" if "HAND-BUILT" in str(statements.get("evidence_label")) else "derived padded program",
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "target_inactive_slot": target, "results": results,
           "all_ok": all(r["ok"] for r in results)}
    (out / "negatives.json").write_text(json.dumps(doc, indent=1) + "\n")
    print("NEGATIVES", "ALL-OK" if doc["all_ok"] else "FAILURES", f"({sum(r['ok'] for r in results)}/{len(results)})")
    return 0 if doc["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
