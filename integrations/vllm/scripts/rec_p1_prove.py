#!/usr/bin/env python
"""[L1 RECORD 2026-09-10] Compressed proofs of every chunked statement under the approved guest, and the host-side verification (DIRECTIVE v3 §1.1 step 7).

Per chunk: ``veritor-zk-host execute`` (CPU; exact cycles, the guest's verdict) -> ``prove --mode compressed`` (SP1_PROVER=cuda) ->
``verify`` (CPU) -> ``check_binding`` of the 65-byte public values (h_O || h_L || verdict) against the chunk's typed object.  The host
binary's ELF/vk are checked against ``e2e/approved_guests.json`` FIRST; a mismatch refuses everything.  Proof wall times per chunk are
appended to ``costlog.jsonl`` (class ``proving``).  Chunks already proven (``proof-compressed.bin`` + ``prove.json``) are skipped
(resumable).  Nothing here changes a statement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
POC = HERE.parent
ROOT = POC.parent
for p in (str(POC), str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def host_json(stdout: str) -> dict:
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"no JSON line in host output: {stdout[-400:]!r}")


class HostNotCuda(SystemExit):
    """Named refusal: the host binary was built without sp1-sdk's ``cuda`` feature, so ``SP1_PROVER=cuda`` panics on the first prove."""

    code_name = "HOST_NOT_CUDA"


# sp1-sdk 6.4.0 `src/env/mod.rs:110` -- this panic is `#[cfg(not(feature = "cuda"))]`-gated, so the string is present in the binary
# only when the host was built WITHOUT the cuda feature (attempt 6 first pass: 4/4 chunks failed on it after execute succeeded).
NONCUDA_PANIC = b"The CUDA prover requires the `cuda` feature to be enabled"


def preflight_cuda_host(host: str, *, prover: str) -> dict:
    """A9 W-A9-16 (attempt 6): refuse a non-cuda host UP FRONT with a named error instead of failing every chunk's prove.
    With ``prover != "cuda"`` nothing is checked (the CPU prover needs no feature).  The check is a byte scan of the binary for
    sp1-sdk's cfg-gated panic string; a binary that lacks it is not proven cuda-capable (UNVALIDATED here: no host binary on the laptop),
    it merely has no known reason to refuse.  ``--allow-noncuda-host`` skips the refusal (recorded in guest.json)."""
    data = Path(host).read_bytes()
    found = NONCUDA_PANIC in data
    return {"prover": prover, "noncuda_panic_string_present": found, "verdict": "HOST_NOT_CUDA" if (prover == "cuda" and found) else ("not checked (prover is not cuda)" if prover != "cuda" else "no refusal (no non-cuda marker)")}


def run_host(host: str, name: str, args: list[str], log_dir: Path, *, prover: str) -> tuple[dict, float]:
    env = {**os.environ, "SP1_PROVER": prover}
    env.setdefault("RUST_LOG", "info")
    t = time.perf_counter()
    res = subprocess.run([host, *args], capture_output=True, text=True, env=env)
    wall = round(time.perf_counter() - t, 2)
    (log_dir / f"host-{name}.log").write_text(res.stderr + "\n" + res.stdout)
    if res.returncode != 0:
        raise RuntimeError(f"host {name} exited {res.returncode}: {res.stderr[-600:]}")
    return host_json(res.stdout), wall


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--record", required=True)
    ap.add_argument("--host", required=True)
    ap.add_argument("--approved", default=str(POC / "e2e" / "approved_guests.json"))
    # [E10 EOD 2026-09-11] the expected ELF is no longer fixed to 9fd00e60: the host's guest must be an APPROVED (`approval == "reproduced"`) entry of
    # approved_guests.json (E7 is its sole writer; the consolidated FTZ/TokenSelect guest gets a new digest there).  `--expect-elf` pins one on purpose;
    # `--expect-elf-any-of a,b` pins the run config's `guest.approved_elf_sha256` list (run_v5.sh passes it).
    ap.add_argument("--expect-elf", default=None, help="pin exactly this ELF sha256 (default: any approved `reproduced` guest of --approved)")
    ap.add_argument("--expect-elf-any-of", default=None, help="comma-separated ELF sha256 list the host's guest must be in (the run config's guest.approved_elf_sha256)")
    ap.add_argument("--hardware", default=os.environ.get("VERITOR_HARDWARE", "unknown GPU (set VERITOR_HARDWARE)"), help="hardware label written into every costlog row (MEASURED on this)")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--skip-execute", action="store_true")
    ap.add_argument("--skip-verify", action="store_true", help="leave host verify + check_binding to rec_p1_verify (parallel); prove only")
    ap.add_argument("--deadline-utc", default=None, help="HH:MM UTC; stop starting new proofs after this")
    ap.add_argument("--prover", default="cuda", choices=("cuda", "cpu"), help="SP1_PROVER for `prove` (default cuda; a non-cuda host is refused up front)")
    ap.add_argument("--allow-noncuda-host", action="store_true", help="do not refuse a host whose binary carries sp1-sdk's non-cuda panic marker")
    a = ap.parse_args(argv)
    rec = Path(a.record)
    odir = rec / "obligations"
    pdir = rec / "proofs"
    pdir.mkdir(exist_ok=True)
    logs = pdir / "logs"
    logs.mkdir(exist_ok=True)
    from veritor.interface import TypedObligationSet, check_binding

    # [E10 EOD 2026-09-11] the host-side check_binding validates the typed object -> its family must be resolvable in veritor.interface; the
    # CK-ELEM / FTZ / select families are folded in lazily by the checkout's builders (E7 `builders.ensure_families`, verifier side only).
    # Without it every chunk of a non-legacy family reads "PROOF DOES NOT VERIFY ... E_UNSUPPORTED_FAMILY" (E7 `e7-guest-build` step 8, 22:23Z).
    # Every fold-in the checkout has is called (each idempotent): E7's `builders.ensure_families` (eod/e7 fccf9291, folds both), else CK-ELEM's
    # and E4's `ck_ftz.ensure_families` directly (vllm-poc @ 782e95f5 has those two, not the builders one).
    _folded = []
    for _mod in ("e2e.coverage_v1.builders", "e2e.coverage_v1.ck_elem", "e2e.coverage_v1.ck_ftz"):
        try:
            getattr(__import__(_mod, fromlist=["ensure_families"]), "ensure_families")()
            _folded.append(_mod.rsplit(".", 1)[1])
        except Exception as e:  # older checkouts: the interface carries the families itself
            print(f"[rec_p1_prove] note: {_mod}.ensure_families unavailable ({type(e).__name__}: {str(e)[:80]})", flush=True)
    print(f"[rec_p1_prove] families folded into veritor.interface by: {_folded or 'none (interface built-ins only)'}", flush=True)

    # W-A9-16: the cuda preflight comes BEFORE the (cheap) `info` call and before any chunk is touched
    pre = preflight_cuda_host(a.host, prover=a.prover)
    if pre["verdict"] == HostNotCuda.code_name and not a.allow_noncuda_host:
        (pdir / "prover_exit.json").write_text(json.dumps({"kind": HostNotCuda.code_name, "host": a.host, "preflight": pre,
                                                           "reason": "host binary was built without sp1-sdk's `cuda` feature; SP1_PROVER=cuda would panic on every prove "
                                                                     "(attempt 6 first pass: 4/4). Rebuild with --features cuda, or pass --prover cpu / --allow-noncuda-host"}, indent=1) + "\n")
        print(f"[rec_p1_prove] {HostNotCuda.code_name}: {a.host} lacks the cuda feature (non-cuda panic marker present); no chunk was attempted", file=sys.stderr, flush=True)
        raise HostNotCuda(3)

    info, _ = run_host(a.host, "info", ["info"], logs, prover="cpu")
    approved = json.loads(Path(a.approved).read_text())
    guests = {g["elf_sha256"]: g for g in approved["guests"]}
    g = guests.get(info["elf_sha256"])
    allowed = [x.strip() for x in (a.expect_elf_any_of or "").split(",") if x.strip()]
    if a.expect_elf:
        allowed = [a.expect_elf]
    if g is None or g.get("approval") != "reproduced" or str(g["vk_hash"]).lower() != str(info["vk_hash"]).lower():
        raise SystemExit(f"NOT BOUND: host guest {info.get('elf_sha256')} / {info.get('vk_hash')} is not an approved (reproduced) guest of {a.approved} "
                         f"(entry: {(g or {}).get('approval')} / {(g or {}).get('vk_hash')})")
    if allowed and info["elf_sha256"] not in allowed:
        raise SystemExit(f"NOT BOUND: host guest {info['elf_sha256']} is approved but not among the run's declared ELFs {allowed}")
    guest_doc = {"elf_sha256": info["elf_sha256"], "vk_hash": info["vk_hash"], "approval": g["approval"], "sp1_version": info.get("sp1_version"), "guest_features": info.get("guest_features"),
                 "host_sha256": sha256_file(Path(a.host)), "host": a.host, "approved_guests_sha256": sha256_file(Path(a.approved)),
                 "elf_pin": {"expect_elf": a.expect_elf, "expect_elf_any_of": allowed or None, "rule": "host guest must be an approved `reproduced` entry; and in the declared list when one is given"},
                 "hardware": a.hardware, "cuda_preflight": {**pre, "allow_noncuda_host": bool(a.allow_noncuda_host)}}
    (pdir / "guest.json").write_text(json.dumps(guest_doc, indent=1, sort_keys=True) + "\n")
    print(f"[rec_p1_prove] guest {info['elf_sha256'][:12]} vk {info['vk_hash'][:14]} approved ({g['approval']}); host {guest_doc['host_sha256'][:12]}", flush=True)

    def ready_chunks() -> list[Path]:
        # composed.json is written last by rec_p1_obligations: a chunk is ready when it exists
        return sorted(p for p in odir.iterdir() if p.is_dir() and p.name.startswith("chunk") and (p / "statement.bin").exists() and (p / "composed.json").exists())

    results = []
    stop = False
    done: set[str] = set()
    chunks: list[Path] = []
    while True:
        pending = [c for c in ready_chunks() if c.name not in done]
        if not pending:
            if (odir / "obligations.json").exists():
                break
            time.sleep(10)
            continue
        cdir = pending[0]
        done.add(cdir.name)
        chunks.append(cdir)
        if a.deadline_utc:
            hh, mm = map(int, a.deadline_utc.split(":"))
            now = time.gmtime()
            if (now.tm_hour, now.tm_min) >= (hh, mm):
                stop = True
        out = pdir / f"{cdir.name}-compressed.bin"
        rec_path = pdir / f"{cdir.name}.json"
        if rec_path.exists() and out.exists():
            results.append(json.loads(rec_path.read_text()))
            continue
        if stop:
            results.append({"chunk": cdir.name, "status": "NOT PROVEN: deadline reached before this chunk started"})
            print(f"[rec_p1_prove] {cdir.name}: deadline reached, not started", flush=True)
            continue
        record = json.loads((cdir / "composed.json").read_text())
        h_o, h_l, fmt = record["digest"], record["statement_digest"], int(record.get("statement_format", 2))
        st, wt = str(cdir / "statement.bin"), str(cdir / "witness.bin")
        common = ["--statement", st, "--witness", wt, "--typed-object-digest", h_o]
        r: dict = {"chunk": cdir.name, "h_O": h_o, "h_L": h_l, "statement_format": fmt, "statement_sha256": sha256_file(cdir / "statement.bin"),
                   "witness_sha256": sha256_file(cdir / "witness.bin"), "obligations": len(record["object"]["obligations"])}
        if not a.skip_execute:
            ex, w = run_host(a.host, f"{cdir.name}-execute", ["execute", *common], logs, prover="cpu")
            r["execute"] = {"total_cycles": ex.get("total_cycles"), "verdict": ex.get("verdict"), "unsound": ex.get("unsound"), "cycle_tracker": ex.get("cycle_tracker"),
                            "syscalls": ex.get("syscalls"), "seconds": w}
        attempts = 0
        while True:
            attempts += 1
            try:
                pr, w = run_host(a.host, f"{cdir.name}-prove", ["prove", *common, "--out", str(out), "--mode", "compressed"], logs, prover=a.prover)
                break
            except RuntimeError as e:
                if attempts > a.retries:
                    r["status"] = f"NOT PROVEN: {str(e)[-300:]}"
                    print(f"[rec_p1_prove] {cdir.name}: FAILED after {attempts} attempts: {str(e)[-200:]}", flush=True)
                    pr = None
                    break
                print(f"[rec_p1_prove] {cdir.name}: prove attempt {attempts} failed ({str(e)[-160:]!r}); retrying in 20 s", flush=True)
                time.sleep(20)
        if pr is None:
            results.append(r)
            rec_path.write_text(json.dumps(r, indent=1, sort_keys=True) + "\n")
            continue
        r["prove"] = {"attempts": attempts, "wall_seconds": w, "proof_bytes": pr.get("proof_bytes"), "shards": pr.get("shards"), "setup_seconds": pr.get("setup_seconds"),
                      "prove_seconds": pr.get("prove_seconds"), "host_verify_seconds": pr.get("verify_seconds"), "proof_sha256": sha256_file(out)}
        if a.skip_verify:
            r["status"] = "PROVEN (verification deferred to rec_p1_verify)"
            rec_path.write_text(json.dumps(r, indent=1, sort_keys=True) + "\n")
            with (rec / "costlog.jsonl").open("a") as f:
                f.write(json.dumps({"stage": f"prove {cdir.name} (compressed, SP1 cuda)", "class": "proving", "seconds": r["prove"]["wall_seconds"], "hardware": a.hardware,
                                    "batch": f"{r['obligations']} obligations, {r.get('execute', {}).get('total_cycles')} cycles", "label": "MEASURED",
                                    "note": f"setup {pr.get('setup_seconds')} s, prove {pr.get('prove_seconds')} s, shards {pr.get('shards')}; execute (cpu) {r.get('execute', {}).get('seconds')} s"}) + "\n")
            results.append(r)
            print(f"[rec_p1_prove] {cdir.name}: PROVEN; cycles {r.get('execute', {}).get('total_cycles')}; prove {r['prove']['wall_seconds']} s ({pr.get('proof_bytes')} B)", flush=True)
            continue
        vr, w = run_host(a.host, f"{cdir.name}-verify", ["verify", "--proof", str(out), "--statement", st], logs, prover="cpu")
        pv = bytes.fromhex(vr["public_values"])
        b = check_binding(TypedObligationSet.from_json(record["object"]), pv, statement_format=fmt)
        ok = bool(vr.get("ok")) and bool(vr.get("statement_match")) and bool(vr.get("verdict")) and not vr.get("unsound") and str(vr.get("vk_hash")).lower() == str(info["vk_hash"]).lower() and b.ok
        r["verify"] = {"ok": vr.get("ok"), "statement_match": vr.get("statement_match"), "verdict": vr.get("verdict"), "unsound": vr.get("unsound"), "vk_hash": vr.get("vk_hash"),
                       "public_values": vr.get("public_values"), "wall_seconds": w,
                       "binding": {"ok": b.ok, "code": str(getattr(b, "code", "")), "detail": getattr(b, "detail", ""), "verdict": getattr(b, "verdict", None), "public_values_len": len(pv)}}
        r["status"] = "PROVEN+VERIFIED" if ok else "PROOF DOES NOT VERIFY"
        rec_path.write_text(json.dumps(r, indent=1, sort_keys=True) + "\n")
        with (rec / "costlog.jsonl").open("a") as f:
            f.write(json.dumps({"stage": f"prove {cdir.name} (compressed, SP1 cuda)", "class": "proving", "seconds": r["prove"]["wall_seconds"], "hardware": a.hardware,
                                "batch": f"{r['obligations']} obligations, {r.get('execute', {}).get('total_cycles')} cycles", "label": "MEASURED",
                                "note": f"setup {pr.get('setup_seconds')} s, prove {pr.get('prove_seconds')} s, shards {pr.get('shards')}; execute (cpu) {r.get('execute', {}).get('seconds')} s"}) + "\n")
            f.write(json.dumps({"stage": f"verify {cdir.name} (host verify + check_binding)", "class": "verifier", "seconds": w, "hardware": a.hardware + " CPU", "batch": "1 compressed proof", "label": "MEASURED"}) + "\n")
        results.append(r)
        print(f"[rec_p1_prove] {cdir.name}: {r['status']}; cycles {r.get('execute', {}).get('total_cycles')}; prove {r['prove']['wall_seconds']} s ({pr.get('proof_bytes')} B); verify {w} s; binding {b.ok}", flush=True)
    summary = {"schema": "veritor/rec/p1-proofs/v1", "guest": guest_doc, "chunks": len(chunks), "proven_verified": sum(str(r.get("status", "")).startswith("PROVEN") for r in results),
               "not_proven": [r["chunk"] for r in results if not str(r.get("status", "")).startswith("PROVEN")],
               "total_cycles": sum((r.get("execute") or {}).get("total_cycles") or 0 for r in results),
               "prove_wall_seconds": round(sum((r.get("prove") or {}).get("wall_seconds") or 0 for r in results), 1), "results": results,
               "provenance": {"label": "MEASURED", "hardware": a.hardware + " (prove on its GPU; execute/verify on its CPU)", "script": "scripts/rec_p1_prove.py"}}
    (pdir / "proofs.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n")
    print(f"[rec_p1_prove] DONE: {summary['proven_verified']}/{len(chunks)} chunks proven+verified; {summary['total_cycles']} cycles; prove wall {summary['prove_wall_seconds']} s; not proven {summary['not_proven']}")
    return 0 if not summary["not_proven"] else 1


if __name__ == "__main__":
    sys.exit(main())
