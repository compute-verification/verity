#!/usr/bin/env python
"""[L1 RECORD 2026-09-10] Independent-host verification of the REC-P1 proofs: on a machine that never ran the prover (record-cpu), given only
the compressed proofs, the statement bytes and the typed objects (composed.json without witnesses is enough), run ``veritor-zk-host verify`` per
chunk and ``check_binding`` of the public values against the typed object (h_O || h_L || verdict = 1).  Writes verifier-independent.log + .json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
POC = HERE.parent
ROOT = POC.parent
for p in (str(POC), str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)


def host(hbin: str, args: list[str]) -> dict:
    res = subprocess.run([hbin, *args], capture_output=True, text=True, env={**os.environ, "SP1_PROVER": "cpu", "RUST_LOG": "warn"})
    if res.returncode != 0:
        raise RuntimeError(res.stderr[-300:])
    for line in reversed(res.stdout.strip().splitlines()):
        if line.strip().startswith("{"):
            return json.loads(line)
    raise RuntimeError("no json")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--proofs", required=True, help="dir with chunkNN-compressed.bin")
    ap.add_argument("--statements", required=True, help="dir with chunkNN/statement.bin + chunkNN/composed.json")
    ap.add_argument("--host", required=True)
    ap.add_argument("--approved", default=str(POC / "e2e" / "approved_guests.json"))
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from veritor.interface import TypedObligationSet, check_binding

    T0 = time.perf_counter()
    info = host(a.host, ["info"])
    approved = {g["elf_sha256"]: g for g in json.loads(Path(a.approved).read_text())["guests"]}
    g = approved.get(info["elf_sha256"])
    lines = [f"host {a.host} elf {info['elf_sha256']} vk {info['vk_hash']} approved={'yes' if g and str(g['vk_hash']).lower() == str(info['vk_hash']).lower() else 'NO'} sp1 {info.get('sp1_version')} machine {os.uname().nodename} ({os.cpu_count()} cpus visible)"]
    chunks = sorted(p for p in Path(a.statements).iterdir() if p.is_dir() and p.name.startswith("chunk"))

    def one(cdir: Path):
        t = time.perf_counter()
        pf = Path(a.proofs) / f"{cdir.name}-compressed.bin"
        if not pf.exists():
            return cdir.name, False, "no proof", 0.0
        try:
            vr = host(a.host, ["verify", "--proof", str(pf), "--statement", str(cdir / "statement.bin")])
        except RuntimeError as e:
            return cdir.name, False, f"host verify error {e}", time.perf_counter() - t
        rec = json.loads((cdir / "composed.json").read_text())
        obj = TypedObligationSet.from_json(rec["object"])
        pv = bytes.fromhex(vr["public_values"])
        b = check_binding(obj, pv, statement_format=int(rec.get("statement_format", 2)))
        h_l = hashlib.sha256((cdir / "statement.bin").read_bytes()).hexdigest()
        ok = bool(vr.get("ok")) and bool(vr.get("statement_match")) and bool(vr.get("verdict")) and not vr.get("unsound") \
            and str(vr.get("vk_hash")).lower() == str(info["vk_hash"]).lower() and b.ok and b.verdict == 1 and pv[32:64].hex() == h_l and pv[:32].hex() == rec["digest"]
        return cdir.name, ok, f"ok={vr.get('ok')} statement_match={vr.get('statement_match')} verdict={vr.get('verdict')} vk_match={str(vr.get('vk_hash')).lower() == str(info['vk_hash']).lower()} binding={b.ok} b.verdict={b.verdict} obligations={len(obj.obligations)} proof_bytes={pf.stat().st_size}", time.perf_counter() - t

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        results = list(ex.map(one, chunks))
    n_ok = 0
    rows = []
    for name, ok, detail, sec in results:
        n_ok += ok
        lines.append(f"{name} {'PASS' if ok else 'FAIL'} {detail} {sec:.1f}s")
        rows.append({"chunk": name, "ok": ok, "detail": detail, "seconds": round(sec, 2)})
    total = round(time.perf_counter() - T0, 1)
    lines.append(f"independent verification: {n_ok}/{len(chunks)} proofs verified and bound; wall {total} s with {a.workers} workers; approved guest {'yes' if g else 'NO'}")
    out = Path(a.out)
    out.write_text("\n".join(lines) + "\n")
    out.with_suffix(".json").write_text(json.dumps({"schema": "veritor/rec-p1-independent-verify/v1", "host": info, "machine": os.uname().nodename, "verified": n_ok, "chunks": len(chunks), "wall_seconds": total, "workers": a.workers, "rows": rows}, indent=1) + "\n")
    print(lines[-1])
    return 0 if n_ok == len(chunks) else 1


if __name__ == "__main__":
    sys.exit(main())
