"""The ACTUAL-GUEST smoke test (GUIDANCE §57.10, S12 caveat): valid / invalid statements through the real zkVM build.

Native Rust tests do not establish the behaviour of the zkVM build; this runs the built host's SP1 *executor*
(``host execute``, no proving) on the real ELF over a small matrix and compares the guest's public values with
what the Python side computes independently:

  (a) a valid format-3 statement + packed witness           -> accept; public values = h_O || sha256(v3 bytes) || 01
  (b) the SAME object as format 2                            -> accept; identical public values except bytes 32..64 (h_L)
  (c) a tampered leaf value (format 3)                       -> reject (verdict 0 over the same h_L)
  (d) malformed ranges (count 0 / beyond the leaf)           -> reject (parse class: verdict 0 over sha256 of the bytes)
  (e) an unauthenticated shared-path reuse (format-2 witness) -> reject
  plus controls: the honest multiproof vector accepts; every vectors.json accept/reject family member agrees.

Then ONE actual CPU proof (``host prove --mode core``) of the tiny valid format-3 object and ``host verify`` with
the vk, wall time + peak RSS measured with ``/usr/bin/time -v``.

    python -m e2e.guest_smoke --host zk/sp1/target/release/veritor-zk-host --out results/guest-smoke [--prove]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE.parent))

from e2e.run_composed import host_json  # noqa: E402

VECTORS = ROOT / "fixtures" / "proof-format-v3" / "vectors.json"
TYPED = ROOT / "fixtures" / "typed-obligation-v0"


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def run_host(host: Path, log: Path, *args: str, timed: bool = False) -> tuple[dict[str, Any], float, int | None, str]:
    """``host <args>`` under SP1_PROVER=cpu; returns (json, wall_s, peak_rss_kb | None, stderr_tail)."""

    cmd = [str(host), *args]
    time_v = Path("/usr/bin/time")
    if timed and time_v.exists():
        cmd = [str(time_v), "-v", *cmd]
    env = {**os.environ, "SP1_PROVER": "cpu", "RUST_LOG": os.environ.get("RUST_LOG", "warn")}
    t = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    wall = time.perf_counter() - t
    log.write_text(f"$ {' '.join(cmd)}\n--- stdout\n{p.stdout}\n--- stderr\n{p.stderr}\n--- rc {p.returncode} wall {wall:.2f}s\n")
    rss = None
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", p.stderr)
    if m:
        rss = int(m.group(1))
    if p.returncode != 0 and not p.stdout.strip().startswith("{"):
        return {"error": f"rc {p.returncode}", "stderr_tail": p.stderr[-800:]}, wall, rss, p.stderr[-800:]
    try:
        doc = host_json(p.stdout)
    except RuntimeError as e:
        doc = {"error": str(e), "stderr_tail": p.stderr[-800:]}
    return doc, wall, rss, p.stderr[-400:]


def execute(host: Path, out: Path, name: str, statement: bytes, witness: bytes, *, plan: bytes | None = None, object_digest: str | None = None) -> dict[str, Any]:
    d = out / name.replace("/", "__")
    d.mkdir(parents=True, exist_ok=True)
    (d / "statement.bin").write_bytes(statement)
    (d / "witness.bin").write_bytes(witness)
    args = ["execute", "--statement", str(d / "statement.bin"), "--witness", str(d / "witness.bin")]
    if plan:
        (d / "plan.bin").write_bytes(plan)
        args += ["--plan", str(d / "plan.bin")]
    if object_digest:
        args += ["--typed-object-digest", object_digest]
    doc, wall, _, _ = run_host(host, d / "execute.log", *args)
    return {"dir": str(d), "wall_seconds": round(wall, 3), "statement_sha256": sha256(statement), "statement_bytes": len(statement), "witness_bytes": len(witness), **doc}


def expect_pv(statement: bytes, verdict: bool, object_digest: str | None) -> str:
    """What a sound guest must commit: [h_O ||] sha256(statement bytes) || verdict byte."""

    return (object_digest or "") + sha256(statement) + ("01" if verdict else "00")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prove", action="store_true", help="also produce ONE core proof of the valid format-3 object and verify it")
    ap.add_argument("--prove-mode", default="core", choices=("core", "compressed"))
    ap.add_argument("--all-vectors", action="store_true", help="execute every vectors.json vector (default: the (a)-(e) matrix + controls)")
    a = ap.parse_args(argv)
    host, out = Path(a.host).resolve(), Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    from veritor.interface.binding import bound_public_values

    info, _, _, _ = run_host(host, out / "info.log", "info")
    rows: list[dict[str, Any]] = []

    def row(label: str, name: str, statement: bytes, witness: bytes, expect_verdict: bool, *, plan: bytes | None = None, object_digest: str | None = None,
            expected_class: str | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        r = execute(host, out, name, statement, witness, plan=plan, object_digest=object_digest)
        want_pv = expect_pv(statement, expect_verdict, object_digest)
        got_pv = r.get("public_values")
        ok = (r.get("verdict") is expect_verdict) and (got_pv == want_pv) and (r.get("unsound") is False) and (r.get("statement_digest") == sha256(statement))
        r.update({"label": label, "name": name, "expect_verdict": expect_verdict, "expected_class": expected_class, "expected_public_values": want_pv,
                  "public_values_match": got_pv == want_pv, "ok": ok, **(extra or {})})
        rows.append(r)
        print(f"[guest_smoke] {label:5s} {name:60s} verdict={r.get('verdict')!s:5s} expect={expect_verdict!s:5s} pv_match={got_pv == want_pv} cycles={r.get('total_cycles')} {'OK' if ok else 'MISMATCH'}", flush=True)
        return r

    # --- (a)/(b): the tiny real typed object, format 3 then format 2, with h_O carried through -------------------------
    obj_vec = json.loads((TYPED / "accept--packed-word-row.json").read_text())
    obj, h_o = obj_vec["object"], obj_vec["digest"]
    v3, v2, wit = bytes.fromhex(obj_vec["statement_v3"]), bytes.fromhex(obj_vec["statement"]), bytes.fromhex(obj_vec["witness"])
    ra = row("(a)", "typed/packed-word-row/format-3", v3, wit, True, object_digest=h_o,
             extra={"python_bound_public_values": bound_public_values(obj, statement_format=3).hex(), "h_O": h_o, "h_L_v3": obj_vec["statement_v3_digest"]})
    rb = row("(b)", "typed/packed-word-row/format-2", v2, wit, True, object_digest=h_o,
             extra={"python_bound_public_values": bound_public_values(obj).hex(), "h_O": h_o, "h_L_v2": obj_vec["statement_digest"]})
    ab = {"same_h_O": (ra.get("public_values") or "")[:64] == (rb.get("public_values") or "")[:64],
          "same_verdict_byte": (ra.get("public_values") or "")[128:] == (rb.get("public_values") or "")[128:],
          "h_L_differs": (ra.get("public_values") or "")[64:128] != (rb.get("public_values") or "")[64:128],
          "a_matches_python_bound_public_values": ra.get("public_values") == ra["python_bound_public_values"],
          "b_matches_python_bound_public_values": rb.get("public_values") == rb["python_bound_public_values"],
          "cycles": {"format_3": ra.get("total_cycles"), "format_2": rb.get("total_cycles")}}
    # --- (c): tampered leaf value, format 3 (the real object's negative, and the synthetic vector) ----------------------
    tam = json.loads((TYPED / "accept--packed-word-tampered-leaf.json").read_text())
    row("(c)", "typed/packed-word-tampered-leaf/format-3", bytes.fromhex(tam["statement_v3"]), bytes.fromhex(tam["witness"]), False, object_digest=tam["digest"],
        expected_class=tam["witness_verdict"])
    vec = json.loads(VECTORS.read_text())
    by_name = {v["name"]: v for v in vec["vectors"]}

    def vrow(label: str, name: str, fmt: int) -> None:
        v = by_name[name]
        st = bytes.fromhex(v["statement_v3"] if fmt == 3 else v["statement_v2"])
        wit_b = bytes.fromhex(v["witness"]) if v.get("witness") else b""
        row(label, f"{name}/format-{fmt}", st, wit_b, v["expected"] == "accept", expected_class=v["expected"], extra={"family": v["family"], "witness_format": v.get("witness_format")})

    vrow("(c)", "reject/tampered-leaf-value", 3)
    vrow("(c)", "reject/wrong-root", 3)
    # --- (d): malformed ranges -----------------------------------------------------------------------------------------
    for n in ("parse-reject/count-zero", "parse-reject/beyond-leaf", "parse-reject/stride-zero", "parse-reject/slot-outside-expansion"):
        vrow("(d)", n, 3)
    # --- (e): unauthenticated shared-path reuse (format-2 witness) + the honest control ----------------------------------
    vrow("(e)", "multiproof/shared-path-reused-for-unopened-rank", 2)
    vrow("(e)", "multiproof/shared-path-with-honest-value-of-unopened-rank", 2)
    vrow("ctl", "multiproof/honest", 2)
    vrow("ctl", "accept/one-tree-range-plus-single", 3)
    vrow("ctl", "accept/one-tree-range-plus-single", 2)
    if a.all_vectors:
        done = {r["name"] for r in rows}
        for v in vec["vectors"]:
            for fmt in (3, 2):
                if not v.get("statement_v3" if fmt == 3 else "statement_v2"):
                    continue  # parse-reject vectors exist only as format-3 bytes
                if v.get("witness_format") == 2 and fmt == 3:
                    continue  # a format-2 (multiproof) witness pairs with a format-2 statement
                if f"{v['name']}/format-{fmt}" not in done:
                    vrow("all", v["name"], fmt)

    result: dict[str, Any] = {
        "host": {"path": str(host), "sha256": sha256(host.read_bytes()), **{k: info.get(k) for k in ("elf_sha256", "vk_hash", "sp1_version", "guest_features", "public_values_len")}},
        "rows": rows, "a_vs_b": ab,
        "matrix_ok": all(r["ok"] for r in rows) and all(ab[k] for k in ("same_h_O", "same_verdict_byte", "h_L_differs", "a_matches_python_bound_public_values", "b_matches_python_bound_public_values")),
        "rows_total": len(rows), "rows_ok": sum(r["ok"] for r in rows),
    }
    print(f"[guest_smoke] matrix {'OK' if result['matrix_ok'] else 'FAILED'}: {result['rows_ok']}/{result['rows_total']} rows agree; (a) vs (b): {ab}", flush=True)

    if a.prove:
        pdir = out / "proof"
        pdir.mkdir(exist_ok=True)
        d = out / "typed__packed-word-row__format-3"
        proof = pdir / f"proof-{a.prove_mode}.bin"
        pr, wall, rss, _ = run_host(host, pdir / f"prove-{a.prove_mode}.log", "prove", "--statement", str(d / "statement.bin"), "--witness", str(d / "witness.bin"),
                                    "--typed-object-digest", h_o, "--mode", a.prove_mode, "--out", str(proof), timed=True)
        vr, vwall, vrss, _ = run_host(host, pdir / f"verify-{a.prove_mode}.log", "verify", "--proof", str(proof), "--statement", str(d / "statement.bin"), timed=True)
        result["proof"] = {"mode": a.prove_mode, "prove": {**pr, "wall_seconds": round(wall, 2), "peak_rss_kb": rss}, "verify": {**vr, "wall_seconds": round(vwall, 2), "peak_rss_kb": vrss},
                           "proof_bytes": proof.stat().st_size if proof.exists() else None, "proof_sha256": sha256(proof.read_bytes()) if proof.exists() else None,
                           "verify_ok": bool(vr.get("ok")) and bool(vr.get("statement_match")) and bool(vr.get("verdict")) and not vr.get("unsound") and vr.get("vk_hash") == info.get("vk_hash"),
                           "verify_public_values_match": vr.get("public_values") == ra["python_bound_public_values"]}
        print(f"[guest_smoke] proof {a.prove_mode}: prove {wall:.1f}s peak {rss} kB, verify ok={result['proof']['verify_ok']} {vwall:.1f}s; {result['proof']['proof_bytes']} B", flush=True)
    (out / "smoke.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    return 0 if result["matrix_ok"] and (not a.prove or result["proof"]["verify_ok"]) else 1


if __name__ == "__main__":
    sys.exit(main())
