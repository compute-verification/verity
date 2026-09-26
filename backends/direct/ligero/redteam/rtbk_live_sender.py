"""red-team-bligero-real-k: the streaming sender's verifier side (PR #55: head-only live checks on a check pool, several
data connections per session, ``serve --drop-files``) against malformed and replayed PROOF frames.

A loopback ``live.Server`` with the Rust core (``--ligero-verify``) and ``drop_files=True`` serves 2-sub-batch sessions
of a small relation; the prover runs the real coin exchange (COMMIT / STMT / ROOT / OPEN1 / TESTS / OPEN2) and then puts
PROOF frames on its data connections as each case dictates.  Every malformed case must end in a session REJECT; the
body-only cases must pass the head-only live check and be refused by the Rust verifier (so the check the stream skipped is
still made), and the honest session must be accepted with every sub-batch's ``proof_sha256`` / ``stmt_sha256`` in
``session.json`` equal to the bytes sent (what links a prover's dump to a session once the files are dropped).

    python -m backends.direct.ligero.redteam.rtbk_live_sender --bin BIN --out DIR [--relation fp8-ada]

Cases: honest; dup (PROOF 0 twice, on two connections); swap (PROOF 0 carries sub-batch 1's proof and vice versa);
extra (a PROOF for sub-batch 7 of 2); tail (4 bytes after the proof); flag-q2 / flag-v2 (a head flag byte 2: truthy for
the head reader, not 0/1 for Rust); hash-id (hash id 2); body-flip-<n> (one byte flipped behind the head: opened columns,
column indices, paths); truncated (the last path cut)."""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
import threading
import time
from pathlib import Path


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--relation", default="fp8-ada")
    ap.add_argument("--flips", type=int, default=6)
    a = ap.parse_args()
    import numpy as np

    from .. import live
    from .. import protocol as pm
    from ..relations import RELATIONS
    from ..relchain import RelationChainRunner, instances
    from ..serialize import proof_bytes, read_proof_head, system_bytes

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    srv = live.Server(f"127.0.0.1:{port}", out / "verifier", live.RustVerifier(str(Path(a.bin).resolve())), jobs=2, log=lambda m: None, drop_files=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    time.sleep(0.3)
    url = f"tcp://127.0.0.1:{port}"
    n_proofs = 2
    rel = RELATIONS[a.relation]
    R = RelationChainRunner(rel, "cpu", -100.0, 2, n_proofs=n_proofs, zk=True, mode=pm.MODE_INTERACTIVE)
    data = instances(rel, 2 * n_proofs, procs=1)
    hello = {"relation": a.relation, "mode": pm.MODE_INTERACTIVE, "zk": True, "batch": 2 * R.steps, "target": -100.0,
             "target_bits": 96.0, "run_id": "rtbk-live-sender", "device": "cpu", "n_proofs": n_proofs, "total_vus": 2 * n_proofs}
    V = live.Verifier(url, hello, system_bytes(R.sys))

    def run(case: str) -> dict:
        sess = V.session(n_proofs, 1, {"negative": case})
        pbs = []
        for i in range(n_proofs):
            proof, _pubs, _lay = R.prove_vus(data[2 * i:2 * i + 2], check=False, min_l=256, coins=sess.coins(i))
            pbs.append(proof_bytes(proof))
        head = read_proof_head(pbs[0])
        # magic, 4 flag bytes, 8 u32 sizes, 6 coin fields, root, then w, h, q, v: what read_proof_head reads
        head_end = 8 + 4 + 32 + (6 * 32 if head.coins is not None else 0) + 32 + sum(
            x.nbytes for x in (head.w, head.h, head.q, head.v) if x is not None)
        sends: list[tuple[int, bytes]] = [(0, pbs[0]), (1, pbs[1])]
        if case == "dup":
            sends = [(0, pbs[0]), (0, pbs[0]), (1, pbs[1])]
        elif case == "swap":
            sends = [(0, pbs[1]), (1, pbs[0])]
        elif case == "extra":
            sends = [(0, pbs[0]), (1, pbs[1]), (7, pbs[1])]
        elif case == "tail":
            sends = [(0, pbs[0] + b"\0\0\0\0"), (1, pbs[1])]
        elif case in ("flag-q2", "flag-v2", "hash-id"):
            b = bytearray(pbs[0])
            b[{"hash-id": 8, "flag-v2": 10, "flag-q2": 11}[case]] = 2
            sends = [(0, bytes(b)), (1, pbs[1])]
        elif case.startswith("body-flip-"):
            k = int(case.rsplit("-", 1)[1])
            rng = np.random.default_rng(1000 + k)
            pos = int(rng.integers(head_end, len(pbs[0])))
            b = bytearray(pbs[0])
            b[pos] ^= 1 << int(rng.integers(0, 8))
            sends = [(0, bytes(b)), (1, pbs[1])]
        elif case == "truncated":
            sends = [(0, pbs[0][:-32]), (1, pbs[1])]
        for i, pb in sends:
            sess.submit(i, pb)
        try:
            v = sess.end({"case": case})
        except (live.LiveError, ConnectionError, OSError) as e:
            v = {"accepted": False, "reason": f"prover side: {e}", "session_id": sess.session_id, "live_rejections": [], "errors": []}
        sd = out / "verifier" / v["session_id"]
        rec = json.loads((sd / "session.json").read_text()) if (sd / "session.json").is_file() else {"subbatches": []}
        links = [{"sub": s["sub"], "proof_sha256_ok": s.get("proof_sha256") == hashlib.sha256(pbs[s["sub"]]).hexdigest(),
                  "live_check": (s.get("live_check") or {}).get("accepted"), "verdict": (s.get("verdict") or {}).get("accepted"),
                  "verdict_reason": str((s.get("verdict") or {}).get("reason"))[:140]} for s in rec["subbatches"] if "sub" in s]
        dropped = not any(sd.glob("*.proof")) and not any(sd.glob("*.stmt"))
        return {"case": case, "accepted": bool(v["accepted"]), "reason": str(v["reason"])[:300], "session": v["session_id"],
                "live_rejections": v.get("live_rejections"), "errors": [str(e)[:160] for e in v.get("errors") or []],
                "subbatches": links, "files_dropped": dropped, "head_end": head_end, "proof_bytes": len(pbs[0])}

    cases = ["honest", "dup", "swap", "extra", "tail", "flag-q2", "flag-v2", "hash-id", "truncated"] + \
            [f"body-flip-{k}" for k in range(a.flips)]
    rows, bad = [], []
    for c in cases:
        r = run(c)
        body_only = c.startswith("body-flip-") or c in ("tail", "truncated", "flag-q2", "flag-v2", "hash-id")
        if c == "honest":
            ok = r["accepted"] and all(s["proof_sha256_ok"] and s["verdict"] for s in r["subbatches"]) and r["files_dropped"]
        elif body_only:
            s0 = next((s for s in r["subbatches"] if s["sub"] == 0), {})
            # the head-only live check lets it through (or cannot read it); the Rust verdict refuses it; the session rejects
            ok = not r["accepted"] and s0.get("verdict") is False
        else:
            ok = not r["accepted"]
        r["ok"] = bool(ok)
        rows.append(r)
        print(f"{'OK ' if ok else 'BAD'} {c}: accepted={r['accepted']} live_rejections={len(r['live_rejections'] or [])} "
              f"sub0={next((s for s in r['subbatches'] if s['sub'] == 0), {})} :: {r['reason'][:150]}", flush=True)
        if not ok:
            bad.append(c)
    srv.stop.set()
    (out / "live_sender.json").write_text(json.dumps({"ok": not bad, "bad": bad, "cases": rows}, indent=1))
    print(f"\n{len(rows) - len(bad)}/{len(rows)} as expected; unexpected: {bad or 'none'}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
