"""R3-7 / R3-10 end to end (GPU): a loopback ``live.Server`` (``ChallengeServerSession``: the verifier's own coins and
session record), one honest live proof, and cheating provers that run a full, ACCEPTED session against it but prove with
coin slots the verifier never issued, writing a dump whose ``.coins`` / ``stream_binding`` re-derive those slots (so
``_file_coins`` -- the cold ``verify-dir`` path, which claims nothing -- accepts them).  ``verify_session`` must reject
every one: it replays with ``record_slots`` (the record alone).

  A  label grinding: after r_0 is opened, the prover tries labels until slot 0 has 8 zero bits; the wire carries the
     protocol label, the dump the ground one.
  B  context grinding: every slot from ``batch_context(stmt, commitments + [junk])``.
  D  one record batch claimed by two proofs (the honest proof twice).
  E  abort-and-retry (R3-10): one more session for the same STMT, dropped after its first coin; the honest proof's claim
     must count every batch opened for its STMT (honest, A, B, E: +2 bits), and without ``--sessions-root`` there is no claim.

    python -m backends.direct.ligerito.live_session_gpu_test [--relation fp8-ada] [--zk] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import tempfile
import threading
import time
from pathlib import Path

from backends.direct.ligero import live as live_mod

from . import prove as prove_mod
from . import run as run_mod
from .proof import sha256


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class GroundLabel(prove_mod.StreamCoins):
    """Attack A: slot 0 re-derived under a label chosen after r_0 was opened."""

    def _slot(self, label: bytes) -> None:
        first = self.rounds == 0
        super()._slot(label)
        if first:
            ctx, r0 = self.stream._ctx, self.stream.openings()[0][0]
            t = 0
            while True:
                g = label + b"#" + str(t).encode()
                c = live_mod.expand_challenge(ctx, 0, g, r0, 32)
                if c[:1] == b"\0":
                    break
                t += 1
            self.slots_out[0], self.labels[0], self._coin = c, g, c


class ExtraCommitment(prove_mod.StreamCoins):
    """Attack B: every slot from a batch context with a trailing junk commitment."""

    junk = os.urandom(32)

    def _slot(self, label: bytes) -> None:
        k = self.rounds
        super()._slot(label)
        ctx = live_mod.batch_context(self.statement, self.stream.commitments() + [self.junk])
        c = live_mod.expand_challenge(ctx, k, label, self.stream.openings()[k][0], 32)
        self.slots_out[k], self._coin = c, c

    def binding(self) -> dict:
        b = super().binding()
        b["commitments"].append(self.junk.hex())
        return b


def _session(port: int, R: int, note: str) -> live_mod.ChallengeSession:
    return live_mod.ChallengeSession("127.0.0.1", port, {"lane": "ligerito-relation-3", "note": note}, 1, R, 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--relation", default="fp8-ada")
    ap.add_argument("--vus", type=int, default=12)
    ap.add_argument("--l", type=int, default=None)
    ap.add_argument("--zk", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    out = Path(args.out or tempfile.mkdtemp(prefix="lr3-live-"))
    if out.exists():
        shutil.rmtree(out)
    store = out / "verifier_sessions"
    port = _free_port()
    srv = live_mod.Server(f"127.0.0.1:{port}", store, None, log=lambda m: None)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    time.sleep(0.5)

    pv = run_mod.Prover(args.relation, args.device, args.l)
    bt = pv.batch(run_mod.instances(pv.rel, args.vus), "live", zk=args.zk)
    stmt = prove_mod.statement_for_stream(bt, bt.key_sha256)
    warm = prove_mod.StreamCoins(live_mod.LocalChallenges(4096), stmt)
    pv.prove(bt, warm)
    R = warm.rounds

    def run_session(cls, name):
        sess = _session(port, R, name)
        pc = cls(sess.stream(0), stmt)
        r = pv.prove(bt, pc)
        verdict = sess.end({})
        d = out / name
        dump = run_mod.Dump(d)
        dump.set_key(bt.key_bytes)
        extra = dump.coins_file("batch_live_0", pc.slots_out)
        extra["stream_binding"] = pc.binding()
        dump.add("batch_live_0", bt.stmt, r.proof, "accept", (True, "ok"), "live", extra)
        return sess.id, verdict, d, dump, r, pc

    results, fails = {}, 0

    def check(name, cond, detail):
        nonlocal fails
        fails += 0 if cond else 1
        results[name] = {"ok": bool(cond), **detail}
        print(f"{'ok  ' if cond else 'FAIL'} {name}: {json.dumps(detail, default=str)[:300]}", flush=True)

    meta = {"relation": pv.rel.name, "l": pv.l, "vus": args.vus, "kind": "live-session-test", "n_proofs": 1}
    sid, verdict, d_h, dump_h, r_h, pc_h = run_session(prove_mod.StreamCoins, "honest")
    dump_h.close(meta)
    vs = run_mod.verify_session(d_h, [store / sid], args.device, store)
    it = vs["items"][0]
    check("honest", verdict["accepted"] and it["authenticated"] and vs["job_claimed_log2"] is not None
          and it["dump_coins_equal_record"], {"verdict": verdict["accepted"], "claim": it.get("claim"), "why": it.get("why")})
    honest_claim = it.get("claim", {}).get("claimed_log2")

    for cls, name in ((GroundLabel, "A_label_grinding"), (ExtraCommitment, "B_context_grinding")):
        sid_x, verdict_x, d_x, dump_x, r_x, pc_x = run_session(cls, name)
        dump_x.close(meta)
        cold = run_mod.verify_dir(d_x, args.device)
        vs = run_mod.verify_session(d_x, [store / sid_x], args.device, store)
        it = vs["items"][0]
        check(name, verdict_x["accepted"] and not it["authenticated"] and vs["job_claimed_log2"] is None,
              {"verifier_verdict": verdict_x["accepted"], "cold_verify_dir_accepts": cold["agree"] == cold["total"],
               "why": it.get("why")})

    d_d = out / "D_two_proofs_one_batch"
    shutil.copytree(d_h, d_d)
    man = json.loads((d_d / "manifest.json").read_text())
    for suf in ("stmt", "lgto", "coins"):
        shutil.copy(d_d / f"batch_live_0.{suf}", d_d / f"batch_live_1.{suf}")
    e2 = dict(man["files"][0], proof="batch_live_1.lgto", stmt="batch_live_1.stmt", coins_file="batch_live_1.coins")
    man["files"].append(e2)
    (d_d / "manifest.json").write_text(json.dumps(man, indent=1))
    vs = run_mod.verify_session(d_d, [store / sid], args.device, store)
    check("D_two_proofs_one_batch", [i["authenticated"] for i in vs["items"]] == [True, False] and vs["job_claimed_log2"] is None,
          {"auth": [i["authenticated"] for i in vs["items"]], "why": vs["items"][1].get("why")})

    s2 = _session(port, R, "E_abort_retry")
    st2 = s2.stream(0)
    st2.begin(stmt)
    st2.challenge(b"zc/r", os.urandom(32), 32)
    s2.ctrl.close()
    for _ in range(100):
        if (store / s2.id / "session.json").exists():
            break
        time.sleep(0.1)
    vs = run_mod.verify_session(d_h, [store / sid], args.device, store)
    c = vs["items"][0].get("claim", {})
    vs0 = run_mod.verify_session(d_h, [store / sid], args.device, None)
    c0 = vs0["items"][0].get("claim", {})
    # every session for this STMT counts: honest, A, B and the dropped one
    check("E_abort_retry", c.get("attempts") == 4 and honest_claim is not None and abs(c["claimed_log2"] - (honest_claim + 2)) < 1e-9
          and vs0["items"][0]["authenticated"] and c0.get("claimed_log2") is None and vs0["job_claimed_log2"] is None,
          {"with_root": {k: c.get(k) for k in ("attempts", "per_session_log2", "claimed_log2")},
           "without_root": {k: c0.get(k) for k in ("claimed_log2", "basis")}})

    srv.stop.set()
    (out / "live_session_gpu_test.json").write_text(json.dumps({"relation": pv.rel.name, "zk": args.zk, "rounds": R,
                                                                "commit": run_mod._git_commit(), "failures": fails,
                                                                "results": results}, indent=1, default=str))
    print(f"live_session_gpu_test {pv.rel.name}{' --zk' if args.zk else ''}: {len(results) - fails}/{len(results)} ok, "
          f"{fails} failures ({out})", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
