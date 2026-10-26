"""The memory challenger's side of the band-fill wipe, on a machine the prover doesn't control (`wire.py` has the control
channel, `prover.py` the other side).

    python -m band_wipe.challenger run --out DIR (--peer NAME [--peers /etc/vy/peers] | --direct HOST)
        [--ctrl-port 47610] [--resp-port 47611] [--modes honest,cheat] [--sessions 1] [--cpu C] [--ssh-cpu C]
        [--signer lms [--lms-key DIR] [--session-log FILE] | --signer test] [--pings 2000] [--workers N] [--no-verify]
    python -m band_wipe.challenger verify --record DIR --out DIR [--workers N]

**What it runs under.** `Pous.Guarantees.SecureErasureMeetsFamily` (`ERASE`): the band `chainBandSeg N (2^9) d` keyed by
the memory challenger's fresh 192-bit salt meets `EraseMeets` at `f = 1/19`, `(D, Q) = (5, 2^20)`, 2152 spot reads and
`ε = 2^-128`; a prover cannot keep `M/19` bytes of foreign content through all the reads.  `D = 5` rounds of `Π₂` per read
is the deadline `Δ + RTT < (D + 1)·call_ns`, so the limit is `6 × 241,500 − 1 = 1,448,999` ns (`limit_ns`), and
`memory_challenger.covering_certificate` must cover `k = 2152` within it before any round.

**`run`.** With `--direct HOST`, it connects to the prover listening on HOST (`prover.py --listen HOST:PORT`, a path the
prover's network admits from this machine); with `--peer`, through an SSH tunnel to that machine of `--peers` (`ssh -L`, the
control port and the responder's port forwarded to the prover's 127.0.0.1, `SSH`), too slow for a session meant to pass.
It retries until the prover's hello arrives (`--wait-s`).  Each wipe of `--modes` (`honest`, `cheat`):

1. a fresh salt (`os.urandom(24)`), and `--sessions` memory challengers with their coins drawn and committed and their
   session ids claimed in the session log, all before the salt is sent; each session's binding is `wire.binding`;
2. the salt to the prover, which fills and answers `filled`;
3. `--pings` round trips to the responder (recorded, not used: the limit is the guarantee's), then each session's
   `k` rounds after `WARMUP_PINGS` pings, signed and its coins opened; the first late round ends a session;
4. the signed sessions to `sessions-wipe{j}.json` (`live.write_sessions`) before any verdict runs.

`challenger.json` records the run; then, unless `--no-verify`, `verify` runs on it.

**`verify`.** For every round revealed, the expected `block_digest` of that block of `C`, recomputed from the salt by the
reference encoder (`wire.segment_digests`, a segment's blocks up to the highest one read, in `--workers` processes), never
taken from the prover.  Each session is judged by `memory_challenger.verdict` against the commitment published before its
wipe's fill, under `memory_challenger.SCHEMES` (the record's: S1's LMS alone) and under the signer's own mapping (a test
signer's, for `--signer test`).  `verdict.json` holds each session's verdict, its rounds' times and, for a session that
ended late, the round that did and whether the cheat had dropped its block.  Exit 0 when every honest session is accepted
(under the signer's mapping) and every cheat session rejected.
"""
from __future__ import annotations

import argparse
import gc
import json
import multiprocessing
import os
import socket
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from band_gpu.memory_challenger import (  # noqa: E402
    HEADER,
    OP_CHALLENGE,
    OP_PING,
    LiveResponder,
    calibrate,
    cgroup_cpu,
    quantiles,
    scheme_of,
)
from band_wipe.wire import SCHEME, Channel, binding, dropped, segment_digests  # noqa: E402
from memory_challenger.live import (  # noqa: E402
    WARMUP_PINGS,
    lms_signer,
    session_log,
    signer_schemes,
    test_signer,
    write_sessions,
)

ERASE = {"guarantee": "Pous.Guarantees.SecureErasureMeetsFamily",
         "statement": "Security/Models/Pous/Guarantees/SecureErasure.lean", "proof": "PousErase.secure_erasure_meets_family",
         "f": "1/19", "D": 5, "Q": 2**20, "k": 2152, "epsilon": "2^-128",
         "reads_as": "a prover cannot keep M/19 bytes of foreign content through all the reads (M its whole volatile "
                     "memory; memory outside the fill adds to what it can keep)",
         "model": "the ideal-primitive model: at the reference Feistel-SHAKE256 Π₂ a label is rebuilt from 1.53% of C in "
                  "481 Keccak-f (art:7617c8017e80), so this run demonstrates the bound against a prover that recomputes "
                  "whole blocks, not against that one"}
K = ERASE["k"]
STATE = Path.home() / ".local" / "state" / "verity-band-wipe"
SSH = ["ssh", "-tt", "-o", "ObscureKeystrokeTiming=no", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
       "-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=15", "-o", "Compression=no", "-o", "IPQoS=lowdelay",
       "-c", "aes128-gcm@openssh.com"]
"""The tunnel's ssh, holding a pty session (`TUNNEL_HOLD`) so that both ends set TCP_NODELAY: under `-N` Nagle holds a 64 KiB
answer's tail for a delayed ACK (p99.9 42 ms, r20261026-010824-7e2d), and a pty's keystroke-timing obfuscation sends every
20 ms (p50 20.5 ms, same run) unless turned off.  Between two uk-south2 nodes this tunnel still took p50 1.27 ms, p99.9
1.85 ms for a 64 KiB answer, 330 of 2,000 over `limit_ns` (r20261026-010951-afe5), so it is a rehearsal's path; a session
meant to pass needs `--direct`."""
TUNNEL_HOLD = "exec sleep infinity"


def limit_ns(params: Any) -> int:
    """`Δ + RTT` under `(D + 1)` calls of `params.call_ns`: the longest an answer may take and leave `D` rounds."""
    return (ERASE["D"] + 1) * params.call_ns - 1


def run(ch: Channel, hello: dict[str, Any], resp_host: str, resp_port: int | None, out: Path, *, modes: list[str],
        sessions: int, signer: Any, log: Any, k: int = K, pings: int = 2000,
        clock: Callable[[], int] = time.perf_counter_ns, base: Any = None, strict: bool = True) -> dict[str, Any]:
    """The wipes of `modes` over the control channel `ch` (its hello read), each with `sessions` signed sessions against the
    responder at `resp_host:resp_port` (the port the prover names when `resp_port` is None).  Returns the run's record."""
    from verity.core.protocols.pous import band
    from verity.core.protocols.pous import memory_challenger as mc

    if hello.get("op") != "hello" or hello.get("scheme") != SCHEME:
        raise ValueError(f"not a band-fill prover's hello: {hello}")
    params = (base or band.SECURE_FIRST).with_segments(int(hello["segments"]))
    if hello["params"] != params.manifest():
        raise ValueError(f"the prover's geometry {hello['params']} is not {params.manifest()}")
    scheme = scheme_of(SCHEME, params)
    limit = limit_ns(params)
    try:
        cert = mc.covering_certificate(scheme, k, limit, experimental=True)
        covered = {"theorem": cert.theorem, "k": cert.k, "rounds": cert.rounds, "call_ns": cert.call_ns}
    except ValueError as e:
        if strict:
            raise
        covered = {"refused": str(e)}
    nb = params.block_bytes
    resolution = None if clock is time.perf_counter_ns else 1
    doc: dict[str, Any] = {"scheme": SCHEME, "params": params.manifest(), "segments": params.segments, "k": k,
                           "limit_ns": limit, "erase": ERASE, "certificate": covered, "hello": hello,
                           "signer": {"scheme": signer.scheme, "public_key": signer.public_key.hex(),
                                      "test": signer.scheme not in mc.SCHEMES, "session_log": str(log.path)},
                           "challenger_host": {"hostname": socket.gethostname(), "cpus": sorted(os.sched_getaffinity(0)),
                                               "cgroup_cpu_before": cgroup_cpu()},
                           "wipes": []}
    for j, mode in enumerate(modes):
        salt = os.urandom(24)
        bind = binding(SCHEME, params.manifest(), salt)
        challengers = [mc.MemoryChallenger(mc.CoinTree.draw(os.urandom(32), os.urandom(32), params.blocks, k), signer,
                                           log=log, limit_ns=limit, block_bytes=nb, binding=bind, clock=clock,
                                           resolution_ns=resolution) for _ in range(sessions)]
        wipe: dict[str, Any] = {"wipe": j, "mode": mode, "salt": salt.hex(), "binding": bind.hex(),
                                "commitments": [c.commitment.to_json() for c in challengers]}
        t0 = time.perf_counter_ns()
        ch.send({"op": "fill", "wipe": j, "salt": salt.hex(), "cheat": mode == "cheat"})
        filled = ch.recv()
        wipe["salt_to_filled_s"] = (time.perf_counter_ns() - t0) / 1e9
        wipe["prover_filled"] = filled
        if filled.get("op") != "filled":
            wipe["refused"] = filled
            doc["wipes"].append(wipe)
            continue
        r = LiveResponder(resp_host, resp_port or int(filled["port"]), nb, clock)
        done: list[dict[str, Any]] = []
        before = after = []
        gc.collect()
        gc.freeze()
        gc.disable()
        try:
            before = calibrate(r, pings)
            for c in challengers:
                for _ in range(WARMUP_PINGS):
                    r.roundtrip(OP_PING, 0)
                reports = []
                while not c.done:
                    i = c.challenge()
                    r.roundtrip(OP_CHALLENGE, i)
                    c.receive(r.view[HEADER:])
                    reports.append((r.device_ns, r.server_ns))
                t1 = time.perf_counter_ns()
                signed = c.close()
                done.append({"commitment": c.commitment, "signed": signed, "openings": c.openings(), "reports": reports,
                             "sign_ns": time.perf_counter_ns() - t1})
            after = calibrate(r, pings)
        finally:
            r.close()
            gc.enable()
            gc.unfreeze()
            wipe["sessions_file"] = f"sessions-wipe{j}.json"
            write_sessions(out / wipe["sessions_file"], signer, done)
        ch.send({"op": "done"})
        wipe["prover_report"] = ch.recv()
        wipe["pings"] = {"before_s": quantiles(before), "after_s": quantiles(after),
                         "over_limit": sum(x * 1e9 > limit for x in before + after), "n": len(before) + len(after)}
        doc["wipes"].append(wipe)
    ch.send({"op": "bye"})
    doc["challenger_host"]["cgroup_cpu_after"] = cgroup_cpu()
    return doc


def expected_digests(params: Any, salt: bytes, indices: set[int], workers: int) -> dict[int, bytes]:
    """`block_digest` of each block of `indices`, from the reference encoder: each segment's blocks up to the highest one
    read, the longest segments first."""
    b = params.blocks_per_segment
    by: dict[int, set[int]] = defaultdict(set)
    for i in indices:
        by[i // b].add(i % b)
    jobs = sorted(((params, salt, s, sorted(vs)) for s, vs in by.items()), key=lambda job: -job[3][-1])
    out: dict[int, bytes] = {}
    if workers <= 1:
        results = map(segment_digests, jobs)
        pool = None
    else:
        pool = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"))
        results = pool.map(segment_digests, jobs, chunksize=1)
    try:
        for n, (s, got) in enumerate(results, 1):
            out.update({s * b + v: d for v, d in got.items()})
            if n % 50 == 0 or n == len(jobs):
                print(f"expected digests: {n} of {len(jobs)} segments", flush=True)
    finally:
        if pool is not None:
            pool.shutdown()
    return out


def verify(doc: dict[str, Any], record: Path, *, workers: int, base: Any = None) -> dict[str, Any]:
    """Every session of the run's record `doc` (its signed sessions in `record`) judged against `C`'s blocks recomputed
    from each wipe's salt (module docstring)."""
    from verity.core.protocols.pous import band
    from verity.core.protocols.pous import memory_challenger as mc

    params = (base or band.SECURE_FIRST).with_segments(int(doc["segments"]))
    if doc["params"] != params.manifest():
        raise ValueError(f"the record's geometry {doc['params']} is not {params.manifest()}")
    limit = int(doc["limit_ns"])
    out: dict[str, Any] = {"limit_ns": limit, "k": doc["k"], "erase": doc["erase"], "wipes": []}
    for wipe in doc["wipes"]:
        row: dict[str, Any] = {"wipe": wipe["wipe"], "mode": wipe["mode"], "sessions": []}
        out["wipes"].append(row)
        if "refused" in wipe:
            row["refused"] = wipe["refused"]
            continue
        salt = bytes.fromhex(wipe["salt"])
        sdoc = json.loads((record / wipe["sessions_file"]).read_text())
        trusted = {bytes.fromhex(sdoc["signer"]["public_key"])}
        under_signer = signer_schemes(sdoc["signer"]["scheme"])
        published = [mc.SessionCommitment.from_json(c) for c in wipe["commitments"]]
        sessions = [(mc.SignedSession.from_json(s["signed"]), [mc.CoinOpening.from_json(o) for o in s["openings"]],
                     s["reports"]) for s in sdoc["sessions"]]
        t0 = time.perf_counter_ns()
        digests = expected_digests(params, salt, {r.index for signed, _, _ in sessions for r in signed.rounds
                                                  if 0 <= r.index < params.blocks}, workers)
        row["expected_digests"] = {"blocks": len(digests), "s": (time.perf_counter_ns() - t0) / 1e9, "workers": workers}
        expected = lambda i, d=digests: d.get(i, b"")  # noqa: E731
        for n, (signed, openings, reports) in enumerate(sessions):
            commitment = published[n] if n < len(published) else signed.commitment
            v = mc.verdict(signed, commitment, openings, expected, trusted=trusted)
            vs = mc.verdict(signed, commitment, openings, expected, trusted=trusted, schemes=under_signer)
            rounds = signed.rounds
            late = [r for r in rounds if not 0 <= r.elapsed_ns <= limit]
            s_row = {"accepted": v.accepted, "accepted_under_signer": vs.accepted, "reasons": list(v.reasons),
                     "reasons_under_signer": list(vs.reasons), "ended": signed.ended, "rounds": len(rounds),
                     "late": vs.late, "wrong": vs.wrong, "bound_to_salt": commitment.binding.hex() == wipe["binding"],
                     "elapsed_s": quantiles([r.elapsed_ns / 1e9 for r in rounds]),
                     "margin_min_s": min((limit - r.elapsed_ns) / 1e9 for r in rounds) if rounds else None,
                     "server_s": quantiles([sv / 1e9 for _, sv in reports]),
                     "dropped_hits": sum(dropped(r.index) for r in rounds)}
            if late:
                r = late[0]
                s_row["first_late"] = {"j": r.j, "index": r.index, "elapsed_ns": r.elapsed_ns, "dropped_by_cheat": dropped(r.index),
                                       "correct": r.digest == expected(r.index), "server_ns": reports[r.j][1]}
            row["sessions"].append(s_row)
    out["passed"] = all(all(s["accepted_under_signer"] == (w["mode"] == "honest") for s in w["sessions"]) and w["sessions"]
                        for w in out["wipes"])
    return out


def free_port() -> int:
    with socket.create_server(("127.0.0.1", 0)) as s:
        return s.getsockname()[1]


def peer_ssh(peers: Path, peer: str) -> list[str]:
    """ssh's options and destination for `peer`, its line of `peers` (as `benchmarks/live_challenger/serve.py` reads it)."""
    import shlex

    for line in peers.read_text().splitlines():
        name, _, rest = line.partition(" ")
        if name == peer and rest.strip():
            return shlex.split(rest)
    raise ValueError(f"no peer {peer} in {peers}")


def connect(host: str, port: int, wait_s: float, alive: Callable[[], bool]) -> tuple[Channel, dict[str, Any]]:
    """The control channel and the prover's hello, retried until it arrives: through a tunnel, a connection the prover isn't
    listening for yet is accepted and then closed."""
    deadline = time.monotonic() + wait_s
    while True:
        sock = None
        try:
            sock = socket.create_connection((host, port), timeout=60)
            ch = Channel(sock)
            hello = ch.recv()
            sock.settimeout(None)
            return ch, hello
        except (OSError, ConnectionError, ValueError):
            if sock is not None:
                sock.close()
            if not alive():
                raise RuntimeError("the tunnel exited before the prover answered") from None
            if time.monotonic() > deadline:
                raise
            time.sleep(2)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, required=True)
    where = r.add_mutually_exclusive_group(required=True)
    where.add_argument("--peer", help="the prover's machine, a line of --peers")
    where.add_argument("--direct", help="the prover's host, reached without a tunnel")
    r.add_argument("--peers", type=Path, default=Path("/etc/vy/peers"))
    r.add_argument("--ctrl-port", type=int, default=47610)
    r.add_argument("--resp-port", type=int, default=47611)
    r.add_argument("--modes", default="honest,cheat")
    r.add_argument("--sessions", type=int, default=1, help="signed sessions per wipe")
    r.add_argument("--k", type=int, default=K)
    r.add_argument("--pings", type=int, default=2000)
    r.add_argument("--cpu", type=int, default=None, help="the core the memory challenger runs on")
    r.add_argument("--ssh-cpu", type=int, default=None, help="the core the tunnel's ssh runs on")
    r.add_argument("--signer", default="lms", choices=["lms", "test"])
    r.add_argument("--lms-key", type=Path, default=STATE / "lms")
    r.add_argument("--session-log", type=Path, default=STATE / "sessions")
    r.add_argument("--keygen-workers", type=int, default=len(os.sched_getaffinity(0)))
    r.add_argument("--wait-s", type=float, default=3600.0, help="how long to wait for the prover's hello")
    r.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0)))
    r.add_argument("--no-verify", action="store_true")
    v = sub.add_parser("verify")
    v.add_argument("--record", type=Path, required=True, help="a run's --out: challenger.json and its sessions files")
    v.add_argument("--out", type=Path, required=True)
    v.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0)))
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "verify":
        result = verify(json.loads((a.record / "challenger.json").read_text()), a.record, workers=a.workers)
        (a.out / "verdict.json").write_text(json.dumps(result, indent=1) + "\n")
        return 0 if result["passed"] else 1

    tunnel = None
    if a.peer:
        lc, lr = free_port(), free_port()
        fwd = ["-L", f"127.0.0.1:{lc}:127.0.0.1:{a.ctrl_port}", "-L", f"127.0.0.1:{lr}:127.0.0.1:{a.resp_port}"]
        cmd = [*(["taskset", "-c", str(a.ssh_cpu)] if a.ssh_cpu is not None else []), *SSH, *fwd, *peer_ssh(a.peers, a.peer),
               TUNNEL_HOLD]
        tunnel = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=open(a.out / "tunnel.log", "w"),
                                  stderr=subprocess.STDOUT)
        host, ctrl, resp = "127.0.0.1", lc, lr
    else:
        host, ctrl, resp = a.direct, a.ctrl_port, None
    allowed = os.sched_getaffinity(0)
    scratch = tempfile.mkdtemp(prefix="band-wipe-") if a.signer == "test" else None
    try:
        if a.signer == "lms":
            new_key = not (a.lms_key / "seed").exists()
            sig = lms_signer(a.lms_key, a.keygen_workers)
            log = session_log(a.session_log, new_key)
        else:
            sessions = a.sessions * len(a.modes.split(","))
            sig = test_signer().TestSigner(os.urandom(32), height=max(1, (sessions - 1).bit_length()), directory=scratch)
            log = session_log(Path(scratch) / "sessions", True)
        ch, hello = connect(host, ctrl, a.wait_s, lambda: tunnel is None or tunnel.poll() is None)
        if a.cpu is not None:
            os.sched_setaffinity(0, {a.cpu})
        doc = run(ch, hello, host, resp, a.out, modes=a.modes.split(","), sessions=a.sessions, signer=sig, log=log, k=a.k,
                  pings=a.pings)
        doc["tunnel"] = {"peer": a.peer, "ssh": SSH, "ssh_cpu": a.ssh_cpu} if a.peer else {"direct": a.direct}
        doc["challenger_host"]["cpu"] = a.cpu
        ch.close()
        sig.state.close()
    finally:
        os.sched_setaffinity(0, allowed)
        if tunnel is not None:
            tunnel.terminate()
            tunnel.wait()
        if scratch is not None:
            import shutil

            shutil.rmtree(scratch, ignore_errors=True)
    (a.out / "challenger.json").write_text(json.dumps(doc, indent=1) + "\n")
    if a.no_verify:
        return 0
    result = verify(doc, a.out, workers=a.workers)
    (a.out / "verdict.json").write_text(json.dumps(result, indent=1) + "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
