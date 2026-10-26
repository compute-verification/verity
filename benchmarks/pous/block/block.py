"""PoUS's block run (Demo B, MEM-71): the prover holds a P2 v4 store in MEM-69's layout (`store.json`, `seg/<s>.c`,
`rows/<s>.rows`), the memory challenger (`challenger_party`) runs where its command starts it, on its own host in the
demo (vy-auditor-2, reached over direct TCP), and each signed session is judged by the run's check at the store's own
point (`run.check_at`), its public output `period.certify`'s: at C*'s 68,026,368 blocks, `p2.wide_certificate`
(`periodHeldAt`), so an accepted honest session reports `S_cert` = 1,060,008,348,186 bits, certified, at 1/100 + 2^-128.

    python benchmarks/pous/block/block.py --store DIR [OPTIONS]
    python benchmarks/pous/block/block.py --make-fixture DIR [--blocks-per-segment B] [--segments S]

    --memory hbm|host|disk  where the prover holds C*: the GPU's memory through the CUDA driver (`wipe_store.Cuda`, no
                            toolkit), host memory (`wipe_store.HostMemory`, a CPU node's stand-in), or the segment files
                            mmapped (default hbm)
    --phase all|sessions|check|smoke
                            all: the sessions, then their checks (C* released first); sessions: take them and stop;
                            check: judge the reports in --sessions; smoke: hold C*, check 64 random blocks against rows/
                            and print the certificate at the store's point, with no session (default all)
    --honest N --drop N     the sessions: N honest, then N of a prover answering zeros for the 10.2% of blocks whose
                            tag's first byte is below 26 (`dropped`), which the check must refuse for wrong answers
                            (defaults 1 and 1)
    --challenger HOST --challenger-address H:P --challenger-command 'ARGV'
                            the memory challenger on a host of its own (all three; INF-258's gate on vy-auditor-2, which
                            starts `challenger_party --device DIR --listen H:P` there); without them a process here, its
                            key in --device (made there if absent, minutes at S1's LMS), the premise unmet
    --trusted HEX           the verifier's memory-challenger key (repeatable): from the challenger's host, never from
                            its report; without it the check trusts the key the report names, and says so
    --node NAME             the prover's node, which the session binds (default: hostname)
    --cpus LO-HI            pin the prover (and a local challenger) to these cores
    --out DIR               outputs (default $OUT, else ./out); --sessions DIR: the reports --phase check judges
                            (default OUT/sessions)

The setup is cited by reference: `store.json`'s commitment, salt and root (MEM-69's encode and its `verify --decode`), not
tied in the run's case, since the run protocol's tie decodes all of C* (note:20261026T0035Z-report-pous-block, Q2). Each
check rebuilds the table's root from rows/ (step 2 of `run.check`), so it costs minutes and tens of GB at C*'s size.

Writes under OUT: summary.json (the point, the certificate, the placement, each session's record and answer), row.json
(the block record's pous row, from the first honest session), sessions/<case>-<j>.report.json (the challenger side's
report, the prover's clock around it as t0 and t1) and sessions/<case>-<j>.json (its answer and record). Prints one
`VERDICT` line a session and `PASS`/`FAIL` a check; exits 0 when every honest session is accepted with none late and
every drop session refused for wrong answers with none late (smoke: the spots match and the point is certified).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import secrets
import shlex
import shutil
import socket
import statistics
import sys
import tempfile
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from verity.core.protocols.pous import challenger_party as CP
from verity.core.protocols.pous import memory_challenger as mc
from verity.core.protocols.pous import period
from verity.core.protocols.pous import run as R
from verity.core.protocols.pous import wipe_store as WS
from verity.core.protocols.pous.schemes import p2 as P2
from verity.core.protocols.pous.verifier import DIGEST_BYTES
from verity.ml.commitments.identity import canonical_json_bytes

DROP = "drop10"
DROP_TAG, DROP_BELOW = b"verity/mem80/block-drop10", 26
"""A drop session's prover answers zeros for block `i` when `sha256(DROP_TAG ‖ u64be i)[0] < DROP_BELOW`: 26/256 = 10.2%."""
SPOTS = 64
WRONG = "answers are not their blocks of C"


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{time.time_ns() % 10**9 // 1000:06d}Z"


def dropped(i: int) -> bool:
    return hashlib.sha256(DROP_TAG + i.to_bytes(8, "big")).digest()[0] < DROP_BELOW


def load(store: Path) -> tuple[R.Point, R.Setup, dict[str, Any]]:
    """The store's point (its manifest at PoUS's `k` and `Δ + RTT`) and what its setup published."""
    doc = json.loads((store / "store.json").read_text())
    m = doc["manifest"]
    q = P2.P2Params(m["w"], m["c0"], m["blocks_per_segment"], m["segments"])
    if q.manifest() != m:
        raise ValueError(f"{store}/store.json's manifest {m} is not P2 v4's {q.manifest()}")
    published = R.Setup(q.blocks, bytes.fromhex(doc["commitment"]), bytes.fromhex(doc["root"]),
                        bytes.fromhex(doc["salt"]))
    return R.Point(q, period.K, period.LIMIT_NS), published, doc


def table(store: Path, q: P2.P2Params) -> bytes:
    """The memory challenger's table of C*, every block's 64-byte digest in block order, from rows/."""
    rows = b"".join((store / "rows" / f"{s:05d}.rows").read_bytes() for s in range(q.segments))
    if len(rows) != DIGEST_BYTES * q.blocks:
        raise ValueError(f"rows/ holds {len(rows)} bytes, not {DIGEST_BYTES} for each of {q.blocks} blocks")
    return rows


def _read_segment(path: Path, into: memoryview, n: int) -> None:
    with open(path, "rb", buffering=0) as f:
        got = 0
        while got < n:
            k = f.readinto(into[got:n])
            if not k:
                raise ValueError(f"{path} is {got} bytes, not {n}")
            got += k
        if f.read(1):
            raise ValueError(f"{path} is past {n} bytes")


class Held:
    """C* as the prover holds it: `respond(i)` is block `i`'s bytes; `info` says where; `close` releases it."""

    def __init__(self, store: Path, q: P2.P2Params, memory: str) -> None:
        self.q, self.memory = q, memory
        nb, per = q.block_bytes, q.blocks_per_segment
        self.seg_bytes = per * nb
        t = time.monotonic()
        if memory == "disk":
            self.maps = []
            for s in range(q.segments):
                with open(store / "seg" / f"{s:05d}.c", "rb") as f:
                    mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
                if len(mm) != self.seg_bytes:
                    raise ValueError(f"segment {s} is {len(mm)} bytes, not {self.seg_bytes}")
                self.maps.append(mm)
            self.info: dict[str, Any] = {"backend": "disk", "host": socket.gethostname(), "tier": "disk"}
        elif memory in ("hbm", "host"):
            self.mem: WS.Memory = WS.Cuda() if memory == "hbm" else WS.HostMemory()
            free, total = self.mem.mem_info()
            if q.blocks * nb > free:
                raise ValueError(f"C* is {q.blocks * nb} bytes; the {self.mem.tier} has {free} of {total} free")
            self.base = self.mem.alloc(q.blocks * nb)
            stage_addr, stage = self.mem.pinned(self.seg_bytes)
            try:
                for s in range(q.segments):
                    _read_segment(store / "seg" / f"{s:05d}.c", stage, self.seg_bytes)
                    self.mem.htod(self.base + s * self.seg_bytes, stage_addr, self.seg_bytes)
            finally:
                self.mem.free_pinned(stage_addr)
            self.answer_addr, self.answer = self.mem.pinned(nb)
            self.info = self.mem.info() | {"bytes": q.blocks * nb, "free_before": free}
        else:
            raise ValueError(f"--memory {memory}: hbm, host or disk")
        self.load_s = round(time.monotonic() - t, 1)

    def respond(self, i: int) -> bytes:
        q, nb = self.q, self.q.block_bytes
        if not 0 <= i < q.blocks:
            raise ValueError(f"index {i} is outside C*'s {q.blocks} blocks")
        if self.memory == "disk":
            s, v = divmod(i, q.blocks_per_segment)
            return self.maps[s][nb * v:nb * (v + 1)]
        self.mem.dtoh(self.answer_addr, self.base + nb * i, nb)
        return bytes(self.answer)

    def close(self) -> None:
        if self.memory == "disk":
            for mm in self.maps:
                mm.close()
            return
        self.mem.free_pinned(self.answer_addr)
        self.mem.free(self.base)
        self.mem.close()


def spot(held: Held, digests: bytes, n: int = SPOTS) -> list[int]:
    """`n` random blocks held against the table; the ones that differ."""
    picks = [secrets.randbelow(held.q.blocks) for _ in range(n)]
    return [i for i in picks if mc.block_digest(held.respond(i)) != digests[64 * i:64 * (i + 1)]]


def prover(respond: Callable[[int], bytes], drop: bool, block_bytes: int) -> Callable[[int], bytes]:
    zero = bytes(block_bytes)
    return (lambda i: zero if dropped(i) else respond(i)) if drop else respond


def party_of(a: argparse.Namespace, node: str, scratch: Path) -> CP.Party:
    if a.challenger:
        h, _, p = a.challenger_address.rpartition(":")
        return CP.Party(a.challenger, tuple(shlex.split(a.challenger_command)), (h, int(p)))
    return CP.Party.local(node, Path(a.device) if a.device else scratch / "pous-challenger")


def take(party: CP.Party, request: Mapping[str, Any], held: Held, cases: Sequence[tuple[str, int]],
         out: Path) -> list[tuple[str, int, dict[str, Any]]]:
    """Each case's sessions with the challenger's side `party` starts; each report written as it comes."""
    taken = []
    for name, n in cases:
        for j in range(n):
            t, t0, load1 = time.monotonic(), utc(), round(os.getloadavg()[0], 1)
            got = CP.session(party, request, prover(held.respond, name == DROP, held.q.block_bytes))
            got |= {"t0": t0, "t1": utc(), "clock": "the prover's, around the whole session", "load1": load1,
                    "wall_s": round(time.monotonic() - t, 2)}
            (out / "sessions" / f"{name}-{j}.report.json").write_text(json.dumps(got, sort_keys=True))
            s = R.Session.from_json(got["session"])
            print(f"[{utc()}] session {name}-{j}: ended {s.signed.ended}, {len(s.signed.rounds)} rounds, "
                  f"{got['wall_s']} s", flush=True)
            taken.append((name, j, got))
    return taken


def judge(point: R.Point, published: R.Setup, digests: bytes, got: Mapping[str, Any], trusted: Collection[bytes],
          schemes: Mapping[str, mc.Verify] = mc.SCHEMES) -> tuple[dict[str, Any], dict[str, Any]]:
    """The run's check of one reported session, against `trusted` (else the key the report names): its answer, with
    `period.certify`'s public output when accepted, and its record (timing, the rounds late)."""
    sid, key = bytes.fromhex(got["session_id"]), bytes.fromhex(got["public_key"])
    s = R.Session.from_json(got["session"])
    keys = tuple(trusted) or (key,)
    commitment = hashlib.sha512(canonical_json_bytes(published.to_json())).hexdigest()
    t = time.monotonic()
    try:
        R.check_at(point, schemes, keys, published, digests, s, sid)
        a = {"commitment": commitment, "profile": period.profile(point) | {"setup": "cited"}, "protocol": period.PROTOCOL,
             "public": period.certify(point), "reason": None, "verdict": "accept"}
    except R.Refused as e:
        a = {"commitment": commitment, "profile": None, "protocol": period.PROTOCOL, "public": None, "reason": str(e),
             "verdict": "reject"}
    el = [r.elapsed_ns for r in s.signed.rounds]
    rec = {"verdict": a["verdict"], "reason": a["reason"], "public": a["public"], "ended": s.signed.ended,
           "rounds": len(el), "late": sum(e > point.limit_ns for e in el),
           "late_rounds": [k for k, e in enumerate(el) if e > point.limit_ns],
           "median_us": round(statistics.median(el) / 1e3, 1) if el else None,
           "p99_us": round(sorted(el)[min(len(el) - 1, int(0.99 * len(el)))] / 1e3, 1) if el else None,
           "max_us": round(max(el) / 1e3, 1) if el else None, "check_s": round(time.monotonic() - t, 1),
           "trusted": "pinned" if trusted else "reported", "public_key": got["public_key"],
           "session_id": got["session_id"], "host": got.get("host"), "t0": got.get("t0"), "t1": got.get("t1"),
           "wall_s": got.get("wall_s"), "load1": got.get("load1")}
    if s.signed.rounds:
        rec["dropped_drawn"] = sum(dropped(r.index) for r in s.signed.rounds)
    return a, rec


def verdict_line(name: str, a: Mapping[str, Any], rec: Mapping[str, Any]) -> str:
    pub = a["public"] or {"certified": False, "eps": "0", "s_bits": 0}
    return (f"VERDICT {name}: {a['verdict']} S_cert={pub['s_bits']} certified={str(pub['certified']).lower()} "
            f"eps={float(Fraction(pub['eps'])):.10g} ({pub['eps']}) rounds={rec['rounds']} late={rec['late']} "
            f"median_us={rec['median_us']} "
            f"max_us={rec['max_us']} trusted={rec['trusted']}" + (f" reason={a['reason']!r}" if a["reason"] else ""))


def checks_of(sessions: Mapping[str, list[dict[str, Any]]]) -> dict[str, bool]:
    honest, drop = sessions.get("honest", []), sessions.get(DROP, [])
    out = {}
    if honest:
        out[f"honest: {len(honest)} accepted, none late, S_cert certified"] = all(
            r["verdict"] == "accept" and r["late"] == 0 and r["ended"] == mc.COMPLETE and r["public"]["certified"]
            for r in honest)
    if drop:
        out[f"{DROP}: {len(drop)} refused for wrong answers, none late"] = all(
            r["verdict"] == "reject" and r["late"] == 0 and WRONG in (r["reason"] or "") for r in drop)
    return out


def make_fixture(d: Path, q: P2.P2Params, *, store: bytes, coins: bytes, workers: int = 1) -> R.Setup:
    """A store in MEM-69's layout at `q` (`period.setup_store` on `store`, padded): store.json, seg/, rows/."""
    held = period.setup_store(store, coins, R.Point(q, period.K, period.LIMIT_NS), workers=workers)
    (d / "seg").mkdir(parents=True)
    (d / "rows").mkdir()
    n = q.blocks_per_segment
    for s in range(q.segments):
        (d / "seg" / f"{s:05d}.c").write_bytes(b"".join(held.blocks[n * s:n * (s + 1)]))
        (d / "rows" / f"{s:05d}.rows").write_bytes(held.digests[64 * n * s:64 * n * (s + 1)])
    (d / "store.json").write_text(json.dumps({"manifest": q.manifest(), **held.published.to_json()}, sort_keys=True))
    return held.published


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--store")
    ap.add_argument("--make-fixture", metavar="DIR")
    ap.add_argument("--blocks-per-segment", type=int, default=128)
    ap.add_argument("--segments", type=int, default=2)
    ap.add_argument("--memory", default="hbm", choices=("hbm", "host", "disk"))
    ap.add_argument("--phase", default="all", choices=("all", "sessions", "check", "smoke"))
    ap.add_argument("--honest", type=int, default=1)
    ap.add_argument("--drop", type=int, default=1)
    ap.add_argument("--challenger")
    ap.add_argument("--challenger-address")
    ap.add_argument("--challenger-command")
    ap.add_argument("--device")
    ap.add_argument("--trusted", action="append", default=[])
    ap.add_argument("--node", default=socket.gethostname())
    ap.add_argument("--cpus")
    ap.add_argument("--out", default=os.environ.get("OUT") or "out")
    ap.add_argument("--sessions")
    a = ap.parse_args(argv)
    workers = min(period.SETUP_WORKERS, len(os.sched_getaffinity(0)))
    if a.make_fixture:
        q = P2.P2Params(*P2.POINT[:2], a.blocks_per_segment, a.segments)
        t = time.monotonic()
        pub = make_fixture(Path(a.make_fixture), q, store=os.urandom(q.w_bytes), coins=secrets.token_bytes(64),
                           workers=workers)
        print(json.dumps({"fixture": a.make_fixture, "manifest": q.manifest(), "root": pub.root.hex(),
                          "setup_s": round(time.monotonic() - t, 1), "workers": workers}), flush=True)
        return 0
    if not a.store:
        ap.error("--store DIR (or --make-fixture DIR)")
    wired = [a.challenger, a.challenger_address, a.challenger_command]
    if any(wired) and not all(wired):
        ap.error("--challenger, --challenger-address and --challenger-command go together")
    store, out = Path(a.store), Path(a.out)
    (out / "sessions").mkdir(parents=True, exist_ok=True)
    point, published, doc = load(store)
    q = point.params
    trusted = [bytes.fromhex(k) for k in a.trusted]
    summary: dict[str, Any] = {
        "node": a.node, "memory": a.memory, "phase": a.phase, "store": str(store), "point": period.point_json(point),
        "blocks": q.blocks, "c_bytes": q.blocks * q.block_bytes, "certificate": period.certify(point),
        "profile": period.profile(point),
        "setup": {"by": "reference", "commitment": doc["commitment"], "salt": doc["salt"], "root": doc["root"],
                  "tie": doc.get("tie"), "why": "the run protocol's tie decodes all of C* from its case; cited, not tied"},
        "trusted": [k.hex() for k in trusted] or "the key each report names", "sessions": {}}
    print(f"[{utc()}] point {json.dumps(summary['point'])} blocks {q.blocks} certificate "
          f"{json.dumps(summary['certificate'])}", flush=True)
    scratch = Path(tempfile.mkdtemp(prefix="pous-block-"))
    try:
        digests = b""
        if a.phase != "check":
            t = time.monotonic()
            digests = table(store, q)
            summary["table_read_s"] = round(time.monotonic() - t, 1)
            party = None
            if a.phase != "smoke":
                party = party_of(a, a.node, scratch)
                summary["placement"] = CP.placement(a.node, party.host)
                summary["challenger"] = {"host": party.host, "address": None if party.address is None else
                                         f"{party.address[0]}:{party.address[1]}"}
                if party.address is None:
                    t = time.monotonic()
                    period.challenger(party.command[party.command.index("--device") + 1], workers=workers)
                    summary["challenger_keygen_s"] = round(time.monotonic() - t, 1)
            if a.cpus:
                lo, _, hi = a.cpus.partition("-")
                os.sched_setaffinity(0, range(int(lo), int(hi or lo) + 1))
            summary["timed_cpus"] = sorted(os.sched_getaffinity(0))
            held = Held(store, q, a.memory)
            summary |= {"held": held.info, "load_s": held.load_s}
            try:
                bad = spot(held, digests)
                summary["spots"] = {"blocks": SPOTS, "differ": bad}
                print(f"[{utc()}] C* held in {a.memory} in {held.load_s} s; {SPOTS} random blocks against rows/: "
                      f"{len(bad)} differ", flush=True)
                if bad:
                    raise SystemExit(f"C* in {a.memory} differs from rows/ at {bad}")
                if a.phase == "smoke":
                    ok = summary["certificate"]["certified"]
                    print(f"SMOKE {'PASS' if ok else 'FAIL'}: certified={str(ok).lower()} "
                          f"S_cert={summary['certificate']['s_bits']} "
                          f"eps={float(Fraction(summary['certificate']['eps'])):.10g} "
                          f"gaps={summary['profile']['gaps']} load_s={held.load_s} "
                          f"table_read_s={summary['table_read_s']}", flush=True)
                    (out / "summary.json").write_text(json.dumps(summary, sort_keys=True, indent=1))
                    return 0 if ok else 1
                request = CP.request_of(a.node, point, published.root)
                sessions = take(party, request, held, (("honest", a.honest), (DROP, a.drop)), out)
            finally:
                held.close()
        if a.phase == "sessions":
            (out / "summary.json").write_text(json.dumps(summary, sort_keys=True, indent=1))
            print(f"[{utc()}] sessions taken; judge them with --phase check --sessions {out / 'sessions'}", flush=True)
            return 0
        if a.phase == "check":
            src = Path(a.sessions) if a.sessions else out / "sessions"
            sessions = []
            for p in sorted(src.glob("*.report.json")):
                name, j = p.name.split(".")[0].rsplit("-", 1)
                sessions.append((name, int(j), json.loads(p.read_text())))
            if not sessions:
                raise SystemExit(f"no session reports in {src}")
            summary["sessions_from"] = str(src)
            if (src.parent / "summary.json").exists():
                taken = json.loads((src.parent / "summary.json").read_text())
                summary |= {k: taken[k] for k in ("placement", "challenger", "held", "load_s", "spots", "timed_cpus",
                                                  "challenger_keygen_s") if k in taken}
            t = time.monotonic()
            digests = table(store, q)
            summary["table_read_s"] = round(time.monotonic() - t, 1)
        first = None
        for name, j, got in sessions:
            ans, rec = judge(point, published, digests, got, trusted)
            summary["sessions"].setdefault(name, []).append(rec | {"answer": ans})
            (out / "sessions" / f"{name}-{j}.json").write_text(json.dumps({"answer": ans, "record": rec}, sort_keys=True))
            print(verdict_line(f"{name}-{j}", ans, rec), flush=True)
            if name == "honest" and first is None:
                first = ans
        checks = checks_of(summary["sessions"])
        summary["checks"] = checks
        if first is not None:
            pub = first["public"] or {"certified": False, "eps": "0", "s_bits": 0}
            (out / "row.json").write_text(json.dumps({
                "answer": first, "owner": "memory-accounting", "verdict": first["verdict"],
                "quantity": {"s_bits": pub["s_bits"], "certified": pub["certified"], "eps": pub["eps"]},
                "profile": {"gaps": [*summary["profile"]["gaps"], "the setup is cited by reference, not tied in the run's case"],
                            "memory": a.memory, "setup": "cited", "placement": summary.get("placement")}},
                sort_keys=True, indent=1))
        (out / "summary.json").write_text(json.dumps(summary, sort_keys=True, indent=1))
        for k, v in checks.items():
            print(("PASS " if v else "FAIL ") + k, flush=True)
        return 0 if checks and all(checks.values()) else 1
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
