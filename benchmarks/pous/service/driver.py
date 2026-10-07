"""PoUS end to end through the proof service: setup, the memory challenger's signed timed audit, the answers proved
through the service, the verifier of record, and a verdict-only output.

    python benchmarks/pous/service/driver.py --out DIR [--scheme NAME --bps 2 --segments 1] [--k 4]
        [--signer test | --signer lms --lms-key DIR [--keygen-workers N]] [--flock BIN --lean BIN [--no-zk]]
        [--from-audit SERVICE-INPUT.json]

1. **Setup** (CPU): `W` at random, its commitment, the verifier's salt, `C = Enc(W)` under `--scheme` (by default P2's
   conforming codec, `schemes.p2.P2().name`; any registered scheme at `--bps`/`--segments`), and C's hiding commitment
   (`spec.commit_c`) under a fresh secret salt key.
2. **The audit**: one session of `memory_challenger.MemoryChallenger` over `C` on a logical clock, its header binding
   C's root (`spec.session_binding`), signed by the PoUS tests' test signer or S1's LMS key (`--lms-key`, at
   `lms.DEPLOYED`), its coins opened after.  `--from-audit` takes a session the GPU harness ran instead
   (`p2_v1/harness.py`'s `service-input.json`: the signed session, its openings, C's root and the answered blocks with
   their salts and paths), and skips 1 and 2.
3. **The proof**: the answered blocks staged as `spec.unit`'s instances (`verity_flock.class_statement.stage_unit`), one
   a round, `block` under C's salts and `o0` under the zero salt; registered with a `service.Stream` over `spec.spec`,
   drawn (`All`), proved by `flock-circuit prove` against its own `serve`, verified by the Lean `flock-verify` (the
   verifier of record; without binaries a stub verdict, labelled, for the CPU tests), delivered by the session record's
   SHA-512 (the minimum the firewall's delivery carries), and decided with PoUS's claims check (`spec.claims_check`)
   as a second verdict of record.
4. **The controls**, each a stream of its own: `tampered-block` (round j's answer has one bit flipped: the session is
   signed over it, the proof is honest about what was answered, and its `block` row is not C's leaf) and
   `tampered-record` (the signed records edited after signing: one receipt time moved 1 ns earlier).  Both must be
   rejected.

`DIR/verdict.json` is the output, `{"accepted": ...}` and nothing else.  `DIR/summary.json` is the run's own record: the
stream's outcome symbols and records (public items only), each control's symbol, and the timings.  `DIR/witness/` keeps
the prover's side (the signed session, the staged files, the sessions) for inspection; none of it is an output.
Exit 0 when the honest audit is accepted and every control is rejected.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from service import spec as SP  # noqa: E402
from verity.experimental.protocols.pous import memory_challenger as mc  # noqa: E402
from verity.experimental.protocols.sampled_proofs import service as S  # noqa: E402

ROOT = HERE.parents[2]
TEST_SIGNER = ROOT / "verity" / "experimental" / "protocols" / "pous" / "tests" / "pous_test_signer.py"
DELTA = 2.0 ** -40
STEP_NS = 26_000
"""The logical clock's advance per answer: the responder's time on the GPU (`r20261006-061825-bba8`'s 26 µs)."""


def test_signer() -> Any:
    import importlib.util

    s = importlib.util.spec_from_file_location("pous_test_signer", TEST_SIGNER)
    mod = importlib.util.module_from_spec(s)
    s.loader.exec_module(mod)
    return mod


def signer_of(kind: str, lms_key: str | None, workers: int) -> tuple[Any, Any]:
    """The session's signer and the scheme mapping its sessions are judged under: S1's LMS key (`memory_challenger.SCHEMES`)
    or the test signer (its own mapping, which `memory_challenger.SCHEMES` never accepts)."""
    if kind == "lms":
        sys.path.insert(0, str(HERE.parent))
        from memory_challenger.live import lms_signer

        return lms_signer(lms_key, workers), mc.SCHEMES
    ts = test_signer()
    return ts.TestSigner(os.urandom(32), height=3), ts.SCHEMES


def setup_cpu(scheme_name: str, bps: int, segments: int) -> dict[str, Any]:
    """Setup on the CPU: `W` at random, its commitment, the verifier's salt and `C = Enc(W)` segment by segment."""
    from verity.experimental.protocols.pous import schemes, setup
    from verity.experimental.protocols.pous.schemes.p2 import P2Params

    base = schemes.get(scheme_name)
    params = dataclasses.replace(base.params, blocks_per_segment=bps, segments=segments) \
        if isinstance(base.params, P2Params) else base.params.with_segments(segments)
    scheme = schemes.get(scheme_name, params=params)
    w = os.urandom(params.w_bytes)
    commitment = setup.commit(params, w)
    salt = setup.draw_salt(os.urandom(32), commitment)
    t0 = time.monotonic()
    seg = params.segment_bytes
    blocks = [b for s in range(params.segments) for b in scheme.encode_segment(salt, s, w[s * seg:(s + 1) * seg])]
    return {"scheme": scheme.name, "manifest": params.manifest(), "blocks": blocks, "salt": salt, "commitment": commitment,
            "setup": salt + commitment, "encode_s": time.monotonic() - t0}


class Logical:
    def __init__(self) -> None:
        self.now = 0

    def __call__(self) -> int:
        return self.now


def run_session(blocks: Sequence[bytes], ctree: SP.CTree, k: int, signer: Any, limit_ns: int,
                respond: Callable[[int], bytes] | None = None) -> dict[str, Any]:
    """One session of the memory challenger over `blocks` on a logical clock, its header binding C's root."""
    clock = Logical()

    def answer(i: int) -> bytes:
        clock.now += STEP_NS
        return (respond or blocks.__getitem__)(i)

    c = mc.MemoryChallenger(mc.CoinTree.draw(os.urandom(32), os.urandom(32), len(blocks), k), signer, limit_ns=limit_ns,
                            block_bytes=len(blocks[0]), binding=SP.session_binding(ctree.root, len(blocks)), clock=clock,
                            resolution_ns=1)
    answered: list[bytes] = []
    signed = c.run(lambda i: answered.append(answer(i)) or answered[-1])
    return {"signed": signed, "commitment": c.commitment, "openings": c.openings(), "answered": answered}


def stage(d: Path, signed: mc.SignedSession, answered: Sequence[bytes], salts: Sequence[bytes]) -> dict[str, Any]:
    """The answered blocks as the unit's instances, one a round: `block` under C's salts, `o0` under the zero salt."""
    from verity_flock import class_statement as CS

    nb, k = len(answered[0]), len(answered)
    t0 = time.monotonic()
    r = CS.stage_unit(SP.unit(nb), SP.set_name(signed.commitment.session), [("block", SP.SCHEMA, 8 * nb, ["block"])],
                      [(SP.SCHEMA, [0])], SP.answer_rows(answered), d, os.urandom(32),
                      salts={"block": list(salts), "o0": [SP.ZERO_SALT] * k})
    pub = (d / "pub.bin").read_bytes()
    head, _ = SP.read_public(pub)
    return {"pub": pub, "head": head, "roots": head["frame_v3"]["roots"], "pin": r["circuit_sha512"], "ands": r["ands"],
            "stage_s": time.monotonic() - t0}


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def session_digest(sdir: Path) -> str:
    """The SHA-512 the firewall delivers for a session record: over each file's name and SHA-512, in name order."""
    h = hashlib.sha512()
    for f in sorted(p for p in sdir.rglob("*") if p.is_file()):
        h.update(f.relative_to(sdir).as_posix().encode() + b"\0" + hashlib.sha512(f.read_bytes()).digest())
    return h.hexdigest()


def prove_and_verify(flock: str, lean: str, d: Path, pin: str, zk: bool = True) -> dict[str, Any]:
    """`flock-circuit prove` against its own `serve` (one session), then the Lean `flock-verify` on that session and the
    public file: the verifier of record's verdict."""
    z = ["--zk"] if zk else []
    circ, inst, pub, sessions = d / "circuit.txt", d / "inst.bin", d / "pub.bin", d / "sessions"
    port = _free_port()
    t0 = time.monotonic()
    with open(d / "serve.log", "w") as log:
        srv = subprocess.Popen([flock, "serve", *z, "--listen", f"127.0.0.1:{port}", "--out", str(sessions), "--instances",
                                str(pub), "--circuit", str(circ), "--pin", pin, "--operator", "pous-service", "--sessions", "1"],
                               stdout=log, stderr=subprocess.STDOUT)
    while "SERVING " not in (d / "serve.log").read_text():
        if srv.poll() is not None:
            return {"accepted": False, "why": f"serve exited: {(d / 'serve.log').read_text()[-400:]}", "sessions": []}
        time.sleep(0.2)
    pr = subprocess.run([flock, "prove", *z, "--verifier", f"127.0.0.1:{port}", "--instances", str(inst), "--circuit", str(circ),
                         "--runs", "1", "--warm", "0"], capture_output=True, text=True, check=False)
    (d / "prove.out").write_text(pr.stdout)
    (d / "prove.err").write_text(pr.stderr)
    try:
        srv.wait(timeout=600)
    except subprocess.TimeoutExpired:
        srv.kill()
        srv.wait()
    prove_s = time.monotonic() - t0
    dirs = sorted(p for p in sessions.iterdir() if p.is_dir()) if sessions.is_dir() else []
    if pr.returncode or len(dirs) != 1:
        return {"accepted": False, "why": f"prove exited {pr.returncode}, {len(dirs)} sessions: {pr.stderr[-400:]}",
                "sessions": [], "prove_s": prove_s}
    t0 = time.monotonic()
    v = subprocess.run([lean, "verify", *z, "--circuit", str(circ), "--statement", "verity/flock-circuit", "--public", str(pub),
                        "--session", str(dirs[0])], capture_output=True, text=True, check=False)
    (d / "lean.out").write_text(v.stdout)
    (d / "lean.err").write_text(v.stderr)
    lines = [json.loads(ln[8:]) for ln in v.stdout.splitlines() if ln.startswith("VERDICT ")]
    accepted = len(lines) == 1 and lines[0].get("accepted") is True and v.returncode == 0
    live = [json.loads(ln[5:]) for ln in pr.stdout.splitlines() if ln.startswith("LIVE")]
    return {"accepted": accepted, "why": (lines[0].get("why") if lines else v.stderr[-400:]) or None,
            "sessions": [session_digest(dirs[0])], "prove_s": prove_s, "verify_s": time.monotonic() - t0,
            "proof_bytes": (live[0].get("proof_bytes") if live else None), "zk": zk}


def audit(tag: str, out: Path, nb: int, s: dict[str, Any], salts: Sequence[bytes], c_root: bytes, c_setup: bytes,
          c_paths: Sequence[Sequence[bytes]], schemes: Any, trusted: set[bytes], flock: str | None, lean: str | None,
          zk: bool, signed: mc.SignedSession | None = None) -> dict[str, Any]:
    """One session's answers through the service: a stream of its own (budget 1), registered, drawn, proved, delivered and
    decided.  `signed` replaces the session's signed record in the claims check (the tampered-record control)."""
    d = out / "witness" / tag
    d.mkdir(parents=True, exist_ok=True)
    honest = s["signed"]
    k = len(s["answered"])
    st = stage(d, honest, s["answered"], salts)
    stream = S.Stream(SP.spec(nb), SP.unit(nb), budget=1, delta=DELTA, challenger=SP.CHALLENGER)
    reg = stream.register(SP.record(nb, honest.commitment.session, k, st["roots"]), SP.ports(honest.commitment.session, k))
    draw = stream.select(reg)
    if flock and lean:
        lv = prove_and_verify(flock, lean, d, st["pin"], zk)
    else:
        lv = {"accepted": True, "sessions": [hashlib.sha512(st["pub"]).hexdigest()], "stub": True,
              "why": "no C-Flock and Lean binaries: a stub verdict of record (the CPU tests)"}
    sessions = lv["sessions"] or [hashlib.sha512(b"no session").hexdigest()]
    late = stream.deliver(draw, sessions)
    claims = SP.claims_check(signed or honest, s["commitment"], s["openings"], c_root, c_setup, c_paths, st["pub"],
                             trusted=trusted, schemes=schemes)
    claims["domains"] = st["head"]["frame_v3"]["domain_ids"] == {p: v["domain"] for p, v in SP.ports(honest.commitment.session, k).items()}
    if not claims["domains"]:
        claims["accepted"] = False
        claims["reasons"].append("the public file's domains are not the challenger's")
    verdicts = {"lean": {"of_record": True, "accepted": bool(lv["accepted"]), "digest": hashlib.sha512(st["pub"]).hexdigest(),
                         "sessions": sessions},
                "pous": {"of_record": True, "accepted": bool(claims["accepted"]), "sessions": sessions}}
    outcome = late or stream.outcome(draw, verdicts, {"statement": "verity/flock-circuit", "circuit_sha512": st["pin"],
                                                      "public_sha512": hashlib.sha512(st["pub"]).hexdigest()},
                                     session_roots=st["roots"])
    (d / "signed.json").write_text(json.dumps((signed or honest).to_json(), indent=1))
    return {"tag": tag, "accepted": outcome.accepted, "symbol": stream.symbols[-1], "exhaustive": outcome.exhaustive is not None,
            "record": outcome.record, "stream": stream.records[-1], "lean": {k_: v for k_, v in lv.items() if k_ != "sessions"},
            "claims": claims, "k": k, "ands": st["ands"], "stage_s": st["stage_s"]}


def tampered(block: bytes) -> bytes:
    return block[:-1] + bytes([block[-1] ^ 0x80])


def edited(signed: mc.SignedSession) -> mc.SignedSession:
    """The signed records with one receipt time moved 1 ns earlier after signing."""
    rounds = list(signed.rounds)
    r = rounds[len(rounds) // 2]
    rounds[r.j] = mc.Round(r.j, r.index, r.send_ns, r.receipt_ns - 1, r.digest)
    return mc.SignedSession(signed.commitment, signed.ended, tuple(rounds), signed.scheme, signed.public_key, signed.signature)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--scheme", default=None, help="a registered scheme (CPU setup; default P2's, `schemes.p2.P2().name`)")
    ap.add_argument("--bps", type=int, default=2, help="blocks per segment (CPU setup)")
    ap.add_argument("--segments", type=int, default=1)
    ap.add_argument("--k", type=int, default=4, help="rounds (CPU setup)")
    ap.add_argument("--limit-ns", type=int, default=0)
    ap.add_argument("--signer", default="test", choices=["test", "lms"])
    ap.add_argument("--lms-key", default=None)
    ap.add_argument("--keygen-workers", type=int, default=len(os.sched_getaffinity(0)))
    ap.add_argument("--flock", default=None, help="flock-circuit (sha512, glue)")
    ap.add_argument("--lean", default=None, help="the Lean flock-verify")
    ap.add_argument("--no-zk", action="store_true")
    ap.add_argument("--from-audit", default=None, help="the GPU harness's service-input.json")
    a = ap.parse_args(argv)
    if (a.signer == "lms") != (a.lms_key is not None):
        ap.error("--signer lms takes --lms-key DIR; --signer test takes none")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    doc = run(a, out)
    (out / "summary.json").write_text(json.dumps(doc, indent=1, default=str))
    (out / "verdict.json").write_text(json.dumps({"accepted": doc["accepted"]}) + "\n")
    print(json.dumps({"accepted": doc["accepted"], "controls": {c["tag"]: c["symbol"] for c in doc["controls"]},
                      "passed": doc["passed"]}))
    return 0 if doc["passed"] else 1


def run(a: argparse.Namespace, out: Path) -> dict[str, Any]:
    """The honest audit and the controls (module docstring)."""
    from verity.experimental.protocols.pous.schemes import p2

    zk = not a.no_zk
    if a.from_audit:
        return from_audit(a, out, zk)
    su = setup_cpu(a.scheme or p2.P2().name, a.bps, a.segments)
    blocks, nb = su["blocks"], len(su["blocks"][0])
    salt_key = os.urandom(32)
    t0 = time.monotonic()
    ctree = SP.commit_c([mc.block_digest(b) for b in blocks], salt_key, su["setup"])
    commit_s = time.monotonic() - t0
    signer, schemes = signer_of(a.signer, a.lms_key, a.keygen_workers)
    trusted = {signer.public_key}
    limit = a.limit_ns or 500_000 + p2.RTT_ALLOWANCE_CAP_NS

    def prove(tag: str, s: dict[str, Any], signed: mc.SignedSession | None = None) -> dict[str, Any]:
        idx = [r.index for r in s["signed"].rounds]
        return audit(tag, out, nb, s, [SP.c_salt(salt_key, i) for i in idx], ctree.root, ctree.setup,
                     [ctree.path(i) for i in idx], schemes, trusted, a.flock, a.lean, zk, signed)

    honest_s = run_session(blocks, ctree, a.k, signer, limit)
    honest = prove("honest", honest_s)
    j = a.k // 2
    hit: dict[str, int] = {}

    def respond(i: int) -> bytes:
        n = hit.setdefault("n", 0)
        hit["n"] = n + 1
        return tampered(blocks[i]) if n == j else blocks[i]

    controls = [prove("tampered-block", run_session(blocks, ctree, a.k, signer, limit, respond)),
                prove("tampered-record", honest_s, edited(honest_s["signed"]))]
    return summary(honest, controls, {"scheme": su["scheme"], "manifest": su["manifest"],
                                      "w_commitment": su["commitment"].hex(), "blocks": len(blocks),
                                      "block_bytes": nb, "encode_s": su["encode_s"], "commit_c_s": commit_s,
                                      "signer": {"scheme": signer.scheme, "test": signer.scheme not in mc.SCHEMES},
                                      "limit_ns": limit, "clock": "logical", "zk": zk})


def from_audit(a: argparse.Namespace, out: Path, zk: bool) -> dict[str, Any]:
    """The GPU harness's session (`service-input.json`) through the service, and the tampered-record control on it."""
    doc = json.loads(Path(a.from_audit).read_text())
    signed = mc.SignedSession.from_json(doc["signed"])
    s = {"signed": signed, "commitment": mc.SessionCommitment.from_json(doc["commitment"]),
         "openings": [mc.CoinOpening.from_json(o) for o in doc["openings"]],
         "answered": [bytes.fromhex(x["block"]) for x in doc["answered"]]}
    salts = [bytes.fromhex(x["salt"]) for x in doc["answered"]]
    paths = [[bytes.fromhex(p) for p in x["path"]] for x in doc["answered"]]
    c_root, c_setup = bytes.fromhex(doc["c_root"]), bytes.fromhex(doc["c_setup"])
    schemes = mc.SCHEMES if signed.scheme in mc.SCHEMES else test_signer().SCHEMES
    trusted = {signed.public_key}
    nb = len(s["answered"][0])
    honest = audit("honest", out, nb, s, salts, c_root, c_setup, paths, schemes, trusted, a.flock, a.lean, zk)
    bad = dict(s, answered=[tampered(b) if j == len(s["answered"]) // 2 else b for j, b in enumerate(s["answered"])])
    controls = [audit("tampered-block", out, nb, bad, salts, c_root, c_setup, paths, schemes, trusted, a.flock, a.lean, zk),
                audit("tampered-record", out, nb, s, salts, c_root, c_setup, paths, schemes, trusted, a.flock, a.lean, zk,
                      edited(signed))]
    return summary(honest, controls, {"from_audit": doc.get("source"), "scheme": doc.get("scheme"),
                                      "manifest": doc.get("manifest"), "w_commitment": doc.get("w_commitment"),
                                      "blocks": s["commitment"].blocks, "block_bytes": nb,
                                      "signer": {"scheme": signed.scheme, "test": signed.scheme not in mc.SCHEMES},
                                      "limit_ns": s["commitment"].limit_ns, "zk": zk,
                                      "timing": doc.get("timing")})


def summary(honest: dict[str, Any], controls: list[dict[str, Any]], run: dict[str, Any]) -> dict[str, Any]:
    stub = honest["lean"].get("stub", False)
    return {"accepted": honest["accepted"], "passed": honest["accepted"] and not any(c["accepted"] for c in controls),
            "verifier_of_record": "stub (no binaries)" if stub else "Lean flock-verify", "run": run,
            "honest": honest, "controls": controls}


if __name__ == "__main__":
    raise SystemExit(main())
