"""One PoUS period on one node, at a small point: the setup, the memory challenger's session and the run's check, with the
answer of PoUS's component of onsite Verity (`verity/pous/p2/v4/run`, `PousCheck.RunProtocol` in `check/`).

`run(node, store, coins, signer=..., log=...)` plays every party in one process, in the protocol's order:

1. the prover commits to `W`, the node's `store` padded with zeros to the point's payload (`setup.commit`);
2. the verifier draws the salt after the commitment, from its live `coins` (`setup.draw_salt`);
3. the prover encodes `C*` (`P2.encode_segment`), and the verifier decodes it and checks `Dec(C*)` against the commitment
   (`setup.verifier_key_from_encoding`, the public-encoder setup; in Lean `verity/pous/p2/v4/setup`), publishing its root;
4. the memory challenger takes its table of `C*` (each block's `memory_challenger.block_digest`), draws its `k` indices
   from `coins` under a session id the auditor designates from them, and runs its timed session against the prover's
   answers (`prover`, by default the honest one that holds `C*`), signed under `signer`;
5. the verifier judges the run (`run.check_at`) and answers in the core interface's canonical form (`answer`).

What an accepted run certifies is `certify`'s, as `PousCheck.RunProtocol.certify` computes it: `s_bits = ⌊ρ·blocks·w⌋`
at `eps = δ + 2^eps_log2` when P2's one certificate covers the point, else 0 and 0 with the profile naming each gap. The
certificate is at P2's deployed point only (`run.DEPLOYED`, a 1 GiB `C*`), so a small point's accepted run certifies no
storage: its verdict is real and its quantity vacuous.

`leg(period_dir, node, out)` is `benchmarks/all_protocols`' leg for space: `W` is the period's records, the coins are
fresh from the OS, and the signer is a key at S1's deployed parameters (`memory_challenger.LMS`) that `challenger` makes
once beside the periods. Every party runs on the one node, so `trusted-nebius-host`'s premise that the memory challenger
runs on a machine the prover doesn't control is not met by this run: it exercises the code, not the deployment.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import secrets
import subprocess
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from verity.ml.commitments.identity import canonical_json_bytes
from verity.ml.crypto import lms
from verity.ml.randomness import derive

from . import memory_challenger as mc
from . import run as R
from . import setup
from .schemes import p2 as P2

PROTOCOL = "verity/pous/p2/v4/run"
"""`PousCheck.RunProtocol.ID`; a test signer's runs answer under `memory_challenger.TEST_PREFIX` + it."""
SETUP_PROTOCOL = "verity/pous/p2/v4/setup"
SESSION_DOMAIN = "verity/pous-period/session/v1"
W, C0 = 1024, 105
"""The small point's width: a 1,024-bit prime `2^w − c0` of P2's ladder, 126 payload bytes a 128-byte block."""
K = P2.CERT_K
LIMIT_NS = P2.DELTA_NS + P2.RTT_ALLOWANCE_CAP_NS


def point_for(size: int, w: int = W, c0: int = C0, k: int = K, limit_ns: int = LIMIT_NS) -> R.Point:
    """The smallest point at width `w` whose payload holds `size` bytes: one segment of up to 2^16 blocks, then more."""
    payload = (w - 16) // 8
    blocks = max(1, -(-size // payload))
    segments = -(-blocks // (1 << 16))
    return R.Point(P2.P2Params(w, c0, -(-blocks // segments), segments), k, limit_ns)


def gaps(point: R.Point) -> tuple[Any, list[str]]:
    """`PousCheck.RunProtocol.gaps`: P2's certificate at the point, and why it doesn't cover the run (none when it does)."""
    q = point.params
    cert = P2.P2(P2.P2Params(q.w, q.c0, q.blocks_per_segment, q.segments)).certificate
    if cert is None:
        return None, [f"P2 has no certificate at (w, c0, blocks per segment) = ({q.w}, {q.c0}, {q.blocks_per_segment})"]
    segments = ([f"W has {q.segments} segments; {cert.theorem} is about {cert.segments}"]
                if cert.segments is not None and q.segments != cert.segments else [])
    return cert, [*cert.refusals(point.k, point.limit_ns), *segments]


def certify(point: R.Point) -> dict[str, Any]:
    """`PousCheck.RunProtocol.certify`'s public outputs: `{"certified", "eps", "s_bits"}`, eps an exact ratio."""
    cert, why = gaps(point)
    if cert is None or why:
        return {"certified": False, "eps": "0", "s_bits": 0}
    q = point.params
    eps = cert.delta + Fraction(1, 2 ** -cert.eps_log2)
    return {"certified": True, "eps": str(eps), "s_bits": q.blocks * q.w * cert.rho.numerator // cert.rho.denominator}


def profile(point: R.Point) -> dict[str, Any]:
    """`PousCheck.RunProtocol.profileJson`: the store and the audit the run was checked at, the certificate, every gap."""
    cert, why = gaps(point)
    q = point.params
    return {"block_bits": q.w, "blocks": q.blocks, "certificate": None if cert is None else cert.to_json(),
            "experimental": P2.EXPERIMENTAL, "gaps": why, "k": point.k, "limit_ns": point.limit_ns}


def point_json(point: R.Point) -> dict[str, Any]:
    q = point.params
    return {"k": point.k, "limit_ns": point.limit_ns,
            "params": {"blocks_per_segment": q.blocks_per_segment, "c0": q.c0, "segments": q.segments, "w": q.w}}


def case(point: R.Point, session_id: bytes, trusted: Sequence[bytes], published: R.Setup, digests: bytes,
         session: R.Session) -> dict[str, Any]:
    """The run protocol's case, `{"commitment": hex, "evidence": hex, "params": JSON}` (`PousCheck.RunProtocol`)."""
    params = {"point": point_json(point), "session_id": session_id.hex(), "trusted": [t.hex() for t in trusted]}
    evidence = {"digests": digests.hex(), "session": session.to_json()}
    return {"commitment": canonical_json_bytes(published.to_json()).hex(),
            "evidence": canonical_json_bytes(evidence).hex(), "params": params}


def answer(point: R.Point, schemes: Mapping[str, mc.Verify], trusted: Collection[bytes], published: R.Setup,
           digests: bytes, session: R.Session, session_id: bytes, protocol: str = PROTOCOL) -> dict[str, Any]:
    """The run protocol's answer (`Core.Answer.canonical`) on decoded inputs: `run.check_at`, then `certify`."""
    digest = hashlib.sha512(canonical_json_bytes(published.to_json())).hexdigest()
    try:
        R.check_at(point, schemes, trusted, published, digests, session, session_id)
    except R.Refused as e:
        return {"commitment": digest, "profile": None, "protocol": protocol, "public": None, "reason": str(e),
                "verdict": "reject"}
    return {"commitment": digest, "profile": profile(point), "protocol": protocol, "public": certify(point),
            "reason": None, "verdict": "accept"}


def setup_case(q: P2.P2Params, commitment: bytes, salt: bytes, encoded: Sequence[Sequence[bytes]]) -> dict[str, Any]:
    """The setup protocol's case (`PousCheck.SetupProtocol`): the manifest, `commitment ‖ salt`, and `C*` framed."""
    u32 = lambda n: n.to_bytes(4, "big")  # noqa: E731
    evidence = u32(len(encoded)) + b"".join(u32(len(seg)) + b"".join(u32(len(b)) + b for b in seg) for seg in encoded)
    return {"commitment": (commitment + salt).hex(), "evidence": evidence.hex(), "params": q.manifest()}


type Prover = Callable[[Sequence[bytes]], Callable[[int], bytes]]
"""The prover's side of the session: from the blocks of `C*` it was sent, its answer to each index."""


def honest(blocks: Sequence[bytes]) -> Callable[[int], bytes]:
    """The prover that holds every block of `C*`."""
    return lambda i: blocks[i]


@dataclass(frozen=True)
class Period:
    """A period's PoUS: the run protocol's answer and case, and the setup protocol's case that published its commitment."""

    answer: dict[str, Any]
    case: dict[str, Any]
    setup_case: dict[str, Any]

    @property
    def verdict(self) -> str:
        return self.answer["verdict"]

    @property
    def s_bits(self) -> int:
        """The storage certified, in bits: 0 unless accepted under a certificate that covers the point."""
        return self.answer["public"]["s_bits"] if self.verdict == "accept" else 0

    @property
    def eps(self) -> Fraction:
        """The error `s_bits` holds at: 0 (vacuous) when nothing is certified."""
        return Fraction(self.answer["public"]["eps"]) if self.verdict == "accept" else Fraction(0)


def run(node: str, store: bytes, coins: bytes, *, signer: mc.Signer, log: mc.SessionLog, point: R.Point | None = None,
        trusted: Collection[bytes] | None = None, schemes: Mapping[str, mc.Verify] = mc.SCHEMES,
        protocol: str = PROTOCOL, prover: Prover = honest,
        clock: Callable[[], int] = time.perf_counter_ns) -> Period:
    """The period's PoUS on `node` (module docstring): `store` is what the node keeps, `coins` the verifier's and the
    memory challenger's live secret for the period, `signer` the memory challenger's key and `log` its session log.
    `trusted` is the verifier's list of memory-challenger keys (by default the signer's), `schemes` the signature schemes
    it accepts and `protocol` the id it answers under (a test signer's: `TEST_PREFIX` and `pous_test_signer.SCHEMES`).
    A setup the verifier refuses raises `ValueError`, since no run of it exists."""
    point = point_for(len(store)) if point is None else point
    q = point.params
    if len(store) > q.w_bytes:
        raise ValueError(f"the store is {len(store)} bytes, past the point's payload of {q.w_bytes}")
    w = bytes(store) + bytes(q.w_bytes - len(store))
    commitment = setup.commit(q, w)
    salt = setup.draw_salt(coins, commitment)
    encoded = [P2.encode_segment(q, None, salt, s, w[s * q.segment_bytes:(s + 1) * q.segment_bytes])
               for s in range(q.segments)]
    root = setup.verifier_key_from_encoding(q, None, commitment, salt, encoded).root
    published = R.Setup(q.blocks, commitment, root, salt)
    blocks = [b for seg in encoded for b in seg]
    digests = b"".join(mc.block_digest(b) for b in blocks)
    session_id = derive(coins, SESSION_DOMAIN, {"node": node, "setup": root}).stream("session", 32)
    tree = mc.CoinTree.draw(coins, session_id, q.blocks, point.k)
    challenger = mc.MemoryChallenger(tree, signer, log=log, limit_ns=point.limit_ns, block_bytes=q.block_bytes,
                                     binding=root, clock=clock)
    signed = challenger.run(prover(blocks))
    session = R.Session(signed, challenger.commitment, challenger.openings())
    trusted = (signer.public_key,) if trusted is None else tuple(trusted)
    return Period(answer(point, schemes, trusted, published, digests, session, session_id, protocol),
                  case(point, session_id, trusted, published, digests, session), setup_case(q, commitment, salt, encoded))


def serve(exe: str | os.PathLike, cases: Sequence[Mapping[str, Any]], *args: str) -> list[dict[str, Any]]:
    """`EXE [ARGS] serve` on `cases`, one canonical JSON line each: the Lean's answers (`Core.Protocol` CLI)."""
    lines = "".join(canonical_json_bytes(dict(c)).decode() + "\n" for c in cases)
    p = subprocess.run([str(exe), *args, "serve"], input=lines, capture_output=True, text=True, check=True)
    return [json.loads(x) for x in p.stdout.splitlines()]


def challenger(directory: str | os.PathLike, *, workers: int | None = None) -> mc.LmsSigner:
    """The memory challenger's key at S1's deployed parameters in `directory`, made there once (its seed from the OS, its
    tree computed in `workers` processes, all of them by default) with its leaf state beside it, so each session signs at
    a fresh leaf, and its session log (`sessions`). Windows that run at once share them: the first makes them under the
    directory's lock (`device.lock`) while the others wait on it, and each window's signer takes the leaf state only to
    sign (`memory_challenger.LmsSigner`)."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    p, top = lms.DEPLOYED, 2 ** (lms.DEPLOYED.H + 1)
    seed_file, tree_file, state = d / "seed", d / "tree.bin", d / "leaf-state.json"
    with open(d / "device.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not seed_file.exists():
            seed = secrets.token_bytes(p.N)
            key = lms.keygen(p, seed, (os.cpu_count() or 1) if workers is None else workers)
            tree_file.write_bytes(b"".join(key.tree[1:]))
            fd = os.open(seed_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(seed)
            lms.LeafState.create(state, lms.public_key(key), p.H)
        else:
            seed, data = seed_file.read_bytes(), tree_file.read_bytes()
            tree = (b"", *(data[p.N * r:p.N * (r + 1)] for r in range(top - 1)))
            key = lms.PrivateKey(p, hashlib.sha512(b"verity/lms/v0/I" + seed).digest()[:16], seed, tree)
        if not (d / "sessions").exists():
            mc.SessionLog.create(d / "sessions")
    return mc.LmsSigner(key, state)


def records(period_dir: str | os.PathLike, skip: Collection[str] = ("answers",)) -> bytes:
    """The period's records as one byte string, the store the node keeps for the period: each file under `period_dir`
    (but those under `skip`) in path order, framed `u32 len(path) ‖ path ‖ u64 len(bytes) ‖ bytes`."""
    root = Path(period_dir)
    out = bytearray()
    for f in sorted(p for p in root.rglob("*") if p.is_file() and p.relative_to(root).parts[0] not in skip):
        name, data = f.relative_to(root).as_posix().encode(), f.read_bytes()
        out += len(name).to_bytes(4, "big") + name + len(data).to_bytes(8, "big") + data
    return bytes(out)


def leg(period_dir: Path, node: str, out: Path) -> dict[str, Any]:
    """Space's leg of `benchmarks/all_protocols` (`leg(period_dir, node, out) -> answer`): `run` on the period's
    records with fresh coins, the memory challenger's key and session log beside the periods. With `pous-run-protocol`
    and `pous-setup` among the period's tools, the Lean answers (the verifier of record) and must agree with the Python;
    it writes `out/pous.case.json` (onsite's PoUS case), `out/pous-setup.case.json` and `out/pous.json`."""
    tools = json.loads((Path(period_dir) / "period.json").read_text()).get("tools") or {}
    device = Path(period_dir).parent / "pous-challenger"
    signer = challenger(device)
    if not (device / "sessions").exists():
        mc.SessionLog.create(device / "sessions")
    store = records(period_dir)
    t = time.monotonic()
    period = run(node, store, secrets.token_bytes(64), signer=signer, log=mc.SessionLog(device / "sessions"))
    seconds = round(time.monotonic() - t, 2)
    answer, lean = period.answer, {}
    if "pous-run-protocol" in tools:
        lean["run"] = serve(tools["pous-run-protocol"], [period.case])[0]
        if {k: v for k, v in lean["run"].items() if k != "reason"} != {k: v for k, v in answer.items() if k != "reason"}:
            raise RuntimeError(f"pous-run-protocol answered {lean['run']}, the Python {answer}")
        answer = lean["run"]
    if "pous-setup" in tools:
        lean["setup"] = serve(tools["pous-setup"], [period.setup_case])[0]
        if lean["setup"]["verdict"] != "accept":
            raise RuntimeError(f"pous-setup refused the setup the Python accepted: {lean['setup']}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "pous.case.json").write_text(json.dumps(period.case, sort_keys=True))
    (out / "pous-setup.case.json").write_text(json.dumps(period.setup_case, sort_keys=True))
    (out / "pous.json").write_text(json.dumps({"answer": answer, "eps": str(period.eps), "lean": lean,
                                               "s_bits": period.s_bits, "seconds": seconds, "store_bytes": len(store)},
                                              sort_keys=True, indent=1))
    return answer


__all__ = ["PROTOCOL", "Period", "Prover", "answer", "case", "certify", "challenger", "gaps", "honest", "leg", "point_for",
           "profile", "records", "run", "serve", "setup_case"]
