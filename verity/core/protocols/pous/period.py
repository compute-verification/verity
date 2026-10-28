"""One PoUS period on one node: the setup, the memory challenger's session and the run's check, with the answer of PoUS's
component of onsite Verity (`verity/pous/p2/v4/run`, `PousCheck.RunProtocol` in `check/`).

`run(node, store, coins, signer=..., log=...)` plays every party in one process, in the protocol's order (`setup_store`
is steps 1 to 3, `audit` steps 4 and 5, so a node's store can be set up once and audited every period):

1. the prover commits to `W`, the node's `store` padded with zeros to the point's payload (`setup.commit`);
2. the verifier draws the salt after the commitment, from its live `coins` (`setup.draw_salt`);
3. the prover encodes `C*` (`P2.encode_segment`), and the verifier decodes it and checks `Dec(C*)` against the commitment
   (`setup.verifier_key_from_encoding`, the public-encoder setup; in Lean `verity/pous/p2/v4/setup`), publishing its root;
4. the memory challenger takes its table of `C*` (each block's `memory_challenger.block_digest`), draws its `k` indices
   from `coins` under a session id the auditor designates from them, and runs its timed session against the prover's
   answers (`prover`, by default the honest one that holds `C*`), signed under `signer`;
5. the verifier judges the run, setup's acceptance of `C*` publishing exactly the run's commitment (`tie`) and then
   `run.check_at`, and answers in the core interface's canonical form (`answer`); the case carries setup's own inputs.

What an accepted run certifies is `certify`'s, as `PousCheck.RunProtocol.certify` computes it: `s_bits = ⌊ρ·blocks·w⌋`
at `eps = δ + 2^eps_log2` = 1/100 + 2^-128 when a certificate covers the point (`certificate`: P2's registered one at
`run.DEPLOYED`, a 1 GiB `C*`, else P2's proved family `p2.family_certificate`, any `2^8` to `2^23` blocks at the
registered width and prime, else the per-period guarantee `p2.wide_certificate`, `periodHeldAt`, to `2^27` blocks, which
holds Demo B's `C*` of 1038 segments, 1,060,008,348,186 bits), else 0 and 0 with the profile naming each gap.
`point_for` gives the smallest such point holding a store, 256 blocks (526 KB) at least; the small point
(`point_for(size, W, C0)`, 1,024-bit blocks) certifies nothing.

Encoding is the cost: two 16,448-bit roots a block, about 3.7 s a block on one core in this pure-Python reference path,
so 16 core-minutes at 256 blocks and 550 core-hours at the deployed `2^19`; `encode` splits it
over processes. The toy's 6,240,150-byte store is 3,039 blocks there, set up in 367 s in 32 processes (3.86 core-seconds
a block). An audit is cheap: about 1.7 s at `2^19` blocks, mostly the table's root.

`leg(period_dir, node, out)` is `benchmarks/all_protocols`' leg for space: by default it sets the node's store up once at
`point_for`'s certified point and audits it every period, the coins fresh from the OS. The memory challenger is a party
of its own (`challenger_party`) on `period.json`'s `challenger_host`, with its own coins and a key at S1's deployed
parameters (`memory_challenger.LMS`) that `challenger` makes once on that host; the prover and the verifier run on the
node.
`trusted-host`'s premise, that the memory challenger runs on a machine the prover doesn't control, holds only
when the challenger host isn't the node, and the leg says which (`challenger_party.placement`). The setup's salt is
still drawn on the node.
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
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from verity.core.deploy.stream import Side
from verity.ml.commitments.identity import canonical_json_bytes
from verity.ml.crypto import lms
from verity.ml.randomness import derive

from . import challenger_party
from . import memory_challenger as mc
from . import protocol as pous_protocol
from . import run as R
from . import setup
from .schemes import p2 as P2
from .verifier import DIGEST_BYTES

PROTOCOL = "verity/pous/p2/v4/run"
"""`PousCheck.RunProtocol.ID`; a test signer's runs answer under `memory_challenger.TEST_PREFIX` + it."""
SETUP_PROTOCOL = "verity/pous/p2/v4/setup"
SESSION_DOMAIN = "verity/pous-period/session/v1"
W, C0 = 1024, 105
"""The small point's width: a 1,024-bit prime `2^w − c0` of P2's ladder, 126 payload bytes a 128-byte block."""
K = P2.CERT_K
LIMIT_NS = P2.DELTA_NS + P2.RTT_ALLOWANCE_CAP_NS


def point_for(size: int, w: int = P2.POINT[0], c0: int = P2.POINT[1], k: int = K, limit_ns: int = LIMIT_NS) -> R.Point:
    """The smallest point at width `w` whose payload holds `size` bytes: one segment of up to 2^16 blocks, then more. At
    the registered width and prime, the default, it has at least `2^8` blocks (`p2.CERT_FAMILY_BLOCKS`), so with `k =
    105` and `Δ + RTT` at the cap `certificate` covers it to `2^27` blocks; `point_for(size, W, C0)` is the small point."""
    payload = (w - 16) // 8
    least = P2.CERT_FAMILY_BLOCKS[0] if (w, c0) == P2.POINT[:2] else 1
    blocks = max(least, -(-size // payload))
    segments = -(-blocks // (1 << 16))
    return R.Point(P2.P2Params(w, c0, -(-blocks // segments), segments), k, limit_ns)


def certificate(q: P2.P2Params) -> Any:
    """`PousCheck.RunProtocol.certificate`: P2's registered certificate when `q` is its point, segment count and all,
    else P2's proved family's (`p2.family_certificate`, `2^8` to `2^23` blocks), else the per-period guarantee's
    (`p2.wide_certificate`, to `2^27` blocks), else none."""
    registered = P2.P2(P2.P2Params(q.w, q.c0, q.blocks_per_segment, q.segments)).certificate
    if registered is not None and registered.segments == q.segments:
        return registered
    return P2.family_certificate(q) or P2.wide_certificate(q)


def gaps(point: R.Point) -> tuple[Any, list[str]]:
    """`PousCheck.RunProtocol.gaps`: the certificate at the point, and why it doesn't cover the run (none when it does)."""
    q = point.params
    cert = certificate(q)
    if cert is None:
        return None, [f"P2 has no certificate at (w, c0, blocks) = ({q.w}, {q.c0}, {q.blocks})"]
    return cert, list(cert.refusals(point.k, point.limit_ns))


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


def point_of(doc: Mapping[str, Any]) -> R.Point:
    q = doc["params"]
    return R.Point(P2.P2Params(q["w"], q["c0"], q["blocks_per_segment"], q["segments"]), doc["k"], doc["limit_ns"])


def case(point: R.Point, session_id: bytes, trusted: Sequence[bytes], published: R.Setup, digests: bytes,
         session: R.Session, setup_of: Mapping[str, Any]) -> dict[str, Any]:
    """The run protocol's case, `{"commitment": hex, "evidence": hex, "params": JSON}` (`PousCheck.RunProtocol`): its
    evidence carries the setup's own commitment and evidence, from `setup_of`, the setup protocol's case."""
    params = {"point": point_json(point), "session_id": session_id.hex(), "trusted": [t.hex() for t in trusted]}
    evidence = {"digests": digests.hex(), "session": session.to_json(),
                "setup": {"commitment": setup_of["commitment"], "evidence": setup_of["evidence"]}}
    return {"commitment": canonical_json_bytes(published.to_json()).hex(),
            "evidence": canonical_json_bytes(evidence).hex(), "params": params}


SetupInputs = tuple[bytes, bytes, Sequence[Sequence[bytes]]]
"""The setup's own inputs (`PousCheck.RunProtocol.SetupEvidence`): the server's commitment to `W`, the verifier's salt,
and `C*` segment by segment."""


def tie(point: R.Point, published: R.Setup, inputs: SetupInputs) -> None:
    """`PousCheck.RunProtocol.tie`: setup accepts `C*` against the commitment and salt at the point's parameters (the
    public-encoder path, `protocol.verifier_key`) and publishes exactly `published`, the run's commitment; else raises
    `run.Refused`."""
    commitment, salt, store = inputs
    q = point.params
    try:
        vk = pous_protocol.verifier_key(P2.P2(q), commitment, salt, mode=pous_protocol.PUBLIC, encoded=store)
    except ValueError as e:
        raise R.Refused(f"setup: {e}") from None
    if R.Setup(q.blocks, commitment, vk.root, salt) != published:
        raise R.Refused("the run's commitment is not what its setup published")


def answer(point: R.Point, schemes: Mapping[str, mc.Verify], trusted: Collection[bytes], published: R.Setup,
           digests: bytes, session: R.Session, session_id: bytes, protocol: str = PROTOCOL, *,
           setup_inputs: SetupInputs) -> dict[str, Any]:
    """The run protocol's answer (`Core.Answer.canonical`) on decoded inputs: `tie` on the setup's own inputs, then
    `run.check_at`, then `certify`."""
    digest = hashlib.sha512(canonical_json_bytes(published.to_json())).hexdigest()
    try:
        tie(point, published, setup_inputs)
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


def setup_inputs(setup_of: Mapping[str, Any]) -> SetupInputs:
    """The setup's own inputs from its case's `commitment` and `evidence`, read as `SetupProtocol`'s decoders read them
    (each count at most a quarter of the bytes left, nothing after the last segment); `ValueError` on bytes they refuse."""
    c, b = bytes.fromhex(setup_of["commitment"]), bytes.fromhex(setup_of["evidence"])
    if len(c) < DIGEST_BYTES:
        raise ValueError(f"the commitment to W is 64 bytes, then the salt; {len(c)} bytes is too short")
    pos = 0

    def take(n: int) -> bytes:
        nonlocal pos
        if pos + n > len(b):
            raise ValueError(f"C* is truncated at byte {pos}")
        pos += n
        return b[pos - n:pos]

    def count() -> int:
        n = int.from_bytes(take(4), "big")
        if 4 * n > len(b) - pos:
            raise ValueError(f"a count {n} exceeds the bytes left")
        return n

    store = [[take(int.from_bytes(take(4), "big")) for _ in range(count())] for _ in range(count())]
    if pos != len(b):
        raise ValueError(f"{len(b) - pos} bytes follow the last segment")
    return c[:DIGEST_BYTES], c[DIGEST_BYTES:], store


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


@dataclass(frozen=True)
class Store:
    """What a node's setup leaves: the point, what setup published, `C*` (which the prover keeps), the memory challenger's
    table of it, and the setup protocol's case."""

    point: R.Point
    published: R.Setup
    blocks: tuple[bytes, ...]
    digests: bytes
    setup_case: dict[str, Any]

    def save(self, directory: str | os.PathLike) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        (d / "c-star.bin").write_bytes(b"".join(self.blocks))
        (d / "digests.bin").write_bytes(self.digests)
        (d / "setup-case.json").write_text(json.dumps(self.setup_case, sort_keys=True))
        (d / "store.json").write_text(json.dumps({"point": point_json(self.point), "published": self.published.to_json()},
                                                 sort_keys=True))

    @classmethod
    def load(cls, directory: str | os.PathLike) -> Store:
        d = Path(directory)
        doc = json.loads((d / "store.json").read_text())
        point, data = point_of(doc["point"]), (d / "c-star.bin").read_bytes()
        n = point.params.block_bytes
        return cls(point, R.Setup.from_json(doc["published"]), tuple(data[i:i + n] for i in range(0, len(data), n)),
                   (d / "digests.bin").read_bytes(), json.loads((d / "setup-case.json").read_text()))


def _encode_part(job: tuple[P2.P2Params, bytes, int, int, bytes]) -> list[bytes]:
    return P2.encode_blocks(*job)


def encode(q: P2.P2Params, salt: bytes, w: bytes, workers: int = 1) -> list[list[bytes]]:
    """`C*` of `W` under `salt`, segment by segment: in this process, or split by blocks (`p2.encode_blocks`) over a pool of
    `workers` processes when more than one, which gives the same blocks."""
    seg, pb, bps = q.segment_bytes, q.payload_bytes, q.blocks_per_segment
    if workers <= 1:
        return [P2.encode_segment(q, None, salt, s, w[s * seg:(s + 1) * seg]) for s in range(q.segments)]
    step = max(1, -(-q.blocks // (4 * workers)))
    jobs = [(q, salt, s, j, w[s * seg + j * pb:s * seg + min(j + step, bps) * pb])
            for s in range(q.segments) for j in range(0, bps, step)]
    with ProcessPoolExecutor(workers) as pool:
        parts = list(pool.map(_encode_part, jobs))
    out: list[list[bytes]] = [[] for _ in range(q.segments)]
    for job, part in zip(jobs, parts, strict=True):
        out[job[2]] += part
    return out


def setup_store(store: bytes, coins: bytes, point: R.Point | None = None, *, workers: int = 1) -> Store:
    """Setup on the node (module docstring, steps 1 to 3): `W` is `store` padded to the point's payload, the salt drawn
    from the verifier's `coins` after the commitment, `C*` encoded in `workers` processes, and the public-encoder check of
    it (`setup.verifier_key_from_encoding`), which raises `ValueError` on a setup it refuses."""
    point = point_for(len(store)) if point is None else point
    q = point.params
    if len(store) > q.w_bytes:
        raise ValueError(f"the store is {len(store)} bytes, past the point's payload of {q.w_bytes}")
    w = bytes(store) + bytes(q.w_bytes - len(store))
    commitment = setup.commit(q, w)
    salt = setup.draw_salt(coins, commitment)
    encoded = encode(q, salt, w, workers)
    root = setup.verifier_key_from_encoding(q, None, commitment, salt, encoded).root
    blocks = tuple(b for seg in encoded for b in seg)
    return Store(point, R.Setup(q.blocks, commitment, root, salt), blocks, b"".join(mc.block_digest(b) for b in blocks),
                 setup_case(q, commitment, salt, encoded))


def audit(node: str, held: Store, coins: bytes, *, signer: mc.Signer | None = None, log: mc.SessionLog | None = None,
          trusted: Collection[bytes] | None = None, schemes: Mapping[str, mc.Verify] = mc.SCHEMES,
          protocol: str = PROTOCOL, prover: Prover = honest, clock: Callable[[], int] = time.perf_counter_ns,
          party: Side | None = None) -> Period:
    """One period's audit of the store `held` on `node` (module docstring, steps 4 and 5): the memory challenger's session
    under the id the auditor designates from `coins` and the setup's root, then the run's check and answer, the tie on
    the setup case `held` keeps, which the run's case carries. With `party`, the memory challenger is a party of its own
    (`challenger_party`): its side runs where `party` (a `stream.Side`) starts it, with its own coins, key and session
    log, and the prover answers it over a socket; `signer`, `log` and `clock` are unused, and the verifier's list is by
    default the key the side reports."""
    point, published = held.point, held.published
    q = point.params
    if party is None:
        session_id = derive(coins, SESSION_DOMAIN, {"node": node, "setup": published.root}).stream("session", 32)
        tree = mc.CoinTree.draw(coins, session_id, q.blocks, point.k)
        challenger = mc.MemoryChallenger(tree, signer, log=log, limit_ns=point.limit_ns, block_bytes=q.block_bytes,
                                         binding=published.root, clock=clock)
        signed = challenger.run(prover(held.blocks))
        session, key = R.Session(signed, challenger.commitment, challenger.openings()), signer.public_key
    else:
        got = challenger_party.session(party, challenger_party.request_of(node, point, published.root),
                                       prover(held.blocks))
        session_id, key = bytes.fromhex(got["session_id"]), bytes.fromhex(got["public_key"])
        session = R.Session.from_json(got["session"])
    trusted = (key,) if trusted is None else tuple(trusted)
    return Period(answer(point, schemes, trusted, published, held.digests, session, session_id, protocol,
                         setup_inputs=setup_inputs(held.setup_case)),
                  case(point, session_id, trusted, published, held.digests, session, held.setup_case), held.setup_case)


def run(node: str, store: bytes, coins: bytes, *, signer: mc.Signer | None = None, log: mc.SessionLog | None = None,
        point: R.Point | None = None, trusted: Collection[bytes] | None = None,
        schemes: Mapping[str, mc.Verify] = mc.SCHEMES, protocol: str = PROTOCOL, prover: Prover = honest,
        clock: Callable[[], int] = time.perf_counter_ns, workers: int = 1,
        party: Side | None = None) -> Period:
    """The period's PoUS on `node`, setup and audit (module docstring): `store` is what the node keeps, `coins` the
    verifier's and the memory challenger's live secret for the period, `signer` the memory challenger's key and `log` its
    session log. `trusted` is the verifier's list of memory-challenger keys (by default the signer's), `schemes` the
    signature schemes it accepts and `protocol` the id it answers under (a test signer's: `TEST_PREFIX` and
    `pous_test_signer.SCHEMES`); `party` as for `audit`. A setup the verifier refuses raises `ValueError`, since no run
    of it exists."""
    return audit(node, setup_store(store, coins, point, workers=workers), coins, signer=signer, log=log, trusted=trusted,
                 schemes=schemes, protocol=protocol, prover=prover, clock=clock, party=party)


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


LEDGER_SCHEMAS = ("verity/challenger-auditor/v1", "verity/challenger-auditor/v2")
"""The schemas of the auditor's ledger entries (`sampled_proofs.challenger.LEDGER_TAGS`), whose copies `cite` cuts."""
_ENTRY = ["body", "kind", "prev", "schema", "seq", "session", "sig"]
_CLOSE = ["chain", "ended", "lines", "session", "signature"]


def _json(data: str | bytes) -> Any:
    """`data` parsed, or the `ValueError` saying why it isn't JSON, for the caller to keep the file as it is."""
    try:
        return json.loads(data)
    except ValueError as e:
        return e


def _entry(line: Any) -> dict | None:
    """`line` as an entry of the auditor's ledger (`sampled_proofs.challenger.ledger_text`'s form), else None."""
    if type(line) is not str or not line.startswith('{"body":'):
        return None
    e = _json(line)
    return e if isinstance(e, dict) and sorted(e) == _ENTRY and e["schema"] in LEDGER_SCHEMAS else None


def _ledger(lines: Sequence[Any]) -> list[dict] | None:
    entries = [_entry(x) for x in lines]
    return entries if entries and all(e is not None for e in entries) else None


def cited(entries: Sequence[dict], sessions: Collection[str]) -> list[int]:
    """The indices of a ledger's `entries` its window cites: every entry of `sessions` and the open and close of each
    period holding one of them (a period holds the sessions whose reserve, or one-step v2 open, falls in its lane between
    its open and its close, as `challenger.verify_periods` counts them). Raises `ValueError` naming a session of
    `sessions` the ledger doesn't open, or when `sessions` is empty: a reader of what's kept would find no entry of it."""
    if not sessions:
        raise ValueError("no challenger session is cited beside the auditor's ledger, so none of its entries would be kept")
    keep: set[int] = set()
    current: dict[Any, int] = {}
    held: set[int] = set()
    for i, e in enumerate(entries):
        b, kind = e["body"], e["kind"]
        if kind == "period-open":
            current[b.get("lane")] = i
        elif kind == "period-close":
            if current.pop(b.get("lane"), None) in held:
                keep.add(i)
        elif e["session"] in sessions:
            keep.add(i)
            if kind == "reserve" or (kind == "open" and e["schema"] == LEDGER_SCHEMAS[1] and "reserve" not in b):
                if (o := current.get(b.get("for") or b.get("by"))) is not None:
                    keep.add(o)
                    held.add(o)
    opened = {e["session"] for e in entries if e["kind"] == "open"}
    if missing := sorted(set(sessions) - opened):
        raise ValueError(f"the auditor's ledger opens no session {missing[0]}, which a challenger close record beside it "
                         f"cites ({len(entries)} entries, {len(missing)} of {len(sessions)} cited sessions unopened)")
    return sorted(keep)


def cite(files: Mapping[str, bytes]) -> dict[str, bytes]:
    """`files` (path -> bytes) with every copy of the auditor's ledger cut to the entries its window cites (`cited`). A
    copy is a `.jsonl` file each of whose lines is a ledger entry (infra's ``windows.jsonl``, ASCII as `ledger_text`
    writes it) or a list of such lines in a `.json` document (`window_salt.Issued.live`'s ``ledger``); the sessions it
    cites are those of the challenger close records (``{chain, ended, lines, session, signature}``, as `.json` files or
    inside one) under the copy's directory, its own document included. A copy with no close record beside it is kept
    whole: nothing cites its entries, so nothing says what a reader of it needs. A document holding a cut copy is
    written back as `json.dumps(doc, sort_keys=True)`, as ``salt.json`` is written; every other file is kept as it is.
    Raises `ValueError` naming the file when a copy would keep no entry of a session cited beside it, so the records
    never hold a ledger cut short of what their window cites. PoUW's verifier reads the whole ledger (every entry,
    chained from seq 0), so the uncut copy it reads stays on disk."""
    parsed = {name: _json(data) for name, data in files.items() if name.endswith(".json") and b'"session"' in data}
    docs = {name: doc for name, doc in parsed.items() if not isinstance(doc, ValueError)}
    closes: list[tuple[str, str]] = []

    def walk(name: str, v: Any) -> None:
        if isinstance(v, dict):
            if sorted(v) == _CLOSE and type(v["session"]) is str:
                closes.append((name, v["session"]))
            for x in v.values():
                walk(name, x)
        elif isinstance(v, list):
            for x in v:
                walk(name, x)

    for name, doc in docs.items():
        walk(name, doc)

    def cut(name: str, lines: list[str], entries: list[dict]) -> list[str] | None:
        scope = name.rsplit("/", 1)[0] + "/" if "/" in name else ""
        if not (sessions := {s for n, s in closes if n.startswith(scope)}):
            return None
        try:
            return [lines[i] for i in cited(entries, sessions)]
        except ValueError as e:
            raise ValueError(f"{name}: {e}") from None

    out = dict(files)
    for name, data in files.items():
        if name.endswith(".jsonl") and b"verity/challenger-auditor/" in data and data.isascii():
            lines = [x for x in data.decode("ascii").split("\n") if x]
            if (entries := _ledger(lines)) is not None and (kept := cut(name, lines, entries)) is not None:
                out[name] = "".join(x + "\n" for x in kept).encode()

    def copies(name: str, v: Any) -> Any:
        if isinstance(v, dict):
            return {k: copies(name, x) for k, x in v.items()}
        if isinstance(v, list):
            if (entries := _ledger(v)) is not None:
                return v if (kept := cut(name, v, entries)) is None else kept
            return [copies(name, x) for x in v]
        return v

    for name, doc in docs.items():
        if b"verity/challenger-auditor/" in files[name] and (kept := copies(name, doc)) != doc:
            out[name] = json.dumps(kept, sort_keys=True).encode()
    return out


def records(period_dir: str | os.PathLike, skip: Collection[str] = ("answers",)) -> bytes:
    """The period's records as one byte string, the store the node keeps for the period: each file under `period_dir`
    (but those under `skip`) in path order, every copy of the auditor's ledger cut to the entries its window cites
    (`cite`), framed `u32 len(path) ‖ path ‖ u64 len(bytes) ‖ bytes`."""
    root = Path(period_dir)
    files = {f.relative_to(root).as_posix(): f.read_bytes() for f in sorted(
        p for p in root.rglob("*") if p.is_file() and p.relative_to(root).parts[0] not in skip)}
    out = bytearray()
    for name, data in cite(files).items():
        out += len(name.encode()).to_bytes(4, "big") + name.encode() + len(data).to_bytes(8, "big") + data
    return bytes(out)


SETUP_WORKERS = 32
"""The processes the leg encodes a node's `C*` in, and the memory challenger's side makes its key in, at most, within the
CPUs each may run on (`setup_store`; `challenger`, from `challenger_party.main`)."""


def leg(period_dir: Path, node: str, out: Path) -> dict[str, Any]:
    """Space's leg of `benchmarks/all_protocols` (`leg(period_dir, node, out) -> answer`). By default the node's store is
    set up once, at the smallest point P2's certificates cover (`point_for`): `W` is the node's records at its
    first period (`records`, the auditor's ledger in them cut to the entries their window cites), encoded in up to
    `SETUP_WORKERS` processes and kept beside the periods (`pous-store`), and every period
    audits it with fresh coins, so an accepted period certifies `s_bits > 0` at the certificate's `eps`. With
    `{"pous": {"point": "small"}}` in `period.json`, each period sets the period's own records up at the small point
    instead, certifying nothing. The memory challenger is a party of its own on `period.json`'s `challenger_host`
    (`challenger_party.side_of`; on this node by default, a process whose key and session log are beside the periods)
    and the prover is here. With `pous-run-protocol` and `pous-setup` among the period's tools, the Lean verifier
    answers and must agree with the Python; it writes `out/pous.case.json` (onsite's PoUS case),
    `out/pous.json` and, in a period that set a store up, `out/pous-setup.case.json`. Which host ran each party, and
    whether `trusted-host`'s premise holds (`challenger_party.placement`), go in `pous.json` and in an accepted
    answer's profile (`placement`)."""
    doc = json.loads((Path(period_dir) / "period.json").read_text())
    tools, small = doc.get("tools") or {}, (doc.get("pous") or {}).get("point") == "small"
    workers = min(SETUP_WORKERS, len(os.sched_getaffinity(0)))
    party = challenger_party.side_of(doc, node, Path(period_dir).parent / "pous-challenger")
    where = challenger_party.placement(node, party.host)
    kept = Path(period_dir).parent / "pous-store"
    t = time.monotonic()
    if small or not (kept / "store.json").exists():
        store = records(period_dir)
        point = point_for(len(store), W, C0) if small else point_for(len(store))
        held = setup_store(store, secrets.token_bytes(64), point, workers=1 if small else workers)
        if not small:
            held.save(kept)
        fresh = True
    else:
        held, fresh = Store.load(kept), False
    setup_s = round(time.monotonic() - t, 2)
    t = time.monotonic()
    period = audit(node, held, secrets.token_bytes(64), party=party)
    seconds = round(time.monotonic() - t, 2)
    answer, lean = period.answer, {}
    if "pous-run-protocol" in tools:
        lean["run"] = serve(tools["pous-run-protocol"], [period.case])[0]
        if {k: v for k, v in lean["run"].items() if k != "reason"} != {k: v for k, v in answer.items() if k != "reason"}:
            raise RuntimeError(f"pous-run-protocol answered {lean['run']}, the Python {answer}")
        answer = lean["run"]
    if fresh and "pous-setup" in tools:
        lean["setup"] = serve(tools["pous-setup"], [period.setup_case])[0]
        if lean["setup"]["verdict"] != "accept":
            raise RuntimeError(f"pous-setup refused the setup the Python accepted: {lean['setup']}")
    if answer["profile"] is not None:
        answer = answer | {"profile": answer["profile"] | {"placement": where}}
    out.mkdir(parents=True, exist_ok=True)
    (out / "pous.case.json").write_text(json.dumps(period.case, sort_keys=True))
    if fresh:
        (out / "pous-setup.case.json").write_text(json.dumps(period.setup_case, sort_keys=True))
    q = held.point.params
    (out / "pous.json").write_text(json.dumps({
        "answer": answer, "blocks": q.blocks, "eps": str(period.eps), "lean": lean, "placement": where,
        "s_bits": period.s_bits, "seconds": seconds, "setup": {"fresh": fresh, "seconds": setup_s if fresh else 0},
        "w_bytes": q.w_bytes}, sort_keys=True, indent=1))
    return answer


__all__ = ["LEDGER_SCHEMAS", "PROTOCOL", "SETUP_WORKERS", "Period", "Prover", "SetupInputs", "Store", "answer", "audit",
           "case", "certificate", "certify", "challenger", "cite", "cited", "encode", "gaps", "honest", "leg", "point_for",
           "point_of", "profile", "records", "run", "serve", "setup_case", "setup_inputs", "setup_store", "tie"]
