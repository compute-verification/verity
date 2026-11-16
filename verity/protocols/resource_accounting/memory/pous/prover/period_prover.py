"""The prover's side of one PoUS period on one node: the setup, the session with the memory challenger and the period's
subprotocol, each ending in the verifier's check (`verity.protocols.resource_accounting.memory.pous.verifier.period`).

`run(node, store, coins, signer=..., log=...)` plays every party in one process, in the protocol's order (`setup_store`
is steps 1 to 3, `audit` steps 4 and 5, so a node's store can be set up once and audited every period):

1. the prover commits to `W`, the node's `store` padded with zeros to the point's payload (`setup.commit`);
2. the verifier samples the salt after the commitment, from its live `coins` (`setup.sample_salt`);
3. the prover encodes `C*` (`P2.encode_segment`), and the verifier decodes it and checks `Dec(C*)` against the commitment
   (`setup.verifier_key_from_encoding`, the public-encoder setup; in Lean `verity/pous/p2/v4/setup`), publishing its root;
4. the memory challenger takes its table of `C*` (each block's `memory_challenger.block_digest`), samples its `k` indices
   from `coins` under a session id the auditor designates from them, and runs its timed session against the prover's
   answers (`prover`, by default the honest one that holds `C*`), signed under `signer`;
5. the verifier judges the run, setup's acceptance of `C*` publishing exactly the run's commitment (`tie`) and then
   `run.check_at`, and answers in the core interface's canonical form (`period.answer`); the case carries setup's own
   inputs.

Encoding is the cost: two 16,448-bit roots a block, about 3.7 s a block on one core in this pure-Python reference path,
so 16 core-minutes at 256 blocks and 550 core-hours at the deployed `2^19`; `encode` splits it
over processes. The toy's 6,240,150-byte store is 3,039 blocks there, set up in 367 s in 32 processes (3.86 core-seconds
a block). An audit is cheap: about 1.7 s at `2^19` blocks, mostly the table's root.

`subprotocol(period_dir, node, out, side_of=...)` is `benchmarks/all_protocols`' subprotocol for space: by default it sets the node's
store up once at `point_for`'s certified point and audits it every period, the coins fresh from the OS. The memory
challenger is a party of its own (`challenger_party`) on `period.json`'s `challenger_host`, with its own coins and a key
at S1's deployed parameters (`memory_challenger.LMS`) that `onsite.device_key` makes once on that host; the
prover and the verifier run on the node. `records` is the store it sets up: the period's files, the auditor's ledger in
them cut to the entries their window cites (`cite`).
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity.protocols.resource_accounting.memory.pous.attester import challenger_session
from verity.protocols.resource_accounting.memory.pous.attester import memory_challenger as mc
from verity.protocols.resource_accounting.memory.pous.verifier import run as R
from verity.protocols.resource_accounting.memory.pous.verifier import setup
from verity.protocols.resource_accounting.memory.pous.verifier.period import (
    C0,
    PROTOCOL,
    SESSION_DOMAIN,
    SETUP_WORKERS,
    W,
    Run,
    answer,
    case,
    deployment,
    point_for,
    point_json,
    point_of,
    serve,
    setup_case,
    setup_inputs,
    unjudged,
)
from verity.protocols.resource_accounting.memory.pous.verifier.schemes import p2 as P2


type Prover = Callable[[Sequence[bytes]], Callable[[int], bytes]]
"""The prover's side of the session: from the blocks of `C*` it was sent, its answer to each index."""

type Session = Callable[[challenger_session.Side, Mapping[str, Any], Callable[[int], bytes]], dict[str, Any]]
"""The prover's end of one session with a memory challenger of its own (`challenger_prover.session`): the challenger's
side, the request and the prover's answer to each index; the side's report."""


def honest(blocks: Sequence[bytes]) -> Callable[[int], bytes]:
    """The prover that holds every block of `C*`."""
    return lambda i: blocks[i]


@dataclass(frozen=True)
class EncodedStore:
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
    def load(cls, directory: str | os.PathLike) -> EncodedStore:
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


def setup_store(store: bytes, coins: bytes, point: R.Point | None = None, *, workers: int = 1) -> EncodedStore:
    """Setup on the node (module docstring, steps 1 to 3): `W` is `store` padded to the point's payload, the salt sampled
    from the verifier's `coins` after the commitment, `C*` encoded in `workers` processes, and the public-encoder check of
    it (`setup.verifier_key_from_encoding`), which raises `ValueError` on a setup it refuses."""
    point = point_for(len(store)) if point is None else point
    q = point.params
    if len(store) > q.w_bytes:
        raise ValueError(f"the store is {len(store)} bytes, past the point's payload of {q.w_bytes}")
    w = bytes(store) + bytes(q.w_bytes - len(store))
    commitment = setup.commit(q, w)
    salt = setup.sample_salt(coins, commitment)
    encoded = encode(q, salt, w, workers)
    root = setup.verifier_key_from_encoding(q, None, commitment, salt, encoded).root
    blocks = tuple(b for seg in encoded for b in seg)
    return EncodedStore(point, R.Setup(q.blocks, commitment, root, salt), blocks,
                        b"".join(mc.block_digest(b) for b in blocks), setup_case(q, commitment, salt, encoded))


def audit(node: str, held: EncodedStore, coins: bytes, *, signer: mc.Signer | None = None, log: mc.SessionLog | None = None,
          trusted: Collection[bytes] | None = None, schemes: Mapping[str, mc.Verify] = mc.SCHEMES,
          protocol: str = PROTOCOL, prover: Prover = honest, clock: Callable[[], int] = time.perf_counter_ns,
          party: challenger_session.Side | None = None, session: Session | None = None) -> Run:
    """One period's audit of the store `held` on `node` (module docstring, steps 4 and 5): the memory challenger's
    session under the id the auditor designates from `coins` and the setup's root, then the run's check and answer, the
    tie on the setup case `held` keeps, which the run's case carries. With `party`, the memory challenger is a party of
    its own (`challenger_party`): its side runs where `party` (a `challenger_session.Side`) starts it, with its own
    coins, key and session log, and the prover answers it over a socket in `session` (`challenger_prover.session`, which
    the caller hands in); `signer`, `log` and `clock` are unused, and the verifier's list is by default the key the side
    reports."""
    point, published = held.point, held.published
    q = point.params
    if party is None:
        session_id = challenger_session.coin_session_id(coins, SESSION_DOMAIN, node, published.root)
        tree = mc.CoinTree.draw(coins, session_id, q.blocks, point.k)
        challenger = mc.MemoryChallenger(tree, signer, log=log, limit_ns=point.limit_ns, block_bytes=q.block_bytes,
                                         binding=published.root, clock=clock)
        signed = challenger.run(prover(held.blocks))
        session, key = R.ChallengePhaseRecord(signed, challenger.commitment, challenger.openings()), signer.public_key
    else:
        if session is None:
            raise ValueError("a memory challenger of its own takes the prover's session (`challenger_prover.session`)")
        got = session(party, challenger_session.request_of(node, point, published.root), prover(held.blocks))
        session_id, key = bytes.fromhex(got["session_id"]), bytes.fromhex(got["public_key"])
        session = R.ChallengePhaseRecord.from_json(got["session"])
    trusted = (key,) if trusted is None else tuple(trusted)
    return Run(answer(point, schemes, trusted, published, held.digests, session, session_id, protocol,
                      setup_inputs=setup_inputs(held.setup_case)),
               case(point, session_id, trusted, published, held.digests, session, held.setup_case), held.setup_case)


def run(node: str, store: bytes, coins: bytes, *, signer: mc.Signer | None = None, log: mc.SessionLog | None = None,
        point: R.Point | None = None, trusted: Collection[bytes] | None = None,
        schemes: Mapping[str, mc.Verify] = mc.SCHEMES, protocol: str = PROTOCOL, prover: Prover = honest,
        clock: Callable[[], int] = time.perf_counter_ns, workers: int = 1,
        party: challenger_session.Side | None = None, session: Session | None = None) -> Run:
    """The period's PoUS on `node`, setup and audit (module docstring): `store` is what the node keeps, `coins` the
    verifier's and the memory challenger's live secret for the period, `signer` the memory challenger's key and `log` its
    session log. `trusted` is the verifier's list of memory-challenger keys (by default the signer's), `schemes` the
    signature schemes it accepts and `protocol` the id it answers under (a test signer's: `TEST_PREFIX` and
    `pous_test_signer.SCHEMES`); `party` and `session` as for `audit`. A setup the verifier refuses raises `ValueError`,
    since no run of it exists."""
    return audit(node, setup_store(store, coins, point, workers=workers), coins, signer=signer, log=log, trusted=trusted,
                 schemes=schemes, protocol=protocol, prover=prover, clock=clock, party=party, session=session)


LEDGER_SCHEMAS = ("verity/challenger-auditor/v1", "verity/challenger-auditor/v2")
"""The schemas of the auditor's ledger entries (`spot.challenger.LEDGER_TAGS`), whose copies `cite` cuts."""
_ENTRY = ["body", "kind", "prev", "schema", "seq", "session", "sig"]
_CLOSE = ["chain", "ended", "lines", "session", "signature"]


def _json(data: str | bytes) -> Any:
    """`data` parsed, or the `ValueError` saying why it isn't JSON, for the caller to keep the file as it is."""
    try:
        return json.loads(data)
    except ValueError as e:
        return e


def _entry(line: Any) -> dict | None:
    """`line` as an entry of the auditor's ledger (`spot.challenger.ledger_text`'s form), else None."""
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
            reserves = kind == "reserve" or (kind == "open" and e["schema"] == LEDGER_SCHEMAS[1] and "reserve" not in b)
            if reserves and (o := current.get(b.get("for") or b.get("by"))) is not None:
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


def subprotocol(period_dir: Path, node: str, out: Path, *, side_of: Callable[..., challenger_session.Side],
        session: Session) -> dict[str, Any]:
    """Space's subprotocol of `benchmarks/all_protocols` (`subprotocol(period_dir, node, out, side_of=...) -> answer`). By default the
    node's store is set up once, at the smallest point P2's certificates cover (`point_for`): `W` is the node's records
    at its first period (`records`, the auditor's ledger in them cut to the entries their window cites), encoded in up
    to `SETUP_WORKERS` processes and kept beside the periods (`pous-store`), and every period audits it with fresh
    coins, so an accepted period certifies `s_bits > 0` at the certificate's `eps`. With `{"pous": {"point": "small"}}`
    in `period.json`, each period sets the period's own records up at the small point instead, certifying nothing. The
    memory challenger is a party of its own on `period.json`'s `challenger_host` (`side_of(period, node, key, local)`,
    which the caller that starts the run hands in, the runtime's `Side.of`; on this node by default, a process whose key
    and session log are beside the periods) and the prover is here, answering it in `session`
    (`challenger_prover.session`, which the caller hands in too). With `pous-run-protocol` and `pous-setup` among the
    period's tools, the Lean verifier answers and must agree with the Python; without `pous-run-protocol` the Python's
    reject or undecided stands and its accept is undecided (`unjudged`). It writes `out/pous.case.json` (onsite's PoUS
    case), `out/pous.json` and, in a period that set a store up, `out/pous-setup.case.json`. Which host ran each party,
    whether `trusted-device`'s premise holds for the session's key (`challenger_session.placement`, under
    `period.json`'s `deployment`), and that setup was checked by the developer, not the auditor
    (`challenger_session.SETUP`), go in `pous.json` and in an accepted answer's profile (`placement`)."""
    doc = json.loads((Path(period_dir) / "period.json").read_text())
    tools, small = doc.get("tools") or {}, (doc.get("pous") or {}).get("point") == "small"
    workers = min(SETUP_WORKERS, len(os.sched_getaffinity(0)))
    party = side_of(doc, node, "pous", challenger_session.command(Path(period_dir).parent / "pous-challenger"))
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
        held, fresh = EncodedStore.load(kept), False
    setup_s = round(time.monotonic() - t, 2)
    t = time.monotonic()
    period = audit(node, held, secrets.token_bytes(64), party=party, session=session)
    seconds = round(time.monotonic() - t, 2)
    key_sha512 = hashlib.sha512(bytes.fromhex(period.case["params"]["trusted"][0])).hexdigest()
    where = challenger_session.placement(node, party.host, key_sha512, **deployment(doc))
    answer, lean = period.answer, {}
    if "pous-run-protocol" in tools:
        lean["run"] = serve(tools["pous-run-protocol"], [period.case])[0]
        if {k: v for k, v in lean["run"].items() if k != "reason"} != {k: v for k, v in answer.items() if k != "reason"}:
            raise RuntimeError(f"pous-run-protocol answered {lean['run']}, the Python {answer}")
        answer = lean["run"]
    else:
        answer = unjudged(answer)
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
        "s_bits": period.public("s_bits"), "seconds": seconds,
        "setup": {"fresh": fresh, "seconds": setup_s if fresh else 0}, "w_bytes": q.w_bytes}, sort_keys=True, indent=1))
    return answer


__all__ = ["LEDGER_SCHEMAS", "EncodedStore", "Prover", "Session", "audit", "cite", "cited", "encode", "honest", "subprotocol",
           "records", "run", "setup_store"]
