"""The verifier and its challenger of a PoUB session over PoUS's P2 store (`p2.py`) as a party of their own,
so they run on a host the developer doesn't control (`trusted-device`) while the prover serves on its node.

The channel is a pair of byte streams carrying one JSON object of one key a line: the ssh session that starts this side
(`--stdio`, as a gate's verb starts it) or a TCP connection (`--listen`). The session has no round-trip floor, so ssh's
latency changes nothing in its verdict. The verifier holds its own registration, never the prover's word for it: the
memory attester's own record of a PoUS session's setup on this host (`--pous-record DIR --pous-session SID`,
`pous_record`: the store PoUS's session SID bound, its whole store the plan), or the auditor's registration file
(`--registration FILE`, `registration_json`); and it flips its coins from its own host's OS.
The record's `plan` is the units every pass reads (`plan_of`): the units holding a tensor's block, so not the store's
padding past the last tensor, which is PoUS's and which the decode never reads; without one, every unit.

- The prover opens with the request `{"request": {"deadline_ns", "node", "passes", "samples", "window_ns",
  "windows"}}`, whose node must be the registration's and the rest the auditor's session parameters as they are
  (`SESSION`, `params_of`): the request only restates them, so the prover chooses none. `main` holds its sessions to
  `SESSION` unless `--windows`, `--passes`, `--samples`, `--window-ns` or `--deadline-ns` names another value (a toy
  party's); the auditor's gate passes none of them. The Lean holds them too: its own pinned `P2Session.SESSION`, or
  the toy's that `hold` gives it, and the case's replay `SESSION`'s alone, so a toy party's case is refused there.
- The verifier answers `{"hold": {"registration"}}`: the registration with its plan, which the prover checks against
  its store and reads each pass in.
- It then drives the session on its clock (`Verifier`): `{"run": {"challenge", "i", "previous", "t"}}` asks for pass i
  of window t under the window's challenge, its value and the root it was opened after (`coins.Released`), after the
  root `previous`, answered `{"root": hex}` when the step that read it is done. Once a window's roots are final (the
  next window's first root, or its deadline for the last) it sends `{"open": {"draws", "t"}}`, answered `{"openings":
  [opening_json, ...]}`, and verifies them while the prover serves on.
- Once the session has its outcome, `{"end": {"commitment", "openings"}}`: the session's coin commitment and every
  released coin's opening (`audit.AuditRun.disclose`), against which the prover checks each challenge it was sent
  (`audit_prover.Prover.end`). The commitment is a function of every coin, so the prover gets it only after its last root.
- `{"case": {...}}`: the session's core case (`p2_case`), the log of every request the Lean answered with the
  registration, the request, the verifier's clock floor, the session id and the memory attester's close of the case
  (its registration's digest, its clock floor and its log), signed with its key (`Closer`), which
  `Poub.P2Protocol` checks and replays; it is witness, which the prover
  keeps private. A session with no memory attester to sign it has none.
- `{"answer": {...}}` ends the session (`answer`): the verdict, the failure and where, and on acceptance the credited
  reads `q_bytes`, the receipts less the wrong ones the pooled bound leaves (`pooled.bound` at δ = 2^-40), a unit's
  8,224 bytes each, over the session's span. `core_answer` puts it in the core interface's form.

The two sides are parties of the deployment layer (`MANIFEST`): the verifier and its challenger (`Verifier`, here) and
the prover (`verity/protocols/accounting/memory/poub/prover/p2_party_prover.py`'s `Prover`). Each message is one line, carried
whole as its body (`Lines`), so the Lean reads every line the prover sent as it sent it, and the verifier takes the line
it waits for whatever its key, one it can't name as `Malformed`. `session` and the prover's `prove` step each side over
the channel with PoUS's loop (`drive`, `challenger_session.run`), the verifier's side handed a failed write as its
channel failing (`broken`). The runtime's start of the verifier from a configuration and its deploy bindings (`start`,
`starts`) raises `NotImplementedError` (`GOAL`): no source the runtime gives a party holds its registration or session.

The verifier is the Lean, since the trusted base is the compiled Lean verifier, which parses bytes of
untrusted provenance, and Python is never trusted: the process `poub-verifier` (`p2_lean.py`,
`Poub.P2Session`) holds the registration file's bytes, recomputes the registration's digest from its fields, and
reads the prover's request and every root and openings line as the prover sent it, each opening's four blocks checked
on the registered vk root, and its answer is the session's alone (`LeanSession`). What the Lean takes as given is the
clock's readings and the coins the challenger (`Challenger`, `coins.local` on this host) releases after the roots the
Lean took, which decide nothing.

`python -m verity.protocols.accounting.memory.poub.verifier.p2_parties (--registration FILE | --pous-record DIR --pous-session SID)
(--stdio | --listen HOST:PORT [--sessions N]) [--device DIR] [--lean-exe EXE] [--challenger DIR
--session SID --trusted-sha512 HEX]` is the verifier's side (`main`), each session with fresh coins, the memory
attester's key and session log in `--challenger`'s directory (the key loaded by the `device_key` the caller hands
`main`, PoUS's device's `onsite.device_key` in the auditor's gate, refused unless `--trusted-sha512` pins it) signing the
close of the one session the auditor named `--session SID`, which is claimed in its log before the session's first coin, the
Lean built in its package (`lake build poub-verifier`) unless `--lean-exe` names a built one, or, without `lake`
on the PATH, the one already built there (`p2_lean.exe`). Listening, it prints `{"listening": port}` on stdout and no
record of any answer: the answer of record is the replay of the memory attester's signed close (`Poub.P2Protocol`), and
the session's own is the Lean's, sent to the prover alone. It is the shape of a party the auditor's gate starts over ssh
with its `--device` directory
(`verity/protocols/accounting/memory/poub/attester/live_verifier/host.py`'s party verbs), where it keeps each session's case's
digest, the chain its signed close signs, a line of `private/case-digests.txt` (`CASE_DIGESTS`, `case_digest`;
0700, the file 0600), and none of the case's values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import socket
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any, BinaryIO

from verity.channel import Body, Message, Schema
from verity.party import Finish, Identity, Received, Send, Timer
from verity.protocol import Channel, Manifest, ProtocolDef, Role, Start, Starter
from verity.protocols.proof.spot.verifier.profile import WINDOW_DELTA
from verity.protocols.accounting.memory.pous.attester import challenger_session as CS
from verity.protocols.accounting.memory.pous.attester import memory_attester as mc
from verity.protocols.accounting.memory.pous.verifier.schemes.p2 import P2Params

from . import audit, p2, p2_lean
from . import coins as C
from . import period as P
from .p2_lean import Lean, LeanError
from .receipts import Plan

MODULE = "verity.protocols.accounting.memory.poub.verifier.p2_parties"
PROTOCOL = "verity/remote-bandwidth/p2/v0"
ID, VERSION = PROTOCOL, 1
VERIFIER, PROVER = "verifier", "prover"
COIN_BYTES = 64
"""The session's secret, which its challenger's coins come from (`coins.local`)."""
LINE = Body(max=(1 << 32) - 1)
MANIFEST = Manifest(
    ID, VERSION, "verity.protocols.accounting.memory.poub.verifier.p2_parties:request_of",
    parties=(Role(VERIFIER, "auditor", needs=("clock", "coins", "signing-key")),
             Role(PROVER, "developer", per="server", needs=("store",))),
    channels=(Channel(VERIFIER, PROVER, ("Hold", "Run", "Open", "End", "Case", "Answer"), carries_coins=True),
              Channel(PROVER, VERIFIER, ("Request", "Root", "Openings", "Malformed"))),
    messages=(Schema("Request", 1, "structure", body=LINE), Schema("Hold", 1, "coins", body=LINE),
              Schema("Run", 1, "coins", body=LINE), Schema("Root", 1, "commitment", body=LINE),
              Schema("Open", 1, "coins", body=LINE), Schema("Openings", 1, "private", body=LINE),
              Schema("End", 1, "coins", body=LINE), Schema("Case", 1, "private", body=LINE),
              Schema("Answer", 1, "output", body=LINE), Schema("Malformed", 1, "private", body=LINE)))
LINES = {PROVER: {"request": "Request", "root": "Root", "openings": "Openings"},
         VERIFIER: {"hold": "Hold", "run": "Run", "open": "Open", "end": "End", "case": "Case", "answer": "Answer"}}
"""Each side's lines by their one key, the message types it sends."""
CASE_DIGESTS = "case-digests.txt"
"""The file under a party's ``private/`` that `_keep` appends each session's `case_digest` to."""
REQUEST_KEYS = ("deadline_ns", "node", "passes", "samples", "window_ns", "windows")
SESSION = {"deadline_ns": 10**9, "passes": 4, "samples": 1024, "window_ns": 20 * 10**9, "windows": 6}
"""The auditor's session parameters, which a request must carry as they are (`params_of`): the served bandwidth piece's
(`p2_stage/bandwidth.py`'s defaults, over the store the PoUS piece holds), six windows of at least 20 s,
four passes each, 1,024 openings a window and an allowance of 1 s. At δ = 2^-40 an accepted session's ε is
δ + 6,144/2^64 (`EPS_TERMS`), about 9.1e-13, which leaves out Σ_t adv_t (SHA-512 stream as PRF: not priced, not
assumed)."""
DELTA = float(WINDOW_DELTA)
"""PoUB-P2's δ, the proof system's per-window δ (`profile.WINDOW_DELTA`, 2^-40; `P2Session.DELTA` in the Lean): each
window's profile δ and the pooled bound's.  It is not PoUB's `audit.DELTA`, and not at an error of 2^-100."""
#: The samples a thread of the Lean checks at once (`P2Session.checkOpeningsChunked`).
CHUNK = 64
#: what an accepted session's `eps` sums (`core_answer`)
EPS_TERMS = ("δ and the draws' rejected first words, δ + Σ draws/2^64, the terms `Poub.Protocol.certify` sums; "
             "not the sketch's term at units (`SketchBoundP2`, drafted), and not Σ_t adv_t (SHA-512 stream as PRF: "
             "not priced, not assumed)")
#: what an accepted session's `gamma` is against, and what it and `q_bytes` are before (`core_answer`)
GAMMA_TERMS = ("μ_timing + μ_sample only, 1 − q_bytes/(R · span), R the read rate `r_bytes_per_s`; q_bytes and γ are "
               "before μ_chip, λ, μ_rebuild, μ_host, μ_link and μ_carrier, which `audit.ledger` does not take off a P2 "
               "record")
_HEX = re.compile(r"(?:[0-9a-f]{2})*")


def send(f: BinaryIO, doc: Mapping[str, Any]) -> None:
    f.write(json.dumps(dict(doc), sort_keys=True).encode() + b"\n")
    f.flush()


def _pairs(kvs: list) -> dict:
    if len({k for k, _ in kvs}) != len(kvs):
        raise ValueError("an object names a key twice")
    return dict(kvs)


def _refused(text: str) -> Any:
    raise ValueError(f"{text} is not an integer or a JSON constant")


def _strings(doc: Any) -> Iterator[str]:
    if isinstance(doc, str):
        yield doc
    elif isinstance(doc, list):
        for x in doc:
            yield from _strings(x)
    elif isinstance(doc, dict):
        for k, v in doc.items():
            yield k
            yield from _strings(v)


def line_json(b: bytes) -> Any:
    """A line's JSON as the prover alone reads it (`P2Session.lineJsonOf`): ValueError for bytes that aren't UTF-8 JSON,
    or name a key twice in an object, or hold a number with a fraction or an exponent, `NaN` or `Infinity`, or a string
    with a surrogate or U+FFFD (which Lean's reader makes of a lone surrogate escape)."""
    try:
        doc = json.loads(b.decode(), object_pairs_hook=_pairs, parse_float=_refused, parse_constant=_refused)
    except RecursionError:
        raise ValueError("a message nests too deep") from None
    if any(c == "\ufffd" or "\ud800" <= c <= "\udfff" for s in _strings(doc) for c in s):
        raise ValueError("a string holds a surrogate or U+FFFD")
    return doc


def message(line: bytes, key: str | None = None) -> Any:
    """The message `line` carries (its one key `key`'s value, if given; `P2Session.messageOf`): ConnectionError for a
    line its sender didn't end, ValueError for one that isn't JSON as `line_json` reads it, an object of one key
    (`key`)."""
    if not line.endswith(b"\n"):
        raise ConnectionError("the peer closed mid-session")
    doc = line_json(line[:-1])
    if not isinstance(doc, dict) or len(doc) != 1:
        raise ValueError("a message is a JSON object of one key")
    if key is None:
        return doc
    if key not in doc:
        raise ValueError(f"expected {key!r}, got {next(iter(doc))!r}")
    return doc[key]


def recv(f: BinaryIO, key: str | None = None) -> Any:
    """The next message's value (its one key `key`, if given)."""
    return message(f.readline(), key)


def _unhex(x: Any) -> bytes:
    """Lowercase hex's bytes (`Flock.unhex`), else ValueError."""
    if type(x) is not str or not _HEX.fullmatch(x):
        raise ValueError("not lowercase hex")
    return bytes.fromhex(x)


def _hexes(xs: Any) -> tuple[bytes, ...]:
    if type(xs) is not list:
        raise ValueError("not a list")
    return tuple(_unhex(x) for x in xs)


def _nat(doc: Mapping[str, Any], k: str) -> int:
    v = doc[k]
    if type(v) is not int or v < 0:
        raise ValueError(f"{k} is a natural number")
    return v


def registration_json(r: p2.Registration, plan: Plan | None = None) -> dict[str, Any]:
    c = r.codec
    out = {"blocks_per_segment": c.blocks_per_segment, "c0": c.c0, "commitment": r.commitment.hex(),
           "digest": r.digest.hex(), "expander": c.expander, "node": r.node.hex(), "root": r.root.hex(),
           "rows": r.rows, "salt": r.salt.hex(), "segments": c.segments, "w": c.w}
    return out if plan is None else out | {"plan": [list(run) for run in plan.runs]}


def plan_of(doc: Mapping[str, Any], reg: p2.Registration) -> Plan:
    """The plan of `registration_json`'s record `doc`, its `plan` runs `[first, count]` over `reg`'s units, or every
    unit without one; refused (ValueError) unless it is a plan of them."""
    runs = doc.get("plan")
    if runs is None:
        return Plan.whole(reg.blocks)
    if not isinstance(runs, list) or not all(isinstance(r, list) and len(r) == 2 and all(type(x) is int for x in r)
                                             for r in runs):
        raise ValueError("a registration's plan is a list of runs [first, count]")
    return Plan(tuple(tuple(r) for r in runs), reg.blocks)


def registration_of(doc: Mapping[str, Any]) -> p2.Registration:
    """`registration_json`'s inverse (`P2Session.heldOf`), refused if its `digest` (when given) isn't the fields'."""
    if type(doc["expander"]) is not str:
        raise ValueError("expander is a string")
    params = P2Params(_nat(doc, "w"), _nat(doc, "c0"), _nat(doc, "blocks_per_segment"), _nat(doc, "segments"),
                      doc["expander"])
    reg = p2.Registration(_unhex(doc["node"]), p2.codec_of(params), _unhex(doc["commitment"]), _unhex(doc["salt"]),
                          _unhex(doc["root"]), _nat(doc, "rows"))
    if "digest" in doc and _unhex(doc["digest"]) != reg.digest:
        raise ValueError("the registration's digest is not its fields'")
    return reg


def registration_of_setup(record: Mapping[str, Any]) -> p2.Registration:
    """The registration of the PoUS setup the memory attester kept for one session (`challenger_party.keep`): the
    request's node, the setup's P2 parameters, commitment and salt, and the session's binding as the vk root, at
    sketch-sha512's rows; ValueError for a record without a setup, or one `challenger_session.setup_of` refuses."""
    request = record["request"]
    s = CS.setup_of(request)
    if s is None:
        raise ValueError("the challenger's record of the session has no setup")
    q = s["params"]
    params = P2Params(q["w"], q["c0"], q["blocks_per_segment"], q["segments"], q["expander"])
    return p2.Registration(request["node"].encode(), p2.codec_of(params), _unhex(s["commitment"]), _unhex(s["salt"]),
                           _unhex(request["binding"]))


def pous_record(directory: Path, session_id: str) -> bytes:
    """The registration file's bytes (`registration_json`, every unit its plan) of the setup the memory attester whose
    device is `directory` kept for the session `session_id` (64 lowercase hex), on this host (`registration_of_setup`)."""
    if not re.fullmatch(r"[0-9a-f]{64}", session_id):
        raise ValueError("a session id is 64 lowercase hex digits")
    doc = json.loads((directory / CS.SETUPS / f"{session_id}.json").read_text())
    if doc.get("session_id") != session_id:
        raise ValueError(f"the record names the session {doc.get('session_id')}, not {session_id}")
    return json.dumps(registration_json(registration_of_setup(doc)), sort_keys=True).encode()


def opening_json(o: p2.Opening) -> dict[str, Any]:
    return {"blocks": [b.hex() for b in o.blocks], "leaf": o.leaf.hex(), "leaf_salt": o.leaf_salt.hex(),
            "pass": o.pass_index, "paths": [[x.hex() for x in path] for path in o.paths], "position": o.position,
            "receipt_path": [x.hex() for x in o.receipt_path]}


def opening_of(doc: Any) -> p2.Opening | None:
    """`opening_json`'s inverse (`P2Session.openingOf`), None for anything else (the session rejects it
    `malformed`)."""
    try:
        if (not isinstance(doc, dict) or type(doc["pass"]) is not int or type(doc["position"]) is not int
                or type(doc["paths"]) is not list):
            return None
        return p2.Opening(doc["pass"], doc["position"], _hexes(doc["blocks"]),
                          tuple(_hexes(path) for path in doc["paths"]), _unhex(doc["leaf"]),
                          _unhex(doc["leaf_salt"]), _hexes(doc["receipt_path"]))
    except (KeyError, ValueError):
        return None


def request_of(node: bytes, *, windows: int, passes: int, samples: int, window_ns: int,
               deadline_ns: int) -> dict[str, Any]:
    return {"deadline_ns": deadline_ns, "node": node.hex(), "passes": passes, "samples": samples,
            "window_ns": window_ns, "windows": windows}


def params_of(reg: p2.Registration, request: Mapping[str, Any],
              auditor: Mapping[str, int] = SESSION) -> audit.Params:
    """The session's parameters, the auditor's (`auditor`, `SESSION`'s keys), refused unless the prover's request is
    for the registration's node and carries each of the auditor's values as it is, by name. The Lean's
    `P2Session.paramsOf` is this under the session parameters its process holds (`hold`), with these refusals' words
    (`P2Session.heldTo`), and its case's replay is this under `SESSION` (`P2Protocol.replay`)."""
    if not isinstance(request, dict) or sorted(request) != sorted(REQUEST_KEYS):
        raise ValueError(f"a request is {list(REQUEST_KEYS)}")
    if _unhex(request["node"]) != reg.node:
        raise ValueError("the request's node is not the registration's")
    if any(type(request[k]) is not int or request[k] < 1 for k in REQUEST_KEYS if k != "node"):
        raise ValueError("a request's deadline_ns, passes, samples, window_ns and windows are positive integers")
    if off := [k for k in SESSION if request[k] != auditor[k]]:
        raise ValueError("; ".join(f"the request's {k} is not the auditor's ({request[k]}, not {auditor[k]})"
                                   for k in off))
    return audit.Params(blocks=reg.blocks, deadline_ns=auditor["deadline_ns"], windows=auditor["windows"],
                        samples=auditor["samples"], delta=DELTA, min_passes=auditor["passes"],
                        window_ns=auditor["window_ns"], codec=reg.codec, receipt=reg.receipt, layout=reg.layout,
                        rows=reg.rows)


def core_answer(a: Mapping[str, Any], r_bytes_per_s: int | None = None) -> dict[str, Any]:
    """The session's answer `a` (`answer`, as the verifier sent it) in the core interface's form (`period.answer`'s),
    its commitment the registration's digest. Accepted, its public outputs are the credited reads `q_bytes` over the
    span `span_ns` at `eps` (`EPS_TERMS`), and with `r_bytes_per_s` (R) `gamma` against it (`GAMMA_TERMS`); its profile
    is δ, the receipts, the wrong ones the pooled bound leaves, each window's samples and the store's units. Any other
    answer is a reject (`undecided` while a window is unverified) with its failure, where, and its reason, and neither.
    ValueError for an answer of another protocol."""
    if a.get("protocol") != PROTOCOL:
        raise ValueError(f"not a {PROTOCOL} answer: {a.get('protocol')!r}")
    reg = a.get("registration")
    out: dict[str, Any] = {"commitment": reg if type(reg) is str and _HEX.fullmatch(reg) else None, "profile": None,
                           "protocol": PROTOCOL, "public": None, "reason": None}
    verdict, why = a.get("verdict"), a.get("reason")
    if verdict == "accept" and (a.get("span_ns") or 0) > 0:
        delta, span, q = Fraction(a["delta"]), a["span_ns"], a["q_bytes"]
        public: dict[str, Any] = {"eps": str(delta + Fraction(sum(a["draws"]), 2**64)), "q_bytes": q, "span_ns": span}
        profile: dict[str, Any] = {
            "delta": str(delta), "draws": a["draws"], "eps_terms": EPS_TERMS, "plan_units": a.get("plan_units", a["units"]),
            "receipts": a["receipts"], "unit_bytes": a["unit_bytes"], "units": a["units"], "verifier_host": None if a.get("host") is None else mc.UNNAMED,
            "windows": len(a["windows"]), "wrong": a["wrong_at_most"]}
        if r_bytes_per_s is not None:
            public |= {"gamma": str(1 - Fraction(q * 10**9, r_bytes_per_s * span)), "r_bytes_per_s": r_bytes_per_s}
            profile["gamma_terms"] = GAMMA_TERMS
        return out | {"profile": profile, "public": public, "verdict": "accept"}
    if verdict == "undecided":
        return out | {"reason": f"the session has a window not yet verified{f': {why}' if why else ''}",
                      "verdict": "undecided"}
    if verdict == "reject":
        where = f"{a.get('failure')} at window {a.get('window')}, draw {a.get('at')}"
        return out | {"reason": f"the session was rejected: {where}{f' ({why})' if why else ''}", "verdict": "reject"}
    return out | {"reason": f"the session has no accepted answer: {verdict}{f': {why}' if why else ''}",
                  "verdict": "reject"}


class _Refused(Exception):
    """The session is refused before its first coin."""


class _Ended(Exception):
    """The session has its outcome: a root or a window rejected, or the Lean answered otherwise than it goes on."""

    def __init__(self, reason: str | None = None) -> None:
        super().__init__(reason)
        self.reason = reason


class Challenger:
    """The session's challenger on the verifier's host (`coins.local`, on coins from `secret`) and the clock's schedule
    it keeps, which decide nothing: each coin is released in order after the last root the Lean took (the
    registration's window root before any), a window's sample's coins after its last root once it holds the auditor's
    passes, and a window's boundary is due `window_ns` after its challenge left and its deadline `deadline_ns` after
    its boundary. `released` is the challenge the last step released, None for none, and `sample` the coins of the
    sample the last close released, whole, None for none."""

    def __init__(self, registration: p2.Registration, auditor: Mapping[str, int], secret: bytes) -> None:
        self.coins = C.local(registration.digest, 0, auditor["windows"], P._Stream(secret, registration.node),
                             samples=auditor["samples"])
        self.auditor, self.latest = auditor, self.coins.window
        self.roots: list[list[bytes]] = []
        self.released_ns: list[int] = []
        self.boundary_ns: list[int | None] = []
        self.released: C.LiveOpening | None = None
        self.sample: bytes | None = None

    def _challenge(self, t: int, now: int) -> C.LiveOpening:
        self.released = self.coins.challenge(t, self.latest)
        self.roots.append([])
        self.released_ns.append(now)
        self.boundary_ns.append(None)
        return self.released

    def start(self, now: int) -> None:
        self.sample = None
        self._challenge(0, now)

    def _unended(self) -> int | None:
        return next((t for t, b in enumerate(self.boundary_ns) if b is None), None)

    def boundary(self, now: int) -> None:
        t = self._unended()
        self.released = self.sample = None
        if t is None:
            return
        self.boundary_ns[t] = now
        if t + 1 < self.auditor["windows"]:
            self._challenge(t + 1, now)

    def root(self, t: int, root: bytes) -> None:
        self.roots[t].append(root)
        self.latest = root

    def close(self, t: int) -> None:
        rs = self.roots[t] if 0 <= t < len(self.roots) else []
        self.released = None
        self.sample = self.coins.sample(t, rs[-1]) if len(rs) >= self.auditor["passes"] else None

    def next_boundary_ns(self) -> int | None:
        t = self._unended()
        return None if t is None else self.released_ns[t] + self.auditor["window_ns"]

    def deadline(self, t: int) -> int | None:
        b = self.boundary_ns[t]
        return None if b is None else b + self.auditor["deadline_ns"]

    def disclose(self) -> tuple[C.LiveCommitment, tuple[C.LiveOpening, ...]]:
        """The coin commitment and every coin released, opened, family by family (`audit.AuditRun.disclose`)."""
        c = self.coins
        return c.commitment, tuple(o for label, _ in c.commitment.families for o in c.opened(label))


def _why(line: bytes, key: str) -> str | None:
    """Why the prover's `line` is out of form as a `key` line, for the answer's reason; None when it isn't."""
    try:
        message(line, key)
    except (ConnectionError, ValueError) as e:
        return f"the prover's side failed: {type(e).__name__}: {e}"
    return None


class LeanSession:
    """A P2 session decided by the Lean verifier alone (session `k` of `lean`, `Poub.P2Session`), on the coins its
    `challenger` releases: each event's result is the Lean's, a Lean that fails answering `{"error"}`."""

    def __init__(self, lean: Lean, k: int, registration: p2.Registration, challenger: Challenger,
                 request: dict[str, Any], not_before_ns: int = 0) -> None:
        self.lean, self.k, self.registration, self.challenger = lean, k, registration, challenger
        self.request, self.not_before_ns = request, not_before_ns
        self.why: str | None = None
        self.log: list[dict[str, Any]] = []
        """Every request after the session opened, as `Poub.P2Protocol` replays them (`p2_case`), each coin
        whole, as the challenger released it, where the Lean is sent its first `coins.COIN_BYTES`."""

    @classmethod
    def open(cls, lean: Lean, registration: p2.Registration, line: bytes, secret: bytes, not_before_ns: int,
             auditor: Mapping[str, int] = SESSION) -> LeanSession:
        """The session the prover's request `line` asks for, on coins from `secret`: `_Refused` where the Lean refuses
        it, as for a request off the auditor's parameters it holds (`hold`)."""
        try:
            got = lean.call({"op": "p2_session", "bytes": len(line), "not_before_ns": not_before_ns}, line)
        except LeanError as e:
            raise _Refused(f"the Lean verifier failed: {e}") from None
        if not isinstance(got, dict) or type(got.get("session")) is not int:
            why = got.get("refused") if isinstance(got, dict) else None
            raise _Refused(why if type(why) is str else f"the Lean verifier answered {got}")
        return cls(lean, got["session"], registration, Challenger(registration, auditor, secret),
                   message(line, "request"), not_before_ns)

    def _ask(self, request: dict[str, Any], body: bytes = b"") -> Any:
        try:
            return self.lean.call(request, body)
        except LeanError as e:
            return {"error": str(e)}

    def _event(self, event: dict[str, Any]) -> Any:
        c, sample = self.challenger.released, self.challenger.sample
        whole = sample if sample is not None else None if c is None else c.coin
        coin = sample if sample is not None else None if c is None else C.coin(c)
        self.log.append({"coin": None if whole is None else whole.hex(), "event": event, "op": "p2_event"})
        return self._ask({"op": "p2_event", "session": self.k, "event": event,
                          "coin": None if coin is None else coin.hex()})

    def start(self, now: int) -> tuple[Any, C.LiveOpening | None]:
        """The session's first coin at `now`: the result, and the challenge to send."""
        self.challenger.start(now)
        return self._event({"kind": "start", "now": now}), self.challenger.released

    def boundary(self, now: int) -> tuple[Any, C.LiveOpening | None]:
        """The next unended window's boundary at `now`: the result, and the next window's challenge, or None."""
        self.challenger.boundary(now)
        return self._event({"kind": "boundary", "now": now}), self.challenger.released

    def root(self, t: int, i: int, line: bytes, now: int) -> Any:
        """Root i of window t from the prover's `line`, arriving at `now`; the challenger's next coin follows it once
        the Lean takes it."""
        self.log.append({"i": i, "line": line.hex(), "now": now, "op": "p2_root", "t": t})
        got = self._ask({"op": "p2_root", "session": self.k, "t": t, "i": i, "now": now, "bytes": len(line)}, line)
        if got == {"ok": True}:
            self.challenger.root(t, bytes.fromhex(message(line, "root")))
        else:
            self.why = _why(line, "root")
        return got

    def close(self, t: int, now: int) -> Any:
        """Window t's samples at `now`."""
        self.challenger.close(t)
        return self._event({"kind": "close", "now": now, "t": t})

    def openings(self, t: int, line: bytes) -> Any:
        """Window t's openings from the prover's `line`."""
        self.log.append({"line": line.hex(), "op": "p2_openings", "t": t})
        got = self._ask({"op": "p2_openings", "session": self.k, "t": t, "bytes": len(line)}, line)
        if not (isinstance(got, dict) and "accepted" in got):
            self.why = _why(line, "openings")
        return got

    def fail(self) -> Any:
        """The session rejected `malformed` at the window filling: the prover is gone."""
        self.log.append({"op": "p2_fail"})
        return self._ask({"op": "p2_fail", "session": self.k})

    def _answer(self) -> Any:
        return self._ask({"op": "p2_answer", "session": self.k})

    def end(self, reason: str | None) -> dict[str, Any]:
        """The session's answer, the Lean's (`P2Session.answer`) with `reason`, the window filling failed first while
        the session is undecided (the prover gone); undecided, saying why, where the Lean gives none."""
        got = self._answer()
        if isinstance(got, dict) and got.get("verdict") == "undecided":
            self.fail()
            got = self._answer()
        if isinstance(got, dict) and got.get("protocol") == PROTOCOL and "error" not in got:
            return got | {"reason": reason}
        return {"protocol": PROTOCOL, "registration": self.registration.digest.hex(), "verdict": "undecided",
                "reason": f"the Lean verifier gave no answer: {got}"}


def hold(lean: Lean, data: bytes, chunk: int = CHUNK,
         auditor: Mapping[str, int] = SESSION) -> tuple[p2.Registration, Plan]:
    """The registration file's bytes `data`, held by the Lean, which holds every request to the auditor's session
    parameters `auditor`, every `chunk` samples checked on a thread (`P2Session.heldOf`): the registration
    (`registration_of`) and the plan the Lean holds. The Lean is sent `auditor` only when it isn't `SESSION` (a toy
    party's): otherwise it holds its own pinned `P2Session.SESSION`. ValueError where the Lean refuses it."""
    reg = registration_of(json.loads(data.decode()))
    request = {"op": "p2_hold", "bytes": len(data), "chunk": chunk}
    if dict(auditor) != SESSION:
        request["session"] = {k: auditor[k] for k in SESSION}
    try:
        got = lean.call(request, data)
    except LeanError as e:
        raise ValueError(f"the Lean verifier refuses the registration: {e}") from None
    if not isinstance(got, dict) or "plan" not in got:
        raise ValueError(f"the Lean verifier refuses the registration: {got}")
    return reg, Plan(tuple(tuple(r) for r in got["plan"]), reg.blocks)


class Closer:
    """The memory attester that signs a session's close (`p2_case.close_of`): its `signer`
    (`memory_attester.Signer`, the key it signs PoUS's sessions with), its session log (`log`,
    `memory_attester.SessionLog`), and the session id the auditor gave it for this session (`session`, 32 bytes),
    which `claim` records in the log before the session's first coin and refuses once it is there."""

    def __init__(self, signer: Any, log: Any, session: bytes) -> None:
        if len(session) != 32:
            raise ValueError("a session id is 32 bytes")
        self.signer, self.log, self.session = signer, log, bytes(session)

    def claim(self) -> str | None:
        """The id claimed in the log: None, or why it is refused."""
        try:
            self.log.claim(self.session)
        except mc.SessionLogError as e:
            return f"the memory attester refuses the session id: {e}"
        return None


def line_of(key: str, value: Any) -> bytes:
    """The line `send` writes for `{key: value}`."""
    return json.dumps({key: value}, sort_keys=True).encode() + b"\n"


class Lines:
    """The parties' messages as the channel's bytes (`MANIFEST`): each message's body is its line, written whole, and
    a line read is named by its one key (`names`, the types the other side sends). A side that is `strict` raises as
    `recv` does on a line it can't name; the verifier's side names it `Malformed`, and reads the end of the channel
    or an OSError on it as the runtime's `Closed` (`how` is `end`, or the error)."""

    def __init__(self, names: Mapping[str, str], strict: bool) -> None:
        self.names, self.strict = names, strict

    def write(self, f: BinaryIO, m: Message) -> None:
        f.write(m.body)
        f.flush()

    def read(self, f: BinaryIO) -> Message:
        if self.strict:
            line = f.readline()
            (key, _), = message(line).items()
            if key not in self.names:
                raise ValueError(f"no message {key!r}")
            return Message(self.names[key], body=line)
        try:
            line = f.readline()
        except OSError as e:
            return Message("Closed", {"how": f"{type(e).__name__}: {e}"})
        if line == b"":
            return Message("Closed", {"how": "end"})
        try:
            (key, _), = message(line).items()
        except (ConnectionError, ValueError):
            key = None
        return Message(self.names.get(key, "Malformed"), body=line)


class _Pair:
    """The channel's two streams as one, which `challenger_session.run` reads and writes."""

    def __init__(self, f_in: BinaryIO, f_out: BinaryIO) -> None:
        self.readline, self.write, self.flush = f_in.readline, f_out.write, f_out.flush


def broken(e: OSError) -> Message:
    """A failed write to the prover, as the runtime tells a party its channel failed."""
    return Message("Closed", {"how": f"{type(e).__name__}: {e}"})


def drive(party: Any, params: Mapping[str, Any], identity: Identity, f_in: BinaryIO, f_out: BinaryIO, *, peer: str,
          clock: Callable[[], int] | None = None, coins: Callable[[int], bytes] | None = None,
          wait: Callable[[Callable[[], int], int], None] = P._sleep_until) -> Finish:
    """`party` against `peer` over the channel (`f_in`, `f_out`) until it finishes, stepped by
    `challenger_session.run` on the codec of its side (`Lines`): its finish. The verifier's side is handed a failed
    write as the prover's side failing (`broken`)."""
    side = VERIFIER if peer == PROVER else PROVER
    codec = Lines(LINES[peer], strict=side == PROVER)
    done, _ = CS.run(party, params, identity, _Pair(f_in, f_out), codec, peer=peer, clock=clock, coins=coins,
                     wait=wait, broken=broken if side == VERIFIER else None)
    return done


class Verifier:
    """The verifier and its challenger: one session, decided by the Lean alone (`LeanSession`), on its clock and
    coins. Its capabilities are `lean` (holding `registration` and `plan`: `hold`), the auditor's session parameters
    `auditor`, its `host`'s name, the memory attester signing the close (`closer`, or None), where its case is kept
    (`keep`, or None) and a progress `log`.
    It claims the closer's session id, flips the session's coins, and reads the prover's request; then each pass is
    asked for in order under the plan, each window's openings sampled once its roots are final and verified, and the
    session ends with its coins disclosed, its case and the answer sent, finishing with `{"answer", "case"}`. Every
    line the prover sends reaches the Lean whole whatever its type, the end of the channel as an empty line; a channel
    that fails rejects the session `malformed`, as the prover's side failing."""

    def start(self, params: Mapping[str, Any], identity: Identity) -> list[Any]:
        c = identity.capabilities
        self.lean, self.registration, self.closer = c["lean"], c["registration"], c.get("closer")
        self.plan = Plan.whole(self.registration.blocks) if c.get("plan") is None else c["plan"]
        self.auditor, self.host = c.get("auditor", SESSION), c["host"]
        self.keep: Callable[[dict[str, Any]], None] | None = c.get("keep")
        self.log: Callable[[str], None] = c.get("log") or (lambda _m: None)
        self.twin: LeanSession | None = None
        self.coins = b""
        self.due: Callable[[bytes], list[Any]] | None = self._request
        self.then: Callable[[int], list[Any]] | None = None
        self.at: int | None = None
        self.after: Callable[[], list[Any]] = self._next
        self.gone = False
        if self.closer is not None and (why := self.closer.claim()) is not None:
            return self._refused(why)
        return [Send("coins", Message("CoinsDraw", {"n": COIN_BYTES}))]

    def on(self, r: Received) -> list[Any]:
        m = r.message
        if m.type == "Coins":
            self.coins = m.body
            return []
        if m.type in ("ClockReading", "ClockTimer"):
            if self.then is None or (m.type == "ClockTimer" and m.fields["at"] != self.at):
                return []
            then, self.then, self.at = self.then, None, None
            return then(m.fields["ns"]) if then == self._open else self._step(then, m.fields["ns"])
        if m.type in ("Closed", "Refused"):
            how = m.fields["how"] if m.type == "Closed" else f"refused {m.fields['name']}: {m.fields['detail']}"
            if how == "end" and self.due is not None:
                return self._line(b"")
            if how == "end" and self.twin is not None:
                self.gone = True
                return []
            self.then = self.at = None
            why = f"the prover's side failed: {how}"
            return self._refused(why) if self.twin is None else self._end(why)
        return self._line(m.body)

    def _line(self, line: bytes) -> list[Any]:
        due, self.due = self.due, None
        if due is None:
            return self._end("out-of-phase") if self.twin is not None else self._refused("out-of-phase")
        return due(line) if due == self._request else self._step(due, line)

    def _step(self, f: Callable[..., list[Any]], *args: Any) -> list[Any]:
        try:
            return f(*args)
        except _Ended as e:
            return self._end(e.reason)
        except (OSError, ValueError, KeyError, TypeError, audit.SessionError) as e:
            return self._end(f"the prover's side failed: {type(e).__name__}: {e}")

    def _read(self, then: Callable[[int], list[Any]]) -> list[Any]:
        self.then = then
        return [Send("clock", Message("ClockRead"))]

    def _alarm(self, at: int, then: Callable[[int], list[Any]]) -> list[Any]:
        self.then, self.at = then, at
        return [Timer(at)]

    def _request(self, line: bytes) -> list[Any]:
        self.line = line
        return self._read(self._open)

    def _open(self, now: int) -> list[Any]:
        try:
            self.twin = LeanSession.open(self.lean, self.registration, self.line, self.coins, now, self.auditor)
        except _Refused as e:
            return self._refused(str(e))
        self.windows, self.passes, self.t, self.i = self.auditor["windows"], self.auditor["passes"], 0, 0
        hold = line_of("hold", {"registration": registration_json(self.registration, self.plan)})
        return [Send(PROVER, Message("Hold", body=hold)), *self._read(self._start)]

    def _start(self, now: int) -> list[Any]:
        got, self.challenge = self.twin.start(now)
        if got != {"released": 0}:
            raise _Ended("the session's first coin was refused")
        self.previous = self.registration.digest
        return self._run()

    def _expect(self, due: Callable[[bytes], list[Any]], sent: Message) -> list[Any]:
        """`sent` to the prover, then its next line due to `due`: at once, an empty line, if it has gone."""
        self.due = due
        return [Send(PROVER, sent), *(self._line(b"") if self.gone else [])]

    def _run(self) -> list[Any]:
        run = {"challenge": C.Released.of(self.challenge).to_json(), "i": self.i, "previous": self.previous.hex(),
               "t": self.t}
        return self._expect(self._root, Message("Run", body=line_of("run", run)))

    def _root(self, line: bytes) -> list[Any]:
        self.line = line
        return self._read(self._rooted)

    def _rooted(self, now: int) -> list[Any]:
        t, i = self.t, self.i
        if self.twin.root(t, i, self.line, now) != {"ok": True}:
            raise _Ended(self.twin.why)
        self.previous = self.twin.challenger.roots[t][i]
        self.log(f"window {t} pass {i}: root {self.previous.hex()[:16]}")
        if i == 0 and t > 0:
            self.after = self._next
            return self._read(self._close)
        return self._next()

    def _next(self) -> list[Any]:
        self.i += 1
        if self.i < self.passes:
            return self._run()
        return self._alarm(self.twin.challenger.next_boundary_ns(), self._boundary)

    def _boundary(self, now: int) -> list[Any]:
        got, self.challenge = self.twin.boundary(now)
        if not isinstance(got, dict) or "released" not in got:
            raise _Ended(f"window {self.t}'s boundary was refused")
        self.t, self.i = self.t + 1, 0
        if self.t < self.windows:
            return self._run()
        self.after = self._done
        return self._alarm(self.twin.challenger.deadline(self.windows - 1) + 1, self._close)

    def _close(self, now: int) -> list[Any]:
        t = self.t - 1
        got = self.twin.close(t, now)
        if not isinstance(got, dict) or "draws" not in got:
            raise _Ended
        self.draws = len(got["draws"])
        return self._expect(self._openings, Message("Open", body=line_of("open", {"draws": got["draws"], "t": t})))

    def _openings(self, line: bytes) -> list[Any]:
        t = self.t - 1
        v = self.twin.openings(t, line)
        self.log(f"window {t}: {self.draws} openings, "
                 f"{'accepted' if 'accepted' in v else 'rejected ' + str(v.get('failure'))}")
        if v != {"accepted": t}:
            raise _Ended(self.twin.why)
        return self.after()

    def _done(self) -> list[Any]:
        return self._end(None)

    def _refused(self, why: str) -> list[Any]:
        out = {"protocol": PROTOCOL, "registration": self.registration.digest.hex(), "verdict": "refused",
               "reason": why, "host": self.host}
        return [Send(PROVER, Message("Answer", body=line_of("answer", out))),
                Finish({"answer": out, "case": None}, kind="private")]

    def _end(self, reason: str | None) -> list[Any]:
        """The session's end (`reason`, None when it ran to its last window): the Lean's answer, the prover gone failing
        the window filling (`LeanSession.end`), its coins disclosed, its case unless no closer signs its close, and the
        answer, each sent while the prover is there to read them."""
        self.due = self.then = self.at = None
        twin = self.twin
        out = twin.end(reason) | {"host": self.host}
        commitment, openings = twin.challenger.disclose()
        case = None
        if self.closer is not None:
            from verity.commitments.identity import canonical_json_bytes

            from . import p2_case

            held = registration_json(self.registration, self.plan)
            head = p2_case.head(canonical_json_bytes(held), twin.not_before_ns)
            close = p2_case.close_of(self.closer.signer, self.closer.session, head, twin.log)
            case = p2_case.case_of(held, twin.request, twin.not_before_ns, twin.log, session=self.closer.session,
                                   trusted=[self.closer.signer.public_key], close=close)
            if self.keep is not None:
                self.keep(case)
        end = {"commitment": commitment.to_json(), "openings": [o.to_json() for o in openings]}
        sent = [Send(PROVER, Message("End", body=line_of("end", end)))]
        if case is not None:
            sent.append(Send(PROVER, Message("Case", body=line_of("case", case))))
        return [*sent, Send(PROVER, Message("Answer", body=line_of("answer", out))),
                Finish({"answer": out, "case": case}, kind="private")]


def session(f_in: BinaryIO, f_out: BinaryIO, registration: p2.Registration, *, coins: bytes, lean: Lean,
            plan: Plan | None = None, clock: Callable[[], int] = time.monotonic_ns,
            sleep_until: Callable[[Callable[[], int], int], None] = P._sleep_until,
            log: Callable[[str], None] = lambda _m: None, host: str | None = None,
            keep: Callable[[dict[str, Any]], None] | None = None,
            auditor: Mapping[str, int] = SESSION, closer: Closer | None = None) -> dict[str, Any]:
    """The verifier's side of one session over the channel (`f_in`, `f_out`), the `Verifier` party on `coins`, `clock`
    and `sleep_until`, decided by `lean`, which holds `registration` and `plan` (`hold`; every unit by default): the
    answer it sent last, naming the verifier's `host` (this one's name by default), its case given to `keep`. The
    `closer`'s session id is claimed before the request is read, and the session refused when the claim is. A line out
    of form or a closed channel rejects the session (its answer says why)."""
    capabilities = {"lean": lean, "registration": registration, "plan": plan, "auditor": auditor, "closer": closer,
                    "keep": keep, "log": log, "host": host or socket.gethostname()}
    done = drive(Verifier(), {}, Identity(VERIFIER, capabilities=capabilities), f_in, f_out, peer=PROVER, clock=clock,
                 coins=lambda _n: coins, wait=sleep_until)
    return done.outcome["answer"]


def case_digest(case: Mapping[str, Any]) -> str:
    """The one SHA-512 a verifier's device keeps of `case` (`p2_case`), as 128 lowercase hex digits: its close's
    chain, which the memory attester's signed close signs, over the case's head (its registration's digest and its
    clock floor) and every entry of its log (`p2_case.chain`)."""
    return str(json.loads(bytes.fromhex(case["evidence"]))["close"]["chain"])


def _keep(path: Path, case: Mapping[str, Any]) -> None:
    """`case_digest(case)` appended to `path` as one line, its directory 0700 and the file 0600: the verifier's own
    record of the case, the digest its signed close signs, which holds none of its values."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a") as f:
        f.write(case_digest(case) + "\n")
    os.chmod(path, 0o600)


def main(argv: Sequence[str] | None = None, *, device_key: Callable[[Path], mc.LmsSigner] | None = None) -> int:
    ap = argparse.ArgumentParser(prog=MODULE, description=__doc__.split("\n\n")[0])
    held = ap.add_mutually_exclusive_group(required=True)
    held.add_argument("--registration", type=Path, help="the auditor's registration with its plan (registration_json)")
    held.add_argument("--pous-record", type=Path, help="the memory attester's device on this host, whose record of "
                                                       "--pous-session's setup is the registration (pous_record)")
    ap.add_argument("--pous-session", help="with --pous-record: the PoUS session id (64 hex) whose setup is the store")
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument("--stdio", action="store_true", help="the channel is this process's stdin and stdout")
    where.add_argument("--listen", help="HOST:PORT to accept provers on (port 0: any, printed as {\"listening\"})")
    ap.add_argument("--sessions", type=int, default=1, help="with --listen: the sessions to serve, one after another")
    held_to = ap.add_argument_group("the auditor's session parameters, which a request must carry as they are "
                                    "(default: SESSION's, the served bandwidth piece's; the auditor's gate passes none)")
    for k, v in SESSION.items():
        held_to.add_argument("--" + k.replace("_", "-"), type=int, default=v, metavar="N")
    ap.add_argument("--device", type=Path, help="the party's directory, as the auditor's gate gives it: each "
                                                "session's case digest is kept in it")
    ap.add_argument("--lean-exe", type=Path, help=f"the built {p2_lean.EXE}, the Lean verifier (default: built "
                                                  "in its Lake package, or without lake on the PATH, the one built "
                                                  "there)")
    ap.add_argument("--challenger", type=Path, help="the memory attester's device on this host: its LMS key, which "
                                                    "the caller's `device_key` loads, and session log, which sign the "
                                                    "session's close (`Closer`)")
    ap.add_argument("--session", help="with --challenger: the session id (64 hex) the auditor gave the memory "
                                      "attester for this session; without them, no session has a case")
    ap.add_argument("--trusted-sha512", help="with --challenger: the SHA-512 (128 hex) of the key the configuration "
                                             f"pins for the close ({CLOSER_PARTY}'s for {COMPONENT}); the "
                                             "challenger's key is refused before any session unless it is that key")
    a = ap.parse_args(argv)
    if (a.pous_record is None) != (a.pous_session is None):
        ap.error("--pous-record and --pous-session go together")
    if not ((a.challenger is None) == (a.session is None) == (a.trusted_sha512 is None)):
        ap.error("--challenger, --session and --trusted-sha512 go together: the close is signed only under the pinned key")
    if a.trusted_sha512 is not None and not re.fullmatch(r"[0-9a-f]{128}", a.trusted_sha512):
        ap.error("--trusted-sha512 is a key's SHA-512, 128 lowercase hex digits")
    if a.session is not None and not re.fullmatch(r"[0-9a-f]{64}", a.session):
        ap.error("--session is a session id, 64 lowercase hex digits")
    if a.session is not None and a.listen is not None and a.sessions != 1:
        ap.error("--session names one session: --sessions 1")
    if a.challenger is not None and device_key is None:
        ap.error("--challenger needs the memory attester's key loader handed to main (device_key): run the party as "
                 "the auditor's gate does (live_verifier/host.py's bw_party)")
    closer = None
    if a.challenger is not None:
        signer = device_key(a.challenger)
        if hashlib.sha512(signer.public_key).hexdigest() != a.trusted_sha512:
            ap.error(f"the memory attester's key in {a.challenger} is not the one --trusted-sha512 pins for the close: "
                     "it signs no session's close")
        closer = Closer(signer, mc.SessionLog(a.challenger / "sessions"), bytes.fromhex(a.session))
    data = a.registration.read_bytes() if a.registration is not None else pous_record(a.pous_record, a.pous_session)
    cases = None
    if a.device is not None:
        a.device.mkdir(parents=True, exist_ok=True)
        cases = a.device / "private" / CASE_DIGESTS

    def log(m: str) -> None:
        print(f"p2_parties {time.strftime('%H:%M:%SZ', time.gmtime())}: {m}", file=sys.stderr, flush=True)

    def done(out: dict[str, Any]) -> None:
        log(f"answer {out['verdict']} {out.get('failure') or ''} q_bytes={out.get('q_bytes')} "
            f"host={socket.gethostname()}")

    auditor = {k: getattr(a, k) for k in SESSION}
    if any(v < 1 for v in auditor.values()):
        ap.error(f"the session's parameters are positive integers, not {auditor}")
    exe = p2_lean.exe(a.lean_exe)
    with Lean(exe) as lean:
        reg, plan = hold(lean, data, auditor=auditor)
    log(f"registration {reg.digest.hex()[:16]} ({reg.blocks} units, {plan.length} a pass) on "
        f"{socket.gethostname()}, of record in {p2_lean.EXE}; every session held to {auditor}")

    def kept(case: dict[str, Any]) -> None:
        if cases is not None:
            _keep(cases, case)

    def served(r: BinaryIO, w: BinaryIO) -> dict[str, Any]:
        """One session on a Lean process of its own, so that one the prover crashes ends only its session."""
        with Lean(exe) as lean:
            hold(lean, data, auditor=auditor)
            return session(r, w, reg, coins=secrets.token_bytes(64), lean=lean, plan=plan, log=log, keep=kept,
                           auditor=auditor, closer=closer)

    if a.stdio:
        done(served(sys.stdin.buffer, sys.stdout.buffer))
        return 0
    host, _, port = a.listen.rpartition(":")
    with socket.create_server((host, int(port))) as srv:
        print(json.dumps({"listening": srv.getsockname()[1]}), flush=True)
        for _ in range(a.sessions):
            conn, _ = srv.accept()
            with conn, conn.makefile("rb") as r, conn.makefile("wb") as w:
                out = served(r, w)
            done(out)
    return 0


GOAL = CS.ONSITE_GOAL
#: PoUB's P2 session's name in a deployment configuration: in its `protocols` and a binding's table
COMPONENT = "bandwidth"
#: its `[[pin]]` names PoUS's key: it signs the close with the memory attester's PoUS key
CLOSER_PARTY = "bandwidth-verifier"


def deployed_request(configuration: Mapping[str, Any], bound: Mapping[str, Any]) -> dict[str, Any]:
    """The prover's request (`request_of`) as a deployment starts it: the session parameters `SESSION`, and `node` the
    node the store was registered under, which PoUS's setup over the served store gives and no configuration states."""
    raise NotImplementedError(GOAL)


def starts() -> dict[str, Starter]:
    return {VERIFIER: start}


def start(configuration: Mapping[str, Any], bound: Mapping[str, Any], state: Path | None) -> Start:
    """Raises `NotImplementedError(GOAL)`: the runtime gives a party no registration, session id or held Lean yet."""
    raise NotImplementedError(GOAL)


PROTOCOLS = {ID: ProtocolDef(MANIFEST, {VERIFIER: Verifier})}


__all__ = ["CASE_DIGESTS", "CLOSER_PARTY", "COIN_BYTES", "COMPONENT", "Closer", "DELTA", "EPS_TERMS", "GAMMA_TERMS", "GOAL",
           "ID", "LINES", "Lines", "MANIFEST", "PROTOCOL", "PROTOCOLS", "PROVER", "SESSION", "Challenger", "LeanSession",
           "VERIFIER", "VERSION", "Verifier", "broken", "case_digest", "core_answer", "deployed_request",
           "drive", "hold", "line_of", "main", "message", "opening_json", "opening_of", "params_of",
           "plan_of", "pous_record", "recv", "registration_json", "registration_of",
           "registration_of_setup", "request_of", "send", "session", "start", "starts"]

if __name__ == "__main__":
    sys.exit(main())
