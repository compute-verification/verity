"""A served run's record composed off the served node, each part from the source that produced its verdict.

    served_sources.py record --serve-dir D --replay DIR --private DIR --out OUT --gate GATE --key-sha512 PIN
        --main FILE --controlled-by developer|auditor --published-coin FILE [--coin-put FILE]
        [--sampled-causes FILE] [--verifier-commit SHA] [--exact-totals private|public]

`served_verdicts.py record` composes from each piece's answer file as the served node wrote it, so an edit to such a
file on that node changes its row. This composes on another host, from the copy D of the run's serve dir there, and
reads no answer file of the served node for a part it verifies again. The settings (the window, its engine steps, the
served node, the product, the audit's directory) are read from D's ``private/compose/compose.env``, so no window's
number is on a command line.

**PoUW (Pearl+): verified again off the served node.** The row is the verdict `case_replay.sh` took from the run's
case (``--replay``, its output: `REPLAY`, the Python answer and the Lean answer), by Python and by Lean. `replay_held`
holds that replay to the inputs here: the coin as drawn, the window record W and the case's parts are the files it
hashed, the Python verifier opened no file outside the case, and the Lean check found no difference. The conditions of
record are `served_verdicts.pearlplus_of_record`'s, on a private piece directory built here from the case's manifests,
the coin as drawn and the two answers: the served node's `development.json`, `pearlplus-answer.json`,
`pearlplus-lean.json` and `piece.json` are never opened, so an edit to one changes nothing in the row. A tie that
names its weights (a version 1 W: ``public.tie.weights``) puts `WEIGHTS_COST` on the row. The row's verdict is a bare
``accept`` only when every record it reads accepted: its tie to the window reads the audit of record's registration and
summary, so beside an audit that did not accept it is `ACCEPT_UNAUDITED`, which the composed verdict never counts as an
accept.

**The gate's three reads.** The salt session, the draw session, the auditor's ledger and the host's pins log are
read live from the challenger's gate (`GateReads`: ``fetch SID``, ``ledger``, ``pins``, each asked once; any other word
is refused before the gate is asked, so nothing here writes, draws or opens). The conditions of record read the
gate's ledger, not the served node's copy of it: that copy could be cut short and so hide a later open on the same
commitment, which the rule that a draw is taken once must see. The record says, for each read, that it was read live
and when, with its length and SHA-512, and `held_to_gate` holds each copy the served node handed off to it (its
ledger, line for line the start of the gate's; each session file it kept, byte for byte the gate's). A difference is the
row's refusal with both sides named, never a warning. The gate's key is the file ssh reads by its path in ``--gate``;
nothing here reads, copies or prints it.

**Sampled proofs: the audit of record, as it answered.** The row is `served_verdicts.sampled_proofs_row` on that
audit's own ``summary.json`` and ``window/audit.json`` under D. Nothing is rerun, and a rejection stays one;
``--sampled-causes`` (a JSON list) puts the lane's diagnosis beside the audit's own reason, marked as that.

**The inference-only policy: the served node's own answer.** Not recomputed, because the block count it is judged on
stays on that node (`INFERENCE_FROM`, by source: one today); its part says so.

**What the record says of itself.** ``composed_on``: where it was composed (`WHERE`), ``served_node: false``, who
controls the host (``--controlled-by``: required, one of `CONTROLLERS`, never a default) and `SOURCES_FROM`.
``composed``: the verdict `served_verdicts.record` composes over the rows
(a rejected audit rejects it; an accept of every row is left pending, since one part is not verified again here), the
sentence `parts_said` (which part was verified again off the served node and which was not), and `NOT_COMPOSITE`: no
theorem of the remote protocol is cited and no block record is written.

**What leaves.** ``--out`` gets the public record (``record/verdicts.json``), its ``RECORD`` line (``verdicts.txt``)
and the coin record, each in its public form (`served_verdicts.public_doc`); the full record, the coin as drawn, the
manifests and the gate's reads stay in ``--private``, a new 0700 directory outside ``--out``. The case's blobs are
opened tile values, witness: they are read where they are and never copied. The public record is `public_report`'s:

- **A row is its judgment** (`JUDGMENT`): its verdict with its ε and δ, where its answer is from and what it rests on.
  Its exact totals (`TOTALS`: the answer and its public quantity) stay in the private record unless
  ``--exact-totals public``, which also lets the two Pearl+ answers leave.
- **A row that did not accept gives the kind of its reason** (`kind_of`): a rejected audit's own words name drawn
  members by index and their session paths, and a copy that differs from the gate's is named by hashes of both sides.
  Those stay in the private record.
- **No hash of a file whose redacted form is published** (`Withheld`): the SHA-512 of the coin as drawn, published
  beside the redacted coin record, would give the window's number to anyone who tries each number in its place, and
  the same holds of the ledger, the pins, the sessions' files and the record itself. The record names the redacted
  coin record's hash (held to ``--published-coin``, the run's published one) and says the other stays on this host
  (`COIN_HELD`); the gate's reads leave as when each was read and how the served node's copy was held to it; and
  `write` refuses, before it writes anything, a public file that holds such a digest or its first 16 hex digits.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shlex
import sys
import tarfile
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import served_verdicts as S

#: where such a record is composed, in its ``composed_on.where``, by who controls the host
OFF_NODE = "off the served node, on a CPU node the lane runs; not the auditor's host"
WHERE = {"developer": OFF_NODE, "auditor": "off the served node, on an auditor's host"}
#: who controls the composing host (``--controlled-by``), in its ``composed_on.controlled_by``: a fixed vocabulary, which
#: the command line requires and never defaults
CONTROLLERS = {"developer": "the developer's side, which also ran the served node; no auditor controls this host",
               "auditor": "an auditor's host: the developer's side controls neither it nor what runs on it"}
#: what it composes from, in its ``composed_on.from``
SOURCES_FROM = ("each part from the source its answer_from names; no answer file the served node wrote is read for a "
                "part verified again here, and the inference-only verdict is that node's own answer")
#: Pearl+'s row's ``answer_from``, what it adds once the gate was read (`GATE_HELD`), and its clause in the composed reason
REPLAYED_FROM = ("the replay off the served node, by Python (served_passes.py verify) and by Lean (PouwCheck.PearlPlus), "
                 "of the passes rebuilt from the run's case on the coin as drawn; no answer file of the served node is "
                 "read")
GATE_HELD = "; its coins held to the challenger's sessions, ledger and pins as read live from the gate"
REPLAYED_SAID = "re-verified off the served node (Python and Lean)"
#: sampled proofs' row's ``answer_from``
AUDIT_FROM = ("the audit of record's own summary.json and window/audit.json, read where that audit left them (each "
              "one's SHA-512 in files); nothing is rerun, retried or recomputed here")
#: what the causes beside a refused audit's own reason are
CAUSES_BY = "the lane's diagnosis of the audit of record's refusal, beside the audit's own reason; no verifier's finding"
#: the inference-only part's ``answer_from``, by its source
INFERENCE_FROM = {"served-node": "the served node's own answer, not recomputed, because the block count stays on that node"}
#: what a tie that names its weights costs (`Core.Remote`'s account of a version 1 W)
WEIGHTS_COST = ("for a version 1 W the tie is by the record alone: nothing holds PoUW's committed weights to the weights "
                "the record registers, and the remote protocol doesn't establish that PoUW's credited work and the "
                "audited inference read the same weights")
#: PoUW's verdict when both verifiers accept and the audit its tie reads did not: never a bare accept, and no accept to
#: the composed verdict (`served_verdicts.record` takes only ``accept`` as one)
ACCEPT_UNAUDITED = "accept, tie unaudited"
#: what the tie to the window rests on when the audit it reads did not accept
TIE_UNAUDITED = ("W is tied to the window by the audit of record's registration and summary, and that audit did not "
                 "accept: the tie says which window the credited passes belong to, not that its inference was audited")
#: a row's fields the public record carries: its judgment with its ε and δ, where its answer is from and what it rests on
JUDGMENT = ("protocol", "owner", "verdict", "commitment", "delta", "epsilon", "runs", "answer_from", "said", "weights",
            "tie_rests_on", "coin", "causes")
#: a row's exact totals (its answer and that answer's public quantity), public only under ``--exact-totals public``
TOTALS = ("quantity", "answer")
TOTALS_HELD = "exact totals are in the private record alone (--exact-totals private)"
#: what the public record says of a reason it gives the kind of, and the kind of one that has no plain words
DETAIL_HELD = ("the kind of reason alone: the words it was given in, which can name drawn members, session paths or "
               "hashes of private files, are in the private record")
NO_KIND = "not accepted"
#: what ``composed`` says of the parts together
NOT_COMPOSITE = ("parts, each verified where its clause says, not the remote protocol's composite: this record cites no "
                 "theorem of the remote protocol, is no verity/remote/v1 accept and writes no block record")
#: why an accept of every row composes pending here
NOT_ALL_HERE = ("every row accepts where it was verified, and the inference-only verdict is not verified again off the "
                "served node: no composite accept is taken here")
#: what the record says of the coin as drawn
COIN_HELD = ("the coin as drawn is private (its pass paths name the window's number) and stays on the lane's node with "
             "its SHA-512: published beside the redacted record, that hash would give the number to anyone who tries each "
             "one in its place; published_sha512 is the redacted coin record's own")
#: `case_replay.sh`'s record of a replay, and its schema
REPLAY, REPLAY_SCHEMA = "replay.json", "verity-pouw/pearl-plus-case-replay/v1"
#: the case's parts (`case_passes.SCHEMA`'s file)
PARTS = "parts.json"
READ_LIVE = "live from the gate"


class Refused(ValueError):
    """Why a row can't be taken from its source; the row's ``error``."""


def sha512(path: Path) -> str:
    h = hashlib.sha512()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def kept(d: Path) -> Path:
    """``d``, made with each directory above it that isn't there, each for its owner alone (0700)."""
    for p in [*reversed(d.parents), d]:
        if not p.exists():
            p.mkdir(mode=0o700)
    return d


def settings(path: Path) -> dict[str, str]:
    """The hand-off's settings (``compose.env``, one ``KEY=value`` a line as `printf %q` wrote it), by key."""
    out: dict[str, str] = {}
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        words = shlex.split(line)
        if len(words) != 1 or "=" not in words[0]:
            raise ValueError(f"{path.name} line {n} is not one KEY=value")
        k, _, v = words[0].partition("=")
        out[k] = v
    return out


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class Withheld:
    """The digests the public record must not hold: SHA-512 and SHA-256 of each file whose redacted form is published,
    by this writer (the coin as drawn, the record, its line) or by the run's own record (the ledger, the pins, the
    sessions' files). Such a hash beside the redacted file gives back what was cut to anyone who tries each value in
    its place. A file that is its own public form has nothing cut, and its hash is not withheld. `found` names each
    one a text holds, by its first 16 hex digits."""

    def __init__(self) -> None:
        self._names: dict[str, str] = {}

    def add(self, name: str, data: bytes) -> None:
        text = data.decode(errors="replace")
        if text.encode() == data and S.public_doc(text) == text:
            return
        for h in (hashlib.sha512, hashlib.sha256):
            self._names.setdefault(h(data).hexdigest()[:16], name)

    def add_file(self, name: str, path: Path) -> None:
        if path.is_file():
            self.add(name, path.read_bytes())

    def found(self, text: str) -> list[str]:
        return sorted({name for head, name in self._names.items() if head in text})


_PLAIN = re.compile(r"[A-Za-z][A-Za-z ,.'+-]*")
_CUT = re.compile(r" \(|;")


def kind_of(reason: Any) -> str:
    """The kind of a reason, for the public record: its leading clauses (split at ``": "``) while each is plain words
    (letters, spaces, commas, full stops, apostrophes and hyphens: no digit, path, quote or bracket), the last cut at a
    parenthesis or a semicolon. So a rejected audit's ``audit record: coins, proved (drawn units not proved: ...);
    lean-verify/m...`` is ``audit record: coins, proved``; a reason with no plain leading clause is `NO_KIND`."""
    kept = []
    for clause in str(reason or "").split(": "):
        plain = _CUT.split(clause)[0].strip()
        if not _PLAIN.fullmatch(plain):
            break
        kept.append(plain)
        if plain != clause.strip():
            break
    return ": ".join(kept)[:300] or NO_KIND


class GateReads:
    """The challenger gate's answers to its read verbs, each asked once and kept with when it was read: ``fetch SID``
    (a session's tree), ``ledger`` (the auditor's ledger) and ``pins`` (the host's pins log). Any other word raises
    ValueError before the gate is asked: nothing here writes, draws or opens. Called as
    `served_verdicts.pearlplus_coins` and `pearlplus_draw` call their ``fetch``, so both checks and `held_to_gate` read
    one ledger. ``ask`` asks the gate (`served_verdicts._gate_out` on ``--gate``); ``now`` is the clock."""

    def __init__(self, ask: Callable[..., bytes], now: Callable[[], str] = utc_now) -> None:
        self._ask, self._now = ask, now
        self._got: dict[str, tuple[bytes, str]] = {}

    @staticmethod
    def verb(words: Sequence[str]) -> str:
        from verity.core.protocols.pouw import salt as PS

        if tuple(words) in (("ledger",), ("pins",)):
            return words[0]
        if len(words) == 2 and words[0] == "fetch" and isinstance(words[1], str) and PS.SESSION.fullmatch(words[1]):
            return f"fetch {words[1]}"
        raise ValueError(f"the gate is asked to read alone (fetch SID, ledger, pins), never {' '.join(map(str, words))[:80]!r}")

    def __call__(self, *words: str) -> bytes:
        verb = self.verb(words)
        if verb not in self._got:
            self._got[verb] = (self._ask(*words), self._now())
        return self._got[verb][0]

    def said(self) -> dict[str, dict[str, Any]]:
        """Each read made, by verb: that it was read live, when, its length and its SHA-512."""
        return {verb: {"read": READ_LIVE, "read_utc": at, "bytes": len(data), "sha512": hashlib.sha512(data).hexdigest()}
                for verb, (data, at) in sorted(self._got.items())}

    def raw(self) -> dict[str, bytes]:
        """Each read made, by verb: the gate's answer."""
        return {verb: data for verb, (data, _at) in self._got.items()}


def _lines(data: bytes) -> list[bytes]:
    return [x for x in data.split(b"\n") if x.strip()]


def _start_of(name: str, mine: bytes, gate: bytes, at: str) -> dict[str, Any]:
    """The served node's copy ``mine`` of a log held to the gate's ``gate`` (read at ``at``): line for line its start.
    `Refused` naming both sides at the first line that differs, or where the copy is the longer."""
    a, b = _lines(mine), _lines(gate)
    ha, hb = hashlib.sha512(mine).hexdigest(), hashlib.sha512(gate).hexdigest()
    sides = (f"the served node's copy ({len(a)} lines, sha512 {ha[:16]}...) and the gate's, read {at} ({len(b)} lines, "
             f"sha512 {hb[:16]}...)")
    if len(a) > len(b):
        raise Refused(f"the served node's copy of {name} is longer than the one the gate reads now: {sides}")
    for i, (x, y) in enumerate(zip(a, b, strict=False), 1):
        if x != y:
            raise Refused(f"the served node's copy of {name} is not the start of the one the gate reads now: line {i} is "
                          f"sha512 {hashlib.sha512(x).hexdigest()[:16]}... in the node's copy and "
                          f"{hashlib.sha512(y).hexdigest()[:16]}... at the gate; {sides}")
    return {"lines": len(a), "sha512": ha, "held": "line for line the start of the gate's", "gate_lines_after": len(b) - len(a)}


def held_to_gate(reads: GateReads, node: Path, sessions: Sequence[str]) -> dict[str, dict[str, Any]]:
    """The gate's reads for ``sessions`` (`GateReads.said`), each with the copy the served node handed off under
    ``node`` (its piece's directory: ``ledger.jsonl``, ``pins.jsonl``, ``session/<id>/...``) held to it: a log line for
    line the start of the gate's, a session's file byte for byte the gate's. A copy the hand-off doesn't hold is said to
    be none. `Refused` on the first difference, naming both sides."""
    got = {"ledger": reads("ledger"), "pins": reads("pins"), **{f"fetch {s}": reads("fetch", s) for s in sessions}}
    said = reads.said()
    for name, what in (("ledger", "the auditor's ledger"), ("pins", "the host's pins log")):
        f = node / f"{name}.jsonl"
        said[name]["lines"] = len(_lines(got[name]))
        said[name]["node_copy"] = (_start_of(what, f.read_bytes(), got[name], said[name]["read_utc"]) if f.is_file()
                                   else f"none: the served node's hand-off holds no copy of {what}")
    for s in sessions:
        verb, d = f"fetch {s}", node / "session" / s
        files = sorted(p for p in d.rglob("*") if p.is_file()) if d.is_dir() else []
        if not files:
            said[verb]["node_copy"] = "none: the served node's hand-off holds no file of this session"
            continue
        with tarfile.open(fileobj=io.BytesIO(got[verb])) as t:
            members = {m.name: m for m in t.getmembers() if m.isfile()}
            held = {}
            for p in files:
                rel = p.relative_to(d).as_posix()
                mine, m = p.read_bytes(), members.get(f"{s}/{rel}")
                theirs = t.extractfile(m).read() if m is not None else None
                ha = hashlib.sha512(mine).hexdigest()
                if theirs != mine:
                    hb = "no such file" if theirs is None else f"sha512 {hashlib.sha512(theirs).hexdigest()[:16]}..."
                    raise Refused(f"the served node's copy of session {s}'s {rel} is not the gate's: sha512 {ha[:16]}... in "
                                  f"the node's copy and {hb} in the session's tree the gate gives, read "
                                  f"{said[verb]['read_utc']}")
                held[rel] = ha
        said[verb]["node_copy"] = {"files": held, "held": "each byte for byte the gate's"}
    return said


def replay_held(replay: Path, coin: Path, w: Path, parts: Path, commit: str | None = None) -> dict[str, Any]:
    """`case_replay.sh`'s record under ``replay`` held to the inputs here, and what the row says of it: the coin as
    drawn ``coin``, the window record ``w`` and the case's ``parts`` are the files it hashed, the coin's bytes didn't
    change, the Python verifier opened no file outside the case, the Lean check found no difference from the Python,
    and, where ``commit`` names one, the verifiers' tree is that commit. `Refused` on the first that fails. The coin's
    own hash is checked and not returned (`COIN_HELD`)."""
    try:
        doc = json.loads((replay / REPLAY).read_text())
    except (OSError, ValueError) as e:
        raise Refused(f"no replay record ({REPLAY}: {type(e).__name__}: {e})") from e
    if not isinstance(doc, dict) or doc.get("schema") != REPLAY_SCHEMA:
        raise Refused(f"{REPLAY} is not a replay's record ({REPLAY_SCHEMA})")
    for key, path, what in (("coin_sha512", coin, "the coin as drawn"), ("w_sha512", w, "the window record W"),
                            ("parts_sha512", parts, "the case's parts")):
        try:
            have = sha512(path)
        except OSError as e:
            raise Refused(f"{what} can't be read ({type(e).__name__})") from e
        if doc.get(key) != have:
            raise Refused(f"the replay was not run on {what} here: its {key} is another file's")
    if doc.get("coin_unchanged") is not True:
        raise Refused("the replay doesn't say the coin record's bytes were unchanged")
    opened = doc.get("opened")
    if not isinstance(opened, dict) or opened.get("opened_outside_the_case") != []:
        outside = opened.get("opened_outside_the_case") if isinstance(opened, dict) else None
        raise Refused(f"the replay's Python verifier opened files outside the case, or its opens weren't recorded: {outside!r}"[:600])
    py, lean = doc.get("python") or {}, doc.get("lean") or {}
    if commit and doc.get("verifier_commit") != commit:
        raise Refused(f"the replay's verifiers are tree {str(doc.get('verifier_commit'))[:12]}'s, not {commit[:12]}'s")
    if py.get("verdict") == "accept" and lean.get("verdict") == "accept" and (
            py.get("exit"), lean.get("exit"), lean.get("differ")) != (0, 0, []):
        raise Refused(f"the replay's verifiers accept but didn't end clean: python exit {py.get('exit')}, lean exit "
                      f"{lean.get('exit')}, differing on {lean.get('differ')!r}"[:600])
    return {"schema": doc["schema"], "verifier_commit": doc.get("verifier_commit"), "parts_sha512": doc["parts_sha512"],
            "w_sha512": doc["w_sha512"], "exe_sha256": doc.get("exe_sha256"), "coin_unchanged": True,
            "python": {k: py.get(k) for k in ("exit", "verdict", "reason", "tie", "draw", "line")},
            "lean": {k: lean.get(k) for k in ("exit", "verdict", "reason", "tie", "differ")},
            "opened": {k: v if isinstance(v, int) else len(v) for k, v in opened.items() if isinstance(v, (int, list))}}


def coin_held(coin: Path, answer: Mapping[str, Any], published: Path | None) -> dict[str, Any]:
    """The coin as drawn ``coin`` held to Pearl+'s verdict ``answer`` and to the coin record the run published
    (``published``): ``{published_sha512, as_drawn, reason}``, the SHA-512 of the coin's public form, `COIN_HELD`, and
    None or why the verdict is not held to the published coin. The verdict's run commitment and draw source must be the
    coin's, and the coin's public form must be the published record byte for byte."""
    got: dict[str, Any] = {"published_sha512": None, "as_drawn": COIN_HELD, "reason": None}
    try:
        text = coin.read_text()
        rec = json.loads(text)
        public = S.public_doc(text).encode()
        got["published_sha512"] = hashlib.sha512(public).hexdigest()
        if (answer.get("commitment"), (answer.get("profile") or {}).get("source")) != (rec.get("commitment"), rec.get("source")):
            got["reason"] = "the verdict's run commitment or draw source is not the coin record's"
        elif published is None:
            got["reason"] = "no published coin record to hold the coin as drawn to (--published-coin)"
        elif published.read_bytes() != public:
            got["reason"] = (f"the coin as drawn, in its public form (sha512 {got['published_sha512'][:16]}...), is not the "
                             f"coin record the run published (sha512 {sha512(published)[:16]}...)")
    except (OSError, ValueError, AttributeError) as e:
        got["reason"] = f"the coin record can't be held to the published one: {type(e).__name__}: {e}"[:600]
    return got


def _refusal(owner: str, verdict: str, why: str, **more: Any) -> dict[str, Any]:
    key = "why" if verdict == "missing" else "reason"
    return {"verdict": verdict, "owner": owner, key: S.public_text(why)[:2000], "answer_from": REPLAYED_FROM, **more}


def pearlplus_row(*, replay: Path, coin: Path, w: Path, case: Path, audit: Path, circuit_record: Path, node: Path,
                  private: Path, reads: GateReads, key: str | None, lane: str, main: Path | None,
                  published_coin: Path | None = None, commit: str | None = None, runs: Sequence[str] = (),
                  withheld: Withheld | None = None) -> dict[str, Any]:
    """PoUW's row from the replay under ``replay`` (module docstring): `replay_held`'s facts, the gate's reads with the
    served node's copies under ``node`` held to them (`held_to_gate`), and `served_verdicts.pearlplus_of_record` on the
    piece directory built in ``private`` (the coin as drawn, the two answers, the gate's sessions, ledger and pins as
    `pearlplus_coins` and `pearlplus_draw` leave them, and the tie's files: the audit ``audit``'s registration and
    summary, and ``circuit_record``). An accepting replay under every condition is the row, its answer Pearl+'s with the
    Lean's work; a verifier's reject is its ``reject``; a condition of record that fails leaves it ``missing``, naming
    the condition; a replay not held to its inputs, a copy that differs from the gate's, or an input that can't be read
    is its ``error``. ``withheld`` takes the coin as drawn, each of the gate's answers and each copy the served node
    handed off of one: files whose hashes the public record must not hold."""
    owner = S.PIECES["work"][2]
    if withheld is not None:
        withheld.add_file("the coin as drawn", coin)
    try:
        held = replay_held(replay, coin, w, case / PARTS, commit)
    except Refused as e:
        return _refusal(owner, "error", str(e))
    more: dict[str, Any] = {"replay": held, "runs": list(runs)}
    try:
        answer = S.check_answer(json.loads((replay / S.PEARLPLUS_ANSWER).read_text()))
        lean = json.loads((replay / S.PEARLPLUS_LEAN).read_text()) if (replay / S.PEARLPLUS_LEAN).is_file() else None
        if answer["verdict"] != "accept":
            return {**S.row_of(owner, answer, replay), "answer_from": REPLAYED_FROM, **more}
        if isinstance(lean, dict) and lean.get("verdict") == "reject":
            return _refusal(owner, "reject", f"the Lean PouwCheck.PearlPlus verdict is reject: {lean.get('reason')}", **more)
        if (held["python"]["verdict"], held["lean"]["verdict"]) != ("accept", (lean or {}).get("verdict")):
            raise Refused("the replay's record and its answers don't name the same verdicts")
        parts = json.loads((case / PARTS).read_text())
        rec = json.loads(coin.read_text())
        manifests = [m for m in parts["manifests"] if m.get("matmuls")]
        salts = sorted({str((m.get("session") or {}).get("session")) for m in manifests})
        if len(salts) != 1:
            raise Refused(f"the case's passes name {len(salts)} salt sessions, not one")
        draw = ((rec.get("draw") or {}).get("reserved") or {}).get("session")
        piece = private / "pouw"
        kept(piece / "passes")
        kept(piece / "tie" / "window")
        for name, src in ((S.COIN_RECORD, coin), (S.PEARLPLUS_ANSWER, replay / S.PEARLPLUS_ANSWER),
                          (S.PEARLPLUS_LEAN, replay / S.PEARLPLUS_LEAN)):
            if src.is_file():
                (piece / name).write_bytes(src.read_bytes())
        for name, src in ((S.WINDOW_REGISTRATION, audit / S.WINDOW_REGISTRATION), (S.AUDIT_SUMMARY, audit / S.AUDIT_SUMMARY),
                          (S.CIRCUIT_RECORD, circuit_record)):
            os.symlink(src.resolve(), piece / "tie" / name)
        served = {"coins": {"session": {"session": salts[0]}}, "passes": []}
        for i, m in enumerate(manifests):
            (f := piece / "passes" / f"pass-{i:05d}.json").write_text(json.dumps(m))
            served["passes"].append({"pass": i, "units": len(m["matmuls"]), "manifest": str(f)})
        sessions = [salts[0], *([draw] if isinstance(draw, str) and draw != salts[0] else [])]
        try:
            more["gate"] = held_to_gate(reads, node, sessions)
        finally:
            if withheld is not None:
                for verb, data in reads.raw().items():
                    withheld.add(f"the gate's answer to {verb}", data)
                for f in (node / "ledger.jsonl", node / "pins.jsonl",
                          *(f for sid in sessions for f in sorted((node / "session" / sid).rglob("*")))):
                    withheld.add_file(f"the served node's copy of {f.relative_to(node).as_posix()}", f)
        check = S.pearlplus_coins(served, piece, None, key, main, fetch=reads)
        drew = S.pearlplus_draw(piece / S.COIN_RECORD, piece, None, key, main, fetch=reads)
        more["coin"] = coin_held(coin, answer, published_coin)
        doc = {"coin_source": S.LIVE_SOURCE, "coins": {"check": check, "draw": drew}, "coin": more["coin"]}
        of_record, why = S.pearlplus_of_record(doc, piece, piece / "tie", key, lane)
    except Refused as e:
        return _refusal(owner, "error", str(e), **more)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, tarfile.TarError) as e:
        return _refusal(owner, "error", f"{type(e).__name__}: {e}", **more)
    if of_record is None:
        return _refusal(owner, "missing", f"not PoUW's verdict of record: {why}", **more)
    tie = of_record["public"]["tie"]
    if (held["lean"]["tie"] or {}).get("weights") != tie.get("weights"):
        return _refusal(owner, "missing", "not PoUW's verdict of record: the Lean verdict's tie names its weights "
                        f"{(held['lean']['tie'] or {}).get('weights')!r}, the Python's {tie.get('weights')!r}", **more)
    row = {**S.row_of(owner, of_record, replay), "answer_from": REPLAYED_FROM + GATE_HELD, "said": REPLAYED_SAID, **more}
    if "weights" in tie:
        row["weights"] = {"tie": tie["weights"], "cost": WEIGHTS_COST}
        row["said"] += f", {tie['weights']}"
    return row


def integrity_row(audit: Path, runs: Sequence[str] = (), causes: Sequence[str] = ()) -> dict[str, Any]:
    """Sampled proofs' row from the audit of record under ``audit`` as it answered (`AUDIT_FROM`), with the SHA-512 of
    each of the two files read; one that didn't accept carries ``causes``, marked `CAUSES_BY`, beside its own reason."""
    row = {**S.sampled_proofs_row(audit), "answer_from": AUDIT_FROM, "runs": list(runs),
           "files": {name: sha512(audit / name) if (audit / name).is_file() else None
                     for name in (S.AUDIT_SUMMARY, "window/audit.json")}}
    if row.get("verdict") != "accept" and causes:
        row["causes"] = {"by": CAUSES_BY, "of": [str(c) for c in causes]}
    return row


def inference_only(path: Path, source: str = "served-node") -> dict[str, Any]:
    """The inference-only part from the verdict at ``path``, taken from ``source`` (`INFERENCE_FROM`'s keys): its
    verdict and why, and the SHA-512 of the file's public form. One that can't be read is its ``error``."""
    part = {"source": source, "answer_from": INFERENCE_FROM[source]}
    try:
        text = path.read_text()
        doc = json.loads(text)
        return {**part, **{k: doc.get(k) for k in ("format", "verdict", "complies", "reason", "failure")},
                "published_sha512": hashlib.sha512(S.public_doc(text).encode()).hexdigest()}
    except (OSError, ValueError, AttributeError) as e:
        return {**part, "verdict": "error", "reason": f"{type(e).__name__}: {e}"[:600]}


def parts_said(rows: Mapping[str, Mapping[str, Any]], policy: Mapping[str, Any]) -> tuple[str, list[str]]:
    """The record's sentence over its three parts, each with where its verdict is from, and the names of those verified
    again off the served node. Pearl+ verified again says so, with `ACCEPT_UNAUDITED` where that is its verdict."""
    work, audit = rows["work"], rows["integrity"]
    again = ([S.NAMES["work"]] if work.get("answer_from") == REPLAYED_FROM + GATE_HELD
             and work.get("verdict") in ("accept", ACCEPT_UNAUDITED) else [])
    why = work.get("reason") or work.get("why")
    pouw = (f"Pearl+ {REPLAYED_SAID}{'' if work.get('verdict') == 'accept' else ': ' + ACCEPT_UNAUDITED}" if again
            else f"Pearl+ not accepted off the served node ({work.get('verdict')}: {why})")
    v = policy.get("verdict")
    served = (f"the inference-only verdict{'' if v == 'accept' else f' ({v})'} is "
              f"{policy['answer_from']}")
    said = {"accept": "accepted", "reject": "rejected"}.get(str(audit.get("verdict")), f"gave no verdict ({audit.get('verdict')})")
    return f"{pouw}; {served}; the sampled-proofs audit of record {said}.", again


def compose(rows: Mapping[str, Mapping[str, Any]], policy: Mapping[str, Any], *, window: int, steps: tuple[int, int],
            node: str, controlled_by: str, product: str, run: str | None = None, tree: Mapping[str, str] | None = None,
            notes: Sequence[str] = ()) -> tuple[dict[str, Any], dict[str, Any]]:
    """The record over sampled proofs' and PoUW's ``rows`` and the inference-only ``policy`` (module docstring, "What the
    record says of itself"), and its ``RECORD`` line. ``controlled_by`` is one of `CONTROLLERS`' keys. PoUW's accept
    beside an audit that didn't accept is `ACCEPT_UNAUDITED`, which composes as no accept, and says what its tie then
    rests on (`TIE_UNAUDITED`). ValueError on the served node itself, or for another ``controlled_by``."""
    if controlled_by not in CONTROLLERS:
        raise ValueError(f"who controls the composing host is one of {', '.join(sorted(CONTROLLERS))}, not {controlled_by!r}")
    rows = {k: dict(r) for k, r in rows.items()}
    if rows["work"].get("verdict") == "accept" and rows["integrity"].get("verdict") != "accept":
        rows["work"] |= {"verdict": ACCEPT_UNAUDITED, "tie_rests_on": TIE_UNAUDITED}
    report, rec, line = S.record(rows, window=window, steps=steps, node=node, run=run, servers={"work": (node, "real")},
                                 product=product, tree=tree,
                                 sources={"where": WHERE[controlled_by], "controlled_by": CONTROLLERS[controlled_by],
                                          "from": SOURCES_FROM})
    said, again = parts_said(report["protocols"], policy)
    composed = dict(report["composed"])
    if rec is not None or composed["verdict"] == "accept":
        composed = {**composed, "verdict": "pending", "reason": f"{NOT_ALL_HERE}; {composed['reason']}"}
    composed |= {"parts": said, "reverified_off_served_node": again, "composite": NOT_COMPOSITE}
    report |= {"composed": composed, "inference_only": dict(policy), "notes": list(notes)}
    line = {k: v for k, v in line.items() if k not in ("block_record", "why", "check", "steps", "deployment", "record_missing")}
    line |= {"composed": composed["verdict"], "reason": composed["reason"], "parts": said,
             "reverified_off_served_node": again, "composite": NOT_COMPOSITE,
             "inference_only": {k: policy.get(k) for k in ("source", "verdict", "answer_from")}, "notes": list(notes)}
    return report, line


def public_row(row: Mapping[str, Any], totals: str = "private") -> dict[str, Any]:
    """A row as the public record carries it (module docstring, "What leaves"): its `JUDGMENT`, Pearl+'s ε from its
    work where the row states none, the kind of a reason (`kind_of`) with `DETAIL_HELD`, each of the gate's reads as
    when it was read and how the served node's copy was held to it, and the replay's verifiers' verdicts. A row with a
    quantity carries its `TOTALS` under ``totals`` ``public`` and `TOTALS_HELD` otherwise; a refusal has none, and its
    answer, which holds its reason in full, never leaves."""
    if totals not in ("private", "public"):
        raise ValueError(f"exact totals are private or public, not {totals!r}")
    pub = {k: row[k] for k in JUDGMENT if k in row}
    work = (row.get("quantity") or {}).get("work") if isinstance(row.get("quantity"), Mapping) else None
    if pub.get("epsilon") is None and isinstance(work, Mapping) and work.get("epsilon") is not None:
        pub["epsilon"] = work["epsilon"]
    for key in ("reason", "why"):
        if key in row:
            pub[key] = None if row[key] is None else kind_of(row[key])
            if row[key] is not None:
                pub["detail"] = DETAIL_HELD
    if row.get("quantity") is not None:
        pub |= {k: row[k] for k in TOTALS if k in row} if totals == "public" else {"totals": TOTALS_HELD}
    if isinstance(row.get("gate"), Mapping):
        pub["gate"] = {verb: {"read": r.get("read"), "read_utc": r.get("read_utc"),
                              "node_copy": c if isinstance(c := r.get("node_copy"), str) else (c or {}).get("held")}
                       for verb, r in row["gate"].items()}
    if isinstance(row.get("replay"), Mapping):
        held = row["replay"]
        pub["replay"] = {"schema": held.get("schema"), "verifier_commit": held.get("verifier_commit"),
                         "python": {k: (held.get("python") or {}).get(k) for k in ("exit", "verdict")},
                         "lean": {k: (held.get("lean") or {}).get(k) for k in ("exit", "verdict", "differ")}}
    return pub


def _swap(o: Any, swaps: Sequence[tuple[str, str]]) -> Any:
    if isinstance(o, Mapping):
        return {k: _swap(v, swaps) for k, v in o.items()}
    if isinstance(o, list):
        return [_swap(v, swaps) for v in o]
    if isinstance(o, str):
        for full, kind in swaps:
            o = o.replace(full, kind)
    return o


def _strings(o: Any):
    if isinstance(o, Mapping):
        for k, v in o.items():
            yield from _strings(k)
            yield from _strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from _strings(v)
    elif isinstance(o, str):
        yield o


def public_report(report: Mapping[str, Any], line: Mapping[str, Any], totals: str = "private"
                  ) -> tuple[dict[str, Any], dict[str, Any]]:
    """`compose`'s record and line as they leave: each row its `public_row`, and each reason given in full anywhere
    else (the composed reason, the line) given as its kind. RuntimeError if a reason's own words would still leave."""
    rows = report["protocols"]
    swaps = sorted({(str(r[k]), kind_of(r[k])) for r in rows.values() for k in ("reason", "why") if r.get(k) is not None},
                   key=lambda x: -len(x[0]))
    swaps = [(full, kind) for full, kind in swaps if full != kind]
    pub = {**_swap({k: v for k, v in report.items() if k != "protocols"}, swaps),
           "protocols": {k: public_row(r, totals) for k, r in rows.items()}}
    said = _swap(dict(line), swaps)
    for full, _kind in swaps:
        if any(full in x for x in _strings([pub, said])):
            raise RuntimeError("a reason's own words would leave with the public record")
    return pub, said


def coin_note(put: Path | None) -> str:
    """The record's one line on whether the coin record's own artifact is in the evidence store, from the
    ``coin-put.json`` the run published (``--coin-put``; the served node's hand-off carries none); outside PoUW's row,
    which that file can't change."""
    unknown = "whether the coin record's own artifact is in the evidence store is not known here"
    if put is None:
        return f"{unknown} (no coin-put.json given)"
    try:
        doc = json.loads(put.read_text())
    except (OSError, ValueError) as e:
        return f"{unknown} ({put.name}: {type(e).__name__})"
    if isinstance(doc, dict) and doc.get("preserved") is True:
        return f"the coin record's own artifact is preserved in the evidence store ({put.name}: preserved: true, {doc.get('art')})"
    return (f"the coin record's own artifact was never preserved in the evidence store ({put.name}: preserved: false); "
            "the store holds the redacted coin record in the run's published record")


def write(out: Path, private: Path, report: Mapping[str, Any], line: Mapping[str, Any], pouw: Mapping[str, Path],
          totals: str = "private", withheld: Withheld | None = None) -> list[str]:
    """The record in full under ``private`` and what leaves under ``out`` (module docstring, "What leaves"): the public
    record and its line (`public_report` at ``totals``) and PoUW's files ``pouw`` (by name; one that isn't there is left
    out), each in its public form. The files written under ``out``. RuntimeError, before anything is written under
    ``out``, if a file that would leave isn't its own public form or holds a digest of ``withheld`` or of a file this
    leaves in redacted form."""
    withheld = withheld if withheld is not None else Withheld()
    kept(private / "record")
    text = json.dumps(report, indent=1, sort_keys=True) + "\n"
    said = f"RECORD {json.dumps(line, sort_keys=True)}\n"
    (private / "record" / "verdicts.json").write_text(text)
    (private / "verdicts.txt").write_text(said)
    pub, pub_line = public_report(report, line, totals)
    sources = {"record/verdicts.json": json.dumps(pub, indent=1, sort_keys=True) + "\n",
               "verdicts.txt": f"RECORD {json.dumps(pub_line, sort_keys=True)}\n"}
    for name, src in pouw.items():
        if src.is_file():
            sources[f"pieces/pouw/{name}"] = src.read_text()
    leaves = {rel: S.public_doc(body) for rel, body in sources.items()}
    for name, body in (("the private record", text), ("the private record's line", said), *sources.items()):
        withheld.add(name, body.encode())
    for rel, body in leaves.items():
        if S.public_doc(body) != body:
            raise RuntimeError(f"{rel} would leave in a form that is not its own public form")
        if names := withheld.found(body):
            raise RuntimeError(f"{rel} would leave holding a hash of {', '.join(names)}: a file whose redacted form is "
                               "published, which that hash would undo")
    for rel, body in leaves.items():
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_text(body)
    return sorted(leaves)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--serve-dir", type=Path, required=True, help="this host's copy of the run's serve dir")
    r.add_argument("--replay", type=Path, required=True, help="case_replay.sh's output directory")
    r.add_argument("--private", type=Path, required=True, help="a new directory, made 0700, for what stays on this host")
    r.add_argument("--out", type=Path, required=True, help="what leaves, each file in its public form")
    r.add_argument("--gate", required=True, help="the challenger gate's command, asked fetch SID, ledger and pins alone")
    r.add_argument("--key-sha512", required=True, help="the challenger key's pin")
    r.add_argument("--main", type=Path, required=True, help="main's first-parent line, a commit a line")
    r.add_argument("--lane", default="vllm")
    r.add_argument("--controlled-by", required=True, choices=sorted(CONTROLLERS),
                   help="who controls this host, which composed_on names: the developer's side, or an auditor")
    r.add_argument("--exact-totals", default="private", choices=("private", "public"),
                   help="where a row's exact totals go: the private record alone (default), or the public one too")
    r.add_argument("--published-coin", type=Path, help="the coin record the run published, which the coin as drawn's "
                                                       "public form is held to")
    r.add_argument("--coin-put", type=Path, help="the coin-put.json the run published, for the record's note on whether "
                                                 "the coin record's own artifact is in the store")
    r.add_argument("--sampled-causes", type=Path, help="a JSON list: the lane's diagnosis of a refused audit of record")
    r.add_argument("--verifier-commit", help="the commit of the tree whose verifiers the replay must have run")
    r.add_argument("--inference-only", default="served-node", choices=sorted(INFERENCE_FROM))
    r.add_argument("--run", help="the run composing the record, which it names")
    r.add_argument("--replay-run", action="append", default=[], help="a run PoUW's row names (the replay's)")
    r.add_argument("--audit-run", action="append", default=[], help="a run sampled proofs' row names (the audit of record's)")
    r.add_argument("--tree", help="the commit of the tree composing the record")
    r.add_argument("--tree-on", help="the branch that tree is on, with --tree")
    a = ap.parse_args(argv)
    d, out, private = a.serve_dir.resolve(), a.out.resolve(), a.private.resolve()
    try:
        if out == private or out in private.parents or private in out.parents:
            raise ValueError("--private and --out must be apart: what stays is never under what leaves")
        private.mkdir(mode=0o700, parents=True)
        env = settings(d / "private" / "compose" / "compose.env")
        k, node = int(env["K"]), env["SERVED_NODE"]
        lo, hi = (int(x) for x in env["STEPS"].split(","))
        handed, window = d / "private" / "compose", d / "pearlplus" / f"w{k}"
        audit = d / env["SAMPLED_AUDIT"]
        causes = json.loads(a.sampled_causes.read_text()) if a.sampled_causes else []
        if not (isinstance(causes, list) and all(isinstance(c, str) for c in causes)):
            raise ValueError("--sampled-causes is not a JSON list of strings")
        reads, withheld = GateReads(lambda *words: S._gate_out(a.gate, *words)), Withheld()
        rows = {"integrity": integrity_row(audit, a.audit_run, causes),
                "work": pearlplus_row(replay=a.replay, coin=handed / "pieces" / "pouw" / S.COIN_RECORD,
                                      w=window / "registration.json", case=d / "private" / "pearlplus" / "case",
                                      audit=audit, circuit_record=window / "tie" / S.CIRCUIT_RECORD,
                                      node=handed / "pieces" / "pouw", private=private, reads=reads, key=a.key_sha512,
                                      lane=a.lane, main=a.main, published_coin=a.published_coin,
                                      commit=a.verifier_commit, runs=[*([a.run] if a.run else []), *a.replay_run],
                                      withheld=withheld)}
        tree = {k_: v for k_, v in (("tree", a.tree), ("tree_on", a.tree_on)) if v}
        report, line = compose(rows, inference_only(handed / "inference-only.json", a.inference_only), window=k,
                               steps=(lo, hi), node=node, controlled_by=a.controlled_by, product=env.get("PRODUCT") or "remote",
                               run=a.run, tree=tree, notes=[coin_note(a.coin_put)])
        answers = "replay" in rows["work"] and a.exact_totals == "public"
        left = write(out, private, report, line, {
            **({S.PEARLPLUS_ANSWER: a.replay / S.PEARLPLUS_ANSWER, S.PEARLPLUS_LEAN: a.replay / S.PEARLPLUS_LEAN} if answers else {}),
            S.COIN_RECORD: private / "pouw" / S.COIN_RECORD}, a.exact_totals, withheld)
    except (OSError, ValueError, KeyError, RuntimeError) as e:
        print(S.public_line(f"RECORD refused: {type(e).__name__}: {e}"[:1000]), flush=True)
        return 2
    print((out / "verdicts.txt").read_text(), end="", flush=True)
    print(f"RECORD-FILES {len(left)} under the published directory: {', '.join(left)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
