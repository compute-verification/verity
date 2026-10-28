"""The bandwidth leg's store, the period's records (`period.records`, PoUS's own): every copy of the auditor's ledger in
them cut to the entries their window cites. PoUW's reader of the ledger (`challenger.audited_session`) checks
every entry, chained from seq 0, so it takes the window's whole ledger, which stays on disk, and refuses the cut copy
rather than credit from it; a copy that doesn't open the cited session is refused when the records are written; and the
bandwidth session accepts the cut store as it accepts the whole."""
from __future__ import annotations

import hashlib
import json

import pytest

from remote_bandwidth_fakes import Clock, Entropy
from test_verifier_party import COINS
from verity.core.protocols.pous import period as pous_period
from verity.core.protocols.remote_bandwidth import period as P
from verity.core.protocols.sampled_proofs import challenger as C

KEY = C.ChallengerKey(4, 10, hashlib.sha512(b"the challenger key").digest() + bytes(16))
LANE = "compute-accounting"
T = "2026-10-26T21:44:45Z"


def verify(key: C.ChallengerKey, message: bytes, signature: bytes) -> bool:
    """A stand-in for LMS that accepts every signature, so the ledger can be long: the rest of its check is infra's."""
    return True


class Ledger:
    """The auditor's ledger, each entry signed at the challenger key's next leaf (`verify`'s stand-in signature)."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.leaf, self.key_sha512 = 0, hashlib.sha512(KEY.public).hexdigest()

    def sign(self, message: bytes) -> bytes:
        self.leaf += 1
        return self.leaf.to_bytes(4, "big") + bytes(60)

    def add(self, kind: str, session: str | None, body: dict) -> int:
        self.lines.append(C.ledger_line(self.lines, kind, session, body, KEY.public[:16], self.sign))
        return len(self.lines) - 1

    def period(self, name: str, sessions: list[str]) -> dict[str, int]:
        """Period `name` of `LANE`'s with a window a session, each reserved, opened, its record signed and ended: each
        entry's index."""
        at = {"period-open": self.add("period-open", None, {"period": name, "lane": LANE, "key_sha512": self.key_sha512,
                                                            "cap": len(sessions), "circuit": "cd" * 64})}
        for s in sessions:
            reserve = {"by": LANE, "minutes": 30, "started_utc": T, "expires_utc": C.later(T, 30)}
            at[f"reserve {s}"] = self.add("reserve", s, reserve)
            at[f"open {s}"] = self.add("open", s, {
                **reserve, "reserve": at[f"reserve {s}"], "opened_by": LANE, "window": {"root": root(s)},
                "host": "vy-challenger-1", "pinned": "ab" * 20, "key_sha512": self.key_sha512})
            self.leaf += 1
            at[f"end {s}"] = self.add("end", s, {
                "closed": True, "complete": True, "files": {f: "00" * 64 for f in C.SEALED_FILES},
                "record_leaf": self.leaf, "commitment": "ee" * 64, "refused_hellos": 0})
        at["period-close"] = self.add("period-close", None, {
            "period": name, "lane": LANE, "complete": True,
            "windows": [{"root": root(s), "session": s, "complete": True} for s in sessions]})
        return at


def root(s: str) -> str:
    return hashlib.sha512(s.encode()).hexdigest()


def window(d, n: int) -> tuple[list[str], str, list[int]]:
    """A live window's records under `d` whose salt session is the middle of `n` in the first of two periods: the
    ledger, the session and the indices of the entries it cites."""
    led = Ledger()
    sessions = [f"s20261026T{i:06d}Z-{i % 65536:04x}" for i in range(2 * n)]
    at = led.period("p-1", sessions[:n])
    led.period("p-2", sessions[n:])
    s = sessions[n // 2]
    close = C.ChallengerClose(s, 3, bytes(64), None, bytes(40)).to_json()
    files = {"period.json": b"{}", "pouw/salt-session/windows.jsonl": "".join(x + "\n" for x in led.lines).encode(),
             f"pouw/salt-session/{s}/challenger.json": json.dumps(close).encode(),
             "pouw/salt.json": json.dumps({"answer": {"session": {"session": s}}, "live": {
                 "key": KEY.to_json(), "ledger": led.lines, "main": [], "pins": [],
                 "session": {"close": close, "hellos": [], "log": [], "record": {}}}}, sort_keys=True).encode()}
    for name, data in files.items():
        (d / name).parent.mkdir(parents=True, exist_ok=True)
        (d / name).write_bytes(data)
    return led.lines, s, [at[k] for k in ("period-open", f"reserve {s}", f"open {s}", f"end {s}", "period-close")]


def unframe(b: bytes) -> dict[str, bytes]:
    out, pos = {}, 0
    while pos < len(b):
        n = int.from_bytes(b[pos:pos + 4], "big")
        name, pos = b[pos + 4:pos + 4 + n].decode(), pos + 4 + n
        m = int.from_bytes(b[pos:pos + 8], "big")
        out[name], pos = b[pos + 8:pos + 8 + m], pos + 8 + m
    return out


def test_pouws_reader_takes_the_whole_ledger_on_disk_and_refuses_the_cut_copy(tmp_path):
    lines, s, kept = window(tmp_path, 100)
    assert P.records is pous_period.records and pous_period.LEDGER_SCHEMAS == tuple(C.LEDGER_TAGS)
    got = unframe(P.records(tmp_path))
    whole = json.loads((tmp_path / "pouw/salt.json").read_text())["live"]["ledger"]
    assert whole == lines and (tmp_path / "pouw/salt-session/windows.jsonl").read_text().split("\n")[:-1] == lines
    audited = C.audited_session(whole, {"root": root(s)}, KEY, verify)
    assert (audited.session, audited.period, audited.cap) == (s, "p-1", 100)
    cut = json.loads(got["pouw/salt.json"])["live"]["ledger"]
    assert cut == [lines[i] for i in kept] == got["pouw/salt-session/windows.jsonl"].decode().split("\n")[:-1]
    for copy in (cut, cut[1:]):
        with pytest.raises(C.CoinError, match="^the auditor's ledger is refused: entry [01]: not entry [01] of a "):
            C.audited_session(copy, {"root": root(s)}, KEY, verify)


def test_a_copy_that_doesnt_open_the_cited_session_is_refused_when_the_records_are_written(tmp_path):
    lines, s, kept = window(tmp_path, 8)
    doc = json.loads((tmp_path / "pouw/salt.json").read_text())
    doc["live"]["ledger"] = lines[:kept[2]]
    (tmp_path / "pouw/salt.json").write_text(json.dumps(doc, sort_keys=True))
    with pytest.raises(ValueError, match=f"^pouw/salt.json: the auditor's ledger opens no session {s}, which a "):
        P.records(tmp_path)


def test_the_bandwidth_session_accepts_the_cut_store_as_the_whole(tmp_path):
    window(tmp_path, 2)
    whole = b"".join(len(k.encode()).to_bytes(4, "big") + k.encode() + len(v).to_bytes(8, "big") + v for k, v in (
        (p.relative_to(tmp_path).as_posix(), p.read_bytes()) for p in sorted(tmp_path.rglob("*")) if p.is_file()))
    blocks = {}
    for name, store in (("whole", whole), ("cut", P.records(tmp_path))):
        c = Clock(10 ** 9, 1_000)
        got = P.run("n0", store, COINS, clock=c, sleep_until=c.sleep_until, entropy=Entropy(name.encode()))
        assert got.verdict == "accept", got.answer
        blocks[name] = len(P.payloads(store))
    assert blocks["cut"] < blocks["whole"]
