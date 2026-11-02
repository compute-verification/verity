"""The auditor's challenger for a test, in process: the only coins a `service.Stream` that reads coins takes are a
challenger's (`coins.ChallengerCoins`), and the verifier's run accepts a draw only with the evidence that the challenger
released them after the registration's digest, so a test's stream draws from this one.

`Challenger` is a `ChallengerCoins` each of whose commitments opens a session of its own: a `LiveChallenger` on
seeded entropy whose record is signed under the pinned key (`test_remote.PINNED`) and which an auditor's ledger of its
own opens for `WINDOW`, withheld (the verifier takes only a withheld session's coins, `challenger.withheld_on`), and
ends once the draw is read, as a served session's would (`remote.open_session`, `remote.close_session`). `evidence(digest)` is then `coins.coins_reason`'s evidence for the draw after `digest`.
`Stream` is a `service.Stream` over one, as a deployment drives it: `register` names the window, `select` commits the
coins to the registration's digest first, and `outcome` hands the verifier the draw's evidence."""

from __future__ import annotations

import itertools
import shutil
import tempfile
import weakref
from pathlib import Path
from typing import Any, Mapping

from verity.core.protocols.sampled_proofs import challenger as C, coins as T, remote as R, service as S
from verity.ml.commitments.identity import canonical_json_text

from test_challenger import HOST, entropy
from test_remote import HOST_PINS, MAIN, SCHEME, WINDOW, Ledger, at, pinned_key

LABEL = "positions"


class Challenger(T.ChallengerCoins):
    """The challenger's `LABEL` coins (`coins` of them a session) on entropy seeded from `seed`, one session for each
    commitment, its files in a directory of its own."""

    def __init__(self, seed: str = "fake", coins: int = 64) -> None:
        self.seed, self.coins = seed, coins
        self.directory = Path(tempfile.mkdtemp(prefix="fake-challenger-"))
        weakref.finalize(self, shutil.rmtree, self.directory, True)
        self.sessions: dict[bytes, tuple[R.Session, Ledger, C.LiveRecord]] = {}
        self._evidence: dict[bytes, dict[str, Any]] = {}
        super().__init__(None)

    def _session(self, after: bytes) -> C.LiveRecord:
        sid = f"s-{len(self.sessions) + 1}"
        d = self.directory / sid
        d.mkdir()
        ledger = Ledger(d / "windows.jsonl", at(0, 3))
        ledger.open(sid, WINDOW, withheld=True)
        challenger = C.LiveChallenger(SCHEME, 0, [(LABEL, self.coins)], entropy(f"{self.seed}/{sid}"))
        session = R.open_session(challenger, pinned_key(), f"{HOST}.{sid}", at(1, 2), d, sid, withhold=True)
        (d / R.HELLOS).write_text("")
        record = C.LiveRecord(challenger, LABEL, itertools.count().__next__)
        self.sessions[after] = (session, ledger, record)
        return record

    def commit(self, after: bytes) -> T.Segment:
        self.record = self._session(bytes(after))
        return super().commit(after)

    def evidence(self, digest: str) -> dict[str, Any] | None:
        """The evidence for the draw after registration `digest` (hex): its session closed by its client, its release
        log as a served challenger writes it (each root, then its coin's release, as `remote.serve` answers one open at a
        time), and the ledger's end sealing it; None for a draw that read no coins."""
        after = bytes.fromhex(digest)
        segment = next(s for s in reversed(self.segments) if s.after == after)
        if not segment.openings:
            return None
        if after not in self._evidence:
            session, ledger, record = self.sessions[after]
            released = {e["index"]: e for e in record.log if e["event"] == "release"}
            lines = [canonical_json_text(x) for e in record.log if e["event"] == "root" for x in (e, released[e["index"]])]
            with open(session.directory / R.RELEASES, "a", encoding="utf-8") as log:
                log.write("".join(line + "\n" for line in lines))
            session.lines.extend(lines)
            R.close_session(session, None)
            ledger.end(session.id, session.directory, True)
            self._evidence[after] = {"key": pinned_key().to_json(), "ledger": ledger.lines, "main": MAIN,
                                     "pins": HOST_PINS, "segment": segment.to_json(),
                                     "session": R.session_files(session.directory)}
        return self._evidence[after]


class Stream(S.Stream):
    """A `service.Stream` on a test's own challenger (`coins`, a new `Challenger` unless given; none for a stream
    that reads no coins), driven as a deployment drives it: a registration names `WINDOW`'s root unless it names
    another, a draw's coins are committed to its registration's digest before it is drawn, and an outcome's run is
    given the draw's evidence unless the test gives its own (`coins`, None for none)."""

    _NONE: Any = object()

    def __init__(self, spec: S.Spec, unit: Any, *, coins: T.ChallengerCoins | None = None, **kwargs: Any) -> None:
        if coins is None and kwargs.get("sampler") is None and not isinstance(spec.selection, S.All):
            coins = Challenger()
        self.fake = coins if isinstance(coins, Challenger) else None
        super().__init__(spec, unit, coins=coins, **kwargs)

    def register(self, record: Mapping[str, Any], ports: Mapping[str, Mapping[str, Any]], **kwargs: Any) -> S.Registration:
        return super().register(record, ports, **{"window_root": WINDOW["root"], **kwargs})

    def select(self, registration: S.Registration, **kwargs: Any) -> S.Draw:
        if self.fake is not None and bytes.fromhex(registration.digest) not in self.fake.sessions:
            self.fake.commit(bytes.fromhex(registration.digest))
        return super().select(registration, **kwargs)

    def outcome(self, draw: S.Draw, verdicts: Mapping[str, Mapping[str, Any]], statement: Mapping[str, Any], *,
                coins: Mapping[str, Any] | None = _NONE, **kwargs: Any) -> S.Outcome | S.Pending:
        if coins is Stream._NONE:
            after = bytes.fromhex(draw.registration.digest)
            coins = self.fake.evidence(draw.registration.digest) if self.fake and after in self.fake.sessions else None
        return super().outcome(draw, verdicts, statement, coins=coins, **kwargs)
