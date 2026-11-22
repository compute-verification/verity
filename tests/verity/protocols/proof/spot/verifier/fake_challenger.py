"""The auditor's challenger for a test, in process: the only coins a `service.Stream` that reads coins takes are a
challenger's (`coins.ChallengerCoins`), and the verifier's run accepts a sample only with the evidence that the challenger
released them after the docket's digest, so a test's stream samples from this one.

`Challenger` is a `ChallengerCoins` each of whose commitments opens a session of its own: a `LiveChallenger` on
seeded entropy whose record is signed under the pinned key (`test_remote.PINNED`) and which an auditor's ledger of its
own opens for `WINDOW`, withheld (the verifier takes only a withheld session's coins, `challenger.withheld_on`), and
ends once the sample is read, as a served session's would (`remote.open_session`, `remote.close_session`). `evidence(digest)` is then `coins.coins_reason`'s evidence for the sample after `digest`.
`Stream` is a `service.Stream` over one, as a deployment drives it: `register` names the window, `select` commits the
coins to the docket's digest first, and `outcome` hands the verifier the sample's evidence."""

from __future__ import annotations

import itertools
import shutil
import tempfile
import weakref
from pathlib import Path
from typing import Any, Mapping

from verity.protocols.proof.spot.verifier.challenger import challenger as C, coins as T, remote as R
from verity.protocols.proof.spot.verifier import service as S
from verity.commitments.identity import canonical_json_text

from test_challenger import HOST, entropy
from test_remote import HOST_PINS, MAIN, SCHEME, WINDOW, Ledger, at, pinned_key

LABEL = "positions"


class Challenger(T.ChallengerCoins):
    """The challenger's coins of one family (`label`, `LABEL` unless given; `coins` of them a session) on entropy seeded
    from `seed`, one session for each commitment, its files in a directory of its own, opened for `window` as it is at
    the commitment (`WINDOW` unless a test sets another)."""

    def __init__(self, seed: str = "fake", coins: int = 64, label: str = LABEL) -> None:
        self.seed, self.coins, self.window, self.family = seed, coins, WINDOW, label
        self.directory = Path(tempfile.mkdtemp(prefix="fake-challenger-"))
        weakref.finalize(self, shutil.rmtree, self.directory, True)
        self.sessions: dict[bytes, tuple[R.Session, Ledger, C.LiveRecord]] = {}
        self._evidence: dict[bytes, dict[str, Any]] = {}
        super().__init__(None)

    @property
    def label(self) -> str:
        return self.family

    def _session(self, after: bytes) -> C.LiveRecord:
        sid = f"s-{len(self.sessions) + 1}"
        d = self.directory / sid
        d.mkdir()
        ledger = Ledger(d / "windows.jsonl", at(0, 3))
        ledger.open(sid, self.window, withheld=True)
        challenger = C.LiveChallenger(SCHEME, 0, [(self.family, self.coins)], entropy(f"{self.seed}/{sid}"))
        session = R.open_session(challenger, pinned_key(), f"{HOST}.{sid}", at(1, 2), d, sid, withhold=True)
        (d / R.HELLOS).write_text("")
        record = C.LiveRecord(challenger, self.family, itertools.count().__next__)
        self.sessions[after] = (session, ledger, record)
        return record

    def commit(self, after: bytes) -> T.Segment:
        self.record = self._session(bytes(after))
        return super().commit(after)

    def evidence(self, digest: str) -> dict[str, Any] | None:
        """The evidence for the sample after docket `digest` (hex): its session closed by its client, its release
        log as a served challenger writes it (each root, then its coin's release, as `remote.serve` answers one open at a
        time), and the ledger's end sealing it; None for a sample that read no coins."""
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
            self._evidence[after] = {"ledger": ledger.lines, "main": MAIN, "pins": HOST_PINS,
                                     "segment": segment.to_json(), "session": R.session_files(session.directory)}
        return self._evidence[after]


class Stream(S.Stream):
    """A `service.Stream` on a test's own challenger (`coins`, a new `Challenger` unless given; none for a stream
    that reads no coins), driven as a deployment drives it: a docket names `WINDOW`'s root unless it names
    another, a sample's coins are committed to its docket's digest before it is sampled, the stream holds the
    challenger's key (`pinned_key`) and the verifier's copy of main's history (`MAIN`) unless the test gives its own
    (`key`, `main`), and an outcome's run is
    given the sample's evidence unless the test gives its own (`coins`, None for none)."""

    _NONE: Any = object()

    def __init__(self, spec: S.Spec, unit: Any, *, coins: T.ChallengerCoins | None = None, **kwargs: Any) -> None:
        if coins is None and kwargs.get("sampler") is None and not isinstance(spec.selection, S.All):
            coins = Challenger()
        self.fake = coins if isinstance(coins, Challenger) else None
        super().__init__(spec, unit, coins=coins, **{"key": pinned_key(), "main": MAIN, **kwargs})

    def register(self, record: Mapping[str, Any], ports: Mapping[str, Mapping[str, Any]],
                 **kwargs: Any) -> S.WindowCommitmentRecord:
        return super().register(record, ports, **{"window_root": WINDOW["root"], **kwargs})

    def select(self, docket: S.WindowCommitmentRecord, **kwargs: Any) -> S.Draw:
        if self.fake is not None and bytes.fromhex(docket.digest) not in self.fake.sessions:
            self.fake.commit(bytes.fromhex(docket.digest))
        return super().select(docket, **kwargs)

    def _evidence(self, docket: S.WindowCommitmentRecord) -> dict[str, Any] | None:
        after = bytes.fromhex(docket.digest)
        return self.fake.evidence(docket.digest) if self.fake and after in self.fake.sessions else None

    def outcome(self, draw: S.Draw, verdicts: Mapping[str, Mapping[str, Any]], statement: Mapping[str, Any], *,
                coins: Mapping[str, Any] | None = _NONE, **kwargs: Any) -> S.Outcome | S.Pending:
        if coins is Stream._NONE:
            coins = self._evidence(draw.docket)
        return super().outcome(draw, verdicts, statement, coins=coins, **kwargs)
