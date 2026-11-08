"""The auditor's challenger for a test, in process: the only coins a `service.Stream` that reads coins takes are a
challenger's (`coins.ChallengerCoins`), and the verifier's run accepts a sample only with the evidence that the challenger
released them after the registration's digest, so a test's stream samples from this one.

`Challenger` is a `ChallengerCoins` each of whose commitments opens a session of its own: a `LiveChallenger` on
seeded entropy whose record is signed under the pinned key (`test_remote.PINNED`) and which an auditor's ledger of its
own opens for `WINDOW`, withheld (the verifier takes only a withheld session's coins, `challenger.withheld_on`), and
ends once the sample is read, as a served session's would (`remote.open_session`, `remote.close_session`). `evidence(digest)` is then `coins.coins_reason`'s evidence for the sample after `digest`.
`Stream` is a `service.Stream` over one, as a deployment drives it: `register` names the window, `select` commits the
coins to the registration's digest first (a second stage's to the interior registration's), and `outcome` hands the
verifier the sample's evidence (a two-stage sample's, both stages'), and the window's seed salt's when the stream drew
it from a `Challenger` of the seed-salt family (`seed_salt`)."""

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
    """The challenger's `label` coins (`coins` of them a session) on entropy seeded from `seed`, one session for each
    commitment, its files in a directory of its own. One of the seed-salt family opens its session for `WINDOW`'s
    open root at once, as `T.draw_seed_salt` reads the family's record before it commits."""

    def __init__(self, seed: str = "fake", coins: int = 64, label: str = LABEL) -> None:
        self.seed, self.coins, self.label = seed, coins, label
        self.directory = Path(tempfile.mkdtemp(prefix="fake-challenger-"))
        weakref.finalize(self, shutil.rmtree, self.directory, True)
        self.sessions: dict[bytes, tuple[R.Session, Ledger, C.LiveRecord]] = {}
        self._evidence: dict[bytes, dict[str, Any]] = {}
        super().__init__(None)
        if label == T.SEED_SALT_FAMILY:
            self.record = self._session(bytes.fromhex(WINDOW["root"]))

    def _session(self, after: bytes) -> C.LiveRecord:
        sid = f"s-{len(self.sessions) + 1}"
        d = self.directory / sid
        d.mkdir()
        ledger = Ledger(d / "windows.jsonl", at(0, 3))
        ledger.open(sid, WINDOW, withheld=True)
        challenger = C.LiveChallenger(SCHEME, 0, [(self.label, self.coins)], entropy(f"{self.seed}/{sid}"))
        session = R.open_session(challenger, pinned_key(), f"{HOST}.{sid}", at(1, 2), d, sid, withhold=True)
        (d / R.HELLOS).write_text("")
        record = C.LiveRecord(challenger, self.label, itertools.count().__next__)
        self.sessions[after] = (session, ledger, record)
        return record

    def commit(self, after: bytes) -> T.Segment:
        if bytes(after) not in self.sessions:
            self.record = self._session(bytes(after))
        return super().commit(after)

    def evidence(self, digest: str) -> dict[str, Any] | None:
        """The evidence for the sample after registration `digest` (hex), or for a window's seed salt after the window's
        open root `digest` (a `T.SEED_SALT_FAMILY` challenger's): its session closed by its client, its release
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
            self._evidence[after] = {"key": pinned_key().to_json(), "ledger": ledger.lines, "main": MAIN,
                                     "pins": HOST_PINS, "segment": segment.to_json(),
                                     "session": R.session_files(session.directory)}
        return self._evidence[after]


class Stream(S.Stream):
    """A `service.Stream` on a test's own challenger (`coins`, a new `Challenger` unless given; none for a stream
    that reads no coins), driven as a deployment drives it: a registration names `WINDOW`'s root unless it names
    another, a sample's coins are committed to its registration's digest before it is sampled, and an outcome's run is
    given the sample's evidence unless the test gives its own (`coins`, None for none)."""

    _NONE: Any = object()

    def __init__(self, spec: S.Spec, unit: Any, *, coins: T.ChallengerCoins | None = None, **kwargs: Any) -> None:
        if coins is None and kwargs.get("sampler") is None and not isinstance(spec.selection, S.All):
            coins = Challenger()
        self.fake = coins if isinstance(coins, Challenger) else None
        self.salted: tuple[Challenger, str] | None = None
        super().__init__(spec, unit, coins=coins, **kwargs)

    def seed_salt(self, coins: T.ChallengerCoins, window_root: str) -> bytes:
        salt = super().seed_salt(coins, window_root)
        if isinstance(coins, Challenger):
            self.salted = (coins, window_root)
        return salt

    def register(self, record: Mapping[str, Any], ports: Mapping[str, Mapping[str, Any]],
                 **kwargs: Any) -> S.WindowCommitmentRecord:
        return super().register(record, ports, **{"window_root": WINDOW["root"], **kwargs})

    def register_interiors(self, draw: S.Draw, record: Mapping[str, Any], ports: Mapping[str, Mapping[str, Any]],
                           **kwargs: Any) -> S.WindowCommitmentRecord:
        return super().register_interiors(draw, record, ports, **{"window_root": WINDOW["root"], **kwargs})

    def select(self, registration: S.WindowCommitmentRecord, **kwargs: Any) -> S.Draw:
        if self.fake is not None and bytes.fromhex(registration.digest) not in self.fake.sessions:
            self.fake.commit(bytes.fromhex(registration.digest))
        return super().select(registration, **kwargs)

    def _evidence(self, registration: S.WindowCommitmentRecord) -> dict[str, Any] | None:
        after = bytes.fromhex(registration.digest)
        return self.fake.evidence(registration.digest) if self.fake and after in self.fake.sessions else None

    def outcome(self, draw: S.Draw, verdicts: Mapping[str, Mapping[str, Any]], statement: Mapping[str, Any], *,
                coins: Mapping[str, Any] | None = _NONE, **kwargs: Any) -> S.Outcome | S.Pending:
        """A two-stage sample's evidence is both stages': the first stage's after the registration, the second's
        (`interior_coins`) after the interior registration. The window's seed salt's is both registrations' (`seed_salt`,
        `interior_seed_salt`): they register in the one window."""
        first = draw.replay_sample or draw
        if coins is Stream._NONE:
            coins = self._evidence(first.registration)
        if self.salted is not None:
            salted, root = self.salted
            kwargs.setdefault("seed_salt", salted.evidence(root))
        if draw.replay_sample is not None:
            kwargs.setdefault("interior_coins", self._evidence(draw.registration))
            kwargs.setdefault("interior_seed_salt", kwargs.get("seed_salt"))
        return super().outcome(draw, verdicts, statement, coins=coins, **kwargs)
