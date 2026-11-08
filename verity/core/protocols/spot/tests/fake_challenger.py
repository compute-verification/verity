"""The auditor's challenger for a test, in process: the only coins a `service.Stream` that reads coins takes are a
challenger's (`coins.ChallengerCoins`), and the verifier's run accepts a sample only with the evidence that the challenger
released them after the registration's digest, so a test's stream samples from this one.

`Challenger` is a `ChallengerCoins` each of whose commitments opens a session of its own: a `LiveChallenger` on
seeded entropy whose record is signed under the pinned key (`test_remote.PINNED`) and which an auditor's ledger of its
own opens for `WINDOW`, withheld (the verifier takes only a withheld session's coins, `challenger.withheld_on`), and
ends once the sample is read, as a served session's would (`remote.open_session`, `remote.close_session`). `evidence(digest)` is then `coins.coins_reason`'s evidence for the sample after `digest`.
`Stream` is a `service.Stream` over one, as a deployment drives it: `register` names the window, `select` commits the
coins to the registration's digest first (a second stage's to the interior registration's), and `outcome` hands the
verifier the sample's evidence (a two-stage sample's, both stages') and a two-stage sample's second-stage sessions'
statements, an honest prover's (`statements`)."""

from __future__ import annotations

import hashlib
import itertools
import shutil
import tempfile
import weakref
from pathlib import Path
from typing import Any, Mapping

from verity.core.protocols.spot import (challenger as C, coins as T, remote as R, second_stage as SS,
                                                  service as S)
from verity.core.protocols.spot.one_stage import draw as D, partition as P
from verity.ml.circuits.codec import canonical_json, descriptor_id
from verity.ml.commitments.identity import canonical_json_text

from test_challenger import HOST, entropy
from test_remote import HOST_PINS, MAIN, SCHEME, WINDOW, Ledger, at, pinned_key

LABEL = "positions"


def checked(statement: Mapping[str, Any], tables: int | None = 1) -> dict[str, Any]:
    """What `flock-verify verify`'s verdict names of the session of second-stage statement `statement` it checked, which
    round 2 holds the statement to (`second_stage.verified_of`): the SHA-512 of its header's canonical JSON, of its
    registered reads' when it reads any, its sample, and the `tables` it set the session up as (one, as the verifier of
    record runs it; None: a verdict naming no count)."""
    names = {"public_header_sha512": hashlib.sha512(canonical_json(statement.get("header"))).hexdigest(),
             "unit_draw": statement.get("draw"), **({} if tables is None else {"session_tables": tables})}
    if statement.get("registered") is not None:
        names["registered_sha512"] = hashlib.sha512(canonical_json(statement["registered"])).hexdigest()
    return names


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
        """The evidence for the sample after registration `digest` (hex): its session closed by its client, its release
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
    given the sample's evidence unless the test gives its own (`coins`, None for none), and a two-stage one the honest
    second-stage statements unless the test gives its own (`statements`, None for none)."""

    _NONE: Any = object()

    def __init__(self, spec: S.Spec, unit: Any, *, coins: T.ChallengerCoins | None = None, **kwargs: Any) -> None:
        if coins is None and kwargs.get("sampler") is None and not isinstance(spec.selection, S.All):
            coins = Challenger()
        self.fake = coins if isinstance(coins, Challenger) else None
        super().__init__(spec, unit, coins=coins, **kwargs)

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
                coins: Mapping[str, Any] | None = _NONE, statements: Any = _NONE, **kwargs: Any) -> S.Outcome | S.Pending:
        """A two-stage sample's evidence is both stages': the first stage's after the registration, the second's
        (`interior_coins`) after the interior registration."""
        first = draw.replay_sample or draw
        if coins is Stream._NONE:
            coins = self._evidence(first.registration)
        if draw.replay_sample is not None:
            kwargs.setdefault("interior_coins", self._evidence(draw.registration))
        if statements is Stream._NONE:
            statements = None if draw.replay_sample is None else self.statements(draw)
        return super().outcome(draw, verdicts, statement, coins=coins, statements=statements, **kwargs)

    def statements(self, second: S.Draw) -> list[dict[str, Any]]:
        """The statements of the second-stage sessions an honest prover runs (`second_stage`): one for each template of
        the interior program, its header over the template's slots of the interior registration at its roots (the ports
        `<template>/<port>` there where the record names ports by template), its sample the second stage's units among
        them, counted from its first, and its registered reads: each port the replay block's Definition links
        (`second_stage.replay_links`) of the boundary port its link names, each slot the first row of its cell's replay
        block, and each value the interior registration registers, under a port of its name (None when it reads
        none)."""
        e = self._entries[second.registration.order]
        rec, boundary = second.registration.record, second.replay_sample.registration.record
        named = any("/" in r["port"] for r in rec["roots"])
        values = {x["value"]: {k: x[k] for k in ("value", "root", "domain", "leaves")}
                  for x in rec.get("registered", [])}
        at = {r["port"]: r for r in boundary["roots"]}
        links = SS.replay_links(e.program, descriptor_id(self.replay), boundary) or ()
        sample = D.proved(second.replay_sample.draw)
        out = []
        for t, ((lo, hi),) in P.strata(e.interior_program, self.spec.check.query):
            prefix = f"{t}/" if named else ""
            roots = {r["port"].removeprefix(prefix): r for r in rec["roots"] if r["port"].startswith(prefix)}
            units = [u - lo for u in D.proved(second.draw) if lo <= u < hi]
            n = (hi - lo) // len(sample)
            reads = {**values, **{x.port: {
                "value": x.value, **{k: at[x.value][k] for k in ("root", "domain", "leaves")}, "reads_from": lo,
                "reads": [at[x.value]["leaves"] // boundary["population"] * sample[u // n] for u in range(hi - lo)]}
                for x in links if x.template == t and x.value in at}}
            out.append({"template": t, "registered": {"ports": reads} if reads else None,
                        "header": {"scheme": rec["scheme"], "instances": hi - lo, "units": {"indices": list(range(lo, hi))},
                                   "frame_v3": {"roots": {p: r["root"] for p, r in roots.items()},
                                                "domain_ids": {p: r["domain"] for p, r in roots.items()}}},
                        "draw": {"law": "subset", "population": hi - lo, "k": len(units), "units": units}})
        return out
