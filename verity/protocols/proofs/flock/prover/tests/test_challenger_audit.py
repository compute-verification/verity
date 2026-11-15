"""`pod/challenger.sh`'s audits of served sessions against their challenger's release log, run as the pods run them, on the
plain-tree fixture's challenger and session (`test_lean_plain_tree.py`): `challenger_audit` through `challengers.end`, as the
sweeps' servers and eight pod jobs run it, and `challenger_cases`, 93-zk-tree-from.sh's. Both hold each record at the tables
its link's `root_b` keys, in Python (`challenger.sessions_tree_reason`) and in Lean (`spot-difftest`'s
`challenger_session_trees`), so either side called with (name, record) alone fails here. It builds `spot-difftest`, which no other test of this
suite runs.

A recursive session's inner coins held to the challenger's release log: `Spot.Challenger.innerTreesReason`
(`ChallengerInnerTrees.lean`, the verifier, which `VStar.Recursive.checkBlobs` reads) against its Python mirror
`verity_flock.rec_live.tree_held_reason`, through `spot-difftest`'s piece `challenger_inner_trees`.

The sampled cases are inner sessions as rec_live's live-coin part (`rec_live.LiveCoinPart`) runs them on coin trees a
challenger (`challenger.LiveChallenger`) samples and opens, their release log lines among the challenger's roots and
releases, one tree a session; then one mutation each (`INNER_TREES_NAMES`): a tree no line logged, or another than the one
logged under its root; a link root, table or point edited; a round's coins, entry, number or stream edited, so it is a
round the challenger didn't open after those bytes; a round read twice; a round the log shows opened and the record
doesn't read; a tree the log shows sampled and no session uses, or one two sessions list; one session's record under
another's tree; each malformed form of the record, the log and the request.

`test_a_served_inner_session_is_held_and_each_bent_copy_refused` holds the two to a served inner session (`SERVED`, as
`rec_inner.py tree-held` reads one: the challenger's release log, the session's ``coin_tree`` and its live-coin part's
``public.json``, gzipped), accepted, and to that session with a round moved, a second tree and an unread opening, each refused.

The inner-trees cases are here, not in SPOT's Lean tests, because their mirror is `verity_flock`'s and no test
under a `verifier/` loads `verity_flock`; they borrow the session and log builders of SPOT's `test_challenger_difftest.py`.
"""

from __future__ import annotations

import gzip
import json
import random
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import verity_difftest as D
from verity.commitments import coin_tree as CT, hm96
from verity.protocols.proofs.spot.verifier.challenger import challenger as C, remote as R

from verity_flock import challengers as CH, rec_live as RL

SP = Path(__file__).resolve().parents[3] / "spot" / "verifier" / "lean"
sys.path.insert(0, str(SP / "tests"))
from test_challenger_difftest import (DROP, PACKAGE, PIECES, SESSIONS, TABLES, Session, _body, _cases,  # noqa: E402
                                      _edit, _flip, _interleave, _log, _refusing, _request, _seeded)

#: ``challenger/`` (its record and release log) and ``session/`` (the plain session it sampled and opened the tree of)
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "plain-tree"
#: a served inner session, as `rec_inner.py tree-held` reads one
SERVED = Path(__file__).resolve().parent / "fixtures" / "inner-trees-served"
#: one coin-tree node: a whole SHA-512 digest (`rec_live`'s node size)
NODE_BYTES = hm96.SHA512.digest_bytes

pytestmark = [pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain"), pytest.mark.slow]


class _Exited:
    def wait(self, timeout=None):
        return 0


def _bumped(h: str) -> str:
    return h[:-1] + "0123456789abcdef"[("0123456789abcdef".index(h[-1]) + 1) % 16]


def _session(d: Path, rec: dict) -> Path:
    d.mkdir(parents=True)
    (d / "session.json").write_text(json.dumps(rec))
    return d


@pytest.fixture(scope="module")
def built():
    assert (FIXTURE / "challenger" / "releases.jsonl").is_file(), (
        f"{FIXTURE} is not fetched: `uv run research data fetch-fixtures verity/protocols/proofs/flock/prover/tests/fixtures/plain-tree`")
    D.build(SP, "spot-difftest")


@pytest.fixture
def ch(built, tmp_path: Path) -> Path:
    """The fixture's challenger directory, copied, since the audit writes its audit.json there."""
    d = tmp_path / "challenger"
    shutil.copytree(FIXTURE / "challenger", d)
    (d / "audit.json").unlink(missing_ok=True)
    return d


@pytest.fixture
def rec() -> dict:
    return json.loads((FIXTURE / "session" / "session.json").read_text())


def test_challenger_audit_passes_the_served_session_and_refuses_it_with_a_link_point_changed(ch, rec, tmp_path):
    honest = _session(tmp_path / "sessions" / "honest", rec)
    assert CH.end(ch, _Exited(), [str(honest)]) == {"exit": 0, "sessions": 1, "audit": True, "ok": True}
    assert json.loads((ch / "audit.json").read_text()) == {"sessions": ["honest"], "python": None, "lean": None, "ok": True}
    rec["link"]["points"][0][0] = _bumped(rec["link"]["points"][0][0])
    bent = _session(tmp_path / "sessions" / "bent", rec)
    assert CH.end(ch, _Exited(), [str(bent)])["ok"] is False
    audit = json.loads((ch / "audit.json").read_text())
    assert audit["python"] == audit["lean"] and "link points are not the coins" in audit["python"], audit


def test_challenger_cases_passes_the_tree_session_and_refuses_the_rest_for_the_leans_reason(ch, rec, tmp_path):
    """93-zk-tree-from.sh's cases, its own-tree session one whose coin tree's root no release names."""
    tree = _session(tmp_path / "tree-from", rec)
    own = _session(tmp_path / "own-tree", {**rec, "coin_tree": {**rec["coin_tree"], "root": _bumped(rec["coin_tree"]["root"])}})
    out = tmp_path / "audit.json"
    p = subprocess.run(["bash", "-c", '. "$0" && challenger_cases "$@"', str(CH.SH), str(ch), str(tree), str(own), str(out)],
                       capture_output=True, text=True, env=CH._env(ch))
    assert p.returncode == 0, p.stderr[-2000:]
    audit = json.loads(out.read_text())
    assert (audit["trees"], audit["hellos_refused"]) == (1, 0) and all(n > 0 for n in audit["opened"].values()), audit
    own_why = f"session own-tree's coins are not the challenger's: {C.NO_TREE}"
    tree_why = "session tree-from's coins are not the challenger's: its "
    assert audit["cases"] == {name: {"python": why, "lean": why} for name, why in {
        "tree-from": None, "own-tree": own_why, "both": own_why,
        "coin-changed": tree_why + "circuit/rep0 round 0 is not coins the challenger opened under its tree after that round's message",
        "point-changed": tree_why + "link points are not the coins the challenger opened as its tree's session round 0 after its "
                                    "link's roots"}.items()}, audit["cases"]


def inner_trees(j: Any) -> Any:
    _request(j, ["log", "sessions"])
    log, ss = _log(j["log"]), j["sessions"]
    if type(ss) is not list or not all(isinstance(s, dict) and sorted(s) == ["name", "public", "tree"]
                                       and type(s["name"]) is str for s in ss):
        raise D.Reject("sessions is not a list of {name, public, tree}")
    return _refusing(lambda: {"reason": RL.tree_held_reason(log, [(s["name"], s["tree"], s["public"]) for s in ss])})


class _Source:
    """`rec_live.TreeSource` on challenger `ch`: the trees it samples and the rounds it opens, each one's release log
    line in `lines`."""

    def __init__(self, ch: C.LiveChallenger) -> None:
        self.ch, self.lines = ch, []

    def tree(self, spec: dict, nonce: bytes | None = None) -> tuple[bytes, bytes, bytes]:
        tree, line = self.ch.sample_tree(spec, nonce)
        self.lines.append(line)
        return tree.nonce, tree.root(), tree.key

    def tree_open(self, root: bytes, stream: str, rnd: int, n: int, after: bytes) -> tuple[list[int], list[int]]:
        coins, words, line = self.ch.open_tree(root.hex(), stream, rnd, n, after)
        self.lines.append(line)
        return coins, words

    def close(self) -> None:
        pass


class Inner:
    """An inner session as rec_live's live-coin part runs it on a coin tree challenger `ch` samples and opens: `lines` the
    tree's release log lines in order, `tree` the session record's ``coin_tree`` and `public` the challenger's live-coin part's record as
    ``public.json`` holds it (`rec_live.record_bytes`). Its ``Hello`` asks for some link points, its ``Commit`` gives each
    table's salted top, then each table's two streams some rounds in a random order across them, each of sampled tops,
    commits, level-0 copies and coin count."""

    def __init__(self, rng: random.Random, ch: C.LiveChallenger) -> None:
        tables = sorted(rng.sample(TABLES, rng.randint(1, 3)))
        self.streams = [f"{t}/rep{r}" for t in tables for r in (0, 1)]
        self.spec = CT.Spec.of_json({"block": rng.choice([1, 2, 3, 5]), "rounds_log": rng.choice([0, 1, 2]),
                                     "scheme": CT.SCHEME, "streams": [CT.SESSION] + self.streams})
        self.source = _Source(ch)
        cs = RL.LiveCoinPart(rng.randbytes, self.spec, self.source)
        block = self.spec.block
        points = rng.randint(1, block)
        cs.start({"link": {"m": rng.randint(1, block // points), "points": points}})
        cs.commit({t: [rng.randbytes(2 * NODE_BYTES) for _ in range(rng.randint(1, 2))] for t in tables})
        cs.y([rng.getrandbits(128) for _ in range(rng.randint(0, 2))])
        picks = [s for s in self.streams for _ in range(rng.randint(0, 1 << self.spec.rounds_log))]
        rng.shuffle(picks)
        for s in picks:
            if s not in cs.streams:
                cs.open(s, rng.choice([RL.LINK_COINS] + self.streams), rng.randrange(4))
            cs.round(s, [[rng.randbytes(2 * NODE_BYTES)] for _ in range(rng.randint(0, 2))],
                     [rng.randbytes(64) for _ in range(rng.randint(0, 1))], rng.randint(0, block), rng.randint(0, 1))
        self.coins = cs
        self.tree = dict(cs.tree_record)
        self.public = json.loads(RL.record_bytes(cs.record()))

    @property
    def lines(self) -> list[str]:
        return self.source.lines

    def open_unread(self, rng: random.Random) -> bool:
        """The challenger opens a round of the tree past what the record reads (its line appended to `lines`), if a
        stream has one left."""
        opened = Counter({CT.SESSION: 1})
        opened.update(r["stream"] for r in self.public["rounds"])
        left = [s for s in [CT.SESSION] + self.streams if opened[s] < 1 << self.spec.rounds_log]
        if not left:
            return False
        s = rng.choice(left)
        self.source.tree_open(bytes.fromhex(self.tree["root"]), s, opened[s], rng.randint(0, self.spec.block),
                              rng.randbytes(64))
        return True


INNER_TREES_NAMES = ["honest", "no-tree", "coin-tree", "link-root", "link-table", "points", "round-coins", "round-entry",
                     "round-k", "round-stream", "round-twice", "unread", "second-tree", "shared", "unused",
                     "other-record", "record-form", "log-malformed", "request"]


def _record_form(rng: random.Random, p: dict) -> Any:
    """`p`, a live-coin part's record, in a form rec_live never writes."""
    link, rounds = p["link"], p["rounds"]
    t = rng.choice(sorted(link["roots"]))
    what = rng.choice(["public", "link", "points", "point", "roots", "roots-value", "root", "rounds", "round", "stream",
                       "k", "coins", "coin"] if rounds else ["public", "link", "points", "point", "roots", "roots-value",
                                                              "root", "rounds"])
    r = rng.choice(rounds) if rounds else None
    match what:
        case "public":
            return rng.choice([None, [], "x", 5])
        case "link":
            p["link"] = rng.choice([DROP, None, [], "x"])
        case "points":
            link["points"] = rng.choice([DROP, None, {}, "00" * 16])
        case "point":
            link["points"].append(rng.choice([None, 5, "ab" * 15 + "a", "AB" * 16, ["00"]]))
        case "roots":
            link["roots"] = rng.choice([DROP, None, [], "x"])
        case "roots-value":
            link["roots"][t] = rng.choice([None, "", link["roots"][t][0], {}])
        case "root":
            link["roots"][t][0] = rng.choice([link["roots"][t][0].upper(), link["roots"][t][0][:-1], 5, None])
        case "rounds":
            p["rounds"] = rng.choice([DROP, None, {}, "x"])
        case "round":
            rounds[rounds.index(r)] = rng.choice([None, [], "x", 5])
        case "stream":
            r["stream"] = rng.choice([DROP, None, 5, [r["stream"]]])
        case "k":
            r["k"] = rng.choice([DROP, None, -1, True, False, str(r["k"]), [r["k"]]])
        case "coins":
            r["coins"] = rng.choice([DROP, None, "", {}, "00" * 16])
        case "coin":
            r["coins"].append(rng.choice([None, 5, "AB" * 16, "a", ["00"]]))
    return _dropped(p)


def _dropped(v: Any) -> Any:
    """`v` without each key set to `DROP`."""
    if isinstance(v, dict):
        return {k: _dropped(x) for k, x in v.items() if x is not DROP}
    return [_dropped(x) for x in v] if isinstance(v, list) else v


def _mutated(rng: random.Random, s: Inner, others: list[Inner], name: str) -> tuple[Any, Any]:
    """`s`'s tree and record mutated as `name` names: each one `tree_held_reason` refuses, but those of the log and the
    session list (`inner_trees_case`)."""
    tree, p = json.loads(json.dumps(s.tree)), json.loads(json.dumps(s.public))
    link, rounds = p["link"], p["rounds"]
    match name:
        case "no-tree":
            tree = rng.choice([{**tree, "root": rng.randbytes(64).hex()}, _body(tree, root=DROP), {**tree, "root": 5},
                               None, [tree]])
        case "coin-tree":
            f = rng.choice(["key", "nonce", "spec", "extra"])
            if f == "spec":
                tree["spec"] = {**tree["spec"], "block": tree["spec"]["block"] + 1}
            elif f == "extra":
                tree["x"] = 1
            else:
                tree[f] = _flip(tree[f], rng.randrange(64))
        case "link-root":
            t = rng.choice(sorted(link["roots"]))
            nodes = link["roots"][t]
            i = rng.randrange(len(nodes))
            nodes[i] = _flip(nodes[i], rng.randrange(len(nodes[i]) // 2))
        case "link-table":
            t = rng.choice(sorted(link["roots"]))
            if rng.random() < 0.5:
                link["roots"]["x-" + t] = link["roots"].pop(t)
            else:
                link["roots"]["x-" + t] = list(link["roots"][t])
        case "points":
            pts = link["points"]
            what = rng.choice(["flip", "add", "drop"])
            if what == "flip":
                i = rng.randrange(len(pts))
                pts[i] = _flip(pts[i], rng.randrange(16))
            elif what == "add":
                pts.insert(rng.randint(0, len(pts)), rng.choice(pts))
            else:
                pts.pop()
        case "round-coins" | "round-entry" | "round-k" | "round-stream" | "round-twice" | "unread" if rounds:
            r = rng.choice(rounds)
            if name == "round-coins":
                if r["coins"] and rng.random() < 0.5:
                    i = rng.randrange(len(r["coins"]))
                    r["coins"][i] = _flip(r["coins"][i], rng.randrange(16))
                else:
                    r["coins"].append(RL.f128_hex(rng.getrandbits(128)))
            elif name == "round-entry":
                f = rng.choice(["commits", "tops", "level0", "n", "extra"])
                if f == "commits":
                    r["commits"] = r["commits"][1:] if r["commits"] else [rng.randbytes(64).hex()]
                elif f == "tops":
                    r["tops"] = r["tops"] + [[rng.randbytes(2 * NODE_BYTES).hex()]]
                elif f == "extra":
                    r["x"] = rng.choice([0, "x", None])
                else:
                    r[f] += 1
            elif name == "round-k":
                r["k"] = rng.choice([k for k in range(2 << s.spec.rounds_log) if k != r["k"]])
            elif name == "round-stream":
                r["stream"] = rng.choice([x for x in s.streams + [CT.SESSION, "x/rep0"] if x != r["stream"]])
            elif name == "round-twice":
                rest = [x for x in rounds if x != r]
                rounds.insert(rng.randint(0, len(rounds)), dict(r))
                if rest and rng.random() < 0.5:
                    rounds.remove(rng.choice(rest))
            else:
                rounds.remove(r)
        case "round-coins" | "round-entry" | "round-k" | "round-stream" | "round-twice" | "unread":
            link["points"].append(RL.f128_hex(rng.getrandbits(128)))
        case "other-record":
            o = rng.choice(others) if others else Inner(rng, Session(rng).ch)
            p = json.loads(json.dumps(o.public))
        case "record-form":
            p = _record_form(rng, p)
    return tree, p


def inner_trees_case(rng: random.Random, name: str) -> Any:
    rng = _seeded(rng)
    s = Session(rng)
    inners = [Inner(rng, s.ch) for _ in range(rng.randint(1, 3))]
    k = rng.randrange(len(inners))
    t = inners[k]
    extra: list[str] = []
    if name == "unread" and (not t.public["rounds"] or rng.random() < 0.5) and t.open_unread(rng):
        name = "unread-log"
    if name == "second-tree":
        _, line = s.ch.sample_tree(t.spec.to_json())
        extra.append(line)
    log = _interleave(rng, s.lines, [i.lines for i in inners] + [extra])
    sessions = [{"name": rng.choice(SESSIONS) + f"-{n}", "public": i.public, "tree": i.tree} for n, i in enumerate(inners)]
    match name:
        case "shared":
            sessions.insert(rng.randint(k + 1, len(sessions)), {**sessions[k], "name": "again"})
        case "unused":
            del sessions[k]
        case "log-malformed":
            i = log.index(t.lines[0])
            log = log[:i + 1] + [_edit(t.lines[0], key=rng.randbytes(CT.KEY_BYTES).hex())] + log[i + 1:]
        case "honest" | "request" | "second-tree" | "unread-log":
            pass
        case _:
            tree, public = _mutated(rng, t, [o for o in inners if o is not t], name)
            sessions[k] = {**sessions[k], "public": public, "tree": tree}
    j = {"log": log, "sessions": sessions}
    if name == "request":
        what = rng.choice(["keys", "log", "sessions", "entry"])
        if what == "keys":
            j = rng.choice([{"log": log}, {**j, "records": []}, {"sessions": sessions}])
        elif what == "log":
            j["log"] = rng.choice([None, "x", log + [5]])
        elif what == "sessions":
            j["sessions"] = rng.choice([None, {}, sessions[k]])
        else:
            sessions[k] = rng.choice([_body(sessions[k], name=DROP), {**sessions[k], "name": 5},
                                      {**sessions[k], "x": 1}, [sessions[k]["tree"], sessions[k]["public"]]])
    return j


INNER_TREES = D.Port(PIECES, "challenger_inner_trees", inner_trees, _cases(inner_trees_case, INNER_TREES_NAMES),
                     package=PACKAGE, of_record="lean")

pieces = D.server_fixture(INNER_TREES)


def _clean(kind: str, text: str) -> bool:
    return kind == "output" and '"refused"' not in text and '"reason":"' not in text


def test_the_inner_trees_piece_agrees(pieces):
    """The piece agrees on sampled cases and on every mutation over a few seeds, the honest ones accepted (no reason, no
    refusal) and each mutation refused."""
    stats = D.run(INNER_TREES, examples=200, server=pieces)
    outcomes = Counter()
    for name in INNER_TREES_NAMES:
        for seed in range(6):
            kind, text = D.replay(INNER_TREES, inner_trees_case(random.Random(f"{name}/{seed}"), name), pieces)
            outcomes[name, _clean(kind, text)] += 1
            assert _clean(kind, text) == (name == "honest"), (name, seed, kind, text)
    print(f"challenger_inner_trees: {stats}; swept {sum(outcomes.values())}")


def test_a_served_inner_session_is_held_and_each_bent_copy_refused(pieces):
    """A served inner session on the challenger's tree (`SERVED`) is accepted, every opening the log shows read by its
    record; with one round's number moved, a second tree in the log, or its last round dropped (an opening the record
    doesn't read), both sides refuse it, for the same reason."""
    log = R.log_lines(SERVED / R.RELEASES)
    tree = json.loads((SERVED / "coin_tree.json").read_text())
    public = json.loads(gzip.decompress((SERVED / "public.json.gz").read_bytes()))
    events = Counter(json.loads(line).get("event") for line in log)
    assert events[C.TREE_EVENT] == 1 and events[C.OPEN_EVENT] == 1 + len(public["rounds"])

    def held(log: list[str], sessions: list[tuple[str, dict, dict]]) -> str | None:
        kind, text = D.replay(INNER_TREES, {"log": log, "sessions": [{"name": n, "public": p, "tree": t}
                                                                     for n, t, p in sessions]}, pieces)
        assert kind == "output", text
        return json.loads(text)["reason"]

    assert held(log, [("inner", tree, public)]) is None

    moved = json.loads(json.dumps(public))
    last = moved["rounds"][-1]
    last["k"] += 1
    assert f"{last['stream']} round {last['k']} is not coins the challenger opened" in held(log, [("inner", tree, moved)])

    _, line = C.LiveChallenger("verity/recursion/inner", 0, [("a", 1)], random.Random(0).randbytes).sample_tree(tree["spec"])
    assert C.UNUSED_TREE in held(log + [line], [("inner", tree, public)])

    unread = json.loads(json.dumps(public))
    unread["rounds"].pop()
    assert "openings of its tree and its record" in held(log, [("inner", tree, unread)])
