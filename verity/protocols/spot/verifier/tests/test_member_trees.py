"""A member's session's closing round and the members' sessions one after another (`challenger.closes`,
`member_tree_reason`, `members_order`, `members_tree_reason`), on the cases the Lean's own guards hold
(`Spot.Challenger`'s, the same lines, records and reasons), and on a challenger's own log: the device opens a tree's
closing round as it opens any round, so two sessions it serves one after another, each closed after its evidence, pass,
and the same sessions closed after other evidence, left open, interleaved or named in another order are refused. The
evidence's bytes (`closing.evidence_bytes`, `session_evidence`, `closing_after`) decode to their parts, and a
directory's digest is its files' evidence's."""

from __future__ import annotations

import hashlib
import json

import pytest

from verity.commitments import coin_tree as CT
from verity.commitments.identity import canonical_json_text
from verity.protocols.spot.device import challenger as C
from verity.protocols.spot.verifier.one_stage import closing as K

from test_challenger import entropy

EMPTY = hashlib.sha512(b"").hexdigest()
SPEC = {"block": 512, "rounds_log": 9, "scheme": CT.SCHEME, "streams": [CT.SESSION, "t/rep0", "t/rep1"]}
TABLES = {"t": 8}
ROOT_F, ROOT_B = "f" * 16, "b" * 16
POINTS = ["1" * 32, "2" * 32]


def closing(after: str, **changes) -> C.TreeOpening:
    return C.TreeOpening(**{"stream": CT.SESSION, "round": C.CLOSING_ROUND, "n": 0, "after": after, "coins": EMPTY,
                            **changes})


def logged(*opened: C.TreeOpening) -> dict[str, C.LoggedTree]:
    return {"01": C.LoggedTree("02", "03", "01", {}, list(opened))}


def open_at(tree: str, stream: str) -> str:
    return canonical_json_text({"after": "cd", "coins": "ef", "event": C.OPEN_EVENT, "n": 1, "round": 0,
                                "stream": stream, "tree": tree})


TREE = canonical_json_text({"event": C.TREE_EVENT, "key": "02", "nonce": "03", "root": "01", "spec": {}})
ROOT = canonical_json_text({"event": "root", "index": 0, "label": "sample", "received_at": 0, "root": "ab"})
RELEASE = canonical_json_text({"after": "ab", "event": "release", "index": 0, "label": "sample", "released_at": 0})
S, REP = CT.SESSION, "circuit/rep0"


def test_a_closing_round_is_the_session_streams_second_of_no_coins_after_the_evidence():
    assert C.closes(closing("ab"), "ab") and not C.closes(closing("ab"), "cd")
    assert not C.closes(closing("ab", n=1), "ab")
    assert not C.closes(closing("ab", round=0), "ab")
    assert not C.closes(closing("ab", stream=REP), "ab")
    assert not C.closes(closing("ab", coins="ef"), "ab")


def test_a_members_session_is_refused_unless_its_trees_last_opening_closes_it_after_its_evidence():
    record = {"coin_tree": {"root": "01"}}
    assert C.member_tree_reason(logged(), {}, record, "ab") == C.NO_CLOSING
    assert C.member_tree_reason(logged(closing("cd")), {}, record, "ab") == C.NO_CLOSING
    assert C.member_tree_reason(logged(closing("ab"), C.TreeOpening(REP, 0, 1, "cd", "ef")), {}, record,
                                "ab") == C.NO_CLOSING
    assert C.member_tree_reason({}, {}, record, "ab") == C.NO_TREE
    assert C.member_tree_reason(logged(closing("ab")), {}, {}, "ab") == C.NO_TREE
    assert C.member_tree_reason(logged(closing("ab")), {}, record, "ab") not in (None, C.NO_CLOSING, C.NO_TREE)


@pytest.mark.parametrize("log, order", [
    ([TREE, ROOT, RELEASE], []),
    ([open_at("01", S), open_at("01", REP), open_at("01", S), open_at("02", S), open_at("02", S)], ["01", "02"]),
    ([open_at("01", S), open_at("02", S)], None),
    ([open_at("01", S), open_at("01", REP), open_at("02", S)], None),
    ([open_at("01", S), open_at("01", S), open_at("01", REP)], None),
    ([open_at("01", S), open_at("01", S), open_at("02", S), open_at("02", S), open_at("01", S)], ["01", "02", "01"]),
    (["not a line", open_at("01", S), "{}", open_at("01", S)], ["01"]),
])
def test_the_members_order_is_each_trees_rounds_together_its_closing_round_before_the_next_trees_first(log, order):
    assert C.members_order(log) == order


def coin_tree(c: str) -> dict:
    return {"key": c * 512, "nonce": c * 64, "root": c * 128, "spec": SPEC}


def hello(c: str) -> str:
    return canonical_json_text({"event": C.TREE_EVENT, **coin_tree(c)})


def round_line(c: str, stream: str, k: int, n: int, after: str, coins: str) -> str:
    return canonical_json_text({"after": after, "coins": coins, "event": C.OPEN_EVENT, "n": n, "round": k,
                                "stream": stream, "tree": c * 128})


def member_record(c: str) -> dict:
    return {"coin_tree": coin_tree(c),
            "link": {"points": [[p] for p in POINTS], "root_b": {"t": ROOT_B}, "root_f": ROOT_F},
            "streams": [{"rounds": [{"coins": ["3" * 32], "msg_sha512": "e" * 128, "n": 1}], "stream": "t/rep0"}]}


def evidence(s: str) -> str:
    return hashlib.sha512(s.encode()).hexdigest()


def member_log(c: str, after: str) -> list[str]:
    link = C.link_after(bytes.fromhex(ROOT_F), {"t": bytes.fromhex(ROOT_B)}).hex()
    return [round_line(c, CT.SESSION, 0, 2, link, hashlib.sha512(bytes.fromhex("".join(POINTS))).hexdigest()),
            round_line(c, "t/rep0", 0, 1, "e" * 128, hashlib.sha512(bytes.fromhex("3" * 32)).hexdigest()),
            round_line(c, CT.SESSION, C.CLOSING_ROUND, 0, after, EMPTY)]


def closed(c: str) -> list[str]:
    return member_log(c, evidence(c))


def members(log: list[str], named: list[tuple[str, str]]) -> str | None:
    return C.members_tree_reason(C.coin_trees_of(log), log, [(c, TABLES, member_record(c), ev) for c, ev in named])


A, B = ("a", evidence("a")), ("b", evidence("b"))
HELLOS = [hello("a"), hello("b")]
REFUSED = "session {}'s coins are not the challenger's: {}"
TWO_MEMBERS = {
    "both closed, the trees sampled first": (HELLOS + closed("a") + closed("b"), [A, B], None),
    "both closed, each tree sampled at its session": ([hello("a")] + closed("a") + [hello("b")] + closed("b"), [A, B],
                                                      None),
    "the second closed after other evidence": (HELLOS + closed("a") + closed("b"), [A, ("b", evidence("x"))],
                                               REFUSED.format("b", C.NO_CLOSING)),
    "the first left open": (HELLOS + closed("a")[:-1] + closed("b"), [A, B], REFUSED.format("a", C.NO_CLOSING)),
    "the second begun before the first closed": (HELLOS + closed("a")[:-1] + closed("b")[:1] + closed("a")[2:]
                                                 + closed("b")[1:], [A, B], C.MEMBERS_ORDER),
    "the sessions in the other order": (HELLOS + closed("b") + closed("a"), [A, B], C.MEMBERS_ORDER),
    "one tree two sessions'": ([hello("a")] + closed("a"), [A, A],
                               f"two sessions' coins are one tree's: {C.SHARED_TREE}"),
    "a tree no session's": (HELLOS + closed("a"), [A],
                            f"the challenger's release log shows 2 coin trees and the sessions use 1: {C.UNUSED_TREE}"),
}


@pytest.mark.parametrize("case", sorted(TWO_MEMBERS))
def test_members_sessions_are_held_to_their_trees_each_closed_before_the_next_begins(case):
    log, named, why = TWO_MEMBERS[case]
    assert members(log, named) == why


def test_the_one_statements_check_refuses_a_closed_tree_and_takes_it_without_its_closing_round():
    one = [("a", TABLES, member_record("a"))]
    assert C.sessions_tree_reason(C.coin_trees_of([hello("a")] + closed("a")[:-1]), one) is None
    assert C.sessions_tree_reason(C.coin_trees_of([hello("a")] + closed("a")), one) == REFUSED.format("a", C.EXTRA_OPENING)


def f128(w: int) -> str:
    return f"{w:032x}"


class Served:
    """A challenger and its release log's tree lines, as its host writes them: `session` is one ``--zk`` session on a
    tree of `SPEC`, its link points and one round of ``t/rep0``, and its record; `close` its closing round."""

    def __init__(self, seed: str) -> None:
        self.challenger = C.LiveChallenger("verity/test/coins/v1", 0, [("sample", 1)], entropy(seed))
        self.log: list[str] = []

    def opened(self, root: str, stream: str, rnd: int, n: int, after: bytes) -> list[int]:
        coins, _, line = self.challenger.open_tree(root, stream, rnd, n, after)
        self.log.append(line)
        return coins

    def session(self, nonce: bytes) -> dict:
        tree, line = self.challenger.sample_tree(SPEC, nonce)
        self.log.append(line)
        root, root_f, root_b, message = tree.root().hex(), bytes([7]) * 8, bytes([8]) * 8, bytes([9]) * 64
        points = self.opened(root, CT.SESSION, 0, 2, C.link_after(root_f, {"t": root_b}))
        coins = self.opened(root, "t/rep0", 0, 3, message)
        return {"coin_tree": tree.record(),
                "link": {"points": [[f128(c)] for c in points], "root_b": {"t": root_b.hex()}, "root_f": root_f.hex()},
                "streams": [{"stream": "t/rep0",
                             "rounds": [{"coins": [f128(c) for c in coins], "k": 0, "msg_sha512": message.hex(), "n": 3}]}]}

    def close(self, record: dict, after: bytes) -> None:
        assert self.opened(record["coin_tree"]["root"], CT.SESSION, C.CLOSING_ROUND, 0, after) == []

    def reason(self, *named: tuple[str, dict, bytes]) -> str | None:
        return C.members_tree_reason(C.coin_trees_of(self.log), self.log,
                                     [(name, TABLES, record, after.hex()) for name, record, after in named])


def session_evidence(record: dict, proof: bytes) -> bytes:
    return hashlib.sha512(K.session_evidence(json.dumps(record, sort_keys=True).encode(), [("t.rep0.bin", proof)])).digest()


def test_a_challenger_opens_the_closing_round_it_is_asked_for_and_its_log_holds_two_sessions_one_after_another():
    """The device has no rule of its own for the closing round: it opens it as the session stream's next round, of no
    coins, and logs it; the check of the log is what holds the sessions to it."""
    host = Served("members")
    a = host.session(bytes(range(32)))
    after_a = session_evidence(a, b"proof of a")
    host.close(a, after_a)
    b = host.session(bytes(range(32, 64)))
    after_b = session_evidence(b, b"proof of b")
    assert host.reason(("a", a, after_a), ("b", b, after_b)) == REFUSED.format("b", C.NO_CLOSING)
    host.close(b, after_b)
    closing_line = json.loads(host.log[3])
    assert closing_line == {"after": after_a.hex(), "coins": EMPTY, "event": C.OPEN_EVENT, "n": 0,
                            "round": C.CLOSING_ROUND, "stream": CT.SESSION, "tree": a["coin_tree"]["root"]}
    assert host.reason(("a", a, after_a), ("b", b, after_b)) is None
    assert host.reason(("b", b, after_b), ("a", a, after_a)) == C.MEMBERS_ORDER
    assert host.reason(("a", a, after_a), ("b", b, session_evidence(b, b"another proof"))) == REFUSED.format("b", C.NO_CLOSING)
    assert host.reason(("a", a, session_evidence({**a, "streams": []}, b"proof of a")),
                       ("b", b, after_b)) == REFUSED.format("a", C.NO_CLOSING)
    assert host.reason(("a", a, after_a)) == (f"the challenger's release log shows 2 coin trees and the sessions use 1: "
                                              f"{C.UNUSED_TREE}")
    with pytest.raises(C.CoinError, match="the next round of the coin tree's session stream is 2"):
        host.close(a, after_a)


def test_a_challenger_asked_for_the_next_sessions_link_before_the_closing_round_logs_an_order_the_check_refuses():
    host = Served("interleaved")
    a = host.session(bytes(range(32)))
    b = host.session(bytes(range(32, 64)))
    after_a, after_b = session_evidence(a, b"proof of a"), session_evidence(b, b"proof of b")
    host.close(a, after_a)
    host.close(b, after_b)
    assert host.reason(("a", a, after_a), ("b", b, after_b)) == C.MEMBERS_ORDER


PARTS = [[], [b""], [b"record"], [b"record", b"t.rep0.bin", b"\x00" * 9, b"t.rep1.bin", b""]]


@pytest.mark.parametrize("parts", PARTS)
def test_the_evidences_bytes_hold_each_parts_length_and_bytes_and_decode_to_the_parts(parts):
    b = K.evidence_bytes(parts)
    assert b[:8] == len(parts).to_bytes(8, "big") and len(b) == 8 + sum(8 + len(p) for p in parts)
    assert K.evidence_parts(b) == parts
    assert K.evidence_parts(b + b"\x00") is None
    assert K.evidence_parts(b[:-1]) is None


def test_two_evidences_of_other_parts_are_other_bytes():
    assert K.evidence_bytes([b"ab", b"c"]) != K.evidence_bytes([b"a", b"bc"])
    assert K.evidence_bytes([b"ab"]) != K.evidence_bytes([b"a", b"b"])
    assert K.evidence_parts(b"\x00" * 7) is None
    assert K.evidence_parts((1 << 60).to_bytes(8, "big")) is None


def test_a_sessions_evidence_is_its_record_then_each_proof_files_name_and_bytes_by_name(tmp_path):
    files = {"session.json": b'{"session":"s"}', "t.rep1.bin": b"second", "t.rep0.bin": b"first", "u.rep0.bin": b"",
             "serve.log": b"not evidence"}
    for name, content in files.items():
        (tmp_path / name).write_bytes(content)
    reps = [(name, files[name]) for name in ("t.rep0.bin", "t.rep1.bin", "u.rep0.bin")]
    want = K.session_evidence(files["session.json"], reps)
    assert K.evidence_parts(want) == [files["session.json"], b"t.rep0.bin", b"first", b"t.rep1.bin", b"second",
                                      b"u.rep0.bin", b""]
    assert K.session_files(tmp_path) == (tmp_path / "session.json", [tmp_path / name for name, _ in reps])
    assert K.closing_after(tmp_path) == hashlib.sha512(want).digest()
    (tmp_path / "t.rep0.bin").write_bytes(b"First")
    assert K.closing_after(tmp_path) != hashlib.sha512(want).digest()
