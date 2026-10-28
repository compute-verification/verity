"""The firewall's contract (`Flock/Firewall/Contract.lean`, `flock-firewall contract`) against the inner phase's firewall
(`verity_flock.rec_live.Proxy`), on the transcripts `verity_flock.firewall_transcript` records (`firewall_agree.py`):
honest sessions accepted by both, the red team's six sessions against the firewall refused by both or sent as the schedule has them,
tamper controls (firewalls that let out what the contract forbids) refused by the contract, the firewall review's leaks
each cut at its item, every session `test_firewall_agreement` holds the proxy to `Flock.Firewall.run` on, the inner
session's live coins, coins that do not open a challenger's commitment refused by both, and no salt used twice."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from verity.ml.commitments.rowleaf import ROLE_X, sha512_row_digest
from verity_flock import firewall_transcript as FT
from verity_flock import rec_live as RL

VERIFIER = Path(__file__).resolve().parents[1] / "verifier"
LEAN = VERIFIER.parents[2] / "core" / "service" / "flock"
BIN = LEAN / ".lake/build/bin/flock-firewall"


def _agree():
    spec = importlib.util.spec_from_file_location("firewall_agree", VERIFIER / "firewall_agree.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FA = _agree()


@pytest.fixture(scope="module")
def lean():
    if shutil.which("lake") is None:
        pytest.skip("no Lean toolchain")
    subprocess.run(["lake", "build", "flock-firewall"], cwd=LEAN, check=True, capture_output=True)
    return str(BIN)


def test_a_commit_item_is_the_row_the_coin_server_receives():
    """A commit item's message is the bytes the registered row's digest hashes, and the firewall's ``b || c`` is hm96-sha512's
    commit string of that digest: what `Flock.Firewall.Contract.commitBC` recomputes."""
    s = FA.run_case("honest", FA.requests(), RL.Proxy, None)
    t = s.transcript()
    row = s.proxy.private[1]["rows"][0]
    assert hashlib.sha512(FT.row_message(row)).digest() == sha512_row_digest(row, 16, ROLE_X)
    commits = [i["commit"] for i in t["items"] if "commit" in i]
    assert len(commits) == 2 + 2 * (1 + 1), "the level-0 top's two nodes, then per rep its fresh top's node and its message row"
    for c in commits:
        msg, salt = bytes.fromhex(c["msg"]), bytes.fromhex(t["tape"]["salts"][c["salt"]])
        assert bytes.fromhex(c["sent"]) == FT.INST.commit_string(FT.INST.default_key, hashlib.sha512(msg).digest(), salt)
    assert sorted(c["salt"] for c in commits) == list(range(len(t["tape"]["salts"]))), "each salt drawn is used once"
    sent = {bytes.fromhex(x) for r in s.coins.record()["rounds"] for x in r["commits"] + [y for t in r["tops"] for y in t]}
    sent |= {bytes.fromhex(x) for v in s.coins.record()["link"]["roots"].values() for x in v}
    joined = b"".join(sent)
    assert all(bytes.fromhex(c["sent"]) in joined for c in commits)
    assert t["exposed"] == {"salts": [], "pads": []} and t["firewall"] == {"accepted": True, "why": None}
    assert "schedule" not in t and "relations" not in t, "the public input is the verifier's, never the transcript's"
    assert s.public() == FT.public(FA.SCHEDULE)


def test_the_transcript_is_the_schedule_in_its_order():
    """Each stream's Open just before its first round, wherever the prover sent it."""
    t = FA.run_case("honest", FA.requests(), RL.Proxy, None).transcript()
    fixed = lambda t: [i["fixed"]["what"] for i in t["items"] if "fixed" in i]
    assert fixed(t) == ["hello", "roots", "link.y", "open:circuit/rep0", "round:0", "round:1", "open:circuit/rep1", "round:2",
                        "round:3", "finish"]
    for o in ("reversed", "after-link", "early"):
        assert fixed(FA.run_case(o, FA.requests(order=o), RL.Proxy, None).transcript()) == fixed(t)
    assert list(t["items"][-1]) == ["finish"]
    hello = next(i["fixed"] for i in t["items"] if "fixed" in i and i["fixed"]["what"] == "hello")
    assert bytes.fromhex(hello["sent"]) == FT.canon(FA.SCHEDULE["hello"])


def test_the_contract_and_the_firewall_agree_on_every_case(lean, tmp_path):
    """Every case meets its expectation (`firewall_agree.cases`): the honest sessions accepted by both; the red team's six
    a stop for both, or the schedule's value sent in its place (a Hello spelled with its keys in another order, each
    stream's Open wherever the prover sent it); each tamper control refused by the contract alone."""
    assert FA.main(["--lean", lean, "--out", str(tmp_path), "--cases"]) == 0
    rows = [r.split("\t") for r in (tmp_path / "agreement.tsv").read_text().splitlines()[1:]]
    got = {r[0]: (r[1], r[2]) for r in rows}
    assert len(rows) == len(FA.cases()) == 29
    assert [n for n, v in got.items() if v == ("True", "True")] == ["honest", "honest-pings", "hello-keys-reordered",
                                                                     "gap3-open-reversed", "gap3-open-after-link", "gap3-open-early",
                                                                     "honest-committed"]
    gaps = [n for n in got if n.startswith("gap")]
    assert {n.split("-")[0] for n in gaps} == {f"gap{i}" for i in range(1, 7)}
    assert all(got[n] == ("False", "False") for n in gaps if not n.startswith("gap3"))
    assert all(got[n] == ("True", "False") for n in got if n.startswith("tamper"))
    assert json.loads((tmp_path / "result.json").read_text())["divergences"] == []


def test_the_contract_holds_on_every_session_of_the_agreement_test(lean, tmp_path):
    """`test_firewall_agreement`'s sessions (its cases: each schedule edit, prefix, drop, duplicate and swap of a request,
    requests after Finish, forgeries, byte flips; and every placement of the Opens) through the proxy: the contract accepts
    each session the proxy
    finishes, and of each one it stops releases every item, save where the proxy hands the coin server a call the coin server
    refuses (a Commit before Hello or a second one, a Link before Commit or a second one). Such a call reaches the coin
    server, so the contract's cut is that call; `Flock.Firewall.run` refuses it before it sends anything, and
    `test_firewall_agreement` compares the calls the coin server takes."""
    from test_firewall_agreement import SALTS, _coins, cases, placements

    sessions = []
    for name, schedule, reqs in cases() + placements():
        salts = iter(SALTS)
        s = FT.Session(schedule, _coins, lambda salts=salts: next(salts))
        for r in reqs:
            s.handle(r)
        sessions.append((name, s))
    got = FA.verdicts(lean, tmp_path, [(f"s{i}", s) for i, (_, s) in enumerate(sessions)])
    cut = {"roots": "Commit's roots: not just after Hello", "y": "Link's y: not after Commit"}
    wrong = []
    for i, ((name, s), v) in enumerate(zip(sessions, got)):
        n = len(json.loads((tmp_path / f"s{i}.json").read_text())["items"])
        bad = next((c["call"] for c in s.coins.calls if "refused" in c), None)
        if bad is not None:
            ok = v["accepted"] is False and v["released"] < n and v["why"].startswith(cut[bad])
        else:
            ok = v["accepted"] == s.accepted() and v["released"] == n
            ok = ok and (v["accepted"] or v["why"] == "the session ended without Finish")
        if not ok:
            wrong.append((name, bad, v))
    assert len(sessions) > 1000 and not wrong, wrong[:3]


def test_a_firewall_that_lets_out_what_the_contract_forbids_fails_the_check(lean, tmp_path, monkeypatch):
    """Negative control of the check itself: a tamper control expected to agree (as a firewall that reused salts would be,
    if the contract missed it) is a miss, and the check exits 1."""
    name, reqs, proxy, fresh, _, challenger = next(c for c in FA.cases() if c[0] == "tamper-reused-salt")
    monkeypatch.setattr(FA, "cases", lambda: [(name, reqs, proxy, fresh, {"firewall": True, "lean": True, "why": None}, challenger)])
    assert FA.main(["--lean", lean, "--out", str(tmp_path), "--cases"]) == 1
    missed = json.loads((tmp_path / "result.json").read_text())["missed"]
    assert [m["case"] for m in missed] == [name] and missed[0]["lean"]["why"].startswith("the tape's salts are not pairwise distinct")


def test_the_contract_cuts_every_leak_of_review_r6_at_its_item(lean, tmp_path):
    """The firewall review's leaks (`firewall_agree.leaks`), each a bit the firewall could choose and each accepted before:
    the commitments' count and place (B1), a tape number naming another's value (B2), the public input taken from the
    transcript (B3) and a value of the firewall's own under a name it chose (N1). Each is cut at its item for its reason;
    the honest transcript and a masked value checked against the verifier's relation are accepted."""
    cases = FA.leaks()
    got = FA.decide(lean, tmp_path, [(n, t, p) for n, t, p, _ in cases])
    assert not [(n, v, e) for (n, _, _, e), v in zip(cases, got) if not FA.meets(v, e)]
    review = {"extra-commitment", "dropped-commitment", "moved-commitment", "salt-bytes-reused", "salt-shown-under-another-id",
              "pad-shown-under-another-id", "masked-no-relations", "masked-own-relation", "statement-value-unchecked",
              "schedule-from-the-transcript", "own-item-chosen"}
    names = [n for n, *_ in cases]
    assert review <= set(names) and len(names) == len(set(names))
    assert {n for n, (*_, e) in zip(names, cases) if e["accepted"]} == {"honest", "honest-committed", "masked-checked"}


def test_the_contract_reads_a_round_only_in_rec_step3s_form(lean, tmp_path):
    """The public schedule's rounds are read in `rec_live.check_schedule`'s form and no other, and every field of one is
    pinned: the honest transcript under a schedule whose round 1 has v0's ``caps``, no ``level0``, another key or two
    message rows is cut at that round's item, as it is when the round's ``tops`` or ``level0`` differ from what the
    firewall sent; under v0's kind it is cut at Hello."""
    s = FA.run_case("honest", FA.requests(), RL.Proxy, None)
    t, sched = s.transcript(), s.public()["schedule"]
    at = {it["fixed"]["what"]: i for i, it in enumerate(t["items"]) if "fixed" in it}
    form = f"the schedule's round:1: not a round of {RL.KIND}/schedule"
    edits = {"v0-caps": (lambda r: r.update(caps=r.pop("tops")), form), "no-level0": (lambda r: r.pop("level0"), form),
             "another-key": (lambda r: r.update(note=0), form),
             "two-rows": (lambda r: r.update(commits=2), "the schedule's round:1: 2 message rows"),
             "other-tops": (lambda r: r.update(tops=[2]), "round:1: not the value the schedule fixes"),
             "other-level0": (lambda r: r.update(level0=1), "round:1: not the value the schedule fixes")}
    named, want = [], []
    for name, (edit, why) in edits.items():
        p = copy.deepcopy(sched)
        edit(p["rounds"][1])
        named.append((name, t, FT.public(p)))
        want.append((at["round:1"], why))
    named.append(("v0-kind", t, FT.public({**sched, "kind": "rec/inner-transcript/v0/schedule"})))
    want.append((at["hello"], f"Hello: the public schedule is not a {RL.KIND}/schedule"))
    got = FA.decide(lean, tmp_path, named)
    assert [(v["accepted"], v["released"]) for v in got] == [(False, n) for n, _ in want], got
    assert all(v["why"].startswith(w) for v, (_, w) in zip(got, want)), got


def test_a_public_input_the_contract_cannot_read_decides_nothing(lean, tmp_path):
    """`contract --public` with a file that is not a public input exits 2 and prints no transcript's verdict."""
    s = FA.run_case("honest", FA.requests(), RL.Proxy, None)
    (tmp_path / "t.json").write_text(json.dumps(s.transcript()))
    (tmp_path / "p.json").write_text(json.dumps({**s.public(), "kind": FT.FORMAT}))
    p = subprocess.run([lean, "contract", "--public", str(tmp_path / "p.json"), str(tmp_path / "t.json")],
                       capture_output=True, text=True)
    assert p.returncode == 2 and [list(json.loads(x)) for x in p.stdout.splitlines()] == [["error", "public"]]


@pytest.mark.parametrize("commit", [False, True], ids=["live", "committed"])
def test_a_recorded_session_replays_to_its_record(lean, tmp_path, commit):
    """`firewall_agree.replay` on a session `rec_live` wrote, on live coins or on coins the challenger committed to (then
    with its ``coins.json``): the same coin server record from its own coins and salts, and both accept it."""
    s = FA.run_case("honest", FA.requests(), RL.Proxy, None, challenger={"commit": commit})
    d = tmp_path / "s"
    s.proxy.write(d)
    (d / "schedule.json").write_text(json.dumps(FA.SCHEDULE))
    assert (d / "coins.json").exists() == commit
    r, bad = FA.replay(d)
    assert bad is None and r.coins.record() == s.coins.record() and r.accepted()
    assert FA.main(["--lean", lean, "--out", str(tmp_path / "out"), "--sessions", str(d)]) == 0


def test_the_inner_sessions_coins_are_live():
    """The inner session's challenger commits to no coin (`rec_live.serve`'s, `firewall_transcript.Session`'s by default):
    its public input holds no commitment, no coin item of the honest transcript carries an opening, and each coin item
    comes after every commitment the schedule owes before it, so each round's coins are drawn after its commitments."""
    s = FA.run_case("honest", FA.requests(), RL.Proxy, None)
    t, pub = s.transcript(), s.public()
    assert s.commitments is None and s.coins.committed is None and "coins" not in pub
    items = t["items"]
    coins = [i for i, it in enumerate(items) if "coins" in it]
    assert [(items[i]["coins"]["stream"], items[i]["coins"]["round"]) for i in coins] == \
        [(RL.LINK_COINS, 0)] + [(st, k) for st in FA.STREAMS for k in (0, 1)]
    assert not any("opening" in items[i]["coins"] for i in coins)
    # Commit's two level-0 nodes before the link points; each rep's round 1, its fresh top's node and its message row
    for i, owed in zip(coins, [2, 0, 2, 0, 2]):
        last = max(j for j in range(i) if "fixed" in items[j])
        assert [list(items[j]) for j in range(last + 1, i)] == [["commit"]] * owed, i


def test_each_coin_item_opens_its_commitment_in_the_public_input():
    """A challenger that commits to every round's coins before the session (`rec_live.CoinServer.commit_coins`): the public
    input holds one commitment a round, the link points as round 0 of ``link``, and each coin item of the committed
    transcript opens its own under the salt sent with it (`rec_live.coin_commitment`, `Flock.Firewall.Contract.commitBC` of
    `coinMsg`). The challenger's salts are its own: none is the firewall's, and none opens two rounds."""
    s = FA.run_case("honest-committed", FA.requests(), RL.Proxy, None, challenger={"commit": True})
    t, pub = s.transcript(), s.public()
    items = [i["coins"] for i in t["items"] if "coins" in i]
    assert [(c["stream"], c["round"]) for c in items] == [(RL.LINK_COINS, 0)] + [(st, k) for st in FA.STREAMS for k in (0, 1)]
    assert {st: len(cs) for st, cs in pub["coins"].items()} == {RL.LINK_COINS: 1, **{st: 2 for st in FA.STREAMS}}
    for c in items:
        want = RL.coin_commitment(c["stream"], c["round"], [int(x, 16) for x in c["coins"]], bytes.fromhex(c["opening"]))
        assert want.hex() == pub["coins"][c["stream"]][c["round"]] and c["relayed"] == c["coins"]
    openings = [c["opening"] for c in items]
    assert len(set(openings)) == len(openings) and not set(openings) & set(t["tape"]["salts"])
    assert s.coins.commitment_record() == {"kind": RL.COINS_KIND, "coins": pub["coins"],
                                           "openings": {st: [c["opening"] for c in items if c["stream"] == st] for st in pub["coins"]}}


def test_coins_that_do_not_open_the_challengers_commitment_are_refused(lean, tmp_path):
    """The firewall stops a session whose live coins come with an opening, or whose committed coins come with a salt that
    does not open their round's commitment, or with none (before those coins reach the workers); the contract refuses each
    such session, and refuses it at the coin item when a firewall serves it anyway; it accepts the honest session on live
    coins and the honest committed session."""
    named = {c[0]: c for c in FA.cases()}
    stops = {"challenger-stray-opening": "link round 0: an opening of coins the challenger never committed to",
             "challenger-wrong-opening": "link round 0: coins that do not open the challenger's commitment",
             "challenger-no-opening": "circuit/rep0 round 1: coins without the opening of the challenger's commitment"}
    names = ["honest", "honest-committed", *stops, "tamper-stray-opening", "tamper-wrong-opening", "tamper-no-opening"]
    sessions = []
    for name in names:
        _, reqs, proxy, fresh, expect, challenger = named[name]
        sessions.append((name, FA.run_case(name, reqs, proxy, fresh, challenger=challenger), expect))
    got = FA.verdicts(lean, tmp_path, [(n, s) for n, s, _ in sessions])
    for (name, s, expect), v in zip(sessions, got):
        assert (s.accepted(), v["accepted"]) == (expect["firewall"], expect["lean"]) and (v["why"] or "").startswith(expect["why"] or ""), \
            (name, v)
    s = {n: s for n, s, _ in sessions}
    assert {n: s[n].proxy.stopped for n in stops} == {n: f"rec proxy: FIREWALL-REFUSED: {w}" for n, w in stops.items()}
    assert [len(s[n].relayed) for n in stops] == [0, 0, 2]
    cut = {n: v for (n, _, _), v in zip(sessions, got)}
    assert cut["honest"]["accepted"] and cut["honest-committed"]["accepted"]
    for n, (st, k) in {"tamper-stray-opening": (RL.LINK_COINS, 0), "tamper-wrong-opening": (RL.LINK_COINS, 0),
                       "tamper-no-opening": ("circuit/rep0", 1)}.items():
        items = json.loads((tmp_path / f"{n}.json").read_text())["items"]
        assert items[cut[n]["released"]]["coins"]["stream"] == st and items[cut[n]["released"]]["coins"]["round"] == k


def test_a_salt_used_twice_in_a_session_is_refused(lean, tmp_path):
    """Within a session the contract refuses any salt used twice, whoever drew it: a firewall whose salt source repeats (its
    tape's salts not pairwise distinct), two commitments under one salt number, and one salt's bytes under two numbers."""
    _, reqs, proxy, fresh, expect, challenger = next(c for c in FA.cases() if c[0] == "tamper-reused-salt")
    s = FA.run_case("tamper-reused-salt", reqs, proxy, fresh, challenger=challenger)
    t = s.transcript()
    assert s.accepted() and len(set(t["tape"]["salts"])) == 1 < len(t["tape"]["salts"])
    leaks = {n: (t, p, e) for n, t, p, e in FA.leaks() if n in ("salt-number-reused", "salt-bytes-reused")}
    got = FA.decide(lean, tmp_path, [("reused", t, s.public())] + [(n, t, p) for n, (t, p, _) in leaks.items()])
    assert got[0]["accepted"] is False and got[0]["released"] == 0 and got[0]["why"].startswith("the tape's salts are not pairwise distinct")
    assert all(FA.meets(v, e) for v, (_, _, e) in zip(got[1:], leaks.values())), got


def test_no_salt_repeats_across_a_thousand_sessions():
    """Across 1,000 honest sessions, each on the firewall's own salt source (`hm96.SHA512.fresh_salt`, 192 bytes from the
    operating system's generator) and that of a challenger that commits to its coins, every salt either drew is distinct
    from every other: the firewall's tapes and the challenger's openings, within and across sessions."""
    salts: list[str] = []
    for _ in range(1000):
        s = FT.Session(FA.SCHEDULE, commit=True)
        for r in FA.requests():
            s.handle(r)
        t = s.transcript()
        assert s.accepted() and t["exposed"] == {"salts": [], "pads": []}
        salts += t["tape"]["salts"] + [i["coins"]["opening"] for i in t["items"] if "coins" in i]
    assert len(salts) == 1000 * (6 + 5) and len(set(salts)) == len(salts)
