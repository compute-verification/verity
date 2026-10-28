"""N2's period draw (`LinkWindows.period_draw`) and the sampled network verdict (`network_verdict(drawn=…)`): the
period's salted roots committed before any coin is read, K of its link-windows drawn, only those registered and proved,
the sample of record the Lean verifier's on the coins the sample read (`lean_draw`, stood in here by the sampler
`coins.OwnRandomness` that `test_coins` holds byte for byte to Lean's `Flock.Draw.drawOn`), and what the sample
established stated as wrong units at δ."""
from __future__ import annotations

import itertools
import json
import sys
from dataclasses import replace

import pytest

from verity.core.protocols.certifier import service as SVC
from verity.core.protocols.certifier.certificate import CertifiedRecord
from verity.core.protocols.certifier.service import PERIOD_DRAW, PERIOD_DRAWN
from verity.core.protocols.sampled_proofs import service as S
from verity.core.protocols.sampled_proofs.coins import OwnRandomness

from test_network_certifier_service import SALTS, Run, lean  # noqa: F401  (lean: the stand-in Lean, autouse)


@pytest.fixture(autouse=True)
def lean_draw(monkeypatch):
    """The Lean's sample stood in by its Python twin on the same bytes; each call's (k, n, coins) recorded."""
    calls = []

    def draw(k: int, n: int, read: bytes) -> dict:
        calls.append((k, n, read))
        at = [0]

        def source(m: int) -> bytes:
            at[0] += m
            return read[at[0] - m:at[0]]
        try:
            units = OwnRandomness(source).subset(k, n)
        except ValueError:
            return {"refused": "draw: the stream ran out"}
        return {"law": "subset", "population": n, "k": k, "units": sorted(units)}
    monkeypatch.setattr(SVC, "lean_draw", draw)
    return calls


class Coins:
    """A coin source with a `commit`, as a remote challenger's: what it was committed to, and when it was read."""

    def __init__(self, byte: int = 0x5a) -> None:
        self.byte, self.events = byte, []

    def commit(self, after: bytes) -> None:
        self.events.append(("commit", after.hex()))

    def __call__(self, n: int) -> bytes:
        self.events.append(("read", n))
        return bytes([self.byte]) * n


def committed(run: Run, windows=range(3)) -> dict:
    """The prover's openings of the period's windows, none registered: window -> opening."""
    return {w: run.honest(w) for w in windows}


def roots(run: Run, openings: dict) -> list[str]:
    return [run.lw.root(openings[w]) for w in sorted(openings)]


def prove(run: Run, drawn: dict, openings: dict, windows=None) -> list[S.Outcome]:
    """The drawn windows (or `windows`) registered under the period draw, in order, and each decided on a proof."""
    out = []
    for w in drawn["units"] if windows is None else windows:
        reg = run.lw.register(run.stream, run.lw.registration(openings[w], run.previous), drawn)
        run.previous = reg.digest
        out.append(run.decide(run.stream.select(reg), openings[w]))
    return out


def test_the_period_draw_commits_every_root_before_it_reads_a_coin(tmp_path):
    run, coins = Run(tmp_path), Coins()
    openings = committed(run)
    drawn = run.lw.period_draw(run.stream, roots(run, openings), 2, coins)
    assert coins.events[0] == ("commit", drawn["commitment"]) and len(coins.events) > 1
    assert all(e[0] == "read" for e in coins.events[1:])
    assert drawn["read"] == (bytes([coins.byte]) * sum(e[1] for e in coins.events[1:])).hex()
    assert drawn["commitment"] == S._sha512({"format": PERIOD_DRAW, "stream": run.stream.digest,
                                             "roots": roots(run, openings), "advice": None})
    assert drawn["format"] == PERIOD_DRAW and drawn["k"] == 2 and drawn["population"] == 3
    assert len(drawn["units"]) == 2 and drawn["units"] == sorted(set(drawn["units"])) <= [0, 1, 2]
    with pytest.raises(ValueError, match="1 <= K <= 3"):
        run.lw.period_draw(run.stream, roots(run, openings), 4, Coins())
    with pytest.raises(ValueError, match="SHA-512 in lowercase hex"):
        run.lw.period_draw(run.stream, ["00"] * 3, 2, Coins())
    with pytest.raises(ValueError, match="advice's digest"):
        run.lw.period_draw(run.stream, roots(run, openings), 2, Coins(), advice="00" * 64)
    prove(run, drawn, openings)
    with pytest.raises(ValueError, match="before the stream's first registration"):
        run.lw.period_draw(run.stream, roots(run, openings), 2, Coins())


def test_all_samples_every_window_and_reads_no_coin(tmp_path, lean_draw):
    """K = n leaves nothing to sample: no coin is read, no Lean draw runs, and the sample of record is every window."""
    run, coins = Run(tmp_path), Coins()
    openings = committed(run)
    drawn = run.lw.period_draw(run.stream, roots(run, openings), None, coins)
    assert drawn["units"] == [0, 1, 2] and drawn["k"] == 3 and coins.events == [] and drawn["read"] == ""
    assert all(o.accepted for o in prove(run, drawn, openings))
    verdict = run.lw.network_verdict(run.stream, drawn=drawn)
    assert verdict["accepted"] and verdict["wrong_units"]["bound"] == 0 and verdict["why"] == {}
    assert verdict["selection"]["by"] == "all" and lean_draw == []
    assert verdict["records"] == run.lw.network_verdict(run.stream)["records"]


def test_a_sampled_link_is_accepted_on_its_sampled_windows_and_states_its_incorrect_blocks_at_delta(tmp_path, lean_draw):
    """Two of three windows drawn, registered and proved, the third never registered: the AND over every window refuses
    the missing one, the sampled verdict accepts on the Lean's sample from the very coins the period flip read, and
    K = 2 of 3 leaves at most one wrong window at any δ < 1/3."""
    run = Run(tmp_path)
    openings = committed(run)
    drawn = run.lw.period_draw(run.stream, roots(run, openings), 2, Coins())
    assert all(o.accepted for o in prove(run, drawn, openings))
    (undrawn,) = {0, 1, 2} - set(drawn["units"])
    assert [r["window"]["window"] for r in run.stream.log] == drawn["units"]
    assert not run.lw.network_verdict(run.stream)["accepted"]
    verdict = run.lw.network_verdict(run.stream, drawn=drawn)
    assert lean_draw == [(2, 3, bytes.fromhex(drawn["read"]))] and drawn["read"]
    assert verdict["accepted"] and verdict["refused"] == [] and verdict["records"][undrawn] is None
    assert verdict["why"] == {}
    assert all(verdict["records"][w] is not None for w in drawn["units"])
    assert verdict["selection"] == {"law": "subset", "k": 2, "population": 3, "units": drawn["units"],
                                    "commitment": drawn["commitment"], "by": "lean"}
    wrong = verdict["wrong_units"]
    assert (wrong["bound"], wrong["of"], wrong["delta"]) == (1, 3, run.stream.delta) and wrong["epsilon"] == 1 / 3


def _other(drawn: dict, n: int = 3) -> list[int]:
    """The first K-subset of the n windows, in order, that isn't the sample's."""
    return next(list(c) for c in itertools.combinations(range(n), drawn["k"]) if list(c) != drawn["units"])


@pytest.mark.parametrize("k", [1, 2])
def test_a_prover_that_proves_another_k_subset_is_refused_by_name(tmp_path, k):
    """The prover registers and proves K windows the sample didn't hold, every one accepted, under a sample object listing
    them (the period's own coins and commitment, which hold): the verdict draws its own with the Lean on those coins and
    refuses the link by name, `period-draw`, whether it is handed the prover's object or the challenger's."""
    run = Run(tmp_path)
    openings = committed(run)
    drawn = run.lw.period_draw(run.stream, roots(run, openings), k, Coins())
    forged = {**drawn, "units": _other(drawn)}
    assert all(o.accepted for o in prove(run, forged, openings))
    for given in (forged, drawn):
        verdict = run.lw.network_verdict(run.stream, drawn=given)
        assert not verdict["accepted"] and set(verdict["why"]) == {PERIOD_DRAWN}
        assert verdict["why"][PERIOD_DRAWN] | {"why": None} == {"lean": drawn["units"], "drawn": given["units"],
                                                                "registered": forged["units"], "why": None}
        assert verdict["selection"]["units"] == drawn["units"] and verdict["selection"]["by"] == "lean"
        assert set(verdict["refused"]) <= set(drawn["units"]) and verdict["refused"]


def test_a_lean_that_refuses_or_samples_otherwise_refuses_the_link_by_name(tmp_path, monkeypatch):
    """An honest period, proved: the verdict refuses it by name where the Lean can't draw on its coins, draws other
    windows than the sample object lists, or answers with a sample that isn't subset:K over the period's windows."""
    run = Run(tmp_path)
    openings = committed(run)
    drawn = run.lw.period_draw(run.stream, roots(run, openings), 2, Coins())
    assert all(o.accepted for o in prove(run, drawn, openings))
    assert run.lw.network_verdict(run.stream, drawn=drawn)["accepted"]
    other = {"law": "subset", "population": 3, "k": 2, "units": _other(drawn)}
    for answer in ({"refused": "draw: the stream ran out"}, other, {**other, "units": drawn["units"], "k": 3},
                   {**other, "units": drawn["units"], "population": 4}, {**other, "units": [drawn["units"][0]] * 2},
                   {**other, "units": drawn["units"], "law": "bernoulli"}):
        monkeypatch.setattr(SVC, "lean_draw", lambda k, n, read, answer=answer: answer)
        verdict = run.lw.network_verdict(run.stream, drawn=drawn)
        assert not verdict["accepted"] and set(verdict["why"]) == {PERIOD_DRAWN}, answer
        why = verdict["why"][PERIOD_DRAWN]
        if answer is other:
            assert why["lean"] == other["units"] and verdict["selection"]["by"] == "lean"
        else:
            assert "refused" in why["lean"] and why["why"] and verdict["selection"]["by"] is None


def test_lean_draw_runs_flock_verify_draw_on_the_coins(tmp_path, monkeypatch):
    """`lean_draw` runs the Lean verifier of record (`proofs.executable`) as ``draw --population n --k k
    --stream-file F``, F the coins, and returns its one ``DRAW`` object; a failed run, no ``DRAW`` line or one that
    isn't JSON is a refusal, and so is a Lean that can't be built."""
    monkeypatch.undo()
    seen = tmp_path / "seen.json"
    fake = tmp_path / "flock-verify"
    answer = json.dumps({"law": "subset", "population": 5, "k": 2, "units": [1, 4]})
    fake.write_text(f"#!{sys.executable}\n"
                    "import json, os, sys\n"
                    "a = sys.argv[1:]\n"
                    "coins = open(a[a.index('--stream-file') + 1], 'rb').read().hex()\n"
                    f"open({str(seen)!r}, 'w').write(json.dumps({{'argv': a[:-1], 'coins': coins}}))\n"
                    "mode = os.environ.get('FAKE_DRAW', 'ok')\n"
                    f"if mode == 'ok': print('DRAW ' + {answer!r})\n"
                    "if mode == 'bad': print('DRAW {')\n"
                    "sys.exit(1 if mode == 'exit' else 0)\n")
    fake.chmod(0o755)
    monkeypatch.setattr(SVC.PF, "executable", lambda: fake)
    assert SVC.lean_draw(2, 5, b"\x01\x02\xff") == {"law": "subset", "population": 5, "k": 2, "units": [1, 4]}
    assert json.loads(seen.read_text()) == {"argv": ["draw", "--population", "5", "--k", "2", "--stream-file"],
                                            "coins": "0102ff"}
    for mode in ("exit", "none", "bad"):
        monkeypatch.setenv("FAKE_DRAW", mode)
        assert "refused" in SVC.lean_draw(2, 5, b"\x01"), mode

    def unbuilt():
        raise SVC.PF.LeanError("lake is not on the PATH")
    monkeypatch.setattr(SVC.PF, "executable", unbuilt)
    assert "can't run" in SVC.lean_draw(2, 5, b"\x01")["refused"]


def test_registration_follows_the_sample_and_the_committed_roots(tmp_path):
    run = Run(tmp_path)
    openings = committed(run)
    drawn = run.lw.period_draw(run.stream, roots(run, openings), 1, Coins())
    (w,) = drawn["units"]
    other = min({0, 1, 2} - {w})
    with pytest.raises(S.Refused, match="window"):
        run.lw.register(run.stream, run.lw.registration(openings[other]), drawn)
    eg, ig = openings[w].records
    resalted = run.lw.commit(w, eg, ig, SALTS[other])
    with pytest.raises(S.Refused, match="period"):
        run.lw.register(run.stream, run.lw.registration(resalted), drawn)
    assert not run.stream.log
    prove(run, drawn, openings)
    with pytest.raises(S.Refused, match="period"):
        run.lw.register(run.stream, run.lw.registration(openings[other], run.previous), drawn)


def test_a_rejected_sampled_window_rejects_the_link(tmp_path):
    run = Run(tmp_path)
    openings = committed(run)
    w = 1
    eg, ig = openings[w].records
    forged = CertifiedRecord(eg.head, eg.salt, replace(eg.certificate, signature=bytes(len(eg.certificate.signature))))
    openings[w] = run.lw.commit(w, forged, ig, SALTS[w])
    drawn = run.lw.period_draw(run.stream, roots(run, openings), 3, Coins())
    outs = prove(run, drawn, openings)
    assert [o.accepted for o in outs] == [True, False, True]
    verdict = run.lw.network_verdict(run.stream, drawn=drawn)
    assert not verdict["accepted"] and verdict["refused"] == [w]


def test_the_sample_is_held_to_this_stream_and_the_period_it_committed(tmp_path):
    run = Run(tmp_path)
    openings = committed(run)
    drawn = run.lw.period_draw(run.stream, roots(run, openings), 2, Coins())
    another = run.lw.stream(budget=8, delta=run.stream.delta, challenger="another", coins=lambda n: b"\x07" * n)
    with pytest.raises(ValueError, match="this stream's"):
        run.lw.network_verdict(another, drawn=drawn)
    swapped = list(reversed(drawn["roots"]))
    for bad in ({**drawn, "roots": swapped}, {**drawn, "advice": "00" * 64}, {**drawn, "stream": "00" * 64}):
        with pytest.raises(ValueError, match="this stream's"):
            run.lw.network_verdict(run.stream, drawn=bad)
    for bad in ({**drawn, "units": drawn["units"][:1]}, {**drawn, "population": 4}, {**drawn, "units": [2, 0]},
                {**drawn, "format": "other"}, {**drawn, "k": 0, "units": []}, {**drawn, "roots": drawn["roots"][:2]},
                {**drawn, "read": "zz"}, {**drawn, "read": "abc"}, {k: v for k, v in drawn.items() if k != "read"}):
        with pytest.raises(ValueError, match="not a period draw"):
            run.lw.network_verdict(run.stream, drawn=bad)


def test_a_sampled_window_missing_or_past_the_sample_refuses_a_sampled_link(tmp_path):
    run = Run(tmp_path)
    openings = committed(run)
    drawn = run.lw.period_draw(run.stream, roots(run, openings), 2, Coins())
    first, last = drawn["units"]
    prove(run, drawn, openings, [first])
    assert run.lw.network_verdict(run.stream, drawn=drawn)["refused"] == [last]
    prove(run, drawn, openings, [last])
    assert run.lw.network_verdict(run.stream, drawn=drawn)["accepted"]
    (undrawn,) = {0, 1, 2} - set(drawn["units"])
    reg = run.stream.register(run.lw.registration(openings[undrawn], run.previous), run.lw.ports(undrawn))
    assert reg.order == 2
    assert run.lw.network_verdict(run.stream, drawn=drawn)["refused"] == [3]
