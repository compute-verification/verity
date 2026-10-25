"""N2's period draw (`LinkWindows.period_draw`) and the sampled network verdict (`network_verdict(drawn=…)`): the
period's salted roots committed before any coin is read, K of its link-windows drawn, only those registered and proved,
and what the draw established stated as wrong units at δ."""
from __future__ import annotations

from dataclasses import replace

import pytest

from verity.core.protocols.certifier.certificate import CertifiedRecord
from verity.core.protocols.certifier.service import PERIOD_DRAW
from verity.core.protocols.sampled_proofs import service as S

from test_network_certifier_service import SALTS, Run, lean  # noqa: F401  (lean: the stand-in Lean, autouse)


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


def test_all_draws_every_window_and_reads_no_coin(tmp_path):
    run, coins = Run(tmp_path), Coins()
    openings = committed(run)
    drawn = run.lw.period_draw(run.stream, roots(run, openings), None, coins)
    assert drawn["units"] == [0, 1, 2] and drawn["k"] == 3 and coins.events == []
    assert all(o.accepted for o in prove(run, drawn, openings))
    verdict = run.lw.network_verdict(run.stream, drawn=drawn)
    assert verdict["accepted"] and verdict["wrong_units"]["bound"] == 0
    assert verdict["records"] == run.lw.network_verdict(run.stream)["records"]


def test_a_sampled_link_is_accepted_on_its_drawn_windows_and_states_its_wrong_units_at_delta(tmp_path):
    """Two of three windows drawn, registered and proved, the third never registered: the AND over every window refuses
    the missing one, the sampled verdict accepts, and K = 2 of 3 leaves at most one wrong window at any δ < 1/3."""
    run = Run(tmp_path)
    openings = committed(run)
    drawn = run.lw.period_draw(run.stream, roots(run, openings), 2, Coins())
    assert all(o.accepted for o in prove(run, drawn, openings))
    (undrawn,) = {0, 1, 2} - set(drawn["units"])
    assert [r["window"]["window"] for r in run.stream.log] == drawn["units"]
    assert not run.lw.network_verdict(run.stream)["accepted"]
    verdict = run.lw.network_verdict(run.stream, drawn=drawn)
    assert verdict["accepted"] and verdict["refused"] == [] and verdict["records"][undrawn] is None
    assert all(verdict["records"][w] is not None for w in drawn["units"])
    assert verdict["selection"] == {"law": "subset", "k": 2, "population": 3, "units": drawn["units"],
                                    "commitment": drawn["commitment"]}
    wrong = verdict["wrong_units"]
    assert (wrong["bound"], wrong["of"], wrong["delta"]) == (1, 3, run.stream.delta) and wrong["epsilon"] == 1 / 3


def test_registration_follows_the_draw_and_the_committed_roots(tmp_path):
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


def test_a_rejected_drawn_window_rejects_the_link(tmp_path):
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


def test_the_draw_is_held_to_this_stream_and_the_period_it_committed(tmp_path):
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
                {**drawn, "format": "other"}, {**drawn, "k": 0, "units": []}, {**drawn, "roots": drawn["roots"][:2]}):
        with pytest.raises(ValueError, match="not a period draw"):
            run.lw.network_verdict(run.stream, drawn=bad)


def test_a_drawn_window_missing_or_past_the_draw_refuses_a_sampled_link(tmp_path):
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
