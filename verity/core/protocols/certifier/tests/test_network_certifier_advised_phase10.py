"""The advised link's tie (`advised.IoTieVerdict`), held as regressions. The verdict comes only from IoTieRequest
proofs the verifier of record verified, so that without them every advised link is refused (PROTOCOL.md, "The tie is
IoTieRequest's verdict"). Were the verifier to read it off the delivered `LinkAdvice`, with the verdict holding only
the advice's digest, which the delivered advice fixes, a prover that delivered its own verdict would reopen the
swapped split's acausal release through the verifier of record. So the advice carries no tie, the verifier takes the
verdict only from its own tie source (`LinkWindows.ties`: `tie_service.Ties` in the package, here the stand-in
`advised_traffic.StandInTie`), and the verdict names the link and its ingress registrations, which `view` holds to the
verifier's, so one link's verdict isn't another's.
"""

import ast
import inspect
from dataclasses import fields, replace
from pathlib import Path

import pytest

from verity.core.protocols.certifier import advice as ADV, advised as AD, service as SVC
from verity.core.protocols.certifier.service import LinkWindows

import signers
import standin
from advised_traffic import (FRAMING, LINK, P, PIN, SYNC_SET, SYNCS, T, WINDOWS, Session, StandInClocks, StandInTie,
                             counts, egress_records, held, ingress_records)
from test_network_certifier_advised import Run

#: the certified ingress: a 7-token request (3 frames) and then a 2-token one (1 frame), both arriving in bucket 0,
#: released 2 a bucket, so the first request's last frame is in only in bucket 1 (ready in bucket 2)
CERTIFIED = (7, 2)
FRAMES = tuple(FRAMING[p - 1] for p in CERTIFIED)
#: request 0's output released in bucket 1 (acausal for the 7-token request), request 1's in bucket 3
RELEASE = (1, 3)


@pytest.fixture(autouse=True)
def lean(monkeypatch):
    monkeypatch.setattr(SVC.PF, "verify", standin.lean)


def _advice(prompt_lens, release=RELEASE):
    """Requests of `prompt_lens` tokens in first-row order, each prefilled whole in its own step, one kept output
    each, released at `release`."""
    reqs = tuple(ADV.RequestAdvice(f"r{i}", p, 1, "length") for i, p in enumerate(prompt_lens))
    steps = tuple(ADV.StepAdvice((ADV.Row(i, "prefill", 0, p),)) for i, p in enumerate(prompt_lens))
    return ADV.Advice(reqs, steps, release)


def _session(release=RELEASE):
    """The certified records of the case: the ingress of `CERTIFIED`, the egress handing one frame in each of
    `release`'s buckets, which an egress allocation of 3 releases at once."""
    ingress, _ = ingress_records([0, 0], FRAMES, 0)
    handed = [0] * (WINDOWS * T)
    for b in release:
        handed[b] += 1
    egress = egress_records(handed)
    received = tuple(sum(row.count(b"q") for row in r.grid) for r in ingress)
    assert received == (sum(FRAMES), 0, 0, 0) and counts(ingress)[0][:2] == [2, 2]
    return Session(AD.LinkAdvice(_advice(CERTIFIED, release), received), egress, ingress, [0, 0], [0, 1], [], handed,
                   (), CERTIFIED)


def _self_built(link):
    """The verdict a prover builds from its own advice alone: the link, the verifier's registrations (public), the
    digest, which the delivered advice fixes, and every ingress window accepted. No proof, key or session goes in."""
    regs = StandInTie().registrations(LINK)
    return AD.IoTieVerdict(LINK, regs, AD.digest(link), (True,) * len(link.received))


def test_a_verdict_the_prover_builds_never_reaches_the_verifier_of_record(tmp_path):
    """Was LIVE. The honest advice (7 then 2) releases the first output before its request's last frame is in, and
    window 0 refuses it. The swapped split (2 then 7) lowers the first request's end from 3 frames to 1, which the
    certified ingress had released by bucket 0, and under a verdict the prover built (`_self_built`) its ends are
    (1, 4). That verdict has no way in: the advice takes none, and the verifier's tie source, on the certified
    prompts, refuses the split in every window, as no source at all does; the network verdict refuses the link."""
    session = _session()
    honest = session.link
    forged = AD.LinkAdvice(_advice(CERTIFIED[::-1]), honest.received)
    assert ADV.total_bits(honest.advice) == ADV.total_bits(forged.advice)
    assert AD.ends(honest, FRAMING, **held(honest, CERTIFIED)) == (3, 4)
    assert AD.ends(forged, FRAMING, **(held(forged) | {"tie": _self_built(forged)})) == (1, 4)
    with pytest.raises(TypeError):
        AD.LinkAdvice(forged.advice, forged.received, _self_built(forged))
    (tmp_path / "honest").mkdir()
    assert Run(tmp_path / "honest", ties=StandInTie(CERTIFIED)).link(session, honest) == \
        [False] + [True] * (WINDOWS - 1)
    for name, ties in (("certified", StandInTie(CERTIFIED)), ("none", None)):
        (tmp_path / name).mkdir()
        run = Run(tmp_path / name, ties=ties)
        assert run.link(session, forged) == [False] * WINDOWS
        assert not run.lw.network_verdict(run.stream)["accepted"]


def test_the_verdict_names_its_link_and_ingress_registrations_and_the_advice_carries_none():
    """The verdict's fields are the link, its ingress windows' registrations, the advice digest and the accepted bits,
    and `LinkAdvice` is the advice and `received` alone: a stated tuple, a dict or a verdict beside them can't be
    built."""
    assert [f.name for f in fields(AD.IoTieVerdict)] == ["link", "registrations", "advice", "accepted"]
    assert [f.name for f in fields(AD.LinkAdvice)] == ["advice", "received"]
    link = AD.LinkAdvice(_advice(CERTIFIED[::-1]), (4, 0, 0, 0))
    for stated in (CERTIFIED[::-1], {"prompt_len": CERTIFIED[::-1]}, _self_built(link)):
        with pytest.raises(TypeError):
            AD.LinkAdvice(link.advice, link.received, stated)


def test_without_a_tie_source_the_verifier_of_record_refuses_every_window(tmp_path):
    """PROTOCOL.md's fail closed: the honest advice of a causal release is refused in every window without a tie
    source, as in the package, and accepted in every one under the source on the certified ingress."""
    later = _session((2, 3))
    (tmp_path / "none").mkdir()
    assert Run(tmp_path / "none", ties=None).link(later) == [False] * WINDOWS
    (tmp_path / "source").mkdir()
    assert Run(tmp_path / "source", ties=StandInTie(CERTIFIED)).link(later) == [True] * WINDOWS


@pytest.mark.parametrize("why", ["digest", "partial", "length"])
def test_ends_refuses_another_digest_and_every_partial_verdict(why):
    """`ends` refuses a verdict whose digest is another advice's (another release, another `received`, the honest
    advice's on the forged one), one with any window refused or every one, and one covering fewer or more windows
    than `received`."""
    link = AD.LinkAdvice(_advice(CERTIFIED[::-1]), (4, 0, 0, 0))
    tie = held(link)
    d, regs = AD.digest(link), tie["registrations"]
    if why == "digest":
        others = [AD.LinkAdvice(replace(link.advice, release=(1, 4)), link.received),
                  AD.LinkAdvice(link.advice, (3, 1, 0, 0)), AD.LinkAdvice(_advice(CERTIFIED), link.received)]
        bad = [AD.IoTieVerdict(LINK, regs, AD.digest(o), (True,) * WINDOWS) for o in others]
        match = "another advice"
    elif why == "partial":
        bad = [AD.IoTieVerdict(LINK, regs, d, tuple(i != k for i in range(WINDOWS))) for k in range(WINDOWS)]
        bad.append(AD.IoTieVerdict(LINK, regs, d, (False,) * WINDOWS))
        match = "doesn't accept"
    else:
        for n in (0, 1, WINDOWS - 1, WINDOWS + 1):
            source = StandInTie(None, LINK, n)
            with pytest.raises(ValueError, match="doesn't accept"):
                AD.ends(link, FRAMING, tie=AD.IoTieVerdict(LINK, source.registrations(LINK), d, (True,) * n),
                        link_name=LINK, registrations=source.registrations(LINK))
        bad = [AD.IoTieVerdict(LINK, regs[:n], d, (True,) * n) for n in (0, 1, WINDOWS - 1)]
        match = "other ingress registrations"
    for verdict in bad:
        with pytest.raises(ValueError, match=match):
            AD.ends(link, FRAMING, **(tie | {"tie": verdict}))
    assert AD.ends(link, FRAMING, **tie) == (1, 4)


def test_the_stand_in_is_unreachable_from_the_package_and_only_the_tie_service_builds_a_verdict():
    """No module of the certifier outside its tests imports `advised_traffic` or `standin`, none but the tie service
    (`tie_service.IngressTie.tie`, from the stream's outcomes of record) calls `IoTieVerdict(…)`, and the verifier
    takes the verdict only from its own tie source: `_view` reads `self.ties`, which is None unless the verifier sets
    it."""
    pkg = Path(SVC.__file__).parent
    built, imported = [], []
    for path in sorted(pkg.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                f = node.func
                if (isinstance(f, ast.Name) and f.id == "IoTieVerdict") or \
                        (isinstance(f, ast.Attribute) and f.attr == "IoTieVerdict"):
                    built.append(path.name)
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] + [getattr(node, "module", None) or ""]
                if any(n.split(".")[-1] in ("advised_traffic", "standin") for n in names):
                    imported.append(path.name)
    assert built == ["tie_service.py"] and imported == []
    source = inspect.getsource(LinkWindows._view)
    assert "self.ties.verdict(" in source and "advice.tie" not in source
    assert {f.name: f.default for f in fields(LinkWindows)}["ties"] is None


def test_a_verdict_on_another_link_or_ingress_record_isnt_reused(tmp_path):
    """A verdict that named no link or registration would pass one link's verdict on another's view, so `view` takes
    the link and the registrations the verifier holds, and refuses a verdict naming another link or
    other registrations: the verifier of link A's tie source gives a verdict that link B's view refuses, and a
    verdict on other registrations than the source records (another ingress record of the same link) is refused."""
    assert list(inspect.signature(AD.view).parameters) == ["link", "window", "T", "R", "syncs", "sync_set",
                                                            "request_frames", "tie", "link_name", "registrations",
                                                            "clocks"]
    link = AD.LinkAdvice(_advice(CERTIFIED), (4, 0, 0, 0))
    key = signers.signer(tmp_path).public_key()
    views = {}
    for name in (LINK, "another-link"):
        lw = LinkWindows(name, P, SYNC_SET, key, "r-phase10", signers.SESSION, ingress=PIN, proof_deadline=60.0,
                         advised=AD.AdvisedProfile(WINDOWS, FRAMING, 0), ties=StandInTie(CERTIFIED),
                         clocks=StandInClocks())
        views[name] = [lw._view(w, SYNCS, link) for w in range(WINDOWS)]
    assert None not in views[LINK] and views["another-link"] == [None] * WINDOWS
    tie = held(link, CERTIFIED)
    elsewhere = StandInTie(CERTIFIED, "another-link").registrations(LINK)
    for bad, why in ((replace(tie["tie"], link="another-link"), "another link"),
                     (replace(tie["tie"], registrations=elsewhere), "other ingress registrations")):
        for w in range(WINDOWS):
            with pytest.raises(ValueError, match=why):
                AD.view(link, w, T, P.allocation, SYNCS, SYNC_SET, FRAMING, clocks=SYNCS, **(tie | {"tie": bad}))
