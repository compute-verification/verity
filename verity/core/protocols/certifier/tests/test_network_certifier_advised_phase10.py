"""REDTEAM phase 10, target 1, against #3502's fail-closed tie (`advised.IoTieVerdict`): the verdict is meant to be
built only by the verifier of record from IoTieRequest proofs it verified, so that until IoTieRequest runs in the proof
service every advised link is refused (PROTOCOL.md, "The tie is IoTieRequest's verdict"). `LinkWindows.verify` reads
it off the delivered `LinkAdvice`, though, and the verdict holds only the advice's digest, which the delivered advice
fixes: a prover that delivers its own verdict reopens phase 6's acausal release through the verifier of record. `ends`
does refuse another advice's verdict and every partially accepted one, and the stand-in (`advised_traffic.tied`) is
unreachable from the package; the verdict names no link, so one link's verdict passes on another's view.
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
from advised_traffic import (FRAMING, LINK, P, PIN, SYNC_SET, SYNCS, T, WINDOWS, Session, counts, egress_records,
                             ingress_records, tied)
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


def _advice(prompt_lens):
    """Requests of `prompt_lens` tokens in first-row order, each prefilled whole in its own step, one kept output
    each, released at `RELEASE`."""
    reqs = tuple(ADV.RequestAdvice(f"r{i}", p, 1, "length") for i, p in enumerate(prompt_lens))
    steps = tuple(ADV.StepAdvice((ADV.Row(i, "prefill", 0, p),)) for i, p in enumerate(prompt_lens))
    return ADV.Advice(reqs, steps, RELEASE)


def _session():
    """The certified records of the case: the ingress of `CERTIFIED`, the egress handing one frame in each of
    `RELEASE`'s buckets, which an egress allocation of 3 releases at once."""
    ingress, _ = ingress_records([0, 0], FRAMES, 0)
    handed = [0] * (WINDOWS * T)
    for b in RELEASE:
        handed[b] += 1
    egress = egress_records(handed)
    received = tuple(sum(row.count(b"q") for row in r.grid) for r in ingress)
    assert received == (sum(FRAMES), 0, 0, 0) and counts(ingress)[0][:2] == [2, 2]
    return Session(AD.LinkAdvice(_advice(CERTIFIED), received), egress, ingress, [0, 0], [0, 1], [], handed, ())


def _self_tied(link):
    """The verdict a prover builds from its own advice alone: the digest, which the delivered advice fixes, and every
    ingress window accepted. No proof, key, session or record goes in."""
    return replace(link, tie=AD.IoTieVerdict(AD.digest(link), (True,) * len(link.received)))


def test_a_verdict_the_prover_builds_reopens_phase_6_through_the_verifier_of_record(tmp_path):
    """LIVE. The honest advice (7 then 2) releases the first output before its request's last frame is in, and window
    0 refuses it, so the link is refused. The swapped split (2 then 7) lowers the first request's end from 3 frames to 1, which
    the certified ingress had released by bucket 0: the stand-in verdict on the certified prompts refuses it, but a
    verdict the prover built from its own advice is an `IoTieVerdict` for the same digest, every window accepted, and
    `LinkWindows.verify` takes the delivered `LinkAdvice`'s tie as it is. Every window is accepted on the proof, the
    network verdict accepts the link, and its advice output can't tell this tie from IoTieRequest's."""
    session = _session()
    honest = tied(session.link, CERTIFIED)
    forged = AD.LinkAdvice(_advice(CERTIFIED[::-1]), session.link.received)
    assert ADV.total_bits(honest.advice) == ADV.total_bits(forged.advice)
    assert AD.ends(honest, FRAMING) == (3, 4)
    (tmp_path / "honest").mkdir()
    assert Run(tmp_path / "honest").link(session, honest) == [False] + [True] * (WINDOWS - 1)
    (tmp_path / "standin").mkdir()
    assert Run(tmp_path / "standin").link(session, tied(forged, CERTIFIED)) == [False] * WINDOWS
    stated = _self_tied(forged)
    assert AD.ends(stated, FRAMING) == (1, 4)
    (tmp_path / "stated").mkdir()
    run = Run(tmp_path / "stated")
    assert run.link(session, stated) == [True] * WINDOWS
    verdict = run.lw.network_verdict(run.stream)
    assert verdict["accepted"] and verdict["advice"] == AD.digest(forged) == AD.digest(stated)


def test_the_verdict_holds_nothing_the_delivered_advice_doesnt_fix():
    """The verdict's fields are the advice digest and the accepted bits: `digest` reads the delivered advice and
    `received` alone, and `LinkAdvice` takes any `IoTieVerdict`, so the type check on the tie (phase 7's fix) stops a
    stated tuple or dict but not a stated verdict."""
    assert [f.name for f in fields(AD.IoTieVerdict)] == ["advice", "accepted"]
    assert list(inspect.signature(AD.digest).parameters) == ["link"]
    link = AD.LinkAdvice(_advice(CERTIFIED[::-1]), (4, 0, 0, 0))
    for stated in (CERTIFIED[::-1], {"prompt_len": CERTIFIED[::-1]}):
        with pytest.raises(ADV.MalformedAdvice, match="IoTieVerdict"):
            AD.LinkAdvice(link.advice, link.received, stated)
    assert AD.LinkAdvice(link.advice, link.received, _self_tied(link).tie).tie == _self_tied(link).tie


def test_without_a_delivered_tie_the_verifier_of_record_refuses_every_window(tmp_path):
    """PROTOCOL.md's fail closed holds only for a delivery without a tie: the honest advice of a causal release,
    untied, is refused in every window, and with any `IoTieVerdict` the prover attaches, accepted in every one."""
    session = _session()
    causal = AD.LinkAdvice(replace(session.link.advice, release=(2, 3)), session.link.received)
    handed = [0] * (WINDOWS * T)
    for b in (2, 3):
        handed[b] += 1
    later = Session(causal, egress_records(handed), session.ingress, [0, 0], [0, 1], [], handed, ())
    (tmp_path / "untied").mkdir()
    assert Run(tmp_path / "untied").link(later, causal) == [False] * WINDOWS
    (tmp_path / "tied").mkdir()
    assert Run(tmp_path / "tied").link(later, _self_tied(causal)) == [True] * WINDOWS


@pytest.mark.parametrize("why", ["digest", "partial", "length"])
def test_ends_refuses_another_digest_and_every_partial_verdict(why):
    """CLOSED. `ends` refuses a verdict whose digest is another advice's (another release, another `received`, the
    honest advice's on the forged one), one with any window refused or every one, and one covering fewer or more
    windows than `received`."""
    link = AD.LinkAdvice(_advice(CERTIFIED[::-1]), (4, 0, 0, 0))
    d = AD.digest(link)
    if why == "digest":
        others = [AD.LinkAdvice(replace(link.advice, release=(1, 4)), link.received),
                  AD.LinkAdvice(link.advice, (3, 1, 0, 0)), AD.LinkAdvice(_advice(CERTIFIED), link.received)]
        bad = [AD.IoTieVerdict(AD.digest(o), (True,) * WINDOWS) for o in others]
        match = "another advice"
    elif why == "partial":
        bad = [AD.IoTieVerdict(d, tuple(i != k for i in range(WINDOWS))) for k in range(WINDOWS)]
        bad.append(AD.IoTieVerdict(d, (False,) * WINDOWS))
        match = "doesn't accept"
    else:
        bad = [AD.IoTieVerdict(d, (True,) * n) for n in (0, 1, WINDOWS - 1, WINDOWS + 1)]
        match = "doesn't accept"
    for tie in bad:
        with pytest.raises(ValueError, match=match):
            AD.ends(replace(link, tie=tie), FRAMING)
    assert AD.ends(_self_tied(link), FRAMING) == (1, 4)


def test_the_stand_in_is_unreachable_from_the_package_and_nothing_there_builds_a_verdict():
    """CLOSED in the package: no module of the certifier outside its tests imports `advised_traffic` or `standin`, or
    calls `IoTieVerdict(…)`, and the verifier of record has no other way to a verdict than the delivered advice's
    field (`_view` passes `advice` to `advised.view` as it came)."""
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
    assert built == [] and imported == []
    assert "AD.view(advice," in inspect.getsource(LinkWindows._view).replace("\n", " ").replace("  ", " ")


def test_a_verdict_names_no_link_so_one_links_verdict_passes_on_anothers_view(tmp_path):
    """The verdict names the advice digest only, and `advised.view` takes no link, registration or stream: a verdict
    IoTieRequest gave on link A's certified ingress passes `ends` on link B carrying the same advice bytes and
    `received`, and B's view of it is A's. The verifier that builds it must bind it to the link-window's ingress
    registrations itself."""
    assert list(inspect.signature(AD.view).parameters) == ["link", "window", "T", "R", "syncs", "sync_set",
                                                            "request_frames"]
    assert [f.name for f in fields(AD.LinkAdvice)] == ["advice", "received", "tie"]
    link = tied(AD.LinkAdvice(_advice(CERTIFIED), (4, 0, 0, 0)))
    key = signers.signer(tmp_path).public_key()
    views = []
    for name in (LINK, "another-link"):
        lw = LinkWindows(name, P, SYNC_SET, key, "r-phase10", signers.SESSION, ingress=PIN,
                         proof_deadline=60.0, advised=AD.AdvisedProfile(WINDOWS, FRAMING, 0))
        views.append([lw._view(w, SYNCS, link) for w in range(WINDOWS)])
    assert views[0] == views[1] and None not in views[0]
