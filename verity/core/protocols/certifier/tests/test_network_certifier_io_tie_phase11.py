"""REDTEAM phase 11 against the ingress tie in zero knowledge (#3701, `tie_service.IngressTie`): the lead's three
questions re-run against the verifier of record, each answered closed. The PR's own
`test_network_certifier_tie_service.py` covers phase 6/9, the short, out-of-order and past streams, an outcome on
another advice, in the clear or late, a mis-framing profile and an unpinned size class; these cases exercise the
substitution and forge levers it leaves untested:

- A proof of one window can't stand for another (window 0's delivered proof presented for window 1's draw): the
  verifier derives the public row from the registration's own root, its `domain(w)` and the advice's view of window w,
  so a proof whose public row is window 0's opens to none of window 1's and is refused.
- A proof of one link can't stand for another (the same cells committed to two links; link A's proof presented for
  link B's draw): the served domain binds `<run>/<link>/<window>`, so the public row's domain is another's and refused.
- No path from the delivered bytes to a verdict: a forged output word, or a forged C-Flock/Lean acceptance over cells
  that don't satisfy the advice's view, leaves the proof's committed output row at 0, which opens to 0 and not 1.
- A forged public-file unit index or session program fails the verifier's `proved`, so a Lean-accepted session of the
  wrong unit or Program is refused.
"""

from dataclasses import replace

import pytest

from verity.core.protocols.certifier import advice as ADV, advised as AD, io_request as Q, io_tie as IT, \
    service as SVC, tie_service as TS
from verity.core.protocols.certifier.active import id_request
from verity.core.protocols.sampled_proofs import service as S

import standin

B = 512
SHAPE = Q.Shape(B, 4, 2, 1)
RUN = "r-tie-p11"
#: REDTEAM phase 9's span: request 1 (40 tokens) takes two frames, one in each window, behind each window's own frame
SPAN, COUNTS = (5, 40, 5), (1, 0, 0, 1, 1, 1, 0, 0)
FRAMING = TS.framing(B, 64)
PROFILE = AD.AdvisedProfile(2, FRAMING, 1)
OWN_VIEW = object()


@pytest.fixture(autouse=True)
def lean(monkeypatch):
    monkeypatch.setattr(SVC.PF, "verify", standin.lean)


def _requests(prompt_lens):
    return [id_request(model="glm", user=f"u{i}", prompt=[(7919 * i + 104729 * k) % 151329 for k in range(n)],
                       max_tokens=16, temperature=0.7, top_p=0.95, seed=i, frame_bytes=B)
            for i, n in enumerate(prompt_lens)]


def _received(counts, shape=SHAPE):
    return tuple(sum(counts[w * shape.T:(w + 1) * shape.T]) for w in range(len(counts) // shape.T))


def _advice(prompt_lens, received, release=None):
    requests = tuple(ADV.RequestAdvice(f"u{j}", n, 1, "eos") for j, n in enumerate(prompt_lens))
    steps = (ADV.StepAdvice(tuple(ADV.Row(j, "prefill", 0, n) for j, n in enumerate(prompt_lens))),)
    release = tuple(range(len(prompt_lens))) if release is None else tuple(release)
    return AD.LinkAdvice(ADV.Advice(requests, steps, release), tuple(received))


HONEST = Q.windows_of(SHAPE, _requests(SPAN), COUNTS)
RECEIVED = _received(COUNTS)


class Run:
    """One ingress link's tie through the proof service, as the PR's harness: the warden commits each window, the
    challenger registers and draws it, and the verifier of record decides on a stand-in proof."""

    def __init__(self, tmp_path, link="requests", shape=SHAPE) -> None:
        self.tmp, self.now, self.previous = tmp_path, [0.0], None
        self.tie = TS.IngressTie(link, shape, RUN, proof_deadline=60.0)
        self.stream = self.tie.stream(budget=16, delta=2.0 ** -40, challenger="challenger-test",
                                      coins=lambda n: b"\x07" * n, clock=lambda: self.now[0])

    def draw(self, w, cells):
        c = self.tie.commit(w, cells, [IT.fake_salt(b"warden" + bytes([w]), t) for t in range(self.tie.shape.T)])
        reg = self.tie.register(self.stream, self.tie.registration(c, self.previous))
        self.previous = reg.digest
        return c, self.stream.select(reg)

    def prove(self, w, cells, advice, view=OWN_VIEW, **kw):
        c, draw = self.draw(w, cells)
        view = self.tie.view(advice, w) if view is OWN_VIEW else view
        delivered = standin.prove_tie(self.tie, c, draw, view, self.tmp / f"tie-{draw.registration.order}", **kw)
        return c, draw, delivered

    def deliver(self, draw, delivered):
        return self.stream.deliver(draw, delivered.sessions())

    def verdict(self, draw, delivered, advice, opening=None):
        return self.tie.verify(draw, delivered, circuit_sha512=standin.circuit_sha512(delivered), advice=advice,
                               opening=opening)

    def record(self, draw, delivered, advice, opening=None) -> S.Outcome:
        """Deliver an already-proved window and record its outcome of record, so the next window may register."""
        late = self.deliver(draw, delivered)
        if late is not None:
            return late
        return self.tie.outcome(self.stream, draw, delivered, circuit_sha512=standin.circuit_sha512(delivered),
                                advice=advice, opening=opening)

    def round(self, w, cells, advice, view=OWN_VIEW, **kw) -> S.Outcome:
        c, draw, delivered = self.prove(w, cells, advice, view, **kw)
        return self.record(draw, delivered, advice, c)

    def link(self, windows, advice):
        return [self.round(w, cells, advice).accepted for w, cells in enumerate(windows)]


def test_an_honest_link_is_tied_and_advised_ends_reads_it(tmp_path):
    run, advice = Run(tmp_path), _advice(SPAN, RECEIVED)
    assert run.link(HONEST, advice) == [True, True]
    verdict = run.tie.tie(run.stream, advice, PROFILE)
    assert verdict == AD.IoTieVerdict(AD.digest(advice), (True, True))
    assert AD.ends(replace(advice, tie=verdict), PROFILE.request_frames) == Q.ends(SPAN, B)


def test_window_0s_proof_cannot_stand_for_window_1(tmp_path):
    """A proof of one window for another: the verifier derives the public row from the registration's own root, its
    `domain(w)` and the advice's view of window w, so window 0's delivered proof, presented for window 1's draw, opens
    to none of window 1's public row and is refused — with a forged Lean acceptance, which decides nothing on its own."""
    run, advice = Run(tmp_path), _advice(SPAN, RECEIVED)
    c0, draw0, delivered0 = run.prove(0, HONEST[0], advice)
    assert run.record(draw0, delivered0, advice, c0).accepted  # records window 0's outcome, so window 1 may register
    _c1, draw1, _delivered1 = run.prove(1, HONEST[1], advice)
    # window 1's own proof is honest; here window 0's delivered proof is presented against window 1's draw
    assert delivered0.output == 1
    cross = run.tie.verify(draw1, delivered0, circuit_sha512=standin.circuit_sha512(delivered0), advice=advice)
    assert not cross["accepted"], "window 0's public row is not window 1's; the window binds the proof"


def test_a_proof_of_one_link_cannot_stand_for_another(tmp_path):
    """A proof of one link for another: the served domain binds `<run>/<link>/<window>`, so even the same cells
    committed to two links register under different roots and domains, and link A's window-0 proof, presented for link
    B's window-0 draw, opens to another domain's public row and is refused (the unit's Program is the shape's, shared)."""
    advice = _advice(SPAN, RECEIVED)
    a, b = Run(tmp_path / "a", link="link-a"), Run(tmp_path / "b", link="link-b")
    assert a.tie.program == b.tie.program, "the unit's Program is the shape's, so a cross-link reuse isn't a program gap"
    _ca, drawa, delivered_a = a.prove(0, HONEST[0], advice)
    assert a.deliver(drawa, delivered_a) is None and a.verdict(drawa, delivered_a, advice)["accepted"]
    _cb, drawb, _delivered_b = b.prove(0, HONEST[0], advice)
    cross = b.tie.verify(drawb, delivered_a, circuit_sha512=standin.circuit_sha512(delivered_a), advice=advice)
    assert not cross["accepted"], "link A's domain is not link B's; the domain binds the link"


def test_a_forged_output_word_opens_to_the_committed_zero_not_one(tmp_path):
    """No path from the delivered bytes to a verdict: on cells that don't satisfy the advice's view (all pads, no
    request frames) the proof's committed output row is 0, so a forged output word of 1, with a forged Lean acceptance,
    opens to 0 and not 1 and is refused."""
    run, advice = Run(tmp_path), _advice(SPAN, RECEIVED)
    assert run.round(0, HONEST[0], advice).accepted
    pads = [IT.pad_cell(B)] * SHAPE.N
    c, draw = run.draw(1, pads)
    view = run.tie.view(advice, 1)
    assert run.tie.outputs(c, bytes.fromhex(draw.registration.record["roots"][0]["root"]), view) == 0
    delivered = standin.prove_tie(run.tie, c, draw, view, tmp_path / "forged", word=1)
    assert delivered.output == 1 and run.deliver(draw, delivered) is None
    assert not run.verdict(draw, delivered, advice)["accepted"], "the output row commits to 0; a forged word can't open it to 1"


def test_a_forged_lean_acceptance_over_mismatched_cells_is_refused(tmp_path):
    """A forged C-Flock/Lean acceptance over cells that don't satisfy the advice's view decides nothing: the committed
    output row is 0, so the honest output word opens to 0 and the window is refused even though every session accepted."""
    run, advice = Run(tmp_path), _advice(SPAN, RECEIVED)
    assert run.round(0, HONEST[0], advice).accepted
    pads = [IT.pad_cell(B)] * SHAPE.N
    c, draw = run.draw(1, pads)
    view = run.tie.view(advice, 1)
    delivered = standin.prove_tie(run.tie, c, draw, view, tmp_path / "liar", accepted=True)
    assert delivered.output == 0 and run.deliver(draw, delivered) is None
    verdict = run.verdict(draw, delivered, advice)
    assert not verdict["accepted"], "Lean acceptance alone, with a committed 0 output, is refused"


def test_a_forged_unit_index_or_program_is_refused(tmp_path):
    """A Lean-accepted session of the wrong unit or Program fails the verifier's `proved`: a public file whose unit
    index isn't the drawn unit opens no commitment for it, and a session naming another Program is not this tie's."""
    advice = _advice(SPAN, RECEIVED)

    run = Run(tmp_path / "units")
    assert run.round(0, HONEST[0], advice).accepted
    c, draw = run.draw(1, HONEST[1])
    view = run.tie.view(advice, 1)
    bad = [u + 7 for u in draw.units]
    delivered = standin.prove_tie(run.tie, c, draw, view, tmp_path / "units-proof", units=bad)
    assert run.deliver(draw, delivered) is None
    assert not run.verdict(draw, delivered, advice)["accepted"], "the public file names another unit than the drawn one"

    run = Run(tmp_path / "program")
    assert run.round(0, HONEST[0], advice).accepted
    c, draw = run.draw(1, HONEST[1])
    view = run.tie.view(advice, 1)
    delivered = standin.prove_tie(run.tie, c, draw, view, tmp_path / "program-proof", program="another-program")
    assert run.deliver(draw, delivered) is None
    assert not run.verdict(draw, delivered, advice)["accepted"], "the session names another Program than the tie's"
