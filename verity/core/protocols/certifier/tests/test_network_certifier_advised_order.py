"""A runner's Commit order other than the certified arrival order, against N2's advised counts (`advised`) under the tie
(IoTieRequest's verdict, `advised.IoTieVerdict`). network-e2e's closer on a served run (r20261024-150521-0551) had its
endpoint's Commit windows number two pairs of requests opposite to their arrival: prompts of 239 and 230 tokens, one
frame each at B = 4,096, so `advised.ends` doesn't move, and of 377 and 142, two frames and one. v0 numbers requests in
order of first row and advice request j is the link's j-th arrival, so the swapped prompt lengths are the tie's to
refuse: the stand-in on the prompt lengths refuses every window, and IoTieRequest's reference on the certified cells
the windows a swapped pair ends in, which `advised.view` reads as a refusal of every window. The verifier of record's
view (`LinkWindows._view`) is None in each, never an exception. With equal prompt lengths the tie has one reading, and
the advised counts refuse a swap only where it moves a floor past the ingress; which body each output answers is the
egress tie's (`test_network_certifier_io_tie.test_a_request_admitted_ahead_of_an_earlier_arrival_is_refused`)."""

import hashlib
from dataclasses import dataclass
from fractions import Fraction

import pytest

from verity.core.protocols.certifier import advice as I1, advised as AD, io_request as Q
from verity.core.protocols.certifier.active import id_request, id_request_frames
from verity.core.protocols.certifier.grid import GridParams
from verity.core.protocols.certifier.schedule import ClockSync, SyncSet
from verity.core.protocols.certifier.service import LinkWindows

import signers
from advised_traffic import LINK, StandInClocks, StandInTie, held, prompts

B, T, R, RIN = 4096, 4, 32, 9
RF = tuple(id_request_frames(n, B) for n in range(1, 400))
SYNC = ClockSync(0, Fraction(1))
SS = SyncSet((0,), (Fraction(1),))
#: the certified arrival, each request's (id, prompt_len, the bucket its one output is released in), and each window's
#: request frames by bucket: 239 and 230 tokens end in buckets 0 and 1, 100 in bucket 4, 377 in buckets 8 and 9 and
#: 142 in bucket 10
ARRIVAL = (("a", 239, 1), ("b", 230, 2), ("m", 100, 5), ("c", 377, 10), ("d", 142, 11))
INGRESS = ([1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 1, 0])
#: the runner's Commit order, each pair admitted later arrival first
COMMIT = tuple(ARRIVAL[i] for i in (1, 0, 2, 4, 3))


def _link(order, received):
    """v0 of the requests `order` ((id, prompt_len, release bucket), in first-row order), each prefilled whole in its
    own step, with one kept output."""
    reqs = tuple(I1.RequestAdvice(i, p, 1, "length") for i, p, _ in order)
    steps = tuple(I1.StepAdvice((I1.Row(j, "prefill", 0, p),)) for j, (_, p, _) in enumerate(order))
    return AD.LinkAdvice(I1.Advice(reqs, steps, tuple(b for *_, b in order)), received)


@dataclass(frozen=True)
class CellsTie:
    """A tie source (`advised.TieSource`) whose verdict is IoTieRequest's reference in the clear on the certified
    cells, where `tie_service.Ties` takes it from proofs: the warden's cells of id requests of the certified prompt
    lengths, in arrival order, released as the ingress counts say, each window registered on `LINK`; each window's
    verdict the reference on the statement the verifier computes from the advice (`io_request.statements`), every
    window refused when the statements don't chain."""

    shape: Q.Shape
    registered: tuple[Q.Registration, ...]
    opened: tuple

    @classmethod
    def of(cls, prompt_lens, counts):
        shape = Q.Shape(B, T, RIN)
        reqs = [id_request(model="m", user=f"u{i}", prompt=[(7919 * i + 104729 * k) % 151329 for k in range(n)],
                           max_tokens=1, frame_bytes=B) for i, n in enumerate(prompt_lens)]
        none = Q.RequestView(Q.Carry(0, 0, 0), (), 0, Q.Carry(0, 0, 0))
        insts = [Q.instance(shape, cells, none, w) for w, cells in enumerate(Q.windows_of(shape, reqs, counts))]
        regs = tuple(Q.Registration(LINK, hashlib.sha512(f"{LINK}/ingress/{w}".encode() + i.root).digest(), i.root,
                                    i.domain) for w, i in enumerate(insts))
        return cls(shape, regs, tuple((i.cells, i.salts) for i in insts))

    def registrations(self, link):
        return tuple(r.digest.hex() for r in self.registered)

    def windows(self, link):
        try:
            stmts = Q.statements(self.shape, prompts(link), link.received, self.registered)
        except ValueError:
            stmts = ()
        if not stmts or not Q.chained(stmts):
            return (False,) * len(self.registered)
        return tuple(bool(Q.reference(self.shape, Q.Instance(c, s, st.root, st.domain, st.view, st.window)))
                     for st, (c, s) in zip(stmts, self.opened, strict=True))

    def verdict(self, link_name, link, profile=None):
        return AD.IoTieVerdict(link_name, self.registrations(link_name), AD.digest(link), self.windows(link))


def _decided(key, ties, link, ingress):
    """Each window's decision by the verifier of record on a link whose tie source is `ties`: None where its view
    (`LinkWindows._view`) is refused, else the advised counts' reference on the window's certified counts."""
    n = len(ingress)
    lw = LinkWindows(LINK, GridParams(T, R, 1), SS, key, "r-order", signers.SESSION, ingress=GridParams(T, RIN, 1),
                     proof_deadline=60.0, advised=AD.AdvisedProfile(n, RF, 0), ties=ties,
                     clocks=StandInClocks((SYNC,) * n))
    egress = [[0] * T for _ in range(n)]
    for b in link.advice.release:
        egress[b // T][b % T] += 1
    out = []
    for w in range(n):
        v = lw._view(w, [SYNC] * n, link)
        out.append(None if v is None else AD.reference(T, RIN, 0, ingress[w], egress[w], v.released, v.floor,
                                                       v.received))
    return out


@pytest.fixture(scope="module")
def key(tmp_path_factory):
    return signers.signer(tmp_path_factory.mktemp("order")).public_key()


def test_a_commit_order_other_than_the_arrival_is_refused_in_every_window(key):
    """The run's two inversions on one link of three windows: the honest advice is accepted in every window under
    either tie source, and the Commit's order refused in every one. IoTieRequest's reference refuses windows 0 and 2,
    where the swapped pairs end, and accepts window 1, whose request is its own; the verdict accepts no window then.
    The refusal is the tie's: under a tie on the Commit's own prompt lengths, what N2 decides without IoTieRequest,
    the floors hold, and the first pair's ends are the honest ones."""
    honest, commit = _link(ARRIVAL, tuple(map(sum, INGRESS))), _link(COMMIT, tuple(map(sum, INGRESS)))
    assert [RF[p - 1] for _, p, _ in ARRIVAL] == [1, 1, 1, 2, 1]
    cells = CellsTie.of([p for _, p, _ in ARRIVAL], [c for w in INGRESS for c in w])
    for ties in (StandInTie(prompts(honest), LINK, len(INGRESS)), cells):
        assert _decided(key, ties, honest, INGRESS) == [True] * len(INGRESS)
        assert _decided(key, ties, commit, INGRESS) == [None] * len(INGRESS)
    assert (cells.windows(honest), cells.windows(commit)) == ((True, True, True), (False, True, False))
    with pytest.raises(ValueError, match="doesn't accept"):
        AD.ends(commit, RF, tie=cells.verdict(LINK, commit), link_name=LINK, registrations=cells.registrations(LINK))
    assert _decided(key, StandInTie(None, LINK, len(INGRESS)), commit, INGRESS) == [True] * len(INGRESS)
    assert AD.ends(commit, RF, **held(commit)) == (1, 2, 3, 4, 6)
    assert AD.ends(honest, RF, **held(honest)) == (1, 2, 3, 5, 6)


@pytest.mark.parametrize("ingress, release, accepted", [
    ([2, 0, 0, 0], (1, 2), True),
    ([1, 0, 1, 0], (2, 3), True),
    ([1, 0, 1, 0], (1, 3), False),
])
def test_equal_prompt_lengths_swapped_are_refused_only_where_a_floor_moves_past_the_ingress(key, ingress, release,
                                                                                            accepted):
    """Two 16-token requests, one frame each, swapped. The tie can't tell them apart. Ending in one bucket, the swap is
    accepted; across buckets, it holds the earlier arrival's output to the later one's end, refused when the output is
    released before that end and accepted after. The egress tie refuses each accepted swap: the map puts advice
    request j's outputs on the j-th arrival's stream."""
    arrival = (("a", 16, release[0]), ("b", 16, release[1]))
    ties = StandInTie((16, 16), LINK, 1)
    assert _decided(key, ties, _link(arrival, (2,)), [ingress]) == [True]
    assert _decided(key, ties, _link(arrival[::-1], (2,)), [ingress]) == [accepted]
