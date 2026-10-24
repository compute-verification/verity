"""REDTEAM phase 16, Target 5 (#3838, N2-ADVICE): IoTieRequest's statement names its link and its registration, and
`negatives.request_tie`'s 4-line change states each window on `io_request.Registration(N2_LINK, …)` with
`n2_registrations`' stand-in digest, over the certified (root, domain). The lead's ask: try a statement bound to
another link's registration. These cases confirm the binding holds at both layers the verdict passes through:
`io_request.statements`/`chained` (one link, each window its own registration) and `advised._held`/`ends`, which holds
the verdict to the verifier's own link and registration record. A confirmation, not a finding."""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
FRAMING = (1, 1, 2, 2, 2)  # the link's request frames by prompt length: r0's prompt of 5 two, r1's of 2 one


def _load(name: str):
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location(f"network_traces_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


NG = _load("negatives")
IO = importlib.import_module(NG.IO_REQUEST_MODULE)
AD = importlib.import_module(NG.ADVISED_MODULE)
ADV = importlib.import_module("verity.core.protocols.certifier.advice")


def _advice():
    """The suite's honest advice (`test_negatives._advice`): r0 a prompt of 5, 3 out; r1 a prompt of 2, 2 out."""
    rows = [[(0, "prefill", 0, 3)], [(0, "prefill", 3, 2), (1, "prefill", 0, 2)],
            [(0, "decode", 5, 1), (1, "decode", 2, 1)], [(0, "decode", 6, 1)]]
    return ADV.Advice((ADV.RequestAdvice("r0", 5, 3, "eos"), ADV.RequestAdvice("r1", 2, 2, "length")),
                      tuple(ADV.StepAdvice(tuple(ADV.Row(*r) for r in st)) for st in rows), (3, 3, 3, 3, 5))


@pytest.fixture(scope="module")
def two_windows():
    """A link whose r0 (two request frames) fills window 0 and r1 (one) window 1, certified on a fake upstream, its
    honest `(link, cells, statements)` with the verifier's own N2_LINK registrations."""
    from verity.core.protocols.certifier.active import id_request
    B = 404
    reqs = [id_request(model="", user="r0", prompt=[0] * 5, max_tokens=3, frame_bytes=B),
            id_request(model="", user="r1", prompt=[0] * 2, max_tokens=2, frame_bytes=B)]
    cells = NG.certified_ingress(IO, reqs, [2] + [0] * 7 + [1] + [0] * 7, T=8, frame_bytes=B)
    link = AD.LinkAdvice(_advice(), (2, 1))
    prompt_lens = [q.prompt_len for q in _advice().requests]
    regs = [IO.Registration(NG.N2_LINK, bytes.fromhex(d), root, domain)
            for d, (root, domain) in zip(NG.n2_registrations(len(cells.registered)), cells.registered)]
    stmts = IO.statements(cells.shape, prompt_lens, list(link.received), regs)
    return link, cells, prompt_lens, regs, stmts


def test_the_honest_link_chains_and_is_tied_on_every_window(two_windows):
    """The baseline the attacks must not reach: the verifier's own N2_LINK statements chain and IoTieRequest's
    reference accepts every window."""
    link, cells, _, _, stmts = two_windows
    assert IO.chained(stmts)
    assert NG.request_tie(IO, cells, link) == (True, True)


def test_statements_refuse_a_window_registered_under_another_link(two_windows):
    """A statement bound to another link's registration: window 1's registration names `attacker`, not N2_LINK.
    `statements` takes one link's registrations, so it refuses the mix before any view is built."""
    _, cells, prompt_lens, regs, _ = two_windows
    mixed = [regs[0], regs[1]._replace(link="attacker")]
    with pytest.raises(ValueError, match="one link's registrations"):
        IO.statements(cells.shape, prompt_lens, [2, 1], mixed)


def test_statements_refuse_two_windows_sharing_a_registration(two_windows):
    """One registration cannot stand in for two windows: window 1 reuses window 0's digest. `statements` requires each
    window its own, so it refuses."""
    _, cells, prompt_lens, regs, _ = two_windows
    dup = [regs[0], regs[1]._replace(digest=regs[0].digest)]
    with pytest.raises(ValueError, match="one link's registrations"):
        IO.statements(cells.shape, prompt_lens, [2, 1], dup)


def test_a_statement_bound_to_another_links_registration_breaks_the_chain(two_windows):
    """Past `statements`, a forged statement whose link and registration are another link's does not chain: `chained`
    holds every statement to the first's link, so window 1 on `attacker` with a foreign digest is refused."""
    _, _, _, _, stmts = two_windows
    forged = stmts[0], stmts[1]._replace(link="attacker", registration=hashlib.sha512(b"other-link").digest())
    assert not IO.chained(forged)


def test_reusing_a_windows_registration_for_the_next_breaks_the_chain(two_windows):
    """`chained` holds the statements to distinct registrations, so replaying window 0's registration as window 1's is
    refused even with window 1's own link and view."""
    _, _, _, _, stmts = two_windows
    replayed = stmts[0], stmts[1]._replace(registration=stmts[0].registration)
    assert not IO.chained(replayed)


def test_a_verdict_on_another_link_is_not_this_links():
    """At the advised layer, the tie `n2_advice` carries is held to the verifier's own link. A verdict that names
    another link (`advised._held` via `ends`) is refused, so a verdict proved on another link isn't reused here."""
    a = _advice()
    link, honest = NG.n2_advice(a, FRAMING, [3])
    assert AD.ends(link, FRAMING, tie=honest, link_name=NG.N2_LINK, registrations=NG.n2_registrations(1)) == (2, 3)
    bad = AD.IoTieVerdict("attacker-link", NG.n2_registrations(1), AD.digest(link), (True,))
    with pytest.raises(ValueError, match="on another link"):
        AD.ends(link, FRAMING, tie=bad, link_name=NG.N2_LINK, registrations=NG.n2_registrations(1))


def test_a_verdict_on_another_links_registrations_is_not_this_links():
    """A verdict on N2_LINK but on another link's stand-in registration digests is refused: `_held` holds the verdict
    to the verifier's own record of the link's ingress registrations."""
    a = _advice()
    link, _ = NG.n2_advice(a, FRAMING, [3])
    other = (hashlib.sha512(b"negatives ingress registration other-link/0").hexdigest(),)
    bad = AD.IoTieVerdict(NG.N2_LINK, other, AD.digest(link), (True,))
    with pytest.raises(ValueError, match="other ingress registrations"):
        AD.ends(link, FRAMING, tie=bad, link_name=NG.N2_LINK, registrations=NG.n2_registrations(1))
