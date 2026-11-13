"""The hidden inert-work bound (`circuit.cap`'s `Pc8TileCap` at `cap.row_rho`'s ρr, the template `cap.served` serves)
and the clear path's (`clear_work.cap_failure`) on the same tile of the H100 record: one verdict on every tile, accepted
and refused, under -h3's bound on a row (4ρ at its census k = 256, ρ at k = 1,024) and -h0's bound on the tile alone.
The clear path's forming term (`pearl_c_debit.forming_elements`) reads each row's forming, here α = β = 1 and x = 0,
which has no salt-dead element, and the tiles have no atom block that every word of a row flags, so it is 0, as the
circuit's is on no salt-dead bits."""
from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "verity" / "protocols" / "pouw" / "clear"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "verity" / "protocols" / "pouw" / "verifier" / "tests"))
import clear_work as CW
import test_circuit_cap as T
from verity.protocols.pouw.prover.circuit import cap
from verity.circuits.evaluation import evaluate
from verity.protocols.pouw.verifier.protocol import Matmul, Shape, Tile
from verity.protocols.pouw.prover.schemes import pearl_c as C
from verity.protocols.pouw.prover.schemes.pearl_c_device import DEVICES
from verity.protocols.pouw.prover.schemes.pearl_c_work import UnitRecord

H100 = DEVICES["h100"]
H3, H0 = C.PearlC(forming="v1", device=H100, hashing="h3"), C.PearlC(forming="v1", device=H100, hashing="h0")
#: GLM-5.3's down projection's n, in tiles of 8 rows and 16 columns: a flagged atom-word is 40 units of inert work, a
#: cell earns 420 at k = 256 and 1,392 at k = 1,024, so a row's bound at k = 256 is 16.8 at ρ = 1/400 and 67.2 at 4ρ,
#: and 55.68 at ρ at k = 1,024; the tile's is 134.4 at k = 256
N, TILE = 6144, (8, 16)
ONE = 0x3F800000


def _verdicts(scheme, k: int, at, voluntary=()) -> tuple[bool, bool, str | None]:
    """(the hidden verdict, the clear verdict, the clear reason) on the first unflagged random tile on which `T.flag_words`
    flags the words of ``at`` and no other (zeroing an atom can flag a later one of its word), the hidden bound's
    credited rows the record's (`UnitRecord.credited`: a ``voluntary`` row's band of 64 has none, so the tile has
    none)."""
    tm, tn = TILE
    d = scheme.device
    want = [0] * tm
    for i, _j, _t in at:
        want[i] += 1
    for s in itertools.count():
        ac, bc = T._unflagged_tile(d.name, f"cap-agree-{k}-{at}-{s}", k, tm, tn)
        T.flag_words(ac, bc, at)
        if T._row_flags(d.name, ac, bc) == want:
            break
    record = UnitRecord(Matmul(0, Shape(tm, k, N), 0), tm, k, voluntary_a=tuple(voluntary))
    tile = Tile(tuple(range(tm)), tuple(range(tn)))
    credited = set(record.credited(tile)[0])
    ok = [i in credited for i in range(tm)]
    pa, pb = T._peel_rows(k, TILE)
    cert = [cap.certificates(ac[i], bc[j], d.name) for i in range(tm) for j in range(tn)]
    flat = lambda rows: [x for r in rows for x in r]  # noqa: E731
    hidden = evaluate(cap.templates(k, d.name, TILE, N, row_rho=cap.row_rho(scheme, k))["tile"], flat(ac), flat(bc),
                      flat(pa), flat(pb), [int(o) for o in ok], [0] * (tm * k // 32), flat(cert))[-1]
    reason = CW.cap_failure(scheme, record, tile, SimpleNamespace(
        a=SimpleNamespace(codes=ac, alpha=[ONE] * tm, x=[[0] * k] * tm, beta=[ONE] * tm), b=SimpleNamespace(codes=bc)))
    return bool(hidden), reason is None, reason


@pytest.mark.slow
@pytest.mark.parametrize("scheme, k, at, voluntary, accepted, why", [
    (H3, 256, [], (), True, None),
    (H3, 256, [(2, 5, 1)], (), True, None),
    (H3, 256, [(5, 3, 2), (5, 9, 4)], (), False, "A row 5's replayed debit 80 exceeds its cap 67 (the row cap: 4ρ"),
    (H3, 256, [(5, 3, 2), (5, 9, 4)], (5,), True, None),
    (H3, 256, [(2, 5, 1), (3, 3, 2), (5, 9, 4)], (), True, None),
    (H3, 256, [(i, 2 * i, i + 1) for i in range(4)], (), False, "the replayed debit 160 exceeds the cap 134"),
    (H3, 1024, [(2, 5, 1)], (), True, None),
    (H3, 1024, [(5, 3, 2), (5, 9, 4)], (), False, "A row 5's replayed debit 80 exceeds its cap 56 (the row cap: ρ"),
    (H0, 1024, [(5, 3, 2), (5, 9, 4)], (), True, None),
], ids=["h3-256-none", "h3-256-one", "h3-256-two-over-4rho", "h3-256-two-voluntary", "h3-256-three-rows",
        "h3-256-tile-over-rho", "h3-1024-one", "h3-1024-two-over-rho", "h0-1024-two"])
def test_the_hidden_and_clear_caps_agree(scheme, k, at, voluntary, accepted, why):
    """Each case's verdict is the hidden bound's and the clear path's, and the clear reason names the bound that refuses
    it: a row over 4ρ at k = 256 (or over ρ at k = 1,024) under -h3 is refused by both and accepted once given up as
    voluntary, a row within 4ρ but over ρ is accepted by both, a tile over ρ is refused by both, and -h0 bounds the tile
    alone."""
    hidden, clear, reason = _verdicts(scheme, k, at, voluntary)
    assert hidden == clear == accepted, reason
    assert reason is None if why is None else reason.startswith(why), reason


@pytest.mark.slow
def test_the_hidden_and_clear_caps_agree_on_random_flags():
    """Random flag placements at k = 256 under -h3 (up to three flags in a row, up to five in the tile, each in an atom
    block of its own so that `T.flag_words` flags no other word, a random row voluntary): the hidden and clear verdicts
    agree on every tile, and both verdicts occur."""
    rng, seen = random.Random("cap-agree-random"), set()
    tm, tn = TILE
    for trial in range(8):
        at = []
        for t in rng.sample(range(256 // 32), rng.randint(1, 5)):
            i = rng.randrange(tm)
            if sum(1 for a in at if a[0] == i) < 3:
                at.append((i, rng.randrange(tn), t))
        voluntary = (rng.randrange(tm),) if trial % 2 else ()
        hidden, clear, reason = _verdicts(H3, 256, at, voluntary)
        assert hidden == clear, (at, voluntary, reason)
        seen.add(clear)
    assert seen == {True, False}
