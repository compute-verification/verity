"""PoUS and the wipe as resource schemes (`verity.protocols.accounting.memory.pous.scheme`,
`verity.protocols.accounting.memory.wipe.scheme`): found by name, at their verifiers' own ids, with the codec the
adapter attaches, and citing only Lean names the locks record."""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from verity.protocols.run import schemes as table
from verity.protocols.accounting import scheme as S
from verity.protocols.accounting.memory.pous.scheme import POUS, POUS_TERM
from verity.protocols.accounting.memory.wipe.scheme import WIPE
from verity.protocols.accounting.memory.pous.verifier import period, schemes
from verity.protocols.accounting.memory.wipe.verifier import wipe_period

ROOT = Path(__file__).resolve().parents[6]
LOCKS = (ROOT / "security" / "lean-audit.json", ROOT / "security" / "Properties" / "lean-audit.json")


def test_found_by_name_at_their_verifiers_ids() -> None:
    assert table.get("pous") is POUS and table.get("wipe") is WIPE
    assert (POUS.protocol, WIPE.protocol) == (period.PROTOCOL, wipe_period.PROTOCOL), (
        "the trusted verifiers keep their own ids (they may not import the registry): change period.PROTOCOL or "
        "wipe_period.PROTOCOL and its declaration's protocol (memory/pous/scheme.py or memory/pous/wipe_scheme.py, through domains) together")
    assert POUS.unit == WIPE.unit == "bits"


def test_pous_attaches_its_registered_codec_and_the_wipe_none() -> None:
    name, _, codec = POUS.option.partition(":")
    assert name == POUS.name and codec in schemes.REGISTRY
    assert WIPE.option == "" and WIPE.hooks == ()


def test_every_lean_name_cited_is_in_a_lock() -> None:
    locks = "".join(p.read_text() for p in LOCKS)
    cited = {*POUS.assumptions, *WIPE.assumptions, POUS_TERM,
             *(c.bound for c in (*POUS.charges, *WIPE.charges) if c.bounded)}
    assert sorted(n for n in cited if f'"{n}"' not in locks) == []


def test_pous_takes_its_challenges_setting() -> None:
    assert S.settings(POUS, {"challenges": 105}) == {"challenges": 105}


def test_pous_says_its_own_term_and_is_refused_a_bound_while_its_help_is_unbounded() -> None:
    assert POUS.delta_scheme == Fraction(1, 2**128) and isinstance(WIPE.delta_scheme, S.Lacks)
    assert [c.name for c in POUS.charges if not c.bounded] == ["help"]
    assert all((s.over, s.under) for s in (POUS, WIPE))


@pytest.mark.slow
def test_pous_credits_every_store_block_at_its_bits() -> None:
    """PoUS's blocks are the store's: on the setup's registered C, every unit of `PousP2SetupCheck_v2` is credited, each
    at the bits one store block holds (P2's w, the unit's W), and `bound` still refuses PoUS by its unbounded help. Slow:
    building the setup's Program takes about 15 s, and no smaller circuit is the setup's."""
    from verity.protocols.proof.spot.verifier.one_stage import partition as P
    from verity.protocols.accounting.memory.pous.prover import window as WIN

    setup = WIN.Setup(64, 189, 16, 4, 4)
    strata = P.strata(*setup.circuit().program())
    assert [t.split("{")[0] for t, _ in strata] == ["PousP2SetupCheck_v2"]
    assert S.credited([POUS], strata) == frozenset(range(setup.blocks)) and setup.blocks == 12
    (block,) = POUS.blocks
    assert block.cost({"W": 64, "C0": 189, "N": 16}) == 64 and block.cost({"W": 16448, "C0": 21065}) == 16448
    with pytest.raises(S.Unbounded, match="known charges nothing bounds yet: help") as e:
        S.bound(POUS, Fraction(1, 100), 0, {"chip": 0, "side-channels": 0, "help": 0}, epsilon=Fraction(0))
    assert e.value.terms == ("help",)
