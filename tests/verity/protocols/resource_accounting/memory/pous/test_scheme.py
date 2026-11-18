"""PoUS and the wipe as resource schemes (`verity.protocols.resource_accounting.memory.pous.scheme`): found by name, at
their verifiers' own ids, with the codec the adapter attaches, and citing only Lean names the locks record."""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

from verity.protocols.resource_accounting import scheme as S
from verity.protocols.resource_accounting.memory.pous.scheme import POUS, POUS_TERM, WIPE
from verity.protocols.resource_accounting.memory.pous.verifier import period, schemes, wipe_period

ROOT = Path(__file__).resolve().parents[6]
LOCKS = (ROOT / "security" / "lean-audit.json", ROOT / "security" / "Properties" / "lean-audit.json")


def test_found_by_name_at_their_verifiers_ids() -> None:
    assert S.get("pous") is POUS and S.get("wipe") is WIPE
    assert (POUS.protocol, WIPE.protocol) == (period.PROTOCOL, wipe_period.PROTOCOL)
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
    assert S.settings(POUS, {"challenges": 105})["challenges"] == 105


def test_pous_says_its_own_term_and_is_refused_a_bound_while_its_help_is_unbounded() -> None:
    assert POUS.delta_scheme == Fraction(1, 2**128) and WIPE.delta_scheme is None
    assert [c.name for c in POUS.charges if not c.bounded] == ["help"]
