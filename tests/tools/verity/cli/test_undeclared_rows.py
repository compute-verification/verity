"""The served window's table holds a component of its own only where no scheme declares it (`served_table.UNDECLARED`),
a table that only shrinks; every other component is its scheme's declaration (`schemes.get`), and a declared
``depends`` is a requirement a configuration is refused without."""
from __future__ import annotations

import sys

import pytest

from verity import config as K
from verity.channel import Refused
from verity.protocols.compose import schemes
from verity_cli import run as R

REG = R.registry()
T = sys.modules["served_table"]
#: who declares each undeclared row's scheme, and then deletes the row
OWNERS = {"sampled-proofs": "proofs", "pouw": "compute-accounting", "inference-only": "proofs (workload classification)",
          "certifier": "network-accounting"}


def test_the_undeclared_rows_only_shrink() -> None:
    grown = sorted(set(T.UNDECLARED) - set(OWNERS))
    assert not grown, (f"served_table.UNDECLARED gained {grown}: a component is a scheme's declaration "
                       "(verity.protocols.compose.schemes), never a new row here")
    now = sorted(n for n in T.UNDECLARED if n in schemes.DECLARATIONS)
    assert not now, "; ".join(f"{n} is declared now: {OWNERS[n]} deletes its row from served_table.UNDECLARED "
                              "(and from OWNERS here)" for n in now)


@pytest.mark.parametrize("name", sorted(set(T.MANIFESTS) - set(OWNERS)))
def test_a_declared_component_is_its_declaration(name: str) -> None:
    s, c = schemes.get(name), T.MANIFESTS[name]
    assert (c.parties, c.channels, c.colocated, c.needs, c.requires, T.ROWS[name]) == (
        s.parties, s.channels, s.colocated, s.needs, s.depends, s.record)


def test_the_bandwidth_audit_without_storage_is_refused_by_name() -> None:
    """The bandwidth audit runs over storage's store (its declared ``depends``)."""
    without = [n for n in K.COMPOSITES[T.ONSITE] if n != "pous"]
    with pytest.raises(Refused, match="bandwidth requires pous, which protocols doesn't list"):
        K.new(T.SERVED_WINDOW, without, REG.components, REG.workloads, "0" * 40)


@pytest.mark.parametrize(("name", "why"), [("nonesuch", "nonesuch is no declared scheme"),
                                           ("dither", "dither is declared, but served_table.PARTIES has no pieces")])
def test_the_launch_data_refuses_a_name_it_has_no_pieces_for_by_name(name: str, why: str) -> None:
    cfg = R.load(R.ROOT / "applications" / "onsite_full.toml", REG)
    cfg["protocols"] = [*cfg["protocols"], name]
    for read in (lambda: T.pieces(cfg, T.WORKERS), lambda: T.variables(cfg, {}), lambda: T.unbound(cfg, {})):
        with pytest.raises(Refused, match=why):
            read()
