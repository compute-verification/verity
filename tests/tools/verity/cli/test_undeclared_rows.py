"""The served window's table holds a component of its own only where no scheme declares it (`served_table.UNDECLARED`),
a table that only shrinks; every other component is its scheme's declaration (`schemes.get`), and a declared
``depends`` is a requirement a configuration is refused without."""
from __future__ import annotations

import importlib
import sys

import pytest

from verity import config as K
from verity.channel import Refused
from verity.protocols.run import schemes
from verity.protocols.accounting.scheme import Launch
from verity_cli import run as R

REG = R.registry()
T = sys.modules["served_table"]
#: who declares each undeclared row's scheme, and then deletes the row
OWNERS = {"pouw": "compute-accounting"}


def test_the_undeclared_rows_only_shrink() -> None:
    grown = sorted(set(T.UNDECLARED) - set(OWNERS))
    assert not grown, (f"served_table.UNDECLARED gained {grown}: a component is a scheme's declaration "
                       "(verity.protocols.run.schemes), never a new row here")
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
                                           ("dither", "dither is declared, but served_table.PIECES has no pieces")])
def test_the_launch_data_refuses_a_name_it_has_no_pieces_for_by_name(name: str, why: str) -> None:
    cfg = R.load(R.ROOT / "applications" / "onsite_full.toml", REG)
    cfg["protocols"] = [*cfg["protocols"], name]
    for read in (lambda: T.pieces(cfg, T.WORKERS), lambda: T.variables(cfg, {}), lambda: T.unbound(cfg, {})):
        with pytest.raises(Refused, match=why):
            read()


@pytest.mark.parametrize(("given", "name", "detail"), [
    ({"extra": 1}, "unknown-key", "pous has no setting ['extra']: its settings are ['challenges']"),
    ({"challenges": 0}, "params", "pous: challenges is a positive integer, not 0"),
    ({"challenges": "x"}, "params", "pous.challenges is a int, not 'x'")])
def test_a_declared_schemes_parameters_are_its_settings_refused_by_name(given: dict, name: str, detail: str) -> None:
    pous = T.MANIFESTS["pous"]
    assert pous.default_params == schemes.get("pous").settings
    with pytest.raises(Refused) as e:
        pous.params({**pous.default_params, **given})
    assert (e.value.name, e.value.detail) == (name, detail)


def test_every_declared_piece_names_its_entry_and_executables() -> None:
    said = {(n, p): x for n in T.MANIFESTS for p, x in T.launches(n).items()}
    lacking = [k for k, x in said.items() if isinstance(x.entry, T.Lacks) or isinstance(x.executables, T.Lacks)]
    assert not lacking, "each declared piece declares its entry and executables (`Launch.entry`, `Launch.executables`)"
    for (n, p), x in said.items():
        piece = T.PIECES[n][p]
        module, attribute = x.entry.split(":")
        made = getattr(T if module == "served_table" else importlib.import_module(module), attribute)(
            T.UNDECLARED[n][0] if n in T.UNDECLARED else schemes.get(n))
        assert (piece.hooks, piece.scripts, piece.executables) == (made[0], made[1], x.executables)


def test_an_entry_is_the_declarations_alone_and_nothing_a_node_sets_overrides_it(monkeypatch) -> None:
    """`Launch.entry` and `Launch.executables` are read from the registered declaration and nowhere else: the table's
    former by-name launchers (``ENTRIES``, the ``*_LAUNCHER`` attributes) and the scripts' former program overrides
    (``POUS_EXE``, ``BW_CASE_EXE``, ``BW_PARTY_EXE``, ``WIPE_PROTOCOL_EXE``) set to another change no piece."""
    def launched() -> dict:
        return {(n, r): (p.hooks.__code__, p.scripts, p.executables, p.what)
                for n, c in T.MANIFESTS.items() for r, p in T._pieces(c).items()}

    before = launched()
    other = T.Launcher(lambda cfg, params, bound, pinned: {"POUS_CMD": "evil"}, ("evil.sh",))
    monkeypatch.setattr(T, "ENTRIES", {(n, p): other for n in T.MANIFESTS for p in T.launches(n)}, raising=False)
    for name in ("SPOT", "POUW", "PEARLPLUS", "POUS", "BANDWIDTH", "ATTESTER", "WIPE"):
        monkeypatch.setattr(T, f"{name}_LAUNCHER", other, raising=False)
    for var in ("POUS_EXE", "BW_CASE_EXE", "BW_PARTY_EXE", "WIPE_PROTOCOL_EXE", "VERITY_ENTRY"):
        monkeypatch.setenv(var, "/tmp/evil")
    assert launched() == before


def test_the_attesters_piece_is_its_declarations_and_neither_the_tables_former_names_nor_a_node_replace_it(
        monkeypatch) -> None:
    """The network attester's piece is its scheme's `launch_attester` with network-period and network-rows of its
    Lake package: the table's former ``attester_entry``, ``_attester_piece`` and ``ATTESTER_PIECE``, and the
    variables served_verdicts.sh reads the piece's commands from, set to another, change none of it."""
    from verity.protocols.accounting.network import scheme as NS

    def piece() -> tuple:
        p = T._pieces(T.MANIFESTS["network-attester"])[T.WORKERS]
        return p.hooks.__code__, p.scripts, p.executables

    before = piece()
    package = "verity/protocols/accounting/network/verifier/lean"
    assert before == (NS._attester_hooks.__code__, (NS.ATTESTER_PIECE,),
                      tuple(schemes.get("network-attester").pieces[T.WORKERS].executables))
    assert [(e.package, e.target) for e in before[2]] == [(package, "network-period"), (package, "network-rows")]
    other = T.Launcher(lambda cfg, params, bound, pinned: {"CERTIFIER_PERIOD_CMD": "evil"}, ("evil.py",))
    for name, value in (("attester_entry", lambda declaration: other), ("_attester_piece", other.hooks),
                        ("ATTESTER_PIECE", "evil.py"), ("ATTESTER_LAUNCHER", other)):
        monkeypatch.setattr(T, name, value, raising=False)
    for var in ("CERTIFIER_CMD", "CERTIFIER_CHAIN_CMD", "CERTIFIER_PERIOD_CMD", "CERTIFIER_FRONT"):
        monkeypatch.setenv(var, "/tmp/evil")
    assert piece() == before


def test_an_entry_that_gives_no_hooks_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="gives .* not its hooks and scripts"):
        T._launcher("x", T.PROVER, Launch(entry="builtins:str", executables=()), None)


def test_a_piece_reaches_its_declared_channels_gate_party_and_says_one_only_where_none_joins() -> None:
    from dataclasses import replace

    from verity.protocol import Channel
    pous = T.MANIFESTS["pous"]
    assert T._reaches(pous, T.PROVER, "") == T.MEMORY_ATTESTER
    with pytest.raises(ValueError, match="by its declared channels, so its piece says no reaches"):
        T._reaches(pous, T.PROVER, T.MEMORY_ATTESTER)
    spot = T.MANIFESTS["sampled-proofs"]
    assert T._reaches(spot, T.WORKERS, "") == "" and T.PIECES["sampled-proofs"][T.WORKERS].reaches == T.CHALLENGER
    joined = replace(spot, channels=(*spot.channels, Channel(T.CHALLENGER, T.WORKERS, ("Sample",))))
    with pytest.raises(ValueError, match="so its piece says no reaches"):
        T._reaches(joined, T.WORKERS, T.CHALLENGER)
    assert T.PIECES["bandwidth"][T.PROVER].follows == "pous" and not T.PIECES["network-attester"][T.WORKERS].follows
