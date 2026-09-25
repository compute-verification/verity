"""Tests for :mod:`accumulation.exact`.

1. The SPEC §1 checker (``is_legal`` / ``partition_cost``) on hand-written circuits with hand-computed answers.
2. Solver agreement: brute force == MILP (both formulations) on every micro program brute force can handle.
3. The sandwich ``L <= I* <= U`` for the certified lower bound and the constructive upper bound on the
   flattened micro circuits (skipped while the bound modules are not importable).
4. The op-graph extractor + ``check_against_flat`` on every micro program.

Run: ``PYTHONPATH=. .venv/bin/pytest accumulation/tests/test_exact.py -q``
"""

from __future__ import annotations

import functools
import importlib
import inspect
import os

import pytest

from accumulation.exact import (
    BRUTE_MAX_GATES,
    Infeasible,
    exact_min_input,
    flatten,
    is_legal,
    make_flat,
    partition_cost,
    up_set,
)
from accumulation.exact.micro import (
    MICRO_SUITE,
    gate_count,
    micro_programs,
    smallest_registry_programs,
)

FS = (2, 4, 1 << 30)
XS = (8, 16, 32, 64, 1 << 20)
GRID = [(F, X) for F in FS for X in XS]

#: the sandwich sweeps cover programs up to this many gates by default (whole file < 1 min); the three
#: 32-36 gate programs add ~5 min (MILP cells of up to ~1 min each) -- set EXACT_ALL=1 to include them
SWEEP_MAX_GATES = 40 if os.environ.get("EXACT_ALL") else 28


# ---------------------------------------------------------------------------------------------------------
# hand-written circuits
# ---------------------------------------------------------------------------------------------------------

def chain_circuit():
    """0: a (token, 16b)  1: w (accumulated, 16b)  2: f (fixed, 16b)
       3 = g(a, w)      work 1, 32b
       4 = g(3, f)      work 1, 32b
       5 = round(4)     work 1, 16b"""
    return make_flat(work=[0, 0, 0, 1, 1, 1], width_bits=[16, 16, 16, 32, 32, 16],
                     operands=[(), (), (), (0, 1), (3, 2), (4,)], input_role=["token", "accumulated", "fixed", None, None, None],
                     outputs=[5])


def test_partition_cost_hand():
    f = chain_circuit()
    assert f.gates == (3, 4, 5)
    assert f.nonfixed_inputs == frozenset({0, 1})
    # one RU: a + w (fixed f free) = 4 bytes
    assert partition_cost(f, {3: 0, 4: 0, 5: 0}) == 4
    # {3} | {4, 5}: RU0 imports a, w (4); RU1 imports the 32-bit output of 3 (4) -- f is free
    assert partition_cost(f, {3: 0, 4: 1, 5: 1}) == 8
    # singletons: 4 + 4 + 4
    assert partition_cost(f, {3: 0, 4: 1, 5: 2}) == 12
    # RU labels are arbitrary hashables
    assert partition_cost(f, {3: "x", 4: "y", 5: "y"}) == 8


def test_is_legal_hand():
    f = chain_circuit()
    one = {3: 0, 4: 0, 5: 0}
    assert up_set(f, one, 5) == frozenset({3, 4, 5})
    assert is_legal(f, one, F=3, X=4) == (True, "ok")
    ok, why = is_legal(f, one, F=2, X=4)          # Up(5) = {3, 4, 5} has work 3 > 2
    assert not ok and "gate 5" in why and "F=2" in why
    ok, why = is_legal(f, one, F=3, X=3)          # in(RU0) = 4 > 3
    assert not ok and "X=3" in why
    split = {3: 0, 4: 1, 5: 1}
    assert up_set(f, split, 5) == frozenset({4, 5})
    assert is_legal(f, split, F=2, X=4)[0]
    assert not is_legal(f, split, F=1, X=4)[0]
    assert is_legal(f, {3: 0, 4: 1, 5: 2}, F=1, X=4)[0]
    # incomplete assignment / assigning an Input gate are rejected
    assert not is_legal(f, {3: 0, 4: 0}, F=9, X=99)[0]
    assert not is_legal(f, {0: 0, 3: 0, 4: 0, 5: 0}, F=9, X=99)[0]


def test_repeated_operand_counted_once():
    # gate 2 reads Input 0 twice; gate 3 reads gate 2 twice
    f = make_flat(work=[0, 0, 1, 1], width_bits=[16, 16, 32, 16], operands=[(), (), (0, 0, 1), (2, 2)],
                  input_role=["accumulated", "token", None, None])
    assert f.operands[2] == (0, 1) and f.operands[3] == (2,)
    assert partition_cost(f, {2: 0, 3: 0}) == 4          # a (2) + t (2), not 6
    assert partition_cost(f, {2: 0, 3: 1}) == 4 + 4      # + one 32-bit import, not two


def test_fixed_inputs_free_and_zero_work():
    # 0, 1 fixed; 2 = const (work 0); 3 = g(0, 1, 2) work 1; 4 = g(3) work 0
    f = make_flat(work=[0, 0, 0, 1, 0], width_bits=[16, 16, 32, 32, 16], operands=[(), (), (), (0, 1, 2), (3,)],
                  input_role=["fixed", "fixed", None, None, None])
    assert f.nonfixed_inputs == frozenset()
    assert partition_cost(f, {2: 0, 3: 0, 4: 0}) == 0
    assert partition_cost(f, {2: 0, 3: 1, 4: 1}) == 4    # the constant's 32-bit output crosses
    assert is_legal(f, {2: 0, 3: 1, 4: 2}, F=1, X=4)[0]
    assert is_legal(f, {2: 0, 3: 0, 4: 0}, F=1, X=0)[0]  # zero-work gates never push Up over F=1
    assert not is_legal(f, {2: 0, 3: 0, 4: 0}, F=0, X=0)[0]
    r = exact_min_input(f, 1, 0)
    assert r.total == 0


@pytest.mark.parametrize("method", ["brute", "milp", "milp-assign"])
def test_exact_hand_answers(method):
    f = chain_circuit()
    assert exact_min_input(f, 3, 4, method=method).total == 4      # all in one RU
    assert exact_min_input(f, 2, 4, method=method).total == 8      # F forces one split: one 32-bit wire crosses
    assert exact_min_input(f, 1, 4, method=method).total == 12     # singletons
    r = exact_min_input(f, 2, 100, method=method)
    assert r.total == 8 and is_legal(f, r.assignment, 2, 100)[0] and partition_cost(f, r.assignment) == 8
    assert set(r.assignment) == {3, 4, 5}
    with pytest.raises(Infeasible):
        exact_min_input(f, 1, 3, method=method)                     # gate 3 alone needs 4 bytes > X=3
    with pytest.raises(Infeasible):
        exact_min_input(f, 0, 100, method=method)                   # work 1 gates cannot satisfy F=0


def test_spec_call_form():
    bp = MICRO_SUITE[0].build()
    a = exact_min_input(bp, 4, 16)
    b = exact_min_input(bp.program, bp.roles, 4, 16)
    assert a.total == b.total


# ---------------------------------------------------------------------------------------------------------
# micro programs
# ---------------------------------------------------------------------------------------------------------

@functools.cache
def _programs():
    return dict(micro_programs(max_gates=40))


@functools.cache
def _flat(name):
    return flatten(_programs()[name])


@functools.cache
def _exact(name, F, X):
    return exact_min_input(_flat(name), F, X, method="auto")


def _names(max_gates, min_gates=0):
    return [s.name for s in MICRO_SUITE if min_gates <= gate_count(s.build()) <= max_gates]


def test_micro_gate_counts_match_docstrings():
    for s in MICRO_SUITE:
        bp = s.build()
        k = bp.notes
        f = flatten(bp)
        n = gate_count(bp)
        assert n == f.n_gates
        if s.factory.__name__ == "matmul_only":
            assert n == k["M"] * k["N"] * (k["K"] + 2)
        elif s.factory.__name__ == "two_matmul_chain":
            assert n == k["M"] * k["N1"] * (k["K"] + 2) + k["M"] * k["N2"] * (k["N1"] + 2)
        elif s.factory.__name__ == "matmul_add":
            assert n == k["M"] * k["N"] * (k["K"] + 2) + k["M"] * k["N"]
        elif s.factory.__name__ == "fixed_then_accum":
            assert n == k["M"] * k["N"] * (k["K"] + 2) + k["M"] * k["N2"] * (k["N"] + 2)
        assert 3 <= n <= 40


def test_micro_roles_flatten():
    f = flatten(_programs()["fixacc_1x1x1"])
    roles = {f.input_param[i]: f.input_role[i] for i in range(f.n) if f.is_input[i]}
    assert roles == {"A": "token", "Wf": "fixed", "Wa": "accumulated"}
    assert f.nonfixed_input_bytes() == 2 + 2                # A (1 elem) + Wa (1 elem); Wf free
    r = exact_min_input(f, 1 << 30, 1 << 20)
    assert r.total == 4 and r.n_units == 1


@pytest.mark.parametrize("name", _names(BRUTE_MAX_GATES))
@pytest.mark.parametrize("F,X", GRID)
def test_brute_equals_milp_small(name, F, X):
    f = _flat(name)
    b = exact_min_input(f, F, X, method="brute")
    m = exact_min_input(f, F, X, method="milp")
    a = exact_min_input(f, F, X, method="milp-assign")
    assert b.optimal and m.optimal and a.optimal
    assert b.total == m.total == a.total, (name, F, X, b.total, m.total, a.total)
    for r in (b, m, a):
        assert is_legal(f, r.assignment, F, X)[0]
        assert partition_cost(f, r.assignment) == r.total


@pytest.mark.parametrize("name", _names(24, BRUTE_MAX_GATES + 1))
@pytest.mark.parametrize("F,X", GRID)
def test_brute_equals_milp_medium(name, F, X):
    f = _flat(name)
    b = exact_min_input(f, F, X, method="brute")
    m = exact_min_input(f, F, X, method="milp")
    assert b.optimal and m.optimal
    assert b.total == m.total, (name, F, X, b.total, m.total)


@pytest.mark.parametrize("name", _names(BRUTE_MAX_GATES))
def test_presolve_is_lossless(name):
    f = _flat(name)
    for F, X in ((2, 8), (4, 16), (1 << 30, 1 << 20)):
        a = exact_min_input(f, F, X, method="brute", presolve=True)
        b = exact_min_input(f, F, X, method="brute", presolve=False)
        assert a.total == b.total


# ---------------------------------------------------------------------------------------------------------
# operator graph on the micro programs
# ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", _names(40))
def test_extract_and_validate(name):
    from accumulation.graph import extract
    from accumulation.graph.validate import check_against_flat
    bp = _programs()[name]
    g = extract(bp)
    summary = check_against_flat(bp, g)
    assert summary["ops"] >= 1
    # the graph's non-fixed root bytes must equal the flattened circuit's
    f = _flat(name)
    nonfixed = sum(t.bytes for t in g.tensors if t.kind == "param" and t.role != "fixed")
    assert nonfixed == f.nonfixed_input_bytes()
    assert g.total_work() == f.total_work()


# ---------------------------------------------------------------------------------------------------------
# sandwich: L <= I* <= U
# ---------------------------------------------------------------------------------------------------------

def _try_import(modname):
    try:
        return importlib.import_module(modname)
    except Exception:  # noqa: BLE001 - module being written concurrently
        return None


def _lower_total(mod, g, bp, F, X):
    fn = mod.lower_bound
    n_req = sum(1 for p in inspect.signature(fn).parameters.values()
                if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD))
    res = fn(g, bp.roles, F, X) if n_req >= 4 else fn(g, F, X)
    return int(res.total), res


def _upper_total(mod, g, bp, F, X, **kw):
    fn = mod.upper_bound
    params = inspect.signature(fn).parameters
    kw = {k: v for k, v in kw.items() if k in params}
    if "program" in params:
        kw.setdefault("program", bp.program)
    res = fn(g, F, X, **kw)
    return int(res.total), res


def _dump(name, F, X, r):
    f = _flat(name)
    lines = [f"{name} F={F} X={X}: I*={r.total} ({r.n_units} RUs, {r.method})"]
    for ru, gs in r.units().items():
        lines.append(f"  RU{ru}: gates {gs}")
    lines.append(f.describe())
    return "\n".join(lines)


@pytest.mark.parametrize("name", _names(SWEEP_MAX_GATES))
@pytest.mark.parametrize("F,X", GRID)
def test_lower_bound_sound(name, F, X):
    mod = _try_import("accumulation.bounds.lower")
    if mod is None:
        pytest.skip("accumulation.bounds.lower not importable yet")
    from accumulation.graph import extract
    bp = _programs()[name]
    r = _exact(name, F, X)
    L, res = _lower_total(mod, extract(bp), bp, F, X)
    assert L <= r.total, (f"UNSOUND LOWER BOUND: L={L} > I*={r.total} on {name} (F={F}, X={X}); per_op={res.per_op} "
                          f"source={getattr(res, 'source', None)} cap={getattr(res, 'cap', None)}\n" + _dump(name, F, X, r))


@pytest.mark.parametrize("name", _names(SWEEP_MAX_GATES))
@pytest.mark.parametrize("F,X", GRID)
def test_upper_bound_legal(name, F, X):
    """``recompute=False``: the plan is a partition of the literal gate set, exactly SPEC §1's object."""
    mod = _try_import("accumulation.bounds.upper")
    if mod is None:
        pytest.skip("accumulation.bounds.upper not importable yet")
    from accumulation.graph import extract
    bp = _programs()[name]
    r = _exact(name, F, X)
    U, res = _upper_total(mod, extract(bp), bp, F, X, recompute=False)
    assert U >= r.total, (f"ILLEGAL UPPER BOUND: U={U} < I*={r.total} on {name} (F={F}, X={X}); per_op={res.per_op} "
                          f"notes={res.notes}\n" + _dump(name, F, X, r))


@pytest.mark.parametrize("name", _names(SWEEP_MAX_GATES))
@pytest.mark.parametrize("F,X", GRID)
def test_lower_le_upper(name, F, X):
    lo = _try_import("accumulation.bounds.lower")
    up = _try_import("accumulation.bounds.upper")
    if lo is None or up is None:
        pytest.skip("bound modules not importable yet")
    from accumulation.graph import extract
    bp = _programs()[name]
    g = extract(bp)
    L, _ = _lower_total(lo, g, bp, F, X)
    U, _ = _upper_total(up, g, bp, F, X)
    assert L <= U, (name, F, X, L, U)


# ---------------------------------------------------------------------------------------------------------
# registry programs at the exact-solver scale
# ---------------------------------------------------------------------------------------------------------

def test_smallest_registry_programs_report():
    kept, skipped = smallest_registry_programs(limit=40)
    # every registry algorithm is far above 40 gates even at `tiny`; the micro factories are the test bed
    assert isinstance(kept, list) and isinstance(skipped, dict)
    assert kept == [] or all(n <= 40 for _, _, _, n in kept)
    assert skipped, "expected the registry algorithms to be reported as too large or unbuildable"
