"""What H1's readers make of a part their input leaves out: a rank's release whose committer states no errors, a pull
from a rank that shipped no acquisitions, a window whose details state no tie to the Commit, a call with no tables,
commitments that name no session, a manifest row with no tables, a window's record with no commitment, a slot of a
round the round log does not hold, an engine that holds no executor in its own process, and a window that holds no
Program digest; and of a round the log names by no number."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from verity.protocols.proofs.spot.verifier.challenger.challenger import (
    OPENING_FORM,
    LiveCommitment,
)
from verity.protocols.resource_accounting.compute.pouw.prover.serving.h1 import (
    CHALLENGER,
    H1Run,
    registered,
)
from verity.protocols.resource_accounting.compute.pouw.prover.serving.h1 import prover as PV
from verity.protocols.resource_accounting.compute.pouw.prover.serving.coins.challenger import (
    drawn_of,
)
from verity.protocols.resource_accounting.compute.pouw.prover.serving.h1.commit import (
    ROOT_BYTES,
    SCHEME,
    UNTIED,
)
from verity.protocols.resource_accounting.compute.pouw.prover.serving.coins.requests import (
    RoundOpening,
    slot_reason,
    step_digest,
    table_root,
    tables_reason,
)
from verity.protocols.resource_accounting.compute.pouw.prover.serving.h1.served import (
    Served,
    declared,
    descriptor,
)

from tests.protocols.test_pouw_h1_served import _unit

#: what a rank's release hands on of its committer's finalize beside the errors (`h1rank_release`)
COUNTS = {"expected_tensors": 2, "committed_tensors": 2, "errors_n": 0}


class _Committer:
    """What `RankExtension.h1rank_release` reads of a rank's committer: rows-sha512's scheme, its name and its finalize's
    ``extra``."""

    commit_scheme, name = SCHEME, "stand-in"

    def __init__(self, extra: dict[str, Any]) -> None:
        self.extra, self.closed = extra, False

    def finalize(self) -> SimpleNamespace:
        return SimpleNamespace(run_root=bytes(ROOT_BYTES), extra=self.extra)

    def close(self) -> None:
        self.closed = True


class _Tap:
    """What `Holder.pull` and `ranks.feed` use of the window's tap: its port's name, what it took (the name, the engine
    step and the count of ids) and what it refused."""

    port = "ids"

    def __init__(self) -> None:
        self.took: list[tuple[str, int, int | None]] = []
        self.refused: list[str] = []
        self._ids: dict[int, Any] = {}

    def take(self, name: str, t: Any, step: int, n: int | None = None) -> None:
        self.took.append((name, step, n))


def _rank(lo: int, **more: Any) -> dict[str, Any]:
    """A rank's answer to ``h1rank_rows`` with no call since the last pull."""
    return {"lo": lo, "rows": [], "refused": [], "order_ok": True, "unopened": 0, **more}


SHIPPED = [("site/0", 2, b"x_q"), ("ids", 2, [5, 6])]


#: the gate's verdict for the served call of `_window`
CREDITED = {"linear": "q", "credit": True, "reason": None}
#: a table's root as a manifest row states it
ROOT = "ab" * 64


def _window(details: dict[str, Any], tables: dict[int, Any]) -> Any:
    """What `H1Run.gated` and `H1Run.credit_bits` read of a window: a served call (0) that the gate credits and a call
    (1) that was not served, under the challenger's coins and with every request entered; its `details` and its calls'
    `tables`."""
    calls = [SimpleNamespace(index=0, m=1, module="q"), SimpleNamespace(index=1, m=0, module="r")]
    run = SimpleNamespace(calls=calls, details=details, tables=tables, coins_from=CHALLENGER, request_gaps=list,
                          verdict=lambda call: dict(CREDITED))
    run.gated = lambda: H1Run.gated(run)
    return run


@pytest.mark.parametrize("details", [{}, {"roots1": None}, {"roots1": {}}, {"roots1": 0}, {"roots1": {"reason": None}}],
                         ids=["absent", "null", "empty", "a false value", "tied"])
def test_a_window_whose_details_state_nothing_against_its_tie_to_the_commit_keeps_the_gates_credit(details):
    assert H1Run.gated(_window(details, {})) == {0: CREDITED}


def test_a_window_whose_tie_to_the_commit_states_a_reason_loses_the_gates_credit():
    assert H1Run.gated(_window({"roots1": {"reason": "ids"}}, {})) == {0: {**CREDITED, "credit": False, "reason": UNTIED}}


@pytest.mark.parametrize("tables", [{}, {0: None}, {0: {}}, {0: 0}, {0: {"a": ROOT}}, {0: {"a": ROOT, "z": None}}],
                         ids=["absent", "null", "empty", "a false value", "an A table alone", "a null z"])
def test_a_credited_call_with_no_z_table_has_no_credit_bit(tables):
    assert H1Run.credit_bits(_window({}, tables)) == {0: False, 1: False}


def test_a_credited_call_with_a_z_table_has_its_credit_bit_and_a_call_not_served_has_none():
    assert H1Run.credit_bits(_window({}, {0: {"z": ROOT}, 1: {"z": ROOT}})) == {0: True, 1: False}


class _Settling(SimpleNamespace):
    """`_window`'s window with its `tables` read as `H1Run.tables` is, a property that settles the tables still hashing:
    it counts its reads and raises what a failed hash raises."""

    reads = 0

    @property
    def tables(self) -> dict[int, Any]:
        self.reads += 1
        raise RuntimeError("a table's hash failed")


def _settling(details: dict[str, Any]) -> Any:
    run: Any = _Settling(**{name: part for name, part in vars(_window(details, {})).items() if name not in ("tables", "gated")})
    run.gated = lambda: H1Run.gated(run)
    return run


def test_a_window_with_no_call_the_gate_credits_gives_its_credit_bits_without_settling_its_tables():
    run = _settling({"roots1": {"reason": "ids"}})
    assert H1Run.credit_bits(run) == {0: False, 1: False} and run.reads == 0


def test_a_call_the_gate_credits_settles_the_windows_tables_and_a_failed_hash_raises_there():
    run = _settling({})
    with pytest.raises(RuntimeError, match="a table's hash failed"):
        H1Run.credit_bits(run)
    assert run.reads == 1


@pytest.mark.parametrize("session", [{}, {"session": None}, {"session": {}}, {"session": 0}, {"session": {"session": None}}],
                         ids=["absent", "null", "empty", "a false value", "an object that names none"])
def test_commitments_that_name_no_session_give_a_window_with_no_key_of_its_own_none(session):
    commits = {"weights_root": "00" * 64, "requests_root": "00" * 64, **session}
    with pytest.raises(ValueError, match="the window has no key: neither one given nor a window-open root"):
        Served.key(SimpleNamespace(window=None), commits)


def test_a_window_that_holds_no_program_digest_commits_a_tree_whose_domain_names_no_program():
    """`Served._tree` hands `registered.commit` the Program digest as the window holds it: None until `Served.declare`,
    which the tree's domain then names as its program, as it names any digest."""
    words, run = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], np.uint16), SimpleNamespace(_salts=PV.RunSalts())
    tree = Served()._tree(run, "t/x", words, 0, ROOT)
    salts = run._salts.at(0, 2)
    assert tree.program is None and tree.root == registered.commit(None, "t/x", words, salts, window=ROOT).root
    assert tree.root != registered.commit("p", "t/x", words, salts, window=ROOT).root


@pytest.mark.parametrize("row", [{}, {"tables": None}, {"tables": {}}, {"tables": 0}, {"tables": {"z": ROOT}}],
                         ids=["absent", "null", "empty", "a false value", "a z table alone"])
def test_a_manifest_row_with_no_a_table_states_no_root_of_it(row):
    assert table_root(row, "a") is None


def test_a_manifest_rows_table_root_is_its_tables_entry_of_that_name():
    assert table_root({"tables": {"a": ROOT, "z": None}}, "a") == ROOT


def test_a_manifest_row_whose_tables_are_a_list_raises_at_the_read_of_its_entry():
    with pytest.raises(AttributeError):
        table_root({"tables": ["x"]}, "a")


def _descriptor() -> bytes:
    """The descriptor bytes of a declared window of one call (N = 64 over K = 64, 32 slots, one step) on the served
    test's stand-in proof block: a window whose layout core reads, so a refusal below is the record's."""
    return descriptor(declared([(64, 64)], {0: "q"}, 32, 1, unit=_unit()))


def _sampled(record: dict[str, Any]) -> tuple[Any, Any]:
    """What core samples from the window's `record` at K = 1 over that descriptor, or its refusal."""
    return drawn_of(record, {"root": "00"}, 1, _descriptor())


def _refusal(why: str) -> tuple[None, tuple[str, str]]:
    """The refusal of a record whose sampling coin cannot be read, for the reason `why`."""
    return None, ("draw", f"the window's draw coin is malformed ({why})")


@pytest.mark.parametrize("record", [{}, {"commitment": None}, {"commitment": []}, {"commitment": 0}],
                         ids=["absent", "null", "empty", "a false value"])
def test_a_record_with_no_commitment_is_refused_at_its_sampling_coin_as_one_whose_commitment_states_no_sample(record):
    assert _sampled(record) == _refusal(OPENING_FORM)


@pytest.mark.parametrize("sample", [..., None, {}, {"coin": {}}])
def test_a_commitment_that_holds_no_opened_sampling_coin_is_refused_at_it(sample):
    """The commitment object states no sample (``...``), or one that is no object or holds no opened coin."""
    commitment = {} if sample is ... else {"draw": sample}
    assert _sampled({"commitment": commitment}) == _refusal(OPENING_FORM)


def test_a_slot_of_a_round_the_log_does_not_hold_is_refused_before_anything_is_opened():
    """`slot_reason` hands on the round's own refusal: the coins' commitment, the salt and the step's calls go unread."""
    coins = LiveCommitment("pouw-h1/rounds", 0, (("round", 2),), bytes(64))
    slot = RoundOpening(round=3, position=0, value=b"", path=())
    assert slot_reason([{"round": 0}, {"round": 1}], coins, b"salt", 4, [], slot) == "the round log has no round 3"


def _step(r: Any) -> list[dict[str, Any]]:
    """The manifest row of the one call of the step the log names `r`, with its A table's root."""
    return [{"step": r, "index": 0, "tables": {"a": "ab" * 64}}]


@pytest.mark.parametrize("r", [0, "0"], ids=["a number", "a string"])
def test_a_step_is_held_to_its_rounds_calls_digest_whatever_the_log_names_the_round(r):
    """`tables_reason` holds a step to `step_digest` over the name the log gives its round, and the name to nothing else."""
    rows = _step(r)
    assert tables_reason(b"salt", [{"round": r, "calls": step_digest(b"salt", r, rows).hex()}], rows) is None
    assert tables_reason(b"salt", [{"round": r, "calls": "00"}], rows) == (
        f"step {r}'s calls in the record (their tables' roots among them) are not the ones under round {r}'s root")


def test_a_round_the_log_names_by_what_no_digest_takes_raises_as_the_digest_does():
    with pytest.raises(TypeError, match="parts are bytes, str or nonnegative int, got float"):
        tables_reason(b"salt", [{"round": 0.0, "calls": "00"}], _step(0))


def _engine(**holds: Any) -> SimpleNamespace:
    """What `executor_of` reads of vLLM's ``LLM``: its engine, which holds `holds`."""
    return SimpleNamespace(llm_engine=SimpleNamespace(**holds))


