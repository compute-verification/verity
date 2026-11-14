"""`verity_vllm.check.replay.pouw_circuit` beside a whole replay, on the calls the PoUW end-to-end world commits
(`tests.check.test_pouw_ncp_v2_end_to_end`): the record of a window that committed no call, the reasons a record names
beside the reasons it counts, the composition entry the live run is asked for under, and what the tile tamper commits
wrong."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from verity.protocols.accounting.pouw.prover.circuit import plan as CPL

from tests.check import test_pouw_ncp_v2_end_to_end as E2E
from verity_vllm.check.replay import pouw_circuit as RC
from verity_vllm.protocols import pouw
from verity_vllm.protocols.pouw import native as PN
from verity_vllm.records.schema_ids import POUW_CIRCUIT_REPLAY_SCHEMA

_module_body_query = E2E._module_body_query
world = E2E.world

NO_CALL = "no ncp-v2 call was committed"


def test_a_window_that_committed_no_call_is_refused_by_name_and_its_record_holds_nothing_else(world):
    """Beside the reason and its count the record holds its schema and the budget of record; every other field is zero or
    empty, but the time the replay took and the note of what is not of record."""
    record = RC.replay([], world.commitment)
    said = {"schema": POUW_CIRCUIT_REPLAY_SCHEMA, "result": False, "why": [NO_CALL], "why_n": 1, "record_k": CPL.RECORD_K}
    assert {k: record[k] for k in said} == said
    assert not [k for k in set(record) - set(said) - {"seconds", "not_of_record"} if record[k]]
    assert record["not_of_record"].startswith("the quantizer and the int7 linear of the model rows are evaluated natively")


def test_a_record_names_its_first_twelve_reasons_and_counts_them_all(world, monkeypatch):
    """Fourteen reasons: the window's own, and thirteen its anchors give, stated here in place of thirteen calls anchored wrong."""
    anchored = [f"call {i}: anchored salt is not the verifier's" for i in range(13)]
    monkeypatch.setattr(RC, "anchors", lambda records, salt, first: anchored)
    record = RC.replay([], world.commitment)
    assert (record["result"], record["why"], record["why_n"]) == (False, [NO_CALL, *anchored[:11]], 14)


@pytest.mark.parametrize("entries, asked_for", [
    (None, []), ((), []), (("spot",), []), (("pouw:ncp-v1",), []), (("other:ncp-v2",), []), (("pouw:ncp-v2:x",), []),
    (("spot", "pouw:ncp-v2"), ["pouw:ncp-v2"])],
    ids=["no entries stated", "no entry", "spot alone", "another scheme", "another protocol", "more than a scheme", "beside spot"])
def test_the_live_run_is_asked_for_under_the_composition_s_ncp_v2_entry_and_under_no_other(monkeypatch, entries, asked_for):
    """An entry is a protocol's name, or its name and scheme; a composition without ``pouw:ncp-v2`` has no circuit replay."""
    asked: list[str] = []
    monkeypatch.setattr(pouw, "circuit_run", asked.append)
    with RC.around(SimpleNamespace(entries=entries)) as circuit:
        assert circuit is None
    assert asked == asked_for


def test_the_tile_tamper_commits_one_word_of_each_tile_of_the_first_call_with_its_low_bit_flipped_and_keeps_the_other_calls(world):
    records, where = RC.tamper_tiles(world.records)
    first, wrong = world.records[0], records[0]
    native = PN.Native(first.call, first.scheme)
    tiles = [native.index("tile", rs, cs) for rs in range(native.ms) for cs in range(native.ns)]
    assert where == {"what": "tile", "call": first.index, "units": tiles, "leaf": RC.TAMPER_LEAF} and len(tiles) > 1
    assert len(world.records) > 1 and records[1:] == world.records[1:]
    for u in tiles:
        flipped = [int(a) ^ int(b) for a, b in zip(wrong.opening(u), first.opening(u), strict=True)]
        assert flipped == [int(leaf == RC.TAMPER_LEAF) for leaf in range(len(flipped))]
