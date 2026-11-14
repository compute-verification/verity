"""The second stage's reads of a replay block whose fused blocks open registered tables (`two_stage.verdicts`,
`two_stage.verifier_reads`): each block's ``root`` and ``domain`` are values the verifier supplies from the entries it
pinned (`interior_rows.Supply`'s ``tables``), and its verdict ``o1`` reads a value the verifier fills with 1
(`second_stage.VERDICT_ONE`), placed as its root, so Lean's registered reads open a block's verdict only where it is 1.
Without the verdicts the second stage reads ``o1`` at the block's own port, which holds it only to the prover's
commitment of it.

The block is the registered fused family with a stand-in for its read, as `adapters/vllm`'s registered-tables test
builds it, so its member is laid out in a quick test's seconds; the interior record is written by hand, its roots
placeholders, since only the reads the verifier derives from it are checked here. No interior is committed and no
session is proved."""

from __future__ import annotations

import functools
import importlib
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from verity.circuits import boolean as BL
from verity.circuits.boolean import trace as TR
from verity.circuits.boolean.forms import ONE
from verity.circuits.codec import descriptor_id
from verity.circuits.defs import bind
from verity.circuits.ops.boolean import gemm as BG
from verity.circuits.types import Array, Tuple
from verity.commitments.gates import embedding as EMB
from verity.definitions.registry import boolean_fp8_moe as BM
from verity.protocols.proofs.spot.verifier import second_stage as SS
from verity_vllm.pipeline import chunk_rows as CR
from verity_vllm.pipeline import interior_links as IL
from verity_vllm.pipeline import interior_rows as IR
from verity_vllm.pipeline import serving_rows as SRW
from verity_vllm.pipeline import serving_templates as ST

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
TS = importlib.import_module("two_stage")

#: experts a table, a block's K e4m3 bytes, a table row's u16 words; the tables a span reads and the blocks on each
E, K, H, T, W = 4, 32, 16, 2, 2
CELLS, Q, SAMPLE, BOUNDARY = T * W, 2, (0, 2), "b1" * 64
KEY = "replay-block-test"
PARAMS = tuple(f"weights/model.layers.0.mlp.experts.{p}@{H}/b0/table" for p in ("gate_up_proj", "down_proj"))
#: unit j of a span opens table j // W
AT = tuple(j // W for j in range(CELLS))


def _read_signature(S_):
    return EMB.read(S_.V, S_.H).signature()[0], Tuple(Array(S_.H, EMB.BF16), BL.BIT)


READ = TR.part("TwoStageVerdictsTestRead", ("V", "H"), _read_signature,
               lambda C, S_, *e: [*e[17:17 + S_.H], [ONE]], "h = row, ok = 1: a read's ports with none of its hashes",
               version=0, register=False)
FUSED = bind(BM.MoeExpertFp8ReadGemmV1, READ=bind(READ, V=E, H=H), K=K, N=1, G=K, KB=K, SWAP=False,
             DOT=BG.HopperE4m3MmaV2Dot32)
TID = descriptor_id(FUSED)
SUPPLIES = tuple(IR.Supply(TID, p, f"{KEY}/{p}", T, AT, tables=PARAMS) for p in IR.TABLE)


@pytest.fixture(scope="module", autouse=True)
def _template():
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(ST._TEMPLATES, TID, SRW.Template(TID, FUSED, TID, "fused"))
        yield


@functools.cache
def _member():
    return IR.named(CR.unit_member(SRW.template(TID), CELLS))


def _links() -> tuple[IR.Link, ...]:
    ops = tuple(IR.Operand(TID, p, f"{KEY}/{p}", CELLS, tuple(range(CELLS)), w) for p, w in _member().rows
                if p not in IR.TABLE)
    return (*SUPPLIES, *ops)


def _design(**kw) -> TS.Design:
    links = _links()
    return replace(TS.Design(KEY, (_member(),), (CELLS,), links, (), (), verdicts=TS.verdicts([_member()], links)), **kw)


def _interior(d: TS.Design, window, *, verdicts: bool = True) -> dict:
    """An interior record registering every value of ``d``'s links (and its verdicts' values with ``verdicts``) and
    committing every other port of ``window``'s member, each under a placeholder root."""
    m, = window.members
    registered = [{"value": v, "root": f"{k:02x}" * 64, "leaves": Q * per, "domain": "d0" * 64}
                  for k, (v, per, _leaf) in enumerate(IR.link_values(d.links, Q))]
    registered += [{"value": x.value, "root": "0e" * 64, "leaves": Q * x.per, "domain": "d1" * 64}
                   for x in d.verdicts if verdicts]
    linked = {x.port for x in d.links}
    roots = [{"port": window.held(m, p), "root": "aa" * 64, "leaves": max(window.held_leaves(m, p)) + 1,
              "domain": "da" * 64} for p, _w in m.all_rows if p not in linked]
    return {"registered": registered, "roots": roots}


def test_a_fused_block_on_registered_tables_has_its_verdict_at_a_value_placed_as_its_root():
    """The block's one verdict is its ``o1`` at ``<root's value>/opened``, a row a table, unit j reading row j // W;
    a fused block whose root and domain are left to round 1's history is refused, as the interior refuses it, and the
    verifier's copy of the value is `second_stage.VERDICT_ONE` at every row, which round 2 reads as the verifier's
    bytes of 1."""
    (v,) = TS.verdicts([_member()], _links())
    assert v == TS.Verdict(TID, SS.OPENING_VERDICT, f"{KEY}/root/opened", T, AT)
    assert IL.tied([_member()], _links())[0][0] == SUPPLIES[0]
    history = (*(replace(x, tables=None) for x in SUPPLIES), *_links()[len(SUPPLIES):])
    with pytest.raises(ValueError, match="a fused block's root and domain are left untied"):
        TS.verdicts([_member()], history)
    src = TS.verdict_sources((v,), Q)
    assert src == {v.value: [[{"bytes": "00000001"}]] * (Q * T)}
    assert SS.supplied_of(src) == ((v.value, (SS.VERDICT_ONE,) * (Q * T)),)


def test_the_verifier_reads_the_verdict_at_its_value_and_never_at_the_blocks_own_port():
    """`verifier_reads`: with the verdicts, ``o1`` reads the value the verifier fills with 1, at the root, leaves and
    domain the interior record gives it, unit j of sampled span s its row ``s·T + j // W``, as ``root`` and ``domain``
    read theirs; without them (the second stage before this change) ``o1`` reads the block's own committed port,
    which nothing the verifier fills holds to 1. A verdict whose value the interior does not register is refused by
    name, and never read at the block's own port."""
    d = _design()
    window = d.window(BOUNDARY, SAMPLE)
    m, = window.members
    got = TS.verifier_reads(d, window, _interior(d, window), {"roots": []}, SAMPLE)[m.key]["ports"]
    rows = [s * T + a for s in range(Q) for a in AT]
    o1 = got[SS.OPENING_VERDICT]
    assert (o1["value"], o1["root"], o1["leaves"], o1["domain"], o1["reads"]) == (f"{KEY}/root/opened", "0e" * 64,
                                                                                   Q * T, "d1" * 64, rows)
    assert all(got[p]["value"] == f"{KEY}/{p}" and got[p]["reads"] == rows for p in IR.TABLE)
    before = replace(d, verdicts=())
    own = TS.verifier_reads(before, window, _interior(before, window), {"roots": []}, SAMPLE)[m.key]["ports"]
    assert own[SS.OPENING_VERDICT]["value"] == window.held(m, SS.OPENING_VERDICT) != o1["value"]
    with pytest.raises(ValueError, match=f"'o1' reads {KEY}/root/opened, which the interior registration doesn't register"):
        TS.verifier_reads(d, window, _interior(d, window, verdicts=False), {"roots": []}, SAMPLE)


def _pinned_round_two(pins: dict, boundary: dict | None = None) -> SS.Own:
    """What round 2 holds of the fused block's read (`second_stage.Own`, as `two_stage.round_two` builds it): its key
    an index the verifier supplies, its root and domain the tables' entries it pinned (`interior_rows.sources`), its
    verdict ``o1`` the value it fills with 1 (`two_stage.verdict_sources`), and the block's other operands at round 1's
    ports, which the interior registers none of. The interior registers the read's values; the auditor's params record
    the template as a read of the tables' E rows and pin ``pins`` (`service.pinned_tables`); round 1's record registers
    ``boundary``'s tables, none by default."""
    key = IR.Supply(TID, "tok", f"{KEY}/tok", CELLS, tuple(range(CELLS)), index=tuple(range(CELLS)))
    read = (*SUPPLIES, key, *(IR.Operand(TID, p, f"{KEY}/{p}", CELLS, tuple(range(CELLS)), w)
                              for p, w in _member().rows if p in ("row", "salt", "path")))
    (v,) = TS.verdicts([_member()], read)
    supplied = {**IR.sources(read, Q, pins=pins), **TS.verdict_sources((v,), Q)}
    links = (*(SS.Link(TID, x.port, x.value) for x in read),
             *(SS.Link(TID, p, f"b.{p}") for p, _w in _member().rows if p in ("xq", "sx", "ws")))
    runs = tuple(SS.Run(j, W, t, 0) for j, t in zip(range(0, CELLS, W), range(T)))
    return SS.Own(SAMPLE, (), {"registered": [{"value": x.value} for x in read] + [{"value": v.value}]},
                  boundary or {"roots": []}, (), links, supplied=SS.supplied_of(supplied),
                  ref_reads=((TID, {"ports": {SS.OPENING_VERDICT: {"value": v.value}}}),),
                  rows=tuple(SS.Rows(TID, p, T, runs) for p in IR.TABLE), reads=((TID, E),),
                  tables=tuple((p, e) for p, e in sorted(pins.items())))


#: The tables' entries the auditor pins (`verity-vllm/expert-tables/v1`'s ``values``), each of E rows of H words.
PINS = {p: {"value": p, "root": f"{k + 1:02x}" * 64, "leaves": E, "words": H, "domain": f"{k + 0xd1:02x}" * 64}
        for k, p in enumerate(PARAMS)}


def test_round_two_takes_a_fused_blocks_tables_from_the_auditors_pins():
    """`second-inputs` at round 2: a fused block's row, salt and path pass where its root and domain are the tables the
    auditor's params pin, round 1's record registering none of them. They are refused, with `second_stage.INPUTS_WHY`
    and the row named, where the params pin no table or one of the two, where the pin's root or domain id is another
    than the one the verifier filled from the tables' registration, or where it pins the table at another count of
    rows; the same entries registered by round 1's record instead pass as before. The run is refused either way: no
    statement is given here, so ``second-statements`` or the design's own codes refuse it before the form's."""
    own = _pinned_round_two(PINS)
    assert SS.inputs_failure(own) is None
    first = SS.check(SS.FORM, own, None)
    assert "second-inputs" not in first.codes and not first.ok
    for changed in (replace(own, tables=()),
                    replace(own, tables=tuple((p, {**e, "root": "ee" * 64}) for p, e in own.tables)),
                    replace(own, tables=tuple((p, {**e, "domain": "ee" * 64}) for p, e in own.tables)),
                    replace(own, tables=tuple((p, {**e, "leaves": 2 * E}) for p, e in own.tables)),
                    replace(own, tables=own.tables[:1])):
        assert SS.inputs_failure(changed) == f"{KEY}/row"
        assert dict(SS._interior_failures(changed))["second-inputs"] == {"value": f"{KEY}/row", "why": SS.INPUTS_WHY}
        assert "second-inputs" in SS.check(SS.FORM, changed, None).codes
    recorded = {"roots": [{"port": f"t{k}", **{x: e[x] for x in ("root", "domain", "leaves")}}
                          for k, e in enumerate(PINS.values())]}
    assert SS.inputs_failure(replace(own, boundary=recorded, tables=())) is None
