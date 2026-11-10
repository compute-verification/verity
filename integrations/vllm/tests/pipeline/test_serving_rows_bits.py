"""A Boolean row template with no word Definition (`serving_templates.template`'s ``bits`` layout): DSA's long row's units,
the mask's cut and the index, whose ports are bits that are not whole u16 words (a digit prefix of 8, a rank of 12, a mask of
a block's N).

* Every unit of the long row's check template (`long_row.units`) but its table read is a ``bits`` template, at the served
  statics too: `chunk_rows.unit_rows` gives each parameter one row of its bits zero-padded to whole u16 words and each
  return field one, `chunk_rows.statement` keeps its reads inline, and the rows the verifier supplies (a chunk's ``c``, a
  slot's ``j``) are in `chunk_rows.INDEX_SCHEMA`. `interface.layout` still refuses such a port.
* At small statics each unit but the pick is staged by `UnitStager` on those rows in the form the sampled audit's knowledge
  soundness covers (`one_stage.proofs.ks_form_why`), inside Flock's range (k_log, and a region a row against
  `REGIONS_MAX`), on random rows; a row whose padding is not zero is refused (slow).
* ``VERITY_VLLM_LONG_ROW_UNITS=1``: the same at statics whose blocks hold more than one key a digit, with the pick, and
  where `flock-verify` is built its ``ks-form`` on each staged statement's files, which holds the unit's rows to their port
  bits (`ProvedScope.scopeOkH`), as the mirror does not.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from verity.core.protocols.spot.one_stage import proofs as PF
from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import bind
from verity.ml.commitments.gates.embedding import read
from verity.ml.windows import interface as IF
from verity_flock import circuit as FC
from verity_flock.class_statement import UnitStager
from verity_vllm.pipeline import chunk_rows as CR
from verity_vllm.pipeline import long_row as LR
from verity_vllm.pipeline import serving_templates as ST
from verity_vllm.query.program_view import spec_base

#: `Flock.Statement.checkInRange`'s bound on a statement's regions; a unit's are one a row (`Flock.HmOut.regions`)
REGIONS_MAX = 1024
BIND = {"partition": "0" * 128, "program": "0" * 128}
VERIFY = Path(__file__).resolve().parents[4] / "verity" / "core" / "service" / "flock" / ".lake" / "build" / "bin" / "flock-verify"


def _templates(t) -> dict[str, object]:
    """The long row ``t``'s templates but its read, by id, in body order."""
    return {descriptor_id(d): d for _k, d in LR.units(t) if spec_base(descriptor_id(d)) != "EmbeddingRead_v1"}


def _cut(N: int, TOPK: int, BLOCK: int):
    return bind(LR.DsaTopKReadRowsCut, N=N, TOPK=TOPK, BLOCK=BLOCK, READ=read(max(16, N), 1))


def _random_rows(tp, in_rows, n: int, seed: int) -> dict[str, np.ndarray]:
    """``n`` units' rows: each parameter's bits random, low first, then zeros to its row's whole words."""
    rng, rows = np.random.default_rng(seed), {}
    for (r, _s, b, _ps), (_nm, ty) in zip(in_rows, tp.fn.signature()[0], strict=True):
        bits = np.zeros((n, b), np.uint8)
        bits[:, :ty.leaves] = rng.integers(0, 2, (n, ty.leaves))
        rows[r] = np.packbits(bits, axis=1, bitorder="little").view("<u2")
    return rows


def _staged(tid: str, d: Path, lean: bool = False) -> dict:
    """Template ``tid`` staged on its `chunk_rows` rows as a member's units are, two units of random rows: what the tests hold
    it to."""
    tp = ST.template(tid)
    in_rows, out_rows = CR.unit_rows(tp)
    st = UnitStager(tp.fn, f"unit/{tid}", in_rows, out_rows, circuit=d / "circuit.txt", bind=BIND, **CR.statement(tp))
    rows = _random_rows(tp, in_rows, 2, 11)
    st.stage(rows, d, b"k" * 32, indices=[0, 1])
    got = {"why": PF.ks_form_why(d / "circuit.txt", d / "pub.bin"), "k_log": st.meta["k_log"], "regions": len(st.ports)}
    if lean:
        p = subprocess.run([str(VERIFY), "ks-form", "--statement", PF.STATEMENT, "--circuit", str(d / "circuit.txt"), "--public",
                            str(d / "pub.bin")], capture_output=True, text=True, check=False)
        got["lean"] = [ln for ln in p.stdout.splitlines() if ln.startswith("KS-FORM ")] or [p.stderr[-300:]]
    return {**got, "stager": st, "rows": rows, "in_rows": in_rows}


@pytest.mark.parametrize("t", [LR.served(), _cut(64, 32, 32)], ids=["served", "small"])
def test_every_unit_of_the_long_row_but_its_read_is_a_bits_template_on_rows_of_whole_words(t) -> None:
    padded = 0
    for tid, d in _templates(t).items():
        tp = ST.template(tid)
        assert (tp.layout, tp.packed, tp.fn.id) == ("bits", None, d.id), tid
        assert CR.statement(tp) == {"reads": "inline", "pad": True}, tid
        params, ret = d.signature()
        in_rows, out_rows = CR.unit_rows(tp)
        index = CR.INDEX_ROWS.get(spec_base(tid))
        assert in_rows == [(f"in{g}", CR.INDEX_SCHEMA if nm == index else IF.V2, 16 * -(-ty.leaves // 16), [nm])
                           for g, (nm, ty) in enumerate(params)], tid
        assert out_rows == [(IF.V2, (j,)) for j in range(len(getattr(ret, "fields", None) or (ret,)))], tid
        padded += any(ty.leaves % 16 for _nm, ty in params)
    assert padded and {spec_base(x) for x in _templates(t)} == {
        "DsaTopKDigitCount_v1", "DsaTopKDigitPick_v1", "DsaTopKMaskBlock_v1", "DsaMaskPositions_v1", "DsaPositionsMerge_v1",
        "DsaPositionsChunk_v1", "DsaPositionsSlot_v1"}
    assert ST.template(descriptor_id(t.bindings["READ"])).layout == "read"
    assert [nm for tid in _templates(t) for nm, _ty in ST.template(tid).fn.signature()[0]
            if nm == CR.INDEX_ROWS.get(spec_base(tid))] == ["c", "j"]


def test_the_window_builders_layout_still_refuses_a_port_that_is_not_whole_words() -> None:
    count = next(d for tid, d in _templates(_cut(64, 32, 32)).items() if tid.startswith("DsaTopKDigitCount_v1") and "K=1" in tid)
    assert dict(count.signature()[0])["prefix"].leaves == 8
    with pytest.raises(ValueError):
        IF.layout(count)


@pytest.mark.slow
def test_the_small_units_but_the_pick_stage_padded_in_the_proved_form(tmp_path) -> None:
    """Each unit of a small long row but the pick (whose circuit is 17 GB to build at any statics) staged on its padded rows:
    the stager's reference check passes on random units, which holds the lowering to reading each parameter at its row's first
    bit; the file is in the proved form and range; and a one in a row's padding is refused. The cheapest test that builds
    these units' statements."""
    seen = set()
    for k, tid in enumerate(t for t in _templates(_cut(64, 32, 32)) if spec_base(t) != "DsaTopKDigitPick_v1"):
        fam = (spec_base(tid), tuple(b for _r, _s, b, _ps in CR.unit_rows(ST.template(tid))[0]))
        if fam in seen:
            continue
        seen.add(fam)
        got = _staged(tid, tmp_path / str(k))
        assert got["why"] is None and got["k_log"] <= FC.K_MAX and got["regions"] <= REGIONS_MAX, (tid, got)
        tp = ST.template(tid)
        odd = next((r for (r, _s, _b, _ps), (_nm, ty) in zip(got["in_rows"], tp.fn.signature()[0], strict=True) if ty.leaves % 16),
                   None)
        if odd is not None:
            bad = {**got["rows"], odd: got["rows"][odd] | np.uint16(1 << 15)}
            with pytest.raises(ValueError, match=rf"row {odd}'s bits past its parameters' \d+ are not zero$"):
                got["stager"].stage(bad, tmp_path / f"{k}-bad", b"k" * 32, indices=[0, 1])
    assert {f for f, _b in seen} == {"DsaTopKDigitCount_v1", "DsaTopKMaskBlock_v1", "DsaMaskPositions_v1",
                                     "DsaPositionsMerge_v1", "DsaPositionsChunk_v1", "DsaPositionsSlot_v1"}


@pytest.mark.skipif(not os.environ.get("VERITY_VLLM_LONG_ROW_UNITS"),
                    reason="VERITY_VLLM_LONG_ROW_UNITS=1: stages every unit of a long row of 1,024 keys, about 15 minutes and "
                           "17 GB (the pick's circuit); run it as a recorded run")
def test_every_unit_of_a_long_row_stages_padded_in_the_proved_form(tmp_path) -> None:
    """Every template of the long row at N = 1,024, TOPK = 64 and blocks of 256 but its read, the pick included: staged as
    above, and held to Lean's ``ks-form`` where `flock-verify` is built."""
    lean = VERIFY.is_file()
    for k, tid in enumerate(_templates(_cut(1024, 64, 256))):
        got = _staged(tid, tmp_path / str(k), lean)
        assert got["why"] is None and got["k_log"] <= FC.K_MAX and got["regions"] <= REGIONS_MAX, (tid, got)
        assert not lean or got["lean"] == ["KS-FORM ok"], (tid, got["lean"])
