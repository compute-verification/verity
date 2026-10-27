"""`circuit.h1device`: H1's device records. `h1window.h1_tile` without ``arch`` is the sm_120 tile it was before records
existed (template and Program digests recorded on the tree before), sm_90's binds the same step, so its tiles are sm_120's
id for id, and sm_90's domain is h1u's; each record's sheet numbers are its data sheet's."""
from __future__ import annotations

import hashlib
import re
import tomllib
from pathlib import Path

import pytest

from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.silicon.instructions import INSTRUCTIONS

from verity.core.protocols.pouw.circuit import h1device as D
from verity.core.protocols.pouw.circuit import h1tile as H
from verity.core.protocols.pouw.circuit import h1window as HW
from verity.core.protocols.pouw.circuit import h1zy as Z
from verity.core.protocols.sampled_proofs.one_stage import partition as P

ROOT = Path(__file__).resolve().parents[5]
SHEET_90 = ROOT / "benchmarks" / "pouw" / "machine_model" / "data" / "devices" / "sm_90" / "sheet.toml"
SM120_LEAN = ROOT / "Security" / "Definitions" / "Accounting" / "Headline" / "Sm120.lean"

#: sha512 of the template descriptor of `HW.template(K, HW.tile_unit(K, (32, 32), definition))`, recorded before `h1_tile`
#: took an architecture: (PouwH1Tile v1's, v3's) at each K
TEMPLATE_IDS = {
    512: ("1ed0153a7c1d7baec3cd0ddf42baae4a33c1ba6682b9d509e3aeed1e8dfabcb9001b07995db26f72cca8c6c77e261693080733e0c9a2ca0de7f551c6da573569",
          "41404a95e38c368a700b248ee586e5de7eb17163c774d994df5862a8b97d1798bc0c7dddc460d6aacb1e1d3c7a549d1bda9b4bdc5ccc585b24ec7f1a0ecbf40f"),
    1536: ("fc961dbac7ae6f1206152c254372057d8274d747b5cde12f5e17e8e1d9bfd704638cb232766b38471521d518f509fa172f3623b0a8fb73f955dcf04352a17df4",
           "f968603b7b6ad5bd4f481f245c364bd6a8c76fe500839d24d26623529d935c507aad900eeceb4799021ba8f0c5b7ddc6bc35a880696197c77f06c593d679b3ac"),
    2048: ("c72529b63f6f9d7f5c0a527f4b1ab21d74f71c85ddb064ff33410187118d42a9e295b911c325e7a78d191778a1cd6674ae4377d5a789628840e81a3f21888d16",
           "5b6d3850b2401ed65a009550a4e993b9d359baaa9f1060877b93e24810c4ca937e546c42d73499e4e89543d160dc5e045b42f00c716356985b8a956121b016b5"),
    6144: ("b27a2770a32937fe8462b596151aefbecf193c07c7a9c9c06b0d9ed1f943bc69e7a48c3cfe07b27562a7f583b78790cef09c4f3a31edd63e3fa2ed1a2e5692af",
           "fb95db196ed9c54304c5a9acaf65cfa244252c9964fe6c7e0f93d6ca60aa5488d65664abd958baee233bb118b7969c275d0ec3cc9478a30762311d45e299a0e9"),
}

#: `P.program_digest(HW.program([("q", 64, 8)], [("q", 0)], 4, (4, 8)))`: a change to the window's layout moves it
PROGRAM_64 = ("e009cbeb20775cd21cb183eab99a34432e130840d90a66b8110a34a776c05669d9043c3bdfbe78cc21570f8a9b8a27c925263e4e5cf8a544f11ff8d9053b0c17")


def _template_id(k: int, definition, **arch) -> str:
    return hashlib.sha512(descriptor_id(HW.template(k, HW.tile_unit(k, (32, 32), definition, **arch))).encode()).hexdigest()


@pytest.mark.parametrize("k", TEMPLATE_IDS)
def test_the_default_and_sm120_and_sm90_tiles_are_the_recorded_ones(k) -> None:
    """Without ``arch``, with ``arch="sm_120"`` and with ``arch="sm_90"``, each K's template is the one recorded before."""
    for definition, want in zip((None, H.PouwH1Tile3), TEMPLATE_IDS[k]):
        for arch in ({}, {"arch": "sm_120"}, {"arch": "sm_90"}):
            assert _template_id(k, definition, **arch) == want, (k, definition, arch)


@pytest.mark.slow
def test_the_default_program_digest_is_the_recorded_one() -> None:
    """A whole window's Program (the real tile at K = 64), as recorded before records existed. It builds the
    Program end to end (about 40 s), which the template ids above don't: a change in how `program` reaches `tile_unit`
    would show only here."""
    assert P.program_digest(HW.program([("q", 64, 8)], [("q", 0)], 4, (4, 8))) == PROGRAM_64


def test_sm90s_domain_is_h1us_and_the_combines() -> None:
    """sm_90 serves K >= `h1zy.LEAST_K` (384) with 512 | K, h1u's own refusal (``if (K % 512) return 2``), so its least K is
    512; sm_120 keeps the Definitions' own refusals."""
    h1u = (ROOT / "benchmarks" / "hash" / "dither" / "h1u.cu").read_text()
    assert "if (K % 512) return 2;" in h1u
    assert Z.LEAST_K == 384 and D.SM90.domain == (384, 512)
    for k in (64, 256, 384, 640, 1000, 1024 + 32):
        with pytest.raises(ValueError, match=re.escape(f"K={k} is outside H1's domain on sm_90 (H200 SXM (sm_90a)): "
                                                       f"K >= 384 (h1zy.LEAST_K")):
            HW.h1_tile(k, (32, 32), arch="sm_90")
    for k in (512, 1536, 2048, 6144, 14336, (1 << 17) - 512):
        assert D.SM90.refuse(k) is None
    assert HW.h1_tile(14336, (32, 32), arch="sm_90").id == HW.h1_tile(14336, (32, 32)).id
    assert D.SM120.refuse(64) is None and HW.h1_tile(64, (4, 8), arch="sm_120").complete


def test_an_architecture_without_a_record_is_refused() -> None:
    for arch in ("sm_100", "sm_89", "", None):
        with pytest.raises(ValueError, match=re.escape(f"H1 has no device record for {arch!r}: its records are sm_120, sm_90")):
            HW.h1_tile(512, (32, 32), arch=arch)


def test_each_record_binds_its_pinned_step() -> None:
    """Each record's instruction is a registered one whose model is the record's step, on the record's architecture, and its
    binding is sm_120's step on bits."""
    for dev in D.DEVICES.values():
        ins = INSTRUCTIONS[dev.instruction]
        assert (ins.sm, ins.model, ins.status) == (dev.arch, dev.step, "pinned")
        assert dev.gemm == BG.GEMM_SM120_S8
    assert D.SM90.gemm == BG.GEMM_SM90_S8


def test_the_sm90_record_is_its_data_sheet() -> None:
    """SMs, the tensor clock, f_max and the dense s8 MACs are DS-90's, and 2 x MACs x SMs x clock is half the H200 data sheet's
    sparse INT8 3,958 TOPS to within its tolerance."""
    sheet = tomllib.loads(SHEET_90.read_text())
    s8 = sheet["peaks"]["s8"]
    assert (D.SM90.gpu, D.SM90.sms, D.SM90.fmax_mhz) == (sheet["gpu"], sheet["sms"]["value"], sheet["fmax"]["mhz"])
    assert (D.SM90.tensor_mhz, D.SM90.s8_macs) == (sheet["clocks"][s8["clock"]]["mhz"], s8["macs"])
    assert s8["sparse"] and D.SM90.sheet_s8_tops == s8["sheet"] / 2
    assert abs(D.SM90.s8_dense_tops - s8["sheet"] / 2) <= s8["tol"]


def test_the_sm120_record_is_ds120() -> None:
    """f_max and the IMMA16832 MACs are `Accounting.TS.Headline.Sm120.ds120`'s."""
    lean = SM120_LEAN.read_text()
    assert re.search(r"def ds120 : SheetData Pipe where\n  fmax := 2430\n", lean)
    assert (D.SM120.fmax_mhz, D.SM120.tensor_mhz, D.SM120.s8_macs, D.SM120.sheet_s8_tops) == (2430, 2430, 1024, None)
