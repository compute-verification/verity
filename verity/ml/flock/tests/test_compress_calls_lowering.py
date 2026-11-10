"""C-Flock lowers the hashing Definitions traced with calls (each SHA-512 compression a call of
`commitments.gates.sha512.Sha512CompressFolded`, which has no builder) to the circuit it lowered them to before: an activation
of a part is lowered by its builder (`ir_lower._boolean_activation`), never through the calls. Each pin is the flock-ir-unit/v2
netlist's sha256 and useful rows that `ir_lower.whole` gave with every compression's gates written out in the parts' bodies:
a part alone, the reads that hold the parts, and the served segment and outer hash."""
from __future__ import annotations

import hashlib

import pytest

from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.boolean.forms import const_bits
from verity.ml.circuits.defs import bind
from verity.ml.commitments.gates import embedding as E, frame_v3 as F, hm96 as H, rowleaf as R
from verity.ml.commitments.gates.sha512 import IV, Sha512CompressFolded
from verity_flock.ir_lower import _sha256, layout, netlist_pieces, whole

ROW = "hm96-sha512/row/v2"
#: target -> (useful rows, netlist sha256) with the compressions' gates written out
PINS = {
    "Sha512RowSeg{N_BITS=13}": (154625, "1059ce504fd5ccfe9da2ce9095313e53f337e10eb9127979ea7391116261ef01"),
    "Hm96Sha512Leaf": (307201, "d16dac80db93fab68495b9d6f41f942d53fed7aea901ea3877d2a7796f7631c1"),
    "FrameV3Read{D=1}": (394753, "2710714a668b642b8fd6beb085183c8b70c42b51de5ac3ca96719f8b0b07f231"),
    "FrameV3ReadAt{D=1,B=4}": (397313, "0011bc97eecb13fe4884d54767f08543e0208ea0adb3b630195c04bcc697afc0"),
    "EmbeddingRead{V=2,H=1}": (851969, "2b1b419cef11459b2c091595109eaa73a8f7ad55b6035280d62a56c52dc77cfa"),
    "Segment{M=8192}": (730113, "8ce5880e4f8de5276a033cef3dfe48b63f9b5b21892cdd75ee5ec886b960a31c"),
    "Outer{N_BITS=49152}": (309761, "cb8f2b0ee00e43050e0c0ba7f81d6736d4564bf022f14a4ee9d8b0e518c2b839"),
}


def _target(name: str):
    return {
        "Sha512RowSeg{N_BITS=13}": lambda: R.digest(13),
        "Hm96Sha512Leaf": lambda: bind(H.Hm96Sha512Leaf),
        "FrameV3Read{D=1}": lambda: F.read(1, ROW),
        "FrameV3ReadAt{D=1,B=4}": lambda: F.read_at(1, 4, ROW),
        "EmbeddingRead{V=2,H=1}": lambda: E.read(2, 1),
        "Segment{M=8192}": lambda: bind(R.Sha512RowSegSegment, M=8192),
        "Outer{N_BITS=49152}": lambda: bind(R.Sha512RowSegOuter, N_BITS=49152),
    }[name]()


@pytest.mark.slow
@pytest.mark.parametrize("name", PINS)
def test_a_call_trace_lowers_to_the_whole_circuit(name: str) -> None:
    """10 to 100 s and up to 7 GB a target (the lowering): what holds the lowered circuit, the one a proof is of, to the one
    before calls; the commitments suite's tests hold only the Definitions' own gates."""
    U = whole(_target(name))
    lay = layout(U.circuit(), [f"in{j}" for j in range(len(U.in_widths))], [f"out{j}" for j in range(len(U.out_widths))])
    h = hashlib.sha256()
    for piece in netlist_pieces(lay, "unit"):
        h.update(_sha256(piece).digest())
    assert (lay.useful, h.hexdigest()) == PINS[name]


def test_a_folded_definition_alone_is_lowered_by_its_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    """A compression no part holds (circuit-check's binding) is lowered by its own builder at its pattern (``fold_build``), not
    gate by gate (15 minutes of linear forms growing round over round)."""
    iv = ["o" if b.c else "z" for v in IV for b in const_bits(v, 64)]
    P = TR.encode_pattern(iv + [((i,), 0) for i in range(1024)])
    built, fold_build = [], Sha512CompressFolded.fold_build
    monkeypatch.setattr(Sha512CompressFolded, "fold_build", lambda S, C, x: built.append(S.P) or fold_build(S, C, x))
    whole(bind(Sha512CompressFolded, P=P)).circuit()
    assert built == [P]
