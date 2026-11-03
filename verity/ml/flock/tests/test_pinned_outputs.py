"""Pinned outputs (META ``pinned_outputs``, form B): the last output ports of a statement of bit rows have no row,
and Δ copies the unit's forced-zero row onto each of their bits, so a proof holds only where they are zero. A writer that
reads a shared row ``S`` and pins ``d = S[j] XOR y`` commits no output row. The Rust prover's parse and Δ are its own
tests (`circuit::tests::pinned_outputs_are_copies_of_the_forced_zero_row`); these hold the composer to the same form, for
any pinned width and with or without an output row before it."""
import hashlib
import json
import os
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

#: the live server's refusal of the form (`proved_scope::PINNED_WHY`)
PINNED_WHY = "pinned outputs (META pinned_outputs): a form no soundness theorem covers yet"


def pinned_toy(pinned_words: int, rowed: bool = False, flip: bool = False):
    """A unit reading one 1024-bit `sha512/row/v2` input row ``p0`` (64 words) that pins ``d`` of ``pinned_words`` words,
    word ``o`` the XOR of input words ``o`` and ``o + pinned_words`` (``flip``: and the constant one, so the all-zero
    instance's ``d`` is not zero); ``rowed``: before it, an output port ``a0`` of one word (input word 0) with its own row.
    Returns the lowering, the subcircuit and the input set."""
    from verity.ml.circuits.types import Array, Tuple, Value
    from verity_flock import class_statement as CS
    from verity_flock import gf2
    from verity_flock import ir_lower as IL

    assert 2 * pinned_words <= 64
    C = gf2.Circuit()
    ins = [C.input(f"in{j}", 16) for j in range(64)]
    names = (["a0"] if rowed else []) + ["d"]
    outs = []
    if rowed:
        C.output("out0", list(ins[0]))
        outs.append("out0")
    for o in range(pinned_words):
        C.output(f"out{len(outs)}", [ins[o][t] ^ ins[o + pinned_words][t] ^ (gf2.ONE if flip else gf2.ZERO) for t in range(16)])
        outs.append(f"out{len(outs)}")
    widths = [1] * rowed + [pinned_words]
    units = IL.Units(target=None, in_widths=[16] * 64, out_widths=[16] * len(outs), in_src=[[("leaf", p) for p in range(64)]],
                     out_src=[[("ret", o) for o in range(len(outs))]], signature=("toy",), gates=[], arg_gates=[], ret_gates=[])
    sig = ((("p0", Array(64, Value(16))),), Tuple(tuple(Array(w, Value(16)) for w in widths)))
    definition = SimpleNamespace(id="unit-class/pinned-writer", signature=lambda: sig)
    low = IL.Lowering("unit-class/pinned-writer", definition, units, C, IL.layout(C, [f"in{j}" for j in range(64)], outs))
    CS.cache_slot_circuits()
    port = lambda nm: SimpleNamespace(name=nm, dtype="u16")  # noqa: E731
    content = json.dumps([pinned_words, rowed, flip]).encode()
    sub = SimpleNamespace(id="unit-class/pinned-writer", template=SimpleNamespace(id="unit-class"), inputs=[port("p0")],
                          outputs=[port(nm) for nm in names],
                          program_digests=lambda statics: {"unit-class/pinned-writer": hashlib.sha256(content).hexdigest()})
    iset = SimpleNamespace(subcircuit=sub, name="unit-class-pinned-writer", content_digest=hashlib.sha256(content).hexdigest())
    return low, sub, iset


def _meta(text: str) -> dict:
    return json.loads(text.split("\n", 2)[1][5:])


@pytest.mark.parametrize("pinned_words,rowed", [(1, False), (8, False), (32, False), (16, True)])
def test_a_pinned_output_has_no_row_and_meta_names_it(pinned_words, rowed, tmp_path):
    """META names the pinned port, it has no row (no port, no ``rows`` entry, no ``b || c``), the output rows before it
    keep theirs, and the all-zero instance's pinned words are zero. Staging an instance whose ``d`` is zero writes one
    ``b || c`` per row per instance."""
    from verity_flock import circuit as FC

    low, sub, iset = pinned_toy(pinned_words, rowed)
    text = FC.compose(low, sub, port_bits=[1024], pinned_outputs=("d",))
    meta = _meta(text)
    assert meta["pinned_outputs"] == ["d"]
    assert [p[0] for p in meta["ports"]] == ["p0"] + ["a0"] * rowed
    assert [p[0] for p in meta["out_ports"]] == ["a0"] * rowed + ["d"]
    assert "out_row_ports" not in meta
    assert len(meta["rows"]) == len(meta["dummy"]["bc"]) == 1 + rowed
    assert meta["dummy"]["outputs"][rowed:] == [0] * pinned_words
    n = 4
    s = np.random.default_rng(1).integers(0, 1 << 16, (n, 64), dtype=np.uint64).astype(np.uint16)
    s[:, pinned_words:2 * pinned_words] = s[:, :pinned_words]
    got, _cut, ok = low.evaluate(s.astype(np.uint64))
    assert int(ok) == (1 << n) - 1 and not got[:, rowed:].any()
    outs = {"d": got[:, rowed:].astype(np.uint16), **({"a0": got[:, :1].astype(np.uint16)} if rowed else {})}
    FC.write(text, iset, 0, n, tmp_path / "inst.bin", tmp_path / "pub.bin", b"k" * 32, rows={"p0": s}, outs=outs)
    head, body = (tmp_path / "pub.bin").read_bytes().split(b"\n", 1)
    assert sorted(json.loads(head)["frame_v3"]["roots"]) == sorted(["p0"] + ["a0"] * rowed)
    assert len(body) == 128 * n * (1 + rowed)


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("FLOCK_CIRCUIT"), reason="FLOCK_CIRCUIT names no flock-circuit selftest build")
@pytest.mark.parametrize("pinned_words,rowed", [(1, False), (16, True)])
def test_the_prover_proves_a_zero_pinned_output_and_no_nonzero_one(pinned_words, rowed, tmp_path):
    """`flock-circuit selftest --only honest` (a selftest build, ``FLOCK_CIRCUIT``) refuses the form (`ProvedScope`'s
    `pinnedWhy`, until a soundness theorem covers it), and under the refusal both replays verify a session of instances
    whose pinned words are zero, with no output row or with one before them, and neither verifies one whose pinned words
    are not: Δ ties each pinned bit to the forced-zero row, so the block's R1CS has no witness there. What it catches that
    the parse test can't: the prover's witness and Δ agree on the pinned columns, and a nonzero pinned word reaches no
    verified proof."""
    from verity_flock import circuit as FC

    low, sub, iset = pinned_toy(pinned_words, rowed)
    text = FC.compose(low, sub, port_bits=[1024], pinned_outputs=("d",))
    verdicts = {}
    for zero in (True, False):
        d = tmp_path / ("zero" if zero else "nonzero")
        d.mkdir()
        n = 4
        s = np.random.default_rng(2).integers(0, 1 << 16, (n, 64), dtype=np.uint64).astype(np.uint16)
        if zero:
            s[:, pinned_words:2 * pinned_words] = s[:, :pinned_words]
        got, _cut, _ok = low.evaluate(s.astype(np.uint64))
        assert got[:, rowed:].any() != zero
        outs = {"d": got[:, rowed:].astype(np.uint16), **({"a0": got[:, :1].astype(np.uint16)} if rowed else {})}
        (d / "circuit.txt").write_text(text)
        FC.write(text, iset, 0, n, d / "inst.bin", d / "pub.bin", b"k" * 32, rows={"p0": s}, outs=outs)
        r = subprocess.run([os.environ["FLOCK_CIRCUIT"], "selftest", "--only", "honest", "--instances", str(d / "inst.bin"),
                            "--circuit", str(d / "circuit.txt"), "--tables", str(d / "tables"), "--forged", str(d / "none.txt")],
                           capture_output=True, text=True, check=False)
        neg = [json.loads(ln.split("\t", 1)[1]) for ln in r.stdout.splitlines() if ln.startswith("NEG\t")]
        (tmp_path / f"{d.name}.log").write_text(r.stdout + r.stderr)
        assert len(neg) == 1 and PINNED_WHY in (neg[0]["refusal"] or "") and not neg[0]["accepted"], r.stdout + r.stderr
        verdicts[zero] = [ok for _stream, ok, _why in neg[0]["reps_ok"]]
    assert verdicts == {True: [True, True], False: [False, False]}, verdicts


def test_pinned_outputs_are_refused_outside_their_form():
    """The composer refuses pinned outputs that are not the last output ports, a statement of `sha512/row/v1` rows (no
    ``port_bits``), and a unit whose all-zero instance's pinned words are not zero (it pads every block)."""
    from verity_flock import circuit as FC

    low, sub, _ = pinned_toy(2, rowed=True)
    with pytest.raises(ValueError, match="the last output ports"):
        FC.compose(low, sub, port_bits=[1024], pinned_outputs=("a0",))
    with pytest.raises(ValueError, match="the last output ports"):
        FC.compose(low, sub, pinned_outputs=("d",))
    low, sub, _ = pinned_toy(2, flip=True)
    with pytest.raises(ValueError, match="pinned outputs are not zero"):
        FC.compose(low, sub, port_bits=[1024], pinned_outputs=("d",))


def test_out_rows_writes_no_row_past_the_runs():
    """`circuit.out_rows` writes each output row's run and nothing for the output ports past them: none at all for a
    statement whose only outputs are pinned."""
    from verity_flock import circuit as FC

    assert FC.out_rows({"d": np.zeros((2, 3))}, [("d", 3)], [], []) == {}
    rows = FC.out_rows({"a": np.ones((2, 1)), "d": np.zeros((2, 3))}, [("a", 1), ("d", 3)], [FC.out_row("a", 1, bit_row=True)], None)
    assert list(rows) == ["a"] and rows["a"].shape == (2, 64)
