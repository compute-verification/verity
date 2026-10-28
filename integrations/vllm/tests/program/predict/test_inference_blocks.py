"""verity_vllm.program.predict.inference_blocks: B's partition of a window circuit into inference blocks, one a slot, as a
private table under a hiding root with its budgets and padded counts public. Held, on `test_window_circuit`'s TINY window
circuit (its class, windows and grain), to the circuit unrolled unit by unit; with ``VERITY_VLLM_FA3_LLAMA_1B=1``, on int8
Llama-3.2-1B's window circuit on the H200 at 4,32,4 too.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest
from tests.program.predict import test_steps as S
from tests.program.predict.test_window_circuit import WINDOW, _circuit, _labels, held
from tests.program.predict.test_window_circuit_fa3 import H200, LLAMA_1B
from verity_vllm.program.model import INT8_EXACT_QUANT
from verity_vllm.program.predict import inference_blocks as IB
from verity_vllm.program.predict.scan_state import SCAN_STATE
from verity_vllm.program.predict.window_builder import WEIGHTS
from verity_vllm.program.predict.window_circuit import WindowClass, class_circuit

from verity.ml.circuits import partition_v2 as PV2
from verity.ml.commitments import advice as AC
from verity.ml.windows import interface as IF

INT8_H200 = {**H200, "int8_exact": "linears"}
#: the Program digest the header names, as the record gives it (`inference_blocks.program_digest`, about a minute at TINY)
PROGRAM = "ab" * 64


def _registered(config: dict = S.TINY):
    phi = S.Phi(model={"repo": "org/m", "revision": "r", "hf_config": {**config, "quantization_config": dict(INT8_EXACT_QUANT)}},
                target=INT8_H200)
    return S.Registered.of(phi, default_seed=7)


@pytest.fixture(scope="module")
def window():
    return _circuit(WINDOW), PROGRAM


def _slots(circuit, rs) -> dict[str, np.ndarray]:
    """Each template's units' blocks, by descriptor id, from the table alone: each entry's units' slot digits."""
    tid = {PV2.definition_digest(circuit.template(t)): t for t in circuit.ids}
    out = {t: np.full(n, -1, dtype=np.int64) for t, n in circuit.units.items()}
    for digest, at, radix, d in IB.decode(rs)[2]:
        i = np.arange(math.prod(radix), dtype=np.int64)
        out[tid[digest]][at:at + len(i)] = i // math.prod(radix[d + 1:]) % radix[d]
    return out


def _unrolled(circuit, slots: dict[str, np.ndarray]) -> tuple[list[int], list[int]]:
    """Each block's gates and fresh-input bits, unit by unit: a template's gates for each of its units in the block, and
    the rows its units read from registered parameters outside θ (the scan state), every wire within one block and every
    other input row wired or θ's."""
    m = circuit.blocks.slots
    for w in circuit.wires:
        i = np.arange(w.units, dtype=np.int64)
        assert (slots[w.reader][w.at + i // w.cut[1]] == slots[w.writer][w.source(i)]).all(), w.doc()
    gates = sum(np.bincount(slots[t], minlength=m) * circuit.template(t).gates for t in circuit.ids)
    lays, fresh = circuit.layouts, np.zeros(m, dtype=np.int64)
    for r in circuit.registered.reads:
        if r.param.split("/", 1)[0] != WEIGHTS and r.port in dict(lays[r.reader][0].rows):
            bits = 16 * dict(lays[r.reader][0].rows)[r.port] // int(circuit.cuts.get(r.reader, {}).get(r.port, 1))
            fresh += np.bincount(slots[r.reader][r.at + np.arange(r.units) // r.cut[1]], minlength=m) * bits
    assert not circuit.ingress and all(t.param.split("/", 1)[0] == WEIGHTS for t in circuit.registered.tables)
    assert IF.input_count(json.loads(json.dumps(circuit.description())), **held(circuit))["free"] == 0
    return [int(g) for g in gates], [int(f) for f in fresh]


def test_the_table_is_the_windows_entries_one_block_a_slot(window):
    """The table is a header (the Program it is given, its 2 blocks, its entries) and one row an inference entry, never one a
    unit: each entry's template, first unit, radix and which digit is the slot. Read back alone it puts every unit in its
    slot's block, as the circuit's entries do."""
    circuit, program = window
    rs = IB.rows(circuit, program)
    assert IB.decode(rs)[:2] == (program, 2)
    entries = IB.decode(rs)[2]
    assert len(rs) == 1 + len(entries)
    assert entries == [(PV2.definition_digest(circuit.template(e.template)), e.at, e.radix, e.axes.index("slot"))
                       for e in circuit.blocks.inference]
    assert all(len(r) == IB.ROW_BYTES for r in rs)
    labels, slots = _labels(circuit), _slots(circuit, rs)
    assert all((slots[t] == labels[t][:, 0]).all() for t in circuit.ids)


def test_every_block_has_c_gates_and_k_fresh_input_bits(window):
    """Every block has exactly c gates and takes k fresh-input bits, the circuit unrolled unit by unit agreeing: no wire
    crosses blocks, and a block's fresh input is its slot's lane of the scan state (4,548 words), the weights θ."""
    circuit, program = window
    c, k = IB.budgets(circuit)
    gates, fresh = _unrolled(circuit, _slots(circuit, IB.rows(circuit, program)))
    assert gates == [c, c] and fresh == [k, k] and k == 4548 * 16
    assert {r.param for r in circuit.registered.reads if r.param.split("/", 1)[0] != WEIGHTS} == {SCAN_STATE}


def test_the_statement_is_the_budgets_and_padded_counts_under_a_hiding_root(window):
    """The statement is the query, c and k, the block and row counts on the grid and the root: no entry, template or unit
    of the table. Each row opens under the root at the padded count, the padding rows zero; the same rows under fresh salts
    have another root, and under the same salts the same one."""
    circuit, program = window
    st, reg = IB.commit(circuit, program)
    rs = IB.rows(circuit, program)
    assert st == {"format": IB.FORMAT, "query": IB.QUERY, "c": IB.budgets(circuit)[0], "k": 4548 * 16, "blocks": 2,
                  "rows": PV2.grid(len(rs)), "root": reg.root.hex()}
    assert list(reg.rows[:len(rs)]) == rs and set(reg.rows[len(rs):]) <= {bytes(IB.ROW_BYTES)}
    root = bytes.fromhex(st["root"])
    assert all(AC.verify(root, st["rows"], IB.ROW_BITS, IB.BINDING, reg.open(i)) for i in range(st["rows"]))
    assert not AC.verify(root, st["rows"], IB.ROW_BITS, PV2.BINDING, reg.open(0))
    assert IB.commit(circuit, program)[0]["root"] != st["root"] and IB.commit(circuit, program, reg.salts)[0] == st


@pytest.mark.skipif(not os.environ.get("VERITY_VLLM_FA3_LLAMA_1B"),
                    reason="int8 Llama-3.2-1B on sm_90 at 4,32,4: its lowering, minutes on a pod; set VERITY_VLLM_FA3_LLAMA_1B=1")
def test_int8_llama_3_2_1b_on_the_h200_is_one_block_a_slot():
    """int8 Llama-3.2-1B's window circuit on the H200 at 4,32,4: its table, read back, puts every unit in its slot's block,
    and every block has c gates and k fresh-input bits, the circuit unrolled unit by unit agreeing. Its statement is
    printed as a JSON line."""
    circuit = class_circuit(_registered(json.loads(LLAMA_1B.read_text())), WindowClass(4, 32, 4))
    program = IB.program_digest(circuit)
    st, _ = IB.commit(circuit, program)
    rs = IB.rows(circuit, program)
    labels, slots = _labels(circuit), _slots(circuit, rs)
    assert all((slots[t] == labels[t][:, 0]).all() for t in circuit.ids)
    gates, fresh = _unrolled(circuit, slots)
    print(json.dumps({"statement": st, "entries": len(rs) - 1, "units": sum(circuit.units.values())}, sort_keys=True),
          flush=True)
    assert gates == [st["c"]] * 4 and fresh == [st["k"]] * 4 and st["blocks"] == 4
