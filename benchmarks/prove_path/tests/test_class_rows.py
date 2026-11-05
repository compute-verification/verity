"""class_rows: the class circuit's rows by what fills them, on TINY's class circuit.

* no input row of any unit is free: each is wired, a registered read of the weights or of the network certifier's scan
  state rows, a table root or an opening;
* a GEMM slice reads one wired row (o_proj's, its 4 heads' rows) and one registered row and writes one row;
* the scan reads its state row and writes each lane's four outputs, and no unit is a gather or a slot row;
* the census counts every unit once and its compressions are the templates' sum;
* SHA-512 pads a message by 17 bytes, so 111 bytes are one compression and 112 two.
"""
from __future__ import annotations

import sys
from pathlib import Path

from verity_vllm.program.predict.phi import Phi
from verity_vllm.program.predict.steps import Registered
from verity_vllm.program.predict.window_circuit import Grain, WindowClass, class_circuit

from verity.ml.windows import interface as IF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import class_rows as CR  # noqa: E402

#: integrations/vllm's `tests.program.predict.test_steps.TINY` and its grain in test_window_circuit
TINY = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": 64, "intermediate_size": 128,
        "num_attention_heads": 4, "num_key_value_heads": 2, "num_hidden_layers": 2, "vocab_size": 256, "rms_norm_eps": 1e-5,
        "tie_word_embeddings": True, "max_position_embeddings": 64, "hidden_act": "silu"}
GRAIN = Grain(gemm_words=1024, rows=256, select=64)


def _circuit():
    phi = Phi(model={"repo": "org/m", "revision": "r", "hf_config": TINY}, target={"compute_capability": [12, 0], "num_sms": 188})
    return class_circuit(Registered.of(phi, default_seed=7), WindowClass(2, 4, 2), None, GRAIN, provable=False)


def test_every_input_row_of_tinys_class_circuit_is_filled_and_a_gemm_slice_is_wired_registered_and_one_row_out():
    circuit = _circuit()
    doc = CR.census(circuit, IF)
    total, templates = doc["total"], doc["templates"]
    assert not any(k.endswith(".free") for k in total)
    assert total["units"] == sum(n for _, n in circuit.templates) == sum(t["units"] for t in templates.values())
    assert total["compressions"] == sum(t["units"] * t["compressions_a_unit"] for t in templates.values())
    gemms = {lab: t["rows"] for lab, t in templates.items() if lab.startswith("Gemm_v2")}
    assert gemms.pop("Gemm_v2{K=64,N=8}") == {"out": 1, "registered": 1, "wired": 4}
    assert gemms and all(rows == {"out": 1, "registered": 1, "wired": 1} for rows in gemms.values())
    assert {"rows.value", "rows.opening", "rows.table_root"} <= set(total)
    assert [t["rows"] for lab, t in templates.items() if lab.startswith("ScanSlots_v2")] == [{"out": 8, "value": 1}]
    assert not [lab for lab in templates if lab.startswith(("ScanSlots_v1", "SlotRow", "AllGather"))]


def test_a_compression_is_128_bytes_after_sha512s_17_bytes_of_padding():
    assert [CR.blocks(n) for n in (0, 111, 112, 239, 240)] == [1, 1, 2, 2, 3]
