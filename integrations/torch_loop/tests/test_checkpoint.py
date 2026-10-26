"""The loop's int8 checkpoint role (`checkpoint.ROLE`): its files are the weights the loop serves, bit for bit; its manifest
entry pins them by size and sha256 and names the loop's linears in order; and the auditor's gate
(`benchmarks/live_challenger/models.py`) reads the loop's rows from them and digests them to the role's checkpoint digest."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
from pathlib import Path

from verity.ml.pouw import weights as Wt
from verity_torch_loop import checkpoint as K, loop as L

GATE = Path(__file__).resolve().parents[3] / "benchmarks" / "live_challenger" / "models.py"
LINEARS = ["model.layers.0.mlp.up_proj", "model.layers.1.mlp.down_proj"]


def served() -> L.Model:
    """The model `serve` builds at its default width."""
    return L.Model(h=L.H)


def blobs() -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(K.CHECKPOINT.iterdir())}


def entry() -> dict:
    (e,) = json.loads(K.MANIFEST.read_text())["checkpoints"]
    return e


def test_the_committed_files_are_the_ones_the_model_writes(capsys):
    assert K.main(["--check"]) == 0, f"{capsys.readouterr().out}: rewrite them with `python -m verity_torch_loop.checkpoint`"


def test_the_safetensors_hold_the_served_weights_bit_for_bit():
    m, blob = served(), (K.CHECKPOINT / "model.safetensors").read_bytes()
    (n,) = struct.unpack("<Q", blob[:8])
    header = json.loads(blob[8:8 + n])
    want = {K.EMBED: m.embed}
    for i, layer in enumerate(m.layers):
        want |= {f"{m.op_path(i)}.weight": layer.wq, f"{m.op_path(i)}.weight_scale": layer.sw}
    assert sorted(header) == sorted(want) and "__metadata__" not in header
    for name, t in header.items():
        a = want[name].contiguous().numpy()
        lo, hi = t["data_offsets"]
        assert (t["dtype"], t["shape"]) == (K.DTYPES[a.dtype], list(a.shape)), name
        assert blob[8 + n + lo:8 + n + hi] == a.tobytes(), name
    assert 8 + n + max(hi for _lo, hi in (t["data_offsets"] for t in header.values())) == len(blob)


def test_the_entry_pins_the_files_and_names_the_loops_linears_in_order():
    m, e, files = served(), entry(), blobs()
    assert e["role"] == K.ROLE and e["linears"] == LINEARS == [m.op_path(i) for i in range(len(m.layers))]
    assert sorted(e["files"]) == sorted(files) == ["config.json", "model.safetensors"]
    for name, b in files.items():
        assert e["files"][name] == {"size": len(b), "sha256": hashlib.sha256(b).hexdigest()}, name
    assert e["revision"] == K.checkpoint_sha512(files)[:40]
    cfg = json.loads(files["config.json"])
    assert (cfg["hidden_size"], cfg["num_hidden_layers"], cfg["seed"]) == (m.h, len(m.layers), m.seed)


def test_the_gate_reads_the_loops_rows_and_digests_the_files_to_the_roles_checkpoint():
    spec = importlib.util.spec_from_file_location("live_challenger_models", GATE)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    m, e = served(), entry()
    assert gate.checkpoint_sha512(gate.check_files(K.CHECKPOINT, e)) == K.checkpoint_sha512(blobs())
    ckpt = gate.Checkpoint(K.CHECKPOINT, sorted(e["files"]))
    for i, name in enumerate(e["linears"]):
        wq, sw = m.layers[i].wq.numpy(), m.layers[i].sw.numpy()
        assert gate.parts(name) == [name]
        assert list(gate.rows_of(ckpt, name, m.h)) == [Wt.row_bytes(wq[j].tobytes(), sw[j:j + 1].tobytes()) for j in range(m.h)]
