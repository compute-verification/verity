"""The auditor's model registry: `model-register` commits a pinned int8 checkpoint's b/<K> under the auditor's own salt key
for the window Program its descriptor declares, `period-open --model` (or a registered `--weights-root`) signs that
registration's root, and `model-key` hands a lane the key."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

from verity.core.protocols.pouw import credit as CR
from verity.core.protocols.pouw import weights as Wt
from verity.ml.circuits import codec

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import auditor as A  # noqa: E402
import models as M  # noqa: E402
from test_auditor_periods import CIRCUIT, PIN, T0, admin, host, hooks, ok, refused, signer, ssh  # noqa: E402,F401

_spec = importlib.util.spec_from_file_location(
    "h1window_case", HERE.parents[2] / "verity/core/protocols/pouw/tests/test_circuit_h1window.py")
H1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H1)

ROLE = "TINY_INT8"
QKV, DOWN = "model.layers.0.self_attn.qkv_proj", "model.layers.0.mlp.down_proj"
LINEARS = ((QKV, 32, 16), (DOWN, 64, 8))
CALLS = ((QKV, 0), (DOWN, 0))
PARTS = {"model.layers.0.self_attn.q_proj": (8, 32), "model.layers.0.self_attn.k_proj": (4, 32),
         "model.layers.0.self_attn.v_proj": (4, 32), "model.layers.0.mlp.down_proj": (8, 64)}
KEY = bytes(range(32))


def descriptor(prefix: str = "h1/") -> bytes:
    desc = codec.encode_program(H1.HW.program(LINEARS, CALLS, 8, H1.TILE, prefix=prefix, unit=H1.unit))
    return codec.canonical_json({k: v for k, v in desc.items() if k != "annotations"})


def safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    header, blobs, at = {}, [], 0
    for name, t in tensors.items():
        b = t.tobytes()
        header[name] = {"dtype": {"int8": "I8", "float32": "F32"}[str(t.dtype)], "shape": list(t.shape),
                        "data_offsets": [at, at + len(b)]}
        blobs.append(b)
        at += len(b)
    h = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))


@pytest.fixture
def model(tmp_path):
    """A pinned tiny checkpoint (two shards), its manifest in a pinned tree, and its tensors."""
    rng = np.random.default_rng(7)
    ts = {}
    for name, (n, k) in PARTS.items():
        ts[f"{name}.weight"] = rng.integers(-127, 128, (n, k), dtype=np.int8)
        ts[f"{name}.weight_scale"] = rng.uniform(1e-3, 1, (n, 1)).astype("<f4")
    ck, tree = tmp_path / "ckpt", tmp_path / "tree"
    ck.mkdir()
    names = sorted(ts)
    safetensors(ck / "model-00001-of-00002.safetensors", {n: ts[n] for n in names[:4]})
    safetensors(ck / "model-00002-of-00002.safetensors", {n: ts[n] for n in names[4:]})
    (ck / "config.json").write_text("{}")
    files = {p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "size": p.stat().st_size,
                      "digest_source": "recipe"} for p in sorted(ck.iterdir())}
    (tree / M.MANIFEST).parent.mkdir(parents=True)
    (tree / M.MANIFEST).write_text(json.dumps({"checkpoints": [
        {"role": ROLE, "repo": "verity-int8/Tiny", "revision": "ab" * 20, "files": files,
         "recipe": {"id": "verity-int8-channel127/v1"}}]}))
    d = descriptor()
    (tmp_path / "desc.json").write_bytes(d)
    return {"ckpt": ck, "tree": tree, "tensors": ts, "descriptor": d, "circuit": hashlib.sha512(d).hexdigest(),
            "desc_path": tmp_path / "desc.json"}


def expected_roots(ts, circuit, key=KEY):
    def fused(parts):
        w = np.concatenate([ts[f"{p}.weight"] for p in parts])
        s = np.concatenate([ts[f"{p}.weight_scale"] for p in parts]).reshape(-1)
        return [w[j].tobytes() + s[j].tobytes() for j in range(len(w))]
    rows = [fused([f"model.layers.0.self_attn.{x}_proj" for x in "qkv"]), fused([DOWN])]
    return Wt.weights_roots(circuit, "h1/", [(16, 32), (8, 64)], lambda w: rows[w], key)


def test_a_registration_commits_the_pinned_checkpoints_rows_for_the_descriptors_linears(model):
    """qkv_proj is q, k and v's rows in turn, each its int8 codes and s_w, in the descriptor's linear order; the roots, the
    entries and their fold are `weights`', under the circuit's domain, with the auditor's key."""
    line = M.register(model["tree"], ROLE, model["circuit"], model["descriptor"], model["ckpt"], KEY, pinned=PIN,
                      now="2026-10-22T19:00:00Z")
    roots = expected_roots(model["tensors"], model["circuit"])
    assert line["roots"] == {str(k): r.hex() for k, r in roots.items()}
    assert line["weights_root"] == Wt.weights_root(roots).hex()
    assert line["prefix"] == CR.shape_of(model["descriptor"])["prefix"] == "h1/" and line["model"] == ROLE and line["circuit"] == model["circuit"]
    assert line["entries"] == [Wt.entry(model["circuit"], "h1/", 32, 16, roots[32]),
                               Wt.entry(model["circuit"], "h1/", 64, 8, roots[64])]
    assert line["key_sha512"] == hashlib.sha512(KEY).hexdigest()
    assert set(line["checkpoint"]) == {p.name for p in model["ckpt"].iterdir()}


def test_a_registration_refuses_a_checkpoint_or_descriptor_that_isnt_the_pinned_one(model):
    shard = model["ckpt"] / "model-00002-of-00002.safetensors"
    good = shard.read_bytes()
    shard.write_bytes(good[:-1] + bytes([good[-1] ^ 1]))
    with pytest.raises(M.Refused, match="sha256"):
        M.register(model["tree"], ROLE, model["circuit"], model["descriptor"], model["ckpt"], KEY, pinned=PIN, now="t")
    shard.write_bytes(good)
    with pytest.raises(M.Refused, match="not --circuit"):
        M.register(model["tree"], ROLE, "ab" * 64, model["descriptor"], model["ckpt"], KEY, pinned=PIN, now="t")
    with pytest.raises(M.Refused, match="one role"):
        M.register(model["tree"], "OTHER", model["circuit"], model["descriptor"], model["ckpt"], KEY, pinned=PIN, now="t")
    other = descriptor("h1/w-2/")
    with pytest.raises(M.Refused, match="not --circuit"):
        M.register(model["tree"], ROLE, model["circuit"], other, model["ckpt"], KEY, pinned=PIN, now="t")


def test_a_linear_the_checkpoint_lacks_or_shapes_otherwise_is_refused(model, monkeypatch):
    monkeypatch.setattr(M, "layout", lambda d, c: ("h1/", [("model.layers.0.self_attn.o_proj", 32, 16)]))
    with pytest.raises(M.Refused, match="no tensor"):
        M.register(model["tree"], ROLE, model["circuit"], model["descriptor"], model["ckpt"], KEY, pinned=PIN, now="t")
    monkeypatch.setattr(M, "layout", lambda d, c: ("h1/", [(QKV, 32, 24)]))
    with pytest.raises(M.Refused, match="gave 16 rows"):
        M.register(model["tree"], ROLE, model["circuit"], model["descriptor"], model["ckpt"], KEY, pinned=PIN, now="t")


def register_hooks(host, signer, model, **kw):
    return dict(hooks(host, signer, **kw), seal=lambda p: None,
                register=lambda args: A.model_register(host, args, pinned=PIN, now=T0, key=KEY, tree=model["tree"]))


def test_vy_admin_registers_and_a_period_open_signs_the_registered_root_and_model(host, signer, model):
    words = (f"model-register --model {ROLE} --circuit {model['circuit']} --descriptor {model['desc_path']} "
             f"--checkpoint {model['ckpt']}")
    rc, out = A.gate(["--ssh", words, "--key", "hash"], host, **register_hooks(host, signer, model))
    assert rc == 3 and "vy-admin alone" in json.loads(out)["refused"]
    rc, out = A.gate(words.split(), host, by="vy-admin", **register_hooks(host, signer, model))
    reg = json.loads(out)
    assert rc == 0 and reg["weights_root"] == Wt.weights_root(expected_roots(model["tensors"], model["circuit"])).hex()
    assert [json.loads(x)["weights_root"] for x in host.models.read_text().splitlines()] == [reg["weights_root"]]
    assert (A.gate(["--ssh", "models"], host)[1] == host.models.read_bytes())

    c = model["circuit"]
    body = ok(ssh(host, signer, f"period-open --name p1 --cap 2 --circuit {c} --draw-k 27 --model {ROLE}", "hash"))["body"]
    assert body["weights_root"] == reg["weights_root"]
    assert body["model"] == {"name": ROLE, "checkpoint_sha512": reg["checkpoint_sha512"]}
    ok(ssh(host, signer, "period-close --name p1", "hash"))
    body = ok(admin(host, signer, f"period-open --name k1 --lane kernels --cap 1 --circuit {c} "
                                  f"--weights-root {reg['weights_root']}"))["body"]
    assert body["model"]["name"] == ROLE and body["weights_root"] == reg["weights_root"]
    ok(admin(host, signer, "period-close --name k1"))

    key = ok(ssh(host, signer, f"model-key --model {ROLE} --circuit {c}", "hash"))
    assert key["key"] == KEY.hex() and key["weights_root"] == reg["weights_root"]
    rc, out = A.gate(["--ssh", f"model-key --model {ROLE} --circuit {c}"], host)
    refused((rc, json.loads(out)), "reads only")


@pytest.mark.parametrize("args", ["--model OTHER", "--weights-root " + "5e" * 64, "--model TINY_INT8 --weights-root " + "5e" * 64])
def test_a_period_open_refuses_a_model_or_root_the_registry_has_no_entry_of(host, signer, model, args):
    """A period's weights' root is the auditor's registration, never the prover's: an unregistered model, root, or a
    registered model with another root, is refused, as is a registered model under another circuit."""
    words = (f"model-register --model {ROLE} --circuit {model['circuit']} --descriptor {model['desc_path']} "
             f"--checkpoint {model['ckpt']}")
    assert A.gate(words.split(), host, by="vy-admin", **register_hooks(host, signer, model))[0] == 0
    refused(ssh(host, signer, f"period-open --name p1 --cap 2 --circuit {model['circuit']} {args}", "hash"),
            "no entry")
    refused(ssh(host, signer, f"period-open --name p1 --cap 2 --circuit {CIRCUIT} --model {ROLE}", "hash"), "no entry")
    body = ok(ssh(host, signer, f"period-open --name p1 --cap 2 --circuit {CIRCUIT}", "hash"))["body"]
    assert "weights_root" not in body and "model" not in body
