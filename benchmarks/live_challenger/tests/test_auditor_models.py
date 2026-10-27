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
from test_auditor_periods import CIRCUIT, PIN, T0, admin, checked, host, hooks, ok, refused, signer, ssh  # noqa: E402,F401

_spec = importlib.util.spec_from_file_location(
    "h1window_case", HERE.parents[2] / "verity/core/protocols/pouw/tests/test_circuit_h1window.py")
H1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H1)

ROLE = "TINY_INT8"
L0 = "model.layers.0."
QKV, O, GU, DOWN = (L0 + m for m in M.DECODER)
LINEARS = ((QKV, 32, 48), (O, 32, 32), (GU, 32, 128), (DOWN, 64, 32))
CALLS = tuple((name, 0) for name, _k, _n in LINEARS)
PARTS = {L0 + "self_attn.q_proj": (32, 32), L0 + "self_attn.k_proj": (8, 32), L0 + "self_attn.v_proj": (8, 32),
         L0 + "self_attn.o_proj": (32, 32), L0 + "mlp.gate_proj": (64, 32), L0 + "mlp.up_proj": (64, 32),
         L0 + "mlp.down_proj": (32, 64)}
KEY = bytes(range(32))


def descriptor(prefix: str = "h1/", linears=LINEARS) -> bytes:
    calls = tuple((name, 0) for name, _k, _n in linears)
    desc = codec.encode_program(H1.HW.program(linears, calls, 8, H1.TILE, prefix=prefix, unit=H1.unit))
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


def pinned(tmp_path, parts, window, split, config=None, role=ROLE, manifest=M.MANIFEST, **fields):
    """A pinned checkpoint of int8 modules `parts` (two shards, the first holding `split` tensors), role `role`'s entry
    (with `fields`) in the pinned tree's `manifest` (vLLM's holding another role when `manifest` is another file), core's
    pin, and the descriptor of the window's linears `window`."""
    rng = np.random.default_rng(7)
    ts = {}
    for name, (n, k) in parts.items():
        ts[f"{name}.weight"] = rng.integers(-127, 128, (n, k), dtype=np.int8)
        ts[f"{name}.weight_scale"] = rng.uniform(1e-3, 1, (n, 1)).astype("<f4")
    ck, tree = tmp_path / "ckpt", tmp_path / "tree"
    ck.mkdir(mode=0o755)
    names = sorted(ts)
    safetensors(ck / "model-00001-of-00002.safetensors", {n: ts[n] for n in names[:split]})
    safetensors(ck / "model-00002-of-00002.safetensors", {n: ts[n] for n in names[split:]})
    if config is not None:
        (ck / "config.json").write_text(json.dumps(config))
    for p in ck.iterdir():
        p.chmod(0o644)
    files = {p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "size": p.stat().st_size,
                      "digest_source": "recipe"} for p in sorted(ck.iterdir())}
    entry = {"role": role, "repo": "verity-int8/Tiny", "revision": "ab" * 20, "recipe": {"id": "verity-int8-channel127/v1"},
             **fields, "files": files}
    for path in {M.MANIFEST, manifest}:
        (tree / path).parent.mkdir(parents=True)
        (tree / path).write_text(json.dumps({"checkpoints": [entry if path == manifest else {"role": "LLAMA32_1B_INT8"}]}))
    pin = M.checkpoint_sha512({p.name: hashlib.sha512(p.read_bytes()).hexdigest() for p in ck.iterdir()})
    (tree / M.PINS).parent.mkdir(parents=True)
    (tree / M.PINS).write_text(json.dumps({role: {"checkpoint_sha512": pin}}))
    d = descriptor(linears=window)
    (tmp_path / "desc.json").write_bytes(d)
    return {"ckpt": ck, "tree": tree, "tensors": ts, "descriptor": d, "circuit": hashlib.sha512(d).hexdigest(),
            "desc_path": tmp_path / "desc.json", "pin": pin, "role": role, "entry": entry}


@pytest.fixture
def model(tmp_path):
    """A pinned tiny checkpoint (two shards), its manifest in a pinned tree, and its tensors."""
    return pinned(tmp_path, PARTS, LINEARS, 6, config={"num_hidden_layers": 1})


def expected_roots(ts, circuit, key=KEY):
    def fused(parts):
        w = np.concatenate([ts[f"{L0}{p}.weight"] for p in parts])
        s = np.concatenate([ts[f"{L0}{p}.weight_scale"] for p in parts]).reshape(-1)
        return [w[j].tobytes() + s[j].tobytes() for j in range(len(w))]
    rows = [fused(["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]), fused(["self_attn.o_proj"]),
            fused(["mlp.gate_proj", "mlp.up_proj"]), fused(["mlp.down_proj"])]
    return Wt.weights_roots(circuit, "h1/", [(n, k) for _name, k, n in LINEARS], lambda w: rows[w], key)


def test_a_registration_commits_the_pinned_checkpoints_rows_for_the_descriptors_linears(model):
    """qkv_proj is q, k and v's rows in turn, each its int8 codes and s_w, in the descriptor's linear order; the roots, the
    entries and their fold are `weights`', under the circuit's domain, with the auditor's key."""
    line = M.register(model["tree"], ROLE, model["circuit"], model["descriptor"], model["ckpt"], KEY, pinned=PIN,
                      now="2026-10-22T19:00:00Z")
    roots = expected_roots(model["tensors"], model["circuit"])
    assert line["roots"] == {str(k): r.hex() for k, r in roots.items()}
    assert line["weights_root"] == Wt.weights_root(roots).hex()
    assert line["prefix"] == CR.shape_of(model["descriptor"])["prefix"] == "h1/" and line["model"] == ROLE and line["circuit"] == model["circuit"]
    assert line["entries"] == [Wt.entry(model["circuit"], "h1/", 32, 208, roots[32]),
                               Wt.entry(model["circuit"], "h1/", 64, 32, roots[64])]
    assert line["checkpoint_sha512"] == model["pin"]
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


def reg(model, descriptor=None, circuit=None):
    d = descriptor or model["descriptor"]
    return M.register(model["tree"], model["role"], circuit or hashlib.sha512(d).hexdigest(), d, model["ckpt"], KEY,
                      pinned=PIN, now="t")


def test_a_checkpoint_that_isnt_core_s_pin_or_that_another_user_can_write_is_refused(model):
    """The registration signs only the checkpoint credit's pin names, from files nothing but root and auditor could
    change between their hash and their read."""
    pins = model["tree"] / M.PINS
    pins.write_text(json.dumps({ROLE: {"checkpoint_sha512": "0" * 128}}))
    with pytest.raises(M.Refused, match="is not .*'s pin"):
        reg(model)
    pins.write_text("{}")
    with pytest.raises(M.Refused, match="pins no checkpoint_sha512"):
        reg(model)
    pins.write_text(json.dumps({ROLE: {"checkpoint_sha512": model["pin"]}}))
    shard = model["ckpt"] / "model-00001-of-00002.safetensors"
    shard.chmod(0o664)
    with pytest.raises(M.Refused, match="only root or auditor"):
        reg(model)
    shard.chmod(0o644)
    link = model["ckpt"] / "config.json"
    real = model["ckpt"].parent / "config.json"
    link.rename(real)
    link.symlink_to(real)
    with pytest.raises(M.Refused, match="no symlink"):
        reg(model)


def test_a_shard_outside_the_pin_never_shadows_a_pinned_tensor(model):
    """An extra safetensors file in DIR is never opened, and a tensor in two pinned shards is refused."""
    extra = {f"{L0}self_attn.q_proj.weight": np.zeros((32, 32), np.int8)}
    safetensors(model["ckpt"] / "zz.safetensors", extra)
    roots = expected_roots(model["tensors"], model["circuit"])
    assert reg(model)["weights_root"] == Wt.weights_root(roots).hex()
    with pytest.raises(M.Refused, match="is in both"):
        M.Checkpoint(model["ckpt"], ["model-00001-of-00002.safetensors", "zz.safetensors",
                                     "model-00002-of-00002.safetensors"])


@pytest.mark.parametrize("linears", [
    (LINEARS[1], LINEARS[0], LINEARS[2], LINEARS[3]),
    LINEARS[:3],
    LINEARS + (("model.layers.1.self_attn.qkv_proj", 32, 48),),
    (("model.layers.0.self_attn.kqv_proj", 32, 48),) + LINEARS[1:],
])
def test_a_descriptor_whose_linears_arent_the_models_decoder_layers_in_order_is_refused(model, linears):
    """A descriptor that permutes, drops, adds or renames a linear would sign the pinned checkpoint's root for a function
    the model doesn't compute."""
    with pytest.raises(M.Refused, match="decoder layers"):
        reg(model, descriptor(linears=linears))


UP, DN = "model.layers.0.mlp.up_proj", "model.layers.1.mlp.down_proj"
MLP = ((UP, 32, 64), (DN, 64, 32))
#: a role whose entry declares its linears (an MLP's), its files aside
MLP_ROLE = "MLP_H128_W8A8"
MLP_ENTRY = {
    "linears": [UP, DN],
    "note": "a two-layer MLP's int8 weights, whose entry declares its linears",
    "recipe": {"h": 128, "id": "mlp-w8a8/v1", "seed": 0, "vocab": 16},
    "repo": "verity-int8/mlp-w8a8-h128", "revision": "ef47a4f92c408f8c64778a9934b81c9729f2df2b"}


@pytest.fixture
def mlp(tmp_path):
    """An MLP role: a two-layer MLP in vLLM's manifest, whose entry declares its linears, up_proj then down_proj, and
    whose checkpoint has no config.json."""
    (tmp_path / "mlp").mkdir()
    return pinned(tmp_path / "mlp", {UP: (64, 32), DN: (32, 64)}, MLP, 2, role=MLP_ROLE, **MLP_ENTRY)


def test_a_role_that_declares_its_linears_commits_each_unmerged_module_as_itself(mlp):
    """An unmerged module is one part, itself: each linear's rows are its own int8 codes and s_w, in the declared order,
    and the role's checkpoint needs no config.json."""
    assert M.parts(UP) == [UP] and M.parts(DN) == [DN]
    ts = mlp["tensors"]
    rows = [[ts[f"{m}.weight"][j].tobytes() + ts[f"{m}.weight_scale"][j].tobytes() for j in range(n)]
            for m, _k, n in MLP]
    roots = Wt.weights_roots(mlp["circuit"], "h1/", [(n, k) for _m, k, n in MLP], lambda w: rows[w], KEY)
    line = reg(mlp)
    assert line["weights_root"] == Wt.weights_root(roots).hex() and line["model"] == MLP_ROLE
    assert line["roots"] == {str(k): r.hex() for k, r in roots.items()}
    assert line["source"]["repo"] == MLP_ENTRY["repo"]


def test_a_role_listed_twice_or_not_at_all_is_refused(mlp):
    """Exactly one entry: a role the manifest lists twice is ambiguous, and an unlisted one unknown."""
    vllm = mlp["tree"] / M.MANIFEST
    assert M.manifest_entry(mlp["tree"], MLP_ROLE) == mlp["entry"]
    vllm.write_text(json.dumps({"checkpoints": [mlp["entry"], {"role": MLP_ROLE, "repo": "elsewhere"}]}))
    with pytest.raises(M.Refused, match=f"2 entries .*not one role of {M.MANIFESTS[0]}"):
        M.manifest_entry(mlp["tree"], MLP_ROLE)
    with pytest.raises(M.Refused, match=f"0 entries \\(none\\), not one role of {M.MANIFESTS[0]}"):
        M.manifest_entry(mlp["tree"], "QWEN3_06B_INT8")


def test_an_unreadable_or_missing_manifest_is_refused(mlp):
    (mlp["tree"] / M.MANIFEST).write_text("{")
    with pytest.raises(M.Refused, match="integrations/vllm/manifests/checkpoints-int8.json is not readable JSON"):
        M.manifest_entry(mlp["tree"], MLP_ROLE)
    (mlp["tree"] / M.MANIFEST).unlink()
    with pytest.raises(M.Refused, match="the pinned tree has no integrations/vllm"):
        M.manifest_entry(mlp["tree"], MLP_ROLE)


@pytest.mark.parametrize("linears", [
    (MLP[1], MLP[0]),
    MLP[:1],
    MLP + (("model.layers.2.mlp.up_proj", 32, 64),),
    LINEARS,
])
def test_a_descriptor_whose_linears_arent_the_roles_declared_ones_in_order_is_refused(mlp, linears):
    """Reordered, missing, extra, or the decoder's: the declared list is the whole of what the role's root may sign."""
    with pytest.raises(M.Refused, match="the role's declared linears"):
        reg(mlp, descriptor(linears=linears))


@pytest.mark.parametrize("declared", [[], [UP, UP], "model.layers.0.mlp.up_proj", [UP, 3], {"0": UP}])
def test_a_role_whose_declared_linears_are_malformed_is_refused(tmp_path, declared):
    m = pinned(tmp_path, {UP: (64, 32), DN: (32, 64)}, MLP, 2, linears=declared)
    with pytest.raises(M.Refused, match="not a nonempty list of distinct module names"):
        reg(m)


def test_a_role_that_declares_no_linears_keeps_the_decoder_rule(model, tmp_path):
    """Without the field, the role's linears are config.json's decoder layers', as before: the MLP's descriptor is
    refused for it, and so is a checkpoint with no config.json."""
    assert M.declared_linears(M.manifest_entry(model["tree"], ROLE)) is None
    with pytest.raises(M.Refused, match="decoder layers"):
        reg(model, descriptor(linears=MLP))
    (tmp_path / "bare").mkdir()
    bare = pinned(tmp_path / "bare", {UP: (64, 32), DN: (32, 64)}, MLP, 2)
    with pytest.raises(M.Refused, match="no readable config.json"):
        reg(bare)


def test_a_linear_the_checkpoint_shapes_otherwise_is_refused(model, monkeypatch):
    monkeypatch.setattr(M, "layout", lambda d, c: ("h1/", [(QKV, 32, 40), *LINEARS[1:]]))
    with pytest.raises(M.Refused, match="gave 48 rows"):
        reg(model)
    monkeypatch.setattr(M, "layout", lambda d, c: ("h1/", [(QKV, 64, 48), *LINEARS[1:]]))
    with pytest.raises(M.Refused, match="not n × 64"):
        reg(model)


def register_hooks(host, signer, model, key=KEY, **kw):
    return dict(hooks(host, signer, **kw), seal=lambda p: None,
                register=lambda args: A.model_register(host, args, pinned=PIN, now=T0, key=key, tree=model["tree"]),
                period=lambda args: A.period(host, args, signer=signer, pinned=PIN, now=T0, active=lambda s: False,
                                             tree=model["tree"]))


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
    assert ok(ssh(host, signer, f"model-key --model {ROLE} --circuit {c} --weights-root {reg['weights_root']}",
                  "hash"))["key"] == KEY.hex()
    refused(ssh(host, signer, f"model-key --model {ROLE} --circuit {c} --weights-root {'5e' * 64}", "hash"), "no entry")
    rc, out = A.gate(["--ssh", f"model-key --model {ROLE} --circuit {c}"], host)
    refused((rc, json.loads(out)), "reads only")


def gate(host, signer, model, words, key=None, **kw):
    """A lane key's request (`key`), or vy-admin's, through the gate with the fixture's pinned tree."""
    argv = ["--ssh", words, "--key", key] if key else words.split()
    rc, out = A.gate(argv, host, **({} if key else {"by": "vy-admin"}), **register_hooks(host, signer, model, **kw))
    return rc, json.loads(out)


@pytest.mark.parametrize("args, why", [
    ("--model OTHER", "pins no checkpoint_sha512 of OTHER"),
    ("--weights-root " + "5e" * 64, "no entry"),
    ("--model TINY_INT8 --weights-root " + "5e" * 64, "no entry"),
])
def test_a_period_open_refuses_a_root_the_registry_lacks_or_a_model_core_doesnt_pin(host, signer, model, args, why):
    """A period's weights' root is the auditor's registration, never the prover's: an unregistered root, or a registered
    model with another root, is refused, and so is a model core's pin doesn't name."""
    words = (f"model-register --model {ROLE} --circuit {model['circuit']} --descriptor {model['desc_path']} "
             f"--checkpoint {model['ckpt']}")
    assert A.gate(words.split(), host, by="vy-admin", **register_hooks(host, signer, model))[0] == 0
    refused(gate(host, signer, model, f"period-open --name p1 --cap 2 --circuit {model['circuit']} {args}", "hash"), why)
    body = ok(gate(host, signer, model, f"period-open --name p1 --cap 2 --circuit {CIRCUIT}", "hash"))["body"]
    assert "weights_root" not in body and "model" not in body


def test_a_pinned_model_under_a_circuit_the_registry_lacks_signs_the_model_alone(host, signer, model):
    """A remote memory audit's period (its circuit its verifier tree, no b/<K> registered) names its pinned model: the
    open signs `model` with core's pinned digest and no weights' root, so H1 credit, which needs one, refuses its windows;
    a role core doesn't pin is refused."""
    body = ok(gate(host, signer, model, f"period-open --name m1 --cap 2 --circuit {CIRCUIT} --model {ROLE}",
                   "hash"))["body"]
    assert body["model"] == {"name": ROLE, "checkpoint_sha512": model["pin"]} and "weights_root" not in body
    ok(gate(host, signer, model, "period-close --name m1", "hash"))
    refused(gate(host, signer, model, f"period-open --name m2 --cap 2 --circuit {CIRCUIT} --model OTHER", "hash"),
            "pins no checkpoint_sha512")
    assert set(checked(host)) == {"m1"}


def test_model_key_names_the_registration_when_the_model_has_two(host, signer, model):
    """After a re-registration under the same circuit, model-key without --weights-root is refused (the period's signed
    root picks the key); with it, each period's key is its own registration's."""
    words = (f"model-register --model {ROLE} --circuit {model['circuit']} --descriptor {model['desc_path']} "
             f"--checkpoint {model['ckpt']}")
    roots = []
    for key in (KEY, bytes(32)):
        rc, out = A.gate(words.split(), host, by="vy-admin", **register_hooks(host, signer, model, key=key))
        assert rc == 0, out
        roots.append(json.loads(out)["weights_root"])
    c = model["circuit"]
    refused(gate(host, signer, model, f"model-key --model {ROLE} --circuit {c}", "hash"), "registered 2 times")
    for key, root in zip((KEY, bytes(32)), roots):
        assert ok(gate(host, signer, model, f"model-key --model {ROLE} --circuit {c} --weights-root {root}",
                       "hash"))["key"] == key.hex()
