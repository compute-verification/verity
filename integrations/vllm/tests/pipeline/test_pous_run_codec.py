"""`verity-vllm pous` under ``--scheme band-chain/d12/v1``, on a toy model through the protocol option: `W` of record and its
commitment as `protocol_options.pous` lays `W` out, the verifier's own band encoding and `vk`, the layer-0 reference decode, the
served store's settings, and a timed audit against the stand-in responder of `test_pous_verifier`, at a 128-byte band segment.
CPU torch, no GPU: the served generation and the GPU twin are the pod run's."""
from __future__ import annotations

import dataclasses
import json
import types

import pytest
from verity_pous import band, protocol

from tests.pipeline.test_pous_verifier import _run
from verity_vllm.pipeline import cli, pous_run
from verity_vllm.protocol_options import pous

torch = pytest.importorskip("torch")

BAND = "band-chain/d12/v1"
SMALL = band.BandParams(16, 64, 12)


class Linear:
    def __init__(self, weight) -> None:
        self.weight = weight


class Model:
    """Three linears, one of them tied to another, as `pous._plaintext` finds them."""

    def __init__(self) -> None:
        g = torch.Generator().manual_seed(5)
        a, b = torch.randn(4, 12, generator=g).to(torch.bfloat16), torch.randn(5, 9, generator=g).to(torch.bfloat16)
        self.layers = {"model.layers.0.self_attn.q_proj": Linear(a), "model.layers.1.mlp.down_proj": Linear(b), "lm_head": Linear(a)}

    def named_modules(self):
        yield "", self
        yield from self.layers.items()


def _args(out, scheme: str = BAND, prefill: str = "", decode_tokens: int = 0, **kw) -> types.SimpleNamespace:
    return types.SimpleNamespace(**{"out": str(out), "case": "TOY", "build_dir": str(out / "build"), "scheme": scheme, "workers": 2,
                                    "tag": "honest", "drop_fraction": 0.0, "responder_cpu": None, "prefill": prefill,
                                    "decode_tokens": decode_tokens, "kernels": "", **kw})


@pytest.fixture
def toy(tmp_path, monkeypatch):
    monkeypatch.setattr(pous, "linear_types", lambda: (Linear,))
    monkeypatch.setattr(pous, "_base", lambda scheme: SMALL if scheme == BAND else pous_run.geometry(scheme))
    monkeypatch.setattr(pous_run, "geometry", lambda name, real=pous_run.geometry: SMALL if name == BAND else real(name))
    model = Model()
    return tmp_path, model, pous_run.record_w(BAND, model, tmp_path)


def test_the_verifier_roles_encode_and_audit_under_the_band(toy):
    out, model, commit = toy
    q, down = (model.layers[n].weight.contiguous().view(torch.uint8).numpy().tobytes()
               for n in ("model.layers.0.self_attn.q_proj", "model.layers.1.mlp.down_proj"))
    assert commit["scheme"] == BAND and commit["params"] == SMALL.with_segments(2).manifest()    # 96 + 90 bytes: 2 segments
    assert commit["tensors"] == [["model.layers.0.self_attn.q_proj", len(q)], ["model.layers.1.mlp.down_proj", len(down)]]
    doc, layout, scheme, w = pous_run.committed_w(out)
    assert layout.unpack(w, "model.layers.0.self_attn.q_proj") == q and protocol.commit(scheme, w).hex() == commit["commitment"]

    pous_run.cmd_vk(_args(out))
    vkdoc = json.loads((out / "vk.json").read_text())
    assert vkdoc["scheme"] == BAND and vkdoc["reference_decode_matches_w"]
    assert list(vkdoc["reference_decode_layer0"]) == ["model.layers.0.self_attn.q_proj"]
    loaded, sch, vk = pous_run.load_vk(out)
    assert loaded == vkdoc and (sch.name, sch.params) == (scheme.name, scheme.params) and vk.root.hex() == vkdoc["root"] and vk.params == SMALL.with_segments(2)

    salt = bytes.fromhex(vkdoc["salt"])
    canonical = protocol.encode(scheme, w, salt)                                     # the band's own encoding
    assert vk == protocol.verifier_key(scheme, bytes.fromhex(commit["commitment"]), salt, w=w)
    blocks = [b for seg in canonical for b in seg]
    assert _run(blocks, scheme, vk)["audit_of_record"]["accepted"]                   # the server's encoding, as the twin makes it
    dropped = [bytes(len(b)) if i % 4 == 0 else b for i, b in enumerate(blocks)]
    assert not _run(dropped, scheme, vk)["audit_of_record"]["accepted"]


def test_the_served_store_is_the_gpu_codec_under_the_verifiers_salt_and_the_default_is_the_band(toy):
    out, _, _ = toy
    (out / "salt.json").write_text(json.dumps({"salt": "00" * 24}))
    own = pous_run.serve_config(_args(out, drop_fraction=0.1, responder_cpu=3, kernels='{"chunked": true, "thread_rows": 500}'), out)["pous"]
    assert own["codec"] == "gpu" and own["salt"] == "00" * 24 and own["port"] == 0 and own["drop_fraction"] == 0.1
    assert own["responder_cpu"] == 3 and own["build_dir"] == str(out / "build") and set(own) <= set(pous.SETTINGS)
    assert own["kernels"] == {"chunked": True, "thread_rows": 500} and "kernels" not in pous_run.serve_config(_args(out), out)["pous"]
    assert pous_run._base(_args(out, prefill="2048x4", decode_tokens=4))[-8:] == [
        "--scheme", BAND, "--prefill", "2048x4", "--decode-tokens", "4", "--kernels", ""]
    default = {f.name: f.default for f in dataclasses.fields(pous_run.COMMANDS["e2e"])}["scheme"]
    assert default == BAND == pous.DEFAULT_SCHEME
    common = ["--out", str(out), "--case", "TOY", "--build-dir", str(out)]
    assert cli.parse("pous", ["vk", *common, "--codec", "p3/v1"]).scheme == "p3/v1"            # #333's --codec is --scheme
    assert cli.parse("pous", ["e2e", *common]).scheme == BAND


def test_prefill_and_decode_slowdowns_pair_the_plain_and_served_configurations(toy):
    out, _, _ = toy
    assert pous_run.engine_args(_args(out, prefill="2048x4")) == {"max_num_batched_tokens": pous_run.MAX_BATCHED_TOKENS}
    assert pous_run.engine_args(_args(out)) is None                     # the plain and served engines get the same budget
    plain = [{"kind": "prefill", "length": 2048, "batch": 4, "tokens_per_s": 150_000.0},
             {"kind": "decode", "batch": 1, "tokens_per_s": 100.0}]
    served = [{"kind": "prefill", "length": 2048, "batch": 4, "tokens_per_s": 7_500.0, "decode_s": 1.0, "decodes": 3},
              {"kind": "decode", "batch": 1, "tokens_per_s": 1.25, "decode_s": 3.2, "decodes": 12}]
    got = pous_run.slowdowns(plain, served)
    assert list(got) == ["prefill 2048x4", "decode batch 1"]
    assert got["prefill 2048x4"]["slowdown"] == 20.0 and got["decode batch 1"]["slowdown"] == 80.0
    assert got["decode batch 1"]["pous_decodes"] == 12 and pous_run.slowdowns(plain, []) == {}


def test_the_verifier_refuses_another_scheme_and_a_w_that_is_not_the_committed_one(toy):
    out, _, _ = toy
    with pytest.raises(SystemExit, match=f"committed under {BAND}, not p3/v1"):
        pous_run.cmd_vk(_args(out, scheme="p3/v1"))
    w = bytearray((out / "w.bin").read_bytes())
    w[0] ^= 1
    (out / "w.bin").write_bytes(bytes(w))
    with pytest.raises(SystemExit, match="do not give commit.json's parameters and commitment"):
        pous_run.cmd_vk(_args(out))
