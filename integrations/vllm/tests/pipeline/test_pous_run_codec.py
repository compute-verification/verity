"""`verity-vllm pous` under ``--codec band-chain/d12/v1``: the server's commitment, the verifier's own band encoding and
`vk`, the layer-0 reference decode, the served store's spec, and a timed audit against the stand-in responder of
`test_pous_verifier`, on a synthetic three-tensor checkpoint at a 128-byte band segment.  No torch, no GPU: the served
generation and the GPU twin are the pod run's."""
from __future__ import annotations

import dataclasses
import json
import struct
import types

import pytest
from verity_pous import band

from tests.pipeline.test_pous_verifier import _run
from verity_vllm.pipeline import pous_run
from verity_vllm.pipeline import pous_verifier as pv

BAND = "band-chain/d12/v1"
SMALL = band.BandParams(16, 64, 12)


def _checkpoint(path) -> bytes:
    tensors = {"model.layers.0.self_attn.q_proj.weight": bytes(range(96)),
               "model.layers.0.mlp.down_proj.weight": bytes(range(100, 200)),
               "lm_head.weight": bytes(range(40))}
    header, data = {}, b""
    for name, raw in tensors.items():
        header[name] = {"dtype": "BF16", "shape": [len(raw) // 2], "data_offsets": [len(data), len(data) + len(raw)]}
        data += raw
    h = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(h)) + h + data)
    return data


def _args(out, codec: str = BAND) -> types.SimpleNamespace:
    return types.SimpleNamespace(out=str(out), case="TOY", build_dir=str(out), codec=codec, workers=2, tag="honest",
                                 drop_fraction=0.0, layers_per_decode=1)


@pytest.fixture
def toy(tmp_path, monkeypatch):
    data = _checkpoint(tmp_path / "model.safetensors")
    monkeypatch.setattr(pous_run, "checkpoint_files", lambda case: [str(tmp_path / "model.safetensors")])
    served = pous_run.reference_codec
    monkeypatch.setattr(pous_run, "reference_codec", lambda name: (served(name)[0], SMALL) if name == BAND else served(name))
    return tmp_path, data


def test_the_verifier_roles_encode_and_audit_under_the_band(toy):
    out, data = toy
    pous_run.cmd_commit(_args(out))
    commit = json.loads((out / "commit.json").read_text())
    assert commit["codec"] == BAND and commit["params"] == SMALL.with_segments(2).manifest()        # 236 bytes: 2 segments
    pous_run.cmd_vk(_args(out))
    doc = json.loads((out / "vk.json").read_text())
    assert doc["codec"] == BAND and doc["reference_decode_matches_w"] and len(doc["reference_decode_layer0"]) == 2
    loaded, vk = pous_run.load_vk(out)
    assert loaded == doc and vk.root.hex() == doc["root"] and vk.params == SMALL.with_segments(2)

    salt = bytes.fromhex(doc["salt"])
    blocks = [b for seg in pv.encode(vk.params, data + bytes(vk.params.w_bytes - len(data)), salt, 2, band.BAND12) for b in seg]
    assert _run(blocks, vk)["audit_of_record"]["accepted"]                        # the server's encoding, as the twin makes it
    dropped = [bytes(len(b)) if i % 4 == 0 else b for i, b in enumerate(blocks)]
    assert not _run(dropped, vk)["audit_of_record"]["accepted"]


def test_every_role_is_handed_the_codec_and_the_default_stays_p3(toy):
    out, _ = toy
    (out / "salt.json").write_text(json.dumps({"salt": "00" * 24}))
    assert pous_run.serve_spec(_args(out), out)["codec"] == BAND
    assert pous_run._base(_args(out))[-2:] == ["--codec", BAND]
    default = {f.name: f.default for f in dataclasses.fields(pous_run.COMMANDS["e2e"])}["codec"]
    assert default == "p3/v1" == pous_run.reference_codec(default)[0].name


def test_the_verifier_refuses_a_commitment_under_another_codec(toy):
    out, _ = toy
    pous_run.cmd_commit(_args(out))
    with pytest.raises(SystemExit, match=f"committed under {BAND}, not p3/v1"):
        pous_run.cmd_vk(_args(out, codec="p3/v1"))
