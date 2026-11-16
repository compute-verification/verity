"""stage_window.py's choice of a served window's member (`pick`: the first member of a projection's fused block on the kernel
asked, or of its scale read, at the size asked; a refusal names the sizes there are) and its public-body check (`public_body`:
the staged public file's body must be the window's member file, with the digest its record states)."""
from __future__ import annotations

import hashlib
import json

import pytest

from conftest import load

W13 = ('MoeExpertFp8ReadGemm_v1{READ={"fn":"EmbeddingRead_v1{V=256,H=3072}"},K=6144,N=1,G=128,KB=128,SWAP=%s,'
       'DOT={"fn":"HopperE4m3%sDot32_v2"}}')
W2 = ('MoeExpertFp8ReadGemmW_v1{READ={"fn":"EmbeddingRead_v1{V=256,H=4096}"},K=256,N=32,G=128,KB=128,SWAP=%s,'
      'DOT={"fn":"HopperE4m3%sDot32_v2"}}')
MEMBERS = [{"template": "EmbeddingRead_v1{V=256,H=96}", "n": 287},
           {"template": W13 % ("true", "Qgmma"), "n": 256},
           {"template": W13 % ("false", "MmaV2"), "n": 31},
           {"template": W2 % ("false", "MmaV2"), "n": 31},
           {"template": W13 % ("true", "Qgmma"), "n": 256},
           {"template": "EmbeddingRead_v1{V=256,H=4}", "n": 287}]
BODY = b"\x00\x01the member's public body"


@pytest.fixture
def sw(monkeypatch):
    return load(monkeypatch, "stage_window.py")


def test_pick_takes_the_first_member_of_the_kind_and_size(sw):
    assert sw.pick(MEMBERS, "w13", 256, None) == 1
    assert sw.pick(MEMBERS, "w13", 256, 1) == 1
    assert sw.pick(MEMBERS, "w13", 31, 0) == 2
    assert sw.pick(MEMBERS, "w2", 31, None) == 3
    assert sw.pick(MEMBERS, "s13", 287, None) == 0
    assert sw.pick(MEMBERS, "s2", 287, None) == 5


def test_pick_refuses_a_size_or_kernel_the_window_has_not(sw):
    with pytest.raises(SystemExit, match=r"no w13 member of 64 units \(sizes: \[31, 256\]\)"):
        sw.pick(MEMBERS, "w13", 64, None)
    with pytest.raises(SystemExit, match=r"no w2 member of 31 units at SWAP=1 \(sizes: \[\]\)"):
        sw.pick(MEMBERS, "w2", 31, 1)
    with pytest.raises(SystemExit, match="--pick takes"):
        sw.pick(MEMBERS, "w7", 31, None)


def window(tmp_path, recorded: str):
    d = tmp_path / "window"
    d.mkdir()
    (d / "m1-public.bin").write_bytes(BODY)
    (d / "window.json").write_text(json.dumps({"members": [{}, {"public_sha512": recorded}]}))
    return d


def staged(tmp_path, body: bytes):
    pub = tmp_path / "pub.bin"
    pub.write_bytes(b'{"format": "flock-public"}\n' + body)
    return pub


def test_public_body_is_the_windows(sw, tmp_path):
    d = window(tmp_path, hashlib.sha512(BODY).hexdigest())
    assert sw.public_body(d, 1, staged(tmp_path, BODY)) == hashlib.sha512(BODY).hexdigest()
    with pytest.raises(SystemExit, match="member 1's public body"):
        sw.public_body(d, 1, staged(tmp_path, BODY + b"\x00"))


def test_public_body_is_held_to_the_record(sw, tmp_path):
    d = window(tmp_path, hashlib.sha512(b"another body").hexdigest())
    with pytest.raises(SystemExit, match="is not the window's"):
        sw.public_body(d, 1, staged(tmp_path, BODY))
