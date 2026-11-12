"""training_slowdown.data: the stream and the held-out sets, read only when they are the registry's."""
import dataclasses
import json

import numpy as np
import pytest
import torch
from synthetic import MERGES, PARTS, SETS, sha256, tiny_data

from training_slowdown import policy, starts
from training_slowdown.data import Data, Stream, sample_slice, held_out, windows

CTX = 16


def test_a_pass_of_the_stream_reads_every_window_of_every_part_once(tmp_path):
    data = tiny_data(tmp_path)
    parts = [np.fromfile(data.train / "parts" / f"{i:03d}.bin", dtype=np.uint16) for i in range(len(PARTS))]
    every = {p[j * CTX: j * CTX + CTX + 1].tobytes() for p in parts for j in range((len(p) - 1) // CTX)}
    s = Stream(data, seed=3, ctx=CTX)
    assert len(s.order) == len(every) == sum((n - 1) // CTX for n in PARTS)
    batch = 8
    assert len(every) % batch == 0, "a pass reads whole batches: a batch past its end starts the order again"
    rows = [s.next(batch) for _ in range(len(every) // batch)]
    seen = [r.numpy().astype(np.uint16).tobytes() for b in rows for r in b]
    assert sorted(seen) == sorted(every)
    assert torch.equal(s.next(batch), rows[0])  # the order starts again
    assert torch.equal(Stream(data, seed=3, ctx=CTX).next(batch), rows[0])
    assert not torch.equal(Stream(data, seed=4, ctx=CTX).next(batch), rows[0])


def test_a_stream_that_isnt_the_registrys_is_refused(tmp_path):
    data = tiny_data(tmp_path)
    man_path = data.train / "manifest.json"
    man = json.loads(man_path.read_text())
    with pytest.raises(SystemExit, match="stream manifest"):
        Stream(dataclasses.replace(data, manifest_sha256="0" * 64), seed=0, ctx=CTX)
    other = json.dumps(man | {"selection": {"sha256": "1" * 64}}).encode()
    man_path.write_bytes(other)
    with pytest.raises(SystemExit, match="selection"):
        Stream(dataclasses.replace(data, manifest_sha256=sha256(other)), seed=0, ctx=CTX)
    man_path.write_text(json.dumps(man))
    part = data.train / "parts" / "001.bin"
    part.write_bytes(part.read_bytes()[:-2])
    with pytest.raises(SystemExit, match="isn't its manifest's"):
        Stream(dataclasses.replace(data, manifest_sha256=sha256(man_path.read_bytes())), seed=0, ctx=CTX)


def test_a_held_out_set_is_read_only_when_its_tokens_are_the_registrys(tmp_path):
    data = tiny_data(tmp_path)
    toks, man = held_out(data, "final")
    assert len(toks) == SETS["final"] and man["bytes"] > 0
    f = data.sets["search"] / "tiny.bin"
    b = bytearray(f.read_bytes())
    b[10] ^= 1
    f.write_bytes(bytes(b))
    with pytest.raises(SystemExit, match="search set"):
        held_out(data, "search")
    with pytest.raises(SystemExit, match="no held-out split"):
        held_out(data, "val")


def test_every_target_is_in_one_window():
    n = 2 * 1024 + 50
    full, tail = windows(n)
    assert (full, tail) == ([0, 1024], [2048])
    targets = [t for s in full + tail for t in range(s + 1, min(s + 1025, n))]
    assert targets == list(range(1, n))


def test_a_slice_is_sampled_from_its_coin_and_the_weights_digest():
    a = sample_slice(50, "w1", size=10)
    assert len(a["coin"]) == 32 and len(a["windows"]) == 10 == len(set(a["windows"]))
    assert sample_slice(50, "w1", a["coin"], size=10) == a
    assert sample_slice(50, "w2", a["coin"], size=10)["windows"] != a["windows"]
    assert len(sample_slice(5, "w1", size=10)["windows"]) == 5


def test_token_bytes_are_the_tokenizers_byte_level_vocabulary(tmp_path):
    data = tiny_data(tmp_path)
    tb = data.token_bytes()
    assert tb[0] == 0 and set(tb[1:257]) == {1} and list(tb[257:]) == [len(m) for m in MERGES]
    (data.train / "tokenizer.json").write_text("{}")
    with pytest.raises(SystemExit, match="tokenizer"):
        data.token_bytes()


def test_the_registry_states_data_v3_and_its_six_starts():
    v3 = Data.of("v3")
    assert v3.tokenizer == "bpe32k" and v3.vocab == 32000 and set(v3.sets) == {"search", "final"}
    with pytest.raises(SystemExit, match="no data"):
        Data.of("v9")
    reg = starts.registry()
    assert list(reg["starts"]) == ["s-t20", "s-t200", "m-t20", "m-t200", "l16-t20", "xl-t20"]
    for name in reg["starts"]:
        s = starts.start(name)
        assert s.data == "v3" and s.path == f"starts/{name}" and s.art.startswith("art:")
        assert policy.parameters(s.shape) == s.shape["parameters"]
    with pytest.raises(SystemExit, match="no start"):
        starts.start("s-t2")
