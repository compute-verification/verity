"""``--weights-file`` (`store.FileStore`): W read from the file network's load sink writes serves what the generator serves,
and a file that isn't its manifest's W is refused before anything is served."""
from __future__ import annotations

import json

import numpy as np
import pytest

import run as RUN
from e2e_h200.engine import Engine
from e2e_h200.shape import TINY
from e2e_h200.store import FileStore, GeneratorStore, Mismatch
from e2e_h200.weights import SW, Weights


@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    d = tmp_path_factory.mktemp("w")
    w = Weights(TINY, SW.DEMO_SEED)
    (d / "w.bin").write_bytes(w.stream(0, w.nbytes))
    (d / "manifest.json").write_text(json.dumps(SW.manifest(TINY, SW.DEMO_SEED, digests=True)))
    return w, d / "w.bin", d / "manifest.json"


def tampered(tmp_path, path, at: int):
    raw = bytearray(path.read_bytes())
    raw[at] ^= 1
    out = tmp_path / "bad.bin"
    out.write_bytes(bytes(raw))
    return out


def test_the_file_serves_every_tensor_the_generator_does(loaded):
    w, path, manifest = loaded
    fs, gen = FileStore(w, path, manifest, workers=4), GeneratorStore(w)
    assert fs.describe()["tensors_checked"] == len(w.t)
    for name, t in w.t.items():
        assert np.array_equal(fs.load(name), gen.load(name)), name
        if t.kind == "s8":
            dst = np.empty(w.dims[name], np.int8)
            assert fs.stage(name, dst) is dst and np.array_equal(dst, gen.stage(name)), name


def test_a_file_that_isnt_the_manifests_w_is_refused(loaded, tmp_path):
    w, path, manifest = loaded
    t = w.t["l0.wv"]
    with pytest.raises(Mismatch, match="l0.wv"):
        FileStore(w, tampered(tmp_path, path, t.offset + t.nbytes - 1), manifest)
    short = tmp_path / "short.bin"
    short.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(Mismatch, match="bytes"):
        FileStore(w, short, manifest)
    with pytest.raises(Mismatch, match="seed"):
        FileStore(Weights(TINY, SW.TEST_SEED), path, manifest)


def test_the_engine_serves_the_file_as_it_serves_the_generator(loaded, monkeypatch):
    _, path, manifest = loaded
    seqs = [[104, 105, 32], [111, 107]]
    monkeypatch.setenv("E2E_H200_SEED", "demo")
    want = Engine(TINY, device="cpu", pool=2, commit="reference").next_tokens(seqs)
    monkeypatch.setenv("E2E_H200_WEIGHTS_FILE", str(path))
    monkeypatch.setenv("E2E_H200_WEIGHTS_MANIFEST", str(manifest))
    e = Engine(TINY, device="cpu", pool=2, commit="reference")
    assert e.name.endswith("/seeddemo") and e.next_tokens(seqs) == want


def test_run_refuses_a_file_that_isnt_its_manifests(loaded, tmp_path):
    w, path, manifest = loaded
    bad = tampered(tmp_path, path, w.t["head"].offset)
    rc = RUN.main(["--out", str(tmp_path / "out"), "--shape", "tiny", "--seed", "demo", "--weights-file", str(bad),
                   "--weights-manifest", str(manifest), "--commit", "reference"])
    assert rc == 5 and not (tmp_path / "out" / "record.json").exists()
