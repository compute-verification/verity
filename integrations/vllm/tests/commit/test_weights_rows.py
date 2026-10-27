"""The registered weights under rows-sha512 (`NativeHostCommitter.register_weights`, `weights_rows`): the committer's tree is
the one the CPU replay opens against (`cpu_replay.open_weights`), with a 64-byte root the record names under its scheme, a
re-registration reproduces it, and the device path's roots are the host path's byte for byte (CUDA only)."""
from __future__ import annotations

import pytest

from tests.check.opened import committer
from verity_vllm.commit import scheme, step_rows
from verity_vllm.commit.committer import weights_rows as WR
from verity_vllm.pipeline import cpu_replay as CPU

torch = pytest.importorskip("torch")


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(5)
        self.proj = torch.nn.Linear(40, 120, bias=True)
        self.big = torch.nn.Parameter(torch.randint(-100, 100, (step_rows.ROW_BYTES + 77,), dtype=torch.int8, generator=g), requires_grad=False)
        self.register_buffer("scale", torch.tensor(0.5))
        with torch.no_grad():
            self.proj.weight.copy_(torch.randn(120, 40, generator=g).to(torch.bfloat16).float())


def _registered(commit_scheme: str | None, monkeypatch) -> tuple:
    monkeypatch.setenv("VERITY_WEIGHTS_HASH", "host")
    com = committer(commit_scheme=commit_scheme)
    com.model = _Model()
    return com, com.register_weights()


def _open(com, tmp_path, root):
    """Every registered tensor carried in the bundle (no checkpoint), opened against ``root`` as the CPU replay opens it."""
    tree = dict(com.weights_tree)
    reg = dict(list(com.model.named_parameters()) + list(com.model.named_buffers()))
    (tmp_path / "weights" / "extra").mkdir(parents=True, exist_ok=True)
    tree["extra"] = {}
    for i, (name, *_rest) in enumerate(tree["names"]):
        t = reg[name].detach().contiguous().reshape(-1)
        (tmp_path / "weights" / "extra" / f"{i}.bin").write_bytes(t.view(torch.uint8).numpy().tobytes())
        tree["extra"][name] = i
    return CPU.open_weights(tmp_path, tree, root, None, threads=1)


def test_a_rows_sha512_registration_is_a_sha512_tree_the_replay_opens_and_a_reregistration_reproduces(tmp_path, monkeypatch):
    com, w = _registered(step_rows.NAME, monkeypatch)
    tree = com.weights_tree
    assert (w["scheme"], tree["scheme"], len(bytes.fromhex(w["root"])), tree["chunk"]) == (step_rows.NAME,) * 2 + (step_rows.ROOT_BYTES, step_rows.ROW_BYTES)
    assert "key" not in w and len(bytes.fromhex(tree["key"])) == step_rows.KEY_BYTES, "the salt key stays off the record"
    assert CPU.folded_root(tree) == w["root"] == tree["root"]
    words, wc = _open(com, tmp_path, w["root"])
    assert wc["equal"] is True and wc["opened"] == wc["servable"] == len(tree["names"]) == 4, wc
    assert com.register_weights()["root"] == w["root"], "a re-registration takes the registration's key (live_weights_check)"
    _other, w2 = _registered(step_rows.NAME, monkeypatch)
    assert w2["root"] != w["root"], "a fresh registration salts its rows with a fresh key"
    with torch.no_grad():
        com.model.proj.bias[3] += 1
    assert com.register_weights()["root"] != w["root"]


def test_a_vllm_v1_registration_stays_sha256_and_is_refused_under_a_sha512_root_of_record(tmp_path, monkeypatch):
    com, w = _registered(None, monkeypatch)
    assert (w["scheme"], "scheme" in com.weights_tree, len(bytes.fromhex(w["root"]))) == (scheme.NAME, False, 32)
    _words, wc = _open(com, tmp_path, w["root"])
    assert wc["equal"] is True, wc
    _words, wc = _open(com, tmp_path, "ab" * step_rows.ROOT_BYTES)
    assert wc["equal"] is False and "weights tree is vllm-v1's, and the committed weights root is rows-sha512's" in wc["why"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for the device row hash")
@pytest.mark.parametrize("nbytes", [1, 100, step_rows.ROW_BYTES, step_rows.ROW_BYTES + 1, 3 * step_rows.ROW_BYTES - 5, 5 * step_rows.ROW_BYTES])
def test_the_device_rows_root_is_the_host_rows_root(nbytes, monkeypatch):
    wr = WR.WeightsRows(committer(commit_scheme=step_rows.NAME)._geo_digest, bytes(range(64)))
    g = torch.Generator().manual_seed(nbytes)
    t = torch.randint(0, 256, (nbytes + 3,), dtype=torch.uint8, generator=g).cuda()
    entry = ["t", "uint8", [nbytes], nbytes]
    for flat in (t[:nbytes], t[3:]):                               # 16-byte aligned, and not
        assert wr.device_root(2, entry, flat) == wr.host_root(2, entry, flat)
        assert wr.root(2, entry, flat, "device-checked") == (wr.host_root(2, entry, flat), 1)
