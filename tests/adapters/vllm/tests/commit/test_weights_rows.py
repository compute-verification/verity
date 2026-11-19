"""The registered weights under rows-sha512 (`NativeHostCommitter.register_weights`, `weights_rows`): the committer's tree is
the one the CPU replay opens against (`cpu_replay.open_weights`), with a 64-byte root the record names under its scheme, a
re-registration reproduces it, and the device path's roots are the host path's byte for byte (CUDA only)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.check.opened import committer
from verity_vllm.commit import step_rows
from verity_vllm.commit.committer import weights_rows as WR
from verity_vllm.pipeline import cpu_replay as CPU

if TYPE_CHECKING:
    import torch
else:
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


def _registered(monkeypatch) -> tuple:
    monkeypatch.setenv("VERITY_WEIGHTS_HASH", "host")
    com = committer()
    com.model = _Model()
    return com, com.register_weights()


def _open(com, tmp_path, root):
    """Every registered tensor carried in the bundle (no checkpoint), opened against ``root`` as the CPU replay opens it."""
    from verity.commitments import salt_file as SF

    tree = dict(com.weights_tree)
    SF.write(CPU.salts_path(tmp_path), com.weights_salts)
    reg = dict(list(com.model.named_parameters()) + list(com.model.named_buffers()))
    (tmp_path / "weights" / "extra").mkdir(parents=True, exist_ok=True)
    tree["extra"] = {}
    for i, (name, *_rest) in enumerate(tree["names"]):
        t = reg[name].detach().contiguous().reshape(-1)
        (tmp_path / "weights" / "extra" / f"{i}.bin").write_bytes(t.view(torch.uint8).numpy().tobytes())
        tree["extra"][name] = i
    return CPU.open_weights(tmp_path, tree, root, None, threads=1)


def test_a_rows_sha512_registration_is_a_sha512_tree_the_replay_opens_and_a_reregistration_reproduces(tmp_path, monkeypatch):
    com, w = _registered(monkeypatch)
    tree = com.weights_tree
    assert (w["scheme"], tree["scheme"], len(bytes.fromhex(w["root"])), tree["chunk"]) == (step_rows.NAME,) * 2 + (step_rows.ROOT_BYTES, step_rows.ROW_BYTES)
    assert not {"key", "salts"} & (w.keys() | tree.keys()), "the salts stay off the record and the tree"
    assert {n: len(ys) for n, ys in com.weights_salts.items()} == {n: step_rows.SALT_BYTES * step_rows.tensor_rows(nb) for n, _d, _s, nb in tree["names"]}
    assert CPU.folded_root(tree) == w["root"] == tree["root"]
    words, wc = _open(com, tmp_path, w["root"])
    assert wc["equal"] is True and wc["opened"] == wc["servable"] == len(tree["names"]) == 4, wc
    assert com.register_weights(replay=True)["root"] == w["root"], "a replay re-registers under the recorded salts (live_weights_check)"
    _other, w2 = _registered(monkeypatch)
    assert w2["root"] != w["root"], "a fresh registration draws fresh salts"
    with torch.no_grad():
        com.model.proj.bias[3] += 1
    assert com.register_weights(replay=True)["root"] != w["root"]


def test_every_new_registration_draws_its_own_salts_even_on_the_same_committer(monkeypatch):
    com, w = _registered(monkeypatch)
    s1 = dict(com.weights_salts)
    w2 = com.register_weights()
    assert not {n for n, ys in com.weights_salts.items() if ys == s1[n]} and w2["root"] != w["root"], "a new registration never reuses a salt"
    assert com.register_weights(replay=True)["root"] == w2["root"], "a replay takes the latest registration's salts"


def test_a_registration_s_salt_file_replays_it(tmp_path, monkeypatch):
    """The bundle's salt file (`replay_bundle.write_store_files`) is the registration's: a committer given it re-registers the root."""
    from verity.commitments import salt_file as SF

    com, w = _registered(monkeypatch)
    SF.write(tmp_path / "w.salts", com.weights_salts)
    again = committer()
    again.model = com.model
    again.weights_salts = {n: SF.read(tmp_path / "w.salts", n) for n in SF.params(tmp_path / "w.salts")}
    assert again.register_weights(replay=True)["root"] == w["root"]


def test_a_replay_with_no_recorded_salts_is_refused(monkeypatch):
    monkeypatch.setenv("VERITY_WEIGHTS_HASH", "host")
    com = committer()
    com.model = _Model()
    with pytest.raises(ValueError, match="this committer holds none"):
        com.register_weights(replay=True)
    com.weights_salts = {"scale": b"x" * 192}
    with pytest.raises(ValueError, match="salt file holds none for it"):
        com.register_weights(replay=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for the device row hash")
@pytest.mark.parametrize("nbytes", [1, 100, step_rows.ROW_BYTES, step_rows.ROW_BYTES + 1, 3 * step_rows.ROW_BYTES - 5, 5 * step_rows.ROW_BYTES])
def test_the_device_rows_root_is_the_host_rows_root(nbytes, monkeypatch):
    wr = WR.WeightsRows(committer()._geo_digest)
    g = torch.Generator().manual_seed(nbytes)
    t = torch.randint(0, 256, (nbytes + 3,), dtype=torch.uint8, generator=g).cuda()
    entry = ["t", "uint8", [nbytes], nbytes]
    ys = step_rows.fresh_tensor_salts(nbytes)
    for flat in (t[:nbytes], t[3:]):                               # 16-byte aligned, and not
        assert wr.device_root(2, entry, flat, ys) == wr.host_root(2, entry, flat, ys)
        assert wr.root(2, entry, flat, "device-checked", ys) == (wr.host_root(2, entry, flat, ys), 1)
