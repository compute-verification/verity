"""serve-openai's draw-gated release (`pipeline.serve_release`): a window's device-held values are released only once
every named subprotocol has drawn it; the union of the leaves the drawn blocks open is kept, inputs an unsampled block wrote
included, and opens against the pre-coin root after the drop; a later draw, or a read of a dropped leaf, is refused by name."""
from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from tests.commit.test_device_pool import committed, host_leaf
from verity_vllm.commit import step_rows as R
from verity_vllm.commit import store_dump as SD
from verity_vllm.commit.native_ranges import DROPPED
from verity_vllm.commit.opened import OpenedReader
from verity_vllm.pipeline import serve_release as SL

PROTOCOLS = ("sampled_proofs", "pouw_h1")


@pytest.fixture(params=[None, R.NAME], ids=["vllm-v1", R.NAME])
def cs(request):
    return request.param


class _Waits(threading.Condition):
    """A condition that counts each wait as it starts."""

    def __init__(self):
        super().__init__()
        self.entered = threading.Semaphore(0)

    def wait(self, timeout=None):
        self.entered.release()
        return super().wait(timeout)


def test_no_release_before_the_last_protocols_draw(cs):
    _host, _rc, dev = committed(cs)
    draws = SL.Draws(PROTOCOLS)
    draws._cv = _Waits()
    got = {}
    t = threading.Thread(target=lambda: got.update(rec=draws.release(0, dev)))
    t.start()
    draws._cv.entered.acquire()
    draws.put(0, "sampled_proofs", [(2, "model.layers.0.mlp.down_proj", 0, 1)])
    draws._cv.entered.acquire()                 # woken by the first draw, and waiting again for the second
    assert "rec" not in got and getattr(dev, "released", None) is None and dev.device_pool.held > 0
    draws.put(0, "pouw_h1", [])
    t.join()
    assert got["rec"]["protocols"] == list(PROTOCOLS) and dev.released is not None and dev.device_pool.held == 0


def test_the_union_is_kept_with_an_input_an_unsampled_block_wrote_and_opens_against_the_pre_coin_root(cs, tmp_path):
    host, rc, dev = committed(cs)
    C = dev.chunk
    down = (2, "model.layers.0.mlp.down_proj", 0, C)
    written_by_unsampled = (0, "model.embed_tokens", C, 2 * C)      # the drawn block's input, from a block nobody drew
    h1 = (0, "model.layers.0.self_attn.qkv_proj", 10, 20)
    draws = SL.Draws(PROTOCOLS)
    draws.put(0, "sampled_proofs", [down, written_by_unsampled])
    draws.put(0, "pouw_h1", [h1])
    rec = draws.release(0, dev)
    want = SL.leaves_of(dev, [down, written_by_unsampled, h1])
    assert rec["union_leaves"] == sum(len(v) for v in want.values()) and rec["kept_bytes"] == rec["union_leaves"] * C
    assert rec["leaves_by_protocol"] == {"sampled_proofs": 2, "pouw_h1": 1}
    SD.dump(dev, rc, tmp_path)
    store, run = SD.load(tmp_path)
    assert [s.root for s in run.steps] == [s.root for s in rc.steps]
    base = {(s, m.name): m.dev_off for s, metas in dev._layouts.items() for m in metas}
    for com in (dev, store):
        reader = OpenedReader(com, rc)
        for step, name, lo, hi in (down, written_by_unsampled, h1):
            b = base[step, name]
            assert reader(step, name, lo, hi) == host[step][b + lo:b + hi]
        assert reader.stats["failed"] == 0
        for s, leaves in want.items():
            for i in leaves:
                assert bytes(com.open_range(s, i, i + 1).value) == host_leaf(host, C, s, i)
        with pytest.raises(RuntimeError, match="draw-gated release"):
            com.open_range(1, 0, 1)                 # no draw read step 1


def test_a_later_draw_or_a_read_of_a_dropped_leaf_is_refused_by_name(cs):
    _host, rc, dev = committed(cs)
    draws = SL.Draws(("sampled_proofs",))
    draws.put(0, "sampled_proofs", [(2, "model.layers.0.mlp.down_proj", 0, 1)])
    draws.release(0, dev)
    assert draws.refused(0, [(2, "model.layers.0.mlp.down_proj", 0, 1)], dev) == []
    lost = draws.refused(0, [(1, "runner.sampler", 0, 4), (2, "model.layers.0.mlp.down_proj", 0, 1)], dev)
    assert len(lost) == 1 and lost[0].startswith("runner.sampler [0, 4) at step 1") and DROPPED in lost[0]
    with pytest.raises(SL.DrawRefused, match="was released"):
        draws.put(0, "sampled_proofs", [(1, "runner.sampler", 0, 4)])
    with pytest.raises(RuntimeError, match="draw-gated release"):
        OpenedReader(dev, rc)(1, "runner.sampler", 0, 4)


def test_a_draw_is_never_redrawn_names_only_its_protocols_and_reads_only_what_was_committed(cs):
    _host, _rc, dev = committed(cs)
    draws = SL.Draws(PROTOCOLS)
    draws.put(0, "pouw_h1", [])
    with pytest.raises(SL.DrawRefused, match="drawn by pouw_h1 already"):
        draws.put(0, "pouw_h1", [])
    with pytest.raises(SL.DrawRefused, match="not a protocol the release waits for"):
        draws.put(0, "pous", [])
    with pytest.raises(ValueError, match="0 <= lo < hi"):
        draws.put(0, "sampled_proofs", [(0, "model.embed_tokens", 4, 4)])
    draws.put(0, "sampled_proofs", [(0, "model.layers.9.mlp.down_proj", 0, 1)])
    with pytest.raises(KeyError, match="no committed member by that name"):
        draws.release(0, dev)
    assert getattr(dev, "released", None) is None          # a window whose draws can't be read is not released
    with pytest.raises(KeyError, match="did not commit"):
        SL.leaves_of(dev, [(9, "model.embed_tokens", 0, 1)])


def test_the_draws_dir_puts_each_file_once_and_a_bad_one_refuses_its_window_by_name(cs, tmp_path):
    _host, _rc, dev = committed(cs)
    draws = SL.Draws(PROTOCOLS)
    d = SL.DrawDir(tmp_path / "draws", draws)
    for p, reads in (("sampled_proofs", [[2, "model.layers.0.mlp.down_proj", 0, 1]]), ("pouw_h1", [])):
        (d.path / f"w0.{p}.json").write_text(json.dumps({"window": 0, "protocol": p, "reads": reads}))
    (d.path / "w1.sampled_proofs.json").write_text(json.dumps({"window": 0, "protocol": "sampled_proofs", "reads": []}))
    (d.path / "w0.pouw_h1.json.tmp").write_text("{")
    assert d.scan() == 3 and d.scan() == 0
    assert draws.release(0, dev)["union_leaves"] == 1
    with pytest.raises(SL.DrawRefused, match="w1.sampled_proofs.json: ValueError: its body names window 0"):
        draws.release(1, dev)
    assert draws.summary()["refused"] == {1: draws._refused[1]}


def test_a_failed_run_ends_every_waiting_release():
    draws = SL.Draws(PROTOCOLS)
    draws._cv = _Waits()
    got = {}
    t = threading.Thread(target=lambda: got.update(e=pytest.raises(SL.DrawRefused, draws.wait, 3)))
    t.start()
    draws._cv.entered.acquire()
    draws.fail("the server stopped")
    t.join()
    assert "window 3: no release: the server stopped" in str(got["e"].value)


def opts(**retention):
    r = {"retain": "device", "retain_device_gb": 32.0, "retain_exclude": "", "release_after": "sampled_proofs,pouw_h1", "draws_dir": "",
         **retention}
    return SimpleNamespace(retention=SimpleNamespace(**r), commit=True, scheme=R.NAME, window_gb=32.0, seal=True)


def test_serve_openai_gates_every_device_held_seal_on_named_draws(tmp_path):
    from verity_vllm.pipeline import serve_seal as SS

    assert SS.retention_refusal(opts()) is None
    assert "--release-after" in SS.retention_refusal(opts(release_after=""))
    assert "needs --retain device and --seal" in SS.retention_refusal(opts(retain="host"))
    assert "distinct protocol names" in SS.retention_refusal(opts(release_after="sampled_proofs,sampled_proofs"))
    assert "distinct protocol names" in SS.retention_refusal(opts(release_after="a/b"))
    rec = SS.retention_record(opts(), SimpleNamespace())
    assert rec["release_after"] == "sampled_proofs,pouw_h1" and rec["draws_dir"] == "OUT/draws"
    draws, d = SL.release_of(opts(), tmp_path)
    assert draws.protocols == ("sampled_proofs", "pouw_h1") and d.path == tmp_path / "draws" and d.path.is_dir()
    assert SL.release_of(opts(retain="host", release_after=""), tmp_path) == (None, None)
    from verity_vllm.pipeline import serve_openai as so
    assert so.RL is SL, "serve-openai reaches its draw-gated release under its own alias, which no other module shadows"
