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


def test_the_seal_reads_the_port_positions_of_a_step_no_draw_named(cs, tmp_path):
    """The seal holds the step-input port's committed positions to the rows the window ran, at every step
    (`serve_seal.port_positions`, and `seal_process`'s handoff): they are read just before the release, which still keeps
    only the draw's leaves."""
    from verity_vllm.commit.served_record import PORT_NAMES, committed_raw

    pos = bytes(range(40, 48)) * 8
    _host, rc, dev = committed(cs, extra={1: [(PORT_NAMES[1], pos)]})
    before = committed_raw(dev, 1, PORT_NAMES[1])
    assert before is not None and before[0] == pos
    draws = SL.Draws(("pouw",))
    draws.put(0, "pouw", [(2, "model.layers.0.mlp.down_proj", 0, 1)])
    draws.release(0, dev)
    assert committed_raw(dev, 1, PORT_NAMES[1]) == before
    with pytest.raises(RuntimeError, match="draw-gated release"):
        dev.open_range(1, 0, 1)
    SD.dump(dev, rc, tmp_path)
    assert [s["ranges"] for s in json.loads((tmp_path / "store.json").read_text())["steps"]][:2] == [[], []]
    del dev.read_before_release
    with pytest.raises(KeyError, match="draw-gated release"):
        committed_raw(dev, 1, PORT_NAMES[1])


def test_a_released_window_seals_and_its_bundle_holds_exactly_the_draws_leaves(tmp_path):
    import torch

    from tests.pipeline.test_serve_seal import NAMES, ROWS, _ledgers
    from verity_vllm.pipeline import deferred_rows as DR
    from verity_vllm.pipeline import replay_bundle as RB
    from verity_vllm.pipeline import serve_seal as SS

    host, rc, dev = committed(R.NAME)
    dev.weights_tree, dev.model = {"root": (rc.weight_registration or {}).get("root"), "names": []}, torch.nn.Module()
    down = (2, "model.layers.0.mlp.down_proj", 0, dev.chunk)
    draws = SL.Draws(("pouw",))
    draws.put(0, "pouw", [down])
    draws.release(0, dev)
    leds = _ledgers()
    reqs, rows = SS.window_requests(ROWS, NAMES.get, leds.get)
    d = tmp_path / "windows" / "w0"
    s = SS.seal_window(dev, rc, d, rec={"window": 0, "run_id": "t/w0", "steps": len(rc.steps), "passed": True}, reqs=reqs,
                       request_rows=rows, tokens={k: v.streamed for k, v in leds.items()}, sampling={}, label="serve-openai:t")
    assert "error" not in s
    assert DR.of_record(s["bundle"], str(d)).run.run_root == rc.run_root
    store, run = SD.load(RB.path(d / "commit", 0) / "store")
    doc = json.loads((RB.path(d / "commit", 0) / "store" / "store.json").read_text())
    assert [s_["ranges"] for s_ in doc["steps"]] == [[], [], [[0, dev.chunk]]]
    base = {m.name: m.dev_off for m in dev._layouts[2]}["model.layers.0.mlp.down_proj"]
    assert OpenedReader(store, run)(*down) == host[2][base:base + dev.chunk]


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
    assert dev.released["discarded"].startswith("KeyError") and dev.released["kept_bytes"] == 0   # a window whose draws can't
    assert dev.device_pool.held == 0                       # be read keeps nothing: never redrawn, so it would hold the pool forever
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


def _waiting_take(pool, nbytes, key):
    """A take of ``nbytes`` for window ``key`` on a thread, returned once it waits on the full pool."""
    from tests.commit.test_device_pool import _Seen

    pool._cv, got = _Seen(), {}

    def take():
        try:
            got["v"] = pool.take(nbytes, key)
        except RuntimeError as e:
            got["e"] = e
    t = threading.Thread(target=take)
    t.start()
    pool._cv.waiting.wait()
    assert not got
    return t, got


def test_a_failed_run_ends_every_step_waiting_on_the_pool_by_name(cs):
    _host, _rc, dev = committed(cs)
    draws = SL.Draws(PROTOCOLS, dev.device_pool)
    t, got = _waiting_take(dev.device_pool, dev.device_pool.capacity, "next window")
    draws.fail("the server failed: RuntimeError: boom")
    t.join()
    assert "device pool: no release is coming: the server failed: RuntimeError: boom" in str(got["e"])
    with pytest.raises(SL.DrawRefused, match="the server failed"):
        draws.release(0, dev)
    assert draws.discarded == {0: "Refused: window 0: no release: the server failed: RuntimeError: boom"}
    assert dev.device_pool.held == 0 and dev.released["discarded"].startswith("Refused")


def test_a_window_that_reaches_no_release_frees_its_pool_and_is_refused_by_name(cs):
    _host, _rc, dev = committed(cs)
    draws = SL.Draws(PROTOCOLS, dev.device_pool)
    held, cap = dev.device_pool.held, dev.device_pool.capacity
    t, got = _waiting_take(dev.device_pool, cap, "next window")
    draws.refuse(0, "--draws-dir: w0.pouw_h1.json: ValueError: not JSON")
    with pytest.raises(SL.DrawRefused, match="not JSON"):
        draws.release(0, dev)
    t.join()
    assert got["v"][0].numel() == cap and dev.device_pool.held_by("next window") == cap
    assert dev.released["kept_bytes"] == 0 and dev.released["device_freed_bytes"] == held
    with pytest.raises(RuntimeError, match="draw-gated release"):
        dev.open_range(0, 0, 1)
    with pytest.raises(SL.DrawRefused, match="reached no release"):
        draws.put(0, "sampled_proofs", [])
    reads = [(2, "model.layers.0.mlp.down_proj", 0, 1)]
    assert len(draws.refused(0, reads, dev)) == 1 and DROPPED in draws.refused(0, reads, dev)[0]
    assert draws.summary()["discarded"] == {0: draws.discarded[0]}
    assert draws.summary()["device_pool"]["waits"] == 1 and draws.summary()["device_pool"]["peak_bytes"] == cap
    assert draws.discard(0, dev, "again") is None


def test_a_window_whose_close_fails_is_discarded_not_waited_on(cs):
    from verity_vllm.pipeline.serve_commit import CommitWindows

    _host, _rc, dev = committed(cs)
    draws = SL.Draws(PROTOCOLS, dev.device_pool)
    rec = {"window": 4, "error": "RuntimeError: finalize failed"}
    CommitWindows._no_release(SimpleNamespace(release=draws), rec, dev, None)
    assert rec["release"]["discarded"] == "not sealed: RuntimeError: finalize failed" and dev.device_pool.held == 0
    rec = {"window": 5}
    CommitWindows._no_release(SimpleNamespace(release=draws), rec, dev, ("com", "rc", {}))
    assert "release" not in rec                 # a window with a seal job is released before its seal


def opts(**retention):
    r = {"retain": "device", "retain_device_gb": 32.0, "retain_exclude": "", "release_after": "sampled_proofs,pouw_h1", "draws_dir": "",
         **retention}
    return SimpleNamespace(retention=SimpleNamespace(**r), commit=True, scheme=R.NAME, window_gb=32.0, seal=True)


def test_serve_openai_gates_every_device_held_seal_on_named_draws(tmp_path):
    assert SL.retention_refusal(opts()) is None
    assert "--release-after" in SL.retention_refusal(opts(release_after=""))
    assert "needs --retain device and --seal" in SL.retention_refusal(opts(retain="host"))
    assert "distinct protocol names" in SL.retention_refusal(opts(release_after="sampled_proofs,sampled_proofs"))
    assert "distinct protocol names" in SL.retention_refusal(opts(release_after="a/b"))
    rec = SL.retention_record(opts(), SimpleNamespace())
    assert rec["release_after"] == "sampled_proofs,pouw_h1" and rec["draws_dir"] == "OUT/draws"
    draws, d = SL.release_of(opts(), tmp_path)
    assert draws.protocols == ("sampled_proofs", "pouw_h1") and d.path == tmp_path / "draws" and d.path.is_dir()
    assert SL.release_of(opts(retain="host", release_after=""), tmp_path) == (None, None)
    assert SL.release_of(opts(), tmp_path, SimpleNamespace(device_pool="the pool"))[0].pool == "the pool"
    unsealed = opts()
    unsealed.seal = False
    assert "it needs --seal and the subprotocols" in SL.retention_refusal(unsealed)
    from verity_vllm.pipeline import serve_openai as so
    assert so.RL is SL, "serve-openai reaches its draw-gated release under its own alias, which no other module shadows"
