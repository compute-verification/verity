"""`protocol_options.pouw_h1_handoff`, the rehearsal's hand-off: a served H1 window saved by its serving process, its
challenger challenge phase's stream handed to a holder, and the window closed and released by another process on the same
challenge phase, with the verifier's checks as they are for a window closed where it served."""
from __future__ import annotations

import json
import os
import pickle
import shlex
import socket
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from verity_vllm import protocols as PO  # noqa: E402
from verity_vllm.engine import int8_ops  # noqa: E402
from verity_vllm.protocols import pouw  # noqa: E402
from verity_vllm.protocols.pouw import h1 as PH  # noqa: E402
from verity_vllm.protocols.pouw.h1 import challenger as PHC  # noqa: E402
from verity_vllm.protocols.pouw.h1 import handoff as HO  # noqa: E402
from verity_vllm.protocols.pouw.h1 import kept as KP  # noqa: E402
from verity_vllm.protocols.pouw.h1 import prover as PV  # noqa: E402
from verity_vllm.protocols.pouw.h1 import served as SV  # noqa: E402
from verity_vllm.protocols.pouw.h1 import window as PHW  # noqa: E402

from tests.protocols import challenger_host as CH  # noqa: E402
from tests.protocols.test_pouw_h1 import K, Int8Arm, Linear, Model, _entry, _host, _remote, _so  # noqa: E402


@pytest.fixture(autouse=True)
def _linear_is_the_stand_in(monkeypatch):
    monkeypatch.setattr(pouw, "linear_types", lambda: (Linear,))


class ZyArm(Int8Arm):
    """`Int8Arm` as `engine.int8_vllm.Int8PlainLinearMethod` serves under H1: with ``zy_rows`` (which H1's install sets)
    ``int8_ops.linear_zy`` over that many slots, its records handed to ``observe`` as z."""

    zy_rows: int | None = None

    def apply(self, layer, x, bias=None):
        self.calls += 1
        xq, s_a, y, zy = int8_ops.linear_zy(x, layer.weight, layer.weight_scale, self.zy_rows)
        self.observe(layer, xq, s_a, y, zy)
        return y

    def observe(self, layer, xq, s_a, y, z=None):
        pass


def _served(settings, n=40, arm=Int8Arm):
    """Two served steps of a 3-step window under `settings` (its second linear's N `n`, its linears on `arm`), its option's
    hooks off and its release not run: the run, as its serving process holds it before its close."""
    model, runner = Model(Linear(32, 1), Linear(n, 2)), settings["pouw"]["runner"]
    for layer in model.layers:
        layer.quant_method = arm()
    comp = PO.install(model, PO.Context(stage="serve", enabled=(_entry(),), settings=settings))
    g = torch.Generator().manual_seed(9)
    for so, rows in ((_so(("a", [5], 4)), (1, 5)), (_so(("b", [7], 2), ("c", list(range(8, 39)), 3), running={"a": 1}),
                                                    (33, 2))):
        runner.execute_model(so)
        for m, layer in zip(rows, model.layers):
            layer.quant_method.apply(layer, (torch.randn(m, K, generator=g) * 3).bfloat16())
    run = pouw.h1_run(_entry())
    comp.installed[0].release = None
    comp.close()
    return run


def test_a_window_closed_by_another_process_is_the_one_its_session_served(tmp_path, monkeypatch):
    """The serving process saves its window and hands its challenge phase's stream to a holder, and its own end of the stream
    goes; the window resumed from the file on that descriptor closes, opens its last round's coin and its sample, and
    closes the challenge phase: the record is one manifest under the coins the auditor's challenge phase released, in order, and each
    call's A table is under its round's root."""
    host = _host(tmp_path, monkeypatch)
    run = _served(_remote(transcript_dir=str(tmp_path / "w"), served=SV.Served(window=bytes(64))))
    assert len(run.round_log) == 2 and not run.closed
    path = HO.suspend(run, tmp_path / "state")
    assert path.stat().st_mode & 0o777 == 0o600
    holder = HO.Holder(tmp_path / "stream.sock")
    client = run.rounds.challenger
    HO.send_stream(client, holder.path)
    fd = holder.take()
    client._wire.sock.close()
    resumed, rclient = HO.resume(tmp_path / "state", fd)
    assert resumed.round_log == run.round_log and [c.index for c in resumed.calls] == [c.index for c in run.calls]
    resumed.listen(lambda call, a, s_a, w, s: {"listener": "close"} if call.m == 0 and tuple(a.shape) == (0, call.k) else None)
    PHW.finish(resumed, rclient)
    assert resumed.details["challenger"] == {"closed": True} and host.ended() is None
    assert resumed.details["requests"]["a_roots"].startswith("bound") and resumed.static()
    assert [c.step for c in resumed.calls] == [0, 0, 1, 1, 2, 2] and resumed.bound[5] == {"listener": "close"}
    record = json.loads((tmp_path / "w" / PH.CALLS_FILE).read_text())
    assert PHC.manifest_reason(record) is None and len(record["commitment"]["rounds"]) == 3
    assert PHC.trusted_rounds_reason(record["commitment"], host.directory, [host.ledger.path], host.window, host.key,
                                     pins=[host.ledger.pins_path], main=CH.MAIN) is None


def test_a_window_hands_off_once_open_and_unclosed_and_a_table_off_its_root_is_refused(tmp_path, monkeypatch):
    """Only a window open on the auditor's challenge phase and not yet closed hands off, a tensor left in its state isn't saved,
    and a saved window whose round's slot owners were changed rebuilds a slot table its round's root isn't over, which
    `resume` refuses."""
    host = _host(tmp_path, monkeypatch)
    run = _served(_remote())
    HO.suspend(run, tmp_path / "a")
    a, _b = socket.socketpair()
    run.owners[1][0] = None
    HO.suspend(run, tmp_path / "b")
    with pytest.raises(ValueError, match="round 1's request or slot table, rebuilt, is not the one its root is over"):
        HO.resume(tmp_path / "b", a.fileno())
    a.close()
    run.stray = torch.zeros(2, dtype=torch.int8)
    with pytest.raises(ValueError, match="tensor on cpu in the window's saved state"):
        HO.suspend(run, tmp_path / "d")
    del run.stray
    run.close()
    with pytest.raises(ValueError, match="before its close"):
        HO.suspend(run, tmp_path / "c")
    run.rounds.challenger._wire.sock.close()
    assert host.ended() is not None


@pytest.mark.slow
def test_the_close_stage_completes_the_serving_stages_record(tmp_path, monkeypatch):
    """`pouw_h1_gpu.py --close-from` on a handed-off window and the record its serving phase left: the window closed,
    measured and released on the challenge phase's stream, and the record (``pouw_h1_gpu.json``) completed with its commitment,
    the challenge phase closed and each call's A table under its round's root."""
    from tests.protocols import pouw_h1_gpu as GPU
    host, out = _host(tmp_path, monkeypatch), tmp_path / "out"
    run = _served(_remote(transcript_dir=str(out), served=SV.Served(window=bytes(64))), n=64, arm=ZyArm)
    HO.suspend(run, tmp_path / "state")
    fd = os.dup(run.rounds.challenger._wire.sock.fileno())
    run.rounds.challenger._wire.sock.close()
    out.mkdir()
    (out / GPU.SERVING).write_text(json.dumps({
        "plain": {"static": {"ok": True}}, "engine": {"enforce_eager": True},
        "h1": {"same_tokens": True, "static_schedule": {"ok": True}, "hook_bad": [], "handoff": {"requests": 3},
               "serve": {"tokens_per_s": 1.0}, "decode_step_s_median": None, "tokens_digest": "0" * 16, "step_s": []}}))
    GPU.main(["--out", str(out), "--close-from", str(tmp_path / "state"), "--stream-fd", str(fd)])
    doc = json.loads((out / GPU.RECORD).read_text())
    h = doc["h1"]
    assert h["challenger"] == {"closed": True} and host.ended() is None and doc["close_argv"]
    assert h["requests"]["a_roots"].startswith("bound") and h["requests"]["admitted"] == 3 and not h["hook_bad"]
    assert h["calls"] == 6 and h["layout_static"] and h["commitment"]["draw"]["coin"] and h["served"]["committed"]
    record = json.loads((out / PH.CALLS_FILE).read_text())
    assert record["commitment"] == h["commitment"] and PHC.manifest_reason(record) is None


def test_a_gate_verb_reaches_the_gate_over_ssh_word_for_word():
    """The live gate's forced command splits ssh's remote command line with ``shlex.split``: a reserve's ``--families``
    (JSON, with spaces) arrives as one word, and a local gate gets the words as they are."""
    from tests.protocols import pouw_h1_gpu as GPU
    ssh = ["ssh", "-i", "k", "-o", "BatchMode=yes", "research@host"]
    verb = ["reserve", "--peer", "vy-nebius-2", "--families", PHC.families(18)]
    got = GPU.gate_argv(ssh, verb)
    assert got[:len(ssh)] == ssh and shlex.split(" ".join(got[len(ssh):])) == verb
    assert GPU.gate_argv(["python", "-I", "local_gate.py", "d"], verb) == ["python", "-I", "local_gate.py", "d", *verb]


def test_the_stream_outlives_the_process_that_handed_it_over(tmp_path):
    """A child process hands a stream's descriptor to the holder and exits; the holder's descriptor still reaches the
    stream's other end, so the challenge phase isn't closed by the serving process's exit."""
    here, there = socket.socketpair()
    holder = HO.Holder(tmp_path / "stream.sock")
    child = ("import socket, sys\nfrom types import SimpleNamespace as N\n"
             "from verity_vllm.protocols.pouw.h1 import handoff as HO\n"
             "s = socket.socket(fileno=int(sys.argv[1]))\n"
             "HO.send_stream(N(_wire=N(sock=s)), sys.argv[2])\n")
    subprocess.run([sys.executable, "-c", child, str(here.fileno()), str(holder.path)], check=True,
                   pass_fds=(here.fileno(),))
    here.close()
    fd = holder.take()
    with socket.socket(fileno=fd) as got:
        got.sendall(b"round 2")
        assert there.recv(7) == b"round 2"


def test_every_served_row_of_a_saved_window_rebuilds_from_the_provers_directory(tmp_path, monkeypatch):
    """The served trees a window saves keep no rows, and `pouw_h1_kept.kept_row` rebuilds every row of each
    of its step A tables, z tables and its ``b/<K>`` from the prover's directory and the saved state, each committing to
    its tree's ``b || c`` under the tree's own salt; a kept A code changed afterwards rebuilds a row that isn't."""
    from verity.core.protocols.sampled_proofs.one_stage import registered as G
    _host(tmp_path, monkeypatch)
    prover = tmp_path / "prover"
    run = _served(_remote(transcript_dir=str(tmp_path / "w"), served=SV.Served(window=bytes(64), prover=str(prover))),
                  n=64, arm=ZyArm)
    assert not run.served.refused, run.served.refused
    try:
        calls = [c.summary() for c in run.calls]
        HO.suspend(run, prover / PV.STATE)
        with open(prover / PV.STATE / HO.FILE, "rb") as f:
            state = pickle.load(f)["state"]
        trees = state["served"][1]["registered"]
        values = [v for v in trees if v.split("/", 1)[1].startswith(("a/", "zy@", "b/"))]
        assert {v.split("/", 1)[1].split("@")[0].split("/")[0] for v in values} == {"a", "zy", "b"}

        def bc(tree, j, words):
            return G.Registered.commit(tree.program, tree.value, [words], [tree.salts[j]], tree.word_bits, G.SCHEMA_V2,
                                       tree.window, tree.produced).bcs[0]
        for v in values:
            tree = trees[v]
            with pytest.raises(LookupError):
                tree.rows[0]
            for j in range(tree.leaves):
                assert bc(tree, j, KP.kept_row(prover, state, calls, v, j)) == tree.bcs[j], (v, j)
        a = f"{state['served'][1]['prefix']}a/{K}@0"
        kept = prover / PV.CALLS / "0.npz"
        with np.load(kept) as f:
            codes, s_a, records = f["a"].copy(), f["s_a"], f["records"]
        codes.reshape(-1)[0] ^= 1
        PV.keep_call(prover, 0, codes, s_a, records)
        assert bc(trees[a], 0, KP.kept_row(prover, state, calls, a, 0)) != trees[a].bcs[0]
        with pytest.raises(ValueError, match="no kept value rebuilds it"):
            KP.kept_row(prover, state, calls, "h1/pub@0", 0)
    finally:
        run.close()
        run.rounds.challenger._wire.sock.close()
