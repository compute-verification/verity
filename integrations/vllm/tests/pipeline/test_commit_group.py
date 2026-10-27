"""A grouped Commit (`pipeline.commit_group`): the Commits of several rows of one engine in one process under one GPU lease.

* each row runs in its own environment, the leased devices kept; its output goes to its own log, and a refused or crashed Commit is
  that row's exit, not the group's;
* a row whose Commit the group can't run (a hot-unsafe or weight-changing flag, a replay that isn't deferred, another row's output,
  an engine other than the first row's) is refused by name before any row runs;
* only the last row to run may release the lease, and after any other row every committer is closed and the engine is as built, or
  the rows after it are refused by name;
* the later rows reuse the first row's pinned pool (every block free), and its weights root and live digests only while the weights
  are the tensors and words they were computed from (`group_reuse`); outside a grouped Commit each is the plain call;
* a process that holds its lease keeps it: a later row's ``acquire`` takes no second lease.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity_vllm.config import UsageError
from verity_vllm.pipeline import cli
from verity_vllm.pipeline import commit_group as CG
from verity_vllm.pipeline import gpu_lease as GL
from verity_vllm.pipeline import group_reuse as GR

TREE = Path(__file__).resolve().parents[2]


def _spec(tmp_path: Path, *rows: str) -> dict:
    return {"schema": CG.SCHEMA, "lead": rows[0], "result": str(tmp_path / "result.json"),
            "rows": [{"row": r, "dir": str(tmp_path / r), "argv": ["--label", r, "--out", str(tmp_path / r / "commit")],
                      "env": {"SWEEP_DIR": str(tmp_path), "VERITY_TIMELINE": str(tmp_path / r / "timeline.jsonl")},
                      "log": str(tmp_path / f"{r}.log")} for r in rows]}


def _args(argv: list[str], **kw) -> SimpleNamespace:
    out = argv[argv.index("--out") + 1]
    return SimpleNamespace(out=out, replay_deferred=True, flip_weight=None, committer="native_collect_v2b", hidden_so="/x/h.so",
                           hidden_ns="ns", tap_sources=None, label=argv[argv.index("--label") + 1], **kw)


def _hot() -> dict:
    return {"jobs": 0, "coms": [], "last": False, "group": {"pool": None, "weights": None, "live": None, "patches": None, "events": []}}


# ---------------------------------------------------------------------------------------------------------------- the spec

def test_a_spec_names_its_rows_their_command_lines_environments_and_logs_and_its_result(tmp_path):
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(_spec(tmp_path, "a", "b")))
    assert [r["row"] for r in CG.load(str(p))["rows"]] == ["a", "b"]
    for bad, why in [({"schema": "x"}, "not a"), ({"schema": CG.SCHEMA, "rows": []}, "no rows"),
                     ({"schema": CG.SCHEMA, "rows": [{"row": "a"}], "result": "r"}, "a row needs"),
                     ({**_spec(tmp_path, "a"), "result": None}, "no result")]:
        p.write_text(json.dumps(bad))
        with pytest.raises(UsageError, match=why):
            CG.load(str(p))


def test_the_command_is_on_the_cli_and_takes_its_spec_and_the_environment(tmp_path):
    assert "commit-group" in cli.COMMANDS
    o = cli.parse("commit-group", [str(tmp_path / "spec.json")], {"X": "1"})
    assert o.spec == str(tmp_path / "spec.json") and o.environ == {"X": "1"}


def test_each_row_runs_in_its_own_environment_with_the_leased_devices_kept(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    monkeypatch.setenv("GPU_LEASE_UUID", "GPU-5")
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)       # leased too (GL.LEASED); a GPU pod's shell sets it
    monkeypatch.setenv("ONLY_BEFORE", "1")
    was = dict(os.environ)
    try:
        cli.set_environment({"CUDA_VISIBLE_DEVICES": "", "SWEEP_DIR": "/s/b", "PATH": was.get("PATH", "")}, keep=GL.LEASED)
        assert dict(os.environ) == {"CUDA_VISIBLE_DEVICES": "5", "GPU_LEASE_UUID": "GPU-5", "SWEEP_DIR": "/s/b", "PATH": was.get("PATH", "")}
    finally:
        os.environ.clear()
        os.environ.update(was)


def test_a_row_on_the_reused_engine_has_what_the_engines_build_wrote_into_the_environment():
    r = {"env": {"SWEEP_DIR": "/s/b", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}}
    env0 = {"PATH": "/bin", "SWEEP_DIR": "/s/a"}
    assert CG.row_environ(env0, r, None) == {"PATH": "/bin", "SWEEP_DIR": "/s/b", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    eng = SimpleNamespace(environ={"CUBLAS_WORKSPACE_CONFIG": ":16:8", "CUBLASLT_WORKSPACE_SIZE": "1"})
    assert CG.row_environ(env0, r, eng) == {"PATH": "/bin", "SWEEP_DIR": "/s/b", "CUBLAS_WORKSPACE_CONFIG": ":16:8",
                                            "CUBLASLT_WORKSPACE_SIZE": "1"}
    assert CG.row_environ(env0, r, SimpleNamespace()) == CG.row_environ(env0, r, None)
    src = Path(CG.__file__).read_text()
    assert 'cli.set_environment(row_environ(env0, r, (cd._HOT or {}).get("eng")), keep=GL.LEASED)' in src


def test_the_devices_a_later_row_keeps_are_the_ones_the_lease_holder_sets():
    assert "(" + ", ".join(f'"{k}"' for k in GL.LEASED) + ")" in GL._HOLDER


# ---------------------------------------------------------------------------------------------------------------- refusals

def _key(a, argv):
    return {"workload": {"model": getattr(a, "model", "m"), "B": 1}, "committer": a.committer}


def test_a_row_the_group_cannot_commit_is_refused_by_name_before_any_row_runs(tmp_path):
    rows = _spec(tmp_path, "a", "b", "c", "d", "e", "f", "g")["rows"]
    rows[5]["argv"] = ["--label", "f", "--out", rows[0]["argv"][-1]]          # row a's output
    entered: list[str] = []

    def parse(argv):
        label = argv[argv.index("--label") + 1]
        if label == "g":
            raise SystemExit(2)
        return _args(argv, **({"model": "other"} if label == "b" else {}), **({"unsafe": True} if label == "c" else {}),
                     **({"flip": True} if label == "d" else {}))

    def args_of(a):
        if getattr(a, "flip", False):
            a.flip_weight = "x:0:1"
        if a.label == "e":
            a.replay_deferred = None
        return a

    why = CG.refusals(rows, lambda argv: args_of(parse(argv)), lambda a: ["--arms (multi-committer arms)"] if getattr(a, "unsafe", False) else [],
                      _key, lambda r: entered.append(r["row"]))
    assert entered == ["a", "b", "c", "d", "e", "f", "g"]
    assert set(why) == {"b", "c", "d", "e", "f", "g"}
    assert why["b"] == "its engine differs from row a's (workload.model)"
    assert why["c"] == "--arms (multi-committer arms)" and "--flip-weight" in why["d"] and "not deferred" in why["e"]
    assert why["f"] == f"it commits into {rows[0]['argv'][-1]}, row a's output" and why["g"] == "its Commit's command line does not parse (exit 2)"


def test_the_engine_key_is_the_hot_workers_engine_identity_but_the_rows_own_workload_fields():
    ident = {"workload": {"row": "a", "model": "m", "max_model_len": 288}, "workload_path": "workloads/a.json", "engine_kwargs": {"x": 1}}
    taps = SimpleNamespace(given=lambda: {"norm": "/x/norm.so"})
    k = CG.engine_key(SimpleNamespace(committer="c", hidden_so="/h.so", hidden_ns="n", tap_sources=taps), lambda argv: json.loads(json.dumps(ident)), [])
    assert k == {"workload": {"model": "m", "max_model_len": 288}, "engine_kwargs": {"x": 1}, "committer": "c", "hidden_so": "/h.so",
                 "hidden_ns": "n", "taps": {"norm": "/x/norm.so"}}


# ---------------------------------------------------------------------------------------------------------------- one row

def _commit(code):
    def run(argv):
        if isinstance(code, BaseException):
            raise code
        if code is not None:
            raise SystemExit(code)
    return run


def test_a_rows_refusal_or_crash_is_its_own_exit(tmp_path):
    r = {"row": "a", "argv": ["x"], "log": str(tmp_path / "a.log")}
    assert CG.run_row(r, _commit(None)) == (0, None) and CG.run_row(r, _commit(3)) == (3, None)
    assert CG.run_row(r, _commit("HOT-KEY-MISMATCH: engine kwargs differ")) == (CG.RC_REFUSED, "HOT-KEY-MISMATCH: engine kwargs differ")
    assert CG.run_row(r, _commit("HOT-UNSAFE: --arms")) == (CG.RC_REFUSED, "HOT-UNSAFE: --arms")
    assert CG.run_row(r, _commit("--flip-weight did not change the element")) == (1, None)
    assert CG.run_row(r, _commit(RuntimeError("boom"))) == (1, None)


def test_a_rows_output_and_exit_message_go_to_its_log_and_the_process_output_comes_back(tmp_path):
    script = ("import os, sys\nfrom verity_vllm.pipeline import commit_group as CG\n"
              "def commit(code):\n    def run(argv):\n        print(f'row output {argv}', flush=True)\n        os.write(2, b'native stderr\\n')\n"
              "        if isinstance(code, BaseException):\n            raise code\n        raise SystemExit(code)\n    return run\n"
              f"a, b = {{'row': 'a', 'argv': ['x'], 'log': {str(tmp_path / 'a.log')!r}}}, {{'row': 'b', 'argv': ['y'], 'log': {str(tmp_path / 'b.log')!r}}}\n"
              "print(CG.run_row(a, commit('--flip-weight did not change the element')), CG.run_row(b, commit(RuntimeError('boom'))))\n")
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=TREE)
    assert r.returncode == 0 and r.stdout == "(1, None) (1, None)\n", r.stderr
    a, b = (tmp_path / "a.log").read_text(), (tmp_path / "b.log").read_text()
    assert "row output ['x']" in a and "native stderr" in a and "--flip-weight did not change the element" in a
    assert "row output ['y']" in b and "RuntimeError: boom" in b and "row output" not in r.stderr


# ---------------------------------------------------------------------------------------------------------------- the group

def _clock():
    t = iter(range(1000))
    return lambda: float(next(t))


def test_the_rows_run_in_order_and_only_the_last_to_run_may_release_the_lease(tmp_path):
    spec = _spec(tmp_path, "a", "b", "c")
    hot, seen, cleaned = _hot(), [], []
    env = {}

    def commit(argv):
        label = argv[argv.index("--label") + 1]
        seen.append((label, hot["jobs"], hot["last"], env["row"]))
        if hot.get("eng") is None:
            hot["eng"] = "engine"
        hot["group"]["events"].append({"job": hot["jobs"], "what": "weights.reused" if hot["jobs"] > 1 else "weights.computed"})

    doc = CG.run(spec, enter=lambda r: env.update(row=r["row"]), parse=_args, unsafe=lambda a: [], key=_key, commit=commit, hot=hot,
                 clean=lambda: cleaned.append(1) or {"ok": True, "why": None}, clock=_clock())
    assert seen == [("a", 1, False, "a"), ("b", 2, False, "b"), ("c", 3, True, "c")] and len(cleaned) == 2
    rows = doc["rows"]
    assert [rows[r]["rc"] for r in "abc"] == [0, 0, 0] and [rows[r]["engine_built"] for r in "abc"] == [True, False, False]
    assert [rows[r]["engine_reused"] for r in "abc"] == [False, True, True] and "after" not in rows["c"]
    assert rows["b"]["reuse"] == [{"job": 2, "what": "weights.reused"}] and rows["a"]["after"] == {"ok": True, "why": None}
    assert json.loads(Path(spec["result"]).read_text()) == doc and doc["schema"] == CG.RESULT_SCHEMA and "ended" in doc


def test_a_refused_row_runs_nothing_and_the_last_row_that_runs_is_the_last(tmp_path):
    spec = _spec(tmp_path, "a", "b", "c")
    hot, seen = _hot(), []
    doc = CG.run(spec, enter=lambda r: None, parse=lambda argv: _args(argv, model="other" if "c" in argv else "m"),
                 unsafe=lambda a: [], key=_key, commit=lambda argv: seen.append((argv[1], hot["last"])), hot=hot,
                 clean=lambda: {"ok": True, "why": None}, clock=_clock())
    assert seen == [("a", False), ("b", True)]
    assert doc["rows"]["c"]["rc"] == CG.RC_REFUSED and doc["rows"]["c"]["refused"].startswith("REFUSED by the grouped Commit: its engine differs")


def test_an_engine_left_unlike_it_was_built_refuses_every_later_row_by_name(tmp_path):
    spec = _spec(tmp_path, "a", "b", "c")
    hot, seen = _hot(), []
    doc = CG.run(spec, enter=lambda r: None, parse=_args, unsafe=lambda a: [], key=_key, commit=lambda argv: seen.append(argv[1]), hot=hot,
                 clean=lambda: {"ok": False, "why": "the engine has unfinished requests"}, clock=_clock())
    assert seen == ["a"]
    for r in "bc":
        assert doc["rows"][r]["refused"] == ("REFUSED by the grouped Commit: after row a the engine is not as it was built "
                                             "(the engine has unfinished requests)")
    assert json.loads(Path(spec["result"]).read_text())["rows"]["c"]["rc"] == CG.RC_REFUSED


def test_after_a_row_its_committers_are_closed_and_the_engine_checked_as_the_hot_worker_checks_it():
    calls: list[str] = []
    done = threading.Event()
    bundle = threading.Thread(target=lambda: (done.wait(), calls.append("bundle written")), name="replay-bundle_0")
    bundle.start()
    done.set()

    class Com:
        def __init__(self, name, fail=False):
            self.name, self.fail = name, fail

        def uninstall(self):
            calls.append(f"{self.name}.uninstall")

        def close(self):
            if self.fail:
                raise RuntimeError("worker gone")
            calls.append(f"{self.name}.close")

    base, extra = object(), SimpleNamespace(name="vllm.forward")
    llm = SimpleNamespace(llm_engine=SimpleNamespace(has_unfinished_requests=lambda: False))
    cd = SimpleNamespace(_HOT={"coms": [Com("c0"), Com("c1", fail=True)], "eng": SimpleNamespace(llm=llm), "base0": {"b": 0}, "cuda0": None,
                               "group": {"patches": (base,)}},
                         assert_pristine=lambda eng, b: calls.append(f"pristine {b}"))
    torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: calls.append("sync"), empty_cache=lambda: calls.append("empty")))
    out = CG.clean(cd, torch, lambda com: [f"{com.name}.pool"], lambda: (base,))
    assert not bundle.is_alive()
    assert calls[0] == "bundle written" and calls[1:] == ["c0.uninstall", "c0.close", "c1.uninstall", "sync", "empty", "pristine {'b': 0}"]
    assert out["ok"] and out["committers_closed"] == 2 and out["released"] == ["c0.pool", "c1.pool"]
    assert out["close_errors"] == ["Com.close: RuntimeError('worker gone')"] and "coms" not in cd._HOT
    cd._HOT.update(coms=[Com("c2")])
    out = CG.clean(cd, torch, lambda com: ["SKIPPED:cmt-worker-alive"], lambda: (base, extra))
    assert not out["ok"] and "outlived its close()" in out["why"] and "1 patch(es) left by the row: ['SimpleNamespace:vllm.forward']" in out["why"]


# ---------------------------------------------------------------------------------------------------------------- reuse

class Pool:
    def __init__(self, mode="exact-registered-shared"):
        self.mode, self.all, self.allocs, self.alloc_s, self.resets = mode, [b"x" * 8, b"y" * 8], 2, 1.5, 0

    @property
    def bytes_total(self):
        return 16

    def reset(self):
        self.resets += 1


def test_the_later_rows_take_the_first_rows_pinned_pool_with_every_block_free(monkeypatch):
    patches = ("p",)
    monkeypatch.setattr(GR.hooks, "live", lambda: patches)
    com0, com1 = SimpleNamespace(pool=Pool()), SimpleNamespace(pool=Pool())
    assert GR.adopt(None, com0) is com0 and GR.adopt({"jobs": 1}, com0).pool is com0.pool       # outside a grouped Commit: unchanged
    hot = _hot()
    hot["jobs"] = 1
    GR.adopt(hot, com0)
    assert hot["group"]["pool"] is com0.pool and hot["group"]["patches"] == patches and com0.pool.resets == 0
    hot["jobs"] = 2
    GR.adopt(hot, com1)
    assert com1.pool is com0.pool and com0.pool.resets == 1 and (com0.pool.allocs, com0.pool.alloc_s) == (0, 0.0)
    assert [e["what"] for e in hot["group"]["events"]] == ["pool.kept", "pool.adopted"] and hot["group"]["events"][1]["blocks"] == 2


def _model():
    torch = pytest.importorskip("torch")
    m = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.LayerNorm(3))
    m.register_buffer("odd", torch.arange(3, dtype=torch.uint8))
    return torch, m


def test_the_fingerprint_moves_with_any_change_to_the_weights_words_or_tensors():
    torch, m = _model()
    fp = GR.fingerprint(m)
    assert GR.fingerprint(m) == fp
    with torch.no_grad():
        m[0].weight[0, 0] += 1                                      # in place: the version counter and the words
    assert GR.fingerprint(m) != fp
    fp = GR.fingerprint(m)
    m[0].weight.data[0, 1] += 1                                     # behind autograd's back: the words alone
    assert GR.fingerprint(m) != fp
    fp = GR.fingerprint(m)
    m[0].bias = torch.nn.Parameter(m[0].bias.detach().clone())      # the same words in another tensor
    assert GR.fingerprint(m) != fp
    with torch.inference_mode():
        t = torch.ones(2, 2)
    inf = torch.nn.Module()
    inf.register_buffer("w", t)
    with pytest.raises(RuntimeError):
        t._version
    assert GR.fingerprint(inf) == GR.fingerprint(inf)               # an inference tensor keeps no version counter


class Com:
    calls = 0

    def __init__(self, model, chunk=256):
        self.model, self.chunk, self._geo_digest = model, chunk, b"\x01\x02"

    def register_weights(self):
        Com.calls += 1
        self.weights_tree = {"root": "ab" * 32, "tensor_roots": [Com.calls]}
        self.weights = {"root": "ab" * 32, "tensors": 4, "wall_s": 2.5}
        return self.weights


def test_the_weights_root_is_registered_once_while_the_weights_are_the_same(monkeypatch, capsys):
    torch, m = _model()
    Com.calls = 0
    first = Com(m)
    assert GR.register_weights(None, first, "host") == first.weights and Com.calls == 1          # outside a grouped Commit
    hot = _hot()
    for job in (1, 2):
        hot["jobs"] = job
        c = Com(m)
        w = GR.register_weights(hot, c, "host")
    assert Com.calls == 2 and c.weights_tree == {"root": "ab" * 32, "tensor_roots": [2]} and w is c.weights
    assert w["wall_s"] == 0.0 and w["reused"]["from_job"] == 1 and w["reused"]["wall_s"] == 2.5 and w["root"] == "ab" * 32
    assert "weights root abababababababab reused from row 1" in capsys.readouterr().out
    hot["jobs"] = 3
    GR.register_weights(hot, Com(m), "device")                       # another hashing mode
    assert Com.calls == 3
    hot["jobs"] = 4
    GR.register_weights(hot, Com(m, chunk=512), "device")            # another chunk
    assert Com.calls == 4
    hot["jobs"] = 5
    with torch.no_grad():
        m[1].weight.mul_(2)
    GR.register_weights(hot, Com(m, chunk=512), "device")            # other weights
    assert Com.calls == 5
    whats = [e["what"] for e in hot["group"]["events"] if e["what"].startswith("weights.") and e["what"] != "weights.fingerprint"]
    assert whats == ["weights.computed", "weights.reused", "weights.computed", "weights.computed", "weights.computed"]


class ModeCom(Com):
    """Hashes in the mode `native_host.register_weights` reads (``VERITY_WEIGHTS_HASH``) and records it as ``hash``."""

    def register_weights(self):
        super().register_weights()
        self.weights["hash"] = os.environ["VERITY_WEIGHTS_HASH"]
        return self.weights


def test_a_reused_weights_root_was_hashed_in_the_rows_own_mode(monkeypatch):
    """The group's one leased process hashes on the device (`gpu_lease.run` sets ``VERITY_WEIGHTS_HASH``) and every row's environment
    keeps it; `commit.main` hands the committer the row's ``--weights-hash`` before it registers, and a later row reuses only a root
    registered in that mode, so a reused root's ``hash`` is the row's ``--weights-hash`` (its versions.json)."""
    src = (TREE / "verity_vllm" / "pipeline" / "commit.py").read_text()
    assert src.index('settings.put("VERITY_WEIGHTS_HASH", args.weights_hash)') < src.index("GR.register_weights(_HOT, com, args.weights_hash)")
    torch, m = _model()
    Com.calls = 0
    env0 = {"VERITY_WEIGHTS_HASH": "device", "CUDA_VISIBLE_DEVICES": "3"}
    eng = SimpleNamespace(environ={"CUBLAS_WORKSPACE_CONFIG": ":16:8", "CUBLASLT_WORKSPACE_SIZE": "1"})
    hot = _hot()
    for job, row in enumerate(_spec(Path("/sweep"), "a", "b", "c")["rows"], start=1):
        hot["jobs"] = job
        mode = CG.row_environ(env0, row, eng)["VERITY_WEIGHTS_HASH"]
        monkeypatch.setenv("VERITY_WEIGHTS_HASH", mode)
        w = GR.register_weights(hot, ModeCom(m), mode)
        assert mode == "device" and w["hash"] == "device"
    assert Com.calls == 1 and w["reused"]["from_job"] == 1
    hot["jobs"] = 4
    monkeypatch.setenv("VERITY_WEIGHTS_HASH", "host")
    assert GR.register_weights(hot, ModeCom(m), "host")["hash"] == "host" and Com.calls == 2


def test_the_live_digests_are_taken_once_while_the_weights_are_the_same():
    torch, m = _model()
    taken: list[int] = []

    def digests(model):
        taken.append(1)
        return {"params": {"0.weight": "aa"}, "n": 4, "wall_s": 1.0}

    assert GR.live_digests(None, m, digests)["n"] == 4 and len(taken) == 1
    hot = _hot()
    hot["jobs"] = 1
    d1 = GR.live_digests(hot, m, digests)
    hot["jobs"] = 2
    d2 = GR.live_digests(hot, m, digests)
    assert len(taken) == 2 and d2 == {**d1, "reused": {"from_job": 1}}
    d2["params"]["0.weight"] = "changed"
    hot["jobs"] = 3
    assert GR.live_digests(hot, m, digests)["params"] == {"0.weight": "aa"}         # the cached copy is the first row's
    with torch.no_grad():
        m[0].bias.zero_()
    hot["jobs"] = 4
    GR.live_digests(hot, m, digests)
    assert len(taken) == 3


# ---------------------------------------------------------------------------------------------------------------- the lease

def test_a_process_that_holds_its_lease_takes_no_second_one(monkeypatch):
    monkeypatch.setattr(GL, "_HELD", [("holder",)])
    monkeypatch.setattr(GL, "cuda_initialised", lambda: pytest.fail("a held lease is not taken again"))
    env = {"GPU_LEASE": "/x/gpu-lease", GL.STARTED: "/row/gpu_lease.started", "CUDA_VISIBLE_DEVICES": "5"}
    GL.acquire(environ=env)
    assert env == {"GPU_LEASE": "/x/gpu-lease", GL.STARTED: "/row/gpu_lease.started", "CUDA_VISIBLE_DEVICES": "5"} and GL._HELD == [("holder",)]


def test_the_later_rows_of_a_grouped_commit_keep_the_first_rows_lease_and_devices(tmp_path):
    from tests.pipeline.test_gpu_lease import commit, fake_lease, released

    env = {"PATH": os.environ["PATH"], "GPU_LEASE": fake_lease(tmp_path)}
    GL.hide_devices(env)
    seen = tmp_path / "seen"
    cmd = commit("holder = GL._HELD[0][0]\nrow = dict(os.environ)\n"
                 "cli.set_environment({k: v for k, v in row.items() if k not in GL.LEASED} | {'CUDA_VISIBLE_DEVICES': '', 'ROW': 'b'}, keep=GL.LEASED)\n"
                 "GL.acquire()\n"
                 f"fds = [f for f in os.listdir('/proc/self/fd') if os.path.realpath('/proc/self/fd/' + f) == {str(tmp_path / '5.lock')!r}]\n"
                 f"json.dump(dict(devices=os.environ['CUDA_VISIBLE_DEVICES'], uuid=os.environ['GPU_LEASE_UUID'], row=os.environ['ROW'], "
                 f"held=len(GL._HELD), same=GL._HELD[0][0] is holder, locks=len(fds)), open({str(seen)!r}, 'w'))\n")
    log = tmp_path / "commit.log"

    def run_tl(to: int, argv: list[str]) -> int:
        with open(log, "a") as f:
            return subprocess.run(argv, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=TREE).returncode

    rc, _ = GL.run(str(tmp_path), str(log), cmd, 600, env, run_tl)
    assert rc == 0, log.read_text()
    assert json.loads(seen.read_text()) == {"devices": "5", "uuid": "GPU-5", "row": "b", "held": 1, "same": True, "locks": 1}
    assert released(tmp_path / "5.lock")
