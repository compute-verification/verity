"""serve-openai's Pearl+ arm (`serve_pearl`) on CPU, over test_serve_openai's stand-in engine and committer: one Pearl+
epoch a Commit window, its composition Pearl-C beside sampled proofs cut at Pearl-C's sites, each engine step's forward
one pass ended at the model runner the committer patched first (`pearl_c.passes.hook_steps`' ``runner``; the class's
wrapper, under the committer's patch, never runs), and each window's ``pearl/w<k>/served.json`` and ``coins.json``, from
which Pearl+'s window record W is written (`pearl_c.window.write`).

The numbering is test_pearltie's toy's: the toy's class (1 slot, a prompt of 27, 2 outputs) served on one layer's 4
linears (q/k/v, o, gate/up and down), so round s is step s, its call c the c-th linear at index s · 4 + c of the epoch,
with `window_class.PearlTables`' m₁₆ A rows and real rows, and each call's A rows (its x widened to FP32, zero rows up to
m₁₆) and y committed as Pearl+'s reference commits them (`audit.commit_rows`, `served_y.commit_y`) from the x the forward
handed the linear and the y it served. The device run is a stand-in on the real pass numbering and ends (`EpochCalls`,
`PassEnds`), its y the BF16 product. A window whose pass is short of a linear's call, or whose passes aren't its steps, is
that window's visible failure, and serve-openai refuses what the arm can't serve before the model loads. A window's epoch
is made ahead of it (`PearlWindows.make_ahead`: in-process coins' always, a live session's only when asked) in the order
its open makes it, and the open then only hooks the model.
"""
from __future__ import annotations

import itertools
import json
import logging
import random
import shlex
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from verity.core.protocols.pouw import verifier
from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.pouw.audit import HM96, Workload, activation_binding, commit_rows, weights_root
from verity.core.protocols.pouw.coins import Challenger
from verity.core.protocols.pouw.protocol import Matmul, Shape
from verity_vllm.engine import hooks
from verity_vllm.pipeline import cli
from verity_vllm.pipeline import serve_openai as so
from verity_vllm.pipeline import serve_pearl as SPL
from verity_vllm.program.predict.window_class import PearlTables, WindowClass
from verity_vllm.protocols import pouw
from verity_vllm.protocols.pouw import pearl_c as PPD
from verity_vllm.protocols.pouw.h1 import window as PHW
from verity_vllm.protocols.pouw.pearl_c import passes as PP
from verity_vllm.protocols.pouw.pearl_c import served_y as SY
from verity_vllm.protocols.pouw.pearl_c import window as PWR
from verity_vllm.protocols.pouw.pearl_c.ends import PassEnds, _Pass
from verity_vllm.protocols.pouw.pearl_c.manifest import SESSION_KEYS
from verity_vllm.protocols.pouw.pearl_c.parts import Parts

from tests.pipeline import predicted
from tests.pipeline.test_serve_openai import CERTIFIER_BODY, Runner, Serving, records

CLASS, PEARL = WindowClass(1, 27, 2), PearlTables(SPL.SCHEME, words=64)
#: a round's calls: the layer's q/k/v, o, gate/up and down
CALLS = 4
HIDDEN, INTER = 256, 1024
#: each linear's (n, k), in Pearl-C's domain on the H100 kernels
SHAPES = ((2 * HIDDEN, HIDDEN), (HIDDEN, HIDDEN), (2 * INTER, HIDDEN), (HIDDEN, INTER))
CONFIG = {"pouw": {"device": "cuda", "kernel": "k"}}
#: a step's pass's seconds on the stand-in under the fixture's clock, a tick a reading: its forward, to the end's one
#: switch, and the next pass's begin
STEP_PARTS = {"forward": 1.0, "next": 1.0}
PROMPTS = ([(7 * i + 3) % 500 for i in range(CLASS.prompt)], [(5 * i + 11) % 500 for i in range(CLASS.prompt)])
RUNS: list = []


class Apply:
    def apply(self, layer, x, bias=None):
        return x @ layer.weight.T


class Linear:
    def __init__(self, n, k, rng):
        self.weight = (0.05 * torch.from_numpy(rng.standard_normal((n, k), np.float32))).to(torch.bfloat16)
        self.quant_method = Apply()


class Model:
    def __init__(self):
        rng = np.random.default_rng(0)
        self.layers = [Linear(n, k, rng) for n, k in SHAPES]

    def modules(self):
        return iter(self.layers)


class Forward(Runner):
    """Model Runner V2 as the committer and Pearl+ read it: ``execute_model`` runs the toy's forward on the step's rows,
    each linear through its ``quant_method.apply`` (but the down projection at a step in ``skip``), recording each call's
    x and the y it served."""

    decode_query_len = 1

    def __init__(self, model, skip=()):
        super().__init__()
        self.model, self.skip, self.calls, self.steps = model, set(skip), [], 0

    def call(self, layer, x):
        y = layer.quant_method.apply(layer, x)
        self.calls.append((x, y))
        return y

    def execute_model(self, sched):
        ids = torch.tensor(sched.ids)
        x = (((ids[:, None] * 31 + torch.arange(HIDDEN) * 7) % 97 - 48) / 64).to(torch.bfloat16)
        qkv, o, gate_up, down = self.model.layers
        o_out = self.call(o, self.call(qkv, x)[:, :HIDDEN])
        g = self.call(gate_up, x + o_out)
        if self.steps not in self.skip:
            self.call(down, g[:, :INTER] * g[:, INTER:])
        self.steps += 1


def _rows(x, y):
    """A call's A rows and y as Pearl+ commits them: x widened to FP32 in u16 words, zero rows up to m₁₆, and y's BF16 words."""
    m, k = x.shape
    a = np.zeros((-(-m // PPD.ROW_ALIGN) * PPD.ROW_ALIGN, 2 * k), np.uint16)
    a[:m] = x.float().numpy().view("<u2")
    return a, np.ascontiguousarray(y.float().numpy().view("<u2")[:, 1::2])


class Run(PP.EpochCalls, PassEnds):
    """Pearl-C's device run with no device: the real pass numbering (`passes.call_index`, `passes.pass_dir`) and ends
    (`PassEnds.end_pass`, `finish`); a call's y the BF16 product and its A rows and y committed as test_pearltie's toy
    commits them, in a retained pass's manifest beside each weight's n, k and root (as `host.write_manifest` writes
    them)."""

    lag_ends = False

    def __init__(self, s, entry, kernel, weights, roots, coins, opening, schedule="serial", salt_key=None, pass_index=None,
                 rows=None):
        self.rows = rows
        workload = Workload(tuple(Matmul(w, Shape(16, t.shape[1], t.shape[0]), w) for w, t in weights.items()))
        self.epoch = PPD.epoch_of(workload, roots, coins, opening, pass_index)
        self.scheme, self.entry, self.salt, self.salt_key, self.coins = s, entry, self.epoch.salt, salt_key, coins
        self.weights = {str(w): {"n": t.shape[0], "k": t.shape[1], "root": roots[w].hex()} for w, t in weights.items()}
        self.commits = {"scheme": s.name, "salt": self.salt.hex(), "weights_root": self.epoch.weights_root.hex(),
                        "coins": coins.to_json(), "salt_opening": opening.to_json()}
        self.details, self.retain_dir, self.retained, self.cur = {"passes": [], "seconds": []}, None, 0, None
        self.finished = False
        RUNS.append(self)
        self.begin()

    def begin(self, retain=None, run=None, schedule=None, stepped=False, at=None):
        retain = retain or PP.pass_dir(self)
        self.cur = _Pass(Path(retain) if retain else None, run, stepped=stepped, base=self.pass_base(at))

    def settle(self):
        pass

    def gemm(self, w, x, weight, bias):
        (m, k), n = x.shape, weight.shape[0]
        index = PP.call_index(self)
        y = (x.float() @ weight.float().T).to(torch.bfloat16)
        a, y16 = _rows(x, y)
        root_a = commit_rows(activation_binding(self.salt, index), index, [r.tobytes() for r in a], HM96, self.salt_key,
                             index).commitment.root
        self.cur.matmuls.append(Matmul(index, Shape(len(a), k, n), w))
        self.cur.records.append({"index": index, "m": len(a), "k": k, "n": n, "weight": w, "real_rows": m,
                                 "root_a": root_a.hex(),
                                 "y": {"root": SY.commit_y(y16.tobytes(), n, self.salt, index, self.salt_key)[0].hex()}})
        return y

    def end(self, lag=False, behind=False):
        cur = self.cur
        out = {"units": len(cur.matmuls), "tiles": len(cur.matmuls)} | ({"base": cur.base} if cur.matmuls else {})
        if cur.retain is not None and cur.matmuls:
            cur.retain.mkdir(mode=0o700)
            (path := cur.retain / "manifest.json").write_text(json.dumps(
                {"entry": self.entry, "scheme": self.scheme.name, "salt": self.salt.hex(),
                 "hm96_salt_key": self.salt_key.hex(), "weights": self.weights, "matmuls": cur.records}))
            out["manifest"] = str(path)
        cur.parts.switch(None)
        if cur.matmuls:
            self.details["seconds"].append(cur.parts.seconds)
        self.details["passes"].append(out)
        return out

    def finish(self):
        PassEnds.finish(self)
        self.finished = True


@pytest.fixture(autouse=True)
def _device_run_is_the_stand_in(monkeypatch):
    monkeypatch.setattr(pouw, "linear_types", lambda: (Linear,))
    monkeypatch.setattr(PP, "model_runners", lambda: (Forward,))
    monkeypatch.setattr(PPD.Kernel, "load", staticmethod(lambda *a: None))
    monkeypatch.setattr(PPD, "weight_roots", lambda kernel, weights, scheme, salt_key=None: {
        w: bytes([w + 1]) * 32 + salt_key for w in weights})
    monkeypatch.setattr(PPD, "keep_rows", lambda kernel, weights, scheme, directory: {
        f"{w}.rows": (directory / f"{w}.rows", {"path": f"{w}.rows", "bytes": 2 * t.numel()}) for w, t in weights.items()})
    monkeypatch.setattr(PPD, "PearlCDeviceRun", Run)
    monkeypatch.setattr(pouw, "_SALT_KEYS", {})
    monkeypatch.setattr(verifier, "coins", random.Random(0).randbytes)
    monkeypatch.setattr(verifier, "executable", lambda: Path("lean"))
    ticks = itertools.count()
    monkeypatch.setattr(Parts, "clock", staticmethod(lambda: float(next(ticks))))
    RUNS.clear()


def _serving(tmp_path, runner, coins=None, ahead=None):
    """The server over ``runner``, its arm a `PearlWindows` over the committer `Serving` makes, ready."""
    s = Serving(tmp_path, window_steps=CLASS.outputs, runner=runner,
                arm=lambda r: SPL.PearlWindows(runner.model, r, None, out=tmp_path, config=CONFIG, run_id="t", coins=coins,
                                               ahead=ahead, clock=lambda: 0.0))
    s.windows.arm.com = s.windows.com
    s.windows.arm.ready()
    return s


def _serve(s, prompts):
    for prompt in prompts:
        status, _ = s.post("/v1/completions", dict(CERTIFIER_BODY, prompt=prompt, max_tokens=CLASS.outputs,
                                                   min_tokens=CLASS.outputs))
        assert status == 200


def test_each_window_is_one_epoch_a_pass_a_step_numbered_as_the_toys_and_its_w_registers_what_it_served(tmp_path):
    """Two requests of the toy's class, one a window: each window's epoch has its own salt and hm96 key, its passes are
    its steps, each with every linear's call at the toy's index, m₁₆ and real rows, and its W's entries are the roots of the
    x each linear was handed and the y it served, then each linear's weight's B root, once; its line's ``pouw_pearl`` is
    its outcome, its seconds with where its passes' went (the first pass's parts, the later ones' summed), and every hook
    is off after.  The weights' rows are kept once, when the arm is made ready, in
    ``pearl/rows/``, and each window's run is given them.  The coins are in-process, so each window's epoch is made
    ahead: window 0's when the arm is made ready, with no hook on the model, and the next window's at each close, the
    last of which no window takes and the drain closes."""
    original = vars(Forward)["execute_model"]
    runner = Forward(Model())
    with _serving(tmp_path, runner) as srv:
        arm = srv.windows.arm
        assert sorted(arm.rows) == [f"{c}.rows" for c in range(CALLS)] and len(RUNS) == 1 and arm.ahead
        assert hooks.live(SPL.ENTRY) == () and all("apply" not in vars(m.quant_method) for m in runner.model.layers)
        assert {p for p, _ in arm.rows.values()} == {tmp_path / SPL.DIR / SPL.ROWS / f"{c}.rows" for c in range(CALLS)}
        assert (tmp_path / SPL.DIR / SPL.ROWS).stat().st_mode & 0o777 == 0o700
        assert arm.describe()["ready"] == {"seconds": {"kernel": 0.0, "lean": 0.0, "rows": 0.0, "epoch": 0.0}, "rows": {
            "files": CALLS, "bytes": sum(2 * n * k for n, k in SHAPES)}}
        _serve(srv, PROMPTS)
    assert [r.rows for r in RUNS] == [arm.rows] * 3 and [r.finished for r in RUNS] == [True, True, False]
    arm.drain()
    assert RUNS[2].finished and arm.made is None
    wins, salts = records(tmp_path, "windows.jsonl"), []
    assert [w["window"] for w in wins] == [0, 1] and all("pouw_h1" not in w for w in wins)
    for k, line in enumerate(wins):
        d = tmp_path / SPL.DIR / f"w{k}"
        served = json.loads((d / "served.json").read_text())
        assert served["schema"] == PWR.SERVED_SCHEMA and served["ok"] and served["error"] is None
        assert served["commit_steps"] == served["passes_with_calls"] == CLASS.outputs
        assert [(p["pass"], p["units"], p["timed"], p["base"]) for p in served["passes"]] == [
            (s, CALLS, True, s * CALLS) for s in range(CLASS.outputs)]
        assert [Path(p["manifest"]).parent.name for p in served["passes"]] == ["pass-00000", "pass-00001"]
        assert json.loads((d / "coins.json").read_text()) == RUNS[k].coins.to_json()
        assert served["coins"]["source"] == "in-process" and served["coins"]["salt"] == RUNS[k].salt.hex()
        W = PWR.write(tmp_path / f"W{k}.json", served["passes"], range(CLASS.outputs))
        assert W["window"]["salt"] == RUNS[k].salt.hex() and W["window"]["entry"] == SPL.ENTRY
        manifests = [json.loads(Path(p["manifest"]).read_text()) for p in served["passes"]]
        key, entries = bytes.fromhex(manifests[0]["hm96_salt_key"]), iter(W["registered"])
        for s in range(CLASS.outputs):
            for c in range(CALLS):
                u, i = manifests[s]["matmuls"][c], s * CALLS + c
                assert (u["index"], u["weight"], u["m"], u["real_rows"], (u["n"], u["k"])) == (
                    i, c, PEARL.a_rows(CLASS, s), PEARL.rows(CLASS, s), SHAPES[c])
                a, y16 = _rows(*runner.calls[(k * CLASS.outputs + s) * CALLS + c])
                ea, ey = next(entries), next(entries)
                assert (ea["value"], ea["leaves"], ey["value"], ey["leaves"]) == (
                    PEARL.a(s, c), PEARL.a_rows(CLASS, s), PEARL.y(s, c), PEARL.rows(CLASS, s))
                assert ea["root"] == commit_rows(activation_binding(RUNS[k].salt, i), i, [r.tobytes() for r in a], HM96,
                                                 key, i).commitment.root.hex()
                assert ey["root"] == SY.commit_y(y16.tobytes(), u["n"], RUNS[k].salt, i, key)[0].hex()
        assert [(e["value"], e["root"], (e["leaves"], e["words"])) for e in entries] == [
            (PEARL.b(c), (bytes([c + 1]) * 32 + key).hex(), SHAPES[c]) for c in range(CALLS)]
        salts.append((RUNS[k].salt, key))
        roots = {w: bytes([w + 1]) * 32 + key for w in range(CALLS)}
        assert RUNS[k].epoch.weights_root == weights_root(roots)
        assert {**line["pouw_pearl"], "seconds": None} == {
            "entry": SPL.ENTRY, "window": k, "steps": CLASS.outputs, "passes": CLASS.outputs, "calls": CLASS.outputs * CALLS,
            "salt": RUNS[k].salt.hex(), "weights_root": weights_root(roots).hex(), "coin_source": "in-process",
            "served": f"{SPL.DIR}/w{k}/served.json", "seconds": None}
        seconds = line["pouw_pearl"]["seconds"]
        assert sorted(seconds) == ["ahead", "epoch", "hooks", "install", "next", "passes", "record", "release", "unhook"]
        assert sorted(seconds["epoch"]) == ["coins", "run", "salt", "weights"]
        assert seconds["passes"] == {"count": CLASS.outputs, "first": STEP_PARTS, "later": STEP_PARTS}
        assert d.stat().st_mode & 0o777 == 0o700 and RUNS[k].finished
    assert salts[0][0] != salts[1][0] and salts[0][1] != salts[1][1]
    assert srv.windows.summary()["pouw_pearl"] == {"windows": 2, "served": 2, "passes": 4, "calls": 16, "failed": [],
                                                   "closing": [], "ahead": {"made": 3, "taken": 2, "unused": 1,
                                                                            "failed": []}}
    assert hooks.live(SPL.ENTRY) == () and "execute_model" not in vars(runner) and vars(Forward)["execute_model"] is original
    assert all("apply" not in vars(m.quant_method) for m in runner.model.layers)


def test_a_window_short_of_a_linears_call_is_its_visible_failure(tmp_path):
    """The forward skips the down projection at step 1 (window 0's second step): window 0's served.json names its pass
    short of a call and isn't ok, and window 1, a fresh epoch, is served."""
    runner = Forward(Model(), skip={1})
    coins = lambda k: {"coins": Challenger(k, random.Random(k).randbytes)}  # noqa: E731
    with _serving(tmp_path, runner, coins) as s:
        _serve(s, PROMPTS)
    bad = json.loads((tmp_path / SPL.DIR / "w0" / "served.json").read_text())
    assert not bad["ok"] and bad["error"] == f"pass 1 has {CALLS - 1} calls, not one a linear ({CALLS})"
    lines = records(tmp_path, "windows.jsonl")
    assert lines[0]["pouw_pearl"]["error"] == bad["error"] and "error" not in lines[1]["pouw_pearl"]
    assert json.loads((tmp_path / SPL.DIR / "w1" / "served.json").read_text())["ok"]
    assert s.windows.summary()["pouw_pearl"] == {"windows": 2, "served": 1, "passes": 2, "calls": 8, "failed": [0],
                                                 "closing": []}
    assert s.windows.arm.describe()["coins"] == "given" and not s.windows.arm.ahead and len(RUNS) == 2
    assert all(sorted(w["pouw_pearl"]["seconds"]) == ["epoch", "hooks", "install", "passes", "record", "release", "unhook"]
               for w in lines)


def _registered(tmp_path, ahead, monkeypatch):
    """Two windows served with seeded coins and each epoch's hm96 key its number's, their epochs made ahead or not: each
    window's line less its seconds, its coins.json and its W."""
    made = iter(range(8))
    monkeypatch.setattr(PPD, "new_salt_key", lambda scheme: bytes([next(made) + 1]) * 32)
    coins = lambda k: {"coins": Challenger(k, random.Random(k).randbytes)}  # noqa: E731
    RUNS.clear()
    with _serving(tmp_path, Forward(Model()), coins, ahead) as s:
        made_at_ready = len(RUNS)
        _serve(s, PROMPTS)
    s.windows.arm.drain()
    out = []
    for k, line in enumerate(records(tmp_path, "windows.jsonl")):
        d = tmp_path / SPL.DIR / f"w{k}"
        served = json.loads((d / "served.json").read_text())
        W = PWR.write(tmp_path / f"W{k}.json", served["passes"], range(CLASS.outputs))
        out.append(({**line["pouw_pearl"], "seconds": None}, (d / "coins.json").read_text(), W["window"], W["registered"],
                    served["coins"], served["ok"]))
    return out, made_at_ready, len(RUNS), s.windows.summary()["pouw_pearl"].get("ahead")


def test_a_window_on_an_epoch_made_ahead_registers_what_it_registers_when_its_open_makes_it(tmp_path, monkeypatch):
    """The same two windows, coins and keys, served with each epoch made at its window's open and with each made ahead:
    every window's outcome, coins.json, coins of record and W (its salt, and each root it registers) are the same.  Ahead,
    window 0's epoch exists before the server listens and a third is made at the last close, which no window takes."""
    (tmp_path / "at-open").mkdir()
    (tmp_path / "ahead").mkdir()
    at_open = _registered(tmp_path / "at-open", False, monkeypatch)
    ahead = _registered(tmp_path / "ahead", True, monkeypatch)
    assert at_open[0] == ahead[0] and all(ok for *_, ok in ahead[0]) and len(ahead[0]) == 2
    assert at_open[1:] == (0, 2, None) and ahead[1:] == (1, 3, {"made": 3, "taken": 2, "unused": 1, "failed": []})


def test_an_epoch_that_cant_be_made_ahead_is_its_windows_failure_at_its_open_and_is_not_made_again(tmp_path):
    """Window 1's coins refuse.  Made ahead at window 0's close, that refusal is kept and is window 1's failure when it
    opens, as when its open asked: its coins are asked for once, never again.  Window 1 holds no epoch, so none is made
    ahead at its close, and window 2 makes its own at its open and is served."""
    asked: list = []

    def coins(k):
        asked.append(k)
        if k == 1:
            raise ValueError("no coins for window 1")
        return {"coins": Challenger(k, random.Random(k).randbytes)}

    with _serving(tmp_path, Forward(Model()), coins, True) as s:
        _serve(s, [*PROMPTS, PROMPTS[0]])
    lines = records(tmp_path, "windows.jsonl")
    assert [w["pouw_pearl"].get("error") for w in lines] == [None, "open: ValueError: no coins for window 1", None]
    assert asked == [0, 1, 2, 3] and "ahead" in lines[0]["pouw_pearl"]["seconds"]
    assert "ahead" not in lines[2]["pouw_pearl"]["seconds"] and not (tmp_path / SPL.DIR / "w1").exists()
    assert s.windows.summary()["pouw_pearl"] == {"windows": 2, "served": 2, "passes": 4, "calls": 16, "failed": [1],
                                                 "closing": [], "ahead": {"made": 2, "taken": 1, "unused": 0,
                                                                          "failed": [1]}}
    s.windows.arm.drain()
    assert s.windows.arm.aheads["unused"] == 1 and [r.finished for r in RUNS] == [True, True, True]


def test_serve_openai_refuses_what_the_pearl_arm_cant_serve_before_the_model_loads(tmp_path):
    man = predicted.row(tmp_path)
    base = ["--model", "m", "--revision", "r", "--required-manifest", str(man)]
    arm = ("--protocols", SPL.ENTRY, "--precision", "bfloat16", "--protocols-config", json.dumps(CONFIG))

    def refusal(*more: str) -> str | None:
        try:
            so.preflight(cli.parse("serve-openai", [*base, *more], environ={}), which=lambda _: "/v/bin/ninja")
        except cli.UsageError as e:
            return str(e)
        return None

    assert refusal(*arm) is None
    opts = cli.parse("serve-openai", [*base, *arm], environ={})
    assert so.engine_options(opts)["dtype"] == "bfloat16" and "quantization" not in so.engine_options(opts)
    assert "it needs --precision bfloat16, not verity_int8" in refusal(*arm[:2], *arm[4:])
    assert "--commit, --scheme rows-sha512 and the row's --required-manifest" in refusal(*arm, "--no-commit")
    live = ("--challenger-gate", "g", "--challenger-peer", "p", "--challenger-lane", "l")
    assert refusal(*arm, *live) is None and refusal(*arm, *live, "--challenger-ahead") is None
    assert "--challenger-ahead makes each window's live session ahead of the window" in refusal(*arm, "--challenger-ahead")
    assert "--challenger-ahead is Pearl+'s" in refusal("--challenger-ahead")
    for short in (live[:4], live[2:], (*live[:2], *live[4:])):
        assert "--challenger-lane go together under Pearl+" in refusal(*arm, *short)
    assert "--h1-model: PoUW H1's, not Pearl+'s" in refusal(*arm, *live, "--h1-model", "m")
    assert "--challenger-lane is Pearl+'s" in refusal("--challenger-lane", "l")
    assert "not Pearl+'s to give" in refusal(*arm[:4], "--protocols-config", json.dumps({"pouw": {"transcript_dir": "/t"}}))
    assert "are not Pearl+'s (pouw)" in refusal(*arm[:4], "--protocols-config", json.dumps({"pous": {}}))
    after = cli.parse("serve-openai", [*base, *arm, "--release-after", SPL.DRAWER], environ={})
    assert SPL.refusal(after).startswith(f"--release-after {SPL.DRAWER} waits for PoUW H1's draw")
    assert SPL.refusal(cli.parse("serve-openai", base, environ={})) is None



GATE, PEER, LANE = "ssh -i KEY -o BatchMode=yes research@gate", "vy-dc-h200", "compute-accounting"


def _gate(ledger, asked):
    """The gate's subprocess: its ledger ``ledger``, and each reserve answered with a session named for its epoch, joining
    the lane's open period there."""
    def run(argv):
        assert argv[:-1] == shlex.split(GATE)
        asked.append(shlex.split(argv[-1]))
        if asked[-1][0] == "ledger":
            return 0, "".join(json.dumps(e) + "\n" for e in ledger)
        return 0, json.dumps(_reserved(asked[-1][-1], ledger[-1]["body"]["period"] if ledger else None))
    return run


def _reserved(epoch, period):
    return {"session": f"s-{epoch}", "seq": 7, "peer": PEER, "by": LANE, "period": period,
            "expires_utc": "2026-11-04T16:00:00Z"}


def test_live_coins_reserve_each_window_a_salt_only_session_under_the_lane_which_the_install_takes(tmp_path, monkeypatch):
    """`serve_pearl.live`: window k's coins are a salt-only session the gate reserves when the window opens, under the
    lane's open period (``ledger``, then ``reserve --peer P --scheme S --families [["pouw/salt",1]] --epoch k``), as the
    pouw settings ``challenger``, ``session``, ``gate`` and ``period``, which the window's composition takes; `arm_of`
    gives them under ``--challenger-gate``. A reserve refused (the lane's open period H1's) is that window's visible
    failure, before any epoch is opened."""
    asked: list = []
    period = {"period": "p7", "lane": LANE}
    coins = SPL.live(GATE, PEER, LANE, _gate([{"kind": "period-open", "session": None, "body": period}], asked))
    families = json.dumps([list(f) for f in Wd.SALT_FAMILIES], separators=(",", ":"))
    for k in (0, 1):
        got = coins(k)
        assert {x: got[x] for x in ("challenger", "session", "period")} == {
            "challenger": SPL.AT, "session": _reserved(str(k), "p7"), "period": period}
        assert asked[-2:] == [["ledger"], ["reserve", "--peer", PEER, "--scheme", Wd.LIVE_SCHEME, "--families", families,
                                           "--epoch", str(k)]]
        settings = pouw.Settings.of(SPL.context(CONFIG, {"transcript_dir": str(tmp_path), **got}).settings["pouw"])
        assert (settings.coins, settings.challenger, settings.session["session"], settings.period) == (
            None, SPL.AT, f"s-{k}", period)

    made: dict = {}

    def windows(*a, **kw):
        made["coins"], made["ahead"] = kw["coins"], kw["ahead"]
        return SimpleNamespace(ready=lambda: None)

    served = ["--model", "m", "--revision", "r", "--protocols", SPL.ENTRY, "--precision", "bfloat16",
              "--protocols-config", json.dumps(CONFIG)]
    gate = ["--challenger-gate", GATE, "--challenger-peer", PEER, "--challenger-lane", LANE]
    with monkeypatch.context() as m:
        m.setattr(SPL, "live", lambda *a: ("live", *a))
        m.setattr(SPL, "PearlWindows", windows)
        m.setattr("verity_vllm.engine.handle.model_runner", lambda v: None)
        for more, want in ((gate, (("live", GATE, PEER, LANE), False)),
                           ([*gate, "--challenger-ahead"], (("live", GATE, PEER, LANE), True)), ([], (None, None))):
            opts = cli.parse("serve-openai", [*served, *more], environ={})
            SPL.arm_of(opts, SimpleNamespace(model=None, vllm=None), None, tmp_path)
            assert (made["coins"], made["ahead"]) == want

    h1 = {"kind": "period-open", "session": None, "body": {"period": "p8", "lane": LANE, **dict.fromkeys(PHW.H1_PERIOD, 1)}}
    with _serving(tmp_path, Forward(Model()), SPL.live(GATE, PEER, LANE, _gate([h1], []))) as s:
        _serve(s, PROMPTS[:1])
    assert records(tmp_path, "windows.jsonl")[0]["pouw_pearl"]["error"].startswith(
        "open: ValueError: refused before reserving a salt-only session: the lane's open period p8 is H1's")
    assert not (tmp_path / SPL.DIR / "w0").exists() and RUNS == []


class _Session:
    """A live session's client, as the install closes it: its close is the gate's last word on it."""

    def __init__(self, sid, asked):
        self.sid, self.asked = sid, asked

    def close(self):
        self.asked.append(["close", self.sid])


def _salt_session(asked):
    """`pouw_h1_window.salt_session` over the stand-in gate: the reserved session opened on the weights' root handed it
    (in ``asked``, after the reserve), its coins an in-process challenger's, which the stand-in run reads."""
    def salt_session(entry, settings, root, opens):
        sid = settings.session["session"]
        asked.append(["open", sid, root.hex()])
        coins = Challenger(0, random.Random(len(asked)).randbytes)
        carried = {**dict.fromkeys(SESSION_KEYS), "session": sid}
        return _Session(sid, asked), coins.commitment, coins.open("salt", root), b"", carried, None
    return salt_session


def _said(caplog):
    """The arm's log lines about a window's epoch, in order."""
    return [m for m in (r.getMessage() for r in caplog.records if r.name == SPL.logger.name) if "its epoch" in m]


def _calls(asked):
    return [a[:2] if a[0] in ("open", "close") else [a[0], *a[-2:]] if a[0] == "reserve" else a for a in asked]


def test_a_live_session_is_made_ahead_only_when_asked_and_the_window_then_opens_without_the_gate(tmp_path, monkeypatch,
                                                                                                caplog):
    """Live coins.  Without ``ahead`` nothing reaches the gate until a window opens: the ledger, its reserve, its open on
    the weights' root, and its close at the window's close.  With it, window 0's session is reserved and opened on its
    weights' root when the arm is made ready, each reserve before its open; the window's open asks the gate nothing (the
    gate's next word after the session's open is its close, at the window's close), and the next window's session
    follows at that close, the old epoch freed.  The one made ahead that no window takes is closed at the drain, unused.
    The arm logs one line when it makes a session ahead and one when it closes one unused.  While the lane has no
    open period, which a session reserved then would join none of, none is made ahead: the gate is asked its ledger
    alone, and the window's open reserves its session as without ``ahead``, joining the period opened meanwhile."""
    caplog.set_level(logging.INFO, logger="verity_vllm")
    period = [{"kind": "period-open", "session": None, "body": {"period": "p7", "lane": LANE}}]
    reserve = lambda k: ["reserve", "--epoch", str(k)]  # noqa: E731
    for name, ahead in (("at-open", None), ("ahead", True), ("no-period-yet", True)):
        asked: list = []
        ledger = [] if name == "no-period-yet" else period
        RUNS.clear()
        caplog.clear()
        (tmp_path / name).mkdir()
        monkeypatch.setattr(PHW, "salt_session", _salt_session(asked))
        with _serving(tmp_path / name, Forward(Model()), SPL.live(GATE, PEER, LANE, _gate(ledger, asked)), ahead) as s:
            ready = _calls(asked)
            if ledger == []:
                ledger += period
            _serve(s, PROMPTS[:1])
        arm = s.windows.arm
        served = _calls(asked)
        arm.drain()
        line = records(tmp_path / name, "windows.jsonl")[0]["pouw_pearl"]
        coins = json.loads((tmp_path / name / SPL.DIR / "w0" / "served.json").read_text())["coins"]
        assert "error" not in line and line["coin_source"] == pouw.PH.CHALLENGER
        assert coins["reserved"] == _reserved("0", "p7") and coins["session"]["session"] == "s-0"
        assert next(a for a in asked if a[:2] == ["open", "s-0"])[2] == RUNS[0].epoch.weights_root.hex()
        if name == "no-period-yet":
            assert ready == [["ledger"]] and "ahead" not in line["seconds"]
            assert served[:5] == [*ready, ["ledger"], reserve(0), ["open", "s-0"], ["close", "s-0"]]
            assert arm.summary()["ahead"] == {"made": 1, "taken": 0, "unused": 1, "failed": []}
            said = _said(caplog)
            assert said[0] == ("window 0: its epoch is left to its open: its lane has no open period for a session "
                               "reserved now to join")
        elif not ahead:
            assert ready == [] and served == [["ledger"], reserve(0), ["open", "s-0"], ["close", "s-0"]]
            assert _calls(asked) == served and len(RUNS) == 1 and not {"ahead", "next"} & set(line["seconds"])
            assert "made ahead" not in caplog.text
        else:
            assert ready == [["ledger"], reserve(0), ["open", "s-0"]]
            assert served == [*ready, ["close", "s-0"], ["ledger"], reserve(1), ["open", "s-1"]]
            assert asked[6][2] == RUNS[1].epoch.weights_root.hex() != asked[2][2]
            assert _calls(asked) == [*served, ["close", "s-1"]] and [r.finished for r in RUNS] == [True, True]
            assert arm.summary()["ahead"] == {"made": 2, "taken": 1, "unused": 1, "failed": []}
            assert {"ahead", "next"} <= set(line["seconds"])
            said = _said(caplog)
            assert [x.split(": its epoch made ahead ")[0] for x in said] == [f"window {k}" for k in (0, 1, 1)]
            assert "live session s-1" in said[2] and "closed unused" in said[2] and "refuses its lane's open period" in said[2]
            assert "live session s-0" in said[0] and "closed unused" not in said[0] + said[1]
