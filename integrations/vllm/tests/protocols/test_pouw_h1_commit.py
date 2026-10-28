"""`protocol_options.pouw_h1_commit` (ROOTS1): with the Commit in H1's serving engine (`acquire.h1_commit.attach`:
the int8 source at each linear's Int8RowQuant_v1 site and the step-input port, on a rows-sha512 committer whose acquires a
`CommitTap` sees, refused under any other scheme or run root), its record
registers each step's A table of K ``a/<K>@<s>`` (both linears' rows, call after call) as its AROW of the
step and K and each round's input-ids table ``ids@<s>`` as its step's input ids, the window's own trees, and calls_root
binds the record. A window whose served A is all zero codes, or one A row off the rows the Commit acquired, or whose
step table holds its calls' rows shifted by a call, or whose Commit ids aren't its prompts, each with an honest Commit,
credits nothing (`UNTIED`); core's credit refuses each at ``arow`` (here and `test_credit_gate`). The window here is a
two-linear stand-in under the auditor host's coins (`test_pouw_h1._host`), or a linear and a MoE layer's routed experts
(`MoEModel`), whose calls' rows the Commit acquires under their H1 names, an idle expert's as its zero rows."""
from __future__ import annotations

import hashlib
import itertools
import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
# ruff: noqa: E402  (the imports below need torch, so they follow its importorskip)
from torch import nn

from verity.core.protocols.pouw import credit as CR
from verity.core.protocols.pouw import weights as Wt
from verity.core.protocols.pouw import window as Wd
from verity.ml.protocols.pouw.circuit import h1window as HW
from verity.core.protocols.sampled_proofs import remote as R
from verity.ml.circuits.codec import canonical_json
from verity_vllm import protocols as PO
from verity_vllm.engine import int8_exact as E
from verity_vllm.engine import int8_ops
from verity_vllm.protocols import pouw
from verity_vllm.protocols.pouw import h1 as PH
from verity_vllm.protocols.pouw.h1 import challenger as PHC
from verity_vllm.protocols.pouw.h1 import hooks as PHK
from verity_vllm.acquire import h1_commit as HC
from verity_vllm.commit.served_record import StepInputPort
from verity_vllm.engine import int8_experts as X
from verity_vllm.engine.hooks import Hooks
from verity_vllm.protocols.pouw.h1 import commit as CT
from verity_vllm.protocols.pouw.h1 import served as SV

from tests.protocols import challenger_host as CH
from tests.protocols.test_pouw_h1 import AUDITOR, Experts, Runner, _host, _remote, _so
from tests.protocols.test_pouw_h1_served import _unit

K, ROWS, STEPS = 64, 32, 2
ENTRY = f"pouw:{PH.SCHEME}"
#: the committer's engine step at the window's step 0 (its steps count from the engine's start, H1's from the window's)
FIRST = 7
#: step 0 admits a and b (prompt tokens, cap); step 1 decodes one row of each
PROMPTS = {"a": [5, 6, 7], "b": [8, 9]}


class Arm:
    """`engine.int8_vllm.Int8PlainLinearMethod`'s apply under H1 (``zy_rows`` set): ``int8_ops.linear_zy``, whose
    ``row_quant`` the Commit acquires, then ``observe``, handed the codes ``edit`` makes of them (call order, codes)."""

    int8_arm = E.ARM
    zy_rows: int | None = None

    def __init__(self, edit=None, calls=None):
        self.edit, self.calls = edit, calls if calls is not None else [0]

    def apply(self, layer, x, bias=None):
        xq, s_a, y, zy = int8_ops.linear_zy(x, layer.weight, layer.weight_scale, self.zy_rows)
        served = xq if self.edit is None else self.edit(self.calls[0], xq.clone())
        self.calls[0] += 1
        self.observe(layer, served, s_a, y, zy)
        return y

    def observe(self, layer, xq, s_a, y, z=None):
        pass


class Linear(nn.Module):
    def __init__(self, n, seed, arm, k=K):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.weight = torch.randint(-127, 128, (n, k), generator=g, dtype=torch.int8)
        self.weight_scale = (torch.rand(n, 1, generator=g) * 0.01 + 1e-4).float()
        self.quant_method = arm

    def forward(self, x):
        return self.quant_method.apply(self, x)


class Model(nn.Module):
    """Two linears of contractions ``ks``, each on its own arm, ``edit`` theirs (counting their calls together)."""

    def __init__(self, edit=None, ks=(K, K)):
        super().__init__()
        calls = [0]
        self.model = nn.Module()
        self.model.embed_tokens = nn.Module()
        self.model.layers = nn.ModuleList([Linear(64, 1, Arm(edit, calls), ks[0]), Linear(32, 2, Arm(edit, calls), ks[1])])


#: the MoE layer's experts and intermediate width: gate_up [2I, K], down [K, I], so its downs are the calls of K = I
EXPERTS, I = 3, 32
MOE = "model.layers.1.experts"
#: each step's top-1 expert per batch row: expert 2 gets no row at step 0, expert 0 none at step 1
ROUTES = ([0, 1, 0, 1, 1], [2, 1])


class MoE(nn.Module):
    """A MoE layer's routed experts on `int8_experts.forward` (`Int8ExpertsMethod`'s, the `test_pouw_h1.Experts`
    stand-in holding its sites), each step routed top-1 by `ROUTES`."""

    def __init__(self, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        w13, w2 = (torch.randint(-127, 128, s, generator=g, dtype=torch.int8) for s in ((EXPERTS, 2 * I, K), (EXPERTS, K, I)))
        s13, s2 = ((torch.rand(*s, generator=g) * 0.01 + 1e-4).float() for s in ((EXPERTS, 2 * I, 1), (EXPERTS, K, 1)))
        self.quant_method, self.step = Experts(X.sites(MOE, w13, s13, w2, s2)), 0

    def forward(self, x):
        ids = torch.tensor(ROUTES[self.step]).reshape(-1, 1)
        self.step += 1
        return X.forward(self.quant_method.sites, x, torch.ones(len(x), 1), ids)


class MoEModel(nn.Module):
    """A dense linear, then a MoE layer of `EXPERTS` experts: H1's linears are the linear then each expert's gate_up and
    down, 7 calls a step at K = 64 (the linear and the gate_ups) and K = I (the downs)."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Module()
        self.model.layers = nn.ModuleList([Linear(64, 1, Arm()), MoE(3)])


class Committer:
    """The rows-sha512 committer's face (`h1_commit.committer`) that `h1_commit.attach`, the int8 source, the step-input
    port and the tap use: the step being committed (``_cur``) and its scheduled tokens (``_plan_shape``), its acquire, its
    sources, its staging (``sized``) and window ring (``ring``), and its finalized 64-byte run root."""

    name, commit_scheme, chunk = "stand-in", CT.SCHEME, 4096

    def __init__(self, engine=None):
        self.engine, self.model, self.run_id = engine, getattr(engine, "model", None), None
        self._cur, self._plan_shape, self.errors, self.got, self.sources = None, None, [], [], []
        self.plan_inputs = self.sized = self.ring = None
        self.installed = self.closed = False

    def _acquire(self, name, t, job):
        self.got.append((int(job.step), name))

    def attach_source(self, src):
        self.sources.append(src)

    def size_arena(self, envelope):
        self.sized = dict(envelope)
        return {"envelope": self.sized}

    def enable_windows(self, rec):
        self.ring = rec

    def install(self):
        self.installed = True
        for s in self.sources:
            s.install()

    def finalize(self):
        return SimpleNamespace(run_root=bytes(CT.ROOT_BYTES), extra={"expected_tensors": 0, "committed_tensors": 0, "errors_n": 0})

    def close(self):
        for s in self.sources:
            s.uninstall()
        self.closed = True


def vllm_config(tokens=ROWS, seqs=8, model_len=64):
    """A rank's vLLM config as `h1_commit.RankCommit` reads it: the token budget, the batch and the model length."""
    return SimpleNamespace(scheduler_config=SimpleNamespace(max_num_batched_tokens=tokens, max_num_seqs=seqs),
                           model_config=SimpleNamespace(max_model_len=model_len))


def _engine(model):
    """What `h1_commit.attach` reads of an engine (`engine.build.build_engine`'s): its model and what it executes
    (``effective``)."""
    return SimpleNamespace(model=model, effective={"max_num_batched_tokens": ROWS, "max_num_seqs": 8, "max_model_len": 64})


def bound(com):
    """`h1_commit.committer` as the stand-in ``com`` on the engine and run it is given."""
    def committer(engine, run_id):
        com.engine, com.model, com.run_id = engine, engine.model, run_id
        return com
    return committer


@pytest.fixture(autouse=True)
def _linear_is_the_stand_in(monkeypatch):
    monkeypatch.setattr(pouw, "linear_types", lambda: (Linear,))


def _ids(run, s):
    """Step s's honest input ids in its batch order: each prefill row its request's next prompt token, a decode row a
    generated token."""
    seen: dict[int, int] = {}
    for r in range(s):
        for q in run.owners.get(r, []):
            seen[q] = seen.get(q, 0) + 1
    out = []
    for q in run.owners[s]:
        p, prompt = seen.get(q, 0), run.requests[q].prompt
        seen[q] = p + 1
        out.append(prompt[p] if p < len(prompt) else 1000 + q)
    return out


class _Laid(dict):
    """`Served`'s open step tables (``_a``, each call's block by call) laid out by ``lay`` when a step's table of K is
    taken whole: the rows the window registers and hands the tap as its table, the Commit's own rows untouched."""

    def __init__(self, lay):
        super().__init__()
        self.lay = lay

    def pop(self, key, *default):
        return dict(enumerate(self.lay([b for _c, b in sorted(super().pop(key, *default).items())])))


#: the server's step-input port's T (`served_record.StepInputPort`: whole 4096-byte chunks of i64 ids)
PORT_T = 512
#: the positions a CUDA graph pads each step's model input with past its scheduled tokens
GRAPH_PAD = 3


def _manifest_sites(steps=range(FIRST, FIRST + STEPS), modules=("model.layers.0", "model.layers.1")):
    """The bench workload's manifest's Int8RowQuant_v1 sites (`int8_source.quant_sites`): (engine step, linear) -> its op path
    ``<module>/<fx node>``, the fx nodes counted on through the run."""
    return {(e, m): f"{m}/row_quant_{i}" for i, (e, m) in enumerate((e, m) for e in steps for m in modules)}


def _served_commit(com, model, quant=None):
    """serve-openai's committer as H1 sees it (`pipeline.serve_seal.committer`): the step-input port and the int8 source
    of the bench workload's manifest, ``com.sources``."""
    from verity_vllm.acquire.sources.int8_source import Int8Source
    from verity_vllm.commit.served_record import StepInputPort
    port, src = StepInputPort(com, PORT_T), Int8Source(com, model, _manifest_sites() if quant is None else quant, [])
    com.sources = [port, src]
    return port, src


def _window(tmp_path, monkeypatch, edit=None, ids=lambda s, x: x, lay=None, model=None, lie=None, com=None,
            order=None, coins=None, key=hashlib.sha512(b"w").digest(), served_commit=False):
    """A served two-step window with the Commit in its engine: the record (`pouw_h1.CALLS_FILE`), the run and the tap.
    ``edit`` makes the codes H1 is handed of what the arm quantized; ``ids`` the input ids the Commit acquires at step s
    of the honest ones; ``lay`` the blocks of a step's A table of K (one a call, in call order) as the window lays them
    out (`_Laid`). ``model`` is the two-linear `Model` unless given (`MoEModel`), H1's linears `pouw_h1_hooks.layers`
    of it; ``lie`` wraps them once H1 is installed (the GPU harness's `edited`); ``order`` the model's layers in the
    order the forward calls them (module order unless given). ``coins(eng, kw)`` is the window's settings as the GPU
    harness makes them (`pouw_h1_gpu.h1_settings` with `kw`), else one challenge phase of `_host`'s; ``key`` the window's key
    (`Served.window`), None for one keyed by its window-open root, as the harness's under a challenger. The Commit is
    H1's own (`h1_commit.attach` on the stand-in `Committer`), or with ``served_commit`` the server's (`_served_commit`,
    H1's tap on it `h1_commit.tap`); either way the ids are its step-input port's, each step's model input padded by
    `GRAPH_PAD` positions past its scheduled tokens."""
    if coins is None:
        _host(tmp_path, monkeypatch, rounds=STEPS)
    model, com = model or Model(edit), com or Committer()
    layers = PHK.layers(model, (Linear,))
    if served_commit:
        port, src = _served_commit(com, model)
        tap = HC.tap(com, model, layers, range(FIRST, FIRST + STEPS))
        src.install()
    else:
        monkeypatch.setattr(HC, "committer", bound(com))
        _com, src, tap = HC.attach(_engine(model), layers)
        port = next(s for s in com.sources if isinstance(s, StepInputPort))
    served = SV.Served(PH.TILE, _unit(), window=key, tap=tap)
    if lay is not None:
        served._a = _Laid(lay)
    kw = {"transcript_dir": str(tmp_path / "w"), "served": served}
    settings = (_remote(rows=ROWS, steps=STEPS, **kw) if coins is None
                else coins(SimpleNamespace(runner=Runner()), {**kw, "admission": "round-0"}))
    runner = settings["pouw"]["runner"]
    comp = PO.install(model, PO.Context(stage="serve", enabled=(ENTRY,), settings=settings))
    lied = Hooks("test-lie")
    if lie is not None:
        for m in layers:
            lied.wrap(m.quant_method, "observe", lie)
    try:
        g, run = torch.Generator().manual_seed(9), pouw.h1_run(ENTRY)
        for s, so in enumerate((_so(*((q, p, 4) for q, p in PROMPTS.items())), _so(running={"a": 1, "b": 1}))):
            runner.execute_model(so)
            com._cur = SimpleNamespace(step=FIRST + s)
            src.on_step_begin()
            got = ids(s, _ids(run, s))
            com._plan_shape, n = (len(got), len(set(run.owners[s]))), len(got) + GRAPH_PAD
            port._pre(None, (torch.tensor([*got, *[0] * GRAPH_PAD]), torch.arange(n)), {})
            for layer in (model.model.layers if order is None else [model.model.layers[w] for w in order]):
                k = layer.weight.shape[1] if isinstance(layer, Linear) else K
                layer((torch.randn(len(run.owners[s]), k, generator=g) * 3).bfloat16())
            com._cur = None
    finally:
        lied.uninstall()
        comp.close()
        if served_commit:
            src.uninstall()
        else:
            com.close()
        tap.untap()
    return json.loads((tmp_path / "w" / PH.CALLS_FILE).read_text()), run, tap


def _record_of(record):
    return json.loads(bytes.fromhex(record["commitment"]["commit"]))


def _core(record, run):
    """Core credit's two ROOTS1 steps over the window's record: each A table and input-ids table committed whole by its
    round's root at the layout's rows (`credit._a_link`, `credit.table_rows`), then each the Commit's value of its rows
    (`credit._arow`)."""
    c = record["commitment"]
    entries, layout = c["served"]["entries"], CR.layout_of(SV.descriptor(run.served.program))[0]
    return (CR._a_link(c["rounds"], entries, len(run.calls), CR.table_rows(layout, len(c["rounds"]), SV.PREFIX)),
            CR._arow(c["rounds"], entries, CR.commit_of(c["commit"])[0][1], layout, SV.PREFIX))


def test_each_steps_a_table_is_the_commits_arow_and_each_rounds_ids_its_input_ids(tmp_path, monkeypatch):
    """An honest window: the Commit acquired each linear's x_q and s_a at its site at engine step FIRST + s, and its
    record registers each step's A table of K = 64, both linears' rows call after call, as `credit.commit_rows` of its
    round and K and each round's input-ids table ``ids@<s>`` as `credit.commit_ids`, the window's own trees (none of the
    Commit's own); round 0's ids are its prompts' tokens in step 0's batch order, each round's tables are listed whole
    under its round's root, the record is under calls_root, core's ROOTS1 steps pass, and every served call is
    credited."""
    record, run, tap = _window(tmp_path, monkeypatch)
    assert run.details["roots1"] == {"tied": True, "reason": None, "values": 4, "own_trees": 0,
                                     "steps": {"0": FIRST, "1": FIRST + 1}}
    assert run.owners[0] == [1, 1, 0, 0, 0] and tap.round_ids[0] == [8, 9, 5, 6, 7]
    assert np.array_equal(run.served.ids_of(0), SV.ids_words([8, 9, 5, 6, 7], run.served.m))
    rec, entries = _record_of(record), {e["value"]: e for e in record["commitment"]["served"]["entries"]}
    assert rec["format"] == CR.COMMIT_FORMAT and rec["program"] == run.served.program_digest
    theirs = {e["value"]: e for e in rec["registered"]}
    for s in range(STEPS):
        name, a = CR.commit_rows(s, K), HW.table(run.served.program, "a", 2 * s, STEPS)
        assert (name, a) == (f"row-table/{s}/k{K}", f"{SV.PREFIX}a/{K}@{s}")
        assert {**theirs[name], "value": a} == entries[a] and entries[a]["leaves"] == 2 * run.served.m
        assert tap.where[name] == {"round": s, "calls": [2 * s, 2 * s + 1], "steps": [FIRST + s] * 2,
                                   "sites": [CT.site(f"model.layers.{w}") for w in range(2)],
                                   "rows": [c.m for c in run.calls if c.step == s]}
        v = f"{SV.PREFIX}ids@{s}"
        assert {**theirs[CR.commit_ids(s)], "value": v} == entries[v]
        for t in (a, v):
            assert t in run.round_log[s]["a"] and all(t not in r["a"] for r in run.round_log if r["round"] != s)
    assert sorted(theirs) == sorted([CR.commit_rows(s, K) for s in range(STEPS)] + [CR.commit_ids(s) for s in range(STEPS)])
    assert PHC.manifest_reason(record) is None and _core(record, run) == (None, None)
    assert all(v["credit"] for v in run.gated().values())


def test_a_chunked_llama_windows_a_tables_are_the_commits_arow(tmp_path, monkeypatch):
    """SPLITK §3–4: Llama's up_proj at K = 4096 and down_proj at K = 14336 (n = 4 chunk leaves a row). The Commit builds
    its AROW from `Served.a_words` as the window does, so each step's ``row-table/<s>/k<K>`` is the window's own tree of
    ``a/<K>@<s>``, M·|calls|·n leaves of `weights.leaf_words` (K) words (n = 1 at K = 4096, its words K/2 + 2 as before),
    and core's ROOTS1 steps pass: `credit._a_link` at the layout's rows and `credit._arow`."""
    ks = (4096, 14336)
    two = Model
    monkeypatch.setitem(globals(), "Model", lambda edit=None: two(edit, ks))
    record, run, tap = _window(tmp_path, monkeypatch)
    assert run.details["roots1"]["tied"] and run.details["roots1"]["own_trees"] == 0 and tap.own == 0
    theirs = {e["value"]: e for e in _record_of(record)["registered"]}
    entries = {e["value"]: e for e in record["commitment"]["served"]["entries"]}
    for s in range(STEPS):
        for k in ks:
            n, a = Wt.chunks(k), f"{SV.PREFIX}a/{k}@{s}"
            assert {**theirs[CR.commit_rows(s, k)], "value": a} == entries[a]
            assert (entries[a]["leaves"], entries[a]["words"]) == (n * run.served.m, Wt.leaf_words(k))
    assert [Wt.chunks(k) for k in ks] == [1, 4] and Wt.leaf_words(4096) == 4096 // 2 + 2
    assert PHC.manifest_reason(record) is None and _core(record, run) == (None, None)
    assert all(v["credit"] for v in run.gated().values())


def _zero(i, xq):
    return torch.zeros_like(xq)


def _one_row(i, xq):
    if i == 0:
        xq[1, 0] = 1 if int(xq[1, 0]) != 1 else 2
    return xq


UNTIED_A = "roots1: round 0's A table h1/a/64@0 of calls [0, 1] is not the Commit's row-table/0/k64 "


@pytest.mark.parametrize("edit,own", [(_zero, 2), (_one_row, 1)], ids=["a-all-zero", "one-a-row-off"])
def test_a_served_a_off_the_commits_arow_credits_nothing(tmp_path, monkeypatch, edit, own):
    """H1 handed codes other than the ones the Commit acquired (every A all zero, the A = 0 probe; or one row of call
    0), with an honest Commit: the Commit registers those steps' rows as trees of its own at another root, so the
    serving process credits no call (`UNTIED`), and core's credit refuses the window at ``arow`` (also
    `test_credit_gate.test_a_window_whose_tables_are_not_the_commits_is_refused_by_core`)."""
    record, run, tap = _window(tmp_path, monkeypatch, edit)
    assert run.details["roots1"]["reason"].startswith(UNTIED_A)
    assert tap.own == own and run.details["roots1"]["own_trees"] == own
    assert PHC.manifest_reason(record) is None
    assert {v["reason"] for v in run.gated().values()} == {CT.UNTIED}
    a_link, arow = _core(record, run)
    assert a_link is None and arow[0] == "arow" and "h1/a/64@0 is not the Commit's row-table/0/k64" in arow[1], arow


def _rotated(blocks):
    return blocks[1:] + blocks[:1]


def _offset(blocks):
    return [np.zeros_like(blocks[0]), *blocks]


@pytest.mark.parametrize("lay,at,why", [
    (_rotated, "arow", "round 0's A table of K = 64 h1/a/64@0 is not the Commit's row-table/0/k64"),
    (_offset, "a-link", f"registered A value h1/a/64@0 has {3 * ROWS} rows, not the {2 * ROWS} the window's layout gives it"),
], ids=["rotated", "offset"])
def test_a_step_table_with_a_calls_rows_shifted_by_a_call_is_refused_by_core(tmp_path, monkeypatch, lay, at, why):
    """A step's A table of K holds its calls of K's rows call after call, the j-th from row M·j, and a proof block reads
    its call's rows at that offset. The window lays step s's table out shifted by a call, every row the served one: call 1's
    rows at row 0 and call 0's at row M (rotated), or each call's a call late behind an empty block (offset), with an
    honest Commit, which registers its own rows in call order at another root, so the serving process credits nothing.
    Core refuses both: the offset table is not the M rows a call the layout gives it, which `credit._a_link` checks
    before hashing any pad (`credit.table_rows`); the rotated one is committed whole by its round, and `credit._arow`
    refuses it as not the Commit's value."""
    record, run, tap = _window(tmp_path, monkeypatch, lay=lay)
    assert run.details["roots1"]["reason"].startswith(UNTIED_A) and tap.own == STEPS
    assert PHC.manifest_reason(record) is None
    assert {v["reason"] for v in run.gated().values()} == {CT.UNTIED}
    a_link, arow = _core(record, run)
    refused = a_link if at == "a-link" else arow
    assert (at == "a-link" or a_link is None) and refused[0] == at and refused[1].startswith(why), (a_link, arow)


def test_input_ids_off_the_prompts_credit_nothing(tmp_path, monkeypatch):
    """Step 0's Commit ids with one id that isn't its request's prompt token: round 0's input-ids table is the prompts'
    (built before step 0's forward), so the Commit registers its own tree as input-ids/0 and the serving process credits
    nothing (`UNTIED`)."""
    _record, run, tap = _window(tmp_path, monkeypatch, ids=lambda s, x: [*x[:3], 99, *x[4:]] if s == 0 else x)
    assert run.details["roots1"]["reason"].startswith("roots1: round 0's input-ids table is not the Commit's input-ids/0")
    assert tap.own == 1
    assert CT.ids_reason(run, tap) == ("roots1: step 0's slot 3 has input id 99 in the Commit, and request 0's prompt "
                                       "token 1 is 6: layer 0's input is not its round's request table")
    assert {v["reason"] for v in run.gated().values()} == {CT.UNTIED}


def test_the_commits_record_is_under_calls_root(tmp_path, monkeypatch):
    """The Commit's record is bound into calls_root (`pouw_h1.calls_root`, core's `credit.calls_root`): without it, or with
    one value's root edited, the record is refused, and one that isn't a registration v2 record is malformed."""
    record, _run, _tap = _window(tmp_path, monkeypatch)
    c = record["commitment"]

    def with_commit(commit):
        return PHC.manifest_reason({**record, "commitment": {**c, "commit": commit}})
    rec = _record_of(record)
    rec["registered"][0] = {**rec["registered"][0], "root": "00" * 64}
    assert "calls_root is not over" in PHC.manifest_reason({**record, "commitment": {k: v for k, v in c.items()
                                                                                       if k != "commit"}})
    assert "and its Commit record" in with_commit(canonical_json(rec).hex())
    assert with_commit("00").startswith("the window's Commit record is malformed")
    assert CR.calls_root(PHC.program_salt_of(c), record["calls"],
                         SV.digest_of(c["served"]["entries"], c["served"]["refused"]),
                         CT.digest(c["commit"])).hex() == c["calls_root"]


def _experts(name):
    return [f"{MOE}.{e}.{name}" for e in range(EXPERTS)]


def test_each_routed_expert_calls_rows_are_the_commits_and_an_idle_experts_its_zero_rows(tmp_path, monkeypatch):
    """A MoE layer's routed experts are no modules, and the Commit acquires each expert call's Int8RowQuant_v1
    rows under the call's H1 name (`acquire.h1_commit.H1Source`), every expert at every step as `int8_experts.forward`
    calls them, an expert no row routed to as its zero rows. Its record registers each step's A table of K = 64 (the
    linear and the gate_ups) and of K = I (the downs) as their `credit.commit_rows`, the window's own trees; core's
    ROOTS1 steps pass and every served call is credited."""
    com = Committer()
    record, run, tap = _window(tmp_path, monkeypatch, model=MoEModel(), com=com)
    assert run.details["roots1"] == {"tied": True, "reason": None, "values": 6, "own_trees": 0,
                                     "steps": {"0": FIRST, "1": FIRST + 1}}
    assert [c.module for c in run.calls[:7]] == ["model.layers.0", *(f"{MOE}.{e}.{p}" for e in range(EXPERTS)
                                                                      for p in X.PROJECTIONS)]
    sites = ["model.layers.0", *_experts(X.GATE_UP), *_experts(X.DOWN)]
    for s, rows in ((0, [2, 3, 0]), (1, [0, 1, 1])):
        assert tap.where[CR.commit_rows(s, I)] == {"round": s, "calls": [7 * s + 2, 7 * s + 4, 7 * s + 6],
                                                   "steps": [FIRST + s] * 3, "sites": [CT.site(n) for n in _experts(X.DOWN)],
                                                   "rows": rows}
        assert tap.where[CR.commit_rows(s, K)]["calls"] == [7 * s, 7 * s + 1, 7 * s + 3, 7 * s + 5]
        assert tap.where[CR.commit_rows(s, K)]["rows"] == [len(ROUTES[s]), *rows]
        assert sorted(n for j, n in com.got if j == FIRST + s and "/row_quant/" in n) == sorted(
            f"{CT.site(n)}/{p}" for n in sites for p in "01")
    assert PHC.manifest_reason(record) is None and _core(record, run) == (None, None)
    assert all(v["credit"] for v in run.gated().values())


def test_one_expert_calls_rows_off_the_commits_are_refused_by_core(tmp_path, monkeypatch):
    """H1 handed all-zero codes at one routed expert call (`pouw_h1_gpu.editor`'s ``a-zero-expert``, expert 0's down at
    step 0), with an honest Commit: the Commit registers its own tree for that step's A table of K = I, so the serving
    process credits nothing and core's credit refuses the window at ``arow``, as the GPU harness's verdict reads it."""
    from tests.protocols import pouw_h1_gpu as G
    ed = G.editor(G.EXPERT_NEGATIVE)
    record, run, tap = _window(tmp_path, monkeypatch, model=MoEModel(), lie=lambda orig: G.edited(orig, ed))
    assert ed.at == f"{MOE}.0.down" and tap.own == 1
    assert run.details["roots1"]["reason"].startswith(
        f"roots1: round 0's A table h1/a/{I}@0 of calls [2, 4, 6] is not the Commit's row-table/0/k{I} ")
    assert {v["reason"] for v in run.gated().values()} == {CT.UNTIED}
    a_link, arow = _core(record, run)
    assert a_link is None and arow[0] == "arow" and arow[1].startswith(
        f"round 0's A table of K = {I} h1/a/{I}@0 is not the Commit's row-table/0/k{I}"), arow
    v = G.negative_verdict({"core": G.core_tie(record, run.served.program, tmp_path / "ev"),
                            "roots1": run.details["roots1"], "edited": ed.at})
    assert v["refused"] and v["edited"] == f"{MOE}.0.down", v


def test_the_source_takes_every_linears_site_at_every_engine_step():
    """`source` names each of H1's linears by its module's name and taps its Int8RowQuant_v1 site at every engine step, so
    The targets at any step are each site's x_q and s_a; a layer not in the model is refused."""
    model = Model()
    src = HC.source(Committer(), model, list(model.model.layers))
    want = {f"{CT.site(f'model.layers.{w}')}/{p}": 1 for w in range(2) for p in "01"}
    assert src.targets_at(0) == src.targets_at(41) == want
    assert src.quant.get((41, "model.layers.1")) == CT.site("model.layers.1") and src.quant.get((0, "model")) is None
    with pytest.raises(ValueError, match="not named modules"):
        HC.source(Committer(), model, [Linear(32, 3, Arm())])
    assert CT.Sites(["q"]).get((5, "q")) == "q/row_quant"
    moe = MoEModel()
    layers = PHK.layers(moe, (Linear,))
    src = HC.source(Committer(), moe, layers)
    names = ["model.layers.0", *(f"{MOE}.{e}.{p}" for e in range(EXPERTS) for p in X.PROJECTIONS)]
    assert src.modules == ["model.layers.0"] and list(src.experts) == names[1:]
    assert src.targets_at(3) == {f"{CT.site(n)}/{p}": 1 for n in names for p in "01"}
    from tests.protocols import pouw_h1_gpu as G
    assert G.negatives_of(layers) == (*G.NEGATIVES, G.EXPERT_NEGATIVE)
    assert G.negatives_of(list(model.model.layers)) == G.NEGATIVES


def test_on_the_servers_commit_each_a_table_is_its_int8_sources_rows_and_each_rounds_ids_its_ports(tmp_path, monkeypatch):
    """serve-openai's rows-sha512 collector in the engine (`_served_commit`), H1's tap on it (`h1_commit.tap`) and no
    Commit of H1's own: the tap reads each linear's x_q and s_a where the bench workload's int8 source acquires them, under
    the manifest's op paths, as the linear's site, and each step's ids at the port, its scheduled tokens only, so the
    graph's padded positions are neither ids nor cut from them again. The window ties as on H1's own Commit, core's
    ROOTS1 steps pass and every served call is credited; one whose served A is all zero, with the same honest Commit,
    credits nothing."""
    com = Committer()
    record, run, tap = _window(tmp_path, monkeypatch, com=com, served_commit=True)
    assert run.details["roots1"] == {"tied": True, "reason": None, "values": 4, "own_trees": 0,
                                     "steps": {"0": FIRST, "1": FIRST + 1}}
    assert tap.round_ids[0] == [8, 9, 5, 6, 7] and tap.round_ids[1] == _ids(run, 1)
    for s in range(STEPS):
        assert tap.where[CR.commit_rows(s, K)]["sites"] == [CT.site(f"model.layers.{w}") for w in range(2)]
        assert tap.where[CR.commit_rows(s, K)]["steps"] == [FIRST + s] * 2
    assert sorted(n for j, n in com.got if j == FIRST) == sorted(
        ["step.input/ids", "step.input/positions", *(f"model.layers.{w}/row_quant_{w}/{p}" for w in range(2) for p in "01")])
    assert PHC.manifest_reason(record) is None and _core(record, run) == (None, None)
    assert all(v["credit"] for v in run.gated().values())
    (tmp_path / "zero").mkdir()
    _record, run, tap = _window(tmp_path / "zero", monkeypatch, _zero, served_commit=True)
    assert run.details["roots1"]["reason"].startswith(UNTIED_A) and tap.own == STEPS
    assert {v["reason"] for v in run.gated().values()} == {CT.UNTIED}


def test_the_ports_ids_are_the_steps_scheduled_tokens_and_keep_no_graph_pad():
    """The port's T ids, zero past the step's N scheduled tokens: the tap keeps the first N, which hold no CUDA graph's
    padding, so round r's ids are those N; port ids with no count of the step's tokens are refused, and so is a round at
    an engine step whose ids the Commit never acquired."""
    tap = CT.CommitTap(port="step.input/ids")
    tap.take("step.input/ids", np.array([5, 6, 7] + [0] * (PORT_T - 3)), FIRST, 3)
    tap._steps[0] = FIRST
    assert tap.ids_at(0) == [5, 6, 7] and tap.refused == []
    tap.take("step.input/ids", np.zeros(PORT_T, np.int64), FIRST + 1)
    assert tap.refused == [f"engine step {FIRST + 1}: the port's input ids with no count of the step's scheduled tokens"]
    tap._steps[1] = FIRST + 1
    assert tap.ids_at(1) == [] and tap.refused[-1] == f"step 1: the Commit acquired no input ids at engine step {FIRST + 1}"


def test_h1_taps_the_servers_commit_only_where_it_acquires_every_linear_at_every_step():
    """`h1_commit.tap` puts no source on the server's Commit: it needs the Commit's step-input port and int8 source, and
    refuses a window step at which the manifest names no site of one of H1's linears, and H1's routed experts, whose
    sites only H1's own source acquires; a tap it refuses is never put on."""
    model = Model()
    layers = PHK.layers(model, (Linear,))
    steps = range(FIRST, FIRST + STEPS)
    com = Committer()
    with pytest.raises(ValueError, match="no step-input port"):
        HC.tap(com, model, layers, steps)
    _served_commit(com, model)
    com.sources = com.sources[:1]
    with pytest.raises(ValueError, match="no int8 source"):
        HC.tap(com, model, layers, steps)
    _served_commit(com, model, _manifest_sites(steps=[FIRST]))
    with pytest.raises(ValueError, match=rf"acquires 2 of H1's .* sites nowhere, \[\({FIRST + 1}, 'model.layers.0'\)"):
        HC.tap(com, model, layers, steps)
    assert "_acquire" not in vars(com)
    moe = MoEModel()
    _served_commit(com, moe, _manifest_sites(modules=("model.layers.0",)))
    with pytest.raises(ValueError, match="routed experts"):
        HC.tap(com, moe, PHK.layers(moe, (Linear,)), steps)
    _served_commit(com, model)
    tap = HC.tap(com, model, layers, steps)
    assert tap.port == "step.input/ids"
    assert tap.ops == {f"model.layers.{w}/row_quant_{2 * i + w}": CT.site(f"model.layers.{w}")
                       for i in range(STEPS) for w in range(2)}
    tap.untap()
    assert "_acquire" not in vars(com)


class Top(nn.Module):
    """vLLM's top model as the step-input port's pre-hook sees it, called as ``(input_ids, positions)``: two linears and
    an embedding it never calls, as GLM-5.3 at TP1 gathers its table fused."""

    def __init__(self):
        super().__init__()
        self.model = Model().model

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        return positions


def test_attach_builds_a_rows_sha512_commit_whose_port_is_the_steps_one_copy_of_its_ids(monkeypatch):
    """`h1_commit.attach`: H1's committer (`h1_commit.committer`, the stand-in here), the int8 source at each linear, and
    the step-input port over the engine's token budget rounded up to whole chunks of i64 ids (32 tokens, a port of 512),
    with no module's input acquired beside it (``plan_inputs`` empty); the staging sized for the engine's workload and its
    window ring enabled on that sizing; then installed. The port takes each step's ids at the top model's forward, so a
    model that never calls its embedding gives them, and the tap keeps the step's scheduled tokens only."""
    top, com = Top(), Committer()
    monkeypatch.setattr(HC, "committer", bound(com))
    got, src, tap = HC.attach(_engine(top), PHK.layers(top, (Linear,)), run_id="h1/w")
    port = com.sources[-1]
    try:
        assert got is com and com.run_id == "h1/w" and com.sources == [src, port] and isinstance(port, StepInputPort)
        assert port.T == HC.port_tokens(com, ROWS) == 512 and com.plan_inputs == set()
        assert com.sized == {"batch": 8, "max_prompt": ROWS, "max_new": 1, "max_num_batched_tokens": ROWS,
                             "max_model_len": 64}
        assert com.ring == {"envelope": com.sized} and com.installed and tap.port == "step.input/ids"
        com._cur, com._plan_shape = SimpleNamespace(step=FIRST), (3, 2)
        top(torch.tensor([5, 6, 7, 0]), torch.arange(4))
        assert com.got == [(FIRST, "step.input/ids"), (FIRST, "step.input/positions")] and com.errors == []
        assert tap._ids[FIRST].tolist() == [5, 6, 7] and tap.refused == []
    finally:
        com.close()
        tap.untap()
    assert com.closed and port._h is None and "_acquire" not in vars(com)
    assert [HC.port_tokens(com, t) for t in (1, 512, 513, 8192)] == [512, 512, 1024, 8192]
    with pytest.raises(ValueError, match=r"gives no \['max_num_batched_tokens', 'max_num_seqs'\]"):
        HC.envelope(SimpleNamespace(effective={"max_model_len": 64}))


def test_h1s_commit_refuses_a_sha256_run_root(monkeypatch):
    """vllm-v1's committer (SHA-256 rows, 32-byte roots) is no Commit of H1's. `h1_commit.attach` refuses one built under
    it, closed before anything is put on the engine; and `pouw_h1_commit.committed`, what a run records of the Commit
    (the GPU harness's ``committer`` and each rank's release), refuses vllm-v1's real 32-byte run root, and a
    rows-sha512 committer's root of any length but 64. The offline core's rows-sha512 run root passes."""
    from tests.check import opened as O
    steps = {0: [("step.input/ids", bytes(range(64)))]}
    sha512 = O.commit_steps(steps, commit_scheme=CT.SCHEME, run_id="h1-rows")
    sha256 = O.commit_steps(steps, run_id="h1-v1")
    assert (sha512[0].commit_scheme, len(sha512[1].run_root)) == (CT.SCHEME, CT.ROOT_BYTES)
    assert (sha256[0].commit_scheme, len(sha256[1].run_root)) == ("vllm-v1", 32)
    assert CT.committed(*sha512) == {"run_root": sha512[1].run_root.hex(), "commit_scheme": CT.SCHEME,
                                     "name": sha512[0].name}
    with pytest.raises(ValueError, match=f"committed under vllm-v1 to a 32-byte run root, not under {CT.SCHEME} to a 64-byte"):
        CT.committed(*sha256)
    short = SimpleNamespace(run_root=bytes(32), extra={})
    with pytest.raises(ValueError, match=f"committed under {CT.SCHEME} to a 32-byte run root"):
        CT.committed(Committer(), short)
    old, model = sha256[0], Model()
    closed = []
    monkeypatch.setattr(old, "close", lambda: closed.append(True))
    monkeypatch.setattr(HC, "committer", lambda engine, run_id: old)
    with pytest.raises(ValueError, match=f"commits under vllm-v1, not {CT.SCHEME}"):
        HC.attach(_engine(model), PHK.layers(model, (Linear,)))
    assert closed == [True] and old.sources == [] and "_acquire" not in vars(old)


def test_the_gpu_harness_holds_each_window_to_cores_roots1_steps(tmp_path, monkeypatch):
    """`pouw_h1_gpu.core_verdict`, the GPU run's check of each Commit window under in-process coins: the honest window's
    record is one manifest under calls_root, its tables committed by their rounds (`credit._a_link`) and each the
    Commit's value (`credit._arow`); the all-zero and one-row-off windows, each with an honest Commit, pass the first two
    and are refused at ``arow``. ``--core-from`` reaches the same verdicts from the saved evidence."""
    from tests.protocols import pouw_h1_gpu as G
    got = {}
    for name, edit in (("commit", None), ("commit-a-zero", _zero), ("commit-a-row-off", _one_row)):
        (tmp_path / name).mkdir()
        record, run, _tap = _window(tmp_path / name, monkeypatch, edit)
        got[name] = {"core": G.core_tie(record, run.served.program, tmp_path / "ev" / name),
                     "roots1": run.details["roots1"]}
    assert got["commit"]["core"]["tied"] and got["commit"]["core"]["commit_values"] == 4
    for name in ("commit-a-zero", "commit-a-row-off"):
        v = G.negative_verdict(got[name])
        assert v["refused"] and v["arow"][1].startswith("round 0's A table of K = 64 h1/a/64@0 is not the Commit's "), v
    assert G.core_from(str(tmp_path / "ev"), str(tmp_path / "out"))["ok"]


def test_a_forward_out_of_module_order_ties_and_its_a_zero_is_refused_at_arow(tmp_path, monkeypatch):
    """A forward that calls linear 1 before linear 0, as GLM-5.3's DSA attention runs fused_qkv_a_proj and q_b_proj before
    its indexer's wq_b, which module order puts first (refused on layout before): each call sits at its
    (step, linear) index, each round's root is over its step's calls, and step s maps to the Commit's engine step at its
    first call, whichever linear that is. The two share K, so step s's A table of K is taken whole at linear 0's call,
    the last to come, with its rows (and the Commit's) in call order, linear 0's first. Core ties the honest window, and
    refuses its all-zero A at ``arow``."""
    from tests.protocols import pouw_h1_gpu as G
    got = {}
    for name, edit in (("commit", None), ("commit-a-zero", _zero)):
        (tmp_path / name).mkdir()
        record, run, tap = _window(tmp_path / name, monkeypatch, edit, order=(1, 0))
        assert [(c["index"], c["step"], c["layer"]) for c in record["calls"]] == [(0, 0, 0), (1, 0, 1), (2, 1, 0), (3, 1, 1)]
        assert run.refused == [] and run.static() and run.order_ok
        got[name] = {"core": G.core_tie(record, run.served.program, tmp_path / "ev" / name),
                     "roots1": run.details["roots1"]}
    for s in range(STEPS):
        assert tap.where[CR.commit_rows(s, K)]["calls"] == [2 * s, 2 * s + 1]
        assert tap.where[CR.commit_rows(s, K)]["sites"] == [CT.site(f"model.layers.{w}") for w in range(2)]
    assert got["commit"]["roots1"]["tied"] and got["commit"]["roots1"]["steps"] == {"0": FIRST, "1": FIRST + 1}
    assert got["commit"]["core"]["tied"] and got["commit"]["core"]["commit_values"] == 4
    v = G.negative_verdict(got["commit-a-zero"])
    assert v["refused"] and v["arow"][1].startswith("round 0's A table of K = 64 h1/a/64@0 is not the Commit's "), v


class _Gate:
    """The challenger gate for a run of several windows (`pouw_h1_gpu.window_sessions`): each ``reserve`` is a challenge phase of its own
    on a challenger host of its own (`challenger_host.Host`, gated), all under one auditor key and one ledger, which
    reserves, opens and ends them in turn. Calling it is the gate's open (`pouw_h1_window.Gate`), the serving side
    reaches the challenge phase opened last, and ``ended`` is the challenge phase's ``status`` once its end is in the ledger."""

    def __init__(self, tmp_path, monkeypatch):
        self.dir, self.leaves, self.hosts, self.opened = tmp_path / "gate", itertools.count(), {}, None
        self.dir.mkdir()
        self.ledger = CH.Ledger(self.dir / "auditor.jsonl", 1, self.leaves)
        monkeypatch.setattr(R.RemoteChallenger, "connect",
                            classmethod(lambda cls, at, secret=None: self.opened.client if at == AUDITOR else None))

    def reserve(self, peer, steps):
        n = len(self.hosts) + 1
        sid = CH.session_id(n)
        self.hosts[sid] = CH.Host(self.dir / sid, Wd.live_challenger(0, steps, CH.entropy(n)), 1, sid, self.leaves,
                                  self.ledger, gated=True)
        return {"session": sid, "seq": len(self.ledger.lines) - 1, "peer": peer, "by": "vllm",
                "expires_utc": "2026-10-20T11:18:00Z"}

    def __call__(self, session, root):
        self.opened = self.hosts[session]
        return self.opened.open(session, root)

    def ended(self, session):
        self.hosts[session].ended()
        lines = [json.loads(line) for line in self.ledger.lines]
        entry = {e["kind"]: e for e in lines if e["session"] == session}
        return {"session": session, "open": entry.get("open"), "end": entry.get("end")}


def _live(gate, sessions, name):
    """``coins`` for `_window`: window `name`'s settings as the GPU harness makes them with ``--challenger --gate --peer``,
    its challenge phase the one `sessions` (`pouw_h1_gpu.window_sessions`) reserves for it."""
    from tests.protocols import pouw_h1_gpu as G
    return lambda eng, kw: G.h1_settings(eng, ROWS, STEPS, AUDITOR, sessions(name), gate, **kw)


def _in_process(eng, kw):
    """``coins`` for `_window`: the GPU harness's settings without ``--challenger``, the window's coins its own."""
    from tests.protocols import pouw_h1_gpu as G
    return G.h1_settings(eng, ROWS, STEPS, **kw)


def _summary_row(gate, sessions, name, record, run):
    """Window `name` as the GPU harness's record keeps it for `pouw_h1_gpu.coins_summary`, its challenge phase ended."""
    from tests.protocols import pouw_h1_gpu as G
    end = G.session_end(gate, sessions.held[name]) if name in getattr(sessions, "held", {}) else None
    return {"coin_source": record["commitment"]["coin_source"], "gate": run.credit(), **({"session": end} if end else {})}


def test_each_commit_window_draws_every_coin_from_a_session_of_its_own(tmp_path, monkeypatch):
    """`pouw_h1_gpu.py --commit --challenger --gate --peer`: the honest Commit window and a negative each reserve their own
    challenge phase at the gate (`window_sessions`), opened on that window's open root; every coin the credit gate checks, the
    salt, each round's and the draw, is that challenge phase's, which the verifier trusts (`trusted_rounds_reason`) and which
    ends closed and complete before the next is reserved (`session_end`). The honest window's credit gate passes every
    served call, the negative's refuses every one as untied (never as in-process), core still refuses the negative at
    ``arow``, the harness's summary is ok, and a window asks for its challenge phase once."""
    from tests.protocols import pouw_h1_gpu as G
    gate = _Gate(tmp_path, monkeypatch)
    sessions = G.window_sessions(gate, "vy-nebius-1", STEPS)
    doc, runs = {"negatives": {}}, {}
    for name, edit in (("commit", None), ("commit-a-zero", _zero)):
        (tmp_path / name).mkdir()
        record, run, _tap = _window(tmp_path / name, monkeypatch, edit, coins=_live(gate, sessions, name), key=None)
        row = {**_summary_row(gate, sessions, name, record, run),
               "core": G.core_tie(record, run.served.program, tmp_path / "ev" / name), "roots1": run.details["roots1"]}
        runs[name] = (record, run, row)
        if name == "commit":
            doc["h1_commit"] = row
        else:
            doc["negatives"]["a-zero"] = {"window": row}
    assert [s["session"] for s in sessions.held.values()] == [CH.session_id(1), CH.session_id(2)]
    for name, (record, run, row) in runs.items():
        c, sid = record["commitment"], sessions.held[name]["session"]
        host = gate.hosts[sid]
        assert c["coin_source"] == PH.CHALLENGER and run.details["challenger"] == {"closed": True}
        assert c["session"]["session"] == sid and c["coins"] == c["round_coins"] == host.challenger.commitment.to_json()
        assert row["session"]["ok"] and row["session"]["end"]["complete"]
        assert PHC.trusted_rounds_reason(c, host.directory, [gate.ledger.path], host.window, host.key,
                                         pins=[gate.ledger.pins_path], main=CH.MAIN) is None
        received = R.released_roots(host.directory / R.RELEASES, host.challenger.commitment)
        assert received[Wd.SALT] == (bytes.fromhex(c["session"]["root"]),)
        assert received[Wd.ROUND] == tuple(bytes.fromhex(r["root"]) for r in c["rounds"]) and len(c["rounds"]) == STEPS
        assert received[Wd.DRAW] == (bytes.fromhex(c["draw"]["root"]),) and c["draw"]["coin"]
    honest, zero = runs["commit"][2]["gate"], runs["commit-a-zero"][2]["gate"]
    served = sum(len(runs["commit"][1].owners[s]) for s in range(STEPS)) * 2
    assert honest["refused"]["calls"] == 0 and honest["gate_passed"]["rows"] == served and honest["coins"] == PH.CHALLENGER
    assert zero["gate_passed"]["rows"] == 0 and set(zero["refused"]["reasons"]) == {CT.UNTIED}
    assert runs["commit"][2]["core"]["tied"] and G.negative_verdict(runs["commit-a-zero"][2])["refused"]
    summary = G.coins_summary(doc, True)
    assert summary["ok"] and summary["windows"]["commit"]["credited_share"] == 1.0
    assert summary["windows"]["commit-a-zero"]["credited_share"] == 0.0
    with pytest.raises(ValueError, match="one session a window"):
        sessions("commit")


def test_a_commit_window_on_coins_held_in_process_credits_nothing(tmp_path, monkeypatch):
    """`pouw_h1_gpu.py --commit` without ``--challenger``: the Commit ties and core's
    ROOTS1 steps pass, but the window's coins were drawn in the serving process, so its credit gate refuses every served
    call as ``coins-not-the-auditors`` and the harness's summary under a challenger run would not be ok."""
    from tests.protocols import pouw_h1_gpu as G
    record, run, _tap = _window(tmp_path, monkeypatch, coins=_in_process)
    assert record["commitment"]["coin_source"] == "in-process" and run.details["roots1"]["tied"]
    assert G.core_tie(record, run.served.program, tmp_path / "ev")["tied"]
    g = run.credit()
    calls = sum(1 for c in run.calls if c.m)
    assert g["gate_passed"] == {"rows": 0, "macs": 0} and g["refused"]["reasons"] == {PH.IN_PROCESS: calls}
    assert PH.IN_PROCESS.startswith("credit: coins-not-the-auditors: ")
    doc = {"h1_commit": {"coin_source": "in-process", "gate": g}}
    assert G.coins_summary(doc, False)["ok"] and not G.coins_summary(doc, True)["ok"]
    assert G.coins_summary(doc, True)["windows"]["commit"]["credited_share"] == 0.0


def test_a_run_mixing_live_and_in_process_coins_credits_only_the_live_window(tmp_path, monkeypatch):
    """A run whose Commit window drew its coins from its own challenge phase and whose next window drew them in process: the live
    window credits every served call, the in-process one none (``coins-not-the-auditors``), and the summary is not ok,
    since every window of a challenger run must be the challenger's. One window's coins come from one source: settings
    with both the challenger and coins held in process are refused before anything is served."""
    from tests.protocols import pouw_h1_gpu as G
    gate = _Gate(tmp_path, monkeypatch)
    sessions = G.window_sessions(gate, "vy-nebius-1", STEPS)
    (tmp_path / "commit").mkdir()
    record, run, _tap = _window(tmp_path / "commit", monkeypatch, coins=_live(gate, sessions, "commit"), key=None)
    live = _summary_row(gate, sessions, "commit", record, run)
    (tmp_path / "fill").mkdir()
    record2, run2, _tap = _window(tmp_path / "fill", monkeypatch, coins=_in_process)
    local = _summary_row(gate, sessions, "fill", record2, run2)
    assert live["gate"]["refused"]["calls"] == 0 and live["gate"]["gate_passed"]["rows"] > 0
    assert local["gate"]["gate_passed"]["rows"] == 0 and set(local["gate"]["refused"]["reasons"]) == {PH.IN_PROCESS}
    summary = G.coins_summary({"h1_commit": live, "h1": local}, True)
    assert not summary["ok"] and summary["windows"]["fill"]["coin_source"] == "in-process"
    assert summary["windows"]["fill"]["session"] is None and summary["windows"]["commit"]["session_ok"]
    both = G.h1_settings(SimpleNamespace(runner=Runner()), ROWS, STEPS, AUDITOR, {"session": CH.session_id(9)}, gate)
    both["pouw"].update({k: v for k, v in _in_process(SimpleNamespace(runner=Runner()), {})["pouw"].items()
                         if k in ("coins", "rounds")})
    assert any("come from one source" in r for r in PO.refusals(PO.Context(stage="serve", enabled=(ENTRY,), settings=both)))


def test_a_refused_reserve_or_a_reused_session_serves_no_window():
    """`window_sessions` takes each window's reserve from the gate as it is: one the gate refuses, or an answer naming a
    challenge phase another window of the run already holds, raises before the window is installed, so no coin is drawn."""
    from tests.protocols import pouw_h1_gpu as G

    class Refusing:
        def reserve(self, peer, steps):
            raise ValueError("the gate refused the reserve (exit 1): period p holds its cap of 1 windows")

    with pytest.raises(ValueError, match="holds its cap"):
        G.window_sessions(Refusing(), "vy-nebius-1", STEPS)("commit")

    class Same:
        def reserve(self, peer, steps):
            return {"session": CH.session_id(1), "seq": 0, "peer": peer, "by": "vllm"}

    sessions = G.window_sessions(Same(), "vy-nebius-1", STEPS)
    assert sessions("commit")["session"] == CH.session_id(1)
    with pytest.raises(ValueError, match="no fresh session"):
        sessions("fill")


@pytest.mark.parametrize("argv,why", [
    (["--commit", "--challenger", "127.0.0.1:7811"], "needs --peer"),
    (["--commit", "--challenger", "127.0.0.1:7811", "--session", '{"session": "s20261020T000000Z-0001"}'], "needs --peer"),
    (["--peer", "vy-nebius-1"], "--peer is the fill workload's"),
    (["--commit", "--challenger", "127.0.0.1:7811", "--gate", "true", "--peer", "vy-nebius-1", "--session",
      '{"session": "s20261020T000000Z-0001"}'], "--peer is the fill workload's"),
])
def test_the_harness_takes_commit_with_the_challenger_only_with_a_session_a_window(tmp_path, capsys, argv, why):
    """`pouw_h1_gpu.py --commit --challenger` serves several windows, so it needs ``--peer`` (each window reserves its own
    challenge phase) and not one ``--session``; ``--peer`` needs ``--challenger`` and ``--gate``."""
    from tests.protocols import pouw_h1_gpu as G
    with pytest.raises(SystemExit):
        G.main(["--out", str(tmp_path), "--at", "t", "--by", "b", "--workload", "fill", "--served", *argv])
    assert why in capsys.readouterr().err


def test_credit_reads_round_zero_under_the_gate_session_the_window_was_served_in(tmp_path, monkeypatch):
    """`pouw_h1_gpu.py --credit` on a challenger window: round 0's input-ids and request tables bind the gate's reserved
    challenge phase, so the stand-in's registration step (`pouw_h1_credit.check`) reads them under that challenge phase, never under the
    in-process one (`LOCAL_SESSION`), which gives them another domain."""
    from tests.protocols import pouw_h1_credit as PC
    from tests.protocols import pouw_h1_gpu as G
    from verity_vllm.protocols.pouw.h1 import served as SV
    gate = _Gate(tmp_path, monkeypatch)
    sessions = G.window_sessions(gate, "vy-nebius-1", STEPS)
    _record, run, _tap = _window(tmp_path, monkeypatch, None, coins=_live(gate, sessions, "commit"), key=None)
    w = PC.of_run(run, run.served)
    assert w.session == sessions.held["commit"]["session"]
    first = [e for e in w.entries if CR.ids_step(e["value"]) == 0 or CR.is_requests(e["value"])]
    assert first
    for e in first:
        assert e["domain"] == CR.domain_of(w.program, w.window, e, w.session)
        assert e["domain"] != CR.domain_of(w.program, w.window, e, SV.LOCAL_SESSION)
