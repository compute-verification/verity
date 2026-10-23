"""`protocol_options.pouw_h1_commit` (ROOTS1, VLLM-73): with the Commit in H1's serving engine (the int8 source at each
linear's Int8RowQuant_v1 site, `pouw_h1_commit.source`, on a committer whose acquires a `CommitTap` sees), each call's
registered A table ``a/<K>@<c>`` is the Commit's AROW of its (step, linear), one tree, and each round's root is over the
Commit's input ids of its step, which are its requests' prompt tokens. A window whose served A is all zero codes, or one A
row off the rows the Commit acquired, each with an honest Commit, is refused by `manifest_reason` and credits nothing
(`UNTIED`); so is one whose input ids aren't its prompts. Its Commit record is under calls_root, so a record without it or
with it edited is refused too. These checks stand in for core's `credit._arow` and Served.lean's layer-0 tie until
they land; the window here is a two-linear stand-in under the auditor host's coins (`test_pouw_h1._host`)."""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from verity.core.protocols.pouw.circuit import h1window as HW  # noqa: E402
from verity_vllm import protocol_options as PO  # noqa: E402
from verity_vllm.engine import int8_exact as E  # noqa: E402
from verity_vllm.engine import int8_ops  # noqa: E402
from verity_vllm.protocol_options import pouw  # noqa: E402
from verity_vllm.protocol_options import pouw_h1 as PH  # noqa: E402
from verity_vllm.protocol_options import pouw_h1_challenger as PHC  # noqa: E402
from verity_vllm.protocol_options import pouw_h1_commit as CT  # noqa: E402
from verity_vllm.protocol_options import pouw_h1_served as SV  # noqa: E402

from tests.protocol_options.test_pouw_h1 import _host, _remote, _so  # noqa: E402
from tests.protocol_options.test_pouw_h1_served import _unit  # noqa: E402

K, ROWS, STEPS = 64, 32, 2
ENTRY = f"pouw:{PH.SCHEME}"
EMBED = "model.embed_tokens"
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
    def __init__(self, n, seed, arm):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.weight = torch.randint(-127, 128, (n, K), generator=g, dtype=torch.int8)
        self.weight_scale = (torch.rand(n, 1, generator=g) * 0.01 + 1e-4).float()
        self.quant_method = arm

    def forward(self, x):
        return self.quant_method.apply(self, x)


class Model(nn.Module):
    """Two linears, each on its own arm, ``edit`` theirs (counting their calls together)."""

    def __init__(self, edit=None):
        super().__init__()
        calls = [0]
        self.model = nn.Module()
        self.model.embed_tokens = nn.Module()
        self.model.layers = nn.ModuleList([Linear(64, 1, Arm(edit, calls)), Linear(32, 2, Arm(edit, calls))])


class Committer:
    """The committer's face the int8 source and the tap use: the step being committed (``_cur``) and its acquire."""

    def __init__(self):
        self._cur, self.errors, self.got = None, [], []

    def _acquire(self, name, t, job):
        self.got.append((int(job.step), name))


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


def _window(tmp_path, monkeypatch, edit=None, ids=lambda s, x: x):
    """A served two-step window with the Commit in its engine: the record (`pouw_h1.CALLS_FILE`), the run and the tap.
    ``edit`` makes the codes H1 is handed of what the arm quantized; ``ids`` the input ids the Commit acquires at step s
    of the honest ones."""
    _host(tmp_path, monkeypatch, rounds=STEPS)
    model, com, tap = Model(edit), Committer(), CT.CommitTap(EMBED)
    src = CT.source(com, model, list(model.model.layers))
    tap.tap(com)
    src.install()
    settings = _remote(rows=ROWS, steps=STEPS, transcript_dir=str(tmp_path / "w"),
                       served=SV.Served(PH.TILE, _unit(), window=hashlib.sha512(b"w").digest(), tap=tap))
    runner = settings["pouw"]["runner"]
    comp = PO.install(model, PO.Context(stage="serve", enabled=(ENTRY,), settings=settings))
    try:
        g, run = torch.Generator().manual_seed(9), pouw.h1_run(ENTRY)
        for s, so in enumerate((_so(*((q, p, 4) for q, p in PROMPTS.items())), _so(running={"a": 1, "b": 1}))):
            runner.execute_model(so)
            com._cur = SimpleNamespace(step=FIRST + s)
            src.on_step_begin()
            com._acquire(f"{EMBED}/in", torch.tensor(ids(s, _ids(run, s))), com._cur)
            for layer in model.model.layers:
                layer((torch.randn(len(run.owners[s]), K, generator=g) * 3).bfloat16())
            com._cur = None
    finally:
        comp.close()
        src.uninstall()
        tap.untap()
    return json.loads((tmp_path / "w" / PH.CALLS_FILE).read_text()), run, tap


def test_each_calls_a_table_is_the_commits_arow_and_each_round_is_over_its_input_ids(tmp_path, monkeypatch):
    """An honest window: the Commit acquired each linear's x_q and s_a at its site at engine step FIRST + s, and each call's
    registered A table is that AROW, H1's own tree (no tree of the Commit's own); round s's root is over the Commit's
    ids of step s, its prompts' tokens; the record passes `manifest_reason` and every served call is credited."""
    record, run, tap = _window(tmp_path, monkeypatch)
    sites = [CT.site(f"model.layers.{w}") for w in range(2)]
    assert run.details["roots1"] == {"tied": True, "reason": None, "arows": 4, "own_trees": 0, "rounds": 2,
                                     "steps": {"0": FIRST, "1": FIRST + 1}}
    assert sorted(tap.round_ids[0]) == [5, 6, 7, 8, 9] and sorted(tap.round_ids[1]) == [1000, 1001]
    for c in run.calls:
        got, a = tap.arows[c.index], HW.table(run.served.program, "a", c.index)
        assert got["value"] == a and got["root"] == run.tables[c.index]["registered"][a]
        assert got["root"] == run.served.registered[a].root.hex()
        assert (got["site"], got["step"], got["rows"]) == (sites[c.layer], FIRST + c.step, c.m)
    assert [r["ids"] for r in run.round_log] == [tap.ids_roots[r]["root"] for r in range(STEPS)]
    assert record["commitment"]["served"]["commit"]["arow"] == [tap.arows[c] for c in range(4)]
    assert PHC.manifest_reason(record) is None
    assert all(v["credit"] for v in run.gated().values())


def _zero(i, xq):
    return torch.zeros_like(xq)


def _one_row(i, xq):
    if i == 0:
        xq[1, 0] = 1 if int(xq[1, 0]) != 1 else 2
    return xq


@pytest.mark.parametrize("edit,own", [(_zero, 4), (_one_row, 1)], ids=["a-all-zero", "one-a-row-off"])
def test_a_served_a_off_the_commits_arow_is_refused(tmp_path, monkeypatch, edit, own):
    """H1 handed codes other than the ones the Commit acquired (every A all zero, the RT4 A = 0 probe; or one row of call
    0), with an honest Commit: those calls' AROWs are trees of the Commit's own at another root, so the window's record
    is refused (`manifest_reason`, until core's `credit._arow`) and no call is credited (`UNTIED`)."""
    record, run, tap = _window(tmp_path, monkeypatch, edit)
    why = run.details["roots1"]["reason"]
    assert why.startswith("roots1: call 0's A table ") and "not the Commit's AROW" in why
    assert tap.own == own and run.details["roots1"]["own_trees"] == own
    assert PHC.manifest_reason(record) == why
    assert {v["reason"] for v in run.gated().values()} == {CT.UNTIED}


def test_input_ids_off_the_prompts_are_refused(tmp_path, monkeypatch):
    """Step 0's Commit ids with one id that isn't its request's prompt token: the round roots are over the Commit's ids, so
    the record is one manifest, but the serving process refuses the window (Served.lean's layer-0 tie replaces it)."""
    record, run, _tap = _window(tmp_path, monkeypatch, ids=lambda s, x: [*x[:3], 99, *x[4:]] if s == 0 else x)
    assert run.owners[0] == [1, 1, 0, 0, 0]
    assert run.details["roots1"]["reason"] == ("roots1: step 0's slot 3 has input id 99 in the Commit, and request 0's "
                                               "prompt token 1 is 6: layer 0's input is not its round's request table")
    assert PHC.manifest_reason(record) is None
    assert {v["reason"] for v in run.gated().values()} == {CT.UNTIED}


def test_the_commits_record_is_under_calls_root(tmp_path, monkeypatch):
    """The Commit's record is bound into the served digest calls_root binds (`bind`): without it, or with an AROW or ids
    root edited, the record is refused."""
    record, _run, _tap = _window(tmp_path, monkeypatch)

    def edited(edit):
        doc = json.loads(json.dumps(record))
        edit(doc["commitment"]["served"])
        return PHC.manifest_reason(doc)
    assert "calls_root is not over" in edited(lambda s: s.pop("commit"))
    assert "calls_root is not over" in edited(lambda s: s["commit"]["arow"][0].update(root="00" * 64))
    assert "calls_root is not over" in edited(lambda s: s["commit"]["ids"][1].update(root="00" * 64))
    served = record["commitment"]["served"]
    plain = SV.digest_of(served["entries"], served["refused"])
    assert SV.served_digest(served) == CT.bind(plain, served["commit"]) != plain
    assert SV.served_digest({k: v for k, v in served.items() if k != "commit"}) == plain


def test_a_commit_record_off_its_window_is_refused():
    """`arow_reason` on records the window's own rows don't match: a refused tap, an AROW missing or of another call, and a
    round whose ids root isn't the Commit's."""
    entry = {"value": "p/a/64@0", "root": "aa" * 64, "domain": "dd", "leaves": 32, "words": 34}
    rows = [{"index": 0, "tables": {"registered": {"p/a/64@0": "aa" * 64, "p/zy@0": "bb" * 64}}}]
    commit = {"arow": [{**entry, "call": 0, "step": 3, "site": "q/row_quant", "rows": 1}], "ids": [{"round": 0, "root": "cc"}],
              "refused": []}

    def reason(commit=commit, rows=rows, ids="cc"):
        return CT.arow_reason({"commitment": {"served": {"entries": [entry], "commit": commit},
                                              "rounds": [{"round": 0, "ids": ids}]}, "calls": rows})
    assert reason() is None
    assert reason(commit={**commit, "refused": ["x"]}).startswith("roots1: the Commit's tap refused")
    assert reason(commit={**commit, "arow": []}) == "roots1: call 0 has no AROW in the Commit's record"
    assert reason(rows=rows + [{"index": 1, "tables": {"registered": {}}}]).startswith("roots1: call 1's manifest row")
    assert reason(commit={**commit, "arow": [{**commit["arow"][0], "words": 33}]}).startswith("roots1: call 0's A table")
    assert reason(ids="dd").startswith("roots1: round 0's root is not over the Commit's ids")
    assert reason(ids=None).startswith("roots1: round 0's root")
    assert CT.arow_reason({"commitment": {"served": {"entries": []}}, "calls": rows}) is None


def test_the_source_takes_every_linears_site_at_every_engine_step():
    """`source` names each of H1's linears by its module's name and taps its Int8RowQuant_v1 site at every engine step, so
    X-03's targets at any step are each site's x_q and s_a; a layer not in the model is refused."""
    model = Model()
    src = CT.source(Committer(), model, list(model.model.layers))
    want = {f"{CT.site(f'model.layers.{w}')}/{p}": 1 for w in range(2) for p in "01"}
    assert src.targets_at(0) == src.targets_at(41) == want
    assert src.quant.get((41, "model.layers.1")) == CT.site("model.layers.1") and src.quant.get((0, "model")) is None
    with pytest.raises(ValueError, match="not named modules"):
        CT.source(Committer(), model, [Linear(32, 3, Arm())])
    assert CT.Sites(["q"]).get((5, "q")) == "q/row_quant"
