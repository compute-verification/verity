"""The narrowed-rows source (`acquire.sources.narrowed_source`) on CPU tensors with a recording committer: at a step with prompt rows
it commits the manifest's columns of every row of the one tensor the module returns, after the module's own acquisition and before
anything else runs (the gather collector flushes at once); nothing at a decode step or outside a step; a return of two tensors or of
too few columns is a named collector error; a manifest whose narrowed rows lack their columns refuses the Commit, an external
acquisition attaches nothing, and on a TP Commit once one rank attaches it every rank must.  That its words are the Program's
values, and that a Commit without them fails coverage by name: `tests/query/test_narrowed_producers.py`."""
from __future__ import annotations

import json
import types

import pytest

try:
    import torch
except ModuleNotFoundError:
    pytest.skip("could not import 'torch'", allow_module_level=True)

from verity_vllm.acquire.sources import narrowed_source as NS
from verity_vllm.acquire.sources import taps as TAPS

OP = "model.layers.1.self_attn.qkv_proj"
T, QKV = 5, 12


def _row(**kw):
    return {"family": "instance_outputs", "op_path": OP, "output_member": "out@8", "step": 0, "request_id": "r0", "rows": T - 1,
            "width": 8, "dtype": "bf16", "narrowed_of": "out", "columns": [4, 12], **kw}


MANIFEST = {"identities": [{"op_path": OP, "output_member": "out", "step": 0, "rows": 1, "width": QKV, "runtime_rows": [T - 1, T]},
                           _row()]}


class Qkv(torch.nn.Module):
    def __init__(self, ret) -> None:
        super().__init__()
        self.ret = ret

    def forward(self, _x):
        return self.ret


def _model(ret) -> torch.nn.Module:
    m = torch.nn.Module()
    m.model = torch.nn.Module()
    m.model.layers = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()])
    m.model.layers[1].self_attn = torch.nn.Module()
    m.model.layers[1].self_attn.qkv_proj = Qkv(ret)
    return m


class Committer:
    def __init__(self, model=None, gather: bool = False) -> None:
        self.model, self.name, self.errors, self.sources, self.events = model, "rec", [], [], []
        self._cur, self._plan_shape, self.gather = None, None, gather
        self._nc = types.SimpleNamespace(flush=lambda: self.events.append(("flush",)))

    def attach_source(self, src) -> None:
        self.sources.append(src)

    def _acquire(self, name, t, _job) -> None:
        self.events.append(("acquire", name, t.clone()))

    def acquired(self) -> list:
        return [e for e in self.events if e[0] == "acquire"]


def _run(com: Committer, src: NS.NarrowedRowsSource, shape):
    com._cur, com._plan_shape = types.SimpleNamespace(step=0), shape
    return com.model.model.layers[1].self_attn.qkv_proj(None)


def test_it_commits_the_columns_of_every_row_at_steps_with_prompt_rows_only() -> None:
    out = torch.arange(T * QKV, dtype=torch.float32).reshape(T, QKV).to(torch.bfloat16)
    com = Committer(_model((out, None)), gather=True)
    src = NS.attach(com, MANIFEST)
    assert com.sources == [src] and src.own_names == {f"{OP}/out@8"}
    src.install()
    com.model.model.layers[1].self_attn.qkv_proj(None)                       # outside a step
    _run(com, src, (1, 1))                                                    # a decode step
    _run(com, src, (3, 3))                                                    # every request at its last row
    assert com.events == []
    _run(com, src, (T, 1))
    assert [e[:2] for e in com.events] == [("acquire", f"{OP}/out@8"), ("flush",)]
    assert torch.equal(com.events[0][2], out[:, 4:12]) and src.stats == {"tensors": 1, "bytes": T * 8 * 2, "rows": T, "sites": 1, "errors": 0}
    com.events.clear()
    _run(com, src, None)                                                      # a committer that records no step shape: every step
    assert len(com.acquired()) == 1
    src.uninstall()
    _run(com, src, (T, 1))
    assert len(com.acquired()) == 1 and not com.errors


def test_an_arena_committer_is_not_flushed_and_a_bare_tensor_is_its_rows() -> None:
    out = torch.randn(T, 3, 4)
    com = Committer(_model(out))
    src = NS.attach(com, MANIFEST)
    src.install()
    _run(com, src, (T, 2))
    assert [e[:2] for e in com.events] == [("acquire", f"{OP}/out@8")] and torch.equal(com.events[0][2], out.reshape(T, QKV)[:, 4:12])


@pytest.mark.parametrize("ret,msg", [((torch.zeros(T, QKV), torch.zeros(T, QKV)), "with 2 tensors"),
                                     ((torch.zeros(T, 10), None), "rows of 10 columns; out@8 is its columns [4, 12)")])
def test_a_return_it_cannot_read_the_columns_of_is_a_named_error(ret, msg) -> None:
    com = Committer(_model(ret))
    src = NS.attach(com, MANIFEST)
    src.install()
    _run(com, src, (T, 1))
    assert com.acquired() == [] and len(com.errors) == 1 and msg in com.errors[0] and src.stats["errors"] == 1


def test_the_sites_are_the_manifests_narrowed_rows_with_their_columns() -> None:
    assert NS.manifest_sites(MANIFEST) == {OP: {"out@8": {"columns": (4, 12), "bytes": 2}}}
    assert NS.manifest_sites({"identities": [_row(), _row(step=3, request_id="r1", dtype="f32")]}) == {OP: {"out@8": {"columns": (4, 12), "bytes": 2}}}
    for bad, msg in ((_row(columns=None), "records columns None for width 8"), (_row(columns=[0, 12]), "records columns [0, 12] for width 8")):
        with pytest.raises(ValueError, match=msg.replace("[", r"\[").replace("]", r"\]")):
            NS.manifest_sites({"identities": [bad]})
    with pytest.raises(ValueError, match=r"out@8 at columns \(4, 12\) and \(0, 8\)"):
        NS.manifest_sites({"identities": [_row(), _row(step=1, columns=[0, 8])]})
    src = NS.NarrowedRowsSource(Committer(), None, NS.manifest_sites(MANIFEST))
    assert src.extra_step_bytes({}, None) == {"per_token_bytes": 16, "per_seq_bytes": 0, "tensors_per_step": 1}
    assert [c["class"] for c in src.declaration_classes()] == [NS.SHORT_NAME]


def test_attach_refuses_a_manifest_without_columns_and_leaves_an_external_acquisition_alone() -> None:
    assert NS.attach(Committer(), {"identities": [MANIFEST["identities"][0]]}) is None
    with pytest.raises(SystemExit):
        NS.attach(Committer(), {"identities": [_row(columns=None)]})
    com = Committer()
    com.external_acquisition = True
    assert NS.attach(com, MANIFEST) is None and com.sources == []
    with pytest.raises(RuntimeError, match="not modules of the model"):
        NS.attach(Committer(torch.nn.Linear(2, 2)), MANIFEST).install()


def test_the_taps_attach_it_whenever_the_manifest_names_narrowed_rows() -> None:
    assert NS.SHORT_NAME in TAPS.attached(MANIFEST) and NS.SHORT_NAME in TAPS.attached({**MANIFEST, "ranks": 2})
    assert NS.SHORT_NAME not in TAPS.attached({"identities": [MANIFEST["identities"][0]]})
    com = Committer(_model((torch.zeros(T, QKV), None)))
    TAPS.attach(com, MANIFEST, (), TAPS.TapSources(), explicit_classes=True)
    assert [type(s).__name__ for s in com.sources] == ["NarrowedRowsSource"]


def test_every_tp_rank_whose_rows_are_narrowed_must_attach_it(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"identities": [_row(rank=r) for r in (0, 1)]}))
    attach = TAPS.for_ranks(TAPS.TapSources())
    made = [{"rank": r, "tap_sources": attach(Committer(), _model((torch.zeros(T, QKV), None)), str(path), r)} for r in (0, 1)]
    versions: dict = {}
    assert TAPS.require_on_every_rank(made, (), versions, TAPS.TapSources(), explicit_classes=True) == ()
    assert versions["narrowed_source"] == {0: {"sites": 1}, 1: {"sites": 1}}
    path.write_text(json.dumps({"identities": [_row(rank=0), _row(rank=1, columns=None)]}))
    other = attach(Committer(), _model((torch.zeros(T, QKV), None)), str(path), 1)
    assert "SystemExit" in other["narrowed_source"]["error"]
    with pytest.raises(SystemExit, match=r"narrowed source did not attach on ranks \[1\]"):
        TAPS.require_on_every_rank([made[0], {"rank": 1, "tap_sources": other}], (), {}, TAPS.TapSources(), explicit_classes=True)
