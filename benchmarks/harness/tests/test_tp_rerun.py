"""served_commit's rerun driver (`benchmarks/served_commit/tp_rerun.py`): `Rerun` records each forward's rows as the schedule and
writes every rank's fingerprints with the names their codes index; `compare` finds two engine processes that served a window alike
bit-exact even when they coded its requests differently, names a perturbed rank's first difference by step and module, and tells a
changed schedule from changed values. Fake engine processes on CPU tensors, no engine."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

BENCH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BENCH))
try:
    from served_commit import tp_replay as R
    from served_commit import tp_rerun as M
finally:
    sys.path.remove(str(BENCH))


def test_runs_of_cuts_at_each_request_and_each_gap():
    req, pos = np.array([0, 0, 0, 1, 2, 2, 0]), np.array([4, 5, 6, 9, 0, 2, 7])
    assert M.runs_of(req, pos, ["d", "c", "e"]) == [["d", 4, 3], ["c", 9, 1], ["e", 0, 1], ["e", 2, 1], ["d", 7, 1]]
    assert M.runs_of(np.zeros(0, np.int64), np.zeros(0, np.int64), []) == []


def test_tokens_and_schedules_differ_where_they_first_do():
    assert M.tokens_differ({"d": [1, 2, 3], "c": [4]}, {"d": [1, 2, 3], "c": [4]}) == {}
    assert M.tokens_differ({"d": [1, 2, 3], "c": [4]}, {"d": [1, 9, 3], "e": [4]}) == {"c": 0, "d": 1, "e": 0}
    assert M.tokens_differ({"d": [1, 2]}, {"d": [1, 2, 3]}) == {"d": 2}
    s = [[["d", 0, 3]], [["d", 3, 1], ["c", 0, 2]]]
    assert M.schedule_differs(s, s) is None and M.schedule_differs(s, s[:1]) == 1
    assert M.schedule_differs(s, [[["d", 0, 2]], *s[1:]]) == 0


class _Attn(torch.nn.Module):
    """A DSA attention layer's shape: ``_sparse_indexer_and_attn`` fills ``output`` and the shared top-k buffer; rows don't mix."""

    def __init__(self, buf, w):
        super().__init__()
        self.topk_indices_buffer, self.w, self.indexer, self.skip_topk = buf, w, object(), False

    def forward(self, h):
        out = torch.empty_like(h)
        self._sparse_indexer_and_attn(h, output=out)
        return out

    def _sparse_indexer_and_attn(self, h, output):
        output.copy_(torch.tanh(h * self.w))
        self.topk_indices_buffer[:h.shape[0]] = (h[:, :4] * 100).to(torch.int32)


class _Scale(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.w = w

    def forward(self, h):
        return h * self.w


class _Layer(torch.nn.Module):
    def __init__(self, buf, w):
        super().__init__()
        self.self_attn, self.mlp = _Attn(buf, w), _Scale(w * 0.5)

    def forward(self, h):
        h = h + self.self_attn(h)
        return h + self.mlp(h)


class _Logits(torch.nn.Module):
    def forward(self, h):
        return torch.cat([h, -h], dim=-1)


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        buf = torch.full((16, 4), -1, dtype=torch.int32)
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(50, 8)
        self.model.layers = torch.nn.ModuleList([_Layer(buf, 0.5), _Layer(buf, 0.25)])
        self.model.norm = _Scale(2.0)
        self.logits_processor = _Logits()
        torch.nn.init.normal_(self.model.embed_tokens.weight, generator=torch.Generator().manual_seed(6))

    def forward(self, ids):
        h = self.model.embed_tokens(ids)
        for layer in self.model.layers:
            h = layer(h)
        return self.model.norm(h)


class _Runner:
    """The V2 runner's calls `Interior` wraps: ``execute_model`` calls ``prepare_inputs`` then the forward; ``sample`` the logits."""

    PROMPT = {"d": 3, "c": 2}

    def __init__(self):
        self.model = _Tiny()

    def get_model(self):
        return self.model

    def prepare_inputs(self, step):
        ids, rows = step
        qsl = np.concatenate([[0], np.cumsum([n for _r, n, _c in rows])]).astype(np.int32)
        return SimpleNamespace(num_reqs=len(rows), req_ids=[f"{r}-0a1b2c3d" for r, _n, _c in rows], query_start_loc_np=qsl,
                               num_computed_tokens_np=np.array([c for _r, _n, c in rows], dtype=np.int32),
                               prefill_len_np=np.array([self.PROMPT[r] for r, _n, _c in rows], dtype=np.int32),
                               logits_indices=torch.zeros(len(rows)), num_tokens_after_padding=len(ids), input_ids=torch.tensor(ids))

    def execute_model(self, step):
        self.ib = self.prepare_inputs(step)
        self.h = self.model(self.ib.input_ids)

    def sample(self, hidden, input_batch, grammar=None):
        return self.model.logits_processor(hidden[torch.tensor(input_batch.query_start_loc_np[1:] - 1, dtype=torch.long)])


#: the window: d's 3-token prompt and c's 2-token prompt prefilled together, then two decode steps
WINDOW = [([3, 7, 11, 5, 9], [("d", 3, 0), ("c", 2, 0)]), ([13, 1], [("d", 1, 3), ("c", 1, 2)]), ([4, 6], [("d", 1, 4), ("c", 1, 3)])]
#: the same rows, c's prompt split over two steps
OTHER_SCHEDULE = [([3, 7, 11, 5], [("d", 3, 0), ("c", 1, 0)]), ([13, 9], [("d", 1, 3), ("c", 1, 1)]), ([4, 1], [("d", 1, 4), ("c", 1, 2)]),
                  ([6], [("c", 1, 3)])]
ALONE_C = [([5, 9], [("c", 2, 0)]), ([1], [("c", 1, 2)])]
TOKENS = {"d": [13, 4], "c": [1, 6]}


def _process(tmp: Path, monkeypatch, name: str, arms: dict, *, world: int = 2, perturb: int | None = None) -> Path:
    """A fake engine process: ``world`` ranks serve ``arms`` (name -> steps) in order and `tp_rerun.run`'s files are written."""
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace())
    out, dump = tmp / name, tmp / "dump" / name
    out.mkdir(parents=True)
    dump.mkdir(parents=True)
    workers, rows, broken = [], [], []
    try:
        for rank in range(world):
            monkeypatch.setitem(sys.modules, "vllm.distributed", SimpleNamespace(get_tensor_model_parallel_rank=lambda rank=rank: rank))
            w = SimpleNamespace(model_runner=_Runner())
            if rank == perturb:
                w.model_runner.model.model.layers[1].mlp.w = 0.126
            workers.append(w)
            assert M.install(w)["dsa_attention"] == 2
        for arm, steps in arms.items():
            ranks = []
            for w in workers:
                R.begin(w)
                for s in steps:
                    w.model_runner.execute_model(s)
                    w.model_runner.sample(w.model_runner.h, w.model_runner.ib)
                drawn = sorted({r for _ids, rs in steps for r, _n, _c in rs})
                ranks.append(M.end(w, arm=arm, drawn=drawn, reference=R.REFERENCE, dump=str(dump)))
            result = {"wall_s": 0.5, "iterations": len(steps), "step_wall_s": [0.5 / len(steps)] * len(steps),
                      "tokens": {k: v for k, v in TOKENS.items() if any(r == k for _ids, rs in steps for r, _n, _c in rs)},
                      "finish_reasons": {}}
            row, bad = M.record(arm, result, ranks, world)
            rows.append(row)
            broken += bad
    finally:
        # every rank of this fake process shares one Python, whose live patches each rank's uninstall counts
        left = [R.uninstall(w) for w in workers if hasattr(w, "_tp_replay")]
    assert [x["hooks_left"] for x in left] == [0] * world and left[-1]["patches_left"] == 0
    (out / "versions.json").write_text(json.dumps({"dump": str(dump)}))
    (out / "arms.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "run.json").write_text(json.dumps({"world": world, "build_s": 1.0, "wall_s": 2.0, "gpu_s": 2.0 * world, "instrument_failures": broken}))
    return out


def test_rerun_records_the_schedule_and_dumps_named_fingerprints(tmp_path, monkeypatch):
    run = _process(tmp_path, monkeypatch, "a", {"served": WINDOW})
    row = json.loads((run / "arms.jsonl").read_text())
    assert row["schedule"] == [[["d", 0, 3], ["c", 0, 2]], [["d", 3, 1], ["c", 2, 1]], [["d", 4, 1], ["c", 3, 1]]]
    assert [x["rank"] for x in row["ranks"]] == [0, 1] and json.loads((run / "run.json").read_text())["instrument_failures"] == []
    keys, fp, requests, groups = M.load(tmp_path / "dump" / "a", "served", 1)
    assert requests == ["d", "c"] and groups[:4] == ["model.embed_tokens", "L0.attn", "L0.topk", "L0.self_attn"]
    assert "logits_processor" in groups and fp.shape == (len(keys), 2) and len(keys) == row["ranks"][1]["entries"] > 0
    assert len(np.unique((keys >> R.FIELD) & R.MASK)) == 5


def test_compare_finds_reruns_bit_exact_and_names_what_differs(tmp_path, monkeypatch, capsys):
    a = _process(tmp_path, monkeypatch, "a", {"served": WINDOW, "served-2": WINDOW})
    b = _process(tmp_path, monkeypatch, "b", {"alone-c": ALONE_C, "served": WINDOW, "served-2": WINDOW})
    c = _process(tmp_path, monkeypatch, "c", {"served": WINDOW}, perturb=1)
    d = _process(tmp_path, monkeypatch, "d", {"served": OTHER_SCHEDULE})
    assert M.load(tmp_path / "dump" / "b", "served", 0)[2] == ["c", "d"]
    assert M.compare([a, b, c, d], tmp_path / "cross.json") == 0
    doc = json.loads((tmp_path / "cross.json").read_text())
    same, perturbed, moved = (doc["across"][str(p)] for p in (b, c, d))
    assert set(same) == {"served", "served-2"} and all(v["bit_exact"] for v in same.values())
    v = same["served"]
    assert (v["ranks"], v["differ"], v["only_first"], v["only_this"], v["first_differ"], v["schedule_first_differ"]) == (2, 0, 0, 0, None, None)
    assert v["compared"] == v["equal"] > 0
    v = perturbed["served"]
    assert not v["bit_exact"] and v["ranks_differ"] == 1 and v["per_rank"]["0"]["differ"] == 0
    assert v["first_differ"] == {"rank": 1, "step": 0, "request": "d", "pos": 0, "group": "L1.mlp"}
    assert set(v["kinds_differ"]) == {"mlp", "layer", "model.norm", "root", "logits_processor"} and v["per_rank"]["1"]["positions_differ"] == 9
    v = moved["served"]
    assert not v["bit_exact"] and v["schedule_first_differ"] == 0 and v["forwards"] == [3, 4]
    assert (v["differ"], v["only_first"], v["only_this"], v["tokens_differ"]) == (0, 0, 0, {})
    assert doc["in_engine"][str(a)]["served-2"]["bit_exact"] and doc["in_engine"][str(b)]["served-2"]["bit_exact"]
    assert doc["in_engine"][str(b)]["alone-c"]["compared"] == 0
    assert doc["gpu_s"][str(a)] == {"total": 4.0, "build_s": 1.0, "served": 1.0, "served-2": 1.0}
    assert f"{b} vs {a}, served: bit-exact True" in capsys.readouterr().out
