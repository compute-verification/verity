"""served_commit's replay driver (`benchmarks/served_commit/tp_replay.py`): its fingerprint is the keyed sum of a row's 32-bit words
mod 2^64 for every dtype and byte width, sees one flipped bit, doesn't depend on the row's place or alignment, and refuses a row
wider than its keys; its rows and logits rows read an input batch as the runner builds one; `flatten` and `compare` count equal,
differing and one-sided entries and name the first difference; its arms keep the drawn requests the same across arms and the
fillers in their companions' shapes; `span` and `summarize` read a run's steps. CPU tensors, no engine."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

DRIVER = Path(__file__).resolve().parents[2] / "served_commit" / "tp_replay.py"


def _driver():
    spec = importlib.util.spec_from_file_location("served_commit_tp_replay", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


D = _driver()
KEYS = D.key_table()


def _reference(row: bytes, keys) -> tuple[int, int]:
    row = row + b"\0" * (-len(row) % 4)
    w = [int.from_bytes(row[i:i + 4], "little") for i in range(0, len(row), 4)]
    out = []
    for lane in range(2):
        s = sum(x * int(k) for x, k in zip(w, keys[lane, :len(w)].tolist())) % (1 << 64)
        out.append(s - (1 << 64) if s >= 1 << 63 else s)
    return tuple(out)


@pytest.mark.parametrize("dtype,shape", [(torch.bfloat16, (3, 6)), (torch.float32, (2, 5)), (torch.int32, (4, 3)),
                                         (torch.uint8, (3, 5)), (torch.bool, (2, 7)), (torch.int64, (5,))])
def test_fingerprint_is_the_keyed_word_sum_mod_2_64(dtype, shape):
    g = torch.Generator().manual_seed(1)
    t = (torch.randint(0, 255, shape, generator=g) % 2).to(dtype) if dtype == torch.bool else torch.randint(-99, 99, shape, generator=g).to(dtype)
    fp = D.fingerprint(t, shape[0], KEYS)
    rows = t.contiguous().view(torch.uint8).reshape(shape[0], -1)
    assert [tuple(r) for r in fp.tolist()] == [_reference(bytes(rows[i].tolist()), KEYS) for i in range(shape[0])]


def test_fingerprint_sees_one_bit_and_only_in_its_row():
    t = torch.randn(4, 64, generator=torch.Generator().manual_seed(2)).to(torch.bfloat16)
    u = t.clone()
    u.view(torch.int16)[2, 17] ^= 1 << 9
    a, b = D.fingerprint(t, 4, KEYS), D.fingerprint(u, 4, KEYS)
    assert (a[[0, 1, 3]] == b[[0, 1, 3]]).all() and (a[2] != b[2]).all()


def test_fingerprint_of_a_row_is_the_same_at_any_place_and_alignment():
    t = torch.randint(0, 255, (6, 5), dtype=torch.uint8, generator=torch.Generator().manual_seed(3))
    whole = D.fingerprint(t, 6, KEYS)
    assert t[1:].storage_offset() == 5
    assert torch.equal(D.fingerprint(t[1:], 5, KEYS), whole[1:])
    assert torch.equal(D.fingerprint(t[:, :3], 6, KEYS), D.fingerprint(t[:, :3].clone(), 6, KEYS))
    assert torch.equal(D.fingerprint(t, 2, KEYS), whole[:2])


def test_fingerprint_chunks_rows_the_same(monkeypatch):
    t = torch.randint(-9, 9, (7, 10), dtype=torch.int32, generator=torch.Generator().manual_seed(4))
    whole = D.fingerprint(t, 7, KEYS)
    monkeypatch.setattr(D, "CHUNK_WORDS", 20)
    assert torch.equal(D.fingerprint(t, 7, KEYS), whole)


def test_fingerprint_refuses_a_row_wider_than_its_keys():
    with pytest.raises(ValueError, match="wider than the key table's 4 .KEY_WORDS."):
        D.fingerprint(torch.zeros(2, 5, dtype=torch.int32), 2, KEYS[:, :4])


def test_keys_are_odd_and_fixed():
    assert KEYS.shape == (2, D.KEY_WORDS) and bool((KEYS & 1).all())
    assert torch.equal(D.key_table(), KEYS) and not torch.equal(KEYS[0], KEYS[1])
    a, b = D.name_key("model.layers.3.mlp#0/0")
    assert a & 1 and b & 1 and D.name_key("model.layers.3.mlp#1/0") != (a, b)


def test_groups_and_kinds():
    assert [D.group_of(n) for n in ("model.layers.12.self_attn.q_b_proj", "model.layers.12.mlp", "model.layers.12",
                                    "model.layers.4.topk", "model.embed_tokens", "logits_processor", "")] == \
        ["L12.self_attn", "L12.mlp", "L12", "L4.topk", "model.embed_tokens", "logits_processor", "root"]
    assert [D.kind_of(g) for g in ("L12.self_attn", "L12", "model.norm", "logits_processor")] == ["self_attn", "layer", "model.norm",
                                                                                                    "logits_processor"]


def _ib(req_ids, scheduled, computed, prompt, logits=None):
    qsl = np.concatenate([[0], np.cumsum(scheduled)]).astype(np.int32)
    return SimpleNamespace(num_reqs=len(req_ids), req_ids=list(req_ids), query_start_loc_np=qsl,
                           num_computed_tokens_np=np.array(computed, dtype=np.int32), prefill_len_np=np.array(prompt, dtype=np.int32),
                           logits_indices=torch.zeros(len(req_ids) if logits is None else logits))


def test_rows_of_and_logit_rows_read_the_input_batch():
    codes: dict[str, int] = {}
    code = lambda r: codes.setdefault(r, len(codes))  # noqa: E731
    ib = _ib(["d-long-0a1b2c3d", "c0-deadbeef", "d-short-01234567"], [3, 1, 2], [2048, 127, 0], [3072, 128, 256])
    req, pos = D.rows_of(ib, code)
    assert codes == {"d-long": 0, "c0": 1, "d-short": 2}
    assert req.tolist() == [0, 0, 0, 1, 2, 2] and pos.tolist() == [2048, 2049, 2050, 127, 0, 1]
    lreq, lpos, keep = D.logit_rows(ib, code)
    assert lreq.tolist() == [0, 1, 2] and lpos.tolist() == [2050, 127, 1] and keep.tolist() == [False, True, False]
    assert D.logit_rows(_ib(["a-0a1b2c3d"], [2], [0], [2], logits=2), code) is None


def _step(req, pos, groups, fp):
    return (np.array(req, np.int64), np.array(pos, np.int64), np.array(groups, np.int64), np.array(fp, np.int64))


def test_flatten_and_compare_count_and_name_the_first_difference():
    groups = ["model.embed_tokens", "L0.self_attn", "L0.topk"]
    fp = lambda *v: [[[x, x + 1] for x in row] for row in v]  # noqa: E731
    ref = D.flatten([_step([0, 0, 1], [0, 1, 0], [0, 1, 2], fp([1, 2, 3], [4, 5, 6], [7, 8, 9])),
                     _step([0], [2], [0, 1, 2], fp([10], [11], [12]))])
    run = D.flatten([_step([0, 0], [0, 1], [0, 1, 2], fp([1, 2], [4, 5], [7, 8])),
                     _step([0, 0], [1, 3], [0, 1, 2], fp([2, 13], [50, 14], [80, 15]))])
    assert ref[2] == 0 and run[2] == 3
    res = D.compare(ref[:2], run[:2], {"d": 0, "e": 1}, groups)
    assert res["d"] == {"compared": 6, "equal": 4, "differ": 2, "only_ref": 3, "only_run": 3, "positions": 2, "positions_differ": 1,
                        "first_differ": {"pos": 1, "group": "L0.self_attn"}, "kinds_differ": {"self_attn": 1, "topk": 1}}
    assert res["e"]["compared"] == 0 and res["e"]["only_ref"] == 3 and res["e"]["first_differ"] is None
    same = D.compare(ref[:2], ref[:2], {"d": 0}, groups)["d"]
    assert (same["compared"], same["equal"], same["differ"], same["first_differ"]) == (9, 9, 0, None)


def test_arms_keep_the_drawn_requests_and_their_companions_shapes():
    base = np.random.default_rng(5).integers(3, 1000, size=256).tolist()
    A = D.arms(base, 1000)
    served = {r["request_id"]: r for r in A["served"]}
    assert list(A) == ["served", "served-2", "shape", "alone-d-short", "alone-d-long", "alone-d-seed", "other", "warm"]
    assert [(r["request_id"], len(r["prompt_token_ids"])) for r in A["served"]][:3] == [("d-short", 256), ("d-long", 3072), ("d-seed", 512)]
    firsts = [tuple(r["prompt_token_ids"][:64]) for arm in ("served", "other") for r in A[arm] if not (arm == "other" and r["request_id"] in D.DRAWN)]
    assert len(set(firsts)) == len(firsts) == len(D.SERVED) + len(D.OTHER)
    for arm, reqs in A.items():
        assert len(reqs) <= D.MAX_SEQS and max(len(r["prompt_token_ids"]) + r["max_tokens"] for r in reqs) <= 4096
        for r in reqs:
            if r["request_id"] in D.DRAWN:
                assert {k: v for k, v in r.items() if k != "arrive_step"} == {k: v for k, v in served[r["request_id"]].items() if k != "arrive_step"}
    for s, f in zip(A["served"], A["shape"]):
        if s["request_id"] in D.DRAWN:
            assert f is s
        else:
            assert f["request_id"].startswith("filler-") and f["prompt_token_ids"] != s["prompt_token_ids"]
            assert {k: v for k, v in f.items() if k not in ("request_id", "prompt_token_ids")} == \
                {k: v for k, v in s.items() if k not in ("request_id", "prompt_token_ids")} and len(f["prompt_token_ids"]) == len(s["prompt_token_ids"])
    assert sum(len(r["prompt_token_ids"]) > 2048 for r in A["other"] if r["request_id"] not in D.DRAWN) == 4
    assert [(r["request_id"], r["arrive_step"]) for r in A["other"] if r["request_id"] in D.DRAWN] == list(D.OTHER_DRAWN)
    prod, long = A["warm"]
    assert prod["request_id"] == D.PRODUCER and prod["max_tokens"] == 1 and prod["prompt_token_ids"] == long["prompt_token_ids"]
    assert long["request_id"] == "d-long" and long["arrive_step"] > 10 ** 6


def test_steps_and_span():
    st = D.Steps()
    outs = [[SimpleNamespace(request_id="p", finished=False, num_cached_tokens=0)],
            [SimpleNamespace(request_id="p", finished=True, num_cached_tokens=0)],
            [SimpleNamespace(request_id="d-long", finished=False, num_cached_tokens=3008)],
            [SimpleNamespace(request_id="d-long", finished=True, num_cached_tokens=3008)]]
    for o in outs:
        assert st(lambda o=o: o) is o
    assert st.done == {"p": 1, "d-long": 3} and st.cached == {"p": 0, "d-long": 3008}
    walls = [0.5, 0.25, 0.125, 1.0]
    assert D.span(walls, st.done, "d-long", "p") == 1.125 and D.span(walls, st.done, "d-long") == 1.875
    assert D.span(walls, st.done, "x") is None


class _Attn(torch.nn.Module):
    """A DSA attention layer's shape: its forward fills ``output`` in ``_sparse_indexer_and_attn``, which also writes the shared
    top-k buffer; rows don't mix."""

    def __init__(self, buf, w):
        super().__init__()
        self.topk_indices_buffer, self.w, self.indexer, self.skip_topk = buf, w, object(), False
        self.o_proj = _Scale(w + 1)

    def forward(self, h):
        out = torch.empty_like(h)
        self._sparse_indexer_and_attn(h, output=out)
        return self.o_proj(out)

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

    def __init__(self):
        self.model = _Tiny()

    def get_model(self):
        return self.model

    def prepare_inputs(self, step):
        ids, rows = step
        ib = _ib([f"{r}-0a1b2c3d" for r, _n, _c in rows], [n for _r, n, _c in rows], [c for _r, _n, c in rows], [3] * len(rows))
        return SimpleNamespace(**vars(ib), num_tokens_after_padding=len(ids), input_ids=torch.tensor(ids))

    def execute_model(self, step):
        self.ib = self.prepare_inputs(step)
        self.h = self.model(self.ib.input_ids)

    def sample(self, hidden, input_batch, grammar=None):
        return self.model.logits_processor(hidden[torch.tensor(input_batch.query_start_loc_np[1:] - 1, dtype=torch.long)])


def _serve(runner, steps):
    for s in steps:
        runner.execute_model(s)
        runner.sample(runner.h, runner.ib)


def test_interior_folds_every_module_per_row_and_compares_runs_across_batches(monkeypatch):
    """A request's rows fold to the same values whether it was prefilled in one chunk beside another request or in two alone; a
    changed prompt token differs at its position only (these rows don't mix), from its embedding on; the partial prefill's logits
    row is dropped; and nothing stays installed."""
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "vllm.distributed", SimpleNamespace(get_tensor_model_parallel_rank=lambda: 0))
    worker = SimpleNamespace(model_runner=_Runner())
    assert D.install(worker) == {"rank": 0, "modules": 14, "dsa_attention": 2}
    d, c = [3, 7, 11, 13], [5, 9, 2]
    batched = [(d[:3] + c, [("d", 3, 0), ("c", 3, 0)]), ([d[3], 1], [("d", 1, 3), ("c", 1, 3)])]
    alone = [(d[:2], [("d", 2, 0)]), ([d[2]], [("d", 1, 2)]), ([d[3]], [("d", 1, 3)])]
    changed = [(d[:2], [("d", 2, 0)]), ([d[2] + 1], [("d", 1, 2)]), ([d[3]], [("d", 1, 3)])]
    out = {}
    for arm, steps in (("served", batched), ("alone", alone), ("changed", changed)):
        D.begin(worker)
        _serve(worker.model_runner, steps)
        out[arm] = D.end(worker, arm=arm, drawn=["d", "c"], reference="served")
    ref, al, ch = (out[a]["per_request"]["d"] for a in ("served", "alone", "changed"))
    forward = ["model.embed_tokens", "L0.attn", "L0.topk", "L0.self_attn", "L0.mlp", "L0", "L1.attn", "L1.topk", "L1.self_attn", "L1.mlp",
               "L1", "model.norm", "root"]
    assert out["served"]["groups"] == len(forward) + 1 and out["served"]["duplicates"] == 0 and out["served"]["skipped"] == {}
    entries = 4 * len(forward) + 2
    assert (ref["compared"], ref["equal"], ref["differ"], ref["positions"]) == (entries, entries, 0, 4)
    assert (al["compared"], al["equal"], al["only_ref"], al["only_run"]) == (entries, entries, 0, 0)
    assert ch["first_differ"] == {"pos": 2, "group": "model.embed_tokens"}
    assert (ch["positions_differ"], ch["differ"], ch["equal"], ch["only_ref"], ch["only_run"]) == (1, len(forward) + 1, entries - len(forward) - 1, 0, 0)
    assert ch["kinds_differ"]["topk"] == 2 and ch["kinds_differ"]["logits_processor"] == 1
    assert out["alone"]["per_request"]["c"]["compared"] == 0 and out["alone"]["per_request"]["c"]["only_ref"] == entries
    assert D.uninstall(worker) == {"rank": 0, "hooks_left": 0, "patches_left": 0}
    runner = worker.model_runner
    assert not {"execute_model", "prepare_inputs", "sample"} & set(vars(runner))
    assert not any("_sparse_indexer_and_attn" in vars(m) for m in runner.model.modules())


def test_summarize_sums_ranks_and_takes_medians():
    per = lambda differ, first: {"compared": 10, "equal": 10 - differ, "differ": differ, "only_ref": 0, "only_run": 0,  # noqa: E731
                                 "positions_differ": int(differ > 0), "first_differ": first, "kinds_differ": {"topk": differ} if differ else {}}
    rows = [{"pass": 1, "arm": "served", "tokens_equal_served": {"d-long": True}, "ranks": [{"per_request": {"d-long": per(0, None)}}] * 2},
            {"pass": 1, "arm": "alone-d-long", "tokens_equal_served": {"d-long": True},
             "ranks": [{"per_request": {"d-long": per(2, {"pos": 3100, "group": "L5.topk"})}}, {"per_request": {"d-long": per(1, {"pos": 3090, "group": "L5.attn"})}}]},
            *[{"pass": 2, "arm": "alone-d-long", "wall_s": w, "tokens_equal_pass1": True, "spans": {"d-long": w}, "cached_tokens": {}} for w in (2.0, 3.0)],
            {"pass": 2, "arm": "warm", "wall_s": 2.0, "tokens_equal_pass1": True, "spans": {"d-long": 0.5}, "cached_tokens": {"d-long": 3008}}]
    s = D.summarize(rows, 8)
    assert s["arms"]["served"]["drawn"]["d-long"]["bit_exact"] is True
    d = s["arms"]["alone-d-long"]["drawn"]["d-long"]
    assert (d["bit_exact"], d["differ"], d["ranks_differ"], d["first_differ"], d["kinds_differ"]) == (False, 3, 2, {"pos": 3090, "group": "L5.attn"}, {"topk": 3})
    assert s["arms"]["alone-d-long"]["gpu_s"] == 20.0
    assert s["replay"]["d-long cold"] == {"gpu_s": 20.0, "reps": 2}
    assert s["replay"]["d-long warm"] == {"gpu_s": 4.0, "reps": 1, "cached_tokens": [3008]}
