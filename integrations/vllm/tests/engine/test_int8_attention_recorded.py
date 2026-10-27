"""The int8 attention fold (`observe.fold.patterns.int8.Int8AttentionScope`) on RECORDED events and values: Llama-3.1-8B served
with `TargetProfile.int8_exact` "all" on one RTX PRO 6000 (sm_120), the batched Triton ops of `engine.int8_ops`.

`fixtures/llama31_8b_int8_attn_l0.jsonl.gz` is layer 0 of the Match capture `int8_attention_capture_gpu` took in run RUN (vLLM
d9105ea80, eager, observer on, steps 0 and 1 snapshotted): two requests of 5 and 3 prompt tokens, step 0 their batched prefill (8
rows) and step 1 their batched decode (2 rows), cut by `int8_attention_cut` (its Note names the source log and its sha256) with the
words the scope's fold reads and writes beside it (`.values.npz`: the requests' KV slots of every head, the query and output rows,
kv_quant_batch's padded results).  Each step folds as recorded, every event of the scope claimed, to `Int8KvQuant_v1{s,128}` per
request and KV head then `Int8LutAttentionRow_v1{128,pos+1,SCALE}` per (row, head), and every Compute evaluated by its registered
row kernel (`program.kernels.int8_rows`, the replay's) on the captured operand words gives the captured output words, word for word
(272 Computes and 53,248 words at step 0, 80 and 32,768 at step 1).
A flipped word of a request's K window, a flipped output word, or a doubled K scale in the header is refused, at exactly the
Computes that read it.
"""
from __future__ import annotations

__models__ = ("unsloth/Llama-3.1-8B",)

import dataclasses
import functools
from pathlib import Path

import numpy as np
import pytest

from tests.engine import int8_attention_cut as C
from verity_vllm.observe.events import Note, RunHeader, ViewDesc
from verity_vllm.observe.fold.resolve_log import collect_roots
from verity_vllm.observe.fold.resolver import Compute, Const, Group, spec_label
from verity_vllm.observe.log import read_log, split_steps
from verity.ml.definitions.numerics.int8_rows import ROWS
from verity_vllm.query.manifest import format as RM

FIXTURE = Path(__file__).parent / "fixtures" / "llama31_8b_int8_attn_l0.jsonl.gz"
RUN = "r20261024-120845-b375"
SOURCE_SHA256 = "ebe1bc6a6f289dac162d85655f64e46bbe742d03be8b70c3a3c8728c242ea8be"
LAYER, H, G, D, PROMPTS = 0, 32, 8, 128, (5, 3)
SCALE = D ** -0.5


class Memory:
    """One step's words of the cut (`int8_attention_cut.image`): alloc -> runs of storage elements and their words."""

    def __init__(self, allocs: dict[int, tuple[np.ndarray, np.ndarray]]):
        self.allocs = allocs

    def _at(self, v: ViewDesc) -> tuple[np.ndarray, np.ndarray]:
        runs, words = self.allocs[v.alloc]
        el = C._storage_indices(v).reshape(-1)
        offs = np.r_[0, np.cumsum(runs[:, 1])[:-1]]
        i = np.searchsorted(runs[:, 0], el, side="right") - 1
        if (i < 0).any() or (el - runs[i, 0] >= runs[i, 1]).any():
            raise KeyError(f"{v!r} reads elements the capture did not keep")
        return words, offs[i] + el - runs[i, 0]

    def words(self, v: ViewDesc) -> np.ndarray:
        words, at = self._at(v)
        return words[at]

    def flipped(self, v: ViewDesc, bit: int = 0) -> "Memory":
        """This memory with bit `bit` of the first word of `v` flipped."""
        words, at = self._at(v)
        w = words.copy()
        w[at[0]] ^= w.dtype.type(1 << bit)
        return Memory({**self.allocs, v.alloc: (self.allocs[v.alloc][0], w)})


def operand(o, mem: Memory) -> np.ndarray:
    if isinstance(o, Const):
        return np.asarray(o.bits if isinstance(o.bits, tuple) else (o.bits,), dtype=np.uint32)
    return np.concatenate([mem.words(v) for v in C.rows_of(o)])


def _words(x, like: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(np.asarray(x)).reshape(-1)
    if a.dtype.kind == "f":
        a = a.view(f"u{a.itemsize}")
    return a.astype(like.dtype)


def refused(c: Compute, mem: Memory) -> list[str]:
    """Each output of `c` its row kernel's words on the captured operands do not reproduce, by name."""
    fn, names = ROWS[c.spec.id.split("{")[0]]
    got = fn(RM.spec_statics(c.spec.id), [operand(o, mem) for o in c.operands])
    out = []
    for name, o in zip(names, c.outputs):
        want = mem.words(o.view)
        n = int((_words(got[name], want) != want).sum())
        if n:
            out.append(f"{spec_label(c.spec)} {c.hint} {c.attribution} output {name}: {n} of {want.size} words differ")
    return out


@functools.lru_cache(maxsize=None)
def _cut() -> tuple[RunHeader, dict, dict[int, list], dict[int, Memory]]:
    header, roots, steps = None, {}, {}
    for pre, step in split_steps(read_log(FIXTURE)):
        header = next((e for e in pre if isinstance(e, RunHeader)), header)
        collect_roots(pre, roots)
        if step:
            steps[step[0].step] = step
    with np.load(C.values_path(FIXTURE)) as z:
        arrays = {k: z[k] for k in z.files}
    mems = {s: Memory({int(k.split("_")[1]): (arrays[k], arrays["w" + k[1:]]) for k in arrays if k.startswith(f"r{s}_")}) for s in steps}
    return header, roots, steps, mems


def _fold(step: int, header: RunHeader | None = None) -> tuple[Group, object]:
    h, roots, steps, _ = _cut()
    return C.fold_scope(header or h, steps[step], roots)


def _computes(group: Group) -> list[Compute]:
    return [p for p in group.parts if isinstance(p, Compute)]


def test_fixture_is_layer_0_of_the_served_int8_capture():
    header, _, steps, mems = _cut()
    p = header.profile
    assert p["case"] == C.ROLE and p["model"]["repo"] == "verity-int8/Llama-3.1-8B-INT8"
    assert "RTX PRO 6000" in p["gpu"]["name"] and p["target"]["int8_exact"] == "all" and p["vllm"]["commit"] == "d9105ea80"
    assert len(p["backends"]["int8_k_scales"][f"model.layers.{LAYER}.self_attn.attn"]) == G
    (note,) = [e for e in read_log(FIXTURE) if isinstance(e, Note)]
    assert note.text == "int8_attention_cut" and (note.data["layer"], note.data["steps"]) == (LAYER, [0, 1])
    assert note.data["sha256"] == SOURCE_SHA256
    windows = {s: sorted(1 + max(p for q, p in st[0].rows if q == r) for r, _ in st[0].rows) for s, st in steps.items()}
    assert {s: len(st[0].rows) for s, st in steps.items()} == {0: sum(PROMPTS), 1: len(PROMPTS)}
    assert {s: sorted(set(w)) for s, w in windows.items()} == {0: sorted(PROMPTS), 1: sorted(n + 1 for n in PROMPTS)}
    assert sorted(mems) == [0, 1] and max(C.values_path(FIXTURE).stat().st_size, FIXTURE.stat().st_size) < 256 * 1024


@pytest.mark.parametrize("step", [0, 1])
def test_each_step_folds_to_kv_quant_per_request_and_head_then_a_row_per_row_and_head(step):
    """In the step's request order (vLLM's batch: the 3-token request first at prefill, the 5-token one first at decode), each
    request's window its last row's position + 1."""
    group, scope = _fold(step)
    assert group.consumed == tuple(sorted(scope.ids())) and group.pattern == "int8.window_attention"
    rows = _cut()[2][step][0].rows
    want = []
    for r in dict.fromkeys(q for q, _ in rows):
        ps = [p for q, p in rows if q == r]
        want += [f"Int8KvQuant_v1{{S={ps[-1] + 1},D={D}}}"] * G
        want += [f"Int8LutAttentionRow_v1{{D={D},S={p + 1},SCALE={SCALE}}}" for p in ps for _ in range(H)]
    assert [spec_label(c.spec) for c in _computes(group)] == want


@pytest.mark.parametrize("step, computes, words", [(0, 2 * G + sum(PROMPTS) * H, 53248), (1, 2 * G + 2 * H, 32768)])
def test_every_folded_compute_is_word_equal_to_the_capture(step, computes, words):
    group, _ = _fold(step)
    mem = _cut()[3][step]
    cs = _computes(group)
    assert (len(cs), sum(mem.words(o.view).size for c in cs for o in c.outputs)) == (computes, words)
    bad = [r for c in cs for r in refused(c, mem)]
    assert not bad, bad[:5]


def test_a_flipped_word_of_a_k_window_is_refused_at_its_heads_quantization():
    group, _ = _fold(1)
    mem = _cut()[3][1]
    cs = _computes(group)
    kq = [c for c in cs if c.spec.id.startswith("Int8KvQuant")]
    target = kq[G + 3]                                  # the second request's KV head 3; its third key's first word, the exponent's top bit
    bad = mem.flipped(C.rows_of(target.operands[0])[2], bit=14)
    assert [c for c in cs if refused(c, bad)] == [target]


def test_a_flipped_output_word_is_refused_at_its_row():
    group, _ = _fold(0)
    mem = _cut()[3][0]
    rows = [c for c in _computes(group) if c.spec.id.startswith("Int8LutAttentionRow")]
    target = rows[-1]
    bad = mem.flipped(target.outputs[0].view)
    assert [c for c in rows if refused(c, bad)] == [target]


def test_a_doubled_k_scale_is_refused_at_its_kv_heads_quantizations():
    header, _, _, mems = _cut()
    layer = f"model.layers.{LAYER}.self_attn.attn"
    ks = list(header.profile["backends"]["int8_k_scales"][layer])
    ks[5] = int(ks[5]) + (1 << 23)
    off = dataclasses.replace(header, profile={**header.profile, "backends": {**header.profile["backends"],
                                                                                "int8_k_scales": {**header.profile["backends"]["int8_k_scales"], layer: ks}}})
    group, _ = _fold(1, off)
    cs = _computes(group)
    kq = [c for c in cs if c.spec.id.startswith("Int8KvQuant")]
    hit = {id(c) for c in cs if refused(c, mems[1])}
    assert {id(c) for c in kq if id(c) in hit} == {id(kq[5]), id(kq[G + 5])}
    assert all(c.operands[2] == Const(32, ks[5], "k_scale") for c in cs if id(c) in hit and c.spec.id.startswith("Int8LutAttentionRow"))
