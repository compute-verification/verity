"""`compare.py` on a synthetic window of two requests: schedules and tokens, a replay bit-exact or one bit off, a request
alone against the window over the positions whose inputs agree, and each op's batch variance over the rows whose inputs
(for attention, the keys and values of every earlier position too) are equal."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compare as c  # noqa: E402


def gqa_labels(layers: int) -> list[str]:
    """A dense model's module outputs in the forward's order, as the recorder labels them: the rotary embedding every layer
    shares under layer 0's name, its k-th call suffixed #k."""
    labels = ["model.embed_tokens"]
    for n in range(layers):
        k = f"#{n}" if n else ""
        labels += ["model.layers.0.input_layernorm"] if n == 0 else [f"model.layers.{n}.input_layernorm:0",
                                                                    f"model.layers.{n}.input_layernorm:1"]
        labels += [f"model.layers.{n}.self_attn.qkv_proj", f"model.layers.0.self_attn.rotary_emb:0{k}",
                   f"model.layers.0.self_attn.rotary_emb:1{k}", f"model.layers.{n}.self_attn.attn",
                   f"model.layers.{n}.self_attn.o_proj", f"model.layers.{n}.post_attention_layernorm:0",
                   f"model.layers.{n}.post_attention_layernorm:1", f"model.layers.{n}.mlp.gate_up_proj",
                   f"model.layers.{n}.mlp.act_fn", f"model.layers.{n}.mlp.down_proj", f"model.layers.{n}.mlp"]
    return labels + ["model.norm:0"]


LABELS = gqa_labels(2)
PROMPTS = {0: [11, 12, 13], 1: [21, 22]}
WORKLOAD = {"requests": [{"id": "r0", "prompt": PROMPTS[0], "max_tokens": 3, "temperature": 0.0},
                         {"id": "r1", "prompt": PROMPTS[1], "max_tokens": 2, "temperature": 0.8}]}
#: each step's rows: (request, first position, length)
BATCHED = [[(0, 0, 3), (1, 0, 2)], [(0, 3, 1), (1, 2, 1)], [(0, 4, 1)]]
ALONE = [[(0, 0, 3)], [(0, 3, 1)], [(0, 4, 1)], [(1, 0, 2)], [(1, 2, 1)]]
TOKENS = {"r0": [7, 8, 9], "r1": [4, 5]}


def value(req: int, pos: int) -> np.ndarray:
    return np.random.default_rng([req, pos]).integers(-(1 << 62), 1 << 62, len(LABELS), dtype=np.int64)


def phase(d: Path, schedule: list, tokens: dict, capture: bool = True, row=value, logit=lambda req, pos: req * 100 + pos) -> c.Run:
    """Write one phase's directory as `replay_det.py` does, its rows' values from `row` and its logits rows' from `logit`."""
    d.mkdir(parents=True)
    steps, keys, rows, lkeys, ldig = [], [], [], [], []
    for i, step in enumerate(schedule):
        steps.append({"step": i, "T": sum(n for _, _, n in step), "rows": [[f"r{q}", s, n] for q, s, n in step], "pages": {},
                      "new": [], "resumed": [], "preempted": [], "finished": []})
        for q, s, n in step:
            seq = PROMPTS[q] + tokens[f"r{q}"]
            for p in range(s, s + n):
                keys.append([i, q, p, seq[p]])
                rows.append(row(q, p))
            lkeys.append([i, q, s + n - 1])
            ldig.append(logit(q, s + n - 1))
    (d / "steps.jsonl").write_text("".join(json.dumps(s) + "\n" for s in steps))
    (d / "tokens.json").write_text(json.dumps(tokens))
    (d / "timing.json").write_text(json.dumps({"load_s": 1.0, "serve_s": 2.0, "total_s": 3.0, "steps": len(steps)}))
    if capture:
        np.save(d / "rows.npy", np.array(rows, dtype=np.int64))
        np.save(d / "keys.npy", np.array(keys, dtype=np.int64))
        (d / "labels.json").write_text(json.dumps(LABELS))
        np.save(d / "logit_keys.npy", np.array(lkeys, dtype=np.int64))
        np.save(d / "logit_digest.npy", np.array(ldig, dtype=np.int64))
        np.save(d / "logit_top.npy", np.zeros((len(lkeys), 8), dtype=np.float32))
        np.save(d / "logit_topi.npy", np.zeros((len(lkeys), 8), dtype=np.int64))
        (d / "whole.jsonl").write_text("{}\n" * len(steps))
    return c.Run(d)


def flipped(at: tuple[int, int], label: str, through: bool = False):
    """`value`, with one bit of `label`'s output flipped at (request, position), and with `through` every later output's."""
    j = LABELS.index(label)

    def row(req: int, pos: int) -> np.ndarray:
        v = value(req, pos)
        if (req, pos) == at:
            v[j:None if through else j + 1] ^= 1
        return v
    return row


def test_labels() -> None:
    assert c.canonical("model.layers.0.self_attn.rotary_emb:1#3") == "model.layers.3.self_attn.rotary_emb:1"
    assert c.canonical("model.layers.0.self_attn.rotary_emb:1") == "model.layers.0.self_attn.rotary_emb:1"
    assert c.op_of("model.layers.0.self_attn.rotary_emb:0#5") == "model.layers.N.self_attn.rotary_emb:0"
    assert c.layer_of("model.layers.0.self_attn.rotary_emb:0#5") == 5 and c.layer_of("model.norm:0") is None
    assert c.family(LABELS) == "gqa" and c.family(["model.layers.0.self_attn.mla_attn.mla_attn"]) == "mla"


def test_schedule_and_tokens(tmp_path: Path) -> None:
    a = phase(tmp_path / "serve", BATCHED, TOKENS)
    assert c.schedule(a, phase(tmp_path / "replay", BATCHED, TOKENS))["equal"]
    other = phase(tmp_path / "other", [BATCHED[0], [(0, 3, 1)], [(0, 4, 1), (1, 2, 1)]], TOKENS)
    assert c.schedule(a, other) == {"steps": [3, 3], "equal": False, "first_difference": {"step": 1, "fields": ["T", "rows"]}}
    assert not c.schedule(a, phase(tmp_path / "short", BATCHED[:2], TOKENS))["equal"]
    t = c.tokens(a, phase(tmp_path / "solo", ALONE, {"r0": [7, 5, 6], "r1": [4]}))
    assert t["equal"] == 0 and t["first_divergence"] == {"r0": 1, "r1": 1}
    assert c.first_divergence([1, 2], [1, 2]) is None and c.first_divergence([1, 2], [1, 3]) == 1


def test_values_bit_exact_and_one_bit_off(tmp_path: Path) -> None:
    a = phase(tmp_path / "serve", BATCHED, TOKENS)
    v = c.values(a, phase(tmp_path / "replay1", BATCHED, TOKENS))
    assert v["bit_exact"] and v["rows_differing"] == 0 and v["logits"]["rows_differing"] == 0
    v = c.values(a, phase(tmp_path / "replay2", BATCHED, TOKENS, row=flipped((1, 2), "model.layers.0.mlp.down_proj")))
    assert not v["bit_exact"] and (v["rows_differing"], v["cells_differing"], v["steps_differing"]) == (1, 1, 1)
    assert v["first_difference"] == {"step": 1, "request": 1, "position": 2, "label": "model.layers.0.mlp.down_proj"}
    v = c.values(a, phase(tmp_path / "replay3", BATCHED, TOKENS, logit=lambda req, pos: 1))
    assert not v["bit_exact"] and v["rows_differing"] == 0 and v["logits"]["rows_differing"] == 5


def test_prefix_all() -> None:
    req, pos = np.array([1, 0, 1, 0, 1]), np.array([2, 1, 0, 0, 1])
    ok = np.array([True, True, True, True, False])
    assert c._prefix_all(ok, req, pos).tolist() == [False, True, True, True, False]


def table(eq: np.ndarray) -> dict:
    req, pos = np.array([0, 0, 1, 1, 2, 2]), np.array([0, 1, 0, 1, 0, 1])
    return c.op_variance(eq, np.ones_like(eq), LABELS, req, pos, pos == 1)


def test_op_variance_counts_an_op_whose_inputs_agree() -> None:
    eq = np.ones((6, len(LABELS)), bool)
    eq[4:, LABELS.index("model.layers.1.self_attn.attn"):] = False
    t = table(eq)
    assert t["model.layers.N.self_attn.attn"] == {"inputs_equal_prefill": 6, "differs_prefill": 1,
                                                  "inputs_equal_decode": 6, "differs_decode": 1}
    assert t["model.layers.N.self_attn.o_proj"]["inputs_equal_prefill"] == 5
    assert t["model.layers.N.self_attn.rotary_emb:1"]["inputs_equal_prefill"] == 6


def test_op_variance_attention_reads_every_earlier_position() -> None:
    eq = np.ones((6, len(LABELS)), bool)
    eq[2, LABELS.index("model.layers.0.self_attn.qkv_proj")] = False
    t = table(eq)
    assert t["model.layers.N.self_attn.attn"] == {"inputs_equal_prefill": 5, "differs_prefill": 0,
                                                  "inputs_equal_decode": 5, "differs_decode": 0}
    assert t["model.layers.N.self_attn.rotary_emb:0"] == {"inputs_equal_prefill": 5, "differs_prefill": 0,
                                                         "inputs_equal_decode": 6, "differs_decode": 0}


def test_alone_against_the_window(tmp_path: Path) -> None:
    batch = phase(tmp_path / "serve", BATCHED, TOKENS)
    solo = phase(tmp_path / "alone", ALONE, {"r0": [7, 5, 6], "r1": [4, 5]},
                 row=flipped((1, 1), "model.layers.1.self_attn.attn", through=True),
                 logit=lambda req, pos: -1 if (req, pos) == (1, 1) else req * 100 + pos)
    a = c.alone(batch, solo, WORKLOAD)
    assert (a["rows_compared"], a["inputs_equal"]) == (7, True)
    assert a["bit_identical"] == {"all": 6, "prefill": 4, "decode": 2} and a["rows"] == {"prefill": 5, "decode": 2}
    assert a["first_op_counts"] == {"model.layers.N.self_attn.attn": 1} and a["first_layer_counts"] == {"1": 1}
    assert a["final_hidden_differing"] == 1 and "router" not in a
    assert a["prefill_by_chunk"]["same_chunk"]["rows"] == 5 and a["prefill_by_chunk"]["other_chunk"]["rows"] == 0
    assert a["op_variance"]["model.layers.N.self_attn.attn"] == {"inputs_equal_prefill": 10, "differs_prefill": 1,
                                                                 "inputs_equal_decode": 4, "differs_decode": 0}
    g = a["logits"]
    assert (g["rows_compared"], g["bit_identical"], g["final_hidden_equal"], g["differs_where_final_hidden_equal"]) == (4, 3, 3, 0)
    assert a["tokens"] == {"requests": 2, "identical": 1, "identical_greedy": 0, "greedy": 1, "first_divergence": {"r0": 1}}


def test_main_writes_the_report(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    phase(runs / "serve", BATCHED, TOKENS)
    phase(runs / "replay1", BATCHED, TOKENS)
    phase(runs / "replay2", BATCHED, TOKENS, row=flipped((0, 3), "model.layers.1.mlp"))
    phase(runs / "replay_plain", BATCHED, TOKENS, capture=False)
    phase(runs / "alone", ALONE, TOKENS)
    (tmp_path / "workload.json").write_text(json.dumps(WORKLOAD))
    out = tmp_path / "out"
    out.mkdir()
    (out / "lease.json").write_text(json.dumps({"lease_wall_s": 12.5, "gpus": 1}))
    c.main(["--runs", str(runs), "--workload", str(tmp_path / "workload.json"), "--out", str(out)])
    res = json.loads((out / "compare.json").read_text())
    p = res["pairs"]
    assert p["serve vs replay1"]["values"]["bit_exact"] and not p["serve vs replay2"]["values"]["bit_exact"]
    assert "values" not in p["serve vs replay_plain"] and p["serve vs replay_plain"]["tokens"]["all_equal"]
    assert set(p) == {"serve vs replay1", "serve vs replay2", "serve vs replay_plain", "replay1 vs replay2",
                      "replay1 vs replay_plain", "replay2 vs replay_plain"}
    assert res["alone"]["bit_identical"]["all"] == res["alone"]["rows_compared"] == 8
    assert res["gpu_seconds"]["lease_wall_s"] == 12.5 and res["gpu_seconds"]["total_s"] == 15.0
    md = (out / "compare.md").read_text()
    assert "serve vs replay1: schedule equal; tokens 2/2 requests equal; values bit-exact" in md
    assert "| `self_attn.attn` |" in md
