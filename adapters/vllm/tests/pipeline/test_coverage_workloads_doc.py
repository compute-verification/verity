"""The coverage campaign's generator (`verity_vllm.pipeline.coverage_workloads`): a cell's document, made here with a tokenizer of
one id a character.  It states the cell's lengths, each prompt as the corpus from its own paragraph on, the schedule if every
request runs to its end and the engine arguments that keep a prefill in one step.  A replicate keeps the lengths and changes the
prompts; the sampled column, the format column and the chunked column each add their own keys to the same requests.  The command
prints a dry run's lengths and reads no tokenizer for it, and writes each cell under its variant's name, for the model named or
else the role's."""
from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

import pytest

from tests.pipeline import tokenizers_stand_in
from verity_vllm.config import VLLM
from verity_vllm.engine.profiles.coverage_roles import ROLE_REPOS
from verity_vllm.pipeline import coverage_workloads as gw

GREEDY, T1 = {"temperature": 0.0, "top_p": 1.0, "top_k": 0}, {"temperature": 1.0, "top_p": 1.0, "top_k": 0}
REVISION = "ab" * 20
TOKENIZER = b'{"model": "chars"}'


class _CharTok:
    """What the generator's `Tok` presents of a tokenizer file, an id a character."""

    path, sha256 = Path("snap/tokenizer.json"), "5a" * 32

    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(map(chr, ids))


def _doc(profile="B8-mixed", ctx="C256", **given):
    return gw.workload_doc(**{"case": "B0", "model": "org/m", "tok": _CharTok(), "profile": profile, "ctx": ctx, "seed": 3,
                              "tokenizer_revision": REVISION, "sampling_seed": 7, **given})


def _lengths(doc):
    return [{k: r[k] for k in ("prompt_len", "max_tokens", "arrive_step")} for r in doc["requests"]]


def _ids(tag):
    return [str(uuid.uuid5(uuid.NAMESPACE_URL, f"verity-coverage/B0/B8-mixed/C256/{tag}/{i}")) for i in range(8)]


#: The cell's eight requests, the one of seed 3: each prompt's length, the tokens it is to make and the step it arrives at.
LENGTHS = [(98, 9, 0), (128, 12, 1), (108, 10, 1), (244, 24, 1), (108, 10, 9), (233, 23, 13), (138, 13, 13), (164, 16, 14)]


def test_a_cell_s_document_states_its_lengths_its_schedule_and_the_arguments_that_keep_a_prefill_in_one_step():
    doc = _doc()
    lengths = _lengths(doc)
    sim = gw.simulate_schedule(lengths)
    assert [tuple(d.values()) for d in lengths] == LENGTHS and [r["index"] for r in doc["requests"]] == list(range(8))
    assert [r["request_id"] for r in doc["requests"]] == _ids("3")
    told = ("requests", "note", "tokenizer_note", "length_rule", "schedule_if_full_length", "prompt_lengths", "max_tokens")
    assert {k: v for k, v in doc.items() if k not in told} == {
        "schema": gw.SCHEMA, "model": "org/m", "case": "B0", "engine": {"revision": REVISION, "vllm": VLLM, "args": {"max_num_batched_tokens": 2048 + 64}},
        "seed": 0, "variants": {"greedy": GREEDY}, "seeds": [], "sampling": {"variant": "greedy", "seed": None, "n": 1},
        "serving_profile": "B8-mixed", "context_class": "C256", "length_seed": 3, "prompt_seed": 3, "replicate": 0,
        "expected_engine_steps": sim["expected_engine_steps"]}
    assert doc["schedule_if_full_length"] == sim
    assert (doc["prompt_lengths"], doc["max_tokens"]) == (sorted({n for n, _, _ in LENGTHS}), sorted({n for _, n, _ in LENGTHS}))
    rule = doc["length_rule"]
    assert list(rule) == ["prompt_len", "max_tokens", "arrive_step", "depends_on"]
    assert (rule["prompt_len"], rule["max_tokens"], rule["depends_on"]) == ("U{32..256}", "max(2, prompt_len // 10)", "(profile, ctx, seed) only")
    assert doc["tokenizer_note"] == ("prompt_token_ids from tokenizer.json sha256 5a5a5a5a5a5a5a5a (tokenizers, add_special_tokens=False), "
                                     "natural text from corpus_coverage.json, truncated to prompt_len; prompt_text is the decode of the ids (informational)")
    longer = _doc(ctx="C1024")
    assert longer["engine"]["args"] == {"max_num_batched_tokens": 8192 + 64} and longer["context_class"] == "C1024"
    assert (longer["length_rule"]["prompt_len"], longer["length_rule"]["max_tokens"]) == ("U{128..1024}", "max(4, prompt_len // 10)")
    one = _doc(profile="B1")                                                    # one request: no budget to keep
    (r,) = one["requests"]
    assert one["engine"]["args"] == {} and one["serving_profile"] == "B1"
    assert one["note"].partition(": ")[2] == (
        f"B1 / C256, seed 3: 1 request(s), prompt lens [{r['prompt_len']}], max_tokens [{r['max_tokens']}], arrive steps [0]; "
        f"every request to max_tokens => {r['max_tokens']} engine steps, peak 1 concurrent, 0 batch-membership changes, "
        "0 steps with a prefill next to decodes.  Greedy.  "
        "The control run's realized step count is the gates' reference (EOS may end a request early).")


def test_each_prompt_is_the_corpus_from_its_own_paragraph_on_cut_at_its_length():
    paras = gw.corpus_paragraphs()
    at = {f"{p['source']} \N{PILCROW SIGN}{p['index']}": j for j, p in enumerate(paras)}
    reqs = _doc()["requests"]
    for r in reqs:
        used = [at[name] for name in r["origin"]["paragraphs"]]
        assert used == [(used[0] + k) % len(paras) for k in range(len(used))]
        assert r["prompt_text"] == " ".join(paras[j]["text"] for j in used)[:r["prompt_len"]]
        assert len(" ".join(paras[j]["text"] for j in used[:-1])) < r["prompt_len"]            # no paragraph the length does not reach
        assert r["prompt_token_ids"] == [ord(c) for c in r["prompt_text"]] and r["origin"]["corpus"] == "corpus_coverage.json"
    assert len({r["origin"]["paragraphs"][0] for r in reqs}) == 8                           # no two prompts start at one paragraph


def test_a_replicate_keeps_the_lengths_and_the_schedule_and_takes_its_prompts_from_its_own_seed():
    base, again = _doc(), _doc(replicate=2)
    assert _lengths(again) == _lengths(base) and again["schedule_if_full_length"] == base["schedule_if_full_length"]
    assert (again["replicate"], again["length_seed"], again["prompt_seed"]) == (2, 3, 2003)
    assert [r["request_id"] for r in again["requests"]] == _ids("3/p2003")
    assert [r["origin"]["paragraphs"][0] for r in again["requests"]] != [r["origin"]["paragraphs"][0] for r in base["requests"]]
    assert ", replicate 2 (prompt seed 2003): 8 request(s)" in again["note"] and "seed 3: 8 request(s)" in base["note"]


def test_the_sampled_column_seeds_its_requests_and_the_format_column_asks_the_engine_for_its_dtype():
    base, sampled = _doc(), _doc(variant="t1")
    assert sampled["variants"] == {"greedy": GREEDY, "t1": T1} and sampled["sampling"] == {"variant": "t1", "seed": 7, "n": 1}
    assert sampled["requests"] == base["requests"]
    assert f"  Sampling variant t1 {T1} seed 7.  " in sampled["note"] and "  Greedy.  " in base["note"]
    for dtype in ("float16", "half"):
        half = _doc(dtype=dtype)
        assert half["engine"]["args"] == {"max_num_batched_tokens": 2048 + 64, "dtype": dtype}
        assert (half["dtype"], half["numerical_format"]) == (dtype, "f16") and half["requests"] == base["requests"]
        assert f", dtype {dtype} (engine.args.dtype; the same requests as the bf16 workload): " in half["note"]
    assert not {"dtype", "numerical_format", "feature_variant"} & set(base) and "dtype" not in base["note"]
    for given, refusal in (({"variant": "t2"}, "unknown sampling variant 't2'; known: ['greedy', 't1']"),
                           ({"dtype": "float8"}, "unknown dtype variant 'float8'; known: ['float16', 'half']")):
        with pytest.raises(KeyError) as refused:
            _doc(**given)
        assert refused.value.args == (refusal,)


def test_the_chunked_column_declares_its_budget_and_the_schedule_under_it():
    base, doc = _doc(), _doc(chunked=64)
    sim = gw.simulate_schedule_chunked(doc["requests"], 64, 8)
    assert doc["engine"]["args"] == {"max_num_batched_tokens": 64, "enable_chunked_prefill": True} and doc["feature_variant"] == {"chunked": 64}
    assert doc["requests"] == base["requests"]
    assert doc["schedule_if_full_length"] == sim and doc["expected_engine_steps"] == sim["expected_engine_steps"] > base["expected_engine_steps"]
    assert (f"  CHUNKED PREFILL: max_num_batched_tokens 64 (enable_chunked_prefill) forces prefill chunks {sim['prefill_chunks']} "
            f"(partial-prefill steps {sim['partial_prefill_steps']}: vLLM samples and discards a token at each), "
            f"first served token at steps {sim['first_token_step']}.  The control run's") in doc["note"]
    assert "CHUNKED" not in base["note"]
    assert _doc(chunked=64, dtype="float16")["engine"]["args"] == {"max_num_batched_tokens": 64, "enable_chunked_prefill": True, "dtype": "float16"}


def test_an_idle_engine_admits_the_next_arrival_at_once_and_a_budget_that_fits_no_token_stalls():
    reqs = [{"prompt_len": 4, "max_tokens": 1, "arrive_step": 0}, {"prompt_len": 4, "max_tokens": 2, "arrive_step": 9}]
    sim = gw.simulate_schedule_chunked(reqs, 8)
    assert (sim["expected_engine_steps"], sim["admitted_step"], sim["first_token_step"], sim["finish_step"]) == (3, [0, 1], [0, 1], [0, 2])
    assert (sim["prefill_chunks"], sim["partial_prefill_steps"], sim["peak_concurrent"], sim["membership_changes"]) == ([[4], [4]], [[], []], 1, 1)
    assert (sim["max_num_batched_tokens"], sim["max_num_seqs"], sim["steps_with_partial_prefill"], sim["steps_with_mixed_prefill_decode"]) == (8, 8, 0, 0)
    for budget, seqs in ((0, 8), (8, 0)):
        with pytest.raises(RuntimeError, match=rf"^chunked schedule stalled at step 0 \(budget {budget}, max_num_seqs {seqs}\)$"):
            gw.simulate_schedule_chunked(reqs, budget, seqs)


@pytest.mark.parametrize("rows, budget, steps, chunks, partial_steps, first, finish, partial, mixed", [
    # 4 a step: the first prompt takes it whole, then its rest goes before the second's first chunk, and a decode costs one
    ([(6, 3), (3, 1), (2, 1)], 4, 4, [[4, 2], [2, 1], [2]], [[0], [1], []], [1, 2, 2], [3, 2, 2], 2, 1),
    # 3 a step: a waiting request, then a running one, is given the one token that is left
    ([(2, 3), (2, 2), (3, 1)], 3, 4, [[2], [1, 1], [1, 1, 1]], [[], [0], [1, 2]], [0, 1, 3], [2, 2, 3], 3, 2),
])
def test_a_tight_budget_goes_to_the_running_requests_first_and_cuts_a_prompt_into_chunks(rows, budget, steps, chunks, partial_steps, first, finish,
                                                                                        partial, mixed):
    sim = gw.simulate_schedule_chunked([{"prompt_len": p, "max_tokens": m, "arrive_step": 0} for p, m in rows], budget)
    assert sim.pop("policy") == ("vLLM V1: running first (1 decode token / remaining prompt), then waiting in arrival order with the remaining "
                                 "budget; first token sampled at the step that completes the prompt; no preemption; every request to max_tokens")
    assert sim == {"expected_engine_steps": steps, "peak_concurrent": 3, "membership_changes": 1, "steps_with_mixed_prefill_decode": mixed,
                   "steps_with_partial_prefill": partial, "max_num_batched_tokens": budget, "max_num_seqs": 8, "prefill_chunks": chunks,
                   "partial_prefill_steps": partial_steps, "first_token_step": first, "finish_step": finish, "admitted_step": [0, 0, 0]}


@pytest.mark.parametrize("spec, parsed, suffix", [
    ("", ("greedy", {}), ""),
    ("greedy,,t1", ("t1", {}), "_t1"),                                          # an empty item is none, and the last name wins
    (" t1 , chunked=64 ", ("t1", {"chunked": 64}), "_t1_ck64"),
    ("dtype=float16", ("greedy", {"dtype": "float16"}), "_f16"),
    ("chunked=64,dtype=half", ("greedy", {"chunked": 64, "dtype": "half"}), "_ck64_f16"),
])
def test_a_variant_is_a_sampling_name_and_features_by_key_and_gives_its_file_s_suffix(spec, parsed, suffix):
    assert gw.parse_variant(spec) == parsed and gw.variant_suffix(*parsed) == suffix


@pytest.mark.parametrize("spec, refusal", [
    ("dtype=float8", "unknown dtype variant 'float8'; known: ['float16', 'half']"),
    ("x=1", "unknown feature variant 'x'; known: ['chunked', 'dtype']"),
    ("t2", "unknown sampling variant 't2'; known: ['greedy', 't1']"),
])
def test_a_variant_the_generator_does_not_know_is_refused_by_name(spec, refusal):
    with pytest.raises(KeyError) as refused:
        gw.parse_variant(spec)
    assert refused.value.args == (refusal,)


def test_the_tokenizer_is_the_file_named_or_tokenizer_json_of_the_directory_named_and_adds_no_special_token(tmp_path, monkeypatch):
    stand_in = tokenizers_stand_in.install(monkeypatch)
    (tmp_path / "tokenizer.json").write_bytes(TOKENIZER)
    for named in (tmp_path, str(tmp_path), tmp_path / "tokenizer.json"):
        tok = gw.Tok(named)
        assert tok.path == tmp_path / "tokenizer.json" and tok.tk.path == str(tmp_path / "tokenizer.json")
        assert tok.sha256 == hashlib.sha256(TOKENIZER).hexdigest()
    assert tok.encode("a\N{PILCROW SIGN}") == [97, 182] and tok.decode([97, 182]) == "a\N{PILCROW SIGN}"
    assert stand_in.asked == {("encode", False), ("decode", False)}


def test_a_dry_run_prints_each_cell_s_lengths_and_schedule_and_reads_no_tokenizer(tmp_path, monkeypatch, capsys):
    def no_tokenizer(path):
        raise AssertionError("a dry run reads no tokenizer")

    def line(cell, schedule):
        return f"{cell[0]} {cell[1]} {[tuple(d.values()) for d in cells[cell]]} {schedule(cells[cell])}"

    # the lengths each cell's document states, in the order the command goes through the cells
    cells = {(profile, ctx): _lengths(_doc(profile, ctx)) for profile in ("B1", "B8-mixed") for ctx in ("C256", "C1024")}
    assert [tuple(d.values()) for d in cells["B8-mixed", "C256"]] == LENGTHS and [len(c) for c in cells.values()] == [1, 1, 8, 8]
    monkeypatch.setattr(gw, "Tok", no_tokenizer)
    given = {"case": "B0", "model": "m", "tokenizer": "t", "profiles": "B1,B8-mixed", "ctx": "C256,C1024", "seed": 3,
             "out_dir": str(tmp_path / "out"), "dry_run": True}
    assert gw.main(gw.Options(**given)) == 0
    assert capsys.readouterr().out.splitlines() == [line(cell, gw.simulate_schedule) for cell in cells]
    assert gw.main(gw.Options(**given, variant=["t1", "chunked=64"])) == 0
    assert capsys.readouterr().out.splitlines() == [line(cell, lambda lengths: gw.simulate_schedule_chunked(lengths, 64)) for cell in cells]
    assert not (tmp_path / "out").exists()


def test_the_model_written_is_the_one_named_else_the_role_s_in_the_checkpoints_file_else_the_role_s_own_else_unknown(tmp_path, monkeypatch):
    tokenizers_stand_in.install(monkeypatch)
    (tmp_path / "tokenizer.json").write_bytes(TOKENIZER)
    listed = tmp_path / "checkpoints.json"
    listed.write_text(json.dumps({"checkpoints": [{"role": "B0", "repo": "org/listed"}]}))

    def model(case, **given):
        assert gw.main(gw.Options(case=case, tokenizer=str(tmp_path), profiles="B1", ctx="C256", out_dir=str(tmp_path / "out"), **given)) == 0
        return json.loads((tmp_path / "out" / f"workload_cov_{case.lower()}_b1_c256.json").read_text())["model"]

    assert model("B0", model="org/named", checkpoints=str(listed)) == "org/named"
    assert model("B0", checkpoints=str(listed)) == "org/listed"
    assert model("B1", checkpoints=str(listed)) == ROLE_REPOS["B1"] != "org/listed"         # a role the file does not list
    assert model("B0", checkpoints=str(tmp_path / "none.json")) == ROLE_REPOS["B0"]
    assert model("B9", checkpoints=str(listed)) == "unknown"


def test_the_command_writes_each_cell_s_document_under_its_variant_s_name_its_text_not_escaped(tmp_path, monkeypatch):
    tokenizers_stand_in.install(monkeypatch)
    (tmp_path / "tokenizer.json").write_bytes(TOKENIZER)
    cells = tmp_path / "out" / "cells"                                          # made, with the directory above it
    given = {"case": "B0", "model": "org/m", "tokenizer": str(tmp_path), "tokenizer_revision": REVISION, "profiles": "B1", "ctx": "C256,C1024",
             "seed": 3, "out_dir": str(cells)}
    plain = gw.Options(**given)
    assert gw.main(plain) == 0
    assert gw.main(gw.Options(**given, replicate=2, variant=["t1,chunked=64", "dtype=float16"], sampling_seed=7)) == 0
    column = {"replicate": 2, "variant": "t1", "chunked": 64, "dtype": "float16", "sampling_seed": 7}
    written = {f"workload_cov_b0_b1_{ctx.lower()}{suffix}.json": (ctx, kw)
               for ctx in ("C256", "C1024") for suffix, kw in (("", {}), ("_r2_t1_ck64_f16", column))}
    assert sorted(p.name for p in cells.iterdir()) == sorted(written)
    tok = gw.Tok(tmp_path)
    for name, (ctx, kw) in written.items():
        doc = gw.workload_doc(**{"case": "B0", "model": "org/m", "tok": tok, "profile": "B1", "ctx": ctx, "seed": 3,
                                 "tokenizer_revision": REVISION, "sampling_seed": plain.sampling_seed, **kw})
        text = (cells / name).read_text()
        assert text == json.dumps(doc, indent=1, ensure_ascii=False) + "\n" and "\N{PILCROW SIGN}" in text, name
        assert doc["tokenizer_note"].startswith(f"prompt_token_ids from tokenizer.json sha256 {hashlib.sha256(TOKENIZER).hexdigest()[:16]} (")
