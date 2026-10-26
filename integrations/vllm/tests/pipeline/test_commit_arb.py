"""`pipeline.commit_arb`'s traffic, verdict and Commit, without an engine: CERT-VARLEN's lengths and caps, the sampling mix,
seeded and in arrival order; a run passes only when every request and every window did; it commits under serve-openai's
rows-sha512 committer."""
from __future__ import annotations

import pytest

from verity_vllm.config import UsageError
from verity_vllm.pipeline import cli
from verity_vllm.pipeline import commit_arb as CA
from verity_vllm.pipeline import serve_seal as SS


def encode(text):
    return [0] + [ord(c) for c in text]


def test_traffic_is_seeded_varied_and_in_arrival_order():
    reqs = CA.traffic(encode, 400, seed=3, prompt_tokens=(16, 512), max_tokens=32, rate=0.5)
    assert reqs == CA.traffic(encode, 400, seed=3, prompt_tokens=(16, 512), max_tokens=32, rate=0.5)
    assert reqs != CA.traffic(encode, 400, seed=4, prompt_tokens=(16, 512), max_tokens=32, rate=0.5)
    steps = [q["arrive_step"] for q in reqs]
    assert steps == sorted(steps) and len({q["request_id"] for q in reqs}) == 400
    lengths = [len(q["prompt"]) for q in reqs]
    assert min(lengths) >= 16 and max(lengths) <= 512 and max(lengths) > 4 * min(lengths)
    tail = encode(CA.CHAT_TAIL)[1:]
    assert all(q["prompt"][-len(tail):] == tail for q in reqs if len(q["prompt"]) >= len(tail))
    s = [q["sampling"] for q in reqs]
    assert all(1 <= x["max_tokens"] <= 32 for x in s)
    temps = {x["temperature"] for x in s}
    assert temps == {0.0, 0.7, 1.0}
    assert any("seed" in x for x in s) and any(x["temperature"] and "seed" not in x for x in s)
    assert not any("seed" in x for x in s if x["temperature"] == 0.0)
    assert any("stop" in x for x in s) and any("min_tokens" in x for x in s)
    assert all(x.get("min_tokens", 0) <= x["max_tokens"] for x in s)


def test_a_template_frame_wraps_the_same_turns():
    class Tok:
        def apply_chat_template(self, turns, *, tokenize, add_generation_prompt, enable_thinking):
            assert not tokenize and add_generation_prompt and not enable_thinking
            return f"<u>{turns[0]['content']}</u><a>"

    frame = CA.template_frame(Tok())
    assert frame == ("<u>", "</u><a>")
    reqs = CA.traffic(encode, 20, seed=1, prompt_tokens=(64, 128), max_tokens=8, rate=1.0, frame=frame)
    assert all(q["prompt"][1:4] == encode("<u>")[1:] and q["prompt"][-7:] == encode("</u><a>")[1:] for q in reqs)


def test_shared_prefix_traffic_opens_with_system_prompts_and_goes_on_in_turns():
    kw = dict(seed=3, prompt_tokens=(16, 128), max_tokens=8, rate=0.5)
    first = CA.traffic(encode, 40, **kw)
    reqs = CA.traffic(encode, 40, **kw, system_prompts=3, system_tokens=200, turns=2)
    assert CA.traffic(encode, 40, **kw, system_prompts=0, turns=0) == first
    systems = [CA.system_prompt(encode, i, 200, True) for i in range(3)]
    assert all(200 <= len(s) < 400 for s in systems) and len({tuple(s) for s in systems}) == 3
    heads = [q for q in reqs if "follows" not in q]
    assert [q["request_id"] for q in heads] == [q["request_id"] for q in first]
    assert all(q["prompt"][0] == 0 and any(q["prompt"][1:1 + len(s)] == s for s in systems) for q in heads)
    assert {sum(q["prompt"][1:1 + len(s)] == s for q in heads) > 1 for s in systems} == {True}   # each shared
    later = [q for q in reqs if "follows" in q]
    assert later and all("prompt" not in q for q in later) and {q["request_id"].rsplit(".", 1)[1] for q in later} == {"t1", "t2"}
    ids = {q["request_id"] for q in reqs}
    assert all(q["follows"] in ids and q["suffix"][:len(encode("<|eot_id|>")) - 1] == encode("<|eot_id|>")[1:] for q in later)


def test_a_turn_is_its_last_turns_prompt_and_stream_then_its_suffix():
    parent = {"request_id": "a", "prompt": [0, 1, 2]}
    q = {"request_id": "a.t1", "follows": "a", "gap": 2, "suffix": [7, 8], "sampling": {"max_tokens": 4}}
    assert CA.next_turn(q, parent, [5, 6], 10, max_model_len=64)
    assert q["prompt"] == [0, 1, 2, 5, 6, 7, 8] and q["arrive_step"] == 13
    long = {**q, "sampling": {"max_tokens": 60}}
    del long["prompt"]
    assert not CA.next_turn(long, parent, [5, 6], 10, max_model_len=64) and "pass --max-model-len 64" in long["skipped"]


def three_turn_checks(flip=None):
    from tests.commit.test_served_record import A_OUT, A_PROMPT, B_OUT, B_PROMPT, C_OUT, C_PROMPT, SEED, three_turns
    rec = three_turns(flip)
    served = {"a": (A_PROMPT, A_OUT), "b": (B_PROMPT, B_OUT), "c": (C_PROMPT, C_OUT)}
    checks = [({"request_id": r, "prompt": p, "sampling": {"seed": SEED}}, None, rec.check(r, p, o, seed=SEED))
              for r, (p, o) in served.items()]
    return rec, checks, {r: o for r, (_p, o) in served.items()}


def test_the_negative_control_flips_a_reused_prefix_and_both_checks_fail():
    rec, checks, streamed = three_turn_checks()
    assert all(c["passed"] for _q, _f, c in checks)
    neg = CA.prefix_negative(rec, checks, streamed, lambda q: q["sampling"]["seed"])
    assert neg["reader"] == "b" and neg["source"] == "a" and neg["position"] == 2 and neg["caught"], neg
    assert rec.requests["a"].inputs[2] == 3, "the flip is on a copy of the record"
    plain = [(q, f, {**c, "prefix_spans": []}) for q, f, c in checks]
    assert CA.prefix_negative(rec, plain, streamed, lambda q: 0) is None


def test_with_the_cache_on_a_run_passes_only_with_a_hit_and_its_negative_caught():
    win = {"window": 0, "passed": True, "commit": {"steps": 4}}
    _rec, checks, _s = three_turn_checks()
    neg = {"caught": True}
    got = CA.summarize(checks, [win], engine_steps=5, idle_steps=0, prefix_caching=True, negative=neg)
    assert got["passed"] and got["cache_hits"] == 2 and got["cached_positions"] == 14 and got["multi_source_hits"] == 1, got
    assert not CA.summarize(checks, [win], engine_steps=5, idle_steps=0, prefix_caching=True, negative={"caught": False})["passed"]
    cold = [(q, f, {**c, "cache_hit": 0, "prefix_spans": []}) for q, f, c in checks]
    assert not CA.summarize(cold, [win], engine_steps=5, idle_steps=0, prefix_caching=True)["passed"]
    assert CA.summarize(cold, [win], engine_steps=5, idle_steps=0)["passed"]


@pytest.mark.parametrize("text", ["16", "a:b", "0:4", "9:3"])
def test_prompt_range_refuses(text):
    with pytest.raises(UsageError):
        CA.prompt_range(text)


def test_summary_passes_only_when_every_request_and_window_did():
    q = {"request_id": "r", "sampling": {"temperature": 0.0}, "prompt": [1, 2]}
    ok = {"passed": True, "streamed_ids": 3, "discarded_samples": 1, "preemptions": 1}
    win = {"window": 0, "passed": True, "commit": {"steps": 4}}
    fin = {"reason": "stop", "stop_reason": "."}
    got = CA.summarize([(q, fin, ok)], [win], engine_steps=5, idle_steps=1)
    assert got["passed"] and got["finishes"] == {"stop-string": 1} and got["chunked_requests"] == 1 and got["committed_steps"] == 4
    assert not CA.summarize([(q, fin, {**ok, "passed": False})], [win], engine_steps=5, idle_steps=1)["passed"]
    assert not CA.summarize([(q, fin, ok)], [{**win, "passed": False}], engine_steps=5, idle_steps=1)["passed"]
    assert not CA.summarize([], [win], engine_steps=0, idle_steps=0)["passed"]


def test_it_commits_under_serve_openais_rows_sha512_committer_and_needs_the_rows_manifest(tmp_path):
    """commit-arb's Commit is serve-openai's rows-sha512 one (`serve_seal.committer`), never SHA-256: without the bench row's
    manifest, or ninja for its kernels, it is refused before the model loads, and the committer refuses it too."""
    def parse(*more: str) -> CA.Options:
        return cli.parse("commit-arb", ["--model", "m", "--revision", "r", "--out", str(tmp_path), *more], environ={})

    with pytest.raises(UsageError, match="under rows-sha512 .*--required-manifest ROW_DIR/manifest.json"):
        CA.preflight(parse(), which=lambda _: "/v/bin/ninja")
    with pytest.raises(UsageError, match="rows-sha512 .*commits with the Commit of record's parts"):
        SS.committer(CA.committer_opts(parse()), llm=None, label="commit-arb")
    with pytest.raises(UsageError, match="no such file"):
        CA.preflight(parse("--required-manifest", str(tmp_path / "none.json")), which=lambda _: "/v/bin/ninja")
    (man := tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(UsageError, match="ninja"):
        CA.preflight(parse("--required-manifest", str(man)), which=lambda _: None)
    CA.preflight(parse("--required-manifest", str(man)), which=lambda _: "/v/bin/ninja")
    opts = CA.committer_opts(parse("--required-manifest", str(man), "--hidden-so", "h.so", "--norm-tap-so", "n.so"))
    assert (opts.scheme, opts.of_record, opts.commit, opts.seal, opts.gpu_tree) == (SS.ROWS, None, True, False, True)
    assert SS.scheme_refusal(opts) is None
    args = SS.committer_args(opts)
    assert (args.commit_scheme, args.required_manifest, args.hidden_so, args.tap_sources.norm_tap_so) == (SS.ROWS, str(man), "h.so", "n.so")
