"""Advice v1's charge by class (`verity_vllm.pipeline.advice_code.by_class`, and `by_class.py`'s report over a trace).
Each window's parts sum to its charge, and its segment's parts are the arithmetic decoder's own shifts, under the
tick and without it, under the uniform book and a fit one. Under the certified tick, with the prefix cache on and
prompts that share prefixes, every field but the lengths is 0 by construction, each by rules whose lines are in the
code; without the tick the admission is charged and the hits still aren't. The observer reads off the replay the
evictions, the lookups they cut short, the waits with what held them, and the preemptions. The report counts each
class's events beside its bits, a class with no bits and no event reading "not exercised"; its hits are the engine's
own; where the decoder refuses the string it gives the decoder's reason and the first step at which the replay
differs from the engine's schedule; and a trace with an abort is its refusal and the aborts its directory records."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from test_tick import EOS, N, R, buckets, engine, plan, serve, tick_case, write_corpus

from verity_vllm.pipeline.advice_code import by_class as BC
from verity_vllm.pipeline.advice_code import pool as P
from verity_vllm.pipeline.advice_code import replay as RP
from verity_vllm.pipeline.advice_code.codec import length_widths

HERE = Path(__file__).resolve().parents[1]


def _load(name: str):
    key = f"enc_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, HERE / f"{name}.py")
        sys.modules[key] = mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return sys.modules[key]


V, CV, BYC = _load("v1"), _load("code_v1"), _load("by_class")

SEGMENT = ("decisions", "order", "hits", "resumes", "lag")
LENGTHS = ("output_len", "finish")


def registrations(asy: bool) -> dict:
    """The plan's tick-served run under each registration v1 reads it with: the tick's, with a fit book and without
    lengths too, and under sync scheduling the same run registered without the tick (under async it has no advice)."""
    _, rec, adv, reg = tick_case(asy)
    fit = dataclasses.replace(reg, book=CV.fit_book([V.residual(adv, rec, reg)], V.TICK_TABLE_CAPS))
    out = {"tick": reg, "tick, fit book": fit, "tick, no lengths": dataclasses.replace(reg, send_lengths=False)}
    if not asy:
        plain = dataclasses.replace(reg, tick=None)
        out["plain"] = plain
        out["plain, fit book"] = dataclasses.replace(plain, book=CV.fit_book([V.residual(adv, rec, plain)], V.CAPS))
    return {name: (rec, adv, r) for name, r in out.items()}


@pytest.mark.parametrize("asy", (False, True), ids=("sync", "async"))
def test_the_parts_sum_to_the_charge_and_are_the_decoders_shifts(asy):
    """Each window's parts are `bits_by_window`'s charge exactly. Its segment's parts sum to the shifts the arithmetic
    decoder counts over the same string, its segment's end is the 2 bits of a segment that coded anything, and its
    length terms are the tail's widths."""
    for name, (rec, adv, reg) in registrations(asy).items():
        data = V.encode(adv, rec, reg)
        charge, parts = V.bits_by_window(data, rec, reg), BC.bits_by_class(data, rec, reg)
        assert {w: sum(p.values()) for w, p in parts.items()} == charge, name
        assert all(tuple(p) == BC.FIELDS and min(p.values()) >= 0 for p in parts.values()), name
        d, st = CV.Decoder(data, sum(charge.values()), V.book_of(reg)), CV.CoderState()
        for w, widths in enumerate(length_widths(rec, reg)):
            d.window()
            CV.code_window(d, None, st)
            assert sum(parts[w][f] for f in SEGMENT) == d.seg.shifts, (name, w)
            assert parts[w]["segment_end"] == (2 if d.seg.used else 0), (name, w)
            for n in widths:
                d.raw(n)
            assert (parts[w]["output_len"], parts[w]["finish"]) == (sum(widths[0::2]), len(widths) // 2), (name, w)
        d.finish()


@pytest.mark.parametrize("asy", (False, True), ids=("sync", "async"))
def test_under_the_tick_only_the_lengths_are_charged(asy):
    """A tick-served run with the prefix cache on, whose prompts share prefixes and whose admissions start from hits:
    every window's charge is its length terms, every other field is 0, and the registration holds each of those at 0
    by rules of the code. Where it sends no lengths, every field is."""
    cases = registrations(asy)
    rec, adv, reg = cases["tick"]
    assert reg.engine.enable_prefix_caching and sum(1 for st in adv.steps for x in st.rows if x.cache_hit) > 1
    for name in ("tick", "tick, fit book"):
        rec, adv, r = cases[name]
        data = V.encode(adv, rec, r)
        charge, parts = V.bits_by_window(data, rec, r), BC.bits_by_class(data, rec, r)
        assert sum(charge.values()) == sum(q.max_tokens.bit_length() + 1 for q in rec.requests) > 0
        for w, p in parts.items():
            assert p["output_len"] + p["finish"] == charge[w] and not any(p[f] for f in (*SEGMENT, "segment_end"))
        shut = BC.closed(r)
        assert tuple(shut) == (*SEGMENT, "segment_end")
        assert shut["hits"] == (("tick book",) if r.book is not None else ("caps",)) + ("one value", "refused")
        assert shut["decisions"] == shut["order"] == (("tick book",) if r.book is not None else ("tick caps",)) + (
            "one value", "tick replay", "tick refused")
        assert shut["segment_end"] == ("no segment",)
    rec, adv, r = cases["tick, no lengths"]
    assert V.encode(adv, rec, r) == b"" and tuple(BC.closed(r)) == BC.FIELDS
    assert BC.closed(r)["finish"] == ("no lengths",)
    assert all(not any(p.values()) for p in BC.bits_by_class(b"", rec, r).values())


def test_without_the_tick_the_admission_is_charged():
    """The same run registered without the tick: its decisions are charged, each window's segment ends in the
    termination's 2 bits, and no window sends a hit, a resume or a lag bit, which v1's own caps hold at 0. A book with
    room for hits charges their count even at none, so there the hits aren't 0 by construction."""
    cases = registrations(False)
    for name in ("plain", "plain, fit book"):
        rec, adv, r = cases[name]
        parts = BC.bits_by_class(V.encode(adv, rec, r), rec, r)
        assert sum(p["decisions"] for p in parts.values()) > 0, name
        assert all(p["segment_end"] == 2 and not (p["hits"] or p["resumes"] or p["lag"]) for p in parts.values()), name
        shut = BC.closed(r)
        assert tuple(shut) == ("hits", "resumes", "lag")
        assert shut["hits"] == (("caps",) if r.book is None else ()) + ("one value", "refused")
    rec, adv, plain = cases["plain"]
    roomy = dataclasses.replace(plain, book=CV.Book.uniform(dataclasses.replace(V.CAPS, hits=64)))
    assert "hits" not in BC.closed(roomy)
    parts = BC.bits_by_class(V.encode(adv, rec, roomy), rec, roomy)
    assert sum(p["hits"] for p in parts.values()) > 0


def test_the_fields_are_the_window_codes_and_each_rule_is_on_one_line(monkeypatch):
    assert set(BC.SYMBOLS) == set(CV.ints(CV.DEFAULT_CAPS)) | set(CV.BITS)
    assert sorted(f for fs in BC.CLASSES.values() for f in fs) == sorted(BC.FIELDS)
    assert set(BC.SYMBOLS.values()) == set(SEGMENT)
    package = Path(BC.__file__).parent
    for name, rule in BC.RULES.items():
        file, line = BC.where(name)
        first = rule.line.split()[0] if rule.prose else rule.line
        assert first in (package / file).read_text().splitlines()[line - 1], name
    assert {r for rules in BC.NO_FIELD.values() for r in rules} <= set(BC.RULES)
    assert set(BYC.OF) == {*BC.CLASSES, *BC.NO_FIELD} == {name for name, _ in BYC.COLUMNS}
    assert {e for es in BYC.OF.values() for e in es} | {e for _, es in BYC.COLUMNS for e, _ in es} <= set(BYC.EVENTS)
    _, _, _, reg = tick_case(False)
    assert BC.no_field(reg)["release buckets"] == ("release certified", "tick replay")
    assert BC.no_field(dataclasses.replace(reg, tick=None)) == BC.NO_FIELD
    monkeypatch.setitem(BC.RULES, "caps", BC.Rule("as before", "codec", "CAPS = CV.Caps(hits=1)"))
    with pytest.raises(V.AdviceCodeError, match="the rule 'caps' is on 0 lines of codec.py"):
        BC.where("caps")
    monkeypatch.setitem(BC.RULES, "caps", BC.Rule("as before", "codec", "CAPS"))
    with pytest.raises(V.AdviceCodeError, match="moved"):
        BC.where("caps")


def test_a_rule_that_cites_a_docstring_is_found_however_its_lines_break(monkeypatch, tmp_path):
    """A rule that cites a docstring's words (`prose`) is found wherever the docstring's lines break, at the line its
    words begin on, and is refused once the words are changed or there twice. A rule that cites a line of code is
    found on one line only, as before: the same words broken over two lines are refused."""
    monkeypatch.setattr(BC, "__file__", str(tmp_path / "by_class.py"))
    monkeypatch.setitem(BC.RULES, "said", BC.Rule("as said", "said", "aborts and stop strings", prose=True))
    monkeypatch.setitem(BC.RULES, "coded", BC.Rule("as coded", "said", "aborts and stop strings"))
    file = tmp_path / "said.py"
    file.write_text("It refuses what it doesn't model\n(structured output, aborts and stop strings) rather\nthan guess.\n")
    assert BC.where("said") == BC.where("coded") == ("said.py", 2)
    file.write_text("It refuses what it doesn't model (a draft,\nstructured output, aborts\nand stop strings) rather.\n")
    assert BC.where("said") == ("said.py", 2)
    with pytest.raises(V.AdviceCodeError, match="the rule 'coded' is on 0 lines of said.py"):
        BC.where("coded")
    file.write_text("It refuses\n\n    what it doesn't model: aborts\n    and  stop\n    strings, rather than guess.\n")
    assert BC.where("said") == ("said.py", 3)
    file.write_text("It refuses aborts or stop strings.\n")
    with pytest.raises(V.AdviceCodeError, match="the rule 'said' is on 0 lines of said.py"):
        BC.where("said")
    file.write_text("It refuses aborts and stop strings.\nAgain: it refuses aborts and\nstop strings.\n")
    with pytest.raises(V.AdviceCodeError, match="the rule 'said' is on 2 lines of said.py"):
        BC.where("said")


def _io(prompt, n: int = 1):
    """A request that runs to its length of `n` tokens."""
    return R.RequestIO(prompt_len=len(prompt), max_tokens=n, output_len=n, eos=False, tokens=tuple(prompt) + (3,) * n)


def _engine(**kw):
    return R.Engine(**{"max_num_seqs": 4, "max_num_batched_tokens": 64, "max_model_len": 256, "num_gpu_blocks": 64,
                       "block_size": 16, "enable_prefix_caching": True, "async_scheduling": False, **kw})


def _seen(e, ios, admit) -> list:
    with BC.observing() as seen:
        rp = RP.replay(e, ios, admit=admit)
    assert len(seen) == len(rp.steps)
    return seen


def test_the_observer_reads_evictions_waits_and_preemptions_off_the_replay():
    """Four replays small enough to follow by hand, 16-token blocks.
    - One request a step and four blocks: while the first runs the second waits on `max_num_seqs`; the second's 49
      tokens take all four blocks, the two that cached the first's 32-token prefix among them; the third, which shares
      that prefix, then finds none of it: its lookup is cut, 2 blocks it would have hit.
    - Two blocks and two requests that each grow into a second: the first preempts the second and takes its block,
      whose hash goes; the second waits on that preemption, then resumes past a lookup cut 1 block short.
    - A step whose 32 tokens the first prompt fills leaves the second waiting on `tokens`; a pool the first prompt
      fills leaves it waiting on `blocks`.
    The observer leaves the scheduler and the pool as it found them, after an error inside it too, and counts nothing
    on a what-if fork."""
    before = (P.BlockPool.hit, P.BlockPool._pop_hash, P.BlockPool._get_new_blocks, P.BlockPool.cache_blocks, RP.Scheduler.schedule)
    shared = tuple(range(100, 132))
    one = _seen(_engine(max_num_seqs=1, num_gpu_blocks=5),
                [_io(shared + (7,)), _io(range(200, 249)), _io(shared + (8,))], [0, 0, 1])
    assert one == [BC.Seen(waiting=(1,), why="max_num_seqs"), BC.Seen(evicted=2, waiting=(2,), why="max_num_seqs"),
                   BC.Seen(evicted=2, cut=[(2, 2)])]
    for asy in (False, True):
        two = _seen(_engine(max_num_seqs=2, num_gpu_blocks=3, async_scheduling=asy),
                    [_io(range(300, 316), 2), _io(range(400, 416), 2)], [0, 0])
        assert two == [BC.Seen(), BC.Seen(evicted=1, waiting=(1,), why="preempted", preempted=(1,)),
                       BC.Seen(evicted=1, cut=[(1, 1)], resumed=(1,))]
    ios = [_io(range(300, 340)), _io(range(400, 410))]
    assert _seen(_engine(max_num_batched_tokens=32), ios, [0, 0]) == [BC.Seen(waiting=(1,), why="tokens"), BC.Seen()]
    assert _seen(_engine(num_gpu_blocks=4), ios, [0, 0]) == [BC.Seen(waiting=(1,), why="blocks"), BC.Seen()]
    assert set(BC.WHY) >= {"max_num_seqs", "tokens", "blocks", "preempted"} >= set(BYC.FULL) | {"blocks"}
    e, (ios, _) = engine(False), plan()
    runs = []
    for ask in (False, True):
        forks = []

        def decide(i, k, o, ask=ask, forks=forks):
            forks.extend(o.whatif(now, 2) for now in ((True, False) if ask else ()))
            return True

        with BC.observing() as seen:
            rp = RP.replay(e, ios, decide=decide)
        assert len(seen) == len(rp.steps) and bool(forks) == ask
        runs.append(seen)
    assert runs[0] == runs[1] and sum(s.evicted for s in runs[0]) > 0 < sum(len(s.preempted) for s in runs[0])
    with pytest.raises(ZeroDivisionError), BC.observing():
        raise ZeroDivisionError
    assert before == (P.BlockPool.hit, P.BlockPool._pop_hash, P.BlockPool._get_new_blocks, P.BlockPool.cache_blocks,
                      RP.Scheduler.schedule)


def test_the_report_splits_a_trace_by_class_and_counts_its_events(tmp_path, capsys):
    """`by_class.py` over the plan's tick-served run and the same run without the tick, as trace-corpus directories:
    the tick's charge is its length terms, the terminator's bit a request among them, every other class 0 by
    construction or without a field, each rule with its file and line; the replay's admissions from a hit are the
    engine's own; every class's events are counted by window, and the plan's small engine makes each happen: hits,
    evictions that cut a later admission's lookup, waits on a full batch, a preemption, releases in several buckets
    and both ends. A class with no bits and no event in a window reads "not exercised" there, and one with events and
    no bits reads 0. Both strings decode back to their traces."""
    e, (ios, times) = engine(False), plan()
    run = serve(e, ios, times, seed=1)
    tick = write_corpus(tmp_path / "tick", e, ios, times, run, tick=True)
    plain = write_corpus(tmp_path / "plain", e, ios, times, run, tick=False)
    out = tmp_path / "report.json"
    assert BYC.main([f"tick={tick}", f"plain={plain}", "--stops", str(EOS), "--T", "10", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "| all |" in text and "- hits: 0 by construction: " in text and "- aborts: no field: " in text
    assert "decode(encode(a)) == a: yes." in text and "NO" not in text
    assert len({line.count("|") for line in text.splitlines() if line.startswith("|")}) == 1
    rows = {r["trace"]: r for r in json.loads(out.read_text())["rows"]}
    t, p = rows["tick"], rows["plain"]
    lengths = sum(io.max_tokens.bit_length() + 1 for io in ios)
    trace = R.read_corpus_trace(tick)
    hits = [k for st in trace.steps for k in st.cache_hit.values()]
    assert len(hits) > 1 and all(hits)
    assert t["round_trip"] and p["round_trip"] and t["first_step_that_differs"] is None
    assert t["bits"] == lengths and t["windows"] > 1 and t["prefix_caching"]
    assert t["total"]["classes"]["terminator"] == N and t["total"]["classes"]["output lengths"] == lengths - N
    assert {c: v["standing"] for c, v in t["classes"].items()} == {
        "hits": "0 by construction", "admission": "0 by construction", "terminator": "charged",
        "output lengths": "charged", "resumes and lag": "0 by construction", "segment end": "0 by construction",
        "aborts": "no field", "release buckets": "no field"}
    assert p["classes"]["admission"]["standing"] == p["classes"]["segment end"]["standing"] == "charged"
    assert p["classes"]["hits"]["standing"] == "0 by construction" and p["total"]["classes"]["admission"] > 0
    assert p["bits"] == sum(p["total"]["classes"].values()) > lengths
    assert [r["rule"] for r in t["classes"]["release buckets"]["rules"]] == ["release certified", "tick replay"]
    for row in (t, p):
        for c in row["classes"].values():
            assert (c["standing"] == "charged") == (not c["rules"])
            for r in c["rules"]:
                assert r["file"].startswith("advice_code/") and r["line"] > 0 and r["why"] == BC.RULES[r["rule"]].why
        total, ev = row["total"], row["total"]["events"]
        assert set(ev) == set(BYC.EVENTS) and all(set(w["events"]) == set(BYC.EVENTS) for w in row["by_window"])
        for key in BYC.EVENTS:
            windows = sum(w["events"][key] for w in row["by_window"])
            assert windows >= ev[key] if key in ("waited", "waited_full", "release_buckets") else windows == ev[key], key
        assert {why: sum(w["steps_waiting"][why] for w in row["by_window"]) for why in BC.WHY} == total["steps_waiting"]
        assert all(sum(w["classes"].values()) == w["charge"] == sum(w["fields"].values()) for w in row["by_window"])
        assert ev["admissions"] == ev["length_terms"] == ev["released"] == N and ev["lookups"] >= N
        assert row["engine_hits"] == {"admissions_hit": len(hits), "hit_tokens": sum(hits),
                                      "hit_blocks": sum(hits) // e.block_size}
        assert (ev["admissions_hit"], ev["hit_blocks"]) == (len(hits), sum(hits) // e.block_size)
        assert ev["admissions_hit"] <= ev["lookups_hit"] <= ev["lookups"]
        assert (ev["ended_stop"], ev["ended_length"]) == (sum(io.eos for io in ios), sum(not io.eos for io in ios))
        assert min(ev["ended_stop"], ev["ended_length"]) > 0 and ev["lagged"] == 0
        assert ev["preemptions"] == ev["resumptions"] == sum(len(st.preempted) for st in trace.steps) > 0
        assert ev["evicted_blocks"] >= ev["admissions_cut"] > 0 and ev["cut_blocks"] >= ev["admissions_cut"]
        assert ev["waited"] >= ev["waited_full"] and ev["segments"] == sum(w["classes"]["segment end"] > 0 for w in row["by_window"])
        assert row["engine"] == {"num_gpu_blocks": e.num_gpu_blocks, "block_size": e.block_size, "max_num_seqs": e.max_num_seqs,
                                 "max_num_batched_tokens": e.max_num_batched_tokens, "async_scheduling": False}
        assert row["aborts"] == {"freed": N, "finishes": {"length": ev["ended_length"], "stop": ev["ended_stop"]}, "aborted": 0,
                                 "released_never_freed": None, "certifier_failed": None}
    ev = t["total"]["events"]
    assert ev["waited_full"] > 0 and t["total"]["steps_waiting"]["tokens"] > 0 == t["total"]["steps_waiting"]["in flight"]
    assert ev["release_buckets"] == len(set(buckets(times))) > 1 and ev["segments"] == 0 < p["total"]["events"]["segments"]
    quiet = next(w for w in t["by_window"] if not w["charge"])
    assert BYC._cell("terminator", quiet) == BYC._cell("release buckets", quiet) == BYC._cell("aborts", quiet) == "not exercised"
    busy = next(w for w in t["by_window"] if w["events"]["lookups_hit"] and w["events"]["released"])
    assert (BYC._cell("hits", busy), BYC._cell("release buckets", busy)) == ("0", "no field")
    assert BYC._cell("terminator", busy) == str(busy["events"]["length_terms"]) and BYC._cell("segment end", busy) == "not exercised"
    assert text.count("- hits: 0 by construction: ") == 2 and BYC.render_all([t, p]) == text
    assert f"The pool evicted {ev['evicted_blocks']} cached blocks; {ev['admissions_cut']} admissions' lookups" in text
    assert f"Aborts in its record: the engine freed {N} requests" in text and ", 0 of them aborted." in text
    twice = BYC.render_all([t, dict(t, trace="again"), p])
    assert twice.count("- output lengths: charged.") == 2 and twice.count("decode(encode(a)) == a: yes.") == 3


def test_a_trace_with_an_abort_is_its_refusal_and_its_aborts(tmp_path, capsys):
    """The plan's run with one request the engine ended `abort`: the reader refuses the trace, so it has no table, and
    the report is the refusal and what the directory records: the engine's finishes, the released requests it never
    freed, and the connections the network certifier failed, apart for those whose request it had released."""
    e, (ios, times) = engine(False), plan()
    d = write_corpus(tmp_path / "tick", e, ios, times, serve(e, ios, times, seed=1), tick=True)
    reqs = [json.loads(line) for line in (d / "requests.jsonl").read_text().splitlines()]
    reqs[5]["finish_reason"] = "abort"
    (d / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reqs))
    (d / "client.jsonl").write_text("".join(json.dumps({"id": i}) + "\n" for i in ("r0", "r5", "gone-1", "gone-2")))
    (d / "warden").mkdir()
    conns = [{"release": 0, "failed": None}, {"release": 5, "failed": "limit"}, {"release": 6, "failed": "late"},
             {"release": 7, "failed": "limit"}, {"release": None, "failed": "ingress"}]
    (d / "warden" / "warden.json").write_text(json.dumps({"connections": conns}))
    out = tmp_path / "report.json"
    assert BYC.main([f"abort={d}", "--stops", str(EOS), "--T", "10", "--out", str(out)]) == 1
    row = json.loads(out.read_text())["rows"][0]
    stops = sum(io.eos for k, io in enumerate(ios) if k != 5)
    assert set(row) == {"trace", "error", "aborts"} and "request r5: finish 'abort'" in row["error"]
    assert row["aborts"] == {"freed": N, "finishes": {"abort": 1, "length": N - 1 - stops, "stop": stops}, "aborted": 1,
                             "released_never_freed": 2,
                             "certifier_failed": {"released": {"limit": 2, "late": 1}, "unreleased": {"ingress": 1}}}
    text = capsys.readouterr().out
    assert text.startswith("**abort**: refused, so it has no table: request r5: finish 'abort'") and "|" not in text
    assert f"Aborts in its record: the engine freed {N} requests (1 `abort`, " in text and "1 of them aborted; 2 released" in text
    assert "after their request's release: 1 `late`, 2 `limit`; and before it: 1 `ingress`." in text


def test_the_report_names_the_first_step_that_differs(tmp_path, capsys):
    """A trace whose registration isn't the engine's: read as served with the prefix cache off, the plan's run has rows
    that start from hits the replay's pool never gives. The decoder refuses its string, and the report says so with
    the decoder's reason, exits 1 and names the first step at which the replay differs from the engine's schedule, the
    first one the engine admitted from a hit. The split by class is still each window's charge."""
    e, (ios, times) = engine(False), plan()
    run = serve(e, ios, times, seed=1)
    d = write_corpus(tmp_path / "tick", e, ios, times, run, tick=True)
    meta = json.loads((d / "meta.json").read_text())
    meta["server_argv"].remove("--enable-prefix-caching")
    (d / "meta.json").write_text(json.dumps(meta))
    out = tmp_path / "report.json"
    assert BYC.main([f"tick={d}", "--stops", str(EOS), "--T", "10", "--out", str(out)]) == 1
    row = json.loads(out.read_text())["rows"][0]
    first = min(s for s, st in enumerate(R.read_corpus_trace(d).steps) if any(st.cache_hit.values()))
    assert not row["round_trip"] and not row["prefix_caching"] and row["total"]["events"]["admissions_hit"] == 0
    assert "the walk's rows" in row["refused"]
    assert row["first_step_that_differs"]["step"] == first and row["engine_hits"]["admissions_hit"] > 1
    assert all(sum(w["fields"].values()) == w["charge"] for w in row["by_window"]) and row["bits"] > 0
    text = capsys.readouterr().out
    assert f"decode(encode(a)) == a: NO. The decoder refuses the string: {row['refused']}." in text
    assert f"The replay's schedule differs from the engine's from step {first}, in window " in text
