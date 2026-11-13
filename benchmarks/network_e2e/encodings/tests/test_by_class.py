"""Advice v1's charge by class (`verity_vllm.pipeline.advice_code.by_class`, and `by_class.py`'s report over a trace).
Each window's parts sum to its charge, and its segment's parts are the arithmetic decoder's own shifts, under the
tick and without it, under the uniform book and a fit one. Under the certified tick, with the prefix cache on and
prompts that share prefixes, every field but the lengths is 0 by construction, each by rules whose lines are in the
code; without the tick the admission is charged and the hits still aren't. The observer reads off the replay the
evictions, the lookups they cut short, the waits with what held them, and the preemptions. The report counts each
class's events beside its bits, a class with no bits and no event reading "not exercised"; its hits are the engine's
own; where the decoder refuses the string it gives the decoder's reason and the first step at which the replay
differs from the engine's schedule; and a trace with an abort is its refusal, the aborts its directory records, by
window, and that the run shows none of its windows. It prints ℓ beside each window's charge, the lengths' rule with
the load's `max_tokens`, and the charge scaled to a week, a month and a year at the serve's own rate; it refuses a
trace served with a draft; and set beside another serve's trace, read under that serve's own stop ids, it names what
that serve exercised and this one did not, and where the two serves differ."""

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
    termination's 2 bits, and no window sends a hit, a resume or a lag bit, which v1's own bounds hold at 0. A book with
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
    no bits reads 0. Both strings decode back to their traces. Each row has ℓ beside its charge; the lengths are their
    rule's line, the plan's `max_tokens` and the widths they give; the charge is scaled to each monitoring period by the
    seconds from the first release to the last; and the aborted requests are 0 in every window, a count and no blank."""
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
        held = len({(a - min(buckets(times))) // 10 for a in buckets(times)})
        assert row["aborts"] == {"freed": N, "finishes": {"length": ev["ended_length"], "stop": ev["ended_stop"]}, "aborted": 0,
                                 "windows_released": held if row is t else None, "by_window": [] if row is t else None,
                                 "released_never_freed": None, "certifier_failed": None}
        assert ev["aborted"] == 0 and row["ell"] == 1 << 23
        assert row["serve"] == {"model": None, "cards": 1, "draft": False, "vllm": None, "certifier": None}
        n, widths = row["lengths"], [io.max_tokens.bit_length() for io in ios]
        assert (n["terms"], n["bits"]) == (N, lengths) and n["rule"]["file"] == "advice_code/lengths.py"
        assert "C.width(requests[k].max_tokens + 1), 1)" in (Path(BC.__file__).parent / "lengths.py").read_text().splitlines()[
            n["rule"]["line"] - 1]
        assert n["max_tokens"] == {"min": min(io.max_tokens for io in ios), "max": max(io.max_tokens for io in ios),
                                   "mean": sum(io.max_tokens for io in ios) / N}
        assert n["width"] == {"min": min(widths), "max": max(widths), "mean": sum(widths) / N}
        assert row["scaled"] == {name: round(row["bits"] * s / n["release_span_s"]) for name, s in BYC.PERIODS.items()}
    span = (max(buckets(times)) - min(buckets(times)) + 1) * 0.1
    assert t["lengths"]["release_span_s"] == pytest.approx(span) and BYC.PERIODS["week"] == 604800
    assert t["scaled"]["year"] == round(lengths * 365 * 86400 / t["lengths"]["release_span_s"]) > t["scaled"]["month"]
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
    assert text.count("| window | charge | ℓ | hits |") == 2 and text.count("| aborts | aborted requests |") == 2
    assert all(line.split("|")[3].strip() == f"{1 << 23:,}" for line in text.splitlines() if line.startswith(("| all", "| 0")))
    assert text.count(f"ℓ = {1 << 23:,} bits bounds the whole string, every window's charge together.") == 2
    assert text.count("`decode`, the registered codebook and ℓ are held by Python.") == 2
    assert f"{lengths:,} bits charged by the registered code." in text and text.count("served with the draft off") == 2
    n = t["lengths"]
    assert (f"(`advice_code/lengths.py:{n['rule']['line']}`). The load's `max_tokens` runs from {n['max_tokens']['min']} to "
            f"{n['max_tokens']['max']}, mean {n['max_tokens']['mean']:.1f}; the widths from") in text
    assert f"{N} terms are {lengths:,} bits, {lengths / N:.3f} a request." in text
    assert (f"Scaled, not measured: at this serve's own rate on its 1 card, its {lengths:,} bits over the {span:,.1f} s from its first "
            f"release to its last, the charge is {t['scaled']['week']:,} bits a week, {t['scaled']['month']:,} a month of "
            f"30 days and {t['scaled']['year']:,} a year.") in text
    _, rec, _, reg = tick_case(False)
    none = BYC.length_terms(rec, dataclasses.replace(reg, send_lengths=False), 0.1)
    assert (none["terms"], none["bits"], none["max_tokens"], none["width"]) == (0, 0, None, None)
    assert "The lengths: the registration sends none." in BYC.render({**t, "lengths": none})
    twice = BYC.render_all([t, dict(t, trace="again"), p])
    assert twice.count("- output lengths: charged.") == 2 and twice.count("decode(encode(a)) == a: yes.") == 3


def test_a_trace_with_an_abort_is_its_refusal_and_its_aborts(tmp_path, capsys):
    """The plan's run with one request the engine ended `abort`: the reader refuses the trace, so it has no table, and
    the report is the refusal and what the directory records: the engine's finishes, the released requests it never
    freed, and the connections the network certifier failed, apart for those whose request it had released. The
    aborted request is counted in the window its release falls in, and the report says that the run shows the advice
    of none of the record's windows."""
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
    a = buckets(times)
    held, at = len({(x - min(a)) // 10 for x in a}), (a[5] - min(a)) // 10
    assert set(row) == {"trace", "error", "T", "serve", "aborts"} and "request r5: finish 'abort'" in row["error"]
    assert row["aborts"] == {"freed": N, "finishes": {"abort": 1, "length": N - 1 - stops, "stop": stops}, "aborted": 1,
                             "windows_released": held, "by_window": [{"window": at, "aborted": 1}],
                             "released_never_freed": 2,
                             "certifier_failed": {"released": {"limit": 2, "late": 1}, "unreleased": {"ingress": 1}}}
    text = capsys.readouterr().out
    assert text.startswith("**abort**: refused, so it has no table: request r5: finish 'abort'") and "|" not in text
    assert f"Aborts in its record: the engine freed {N} requests (1 `abort`, " in text and "1 of them aborted; 2 released" in text
    assert "after their request's release: 1 `late`, 2 `limit`; and before it: 1 `ingress`." in text
    assert held > 1 and (f"Aborted requests: 1, by the window of T = 10 buckets their release falls in: window {at}: 1. The "
                         f"code has no field for an abort and the reader refuses the whole record for one: this run does not "
                         f"show the advice of any of its {held} windows that hold a release.") in text
    (d / "requests.jsonl").write_text("".join(json.dumps({k: v for k, v in r.items() if k != "bucket"}) + "\n" for r in reqs))
    assert BYC.aborts(d, 10)["by_window"] is None is BYC.aborts(d, 10)["windows_released"]
    assert "so they are not counted by window. The record is refused whole" in BYC._unshown(BYC.aborts(d, 10), 10)


def _name(d: Path, model: str, cards: int, where: str, engine: bool = True) -> None:
    """Give a trace-corpus directory a serve's own facts: its model's name, its cards, where its network certifier ran
    and the version its server's log prints: another program's version and the banner's, which are not read, then
    with `engine` the engine's own line, which is."""
    meta = json.loads((d / "meta.json").read_text())
    meta["server_argv"] += ["--served-model-name", model, "--tensor-parallel-size", str(cards)]
    meta["tick"]["where"] = where
    (d / "meta.json").write_text(json.dumps(meta))
    with (d / "server.log").open("a") as log:
        log.write("INFO CUDA version 12.4\nINFO vLLM API server version 0.9.1rc2.dev3+gabc123\n")
        if engine:
            log.write("INFO [core.py:123] Initializing a V1 LLM engine (v0.9.1rc2.dev3+gabc123) with config: model='m'\n")


def test_a_serve_is_set_beside_the_serve_that_exercised_what_it_did_not(tmp_path, capsys):
    """The plan served by an engine with room for all of it, so that nothing is evicted or preempted, set beside the
    plan's run on the small engine and beside that run with an abort, which is refused: the report's own table is the
    roomy serve's alone; beside it are each class it did not exercise, the events with none in it and some in the small
    serve with that serve's counts, the refused trace's aborted request, and the two serves' facts, those that are the
    same (the vLLM version, the tick, the codebook, ℓ, the draft) apart from those that differ (the model, the cards,
    the pool, where the network certifier ran). A refused trace beside it doesn't fail the report."""
    small, (ios, times) = engine(False), plan()
    roomy = dataclasses.replace(small, num_gpu_blocks=4096)
    here = write_corpus(tmp_path / "here", roomy, ios, times, serve(roomy, ios, times, seed=1), tick=True)
    there = write_corpus(tmp_path / "there", small, ios, times, serve(small, ios, times, seed=1), tick=True)
    gone = write_corpus(tmp_path / "gone", small, ios, times, serve(small, ios, times, seed=1), tick=True)
    _name(here, "big", 8, "on the auditor's host")
    _name(there, "little", 1, "beside the server")
    _name(gone, "little", 1, "beside the server", engine=False)
    reqs = [json.loads(line) for line in (gone / "requests.jsonl").read_text().splitlines()]
    reqs[5]["finish_reason"] = "abort"
    (gone / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reqs))
    out = tmp_path / "report.json"
    assert BYC.main([f"here={here}", "--beside", f"there={there}", "--beside", f"gone={gone}", "--stops", str(EOS),
                     "--T", "10", "--out", str(out)]) == 0
    doc, text = json.loads(out.read_text()), capsys.readouterr().out
    row, (other, refused) = doc["rows"][0], doc["beside"]
    assert [r["trace"] for r in doc["rows"]] == ["here"] and "error" in refused and "error" not in other
    assert row["serve"] == {"model": "big", "cards": 8, "draft": False, "vllm": "0.9.1rc2.dev3+gabc123",
                            "certifier": "on the auditor's host"}
    ev, theirs = row["total"]["events"], other["total"]["events"]
    assert ev["preemptions"] == ev["evicted_blocks"] == ev["admissions_cut"] == 0 < theirs["preemptions"]
    a, b = row["beside"]
    assert (a["trace"], a["refused"], b["trace"]) == ("there", None, "gone") and b["refused"] == refused["error"]
    quiet = {c for c in BYC.OF if BYC._cell(c, row["total"]) == "not exercised"}
    assert quiet == {"aborts", "resumes and lag", "segment end"} == set(a["not_exercised"]) == set(b["not_exercised"])
    assert a["events"] == {"admissions_cut": theirs["admissions_cut"], "preemptions": theirs["preemptions"]} and not b["events"]
    assert set(BYC.CLASS_OF) == set(BYC.HEADS) and set(BYC.CLASS_OF.values()) == set(BYC.OF)
    assert a["not_exercised"]["aborts"] == {"aborted": 0} and b["not_exercised"]["aborts"] == {"aborted": 1}
    assert a["not_exercised"]["resumes and lag"] == {"resumptions": theirs["resumptions"], "lagged": 0}
    assert b["not_exercised"]["resumes and lag"] == {"resumptions": None, "lagged": None}
    assert b["not_exercised"]["segment end"] == {"segments": None} and a["not_exercised"]["segment end"] == {"segments": 0}
    assert refused["serve"]["vllm"] is None and other["serve"]["vllm"] == row["serve"]["vllm"]
    assert b["differs"]["vLLM"] == ["0.9.1rc2.dev3+gabc123", "not in the server's log"]
    assert a["differs"] == {"model": ["big", "little"], "cards": ["8", "1"],
                            "where the network certifier ran": ["on the auditor's host", "beside the server"],
                            "the pool": ["4,096 blocks of 16 tokens", "8 blocks of 16 tokens"]}
    assert set(a["same"]) == {"vLLM", "the draft", "requests a step", "tokens a step", "admission", "the prefix cache",
                              "the registered codebook", "ℓ", "T"}
    assert a["same"]["admission"] == "under the certified tick (τ_b = 100,000, τ_sat = 30,000, τ_free = 40,000 µs)"
    assert set(b["differs"]) == {"model", "cards", "vLLM", "where the network certifier ran"} and set(b["same"]) == {"the draft"}
    assert text.count("| all |") == 1 and "**there**" not in text and "**gone**" not in text
    assert "Beside `there`:\n\n- aborts: not exercised here; not exercised in `there` either.\n" in text
    assert f"- resumes and lag: not exercised here; in `there`: resumptions {theirs['resumptions']}.\n" in text
    assert (f"- in the classes exercised here, none here and some in `there`: admissions an eviction cut "
            f"{theirs['admissions_cut']}, preemptions {theirs['preemptions']}.\n") in text
    assert "none here and some in `gone`" not in text
    assert ("- the same in both serves: vLLM: 0.9.1rc2.dev3+gabc123; the draft: off; requests a step: at most 4; tokens a "
            "step: at most 64; admission: under the certified tick") in text
    assert ("- differing: model: big here, little in `there`; cards: 8 here, 1 in `there`; where the network certifier "
            "ran: on the auditor's host here, beside the server in `there`; the pool: 4,096 blocks of 16 tokens here, 8 "
            "blocks of 16 tokens in `there`.\n") in text
    assert "Scaled, not measured: at this serve's own rate on its 8 cards, its " in text
    assert "request r5: finish 'abort'" in refused["error"]
    assert (f"Beside `gone`, which is refused ({refused['error']}), so that only its aborts and its directory's facts "
            f"are read:\n\n- aborts: not exercised here; in `gone`: aborted requests 1.\n") in text
    assert "- segment end: not exercised here; `gone` has no table to say.\n" in text
    assert text.index("Beside `there`") < text.index("Beside `gone`") < text.index("- hits: 0 by construction")
    assert BYC.render_all(doc["rows"]) == text and doc["beside_stops"] == doc["stops"] == [EOS]


def test_a_serve_of_another_model_is_set_beside_under_its_own_stop_ids(tmp_path, capsys):
    """The small engine's serve as another model's, each of its stops on that model's stop id, set beside the roomy
    serve: read under the report's stop ids its record is refused, so it has no table to set beside and its line says
    why; read under its own (``--beside-stops``) it is set beside as the same serve on the report's stop id is."""
    small, (ios, times) = engine(False), plan()
    roomy = dataclasses.replace(small, num_gpu_blocks=4096)
    here = write_corpus(tmp_path / "here", roomy, ios, times, serve(roomy, ios, times, seed=1), tick=True)
    same = write_corpus(tmp_path / "same", small, ios, times, serve(small, ios, times, seed=1), tick=True)
    there = write_corpus(tmp_path / "there", small, ios, times, serve(small, ios, times, seed=1), tick=True)
    theirs = EOS + 7
    reqs = [json.loads(line) for line in (there / "requests.jsonl").read_text().splitlines()]
    assert any(r["eos"] for r in reqs) and not any(theirs in r["prompt_ids"] + r["output_ids"] for r in reqs)
    for r in reqs:
        if r["eos"]:
            r["output_ids"][-1] = theirs
    (there / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reqs))
    out = tmp_path / "report.json"
    args = [f"here={here}", "--stops", str(EOS), "--T", "10", "--out", str(out)]

    def beside(d: Path, *more: str) -> tuple[dict, dict, str]:
        assert BYC.main([*args, "--beside", f"there={d}", *more]) == 0
        doc = json.loads(out.read_text())
        return doc, doc["rows"][0]["beside"][0], capsys.readouterr().out

    doc, b, text = beside(there)
    assert doc["beside_stops"] == [EOS] and f"stopped on a stop id, its last id {theirs}" in b["refused"]
    assert f"Beside `there`, which is refused ({b['refused']}), so that only its aborts" in text
    assert "- segment end: not exercised here; `there` has no table to say.\n" in text
    doc, b, text = beside(there, "--beside-stops", str(theirs))
    assert doc["beside_stops"] == [theirs] and doc["stops"] == [EOS] and b["refused"] is None
    assert "Beside `there`:\n" in text and "no table to say" not in text and b["events"]["preemptions"] > 0
    assert beside(same)[1] == b


def test_a_trace_served_with_a_draft_is_refused(tmp_path, capsys):
    """The reader takes no draft from the server's arguments, so its replay of a serve that had one would be of
    another serve: such a trace is refused by name before it is read, and so a trace that has a table was served with
    the draft off."""
    e, (ios, times) = engine(False), plan()
    d = write_corpus(tmp_path / "tick", e, ios, times, serve(e, ios, times, seed=1), tick=True)
    meta = json.loads((d / "meta.json").read_text())
    meta["server_argv"] += ["--speculative-config", "{}"]
    (d / "meta.json").write_text(json.dumps(meta))
    out = tmp_path / "report.json"
    assert BYC.main([f"draft={d}", "--stops", str(EOS), "--T", "10", "--out", str(out)]) == 1
    row = json.loads(out.read_text())["rows"][0]
    assert row["serve"]["draft"] and row["error"].startswith("the server had a draft") and "bits" not in row
    text = capsys.readouterr().out
    assert text.startswith("**draft**: refused, so it has no table: the server had a draft") and "|" not in text
    assert "Aborted requests: 0, by the window of T = 10 buckets their release falls in: none in any window. The record is " \
           "refused whole: this run does not show the advice of any of its" in text


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
