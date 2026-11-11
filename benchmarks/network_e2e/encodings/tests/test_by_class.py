"""Advice v1's charge by class (`verity_vllm.pipeline.advice_code.by_class`, and `by_class.py`'s report over a trace).
Each window's parts sum to its charge, and its segment's parts are the arithmetic decoder's own shifts, under the
tick and without it, under the uniform book and a fit one. Under the certified tick, with the prefix cache on and
prompts that share prefixes, every field but the lengths is 0 by construction, each by rules whose lines are in the
code; without the tick the admission is charged and the hits still aren't. The report counts the replay's lookups and
its hits, which are the engine's own, and where the decoder refuses the string it gives the decoder's reason and the
first step at which the replay differs from the engine's schedule."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from test_tick import EOS, N, R, engine, plan, serve, tick_case, write_corpus

from verity_vllm.pipeline.advice_code import by_class as BC
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
    decoder counts over the same string, its terminator is the 2 bits of a segment that coded anything, and its
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
            assert parts[w]["terminator"] == (2 if d.seg.used else 0), (name, w)
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
            assert p["output_len"] + p["finish"] == charge[w] and not any(p[f] for f in (*SEGMENT, "terminator"))
        shut = BC.closed(r)
        assert tuple(shut) == (*SEGMENT, "terminator")
        assert shut["hits"] == (("tick book",) if r.book is not None else ("caps",)) + ("one value", "refused")
        assert shut["decisions"] == shut["order"] == (("tick book",) if r.book is not None else ("tick caps",)) + (
            "one value", "tick replay", "tick refused")
        assert shut["terminator"] == ("no segment",)
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
        assert all(p["terminator"] == 2 and not (p["hits"] or p["resumes"] or p["lag"]) for p in parts.values()), name
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
        assert rule.line in (package / file).read_text().splitlines()[line - 1], name
    assert {r for rules in BC.NO_FIELD.values() for r in rules} <= set(BC.RULES)
    monkeypatch.setitem(BC.RULES, "caps", BC.Rule("as before", "codec", "CAPS = CV.Caps(hits=1)"))
    with pytest.raises(V.AdviceCodeError, match="the rule 'caps' is on 0 lines of codec.py"):
        BC.where("caps")
    monkeypatch.setitem(BC.RULES, "caps", BC.Rule("as before", "codec", "CAPS"))
    with pytest.raises(V.AdviceCodeError, match="moved"):
        BC.where("caps")


def test_the_report_splits_a_trace_by_class_and_counts_its_hits(tmp_path, capsys):
    """`by_class.py` over the plan's tick-served run and the same run without the tick, as trace-corpus directories:
    the tick's charge is its length terms, every other class 0 by construction or without a field, each rule with its
    file and line; the replay's admissions from a hit are the engine's own; the windows sum to the totals; and both
    strings decode back to their traces."""
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
    hits = [k for st in R.read_corpus_trace(tick).steps for k in st.cache_hit.values()]
    assert len(hits) > 1 and all(hits)
    assert t["round_trip"] and p["round_trip"] and t["first_step_that_differs"] is None
    assert t["bits"] == t["total"]["classes"]["lengths"] == lengths and t["windows"] > 1 and t["prefix_caching"]
    assert {c: v["standing"] for c, v in t["classes"].items()} == {
        "hits": "0 by construction", "admission": "0 by construction", "terminator": "0 by construction",
        "lengths": "charged", "resumes and lag": "0 by construction", "aborts": "no field", "release buckets": "no field"}
    assert p["classes"]["admission"]["standing"] == p["classes"]["terminator"]["standing"] == "charged"
    assert p["classes"]["hits"]["standing"] == "0 by construction" and p["total"]["classes"]["admission"] > 0
    assert p["bits"] == sum(p["total"]["classes"].values()) > lengths == p["total"]["classes"]["lengths"]
    for row in (t, p):
        for c in row["classes"].values():
            assert (c["standing"] == "charged") == (not c["rules"])
            for r in c["rules"]:
                assert r["file"].startswith("advice_code/") and r["line"] > 0 and r["why"] == BC.RULES[r["rule"]].why
        total = row["total"]
        for key in BYC.COUNTS:
            assert sum(w[key] for w in row["by_window"]) == total[key]
        assert all(sum(w["classes"].values()) == w["charge"] == sum(w["fields"].values()) for w in row["by_window"])
        assert total["admissions"] == total["length_terms"] == N and total["lookups"] >= N
        assert row["engine_hits"] == {"admissions_hit": len(hits), "hit_tokens": sum(hits),
                                      "hit_blocks": sum(hits) // e.block_size}
        assert (total["admissions_hit"], total["hit_blocks"]) == (len(hits), sum(hits) // e.block_size)
        assert total["admissions_hit"] <= total["lookups_hit"] <= total["lookups"]
    assert text.count("- hits: 0 by construction: ") == 2 and BYC.render_all([t, p]) == text
    twice = BYC.render_all([t, dict(t, trace="again"), p])
    assert twice.count("- lengths: charged.") == 2 and twice.count("decode(encode(a)) == a: yes.") == 3


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
    assert not row["round_trip"] and not row["prefix_caching"] and row["total"]["admissions_hit"] == 0
    assert "the walk's rows" in row["refused"]
    assert row["first_step_that_differs"]["step"] == first and row["engine_hits"]["admissions_hit"] > 1
    assert all(sum(w["fields"].values()) == w["charge"] for w in row["by_window"]) and row["bits"] > 0
    text = capsys.readouterr().out
    assert f"decode(encode(a)) == a: NO. The decoder refuses the string: {row['refused']}." in text
    assert f"The replay's schedule differs from the engine's from step {first}, in window " in text
