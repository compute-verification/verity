"""Advice v1's charge by class (`verity_vllm.pipeline.advice_code.by_class`).
Each window's parts sum to its charge, and its segment's parts are the arithmetic decoder's own shifts, under the
tick and without it, under the uniform book and a fit one. Under the attested tick, with the prefix cache on and
prompts that share prefixes, every field but the lengths is 0 by construction, each by rules whose lines are in the
code; without the tick the admission is charged and the hits still aren't. The observer reads off the replay the
evictions, the lookups they cut short, the waits with what held them, and the preemptions. v1's own table refuses a
trace served with a draft by name."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest
from test_tick import EOS, R, TICK, buckets, engine, plan, serve, tick_case, write_corpus

from verity_vllm.pipeline.advice_code import by_class as BC
from verity_vllm.pipeline.advice_code import pool as P
from verity_vllm.pipeline.advice_code import replay as RP
from verity_vllm.pipeline.advice_code.codec import length_widths

HERE = (Path(__file__).resolve().parents[4] / "benchmarks" / "network_e2e" / "encodings")


def _load(name: str):
    key = f"enc_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, HERE / f"{name}.py")
        sys.modules[key] = mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return sys.modules[key]


V, CV = _load("v1"), _load("code_v1")

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
            CV.code_executionPhase(d, None, st)
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
    assert set(BC.WHY) >= {"max_num_seqs", "tokens", "blocks", "preempted"}
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


#: the drafts a row carries in the drafted serve
K = 3
MTP = ("--speculative-config.method", "mtp", "--speculative-config.num_speculative_tokens", str(K))


def drafted(asy: bool, seed: int = 11):
    """The plan under the attested tick on an engine that drafts `K` tokens a row: the engine, the plan's requests,
    each with the drafts its emitting rows kept, their release times, and the tokens each executed step's rows
    emitted. A row's kept count is random, seeded by `seed`, up to the drafts it carries and the outputs its request
    still lacks, as an accept rule might give; each is chosen as the tick's replay asks for it."""
    e = R.DraftEngine(**dataclasses.asdict(engine(asy)), num_speculative_tokens=K)
    (ios, times), rng = plan(), random.Random(seed)
    kept, emitted, now = [[] for _ in ios], [], {}
    update0 = RP.Scheduler.update

    def update(s, step, samples):
        if not s.whatif:
            now.update(s=s, step=step)
            if step.rows:
                emitted.append({})
        return update0(s, step, samples)

    def ask(i: int, _row: int) -> int:
        k = rng.randint(0, min(now["step"].drafts.get(i, 0), ios[i].output_len - now["s"].reqs[i].n_out - 1))
        kept[i].append(k)
        emitted[-1][i] = 1 + k
        return k

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(RP.Scheduler, "update", update)
        rp = R.replay(e, ios, tick=R.Tick(*TICK), released=buckets(times), kept=ask)
    assert len(emitted) == len(rp.steps)
    return e, [dataclasses.replace(io, kept=tuple(k)) for io, k in zip(ios, kept)], times, emitted


def with_draft(d: Path, ios, emitted, *argv: str) -> Path:
    """A trace-corpus directory (`write_corpus`'s) as a serve with a draft leaves it: the server's draft arguments,
    and for each step what the hook records of the model runner's output, the tokens it sampled for each row of its
    batch. An emitting row's are the tokens it emitted (`emitted`) and, where they end its request, one more the
    runner accepted past the stop; a row run after its request ended has one too, which the scheduler never took;
    a prefill chunk's are none."""
    meta = json.loads((d / "meta.json").read_text())
    meta["server_argv"] += argv
    (d / "meta.json").write_text(json.dumps(meta))
    steps = [json.loads(line) for line in (d / "steps.jsonl").read_text().splitlines()]
    done = [0] * len(ios)
    for s, out in zip(steps, emitted, strict=True):
        s["sampled"] = []
        for i in (int(rid[1:]) for rid in s["batch"]):
            io, at = ios[i], ios[i].prompt_len + done[i]
            done[i] += out.get(i, 0)
            over = [9] if done[i] == io.output_len else []
            s["sampled"].append([*io.tokens[at:at + out.get(i, 0)], *over])
    (d / "steps.jsonl").write_text("".join(json.dumps(s) + "\n" for s in steps))
    return d


@pytest.mark.parametrize("asy", (False, True), ids=("sync", "async"))
def test_the_corpus_table_refuses_a_serve_with_a_draft_by_name(tmp_path, capsys, asy):
    """v1's own table (`v1.py`) is of I1 v0, whose rows emit one output each: for the drafted serve's directory it has
    an error row that says so and names what reads the schedule, where the same serve without a draft has its row."""
    e, ios, times, emitted = drafted(asy)
    run = serve(e, ios, times, seed=1)
    d = with_draft(write_corpus(tmp_path / "draft", e, ios, times, run, tick=True), ios, emitted, *MTP)
    small, (plain, _) = engine(asy), plan()
    there = write_corpus(tmp_path / "there", small, plain, times, serve(small, plain, times, seed=1), tick=True)
    out = tmp_path / "table.json"
    assert V.main([f"draft={d}", f"there={there}", "--stops", str(EOS), "--T", "10", "--jobs", "1", "--out", str(out)]) == 1
    draft, other = json.loads(out.read_text())["rows"]
    why = "the trace was served with a draft, and I1 v0's rows emit one output each: `schedule_of` gives its schedule"
    assert draft == {"trace": "draft", "error": why} and "error" not in other and other["held_out"]["round_trip"]
    assert f"| draft | error: {why} |" in capsys.readouterr().out
    with pytest.raises(V.AdviceCodeError, match="^the trace was served with a draft"):
        V.advice_of(R.read_corpus_trace(d), V.record_of(R.read_corpus_trace(d), (EOS,), bucket_s=TICK[0] / 1e6))
