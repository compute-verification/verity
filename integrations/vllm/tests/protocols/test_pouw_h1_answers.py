"""`answers`: the verifier of record's answers on a window's statements, run once and shared by the steps that judge them
(credit's window check and the period's, ``--lean-answers``), the drawn ones `fill`ed J at a time. Over the statements kit
of core's credit tests (`test_pouw_credit_statements`: periods of proof files and a stand-in ``flock-verify`` that
accepts a session only when it proves every file credit hands it), under the Lean verdict of record and the Python
reference: two steps over the same statements run the verifier once a statement, the second taking every answer by key;
an input changed between them is another key, which the verifier answers afresh; the order the fill's runs end in changes
no answer and no verdict; and a statement the verifier rejects, refuses or can't run refuses both steps, as it does with
no memo."""
from __future__ import annotations

import functools
import importlib.util
import json
import random
import re
import shutil
import stat
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity.core.protocols.pouw import credit as Cr
from verity.core.protocols.pouw import verifier as V
from verity.core.protocols.sampled_proofs.one_stage import proofs as PF
from verity_vllm.protocols.pouw.h1 import answers as A
from verity_vllm.protocols.pouw.h1 import drawn as DRAWN

STATEMENTS = Path(__file__).resolve().parents[4] / "verity/core/protocols/pouw/tests/test_pouw_credit_statements.py"
SEEDS = range(2)


@functools.cache
def kit():
    """credit's statements kit (`test_pouw_credit_statements`), as ``benchmarks/one_stage``'s pouw_piece tests load it."""
    spec = importlib.util.spec_from_file_location("pouw_credit_statements", STATEMENTS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=["lean", "python"])
def record(request, monkeypatch):
    """Credit's verdict of record: the Lean's (`verifier.RECORD_VERDICT`, as served) and the Python reference's."""
    monkeypatch.setattr(V, "RECORD_VERDICT", request.param)
    return request.param


@pytest.fixture
def exe(tmp_path, monkeypatch) -> Path:
    exe = tmp_path / "flock-verify"
    exe.write_text(kit().FAKE)
    exe.chmod(0o755)
    monkeypatch.setattr(PF, "executable", lambda: exe)
    return exe


class Counted:
    """`proofs.verify`, or `answer`'s answer where it gives one (not None), each run's public file recorded."""

    def __init__(self, answer=None):
        self.runs: list[str] = []
        self.answer = answer
        self.lock = threading.Lock()

    def __call__(self, proof, **kw):
        with self.lock:
            self.runs.append(str(proof.public))
        got = None if self.answer is None else self.answer(proof)
        return PF.verify(proof, **kw) if got is None else got


def committed(d: dict) -> tuple[dict, int, int]:
    """The circuit files, K and table rows period `d`'s open committed, which credit checks each of its windows on."""
    (files, why), (k, why_k), (t, why_t) = (Cr.committed_files(d["ledger"], d["name"]), Cr.committed_k(d["ledger"], d["name"]),
                                            Cr.committed_rows(d["ledger"], d["name"]))
    assert why is None and why_k is None and why_t is None, (why, why_k, why_t)
    return files, k, t


def window_check(d: dict, w: dict, run) -> tuple:
    """Step 3: credit's check of window `w` of period `d` (`credit.window`, as `challenger_check.credit_reason` runs it)."""
    files, k, t = committed(d)
    return Cr.window(d["key"], d["ledger"], d["pins"], d["main"], *(w[x] for x in kit().cases.WINDOW_ARGS[4:14]),
                     w["statements"], w["live"], files, k, t, kit().rc.lms_verify, run=run)


def period_check(d: dict, windows: list[dict], run) -> tuple:
    """Step 4: the period's credit (`credit.period_credit`, as pouw_piece.py's ``period`` runs it)."""
    return Cr.period_credit(d["key"], d["ledger"], d["pins"], d["main"], d["name"], windows, kit().rc.lms_verify, run=run)


def steps(d: dict, windows: list[dict], window_run, period_run) -> tuple[list[tuple], tuple]:
    return [window_check(d, w, window_run) for w in windows], period_check(d, windows, period_run)


def runs(d: dict, windows: list[dict]) -> list:
    """`answers.window_runs` of each window's drawn statements, in window order: the runs ``drawn.py fill`` makes."""
    files, k, _t = committed(d)
    out = []
    for w in windows:
        got, refused = A.window_runs(w["statements"], files, bytes.fromhex(w["descriptor"]), kit()._layout(w),
                                     w["entries"], w["window"]["root"], kit()._units(w, k), len(w["rounds"]), w["salt"],
                                     w["draw"])
        assert refused is None, refused
        out += got
    return out


def statements(windows: list[dict]) -> list[str]:
    return [s["public"] for w in windows for s in w["statements"] + w["live"]["statements"]]


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("memo", ["scratch directory", "one process"])
def test_two_steps_over_the_same_statements_run_the_verifier_once_a_statement(seed, memo, record, exe, tmp_path):
    """Credit's window check and the period's over one memo: the first runs the verifier once on each statement, drawn
    and live; the second runs it on none, taking each answer by key; both judge as they do with no memo. The memo's
    directory is the verifier's alone (0700, each answer 0600)."""
    d, windows = kit()._period(seed, tmp_path / "proofs")
    every = statements(windows)
    plain = Counted()
    want = steps(d, windows, plain, plain)
    assert want[1][1] is None and all(why is None for _got, why in want[0])
    assert sorted(plain.runs) == sorted(every * 2)

    ran, said = Counted(), []
    if memo == "scratch directory":
        scratch = tmp_path / "scratch" / "answers"
        piece = A.Answers(scratch, by="the challenger check", verify=ran, say=said.append)
        period = A.Answers(scratch, by="the period check", verify=ran, say=said.append)
    else:
        piece = period = A.Answers(by="the verifier", verify=ran, say=said.append)
    assert [window_check(d, w, piece) for w in windows] == want[0]
    assert sorted(ran.runs) == sorted(every)
    assert period_check(d, windows, period) == want[1]
    assert sorted(ran.runs) == sorted(every)
    n = len(every)
    assert [x.split(": ")[0].split(" ", 2)[2] for x in said] == ["ran"] * n + [f"by key from {piece.by}"] * n
    assert all(x.endswith(": accepted 1 of 1 sessions") for x in said)
    if memo == "scratch directory":
        assert (piece.ran, piece.taken, period.ran, period.taken) == (n, {}, 0, {"the challenger check": n})
        assert period.summary() == f"{n} of {n} runs by key ({n} from the challenger check), 0 ran"
        assert stat.S_IMODE(scratch.stat().st_mode) == 0o700
        assert sorted(stat.S_IMODE(p.stat().st_mode) for p in scratch.iterdir()) == [0o600] * n
    else:
        assert (piece.ran, piece.taken) == (n, {"the verifier": n})


def _flip(p: Path, at: int = -2) -> None:
    """One byte of the file at `p` changed in place, its size kept."""
    b = bytearray(p.read_bytes())
    b[at] ^= 1
    with open(p, "r+b") as f:
        f.write(b)


#: each input of a run changed: its files in place (each keeping its size), its files at other paths, the verifier's
#: executable, and each argument credit builds
CHANGES = {
    "a session file": lambda proof, kw, exe: (_flip(proof.sessions[0] / "session.json"), proof, kw),
    "a file added to a session": lambda proof, kw, exe: ((proof.sessions[0] / "extra").write_text("x"), proof, kw),
    "the public file": lambda proof, kw, exe: (_flip(proof.public), proof, kw),
    "the circuit file": lambda proof, kw, exe: (_flip(proof.circuit), proof, kw),
    "the sessions' paths": lambda proof, kw, exe: (
        shutil.copytree(proof.sessions[0], proof.sessions[0].with_name("moved")),
        PF.Proof(proof.circuit, proof.public, (proof.sessions[0].with_name("moved"),)), kw),
    "the executable": lambda proof, kw, exe: (exe.write_text(exe.read_text() + "# another build\n"), proof, kw),
    "the circuit digest": lambda proof, kw, exe: (None, proof, {**kw, "circuit_sha512": "0" * 128}),
    "the program": lambda proof, kw, exe: (None, proof, {**kw, "program": kw["program"] + b" "}),
    "the partition": lambda proof, kw, exe: (None, proof, {**kw, "partition": {**kw["partition"], "format": "other"}}),
    "the registered file": lambda proof, kw, exe: (None, proof, {**kw, "registered": {**kw["registered"], "ports": {}}}),
}


@pytest.mark.parametrize("change", CHANGES)
def test_a_changed_input_is_another_key(change, exe, tmp_path):
    d, windows = kit()._period(0, tmp_path / "proofs")
    _label, proof, kw = runs(d, windows)[0]
    before = A.key(proof, exe=exe, **kw)
    assert A.key(proof, exe=exe, **kw) == before
    _done, proof, kw = CHANGES[change](proof, kw, exe)
    assert A.key(proof, exe=exe, **kw) != before


def test_a_statement_changed_between_the_steps_is_answered_afresh(record, exe, tmp_path):
    """A session changed after the window check answered it is another key: the period's check runs the verifier on it
    again (and on nothing else), and refuses as it does with no memo."""
    d, windows = kit()._period(0, tmp_path / "proofs")
    scratch = tmp_path / "scratch"
    assert all(why is None for _got, why in (window_check(d, w, A.Answers(scratch, by="the challenger check", say=None))
                                             for w in windows))
    first = windows[0]["statements"][0]
    rec = Path(first["sessions"][0]) / "session.json"
    rec.write_text(json.dumps({**json.loads(rec.read_text()), "proof": "00" * 64}))
    want = period_check(d, windows, Counted())
    assert want[0] is None and want[1][0] == "window", want
    ran = Counted()
    period = A.Answers(scratch, by="the period check", verify=ran, say=None)
    assert period_check(d, windows, period) == want
    assert ran.runs == [first["public"]] and period.ran == 1


def _ending(order: list[int], said: list[str]):
    """`proofs.verify` whose runs end in `order` (places among the fill's runs, each identified by its public file),
    and the fill's ``say``: the run after each in `order` starts only once the fill has said the one before it ended."""
    go = [threading.Event() for _ in order]
    go[order[0]].set()
    after = dict(zip(order, order[1:]))
    place: dict[str, int] = {}

    def verify(proof, **kw):
        go[place[str(proof.public)]].wait()
        return PF.verify(proof, **kw)

    def say(line: str) -> None:
        said.append(line)
        m = re.match(r"LEAN \d+/\d+ run (\d+) of ", line)
        if m and int(m[1]) - 1 in after:
            go[after[int(m[1]) - 1]].set()
    return verify, say, place


@pytest.mark.parametrize("seed", SEEDS)
def test_the_order_the_fills_runs_end_in_changes_no_answer_and_no_verdict(seed, record, exe, tmp_path):
    """The fill runs every drawn statement of the period at once and they end in a forced order (in order, reversed and
    shuffled): its ``LEAN`` lines follow that order, one a verdict, its memo holds the same answers whatever the order,
    and credit's window checks over it take every drawn answer by key and judge as with no memo. A fill over a full
    memo runs nothing."""
    d, windows = kit()._period(seed, tmp_path / "proofs")
    todo = runs(d, windows)
    n = len(todo)
    assert n >= 2
    want = [window_check(d, w, PF.verify) for w in windows]
    kept = []
    for i, order in enumerate([list(range(n)), list(reversed(range(n))), random.Random(seed).sample(range(n), n)]):
        memo, said = tmp_path / f"memo{i}", []
        verify, say, place = _ending(order, said)
        place.update({str(proof.public): j for j, (_l, proof, _kw) in enumerate(todo)})
        answers = A.Answers(memo, by="the fill", verify=verify, say=None, clock=lambda: 0.0)
        assert A.fill(answers, todo, n, say=say) == {"ran": n, "held": 0, "unkeyed": 0}
        assert said[0] == f"LEAN {n} runs: {n} to run, {n} at a time, J set (--jobs) (0 already answered, 0 unkeyed)"
        assert [int(re.match(r"LEAN \d+/\d+ run (\d+) of ", x)[1]) - 1 for x in said[1:]] == order
        assert all(x.endswith(": accepted 1 of 1 sessions in 0.0 s") for x in said[1:])
        kept.append({p.name: p.read_text() for p in sorted(memo.iterdir())})
        check = A.Answers(memo, by="the challenger check", say=None)
        assert [window_check(d, w, check) for w in windows] == want
        assert check.taken == {"the fill": n} and check.ran == sum(len(w["live"]["statements"]) for w in windows)
    assert kept[1] == kept[0] and kept[2] == kept[0]
    said = []
    assert A.fill(A.Answers(tmp_path / "memo0", by="the fill", say=None), todo, 2, say=said.append) == {
        "ran": 0, "held": n, "unkeyed": 0}
    assert said == [f"LEAN {n} runs: 0 to run, 2 at a time, J set (--jobs) ({n} already answered, "
                    "0 unkeyed)"]


def test_drawn_fill_answers_a_windows_runs_for_its_window_check_to_take_by_key(record, exe, tmp_path, monkeypatch,
                                                                               capsys):
    """``drawn.py fill`` (`drawn.fill`) on a window's statements, its window read as `drawn._checked` reads a serve dir:
    its header, one ``LEAN`` line a verdict and its counts; credit's window check then takes each drawn answer by key."""
    d, windows = kit()._period(0, tmp_path / "proofs")
    w = windows[0]
    files, k, _t = committed(d)
    win = {"raw": bytes.fromhex(w["descriptor"]), "layout": kit()._layout(w), "entries": w["entries"],
           "root": w["window"]["root"], "steps": len(w["rounds"]), "salt": w["salt"], "draw": w["draw"]}
    monkeypatch.setattr(DRAWN, "_checked", lambda a: (win, kit()._units(w, k), files, w["statements"]))
    memo = tmp_path / "scratch"
    assert DRAWN.run("fill", SimpleNamespace(answers=memo, jobs=2, headroom_gb=None)) == 0
    out, n = capsys.readouterr().out.splitlines(), len(w["statements"])
    assert out[0] == (f"LEAN {n} runs: {n} to run, 2 at a time, J set (--jobs) (0 already answered, "
                      "0 unkeyed)")
    assert len(out) == n + 2 and all(": accepted 1 of 1 sessions in " in x for x in out[1:-1])
    assert out[-1] == f"LEAN filled: {n} ran, 0 already answered, 0 unkeyed"
    check = A.Answers(memo, by="the challenger check", say=None)
    assert window_check(d, w, check) == window_check(d, w, PF.verify)
    assert check.taken == {"the fill": n}


def _tamper(s: dict) -> None:
    rec = Path(s["sessions"][0]) / "session.json"
    rec.write_text(json.dumps({**json.loads(rec.read_text()), "proof": "00" * 64}))


def _raise(e: Exception):
    def answer(proof):
        raise e
    return answer


#: what the verifier does on one statement: rejects its session (its proof tampered), refuses it, can't run (a
#: `proofs.LeanError`, or a KeyError, whose ``str`` quotes its message), or meets its session missing, so no key
FAILURES = {
    "rejected": (_tamper, lambda public: None),
    "refused": (lambda s: None, lambda public: lambda proof: (
        (None, ("registered", "the registered file names no port")) if str(proof.public) == public else None)),
    "LeanError": (lambda s: None, lambda public: lambda proof: (
        _raise(PF.LeanError("lake build flock-verify failed"))(proof) if str(proof.public) == public else None)),
    "KeyError": (lambda s: None, lambda public: lambda proof: (
        _raise(KeyError("ports"))(proof) if str(proof.public) == public else None)),
    "missing": (lambda s: shutil.rmtree(s["sessions"][0]), lambda public: None),
}


@pytest.mark.parametrize("failure", FAILURES)
def test_a_statement_the_verifier_refuses_refuses_both_steps(failure, record, exe, tmp_path):
    """Window 0's last drawn statement rejected, refused or not verifiable: its window check and the period's both
    refuse, with the refusals they give with no memo; the period's takes the window check's answer by key without
    running the verifier, except for a session that is missing, which has no key and which each step's verifier meets."""
    d, windows = kit()._period(0, tmp_path / "proofs")
    target = windows[0]["statements"][-1]
    change, answer = FAILURES[failure]
    change(target)
    answer = answer(target["public"])
    want = steps(d, windows, Counted(answer), Counted(answer))
    assert want[0][0][0] is None and want[1][0] is None and want[1][1][0] == "window", want
    assert all(why is None for _got, why in want[0][1:])

    scratch, ran, said = tmp_path / "scratch", Counted(answer), []
    piece = A.Answers(scratch, by="the challenger check", verify=ran, say=said.append)
    got = [window_check(d, w, piece) for w in windows]
    assert got == want[0]
    before = len(ran.runs)
    period = A.Answers(scratch, by="the period check", verify=ran, say=said.append)
    assert period_check(d, windows, period) == want[1]
    if failure == "missing":
        assert ran.runs[before:] == [target["public"]] and (piece.unkeyed, period.unkeyed) == (1, 1)
    else:
        assert ran.runs[before:] == [] and period.ran == 0 and period.unkeyed == 0
        verdicts = {"rejected": "rejected session ", "refused": "refused registered: the registered file names no port",
                    "LeanError": "error LeanError: lake build flock-verify failed", "KeyError": "error KeyError: 'ports'"}
        assert said[-1].split(": ", 1)[1].startswith(verdicts[failure]) and " by key from the challenger check: " in said[-1]


def test_j_is_the_available_memory_less_the_headroom_over_one_runs_peak_unless_set(tmp_path):
    assert A.jobs(memory=190.6) == (3, "floor((190.6 GiB available - 16 GiB headroom) / 47.5 GiB a run)")
    assert [A.jobs(memory=m)[0] for m in (663.0, 63.5, 40.0)] == [13, 1, 1]
    assert A.jobs(memory=190.6, headroom=60) == (2, "floor((190.6 GiB available - 60 GiB headroom) / 47.5 GiB a run)")
    assert A.jobs("7", memory=190.6, headroom=60) == (7, "set (--jobs)") and A.jobs(0)[0] == 1
    said = []
    A.fill(A.Answers(by="the fill", exe=lambda: tmp_path, say=None), [], headroom=60, say=said.append)
    assert " GiB available - 60 GiB headroom) / 47.5 GiB a run) (0 already answered, 0 unkeyed)" in said[0]
