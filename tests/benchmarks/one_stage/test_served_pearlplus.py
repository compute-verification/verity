"""`served_pearlplus.sh` in toy trees whose steps each note their argv.

Under PEARLPLUS_TIE: Pearl+'s piece over the window's own epoch, which serve-openai's Pearl+ (`serve_openai.scheme_settings`) retained
in the serve dir's ``pearl/w<k>/``: the epoch taken as the piece's passes, the give-up on the card, ``retain
--window-record W``, and ``verify --tie W``.

The give-up before the verifier's coin, with stand-ins for served_passes.py, debit.py, served_verdicts.py,
pearl_c_sm90.sh's setup and gpu-lease: it asks ``served_passes.py volunteer-needed`` before anything is built or served,
gives the rows over a debit cap up (on the card by default) exactly when the answer is yes, refuses to serve with the
give-up turned off then, leaves it off when the answer is no unless PEARLPLUS_VOLUNTEER asks for it, stops on any other
answer, and hands PEARLPLUS_OVER_CAP to both the question and the verdict."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

HERE = Path(__file__).resolve().parent

#: the toy venv's python: each served_passes.py step, debit.py and served_verdicts.py noted and answered as the real one
#: writes its files; anything else (the script's inline checks) is the real interpreter
STEP = r'''
import json, os, sys
from pathlib import Path
argv = sys.argv[1:]
with open(sys.argv[0] + ".log", "a") as f:
    f.write(json.dumps(argv) + "\n")
def opt(name):
    return argv[argv.index(name) + 1] if name in argv else None
script, cmd = Path(argv[0]).name, argv[1] if len(argv) > 1 else None
if script == "served_passes.py" and cmd == "volunteer-needed":
    print("yes reject")
elif script == "debit.py":
    Path(opt("--out")).write_text("{}")
elif script == "served_passes.py" and cmd == "volunteer":
    Path(opt("--out")).write_text(json.dumps({"s": 1.0, "jobs": 1, "passes": []}))
    print("volunteer PASS: 0 rows given up")
elif script == "served_passes.py" and cmd == "retain":
    if (w := opt("--window-record")) and not os.environ.get("TOY_NO_W"):
        Path(w).write_text(json.dumps({"window": {"scheme": "pearl-c-h100-v1-h3"}}))
        print("retain PASS: 3 of 3 passes with calls retained")
        print(f"retain W: {w}, rounds 0..2 (3), 384 values, digest abcd…")
    else:
        print("retain PASS: 3 of 3 passes with calls retained")
elif script == "served_passes.py" and cmd == "verify":
    Path(opt("--passes"), "coin.json").write_text("{}")
    Path(opt("--out")).write_text(json.dumps({"answer": {"verdict": "accept"}}))
    print("ACCEPT verity/pouw/pearl-plus/v1")
elif script == "served_passes.py" and cmd == "lean":
    print("lean: no executable")
elif script == "served_verdicts.py" and cmd == "coin":
    print("coin: kept")
elif script == "served_verdicts.py" and argv[1:3] == ["pearlplus-period", "rule"]:
    print(f"RULE no other reserve on lane {opt('--lane')} between period pp-r1's close and the draw: HELD (toy)")
elif script == "served_verdicts.py" and cmd == "pearlplus":
    print("POUW-DEV {}")
else:
    sys.exit(f"unexpected step {argv}")
'''


def _piece(tmp_path: Path, served: dict | None, period: dict | None = None, first: dict | None = None,
           declared: dict | None = None, **extra: str) -> tuple[subprocess.CompletedProcess, Path, Path]:
    """The piece on window 7 of a toy serve dir whose ``pearl/w7/served.json`` is ``served`` (none if None), whose
    window serve left ``pearlplus/period.json`` `period` (none if None), ``deployment-registration.json`` `first` (none
    if None) and ``declaration.json`` `declared` (none if None), under PEARLPLUS_TIE with the window serve's cubin and a
    toy gpu-lease: the run, the serve dir and the toy tree."""
    tree, bin_, venv, sd = tmp_path / "tree", tmp_path / "bin", tmp_path / "venv" / "bin", tmp_path / "serve"
    for p in (tree / "benchmarks" / "one_stage", tree / "benchmarks" / "harness", tree / ".venv" / "bin", bin_, venv,
              sd / "pearl" / "w7", tmp_path / "piece", tmp_path / "cubin"):
        p.mkdir(parents=True)
    (tree / "benchmarks" / "one_stage" / "served_pearlplus.sh").symlink_to(HERE.parents[2] / "benchmarks" / "one_stage" / "served_pearlplus.sh")
    (tree / "benchmarks" / "harness" / "env.sh").write_text("")
    (tmp_path / "step.py").write_text(STEP)
    py = (f'#!/bin/sh\ncase "$1" in -c|-) exec {sys.executable} "$@";; esac\n'
          f'exec {sys.executable} {tmp_path}/step.py "$@"\n')
    for path, body in ((venv / "python", py), (tree / ".venv" / "bin" / "python", py),
                       (bin_ / "gpu-lease", '#!/bin/sh\nwhile [ "$1" != -- ]; do shift; done\nshift\nexec "$@"\n')):
        path.write_text(body)
        path.chmod(0o755)
    (tmp_path / "cubin" / "pearl_c_sm90.cubin").write_bytes(b"cubin")
    if served is not None:
        (sd / "pearl" / "w7" / "served.json").write_text(json.dumps(served))
    if period is not None:
        (sd / "pearlplus").mkdir()
        (sd / "pearlplus" / "period.json").write_text(json.dumps(period))
    if first is not None:
        (sd / "deployment-registration.json").write_text(json.dumps(first))
    if declared is not None:
        (sd / "declaration.json").write_text(json.dumps(declared))
    env = {"PATH": f"{bin_}:{os.defpath}", "HOME": str(tmp_path), "VENV": str(venv.parent), "PEARLPLUS_TIE": "1",
           "PEARLC_KERNEL": "pearl_c_sm90", "PEARLC_CUBIN": str(tmp_path / "cubin" / "pearl_c_sm90.cubin"),
           "PEARLPLUS_JOBS": "1", **extra}
    run = subprocess.run(["bash", str(tree / "benchmarks" / "one_stage" / "served_pearlplus.sh"), str(sd), "7",
                          str(tmp_path / "piece")], cwd=tmp_path, env=env, check=False, capture_output=True, text=True)
    return run, sd, tree


def _steps(tmp_path: Path) -> list[list[str]]:
    log = tmp_path / "step.py.log"
    return [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []


def _epoch(ok: bool = True) -> dict:
    return {"schema": "verity-pouw/pearl-c-served-passes/v1", "entry": "pouw:pearl-c-h100-v1-h3", "window": 7,
            "commit_steps": 3, "passes_with_calls": 3, "coins": {"source": "in-process"}, "ok": ok,
            "error": None if ok else "its passes with calls are 2 and its Commit's steps 3, so round s isn't step s",
            "passes": [{"pass": i, "timed": True, "units": 64, "manifest": f"m{i}"} for i in range(3)]}


def test_under_pearlplus_tie_the_piece_ties_the_windows_own_epoch_to_its_w(tmp_path):
    """The piece serves nothing: its passes are the window's epoch (a symlink to ``pearl/w7``), the card gives up its rows
    over a cap (VOLUNTEER's default), `retain` writes W, `verify` holds the passes to W, and the piece reads the epoch's
    served.json."""
    run, sd, tree = _piece(tmp_path, _epoch())
    q = sd / "private" / "pearlplus"
    assert run.returncode == 0, (run.stdout, run.stderr)
    assert (q / "passes").is_symlink() and (q / "passes").resolve() == (sd / "pearl" / "w7").resolve()
    steps = [(Path(a[0]).name, a[1] if len(a) > 1 else None) for a in _steps(tmp_path)]
    assert steps == [("served_passes.py", "volunteer-needed"), ("debit.py", "--cubin"), ("served_passes.py", "volunteer"),
                     ("served_passes.py", "retain"), ("served_passes.py", "verify"), ("served_verdicts.py", "coin"),
                     ("served_passes.py", "lean"), ("served_verdicts.py", "pearlplus")]
    argv = {Path(a[0]).name + " " + a[1]: a for a in _steps(tmp_path)}
    w = str(q / "window-record.json")
    retain = argv["served_passes.py retain"]
    assert retain[retain.index("--passes") + 1] == str(q / "passes") and retain[retain.index("--window-record") + 1] == w
    assert retain[retain.index("--volunteer") + 1] == str(q / "volunteer.json") and "--previous-record" not in retain
    assert "--day-declaration" not in retain
    verify = argv["served_passes.py verify"]
    assert verify[verify.index("--tie") + 1] == w and verify[verify.index("--passes") + 1] == str(q / "passes")
    assert "--class" not in verify
    pearlplus = argv["served_verdicts.py pearlplus"]
    assert pearlplus[pearlplus.index("--served") + 1] == str(q / "passes" / "served.json")
    assert pearlplus[pearlplus.index("--tie") + 1] == w
    lines = run.stdout.splitlines()
    assert any(x.startswith("POUW window: the window's own Pearl+ epoch ") and x.endswith("coins in-process -> OK")
               for x in lines), lines
    assert "POUW W: rounds 0..2 (3), 384 values, digest abcd…" in lines
    assert (sd / "pearl" / "w7" / "coin.json").exists()


def test_under_pearlplus_tie_verify_holds_the_passes_to_the_windows_class(tmp_path):
    """With PEARLPLUS_CLASS (served_verdicts.sh's, the class the window is committed at) `verify` gets ``--class`` beside
    ``--tie W``, so it refuses passes that aren't the class's fixed load."""
    run, sd, _tree = _piece(tmp_path, _epoch(), PEARLPLUS_CLASS="4,32,3")
    assert run.returncode == 0, (run.stdout, run.stderr)
    verify = next(a for a in _steps(tmp_path) if Path(a[0]).name == "served_passes.py" and a[1] == "verify")
    w = str(sd / "private" / "pearlplus" / "window-record.json")
    assert verify[verify.index("--tie"):verify.index("--tie") + 4] == ["--tie", w, "--class", "4,32,3"]
    script = (HERE.parents[2] / "benchmarks" / "one_stage" / "served_verdicts.sh").read_text()
    assert "export PEARLPLUS_TIE PEARLPLUS_CLASS=${V3C_CLASS-4,32,4} " in script


def test_under_pearlplus_tie_w_is_chained_after_the_serves_first_registration_record(tmp_path):
    """Where the window serve wrote its first registration record (served_verdicts.sh's weights():
    SERVE_DIR/deployment-registration.json, which registers the weights before the first request), `retain` gets it as
    ``--previous-record`` beside ``--window-record W``, so W is chained after it and the weights' record is the log's
    first; a serve dir with none leaves W chained after nothing (the test above)."""
    run, sd, _tree = _piece(tmp_path, _epoch(), first={"previous": None})
    assert run.returncode == 0, (run.stdout, run.stderr)
    retain = next(a for a in _steps(tmp_path) if Path(a[0]).name == "served_passes.py" and a[1] == "retain")
    w = str(sd / "private" / "pearlplus" / "window-record.json")
    at = retain.index("--window-record")
    assert retain[at:at + 4] == ["--window-record", w, "--previous-record", str(sd / "deployment-registration.json")]


def test_under_pearlplus_tie_w_names_the_day_the_serve_declared(tmp_path):
    """Where the window serve declared a day's program (serve-openai's --day: SERVE_DIR/declaration.json), `retain` gets
    that file as ``--day-declaration`` beside ``--window-record W``, so W names the day as the windows' dockets do
    and the audit tied to W holds it to that day; a serve that declared none leaves W naming none (the tests above)."""
    run, sd, _tree = _piece(tmp_path, _epoch(), declared={"day": 3, "declaration": "5d" * 64})
    assert run.returncode == 0, (run.stdout, run.stderr)
    retain = next(a for a in _steps(tmp_path) if Path(a[0]).name == "served_passes.py" and a[1] == "retain")
    at = retain.index("--window-record")
    assert retain[at:at + 4] == ["--window-record", str(sd / "private" / "pearlplus" / "window-record.json"),
                                 "--day-declaration", str(sd / "declaration.json")]


def test_under_pearlplus_tie_a_window_whose_epoch_failed_or_is_missing_gets_no_retain_tie_or_verdict(tmp_path):
    """No retries: an epoch whose served.json isn't ok, or none at all, is the window's failure, and the piece gives up,
    retains, ties, verifies or asks the Lean nothing, still writing its piece over what it has, as the window's own
    epoch (``--tie W``)."""
    for name, served in (("failed", _epoch(ok=False)), ("missing", None)):
        d = tmp_path / name
        run, sd, _tree = _piece(d, served)
        assert [Path(a[0]).name + " " + a[1] for a in _steps(d)] == ["served_passes.py volunteer-needed",
                                                                     "served_verdicts.py pearlplus"], run.stdout
        pearlplus = _steps(d)[-1]
        assert pearlplus[pearlplus.index("--tie") + 1] == str(sd / "private" / "pearlplus" / "window-record.json")
        said = [x for x in run.stdout.splitlines() if x.startswith("POUW window: ")]
        assert len(said) == 1 and (said[0].endswith("so round s isn't step s") if served else
                                   said[0].endswith("so there are no passes to tie")), said


def test_under_pearlplus_tie_a_window_whose_w_retain_did_not_write_gets_no_sample_verdict_or_line(tmp_path):
    """Where `retain` wrote no W to tie, the piece says so and stops before the coin: no sample, no `verify` (tied or not), no
    Lean, and a piece written over a verdict document that isn't there, which `served_verdicts.pearlplus` rejects, so no
    Pearl+ line is published for a window nothing holds to its class."""
    run, sd, _tree = _piece(tmp_path, _epoch(), TOY_NO_W="1")
    assert run.returncode == 0, (run.stdout, run.stderr)
    names = [Path(a[0]).name + " " + a[1] for a in _steps(tmp_path)]
    assert names == ["served_passes.py volunteer-needed", "debit.py --cubin", "served_passes.py volunteer",
                     "served_passes.py retain", "served_verdicts.py pearlplus"], run.stdout
    pearlplus = _steps(tmp_path)[-1]
    assert not Path(pearlplus[pearlplus.index("--verify") + 1]).exists()
    assert not (sd / "pearl" / "w7" / "coin.json").exists()
    lines = run.stdout.splitlines()
    assert ("POUW W: retain wrote no window record, so nothing ties the passes to the window and no sample or Pearl+ "
            "line follows") in lines
    assert not any(x.startswith(("POUW verdict: ", "POUW coin: ", "POUW lean: ")) for x in lines), lines


@pytest.mark.parametrize("case", ["held", "unclosed", "refused", "unread"])
def test_under_pearlplus_tie_and_the_gate_the_piece_samples_only_after_a_period_whose_reserves_held(tmp_path, case):
    """Under the gate the window serve closed its Pearl-C period (served_verdicts.sh's serve()), its period.json saying
    whether the lane's reserves since the open were the window's salt sessions alone (``reserves``) and holding the gate's
    close (``close``): the piece samples only after one that closed and held, and once the coin is flipped its RULE line reads
    the ledger (``pearlplus-period rule`` on the serve dir's pearlplus/, the lane and the coin record). Otherwise, the
    gate's close refused too, it says why, samples nothing and still writes its piece."""
    reserves = {"held": {"ok": True}, "unclosed": {"ok": True}, "refused": {"ok": False}, "unread": None}[case]
    period = {"name": "pp-r1", **({} if reserves is None else {"reserves": reserves}),
              **({} if case == "unclosed" else {"close": {"kind": "period-close"}})}
    run, sd, _tree = _piece(tmp_path, _epoch(), period=period, PEARLC_CHALLENGER_GATE="ssh gate",
                            PEARLC_CHALLENGER_LANE="vllm")
    assert run.returncode == 0, (run.stdout, run.stderr)
    names = [" ".join([Path(a[0]).name, *a[1:3 if a[1:2] == ["pearlplus-period"] else 2]]) for a in _steps(tmp_path)]
    given = ["served_passes.py volunteer-needed", "debit.py --cubin", "served_passes.py volunteer", "served_passes.py retain"]
    lines = run.stdout.splitlines()
    if case == "held":
        assert names == [*given, "served_passes.py verify", "served_verdicts.py coin",
                         "served_verdicts.py pearlplus-period rule", "served_passes.py lean",
                         "served_verdicts.py pearlplus"], lines
        rule = _steps(tmp_path)[names.index("served_verdicts.py pearlplus-period rule")]
        q = sd / "private" / "pearlplus"
        assert rule[3:] == [str(sd / "pearlplus"), "--gate", "ssh gate", "--lane", "vllm", "--coin",
                            str(q / "passes" / "coin.json")]
        assert "POUW RULE no other reserve on lane vllm between period pp-r1's close and the draw: HELD (toy)" in lines
    else:
        assert names == [*given, "served_verdicts.py pearlplus"], lines
        assert ("POUW draw: refused before it: the window serve's Pearl-C period didn't close with the lane's reserves "
                "since its open the window's salt sessions alone (served_verdicts.sh's PEARLPLUS-PERIOD line), so no draw "
                "follows it") in lines
        assert not (sd / "pearl" / "w7" / "coin.json").exists()


RECORD = '''import json, os, sys
from pathlib import Path
with open(os.environ["CALLS"], "a") as f:
    f.write(json.dumps({"tool": Path(sys.argv[0]).name, "argv": sys.argv[1:]}) + "\\n")
rest = sys.argv[1:]
at = lambda flag: Path(rest[rest.index(flag) + 1])
'''
PASSES = '''if rest[0] == "volunteer-needed":
    print(os.environ["NEEDED"])
    sys.exit(0 if os.environ["NEEDED"] else 2)
if rest[0] == "serve":
    at("--passes").mkdir(parents=True, exist_ok=True)
if rest[0] in ("serve", "volunteer", "verify"):
    at("--out").write_text("{}")
print(rest[0], "ok")
'''
SETUP = '''echo '{"tool": "pearl_c_sm90.sh", "argv": []}' >> "$CALLS"
mkdir -p "$RESEARCH_RUN_DIR/k/debit" && touch "$RESEARCH_RUN_DIR/k/pearl_c_sm90.cubin" "$RESEARCH_RUN_DIR/k/debit/debit_sm90.cubin"
echo '{"ok": true}' > "$RESEARCH_RUN_DIR/k/harness_gate.json"
PEARLC_KERNEL=pearl_c_sm90 PEARLC_CUBIN=$RESEARCH_RUN_DIR/k/pearl_c_sm90.cubin
'''
SERVED = ["volunteer-needed", "pearl_c_sm90.sh", "serve"]
CHECKED = ["verify", "lean"]


def _untied(tmp_path: Path, needed: str, volunteer: str | None = None) -> Any:
    """The untied piece on window 3 in a toy tree whose served_passes.py, served_verdicts.py, debit.py and setup each note
    their argv in ``calls``, ``volunteer-needed`` answering `needed`: a function running it under more environment, which
    returns its exit code, its output and the calls in order."""
    tree, d, p, calls = tmp_path / "tree", tmp_path / "serve", tmp_path / "piece", tmp_path / "calls"
    one = tree / "benchmarks" / "one_stage"
    one.mkdir(parents=True)
    shutil.copy(HERE.parents[2] / "benchmarks" / "one_stage" / "served_pearlplus.sh", one)
    (tree / ".venv" / "bin").mkdir(parents=True)
    (tree / ".venv" / "bin" / "python").symlink_to(sys.executable)
    (one / "served_verdicts.py").write_text(RECORD + 'print("POUW-DEV {}")\n')
    pouw = tree / "benchmarks" / "pouw"
    (pouw / "pearl_c_vllm").mkdir(parents=True)
    (pouw / "pearl_c_vllm" / "served_passes.py").write_text(RECORD + PASSES)
    (pouw / "pearl_c_sm90").mkdir()
    (pouw / "pearl_c_sm90" / "debit.py").write_text(RECORD + 'at("--out").write_text("{}")\n')
    (tree / "benchmarks" / "harness" / "setup").mkdir(parents=True)
    (tree / "benchmarks" / "harness" / "env.sh").write_text('VENV=$VERITY_BENCH_VENV\n')
    (tree / "benchmarks" / "harness" / "setup" / "pearl_c_sm90.sh").write_text(SETUP)
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "gpu-lease").write_text('#!/bin/sh\nwhile [ "$1" != -- ]; do shift; done\nshift\nexec "$@"\n')
    (bin_ / "gpu-lease").chmod(0o755)
    (bin_ / "nvidia-smi").write_text('#!/bin/sh\necho "GPU 0: toy (UUID: GPU-0)"\n')
    (bin_ / "nvidia-smi").chmod(0o755)
    d.mkdir()
    env = {"PATH": f"{bin_}:{os.defpath}", "HOME": str(tmp_path), "CALLS": str(calls), "NEEDED": needed,
           "VERITY_BENCH_VENV": str(tree / ".venv"), "PEARLPLUS_PASSES_RAM": "", "PEARLPLUS_UNRETAINED": "",
           **({} if volunteer is None else {"PEARLPLUS_VOLUNTEER": volunteer})}

    def run(**more: str) -> tuple[int, str, list[dict]]:
        calls.unlink(missing_ok=True)
        r = subprocess.run(["bash", "benchmarks/one_stage/served_pearlplus.sh", str(d), "3", str(p)], cwd=tree,
                           env={**env, **more}, capture_output=True, text=True, check=False)
        return r.returncode, r.stdout + r.stderr, [json.loads(x) for x in
                                                   (calls.read_text().splitlines() if calls.exists() else ())]
    return run


@pytest.mark.parametrize("needed, volunteer, steps", [
    ("yes reject", None, [*SERVED, "debit.py", "volunteer --device", "retain --volunteer", *CHECKED, "pearlplus --volunteer"]),
    ("yes reject", "cpu", [*SERVED, "volunteer", "retain --volunteer", *CHECKED, "pearlplus --volunteer"]),
    ("yes reject", "", ["volunteer-needed"]),
    ("no uncredited", None, [*SERVED, *CHECKED, "pearlplus"]),
    ("no uncredited", "", [*SERVED, *CHECKED, "pearlplus"]),
    ("no uncredited", "1", [*SERVED, "debit.py", "volunteer --device", "retain --volunteer", *CHECKED, "pearlplus --volunteer"]),
    ("", None, ["volunteer-needed"]),
])
def test_the_give_up_runs_exactly_when_volunteer_needed_says_yes_and_cant_be_skipped_then(tmp_path, needed, volunteer, steps):
    d, run = tmp_path / "serve", _untied(tmp_path, needed, volunteer)
    rc, said, got = run()
    flags = {"volunteer": "--device", "retain": "--volunteer", "pearlplus": "--volunteer"}
    names = []
    for c in got:
        name = c["argv"][0] if c["tool"] in ("served_passes.py", "served_verdicts.py") else c["tool"]
        names.append(f"{name} {flags[name]}" if flags.get(name) in c["argv"] else name)
    assert names == steps, said
    if steps == ["volunteer-needed"]:
        assert rc == 2 and not (d / "private" / "pearlplus" / "setup.log").exists(), said
        assert ("POUW volunteer: refused, so nothing is served" if needed else "volunteer-needed answered ''") in said
        return
    assert rc == 0, said
    if "verify" in steps and not any(s.startswith("volunteer ") or s == "volunteer" for s in steps):
        assert f"POUW volunteer: not run: Pearl+'s verdict credits a drawn row over its debit cap nothing (over-cap rule " \
               f"{needed.split()[1]})" in said
    if needed == "yes reject" and volunteer is None:
        rc, said, got = run(PEARLPLUS_OVER_CAP="reject")
        asked = {c["argv"][0]: c["argv"] for c in got if c["tool"] == "served_passes.py"}
        assert rc == 0 and asked["volunteer-needed"] == ["volunteer-needed", "--over-cap", "reject"], said
        assert asked["verify"][asked["verify"].index("--over-cap") + 1] == "reject"


def test_on_a_node_with_no_card_the_untied_piece_refuses_by_name_and_writes_no_answer(tmp_path):
    """A node that lists no card serves nothing, so the piece has no verdict: it refuses (pearlplus-no-card) before any
    step, its one sentence its POUW-REFUSED line and the piece's refused.txt, exits 2 and writes no development.json,
    which `piece_row` makes refused, never an answer of nulls."""
    sys.path.insert(0, str(HERE.parents[2] / "benchmarks" / "one_stage"))
    import served_verdicts as S
    from verity_e2e_demo import block_record as BR

    run, p = _untied(tmp_path, "no uncredited"), tmp_path / "piece"
    (tmp_path / "bin" / "nvidia-smi").write_text("#!/bin/sh\nexit 9\n")
    rc, said, got = run()
    sentence = (tmp_path / "piece" / "refused.txt").read_text().strip()
    assert (rc, got) == (2, []), said
    assert sentence.startswith("pearlplus-no-card: this node lists no card") and f"POUW-REFUSED {sentence}" in said
    assert not (p / S.DEVELOPMENT).exists()
    (pieces := tmp_path / "pieces").mkdir()
    p.rename(pieces / "pouw")
    assert S.piece_row("work", pieces) == {"verdict": BR.REFUSED, "owner": S.PIECES["work"][2], "reason": sentence}


def test_the_untied_pieces_build_reads_the_runs_machine_record_and_a_failed_build_refuses_by_name(tmp_path):
    """pearl_c_sm90.sh's target is machine.py's `arch`, read from $RESEARCH_RUN_DIR/inputs/$RESEARCH_MACHINE.toml, and the
    build runs with RESEARCH_RUN_DIR the piece's own dir: the piece carries the run's record there, so a run whose record
    `research run --on` copied builds and serves. Where the build fails anyway the untied piece serves nothing and refuses
    by the build's own reason (pearlplus-no-cubin), never as a serve that wrote no document."""
    run, p = _untied(tmp_path, "no uncredited"), tmp_path / "piece"
    setup = tmp_path / "tree" / "benchmarks" / "harness" / "setup" / "pearl_c_sm90.sh"
    setup.write_text('[ -f "$RESEARCH_RUN_DIR/inputs/$RESEARCH_MACHINE.toml" ] '
                     '|| { echo "target \'\': no architecture \'\'" >&2; return 4; }\n' + SETUP)
    (inputs := tmp_path / "run" / "inputs").mkdir(parents=True)
    (inputs / "toy.toml").write_text("[properties]\n")
    rc, said, got = run(RESEARCH_RUN_DIR=str(tmp_path / "run"), RESEARCH_MACHINE="toy")
    assert rc == 0 and [c["argv"][0] if c["argv"] else c["tool"] for c in got][:3] == [
        "volunteer-needed", "pearl_c_sm90.sh", "serve"], said
    (inputs / "toy.toml").unlink()
    rc, said, got = run(RESEARCH_RUN_DIR=str(tmp_path / "run"), RESEARCH_MACHINE="toy")
    sentence = (p / "refused.txt").read_text().strip()
    assert (rc, [c["argv"][0] for c in got]) == (2, ["volunteer-needed"]), said
    assert sentence.startswith("pearlplus-no-cubin: pearl_c_sm90.sh failed (exit 4)") and "no architecture" in sentence
    assert f"POUW-REFUSED {sentence}" in said


def test_pearl_c_sm90_refuses_an_architecture_machine_py_did_not_name_and_never_passes_it_on(tmp_path):
    """pearl_c_sm90.sh, the piece's step 1, stops at exit 4 with the fix in its message where machine.py's `arch` names no
    architecture (its machine description absent), and never asks `gencode` for an empty target."""
    tree, calls = tmp_path / "tree", tmp_path / "calls"
    (tree / "benchmarks" / "kernels").mkdir(parents=True)
    (tree / "benchmarks" / "harness" / "setup").mkdir(parents=True)
    shutil.copy(HERE.parents[2] / "benchmarks" / "harness" / "setup" / "pearl_c_sm90.sh", tree / "benchmarks" / "harness" / "setup")
    (tree / "benchmarks" / "kernels" / "machine.py").write_text(
        'import os, sys\nopen(os.environ["CALLS"], "a").write(sys.argv[1] + "\\n")\n'
        'print("/cuda") if sys.argv[1] == "toolkit" else sys.exit("machine: inputs/toy.toml is not there")\n')
    r = subprocess.run(["bash", "-c", ". benchmarks/harness/setup/pearl_c_sm90.sh"], cwd=tree, capture_output=True, text=True,
                       env={"PATH": os.defpath, "CALLS": str(calls), "RESEARCH_RUN_DIR": str(tmp_path / "q"),
                            "RESEARCH_MACHINE": "toy"}, check=False)
    assert r.returncode == 4 and calls.read_text().split() == ["toolkit", "arch"], r.stderr
    assert f"machine.py arch named no architecture, so no cubin is built: it reads the run's machine description at " \
           f"{tmp_path / 'q'}/inputs/toy.toml" in r.stderr


def test_under_the_gate_the_untied_piece_holds_its_pearl_c_period_open_from_its_serve_to_right_before_its_sample(tmp_path):
    """Under the gate the untied piece opens the lane's Pearl-C period right before its retained serve and closes it only
    right before verify samples, after the give-up and retain: while it is open its cap of 1, which the serve's salt session
    fills, refuses every other reserve of the lane, so none comes between its close and the sample, which a compose takes
    only as the lane's first reserve after the close."""
    rc, said, got = _untied(tmp_path, "yes reject")(PEARLC_CHALLENGER_GATE="ssh gate")
    names = [" ".join(c["argv"][:2]) if c["argv"][:1] == ["pearlplus-period"] else
             c["argv"][0] if c["tool"] in ("served_passes.py", "served_verdicts.py") else c["tool"] for c in got]
    assert rc == 0 and names == ["volunteer-needed", "pearl_c_sm90.sh", "pearlplus-period open", "serve", "debit.py",
                                 "volunteer", "retain", "pearlplus-period close", *CHECKED, "pearlplus"], said
