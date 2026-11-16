"""`served_verdicts.sh`'s advice call (`advice_of`, `verity-vllm advice`) under DAY: the call registers the compiler in the
day's registration log (`serve_day.register_compiler`), which a log with site entries checks under DAY_PARAMS' auditor, so
it passes ``--day-params`` whenever the run declared a day with them, and nothing new otherwise. The call comes before
every coin that samples what the compiler builds: in ``tie`` and ``pearl_tie`` before window-commit, whose R the window's
sample coin waits for, and in the main flow before step 3's sampled proofs and step 6's window commit and V3c's audit. A
PoUW H1 window's own coins, drawn at its close while the endpoint serves, sample H1's blocks, not the circuit the compiler
builds, and without the tie they come before the call."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parents[2] / "benchmarks" / "one_stage" / "served_verdicts.sh"


def _function(text: str, name: str) -> str:
    """The shell function ``name``'s definition, its first line to its closing brace."""
    start = text.index(f"\n{name}() {{") + 1
    return text[start:text.index("\n}\n", start) + 3]


@pytest.mark.parametrize(("day", "want"), [({}, []), ({"DAY_PARAMS": "params.json"}, []), ({"DAY": "0"}, []),
                                           ({"DAY": "0", "DAY_PARAMS": "params.json"}, ["--day-params", "params.json"])])
def test_the_advice_call_passes_the_days_parameters_when_the_run_declared_a_day_with_them(tmp_path, day, want):
    tree, d, args = tmp_path / "tree", tmp_path / "serve", tmp_path / "args"
    (tree / "adapters" / "vllm").mkdir(parents=True)
    d.mkdir()
    body = (f'set -uo pipefail\nvv() {{ printf "%s\\n" "$@" > "$ARGS"; }}\n{_function(SCRIPT.read_text(), "advice_of")}'
            'advice_of 4 "--compiler-name n --compiler-digest d"\n')
    env = {"PATH": os.environ["PATH"], "TREE": str(tree), "D": str(d), "CAP": "4096", "WREG": str(tmp_path / "none"),
           "ARGS": str(args), **day}
    run = subprocess.run(["bash", "-c", body], env=env, capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr
    assert args.read_text().splitlines() == ["advice", str(d), "--window", "4", "--cap", "4096", "--compiler-name", "n",
                                             "--compiler-digest", "d", *want, "--out", f"{d}/advice/.w4"]


def test_the_advice_call_comes_before_each_coin_that_samples_what_the_compiler_builds():
    text = SCRIPT.read_text()
    for name, call in (("tie", 'advice_of "$K" "$ca"'), ("pearl_tie", 'advice_of "$K" "$ca"')):
        fn = _function(text, name)
        assert fn.index(call) < fn.index("window_commit "), name
    main = text[text.index("\nCA=$(compiler_args)\n"):]
    first = main.index('advice_of "$k" "$CA"')
    for later in ("bash benchmarks/prove_path/run.sh", "v3c_commit", "v3c_proved"):
        assert first < main.index(later), later
    assert text.count("bash benchmarks/prove_path/run.sh") == 1
