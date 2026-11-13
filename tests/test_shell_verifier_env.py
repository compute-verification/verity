"""Every tracked shell script that starts the Lean verifier's `verify` itself starts it under the verifier's allocator
setting, as the Python callers do (`proofs.verifier_env`): its command carries `$(verifier_env)`, the words
`verifier_env.sh` prints, and the script sources that file or a script that does. A figure taken through a shell job is
then under the same setting as one taken through Python.

A command is found by the word `verify` followed by one of the verifier's options or by a variable that holds them, on
one logical line (a line with its `\\` continuations), outside comment lines. `ALLOWED` names the scripts that still
start the verifier under the allocator's default; it only shrinks.

The finder reads text, so another tool's `verify` followed by a variable reads as the verifier's. `OTHER_TOOLS` names
each such command with its script and its tool, and only that is let through: the command word for word, in that script,
while the script sets the variable the command starts with once, and to that tool. Any other spelling is read as the
verifier's, so a script can't leave the rule by renaming a variable.
"""
from __future__ import annotations

import os
import re
import subprocess

import pytest
from test_repository import ROOT, _tracked

from verity.protocols.spot.verifier.one_stage import proofs

HELPER = "verity/protocols/spot/verifier/one_stage/verifier_env.sh"
USE = "$(verifier_env)"
FIX = (f"start it as `{USE} VERIFIER verify ...` in a script that sources {HELPER}; another tool's `verify` goes in "
       "OTHER_TOOLS with its script and what the variable it starts with is set to")
#: Flock's pod jobs, which start the verifier under the allocator's default
ALLOWED = frozenset({
    "verity/protocols/flock/pod/76-read-gadget.sh",
    "verity/protocols/flock/pod/85-rec-reprice.sh",
    "verity/protocols/flock/pod/86-zk-cpu-steps.sh",
    "verity/protocols/flock/pod/89-served-registered.sh",
    "verity/protocols/flock/pod/90-tmpl-unit.sh",
    "verity/protocols/flock/pod/91-plain-hidden-tables.sh",
    "verity/protocols/flock/pod/91-plain-hidden.sh",
    "verity/protocols/flock/pod/92-zk-sessions.sh",
    "verity/protocols/flock/pod/93-zk-tree-from.sh",
})
#: (script, another tool's `verify` as the script writes it): what the script sets the command's first word to. An entry
#: whose command its script doesn't hold lets nothing through.
OTHER_TOOLS = {
    ("benchmarks/prove_path/run.sh",
     '$WT verify "${common[@]}" --registration $O/$name/registration.json --out $O/$name/verifier'
     " | tee $O/$name/verify.json || return 1"): '"$PY -m verity_vllm.pipeline.cli weights-tie"',
}
_OPTIONS = "archive|circuit|coins|partition|plain-tables|program|public|registered|session|statement|stratified|zk"
_CALL = re.compile(rf'(?<![-\w/.])verify\s+(?:--(?:{_OPTIONS})\b|"?\$\{{?\w)')
_SOURCED = re.compile(r'(?:^|[;&|(]\s*|\s)(?:\.|source)\s+"?(?:\$\{?\w+\}?/)?([\w./-]+\.sh)\b')


def other_tool(path: str, text: str, command: str) -> bool:
    """Whether `command`, a logical line of the script `path`, is another tool's `verify` as OTHER_TOOLS names it."""
    tool = OTHER_TOOLS.get((path, command.strip()))
    first = re.match(r"\$(\w+) ", command.strip())
    return bool(tool and first) and re.findall(rf"(?<![\w$]){first[1]}\+?=(.*)", text) == [tool]


def calls(path: str, text: str) -> list[tuple[int, str]]:
    """Each logical line of the script `path` that starts the verifier's `verify`, with the number of its first line."""
    found, start, held = [], 0, ""
    for number, line in enumerate(text.split("\n"), 1):
        if line.lstrip().startswith("#"):
            continue
        if not held:
            start = number
        if line.endswith("\\"):
            held += line[:-1] + " "
            continue
        if _CALL.search(held + line) and not other_tool(path, text, held + line):
            found.append((start, held + line))
        held = ""
    return found


def scripts() -> dict[str, str]:
    return {p: (ROOT / p).read_text(errors="replace") for p in _tracked() if p.endswith(".sh") and not p.startswith("archive/")}


def unset(path: str, text: str) -> list[int]:
    return [number for number, command in calls(path, text) if USE not in command]


_JOB = "benchmarks/job.sh"
(_TIE, _TIE_VERIFY), _TIE_TOOL = next(iter(OTHER_TOOLS.items()))
_TIE_TEXT = f"  WT={_TIE_TOOL}\n    {_TIE_VERIFY}"


@pytest.mark.parametrize("path, text, found, takes", [
    (_JOB, '$TIME "$O/t" bash -c "$(vlimit $gb) exec $LEAN verify --zk --circuit $D/circuit.txt"', True, False),
    (_JOB, '$TIME "$O/t" bash -c "$(vlimit $gb) exec $(verifier_env) $LEAN verify --zk --circuit $D/circuit.txt"', True, True),
    (_JOB, '/usr/bin/time -v "$LEAN" verify "${coins[@]}" --statement s', True, False),
    (_JOB, '$(verifier_env) $LV verify "${V[@]}" --circuit c', True, True),
    (_JOB, 'V=(verify --statement s --coins os)', True, False),
    (_JOB, '  # $LEAN verify --zk --circuit c', False, False),
    (_JOB, '"$PY" "$D/passes.py" verify --out "$O/passes" --fast', False, False),
    (_JOB, 'ZK verify --lean "$LEAN" 2>&1 | tee -a "$O/zk.log"', False, False),
    (_JOB, 'gpg --verify $sig $file', False, False),
    (_JOB, 'echo "plain: $(verify "$(one plain)" plain)"', False, False),
    pytest.param(_TIE, _TIE_TEXT, False, False, id="another tool's, as OTHER_TOOLS names it"),
    pytest.param(_JOB, _TIE_TEXT, True, False, id="the same lines in another script"),
    pytest.param(_TIE, _TIE_TEXT.replace("WT", "TIE"), True, False, id="its variable renamed"),
    pytest.param(_TIE, _TIE_TEXT.replace("--out", "--zk --out"), True, False, id="its command edited"),
    pytest.param(_TIE, f"{_TIE_TEXT}\n  WT=$LEAN", True, False, id="its variable set a second time"),
    pytest.param(_TIE, f"  WT=$LEAN\n    {_TIE_VERIFY}", True, False, id="its variable set to something else"),
    pytest.param(_TIE, _TIE_VERIFY, True, False, id="its variable not set in the script"),
])
def test_the_finder_reads_each_form(path: str, text: str, found: bool, takes: bool) -> None:
    assert [USE in command for _, command in calls(path, text)] == ([takes] if found else [])


def test_a_command_is_read_with_its_continuation_lines() -> None:
    text = 'x=1\n$(verifier_env) \\\n  "$LEAN" verify --zk \\\n  --circuit c\n"$LEAN" verify --zk \\\n  --circuit c\n'
    assert [(number, USE in command) for number, command in calls(_JOB, text)] == [(2, True), (5, False)]


def test_every_shell_call_of_the_verifier_takes_its_allocator_setting() -> None:
    found = [f"{path}:{number}" for path, text in scripts().items() if path not in ALLOWED for number in unset(path, text)]
    assert not found, f"the Lean verifier started under the allocator's default; {FIX}:\n" + "\n".join(found)


def test_a_script_that_takes_the_setting_sources_the_helper() -> None:
    """`$(verifier_env)` in a script that never defines it expands to nothing, and the verifier runs under the default."""
    all_ = scripts()
    missing = []
    for path, text in all_.items():
        if USE not in text or path == HELPER:
            continue
        sourced = [s for s in _SOURCED.findall(text) if s in all_]
        if not any(HELPER in t for t in (text, *(all_[s] for s in sourced))):
            missing.append(path)
    assert not missing, f"scripts that use {USE} and source neither {HELPER} nor a script that does:\n" + "\n".join(missing)


@pytest.mark.parametrize("own", [None, "0", ""], ids=["unset", "the caller's own", "the caller's empty"])
def test_the_shell_helper_prints_the_python_callers_setting(own: str | None) -> None:
    environ = {k: v for k, v in os.environ.items() if k not in proofs.VERIFIER_ENV}
    if own is not None:
        environ.update(dict.fromkeys(proofs.VERIFIER_ENV, own))
    out = subprocess.run(["bash", "-uc", '. "$0" && verifier_env', str(ROOT / HELPER)], env=environ, capture_output=True,
                         text=True, check=True).stdout
    want = proofs.verifier_env(environ)
    assert out == "env " + " ".join(f"{name}={want[name]}" for name in proofs.VERIFIER_ENV) + "\n"


def test_allowed_names_only_scripts_that_keep_the_default() -> None:
    all_ = scripts()
    stale = sorted(path for path in ALLOWED if path not in all_ or not unset(path, all_[path]))
    assert not stale, "ALLOWED names scripts that no longer start the verifier under the default; drop them:\n" + "\n".join(stale)
