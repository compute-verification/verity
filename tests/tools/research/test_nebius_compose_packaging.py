"""The compose gate's packaging (`pods/nebius/compose/`) against the tree it pins: `pin.sh` requires files the tree has and
prebuilds every Lean executable the compose builds at compose time, at its package, and `vy-compose-gate` execs the
pinned tree's `host.py` in the form its `gate` parses and the sudoers line (`research ... vy-compose-gate --ssh *`)
admits."""
from __future__ import annotations

import importlib.util
import io
import json
import re
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PACKAGING = REPO / "tools/research/src/research/pods/nebius/compose"
HOST_PY = REPO / "verity/protocols/resource_accounting/memory/poub/attester/live_verifier/host.py"
ONE_STAGE = REPO / "benchmarks/one_stage"
_spec = importlib.util.spec_from_file_location("live_verifier_host", HOST_PY)
H = sys.modules[_spec.name] = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)

PIN = (PACKAGING / "pin.sh").read_text()
GATE = (PACKAGING / "vy-compose-gate").read_text()


def builds() -> set[tuple[str, str]]:
    block = re.search(r"^builds=\((.*?)\)$", PIN, re.M | re.S).group(1)
    return set(re.findall(r'"([a-z0-9-]+) ([^" ]+)"', block))


def compose_builds() -> set[tuple[str, str]]:
    """The Lean executables the compose builds: served_composite.sh's products and served_cases.py's cases."""
    composite = (ONE_STAGE / "served_composite.sh").read_text()
    package = re.search(r"^package\(\) \{.*?^\}", composite, re.M | re.S).group(0)
    found = {(x, p) for xs, p in re.findall(r"^\s*([a-z0-9| -]+)\) echo (\S+) ;;", package, re.M)
             for x in xs.replace(" ", "").split("|")}
    cases = (ONE_STAGE / "served_cases.py").read_text()
    lean = re.compile(r'\(ROOT / "([^"]+)", "([^"]+)"\)')
    found.add(tuple(reversed(lean.search(re.search(r"^SAMPLED_LEAN = .*$", cases, re.M).group(0)).groups())))
    found.add(tuple(reversed(lean.search((ONE_STAGE / "inference_only_lean.py").read_text()).groups())))
    spot = (REPO / "verity/protocols/proofs/spot/verifier/one_stage/verifier.py").read_text()
    found.add((re.search(r'^V1_LEAN = \(VF\.PACKAGE, "([^"]+)"\)$', cases, re.M).group(1),
               re.search(r'^PACKAGE = ROOT / "([^"]+)"$', spot, re.M).group(1)))
    return found


def test_the_pin_prebuilds_every_lean_executable_the_compose_builds():
    """A compose that builds Lean spends its minutes on it, and a package the pin left out isn't held to the pin's
    one toolchain: the pin builds each executable the compose names, which its package's lakefile declares."""
    pinned, needed = builds(), compose_builds()
    assert needed <= pinned, f"pin.sh's builds lack {sorted(needed - pinned)}: add each to its builds list"
    assert H.SAMPLED_EXE == Path(dict(pinned)["vstar-recursive"]) / ".lake/build/bin/vstar-recursive"
    toolchains = set()
    for exe, pkg in sorted(pinned):
        lakefile = tomllib.loads((REPO / pkg / "lakefile.toml").read_text())
        assert exe in [e["name"] for e in lakefile.get("lean_exe", [])], f"{pkg}/lakefile.toml declares no {exe}"
        toolchains.add((REPO / pkg / "lean-toolchain").read_text().strip())
    assert len(toolchains) == 1, f"the compose's packages name two toolchains: {sorted(toolchains)}"


def test_the_pin_requires_files_the_tree_has_and_logs_each_pin_before_current_names_it():
    code = "\n".join(ln for ln in PIN.splitlines() if not ln.lstrip().startswith("#"))
    required = re.search(r"for f in (.*?); do", code, re.S).group(1).replace("\\\n", " ").split()
    assert required and [f for f in required if not (REPO / f).is_file()] == []
    rel = HOST_PY.relative_to(REPO).as_posix()
    assert f'"$T/{rel}" pinned "$c" "$br"' in code
    assert code.index('= --prepare ]') < code.index(' pinned "$c"')
    assert code.index(' pinned "$c"') < code.index('ln -sfn "$c" "$O/current.tmp"')
    assert "curl" not in code and not re.search(r"\|\s*(ba)?sh\b", code)


def test_the_gate_execs_the_pinned_trees_host_py_in_the_form_its_gate_parses():
    """The wrapper re-execs itself through sudo as `--ssh WORDS --key LANE`, which the sudoers line admits and the
    compose host's gate takes; it answers a verb but compose with its refusal, never the forced command's."""
    rel = HOST_PY.relative_to(REPO).as_posix()
    assert f"exec /usr/bin/python3 -I /opt/vy-compose/current/{rel} gate \"$@\"\n" in GATE
    assert 'exec sudo -n /usr/local/sbin/vy-compose-gate --ssh "${SSH_ORIGINAL_COMMAND:-}" --key "${1:--}"\n' in GATE
    assert H.COMPOSE_HOST.pins == Path("/opt/vy-compose/pins.jsonl") and H.COMPOSE_HOST.compose
    sink = io.BytesIO()
    assert H.gate(["--ssh", "status", "--key", "vllm"], H.COMPOSE_HOST, sink, busy=list) == 3
    assert "serves compose alone" in json.loads(sink.getvalue())["refused"]
