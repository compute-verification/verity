"""The adapter's sites in `workload_messages.toml` go through the messages' declarations: within `WINDOW` lines of a
line its ``find`` matches, a call of `verity_vllm.messages` (``M.read``, ``M.write``, ``M.lines``, ``M.line``,
``M.append``, ``M.body``) names the site's type. `LEFT` holds the adapter's sites that don't, each with why, and only
shrinks: a site missing from it that doesn't fails, and a listed site that now does warns, to be dropped."""
from __future__ import annotations

import re
import tomllib
import warnings
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = "adapters/vllm/"
WINDOW = 15
CALL = re.compile(r"\bM\.(?:read|write|lines|line|append|body)\(")

FLAG = "a flag whose value names a file, a directory or a command that another site reads; it carries no body"
IN_PROCESS = "a Python call inside the adapter's process: no body crosses between two processes"

#: The adapter's sites that don't go through the declarations, ``"TYPE SIDE FILE"`` (a second form keeps its type), with
#: why; one entry a site.
LEFT: list[tuple[str, str]] = [
    ("Advice writer adapters/vllm/verity_vllm/engine/step_trace.py",
     "the engine's trace writer: suites outside the adapter list the files this module imports, so it stays off the core"),
    ("Advice writer adapters/vllm/verity_vllm/pipeline/serve_commit.py",
     "installs step_trace's writer, which writes the trace"),
    ("CanonicalServing reader adapters/vllm/verity_vllm/settings.py", "environment variables: no body"),
    ("Circuit reader adapters/vllm/verity_vllm/pipeline/serve_openai.py", FLAG),
    ("Circuit writer adapters/vllm/verity_vllm/program/predict/aot.py", "the ahead-of-time compiler's Build writer"),
    ("Circuit writer adapters/vllm/verity_vllm/records/predicted_row.py", "declares the record, which aot.py writes"),
    ("CommitmentScheme reader adapters/vllm/verity_vllm/pipeline/serve_openai.py", "flags whose values are the scheme's "
     "settings, read by the parser"),
    ("H1RunCommitment reader adapters/vllm/verity_vllm/pipeline/serve_h1.py", FLAG),
    ("H1RunCommitment writer adapters/vllm/verity_vllm/pipeline/serve_tie.py", "runs the prover's command; the record it "
     "writes is read as H1Registration"),
    ("H1State reader adapters/vllm/verity_vllm/pipeline/serve_h1.py", FLAG),
    ("PoUSSetup reader adapters/vllm/verity_vllm/protocols/__init__.py", IN_PROCESS),
    ("PoUSSetup reader adapters/vllm/verity_vllm/protocols/pous.py", IN_PROCESS),
    ("PoUSStore writer adapters/vllm/verity_vllm/protocols/pous.py", IN_PROCESS),
    ("PoUWScheme reader adapters/vllm/verity_vllm/pipeline/serve_openai.py", FLAG),
    ("PoUWValues writer adapters/vllm/verity_vllm/pipeline/serve_pearl.py", IN_PROCESS),
    ("PoUWValues writer adapters/vllm/verity_vllm/protocols/pouw/__init__.py", IN_PROCESS),
    ("RegisteredCompiler reader adapters/vllm/verity_vllm/pipeline/advice.py", "flags whose values are the compiler's "
     "name and digest, which the advice binds"),
    ("RegisteredWeights writer adapters/vllm/verity_vllm/pipeline/serve_seal.py", "calls match_capture's writer"),
    ("ReportedAdvice writer adapters/vllm/verity_vllm/pipeline/advice_code/window_code.py", IN_PROCESS),
    ("ReportedAdvice writer adapters/vllm/verity_vllm/program/predict/scan_state.py", IN_PROCESS),
    ("ReportedAdvice writer adapters/vllm/verity_vllm/program/predict/window_rows.py", IN_PROCESS),
    ("RunTotals reader adapters/vllm/tools/served_run/serve.sh", "a shell script that copies the file; it reads no body"),
    ("Values writer adapters/vllm/verity_vllm/pipeline/serving_rows.py", IN_PROCESS),
]


def through_declaration(type_: str, site: dict[str, str]) -> bool:
    """Whether a call of `verity_vllm.messages` naming ``type_`` is within `WINDOW` lines of a line ``site`` finds."""
    lines = (ROOT / site["file"]).read_text().splitlines()
    find, named = re.compile(site["find"]), re.compile(rf"""["']{re.escape(type_)}["']""")
    for i in (i for i, line in enumerate(lines) if find.search(line)):
        near = lines[max(0, i - WINDOW): i + WINDOW + 1]
        if any(CALL.search(line) and named.search(line) for line in near):
            return True
    return False


def adapter_sites() -> list[tuple[str, bool]]:
    """Each adapter site's ``"TYPE SIDE FILE"`` and whether it goes through the declarations."""
    out = []
    for entry in tomllib.loads((ROOT / "tests/workload_messages.toml").read_text())["message"]:
        for side in ("writers", "readers"):
            for site in entry[side]:
                if site["file"].startswith(ADAPTER):
                    out.append((f"{entry['type']} {side[:-1]} {site['file']}", through_declaration(entry["type"], site)))
    return out


def test_no_new_adapter_site_bypasses_the_declarations():
    left = Counter(key for key, _ in LEFT)
    bypassing = Counter(key for key, through in adapter_sites() if not through)
    if new := bypassing - left:
        raise AssertionError(
            f"these adapter sites read or write their message without `verity_vllm.messages` "
            f"(`M.read(path, TYPE)`, `M.write(path, TYPE, data)`, `M.lines`, `M.line`, `M.append`, `M.body`) within "
            f"{WINDOW} lines of what their `find` in tests/workload_messages.toml matches: {sorted(new.elements())}. "
            "Send them through the declaration, or point `find` at the line that does")
    if stale := left - bypassing:
        warnings.warn(f"LEFT lists adapter sites that now go through the declarations: drop {sorted(stale.elements())}",
                      stacklevel=1)


def test_left_is_well_formed():
    keys = [key for key, _ in LEFT]
    assert keys == sorted(keys), "keep LEFT sorted"
    assert all(why.strip() for _, why in LEFT), "each LEFT entry says why"
    sites = Counter(key for key, _ in adapter_sites())
    assert not (missing := Counter(keys) - sites), f"LEFT names sites tests/workload_messages.toml doesn't: {missing}"
