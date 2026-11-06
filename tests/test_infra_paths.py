"""Verity keeps none of the team's infra (AGENTS.md "Map and commands"): a new infra component starts in the infra
repository, and ``INFRA_PATHS`` lists each infra path still here with its destination.  It only shrinks.  A new entry under
``tools/`` or ``infra/`` fails unless ``VERITY_TOOLS`` names it as Verity's own, a new skill fails unless ``VERITY_SKILLS``
does, and a listed path that is gone fails until its entry is removed, so the list stays exact.  What a listed path holds
isn't counted."""

from __future__ import annotations

import subprocess
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: What becomes of a listed path: it moves to the infra repository (``infra``), it is deleted with the coarse check once
#: drive reads the graph (``coarse-check``), it stays here as landing policy infra's drive reads (``data``), or it stays as
#: a stub handing off to infra's copy while something outside the repository runs it by this path (``shim``).
DESTINATIONS = {"infra", "coarse-check", "data", "shim"}

#: The entries under ``tools/`` that are Verity's own, each with what it is.
VERITY_TOOLS = {
    "tools/verity": "the `verity` command: the check's suites and guard, circuit-check and the Lean check",
    "tools/config": "`verity_config`, the strict configuration and canonical identities verity, benchmarks and "
                    "integrations read",
}

#: The skills under ``.agents/skills/`` that are Verity's own: how to work on its code, not how the team works.
VERITY_SKILLS = {"testing", "lean-proofs", "kernel-engineering", "smell-check"}

#: Each infra path still in Verity, with its destination.  It only shrinks: remove an entry with its path, and a path that
#: moves inside Verity takes its entry along.
INFRA_PATHS = {
    # the team's machines and services
    "infra/cluster": "infra",
    "infra/control": "infra",
    "infra/cursor": "infra",
    "infra/nebius": "infra",
    "infra/ops": "infra",
    "infra/slack-events": "infra",
    "tools/research": "infra",
    # the guard: Cursor's project hooks and the environment's start run it by its path here
    ".cursor/hooks.json": "shim",
    "tools/agent-guard": "shim",
    "tests/test_agent_guard_boot.py": "infra",
    "tests/test_agent_guard_hook.py": "infra",
    "tests/test_agent_guard_oom.py": "infra",
    "tests/test_shell_hook.py": "infra",
    "tests/test_task_hook.py": "infra",
    # the environment's install command runs it by its path here
    "tools/lean": "shim",
    # the layout move, which drive regenerates and certifies
    "tools/move": "infra",
    "tools/verity/check/anchors.py": "infra",
    "tools/verity/check/move_check.py": "infra",
    "tools/verity/check/tests/test_anchors.py": "infra",
    "tools/verity/check/tests/test_move_check.py": "infra",
    # drive's landing policy, and tests that check it, or the machines' uv pin, with infra's code
    "tools/verity/check/drive.toml": "data",
    "tools/verity/check/queue.toml": "data",
    "tools/verity/check/tests/test_queue_rules.py": "infra",
    "tests/test_drive_toml.py": "infra",
    "tests/test_merge_drivers.py": "infra",
    "tests/test_uv_pin.py": "infra",
    # the coarse check
    "tools/verity/check/check.py": "coarse-check",  # its execution half
    "tools/verity/check/ci.toml": "coarse-check",
    "tools/verity/check/declaration.py": "coarse-check",
    "tools/verity/check/lean_slot.py": "coarse-check",
    "tools/verity/check/live.py": "coarse-check",
    "tools/verity/check/plan.py": "coarse-check",
    "tools/verity/check/pod_setup.sh": "coarse-check",
    "tools/verity/check/post_train.py": "coarse-check",
    "tools/verity/check/preflight.py": "coarse-check",
    "tools/verity/check/shards.py": "coarse-check",
    "tools/verity/check/slot.py": "coarse-check",
    "tools/verity/check/spans.py": "coarse-check",
    "tools/verity/check/tool.py": "coarse-check",
    "tools/verity/check/train.sh": "coarse-check",
    "tools/verity/check/verdicts.py": "coarse-check",
    "tools/verity/check/tests/test_check.py": "coarse-check",
    "tools/verity/check/tests/test_declaration.py": "coarse-check",
    "tools/verity/check/tests/test_deferred_run.py": "coarse-check",
    "tools/verity/check/tests/test_env.py": "coarse-check",
    "tools/verity/check/tests/test_lean_slot.py": "coarse-check",
    "tools/verity/check/tests/test_live.py": "coarse-check",
    "tools/verity/check/tests/test_placement.py": "coarse-check",
    "tools/verity/check/tests/test_plan.py": "coarse-check",
    "tools/verity/check/tests/test_post_train.py": "coarse-check",
    "tools/verity/check/tests/test_preflight.py": "coarse-check",
    "tools/verity/check/tests/test_record_clone.py": "coarse-check",
    "tools/verity/check/tests/test_record_pool.py": "coarse-check",
    "tools/verity/check/tests/test_shard_plan.py": "coarse-check",
    "tools/verity/check/tests/test_shards.py": "coarse-check",
    "tools/verity/check/tests/test_shipped_checkout.py": "coarse-check",
    "tools/verity/check/tests/test_slot.py": "coarse-check",
    "tools/verity/check/tests/test_spans.py": "coarse-check",
    "tools/verity/check/tests/test_verdicts.py": "coarse-check",
    "tools/verity/circuit_check/src/circuit_check/cache.py": "coarse-check",
    "tools/verity/circuit_check/src/circuit_check/units.py": "coarse-check",
    "tools/verity/circuit_check/tests/test_cache.py": "coarse-check",
    "tools/verity/circuit_check/tests/test_cache_redteam.py": "coarse-check",
    "tools/verity/circuit_check/tests/test_cache_redteam_1176.py": "coarse-check",
    "tools/verity/circuit_check/tests/test_units.py": "coarse-check",
    # the team's skills
    ".agents/skills/asking-daniel": "infra",
    ".agents/skills/authoring-skills": "infra",
    ".agents/skills/coordinating": "infra",
    ".agents/skills/friction": "infra",
    ".agents/skills/process-design": "infra",
    ".agents/skills/team": "infra",
    ".agents/skills/using-slack": "infra",
    ".agents/skills/writing-runs": "infra",
}

FIX = ("Infra starts in the infra repository (compute-verification/infra), never here, and INFRA_PATHS in "
       "tests/test_infra_paths.py only shrinks.")


@cache
def _tracked() -> tuple[str, ...]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True).stdout
    return tuple(p for p in out.decode().split("\0") if p)


def test_tools_and_infra_hold_only_verity_tools_and_infra_paths() -> None:
    entries = {"/".join(p.split("/")[:2]) for p in _tracked() if p.split("/")[0] in ("tools", "infra")}
    new = sorted(entries - set(VERITY_TOOLS) - set(INFRA_PATHS))
    assert not new, (f"new under tools/ or infra/: {new}. {FIX} Move it to the infra repository, or, if it is a tool of "
                     "Verity's own, add it to VERITY_TOOLS with what it is.")


def test_every_skill_is_a_verity_skill_or_an_infra_path() -> None:
    skills = {p.split("/")[2] for p in _tracked() if p.startswith(".agents/skills/") and p.count("/") > 2}
    new = sorted(s for s in skills - VERITY_SKILLS if f".agents/skills/{s}" not in INFRA_PATHS)
    assert not new, (f"new skills: {new}. {FIX} The team's skills are written there; a skill for working on Verity's code "
                     "goes in VERITY_SKILLS.")


def test_every_listed_path_exists() -> None:
    present = {"/".join(parts[:i]) for p in _tracked() for parts in [p.split("/")] for i in range(1, len(parts) + 1)}
    listed = [*INFRA_PATHS, *VERITY_TOOLS, *(f".agents/skills/{s}" for s in VERITY_SKILLS)]
    gone = sorted(p for p in listed if p not in present)
    assert not gone, f"listed in tests/test_infra_paths.py but gone from the tree; remove their entries: {gone}"


def test_every_destination_is_known() -> None:
    odd = {p: d for p, d in INFRA_PATHS.items() if d not in DESTINATIONS}
    assert not odd, f"INFRA_PATHS destinations outside DESTINATIONS {sorted(DESTINATIONS)}: {odd}"
