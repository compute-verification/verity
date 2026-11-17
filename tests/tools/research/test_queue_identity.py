"""The identity `research queue cross` gives a crossing's commits (`IDENTITY`) is the one identity the agent guard's pre-push
hook lets through beside a checkout's own (`OTHERS` in tools/agent-guard/pre_push.py). `cross_push` pushes a crossing with
the caller's own git, so where an agent runs it that hook judges the push by who wrote the commits it adds. The guard is
standard library only and imports nothing of research, so it holds the identity as text: this test holds that text to
`IDENTITY`, read from the hook's source without running it."""
from __future__ import annotations

import ast
from pathlib import Path

from research.queue import IDENTITY

HOOK = "tools/agent-guard/pre_push.py"


def test_the_pre_push_hooks_one_allowance_is_the_identity_a_crossing_commits_under():
    source = (Path(__file__).resolve().parents[3] / HOOK).read_text()
    [others] = [ast.literal_eval(node.value) for node in ast.parse(source).body
                if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "OTHERS"]
    crossing = {f"{IDENTITY[f'GIT_{role}_NAME']} <{IDENTITY[f'GIT_{role}_EMAIL']}>" for role in ("AUTHOR", "COMMITTER")}
    assert set(others) == crossing, (
        f"{HOOK}'s `OTHERS` lets through {sorted(others)}, and `research queue cross` commits as {sorted(crossing)} "
        "(`IDENTITY` in tools/research/src/research/queue.py): an agent's `research queue cross --push` would be refused by "
        "the hook, or a name no crossing carries would pass it. Change `OTHERS`' one entry and `IDENTITY` together.")
