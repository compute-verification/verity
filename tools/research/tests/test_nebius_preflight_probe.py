"""The hourly preflight probe (`pods/nebius/preflight_probe.py`): what a probe run's output says, and one alert per machine whose
preflight fails, cleared once it passes."""
from __future__ import annotations

import importlib.util
from pathlib import Path

NEBIUS = Path(__file__).resolve().parents[1] / "src/research/pods/nebius"
_spec = importlib.util.spec_from_file_location("nebius_preflight_probe", NEBIUS / "preflight_probe.py")
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

# what `research run --on vy-nebius-1 --preflight check -- true` printed on 2 Oct (00:51Z without the bundles' URLs, 00:52Z with)
REFUSED = """research: preflight tools/verity/check/preflight.py on vy-nebius-1: FAILED in 0.7 s
research: preflight: uv 0.12.20
research: run r20261002-005132-46b6 refused on vy-nebius-1 (PreflightFailed): the machine failed the preflight tools/verity/check/preflight.py (exit 1): Security/Proofs: its Lean dependencies (32c884ee1ea4) are not warm here and the run carries no URL for their bundle
"""
LAUNCHED = """research: preflight: store: Security/Proofs's Lean bundle 32c884ee1ea4 read with the run's custody key
research: vy-nebius-1's lease was not extended: the run has no --timeout, so a pod lease (if any) may end it
research: run r20261002-005227-a0ab launched on vy-nebius-1 (None) pid=1398558 -> /workspace/research/runs/r20261002-005227-a0ab
research: workload cwd: --cwd source
"""


def test_a_probe_runs_output_is_a_launch_a_preflight_refusal_or_an_error():
    assert P.verdict(0, LAUNCHED) == ("ok", "r20261002-005227-a0ab")
    v, detail = P.verdict(3, REFUSED)
    assert v == "failed" and detail.startswith("the machine failed the preflight") and "32c884ee1ea4" in detail
    assert P.verdict(3, "research: run r1 refused on vy-nebius-1 (Refused): machine is exclusively held by run r0\n")[0] == "error"
    assert P.verdict(255, "") == ("error", "exit 255, no output")


def test_one_alert_per_failing_machine_and_one_cleared_reply_once_it_passes():
    posts: list[tuple[str, str | None]] = []

    def post(text, thread):
        posts.append((text, thread))
        return f"ts{len(posts)}"

    state = {"open": {}}
    failed = ("failed", "uv is 0.12.21, not the 0.12.20 pod_setup.sh pins")
    P.step(state, {"vy-nebius-1": failed, "vy-nebius-2": ("ok", "r2")}, post)
    P.step(state, {"vy-nebius-1": failed, "vy-nebius-2": ("ok", "r3")}, post)
    assert len(posts) == 1 and "`vy-nebius-1`" in posts[0][0] and "0.12.21" in posts[0][0] and posts[0][1] is None
    P.step(state, {"vy-nebius-1": ("error", "ssh: connect timed out")}, post)     # an error neither alerts nor clears
    assert len(posts) == 1 and set(state["open"]) == {"vy-nebius-1"}
    P.step(state, {"vy-nebius-1": ("ok", "r4")}, post)
    assert posts[1] == ("*Cleared:* check's preflight passes on `vy-nebius-1` again (run r4).", "ts1") and state["open"] == {}
