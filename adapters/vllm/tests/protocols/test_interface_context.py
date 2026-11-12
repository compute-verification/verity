"""`verity_vllm.protocols.interface.Context`: what an adapter reads of its own settings, and of the settings the built
engine will supply, where the run's settings or its fills leave the protocol out."""
from __future__ import annotations

from typing import Any

from verity_vllm.protocols.interface import Context

ENABLED = ("pouw:ncp-v1",)


def test_a_protocol_the_settings_leave_out_reads_no_settings_and_no_fills() -> None:
    bare = Context(stage="serve", enabled=ENABLED)
    other = Context(stage="serve", enabled=ENABLED, settings={"pous": {"key": "k"}}, filled={"pous": ("key",)})
    for ctx in (bare, other):
        assert ctx.settings_of("pouw") == {}
        assert ctx.filled_of("pouw") == ()


def test_a_protocol_reads_its_own_settings_and_fills_and_not_another_s() -> None:
    own = {"rows": 4, "steps": 2}
    fills: dict[str, Any] = {"pouw": ["runner", "coins"], "pous": ("key",)}
    ctx = Context(stage="serve", enabled=ENABLED, settings={"pouw": own, "pous": {"key": "k"}}, filled=fills)
    assert ctx.settings_of("pouw") is own
    assert ctx.filled_of("pouw") == ("runner", "coins")
    assert ctx.settings_of("pous") == {"key": "k"}
    assert ctx.filled_of("pous") == ("key",)


def test_an_entry_that_is_none_reads_as_an_absent_one() -> None:
    """`load_settings` refuses such an entry from a file, so only a `Context` built by hand holds one: it reads as no
    settings and no fills, never as None."""
    none: dict[str, Any] = {"pouw": None}
    ctx = Context(stage="serve", enabled=ENABLED, settings=none, filled=none)
    assert ctx.settings_of("pouw") == {}
    assert ctx.filled_of("pouw") == ()
