"""The distribution's vLLM general plugins, and the environment that loads exactly the ones a run names.

vLLM calls every installed entry point of the group ``vllm.general_plugins`` in each of its processes
(``vllm.plugins.load_plugins_by_group`` at the pin), or, when ``VLLM_PLUGINS`` is set, only the ones it names,
comma-separated. A named plugin it can't find, or one that fails to load, it skips with a log line. `PLUGINS` is each of
this distribution's plugins: its entry point's target, and the environment variable its ``register`` reads to turn on (None
for a plugin that is always on). ``pyproject.toml`` declares exactly these, which a test holds. `env` is the environment of
a vLLM server or engine whose processes load the always-on plugins and the ones it names, each with its setting, so a name
`PLUGINS` lacks fails here rather than being skipped there.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

__all__ = ["GROUP", "PLUGINS", "Plugin", "env"]

#: vLLM's entry point group of general plugins
GROUP = "vllm.general_plugins"


@dataclass(frozen=True)
class Plugin:
    """``target`` is the entry point, ``module:function``. ``setting`` is the environment variable whose value turns the
    plugin on and configures it, or None for a plugin that is always on."""

    target: str
    setting: str | None = None


PLUGINS = {
    "verity_moe_align": Plugin("verity_vllm.engine.moe_align:register"),
}


def env(*names: str, **settings: Any) -> dict[str, str]:
    """The environment that loads, in each of vLLM's processes, every always-on plugin and each plugin ``names`` lists,
    with its setting given as the keyword of its name (a string as it is, anything else as compact JSON). Refused
    (`ValueError`) for a name `PLUGINS` lacks, for a named plugin whose setting isn't given, and for a setting given to a
    plugin that isn't named or takes none."""
    unknown = [n for n in names if n not in PLUGINS]
    if unknown:
        raise ValueError(f"no vLLM plugin named {unknown} in engine.plugins.PLUGINS, which has {sorted(PLUGINS)}")
    stray = sorted(set(settings) - set(names))
    if stray:
        raise ValueError(f"settings for plugins that aren't named: {stray}")
    out: dict[str, str] = {}
    for n in names:
        p = PLUGINS[n]
        if p.setting is None:
            if n in settings:
                raise ValueError(f"{n} takes no setting: it is always on")
            continue
        if n not in settings:
            raise ValueError(f"{n} is on only with {p.setting} set: pass its setting as {n}=...")
        v = settings[n]
        out[p.setting] = v if isinstance(v, str) else json.dumps(v, separators=(",", ":"))
    on = [n for n, p in PLUGINS.items() if p.setting is None or n in names]
    return {"VLLM_PLUGINS": ",".join(on), **out}
