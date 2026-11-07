"""The replay ladder's environment variables, each declared once (`verity_config.settings`): the arm run.sh serves and
replays a model in, where a replay's processes log Inductor's run-time choices, the route capture, and the DSA top-k probe.
ladder.py reads them (``of(__file__)``) and hands them to inductor_pin and topk_probe, which read none themselves: serving
loads inductor_pin from a site directory outside the tree.
"""
from __future__ import annotations

from verity_config.settings import VALUE, WHERE, Registry, Setting as S, choice

REGISTRY = Registry(
    __file__, ("LADDER_",),
    S("LADDER_ARM", choice("pinned", "rblock", "unpinned"), "unpinned", VALUE,
      "the arm run.sh serves and replays a model in: pinned keeps Inductor's first candidate at each on-device choice "
      "(inductor_pin), rblock runs with TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0, unpinned with neither"),
    S("LADDER_TILINGS", str, None, WHERE, "the directory a replay's engine processes log Inductor's run-time choices to "
                                          "(ladder.py sets it for the processes a replay starts)"),
    S("LADDER_ROUTE_CAPTURE", str, None, WHERE,
      "a directory holding the route capture's plugin (route_capture_plugin.py and its settings.py): ladder.py serve then "
      "loads it in both servers, each writing its capture under its own route_capture/ (ROUTECAP_OUT); replays never "
      "load it"),
    S("LADDER_TOPK_PROBE", choice("record", "sort"), None, VALUE,
      "probe each replay's DSA prefill top-k (topk_probe.py): record its checksums per call under replay-I/topk, and in "
      "sort mode also sort each row's indices before attention reads them"),
    S("LADDER_TOPK_PROBE_REAL", str, None, VALUE,
      "module:function, a replacement for vLLM's top_k_per_row_prefill (its signature, writing the rows in place) that the "
      "probe calls and records instead; it must be importable in every engine process"),
    S("LADDER_TOPK_PROBE_OUT", str, None, WHERE, "the directory a replay's engine processes write the top-k probe's records "
                                                 "to (ladder.py sets it for the processes a replay starts)"),
)
