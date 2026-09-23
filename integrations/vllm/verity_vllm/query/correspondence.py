"""The per-Call runtime-correspondence annotations of a derived Program, read -- not invented.

Today the frontend records ONE annotation per Call: its naming hint (`derive.Ctx.hint_name` + the rule's member / row suffix),
carried as `annotations["node_hints"][k]` in the descriptor and as `name` in the instance sequence::

    <module path>/<fx node>[/<member hint>][<row>]        model.model.layers.0.self_attn.rotary_emb/rotary_embedding/query[7]
    <fx node>[<row>]                                       argmax_2[0]            (a Call outside every module: the root body)

`<module path>` is the owning module in the SOURCE module's own vocabulary (`named_modules()` of the export wrapper), from the fx
node's `nn_module_stack`; `<member hint>` is the slot the module RETURNS the value as, when the rule knew it (`out`, `0`, `1`,
`query`, `key`); `<row>` is the token row (`[7]`) or the attention position (`[t=3]`) the Call is the instance for.  A collective's
row name ends in `/all_reduce` after the row (`...[i]/all_reduce`).

The second correspondence fact is the map from a Program module path to the IMPLEMENTATION's path (the collector's
`named_modules()` name): the export wrapper holds the served model under an attribute, so `model.model.layers.0...` in the Program
is `model.layers.0...` at the collector.  It is determined from the Build's observed implementation paths
(`result.json: attention_impls_observed`, keys = implementation module paths) by suffix match against the Program's module paths --
the one prefix every observed path is a suffix under -- never assumed.  Phase 3 records both facts as first-class annotations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Iterable, Mapping

from .program_view import ProgramView

__all__ = ["CallName", "parse_call_name", "Correspondence", "implementation_prefix"]

_ROW_RE = re.compile(r"\[([^\[\]]*)\]$")


@dataclass(frozen=True, slots=True)
class CallName:
    module: str                      # owning module path in the Program's (export wrapper's) vocabulary; "" = no module (root body)
    node: str                        # the fx node
    hint: str                        # the recorded return-slot hint ("" when the rule recorded none)
    row: str | None                  # the row / position label inside `[...]`, None when absent
    suffix: str                      # a trailing `/<tag>` after the row (`all_reduce`), "" otherwise


def parse_call_name(name: str) -> CallName:
    """`model.a.b/node/hint[7]` -> CallName("model.a.b", "node", "hint", "7", "")."""
    suffix = ""
    row = None
    m = _ROW_RE.search(name)
    if m is None and "]/" in name:                         # `<...>[i]/all_reduce`
        head, _, suffix = name.rpartition("/")
        m = _ROW_RE.search(head)
        if m is not None:
            name = head
    if m is not None:
        row = m.group(1)
        name = name[: m.start()]
    parts = name.split("/")
    if len(parts) == 1:
        return CallName("", parts[0], "", row, suffix)
    module, node = parts[0], parts[1]
    hint = parts[2] if len(parts) > 2 else ""
    return CallName(module, node, hint, row, suffix)


def implementation_prefix(program_modules: Iterable[str], implementation_paths: Iterable[str]) -> str:
    """The wrapper prefix `p` such that an observed implementation path `a` is the Program module path `p + a`: the shortest prefix
    under which EVERY observed path is found as a suffix of some Program module path.  "" when nothing is observed or the
    Program's paths are already the implementation's."""
    mods = [m for m in set(program_modules) if m]
    obs = [a for a in set(implementation_paths) if a]
    if not obs or not mods:
        return ""
    candidates: list[set[str]] = []
    for a in obs:
        c = set()
        for m in mods:
            if m == a:
                c.add("")
            elif m.endswith("." + a):
                c.add(m[: len(m) - len(a)])
        candidates.append(c)
    common = set.intersection(*candidates) if candidates else set()
    if not common:
        return ""
    return min(common, key=len)


class Correspondence:
    """The correspondence table of one ProgramView: Call id -> CallName, plus the implementation-path map."""

    def __init__(self, P: ProgramView, *, implementation_paths: Iterable[str] = ()) -> None:
        self.P = P
        self.names: dict[int, CallName] = {c.id: parse_call_name(c.name) for c in P.calls}
        self._impl_paths = tuple(implementation_paths)

    def module_of(self, cid: int) -> str:
        return self.names[cid].module

    def name(self, cid: int) -> CallName:
        return self.names[cid]

    @cached_property
    def modules(self) -> tuple[str, ...]:
        return tuple(sorted({n.module for n in self.names.values()}))

    @cached_property
    def prefix(self) -> str:
        return implementation_prefix(self.modules, self._impl_paths)

    def implementation_path(self, module: str) -> str:
        """Program module path -> the implementation's (collector's) path."""
        p = self.prefix
        if p and module[: len(p)] == p:
            return module[len(p):]
        return module

    @staticmethod
    def implementation_paths_of(result_json: Mapping[str, Any] | None) -> tuple[str, ...]:
        """The observed implementation module paths a Build's `result.json` records (`attention_impls_observed` keys)."""
        if not result_json:
            return ()
        obs = result_json.get("attention_impls_observed") or {}
        return tuple(str(k) for k in obs)
