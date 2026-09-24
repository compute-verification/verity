"""Runtime correspondence for the acquisition plan: ONE reader, an explicit fallback.

The plan asks three things of a required Value: `owning_module(call)` (the runtime module path whose forward produced / consumes it),
`return_slot(call, port)` (where on that module's return it sits: the whole return `out`, tuple slot `k`, an attribute of the runner's
`SamplerOutput`) and `arg_slot(call, i)` (the i-th positional argument of the module's forward).  `call` is a required-value manifest
identity row (`op_path`, `output_member`, ...), a Program instance row (`name`, `spec`[, `i`]) or `{"call": k[, "program": label]}`.

WHERE the owning module comes from -- `source`:

  correspondence   the `runtime-correspondence/v1` record the frontend emitted into `annotations["correspondence"]` of the Build's request
                   Program descriptors (`verity_vllm.correspondence.runtime.load_correspondence`; one table per component Program).  A
                   Value's module path is the record's owning module (`occurrences[occ].module`, the source module's own vocabulary)
                   mapped to the implementation's `named_modules()` name by the manifest's `wrapper_prefix` (`collector_path`) and
                   `op_path_aliases` -- the SAME mapping the manifest applied to the naming hints.
  v1_annotations   a v1-frontend Build (no such namespace): the identity row's `op_path` as `required_manifest` read it off the naming
                   hints (`annotations["node_hints"]`: `split_name(name)[0]` -> `collector_path`) -- what this adapter read before the
                   record existed.  Chosen ONLY when the record is absent, and recorded (`AcquisitionPlan.correspondence`), never silent.

Under `correspondence` every module-site Value is CONFIRMED against the record (its op_path is a module the record owns Calls under) or
listed `unconfirmed` in the plan -- a finding, not a changed site: the manifest's op_path stands either way, so the plan's content is the
same under both sources and the plan digest (which excludes the `correspondence` provenance block) says whether anything changed.

The runtime side is `runtime_tree.RuntimeTree` (module-tree facts: which modules the runtime HAS; re-exported here for callers).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from verity_vllm.query.manifest.format import collector_path, split_name
from verity_vllm.correspondence.runtime_tree import RUNNER_PATH, RUNTIME_TREE_SCHEMA, RuntimeTree, mro_names, runtime_tree_from_paths, runtime_tree_of
from verity_vllm.correspondence.runtime import NAMESPACE, SCHEMA, RuntimeCorrespondence, read_program_dir

SOURCES = ("correspondence", "v1_annotations")
FALLBACK_READ = "the identity row's op_path: required_manifest.split_name(annotations.node_hints)[0] -> collector_path(wrapper_prefix) -> op_path_aliases"


@dataclass(frozen=True)
class ReturnSlot:
    """Where a Value sits on a module's return.  `index=None, attr=None` = the whole return (a single tensor); `index=k` = tuple slot k;
    `attr` = an attribute of the returned object (the runner's `SamplerOutput.sampled_token_ids`)."""
    index: int | None = None
    attr: str | None = None

    def label(self) -> str:
        if self.attr is not None:
            return f".{self.attr}"
        return "out" if self.index is None else str(self.index)


@dataclass(frozen=True)
class ArgSlot:
    index: int

    def label(self) -> str:
        return f"in[{self.index}]"


def read_program_correspondence(program_dir: str | Path) -> tuple[RuntimeCorrespondence | None, str | None, str | None]:
    """The `runtime-correspondence/v1` table of one derive_step output dir, (table, digest, problem) -- `runtime_correspondence.read_program_dir`,
    the one loader: `result.json:program.correspondence_digest` is the presence test (a v1-frontend Build -> (None, None, None) without
    decoding the descriptor); with it the record is loaded and its digest checked against the recorded one."""
    return read_program_dir(program_dir)


def infer_wrapper_prefix(program_modules: Iterable[str], runtime: RuntimeTree) -> str:
    """The export wrapper's attribute prefix, determined: of every dotted prefix of the record's module paths, the one under which the
    most paths are modules of the runtime tree (`""` when the paths already are; ties to the shortest).  A record whose paths meet the
    tree under no prefix gets `""` -- every module then reads unconfirmed, which the plan's provenance shows."""
    votes: dict[str, int] = {}
    for m in program_modules:
        if not m:
            continue
        if runtime.has_module(m):
            votes[""] = votes.get("", 0) + 1
        for i, ch in enumerate(m):
            if ch == "." and runtime.has_module(m[i + 1:]):
                votes[m[:i + 1]] = votes.get(m[:i + 1], 0) + 1
    if not votes:
        return ""
    return max(sorted(votes, key=len), key=lambda p: votes[p])


class AcquireCorrespondence:
    """`owning_module` / `return_slot` / `arg_slot` for the plan, over the canonical record when the Build carries it (`tables`), else the
    manifest's v1-annotation reading; plus the runtime tree and the (optional) in-place facts.

    `tables`: {program label: RuntimeCorrespondence} -- one per component request Program of the manifest (a global manifest has one per
    request shape).  `inplace_facts`: {module path: {"mutates_input": bool, "output_mutated_after": bool}} -- alias / in-place facts;
    today none are read, so a consumer-input site's lifetime is `unknown` unless a fact says the input is left unmutated."""

    def __init__(self, runtime: RuntimeTree, *, wrapper_prefix: str = "", aliases: Mapping[str, Any] | None = None,
                 inplace_facts: Mapping[str, Mapping[str, Any]] | None = None, tables: Mapping[str, RuntimeCorrespondence] | None = None,
                 digests: Mapping[str, str] | None = None, absent: Iterable[str] = (), problems: Iterable[str] = ()) -> None:
        self.runtime = runtime
        self.wrapper_prefix = wrapper_prefix
        self.aliases = dict(aliases or {})                      # program op_path -> [canonical op_path, invocation] (`op_path_aliases`)
        self.inplace_facts = {str(k): dict(v) for k, v in (inplace_facts or {}).items()}
        self.tables: dict[str, RuntimeCorrespondence] = dict(tables or {})
        self.digests: dict[str, str] = dict(digests or {})
        self.absent: list[str] = list(absent)
        self.problems: list[str] = list(problems)
        self.source = "correspondence" if self.tables else "v1_annotations"
        # the export wrapper's attribute prefix (Program `model.model.layers.0...` -> collector `model.layers.0...`): the request manifest
        # carries it (`wrapper_prefix`); a GLOBAL manifest does not -- then it is determined from the runtime tree (the one prefix under
        # which the record's module paths are the tree's), never assumed
        self.wrapper_prefix_source = "manifest" if wrapper_prefix else None
        if not wrapper_prefix and self.tables:
            wrapper_prefix = infer_wrapper_prefix((m for t in self.tables.values() for m in t.modules()), runtime)
            self.wrapper_prefix = wrapper_prefix
            self.wrapper_prefix_source = "runtime_tree"
        # collector path -> the Program module paths (per table) the record owns Calls under
        self._owned: dict[str, set[tuple[str, str]]] = {}
        for label, t in self.tables.items():
            for m in t.modules():
                op = collector_path(m, wrapper_prefix)
                alias = self.aliases.get(op)
                self._owned.setdefault(str(alias[0]) if alias else op, set()).add((label, m))

    # -- construction from a Build --------------------------------------------------------------------------------------------------

    @classmethod
    def from_programs(cls, runtime: RuntimeTree, programs: Iterable[str | Path], *, wrapper_prefix: str = "", aliases: Mapping[str, Any] | None = None,
                      inplace_facts: Mapping[str, Mapping[str, Any]] | None = None) -> "AcquireCorrespondence":
        """Read the record of every component Program dir.  `source` is `correspondence` only when EVERY program carries it; a Build with
        the record on some components and not others falls back as a whole (uniform decision), the components lacking it listed `absent`."""
        tables: dict[str, RuntimeCorrespondence] = {}
        digests: dict[str, str] = {}
        absent: list[str] = []
        problems: list[str] = []
        for p in programs:
            label = os.path.basename(os.path.normpath(str(p)))
            t, dg, prob = read_program_correspondence(p)
            if prob:
                problems.append(prob)
            if t is None:
                absent.append(label)
            else:
                tables[label] = t
                digests[label] = dg or ""
        if absent and tables:
            problems.append(f"{len(absent)} of {len(absent) + len(tables)} component Programs carry no {SCHEMA} record: the plan falls back to v1_annotations as a whole")
            tables, digests = {}, {}
        return cls(runtime, wrapper_prefix=wrapper_prefix, aliases=aliases, inplace_facts=inplace_facts, tables=tables, digests=digests, absent=absent, problems=problems)

    # -- the three functions of the correspondence namespace ------------------------------------------------------------------

    def owning_module(self, call: Mapping[str, Any]) -> str:
        if "op_path" in call:
            op = str(call["op_path"])
        elif "call" in call and self.tables:
            label = str(call.get("program") or next(iter(self.tables)))
            op = collector_path(self.tables[label].owning_module(int(call["call"])), self.wrapper_prefix)
        elif "i" in call and "name" in call and len(self.tables) == 1:
            op = collector_path(next(iter(self.tables.values())).owning_module(int(call["i"])), self.wrapper_prefix)
        elif "name" in call:
            op = collector_path(split_name(str(call["name"]))[0], self.wrapper_prefix)
        else:
            raise KeyError("call carries neither op_path, call nor name")
        alias = self.aliases.get(op)
        return str(alias[0]) if alias else op

    def return_slot(self, call: Mapping[str, Any], port: str | None = None) -> ReturnSlot | None:
        port = str(call.get("output_member", "out") if port is None else port)
        if port == "in":
            return None                                          # an input of the module: no return slot
        if port == "out":
            return ReturnSlot()
        if port.isdigit():
            return ReturnSlot(index=int(port))
        if self.owning_module(call) == RUNNER_PATH:
            return ReturnSlot(attr=port)                         # `sampled_token_ids` / `seed` / `splits` of the runner's sample()
        return ReturnSlot()                                      # a hidden-stream geometry name, a weight, a tap plane: the whole return of its site

    def arg_slot(self, call: Mapping[str, Any], i: int = 0) -> ArgSlot:
        return ArgSlot(index=int(i))

    # -- confirmation against the record ---------------------------------------------------------------------------------------------

    def confirmed(self, module: str) -> bool | None:
        """Under `correspondence`: does the record own Calls under this (collector-vocabulary) module path?  None under the fallback."""
        if not self.tables:
            return None
        return module in self._owned

    def owned_modules(self) -> list[str]:
        return sorted(self._owned)

    def provenance(self) -> dict[str, Any]:
        """The plan's `correspondence` block (provenance: excluded from the plan digest)."""
        out: dict[str, Any] = {"source": self.source, "schema": SCHEMA if self.tables else None, "namespace": NAMESPACE if self.tables else None,
                               "programs": dict(sorted(self.digests.items())), "absent": list(self.absent), "problems": list(self.problems),
                               "module_paths": len(self._owned), "wrapper_prefix": self.wrapper_prefix, "wrapper_prefix_source": self.wrapper_prefix_source}
        if not self.tables:
            out["fallback"] = FALLBACK_READ
        return out

    # -- lifetime facts -------------------------------------------------------------------------------------------------------------

    def input_lifetime(self, consumer: str) -> str:
        f = self.inplace_facts.get(consumer)
        if f is None:
            return "unknown"
        return "unmutated" if not f.get("mutates_input", True) else "mutated-by-consumer"

    def output_lifetime(self, producer: str) -> str:
        f = self.inplace_facts.get(producer)
        if f is None:
            return "unknown"
        return "unmutated" if not f.get("output_mutated_after", True) else "mutated-after-hook"


AnnotationCorrespondence = AcquireCorrespondence      # the adapter's former name (callers written against it keep working)

__all__ = ["SOURCES", "FALLBACK_READ", "ReturnSlot", "ArgSlot", "AcquireCorrespondence", "AnnotationCorrespondence", "read_program_correspondence",
           "RuntimeTree", "RUNTIME_TREE_SCHEMA", "RUNNER_PATH", "mro_names", "runtime_tree_of", "runtime_tree_from_paths", "RuntimeCorrespondence"]
