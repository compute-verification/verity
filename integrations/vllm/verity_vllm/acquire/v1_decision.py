"""v1's hook DECISION as a pure function of (runtime tree, required-value manifest) -- what `NativeHostCommitter._select_modules`,
`_acquires_input`, `_input_only_modules`, the runner wraps in `install()`, and `bench.commit_delta`'s manifest-driven arming decide,
without an engine and without executing anything.  The CPU gate compares its site set with the plan's (`gate.compare_sites`).

The decision, in v1's own terms (tables imported from v1, never copied):
  outputs   every module whose MRO meets ACQUIRE_CLASSES and not SKIP_CLASSES (`_select_modules`); + `runner.sampler` when the runner's sampler
            is an nn.Module (its forward hook never fires); + [commit_delta] every non-promoted `instance_outputs` op_path of the manifest that
            is a module of the model, not selected, and not a site a source binds (a TP collective's received output)
  inputs    a selected module whose MRO meets ACQUIRE_INPUT_CLASSES, or that the manifest's promoted rows name as a first consumer
            (`promoted_observation` -> `acquire_inputs`); + the consumers the committer did not select (`_input_only_modules`)
  runner    `runner.sampler/sampled_token_ids` from the wrapped `sample()`; `seed` when the manifest has `sampler_seed`; `splits` when it has
            `sampler_splits` (the launch proxies)
  taps      the attention hidden-stream tap per attention layer when the manifest has a hidden family (row_pod always passes --hidden-so);
            the MoE block source when it names `moe_block_stream`; the TP partial source when it names `tp_rank_partials`
  weights   `register_weights`: every parameter and buffer, once

`test_v1_decision_matches_committer` holds this function to the real committer's decision on a module tree.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from verity_capture.commit.manifest_format import promoted_observation
from verity_vllm.acquire.runtime_tree import RUNNER_PATH, RuntimeTree
from verity_vllm.acquire.plan import HIDDEN_TAP_OF_FAMILY, MOE_FAMILY, TP_FAMILY, _v1_tables


@dataclass
class V1Decision:
    outputs: list[str] = field(default_factory=list)         # forward hooks (module path), v1 order
    inputs: list[str] = field(default_factory=list)          # forward pre-hooks (arg 0)
    runner: list[str] = field(default_factory=list)          # `sampled_token_ids` | `seed` | `splits`
    taps: list[tuple[str, str]] = field(default_factory=list)   # (tap path, plane / `*`)
    weights: bool = True
    causes: dict[str, str] = field(default_factory=dict)     # site key string -> the table / rule that put it there
    problems: list[str] = field(default_factory=list)

    def sites(self) -> set[tuple[str, str, str]]:
        """The (module path, kind, slot) vocabulary of the plan; a v1 output hook takes every returned tensor -> slot `*`."""
        s: set[tuple[str, str, str]] = set()
        s |= {(m, "producer_output", "*") for m in self.outputs}
        s |= {(m, "consumer_input", "in[0]") for m in self.inputs}
        kinds = {"sampled_token_ids": "runner_return", "seed": "runner_state", "splits": "launch_proxy"}
        s |= {(RUNNER_PATH, kinds[r], f".{r}") for r in self.runner}
        s |= {(p, "native_tap", plane) for p, plane in self.taps}
        if self.weights:
            s.add(("weights", "registered_input", "*"))
        return s

    def to_json(self) -> dict[str, Any]:
        return {"schema": "verity-vllm/v1-hook-decision/v1", "outputs": list(self.outputs), "inputs": list(self.inputs), "runner": list(self.runner),
                "taps": [list(t) for t in self.taps], "weights": self.weights, "causes": dict(self.causes), "problems": list(self.problems),
                "sites": sorted(list(k) for k in self.sites())}


def v1_decision(rt: RuntimeTree, manifest: Mapping[str, Any] | None, *, tables: Mapping[str, Any] | None = None) -> V1Decision:
    t = dict(tables or _v1_tables())
    d = V1Decision()
    rows = (manifest or {}).get("identities") or []
    fams = set((manifest or {}).get("populations", {}).get("by_family") or {}) or {str(r.get("family")) for r in rows}
    # _select_modules
    for path in rt.modules:                                             # named_modules() order
        classes = rt.classes(path)
        if classes & t["SKIP_CLASSES"]:
            continue
        if classes & t["ACQUIRE_CLASSES"]:
            d.outputs.append(path)
            d.causes[f"{path}|producer_output"] = f"ACQUIRE_CLASSES ({(classes & t['ACQUIRE_CLASSES']) and sorted(classes & t['ACQUIRE_CLASSES'])[0]})"
    if rt.runner.get("sampler_module"):
        d.outputs.append(RUNNER_PATH)
        d.causes[f"{RUNNER_PATH}|producer_output"] = "ACQUIRE_CLASSES (the runner's sampler nn.Module; forward hook never fires)"
    # commit_delta: acquisition extended by the manifest (non-promoted instance_outputs op_paths of the model the committer did not select)
    if manifest is not None:
        tp_sites = {str(r.get("op_path")) for r in rows if r.get("family") == TP_FAMILY}
        named = sorted({str(r.get("op_path")) for r in rows if r.get("family") == "instance_outputs" and r.get("op_path") and not r.get("promoted")})
        selected = set(d.outputs)
        for n in named:
            if n in rt.modules and n not in selected and n not in tp_sites:
                d.outputs.append(n)
                d.causes[f"{n}|producer_output"] = "commit_delta: manifest instance_outputs op_path the class table did not select"
        # promoted members -> consumer inputs
        pia = promoted_observation(manifest)
        if pia["problems"]:
            d.problems += [f"promoted_observation: {p}" for p in pia["problems"]]
        acquire_inputs = set(pia["consumers"])
    else:
        acquire_inputs = set()
    # _acquires_input on the selected modules, then _input_only_modules
    for path in d.outputs:
        if path == RUNNER_PATH:
            continue
        if rt.classes(path) & t["ACQUIRE_INPUT_CLASSES"]:
            d.inputs.append(path)
            d.causes[f"{path}|consumer_input"] = "ACQUIRE_INPUT_CLASSES"
        elif path in acquire_inputs:
            d.inputs.append(path)
            d.causes[f"{path}|consumer_input"] = "manifest promoted consumer (acquire_inputs)"
    selected = set(d.outputs)
    for n in sorted(acquire_inputs):
        if n not in selected and n in rt.modules:
            d.inputs.append(n)
            d.causes[f"{n}|consumer_input"] = "manifest promoted consumer of no acquire class (_input_only_modules)"
    # runner wraps
    if rt.runner.get("sample"):
        d.runner.append("sampled_token_ids")
        d.causes[f"{RUNNER_PATH}|runner_return"] = "install(): wrapped runner.sample()"
        if "sampler_seed" in fams:
            d.runner.append("seed")
            d.causes[f"{RUNNER_PATH}|runner_state"] = "install(): sampler seeds gathered at sample() (stochastic manifest)"
        if "sampler_splits" in fams:
            d.runner.append("splits")
            d.causes[f"{RUNNER_PATH}|launch_proxy"] = "install(): split-kernel launch proxies (derived-S manifest)"
    # sources by manifest family
    for fam, tap in HIDDEN_TAP_OF_FAMILY.items():
        if fam in fams:
            for p in sorted({str(r.get("op_path")) for r in rows if r.get("family") == fam}):
                d.taps.append((p, "*"))
                d.causes[f"{p}|native_tap"] = f"hidden source ({tap}) attached by commit_delta --hidden-so"
    if MOE_FAMILY in fams:
        for p, plane in sorted({(str(r.get("op_path")), str(r.get("output_member"))) for r in rows if r.get("family") == MOE_FAMILY}):
            d.taps.append((p, plane))
            d.causes[f"{p}|native_tap"] = "MoE block source (manifest names moe_block_stream)"
    if TP_FAMILY in fams:
        for p, member in sorted({(str(r.get("op_path")), str(r.get("output_member"))) for r in rows if r.get("family") == TP_FAMILY}):
            d.taps.append((p, member))
            d.causes[f"{p}|native_tap"] = "TP partial source (manifest names tp_rank_partials)"
    d.causes["weights|registered_input"] = "register_weights(): every parameter and buffer"
    return d
