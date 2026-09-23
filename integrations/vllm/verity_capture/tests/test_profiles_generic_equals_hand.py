"""[P6, 2026-09-23 -- TEMPORARY: leaves with the hand-written modules] Object equality, hand module vs `profiles.generic`, per role.

For every role that had a hand-written κ profile module, the canonical form (`profiles.canonical`) of the hand module's PROFILE equals the
generic constructor's, on the load-bearing surface: the ordered pattern list with every pinned field, `C`, the κ pins, `weights_type`,
`weight_field` / `root_class` by extension over the probe set, `check_header` over the role's config.json and its perturbations.

The DECLARED deltas (each one a decision, not an accident), removed from both sides before the comparison:
  * `GemmLaunchTuned` (`gen_llama_patterns`) precedes the base `GemmLaunch` in every generic profile; the hand B0 / B1 / Qwen-0.5B / Phi-3 /
    Gemma-2 / Pythia / Llama-TP2 modules did not register it (a TARGET fact: it owns only launches whose (N, K) is a key of the record
    target's table, so on those roles' launches it never fires) and the two MoE modules registered its Ada twin `GemmLaunchTunedAda`
    (`gen_moe_patterns`, the same table, a base-constexpr pin the target-agnostic one derives) -- the `.tuned` / `.tuned_ada` entry is dropped;
  * `C["ROUTER_VPT"]` (MoE): the router kernel's values-per-thread, derived from E by `generic.router_vpt_bf16` and asserted against the hand
    table `ROUTER_VPT_BY_E`; the hand modules passed the table to the scope pattern instead (the pattern's `router_vpt` is compared);
  * `C["WORLD"]` (TP ranks) and the `unmodelled-moe-variant` root class (a finer answer than the hand modules' "unknown" for shared-expert /
    EP-map / router-bias parameters the constructive block does not state);
  * `id` (the generic id is `...-<role>` / `...-<role>-tp<W>-rank`), `config.profile` (free text), `config.revision` / `config.model` where the
    hand module pinned none, `config.family` / `config.rope_scaling` where the hand module's `C` predates `analytic.config_of`, `notes` (free
    text), and the `check_header` probe where the hand module has no `HF_CONFIG` attribute (the probe needs the config);
  * the TP2 sampler's vocabulary on `TokenSelectGumbelTopP`: the two older `*_tp2rank` modules set the FULL vocabulary on `TokenSelectArgmax`
    and `TokenSelectGumbelTwoStage` only (their docstrings say "`TokenSelect*` patterns take `vocab = V`"; the top-p pattern post-dates them),
    `gen_ov_moe_qwen3` and the generic `world > 1` branch set it on all three -- the generic branch does what the docstrings state."""
from __future__ import annotations

import importlib
import json

import pytest

from verity_capture.profiles import canonical as K

#: role (world) -> the hand module it had
HAND = {
    ("B0", 1): "vllm_d9105ea80_sm89_eager",
    ("B1", 1): "vllm_d9105ea80_sm89_eager_qwen15",
    ("B1", 2): "vllm_d9105ea80_sm89_eager_qwen15_tp2rank",
    ("LLAMA32_1B", 1): "gen_llama_llama32_1b",
    ("LLAMA32_1B", 2): "gen_llama_llama32_1b_tp2rank",
    ("QWEN05", 1): "gen_llama_qwen05",
    ("B7", 1): "gen_ov_easy_qwen25_7b",
    ("GEMMA2_2B", 1): "gen_dense2_gemma2_2b",
    ("PHI3_MINI", 1): "gen_dense2_phi3_mini",
    ("PYTHIA160M", 1): "gen_ln_pythia160m",
    ("OLMOE", 1): "gen_ov_moe_olmoe",
    ("QWEN3_30B_A3B", 2): "gen_ov_moe_qwen3",
    ("MISTRAL7B", 1): "dense_MISTRAL7B",
    ("QWEN3_4B", 1): "dense_QWEN3_4B",
    ("QWEN3_4B_FP8", 1): "dense_QWEN3_4B_FP8",
    ("TINYLLAMA", 1): "dense_TINYLLAMA",
}
IGNORE_CONFIG = {"profile", "revision", "model", "family", "WORLD", "ROUTER_VPT", "ROPE_THETA"}   # ROPE_THETA: int vs float, compared as numbers
IGNORE_PATTERN_NAMES = {"gemm.matmul_kernel_persistent.tuned", "gemm.matmul_kernel_persistent.tuned_ada"}


OLDER_TP2 = {"vllm_d9105ea80_sm89_eager_qwen15_tp2rank", "gen_llama_llama32_1b_tp2rank"}


def _strip(doc: dict, *, older_tp2: bool = False) -> dict:
    d = json.loads(json.dumps(doc, default=repr))
    d.pop("id"); d.pop("check_header"); d.pop("probe", None)
    cfg = d["config"]
    d["rope_theta_numeric"] = float(cfg["ROPE_THETA"])
    if cfg.get("rope_scaling") is None:
        cfg.pop("rope_scaling", None)
    for k in IGNORE_CONFIG:
        cfg.pop(k, None)
    pats = [p for p in d["patterns"] if p.get("name") not in IGNORE_PATTERN_NAMES]
    for p in pats:
        if isinstance(p.get("config"), dict):
            for k in ("ROUTER_VPT", "WORLD", "profile", "revision", "model", "family"):
                p["config"].pop(k, None)
            p["config"]["ROPE_THETA"] = float(p["config"]["ROPE_THETA"])
            if p["config"].get("rope_scaling") is None:
                p["config"].pop("rope_scaling", None)
        if p.get("name") == "sample.gumbel_topp_select" and older_tp2:
            p["vocab"] = "full-or-rank (declared delta on the two older tp2rank modules)"
    d["patterns"] = pats
    d["root_class"] = {n: c for n, c in (d.get("root_class") or {}).items() if c != "unmodelled-moe-variant"}
    if isinstance(d.get("pins"), dict):
        d["pins"].pop("gemm_tuned_ada", None)
    return d


def _canon_hand(name: str) -> dict:
    mod = importlib.import_module(f"verity_capture.profiles.{name}")
    hf = getattr(mod, "HF_CONFIG", None)
    if hf is None and getattr(mod, "CONFIG_JSON", None):
        hf = json.loads(open(mod.CONFIG_JSON).read())
    return K.canonical(mod.PROFILE, mod, hf_config=hf)


def _diff(a, b, path="", out=None):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}.{k}: {'only hand' if k in a else 'only generic'}: {json.dumps(a.get(k, b.get(k)), default=repr)[:160]}")
            else:
                _diff(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} (hand) != {len(b)} (generic)")
        for i, (x, y) in enumerate(zip(a, b)):
            _diff(x, y, f"{path}[{i}]", out)
    elif a != b:
        out.append(f"{path}: hand {json.dumps(a, default=repr)[:160]} generic {json.dumps(b, default=repr)[:160]}")
    return out


@pytest.mark.parametrize("role,world", sorted(HAND), ids=[f"{r}_tp{w}" if w > 1 else r for r, w in sorted(HAND)])
def test_generic_equals_hand(role, world):
    older = HAND[(role, world)] in OLDER_TP2
    hand = _strip(_canon_hand(HAND[(role, world)]), older_tp2=older)
    gen = _strip(K.canonical_of_role(role, world), older_tp2=older)
    delta = _diff(hand, gen)
    assert not delta, f"{role} tp{world} ({HAND[(role, world)]}): {len(delta)} differences\n  " + "\n  ".join(delta[:40])


def test_declared_pattern_delta_is_exactly_the_tuned_gemm():
    """The only pattern the generic list has and a hand list lacks (or names differently) is the tuned-GEMM specialization."""
    for (role, world), name in sorted(HAND.items()):
        hand = [p["name"] for p in _canon_hand(name)["patterns"]]
        gen = [p["name"] for p in K.canonical_of_role(role, world)["patterns"]]
        assert [n for n in gen if n not in IGNORE_PATTERN_NAMES] == [n for n in hand if n not in IGNORE_PATTERN_NAMES], (role, world)
        assert set(gen) - set(hand) <= {"gemm.matmul_kernel_persistent.tuned"}, (role, world)
        assert set(hand) - set(gen) <= {"gemm.matmul_kernel_persistent.tuned_ada"}, (role, world)
