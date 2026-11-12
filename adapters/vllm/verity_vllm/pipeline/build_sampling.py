"""The sampling policy the Build constructs a row's Program under: the token-select kinds this builder binds
(`SAMPLERS_BOUND`), the row's declared policy as a structural input of the construction (`sampling_policy`), and the
pinned stochastic sampler's launch geometry, or its refusal (`sampler_geometry`).  `pipeline.build` reads the three
and hands `sampler_geometry` and `sampling_policy` on under its own name."""
from __future__ import annotations

import json

from verity_vllm.config import workload_file
from verity_vllm.program.frontend.seed_rule import rid_entry

__all__ = ["SAMPLERS_BOUND", "sampler_geometry", "sampling_policy"]


#: the token-select vocabulary of this builder: sampling policy kind -> the rule family that binds it.  A row whose policy is
#: not a key here is REFUSED before export (`reason: "sampler"`) — never built as the greedy Program (a fail-open Build).
SAMPLERS_BOUND = {"greedy": "ArgmaxTokenSelect_v1 (aten.argmax over the bf16 logits row -> TokenSelect_v1{V})",
                  # temperature > 0 with top-p, an explicit per-request seed and no top-k: the wrapper authors ONE
                  # `vllm_iface.gumbel_topp_token_select(logits, seed, temperature, top_p, pos, S)` per sampled step; the rule binds it to
                  # `GumbelTopPTokenSelect_v2{V,S}` (registry.sampling: f32 widening -> div.full.f32 temperature -> the S-split top-p mask
                  # `_topp_sb_{stats,step,mask}_kernel` -> Gumbel-max keyed `tl.randint(seed, pos)` -> first maximum).  The seed is a runtime
                  # INPUT of the Program; temperature / top_p / pos and the registered S (`sampler_geometry`) are constants, S the 6th operand.
                  "stochastic": "GumbelTopPTokenSelect[vllm]_v4 (vllm_iface.gumbel_topp_token_select -> GumbelTopPTokenSelect_v2{V,S} with the seed input and the registered S: its keep word as word gates)"}


def sampler_geometry(sampling: dict, target, LP: int | None = None, T: int | None = None) -> dict | None:
    """The launch geometry of the pinned stochastic sampler for this row, or the refusal (`unsupported` dict) when the policy has no
    Definition, or a split count the session declares that isn't the registered one.  On success `sampling["splits"]` is the registered
    split count S and `sampling["geometry"]` records it with its rule:
    nothing of the workload's batch, so every component of every workload on the target declares the same (GP-01 agreement by
    construction).

    vLLM d9105ea80 `apply_top_k_top_p` for p-only requests and a sampler batch <= 64 runs `_apply_topp_split`, whose keep set depends
    on its split count S.  vLLM picks S from the live sampler batch, so a request's batch-mates would pick its nucleus at top_p < 1
    (as `check.batch.check` over `predict.batch.templates` shows).  The served engine runs every launch at the registered S instead
    (`topp_geometry.registered_splits`, from the target's SM count, never hard-coded; `engine.topp_pin`),
    so the Program takes S as a per-event constant (`Const32[S]`, the select's 6th operand) and reads no live row count; the Match
    holds every launch's S to it."""
    from verity.definitions.registry.topp_geometry import (
        SPLIT_MAX_BATCH,
        registered_splits,
    )
    kind = sampling.get("kind")
    if kind != "stochastic":
        return None
    def refuse(hint: str, reason: str = "sampler", **detail) -> dict:
        return {"reason": reason, "phase": "applicability", "node": None, "module_path": None, "hint": hint,
                "detail": {"sampling": {k: v for k, v in sampling.items() if k != "workload"}, "samplers_bound": SAMPLERS_BOUND, **detail}}
    if int(sampling.get("top_k", 0) or 0) != 0:
        return refuse(f"top_k = {sampling.get('top_k')}: vLLM's top-k path (`_topk_topp_kernel` / `_apply_topk_topp`) has no exact Definition; "
                      f"only the p-only split pipeline (`_apply_topp_split`) is modelled (TopPMask_v1{{V}})", missing=["TopKMask"])
    if not sampling.get("seeded", False):
        return refuse("the policy has no explicit per-request seed: vLLM picks a random seed when `SamplingParams.seed` is None "
                      "(SamplingStates.add_request), so the RNG stream is not a declared input and the sampled token is not reproducible "
                      "from the manifest — declare `seed` per request (sweep `sampling_seeds`)", missing=["declared seed"])
    reqs = []
    wl = sampling.get("workload")
    if wl:
        try:
            w = json.loads(workload_file(wl).read_text())
            reqs = list(stated) if (stated := w.get("requests")) else []   # a workload that states none: a batch of one
        except Exception as e:  # noqa: BLE001
            return refuse(f"cannot read the workload manifest {wl!r} for the sampler batch: {e!r}", reason="sampler_geometry")
    B = max(1, len(reqs))
    if B > SPLIT_MAX_BATCH:
        return refuse(f"sampler batch {B} > _SPLIT_MAX_BATCH {SPLIT_MAX_BATCH}: vLLM runs `_topk_topp_kernel`, not the split pipeline", reason="sampler_geometry")
    if getattr(target, "num_sms", None) is None:
        return refuse("the target profile declares no num_sms: the registered split count registered_splits(num_SMs) has no num_SMs",
                      reason="sampler_geometry")
    nsm = int(target.num_sms)
    S = int(registered_splits(nsm))
    if sampling.get("splits", S) != S or type(sampling.get("splits", S)) is not int:
        return refuse(f"unregistered split count: the session declares top-p split count {sampling['splits']!r}, not the registered one, "
                      f"registered_splits({nsm}) = {S}: serving runs every launch at it (engine.topp_pin), and the Program carries it "
                      f"as Const32[{S}]", reason="sampler_geometry")
    sampling["splits"] = S
    sampling["geometry"] = {"num_sms": nsm, "pipeline": "_apply_topp_split", "relation": "registered", "splits": S,
                            "rule": "S = registered_splits(num_SMs) at every sampling event, whatever the live "
                                    "batch: the served engine pins vLLM's split count to it (engine.topp_pin)",
                            "constant": f"Const32[{S}]: the select's 6th operand at every sampling event (no `splits` input)"}
    return None


def sampling_policy(arg: str | None, workload: str | None) -> dict:
    """The row's sampling policy as a STRUCTURAL INPUT of the construction (as are prompt length and max_tokens).

    Sources, in order: `--workload <manifest>` (the row's manifest: `sweep.sampling` + `sweep.sampling_params` from `sweep/workload.py`,
    else the legacy `sampling.variant` -> `variants[...]`, else per-request `variant` / `seed` fields), and/or `--sampling <policy>` with
    `greedy` | `stoch-t<T>-p<P>[-k<K>]` | a JSON object.  Both given -> they must agree (refused otherwise).  Neither given -> greedy,
    recorded as `source: "default"` (a caller without a manifest declares greedy by omission; the row procedure passes the manifest).
    Returns {"kind": "greedy"} or {"kind": "stochastic", "temperature", "top_p", "top_k", "seeded", "label"} (+ "source", "mixed", "rid": the id length the seed rule reads, `seed_rule.rid_entry`)."""
    def _norm(params: dict, label: str | None = None) -> dict:
        t = float(params.get("temperature", 1.0))
        if t == 0.0:
            return {"kind": "greedy"}
        return {"kind": "stochastic", "temperature": t, "top_p": float(params.get("top_p", 1.0)), "top_k": int(params.get("top_k", 0) or 0),
                "seeded": bool(params.get("seeded", True)), "label": label or f"stoch-t{t:g}-p{float(params.get('top_p', 1.0)):g}", **({"rid": int(params["rid"])} if params.get("rid") is not None else {})}

    def _parse_arg(s: str) -> dict:
        s = s.strip()
        if s.startswith("{"):
            return _norm(json.loads(s))
        if s == "greedy":
            return {"kind": "greedy"}
        import re
        m = re.fullmatch(r"stoch-t([0-9.]+)-p([0-9.]+)(?:-k([0-9]+))?", s)
        if not m:
            raise ValueError(f"unknown sampling policy {s!r} (greedy | stoch-t<T>-p<P>[-k<K>] | JSON)")
        return _norm({"temperature": float(m.group(1)), "top_p": float(m.group(2)), "top_k": int(m.group(3) or 0)}, s)

    def _from_manifest(path: str) -> dict:
        with open(path) as f:
            w = json.load(f)
        sw = w.get("sweep")   # a generated workload's block; a row file has none
        if sw and sw.get("sampling") is not None:
            pol = _norm(dict(params) if (params := sw.get("sampling_params")) else {}, sw["sampling"]) if sw["sampling"] != "greedy" else {"kind": "greedy"}
            if sw.get("sampling_seeds"):
                pol["seeded"] = True
            return pol
        named = w.get("variants")   # the legacy form's named policies; a workload that names none: every request's is its own fields'
        variants: dict = named if named else {}
        top = block.get("variant") if (block := w.get("sampling")) else None
        pols = []
        for r in w.get("requests", []):
            v = r.get("variant", top)
            p = variants.get(v) if v is not None else None
            if p is None:
                p = {"temperature": 0.0} if r.get("seed") is None and v in (None, "greedy") else {"temperature": 1.0}
            pols.append(_norm(dict(p), v))
        if not pols:
            return _norm(dict(variants.get(top) or {"temperature": 0.0}), top) if top else {"kind": "greedy"}
        kinds = {json.dumps(p, sort_keys=True) for p in pols}
        if len(kinds) > 1:
            return {"kind": "mixed", "policies": [json.loads(k) for k in sorted(kinds)]}
        return pols[0]

    pol_w = _from_manifest(workload) if workload else None
    pol_a = _parse_arg(arg) if arg else None
    if pol_w is not None and pol_a is not None:
        a, b = ({k: v for k, v in p.items() if k not in ("label", "source")} for p in (pol_w, pol_a))
        if a != b:
            raise ValueError(f"--sampling {arg!r} disagrees with the manifest's sampling policy {pol_w} ({workload})")
    pol = dict(pol_w if pol_w is not None else (pol_a if pol_a is not None else {"kind": "greedy"}))
    pol["source"] = "workload" if pol_w is not None else ("arg" if pol_a is not None else "default")
    if workload:
        with open(workload) as f:
            pol.update(workload=workload, **rid_entry(pol, json.load(f).get("requests", [])))
    return pol
