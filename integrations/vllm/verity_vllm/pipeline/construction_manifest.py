"""A derive dir's `construction_manifest.json` (`cb-a/construction_manifest/v3`), from the Build's result (`pipeline.build`)."""
from __future__ import annotations

import json
import os


def write_construction_manifest(out: str, res: dict) -> None:
    """`construction_manifest.json`: what the construction read (inputs trace summary), which rules fired, the declared
    facts it relied on, wall/RSS, and the program identity — for the independent validator."""
    tr = {}
    try:
        tr = json.load(open(os.path.join(out, "inputs.json")))
    except Exception:  # noqa: BLE001
        pass
    m = {"schema": "cb-a/construction_manifest/v3", "utc": res.get("utc"), "wrapper": res.get("wrapper"),
         # [X-05] `sampling` is the row's declared policy (manifest / --sampling / default-by-omission), never a literal
         "structural_inputs": {"LP": res.get("LP"), "T": res.get("T"), "M": res.get("M"), "pos0": res.get("pos0"), "eos": res.get("eos"),
                               "sampling": res.get("sampling"), "tp": (res.get("tp") or {}).get("world", 1), "rank": (res.get("tp") or {}).get("rank", 0),
                               "launch_context": res.get("launch_context")},
         # [R13 tp] world > 1: the rank's collectives (peer-input signature), embedding shard, engine args + env the admitted TP path requires,
         # the sharded parameter shapes and the cross-rank binding summary (full record: result.json `tp`, tp_links.json)
         "tp": ({k: v for k, v in res["tp"].items() if k != "collectives"} | {"collectives": {k: v for k, v in res["tp"].get("collectives", {}).items() if k != "record"}}
                if (res.get("tp") or {}).get("world", 1) > 1 else None),
         "model": {"class": res.get("model_class"), "hf_config": res.get("hf_config"), "pin": res.get("model_pin"),
                   # [R15 fp8] `hf_config.tie_word_embeddings` above is what vLLM SET for loading (False when the checkpoint ships an
                   # lm_head); the loader's re-tie decision, reproduced from the checkpoint's bytes, is the served model's fact
                   "lm_head_tying": res.get("lm_head_tying")},
         "code_rev": res.get("code_rev"), "profile": res.get("profile"),
         # v3: `target.declared` = the TargetProfile (the only source of every applied value); `target.applied` / `applied_after_export` =
         # read back from vLLM inside the applied shim; `target.host` = the construction host (informational, in no identity);
         # `target.consulted` = config facts + env split into consulted / ignored (dead in this pin) / non-vLLM
         "target": res.get("target"), "artifact": res.get("artifact"),
         "declared": (res.get("declared_derived") or {}).get("declared"), "derived": (res.get("declared_derived") or {}).get("derived"),
         "observed": {"attention_impls": res.get("attention_impls_observed"), "export_facts": res.get("export_facts")},
         "construction": {"mechanism": "torch.export(strict=False) of the vLLM model class on meta + verity_vllm.program.frontend.derive (profile mode)",
                          "export": res.get("export"), "derive_secs": res.get("derive_secs"), "wall_secs_total": res.get("wall_secs_total"),
                          "max_rss_mb": res.get("max_rss_mb"), "cuda_available": res.get("cuda_available"), "param_device": res.get("param_device"),
                          "real_inference": False, "weights_loaded": False},
         "rules_applied": res.get("rules_applied"), "n_applications": res.get("n_applications"), "bindings_rules": res.get("bindings_rules"),
         "inputs_trace": {"n_open_events": tr.get("n_open_events"), "distinct_paths": tr.get("distinct_paths"), "forbidden_hits": tr.get("forbidden_hits"),
                          "site_packages_by_package": tr.get("site_packages_by_package"),
                          "non_site_packages_paths": [p for p in tr.get("non_site_packages_paths", []) if not p.startswith("/usr/lib/python")]},
         "program": res.get("program"), "spec_histogram": res.get("spec_histogram"), "ok": res.get("ok"), "unsupported": res.get("unsupported"),
         "files": ["descriptor.json.gz (verity.ml.circuits descriptor v1)", "instances.json.gz (instance sequence: spec + operand run progressions)", "derive-report.json", "inputs.json", "result.json",
                   "artifact.json (verity/aot-artifact/v1: identity + construction_version + target_profile + applicability + program_digest)"]}
    if res.get("relabeled_from"):
        m["relabeled_from"] = res["relabeled_from"]
    json.dump(m, open(os.path.join(out, "construction_manifest.json"), "w"), indent=1, default=str)
