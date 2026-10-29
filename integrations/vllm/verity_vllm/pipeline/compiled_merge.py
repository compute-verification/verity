"""Merge the control-only and treatment-only commit_delta outputs of an Execution=compiled row into ONE paired summary
(summary.json / summary.md / runs.jsonl / verdict.json in --out), in the shape row_pod.sh's Commit phase reads.

The two arms are separate processes because the treatment's record ops live inside the compiled graphs for the whole process; pairing is
by run index (control run i <-> treatment run i).  Refuses when the two processes' compilation settings differ (CompilationMode,
CUDAGraphMode, capture sizes, enforce_eager) or when the generated tokens differ (tokens_digest), so an eager control cannot be paired with
a compiled treatment (and vice versa)."""
from __future__ import annotations

import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from verity_vllm.config import option


def _rows(p: Path) -> list[dict]:
    f = p / "runs.jsonl"
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()] if f.exists() else []


def _json(p: Path, name: str) -> dict:
    f = p / name
    return json.loads(f.read_text()) if f.exists() else {}


def _bootstrap_ci(xs: list[float], n: int = 4000, *, seed: int) -> tuple[float, float]:
    from verity_vllm.commit import challenge as CH

    if len(xs) < 2:
        return (float("nan"), float("nan"))
    rng = CH.stream(seed)
    means = sorted(statistics.fmean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


COMPILE_KEYS = ("compilation_mode", "cudagraph_mode", "cudagraph_capture_sizes", "enforce_eager", "use_inductor_graph_partition", "splitting_ops",
                "inductor_compile_config")


@dataclass(frozen=True, kw_only=True)
class Options:
    control: list[str] = option("--control", action="append",
                                help="control-process output dir (repeatable: one per round, in run order)")
    treatment: list[str] = option("--treatment", action="append", help="treatment-process output dir (repeatable)")
    out: str = option("--out")
    manifest_control: str | None = option("--manifest-control", default=None,
                                          help="P_compiled manifest built from the CONTROL process's Inductor output code (taps=kernels)")
    manifest_treatment: str | None = option("--manifest-treatment", default=None,
                                            help="the same derivation from the TREATMENT process's output code (must agree on program_digest)")


def main(a: Options) -> None:
    Cs, Ts, out = [Path(x) for x in a.control], [Path(x) for x in a.treatment], Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ctl = [r for C in Cs for r in _rows(C)]
    trt = [r for T in Ts for r in _rows(T)]
    for i, r in enumerate(ctl):
        r["pair"] = i
    for i, r in enumerate(trt):
        r["pair"] = i
    C, T = Cs[0], Ts[0]
    vc, vt = _json(C, "versions.json"), _json(T, "versions.json")
    cc, ct = vc.get("compilation", {}), vt.get("compilation", {})
    problems = []
    for X in Cs[1:] + Ts[1:]:                                   # every process of both arms must report the same compilation settings
        cx = _json(X, "versions.json").get("compilation", {})
        for k in COMPILE_KEYS:
            if cx.get(k) != ct.get(k):
                problems.append(f"compilation setting differs in {X.name}: {k}: {cx.get(k)!r} vs {ct.get(k)!r}")
    rounds = len(Cs)
    if not ctl:
        problems.append("control: no runs")
    if not trt:
        problems.append("treatment: no runs")
    for k in COMPILE_KEYS:
        if cc.get(k) != ct.get(k):
            problems.append(f"compilation setting differs: {k}: control {cc.get(k)!r} vs treatment {ct.get(k)!r}")
    if cc.get("enforce_eager") is not False:
        problems.append(f"control is not compiled (enforce_eager={cc.get('enforce_eager')!r})")
    if ct.get("compiled_taps") not in ("on", "kernels", "served"):  # kernels = the S3 path of record (P_compiled kernel buffers); on = withdrawn record ops
        problems.append("treatment process did not run with --compiled-taps on|kernels|served")
    if cc.get("compiled_taps") not in (None, "off"):
        problems.append("control process ran with compiled taps")
    # the generated-kernel identity of both processes: the Triton kernel SOURCES Inductor emitted (kernel_set_sha256 over
    # {(name, source sha256)}) and the per-piece launch programs.  The treatment's pieces carry the record op as an extern call, so its
    # launch programs (call_set) legitimately differ; its KERNEL SET should equal the control's -- a difference means the record op changed
    # what Inductor fused, and is listed kernel by kernel (reported, labelled; not a refusal: the workload's implementation is then the treatment's).
    gkc, gkt = cc.get("generated_kernels") or {}, ct.get("generated_kernels") or {}
    kernels_identified = bool(gkc.get("kernel_set_sha256")) and bool(gkt.get("kernel_set_sha256"))
    kernel_set_equal = kernels_identified and gkc.get("kernel_set_sha256") == gkt.get("kernel_set_sha256")
    kc, kt = gkc.get("kernels") or {}, gkt.get("kernels") or {}
    kernel_diff = {"only_control": sorted(set(kc) - set(kt))[:40], "only_treatment": sorted(set(kt) - set(kc))[:40],
                   "same_name_different_source": sorted(k for k in set(kc) & set(kt) if kc[k] != kt[k])[:40]} if kernels_identified else None
    # on the P_compiled path (taps=kernels) the manifest built from the CONTROL's Inductor dir gates the TREATMENT's
    # commit, and the kernels are untouched by the hooks (wrapper-level seam): the kernel set must be identical across arms, and the
    # PROGRAM derived from each arm's own output code (compiled_manifest.build: kernel set x launch table x dataflow = program_digest) must
    # be the same Program -- else the Program of record is not the executed implementation: a refusal, not a label.  The raw wrapper text
    # (call_set_sha256) legitimately differs: the treatment's wrappers carry Inductor's own debug-printer statements after each launch
    # (the seam the hooks ride), which is not a launch, so it is reported, not compared.
    program_digests = None
    loaded_identity = None
    # served: the Build of record is derived from the treatment's own Inductor artifact and rebuilt after its runs (manifest_verify); the
    # control must have run the same generated kernels, else the record is not the served implementation
    if ct.get("compiled_taps") == "served":
        if not kernels_identified:
            problems.append("served path: generated kernels not identified in both arms (inductor_kernels.json)")
        elif not kernel_set_equal:
            problems.append(f"served path: the treatment's generated kernel set differs from the control's: {kernel_diff}")
    if ct.get("compiled_taps") == "kernels":
        mc = json.loads(Path(a.manifest_control).read_text()) if a.manifest_control and Path(a.manifest_control).is_file() else None
        mt = json.loads(Path(a.manifest_treatment).read_text()) if a.manifest_treatment and Path(a.manifest_treatment).is_file() else None
        program_digests = {"control": (mc or {}).get("program_digest"), "treatment": (mt or {}).get("program_digest"),
                           "manifest_control": (mc or {}).get("manifest_digest"), "manifest_treatment": (mt or {}).get("manifest_digest"),
                           "equal": bool(mc and mt and mc.get("program_digest") and mc.get("program_digest") == mt.get("program_digest"))}
        if not kernels_identified:
            problems.append("P_compiled path: generated kernels not identified in both arms (inductor_kernels.json)")
        elif not kernel_set_equal:
            problems.append(f"P_compiled path: the treatment's generated kernel set differs from the control's: {kernel_diff}")
        elif mc is None or mt is None:
            problems.append("P_compiled path: P_compiled not derived from BOTH arms' Inductor output code (--manifest-control/--manifest-treatment)")
        elif not program_digests["equal"]:
            problems.append(f"P_compiled path: the Program derived from the treatment's output code differs from the control's: {program_digests}")
        # runtime <-> pinned artifact identity: the code objects LOADED in each running process (PyCodeCache registry:
        # Triton sources + cubins) must be one set across arms, and every kernel P_compiled names must be among the loaded objects
        lc, lt = gkc.get("loaded") or {}, gkt.get("loaded") or {}
        # refusal = the LOADED KERNEL SOURCES (the code objects) differ; the binaries (cubin sha256 where Inductor retains the bytes, else
        # Triton's compiled-kernel hash, which also covers compile options/env) are recorded per kernel and their equality LABELLED --
        # autotune candidates (compile_results) and env-dependent Triton hashes can differ between processes that run the same code objects
        loaded_ok = bool(lc.get("kernel_set_sha256")) and lc.get("kernel_set_sha256") == lt.get("kernel_set_sha256")
        pc, pt = lc.get("per_kernel") or {}, lt.get("per_kernel") or {}
        binaries_differ = sorted(k for k in set(pc) | set(pt) if (pc.get(k) or {}).get("binaries") != (pt.get(k) or {}).get("binaries"))
        binaries_equal = bool(pc) and not binaries_differ
        _lt = ((mc or {}).get("program") or {}).get("launch_table") or {}
        manifest_kernels = sorted({str(r.get("kernel")) for rows in (_lt.values() if isinstance(_lt, dict) else _lt) for r in rows if isinstance(r, dict)})
        # P_compiled names kernels by LAUNCH SITE (Inductor's wrapper symbol: matmul_kernel_persistent_0..3 for the 4 GEMM sites of a template
        # kernel); the loaded code object's identity is the JIT function name inside the source (matmul_kernel_persistent) -- strip the site suffix
        import re as _re
        _loaded_names = set(lc.get("kernel_names") or [])
        not_loaded = sorted(k for k in manifest_kernels if k not in _loaded_names and _re.sub(r"_\d+$", "", k) not in _loaded_names)
        # the EXECUTED specialization per kernel: code_sha256 = every ELF section of the executed cubin EXCEPT the
        # named deny-list (debug/line info, embedded PTX text, symbol/string tables) -- SASS, constant banks, initialised globals, rodata all
        # count; the executed autotune config; and -- where the arms' cubins differ -- WHICH sections differ, DECODED: a non-deny-listed
        # section may differ across arms only if its bytes are equal once absolute-path strings are normalised (per-process source path
        # embedded in device data by tl.device_assert) and the record carries those strings; any other non-deny-listed difference REFUSES.
        def _exec(pk: dict) -> list[dict]:
            return [e for e in (pk.get("executed") or []) if isinstance(e, dict)]

        def _cfg_key(e: dict) -> str:
            return json.dumps(e.get("config"), sort_keys=True)

        _DENY_PREFIX = (".debug", ".rel.debug", ".rela.debug", ".nv.debug", ".nv_debug", ".rel.nv_debug", ".rela.nv_debug", ".rel.nv.debug", ".rela.nv.debug")   # = vllm_adapter.CUBIN_DENY_LIST
        _DENY_EXACT = (".symtab", ".strtab", ".shstrtab")

        def _denied(s: str) -> bool:
            return s in _DENY_EXACT or s.startswith(_DENY_PREFIX)

        # a path-string difference is decodable ONLY in a DATA section (initialised globals, constant banks, rodata): SASS
        # (.text*), kernel info (.nv.info*) and relocations never legitimately embed the module path, so a difference there is never
        # 'per-process path' whatever its bytes look like once a slash-led run is normalised.  And every string must BE the generated
        # module's path: ends with .py or .py:<line>, and is a loaded module path of its arm (modules_list) or lives under the inductor cache.
        _DATA_SECTIONS = (".nv.global", ".nv.constant", ".rodata")
        _MODULE_PATH_RE = _re.compile(r"\.py(:\d+)?$")
        _module_paths_c = {str(m.get("path")) for m in (lc.get("modules_list") or []) if isinstance(m, dict) and m.get("path")}
        _module_paths_t = {str(m.get("path")) for m in (lt.get("modules_list") or []) if isinstance(m, dict) and m.get("path")}

        def _is_module_path(p: str, arm_paths: set[str]) -> bool:
            if not isinstance(p, str) or not p.startswith("/") or " " in p or not _MODULE_PATH_RE.search(p):
                return False
            base = _re.sub(r":\d+$", "", p)
            return base in arm_paths or "inductor_cache" in base or "/torchinductor" in base.lower()

        def _decode_pair(e1: dict, e2: dict) -> tuple[list[str], list[dict], list[str]]:
            """(sections differing, decoded non-denied differences, UNEXPLAINED non-denied sections) for one executed-binary pair."""
            s1, s2 = e1.get("sections") or {}, e2.get("sections") or {}
            diff = sorted(s for s in set(s1) | set(s2) if s1.get(s) != s2.get(s))
            d1, d2 = e1.get("section_detail") or {}, e2.get("section_detail") or {}
            decoded, unexplained = [], []
            for s in diff:
                if _denied(s):
                    continue
                a, b = d1.get(s) or {}, d2.get(s) or {}
                res_eq = bool(a.get("residue_sha256")) and a.get("residue_sha256") == b.get("residue_sha256")
                pa, pb = list(a.get("path_strings") or []), list(b.get("path_strings") or [])
                data_section = s.startswith(_DATA_SECTIONS)
                module_paths = bool(pa) and bool(pb) and all(_is_module_path(p, _module_paths_c) for p in pa) and all(_is_module_path(p, _module_paths_t) for p in pb)
                ok = res_eq and data_section and module_paths and len(pa) == len(pb) and pa != pb \
                    and int(a.get("n_path_strings") or 0) == int(b.get("n_path_strings") or 0) and int(a.get("n_path_strings") or 0) <= 16
                kind = ("per-process absolute-path strings in device data; bytes equal once normalised" if ok
                        else ("section_detail not recorded (older record) -> fail closed" if not a or not b
                              else (f"{s} is not a data section: a difference there is never a path string -> not one specialization" if res_eq and not data_section
                                    else ("strings are not the generated module's .py path -> NON-PATH DATA DIFFERS" if res_eq and not module_paths
                                          else "NON-PATH DATA DIFFERS -> not one specialization"))))
                decoded.append({"section": s, "residue_equal": res_eq, "data_section": data_section, "module_paths": module_paths,
                                "size_control": a.get("size"), "size_treatment": b.get("size"),
                                "path_strings_control": pa[:8], "path_strings_treatment": pb[:8], "kind": kind})
                if not ok:
                    unexplained.append(s)
            return diff, decoded, unexplained

        exec_detail: dict[str, Any] = {}
        code_differ, code_differ_unexplained, code_equal_modulo_paths, config_differ, code_equal_denied_differ = [], [], [], [], []
        for k in sorted(set(pc) | set(pt)):
            ec, et = _exec(pc.get(k) or {}), _exec(pt.get(k) or {})
            if not ec and not et:
                continue
            cc_codes, ct_codes = sorted({e.get("code_sha256") for e in ec if e.get("code_sha256")}), sorted({e.get("code_sha256") for e in et if e.get("code_sha256")})
            cc_norm, ct_norm = sorted({e.get("code_sha256_normalized") for e in ec if e.get("code_sha256_normalized")}), sorted({e.get("code_sha256_normalized") for e in et if e.get("code_sha256_normalized")})
            cc_cfg, ct_cfg = sorted({_cfg_key(e) for e in ec}), sorted({_cfg_key(e) for e in et})
            cc_bytes, ct_bytes = sorted({e.get("cubin_sha256") for e in ec if e.get("cubin_sha256")}), sorted({e.get("cubin_sha256") for e in et if e.get("cubin_sha256")})
            sec_diff, decoded_all, unexplained_all = [], [], []
            for e1 in ec:
                for e2 in et:
                    if _cfg_key(e1) == _cfg_key(e2) and e1.get("sections") and e2.get("sections") and e1.get("cubin_sha256") != e2.get("cubin_sha256"):
                        diff, decoded, unexplained = _decode_pair(e1, e2)
                        sec_diff += diff
                        decoded_all += decoded
                        unexplained_all += unexplained
            code_equal = bool(cc_codes) and cc_codes == ct_codes
            code_equal_norm = bool(cc_norm) and cc_norm == ct_norm
            d = {"control": {"code": [c[:16] for c in cc_codes], "code_normalized": [c[:16] for c in cc_norm], "configs": [json.loads(c) for c in cc_cfg], "cubin": [c[:16] for c in cc_bytes],
                             "triton": sorted({str(e.get("triton_hash"))[:16] for e in ec if e.get("triton_hash")}),
                             "compile_meta": [e.get("compile_meta") for e in ec][:2], "candidates": (pc.get(k) or {}).get("n_configs_candidate")},
                 "treatment": {"code": [c[:16] for c in ct_codes], "code_normalized": [c[:16] for c in ct_norm], "configs": [json.loads(c) for c in ct_cfg], "cubin": [c[:16] for c in ct_bytes],
                               "triton": sorted({str(e.get("triton_hash"))[:16] for e in et if e.get("triton_hash")}),
                               "compile_meta": [e.get("compile_meta") for e in et][:2], "candidates": (pt.get(k) or {}).get("n_configs_candidate")},
                 "code_equal": code_equal, "code_equal_normalized": code_equal_norm, "config_equal": cc_cfg == ct_cfg,
                 "cubin_bytes_equal": bool(cc_bytes) and cc_bytes == ct_bytes, "sections_differing": sorted(set(sec_diff))[:20],
                 "sections_differing_denied": sorted({s for s in sec_diff if _denied(s)})[:20], "sections_differing_non_denied": sorted({s for s in sec_diff if not _denied(s)})[:20],
                 "device_data_path_strings": decoded_all[:8], "non_denied_unexplained": sorted(set(unexplained_all))[:20],
                 "autotuned": bool((pc.get(k) or {}).get("autotuned") or (pt.get(k) or {}).get("autotuned"))}
            # fail closed: a NON-deny-listed section that differs and is not decoded as a path-string difference makes the pair DIFFER even when
            # the recorded code digest agrees (records written under the earlier .text-only rule carry section hashes but no section_detail)
            d["identity_verdict"] = ("DIFFERS" if unexplained_all else
                                     ("equal" if code_equal else
                                      ("equal modulo per-process absolute-path strings in device data (decoded)" if (code_equal_norm and cc_codes and ct_codes and decoded_all) else
                                       ("DIFFERS" if (cc_codes and ct_codes) else "no executed code recorded"))))
            exec_detail[k] = d
            if (cc_codes and ct_codes and cc_codes != ct_codes) or unexplained_all:
                code_differ.append(k)
                if d["identity_verdict"] == "DIFFERS":
                    code_differ_unexplained.append(k)
                else:
                    code_equal_modulo_paths.append(k)
            if cc_cfg != ct_cfg:
                config_differ.append(k)
            if code_equal and cc_bytes and ct_bytes and cc_bytes != ct_bytes:
                code_equal_denied_differ.append(k)
        # [the compiled-kernel autotune rule] a kernel whose arms executed DIFFERENT autotune configs of the SAME source is two
        # specializations -- lifted only when the treatment process SHOWED, on committed inputs of sampled launches, that the kernel's own
        # compiled source under EVERY candidate config of its heuristic reproduces the committed outputs bit for bit (compiled_autotune),
        # every treatment run agreeing, the control's executed config(s) among the candidates, and the sources equal; else it stays DIFFERS.
        def _autotune_records(k: str) -> list[dict]:
            out = []
            for r in trt:
                ae = ((((r.get("commit") or {}).get("value_correspondence") or {}).get("compiled_check") or {}).get("autotune_equivalence") or {})
                if isinstance(ae, dict) and isinstance(ae.get(k), dict):
                    out.append(ae[k])
            return out

        autotune_shown: dict[str, dict] = {}
        for k in list(code_differ_unexplained):
            d = exec_detail.get(k) or {}
            recs = _autotune_records(k)
            src_equal = bool((pc.get(k) or {}).get("src_sha256")) and (pc.get(k) or {}).get("src_sha256") == (pt.get(k) or {}).get("src_sha256")
            if not recs or d.get("config_equal") or not src_equal or d.get("non_denied_unexplained"):
                continue
            cands = [json.dumps(c, sort_keys=True) for c in (recs[0].get("candidates") or [])]
            ctrl_in = bool(d.get("control", {}).get("configs")) and all(json.dumps(c, sort_keys=True) in cands for c in d["control"]["configs"])
            trt_in = bool(d.get("treatment", {}).get("configs")) and all(json.dumps(c, sort_keys=True) in cands for c in d["treatment"]["configs"])
            # structural = POINTWISE heuristic (one thread per element; the config cannot change the arithmetic); a reduction's
            # config sets the summation order, so its sampled equality is recorded but never lifts -- DIFFERS stands
            structural = bool(recs) and all(str(r.get("heuristic") or "").endswith("POINTWISE") for r in recs)
            shown = all(bool(r.get("all_shown")) and int(r.get("launches") or 0) > 0 for r in recs) and len(cands) > 1 and ctrl_in and trt_in
            if shown and structural:
                code_differ_unexplained.remove(k)
                autotune_shown[k] = {"candidates": recs[0].get("candidates"), "control_configs": d["control"]["configs"], "treatment_configs": d["treatment"]["configs"],
                                     "launches_shown": sum(int(r.get("launches") or 0) for r in recs), "treatment_runs": len(recs),
                                     "exact_by_config": recs[0].get("exact_by_config"), "derived_scalars": recs[0].get("derived_scalars_sample"),
                                     "heuristic": recs[0].get("heuristic"), "structural": True,
                                     "rule": "pointwise kernel (config independence structural: one thread per element) AND every candidate config of the kernel's heuristic reproduces the committed outputs bit for bit from the committed inputs (the executed config included); the autotune choice does not affect the numerics -> not part of the specialization identity, recorded"}
                d["identity_verdict"] = "equal modulo autotune config (pointwise; every candidate shown output-equivalent on committed inputs)"
                d["autotune_equivalence"] = autotune_shown[k]
            else:
                d["autotune_equivalence"] = {"shown": False, "structural": structural, "heuristic": recs[0].get("heuristic"), "candidates": recs[0].get("candidates"),
                                             "control_in_candidates": ctrl_in, "treatment_in_candidates": trt_in,
                                             "why": (None if structural else "non-pointwise heuristic -- the config sets the reduction order; a different config is a different specialization (DIFFERS stands)"),
                                             "runs": [{"launches": r.get("launches"), "shown": r.get("shown"), "why": r.get("why")} for r in recs][:6]}
        # in P_compiled (launch-site names -> JIT names)
        _pk = {_re.sub(r"_\d+$", "", k) for k in manifest_kernels} | set(manifest_kernels)
        text_differ = code_differ                       # name kept for readers of earlier records: 'code' now = every non-deny-listed section
        text_differ_in_program = [k for k in code_differ if k in _pk]
        code_differ_unexplained_in_program = [k for k in code_differ_unexplained if k in _pk]
        autotune_shown_in_program = [k for k in autotune_shown if k in _pk]
        config_differ_in_program = [k for k in config_differ if k in _pk]
        text_equal_bytes_differ = code_equal_denied_differ
        rule_recorded = bool(lc.get("code_identity_rule")) and bool(lt.get("code_identity_rule"))     # else the digests are the earlier .text-only rule
        strict_equal = rule_recorded and bool(lc.get("executed_code_set_sha256")) and lc.get("executed_code_set_sha256") == lt.get("executed_code_set_sha256") and not code_differ_unexplained
        norm_equal = bool(lc.get("executed_code_set_sha256_normalized")) and lc.get("executed_code_set_sha256_normalized") == lt.get("executed_code_set_sha256_normalized")
        modulo_paths_equal = rule_recorded and norm_equal and not code_differ_unexplained and bool(code_equal_modulo_paths) and all(
            k in code_equal_modulo_paths for k in code_differ)       # an autotune-lifted kernel is named by the autotune verdict, not hidden here
        # the set digests cannot agree when an autotune config differs (different SASS): equality is then per kernel -- every differing kernel
        # either decoded (path strings) or shown (autotune), none unexplained
        modulo_autotune_equal = rule_recorded and not code_differ_unexplained and bool(autotune_shown) and all(
            k in autotune_shown or k in code_equal_modulo_paths for k in code_differ)
        _autotune_text = (f"equal modulo autotune config of {len(autotune_shown)} kernel(s) {sorted(autotune_shown)[:6]} (every candidate config shown "
                          f"output-equivalent on committed inputs: executed[k].autotune_equivalence)"
                          + (f" + path strings in device data of {len(code_equal_modulo_paths)} kernel(s)" if code_equal_modulo_paths else ""))
        identity_verdict = ("equal" if strict_equal else
                            (f"equal modulo per-process absolute-path strings in device data of {len(code_equal_modulo_paths)} kernel(s) (decoded on record: executed[k].device_data_path_strings)"
                             if modulo_paths_equal else
                             (_autotune_text if modulo_autotune_equal else
                              (f"DIFFERS: {code_differ_unexplained[:8]}" if code_differ_unexplained else
                               ("not recorded" if not lc.get("executed_code_set_sha256") else
                                (f"recorded under the earlier .text-only rule (no code_identity_rule / section_detail): not the deny-list identity" if not rule_recorded
                                 else f"differs: {code_differ[:8]}"))))))
        # repeated graph replay covered: each arm's build-time snapshot == its end-of-run snapshot (commit_delta writes replay_covered)
        rc_c, rc_t = gkc.get("replay_covered") or {}, gkt.get("replay_covered") or {}
        replay_covered = {"control": {k: rc_c.get(k) for k in ("ok", "why", "new_modules", "new_kernels", "executed_changed", "binaries_changed")} if rc_c else None,
                          "treatment": {k: rc_t.get(k) for k in ("ok", "why", "new_modules", "new_kernels", "executed_changed", "binaries_changed")} if rc_t else None,
                          "ok": bool(rc_c) and bool(rc_t) and rc_c.get("ok") is True and rc_t.get("ok") is True}
        _keys = ("modules", "kernels", "kernels_with_binaries", "kernel_set_sha256", "cubin_set_sha256", "module_keys_sha256", "executed_set_sha256",
                 "executed_code_set_sha256", "executed_code_set_sha256_normalized", "kernels_with_executed_code", "kernels_autotuned")
        loaded_identity = {"control": {k: lc.get(k) for k in _keys}, "treatment": {k: lt.get(k) for k in _keys},
                           "code_identity_rule": lc.get("code_identity_rule") or lt.get("code_identity_rule"),
                           "equal_across_arms": loaded_ok, "binaries_equal_across_arms": binaries_equal, "kernels_whose_binaries_differ": binaries_differ[:40],
                           "binaries_differ_detail": {k: {"control": (pc.get(k) or {}).get("binaries"), "treatment": (pt.get(k) or {}).get("binaries")} for k in binaries_differ[:8]},
                           "executed": exec_detail,
                           # STRICT: the deny-list identity of the executed set is byte-equal across arms (every non-deny-listed section)
                           "executed_code_equal_across_arms": strict_equal,
                           # DECODED: equal once per-process absolute-path strings in device data are normalised, every such difference on record
                           "executed_code_equal_modulo_device_path_strings": modulo_paths_equal,
                           "executed_identity_verdict": identity_verdict,
                           "executed_code_differs": code_differ[:40], "executed_code_differs_in_program": text_differ_in_program[:40],
                           "executed_code_differs_unexplained": code_differ_unexplained[:40], "executed_code_differs_unexplained_in_program": code_differ_unexplained_in_program[:40],
                           "executed_code_equal_modulo_path_strings": code_equal_modulo_paths[:40],
                           # AUTOTUNE: equal modulo the autotune config of kernels shown output-equivalent under every candidate (compiled_autotune)
                           "executed_code_equal_modulo_autotune": modulo_autotune_equal,
                           "executed_code_equal_modulo_autotune_kernels": autotune_shown_in_program[:40], "autotune_equivalence": autotune_shown,
                           "executed_text_differs": text_differ[:40], "executed_text_differs_in_program": text_differ_in_program[:40],   # legacy names (= code_differs)
                           "executed_config_differs": config_differ[:40], "executed_config_differs_in_program": config_differ_in_program[:40],
                           "code_equal_denied_sections_differ": code_equal_denied_differ[:40],
                           "text_equal_cubin_bytes_differ": text_equal_bytes_differ[:40],                                                 # legacy name
                           "replay_covered": replay_covered,
                           "manifest_kernels": len(manifest_kernels), "manifest_kernels_not_loaded": not_loaded[:40],
                           "read_from": lc.get("read_from"), "files_vs_loaded": lc.get("files_vs_loaded")}
        if loaded_ok is False or not_loaded:
            problems.append(f"P_compiled path: runtime<->artifact identity: loaded code objects equal across arms {loaded_ok}, "
                            f"manifest kernels not among the loaded objects {not_loaded[:8]}")
        if code_differ_unexplained_in_program:
            problems.append(f"P_compiled path: a NON-deny-listed ELF section of the executed cubin of a P_compiled kernel differs across arms and "
                            f"is not a decoded path-string difference: {code_differ_unexplained_in_program[:8]} "
                            f"{[exec_detail[k].get('non_denied_unexplained') for k in code_differ_unexplained_in_program[:4]]} -> not one specialization")
        if text_differ_in_program and not code_differ_unexplained_in_program and not modulo_paths_equal and not modulo_autotune_equal:
            problems.append(f"P_compiled path: executed code of a P_compiled kernel differs across arms ({text_differ_in_program[:8]}) and the "
                            f"normalised set digests do not agree -> not one specialization")
        if replay_covered["control"] is not None and replay_covered["treatment"] is not None and not replay_covered["ok"]:
            problems.append(f"P_compiled path: loaded code objects changed between build and end of run (replay not covered): control {rc_c.get('why')}; treatment {rc_t.get('why')}")
    tok_c = {r.get("tokens_digest") for r in ctl}
    tok_t = {r.get("tokens_digest") for r in trt}
    tokens_equal = bool(tok_c) and tok_c == tok_t and len(tok_c) == 1
    if not tokens_equal:
        problems.append(f"generated tokens differ across arms/runs: control {sorted(x[:12] for x in tok_c if x)} treatment {sorted(x[:12] for x in tok_t if x)}")
    n = min(len(ctl), len(trt))
    pairs = []
    for i in range(n):
        c, t = ctl[i], trt[i]
        pairs.append({"pair": i, "control_wall_s": c["wall_s"], "treatment_wall_s": t["wall_s"], "delta_wall_s": t["wall_s"] - c["wall_s"],
                      "control_decode_mean_s": c.get("step_wall_decode_mean_s"), "treatment_decode_mean_s": t.get("step_wall_decode_mean_s"),
                      "control_prefill_s": c.get("step_wall_prefill_s"), "treatment_prefill_s": t.get("step_wall_prefill_s"),
                      "commit_pass": (t.get("commit") or {}).get("commit_pass"), "run_root": (t.get("commit") or {}).get("run_root")})
    deltas = [p["delta_wall_s"] for p in pairs]
    ratio = [p["treatment_wall_s"] / p["control_wall_s"] for p in pairs if p["control_wall_s"]]
    lo, hi = _bootstrap_ci(deltas, seed=0)
    commit_pass = bool(trt) and all(bool((r.get("commit") or {}).get("commit_pass")) for r in trt) and not problems
    fail_reasons = [(r.get("commit") or {}).get("commit_fail_reason") for r in trt if not (r.get("commit") or {}).get("commit_pass")]
    tv = _json(T, "verdict.json")
    for X in Ts[1:]:
        v2 = _json(X, "verdict.json")
        tv["run_roots"] = list(tv.get("run_roots") or []) + list(v2.get("run_roots") or [])
        tv["runs_instrumented"] = int(tv.get("runs_instrumented") or 0) + int(v2.get("runs_instrumented") or 0)
        tv["runs_failed"] = int(tv.get("runs_failed") or 0) + int(v2.get("runs_failed") or 0)
    summary = {
        "schema": "verity-commit-delta/v1/compiled-paired", "label": (trt[0].get("label") if trt else None), "pairs": n, "rounds": rounds,
        "pairing": f"separate processes (control: no record ops; treatment: record ops compiled into the graphs), {rounds} alternating round(s) "
                   f"(c,t / t,c), {len(ctl)} control + {len(trt)} treatment runs paired by global run index",
        "compilation": ct, "compilation_control": cc, "compilation_settings_equal": all(cc.get(k) == ct.get(k) for k in COMPILE_KEYS),
        "tokens_equal": tokens_equal,
        "generated_kernels": {"identified": kernels_identified, "kernel_set_equal": kernel_set_equal,
                              "control": {k: gkc.get(k) for k in ("wrappers", "distinct_kernel_sources", "kernel_set_sha256", "call_set_sha256", "wrapper_norm_set_sha256", "per_process_cache_dir")},
                              "treatment": {k: gkt.get(k) for k in ("wrappers", "distinct_kernel_sources", "kernel_set_sha256", "call_set_sha256", "wrapper_norm_set_sha256", "per_process_cache_dir")},
                              "kernel_diff": kernel_diff},
        "program_digests": program_digests,
        "loaded_identity": loaded_identity,
        "control_wall_s": [r["wall_s"] for r in ctl], "treatment_wall_s": [r["wall_s"] for r in trt],
        "delta_wall_mean": statistics.fmean(deltas) if deltas else None, "delta_wall_ci95": [lo, hi], "delta_wall": deltas,
        "wall_ratio_mean": statistics.fmean(ratio) if ratio else None,
        "decode_mean_control_s": statistics.fmean([p["control_decode_mean_s"] for p in pairs if p["control_decode_mean_s"]]) if pairs else None,
        "decode_mean_treatment_s": statistics.fmean([p["treatment_decode_mean_s"] for p in pairs if p["treatment_decode_mean_s"]]) if pairs else None,
        "prefill_control_s": statistics.fmean([p["control_prefill_s"] for p in pairs if p["control_prefill_s"]]) if pairs else None,
        "prefill_treatment_s": statistics.fmean([p["treatment_prefill_s"] for p in pairs if p["treatment_prefill_s"]]) if pairs else None,
        "per_pair": pairs, "commit_pass": commit_pass, "commit_fail_reasons": fail_reasons, "problems": problems,
        "openings_verified": [((r.get("commit") or {}).get("openings") or {}).get("verified") for r in trt],
        "coverage": [((r.get("commit") or {}).get("coverage") or {}).get("ok") for r in trt],
        "value_correspondence": [((r.get("commit") or {}).get("value_correspondence") or {}).get("result") for r in trt],
        "compiled_source": [((r.get("commit") or {}).get("compiled_source") or {}).get("stats") for r in trt],
        "run_roots": [(r.get("commit") or {}).get("run_root") for r in trt],
        "treatment_verdict": tv, "versions": vt, "versions_control": vc,
    }
    md = [f"### Execution=compiled paired Commit ({n} pairs over {rounds} alternating round(s); control/treatment separate processes, same compilation settings: {summary['compilation_settings_equal']})",
          f"- compiled: mode {ct.get('compilation_mode')} cudagraph {ct.get('cudagraph_mode')} sizes {ct.get('cudagraph_capture_sizes')} enforce_eager {ct.get('enforce_eager')} inductor-partition {ct.get('use_inductor_graph_partition')}",
          f"- wall control {summary['control_wall_s']} s; treatment {summary['treatment_wall_s']} s; delta mean {summary['delta_wall_mean']} s (95% CI {lo:.4f}..{hi:.4f}); ratio {summary['wall_ratio_mean']}",
          f"- decode step mean control {summary['decode_mean_control_s']} s vs treatment {summary['decode_mean_treatment_s']} s; prefill {summary['prefill_control_s']} vs {summary['prefill_treatment_s']} s",
          f"- tokens equal across arms: {tokens_equal}; openings verified {summary['openings_verified']}; manifest coverage {summary['coverage']}; value correspondence {summary['value_correspondence']}",
          f"- generated kernels (Inductor output code): identified {kernels_identified}; control {gkc.get('wrappers')} pieces / {gkc.get('distinct_kernel_sources')} kernel sources "
          f"kernel_set {str(gkc.get('kernel_set_sha256'))[:16]}; treatment {gkt.get('wrappers')} pieces / {gkt.get('distinct_kernel_sources')} sources kernel_set "
          f"{str(gkt.get('kernel_set_sha256'))[:16]}; kernel set equal across arms: {kernel_set_equal}"
          + (f"; diff {kernel_diff}" if kernels_identified and not kernel_set_equal else ""),
          *( [f"- runtime<->artifact identity: LOADED code objects read from each running process (PyCodeCache registry): control "
              f"{loaded_identity['control']['modules']} modules / {loaded_identity['control']['kernels']} kernels ({loaded_identity['control']['kernels_with_binaries']} with cubins) "
              f"kernel_set {str(loaded_identity['control']['kernel_set_sha256'])[:16]} cubin_set {str(loaded_identity['control']['cubin_set_sha256'])[:16]}; treatment "
              f"{loaded_identity['treatment']['modules']} / {loaded_identity['treatment']['kernels']} ({loaded_identity['treatment']['kernels_with_binaries']}) kernel_set "
              f"{str(loaded_identity['treatment']['kernel_set_sha256'])[:16]} cubin_set {str(loaded_identity['treatment']['cubin_set_sha256'])[:16]}; equal across arms "
              f"{loaded_identity['equal_across_arms']} (kernel SOURCES of the loaded code objects; refusal if not); binaries equal across arms "
              f"{loaded_identity['binaries_equal_across_arms']} (labelled; differ: {loaded_identity['kernels_whose_binaries_differ'] or 'none'}); "
              f"P_compiled kernels {loaded_identity['manifest_kernels']}, not among the loaded objects {loaded_identity['manifest_kernels_not_loaded'] or 'none'}",
              f"- EXECUTED specialization (deny-list identity): executed_code_set (every ELF section of the launchers' cubins except "
              f"debug/line-info/PTX-text/symbol tables) control {str(loaded_identity['control'].get('executed_code_set_sha256'))[:16]} "
              f"treatment {str(loaded_identity['treatment'].get('executed_code_set_sha256'))[:16]} STRICT equal {loaded_identity['executed_code_equal_across_arms']}; "
              f"normalised (absolute-path strings in device data replaced) control {str(loaded_identity['control'].get('executed_code_set_sha256_normalized'))[:16]} "
              f"treatment {str(loaded_identity['treatment'].get('executed_code_set_sha256_normalized'))[:16]} equal {loaded_identity['executed_code_equal_modulo_device_path_strings']}; "
              f"VERDICT: {loaded_identity['executed_identity_verdict']} "
              f"({loaded_identity['control'].get('kernels_with_executed_code')} / {loaded_identity['treatment'].get('kernels_with_executed_code')} kernels with readable cubins); "
              f"code differs across arms: {loaded_identity['executed_code_differs'] or 'none'} (in P_compiled: {loaded_identity['executed_code_differs_in_program'] or 'none'}; "
              f"of these UNEXPLAINED -> refusal: {loaded_identity['executed_code_differs_unexplained_in_program'] or 'none'}; explained as per-process path strings, decoded on record: "
              f"{loaded_identity['executed_code_equal_modulo_path_strings'] or 'none'}); "
              f"executed autotune config differs: {loaded_identity['executed_config_differs'] or 'none'} (autotuned kernels: {loaded_identity['control'].get('kernels_autotuned')}; "
              f"shown output-equivalent under every candidate config on committed inputs, recorded: {loaded_identity['executed_code_equal_modulo_autotune_kernels'] or 'none'}); "
              f"code equal, only deny-listed sections differ: {loaded_identity['code_equal_denied_sections_differ'] or 'none'}; "
              f"repeated graph replay covered (build-time == end-of-run loaded objects, both arms): {loaded_identity['replay_covered']['ok']}"] if loaded_identity else [] ),
          *( [f"- P_compiled derived from each arm's own output code: program_digest control {str(program_digests['control'])[:16]} treatment "
              f"{str(program_digests['treatment'])[:16]} equal {program_digests['equal']} (wrapper text differs by the debug-printer statements the "
              f"hooks ride: call_set control {str(gkc.get('call_set_sha256'))[:12]} treatment {str(gkt.get('call_set_sha256'))[:12]})"] if program_digests else [] ),
          f"- Commit {'PASS' if commit_pass else 'FAIL'} {fail_reasons if fail_reasons else ''} {problems if problems else ''}"]
    summary["table_md"] = "\n".join(md)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    (out / "summary.md").write_text(summary["table_md"] + "\n")
    with (out / "runs.jsonl").open("w") as f:
        for r in ctl + trt:
            f.write(json.dumps(r, default=str) + "\n")
    verdict = dict(tv)
    verdict.update({"commit_pass": commit_pass, "paired": True, "problems": problems, "compilation": ct, "compilation_control": cc, "tokens_equal_across_arms": tokens_equal,
                    "first_fail_reason": (fail_reasons[0] if fail_reasons else (problems[0] if problems else None))})
    (out / "verdict.json").write_text(json.dumps(verdict, indent=1, default=str) + "\n")
    print(summary["table_md"])
    sys.exit(0 if commit_pass else 3)
