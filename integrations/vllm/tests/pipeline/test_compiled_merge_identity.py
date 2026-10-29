"""compiled_merge's executed-code identity is a DENY-list over the ELF sections of the executed cubins (debug /
line info / PTX text / symbol tables excluded; SASS, constant banks, initialised globals, rodata all count), and a difference in a
non-deny-listed section is accepted ONLY when decoded on record as per-process absolute-path strings in device data (bytes equal once
normalised).  Fixture: the two round-1 arms of the B1 record r16_0434_b1_unitB (written under the earlier .text-only rule: section hashes,
no section_detail).  Torch-free: drives compiled_merge.main() on copies of the arms."""
from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REC = ROOT / "out/gen/sweep/evidence/compiled/llama32-1b__bf16__l40s__tp1__b1__i256__o32__mixed__greedy__bi-compiled/r16_0434_b1_unitB"
PATH_KERNELS = ("triton_", "triton_poi_fused_1", "triton_poi_fused_3", "triton_red_fused__to_copy_embedding_rms_norm_t_0")
RULE = "sha256 over every ELF section except the deny-list (test)"

pytestmark = pytest.mark.skipif(not (REC / "commit_control_r1" / "versions.json").exists(), reason="fixture record r16_0434_b1_unitB not checked out")


def _arms(tmp: Path) -> tuple[Path, Path, Path, Path]:
    c, t = tmp / "control", tmp / "treatment"
    for src, dst in ((REC / "commit_control_r1", c), (REC / "commit_treatment_r1", t)):
        dst.mkdir()
        for n in ("versions.json", "runs.jsonl", "verdict.json", "summary.json"):
            if (src / n).exists():
                shutil.copy(src / n, dst / n)
    mc, mt = tmp / "manifest_control.json", tmp / "manifest_treatment.json"
    mc.write_bytes(gzip.open(REC / "compiled_manifest.json.gz").read())
    mt.write_bytes(gzip.open(REC / "compiled_manifest_treatment.json.gz").read())
    return c, t, mc, mt


def _merge(tmp: Path, c: Path, t: Path, mc: Path, mt: Path) -> dict:
    out = tmp / "merged"
    r = subprocess.run([sys.executable, str(ROOT / "verity_vllm" / "pipeline" / "compiled_merge.py"), "--control", str(c), "--treatment", str(t), "--out", str(out),
                        "--manifest-control", str(mc), "--manifest-treatment", str(mt)], capture_output=True, text=True, cwd=str(ROOT))
    assert (out / "summary.json").exists(), r.stderr[-2000:]
    return json.loads((out / "summary.json").read_text())


def _loaded(arm: Path) -> tuple[dict, dict]:
    V = json.loads((arm / "versions.json").read_text())
    return V, V["compilation"]["generated_kernels"]["loaded"]


def _write(arm: Path, V: dict) -> None:
    (arm / "versions.json").write_text(json.dumps(V))


def _h(x: str) -> str:
    return hashlib.sha256(x.encode()).hexdigest()


def _decorate(arm: Path, tag: str, *, constant_residue: dict[str, str] | None = None) -> None:
    """Give every executed binary the deny-list identity fields as loaded_code_objects() writes them: code_sha256 over the non-deny-listed
    sections (differs across arms for PATH_KERNELS because .nv.global.init embeds the arm's module path), code_sha256_normalized (equal),
    section_detail with residue_sha256 + path_strings.  `constant_residue` = {kernel: residue} to make a .nv.constant2 section differ for real."""
    V, L = _loaded(arm)
    code_set, norm_set = [], []
    for k, pk in L["per_kernel"].items():
        for e in pk.get("executed") or []:
            secs = e.get("sections") or {}
            det = {}
            for s, h in secs.items():
                if s in (".symtab", ".strtab", ".shstrtab") or s.startswith((".debug", ".rel.debug", ".rela.debug", ".nv.debug", ".nv_debug", ".rel.nv_debug")):
                    continue
                d = {"sha256": h, "size": 100, "residue_sha256": h}
                if s == ".nv.global.init" and k in PATH_KERNELS:
                    d["residue_sha256"] = "residue-" + k                    # equal across arms: only the path string differs
                    d["path_strings"] = [f"/workspace/{tag}/inductor_cache/ab/c{k}.py"]
                    d["n_path_strings"] = 1
                if constant_residue and k in constant_residue and s.startswith(".nv.constant0"):
                    d["residue_sha256"] = constant_residue[k]
                    d["sha256"] = constant_residue[k]
                    secs[s] = constant_residue[k]                             # the per-section hash table sees the difference too
                det[s] = d
            e["section_detail"] = det
            e["code_identity_rule"] = RULE
            e["code_sha256"] = _h(json.dumps(sorted((s, d["sha256"]) for s, d in det.items())))
            e["code_sha256_normalized"] = _h(json.dumps(sorted((s, d["residue_sha256"]) for s, d in det.items())))
            code_set.append((k, e["code_sha256"]))
            norm_set.append((k, e["code_sha256_normalized"]))
    L["executed_code_set_sha256"] = _h(json.dumps(sorted(code_set)))
    L["executed_code_set_sha256_normalized"] = _h(json.dumps(sorted(norm_set)))
    L["code_identity_rule"] = RULE
    _write(arm, V)


def test_old_text_only_record_fails_closed(tmp_path: Path) -> None:
    """The r16_0434 arms carry section hashes but no section_detail: .nv.global.init (non-deny-listed) differs for 4 kernels and cannot be
    decoded -> the identity verdict is DIFFERS / fail closed and the merge records the refusal (no silent 'debug-only' label)."""
    c, t, mc, mt = _arms(tmp_path)
    S = _merge(tmp_path, c, t, mc, mt)
    L = S["loaded_identity"]
    assert L["executed_code_equal_across_arms"] is False
    assert L["executed_code_equal_modulo_device_path_strings"] is False
    assert L["executed_identity_verdict"].startswith("DIFFERS")
    assert set(L["executed_code_differs_unexplained"]) >= set(PATH_KERNELS)
    assert set(L["executed_code_differs_unexplained_in_program"]) == set(PATH_KERNELS) - {"triton_"}     # 'triton_' is loaded but not a P_compiled kernel
    ex = L["executed"]["triton_poi_fused_1"]
    assert ".nv.global.init" in ex["sections_differing_non_denied"] and ".symtab" in ex["sections_differing_denied"]
    assert ex["device_data_path_strings"][0]["kind"].startswith("section_detail not recorded")
    assert any("a NON-deny-listed ELF section" in p for p in S["problems"])
    assert S["commit_pass"] is False


def test_decoded_path_strings_are_one_specialization(tmp_path: Path) -> None:
    """Same arms with the deny-list identity fields: the strict digests differ (path strings in .nv.global.init), the normalised digests agree,
    every difference is decoded on record -> verdict 'equal modulo ... path strings', no problem, the kernels listed as explained."""
    c, t, mc, mt = _arms(tmp_path)
    _decorate(c, "control")
    _decorate(t, "treatment")
    S = _merge(tmp_path, c, t, mc, mt)
    L = S["loaded_identity"]
    assert L["executed_code_equal_across_arms"] is False                       # strict: the bytes DO differ
    assert L["executed_code_equal_modulo_device_path_strings"] is True
    assert L["executed_identity_verdict"].startswith("equal modulo per-process absolute-path strings")
    assert sorted(L["executed_code_equal_modulo_path_strings"]) == sorted(PATH_KERNELS)
    assert L["executed_code_differs_unexplained"] == []
    ex = L["executed"]["triton_poi_fused_1"]
    assert ex["identity_verdict"].startswith("equal modulo")
    dec = ex["device_data_path_strings"][0]
    assert dec["section"] == ".nv.global.init" and dec["residue_equal"] is True
    assert dec["path_strings_control"] != dec["path_strings_treatment"] and all(p.startswith("/workspace/") for p in dec["path_strings_control"] + dec["path_strings_treatment"])
    assert L["executed"]["matmul_kernel_persistent"]["identity_verdict"] == "equal"
    assert "matmul_kernel_persistent" in L["code_equal_denied_sections_differ"]
    assert not any("of a P_compiled kernel differs across arms" in p for p in S["problems"])
    assert S["commit_pass"] is True
    assert "VERDICT: equal modulo" in S["table_md"] and "STRICT equal False" in S["table_md"]


def test_constant_bank_difference_refuses(tmp_path: Path) -> None:
    """A .nv.constant0 (param bank / compiler constants) difference whose bytes differ after normalisation is NOT debug: refusal for the
    P_compiled kernel, named with the section."""
    c, t, mc, mt = _arms(tmp_path)
    _decorate(c, "control", constant_residue={"triton_red_fused_fused_add_rms_norm_2": "const-A"})
    _decorate(t, "treatment", constant_residue={"triton_red_fused_fused_add_rms_norm_2": "const-B"})
    S = _merge(tmp_path, c, t, mc, mt)
    L = S["loaded_identity"]
    assert "triton_red_fused_fused_add_rms_norm_2" in L["executed_code_differs_unexplained_in_program"]
    ex = L["executed"]["triton_red_fused_fused_add_rms_norm_2"]
    assert ex["identity_verdict"] == "DIFFERS"
    assert any(s.startswith(".nv.constant0") for s in ex["non_denied_unexplained"])
    assert any(d["kind"].startswith("NON-PATH DATA DIFFERS") for d in ex["device_data_path_strings"])
    assert L["executed_identity_verdict"].startswith("DIFFERS")
    assert any("of a P_compiled kernel differs across arms" in p and "triton_red_fused_fused_add_rms_norm_2" in p for p in S["problems"])
    assert S["commit_pass"] is False


def _decorate_text_path(arm: Path, tag: str, *, section_prefix: str = ".text", kernel: str = "triton_poi_fused_1", strings: list[str] | None = None) -> None:
    """The honest decoration PLUS one kernel whose `section_prefix` section differs across arms
    with bytes 'equal once normalised' and a slash-led printable run as the only difference (SASS bytes can look like that)."""
    V, L = _loaded(arm)
    code_set, norm_set = [], []
    for k, pk in L["per_kernel"].items():
        for e in pk.get("executed") or []:
            secs = e.get("sections") or {}
            det = {}
            for s, h in secs.items():
                if s in (".symtab", ".strtab", ".shstrtab") or s.startswith((".debug", ".rel.debug", ".rela.debug", ".nv.debug", ".nv_debug", ".rel.nv_debug")):
                    continue
                d = {"sha256": h, "size": 100, "residue_sha256": h}
                if s == ".nv.global.init" and k in PATH_KERNELS:
                    d["residue_sha256"] = "residue-" + k
                    d["path_strings"] = [f"/workspace/{tag}/inductor_cache/ab/c{k}.py"]
                    d["n_path_strings"] = 1
                if k == kernel and s.startswith(section_prefix):
                    d["sha256"] = f"{section_prefix}-{tag}"
                    secs[s] = d["sha256"]
                    d["residue_sha256"] = f"residue-{section_prefix}-{k}"
                    d["path_strings"] = list(strings) if strings is not None else [f"/opc/{tag}/x"]
                    d["n_path_strings"] = len(d["path_strings"])
                det[s] = d
            e["section_detail"] = det
            e["code_identity_rule"] = RULE
            e["code_sha256"] = _h(json.dumps(sorted((s, d["sha256"]) for s, d in det.items())))
            e["code_sha256_normalized"] = _h(json.dumps(sorted((s, d["residue_sha256"]) for s, d in det.items())))
            code_set.append((k, e["code_sha256"]))
            norm_set.append((k, e["code_sha256_normalized"]))
    L["executed_code_set_sha256"] = _h(json.dumps(sorted(code_set)))
    L["executed_code_set_sha256_normalized"] = _h(json.dumps(sorted(norm_set)))
    L["code_identity_rule"] = RULE
    _write(arm, V)


def test_f16_14_text_difference_dressed_as_path_string_is_refused(tmp_path: Path) -> None:
    """A .text (SASS) difference whose bytes are 'equal once a slash-led run is normalised' is NOT a per-process path -- SASS
    never embeds the module path; the section name decides. Refusal named with the section."""
    c, t, mc, mt = _arms(tmp_path)
    _decorate_text_path(c, "control")
    _decorate_text_path(t, "treatment")
    S = _merge(tmp_path, c, t, mc, mt)
    L = S["loaded_identity"]
    assert "triton_poi_fused_1" in L["executed_code_differs_unexplained"]
    ex = L["executed"]["triton_poi_fused_1"]
    assert ex["identity_verdict"] == "DIFFERS"
    assert any(s.startswith(".text") for s in ex["non_denied_unexplained"])
    assert any(d["section"].startswith(".text") and d["data_section"] is False and "not a data section" in d["kind"] for d in ex["device_data_path_strings"])
    assert L["executed_identity_verdict"].startswith("DIFFERS")
    assert S["commit_pass"] is False


def test_f16_14_data_section_with_non_module_strings_is_refused(tmp_path: Path) -> None:
    """A DATA-section difference whose differing strings are not the generated module's .py path is not decoded either."""
    c, t, mc, mt = _arms(tmp_path)
    _decorate_text_path(c, "control", section_prefix=".nv.constant0", strings=["/opc/control/x"])
    _decorate_text_path(t, "treatment", section_prefix=".nv.constant0", strings=["/opc/treatment/x"])
    S = _merge(tmp_path, c, t, mc, mt)
    L = S["loaded_identity"]
    assert "triton_poi_fused_1" in L["executed_code_differs_unexplained"]
    ex = L["executed"]["triton_poi_fused_1"]
    assert any(d["section"].startswith(".nv.constant0") and d["data_section"] is True and d["module_paths"] is False and d["kind"].startswith("strings are not the generated module's .py path")
               for d in ex["device_data_path_strings"])
    assert S["commit_pass"] is False


def test_f16_14_module_path_with_line_number_in_data_section_is_decoded(tmp_path: Path) -> None:
    """The honest case still decodes: a .nv.global.init difference whose strings are the arm's generated module path (with a :<line> suffix)."""
    c, t, mc, mt = _arms(tmp_path)
    _decorate_text_path(c, "control", section_prefix=".nv.global.init", kernel="triton_red_fused_fused_add_rms_norm_2", strings=["/workspace/control/inductor_cache/zz/czz.py:17"])
    _decorate_text_path(t, "treatment", section_prefix=".nv.global.init", kernel="triton_red_fused_fused_add_rms_norm_2", strings=["/workspace/treatment/inductor_cache/zz/czz.py:17"])
    S = _merge(tmp_path, c, t, mc, mt)
    L = S["loaded_identity"]
    assert L["executed_code_differs_unexplained"] == []
    assert L["executed_code_equal_modulo_device_path_strings"] is True
    assert S["commit_pass"] is True


# ---- [compiled: the autotune rule] ------------------------------------------------------------------------------------------------------
POI = "triton_poi_fused_1"                                       # pointwise (RoPE) -- config independence structural
RED = "triton_red_fused__to_copy_embedding_rms_norm_t_0"        # reduction (norm) -- the config sets the summation order
OTHER_CFG = {"kwargs": {"XBLOCK": 512, "R0_BLOCK": 1024}, "num_warps": 8, "num_stages": 1}


def _retune(arm: Path, kernel: str, cfg: dict) -> dict:
    """Make the arm's executed binary of `kernel` a DIFFERENT autotune config of the same source (other SASS: code digests differ). Returns
    the config the arm executed before."""
    V, L = _loaded(arm)
    pk = L["per_kernel"][kernel]
    before = pk["executed"][0]["config"]
    for e in pk["executed"]:
        e["config"] = cfg
        e["code_sha256"] = _h("sass-of-" + kernel + json.dumps(cfg, sort_keys=True))
        e["code_sha256_normalized"] = e["code_sha256"]
        e["cubin_sha256"] = _h("cubin-" + e["code_sha256"])
        for s in list(e.get("sections") or {}):
            if s.startswith(".text"):
                e["sections"][s] = e["code_sha256"]
                if e.get("section_detail") and s in e["section_detail"]:
                    e["section_detail"][s].update({"sha256": e["code_sha256"], "residue_sha256": e["code_sha256"]})
    # the arm's executed-set digests follow (as loaded_code_objects() writes them at the end of the run): other SASS -> other set digests
    code_set = sorted((k, e["code_sha256"]) for k, p in L["per_kernel"].items() for e in p.get("executed") or [])
    norm_set = sorted((k, e["code_sha256_normalized"]) for k, p in L["per_kernel"].items() for e in p.get("executed") or [])
    L["executed_code_set_sha256"], L["executed_code_set_sha256_normalized"] = _h(json.dumps(code_set)), _h(json.dumps(norm_set))
    _write(arm, V)
    return before


def _autotune_record(arm: Path, kernel: str, heuristic: str, cands: list[dict], executed: dict, *, launches: int = 8, shown: int | None = None) -> None:
    """What CompiledKernelCheck._autotune_equivalence writes into every treatment run: the kernel's sampled launches replayed under every
    candidate config of its heuristic on committed inputs."""
    shown = launches if shown is None else shown
    lines = []
    for line in (arm / "runs.jsonl").read_text().splitlines():
        r = json.loads(line)
        cc = r["commit"]["value_correspondence"].setdefault("compiled_check", {})
        cc.setdefault("autotune_equivalence", {})[kernel] = {
            "heuristic": heuristic, "structural": heuristic.endswith("POINTWISE"), "candidates": cands, "n_candidates": len(cands),
            "executed": [executed], "executed_in_candidates": True, "launches": launches, "shown": shown, "not_shown": launches - shown,
            "why": None if shown == launches else "a candidate config does not reproduce the committed output",
            "exact_by_config": {json.dumps(c, sort_keys=True): shown for c in cands}, "all_shown": shown == launches,
            "lift_eligible": shown == launches and heuristic.endswith("POINTWISE")}
        lines.append(json.dumps(r))
    (arm / "runs.jsonl").write_text("\n".join(lines) + "\n")


def test_pointwise_kernel_shown_under_every_candidate_is_equal_modulo_autotune_config(tmp_path: Path) -> None:
    """The arms executed different autotune configs of the SAME pointwise source (other SASS); every treatment run showed the kernel's own source
    under both candidates output-equivalent on committed inputs -> 'equal modulo autotune config', no unexplained kernel, Commit stands."""
    c, t, mc, mt = _arms(tmp_path)
    _decorate(c, "control")
    _decorate(t, "treatment")
    ctrl_cfg = _retune(t, POI, OTHER_CFG)                                     # treatment executes the other candidate; control keeps ctrl_cfg
    _autotune_record(t, POI, "HeuristicType.POINTWISE", [ctrl_cfg, OTHER_CFG], OTHER_CFG)
    S = _merge(tmp_path, c, t, mc, mt)
    L = S["loaded_identity"]
    assert POI in L["executed_config_differs"]
    ex = L["executed"][POI]
    assert ex["code_equal"] is False and ex["config_equal"] is False
    assert ex["identity_verdict"].startswith("equal modulo autotune config")
    assert ex["autotune_equivalence"]["structural"] is True and ex["autotune_equivalence"]["launches_shown"] == 24    # 8 launches x 3 runs
    assert L["executed_code_differs_unexplained"] == []
    assert L["executed_code_equal_modulo_autotune"] is True and L["executed_code_equal_modulo_autotune_kernels"] == [POI]
    assert L["executed_identity_verdict"].startswith("equal modulo autotune config")
    assert S["commit_pass"] is True


def test_f16_22_reduction_kernel_is_never_lifted_by_sampled_equality(tmp_path: Path) -> None:
    """For a REDUCTION the config sets the summation/fma order -- bit-equality on the sampled launches is a property of the data,
    not of the kernel; a different executed config is a different specialization: DIFFERS stands even though every sampled launch was shown."""
    c, t, mc, mt = _arms(tmp_path)
    _decorate(c, "control")
    _decorate(t, "treatment")
    ctrl_cfg = _retune(t, RED, OTHER_CFG)
    _autotune_record(t, RED, "HeuristicType.REDUCTION", [ctrl_cfg, OTHER_CFG], OTHER_CFG)
    S = _merge(tmp_path, c, t, mc, mt)
    L = S["loaded_identity"]
    ex = L["executed"][RED]
    assert ex["identity_verdict"] == "DIFFERS"
    assert ex["autotune_equivalence"]["shown"] is False and ex["autotune_equivalence"]["structural"] is False
    assert ex["autotune_equivalence"]["why"].startswith("non-pointwise heuristic")
    assert RED in L["executed_code_differs_unexplained"] and L["executed_code_equal_modulo_autotune"] is False
    assert L["executed_identity_verdict"].startswith("DIFFERS")
    assert S["commit_pass"] is False


def test_pointwise_kernel_with_one_launch_not_shown_keeps_differs(tmp_path: Path) -> None:
    """Fail closed: one sampled launch not reproduced under a candidate (or not derivable) -> no lift, even for a pointwise kernel."""
    c, t, mc, mt = _arms(tmp_path)
    _decorate(c, "control")
    _decorate(t, "treatment")
    ctrl_cfg = _retune(t, POI, OTHER_CFG)
    _autotune_record(t, POI, "HeuristicType.POINTWISE", [ctrl_cfg, OTHER_CFG], OTHER_CFG, launches=8, shown=7)
    S = _merge(tmp_path, c, t, mc, mt)
    L = S["loaded_identity"]
    assert L["executed"][POI]["identity_verdict"] == "DIFFERS"
    assert L["executed"][POI]["autotune_equivalence"]["shown"] is False
    assert POI in L["executed_code_differs_unexplained"]
    assert S["commit_pass"] is False
