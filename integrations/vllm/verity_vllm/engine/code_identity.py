"""Code identity of what this process executes: the ELF sections of a compiled cubin (identity = every section except a
deny-list), the Inductor code objects loaded in this process with their Triton binaries and executed autotune configs, and the
Inductor output code the process compiled.
"""

from __future__ import annotations

import json
import os
from typing import Any

from verity_vllm import settings
from verity_vllm.engine import env as engine_env


_VERITY_INDUCTOR_DIR_MARK = engine_env.INDUCTOR_DIR_MARK


# ELF sections of a cubin that the DEVICE DOES NOT READ: debug/line info, embedded PTX text, and the host-side
# symbol/string tables.  The code identity is every OTHER section (deny-list, not a .text allow-list): .text.* (SASS), .nv.constant0.* (param
# bank + compiler constants), .nv.constant2/3 (constant-memory data), .nv.global(.init) (initialised device globals), .rodata, .nv.info*, ...
CUBIN_DENY_LIST: tuple[str, ...] = (".debug*", ".rel.debug*", ".rela.debug*", ".nv.debug*", ".nv_debug*", ".rel.nv_debug*", ".rela.nv_debug*", ".rel.nv.debug*",
                                    ".rela.nv.debug*", ".symtab", ".strtab", ".shstrtab")
_CUBIN_DENY_PREFIX = (".debug", ".rel.debug", ".rela.debug", ".nv.debug", ".nv_debug", ".rel.nv_debug", ".rela.nv_debug", ".rel.nv.debug", ".rela.nv.debug")
CUBIN_CODE_IDENTITY_RULE = ("sha256 over every ELF section of the executed cubin EXCEPT the named deny-list " + repr(CUBIN_DENY_LIST)
                            + " (debug/line info, embedded PTX text, symbol/string tables); NOT a .text allow-list")
_PATH_STRING_RE = None


def cubin_section_denied(name: str) -> bool:
    return name in (".symtab", ".strtab", ".shstrtab") or name.startswith(_CUBIN_DENY_PREFIX)


def _elf_section_bytes(b: bytes) -> dict[str, bytes]:
    """Pure-Python ELF32/64 section walk -> {section name: bytes} ({} if not an ELF)."""
    import struct

    if len(b) < 64 or b[:4] != b"\x7fELF":
        return {}
    is64, le = b[4] == 2, b[5] == 1
    E = "<" if le else ">"
    if is64:
        (e_shoff,) = struct.unpack_from(E + "Q", b, 0x28)
        e_shentsize, e_shnum, e_shstrndx = struct.unpack_from(E + "HHH", b, 0x3A)
    else:
        (e_shoff,) = struct.unpack_from(E + "I", b, 0x20)
        e_shentsize, e_shnum, e_shstrndx = struct.unpack_from(E + "HHH", b, 0x2E)
    hdrs = []
    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        if is64:
            sh_name, sh_type, _fl, _ad, sh_offset, sh_size = struct.unpack_from(E + "IIQQQQ", b, off)
        else:
            sh_name, sh_type, _fl, _ad, sh_offset, sh_size = struct.unpack_from(E + "IIIIII", b, off)
        hdrs.append((sh_name, sh_type, sh_offset, sh_size))
    if e_shstrndx >= len(hdrs):
        return {}
    _, _, stroff, strsz = hdrs[e_shstrndx]
    strtab = b[stroff:stroff + strsz]
    out: dict[str, bytes] = {}
    for sh_name, sh_type, sh_offset, sh_size in hdrs:
        end = strtab.find(b"\0", sh_name)
        name = strtab[sh_name:end if end >= 0 else None].decode(errors="replace") or "<null>"
        out[name] = b"" if sh_type == 8 else b[sh_offset:sh_offset + sh_size]          # SHT_NOBITS
    return out


def cubin_sections(b: bytes) -> dict[str, str]:
    """Per-section sha256 of a cubin (ELF), every section (deny-listed ones included, so that WHICH sections differ between
    two cubins is on record).  {} if not an ELF."""
    import hashlib

    return {n: hashlib.sha256(d).hexdigest() for n, d in _elf_section_bytes(b).items()}


def _split_path_strings(data: bytes) -> tuple[bytes, list[str]]:
    """Replace every printable run (>= 4 bytes) that is an ABSOLUTE PATH (starts with '/', no whitespace) by b'<PATH>' and return the
    normalised bytes + the removed strings.  Device data of Inductor/Triton kernels embeds the generated module's absolute source path
    (tl.device_assert / assert message strings in .nv.global.init) -- a per-process string, not computation."""
    import re

    global _PATH_STRING_RE
    if _PATH_STRING_RE is None:
        _PATH_STRING_RE = re.compile(rb"/[\x21-\x7e]{3,}")
    found: list[str] = []

    def _sub(m: "re.Match[bytes]") -> bytes:
        found.append(m.group().decode("latin1"))
        return b"<PATH>"

    return _PATH_STRING_RE.sub(_sub, data), found


def cubin_section_detail(b: bytes) -> dict[str, dict[str, Any]]:
    """For every NON-deny-listed section: sha256, size, sha256 of the bytes with absolute-path strings normalised
    (`residue_sha256`), and the path strings themselves -- so a cross-arm difference in device data can be DECODED from the record:
    residue equal + only path strings differ = per-process source path embedded in device data; anything else = not one specialization."""
    import hashlib

    out: dict[str, dict[str, Any]] = {}
    for n, d in _elf_section_bytes(b).items():
        if cubin_section_denied(n):
            continue
        norm, paths = _split_path_strings(d)
        ent: dict[str, Any] = {"sha256": hashlib.sha256(d).hexdigest(), "size": len(d), "residue_sha256": hashlib.sha256(norm).hexdigest()}
        if paths:
            ent["path_strings"] = paths[:16]
            ent["n_path_strings"] = len(paths)
        out[n] = ent
    return out


def cubin_code_sha256(sections: dict[str, str]) -> str | None:
    """sha256 over the sorted {(section name, sha256)} of every section NOT on CUBIN_DENY_LIST = everything the device loads (SASS, constant
    banks, initialised globals, rodata, kernel info), without debug/line-info/symbol sections.  None if the cubin has no .text section."""
    import hashlib

    code = sorted((k, v) for k, v in sections.items() if not cubin_section_denied(k))
    return hashlib.sha256(json.dumps(code).encode()).hexdigest() if any(k.startswith(".text") for k, _ in code) else None


def cubin_code_sha256_normalized(detail: dict[str, dict[str, Any]]) -> str | None:
    """As cubin_code_sha256 but over each non-deny-listed section's residue_sha256 (absolute-path strings normalised): equal across two
    processes iff every difference is a path string.  The strict digest is the identity; this one names what the difference is."""
    import hashlib

    code = sorted((k, v.get("residue_sha256")) for k, v in detail.items())
    return hashlib.sha256(json.dumps(code).encode()).hexdigest() if any(k.startswith(".text") for k, _ in code) else None


_TRITON_CACHE_INDEX: dict[str, dict[str, str]] = {}


def _triton_cache_files(h: str | None) -> dict[str, Any]:
    """Triton's compiled-kernel cache entry for hash `h` (TRITON_CACHE_DIR/<hash>/): the cubin bytes (when Inductor no longer holds asm) and
    the compile metadata json (num_warps / num_stages / target / options) -- what the executed binary was compiled WITH."""
    if not h:
        return {}
    import base64
    import glob

    root = settings.TRITON_CACHE_DIR or os.path.join(os.path.expanduser("~"), ".triton", "cache")
    cands = [os.path.join(root, str(h))]                          # Triton <= 3.3: <hex hash>/
    try:                                                           # Triton >= 3.4: base32(sha256 digest)/ (52 chars, no padding)
        cands.append(os.path.join(root, base64.b32encode(bytes.fromhex(str(h))).decode().rstrip("=")))
    except ValueError:
        pass
    d = next((c for c in cands if os.path.isdir(c)), None)
    if d is None:
        # last resort: the metadata json of every entry carries "hash": <hex> (indexed once per root)
        idx = _TRITON_CACHE_INDEX.get(root)
        if idx is None:
            idx = {}
            for mj in glob.glob(os.path.join(root, "*", "*.json")):
                if os.path.basename(mj).startswith("__grp__"):
                    continue
                try:
                    md = json.load(open(mj))
                    if isinstance(md, dict) and md.get("hash"):
                        idx[str(md["hash"])] = os.path.dirname(mj)
                except (OSError, ValueError):
                    continue
            _TRITON_CACHE_INDEX[root] = idx
        d = idx.get(str(h))
    if d is None:
        return {}
    out: dict[str, Any] = {"dir": d}
    cub = sorted(glob.glob(os.path.join(d, "*.cubin")))
    if cub:
        try:
            out["cubin"] = open(cub[0], "rb").read()
        except OSError:
            pass
    for mj in sorted(glob.glob(os.path.join(d, "*.json"))):
        if os.path.basename(mj).startswith("__grp__"):
            continue
        try:
            md = json.load(open(mj))
            if isinstance(md, dict) and ("num_warps" in md or "target" in md):
                out["meta"] = {k: md.get(k) for k in ("name", "num_warps", "num_stages", "num_ctas", "cluster_dims", "enable_fp_fusion", "debug", "shared",
                                                      "ptx_version", "target", "maxnreg", "launch_cooperative_grid") if k in md}
                break
        except (OSError, ValueError):
            continue
    return out


def _binary_identity(k: Any, cfg: Any = None) -> dict[str, Any]:
    """One compiled Triton binary: triton hash (source x signature/specialization x options x env), cubin sha256 (bytes), per-section
    sha256 + code_sha256 (every section except CUBIN_DENY_LIST) + code_sha256_normalized / section_detail (path strings in
    device data decoded), the compile metadata, and the autotune config it was compiled for."""
    import hashlib

    ent: dict[str, Any] = {"triton_hash": str(getattr(k, "hash", None)) if getattr(k, "hash", None) else None}
    asm = getattr(k, "asm", None)
    cub = asm.get("cubin") if isinstance(asm, dict) else None
    src = "asm" if cub is not None else None
    tc = _triton_cache_files(ent["triton_hash"])
    if cub is None and tc.get("cubin") is not None:
        cub, src = tc["cubin"], "triton_cache_dir"
    if tc.get("meta"):
        ent["compile_meta"] = tc["meta"]
    if cub is not None:
        secs = cubin_sections(cub)
        det = cubin_section_detail(cub)
        ent.update({"cubin_sha256": hashlib.sha256(cub).hexdigest(), "cubin_bytes": len(cub), "cubin_from": src,
                    "sections": secs, "code_sha256": cubin_code_sha256(secs), "code_sha256_normalized": cubin_code_sha256_normalized(det),
                    "section_detail": det, "code_identity_rule": CUBIN_CODE_IDENTITY_RULE,
                    "sections_denied": sorted(k for k in secs if cubin_section_denied(k))})
    md = getattr(k, "metadata", None)
    if md is not None and "compile_meta" not in ent:
        ent["compile_meta"] = {a: getattr(md, a, None) for a in ("num_warps", "num_stages", "num_ctas", "shared", "enable_fp_fusion", "debug") if hasattr(md, a)}
    if cfg is not None:
        ent["config"] = {"kwargs": {str(a): (int(b) if isinstance(b, int) else str(b)) for a, b in (getattr(cfg, "kwargs", None) or {}).items()},
                         "num_warps": getattr(cfg, "num_warps", None), "num_stages": getattr(cfg, "num_stages", None)}
    ent["id"] = ("code:" + ent["code_sha256"]) if ent.get("code_sha256") else (("triton:" + ent["triton_hash"]) if ent.get("triton_hash") else None)
    return ent


def loaded_code_objects() -> dict[str, Any]:
    """Identity of the Inductor code objects LOADED IN THIS PROCESS -- read from the running interpreter,
    not from files: `torch._inductor.codecache.PyCodeCache` is the registry of every output-code module Inductor exec'd (torch >= 2.9:
    `modules` list + `modules_no_attr` by path; older: `cache` by content-hash key), and each `CachingAutotuner` bound in those modules is
    a compiled Triton kernel: its JIT source (`fn.src`), and the binaries the device runs (`compile_results[*].kernel` / `launchers[*].bin`:
    Triton's compiled-kernel `hash`, and the cubin bytes' sha256 while Inductor still holds them).  Digests: kernel_set_sha256 over the sorted
    {(kernel name, source sha256)}, binary_set_sha256 over the sorted binary identities (cubin sha256 or triton hash), module_keys_sha256.

    Per compiled binary: triton hash + cubin sha256 + per-ELF-section sha256 + code_sha256 (every section except CUBIN_DENY_LIST) + compile metadata +
    the autotune config it was compiled for (`compile_results[*]` = every candidate the process compiled; `launchers[*]` = the one(s) it
    EXECUTES after autotuning); `executed_set_sha256` over the sorted {(kernel, src sha, executed code id, executed config)} is the identity
    of the specialization the graphs replay.  Two processes running the same code objects differ in cubin BYTES when the debug/line-info
    sections carry the per-process module path (code_sha256 equal) or in EXECUTED config when autotuning picked another candidate."""
    import hashlib

    def _sha(b: Any) -> str:
        return hashlib.sha256(b if isinstance(b, (bytes, bytearray)) else str(b).encode()).hexdigest()

    try:
        from torch._inductor.codecache import PyCodeCache
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner
        reg: list[tuple[str, Any]] = []
        seen: set[int] = set()
        for src_name in ("modules", "modules_no_attr", "cache"):
            coll = getattr(PyCodeCache, src_name, None)
            items = list(coll.items()) if isinstance(coll, dict) else [(getattr(m, "key", None) or getattr(m, "__file__", None) or str(id(m)), m) for m in (coll or [])]
            for k, m in items:
                if id(m) in seen:
                    continue
                seen.add(id(m)); reg.append((str(k), m))
        mods, kernels, bins_all = [], {}, set()
        for key, mod in reg:
            ks = 0
            for name, obj in list(vars(mod).items()):
                if not isinstance(obj, CachingAutotuner):
                    continue
                fn = getattr(obj, "fn", None)
                kname = getattr(fn, "__name__", None) or name
                src = getattr(fn, "src", None)
                bins: list[str] = []
                compiled: list[dict[str, Any]] = []          # every candidate binary this process compiled (autotune candidates included)
                for cr in list(getattr(obj, "compile_results", None) or []):
                    k = getattr(cr, "kernel", None)
                    b_ = _binary_identity(k, getattr(cr, "config", None))
                    compiled.append(b_)
                    asm = getattr(k, "asm", None)
                    cub = asm.get("cubin") if isinstance(asm, dict) else None
                    if cub is not None:
                        bins.append("cubin:" + _sha(cub))
                    elif getattr(k, "hash", None):
                        bins.append("triton:" + str(k.hash))
                # the EXECUTED specialization: the launcher(s) left after autotuning (Inductor keeps ONE launcher = the benchmark winner; the
                # graphs captured / the eager steps launch that binary) -- autotuning picks it per process by benchmark, so two processes
                # running the same code objects can execute differently configured binaries
                executed: list[dict[str, Any]] = []
                cfgs = []
                for l in list(getattr(obj, "launchers", None) or []):
                    c = getattr(l, "config", None)
                    b = getattr(l, "bin", None)
                    e_ = _binary_identity(b, c) if b is not None else ({"config": None, "id": None})
                    if b is None and c is not None:
                        # no `bin` on the launcher (older Inductor): the executed binary is the compile result whose config equals the launcher's
                        for b_ in compiled:
                            if b_.get("config") and c is not None and b_["config"] == {"kwargs": {str(a): (int(v) if isinstance(v, int) else str(v)) for a, v in (getattr(c, "kwargs", None) or {}).items()},
                                                                                        "num_warps": getattr(c, "num_warps", None), "num_stages": getattr(c, "num_stages", None)}:
                                e_ = dict(b_)
                                break
                    if c is not None:
                        cfgs.append({"kwargs": {str(a): (int(b2) if isinstance(b2, int) else str(b2)) for a, b2 in (getattr(c, "kwargs", None) or {}).items()},
                                     "num_warps": getattr(c, "num_warps", None), "num_stages": getattr(c, "num_stages", None)})
                        e_.setdefault("config", cfgs[-1])
                    executed.append(e_)
                    if b is not None:
                        asm = getattr(b, "asm", None)
                        if isinstance(asm, dict) and asm.get("cubin") is not None:
                            bins.append("cubin:" + _sha(asm["cubin"]))
                        elif getattr(b, "hash", None):
                            bins.append("triton:" + str(b.hash))
                if not executed and len(compiled) == 1:
                    executed = [dict(compiled[0])]
                bins_all.update(bins)
                candidates = [{"kwargs": {str(a): (int(b2) if isinstance(b2, int) else str(b2)) for a, b2 in (getattr(c, "kwargs", None) or {}).items()},
                               "num_warps": getattr(c, "num_warps", None), "num_stages": getattr(c, "num_stages", None)} for c in list(getattr(obj, "configs", None) or [])]
                ent = {"src_sha256": _sha(src) if src else None, "binaries": sorted(set(bins)), "n_launchers": len(getattr(obj, "launchers", None) or []),
                       "n_compile_results": len(compiled), "executed_configs": cfgs, "n_configs_candidate": len(candidates), "candidate_configs": candidates,
                       "compiled": compiled, "executed": executed, "module_key": key,
                       "autotuned": len(candidates) > 1 or len(compiled) > 1}
                if kname in kernels and kernels[kname]["src_sha256"] == ent["src_sha256"]:   # the same kernel bound in kernel module + wrapper namespace
                    kernels[kname]["binaries"] = sorted(set(kernels[kname]["binaries"]) | set(bins))
                    have = {json.dumps(x.get("id")) + json.dumps(x.get("config"), sort_keys=True) for x in kernels[kname]["compiled"]}
                    kernels[kname]["compiled"] += [x for x in compiled if json.dumps(x.get("id")) + json.dumps(x.get("config"), sort_keys=True) not in have]
                    have_e = {json.dumps(x.get("id")) + json.dumps(x.get("config"), sort_keys=True) for x in kernels[kname]["executed"]}
                    kernels[kname]["executed"] += [x for x in executed if json.dumps(x.get("id")) + json.dumps(x.get("config"), sort_keys=True) not in have_e]
                    kernels[kname]["executed_configs"] = kernels[kname]["executed_configs"] + [c for c in cfgs if c not in kernels[kname]["executed_configs"]]
                    kernels[kname]["n_compile_results"] = len(kernels[kname]["compiled"])
                    kernels[kname]["n_launchers"] += ent["n_launchers"]
                    kernels[kname]["autotuned"] = kernels[kname]["autotuned"] or ent["autotuned"]
                    kernels[kname].setdefault("module_keys", [kernels[kname]["module_key"]]).append(key)
                else:
                    kernels[kname] = ent
                ks += 1
            mods.append({"key": key[-64:], "path": getattr(mod, "__file__", None), "kernels": ks})
        kset = sorted((k, v["src_sha256"]) for k, v in kernels.items())
        # the EXECUTED specialization set: per kernel the code id(s) the launchers run (code_sha256 of the non-deny-listed sections when the
        # cubin is readable, else the triton hash) x the executed autotune config -- the identity of what the CUDA graphs replay
        exec_set = sorted((k, v["src_sha256"], json.dumps([(e.get("id"), json.dumps(e.get("config"), sort_keys=True)) for e in v["executed"]], sort_keys=True))
                          for k, v in kernels.items())
        code_set = sorted((k, e.get("code_sha256")) for k, v in kernels.items() for e in v["executed"] if e.get("code_sha256"))
        norm_set = sorted((k, e.get("code_sha256_normalized")) for k, v in kernels.items() for e in v["executed"] if e.get("code_sha256_normalized"))
        return {"modules": len(mods), "kernels": len(kernels), "kernels_with_binaries": sum(1 for v in kernels.values() if v["binaries"]),
                "kernel_set_sha256": _sha(json.dumps(kset)) if kset else None, "binary_set_sha256": _sha(json.dumps(sorted(bins_all))) if bins_all else None,
                "cubin_set_sha256": _sha(json.dumps(sorted(bins_all))) if bins_all else None,     # alias (compiled_merge reads it)
                "executed_set_sha256": _sha(json.dumps(exec_set)) if exec_set else None,
                "executed_code_set_sha256": _sha(json.dumps(code_set)) if code_set else None,
                "executed_code_set_sha256_normalized": _sha(json.dumps(norm_set)) if norm_set else None,
                "code_identity_rule": CUBIN_CODE_IDENTITY_RULE,
                "kernels_with_executed_code": len({k for k, _ in code_set}),
                "kernels_autotuned": sorted(k for k, v in kernels.items() if v.get("autotuned")),
                "binary_kinds": sorted({b.split(":", 1)[0] for b in bins_all}),
                "module_keys_sha256": _sha(json.dumps(sorted(m["key"] for m in mods))) if mods else None,
                "kernel_names": sorted(kernels), "per_kernel": kernels, "modules_list": mods,
                "triton_cache_dir": settings.raw("TRITON_CACHE_DIR"),
                "read_from": "torch._inductor.codecache.PyCodeCache (loaded modules) x CachingAutotuner.fn.src / compile_results[*].kernel + launchers[*].bin "
                             "(triton hash, cubin sha256 from asm or TRITON_CACHE_DIR/<hash>/*.cubin, per-ELF-section sha256, code_sha256 = every section "
                             "except CUBIN_DENY_LIST, code_sha256_normalized + section_detail.path_strings (absolute-path strings in device data decoded), "
                             "compile metadata, autotune config) in this process"}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)[:200]}


def loaded_code_objects_delta(start: dict[str, Any], end: dict[str, Any]) -> dict[str, Any]:
    """Compare two loaded_code_objects() snapshots of ONE process (after engine build = compile + Inductor autotune + CUDA-graph capture,
    and after the last measured run), so that repeated graph replay is covered and not only compilation/warm-up: the graphs replay
    exactly the launchers that existed at capture, so an equal executed set + no new modules/kernels/binaries means the artifact identity
    recorded at build is the identity of every replayed step; anything new (a recompile, a re-tune, a new specialization) is named."""
    if not isinstance(start, dict) or not isinstance(end, dict) or start.get("error") or end.get("error"):
        return {"ok": None, "why": f"snapshot unavailable: start {str((start or {}).get('error'))[:80]} end {str((end or {}).get('error'))[:80]}"}
    ps, pe = start.get("per_kernel") or {}, end.get("per_kernel") or {}
    new_k = sorted(set(pe) - set(ps))
    gone_k = sorted(set(ps) - set(pe))
    changed_exec = sorted(k for k in set(ps) & set(pe)
                          if json.dumps([(e.get("id"), e.get("config")) for e in ps[k].get("executed") or []], sort_keys=True)
                          != json.dumps([(e.get("id"), e.get("config")) for e in pe[k].get("executed") or []], sort_keys=True))
    changed_bins = sorted(k for k in set(ps) & set(pe) if sorted(ps[k].get("binaries") or []) != sorted(pe[k].get("binaries") or []))
    changed_src = sorted(k for k in set(ps) & set(pe) if ps[k].get("src_sha256") != pe[k].get("src_sha256"))
    out = {"modules_start": start.get("modules"), "modules_end": end.get("modules"), "new_modules": (end.get("modules") or 0) - (start.get("modules") or 0),
           "kernels_start": start.get("kernels"), "kernels_end": end.get("kernels"), "new_kernels": new_k[:40], "kernels_gone": gone_k[:40],
           "executed_changed": changed_exec[:40], "binaries_changed": changed_bins[:40], "sources_changed": changed_src[:40],
           "kernel_set_equal": start.get("kernel_set_sha256") == end.get("kernel_set_sha256"),
           "executed_set_equal": start.get("executed_set_sha256") == end.get("executed_set_sha256"),
           "executed_set_sha256": {"start": start.get("executed_set_sha256"), "end": end.get("executed_set_sha256")}}
    out["ok"] = bool(out["kernel_set_equal"] and out["executed_set_equal"] and not new_k and not changed_exec and not changed_bins and not changed_src
                     and out["new_modules"] == 0)
    out["why"] = None if out["ok"] else ("the code objects / executed specialization changed between build and end of run: " +
                                          ", ".join(f"{k}={v}" for k, v in out.items() if k in ("new_modules", "new_kernels", "executed_changed", "binaries_changed", "sources_changed") and v))
    return out


def generated_kernels_of_this_process(inductor_dir: str | None = None) -> dict[str, Any]:
    """Digest of the Inductor output code this process compiled (W1c's kernel identity: kernel_set_sha256 = sha256 of the sorted
    {(kernel name, source sha256)}; call_set_sha256 over the pieces' launch programs), plus where it was read from.  Shared by the record
    path (RunHeader.compilation.generated_kernels) and commit_delta's versions.json so the two pins are comparable byte for byte."""
    try:
        from verity_vllm.engine import kernel_identity as KI

        ind = inductor_dir or settings.TORCHINDUCTOR_CACHE_DIR or KI.inductor_cache_dir()
        doc = KI.collect(ind, KI.triton_cache_dir())
        gk = dict(doc["summary"])
        gk["inductor_cache_dir"] = ind
        gk["per_process_cache_dir"] = settings.raw(_VERITY_INDUCTOR_DIR_MARK) == "1"      # set by prepare_compiled_process_cache / compiled_commit.sh
        gk["pieces"] = [{"path": w["path"], "norm_sha256": w["norm_sha256"], "call_sha256": w["call_sha256"], "kernels": len(w["kernels"]),
                         "launches": w["launches"], "extern_calls": w["extern_calls"]} for w in doc["wrappers"]]
        gk["kernels"] = doc["kernels"]
        gk["loaded"] = loaded_code_objects()                                       # the code objects this process actually loaded
        if gk["loaded"].get("kernel_names") is not None:
            gk["loaded"]["files_vs_loaded"] = {"in_files_not_loaded": sorted(set(doc["kernels"]) - set(gk["loaded"]["kernel_names"]))[:40],
                                               "loaded_not_in_files": sorted(set(gk["loaded"]["kernel_names"]) - set(doc["kernels"]))[:40]}
        if not gk.get("wrappers"):
            # kernel_identity digests the EMPTY set to a hex string; a pin over nothing is not evidence -> null it and say why
            gk["error"] = f"no Inductor output code under {ind!r} (compile cache hit from another process, or the dir is not this process's)"
        elif not gk["per_process_cache_dir"]:
            # torch's default /tmp/torchinductor_<user> holds every process's output code on the pod: a digest over it is not THIS engine's
            gk["error"] = f"Inductor output code read from a shared dir {ind!r} ({gk['wrappers']} pieces): not this process's identity"
        if gk.get("error"):
            for k in ("kernel_set_sha256", "call_set_sha256", "wrapper_set_sha256", "wrapper_norm_set_sha256"):
                gk[k] = None
        return gk
    except Exception as e:  # noqa: BLE001
        return {"kernel_set_sha256": None, "call_set_sha256": None, "wrappers": 0, "distinct_kernel_sources": 0, "error": repr(e)[:200]}
