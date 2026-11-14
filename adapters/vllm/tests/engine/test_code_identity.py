"""`verity_vllm.engine.code_identity` on a CPU: the ELF walk and the digests of a cubin on hand-written ELF files, Triton's
cache entry of a kernel on a hand-written cache, and the loaded code objects on stand-ins for Inductor's registry
(`torch._inductor.codecache.PyCodeCache`) and its compiled kernels (`CachingAutotuner`).  No torch is imported.
"""
from __future__ import annotations

import base64
import hashlib
import json
import struct
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from verity_vllm.engine import code_identity as CI


def sha(data: bytes | str) -> str:
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


def elf(sections: dict[str, bytes], *, bits: int = 64, little: bool = True, nobits: tuple[str, ...] = ()) -> bytes:
    """An ELF file of a null section, `sections` in order and the section-name table, with nothing but what the walk reads."""
    order = "<" if little else ">"
    names = [*sections, ".shstrtab"]
    table, at = b"\0", {}
    for name in names:
        at[name] = len(table)
        table += name.encode() + b"\0"
    bodies = {**sections, ".shstrtab": table}
    head, entry, fields = (64, 64, "IIQQQQ") if bits == 64 else (52, 40, "IIIIII")
    blob, where = b"", {}
    for name in names:
        where[name] = head + len(blob)
        blob += bodies[name]
    headers = struct.pack(order + fields, 0, 0, 0, 0, 0, 0).ljust(entry, b"\0")
    for name in names:
        kind = 8 if name in nobits else 1
        headers += struct.pack(order + fields, at[name], kind, 0, 0, where[name], len(bodies[name])).ljust(entry, b"\0")
    header = bytearray(head)
    header[:6] = b"\x7fELF" + bytes([2 if bits == 64 else 1, 1 if little else 2])
    if bits == 64:
        struct.pack_into(order + "Q", header, 0x28, head + len(blob))
        struct.pack_into(order + "HHH", header, 0x3A, entry, len(names) + 1, len(names))
    else:
        struct.pack_into(order + "I", header, 0x20, head + len(blob))
        struct.pack_into(order + "HHH", header, 0x2E, entry, len(names) + 1, len(names))
    return bytes(header) + blob + headers


def kernel_sections(text: bytes, path: bytes = b"/tmp/inductor_one/ab/k.py", line: bytes = b"line-info") -> dict[str, bytes]:
    return {".text.k": text, ".nv.info": b"info", ".nv.global.init": b"\x00\x01assert at " + path + b" line 7\x00",
            ".bss": b"never-read", ".debug_line": line}


def cubin(text: bytes, **how: Any) -> bytes:
    return elf(kernel_sections(text, **how), nobits=(".bss",))


NAMES = ["<null>", ".text.k", ".nv.info", ".nv.global.init", ".bss", ".debug_line", ".shstrtab"]
NAME_TABLE = b"\0.text.k\0.nv.info\0.nv.global.init\0.bss\0.debug_line\0.shstrtab\0"


@pytest.mark.parametrize("bits", [64, 32])
@pytest.mark.parametrize("little", [True, False])
def test_the_walk_reads_every_section_of_an_elf_of_either_width_and_byte_order(bits: int, little: bool) -> None:
    sections = kernel_sections(b"\x01\x02SASS")
    got = CI._elf_section_bytes(elf(sections, bits=bits, little=little, nobits=(".bss",)))
    assert list(got) == NAMES
    assert got == {**sections, "<null>": b"", ".bss": b"", ".shstrtab": NAME_TABLE}


def test_the_walk_gives_nothing_for_what_is_not_an_elf() -> None:
    whole = cubin(b"SASS")
    assert CI._elf_section_bytes(b"") == {} and CI._elf_section_bytes(whole[:63]) == {}
    assert CI._elf_section_bytes(b"\x7fELG" + whole[4:]) == {}
    no_names = bytearray(whole)
    struct.pack_into("<H", no_names, 0x3E, 99)                 # the name table's index points past the last section
    assert CI._elf_section_bytes(bytes(no_names)) == {}
    assert CI.cubin_sections(b"not an ELF, but longer than an ELF header: " + bytes(64)) == {}


def test_a_cubins_code_identity_is_every_section_but_the_denied_ones() -> None:
    whole = cubin(b"SASS")
    sections = CI.cubin_sections(whole)
    assert sections == {name: sha(data) for name, data in CI._elf_section_bytes(whole).items()} and list(sections) == NAMES
    denied = [name for name in sections if CI.cubin_section_denied(name)]
    assert denied == [".debug_line", ".shstrtab"]
    assert all(CI.cubin_section_denied(name) for name in (".symtab", ".strtab", ".rela.debug_info", ".nv_debug_ptx_txt", ".rel.nv.debug"))
    assert not any(CI.cubin_section_denied(name) for name in (".text.k", ".nv.constant0.k", ".rodata", ".nv.info.k", ".symtab_shndx"))
    code = sorted([name, digest] for name, digest in sections.items() if name not in denied)
    assert CI.cubin_code_sha256(sections) == sha(json.dumps(code))
    assert CI.cubin_code_sha256(CI.cubin_sections(cubin(b"SASS", line=b"other line info"))) == CI.cubin_code_sha256(sections)
    assert CI.cubin_code_sha256(CI.cubin_sections(cubin(b"SASt"))) != CI.cubin_code_sha256(sections)
    assert CI.cubin_code_sha256({name: digest for name, digest in sections.items() if name != ".text.k"}) is None


def test_a_path_in_device_data_is_named_and_left_out_of_the_normalised_identity() -> None:
    one, two = cubin(b"SASS"), cubin(b"SASS", path=b"/tmp/inductor_two/cd/k.py")
    first, second = CI.cubin_section_detail(one), CI.cubin_section_detail(two)
    assert list(first) == [name for name in NAMES if name not in (".debug_line", ".shstrtab")]
    data = kernel_sections(b"SASS")[".nv.global.init"]
    assert first[".nv.global.init"] == {"sha256": sha(data), "size": len(data), "n_path_strings": 1, "path_strings": ["/tmp/inductor_one/ab/k.py"],
                                        "residue_sha256": sha(b"\x00\x01assert at <PATH> line 7\x00")}
    assert first[".text.k"] == {"sha256": sha(b"SASS"), "size": 4, "residue_sha256": sha(b"SASS")}
    assert second[".nv.global.init"]["path_strings"] == ["/tmp/inductor_two/cd/k.py"]
    assert second[".nv.global.init"]["residue_sha256"] == first[".nv.global.init"]["residue_sha256"]
    assert CI.cubin_code_sha256(CI.cubin_sections(one)) != CI.cubin_code_sha256(CI.cubin_sections(two))
    normalised = CI.cubin_code_sha256_normalized(first)
    assert normalised == CI.cubin_code_sha256_normalized(second)
    assert normalised == sha(json.dumps(sorted([name, entry["residue_sha256"]] for name, entry in first.items())))
    assert CI.cubin_code_sha256_normalized({name: entry for name, entry in first.items() if name != ".text.k"}) is None
    assert CI._split_path_strings(b"a/b /ab /abc\x00/x/y z") == (b"a/b /ab <PATH>\x00<PATH> z", ["/abc", "/x/y"])
    many = CI.cubin_section_detail(elf({".text.k": b" ".join(b"/p/%d" % i for i in range(20))}))[".text.k"]
    assert many["n_path_strings"] == 20 and many["path_strings"] == [f"/p/{i}" for i in range(16)]


# ---- Triton's cache entry of a kernel ---------------------------------------------------------------------------------

HEX = "aa11"
CACHED = cubin(b"SASS-of-the-cache")
META = {"hash": HEX, "name": "triton_red_fused_1", "num_warps": 8, "num_stages": 3, "debug": False, "not_recorded": 1}
RECORDED_META = {"name": "triton_red_fused_1", "num_warps": 8, "num_stages": 3, "debug": False}


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A Triton cache at the relative path `triton` of the working directory, so no recorded string holds a temporary path."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRITON_CACHE_DIR", "triton")
    monkeypatch.setattr(CI, "_TRITON_CACHE_INDEX", {})
    root = tmp_path / "triton"
    root.mkdir()
    return root


def entry_of(root: Path, name: str, meta: dict[str, Any] | None = None, binary: bytes | None = CACHED) -> Path:
    entry = root / name
    entry.mkdir()
    (entry / "__grp__k.json").write_text(json.dumps({"num_warps": "a group file, never read as the metadata"}))
    (entry / "a.json").write_text("{not json")
    (entry / "b.json").write_text(json.dumps({"hash": (meta or META)["hash"], "child_paths": {}}))
    (entry / "k.json").write_text(json.dumps(meta or META))
    if binary is not None:
        (entry / "k.cubin").write_bytes(binary)
    return entry


def test_a_kernels_cache_entry_is_found_under_each_name_triton_gives_it(cache: Path) -> None:
    newer = base64.b32encode(bytes.fromhex(HEX)).decode().rstrip("=")
    assert newer == "VIIQ"
    assert CI._triton_cache_files(None) == {} and CI._triton_cache_files("") == {} and CI._triton_cache_files(HEX) == {}
    entry_of(cache, newer)
    assert CI._triton_cache_files(HEX) == {"dir": f"triton/{newer}", "cubin": CACHED, "meta": RECORDED_META}
    entry_of(cache, HEX, binary=b"the older name is looked at first")
    assert CI._triton_cache_files(HEX)["cubin"] == b"the older name is looked at first"


def test_a_kernel_whose_hash_names_no_directory_is_found_by_its_metadata(cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    entry_of(cache, "any-name", {**META, "hash": "not hex"}, binary=None)
    (cache / "any-name" / "k.cubin").mkdir()                       # a binary that cannot be read is left out
    assert CI._triton_cache_files("not hex") == {"dir": "triton/any-name", "meta": RECORDED_META}
    assert CI._triton_cache_files("another") == {}
    entry_of(cache, "later", {**META, "hash": "another"})
    assert CI._triton_cache_files("another") == {}, "the index of a cache directory is made once in a process"
    monkeypatch.setattr(CI, "_TRITON_CACHE_INDEX", {})
    assert CI._triton_cache_files("another") == {"dir": "triton/later", "cubin": CACHED, "meta": RECORDED_META}


def test_an_entry_with_no_compile_metadata_gives_its_binary_alone(cache: Path) -> None:
    entry = cache / HEX
    entry.mkdir()
    (entry / "k.json").write_text(json.dumps({"hash": HEX}))
    (entry / "k.cubin").write_bytes(CACHED)
    assert CI._triton_cache_files(HEX) == {"dir": f"triton/{HEX}", "cubin": CACHED}
    (entry / "k.json").write_text(json.dumps({"target": "cuda:90"}))
    assert CI._triton_cache_files(HEX) == {"dir": f"triton/{HEX}", "cubin": CACHED, "meta": {"target": "cuda:90"}}


def test_the_cache_is_tritons_default_when_no_directory_is_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(CI, "_TRITON_CACHE_INDEX", {})
    root = tmp_path / ".triton" / "cache"
    root.mkdir(parents=True)
    entry_of(root, HEX)
    assert CI._triton_cache_files(HEX) == {"dir": str(root / HEX), "cubin": CACHED, "meta": RECORDED_META}


# ---- one compiled binary ----------------------------------------------------------------------------------------------

def config(block: int, warps: int = 4, stages: int = 2) -> types.SimpleNamespace:
    return types.SimpleNamespace(kwargs={"BLOCK": block, "mode": "x"}, num_warps=warps, num_stages=stages)


def recorded(block: int, warps: int = 4, stages: int = 2) -> dict[str, Any]:
    return {"kwargs": {"BLOCK": block, "mode": "x"}, "num_warps": warps, "num_stages": stages}


def binary(name: str | None = None, held: bytes | None = None, **more: Any) -> types.SimpleNamespace:
    return types.SimpleNamespace(hash=name, asm={"cubin": held} if held is not None else None, **more)


def identity_of(held: bytes, source: str) -> dict[str, Any]:
    sections, detail = CI.cubin_sections(held), CI.cubin_section_detail(held)
    code = CI.cubin_code_sha256(sections)
    return {"cubin_sha256": sha(held), "cubin_bytes": len(held), "cubin_from": source, "sections": sections, "code_sha256": code,
            "code_sha256_normalized": CI.cubin_code_sha256_normalized(detail), "section_detail": detail,
            "code_identity_rule": CI.CUBIN_CODE_IDENTITY_RULE, "sections_denied": [".debug_line", ".shstrtab"], "id": f"code:{code}"}


def test_a_binary_is_identified_by_the_code_it_holds_else_by_its_cache_entry_else_by_its_hash(cache: Path) -> None:
    held = cubin(b"SASS-A")
    metadata = types.SimpleNamespace(num_warps=4, num_stages=2, shared=0, target="not recorded from here")
    assert CI._binary_identity(binary("h0", held, metadata=metadata), config(8)) == {
        "triton_hash": "h0", **identity_of(held, "asm"), "compile_meta": {"num_warps": 4, "num_stages": 2, "shared": 0}, "config": recorded(8)}
    entry_of(cache, HEX)
    assert CI._binary_identity(binary(HEX, metadata=metadata)) == {
        "triton_hash": HEX, **identity_of(CACHED, "triton_cache_dir"), "compile_meta": RECORDED_META}
    assert CI._binary_identity(binary(HEX, held))["cubin_from"] == "asm", "the code the process holds comes before the cache's"
    assert CI._binary_identity(binary("beef")) == {"triton_hash": "beef", "id": "triton:beef"}
    assert CI._binary_identity(binary(), types.SimpleNamespace(kwargs=None)) == {
        "triton_hash": None, "config": {"kwargs": {}, "num_warps": None, "num_stages": None}, "id": None}
    assert CI._binary_identity(None) == {"triton_hash": None, "id": None}


# ---- the code objects a process has loaded ----------------------------------------------------------------------------

class CachingAutotuner:
    """Stands in for Inductor's: the attributes `loaded_code_objects` reads off a compiled kernel."""

    def __init__(self, fn: Any = None, **read: Any) -> None:
        self.fn = fn
        vars(self).update(read)


def source(name: str | None, text: str | None) -> types.SimpleNamespace:
    return types.SimpleNamespace(src=text) if name is None else types.SimpleNamespace(__name__=name, src=text)


def module(**names: Any) -> types.ModuleType:
    made = types.ModuleType("generated")
    vars(made).update(names)
    return made


A, B, A_ELSEWHERE = cubin(b"SASS-A"), cubin(b"SASS-B"), cubin(b"SASS-A", path=b"/tmp/inductor_two/cd/k.py", line=b"other")
KEY_A = "k" * 70
FILE_B, FILE_C = "/inductor/bb/generated_b.py", "/inductor/cc/generated_c.py"


@pytest.fixture
def registry(cache: Path, monkeypatch: pytest.MonkeyPatch) -> type:
    """Inductor's registry and its kernel class, as modules `loaded_code_objects` imports; the registry is empty."""
    class PyCodeCache:
        modules: list[Any] = []

    for name, holds in (("torch._inductor.codecache", {"PyCodeCache": PyCodeCache}),
                        ("torch._inductor.runtime.triton_heuristics", {"CachingAutotuner": CachingAutotuner})):
        stand_in = types.ModuleType(name)
        vars(stand_in).update(holds)
        monkeypatch.setitem(sys.modules, name, stand_in)
    return PyCodeCache


def load(registry: Any, cache: Path) -> None:
    """Three generated modules of four kernels, one of them bound in two modules, with every shape of binary and launcher."""
    entry_of(cache, base64.b32encode(bytes.fromhex(HEX)).decode().rstrip("="))
    tuned = CachingAutotuner(
        source("triton_poi_fused_0", "SRC0"), configs=[config(8), config(16)],
        compile_results=[types.SimpleNamespace(kernel=binary("h0a", A), config=config(8)),
                         types.SimpleNamespace(kernel=binary("h0b", B), config=config(16))],
        launchers=[types.SimpleNamespace(config=config(16), bin=binary("h0b", B))])
    cached = CachingAutotuner(                               # its binary is in Triton's cache; its launcher holds no `bin`
        source("triton_red_fused_1", "SRC1"), configs=[config(1)],
        compile_results=[types.SimpleNamespace(kernel=binary(HEX), config=config(1))],
        launchers=[types.SimpleNamespace(config=config(1))])
    again = CachingAutotuner(                                # the first kernel as a second module binds it
        source("triton_poi_fused_0", "SRC0"), configs=[config(8)],
        compile_results=[types.SimpleNamespace(kernel=binary("h0a", A), config=config(8))],
        launchers=[types.SimpleNamespace(config=config(8), bin=binary("h0a2", A_ELSEWHERE))])
    bare = CachingAutotuner(                                 # no source, no compile result; one launcher with a hash only, one with nothing
        None, launchers=[types.SimpleNamespace(bin=binary("beef")), types.SimpleNamespace()])
    lone = CachingAutotuner(                                 # one compile result and no launcher: that result is what runs
        source(None, "SRC3"), compile_results=[types.SimpleNamespace(kernel=binary())])
    first = module(key=KEY_A, __file__="/inductor/aa/generated_a.py", triton_poi_fused_0=tuned, triton_red_fused_1=cached, helper=3)
    second = module(__file__=FILE_B, triton_poi_fused_0=again, triton_per_fused_2=bare)
    third = module(__file__="/inductor/cc/another_name.py", triton_poi_fused_3=lone)
    registry.modules = [first, second]
    registry.modules_no_attr = {FILE_C: third, "/the first module again": first}


def test_no_loaded_code_gives_counts_of_zero_and_no_digest(registry: Any) -> None:
    doc = CI.loaded_code_objects()
    assert {name: doc[name] for name in ("modules", "kernels", "kernels_with_binaries", "kernels_with_executed_code")} == {
        "modules": 0, "kernels": 0, "kernels_with_binaries": 0, "kernels_with_executed_code": 0}
    assert [name for name, value in doc.items() if name.endswith(("_sha256", "_sha256_normalized")) and value is not None] == []
    assert doc["per_kernel"] == {} and doc["kernel_names"] == [] and doc["modules_list"] == [] and doc["triton_cache_dir"] == "triton"
    assert CI.loaded_code_objects_delta(doc, doc)["ok"] is True


def test_a_registry_that_cannot_be_read_is_an_error_document(registry: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    registry.modules = [object()]
    doc = CI.loaded_code_objects()
    assert list(doc) == ["error"] and doc["error"].startswith("TypeError(")
    monkeypatch.setitem(sys.modules, "torch._inductor.codecache", None)
    doc = CI.loaded_code_objects()
    assert list(doc) == ["error"] and doc["error"].startswith("ModuleNotFoundError(")


def test_the_loaded_code_objects_of_a_registry(registry: Any, cache: Path) -> None:
    load(registry, cache)
    doc = CI.loaded_code_objects()
    kernels = doc["per_kernel"]
    assert doc["kernel_names"] == sorted(kernels) == ["triton_per_fused_2", "triton_poi_fused_0", "triton_poi_fused_3", "triton_red_fused_1"]
    assert doc["modules_list"] == [{"key": "k" * 64, "path": "/inductor/aa/generated_a.py", "kernels": 2},
                                   {"key": FILE_B, "path": FILE_B, "kernels": 2},
                                   {"key": FILE_C, "path": "/inductor/cc/another_name.py", "kernels": 1}]
    assert (doc["modules"], doc["kernels"], doc["kernels_with_binaries"], doc["kernels_with_executed_code"]) == (3, 4, 3, 2)
    assert doc["kernels_autotuned"] == ["triton_poi_fused_0"] and doc["binary_kinds"] == ["cubin", "triton"]
    assert doc["code_identity_rule"] == CI.CUBIN_CODE_IDENTITY_RULE and doc["triton_cache_dir"] == "triton"

    tuned = kernels["triton_poi_fused_0"]
    of_a, of_b, elsewhere = identity_of(A, "asm"), identity_of(B, "asm"), identity_of(A_ELSEWHERE, "asm")
    assert tuned["compiled"] == [{"triton_hash": "h0a", **of_a, "config": recorded(8)}, {"triton_hash": "h0b", **of_b, "config": recorded(16)}]
    assert tuned["executed"] == [{"triton_hash": "h0b", **of_b, "config": recorded(16)}, {"triton_hash": "h0a2", **elsewhere, "config": recorded(8)}]
    assert tuned["binaries"] == sorted(f"cubin:{sha(held)}" for held in (A, B, A_ELSEWHERE))
    assert tuned["executed_configs"] == [recorded(16), recorded(8)] and tuned["candidate_configs"] == [recorded(8), recorded(16)]
    assert (tuned["n_launchers"], tuned["n_compile_results"], tuned["n_configs_candidate"], tuned["autotuned"]) == (2, 2, 2, True)
    assert tuned["src_sha256"] == sha("SRC0") and tuned["module_key"] == KEY_A and tuned["module_keys"] == [KEY_A, FILE_B]
    assert of_a["code_sha256"] != elsewhere["code_sha256"] and of_a["code_sha256_normalized"] == elsewhere["code_sha256_normalized"]

    cached = kernels["triton_red_fused_1"]
    in_cache = {"triton_hash": HEX, **identity_of(CACHED, "triton_cache_dir"), "compile_meta": RECORDED_META, "config": recorded(1)}
    assert cached["compiled"] == [in_cache] and cached["executed"] == [in_cache] and cached["binaries"] == [f"triton:{HEX}"]
    assert (cached["n_launchers"], cached["executed_configs"], cached["autotuned"]) == (1, [recorded(1)], False)

    bare = kernels["triton_per_fused_2"]
    assert bare["executed"] == [{"triton_hash": "beef", "id": "triton:beef"}, {"config": None, "id": None}]
    assert (bare["src_sha256"], bare["compiled"], bare["binaries"], bare["n_launchers"]) == (None, [], ["triton:beef"], 2)

    lone = kernels["triton_poi_fused_3"]
    assert lone["compiled"] == lone["executed"] == [{"triton_hash": None, "id": None}] and lone["binaries"] == []
    assert (lone["src_sha256"], lone["n_launchers"], lone["candidate_configs"], lone["module_key"]) == (sha("SRC3"), 0, [], FILE_C)

    names = {"triton_per_fused_2": None, "triton_poi_fused_0": sha("SRC0"), "triton_poi_fused_3": sha("SRC3"), "triton_red_fused_1": sha("SRC1")}
    assert doc["kernel_set_sha256"] == sha(json.dumps(sorted([name, digest] for name, digest in names.items())))
    binaries = sorted({b for kernel in kernels.values() for b in kernel["binaries"]})
    assert doc["binary_set_sha256"] == doc["cubin_set_sha256"] == sha(json.dumps(binaries))
    executed = [[name, kernel["src_sha256"], json.dumps([[e.get("id"), json.dumps(e.get("config"), sort_keys=True)] for e in kernel["executed"]],
                                                        sort_keys=True)] for name, kernel in sorted(kernels.items())]
    assert doc["executed_set_sha256"] == sha(json.dumps(executed))
    code = sorted([name, e["code_sha256"]] for name, kernel in kernels.items() for e in kernel["executed"] if e.get("code_sha256"))
    assert [name for name, _ in code] == ["triton_poi_fused_0", "triton_poi_fused_0", "triton_red_fused_1"]
    assert doc["executed_code_set_sha256"] == sha(json.dumps(code))
    normalised = sorted([name, e["code_sha256_normalized"]] for name, kernel in kernels.items() for e in kernel["executed"] if e.get("code_sha256"))
    assert doc["executed_code_set_sha256_normalized"] == sha(json.dumps(normalised))
    assert doc["module_keys_sha256"] == sha(json.dumps(sorted(["k" * 64, FILE_B, FILE_C])))
    assert sha(json.dumps(doc, sort_keys=True)) == WHOLE, "the whole document, as the function gave it when this test was written"


WHOLE = "baba6b60b05c6c93be8a3e7acc58237a20ce3e56b9615285db617c92a0bd8565"


def test_two_snapshots_of_one_process_are_compared_kernel_by_kernel(registry: Any, cache: Path) -> None:
    load(registry, cache)
    start = CI.loaded_code_objects()
    same = CI.loaded_code_objects_delta(start, CI.loaded_code_objects())
    assert same == {"modules_start": 3, "modules_end": 3, "new_modules": 0, "kernels_start": 4, "kernels_end": 4, "new_kernels": [],
                    "kernels_gone": [], "executed_changed": [], "binaries_changed": [], "sources_changed": [], "kernel_set_equal": True,
                    "executed_set_equal": True, "ok": True, "why": None,
                    "executed_set_sha256": {"start": start["executed_set_sha256"], "end": start["executed_set_sha256"]}}

    registry.modules = [*registry.modules, module(__file__="/inductor/dd/generated_d.py", triton_new_4=CachingAutotuner(source("triton_new_4", "SRC4")))]
    registry.modules[0].triton_poi_fused_0.launchers[0].bin = binary("h0c", cubin(b"SASS-C"))       # a re-tune: another binary runs
    registry.modules[0].triton_red_fused_1.fn.src = "SRC1, compiled again"
    del registry.modules[1].triton_per_fused_2
    end = CI.loaded_code_objects()
    moved = CI.loaded_code_objects_delta(start, end)
    assert moved["ok"] is False and (moved["modules_end"], moved["new_modules"], moved["kernels_end"]) == (4, 1, 4)
    assert (moved["new_kernels"], moved["kernels_gone"]) == (["triton_new_4"], ["triton_per_fused_2"])
    assert (moved["executed_changed"], moved["binaries_changed"], moved["sources_changed"]) == (
        ["triton_poi_fused_0"], ["triton_poi_fused_0"], ["triton_red_fused_1"])
    assert not moved["kernel_set_equal"] and not moved["executed_set_equal"]
    assert moved["executed_set_sha256"] == {"start": start["executed_set_sha256"], "end": end["executed_set_sha256"]}
    assert moved["why"] == ("the code objects / executed specialization changed between build and end of run: new_modules=1, "
                            "new_kernels=['triton_new_4'], executed_changed=['triton_poi_fused_0'], binaries_changed=['triton_poi_fused_0'], "
                            "sources_changed=['triton_red_fused_1']")


def test_a_snapshot_that_leaves_a_field_out_compares_as_one_that_holds_it_empty() -> None:
    """The fields the comparison reads with a default: a snapshot's `per_kernel` and `modules`, a kernel's `executed` and `binaries`."""
    whole = {"modules": 0, "kernels": 1, "kernel_set_sha256": "k", "executed_set_sha256": "e",
             "per_kernel": {"triton_poi_fused_0": {"src_sha256": "s", "executed": [], "binaries": []}}}
    short = {"kernels": 1, "kernel_set_sha256": "k", "executed_set_sha256": "e", "per_kernel": {"triton_poi_fused_0": {"src_sha256": "s"}}}
    null = {**short, "modules": None, "per_kernel": {"triton_poi_fused_0": {"src_sha256": "s", "executed": None, "binaries": None}}}
    for start, end in ((whole, short), (short, whole), (short, short), (whole, null), (null, short)):
        delta = CI.loaded_code_objects_delta(start, end)
        assert delta["ok"] is True and delta["new_modules"] == 0 and delta["why"] is None, (start, end)
        assert (delta["executed_changed"], delta["binaries_changed"], delta["sources_changed"]) == ([], [], [])
    for bare in ({"kernels": 0, "kernel_set_sha256": None, "executed_set_sha256": None}, {**short, "kernel_set_sha256": None, "per_kernel": None}):
        gone = CI.loaded_code_objects_delta(whole, bare)
        assert gone["ok"] is False and gone["kernels_gone"] == ["triton_poi_fused_0"] and gone["new_kernels"] == []
        new = CI.loaded_code_objects_delta(bare, whole)
        assert new["ok"] is False and new["new_kernels"] == ["triton_poi_fused_0"] and new["kernels_gone"] == []
    ran = {**whole, "per_kernel": {"triton_poi_fused_0": {"src_sha256": "s", "executed": [{"id": "code:1", "config": None}], "binaries": ["cubin:1"]}}}
    delta = CI.loaded_code_objects_delta(short, ran)
    assert (delta["ok"], delta["executed_changed"], delta["binaries_changed"]) == (False, ["triton_poi_fused_0"], ["triton_poi_fused_0"])


def test_a_snapshot_that_failed_is_no_comparison() -> None:
    good: dict[str, Any] = {"modules": 0, "per_kernel": {}}
    failed = {"error": "ImportError('no torch')"}
    assert CI.loaded_code_objects_delta(failed, good) == {"ok": None, "why": "snapshot unavailable: start ImportError('no torch') end None"}
    assert CI.loaded_code_objects_delta(good, failed) == {"ok": None, "why": "snapshot unavailable: start None end ImportError('no torch')"}
    nothing: Any = None
    assert CI.loaded_code_objects_delta(nothing, good) == {"ok": None, "why": "snapshot unavailable: start None end None"}
