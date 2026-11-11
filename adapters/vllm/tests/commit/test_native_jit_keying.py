"""The native collector's JIT build is keyed by its SOURCE SET, and the loaded .so must expose every
name the source binds -- the two rules that make a stale .so (another tree's build that torch's mtime test sees as up to date)
impossible to adopt and impossible to miss.  Torch-free: runs on the laptop and in the pod gate."""
from __future__ import annotations

import os
import types

from verity_vllm.commit.committer import native_jit as nj

HERE = os.path.dirname(os.path.abspath(__file__))
CPP = os.path.join(HERE, "..", "..", "verity_vllm", "commit", "committer", "native_collect.cpp")


def _write(tmp_path, name, data: bytes):
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


def test_source_set_digest_changes_with_any_source_byte_and_with_the_flags_and_toolchain(tmp_path):
    a = _write(tmp_path, "native_collect.cpp", b"int x;")
    b = _write(tmp_path, "native_gather.cu", b"__global__ void k(){}")
    c = _write(tmp_path, "native_rows.cu", b"__global__ void t(){}")
    d0 = nj.source_set_digest([a, b, c], torch_version="2.8.0", cuda_version="12.9")
    assert d0 == nj.source_set_digest([a, b, c], torch_version="2.8.0", cuda_version="12.9"), "stable over identical input"
    # touching the mtime changes NOTHING -- the digest is the content, never the clock
    os.utime(a, (1, 1))
    assert nj.source_set_digest([a, b, c], torch_version="2.8.0", cuda_version="12.9") == d0
    a2 = _write(tmp_path, "native_collect.cpp", b"int x; int64_t plan_version() { return 1; }")
    assert nj.source_set_digest([a2, b, c], torch_version="2.8.0", cuda_version="12.9") != d0, "one source byte -> another key"
    assert nj.source_set_digest([a, b, c], torch_version="2.9.0", cuda_version="12.9") != d0, "toolchain in the key"
    assert nj.source_set_digest([a, b, c], cflags=("-O2",), torch_version="2.8.0", cuda_version="12.9") != d0, "flags in the key"
    assert len(d0) == 64


def test_keyed_build_dir_is_a_child_of_the_named_parent():
    d = "ab12cd34ef56" + "0" * 52
    assert nj.keyed_build_dir("/workspace/cp/nc_build", d) == "/workspace/cp/nc_build/ab12cd34ef56"
    assert nj.keyed_build_dir("/workspace/cp/stoch/nc_build_int3", d) == "/workspace/cp/stoch/nc_build_int3/ab12cd34ef56"


SAMPLE_CPP = r'''
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<StepHasher>(m, "StepHasher")
      .def(py::init<>())
      .def("__call__", &StepHasher::call);
  py::class_<Collector>(m, "Collector")
      .def("begin_step", &Collector::begin_step)
      .def("plan_used", &Collector::plan_used, py::arg("key"))
      .def("plan_version", &Collector::plan_version, py::arg("plan_key"))
      .def_readwrite("use_gather", &Collector::use_gather)
      .def_readonly("active", &Collector::active)
      .def_property_readonly("stats", &Collector::stats);
  m.def("cuda_runtime_version", []() { int v = 0; return v; });
}
'''


def test_binding_names_reads_every_def_form_and_the_classes(tmp_path):
    p = _write(tmp_path, "native_collect.cpp", SAMPLE_CPP.encode())
    assert nj.binding_names(p) == {"__call__", "begin_step", "plan_used", "plan_version", "use_gather", "active", "stats",
                                   "cuda_runtime_version"}
    assert nj.class_names(p) == {"StepHasher", "Collector"}


def _fake_module(collector_names, module_funcs=("cuda_runtime_version",), classes=("StepHasher", "Collector")):
    m = types.ModuleType("verity_native_collect")
    for c in classes:
        cls = type(c, (), {n: (lambda self: None) for n in (collector_names if c == "Collector" else ("__call__",))})
        setattr(m, c, cls)
    for f in module_funcs:
        setattr(m, f, lambda: 0)
    return m


def test_missing_bindings_names_exactly_what_a_stale_so_lacks(tmp_path):
    p = _write(tmp_path, "native_collect.cpp", SAMPLE_CPP.encode())
    fresh = _fake_module(("begin_step", "plan_used", "plan_version", "use_gather", "active", "stats"))
    assert nj.missing_bindings(fresh, p) == []
    # the cp-big shape: the .so predates fp8's plan_version binding
    stale = _fake_module(("begin_step", "plan_used", "use_gather", "active", "stats"))
    assert nj.missing_bindings(stale, p) == ["plan_version"]
    # a whole class missing is named first
    no_hasher = _fake_module(("begin_step", "plan_used", "plan_version", "use_gather", "active", "stats"), classes=("Collector",))
    assert nj.missing_bindings(no_hasher, p) == ["StepHasher", "__call__"]


def test_the_real_source_binds_the_names_the_python_side_calls():
    names = nj.binding_names(CPP)
    for n in ("begin_step", "end_step", "rows_leaves", "rows_tree_window", "plan_used", "stats", "cuda_runtime_version"):
        assert n in names, n
    assert {"StepHasher", "Collector"} <= nj.class_names(CPP)
    assert len(names) >= 40


def test_release_json_records_the_collector_source_digest():
    """release_json keeps the source_digest= / bindings= a collector build line prints beside the sha."""
    from verity_vllm.pipeline import release_json as rj

    arts = rj.parse_artifacts("FA2-TAP-OK 2222 /b.so arch=8.9",
                              "3333 /workspace/cp/nc_build/ab12cd34ef56/verity_native_collect.so source_digest=ab12cd34ef56ff bindings=48")
    assert arts["verity_native_collect.so"] == {"sha256": "3333", "path": "/workspace/cp/nc_build/ab12cd34ef56/verity_native_collect.so",
                                                "source_digest": "ab12cd34ef56ff", "bindings": "48"}
    # the two-token line still parses (records written by earlier releases)
    assert rj.parse_artifacts("FA2-TAP-OK 2 /b", "3 /nc/verity_native_collect.so")["verity_native_collect.so"] == \
        {"sha256": "3", "path": "/nc/verity_native_collect.so"}


def test_ext_keys_its_build_dir_by_the_digest_and_refuses_a_stale_load():
    src = open(os.path.join(HERE, "..", "..", "verity_vllm", "commit", "committer", "native_ext.py")).read()
    assert "native_jit.jit_load(\"verity_native_collect\", native_jit.source_paths(HERE)" in src
    assert "missing = native_jit.missing_bindings(_EXT, os.path.join(HERE, \"native_collect.cpp\"))" in src
    assert "raise RuntimeError" in src.split("missing = native_jit.missing_bindings")[1][:400]
