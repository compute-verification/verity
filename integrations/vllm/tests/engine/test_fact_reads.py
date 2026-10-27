"""engine/'s fact readers give their fallback, never raise, when a fact can't be read: an engine without the config field, a host
file that is absent or unparsable, a torch without CUDA, a tool that isn't installed, a result type `dataclasses.replace` can't
copy."""
from __future__ import annotations

import dataclasses
import importlib.metadata
import subprocess
import sys
from types import ModuleType
from types import SimpleNamespace as NS

import pytest

from verity_vllm.engine import build, engine_profile, run_facts
from verity_vllm.pipeline.tp import rank_worker


def _module(name: str, **attrs) -> ModuleType:
    m = ModuleType(name)
    m.__dict__.update(attrs)
    return m


@pytest.mark.parametrize("llm, want", [
    (NS(llm_engine=NS(vllm_config=NS(scheduler_config=NS(async_scheduling=1)))), True),
    (NS(llm_engine=NS(vllm_config=NS(scheduler_config=NS(async_scheduling=0)))), False),
    (NS(llm_engine=NS(vllm_config=NS(scheduler_config=NS(async_scheduling=None)))), None),
    (NS(llm_engine=NS(vllm_config=NS(scheduler_config=NS()))), None),
    (NS(), None),
])
def test_resolved_async_scheduling_or_none(llm, want):
    assert build.resolved_async_scheduling(llm) is want


class _Model:
    def __init__(self, *versions):
        self.mods = [(f"l{i}", NS(impl=NS(vllm_flash_attn_version=v))) for i, v in enumerate(versions)]

    def named_modules(self):
        return iter(self.mods)


@pytest.mark.parametrize("model, want", [
    (_Model(2, 2), 2),
    (_Model(2, 3), None),
    (_Model("x"), None),
    (_Model([2]), None),
    (NS(), None),
])
def test_effective_execution_doc_reads_the_flash_attn_version_or_none(model, want):
    doc = build.effective_execution_doc(NS(), model)
    assert doc["flash_attn_version"] == want
    assert "compilation_error" in doc and "config_error" in doc


def _host_files(monkeypatch, files: dict):
    class P:
        def __init__(self, p):
            self.p = p

        def read_text(self):
            v = files.get(self.p, FileNotFoundError(self.p))
            if isinstance(v, BaseException):
                raise v
            return v

    monkeypatch.setattr(run_facts, "Path", P)


CPUINFO, CPU_MAX = "/proc/cpuinfo", "/sys/fs/cgroup/cpu.max"
CLOCK = "/sys/devices/system/clocksource/clocksource0/current_clocksource"
BAD_UTF8 = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")


@pytest.mark.parametrize("files, want", [
    ({}, {}),
    ({CPUINFO: "processor\t: 0\nmodel name\t: Some CPU\n", CPU_MAX: "max 100000", CLOCK: "tsc\n"},
     {"cpu": "Some CPU", "cpu_quota": None, "clocksource": "tsc"}),
    ({CPU_MAX: "200000 100000"}, {"cpu_quota": 2.0}),
    ({CPU_MAX: ""}, {}),
    ({CPU_MAX: "a b"}, {}),
    ({CPU_MAX: "100 0"}, {}),
    ({CPUINFO: BAD_UTF8, CPU_MAX: PermissionError(CPU_MAX), CLOCK: IsADirectoryError(CLOCK)}, {}),
])
def test_host_doc_keeps_the_facts_it_reads(monkeypatch, files, want):
    _host_files(monkeypatch, files)
    doc = run_facts.host_doc()
    assert {k: v for k, v in doc.items() if k not in ("nproc", "clock_read_us")} == want
    assert set(doc) >= {"nproc", "clock_read_us"}


def _no_device(exc):
    def props(_):
        raise exc
    return props


@pytest.mark.parametrize("props, gpu", [
    (_no_device(AssertionError("Torch not compiled with CUDA enabled")), None),
    (_no_device(RuntimeError("Found no NVIDIA driver on your system")), None),
    (_no_device(run_facts.torch.cuda.DeferredCudaCallError("CUDA call failed lazily at initialization")), None),
    (lambda _: NS(name="G", major=9, minor=0, multi_processor_count=132), {"name": "G", "cc": "9.0", "sms": 132}),
])
@pytest.mark.parametrize("export, want_export", [(None, None), ("{bad", None), ('{"a": 1}', {"a": 1}), ("<dir>", None)])
def test_versions_doc_keeps_the_facts_it_reads(monkeypatch, tmp_path, props, gpu, export, want_export):
    monkeypatch.setitem(sys.modules, "triton", _module("triton", __version__="t"))
    monkeypatch.setitem(sys.modules, "vllm", _module("vllm", __version__="v"))
    real = importlib.metadata.version

    def version(name):
        if name == "flashinfer-python":
            raise importlib.metadata.PackageNotFoundError(name)
        return "4.0" if name == "transformers" else real(name)

    monkeypatch.setattr(importlib.metadata, "version", version)
    monkeypatch.setattr(run_facts.torch.cuda, "get_device_properties", props)
    monkeypatch.setattr(run_facts, "host_doc", lambda: {})
    monkeypatch.chdir(tmp_path)
    if export == "<dir>":
        (tmp_path / "EXPORT.json").mkdir()
    elif export is not None:
        (tmp_path / "EXPORT.json").write_text(export)
    doc = run_facts.versions_doc()
    assert (doc["triton"], doc["vllm"], doc["flashinfer"], doc["transformers"]) == ("t", "v", None, "4.0")
    assert doc.get("gpu") == gpu
    assert doc.get("export") == want_export
    assert doc["host"] == {}


@pytest.mark.parametrize("exc", [FileNotFoundError("nvidia-smi"), PermissionError("nvidia-smi"),
                                 subprocess.TimeoutExpired(["nvidia-smi"], 30), BAD_UTF8])
def test_run_gives_empty_output_when_the_tool_cannot_run(monkeypatch, exc):
    def run(*a, **k):
        raise exc

    monkeypatch.setattr(subprocess, "run", run)
    assert engine_profile._run(["nvidia-smi"]) == ""


def test_run_gives_the_tools_stripped_output():
    assert engine_profile._run([sys.executable, "-c", "print(' ok ')"]) == "ok"


@dataclasses.dataclass(frozen=True)
class _RC:
    run_root: bytes
    other: int


@pytest.mark.parametrize("rc", [_RC(b"\x01", 5), NS(run_root=b"\x01", other=5)])
def test_claimed_puts_the_drivers_root_on_a_copy(rc):
    claimed = rank_worker.TP2CaptureWorkerExtension._claimed
    assert claimed(None, rc, None) is rc and claimed(None, rc, "01") is rc
    got = claimed(None, rc, "02")
    assert got is not rc and (got.run_root, got.other, type(got)) == (b"\x02", 5, type(rc)) and rc.run_root == b"\x01"
    with pytest.raises(ValueError):
        claimed(None, rc, "zz")
