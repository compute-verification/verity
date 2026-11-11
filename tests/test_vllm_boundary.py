"""Only the vLLM integration imports vLLM: outside ``adapters/vllm/``, no tracked Python file imports ``vllm``.

Code that touches vLLM goes through ``verity_vllm``'s interfaces, the public modules ``PUBLIC`` in
``tests/test_vllm_public_api.py`` lists: the pinned engine (``verity_vllm.LLM``), a built engine's vLLM objects
(``verity_vllm.engine.handle``), the scheduler's step trace (``verity_vllm.engine.step_trace``).  A plugin, a patch, a
stock kernel or a private vLLM internal that a benchmark needs goes into ``adapters/vllm/`` behind a public module:
the integration's ``vllm.general_plugins`` entry points (``adapters/vllm/pyproject.toml``) load it, its patches are
installed through ``verity_vllm.engine.hooks``, and one place answers for each vLLM version the integration pins.

``ALLOWED`` lists every file outside the integration that imports vLLM today, each with its owner and what it takes from
vLLM.  It only shrinks: a new importer fails, and so does a listed file that no longer imports vLLM or no longer exists.

An import is what ``tests/test_boundaries.py``'s ``_imports`` reads as imported: an import statement at any depth (a
function's lazy import included), or a string handed to ``importlib.import_module``, ``__import__`` or
``pytest.importorskip``.  A ``"module:attr"`` string is not one here, since vLLM names its metrics ``vllm:<name>``; a check
that vLLM is installed (``importlib.util.find_spec("vllm")``) imports nothing, and a shell script's
``python -c 'import vllm'`` is no Python file's import.
"""
from __future__ import annotations

import ast
import subprocess

from test_boundaries import _imports
from test_repository import ROOT

INTEGRATION = "adapters/vllm/"
#: The fix a failure names.
ROUTE = ("route the code through verity_vllm's interface (PUBLIC in tests/test_vllm_public_api.py): verity_vllm.LLM builds "
         "and drives the pinned engine, verity_vllm.engine.handle hands over a built engine's vLLM objects and "
         "verity_vllm.engine.step_trace its scheduler's steps; a plugin, a patch, a stock kernel or a private vLLM internal "
         "goes into adapters/vllm/ behind a public module you import (a vllm.general_plugins entry point in its "
         "pyproject.toml, patches through verity_vllm.engine.hooks)")

#: Every file outside the integration that imports vLLM, with its owner and what it takes.  Move it behind verity_vllm's
#: interface and delete its line.  It only shrinks.
ALLOWED = {
    # stock engines a benchmark builds and drives beside the integration's
    "benchmarks/harness/verity_bench/vllm/runner.py": "benchmarks: verity-bench's vLLM arms, each a stock engine's requests",
    "benchmarks/served_commit/tp_decode.py": "circuits: a stock vllm.LLM at TP8 against a Commit arm in the same engine",
    "archive/benchmarks/pouw/np_kernel/stock_decode.py": "compute-accounting: a stock vllm.LLM step, T2's denominator",
    **dict.fromkeys(("benchmarks/pouw/pearl_c_vllm/e2e.py", "benchmarks/pouw/pearl_c_vllm/profile_decode.py",
                     "benchmarks/pouw/pearl_c_vllm/served_passes.py"),
                    "compute-accounting: Pearl-C served through verity_vllm.LLM, with vLLM's SamplingParams and a stock "
                    "vllm.LLM beside it"),
    **dict.fromkeys(("benchmarks/kv_filler/serve.py", "benchmarks/kv_filler/step_profile.py"),
                    "memory-accounting: a stock vllm.LLM serving under the KV filler's top-up, and its decode step profiled"),
    "benchmarks/pous/lateness/vllm_load.py": "memory-accounting: a stock vllm.LLM as load on the responder's GPU",
    "benchmarks/pous/p2_v1/scale.py": "memory-accounting: P2's store held beside a stock vllm.LLM serving",
    **dict.fromkeys(("benchmarks/remote_session/engine.py", "benchmarks/remote_session/serve.py"),
                    "memory-accounting: an in-process LLMEngine under the remote bandwidth audit's binder"),
    "benchmarks/network_traces/record.py": "network-accounting: per-token timing of served requests (SamplingParams)",
    # the stock kernels a benchmark times or gates against
    **dict.fromkeys(("benchmarks/hash/exactlin/kbench.py", "benchmarks/hash/exactlin/kgate.py",
                     "benchmarks/hash/exactlin/ksweep.py"),
                    "kernels: vLLM's int8 GEMMs (_custom_ops.cutlass_scaled_mm, triton_scaled_mm) as the exact linear's "
                    "kernel-level baseline and gate"),
    **dict.fromkeys(("benchmarks/torch_gpu/s8_linear/bench.py", "benchmarks/torch_gpu/s8_linear/run.py"),
                    "kernels: vLLM's int8 cutlass_scaled_mm as the plain GEMM the exact W8A8 linear is timed against"),
    **dict.fromkeys(("benchmarks/pouw/harness/torch_baselines.py", "benchmarks/pouw/pearl_c_sm90/glm53_timing.py",
                     "benchmarks/pouw/pearl_c_sm90/grouped_check.py"),
                    "compute-accounting: the FP8 GEMMs vLLM serves with (cutlass_scaled_mm, DeepGEMM, fp8_utils) as "
                    "Pearl-C's baselines"),
    "benchmarks/pouw/gemm_rate/attn_rate.py": "compute-accounting: vLLM's flash-attention build, the served prefill attention "
                                              "whose MAC rate is measured",
    # served arms that replace a layer's kernels inside a stock engine
    **dict.fromkeys(("benchmarks/hash/dither/dither_a.py", "benchmarks/hash/dither/dither_fused.py",
                     "benchmarks/hash/dither/dither_one.py", "benchmarks/hash/dither/dither_so.py",
                     "benchmarks/hash/dither/native.py", "benchmarks/hash/dither/sbench.py"),
                    "kernels: Dither's served arms, a stock vllm.LLM per arm with its int8 linears swapped in process"),
    **dict.fromkeys(("benchmarks/hash/exactlin/percall.py", "benchmarks/hash/exactlin/percall_diag.py",
                     "benchmarks/hash/exactlin/share.py"),
                    "kernels: sbench's served arms over stock vLLM int8 (TritonInt8ScaledMMLinearKernel), and why they "
                    "part"),
    **dict.fromkeys(("benchmarks/moe_align/served.py", "benchmarks/moe_align/stress.py"),
                    "kernels: verity_vllm.engine.moe_align's canonical moe_align_block_size held to vLLM's op and to its "
                    "Marlin MoE path, served at TP8"),
    **dict.fromkeys(("benchmarks/vllm_encoded/kernels.py", "benchmarks/vllm_encoded/selftest.py",
                     "benchmarks/vllm_encoded/serve.py", "benchmarks/vllm_encoded/stack.py",
                     "benchmarks/vllm_encoded/tests/properties/encoded_kv_append_capture_gpu.py"),
                    "memory-accounting: the encoded-weights stack inside vLLM, its linears, attention and KV append in "
                    "place of vLLM's"),
    # vLLM general plugins and the engine-core internals they patch
    **dict.fromkeys(("benchmarks/network_e2e/traces/vllm_trace_plugin.py",
                     "benchmarks/network_e2e/traces/vllm_adversarial_plugin.py"),
                    "network-accounting: general plugins that record the V1 scheduler's steps, or make it hostile"),
    **dict.fromkeys(("benchmarks/network_e2e/encodings/sched_trace.py", "benchmarks/network_e2e/encodings/vllm_trace.py"),
                    "network-accounting: vLLM V1's own scheduler driven on CPU over a fake executor, for its step traces"),
    **dict.fromkeys(("benchmarks/dsa_topk/dsa_topk.py", "benchmarks/dsa_topk/stress.py"),
                    "proofs: a general plugin making the DSA indexer's top-k (_custom_ops) a function of its inputs, and "
                    "its stress"),
    **dict.fromkeys(("benchmarks/replay_ladder/ladder.py", "benchmarks/replay_ladder/topk_probe.py"),
                    "proofs: each ladder model's window replayed by a stock engine, and a probe of its DSA prefill top-k"),
    **dict.fromkeys(("benchmarks/serve_tax/rebuild.py", "benchmarks/serve_tax/replay.py", "benchmarks/serve_tax/restore.py",
                     "benchmarks/serve_tax/servetax_plugin.py", "benchmarks/serve_tax/tp_edge.py",
                     "benchmarks/serve_tax/tp_restore.py", "benchmarks/serve_tax/tp_state.py"),
                    "proofs: the two-stage replay's serve tax: an engine core's drained edges, its per-rank boundary "
                    "state and the replay from it (EngineCore, GPUModelRunner, collective_rpc)"),
    "benchmarks/spot_proved/capture.py": "proofs: the recorder, a general plugin recording a replayed step's interior per TP "
                                         "rank under CUDA graphs",
    "benchmarks/spot_proved/tree_allreduce.py": "proofs: the TP all-reduce summed in software over vLLM's CUDA "
                                                "communicator, as the serving node's multimem tree reduced it",
    "benchmarks/spot_proved/discover.py": "proofs: a rehearsal of the capture on a GLM-5.3 vllm.LLM with dummy weights",
    "benchmarks/route_capture/route_capture_plugin.py": "proofs: a general plugin recording each served step's launch-shape "
                                                        "class and kernels",
    "benchmarks/route_capture/tests/test_plugin.py": "proofs: the route capture's plugin on stand-ins for vLLM's model "
                                                     "runners, imported under vLLM's module names",
}


def _importers(texts: dict[str, str]) -> dict[str, list[str]]:
    """Each file of ``texts`` (repository path -> source) outside the integration that imports vLLM, with where."""
    out = {}
    for rel, text in texts.items():
        if rel.startswith(INTEGRATION) or "vllm" not in text:
            continue
        hits = sorted({(line, name) for name, line, imported in _imports(ROOT / rel, ROOT, (), ast.walk(ast.parse(text, filename=rel)))
                       if imported and name.split(".")[0] == "vllm"})
        if hits:
            out[rel] = [f"{line}: {name}" for line, name in hits]
    return out


def _tracked_vllm_texts() -> dict[str, str]:
    """Every tracked Python file outside the integration that names vLLM at all, by path."""
    files = subprocess.run(["git", "grep", "-l", "-F", "vllm", "--", "*.py", f":!{INTEGRATION}"], cwd=ROOT,
                           capture_output=True, text=True, check=True).stdout.split()
    return {rel: (ROOT / rel).read_text(encoding="utf-8") for rel in files}


def test_only_the_integration_imports_vllm() -> None:
    found = _importers(_tracked_vllm_texts())
    new = [f"{f}:{hit}" for f, hits in sorted(found.items()) if f not in ALLOWED for hit in hits]
    assert not new, (f"outside {INTEGRATION} nothing imports vLLM: {ROUTE}. ALLOWED in tests/test_vllm_boundary.py only "
                     "shrinks.\n" + "\n".join(new))
    stale = sorted(set(ALLOWED) - set(found))
    assert not stale, ("these files no longer import vLLM, or are gone; delete them from ALLOWED in "
                       "tests/test_vllm_boundary.py:\n" + "\n".join(f'    "{f}",' for f in stale))


def test_a_vllm_import_is_found() -> None:
    """The check above isn't vacuous: an import at any depth and an import call are each found; a presence check, a
    metric's ``vllm:<name>``, a module whose name only starts with ``vllm``, ``verity_vllm``, a docstring or a string that
    mentions vLLM, and the integration's own files are not."""
    texts = {
        "b/top.py": "import vllm\nfrom vllm.v1.core.sched.scheduler import Scheduler as S\n",
        "b/lazy.py": ("def f():\n    from vllm import LLM\n    class C:\n        def g(self):\n"
                      "            import vllm.distributed as d\n            return d\n    return LLM\n"),
        "b/calls.py": ("import importlib, pytest\npytest.importorskip('vllm')\nimportlib.import_module('vllm._custom_ops')\n"
                       "v = __import__('vllm').__version__\n"),
        "b/clean.py": ('"""read from vllm-flash-attn"""\nimport importlib.util\nimport vllm_v1\n'
                       "from verity_vllm import LLM\nfrom verity_vllm.engine import vllm_adapter\n"
                       "ok = importlib.util.find_spec('vllm') is not None\nNOTE = 'from vllm import LLM'\n"
                       "STEPS = 'vllm:iteration_tokens_total_count'\n"),
        "adapters/vllm/verity_vllm/x.py": "import vllm\n",
    }
    assert _importers(texts) == {
        "b/top.py": ["1: vllm", "2: vllm.v1.core.sched.scheduler"],
        "b/lazy.py": ["2: vllm", "5: vllm.distributed"],
        "b/calls.py": ["2: vllm", "3: vllm._custom_ops", "4: vllm"],
    }, _importers(texts)
