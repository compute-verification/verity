"""Resolve the 16 stops of #9659/#9660 against #9364 (8854b824977): each conflict hunk takes #9364's side, then each file gets
back what #9659 changes in it.  python3 resolve.py WORKTREE"""
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
HUNK = re.compile(r"^<<<<<<< [^\n]*\n(.*?)^=======\n(.*?)^>>>>>>> [^\n]*\n", re.S | re.M)
CS = "from verity.definitions.call_scope import specialization\n"
PVS = "from verity_vllm.query.program_view import specialization\n"
PV = "from verity_vllm.query import program_view as PV\n"


def theirs(text: str) -> str:
    return HUNK.sub(lambda m: m.group(2), text)


def once(text: str, old: str, new: str, path: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{path}: {old!r} found {n} times")
    return text.replace(old, new)


def before(text: str, anchor: str, line: str, path: str) -> str:
    """``line`` inserted just before the line ``anchor`` (a whole line, found once)."""
    return once(text, anchor, line + anchor, path)


def fix(path: str, f) -> None:
    p = root / path
    if "<<<<<<<" not in p.read_text():
        print("already", path)
        return
    t = theirs(p.read_text())
    t = f(t, path)
    if "<<<<<<<" in t or ">>>>>>>" in t:
        raise SystemExit(f"{path}: markers left")
    p.write_text(t)
    print("resolved", path)


def swap_cs(t: str, path: str, anchor: str) -> str:
    """#9364's `call_scope` import gone, `program_view.specialization`'s just before ``anchor``."""
    t = once(t, CS, "", path)
    return before(t, anchor, PVS, path)


T = "adapters/vllm/tests/"
fix(T + "acquire/test_call_boundary_source.py",
    lambda t, p: swap_cs(t, p, "from verity_vllm.query.program_view import definition_ports, spec_statics\n"))
fix(T + "check/test_dense_replay_rows.py",
    lambda t, p: swap_cs(t, p, "from verity_vllm.query.program_view import spec_statics\n"))
fix(T + "pipeline/test_serving_rows_boolean.py",
    lambda t, p: swap_cs(t, p, "from verity_vllm.query.program_view import spec_base, spec_statics\n"))


def manifest_pool(t: str, p: str) -> str:
    return once(t, "    def prefetch(specs, X, W, extras, *, max_gates, jobs, budget):\n",
                "    def prefetch(specs, X, W, extras, *, max_gates, jobs, budget, specialize):\n", p)


fix(T + "pipeline/test_manifest_pool.py", manifest_pool)


def single_request(t: str, p: str) -> str:
    return once(t, "(\"from verity.definitions import word as W; W.specialization('GumbelTopPTokenSelect_v2', {'V': 16, 'S': 4}); \"\n",
                "(\"from verity_vllm.query.program_view import specialization; specialization('GumbelTopPTokenSelect_v2', \"\n"
                "                   \"{'V': 16, 'S': 4}); \"\n", p)


fix(T + "pipeline/test_single_request_build_path.py", single_request)


def with_pv(anchor: str):
    """`PV` imported just before ``anchor``; #9659's uses of it merged without a stop."""
    return lambda t, p: before(t, anchor, PV, p)


for f in ("program/test_fa2_check_inf.py", "program/test_fa2_softcap.py", "program/test_fa3_check_inf.py"):
    fix(T + f, with_pv("from verity_vllm.program.frontend.target_profile import TargetProfile\n"))


def pv_after(anchor: str):
    """`PV` imported just after ``anchor``."""
    return lambda t, p: once(t, anchor, anchor + PV, p)


fix(T + "query/test_cross_call.py", pv_after("from verity_vllm.pipeline import manifest as M\n"))
fix(T + "query/test_norm_scales.py", pv_after("from verity_vllm.query.units import ValueRef\n"))
fix(T + "query/test_router_softmax.py", pv_after("from verity_vllm.query.units import ValueRef\n"))


def word(t: str, p: str) -> str:
    t = once(t, "from verity_vllm.query.program_view import from_instances\n",
             "from verity_vllm.query import program_view as PV\nfrom verity_vllm.query.program_view import from_instances\n", p)
    return once(t, '    W.specialization("Gemm_v1", {"K": 16, "N": 1})\n', '    PV.specialization("Gemm_v1", {"K": 16, "N": 1})\n', p)


fix(T + "query/test_word.py", word)


def popread(t: str, p: str) -> str:
    t = once(t, "    from verity.definitions.call_scope import specialization\n    from verity.definitions.ops.fp32 import Bf16Add\n",
             "    from verity.definitions.ops.fp32 import Bf16Add\n", p)
    anchor = "    from verity.definitions.ops.scalar import BitAnd, I32Eq, SelectBf16\n    from verity.definitions.registry.prims import gather_bf16\n"
    return once(t, anchor, anchor + "    from verity_vllm.query.program_view import specialization\n", p)


fix(T + "tools/test_popread.py", popread)


def plan(t: str, p: str) -> str:
    t = once(t, "    from verity.definitions.call_scope import specialization\n", "", p)
    return once(t, "    from verity_vllm.query.program_view import spec_statics\n",
                "    from verity_vllm.query.program_view import spec_statics, specialization\n", p)


fix("adapters/vllm/verity_vllm/acquire/sources/call_boundary_plan.py", plan)


def manifest(t: str, p: str) -> str:
    return once(t, "from verity_vllm.query.program_view import from_instances, iter_instance_rows\n",
                "from verity_vllm.query.program_view import from_instances, iter_instance_rows, specialization\n", p)


fix("adapters/vllm/verity_vllm/pipeline/manifest.py", manifest)


def graph(t: str, p: str) -> str:
    t = once(t, "    spec_statics,\n)\n", "    spec_statics,\n)\nfrom verity_vllm.query.program_view import specialization as _specialization\n", p)
    if "_specialization = word_query.specialization" in t:
        t = once(t, "_specialization = word_query.specialization\n\n\n", "", p)
    return t


fix("adapters/vllm/verity_vllm/pipeline/program_graph.py", graph)
