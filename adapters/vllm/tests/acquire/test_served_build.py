"""The served Build (`acquire.served`) from a served-compiled process's own artifact: Llama-3.2-1B served at vLLM's defaults on an
H200 (`fixtures/served_llama.tar.gz`: the three Inductor wrappers its 17 pieces run, vLLM's split graph, the kernels' launch configs,
the cuBLAS routes probed on that GPU, and the four-request mixed workload).  The Build reads every wrapper line, binds every launch
to a Definition, and reproduces the CUDA-graph mode, padding and FA3 split count of each of the 14 steps the process served; an
autotuned kernel binds the one config its autotuners ran; a batched-copy flush line is no call; a line, a kernel source, a hook or a
launch config it does not model is refused by name, and a GEMM route it lacks leaves it incomplete."""
from __future__ import annotations

__models__ = ("unsloth/Llama-3.2-1B",)

import copy
import dataclasses
import inspect
import json
import os
import re
import shutil
import tarfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from verity_vllm.acquire import compiled as C
from verity_vllm.acquire import inductor_templates as T
from verity_vllm.acquire import served as SV
from verity_vllm.acquire import served_device as SD
from verity_vllm.acquire.inductor_templates import Unmodelled
from verity_vllm.acquire.sources.inductor_hooks import FLUSH

FIXTURE = Path(__file__).parent / "fixtures" / "served_llama.tar.gz"
#: the fixture's GPU, an H200
CC = (9, 0)
#: (mode, padded, tokens) of each engine step, as the serving process dispatched them (its capture's record)
SERVED = [("NONE", 137, 137), ("FULL", 2, 2), ("FULL", 2, 2), ("PIECEWISE", 16, 11), ("FULL", 4, 3), ("FULL", 4, 3),
          ("NONE", 303, 303), ("FULL", 4, 4), ("FULL", 4, 4), ("FULL", 4, 3), ("FULL", 4, 3), ("FULL", 2, 2), ("FULL", 1, 1),
          ("FULL", 1, 1)]


@pytest.fixture(scope="module")
def art(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("served")
    with tarfile.open(FIXTURE) as t:
        t.extractall(d, filter="data")
    return d / "served_llama"


def _build(d: Path, inductor: Path | None = None, **kw):
    wl = json.loads((d / "workload.json").read_text())
    args = dict(split_graph=str(d / "computation_graph.py"), launch_configs=json.loads((d / "launch_configs.json").read_text()),
                routes=json.loads((d / "routes.json").read_text()), schedule=C.schedule_of_workload(wl), steps=14,
                capture_sizes=[1, 2, 4, 8, 16], max_num_seqs=8, device=SD.device_of(CC, 132))
    return SV.build(str(inductor or d / "inductor"), **{**args, **kw})


def _copy(art: Path, tmp: Path) -> Path:
    shutil.copytree(art / "inductor", tmp / "inductor")
    return tmp / "inductor"


def _edit(inductor: Path, old: str, new: str, count: int = 1) -> None:
    hits = [p for p in sorted(inductor.rglob("*.py")) if old in p.read_text()]
    assert hits, old
    for p in hits:
        p.write_text(p.read_text().replace(old, new, count))


def test_the_build_is_the_served_computation(art: Path) -> None:
    m = _build(art)
    p = m["program"]
    assert m["complete"] and not m["routes_missing"]
    assert m["kind"] == SV.KIND == "verity/served-required-manifest/v1"
    assert len(p["piece_wrappers"]) == 17 and len(set(p["piece_wrappers"])) == 3
    assert Counter(c.get("base", c["op"]) for c in p["calls"]) == {
        "mm": 64, "InductorRoPE_v1": 32, "InductorAddRMSNorm_v1": 16, "InductorSiluMul_v1": 16, "InductorAdd2RMSNorm_v1": 15,
        "Embedding_v1": 1, "InductorRMSNorm_v1": 1, "InductorAdd2RMSNormFinal_v1": 1}
    assert len(p["attention"]) == 16 and len(p["values"]) == 179 and p["final"]["value"] == "c16.mul/buf7"
    assert [(s["mode"], s["padded"], s["tokens"]) for s in m["steps_plan"]] == SERVED
    assert [s["step"] for s in m["steps_plan"] for r in s["requests"] if r["NS"] > 1] == [7, 8, 9, 10, 11]
    assert {r["NS"] for s in m["steps_plan"] for r in s["requests"]} == {1, 2}
    assert len(m["identities"]) == 6697
    assert m["program_digest"] == ("1f2df5e1029f878debf647a562314108b4e15159e90c5a852adefe9ade198707"
                                   "7c2b4cbf5621209148cdafbdc1113f15cbafd7a9c3f38d11a70659c291062a6b")
    assert m["artifact"]["kernels"] == {sha: k["config"] for sha, k in p["kernels"].items()} and len(m["artifact"]["digest"]) == 128


def test_the_gemm_route_follows_the_step(art: Path) -> None:
    plans = {s["step"]: s for s in _build(art)["steps_plan"]}
    def down(s: int) -> dict:
        return plans[s]["gemms"]["8192,2048"]

    assert down(1)["base"] == "CublasSplitKGemm_v1" and down(1)["statics"]["SPLIT_K"] == 3
    assert down(0)["statics"]["SPLIT_K"] == 2
    assert down(6)["base"] == "Gemm_v2"


NO_ROUTES: list[dict] = [{}, {"routes": None}, {"routes": {}}, {"routes": []}, {"routes": ""}, {"routes": 0}, {"routes": False}]


@pytest.mark.parametrize("routes", NO_ROUTES, ids=repr)
def test_a_missing_route_leaves_the_build_incomplete(art: Path, routes: dict) -> None:
    m = _build(art, routes=routes)
    assert not m["complete"] and len(m["routes_missing"]) == 28
    assert all(g is None for s in m["steps_plan"] for g in s["gemms"].values())


@pytest.mark.parametrize("old,new,why", [
    ("        args.clear()\n", "        args.clear()\n        print(args)\n", "wrapper line not modelled"),
    ("tmp3 = libdevice.exp(tmp2)", "tmp3 = libdevice.exp2(tmp2)", "template"),
    ("'enable_fp_fusion': True", "'enable_fp_fusion': False", "enable_fp_fusion"),
])
def test_an_unmodelled_artifact_is_refused(art: Path, tmp_path: Path, old: str, new: str, why: str) -> None:
    ind = _copy(art, tmp_path)
    _edit(ind, old, new, count=-1)
    with pytest.raises(Unmodelled):
        _build(art, inductor=ind)


def test_an_unhooked_buffer_is_refused(art: Path, tmp_path: Path) -> None:
    ind = _copy(art, tmp_path)
    _edit(ind, "        run_intermediate_hooks('verity|mul|buf4', buf4)\n", "", count=-1)
    with pytest.raises(Unmodelled, match="no run_intermediate_hooks names them"):
        _build(art, inductor=ind)


def test_a_piece_argument_of_no_known_source_is_refused(art: Path, tmp_path: Path) -> None:
    graph = tmp_path / "computation_graph.py"
    src = (art / "computation_graph.py").read_text()
    assert "_weight_, l_positions_, " in src
    graph.write_text(src.replace("_weight_, l_positions_, ", "_weight_, positions, ", 1))
    with pytest.raises(Unmodelled, match=r"piece 0 argument 5 \(positions\): a unknown argument"):
        _build(art, split_graph=str(graph))


@pytest.mark.parametrize("operand,why", [
    ("output_3", "an attention op's output buffer is not a piece's return"),
    ("query_2", "piece 1 mm reads storage no launch wrote"),
    ("key_2", "attention 0: its k is not a piece's return"),
    ("value", "attention 0: its v is not a piece's return"),
])
def test_an_attention_op_whose_signature_leaves_an_operand_out_is_refused(art: Path, tmp_path: Path, operand: str, why: str) -> None:
    """The wiring names an attention op's query, key, value and output by its signature's parameter names; one it cannot name is
    left out of the op's record, and the build is refused."""
    graph = tmp_path / "computation_graph.py"
    src = (art / "computation_graph.py").read_text()
    signature = next(line for line in src.splitlines() if line.lstrip().startswith("def forward(self, key_2: "))
    assert f"{operand}: " in signature
    graph.write_text(src.replace(signature, signature.replace(f"{operand}: ", f"x_{operand}: "), 1))
    with pytest.raises(Unmodelled, match=re.escape(why)):
        _build(art, split_graph=str(graph))


def test_a_launch_reads_a_view_at_the_views_width(art: Path, tmp_path: Path) -> None:
    ind = _copy(art, tmp_path)
    kernel = "triton_red_fused_fused_add_rms_norm_0"
    old = f"{kernel}.run(buf0, arg4_1, arg3_1, buf2, s72, 2048, stream=raw_stream0)"
    path = next(p for p in sorted(ind.rglob("*.py")) if old in p.read_text())

    def launch() -> dict:
        return next(la for la in C.parse_wrapper(str(path))["launches"] if la["kernel"] == kernel)

    whole = launch()
    assert whole["args"][0] == "buf0" and whole["widths"][0] == 2048
    view = "reinterpret_tensor(buf0, (s72, 8, 64), (512, 64, 1), 0)"
    _edit(ind, old, old.replace("buf0", view))
    assert launch()["args"][0] == view and launch()["widths"] == [512, *whole["widths"][1:]]


#: the first piece's two launches: its embedding with the first norm, and the RoPE of its first GEMM's rows
EMBED = "triton_red_fused__to_copy_embedding_rms_norm_0"
EMBED_RUN = f"{EMBED}.run(arg0_1, arg2_1, arg3_1, buf0, buf2, s72, 2048, stream=raw_stream0)"
ROPE_RUN = "triton_poi_fused_1.run(buf3, arg5_1, arg6_1, buf4, buf5, triton_poi_fused_1_xnumel_0, triton_poi_fused_1_xnumel_1, stream=raw_stream0)"


@pytest.fixture
def first(art: Path, tmp_path: Path) -> Path:
    """A copy of the fixture's wrappers; the path of the first piece's."""
    return next(p for p in sorted(_copy(art, tmp_path).rglob("*.py")) if EMBED_RUN in p.read_text())


def _edit_call(path: Path, old: str, new: str) -> None:
    """One edit of a wrapper's `call()`: the example-value stubs above it hold some of its lines too."""
    head, mark, call = path.read_text().partition("    def call(self, args):")
    assert call.count(old) == 1, old
    path.write_text(head + mark + call.replace(old, new))


def test_a_view_named_beside_its_live_buffer_is_the_buffers_value_at_the_views_shape(first: Path) -> None:
    """A line that names a view of a buffer and leaves the buffer alive gives the name the view's shape, and the launch that
    wrote the buffer where one did."""
    whole = C.parse_wrapper(str(first))
    _edit_call(first, f"            {ROPE_RUN}", "            cols = reinterpret_tensor(buf3, (s72, 8, 64), (3072, 64, 1), 2560)\n"
               f"            {ROPE_RUN.replace('buf3', 'cols')}")
    _edit_call(first, "        return (", "        emb = reinterpret_tensor(buf0, (s72, 32, 64), (2048, 64, 1), 0)\n"
               "        out = reinterpret_tensor(buf6, (s72, 32, 64), (2048, 64, 1), 0)\n        return (")
    _edit_call(first, "reinterpret_tensor(buf6, (s72, 32, 64), (2048, 64, 1), 0), buf0, )", "out, emb, )")
    doc = C.parse_wrapper(str(first))
    rope, was = doc["launches"][1], whole["launches"][1]
    assert (was["args"][0], was["widths"][0]) == ("buf3", 3072) and (rope["args"][0], rope["widths"]) == ("cols", [512, *was["widths"][1:]])
    assert rope["edges"][0] == {"arg": 0, "role": "qkv", "from": "undefined", "base": "cols"}
    assert doc["returns"][3] == {"pos": 3, "kind": "tensor", "from": "handoff", "base": "out"}
    assert doc["returns"][4] == whole["returns"][4] == {"pos": 4, "kind": "tensor", "from": [EMBED, "emb"]}


@pytest.mark.parametrize("dims", ["s72, s80, 64", "s72, 2*s72"])
def test_a_view_whose_row_holds_a_dim_that_is_no_number_has_no_width(first: Path, dims: str) -> None:
    assert C.parse_wrapper(str(first))["launches"][0]["widths"][3] == 2048
    _edit_call(first, EMBED_RUN, EMBED_RUN.replace("buf0", f"reinterpret_tensor(buf0, ({dims}), (2048, 64, 1), 0)"))
    assert C.parse_wrapper(str(first))["launches"][0]["widths"][3] is None


def test_a_returned_expression_that_is_no_name_is_an_integer(first: Path) -> None:
    whole = C.parse_wrapper(str(first))["returns"]
    _edit_call(first, "buf0, )", "buf0, 2*s72, )")
    assert C.parse_wrapper(str(first))["returns"] == [*whole, {"pos": 5, "kind": "int", "expr": "2*s72"}]


@pytest.mark.parametrize(("run", "why"), [
    ("triton_poi_fused_1.run(buf3, arg5_1, arg6_1, buf4, buf5)", "wrapper line not modelled: 'triton_poi_fused_1.run(buf3, arg5_1, arg6_1, buf4, buf5)'"),
    ("triton_poi_fused_1.run(*args, stream=raw_stream0)", "triton_poi_fused_1.run with 1 arguments, its source takes ['in_ptr0', "),
], ids=["no-stream", "starred"])
def test_a_launch_the_parser_does_not_read_is_counted_and_the_build_refuses_it(art: Path, first: Path, run: str, why: str) -> None:
    _edit_call(first, ROPE_RUN, run)
    doc = C.parse_wrapper(str(first))
    assert doc["unparsed_runs"] == 1 and [la["kernel"] for la in doc["launches"]] == [EMBED]
    with pytest.raises(Unmodelled, match=re.escape(f"{first.name}: {why}")):
        _build(art, inductor=first.parents[1])


GRAPH = '''class GraphModule(torch.nn.Module):
    def forward(self, s72: "Sym(s72)", L_input_ids_: "i32[s72]"):
        submod_0 = self.submod_0(l_input_ids_, s72, plain)
        getitem = submod_0[0]
        getitem_1 = submod_0[1]
        submod_1 = self.submod_1(getitem, getitem_1);  getitem = None
        getitem_2 = submod_1[0]
        getitem_3 = submod_9[4]
        submod_2 = self.submod_2(getitem_2, getitem_3, getitem_1)
        getitem_4 = submod_2[0]
        return (getitem_4,)

    class submod_0(torch.nn.Module):
        def forward(self, l_input_ids_: "i32[s72]", s72: "Sym(s72)", plain):
            return (embedding, view)

    class submod_1(torch.nn.Module):
        def forward(self, query: "bf16[s72, 32, 64]", output: "bf16[s72, 32, 64]"):
            unified_attention_with_output = torch.ops.vllm.unified_attention_with_output(query, output)
            return (output,)

    class submod_2(torch.nn.Module):
        def forward(self, x: "bf16[s72, 2048]", y, z: "bf16[s72, 2048]"):
            return (x,)
'''


def test_the_wiring_names_an_attention_ops_return_and_a_parameter_of_no_annotation(tmp_path: Path) -> None:
    """A piece argument taken from an attention op's return is named by the op's place among the attention ops, None for a
    submodule the graph never calls; a parameter a signature gives no annotation is of no kind."""
    (tmp_path / "graph.py").write_text(GRAPH)
    wiring = C.split_graph_wiring(str(tmp_path / "graph.py"))
    piece = {"kind": "piece_ret", "piece": 0}
    assert wiring["pieces"] == 2 and wiring["attention"] == [{"layer": 0, "submod": "submod_1", "q": {**piece, "ret": 0}, "out": {**piece, "ret": 1}}]
    assert wiring["wiring"] == [
        {"piece": 0, "submod": "submod_0", "args": [{"kind": "input_ids"}, {"kind": "sym", "name": "s72"}, {"kind": "unknown", "name": "plain"}],
         "raw_args": ["l_input_ids_", "s72", "plain"], "arg_kinds": ["tensor", "int", "?"], "n_ret": 2},
        {"piece": 1, "submod": "submod_2", "args": [{"kind": "attn_ret", "layer": 0, "ret": 0}, {"kind": "attn_ret", "layer": None, "ret": 4}, {**piece, "ret": 1}],
         "raw_args": ["getitem_2", "getitem_3", "getitem_1"], "arg_kinds": ["tensor", "?", "tensor"], "n_ret": 1}]


def test_a_piece_that_reads_an_attention_ops_return_is_refused(art: Path, tmp_path: Path) -> None:
    graph = tmp_path / "computation_graph.py"
    src = (art / "computation_graph.py").read_text()
    taken, call = "        getitem_3 = submod_0[3]\n", "submod_2 = self.submod_2(getitem_3, "
    assert src.count(taken) == 1 and src.count(call) == 1
    graph.write_text(src.replace(taken, taken + "        getitem_99 = submod_1[0]\n").replace(call, call.replace("getitem_3", "getitem_99")))
    with pytest.raises(Unmodelled, match=r"piece 1 argument 0 \(getitem_99\): a attn_ret argument"):
        _build(art, split_graph=str(graph))


def test_batched_copy_flushes_are_no_calls(art: Path, tmp_path: Path) -> None:
    ind = _copy(art, tmp_path)
    flush = f"run_intermediate_hooks({FLUSH!r}, None)"
    for p in sorted(ind.rglob("*.py")):
        src = re.sub(r"^( *)(run_intermediate_hooks\('verity\|mul\|buf4', buf4\))$", rf"\1\2\n\1{flush}", p.read_text(), flags=re.M)
        p.write_text(src.replace("        return (", f"        {flush}\n        return ("))
    assert sum(p.read_text().count(flush) for p in ind.rglob("*.py")) > 3
    assert _build(art, inductor=ind)["program_digest"] == _build(art)["program_digest"]


def _autotuned(art: Path, executed) -> dict:
    lc = json.loads((art / "launch_configs.json").read_text())
    key, rec = next((k, v) for k, v in lc.items() if v["candidates"][0]["kwargs"].get("R0_BLOCK"))
    other = {**rec["candidates"][0], "kwargs": {**rec["candidates"][0]["kwargs"], "R0_BLOCK": 1024}}
    return {**lc, key: {"candidates": [rec["candidates"][0], other], "executed": executed(rec["candidates"][0], other)}}


def test_an_autotuned_kernel_binds_the_config_it_ran(art: Path) -> None:
    assert _build(art, launch_configs=_autotuned(art, lambda won, _lost: [won]))["program_digest"] == _build(art)["program_digest"]
    lost = _build(art, launch_configs=_autotuned(art, lambda _won, lost: [lost]))
    assert lost["program_digest"] != _build(art)["program_digest"]


@pytest.mark.parametrize("executed", [lambda won, lost: [won, lost], lambda won, lost: []], ids=["two", "none"])
def test_a_kernel_without_one_executed_config_is_refused(art: Path, executed) -> None:
    with pytest.raises(Unmodelled, match="one config every instance ran is required"):
        _build(art, launch_configs=_autotuned(art, executed))


def test_add_norm_storing_its_rows_twice_binds_its_definition_per_buffer(art: Path) -> None:
    """The add + RMSNorm kernel that Inductor makes store its rows to two buffers (an extern op's two inputs, one of which the op
    mutates) is the same Definition once per buffer."""
    configs = json.loads((art / "launch_configs.json").read_text())
    srcs = [(n, s) for p in sorted((art / "inductor").rglob("*.py")) for n, s in T.kernel_sources(p.read_text()).items()]
    name, src = next((n, s) for n, s in srcs if T.recognize(n, s, T.launch_config(configs, s), cc=CC)["template"] == "ADD_NORM")
    cfg = T.launch_config(configs, src)
    stored = re.search(r"\n( +tl\.store\(out_ptr1 \+ [^\n]*\n)", src)
    assert stored is not None
    store = stored.group(1)
    twice = (src.replace("'out_ptr1': '*bf16'", "'out_ptr1': '*bf16', 'out_ptr2': '*bf16'", 1)
             .replace("out_ptr1, xnumel", "out_ptr1, out_ptr2, xnumel", 1).replace(store, store + store.replace("out_ptr1", "out_ptr2"), 1))
    one, two = T.recognize(name, src, cfg, cc=CC), T.recognize(name, twice, cfg, cc=CC)
    assert two["template"] == "ADD_NORM_COPY" and two["literals"] == one["literals"]
    assert [c["writes"][0]["param"] for c in two["calls"]] == ["out_ptr1", "out_ptr2"]
    assert all({**c, "writes": None} == {**one["calls"][0], "writes": None} for c in two["calls"])
    with pytest.raises(Unmodelled, match="none of the templates"):
        T.recognize(name, twice.replace(store.replace("out_ptr1", "out_ptr2"), store.replace("out_ptr1", "out_ptr2").replace("tmp22", "tmp20")), cfg, cc=CC)


def test_a_file_with_no_runner_call_has_no_launch(tmp_path: Path) -> None:
    (tmp_path / "w.py").write_text("x = 1\n")
    doc = C.parse_wrapper(str(tmp_path / "w.py"))
    assert (doc["launches"], doc["returns"], doc["unparsed_runs"], doc["signatures"]) == ([], [], 0, {}) and not doc["args"] and not doc["arg_kinds"]


def _path_readers(art: Path) -> dict[str, tuple[Path, Callable[[Any], Any]]]:
    """Each function of the Build that reads a file by its path, with a file of the fixture it reads."""
    graph = art / "computation_graph.py"
    wiring = C.split_graph_wiring(str(graph))
    wired = {"returns": [[{"value": f"c{w['piece']:02d}.r{i}"} for i in range(w["n_ret"])] for w in wiring["wiring"]]}
    return {"parse_wrapper": (sorted((art / "inductor").rglob("*.py"))[0], C.parse_wrapper),
            "split_graph_wiring": (graph, C.split_graph_wiring), "_graph_output": (graph, SV._graph_output),
            "_final_value": (graph, lambda path: SV._final_value(path, wiring, wired))}


@pytest.mark.parametrize("reader", ["parse_wrapper", "split_graph_wiring", "_graph_output", "_final_value"])
def test_a_file_is_opened_at_the_path_as_given(art: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reader: str) -> None:
    """A wrapper's or a split graph's path is opened as it is given, nothing normalised on the way: bytes name the same file, a
    file's name with a slash after it names no directory, and the empty path names nothing."""
    path, read = _path_readers(art)[reader]

    def less_path(doc: Any) -> Any:
        return {k: v for k, v in doc.items() if k != "path"} if isinstance(doc, dict) else doc

    assert less_path(read(os.fsencode(path))) == less_path(read(str(path)))
    with pytest.raises(NotADirectoryError):
        read(f"{path}/")
    monkeypatch.chdir(tmp_path)     # the suite's read guard takes an open of "" for a read of the directory the test runs in
    with pytest.raises(FileNotFoundError):
        read("")


def _wrapper(kernel: str) -> dict:
    """What `assign_wrappers` reads of a parsed wrapper: one argument of no kind `get_args()` names, one return, one launch."""
    return {"args": ["arg0_1"], "arg_kinds": ["?"], "returns": [{}], "launches": [{"kernel": kernel, "nargs": 1, "widths": [8]}]}


def test_a_piece_takes_the_first_wrapper_of_its_one_launch_program() -> None:
    graph = {"wiring": [{"piece": 3, "submod": "submod_3", "args": ["name"], "arg_kinds": ["layer"], "n_ret": 1}]}
    wrappers = [_wrapper("k"), {**_wrapper("other"), "returns": [{}, {}]}, _wrapper("k")]
    assert [w is wrappers[0] for w in C.assign_wrappers(wrappers, graph)] == [True]


@pytest.mark.parametrize(("kernels", "found"), [((), "0 wrapper candidates with 0 launch programs"), (("k", "other"), "2 wrapper candidates with 2 launch programs"),
                                                 (("k", "other", "k"), "3 wrapper candidates with 2 launch programs")], ids=["none", "two", "three"])
def test_a_piece_without_one_launch_program_is_refused(kernels: tuple[str, ...], found: str) -> None:
    graph = {"wiring": [{"piece": 3, "submod": "submod_3", "args": ["name"], "arg_kinds": ["layer"], "n_ret": 1}]}
    with pytest.raises(Unmodelled) as e:
        C.assign_wrappers([_wrapper(k) for k in kernels], graph)
    assert str(e.value) == f"compiled_manifest: piece 3 (submod_3, 1 args, 1 returns): {found} -- the wrapper set does not describe the split graph"


def test_a_kernel_that_is_a_template_only_in_its_function_is_refused(art: Path) -> None:
    """A source whose jit function is a template's text is still refused, by name, when its heuristic or its signature is not the
    template's, its literals break the template's relations, its launch config is absent or one the Definition's layout does not
    take, or the template reads another count of buffers than the Definition takes; a source with no jit function is no kernel, and
    one with no recorded config has none."""
    configs = json.loads((art / "launch_configs.json").read_text())
    srcs = [(n, s) for p in sorted((art / "inductor").rglob("*.py")) for n, s in T.kernel_sources(p.read_text()).items()]
    name, src = next((n, s) for n, s in srcs if T.recognize(n, s, T.launch_config(configs, s), cc=CC)["template"] == "ADD_NORM")
    cfg = T.launch_config(configs, src)
    t = next(t for t in T.TEMPLATES if t.name == "ADD_NORM")
    whole = T.recognize(name, src, cfg, cc=CC)
    lit = {h: whole["literals"][h] for h in T.holes(t)}
    assert lit == {"N": 2048, "NF": 2048.0, "EPS": 1e-05} and T.instantiate(name, t, lit, cfg) == {k: v for k, v in whole.items() if k != "source_sha256"}

    def why(of, *a, **kw) -> str:
        with pytest.raises(Unmodelled) as e:
            of(*a, **kw)
        return str(e.value)

    at = f"{name} (ADD_NORM): "
    assert why(T.recognize, name, src.replace("@triton_heuristics.reduction(", "@triton_heuristics.pointwise(", 1), cfg, cc=CC) == \
        at + "heuristic pointwise, the template's is reduction"
    assert why(T.recognize, name, src.replace("@triton_heuristics.", "@heuristics.", 1), cfg, cc=CC) == at + "heuristic None, the template's is reduction"
    assert why(T.recognize, name, src.replace("'out_ptr1': '*bf16'", "'out_ptr1': '*fp32'", 1), cfg, cc=CC).startswith(
        at + "signature {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', "
        "'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'} is not {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr1': '*bf16', ")
    assert why(T.recognize, name, src, None, cc=CC) == at + "its launch config is a static and none was given"
    assert why(T.recognize, name, src.replace("@triton.jit", "@jit"), cfg, cc=CC) == f"{name}: no @triton.jit def"
    assert why(T.instantiate, name, t, {**lit, "N": 2049}, cfg) == at + "literals {'N': 2049, 'NF': 2048.0, 'EPS': 1e-05} break the template's relations"
    assert why(T.instantiate, name, t, lit, {**cfg, "num_warps": 3}).startswith(at + "Inductor norm: N=2048 XBLOCK=2 R0_BLOCK=2048 NUM_WARPS=3 is not ")
    short = dataclasses.replace(t, calls=(dataclasses.replace(t.calls[0], reads=t.calls[0].reads[:2]),))
    assert re.fullmatch(re.escape(at) + r"\S+ takes 3 inputs and gives 1 outputs, the template reads 2 and writes 1", why(T.instantiate, name, short, lit, cfg))
    assert why(T.launch_config, configs, src + " ") == f"no launch config recorded for kernel source {T.source_sha256(src + ' ')[:16]}"
    assert why(T.numel, "x") == "type x is not an array of values"


@pytest.mark.parametrize("n,uniform,want", [(3, True, ("FULL", 4)), (8, True, ("FULL", 8)), (9, True, ("PIECEWISE", 16)),
                                            (11, False, ("PIECEWISE", 16)), (17, True, ("NONE", 17)), (1, False, ("PIECEWISE", 1))])
def test_dispatch(n: int, uniform: bool, want: tuple[str, int]) -> None:
    assert SV.dispatch(n, uniform, [1, 2, 4, 8, 16], 8) == want


def test_the_full_graph_caps_fa3_splits() -> None:
    kw = dict(NH=32, KVH=8, D=64, sms=132)
    assert SD.fa3_splits([1, 1, 1, 1], [107, 44, 13, 301], split_cap=32, **kw)["NS"] == [1, 1, 1, 2]
    assert SD.fa3_splits([1, 1, 1, 300], [106, 43, 12, 300], split_cap=0, **kw)["NS"] == [1, 1, 1, 1]


# ------------------------------------------------------------------------------------------------- the forms the Build refuses
#: a wrapper of one GEMM, the piece's rows against a weight into a new buffer, and the lines the cases below edit
PIECE = """
class Runner:
    def call(self, args):
        arg0_1, arg1_1, arg2_1 = args
        args.clear()
        s7 = arg0_1
        assert_size_stride(arg1_1, (s7, 32), (32, 1), 'input')
        assert_size_stride(arg2_1, (16, 32), (32, 1), 'input')
        buf0 = empty_strided_cuda((s7, 16), (16, 1), torch.bfloat16)
        extern_kernels.mm(arg1_1, reinterpret_tensor(arg2_1, (32, 16), (1, 32), 0), out=buf0)
        run_intermediate_hooks('verity|mm|buf0', buf0)
        return (buf0, )
"""
ROWS, WEIGHT, NEW = "(s7, 32), (32, 1), 'input'", "reinterpret_tensor(arg2_1, (32, 16), (1, 32), 0)", "(s7, 16), (16, 1), torch.bfloat16"
STORED = "(16, 32), (32, 1), 'input'"
HOOK = "        run_intermediate_hooks('verity|mm|buf0', buf0)\n"
#: SUPERSCRIPT TWO: a digit to `str.isdigit`, no decimal, and no number to `int`
SUP2 = "\u00b2"


def _piece(src: str) -> dict[str, Any]:
    return SV._Piece(src, None, "piece.py", CC).program()


def test_a_piece_returns_a_symbol_and_a_column_range_of_an_argument() -> None:
    assert _piece(PIECE)["returns"] == [{"value": 0, "width": 16}]
    view = "        half = reinterpret_tensor(arg1_1, (s7, 16), (32, 1), 16)\n"
    prog = _piece(PIECE.replace("        return (buf0, )", f"{view}        return (buf0, s7, half, )"))
    assert prog["returns"] == [{"value": 0, "width": 16}, {"sym": True}, {"arg": 1, "width": 16, "cols": [16, 32]}]


@pytest.mark.parametrize(("old", "new", "why"), [
    (PIECE, "x = 1\n", "no call()"),
    ("        return (buf0, )\n", "", "call() has no return"),
    (NEW, NEW.replace("(16, 1)", "(32, 1)"), "buf0 ['s7', '16'] strides ['32', '1']: not a row-major buffer of token rows or a constant shape"),
    (NEW, NEW.replace("(s7, 16)", "(s7, s7)"), "buf0 ['s7', 's7'] strides ['16', '1']: not a row-major buffer of token rows or a constant shape"),
    (ROWS, ROWS.replace("(s7, 32)", "(s7, 2*s7)"), "arg1_1 ['s7', '2*s7'] strides ['32', '1']: not a row-major buffer of token rows or a constant shape"),
    (STORED, STORED.replace("(16, 32)", "(s9, 32)"), "arg2_1 ['s9', '32'] strides ['32', '1']: not a row-major buffer of token rows or a constant shape"),
    (NEW, NEW.replace("(s7, 16)", f"(s7, {SUP2})"), f"buf0 ['s7', '{SUP2}'] strides ['16', '1']: not a row-major buffer of token rows or a constant shape"),
    (WEIGHT, WEIGHT.replace("arg2_1", "arg0_1"), "a view of arg0_1, itself a view or of no known shape"),
    ("mm(arg1_1,", "mm(reinterpret_tensor(arg1_1, (s7, 64), (32, 1), 0),", "view ['s7', '64'] ['32', '1'] +0 of arg1_1 ['s7', '32']: not a column range of its rows"),
    ("mm(arg1_1,", f"mm(reinterpret_tensor(arg1_1, (s7, {SUP2}), (32, 1), 0),",
     f"view ['s7', '{SUP2}'] ['32', '1'] +0 of arg1_1 ['s7', '32']: not a column range of its rows"),
    (WEIGHT, WEIGHT.replace("(32, 16), (1, 32)", "(16, 32), (32, 1)"), "view ['16', '32'] ['32', '1'] +0 of arg2_1 ['16', '32']: only a weight's transpose is modelled"),
    (NEW, NEW.replace("s7", "4"), "mm writes buf0, not a buffer of token rows"),
    ("out=buf0)", "out=arg1_1)", "mm writes its argument arg1_1 in place"),
    ("        s7 = arg0_1\n", "        s7 = arg0_1\n        s8 = arg1_1\n", "symbol 's8 = arg1_1': one token count is modelled"),
    (NEW, NEW.replace("bfloat16", "float32"), "'buf0 = empty_strided_cuda((s7, 16), (16, 1), torch.float32)': a float32 buffer"),
    (HOOK, f"{HOOK}        buf1 = buf9; del buf9  # reuse\n", "'buf1 = buf9; del buf9  # reuse': reuse of an unknown buffer"),
    (HOOK, f"{HOOK}        buf1 = reinterpret_tensor(buf0, (s7, 32), (32, 1), 0); del buf0  # reuse\n",
     "'buf1 = reinterpret_tensor(buf0, (s7, 32), (32, 1), 0); del buf0  # reuse': reuse beyond the dead buffer's rows"),
    (HOOK, f"{HOOK}        buf1 = reinterpret_tensor(buf0, (s7, s7), (16, 1), 0); del buf0  # reuse\n",
     "'buf1 = reinterpret_tensor(buf0, (s7, s7), (16, 1), 0); del buf0  # reuse': reuse beyond the dead buffer's rows"),
    (HOOK, f"{HOOK}        buf1 = reinterpret_tensor(buf0, (s7, {SUP2}), (16, 1), 0); del buf0  # reuse\n",
     f"'buf1 = reinterpret_tensor(buf0, (s7, {SUP2}), (16, 1), 0); del buf0  # reuse': reuse beyond the dead buffer's rows"),
    (HOOK, HOOK + HOOK, f"{HOOK.strip()!r}: hooks a buffer no launch of this wrapper wrote, or twice"),
    (f"arg1_1, {WEIGHT}", "arg1_1", "extern_kernels.mm with 1 operands"),
    (WEIGHT, "arg2_1", "mm(arg1_1, arg2_1): an activation's whole rows against a transposed weight is modelled"),
    (ROWS, ROWS.replace("32", "8"), "mm: activation width 8 against a weight of 32 columns"),
    (NEW, NEW.replace("(s7, 16), (16, 1)", "(s7, 8), (8, 1)"), "mm writes buf0 of width 8, the weight has 16 rows"),
    (HOOK, f"{HOOK}        triton_poi_fused_0.run(buf0, s7, stream=raw_stream0)\n", "triton_poi_fused_0.run: no async_compile.triton block defines it"),
], ids=["no-call", "no-return", "strides", "symbol-in-a-buffers-row", "symbol-in-an-arguments-row", "symbol-in-a-constant-shape",
        "superscript-in-a-buffers-row", "view-of-no-shape", "view-past-the-rows", "view-of-a-superscripts-columns", "weight-not-transposed",
        "written-without-tokens", "written-in-place", "second-token-count", "float32", "reuse-of-no-buffer", "reuse-past-the-rows",
        "reuse-of-a-symbols-rows", "reuse-of-a-superscripts-rows", "hooked-twice", "one-operand", "weight-as-stored", "columns", "rows", "no-kernel-source"])
def test_a_piece_of_another_form_is_refused(old: str, new: str, why: str) -> None:
    assert old in PIECE
    with pytest.raises(Unmodelled, match=re.escape(f"piece.py: {why}")):
        _piece(PIECE.replace(old, new))


@pytest.mark.parametrize(("old", "new", "why"), [
    (EMBED_RUN, EMBED_RUN.replace("s72, 2048,", "s72, 1024,"), f"{EMBED}: r0_numel = 1024, the template's is 2048"),
    (EMBED_RUN, EMBED_RUN.replace("s72, 2048,", "2*s72, 2048,"), f"{EMBED}: xnumel = 2*s72, the template's is 1 x tokens"),
    (ROPE_RUN, ROPE_RUN.replace("buf3,", "buf0,"), "triton_poi_fused_1: columns [2048, 2560] of buf0"),
    (ROPE_RUN, ROPE_RUN.replace("arg5_1,", "buf0,"), "triton_poi_fused_1: in_ptr2 indexed by in_ptr1: a 2-d table by one index a token"),
    (EMBED_RUN, EMBED_RUN.replace("arg3_1,", "arg0_1,"), "reads 2048 elements per row from in_ptr2, which has 1"),
    (EMBED_RUN, EMBED_RUN.replace("buf0,", "reinterpret_tensor(buf0, (s72, 2048), (2048, 1), 0),"),
     f"{EMBED} writes reinterpret_tensor(buf0, (s72, 2048), (2048, 1), 0): not a whole buffer"),
    (ROPE_RUN, ROPE_RUN.replace("buf4, buf5", "buf5, buf4"), "writes 512 elements per row to buf5 of width 2048"),
], ids=["constant", "token-count", "columns", "index", "read-width", "written-view", "written-width"])
def test_a_launch_whose_operands_are_not_its_templates_is_refused(art: Path, first: Path, old: str, new: str, why: str) -> None:
    _edit_call(first, old, new)
    with pytest.raises(Unmodelled, match=re.escape(why)):
        SV.parse_piece(str(first), json.loads((art / "launch_configs.json").read_text()), CC)


@pytest.fixture(scope="module")
def pieces(art: Path) -> dict[str, Any]:
    """The fixture's piece programs and its split graph's wiring, as the Build hands them to `wire`."""
    graph = C.split_graph_wiring(str(art / "computation_graph.py"))
    configs = json.loads((art / "launch_configs.json").read_text())
    wrappers = [{**C.parse_wrapper(str(p)), "path": str(p)} for p in sorted((art / "inductor").rglob("*.py"))]
    return {"programs": [SV.parse_piece(w["path"], configs, CC) for w in C.assign_wrappers(wrappers, graph)], "graph": graph}


#: the first piece's embedding rows, its fifth return
EMB = {"value": 0, "width": 2048}


@pytest.mark.parametrize(("at", "value", "why"), [
    (("programs", 0, "values", 1, "name"), "embedding/buf0", "c00.embedding/buf0: two values of one name"),
    (("programs", 0, "returns", 0), {"empty": "buf9", "width": 512}, "attention 0: its k is no value"),
    (("programs", 0, "returns", 3), {"sym": True}, "attention 0: its output {'sym': True} is not a whole buffer of token rows"),
    (("programs", 0, "returns", 0), EMB, "attention 0: its k is not a RoPE's output"),
    (("programs", 0, "returns", 1), EMB, "attention 0: q 32x64, k 8x64, v width 2048, output width 2048"),
    (("programs", 0, "calls", 4, "reads", 0), {"arg": 0, "width": 1}, "c00.cat/buf5: no GEMM projects it"),
    (("graph", "wiring", 1, "args", 4), {"kind": "input_ids"}, "an argument of width 2048 wired to {'value': 'c00.input/ids', 'width': 1}"),
], ids=["one-name-twice", "k-no-value", "output-a-symbol", "k-no-rope", "v-width", "q-no-gemm", "argument-width"])
def test_pieces_wired_another_way_are_refused(pieces: dict[str, Any], at: tuple, value: Any, why: str) -> None:
    doc = copy.deepcopy(pieces)
    assert doc["programs"][0]["returns"][4] == EMB and SV.wire(doc["programs"], doc["graph"], backbone="model")["attention"][0]["NH"] == 32
    *path, last = at
    inner: Any = doc
    for k in path:
        inner = inner[k]
    inner[last] = value
    with pytest.raises(Unmodelled, match=re.escape(why)):
        SV.wire(doc["programs"], doc["graph"], backbone="model")


def test_a_pieces_column_range_is_taken_of_its_arguments_rows() -> None:
    rows = {"value": "c00.mm/buf0", "width": 16}
    assert SV._merge(rows, {"arg": 1, "width": 8, "cols": [4, 12]}) == {**rows, "width": 8, "cols": [4, 12]}
    assert SV._merge({**rows, "cols": [8, 24]}, {"arg": 1, "width": 8, "cols": [4, 12]}) == {**rows, "width": 8, "cols": [12, 20]}
    for env in (rows, {**rows, "cols": [8, 24]}, {"weight": "model.norm.weight"}):
        with pytest.raises(Unmodelled, match=re.escape(f"columns [8, 24] of an argument {env}")):
            SV._merge(env, {"arg": 1, "width": 16, "cols": [8, 24]})


def test_steps_past_the_schedule_have_no_plan(art: Path) -> None:
    m, longer = _build(art), _build(art, steps=16)
    assert (longer["steps"], longer["workload"]["steps"]) == (list(range(16)), 16) and len(longer["steps_plan"]) == 14
    assert all(longer[k] == m[k] for k in ("steps_plan", "program_digest", "identities", "manifest_digest"))


def test_a_route_of_another_kernel_is_refused(art: Path) -> None:
    routes = json.loads((art / "routes.json").read_text())
    routes["routes"]["1,2048,2048"] = {"kernel": "other"}
    with pytest.raises(Unmodelled, match=re.escape("cuBLAS route {'kernel': 'other'}")):
        _build(art, routes=routes)


def test_a_directory_with_no_wrapper_is_refused(art: Path, tmp_path: Path) -> None:
    (tmp_path / "aa").mkdir()
    (tmp_path / "aa" / "module.py").write_text("x = 1\n")
    with pytest.raises(Unmodelled, match=re.escape(f"no Inductor wrapper with a call() under {tmp_path}")):
        _build(art, inductor=tmp_path)


def test_two_wrappers_of_one_launch_signature_and_two_programs_are_refused(art: Path, tmp_path: Path) -> None:
    ind = _copy(art, tmp_path)
    hook = "'verity|mul|buf4', buf4"
    piece = next(p for p in sorted(ind.rglob("*.py")) if p.read_text().count(hook) == 1 and "return (buf7, )" not in p.read_text())
    (ind / "zz").mkdir()
    (ind / "zz" / "twin.py").write_text(piece.read_text().replace(hook, "'verity|silu|buf4', buf4"))
    with pytest.raises(Unmodelled, match=re.escape(f"twin.py and {piece.name}: one launch signature, two programs")):
        _build(art, inductor=ind)


@pytest.mark.parametrize(("twin", "found"), [(False, "0 wrapper candidates with 0 launch programs"), (True, "2 wrapper candidates with 2 launch programs")],
                         ids=["none", "two"])
def test_a_wrapper_set_that_does_not_describe_the_graph_is_refused(art: Path, tmp_path: Path, twin: bool, found: str) -> None:
    """The Build of an artifact whose last piece has no wrapper, or two wrappers that launch two programs: the second is the first
    less its last launch."""
    ind = _copy(art, tmp_path)
    last = next(p for p in sorted(ind.rglob("*.py")) if "return (buf7, )" in p.read_text())
    if twin:
        lines = last.read_text().splitlines(keepends=True)
        launch = max(i for i, line in enumerate(lines) if C._RUN.match(line))
        (ind / "zz").mkdir()
        (ind / "zz" / "twin.py").write_text("".join(lines[:launch] + lines[launch + 1:]))
    else:
        last.unlink()
    with pytest.raises(Unmodelled) as e:
        _build(art, inductor=ind)
    assert str(e.value) == f"compiled_manifest: piece 16 (submod_32, 8 args, 1 returns): {found} -- the wrapper set does not describe the split graph"


@pytest.mark.parametrize(("old", "new", "why"), [
    ("return (getitem_80,)", "return (getitem_80, getitem_79)", "the split graph returns no single value"),
    ("return (getitem_80,)", "return (s72,)", "the split graph's output s72 is not a piece's return"),
    ("        return (getitem_80,)", "        getitem_81 = submod_31[0]\n        return (getitem_81,)", "the split graph's output getitem_81 is not a piece's return"),
], ids=["two-values", "a-symbol", "an-attention-return"])
def test_a_graph_whose_output_is_not_one_pieces_return_is_refused(art: Path, tmp_path: Path, old: str, new: str, why: str) -> None:
    graph = tmp_path / "computation_graph.py"
    src = (art / "computation_graph.py").read_text()
    assert src.count(old) == 1
    graph.write_text(src.replace(old, new))
    with pytest.raises(Unmodelled, match=re.escape(why)):
        _build(art, split_graph=str(graph))


def test_a_backbone_output_that_is_a_column_range_is_refused(art: Path, tmp_path: Path) -> None:
    ind = _copy(art, tmp_path)
    _edit(ind, "return (buf7, )", "return (reinterpret_tensor(buf7, (s72, 1024), (2048, 1), 0), )")
    with pytest.raises(Unmodelled, match=re.escape("the backbone's output {'value': 'c16.mul/buf7', 'width': 1024, 'cols': [0, 1024]} is not a whole value")):
        _build(art, inductor=ind)


@pytest.mark.parametrize(("wired", "why"), [
    ({"attention": []}, "the split graph has no attention op"),
    ({"attention": [{"NH": 32, "KVH": 8, "D": 64}, {"NH": 32, "KVH": 16, "D": 64}]}, "attention ops of different head geometry"),
    ({"attention": [{"NH": 32, "KVH": 8, "D": 64}], "calls": [{"op": "mm"}]}, "no embedding call: the vocabulary is unknown"),
], ids=["no-attention", "two-geometries", "no-embedding"])
def test_a_wired_program_the_steps_cannot_be_planned_for_is_refused(wired: dict[str, Any], why: str) -> None:
    """`assemble` refuses each of the wired Program alone, before it reads another argument: every other one is None."""
    rest: dict[str, Any] = dict.fromkeys(inspect.signature(SV.assemble).parameters)
    with pytest.raises(Unmodelled, match=re.escape(why)):
        SV.assemble(**{**rest, "wired": wired})
