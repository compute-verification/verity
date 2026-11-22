"""Every place the serve tax, prod_repro and the replay ladder write a file is in a registry of writers: a write under
the private directory (its path built from ``private``, ``kept`` or ``_kept``) needs no entry, and any other is
listed here by its file, function and path as public (in the run's record: counts, timings, roots and settings) or
apart (a scratch, site or edge directory, never the run's record). A new writer, whatever its file's name, fails until
it is listed."""
import ast
import fnmatch
import re
from pathlib import Path, PurePosixPath

import pytest
from verity_vllm.pipeline.research_outputs import PRIVATE

ROOT = Path(__file__).resolve().parents[3]
DIRS = ("benchmarks/serve_tax", "benchmarks/prod_repro", "benchmarks/replay_ladder")
PRIVATE_ROOT = re.compile(r"\b(_kept|private|kept)\b")
#: functions that write the path they are given, by name: the argument that is it
PASSES = {"_lines": 0, "_lines_behind": 0, "plan_of": 0, "write_trace": 0, "C.drive": 5, "SC.write": 0, "write_host": 1}
#: inside those, the parameter they write, which their callers' entries stand for
PASSED = {("_lines", "path"), ("_lines_behind", "path"), ("plan_of", "f"), ("write_trace", "f"), ("write_host", "path")}
BARE = {k for k in PASSES if "." not in k}

WRITES = {
    "benchmarks/prod_repro/prod_repro.py": {
        ("keep_topk_files", "out / TOPK_FILES / name / f.name"): "public",
        ("topk_kept", "f"): "public",
        ("topk_by_dir", "f"): "public",
        ("declare", "tmp"): "public",
        ("carry", "dst / name"): "public",
        ("main", "a.out / 'prod_repro.json'"): "public",
        ("main", "a.out / 'prod_repro.txt'"): "public",
    },
    "benchmarks/replay_ladder/inductor_pin.py": {
        ("autotune_to_one_config", "Path(d) / f'tilings.{os.getpid()}.jsonl'"): "public",
        ("site", "info / 'METADATA'"): "apart",
        ("site", "info / 'entry_points.txt'"): "apart",
        ("site", "d / 'inductor_pin.py'"): "apart",
        ("site", "d / CONFIG"): "apart",
    },
    "benchmarks/replay_ladder/ladder.py": {
        ("route_capture_site", "info / 'METADATA'"): "apart",
        ("route_capture_site", "info / 'entry_points.txt'"): "apart",
        ("route_capture_site", "d / f"): "apart",
        ("serve", "plan"): "private",
        ("serve", "out / 'serve.json'"): "public",
        ("alone", "out / 'alone' / 'alone.json'"): "public",
        ("summary", "out / 'summary.json'"): "public",
        ("summary", "out / 'summary.md'"): "public",
    },
    "benchmarks/replay_ladder/prod_plan.py": {
        ("build", "out_f"): "private",
        ("build", "out_f.with_suffix('.meta.json')"): "private",
        ("kinds", "d / 'kinds.json'"): "public",
    },
    "benchmarks/replay_ladder/status_table.py": {
        ("render", "out / 'status.json'"): "public",
        ("render", "out / 'status.md'"): "public",
    },
    "benchmarks/replay_ladder/topk_probe.py": {
        ("flush", "out / f'dev{dev}.jsonl'"): "private",
    },
    "benchmarks/serve_tax/boundary.py": {
        ("save_rank", "Path(edge_dir) / f'rank-{worker.rank}.pkl'"): "apart",
        ("save_edge", "edge_dir / 'state.pkl'"): "apart",
    },
    "benchmarks/serve_tax/cell_shim.py": {
        ("serve", "a.log"): "public",
    },
    "benchmarks/serve_tax/prefix_plan.py": {
        ("main", "Path(argv[2])"): "public",
    },
    "benchmarks/serve_tax/rebuild.py": {
        ("save", "path.with_suffix('.bin')"): "apart",
        ("save", "path.with_suffix('.json')"): "apart",
        ("run", "a.mirror_out / 'start.json'"): "apart",
        ("main", "a.out / name"): "public",
    },
    "benchmarks/serve_tax/replay.py": {
        ("main", "a.out / 'replay.json'"): "public",
    },
    "benchmarks/serve_tax/restore.py": {
        ("main", "a.out / 'restore.json'"): "public",
    },
    "benchmarks/serve_tax/serve_tax.py": {
        ("plugin_site", "d / 'METADATA'"): "apart",
        ("plugin_site", "d / 'entry_points.txt'"): "apart",
        ("kept_client", "d / CLIENT_FILE"): "public",
        ("steps_of_trace", "f"): "public",
        ("stated", "f"): "public",
        ("drive_load", "d / 'meta.json'"): "public",
        ("user_files", "f"): "apart",
        ("fetch", "tmp"): "apart",
        ("client", "jobs[k][0] / 'bench.log'"): "public",
        ("drive_bench", "u / RESULT"): "public",
        ("drive_bench", "d / RESULT"): "public",
        ("drive_bench", "d / 'meta.json'"): "public",
        ("serve", "out / f'{n}.trace.jsonl'"): "public",
        ("serve", "out / 'meta.json'"): "public",
        ("serve", "srv / 'server.log'"): "public",
        ("serve", "d / 'meta.json'"): "public",
        ("certified_server", "out / SERVER_RECORD"): "public",
        ("certified_server", "srv / 'server.log'"): "public",
        ("certified_server", "srv / 'front.log'"): "public",
        ("certified_client", "out / 'meta.json'"): "public",
        ("certified_client", "out / 'attester.log'"): "public",
        ("certified_client", "out / 'shim.log'"): "public",
        ("certified_client", "out / 'warm' / 'meta.json'"): "public",
        ("certified_client", "d / 'meta.json'"): "public",
        ("certified_client", "cert / 'client.json'"): "private",
        ("certified_client", "out / 'check.log'"): "public",
        ("main", "Path(a.out)"): "private",
        ("main", "a.out / 'serving_cost.json'"): "public",
        ("main", "a.out / 'serving_cost.txt'"): "public",
        ("main", 'a.out / f"serving_cost.{r[\'load\']}.json"'): "public",
        ("main", 'a.out / f"serving_cost.{r[\'load\']}.txt"'): "public",
        ("main", "a.out / 'table.json'"): "public",
        ("main", "a.out / 'table.md'"): "public",
    },
    "benchmarks/serve_tax/servetax_plugin.py": {
        ("local_take", "out / f'edges.{pid}.jsonl'"): "public",
        ("write", "kv_dir / f'edge-{k}.state.pkl'"): "private",
        ("write", "d / f'edge-{k}.dump.json'"): "private",
        ("write", "d / f'edge-{k}.fp.npy'"): "private",
        ("write", "str(path)"): "private",
        ("ranks_take", "Path(cfg['out']) / f'edges.{os.getpid()}.jsonl'"): "public",
        ("install", "out / f'prefix_resets.{pid}.jsonl'"): "public",
        ("install", "out / f'tokens.{pid}.jsonl'"): "public",
        ("install", "out / f'boundary.{pid}.jsonl'"): "public",
    },
    "benchmarks/serve_tax/tp_edge.py": {
        ("snapshot", "self.out / 'outputs.jsonl'"): "apart",
        ("snapshot", "d / 'edge.json'"): "apart",
    },
    "benchmarks/serve_tax/tp_restore.py": {
        ("edge_site", "d / 'METADATA'"): "apart",
        ("edge_site", "d / 'entry_points.txt'"): "apart",
        ("drive", "d / 'classes.json'"): "public",
        ("drive", "d / 'meta.json'"): "public",
        ("serve", "base"): "private",
        ("serve", "plan_file"): "private",
        ("serve", "srv / 'server.log'"): "public",
        ("serve", "out / 'meta.json'"): "public",
        ("carry_autotune", "dst"): "apart",
        ("restore", "verdict"): "public",
    },
    "benchmarks/serve_tax/tp_state.py": {
        ("digest", 'Path(out) / f"digests.rank{rank[\'rank\']}.bin"'): "apart",
        ("dump", "d / f'{i:03d}.bin'"): "apart",
        ("dump", "d / 'layers.json'"): "apart",
    },
}


def written(call: ast.Call) -> ast.expr | None:
    """The path ``call`` writes, or None where it writes none."""
    f = call.func
    name = ast.unparse(f)
    last = name.rsplit(".", 1)[-1]

    def mode(i: int) -> str:
        if len(call.args) > i and isinstance(call.args[i], ast.Constant):
            return str(call.args[i].value)
        return next((str(k.value.value) for k in call.keywords if k.arg == "mode" and isinstance(k.value, ast.Constant)),
                    "r")

    if isinstance(f, ast.Attribute) and last in ("write_text", "write_bytes"):
        return f.value
    if name in ("open", "gzip.open", "io.open") and set(mode(1)) & set("wax"):
        return call.args[0]
    if isinstance(f, ast.Attribute) and last == "open" and name not in ("gzip.open", "io.open", "os.open") \
            and set(mode(0)) & set("wax"):
        return f.value
    if name in ("np.save", "np.savez", "np.savez_compressed", "numpy.save"):
        return call.args[0]
    if name in ("shutil.copy", "shutil.copy2", "shutil.copyfile", "shutil.copytree", "shutil.move"):
        return call.args[1]
    i = PASSES.get(name, PASSES.get(last) if last in BARE else None)
    return call.args[i] if i is not None and len(call.args) > i else None


def sites(path: Path) -> list[tuple[int, str, str]]:
    """Each write in ``path``: its line, the function it is in and the path it writes, as source."""
    out = []

    def visit(node: ast.AST, fn: str) -> None:
        for ch in ast.iter_child_nodes(node):
            here = ch.name if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)) else fn
            if isinstance(ch, ast.Call) and (t := written(ch)) is not None:
                target = ast.unparse(t)
                if (fn, target) not in PASSED:
                    out.append((ch.lineno, fn, target))
            visit(ch, here)

    visit(ast.parse(path.read_text()), "<module>")
    return out


FILES = sorted(str(p.relative_to(ROOT)) for d in DIRS for p in (ROOT / d).rglob("*.py"))


@pytest.mark.parametrize("rel", FILES)
def test_every_write_is_under_the_private_directory_or_listed(rel: str) -> None:
    listed = WRITES.get(rel, {})
    found = sites(ROOT / rel)
    unlisted = [f"{rel}:{ln} in {fn}: {t}" for ln, fn, t in found if not PRIVATE_ROOT.search(t) and (fn, t) not in listed]
    assert not unlisted, ("these write outside the private directory and the registry lists none of them: write what "
                          "holds request values under the private directory (`_kept`), or add the write to WRITES as "
                          "public or apart:\n  " + "\n  ".join(unlisted))
    stale = sorted(set(listed) - {(fn, t) for _, fn, t in found})
    assert not stale, f"WRITES lists writes {rel} no longer makes: remove {stale}"


def test_no_public_write_has_a_private_name() -> None:
    named = [f"{rel} in {fn}: {t}" for rel, w in WRITES.items() for (fn, t), kind in w.items() if kind == "public"
             for c in ast.walk(ast.parse(t, mode="eval")) if isinstance(c, ast.Constant) and isinstance(c.value, str)
             and any(fnmatch.fnmatchcase(PurePosixPath(c.value).name, p) for p in PRIVATE)]
    assert not named, ("these public writes have a name the adapter's private list (private_names.json) calls "
                       "private: write them under the private directory:\n  " + "\n  ".join(named))


def test_the_registry_names_only_files_it_scans_and_kinds_it_knows() -> None:
    assert set(WRITES) <= set(FILES), sorted(set(WRITES) - set(FILES))
    assert {k for w in WRITES.values() for k in w.values()} <= {"public", "private", "apart"}


def test_a_new_writer_is_found_whatever_its_name(tmp_path: Path) -> None:
    f = tmp_path / "w.py"
    f.write_text("import json\n"
                 "def go(out, private, rows):\n"
                 "    (out / 'outputs.json').write_text(json.dumps(rows))\n"
                 "    with open(out / 'x.bin', 'wb') as g:\n"
                 "        g.write(b'')\n"
                 "    _lines(private / 'ids.jsonl')\n"
                 "    open(out / 'read.json').read()\n")
    assert sites(f) == [(3, "go", "out / 'outputs.json'"), (4, "go", "out / 'x.bin'"), (6, "go", "private / 'ids.jsonl'")]


def test_plugin_refuses_without_private(tmp_path: Path) -> None:
    import servetax_plugin as P

    with pytest.raises(RuntimeError, match="private"):
        P._kept({"out": str(tmp_path)}, "step_trace.1.jsonl")
    assert P._kept({"out": str(tmp_path / "out"), "private": str(tmp_path / "p")}, "ids.1.jsonl") \
        == tmp_path / "p" / "server" / "ids.1.jsonl"
