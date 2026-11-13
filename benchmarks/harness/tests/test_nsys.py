"""`verity_bench.nsys`'s GPU kernel time by kernel, on Nsight Systems exports built here: a capture range's kernels (a CUDA
graph's nodes, a driver-API launch from a worker thread, two streams that overlap, a second device, another process), the
whole capture, a graph traced whole, an arm's extra a step over stock, and both as Markdown and as JSON."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from verity_bench import nsys

MS = 1_000_000
#: nsys's globalTid: the process above the thread's 24 bits. One server's engine thread and its worker, and another process
ENGINE, WORKER, OTHER = (7 << 24) | 1, (7 << 24) | 2, (8 << 24) | 1
GEMM = "void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_bf16_128x128_32x3_nn_align8>(Params)"
NORM = "triton_red_fused_rms_norm_0"
LEAF = "void rows_leaf_kernel<64>(unsigned char const*, unsigned long*, int)"
SAMPLE = "void argmax_kernel<float>(float const*, long*)"
SCHEMA = """
CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, globalTid INTEGER, text TEXT, textId INTEGER);
CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER, end INTEGER, globalTid INTEGER, correlationId INTEGER);
CREATE TABLE CUPTI_ACTIVITY_KIND_DRIVER (start INTEGER, end INTEGER, globalTid INTEGER, correlationId INTEGER);
CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, deviceId INTEGER, streamId INTEGER, globalPid INTEGER,
    correlationId INTEGER, demangledName INTEGER, shortName INTEGER);
"""
#: a graph traced whole carries no process of its own here, so its launch is found by correlation id alone
GRAPH_SCHEMA = """
CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE (start INTEGER, end INTEGER, deviceId INTEGER, contextId INTEGER,
    streamId INTEGER, correlationId INTEGER, graphId INTEGER);
"""


def export(path: Path, ranges: list[tuple], launches: list[tuple], graphs: tuple[tuple, ...] = ()) -> Path:
    """An export, times in ms. `ranges`: (start, end, thread, text), the first one's text held in StringIds. `launches`:
    (the call's table, its start, thread, correlation id, its kernels as (name, start, end, stream[, device])). `graphs`:
    (correlation id, start, end) of graphs traced whole."""

    db = sqlite3.connect(path)
    db.executescript(SCHEMA + (GRAPH_SCHEMA if graphs else ""))
    names = sorted({k[0] for _, _, _, _, kernels in launches for k in kernels})
    db.executemany("INSERT INTO StringIds VALUES (?, ?)", [(100 + i, name) for i, name in enumerate(names)])
    for i, (start, end, thread, text) in enumerate(ranges):
        if i == 0:
            db.execute("INSERT INTO StringIds VALUES (1, ?)", (text,))
        db.execute("INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, ?)",
                   (start * MS, end * MS, thread, None if i == 0 else text, 1 if i == 0 else None))
    for table, at, thread, corr, kernels in launches:
        db.execute(f"INSERT INTO CUPTI_ACTIVITY_KIND_{table} VALUES (?, ?, ?, ?)", (at * MS, at * MS + 1000, thread, corr))
        for name, start, end, stream, *device in kernels:
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                       (start * MS, end * MS, *(device or [0]), stream, thread >> 24 << 24, corr,
                        100 + names.index(name), 0))
    for corr, start, end in graphs:
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES (?, ?, 0, 1, 7, ?, 5)", (start * MS, end * MS, corr))
    db.commit()
    db.close()
    return path


def arm_export(path: Path) -> Path:
    """A warm-up, then the run `run:arm` of two steps on the engine thread. Each step is a graph launch of a GEMM and a
    norm on stream 7. A worker thread hashes on stream 9: a driver-API launch whose kernel runs beside the first step's
    GEMM, and a launch at the run's end whose kernel runs after it. The second step also launches a GEMM on a second
    device, and its graph launch has a driver record beside its runtime record. Another process reuses a correlation id
    later."""

    return export(path, [(10, 50, ENGINE, "run:arm"), (0, 6, ENGINE, "warm"), (2, 4, ENGINE, "exec:8"),
                         (10, 20, ENGINE, "exec:8"), (20, 30, ENGINE, "exec:8")],
                  [("RUNTIME", 1, ENGINE, 1, [(GEMM, 2, 5, 7)]),
                   ("RUNTIME", 11, ENGINE, 2, [(GEMM, 12, 15, 7), (NORM, 15, 16, 7)]),
                   ("DRIVER", 12, WORKER, 3, [(LEAF, 13, 17, 9)]),
                   ("RUNTIME", 21, ENGINE, 4, [(GEMM, 22, 25, 7), (NORM, 25, 26, 7)]),
                   ("DRIVER", 21, ENGINE, 4, []),
                   ("RUNTIME", 22, ENGINE, 6, [(GEMM, 22, 24, 7, 1)]),
                   ("RUNTIME", 49, WORKER, 5, [(LEAF, 51, 52, 9)]),
                   ("RUNTIME", 100, OTHER, 2, [(SAMPLE, 101, 102, 7)])])


def stock_export(path: Path) -> Path:
    """The run `run:stock` of two steps: shorter GEMMs with nothing beside them, the norm, and a sampler the arm lacks."""

    return export(path, [(10, 50, ENGINE, "run:stock"), (10, 20, ENGINE, "exec:8"), (20, 30, ENGINE, "exec:8")],
                  [("RUNTIME", 11, ENGINE, 1, [(GEMM, 12, 14, 7), (NORM, 14, 15, 7)]),
                   ("RUNTIME", 21, ENGINE, 2, [(GEMM, 22, 24, 7), (NORM, 24, 25, 7)]),
                   ("RUNTIME", 26, ENGINE, 3, [(SAMPLE, 26, 27, 7)])])


def rows(table: dict, *keys: str) -> dict[str, tuple]:
    return {k["name"]: tuple(k[key] for key in keys) for k in table["kernels"]}


def test_a_ranges_kernels_are_those_its_open_time_launched_on_any_thread(tmp_path: Path) -> None:
    """The run's table holds the graphs' nodes, the worker's driver-API launch and the kernel that ran after the run closed,
    each once though one launch has two call records; not the warm-up's GEMM, nor the other process's kernel under a
    correlation id the run used. Its steps are the two step ranges that open inside it."""

    table = nsys.kernel_table(arm_export(tmp_path / "arm.sqlite"), "run:arm", "exec:")
    assert rows(table, "launches", "ms", "ms_step") == {GEMM: (3, 8.0, 4.0), LEAF: (2, 5.0, 2.5), NORM: (2, 2.0, 1.0)}
    assert [k["name"] for k in table["kernels"]] == [GEMM, LEAF, NORM]
    assert {k["name"]: k["share"] for k in table["kernels"]} == pytest.approx({GEMM: 8 / 15, LEAF: 5 / 15, NORM: 2 / 15})
    assert {k: table[k] for k in ("range", "ranges", "steps", "launches", "kernel_ms")} == {
        "range": "run:arm", "ranges": 1, "steps": 2, "launches": 7, "kernel_ms": 15.0}


def test_kernel_time_counts_overlapping_streams_whole_and_the_union_is_beside_it(tmp_path: Path) -> None:
    """The hashing on stream 9 overlaps the first step's GEMM on stream 7 for 2 ms and its norm for 1 ms, so the first
    device's 13 ms of kernels cover 10 ms; the second device's GEMM runs beside them and is united apart, 2 ms more."""

    table = nsys.kernel_table(arm_export(tmp_path / "arm.sqlite"), "run:arm")
    assert (table["kernel_ms"], table["union_ms"], table["streams"]) == (15.0, 12.0, 3)
    assert table["notes"] == [
        "Kernel time is not busy time: kernels on overlapping streams each count their whole duration (15.000 ms summed "
        "over 3 stream(s); the union of their intervals, device by device, is 12.000 ms).",
        "Kernel time is not a cost: the timed repetitions give the cost, and a profiled repetition is never one of them."]
    assert all("\n" not in note for note in table["notes"])


def test_the_whole_capture_holds_every_kernel_and_steps_may_be_a_count(tmp_path: Path) -> None:
    path = arm_export(tmp_path / "arm.sqlite")
    table = nsys.kernel_table(path, steps="exec:")
    assert rows(table, "launches", "ms") == {GEMM: (4, 11.0), LEAF: (2, 5.0), NORM: (2, 2.0), SAMPLE: (1, 1.0)}
    assert (table["range"], table["ranges"], table["steps"], table["kernel_ms"]) == (None, 0, 3, 19.0)
    assert rows(nsys.kernel_table(path, steps=4), "ms_step") == {
        GEMM: (2.75,), LEAF: (1.25,), NORM: (0.5,), SAMPLE: (0.25,)}
    bare = nsys.kernel_table(path, "run:arm")
    assert bare["steps"] is None and rows(bare, "ms_step") == {GEMM: (None,), LEAF: (None,), NORM: (None,)}
    assert nsys.kernel_table(path, "run:arm", "no such step")["steps"] is None


def test_a_graph_traced_whole_is_one_row_of_spans_in_place_of_its_kernels(tmp_path: Path) -> None:
    """Without `--cuda-graph-trace=node` the two steps' graph launches are two spans and no kernel; the worker's launch is
    still a kernel. The warm-up's graph launch is outside the run."""

    path = export(tmp_path / "graph.sqlite", [(10, 50, ENGINE, "run:arm")],
                  [("RUNTIME", 1, ENGINE, 1, []), ("RUNTIME", 11, ENGINE, 2, []),
                   ("DRIVER", 12, WORKER, 3, [(LEAF, 13, 17, 9)]), ("RUNTIME", 21, ENGINE, 4, [])],
                  graphs=((1, 2, 6), (2, 12, 16), (4, 22, 26)))
    table = nsys.kernel_table(path, "run:arm")
    assert rows(table, "launches", "ms") == {nsys.WHOLE_GRAPH: (2, 8.0), LEAF: (1, 4.0)}
    assert "--cuda-graph-trace=node" in nsys.WHOLE_GRAPH
    assert rows(nsys.kernel_table(path), "launches", "ms")[nsys.WHOLE_GRAPH] == (3, 12.0)


def test_an_arms_extra_a_step_over_stock_is_by_kernel(tmp_path: Path) -> None:
    """The hashing only the arm runs is all extra; the GEMM both run is the difference, the arm's longer and more launches;
    the norm is the same in both; the sampler only stock runs is negative. The rows sum to the arm's kernel time a step
    less stock's. Stock's table is its whole capture with the engine's count of steps."""

    arm = nsys.kernel_table(arm_export(tmp_path / "arm.sqlite"), "run:arm", "exec:")
    stock = nsys.kernel_table(stock_export(tmp_path / "stock.sqlite"), steps=2)
    extra = nsys.extra_over_stock(arm, stock)
    assert rows(extra, "arm_ms_step", "stock_ms_step", "extra_ms_step", "share") == {
        LEAF: (2.5, 0.0, 2.5, 0.625), GEMM: (4.0, 2.0, 2.0, 0.5), NORM: (1.0, 1.0, 0.0, 0.0),
        SAMPLE: (0.0, 0.5, -0.5, -0.125)}
    assert [k["name"] for k in extra["kernels"]] == [LEAF, GEMM, NORM, SAMPLE]
    assert (extra["arm_ms_step"], extra["stock_ms_step"], extra["extra_ms_step"]) == (7.5, 3.5, 4.0)
    assert sum(k["extra_ms_step"] for k in extra["kernels"]) == extra["extra_ms_step"]
    assert extra["notes"] == arm["notes"]
    less = nsys.extra_over_stock(stock, arm)
    assert less["extra_ms_step"] == -4.0 and rows(less, "extra_ms_step", "share")[LEAF] == (-2.5, None)
    with pytest.raises(ValueError, match="the stock's table has no steps"):
        nsys.extra_over_stock(arm, nsys.kernel_table(tmp_path / "stock.sqlite"))
    with pytest.raises(ValueError, match="the arm's table has no steps"):
        nsys.extra_over_stock(nsys.kernel_table(tmp_path / "arm.sqlite", "run:arm"), stock)


def test_the_tables_print_as_markdown_with_the_rest_summed_and_both_notes(tmp_path: Path) -> None:
    arm = nsys.kernel_table(arm_export(tmp_path / "arm.sqlite"), "run:arm", "exec:")
    stock = nsys.kernel_table(stock_export(tmp_path / "stock.sqlite"), "run:stock", "exec:")
    assert nsys.kernel_lines(arm, top=2) == "\n".join([
        "GPU kernel time by kernel, 1 NVTX range(s) `run:arm`, 2 step(s)",
        "",
        "| kernel | launches | ms | ms a step | share |",
        "|---|---:|---:|---:|---:|",
        f"| `{GEMM}` | 3 | 8.000 | 4.000 | 53.3% |",
        f"| `{LEAF}` | 2 | 5.000 | 2.500 | 33.3% |",
        "| 1 more | 2 | 2.000 | 1.000 | 13.3% |",
        "| all | 7 | 15.000 | 7.500 | 100.0% |",
        "",
        *arm["notes"]]) + "\n"
    assert nsys.kernel_lines(nsys.extra_over_stock(arm, stock)) == "\n".join([
        "Extra GPU kernel time a step over stock, by kernel: +4.000 ms (the arm's 7.500 ms less stock's 3.500 ms)",
        "",
        "| kernel | arm, ms a step | stock, ms a step | extra, ms a step | share of the extra |",
        "|---|---:|---:|---:|---:|",
        f"| `{LEAF}` | 2.500 | 0.000 | +2.500 | 62.5% |",
        f"| `{GEMM}` | 4.000 | 2.000 | +2.000 | 50.0% |",
        f"| `{NORM}` | 1.000 | 1.000 | +0.000 | 0.0% |",
        f"| `{SAMPLE}` | 0.000 | 0.500 | -0.500 | -12.5% |",
        "| all | 7.500 | 3.500 | +4.000 | 100.0% |",
        "",
        *arm["notes"]]) + "\n"
    whole = nsys.kernel_lines(nsys.kernel_table(tmp_path / "arm.sqlite"))
    assert whole.startswith("GPU kernel time by kernel, the whole capture\n")
    assert "| all | 9 | 19.000 |  | 100.0% |" in whole
    long = {**arm, "kernels": [{**arm["kernels"][0], "name": "f<a|b>" + "x" * 200}]}
    assert f"| `f<a\\|b>{'x' * 93}…` | 3 |" in nsys.kernel_lines(long)


def test_the_command_prints_the_arms_table_and_its_extra_or_every_table_as_json(tmp_path: Path, capsys) -> None:
    arm, stock = arm_export(tmp_path / "arm.sqlite"), stock_export(tmp_path / "stock.sqlite")
    tables = {"arm": nsys.kernel_table(arm, "run:", "exec:"), "stock": nsys.kernel_table(stock, "run:", 2)}
    tables["extra"] = nsys.extra_over_stock(tables["arm"], tables["stock"])
    argv = [str(arm), "--range", "run:", "--steps", "exec:", "--stock", str(stock), "--stock-steps", "2"]
    assert nsys.main([*argv, "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == tables == json.loads(json.dumps(tables))
    assert nsys.main([*argv, "--top", "3"]) == 0
    assert capsys.readouterr().out == "\n".join(nsys.kernel_lines(tables[key], 3) for key in ("arm", "extra")) + "\n"
    assert nsys.main([str(arm)]) == 0
    assert capsys.readouterr().out == nsys.kernel_lines(nsys.kernel_table(arm)) + "\n"


def test_an_export_without_the_range_or_without_kernels_is_refused_by_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no NVTX range of the export starts with 'run:other'"):
        nsys.kernel_table(arm_export(tmp_path / "arm.sqlite"), "run:other")
    sqlite3.connect(tmp_path / "empty.sqlite").close()
    with pytest.raises(ValueError, match="the export has no CUPTI_ACTIVITY_KIND_KERNEL: profile with --trace=cuda,nvtx"):
        nsys.kernel_table(tmp_path / "empty.sqlite")
