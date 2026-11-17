"""The point's node side: one point timed under one lease, and the lease asked of the cards on the NUMA node of the run's
CPUs."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest
from steady_standins import CARDS, CUDA, DRIVER, PIN, POINT, RUN, STAMP, Node, numa
from test_steady import END, SESSIONS, START

from verity_bench import kit
from verity_bench import steady as S
from verity_bench import steady_node as N
from verity_bench import steady_spec as SP


def test_the_node_side_times_the_point_under_one_lease_and_computes_its_verdict_after_it(tmp_path, monkeypatch):
    node = Node(tmp_path, monkeypatch)
    written, write, append = [], N._write, kit.Writer._open
    monkeypatch.setattr(N, "_write", lambda path, doc: (written.append(path.name), write(path, doc))[1])
    monkeypatch.setattr(kit.Writer, "_open", lambda self: (written.append("appended"), append(self))[1])
    assert node.point() == 0
    assert "result.json" not in written and "appended" in written, "the result is the kit's writer's"

    stager, lease_argv, timer = node.ran
    scratch = Path(lease_argv[-1])
    assert scratch.parent == node.scratch_root and not scratch.exists(), "the scratch is removed"
    assert stager == [sys.executable, SP.STAGER, "gemm64", "1", str(scratch / "z" / "gemm64" / "c1" / "stage"), "--unbound"]
    assert kit.lease(1, 90) == ["gpu-lease", "1", "--wait", "--max-min", "90", "--"], "the kit's lease, and beside it:"
    assert lease_argv == ["gpu-lease", "1", "--wait", "--max-min", "90", "--cores", "20-29", "--preemptible",
                          "--mem-gb", "80", "--",
                          sys.executable, "-m", "verity_bench.steady_node", "--spec", node.spec, "--point", POINT,
                          "--identity", node.identity, "--inside", str(scratch)]
    assert timer == [sys.executable, SP.TIMER, "--bin", str(node.binary), "--staged", str(scratch / "z"),
                     "--types", "gemm64", "--provers", "2", "--out", str(scratch / "rate"), "--device", "3",
                     "--sessions", "16", "--warm", "2", "--ahead", "2"]

    [row] = node.read(S.RATES)
    assert row[SP.VERSIONS] == {"driver": DRIVER, "cuda_toolkit": CUDA}, "read on the node and written in the row"
    assert row == node.row | {SP.VERSIONS: row[SP.VERSIONS]}
    start = (node.run_dir / START).read_text()
    assert start == (node.run_dir / END).read_text() == (
        "epoch 1700000000\nlease 0 held 5m pid=11 run=r-other who=other\napp GPU 0 pid 12 100MiB\n")
    kept = sorted(p.relative_to(node.run_dir).as_posix() for p in (node.run_dir / "out" / "rate").rglob("*") if p.is_file())
    assert kept == ["out/rate/gemm64-k2/prover-gpu3-0.log", SESSIONS, S.RATES, "out/rate/server-rate.log"]
    assert (node.run_dir / "out/rate/gemm64-k2/prover-gpu3-0.log").stat().st_size == N.KEPT_LOG_BYTES

    point = node.read(N.POINT_FILE)
    assert point["identity"] == node.identity and point["point"]["provers"] == 2 and "refused" not in point
    assert point["prover"] == {"path": str(node.binary), "sha256": hashlib.sha256(b"#!/bin/sh\n").hexdigest()}
    assert point["cores"] == {"count": 10, "list": "20-29"}
    assert point["statement"]["pin"] == PIN and point["statement"]["from"] == SP.STAGER
    assert point["timed"] == {"refused": None, "device": "3", "host": STAMP, S.CARD_NUMA: 0,
                              SP.VERSIONS: {"driver": DRIVER, "cuda_toolkit": CUDA}, "exit": 0, "rows": 1,
                              "kept_session": {"session": None, "why": "the verifiers filed no session"}}
    assert STAMP["cuda"] != CUDA, "the stamp's `cuda` is the driver's newest, never the toolkit's version"
    verdict = node.read(N.VERDICT_FILE)
    assert (verdict["identity"], verdict["point"], verdict["passed"], verdict["why"]) == (node.identity, POINT, True, None)
    assert list(verdict["verdict"]["checks"]) == list(S.CHECKS)
    result = node.read("result.json")
    assert result["kind"] == N.KIND and result["validation"]["status"] == "passed" and result["run_id"] == RUN
    assert result["workload_fingerprint"]["identity"] == node.identity
    assert result["schema"] == kit.RESULT_SCHEMA
    assert result["measurements"] == [{
        "name": "benchmarks/flock-steady.statements-per-s", "value": row["statements_per_s"], "unit": "1/s", "mark": "measured",
        "commit": STAMP["commit"], "run": RUN, "ci95": None, "target": None, "verdict": None,
        "inputs": ["measured", "spec steady", f"identity {node.identity}"], "member": POINT, "host": STAMP}], (
        "the kit's writer's: the point's own, by its id, from its spec at its identity")

    spec = SP.load_spec(node.root / node.spec, root=node.root)
    assert SP.point_why(row, S.files_in(node.run_dir), spec, SP.point(spec, POINT), own=True) is None


def test_the_lease_is_asked_of_the_cards_on_the_numa_node_of_the_runs_cpus(tmp_path, monkeypatch):
    """A slot gives the run its CPUs and no card: without this the lease takes the first free card of the node."""

    node = Node(tmp_path, monkeypatch)
    numa(node)
    assert node.point() == 0
    lease = node.ran[1]
    assert lease[:lease.index("--")] == ["gpu-lease", "1", "--wait", "--max-min", "90", "--on", "2,3", "--cores", "20-29",
                                         "--preemptible", "--mem-gb", "80"]
    assert node.read(N.POINT_FILE)["lease_on"] == {"numa_nodes": [1], "cards": [2, 3]}


def test_a_booked_card_is_leased_alone_once_it_is_on_the_numa_node_of_the_runs_cpus(tmp_path, monkeypatch):
    """Either card of the NUMA node may be another lane's booking in the same hours."""

    node = Node(tmp_path, monkeypatch)
    numa(node)
    assert node.point(card=3) == 0
    lease = node.ran[1]
    assert lease[:lease.index("--")] == ["gpu-lease", "1", "--wait", "--max-min", "90", "--on", "3", "--cores", "20-29",
                                         "--preemptible", "--mem-gb", "80"]
    assert node.read(N.POINT_FILE)["lease_on"] == {"numa_nodes": [1], "cards": [3]}


@pytest.mark.parametrize("cards, says", [
    (CARDS, ("the point was launched on card 0, which isn't on NUMA node 1 of the run's CPUs (its cards are 2, 3): "
             "launch it with the cores of card 0's slot")),
    (None, ("the point was launched on card 0, and the node doesn't say which cards are on the NUMA node of the run's "
            "CPUs")),
])
def test_a_booked_card_off_the_numa_node_of_the_runs_cpus_is_refused_before_the_statement(tmp_path, monkeypatch,
                                                                                           cards, says):
    node = Node(tmp_path, monkeypatch)
    numa(node, cards=cards)
    assert node.point(card=0) == N.REFUSED
    assert node.ran == [] and says in node.read(N.POINT_FILE)["refused"]


def test_a_node_that_cannot_run_gpu_lease_is_refused_before_a_statement_is_staged_with_the_script_to_install(
        tmp_path, monkeypatch):
    """The kit's lease calls `gpu-lease` by its bare name. A node without it on its path would stage or fetch the
    statement and then fail to take the lease; the point says so first, with what to put on the path."""
    node = Node(tmp_path, monkeypatch)
    answered = node.ask
    node.ask = lambda argv: ((N.NOT_RUN, "FileNotFoundError: [Errno 2] No such file or directory: 'gpu-lease'")
                             if list(argv) == ["gpu-lease", "status"] else answered(argv))

    assert node.point() == N.REFUSED
    assert node.ran == [], "nothing is staged, fetched or leased"
    refused = node.read(N.POINT_FILE)["refused"]
    assert refused == ("`gpu-lease` can't be run on this node (FileNotFoundError: [Errno 2] No such file or directory: "
                       "'gpu-lease'), and the timed step runs under its lease of one card: put "
                       "tools/research/src/research/pods/sh/gpu_lease.sh of this tree on the node's path as "
                       "`gpu-lease`. A copy on the node's own disk goes when the node is reset")
    assert node.read(kit.RESULT)["validation"] == {
        "status": "failed", "checker": "verity_bench.steady_spec.point_why",
        "detail": f"refused, nothing timed: {refused}", "evidence": [N.POINT_FILE]}
    assert not list(node.scratch_root.iterdir()), "no scratch was made"


def test_a_command_the_node_cannot_run_is_asked_with_the_exit_code_a_shell_gives(tmp_path):
    rc, text = N._ask([str(tmp_path / "no-such-program")])
    assert rc == N.NOT_RUN == 127 and text.startswith("FileNotFoundError: ")


@pytest.mark.parametrize("answer", [(0, "0 free\n"), (1, ""), (2, "gpu-lease: no state directory")])
def test_a_gpu_lease_that_runs_is_not_refused_at_the_start_whatever_its_status_says(answer):
    """Only a tool that can't be run at all is refused there: what a lease that can't be taken says is the lease's own."""
    N.lease_tool(lambda argv: answer)


#: a node that doesn't say which cards are on the NUMA node of the run's CPUs: what it says instead, and why the lease
#: is asked of any card
ANY_CARD = [
    ({"node_node0_cpulist": "0-24\n", "node_node1_cpulist": "25-31\n"}, [0, 1], "the node lists 2 NUMA nodes for the run's CPUs"),
    ({"node_node0_cpulist": None, "node_node1_cpulist": None}, [], "the node lists 0 NUMA nodes for the run's CPUs"),
    ({"cards": None}, [1], "`nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader` exited 1"),
    ({"pci_0000_b1_00_0_numa_node": "-1\n", "pci_0000_ca_00_0_numa_node": None}, [1],
     "none of the node's 3 cards says it is on NUMA node 1"),
]


@pytest.mark.parametrize("says, nodes, why", ANY_CARD, ids=[why for *_, why in ANY_CARD])
def test_a_node_that_does_not_say_which_cards_are_near_the_runs_cpus_leases_any_card_and_the_point_says_why(
        tmp_path, monkeypatch, says, nodes, why):
    node = Node(tmp_path, monkeypatch)
    numa(node, **({"cards": CARDS} | says))
    assert node.point() == 0
    assert "--on" not in node.ran[1]
    assert node.read(N.POINT_FILE)["lease_on"] == {"numa_nodes": nodes, "cards": None, "why": why}


def test_a_node_that_lists_nothing_of_its_numa_nodes_leases_any_card(tmp_path, monkeypatch):
    node = Node(tmp_path, monkeypatch)
    assert node.point() == 0 and "--on" not in node.ran[1]
    assert node.read(N.POINT_FILE)["lease_on"]["cards"] is None
