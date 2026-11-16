"""What the node side refuses and how it stops: before the card, inside the lease, on a signal, and a timed point that did
not pass."""

from __future__ import annotations

import re
import signal
import sys
from pathlib import Path

import pytest
from steady_standins import (
    ANSWERS,
    CUDA,
    DRIVER,
    FILES,
    NVCC,
    POINT,
    Node,
    git,
    numa,
    stored,
)
from test_steady import DROP, START, lease

from verity_bench import settings
from verity_bench import steady as S
from verity_bench import steady_cards as C
from verity_bench import steady_node as N
from verity_bench import steady_plan as P
from verity_bench import steady_spec as SP
from verity_bench.tool import STEADY_POINT

#: the node side refusing before any card is asked for: the stand-in's change and what the refusal says
BEFORE_THE_CARD = [
    ({"cpus": 16}, r"the run has 16 CPUs \(20-35\), and the `cores` of spec 'steady' is 10"),
    ({"circuit": "another circuit\n"}, "the statement's circuit has SHA-512 .*, and the `pin` of row 'gemm64' is"),
    ({"said": "ab" * 64}, r"the statement's pin.txt, which the timing script writes in the row, says abababababababab\.\.\."),
    ({"units": 2}, "the statement holds 2 units, and the `n` of row 'gemm64' is 1"),
]


@pytest.mark.parametrize("change, says", BEFORE_THE_CARD, ids=[says[:30] for _, says in BEFORE_THE_CARD])
def test_the_node_side_refuses_another_point_before_any_card_is_asked_for(tmp_path, monkeypatch, change, says):
    node = Node(tmp_path, monkeypatch, **change)
    assert node.point() == N.REFUSED
    assert not any(argv[0] == "gpu-lease" for argv in node.ran), "no card was asked for"
    result, point = node.read("result.json"), node.read(N.POINT_FILE)
    assert result["validation"]["status"] == "failed" and result["measurements"] == []
    assert re.search(says, point["refused"]) and result["validation"]["detail"] == f"refused, nothing timed: {point['refused']}"
    assert not (node.run_dir / N.VERDICT_FILE).exists() and not list(node.scratch_root.iterdir())


def test_the_node_side_refuses_a_tree_whose_point_is_not_the_one_it_was_launched_as(tmp_path, monkeypatch):
    node = Node(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match=f"point identity changed: this tree gives {POINT} {node.identity}"):
        node.point(identity="0" * 64)
    (node.root / "verity/protocols/proofs/flock/prover/src/main.rs").write_text("fn main() { /* edited on the node */ }\n")
    with pytest.raises(ValueError, match="a file its time depends on differs from the launching checkout's"):
        node.point()
    (node.root / "verity/protocols/proofs/flock/prover/src/main.rs").write_text(FILES["verity/protocols/proofs/flock/prover/src/main.rs"])
    # a shipped tree is no checkout: a file a build left under the closure's directories counts, and the refusal says so
    (left := node.root / "verity/protocols/proofs/flock/prover/target/built.o").parent.mkdir()
    left.write_text("object\n")
    with pytest.raises(ValueError) as shipped:
        node.point()
    assert ("This tree is no checkout, so every file under verity/protocols/proofs/flock/prover, verity/protocols/proofs/flock/prover/pod "
            "counts: one the launching checkout doesn't track (what a build or a setup left there) changes the identity "
            "too. Its closure here is ") in str(shipped.value)
    assert re.search(r"Its closure here is \d+ files \(SHA-256 [0-9a-f]{16}\.\.\.\); `plan` prints the launching "
                     r"checkout's as its `toolchain`$", str(shipped.value))
    # in a checkout the same file is untracked and no part of the point, and a tracked file that differs is
    left.unlink()
    git(node.root)
    left.write_text("object\n")
    (node.root / "verity/protocols/proofs/flock/prover/src/main.rs").write_text("fn main() { /* edited in the checkout */ }\n")
    with pytest.raises(ValueError) as checkout:
        node.point()
    assert "differs from the launching checkout's. Its closure here is " in str(checkout.value)
    assert "no checkout" not in str(checkout.value)
    monkeypatch.setenv("RUST_MIN_STACK", "1")
    (node.root / "verity/protocols/proofs/flock/prover/src/main.rs").write_text(FILES["verity/protocols/proofs/flock/prover/src/main.rs"])
    with pytest.raises(ValueError, match=r"differs from spec 'steady''s env and setup in \['RUST_MIN_STACK'\]"):
        node.point()
    assert node.ran == [] and not (node.run_dir / "result.json").exists()
    assert N.main(["--spec", node.spec, "--point", POINT, "--identity", node.identity]) == 2


def test_the_node_side_refuses_a_run_without_the_prover_its_setup_builds(tmp_path, monkeypatch):
    node = Node(tmp_path, monkeypatch)
    monkeypatch.setenv("VERITY_BENCH_STEADY_BIN", str(tmp_path / "no-prover"))
    assert node.point() == N.REFUSED and node.ran == []
    assert "the spec's setup left no prover" in node.read(N.POINT_FILE)["refused"]


#: the timed step refusing inside the lease: what the node answers otherwise, and what the refusal says
INSIDE_THE_LEASE = [
    # the toolkit's version is nvcc's (12.8.93), never the stamp's `cuda`, the newest the driver serves (12.9)
    ({"stamp": {"driver": "570.86.16"}},
     "the node's driver and CUDA toolkit are {'driver': '570.86.16', 'cuda_toolkit': '12.8.93'}, and spec 'steady' says"),
    ({"stamp": {"card": "NVIDIA B200"}},
     "the lease's card is a NVIDIA B200, not the `card` of spec 'steady' (h200: NVIDIA H200)"),
    ({"stamp": {"card": "none", "gpus": [], "driver": None, "cuda": None, "why": "no nvidia-smi on PATH"}},
     "the host stamp doesn't say what the lease's card is (no nvidia-smi on PATH)"),
    ({"stamp": {"gpus": [{"uuid": "GPU-3", "sm_mhz": 1980, "mem_mhz": 3201}, {"uuid": "GPU-4", "sm_mhz": 0, "mem_mhz": 0}]}},
     ("the host stamp doesn't say what the lease's card is (it names 2 GPUs and the driver '570.86.15'): inside a "
      "gpu-lease of one card it is that card's, which gpu-lease names in GPU_LEASE_UUID")),
]


@pytest.mark.parametrize("change, says", INSIDE_THE_LEASE, ids=[says[:40] for _, says in INSIDE_THE_LEASE])
def test_the_timed_step_refuses_a_card_or_versions_that_are_not_the_specs(tmp_path, monkeypatch, change, says):
    node = Node(tmp_path, monkeypatch, **change)
    assert node.point() == N.REFUSED
    assert [argv[0] for argv in node.ran] == [sys.executable, "gpu-lease"] and node.ran[0][1] == SP.STAGER, "nothing timed"
    assert says in node.read(N.POINT_FILE)["refused"] and says in node.read("result.json")["validation"]["detail"]
    assert not (node.run_dir / S.RATES).exists() and not (node.run_dir / START).exists()


def test_the_timed_step_refuses_a_toolkit_that_is_not_the_specs_though_the_point_read_it_before_the_lease(tmp_path,
                                                                                                         monkeypatch):
    """The point refuses a wrong toolkit before the lease, so no run of the point reaches the timed step's own read
    with one: an nvcc that differs inside the lease (another setup, another path) is refused there too."""

    node = Node(tmp_path, monkeypatch, answers={"nvcc --version": NVCC.replace(CUDA, "12.9.41")})
    spec = SP.load_spec(node.root / node.spec, root=node.root)
    (node.run_dir / "out").mkdir()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    assert N.timed(spec, SP.point(spec, POINT), node.scratch_root, node.run_dir, run=node.run, ask=node.ask,
                   now=lambda: 1_700_000_000.0, stamp=lambda: dict(node.host)) == N.REFUSED
    assert node.ran == [], "nothing timed"
    assert node.read(N.TIMED_FILE)["refused"].startswith(
        f"the node's driver and CUDA toolkit are {{'driver': '{DRIVER}', 'cuda_toolkit': '12.9.41'}}, and spec 'steady' "
        f"says {{'driver': '{DRIVER}', 'cuda_toolkit': '{CUDA}'}}")
    assert not (node.run_dir / START).exists()


#: the toolkit's version refused before any statement or card: what the node's nvcc answers, and what the refusal says
TOOLKIT_BEFORE_THE_STATEMENT = [
    ({"nvcc --version": NVCC.replace(CUDA, "12.9.41")},
     ("the node's CUDA toolkit is 12.9.41 (its nvcc), and spec 'steady' says 12.8.93: a row at another toolkit is another "
      "point. Set the spec's `cuda_toolkit` to the node's")),
    ({"nvcc --version": None}, "the nvcc on the path the spec's setup leaves doesn't say its version"),
    ({"nvcc --version": "nvcc: no version here\n"}, "the nvcc on the path the spec's setup leaves doesn't say its version"),
]


@pytest.mark.parametrize("answers, says", TOOLKIT_BEFORE_THE_STATEMENT,
                         ids=[says[:40] for _, says in TOOLKIT_BEFORE_THE_STATEMENT])
def test_a_toolkit_that_is_not_the_specs_is_refused_before_the_statement_and_the_lease(tmp_path, monkeypatch, answers,
                                                                                        says):
    """The toolkit's version needs no card, so a wrong one is refused before the statement is staged or fetched and
    before the lease: nothing is spent on a node whose rows would be another point."""

    node = Node(tmp_path, monkeypatch, answers=answers)
    stored(node)
    assert node.point() == N.REFUSED
    assert node.ran == [] and node.asked == [], "nothing staged, fetched or leased"
    assert says in node.read(N.POINT_FILE)["refused"] and says in node.read("result.json")["validation"]["detail"]
    assert not list(node.scratch_root.iterdir())


def test_a_lease_that_never_ran_the_timed_step_is_a_refusal(tmp_path, monkeypatch):
    node = Node(tmp_path, monkeypatch, lease_exit=75)
    assert node.point() == N.REFUSED
    assert node.read(N.POINT_FILE)["refused"] == "the timed step never began: gpu-lease exited 75 (out/lease.log)"


def test_a_timed_step_stopped_before_its_end_is_a_refusal_that_says_it_began(tmp_path, monkeypatch):
    """A lease preempted mid-step: the step had recorded the node at its start and wrote no record of its end."""

    node = Node(tmp_path, monkeypatch)
    timer = node.run

    def preempted(argv, log):
        if argv[0] == "gpu-lease":
            assert timer(argv, log) == 0
            (node.run_dir / N.TIMED_FILE).unlink()
            return 143
        return timer(argv, log)

    node.run = preempted
    assert node.point() == N.REFUSED
    assert (node.run_dir / START).is_file()
    assert node.read(N.POINT_FILE)["refused"] == (
        "the timed step was stopped before its end and left no row, as a preempted lease stops it: gpu-lease exited "
        "143 (out/lease.log)")
    assert node.read("result.json")["validation"]["status"] == "failed"


@pytest.mark.parametrize("ended, record, says", [
    (RuntimeError("the stand-in fell over"), {"error": "RuntimeError: the stand-in fell over"},
     "the timed step failed: RuntimeError: the stand-in fell over (out/lease.log)"),
    (SystemExit(N.STOPPED), {"stopped": "SystemExit: 143"},
     ("the timed step was stopped before its end and left no row, as a preempted lease stops it: gpu-lease exited 1 "
      "(out/lease.log)")),
], ids=["raised", "stopped"])
def test_a_timed_step_that_raises_or_is_stopped_still_writes_its_record_and_the_point_says_what_ended_it(
        tmp_path, monkeypatch, ended, record, says):
    node = Node(tmp_path, monkeypatch)
    timer = node.run

    def crashing(argv, log):
        if argv[1:2] == [SP.TIMER]:
            raise ended
        if argv[0] == "gpu-lease":
            with pytest.raises(type(ended)):
                timer(argv, log)
            return 1
        return timer(argv, log)

    node.run = crashing
    assert node.point() == N.REFUSED
    step = node.read(N.TIMED_FILE)
    assert step == step | {"refused": None, "device": "3"} | record and "exit" not in step
    assert node.read(N.POINT_FILE)["refused"] == says


def test_the_leased_cards_numa_node_is_the_kernels_word_on_its_pci_address(tmp_path, monkeypatch):
    node = Node(tmp_path, monkeypatch)
    monkeypatch.undo()
    monkeypatch.setattr(C, "DEVICES", node.sys / "pci")
    numa(node, pci_0000_65_00_0_numa_node="-1\n", pci_0000_b1_00_0_numa_node=None)
    assert [C.card_numa_node(card, node.ask) for card in ("0", "1", "2", "3", "7", "GPU-3")] == [0, None, None, 1, None,
                                                                                                None]
    node.answers = node.answers | {" ".join(C.CARDS): None}
    assert C.card_numa_node("3", node.ask) is None


@pytest.fixture
def signals():
    """TERM's and INT's handlers, put back after a test that lets the node side set its own."""

    kept = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}
    yield
    for number, handler in kept.items():
        signal.signal(number, handler)


@pytest.mark.parametrize("number", [signal.SIGTERM, signal.SIGINT], ids=["TERM", "INT"])
def test_a_cancelled_or_timed_out_point_removes_its_scratch_and_exits_143(tmp_path, monkeypatch, signals, number):
    """A cancel or a timeout reaches the node side while it waits for its lease: the run ends with 143 and no result,
    and the statement it staged is gone from the scratch."""

    node = Node(tmp_path, monkeypatch)
    seen = {}

    def lease(argv, log):
        if argv[0] != "gpu-lease":
            return Node.run(node, argv, log)
        seen["scratch"] = scratch = Path(argv[-1])
        seen["staged"] = (scratch / "z" / "gemm64" / "c1" / "stage" / "circuit.txt").is_file()
        assert signal.getsignal(number) is N._stopped, "the signal would end the tests' own process"
        signal.raise_signal(number)
        return 0

    node.run = lease
    N.stop_on_signals()
    with pytest.raises(SystemExit) as stopped:
        node.point()
    assert stopped.value.code == N.STOPPED == 143
    assert seen["staged"] and not seen["scratch"].exists() and list(node.scratch_root.iterdir()) == []
    assert not (node.run_dir / "result.json").exists() and not (node.run_dir / N.POINT_FILE).exists()


def test_main_ends_on_a_signal_and_a_command_it_waits_for_is_ended_first(tmp_path, monkeypatch, signals):
    """`main` sets the handlers; and `_run`, ended while it waits, sends its command TERM and waits for it, so a lease's
    holder stops its step and releases its card before the scratch goes. The command here signals its parent, the
    tests' process, once it can be signalled itself, and writes a file when TERM reaches it."""

    node = Node(tmp_path, monkeypatch)
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, signal.SIG_DFL)
    assert N.main(["--spec", node.spec, "--point", POINT, "--identity", "0" * 64]) == 2
    assert [signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)] == [N._stopped, N._stopped]

    ended = tmp_path / "ended"
    command = ("import os, signal, sys\n"
               "def end(*_):\n"
               "    open(sys.argv[1], 'w').write('ended by TERM')\n"
               "    sys.exit(0)\n"
               "signal.signal(signal.SIGTERM, end)\n"
               "os.kill(os.getppid(), signal.SIGTERM)\n"
               "while True:\n"
               "    signal.pause()\n")
    with pytest.raises(SystemExit) as stopped:
        N._run([sys.executable, "-c", command, str(ended)], tmp_path / "command.log")
    assert stopped.value.code == 143 and ended.read_text() == "ended by TERM"
    assert (tmp_path / "command.log").read_text() == ""


#: a point that was timed and isn't a passed point: the stand-in's change and what the verdict says
TIMED_NOT_PASSED = [
    ({"row": {"flat": False}}, "fail: `flat` (fail): its statement cache's hit rate wasn't flat"),
    ({"row": {"cores": {"count": 8, "list": "0-7", "numa_nodes": [0]}}}, "fail: it ran on 8 cores; the spec's `cores` is 10"),
    ({"row": {"cores": DROP}}, "cannot tell: its row doesn't say which cores it ran on"),
    ({"row": {"cores": {"count": 10, "list": "20-29"}}},
     "cannot tell: `one_numa_node` (cannot tell): its row lacks `cores.numa_nodes`"),
    ({"row": {"cores": {"count": 10, "list": "20-23,24-29", "numa_nodes": [0, 1]}}},
     "fail: `one_numa_node` (fail): its CPUs (20-23,24-29) are on 2 NUMA nodes (0, 1)"),
    ({"row": {"flat": DROP}}, "cannot tell: `flat` (cannot tell): "),
    ({"leases": None}, "cannot tell: `one_lease_own_card` (cannot tell): the run holds no `gpu-lease-usage.jsonl`"),
    ({"leases": lease(0)}, "fail: `one_lease_own_card` (fail): its lease held GPU 0; its provers ran on GPU 3"),
    ({"answers": {"gpu-lease status": None}}, "cannot tell: `card_clear_start_end` (cannot tell): "),
    ({"answers": {"nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader": "GPU-3, 99, 5 MiB\n"}},
     "fail: `card_clear_start_end` (fail): GPU 3 wasn't its own: at its start, a compute process (pid 99 5MiB)"),
    ({"answers": {"gpu-lease status": ANSWERS["gpu-lease status"] + "3 held 9m pid=5 run=r-third who=x cmd=y\n"}},
     "fail: `card_clear_start_end` (fail): GPU 3 wasn't its own: at its start, another run's lease (3 held 9m pid=5 run=r-third who=x)"),
    ({"row": {"type": "gemm32"}}, "cannot tell: the run's `out/rate/rate.json` holds 0 rows of `gemm64` at 2 provers"),
    ({"card_node": 1}, "fail: `one_numa_node` (fail): its card is on NUMA node 1, and its CPUs on NUMA node 0"),
    ({"card_node": None}, ("cannot tell: `one_numa_node` (cannot tell): its node didn't say which NUMA node its card is "
                           "on (its row's `card_numa_node` is null)")),
    # a status that ran and named no card, or other cards and not the row's, or the row's as unreadable, says nothing of it
    ({"answers": {"gpu-lease status": ""}},
     ("cannot tell: `card_clear_start_end` (cannot tell): `out/cotenants-k2-start.txt` has no list of other runs' leases: "
      "gpu-lease status gave no answer (no line of card 3 in its output)")),
    ({"answers": {"gpu-lease status": "0 free\n1 free\n13 held 1m pid=5 run=r-x who=x cmd=y\nforeign 3 pid=9 who=k8s\n"}},
     ("cannot tell: `card_clear_start_end` (cannot tell): `out/cotenants-k2-start.txt` has no list of other runs' leases: "
      "gpu-lease status gave no answer (no line of card 3 in its output)")),
    ({"answers": {"gpu-lease status": "0 free\n1 free\n3 unreadable\n"}},
     ("cannot tell: `card_clear_start_end` (cannot tell): `out/cotenants-k2-start.txt` has no list of other runs' leases: "
      "gpu-lease status gave no answer (card 3 reads `unreadable` in its output)")),
]


@pytest.mark.parametrize("change, says", TIMED_NOT_PASSED, ids=[says[:44] for _, says in TIMED_NOT_PASSED])
def test_a_timed_point_whose_files_do_not_show_all_the_checks_is_not_passed(tmp_path, monkeypatch, change, says):
    node = Node(tmp_path, monkeypatch, **change)
    assert node.point() == SP.NOT_PASSED
    verdict, result = node.read(N.VERDICT_FILE), node.read("result.json")
    assert verdict["passed"] is False and verdict["why"].startswith(says), verdict["why"]
    assert result["validation"] == result["validation"] | {"status": "failed", "detail": verdict["why"]}
    assert result["measurements"] == []


def test_a_point_that_runs_again_in_its_run_directory_leaves_no_earlier_attempts_records(tmp_path, monkeypatch):
    node = Node(tmp_path, monkeypatch)
    assert node.point() == 0
    monkeypatch.setenv("VERITY_BENCH_STEADY_BIN", str(tmp_path / "no-prover"))
    assert node.point() == N.REFUSED
    left = sorted(p.relative_to(node.run_dir).as_posix() for p in node.run_dir.rglob("*") if p.is_file())
    assert left == [S.LEASES, "out/lease.log", N.POINT_FILE, "out/stage.log", "result.json"]
    assert node.read("result.json")["validation"]["status"] == "failed"


def test_the_node_side_reads_the_environment_through_declared_settings_and_names_the_tools_it_runs():
    """The stager and the timing script are named by their paths in the tree, where the tool's closure covers them."""

    root = Path(__file__).resolve().parents[3]
    assert (root / SP.STAGER).is_file() and (root / SP.TIMER).is_file() and SP.TIMER in SP.CLOSURE
    assert f"timed_n = max(a.sessions, {SP.FLOOR} * K)" in (root / SP.TIMER).read_text(), "the floor a point's count takes"
    assert (root / "benchmarks/harness/steady_point.sh").is_file() and STEADY_POINT.command[-1].endswith("steady_point.sh")
    assert {SP.STAGER, SP.TIMER} <= set(STEADY_POINT.closure)
    assert all(any(path.startswith(pattern.removesuffix("/**")) for pattern in STEADY_POINT.closure) for path in SP.CLOSURE)
    assert all(settings.REGISTRY.known(name) for name in (
        "VERITY_BENCH_STEADY_BIN", "VERITY_BENCH_STEADY_PY", "VERITY_BENCH_STEADY_SCRATCH", "CUDA_VISIBLE_DEVICES"))


def test_the_readme_states_a_steady_state_spec_as_the_code_keeps_it():
    readme = (Path(__file__).resolve().parents[3] / "benchmarks" / "harness" / "README.md").read_text()
    section = readme[readme.index("## A steady-state spec"):readme.index("## Settings")]
    for name in (SP.SCHEMA, "verity_bench.tool:STEADY_POINT", "VERITY_BENCH_STEADY_BIN", "VERITY_BENCH_STEADY_PY",
                 "benchmarks/harness/steady_point.sh", SP.STAGER, SP.TIMER, "steady_plan.completed_rows", P.DERIVATION, SP.VERSIONS,
                 *SP.CLOSURE[:2], "out/verdict.json", "out/point.json"):
        assert f"`{name}`" in section, name
    assert (N.VERDICT_FILE, N.POINT_FILE) == ("out/verdict.json", "out/point.json")
    assert (f"exits 0 for a passed point, {SP.NOT_PASSED} for a point that was timed and didn't pass, {N.REFUSED} where "
            f"nothing was timed, and {N.STOPPED} where\nTERM or INT ended it") in section
    assert "`steady_plan.completed_rows`, `completed`" in section and "runs and arts" in section
    assert all(f"`{name}`" in section for name in ("kit.lease", "kit.host_stamp", "kit.Writer", "jobs.dispatch"))
    spec_keys = {"workload", "card", "cores", "sessions", "ahead", "driver", "cuda_toolkit", "setup", "[env]", "[admission]",
                 "max_minutes", "mem_gb", "[[row]]", "type", "n", "pin", "provers", "warm", "period", "staged"}
    assert all(f"`{key}`" in section for key in spec_keys), [key for key in spec_keys if f"`{key}`" not in section]
