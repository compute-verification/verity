"""`member_seconds.py` on a planted audit: two members' logs, records and stamps at fixed times. Each member's seconds
before its prover's clock and before Lean's first check come out exactly, a negative's later processes on the same member
are not the member's, a member with no Lean loop or no setup line has those parts absent rather than zero, the window's
one put before its sessions is the window's and no member's, a wrapper stamps its start and execs its binary under its
own pid, and each reader run alone is the member's own command on the member's files (on stand-in binaries, under an
injected clock)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import member_seconds as MS  # noqa: E402  (a script beside the suite, importable only once its dir is on the path)

T = 1_700_000_000
BUILT = "statement unit/t instances=8 vus_per_block=1 blocks=8 k_log=10 nbl=1 m=3 regions=1 delta=1 built in {}s\n"


def at(f: Path, s: float) -> None:
    """``f`` last written ``s`` seconds after T (to a millisecond)."""
    ns = T * 10**9 + round(s * 1000) * 10**6
    os.utime(f, ns=(ns, ns))


def stamp(program: str, pid: int, ppid: int, s: float, *argv: object) -> str:
    return f"{program} {pid} {ppid} {T + s:.6f} {' '.join(map(str, argv))}\n"


def run_log(f: Path, s: float, stdout: str = "", stderr: str = "") -> None:
    f.write_text(f"$ cmd\n# rc 0 in 0.0 s\n--- stdout\n{stdout}\n--- stderr\n{stderr}")
    at(f, s)


def plant(tmp_path: Path) -> tuple[Path, Path]:
    """An audit of two members: member 0 behind the ZK server with a negative after it, member 1 with no Lean loop and no
    setup line. Returns the audit and the stamps' directory."""
    audit, d = tmp_path / "scratch" / "rank0", tmp_path / "stamped"
    d.mkdir()
    lines = []
    for k, n in ((0, 8), (1, 4)):
        m, s = audit / f"m{k}", audit / f"session-m{k}"
        (m / "tables").mkdir(parents=True)
        (s / "sessions" / "abc").mkdir(parents=True)
        for f, size in ((m / "circuit.txt", 100), (m / f"pub-{n}.bin", 20), (m / f"inst-{n}.bin", 300), (m / "registered.json", 7),
                        (m / "tables" / "t.u32", 16)):
            f.write_bytes(b"x" * size)
        (s / "draw.json").write_text('{"k": 1}')
    (audit / "partition.json").write_bytes(b"p" * 11)
    (audit / "program.json").write_bytes(b"q" * 13)
    (audit / "session-m0-altered").mkdir()
    (audit / "summary.json").write_text(json.dumps({
        "firewall": "/x/flock-firewall", "staging": {"drawn_s": 1.5},
        "members": [{"k": 0, "template": "T0", "n": 8, "base": 0}, {"k": 1, "template": "T1", "n": 4, "base": 8}],
        "timing": {"a": {"sigma": "s0", "session_s": 20.1}, "b": {"sigma": "s1", "session_s": 9.0}}}))

    m, s = audit / "m0", audit / "session-m0"
    serve = ("serve", "--zk", "--listen", "127.0.0.1:7000", "--out", s / "sessions", "--instances", m / "pub-8.bin", "--circuit",
             m / "circuit.txt", "--tables", m / "tables", "--pin", "abc", "--operator", "one-stage-a0", "--draw-file", s / "draw.json")
    prove = ("prove", "--zk", "--verifier", "127.0.0.1:7000", "--instances", m / "inst-8.bin", "--circuit", m / "circuit.txt",
             "--tables", m / "tables", "--runs", "1", "--warm", "0")
    loop = ("session", "--verifier", "127.0.0.1:7000", "--statement", "verity/flock-circuit", "--circuit", m / "circuit.txt",
            "--public", "/tmp/flock-session-200-1.pub", "--tables", m / "tables", "--session-tables", "1",
            "--partition", audit / "partition.json", "--program", audit / "program.json")
    put = ("archive-put", s / "archive", m / "circuit.txt", m / "pub-8.bin", m / "tables" / "t.u32", audit / "partition.json",
           audit / "program.json", m / "registered.json")
    verify = ("verify", "--statement", "verity/flock-circuit", "--archive", s / "archive", "--circuit", "", "--public", "k1",
              "--zk", "--session", s / "sessions" / "abc")
    lines += [stamp("flock-circuit", 100, 1, 0, *serve), stamp("flock-circuit", 200, 1, 3, *prove),
              stamp("flock-firewall", 201, 200, 6.1, "tap"), stamp("flock-firewall", 202, 200, 6.2, *loop),
              stamp("flock-verify", 300, 1, 20.5, "draw-test", s / "u1-u3.cases.json"),
              stamp("flock-verify", 301, 1, 20.7, *put), stamp("flock-verify", 302, 1, 21.75, *verify),
              # a negative on member 0 once its checks are done: the same commands, later
              stamp("flock-verify", 400, 1, 29, *put), stamp("flock-verify", 401, 1, 30, *verify),
              stamp("flock-circuit", 200, 1, 40, *prove), stamp("flock-firewall", 402, 200, 41, *loop)]
    (s / "serve.log").write_text(BUILT.format("2.50") + "SERVING\n")
    run_log(s / "prove.log", 20, "LIVE\t" + json.dumps({"accepted": True, "e2e_s": 10.0, "wait_s": 3.0, "wait_e2e_s": 2.5,
                                                         "witness_s": 4.0, "prove_total_s": 1.0, "sigma": "s0"}), BUILT.format("3.00"))
    run_log(s / "u1-u3.lean.log", 20.7)
    run_log(s / "archive-put.log", 21.7)
    run_log(s / "lean-verify.log", 24.75, "VERDICT {}", f"inputs: 1500 ms\nsetup (drawn): 250 ms\n{s / 'sessions' / 'abc'}: 1000 ms\n")

    m, s = audit / "m1", audit / "session-m1"
    lines += [stamp("flock-circuit", 500, 1, 21, "serve", "--zk", "--out", s / "sessions", "--instances", m / "pub-4.bin",
                    "--circuit", m / "circuit.txt", "--tables", m / "tables", "--pin", "def"),
              stamp("flock-circuit", 501, 1, 22, "prove", "--zk", "--instances", m / "inst-4.bin", "--circuit", m / "circuit.txt",
                    "--tables", m / "tables"),
              stamp("flock-verify", 502, 1, 30.5, "archive-put", s / "archive", m / "pub-4.bin"),
              stamp("flock-verify", 503, 1, 31, "verify", "--archive", s / "archive", "--session", s / "sessions" / "abc")]
    (s / "serve.log").write_text(BUILT.format("0.50"))
    run_log(s / "prove.log", 30, "LIVE\t" + json.dumps({"accepted": True, "e2e_s": 6.0, "wait_s": 1.0, "wait_e2e_s": 1.0, "sigma": "s1"}),
            BUILT.format("0.75"))
    run_log(s / "archive-put.log", 30.75)
    run_log(s / "lean-verify.log", 32, "", "inputs: 400 ms\nstatement inputs: refused\n")
    (d / MS.STAMPS).write_text("".join(lines))
    return audit, d


def test_a_members_seconds_are_its_own_processes(tmp_path):
    audit, d = plant(tmp_path)
    got = MS.measure(audit, MS.stamps((d / MS.STAMPS).read_text()))
    assert {x: got[x] for x in ("audit", "firewall", "stamped", "staging", "window")} == {
        "audit": "rank0", "firewall": True, "stamped": True, "staging": {"drawn_s": 1.5},
        "window": {"archive_put_s": None, "files": None}}
    m0, m1 = got["members"]
    assert (m0["k"], m0["template"], m0["n"], m0["to_next_member_s"]) == (0, "T0", 8, 1.0) and "alone" not in m0
    ns = T * 10**9
    assert m0["starts_ns"] == {"serve": ns, "prove": ns + 3 * 10**9, "tap": ns + 61 * 10**8, "loop": ns + 62 * 10**8,
                               "draw-test": ns + 205 * 10**8, "archive-put": ns + 207 * 10**8, "verify": ns + 2175 * 10**7}
    assert m0["ends_ns"] == {"prove": ns + 20 * 10**9, "draw-test": ns + 207 * 10**8, "archive-put": ns + 217 * 10**8,
                             "verify": ns + 2475 * 10**7}
    assert m0["bytes"] == {"circuit": 100, "partition": 11, "program": 13, "registered": 7, "public:pub-8.bin": 20,
                           "prover:inst-8.bin": 300, "table:t.u32": 16}
    assert m0["session"] == {"verifier_ahead_s": 3.0, "verifier_built_s": 2.5, "prover_s": 17.0, "prover_built_s": 3.0,
                             "tap_at_s": 3.1, "to_loop_s": 3.2, "loop_to_clock_s": 3.3, "before_clock_s": 6.5, "e2e_s": 10.0,
                             "witness_s": 4.0, "prove_total_s": 1.0, "after_clock_wait_s": 0.5, "accepted": True, "session_s": 20.1}
    assert m0["lean"] == {"waited_s": 0.5, "draw_test_s": 0.2, "archive_put_s": 1.0, "verify_s": 3.0, "inputs_s": 1.5,
                          "setup_s": 0.25, "session_s": 1.0, "check_s": 0.75, "verify_rest_s": 0.5, "before_first_check_s": 3.25}
    # no Lean loop, no draw check and a verify that printed no setup: absent, never zero
    assert (m1["k"], m1["template"], "to_next_member_s" in m1) == (1, "T1", False)
    assert m1["session"] == {"verifier_ahead_s": 1.0, "verifier_built_s": 0.5, "prover_s": 8.0, "prover_built_s": 0.75,
                             "tap_at_s": None, "to_loop_s": None, "loop_to_clock_s": None, "before_clock_s": 2.0, "e2e_s": 6.0,
                             "witness_s": None, "prove_total_s": None, "after_clock_wait_s": 0.0, "accepted": True, "session_s": 9.0}
    assert m1["lean"] == {"waited_s": None, "draw_test_s": None, "archive_put_s": 0.25, "verify_s": 1.0, "inputs_s": 0.4,
                          "setup_s": None, "session_s": None, "check_s": None, "verify_rest_s": None, "before_first_check_s": None}


def test_an_audit_that_archived_its_window_once_counts_the_put_once(tmp_path):
    """`a5.audit` puts every sampled member's files before its sessions (``lean-verify/archive``): the put is the window's,
    a member has none of its own (a negative's later put into its archive is not its), and what it spends before Lean's
    first check is its verify less the check."""
    audit, d = plant(tmp_path)
    for k in (0, 1):
        (audit / f"session-m{k}" / "archive-put.log").unlink()
    (audit / "lean-verify").mkdir()
    run_log(audit / "lean-verify" / "archive-put.log", -0.6)
    with open(d / MS.STAMPS, "a") as f:
        f.write(stamp("flock-verify", 90, 1, -1, "archive-put", audit / "lean-verify" / "archive", "a", "b", "c"))
    got = MS.measure(audit, MS.stamps((d / MS.STAMPS).read_text()))
    assert got["window"] == {"archive_put_s": 0.4, "files": 3}
    m0, m1 = got["members"]
    assert m0["starts_ns"]["archive-put"] is None and m0["ends_ns"]["archive-put"] is None
    assert (m0["lean"]["archive_put_s"], m0["lean"]["before_first_check_s"]) == (None, 2.25)
    assert (m1["lean"]["archive_put_s"], m1["lean"]["before_first_check_s"]) == (None, None)


def test_measure_writes_each_audit_under_the_root(tmp_path, capsys):
    audit, d = plant(tmp_path)
    assert MS.main(["measure", str(tmp_path / "scratch"), "--stamped", str(d), "--no-alone"]) == 0
    line = json.loads(capsys.readouterr().out.removeprefix("MEMBER-SECONDS "))
    assert line == {"audit": "rank0", "window_archive_put_s": None, "members": [
        {"k": 0, "template": "T0", "before_clock_s": 6.5, "verifier_ahead_s": 3.0, "e2e_s": 10.0, "before_first_check_s": 3.25,
         "check_s": 0.75},
        {"k": 1, "template": "T1", "before_clock_s": 2.0, "verifier_ahead_s": 1.0, "e2e_s": 6.0, "before_first_check_s": None,
         "check_s": None}]}
    assert [m["k"] for m in json.loads((audit / "member-seconds.json").read_text())["members"]] == [0, 1]
    assert MS.main(["measure", str(tmp_path / "stamped"), "--stamped", str(d), "--no-alone"]) == 1
    assert "no audit under" in capsys.readouterr().err


def test_a_stamp_keeps_an_empty_argument_in_its_place():
    (got,) = MS.stamps("flock-verify 7 3 1700000000.250000 verify --circuit  --public k\nnot a stamp\n")
    assert got == {"program": "flock-verify", "pid": 7, "ppid": 3, "ns": 1_700_000_000_250_000_000,
                   "argv": ["verify", "--circuit", "", "--public", "k"]}
    assert MS.flag(got["argv"], "--circuit") == "" and MS.flag(got["argv"], "--public") == "k" and MS.flag(got["argv"], "--zk") is None


def test_a_wrapper_stamps_its_start_and_execs_its_binary(tmp_path):
    real = []
    for name in MS.PROGRAMS:
        f = tmp_path / "bin" / name
        f.parent.mkdir(exist_ok=True)
        f.write_text(f'#!/bin/bash\necho "{name} $$ $*"\n')
        f.chmod(0o755)
        real.append(str(f))
    wrapped = MS.wrap(tmp_path / "stamped", real)
    assert [w.name for w in wrapped] == list(MS.PROGRAMS)
    assert json.loads((tmp_path / "stamped" / MS.REAL).read_text()) == dict(zip(MS.PROGRAMS, real))
    for w in wrapped:
        p = subprocess.run([str(w), "verify", "--circuit", "", "--zk"], capture_output=True, text=True, check=True)
        name, pid, rest = p.stdout.rstrip("\n").split(" ", 2)
        got = MS.stamps((tmp_path / "stamped" / MS.STAMPS).read_text())[-1]
        # the binary ran with its arguments under the pid the wrapper stamped
        assert (name, rest) == (w.name, "verify --circuit  --zk")
        assert (got["program"], got["pid"], got["ppid"], got["argv"]) == (w.name, int(pid), os.getpid(), ["verify", "--circuit", "", "--zk"])
        assert got["ns"] > 0
    assert len(MS.stamps((tmp_path / "stamped" / MS.STAMPS).read_text())) == 3


STAND_IN = r"""#!/bin/bash
echo "{name} $*" >> {log}
case "$1" in
  statement) echo "statement unit/t instances=8 vus_per_block=1 blocks=8 k_log=10 nbl=1 m=3 regions=1 delta=1 built in 0.25s" >&2;;
  session)
    if [[ " $* " == *" --schedule "* ]]; then cat > /dev/null; echo "ok 3"
    elif [[ " $* " == *" --program "* ]]; then echo "LISTENING 127.0.0.1:1"; echo "FRAMES 127.0.0.1:2"; cat > /dev/null
    else echo "refused: no program"; fi;;
esac
"""


def test_each_reader_alone_is_the_members_command_on_its_files(tmp_path, monkeypatch):
    audit, d = plant(tmp_path)
    log, real = tmp_path / "ran.txt", {}
    for name in MS.PROGRAMS:
        f = tmp_path / "bin" / name
        f.parent.mkdir(exist_ok=True)
        f.write_text(STAND_IN.format(name=name, log=log))
        f.chmod(0o755)
        real[name] = str(f)
    ticks = iter(range(10**6))
    monkeypatch.setattr(MS, "clock", lambda: next(ticks) * 0.5)
    all_stamps = MS.stamps((d / MS.STAMPS).read_text())
    got = MS.measure(audit, all_stamps)
    assert MS.readers(audit, got, all_stamps, real, 2)
    m0, m1 = (m["alone"] for m in got["members"])
    half = {"s": 0.5, "each": [0.5, 0.5]}
    assert m0["verifier_load"] == m0["prover_load"] == {**half, "built_s": 0.25}
    assert m0["loop_start"] == m0["loop_setup_at_sample"] == half
    # a reader that fails is its error
    assert list(m0["loop_start_no_window"]) == ["error"] and "refused: no program" in m0["loop_start_no_window"]["error"]
    assert m0["archive_put"] == {x: half for x in ("none", "circuit.txt", "pub-8.bin", "t.u32", "partition.json", "program.json",
                                                   "registered.json")}
    # member 1 ran no Lean loop, and linked its circuit rather than put it: the circuit is still put alone
    assert sorted(m1) == ["archive_put", "prover_load", "verifier_load"] and sorted(m1["archive_put"]) == ["circuit.txt", "none", "pub-4.bin"]
    m = audit / "m0"
    ran = log.read_text().splitlines()
    state = f"--circuit {m / 'circuit.txt'} --tables {m / 'tables'}"
    loop = (f"flock-firewall session --statement verity/flock-circuit --circuit {m / 'circuit.txt'} --public {m / 'pub-8.bin'} "
            f"--tables {m / 'tables'} --session-tables 1")
    window = f"--partition {audit / 'partition.json'} --program {audit / 'program.json'}"
    assert ran[:4] == [f"flock-circuit statement --zk --instances {m / 'pub-8.bin'} {state} --pin abc"] * 2 + [
        f"flock-circuit statement --zk --instances {m / 'inst-8.bin'} {state}"] * 2
    assert ran[4:9] == [f"{loop} --verifier 127.0.0.1:9 {window}"] * 2 + [f"{loop} --verifier 127.0.0.1:9"] + [f"{loop} {window} --schedule"] * 2
    puts = [x.split(" ") for x in ran[9:23]]
    assert all(x[:2] == ["flock-verify", "archive-put"] and not Path(x[2]).exists() for x in puts)
    assert [x[3:] for x in puts] == [y for f in ([], [str(m / "circuit.txt")], [str(m / "pub-8.bin")], [str(m / "tables" / "t.u32")],
                                                 [str(audit / "partition.json")], [str(audit / "program.json")],
                                                 [str(m / "registered.json")]) for y in (f, f)]
    # a reader whose command can't be run at all is the member's error, and the audit's own seconds stay
    assert not MS.readers(audit, got, all_stamps, {}, 1)
    assert [list(m["alone"]) for m in got["members"]] == [["error"], ["error"]] and got["members"][0]["session"]["before_clock_s"] == 6.5
