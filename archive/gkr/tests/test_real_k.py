"""A-GKR at vLLM's reduction lengths (route (a) on ``bf16-ampere-k<K>+blake3``): the pins are the K = 1536 circuit files at
``steps`` = K / 16 under relation names that carry K, the lowering admits K = 2048 and 8192, the verifier builds its own statement
of an input set (the committed identity B-Ligero binds), and ``tools/cell.py finish`` turns a sweep into the plateau's cell."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import blake3
import pytest

GKR = Path(__file__).resolve().parents[1]
REPO = GKR.parents[1]
sys.path.insert(0, str(GKR / "tools"))
SLICES = REPO / "benchmarks" / "numerical" / "tests" / "bench" / "data" / "captured-101" / "slices"
REAL_KS = (2048, 8192)


def pins() -> dict[str, list[tuple[int, tuple[str, ...]]]]:
    out: dict[str, list] = {}
    for line in (GKR / "verifier" / "pins.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            rel, steps, *rest = line.split()
            out.setdefault(rel, []).append((int(steps), tuple(rest[:3])))
    return out


@pytest.mark.parametrize("base", ["bf16-ampere", "bf16-ampere+blake3"])
def test_real_k_pins_are_the_k1536_circuit_files_at_k_over_16_steps(base):
    p = pins()
    ((steps, files),) = p[base]
    assert steps == 1536 // 16
    for K in REAL_KS:
        assert p[base.replace("bf16-ampere", f"bf16-ampere-k{K}")] == [(K // 16, files)]


def test_the_lowering_names_k_in_the_relation():
    from verity_numerical.bench import lowerings as L
    from verity_numerical.bench import templates as TM

    reg = L.registry("A-GKR")
    for K, rel in ((1536, "bf16-ampere+blake3"), (2048, "bf16-ampere-k2048+blake3"), (8192, "bf16-ampere-k8192+blake3")):
        sub = TM.subcircuit("gemm-coordinate", K=K)
        assert reg.supports(sub, "frame-v3/blake3-keyed") is None and reg.relation(sub, "frame-v3/blake3-keyed") == rel
    assert "pinned at K" in reg.supports(TM.subcircuit("gemm-coordinate", K=4096), "frame-v3/blake3-keyed")
    assert "sha-256" in reg.supports(TM.subcircuit("gemm-coordinate", K=2048), "frame-v3/sha-256")
    assert reg.stage(TM.subcircuit("gemm-coordinate", K=2048)) is not None


def _row_key(role: bytes) -> bytes:
    return (b"verity/blake3-leaf/v1/" + role).ljust(32, b"\0")


@pytest.mark.parametrize("K", REAL_KS)
def test_the_verifiers_own_statement_of_an_input_set(K, tmp_path):
    from verity_numerical.bench.input_sets import InputSet

    s = InputSet.open(SLICES / f"gemm-coordinate-ampere-bf16-k{K}")
    cell = tmp_path / "c"
    r = subprocess.run([sys.executable, str(GKR / "tools" / "cell.py"), "statement", "--cell", str(cell), "--input-set", str(s.path),
                        "--vus", str(s.n), "--verifier-side"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    meta = json.loads((cell / "cell.json").read_text())
    assert (meta["relation"], meta["K"], meta["vus"]) == (f"bf16-ampere-k{K}+blake3", K, s.n)
    assert meta["instances"]["content_digest"] == s.content_digest and meta["instances"]["range"] == [0, s.n]
    text = (cell / "statement" / "commitment.txt").read_text()
    assert f"\nK {K}\n" in text and f"\nvus {s.n}\n" in text and f"instances vllm-vu-set/v1:{s.name} {s.name} {s.manifest_sha256()}\n" in text
    ins = s.instances(0, s.n)
    want = [blake3.blake3(ins[p][v].astype("<u2").tobytes(), key=_row_key(role)).digest()
            for p, role in (("x", b"x"), ("w", b"w")) for v in range(s.n)]
    assert (cell / "leaf_digests.bin").read_bytes() == b"".join(want)
    assert not (cell / "operands.bin").exists() and not (cell / "statement" / "circuit.txt").exists()


def test_the_gate_replays_flock_at_the_statements_k(tmp_path):
    """``cell_gate.py --flock`` builds flock-link's statement at sigma.txt's K (without ``--k`` flock-link builds K = 1536's)."""
    ses = tmp_path / "session"
    ses.mkdir()
    (ses / "session.json").write_text("{}")
    sigma = tmp_path / "sigma.txt"
    sigma.write_text("verity/link-L/v2\nscheme blake3-keyed/row/v2\nvus 4\nk 8192\nbits 16\nm 21\n")
    fake = tmp_path / "flock-link"
    fake.write_text(f"#!{sys.executable}\nimport json, sys\nprint('REPLAY\\t' + json.dumps({{'accepted': True, 'why': sys.argv[1:]}}))\n")
    fake.chmod(0o755)
    out = tmp_path / "gate"
    subprocess.run([sys.executable, str(GKR / "tools" / "cell_gate.py"), "--session", str(ses), "--sigma", str(sigma), "--producer", "p",
                    "--out", str(out), "--flock", str(fake), "--digests", str(sigma), "--prime-commitment", "ab" * 32, "--vus", "4"],
                   capture_output=True, text=True)
    doc = json.loads((out / "gate.json").read_text())
    assert doc["checks"]["flock_replay"]["ok"] and "'--k', '8192'" in doc["checks"]["flock_replay"]["detail"]


def _point(root: Path, i: int, vus: int, t_total: float | None, why: str = "ok", verifier: str = "10.0.0.2:17200") -> Path:
    d = root / f"p{i}-{vus}"
    d.mkdir(parents=True)
    (d / "status.json").write_text(json.dumps({"vus": vus, "rc": 0 if t_total else 1, "why": why, "verifier": verifier}))
    if t_total is None:
        return d
    sessions = [{"session": j, "warmup": j == 0, "t_total": t_total} for j in range(6)]
    doc = {"kind": "bench-result/v1", "validation": {"status": "passed", "evidence": {"sessions": sessions}},
           "measurements": [{"name": "t.total", "value": t_total, "unit": "s"}, {"name": "commit.serving_seconds", "value": 9.5, "unit": "s"}],
           "workload_fingerprint": {"B": vus, "K": 2048, "relation": "bf16-ampere-k2048+blake3", "verifier": {"prime": "..."},
                                    "interaction": {"rtt_method": "median of 20 TCP connects"}}}
    (d / "result.json").write_text(json.dumps(doc))
    (d / "loopback-times.json").write_text(json.dumps({"verifier": "127.0.0.1:7300", "t_flock_prove_s": 2.0, "t_prime_prove_s": 1.0,
                                                       "net": {"rounds": 4000, "prime_wait_before_root_f_s": 0.3, "flock_wait_s": 0.1}}))
    (d / "contention.json").write_text(json.dumps({"contended": True, "reasons": ["CPU throttled in 23% of the cgroup's periods while the GPU was busy"],
                                                   "other_gpu_procs": 0}))
    return d


def test_finish_takes_the_plateau_by_memory_cap(tmp_path):
    import cell

    pts = [_point(tmp_path, 0, 1024, 5.0), _point(tmp_path, 1, 2048, 8.0), _point(tmp_path, 2, 4096, 15.0),
           _point(tmp_path, 3, 6272, None, why="torch.OutOfMemoryError: CUDA out of memory")]
    best, others = cell.finish(pts, "10.0.0.2:17200", "agkr-real-k", "sweep-test")
    fp = best["workload_fingerprint"]
    assert fp["B"] == 4096 and [d["workload_fingerprint"]["B"] for d in others] == [1024, 2048]
    sw = best["sweep"]
    assert sw["plateau"] and sw["total_vus"] == 4096 and sw["stopped"].startswith("6272 VUs failed (rc 1: torch.OutOfMemoryError")
    assert sw["points"] == [[1024, 1024 / 5.0], [2048, 256.0], [4096, 4096 / 15.0], [6272, None]]
    assert not others[0]["sweep"]["plateau"]
    pr = best["protocol"]
    assert pr["warm"] and pr["runs"] == 5 and pr["contended"] is False and pr["contention"]["reasons"]
    m = {x["name"]: x["value"] for x in best["measurements"]}
    assert m["commit.seconds"] == 9.5 and m["live.loopback_round_seconds"] == pytest.approx(0.4 / 4000)
    assert fp["commit"]["in_t_total"] is False and fp["verifier"]["placement"] == "10.0.0.2:17200"
    assert best["interaction"]["loopback_method"] and best["interaction"]["rtt_method"] and best["interaction"] is fp["interaction"]


def test_finish_counts_an_external_gpu_process_as_contention(tmp_path):
    import cell

    d = _point(tmp_path, 0, 1024, 5.0)
    (d / "contention.json").write_text(json.dumps({"contended": True, "reasons": ["1 other GPU compute process (pids seen: 4711)"]}))
    best, _ = cell.finish([d], "10.0.0.2:17200", "agkr-real-k", "sweep-test")
    assert best["protocol"]["contended"] is True
