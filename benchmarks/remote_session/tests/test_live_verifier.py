"""PoUB's verifier host (`verity/protocols/poub/device/live_verifier/host.py`) on what this directory holds, with the
gate's stand-ins of its own tests (`gate_harness.py`): a bound run's command line, which verifier.py binds to a
synthetic checkpoint (`store_layout.synthetic`); the prover's tunnel key, whose client is `run.sh`'s; `auditor_line.py`
on the host's python3; and the slow test, a run through the stub verifier (`stub_verifier.py`, a real paced session
over `test_session.run`): its out/ is verifier.py's, which `auditor_line.py` reads, `fetch` ships it and `replay`
replays it through both verifiers."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
RS = HERE.parent
LIVE = HERE.parents[2] / "verity" / "protocols" / "poub" / "device" / "live_verifier"
sys.path.insert(0, str(RS))
sys.path.insert(0, str(LIVE / "tests"))

import store_layout as SL  # noqa: E402
from gate_harness import (BASE, BOUND, KEY, PACKAGING, PIN, ROLE, ROOT, RUN, T1, H, Units, V, call,  # noqa: E402
                          host_at, parsed)


@pytest.fixture
def svc(tmp_path) -> H.Host:
    return host_at(tmp_path)


@pytest.fixture
def tree(svc, tmp_path) -> Path:
    """A pinned tree whose core pin (`models.json`'s shape) pins ROLE as the checkpoint infra provisioned in the host's
    models dir: a small synthetic BF16 one (`store_layout.synthetic`, an arena of 6 weights blocks)."""
    ckpt = svc.root / "models" / ROLE
    pin = SL.synthetic(ckpt, name=ROLE, layers=1, hidden=16, heads=2, kv_heads=1, intermediate=32, vocab=64)
    (ckpt / "pin.json").unlink()
    t = tmp_path / "tree"
    (t / H.MODEL_PINS).parent.mkdir(parents=True)
    (t / H.MODEL_PINS).write_text(json.dumps({ROLE: {
        "checkpoint_sha512": pin["checkpoint_sha512"], "files": pin["checkpoint"], "recipe": "bf16/v1",
        "repo": "example/synthetic-bf16", "revision": "0" * 40}}, indent=1, sort_keys=True) + "\n")
    return t


def test_a_bound_runs_command_line_names_its_roles_checkpoint_and_cores_pin_which_verifier_py_binds_to(
        svc, tree, monkeypatch, capsys, tmp_path):
    """`--model ROLE` names no path: the gate adds the role's directory in the host's models dir as `--checkpoint` and
    the pinned tree's `models.json` as verifier.py's `--pin`, apart from the gate's own `--pin` (the challenger's key,
    `--challenger-pin`); the rest is the unbound run's command line, and verifier.py binds the store to that pin before
    it listens."""
    rc, out = call(svc, BOUND, tree=tree)
    doc = json.loads(out)
    assert rc == 0, doc
    req = json.loads((svc.runs / RUN / "request.json").read_text())
    assert (req["model"], req["engines"], req["arena-blocks"], req["pin"]) == (ROLE, 2, 7, KEY)
    argv = req["argv"]
    assert argv == doc["argv"]
    i = argv.index("--model")
    assert argv[i:] == ["--model", ROLE, "--checkpoint", str(svc.root / "models" / ROLE), "--pin",
                        str(tree / H.MODEL_PINS), "--engines", "2", "--arena-blocks", "7",
                        "--paced", "2:8000:40:2", "4:16000:40:2"]
    assert argv.count("--pin") == 1 and argv[argv.index("--challenger-pin") + 1] == KEY
    unbound = {k: v for k, v in req.items() if k not in ("model", "engines", "arena-blocks", "argv")}
    assert argv[:i] + argv[i + 10:] == H.verifier_argv(svc, RUN, unbound, svc.settings(), tree=tree)

    monkeypatch.setattr(V.os, "sched_getaffinity", lambda pid: {0})
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    a, listened = parsed(monkeypatch, argv)
    assert listened == ("127.0.0.1", 7951)
    assert (a.model, a.checkpoint, a.pin, a.engines, a.arena_blocks, a.challenger_pin, a.layout_workers) == (
        ROLE, svc.root / "models" / ROLE, tree / H.MODEL_PINS, 2, 7, KEY, 1)
    pinned = json.loads((tree / H.MODEL_PINS).read_text())[ROLE]["checkpoint_sha512"]
    assert f"store: {ROLE} pinned ({pinned[:16]}...), " in capsys.readouterr().out


def test_a_tunnel_key_gets_a_pty_only_while_the_challengers_pinned_client_forces_one():
    """ssh ends a session whose forced pty (`-tt`) the server refuses (OpenSSH 9.6, checked against a real sshd), so the
    prover's key, whose client is run.sh's `ssh -T`, has no `pty`, and the challenger's keeps it only while serve.py
    forces one."""
    setup = (PACKAGING / "setup.sh").read_text()
    assert "opts=restrict,port-forwarding\n" in setup and '[ "$acct" = prover ] || opts+=,pty\n' in setup
    verifier_host = (RS / "run.sh").read_text().split('[ -n "${VERIFIER_SSH:-}" ]; then')[1].split("\nelif ")[0]
    assert re.search(r"\bssh -T\b", verifier_host) and "-tt" not in verifier_host
    serve = (ROOT / "verity" / "composite" / "device" / "live_challenger" / "serve.py").read_text()
    assert re.search(r'^SSH = \["ssh", "-tt"', serve, re.M), (
        "serve.py no longer forces a pty: drop `pty` from the challenger's tunnel key in verifier/setup.sh, and this check")


def test_auditor_line_runs_from_the_hosts_tree_on_a_python_with_no_site_packages(tmp_path):
    """The host runs its code with its own python3 (`-I`), where neither `verity` nor any other package is installed:
    `auditor_line.py`, at its path in the pinned tree, resolves each import within it."""
    code = ("sys.argv = ['auditor_line.py', '--help']; "
            "runpy.run_path(str(host.TREE / 'benchmarks' / 'remote_session' / 'auditor_line.py'), run_name='__main__')")
    p = subprocess.run([sys.executable, "-I", "-S", "-c", f"import runpy, sys; sys.path.insert(0, {str(LIVE)!r}); "
                        f"import host; {code}"], capture_output=True, text=True, cwd=tmp_path)
    assert p.returncode == 0, p.stderr[-2000:]
    assert "site-packages" not in p.stdout


def names(tar: tarfile.TarFile) -> set[str]:
    return {m.name for m in tar.getmembers() if m.isfile()}


@pytest.mark.slow
def test_a_run_through_the_stub_verifier_leaves_verifier_pys_record_which_fetch_ships_and_replay_replays(
        svc, tmp_path):
    """The one test that runs a session through the gate (with the Lean verdict, so it builds the Lean verifier and the
    difftest): what the cheaper tests can't show is that the runner's out/ is the one `auditor_line.py` and `replay.py`
    read, from a real transcript, gzipped, summed and shipped."""
    units = Units()
    rc, out = call(svc, [*BASE, "--paced", "2:8:40:2"], units)
    assert rc == 0, out
    req = json.loads((svc.runs / RUN / "request.json").read_text())
    end = H.execute(svc, RUN, pinned=PIN, now=lambda: T1,
                    argv=[sys.executable, str(HERE / "stub_verifier.py"), *req["argv"][3:]])
    d, o = svc.runs / RUN, svc.runs / RUN / "out"
    assert end["rc"] == 0, (o / "verifier.log").read_text()
    assert sorted(p.name for p in o.iterdir()) == [
        "challenger", "rc.txt", "registry.json", "transcripts", "verifier.json", "verifier.log"]
    assert not (svc.scratch / RUN).exists()
    v = json.loads((o / "verifier.json").read_text())
    (s,) = v["sessions"]
    t = s["transcript"]
    name = Path(t["path"]).name
    assert s["accepted"] and Path(t["path"]).parent == svc.scratch / RUN / "transcripts"
    raw = gzip.decompress((o / "transcripts" / f"{name}.gz").read_bytes())
    assert (len(raw), hashlib.sha256(raw).hexdigest()) == (t["bytes"], t["sha256"])
    assert (o / "transcripts" / "SHA256SUMS").read_text() == (
        f"{hashlib.sha256((o / 'transcripts' / f'{name}.gz').read_bytes()).hexdigest()}  ./{name}.gz\n")
    assert end["transcripts"] == [{"name": name, "bytes": t["bytes"]}]
    assert end["files"] == {str(p.relative_to(d)): H.digest_of(p) for p in sorted(o.rglob("*")) if p.is_file()}
    assert json.loads((o / "registry.json").read_text()) == v["registry"] == json.loads(svc.registry.read_text())
    units.live.clear()
    rc, st = call(svc, ["status", RUN], units, key="-")
    st = json.loads(st)
    assert rc == 0 and st["listening"] and not st["active"] and st["end"]["rc"] == 0 and st["replay"] is None

    import auditor_line
    line = auditor_line.analyse(o, host_bytes=1843153879040, pin=None, period=None, heads=None, peer="auditor-host")
    assert line["registration"] == v["registration"]["digest"]
    (row,) = line["sessions"]
    assert (row["k"], row["accepted"]) == (0, True) and row["skipped"].startswith("no challenger log: coins ")

    rc, shipped = call(svc, ["fetch", RUN], units, key="-")
    assert rc == 0
    with tarfile.open(fileobj=io.BytesIO(shipped), mode="r:gz") as tar:
        light = names(tar)
    assert {f"{RUN}/request.json", f"{RUN}/end.json", f"{RUN}/out/verifier.json", f"{RUN}/out/transcripts/SHA256SUMS",
            f"{RUN}/out/transcripts/transcripts.json"} <= light
    assert not any(n.endswith(".json.gz") for n in light)
    rc, shipped = call(svc, ["fetch", RUN, "--transcripts"], units, key="-")
    assert rc == 0
    with tarfile.open(fileobj=io.BytesIO(shipped), mode="r:") as tar:
        assert names(tar) == light | {f"{RUN}/out/transcripts/{name}.gz"}
        tar.extractall(tmp_path / "fetched", filter="data")
    assert all((tmp_path / "fetched" / n).read_bytes() == (svc.runs / n).read_bytes() for n in light)

    assert call(svc, ["replay", RUN], units)[0] == 0
    replayed = H.replay_run(svc, RUN, now=lambda: T1)
    assert (replayed["rc"], replayed["kind"], replayed["sums_match"], replayed["gz_sums"]) == (
        0, "transcript", True, {f"{name}.gz": True}), (d / "replay" / "replay.log").read_text()
    rows = json.loads((d / "replay" / "replay.json").read_text())
    assert rows and all(r["agree"] and r["held"] and r["recorded"] == "accepted" for r in rows), rows
    assert not (svc.scratch / f"{RUN}-replay").exists()

    gz = o / "transcripts" / f"{name}.gz"
    gz.write_bytes(gzip.compress(raw.replace(b'"events"', b'"events" ', 1)))
    tampered = H.replay_run(svc, RUN, now=lambda: T1)
    assert tampered["rc"] == 1 and tampered["gz_sums"] == {f"{name}.gz": False} and not tampered["sums_match"]
