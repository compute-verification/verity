"""Run custody on the remote (store.custody, `research run --on --custody-r2`): the machine publishes the attempt with every run file
(logs included) under a minted key, the pod guard, `pods drain` and the reaper accept that custody, and any machine can check it.
The machine is a local directory; the remote is an `fs` vault."""
from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from research import cli, notes, remote
from research.pods import drain as DR
from research.store import credentials as C
from research.store import custody as CU
from research.store import preserved as P
from research.store.cli import main as data_main
from research.store.local import LocalStore, object_key
from research.store.model import Label

from test_pod_guard_daemon import QUIET, _env, idle_since
from waits import Gate, wait_runner

PY = sys.executable
SRC = str(Path(remote.__file__).resolve().parents[1])
FIXTURES = Path(__file__).parent / "fixtures"
RID = "r20260925-010000-c0de"
FIXTURE_RID = "r20260922-120000-abcd"
SECRET = "session-token-that-must-not-leak"


def _custody_dir(tmp_path: Path, run_id: str, vault: Path, remote_toml: str | None = None) -> Path:
    d = tmp_path / "root" / "requests" / run_id / remote.CUSTODY_DIR
    d.mkdir(parents=True)
    (d / CU.STORE_TOML).write_text(remote_toml or f'[remote]\ntype = "fs"\npath = "{vault}"\n')
    (d / CU.CRED_FILE).write_text(json.dumps({"AWS_ACCESS_KEY_ID": "AKfake", "AWS_SECRET_ACCESS_KEY": "sk-fake", "AWS_SESSION_TOKEN": SECRET,
                                              "expires_utc": "2026-09-25T07:00:00Z"}))
    return d


def _request(tmp_path: Path, run_id: str, command: list[str], vault: Path) -> tuple[Path, Path]:
    root = tmp_path / "root"
    _custody_dir(tmp_path, run_id, vault)
    req = {"schema": remote.REQUEST_SCHEMA, "run_id": run_id, "runs_root": str(root / "runs"), "tool": SRC, "python": PY, "cwd": None,
           "env": {}, "harness_args": ["--stage", "t", "--no-sampler"], "command": command, "exclusive": False,
           "cost": {"provider": "runpod", "pod_id": "p1", "rate_usd_per_hour": 0.06, "rate_observed_utc": "t0", "currency": "USD", "reason": None},
           "remote": {"machine": "m", "pod_id": "p1"}, "custody": {"schema": CU.SCHEMA, "expires_utc": "2026-09-25T07:00:00Z"}}
    p = root / "requests" / run_id / "request.json"
    p.write_text(json.dumps(req))
    return p, root / "runs"


def _wait_marker(run_dir: Path) -> dict:
    wait_runner(run_dir)
    assert (run_dir / CU.MARKER).exists(), (f"no {CU.MARKER} in {run_dir}: {sorted(p.name for p in run_dir.iterdir())}; launcher.log: "
                                            f"{(run_dir / 'launcher.log').read_text()[-2000:] if (run_dir / 'launcher.log').exists() else '-'}")
    return json.loads((run_dir / CU.MARKER).read_text())


def _other_machine(tmp_path: Path, vault: Path, monkeypatch) -> LocalStore:
    """A fresh store (another machine: nothing local) whose remote is the same vault."""
    cfg = tmp_path / "other.toml"
    cfg.write_text(f'[remote]\ntype = "fs"\npath = "{vault}"\n')
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(cfg))
    return LocalStore(tmp_path / "other-store")


def _fixture_run(runs: Path, run_id: str = FIXTURE_RID) -> Path:
    d = runs / run_id
    shutil.copytree(FIXTURES / "run_v0.2", d)
    (d / "state.json").rename(d / "status.json")
    job = json.loads((d / "job.json").read_text())
    job["run"]["id"] = run_id
    (d / "job.json").write_text(json.dumps(job))
    (d / "stdout.log").write_text("the workload's stdout\n")
    (d / "stderr.log").write_text("the workload's stderr\n")
    return d


# ---- the machine side, end to end --------------------------------------------------------------------------------------------------------

def test_the_machine_publishes_every_run_file_and_any_machine_sees_custody(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    cmd = [PY, "-c", "import os; print('hello from the workload'); print('token seen:', 'AWS_SESSION_TOKEN' in os.environ); "
                     "print('run dir:', os.environ['RESEARCH_RUN_DIR'] == os.getcwd())"]
    req, runs = _request(tmp_path, RID, cmd, vault)
    out = remote.launch_request(req)
    assert out["started"], out
    d = runs / RID
    marker = _wait_marker(d)
    assert marker["preserved"] is True, marker
    assert (d / "stdout.log").read_text().splitlines() == ["hello from the workload", "token seen: False", "run dir: True"], "the key is never the workload's"
    assert not (req.parent / remote.CUSTODY_DIR / CU.CRED_FILE).exists(), "the key is deleted once used"
    assert SECRET not in (d / "launcher.log").read_text() + (d / "job.json").read_text() + req.read_text() + json.dumps(marker)
    assert {"stdout.log", "stderr.log", "launcher.log", "job.json", "status.json"} <= set(marker["files"])
    launch = json.loads((d / "launch.json").read_text())
    assert launch["argv"][launch["argv"].index("--custody-dir") + 1] == str(req.parent / remote.CUSTODY_DIR)
    # nothing touched the run dir after the marker: launcher.log in the record is all of it, and the marker vouches for the dir
    assert CU.marker_problem(d) is None
    assert "custody: publishing" in (d / "launcher.log").read_text()

    # another machine, nothing local: the attempt, its logs and the verdict come from the remote alone
    st2 = _other_machine(tmp_path, vault, monkeypatch)
    assert P.check(st2, [RID], mode="head")["preserved"]
    doc = st2.get_attempt(RID)
    rr = doc["outputs"]["run_record"]
    assert rr == marker["run_record"]
    m = st2.get_manifest(rr)
    assert m.kind == CU.KIND and {f.path for f in m.payload.files} == set(marker["files"])
    stdout = next(f for f in m.payload.files if f.path == "stdout.log")
    st2.remote.get(object_key(stdout.sha256), tmp_path / "got")
    assert (tmp_path / "got").read_text().startswith("hello from the workload")
    assert data_main(["preserved", RID]) == 0

    # the pod guard: idle, no .fetched, but custody on the remote -> it may terminate; a file changed afterwards -> it refuses
    root = tmp_path / "root"
    st = idle_since(root, _env(tmp_path, QUIET), 120)
    assert st["verdict"] == "terminate" and st["unfetched"] == [], st
    with (d / "stdout.log").open("a") as f:
        f.write("late line\n")
    assert "changed after custody" in CU.marker_problem(d)
    st = idle_since(root, _env(tmp_path, QUIET), 120)
    assert st["verdict"] == "refuse" and "stdout.log changed after its custody" in st["unfetched"][0]["why"]


def test_a_stray_sigterm_to_the_runner_does_not_cost_the_run_its_record_or_custody(tmp_path):
    # `pkill -f rows.sh` to stop a queue matched the runner's argv too (it carries the workload's command): the runner died, so the run
    # never recorded its end or published (vllm-rf-epoch #73)
    import signal

    vault = tmp_path / "vault"
    arrived, gate = tmp_path / "arrived", Gate(tmp_path / "gate")
    os.mkfifo(arrived)
    workload = f"import sys; open(sys.argv[2], 'w').close(); {Gate.WAIT}; print('finished')"   # arrives, then waits for the gate
    req, runs = _request(tmp_path, RID, [PY, "-c", workload, str(gate.path), str(arrived)], vault)
    try:
        assert remote.launch_request(req)["started"]
        d = runs / RID
        open(arrived, "rb").close()                              # returns once the workload runs: the runner is past its setup
        runner = json.loads((d / "launch.json").read_text())["runner_pid"]
        os.kill(runner, signal.SIGTERM)
        os.kill(runner, signal.SIGHUP)
        gate.release()
        marker = _wait_marker(d)
    finally:
        gate.close()
    status = json.loads((d / "status.json").read_text())
    assert status["state"] == "done" and status["rc"] == 0 and (d / "stdout.log").read_text() == "finished\n"
    assert marker["preserved"] is True and CU.marker_problem(d) is None
    log = (d / "launcher.log").read_text()
    assert "runner got SIGTERM" in log and "runner got SIGHUP" in log


def test_a_failed_publish_leaves_a_negative_marker_forgets_the_key_and_the_guard_refuses(tmp_path, capsys, monkeypatch):
    from research.store import remote_s3

    monkeypatch.setattr(remote_s3, "RETRY_BACKOFF_S", 0.0)
    runs = tmp_path / "root" / "runs"
    d = _fixture_run(runs)
    cdir = _custody_dir(tmp_path, FIXTURE_RID, tmp_path / "vault",
                        remote_toml='[remote]\ntype = "s3"\nbucket = "b"\nendpoint = "http://127.0.0.1:9"\nregion = "auto"\n'
                                    'access_key_id_env = "AWS_ACCESS_KEY_ID"\nsecret_access_key_env = "AWS_SECRET_ACCESS_KEY"\n')
    rep = CU.runner_publish(d, cdir, sleep=lambda s: None, tries=2)
    assert rep["preserved"] is False and rep["error"]
    assert not (cdir / CU.CRED_FILE).exists()
    assert json.loads((d / CU.MARKER).read_text())["preserved"] is False
    assert "custody NOT on the remote" in capsys.readouterr().err
    assert CU.marker_problem(d).startswith(".custody: not preserved")
    st = idle_since(tmp_path / "root", _env(tmp_path, QUIET), 120)
    assert st["verdict"] == "refuse" and "custody not on the remote" in st["unfetched"][0]["why"]


def test_on_the_machine_a_failed_push_leaves_launcher_log_as_recorded_and_names_an_expired_key(tmp_path):
    # the runner's stderr IS launcher.log: a retry notice or the failure line printed there after the record was taken made the run dir
    # differ from its record for good (x4-hopper-blake3: `data custody` refused even after a repair push); an expired key was a bare 403
    runs = tmp_path / "root" / "runs"
    d = _fixture_run(runs)
    (d / "launcher.log").write_text("research: custody: publishing the attempt and every run file\n")
    cdir = _custody_dir(tmp_path, FIXTURE_RID, tmp_path / "vault",
                        remote_toml='[remote]\ntype = "s3"\nbucket = "b"\nendpoint = "http://127.0.0.1:9"\nregion = "auto"\n'
                                    'access_key_id_env = "AWS_ACCESS_KEY_ID"\nsecret_access_key_env = "AWS_SECRET_ACCESS_KEY"\n')
    cred = json.loads((cdir / CU.CRED_FILE).read_text())
    (cdir / CU.CRED_FILE).write_text(json.dumps({**cred, "expires_utc": "2020-01-01T00:00:00Z"}))
    before = (d / "launcher.log").read_bytes()
    code = ("from pathlib import Path; from research.store import custody as CU, remote_s3 as R; R.RETRY_BACKOFF_S = 0; "
            f"CU.runner_publish(Path({str(d)!r}), Path({str(cdir)!r}), sleep=lambda s: None, tries=1)")
    with open(d / "launcher.log", "ab") as log:
        subprocess.run([PY, "-c", code], stdout=log, stderr=log, env={**os.environ, "PYTHONPATH": SRC}, check=True)
    assert (d / "launcher.log").read_bytes() == before
    marker = json.loads((d / CU.MARKER).read_text())
    assert marker["preserved"] is False and marker["error"].startswith("the custody key expired at 2020-01-01T00:00:00Z: the run outlived")
    assert "custody NOT on the remote" in (cdir / CU.PUBLISH_LOG).read_text()
    assert CU.marker_problem(d).startswith(".custody: not preserved")


def test_research_data_custody_publishes_a_local_run_and_verifies_it_elsewhere(tmp_path, monkeypatch, capsys):
    vault = tmp_path / "vault"
    cfg = tmp_path / "store.toml"
    cfg.write_text(f'[remote]\ntype = "fs"\npath = "{vault}"\n')
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(cfg))
    runs = tmp_path / "runs"
    d = _fixture_run(runs)
    assert data_main(["custody", str(d)]) == 1 and "not in the store" in capsys.readouterr().out
    assert data_main(["custody", FIXTURE_RID, "--runs-dir", str(runs), "--publish"]) == 0
    assert ".custody written" in capsys.readouterr().out
    assert CU.has_custody(d) == (True, "on the remote (.custody)")
    # the verdict from another machine: same answer; the marker there is local bookkeeping, the fact is the remote's
    st2 = _other_machine(tmp_path, vault, monkeypatch)
    rep = CU.verify(st2, d, mark=False)
    assert rep["preserved"] and "stdout.log" in rep["files"]


def test_an_attempt_published_without_a_run_record_is_not_run_custody(tmp_path, monkeypatch):
    cfg = tmp_path / "store.toml"
    cfg.write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "vault"}"\n')
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(cfg))
    d = _fixture_run(tmp_path / "runs")
    from research.store import attempt as A

    st = LocalStore()
    pub = A.publish(d, store=st)
    assert pub["published"] and pub["preserved"] and "run_record" not in pub["outputs"], "the default publish is unchanged"
    rep = CU.verify(st, d)
    assert not rep["preserved"] and "names no run record" in rep["error"] and not (d / CU.MARKER).exists()


# ---- the launcher -------------------------------------------------------------------------------------------------------------------------

PARENT = {"R2_ENDPOINT": "https://" + "0" * 32 + ".r2.cloudflarestorage.com", "R2_BUCKET": "bucket", "AWS_ACCESS_KEY_ID": "parent-id",
          "AWS_SECRET_ACCESS_KEY": "parent-secret"}


def test_the_minted_key_is_delete_free_and_staged_privately(tmp_path):
    public, files = CU.mint_for_machine("2h", environ=dict(PARENT))
    assert public["ttl_s"] == 7200 and public["via"] == "local"
    assert not {"DeleteObject", "DeleteObjects", "CopyObject"} & set(public["actions"])
    names = dict(files)
    cred = json.loads(names[f"{CU.DIR_NAME}/{CU.CRED_FILE}"])
    _, claims = C.decode_jwt(__import__("base64").b64decode(cred["AWS_SESSION_TOKEN"]).decode()[len("jwt/"):])
    assert claims["actions"] == list(CU.ACTIONS) and claims["exp"] - claims["iat"] == 7200 and "scope" not in claims
    toml = names[f"{CU.DIR_NAME}/{CU.STORE_TOML}"].decode()
    for secret in (cred["AWS_SECRET_ACCESS_KEY"], cred["AWS_SESSION_TOKEN"], PARENT["AWS_SECRET_ACCESS_KEY"]):
        assert secret not in toml and secret not in json.dumps(public)
    assert CU.DIR_NAME == remote.CUSTODY_DIR
    # staged beside the request, readable by the machine's user only; request.json carries no secret
    path = remote.stage_request(["sh", "-c"], str(tmp_path / "root"), RID, {"schema": remote.REQUEST_SCHEMA, "custody": public}, [], private=files)
    cf = tmp_path / "root" / "requests" / RID / CU.DIR_NAME / CU.CRED_FILE
    assert stat.S_IMODE(cf.stat().st_mode) == 0o600
    assert cred["AWS_SECRET_ACCESS_KEY"] not in Path(path).read_text()


def test_custody_r2_is_refused_when_this_machine_cannot_mint(tmp_path, monkeypatch):
    mf = tmp_path / "machines.toml"
    mf.write_text('[machines.m]\nprovider = "runpod"\npod_id = "abcdefghijkl12"\nproject = "verity"\n')
    monkeypatch.setenv("RESEARCH_MACHINES", str(mf))
    for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "CLOUDFLARE_API_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(remote, "pod_target", lambda m: pytest.fail("nothing is contacted before the key exists"))
    with pytest.raises(remote.Refused, match="cannot mint the machine's key here"):
        remote.launch(run_id=RID, local_dir=tmp_path / "runs" / RID, machine_name="m", project="verity", command=["true"], harness_args=[],
                      source=None, sends=[], cwd=None, env={}, exclusive=False, custody_ttl="6h")


def test_a_source_tree_that_predates_custody_is_refused(tmp_path):
    pkg = tmp_path / "tree" / "tools" / "research" / "src" / "research"
    (pkg / "store").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "remote.py").write_text("def tool_identity(): pass\n")
    assert remote.knows_custody(None) and remote.knows_custody(Path(SRC) / "research")
    assert not remote.knows_custody(pkg)


def test_custody_flags_need_on(capsys):
    assert cli.main(["run", "--custody-r2", "--", "true"]) == 2
    assert "--custody-r2 / --custody-ttl are for --on" in capsys.readouterr().err
    assert cli.main(["run", "--on", "m", "--project", "p", "--no-custody-r2", "--custody-ttl", "8h", "--", "true"]) == 2
    assert "--no-custody-r2 mints none" in capsys.readouterr().err


def test_custody_is_the_default_with_on_and_opting_out_is_explicit(tmp_path, monkeypatch, capsys):
    seen: list = []

    def launch(**kw):
        seen.append(kw["custody_ttl"])
        return {"phase": "launched", "pod_id": "p1", "remote_run_dir": "/r", "launch": {"runner_pid": 1}}
    monkeypatch.setattr(remote, "launch", launch)
    base = ["run", "--runs-dir", str(tmp_path / "runs"), "--on", "m", "--project", "p"]
    assert cli.main([*base, "--", "true"]) == 0
    assert cli.main([*base, "--custody-ttl", "12h", "--", "true"]) == 0
    assert cli.main([*base, "--custody-r2", "--", "true"]) == 0
    capsys.readouterr()
    assert cli.main([*base, "--no-custody-r2", "--", "true"]) == 0
    assert seen == ["6h", "12h", "6h", None]
    assert "--no-custody-r2: this run's records stay on the machine" in capsys.readouterr().err


def test_a_refused_claim_deletes_the_staged_key(tmp_path):
    req, runs = _request(tmp_path, RID, ["true"], tmp_path / "vault")
    doc = json.loads(req.read_text())
    doc["remote"]["source_sha"] = "f" * 40                    # no READY tree: refused
    req.write_text(json.dumps(doc))
    out = remote.launch_request(req)
    assert out["started"] is False and out["fault"] == "SourceNotReady"
    assert not (req.parent / remote.CUSTODY_DIR / CU.CRED_FILE).exists()


# ---- drain / the reaper from any machine ---------------------------------------------------------------------------------------------------

POD = "abcdefghijkl12"


NOW = 1_800_000_000.0


def _state(verdict="idle", unfetched=(), age_s=30.0):
    return lambda pod_id, root: ({"verdict": verdict, "unfetched": list(unfetched), "runs": 2, "updated_epoch": NOW - age_s}, "")


def test_pod_custody_needs_a_fresh_guard_verdict_over_every_run_dir(tmp_path):
    st = LocalStore(tmp_path / "s")
    ok, why, _ = DR.pod_custody(st, POD, refresh=False, probe=False, now=NOW, state=_state())
    assert ok and "2 run dir(s) with custody" in why
    ok, why, _ = DR.pod_custody(st, POD, refresh=False, probe=False, now=NOW, state=_state(unfetched=[{"run": RID, "why": "no .custody"}]))
    assert not ok and RID in why
    ok, why, _ = DR.pod_custody(st, POD, refresh=False, probe=False, now=NOW, state=_state(age_s=3600))
    assert not ok and "guard dead?" in why
    ok, why, _ = DR.pod_custody(st, POD, refresh=False, probe=False, now=NOW, state=lambda p, r: (None, "no guard state on the pod"))
    assert not ok and "unknown" in why
    ok, why, _ = DR.pod_custody(st, POD, refresh=True, probe=False, now=NOW, state=_state())
    assert not ok and "refresh" in why, "no remote: the catalog cannot be brought up to date, so custody is not claimed"


def test_drain_custody_r2_refuses_until_the_pod_has_custody(tmp_path):
    st = LocalStore(tmp_path / "s")
    killed: list[str] = []
    bad = lambda *a, **k: (False, "1 run dir(s) on it without custody", {"guard": None})  # noqa: E731
    rc, rep = DR.drain(st, POD, probe=False, terminate=killed.append, log_path=tmp_path / "d.log", custody_r2=True, pod_custody_fn=bad)
    assert rc == 1 and not killed and rep["unpreserved"] == ["custody: 1 run dir(s) on it without custody"]
    assert any("NOT preserved  custody:" in line for line in DR.format_report(rep))
    good = lambda *a, **k: (True, "ok", {})  # noqa: E731
    rc, rep = DR.drain(st, POD, probe=False, terminate=killed.append, log_path=tmp_path / "d.log", custody_r2=True, pod_custody_fn=good)
    assert rc == 0 and killed == [POD]


def test_the_reaper_takes_run_custody_from_the_remote_with_custody_r2(monkeypatch):
    seen = {}

    def fake(st, target, **kw):
        seen.update(target=target, **kw)
        return True, "3 attempt(s) preserved, 3 run dir(s) with custody (guard)", {}

    monkeypatch.setattr(DR, "pod_custody", fake)
    assert notes.run_custody_r2(POD) == (True, "3 attempt(s) preserved, 3 run dir(s) with custody (guard)")
    assert seen["target"] == POD and seen["refresh"] is True


# ---- triage (F2) ---------------------------------------------------------------------------------------------------------------------------

def test_triage_sorts_every_run_by_custody_state(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "store.toml"
    cfg.write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "vault"}"\n')
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(cfg))
    runs = tmp_path / "runs"
    st = LocalStore()
    from research.store import attempt as A

    r2 = _fixture_run(runs, "r20260925-010001-aaaa")
    assert data_main(["custody", str(r2), "--publish"]) == 0
    plain = _fixture_run(runs, "r20260925-010002-bbbb")
    assert A.publish(plain, store=st)["preserved"]
    fetched = _fixture_run(runs, "r20260925-010003-cccc")
    (fetched / "preserved.json").write_text("{}")
    never = runs / "r20260925-010004-dddd"
    never.mkdir()
    (never / "remote.json").write_text(json.dumps({"phase": "refused", "pod_id": POD}))
    lost = runs / "r20260925-010005-eeee"
    lost.mkdir()
    (lost / "remote.json").write_text(json.dumps({"phase": "launched", "pod_id": POD}))
    waived = _fixture_run(runs, "r20260925-010006-ffff")
    st.add_label(waived.name, "custody", "waived", "daniel")
    capsys.readouterr()
    states = {r["run"]: r["state"] for r in CU.triage(LocalStore(), runs)}
    assert states == {r2.name: "r2", plain.name: "attempt-no-record", fetched.name: "fetched", never.name: "never-ran",
                      lost.name: "no-custody", waived.name: "waived"}
    assert data_main(["custody", "--triage", "--runs-dir", str(runs)]) == 1
    out = capsys.readouterr().out
    assert "2 need action" in out and lost.name in out and plain.name in out


def test_a_launch_longer_than_the_default_custody_key_needs_an_explicit_ttl_capped_at_24h(capsys):
    assert cli._custody_ttl_problem(None, ["--timeout", "3600"]) is None
    assert "outlives the custody key's default lifetime (6h)" in cli._custody_ttl_problem(None, ["--timeout=30000"])
    assert cli._custody_ttl_problem("12h", ["--timeout", "30000"]) is None
    assert "over the 24 h cap" in cli._custody_ttl_problem("25h", [])
    assert cli.main(["run", "--on", "vy-nowhere", "--timeout", "30000", "--", "true"]) == 2
    assert "pass --custody-ttl" in capsys.readouterr().err


def test_custody_warns_before_pushing_a_run_bigger_than_the_warn_size(tmp_path, monkeypatch):
    (tmp_path / "out.bin").write_bytes(b"x" * 2048)
    assert CU.size_warning(tmp_path) is None
    monkeypatch.setattr(CU, "SIZE_WARN_BYTES", 1024)
    assert "no resume" in CU.size_warning(tmp_path)


# ---- declared outputs: custody uploads what the run declares, and lists the rest ---------------------------------------------------------

def _declare(d: Path, globs: list[str]) -> None:
    job = json.loads((d / "job.json").read_text())
    job.setdefault("execution", {})["declared_outputs"] = globs
    (d / "job.json").write_text(json.dumps(job))


def test_select_keeps_run_files_and_declared_outputs_and_trims_undeclared_past_the_cap(tmp_path):
    d = _fixture_run(tmp_path / "runs")
    shutil.rmtree(d / "out")
    (d / "res").mkdir()
    (d / "res" / "result.json").write_text("{}")
    (d / "res" / "deep").mkdir()
    (d / "res" / "deep" / "big.bin").write_bytes(b"x" * 100)
    (d / "a-small.txt").write_bytes(b"y" * 10)
    (d / "z-junk.bin").write_bytes(b"z" * 100)
    _declare(d, ["res/*"])
    kept, trimmed = CU.select(d, cap=50)
    assert {"stdout.log", "job.json", "res/result.json", "res/deep/big.bin", "a-small.txt"} <= set(kept)
    assert [t["path"] for t in trimmed] == ["z-junk.bin"] and trimmed[0]["bytes"] == 100 and len(trimmed[0]["sha256"]) == 64
    assert [t["path"] for t in CU.select(d, cap=0)[1]] == ["a-small.txt", "z-junk.bin"], "cap 0: never upload undeclared"
    _declare(d, ["*"])
    assert CU.select(d, cap=0)[1] == [], "`*` keeps everything"
    w = CU.trim_warning(trimmed)
    assert "--declared-output" in w and "will be dropped once lanes declare their outputs" in w and CU.SKILL in w


def test_trimmed_files_are_listed_not_uploaded_and_custody_still_verifies(tmp_path, monkeypatch, capsys):
    vault = tmp_path / "vault"
    cfg = tmp_path / "store.toml"
    cfg.write_text(f'[remote]\ntype = "fs"\npath = "{vault}"\n')
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(cfg))
    monkeypatch.setattr(CU, "UNDECLARED_CAP_BYTES", 0)
    runs = tmp_path / "runs"
    d = _fixture_run(runs)
    (d / "res").mkdir()
    (d / "res" / "result.json").write_text('{"ok": true}')
    (d / "junk.bin").write_bytes(b"j" * 1000)
    _declare(d, ["res/*"])
    assert data_main(["custody", FIXTURE_RID, "--runs-dir", str(runs), "--publish"]) == 0
    assert "custody trimmed 3 undeclared file(s)" in capsys.readouterr().err   # junk.bin and the fixture's out/
    assert CU.has_custody(d) == (True, "on the remote (.custody)")
    st2 = _other_machine(tmp_path, vault, monkeypatch)
    m = st2.get_manifest(st2.get_attempt(FIXTURE_RID)["outputs"]["run_record"])
    paths = {f.path for f in m.payload.files}
    assert "res/result.json" in paths and "junk.bin" not in paths
    assert [t["path"] for t in m.meta["trimmed"]] == ["junk.bin", "out/proof.bin", "out/verify.jsonl"]
    assert CU.verify(st2, d, mark=False)["preserved"]
    (d / "junk.bin").write_bytes(b"k" * 1000)
    assert "changed" in CU.verify(st2, d, mark=False)["error"], "a trimmed file is still pinned by its hash"
