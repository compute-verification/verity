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
from research import runs as R
from research.pods import drain as DR
from research.store import credentials as C
from research.store import custody as CU
from research.store import preserved as P
from research.store.cli import main as data_main
from research.store.local import LocalStore
from research.store.model import Label
from research.telemetry.job import set_status

from test_pod_guard_daemon import QUIET, _env, idle_since
from waits import wait_runner

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

URLS = '{"k": "https://r2.example/objects/sha256/k?X-Amz-Signature=sent-url-that-must-not-leak"}'


def _no_url_anywhere(vault: Path) -> None:
    assert not any(b"sent-url-that-must-not-leak" in p.read_bytes() for p in vault.rglob("*") if p.is_file()), "a URL reached the remote"


@pytest.mark.parametrize("command", [
    ["true"],                                                                       # what the hourly preflight probe runs
    [PY, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGTERM)"],          # a check cancelled while slot.py waits
], ids=["probe", "cancelled-before-check"])
def test_a_sent_credential_is_deleted_before_the_record_is_taken_whatever_the_workload(tmp_path, command):
    """Only `check.py` takes the Lean bundles' URLs out of inputs/: a run of anything else, or a check that ends before `check.py`
    starts, leaves them there for custody, which must neither upload them nor fail to vouch for the run dir."""
    vault = tmp_path / "vault"
    req, runs = _request(tmp_path, RID, command, vault)
    (req.parent / "inputs").mkdir()
    (req.parent / "inputs" / "lean-deps-urls.json").write_text(URLS)
    (req.parent / "inputs" / ".lean-deps-urls.json.x1y2z3").write_text(URLS)
    (req.parent / "inputs" / "sent.txt").write_text("x")
    assert remote.launch_request(req)["started"]
    d = runs / RID
    marker = _wait_marker(d)
    assert marker["preserved"] is True and CU.marker_problem(d) is None
    assert os.listdir(d / "inputs") == ["sent.txt"] and "inputs/sent.txt" in marker["files"]
    assert not any("lean-deps-urls" in f for f in marker["files"])
    _no_url_anywhere(vault)


def test_a_machine_runs_workload_writes_to_out_and_its_record_holds_all_of_it(tmp_path, monkeypatch):
    """`$OUT` is the run dir's out/, there before the workload starts, on a machine (and a queued run, which launches the same
    request) as locally; custody uploads all of it with no --declared-output."""
    vault = tmp_path / "vault"
    work = ("import os, pathlib; o = pathlib.Path(os.environ['OUT']); assert o == pathlib.Path(os.environ['RESEARCH_RUN_DIR'], 'out') "
            "and o.is_dir(); (o / 'deep').mkdir(); (o / 'deep' / 'kept.bin').write_bytes(b'k' * 1000)")
    req, runs = _request(tmp_path, RID, [PY, "-c", work], vault)
    assert remote.launch_request(req)["started"]
    d = runs / RID
    marker = _wait_marker(d)
    assert marker["preserved"] is True and json.loads((d / "status.json").read_text())["rc"] == 0
    st2 = _other_machine(tmp_path, vault, monkeypatch)
    m = st2.get_manifest(st2.get_attempt(RID)["outputs"]["run_record"])
    assert "out/deep/kept.bin" in {f.path for f in m.payload.files} and not m.meta.get("trimmed")


def test_a_repair_publish_deletes_a_sent_credential_too(tmp_path, monkeypatch, capsys):
    """A runner killed before it published leaves the run dir as the workload left it; `research data custody --publish` takes its
    record later."""
    vault = tmp_path / "vault"
    cfg = tmp_path / "store.toml"
    cfg.write_text(f'[remote]\ntype = "fs"\npath = "{vault}"\n')
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(cfg))
    runs = tmp_path / "runs"
    d = _fixture_run(runs)
    (d / "inputs").mkdir()
    (d / "inputs" / "lean-deps-urls.json").write_text(URLS)
    assert data_main(["custody", FIXTURE_RID, "--runs-dir", str(runs), "--publish"]) == 0
    assert not (d / "inputs" / "lean-deps-urls.json").exists() and CU.has_custody(d) == (True, "on the remote (.custody)")
    _no_url_anywhere(vault)


@pytest.mark.parametrize("shape", ["inputs-is-a-file", "credential-is-a-dir", "inputs-is-a-symlink"])
def test_a_run_dir_of_any_shape_still_gets_its_record(tmp_path, shape):
    """What a workload leaves at a credential's path is its own: custody deletes only a file there, never through a symlinked inputs/
    (which `run_files` doesn't follow either), and an odd shape never costs the run its record (red team, #1055 F4)."""
    run = tmp_path / "runs" / "r1"
    run.mkdir(parents=True)
    (run / "stdout.log").write_text("ok\n")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "lean-deps-urls.json").write_text("not the run's\n")
    if shape == "inputs-is-a-file":
        (run / "inputs").write_text("a workload's own file\n")
    elif shape == "credential-is-a-dir":
        (run / "inputs" / "lean-deps-urls.json").mkdir(parents=True)
        (run / "inputs" / "lean-deps-urls.json" / "x").write_text("x")
    else:
        (run / "inputs").symlink_to(elsewhere)
    st = LocalStore(tmp_path / "store")
    rr = CU.put_run_record(st, run, "r1")
    assert rr and (elsewhere / "lean-deps-urls.json").exists()
    held = {"inputs-is-a-file": {"inputs"}, "credential-is-a-dir": {"inputs/lean-deps-urls.json/x"}, "inputs-is-a-symlink": set()}[shape]
    assert {f.path for f in st.get_manifest(rr).payload.files} == {"stdout.log", *held}


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


def test_the_launchers_remote_json_does_not_cost_its_copy_custody(tmp_path, monkeypatch, capsys):
    """A remote run's record is taken on its machine; the launching machine's copy also holds remote.json, which changes."""
    vault = tmp_path / "vault"
    cfg = tmp_path / "store.toml"
    cfg.write_text(f'[remote]\ntype = "fs"\npath = "{vault}"\n')
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(cfg))
    d = _fixture_run(tmp_path / "runs")
    assert data_main(["custody", str(d), "--publish", "--keep-local"]) == 0
    capsys.readouterr()
    (d / CU.LAUNCHER_RECORD).write_text(json.dumps({"machine": "n1", "phase": "running"}))
    rep = CU.verify(_other_machine(tmp_path, vault, monkeypatch), d, mark=False)
    assert rep["preserved"], rep["error"]
    assert CU.LAUNCHER_RECORD not in rep["files"], "eviction removes only what verify lists"
    (d / CU.LAUNCHER_RECORD).write_text(json.dumps({"machine": "n1", "phase": "fetched"}))
    assert CU.verify(_other_machine(tmp_path, vault, monkeypatch), d, mark=False)["preserved"]
    (d / "stdout.log").write_text("changed")
    rep = CU.verify(_other_machine(tmp_path, vault, monkeypatch), d, mark=False)
    assert not rep["preserved"] and "stdout.log" in rep["error"] and "changed" in rep["error"], "the skip covers remote.json only"


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


def test_a_local_run_with_run_record_preserves_the_files_its_workload_left_in_the_run_dir(tmp_path, monkeypatch):
    """A pod's `research run` (sky/jobs/port-capture.yaml): without --run-record a capture's out/ stayed on the host's disk."""
    vault = tmp_path / "vault"
    cfg = tmp_path / "store.toml"
    cfg.write_text(f'[remote]\ntype = "fs"\npath = "{vault}"\n')
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(cfg))
    work = ("import os, pathlib; p = pathlib.Path(os.environ['OUT']); assert p == pathlib.Path(os.environ['RESEARCH_RUN_DIR'], 'out') "
            "and p.is_dir(); (p / 'capture.json').write_text('{}')")
    ids = {}
    for name, flags in (("plain", []), ("recorded", ["--run-record"])):
        runs = tmp_path / name
        assert cli.main(["run", "--runs-dir", str(runs), "--project", "t", "--no-sampler", *flags, "--", PY, "-c", work]) == 0
        ids[name] = R.list_runs(runs)[0]["id"]
    st2 = _other_machine(tmp_path, vault, monkeypatch)
    assert "run_record" not in st2.get_attempt(ids["plain"])["outputs"]
    m = st2.get_manifest(st2.get_attempt(ids["recorded"])["outputs"]["run_record"])
    assert m.kind == CU.KIND and {"out/capture.json", "stdout.log", "job.json"} <= {f.path for f in m.payload.files}
    assert P.check(st2, [ids["recorded"]], mode="head")["preserved"]


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


def test_run_record_is_refused_where_custody_keeps_the_run_files(capsys):
    """Neither an --on run nor a queued one reads --run-record: with --no-custody-r2 it would have kept no record, silently."""
    for where in (["--on", "m", "--no-custody-r2"], ["--queue", "--source", "."]):
        assert cli.main(["run", *where, "--project", "p", "--run-record", "--", "true"]) == 2
        assert "--run-record is for a local run" in capsys.readouterr().err


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
    (req.parent / "inputs").mkdir()
    (req.parent / "inputs" / "lean-deps-urls.json").write_text(URLS)
    (req.parent / "inputs" / ".lean-deps-urls.json.k1ll3d").write_text(URLS)    # a preflight killed between write and replace
    out = remote.launch_request(req)
    assert out["started"] is False and out["fault"] == "SourceNotReady"
    assert not (req.parent / remote.CUSTODY_DIR / CU.CRED_FILE).exists()
    assert os.listdir(req.parent / "inputs") == [], "nothing will take the sent URLs out of the request dir"


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
    waived, empty = _fixture_run(runs, "r20260925-010006-ffff"), runs / "r20260925-010007-abab"
    empty.mkdir()
    (empty / "job.json").write_text("{}")
    (empty / "status.json").write_text(json.dumps({"state": "submitted"}))
    assert data_main(["label", waived.name, "custody", "waived", "--by", "daniel"]) == 0, "triage's own next step, in the vocabulary"
    assert data_main(["label", empty.name, "custody", "never-ran", "--by", "infra"]) == 0
    assert data_main(["label", empty.name, "custody", "wavied", "--by", "infra"]) == 2
    capsys.readouterr()
    states = {r["run"]: r["state"] for r in CU.triage(LocalStore(), runs)}
    assert states == {r2.name: "r2", plain.name: "attempt-no-record", fetched.name: "fetched", never.name: "never-ran",
                      lost.name: "no-custody", waived.name: "waived", empty.name: "never-ran"}
    assert data_main(["custody", "--triage", "--runs-dir", str(runs)]) == 1
    out = capsys.readouterr().out
    assert "2 need action" in out and lost.name in out and plain.name in out


def test_a_never_ran_label_holds_only_while_the_run_dir_agrees_and_custody_needed_withdraws_a_verdict(tmp_path, monkeypatch, capsys):
    """Node 1's custody round held six `overnight-sep30` runs as never-ran and asked for the label, but each had a pid, `started`,
    `span_start` and `sampler_start` events and sampler readings of its workload: they ran, were killed and printed nothing. A
    never-ran label read before every record would take them out of triage for good, and the closed set had no value to undo it."""
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(tmp_path / "absent.toml"))
    runs, st = tmp_path / "runs", LocalStore()
    ran = _fixture_run(runs, "r20260925-010001-aaaa")
    assert data_main(["label", ran.name, "custody", "never-ran", "--by", "infra"]) == 0
    row = CU.triage_row(st, ran, host=HOST)
    assert row["state"] == "no-custody" and row["state"] in CU.NEEDS_ACTION
    assert row["why"].startswith("label custody=never-ran ignored, since status.json names pid 4243; no attempt in the store")
    d = runs / "r20260925-010002-bbbb"
    d.mkdir()
    (d / "status.json").write_text(json.dumps({"state": "submitted", "transitions": [{"state": "submitted"}]}))
    for name in ("stdout.log", "stderr.log", "resources.jsonl"):
        (d / name).write_text("")
    (d / "events.jsonl").write_text('{"kind": "submitted"}\nnot json\n[1]\n')
    assert CU.ran_evidence(d) is None
    for name, text, says in (("events.jsonl", '{"kind": "submitted"}\n{"kind": "sampler_start"}\n', "events.jsonl has sampler_start"),
                             ("resources.jsonl", '{"kind": "sample", "processes": [{"pid": 451}]}\n',
                              "resources.jsonl holds sampler readings"),
                             ("stderr.log", "Killed\n", "stderr.log holds output"),
                             ("status.json", json.dumps({"state": "running", "transitions": [{"state": "running", "pid": 451}]}),
                              "status.json names pid 451")):
        keep = (d / name).read_text()
        (d / name).write_text(text)
        assert CU.ran_evidence(d) == says
        (d / name).write_text(keep)
    for p in d.iterdir():
        os.utime(p, (NOW - CU.STALE_S - 3600,) * 2)
    assert data_main(["label", d.name, "custody", "never-ran", "--by", "infra"]) == 0
    assert CU.triage_row(st, d, now=NOW, host=HOST)["state"] == "never-ran"
    assert data_main(["label", d.name, "custody", "needed", "--by", "infra"]) == 0
    assert data_main(["label", ran.name, "custody", "waived", "--by", "infra"]) == 0
    assert CU.triage_row(st, ran, host=HOST)["state"] == "waived", "a waiver needs no sign of anything"
    assert data_main(["label", ran.name, "custody", "needed", "--by", "infra"]) == 0
    assert [CU.triage_row(st, x, now=NOW, host=HOST)["state"] for x in (d, ran)] == ["no-custody", "no-custody"]
    capsys.readouterr()


HOST = ("node-2", "boot-1")


def _on_host(d: Path, host: tuple[str, str], state: str | None, t: float, **transition) -> None:
    """The run dir as the machine that ran it sees it: job.json's host, a status (None: none yet), every file last changed at `t`."""
    job = json.loads((d / "job.json").read_text())
    job["environment"].update(hostname=host[0], boot_id=host[1])
    (d / "job.json").write_text(json.dumps(job))
    (d / "status.json").unlink()
    if state:
        (d / "status.json").write_text(json.dumps({"state": state, "transitions": [{"state": state, **transition}]}))
    for p in d.iterdir():
        os.utime(p, (t, t))


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="a runner's liveness is read from /proc")
def test_triage_on_the_machine_tells_a_live_runner_from_a_dead_one_and_a_pods_run_from_a_stale_one(tmp_path):
    """Node 2 runs `research run` as host processes, node 1 in Kubernetes pods. A run still marked running has ended when its runner's
    pid (same start time) is gone from the host that ran it or that host rebooted since, or, when that runner is in another pid
    namespace, when none of its records changed for 24 h: node 1 had 326 such runs on 3 Oct, preempted before they recorded an end,
    that nothing published."""
    runs, st = tmp_path / "runs", LocalStore(tmp_path / "s", config_path=tmp_path / "absent.toml")
    me = int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19])
    live, dead, pod, stale, starting = (_fixture_run(runs, f"r20260925-01000{i}-aaaa") for i in range(5))
    _on_host(live, HOST, "running", NOW - 7200, pid=os.getpid(), starttime_ticks=me)
    _on_host(dead, HOST, "running", NOW - 7200, pid=2**22 + 7, starttime_ticks=1)
    _on_host(pod, ("pod-xyz", "boot-1"), "running", NOW - 7200, pid=1, starttime_ticks=1)
    _on_host(stale, ("pod-xyz", "boot-1"), "running", NOW - CU.STALE_S - 60, pid=1, starttime_ticks=1)
    _on_host(starting, HOST, None, NOW - 60)
    rows = {r["run"]: r for r in CU.triage(st, runs, now=NOW, host=HOST)}
    assert {k: r["state"] for k, r in rows.items()} == {live.name: "running", dead.name: "no-custody", pod.name: "running",
                                                       stale.name: "no-custody", starting.name: "running"}
    assert rows[dead.name]["why"].startswith("dead: its runner (pid 4194311)") and "--publish --force" in rows[dead.name]["next"]
    assert rows[stale.name]["why"].startswith("stale: status running") and rows[pod.name]["why"].startswith("running elsewhere")
    assert "--publish --force --runner-gone (once its host says its runner is gone)" in rows[stale.name]["next"]
    assert rows[dead.name]["age_min"] == 120.0 and rows[dead.name]["bytes"] > 0 and rows[live.name]["bytes"] is None
    other_boot = {r["run"]: r for r in CU.triage(st, runs, now=NOW, host=(HOST[0], "boot-2"))}
    assert other_boot[live.name]["state"] == "no-custody", "a reboot of the host that ran it ended every runner"
    assert other_boot[live.name]["why"].startswith("dead: this host rebooted since its runner started (boot boot-1, now boot-2)")
    assert other_boot[pod.name]["state"] == "running" and other_boot[starting.name]["state"] == "running"


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="a runner's liveness is read from /proc")
def test_a_live_run_whose_timeout_was_changed_is_running_and_a_status_naming_no_pid_is_never_dead(tmp_path):
    """`tele set-timeout` appends a `running` transition without a pid. Triage and n2_custody.sh read the last transition's pid,
    found none and took a live run for dead, and n2_custody.sh published it --force: an attempt in state running with no result,
    which no later publish can replace. The runner's pid is status.json's own (set_status keeps it), and a status naming no pid
    that reads is never taken for dead."""
    runs, st = tmp_path / "runs", LocalStore(tmp_path / "s", config_path=tmp_path / "absent.toml")
    me = int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19])
    live, dead, nopid = (_fixture_run(runs, f"r20260925-01000{i}-aaaa") for i in range(3))
    for d, pid, ticks in ((live, os.getpid(), me), (dead, 2**22 + 7, 1)):
        _on_host(d, HOST, None, NOW - 7200)
        set_status(d, "running", pid=pid, pgid=pid, starttime_ticks=ticks, stage="t")
        set_status(d, "running", timeout_s=60, timeout_changed_utc="2026-09-25T01:00:00.000Z")
        assert "pid" not in json.loads((d / "status.json").read_text())["transitions"][-1]
    _on_host(nopid, HOST, "running", NOW - 7200, pid="?")
    rows = {r["run"]: r for r in CU.triage(st, runs, now=NOW, host=HOST)}
    assert {k: r["state"] for k, r in rows.items()} == {live.name: "running", dead.name: "no-custody", nopid.name: "running"}
    assert rows[live.name]["why"] == "running here" and rows[dead.name]["why"].startswith("dead: its runner (pid 4194311)")
    assert rows[nopid.name]["why"] == "running here (its status names no pid that reads)"


def test_triage_takes_a_valid_custody_marker_where_the_catalog_cannot_say(tmp_path, monkeypatch):
    """Node 2's runs are published on node 1 (n2_custody.sh), so node 2's own catalog never sees them on the remote: the .custody that
    node 1's verify wrote, copied back beside identical files, says so instead, and stops saying so once a file changes after it."""
    cfg = tmp_path / "store.toml"
    cfg.write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "vault"}"\n')
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(cfg))
    n1_runs, n2_runs = tmp_path / "n1-runs", tmp_path / "n2-runs"
    d1 = _fixture_run(n1_runs, "r20260925-010001-aaaa")
    shutil.copytree(d1, n2_runs / d1.name, copy_function=shutil.copy2)
    assert data_main(["custody", str(d1), "--publish", "--keep-local", "--store", str(tmp_path / "n1-store")]) == 0
    d2 = n2_runs / d1.name
    shutil.copy2(d1 / CU.MARKER, d2 / CU.MARKER)
    n2 = LocalStore(tmp_path / "n2-store", config_path=tmp_path / "absent.toml")
    (row,) = CU.triage(n2, n2_runs, host=HOST)
    assert row["state"] == "r2" and row["why"].startswith("on the remote (.custody, verified ") and row["next"] is None
    (d2 / "late.txt").write_text("written after custody")
    (row,) = CU.triage(n2, n2_runs, host=HOST)
    assert row["state"] == "no-custody"


def test_the_custody_temp_file_a_killed_publish_leaves_is_no_file_of_the_run(tmp_path, monkeypatch):
    """A custody Job stopped at its deadline (its bash exits on SIGTERM, and the kernel SIGKILLs research) can leave the
    `.custody.tmp.<pid>` write_json_atomic writes before its rename. Counted as a run file, verify found it not in the run record,
    and n1_custody.sh would hold the run as changed after its record, telling a person to waive logs the record holds."""
    cfg = tmp_path / "store.toml"
    cfg.write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "vault"}"\n')
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(cfg))
    d, store = _fixture_run(tmp_path / "runs", "r20260925-010003-cccc"), str(tmp_path / "s")
    assert data_main(["custody", str(d), "--publish", "--keep-local", "--store", store]) == 0
    (d / f"{CU.MARKER}.tmp.4242").write_text('{"preserved": tr')
    assert f"{CU.MARKER}.tmp.4242" not in CU.run_files(d) and CU.marker_problem(d) is None
    assert CU.verify(LocalStore(store), d)["preserved"]


def test_a_remote_holding_another_attempt_document_is_a_conflict_that_triage_reports_with_its_repair(tmp_path, monkeypatch, capsys):
    """3 Oct: r20261003-174723-7ea5's attempt was already on R2 with other bytes, and an attempt key is write-once, so a node's custody
    Job failed on it every round. A failed `custody --publish` leaves its verdict in .custody (never over a marker that still vouches
    for the dir), triage reports the run as `conflict` with the repair, and the repair (reconcile, then --publish) makes it r2."""
    from research.store import attempt as A
    from research.store.local import attempt_key

    vault, store = tmp_path / "vault", str(tmp_path / "s")
    cfg = tmp_path / "store.toml"
    cfg.write_text(f'[remote]\ntype = "fs"\npath = "{vault}"\n')
    runs = tmp_path / "runs"
    ok, d = _fixture_run(runs, "r20260925-010001-aaaa"), _fixture_run(runs, "r20260925-010002-bbbb")
    offline = LocalStore(store, config_path=tmp_path / "absent.toml")
    assert A.publish(d, store=offline, remote_push=False, run_record=True)["published"], "published where the remote wasn't visible"
    first = json.loads(offline.attempt_path(d.name).read_bytes())
    first["status"]["rc"] = 99
    (vault / attempt_key(d.name)).parent.mkdir(parents=True)
    (vault / attempt_key(d.name)).write_text(json.dumps(first, sort_keys=True, separators=(",", ":")))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(cfg))
    assert data_main(["custody", str(ok), "--publish", "--keep-local", "--store", store]) == 0
    capsys.readouterr()

    assert data_main(["custody", str(d), "--publish", "--keep-local", "--store", store, "--json"]) == 1
    marker = json.loads((d / CU.MARKER).read_text())
    assert marker["preserved"] is False and P.DIFFERS in marker["error"] and marker["verified_utc"] and marker["host"]
    rows = {r["run"]: r for r in CU.triage(LocalStore(store), runs, host=HOST)}
    assert rows[ok.name]["state"] == "r2" and rows[d.name]["state"] == "conflict" and rows[d.name]["files"] == len(CU.run_files(d))
    assert P.DIFFERS in rows[d.name]["why"] and f"reconcile-attempts {d.name} --apply" in rows[d.name]["next"]
    (verified,) = [r for r in CU.triage(LocalStore(store), runs, verify_remote=True, host=HOST) if r["run"] == d.name]
    assert verified["state"] == "conflict", "--verify asks the remote itself"
    assert data_main(["custody", "--triage", "--runs-dir", str(runs), "--store", store]) == 1
    assert "1 need action" in capsys.readouterr().out

    (vault / attempt_key(ok.name)).write_text("{}")
    assert data_main(["custody", str(ok), "--publish", "--keep-local", "--store", store]) == 1
    assert CU.marker_problem(ok) is None, "a failed check doesn't overwrite a marker that still vouches for the run dir"

    assert data_main(["reconcile-attempts", d.name, "--apply", "--store", store]) == 0
    assert data_main(["custody", str(d), "--publish", "--keep-local", "--store", store]) == 0
    capsys.readouterr()
    (row,) = [r for r in CU.triage(LocalStore(store), runs, host=HOST) if r["run"] == d.name]
    assert row["state"] == "r2" and row["files"] is None


def test_older_min_counts_only_runs_that_ended_n_minutes_ago_across_every_runs_root(tmp_path, capsys, monkeypatch):
    """The monitors' invariant, per node: ended runs not on the remote N minutes after they ended. One read-only command over every
    runs root of the node, `research/custody-triage/v1` with --json, exit 1 on a finding and 3 when triage itself fails; a run that
    just ended is settling (its own custody, or the node's, may still be on the way) and doesn't count."""
    jobs, research = tmp_path / "jobs-runs", tmp_path / "research-runs"
    old, new = _fixture_run(jobs, "r20260925-010001-aaaa"), _fixture_run(research, "r20260925-010002-bbbb")
    for d, t in ((old, NOW - 3600), (new, NOW - 600)):
        for p in d.iterdir():
            os.utime(p, (t, t))
    st = LocalStore(tmp_path / "s", config_path=tmp_path / "absent.toml")
    rep = CU.triage_report(st, [jobs, research], older_min=30, now=NOW, host=HOST)
    assert rep["schema"] == "research/custody-triage/v1" and rep["needs_action"] == 1 and rep["settling"] == 1
    assert {r["run"]: (r["root"], r["settling"]) for r in rep["rows"]} == {old.name: (str(jobs), False), new.name: (str(research), True)}
    assert CU.triage_report(st, [jobs, research], now=NOW, host=HOST)["needs_action"] == 2
    (old / "resources.jsonl").write_text("{}\n")
    os.utime(old / "resources.jsonl", (NOW - 600, NOW - 600))
    assert CU.triage_report(st, [jobs], older_min=30, now=NOW, host=HOST)["settling"] == 1, "the sampler's last sample is a record"
    for d, t in ((old, 1_000_000_000), (new, 4_102_444_800)):    # the CLI reads the clock itself: 2001 is older than any N, 2100 younger
        for p in d.iterdir():
            os.utime(p, (t, t))
    base = ["custody", "--triage", "--store", str(tmp_path / "s"), "--older-min", "30"]
    assert data_main([*base, "--runs-dir", str(jobs), "--runs-dir", str(research), "--json"]) == 1
    doc = json.loads(capsys.readouterr().out)
    assert (doc["needs_action"], doc["settling"], doc["runs_roots"]) == (1, 1, [str(jobs), str(research)])
    assert data_main([*base, "--runs-dir", str(research)]) == 0
    assert "1 more ended under 30 min ago (settling" in capsys.readouterr().out
    assert data_main([*base, "--runs-dir", str(tmp_path / "typo")]) == 2, "a runs root that isn't there is not an empty one"

    def crash(*a, **k):
        raise KeyError("transitions")

    monkeypatch.setattr(CU, "triage_report", crash)
    assert data_main([*base, "--runs-dir", str(jobs), "--json"]) == 3
    assert "crashed, nothing triaged: KeyError" in capsys.readouterr().err
    for bad in (["custody", old.name, "--older-min", "5"], ["custody", "--triage", "--older-min", "-1"], ["custody", old.name, "--force"]):
        with pytest.raises(SystemExit):
            data_main(bad)


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="a runner's liveness is read from /proc")
def test_publish_force_takes_home_a_run_whose_runner_died_and_another_hosts_only_once_that_host_says_so(tmp_path, monkeypatch, capsys):
    """n2_custody.sh publishes a dead run (an OOM-killed scope, a preempted pod) through `research data custody --publish`, which needs
    --force for a status that never became terminal; without it the run is refused, as `attempt publish` refuses it. This host sees
    whether a runner that ran here is gone; a pod's or another machine's it can't, and a pod whose records are quiet for 24 h may
    still run (#1004's red-team F2), so --force publishes such a run only with --runner-gone, which the caller passes once the run's
    own host says so, and never one whose runner runs here."""
    cfg = tmp_path / "store.toml"
    cfg.write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "vault"}"\n')
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(cfg))
    here, me = CU.this_host(), int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19])
    pod, dead, live = (_fixture_run(tmp_path / "runs", f"r20260925-01000{i}-aaaa") for i in (1, 2, 3))
    _on_host(pod, ("pod-xyz", here[1]), "running", 1_000_000_000, pid=1, starttime_ticks=1)
    _on_host(dead, here, "running", 1_000_000_000, pid=2**22 + 7, starttime_ticks=1)
    _on_host(live, here, "running", 1_000_000_000, pid=os.getpid(), starttime_ticks=me)
    assert data_main(["custody", str(pod), "--publish", "--keep-local"]) == 1
    assert "the store only sees completed or failed attempts" in capsys.readouterr().err
    assert data_main(["custody", str(pod), "--publish", "--keep-local", "--force"]) == 3
    err = capsys.readouterr().err
    assert f"{pod.name}: --force refused: stale: status running" in err and not CU.has_custody(pod)[0]
    assert "pass --runner-gone once that host says it is gone (a pod: `kubectl get pod pod-xyz` NotFound)" in err
    assert data_main(["custody", str(pod), "--publish", "--keep-local", "--force", "--runner-gone"]) == 0
    assert CU.has_custody(pod) == (True, "on the remote (.custody)")
    assert data_main(["custody", str(dead), "--publish", "--keep-local", "--force"]) == 0, "this host sees its pid gone"
    assert data_main(["custody", str(live), "--publish", "--keep-local", "--force", "--runner-gone"]) == 3
    assert "--force refused: its runner is running here" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        data_main(["custody", str(pod), "--publish", "--runner-gone"])


def test_a_run_published_force_while_it_ran_is_not_r2_once_it_writes_its_end(tmp_path, monkeypatch):
    """#1004's red-team F2: a pod forced as dead while it still ran ends later; its runner's own publish then returns `already` (an
    attempt is write-once), and triage called the run r2 though its end, its result and its last logs were only on node 1. An
    attempt published --force is r2 only while .custody shows nothing in the run dir changed after its run record was taken."""
    from research.store import attempt as A

    cfg = tmp_path / "store.toml"
    cfg.write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "vault"}"\n')
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(cfg))
    runs, st = tmp_path / "runs", LocalStore(tmp_path / "store", config_path=cfg)
    d = _fixture_run(runs, "r20260925-010001-aaaa")
    _on_host(d, ("pod-xyz", "boot-1"), "running", 1_000_000_000, pid=1, starttime_ticks=1)
    assert A.publish(d, store=st, remote_push=False, run_record=True, force=True)["published"]
    assert CU.push_run(st, d.name) == [] and CU.verify(st, d)["preserved"]
    (row,) = CU.triage(st, runs, host=HOST)
    assert row["state"] == "r2" and "published --force, and .custody shows nothing in the run dir changed since" in row["why"]
    (d / "stdout.log").write_text("the workload's stdout\n... and its last 20 hours, and its result\n")
    (d / "status.json").write_text(json.dumps({"state": "done", "transitions": [{"state": "running", "pid": 1},
                                                                                 {"state": "done", "rc": 0}]}))
    for p in (d / "stdout.log", d / "status.json"):         # 20 h later (the kernel's mtime clock is coarser than verified_from's)
        os.utime(p, (4_102_444_800, 4_102_444_800))
    assert A.publish(d, store=st, remote_push=False, run_record=True).get("already")
    (row,) = CU.triage(st, runs, host=HOST)
    assert row["state"] == "forced-unverified" and row["state"] in CU.NEEDS_ACTION
    assert "published --force with status running, and no .custody shows the record holds the run dir as it is now: " in row["why"]
    assert "changed after custody" in row["why"] and "the attempt is write-once" in row["next"]
    (d / CU.MARKER).unlink()
    (row,) = CU.triage(st, runs, host=HOST)
    assert row["state"] == "forced-unverified" and row["next"].startswith(f"research data custody {d.name} (it writes .custody")


def test_a_publish_that_the_remote_refuses_names_the_conflict_beside_the_missing_record(tmp_path, monkeypatch, capsys):
    """r20261003-174723-7ea5: its runner's custody pushed the attempt and its run record, then a file of the run dir changed, and the
    stopgap published an attempt without a run record on node 1, which the remote's write-once key refuses. n2_custody.sh holds a
    run on the line `research data custody --publish` prints, so that line names the push's RemoteConflict too."""
    from research.store import attempt as A

    cfg = tmp_path / "store.toml"
    cfg.write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "vault"}"\n')
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(cfg))
    d = _fixture_run(tmp_path / "runs", "r20260925-010001-aaaa")
    assert data_main(["custody", str(d), "--publish", "--keep-local", "--store", str(tmp_path / "runner")]) == 0
    (d / "stdout.log").write_text("the workload's stdout, and a line after the record\n")
    assert A.publish(d, store=LocalStore(tmp_path / "n1", config_path=tmp_path / "absent.toml"))["published"]
    capsys.readouterr()
    assert data_main(["custody", str(d), "--publish", "--keep-local", "--store", str(tmp_path / "n1")]) == 1
    out = capsys.readouterr().out
    assert "NO custody on the remote: the attempt names no run record" in out and "; push: RemoteConflict: " in out


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
    assert "--declared-output" in w and "$OUT" in w and "will be dropped once lanes declare their outputs" in w and CU.SKILL in w


def test_select_always_keeps_all_of_out_and_a_declared_glob_adds_to_it(tmp_path):
    d = _fixture_run(tmp_path / "runs")
    (d / "out" / "deep").mkdir()
    (d / "out" / "deep" / "big.bin").write_bytes(b"x" * 100)
    (d / "res").mkdir()
    (d / "res" / "r.json").write_text("{}")
    (d / "outside.bin").write_bytes(b"z" * 100)
    (d / "outx.bin").write_bytes(b"z" * 10)
    kept, trimmed = CU.select(d, cap=0)
    assert {"out/proof.bin", "out/verify.jsonl", "out/deep/big.bin"} <= set(kept), "out/ is kept in full with nothing declared"
    assert [t["path"] for t in trimmed] == ["outside.bin", "outx.bin", "res/r.json"], "only out/, not a name that starts with out"
    _declare(d, ["res/*"])
    kept, trimmed = CU.select(d, cap=0)
    assert {"out/deep/big.bin", "res/r.json"} <= set(kept) and [t["path"] for t in trimmed] == ["outside.bin", "outx.bin"]


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
    assert "custody trimmed 1 undeclared file(s)" in capsys.readouterr().err   # junk.bin; the fixture's out/ is always kept
    assert CU.has_custody(d) == (True, "on the remote (.custody)")
    st2 = _other_machine(tmp_path, vault, monkeypatch)
    m = st2.get_manifest(st2.get_attempt(FIXTURE_RID)["outputs"]["run_record"])
    paths = {f.path for f in m.payload.files}
    assert {"res/result.json", "out/proof.bin", "out/verify.jsonl"} <= paths and "junk.bin" not in paths
    assert [t["path"] for t in m.meta["trimmed"]] == ["junk.bin"]
    assert CU.verify(st2, d, mark=False)["preserved"]
    (d / "junk.bin").write_bytes(b"k" * 1000)
    assert "changed" in CU.verify(st2, d, mark=False)["error"], "a trimmed file is still pinned by its hash"


def test_a_trimmed_listing_too_big_for_the_manifest_goes_in_the_record_and_custody_still_verifies(tmp_path, monkeypatch, capsys):
    """A check run lists thousands of undeclared files; inline in meta that broke the 256 KiB manifest cap and the attempt never
    published (r20261001-091202-7aa6)."""
    vault = tmp_path / "vault"
    cfg = tmp_path / "store.toml"
    cfg.write_text(f'[remote]\ntype = "fs"\npath = "{vault}"\n')
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(cfg))
    monkeypatch.setattr(CU, "UNDECLARED_CAP_BYTES", 0)
    runs = tmp_path / "runs"
    d = _fixture_run(runs)
    (d / "suites").mkdir()
    for i in range(3000):
        (d / "suites" / f"t{i:04d}.log").write_text(str(i))
    assert data_main(["custody", FIXTURE_RID, "--runs-dir", str(runs), "--publish", "--keep-local"]) == 0
    capsys.readouterr()
    st2 = _other_machine(tmp_path, vault, monkeypatch)
    rr = st2.get_attempt(FIXTURE_RID)["outputs"]["run_record"]
    m = st2.get_manifest(rr)
    assert "trimmed" not in m.meta and m.meta["trimmed_listing"] == CU.TRIMMED_LISTING and m.meta["trimmed_files"] == 3000
    assert {f.path for f in m.payload.files} & {CU.TRIMMED_LISTING, "stdout.log"} == {CU.TRIMMED_LISTING, "stdout.log"}
    assert len(CU.trimmed_of(st2, rr)) == 3000
    rep = CU.verify(st2, d, mark=False)
    assert rep["preserved"] and CU.TRIMMED_LISTING not in rep["gone"]
    (d / "suites" / "t2999.log").write_text("changed")
    assert "changed" in CU.verify(st2, d, mark=False)["error"]
