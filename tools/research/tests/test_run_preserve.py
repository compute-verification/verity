"""`research run-preserve PATH` (store.run_preserve): the workload of a `research run --on` asks, and its runner puts one file
of the run's out/ in the evidence store while the workload still runs, with the custody key the workload never holds.  The
machine is a local directory and the remote an `fs` vault.  The first tests launch a real runner; the rest step its loop
themselves, with the runner's child run in this process (`Child`) or stood in for."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import re
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from research import remote
from research.store import custody as CU
from research.store import preserved as P
from research.store import remote_s3
from research.store import run_preserve as RP
from research.store.cli import main as data_main
from research.store.attempt import store_root
from research.store.local import object_key
from research.store.remote import FsRemote
from research.telemetry import run as trun
from research.telemetry.scope import claim

from test_run_custody import FIXTURE_RID, PY, RID, _custody_dir, _fixture_run, _other_machine, _request, _wait_marker
from test_store import _attempt
from waits import FakeClock, Gate, wait_runner

COIN = '{"coin": "0f1e2d3c", "window": 7}\n'
SHA = hashlib.sha256(COIN.encode()).hexdigest()
S3_NOWHERE = ('[remote]\ntype = "s3"\nbucket = "bucket-of-the-run"\nendpoint = "http://127.0.0.1:9"\nregion = "auto"\n'
              'access_key_id_env = "AWS_ACCESS_KEY_ID"\nsecret_access_key_env = "AWS_SECRET_ACCESS_KEY"\n')

#: A workload that stops twice for the test, before it asks and after: ARGV is the two gates, the file's text and the
#: `research run-preserve` command lines to run from $OUT (a JSON list of argument lists).
WORKLOAD = r"""
import json, os, subprocess, sys

out = os.environ["OUT"]
at, go, text, asks = sys.argv[1:5]


def note(name, doc):
    with open(os.path.join(out, name), "w") as f:
        json.dump(doc, f)


def stop(word):
    with open(at, "w") as f:
        f.write(word + "\n")
    with open(go, "rb") as f:
        f.read(1)


word = "failed"
try:
    note("env-before.json", dict(os.environ))
    with open(os.path.join(out, "coin.json"), "w") as f:
        f.write(text)
    stop("ready")
    asked = []
    for argv in json.loads(asks):
        p = subprocess.run(["research", "run-preserve", *argv], capture_output=True, text=True, cwd=out)
        asked.append({"rc": p.returncode, "stdout": p.stdout, "stderr": p.stderr})
    note("asked.json", asked)
    note("env-after.json", dict(os.environ))
    word = "asked"
finally:
    stop(word)
"""


def _key(cdir: Path, vault: Path) -> list[str]:
    """What must reach neither the workload nor a file it reads: the staged custody key's three values and the remote's address."""
    cred = json.loads((cdir / CU.CRED_FILE).read_text())
    return [*(cred[k] for k in CU.KEY_ENV), str(vault)]


def _holding(root: Path, needles: list[str], skip: tuple[str, ...] = ()) -> list[str]:
    """`path: needle` for every file under `root` (but `skip`, top-level names) that holds one of `needles`."""
    return sorted(f"{p.relative_to(root)}: {n}" for p in root.rglob("*")
                  if p.is_file() and p.relative_to(root).parts[0] not in skip for n in needles if n.encode() in p.read_bytes())


def _answers(d: Path) -> dict[str, dict]:
    return {p.name: json.loads(p.read_text()) for p in sorted((d / RP.ANSWERS).iterdir()) if not p.name.startswith(".")}


# ---- a real runner -----------------------------------------------------------------------------------------------------------------------

def test_a_workload_has_its_runner_preserve_a_file_while_it_runs_and_never_holds_the_key(tmp_path, monkeypatch):
    """The whole path, with a runner `launch-request` started and the shipped `research` on the workload's PATH: the art id
    comes back while the workload is still running and another machine finds the artifact on the remote; a path outside out/
    is refused through the same loop; neither the workload's environment nor any file under the machine's root outside the
    runner's requests/ holds the key or the remote's address, before the request, after it, or once the run has ended; and
    the run's custody at the end passes, with the file and the runner's answers in its record."""
    vault, root, store = tmp_path / "vault", tmp_path / "root", store_root()         # the last: the machine's store
    script = tmp_path / "workload.py"
    script.write_text(WORKLOAD)
    at, go = Gate(tmp_path / "at"), Gate(tmp_path / "go")
    asks = [["coin.json", "--meta", '{"what": "coin", "run_id": "not-this-run", "path": "not-this-path"}'], ["../job.json"]]
    req, runs = _request(tmp_path, RID, [PY, str(script), str(at.path), str(go.path), COIN, json.dumps(asks)], vault)
    d, cdir = runs / RID, req.parent / remote.CUSTODY_DIR
    key = _key(cdir, vault)
    assert remote.launch_request(req)["started"]
    try:
        assert at.readline() == b"ready\n"
        assert (d / RP.REQUESTS).is_dir() and _answers(d) == {}
        assert _holding(root, key, skip=("requests",)) == [] and _holding(store, key) == []
        go.release()
        assert at.readline() == b"asked\n"

        # the workload has its answer and is still running
        assert json.loads((d / "status.json").read_text())["state"] == "running" and not (d / CU.MARKER).exists()
        first, second = json.loads((d / "out" / "asked.json").read_text())
        art = first["stdout"].strip()
        assert first["rc"] == 0 and first["stdout"] == f"{art}\n" and re.fullmatch("art:[0-9a-f]{64}", art)
        assert first["stderr"] == (f"research run-preserve: PRESERVED out/coin.json: {len(COIN)} bytes, sha256 {SHA}, verified on the "
                                   "store's remote\n")
        other = _other_machine(tmp_path, vault, monkeypatch)
        assert P.check(other, [art], mode="head")["preserved"] is True
        m = other.get_manifest(art)
        assert (m.kind, m.meta) == (RP.KIND, {"what": "coin", "run_id": RID, "path": "out/coin.json"})
        assert (m.payload.type, m.payload.name, m.payload.sha256, m.payload.bytes) == ("file", "coin.json", SHA, len(COIN))
        assert (vault / object_key(SHA)).read_text() == COIN
        preserved, refused = _answers(d).values()
        assert preserved == {"schema": RP.ANSWER_SCHEMA, "request": preserved["request"], "verdict": "PRESERVED", "art": art,
                             "bytes": len(COIN), "sha256": SHA, "path": "out/coin.json", "kind": RP.KIND, "run_id": RID}

        assert second["rc"] == 1 and second["stdout"] == "" and refused["verdict"] == "REFUSED"
        assert second["stderr"] == f"research run-preserve: REFUSED: {refused['reason']}\n"
        assert f"is not inside this run's out/ ({d / 'out'}): it resolves to {d / 'job.json'}" in refused["reason"]

        # what the puts wrote that can name the remote is beside the key, the runner's alone
        assert stat.S_IMODE((cdir / RP.LOG).stat().st_mode) == 0o600 and stat.S_IMODE((cdir / RP.STORE).stat().st_mode) == 0o700
        assert _holding(cdir, [str(vault)], skip=(CU.STORE_TOML,)) != [], "the scan below would find the address had it left this dir"
        assert _holding(root, key, skip=("requests",)) == [] and _holding(store, key) == []
        env = [json.loads((d / "out" / f"env-{when}.json").read_text()) for when in ("before", "after")]
        assert env[0] == env[1] and env[0]["RESEARCH_RUN_DIR"] == str(d)
        assert not [k for k in env[0] if k in CU.KEY_ENV] and not [n for n in key if n in json.dumps(env)]
        assert env[0].get("RESEARCH_STORE_CONFIG") != str(cdir / CU.STORE_TOML)
    finally:
        go.release()
        go.release()
        wait_runner(d)
        at.close()
        go.close()

    marker = _wait_marker(d)
    assert marker["preserved"] is True and CU.marker_problem(d) is None
    assert json.loads((d / "status.json").read_text())["rc"] == 0
    names = {f"{RP.ANSWERS}/{n}" for n in (*_answers(d), )}
    assert f"{RP.CLOSED}" in names and len(names) == 3
    record = other.get_manifest(other.get_attempt(RID)["outputs"]["run_record"])
    held = {f.path: f.sha256 for f in record.payload.files}
    assert held["out/coin.json"] == SHA and names <= set(held) and not record.meta.get("trimmed")
    assert not (cdir / RP.STORE).exists() and not (cdir / CU.CRED_FILE).exists()
    assert _holding(d, key, skip=(CU.MARKER,)) == [], "but for the marker its custody wrote once the workload was over"


def test_a_run_without_a_custody_key_is_told_so_and_goes_on(tmp_path):
    script = tmp_path / "workload.py"
    script.write_text(WORKLOAD)
    at, go = Gate(tmp_path / "at"), Gate(tmp_path / "go")
    req, runs = _request(tmp_path, RID, [PY, str(script), str(at.path), str(go.path), COIN, json.dumps([["coin.json"]])], tmp_path / "vault")
    doc = json.loads(req.read_text())
    del doc["custody"]
    req.write_text(json.dumps(doc))
    d = runs / RID
    assert remote.launch_request(req)["started"]
    try:
        assert at.readline() == b"ready\n"
        go.release()
        assert at.readline() == b"asked\n"
    finally:
        go.release()
        go.release()
        wait_runner(d)
        at.close()
        go.close()
    (asked,) = json.loads((d / "out" / "asked.json").read_text())
    assert asked["rc"] == 1 and asked["stdout"] == ""
    assert asked["stderr"] == f"research run-preserve: NOT_PRESERVED: {RP.NO_OFFER}. coin.json stays where it is\n"
    assert not (d / RP.DIR).exists() and not (tmp_path / "vault").exists()
    assert json.loads((d / "status.json").read_text())["rc"] == 0


# ---- the runner's loop, stepped here -------------------------------------------------------------------------------------------------------

class Child:
    """The runner's child, run in this process: what `python -m research.store.run_preserve RUN_DIR CUSTODY_DIR NAME` does, with
    its output written where the runner pointed it."""

    started: list[tuple[list[str], dict]] = []

    def __init__(self, argv: list[str], **kw):
        self.started.append((argv, kw))
        assert argv[:3] == [PY, "-m", RP.MODULE]
        said = io.StringIO()
        with contextlib.redirect_stderr(said):
            RP.serve(*argv[3:])
        os.write(kw["stderr"], said.getvalue().encode())

    def poll(self) -> int:
        return 0


class Run:
    """A run dir whose workload is still running, its custody dir and its runner's Preserver; `ask` is the workload's command,
    each of whose waits is one pass of the runner's loop."""

    def __init__(self, tmp_path: Path, monkeypatch, capsys, popen=Child, remote_toml: str | None = None, expires: str = "2999-01-01T00:00:00Z"):
        self.tmp_path, self.monkeypatch, self.capsys, self.vault = tmp_path, monkeypatch, capsys, tmp_path / "vault"
        self.d = _fixture_run(tmp_path / "root" / "runs")
        self.out = self.d / "out"
        self.out.mkdir(exist_ok=True)
        self.cdir = _custody_dir(tmp_path, FIXTURE_RID, self.vault, remote_toml)
        cred = json.loads((self.cdir / CU.CRED_FILE).read_text())
        (self.cdir / CU.CRED_FILE).write_text(json.dumps({**cred, "expires_utc": expires}))
        self.clock = FakeClock()
        monkeypatch.setattr(RP, "time", SimpleNamespace(monotonic=self.clock.module.monotonic, sleep=self._pass))
        monkeypatch.setattr(Child, "started", [])
        self.pres = RP.Preserver(self.d, self.cdir, popen=popen)

    def _pass(self, seconds: float) -> None:
        self.clock.sleep(seconds)
        self.pres.poll()

    def ask(self, *argv: object) -> tuple[int, str, list[str]]:
        """(exit status, stdout, the command's own stderr lines)."""
        rc = RP.main([str(a) for a in argv], environ={"RESEARCH_RUN_DIR": str(self.d)})
        out, err = self.capsys.readouterr()
        return rc, out, [ln.removeprefix("research run-preserve: ") for ln in err.splitlines() if ln.startswith("research run-preserve: ")]

    def file(self, rel: str = "coin.json", data: str | bytes = COIN) -> Path:
        p = self.out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data.encode() if isinstance(data, str) else data)
        return p

    def request(self, name: str, path: Path | str) -> None:
        """A request as the command leaves it, with nobody waiting for its answer."""
        claim(self.d / RP.REQUESTS / name, {"schema": RP.REQUEST_SCHEMA, "path": str(path), "kind": RP.KIND, "meta": {}}, mode=0o644)


@pytest.fixture
def run(tmp_path, monkeypatch, capsys):
    return Run(tmp_path, monkeypatch, capsys)


def test_the_command_prints_the_art_id_of_a_file_the_remote_was_checked_to_hold(run, monkeypatch):
    rc, out, err = run.ask(run.file(), "--kind", "coin-record/v1", "--meta", '{"window": 7}')
    art = out.strip()
    assert rc == 0 and out == f"{art}\n"
    assert err == ["unknown kind 'coin-record/v1' (store/kinds.py lists the known ones); asking anyway",
                   f"PRESERVED out/coin.json: {len(COIN)} bytes, sha256 {SHA}, verified on the store's remote"]
    m = _other_machine(run.tmp_path, run.vault, monkeypatch).get_manifest(art)
    assert (m.kind, m.meta) == ("coin-record/v1", {"window": 7, "run_id": FIXTURE_RID, "path": "out/coin.json"})
    assert (run.vault / object_key(SHA)).read_text() == COIN
    assert (run.d / "out" / "coin.json").read_text() == COIN, "the file stays in out/, for the run's record"


def test_the_runner_hands_its_child_the_custody_dirs_path_and_no_key(run):
    assert run.ask(run.file())[0] == 0
    ((argv, kw),) = Child.started
    (name,) = _answers(run.d)
    assert argv == [PY, "-m", RP.MODULE, str(run.d), str(run.cdir), name]
    assert sorted(kw) == ["env", "stderr", "stdin", "stdout"] and kw["stdin"] == subprocess.DEVNULL and kw["stdout"] == kw["stderr"]
    key = _key(run.cdir, run.vault)
    assert not [k for k in kw["env"] if k in CU.KEY_ENV] and not [n for n in key if n in json.dumps([argv, kw["env"]])]
    assert not [k for k in os.environ if k in CU.KEY_ENV], "the key was in the child's environment for the put alone"
    assert stat.S_IMODE((run.cdir / RP.LOG).stat().st_mode) == 0o600 and '"verdict": "PRESERVED"' in (run.cdir / RP.LOG).read_text()
    assert stat.S_IMODE((run.cdir / RP.STORE).stat().st_mode) == 0o700
    assert _holding(run.d, key) == [] and _holding(store_root(), key) == []


def _outside(run: Run) -> Path:
    return run.d / "stdout.log"


def _dotdot(run: Run) -> Path:
    return run.out / ".." / "job.json"


def _link_to_the_key(run: Run) -> Path:
    (run.out / "coin.json").symlink_to(run.cdir / CU.CRED_FILE)
    return run.out / "coin.json"


def _linked_directory(run: Run) -> Path:
    (run.tmp_path / "elsewhere").mkdir()
    (run.tmp_path / "elsewhere" / "coin.json").write_text(COIN)
    (run.out / "d").symlink_to(run.tmp_path / "elsewhere")
    return run.out / "d" / "coin.json"


def _directory(run: Run) -> Path:
    run.file("d/coin.json")
    return run.out / "d"


def _fifo(run: Run) -> Path:
    os.mkfifo(run.out / "pipe")
    return run.out / "pipe"


def _absent(run: Run) -> Path:
    return run.out / "coin.json"


def _too_big(run: Run) -> Path:
    return run.file("big.bin", b"\0" * (RP.MAX_BYTES + 1))


@pytest.mark.parametrize("make, says", [
    (_outside, "is not inside this run's out/ ({out}). The runner preserves only what the run's own record will hold: write the file "
               "under $OUT and name it there"),
    (_dotdot, "is not inside this run's out/ ({out}): it resolves to {d}/job.json. The runner preserves only"),
    (_link_to_the_key, "is not inside this run's out/ ({out}): it resolves to {cdir}/cred.json. The runner preserves only"),
    (_linked_directory, "is not inside this run's out/ ({out}): it resolves to {tmp}/elsewhere/coin.json. The runner preserves only"),
    (_directory, "out/d is not a regular file: name one file. A directory goes as one file (tar it), or is left to the run's custody"),
    (_fifo, "out/pipe is not a regular file: name one file."),
    (_absent, "the runner can't open out/coin.json (No such file or directory): write the file and close it, then ask"),
    (_too_big, "out/big.bin is 1048577 bytes, over the 1048576 bytes (1 MiB) a file preserved while the run goes on may have: "
               "preserve the small record that can't wait"),
], ids=["outside", "dotdot", "link-to-the-key", "linked-directory", "directory", "fifo", "absent", "too-big"])
def test_a_path_that_is_not_a_small_regular_file_of_out_is_refused_with_what_to_do(run, make, says):
    rc, out, err = run.ask(make(run))
    (answer,) = _answers(run.d).values()
    assert (rc, out, err) == (1, "", [f"REFUSED: {answer['reason']}"]) and answer["verdict"] == "REFUSED"
    assert says.format(out=run.out, d=run.d, cdir=run.cdir, tmp=run.tmp_path) in answer["reason"]
    assert not run.vault.exists(), "nothing was put"
    assert not [k for k in os.environ if k in CU.KEY_ENV]


def test_a_file_of_exactly_the_most_bytes_is_preserved_and_one_that_grows_past_them_is_not(run, tmp_path):
    rc, out, _ = run.ask(run.file("full.bin", b"\1" * RP.MAX_BYTES))
    assert rc == 0 and (run.vault / object_key(hashlib.sha256(b"\1" * RP.MAX_BYTES).hexdigest())).stat().st_size == RP.MAX_BYTES
    with pytest.raises(RP.Refusal, match="out/log is still growing, over the 1048576 bytes"):
        RP._copy(io.BytesIO(b"\1" * (RP.MAX_BYTES + 1)), tmp_path / "copy", "out/log")


@pytest.mark.parametrize("swap", ["file", "directory"])
def test_a_link_put_in_the_files_place_after_its_path_was_checked_is_not_followed(run, monkeypatch, swap):
    """What `realpath` resolved is opened one component at a time with no link followed: here the path is a link all along
    and `realpath` answers as it would have before the swap."""
    if swap == "file":
        path = _link_to_the_key(run)
        says = "the runner can't open out/coin.json (Too many levels of symbolic links): it became a link after its path was checked"
    else:
        path = _linked_directory(run)
        says = "the runner can't open out/d/coin.json (Not a directory): a directory on its path is not one, or became a link"
    run.request("a1.json", path)
    with monkeypatch.context() as m:
        m.setattr(RP.os.path, "realpath", lambda p: str(p))
        run.pres.poll()
    assert _answers(run.d)["a1.json"]["verdict"] == "REFUSED" and says in _answers(run.d)["a1.json"]["reason"]
    assert not run.vault.exists()


@pytest.mark.parametrize("request_file", ["not JSON", '{"schema": "research/run-preserve-request/v1", "path": 7, "kind": "run-file/v1", "meta": {}}',
                                          '{"schema": "research/run-preserve-request/v1", "path": "/x", "kind": "no kind", "meta": {}}', "link"],
                         ids=["not-json", "path-not-a-string", "kind-not-a-kind", "a-link"])
def test_a_request_the_command_did_not_write_is_refused_unread(run, request_file):
    run.file()
    if request_file == "link":
        (run.d / RP.REQUESTS / "a1.json").symlink_to(run.cdir / CU.CRED_FILE)
    else:
        (run.d / RP.REQUESTS / "a1.json").write_text(request_file)
    run.pres.poll()
    answer = _answers(run.d)["a1.json"]
    assert answer["verdict"] == "REFUSED" and answer["reason"].endswith("ask with `research run-preserve PATH`")
    assert not run.vault.exists() and not [n for n in _key(run.cdir, run.vault) if n in json.dumps(answer)]


def _withdraw(run: Run, art: str, *path: str) -> None:
    """Withdraw from another machine's store, so the runner learns of it only from the remote."""
    cfg = run.tmp_path / "control.toml"
    cfg.write_text(f'[remote]\ntype = "fs"\npath = "{run.vault}"\n')
    with run.monkeypatch.context() as m:
        m.setenv("RESEARCH_STORE_CONFIG", str(cfg))
        m.setenv("RESEARCH_WRITE_THROUGH", "1")
        rc = data_main(["withdraw", art, "--reason", "holds a seed", "--ref", "X-1", "--as", "infra", *path, "--cache",
                        str(run.tmp_path / "cache"), "--apply", "--store", str(run.tmp_path / "control")])
    run.capsys.readouterr()
    assert rc == 0


def test_a_file_withdrawn_after_it_was_preserved_is_not_put_back(run):
    rc, out, _ = run.ask(run.file())
    art = out.strip()
    assert rc == 0 and run.ask(run.file("other.json", '{"other": 1}'))[0] == 0
    _withdraw(run, art)
    assert not (run.vault / object_key(SHA)).exists()

    rc, out, err = run.ask(run.file())
    assert (rc, out) == (1, "") and err == [
        f"REFUSED: out/coin.json was withdrawn from this run's {art} on purpose ('holds a seed', by infra, ref X-1), and nothing puts "
        "it back in the shared store: delete the file, or write what may be shared under another name and preserve that"]
    assert not (run.vault / object_key(SHA)).exists()
    assert run.ask(run.file("sub/coin.json", '{"coin": "another"}'))[0] == 0, "the same name elsewhere in out/ is another file"


def test_a_file_withdrawn_from_the_runs_record_is_refused_as_data_put_refuses_it(run, monkeypatch):
    """A run whose record is already on the remote, with out/coin.json withdrawn from it: the runner's store holds nothing of
    the run and finds the record through its attempt."""
    tree = run.tmp_path / "tree"
    (tree / "out").mkdir(parents=True)
    (tree / "out" / "coin.json").write_text(COIN)
    (tree / "out" / "kept.txt").write_text("kept\n")
    st = _other_machine(run.tmp_path, run.vault, monkeypatch)
    record = st.put_artifact("run-record/v1", {"run_id": FIXTURE_RID}, tree=tree)
    assert st.push(record)["preserved"]
    st.put_attempt(_attempt(run_id=FIXTURE_RID, outputs={"run_record": record}))
    st.push_attempt(FIXTURE_RID)
    _withdraw(run, record, "--path", "out/coin.json")

    rc, _, err = run.ask(run.file())
    assert rc == 1 and err[0].startswith(f"REFUSED: out/coin.json was withdrawn from this run's {record} on purpose ('holds a seed', by infra")
    assert run.ask(run.file("kept.txt", "kept\n"))[0] == 0


class Canned:
    """A child that answers at once, without a put."""

    def __init__(self, argv: list[str], **kw):
        claim(Path(argv[3]) / RP.ANSWERS / argv[5], {"schema": RP.ANSWER_SCHEMA, "verdict": RP.REFUSED, "reason": "canned"}, mode=0o644)

    def poll(self) -> int:
        return 0


def test_a_run_makes_at_most_the_stated_number_of_requests(tmp_path, monkeypatch, capsys):
    run = Run(tmp_path, monkeypatch, capsys, popen=Canned)
    (run.d / RP.REQUESTS / "NOT A REQUEST").write_text("x")        # counts as one, and starts nothing
    run.pres.poll()
    assert _answers(run.d) == {}
    for _ in range(RP.MAX_REQUESTS - 1):
        assert run.ask(run.file()) == (1, "", ["REFUSED: canned"])
    assert len(_answers(run.d)) == RP.MAX_REQUESTS - 1 and not (run.d / RP.FULL).exists()

    over = (1, "", ["REFUSED: this run has made 64 requests, the most a run may: keep such records in one file and preserve it at the "
                    "points that matter; everything else in out/ the run's custody records when the run ends"])
    assert run.ask(run.file()) == over
    assert len(os.listdir(run.d / RP.REQUESTS)) == RP.MAX_REQUESTS + 1, "the request that found the run full"
    assert run.ask(run.file()) == over
    assert len(os.listdir(run.d / RP.REQUESTS)) == RP.MAX_REQUESTS + 1, "and once it says so, none is written"
    assert sorted(_answers(run.d))[-1] == "full" and len(_answers(run.d)) == RP.MAX_REQUESTS


def test_a_request_after_the_workload_has_ended_is_refused(run):
    run.file()
    run.request("a1.json", run.out / "coin.json")       # written as the workload ended: the loop never saw it
    run.pres.close()
    ended = {"schema": RP.ANSWER_SCHEMA, "verdict": "REFUSED", "reason": RP.ENDED}
    assert _answers(run.d) == {"a1.json": {**ended, "request": "a1"}, "closed": ended}
    assert run.ask(run.out / "coin.json") == (1, "", [f"REFUSED: {RP.ENDED}"])
    assert os.listdir(run.d / RP.REQUESTS) == ["a1.json"], "a command that finds the run closed leaves no request"
    run.request("a2.json", run.out / "coin.json")
    run.pres.poll()
    run.pres.close()
    assert sorted(_answers(run.d)) == ["a1.json", "closed"] and not run.vault.exists() and Child.started == []
    assert "ask before the workload exits" in RP.ENDED


# ---- when the runner can't --------------------------------------------------------------------------------------------------------------

def test_a_remote_out_of_reach_is_not_preserved_and_the_answer_does_not_name_it(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(remote_s3, "RETRY_BACKOFF_S", 0.0)
    run = Run(tmp_path, monkeypatch, capsys, remote_toml=S3_NOWHERE)
    rc, out, err = run.ask(run.file())
    (answer,) = _answers(run.d).values()
    assert (rc, out, err) == (1, "", [f"NOT_PRESERVED: {answer['reason']}"])
    assert answer == {"schema": RP.ANSWER_SCHEMA, "request": answer["request"], "verdict": "NOT_PRESERVED", "reason": answer["reason"]}
    assert answer["reason"] == ("the runner's put did not reach the store's remote: ask again, or stop if the workload must not go on "
                                "without the record; it stays in out/, which the run's custody records when the run ends. The error is in "
                                "the runner's preserve.log, beside the run's custody key")
    named = ["127.0.0.1", "bucket-of-the-run", "ConnectionRefusedError", *_key(run.cdir, run.vault)[:3]]
    assert _holding(run.d, named) == []
    assert "ConnectionRefusedError: [Errno 111] Connection refused" in (run.cdir / RP.LOG).read_text(), "the error is the runner's to read"


def test_an_expired_custody_key_is_named_with_the_flag_that_avoids_it(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(remote_s3, "RETRY_BACKOFF_S", 0.0)
    run = Run(tmp_path, monkeypatch, capsys, remote_toml=S3_NOWHERE, expires="2020-01-01T00:00:00Z")
    rc, out, err = run.ask(run.file())
    assert (rc, out) == (1, "") and err[0].startswith(
        "NOT_PRESERVED: the run's custody key expired at 2020-01-01T00:00:00Z (the run outlived --custody-ttl), so its runner can put "
        "nothing more: launch the next run with a longer --custody-ttl.")


def test_a_put_the_remote_is_not_checked_to_hold_is_never_called_preserved(run, monkeypatch):
    monkeypatch.setattr(P, "check", lambda st, targets, mode: {"preserved": False, "not_preserved": [{"errors": ["absent"]}]})
    rc, out, err = run.ask(run.file())
    assert (rc, out) == (1, "") and err[0].startswith("NOT_PRESERVED: the store's remote does not hold it after the runner's put: ask again")
    assert "PRESERVED" not in json.dumps([a for a in _answers(run.d).values()]).replace("NOT_PRESERVED", "")


class Stuck:
    """A child that never exits unless killed."""

    def __init__(self, argv: list[str], **kw):
        self.name, self.killed, self.waited = argv[5], False, []
        Stuck.made.append(self)

    made: list[Stuck] = []

    def poll(self) -> int | None:
        return -9 if self.killed else None

    def wait(self, timeout: float | None = None) -> int:
        self.waited.append(timeout)
        if not self.killed:
            raise subprocess.TimeoutExpired("put", timeout)
        return -9

    def kill(self) -> None:
        self.killed = True


def test_a_put_that_does_not_finish_is_stopped_at_its_deadline_and_answered_for(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(Stuck, "made", [])
    run = Run(tmp_path, monkeypatch, capsys, popen=Stuck)
    run.request("a1.json", run.file())
    run.request("a2.json", run.file())
    run.pres.poll()
    run.clock.now = RP.PUT_DEADLINE_S - RP.POLL_S
    run.pres.poll()
    (first,) = Stuck.made
    assert first.name == "a1.json" and not first.killed and _answers(run.d) == {}, "one at a time, and each has its whole deadline"

    run.clock.now = RP.PUT_DEADLINE_S
    run.pres.poll()
    assert first.killed and [c.name for c in Stuck.made] == ["a1.json", "a2.json"]
    assert _answers(run.d) == {"a1.json": {
        "schema": RP.ANSWER_SCHEMA, "request": "a1", "verdict": "NOT_PRESERVED",
        "reason": "the runner's put did not finish within 60 s and was stopped (the store's remote is slow or out of reach from this "
                  "machine): ask again, or stop if the workload must not go on without the record; it stays in out/, which the run's "
                  "custody records when the run ends"}}

    run.clock.now += 20
    run.pres.close()        # the workload ended: the put in flight gets the rest of its deadline and no more
    second = Stuck.made[1]
    assert second.waited[0] == RP.PUT_DEADLINE_S - 20 and second.killed
    assert _answers(run.d)["a2.json"]["verdict"] == "NOT_PRESERVED" and _answers(run.d)["closed"]["reason"] == RP.ENDED


def test_the_command_gives_up_at_its_timeout_when_no_runner_answers(run, monkeypatch):
    monkeypatch.setattr(RP, "time", run.clock.module)       # nothing steps the runner's loop
    rc, out, err = run.ask(run.file(), "--timeout", "5")
    (name,) = os.listdir(run.d / RP.REQUESTS)
    assert (rc, out) == (1, "") and run.clock.now == 5.0
    assert err == ["NOT_PRESERVED: no answer from this run's runner within 5 s (it takes one request at a time and gives each put 60 s), so "
                   "nothing says the file is preserved; it stays in out/, which the run's custody records when the run ends. Its answer, "
                   f"if one comes, will be {run.d / RP.ANSWERS / name}"]


class Died:
    def __init__(self, argv: list[str], **kw):
        pass

    def poll(self) -> int:
        return 3


def test_a_child_that_dies_without_an_answer_is_answered_for(tmp_path, monkeypatch, capsys):
    run = Run(tmp_path, monkeypatch, capsys, popen=Died)
    rc, out, err = run.ask(run.file())
    assert (rc, out) == (1, "") and err == [
        "NOT_PRESERVED: the runner's put ended with status 3 and no answer; it stays in out/, which the run's custody records when the "
        "run ends. Its output is in the runner's preserve.log, beside the run's custody key"]


def test_a_run_that_never_asked_keeps_nothing_of_the_offer(run):
    before = CU.run_files(run.d)
    run.pres.close()
    assert not (run.d / RP.DIR).exists() and CU.run_files(run.d) == before
    assert run.ask(run.file()) == (1, "", [f"NOT_PRESERVED: {RP.NO_OFFER}. {run.out / 'coin.json'} stays where it is"])
    assert "or its workload has ended (ask before the workload exits)" in RP.NO_OFFER


def test_a_runner_that_cannot_start_its_child_says_so(tmp_path, monkeypatch, capsys):
    def popen(argv, **kw):
        raise OSError(12, "Cannot allocate memory")

    run = Run(tmp_path, monkeypatch, capsys, popen=popen)
    rc, out, err = run.ask(run.file())
    assert (rc, out) == (1, "") and err[0].startswith("NOT_PRESERVED: the runner could not start its put (Cannot allocate memory); it stays")


def test_outside_a_run_the_command_says_what_it_is_for(tmp_path, capsys):
    assert RP.main([str(tmp_path / "coin.json")], environ={}) == 1
    assert capsys.readouterr().err == "research run-preserve: RESEARCH_RUN_DIR is not set: this is for a workload that `research run` started\n"
    with pytest.raises(SystemExit) as e:
        RP.main([str(tmp_path / "coin.json"), "--kind", "no kind"], environ={})
    assert e.value.code == 2 and "--kind 'no kind' is not a kind" in capsys.readouterr().err


# ---- the run's custody at the end, and the loop that carries the hook --------------------------------------------------------------------------

def test_the_runs_custody_at_the_end_passes_and_does_not_upload_the_preserved_blob_again(run, monkeypatch):
    puts, real = [], FsRemote.put

    def put(self, key, path):
        rep = real(self, key, path)
        puts.append((key, rep["action"]))
        return rep

    monkeypatch.setattr(FsRemote, "put", put)
    rc, out, _ = run.ask(run.file())
    assert rc == 0 and run.ask(run.d / "job.json")[0] == 1
    run.pres.close()
    asked = sorted(os.listdir(run.d / RP.REQUESTS))
    kept, trimmed = CU.select(run.d, 0)
    assert [t["path"] for t in trimmed if t["path"].startswith(("out/", str(RP.ANSWERS)))] == []
    assert sorted(t["path"] for t in trimmed if t["path"].startswith("control/")) == [f"{RP.REQUESTS}/{n}" for n in asked], \
        "a request is the workload's file, as any other it writes outside out/"

    rep = CU.runner_publish(run.d, run.cdir, sleep=lambda s: None, tries=1)
    assert rep["preserved"] is True and CU.marker_problem(run.d) is None
    assert [a for k, a in puts if k == object_key(SHA)] == ["created", "skipped"]
    other = _other_machine(run.tmp_path, run.vault, monkeypatch)
    record = other.get_manifest(other.get_attempt(FIXTURE_RID)["outputs"]["run_record"])
    held = {f.path: f.sha256 for f in record.payload.files}
    assert held["out/coin.json"] == SHA and not record.meta.get("trimmed")
    assert {f"{RP.ANSWERS}/{n}" for n in _answers(run.d)} | {f"{RP.REQUESTS}/{n}" for n in asked} <= set(held)
    assert other.get_manifest(out.strip()).payload.sha256 == SHA


def test_a_poll_that_raises_is_dropped_with_a_mark_and_the_workload_runs_on(tmp_path):
    d, gate, calls = tmp_path / "r1", Gate(tmp_path / "gate"), []
    ap = argparse.ArgumentParser()
    trun.add_args(ap)

    def poll():
        calls.append(1)
        gate.release()
        raise RuntimeError("the watch broke")

    try:
        a = ap.parse_args(["--attempt-dir", str(d), "--stage", "t", "--no-sampler", "--no-classify"])
        assert trun.run(a, [PY, "-c", Gate.WAIT, str(gate.path)], poll) == 0
    finally:
        gate.close()
    marks = [json.loads(x) for x in (d / "events.jsonl").read_text().splitlines() if '"poll_failed"' in x]
    assert len(calls) == 1 and [m["error"] for m in marks] == ["RuntimeError: the watch broke"]


def test_under_a_job_user_only_requests_is_the_jobs_to_write_and_only_while_it_runs(tmp_path):
    d, cdir = tmp_path / "run", tmp_path / "custody"
    cdir.mkdir()
    pres = RP.Preserver(d, cdir, shared=True)
    mode = {p: stat.S_IMODE((d / p).stat().st_mode) for p in ("control", RP.DIR, RP.ANSWERS, RP.REQUESTS)}
    assert mode == {"control": 0o755, RP.DIR: 0o755, RP.ANSWERS: 0o755, RP.REQUESTS: 0o1777}
    claim(d / RP.REQUESTS / "a1.json", {}, mode=0o644)
    claim(d / "private.json", {})
    assert stat.S_IMODE((d / RP.REQUESTS / "a1.json").stat().st_mode) == 0o644 and stat.S_IMODE((d / "private.json").stat().st_mode) == 0o600
    pres.close()
    assert stat.S_IMODE((d / RP.REQUESTS).stat().st_mode) == 0o755
    assert {stat.S_IMODE(p.stat().st_mode) for p in (d / RP.ANSWERS).iterdir()} == {0o644}
