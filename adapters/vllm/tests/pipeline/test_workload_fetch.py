"""`verity-vllm workload-file` fetches a stored workload by the path the registry (fixtures/artifacts.json) gives it, from whatever
directory it runs in: `research data fetch-fixtures` reads a relative PATH from its working directory, not from --repo."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import subprocess
from pathlib import Path

from research.store import fetchset, fixtures as F
from research.store.cli import main as data_main
from research.store.local import LocalStore, manifest_key, object_key
from research.store.remote import FsRemote

from verity_vllm import config as C
from verity_vllm.pipeline import cli


def test_the_fetch_resolves_the_same_registry_entries_from_the_integration_and_from_the_repository_root(tmp_path, monkeypatch, capsys):
    """Run from the integration (where its shell scripts run it) or from the repository root, it fetches the same registry
    entries. The real command runs in this process, with a directory remote in place of the evidence store's remotes."""
    repo = tmp_path / "repo"
    wl = repo / "adapters" / "vllm" / "workloads"
    wl.mkdir(parents=True)
    data = {n: json.dumps({"x": i}).encode() for i, n in enumerate(("a.json", "b.json"))}
    (wl / C.DIGESTS).write_text("".join(f"{hashlib.sha256(b).hexdigest()}  {n}\n" for n, b in data.items()))
    origin, remote, registry = LocalStore(tmp_path / "origin"), FsRemote(tmp_path / "remote"), {}
    for n, b in data.items():
        (wl / n).write_bytes(b)
        art = origin.put_artifact(F.FIXTURE_KIND, {}, None, file=wl / n)
        for h, _ in origin.get_manifest(art).blobs():
            remote.put(object_key(h), origin.object_path(h))
        remote.put(manifest_key(art.split(":", 1)[1]), origin.manifest_path(art.split(":", 1)[1]))
        registry[art] = f"adapters/vllm/workloads/{n}"
    (repo / "fixtures").mkdir()
    (repo / "fixtures" / "artifacts.json").write_text(json.dumps(registry))
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "store"))
    monkeypatch.setattr(fetchset, "sources", lambda st, **kw: [remote])
    monkeypatch.setattr(F, "tracked", lambda repo: None)
    seen = []

    def run(argv, **kw):
        assert argv[1:5] == ["-m", "research", "data", "fetch-fixtures"], argv
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = data_main(argv[4:])
        seen.append(sorted((e["path"], e["status"]) for e in json.loads(out.getvalue())["entries"]))
        return subprocess.CompletedProcess(argv, rc, out.getvalue(), "")
    monkeypatch.setattr(C.subprocess, "run", run)
    for cwd, rel in ((wl.parent, Path("workloads")), (repo, Path("adapters/vllm/workloads"))):
        for n in data:
            (wl / n).unlink(missing_ok=True)
        monkeypatch.chdir(cwd)
        assert cli.main(["workload-file", *(str(rel / n) for n in data)]) == 0
        assert capsys.readouterr().out == "".join(f"{rel / n}\n" for n in data)
        assert {n: (wl / n).read_bytes() for n in data} == data
    assert seen == [[(p, "fetched")] for p in registry.values()] * 2
