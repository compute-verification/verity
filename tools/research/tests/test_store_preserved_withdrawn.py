"""`research data preserved` on a withdrawn artifact (`research data withdraw --path`, store/withdraw.py): a blob a `withdrawn` label
covers reads WITHDRAWN with the label's reason, never absent, and exits 4; a blob outside the withdrawn paths that is absent still
reads NOT preserved.  `research data put --tree` with that run's id refuses a tree holding a withdrawn file."""
from __future__ import annotations

import json

import pytest

from research.store import preserved as P
from research.store.cli import main as data_main
from research.store.local import LocalStore, object_key

from test_store import _attempt

RUN = "r20261007-040000-0001"
SERVED = "out/sampled-proofs/served"


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(tmp_path / "store.toml"))
    monkeypatch.setenv("RESEARCH_WRITE_THROUGH", "1")
    (tmp_path / "store.toml").write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "vault"}"\n')
    tree = tmp_path / "run"
    (tree / SERVED).mkdir(parents=True)
    (tree / SERVED / "summary.json").write_text('{"advice_bits": 392}')
    (tree / "out" / "verdicts.txt").write_text("accepted\n")
    st = LocalStore()
    art = st.put_artifact("run-record/v1", {"run_id": RUN}, tree=tree)
    assert st.push(art)["preserved"]
    st.put_attempt(_attempt(run_id=RUN, outputs={"run_record": art}))
    st.push_attempt(RUN)
    return tmp_path, st, art, tree


def _withdraw(tmp_path, art: str) -> None:
    """Withdraw the served directory from another store (the control pod's), so this store learns of it only from the remote."""
    rc = data_main(["withdraw", art, "--reason", "unpadded advice length", "--ref", "INF-1", "--as", "infra", "--path", SERVED,
                    "--cache", str(tmp_path / "cache"), "--apply", "--store", str(tmp_path / "control")])
    assert rc == 0


def _key(st: LocalStore, art: str, path: str) -> str:
    return object_key(next(f.sha256 for f in st.get_manifest(art).payload.files if f.path == path))


def test_an_absent_blob_under_a_withdrawn_path_reads_withdrawn_with_the_labels_reason(env, capsys):
    tmp_path, st, art, _ = env
    _withdraw(tmp_path, art)
    capsys.readouterr()
    assert st.labels(art) == [], "the label is on the remote only, until preserved pulls it"
    rep = P.check(st, [RUN])
    a = rep["artifacts"][art]
    assert a["verdict"] == P.WITHDRAWN and a["errors"] == [] and a["preserved"] is False
    assert a["withdrawn"] == [{"key": _key(st, art, f"{SERVED}/summary.json"), "paths": [f"{SERVED}/summary.json"], "under": [SERVED],
                               "reason": "unpadded advice length", "by": "infra", "ref": "INF-1"}]
    assert rep["verdict"] == rep["targets"][0]["verdict"] == P.WITHDRAWN and rep["preserved"] is False
    assert [x["key"] for x in rep["withdrawn"]] == [a["withdrawn"][0]["key"]]
    assert all("absent" not in e for x in rep["not_preserved"] for e in x["errors"]), "withdrawn, never absent"
    assert "WITHDRAWN under out/sampled-proofs/served ('unpadded advice length', by infra, ref INF-1)" in rep["not_preserved"][0]["errors"][0]

    assert data_main(["preserved", RUN, art]) == 4
    out = capsys.readouterr().out.splitlines()
    assert any(ln.startswith("  outputs.run_record") and "WITHDRAWN: 1 file(s) WITHDRAWN under out/sampled-proofs/served" in ln for ln in out)
    assert out[-1].startswith("WITHDRAWN: everything else is PRESERVED") and P.DELETE_COPIES in out[-1]
    assert data_main(["preserved", art, "--json"]) == 4
    assert json.loads(capsys.readouterr().out)["verdict"] == "WITHDRAWN"


def test_an_absent_blob_outside_the_withdrawn_paths_still_reads_not_preserved(env, capsys):
    tmp_path, st, art, _ = env
    _withdraw(tmp_path, art)
    lost = _key(st, art, "out/verdicts.txt")
    (tmp_path / "vault" / lost).unlink()
    rep = P.check(st, [RUN])
    a = rep["artifacts"][art]
    assert a["verdict"] == rep["verdict"] == P.NOT_PRESERVED
    assert a["errors"] == [f"{lost}: absent"] and [x["paths"] for x in a["withdrawn"]] == [[f"{SERVED}/summary.json"]]
    capsys.readouterr()
    assert data_main(["preserved", RUN]) == 1
    last = capsys.readouterr().out.splitlines()[-1]
    assert last.startswith("NOT PRESERVED: 1 item(s)") and "1 file(s) were withdrawn on purpose" in last


def test_an_absent_blob_without_a_withdrawn_label_is_not_preserved(env):
    tmp_path, st, art, _ = env
    (tmp_path / "vault" / _key(st, art, f"{SERVED}/summary.json")).unlink()
    a = P.check(st, [art])["artifacts"][art]
    assert a["verdict"] == P.NOT_PRESERVED and a["withdrawn"] == [] and a["errors"][0].endswith(": absent")


def test_put_tree_refuses_a_file_withdrawn_from_the_same_run(env, capsys):
    tmp_path, st, art, tree = env
    _withdraw(tmp_path, art)
    (tree / "out" / "late.txt").write_text("written after the record")
    n = len(st.artifact_ids())
    capsys.readouterr()
    assert data_main(["put", "--kind", "run-record/v1", "--meta", json.dumps({"run_id": RUN}), "--tree", str(tree)]) == 1
    err = capsys.readouterr().err
    assert f"{SERVED}/summary.json: withdrawn from {art} ('unpadded advice length', by infra, ref INF-1)" in err and "REFUSED" in err
    assert len(st.artifact_ids()) == n, "nothing stored"
    (tree / SERVED / "summary.json").unlink()
    assert data_main(["put", "--kind", "run-record/v1", "--meta", json.dumps({"run_id": RUN}), "--tree", str(tree)]) == 0
