"""reverify.py against a temp store (filesystem remote) and a fake `ligero-verify` (a script that answers system-digest / batch as
told through env vars): PASS writes a preserved verdict and labels; broken custody, a wrong pin and a verifier rejection are FAILs
that write nothing; a dump that never left the pod is an ERROR, not a rejection; the verifier's tag is derived, never typed."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from backends.direct.ligero import reverify as R

RUN = "r20260923-104729-92cb"
FAKE = textwrap.dedent(f"""\
    #!{sys.executable}
    import json, os, sys
    a = sys.argv[1:]
    if a[0] == "system-digest":
        print(json.dumps({{"pinned_relation": os.environ.get("FAKE_PIN") or None, "sys_id": "x"}}))
    elif a[0] == "batch":
        n = len([p for p in os.listdir(a[a.index("--dir") + 1]) if p.endswith(".proof")])
        ok = os.environ.get("FAKE_BATCH", "accept") == "accept"
        doc = {{"n": n, "accepted": n if ok else n - 1, "rejected": 0 if ok else 1, "batch_accepted": ok, "system_pinned": True,
               "batch_reason": "accept" if ok else "reject: sub_00", "batch_bits": 128.3, "python_disagree": 0, "verify_seconds_sum": 0.5}}
        open(a[a.index("--json") + 1], "w").write(json.dumps(doc))
    """)


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "store"))
    (tmp_path / "store.toml").write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "vault"}"\n')
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(tmp_path / "store.toml"))
    monkeypatch.setenv("FAKE_PIN", "fp8-ada")
    monkeypatch.delenv("FAKE_BATCH", raising=False)
    fake = tmp_path / "bin" / "ligero-verify"
    fake.parent.mkdir()
    fake.write_text(FAKE)
    fake.chmod(0o755)
    return tmp_path


def _dump(root: Path, *, n: int = 2, corrupt: bool = False, flat: bool = False) -> Path:
    p = root if flat else root / "proofs"
    (p / "rep1").mkdir(parents=True)
    files = []
    for i in range(n):
        proof, stmt, coins = f"proof-{i}".encode(), f"stmt-{i}".encode(), f"coins-{i}".encode()
        for ext, data in (("proof", proof), ("stmt", stmt), ("coins", coins)):
            (p / f"rep1/sub_{i:02d}.{ext}").write_bytes(data)
        files.append({"rep": 1, "sub": i, "proof": f"rep1/sub_{i:02d}.proof", "proof_sha256": _sha(proof), "stmt": f"rep1/sub_{i:02d}.stmt",
                      "stmt_bytes": len(stmt), "coins": f"rep1/sub_{i:02d}.coins", "coins_sha256": _sha(coins)})
    (p / "system.bin").write_bytes(b"system")
    if corrupt:
        (p / "rep1/sub_00.proof").write_bytes(b"tampered")
    (p / "manifest.json").write_text(json.dumps({"relation": {"name": "fp8-ada"}, "authentication": "excluded", "mode": "interactive", "files": files,
                                                 "system_file": {"path": "system.bin", "sha256": _sha(b"system"), "bytes": 6}}))
    return root


def _result(st, tmp_path: Path, name: str, **kw) -> str:
    rf = st.put_artifact("run-files/v1", {}, {}, tree=_dump(tmp_path / name, **kw))
    res = st.put_artifact("bench-result/v1", {"run_id": RUN, "name": name}, {"run_files": rf})
    st.put_attempt({"schema": "research/attempt/v0.1", "id": RUN, "tool": {"name": "bench_vu_fp8", "version": "1"}, "derivation": "drv:" + "a" * 64,
                    "reuse_key": None, "argv": ["x"], "params": {"relation": "fp8-ada", "mode": "interactive"}, "key_params": [], "inputs": {},
                    "conditions": {}, "key_conditions": [], "source": {"commit": "c" * 40, "dirty": False}, "closure": None,
                    "execution": {"start_utc": "2026-09-23T10:47:29Z"}, "cost": None, "status": {"state": "done", "rc": 0, "validation": "passed"},
                    "outputs": {"result": res, "run_files": rf}, "telemetry": None})
    return res


def _run(env: Path, *args: str) -> int:
    return R.main([*args, "--verifier", str(env / "bin" / "ligero-verify"), "--work", str(env / "work"), "--jobs", "1"])


def test_pass_writes_a_preserved_verdict_and_labels(env, capsys):
    from research.store.local import LocalStore

    st = LocalStore()
    res = _result(st, env, "good")
    assert _run(env, res[:14], "--by", "tester") == 0
    out = capsys.readouterr().out
    assert "custody 7/7, pinned=fp8-ada, 2/2, 2^-128.30 PASS -> verdict art:" in out
    labels = {lab.key: lab for lab in LocalStore().labels(res)}
    assert labels["verified"].value == "accepted" and labels["verified"].by == "tester"
    vid = labels["verified"].ref
    m = LocalStore().get_manifest(vid)
    assert m.kind == "verification-verdict/v1" and m.refs["result"] == res and m.meta["run"] == RUN and m.meta["relation"] == "fp8-ada"
    assert m.meta["verifier"].startswith("ligero-verify(sha256 "), "a binary outside any checkout is identified by its hash, never by a typed commit"
    assert "D7" in m.meta["detail"] and LocalStore().verify(vid)["preserved"]
    assert not list((env / "work").glob("*/proofs")), "the fetched tree is removed"


@pytest.mark.parametrize("case, expect", [("corrupt", "custody: rep1/sub_00.proof sha256 mismatch"),
                                          ("pin", "pinned bf16-hopper but manifest says fp8-ada, attempt says fp8-ada"),
                                          ("unpinned", "system is not pinned to any registered relation"),
                                          ("reject", "rep1: batch REJECT (reject: sub_00); rep1: 1 proof(s) rejected")])
def test_failures_write_nothing(env, monkeypatch, capsys, case, expect):
    from research.store.local import LocalStore

    st = LocalStore()
    res = _result(st, env, case, corrupt=case == "corrupt")
    monkeypatch.setenv("FAKE_PIN", {"pin": "bf16-hopper", "unpinned": ""}.get(case, "fp8-ada"))
    monkeypatch.setenv("FAKE_BATCH", "reject" if case == "reject" else "accept")
    assert _run(env, res, "--by", "tester") == 1
    assert f"FAIL: {expect}" in capsys.readouterr().out
    assert LocalStore().labels(res) == [] and not LocalStore().index.select_artifacts(kind="verification-verdict/v1")


def test_a_result_put_by_hand_with_the_benchs_own_run_id_is_reverified(env, capsys):
    from research.store.local import LocalStore

    st = LocalStore()
    rf = st.put_artifact("run-files/v1", {}, {}, tree=_dump(env / "by-hand"))
    res = st.put_artifact("bench-result/v1", {"run_id": "b-ligero-fp8-ada-vu-1790219342"}, {"run_files": rf})
    assert _run(env, res, "--by", "tester") == 0
    assert "2/2, 2^-128.30 PASS -> verdict art:" in capsys.readouterr().out


def test_a_tree_published_as_the_proof_dir_itself_is_reverified(env, capsys):
    from research.store.local import LocalStore

    st = LocalStore()
    res = _result(st, env, "flat", flat=True)
    assert _run(env, res, "--by", "tester") == 0
    assert "custody 7/7, pinned=fp8-ada, 2/2, 2^-128.30 PASS -> verdict art:" in capsys.readouterr().out


def test_a_bench_cell_result_names_its_dump_in_meta_artifacts(env, capsys):
    # verify-bligero-real-k 2026-09-26: bench.cell results keep the dump under sweep/<point>/proofs and name it in artifacts[0]
    from research.store.local import LocalStore

    st = LocalStore()
    tree = env / "cell"
    _dump(tree / "sweep" / "p4-6272")
    rf = st.put_artifact("run-files/v1", {}, {}, tree=tree)
    res = st.put_artifact("bench-result/v1", {"run_id": "b-ligero-real-k-1", "artifacts": ["sweep/p4-6272/proofs"]}, {"run_files": rf})
    assert _run(env, res, "--by", "tester") == 0
    assert "2/2, 2^-128.30 PASS -> verdict art:" in capsys.readouterr().out


def test_a_dump_that_never_left_the_pod_is_an_error_not_a_rejection(env, capsys):
    from research.store.local import LocalStore

    st = LocalStore()
    tree = _dump(env / "partial")
    for p in (tree / "proofs/rep1").iterdir():
        p.unlink()
    rf = st.put_artifact("run-files/v1", {}, {}, tree=tree)
    res = st.put_artifact("bench-result/v1", {"run_id": RUN}, {"run_files": rf})
    assert _run(env, res, "--dry-run") == 1
    assert "ERROR: incomplete dump: 6 of 7 files" in capsys.readouterr().out


# ---- fp4-nvf4+poseidon2 (fp4/hashed.py FP4_HASHED): the committed trees are recomputed through the relation's lane format (a
# 2-VU dump proved on the CPU by the hashed runner the bench drives, trees bound to the relation's input set) ----

FP4_PIN = "fp4-nvf4+poseidon2"


@pytest.fixture(scope="module")
def fp4(tmp_path_factory):
    """``(emit, man)``: ``emit(tag, committed, vus, xi, wi)`` writes a one-rep dump of ``vus`` proved against the trees of
    ``committed`` (VU v on x row ``xi[v]``, W column ``wi[v]``); ``man`` is the manifest the bench writes for the honest set."""
    pytest.importorskip("torch")
    from backends.direct.ligero.fp4.hashed import FP4_HASHED as rel
    from backends.direct.ligero.hashauth import HashAuth
    from backends.direct.ligero.relchain import HashedRelationRunner, instances, instances_digest
    from backends.direct.ligero.serialize import _statement_of, proof_bytes, statement_bytes, system_bytes

    n = 2
    base, digest = instances(rel, n), instances_digest(rel, n)
    R = HashedRelationRunner(rel, "cpu", -132.0, zk=False, mode="fiat-shamir")
    root = tmp_path_factory.mktemp("fp4")

    def emit(tag, committed, vus, xi, wi):
        R.commit_vus(committed, dataset=rel.instances_dataset, tier=rel.instances_tier, manifest_sha256=digest)
        proof, pubs, lay = R.prove_vus(vus, check=False, vu_ids=list(range(n)), auth=HashAuth.for_vus(R.trees, list(range(n)), xi, wi),
                                       coins=None, min_l=256)
        d = root / tag / "proofs"
        (d / "rep1").mkdir(parents=True, exist_ok=True)
        blobs = {"system.bin": system_bytes(R.sys), "rep1/sub_00.stmt": statement_bytes(_statement_of(R, pubs, lay, R.cfg(lay.l), 1)),
                 "rep1/sub_00.proof": proof_bytes(proof)}
        for rel_path, b in blobs.items():
            (d / rel_path).write_bytes(b)
        man = {"relation": rel.name, "authentication": "included-hash", "mode": "fiat-shamir", "set": {"total_vus": n, "instances": digest},
               "files": [{"rep": 1, "sub": 0, "proof": "rep1/sub_00.proof", "proof_sha256": _sha(blobs["rep1/sub_00.proof"]),
                          "stmt": "rep1/sub_00.stmt", "stmt_bytes": len(blobs["rep1/sub_00.stmt"])}],
               "system_file": {"path": "system.bin", "sha256": _sha(blobs["system.bin"]), "bytes": len(blobs["system.bin"])}}
        (d / "manifest.json").write_text(json.dumps(man))
        return d, man

    return emit, base


def test_fp4_committed_trees_are_the_producers(fp4):
    emit, base = fp4
    d, man = emit("honest", base, base, [0, 1], [0, 1])
    assert R.commitment_problems(d, man, FP4_PIN, ["rep1"]) == (True, [])
    assert R.hashed_relation("fp4-nvf4").hash_format is not None and R.hashed_relation("fp8-ada").name == "fp8-ada"


def test_fp4_honest_dump_is_reverified(fp4, env, monkeypatch, capsys):
    from research.store.local import LocalStore

    emit, base = fp4
    d, _ = emit("reverified", base, base, [0, 1], [0, 1])
    monkeypatch.setenv("FAKE_PIN", FP4_PIN)
    st = LocalStore()
    rf = st.put_artifact("run-files/v1", {}, {}, tree=d.parent)
    res = st.put_artifact("bench-result/v1", {"run_id": "b-ligero-fp4-nvf4-vu-1"}, {"run_files": rf})
    assert _run(env, res, "--by", "tester") == 0
    out = capsys.readouterr().out
    assert f"pinned={FP4_PIN}, 1/1, 2^-128.30 PASS -> verdict art:" in out
    vid = {lab.key: lab for lab in LocalStore().labels(res)}["verified"].ref
    assert "hash commitments recomputed from the instance set" in LocalStore().get_manifest(vid).meta["detail"]


def _flip(v, i=0):
    a = list(v[0])
    a[i] ^= 1
    return (a, *v[1:])


def test_fp4_a_flipped_instance_byte_is_refused(fp4):
    emit, base = fp4
    served = [_flip(base[0], 5), base[1]]        # the producer commits and proves a set other than the one it names
    d, man = emit("flipped", served, served, [0, 1], [0, 1])
    assert R.commitment_problems(d, man, FP4_PIN, ["rep1"]) == (True, ["rep1/sub_00.stmt: tree a root differ from the instance set's"])


def test_fp4_a_remapped_vu_is_refused(fp4):
    emit, base = fp4
    served = [(base[0][0], base[0][1], base[0][2], base[1][3]), base[1]]
    d, man = emit("remap", served, [base[1], base[1]], [1, 1], [1, 1])
    assert R.commitment_problems(d, man, FP4_PIN, ["rep1"]) == (True, [
        "rep1/sub_00.stmt: tree y root differ from the instance set's",
        "rep1/sub_00.stmt: auth: a VU's x row / W column is not the one its index fixes in the committed layout"])


@pytest.mark.parametrize("edit, why", [({"total_vus": 3}, "the regenerated fp4-nvf4 set of 3 VUs has manifest digest"),
                                       ({"instances": "0" * 64}, "the dump claims 00000000"),
                                       ({"tile": {"nx": 1, "nw": 2, "seed": 0}}, "commitment: fp4-nvf4: its lane format (FP4Format)")])
def test_fp4_a_wrong_set_block_is_refused(fp4, edit, why):
    emit, base = fp4
    d, man = emit("set", base, base, [0, 1], [0, 1])
    hashed, bad = R.commitment_problems(d, dict(man, set=dict(man["set"], **edit)), FP4_PIN, ["rep1"])
    assert hashed and len(bad) == 1 and why in bad[0], bad


def test_fp4_a_missing_proof_or_a_statement_without_one_is_refused(fp4, tmp_path):
    import shutil

    emit, base = fp4
    d, man = emit("orphans", base, base, [0, 1], [0, 1])
    e = shutil.copytree(d, tmp_path / "missing")
    (e / "rep1/sub_00.proof").unlink()
    assert R.commitment_problems(e, man, FP4_PIN, ["rep1"])[1] == [
        "rep1: 1 statement(s) without a proof, 0 proof(s) without a statement", "rep1: the manifest's proofs are not the rep's proofs"]
    listed = dict(man, files=[*man["files"], {"rep": 1, "sub": 1, "stmt": "rep1/sub_01.stmt"}])
    assert R.commitment_problems(d, listed, FP4_PIN, ["rep1"])[1] == ["rep1: the manifest lists a statement without a proof"]


def test_manifest_relation_as_name_or_block():
    assert R.expected_relations({"relation": {"name": "fp8-ada"}}, {}, None) == {"manifest": "fp8-ada"}
    assert R.expected_relations({"relation": "fp4-nvf4"}, {}, None) == {"manifest": "fp4-nvf4"}    # fp4/chain.py's manifest
    assert R.expected_relations({}, {"relation": "fp4-nvf4"}, "fp4-nvf4") == {"attempt": "fp4-nvf4", "--relation": "fp4-nvf4"}


def test_verifier_tag_comes_from_the_crate_it_was_built_in(tmp_path):
    crate = tmp_path / "repo" / "backends" / "ligero-verify"
    (crate / "src").mkdir(parents=True)
    (crate / "Cargo.toml").write_text("[package]\nname = 'ligero-verify'\n")
    (crate / "src/main.rs").write_text("fn main() {}\n")
    (crate / ".gitignore").write_text("target/\n")                       # as in the real crate
    git = lambda *a: subprocess.run(["git", "-C", str(tmp_path / "repo"), *a], check=True, capture_output=True, text=True).stdout.strip()  # noqa: E731
    git("init", "-q")
    git("-c", "user.email=t@e", "-c", "user.name=t", "add", "-A")
    git("-c", "user.email=t@e", "-c", "user.name=t", "commit", "-q", "-m", "v")
    binary = crate / "target/release/ligero-verify"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"\x7fELF")
    assert R.load_verifier(binary).tag == f"ligero-verify@{git('rev-parse', 'HEAD')[:7]}"
    later = binary.stat().st_mtime + 10
    os.utime(crate / "src/main.rs", (later, later))
    with pytest.raises(SystemExit, match="src/main.rs is newer than the binary"):
        R.load_verifier(binary)
    assert R.load_verifier(binary, allow_stale=True).src_commit is None
