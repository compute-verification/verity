"""`research data withdraw` (`research.store.withdraw`, README §7.4): a withdrawn artifact's blobs leave the remote even when it is
cited, its manifest stays, and a blob another artifact also holds is kept and named."""
from __future__ import annotations

import pytest

from research.store import canon, gc, ids, withdraw
from research.store.remote import FsRemote
from test_store_gc import artifact, blob, env, label, present  # noqa: F401  (env is a fixture)


def test_a_cited_artifact_holding_private_values_loses_its_blobs_and_keeps_its_manifest(env):
    """Three cited arts held the sealed bundle and the weights key; gc keeps cited arts, so only withdraw removes them."""
    b, _, cache, _ = env
    secret, public = blob(b, b"weights key"), blob(b, b"a README both trees ship")
    leaked = artifact(b, "evidence/v1", secret, public)
    other = artifact(b, "evidence/v1", public)
    label(b, leaked)
    p = withdraw.plan(FsRemote(b), cache, [leaked])
    assert p.blobs == {secret[0]: secret[1]} and p.shared == {public[0]: [other]}, "a blob another art holds is kept and named"
    assert present(b, secret[0]), "planning deletes nothing"
    assert withdraw.apply(FsRemote(b), p) == {"deleted": 1, "absent": 0}
    assert not present(b, secret[0]) and present(b, public[0])
    assert (b / "manifests" / f"{leaked.removeprefix('art:')}.json").exists(), "the citation still resolves"


def test_an_unknown_or_short_id_deletes_nothing(env):
    b, _, cache, _ = env
    kept = blob(b, b"kept")
    artifact(b, "evidence/v1", kept)
    p = withdraw.plan(FsRemote(b), cache, ["art:" + "0" * 64])
    assert p.missing == ["art:" + "0" * 64] and not p.blobs
    with pytest.raises(gc.GcError):
        withdraw.apply(FsRemote(b), p)
    with pytest.raises(ValueError, match="64 hex"):
        withdraw.plan(FsRemote(b), cache, ["art:abcd1234"])
    assert present(b, kept[0])


def test_a_path_withdraws_only_the_files_under_it_and_keeps_a_blob_the_rest_also_holds(env):
    """A run's attempt record published one directory of private values (an unpadded advice length) beside public files."""
    b, _, cache, _ = env
    private, public, both = blob(b, b"advice bits 392"), blob(b, b"verdicts.txt"), blob(b, b"a line both dirs hold")
    files = [{"path": "out/sampled-proofs/served/summary.json", "sha256": private[0], "bytes": private[1]},
             {"path": "out/sampled-proofs/served/audit.log", "sha256": both[0], "bytes": both[1]},
             {"path": "out/verdicts.txt", "sha256": public[0], "bytes": public[1]},
             {"path": "out/sampled-proofs/served-records/x", "sha256": both[0], "bytes": both[1]}]
    payload = {"type": "tree", "sha256": canon.sha256(files), "bytes": sum(f["bytes"] for f in files), "files": files}
    doc = {"schema": "research/artifact/v0.1", "kind": "evidence/v1", "meta": {}, "refs": {}, "payload": payload}
    aid = ids.art_id(doc)
    (b / "manifests").mkdir(parents=True, exist_ok=True)
    (b / "manifests" / f"{ids.hex_of(aid)}.json").write_bytes(canon.dumps(doc))
    p = withdraw.plan(FsRemote(b), cache, [aid], paths=["out/sampled-proofs/served"])
    assert p.blobs == {private[0]: private[1]}, "only the files under the prefix, not served-records/ beside it"
    assert p.shared == {both[0]: [f"{aid} (outside --path)"]}, "a blob the art also holds outside the prefix is kept and named"
    assert p.summary()["paths"] == ["out/sampled-proofs/served"]
    withdraw.apply(FsRemote(b), p)
    assert not present(b, private[0]) and present(b, public[0]) and present(b, both[0])
