"""`research data withdraw` (`research.store.withdraw`, README §7.4): a withdrawn artifact's blobs leave the remote even when it is
cited, its manifest stays, and a blob another artifact also holds is kept and named."""
from __future__ import annotations

import pytest

from research.store import gc, withdraw
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
