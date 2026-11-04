"""covert_witness's starts_store installs every start its manifest lists, each from the artifact holding it (the
manifest's, or an entry's own, as the v3 ladder's starts), from the store or from `pack`'s one tar, so a fresh node's
prep.sh installs the pilot's starts and v3's together."""
import json

import pytest
import starts_store


def artifact(root, starts: dict[str, bytes]):
    """An exported artifact's tree (`starts/<name>/`) and each start's manifest entry."""
    entries = {}
    for name, theta in starts.items():
        d = root / "starts" / name
        d.mkdir(parents=True)
        (d / "start.json").write_text(json.dumps({"digest": f"digest-{name}"}))
        (d / "theta.safetensors").write_bytes(theta)
        entries[name] = {"digest": f"digest-{name}", "files": {f: starts_store.sha256(d / f) for f in starts_store.FILES}}
    return entries


@pytest.fixture
def two(tmp_path):
    """A manifest whose `art` holds s50, with s-t20 in an artifact of its own, and the store's fetch of both."""
    trees = {"art:pilot": tmp_path / "pilot", "art:ladder": tmp_path / "ladder"}
    starts = artifact(trees["art:pilot"], {"s50": b"pilot"})
    starts |= {n: e | {"art": "art:ladder", "path": "starts"}
               for n, e in artifact(trees["art:ladder"], {"s-t20": b"ladder"}).items()}
    man = {"schema": "covert-witness/starts-store/v1", "art": "art:pilot", "path": "starts", "starts": starts}
    return man, lambda art, tmp: trees[art]


def test_install_takes_each_start_from_its_artifact(tmp_path, two):
    man, fetch = two
    to = tmp_path / "node"
    assert sorted(starts_store.install(to, man, fetch)) == ["s-t20", "s50"]
    assert (to / "s-t20" / "theta.safetensors").read_bytes() == b"ladder"
    assert starts_store.install(to, man, fetch) == []


def test_packs_tar_installs_every_artifacts_starts_on_a_fresh_node(tmp_path, two):
    man, fetch = two
    tar = starts_store.pack(man, tmp_path / "starts.tar", fetch)
    to = tmp_path / "fresh"
    assert sorted(starts_store.install(to, man, starts_store.from_tar(tar))) == ["s-t20", "s50"]
    assert all(starts_store.matches(to / n, man["starts"][n]) for n in man["starts"])


def test_a_file_that_is_not_the_manifests_is_refused_naming_its_artifact(tmp_path, two):
    man, fetch = two
    (fetch("art:ladder", None) / "starts" / "s-t20" / "theta.safetensors").write_bytes(b"other")
    with pytest.raises(SystemExit, match="art:ladder's s-t20/theta.safetensors"):
        starts_store.install(tmp_path / "node", man, fetch)
    assert not (tmp_path / "node" / "s50").exists()
