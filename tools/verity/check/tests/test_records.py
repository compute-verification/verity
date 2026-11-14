"""`tools/verity/check/records.py`: circuit-check's binding records, put on the store once, named in a recorded check's verdict
pack in place of their bytes, and fetched by the next check before circuit-check runs; and `check.py`'s side of it."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import sys
import tarfile
from pathlib import Path

import pytest

from research.store.remote import FsRemote

ROOT = Path(__file__).resolve().parents[4]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"test_records_{name}", ROOT / "tools" / "verity" / "check" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


R = _load("records")
V = _load("verdicts")
CHECK = _load("check")


def _machine(tmp_path: Path, monkeypatch, name: str) -> Path:
    """Point both caches at this machine's directories; circuit-check's cache in it."""
    monkeypatch.setenv("CHECK_CACHE", str(tmp_path / name / "check"))
    monkeypatch.setenv("VERITY_TEST_CACHE", str(tmp_path / name / "tests"))
    return V.roots()["check"] / "circuit-check"


def _bytes(doc) -> bytes:
    return json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()


def _put(d: Path, rel_dir: str, data: bytes) -> str:
    """Store ``data`` in ``d`` under its own name, as `circuit_check.cache` does; the name."""
    rel = f"{rel_dir}/{hashlib.sha512(data).hexdigest()}.json"
    (d / rel).parent.mkdir(parents=True, exist_ok=True)
    (d / rel).write_bytes(data)
    return rel


def _entry(d: Path, target: list, value: str, version: str = "v1") -> dict[str, str]:
    """An entry of ``target``'s key in circuit-check's cache ``d``, with its read set and the sited value it needs; each name."""
    v = _put(d, f"{version}/reads/values", _bytes({"value": value}))
    s = _put(d, f"{version}/reads/set", _bytes({"needs": [f"../../{v.split('/', 1)[1]}"], "runs": {"units": [0]}}))
    key = {"request": {"target": target}, "version": version}
    k32 = hashlib.sha512(_bytes(key)).hexdigest()[:32]
    e = _put(d, f"{version}/{k32}", _bytes({"key": key, "needs": [f"../{s.split('/', 1)[1]}"], "own": {}, "report": {},
                                            "shared": 0}))
    return {"entry": e, "set": s, "value": v}


def _binding(d: Path, unit: str, value: str | None = None) -> dict[str, str]:
    return _entry(d, ["catalog-unit", [unit], 1], value or unit)


class _Counted(FsRemote):
    """A remote that keeps each key put and got, and can refuse to give some (``lost``) or give other bytes (``garbled``)."""

    def __init__(self, root: Path, lost: tuple[str, ...] = (), garbled: tuple[str, ...] = ()):
        super().__init__(root)
        self.puts, self.gets, self.lost, self.garbled = [], [], lost, garbled

    def put(self, key, path):
        self.puts.append(key)
        return super().put(key, path)

    def get(self, key, path):
        self.gets.append(key)
        if key.removeprefix(R.PREFIX) in self.lost:
            raise KeyError(key)
        super().get(key, path)
        if key.removeprefix(R.PREFIX) in self.garbled:
            Path(path).write_bytes(Path(path).read_bytes().replace(b'"shared":0', b'"shared":1'))     # still JSON, its length


def _members(pack: Path) -> dict[str, bytes]:
    with tarfile.open(pack, "r:*") as tar:
        return {m.name: tar.extractfile(m).read() for m in tar.getmembers()}


def _publisher(d: Path, since_ns: int, remote) -> object:
    return R.Publisher(d, "check/circuit-check/", since_ns, R.putter(remote) if remote else None, d.parent)


def _pack(tmp_path: Path, d: Path, before: dict, since_ns: int, remote, name: str = "verdicts.tar.gz") -> Path:
    """A recorded check's pack, as `check.Packs` writes it."""
    CHECK.Packs(out := tmp_path / name, before, _publisher(d, since_ns, remote)).close()
    return out


def test_a_pack_names_the_records_its_check_produced_in_place_of_their_bytes(tmp_path, monkeypatch):
    """The records a check's steps built are put on the store under their names and listed in its pack's manifest, and their
    bytes stay out of the pack; a target's entry stays in it with every record it needs, one it shares with a binding's
    record included.  A record built before the check began is no record of its."""
    d = _machine(tmp_path, monkeypatch, "a")
    old = _binding(d, "old")
    since = os.stat(d / old["entry"]).st_mtime_ns + 1
    before = V.snapshot()
    new = _binding(d, "u", value="shared")
    target = _entry(d, ["definition", "D"], "shared")
    for rel in [*new.values(), *target.values()]:
        os.utime(d / rel, ns=(since, since))
    store = _Counted(tmp_path / "store")
    pack = _pack(tmp_path, d, before, since, store)
    got = _members(pack)
    listed = json.loads(got.pop(R.MANIFEST))["records"]
    assert listed == {rel: (d / rel).stat().st_size for rel in new.values()}
    assert sorted(got) == sorted(f"check/circuit-check/{rel}" for rel in {*target.values()})
    assert sorted(store.puts) == sorted(R.PREFIX + rel for rel in new.values())
    assert all((tmp_path / "store" / (R.PREFIX + rel)).read_bytes() == (d / rel).read_bytes() for rel in new.values())


def test_the_manifest_never_lands_in_a_cache_and_old_readers_take_the_pack(tmp_path, monkeypatch):
    """Importing a pack that lists records writes none of the list into the caches, so no later pack carries it as a verdict;
    the pack's verdicts import as before."""
    d = _machine(tmp_path, monkeypatch, "a")
    before = V.snapshot()
    _binding(d, "u")
    target = _entry(d, ["definition", "D"], "x")
    pack = _pack(tmp_path, d, before, 0, _Counted(tmp_path / "store"))
    assert R.MANIFEST in _members(pack)
    _machine(tmp_path, monkeypatch, "b")
    assert V.refusal(pack) is None
    assert V.import_(pack) == (3, 0)
    assert sorted(V.entries()) == sorted(f"check/circuit-check/{rel}" for rel in target.values())


def test_a_record_the_store_holds_is_never_put_again(tmp_path, monkeypatch):
    """A node remembers what it put or fetched, so a second pack write puts nothing; another node, which remembers nothing, finds
    each record on the store and uploads none of their bytes."""
    d = _machine(tmp_path, monkeypatch, "a")
    rec = _binding(d, "u")
    store = _Counted(tmp_path / "store")
    p = _publisher(d, 0, store)
    first = p.pack()
    assert len(store.puts) == 3 and p.stats["put"] == 3
    assert p.pack() == first and _publisher(d, 0, store).pack() == first
    assert len(store.puts) == 3
    b = _machine(tmp_path, monkeypatch, "b")
    for rel in rec.values():
        (b / rel).parent.mkdir(parents=True, exist_ok=True)
        (b / rel).write_bytes((d / rel).read_bytes())
    other = _publisher(b, 0, store)
    other.pack()
    assert other.stats == {"put": 0, "already": 3, "failed": 0, "late": 0}


def test_a_record_the_store_didnt_take_stays_in_the_pack(tmp_path, monkeypatch):
    """A put that fails leaves the record unnamed and its bytes in the pack, so the next check still finds it; a binding record
    whose read set the store refused is named by none either, since nobody could count it from the store."""
    d = _machine(tmp_path, monkeypatch, "a")
    before = V.snapshot()
    rec = _binding(d, "u")

    def put(key, path):
        if key.endswith(rec["set"]):
            raise OSError("the store refused it")
        return "created"

    p = R.Publisher(d, "check/circuit-check/", 0, put, d.parent)
    drop, manifest = p.pack()
    assert (drop, manifest) == (set(), None) and p.stats["failed"] == 1
    V.export(pack := tmp_path / "p.tar.gz", before, None, drop, ())
    assert sorted(_members(pack)) == sorted(f"check/circuit-check/{rel}" for rel in rec.values())


def test_a_corrupt_record_here_is_never_named(tmp_path, monkeypatch):
    """A binding record whose read set's bytes aren't its name's can't count here or anywhere: it is neither put nor named."""
    d = _machine(tmp_path, monkeypatch, "a")
    rec = _binding(d, "u")
    (d / rec["set"]).write_bytes(b"{}")
    store = _Counted(tmp_path / "store")
    assert _publisher(d, 0, store).pack() == (set(), None) and store.puts == []


def _named(tmp_path: Path, monkeypatch, units: tuple[str, ...]) -> tuple[dict[str, int], dict[str, dict[str, str]], Path]:
    """The records of ``units`` a check on machine "a" put on the store and named in its pack: (the pack's list, each unit's
    names, the store's root)."""
    d = _machine(tmp_path, monkeypatch, "a")
    before = V.snapshot()
    recs = {u: _binding(d, u) for u in units}
    pack = _pack(tmp_path, d, before, 0, FsRemote(tmp_path / "store"))
    return R.named_in(pack), recs, d


def test_a_shard_fetches_only_the_records_it_lacks(tmp_path, monkeypatch):
    """A node holding one unit's records fetches only the other's, each into its own name, and logs both counts."""
    named, recs, a = _named(tmp_path, monkeypatch, ("u", "w"))
    b = _machine(tmp_path, monkeypatch, "b")
    for rel in recs["u"].values():
        (b / rel).parent.mkdir(parents=True, exist_ok=True)
        (b / rel).write_bytes((a / rel).read_bytes())
    store = _Counted(tmp_path / "store")
    got = R.fetch(named, b, R.getter(store), base=b.parent)
    assert sorted(store.gets) == sorted(R.PREFIX + rel for rel in recs["w"].values())
    assert {k: got[k] for k in ("named", "here", "fetched")} == {"named": 6, "here": 3, "fetched": 3}
    assert all((b / rel).read_bytes() == (a / rel).read_bytes() for rel in recs["w"].values())
    assert "6 named by the packs imported, 3 here already, 3 fetched" in R.fetch_line(got)
    assert not [p for p in b.rglob("*") if p.name.startswith(".")]
    p = _publisher(b, 0, store)
    p.pack()
    assert p.stats["put"] + p.stats["already"] == 3              # what came from the store is never put back


def test_a_missing_or_corrupt_record_is_left_for_circuit_check_to_build(tmp_path, monkeypatch):
    """One record the store lacks and another whose bytes aren't its name's are left out, and so is each entry one of them is
    needed by: circuit-check finds no entry and builds the binding.  The rest is placed, and nothing raises."""
    named, recs, _a = _named(tmp_path, monkeypatch, ("u", "w", "x"))
    b = _machine(tmp_path, monkeypatch, "b")
    store = _Counted(tmp_path / "store", lost=(recs["u"]["set"],), garbled=(recs["w"]["entry"],))
    got = R.fetch(named, b, R.getter(store))
    assert {k: got[k] for k in ("missing", "corrupt", "incomplete", "fetched")} == \
        {"missing": 1, "corrupt": 1, "incomplete": 1, "fetched": 6}
    assert not (b / recs["u"]["entry"]).exists() and not (b / recs["w"]["entry"]).exists()
    assert all((b / rel).is_file() for rel in recs["x"].values())
    assert "left to circuit-check to build: 1 missing, 1 corrupt, 1 incomplete" in R.fetch_line(got)


def test_a_fetch_is_bounded_in_bytes_and_seconds(tmp_path, monkeypatch):
    """Past the bytes bound nothing more is asked for; past the time bound nothing more is fetched; what is left is built."""
    named, _recs, _a = _named(tmp_path, monkeypatch, ("u", "w"))
    b = _machine(tmp_path, monkeypatch, "b")
    store = _Counted(tmp_path / "store")
    small = sorted(named.values())[0]
    got = R.fetch(named, b, R.getter(store), max_bytes=small)
    assert got["over"] == 5 and len(store.gets) == 1
    ticks = iter(range(100))
    c = _machine(tmp_path, monkeypatch, "c")
    got = R.fetch(named, c, R.getter(store), max_s=0.5, jobs=1, clock=lambda: next(ticks))
    assert got["late"] == 6 and got["fetched"] == 0 and not c.exists()


def test_a_pack_that_names_records_wrongly_names_none(tmp_path):
    """A manifest naming a path outside a version's records, or one that isn't JSON, names nothing to fetch."""
    for doc in (b'{"records": {"../../etc/passwd": 1, "v1/reads/set/x.json": 1}}', b"not json"):
        with tarfile.open(pack := tmp_path / "p.tar.gz", "w:gz") as tar:
            info = tarfile.TarInfo(R.MANIFEST)
            info.size = len(doc)
            tar.addfile(info, io.BytesIO(doc))
        assert R.named_in(pack) == {}


def test_a_pack_names_what_circuit_checks_manifests_list_once_it_writes_them(tmp_path, monkeypatch):
    """Once circuit-check lists the records its catalog took or built, a pack names those, an old one it took included, and
    none another check on the node wrote meanwhile; a manifest of another schema lists nothing."""
    d = _machine(tmp_path, monkeypatch, "a")
    taken, other = _binding(d, "taken"), _binding(d, "other")
    (out := tmp_path / "out").mkdir()
    (out / "circuit-records.step.json").write_text(json.dumps(
        {"schema": R.RUN_SCHEMA, "records": [{"binding": "taken", "record": taken["entry"], "built": False}]}))
    (out / "circuit-records.session.json").write_text(json.dumps({"schema": "v0", "records": [{"record": other["entry"]}]}))
    p = R.Publisher(d, "check/circuit-check/", 0, None, d.parent, out=out)
    assert p.produced() == {rel: (d / rel).stat().st_size for rel in taken.values()}
    assert R.run_records(tmp_path / "none") is None


def test_the_step_and_the_session_are_told_where_to_list_their_records_once_circuit_check_can(tmp_path, monkeypatch):
    """circuit-check's step gets a manifest of its own (each shard of it one), and the pytest step, whose circuit-check suite's
    session builds records too, the variable naming its own; neither while this tree's circuit-check writes none, and no
    step's key reads either."""
    def argv(only=None):
        return {name: a for g in CHECK.groups(tmp_path, 8, only=only) for name, a, _ in g}["circuit-check"]

    class Steps:
        why = "a test's"

        def start(self, name, argv, cwd, e, f):
            self.env = e

        def end(self, rec):
            return rec

    for can in (False, True):
        monkeypatch.setattr(CHECK, "cc_manifests", lambda: can)
        assert ("--records-manifest" in argv()) is can
        CHECK._step("pytest", ["true"], tmp_path, tmp_path, True, steps := Steps())
        assert steps.env.get(R.SESSION_ENV) == (str(tmp_path / "circuit-records.session.json") if can else None)
    assert str(tmp_path / "circuit-records.step.json") in argv()
    sharded = " ".join(argv(["circuit-check:1/4", "circuit-check:3/4"]))
    assert str(tmp_path / "circuit-records.step-1.json") in sharded and str(tmp_path / "circuit-records.step-3.json") in sharded


@pytest.mark.parametrize("parts, builds", [(None, True), (["pytest"], True), (["circuit-check:2/4"], True),
                                           (["lean-audit:security", "ratchet-gate"], False)])
def test_only_a_check_that_runs_circuit_check_fetches(parts, builds):
    check = CHECK
    assert check.builds_records(parts) is builds
    suite = next(s.name for s in check.SUITES.discover() if s.path == check.CIRCUIT_SUITE)
    assert check.builds_records([f"pytest:{suite}:1/2"]) and not check.builds_records(["pytest:repository"])


@pytest.mark.parametrize("lost", [False, True])
def test_a_recorded_check_fetches_named_records_before_its_steps_and_changes_no_key(tmp_path, monkeypatch, lost):
    """A recorded check imports a pack that names records, fetches the ones its node lacks before any step runs, and hands its
    steps the same environment as without them; when the store gives none, the check runs and passes all the same.  Its own
    pack then names what it put, here nothing new."""
    check = CHECK
    named, recs, _a = _named(tmp_path, monkeypatch, ("u",))
    store = _Counted(tmp_path / "store", lost=tuple(named) if lost else ())
    run_dir = tmp_path / "run"
    (run_dir / "inputs").mkdir(parents=True)
    b = _machine(tmp_path, monkeypatch, "node")
    monkeypatch.setenv("RESEARCH_RUN_ID", "r20261112-000000-abcd")
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(run_dir))
    monkeypatch.delenv("VERITY_SKIP_STORE", raising=False)
    monkeypatch.delenv(check.VERDICT_URLS_ENV, raising=False)
    monkeypatch.setattr(check.SUITES, "foreign", lambda members, env: None)
    monkeypatch.setattr(check, "store_remote", lambda: store)
    seen, env = {}, check.env()

    def run(out, n, cache, keep_going, only, packs, reused, regenerate, circuit_cache=None, live=None, splits=None):
        seen.update(here=sorted(rel for rel in recs["u"].values() if (b / rel).is_file()), env=check.env())
        return True, []

    monkeypatch.setattr(check, "run", run)
    assert check.main(["--keep-going", "--verdicts-in", str(tmp_path / "verdicts.tar.gz")]) == 0
    assert seen["here"] == ([] if lost else sorted(recs["u"].values()))
    assert seen["env"] == env
    doc = json.loads((run_dir / "result.json").read_text())
    assert doc["verdicts"]["records"]["fetched"] == (0 if lost else 3)
    assert doc["verdicts"]["out"]["records"]["put"] == 0
    assert R.MANIFEST not in _members(run_dir / "verdicts.tar.gz")
