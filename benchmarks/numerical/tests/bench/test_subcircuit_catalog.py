"""The subcircuit catalog (`verity_numerical.bench.subcircuit_catalog`): one artifact per build, deterministic, every census
subcircuit present or marked missing, its digests core's, its counts cited from circuit-check records that exist.

The records are an excerpt of real circuit-check reports (`data/subcircuit_catalog/circuit-check.json`: attention-head d64
FA2 at T = 1 and 512, the sm90 E4M3 GEMM coordinate, bias-add and rope-head d64), so a test reads counts exactly as a build
cites them. Attention's key-count range is cut to 1..3 here, and the inline limit to 2, so its variant map is a payload file
as a full build's is."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity_numerical.bench import census
from verity_numerical.bench import subcircuit_catalog as SC
from verity_numerical.bench import templates as TM

DATA = Path(__file__).parent / "data" / "subcircuit_catalog"
RECORDS = DATA / "circuit-check.json"
ATTN = "attention-head/d64-bn128/sm80-fa2-bf16"
ROPE, BIAS, E4M3, SILU = "rope-head/d64/neox-bf16", "bias-add/n2048/bf16", "gemm-coordinate/k1536/sm90-wgmma-e4m3", "silu-mul/i1024/bf16"
IDS = [ROPE, BIAS, E4M3, ATTN, SILU]


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """A tmp store, attention's range cut to 1..3 and the inline limit to 2."""
    tmp = tmp_path_factory.mktemp("catalog")
    mp = pytest.MonkeyPatch()
    mp.setenv("RESEARCH_STORE", str(tmp / "store"))
    mp.setenv("RESEARCH_STORE_CONFIG", str(tmp / "store.toml"))
    mp.setenv("RESEARCH_REPO", str(tmp / "repo"))
    (tmp / "repo").mkdir()
    attn = census.SUBCIRCUITS[ATTN]
    cut = SimpleNamespace(**{**vars(attn), "variable_ranges": SimpleNamespace(T=SimpleNamespace(min=1, max=3, source="test"))})
    mp.setitem(census.SUBCIRCUITS, ATTN, cut)
    mp.setattr(SC, "INLINE_VARIANTS", 2)
    from research.store.local import LocalStore

    yield SimpleNamespace(tmp=tmp, store=LocalStore())
    mp.undo()


def _build(env, name, report=None, export=SC.export):
    store = env.store
    (env.tmp / name).mkdir()
    path = env.tmp / name / "circuit-check.json"
    path.write_text(json.dumps(report) if report is not None else RECORDS.read_text())
    report_art = SC.put_report(store, path)
    tree = env.tmp / name / "tree"
    meta = SC.build(tree, json.loads(path.read_text()), report_art, IDS, export=export)
    art = SC.put(store, tree, meta, report_art)
    return SimpleNamespace(art=art, meta=meta, tree=tree, report_art=report_art, manifest=store.get_manifest(art))


@pytest.fixture(scope="module")
def first(env):
    return _build(env, "first")


def _entries(b):
    return {e["id"]: e for e in b.meta["entries"]}


def _files(b):
    return {f.path: f.sha256 for f in b.manifest.payload.files}


def test_one_artifact_of_the_catalogs_kind(first):
    m = first.manifest
    assert m.kind == SC.KIND == "verity/subcircuit-catalog/v1" and m.refs == {"circuit_check": first.report_art}
    assert m.meta["schema"] == SC.KIND and m.payload.type == "tree"


def test_building_twice_gives_the_same_artifact(env, first):
    again = _build(env, "again")
    assert again.art == first.art and again.report_art == first.report_art


def test_one_changed_subcircuit_changes_only_its_blob_and_entry(env, first):
    """A changed export changes that subcircuit's blob and its entry's export digest; a changed record changes only its
    entry's counts. Every other blob and entry is the same."""
    def export(sub, sts):
        blob = SC.export(sub, sts)
        return blob.replace(b'"schema":', b'"changed":true,"schema":', 1) if sub.id == ROPE else blob

    changed = _build(env, "export-changed", export=export)
    assert {p for p, h in _files(changed).items() if _files(first).get(p) != h} == {f"subcircuits/{ROPE}.boolean.json"}
    assert set(_files(changed)) == set(_files(first))
    assert [i for i in IDS if _entries(changed)[i] != _entries(first)[i]] == [ROPE]
    assert {k for k in _entries(first)[ROPE] if _entries(changed)[ROPE][k] != _entries(first)[ROPE][k]} == {"export"}

    report = json.loads(RECORDS.read_text())
    rec = next(t for t in report["targets"] if t["target"] == BIAS)
    rec["definitions"][0]["gates"] += 1
    recounted = _build(env, "record-changed", report)
    assert _files(recounted) == _files(first)
    assert [i for i in IDS if {k: v for k, v in _entries(recounted)[i].items() if k != "counts"}
            != {k: v for k, v in _entries(first)[i].items() if k != "counts"}] == []
    assert _entries(recounted)[BIAS]["counts"]["variants"]["{}"]["gates"] == _entries(first)[BIAS]["counts"]["variants"]["{}"]["gates"] + 1


def test_every_subcircuit_is_present_or_marked_missing(first):
    es = _entries(first)
    assert [e["id"] for e in first.meta["entries"]] == IDS
    for e in es.values():
        assert ("export" in e) != ("export" in e.get("missing", {})), e["id"]
        assert ("counts" in e) != ("counts" in e.get("missing", {})), e["id"]
        assert e["display"] == census.SUBCIRCUITS[e["id"]].display and e["template"] == census.SUBCIRCUITS[e["id"]].template
    assert "silicon reference" in es[E4M3]["missing"]["export"]
    assert "no circuit-check record" in es[SILU]["missing"]["counts"]
    assert set(es[ROPE]) >= {"export", "counts"} and "missing" not in es[ROPE]


def test_a_developer_circuit_never_enters(env):
    with pytest.raises(KeyError, match="developer circuits never enter"):
        SC.build(env.tmp / "dev", {"targets": []}, "art:" + "0" * 64, ["RoPEHead_v1{D=64}"])


def _digests(b, e):
    if "program_digests" in e:
        return e["program_digests"]
    f = e["program_digests_file"]
    blob = (b.tree / f["path"]).read_bytes()
    assert hashlib.sha256(blob).hexdigest() == f["sha256"]
    return json.loads(blob)


def test_digests_match_program_digests_at_the_recorded_statics(first):
    """Every variant's digest, and the descriptor id at the default statics, is `verity_flock.circuit.program_digests`'s; the
    statics each record checked are among the variants."""
    from verity_flock import circuit as FC

    es = _entries(first)
    assert es[ATTN]["variants"] == {"count": 3, "varying": {"T": [1, 3]}} and es[ATTN]["default_statics"] == {"T": 3}
    assert "program_digests_file" in es[ATTN] and "program_digests" in es[ROPE]
    for e in es.values():
        sub = TM.parse_subcircuit_id(e["id"])
        got = _digests(first, e)
        assert len(got) == e["variants"]["count"]
        for key, digest in got.items():
            (_k, v), = FC.program_digests(sub, json.loads(key)).items()
            assert v == digest, (e["id"], key)
        assert e["descriptor_id"] in FC.program_digests(sub, e["default_statics"])
    recorded = set(es[ATTN]["counts"]["variants"])
    assert recorded == {'{"T":1}', '{"T":512}'} and recorded & set(_digests(first, es[ATTN])) == {'{"T":1}'}
    assert set(es[ROPE]["counts"]["variants"]) == set(_digests(first, es[ROPE])) == {"{}"}


def test_counts_cite_records_that_exist(env, first):
    """Each entry's counts name the stored circuit-check report, a target in it, and that target's digest, and are its fields."""
    report = json.loads(Path(env.store.fetch(first.report_art)).read_text())
    assert env.store.get_manifest(first.report_art).kind == SC.RECORDS_KIND
    targets = {t["target"]: t for t in report["targets"]}
    for e in first.meta["entries"]:
        if "counts" not in e:
            continue
        r = e["counts"]["record"]
        rec = targets[r["target"]]
        assert r["report"] == first.report_art and r["target"] == e["id"]
        assert r["sha512"] == hashlib.sha512(json.dumps(rec, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        for d, low in zip(rec["definitions"], rec["lowering"]):
            v = e["counts"]["variants"][json.dumps(low["instance_statics"], sort_keys=True, separators=(",", ":"))]
            assert v["gates"] == d["gates"] and all(v.get(k) == low.get(k) for k in SC.LOWERING_FIELDS)
    rope = _entries(first)[ROPE]["counts"]["variants"]["{}"]
    assert (rope["gates"], rope["units_per_instance"], rope["ands"], rope["rows"]) == (64, 32, 5828, 6145)


@pytest.mark.slow
def test_the_whole_census_is_present_or_marked_missing(env, monkeypatch):
    """Every census subcircuit binds to a template, gets a descriptor id and a digest at each variant, and enters the catalog
    present or marked missing (exports stubbed, each variable range cut to its ends: the recorded full build does the rest)."""
    for sid, c in list(census.SUBCIRCUITS.items()):
        if (vr := getattr(c, "variable_ranges", None)) is not None:
            ends = {k: SimpleNamespace(min=v.min, max=v.min + 1, source="test") for k, v in vars(vr).items()}
            monkeypatch.setitem(census.SUBCIRCUITS, sid, SimpleNamespace(**{**vars(c), "variable_ranges": SimpleNamespace(**ends)}))
    meta = SC.build(env.tmp / "census", copy.deepcopy(json.loads(RECORDS.read_text())), "art:" + "0" * 64,
                    export=lambda sub, sts: json.dumps([sub.id, sts]).encode())
    assert [e["id"] for e in meta["entries"]] == SC.census_ids() and len(meta["entries"]) == len(census.SUBCIRCUITS)
    for e in meta["entries"]:
        assert ("export" in e) != ("export" in e.get("missing", {})) and ("counts" in e) != ("counts" in e.get("missing", {}))
        assert ("descriptor_id" in e) != ("program_digests" in e.get("missing", {})), e["id"]
    model_only = [e["id"] for e in meta["entries"] if "program_digests" in e.get("missing", {})]
    assert model_only and all("NotEvaluable" in census_missing for census_missing in
                              (e["missing"]["program_digests"] for e in meta["entries"] if e["id"] in model_only))
