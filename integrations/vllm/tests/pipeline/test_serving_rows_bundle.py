"""The serving rows of a deferred Commit (`pipeline.deferred_rows`, ``verity-vllm serving-rows``), end to end on CPU: a Program
with RoPE_v1 rows (D=64) executed honestly, its members committed by the production CPU committer and sealed into a replay bundle with
its registered weights, then the verifier's partition (every RoPE head) read back through the bundle opened against the Commit's record.
The registration, a stand-in linked to the record (the in-Commit hook's over the same members is served), passes `registration.check`
against the verifier's own copies (its population program, the run root of record, the public bodies), every hidden row is the served
value, and a partition digest, a template or a binding map that is not the record's is refused."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from verity.ml.circuits.defs import REGISTRY, bind
from verity.core.protocols.sampled_proofs.one_stage import draw as DR, partition as P, registration as R
from verity.ml.windows import interface as IF, rows as CS
from verity_vllm.commit import step_rows
from verity_vllm.pipeline import cli
from verity_vllm.pipeline import cpu_replay as CPU
from verity_vllm.pipeline import replay_bundle as RB
from verity_vllm.pipeline import deferred_rows as DRW
from verity.ml.definitions.registry import b1  # noqa: F401  (registers RoPEHead_v1)

from tests.pipeline import served_rope as SR
from tests.pipeline.served_rope import D, HEADS, LAW, ROPE


def _commit(tmp_path) -> dict:
    """A deferred Commit: its output dir and row dir, and the world that served it."""
    pytest.importorskip("torch")
    return SR.commit(tmp_path)


def test_a_deferred_commits_serving_rows_are_the_verifiers_window_over_the_served_values(tmp_path):
    c = _commit(tmp_path)
    wdir = tmp_path / "window"
    win = DRW.from_bundle(RB.path(c["out"], 1), str(c["row_dir"]), str(c["partition"]), wdir)
    rec = json.loads((wdir / "registration.json").read_text())
    heads = HEADS
    w = rec["window"]
    assert (win["population"], rec["format"], w["kind"], w["run"]) == (heads, R.FORMAT, "stand-in", c["run_root"])
    man = RB.open_bundle(RB.path(c["out"], 1))["manifest_sha256"]
    assert w["of"] == {"commitment": step_rows.NAME, "run_root": c["run_root"], "bundle_manifest_sha256": man, "rows": DRW.ROWS}
    assert w["link"] == DRW.link(step_rows.NAME) and w["committer"]["from"] == "replay bundle" and w["row"] == "rope-d64-cpu"

    # the verifier's own copies: its population program, the run root of record, the public body it holds
    defn = bind(REGISTRY["RoPEHead_v1"], D=D)
    program, query = P.population_program(defn, heads), P.template_instance_query(defn)
    ports = {f"{ROPE}/{p}": {"leaves": heads, "domain": R.window_domain(P.program_digest(program), P.digest(P.partition(program, query)),
                                                                          f"{ROPE}/{p}", CS.ROW_SCHEMA, heads, c["run_root"])}
             for p in ("x", "cs", "out")}
    pub = (wdir / "m0-public.bin").read_bytes()
    v = R.check(rec, program, query, DR.parse_law(LAW), CS.SCHEME, ports, leaf_layer=CS.leaf_layer({ROPE: pub}))
    assert v.ok, v.detail
    # the same copies from the record alone, as the audit's verifier derives them (no committed value read)
    mine = DRW.window_of_record(RB.path(c["out"], 1), str(c["row_dir"]), str(c["partition"]))
    assert (mine.partition_digest, mine.run, CS.expected_ports(mine)) == (win["partition_digest"], c["run_root"], ports)
    # and from the window's own records: its circuit record, whose SHA-512 the registration binds
    assert w["circuit"] == hashlib.sha512((wdir / "circuit.json").read_bytes()).hexdigest()
    got, _doc = IF.of_records(wdir, DR.parse_law(LAW), {ROPE: defn}.__getitem__)
    assert (got.partition_digest, got.run, got.members, CS.expected_ports(got)) == (mine.partition_digest, mine.run, mine.members, ports)

    # every hidden row is what was served: x the qkv row's head slice, cs the cache row at the token's position, out the rotated head
    rows = np.frombuffer((wdir / "m0-private.bin").read_bytes()[:heads * 3 * D * 2], dtype="<u2").reshape(heads, 3, D)
    world, inst = c["world"], c["inst"]
    idx = json.loads((wdir / "index.json").read_text())
    ents = idx["members"][0]["rows"]
    assert sum(e[1] - e[0] for e in ents) == heads and [e[0] for e in ents] == sorted(e[0] for e in ents)
    for lo, hi, rid, _step, layer, _op, _mem, _tok, i in ents:
        r = inst["rows"][i]
        (_k, a, xlo, xhi), = r["args"][0]
        (_k, _p, clo, chi), = r["args"][1]
        assert layer == 0 and r["spec"].startswith("RoPE_v1")
        assert np.array_equal(rows[lo:hi, 0], world.values[(rid, a)][xlo:xhi].reshape(hi - lo, D))
        assert np.array_equal(rows[lo:hi, 1], np.broadcast_to(world.weights[clo:chi], (hi - lo, D)))
        assert np.array_equal(rows[lo:hi, 2], world.values[(rid, i)].reshape(hi - lo, D))


def test_the_in_commit_hook_writes_the_window_as_served(tmp_path):
    """The in-Commit hook (`serving_rows.readers`, its request as the row stage writes it) over the same Commit's members: the same
    program, partition and domains, and the window ``served`` (serving's own commitment, before any draw) with no stand-in link."""
    c = _commit(tmp_path)
    hook = json.loads((SR.hook(c, tmp_path / "hook", run="r-hook") / "registration.json").read_text())
    DRW.from_bundle(RB.path(c["out"], 1), str(c["row_dir"]), str(c["partition"]), tmp_path / "deferred", workers=1)
    deferred = json.loads((tmp_path / "deferred" / "registration.json").read_text())
    assert (hook["window"]["kind"], hook["window"]["run"], hook["window"]["research_run"]) == ("served", c["run_root"], "r-hook")
    assert "of" not in hook["window"] and "link" not in hook["window"] and deferred["window"]["kind"] == "stand-in"
    same = ("format", "program", "query", "law", "partition", "population", "scheme")
    assert {k: hook[k] for k in same} == {k: deferred[k] for k in same}
    assert [(x["port"], x["domain"]) for x in hook["roots"]] == [(x["port"], x["domain"]) for x in deferred["roots"]]

    # the audit's source check (A5's): the deferred window is a stand-in linked to the run root of record and the bundle, and the
    # hook's is served over that SHA-512 root
    man, root = RB.open_bundle(RB.path(c["out"], 1))["manifest_sha256"], c["run_root"]
    assert DRW.source(deferred["window"], root, man, step_rows.NAME) == {"kind": True, "run": True, "commitment": True, "of": True, "link": True}
    assert DRW.source(hook["window"], root, man, step_rows.NAME) == {"kind": True, "run": True, "commitment": True, "sha512": True, "link": True}


def test_a_rows_sha512_records_stand_in_names_that_commitment_and_its_served_window_passes_the_source_check(tmp_path):
    """A deferred Commit, committed under rows-sha512: the stand-in's ``of`` and ``link`` name the commitment the record was
    committed in, and the in-Commit hook's served window over the same record passes the audit's source check; either one naming
    vllm-v1 is refused by name."""
    c = _commit(tmp_path)
    bundle, root = RB.path(c["out"], 1), c["run_root"]
    r = DRW.of_record(bundle, str(c["row_dir"]))
    assert (r.commitment, len(bytes.fromhex(root))) == (step_rows.NAME, DRW.SERVED_ROOT)
    DRW.from_bundle(bundle, str(c["row_dir"]), str(c["partition"]), tmp_path / "deferred", workers=1)
    deferred = json.loads((tmp_path / "deferred" / "registration.json").read_text())["window"]
    hook = json.loads((SR.hook(c, tmp_path / "hook") / "registration.json").read_text())["window"]
    assert (deferred["kind"], deferred["of"]["commitment"], deferred["commitment"], hook["commitment"]) == ("stand-in", *[step_rows.NAME] * 3)
    assert deferred["link"] == DRW.link(step_rows.NAME) and f"{step_rows.NAME} roots" in deferred["link"] and "vllm-v1" not in deferred["link"]

    man = RB.open_bundle(bundle)["manifest_sha256"]

    def refused(window: dict) -> list[str]:
        return [k for k, v in DRW.source(window, root, man, r.commitment).items() if not v]

    assert refused(deferred) == refused(hook) == []
    assert refused(dict(deferred, **DRW.stand_in(root, man, "vllm-v1"))) == ["of", "link"]
    assert refused(dict(hook, commitment="vllm-v1")) == ["commitment"]

    # the record's weights were registered under rows-sha512 as well: a SHA-512 tree, the one the hook's rows were opened against
    wroot = r.row["commit"]["weight_registration"]["root"]
    tree = json.loads((bundle / "weights" / "tree.json").read_text())
    assert (tree["scheme"], tree["root"], len(bytes.fromhex(wroot))) == (step_rows.NAME, wroot, DRW.SERVED_ROOT)
    assert DRW.source(hook, root, man, r.commitment, wroot)["weights_sha512"] is True
    assert [k for k, v in DRW.source(hook, root, man, r.commitment, "ab" * 32).items() if not v] == ["weights_sha512"]


def test_the_audits_source_check_takes_a_served_window_only_over_a_sha512_run_root_of_record():
    """`deferred_rows.source` on a served window over a 64-byte run root of record, and each field altered refused by its name: the
    run, the root's size, the record's commitment named, a stand-in link carried, and a kind that is neither form; a stand-in's
    commitment, ``of`` and ``link`` are the record's."""
    man, root, com = "cd" * 32, "ab" * DRW.SERVED_ROOT, step_rows.NAME
    w = {"kind": "served", "run": root, "commitment": com, "research_run": "r"}

    def refused(window: dict, run_root: str = root, commitment: str = com) -> list[str]:
        return [k for k, v in DRW.source(window, run_root, man, commitment).items() if not v]

    assert refused(w) == []
    short = "ab" * 32
    assert refused(dict(w, run=short, commitment="vllm-v1"), short, "vllm-v1") == ["sha512"]
    assert refused(dict(w, run="00" * DRW.SERVED_ROOT)) == ["run"]
    assert refused(dict(w, commitment="vllm-v1")) == ["commitment"]
    assert refused(dict(w, **DRW.stand_in(root, man, com))) == ["link"]
    assert refused(dict(w, kind="stand-in")) == ["of", "link"]
    assert refused(dict(w, kind="replayed")) == ["kind", "of", "link"]
    standin = dict(w, kind="stand-in", **DRW.stand_in(root, man, com))
    assert refused(standin) == []
    assert refused(dict(standin, commitment="vllm-v1")) == ["commitment"]
    assert refused(dict(standin, **DRW.stand_in(root, man, "vllm-v1"))) == ["of", "link"]


def test_a_partition_or_binding_map_that_is_not_the_records_is_refused(tmp_path):
    c = _commit(tmp_path)
    pf = json.loads(c["partition"].read_text())
    c["partition"].write_text(json.dumps(dict(pf, partition_digest="00" * 64)))
    with pytest.raises(ValueError, match="not the verifier's"):
        DRW.from_bundle(RB.path(c["out"], 1), str(c["row_dir"]), str(c["partition"]), tmp_path / "w1")
    # a template is matched by its descriptor id, so a RoPE head of another D is no member of these rows
    c["partition"].write_text(json.dumps(dict(pf, templates=["RoPEHead_v1{D=128}"])))
    with pytest.raises(LookupError, match=r"no rows of RoPEHead_v1\{D=128\}"):
        DRW.from_bundle(RB.path(c["out"], 1), str(c["row_dir"]), str(c["partition"]), tmp_path / "w1")
    # a scope no row is under names a row it missed, so a wrong prefix shows the op_path spelling
    c["partition"].write_text(json.dumps(dict(pf, scope={"op_path_prefix": "nowhere."})))
    with pytest.raises(LookupError, match=r"under nowhere\.; its rows exist outside the scope, e\.g\. op_path \S+ at step \d+"):
        DRW.from_bundle(RB.path(c["out"], 1), str(c["row_dir"]), str(c["partition"]), tmp_path / "w1")
    c["partition"].write_text(json.dumps(pf))
    bm = c["out"] / "binding_map_p1.json"
    bm.write_text(bm.read_text().replace('"rank": 0', '"rank": 1', 1))
    with pytest.raises(CPU.Refused, match="not the bundle's binding map"):
        DRW.from_bundle(RB.path(c["out"], 1), str(c["row_dir"]), str(c["partition"]), tmp_path / "w2")


def test_the_command_writes_the_window(tmp_path):
    c = _commit(tmp_path)
    o = cli.parse("serving-rows", [str(RB.path(c["out"], 1)), "--row-dir", str(c["row_dir"]), "--partition", str(c["partition"]),
                                   "--out", str(tmp_path / "w"), "--workers", "1"])
    assert DRW.main(o) == 0
    win = json.loads((tmp_path / "w" / "window.json").read_text())
    assert win["registration"] == R.FORMAT and win["run"] == c["run_root"]
    assert json.loads((tmp_path / "w" / "partition.json").read_text())["partition_digest"] == win["partition_digest"]
