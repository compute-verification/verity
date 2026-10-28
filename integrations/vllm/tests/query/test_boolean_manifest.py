"""A Boolean Program's manifest is its word Program's, apart from the Program digest it records.

`TargetProfile.ir = "boolean"` derives the word Program and lifts it to bits (`program.frontend.boolean.derived`), which keeps every Call,
node index and word leaf.  The Build writes the Boolean Program's instance sequence through each Call's word link
(`pipeline.build_instances`: the word Definition `trace.word_of` names, checked against the word node, with the Boolean Definition
as the row's `bits`), so `ProgramView`, the population, the protocol declarations (`SAMPLING_EVENTS`, ...) and the word check read
the word Program's Calls: the same units, boundaries and committed values whichever profile (word or Boolean) the Program is in.  The word check counts the
Boolean Program's size in Boolean gates beside its word gates.  The Build writes the word Program beside the Boolean one
(`program.boolean.write_word`), and the workload composition and the Match read the dir through it (`word_descriptor`).  The row is
a tiny registered-values greedy serve of B0.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from verity.ml.circuits.codec import descriptor_id, encode_program, program_digest  # noqa: E402
from verity_vllm.correspondence.reader_for_query import Correspondence  # noqa: E402
from verity_vllm.correspondence.runtime import correspondence_digest  # noqa: E402
from verity_vllm.check.match import program_compare as PC  # noqa: E402
from verity_vllm.pipeline import components as CO, manifest as MF  # noqa: E402
from verity_vllm.pipeline.build import _spec_histogram  # noqa: E402
from verity_vllm.pipeline.build_instances import write_instances  # noqa: E402
from verity.ml.definitions import boolean as BB  # noqa: E402
from verity.ml.definitions.registry import b1  # noqa: E402
from verity_vllm.query.program_view import from_instances  # noqa: E402
from verity_vllm.query.required import SAMPLING_EVENTS, request_manifest  # noqa: E402

C = {**b1.B0_CONFIG, "H": 64, "NH": 4, "KVH": 2, "D": 16, "I": 128, "QKV": 128, "GU": 256, "NL": 1, "V": 32, "MAX_POS": 64}


def _write(d: Path, rep) -> str:
    """`rep`'s Program as the Build writes a request Program dir (`build._write_program`): descriptor, instances, result."""
    program = rep.program
    desc = {**(rep.descriptor or encode_program(program)), "annotations": dict(program.annotations)}
    digest = program_digest(desc)
    with gzip.open(d / "descriptor.json.gz", "wt") as f:
        json.dump(desc, f)
    write_instances(str(d), program, digest, correspondence_digest(desc), {})
    res = {"program": {"digest": digest, "function": program.fn.id, "correspondence_digest": correspondence_digest(desc)},
           "target": {"declared": {"compute_capability": [8, 0]}},
           "spec_histogram": _spec_histogram(getattr(program, "word", None) or program)}
    if getattr(program, "word", None) is not None:
        res["program"]["word_digest"] = BB.write_word(d, program, rep.provenance["ir"]["word_digest"])
    (d / "result.json").write_text(json.dumps(res))
    (d / "derive-report.json").write_text(json.dumps(rep.to_json(), default=str))
    return digest


@pytest.fixture(scope="module")
def row(tmp_path_factory):
    from verity_vllm.program.frontend import derive_examples as ex
    from verity_vllm.program.frontend.boolean import derived
    from verity_vllm.program.frontend.derive import derive_report

    with torch.device("meta"):
        m = ex.GreedyServe(C, 1)
    word = derive_report(m, (torch.empty(3, dtype=torch.int32, device="meta"),), mode="profile", profile="b1-eager", config=C,
                         name="Serve", registered_values=True)
    assert word.ok, word.unsupported
    bits = derived(word)
    assert bits.ok, bits.unsupported
    dirs, digests = {}, {}
    for ir, rep in (("word", word), ("boolean", bits)):
        dirs[ir] = tmp_path_factory.mktemp(ir)
        digests[ir] = _write(dirs[ir], rep)
    return word, bits, dirs, digests


def _instances(d: Path) -> dict:
    with gzip.open(d / "instances.json.gz", "rt") as f:
        return json.load(f)


def _placeholder(x, digest: str):
    """`x` as JSON with the Program digest one placeholder."""
    s = json.dumps(x)
    assert digest in s
    return json.loads(s.replace(digest, "<program>"))


@pytest.mark.slow
def test_the_boolean_program_names_its_word_program_and_its_rows_are_the_word_rows(row):
    word, bits, dirs, digests = row
    assert bits.program.word is word.program and digests["word"] != digests["boolean"]
    w, b = _instances(dirs["word"]), _instances(dirs["boolean"])
    assert b["ir"] == "boolean" and "ir" not in w
    assert b["digest"] == digests["boolean"] and b["params"] == w["params"]
    assert [{k: v for k, v in r.items() if k != "bits"} for r in b["rows"]] == w["rows"]
    assert [r["bits"] for r in b["rows"]] == [descriptor_id(n.fn) for n in bits.program.fn.body.nodes]
    assert any(r["bits"] != r["spec"] for r in b["rows"])


@pytest.mark.slow
def test_programview_reads_the_word_families_and_the_boolean_spec_of_each_call(row):
    _word, _bits, dirs, _digests = row
    Pw, Pb = (from_instances(str(dirs[ir] / "instances.json.gz")) for ir in ("word", "boolean"))
    assert [(c.definition, c.family, c.port_table, tuple(c.reads)) for c in Pb.calls] == \
           [(c.definition, c.family, c.port_table, tuple(c.reads)) for c in Pw.calls]
    assert Pb.params == Pw.params
    assert any(c.family in SAMPLING_EVENTS for c in Pb.calls)
    assert all(c.bits is None for c in Pw.calls) and all(c.bits for c in Pb.calls)


@pytest.mark.slow
def test_the_boolean_manifest_is_the_word_manifest_apart_from_the_program_digest(row):
    """Every identity and every header field but the Program's own: its digest, its function id (the Boolean root binds
    `IR='boolean'`) and what is digested over them (the manifest digest, the partition's)."""
    word, bits, dirs, digests = row
    m = {}
    for ir, rep in (("word", word), ("boolean", bits)):
        x = _placeholder(MF.build(str(dirs[ir]), word="off"), digests[ir])
        assert x["program"].pop("function") == rep.program.fn.id and x["program"].pop("dir") == str(dirs[ir])
        x["query"]["partition"].pop("digest")
        m[ir] = x
    mw, mb = m["word"], m["boolean"]
    assert mw["identities"] == mb["identities"] and len(mw["identities"]) > 0
    assert {k for k in set(mw) | set(mb) if mw.get(k) != mb.get(k)} <= {"manifest_digest", "unreturned_digest"}


@pytest.mark.slow
def test_the_word_check_cuts_the_word_definitions_and_counts_boolean_gates_beside(row):
    _word, bits, dirs, _digests = row
    out = {}
    for ir in ("word", "boolean"):
        P = from_instances(str(dirs[ir] / "instances.json.gz"))
        corr = Correspondence.of(P, str(dirs[ir]))
        r = request_manifest(P, corr, json.loads((dirs[ir] / "result.json").read_text()), request_id="r0")
        out[ir] = MF.word_check(P, corr, r.required, "16/32", strict=False, unreturned=r.unreturned)
    w, b = out["word"], out["boolean"]
    assert "boolean_gates" not in w and w["calls"] > 0
    assert {k: v for k, v in b.items() if k not in ("groups", "boolean_gates")} == {k: v for k, v in w.items() if k != "groups"}
    assert [{k: v for k, v in g.items() if k not in ("bits", "boolean_gates")} for g in b["groups"]] == w["groups"]
    gates = {descriptor_id(n.fn): n.fn.gates for n in bits.program.fn.body.nodes}
    assert all(g["boolean_gates"] == gates[g["bits"]] * g["calls"] for g in b["groups"])
    assert b["boolean_gates"] == sum(g["boolean_gates"] for g in b["groups"]) > b["gates"]
    assert MF.word_record(b, strict=False)["boolean_gates"] == b["boolean_gates"]


@pytest.mark.slow
def test_the_composition_and_the_match_read_a_boolean_dir_as_its_word_program(row, tmp_path):
    """The word Program beside the Boolean one is the word derive's, and the workload composition (`pipeline.components`) and the
    Match (`program_compare.load_derived`) read it: the same component and the same Program as the word dir's, the Boolean
    digest still the dir's recorded one.  A word descriptor that is not the recorded word digest is refused."""
    import shutil

    from verity.ml.circuits.defs import REGISTRY
    from verity_vllm.program import workload as W

    _word, bits, dirs, digests = row
    b = str(dirs["boolean"])
    assert bits.provenance["ir"]["word_digest"] == digests["word"]
    assert BB.word_descriptor(b).name == BB.WORD_DESCRIPTOR and BB.word_descriptor(dirs["word"]).name == "descriptor.json.gz"
    descs = {ir: CO.load_desc(str(dirs[ir])) for ir in dirs}
    assert program_digest(descs["boolean"]) == digests["word"]
    prov = CO.component_provenance(b, descs["boolean"])
    assert prov["refusal"] is None and (prov["recorded_digest"], prov["word_digest"]) == (digests["boolean"], digests["word"])
    comps = {ir: W.request_component(descs[ir], REGISTRY) for ir in dirs}
    assert comps["boolean"].program_digest == comps["word"].program_digest == digests["word"]
    assert comps["boolean"].local_offsets_sha256 == comps["word"].local_offsets_sha256
    progs = {ir: PC.load_derived(str(dirs[ir]), ir) for ir in dirs}
    assert progs["boolean"].digest == progs["word"].digest
    assert [r["spec"] for r in progs["boolean"].inst] == [r["spec"] for r in progs["word"].inst]

    stale = tmp_path / "stale"
    shutil.copytree(b, stale)
    shutil.copy(dirs["word"] / "descriptor.json.gz", stale / BB.WORD_DESCRIPTOR)
    res = json.loads((stale / "result.json").read_text())
    res["program"]["word_digest"] = "0" * 128
    (stale / "result.json").write_text(json.dumps(res))
    assert "is not the Program derive_step built" in CO.component_provenance(str(stale), CO.load_desc(str(stale)))["refusal"]
    with pytest.raises(ValueError, match="not the derive's word digest"):
        BB.write_word(tmp_path, bits.program, "0" * 128)
