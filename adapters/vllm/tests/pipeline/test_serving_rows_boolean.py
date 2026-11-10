"""The serving rows of a Boolean Program (`pipeline.serving_rows.template`): a template is read through its word link, so the
Boolean Definition is a template like its word Definition, under either IR.

The served toy Commit (`tests.pipeline.served_rope`: RoPE_v1 and Gemm_v1 rows) is read as its word Program and as the Boolean
Program's word view (`pipeline.build_instances`: each row the word Definition its Call's Boolean Definition computes, with that Boolean
Definition as the row's ``bits`` and the header's ``ir``), with the word templates and with their Boolean versions
(``RoPEHead_v2{D=64}``, ``GemmCoordinate_v4{K=16}``).  Every pairing gives the same members, rows, layouts and instance counts; the
population program is over the templates as named, so the Boolean templates' partition is one under both IRs.  A deferred window
under the Boolean template commits the word template's u16 rows, and its registration passes `registration.check` against the
verifier's population program over the Boolean template.  Two templates that compute one word Definition, a template with no rows,
and one with a port that is not whole u16 words are refused by name; a row template of 32- and 16-bit ports is packed."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import REGISTRY, bind
from verity.core.protocols.sampled_proofs.one_stage import draw as DR, partition as P, registration as R
from verity.ml.windows import rows as CS
from verity_vllm.pipeline import deferred_rows as DRW
from verity_vllm.pipeline import replay_bundle as RB
from verity_vllm.pipeline import serving_rows as SRW
from verity.ml.definitions import boolean as BB
from verity.ml.definitions.registry import b1  # noqa: F401  (registers RoPEHead_v1)
from verity.ml.definitions.call_scope import specialization
from verity_vllm.query.program_view import spec_base, spec_statics

from tests.pipeline import served_rope as SR
from tests.pipeline.served_rope import D, HEADS, LAW, ROPE

ROPE_BITS = f"RoPEHead_v2{{D={D}}}"
GEMM, GEMM_BITS = "GemmCoordinate_v1{K=16}", "GemmCoordinate_v4{K=16}"


def _on_bits(inst: dict) -> dict:
    """A word Program's instance doc as its Boolean Program's (`build_instances._word_rows`): every row's ``bits`` the Boolean
    version of its Definition, ``ir: boolean`` in the header; the digest kept, as the toy Commit's record names it."""
    rows = [{**r, "bits": descriptor_id(BB.boolean_version(specialization(spec_base(r["spec"]), spec_statics(r["spec"]))))}
            for r in inst["rows"]]
    assert all(r["bits"] != r["spec"] for r in rows)
    return {**inst, "ir": "boolean", "rows": rows}


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    pytest.importorskip("torch")
    c = SR.commit(tmp_path_factory.mktemp("served"))
    r = DRW.of_record(RB.path(c["out"], 1), str(c["row_dir"]))
    return c, r


def _shape(window, rows) -> tuple:
    return ([(m.rows, m.outs, m.n, m.base, m.tables) for m in window.members],
            [[(rid, x["i"], d["op_path"], d["row"]) for rid, _pi, x, d in rs] for rs in rows])


def test_a_template_is_matched_through_its_word_link_under_either_ir(served):
    c, r = served
    programs = {"word": r.programs, "boolean": {k: _on_bits(v) for k, v in r.programs.items()}}
    templates = {"word": [ROPE, GEMM], "boolean": [ROPE_BITS, GEMM_BITS]}
    got = {(ir, kind): SRW.population(ps, {"templates": ts, "law": LAW}, c["run_root"], r.program_dirs)
           for ir, ps in programs.items() for kind, ts in templates.items()}
    shapes = {k: _shape(*v) for k, v in got.items()}
    assert len({json.dumps(s) for s in shapes.values()}) == 1
    (rope, gemm), _rows = shapes["word", "word"]
    assert rope[:3] == ((("x", D), ("cs", D)), (("out", D),), HEADS) and dict(gemm[4]).keys() == {"x", "w"}
    assert [SRW.template(t).layout for t in templates["boolean"]] == ["rope", "gemm"]

    # the population program is over the templates as named: the Boolean templates' partition is one under both IRs
    for ir in programs:
        for kind, ts in templates.items():
            window = got[ir, kind][0]
            assert [m.template for m in window.members] == ts == window.partition["query"]["params"]["templates"]
    digest = {k: v[0].partition_digest for k, v in got.items()}
    assert digest["word", "boolean"] == digest["boolean", "boolean"] != digest["word", "word"] == digest["boolean", "word"]


def _boolean_row_dir(row_dir: Path) -> None:
    f = row_dir / "build_request" / "instances.json.gz"
    with gzip.open(f, "rt") as fh:
        inst = json.load(fh)
    with gzip.open(f, "wt") as fh:
        json.dump(_on_bits(inst), fh)


def test_a_boolean_templates_window_commits_the_word_rows_and_passes_the_verifiers_check(tmp_path):
    pytest.importorskip("torch")
    c = SR.commit(tmp_path)
    word = DRW.from_bundle(RB.path(c["out"], 1), str(c["row_dir"]), str(c["partition"]), tmp_path / "word", workers=1)
    _boolean_row_dir(c["row_dir"])
    (pf := tmp_path / "bits.json").write_text(json.dumps({"templates": [ROPE_BITS], "law": LAW}))
    wdir = tmp_path / "bits"
    win = DRW.from_bundle(RB.path(c["out"], 1), str(c["row_dir"]), str(pf), wdir, workers=1)
    rec = json.loads((wdir / "registration.json").read_text())
    assert (win["population"], rec["window"]["kind"], rec["window"]["run"]) == (HEADS, "stand-in", c["run_root"])

    # the verifier's own copies, its population program over the Boolean template
    defn = bind(REGISTRY["RoPEHead_v2"], D=D)
    program, query = P.population_program(defn, HEADS), P.template_instance_query(defn)
    part = P.digest(P.partition(program, query))
    ports = {f"{ROPE_BITS}/{p}": {"leaves": HEADS, "domain": R.window_domain(P.program_digest(program), part, f"{ROPE_BITS}/{p}",
                                                                             CS.ROW_SCHEMA, HEADS, c["run_root"])}
             for p in ("x", "cs", "out")}
    v = R.check(rec, program, query, DR.parse_law(LAW), CS.SCHEME, ports,
                leaf_layer=CS.leaf_layer({ROPE_BITS: (wdir / "m0-public.bin").read_bytes()}))
    assert v.ok, v.detail
    assert (win["partition_digest"], word["partition_digest"] != part) == (part, True)
    mine = DRW.window_of_record(RB.path(c["out"], 1), str(c["row_dir"]), str(pf))
    assert (mine.partition_digest, CS.expected_ports(mine)) == (part, ports)

    # the same u16 rows and the same unit -> served identity index as the word template's window
    n = HEADS * 3 * D * 2
    rows = [np.frombuffer((d / "m0-private.bin").read_bytes()[:n], dtype="<u2") for d in (wdir, tmp_path / "word")]
    assert np.array_equal(*rows) and rows[0].any()
    idx = [json.loads((d / "index.json").read_text())["members"][0] for d in (wdir, tmp_path / "word")]
    assert idx[0]["rows"] == idx[1]["rows"] and (idx[0]["template"], idx[1]["template"]) == (ROPE_BITS, ROPE)


def test_templates_that_compute_one_word_definition_have_no_rows_or_are_not_u16_words_are_refused(served):
    c, r = served
    programs = {k: _on_bits(v) for k, v in r.programs.items()}
    with pytest.raises(ValueError, match=r"RoPEHead_v1\{D=64\} and RoPEHead_v2\{D=64\} both compute RoPEHead_v1\{D=64\}"):
        SRW.population(programs, {"templates": [ROPE, ROPE_BITS], "law": LAW}, c["run_root"], r.program_dirs)
    with pytest.raises(LookupError, match=r"no rows of RoPEHead_v2\{D=128\}"):
        SRW.population(programs, {"templates": ["RoPEHead_v2{D=128}"], "law": LAW}, c["run_root"], r.program_dirs)
    for t in ("Embedding_v1{V=8,H=16}", "Embedding_v2{V=8,H=16}"):
        assert (SRW.template(t).layout, SRW.template(t).packed) == ("row", (32, 16, 16))
    with pytest.raises(ValueError, match=r"commit u16 words, and its ports' words are \[8, 16, 32\] bits \(a row template's port"):
        SRW.template("Fp8GroupQuantUe8m0_v1{K=3,G=3}")
    with pytest.raises(LookupError, match="names no registered Definition"):
        SRW.template("NoSuchTemplate_v1")
