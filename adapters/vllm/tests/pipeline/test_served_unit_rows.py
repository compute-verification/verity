"""The served partitions' chunked templates (``chunk_cap`` of ``SERVED``, GLM-5.3's at TP 8: step 16's every family, its sparse
MLA's split chains, and step 0's prefill, both with their greedy select's stages) keep their statements in ``circuit.json``
(`serving_rows.circuit_record`: the rows Flock's stager stages, `chunk_rows.unit_rows`, and their `chunk_rows.statement`) byte
for byte, at the SHA-512 vLLM's own chunk and chain rows gave them. The window stands in: its population program would lower
them."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from verity.commitments.windows import interface as IF
from verity_vllm.pipeline import serving_rows as SRW, weight_rows as WT

SERVED = [Path(__file__).resolve().parents[4] / "benchmarks/one_stage" / f"glm53_{p}.json"
          for p in ("all_families_tp8_step16", "sparse_mla_tp8_step16", "prefill_tp8_step0")]
STATEMENTS = ("cda40e51a9c286c1461348fa69ded063accd4c3dfa90434c6d2f2d917e5b62ec"
              "444cb98bbf31429e2c4d237f02faadf93fd39cf309cf5eb5a09b333c7194bae7")


def test_the_served_templates_statements_are_the_bytes_vllm_s_own_rows_gave(monkeypatch):
    tmpls = sorted({t for f in SERVED for t in json.loads(f.read_text())["chunk_cap"]})
    monkeypatch.setattr(IF, "templates_of", lambda _w: [])
    monkeypatch.setattr(WT, "reads", lambda _ms, _rows, _units: None)
    monkeypatch.setattr(IF, "circuit_record", lambda _w, _name, _templates, sts, **_kw: sts)
    members = [SimpleNamespace(template=t, schemas=True) for t in tmpls]
    sts = SRW.circuit_record(SimpleNamespace(members=members, row_tables=()), [])
    got = hashlib.sha512(IF.circuit_bytes(sts)).hexdigest()
    assert len(sts) == 70 and got == STATEMENTS, got


def test_each_served_key_rows_writer_commits_the_bits_it_returns():
    """Each key rows' writer the served partitions name commits 16 NVIS W bits (`partition._template_bits`, the class the
    partition checks and the weight the leak bound gives a wrong proof block of it), 65,536 at 64 keys of 64 columns: the
    row it returns, which every head's scores and values units read, though it is its input."""
    from verity.protocols.spot.verifier.one_stage import partition as P

    writers = sorted({t for f in SERVED for t in json.loads(f.read_text())["chunk_cap"]
                      if t.startswith("SparseMlaHeadKeysWire_")})
    got = {t: (P._template_bits(fn), 16 * fn.statics["NVIS"] * fn.statics["W"]) for t in writers for fn in [SRW.template(t).fn]}
    assert len(writers) == 18 and all(b == want for b, want in got.values()), got
    assert got["SparseMlaHeadKeysWire_v2{NVIS=64,W=64,C=0}"] == (65536, 65536)
