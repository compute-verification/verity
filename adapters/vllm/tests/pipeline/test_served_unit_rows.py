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
STATEMENTS = ("7abb1051567c13db288dbb50aa8f03962105f9d81809d52f254d22627ccfef12"
              "c87a07e9febe30d26dba0ab89392e2d305036e474f8f52113fc66ce2dc490c4f")


def test_the_served_templates_statements_are_the_bytes_vllm_s_own_rows_gave(monkeypatch):
    tmpls = sorted({t for f in SERVED for t in json.loads(f.read_text())["chunk_cap"]})
    monkeypatch.setattr(IF, "templates_of", lambda _w: [])
    monkeypatch.setattr(WT, "reads", lambda _ms, _rows, _units: None)
    monkeypatch.setattr(IF, "circuit_record", lambda _w, _name, _templates, sts, **_kw: sts)
    members = [SimpleNamespace(template=t, schemas=True) for t in tmpls]
    sts = SRW.circuit_record(SimpleNamespace(members=members, row_tables=()), [])
    got = hashlib.sha512(IF.circuit_bytes(sts)).hexdigest()
    assert len(sts) == 38 and got == STATEMENTS, got
