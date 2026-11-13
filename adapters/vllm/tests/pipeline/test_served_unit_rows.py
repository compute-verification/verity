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
STATEMENTS = ("738cbcd5a8f49d53f620c7bba9eec4d0b6e995c8b78cabff53854d834f7f7787"
              "0e94b2844c3dfa57e63da2ccf0d70ef30270edc5145d3c8c49373a59cd4f4f7e")


def test_the_served_templates_statements_are_the_bytes_vllm_s_own_rows_gave(monkeypatch):
    tmpls = sorted({t for f in SERVED for t in json.loads(f.read_text())["chunk_cap"]})
    monkeypatch.setattr(IF, "templates_of", lambda _w: [])
    monkeypatch.setattr(WT, "reads", lambda _ms, _rows, _units: None)
    monkeypatch.setattr(IF, "circuit_record", lambda _w, _name, _templates, sts, **_kw: sts)
    members = [SimpleNamespace(template=t, schemas=True) for t in tmpls]
    sts = SRW.circuit_record(SimpleNamespace(members=members, row_tables=()), [])
    got = hashlib.sha512(IF.circuit_bytes(sts)).hexdigest()
    assert len(sts) == 38 and got == STATEMENTS, got
