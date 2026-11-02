"""The served partitions' chunked templates (``chunk_cap`` of ``SERVED``, GLM-5.3's at TP 8: step 16's every family, its sparse
MLA's splits, and step 0's prefill) keep their statements in ``circuit.json`` (`serving_rows.circuit_record`: the rows C-Flock's
stager stages, `chunk_rows.unit_rows`, and their `chunk_rows.reads`) byte for byte, at the SHA-512 vLLM's own chunk and split rows
gave them. The window stands in: its population program would lower them."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from verity.ml.windows import interface as IF
from verity_vllm.pipeline import serving_rows as SRW

SERVED = [Path(__file__).resolve().parents[4] / "benchmarks/one_stage" / f"glm53_{p}.json"
          for p in ("all_families_tp8_step16", "sparse_mla_tp8_step16", "prefill_tp8_step0")]
STATEMENTS = ("dbda90083c381ad2053d9bd617ab8ac3f92a90dd4a2fe872014603060828ee21"
              "bb3be5150e804ba58584c4f37f5f4ef9cb2e9608bb9ac87118c2579b21e2f843")


def test_the_served_templates_statements_are_the_bytes_vllm_s_own_rows_gave(monkeypatch):
    tmpls = sorted({t for f in SERVED for t in json.loads(f.read_text())["chunk_cap"]})
    monkeypatch.setattr(IF, "templates_of", lambda _w: [])
    monkeypatch.setattr(IF, "circuit_record", lambda _w, _name, _templates, sts: sts)
    sts = SRW.circuit_record(SimpleNamespace(members=[SimpleNamespace(template=t, schemas=True) for t in tmpls]))
    got = hashlib.sha512(IF.circuit_bytes(sts)).hexdigest()
    assert len(sts) == 27 and got == STATEMENTS, got
