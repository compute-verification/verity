"""`test_served_unit_rows.py`'s own computation, printed: how many statements the served cards' chunked templates have and
their SHA-512, for ``STATEMENTS``."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from verity.commitments.windows import interface as IF
from verity_vllm.pipeline import serving_rows as SRW, weight_rows as WT

SERVED = [Path("benchmarks/one_stage") / f"glm53_{p}.json"
          for p in ("all_families_tp8_step16", "sparse_mla_tp8_step16", "prefill_tp8_step0")]
tmpls = sorted({t for f in SERVED for t in json.loads(f.read_text())["chunk_cap"]})
IF.templates_of = lambda _w: []
WT.reads = lambda _ms, _rows, _units: None
IF.circuit_record = lambda _w, _name, _templates, sts, **_kw: sts
members = [SimpleNamespace(template=t, schemas=True) for t in tmpls]
sts = SRW.circuit_record(SimpleNamespace(members=members, row_tables=()), [])
print(json.dumps({"statements": len(sts), "sha512": hashlib.sha512(IF.circuit_bytes(sts)).hexdigest()}))
