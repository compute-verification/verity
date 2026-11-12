"""names.py TYPES.tsv [NAME...]: each row's type as the census stager would take it, nothing lowered: stage.py's `parse`, and
for a `CENSUS` type `serving_rows.template` and `chunk_rows.statement` on its name (the layout its statement gets,
`chunk_rows.zk_refusal`), and the same on its Boolean version's id. One JSON line each; extra NAMEs are looked up the same
way as census names, and so is each line of names-extra.txt beside TYPES.tsv."""
import importlib.util
import json
import sys
from pathlib import Path

SRC = Path.cwd()
spec = importlib.util.spec_from_file_location("su_stage", SRC / "benchmarks/served_units/stage.py")
stage = importlib.util.module_from_spec(spec)
sys.modules["su_stage"] = stage
spec.loader.exec_module(stage)

from verity.circuits.codec import descriptor_id
from verity.definitions.boolean import boolean_version
from verity_vllm.pipeline import chunk_rows as CR
from verity_vllm.pipeline.serving_rows import template


def kind(tp, stmt) -> str:
    if stmt.get("layout"):
        return "templated read"
    if stmt.get("reads") == "slots":
        return "flat, reads in lookup slots"
    return "flat Boolean, no table" if tp.id != tp.word else "flat, reads inline"


def look(name: str) -> dict:
    rec: dict = {"name": name}
    try:
        tp = template(name)
        stmt = CR.statement(tp)
        rec.update(id=tp.id, word=tp.word, layout=tp.layout, boolean=tp.id != tp.word, statement=stmt, kind=kind(tp, stmt),
                   zk_refusal=CR.zk_refusal(tp))
        try:
            rec["boolean_version"] = descriptor_id(boolean_version(tp.fn))
        except Exception as e:  # noqa: BLE001
            rec["boolean_version_error"] = f"{type(e).__name__}: {e}"[:200]
    except BaseException as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"[:400]
    return rec


names = []
for line in Path(sys.argv[1]).read_text().splitlines():
    if not line.strip():
        continue
    tag, unit, *_ = line.split("\t")
    rec: dict = {"tag": tag, "unit": unit}
    try:
        k, u, S = stage.parse(unit)
        rec["table"] = type(u).__name__
        if isinstance(u, stage.Census):
            rec["census"] = look(u.name(S))
        elif isinstance(u, stage.Unit):
            rec["definition"] = descriptor_id(u.definition(S))
    except BaseException as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"[:400]
    print(json.dumps(rec), flush=True)
extra = Path(sys.argv[1]).with_name("names-extra.txt")
for name in [*sys.argv[2:], *(extra.read_text().splitlines() if extra.exists() else [])]:
    if name.strip():
        print(json.dumps({"tag": None, "census": look(name.strip())}), flush=True)
