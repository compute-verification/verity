"""cover_blocks.py EXTRA.json OUT: CIR-271 Part 2's block-type sweep, read on `main`.

For every block type of `benchmarks/sampling_plan/glm53_blocks.json`, and every name of EXTRA.json's ``units`` the card
lacks (the units of the list templates the card does not cut, `cover_list.py`'s cuts.json): its template (`serving_rows.template`), its Definition's id, the module that
registers it and that module's file, circuit-check's pins (`load_pins`, `PINNED_KINDS`) at its id and at its Boolean
version's id, `stage.py`'s tables that name it, and `census:NAME` resolved as CIR-259's names.py resolves it. Writes
OUT/blocks.json.
"""
import importlib.util
import json
import resource
import sys
import time
import traceback
from pathlib import Path

EXTRA, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()


def rss_gb() -> float:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * resource.getpagesize() / 2**30


def say(*a) -> None:
    print(f"BLOCKS {time.time() - T0:7.1f}s {rss_gb():5.2f}GiB", *a, flush=True)


def err(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"[:500]


from circuit_check import checks as CK  # noqa: E402
from circuit_check import sidecars as SC  # noqa: E402
from circuit_check import targets as TG  # noqa: E402
from verity.circuits.codec import descriptor_id  # noqa: E402
from verity.definitions.boolean import boolean_version  # noqa: E402
from verity.definitions.call_scope import spec_base, spec_statics  # noqa: E402
from verity_vllm.pipeline import chunk_rows as CR  # noqa: E402
from verity_vllm.pipeline.serving_rows import template  # noqa: E402

spec = importlib.util.spec_from_file_location("su_stage", "benchmarks/served_units/stage.py")
stage = importlib.util.module_from_spec(spec)
sys.modules["su_stage"] = stage
spec.loader.exec_module(stage)

TG.load_registries()
PINS = CK.load_pins()


def pinned_at(fid: str) -> list:
    return [k for k in PINS if k in TG.PINNED_KINDS and fid in PINS[k]]


def tables(name: str) -> list[dict]:
    fam, st = spec_base(name), {}
    try:
        st = spec_statics(name)
    except Exception:  # noqa: BLE001
        pass
    hits = []
    for tname, table in (("UNITS", stage.UNITS), ("PARTS", stage.PARTS), ("CENSUS", stage.CENSUS), ("READS", stage.READS)):
        for key, entry in table.items():
            S = dict(entry.defaults)
            over = {k: v for k, v in st.items() if k in S}
            S.update(over)
            try:
                if tname == "CENSUS":
                    got = entry.name(S)
                elif tname == "READS":
                    from verity.commitments.gates import embedding as E
                    got = descriptor_id(E.read(S["V"], S["H"]))
                else:
                    got = descriptor_id(entry.definition(S))
            except Exception:  # noqa: BLE001
                continue
            if spec_base(got) == fam:
                spec_str = key + (":" + ",".join(f"{k}={v}" for k, v in over.items()) if over else "")
                hits.append({"table": tname, "name": key, "spec": spec_str, "id": got, "exact": got == name})
    return hits


def module_file(mod: str) -> str | None:
    try:
        s = importlib.util.find_spec(mod)
        return s.origin if s else None
    except Exception:  # noqa: BLE001
        return None


def read(name: str) -> dict:
    rec: dict = {}
    t = time.time()
    try:
        tp = template(name)
        f = tp.fn
        rec.update(template=True, id=tp.id, word=tp.word, def_id=f.id, exact=pinned_at(f.id))
        try:
            mod = SC.owner(f)
            rec.update(owner=mod, owner_file=module_file(mod))
        except Exception as e:  # noqa: BLE001
            rec["owner_error"] = err(e)
        rec["defn_module"] = getattr(getattr(f, "defn", f), "__module__", None) or getattr(type(f), "__module__", None)
        try:
            bv = boolean_version(f)
            rec["boolean_version"] = {"id": bv.id, "same": bv.id == f.id, "pins": pinned_at(bv.id)}
        except Exception as e:  # noqa: BLE001
            rec["boolean_version"] = {"error": err(e)}
        try:
            stmt = CR.statement(tp)
            rec.update(census_ok=True, statement=stmt, zk_refusal=CR.zk_refusal(tp))
        except BaseException as e:  # noqa: BLE001
            rec.update(census_ok=False, census_error=err(e))
    except BaseException as e:  # noqa: BLE001
        rec.update(template=False, error=err(e))
    rec["tables"] = tables(name)
    rec["s"] = round(time.time() - t, 2)
    return rec


CARD = json.loads(Path("benchmarks/sampling_plan/glm53_blocks.json").read_text())["definitions"]
extra = [u for u in json.loads(EXTRA.read_text())["units"] if u not in CARD]
names = list(CARD) + extra
out: dict = {"card": len(CARD), "extra": extra, "blocks": {}}
for i, b in enumerate(names):
    if rss_gb() > 24:
        out["stopped"] = f"stopped at {b}: {rss_gb():.1f} GiB"
        break
    try:
        out["blocks"][b] = read(b)
    except BaseException as e:  # noqa: BLE001
        out["blocks"][b] = {"error": err(e), "trace": traceback.format_exc()[-1500:]}
    if i % 25 == 0:
        say(i, b[:80])
(OUT / "blocks.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
say("done", len(out["blocks"]))
