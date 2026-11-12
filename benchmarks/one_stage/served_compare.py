"""The served compare: a tied Pearl+ serve dir held to the row of record, on the CPU, once Pearl+'s piece wrote W.

    served_compare.py --serve D --row ROW --class S,P,O --pearl-tables SCHEME --pred PRED --of-record PROGRAM:MANIFEST
        --window-record W --out DIR

- PHI: D's serve.json states row ROW's declared φ. serve.json's φ is read as the registered compiler reads it
  (`prove_path.phi_of_serve`), the row's is `score.phi_of` on its workload file at PRED's checkpoint config; both have the
  same model, revision, checkpoint config, execution and default seed, and lower one window circuit at the class, cut at
  Pearl+'s linears under SCHEME (`window_circuit.lower`): each circuit's facts (its templates' ids and units, its
  registered values' sizes and the cut's calls, DIR/circuit_facts.json) are compared by their SHA-512.
- BUILD: PRED is the row's Build of record: its predicted.json names Program digest PROGRAM and manifest digest MANIFEST.
- CUT: W, Pearl+'s window record, registers exactly the circuit's Pearl+ values (`PEARL_VALUES`), each at its size in the
  circuit (two u16 words an FP32 word), under SCHEME, in 32-bit words.

The seal's B-OF-RECORD (`served_verdicts.sh`) holds each sealed record to PRED's manifest, so this doesn't. A file a
verdict can't read fails that verdict, naming the file. Each verdict prints a line; the last, ``SERVED-COMPARE: EQUAL``
or ``NOT EQUAL``, is the line the record names (`served_verdicts.py record --served-compare`), kept with the verdicts in
DIR/served_compare.json. Exit 0 iff EQUAL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
WORKLOADS = ROOT / "adapters" / "vllm" / "workloads"
#: the circuit's values W registers: each linear's A rows and its y rows
PEARL_VALUES = ("pearl-a/", "pearl-y/")
VERDICTS = ("PHI", "BUILD", "CUT")


def read_json(path: Path) -> Any:
    """``path``'s JSON; a ValueError naming it where it can't be read."""
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        raise ValueError(f"{path}: {type(e).__name__}: {e}") from e


def facts(low: Any) -> dict[str, Any]:
    """A lowered window circuit's facts (`window_circuit.lower`'s): each template's id and units, each registered value's
    size and the cut's calls, with the SHA-512 of their JSON."""
    c = low.circuit
    doc = {"templates": [[tid, n] for tid, (_, n) in zip(c.ids, c.templates, strict=True)],
           "params": dict(sorted(c.registered.params.items())),
           "calls": [list(x) for x in low.programs[0].cut.calls]}
    return {**doc, "digest": hashlib.sha512(json.dumps(doc, sort_keys=True).encode()).hexdigest()}


def lowered_facts(serve: Mapping[str, Any], row: str, config: str, cls: str, scheme: str) -> tuple[dict, dict, dict]:
    """serve.json's φ and row `row`'s declared φ at checkpoint config `config`, each as `phi` compares it, and the facts of
    the window circuit each lowers at class `cls` cut at Pearl+'s linears under `scheme` -> (the φ fields, the served
    circuit's facts, the row's)."""
    sys.path.insert(0, str(ROOT / "benchmarks" / "prove_path"))
    import prove_path as PP
    from verity_vllm.engine.served_seed import DEFAULT_SEED
    from verity_vllm.pipeline.workload_file import workload_file
    from verity_vllm.program.predict import window_circuit as WC
    from verity_vllm.program.predict.phi import Phi
    from verity_vllm.program.predict.score import phi_of
    from verity_vllm.program.predict.steps import Registered
    from verity_vllm.program.predict.window_class import PearlTables

    served = PP.phi_of_serve(serve)
    declared = phi_of({"row": row, "workload": read_json(workload_file(WORKLOADS / f"{row}.json"))}, config=config)
    declared = Phi(model={k: declared.model[k] for k in ("repo", "revision", "hf_config")}, target=dict(declared.target),
                   execution=declared.execution)

    def digest(o: Any) -> str:
        return hashlib.sha512(json.dumps(o, sort_keys=True).encode()).hexdigest()

    st, dt = served.target_profile().to_json(), declared.target_profile().to_json()
    fields = {"model": [served.model["repo"], declared.model["repo"]],
              "revision": [served.model["revision"], declared.model["revision"]],
              "hf_config": [digest(served.model["hf_config"]), digest(declared.model["hf_config"])],
              "execution": [served.execution, declared.execution],
              "default_seed": [serve.get("default_seed"), DEFAULT_SEED],
              "target": sorted(k for k in set(st) | set(dt) if st.get(k) != dt.get(k))}
    window_class, tables = PP.window_class_of(cls), PearlTables(scheme)
    low = [WC.lower(Registered.of(p, DEFAULT_SEED), window_class, tables=tables) for p in (served, declared)]
    return fields, facts(low[0]), facts(low[1])


def phi(fields: Mapping[str, Any], served: Mapping[str, Any], declared: Mapping[str, Any]) -> dict[str, Any]:
    """PHI from `lowered_facts`: every φ field equal and the two circuits' facts one digest. The declared target profile
    may differ from the served one where the circuit doesn't read it (``target_differs``)."""
    differ = sorted(k for k, v in fields.items() if k != "target" and v[0] != v[1])
    if served["digest"] != declared["digest"]:
        differ.append("circuit")
    return {"ok": not differ, "differ": differ, "fields": {k: v for k, v in fields.items() if k != "target"},
            "target_differs": fields["target"], "circuit": [served["digest"], declared["digest"]],
            "templates": len(served["templates"]), "values": len(served["params"])}


def build(pred: Path, of_record: str) -> dict[str, Any]:
    """BUILD: PRED's predicted.json names the Program digest and manifest digest `of_record` (PROGRAM:MANIFEST)."""
    want = of_record.split(":")
    if len(want) != 2:
        raise ValueError(f"--of-record {of_record!r} is PROGRAM_DIGEST:MANIFEST_DIGEST")
    p = read_json(pred / "predicted.json")
    got = [p.get("program_digest"), p.get("manifest_digest")]
    return {"ok": got == want, "names": got, "of_record": want}


def cut(w: Path, params: Mapping[str, int], scheme: str) -> dict[str, Any]:
    """CUT: W at `w` registers exactly the circuit's Pearl+ values of `params` at their sizes, under `scheme`, in 32-bit
    words."""
    rec = read_json(w)
    want = {n: size for n, size in params.items() if n.startswith(PEARL_VALUES)}
    got = {e["value"]: e["leaves"] * e["words"] * 2 for e in rec["registered"]}
    differ = sorted(n for n in set(got) | set(want) if got.get(n) != want.get(n))
    widths = sorted({e.get("word_bits") for e in rec["registered"]}, key=str)
    of = (rec.get("window") or {}).get("scheme")
    return {"ok": not differ and bool(want) and of == scheme and widths == [32], "values": len(got),
            "circuit_values": len(want), "differ": differ[:20], "differ_count": len(differ), "scheme": of,
            "word_bits": widths}


def judged(fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """`fn`'s verdict, or a failed one saying why it couldn't be reached."""
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"[:600]}


def said(name: str, v: Mapping[str, Any], a: argparse.Namespace) -> str:
    """Verdict `name`'s line."""
    end = f": {'PASS' if v['ok'] else 'FAIL'}"
    if "error" in v:
        return f"{name}: can't be reached ({v['error']}){end}"
    if name == "PHI":
        same = "" if v["ok"] else f", differing in {', '.join(v['differ'])}"
        return (f"PHI: serve.json's φ {'is' if v['ok'] else 'is not'} row {a.row}'s declared φ{same} (model, revision, "
                f"checkpoint config, execution, default seed, and the window circuit at class {a.cls} cut at Pearl+'s "
                f"linears under {a.pearl_tables}: facts sha512 {v['circuit'][0][:16]}… against {v['circuit'][1][:16]}…, "
                f"{v['templates']} templates, {v['values']} registered values; the declared target differs in "
                f"{', '.join(v['target_differs']) or 'nothing'}){end}")
    if name == "BUILD":
        return (f"BUILD: {a.pred}/predicted.json names Program digest {str(v['names'][0])[:16]}…, manifest digest "
                f"{str(v['names'][1])[:16]}…; the row's Build of record is {v['of_record'][0][:16]}…, "
                f"{v['of_record'][1][:16]}…{end}")
    return (f"CUT: W ({a.window_record}) registers {v['values']} values and the circuit reads {v['circuit_values']} Pearl+ "
            f"values; {v['differ_count']} differ in name or size (scheme {v['scheme']}, word bits {v['word_bits']}){end}")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--serve", type=Path, required=True, help="the serve dir, holding its serve.json")
    ap.add_argument("--row", required=True, help="the row of record's id (its workload file's name)")
    ap.add_argument("--class", dest="cls", required=True, help="the window class, SLOTS,PROMPT,OUTPUTS")
    ap.add_argument("--pearl-tables", required=True, help="Pearl+'s scheme, whose tables the circuit is cut at")
    ap.add_argument("--pred", type=Path, required=True, help="predict row's Build, its predicted.json")
    ap.add_argument("--of-record", required=True, help="the row's Build of record, PROGRAM_DIGEST:MANIFEST_DIGEST")
    ap.add_argument("--window-record", type=Path, required=True, help="W, Pearl+'s window record")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "served_compare.json").unlink(missing_ok=True)
    circuits: dict[str, Any] = {}

    def phi_verdict() -> dict[str, Any]:
        config = read_json(a.pred / "predicted.json").get("config")
        if not config or not Path(config).is_file():
            raise ValueError(f"{a.pred}/predicted.json names no checkpoint config on this node ({config})")
        fields, served, declared = lowered_facts(read_json(a.serve / "serve.json"), a.row, config, a.cls,
                                                 a.pearl_tables)
        circuits.update(served=served, declared=declared)
        (a.out / "circuit_facts.json").write_text(json.dumps(circuits, indent=1) + "\n")
        return phi(fields, served, declared)

    verdicts = {"PHI": judged(phi_verdict), "BUILD": judged(lambda: build(a.pred, a.of_record))}
    verdicts["CUT"] = (judged(lambda: cut(a.window_record, circuits["served"]["params"], a.pearl_tables)) if circuits
                       else {"ok": False, "error": "PHI lowered no circuit to hold W to"})
    for name in VERDICTS:
        print(said(name, verdicts[name], a), flush=True)
    ok = all(verdicts[n]["ok"] for n in VERDICTS)
    failed = [n for n in VERDICTS if not verdicts[n]["ok"]]
    line = (f"SERVED-COMPARE: {'EQUAL' if ok else 'NOT EQUAL'} (the serve against row {a.row}'s Build of record: "
            + ("PHI, BUILD and CUT PASS" if ok else f"{', '.join(failed)} FAIL") + "; B-OF-RECORD is the seal's)")
    (a.out / "served_compare.json").write_text(json.dumps(
        {"row": a.row, "class": a.cls, "scheme": a.pearl_tables, "ok": ok, "verdicts": verdicts, "line": line},
        indent=1, sort_keys=True, default=str) + "\n")
    print(line, flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
