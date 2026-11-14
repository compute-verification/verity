"""The served compare: a tied Pearl+ serve dir held to the row of record, on the CPU, once Pearl+'s piece wrote W.

    served_compare.py --serve D --row ROW --class S,P,O --pearl-tables SCHEME --pred PRED --of-record PROGRAM:MANIFEST
        --window-record W --out DIR [--revision REV]

- PHI: D's serve.json states row ROW's declared φ. serve.json's φ is read as the registered compiler reads it
  (`prove_path.phi_of_serve`), the row's is `score.phi_of` on its workload file at PRED's checkpoint config, taken at
  revision REV when given (`at_revision`: another revision the manifests pin of the row's model); both have the
  same model, revision, checkpoint config, execution and default seed, and lower one window circuit at the class, cut at
  Pearl+'s linears under SCHEME (`window_circuit.lower`): each circuit's facts (its templates' ids and units, its
  registered values' sizes and the cut's calls, DIR/circuit_facts.json) are compared by their SHA-512. PRED's checkpoint
  config is the copy PRED keeps beside its predicted.json (`CONFIG_COPY`, which `served_verdicts.sh` writes after
  `predict row`), else the file predicted.json names (`checkpoint_config`): that path is in the serving run's own
  tree, so a serve dir kept after the tree is gone is compared again, by any tree, from the copy. The verdict keeps the
  SHA-256 of the config it read, and PHI holds it to the comparing tree's config of record for the served checkpoint.
  PHI's line and verdict name the revision served and the one the row's workload declares, and under REV the last line
  names both too, so EQUAL is never read as a serve at the row's declared revision.
- BUILD: PRED is the row's Build of record: its predicted.json names Program digest PROGRAM and manifest digest MANIFEST.
- CUT: W, Pearl+'s window record, registers exactly the circuit's Pearl+ values (`PEARL_VALUES`), each at its size in the
  circuit (two u16 words an FP32 word), under SCHEME, in 32-bit words, and beside them only what its version registers
  that the circuit never reads: at version 2 the B rows of the weights the cut's calls read (`PEARL_WEIGHTS`), each
  weight's N rows of K words a call's; at version 1 nothing.

The seal's B-OF-RECORD (`served_verdicts.sh`) holds each sealed record to PRED's manifest, so this doesn't. A file a
verdict can't read fails that verdict, naming the file. Each verdict prints a line; the last, ``SERVED-COMPARE: EQUAL``
or ``NOT EQUAL``, is the line the record names (`served_verdicts.py record --served-compare`), kept with the verdicts in
DIR/served_compare.json. Exit 0 iff EQUAL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
WORKLOADS = ROOT / "adapters" / "vllm" / "workloads"
#: PRED's copy of the checkpoint config its Build was predicted from, beside its predicted.json
CONFIG_COPY = "checkpoint_config.json"
#: the circuit's values W registers: each linear's A rows and its y rows
PEARL_VALUES = ("pearl-a/", "pearl-y/")
#: what W registers beside them at its version 2 (`pearl_c.window`): each weight's B rows, ``pearl-b/<w>``, which the
#: circuit, cut at its linears, never reads
PEARL_WEIGHTS = re.compile(r"pearl-b/(0|[1-9][0-9]*)")
VERDICTS = ("PHI", "BUILD", "CUT")


def read_json(path: Path) -> Any:
    """``path``'s JSON; a ValueError naming it where it can't be read."""
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        raise ValueError(f"{path}: {type(e).__name__}: {e}") from e


def checkpoint_config(pred: Path) -> Path:
    """The checkpoint config ``pred``'s Build was predicted from: the copy ``pred`` keeps (`CONFIG_COPY`), else the file
    its predicted.json names; a ValueError, with what to give, where neither is on this node."""
    kept = pred / CONFIG_COPY
    if kept.is_file():
        return kept
    named = read_json(pred / "predicted.json").get("config")
    if named and Path(named).is_file():
        return Path(named)
    raise ValueError(f"{pred} keeps no {CONFIG_COPY}, and its predicted.json names a checkpoint config that is not on this "
                     f"node ({named}): give --pred a dir that holds its predicted.json and, as {CONFIG_COPY}, the config "
                     "its Build was predicted from")


def facts(low: Any) -> dict[str, Any]:
    """A lowered window circuit's facts (`window_circuit.lower`'s): each template's id and units, each registered value's
    size and the cut's calls, with the SHA-512 of their JSON."""
    c = low.circuit
    doc = {"templates": [[tid, n] for tid, (_, n) in zip(c.ids, c.templates, strict=True)],
           "params": dict(sorted(c.registered.params.items())),
           "calls": [list(x) for x in low.programs[0].cut.calls]}
    return {**doc, "digest": hashlib.sha512(json.dumps(doc, sort_keys=True).encode()).hexdigest()}


def at_revision(declared: Any, revision: str | None) -> Any:
    """The row's declared φ `declared` with only its model's repo, revision and checkpoint config, at `revision` when given:
    another revision of the row's model, which the manifests must pin (`pinned_hf_config`). The row's Build and its circuit
    read no weight byte, and its checkpoint config is compared as it is."""
    from verity_vllm.engine.profiles.dense_generic import pinned_hf_config
    from verity_vllm.program.predict.phi import Phi

    repo, own = declared.model["repo"], declared.model["revision"]
    if revision not in (None, own) and pinned_hf_config(repo, revision) is None:
        raise ValueError(f"--revision {revision}: no manifest pins {repo} at it")
    return Phi(model={"repo": repo, "revision": revision or own, "hf_config": declared.model["hf_config"]},
               target=dict(declared.target), execution=declared.execution)


def lowered_facts(serve: Mapping[str, Any], row: str, config: str, cls: str, scheme: str,
                  revision: str | None = None) -> tuple[dict, dict, dict]:
    """serve.json's φ and row `row`'s declared φ at checkpoint config `config` (at `revision` when given, `at_revision`),
    each as `phi` compares it, and the facts of the window circuit each lowers at class `cls` cut at Pearl+'s linears under
    `scheme` -> (the φ fields, the row's own revision among them, the served circuit's facts, the row's)."""
    sys.path.insert(0, str(ROOT / "benchmarks" / "prove_path"))
    from verity.protocols.proofs.spot.prover.audit import prove_path as PP
    from verity_vllm.engine.served_seed import DEFAULT_SEED
    from verity_vllm.pipeline.workload_file import workload_file
    from verity_vllm.program.predict import window_circuit as WC
    from verity_vllm.program.predict.score import phi_of
    from verity_vllm.program.predict.steps import Registered
    from verity_vllm.program.predict.window_class import PearlTables

    served = PP.phi_of_serve(serve)
    row_phi = phi_of({"row": row, "workload": read_json(workload_file(WORKLOADS / f"{row}.json"))}, config=config)
    declared = at_revision(row_phi, revision)

    def digest(o: Any) -> str:
        return hashlib.sha512(json.dumps(o, sort_keys=True).encode()).hexdigest()

    st, dt = served.target_profile().to_json(), declared.target_profile().to_json()
    fields = {"model": [served.model["repo"], declared.model["repo"]],
              "revision": [served.model["revision"], declared.model["revision"]],
              "hf_config": [digest(served.model["hf_config"]), digest(declared.model["hf_config"])],
              "execution": [served.execution, declared.execution],
              "default_seed": [serve.get("default_seed"), DEFAULT_SEED],
              "target": sorted(k for k in set(st) | set(dt) if st.get(k) != dt.get(k)),
              "row_revision": row_phi.model["revision"]}
    window_class, tables = PP.window_class_of(cls), PearlTables(scheme)
    low = [WC.lower(Registered.of(p, DEFAULT_SEED), window_class, tables=tables) for p in (served, declared)]
    return fields, facts(low[0]), facts(low[1])


def phi(fields: Mapping[str, Any], served: Mapping[str, Any], declared: Mapping[str, Any]) -> dict[str, Any]:
    """PHI from `lowered_facts`: every φ field equal and the two circuits' facts one digest. The declared target profile
    may differ from the served one where the circuit doesn't read it (``target_differs``); the row's own revision is
    reported (``row_revision``), never compared."""
    differ = sorted(k for k, v in fields.items() if k not in ("target", "row_revision") and v[0] != v[1])
    if served["digest"] != declared["digest"]:
        differ.append("circuit")
    return {"ok": not differ, "differ": differ,
            "fields": {k: v for k, v in fields.items() if k not in ("target", "row_revision")},
            "row_revision": fields.get("row_revision", fields["revision"][1]),
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


def cut(w: Path, params: Mapping[str, int], calls: Sequence[Sequence[int]], scheme: str) -> dict[str, Any]:
    """CUT: W at `w` registers exactly the circuit's Pearl+ values of `params` at their sizes, under `scheme`, in 32-bit
    words, and beside them only its version's weights (`pearl_c.window.version_of`). At version 2 those are the B rows of
    the weights the cut's `calls` read, each call (K, N, _) of a position's forward: the circuit reads no B row and so
    names no weight, and W registers a weight once however many calls read it, so a weight's B rows are N rows of K words
    for a call's (K, N), every call's (K, N) has a weight, and no (K, N) has more weights than calls. At version 1 W
    registers no weight. A weight's rows that aren't a call's, or past its calls' count, and a call's (K, N) with no
    weight are among `differ`."""
    from verity.protocols.accounting.pouw.prover.serving.pearl_c import window as PWR

    rec = read_json(w)
    version = PWR.version_of(rec)
    want = {n: size for n, size in params.items() if n.startswith(PEARL_VALUES)}
    weights = {e["value"]: (e["leaves"], e["words"]) for e in rec["registered"] if PEARL_WEIGHTS.fullmatch(e["value"])}
    got = {e["value"]: e["leaves"] * e["words"] * 2 for e in rec["registered"] if e["value"] not in weights}
    differ = sorted(n for n in set(got) | set(want) if got.get(n) != want.get(n))
    read = Counter((int(n), int(k)) for k, n, *_ in calls) if version == PWR.VERSION else Counter()
    held = Counter(weights.values())
    differ += sorted(n for n, shape in weights.items() if held[shape] > read[shape])
    differ += [f"a weight of {n} rows of {k} words, which a call reads" for n, k in sorted(read) if not held[n, k]]
    widths = sorted({e.get("word_bits") for e in rec["registered"]}, key=str)
    of = (rec.get("window") or {}).get("scheme")
    return {"ok": not differ and bool(want) and of == scheme and widths == [32], "values": len(got) + len(weights),
            "circuit_values": len(want), "weights": len(weights), "version": version, "differ": differ[:20],
            "differ_count": len(differ), "scheme": of, "word_bits": widths}


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
                f"{', '.join(v['target_differs']) or 'nothing'}; revision served {str(v['fields']['revision'][0])[:12]}…, "
                f"compared with {str(v['fields']['revision'][1])[:12]}…, the row's workload declaring "
                f"{str(v['row_revision'])[:12]}…){end}")
    if name == "BUILD":
        return (f"BUILD: {a.pred}/predicted.json names Program digest {str(v['names'][0])[:16]}…, manifest digest "
                f"{str(v['names'][1])[:16]}…; the row's Build of record is {v['of_record'][0][:16]}…, "
                f"{v['of_record'][1][:16]}…{end}")
    return (f"CUT: W ({a.window_record}) registers {v['values']} values and the circuit reads {v['circuit_values']} Pearl+ "
            f"values; {v['weights']} more are weights' B rows, which W's version {v['version']} registers and the circuit "
            f"never reads; {v['differ_count']} differ in name or size (scheme {v['scheme']}, word bits {v['word_bits']}){end}")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--serve", type=Path, required=True, help="the serve dir, holding its serve.json")
    ap.add_argument("--row", required=True, help="the row of record's id (its workload file's name)")
    ap.add_argument("--class", dest="cls", required=True, help="the window class, SLOTS,PROMPT,OUTPUTS")
    ap.add_argument("--pearl-tables", required=True, help="Pearl+'s scheme, whose tables the circuit is cut at")
    ap.add_argument("--pred", type=Path, required=True, help="predict row's Build: its predicted.json and its copy of the checkpoint config")
    ap.add_argument("--of-record", required=True, help="the row's Build of record, PROGRAM_DIGEST:MANIFEST_DIGEST")
    ap.add_argument("--window-record", type=Path, required=True, help="W, Pearl+'s window record")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--revision", default=None,
                    help="the revision the serve declares, when the manifests pin it beside the row's (`at_revision`)")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "served_compare.json").unlink(missing_ok=True)
    circuits: dict[str, Any] = {}

    def phi_verdict() -> dict[str, Any]:
        config = checkpoint_config(a.pred)
        read = {"file": str(config), "sha256": hashlib.sha256(config.read_bytes()).hexdigest()}
        fields, served, declared = lowered_facts(read_json(a.serve / "serve.json"), a.row, str(config), a.cls,
                                                 a.pearl_tables, a.revision)
        circuits.update(served=served, declared=declared)
        (a.out / "circuit_facts.json").write_text(json.dumps(circuits, indent=1) + "\n")
        return {**phi(fields, served, declared), "config": read}

    verdicts = {"PHI": judged(phi_verdict), "BUILD": judged(lambda: build(a.pred, a.of_record))}
    verdicts["CUT"] = (judged(lambda: cut(a.window_record, circuits["served"]["params"], circuits["served"]["calls"],
                                          a.pearl_tables)) if circuits
                       else {"ok": False, "error": "PHI lowered no circuit to hold W to"})
    for name in VERDICTS:
        print(said(name, verdicts[name], a), flush=True)
    ok = all(verdicts[n]["ok"] for n in VERDICTS)
    failed = [n for n in VERDICTS if not verdicts[n]["ok"]]
    own = verdicts["PHI"].get("row_revision")
    at = (f" at revision {a.revision[:12]}…" if a.revision and a.revision != own else "")
    whose = "" if not at else f", whose workload declares {own[:12]}…" if own else ", whose declared revision PHI didn't reach"
    line = (f"SERVED-COMPARE: {'EQUAL' if ok else 'NOT EQUAL'} (the serve{at} against row {a.row}'s Build of record{whose}: "
            + ("PHI, BUILD and CUT PASS" if ok else f"{', '.join(failed)} FAIL") + "; B-OF-RECORD is the seal's)")
    (a.out / "served_compare.json").write_text(json.dumps(
        {"row": a.row, "class": a.cls, "scheme": a.pearl_tables, "ok": ok, "verdicts": verdicts, "line": line},
        indent=1, sort_keys=True, default=str) + "\n")
    print(line, flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
