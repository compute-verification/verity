"""A window of a class before any serve, through the three reads a served window's close makes of it that need no card,
no gate and no checkpoint (`rehearsal.sh check`):

    rehearsal_window.py SERVE_PHI CLASS SCHEME MAX_MODEL_LEN,T OUT

The window is the class's fixed load as the harness sends it: SLOTS requests of PROMPT prompt ids, each kept to OUTPUTS
outputs by its cap, all admitted at step 0. Its serve dir is a stand-in holding what the three reads read: serve.json,
SERVE_PHI's fields with the engine record a serve at the settings' engine args states (``--max-model-len``
MAX_MODEL_LEN, ``--max-num-batched-tokens`` T); requests.jsonl, each request's body as the load brings it; and the
window's advice, I1's for that schedule. Its prompt ids, roots and salt are made up: no read here opens a value.

- ROW-DIR: the row dir's Builds hold the Programs the registered compiler compiles. The row dir's Builds are
  `advice_compile`'s, compiled under its phi of serve.json (`advice_compile.phi_of`) and written here from that compile
  (`prove_path.own_builds`, a stand-in for vLLM's trace of each, which needs the pinned vLLM). The compiler is the one
  the window's commit and audit run (`prove_path.i2_of_serve`), which reads each request's Build by the digest of the
  Program it compiles (`I2Compiler.load`).
- SERVED-ROWS: the window's commit reads each request's served rows. As ``prove_path.py window-commit`` does: the
  compiler over its own Builds, the class taking what it compiles (`window_class.takes`), and each request's
  `window_rows.Served.of_record` over its Program, here against a record that binds every member of the Program a row
  a prompt position.
- CUT: W as its writer writes it is a W the served compare takes. The class's circuit cut at Pearl+'s linears
  (`window_circuit.lower`), W written by `pearl_c.window.record` over passes whose calls are that cut's (a pass each
  round the circuit reads, each call at the rows the circuit reads of it and reading a weight of its own), and
  `served_compare.cut` on it.

Each verdict is one line, PASS or FAIL with its refusal; OUT gets them as one JSON object, written after each; the exit
status is 0 only where all three pass. Nothing here reads a run.
"""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "adapters/vllm"), str(HERE), str(ROOT / "benchmarks/prove_path")]

VERDICTS = ("ROW-DIR", "SERVED-ROWS", "CUT")
#: a made-up root and salt: 64 bytes each, which W's writer reads as hex
MADE_UP = "00" * 64


def advice_compile() -> Any:
    """vllm's ``tools/advice_compile.py``, which writes a served window's row dir's Builds, loaded by its file."""
    spec = importlib.util.spec_from_file_location("advice_compile", ROOT / "adapters/vllm/tools/advice_compile.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def serve_dir(d: Path, phi: dict, cls: Any, engine: dict) -> bytes:
    """The stand-in serve dir at `d` (module docstring) -> its window 0's advice, I1's canonical bytes."""
    import verity.protocols.accounting.certifier.verifier.advice as I1

    users = [f"fixed.{j:05d}" for j in range(cls.slots)]
    (d / "advice" / "w0").mkdir(parents=True)
    (d / "advice" / "w0" / "registration.json").write_text(json.dumps({"window": 0}) + "\n")
    serve = {**phi, "engine": {"execution": phi["execution"], "env_flags": {}, "engine_kwargs": engine}}
    (d / "serve.json").write_text(json.dumps(serve, indent=1) + "\n")
    (d / "requests.jsonl").write_text("".join(json.dumps({
        "http_id": f"cmpl-{j}", "user": u, "windows": [0], "prompt_ids": [1 + j] * cls.prompt,
        "ingress": {"brought": {"max_tokens": cls.outputs, "temperature": 0.0, "top_p": 1.0, "seed": j, "stop": [],
                                "ignore_eos": True}}}) + "\n" for j, u in enumerate(users)))
    slots = range(cls.slots)
    steps = [I1.StepAdvice(tuple(I1.Row(r, "prefill", 0, cls.prompt) for r in slots)),
             *(I1.StepAdvice(tuple(I1.Row(r, "decode", cls.prompt + s, 1) for r in slots)) for s in range(cls.outputs - 1))]
    return I1.encode(I1.Advice(tuple(I1.RequestAdvice(u, cls.prompt, cls.outputs, "length") for u in users), tuple(steps),
                               (0,) * (cls.slots * cls.outputs)))


def row_dir(PP: Any, run: Path, data: bytes, row: Path) -> dict[str, Any]:
    """ROW-DIR (module docstring) over the Builds it writes into `row`; its refusal names the fields in which the two
    phi of the one serve.json differ."""
    from verity_vllm.engine.profiles.dense_generic import pinned_hf_config
    from verity_vllm.program.predict.steps import Registered

    AC = advice_compile()
    _advice, meta, bodies, _registration = AC.served_inputs(run, 0)
    config = pinned_hf_config(meta["model"], meta["revision"])
    if config is None:
        raise ValueError(f"{meta['model']}@{meta['revision']} is pinned in no checkpoint manifest: advice_compile "
                         "compiles no Build of it")
    built = AC.phi_of(meta, json.loads(config.read_text()), 0)
    theirs = PP.I2Compiler(Registered.of(built, default_seed=AC.registered_seed(meta), cap_bits=PP.CAP_BITS), bodies,
                           str(row), own=True)
    PP.own_builds(theirs, theirs.requests(data))
    i2 = PP.i2_of_serve(str(run), 0, str(row), None, None, PP.CAP_BITS)
    requests = PP.compile_advice(PP.Capped(i2, PP.CAP_BITS), data)
    try:
        i2.load(0, 1, requests)
    except ValueError as e:
        a, b = built.to_json(), i2.registered.phi.to_json()
        differ = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        raise ValueError(f"{e}; advice_compile's phi of serve.json and the compiler's differ in "
                         f"{', '.join(differ) or 'nothing'}") from None
    return {"requests": len(requests), "programs": len({q["digest"] for q in requests}),
            "builds": len(PP.bodies_by_digest(str(row)))}


class Bound:
    """A record that binds each member of a request's Program a row a prompt position, under the op path the Program
    names it by: all `window_rows.Served` asks of a record before it reads a value."""

    def __init__(self, pi: Any, prompt: int) -> None:
        self.prompt = prompt
        self.width = {(d["op_path"], str(m)): int(width) for r in pi.rows if (d := pi.describe(r))
                      for m, width, _dtype in d["members"]}

    def resolve(self, op: str) -> tuple[str, int]:
        return op, 0

    def has(self, rid: str, step: int, op: str, member: str) -> bool:
        return (op, member) in self.width

    def entries(self, rid: str, step: int, op: str, member: str) -> tuple[list[dict], None]:
        return [{"identity": {"element_range": [0, self.prompt * self.width[(op, member)]]}}], None


def served_rows(PP: Any, run: Path, data: bytes, cls: Any, tables: Any, builds: Path) -> dict[str, Any]:
    """SERVED-ROWS (module docstring), the compiler's own Builds written into `builds`."""
    from verity_vllm.check.replay.index import ProgramIndex
    from verity_vllm.program.predict import window_circuit as WC
    from verity_vllm.program.predict import window_rows as WRS
    from verity_vllm.program.predict.steps import compile_steps

    i2 = PP.i2_of_serve(str(run), 0, str(builds), None, None, PP.CAP_BITS, own=True)
    requests = PP.compile_advice(PP.Capped(i2, PP.CAP_BITS), data)
    WC.takes(compile_steps(i2.registered, data, i2.ingress), cls, tables=tables)
    p = i2.load(0, 1, requests)
    rows = []
    for u in p.progs:
        pi = ProgramIndex(p.progs[u], program_dir=p.dirs[u])
        rows.append([len(WRS.Served.of_record(pi, Bound(pi, cls.prompt), u, p.arrivals[u]).rows), len(pi.rows)])
    return {"requests": len(rows), "served_rows": sorted({r[0] for r in rows}), "program_rows": sorted({r[1] for r in rows})}


def cut(PP: Any, serve: dict, cls: Any, tables: Any, scheme: str, w: Path) -> dict[str, Any]:
    """CUT (module docstring), W written to `w`."""
    import served_compare as SC
    from verity_vllm.program.predict import window_circuit as WC
    from verity_vllm.protocols.pouw.pearl_c import window as PWR

    facts = SC.facts(WC.lower(PP.registered_of_serve(serve, None, None, PP.CAP_BITS), cls, tables=tables))
    params, calls = facts["params"], facts["calls"]
    rounds = sorted({int(n.split("/")[1]) for n in params if n.startswith("pearl-a/")})

    def a_pass(s: int) -> dict[str, Any]:
        matmuls = [{"index": c, "weight": c, "k": k, "n": n, "m": params[PWR.a_value(s, c)] // (2 * k),
                    "real_rows": params[PWR.y_value(s, c)] // (2 * n), "root_a": MADE_UP, "y": {"root": MADE_UP}}
                   for c, (k, n, *_) in enumerate(calls)]
        return {"entry": f"pouw:{scheme}", "scheme": scheme, "salt": MADE_UP, "matmuls": matmuls,
                "weights": {str(c): {"n": n, "k": k, "root": MADE_UP} for c, (k, n, *_) in enumerate(calls)}}

    w.write_text(json.dumps(PWR.record([a_pass(s) for s in rounds], rounds=rounds)) + "\n")
    v = SC.cut(w, params, calls, scheme)
    if not v["ok"]:
        raise ValueError(f"W as its writer writes it at version {v.get('version', PWR.VERSION)} registers {v['values']} "
                         f"values and the circuit reads {v['circuit_values']} Pearl+ values; {v['differ_count']} differ in "
                         f"name or size, {', '.join(map(str, v['differ'][:3]))} first (scheme {v['scheme']}, word bits "
                         f"{v['word_bits']})")
    return {"values": v["values"], "circuit_values": v["circuit_values"], "weights": v["weights"], "version": v["version"],
            "rounds": len(rounds), "calls": len(calls)}


def main(argv: list[str]) -> int:
    phi_file, cls_arg, scheme, engine_arg, out = argv

    import prove_path as PP
    from verity_vllm.program.predict.window_class import PearlTables

    cls, tables = PP.window_class_of(cls_arg), PearlTables(scheme)
    max_model_len, tokens = (int(x) for x in engine_arg.split(","))
    engine = {"max_model_len": max_model_len, "max_num_batched_tokens": tokens}
    doc: dict = {"class": cls_arg, "scheme": scheme, "engine": engine, "verdicts": {}}
    with tempfile.TemporaryDirectory() as s:
        run = Path(s) / "serve"
        data = serve_dir(run, json.loads(Path(phi_file).read_text()), cls, engine)
        serve = json.loads((run / "serve.json").read_text())
        parts = {"ROW-DIR": lambda: row_dir(PP, run, data, Path(s) / "row"),
                 "SERVED-ROWS": lambda: served_rows(PP, run, data, cls, tables, Path(s) / "builds"),
                 "CUT": lambda: cut(PP, serve, cls, tables, scheme, Path(s) / "w.json")}
        for name in VERDICTS:
            try:
                v = {"ok": True, **parts[name]()}
            except Exception as e:  # noqa: BLE001 -- a verdict that can't be reached is a failed one, and the next still runs
                v = {"ok": False, "error": f"{type(e).__name__}: {e}"[:800]}
            doc["verdicts"][name] = v
            Path(out).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
            said = v.get("error") or ", ".join(f"{k} {v[k]}" for k in sorted(v) if k != "ok")
            print(f"{name}: {said}: {'PASS' if v['ok'] else 'FAIL'}", flush=True)
    return 0 if all(v["ok"] for v in doc["verdicts"].values()) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
