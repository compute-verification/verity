"""network-e2e's steps after N2 and the proofs, toward the one command over varied traffic: the advice (I1 v0,
`verity.core.protocols.certifier.advice`), its bits per link-window (v0's `bits_by_window` beside the derived rules'
residual, `encodings/derived.py`), the prove path's audit of a committed run's step units compiled from that advice
(`benchmarks/prove_path`), and the negatives (`benchmarks/network_traces/negatives.py`). Each stage's result says what
it stood on; `stubs` lists what is still a stand-in.

* **The advice** (`advice`, ``--advice FILE``): I1's JSON (`to_json`), its `encode` bytes, or a prove-path run's stand-in
  advice from before v0 fixed ``release`` and ``sources`` (its ``advice/advice.json``), upgraded as
  `advice_windows.advice_of_workload` now writes it: no sources, each kept output released in its step's bucket. It is
  bound to the runs only on the advice profile, by the honest run's certified I/O (``io.json``): every advice request is
  a certified request (its ``user``), once.
* **Bits per link-window** (`bits`), T the honest run's egress window in buckets (`window_buckets`): v0's (I1's
  `bits_by_window`) and, given the registered serving config (``--serving-config``: `derived.Config`'s fields, or a step
  trace's ``config``) and the certified I/O the advice is bound to, whose tokens the residual reads (canonical sources,
  derived hits), the residual's (`derived.residual`), unshaped (release sent) and shaped (release derived), charged by
  I1's rule (`derived.charge`), with the forced fields' share. `derived.reconstruct` must give back every step of the
  advice and every output length from the residual, the config and the tokens, and `derived.rebuild` the whole advice
  (the tie's derivation, in the clear on the rehearsal's published witness).
* **The endpoint's own advice** (`served_advice`, e2e's ``--endpoint-run``): `verity-vllm advice` over a `verity-vllm
  serve-openai` endpoint's records (`ENDPOINT`), one window advice (`verity_vllm.pipeline.window_advice`) a closed
  Commit window, each registered (its hiding root, under the window's Commit) with its opening beside it. The honest
  run's windows are bound to it (each window opened to its registered root, every certified request admitted in
  exactly one window, a window that carries requests in joined to the window before it, `window_chain.join`) and each
  chain of windows rebuilt from its certified I/O under the config read off the endpoint's records
  (`endpoint_config`), its bits placed in the run's link-windows.
* **The prove path** (`prove_path`, ``--prove-path DIR|TGZ``): `prove_path.py audit`'s output adopted (`run.sh`'s ``$OUT``,
  each arm a directory with its ``summary.json``): an honest arm passes when every rank's audit accepted on the verifier's
  coins and every negative was refused; a forged arm when Lean refused the forged unit it sampled. The advice the arm
  committed is held to ``--advice``'s: the stand-in's codec committed the file's bytes (its SHA-512), v0's commitment is
  hiding, so there only its ``total_bits``. That holds it to the file, not to these runs: the prove path is bound to them
  only when the file is their advice and the Commit it proves is theirs, and only a bound prove path counts in the OK.
  Beside the endpoint's own advice, an arm of ``prove_path.py audit --serve-run`` names the served run's Commit window
  it proves (`served_commit`); the prove path is bound to these runs when every arm's Commit is the endpoint's (its
  ``windows.jsonl``) and sealed while serving, every arm's committed advice is held by content to the report's advice
  of its window (`held_to`: its root registered under that Commit, its opening opening it to that advice, which the
  report's OK requires), and an honest arm's is one of the honest run's own Commit windows. When each was registered is
  read off the registration and the endpoint's trace (`closes`), never the job's.
* **The negatives** (`negatives`): `negatives.suite` on the runs, handed the advice's v0 bytes for its advice negatives
  and, given each request's sampled ids (``--ids``), the I/O tie's verdict on every negative it reads (`io_tie`'s
  reference on a fake upstream's cells and Commit port of the advice and those ids, under the map the verifier computes
  from the advice).
* **The I/O tie on the runs** (`tie`, on the advice profile, where the negatives registry doesn't run):
  `negatives.tie_run` on each run, `io_tie`'s reference in the clear on its stand-ins (each window's certified frames
  held to their head, the sampled ids as the Commit's port), with each run's ids from the endpoint's records
  (`served_ids`, the requests that arrived in its link-windows) or ``--ids``, which must give each run's certified
  requests an id for each output they served (`uncovered`). Honest owes acceptance, and token, whose certified ids
  aren't the endpoint's, a refusal by content at exactly the link-windows its changed ids fall in (`changed_windows`).
  The tie is required on the advice profile (`tie_gate`): the report isn't OK when it didn't run, gave the honest run no
  verdict or didn't accept it, or when a part it must establish there (`TIE_REQUIRED`) didn't hold.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import struct
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "benchmarks" / "network_traces"))

import active_replay as ar  # noqa: E402
import negatives as NG  # noqa: E402
import window_chain as WC  # noqa: E402
from verity.core.protocols.certifier import advice as I1  # noqa: E402
from verity.core.protocols.certifier.active import compact_record_data  # noqa: E402
from verity.ml.circuits.codec import canonical_json  # noqa: E402
from verity.ml.commitments import advice as AC  # noqa: E402
from verity_vllm.pipeline import window_advice as WA  # noqa: E402


def _derived():
    spec = importlib.util.spec_from_file_location("enc_derived", HERE / "encodings" / "derived.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


D = _derived()
ADVICE = "advice"  # certifier_serve's varied plan's N2 profile: the egress counts are N2's advised counts' to derive


def _sha512(data: bytes) -> str:
    return hashlib.sha512(data).hexdigest()


# --- the advice -------------------------------------------------------------------------------------------------------

def standin(doc: dict) -> dict:
    """A prove-path stand-in's advice from before v0 (no ``schema``, rows without ``sources``, no ``release``) as I1 v0's
    JSON, as `advice_windows.advice_of_workload` now writes it: each kept output released in its step's bucket (every row
    of the stand-in keeps one: a whole-prompt prefill, then one decode a step)."""
    steps = [{"rows": [{"sources": []} | r for r in st["rows"]]} for st in doc["steps"]]
    return {"schema": I1.SCHEMA, "requests": doc["requests"], "steps": steps,
            "release": doc.get("release") or [s for s, st in enumerate(steps) for _ in st["rows"]]}


def advice(path: Path):
    """The advice at ``path`` and what the report says of it: its form, the SHA-512 of the file and of its v0 encoding,
    its v0 bits and its shape."""
    data = path.read_bytes()
    try:
        doc = json.loads(data)
    except ValueError:
        adv, form = I1.decode(data), "I1 v0 bytes"
    else:
        form = "I1 v0 JSON" if "schema" in doc else "a prove-path stand-in's from before v0, upgraded (chain.standin)"
        adv = I1.from_json(doc if "schema" in doc else standin(doc))
    return adv, {"file": str(path), "form": form, "file_sha512": _sha512(data), "v0_sha512": _sha512(I1.encode(adv)),
                 "bits": I1.total_bits(adv), "requests": len(adv.requests), "steps": len(adv.steps),
                 "outputs": len(adv.release)}


def window_buckets(run: Path) -> int:
    """T, the buckets of the run's egress window, from its calibration report as certifier-serve reads it."""
    return ar.Setup(json.loads((run / "calibration.json").read_text())).egress_params.buckets


def holdings(adv, io: list[dict]):
    """The certified I/O (``io.json``'s requests, in ingress order) as `derived.Holdings`, and each advice request's
    certified index; or None and why, when the advice's requests aren't the certified ones, each once."""
    users = [r["user"] for r in io]
    ids = [q.id for q in adv.requests]
    if sorted(users) != sorted(ids) or len(set(ids)) != len(ids):
        return None, None, (f"the advice's {len(ids)} requests aren't the {len(users)} certified ones, each once")
    held = D.Holdings(tuple(D.CertifiedRequest(r["user"], tuple(r["prompt"]), r["max_tokens"]) for r in io),
                      tuple(tuple(r["output"]) for r in io))
    return held, tuple(users.index(i) for i in ids), None


def bits(adv, T: int, windows: int, config: dict | None, io: list[dict] | None) -> dict:
    """The advice's bits per egress window of T buckets: v0's, and with ``config`` and an ``io`` it is bound to the
    residual's (the module's docstring); ``io``, the certified I/O, binds the advice to the run, and its tokens are what
    the residual reads and rebuilds the advice from."""
    v0 = I1.bits_by_window(adv, T)
    out = {"T": T, "v0_total": I1.total_bits(adv), "config": config, "bound": None, "public_fields": None,
           "rebuilt": None, "why": None}
    held = order = None
    if io is None:
        out["bound_why"] = "no certified I/O to hold it to: only a varied run's check writes one (io.json)"
    else:
        held, order, out["bound_why"] = holdings(adv, io)
        out["bound"] = held is not None
    dv = ds = None
    if config is None:
        out["why"] = "no registered serving config (--serving-config)"
    elif held is None:
        out["why"] = "the residual reads the certified tokens, and the advice isn't bound to a certified I/O"
    else:
        cfg = D.config_of({"config": config})
        try:
            res = D.residual(adv, cfg, held, order=order)
            shaped = D.residual(adv, cfg, held, order=order, shaped=True)
            walk = D.reconstruct(res, cfg, held)
            dv = D.charge(adv, T, D.derived_records(res, walk, cfg))
            ds = D.charge(adv, T, D.derived_records(shaped, walk, cfg))
        except ValueError as err:
            out["why"], out["public_fields"], out["rebuilt"] = f"the residual under this config: {err}", False, False
        else:
            out["public_fields"] = walk.steps == adv.steps and walk.output_len == tuple(q.output_len for q in adv.requests)
            try:
                out["rebuilt"] = D.rebuild(res, cfg, held, ()) == adv
            except ValueError as err:
                out["why"], out["rebuilt"] = f"the rebuild from the certified I/O: {err}", False

    def total(d, w, fields=None):
        return None if d is None else sum(b for f, b in d.get(w, {}).items() if fields is None or f in fields)

    out["windows"] = [{"window": w, "v0_bits": v0.get(w, 0), "derived_bits": total(dv, w),
                       "derived_shaped_bits": total(ds, w), "forced_bits": total(dv, w, D.FORCED),
                       "in_run": w < windows} for w in sorted(set(range(windows)) | set(v0))]
    return out


# --- the endpoint's own advice ----------------------------------------------------------------------------------------

ENDPOINT = ("advice_trace.jsonl", "requests.jsonl", "windows.jsonl", "serve.json")
SLACK_S = 1.0  # the endpoint's clock against the certifier's: a Commit window opening this soon before the run's origin
COUNTS = ("v0_bits", "derived_bits", "derived_shaped_bits", "forced_bits")
ROW_ORDER = {"runner": "batch", "scheduling": "scheduling"}


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def endpoint_config(recs: list[dict], serve: dict, eos: list[int]) -> dict:
    """The serving config `derived.Config` takes, read off the endpoint's records, since it registers none: its scheduler
    trace's header (``recs``, ``advice_trace.jsonl``), ``serve.json``'s ``max_model_len``, the plan's EOS ids, and I1's
    rows in the order the trace's steps record them."""
    head = recs[0]["config"]
    orders = {r.get("order") for r in recs if r["kind"] == "step"}
    if len(orders) != 1 or next(iter(orders)) not in ROW_ORDER:
        raise ValueError(f"the trace's steps record their rows in {sorted(map(str, orders))}, not one of {sorted(ROW_ORDER)}")
    return {"block_size": head["block_size"], "num_gpu_blocks": head["num_gpu_blocks"],
            "max_num_batched_tokens": head["max_num_batched_tokens"],
            "max_model_len": serve["engine"]["engine_kwargs"]["max_model_len"], "eos_token_id": eos[0],
            "stop_token_ids": list(eos[1:]), "scheduler": "async" if head.get("async_scheduling") else "sync",
            "long_prefill_token_threshold": head.get("long_prefill_token_threshold", 0),
            "max_num_seqs": head["max_num_seqs"], "enable_prefix_caching": head["enable_prefix_caching"],
            "row_order": ROW_ORDER[next(iter(orders))]}


def opens(reg: dict, opening: dict, adv) -> bool:
    """Whether a Commit window's opening opens its registered root (``reg``, its public registration) to ``adv``, its
    window advice (`window_advice.Advice`, or an I1 v0 advice, a window that carries nothing): the opening's bytes are
    ``adv``'s encoding (`window_advice.encode`, I1 v0's for a window that carries nothing), committed as
    `advice_windows.commit_advice` commits them (``row_bits`` rows, the last zero-padded, under the binding of the
    scheme to the registered compiler)."""
    data = bytes.fromhex(opening["advice"])
    if data != (I1.encode(adv) if type(adv) is I1.Advice else WA.encode(adv)) or opening.get("scheme") != reg["scheme"]:
        return False
    w = reg["row_bits"] // 8
    rows = [data[i:i + w].ljust(w, b"\0") for i in range(0, max(len(data), 1), w)]
    binding = hashlib.sha512(canonical_json({"scheme": reg["scheme"], "compiler": reg["binding"]["compiler"]})).digest()
    got = AC.register(rows, reg["row_bits"], binding, [bytes.fromhex(s) for s in opening["salts"]])
    return got.root.hex() == reg["binding"]["advice"]["root"]


def _first_steps(recs: list[dict]) -> dict[int, float]:
    """Each Commit window's first step's clock (unix seconds): the step with the least index from its open to its
    close, as `verity-vllm advice` cuts the windows (its ``windows_of``) and counts their release from."""
    clock = {r["step"]: r["t"] for r in recs if r["kind"] == "step" and "t" in r}
    edge = {(int(r["window"]), r["event"]): int(r["before_step"]) for r in recs if r["kind"] == "window"}
    out = {}
    for (w, event), lo in edge.items():
        hi = edge.get((w, "close"))
        steps = [s for s in clock if s >= lo and (hi is None or s < hi)] if event == "open" else []
        if steps:
            out[w] = clock[min(steps)]
    return out


def served_advice(ep: Path, adv_dir: Path, deployment: dict, io: list[dict], T: int, eos: list[int]) -> tuple[dict, dict]:
    """The endpoint's own advice for the honest run, as `advice` and `bits` report one: ``adv_dir`` is `verity-vllm
    advice`'s over the endpoint's records ``ep`` (`ENDPOINT`), one window advice (`window_advice`) a closed Commit
    window. The run's windows are those whose first step falls in its link-windows (``deployment``'s origin and span,
    `SLACK_S`). A window that carries requests in joins the window before it into a chain (``chain``), joined into one
    I1 v0 advice (`window_chain.join`): its carried requests and their state are the earlier window's, as that window's
    registered advice closed them, never its own string's, so each request and step is charged once. The advice is
    bound to the run when each window's requests are certified ones, every certified request is admitted in exactly one
    window, each window's opening opens its registered root and its registration names the window's Commit
    (``windows.jsonl``), and each chain joins: a window that carries a request in follows the window before it, which
    the run holds, and no window is left owing a carry. Each chain's advice is rebuilt from its certified I/O under the
    config read off the endpoint (`endpoint_config`) by `bits`, charged per bucket (T 1) and placed at its first
    window's first step's bucket, each later window's release counted from its own first step's, so its bits land in
    the link-windows of T buckets I1's rule puts them in, its release at its own width; a chain's bits are on its first
    window's row."""
    recs = _jsonl(ep / "advice_trace.jsonl")
    serve = json.loads((ep / "serve.json").read_text())
    index = json.loads((adv_dir / "windows.json").read_text())
    commits = {int(d["window"]): {"run_id": d.get("run_id"), "run_root": d.get("run_root")}
               for d in _jsonl(ep / "windows.jsonl")}
    first = _first_steps(recs)
    origin, span = deployment["origin_unix"], deployment["windows"] * deployment["window_s"]
    bucket_s = deployment["window_s"] / T
    users = [r["user"] for r in io]
    certified = set(users)
    out = {"T": T, "served": serve.get("run_id"), "config": None, "config_from": "the endpoint's records (its trace's "
           "header, serve.json's max_model_len, the plan's EOS ids), not one it registered", "bound": False,
           "bound_why": None, "public_fields": None, "rebuilt": None, "why": None, "commit_windows": [], "windows": []}
    try:
        out["config"] = config = endpoint_config(recs, serve, eos)
    except (KeyError, ValueError) as err:
        out["why"], config = f"no serving config off the endpoint's records: {err!r}", None
    problems, rows, links, seen, loaded = [], [], {}, {u: 0 for u in users}, []

    def place(charged: dict[int, dict], at: int) -> None:
        for x in charged:
            lw = links.setdefault((x["window"] + at) // T, dict.fromkeys(COUNTS, 0))
            for k in COUNTS:
                lw[k] = None if lw[k] is None or x.get(k) is None else lw[k] + x[k]

    for w in sorted(w for w, t in first.items() if origin - SLACK_S <= t < origin + span):
        if str(w) not in index["windows"]:
            why = (index.get("refused") or {}).get(str(w)) or "still open"
            problems.append(f"Commit window {w} has no advice ({why})")
            rows.append({"window": w, "refused": why})
            continue
        d = adv_dir / f"w{w}"
        reg = json.loads((d / "registration.json").read_text())
        adv = WA.from_json(json.loads((d / "private" / "advice.json").read_text()))
        ids = [q.id for q in adv.requests]
        for i in ids[len(adv.carried):]:
            if i in seen:
                seen[i] += 1
        foreign = [i for i in ids if i not in certified]
        opened = opens(reg, json.loads((d / "private" / "opening.json").read_text()), adv)
        at = math.floor((first[w] - origin) / bucket_s)
        row = {"window": w, "commit": reg.get("commit"), "commit_is_the_windows": reg.get("commit") == commits.get(w),
               "root": reg["binding"]["advice"]["root"], "opened": opened, "requests": len(ids) - len(adv.carried),
               "carried": len(adv.carried), "steps": len(adv.steps), "outputs": len(adv.release),
               "committed_bits": WA.total_bits(adv), "t0_s": round(first[w] - origin, 3), "bucket": at,
               "link_window": at // T, "chain": [w], "bound": None, "public_fields": None, "rebuilt": None, "why": None,
               **dict.fromkeys(COUNTS, 0)}
        rows.append(row)
        if adv.carried and loaded and loaded[-1][-1][0] == w - 1:
            loaded[-1].append((w, adv, row))
        else:
            loaded.append([(w, adv, row)])
        if foreign:
            problems.append(f"Commit window {w} holds {len(foreign)} requests that aren't the run's certified ones "
                            f"({', '.join(foreign[:3])})")
        if not opened:
            problems.append(f"Commit window {w}'s opening doesn't open its registered root to its advice")
        if not row["commit_is_the_windows"]:
            problems.append(f"Commit window {w}'s registration names another Commit than the endpoint's windows.jsonl")
    for chained in loaded:
        at, ws, last = chained[0][2]["bucket"], [w for w, _, _ in chained], {}

        def released(k: int, rid: str, _: int, b: int, chained=chained, at=at, last=last) -> int:
            last[rid] = max(last.get(rid, 0), b + chained[k][2]["bucket"] - at)
            return last[rid]

        try:
            adv = WC.join([(w, a) for w, a, _ in chained], released)
        except I1.MalformedAdvice as err:
            problems.append(str(err))
            for w, a, row in chained:
                place([{"window": b, "v0_bits": n} for b, n in WA.bits_by_window(a, 1).items()], row["bucket"])
                row.update(chain=ws, bound=False, why=str(err), v0_bits=row["committed_bits"],
                           **dict.fromkeys(COUNTS[1:]))
            continue
        b = bits(adv, 1, 0, config, [r for r in io if r["user"] in {q.id for q in adv.requests}])
        place(b["windows"], at)
        for _, _, row in chained:
            row.update(chain=ws, **{k: b[k] for k in ("bound", "public_fields", "rebuilt", "why")})
        chained[0][2].update({k: sum(x[k] for x in b["windows"]) if all(x[k] is not None for x in b["windows"])
                              else None for k in COUNTS})
    missing, twice = [u for u, n in seen.items() if n == 0], [u for u, n in seen.items() if n > 1]
    if not rows:
        problems.append("no Commit window of the endpoint's opened in the run's link-windows")
    if missing:
        problems.append(f"{len(missing)} certified requests are in none of the run's Commit windows ({', '.join(missing[:3])})")
    if twice:
        problems.append(f"{len(twice)} certified requests are in more than one ({', '.join(twice[:3])})")
    ran = [r for r in rows if "refused" not in r]
    out["bound"] = not problems and all(r["bound"] is True for r in ran)
    out["bound_why"] = "; ".join(problems) or next((f"Commit window {r['window']}: {r['why']}" for r in ran
                                                    if r["bound"] is not True), None)
    out["public_fields"] = bool(ran) and all(r["public_fields"] is True for r in ran)
    out["rebuilt"] = bool(ran) and all(r["rebuilt"] is True for r in ran)
    out["why"] = out["why"] or next((f"Commit window {r['window']}: {r['why']}" for r in ran if r["why"]), None)
    out["commit_windows"] = rows
    out["v0_total"] = sum(r["v0_bits"] for r in ran)
    out["chains"] = [[w for w, _, _ in c] for c in loaded if len(c) > 1]
    n = deployment["windows"]
    out["windows"] = [{"window": w, **links.get(w, dict.fromkeys(COUNTS, 0)), "in_run": w < n}
                      for w in sorted(set(range(n)) | set(links))]
    info = {"file": str(adv_dir), "form": "the endpoint's own (verity-vllm advice): window advice, one a Commit "
            "window, a chain of windows that carry requests across their close joined into one I1 v0 advice",
            "served": serve.get("run_id"), "windows": len(ran), "bits": out["v0_total"],
            "requests": sum(r["requests"] for r in ran), "steps": sum(r["steps"] for r in ran),
            "outputs": sum(r["outputs"] for r in ran), "file_sha512": None, "v0_sha512": None}
    return info, out


# --- the prove path ---------------------------------------------------------------------------------------------------

def _refused(v) -> bool:
    return v if isinstance(v, bool) else bool((v or {}).get("refused"))


def _same(committed: dict | None, info: dict | None) -> bool | None:
    if not committed or info is None or info.get("served"):
        return None
    if "digest" in committed:
        return committed["digest"] in (info["file_sha512"], info["v0_sha512"])
    return committed.get("bits") == info["bits"] if "bits" in committed else None


def per_unit(rank: dict) -> dict[str, dict]:
    """A rank's timing per sampled proof block, by member, as the prove path's `per_unit` reads it: each drawn
    member's prover and Lean verify seconds over the units its one statement proves (``draw_counts``; C-Flock's own
    ``units`` count its virtual units), and its proof's bytes over its reps."""
    out = {}
    for key, t in (rank.get("timing") or {}).items():
        tm, n = t.get("timing") or {}, (rank.get("draw_counts") or {}).get(key) or 0
        pb = tm.get("proof_bytes")
        out[key] = {"units": n, "proof_bytes": sum(pb) if isinstance(pb, list) else pb,
                    **{f"{x}_per_unit": round(v / n, 2) if n and isinstance(v, int | float) else None
                       for x, v in (("prove_s", tm.get("prove_total_s")), ("lean_verify_s", t.get("lean_verify_s")))}}
    return out


def closes(recs: list[dict]) -> dict[int, float]:
    """Each Commit window's close in the endpoint's scheduler trace (``recs``), unix seconds: the clock of the first
    record after its close event (the engine resumes once the window is sealed), or of the last before it when none
    follows."""
    out, waiting, last = {}, [], None
    for r in recs:
        if "t" in r:
            out |= dict.fromkeys(waiting, r["t"])
            waiting, last = [], r["t"]
        if r["kind"] == "window" and r["event"] == "close":
            waiting.append(int(r["window"]))
    return out | ({} if last is None else dict.fromkeys(waiting, last))


def _utc(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def serve_record(d: Path) -> dict:
    """The serving run's own record (``d``, the `research run` directory of the serve): by window, what its advice loop
    registered (``out/advice_windows.jsonl``: the root and ``at``, the UTC time of day to the second, stamped once the
    registration was in place), the last window it logged, and when its session ended (``status.json``'s ``ended_utc``,
    None while it serves)."""
    p = d / "out" / "advice_windows.jsonl"
    lines = _jsonl(p) if p.exists() else []
    ended = (_read_json(d / "status.json") or {}).get("ended_utc")
    return {"dir": str(d), "windows": {int(x["window"]): x for x in lines if x.get("rc") == 0 and x.get("root")},
            "last": max((int(x["window"]) for x in lines), default=None),
            "ended": None if not ended else datetime.fromisoformat(ended.replace("Z", "+00:00")).timestamp()}


def _stamped(at: str, near: float) -> float:
    """A run record's stamp ``at`` (``HH:MM:SSZ``, no date) as unix seconds: the instant with that UTC time of day
    nearest ``near``."""
    h, m, s = (int(x) for x in at.rstrip("Z").split(":"))
    day = datetime.fromtimestamp(near, timezone.utc).replace(hour=h, minute=m, second=s, microsecond=0).timestamp()
    return min((day - 86400, day, day + 86400), key=lambda t: abs(t - near))


def held_to(w: int, root: str | None, given: str | None, adv_dir: Path | None, commit: dict | None,
            closed: float | None, next_closed: float | None = None, before_draw: bool | None = None,
            record: dict | None = None, session_end: float | None = None) -> dict:
    """A ``--serve-run`` arm's committed advice (its ``root``) held by content to the report's advice of its Commit
    window ``w`` (`verity-vllm advice` over the endpoint's records, ``adv_dir/w<w>``): the registration it was committed
    under (``given``, a `verity-vllm advice` directory's ``registration.json`` and ``private/opening.json``, else the
    report's own) has that root and names the window's Commit (``commit``, the endpoint's ``windows.jsonl``), and its
    opening opens the root to the report's advice (`opens`). Its v0 bit count is never enough. When it was registered
    (``at``) is, ahead of the registration's file time (``file_at``; it has no time field), what the serving run's own
    record attests (``record``, `serve_record`: its advice loop's stamp of that root, to the second, so no later than a
    second past it), read against the endpoint's trace (`closes`): its lag after its own window's close (``closed``) is
    always reported, and it was at serve time when it was registered before its window's sample (``before_draw``, the
    arm's own `advice_before_draw`) and no later than the next window's close (``next_closed``), or, for the session's
    last window, its end (``session_end``); after serving when the sample came first or both times are past that bound,
    and unknown (None) otherwise."""
    out = {"registration": None, "the_registered_root": None, "names_the_commit": None, "opens": None, "tied": False,
           "tied_why": None, "at": None, "file_at": None, "attested": None,
           "closed": None if closed is None else _utc(closed), "after_close_s": None,
           "next_closed": None if next_closed is None else _utc(next_closed), "before_next_close_s": None,
           "session_end": None, "before_session_end_s": None, "before_draw": before_draw, "at_serve_time": None}
    report = None if adv_dir is None else adv_dir / f"w{w}"
    src = Path(given) if given else report
    if report is None or not (report / "private" / "advice.json").exists():
        out["tied_why"] = f"the report has no advice of Commit window {w} to hold it to"
        return out
    try:
        reg = json.loads((src / "registration.json").read_text())
        out["registration"], m = str(src), (src / "registration.json").stat().st_mtime
        out["the_registered_root"] = root is not None and root == reg["binding"]["advice"]["root"]
        out["names_the_commit"] = commit is not None and reg.get("commit") == commit
        out["opens"] = opens(reg, json.loads((src / "private" / "opening.json").read_text()),
                             WA.from_json(json.loads((report / "private" / "advice.json").read_text())))
    except (OSError, ValueError, KeyError, TypeError) as err:
        out["tied_why"] = f"no registration of Commit window {w}'s advice to hold it to at {src} ({err!r})"
        return out
    stamp = ((record or {}).get("windows") or {}).get(w)
    t = m
    if stamp and stamp.get("root") == reg["binding"]["advice"]["root"]:
        t = _stamped(stamp["at"], m) + 1.0
        out["attested"] = {"stamp": stamp["at"], "record": record.get("dir")}
    out["at"], out["file_at"] = _utc(t), _utc(m)
    if closed is not None:
        out["after_close_s"] = round(t - closed, 1)
    bound = next_closed
    if next_closed is not None:
        out["before_next_close_s"] = round(next_closed - t, 1)
    elif session_end is not None:
        bound, out["session_end"], out["before_session_end_s"] = session_end, _utc(session_end), round(session_end - t, 1)
    if out["attested"] is not None:
        out["attested"]["within"] = None if bound is None else t <= bound
    if before_draw is False or (bound is not None and t > bound and m > bound):
        out["at_serve_time"] = False
    elif before_draw is True and bound is not None and t <= bound:
        out["at_serve_time"] = True
    why = ([] if out["the_registered_root"] else [f"its advice root isn't the one registered at {src}"]) \
        + ([] if out["names_the_commit"] else [f"the registration at {src} doesn't name Commit window {w}'s Commit "
                                                "(the endpoint's windows.jsonl)"]) \
        + ([] if out["opens"] else [f"the registration's opening at {src} doesn't open its root to the report's advice "
                                    f"of Commit window {w}"])
    out["tied"], out["tied_why"] = not why, "; ".join(why) or None
    return out


def served_commit(s: dict, bits: dict, commits: dict[int, dict], registered: dict, adv_dir: Path | None = None,
                  closed: dict[int, float] | None = None) -> dict:
    """A ``--serve-run`` arm's Commit (its summary's ``serve``: the window it proves, its run id and the root sealed
    while serving) held to the endpoint's records: the endpoint's own when its ``windows.jsonl`` (``commits``) names that
    window with that run id and root, and one of the honest run's when `served_advice` (``bits``) has it among the run's
    Commit windows. Its committed advice is held by content to the report's advice of that window (`held_to`, the
    report's advice directory ``adv_dir``, each window's close ``closed``), whether registered by the report's own run
    or apart. ``registered`` is, by window, who registered its advice (``by``, ``ref``), the registration's directory
    when it isn't the report's (``registration``), the serving run's own record (``record``, its run directory,
    `serve_record`) and the arm's first draw (``first_draw``), as the job states them (``--prove-path-registered``);
    whether it was at serve time is never the job's to say: its sample followed the registration when every rank's
    `advice_before_draw` (the verifier's receipt of it before any coin) says so. A window with no next close in the
    endpoint's records is bounded by the session's end only when the record says the session ended with this window
    the last its advice loop logged."""
    serve = s["serve"]
    d = serve.get("detail") or {}
    w = d.get("window", serve.get("window"))
    rec = commits.get(w) or {}
    row = next((x for x in bits.get("commit_windows") or () if x.get("window") == w and "refused" not in x), None)
    adv = d.get("advice") or s.get("advice") or {}
    g = registered.get(str(w)) or {}
    order = [(x.get("advice_before_draw") or {}).get("ok") for x in (s.get("ranks") or {}).values()]
    record = serve_record(Path(g["record"])) if g.get("record") else None
    nxt = (closed or {}).get(w + 1) if isinstance(w, int) else None
    held = held_to(w, adv.get("root"), g.get("registration"), adv_dir,
                   {"run_id": rec["run_id"], "run_root": rec["run_root"]} if rec else None, (closed or {}).get(w), nxt,
                   False if False in order else None if not order or None in order else all(x is True for x in order),
                   record,
                   record["ended"] if record and nxt is None and record["last"] == w else None)
    return {"window": w, "run_id": d.get("run_id"), "run_root": s.get("run_root"),
            "sealed_while_serving": serve.get("ok") is True and not serve.get("codes"),
            "the_endpoints": bool(rec) and rec.get("run_id") == d.get("run_id")
            and rec.get("run_root") == d.get("published_root") == s.get("run_root"),
            "of_the_honest_run": row is not None, "advice_root": adv.get("root"), "advice_bits": adv.get("bits"),
            "the_reports_root": None if row is None else adv.get("root") == row["root"],
            **{k: held[k] for k in ("registration", "the_registered_root", "names_the_commit", "opens", "tied",
                                    "tied_why")},
            "registered": {k: g.get(k) for k in ("by", "ref", "first_draw")}
            | {k: held[k] for k in ("at", "file_at", "attested", "closed", "after_close_s", "next_closed",
                                    "before_next_close_s", "session_end", "before_session_end_s", "before_draw",
                                    "at_serve_time")}}


def prove_path(src: Path | list[Path], info: dict | None, advice_bound: bool | None = None, bits: dict | None = None,
               ep: Path | None = None, registered: dict | None = None) -> dict:
    """`prove_path.py audit`'s output at ``src`` (one directory or several), one row per arm and rank (the module's
    docstring). Each arm's committed advice is held to the ``--advice`` file (``info``), which says nothing of these
    runs: the prove path is ``bound`` to them only when that file is these runs' advice (``advice_bound``, `bits`'
    ``bound``) and the Commit it proves is theirs, which nothing here checks (no Commit record of these runs is an
    input), so it is never bound and its ``ok`` stays out of the report's. Beside the endpoint's own advice
    (`served_advice`'s ``bits``, its records ``ep``) an arm of the fixed workload proves its own Commit, not this run's;
    an arm of ``prove_path.py audit --serve-run`` names the served run's Commit window it proves (`served_commit`, named
    ``w<k> <arm>``), and the prove path is bound to these runs when every arm's is the endpoint's and sealed while
    serving, every arm's committed advice is held by content to the report's advice of its window (`held_to`: ``tied``,
    which the report's OK requires of every such arm), and an honest arm's is one of the honest run's."""
    srcs = [src] if isinstance(src, Path) else list(src)
    served = info is not None and bool(info.get("served"))
    commits = {int(d["window"]): d for d in _jsonl(ep / "windows.jsonl")} if served and ep is not None else {}
    closed = closes(_jsonl(ep / "advice_trace.jsonl")) if served and ep is not None else {}
    adv_dir = Path(info["file"]) if served and info.get("file") else None
    arms = {}
    for root in srcs:
        for p in sorted(x for x in root.iterdir() if x.is_dir()):
            try:
                s = json.loads((p / "summary.json").read_text())
            except (OSError, ValueError):
                continue
            if "ranks" not in s:
                continue
            ranks = {}
            for r, x in sorted(s["ranks"].items(), key=lambda kv: int(kv[0])):
                a = x.get("audit") or {}
                ranks[r] = {"ok": x.get("ok") is True, "accepted": a.get("accepted"), "population": x.get("population"),
                            "drawn": len(x.get("draw") or ()),
                            "wrong_units_bound": (a.get("wrong_units") or {}).get("bound"),
                            "negatives_refused": bool(x.get("negatives")) and all(map(_refused, x["negatives"].values())),
                            "forged_refused": (x.get("forged") or {}).get("ok") if s.get("forged") else None,
                            "seconds": x.get("seconds"), "source": x.get("source"), "per_unit": per_unit(x)}
            committed = s.get("advice") or {}
            arm = {"ok": s.get("ok") is True and bool(ranks), "forged": bool(s.get("forged")),
                   "world": s.get("world"), "rule": s.get("rule"), "delta": s.get("delta"),
                   "compiler": (s.get("compiler") or {}).get("name"),
                   "tp_run_root": s.get("tp_run_root") or s.get("run_root"),
                   "advice": s.get("advice"), "committed_is_the_advice_file": _same(committed, info),
                   "advice_held_by": "the file's SHA-512" if "digest" in committed else
                   "its bits only: the committed root hides it, and its opening is the prover's" if "root" in committed
                   else None, "gap": s.get("gap"), "ranks": ranks}
            name = p.name
            if served and "serve" in s:
                arm["commit"] = served_commit(s, bits or {}, commits, registered or {}, adv_dir, closed)
                name = f"w{arm['commit']['window']} {p.name}"
            arms[f"{name} ({root})" if name in arms else name] = arm
    if not arms:
        raise SystemExit(f"{', '.join(map(str, srcs))}: no prove-path arm (a directory with prove_path.py audit's "
                         "summary.json)")
    given = [a["committed_is_the_advice_file"] for a in arms.values()]
    committed = None if None in given else all(given)
    roots = sorted({str(a["tp_run_root"])[:16] + "…" for a in arms.values() if a["tp_run_root"]})
    bound, why = False, (
        f"from the fixed workload, not this run: its arms prove their own workload's advice and Commit (run root "
        f"{', '.join(roots) or '—'}), not the endpoint's" if served else
        "its arms' committed advice isn't the --advice file" if committed is not True else
        "the --advice file isn't these runs' advice" if advice_bound is not True else
        "the Commit it proves isn't held to these runs' (no Commit record of these runs is an input)")
    tied = tied_why = None
    if served and all("commit" in a for a in arms.values()):
        problems = []
        for n, a in arms.items():
            c = a["commit"]
            if not c["sealed_while_serving"]:
                problems.append(f"{n}: its run root isn't the one sealed while serving")
            if not c["the_endpoints"]:
                problems.append(f"{n}: Commit window {c['window']} isn't the endpoint's (its windows.jsonl)")
        untied = [f"{n}: {a['commit']['tied_why']}" for n, a in arms.items() if not a["commit"]["tied"]]
        tied, tied_why = not untied, "; ".join(untied) or None
        problems += untied
        if not any(a["commit"]["of_the_honest_run"] for a in arms.values() if not a["forged"]):
            problems.append("no honest arm proves one of the honest run's Commit windows")
        bound, why = not problems, "; ".join(problems) or None
    return {"dir": ", ".join(map(str, srcs)), "arms": arms, "committed_is_the_advice_file": committed, "bound": bound,
            "bound_why": why, "tied": tied, "tied_why": tied_why,
            "ok": all(a["ok"] for a in arms.values()) and any(not a["forged"] for a in arms.values())
            and False not in given}


# --- the negatives ----------------------------------------------------------------------------------------------------

def negatives(runs: dict[str, Path], work: Path, adv, clock, ids: Path | None = None) -> dict:
    """`negatives.suite` on ``runs`` (by strategy), its advice negatives made from ``adv``'s v0 bytes, and with ``ids``
    (each request's sampled ids, `negatives.read_ids`) the I/O tie's verdict on the negatives it reads. ``ok`` unless a
    row that ran says otherwise: a pending check's row (``ok`` None) or one that didn't run is the report's to list, not
    a failure."""
    path = None
    if adv is not None:
        work.mkdir(parents=True, exist_ok=True)
        path = work / "advice.bin"
        path.write_bytes(I1.encode(adv))
    rows = NG.suite(runs, work / "negatives", advice=path, ids=ids, clock=clock)
    return {"ran": True, "format": NG.FORMAT, "rows": rows,
            "ok": all(r.get("ok") is not False for r in rows.values() if r.get("ran"))}


# --- the I/O tie on the runs ------------------------------------------------------------------------------------------

#: the live tampers only the I/O tie refuses, each owed a refusal by content at exactly the link-windows it changed
TIE_OWED = ("token",)
#: what the I/O tie must establish on the honest run beside its acceptance, for the report to be OK (`tie_gate`): each a
#: key of `negatives.tie_run`'s row that must be true, with the report's reason when it isn't. The compact
#: finish (each response's length and finish held to the advice, `active.compact_finish_violations`) goes here if the tie reports it as a key of its own; as a failing part it already refuses the honest run.
TIE_REQUIRED: dict[str, str] = {
    "finish": "the I/O tie didn't establish the honest run's finish (`negatives.tie_run`'s finish: on a compact run "
              "its records held to its sampled ids, `active.compact_finish_violations`)"}


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def served_rows(ep: Path, deployment: dict) -> dict[str, dict]:
    """Each ``requests.jsonl`` row by its ``user``, of the requests that reached the endpoint in a run's link-windows
    (``deployment``'s origin and span, `SLACK_S`): its scheduler trace's ``arrive`` records by engine id, and the rows
    by the same. Every strategy of a rehearsal sends the same users, so a run's rows are the ones that arrived in its
    own span; a user arriving twice in it is refused."""
    origin, span = deployment["origin_unix"], deployment["windows"] * deployment["window_s"]
    arrived = {r["request"] for r in _jsonl(ep / "advice_trace.jsonl")
               if r["kind"] == "arrive" and origin - SLACK_S <= r["t"] < origin + span}
    out: dict[str, dict] = {}
    for r in _jsonl(ep / "requests.jsonl"):
        if r["engine_id"] in arrived and r.get("user"):
            if r["user"] in out:
                raise ValueError(f"{r['user']} reached the endpoint twice in the run's link-windows")
            out[r["user"]] = r
    return out


def served_ids(ep: Path, deployment: dict) -> dict[str, list[int]]:
    """Each request's sampled ids (``streamed_ids``) by its ``user``, of a run's `served_rows`."""
    return {u: list(r["streamed_ids"]) for u, r in served_rows(ep, deployment).items()}


def _output_stream(slot: bytes, compact: bool) -> int | None:
    """The stream (request ordinal) of a certified egress frame that `negatives.tie_run` places as its stream's next
    output: on a compact plan every frame, each a compact record (`active.compact_record_data`), else each id event;
    None for another frame of a plan of id events. Raises ValueError for a frame neither reads: a compact plan's frame
    that isn't a record, or one `decode_frame` can't parse."""
    if compact:
        got = compact_record_data(slot)
        if got is None:
            raise ValueError(f"a compact plan's egress frame that isn't a compact record ({slot[:16].hex()})")
        return got[0]
    try:
        return NG.id_event_data(NG.decode_frame(slot)[2])[0] if NG._is_id_event(slot) else None
    except struct.error as err:
        raise ValueError(f"an egress frame that doesn't decode ({err}): {slot[:16].hex()}") from err


def changed_windows(run: Path) -> list[int] | None:
    """The link-windows a token run's changed ids fall in: each output its shim logged as changed (``server.json``'s
    ``tampered``, (session, index)), placed as `negatives.tie_run` places the certified egress's id events, or on a
    compact plan its compact records (`_output_stream`), each its stream's next output and the stream's user the
    certified I/O's. None when the shim logged none. Raises ValueError for an egress frame it can't place."""
    changed = (_read_json(run / "server.json") or {}).get("tampered")
    if not changed:
        return None
    plan, report = (json.loads((run / f).read_text()) for f in ("plan.json", "calibration.json"))
    setup = ar.Setup(report)
    e, i = setup.egress_params, setup.ingress_params
    ins = [ar.unpack_record(b, i.frame_bytes) for b in ar.read_blobs(run / "requests.records")]
    outs = [ar.unpack_record(b, e.frame_bytes) for b in ar.read_blobs(run / "responses.records")]
    users = {r["stream"]: r["user"] for r in NG.cs.certified_io(plan, setup, ins, outs)["records"]}
    compact, at, nxt = bool(plan.get("compact")), {}, defaultdict(int)
    for rec in sorted(outs, key=lambda r: r.window):
        for q in (_output_stream(s, compact) for row in rec.grid for s in row if s is not None):
            if q is not None:
                at[users.get(q), nxt[q]] = rec.window
                nxt[q] += 1
    return sorted({at[u, k] for u, k in changed if (u, k) in at})


def uncovered(run: Path, ids: dict[str, list[int]]) -> str | None:
    """Why ``ids`` don't cover the run's certified requests (its ``io.json``, which the rehearsal's check writes on the
    advice profile: each request's user and served ``output``), or None when every one has a sampled id for each of
    its served outputs. `negatives.tie_run` gives each sampled id one row of the Commit's port and has no backstop for
    a served output with none (its unplaced count is committed outputs no cell carries), so ids that miss a request,
    or name it with fewer ids than it served, leave those outputs untied."""
    io = _read_json(run / "io.json")
    if not isinstance(io, dict) or not isinstance(io.get("requests"), list):
        return "no certified I/O (io.json) to hold the sampled ids to"
    reqs = io["requests"]
    if not reqs:
        return "no certified request in its io.json to tie"
    if not all(isinstance(r, dict) and isinstance(r.get("output"), list) for r in reqs):
        return "a certified request in its io.json without its served outputs to hold the sampled ids to"
    missing = sum(r.get("user") not in ids for r in reqs)
    short = [(len(ids[r["user"]]), len(r["output"])) for r in reqs
             if r.get("user") in ids and len(ids[r["user"]]) < len(r["output"])]
    why = [f"no sampled ids for {missing}"] * bool(missing) + \
        [f"fewer sampled ids than served outputs for {len(short)}"] * bool(short)
    return None if not why else f"{' and '.join(why)} of its {len(reqs)} certified requests" + (
        f" ({sum(n for n, _ in short)} sampled, {sum(m for _, m in short)} served)" if short else "")


def tie(runs: dict[str, Path], ep: Path | None, ids: Path | None, why: str | None = None) -> dict:
    """The I/O tie's verdict on each run (`negatives.tie_run`: `io_tie`'s reference in the clear, on
    `negatives.TIE_STAND_INS`), each run's sampled ids and client aborts ``ids``' (`negatives.read_ids` and
    `negatives.read_aborted`, one map for every run) or else the endpoint's records' ``ep`` for its own link-windows
    (`served_rows`, `negatives.commit_aborted`), and whether it is the verdict the run owes
    (`negatives.tie_ok`): honest's accepted, a negative's it reads refused by its part, and token's (`TIE_OWED`) at
    exactly the link-windows its changed ids fall in (`changed_windows`). A run has no verdict when its ids don't cover
    its certified requests (`uncovered`), when it lacks what the tie opens, or when its tie raises; the row says why.
    Not run without `io_tie` in the tree or without the ids (``why``: why the endpoint's records aren't here). On the
    advice profile the tie is required: `tie_gate` says what keeps it from passing the report."""
    if NG.io_tie_module() is None:
        return {"ran": False, "why": NG.TIE_MISSING["io-tie"]}
    if ids is None and ep is None:
        return {"ran": False, "why": f"no sampled ids ({why or 'neither --endpoint-run nor --ids given'})"}
    given, rows = None if ids is None else (NG.read_ids(ids), NG.read_aborted(ids)), {}
    for name, d in runs.items():
        row = {"verdict": NG.NOT_APPLICABLE, "ok": None, "codes": [], "failing": {}, "windows": [], "unplaced": None,
               "changed": None, "requests": None, "outputs": None, "placed": None, "held": None, "missing": {},
               "why": None}
        try:
            if given is not None:
                got, gone = given
            else:
                served = served_rows(ep, json.loads((d / "deployment.json").read_text()))
                got = {u: list(r["streamed_ids"]) for u, r in served.items()}
                gone = {u for u, r in served.items() if NG.commit_aborted(r)}
            lacking = uncovered(d, got)
            if lacking:
                rows[name] = row | {"requests": len(got), "why": lacking}
                continue
            t = NG.tie_run(d, got, aborted=gone)
            changed = changed_windows(d) if name in TIE_OWED else None
        except (OSError, KeyError, ValueError, struct.error) as err:
            rows[name] = row | {"why": repr(err)}
            continue
        ok = NG.tie_ok(name, t)
        if name in TIE_OWED and ok:
            ok = changed is not None and sorted(t["failing"]) == changed
        rows[name] = row | {k: t.get(k) for k in ("verdict", "codes", "failing", "unplaced", "held", "missing",
                                                  "outputs", "placed", *TIE_REQUIRED)} | {
            "ok": ok, "windows": sorted(w for w in t["failing"] if w >= 0), "changed": changed, "requests": len(got),
            "why": "; ".join(t["missing"].values()) or None}
    return {"ran": True, "why": None,
            "from": "the endpoint's records (each run's requests by their arrival in its link-windows)" if given is None
            else str(ids), "stand_ins": NG.TIE_STAND_INS, "runs": rows}


def tie_gate(t: dict | None) -> list[str]:
    """Why the I/O tie (`tie`'s result ``t``) keeps the report from being OK on the advice profile, where it is required;
    empty when it passes. It fails when it didn't run (no `io_tie` in the tree, no sampled ids), gave the honest run no
    verdict, didn't accept it, or when a part it must establish beside its verdict (`TIE_REQUIRED`) didn't hold on it.
    The report reads the tie's pass here only: a required part the tie gains fails the report through this."""
    if not t or not t.get("ran"):
        return [f"the I/O tie didn't run, and the advice profile requires it: {(t or {}).get('why') or 'no tie stage'}"]
    honest = t["runs"].get("honest")
    if honest is None or honest["verdict"] not in ("accepted", "refused"):
        return [f"the I/O tie gave no verdict on the honest run: {(honest or {}).get('why') or 'no honest run'}"]
    if honest["ok"] is not True:
        return ["the I/O tie didn't accept the honest run"]
    return [why for k, why in TIE_REQUIRED.items() if honest.get(k) is not True]


# --- what's still a stand-in ------------------------------------------------------------------------------------------

def stubs(r: dict, varied_client: bool) -> list[str]:
    """What the report stood on that isn't yet the chain's own, from what ``r`` (the report) says ran."""
    out, c = [], r.get("chain") or {}
    if not varied_client:
        out.append("Varied traffic: certifier_serve.py's `rehearse --traffic varied` (Poisson arrivals, "
                   "prompts as ids, real EOS) isn't in this tree.")
    if r.get("profile") == ADVICE:
        out.append("N2 on the advice profile: N2's advised counts (NetworkAdvisedCounts_v1) aren't run, so no "
                   "link-window has N2's verdict and nothing is proved by n2_zk; the rehearsal's own check (certified "
                   "I/O, audit, salt) is the verdict.")
    a, b = c.get("advice"), c.get("bits") or {}
    served = "served" in b
    if a is None and not served:
        out.append("Advice: none given (--advice, or --endpoint-run for the endpoint's own); a served run's comes from "
                   "its step trace (`verity-vllm advice`).")
    elif not b.get("bound"):
        out.append(f"Advice: {a['file'] if a else 'the endpoint’s'} is not these runs' ({b.get('bound_why')})"
                   + ("." if served else "; a served run's comes from the committing endpoint's step trace "
                                         "(`verity-vllm advice`)."))
    if a is not None and "upgraded" in a["form"]:
        out.append("Advice form: the stand-in's pre-v0 JSON, upgraded to v0 here; the bits are v0's of the upgrade, while "
                   "the prove path committed the file's bytes.")
    if served and b.get("config") is not None:
        out.append(f"Derived bits: the serving config is read off {b['config_from']}.")
    elif (a is not None or served) and b.get("config") is None:
        out.append("Derived bits: no serving config given (--serving-config); the endpoint registers none yet.")
    elif a is not None:
        out.append("Derived bits: the serving config is the one given (--serving-config), not one the endpoint "
                   "registered.")
    if served:
        out.append("Advice in the clear: the rehearsal opens each Commit window's advice to its report, where only its "
                   "registration is public; the compile unit and N2's advised counts, which take it hidden, aren't run "
                   "here.")
    out.append("Coded bits: the chain doesn't run the advice code (encodings/code_v1.py).")
    p = c.get("prove_path")
    if p is None:
        out.append("Prove path: not given (--prove-path, a prove_path.py run's output).")
    elif served and p.get("bound"):
        cs = {a["commit"]["window"]: a["commit"] for a in p["arms"].values()}
        late = [f"w{w}'s advice was registered {_registered(x['registered'])}"
                + (", not by the endpoint while serving" if (x.get("registered") or {}).get("at_serve_time") is False
                   else "")
                for w, x in sorted(cs.items()) if (x.get("registered") or {}).get("at_serve_time") is not True]
        apart = [f"w{w}" for w, x in sorted(cs.items()) if x["of_the_honest_run"] and not x["the_reports_root"]]
        out.append(f"Prove path: the served run's own Commit windows ({', '.join(f'w{w}' for w in sorted(cs))}), "
                   f"adopted from {p['dir']}" + "".join(f"; {x}" for x in late)
                   + (f"; the advice {', '.join(apart)}'s arms committed is a registration of its own, held to the "
                      "report's advice by its opening, which the report reads in the clear" if apart else "") + ".")
    elif served:
        out.append(f"Prove path: adopted from {p['dir']}, {p['bound_why']}: the endpoint's Commit windows have no sealed "
                   "replay bundle a prover can open, so its step units aren't proved here.")
    else:
        sources = sorted({x["source"] for arm in p["arms"].values() for x in arm["ranks"].values() if x["source"]})
        out.append(f"Prove path: adopted from {p['dir']}, a run of its own (sources {', '.join(sources) or '—'}); it "
                   "proves a TP Commit's step units compiled from that run's advice, not a commit of these runs: the "
                   "committing endpoint's Commit and its own advice (`verity-vllm advice`) are what make it these "
                   "runs'.")
    n = c.get("negatives") or {}
    if not n.get("ran"):
        out.append(f"Negatives (the negatives registry): {n.get('why') or 'not run'}.")
    else:
        pending = sorted(k for k, x in n["rows"].items() if not x.get("ran") or x.get("ok") is None)
        if pending:
            out.append(f"Negatives pending or not run here: {', '.join(pending)} (each row says what it waits on).")
        ties = [x["tie"] for x in n["rows"].values() if "tie" in x]
        if not ties:
            out.append("I/O tie: no sampled ids given (--ids), so no negative has the tie's verdict.")
        elif not all(t["ran"] for t in ties):
            out.append(f"I/O tie: {next(t['why'] for t in ties if not t['ran'])}.")
        else:
            out.append("I/O tie: `io_tie`'s reference in the clear on a fake upstream's cells and Commit port of the "
                       "advice and the given ids, not the warden's cells under its root and the endpoint's Commit "
                       "proved in an IoTieWindow unit.")
    t = c.get("tie")
    if t is not None and t["ran"]:
        out.append("I/O tie: `io_tie`'s reference in the clear on the negatives' stand-ins (each window's certified "
                   "frames held to their head and committed under fake salts in place of the warden's cells, the "
                   f"sampled ids from {t['from']} under fake salts in place of the Commit's port), not the endpoint's "
                   "Commit and the warden's cells in an IoTieWindow unit.")
    where = str(r.get("where") or "")
    if where.startswith("off the prover's node"):
        out.append("Auditor host: the certifier ran on another node than the endpoint's, through an ssh forward, not on "
                   "the auditor host (--from takes a gate run's fetch).")
    elif not where.startswith("gate: "):
        out.append("Auditor host: the certifier ran off the gate (--from takes a gate run's fetch).")
    return out


# --- the report's sections --------------------------------------------------------------------------------------------

def _cell(x) -> str:
    return "—" if x is None else str(x)


def _lag(s: float | None) -> str:
    if s is None:
        return "its Commit window's close not in the endpoint's records"
    return f"{abs(s)} s {'after' if s >= 0 else 'before'} its Commit window closed"


def _registered(g: dict | None) -> str:
    """Who registered a Commit window's advice and when (`served_commit`'s ``registered``): ``by`` and the arm's
    ``first_draw`` as the job states them; the registration's time ``at`` (its serving run's record's stamp, to the
    second, when ``attested``, else its file's), always with its lag in seconds after the window's close in the
    endpoint's records (``after_close_s``); and ``at_serve_time``: before its sample (``before_draw``) and no later than
    the next window's close (``before_next_close_s``) or, for the session's last window, its end
    (``before_session_end_s``)."""
    if not g:
        return "—"
    by, ref = g.get("by") or "whom the job doesn't say", f" ({g['ref']})" if g.get("ref") else ""
    if g.get("at") is None:
        return f"by {by}{ref}, when unknown: no registration read"
    a = g.get("attested")
    lag = ("no later than " if a else "") + _lag(g.get("after_close_s"))
    stamped = f" (its run record's stamp, {a['stamp']} to the second)" if a else ""
    nxt, end = g.get("before_next_close_s"), g.get("before_session_end_s")
    margin, bound = (nxt, "the next window closed") if nxt is not None else \
        (end, "its session ended, with no window after it") if end is not None else (None, None)
    if g.get("at_serve_time"):
        return (f"by {by} at serve time{ref}, {lag}, before its draw and {'at least ' if a else ''}{margin} s before "
                f"{bound}{stamped}")
    if g.get("at_serve_time") is None:
        unknown = ([] if margin is not None else ["the next window's close isn't in the endpoint's records"]) \
            + ([] if g.get("before_draw") is not None else ["no draw receipt orders it against its draw"]) \
            + ([f"its run record's stamp, {a['stamp']} to the second, puts it up to {abs(margin)} s after {bound}, "
                f"though its file's time, {g['file_at']}, is before"] if a and a.get("within") is False else [])
        when = f", no later than {g['at']} ({_lag(g.get('after_close_s'))})" if a else f" at {g['at']}, {lag}"
        return f"by {by}{ref}{when}; whether at serve time is unknown: {' and '.join(unknown)}"
    lead = None if not g.get("first_draw") else round(
        (datetime.fromisoformat(g["first_draw"]) - datetime.fromisoformat(g["at"])).total_seconds(), 1)
    late = ([] if g.get("before_draw") is not False else ["after its draw"]) \
        + ([f"{abs(margin)} s after {bound}"]
           if margin is not None and (margin < 0 or g.get("before_draw") is not False) else [])
    return (f"after serving, by {by}{ref}, {lag}, {' and '.join(late)}{stamped}"
            + (f", {lead} s before the first draw" if lead is not None and lead >= 0 else ""))


def markdown(c: dict) -> list[str]:
    """The chain's sections of the report's Markdown."""
    lines = []
    a, b = c.get("advice"), c.get("bits")
    served = b is not None and "served" in b
    if served:
        lines += ["", "## Advice", "",
                  (f"The endpoint's own ({b['served']}): `verity-vllm advice` over its records, one window advice a "
                   f"closed Commit window, at {a['file']}. The run's Commit windows (those opening in its link-windows): "
                   f"{a['windows']}, with {a['requests']} requests, {a['steps']} steps, {a['outputs']} kept outputs, "
                   f"{a['bits']} bits in v0." if a else "The endpoint's own: none.")
                  + f" Bound to these runs: {_cell(b['bound'])} ({b['bound_why'] or 'every window opened to its registered root under its Commit, every advice request a certified one, each admitted in one window'})."
                  + (" A window that carries requests in is joined to the window before it into one I1 v0 advice, "
                     "its carried requests the earlier window's (windows "
                     + ", ".join("–".join(map(str, c)) for c in b["chains"]) + "): each chain's v0 and derived bits "
                     "are on its first window's row, the committed bits each window's own string's."
                     if b.get("chains") else ""),
                  "", "| Commit window | Commit run id | opens at s | link-window | requests admitted | carried in | "
                  "steps | committed bits | v0 bits | derived bits | derived, shaped | forced | opened | bound | rebuilt |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        lines += [f"| {x['window']} | refused: {x['refused']} |" + " — |" * 13 if "refused" in x else
                  f"| {x['window']} | {_cell((x['commit'] or {}).get('run_id'))} | {x['t0_s']} | {x['link_window']} | "
                  f"{x['requests']} | {_cell(x.get('carried'))} | {x['steps']} | {_cell(x.get('committed_bits'))} | "
                  f"{x['v0_bits']} | {_cell(x['derived_bits'])} | "
                  f"{_cell(x['derived_shaped_bits'])} | {_cell(x['forced_bits'])} | {x['opened']} | {_cell(x['bound'])} "
                  f"| {_cell(x['rebuilt'])} |" for x in b["commit_windows"]]
    elif a is not None:
        lines += ["", "## Advice", "",
                  f"From {a['file']} ({a['form']}): {a['requests']} requests, {a['steps']} steps, {a['outputs']} kept "
                  f"outputs, {a['bits']} bits in v0. SHA-512 of the file {a['file_sha512'][:16]}…, of its v0 encoding "
                  f"{a['v0_sha512'][:16]}…. Bound to these runs: {_cell(b['bound'])} ({b['bound_why'] or 'by the certified I/O'})."]
    if b is not None:
        cfg = b["config"]
        lines += ["", "## Advice bits per link-window", "",
                  f"T = {b['T']} buckets a window. v0: I1's `bits_by_window` ({b['v0_total']} bits in all). Derived: "
                  "the derived rules' residual at its least widths, charged by I1's rule, with the release sent "
                  "(unshaped) and derived (shaped); forced: the residual's fields only witness fixes. "
                  + ("Each Commit window's advice is charged per bucket and placed at its first step's bucket. "
                     if served else "") + "Serving config: "
                  + (f"{json.dumps(cfg, sort_keys=True)}, read off {b['config_from']}, so these bits and the "
                     "residual's own check under it are outside the OK" if cfg and served else
                     f"{json.dumps(cfg, sort_keys=True)}, the one the job was given (--serving-config), not one the "
                     "endpoint registered, so these bits and the residual's own check under it are outside the OK"
                     if cfg else "none") + f". Every step and output length given "
                  f"back by the residual and the config alone: {_cell(b['public_fields'])}; the whole advice rebuilt "
                  f"from the residual and the certified I/O: {_cell(b['rebuilt'])}."
                  + (f" {b['why']}." if b["why"] else ""),
                  "", "| window | v0 bits | derived bits | derived bits, shaped | of them forced |", "|---|---|---|---|---|"]
        lines += [f"| {x['window']}{'' if x['in_run'] else ' (past the run)'} | {x['v0_bits']} | {_cell(x['derived_bits'])} "
                  f"| {_cell(x['derived_shaped_bits'])} | {_cell(x['forced_bits'])} |" for x in b["windows"]]
    p = c.get("prove_path")
    if p is not None:
        commit = [(n, a["commit"]) for n, a in p["arms"].items() if "commit" in a]
        named = {n: a["gap"] for n, a in p["arms"].items() if a.get("gap")}
        gaps = (f" Every arm's gap: {next(iter(named.values()))}." if len(named) == len(p["arms"]) > 1
                and len(set(named.values())) == 1 else "".join(f" {n}'s gap: {g}." for n, g in named.items()))
        if served and commit:
            lines += ["", "## Prove path", "",
                      f"`prove_path.py audit --serve-run`'s output adopted from {p['dir']}: each arm a Commit window of "
                      f"the endpoint's served run ({b['served']}), its step units compiled from that window's advice by "
                      "the registered compiler, drawn on the verifier's own coins, proved --zk behind the Lean firewall "
                      f"and verified by Lean. Bound to these runs: {_cell(p['bound'])}"
                      + (" (every arm's Commit the endpoint's, its run root the one sealed while serving, its advice "
                         "root registered under that Commit and opened to the report's advice of its window, and an "
                         "honest arm's one of the honest run's own Commit windows)" if p["bound"] else
                         f" ({p['bound_why']})" + (", so its ok is not the report's" if p.get("tied") is not False
                                                   else ", and its advice isn't the report's, so the report isn't OK"))
                      + "." + gaps,
                      "", "| arm | Commit window | Commit run id | run root | sealed while serving | the endpoint's | "
                      "one of the honest run's | advice root | v0 bits | the report's root | the registered root | "
                      "registered under its Commit | opens to the report's advice | advice registered |",
                      "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            lines += [f"| {n} | {x['window']} | {_cell(x['run_id'])} | {str(x['run_root'])[:16]}… | "
                      f"{x['sealed_while_serving']} | {x['the_endpoints']} | {x['of_the_honest_run']} | "
                      f"{str(x['advice_root'])[:16]}… | {_cell(x['advice_bits'])} | {_cell(x['the_reports_root'])} | "
                      f"{_cell(x['the_registered_root'])} | {_cell(x['names_the_commit'])} | {_cell(x['opens'])} | "
                      f"{_registered(x['registered'])} |" for n, x in commit]
        else:
            lines += ["", "## Prove path" + (" (from the fixed workload, not this run)" if served else ""), "",
                      f"`prove_path.py audit`'s output adopted from {p['dir']}: each rank's step units compiled from the "
                      "advice by the registered compiler, drawn on the verifier's own coins, proved --zk behind the Lean "
                      "firewall and verified by Lean. The advice each arm committed is the --advice file: "
                      f"{_cell(p['committed_is_the_advice_file'])}. Bound to these runs: {_cell(p['bound'])}"
                      + ("" if p["bound"] else f" ({p['bound_why']}), so its ok is not the report's")
                      + "".join(f"; {n} held by {a['advice_held_by']}" for n, a in p["arms"].items()
                                if a.get("advice_held_by")) + "." + gaps]
        lines += ["", "| arm | ok | rank | audit accepted | units | drawn | wrong units at most | negatives refused | "
                  "forged unit refused | s | prove s a drawn unit | Lean verify s a drawn unit | proof bytes | source |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]

        def each(x, k):
            return _cell("; ".join(f"{m} {_cell(v[k])}" for m, v in x["per_unit"].items()) or None)
        for name, arm in p["arms"].items():
            lines += [f"| {name} | {arm['ok']} | {r} | {_cell(x['accepted'])} | {_cell(x['population'])} | {x['drawn']} | "
                      f"{_cell(x['wrong_units_bound'])} | {x['negatives_refused']} | {_cell(x['forged_refused'])} | "
                      f"{_cell(x['seconds'])} | {each(x, 'prove_s_per_unit')} | {each(x, 'lean_verify_s_per_unit')} | "
                      f"{each(x, 'proof_bytes')} | {_cell(x['source'])} |" for r, x in arm["ranks"].items()]
    n = c.get("negatives")
    if n is not None:
        lines += ["", "## Negatives (the negatives registry)", ""]
        if not n.get("ran"):
            lines.append(f"Not run: {n.get('why')}.")
        else:
            lines += ["Each tampering, the check that must refuse it and its code; ok: refused by that code (a pending "
                      "check's row, ok —, says what it waits on).", "",
                      "| negative | family | tampers | code | ran | N2 accepted | refused windows | codes | I/O tie | ok | "
                      "waits on, or why not run |", "|---|---|---|---|---|---|---|---|---|---|---|"]
            for name, x in n["rows"].items():
                lines.append(f"| {name} | {x['family']} | {x['tampers']} | `{x['code']}` | {x.get('ran')} | "
                             f"{_cell(x.get('n2_accepted'))} | {_cell(x.get('refused_windows'))} | "
                             f"{_cell(', '.join(x['codes']) if x.get('codes') else None)} | {_tie(x.get('tie'))} | "
                             f"{_cell(x.get('ok'))} | {_cell(x.get('why') or ', '.join(x['waits']) or None)} |")
    t = c.get("tie")
    if t is not None:
        lines += ["", "## I/O tie", ""]
        if not t["ran"]:
            lines.append(f"Didn't run: {t['why']}. The advice profile requires it, so the report isn't OK.")
        else:
            lines += ["`negatives.tie_run` on each run: `io_tie`'s reference in the clear, the sampled ids from "
                      f"{t['from']}. Required on the advice profile: the report is OK only when the tie accepts the "
                      "honest run, its ids covering every certified request. Owed: honest accepted, token refused by "
                      "content at exactly the link-windows its changed ids fall in, drop refused by content; late and "
                      "extra owe nothing here. Window -1: committed outputs no certified cell carries.", "",
                      "| run | verdict | ok | codes | refused link-windows (parts) | changed ids' link-windows | requests "
                      "| outputs | placed | unplaced | frames held egress/ingress | why |",
                      "|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for name, x in t["runs"].items():
                failing = "; ".join(f"{w}: {'+'.join(p)}" for w, p in sorted(x["failing"].items(), key=lambda i: int(i[0])))
                held = x["held"] or {}
                lines.append(f"| {name} | {x['verdict']} | {_cell(x['ok'])} | "
                             f"{_cell(', '.join(f'`{c}`' for c in x['codes']) or None)} | {_cell(failing or None)} | "
                             f"{_cell(x['changed'])} | {_cell(x['requests'])} | {_cell(x['outputs'])} | "
                             f"{_cell(x['placed'])} | {_cell(x['unplaced'])} | "
                             f"{_cell(held.get('egress'))}/{_cell(held.get('ingress'))} | {_cell(x['why'])} |")
    return lines


def _tie(t: dict | None) -> str:
    """A negative's I/O tie cell: the parts that refused it, ``accepts``, or why it didn't run."""
    if t is None:
        return "—"
    if not t["ran"]:
        return f"not run: {t['why']}"
    return ", ".join(f"`{c}`" for c in t["codes"]) or "accepts"
