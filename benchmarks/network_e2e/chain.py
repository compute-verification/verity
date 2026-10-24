"""network-e2e's stages after N2 and the proofs, toward the one command over varied traffic (G-A): the advice (I1 v0,
`verity.core.protocols.certifier.advice`), its bits per link-window (v0's `bits_by_window` beside ENC-DERIVED's residual,
`encodings/derived.py`), the prove path's audit of a committed run's step units compiled from that advice (PROVE-PATH,
`benchmarks/prove_path`), and NEG's negatives (`benchmarks/network_traces/negatives.py`). Each stage's result says what it
stood on; `stubs` lists what is still a stand-in.

* **The advice** (`advice`, ``--advice FILE``): I1's JSON (`to_json`), its `encode` bytes, or a prove-path run's stand-in
  advice from before v0 fixed ``release`` and ``sources`` (r20261024-052710-8932's ``advice/advice.json``), upgraded as
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
  serve-openai` endpoint's records (`ENDPOINT`), one advice a closed Commit window, each registered (its hiding root,
  under the window's Commit) with its opening beside it. The honest run's windows are bound to it (each window opened
  to its registered root, every certified request in exactly one window) and each rebuilt from its certified I/O under
  the config read off the endpoint's records (`endpoint_config`), its bits placed in the run's link-windows.
* **The prove path** (`prove_path`, ``--prove-path DIR|TGZ``): `prove_path.py audit`'s output adopted (`run.sh`'s ``$OUT``,
  each arm a directory with its ``summary.json``): an honest arm passes when every rank's audit accepted on the verifier's
  coins and every negative was refused; a forged arm when Lean refused the forged unit it drew. The advice the arm
  committed is held to ``--advice``'s: the stand-in's codec committed the file's bytes (its SHA-512), v0's commitment is
  hiding, so there only its ``total_bits``. That holds it to the file, not to these runs: the prove path is bound to them
  only when the file is their advice and the Commit it proves is theirs, and only a bound prove path counts in the OK.
* **The negatives** (`negatives`): NEG's `suite` on the runs, handed the advice's v0 bytes for its advice negatives and,
  given each request's sampled ids (``--ids``), the I/O tie's verdict on every negative it reads (IO-TIE's reference on
  a fake upstream's cells and Commit port of the advice and those ids, under the map the verifier computes from the
  advice).
* **The I/O tie on the runs** (`tie`, on the advice profile, where NEG's registry doesn't run): NEG's `tie_run` on each
  run, IO-TIE's reference in the clear on its stand-ins (each window's certified frames held to their head, the sampled
  ids as the Commit's port), with each run's ids from the endpoint's records (`served_ids`, the requests that arrived in
  its link-windows) or ``--ids``. Honest owes acceptance, and token, whose certified ids aren't the endpoint's, a
  refusal by content at exactly the link-windows its changed ids fall in (`changed_windows`).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "benchmarks" / "network_traces"))

import active_replay as ar  # noqa: E402
import negatives as NG  # noqa: E402
from verity.core.protocols.certifier import advice as I1  # noqa: E402
from verity.ml.circuits.codec import canonical_json  # noqa: E402
from verity.ml.commitments import advice as AC  # noqa: E402


def _derived():
    spec = importlib.util.spec_from_file_location("enc_derived", HERE / "encodings" / "derived.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


D = _derived()
ADVICE = "advice"  # certifier_serve's varied plan's N2 profile: the egress counts are N2-ADVICE's to derive


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
    """Whether a Commit window's opening opens its registered root (``reg``, its public registration) to ``adv``: the
    opening's bytes are ``adv``'s v0 encoding, committed as `advice_windows.commit_advice` commits them (``row_bits``
    rows, the last zero-padded, under the binding of the scheme to the registered compiler)."""
    data = bytes.fromhex(opening["advice"])
    if data != I1.encode(adv) or opening.get("scheme") != reg["scheme"]:
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
    advice`'s over the endpoint's records ``ep`` (`ENDPOINT`), one I1 v0 advice a closed Commit window. The run's
    windows are those whose first step falls in its link-windows (``deployment``'s origin and span, `SLACK_S`). The
    advice is bound to the run when each window's requests are certified ones, every certified request is in exactly
    one window, and each window's opening opens its registered root and its registration names the window's Commit
    (``windows.jsonl``). Each window's advice is rebuilt from its certified I/O under the config read off the endpoint
    (`endpoint_config`) by `bits`, charged per bucket (T 1) and placed at its first step's bucket, so its bits land in
    the link-windows of T buckets I1's rule puts them in, its release at its own width."""
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
    problems, rows, links, seen = [], [], {}, {u: 0 for u in users}
    for w in sorted(w for w, t in first.items() if origin - SLACK_S <= t < origin + span):
        if str(w) not in index["windows"]:
            why = (index.get("refused") or {}).get(str(w)) or "still open"
            problems.append(f"Commit window {w} has no advice ({why})")
            rows.append({"window": w, "refused": why})
            continue
        d = adv_dir / f"w{w}"
        reg = json.loads((d / "registration.json").read_text())
        adv = I1.from_json(json.loads((d / "private" / "advice.json").read_text()))
        ids = [q.id for q in adv.requests]
        for i in ids:
            if i in seen:
                seen[i] += 1
        foreign = [i for i in ids if i not in certified]
        opened = opens(reg, json.loads((d / "private" / "opening.json").read_text()), adv)
        b = bits(adv, 1, 0, config, [r for r in io if r["user"] in set(ids)])
        at = math.floor((first[w] - origin) / bucket_s)
        for x in b["windows"]:
            lw = links.setdefault((x["window"] + at) // T, dict.fromkeys(COUNTS, 0))
            for k in COUNTS:
                lw[k] = None if lw[k] is None or x[k] is None else lw[k] + x[k]
        row = {"window": w, "commit": reg.get("commit"), "commit_is_the_windows": reg.get("commit") == commits.get(w),
               "root": reg["binding"]["advice"]["root"], "opened": opened, "requests": len(ids),
               "steps": len(adv.steps), "outputs": len(adv.release), "t0_s": round(first[w] - origin, 3),
               "bucket": at, "link_window": at // T, "bound": b["bound"], "public_fields": b["public_fields"],
               "rebuilt": b["rebuilt"], "why": b["why"]}
        rows.append(row | {k: sum(x[k] for x in b["windows"]) if all(x[k] is not None for x in b["windows"])
                           else None for k in COUNTS})
        if foreign:
            problems.append(f"Commit window {w} holds {len(foreign)} requests that aren't the run's certified ones "
                            f"({', '.join(foreign[:3])})")
        if not opened:
            problems.append(f"Commit window {w}'s opening doesn't open its registered root to its advice")
        if not row["commit_is_the_windows"]:
            problems.append(f"Commit window {w}'s registration names another Commit than the endpoint's windows.jsonl")
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
    n = deployment["windows"]
    out["windows"] = [{"window": w, **links.get(w, dict.fromkeys(COUNTS, 0)), "in_run": w < n}
                      for w in sorted(set(range(n)) | set(links))]
    info = {"file": str(adv_dir), "form": "the endpoint's own (verity-vllm advice): I1 v0, one advice a Commit window",
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
    """A rank's timing per drawn unit, by member, as PROVE-PATH's fixed `per_unit` reads it (92e4e50ca5): each drawn
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


def prove_path(src: Path, info: dict | None, advice_bound: bool | None = None) -> dict:
    """`prove_path.py audit`'s output at ``src``, one row per arm and rank (the module's docstring). Each arm's
    committed advice is held to the ``--advice`` file (``info``), which says nothing of these runs: the prove path is
    ``bound`` to them only when that file is these runs' advice (``advice_bound``, `bits`' ``bound``) and the Commit it
    proves is theirs, which nothing here checks yet (no Commit record of these runs is an input), so it is never bound
    and its ``ok`` stays out of the report's. Beside the endpoint's own advice (`served_advice`) there is no file to
    hold it to: its arms prove their own fixed workload, not this run."""
    arms = {}
    for p in sorted(x for x in src.iterdir() if x.is_dir()):
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
                        "drawn": len(x.get("draw") or ()), "wrong_units_bound": (a.get("wrong_units") or {}).get("bound"),
                        "negatives_refused": bool(x.get("negatives")) and all(map(_refused, x["negatives"].values())),
                        "forged_refused": (x.get("forged") or {}).get("ok") if s.get("forged") else None,
                        "seconds": x.get("seconds"), "source": x.get("source"), "per_unit": per_unit(x)}
        committed = s.get("advice") or {}
        arms[p.name] = {"ok": s.get("ok") is True and bool(ranks), "forged": bool(s.get("forged")),
                        "world": s.get("world"), "rule": s.get("rule"), "delta": s.get("delta"),
                        "compiler": (s.get("compiler") or {}).get("name"),
                        "tp_run_root": s.get("tp_run_root") or s.get("run_root"),
                        "advice": s.get("advice"), "committed_is_the_advice_file": _same(committed, info),
                        "advice_held_by": "the file's SHA-512" if "digest" in committed else
                        "its bits only: the committed root hides it, and its opening is the prover's" if "root" in committed
                        else None, "gap": s.get("gap"), "ranks": ranks}
    if not arms:
        raise SystemExit(f"{src}: no prove-path arm (a directory with prove_path.py audit's summary.json)")
    given = [a["committed_is_the_advice_file"] for a in arms.values()]
    committed = None if None in given else all(given)
    roots = sorted({str(a["tp_run_root"])[:16] + "…" for a in arms.values() if a["tp_run_root"]})
    why = (f"from the fixed workload, not this run: its arms prove their own workload's advice and Commit (run root "
           f"{', '.join(roots) or '—'}), not the endpoint's" if info is not None and info.get("served") else
           "its arms' committed advice isn't the --advice file" if committed is not True else
           "the --advice file isn't these runs' advice" if advice_bound is not True else
           "the Commit it proves isn't held to these runs' (no Commit record of these runs is an input)")
    return {"dir": str(src), "arms": arms, "committed_is_the_advice_file": committed, "bound": False,
            "bound_why": why, "ok": all(a["ok"] for a in arms.values()) and any(not a["forged"] for a in arms.values())
            and False not in given}


# --- the negatives ----------------------------------------------------------------------------------------------------

def negatives(runs: dict[str, Path], work: Path, adv, clock, ids: Path | None = None) -> dict:
    """NEG's `suite` on ``runs`` (by strategy), its advice negatives made from ``adv``'s v0 bytes, and with ``ids`` (each
    request's sampled ids, `negatives.read_ids`) the I/O tie's verdict on the negatives it reads. ``ok`` unless a row
    that ran says otherwise: a pending check's row (``ok`` None) or one that didn't run is the report's to list, not a
    failure."""
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


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def served_ids(ep: Path, deployment: dict) -> dict[str, list[int]]:
    """Each request's sampled ids by its ``user``, of the requests that reached the endpoint in a run's link-windows
    (``deployment``'s origin and span, `SLACK_S`): its scheduler trace's ``arrive`` records by engine id, and
    ``requests.jsonl``'s ``streamed_ids`` by the same. Every strategy of a rehearsal sends the same users, so a run's
    ids are the ones that arrived in its own span; a user arriving twice in it is refused."""
    origin, span = deployment["origin_unix"], deployment["windows"] * deployment["window_s"]
    arrived = {r["request"] for r in _jsonl(ep / "advice_trace.jsonl")
               if r["kind"] == "arrive" and origin - SLACK_S <= r["t"] < origin + span}
    out: dict[str, list[int]] = {}
    for r in _jsonl(ep / "requests.jsonl"):
        if r["engine_id"] in arrived and r.get("user"):
            if r["user"] in out:
                raise ValueError(f"{r['user']} reached the endpoint twice in the run's link-windows")
            out[r["user"]] = list(r["streamed_ids"])
    return out


def changed_windows(run: Path) -> list[int] | None:
    """The link-windows a token run's changed ids fall in: each output its shim logged as changed (``server.json``'s
    ``tampered``, (session, index)), placed as `negatives.tie_run` places the certified egress's id events, each its
    stream's next output and the stream's user the certified I/O's. None when the shim logged none."""
    changed = (_read_json(run / "server.json") or {}).get("tampered")
    if not changed:
        return None
    plan, report = (json.loads((run / f).read_text()) for f in ("plan.json", "calibration.json"))
    setup = ar.Setup(report)
    e, i = setup.egress_params, setup.ingress_params
    ins = [ar.unpack_record(b, i.frame_bytes) for b in ar.read_blobs(run / "requests.records")]
    outs = [ar.unpack_record(b, e.frame_bytes) for b in ar.read_blobs(run / "responses.records")]
    users = {r["stream"]: r["user"] for r in NG.cs.certified_io(plan, setup, ins, outs)["records"]}
    at, nxt = {}, defaultdict(int)
    for rec in sorted(outs, key=lambda r: r.window):
        for s in (s for row in rec.grid for s in row if s is not None and NG._is_id_event(s)):
            q = NG.id_event_data(NG.decode_frame(s)[2])[0]
            at[users.get(q), nxt[q]] = rec.window
            nxt[q] += 1
    return sorted({at[u, k] for u, k in changed if (u, k) in at})


def tie(runs: dict[str, Path], ep: Path | None, ids: Path | None, why: str | None = None) -> dict:
    """The I/O tie's verdict on each run (NEG's `tie_run`: IO-TIE's reference in the clear, on
    `negatives.TIE_STAND_INS`), each run's sampled ids ``ids``' (`negatives.read_ids`, one map for every run) or else
    the endpoint's records' ``ep`` for its own link-windows (`served_ids`), and whether it is the verdict the run owes
    (`negatives.tie_ok`): honest's accepted, a negative's it reads refused by its part, and token's (`TIE_OWED`) at
    exactly the link-windows its changed ids fall in (`changed_windows`). Not run without IO-TIE in the tree, without the
    ids (``why``: why the endpoint's records aren't here), or when the honest run has no verdict."""
    if NG.io_tie_module() is None:
        return {"ran": False, "why": NG.TIE_MISSING["io-tie"]}
    if ids is None and ep is None:
        return {"ran": False, "why": f"no sampled ids ({why or 'neither --endpoint-run nor --ids given'})"}
    given, rows = None if ids is None else NG.read_ids(ids), {}
    for name, d in runs.items():
        row = {"verdict": NG.NOT_APPLICABLE, "ok": None, "codes": [], "failing": {}, "windows": [], "unplaced": None,
               "changed": None, "requests": None, "outputs": None, "placed": None, "held": None, "missing": {},
               "why": None}
        try:
            got = given if given is not None else served_ids(ep, json.loads((d / "deployment.json").read_text()))
            t = NG.tie_run(d, got)
            changed = changed_windows(d) if name in TIE_OWED else None
        except (OSError, KeyError, ValueError) as err:
            rows[name] = row | {"why": repr(err)}
            continue
        ok = NG.tie_ok(name, t)
        if name in TIE_OWED and ok:
            ok = changed is not None and sorted(t["failing"]) == changed
        rows[name] = row | {k: t.get(k) for k in ("verdict", "codes", "failing", "unplaced", "held", "missing",
                                                  "outputs", "placed")} | {
            "ok": ok, "windows": sorted(w for w in t["failing"] if w >= 0), "changed": changed, "requests": len(got),
            "why": "; ".join(t["missing"].values()) or None}
    honest = rows.get("honest") or {}
    ran = honest.get("verdict") in ("accepted", "refused")
    return {"ran": ran, "why": None if ran else f"no verdict on the honest run: {honest.get('why') or 'no honest run'}",
            "from": "the endpoint's records (each run's requests by their arrival in its link-windows)" if given is None
            else str(ids), "stand_ins": NG.TIE_STAND_INS, "runs": rows}


# --- what's still a stand-in ------------------------------------------------------------------------------------------

def stubs(r: dict, varied_client: bool) -> list[str]:
    """What the report stood on that isn't yet the chain's own, from what ``r`` (the report) says ran."""
    out, c = [], r.get("chain") or {}
    if not varied_client:
        out.append("Varied traffic: certifier_serve.py's `rehearse --traffic varied` (CERT-VARLEN, Poisson arrivals, "
                   "prompts as ids, real EOS) isn't in this tree.")
    if r.get("profile") == ADVICE:
        out.append("N2 on the advice profile: N2-ADVICE (NetworkAdvisedCounts_v1) hasn't landed, so no link-window has "
                   "N2's verdict and nothing is proved by n2_zk; the rehearsal's own check (certified I/O, audit, salt) "
                   "is the verdict.")
    a, b = c.get("advice"), c.get("bits") or {}
    served = "served" in b
    if a is None and not served:
        out.append("Advice: none given (--advice, or --endpoint-run for the endpoint's own); a served run's comes from "
                   "its step trace (ADVICE-SRC).")
    elif not b.get("bound"):
        out.append(f"Advice: {a['file'] if a else 'the endpoint’s'} is not these runs' ({b.get('bound_why')})"
                   + ("." if served else "; a served run's comes from the committing endpoint's step trace "
                                         "(ADVICE-SRC)."))
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
                   "registration is public; the compile unit and N2-ADVICE, which take it hidden, haven't landed.")
    out.append("Coded bits: ENC-CODE's encoding hasn't landed.")
    p = c.get("prove_path")
    if p is None:
        out.append("Prove path: not given (--prove-path, a prove_path.py run's output).")
    elif served:
        out.append(f"Prove path: adopted from {p['dir']}, {p['bound_why']}: the endpoint's Commit windows have no sealed "
                   "replay bundle a prover can open, so its step units aren't proved here.")
    else:
        sources = sorted({x["source"] for arm in p["arms"].values() for x in arm["ranks"].values() if x["source"]})
        out.append(f"Prove path: adopted from {p['dir']}, a run of its own (sources {', '.join(sources) or '—'}); it "
                   "proves a TP Commit's step units compiled from that run's advice, not a commit of these runs: the "
                   "committing endpoint's Commit and ADVICE-SRC's advice are what make it these runs'.")
    n = c.get("negatives") or {}
    if not n.get("ran"):
        out.append(f"Negatives (NEG's registry): {n.get('why') or 'not run'}.")
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
            out.append("I/O tie: IO-TIE's reference in the clear on a fake upstream's cells and Commit port of the advice "
                       "and the given ids, not the warden's cells under its root (WARDEN) and vllm's Commit proved in an "
                       "IoTieWindow unit.")
    t = c.get("tie")
    if t is not None and not t["ran"]:
        out.append(f"I/O tie: not run ({t['why']}), so token waits on it.")
    elif t is not None:
        out.append(f"I/O tie: IO-TIE's reference in the clear on NEG's stand-ins (each window's certified frames held to "
                   f"their head and committed under fake salts in place of the warden's cells, the sampled ids from "
                   f"{t['from']} under fake salts in place of the Commit's port), not vllm's Commit and the warden's "
                   "cells in an IoTieWindow unit.")
    where = str(r.get("where") or "")
    if where.startswith("off the prover's node"):
        out.append("Auditor host: the certifier ran on another node than the endpoint's, through an ssh forward, not on "
                   "the auditor host (#3433 brings the gate's runs: --from RUN.tgz).")
    elif not where.startswith("gate: "):
        out.append("Auditor host: the certifier ran off the gate (#3433 brings the gate's runs: --from RUN.tgz).")
    return out


# --- the report's sections --------------------------------------------------------------------------------------------

def _cell(x) -> str:
    return "—" if x is None else str(x)


def markdown(c: dict) -> list[str]:
    """The chain's sections of the report's Markdown."""
    lines = []
    a, b = c.get("advice"), c.get("bits")
    served = b is not None and "served" in b
    if served:
        lines += ["", "## Advice", "",
                  (f"The endpoint's own ({b['served']}): `verity-vllm advice` over its records, one I1 v0 advice a "
                   f"closed Commit window, at {a['file']}. The run's Commit windows (those opening in its link-windows): "
                   f"{a['windows']}, with {a['requests']} requests, {a['steps']} steps, {a['outputs']} kept outputs, "
                   f"{a['bits']} bits in v0." if a else "The endpoint's own: none.")
                  + f" Bound to these runs: {_cell(b['bound'])} ({b['bound_why'] or 'every window opened to its registered root under its Commit, every advice request a certified one, each in one window'}).",
                  "", "| Commit window | Commit run id | opens at s | link-window | requests | steps | v0 bits | "
                  "derived bits | derived, shaped | forced | opened | bound | rebuilt |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        lines += [f"| {x['window']} | refused: {x['refused']} |" + " — |" * 11 if "refused" in x else
                  f"| {x['window']} | {_cell((x['commit'] or {}).get('run_id'))} | {x['t0_s']} | {x['link_window']} | "
                  f"{x['requests']} | {x['steps']} | {x['v0_bits']} | {_cell(x['derived_bits'])} | "
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
                  "ENC-DERIVED's residual at its least widths, charged by I1's rule, with the release sent (unshaped) "
                  "and derived (shaped); forced: the residual's fields only witness fixes. "
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
        lines += ["", "## Prove path" + (" (from the fixed workload, not this run)" if served else ""), "",
                  f"`prove_path.py audit`'s output adopted from {p['dir']}: each rank's step units compiled from the "
                  "advice by the registered compiler, drawn on the verifier's own coins, proved --zk behind the Lean "
                  "firewall and verified by Lean. The advice each arm committed is the --advice file: "
                  f"{_cell(p['committed_is_the_advice_file'])}. Bound to these runs: {_cell(p['bound'])}"
                  + ("" if p["bound"] else f" ({p['bound_why']}), so its ok is not the report's")
                  + "".join(f"; {n} held by {a['advice_held_by']}" for n, a in p["arms"].items() if a.get("advice_held_by"))
                  + "." + "".join(f" {n}'s gap: {a['gap']}." for n, a in p["arms"].items() if a.get("gap")), "",
                  "| arm | ok | rank | audit accepted | units | drawn | wrong units at most | negatives refused | "
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
        lines += ["", "## Negatives (NEG's registry)", ""]
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
            lines.append(f"Not run: {t['why']}.")
        else:
            lines += [f"NEG's `tie_run` on each run: IO-TIE's reference in the clear, the sampled ids from {t['from']}. "
                      "Owed: honest accepted, token refused by content at exactly the link-windows its changed ids fall "
                      "in, drop refused by content; late and extra owe nothing here. Window -1: committed outputs no "
                      "certified cell carries.", "",
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
