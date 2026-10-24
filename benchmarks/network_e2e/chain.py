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
* **The prove path** (`prove_path`, ``--prove-path DIR|TGZ``): `prove_path.py audit`'s output adopted (`run.sh`'s ``$OUT``,
  each arm a directory with its ``summary.json``): an honest arm passes when every rank's audit accepted on the verifier's
  coins and every negative was refused; a forged arm when Lean refused the forged unit it drew. The advice the arm
  committed is held to ``--advice``'s: the stand-in's codec committed the file's bytes (its SHA-512), v0's commitment is
  hiding, so there only its ``total_bits``. That holds it to the file, not to these runs: the prove path is bound to them
  only when the file is their advice and the Commit it proves is theirs, and only a bound prove path counts in the OK.
* **The negatives** (`negatives`): NEG's `suite` on the runs, handed the advice's v0 bytes for its advice negatives.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "benchmarks" / "network_traces"))

import active_replay as ar  # noqa: E402
import negatives as NG  # noqa: E402
from verity.core.protocols.certifier import advice as I1  # noqa: E402


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


# --- the prove path ---------------------------------------------------------------------------------------------------

def _refused(v) -> bool:
    return v if isinstance(v, bool) else bool((v or {}).get("refused"))


def _same(committed: dict | None, info: dict | None) -> bool | None:
    if not committed or info is None:
        return None
    if "digest" in committed:
        return committed["digest"] in (info["file_sha512"], info["v0_sha512"])
    return committed.get("bits") == info["bits"] if "bits" in committed else None


def prove_path(src: Path, info: dict | None, advice_bound: bool | None = None) -> dict:
    """`prove_path.py audit`'s output at ``src``, one row per arm and rank (the module's docstring). Each arm's
    committed advice is held to the ``--advice`` file (``info``), which says nothing of these runs: the prove path is
    ``bound`` to them only when that file is these runs' advice (``advice_bound``, `bits`' ``bound``) and the Commit it
    proves is theirs, which nothing here checks yet (no Commit record of these runs is an input), so it is never bound
    and its ``ok`` stays out of the report's."""
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
                        "seconds": x.get("seconds"), "source": x.get("source")}
        committed = s.get("advice") or {}
        arms[p.name] = {"ok": s.get("ok") is True and bool(ranks), "forged": bool(s.get("forged")),
                        "world": s.get("world"), "rule": s.get("rule"), "delta": s.get("delta"),
                        "compiler": (s.get("compiler") or {}).get("name"), "tp_run_root": s.get("tp_run_root"),
                        "advice": s.get("advice"), "committed_is_the_advice_file": _same(committed, info),
                        "advice_held_by": "the file's SHA-512" if "digest" in committed else
                        "its bits only: the committed root hides it, and its opening is the prover's" if "root" in committed
                        else None, "gap": s.get("gap"), "ranks": ranks}
    if not arms:
        raise SystemExit(f"{src}: no prove-path arm (a directory with prove_path.py audit's summary.json)")
    given = [a["committed_is_the_advice_file"] for a in arms.values()]
    committed = None if None in given else all(given)
    why = ("its arms' committed advice isn't the --advice file" if committed is not True else
           "the --advice file isn't these runs' advice" if advice_bound is not True else
           "the Commit it proves isn't held to these runs' (no Commit record of these runs is an input)")
    return {"dir": str(src), "arms": arms, "committed_is_the_advice_file": committed, "bound": False,
            "bound_why": why, "ok": all(a["ok"] for a in arms.values()) and any(not a["forged"] for a in arms.values())
            and False not in given}


# --- the negatives ----------------------------------------------------------------------------------------------------

def negatives(runs: dict[str, Path], work: Path, adv, clock) -> dict:
    """NEG's `suite` on ``runs`` (by strategy), its advice negatives made from ``adv``'s v0 bytes. ``ok`` unless a row
    that ran says otherwise: a pending check's row (``ok`` None) or one that didn't run is the report's to list, not a
    failure."""
    path = None
    if adv is not None:
        work.mkdir(parents=True, exist_ok=True)
        path = work / "advice.bin"
        path.write_bytes(I1.encode(adv))
    rows = NG.suite(runs, work / "negatives", advice=path, clock=clock)
    return {"ran": True, "format": NG.FORMAT, "rows": rows,
            "ok": all(r.get("ok") is not False for r in rows.values() if r.get("ran"))}


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
    if a is None:
        out.append("Advice: none given (--advice); a served run's comes from its step trace (ADVICE-SRC).")
    elif not b.get("bound"):
        out.append(f"Advice: {a['file']} is not these runs' ({b.get('bound_why')}); a served run's comes from the "
                   "committing endpoint's step trace (ADVICE-SRC).")
    if a is not None and "upgraded" in a["form"]:
        out.append("Advice form: the stand-in's pre-v0 JSON, upgraded to v0 here; the bits are v0's of the upgrade, while "
                   "the prove path committed the file's bytes.")
    if a is not None and b.get("config") is None:
        out.append("Derived bits: no serving config given (--serving-config); the endpoint registers none yet.")
    elif a is not None:
        out.append("Derived bits: the serving config is the one given (--serving-config), not one the endpoint "
                   "registered.")
    out.append("Coded bits: ENC-CODE's encoding hasn't landed.")
    p = c.get("prove_path")
    if p is None:
        out.append("Prove path: not given (--prove-path, a prove_path.py run's output).")
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
    if a is not None:
        lines += ["", "## Advice", "",
                  f"From {a['file']} ({a['form']}): {a['requests']} requests, {a['steps']} steps, {a['outputs']} kept "
                  f"outputs, {a['bits']} bits in v0. SHA-512 of the file {a['file_sha512'][:16]}…, of its v0 encoding "
                  f"{a['v0_sha512'][:16]}…. Bound to these runs: {_cell(b['bound'])} ({b['bound_why'] or 'by the certified I/O'})."]
    if b is not None:
        cfg = b["config"]
        lines += ["", "## Advice bits per link-window", "",
                  f"T = {b['T']} buckets a window. v0: I1's `bits_by_window` ({b['v0_total']} bits in all). Derived: "
                  "ENC-DERIVED's residual at its least widths, charged by I1's rule, with the release sent (unshaped) "
                  "and derived (shaped); forced: the residual's fields only witness fixes. Serving config: "
                  + (f"{json.dumps(cfg, sort_keys=True)}, the one the job was given (--serving-config), not one the "
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
        lines += ["", "## Prove path", "",
                  f"`prove_path.py audit`'s output adopted from {p['dir']}: each rank's step units compiled from the "
                  "advice by the registered compiler, drawn on the verifier's own coins, proved --zk behind the Lean "
                  "firewall and verified by Lean. The advice each arm committed is the --advice file: "
                  f"{_cell(p['committed_is_the_advice_file'])}. Bound to these runs: {_cell(p['bound'])}"
                  + ("" if p["bound"] else f" ({p['bound_why']}), so its ok is not the report's")
                  + "".join(f"; {n} held by {a['advice_held_by']}" for n, a in p["arms"].items() if a.get("advice_held_by"))
                  + "." + "".join(f" {n}'s gap: {a['gap']}." for n, a in p["arms"].items() if a.get("gap")), "",
                  "| arm | ok | rank | audit accepted | units | drawn | wrong units at most | negatives refused | "
                  "forged unit refused | s | source |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, arm in p["arms"].items():
            lines += [f"| {name} | {arm['ok']} | {r} | {_cell(x['accepted'])} | {_cell(x['population'])} | {x['drawn']} | "
                      f"{_cell(x['wrong_units_bound'])} | {x['negatives_refused']} | {_cell(x['forged_refused'])} | "
                      f"{_cell(x['seconds'])} | {_cell(x['source'])} |" for r, x in arm["ranks"].items()]
    n = c.get("negatives")
    if n is not None:
        lines += ["", "## Negatives (NEG's registry)", ""]
        if not n.get("ran"):
            lines.append(f"Not run: {n.get('why')}.")
        else:
            lines += ["Each tampering, the check that must refuse it and its code; ok: refused by that code (a pending "
                      "check's row, ok —, says what it waits on).", "",
                      "| negative | family | tampers | code | ran | N2 accepted | refused windows | codes | ok | waits on, "
                      "or why not run |", "|---|---|---|---|---|---|---|---|---|---|"]
            for name, x in n["rows"].items():
                lines.append(f"| {name} | {x['family']} | {x['tampers']} | `{x['code']}` | {x.get('ran')} | "
                             f"{_cell(x.get('n2_accepted'))} | {_cell(x.get('refused_windows'))} | "
                             f"{_cell(', '.join(x['codes']) if x.get('codes') else None)} | {_cell(x.get('ok'))} | "
                             f"{_cell(x.get('why') or ', '.join(x['waits']) or None)} |")
    return lines
