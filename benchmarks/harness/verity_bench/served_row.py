"""What a served run's audit showed, and what stands between it and fully proved: the section `verity bench
exfiltration --served-run` prints from the run's record (`served_record`), read against the tree.

The record holds what the run's files say: the sessions' verdicts counted by kind, the driver's checks, the stream's
outcome and its codes, the layout the serve committed. This module decides nothing about the run; it reads from the
tree who decided each of those and what proves it, and prints (`account`, `render`):

* **The verdict line**, in the verifier's own words (`verdict_line`): how many of the sampled proofs the Lean
  verifier accepted, then the final accept as the stream's outcome gave it: given, refused with its codes, or
  withheld, where every check of the second stage held and the outcome is refused by one code the table names a
  reason for (`TABLE`'s ``withheld``). An accept the verifier withheld or refused is never printed as given.
* **What the audit covered** (`covered`): the replay block as the layout names it: the stand-in, instances of the
  check's template, or a Definition of the class's own.
* **To fully proved** (`open_items`), from the run's own files and the lock entries of its own verdicts: the replay
  block where it is the stand-in; every verdict a verifier that is not Lean decided (`TABLE`'s ``decided_by``);
  every Lean verdict whose lock entry (``proved_by``, in ``security/lean-audit.json``) is missing or lists open
  premises; and the final accept where it was not given. The resampling holes and the cited theorem are not listed
  here: `spot_status` prints them, in the same command.
* **The mark**: ``proved`` only where that list is empty, the final accept is given, the cited theorem's lock entry
  lists no open premise and each resampling hole is closed at the run's commit (`spot_status`'s own readings, so the
  two cannot disagree); `exfiltration.RUN_NOT_PROVED` otherwise. A kind of verdict is ``proved`` only where a Lean
  verifier decided it and its lock entry lists no open premise.
* **The prover's seconds** as the record holds them, said as ``at most`` where they are a session's, wait included.
* **The security level** at the run's own sizes (`security`): the planner's binding bound
  (`headline.binding_bound`) at one verification phase of the run's sessions and sampled proof blocks, modeled, with
  the two inputs it assumes.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from verity_bench import exfiltration as E
from verity_bench import jobs, served_record, spot_status

SCHEMA = "served-run/v1"
#: which verifier decides each kind of verdict, the lock entry that covers a Lean one, and why an accept is withheld:
#: data, read here and never typed into the print
TABLE = Path(__file__).resolve().parent / "data" / "served_row.json"
LEAN, PROVED = "lean", "proved"
#: the layout's fields that make a replay block the stand-in: instances of one check template, so many a prompt
#: token and so many an output
STAND_IN = ("check", "per_prompt", "per_output")
#: the kind of verdict that is the Lean verifier's on a session's proof
PROOFS = "lean-verify"
#: the piece of the stop class's verdicts of record this audit's verdict is (`exfiltration.PIECES`), and the word for
#: an accept the verifier withheld; while the command's words hold none, such a verdict's record reads `refuse`
PIECE, WITHHELD = "second_stage", "withheld"


def table(path: Path | None = None) -> dict:
    """`TABLE`, refused where a kind names no verifier or no decider."""

    path = path or TABLE
    got = json.loads(path.read_text())
    for section in ("verdicts", "checks"):
        for kind, entry in got[section].items():
            if lacks := [k for k in ("what", "verifier", "decided_by") if not entry.get(k)]:
                raise ValueError(f"{path}: `{section}.{kind}` lacks {', '.join(lacks)}")
    return got


def proved_by(lock: Mapping, name: str | None) -> dict:
    """A lock entry as a verdict's proof: its name, whether the lock lists it, and its open premises. `proved` only
    where it is listed and has none."""

    entry = (lock.get("guarantees") or {}).get(name) if name else None
    premises = None if entry is None else list(entry["premises"]["open"])
    return {"name": name, "listed": entry is not None, "open_premises": premises, PROVED: premises == []}


def _proof_said(proof: Mapping) -> str:
    if proof["name"] is None:
        return "the table names no lock entry for it"
    if not proof["listed"]:
        return f"`{proof['name']}` is not in the lock"
    if proof[PROVED]:
        return f"proved by `{proof['name']}`, which takes no open premise"
    n = len(proof["open_premises"])
    return f"its lock entry `{proof['name']}` takes {n} open premise{'s' * (n != 1)}"


def verdicts(counts: Mapping, kinds: Mapping, lock: Mapping) -> list[dict]:
    """The sessions' verdicts by kind: how many are true of how many, the verifier that decided the kind, and whether
    its lock entry proves it."""

    out = []
    for kind, count in counts.items():
        entry = kinds["verdicts"].get(kind) or {"what": "a kind the table does not name", "verifier": "not known",
                                                 "decided_by": "not known", "proved_by": None}
        proof = proved_by(lock, entry.get("proved_by"))
        out.append({"kind": kind, **{k: entry[k] for k in ("what", "verifier", "decided_by")}, **count,
                    "proved_by": proof, PROVED: entry["decided_by"] == LEAN and proof[PROVED]})
    return out


def checks(counts: Mapping, kinds: Mapping) -> list[dict]:
    """The driver's own checks around the sessions, each with who decided it."""

    unnamed = {"what": "a check the table does not name", "verifier": "not known", "decided_by": "not known"}
    return [{"check": name, **(kinds["checks"].get(name) or unnamed), **count} for name, count in counts.items()]


def covered(layout: Mapping | None, interior: Mapping | None) -> dict:
    """What the replay block is, by the layout the serve committed: the stand-in, instances of the check's template
    (the layout's `STAND_IN` fields), or a Definition of the class's own; with what the interior registered as the
    replay's own value. `stand_in` is None where no layout is given."""

    if layout is None:
        return {"stand_in": None, "said": "not known: no layout is given (the first stage's `stage1.json`, or the "
                                          "serve's `commit.json`)"}
    interior = interior or {}
    out = {**layout, "registered": list(interior.get("registered") or []), "unlinked": interior.get("unlinked"),
           "stand_in": all(k in layout for k in STAND_IN)}
    if not out["stand_in"]:
        out["said"] = (f"the replay block is `{layout.get('replay')}`, a Definition of the class's own: the layout holds "
                       f"none of the stand-in's fields ({', '.join(f'`{k}`' for k in STAND_IN)})")
        return out
    counts = [f"{n} for each {what}" for n, what in ((layout["per_output"], "output token"),
                                                    (layout["per_prompt"], "prompt token")) if n]
    reads = (f"what it reads is the replay's own value, {E._names(out['registered'])}, registered after the first "
             "stage's sample" if out["registered"] else
             "the second stage ran unlinked, so no value it reads is held to the first stage's record" if out["unlinked"]
             else "the interior registers no value of the replay's own")
    out["said"] = (f"the replay block is the stand-in, `{layout.get('replay')}`: instances of the check template "
                   f"`{layout['check']}`, {E._listed(counts) or 'none'}{', each one in place' if layout.get('inline') else ''}; "
                   f"{reads}. It checks the choice of each token from a row of scores"
                   + (" and the stop rule the layout registers" if layout.get("stop") else ", under no stop rule")
                   + ", and nothing of the model's layers that produced the scores")
    return out


def outcome(shown: Mapping, kinds: Mapping, theorem: Mapping) -> dict:
    """The final accept as the stream's outcome gave it: ``given``; ``withheld``, where every check of the second
    stage held (``held``) and the outcome is refused by one code the table names a reason for; ``refused``, with
    its codes; or ``none``, where the second stage reached no outcome. Its proof is the cited theorem's."""

    o, entry = shown.get("outcome"), kinds["outcome"]
    proof = {"name": theorem.get("name"), "listed": theorem.get("name") is not None,
             "open_premises": list(theorem["open_premises"]) if theorem.get("name") else None,
             PROVED: theorem.get("name") is not None and not theorem["open_premises"]}
    out = {**{k: entry[k] for k in ("what", "verifier", "decided_by")}, "proved_by": proof, "held": shown.get("held"),
           "codes": [], "withheld": None, "why": None, PROVED: False}
    if not isinstance(o, Mapping):
        refused = shown.get("refused")
        return out | {"final": "none",
                      "why": "the second stage reached no outcome" + (f": {json.dumps(refused)}" if refused else "")}
    out["codes"] = list(o.get("codes") or [])
    if o.get("accepted") is True:
        return out | {"final": "given", PROVED: entry["decided_by"] == LEAN and proof[PROVED]}
    reasons = kinds.get("withheld") or {}
    if shown.get("held") is True and len(out["codes"]) == 1 and out["codes"][0] in reasons:
        return out | {"final": "withheld", "withheld": {"code": out["codes"][0], **reasons[out["codes"][0]]}}
    return out | {"final": "refused", "why": json.dumps(o["refused"]) if o.get("refused") and not out["codes"] else None}


def verdict_line(kinds: Sequence[Mapping], final: Mapping) -> str:
    """The verdict in the verifier's own words: the sampled proofs the Lean verifier accepted, then the final accept
    as the outcome gave it. No accept the verifier withheld or refused is said to be given."""

    proofs = next((k for k in kinds if k["kind"] == PROOFS), None)
    if proofs is None or not proofs["of"]:
        said = "no sampled proof was put to the Lean verifier"
    elif proofs["accepted"] == proofs["of"]:
        said = f"every sampled proof accepted by the Lean verifier ({proofs['of']} of {proofs['of']} sessions)"
    else:
        said = f"{proofs['accepted']} of {proofs['of']} sampled proofs accepted by the Lean verifier"
    if final["final"] == "given":
        return f"{said}; the final accept given by `{final['verifier']}`"
    if final["final"] == "withheld":
        return f"{said}; the final accept withheld, reason `{final['withheld']['code']}` ({final['withheld']['why']})"
    if final["final"] == "refused":
        return (f"{said}; the final accept refused" + (f", {E._names(final['codes'])}" if final["codes"] else "")
                + (f": {final['why']}" if final["why"] else ""))
    return f"{said}; no final accept: {final['why']}"


def open_items(cover: Mapping, kinds: Sequence[Mapping], audit_checks: Sequence[Mapping], final: Mapping) -> list[dict]:
    """What stands between the run and fully proved, from the run's own files and its verdicts' lock entries."""

    items = []
    if cover["stand_in"] is None:
        items.append({"item": "replay block", "said": f"the replay block is {cover['said']}"})
    elif cover["stand_in"]:
        items.append({"item": "replay block", "said": f"the replay block is the stand-in (`{cover['check']}`), not a "
                                                      "Definition of the request: the model's layers are not covered"})
    other = [f"`{k['kind']}` ({k['verifier']})" for k in kinds if k["decided_by"] != LEAN]
    other += [f"`{c['check']}` ({c['verifier']})" for c in audit_checks if c["decided_by"] != LEAN]
    if final["decided_by"] != LEAN:
        other.append(f"the final accept ({final['verifier']})")
    if other:
        items.append({"item": "not Lean", "said": "decided by a verifier that is not Lean: " + ", ".join(other)})
    items += [{"item": "lock entry", "kind": k["kind"], "said": f"`{k['kind']}` ({k['verifier']}): {_proof_said(k['proved_by'])}"}
              for k in kinds if k["decided_by"] == LEAN and not k[PROVED]]
    if final["final"] == "withheld":
        items.append({"item": "final accept", "said": f"the final accept is withheld by `{final['verifier']}`: "
                                                      f"`{final['withheld']['code']}`, {final['withheld']['why']}"})
    elif final["final"] != "given":
        items.append({"item": "final accept",
                      "said": "the final accept is " + ("refused" if final["final"] == "refused" else "not reached")})
    return items


def verdict_record(doc: Mapping) -> dict | None:
    """The audit's verdict as a record of the stop class's verdicts of record (`exfiltration.VERDICT_FIELDS`, the
    piece `PIECE`): the final accept as the outcome gave it, the verifier that gave it at the run's commit, the
    cited theorem only where the whole run is `proved`, and otherwise what stands between it and proved, in plain
    words. None where the second stage reached no outcome: it holds no verdict."""

    final, reads = doc["outcome"], doc["mark_reads"]
    if final["final"] == "none":
        return None
    proved = doc["verdict"]["mark"] == PROVED
    if final["final"] == "withheld":
        word, waits = (WITHHELD if WITHHELD in E.VERDICT_READS else "refuse"), final["withheld"]["why"]
    elif final["final"] == "refused":
        word, waits = "refuse", "the verifier refused it" + (f": {', '.join(final['codes'])}" if final["codes"] else "")
    else:
        n = reads["open_premises"]
        opened = [h for h, status in reads["holes"].items() if status != spot_status.CLOSED]
        word, waits = "accept", None if proved else "; ".join(filter(None, (
            "no Lean theorem is cited for its verifier" if reads["theorem"] is None else
            n and f"{reads['theorem']} takes {n} open premise{'s' * (n != 1)}",
            opened and f"not closed at the run's commit: {', '.join(opened)}",
            *(item["said"] for item in doc["to_fully_proved"]))))
    return {"piece": PIECE, "control": None, "run": doc["run"],
            "verifier": {"module": final["verifier"], "commit": doc["commit"]}, "verdict": word,
            "proved_by": reads["theorem"] if proved else None, "open": waits}


def prover(record: Mapping, notes: Sequence[Mapping] = ()) -> dict:
    """The sessions' proving as the record holds it: which seconds (`served_record.SOURCE`), their sum, the GPUs, and
    the cards the provers ran on, each with what the table notes of that card of the second stage's machine."""

    shown, entries = record["audit"], record.get("prove") or []
    machine = ((record.get("runs") or {}).get("second") or {}).get("machine")
    cards = {card: {**held, "note": next((n["note"] for n in notes if (n["machine"], str(n["card"])) == (machine, card)),
                                         None)} for card, held in sorted((shown.get("cards") or {}).items())}
    source = record.get(served_record.SOURCE)
    seconds = [e["seconds"] for e in entries]
    gpus = [e["gpus"] for e in entries]
    return {"sessions": shown["sessions"], "source": source, "at_most": source != "busy",
            "seconds": None if not seconds or None in seconds else float(sum(seconds)),
            "gpu_s": None if not entries or None in seconds or None in gpus else
            float(sum(s * g for s, g in zip(seconds, gpus))),
            "gpus": None if not gpus or None in gpus else gpus, "cores": shown.get("cores"), "machine": machine,
            "cards": cards, "as_held": shown["seconds"], "sizes": shown["sizes"]}


def security(shown: Mapping, root: Path) -> dict | None:
    """The binding bound at the run's own sizes, the planner's (`headline.binding_bound`): one verification phase of
    the run's sessions and its sampled proof blocks, each session one table. None where the second stage sampled none."""

    sessions, blocks = shown["sessions"], shown["proof_blocks"]
    if not sessions or not blocks:
        return None
    headline = E._load(root, "spot_figures/breakeven", "headline")
    planner = SimpleNamespace(binding=E._load(root, "spot_figures", "binding"))
    return {"sessions": sessions, "sampled_proof_blocks": blocks,
            **headline.binding_bound(planner, {"tables": 1, "draws": blocks, "replay_sample": 1}, 1, sessions, 1)}


def account(record: Mapping[str, Any], root: Path | None = None, *, kinds: Mapping | None = None,
            holes: Mapping[str, str] | None = None) -> dict | None:
    """The section's document from a served run's ``record`` (`served_record.read`), read against the tree at
    ``root``: its lock, the table (``kinds``, default `TABLE`) and each resampling hole's status at the run's commit
    (``holes``, default `spot_status.holes`). None where the record holds no ``audit``."""

    shown = record.get("audit")
    if not isinstance(shown, Mapping):
        return None
    root = root or jobs.repository_root()
    kinds = table() if kinds is None else kinds
    lock = json.loads((root / spot_status.LOCK).read_text())
    theorem = spot_status.theorem(root)
    if holes is None:
        commit = record.get("commit")
        holes = ({h: spot_status.UNKNOWN for h in spot_status.HOLES} if commit is None else
                 {h: s["status"] for h, s in spot_status.holes(commit, root).items()})
    by_kind, audit_checks = verdicts(shown["verdicts"], kinds, lock), checks(shown["checks"], kinds)
    cover, final = covered(shown.get("layout"), shown.get("interior")), outcome(shown, kinds, theorem)
    items = open_items(cover, by_kind, audit_checks, final)
    proved = (not items and final["final"] == "given" and final[PROVED]
              and all(holes.get(h) == spot_status.CLOSED for h in spot_status.HOLES))
    doc = {"schema": SCHEMA, "run": record.get("run"), "commit": record.get("commit"), "runs": record.get("runs"),
           "ties": shown.get("ties") or [], "first_stage": shown.get("first_stage"), "covered": cover,
           "verdicts": by_kind, "checks": audit_checks, "outcome": final,
           "verdict": {"line": verdict_line(by_kind, final), "mark": PROVED if proved else E.RUN_NOT_PROVED},
           "to_fully_proved": items,
           "mark_reads": {"theorem": theorem.get("name"), "open_premises": len(theorem["open_premises"]),
                          "holes": dict(holes)},
           "prover": prover(record, kinds.get("cards") or ()), "security": security(shown, root),
           "one_stage_record": shown.get("one_stage_record")}
    return doc | {"verdict_record": verdict_record(doc)}


# ---- the print ---------------------------------------------------------------------------------------------------------


def _of(count: Mapping) -> str:
    return f"{count['accepted']} of {count['of']}" + (f" ({E._names(count['codes'])})" if count.get("codes") else "")


def _s(x: float | None) -> str:
    return "not written" if x is None else f"{x:,.1f} s"


def _prover_said(p: Mapping) -> list[str]:
    which = {"busy": "by the proof server's own clock on each session's prover (`busy_s`)",
             "session": "by the audit's clock on each session (`session_s`), which counts a session's wait behind the "
                        "others of its prover",
             None: "by the audit's clock on each session (`session_s`), which counts a session's wait behind the others "
                   f"of its prover; no session says which seconds are its proving's (`{served_record.SOURCE}`)"}[p["source"]]
    if p["gpus"] is None:
        where = "The files hold no count of GPUs for a session"
    elif not any(p["gpus"]):
        where = ("No session held a GPU: the sessions proved on the host's CPUs"
                 + (f", {p['cores']} cores" if p["cores"] else "") + ", so the proving holds no GPU-seconds")
    else:
        where = (f"{sum(1 for g in p['gpus'] if g)} of {len(p['gpus'])} sessions held a GPU, {_s(p['gpu_s'])[:-2]} "
                 "GPU-seconds in all")
    held = p["as_held"]
    by_card = [f"- card {card}" + (f" of {p['machine']}" if p["machine"] else "") + ": "
               + ("a session names it" if c["sessions"] is None else f"{c['sessions']} sessions, {_s(c['seconds'])} busy")
               + (f". **{c['note'][0].upper()}{c['note'][1:]}.**" if c["note"] else ".") for card, c in p["cards"].items()]
    return [f"- the sessions' proving: {'at most ' if p['at_most'] else ''}{_s(p['seconds'])} over {p['sessions']} "
            f"sessions, {which}. {where}.",
            f"- as the files hold them, summed over the sessions: `session_s` {_s(held['session_s'])}; `busy_s` "
            f"{_s(held['busy_s'])}; the prover's own timer over its tables (`timing.prove_total_s`) "
            f"{_s(held['prove_total_s'])}; the Lean verifier (`lean_verify_s`) {_s(held['lean_verify_s'])}"
            + (f", on {p['cores']} cores" if p["cores"] else "") + ".", *by_card]


def _security_said(s: Mapping | None) -> list[str]:
    if s is None:
        return [f"- the {E.SECURITY} at the run's own sizes: not computed, the second stage sampled no proof block."]
    calls, compressions = (s["counted"][c] for c in ("calls", "compressions"))
    return [
        f"- the {E.SECURITY} at the run's own sizes ({s['status']}): the probability that an accepted run opens a value "
        f"off the committed trace, by the bound Lean states of a two-stage sample counted by classes of gates "
        f"(`{E.LEAN_BOUND}`), every run at {s['design']['calls_a_run']} calls of SHA-512, over this run's one "
        f"verification phase: **at most {calls['phase']['bound']}** counted in calls, as the Lean counts, and "
        f"{compressions['phase']['bound']} in compressions. Read from the run: {s['sessions']} sessions and "
        f"{s['sampled_proof_blocks']} sampled proof blocks.",
        f"- the {E.SECURITY} assumes: "
        + "; ".join(i["input"] if i["value"] is None else f"{E._n(i['value'])} {i['input']}" for i in s["assumed"])
        + "; one table a session, the files holding no count of a session's tables. It counts each session's knowledge "
        "error and the run's own opening checks as 0.",
        f"- what Lean proves of the {E.SECURITY}: counted in calls, {E._lean(calls['lean'])}. Counted in compressions, "
        f"{E._lean(compressions['lean'])}."]


def render(doc: Mapping[str, Any]) -> str:
    """The section as Markdown: the verdict, what the audit covered and what is open, then the verdicts by kind, the
    runs, the prover and the security level."""

    runs, final = doc["runs"] or {}, doc["outcome"]
    lines = [f"## What the audit showed on {doc['run'] or 'the served run'}, and what stands between it and fully proved",
             "", f"- **verdict: {doc['verdict']['mark']}.** {doc['verdict']['line'][0].upper()}{doc['verdict']['line'][1:]}.",
             f"- **what the audit covered**: {doc['covered']['said']}.",
             "- **to fully proved**:" + ("" if doc["to_fully_proved"] else
                                        " nothing in the run's own files or in its verdicts' lock entries.")]
    lines += [f"  - {item['said']}" for item in doc["to_fully_proved"]]
    body = [[f"`{k['kind']}`", k["what"], _of(k), f"{k['decided_by']}: {k['verifier']}",
             _proof_said(k["proved_by"]) if k["decided_by"] == LEAN else "none: not a Lean verifier's"]
            for k in doc["verdicts"]]
    body += [[f"`{c['check']}`", c["what"], _of(c), f"{c['decided_by']}: {c['verifier']}", "none: not a Lean verifier's"]
             for c in doc["checks"]]
    body.append(["the final accept", final["what"],
                 {"given": "given", "withheld": f"withheld (`{(final['withheld'] or {}).get('code')}`)",
                  "refused": "refused" + (f" ({E._names(final['codes'])})" if final["codes"] else ""),
                  "none": "no outcome"}[final["final"]],
                 f"{final['decided_by']}: {final['verifier']}", "—"])
    lines += ["", *E._table(["verdict", "what", "true", "decided by", "what proves its verifier"], body), ""]
    for stage, what in (("serve", "the serve"), ("first", "the first stage"), ("replay", "the replay of the sampled spans"),
                        ("second", "the second stage")):
        r = runs.get(stage)
        lines.append(f"- {what}: " + ("not given" if r is None else
                     f"{r['run'] or 'no run id'} at " + (f"`{r['commit'][:12]}`" if r["commit"] else f"no commit ({r['why']})")
                     + (" (a dirty tree)" if r["dirty"] else "")))
    if (first := doc["first_stage"]) is not None:
        said = f"- the first stage sampled {len(first['sampled'] or [])} of {first['of']} spans at `{first['rule']}`"
        if (check := first["check"]) is not None:
            error = check["error"] or check["session_error"]
            said += "; its check: " + (f"error {error}" if error else "no error") + (f"; `{check['said']}`" if check["said"] else "")
        lines.append(said + ".")
    lines += [f"- {t['what']}: " + {True: "yes", False: "NO", None: "not checked, a file is not given"}[t["holds"]] + "."
              for t in doc["ties"]]
    lines += _prover_said(doc["prover"]) + _security_said(doc["security"])
    if (record := doc["verdict_record"]) is not None:
        lines += ["", f"The verdict as a record of the stop class's verdicts of record (a line of `{E.VERDICTS_FILE}`, the "
                  f"piece `{PIECE}`):", "", "    " + json.dumps(record, ensure_ascii=False)]
    return "\n".join(lines)
