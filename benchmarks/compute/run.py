"""`verity bench compute`: the share of a served model's multiply-accumulates (MACs) that Pearl+ credits, over the tokens
one served run served: the MACs the Pearl+ verifier credits over the MACs the model computes on those tokens, as text
and as one line of JSON (`value`, `unit`, `commit`, `run`), beside its target (`TARGET`). An answer that isn't an
accept prints no number and exits 1.

    verity bench compute --from RUN                                  a recorded run
    verity bench compute --from RUN --per-call [--uncredited F] [--top N]   and where the rest of its MACs went

The number, for a run whose inference-only policy registers one class of S slots, each P prompt tokens then O output
tokens, served by a model of L layers, hidden size d, H attention heads of d_h = d / H, H_kv key and value heads, MLP
width d_ff and vocabulary V (`MODELS`):

    value     = credited / (linears + attention + LM head)
    credited  = the Pearl+ verifier's `public.work.credited`
    T         = P + O - 1                                      the positions a slot runs: its prompt, then each output
                                                               but the last, fed back
    linears   = S * T * L * (2*d*d + 2*d*H_kv*d_h + 3*d*d_ff)  each position through each layer's q, k, v, o, gate, up
                                                               and down
    attention = S * T*(T+1)/2 * L * H * 2*d_h                  each position's query against its causal keys: QK^T and
                                                               PV, d_h MACs each a head
    LM head   = S * O * d * V                                  one row of logits an output token

What has no multiply-accumulate (the embedding's read, the norms, the activations, RoPE, the softmax) is not counted.
Pearl+ credits linears alone. So the number is at most the linears' share of the model's MACs at the class, which the
text prints beside it, and a credit above the class's linears is refused: its passes are not one load of the class.

`--from RUN` reads the verifier's side of the run's record in the evidence store, and nothing the developer's node
reports of what it served:

- `out/pieces/pouw/pearlplus-answer.json`, the Pearl+ verifier's answer: its verdict, its credit, and its sample's
  ε_s at δ_s (`epsilon_s`, `delta_s`): a sample that accepts misses more than the share ε_s of the credited work with
  probability at most δ_s. They are sampling's alone, not the ε and δ the credit holds at: the assumption's gap ε_a
  and its own δ_a, and the binding of the passes' commitments, are not in them;
- `out/pieces/pouw/pearlplus-lean.json` and `pearlplus-lean-check.json`, the Lean verifier's verdict and the record's
  comparison of it with the answer: where the record holds them they must be the answer's, and where it holds no Lean
  verdict the text says so;
- `out/inference-only-policy.json`, the registered policy: the class and the model's name. Where it also counts its
  class's MACs a slot, that count must be this command's;
- `out/record/verdicts.json`, the run's composed record: PoUW's row. A row that isn't an accept (a development run's
  reads `missing`) is printed beside the number, never left out: the credit is then the Pearl+ piece's alone.

The record holds neither the tokens the run served nor the seconds it served them in (both stay on the developer's
node). So the tokens are the class's, and the capacity share (the credited MACs over the card's rate times the seconds
the verifier holds the passes to) is printed as absent: a Pearl+ answer states no seconds, and the node's own are not
read.

`--per-call` also prints where the model's MACs the credit leaves out went (its give-up: the model's MACs less the
credit), by cause and by matmul call, from the Pearl+ verifier's split of its accepted verdict's credit call by call
(`out/pieces/pouw/verify.json`'s ``uncredited``, which Pearl+'s verifier writes once it is on main, or
``--uncredited F``, such a split written beside the run's passes). Each call's real MACs are the MACs its credited cells
hold at |K| and the cells the band rule leaves out (excluded A rows, excluded B rows' column blocks, voluntary rows'
bands and blocks, and the side channels' k − |K|); its credited cells' MACs are the deduction's terms (ε's, that of
sampled tiles over the inert-work bound, and the inert-work share ι's) and its credit, as exact fractions. The causes
over the model are those, the LM head and attention (which Pearl+ doesn't credit), the linears outside the passes, and
the floor of the credit; they sum to the give-up exactly. The split is the answer's or it is refused (exit 2): its
commitment and its passes' records roots are the answer's, each call's terms sum as above, and its calls' credited
cells' MACs are the answer's ``macs`` and their credit, floored, its ``credited``. The ε deduction is the run's, at its
sample: the verifier takes it off the records whose cells hold the most MACs a ticket, so the calls it lands on carry it
for the whole run. ``--top N`` (default 10) calls are printed, by give-up.

It exits 0 with the number printed, 1 with none (the answer isn't an accept, the Lean verifier's verdict isn't the
answer's, or the run left no answer) and 2 when the command line, the run or its policy can't be read. A record
withdrawn whole, or in a part that holds a file read here, is refused before any of it is fetched; a withdrawal of
other files of it is printed beside the number. Under `research run` the printed document is also `$OUT/compute.json`;
there the evidence store is read with the run's `--store-read` key.

Under `research run` it also writes, through the harness kit's writer (`verity_bench.kit`), `$OUT/result.json` with the
fields every resource's benchmark reports (`amounts`), each stamped with this run and its commit and computed from the
recorded run: the metric (the printed number, its unit and target), the verified MACs (the credit), the capacity and
the free MACs at each data-sheet reading, and the error beside them (the answer's δ_s, sampling's alone). Their names
are every resource's: `metric`, `verified`, `capacity`, `free` and `error`. A reading is a clock of a
registered device's card file (`clock.tensor`, `clock.sm`; `verity.hardware.cards`), capacity the registered devices'
peak e4m3 MACs a second at that clock times the run's span (`capacity`), and free capacity less verified. A recorded
run's record holds no registration of its hardware and no span the verifier holds its passes to, so capacity and free
are written as one entry each, with no value and why.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Any, NamedTuple

from verity.hardware import cards
from verity.hardware.site import Site
from verity_bench import kit
from verity_config.settings import SHARED

UNIT = "share of the served model's MACs that Pearl+ credits, over the tokens served"
#: the share the number is driven to
TARGET = Fraction(97, 100)
PROTOCOL = "verity/pouw/pearl-plus/v1"
#: the files of a run this command reads
ANSWER = "out/pieces/pouw/pearlplus-answer.json"
LEAN = "out/pieces/pouw/pearlplus-lean.json"
LEAN_CHECK = "out/pieces/pouw/pearlplus-lean-check.json"
POLICY = "out/inference-only-policy.json"
VERDICTS = "out/record/verdicts.json"
RECORDS = [ANSWER, LEAN, LEAN_CHECK, POLICY, VERDICTS]
#: the verifier's document, read by `--per-call` alone
VERIFY = "out/pieces/pouw/verify.json"
#: the causes of the per-call split's give-up, in the order printed
CAUSES = {"lm_head": "LM head, which Pearl+ doesn't credit", "attention": "attention, which Pearl+ doesn't credit",
          "outside": "linears outside the passes", "excluded_a": "excluded A rows",
          "excluded_b": "excluded B rows' column blocks", "voluntary": "voluntary rows' bands and blocks",
          "kept_width": "side channels (k beyond |K|)", "epsilon_macs": "ε's deduction",
          "over_bound_macs": "sampled tiles over the inert-work bound", "rho_macs": "ι, the inert-work share",
          "floor": "the credit's floor"}
#: a call's terms the split states, beside its MACs
ROWS = ("excluded_a", "excluded_b", "voluntary", "kept_width")
TERMS = ("epsilon_macs", "over_bound_macs", "rho_macs", "credited_macs")
ABSENT = ("absent: a Pearl+ answer holds its passes to no seconds, and the seconds the run served in stay on the "
          "developer's node")
#: what the answer's ε and δ leave out, printed beside them
SAMPLING_ALONE = "sampling's alone: the assumption's gap and δ, and binding, are not in them"
#: the data sheet's readings of a device's peak, each a clock of its card file (`clock.tensor`, `clock.sm`)
READINGS = ("tensor", "sm")
#: the operand format of the MACs Pearl+ credits
FORMAT = "e4m3"
NO_SITE = "the run's record holds no registration of its hardware: no site entry naming its devices and their sheets"
NO_SPAN = ABSENT.removeprefix("absent: ")
DELTA_S = ("probability that a sample that accepts misses more than the share ε_s of the credited MACs: the answer's δ_s, "
           "sampling's alone (the assumption's gap and δ, and binding, are not in it)")
#: how the store ends the reason of a withdrawal of only the files under some prefixes of a record
SCOPED = re.compile(r" \(only ([^()]+)\)$")


class Shape(NamedTuple):
    """A decoder's shape, as its published configuration gives it."""
    layers: int
    hidden: int
    heads: int
    kv_heads: int
    mlp: int
    vocab: int


#: each served model's shape, by the name its policy serves it under
MODELS = {"unsloth/Llama-3.2-1B": Shape(layers=16, hidden=2048, heads=32, kv_heads=8, mlp=8192, vocab=128256)}


class NoNumber(Exception):
    """The run has no number to print: its answer isn't an accept, or it left none."""


class Unreadable(Exception):
    """A run, its policy or its record can't be read."""


def _under(path: str, prefix: str) -> bool:
    return path == prefix.rstrip("/") or path.startswith(prefix.rstrip("/") + "/")


class EvidenceStore:
    """The evidence store as this command reads it: a run's attempt and some of its files. `store` is research's local
    store, `pull` brings its labels of some targets up to the remote's, `error` is what the store raises, and
    `withdrawn` is the key of the label a withdrawal leaves."""

    def __init__(self, store, pull: Callable, error: type[Exception], withdrawn: str) -> None:
        self.store, self.pull, self.error, self.withdrawn = store, pull, error, withdrawn

    def attempt(self, run: str) -> dict | None:
        try:
            return self.store.get_attempt(run)
        except (self.error, ValueError) as e:
            raise Unreadable(f"the evidence store can't be read: {e}") from e

    def files(self, run: str, paths: list[str]) -> tuple[Path | None, list[str]]:
        """A directory holding the files of `run`'s record at `paths`, each a file's whole path in the run (None when
        the record holds none of them), and what was withdrawn of the rest of the record, each as its label says it.
        A record withdrawn whole, or in a part that holds one of `paths`, is refused before any of it is fetched."""
        a = self.attempt(run)
        if a is None:
            raise Unreadable(f"{run} is not in the evidence store this machine reads")
        art = (a.get("outputs") or {}).get("run_record") or (a.get("outputs") or {}).get("run_files")
        if not art:
            raise Unreadable(f"{run} kept no record")
        self.pull(self.store, [run, art], lambda m: print(f"compute: {m}", file=sys.stderr))
        try:
            names = [f["path"] for f in self.store.listing(art)]
            gone = []
            for label in (lab for target in (art, run) for lab in self.store.labels(target, key=self.withdrawn)):
                said = f"{label.value!r}, by {label.by}"
                scoped = SCOPED.search(str(label.value))
                prefixes = scoped.group(1).split(", ") if scoped else []
                # a reason can end like a list of prefixes without being one: each must hold a file of this record
                partial = bool(prefixes) and all(any(_under(n, p) for n in names) for p in prefixes)
                if not partial or any(_under(f, p) for f in paths for p in prefixes):
                    raise Unreadable(f"{run} was withdrawn ({said}): nothing of it is fetched")
                gone.append(said)
            if not any(n in paths for n in names):
                return None, gone
            return Path(self.store.fetch(art, paths=paths)), gone
        except self.error as e:
            raise Unreadable(f"{run}'s record can't be fetched: {e}") from e


def opened() -> EvidenceStore:
    """The evidence store this machine reads, through research's client."""
    try:
        from research.store import labels_sync, vocab
        from research.store.local import LocalStore, StoreError
    except ImportError as e:
        raise Unreadable(f"the evidence store's client isn't installed here ({e})") from e
    return EvidenceStore(LocalStore(), labels_sync.read_through, StoreError, vocab.WITHDRAWN)


def model_macs(shape: Shape, slots: int, prompt: int, outputs: int) -> dict[str, int]:
    """The MACs a model of `shape` computes serving `slots` slots of `prompt` prompt tokens then `outputs` output
    tokens each: its linears', its attention's and its LM head's (module docstring)."""
    head_dim = shape.hidden // shape.heads
    positions = prompt + outputs - 1
    layer = 2 * shape.hidden ** 2 + 2 * shape.hidden * shape.kv_heads * head_dim + 3 * shape.hidden * shape.mlp
    return {"linears": slots * positions * shape.layers * layer,
            "attention": slots * (positions * (positions + 1) // 2) * shape.layers * shape.heads * 2 * head_dim,
            "lm_head": slots * outputs * shape.hidden * shape.vocab}


def _count(n: Any) -> bool:
    return type(n) is int and n >= 0


def _held(tree: Path, path: str) -> Any:
    """The JSON the record under `tree` holds at `path`; None when it holds no such file."""
    return json.loads((tree / path).read_text()) if (tree / path).is_file() else None


def credit(tree: Path) -> dict:
    """The Pearl+ verifier's credit in the record under `tree`: `NoNumber` unless its answer is Pearl+'s accept with a
    credit, and the Lean verifier's verdict and the record's comparison of the two, where it holds them, are the
    answer's. `epsilon_s` and `delta_s` are the answer's `public.work.epsilon` and `profile.delta`, its sample's."""
    try:
        answer, lean, check = (_held(tree, p) for p in (ANSWER, LEAN, LEAN_CHECK))
        if answer is None:
            raise NoNumber("the run left no Pearl+ answer")
        protocol, verdict, public = answer["protocol"], answer["verdict"], answer.get("public") or {}
        work = public.get("work") or {}
        credited, passes = work.get("credited"), work.get("macs")
        mine = ("accept", answer.get("commitment"), credited)
        theirs = None if lean is None else (lean["verdict"], lean.get("commitment"), (lean.get("work") or {}).get("credited"))
        differ = None if check is None else check["differ"]
    except (OSError, KeyError, TypeError, ValueError, AttributeError) as e:
        raise NoNumber(f"the run's Pearl+ verdicts don't read as verdicts ({type(e).__name__}: {e})") from e
    if protocol != PROTOCOL:
        raise NoNumber(f"the answer is {protocol}'s, not {PROTOCOL}'s")
    if verdict != "accept":
        raise NoNumber(f"Pearl+'s answer is {verdict}, not accept ({answer.get('reason') or 'no reason recorded'})")
    if not _count(credited) or passes is not None and not _count(passes):
        raise NoNumber(f"Pearl+'s answer accepts and holds no credit in MACs (credited {credited!r}, macs {passes!r})")
    for name, lean_s, answer_s in zip(("verdict", "run commitment", "credit"), theirs or mine, mine):
        if lean_s != answer_s:
            raise NoNumber(f"the Lean verifier's {name} is {str(lean_s)[:16]} and the answer's {str(answer_s)[:16]}")
    if differ:
        raise NoNumber(f"the record's comparison of the Lean verifier's verdict with the answer differs on {differ}")
    return {"credited": credited, "passes": passes, "scheme": public.get("scheme"), "epsilon_s": work.get("epsilon"),
            "delta_s": (answer.get("profile") or {}).get("delta"), "lean": theirs is not None}


def load(tree: Path) -> dict:
    """The load the run's inference-only policy registers, from the record under `tree`: the model, the class and the
    MACs the model computes on it. `Unreadable` unless the policy names a class and a model of `MODELS`, and its own
    count of the class's MACs a slot, where it has one, is `model_macs`'s."""
    try:
        policy = _held(tree, POLICY)
        if policy is None:
            raise Unreadable("the run's record holds no inference-only policy, so no class to count the model's MACs at")
        cls, model = policy["class"], policy["serve"]["model"]
        slots, prompt, outputs, positions = (cls.get(k) for k in ("slots", "prompt", "outputs", "positions"))
        priced = (policy.get("n_inputs") or {}).get("work")
    except (OSError, KeyError, TypeError, ValueError, AttributeError) as e:
        raise Unreadable(f"the run's inference-only policy doesn't read as one ({type(e).__name__}: {e})") from e
    if not all(_count(n) and n for n in (slots, prompt, outputs)):
        raise Unreadable(f"the policy's class is not slots, prompt tokens and output tokens, each at least 1: {cls!r}")
    if positions is not None and positions != prompt + outputs - 1:
        raise Unreadable(f"the policy's class runs {positions} positions a slot, not the {prompt + outputs - 1} of "
                         f"{prompt} prompt and {outputs} output tokens")
    if model not in MODELS:
        raise Unreadable(f"no shape for the served model {model!r}: add it to MODELS in {Path(__file__).name} "
                         f"(it holds {', '.join(sorted(MODELS))})")
    macs = model_macs(MODELS[model], slots, prompt, outputs)
    agrees = None
    if isinstance(priced, dict) and all(_count(priced.get(k)) for k in macs):
        if (theirs := {k: slots * priced[k] for k in macs}) != macs:
            raise Unreadable(f"the policy counts its class's MACs as {theirs} and {Path(__file__).name} as {macs}: one "
                             f"of the two has the wrong shape for {model}")
        agrees = True
    return {"model": model, "class": {"slots": slots, "prompt": prompt, "outputs": outputs}, "macs": macs,
            "policy_agrees": agrees,
            "tokens": {"prompt": slots * prompt, "output": slots * outputs, "positions": slots * (prompt + outputs - 1)}}


def row(tree: Path) -> dict:
    """PoUW's row in the composed record under `tree`: its verdict and why, both None where the record composed
    none."""
    try:
        held = _held(tree, VERDICTS)
        work = {} if held is None else held["protocols"].get("work") or {}
        return {"verdict": work.get("verdict"), "why": work.get("why") or work.get("reason")}
    except (OSError, KeyError, TypeError, ValueError, AttributeError) as e:
        raise Unreadable(f"the run's composed record doesn't read as one ({type(e).__name__}: {e})") from e


def linear(shape: Shape, k: int, n: int) -> str:
    """Which of a layer's linears a call of a model of `shape` is, by its k × n."""
    d, head_dim = shape.hidden, shape.hidden // shape.heads
    names = {(d, d + 2 * shape.kv_heads * head_dim): "qkv", (d, d): "o", (d, 2 * shape.mlp): "gate_up", (shape.mlp, d): "down"}
    return names.get((k, n), f"{k}x{n}")


def split(tree: Path, given: Path | None, macs: dict[str, int], credited: int, shape: Shape, top: int) -> dict:
    """The give-up of the record under `tree` by cause and by call (module docstring), from the verifier's split in its
    verify document or in `given`, such a split in a file of its own. `Unreadable` unless the split is the answer's. `macs` are the model's MACs at the class and `credited` the answer's credit."""
    try:
        held = json.loads(given.read_text()) if given is not None else _held(tree, VERIFY)
        if held is None or (given is None and "uncredited" not in held):
            raise Unreadable("the record holds no per-call give-up (`uncredited`): the verifier writes it once Pearl+'s "
                             "verifier side of the per-call give-up is on main")
        answer = _held(tree, ANSWER)
        work, passes = answer["public"]["work"], answer["public"].get("passes")
        got, records = held["uncredited"], held.get("records")
        calls, totals = got["calls"], got["totals"]
        sums = dict.fromkeys(("macs", "checked_macs", *ROWS, *TERMS), Fraction(0))
        for c in calls:
            terms = {x: Fraction(c[x]) for x in TERMS}
            rows = {x: c["uncredited_macs"][x] for x in ROWS}
            if not all(_count(v) for v in (c["macs"], c["checked_macs"], *rows.values())) or min(terms.values()) < 0:
                raise Unreadable(f"the per-call split's call {c['call']} holds a count that is no count of MACs")
            if c["macs"] != c["checked_macs"] + sum(rows.values()) or c["checked_macs"] != sum(terms.values()):
                raise Unreadable(f"the per-call split's call {c['call']} doesn't sum: its MACs are not its checked MACs "
                                 f"and its causes, or its checked MACs not its terms")
            for x, v in (("macs", c["macs"]), ("checked_macs", c["checked_macs"]), *rows.items(), *terms.items()):
                sums[x] += v
        stated = {x: Fraction(totals[x]) for x in sums}
    except (OSError, KeyError, TypeError, ValueError, AttributeError, ZeroDivisionError) as e:
        raise Unreadable(f"the per-call split doesn't read as one ({type(e).__name__}: {e})") from e
    if held["commitment"] != answer["commitment"]:
        raise Unreadable("the per-call split is of another run commitment than the answer's")
    if records is not None and passes is not None and records != [p["records"] for p in passes]:
        raise Unreadable("the per-call split's passes' records are not the answer's")
    if stated != sums or totals.get("credited") != int(sums["credited_macs"]):
        raise Unreadable("the per-call split's totals are not its calls' sums")
    if (sums["checked_macs"], int(sums["credited_macs"])) != (work["macs"], credited):
        raise Unreadable(f"the per-call split's calls credit {int(sums['credited_macs']):,} of "
                         f"{int(sums['checked_macs']):,} MACs and the answer {credited:,} of {work['macs']:,}")
    if sums["macs"] > macs["linears"]:
        raise Unreadable(f"the per-call split's calls hold {int(sums['macs']):,} real MACs, more than the "
                         f"{macs['linears']:,} of the class's linears")
    gave = {"lm_head": macs["lm_head"], "attention": macs["attention"], "outside": macs["linears"] - sums["macs"],
            **{x: sums[x] for x in (*ROWS, *TERMS[:3])}, "floor": sums["credited_macs"] - credited}
    give_up = sum(macs.values()) - credited
    assert sum(gave.values()) == give_up, (gave, give_up)

    def lost(c: dict) -> Fraction:
        return c["macs"] - Fraction(c["credited_macs"])

    worst = sorted(calls, key=lost, reverse=True)[:top]
    return {"source": str(given) if given is not None else VERIFY, "give_up": give_up, "calls": len(calls),
            "causes": [{"cause": x, "macs": str(v), "share": float(v / give_up) if give_up else 0.0}
                       for x, v in gave.items()],
            "top": [{"pass": c["pass"], "call": c["call"], "weight": c["weight"], "linear": linear(shape, c["k"], c["n"]),
                     "m": c["m"], "k": c["k"], "n": c["n"], "real_rows": c["real_rows"], "rows": c["rows"],
                     "give_up": str(lost(c)), "causes": {x: str(v) for x, v in
                                                          (*((x, c["uncredited_macs"][x]) for x in ROWS),
                                                           *((x, c[x]) for x in TERMS[:3])) if Fraction(v)}}
                    for c in worst]}


def document(tree: Path, commit: str | None, run: str | None, withdrawn: list[str]) -> dict:
    """The printed document of the record under `tree`: the number, its unit, the commit the run ran from and the run,
    then what the number is made of. `withdrawn` is what was withdrawn of the rest of the record."""
    c, served, record = credit(tree), load(tree), row(tree)
    macs = served["macs"]
    most = max(c["credited"], c["passes"] or 0)
    if most > macs["linears"]:
        raise Unreadable(f"Pearl+'s answer holds {most:,} MACs, more than the {macs['linears']:,} of the linears of "
                         f"the policy's class: its passes are not one load of the class")
    total = sum(macs.values())
    return {"value": c["credited"] / total, "unit": UNIT, "commit": commit, "run": run, "target": float(TARGET),
            "met": Fraction(c["credited"], total) >= TARGET, "credited_macs": c["credited"], "model_macs": total,
            "macs": macs, "linears_share": macs["linears"] / total, "passes_macs": c["passes"], "scheme": c["scheme"],
            "epsilon_s": c["epsilon_s"], "delta_s": c["delta_s"], "lean": c["lean"], "record": record,
            "withdrawn": withdrawn,
            "model": served["model"], "class": served["class"], "policy_agrees": served["policy_agrees"],
            "tokens": served["tokens"], "tokens_from": "class", "wall_s": None, "capacity_share": None}


def capacity(site: Site, span_s: Fraction) -> dict[str, int]:
    """The registered hardware's MACs over `span_s` seconds at each data-sheet reading (`READINGS`): each registered
    device's peak `FORMAT` MACs a second at its card's clock of that reading (`verity.hardware.cards.rate` on the card
    file of its sheet's SKU), summed over the devices, times the span, rounded up. `Unreadable` where the site registers
    no device, or a device names no registered sheet or an SKU with no card file."""
    if not site.units:
        raise Unreadable("the registered hardware holds no device")
    total = dict.fromkeys(READINGS, Fraction(0))
    for d in site.units:
        if (sheet := site.sheet(d)) is None:
            raise Unreadable(f"the registered device {d} names no registered sheet")
        try:
            card = cards.load(sheet.sku)
        except KeyError as e:
            raise Unreadable(f"the registered device {d}'s SKU has no card file: {e}") from e
        for r in READINGS:
            total[r] += cards.rate(card, fmt=FORMAT, mhz=card["clock"][r]["value"]) * span_s
    return {r: math.ceil(v) for r, v in total.items()}


def amounts(doc: dict, site: Site | None = None, span_s: Fraction | None = None) -> list[dict]:
    """The result a document `doc` gives in the fields every resource's benchmark reports, as the kit writer's
    measurements: the metric (`metric`, the printed number in its unit, beside its target), the verified amount
    (`verified`, the credited MACs), at each data-sheet reading the capacity (`capacity at clock.tensor`, ..., from the
    registered hardware `site` over the run's span `span_s`) and the free amount (`free at ...`, capacity less verified),
    and the probability the verified amount holds except with (`error`, the answer's δ_s, which its `inputs` name).
    Where `site` or `span_s` is None, capacity and free are one entry each, with no value and why. Each is computed from
    the recorded run."""
    inputs = [doc["run"]] if doc["run"] else []
    verified = doc["credited_macs"]
    out = [{"name": "metric", "value": doc["value"], "unit": doc["unit"], "target": doc["target"],
            "verdict": "at or above target" if doc["met"] else "below target", "inputs": inputs},
           {"name": "verified", "value": verified, "unit": "MAC", "inputs": inputs}]
    if site is None or span_s is None:
        why = "absent: " + "; and ".join(w for w, x in ((NO_SITE, site), (NO_SPAN, span_s)) if x is None)
        out += [{"name": x, "value": None, "unit": "MAC", "inputs": inputs, "why": why} for x in ("capacity", "free")]
    else:
        for r, c in capacity(site, span_s).items():
            out += [{"name": f"capacity at clock.{r}", "value": c, "unit": "MAC", "inputs": inputs},
                    {"name": f"free at clock.{r}", "value": c - verified, "unit": "MAC", "inputs": inputs}]
    delta = doc["delta_s"]
    out.append({"name": "error", "value": None if delta is None else float(Fraction(str(delta))), "unit": DELTA_S,
                "inputs": [*inputs, "delta_s"], **({"why": "the answer states no δ_s"} if delta is None else {})})
    return out


def commit_of(attempt: dict) -> str | None:
    source = attempt.get("source") or {}
    return f"{source['commit']}-dirty" if source.get("commit") and source.get("dirty") else source.get("commit")


def from_run(store, run: str, per_call: bool = False, given: Path | None = None, top: int = 10) -> dict:
    """The printed document of recorded run `run`; with `per_call`, its give-up by cause and by call too (`split`)."""
    attempt = store.attempt(run)
    if attempt is None:
        raise Unreadable(f"{run} is not in the evidence store this machine reads")
    tree, withdrawn = store.files(run, RECORDS + [VERIFY] * (per_call and given is None))
    if tree is None or not (tree / ANSWER).is_file():
        raise NoNumber(f"{run} left no Pearl+ answer (it ended rc {(attempt.get('status') or {}).get('rc')})")
    doc = document(tree, commit_of(attempt), run, withdrawn)
    if per_call:
        doc["per_call"] = split(tree, given, doc["macs"], doc["credited_macs"], MODELS[doc["model"]], top)
    return doc


def _m(v: str | int | Fraction) -> str:
    return f"{round(Fraction(v)):,}"


def per_call_text(got: dict, model_macs: int) -> list[str]:
    """The printed lines of a document's `per_call`: its give-up by cause, then its top calls."""
    lines = [f"  per call: the model gives up {got['give_up']:,} of its MACs ({100 * got['give_up'] / model_macs:.2f}%); "
             f"by cause, from the verifier's split of its {got['calls']} calls ({got['source']}):"]
    width = max(len(CAUSES[c["cause"]]) for c in got["causes"])
    lines += [f"    {CAUSES[c['cause']]:<{width}}  {_m(c['macs']):>17} MACs  {100 * c['share']:6.2f}%" for c in got["causes"]]
    lines.append("  ε's deduction is the run's, at its sample: the verifier takes it off the records whose cells hold the most "
                 "MACs a ticket, so the calls it lands on carry it for the whole run")
    lines.append(f"  the {len(got['top'])} calls that give up the most:")
    for c in got["top"]:
        said = ", ".join(f"{CAUSES[x].split(',')[0].split(' (')[0]} {_m(v)}" for x, v in c["causes"].items())
        lines.append(f"    pass {c['pass']} call {c['call']}, weight {c['weight']} {c['linear']}, {c['m']}x{c['k']}x{c['n']} "
                     f"with {c['real_rows']} real rows: {_m(c['give_up'])} MACs" + (f" ({said})" if said else ""))
    return lines


def text(doc: dict) -> str:
    """The printed lines of a document: the number, what accepted it, and what it is made of."""
    macs, cls, tokens, record = doc["macs"], doc["class"], doc["tokens"], doc["record"]
    lines = [f"compute: {100 * doc['value']:.2f}% of the served model's MACs are credited by Pearl+ "
             f"({doc['credited_macs']:,} of {doc['model_macs']:,}), {'at or above' if doc['met'] else 'below'} the "
             f"{100 * doc['target']:g}% target"]
    at = ", ".join(f"{k} {doc[name]}" for k, name in (("ε_s", "epsilon_s"), ("δ_s", "delta_s")) if doc[name] is not None)
    lines.append(f"  {doc['run'] or 'no recorded run'}, commit {doc['commit'] or 'unknown'}: Pearl+'s answer accepts"
                 + (f" (scheme {doc['scheme']})" if doc["scheme"] else "")
                 + (f" at {at} ({SAMPLING_ALONE})" if at else "")
                 + (", and the Lean verifier's verdict is the same" if doc["lean"] else ", and the record holds no Lean "
                    "verdict"))
    lines.append("  PoUW's row in the run's composed record " + ("accepts too" if record["verdict"] == "accept" else
                 (f"is `{record['verdict']}`, not an accept" if record["verdict"] else "is absent")
                 + ", so the credit is the Pearl+ piece's alone" + (f": {record['why']}" if record["why"] else "")))
    if doc["withdrawn"]:
        lines.append(f"  part of the run's record was withdrawn, none of it a file read here: {'; '.join(doc['withdrawn'])}")
    lines.append(f"  {doc['model']} at {cls['slots']} slots of {cls['prompt']} prompt and {cls['outputs']} output "
                 f"tokens: linears {macs['linears']:,}, attention {macs['attention']:,} and LM head {macs['lm_head']:,} "
                 f"MACs" + (", which is the policy's own count of its class" if doc["policy_agrees"] else ""))
    lines.append(f"  Pearl+ credits linears alone, {100 * doc['linears_share']:.2f}% of the model's MACs at this class: "
                 + (f"its passes hold {doc['passes_macs']:,} of them and " if doc["passes_macs"] is not None else "")
                 + f"its credit is {100 * doc['credited_macs'] / macs['linears']:.2f}% of them all")
    lines.append(f"  tokens, the class's since the record holds no count of those served: {tokens['prompt']:,} prompt "
                 f"and {tokens['output']:,} output, over {tokens['positions']:,} positions")
    lines.append(f"  capacity share: {ABSENT}")
    if "per_call" in doc:
        lines += per_call_text(doc["per_call"], doc["model_macs"])
    return "\n".join(lines)


def main(argv: list[str] | None = None, *, store_of: Callable[[], EvidenceStore] = opened) -> int:
    """The command: 0 with the number printed, 1 with none, 2 when the command line, the run or its policy can't be
    read. `store_of` opens the evidence store."""
    ap = argparse.ArgumentParser(prog="verity bench compute", description=__doc__.split("\n\n")[0])
    ap.add_argument("--from", dest="run", metavar="RUN", required=True,
                    help="a recorded run: print from its record in the evidence store")
    ap.add_argument("--per-call", action="store_true",
                    help="also print where the model's MACs the credit leaves out went, by cause and by matmul call")
    ap.add_argument("--uncredited", type=Path, metavar="F",
                    help="with --per-call: a per-call give-up (`uncredited`) in a file of its own, for a record that holds "
                         "none; the verifier writes it once Pearl+'s verifier side of the per-call give-up is on main")
    ap.add_argument("--top", type=int, default=10, metavar="N", help="with --per-call: the calls printed (default 10)")
    a = ap.parse_args(argv)
    if (a.uncredited or a.top != 10) and not a.per_call:
        ap.error("--uncredited and --top go with --per-call")
    try:
        doc = from_run(store_of(), a.run, a.per_call, a.uncredited, a.top)
        measured = amounts(doc)
    except NoNumber as e:
        print(f"compute: no number: {e}", file=sys.stderr)
        return 1
    except Unreadable as e:
        print(f"compute: {e}", file=sys.stderr)
        return 2
    print(text(doc))
    line = json.dumps(doc)
    print(line)
    if out := SHARED.OUT:
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "compute.json").write_text(line + "\n")
        writer = kit.writer(out)
        for m in measured:
            writer.measurement(**m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
