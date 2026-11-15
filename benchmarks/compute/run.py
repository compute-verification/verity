"""`verity bench compute`: the share of a served model's multiply-accumulates (MACs) that Pearl+ credits, over the tokens
one served run served: the MACs the Pearl+ verifier credits over the MACs the model computes on those tokens, as text
and as one line of JSON (`value`, `unit`, `commit`, `run`), beside its target (`TARGET`). An answer that isn't an
accept, is a test instance's (`profile.test_instance`), or that the record holds no Lean verdict of, prints no number
and exits 1.

    verity bench compute --from DIR                          a run's record in a directory on disk
    verity bench compute --from RUN                          a recorded run in our evidence store
    verity bench compute --from DIR|RUN --per-call [--uncredited F] [--top N]   and where the rest of its MACs went

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

The load is the benchmark's, not the run's (`LOAD`, `load.toml`): the served model, its revision and the class S, P, O
it fixes. A record whose policy registers another model, revision or class is refused (exit 2, naming both), since the
share depends on the class (a class of long prompts and one output credits more of its MACs than one with more outputs).

What has no multiply-accumulate (the embedding's read, the norms, the activations, RoPE, the softmax) is not counted.
Pearl+ credits linears, and the LM head where the run's passes hold its calls: the record holds the head when the
verifier's per-call split (below), over the answer's run commitment, names a call of the head's shape, k = d and n = V
(`head_calls`). So the number is at most the share of the model's MACs at the class that its passes can hold, the
linears' (and the LM head's with the head), which the text prints beside it, and a credit above them is refused: its
passes are not one load of the class. A record that holds no split, as runs before Pearl+'s verifier wrote it do, holds
no head.

A run's record is its run directory as `research run` leaves it and the evidence store keeps it: the run's `job.json`,
and what the run wrote under `out/`. `--from` takes a directory or a run id (`main`). Given a directory, it reads the
record from there through no store (`kit.record_dir`): nothing of research's is imported, no credential is looked up
and nothing goes over the network. DIR must hold the answer and the policy below (exit 2 naming the file it lacks). The
run printed is the `run.id` its `job.json` states and the commit its `store.conditions` state (`source.commit`,
`-dirty` when `source.dirty`), as the store's attempt states them; without them, the directory's name and no commit.
Nothing is withdrawn from a directory. Given a run id (`kit_report.RUN_ID`) that names no directory, it reads that
run's record from the evidence store, through research's client. Anything else is refused (exit 2). A record whose
`job.json` says it is made up (``made_up``, as the tests' fixture does) is reported as made up through the kit
(`kit.report`'s ``made_up``): its line under the number, ``made_up: true``, what the record says of itself
(``made_up_says``), and `result.json`'s entries marked modeled and made up, never measured.

Either way it reads the verifier's side of the record, and nothing the developer's node reports of what it served:

- `out/pieces/pouw/pearlplus-answer.json`, the Pearl+ verifier's answer: its verdict, its credit, and its sample's
  ε_s at δ_s (`epsilon_s`, `delta_s`): a sample that accepts misses more than the share ε_s of the credited work with
  probability at most δ_s. They are sampling's alone, not the ε and δ the credit holds at: the assumption's gap ε_a
  and its own δ_a, and the binding of the passes' commitments, are not in them;
- `out/pieces/pouw/pearlplus-lean.json` and `pearlplus-lean-check.json`, the Lean verifier's verdict and the record's
  comparison of it with the answer: the record must hold both and they must be the answer's, or no number is printed
  (exit 1, naming the file the record lacks);
- `out/inference-only-policy.json`, the registered policy: the class and the model's name. Where it also counts its
  class's MACs a slot, that count must be this command's;
- `out/record/verdicts.json`, the run's composed record: PoUW's row. A row that isn't an accept (a development run's
  reads `missing`) is printed beside the number, never left out: PoUW then credits nothing of the run, and the number
  is the Pearl+ verifier's answer alone. Under the answer's verdict a line says what the credit rests on (`RESTS_ON`);
- `out/pieces/pouw/verify.json`, the Pearl+ verifier's document, where the record holds it: its per-call split, which
  says whether the passes hold the LM head's calls, and which `--per-call` prints;
- `out/pieces/pouw/development.json`, the Pearl+ piece's document (`served_verdicts.py pearlplus`), where the record
  holds it: its ``coin_source`` alone, where the passes' salt coins came from (`coins`).

The line under the number is the kit's ``ran with: coins COINS; parameters PARAMETERS (DETAIL)`` (`ran_with`), and
below it the kit's ``a measurement, not an audit``, always: this command never prints an audit, and ``audit`` is false
in `compute.json` and in each of `result.json`'s entries. COINS is who drew the coins: the launcher's ``challenger``
in `out/compute-run.json` (`RAN`, written by `serve.sh`: ``local`` or ``in-process``), else ``in-process`` where the
answer's sample is the node's or the salt coins an in-process challenger's; a record that says neither prints
``ran with: not stated`` and, below it, the coins it does state. PARAMETERS is ``development``. A launcher that says
``deployed`` or ``development`` false exits 2 (`NO_DEPLOYED`): the person's own run writes every file of the record,
so no file of it can show the auditor's challenger drew the coins, and reading a deployed run waits on a check of the
challenger's signature against a pinned key. DETAIL is what the record states
of the rest (`coins`): the answer's draws and δ and scheme, its sample source (`profile.source_from`), the piece's salt
coins (``coin_source``), and whether the Lean verifier ran as of record (``--of-record`` in
`pearlplus-lean-check.json`'s ``argv``; with a ``local`` challenger, ``of record against the local challenger's key``,
the test key the run made), each ``not stated`` where the record holds none. None of them refuses a number.

The record holds neither the tokens the run served nor the seconds it served them in (both stay on the developer's
node). So the tokens are the class's, and the capacity share (the credited MACs over the card's rate times the seconds
the verifier holds the passes to) is printed as absent: a Pearl+ answer states no seconds, and the node's own are not
read.

`--per-call` also prints where the model's MACs the credit leaves out went (its give-up: the model's MACs less the
credit), by cause and by matmul call, from the Pearl+ verifier's split of its accepted verdict's credit call by call
(`out/pieces/pouw/verify.json`'s ``uncredited``, or ``--uncredited F``, such a split written beside the run's passes
by `served_passes.py uncredited`, for a record whose verify document holds none). Each call's real MACs are the MACs its credited cells
hold at |K| and the cells the band rule leaves out (excluded A rows, excluded B rows' column blocks, voluntary rows'
bands and blocks, and the side channels' k − |K|); its credited cells' MACs are the deduction's terms (ε's, that of
sampled tiles over the inert-work bound, and ι's, the inert work: ρ = 1/400 of what ε's deduction leaves) and its
credit, as exact fractions. The causes
over the model are those, the LM head and attention (which Pearl+ doesn't credit), the linears outside the passes, and
the floor of the credit; they sum to the give-up exactly. The split is the answer's or it is refused (exit 2): its
commitment and its passes' records roots are the answer's, each call's terms sum as above, and its calls' credited
cells' MACs are the answer's ``macs`` and their credit, floored, its ``credited``. An answer held to its class's fixed
load publishes its work in the grid form instead (``give_up`` among it: Pearl+'s `verify_run.grid_work`): its ``macs``
are the passes' whole MACs, each call's real MACs less its side channels, and its ``credited`` the whole passes' credit
less ``give_up``, the give-up padded up to H1's window grid. So there the split's whole MACs are the answer's ``macs``,
its credit, floored, is at least the answer's ``credited`` and at most that plus ``give_up``, and what the padding
takes off past the split's credit is the cause ``grid``. Where the passes hold the LM head's
calls, the LM head is no cause of its own: its calls are calls like the linears', and what the head computes outside
the passes is the cause ``lm_head_outside``, as the linears' is ``outside``. The ε deduction is the run's, at its
sample: the verifier takes it off the records whose cells hold the most MACs a ticket, so the calls it lands on carry it
for the whole run. ``--top N`` (default 10) calls are printed, by give-up.

It exits 0 with the number printed, 1 with none (the answer isn't an accept, the record holds no Lean verdict or one
that isn't the answer's, or the run left no answer: the kit prints ``compute: no number: WHY``, under it the coins as
the record states them (`stated`), and writes the result with no value and why) and 2 when the command line, the run or its policy can't be read,
or the policy's load is not the benchmark's. A record
withdrawn whole, or in a part that holds a file read here, is refused before any of it is fetched; a withdrawal of
other files of it is printed beside the number. Under `research run` the printed document is also `$OUT/compute.json`;
there the evidence store is read with the run's `--store-read` key.

Through the harness kit's writer (`verity_bench.kit`) it also writes the run directory's `result.json` under `research run`,
and `$OUT/result.json` outside one where `OUT` is set, with the
fields every resource's benchmark reports (`amounts`), each stamped with this run and its commit and computed from the
recorded run: the metric (the printed number, its unit and target), the two shares it is the product of (`CREDITABLE`,
`OF_CREDITABLE`: the model's MACs Pearl+ can credit at the load, and the share of those credited, each printed too),
the verified MACs (the credit), the capacity and
the free MACs at each data-sheet reading, and the error beside them (the answer's δ_s, sampling's alone). The metric is
named by its id in `metrics.toml` beside this file (`METRIC`), where the console reads it; the rest are every
resource's names: `verified`, `capacity`, `free` and `error`. A reading is a clock of a
registered device's card file (`clock.tensor`, `clock.sm`; `verity.device.load_card`), capacity the registered devices'
peak e4m3 MACs a second at that clock times the run's span (`capacity`), and free capacity less verified. A recorded
run's record holds no registration of its hardware and no span the verifier holds its passes to, so capacity and free
are written as one entry each, with no value and why. A made-up record's entries are marked modeled and made up
(`kit.report`'s ``made_up``), with a number or without.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tomllib
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Any, NamedTuple

from verity.device import Site, load_card, rate
from verity_bench import kit
from verity_bench.kit_report import RUN_ID

__all__ = ["ABSENT", "ANSWER", "CREDITABLE", "OF_CREDITABLE", "RAN", "SHARE", "DELTA_S", "DEVELOPMENT", "JOB", "LEAN", "LEAN_CHECK", "METRIC", "MODELS", "NO_SITE", "NO_SPAN", "OPTIONAL",
           "POLICY",
           "RECORDS", "RESTS_ON", "ROWS", "SAMPLING_ALONE", "TERMS", "UNIT", "VERDICTS", "VERIFY", "EvidenceStore", "Shape", "Unreadable",
           "amounts", "capacity", "coins", "from_dir", "from_run", "main", "model_macs", "uncredited_words"]

UNIT = "share of the served model's MACs that Pearl+ credits, over the tokens served"
#: the unit of the two shares the number is the product of (`CREDITABLE`, `OF_CREDITABLE`)
SHARE = "share of multiply-adds"
#: the number's id in metrics.toml beside this file, the name result.json records it under
METRIC = "compute-accounting/verified-compute-share"
#: the metrics of the two shares the number is the product of: the model's MACs Pearl+ can credit at the load (its
#: linears, and its LM head where the passes hold the head's calls), and the share of those its verifier credits
CREDITABLE, OF_CREDITABLE = "compute-accounting/creditable-share", "compute-accounting/credited-of-creditable"
#: the share the number is driven to
TARGET = Fraction(97, 100)
#: the load the number is of, which a record's policy must register (`fixed_load`)
LOAD = Path(__file__).with_name("load.toml")
#: the load's keys, the policy's names for them
LOAD_KEYS = ("model", "revision", "slots", "prompt", "outputs")
PROTOCOL = "verity/pouw/pearl-plus/v1"
#: the files of a run this command reads
ANSWER = "out/pieces/pouw/pearlplus-answer.json"
LEAN = "out/pieces/pouw/pearlplus-lean.json"
LEAN_CHECK = "out/pieces/pouw/pearlplus-lean-check.json"
POLICY = "out/inference-only-policy.json"
VERDICTS = "out/record/verdicts.json"
#: the Pearl+ piece's own document (`served_verdicts.py pearlplus`), read for its ``coin_source`` alone
DEVELOPMENT = "out/pieces/pouw/development.json"
#: the launcher's statement of the coins and parameters it ran with (`serve.sh`), read for its ``challenger`` and
#: ``development`` alone
RAN = "out/compute-run.json"
#: the run's own record of itself: a record directory's run id and commit, and where a record is made up, ``made_up``
JOB = "job.json"
RECORDS = [ANSWER, LEAN, LEAN_CHECK, POLICY, VERDICTS, DEVELOPMENT, RAN, JOB]
#: the files of `RECORDS` a withdrawal of which leaves the record readable: the file is then neither fetched nor read
OPTIONAL = frozenset({DEVELOPMENT, RAN, JOB})
#: the launcher's ``challenger``, in the kit's words for who drew the coins (`kit.COINS`): never the deployed one's
CHALLENGER = {"local": "local challenger", "in-process": "in-process"}
#: why a record whose launcher says a deployed challenger or deployed parameters is refused: the person's own run
#: writes every file of the record, so no file of it can show that the auditor's challenger drew the coins
NO_DEPLOYED = "this command does not yet read a deployed run: it has no check of the challenger's key"
#: the verifier's document, read by `--per-call` alone
VERIFY = "out/pieces/pouw/verify.json"
#: the files a record directory must hold, and what each is
REQUIRED = {ANSWER: "the Pearl+ verifier's answer", POLICY: "the run's inference-only policy"}
#: the causes of the per-call split's give-up, in the order printed
CAUSES = {"lm_head": "LM head, which Pearl+ doesn't credit", "lm_head_outside": "LM head calls outside the passes",
          "attention": "attention, which Pearl+ doesn't credit",
          "outside": "linears outside the passes", "excluded_a": "excluded A rows",
          "excluded_b": "excluded B rows' column blocks", "voluntary": "voluntary rows' bands and blocks",
          "kept_width": "side channels (k beyond |K|)", "epsilon_macs": "ε's deduction",
          "over_bound_macs": "sampled tiles over the inert-work bound", "rho_macs": "ι, inert work: 1/400 of what ε leaves",
          "floor": "the credit's floor", "grid": "the give-up padded up to the window grid"}
#: a call's terms the split states, beside its MACs
ROWS = ("excluded_a", "excluded_b", "voluntary", "kept_width")
TERMS = ("epsilon_macs", "over_bound_macs", "rho_macs", "credited_macs")
ABSENT = ("absent: a Pearl+ answer holds its passes to no seconds, and the seconds the run served in stay on the "
          "developer's machine")
#: what the answer's ε and δ leave out, printed beside them
SAMPLING_ALONE = ("sampling's alone: they bound what checking a random sample of tiles can miss, not a failure of the "
                  "conjecture or the hash the credit rests on")
#: who drew the answer's sample (`profile.source_from`), as printed
SOURCE = {"node": "the serving machine", "challenger": "a challenger"}
#: what the credit rests on, printed under the answer's verdict
RESTS_ON = ("  the credit rests on Pearl's conjecture TT_OUT rev1, which this repository's red-team rates refuted: that "
            "a credited MAC was computed is an open premise, not a proved guarantee")
#: the data sheet's readings of a device's peak, each a clock of its card file (`clock.tensor`, `clock.sm`)
READINGS = ("tensor", "sm")
#: the operand format of the MACs Pearl+ credits
FORMAT = "e4m3"
NO_SITE = "the run's record holds no registration of its hardware: no site entry naming its devices and their sheets"
NO_SPAN = ABSENT.removeprefix("absent: ")
DELTA_S = ("probability that a sample that accepts misses more than the share ε_s of the credited MACs: the answer's δ_s, "
           "sampling's alone (not a failure of the conjecture or the hash the credit rests on)")
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

    def __init__(self, store: Any, pull: Callable, error: type[Exception], withdrawn: str) -> None:
        self.store, self.pull, self.error, self.withdrawn = store, pull, error, withdrawn

    def attempt(self, run: str) -> dict | None:
        try:
            return self.store.get_attempt(run)
        except (self.error, ValueError) as e:
            raise Unreadable(f"the evidence store can't be read: {e}") from e

    def files(self, run: str, paths: list[str]) -> tuple[Path | None, list[str]]:
        """A directory holding the files of `run`'s record at `paths`, each a file's whole path in the run (None when
        the record holds none of them), and what was withdrawn of the rest of the record, each as its label says it.
        A record withdrawn whole, or in a part that holds one of `paths` other than `OPTIONAL`'s, is refused before any
        of it is fetched; an `OPTIONAL` file withdrawn is left out of the fetch, as if the record held none."""
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
                hit = {f for f in paths for p in prefixes if _under(f, p)}
                if not partial or hit - OPTIONAL:
                    raise Unreadable(f"{run} was withdrawn ({said}): nothing of it is fetched")
                paths = [f for f in paths if f not in hit]
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
        raise Unreadable(f"the evidence store's client isn't installed here ({e}): give --from the run's record "
                         "directory, or install research (uv sync --all-packages)") from e
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
    credit that is no test instance's, and the record holds the Lean verifier's verdict and its comparison with the
    answer, both the answer's. `epsilon_s` and `delta_s` are the answer's `public.work.epsilon` and `profile.delta`, its sample's."""
    try:
        answer, lean, check = (_held(tree, p) for p in (ANSWER, LEAN, LEAN_CHECK))
        if answer is None:
            raise NoNumber("the run left no Pearl+ answer")
        for path, held in ((LEAN, lean), (LEAN_CHECK, check)):
            if held is None:
                raise NoNumber(f"the record holds no {path}, so no Lean verdict: the answer's credit is no number "
                               "without it")
        protocol, verdict, public = answer["protocol"], answer["verdict"], answer.get("public") or {}
        work = public.get("work") or {}
        credited, passes = work.get("credited"), work.get("macs")
        mine = ("accept", answer.get("commitment"), credited)
        theirs = (lean["verdict"], lean.get("commitment"), (lean.get("work") or {}).get("credited"))
        differ = check["differ"]
    except (OSError, KeyError, TypeError, ValueError, AttributeError) as e:
        raise NoNumber(f"the run's Pearl+ verdicts don't read as verdicts ({type(e).__name__}: {e})") from e
    if protocol != PROTOCOL:
        raise NoNumber(f"the answer is {protocol}'s, not {PROTOCOL}'s")
    if verdict != "accept":
        raise NoNumber(f"Pearl+'s answer is {verdict}, not accept ({answer.get('reason') or 'no reason recorded'})")
    if (answer.get("profile") or {}).get("test_instance"):
        raise NoNumber("Pearl+'s answer is a test instance's (its profile's test_instance): its passes are not the served "
                       "instance's, and it credits nothing")
    if not _count(credited) or passes is not None and not _count(passes):
        raise NoNumber(f"Pearl+'s answer accepts and holds no credit in MACs (credited {credited!r}, macs {passes!r})")
    for name, lean_s, answer_s in zip(("verdict", "run commitment", "credit"), theirs, mine):
        if lean_s != answer_s:
            raise NoNumber(f"the Lean verifier's {name} is {str(lean_s)[:16]} and the answer's {str(answer_s)[:16]}")
    if differ:
        raise NoNumber(f"the record's comparison of the Lean verifier's verdict with the answer differs on {differ}")
    return {"credited": credited, "passes": passes, "scheme": public.get("scheme"), "epsilon_s": work.get("epsilon"),
            "delta_s": (answer.get("profile") or {}).get("delta"), "lean": True}


def coins(tree: Path) -> dict:
    """Which coins and parameters the record under `tree` ran with, as it states them, never refused: the source of the
    answer's sample (`profile.source_from`, ``challenger`` or ``node``), of the passes' salt coins (the piece's
    ``coin_source``, ``challenger`` or ``in-process``), whether the Lean verifier ran as of record (``--of-record`` in
    the comparison's ``argv``, which `served_passes.py lean` passes only on live coins), the answer's draws and δ, each
    None where the record doesn't state it, and ``development`` where the record holds PoUW's development piece (whose
    credit is the piece's, not a row of record) or any of the three is not live. ``challenger`` names a challenger
    session, the auditor's or a local one: the record doesn't say whose. Read after `credit`, which has read the answer
    and the comparison; `Unreadable` where the record holds a development piece that doesn't read as JSON."""
    profile = (_held(tree, ANSWER) or {}).get("profile") or {}
    argv = (_held(tree, LEAN_CHECK) or {}).get("argv")
    lean = None if not isinstance(argv, list) else "of record" if "--of-record" in argv else "development"
    piece, ran = _optional(tree, DEVELOPMENT), _optional(tree, RAN)
    sample, salt = profile.get("source_from"), piece.get("coin_source") if isinstance(piece, dict) else None
    ran = ran if isinstance(ran, dict) else {}
    return {"sample": sample, "salt": salt, "lean": lean, "draws": profile.get("draws"), "delta": profile.get("delta"),
            "development": piece is not None or not (sample == salt == "challenger" and lean == "of record"),
            "challenger": ran.get("challenger"), "parameters": ran.get("development")}


def _optional(tree: Path, path: str) -> Any:
    """The JSON an optional file of the record under `tree` holds; None where it holds no such file, and `Unreadable`,
    naming it, where it holds one that doesn't read as JSON."""
    if not (tree / path).is_file():
        return None
    try:
        return json.loads((tree / path).read_text())
    except ValueError as e:
        raise Unreadable(f"the record's {path} doesn't read as JSON ({e})") from e


def ran_with(doc: dict) -> tuple[str, dict] | None:
    """The kit's `ran with` line of a document (`kit.ran_with`), from what its record states (`coins`): who drew the
    coins is the launcher's ``challenger`` (`CHALLENGER`, `RAN`), else ``in-process`` where the sample is the node's or
    the salt coins an in-process challenger's, and otherwise not stated (None): a challenger sample says no more than
    that some challenger drew it. The parameters are always ``development``, so the line is never an audit's.
    `Unreadable` where the launcher's word is none of `CHALLENGER`'s, and `NO_DEPLOYED` where it says the deployed
    challenger or ``development`` false: the person's own run writes every file of the record, so files that agree
    with each other prove nothing, and reading a deployed run waits on a check of the challenger's signature against a
    pinned key."""
    c = doc["coins"]
    # of record here means only that the Lean took its of-record path: against a local challenger, that challenger's key
    lean = c["lean"] + " against the local challenger's key" if c["lean"] == "of record" and c["challenger"] == "local" \
        else c["lean"]
    detail = ", ".join(f"{said} {v}" if v is not None else f"{unsaid} not stated" for said, unsaid, v in (
        ("draws", "draws", c["draws"]), ("delta", "delta", c["delta"]), ("scheme", "scheme", doc["scheme"]),
        ("sample from", "sample source", SOURCE.get(c["sample"], c["sample"])),
        ("salt coins from", "salt coins", c["salt"]),
        ("Lean verdict", "Lean verdict", lean)))
    if c["challenger"] == "deployed" or c["parameters"] is False:
        raise Unreadable(f"{NO_DEPLOYED} (the record's {RAN} says "
                         + ("the deployed challenger" if c["challenger"] == "deployed" else "development false") + ")")
    if c["challenger"] is not None and c["challenger"] not in CHALLENGER:
        raise Unreadable(f"the record's {RAN} names the challenger {c['challenger']!r}, not one of "
                         f"{', '.join(CHALLENGER)}")
    coins = CHALLENGER.get(c["challenger"]) or ("in-process" if c["sample"] == "node" or c["salt"] == "in-process"
                                                else None)
    if coins is None:
        return None
    return kit.ran_with(coins, "development", detail)


def fixed_load() -> dict:
    """The load the benchmark fixes (`LOAD`): the served model, its revision and the class's slots, prompt tokens and
    output tokens a request."""
    with LOAD.open("rb") as f:
        held = tomllib.load(f)
    return {k: held[k] for k in LOAD_KEYS}


def _load_words(d: dict) -> str:
    return f"{d['model']} at revision {d['revision']}, class {d['slots']},{d['prompt']},{d['outputs']}"


def load(tree: Path) -> dict:
    """The load the run's inference-only policy registers, from the record under `tree`: the model, the class and the
    MACs the model computes on it. `Unreadable` unless the policy names a class and a model of `MODELS`, and its own
    count of the class's MACs a slot, where it has one, is `model_macs`'s."""
    try:
        policy = _held(tree, POLICY)
        if policy is None:
            raise Unreadable("the run's record holds no inference-only policy, so no class to count the model's MACs at")
        cls, model, revision = policy["class"], policy["serve"]["model"], policy["serve"].get("revision")
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
    fixed = fixed_load()
    if (theirs := {"model": model, "revision": revision, "slots": slots, "prompt": prompt, "outputs": outputs}) != fixed:
        raise Unreadable(f"the record's policy registers {_load_words(theirs)} and the benchmark's load is "
                         f"{_load_words(fixed)} ({LOAD.name}; class is slots, prompt tokens and output tokens a "
                         "request): the number is of that load alone")
    return {"model": model, "revision": revision, "class": {"slots": slots, "prompt": prompt, "outputs": outputs}, "macs": macs,
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
    names = {(d, d + 2 * shape.kv_heads * head_dim): "qkv", (d, d): "o", (d, 2 * shape.mlp): "gate_up", (shape.mlp, d): "down",
             (d, shape.vocab): "lm_head"}
    return names.get((k, n), f"{k}x{n}")


def head_calls(tree: Path, given: Path | None, shape: Shape) -> list[dict]:
    """The LM head's calls in the run's passes: the calls of the head's shape (k = d, n = V) that the verifier's per-call
    split names, in its verify document or in `given`, where the split is over the answer's run commitment; none where
    the record holds no split. Unreadable files are `split`'s to refuse."""
    try:
        held = json.loads(given.read_text()) if given is not None else _held(tree, VERIFY)
        if held is None or "uncredited" not in held or held["commitment"] != _held(tree, ANSWER)["commitment"]:
            return []
        return [c for c in held["uncredited"]["calls"] if (c["k"], c["n"]) == (shape.hidden, shape.vocab)]
    except (OSError, KeyError, TypeError, ValueError, AttributeError) as e:
        raise Unreadable(f"the per-call split doesn't read as one ({type(e).__name__}: {e})") from e


def uncredited_words(c: dict) -> int:
    """The output words of a per-call split's call `c` that hold at least one MAC its credit leaves out, counted as the
    words touched, not as its uncredited MACs over k. A word in an excluded A row, an excluded B row's column block or a
    voluntary row's band or block holds no credited MAC, so those count their cells, the real words less the credited
    cells (`checked_macs` over |K|). A call that keeps fewer than its k channels (`kept_width`) leaves the side
    channels' MACs out of every real word, so all its real words count, more than its uncredited MACs over k. Where
    only whole rows are excluded the two counts agree."""
    cells, rest = divmod(c["checked_macs"], c["k_kept"])
    if rest:
        raise Unreadable(f"the per-call split's call {c['call']} holds checked MACs that are no whole number of cells at "
                         f"its kept width")
    return c["real_rows"] * c["n"] - (0 if c["uncredited_macs"]["kept_width"] else cells)


def split(tree: Path, given: Path | None, macs: dict[str, int], credited: int, shape: Shape, top: int) -> dict:
    """The give-up of the record under `tree` by cause and by call (module docstring), and the output words that hold
    any of it over every call (`uncredited_words`), from the verifier's split in its verify document or in `given`,
    such a split in a file of its own. `Unreadable` unless the split is the answer's. `macs` are the model's MACs at the class and `credited` the answer's credit."""
    try:
        held = json.loads(given.read_text()) if given is not None else _held(tree, VERIFY)
        if held is None or (given is None and "uncredited" not in held):
            raise Unreadable("the record holds no per-call give-up: its verify document has no `uncredited`, which "
                             "`served_passes.py uncredited` writes from the run's passes and verify.json, and "
                             "--uncredited F reads")
        answer = _held(tree, ANSWER)
        work, passes = answer["public"]["work"], answer["public"].get("passes")
        got, records = held["uncredited"], held.get("records")
        calls, totals = got["calls"], got["totals"]
        sums = dict.fromkeys(("macs", "checked_macs", *ROWS, *TERMS), Fraction(0))
        words = 0
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
            words += uncredited_words(c)
        stated = {x: Fraction(totals[x]) for x in sums}
        heads = [c for c in calls if (c["k"], c["n"]) == (shape.hidden, shape.vocab)]
        in_head = sum(c["macs"] for c in heads)
    except (OSError, KeyError, TypeError, ValueError, AttributeError, ZeroDivisionError) as e:
        raise Unreadable(f"the per-call split doesn't read as one ({type(e).__name__}: {e})") from e
    if held["commitment"] != answer["commitment"]:
        raise Unreadable("the per-call split is of another run commitment than the answer's")
    if records is not None and passes is not None and records != [p["records"] for p in passes]:
        raise Unreadable("the per-call split's passes' records are not the answer's")
    if stated != sums or totals.get("credited") != int(sums["credited_macs"]):
        raise Unreadable("the per-call split's totals are not its calls' sums")
    exact = int(sums["credited_macs"])
    if "give_up" in work:
        whole = sums["checked_macs"] + sum(sums[x] for x in ROWS[:3])
        if whole != work["macs"] or not credited <= exact <= credited + work["give_up"]:
            raise Unreadable(f"the per-call split's calls credit {exact:,} of {int(whole):,} whole MACs and the answer's "
                             f"grid form {credited:,} of {work['macs']:,} less a give-up of {work['give_up']:,}")
    elif (sums["checked_macs"], exact) != (work["macs"], credited):
        raise Unreadable(f"the per-call split's calls credit {exact:,} of "
                         f"{int(sums['checked_macs']):,} MACs and the answer {credited:,} of {work['macs']:,}")
    if sums["macs"] - in_head > macs["linears"]:
        raise Unreadable(f"the per-call split's calls hold {int(sums['macs'] - in_head):,} real MACs, more than the "
                         f"{macs['linears']:,} of the class's linears")
    if in_head > macs["lm_head"]:
        raise Unreadable(f"the per-call split's LM head calls hold {in_head:,} real MACs, more than the "
                         f"{macs['lm_head']:,} of the class's LM head")
    gave = {("lm_head_outside" if heads else "lm_head"): macs["lm_head"] - in_head, "attention": macs["attention"],
            "outside": macs["linears"] - (sums["macs"] - in_head),
            **{x: sums[x] for x in (*ROWS, *TERMS[:3])}, "floor": sums["credited_macs"] - credited}
    if "give_up" in work:
        gave |= {"floor": sums["credited_macs"] - exact, "grid": exact - credited}
    give_up = sum(macs.values()) - credited
    assert sum(gave.values()) == give_up, (gave, give_up)

    def lost(c: dict) -> Fraction:
        return c["macs"] - Fraction(c["credited_macs"])

    worst = sorted(calls, key=lost, reverse=True)[:top]
    return {"source": str(given) if given is not None else VERIFY, "give_up": give_up, "calls": len(calls),
            "uncredited_words": words, "causes": [{"cause": x, "macs": str(v), "share": float(v / give_up) if give_up else 0.0}
                       for x, v in gave.items()],
            "top": [{"pass": c["pass"], "call": c["call"], "weight": c["weight"], "linear": linear(shape, c["k"], c["n"]),
                     "m": c["m"], "k": c["k"], "n": c["n"], "real_rows": c["real_rows"], "rows": c["rows"],
                     "give_up": str(lost(c)), "causes": {x: str(v) for x, v in
                                                          (*((x, c["uncredited_macs"][x]) for x in ROWS),
                                                           *((x, c[x]) for x in TERMS[:3])) if Fraction(v)}}
                    for c in worst]}


def document(tree: Path, commit: str | None, run: str | None, withdrawn: list[str], given: Path | None = None) -> dict:
    """The printed document of the record under `tree`: the number, its unit, the commit the run ran from and the run,
    then what the number is made of. `withdrawn` is what was withdrawn of the rest of the record, and `given` a per-call
    split in a file of its own (`head_calls`). Where the passes hold the LM head's calls it also holds ``lm_head``: their
    count and the MACs Pearl+ can credit at the class, the linears' and the LM head's."""
    c, served, record = credit(tree), load(tree), row(tree)
    macs = served["macs"]
    heads = head_calls(tree, given, MODELS[served["model"]])
    creditable = macs["linears"] + macs["lm_head"] * bool(heads)
    most = max(c["credited"], c["passes"] or 0)
    if most > creditable:
        of = "the linears and the LM head" if heads else "the linears"
        raise Unreadable(f"Pearl+'s answer holds {most:,} MACs, more than the {creditable:,} of {of} of "
                         f"the policy's class: its passes are not one load of the class")
    total = sum(macs.values())
    head = {"lm_head": {"calls": len(heads), "creditable_macs": creditable,
                        "creditable_share": creditable / total}} if heads else {}
    return {"value": c["credited"] / total, "unit": UNIT, "commit": commit, "run": run, "target": float(TARGET),
            "met": Fraction(c["credited"], total) >= TARGET, "credited_macs": c["credited"], "model_macs": total,
            "macs": macs, "linears_share": macs["linears"] / total, "passes_macs": c["passes"], "scheme": c["scheme"],
            "epsilon_s": c["epsilon_s"], "delta_s": c["delta_s"], "lean": c["lean"], "coins": coins(tree), "record": record,
            "withdrawn": withdrawn,
            "model": served["model"], "revision": served["revision"], "class": served["class"], "policy_agrees": served["policy_agrees"],
            "tokens": served["tokens"], "tokens_from": "class", "wall_s": None, "capacity_share": None, **head,
            "made_up_says": made_up(tree)}


def made_up(tree: Path) -> str | None:
    """What the record under `tree` says of itself where it is made up, not a run's: its `job.json`'s ``made_up``, which
    a record written by hand (the tests' fixture) states and a run's never does; None where it says nothing."""
    job = _optional(tree, JOB)
    said = job.get("made_up") if isinstance(job, dict) else None
    return str(said) if said else None


def made_up_text(said: str) -> str:
    """The printed line of what a made-up record says of itself, under the kit's line that it is made up."""
    return f"  what the record says of itself: {said}"


def capacity(site: Site, span_s: Fraction) -> dict[str, int]:
    """The registered hardware's MACs over `span_s` seconds at each data-sheet reading (`READINGS`): each registered
    device's peak `FORMAT` MACs a second at its card's clock of that reading (`verity.device.rate` on the card
    file of its sheet's SKU), summed over the devices, times the span, rounded up. `Unreadable` where the site registers
    no device, or a device names no registered sheet or an SKU with no card file."""
    if not site.units:
        raise Unreadable("the registered hardware holds no device")
    total = dict.fromkeys(READINGS, Fraction(0))
    for d in site.units:
        if (sheet := site.sheet(d)) is None:
            raise Unreadable(f"the registered device {d} names no registered sheet")
        try:
            card = load_card(sheet.sku)
        except KeyError as e:
            raise Unreadable(f"the registered device {d}'s SKU has no card file: {e}") from e
        for r in READINGS:
            total[r] += rate(card, fmt=FORMAT, mhz=card["clock"][r]["value"]) * span_s
    return {r: math.ceil(v) for r, v in total.items()}


def amounts(doc: dict, site: Site | None = None, span_s: Fraction | None = None) -> list[dict]:
    """The result a document `doc` gives in the fields every resource's benchmark reports, as the kit writer's
    measurements: the metric (`METRIC`, the printed number in its unit, beside its target), the two shares it is the
    product of (`CREDITABLE`, `OF_CREDITABLE`, the printed line's "Pearl+ credits ..."), the verified amount
    (`verified`, the credited MACs), at each data-sheet reading the capacity (`capacity at clock.tensor`, ..., from the
    registered hardware `site` over the run's span `span_s`) and the free amount (`free at ...`, capacity less verified),
    and the probability the verified amount holds except with (`error`, the answer's δ_s, which its `inputs` name).
    Where `site` or `span_s` is None, capacity and free are one entry each, with no value and why. Each is computed from
    the recorded run."""
    inputs = [doc["run"]] if doc["run"] else []
    verified = doc["credited_macs"]
    head = doc.get("lm_head")
    can = head["creditable_macs"] if head else doc["macs"]["linears"]
    out = [{"name": METRIC, "value": doc["value"], "unit": doc["unit"], "target": doc["target"],
            "verdict": "at or above target" if doc["met"] else "below target", "inputs": inputs},
           {"name": CREDITABLE, "value": can / doc["model_macs"], "unit": SHARE, "inputs": inputs},
           {"name": OF_CREDITABLE, "value": verified / can, "unit": SHARE, "inputs": inputs},
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


def from_run(store: EvidenceStore, run: str, per_call: bool = False, given: Path | None = None,
             top: int = 10) -> dict:
    """The printed document of recorded run `run`; with `per_call`, its give-up by cause and by call too (`split`)."""
    attempt = store.attempt(run)
    if attempt is None:
        raise Unreadable(f"{run} is not in the evidence store this machine reads")
    tree, withdrawn = store.files(run, RECORDS + [VERIFY] * (given is None))
    try:
        if tree is None or not (tree / ANSWER).is_file():
            raise NoNumber(f"{run} left no Pearl+ answer (it ended rc {(attempt.get('status') or {}).get('rc')})")
        return _with_split(document(tree, commit_of(attempt), run, withdrawn, given), tree, per_call, given, top)
    except NoNumber as e:
        e.tree = tree
        raise


def stated(tree: Path | None) -> tuple[tuple[str, dict] | None, str]:
    """On a refusal, what the record under `tree` states of its coins: the `ran with` pair where it says who drew them
    (`ran_with`), else None with the line naming what it does state (`coins_text`); (None, "") where nothing of it
    reads."""
    if tree is None:
        return None, ""
    try:
        c = coins(tree)
        said = ran_with({"coins": c, "scheme": ((_held(tree, ANSWER) or {}).get("public") or {}).get("scheme")})
    except (Unreadable, OSError, ValueError, AttributeError, TypeError):
        return None, ""
    return said, "" if said else coins_text(c)


def from_dir(record: Path, per_call: bool = False, given: Path | None = None, top: int = 10) -> dict:
    """The printed document of the run record in the directory `record` (module docstring), through no store: the run
    is the `run.id` its `job.json` states, else the directory's name, and the commit the one its `store.conditions`
    state, else None. `Unreadable` unless it holds every file of `REQUIRED`."""
    if not record.is_dir():
        raise Unreadable(f"{record} is not a directory: --from names a run's directory, the one holding out/")
    for path, what in REQUIRED.items():
        if not (record / path).is_file():
            moved = (record / Path(path).relative_to("out")).is_file()
            raise Unreadable(f"the record directory {record} holds no {path} ({what})"
                             + (": it is a run's out/ directory; name the directory holding it" if moved else ""))
    try:
        job = _held(record, JOB) or {}
        run = (job.get("run") or {}).get("id") or record.resolve().name
        conds = (job.get("store") or {}).get("conditions") or {}
        commit = commit_of({"source": {"commit": conds.get("source.commit"), "dirty": conds.get("source.dirty")}})
    except (OSError, ValueError, AttributeError) as e:
        raise Unreadable(f"the record's {JOB} doesn't read as the run's ({type(e).__name__}: {e})") from e
    return _with_split(document(record, commit, str(run), [], given), record, per_call, given, top)


def _with_split(doc: dict, tree: Path, per_call: bool, given: Path | None, top: int) -> dict:
    if per_call:
        doc["per_call"] = split(tree, given, doc["macs"], doc["credited_macs"], MODELS[doc["model"]], top)
    return doc


def _m(v: str | int | Fraction) -> str:
    return f"{round(Fraction(v)):,}"


def per_call_text(got: dict, model_macs: int) -> list[str]:
    """The printed lines of a document's `per_call`: its give-up by cause, then its top calls."""
    lines = [(f"  per call: the model gives up {got['give_up']:,} of its MACs ({100 * got['give_up'] / model_macs:.2f}%); "
              f"by cause, from the verifier's split of its {got['calls']} calls ({got['source']}):")]
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


def coins_text(c: dict) -> str:
    """The printed line of a document's `coins` where the record doesn't say who drew them (`ran_with` None): which
    coins and parameters it does state."""
    stated = "not stated in the record"
    return (f"  coins as the record states them, which doesn't say whose challenger: "
            + (f"sample from {SOURCE.get(c['sample'], c['sample'])}, " if c["sample"] else f"sample source: {stated}, ")
            + (f"salt coins from {c['salt']}" if c["salt"] else f"salt coins: {stated}")
            + f"; Lean verifier run {'as ' + c['lean'] if c['lean'] else stated}"
            + f"; parameters: draws {c['draws'] if c['draws'] is not None else stated}, "
            + f"delta {c['delta'] if c['delta'] is not None else stated}"
            + (", development" if c["development"] else ""))


def text(doc: dict) -> str:
    """The printed lines of a document: the number, what accepted it, and what it is made of."""
    macs, cls, tokens, record = doc["macs"], doc["class"], doc["tokens"], doc["record"]
    lines = [(f"compute: {100 * doc['value']:.2f}% of the served model's MACs ({doc['credited_macs']:,} of "
              f"{doc['model_macs']:,}), {'at or above' if doc['met'] else 'below'} the {100 * doc['target']:g}% target. "
              "This is what the Pearl+ verifier's own answer credits, not PoUW's verdict in the run's record of "
              "verdicts")]
    if doc["made_up_says"]:
        lines.append(made_up_text(doc["made_up_says"]))
    if ran_with(doc) is None:
        lines.append(coins_text(doc["coins"]))
    at = ", ".join(f"{k} {doc[name]}" for k, name in (("ε_s", "epsilon_s"), ("δ_s", "delta_s")) if doc[name] is not None)
    lines.append(f"  {doc['run'] or 'no recorded run'}, commit {doc['commit'] or 'unknown'}: Pearl+'s answer accepts"
                 + (f" (scheme {doc['scheme']})" if doc["scheme"] else "")
                 + (f" at {at} ({SAMPLING_ALONE})" if at else "")
                 + ", and the Lean verifier's verdict is the same")
    lines.append(RESTS_ON)
    lines.append("  PoUW's verdict in the run's record of verdicts " + ("accepts too" if record["verdict"] == "accept" else
                 (f"is `{record['verdict']}`, not an accept" if record["verdict"] else "is absent")
                 + ": PoUW credits nothing of this run, and the number above is the Pearl+ verifier's answer alone"
                 + (f" (the record's reason for PoUW's verdict: {record['why']})" if record["why"] else "")))
    if doc["withdrawn"]:
        lines.append(f"  part of the run's record was withdrawn, none of it a file read here: {'; '.join(doc['withdrawn'])}")
    lines.append(f"  {doc['model']} at revision {doc['revision']}, {cls['slots']} slots of {cls['prompt']} prompt and "
                 f"{cls['outputs']} output tokens, the benchmark's load ({LOAD.name}): linears {macs['linears']:,}, attention {macs['attention']:,} and LM head {macs['lm_head']:,} "
                 f"MACs" + (", which is the policy's own count of its class" if doc["policy_agrees"] else ""))
    head = doc.get("lm_head")
    can, share = (head["creditable_macs"], head["creditable_share"]) if head else (macs["linears"], doc["linears_share"])
    lines.append(("  Pearl+ credits the linears and the LM head, whose calls the passes hold, " if head
                  else "  Pearl+ credits linears alone, ")
                 + f"{100 * share:.2f}% of the model's MACs at this class: "
                 + (f"its passes hold {doc['passes_macs']:,} of them and " if doc["passes_macs"] is not None else "")
                 + f"its credit is {100 * doc['credited_macs'] / can:.2f}% of them all")
    lines.append(f"  tokens, the class's since the record holds no count of those served: {tokens['prompt']:,} prompt "
                 f"and {tokens['output']:,} output, over {tokens['positions']:,} positions")
    lines.append(f"  capacity share: {ABSENT}")
    if "per_call" in doc:
        lines += per_call_text(doc["per_call"], doc["model_macs"])
    return "\n".join(lines)


def main(argv: list[str] | None = None, *, store_of: Callable[[], EvidenceStore] = opened) -> int:
    """The command: 0 with the number printed, 1 with none, 2 when the command line, the run or its policy can't be
    read. `--from` takes a record's directory, read with nothing of research's (`kit.record_dir`), or a run id
    (`kit_report.RUN_ID`) that names no directory, fetched through `store_of`, the evidence store, which honors
    withdrawals of parts of a record (`EvidenceStore.files`)."""
    ap = argparse.ArgumentParser(prog="verity bench compute", description=__doc__.split("\n\n")[0])
    ap.add_argument("--from", dest="source", metavar="DIR|RUN", required=True,
                    help="a run's record: its directory (the run's, holding out/ and job.json), read from disk alone, or "
                         "a run id of our evidence store, read through research's store client")
    ap.add_argument("--per-call", action="store_true",
                    help="also print where the model's MACs the credit leaves out went, by cause and by matmul call")
    ap.add_argument("--uncredited", type=Path, metavar="F",
                    help="with --per-call: a per-call give-up (`uncredited`) in a file of its own, for a record whose "
                         "verify document holds none, as `served_passes.py uncredited` writes it")
    ap.add_argument("--top", type=int, default=10, metavar="N", help="with --per-call: the calls printed (default 10)")
    a = ap.parse_args(argv)
    if (a.uncredited or a.top != 10) and not a.per_call:
        ap.error("--uncredited and --top go with --per-call")
    tree = None
    try:
        if RUN_ID.fullmatch(a.source) and not Path(a.source).is_dir():
            doc = from_run(store_of(), a.source, a.per_call, a.uncredited, a.top)
        else:
            tree = kit.record_dir(a.source)
            doc = from_dir(tree, a.per_call, a.uncredited, a.top)
        measured, said = amounts(doc), ran_with(doc)
    except kit.NoRecord as e:
        print(f"compute: {e}", file=sys.stderr)
        return 2
    except NoNumber as e:
        none = {"value": None, "unit": UNIT, "commit": None, "run": a.source, "target": float(TARGET)}
        try:
            up = made_up(tree) if tree else None
        except Unreadable as u:
            print(f"compute: {u}", file=sys.stderr)
            return 2
        said, line = stated(getattr(e, "tree", None) or tree)
        return kit.report("compute", "\n".join(x for x in (made_up_text(up) if up else "", line) if x),
                          none | {"made_up_says": up},
                          not_an_audit([{"name": METRIC, "unit": UNIT, "target": float(TARGET)}]), ran_with=said,
                          accepted=False, why=str(e), made_up=bool(up))
    except Unreadable as e:
        print(f"compute: {e}", file=sys.stderr)
        return 2
    return kit.report("compute", text(doc), doc, not_an_audit(measured), ran_with=said, accepted=True,
                      made_up=bool(doc["made_up_says"]))


def not_an_audit(measured: list[dict]) -> list[dict]:
    """`measured`, each saying in `result.json` that it is no audit's (``audit`` false), as `compute.json` does."""
    return [m | {"extra": {**m.get("extra", {}), "audit": False}} for m in measured]


if __name__ == "__main__":
    sys.exit(main())
