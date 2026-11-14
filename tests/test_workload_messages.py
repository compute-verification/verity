"""The messages between the workload and the prover are one stated list, and the tree holds to it.

**The list** is every protocol's declaration of its messages with the workload: the module ``workload`` of its
verifier's side (``verity/protocols/GROUP/NAME/verifier/workload.py``), its message types (``MESSAGES``, each a
``verity.channel.Schema``) and its channels between the workload and the prover (``CHANNELS``, each a
``verity.protocol.Channel``).  A message is declared once: one that serves more than one protocol sits in sampled
proofs' module, and the others' channels name it.  ``uv run tests/test_workload_messages.py`` prints the list, and so
does its first test.

**How the tree carries each message today** is ``TODAY``: a file, a record in a log, a function call or a command line,
with the files that write it and the files that read it, each by a line it holds.  A message whose writer or reader
moved fails here, naming the line of ``TODAY`` to change, so the list says where every message is.

**What else crosses** between the two is ``EXCEPTIONS``, one line each with its kind (``KINDS``).  It only shrinks: it
holds at most ``EXCEPTIONS_MAX`` lines, a line whose crossing is gone warns until it is deleted, and nothing crosses by
an import that no line names: every side of a protocol (``verity/protocols/GROUP/NAME/SIDE``) and the values' library
(``LIBRARIES``) that a Python file under ``adapters/`` imports is a line's ``calls``.
"""
from __future__ import annotations

import ast
import functools
import importlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity.channel import Schema
from verity.protocol import Channel

ROOT = Path(__file__).resolve().parents[1]
PROTOCOLS = "verity/protocols"
#: The declaration every other is held to: the two parties' names, and the messages more than one protocol reads.
SPOT = f"{PROTOCOLS}/proofs/spot/verifier/workload.py"
ADAPTERS = "adapters/"
#: What an adapter imports of ``verity`` besides a protocol's side that carries a message: the writer's library for the
#: values, whose name and place are the open entry of ``tests/tree.toml``.
LIBRARIES = ("verity/commitments/windows",)
V = "adapters/vllm/verity_vllm"
ONE_STAGE, PROVE_PATH = "benchmarks/one_stage", "benchmarks/prove_path"
LAUNCHER = f"{ONE_STAGE}/served_verdicts.sh"
DRIVER = f"{PROVE_PATH}/prove_path.py"
POUS_PIECE = "benchmarks/pous/served_piece.py"
CERTIFIER_PIECE = "benchmarks/network_traces/certifier_piece.py"
POUW_PROVER = f"{PROTOCOLS}/accounting/pouw/prover"


@dataclass(frozen=True)
class Line:
    """One line of the list: ``what`` it is in the tree today, the files that write it and the files that read it, each
    ``(file, a regular expression one of its lines holds)``, and what an adapter imports for it (``calls``)."""

    what: str
    writes: tuple[tuple[str, str], ...] = ()
    reads: tuple[tuple[str, str], ...] = ()
    calls: tuple[str, ...] = ()


def _all(pattern: str, *files: str) -> tuple[tuple[str, str], ...]:
    return tuple((f, pattern) for f in files)


#: Each declared message as the tree carries it today.  PoUW's, PoUS's and the certifier's pieces and the launcher are
#: the prover's programs; ``adapters/`` is the workload's side.
TODAY = {
    "Circuit": Line("a file: the Build of record (`predicted.json`, with its Programs), by `verity-vllm predict row`",
                    writes=((f"{V}/program/predict/aot.py", r"predicted\.json"),
                            (f"{V}/records/predicted_row.py", r'FILE = "predicted\.json"')),
                    reads=((LAUNCHER, r"\$PRED/predicted\.json"), (f"{ONE_STAGE}/served_compare.py", r'"--pred"'))),
    "Settings": Line("a file: `serve.json`",
                     writes=((f"{V}/pipeline/serve_openai.py", r"serve\.json"),),
                     reads=_all(r"serve\.json", DRIVER, f"{ONE_STAGE}/served_verdicts.py", f"{ONE_STAGE}/inference_only.py",
                                f"{ONE_STAGE}/served_compare.py", POUS_PIECE, "benchmarks/network_traces/served_session.py",
                                f"{V}/pipeline/advice.py", "adapters/vllm/tools/advice_compile.py")),
    "Window": Line("a record in a log: a line of `windows.jsonl`",
                   writes=((f"{V}/pipeline/serve_commit.py", r"windows\.jsonl"),),
                   reads=_all(r"windows\.jsonl", DRIVER, f"{ONE_STAGE}/served_verdicts.py", f"{ONE_STAGE}/pouw_piece.py",
                              f"{ONE_STAGE}/inference_only.py", LAUNCHER, f"{PROVE_PATH}/run.sh", f"{V}/pipeline/advice.py")),
    "Io": Line("a record in a log: a line of `requests.jsonl`",
               writes=((f"{V}/pipeline/serve_commit.py", r"requests\.jsonl"),),
               reads=_all(r"requests\.jsonl", DRIVER, f"{ONE_STAGE}/served_verdicts.py", f"{ONE_STAGE}/inference_only.py",
                          CERTIFIER_PIECE, f"{V}/pipeline/advice.py")),
    "Summary": Line("a file: `summary.json`",
                    writes=((f"{V}/pipeline/serve_openai.py", r"summary\.json"),),
                    reads=((f"{V}/pipeline/advice.py", r"summary\.json"),)),
    "Advice": Line("a file: `advice/w<k>/private/advice.bin`, by `verity-vllm advice`",
                   writes=((f"{V}/pipeline/advice.py", r"advice\.bin"),),
                   reads=_all(r"advice\.bin", LAUNCHER, DRIVER, "adapters/vllm/tools/advice_compile.py")),
    "Programs": Line("files: `windows/w<K>/build_request_*/` and their manifest, by `tools/advice_compile.py`",
                     writes=(("adapters/vllm/tools/advice_compile.py", r"manifest\.json"),),
                     reads=((DRIVER, r"descriptor\.json\.gz"), (LAUNCHER, r'--row-dir "\$D/windows/w\$K"'))),
    "Values": Line("files: the window commit `v3c-w<K>-<class>/window/`, which the prover's driver makes from the sealed "
                   "window by importing the adapter",
                   writes=((DRIVER, r"def window_commit_of_advice"), (LAUNCHER, r"prove_path\.py window-commit")),
                   reads=((f"{PROVE_PATH}/records_audit.py", r'window / "registration\.json"'),
                          (LAUNCHER, r"records_audit\.py (warm|audit|period) --window"),
                          (CERTIFIER_PIECE, r'v3c / "period\.json"'))),
    "Replay": Line("a message on a channel: SPOT[Flock]'s inference server sends its proof server `Replay`; the served "
                   "run's prover reads the sealed window instead (an exception below)",
                   writes=((f"{PROTOCOLS}/proofs/spot/verifier/parties.py", r'"replay", PROOF_SERVER, "Replay"'),),
                   reads=((f"{PROTOCOLS}/proofs/spot/verifier/parties.py", r'm\.type == "Replay"'),)),
    "Compiler": Line("a command line: `verity-vllm advice --compiler-name --compiler-digest`, from `prove_path.py compiler`",
                     writes=((DRIVER, r'add_parser\("compiler"'), (LAUNCHER, r"--compiler-name")),
                     reads=((f"{V}/pipeline/advice.py", r"compiler_digest"),)),
    "Weights": Line("a file: `weights_registration.json`, by `prove_path.py weights-registration`",
                    writes=((DRIVER, r"def weights_registration_of_serve"), (LAUNCHER, r"prove_path\.py weights-registration")),
                    reads=((f"{V}/pipeline/advice.py", r"weights-of-record"), (POUS_PIECE, r"weights_registration\.json"),
                           (f"{PROVE_PATH}/records_audit.py", r'"--registered"'))),
    "Multiply": Line("a function call: the adapter calls PoUW's prover package, by its schemes and circuits",
                     writes=_all(r"verity\.protocols\.accounting\.pouw\.prover", f"{V}/pipeline/serve_pearl.py",
                                 f"{V}/protocols/pouw/__init__.py"),
                     reads=((f"{POUW_PROVER}/schemes/__init__.py", r"^def get\("), (f"{POUW_PROVER}/circuit/__init__.py", r"^SCHEMES")),
                     calls=(POUW_PROVER,)),
    "WindowProgram": Line("a file: `h1/window-program.json.gz` with `h1/window-program.meta.json`",
                          writes=((f"{V}/protocols/pouw/h1/__init__.py", r"window-program\.json\.gz"),
                                  (f"{V}/protocols/pouw/h1/served_program.py", r"PROGRAM_META")),
                          reads=((f"{ONE_STAGE}/pouw_piece.py", r"window-program\.json\.gz"),
                                 (f"{ONE_STAGE}/served_verdicts.py", r"window-program\.json\.gz"), (DRIVER, r"PROGRAM_META"),
                                 (LAUNCHER, r"window-program\.json\.gz"))),
    "Calls": Line("a file: `h1/w<k>/pouw-h1-calls.json`",
                  writes=((f"{V}/protocols/pouw/h1/__init__.py", r"pouw-h1-calls\.json"),),
                  reads=_all(r"pouw-h1-calls\.json", f"{ONE_STAGE}/pouw_piece.py", f"{ONE_STAGE}/served_verdicts.py")),
    "Store": Line("a function call: PoUS's piece calls the adapter's `protocols.install` on the weights it loaded",
                  writes=((POUS_PIECE, r"PO\.install\(model, ctx\)"),),
                  reads=((f"{V}/protocols/__init__.py", r"^def install\("), (f"{V}/protocols/pous.py", r"^def install\("))),
    "Stored": Line("a function call's result: the installed option's `store` and `prover`, read as attributes",
                   writes=((f"{V}/protocols/pous.py", r"^    def prover\("),),
                   reads=((POUS_PIECE, r"opt\.store"),)),
}

#: The kinds of line ``EXCEPTIONS`` holds.  A line of one of the first three (``ABOUT``) names the declared message it is about.
KINDS = {
    "second name": "a declared message under a second name or form: making the two one changes a file's format",
    "direction": "the code sends the message the other way than ARCHITECTURE.md, which has the workload send it to the prover",
    "not asked": "ARCHITECTURE.md has the prover ask the workload for it; the code asks elsewhere",
    "configuration": "a path or a setting the demo's configuration states, carried by a flag or a variable: no message",
    "direct read": "not a message: one side reads the other's files where they lie",
    "verifier call": "not a message: the workload's side calls a verifier, which an adapter never knows",
    "device call": "not a message: the workload's side reaches an auditor's device, which an adapter never knows",
    "code reach": "not a message: one side runs or loads the other's code past its interface",
    "undeclared": "a message only under a switch that the run `demos/vllm/placement.toml` places doesn't set: not yet declared",
}
ABOUT = ("second name", "direction", "not asked")


@dataclass(frozen=True)
class Exc:
    """One exception: its kind (``KINDS``), the declared message it is about (``of``) where its kind names one, and its
    line."""

    kind: str
    line: Line
    of: str | None = None


def _exc(kind: str, what: str, of: str | None = None, **sites: Any) -> Exc:
    return Exc(kind, Line(what, **sites), of)


#: Everything else that crosses between the workload and the prover.  Delete a line when its crossing is gone, and lower
#: ``EXCEPTIONS_MAX`` with it.
EXCEPTIONS = (
    _exc("second name", "the engine's rotary table (`private/cos_sin_cache.npy`), which the registration is made over",
         "Weights", writes=((f"{V}/pipeline/m1_capture.py", r"cos_sin_cache\.npy"),), reads=((DRIVER, r"cos_sin_cache\.npy"),)),
    _exc("second name", "the window's seal, a line of `seals.jsonl`", "Window",
         writes=((f"{V}/pipeline/serve_commit.py", r"seals\.jsonl"),),
         reads=_all(r"seals\.jsonl", f"{ONE_STAGE}/served_verdicts.py", f"{ONE_STAGE}/inference_only.py")),
    _exc("second name", "the inputs' users alone, a line of `request_users.jsonl`", "Io",
         writes=((f"{V}/pipeline/serve_commit.py", r"request_users\.jsonl"),),
         reads=(("benchmarks/network_traces/served_session.py", r"request_users\.jsonl"),)),
    _exc("second name", "the advice before it is encoded: the step trace `advice_trace.jsonl`", "Advice",
         writes=((f"{V}/engine/step_trace.py", r"advice_trace\.jsonl"),),
         reads=_all(r"advice_trace\.jsonl", f"{V}/pipeline/advice.py", CERTIFIER_PIECE)),
    _exc("second name", "the advice as JSON and with its salts (`private/advice.json`, `private/opening.json`)", "Advice",
         writes=((f"{V}/pipeline/advice.py", r'private / "(opening|advice)\.json"'),), reads=((f"{PROVE_PATH}/run.sh", r"opening\.json"),)),
    _exc("second name", "the advice's registration and the index of windows (`advice/w<k>/registration.json`, "
         "`advice/windows.json`)", "Advice",
         writes=((f"{V}/pipeline/advice.py", r'out / "registration\.json"'),), reads=((f"{PROVE_PATH}/run.sh", r"registration\.json"),)),
    _exc("second name", "the writer's library for the values: the adapter calls `verity.commitments.windows`", "Values",
         writes=((f"{V}/pipeline/serving_rows.py", r"verity\.commitments\.windows"),),
         reads=(("verity/commitments/windows/interface.py", r"^class Window"),), calls=LIBRARIES),
    _exc("second name", "the advice's and the ingress scan's code: the adapter calls the certifier's prover package", "Advice",
         writes=((f"{V}/pipeline/advice_code/window_code.py", r"certifier\.prover\.window_charge"),),
         reads=((f"{PROTOCOLS}/accounting/certifier/prover/window_charge.py", r"^class Advice"),),
         calls=(f"{PROTOCOLS}/accounting/certifier/prover",)),
    _exc("direction", "the prover's driver computes the compiler's name and digest and hands them to the adapter", "Compiler"),
    _exc("direction", "the prover's driver makes the registration and the adapter reads it back", "Weights"),
    _exc("not asked", "PoUS's piece, a program of its own after the run, loads the weights again and asks the adapter: "
         "the serving endpoint is not asked", "Store", writes=((POUS_PIECE, r"PO\.install\(model, ctx\)"),)),
    _exc("configuration", "`--scheme`, `--seal`, `--window-steps`, `--window-requests`, `--admit`: how to commit and seal",
         writes=((LAUNCHER, r"--scheme rows-sha512 --seal"),), reads=((f"{V}/pipeline/serve_openai.py", r'option\("--scheme"'),)),
    _exc("configuration", "`--protocols NAME`, `--protocols-config JSON`: which PoUW scheme the multiplies run under",
         writes=((LAUNCHER, r"--protocols pouw:"),), reads=((f"{V}/pipeline/serve_openai.py", r'option\("--protocols"'),)),
    _exc("configuration", "`VLLM_BATCH_INVARIANT`, `VERITY_RETAIN`, `NATIVE_COLLECT_BUILD`: how the engine computes, and where "
         "raw copies are kept", writes=((LAUNCHER, r"export VLLM_BATCH_INVARIANT=1"),), reads=((f"{V}/settings.py", r"VERITY_RETAIN"),)),
    _exc("configuration", "`--required-manifest FILE`: the path of the Circuit's own manifest",
         writes=((LAUNCHER, r"--required-manifest"),), reads=((f"{V}/pipeline/serve_openai.py", r'option\("--required-manifest"'),)),
    _exc("configuration", "`--h1-prover DIR`: where to keep each call's prover state",
         writes=((LAUNCHER, r"--h1-prover"),), reads=((f"{V}/pipeline/serve_h1.py", r'option\("--h1-prover"'),)),
    _exc("configuration", "`--day DAY --day-log --day-program`: the day's declaration a window is held to",
         writes=((LAUNCHER, r'--day "\$DAY"'),), reads=((f"{V}/pipeline/serve_day.py", r'option\("--day"'),)),
    _exc("direct read", "the checkpoint's files, opened by path (`--checkpoint-dir`, `--hf-home`)",
         reads=((LAUNCHER, r"--checkpoint-dir"), (DRIVER, r'"--checkpoint-dir"'), (POUS_PIECE, r'"--hf-home"'))),
    _exc("direct read", "that the endpoint is up: `serve.log`'s line `SERVE-OPENAI listening`",
         writes=((f"{V}/pipeline/serve_openai.py", r"SERVE-OPENAI listening"),), reads=((LAUNCHER, r"SERVE-OPENAI listening"),)),
    _exc("direct read", "the adapter's pins of the checkpoint (`manifests/checkpoints-int8.json`), opened by path",
         writes=((f"{V}/engine/int8_checkpoint.py", r"checkpoints-int8\.json"),), reads=((POUS_PIECE, r"checkpoints-int8\.json"),)),
    _exc("direct read", "the sealed window (`windows/w<k>/commit/replay_bundle_p0/`), where SPOT[Flock]'s own deployment "
         "sends `Replay`", writes=((f"{V}/pipeline/serve_seal.py", r"^def seal_window"),), reads=((DRIVER, r"replay_bundle_p0"),)),
    _exc("direct read", "PoUW's prover state under `--h1-prover` (`private/h1/w<k>/`: each call's rows and the window's key)",
         writes=((f"{V}/pipeline/serve_h1.py", r'option\("--h1-prover"'),), reads=((LAUNCHER, r"--h1-key"), (DRIVER, r'"--h1-key"'))),
    _exc("verifier call", "the adapter imports protocols' verifiers",
         writes=((f"{V}/pipeline/serve_commit.py", r"from verity\.protocols\.accounting\.certifier\.verifier"),
                 (f"{V}/pipeline/serve_tie.py", r"verity\.protocols\.proofs\.spot\.verifier")),
         calls=tuple(f"{PROTOCOLS}/{p}/verifier" for p in ("accounting/certifier", "accounting/poub", "accounting/pous",
                                                           "accounting/pouw", "accounting/registration", "proofs/spot"))),
    _exc("verifier call", "`--remote-verifier HOST:PORT`: the endpoint connects to the bandwidth audit's verifier itself and "
         "writes `remote_session.json`", writes=((f"{V}/pipeline/serve_remote.py", r'option\("--remote-verifier"'),),
         reads=((f"{ONE_STAGE}/plan_shards.py", r"remote_session\.json"),)),
    _exc("device call", "the adapter imports auditor devices' code",
         writes=((f"{V}/protocols/pous.py", r"verity\.protocols\.accounting\.pous\.device"),),
         calls=(f"{PROTOCOLS}/accounting/pous/device", f"{PROTOCOLS}/proofs/spot/device")),
    _exc("device call", "`--challenger-gate CMD --challenger-peer NAME`: the endpoint opens each window's session on the "
         "auditor host's challenger itself and takes its coins",
         writes=((LAUNCHER, r"--challenger-gate"),), reads=((f"{V}/pipeline/serve_h1.py", r'option\("--challenger-gate"'),)),
    _exc("device call", "the network certifier in front of the endpoint (`CERTIFIER_FRONT`): a request's release order "
         "reaches the engine in the request itself",
         writes=((LAUNCHER, r'bash -c "\$CERTIFIER_FRONT"'),), reads=((f"{V}/pipeline/serve_openai.py", r'option\("--admit"'),)),
    _exc("code reach", "the prover's driver imports the adapter's modules",
         writes=((DRIVER, r"^\s*from verity_vllm"),), reads=((f"{V}/pipeline/advice_windows.py", r"^class Capped"),)),
    _exc("code reach", "the adapter loads code under `benchmarks/` by path",
         writes=((f"{V}/protocols/pous.py", r"P2_V1 = config\.ROOT"),)),
    _exc("code reach", "the verifier's record of a window's circuit (`window-circuit-w<K>-<class>.json`), which the adapter's "
         "`block-counts` judges the inference-only policy's counts from",
         writes=((DRIVER, r"def window_circuit_of_advice"),), reads=((LAUNCHER, r"verity_vllm\.pipeline\.cli block-counts"),)),
    _exc("code reach", "Pearl+'s kernel: a cubin the launcher builds and the endpoint loads by path",
         writes=((LAUNCHER, r"pearl_c_sm90\.sh"),), reads=((f"{V}/pipeline/serve_pearl.py", r"cubin"),)),
    _exc("code reach", "the launcher builds PoUW's circuits with a module of the adapter's tests",
         writes=((LAUNCHER, r"-m tests\.protocols\.pouw_h1_circuits"),)),
    _exc("undeclared", "`--h1-window-commit CMD`: at a window's close the endpoint runs the prover's window commit",
         writes=((f"{V}/pipeline/serve_tie.py", r'argv = \["bash", "-c", self\.command'),), reads=((LAUNCHER, r"^tie\(\) \{"),)),
    _exc("undeclared", "the window's record (`h1/w<k>/record.json`), the answer to `--h1-window-commit`",
         writes=((LAUNCHER, r"window/registration\.json"),), reads=((f"{V}/pipeline/serve_tie.py", r"R_FILE"),)),
    _exc("undeclared", "PoUW's open period and its draw (`period.json`)",
         writes=((f"{ONE_STAGE}/served_verdicts.py", r'PERIOD = "period\.json"'),),
         reads=((f"{V}/protocols/pouw/h1/drawn.py", r'PERIOD = "period\.json"'),)),
    _exc("undeclared", "a window's committed step-input rows and their salts (`served_rows/w<k>.json`)",
         writes=((f"{V}/pipeline/serve_commit.py", r'"served_rows"'),), reads=((f"{PROVE_PATH}/served_tie_window.py", r'"served_rows"'),)),
    _exc("undeclared", "a window's registration for PoUW's tie and the log of ties (`h1/w<k>/registration.json`, `h1/log.jsonl`)",
         writes=((f"{V}/pipeline/serve_tie.py", r'LOG = "log\.jsonl"'),), reads=((DRIVER, r'"--window-registration"'),)),
    _exc("undeclared", "Pearl+'s passes (`pearl/w<k>/served.json`, `coins.json`, `pass-NNNNN/`)",
         writes=((f"{V}/pipeline/serve_pearl.py", r'"served\.json"'),), reads=((DRIVER, r'"--pearl-passes"'),)),
)
EXCEPTIONS_MAX = 38


def declarations() -> dict[str, Any]:
    """Each protocol's declaration, by its path: the module ``workload`` in its ``verifier/``, sampled proofs' first."""
    paths = sorted((p.relative_to(ROOT).as_posix() for p in (ROOT / PROTOCOLS).glob("*/*/verifier/workload.py")),
                   key=lambda p: (p != SPOT, p))
    return {p: importlib.import_module(p[:-len(".py")].replace("/", ".")) for p in paths}


def _name(path: str) -> str:
    return path.split("/")[-3]


def problems(mods: dict[str, Any], today: dict[str, Line], exceptions: tuple[Exc, ...], most: int) -> list[str]:
    """What keeps ``mods``, ``today`` and ``exceptions`` from being one list, each with its fix."""
    spot = mods.get(SPOT)
    if spot is None:
        return [f"{SPOT} is gone: it declares the two parties' names and the messages more than one protocol reads"]
    ends, out = {spot.WORKLOAD, spot.PROVER}, []
    owner: dict[str, str] = {}
    for path, m in mods.items():
        for s in m.MESSAGES:
            if not isinstance(s, Schema):
                out.append(f"{path}: MESSAGES holds {s!r}, which is no verity.channel.Schema")
            elif s.type in owner:
                out.append(f"{s.type} is declared in {owner[s.type]} and in {path}: declare a message once (one that serves "
                           f"more than one protocol in {SPOT}) and name it in the other's channel")
            else:
                owner[s.type] = path
    way: dict[str, tuple[str, str]] = {}
    for path, m in mods.items():
        out += [f"{path}: {a} is {getattr(m, a)!r}, and {SPOT}'s is {getattr(spot, a)!r}: a declaration that states it states the same"
                for a in ("WORKLOAD", "PROVER", "MAX_BODY") if getattr(m, a, getattr(spot, a)) != getattr(spot, a)]
        pairs = [(c.sender, c.receiver) for c in m.CHANNELS if isinstance(c, Channel)]
        if len(pairs) != len(m.CHANNELS) or len(set(pairs)) != len(pairs) or any(set(p) != ends for p in pairs):
            out.append(f"{path}: CHANNELS is at most one verity.protocol.Channel each way between {sorted(ends)}")
            continue
        carried = {t for c in m.CHANNELS for t in c.messages}
        out += [f"{path}: its channels name {t}, which no declaration's MESSAGES holds" for t in sorted(carried - set(owner))]
        out += [f"{path} declares {s.type} and none of its channels carries it" for s in m.MESSAGES
                if isinstance(s, Schema) and s.type not in carried]
        for c in m.CHANNELS:
            for t in c.messages:
                if way.setdefault(t, (c.sender, c.receiver)) != (c.sender, c.receiver):
                    out.append(f"{path} sends {t} from {c.sender}, and another declaration sends it the other way")
    out += [f"{t} ({owner[t]}) has no line in TODAY: say how the tree carries it, with its writers and its readers"
            for t in sorted(set(owner) - set(today))]
    out += [f"TODAY has a line for {t}, which no declaration's MESSAGES holds: declare it or delete the line"
            for t in sorted(set(today) - set(owner))]
    for e in exceptions:
        if e.kind not in KINDS:
            out.append(f"an exception's kind is one of {sorted(KINDS)}, not {e.kind!r} ({e.line.what})")
        elif (e.kind in ABOUT) != (e.of is not None) or (e.of is not None and e.of not in owner):
            out.append(f"the exception {e.line.what!r} names the declared message it is about exactly when its kind "
                       f"({e.kind}) speaks of one, not {e.of!r}")
    if len(exceptions) > most:
        out.append(f"EXCEPTIONS holds {len(exceptions)} lines, above EXCEPTIONS_MAX = {most}: it only shrinks, so declare "
                   "the message, or make the crossing one of the declared messages, rather than add a line")
    return out


def declared_elsewhere(owner: dict[str, str], texts: dict[str, str]) -> list[str]:
    """Each ``Schema`` of a declared message type that a file of ``texts`` other than its declaration constructs."""
    one = re.compile(r'Schema\(\s*"(' + "|".join(map(re.escape, sorted(owner))) + r')"')
    return [f"{rel}:{text.count(chr(10), 0, m.start()) + 1}: Schema({m.group(1)!r}, ...): take it from {owner[m.group(1)]}"
            for rel, text in sorted(texts.items()) for m in one.finditer(text) if rel != owner[m.group(1)]]


def missing(line: Line, texts: dict[str, str | None]) -> list[str]:
    """The stated writers and readers of ``line`` that hold no such line: ``texts`` is each file's text, None if gone."""
    out = []
    for role, sites in (("writes", line.writes), ("reads", line.reads)):
        for f, pattern in sites:
            if texts.get(f) is None:
                out.append(f"{role}: {f} is gone")
            elif not re.search(pattern, texts[f], re.M):
                out.append(f"{role}: no line of {f} holds /{pattern}/")
    return out


def _texts(lines: list[Line]) -> dict[str, str | None]:
    files = {f for line in lines for f, _ in (*line.writes, *line.reads)}
    return {f: (ROOT / f).read_text(encoding="utf-8", errors="replace") if (ROOT / f).is_file() else None for f in files}


def _sides() -> set[str]:
    return {p.relative_to(ROOT).as_posix() for p in (ROOT / PROTOCOLS).glob("*/*/*") if p.is_dir()}


def targets(source: str, sides: set[str]) -> set[str]:
    """The protocol sides (``sides``) and libraries (``LIBRARIES``) that ``source`` imports from, at any depth."""
    out = set()
    for node in ast.walk(ast.parse(source)):
        names = ([a.name for a in node.names] if isinstance(node, ast.Import) else
                 [node.module, *(f"{node.module}.{a.name}" for a in node.names)]
                 if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module else [])
        for name in names:
            path = name.replace(".", "/")
            out |= {t for t in (*sides, *LIBRARIES) if path == t or path.startswith(t + "/")}
    return out


@functools.cache
def adapter_calls() -> dict[str, list[str]]:
    """Each protocol side and library a tracked Python file under ``adapters/`` imports from, with the files that do."""
    tracked = subprocess.run(["git", "ls-files", "-z", "--", f"{ADAPTERS}*.py"], cwd=ROOT, check=True,
                             capture_output=True).stdout.decode().split("\0")
    sides, found = _sides(), {}
    for rel in filter(None, tracked):
        text = (ROOT / rel).read_text(encoding="utf-8", errors="replace") if (ROOT / rel).is_file() else ""
        if "verity.protocols" in text or "verity.commitments" in text or "from verity " in text or "import verity" in text:
            for t in targets(text, sides):
                found.setdefault(t, []).append(rel)
    return found


def listing(mods: dict[str, Any]) -> str:
    """The list as text: every declared message each way with its kind, the protocol that declares it, the protocols whose
    channels name it and how the tree carries it today, then the exceptions by kind."""
    spot = mods[SPOT]
    owner = {s.type: _name(path) for path, m in mods.items() for s in m.MESSAGES}
    kind = {s.type: s.kind for m in mods.values() for s in m.MESSAGES}
    out = []
    for sender, receiver in ((spot.WORKLOAD, spot.PROVER), (spot.PROVER, spot.WORKLOAD)):
        named: dict[str, list[str]] = {}
        for path, m in mods.items():
            for c in m.CHANNELS:
                if (c.sender, c.receiver) == (sender, receiver):
                    for t in c.messages:
                        named.setdefault(t, []).append(_name(path))
        out.append(f"{sender} -> {receiver}: {len(named)} messages")
        for t, names in named.items():
            also = [n for n in names if n != owner.get(t)]
            out.append(f"  {t:14} {kind.get(t, '?'):11} {owner.get(t, '?'):9} {'also ' + ', '.join(also) if also else '':28} "
                       f"{TODAY[t].what if t in TODAY else '?'}")
    out.append(f"exceptions: {len(EXCEPTIONS)} (at most {EXCEPTIONS_MAX})")
    for k, why in KINDS.items():
        lines = [e for e in EXCEPTIONS if e.kind == k]
        out.append(f"  {k}: {len(lines)}. {why}")
        out += [f"    {e.of + ': ' if e.of else ''}{e.line.what}" for e in lines]
    return "\n".join(out)


def test_the_declarations_are_one_list_and_the_tree_carries_each_message_where_it_says() -> None:
    """Every protocol's declaration is well formed, a message is declared once and nowhere else, each has its line in
    ``TODAY``, and each stated writer and reader holds its line."""
    mods = declarations()
    print(listing(mods))
    found = problems(mods, TODAY, EXCEPTIONS, EXCEPTIONS_MAX)
    assert not found, "the messages between the workload and the prover are one list:\n" + "\n".join(found)
    owner = {s.type: path for path, m in mods.items() for s in m.MESSAGES}
    grep = subprocess.run(["git", "grep", "-l", "-E", r"Schema\(", "--", "*.py", ":!tests/test_workload_messages.py"],
                          cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    twice = declared_elsewhere(owner, {rel: (ROOT / rel).read_text(encoding="utf-8", errors="replace") for rel in grep})
    assert not twice, "a message between the workload and the prover is declared once:\n" + "\n".join(twice)
    texts = _texts(list(TODAY.values()))
    moved = [f"{t}: {m}" for t, line in TODAY.items() for m in missing(line, texts)]
    assert not moved, ("a message between the workload and the prover is written or read elsewhere than tests/"
                       "test_workload_messages.py's TODAY says. Name the file that writes or reads it now and a line it "
                       "holds, in that message's line of TODAY:\n" + "\n".join(moved))


def test_an_adapter_imports_only_what_a_line_of_the_list_names() -> None:
    """No new crossing by import: every protocol side and library a Python file under ``adapters/`` imports from is the
    ``calls`` of a declared message or of an exception."""
    named = {t for line in (*TODAY.values(), *(e.line for e in EXCEPTIONS)) for t in line.calls}
    new = {t: files for t, files in adapter_calls().items() if t not in named}
    assert not new, ("an adapter reaches a protocol only by a message of the list (tests/test_workload_messages.py). Declare "
                     "the message in the protocol's verifier/workload.py and name what the adapter imports for it in its "
                     "line of TODAY (`calls`); EXCEPTIONS only shrinks:\n"
                     + "\n".join(f"{t}: {', '.join(files[:3])}{' ...' if len(files) > 3 else ''}" for t, files in sorted(new.items())))


def test_every_exception_is_current() -> None:
    """``EXCEPTIONS`` only shrinks: a line whose crossing is gone (a stated writer or reader holds no such line, nothing
    under ``adapters/`` imports what it calls, or the declaration now sends its message the document's way) is deleted."""
    mods = declarations()
    spot = mods[SPOT]
    texts, calls = _texts([e.line for e in EXCEPTIONS]), adapter_calls()
    stale = [f"{e.kind}: {e.line.what}: {m}" for e in EXCEPTIONS for m in missing(e.line, texts)]
    stale += [f"{e.kind}: {e.line.what}: nothing under {ADAPTERS} imports {t}" for e in EXCEPTIONS for t in e.line.calls
              if t not in calls]
    stale += [f"TODAY's {t}: nothing under {ADAPTERS} imports {c}" for t, line in TODAY.items() for c in line.calls
              if c not in calls]
    to_prover = {t for m in mods.values() for c in m.CHANNELS if c.receiver == spot.PROVER for t in c.messages}
    stale += [f"direction: {e.of} now goes from the workload to the prover" for e in EXCEPTIONS
              if e.kind == "direction" and e.of in to_prover]
    assert not stale, ("tests/test_workload_messages.py's EXCEPTIONS names crossings the tree no longer has. Delete each "
                       "line, or name where its crossing is now, and lower EXCEPTIONS_MAX:\n" + "\n".join(stale))


def test_a_broken_list_is_found() -> None:
    """The checks above aren't vacuous: a party under another name, a message declared twice, one no channel carries, a name no declaration holds, a
    line of ``TODAY`` with no message, a list grown past its cap, a second ``Schema`` of a declared type, a writer that
    moved and an import in each form are each found."""
    mods = dict(declarations())
    spot = mods[SPOT]

    class Twice:
        PROVER = "other"
        MESSAGES = (spot.REPLAY, Schema("Lone", 1, "private"))
        CHANNELS = (Channel(spot.WORKLOAD, spot.PROVER, ("Replay", "Nowhere")),)

    mods["verity/protocols/proofs/twice/verifier/workload.py"] = Twice
    found = "\n".join(problems(mods, {**TODAY, "Gone": Line("a file")}, EXCEPTIONS, len(EXCEPTIONS) - 1))
    for want in ("PROVER is 'other'", "Replay is declared in", "its channels name Nowhere", "declares Lone and none of its channels",
                 "Lone (verity/protocols/proofs/twice/verifier/workload.py) has no line in TODAY", "TODAY has a line for Gone",
                 "above EXCEPTIONS_MAX"):
        assert want in found, (want, found)
    assert declared_elsewhere({"Replay": SPOT}, {SPOT: 'Schema("Replay", 1)', "a.py": 'x = 1\ny = Schema(\n    "Replay", 1)'}) == [
        f"a.py:2: Schema('Replay', ...): take it from {SPOT}"]
    line = Line("a file", writes=(("a.py", r"^def write\("), ("b.py", r"x")), reads=(("c.py", r"read"),))
    assert missing(line, {"a.py": "import x\ndef write():\n", "b.py": None, "c.py": "def write():\n"}) == [
        "writes: b.py is gone", "reads: no line of c.py holds /read/"]
    sides = {"verity/protocols/accounting/pouw/prover", "verity/protocols/proofs/spot/verifier"}
    src = ("from verity.protocols.accounting.pouw import prover\nimport verity.commitments.windows.rows as R\n"
           "def f():\n    from verity.protocols.proofs.spot.verifier.one_stage import registration\n"
           "from verity.protocols.proofs import spot\nfrom . import verifier\n")
    assert targets(src, sides) == {*sides, "verity/commitments/windows"}


if __name__ == "__main__":
    print(listing(declarations()))
