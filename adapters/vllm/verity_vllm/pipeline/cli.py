"""`verity-vllm` -- the one command line of the vLLM integration.

    verity-vllm COMMAND [SUBCOMMAND] [options]          verity-vllm COMMAND --help for its options

A command is a library module with an options dataclass -- `Options`, or `COMMANDS = {name: <options class>}` for a
command with subcommands -- whose fields are `config.option`s, and `main(options)`.  This module is the only place that
parses a command line: it builds the parser from the dataclass fields, takes the environment defaults the fields name
(`option(env=...)`), and reports a `config.UsageError`, the command's or an option type's, as a usage error with its message
(exit 2).  A process it starts refuses a `VERITY_*`
variable nothing in the tree reads (`settings.validate_environ`), naming it, also exit 2.  `python -m verity_vllm.pipeline.cli`
is the same program, for interpreters without the `verity-vllm` launcher.  A command prints what it says to its reader (a
record, a card, a plan) with `write` (`verity_vllm.log`'s, which every layer may load), on stdout; its diagnostics go to
its logger, on stderr.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import os
import sys
from collections.abc import Callable, Mapping, MutableMapping
from types import ModuleType
from typing import Any

from verity_vllm import settings
from verity_vllm.config import UsageError
from verity_vllm.log import write

__all__ = ["main", "parse", "run", "write"]

PROG = "verity-vllm"

#: command -> (module, one-line help).  Modules are imported only when their command runs.
COMMANDS: dict[str, tuple[str, str]] = {
    "advice": ("verity_vllm.pipeline.advice", "a served run's advice (I1) for the precommitted compiler, from its step trace and Commit record"),
    "block-counts": ("verity_vllm.program.predict.block_counts", "a day's (c, k)-inference blocks from its served windows' circuit records, judged against its declared n: the judgment public, the counts private"),
    "build": ("verity_vllm.pipeline.build", "derive one request wrapper's Program (Build step)"),
    "close-record": ("verity_vllm.program.predict.close_plan", "a served window's record after the card is released, from the handoff its close left"),
    "epoch": ("verity_vllm.pipeline.epoch", "the re-baseline epoch's row job (job, resolve, row-state, word-check, digests, ...)"),
    "fa2-tap-apply": ("verity_vllm.acquire.fa2_tap_src.apply_fa2_tap", "patch a pinned vllm-flash-attn tree into the FA2 tapped tree"),
    "fa2-tap-build": ("verity_vllm.acquire.fa2_tap_src.build_ext", "build one FA2 forward variant extension"),
    "fa3-tap-apply": ("verity_vllm.acquire.fa3_tap_src.apply_fa3_tap", "patch a pinned vllm-flash-attn tree into the FA3 tapped tree"),
    "fa3-tap-build": ("verity_vllm.acquire.fa3_tap_src.build_fa3_ext", "build the FA3 forward extension"),
    "fold-compare": ("verity_vllm.check.fold_compare", "folded observation log vs evidence"),
    "fp8-checkpoint": ("verity_vllm.engine.fp8_checkpoint", "make or re-make a block-FP8 checkpoint from a BF16 pin"),
    "int8-checkpoint": ("verity_vllm.engine.int8_checkpoint", "make or re-make an int8 checkpoint (exact-int8 linears) from a BF16 or FP8-block pin"),
    "gemm-correspondence": ("verity_vllm.program.gemm_target_correspondence", "a target's bf16 linear vs its registered GEMM Definition"),
    "device-pin": ("verity_vllm.pipeline.device_pin", "probe this device's cuBLAS routes and route conditions: its device pin (probe)"),
    "m1-capture": ("verity_vllm.pipeline.match_capture", "capture the observation log under the observer"),
    "match": ("verity_vllm.pipeline.match", "Match stage driver of one configuration"),
    "meta-inventory": ("verity_vllm.engine.meta_inventory", "a checkpoint's modules on the meta device, from its config.json alone"),
    "predict": ("verity_vllm.program.predict.cli", "ahead-of-time Program prediction from φ, compared with the traced Builds; a switched row's Build"),
    "served-match": ("verity_vllm.pipeline.served_match", "the Match of a served-compiled row on live data: serve committed, Build from the process's own artifact, check"),
    "served-pin": ("verity_vllm.acquire.served_artifact", "a served row's pinned compile artifact: seed a fresh Inductor dir with its autotune winners (seed), check a dir's (check)"),
    "serve-openai": ("verity_vllm.program.predict.serve", "an OpenAI-compatible HTTP server on the in-process engine, every step committed"),
    "records": ("verity_vllm.pipeline.circuit_db",
                ("the circuit database: a row's record and its Programs, by row id (publish, pull, capture, show, programs, fetch, "
                 "switch), and a checkpoint's staging (staged)")),
    "router-tap-build": ("verity_vllm.acquire.router_tap_src.build_ext", "build or check the router-softmax tap's CUDA op"),
    "resolve-log": ("verity_vllm.observe.fold.resolve_log", "resolve an observation log against a profile"),
    "serving-rows": ("verity_vllm.pipeline.deferred_rows", "a deferred Commit's serving rows and one-stage registration, from its sealed replay bundle"),
    "topp-split-probe": ("verity_vllm.pipeline.topp_split_probe", "top-p split fixture"),
    "tp-register-tables": ("verity_vllm.pipeline.expert_tables", "a TP rank's registration of the per-expert cut's tables, from the checkpoint"),
    "weights-tie": ("verity_vllm.program.predict.weights_tie", "the weights registration's checkpoint tie: its window (commit), the verifier's (verify), the pins (pins)"),
    "workload": ("verity_vllm.pipeline.workload", "the deterministic sweep workload generator"),
    "workload-file": ("verity_vllm.pipeline.workload_file", "workload files fetched by their art ids and checked against SHA256SUMS"),
}

#: commands of the rows' route, refused by name: requests are committed through the window's circuit (`serve-openai --seal`)
DELETED: dict[str, str] = {
    "row": "the traced row driver and its Build are deleted; a workload's circuit is `predict row`'s",
    "commit": "the Commit is deleted; `serve-openai --seal` commits what it serves",
    "manifest": "the row manifest is deleted; the serve commits the window's circuit",
    "neutrality": "the traced Build it compared a predicted one against is deleted",
    "tp-commit-replay": "the tensor-parallel Commit is deleted, and its replay with it",
    "replay-plan": "the slim replay bundle's planner is deleted with the Commit that started it",
    "stoch-recompute": "the stochastic row's value check is deleted with the row driver whose match directory it read",
    "boundary-plan": "the Commit's call-boundary plan is deleted with the Commit",
    "gate": "the row's acquisition gate is deleted with the row",
    "manifest-verify": "the row manifest is deleted; the serve commits the window's circuit",
    "norm-tap-build": "the norm-scale tap is deleted with the row's Build",
    **dict.fromkeys(("compiled", "batch-decomp", "global-match", "weights-of-record", "runtime-tree", "commit-plan", "compiled-merge",
                     "commit-replay", "global-program", "launch-context", "release-json", "research-outputs", "research-result",
                     "source-identity", "timeline", "verdict", "target-family"),
                    "no stage or script runs it since the rows' route went; its module's `main` runs under `cli.run`"),
}


def usage() -> str:
    w = max(map(len, COMMANDS))
    lines = [f"usage: {PROG} COMMAND [SUBCOMMAND] [options]", "", "commands:"]
    lines += [f"  {name:{w}s}  {help_}" for name, (_, help_) in COMMANDS.items()]
    return "\n".join(lines)


def _doc(cls: type) -> str | None:
    doc = cls.__doc__
    return None if not doc or doc.startswith(f"{cls.__name__}(") else doc


def _add_options(parser: argparse.ArgumentParser, cls: type, environ: Mapping[str, str]) -> None:
    parser.register("action", "boolean_optional", argparse.BooleanOptionalAction)
    for f in dataclasses.fields(cls):
        if not f.init or "environ" in f.metadata:
            continue
        if "group" in f.metadata:
            _add_options(parser, f.metadata["group"], environ)
            continue
        if "flags" not in f.metadata:
            raise TypeError(f"{cls.__name__}.{f.name} is not a config.option")
        flags, kw, env = f.metadata["flags"], dict(f.metadata["kw"]), f.metadata["env"]
        positional = not flags[0].startswith("-")
        if env is not None:
            value = environ.get(env)
            if value is not None and (value != "" or f.metadata["env_empty"]):
                kw["default"] = value
                if positional:
                    kw.setdefault("nargs", "?")
        if not positional:
            kw["dest"] = f.name
            if "default" not in kw:
                kw["required"] = True
        if "type" in kw:
            kw["type"] = _typed(kw["type"])
        parser.add_argument(*flags, **kw)


def _typed(convert: Callable[[str], Any]) -> Callable[[str], Any]:
    """An option's ``type`` whose `config.UsageError` argparse reports with its message (of a ValueError it says only "invalid
    <type> value")."""
    def typed(s: str) -> Any:
        try:
            return convert(s)
        except UsageError as e:
            raise argparse.ArgumentTypeError(str(e)) from e
    typed.__name__ = getattr(convert, "__name__", repr(convert))
    return typed


def _options(cls: type, ns: argparse.Namespace, environ: Mapping[str, str]) -> Any:
    kw: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if not f.init:
            continue
        if "environ" in f.metadata:
            kw[f.name] = dict(environ)
        elif "group" in f.metadata:
            kw[f.name] = _options(f.metadata["group"], ns, environ)
        else:
            kw[f.name] = getattr(ns, f.name)
    return cls(**kw)


def _parser(prog: str, description: str | None) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(prog=prog, description=description, formatter_class=argparse.RawDescriptionHelpFormatter)


def parse(command: str | ModuleType, argv: list[str], environ: Mapping[str, str] | None = None) -> Any:
    """The options object of ``verity-vllm COMMAND ARGV...``, or of a command module's ARGV (exits 2 on a usage error, like any
    command line)."""
    environ = os.environ if environ is None else environ
    if isinstance(command, ModuleType):
        return _parse_module(command, f"{PROG} {command.__name__}", argv, environ)[1]
    return _parse(command, argv, environ)[1]


def run(module: ModuleType, argv: list[str], environ: Mapping[str, str] | None = None) -> Any:
    """A command module's `main` on the options of ARGV, as a command line gives them, for a module no command runs."""
    parser, opts, _ = _parse_module(module, f"{PROG} {module.__name__}", argv, os.environ if environ is None else environ)
    try:
        return module.main(opts)
    except UsageError as e:
        parser.error(str(e))


def _parse(command: str, argv: list[str], environ: Mapping[str, str]) -> tuple[argparse.ArgumentParser, Any, Any]:
    return _parse_module(importlib.import_module(COMMANDS[command][0]), f"{PROG} {command}", argv, environ)


def _parse_module(mod: ModuleType, prog: str, argv: list[str], environ: Mapping[str, str]) -> tuple[argparse.ArgumentParser, Any, Any]:
    subs = getattr(mod, "COMMANDS", None)
    if subs is None:
        parser = _parser(prog, _doc(mod.Options) or mod.__doc__)
        _add_options(parser, mod.Options, environ)
        return parser, _options(mod.Options, parser.parse_args(argv), environ), mod
    parser = _parser(prog, mod.__doc__)
    sp = parser.add_subparsers(dest="_sub", required=True, metavar="SUBCOMMAND")
    by_name = {}
    for name, cls in subs.items():
        doc = _doc(cls)
        by_name[name] = sp.add_parser(name, help=(doc or "").split("\n")[0], description=doc,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
        _add_options(by_name[name], cls, environ)
    ns = parser.parse_args(argv)
    return by_name[ns._sub], _options(subs[ns._sub], ns, environ), mod


def process_settings(environ: MutableMapping[str, str]) -> None:
    """The process-wide settings every command takes from the environment: a leased Commit's lease (`gpu_lease.acquire` reads and
    sets this environment).  The dense chain's threads are not among them: `dense_rows` reads `VERITY_DENSE_THREADS` when it is
    imported, and importing it here would need the core, which a Build's top-level process does not have on its path."""
    if environ.get("VERITY_LEASE_STARTED"):
        from verity_vllm.pipeline import gpu_lease
        gpu_lease.use_environ(environ)


def main(argv: list[str] | None = None) -> Any:
    process = argv is None
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(usage(), file=sys.stdout if argv else sys.stderr)
        return 0 if argv else 2
    if argv[0] in DELETED:
        print(f"{PROG} {argv[0]}: refused: {DELETED[argv[0]]}", file=sys.stderr)
        return 2
    if argv[0] not in COMMANDS:
        print(f"{PROG}: unknown command {argv[0]!r}\n\n{usage()}", file=sys.stderr)
        return 2
    if process:
        try:
            settings.validate_environ()
        except settings.UnknownSetting as e:
            print(f"{PROG}: {e}", file=sys.stderr)
            return 2
    try:
        parser, opts, mod = _parse(argv[0], argv[1:], os.environ)
    except ModuleNotFoundError as e:
        if e.name != COMMANDS[argv[0]][0]:
            raise
        print(f"{PROG}: {argv[0]}: its module {e.name} is not on this tree", file=sys.stderr)
        return 2
    process_settings(os.environ)
    if process:
        # run records store sys.argv as the command's own command line, the one `python -m <module>` gave it
        sys.argv[:] = [mod.__file__ or sys.argv[0], *argv[1:]]
    try:
        if process and os.environ.get("VERITY_LEASE_STARTED"):
            from verity_vllm.pipeline import gpu_lease
            return gpu_lease.run_main(mod.main, opts)
        return mod.main(opts)
    except UsageError as e:
        parser.error(str(e))


if __name__ == "__main__":
    raise SystemExit(main())
