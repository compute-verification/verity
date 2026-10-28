"""`verity-vllm` -- the one command line of the vLLM integration.

    verity-vllm COMMAND [SUBCOMMAND] [options]          verity-vllm COMMAND --help for its options

A command is a library module with an options dataclass -- `Options`, or `COMMANDS = {name: <options class>}` for a
command with subcommands -- whose fields are `config.option`s, and `main(options)`.  This module is the only place that
parses a command line: it builds the parser from the dataclass fields, takes the environment defaults the fields name
(`option(env=...)`), and reports a `config.UsageError`, the command's or an option type's, as a usage error with its message
(exit 2).  A process it starts refuses a `VERITY_*`
variable nothing in the tree reads (`settings.validate_environ`), naming it, also exit 2.  `python -m verity_vllm.pipeline.cli`
is the same program, for interpreters without the `verity-vllm` launcher.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import os
import sys
from collections.abc import Callable, Mapping, MutableMapping
from typing import Any

from verity_vllm import settings
from verity_vllm.config import UsageError

PROG = "verity-vllm"

#: command -> (module, one-line help).  Modules are imported only when their command runs.
COMMANDS: dict[str, tuple[str, str]] = {
    "admission-bound": ("verity_vllm.pipeline.admission_bound", "host-retention bound of a --retain host Commit"),
    "advice": ("verity_vllm.pipeline.advice", "a served run's advice (I1) for the precommitted compiler, from its step trace and Commit record"),
    "batch-decomp": ("verity_vllm.check.match.batch_decomp", "B>1 Match by per-request decomposition"),
    "boolean-purity": ("verity_vllm.program.boolean_purity", "the Definitions a Program reaches that are not Boolean (or, --dry-run, would)"),
    "boolean-replay": ("verity_vllm.pipeline.boolean_replay", "a deferred Commit's replay with every pick evaluated on bits"),
    "boundary-plan": ("verity_vllm.pipeline.boundary_plan", "a row's call-boundary check over its Commit's Programs, and the Commit's call-boundary plan, keyed"),
    "build": ("verity_vllm.pipeline.build", "derive one request wrapper's Program (Build step)"),
    "capture-identities": ("verity_vllm.correspondence.capture_identities", "capture identities vs the canonical B1 circuit"),
    "capture-map": ("verity_vllm.check.capture_map", "a capture map against a module inventory (check), its tap counts and value ids"),
    "census": ("verity_vllm.properties.census", "kernel census of a bare run"),
    "commit": ("verity_vllm.pipeline.commit", "Commit stage: pairs protocol, commitment root and overhead"),
    "commit-arb": ("verity_vllm.pipeline.commit_arb", "the served Commit under seeded varied traffic, each request's record held to its streamed ids"),
    "commit-group": ("verity_vllm.pipeline.commit_group", "the Commits of several rows of one engine, in one process under one GPU lease"),
    "commit-plan": ("verity_vllm.pipeline.commit_plan", "a Commit's plan, derived before the GPU: its producer facts, keyed"),
    "commit-replay": ("verity_vllm.pipeline.cpu_replay", "a deferred Commit's C2 replay on CPU, from its sealed replay bundle"),
    "replay-plan": ("verity_vllm.pipeline.replay_plan", "a slim replay bundle's planner process (started by a `--replay-deferred slim` Commit)"),
    "compiled": ("verity_vllm.acquire.compiled", "required-value manifest of an Execution=compiled row"),
    "compiled-merge": ("verity_vllm.pipeline.compiled_merge", "merge control/treatment outputs of a compiled row"),
    "conformance": ("verity_vllm.program.frontend.conformance", "conformance record of the registered library"),
    "coverage-workloads": ("verity_vllm.pipeline.coverage_workloads", "coverage campaign workloads"),
    "descriptor-equivalence": ("verity_vllm.program.descriptor_equivalence", "correspondence of two encodings of one program"),
    "epoch": ("verity_vllm.pipeline.epoch", "the re-baseline epoch's row job (job, resolve, row-state, word-check, digests, ...)"),
    "fa2-tap-build": ("verity_vllm.acquire.fa2_tap_src.build_ext", "build one FA2 forward variant extension"),
    "fa3-tap-build": ("verity_vllm.acquire.fa3_tap_src.build_fa3_ext", "build the FA3 forward extension"),
    "fold-compare": ("verity_vllm.check.fold_compare", "folded observation log vs evidence"),
    "fp8-checkpoint": ("verity_vllm.engine.fp8_checkpoint", "make or re-make a block-FP8 checkpoint from a BF16 pin"),
    "gate": ("verity_vllm.acquire.gate", "CPU acquisition gate of one row"),
    "int8-checkpoint": ("verity_vllm.engine.int8_checkpoint", "make or re-make an int8 checkpoint (exact-int8 linears) from a BF16 or FP8-block pin"),
    "gemm-correspondence": ("verity_vllm.program.gemm_target_correspondence", "a target's bf16 linear vs its registered GEMM Definition"),
    "global-match": ("verity_vllm.check.match.global_match", "global Match of a batched execution"),
    "global-program": ("verity_vllm.pipeline.global_program", "workload Program from the derived request Programs"),
    "glue": ("verity_vllm.glue", "per model, the code its `__models__` markers tie to it and the hours from staged checkpoint to record"),
    "golden": ("verity_vllm.properties.golden", "golden corpus check"),
    "hidden-gpu": ("verity_vllm.commit.committer.hidden_gpu_src.hidden_gpu", "GPU commitment tree: check, bench, commit"),
    "holdout": ("verity_vllm.properties.holdout", "holdout workload generator and assembler"),
    "hot": ("verity_vllm.pipeline.hot", "hot Commit worker"),
    "instances-form": ("verity_vllm.program.instances_form", "stored form of instances.jsonl operand references"),
    "kernel-identity": ("verity_vllm.engine.kernel_identity", "generated-kernel identity"),
    "launch-context": ("verity_vllm.pipeline.launch_context", "launch context of a workload's request shapes"),
    "libdevice-sampling": ("verity_vllm.program.libdevice_sampling", "libdevice sampler transcription check"),
    "m1-capture": ("verity_vllm.pipeline.m1_capture", "capture the observation log under the observer"),
    "manifest": ("verity_vllm.pipeline.manifest", "the required-value manifest (build, build-global, compare)"),
    "manifest-verify": ("verity_vllm.pipeline.manifest_verify", "is manifest.json the manifest of record"),
    "match": ("verity_vllm.pipeline.match", "Match stage driver of one configuration"),
    "meta-inventory": ("verity_vllm.engine.meta_inventory", "a checkpoint's modules on the meta device, from its config.json alone"),
    "mla-capture": ("verity_vllm.program.mla_capture", "MLA kernel captures (produce), against their numpy models (check) and Booleans (check-bits)"),
    "neutrality": ("verity_vllm.pipeline.neutrality", "rebuild the rows of record's Programs; fail if a Program digest moved"),
    "noninterference": ("verity_vllm.properties.noninterference", "observer non-interference check"),
    "norm-tap-build": ("verity_vllm.acquire.norm_tap_src.build_ext", "build or check the norm-scale tap's CUDA op"),
    "pouw-serve": ("verity_vllm.pipeline.pouw_serve", "a workload served under pouw:ncp-v2, its calls replayed honest and tampered"),
    "predict": ("verity_vllm.program.predict.cli", "ahead-of-time Program prediction from φ, compared with the traced Builds; a switched row's Build"),
    "prefix-cache": ("verity_vllm.pipeline.prefix_cache", "prefix-cache linkage tools"),
    "program-graph": ("verity_vllm.pipeline.program_graph", "a served row's Program as a graph (modules, Calls, dataflow, hashes)"),
    "served": ("verity_vllm.acquire.served", "the served-compiled Program and required-value manifest (build), the GEMM routes it needs (routes)"),
    "served-match": ("verity_vllm.pipeline.served_match", "the Match of a served-compiled row on live data: serve committed, Build from the process's own artifact, check"),
    "served-pin": ("verity_vllm.acquire.served_artifact", "a served row's pinned compile artifact: seed a fresh Inductor dir with its autotune winners (seed), check a dir's (check)"),
    "serve-openai": ("verity_vllm.pipeline.serve_openai", "an OpenAI-compatible HTTP server on the in-process engine, every step committed"),
    "serving-view": ("verity_vllm.pipeline.serving_view", "a served row's Programs as vLLM serves them (engine steps, requests, model runner, ranks)"),
    "profile-descriptor": ("verity_vllm.program.profile_descriptor", "where a serialised descriptor's bytes go"),
    "program-compare": ("verity_vllm.check.match.program_compare", "structural comparison of two Programs"),
    "program-dag": ("verity_vllm.check.match.program_dag", "a derived Program and a fold: live-cone DAG multisets equal"),
    "properties-admission": ("verity_vllm.properties.admission", "quarantine admission differential tests"),
    "quarantine-lint": ("verity_vllm.properties.quarantine_lint", "quarantine registry lint"),
    "records": ("verity_vllm.pipeline.circuit_db", "the circuit database: a row's record and its Programs, by row id (publish, pull, capture, show, programs, fetch, "
                            "switch), and a checkpoint's staging (staged)"),
    "rejudge": ("verity_vllm.pipeline.tp.rejudge", "recompute a finished TP Commit's GPU-free gates from its saved outputs"),
    "release-json": ("verity_vllm.pipeline.release_json", "RELEASE.json writer, reader and release line"),
    "router-tap-build": ("verity_vllm.acquire.router_tap_src.build_ext", "build or check the router-softmax tap's CUDA op"),
    "research-outputs": ("verity_vllm.pipeline.research_outputs", "typed outputs of a sweep stage"),
    "research-result": ("verity_vllm.pipeline.research_result", "a sweep row's result.json"),
    "resolve-log": ("verity_vllm.observe.fold.resolve_log", "resolve an observation log against a profile"),
    "rmsnorm-fused-sweep": ("verity_vllm.program.rmsnorm_fused_sweep", "fused rms-norm summation-tree sweep"),
    "row": ("verity_vllm.pipeline.row", "run a sweep row's stages: Build, Match, Commit (run, stage, chain, shed)"),
    "runtime-tree": ("verity_vllm.observe.runtime_tree", "the runtime's module tree"),
    "sampling-rng": ("verity_vllm.program.sampling_rng", "sampler noise recomputation"),
    "serving-rows": ("verity_vllm.pipeline.deferred_rows", "a deferred Commit's serving rows and one-stage registration, from its sealed replay bundle"),
    "source-identity": ("verity_vllm.pipeline.source_identity", "source identity and precheck"),
    "stoch-recompute": ("verity_vllm.check.replay.stoch_recompute", "value check of a stochastic row"),
    "target-family": ("verity_vllm.target_family", "target-family preflight"),
    "telemetry-admission": ("verity_vllm.pipeline.telemetry.admission_commands", "empirical admission planner"),
    "timeline": ("verity_vllm.pipeline.timeline", "stage telemetry (mark, report, collect)"),
    "topp-pin-check": ("verity_vllm.pipeline.topp_pin_check", "the top-p split pin on the GPU: the kernel pinned and not, requests together and alone"),
    "topp-split-probe": ("verity_vllm.pipeline.topp_split_probe", "top-p split fixture"),
    "tp-capture": ("verity_vllm.pipeline.tp.capture", "tensor-parallel M1 capture"),
    "tp-commit": ("verity_vllm.pipeline.tp.commit", "tensor-parallel Commit"),
    "tp-commit-replay": ("verity_vllm.pipeline.tp.commit_replay", "a deferred tensor-parallel Commit's C2 replay on CPU, from its ranks' sealed bundles"),
    "tp-fold-match": ("verity_vllm.pipeline.tp.fold_match", "per-rank fold Match of a tensor-parallel row"),
    "tp-match": ("verity_vllm.pipeline.tp.match", "tensor-parallel Match at the collective level"),
    "tp-serving-rows": ("verity_vllm.pipeline.tp.serving_rows", "a deferred tensor-parallel Commit's stand-in windows, from its ranks' sealed bundles"),
    "advice-windows": ("verity_vllm.pipeline.advice_windows", "a deferred tensor-parallel Commit's stand-in windows over a compiler's Programs for the advice"),
    "twins": ("verity_vllm.program.twins_check", "vectorized twins of the registered bodies"),
    "verdict": ("verity_vllm.pipeline.verdict_record", "a row's Commit verdict record, rebuilt from its evidence"),
    "weights-of-record": ("verity_vllm.check.weights_of_record", "the weights of record"),
    "weights-tie": ("verity_vllm.program.predict.weights_tie", "the weights registration's checkpoint tie: its window (commit), the verifier's (verify), the pins (pins)"),
    "workload": ("verity_vllm.pipeline.workload", "the deterministic sweep workload generator"),
    "workload-file": ("verity_vllm.pipeline.workload_file", "workload files fetched by their art ids and checked against SHA256SUMS"),
}


#: The pod layout's machine paths: the default of an option (``Class.field``) that neither its flag nor its
#: environment variable sets.
MACHINE: dict[str, str] = {
    "RunEnv.py": "/workspace/venv312/bin/python",
    "RunEnv.sweep_dir": "/workspace/cp/sweep",
    "RunEnv.hf_home": "/workspace/hf",
    "RunEnv.cuda_home": "/usr/local/cuda",
    "RunEnv.cuda_bin": "/usr/local/cuda/bin",
    "RunEnv.native_collect_default": "/workspace/cp/nc_build",
    "RunEnv.hidden_so_fa2": "/workspace/cp/fa2/build/matReq/verity_fa2_matReq.so",
    "RunEnv.hidden_so_fa3": "/workspace/cp/fa2/build/fa3_matReq/verity_fa3_matReq.so",
    "RunEnv.hidden_so_fa2_guarded": "/workspace/cp/fa2/build/matReqG/verity_fa2_matReqG.so",
    "RunEnv.hidden_so_fa3_guarded": "/workspace/cp/fa2/build/fa3_matReqG/verity_fa3_matReqG.so",
    "RunEnv.norm_tap_so": "/workspace/cp/norm_tap/build/verity_norm_tap.so",
    "RunEnv.router_tap_so": "/workspace/cp/router_tap/build/verity_router_tap.so",
    "RunEnv.hot_root": "/workspace/cp/hot",
    "RunEnv.guard_preserve_file": "/root/dm/guard.preserve",
    "RunEnv.pod_guard": "/root/dm/pod_guard.sh",
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
        if f"{cls.__name__}.{f.name}" in MACHINE:
            kw["default"] = MACHINE[f"{cls.__name__}.{f.name}"]
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


def parse(command: str, argv: list[str], environ: Mapping[str, str] | None = None) -> Any:
    """The options object of ``verity-vllm COMMAND ARGV...`` (exits 2 on a usage error, like any command line)."""
    return _parse(command, argv, os.environ if environ is None else environ)[1]


def _parse(command: str, argv: list[str], environ: Mapping[str, str]) -> tuple[argparse.ArgumentParser, Any, Any]:
    mod = importlib.import_module(COMMANDS[command][0])
    prog = f"{PROG} {command}"
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


def set_environment(environ: Mapping[str, str], keep: tuple[str, ...] = ()) -> None:
    """Make the process environment ``environ``, but for the ``keep`` keys the process has, which keep their values: each row of a grouped
    Commit (`commit_group`) runs in its own row's environment, on the devices the lease of its first row set."""
    held = {k: os.environ[k] for k in keep if k in os.environ}
    os.environ.clear()
    os.environ.update(environ)
    os.environ.update(held)


def process_settings(environ: MutableMapping[str, str]) -> None:
    """The process-wide settings every command takes from the environment: a leased Commit's lease (`gpu_lease.acquire` reads and
    sets this environment).  The dense chain's threads are not among them: `dense_rows` reads `VERITY_DENSE_THREADS` when it is
    imported, and importing it here would need the core, which a Build's top-level process does not have on its path."""
    if environ.get("VERITY_LEASE_STARTED"):
        from verity_vllm.pipeline import gpu_lease
        gpu_lease.use_environ(environ)


def main(argv: list[str] | None = None) -> Any:
    process = argv is None
    argv = sys.argv[1:] if process else list(argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(usage(), file=sys.stdout if argv else sys.stderr)
        return 0 if argv else 2
    if argv[0] not in COMMANDS:
        print(f"{PROG}: unknown command {argv[0]!r}\n\n{usage()}", file=sys.stderr)
        return 2
    if process:
        try:
            settings.validate_environ()
        except settings.UnknownSetting as e:
            print(f"{PROG}: {e}", file=sys.stderr)
            return 2
    parser, opts, mod = _parse(argv[0], argv[1:], os.environ)
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
