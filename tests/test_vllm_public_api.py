"""Outside `adapters/vllm/`, code imports the vLLM integration (`verity_vllm`) only through its public interfaces.

``PUBLIC`` names each public module and the interface it is. Outside code may import it, and takes from it only the names
its ``__all__`` lists. Each public module states ``__all__`` as a literal, binds every name in it (at top level, or under
``TYPE_CHECKING`` with a module ``__getattr__`` that loads it), and annotates every parameter and return of each function
it exports and of each public method of each class it exports.

``KNOWN`` lists every other module of the package that outside code imports today, and ``KNOWN_PRIVATE`` every ``_name``
it takes from one. Both only shrink: a new outside import of a module outside ``PUBLIC`` fails, and so does a listed one
that is gone, so the lists stay exact. Move an importer onto a public interface (growing the interface when it lacks what
the importer needs) and delete the line.

An import is what `tests/test_integration_boundaries.py` counts: an import statement, a string handed to
``importlib.import_module``, ``__import__`` or ``pytest.importorskip``, or a ``"module:attr"`` string. ``from P import n``
imports the module ``P.n`` when that is a module of the package, else the name ``n`` of ``P``. A stand-in module a test
builds under a package name (``ModuleType("verity_vllm.x")``) is not an import.

A name is taken from a module by ``from M import n``, by a ``"M:n"`` string, or through the module: ``m.n``, read, set or
deleted, and ``getattr``, ``setattr``, ``delattr`` or ``patch.object`` of ``m`` and a literal ``"n"`` (so pytest's
``monkeypatch.setattr(m, "n", v)``). ``m`` is a name that the nearest scope binding it binds to the module by an import
(``import M as m``, ``from P import m``, ``m = importlib.import_module("M")``), or ``verity_vllm.a.b`` under an
``import verity_vllm...``; a local variable of an alias's name is not the module. Through a module, ``m.s`` reaches the
module ``M.s`` when that is one, as ``from M import s`` does. A module's own dunder attributes (``__name__``,
``__file__``) are no part of an interface, and ``hasattr`` takes nothing. Not seen: a name taken through a module that
was handed to a function, returned from one or kept in a container or an attribute, a name computed at run time, and a
dotted string handed to a patcher (``monkeypatch.setattr("verity_vllm.a.b.n", v)``).
"""
from __future__ import annotations

import ast
import functools
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = "adapters/vllm/"
PACKAGE = "verity_vllm"
PACKAGE_DIR = ROOT / INTEGRATION / PACKAGE
MODULE_ATTR = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*")
IMPORT_CALLS = {"import_module", "__import__", "importorskip"}
#: the calls that take a module and a name of it (`object`: mock's `patch.object`)
NAME_CALLS = {"getattr", "setattr", "delattr", "object"}
DUNDER = re.compile(r"__\w+__")
#: what a name is local to; a class body's names are read by nothing below it, so they shadow nothing
SCOPES = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
#: files outside the integration whose `verity_vllm` strings are data, not imports, each with why
NOT_SCANNED = {
    "tests/test_vllm_public_api.py": "this file",
    "tests/test_module_strings.py": "its exemptions name modules that don't resolve, as data",
}

#: The public modules: module -> the interface it is (I1..I10).
PUBLIC = {
    "verity_vllm": "I1: the pinned engine (LLM), its pin, its spec, its target and the serving device's target and execution",
    "verity_vllm.engine.spec": "I1: engine spec documents",
    "verity_vllm.settings": "I1: the record of the value settings in force in an environment (record): each variable set "
                            "there that can change what is computed or committed, a credential by name only, and their "
                            "digest, which benchmarks/serve_tax writes into an arm's meta.json and a replay's replay.json",
    "verity_vllm.engine.handle": "I2: a built engine's vLLM objects",
    "verity_vllm.engine.stock": "I1: a stock engine at the pin, with no protocol on: an in-process LLMEngine, vLLM's "
                                "SamplingParams and its version",
    "verity_vllm.engine.served_seed": "I1: the seed a served request is sampled under (its own, the public default for a greedy "
                                      "one, or derived from its window's seed coin and its id) and the windows' seed coins",
    "verity_vllm.engine.moe_align": "I1: the served Marlin MoE path's moe_align_block_size, each expert's pairs in pair-index "
                                    "order: its reference, kernels and install, which benchmarks/moe_align checks and times",
    "verity_vllm.engine.step_trace": "I2: the scheduler's step trace of an in-process engine core (StepTrace) and its "
                                     "reader, which serve-openai records and benchmarks/serve_tax times and replays",
    "verity_vllm.engine.sampler_capture": "I2: chosen requests' logits rows and ids at a replaying engine's sampler, held "
                                          "across ranks and written as a token select's span, which benchmarks/serve_tax's "
                                          "replay and rebuild take",
    "verity_vllm.engine.core_gate": "I2: the engine core's one seam between steps (EngineCoreGate): gates that hold its "
                                    "input loop and edges that act between two steps once nothing is in flight",
    "verity_vllm.engine.pool_row": "I1: the carried pool's row, a prefix cache's block pool at a window's edge (snapshot, row, "
                                   "export), its genesis reset and its BlockKey_v1 keys (install_keys), which "
                                   "benchmarks/network_e2e/encodings drives over recorded traces",
    "verity_vllm.observe.kernel_trace": "I2: a served engine's device kernels, read from a Kineto trace of its steps, which "
                                         "benchmarks/route_capture reads",
    "verity_vllm.program.predict.step_needs": "I7: what the DSA decoder calls for one served engine step at its launch, each "
                                              "refusal counted in its place, which benchmarks/route_capture's gap reads",
    "verity_vllm.program.predict.naming": "I7: a request Program's Build with the names I2's compile gives it, the verifier's "
                                          "own Build without a trace",
    "verity_vllm.program.predict.block_counts": "I7: a circuit record's inference blocks counted as the Lean count counts "
                                                "them (the session Definition, its Boolean gates by the gate table, a "
                                                "block's fresh bits), which benchmarks/inference_only prices",
    "verity_vllm.program.predict.window_weights": "I7: the window circuit's weights, the committer's registration and the "
                                                  "verifier's pins of it",
    "verity_vllm.pipeline.expert_tables": "I7: the per-expert cut's tables, their layout and record, the committer's "
                                          "registration of a TP rank's and the verifier's pin of it (tp-register-tables)",
    "verity_vllm.commit.history_tables": "I7: the committer's history tables of DSA's long rows by table reads, one a "
                                         "(request, layer), their switch, retention and roots (Histories), which "
                                         "benchmarks/commitments times",
    "verity_vllm.commit.history_tables_device": "I7: a history table on the GPU and its CPU twin (Table), which "
                                                "benchmarks/commitments times",
    "verity_vllm.program.predict.window_rows": "I7: a window's rows, the window circuit's units filled from a served window's "
                                               "values",
    "verity_vllm.program.predict.scan_state": "I7: the network certifier's scan state as a window class's Programs read it: "
                                              "the registered value, its class, its words and a slot's bridge",
    "verity_vllm.program.predict.window_class": "I7: a window class, and the layouts of PoUW's tables a window circuit reads "
                                                "on a fixed load (FixedLoad): H1's under H1's tie (Tables) and Pearl+'s, the "
                                                "circuit cut at its linears (PearlTables)",
    "verity_vllm.program.frontend.train_step": "I7: one training step's Program (step_program, step_report, export_step), its "
                                               "layout and the primitives a census prices, which the grader "
                                               "(benchmarks/training_slowdown/grader) reads",
    "verity_vllm.pipeline.workload_file": "I1: a workload file, fetched from the evidence store when absent and checked against "
                                          "its digest",
    "verity_vllm.pipeline.edge": "I7: the edge between two Commit windows on the advice's side: the requests it holds, each "
                                 "port's fields, and each carried request's, held to window_advice.follows",
    "verity_vllm.pipeline.boundary_rows": "I1: a served window's boundary commitment: each finished request's b.ingress "
                                          "and b.egress rows by size class, a class's layout, record and files (Layout, "
                                          "record, reopen), which benchmarks/one_stage/two_stage.py registers",
    "verity_vllm.pipeline.boundary_stop": "I1: a serve's stop rule on its boundary commitment (Stop), the stop block that "
                                          "computes where a response ends, and a stop class's stop record "
                                          "(stop_registration, stop_record), which benchmarks/one_stage/two_stage.py "
                                          "registers under the verifier's own rule",
    "verity_vllm.pipeline.boundary_windows": "I1: the boundary commitment as the engine core serves: the hook on the "
                                             "scheduler that commits each window's finished requests by class "
                                             "(BoundaryWindows), which benchmarks/serve_tax's plugin installs",
    "verity_vllm.pipeline.boundary_tables": "I1: a two-stage window's history tables at serve: rank 0's step hook on Model "
                                            "Runner V2 (StepHook), the builder and its channel to the boundary commitment "
                                            "(Builder, Channel), and the rows and salts kept on the node for round 2 (Kept), "
                                            "which benchmarks/serve_tax's plugin installs",
    "verity_vllm.pipeline.interior_rows": "I1: a two-stage window's interior at replay: the sampled spans' registration "
                                          "(interior, register), the block parameters it registers as values, the "
                                          "verifier's (Supply) and the replay's own (Operand), and the challenger's "
                                          "copies of them (span_values, link_words, link_schemas, sources, history), which "
                                          "benchmarks/one_stage/two_stage.py registers",
    "verity_vllm.pipeline.serve_h1": "I1: serve-openai's PoUW H1 arm, its --protocols-config (config_of) and the window "
                                     "Program it declares (H1Windows), which a period's offline Program must equal",
    "verity_vllm.pipeline.serve_tie": "I1: serve-openai's H1 tie, and where a serve dir holds each window's registration W, "
                                      "its record R and their log (W_FILE, R_FILE, LOG)",
    "verity_vllm.pipeline.cut_plugin": "I1: the carried pool's cut at each window edge under vLLM's own OpenAI server: its "
                                       "general plugin (register, an edge of the engine core's seam) and ASGI middleware "
                                       "(SentFeed), configured by VERITY_CUT, which the trace corpus's cut traces serve under",
    "verity_vllm.pipeline.serve_pearl": "I1: serve-openai's Pearl+ arm (PearlWindows), its --protocols-config (config_of, "
                                        "CONFIG) and where a serve dir holds each window's epoch (DIR)",
    "verity_vllm.pipeline.tick_plugin": "I1: the certified tick (--admit tick) under vLLM's own OpenAI server: its general "
                                        "plugin (register) and ASGI middleware (TickFeed), configured by VERITY_TICK, which "
                                        "the trace corpus serves under",
    "verity_vllm.pipeline.advice_code.codec": "I1: advice v1, the network certifier's advice code: its certified record, its "
                                              "registration, and I1 v0's encode and decode against the scheduler replay, which "
                                              "the compiler reads and benchmarks/network_e2e measures",
    "verity_vllm.pipeline.advice_code.schedule": "I1: the scheduler replay's engine config, certified tick, steps and request "
                                                 "states",
    "verity_vllm.pipeline.advice_code.pool": "I1: the replay of vLLM's KV block pool, which the certifier's cache vectors are "
                                             "held to",
    "verity_vllm.pipeline.advice_code.replay": "I1: the replay of vLLM's scheduler, which the certifier's replay vectors and "
                                               "the traces' admission are held to",
    "verity_vllm.pipeline.advice_code.derived": "I1: advice v1's derived rules (the walk, the prefixes, the tie), which the "
                                                "certifier's derived vectors are held to",
    "verity_vllm.pipeline.advice_code.window_code": "I1: advice v1's window code and its codebook",
    "verity_vllm.pipeline.advice_code.arith": "I1: the arithmetic coder advice v1's window code runs on",
    "verity_vllm.pipeline.advice_code.by_class": "I1: advice v1's charge by class: each window's bits split by the field "
                                                 "that put them there, and the fields a registration holds at 0 with "
                                                 "the rules that do, which benchmarks/network_e2e reports",
    "verity_vllm.pipeline.advice_code.schedule_code": "I1: advice v1 to and from the replay's schedule, its entry for a "
                                                      "serve with a draft, which no I1 v0 says and "
                                                      "benchmarks/network_e2e holds to vLLM's traces",
    "verity_vllm.pipeline.advice_code.kept": "I1: advice v1's kept counts, the drafts each emitting row kept: their code "
                                             "in a window's second segment, and each step's emissions read off the replay",
    "verity_vllm.protocols.pouw.h1.commit": "I1: a served H1 window's Commit: its registration W (registration), the audited "
                                            "window's tables rebuilt as W's (reuse, reused) and the record H1 names (named)",
    "verity_vllm.protocols.pouw.pearl_c.window": "I1: Pearl+'s window record W over a run's retained timed rounds (record, write), "
                                                 "the names it registers each call's A rows and y under (a_value, y_value), "
                                                 "each weight's B rows at PoUW's weight root (b_entries, b_value), "
                                                 "built at the version W states (version_of, VERSION), "
                                                 "which verify_run.tie and the one-stage audit read, the audited "
                                                 "window's values rebuilt as W's (reuse), and the served run's served.json "
                                                 "W is written from (SERVED_SCHEMA, KEEP, served_passes, macs, coins_of)",
    "verity_vllm.protocols.pouw.h1.words": "I1: the words H1 commits a call's rows and a step's input ids as (chunk_words, "
                                           "ids_words)",
    "verity_vllm.protocols.pouw.h1.drawn": "I1: a served H1 window's drawn calls staged as C-Flock statements and checked as "
                                           "credit does (main), which benchmarks/pouw/h1_prove/drawn.py runs",
    "verity_vllm.protocols.pouw.h1.answers": "I1: the verifier of record's answers on a window's statements, each run once "
                                             "as credit asks for it and shared by key by the steps that judge them "
                                             "(Answers), which drawn.py check and pouw_piece.py period read",
    "verity_vllm.protocols.pouw.h1.receipts": "I1: served H1's receipt producer, the zy receipts serve-openai's "
                                              "--h1-receipts registers (Receipts) and its noise kernel, which PoUW's "
                                              "benchmarks time and gate",
    "verity_vllm.protocols.pouw.pearl_c.served_y": "I1: Pearl+'s served y as a retained pass commits it, a call's real rows "
                                                   "under its domain (y_domain, commit_y, retain_y) and on its device "
                                                   "(DeviceTrees), which verify_run opens on each drawn tile",
    "verity_vllm.acquire.sources.served_compiled_source": "I2: the values of a compiled engine at vLLM's serving defaults "
                                                          "(torch.compile and CUDA graphs), acquired with no Python hook inside "
                                                          "graph replay (ServedCompiledSource, linkage), which "
                                                          "benchmarks/served_commit and the harness's Commit arm time",
    "verity_vllm.check.oracle_compare": "I7: the served C2's value correspondence by oracle compare, and the value checks' one "
                                        "byte source, an opening verified against the run commitment of record before its bytes "
                                        "are returned (OpenedReader), which benchmarks/one_stage and benchmarks/prove_path read a "
                                        "Commit through",
    "verity_vllm.commit.committer.native_host": "I7: the reference committer of the values a served engine materialises, hashed "
                                                "on the host (NativeHostCommitter), its GPU chunk size and its bound on a "
                                                "schedule's stream (schedule_stream_bound), which benchmarks/served_commit, the "
                                                "harness's tests and benchmarks/prove_path's window cost read",
    "verity_vllm.commit.committer.native_jit": "I7: the native collector's JIT build, keyed by what it compiles, and its CUDA "
                                               "flags (CUDA_CFLAGS), which benchmarks/served_commit's rows floor compiles under",
    "verity_vllm.commit.index": "I7: vLLM's labels of a window's units (index.json): per member, each served row's unit and "
                                "served identity (index_doc), which benchmarks/one_stage's A5 runs write",
    "verity_vllm.commit.row_tables": "I7: the row tables of a serving-rows window as serving commits them step by step "
                                     "(step_roots, StepRoots, table_key), which benchmarks/commitments times and "
                                     "benchmarks/one_stage registers",
    "verity_vllm.commit.row_tables_device": "I7: a step's row tables hashed on the GPU on a side stream, and their host twin "
                                            "(step_roots, host_step, plan, DeviceLibrary), which benchmarks/commitments times and "
                                            "benchmarks/serve_tax's plugin installs",
    "verity_vllm.commit.scheme": "I7: the Commit path's vllm-v1 commitment scheme: its leaves, folds and roots (pos_leaf, "
                                 "POS_LEAF_TAG), which benchmarks/commitments prices",
    "verity_vllm.commit.step_rows": "I7: the Commit's rows-sha512 scheme over a step's stream: its row and key sizes, domains, "
                                    "levels and roots (ROW_BYTES, KEY_BYTES, tensor_root, keyed_tensor_root, run_program_digest), "
                                    "which benchmarks/served_commit times and benchmarks/serve_tax and benchmarks/one_stage "
                                    "commit under",
    "verity_vllm.commit.store_dump": "I7: a GPU-tree committer's committed store on disk, and back as a CPU store that opens and "
                                     "verifies like the committer (dump, load), which benchmarks/prove_path's served tie reads",
    "verity_vllm.correspondence.reader_for_query": "I7: the per-Call runtime correspondence of a derived Program for the query, "
                                                   "and a Call's name (call_name_of, parse_call_name), which "
                                                   "benchmarks/served_commit's granularity reads",
    "verity_vllm.correspondence.runtime": "I7: runtime-correspondence/v1: per Call, the facts that resolve a Program's logical "
                                          "value to a runtime occurrence (RuntimeCorrespondence), which "
                                          "benchmarks/served_commit's granularity reads",
    "verity_vllm.llm": "I1: the pinned engine's own module: LLM, the protocol options it resolves and the context it hands them "
                       "(resolve_options, protocol_context), which the harness's tests hold its engine to",
    "verity_vllm.pipeline.commit_coins": "I1: the Commit's coins from the auditor host's challenger, one challenge phase a Commit "
                                         "(ChallengerGate), which benchmarks/one_stage's local Pearl+ run and benchmarks/pouw's "
                                         "Pearl-C arm take their coins from",
    "verity_vllm.pipeline.replay_bundle": "I1: the replay bundle a Commit with --replay-deferred leaves for its C2 replay on CPU "
                                          "(MANIFEST, open_bundle), which benchmarks/one_stage opens",
    "verity_vllm.pipeline.tp.rank_replay": "I1: the C2 of record on one rank of a TP Commit, and a rank's manifest (load, "
                                           "manifest_of_rank), which benchmarks/one_stage reads",
    "verity_vllm.program.model": "I7: what a model configuration is, and the roots that are functions of it (config_of, "
                                 "cos_sin_table, INT8_EXACT_QUANT), which benchmarks/prove_path and benchmarks/inference_only "
                                 "build under",
    "verity_vllm.protocols": "I1: the protocol options of a vLLM run (PoUW, PoUS and SPOT, each enabled on its own): "
                             "their selector, context, install and refusals (parse, adapter, Context, install, Refused, "
                             "requires_build, commit_guard), which the harness's arms and the served benchmarks of PoUW, PoUS and "
                             "the Commit install",
    "verity_vllm.protocols.interface": "I1: what every protocol option implements, and what the selector hands it (Option, "
                                       "Context, Target)",
    "verity_vllm.protocols.linear.api": "I1: the linear-site interface: what computes y at a linear layer, where its weight comes "
                                        "from, and the one object vLLM's quant_method.apply calls (Stack, Arm, Control, PassSpec, "
                                        "LinearSite), which benchmarks/pouw's Pearl-C arms build",
    "verity_vllm.protocols.linear.vllm_site": "I1: the linear sites of a vLLM model, and a Stack attached to them (discover, "
                                              "attach), which benchmarks/pouw's Pearl-C arms attach",
    "verity_vllm.protocols.pouw.h1.handoff": "I1: an H1 window handed from its serving process to a CPU process for its close, "
                                             "and the file it is saved in (FILE), which benchmarks/pouw/h1_lanes reads",
    "verity_vllm.protocols.pouw.h1.kept": "I1: a row of a served H1 window's registered tree, rebuilt from what the prover kept "
                                          "(kept_row), which benchmarks/pouw/h1_lanes opens",
    "verity_vllm.protocols.pouw.h1.prover": "I1: an H1 window's private prover state: what opening a sampled proof block's "
                                            "committed values needs after the window's close (STATE), which "
                                            "benchmarks/pouw/h1_lanes reads",
    "verity_vllm.protocols.pouw.h1.registered": "I1: H1's served tables registered at serving scale as hm96-sha512 bit rows "
                                                "(commit, MODE), which benchmarks/pouw's producer test and "
                                                "benchmarks/prove_path's tests commit under",
    "verity_vllm.protocols.pouw.h1.served": "I1: H1's served tables, committed as registered values of compute accounting's "
                                            "window Program as the window serves them (Served, declared, declared_layout, "
                                            "write_program, nonce), which benchmarks/one_stage's served verdicts, "
                                            "benchmarks/prove_path and benchmarks/pouw read",
    "verity_vllm.protocols.pouw.pearl_c.grouped": "I1: Pearl-C's grouped call for a fused-MoE layer on an H100 or H200, every "
                                                  "routed expert's rows of one layer in one launch a step (gemm_grouped, group, "
                                                  "module, ROTATED), which benchmarks/pouw/pearl_c_sm90 checks and times",
    "verity_vllm.protocols.pouw.pearl_c.manifest": "I1: the Pearl-C device path's manifest as a verifier reads it, and a retained "
                                                   "pass's execution phase re-started from it (manifest_epoch, session_head), "
                                                   "which benchmarks/pouw/pearl_c_vllm's screen reads",
    "verity_vllm.protocols.pouw.pearl_c.whole": "I1: Pearl-C's decode forward replayed as one CUDA graph (WholeStep), which "
                                                "benchmarks/pouw/pearl_c_vllm's arms run",
    "verity_vllm.protocols.remote_bandwidth.binder": "I1: the store's hooks into one in-process vLLM engine (Binder) and the "
                                                     "weights it binds (model_weights, WeightRegion, Every), which "
                                                     "benchmarks/remote_session and benchmarks/vllm_encoded serve under",
    "verity_vllm.protocols.remote_bandwidth.registry": "I1: the node's store as one registry (NodeMemory), the reference "
                                                       "implementation of StoreView and Lifecycle, which "
                                                       "benchmarks/remote_session registers a served engine's weights and "
                                                       "KV pool in",
    "verity_vllm.query.manifest.format": "I7: the required-value manifest's format, the vocabulary every manifest producer and "
                                         "consumer shares (load_program), which benchmarks/prove_path reads",
}

#: Every other module of the package that outside code imports. Move its importers onto a public interface and delete it.
KNOWN = frozenset({
    "verity_vllm.acquire.sources.attention_source",
    "verity_vllm.acquire.sources.taps",
    "verity_vllm.check.checkpoint_index",
    "verity_vllm.check.fold_compare",
    "verity_vllm.check.fp8_ue8m0",
    "verity_vllm.check.replay.evaluate",
    "verity_vllm.check.replay.index",
    "verity_vllm.check.replay.opening",
    "verity_vllm.check.weights_of_record",
    "verity_vllm.commit.committer.host_retention",
    "verity_vllm.commit.committer.native_collect",
    "verity_vllm.commit.opened",
    "verity_vllm.engine.arrivals",
    "verity_vllm.engine.build",
    "verity_vllm.engine.engine_driver",
    "verity_vllm.engine.engine_profile",
    "verity_vllm.engine.env",
    "verity_vllm.engine.hooks",
    "verity_vllm.engine.int8_attention",
    "verity_vllm.engine.int8_exact",
    "verity_vllm.engine.int8_fused",
    "verity_vllm.engine.int8_ops",
    "verity_vllm.engine.int8_ranks",
    "verity_vllm.engine.int8_vllm",
    "verity_vllm.engine.profiles.dense_generic",
    "verity_vllm.engine.profiles.engine_classes",
    "verity_vllm.engine.served",
    "verity_vllm.engine.vllm_adapter",
    "verity_vllm.observe.events",
    "verity_vllm.observe.fold.fold",
    "verity_vllm.observe.fold.patterns.pouw",
    "verity_vllm.observe.log",
    "verity_vllm.pipeline.advice",
    "verity_vllm.pipeline.advice_windows",
    "verity_vllm.pipeline.build_instances",
    "verity_vllm.pipeline.c2_replay",
    "verity_vllm.pipeline.chunk_rows",
    "verity_vllm.pipeline.commit",
    "verity_vllm.pipeline.cpu_replay",
    "verity_vllm.pipeline.deferred_rows",
    "verity_vllm.pipeline.serve_admit",
    "verity_vllm.pipeline.serve_openai",
    "verity_vllm.pipeline.serve_protocol",
    "verity_vllm.pipeline.serving_rows",
    "verity_vllm.pipeline.slice_rows",
    "verity_vllm.pipeline.tp.capture",
    "verity_vllm.pipeline.tp.commit_replay",
    "verity_vllm.pipeline.tp.phases",
    "verity_vllm.pipeline.tp.rank_worker",
    "verity_vllm.pipeline.tp.serving_rows",
    "verity_vllm.pipeline.tp.target",
    "verity_vllm.pipeline.tp.trace",
    "verity_vllm.pipeline.tp.warmup",
    "verity_vllm.pipeline.window_advice",
    "verity_vllm.program.compact",
    "verity_vllm.program.frontend.derive",
    "verity_vllm.program.frontend.derive_examples",
    "verity_vllm.program.frontend.kernel_readback",
    "verity_vllm.program.frontend.rules.family",
    "verity_vllm.program.frontend.targets",
    "verity_vllm.program.predict.batch",
    "verity_vllm.program.predict.build",
    "verity_vllm.program.predict.common",
    "verity_vllm.program.predict.phi",
    "verity_vllm.program.predict.score",
    "verity_vllm.program.predict.steps",
    "verity_vllm.program.predict.window_circuit",
    "verity_vllm.protocols.pouw",
    "verity_vllm.protocols.pouw.h1",
    "verity_vllm.protocols.pouw.h1.challenger",
    "verity_vllm.protocols.pouw.h1.window",
    "verity_vllm.protocols.pouw.pearl_c",
    "verity_vllm.protocols.pouw.pearl_c.graphs",
    "verity_vllm.protocols.pouw.pearl_c.host",
    "verity_vllm.protocols.pouw.pearl_c.lanes",
    "verity_vllm.protocols.pouw.pearl_c.screen",
    "verity_vllm.protocols.remote_bandwidth.remote",
    "verity_vllm.query.program_view",
})

#: Every private name outside code takes from a module of the package, by an import or through the module. Move its
#: readers off it and delete it.
KNOWN_PRIVATE = frozenset({
    ('verity_vllm.check.fold_compare', '_NP'),
    ('verity_vllm.check.fold_compare', '_attach_specs'),
    ('verity_vllm.check.fold_compare', '_component_ranges'),
    ('verity_vllm.check.fold_compare', '_in_place'),
    ('verity_vllm.check.replay.evaluate', '_Unresolved'),
    ('verity_vllm.check.replay.evaluate', '_evaluate_row'),
    ('verity_vllm.check.replay.evaluate', '_weight_runs'),
    ('verity_vllm.pipeline.serving_rows', '_bind'),
    ('verity_vllm.pipeline.serving_rows', '_operand'),
    ('verity_vllm.pipeline.serving_rows', '_read_s8'),
    ('verity_vllm.pipeline.tp.commit_replay', '_of_record'),
    ('verity_vllm.program.frontend.rules.family', '_field_intervals'),
    ('verity_vllm.protocols.pouw.pearl_c', '_Keys'),
    ('verity_vllm.protocols.pouw.pearl_c', '_Uploads'),
    ('verity_vllm.protocols.pouw.pearl_c', '_bytes'),
    ('verity_vllm.protocols.pouw.pearl_c', '_constant_keys'),
    ('verity_vllm.protocols.pouw.pearl_c', '_constants'),
    ('verity_vllm.protocols.pouw.pearl_c', '_pipeline'),
    ('verity_vllm.protocols.pouw.pearl_c', '_steps'),
    ('verity_vllm.protocols.pouw.pearl_c', '_write'),
    ('verity_vllm.protocols.pouw.pearl_c', '_zeros'),
    ('verity_vllm.protocols.pouw.pearl_c.graphs', '_event'),
    ('verity_vllm.protocols.pouw.pearl_c.host', '_Driver'),
    ('verity_vllm.protocols.pouw.pearl_c.host', '_Ring'),
    ('verity_vllm.protocols.pouw.pearl_c.host', '_event'),
    ('verity_vllm.protocols.pouw.pearl_c.host', '_write_host'),
    ('verity_vllm.protocols.pouw.pearl_c.lanes', '_event'),
    ('verity_vllm.protocols.pouw.pearl_c.screen', '_COMPILED'),
})


@dataclass(frozen=True)
class Site:
    file: str
    line: int
    module: str
    name: str | None


def _module_file(module: str) -> Path | None:
    base = PACKAGE_DIR.parent.joinpath(*module.split("."))
    for p in (base.with_suffix(".py"), base / "__init__.py"):
        if p.is_file():
            return p
    return None


def _called(call: ast.Call) -> str | None:
    return call.func.attr if isinstance(call.func, ast.Attribute) else getattr(call.func, "id", None)


def _import_call(node: ast.AST | None) -> str | None:
    """The module of the package a call such as `importlib.import_module("verity_vllm.llm")` names."""
    if (isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)
            and _called(node) in IMPORT_CALLS and node.args[0].value.split(".")[0] == PACKAGE):
        return node.args[0].value
    return None


def _reach(tree: ast.Module) -> tuple[dict[ast.AST, ast.AST], Callable[[ast.AST, list[str]], tuple[str | None, list[str]]]]:
    """Each node's parent, and `reach(expression, names)`: the module of the package that `expression.names...` reaches,
    with the names it takes past that module, or None when the expression's first name is no module of the package there.
    A name is a module where the nearest scope binding it binds it by an import of the package (an import in a scope
    wins over the scope's other bindings of the name)."""
    parent = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    bound: dict[ast.AST, dict[str, str | None]] = {}       # scope -> name -> the module it is there, None for any other binding

    def scope(node: ast.AST) -> ast.AST:
        node = parent[node]
        while not isinstance(node, SCOPES):
            node = parent[node]
        return node

    def bind(node: ast.AST, name: str, module: str | None = None) -> None:
        names = bound.setdefault(scope(node), {})
        names[name] = module or names.get(name)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                top = a.name.split(".")[0]
                bind(node, a.asname or top, (a.name if a.asname else top) if top == PACKAGE else None)
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                sub = f"{node.module}.{a.name}"
                bind(node, a.asname or a.name, sub if node.level == 0 and sub.split(".")[0] == PACKAGE and _module_file(sub) else None)
        elif isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            holder = parent[node]
            bind(node, node.id, _import_call(holder.value) if isinstance(holder, (ast.Assign, ast.AnnAssign, ast.NamedExpr)) else None)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bind(node, node.name)
        elif isinstance(node, ast.arg):
            bind(node, node.arg)
        elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name:
            bind(node, node.name)

    def reach(node: ast.AST, names: list[str]) -> tuple[str | None, list[str]]:
        while isinstance(node, ast.Attribute):
            names = [node.attr, *names]
            node = node.value
        if not isinstance(node, ast.Name):
            return None, names
        at = scope(node)
        while node.id not in bound.get(at, {}) and not isinstance(at, ast.Module):
            at = scope(at)
        module = bound.get(at, {}).get(node.id)
        while module and names and _module_file(f"{module}.{names[0]}"):
            module, names = f"{module}.{names[0]}", names[1:]
        return module, names

    return parent, reach


def _sites_in(path: str, text: str) -> list[Site]:
    out = []
    ours = lambda m: m.split(".")[0] == PACKAGE   # noqa: E731
    tree = ast.parse(text, filename=path)
    parent, reach = _reach(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [Site(path, node.lineno, a.name, None) for a in node.names if ours(a.name)]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module and ours(node.module):
            for a in node.names:
                sub = f"{node.module}.{a.name}"
                out.append(Site(path, node.lineno, sub, None) if _module_file(sub) else Site(path, node.lineno, node.module, a.name))
        elif _import_call(node):
            out.append(Site(path, node.lineno, _import_call(node), None))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and MODULE_ATTR.fullmatch(node.value) and ours(node.value):
            module, _, name = node.value.partition(":")
            out.append(Site(path, node.lineno, module, name))
        elif isinstance(node, ast.Attribute) and not isinstance(parent[node], ast.Attribute):
            module, names = reach(node, [])
            if module and not (names and DUNDER.fullmatch(names[0])):
                out.append(Site(path, node.lineno, module, names[0] if names else None))
        elif (isinstance(node, ast.Call) and _called(node) in NAME_CALLS and len(node.args) > 1 and isinstance(node.args[1], ast.Constant)
              and isinstance(node.args[1].value, str)):
            module, names = reach(node.args[0], [node.args[1].value])
            if module and len(names) < 2 and not (names and DUNDER.fullmatch(names[0])):
                out.append(Site(path, node.lineno, module, names[0] if names else None))
    return out


@functools.cache
def sites() -> tuple[Site, ...]:
    """Every import of the package by a tracked Python file outside the integration."""
    files = subprocess.run(["git", "grep", "-l", "-F", PACKAGE, "--", "*.py", f":!{INTEGRATION}"], cwd=ROOT, capture_output=True,
                           text=True, check=True).stdout.split()
    return tuple(s for f in files if f not in NOT_SCANNED for s in _sites_in(f, (ROOT / f).read_text(encoding="utf-8")))


@functools.cache
def exported(module: str) -> tuple[str, ...]:
    """A public module's `__all__`, as the literal it states."""
    for node in ast.parse(_module_file(module).read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            value = ast.literal_eval(node.value)
            assert isinstance(value, (list, tuple)) and all(isinstance(n, str) for n in value), f"{module}.__all__ is no list of names"
            return tuple(value)
    raise AssertionError(f"{module} is in PUBLIC but states no __all__: list the names outside code may import from it")


def _definitions(module: str) -> dict[str, tuple[str, ast.AST | None]]:
    """name -> (where it is bound: "top", "type-checking" or "lazy", its def or class node when this module defines it, or the
    module it imports it from)."""
    path = _module_file(module)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: dict[str, tuple[str, ast.AST | str | None]] = {}
    package = module.split(".") if path.name == "__init__.py" else module.split(".")[:-1]

    def bind(nodes, where):
        for n in nodes:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                out[n.name] = (where, n)
            elif isinstance(n, (ast.Assign, ast.AnnAssign)):
                for t in n.targets if isinstance(n, ast.Assign) else [n.target]:
                    out.update({x.id: (where, None) for x in ast.walk(t) if isinstance(x, ast.Name)})
            elif isinstance(n, ast.ImportFrom):
                src = n.module if n.level == 0 else ".".join([*package[:len(package) + 1 - n.level], *filter(None, [n.module])])
                out.update({a.asname or a.name: (where, f"{src}.{a.name}") for a in n.names})
            elif isinstance(n, ast.If) and "TYPE_CHECKING" in ast.unparse(n.test):
                bind(n.body, "type-checking")

    bind(tree.body, "top")
    lazy = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "__getattr__"), None)
    if lazy is not None:
        text = ast.unparse(lazy) + "\n" + "\n".join(ast.unparse(n) for n in tree.body if isinstance(n, ast.Assign))
        out.update({k: ("lazy", v[1]) for k, v in out.items() if v[0] == "type-checking" and repr(k) in text})
    return out


def _untyped(fn: ast.FunctionDef | ast.AsyncFunctionDef, method: bool) -> list[str]:
    a = fn.args
    params = [*a.posonlyargs, *a.args, *a.kwonlyargs, *([a.vararg] if a.vararg else []), *([a.kwarg] if a.kwarg else [])]
    static = any(isinstance(d, ast.Name) and d.id == "staticmethod" for d in fn.decorator_list)
    if method and not static and params:
        params = params[1:]
    return [p.arg for p in params if p.annotation is None] + (["return"] if fn.returns is None else [])


def _resolve(module: str, name: str) -> tuple[str, ast.AST | None]:
    """Where `name`, as `module` binds it, is defined: (module, its def or class node, None for any other binding)."""
    where, what = _definitions(module).get(name, (None, None))
    if isinstance(what, str):
        parent, _, leaf = what.rpartition(".")
        if _module_file(what) or not _module_file(parent):
            return what, None                                # a module, or a name from outside the package: not ours to annotate
        return _resolve(parent, leaf)
    return module, what


def untyped(module: str) -> list[str]:
    """`module.name(param)` for each parameter or return of the module's exported functions and public methods without an
    annotation."""
    out = []
    for name in exported(module):
        home, node = _resolve(module, name)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out += [f"{home}.{name}({p})" for p in _untyped(node, method=False)]
        elif isinstance(node, ast.ClassDef):
            for m in node.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and (not m.name.startswith("_") or m.name in ("__init__", "__call__")):
                    out += [f"{home}.{name}.{m.name}({p})" for p in _untyped(m, method=True)]
    return out


def test_each_public_module_states_binds_and_types_its_interface():
    for module in PUBLIC:
        assert _module_file(module), f"PUBLIC names {module}, which is no module of {INTEGRATION}{PACKAGE}"
        defs = _definitions(module)
        unbound = [n for n in exported(module) if defs.get(n, ("type-checking",))[0] == "type-checking" and not _module_file(f"{module}.{n}")]
        assert not unbound, (f"{module}.__all__ lists names it doesn't bind at top level (or load in its __getattr__ when "
                             f"bound only under TYPE_CHECKING): {unbound}")
    missing = [u for module in PUBLIC for u in untyped(module)]
    assert not missing, "a public interface is typed: annotate\n  " + "\n  ".join(missing)


def test_outside_code_imports_only_public_names_and_known_modules():
    bad = []
    for s in sites():
        if s.module in PUBLIC:
            if s.name is not None and s.name not in exported(s.module):
                bad.append(f"{s.file}:{s.line}: {s.name} is not in {s.module}.__all__ (import a listed name, or list it there "
                           "if it is part of that interface)")
        elif s.module not in KNOWN:
            public = ", ".join(sorted(PUBLIC))
            bad.append(f"{s.file}:{s.line}: imports {s.module}, which is not a public interface of {PACKAGE}: import from one of "
                       f"{public} (or grow the interface the import needs, with __all__ and annotations, and add it to PUBLIC in "
                       "tests/test_vllm_public_api.py); KNOWN only shrinks")
        if s.name is not None and s.name.startswith("_") and (s.module, s.name) not in KNOWN_PRIVATE:
            bad.append(f"{s.file}:{s.line}: takes {s.module}.{s.name}, a private name: use a public one")
    assert not bad, "\n".join(bad)


def test_the_allowlists_are_exact():
    reached = {s.module for s in sites()} - set(PUBLIC)
    private = {(s.module, s.name) for s in sites() if s.name is not None and s.name.startswith("_")}
    gone = sorted(KNOWN - reached)
    gone_private = sorted(KNOWN_PRIVATE - private)
    assert not (gone or gone_private), ("nothing outside the integration imports these any more; delete them from tests/"
                                        "test_vllm_public_api.py:\n" + "\n".join([*(f'    "{m}",' for m in gone),
                                                                                *(f"    {p!r}," for p in gone_private)]))
    assert not KNOWN & set(PUBLIC), f"a public module is not also KNOWN: delete {sorted(KNOWN & set(PUBLIC))} from KNOWN"


def test_the_scan_sees_each_form_and_the_tree():
    text = ("import verity_vllm.engine.handle\nfrom verity_vllm import LLM, protocols\nfrom verity_vllm.config import _x\n"
            "importlib.import_module('verity_vllm.llm')\npytest.importorskip('verity_vllm.config')\nX = 'verity_vllm.llm:LLM'\n"
            "import verity_flock\nM = ModuleType('verity_vllm.not_an_import')\n")
    got = {(s.module, s.name) for s in _sites_in("x.py", text)}
    assert got == {("verity_vllm.engine.handle", None), ("verity_vllm", "LLM"), ("verity_vllm.protocols", None),
                   ("verity_vllm.config", "_x"), ("verity_vllm.llm", None), ("verity_vllm.config", None), ("verity_vllm.llm", "LLM")}
    assert ("benchmarks/harness/specs/tests/test_engine_spec_presets.py", "verity_vllm", "EnginePin") in {(s.file, s.module, s.name) for s in sites()}
    assert not any(s.file.startswith(INTEGRATION) for s in sites())
    assert all((ROOT / f).is_file() for f in NOT_SCANNED)


def test_the_scan_sees_a_name_taken_through_a_module():
    text = ("import verity_vllm.engine.handle\nimport verity_vllm.config as c\nfrom verity_vllm import engine as e\n"
            "m = importlib.import_module('verity_vllm.llm')\n"
            "c.a.b()\nc._b = 1\ndel c.c\ngetattr(c, 'd')\nmonkeypatch.setattr(c, '_e', 2)\nmock.patch.object(c, 'f')\n"
            "hasattr(c, 'g')\nc.__name__\nprint(c)\ngetattr(c.a, 'h')\n"
            "e.handle.i\ngetattr(e, 'spec')\nverity_vllm.protocols\nverity_vllm.engine.handle.j\nverity_vllm.LLM\nm.k\n"
            "def f(c, m=1):\n    c.l\n    return [e.n for e in c] + [m.o for _ in c]\n"
            "def g():\n    c.p = 1\n    import verity_vllm.check as m\n    return m.q\n")
    names = {("verity_vllm.config", n) for n in ("a", "_b", "c", "d", "_e", "f", "p")}
    assert {(s.module, s.name) for s in _sites_in("x.py", text)} == names | {
        ("verity_vllm.engine.handle", None), ("verity_vllm.config", None), ("verity_vllm.engine", None), ("verity_vllm.llm", None),
        ("verity_vllm.engine.handle", "i"), ("verity_vllm.engine.spec", None), ("verity_vllm.protocols", None),
        ("verity_vllm.engine.handle", "j"), ("verity_vllm", "LLM"), ("verity_vllm.llm", "k"), ("verity_vllm.check", None),
        ("verity_vllm.check", "q")}


def test_a_name_from_outside_the_package_is_not_ours_to_annotate(tmp_path, monkeypatch):
    package = tmp_path / PACKAGE
    package.mkdir()
    (package / "a.py").write_text("from typing import Any\nfrom verity_vllm.b import f, g\nfrom verity_vllm import b\n", encoding="utf-8")
    (package / "b.py").write_text("from os import getcwd as g\ndef f(x): ...\n", encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "PACKAGE_DIR", package)
    assert [_resolve("verity_vllm.a", n)[0] for n in ("Any", "g", "b", "f")] == ["typing.Any", "os.getcwd", "verity_vllm.b", "verity_vllm.b"]
    assert _untyped(_resolve("verity_vllm.a", "f")[1], method=False) == ["x", "return"]
