"""The Commit scheme and the sealing of `verity_vllm.pipeline.serve_openai`'s windows.

`committer` builds the server's committer: ``--scheme rows-sha512`` (the default) the declared Commit of record's
(`pipeline.commit.make_committer`'s ``native_collect_v2b``: hm96-sha512 rows in frame-v3-sha512 trees, 64-byte roots) with its
acquisition built as the Commit builds it (`pipeline.commit_attach.attach_of_record`) over the bench row's required manifest
(``--required-manifest``): the attention hidden tap (``--hidden-so``), the norm-scale tap (``--norm-tap-so``) and the hook sets of
the acquisition plan, so its name is the Commit's (``...+hidden_m1+norm_scale+plan``) and a part the manifest names that did not
attach is refused (`parts_missing`); ``--scheme vllm-v1`` `NativeHostCommitter` (SHA-256, 32-byte roots), which only a recorded
run's readers take, so a run under it names the recorded vllm-v1 run it serves again (``--of-record``, `scheme_refusal`). Then
`served_record.StepInputPort` (each step's
stream starts with its input ids, then its positions, each padded to ``--max-num-batched-tokens``: the port the certifier's
served-rows tie reads) and `served_record.SamplerInputsSource` are attached and the arena sized; under ``--seal`` it registers
the weights. The port is the step's one copy of its ids (`single_ids`): the embedding's own input isn't committed beside it.
It writes the engine's rotary table to ``OUT/private/cos_sin_cache.npy`` (`m1_capture.dump_cos_sin`), which no record copies:
the words the committer's weights registration roots it at (``prove_path.py weights-registration``) before the first request.

With ``seal`` (`seal_window`, which `serve_commit.CommitWindows` calls at each close), each closed window is sealed as a
deferred Commit's row dir, ``windows/w<k>/``, the way a declared Commit with ``--replay-deferred`` seals its run
(`pipeline.replay_bundle`): ``commit/replay_bundle_p0/`` (the committed store, the weights tree, the binding map, the context,
the manifest), and beside it the record that bundle is opened against (`deferred_rows.of_record`): ``commit/verdict.json``
naming the bundle, ``commit/runs.jsonl`` with the window's run, and ``commit/binding_map_p0.json``. The binding map names
each committed value as the row's Programs read it (`served_binder`: a Linear's ``out``, RoPE's ``query`` / ``key``, a norm's
``norm_scale``, ``fa2.m1/*``, and the port's ids and positions as the identities the Programs read them under,
`port_identities`), with no second tensor under an identity the port is (`copies_beside_port`) and no port position other
than its rows ran (`positions_off`), and the context carries the manifest's ``op_path_aliases``, which the opened store resolves a
shared module's layer-N path through, the window's served-rows record (`serve_commit.CommitWindows`' ``served_rows``) and the
step-input port (`port_of`; `port_layout` is where the window's streams put it). The window's requests are named by the body's ``user`` (the id the advice names them
by), and the bundle names no Programs: a served window's Programs are the registered compiler's for its advice, made after
the window closed. With ``--retain-exclude hidden_m1`` (`RetentionOptions`) the attention interiors (``fa2.m1/*``, ``fa3.m1/*``)
are committed as before, same leaves and roots, but not copied off the device, so the bundle's store doesn't hold them and its
manifest names them ``committed_not_retained``: no reader of a sealed window opens them (no Program of record reads an
``fa*.m1`` identity, and C2 and the oracle compare skip them), and a read of one is refused by name.

A window is sealed off the engine loop (`Sealer`), over a view of the committer that holds the window's own per-step state,
with the pinned blocks its values sit in lent out of the committer's pool until the seal has read them (`thaw`): the view
the window was detached into at its close (`NativeHostCommitter.detach`), finalized on the sealer's first thread before its
seal, or, for a committer that detaches no window, the committer frozen after a finalize at the close (`freeze`). The
registered tensors' bytes (`extras_of`) are read once for each weights root, and `seal_window` runs over the view on the
sealer's second thread, beside the next window's finalize, while the engine serves the window after; the bytes it writes are
the ones an inline seal writes. `DiskFloor` keeps
the serve dir's filesystem above a floor: a window whose seal would take it under is not sealed.

A seal on the sealer's thread shares the engine's GIL, so it holds it only briefly: its binding map is encoded a chunk of
entries at a time (`commit.json_chunks`), built with no cyclic collection running (`gc_paused`) and freed a chunk at a time
(`drop`). By default its Python runs in a spawned process (`seal_process.ProcessSealer`), over a view it is handed (`SealView`,
`seal_view`), the thread here writing only the store's files. `Watchdog` records what still stalls the process (``pauses.jsonl``),
and each seal's line its phases.
"""
from __future__ import annotations

import contextlib
import copy
import functools
import gc
import json
import shutil
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from verity_vllm.commit import json_chunks as JC
from verity_vllm.commit import step_rows
from verity_vllm.commit.served_record import PORT_NAMES, StepInputPort
from verity_vllm.config import UsageError, option

ROWS, V1 = step_rows.NAME, "vllm-v1"
#: the token budget's floor under rows-sha512: the step-input port's ids fill whole ``step_rows.ROW_BYTES`` rows
PORT_IDS = step_rows.ROW_BYTES // StepInputPort.IB
#: the server's Commit schemes, its default first
SCHEMES = (ROWS, V1)
#: what a sealed served window's context says it is (`deferred_rows.of_record` reads no Programs of record for it)
SERVED_WINDOW = "verity-vllm/served-window/v1"


#: the committer a declared Commit commits rows-sha512 with (`pipeline.commit`'s default ``--committer``)
DECLARED_COMMITTER = "native_collect_v2b"


@dataclass(frozen=True, kw_only=True)
class RetentionOptions:
    """serve-openai's ``--retain-exclude`` and ``--retain`` (its ``retention`` group, `retention_refusal`, `retention_record`)."""
    retain_exclude: str = option("--retain-exclude", env="VERITY_RETAIN_EXCLUDE", default="", metavar="FAMILIES",
                                 help="rows-sha512: commit these value families without retaining them (sets VERITY_RETAIN_EXCLUDE, as "
                                      "the Commit's --retain-exclude does): the attention interiors only, hidden_m1 (fa2_hidden_m1, "
                                      "fa3_hidden_m1), which no reader of a sealed window opens. Their leaves and roots are the "
                                      "same; each seal's manifest names them committed_not_retained (store_dump.not_retained)")
    retain: str = option("--retain", default="host", choices=("host", "device"),
                         help="rows-sha512: where a window's committed values wait for its draws. host (the default): the pinned host "
                              "pool. device: a device pool of --retain-device-gb (committer.device_pool), which the window's draw-gated "
                              "release empties, keeping on host only the leaves some subprotocol drew; a full pool makes the step wait")
    retain_device_gb: float = option("--retain-device-gb", type=float, default=32.0, metavar="GB",
                                     help="--retain device: the device pool's size, beside the engine and the window ring; it must hold "
                                          "every window awaiting its release and the open one, so at least --window-gb")
    release_after: str = option("--release-after", default="", metavar="PROTOCOLS",
                                help="--retain device --seal: the subprotocols (comma-separated, e.g. pouw) whose draw of each "
                                     "window its release waits for; the union of the leaves their drawn blocks open is kept "
                                     "(serve_release); pouw is PoUW H1's, put by its arm (--protocols), the rest come in "
                                     "through --draws-dir")
    draws_dir: str = option("--draws-dir", default="", metavar="DIR",
                            help="--release-after: where each draw comes in, w<k>.<protocol>.json (default OUT/draws)")


def retention_record(opts: Any, com: Any) -> dict[str, Any]:
    """serve.json's record of ``opts.retention``: the families left unretained, where values wait for their draws, and the pool."""
    r = opts.retention
    return {"retain_exclude": r.retain_exclude or None, "retain": r.retain, "device_pool": getattr(com, "device_pool_record", None),
            "release_after": r.release_after or None, "draws_dir": (r.draws_dir or "OUT/draws") if r.release_after else None}


def retention_refusal(opts: Any) -> str | None:
    """Why ``opts.retention`` can't hold (None: it can): ``--retain-exclude`` names attention interiors only, and the Commit under
    rows-sha512, whose committer leaves them on the device; ``--retain device`` the Commit under rows-sha512 too, with a pool of at
    least ``--window-gb``."""
    r = opts.retention
    if r.retain == "device":
        if not opts.commit or opts.scheme != ROWS:
            return f"--retain device holds the windowed collector's committed values on the device: it needs the Commit under --scheme {ROWS}"
        if r.retain_device_gb < float(getattr(opts, "window_gb", 0) or 0):
            return (f"--retain-device-gb {r.retain_device_gb:g} is under --window-gb {opts.window_gb:g}: the pool must hold the open window "
                    "beside those awaiting their release")
        if getattr(opts, "seal", False) and not getattr(r, "release_after", ""):
            return ("--retain device --seal: a window is sealed only after its draw-gated release, so name the subprotocols whose draws "
                    "it waits for (--release-after)")
    if getattr(r, "release_after", ""):
        if r.retain != "device" or not getattr(opts, "seal", False):
            return "--release-after gates the release of device-held values before each seal: it needs --retain device and --seal"
        from verity_vllm.pipeline.serve_release import protocols_of
        try:
            protocols_of(r.release_after)
        except ValueError as e:
            return str(e)
    spec = r.retain_exclude
    if not spec:
        return None
    from verity_vllm.commit.committer.host_retention import HIDDEN_M1_FAMILIES, retain_exclude_families

    if other := sorted(retain_exclude_families(spec) - HIDDEN_M1_FAMILIES):
        return (f"--retain-exclude {spec}: {other} aren't attention interiors; serve-openai commits without retaining only "
                f"{sorted(HIDDEN_M1_FAMILIES)} (hidden_m1), which no reader of a sealed window opens")
    if not opts.commit or opts.scheme != ROWS:
        return f"--retain-exclude {spec} commits values without retaining them: it needs the Commit under --scheme {ROWS}"
    return None


def recorded_scheme(rec: Mapping[str, Any]) -> str | None:
    """The scheme a recorded serve-openai run committed under, from its serve.json ``rec``: its ``scheme``, vllm-v1 for a
    committing run recorded before serve.json named one, None for a run that committed nothing."""
    return rec.get("scheme", V1) if rec.get("commit") else None


def scheme_refusal(opts: Any) -> str | None:
    """Why ``opts`` can't commit under its scheme (None: it can). rows-sha512 needs the GPU tree, the bench row's manifest and
    a token budget of whole step-input port rows (`PORT_IDS`). vllm-v1 commits SHA-256, which only a recorded run's readers
    take, so it only serves again a recorded vllm-v1 run of the same model, the one ``--of-record`` names."""
    if why := retention_refusal(opts):
        return why
    if opts.of_record and opts.scheme != V1:
        return f"--of-record names a recorded {V1} run to serve again under --scheme {V1}, not under --scheme {opts.scheme}"
    if not opts.commit:
        return None
    if opts.scheme == ROWS and not opts.gpu_tree:
        return f"--scheme {ROWS} (the default) is the windowed collector's, which hashes its rows on the GPU: drop --no-gpu-tree"
    if opts.scheme == ROWS and not opts.required_manifest:
        return (f"--scheme {ROWS} (the default) commits with the Commit of record's parts, which its bench row's manifest names: "
                "pass --required-manifest ROW_DIR/manifest.json (and the row's --hidden-so, --norm-tap-so), or --no-commit to serve "
                "without a Commit")
    if opts.scheme == ROWS and int(opts.max_num_batched_tokens) % PORT_IDS:
        return (f"--scheme {ROWS} starts each step's stream with its step-input port, the step's ids in whole "
                f"{step_rows.ROW_BYTES}-byte rows: pass --max-num-batched-tokens a multiple of {PORT_IDS}, not "
                f"{opts.max_num_batched_tokens}")
    if opts.scheme != V1:
        return None
    if not opts.of_record:
        return (f"--scheme {V1} commits SHA-256 (32-byte roots), which only a recorded run's readers take: serve under {ROWS}, the "
                f"default, or name the recorded {V1} run this one serves again with --of-record RUN_DIR (its serve.json)")
    try:
        rec = json.loads((Path(opts.of_record) / "serve.json").read_text())
        if not isinstance(rec, Mapping):
            raise ValueError("serve.json is not a JSON object")
    except (OSError, ValueError) as e:
        return f"--of-record {opts.of_record}: not a recorded serve-openai run ({type(e).__name__}: {e})"
    if (got := recorded_scheme(rec)) != V1:
        what = f"under {got}" if got else "nothing"
        return f"--of-record {opts.of_record} committed {what}, not under {V1}: serve under {ROWS} (drop --scheme {V1} and --of-record)"
    if (rec.get("model"), rec.get("revision")) != (opts.model, opts.revision):
        return (f"--of-record {opts.of_record} served {rec.get('model')}@{rec.get('revision')}, not {opts.model}@{opts.revision}: "
                f"a run under {V1} serves its recorded run's model again")
    return None


def committer_args(opts: Any) -> SimpleNamespace:
    """What `pipeline.commit.make_committer` and `pipeline.commit_attach.attach_of_record` read of a Commit's options, for
    `DECLARED_COMMITTER` under ``opts.scheme`` and the bench row's manifest ``opts.required_manifest``."""
    from verity_vllm.acquire.sources.taps import TapSources

    return SimpleNamespace(layout="chunk-leaf-v1", arena_mb=0, profile_hooks=False, pinned_populate_threads=None,
                           commit_scheme=opts.scheme, chunk=opts.chunk, hash_threads=opts.hash_threads,
                           required_manifest=opts.required_manifest, programs_root=programs_root_of(opts), program_dir=None,
                           hidden_so=opts.hidden_so, hidden_ns=opts.hidden_ns, tap_cap_mb=opts.tap_cap_mb, out=opts.out,
                           tap_sources=TapSources(norm_tap_so=opts.norm_tap_so), memory_map=None, moe_source="auto",
                           required_classes=None)


def programs_root_of(opts: Any) -> str | None:
    """The Build's row dir the manifest's partition and the attention record are read from: ``--programs-root``, else the
    manifest's own directory."""
    if opts.programs_root:
        return opts.programs_root
    return str(Path(opts.required_manifest).resolve().parent) if opts.required_manifest else None


def manifest_of_record(opts: Any) -> dict[str, Any]:
    """The bench row's required manifest, held to its partition of record (`commit.partition.require_partition`), as the Commit
    loads it."""
    from verity_vllm.commit.partition import require_partition

    return require_partition(json.loads(Path(opts.required_manifest).read_text()), programs_root_of(opts), None)


def parts_missing(com: Any, manifest: Mapping[str, Any], hidden_required: bool | None) -> list[str]:
    """The committer parts the Commit of record has for ``manifest`` that ``com`` lacks: the hidden tap when the manifest names
    the hidden class, each tap source `taps.attached` names, and the acquisition plan."""
    from verity_vllm.acquire.sources.taps import attached

    have = str(com.name).split("+")
    want = (["hidden_m1"] if hidden_required else []) + attached(manifest) + ["plan"]
    return [p for p in want if not any(h == p or (p == "hidden_m1" and h.startswith(p)) for h in have)]


def target_of(model: Any = None) -> dict[str, Any]:
    """What the compiler's φ reads of the serving device: its target profile (`TargetProfile.from_device`) with, given the served
    ``model``, the int8 scope its layers run read back from them (`kernel_readback.int8_exact_of`: ``linears`` under
    ``--precision verity_int8``, which predict's Build of an exact-int8 checkpoint requires; a mixed scope is refused), and the
    execution, bi-eager where vLLM runs batch-invariant (`vllm.envs.VLLM_BATCH_INVARIANT`)."""
    import vllm.envs as envs

    from verity_vllm.program.frontend.kernel_readback import int8_exact_of
    from verity_vllm.program.frontend.target_profile import TargetProfile

    target = TargetProfile.from_device()
    if model is not None:
        target = target.with_overrides(int8_exact=int8_exact_of(model))
    return {"target": target.to_json(), "execution": "bi-eager" if envs.VLLM_BATCH_INVARIANT else "eager"}


def seal_of(opts: Any) -> dict[str, Any]:
    """`seal_window`'s fixed arguments: the run's label, the tree's code identity and the engine's facts of record."""
    from verity_vllm.check.replay.driver import engine_facts_of_record
    from verity_vllm.pipeline import source_identity as SI

    return {"label": f"serve-openai:{opts.run_id}", "code": SI.code_identity(SI.TREE)["sha256"], "engine_of_record": engine_facts_of_record()}


def committer(opts: Any, llm: Any, label: str = "serve-openai") -> Any:
    """The server's committer over ``llm`` under ``opts.scheme`` (module docstring), its run ``label:<run id>``, refused unless the
    scheme may commit (`scheme_refusal`) and it retains on the host (the served record is read back from there)."""
    if why := scheme_refusal(opts):
        raise UsageError(why)
    from verity_vllm.commit.committer.native_host import NativeHostCommitter
    from verity_vllm.engine.handle import model_runner

    run_id = f"{label}:{opts.run_id}"
    eng = SimpleNamespace(llm=llm.vllm, runner=model_runner(llm.vllm), model=llm.model,
                          cp={"repo": llm.pin.model, "revision": llm.pin.revision})
    manifest, hidden_m1 = None, False
    spec = opts.retention.retain_exclude
    if opts.scheme == ROWS:
        from verity_vllm import settings
        from verity_vllm.commit.committer.host_retention import retain_exclude_families
        from verity_vllm.engine import vllm_adapter as va
        from verity_vllm.pipeline import commit as C, commit_attach as CA

        manifest, args = manifest_of_record(opts), committer_args(opts)
        if spec:
            settings.put("VERITY_RETAIN_EXCLUDE", spec)
        com = C.make_committer(DECLARED_COMMITTER, eng, args, run_id)
        if spec and set(getattr(com, "retain_exclude", ())) != retain_exclude_families(spec):
            raise RuntimeError(f"--retain-exclude {spec}: the committer {com.name} excludes {sorted(getattr(com, 'retain_exclude', ()))}")
    else:
        com = NativeHostCommitter(eng, run_id=run_id, arena_bytes=None, chunk=opts.chunk,
                                  hash_threads=opts.hash_threads, gpu_tree=opts.gpu_tree, fast_copy=opts.cudart_copy)
    if getattr(com, "commit_scheme", V1) != opts.scheme:
        raise RuntimeError(f"--scheme {opts.scheme}: the committer {com.name} commits {getattr(com, 'commit_scheme', V1)}")
    if com.retain != "host":
        raise UsageError(f"VERITY_RETAIN={com.retain}: the served record is read back from the values retained on the host")
    if manifest is not None:
        fa = va.observed_flash_attn_version(llm.model)
        classes, family = CA.classes_of_record(manifest, llm.model, fa)
        att = CA.attach_of_record(com, eng, args, manifest, Path(opts.out) / "acquisition", kind=DECLARED_COMMITTER,
                                 required_classes=classes, required_classes_cli=False, hidden_family=family, fa_version=fa)
        if missing := parts_missing(com, manifest, bool(att.hidden_required and family)):
            raise UsageError(f"--scheme rows-sha512: the committer {com.name} lacks {missing}, which the Commit of record attaches for "
                             f"{opts.required_manifest}; pass the bench row's taps (--hidden-so, --norm-tap-so) as its Commit does")
        com.required_manifest, com.required_manifest_path = manifest, str(Path(opts.required_manifest).resolve())
        hidden_m1 = bool(opts.hidden_so) and att.hidden_required is not False
    from verity_vllm.commit.served_record import SamplerInputsSource, StepInputPort

    tokens = int(opts.max_num_batched_tokens)
    try:
        com.attach_source(StepInputPort(com, tokens))
    except ValueError as e:
        raise UsageError(str(e)) from None
    if manifest is not None:
        single_ids(com, manifest)
    com.attach_source(SamplerInputsSource(com))
    com.arena_sizing_record = com.size_arena({"batch": opts.max_num_seqs, "max_prompt": tokens, "max_new": 1,
                                              "max_num_batched_tokens": tokens, "max_model_len": opts.max_model_len},
                                             hidden_m1=hidden_m1)
    if getattr(com, "windowed", False):
        com.window_ring = com.enable_windows(com.arena_sizing_record)     # windowed staging's ring replaces the per-step blocks
    if opts.retention.retain == "device":
        if not callable(getattr(com, "enable_device_pool", None)) or not getattr(com, "windowed", False):
            raise UsageError(f"--retain device: the committer {com.name} is not the windowed collector")
        com.device_pool_record = com.enable_device_pool(int(opts.retention.retain_device_gb * (1 << 30)))
    if opts.seal:
        com.register_weights()           # the weights tree every sealed window's bundle carries (hashed as VERITY_WEIGHTS_HASH says)
    from verity_vllm.pipeline.m1_capture import dump_cos_sin

    (private := Path(opts.out) / "private").mkdir(exist_ok=True)
    dump_cos_sin(llm, private)
    return com


def window_requests(rows: Mapping[int, Sequence[tuple]], name_of: Callable[[str], str | None],
                    ledger_of: Callable[[str], Any]) -> tuple[list[dict[str, Any]], dict[int, list[tuple[str, int, int]]]]:
    """A closed window's requests as a deferred Commit's context lists them (``request_id`` the body's ``user``, else the HTTP id;
    its index, prompt ids, ``max_tokens`` and first step in the window) and each step's request rows for the binding map
    (`binding.build_binding_map`'s ``request_rows``), from the rows the committer's `served_record.SamplerInputsSource` kept.
    ``name_of`` maps a runner id to the request's name (None: not a served request), ``ledger_of`` a name to its `Ledger`."""
    reqs: dict[str, dict[str, Any]] = {}
    request_rows: dict[int, list[tuple[str, int, int]]] = {}
    for step in sorted(rows):
        for rid, lo, hi, _start, _prefill in rows[step]:
            name = name_of(str(rid))
            if name is None:
                continue
            request_rows.setdefault(int(step), []).append((name, int(lo), int(hi)))
            if name not in reqs:
                led = ledger_of(name)
                params = (led.ingress.get("params") or {}) if led is not None else {}
                reqs[name] = {"request_id": name, "index": len(reqs), "arrive_step": int(step),
                              "prompt_token_ids": list(led.prompt_ids) if led is not None else [],
                              "max_tokens": params.get("max_tokens"), "http_id": led.http_id if led is not None else None,
                              "engine_id": led.engine_id if led is not None else None,
                              "ingress": dict(led.ingress) if led is not None else None}
    return list(reqs.values()), request_rows


def served_binder(manifest: Mapping[str, Any], steps: Sequence[int], sources: Sequence[Any] = ()) -> Any:
    """The `query.manifest.bind.Binder` of a served window's ``steps``: the identities one engine step of the bench row's
    Programs commits (its manifest's rows, weights aside, each (op_path, member, invocation) once, at its first row's place in
    the Program), at every step, with the committing sources' ports. A served step batches requests at any step of their own,
    and a module's output at a step is one identity whatever the requests in it; the binding map splits it between them
    (`binding.build_binding_map`'s request rows). With a `served_record.StepInputPort` among the sources, its tensors bind as
    the identities the Programs read them under (`port_identities`), ahead of any other tensor of the step."""
    from verity_vllm.commit.served_record import StepInputPort
    from verity_vllm.query.manifest.bind import source_ports

    return served_binder_of(manifest, steps, source_ports(sources), any(isinstance(s, StepInputPort) for s in sources))


def served_binder_of(manifest: Mapping[str, Any], steps: Sequence[int], ports: Mapping[str, Any], has_port: bool) -> Any:
    """`served_binder` given the sources' merged ports (`bind.source_ports`) and whether a step-input port is among them."""
    from verity_vllm.query.manifest.bind import Binder

    one: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in manifest.get("identities") or []:
        r = {**row, **(row.get("identity") or {})}
        if r.get("family") != "weights" and r.get("op_path") != "weights":
            one.setdefault((str(r["op_path"]), str(r.get("output_member", "out")), int(r.get("invocation", 0))), r)
    placed = [{**r, "identity": None, "request_id": "*", "step": None, "engine_step": int(s), "rank": 0} for s in steps for r in one.values()]
    binder = Binder.of_manifest({**manifest, "identities": placed, "requests_declared": []}, rank=0, ports=dict(ports))
    if has_port:
        binder.rebind.update(port_identities(manifest))
    return binder


def port_identities(manifest: Mapping[str, Any]) -> dict[str, tuple[str, str]]:
    """The identities the row's Programs read that the step-input port's tensors bind as, (op_path, member) by port name: its
    ids the manifest's one input-ids identity (``c00.input/ids`` on the compiled path, the embedding's ``<module>/in`` on the
    eager one), its positions ``c00.input/positions`` where the manifest has it. An eager row's Programs read no committed
    positions: RoPE's are static structure of the Program (`rules.profile.RotaryEmbeddingRule` refuses a computed one), compiled
    from the advice's rows, and the served-rows tie holds the port's positions to the window's served rows. A manifest naming
    two ids identities is refused."""
    from verity_vllm.query.manifest.format import SERVED_IDS, SERVED_POSITIONS

    fam: dict[tuple[str, str], str] = {}
    for row in manifest.get("identities") or []:
        r = {**row, **(row.get("identity") or {})}
        if r.get("family") != "weights" and r.get("op_path") != "weights":
            fam.setdefault((str(r["op_path"]), str(r.get("output_member", "out"))), str(r.get("family")))
    compiled_ids, compiled_pos = (tuple(n.rsplit("/", 1)) for n in (SERVED_IDS, SERVED_POSITIONS))
    ids = sorted({k for k, f in fam.items() if f == "input_token_ids"} | ({compiled_ids} & set(fam)))
    if len(ids) > 1:
        raise ValueError(f"the row's manifest names {len(ids)} input-ids identities, {ids}: the step-input port's ids can be one of them only")
    return ({PORT_NAMES[0]: ids[0]} if ids else {}) | ({PORT_NAMES[1]: compiled_pos} if compiled_pos in fam else {})


def single_ids(com: Any, manifest: Mapping[str, Any]) -> None:
    """The port's ids as the step's one copy: the module whose ``<module>/in`` they bind as (`port_identities`) is dropped from the
    acquisition plan's input sites, so its own input isn't committed beside them."""
    op, member = port_identities(manifest).get(PORT_NAMES[0], (None, None))
    if member == "in" and getattr(com, "plan_inputs", None) is not None:
        com.plan_inputs.discard(op)


def copies_beside_port(bmap: Mapping[str, Any], aliased: Mapping[str, tuple[str, str]]) -> list[str]:
    """The tensors the binding map binds, beside the port, to an identity the port's tensors are (`port_identities`): each a second
    copy of a step's ids or positions, which the row's Programs could read in place of the port's. As ``step <s>: <name>``."""
    port_as = {tuple(v): k for k, v in aliased.items()}
    out = set()
    for e in bmap.get("entries") or ():
        i = e.get("identity") or {}
        want = port_as.get((i.get("op_path"), i.get("output_member")))
        if want is not None and e.get("name") != want:
            out.add(f"step {i.get('step')}: {e.get('name')}")
    return sorted(out)


def positions_off(com: Any, request_rows: Mapping[int, Sequence[tuple]] | None = None) -> list[str]:
    """Each committed step whose port positions (its stream's [8T, 16T)) aren't the ones its rows ran: start..start+n-1 at each
    row's token rows [lo, hi), zero past the step's tokens. The rows are the committer's (`served_record.served_rows`), the ones
    the window's served rows register and the eager Programs' static RoPE positions are compiled from. Each step's first
    disagreement, by request (``request_rows``' name, else the runner's id)."""
    from verity_vllm.commit.served_record import served_rows

    return positions_off_of(port_positions(com), served_rows(com), request_rows)


def port_positions(com: Any) -> dict[int, list[int] | None]:
    """The port's committed positions at each committed step (`served_record.committed_ints`), None where it committed none."""
    from verity_vllm.commit.served_record import committed_ints

    return {step: committed_ints(com, step, PORT_NAMES[1]) for step in sorted(com._layouts)}


def positions_off_of(positions: Mapping[int, list[int] | None], rows: Mapping[int, Sequence[tuple]],
                     request_rows: Mapping[int, Sequence[tuple]] | None = None) -> list[str]:
    """`positions_off` over each step's committed port positions (`port_positions`) and the committer's rows."""
    out: list[str] = []
    for step in sorted(positions):
        got = positions[step]
        if got is None or not rows.get(step):
            out.append(f"step {step}: " + (f"no committed {PORT_NAMES[1]}" if got is None else "no rows to hold its positions to"))
            continue
        names = {int(lo): n for n, lo, _hi in (request_rows or {}).get(step, ())}
        want, of = [0] * len(got), [None] * len(got)
        for rid, lo, hi, start, _p in rows[step]:
            want[lo:hi], of[lo:hi] = range(start, start + hi - lo), [(names.get(lo, rid), lo, hi, start)] * (hi - lo)
        if (t := next((t for t, (g, w) in enumerate(zip(got, want)) if g != w), None)) is None:
            continue
        if of[t] is None:
            out.append(f"step {step}: token {t}, past the step's {max(hi for _r, _lo, hi, *_ in rows[step])} tokens, is position {got[t]}, not 0")
        else:
            name, lo, hi, start = of[t]
            out.append(f"step {step}: request {name}'s token row {t} (its rows [{lo}, {hi}) from position {start}) is at position "
                       f"{got[t]}, its row ran {want[t]}")
    return out


def seal_window(com: Any, rc: Any, d: Path, *, rec: Mapping[str, Any], reqs: list[dict[str, Any]],
                request_rows: dict[int, list[tuple[str, int, int]]], tokens: Mapping[str, list[int]], sampling: Mapping[str, Any],
                label: str, code: str | None = None, engine_of_record: Mapping[str, Any] | None = None, extras: Any = None) -> dict[str, Any]:
    """Seal one closed window into the row dir ``d`` as a declared Commit with ``--replay-deferred`` seals its run: the binding map
    (`binding.build_binding_map` over the window's layouts and step roots, its request rows by name), the store and weights tree
    (`replay_bundle.write_store`, with ``extras`` the registered tensors' bytes `extras_of` read, else read here), then
    `replay_bundle.seal_all` over the window's one run, its sampled replay and linkage deferred to the bundle; ``commit/runs.jsonl``
    and ``commit/verdict.json`` are the record it is opened against. Returns what the window's line in windows.jsonl records of it.
    The values the committer committed and didn't retain (``--retain-exclude``, `store_dump.not_retained`, which refuses any other
    gap) are named in the context, the bundle's manifest and that line as `replay_bundle.NOT_RETAINED`."""
    from verity_vllm.pipeline import replay_bundle as RB

    t0 = time.perf_counter()
    out = Path(d) / "commit"
    out.mkdir(parents=True, exist_ok=True)
    return seal_view(view_of(com), rc, d, store=lambda hashed: RB.write_store(out, 0, com, rc, digests=hashed, extras=extras), rec=rec,
                     reqs=reqs, request_rows=request_rows, tokens=tokens, sampling=sampling, label=label, code=code,
                     engine_of_record=engine_of_record, t0=t0)


@dataclass
class SealView:
    """What `seal_view` reads of a closed execution block's committer beside its store: its own state (`view_of`), or, in the
    seal's own process (`seal_process`), a copy of it that holds no value. ``positions`` gives the port's committed positions
    by step (`port_positions`) and ``unretained`` what the committer committed and doesn't hold (`store_dump.not_retained`),
    each read only where the seal reads it."""
    program_digest: bytes
    chunk: int
    layouts: Mapping[int, Any]
    name: str
    scheme: str
    retain: str | None
    manifest: Mapping[str, Any] | None
    manifest_path: str | None
    port: dict[str, Any] | None
    ports: Mapping[str, Any]
    has_port: bool
    rows: Mapping[int, Sequence[tuple]]
    positions: Callable[[], Mapping[int, list[int] | None]]
    unretained: Callable[[], dict[str, Any] | None]


def view_of(com: Any) -> SealView:
    """The committer ``com`` as `seal_view` reads it."""
    from verity_vllm.commit import store_dump as SD
    from verity_vllm.commit.served_record import StepInputPort, served_rows
    from verity_vllm.query.manifest.bind import source_ports

    manifest, sources = getattr(com, "required_manifest", None), getattr(com, "sources", None) or ()
    return SealView(program_digest=bytes(com.program_digest), chunk=com.chunk, layouts=com._layouts, name=com.name,
                    scheme=getattr(com, "commit_scheme", V1), retain=getattr(com, "retain", None), manifest=manifest,
                    manifest_path=getattr(com, "required_manifest_path", None), port=port_of(com),
                    ports=source_ports(sources) if manifest is not None else {},
                    has_port=any(isinstance(s, StepInputPort) for s in sources), rows=served_rows(com),
                    positions=functools.partial(port_positions, com), unretained=functools.partial(SD.not_retained, com))


def seal_view(v: SealView, rc: Any, d: Path, *, store: Callable[[dict[str, str]], Any], rec: Mapping[str, Any], reqs: list[dict[str, Any]],
              request_rows: dict[int, list[tuple[str, int, int]]], tokens: Mapping[str, list[int]], sampling: Mapping[str, Any],
              label: str, code: str | None = None, engine_of_record: Mapping[str, Any] | None = None,
              t0: float | None = None) -> dict[str, Any]:
    """`seal_window` over ``v``, its store written by ``store`` (given the dict that takes the sha256 of the files it hashed as it
    wrote them) where `seal_window` writes it; ``t0`` is when the seal began."""
    from verity_vllm.commit import binding as B
    from verity_vllm.pipeline import replay_bundle as RB

    t0 = time.perf_counter() if t0 is None else t0
    out = Path(d) / "commit"
    out.mkdir(parents=True, exist_ok=True)
    pd = v.program_digest.hex()
    manifest = v.manifest
    port = v.port
    with gc_paused():
        binder = served_binder_of(manifest, sorted(v.layouts), v.ports, v.has_port) if manifest is not None else None
        bmap = B.build_binding_map(v.layouts, rc.steps, rc.run_root, pd, request_rows=request_rows, chunk=v.chunk,
                                   weight_registration=rc.weight_registration, binder=binder, padded=PORT_NAMES if port is not None else ())
        aliased = port_identities(manifest) if binder is not None and port is not None else {}
        if twice := copies_beside_port(bmap, aliased):
            raise ValueError(f"the step-input port is the window's one copy of {sorted(aliased.values())}, and {len(twice)} other "
                             f"tensor(s) bind as them too ({twice[:3]}): the row's Programs would read one of two copies")
        if port is not None and (off := positions_off_of(v.positions(), v.rows, request_rows)):
            raise ValueError(f"the step-input port's positions aren't the ones the window's rows ran at {len(off)} step(s) ({off[:3]}): "
                             "the eager Programs' RoPE positions are those rows' start..start+n-1, so the window would commit positions "
                             "they don't read")
        with open(out / "binding_map_p0.json", "w") as fh:
            fh.writelines(JC.dict_pieces(bmap, "entries", separators=(",", ":"), default=str))
        drop(bmap["entries"])
    t_map = time.perf_counter()
    of_record = {} if manifest is None else {
        "required_manifest": {"path": v.manifest_path, "program_digest": manifest.get("program_digest"),
                              "digest": manifest.get("digest")},
        "op_path_aliases": dict(manifest.get("op_path_aliases") or {})}
    hashed: dict[str, str] = {}
    unretained = {RB.NOT_RETAINED: nr} if (nr := v.unretained()) else {}
    store(hashed)
    t_store = time.perf_counter()
    scheme = v.scheme
    sr, bl = RB.deferred_records(RB.path(out, 0))
    row = {"pair": 0, "arm": "instrumented", "label": label, "served_window": {"schema": SERVED_WINDOW, **{k: rec.get(k) for k in ("window", "run_id")}},
           "tokens_equal_ref": None,
           "commit": {"run_root": rc.run_root.hex(), "commit_scheme": scheme, "weight_registration": rc.weight_registration,
                      "steps": rec.get("steps"), "committer": v.name, "passed": rec.get("passed"),
                      "binding": {"map_digest": bmap["digest"], "map_entries_n": bmap["entries_n"], "program_digest": pd},
                      "commit_pass": None, "commit_fail_reason": None, "sampled_replay": sr, "boundary_linkage": bl}}
    layout = port_layout_of(v.layouts, v.chunk, port) if port is not None else None
    args = {k: None for k in RB.ARGS} | {"label": label, "retain": v.retain}
    ctx = {"pair": 0, "arm": "instrumented", "committer": v.name, "run_root": rc.run_root.hex(), "code": code, "args": args,
           "requests": reqs, "sampling": dict(sampling), "tokens": {k: list(t) for k, t in tokens.items()}, "stop_of_record": None,
           "ids_full": None, "retention": v.retain, "required_manifest_sha256": None, "declaration": None,
           "required_classes": [], "min_openings": 0, "vc_result": None, "engine_of_record": dict(engine_of_record or {}), "slim": None,
           "served_window": {"schema": SERVED_WINDOW, "window": rec.get("window"), "run_id": rec.get("run_id"), "commit_scheme": scheme,
                             "programs": "none sealed: the registered compiler's Programs for the window's advice",
                             "served_rows": rec.get("served_rows"), "step_input_port": port if port is None or binder is None else
                             {**port, "binds_as": {k: list(b) for k, b in aliased.items()}}, **of_record},
           RB.HASHED: hashed, **unretained}
    verdict: dict[str, Any] = {"schema": SERVED_WINDOW, "run_id": rec.get("run_id"), "window": rec.get("window"), "commit_pass": None,
                               "commit_scheme": scheme, "run_root": rc.run_root.hex()}
    tl = SimpleNamespace(span=lambda *_a, **_k: _Null())
    RB.seal_all(out, {0: ctx}, [row], verdict, [0], tl=tl)
    (out / "runs.jsonl").write_text(json.dumps(row, default=str) + "\n")
    (out / "verdict.json").write_text(json.dumps(verdict, indent=1, default=str) + "\n")
    b = (verdict.get("replay_deferred") or {}).get("bundles") or [{}]
    return {"row_dir": str(d), "bundle": b[0].get("path"), "manifest_sha256": b[0].get("manifest_sha256"), "bytes": b[0].get("bytes"),
            "commit_scheme": scheme, "binding_map_digest": bmap["digest"], "binding_map_entries": bmap["entries_n"],
            "requests": [q["request_id"] for q in reqs], "port_layout": layout, "seal_s": round(time.perf_counter() - t0, 3),
            "phases_s": {"binding_map": round(t_map - t0, 3), "store": round(t_store - t_map, 3),
                         "bundle": round(time.perf_counter() - t_store, 3)}, **unretained}


def port_of(com: Any) -> dict[str, Any] | None:
    """The committer's step-input port (`served_record.StepInputPort.describe`), None without one."""
    from verity_vllm.commit.served_record import StepInputPort

    src = next((s for s in getattr(com, "sources", None) or () if isinstance(s, StepInputPort)), None)
    return src.describe() if src is not None else None


def port_layout(com: Any, port: Mapping[str, Any]) -> dict[str, Any]:
    """Where each committed step's stream puts the port: the steps whose first two tensors are its ids and positions at
    byte offsets 0 and T·IB with T·IB and T·PB bytes, and the first that isn't (its first two tensors' names, offsets and sizes)."""
    return port_layout_of(com._layouts, com.chunk, port)


def port_layout_of(layouts: Mapping[int, Any], chunk: int, port: Mapping[str, Any]) -> dict[str, Any]:
    """`port_layout` over the committed steps' metas."""
    want = [(n, b[0], b[1] - b[0]) for n, b in zip(port["names"], (port["ids_bytes"], port["positions_bytes"]))]
    ok, bad = 0, None
    for step in sorted(layouts):
        head = []
        for m in list(layouts[step])[:2]:
            so = getattr(m, "stream_off", None)
            head.append((m.name, int(so) if so is not None and int(so) >= 0 else int(m.first_leaf) * int(chunk), int(m.nbytes)))
        if head == want:
            ok += 1
        elif bad is None:
            bad = {"step": int(step), "head": head}
    return {"steps": len(layouts), "at_head": ok, "first_other": bad}


@contextlib.contextmanager
def gc_paused() -> Iterator[None]:
    """No cyclic collection while the block runs: a served window's binding map is up to a few million young objects, and a
    collection that scans them stops every thread, the engine's too. Refcounting still frees what the block drops."""
    was = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was:
            gc.enable()


def drop(items: list[Any], chunk: int = 4096) -> None:
    """``items`` emptied ``chunk`` at a time: freeing a million map entries at once is one C call that holds the GIL."""
    while items:
        del items[-chunk:]


class _Null:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_a: object) -> None:
        return None


# ---------------------------------------------------------------------------------------------------------------- off the loop

#: what a committer's `reset` rebinds for its next window: the closed window's per-step state, as its seal reads it
WINDOW_STATE = ("_layouts", "_gpu_blocks", "_gpu_levels", "_streams", "_levels", "_padding", "_salts", "_done", "row_keys")
GB = 10 ** 9


def freeze(com: Any) -> Any:
    """The closed window's committer as its seal reads it while the committer runs the next window (`pipeline.commit`'s frozen
    view): a shallow copy, never installed, holding the window's own per-step state (`WINDOW_STATE`, each dict copied) and its
    sampler inputs' rows (`served_record.served_rows`), with the pinned blocks its values and levels sit in lent out of the
    committer's pool (`PinnedPool.lend`) until `thaw`. Taken after the window's finalize, before the next window's reset."""
    from verity_vllm.commit.served_record import SamplerInputsSource

    frozen = copy.copy(com)
    frozen.installed = True
    for k in WINDOW_STATE:
        if isinstance(getattr(com, k, None), dict):
            setattr(frozen, k, dict(getattr(com, k)))
    frozen.sources = [_own_rows(s) if isinstance(s, SamplerInputsSource) else s for s in getattr(com, "sources", None) or ()]
    for metas in frozen._layouts.values():
        iter(metas)                     # a plan's TensorMetas built once, here, rather than by the seal and the next window at once
    pool = getattr(com, "pool", None)
    frozen.lent = pool.lend() if hasattr(pool, "lend") else []
    return frozen


def _own_rows(src: Any) -> Any:
    c = copy.copy(src)
    c.rows = dict(src.rows)
    return c


def thaw(frozen: Any) -> None:
    """`freeze`'s lent blocks back to the committer's pool, once the seal has read them: free from the next window's reset. A
    detached window's view (`NativeHostCommitter.detach`) holds them from its window's end."""
    pool = getattr(frozen, "pool", None)
    if getattr(frozen, "lent", None) and hasattr(pool, "restore"):
        pool.restore(frozen.lent)
    frozen.lent = []


#: a warmed block holds any one of the windowed collector's takes (at most a 256 MiB window, `native_collect.WINDOW_MB`)
WARM_BLOCK = 1 << 30


def warm_pool(com: Any, gb: float, clock: Callable[[], float] = time.perf_counter) -> dict[str, Any] | None:
    """The committer's pinned pool grown by ``gb`` GB before serving, in `WARM_BLOCK` blocks faulted in together
    (`PinnedPool.reserve`; free from the first window's reset), so the windows awaiting their seal and the open window find
    blocks ready: grown on the engine loop instead, a burst's windows stall it 0.1-0.2 s a GiB on n1. None for no growth."""
    pool, n = getattr(com, "pool", None), -(-int(gb * GB) // WARM_BLOCK) if gb > 0 else 0
    if pool is None or not n:
        return None
    t0 = clock()
    made = pool.reserve([WARM_BLOCK] * n)
    return {"gb": round(made / GB, 1), "blocks": n, "seconds": round(clock() - t0, 2)}


def extras_of(com: Any, rc: Any) -> Any:
    """The registered tensors' bytes a window's bundle carries (`replay_bundle.weights_extra`), read from the device; None without
    a weights tree (`write_store` then refuses the window by name)."""
    from verity_vllm.pipeline import replay_bundle as RB

    return RB.weights_extra(com, rc) if getattr(com, "weights_tree", None) else None


def committed_bytes(com: Any) -> int:
    """The bytes the open window's absorbed steps committed so far (its `StepCommitment`s), read while the committer's worker adds
    steps: about what its seal writes (the retained values and their tree levels)."""
    return sum(int(s.n_bytes) for s in list((getattr(com, "_done", None) or {}).values()))


class Sealer:
    """Seals closed windows off the engine loop in two stages, each a thread of its own taking windows in order: the first runs
    each window's job (its finalize and records when the close detached it, then its release), and the second what the job
    returns when that is callable (its seal and its line in seals.jsonl), so that one window's seal runs beside the next one's
    finalize. `submit` returns at once while the windows queued or sealing hold under ``hold_bytes`` of committed values (the
    pinned host memory the double-buffered windows take), else it waits for room and says how long; `drain` waits for every job
    and its seal. A job's own failure, or its seal's, is its window's: the next one still runs."""
    process: Any = None                                  # the `seal_process.SealProcess` its seals run in (`seal_process.ProcessSealer`)

    def __init__(self, hold_bytes: int, clock: Callable[[], float] = time.perf_counter) -> None:
        self.hold_bytes, self.clock = int(hold_bytes), clock
        self.held = 0                                    # committed bytes of the windows queued or sealing
        self.errors: list[str] = []
        self._cv = threading.Condition()
        self._jobs: deque = deque()
        self._seals: deque = deque()                     # (seal or None, the bytes its window holds, whether it is the last)
        self.gate = threading.Event()                    # set: the engine loop needs no GIL now (its step's forward is launched, or it waits)
        self.gate.set()
        from verity_vllm import settings
        self.gate_mode = settings.VERITY_SEAL_GATE
        self.pace_s = float(settings.VERITY_SEAL_PACE_S)
        self.paced = [0, 0.0, 0]                         # paces that waited, their seconds, paces that ran out
        self.thread = threading.Thread(target=self._run, name="serve-seal", daemon=True)
        self.seal_thread = threading.Thread(target=self._run_seals, name="serve-seal-2", daemon=True)
        self.thread.start()
        self.seal_thread.start()

    def submit(self, job: Callable[[float], Any], nbytes: int = 0) -> float:
        """Queue ``job``, which holds ``nbytes`` of a window's values until it and the seal it returns have run, and is called with the
        seconds the caller waited for room; returns them too."""
        t0 = self.clock()
        with self._cv:
            while nbytes and self.held and self.held + nbytes > self.hold_bytes:
                self.gate.set()
                self._cv.wait()
            waited = self.clock() - t0
            self.held += int(nbytes)
            self._jobs.append((functools.partial(job, waited), int(nbytes)))
            self._cv.notify_all()
        return waited

    def pace(self) -> None:
        if self.gate.is_set():
            return
        t0 = time.perf_counter()
        ok = self.gate.wait(self.pace_s)
        self.paced[0] += 1
        self.paced[1] += time.perf_counter() - t0
        self.paced[2] += not ok

    def _call(self, f: Callable[[], Any]) -> Any:
        try:
            return f()
        except Exception as e:  # noqa: BLE001
            self.errors.append(f"{type(e).__name__}: {e}"[:300])
            return None

    def _yield(self, frame: Any, event: str, arg: Any) -> None:
        if not self.gate._flag:
            self.pace()

    def _run(self) -> None:
        if self.gate_mode == "mid":
            sys.setprofile(self._yield)
        while True:
            with self._cv:
                while not self._jobs:
                    self._cv.wait()
                job, nbytes = self._jobs[0]
            if job is not None:
                self.pace()
            seal = self._call(job) if job is not None else None
            with self._cv:
                self._jobs.popleft()
                self._seals.append((seal if callable(seal) else None, nbytes, job is None))
                self._cv.notify_all()
            if job is None:
                return

    def _run_seals(self) -> None:
        if self.gate_mode == "mid":
            sys.setprofile(self._yield)
        while True:
            with self._cv:
                while not self._seals:
                    self._cv.wait()
                seal, nbytes, last = self._seals[0]
            if seal is not None:
                self.pace()
                self._call(seal)
            with self._cv:
                self._seals.popleft()
                self.held -= nbytes
                self._cv.notify_all()
            if last:
                return

    def drain(self) -> None:
        self.gate.set()
        with self._cv:
            while self._jobs or self._seals:
                self._cv.wait()

    def close(self) -> None:
        """Every job and seal run, then both threads ended."""
        self.gate.set()
        with self._cv:
            self._jobs.append((None, 0))
            self._cv.notify_all()
        self.thread.join()
        self.seal_thread.join()

    def summary(self) -> dict[str, Any]:
        return {"hold_bytes": self.hold_bytes, "held": self.held, "errors": self.errors[:10], "errors_n": len(self.errors),
                "gate": {"mode": self.gate_mode, "pace_s": self.pace_s, "paced": self.paced[0], "paced_s": round(self.paced[1], 4),
                         "ran_out": self.paced[2], "mid_layer": getattr(self, "mid_layer", None)}}


class DiskFloor:
    """The serve dir's filesystem kept above ``floor`` bytes free: a closed window is sealed only when its committed bytes (about
    its bundle's size) and the bytes the seals still pending will write leave the floor free (`short`)."""

    def __init__(self, path: str | Path, floor: int, usage: Callable[[Any], Any] = shutil.disk_usage) -> None:
        self.path, self.floor, self.usage = Path(path), int(floor), usage

    def free(self) -> int:
        return int(self.usage(self.path).free)

    def short(self, nbytes: int, pending: int = 0) -> dict[str, Any] | None:
        """None when a window of ``nbytes`` fits above the floor; else what was seen, for the window's record."""
        free = self.free()
        if free - int(pending) - int(nbytes) >= self.floor:
            return None
        return {"free_gb": round(free / GB, 1), "pending_gb": round(pending / GB, 1), "window_gb": round(nbytes / GB, 1),
                "floor_gb": round(self.floor / GB, 1), "path": str(self.path)}

    def refusal(self, window: int) -> str | None:
        """Before serving: why the filesystem can't take one window of ``window`` bytes above the floor, else None."""
        free = self.free()
        if free >= self.floor + int(window):
            return None
        return (f"{self.path} has {free / GB:.0f} GB free, under the disk floor ({self.floor / GB:.0f} GB, --disk-floor-gb) plus one "
                f"sealed window ({int(window) / GB:.0f} GB, --window-gb): free space there or lower them")


class Watchdog:
    """A thread that sleeps ``period_s`` at a time beside the engine: each wake ``min_s`` or more late is a pause, a line of
    ``path`` (pauses.jsonl: when it ended, unix seconds as the step trace's ``t_end``, and how late). The engine's thread waits
    as long for the GIL (or a core), so a pause is a stall of the step then running; the window's ``gc`` and the seals'
    ``phases_s`` say what ran."""

    def __init__(self, path: str | Path, *, period_s: float = 0.05, min_s: float = 0.2, clock: Callable[[], float] = time.perf_counter,
                 sleep: Callable[[float], Any] = time.sleep, wall: Callable[[], float] = time.time) -> None:
        self.path, self.period_s, self.min_s = Path(path), float(period_s), float(min_s)
        self.clock, self.sleep, self.wall = clock, sleep, wall
        self.late: list[float] = []
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.run, name="serve-watchdog", daemon=True)

    def run(self) -> None:
        while not self.stopped.is_set():
            t0 = self.clock()
            self.sleep(self.period_s)
            late = self.clock() - t0 - self.period_s
            if late >= self.min_s:
                self.late.append(late)
                with open(self.path, "a") as fh:
                    fh.write(json.dumps({"t_end": round(self.wall(), 3), "late_s": round(late, 3)}) + "\n")

    def stop(self) -> None:
        self.stopped.set()
        if self.thread.is_alive():
            self.thread.join()

    def summary(self) -> dict[str, Any]:
        return {"period_s": self.period_s, "min_s": self.min_s, "pauses": len(self.late),
                "late_s": {"max": round(max(self.late, default=0.0), 3), "total": round(sum(self.late), 3)}}
