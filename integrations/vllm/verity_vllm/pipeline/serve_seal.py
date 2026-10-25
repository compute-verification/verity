"""The Commit scheme and the sealing of `verity_vllm.pipeline.serve_openai`'s windows.

`committer` builds the server's committer: ``--scheme vllm-v1`` `NativeHostCommitter` (32-byte roots), ``--scheme rows-sha512``
the declared Commit of record's (`pipeline.commit.make_committer`'s ``native_collect_v2b``: hm96-sha512 rows in frame-v3-sha512
trees, 64-byte roots) with its acquisition built as the Commit builds it (`pipeline.commit_attach.attach_of_record`) over the bench
row's required manifest (``--required-manifest``): the attention hidden tap (``--hidden-so``), the norm-scale tap
(``--norm-tap-so``) and the hook sets of the acquisition plan, so its name is the Commit's (``...+hidden_m1+norm_scale+plan``)
and a part the manifest names that did not attach is refused (`parts_missing`). Then `served_record.SamplerInputsSource` is
attached and the arena sized; under ``--seal`` it registers the weights.

With ``seal`` (`seal_window`, which `serve_commit.CommitWindows` calls at each close), each closed window is sealed as a
deferred Commit's row dir, ``windows/w<k>/``, the way a declared Commit with ``--replay-deferred`` seals its run
(`pipeline.replay_bundle`): ``commit/replay_bundle_p0/`` (the committed store, the weights tree, the binding map, the context,
the manifest), and beside it the record that bundle is opened against (`deferred_rows.of_record`): ``commit/verdict.json``
naming the bundle, ``commit/runs.jsonl`` with the window's run, and ``commit/binding_map_p0.json``. The binding map names
each committed value as the row's Programs read it (`served_binder`: a Linear's ``out``, RoPE's ``query`` / ``key``, a norm's
``norm_scale``, ``fa2.m1/*``), and the context carries the manifest's ``op_path_aliases``, which the opened store resolves a
shared module's layer-N path through. The window's requests are named by the body's ``user`` (the id the advice names them
by), and the bundle names no Programs: a served window's Programs are the registered compiler's for its advice, made after
the window closed.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from verity_vllm.config import UsageError

SCHEMES = ("vllm-v1", "rows-sha512")
#: what a sealed served window's context says it is (`deferred_rows.of_record` reads no Programs of record for it)
SERVED_WINDOW = "verity-vllm/served-window/v1"


#: the committer a declared Commit commits rows-sha512 with (`pipeline.commit`'s default ``--committer``)
DECLARED_COMMITTER = "native_collect_v2b"


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


def target_of() -> dict[str, Any]:
    """What the compiler's φ reads of the serving device: its target profile (`TargetProfile.from_device`) and the execution,
    bi-eager where vLLM runs batch-invariant (`vllm.envs.VLLM_BATCH_INVARIANT`)."""
    import vllm.envs as envs

    from verity_vllm.program.frontend.target_profile import TargetProfile

    return {"target": TargetProfile.from_device().to_json(), "execution": "bi-eager" if envs.VLLM_BATCH_INVARIANT else "eager"}


def seal_of(opts: Any) -> dict[str, Any]:
    """`seal_window`'s fixed arguments: the run's label, the tree's code identity and the engine's facts of record."""
    from verity_vllm.check.replay.driver import engine_facts_of_record
    from verity_vllm.pipeline import source_identity as SI

    return {"label": f"serve-openai:{opts.run_id}", "code": SI.code_identity(SI.TREE)["sha256"], "engine_of_record": engine_facts_of_record()}


def committer(opts: Any, llm: Any) -> Any:
    """The server's committer over ``llm`` under ``opts.scheme`` (module docstring), refused unless it retains on the host (the served
    record is read back from there)."""
    from verity_vllm.commit.committer.native_host import NativeHostCommitter
    from verity_vllm.engine.engine_profile import get_model_runner

    eng = SimpleNamespace(llm=llm.vllm, runner=get_model_runner(llm.vllm), model=llm.model,
                          cp={"repo": llm.pin.model, "revision": llm.pin.revision})
    manifest, hidden_m1 = None, False
    if opts.scheme == "rows-sha512":
        from verity_vllm.engine import vllm_adapter as va
        from verity_vllm.pipeline import commit as C, commit_attach as CA

        manifest, args = manifest_of_record(opts), committer_args(opts)
        com = C.make_committer(DECLARED_COMMITTER, eng, args, f"serve-openai:{opts.run_id}")
    else:
        com = NativeHostCommitter(eng, run_id=f"serve-openai:{opts.run_id}", arena_bytes=None, chunk=opts.chunk,
                                  hash_threads=opts.hash_threads, gpu_tree=opts.gpu_tree, fast_copy=opts.cudart_copy)
    if getattr(com, "commit_scheme", "vllm-v1") != opts.scheme:
        raise RuntimeError(f"--scheme {opts.scheme}: the committer {com.name} commits {getattr(com, 'commit_scheme', 'vllm-v1')}")
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
    from verity_vllm.commit.served_record import SamplerInputsSource

    com.attach_source(SamplerInputsSource(com))
    tokens = int(opts.max_num_batched_tokens)
    com.arena_sizing_record = com.size_arena({"batch": opts.max_num_seqs, "max_prompt": tokens, "max_new": 1,
                                              "max_num_batched_tokens": tokens, "max_model_len": opts.max_model_len},
                                             hidden_m1=hidden_m1)
    if getattr(com, "windowed", False):
        com.window_ring = com.enable_windows(com.arena_sizing_record)     # windowed staging's ring replaces the per-step blocks
    if opts.seal:
        com.register_weights()           # the weights tree every sealed window's bundle carries (hashed as VERITY_WEIGHTS_HASH says)
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
    (`binding.build_binding_map`'s request rows)."""
    from verity_vllm.query.manifest.bind import Binder, source_ports

    one: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in manifest.get("identities") or []:
        r = {**row, **(row.get("identity") or {})}
        if r.get("family") != "weights" and r.get("op_path") != "weights":
            one.setdefault((str(r["op_path"]), str(r.get("output_member", "out")), int(r.get("invocation", 0))), r)
    placed = [{**r, "identity": None, "request_id": "*", "step": None, "engine_step": int(s), "rank": 0} for s in steps for r in one.values()]
    return Binder.of_manifest({**manifest, "identities": placed, "requests_declared": []}, rank=0, ports=source_ports(sources))


def seal_window(com: Any, rc: Any, d: Path, *, rec: Mapping[str, Any], reqs: list[dict[str, Any]],
                request_rows: dict[int, list[tuple[str, int, int]]], tokens: Mapping[str, list[int]], sampling: Mapping[str, Any],
                label: str, code: str | None = None, engine_of_record: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Seal one closed window into the row dir ``d`` as a declared Commit with ``--replay-deferred`` seals its run: the binding map
    (`binding.build_binding_map` over the window's layouts and step roots, its request rows by name), the store and weights tree
    (`replay_bundle.write_store`), then `replay_bundle.seal_all` over the window's one run, its sampled replay and linkage deferred
    to the bundle; ``commit/runs.jsonl`` and ``commit/verdict.json`` are the record it is opened against. Returns what the
    window's line in windows.jsonl records of it."""
    from verity_vllm.commit import binding as B
    from verity_vllm.pipeline import replay_bundle as RB

    t0 = time.perf_counter()
    out = Path(d) / "commit"
    out.mkdir(parents=True, exist_ok=True)
    pd = bytes(com.program_digest).hex()
    manifest = getattr(com, "required_manifest", None)
    binder = served_binder(manifest, sorted(com._layouts), getattr(com, "sources", None) or ()) if manifest is not None else None
    bmap = B.build_binding_map(com._layouts, rc.steps, rc.run_root, pd, request_rows=request_rows, chunk=com.chunk,
                               weight_registration=rc.weight_registration, binder=binder)
    of_record = {} if manifest is None else {
        "required_manifest": {"path": getattr(com, "required_manifest_path", None), "program_digest": manifest.get("program_digest"),
                              "digest": manifest.get("digest")},
        "op_path_aliases": dict(manifest.get("op_path_aliases") or {})}
    (out / "binding_map_p0.json").write_text(json.dumps({k: v for k, v in bmap.items() if k != "entries"} | {"entries": bmap["entries"]},
                                                        separators=(",", ":"), default=str))
    hashed: dict[str, str] = {}
    RB.write_store(out, 0, com, rc, digests=hashed)
    scheme = getattr(com, "commit_scheme", "vllm-v1")
    sr, bl = RB.deferred_records(RB.path(out, 0))
    row = {"pair": 0, "arm": "instrumented", "label": label, "served_window": {"schema": SERVED_WINDOW, **{k: rec.get(k) for k in ("window", "run_id")}},
           "tokens_equal_ref": None,
           "commit": {"run_root": rc.run_root.hex(), "commit_scheme": scheme, "weight_registration": rc.weight_registration,
                      "steps": rec.get("steps"), "committer": com.name, "passed": rec.get("passed"),
                      "binding": {"map_digest": bmap["digest"], "map_entries_n": bmap["entries_n"], "program_digest": pd},
                      "commit_pass": None, "commit_fail_reason": None, "sampled_replay": sr, "boundary_linkage": bl}}
    args = {k: None for k in RB.ARGS} | {"label": label, "retain": getattr(com, "retain", None)}
    ctx = {"pair": 0, "arm": "instrumented", "committer": com.name, "run_root": rc.run_root.hex(), "code": code, "args": args,
           "requests": reqs, "sampling": dict(sampling), "tokens": {k: list(v) for k, v in tokens.items()}, "stop_of_record": None,
           "ids_full": None, "retention": getattr(com, "retain", None), "required_manifest_sha256": None, "declaration": None,
           "required_classes": [], "min_openings": 0, "vc_result": None, "engine_of_record": dict(engine_of_record or {}), "slim": None,
           "served_window": {"schema": SERVED_WINDOW, "window": rec.get("window"), "run_id": rec.get("run_id"), "commit_scheme": scheme,
                             "programs": "none sealed: the registered compiler's Programs for the window's advice", **of_record},
           RB.HASHED: hashed}
    verdict: dict[str, Any] = {"schema": SERVED_WINDOW, "run_id": rec.get("run_id"), "window": rec.get("window"), "commit_pass": None,
                               "commit_scheme": scheme, "run_root": rc.run_root.hex()}
    tl = SimpleNamespace(span=lambda *_a, **_k: _Null())
    RB.seal_all(out, {0: ctx}, [row], verdict, [0], tl=tl)
    (out / "runs.jsonl").write_text(json.dumps(row, default=str) + "\n")
    (out / "verdict.json").write_text(json.dumps(verdict, indent=1, default=str) + "\n")
    b = (verdict.get("replay_deferred") or {}).get("bundles") or [{}]
    return {"row_dir": str(d), "bundle": b[0].get("path"), "manifest_sha256": b[0].get("manifest_sha256"), "bytes": b[0].get("bytes"),
            "commit_scheme": scheme, "binding_map_digest": bmap["digest"], "requests": [q["request_id"] for q in reqs],
            "seal_s": round(time.perf_counter() - t0, 3)}


class _Null:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_a: object) -> None:
        return None
