"""Prove path: a committed run and its advice through the proof service, for network accounting under arbitrary vLLM usage.

Each window's step units are Programs a registered compiler made from the advice (`pipeline.advice_windows`), not a declared
workload's. The advice is private: the prover commits I1's canonical encoding under hiding rows (`advice_windows.commit_advice`),
and each window's registration binds the compiler's digest and that commitment's root, before any coin. The prover commits each
window over the compiler's Programs from its sealed bundle. The verifier, handed the commitment's opening (the named gap: the
advice is opened to the verifier until the compile unit exists, README.md), compiles the advice with its own copy of the
compiler, derives its own window and audits it through the one-stage audit's `service.Stream` (`benchmarks/one_stage/a5.py`'s
`audit`): register, select on the verifier's own live coins (read only after the receipt), C-Flock `--zk` behind the Lean
firewall on the drawn units, Lean's verdicts, and the outcome, which the Lean verifier of record's `Run.run` must agree with. It
then checks that the advice was fixed before the draw (`advice_before_draw`).

Two kinds of committed run:

* **A TP Commit** (``--tp-commit DIR --build-root DIR``): one window per rank, each from the rank's sealed bundle, under the TP
  run root. The compiler is `Capped` over `advice_windows.Traced` (the Build's Programs of record, refused unless of record).
* **A single-GPU Commit** (``--bundle DIR --row-dir DIR --workload F``, TP1): one window, ``rank0/``, from the Commit's sealed
  bundle (``<row dir>/commit/replay_bundle_p0``) under its run root. The compiler is `Capped` over `I2Compiler`: I2's
  `predict.steps.compile_steps` on the advice and the requests' bodies, each Program the one I2 compiles, its body read from
  the Build's files of that digest (`I2Compiler.load`). The window is a stand-in of the record's SHA-512 rows root (`stand_in1`).

    prove_path.py advice --workload F --out FILE
    prove_path.py commit (--tp-commit DIR --build-root DIR | --bundle DIR --row-dir DIR --workload F [--hf-config F]) \\
        --advice FILE --opening FILE --partition F --out WINDOWS [--cap-bits L] [--ranks 0,1] [--jobs N] [--forge MEMBER:UNIT] \\
        [--transfer-gate T ...]
    prove_path.py audit (--tp-commit DIR --build-root DIR | --bundle DIR --row-dir DIR --workload F [--hf-config F]) \\
        --opening FILE --windows WINDOWS --partition F --template-sets DIR --m0-pythonpath PATH --bin flock-circuit \\
        --lean flock-verify --firewall flock-firewall --out DIR [--cap-bits L] [--ranks 0,1] [--jobs N] [--forged] \\
        [--transfer-gate T ...] [--challenger local --main FILE]

* **advice**: a declared run's I1 advice from its workload file (`advice_windows.advice_of_workload`), its canonical encoding.
  A served run's comes from its step trace instead (ADVICE-SRC); `compile_advice` reads either.
* **commit** (the prover): the advice compiled (`compile_advice`) and committed, its opening (the advice and its salts) written to
  ``--opening``, the prover's private file, never under the run's out/; then each window from its bundle over the compiler's
  Programs for it, ``WINDOWS/rank<r>/``. ``--forge MEMBER:UNIT`` commits that unit with bit 0 of its first output word flipped
  (`forging`): its rows, commit strings and roots all agree with the flipped word, so only the proof can refuse it.
* **audit** (the verifier): each window from the record, the opened advice and its own compiler (`open_advice`, `window_of`),
  audited by a5 on the verifier's coins with the advice negatives beside it (`more_negatives`), then `advice_before_draw` on the
  draw's receipt and the advice-root-after-the-draw negative (`advice_after_draw`); ``OUT/summary.json`` over the windows.
  ``--forged``: the prover runs with its self-check skipped (``FC_ZK_SELF_CHECK=skip``) so it sends proofs of the false
  witness, and a window passes when its registration passed every check before the draw and Lean refused the outcome
  (`forged_row`).

`Capped` adds the public cap ℓ on the advice's length to the compiler's registered digest, and `compile_advice` is total on
every byte string: the requests, or a `ValueError` naming why, the same one on every run.

**Transfer gates** (``--transfer-gate T``, repeatable, the same on both sides): the templates the registered compiler lists as
holding transfer gates (`advice_windows`: the I/O shell's units, proved with selection All, never sampled), which its digest
covers (`I2Compiler`, `advice_windows.Traced`). The window's ``work:K`` law then draws every unit of theirs beside K of the rest
(`serving_rows.work_table_of`), and the stream registers with that work table. Until COMPILER lists the I/O shell's units, the
run names a stand-in template, at TP1 ``RoPEHead_v1{D=64}`` (``llama32_1b_tg_*.json``). Their negatives are a5's
`transfer_gate_negatives` and the registration under the compiler listing none (`more_negatives`).

**Coins**: each window's `service.Stream` draws on the verifier's own coins, or with ``--challenger local --main FILE`` on the
auditor's challenger's (`a5_coins`: the run root's one session, committed to each registration before any coin is read;
``local`` is a development session on this machine, pinned on ``--main``, the verifier's copy of main's first-parent history).
`advice_before_draw` holds the advice to the receipt of the stream that drew.
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import json
import multiprocessing
import os
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "one_stage"))

from verity.core.protocols.sampled_proofs.one_stage import registration as R  # noqa: E402
from verity.ml.circuits.codec import canonical_json  # noqa: E402
from verity_vllm.pipeline import advice_windows as AW  # noqa: E402
from verity_vllm.pipeline import deferred_rows as DRW  # noqa: E402
from verity_vllm.pipeline.tp import serving_rows as SRT  # noqa: E402

CAPPED = "verity/prove-path/capped-compiler/v0"
I2 = "verity/prove-path/i2-compile-steps/v0"
#: ℓ, the public cap on the advice's encoding in bits: 1 MiB, far above a TP Commit's few hundred requests (I1 spends tens of
#: bits a row)
CAP_BITS = 1 << 23
#: the checkpoint config I2's φ reads when none is given: unsloth/Llama-3.2-1B's (I2's test fixture)
LLAMA32_1B_CONFIG = HERE.parents[1] / "integrations/vllm/tests/predict/fixtures/llama32-1b.config.json"
ROWS1 = "prove_path.py commit: a single-GPU Commit's sealed replay bundle -> rows over the registered compiler's Programs"
#: the sampling parameters a request's body carries to I2 (vLLM's `SamplingParams` names the record holds)
BODY_SAMPLING = ("temperature", "top_p", "top_k", "seed", "n")
#: where an audit's ports start when ``--port`` isn't given (`free_ports`): below Linux's ephemeral range (32768-60999), from
#: which outbound connections take theirs
PORTS = range(10000, 32000)


@dataclass(frozen=True)
class Capped:
    """A registered compiler with ℓ, the public cap on the advice's length in bits. Its digest names the inner compiler's and ℓ,
    so the registration's ``compiler`` fixes ℓ before the draw as it fixes the code; `load` and ``transfer_gates`` are the inner
    compiler's."""

    inner: Any
    cap_bits: int = CAP_BITS
    name: str = CAPPED

    def __post_init__(self) -> None:
        if isinstance(self.cap_bits, bool) or not isinstance(self.cap_bits, int) or self.cap_bits < 8:
            raise ValueError(f"the cap ℓ is a whole number of bits, at least 8, not {self.cap_bits!r}")

    @property
    def digest(self) -> str:
        return hashlib.sha512(canonical_json({"name": self.name, "cap_bits": self.cap_bits,
                                              "compiler": {"name": str(self.inner.name), "digest": self.inner.digest}})).hexdigest()

    @property
    def transfer_gates(self) -> tuple[str, ...]:
        return tuple(getattr(self.inner, "transfer_gates", ()))

    def load(self, rank: int, world: int, requests: Any) -> Any:
        return self.inner.load(rank, world, requests)


@dataclass(frozen=True)
class I2Compiler:
    """I2 (`predict.steps.compile_steps`) as a single-GPU window's compiler: the requests and their Programs are the ones I2
    compiles from the advice and each request's body by its I1 id (``ingress``; I2's prompts and sampling are values the request
    brings, never advice), under the registered φ (``registered``: the model, target, engine and ℓ, no workload). `load` reads
    each Program's body from the Build's files of the digest I2 compiled (a ``build_request*/`` under ``programs_root`` whose
    ``result.json`` names it), refused where no Build dir holds it: the Build's files are the bodies' cache, I2's digests are the
    Programs. ``transfer_gates`` are the templates it lists as holding transfer gates (module docstring), the run's stand-in until
    I2 lists the I/O shell's own. Its digest names I2's code (`score.code_version`), φ, the default seed, ℓ and that list."""

    registered: Any
    ingress: Mapping[str, Mapping[str, Any]]
    programs_root: str
    name: str = I2
    transfer_gates: tuple[str, ...] = ()
    compiled: dict = field(default_factory=dict, compare=False, repr=False)

    @functools.cached_property
    def digest(self) -> str:
        from verity_vllm.predict.score import code_version

        r = self.registered
        return hashlib.sha512(canonical_json({"name": self.name, "code": code_version(), "phi": r.phi.to_json(),
                                              "default_seed": r.default_seed, "cap_bits": r.cap_bits,
                                              "transfer_gates": sorted(self.transfer_gates)})).hexdigest()

    def requests(self, data: bytes) -> list[dict]:
        """I2's requests for the advice ``data``: each request's id, first step, prompt length, ``max_tokens`` (its kept outputs)
        and the digest of the Program I2 compiled for it; I2's refusal (`Unexpressible`) is a ValueError naming it."""
        from verity_vllm.predict.build import Unexpressible
        from verity_vllm.predict.steps import compile_steps

        if data not in self.compiled:
            try:
                c = compile_steps(self.registered, data, self.ingress)
            except Unexpressible as e:
                raise ValueError(f"I2 refuses the advice: {e}") from None
            self.compiled[data] = [{"request_id": u.id, "arrive_step": u.first_step, "prompt_len": u.LP, "max_tokens": u.T + 1,
                                    "digest": u.digest} for u in c.requests]
        return [dict(q) for q in self.compiled[data]]

    def load(self, rank: int, world: int, requests: Any) -> SimpleNamespace:
        """The window's Programs for I2's ``requests`` (`requests`), as `deferred_rows.of_record` holds the Build's: by request id
        its instances, its dir and its digest, and each request's arrival, its first step in the advice."""
        from verity_vllm.query.manifest import format as MF

        if (int(rank), int(world)) != (0, 1):
            raise ValueError(f"I2 compiles a single-GPU session: rank {rank} of a world of {world}")
        by = bodies_by_digest(self.programs_root)
        loaded, digs = {}, {}
        for q in requests:
            d = by.get(str(q.get("digest")))
            if d is None:
                raise ValueError(f"request {q['request_id']}: no Build dir under {self.programs_root} holds I2's Program "
                                 f"{str(q.get('digest'))[:16]}")
            v = MF.load_program(d)
            got = str((v.get("result") or {}).get("program_digest") or (v.get("instances") or {}).get("digest") or "")
            if got != q["digest"]:
                raise ValueError(f"request {q['request_id']}: {d}'s instances are Program {got[:16]}, not I2's {q['digest'][:16]}")
            loaded[q["request_id"]], digs[q["request_id"]] = v, got
        return SimpleNamespace(progs={k: v["instances"] for k, v in loaded.items()}, dirs={k: v["dir"] for k, v in loaded.items()},
                               digs=digs, arrivals={q["request_id"]: int(q["arrive_step"]) for q in requests})


def bodies_by_digest(programs_root: str) -> dict[str, str]:
    """Each Build dir under ``programs_root`` (``build_request*/``) by the Program digest its ``result.json`` names."""
    out: dict[str, str] = {}
    for d in sorted(Path(programs_root).glob("build_request*/result.json")):
        res = json.loads(d.read_text())
        if res.get("ok") and (dig := (res.get("program") or {}).get("digest")):
            out.setdefault(str(dig), str(d.parent))
    return out


@functools.lru_cache(maxsize=4)
def record1(bundle: str, row_dir: str) -> SimpleNamespace:
    """A single-GPU Commit's sealed bundle opened against its record (`deferred_rows.of_record`), once per process."""
    return DRW.of_record(bundle, row_dir)


def ingress_of(ctx: Mapping[str, Any]) -> dict[str, dict]:
    """Each request's body by its I1 id (``r<index>``, as the Programs of record and `advice_of_workload` name it), from the
    Commit's record: its prompt token ids, ``max_tokens`` and the sampling parameters the record served it under."""
    from verity_vllm.pipeline import c2_replay as C2

    rl, sampling = C2.request_labels(ctx["requests"]), ctx.get("sampling") or {}
    return {rl[str(q.get("request_id"))]: {"prompt": list(q["prompt_token_ids"]), "max_tokens": int(q["max_tokens"]),
                                            **{k: sampling.get(k) for k in BODY_SAMPLING if k in sampling}} for q in ctx["requests"]}


def i2_of_record(bundle: str, row_dir: str, workload: str, hf_config: str | None, cap_bits: int = CAP_BITS,
                 transfer_gates: Sequence[str] = ()) -> I2Compiler:
    """I2 over a single-GPU Commit: φ from the row's declared inputs (`score.phi_of`: its row id, the workload file the record
    served, refused unless its digest is the record's, and the checkpoint's ``config.json``), the bodies from the record, listing
    ``transfer_gates``."""
    from verity_vllm.predict.score import phi_of
    from verity_vllm.predict.steps import Registered

    r = record1(str(bundle), str(row_dir))
    doc = json.loads(Path(workload).read_text())
    got = hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if got != r.row.get("workload_sha256"):
        raise ValueError(f"{workload}: workload sha256 {got[:16]} is not the record's {str(r.row.get('workload_sha256'))[:16]}")
    phi = phi_of({"row": r.row["label"], "workload": doc}, config=str(hf_config or LLAMA32_1B_CONFIG))
    return I2Compiler(Registered.of(phi, int(doc.get("seed") or 0), cap_bits), ingress_of(r.ctx), str(row_dir),
                      transfer_gates=tuple(dict.fromkeys(map(str, transfer_gates))))


def compiler_of(tp_commit: str, build_root: str, cap_bits: int = CAP_BITS, transfer_gates: Sequence[str] = ()) -> Capped:
    return Capped(AW.Traced.for_commit(tp_commit, build_root, transfer_gates), cap_bits)


def compiler_of_args(a: argparse.Namespace) -> Capped:
    gates = getattr(a, "transfer_gate", None) or ()
    if single(a):
        return Capped(i2_of_record(a.bundle, a.row_dir, a.workload, a.hf_config, a.cap_bits, gates), a.cap_bits)
    return compiler_of(a.tp_commit, a.build_root, a.cap_bits, gates)


def single(a: argparse.Namespace) -> bool:
    """A single-GPU Commit's run (``--bundle``), not a TP Commit's."""
    return bool(getattr(a, "bundle", None))


def compile_advice(compiler: Capped, data: Any, codec: Any = None) -> list[dict]:
    """The compiler on any input: the requests it compiles, or a `ValueError` naming why it refuses the input, a function of the
    input alone. Refused in order: not bytes; over the cap ℓ (before any of it is read); not I1 advice (`MalformedAdvice`, the
    codec's first broken rule); not its canonical encoding; a request the compiler can't run (a chunked prefill, a prefix-cache
    hit, a preemption, a gap: `advice_windows.requests_of`, or I2's own refusals, `I2Compiler.requests`)."""
    if not isinstance(data, bytes):
        raise ValueError(f"the advice is a byte string, not a {type(data).__name__}")
    if 8 * len(data) > compiler.cap_bits:
        raise ValueError(f"the advice is {8 * len(data)} bits, over the registered cap ℓ = {compiler.cap_bits} bits")
    c = AW.codec_of(codec)
    advice = c.decode(data)
    if c.encode(advice) != data:
        raise ValueError("the advice's bytes are not its canonical encoding")
    own = getattr(compiler.inner, "requests", None)
    return own(data) if own is not None else AW.requests_of(advice)


def commit_advice(compiler: Capped, data: bytes) -> tuple[list[dict], dict, dict]:
    """The prover's side: the requests compiled from the advice (`compile_advice`), the window's binding (the compiler's digest
    and the root of the advice's hiding commitment, `advice_windows.of_advice`) and the opening (`advice_windows.opening`)."""
    requests = compile_advice(compiler, data)
    _reqs, bound, reg = AW.of_advice(compiler, data)
    return requests, bound, AW.opening(data, reg)


def open_advice(compiler: Capped, doc: Any) -> tuple[list[dict], dict]:
    """The verifier's side of the opening (the named gap): the requests it compiles from the opened advice and the binding the
    registration must carry (`advice_windows.of_opening`: the root recomputed from the advice and its salts). A malformed opening is
    a ValueError, as `compile_advice`'s refusals are."""
    if not isinstance(doc, Mapping) or doc.get("scheme") != AW.SCHEME or not isinstance(doc.get("advice"), str):
        raise ValueError(f"an advice opening is a {AW.SCHEME} document with the advice in hex")
    try:
        data = bytes.fromhex(doc["advice"])
    except ValueError as e:
        raise ValueError(f"an advice opening holds the advice in hex: {e}") from e
    return compile_advice(compiler, data), AW.of_opening(compiler, doc)[1]


def compiled(a: argparse.Namespace) -> tuple[Capped, list[dict], dict, dict]:
    """The verifier's compiler, requests and binding from the opening ``a.opening``, and the opening."""
    compiler = compiler_of_args(a)
    opening = json.loads(Path(a.opening).read_text())
    return (compiler, *open_advice(compiler, opening), opening)


def stand_in1(r: SimpleNamespace) -> dict[str, Any]:
    """A single-GPU window's link to its Commit's commitment: ``of`` the commitment its store names (``rows-sha512``), the run
    root, the bundle's manifest digest and the command that made the rows, and ``link``."""
    scheme = r.store.commit_scheme
    return {"of": {"commitment": scheme, "run_root": bytes(r.run.run_root).hex(), "bundle_manifest_sha256": r.bundle["manifest_sha256"],
                   "rows": ROWS1},
            "link": f"provenance, not proof: prove_path.py opened the sealed replay bundle against the Commit's record and committed "
                    f"these rows from openings verified against its {scheme} run root; no circuit opens the {scheme} leaves"}


def window1(a: argparse.Namespace, compiler: Capped, requests: list[dict], bound: Mapping[str, Any]) -> tuple[Any, dict]:
    """The verifier's own window of a single-GPU Commit (`serving_rows.population` over ``compiler``'s Programs for the advice's
    ``requests``, the record's run root), from the record alone: no committed value is read. -> (the window, what `source1` holds
    the registration to)."""
    from verity_vllm.pipeline import serving_rows as SRW

    r = record1(str(a.bundle), str(a.row_dir))
    p = compiler.load(0, 1, requests)
    window, _rows = SRW.population(p.progs, json.loads(Path(a.partition).read_text()), bytes(r.run.run_root).hex(), p.dirs,
                                   arrivals=p.arrivals, transfer_gates=compiler.transfer_gates)
    return window, {"commitment": r.store.commit_scheme, "served_programs": {str(k): str(x) for k, x in sorted(p.digs.items())},
                    "stand_in": stand_in1(r), **{k: dict(bound).get(k) for k in SRT.BINDING}}


def source1(registered: dict[str, Any] | None, mine: Any, held: dict[str, Any]) -> dict[str, Any]:
    """A single-GPU window's registration held to the verifier's own (`window1`): a ``stand-in`` of the record's run root, a
    SHA-512 root (``sha512``), under the commitment the record's store names, with `stand_in1`'s ``of`` and ``link``, the
    Programs the verifier loaded and the compiler and advice it compiled (`tp.serving_rows.BINDING`)."""
    w, si = registered or {}, held["stand_in"]
    ok = {"kind": w.get("kind") == "stand-in", "run": w.get("run") == mine.run, "sha512": len(mine.run) == 2 * DRW.SERVED_ROOT,
          "commitment": w.get("commitment") == held["commitment"], "of": w.get("of") == si["of"], "link": w.get("link") == si["link"],
          "served_programs": w.get("served_programs") == held["served_programs"], **{k: w.get(k) == held.get(k) for k in SRT.BINDING}}
    return {"ok": all(ok.values()), "codes": [c for c, v in ok.items() if not v],
            "detail": {"kind": w.get("kind"), "run": w.get("run"), "commitment": w.get("commitment"), "of": w.get("of")}}


def source_of(a: argparse.Namespace) -> Callable[[dict | None, Any, dict], dict]:
    return source1 if single(a) else SRT.source


def window_of(a: argparse.Namespace, r: int, compiler: Capped, requests: list[dict], bound: Mapping[str, Any]) -> tuple[Any, dict]:
    if single(a):
        return window1(a, compiler, requests, bound)
    return AW.window_of_advice(a.tp_commit, r, a.partition, compiler, requests, bound)


def commit1(a: argparse.Namespace, compiler: Capped, requests: list[dict], bound: Mapping[str, Any], out: Path) -> dict:
    """A single-GPU Commit's stand-in window into ``out`` (`serving_rows.after_commit`, as `deferred_rows.from_bundle` writes one)
    over ``compiler``'s Programs for the advice's ``requests``, ``bound`` in its provenance: every value read through an opening
    verified against the record's run root, the weights (none for an activation template) each opened on its first read
    against the committed weights root (`cpu_replay.OnRead`)."""
    from verity_vllm.check.oracle_compare import OpenedReader
    from verity_vllm.check.replay.index import ProgramIndex
    from verity_vllm.pipeline import cpu_replay as CPU
    from verity_vllm.pipeline import serving_rows as SRW

    r = record1(str(a.bundle), str(a.row_dir))
    if r.store.commit_scheme != SRT.ROWS_SCHEME:
        raise ValueError(f"the Commit's store commits under {r.store.commit_scheme}, not {SRT.ROWS_SCHEME}: no stand-in names a SHA-512 root")
    p = compiler.load(0, 1, requests)
    k0 = next(iter(p.progs))
    tree = json.loads((r.bdir / "weights" / "tree.json").read_text())
    words = CPU.OnRead(r.bdir, tree, (r.row["commit"].get("weight_registration") or {}).get("root"), None)
    bmap = json.loads((r.out / f"binding_map_p{r.pair}.json").read_text())
    res = SRW.after_commit(str(out), partition_file=str(a.partition), programs=p.progs, bmap=bmap, committed=OpenedReader(r.store, r.run),
                           weights=CPU.provider(words, ProgramIndex(p.progs[k0], program_dir=p.dirs[k0]).prefix, set()),
                           manifest=r.manifest if isinstance(r.manifest, dict) else None, run_root=r.run.run_root,
                           row=r.row.get("label"), arrivals=p.arrivals, program_dirs=p.dirs, digests=p.digs,
                           committer={"from": "replay bundle", "pair": r.pair, "source": r.ctx.get("code")}, workers=a.workers,
                           stand_in=stand_in1(r), binding=dict(bound), transfer_gates=compiler.transfer_gates)
    return {k: res.get(k) for k in ("run", "population", "partition_digest", "registration_digest", "seconds")}


def i2_against_record(a: argparse.Namespace, requests: list[dict]) -> dict | None:
    """A single-GPU run's I2 Programs against the Build's Programs of record for the record's requests (`deferred_rows.of_record`,
    by the manifest's lookup): each request's two digests and whether they are equal."""
    if not single(a):
        return None
    of = record1(str(a.bundle), str(a.row_dir)).digests
    units = {q["request_id"]: {"i2": q["digest"], "of_record": of.get(q["request_id"]), "equal": q["digest"] == of.get(q["request_id"])}
             for q in requests}
    return {"units": units, "equal": len(units) == len(of) and all(u["equal"] for u in units.values())}


@contextlib.contextmanager
def forging(member: int, unit: int) -> Iterator[None]:
    """Inside, serving rows commit member ``member``'s unit ``unit`` with bit 0 of its first output word flipped, before its commit
    strings and roots are made over it."""
    from verity_vllm.pipeline import serving_rows as SRW

    real = SRW.read_all

    def read_all(ms, rows, store, weights, arrivals=None):
        got = real(ms, rows, store, weights, arrivals)
        port = ms[member].outs[0][0]
        got[member][0]["outs"][port][unit, 0] ^= 1
        return got

    SRW.read_all = read_all
    try:
        yield
    finally:
        SRW.read_all = real


def ranks_of(a: argparse.Namespace) -> tuple[int, list[int]]:
    """The run's world and the windows to commit or audit (``--ranks``, default all); a single-GPU run's one window is rank 0."""
    world = 1 if single(a) else int(SRT.record(a.tp_commit)["world"])
    ranks = [int(x) for x in a.ranks.split(",")] if a.ranks else list(range(world))
    if bad := [r for r in ranks if not 0 <= r < world]:
        raise SystemExit(f"ranks {bad} are not ranks of the world of {world}")
    return world, ranks


def run_root_of(a: argparse.Namespace) -> str:
    """The root the windows are registered under: the TP run root, or a single-GPU Commit's run root."""
    if single(a):
        return bytes(record1(str(a.bundle), str(a.row_dir)).run.run_root).hex()
    return SRT.record(a.tp_commit)["tp_run_root"]


def in_processes(fn, jobs: int, items: list) -> Iterator[tuple[Any, Any]]:
    """``fn(item)`` of each item, in up to ``jobs`` spawned processes (a rank's Programs don't pickle, so a process does a rank);
    -> (item, result) as each finishes."""
    if jobs <= 1 or len(items) <= 1:
        for x in items:
            yield x, fn(x)
        return
    with ProcessPoolExecutor(min(jobs, len(items)), mp_context=multiprocessing.get_context("spawn")) as ex:
        futures = {ex.submit(fn, x): x for x in items}
        for f in as_completed(futures):
            yield futures[f], f.result()


def free_ports(span: int, bases: Iterable[int] | None = None) -> int:
    """A base port whose ``span`` loopback ports are all free now, the first of ``bases`` (default every base of `PORTS` in
    random order). An audit's sessions listen at fixed offsets from it (``a5.RANK_PORTS`` a rank); a fixed default made two
    audits on one node collide, the second's verifier failing to bind and its prover reaching the first's, which refused it."""
    import random
    import socket

    if bases is None:
        bases = list(range(PORTS.start, PORTS.stop - span + 1, span))
        random.SystemRandom().shuffle(bases)
    for base in bases:
        try:
            with contextlib.ExitStack() as held:
                for port in range(base, base + span):
                    held.enter_context(socket.socket()).bind(("127.0.0.1", port))
            return base
        except OSError:
            continue
    raise SystemExit(f"no {span} free loopback ports in a row to listen on; give --port")


def commit_rank(job: tuple[argparse.Namespace, int, list[dict], dict]) -> dict:
    a, r, requests, bound = job
    compiler = compiler_of_args(a)
    forge = [int(x) for x in a.forge.split(":")] if a.forge else None
    out = Path(a.out) / f"rank{r}"
    with forging(*forge) if forge else contextlib.nullcontext():
        if single(a):
            got = commit1(a, compiler, requests, bound, out)
        else:
            got = AW.commit_window(a.tp_commit, r, a.partition, out, compiler, requests, bound, workers=a.workers)
    return {**got, "forged": {"member": forge[0], "unit": forge[1]} if forge else None}


def commit(a: argparse.Namespace) -> int:
    """The advice committed once, so every window's registration binds one root, then each window."""
    _world, ranks = ranks_of(a)
    t0 = time.time()
    requests, bound, opening = commit_advice(compiler_of_args(a), Path(a.advice).read_bytes())
    compile_s = round(time.time() - t0, 1)
    Path(a.opening).write_text(json.dumps(opening) + "\n")
    rows = {}
    for (_a, r, _q, _b), got in in_processes(commit_rank, a.jobs, [(a, r, requests, bound) for r in ranks]):
        rows[str(r)] = got
        print(f"[prove-path commit] rank {r}: {json.dumps(got, sort_keys=True, default=str)}", flush=True)
    doc = {"binding": bound, "ranks": rows, "compile_s": compile_s, **({"i2": i2_against_record(a, requests)} if single(a) else {})}
    Path(a.out, "commit.json").write_text(json.dumps(doc, indent=1, sort_keys=True, default=str))
    return 0


def more_negatives(a: argparse.Namespace, r: int, world: int, compiler: Capped, mine: Any, held: dict, served: Path,
                   opening: dict) -> dict[str, dict]:
    """What the verifier refuses before any draw about the advice: a registration naming another compiler or another advice root,
    the advice's opening under another salt (which opens no registered root), a registration binding no compiler or advice (a
    window over Programs of record passed off as the advice's), one whose compiler names another cap ℓ, one under the compiler
    listing no transfer gates (where it lists some: a window sampling them, as another compiler's), and another rank's."""
    src = source_of(a)
    w = json.loads((served / "registration.json").read_text())["window"]
    cases = {f"registration-naming-another-{key} (source)": {**w, key: {**(w.get(key) or {}), fld: "00" * 64}}
             for key, fld in (("compiler", "digest"), ("advice", "root"))}
    cases["registration-binds-no-compiler-or-advice (source)"] = {k: v for k, v in w.items() if k not in SRT.BINDING}
    cases["registration-names-another-cap (source)"] = {**w, "compiler": {"name": CAPPED,
                                                                          "digest": Capped(compiler.inner, 2 * compiler.cap_bits).digest}}
    if compiler.transfer_gates:
        bare = Capped(replace(compiler.inner, transfer_gates=()), compiler.cap_bits)
        cases["registration-under-the-compiler-listing-no-transfer-gates (source)"] = {**w, "compiler": {"name": CAPPED,
                                                                                                         "digest": bare.digest}}
    theirs = served.parent / f"rank{(r + 1) % world}" / "registration.json"
    if world > 1 and theirs.exists():
        cases["registration-of-another-rank (source)"] = json.loads(theirs.read_text())["window"]
    neg = {}
    for name, x in cases.items():
        v = src(x, mine, held)
        neg[name] = {"refused": not v["ok"], "codes": v["codes"]}
    salts = list(opening["salts"])
    salts[0] = f"{int(salts[0][:2], 16) ^ 1:02x}{salts[0][2:]}"
    _reqs, resalted = AW.of_opening(compiler, {**opening, "salts": salts})
    v = src(w, mine, {**held, **resalted})
    neg["opening-under-another-salt (source)"] = {"refused": not v["ok"], "codes": v["codes"]}
    return neg


def advice_before_draw(receipt: Mapping[str, Any], rec: Mapping[str, Any], bound: Mapping[str, Any], coins: str = "verifier") -> dict:
    """That the advice was fixed before the coins, checked rather than assumed, as `pouw.credit.draw_root` checks the credit bits:
    the digest of the registration the verifier holds the opening to (`registration.digest`, which covers the window's
    ``compiler`` and ``advice`` root) must be the one the draw's receipt names, the receipt the stream wrote when it received the
    registration and before its select read any coin (a5's ``verifier-coins-before-the-registration``: a select on a
    registration the stream never received reads none; on the challenger's coins, ``coins="challenger"``, the coins are also
    committed to that digest before any is read); and that registration must bind the compiler and the root the verifier
    opened. -> {ok, codes, detail}: ``after-the-draw`` where the draw followed another registration, or another stream's,
    ``compiler`` or ``advice`` where the registration binds another."""
    import a5 as A5

    w = rec.get("window") or {}
    drew = {"verifier": A5.VERIFIER, "challenger": A5.CHALLENGER}[coins]
    ok = {"after-the-draw": receipt.get("registration") == R.digest(rec) and receipt.get("verifier") == drew,
          "compiler": w.get("compiler") == bound["compiler"], "advice": w.get("advice") == bound["advice"]}
    return {"ok": all(ok.values()), "codes": [c for c, v in ok.items() if not v],
            "detail": {"drawn_after": receipt.get("registration"), "registration": R.digest(rec), "received": receipt.get("received")}}


def advice_after_draw(compiler: Capped, receipt: Mapping[str, Any], rec: Mapping[str, Any], opening: Mapping[str, Any], mine: Any,
                      held: dict, source: Callable[[dict | None, Any, dict], dict] = SRT.source, coins: str = "verifier") -> dict:
    """The negative: once the coins are read, the prover commits the same advice again (fresh salts, so a fresh root) and hands
    in a registration binding that root, with its opening. Every check before the draw passes on it (``source`` against the new
    opening: the root is a true commitment of the advice), and `advice_before_draw` refuses it (``after-the-draw``): the draw
    followed the registration received first."""
    data = bytes.fromhex(opening["advice"])
    late = AW.opening(data, AW.commit_advice(compiler, data))
    _reqs, bound = open_advice(compiler, late)
    late_rec = {**rec, "window": {**rec["window"], "advice": bound["advice"]}}
    src = source(late_rec["window"], mine, {**held, **bound})
    v = advice_before_draw(receipt, late_rec, bound, coins)
    return {"refused": not v["ok"], "codes": v["codes"], "before_draw_checks_pass": src["ok"]}


def forged_row(s: dict, forged: dict | None) -> dict:
    """A forged window's verdict: its registration passed every check before the draw, the forged unit was drawn, and the outcome
    is a reject with Lean's verifier refusing a drawn member."""
    verdicts = (s.get("audit") or {}).get("verdicts") or {}
    lean = {k: v for k, v in verdicts.items() if k.startswith("lean-verify/")}
    unit = None
    if forged is not None:
        unit = next(m["base"] for m in s.get("members") or [] if m["k"] == forged["member"]) + forged["unit"]
    checks = {"checks_before_draw": bool(s.get("checks_before_draw")) and all(s["checks_before_draw"].values()),
              "forged_unit_drawn": unit is not None and unit in (s.get("draw") or []),
              "rejected": (s.get("audit") or {}).get("accepted") is False,
              "lean_refused": any(v is False for v in lean.values())}
    return {"ok": all(checks.values()), "checks": checks, "why": (s.get("audit") or {}).get("why")}


def per_unit(s: dict) -> dict[str, dict]:
    """Each drawn member's session timing per drawn unit (the units its one statement proves, ``draw_counts``; C-Flock's own
    ``units`` count its virtual units): proving (C-Flock's ``prove_total_s`` and the session's wall), M0's and Lean's verify,
    and the proof's bytes over its reps."""
    out = {}
    for key, t in (s.get("timing") or {}).items():
        tm, n = t.get("timing") or {}, (s.get("draw_counts") or {}).get(key) or 0
        pb = tm.get("proof_bytes")
        out[key] = {"units": n, "k_log": tm.get("k_log"), "proof_bytes": sum(pb) if isinstance(pb, list) else pb,
                    **{f"{x}_per_unit": round(v / n, 2) if n and isinstance(v, int | float) else None
                       for x, v in (("prove_s", tm.get("prove_total_s")), ("session_s", t.get("session_s")),
                                    ("verify_s", t.get("verify_s")), ("lean_verify_s", t.get("lean_verify_s")))}}
    return out


def audit_rank(job: tuple[argparse.Namespace, int, int]) -> dict:
    """Rank ``r``'s audit (`rank_audit`). On the challenger's coins, a rank that ends without its draw says so, so the run root's
    one session closes once every other rank has drawn."""
    a, world, r = job
    try:
        return rank_audit(a, world, r)
    finally:
        if a.coins is not None:
            a.coins.skip(r)


def rank_audit(a: argparse.Namespace, world: int, r: int) -> dict:
    import a5 as A5

    t0 = time.time()
    compiler, requests, bound, opening = compiled(a)
    mine, held = window_of(a, r, compiler, requests, bound)
    window_s = round(time.time() - t0, 2)
    served, out, src = Path(a.windows) / f"rank{r}", Path(a.out) / f"rank{r}", source_of(a)
    more = more_negatives(a, r, world, compiler, mine, held, served, opening)
    s, ok = A5.audit(a, A5.template_sets(Path(a.template_sets)), out, mine, served, lambda w: src(w, mine, held),
                     a.port + A5.RANK_PORTS * r,
                     {"rank": r, "tp": held.get("tp"), "binding": bound, "requests": requests, "verifier_window_s": window_s}, more,
                     rank=r)
    row = {"ok": ok, "seconds": round(time.time() - t0, 1), "verifier_window_s": window_s, "per_unit": per_unit(s),
           **{x: s.get(x) for x in ("run_root", "population", "members", "registration", "source", "refused", "checks_before_draw",
                                    "audit", "draw", "draw_counts", "negatives", "coins", "timing")}}
    if s.get("draw") is not None:
        coins = "challenger" if a.coins is not None else "verifier"
        rec, receipt = (json.loads((out / f).read_text()) for f in ("registration.json", "receipt.json"))
        row["advice_before_draw"] = advice_before_draw(receipt, rec, bound, coins)
        row["negatives"] = {**(row["negatives"] or {}),
                            "advice-root-after-the-draw": advice_after_draw(compiler, receipt, rec, opening, mine, held, src, coins)}
        ordered = row["advice_before_draw"]["ok"] and row["negatives"]["advice-root-after-the-draw"]["refused"]
    else:
        ordered = False
    row["ok"] = ok and ordered
    if a.forged:
        forged = ((json.loads((Path(a.windows) / "commit.json").read_text())["ranks"]).get(str(r)) or {}).get("forged")
        row["forged"] = {"unit": forged, **forged_row(s, forged)}
        row["ok"] = row["forged"]["ok"] and ordered
    return row


def coins_of(a: argparse.Namespace) -> dict[str, Any]:
    """The summary's account of the coins every window's stream draws on (module docstring)."""
    if a.challenger:
        return {"source": "challenger", "challenger": a.challenger,
                "what": "each window's service.Stream on the auditor's challenger's coins (a5_coins: the run root's one session, "
                        f"{a.challenger}), committed to the window's registration before any is read"}
    return {"source": "verifier", "what": "each window's service.Stream on the verifier's own live coins (the Lean's "
                                          "IO.getRandomBytes), read once the stream received its registration"}


def audit(a: argparse.Namespace) -> int:
    import a5 as A5
    import a5_coins as AC

    world, ranks = ranks_of(a)
    if a.port is None:
        a.port = free_ports(A5.RANK_PORTS * world)
    root = run_root_of(a)
    a.coins, a.verifier_coins, a.window_root = None, not a.challenger, root
    a.rule = json.loads(Path(a.partition).read_text()).get("law")
    if a.forged:
        os.environ["FC_ZK_SELF_CHECK"] = "skip"
    if a.challenger and len(ranks) > 1 and a.jobs < len(ranks):
        raise SystemExit(f"--challenger draws every rank from the run root's one session, which ends once every rank has drawn: "
                         f"audit the {len(ranks)} ranks at once (--jobs {len(ranks)}), or fewer ranks (--ranks)")
    t0 = time.time()
    compiler, requests, bound, _opening = compiled(a)
    compile_s = round(time.time() - t0, 1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {**({"run_root": root} if single(a) else {"tp_run_root": root}), "world": world, "compiler": bound["compiler"],
                     "advice": bound["advice"], "cap_bits": compiler.cap_bits,
                     "inner_compiler": {"name": compiler.inner.name, "digest": compiler.inner.digest}, "compile_s": compile_s,
                     "transfer_gates": list(compiler.transfer_gates),
                     "gap": "the advice is opened to the verifier until the compile unit exists (README.md)",
                     **({"i2": i2_against_record(a, requests)} if single(a) else {}),
                     "requests": requests, "rule": a.rule, "delta": a.delta, "forged": a.forged, "coins": coins_of(a), "ranks": {}}
    if a.challenger:
        manager, a.coins, finisher = AC.start(a.challenger, root, ranks, out / "challenger", a.main)
    try:
        for (_a, _w, r), row in in_processes(audit_rank, a.jobs, [(a, world, r) for r in ranks]):
            summary["ranks"][str(r)] = row
            (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    finally:
        if a.coins is not None:
            for r in ranks:
                a.coins.skip(r)
            finisher.join()
            published = a.coins.evidence()
            summary["coins"].update(trusted=published.get("trusted"), error=published.get("error"), **(published.get("plan") or {}))
            manager.shutdown()
    summary["ok"] = sorted(summary["ranks"]) == sorted(map(str, ranks)) and all(x["ok"] for x in summary["ranks"].values()) \
        and not summary["coins"].get("error")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    shown = {r: {x: s.get(x) for x in ("ok", "population", "draw", "draw_counts", "seconds", "per_unit")}
             | {"advice_before_draw": (s.get("advice_before_draw") or {}).get("ok")}
             | ({"forged": s["forged"]} if a.forged else {"accepted": (s.get("audit") or {}).get("accepted"),
                                                         "negatives_refused": all((s.get("negatives") or {"none": False}).values())})
             for r, s in summary["ranks"].items()}
    print("PROVE-PATH " + json.dumps({"run_root" if single(a) else "tp_run_root": root, "forged": a.forged, "ok": summary["ok"],
                                      "coins": summary["coins"]["source"], "transfer_gates": summary["transfer_gates"],
                                      **({"i2_equal": summary["i2"]["equal"]} if single(a) else {}), "ranks": shown},
                                     sort_keys=True, default=str))
    return 0 if summary["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("advice", help="a declared run's I1 advice from its workload file")
    p.add_argument("--workload", required=True)
    p.add_argument("--out", required=True, help="the advice's file: I1's canonical encoding")
    for name in ("commit", "audit"):
        p = sub.add_parser(name)
        p.add_argument("--tp-commit", help="a TP Commit's record (its --out: runs.jsonl, versions.json)")
        p.add_argument("--build-root", help="with --tp-commit: each rank's Programs of record in rank<r>/ (the stand-in compiler's)")
        p.add_argument("--bundle", help="instead of --tp-commit: a single-GPU Commit's sealed replay bundle (<row dir>/commit/"
                       "replay_bundle_p0), its window's Programs I2's")
        p.add_argument("--row-dir", help="with --bundle: the row dir (the Build's Programs, manifest.json, commit/)")
        p.add_argument("--workload", help="with --bundle: the workload file the Commit served (φ's declared inputs, I2's)")
        p.add_argument("--hf-config", help="with --bundle: the checkpoint's config.json (default unsloth/Llama-3.2-1B's)")
        p.add_argument("--opening", required=True, help="the advice's opening (advice_windows.opening): commit writes it, the "
                       "prover's private file; audit reads it, the named gap")
        p.add_argument("--cap-bits", type=int, default=CAP_BITS, help="ℓ, the public cap on the advice's length in bits, "
                       "registered in the compiler's digest")
        p.add_argument("--partition", required=True, help="the verifier's partition file: {templates, law, scope?}")
        p.add_argument("--ranks", default=None, help="comma-separated (default every rank)")
        p.add_argument("--jobs", type=int, default=1, help="ranks at once, each in its own process")
        p.add_argument("--transfer-gate", action="append", default=[], help="a template the registered compiler lists as holding "
                       "transfer gates, in its digest, every unit of which is proved (repeatable; the partition's law is work:K)")
        p.add_argument("--out", required=True)
    p = sub.choices["commit"]
    p.add_argument("--advice", required=True, help="the advice's file: I1's canonical encoding (the prover's, private)")
    p.add_argument("--forge", help="MEMBER:UNIT: commit that unit with bit 0 of its first output word flipped")
    p.add_argument("--workers", type=int, default=None, help="commit workers per rank (default min(16, CPUs))")
    p = sub.choices["audit"]
    p.add_argument("--windows", required=True, help="the prover's windows, rank<r>/ each (prove_path.py commit --out)")
    p.add_argument("--template-sets", required=True, help="one input set per member template, for the lowering")
    p.add_argument("--m0-pythonpath", required=True)
    p.add_argument("--bin", required=True)
    p.add_argument("--lean", required=True)
    p.add_argument("--firewall", required=True)
    p.add_argument("--delta", type=float, default=0.5)
    p.add_argument("--port", type=int, default=None, help="the sessions' base port, rank r's from base + a5.RANK_PORTS·r "
                   "(default a base whose ports are free now, `free_ports`)")
    p.add_argument("--session-timeout", type=float, default=14400)
    p.add_argument("--forged", action="store_true", help="the windows hold a forged unit (commit --forge): every window must be refused")
    p.add_argument("--challenger", choices=["local"], help="draw each window on the auditor's challenger's coins, the run root's one "
                   "session (a5_coins; local: a development session on this machine), not the verifier's own")
    p.add_argument("--main", help="with --challenger: the verifier's own copy of main's first-parent history (git rev-list "
                   "--first-parent origin/main), which must hold the commit the session's host pins")
    a = ap.parse_args(argv)
    if a.cmd == "advice":
        c = AW.codec_of()
        advice = AW.advice_of_workload(json.loads(Path(a.workload).read_text()), c)
        data = c.encode(advice)
        AW.requests_of(advice)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_bytes(data)
        print(json.dumps({"advice": a.out, "requests": len(advice.requests), "steps": len(advice.steps), "bits": c.total_bits(advice)}))
        return 0
    if single(a) == bool(a.tp_commit) or (a.tp_commit and not a.build_root) or (single(a) and not (a.row_dir and a.workload)):
        ap.error("one committed run: --tp-commit DIR --build-root DIR, or --bundle DIR --row-dir DIR --workload F")
    if a.cmd == "audit" and a.challenger and not a.main:
        ap.error("--challenger needs --main: the verifier's `git rev-list --first-parent origin/main`, which must hold the commit "
                 "the session's host pins")
    return commit(a) if a.cmd == "commit" else audit(a)


if __name__ == "__main__":
    raise SystemExit(main())
