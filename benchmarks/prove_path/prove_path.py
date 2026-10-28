"""Prove path: a committed run and its advice through the proof system, for network accounting under arbitrary vLLM usage.

Each window's step units are Programs a registered compiler made from the advice (`pipeline.advice_windows`), not a declared
workload's. The advice is private: the prover commits I1's canonical encoding under hiding rows (`advice_windows.commit_advice`),
and each window's registration binds the compiler's digest and that commitment's root, before any coin. The prover commits each
window over the compiler's Programs from its sealed bundle. The verifier, handed the commitment's opening (the named gap: the
advice is opened to the verifier until the compile unit exists, README.md), compiles the advice with its own copy of the
compiler, derives its own window and audits it through the one-stage audit's `service.Stream` (`benchmarks/one_stage/a5.py`'s
`audit`): register, select on the verifier's own live coins (read only after the receipt), Flock `--zk` behind the Lean
ZK server on the sampled proof blocks, Lean's verdicts, and the outcome, which the Lean verifier of record's `Run.run` must agree with. It
then checks that the advice was fixed before the sample (`advice_before_draw`).

Two kinds of committed run:

* **A TP Commit** (``--tp-commit DIR --build-root DIR``): one window per rank, each from the rank's sealed bundle, under the TP
  run root. The compiler is `Capped` over `advice_windows.Traced` (the Build's Programs of record, refused unless of record).
* **A single-GPU Commit** (``--bundle DIR --row-dir DIR --workload F``, TP1): one window, ``rank0/``, from the Commit's sealed
  bundle (``<row dir>/commit/replay_bundle_p0``) under its run root. The compiler is `Capped` over `I2Compiler`: I2's
  `predict.steps.compile_steps` on the advice and the requests' bodies, each Program the one I2 compiles, its body read from
  the Build's files of that digest and held to it (`I2Compiler.load`, `build_of`). What no digest covers (the rows' names,
  the parameters' names and types, ``supplied``, the correspondence, ``prefix_cache.json``) the verifier reads only from its
  own Build of each Program, which its own compile writes (`own_builds`: I2 names each Program as vLLM's trace of it does,
  `predict.naming`, so the audit needs no vLLM), never from the row dir; the prover's Builds are its row dir's. The window
  is a stand-in of the record's SHA-512 rows root (`stand_in1`).
* **A served window** (``--serve-run DIR --window K --row-dir DIR``, TP1): one closed Commit window of a `verity-vllm
  serve-openai` run, its window ``rank0/``. The run dir is serve-openai's (``serve.json``, ``windows.jsonl``, ``requests.jsonl``)
  with `verity-vllm advice`'s ``advice/w<K>/`` (``registration.json``, public; ``private/advice.bin`` and ``private/opening.json``,
  the prover's); the window's Commit is its sealed row dir (``commit/replay_bundle_p0``, the Commit's records, the
  Build's Programs, ``manifest.json``). The compiler is `Capped` over `I2Compiler` under φ of the served engine
  (`phi_of_serve`), the bodies each request's ingress record (`serve_ingress`), named as `verity-vllm advice` names it
  (`served_identity`: compile_steps over the tree's code identity), its digest a single-GPU window's I2's over that code, so
  it covers what the compile reads of ``serve.json`` (`I2Compiler.digest`). The advice is committed once, by the server:
  the prover commits the window under the published binding (no re-commit), and both sides hold the published records to their
  own (`serve_check`): the registration names this window, a compiler whose digest carries the cap ℓ and is the verifier's own,
  and the root the opening opens; the root serve-openai published for the window is a SHA-512 root, the row dir's run root.

    prove_path.py advice --workload F --out FILE
    prove_path.py commit (--tp-commit DIR --build-root DIR | --bundle DIR --row-dir DIR --workload F [--hf-config F] \\
        | --serve-run DIR --window K --row-dir DIR [--hf-config F] [--target F]) [--advice FILE] [--opening FILE] --partition F \\
        --out WINDOWS [--cap-bits L] [--ranks 0,1] [--jobs N] [--forge MEMBER:UNIT] [--transfer-gate T ...]
    prove_path.py audit (the same committed run) [--opening FILE] --windows WINDOWS --partition F --template-sets DIR \\
        --m0-pythonpath PATH --bin flock-circuit --lean flock-verify --firewall flock-firewall --out DIR [--cap-bits L] \\
        [--ranks 0,1] [--jobs N] [--forged] [--plain] [--transfer-gate T ...] [--challenger local|gate:USER@HOST --main FILE]
        [--peer NAME --challenger-key RECORD [--gate-key KEY]] [--builds DIR]
    prove_path.py compiler --serve-run DIR [--hf-config F] [--target F] [--cap-bits L] [--transfer-gate T ...]
    prove_path.py weights-registration --serve-run DIR --checkpoint-dir DIR --key FILE --out FILE [--checkpoints F] \\
        [--tables DIR] [--hf-config F] [--target F]
    prove_path.py serve-standin --bundle DIR --row-dir DIR --workload F --out RUN [--hf-config F] [--cap-bits L]
    prove_path.py circuit --serve-run DIR --window K --partition F --advice FILE --out FILE [--hf-config F] [--target F] \\
        [--cap-bits L] [--transfer-gate T ...]
    prove_path.py window-circuit --serve-run DIR --window K --row-dir DIR --advice FILE --out FILE [--cap N] [--description] \\
        [--class SLOTS,PROMPT,OUTPUTS[,LANES,EVERY]] [--pin NAME=ROOT ...] [--pins FILE ...] [--hf-config F] [--target F] \\
        [--cap-bits L]
    prove_path.py class-size --serve-run DIR --window K --row-dir DIR --advice FILE \\
        [--class SLOTS,PROMPT,OUTPUTS[,LANES,EVERY]] [--probe SLOTS,PROMPT,OUTPUTS] [--hf-config F] [--target F] [--cap-bits L]
    prove_path.py window-commit --serve-run DIR --window K --row-dir DIR --advice FILE --key FILE --registration FILE \\
        --checkpoint-dir DIR --out DIR [--checkpoints F] [--class SLOTS,PROMPT,OUTPUTS[,LANES,EVERY]] [--law L] [--cap N] \\
        [--workers N] [--hf-config F] [--target F] [--cap-bits L]

* **advice**: a declared run's I1 advice from its workload file (`advice_windows.advice_of_workload`), its canonical encoding.
  A served run's comes from its step trace instead (ADVICE-SRC); `compile_advice` reads either.
* **commit** (the prover): the advice compiled (`compile_advice`) and committed, its opening (the advice and its salts) written to
  ``--opening``, the prover's private file, never under the run's out/; then each window from its bundle over the compiler's
  Programs for it, ``WINDOWS/rank<r>/``. ``--forge MEMBER:UNIT`` commits that unit with bit 0 of its first output word flipped
  (`forging`): its rows, commit strings and roots all agree with the flipped word, so only the proof can refuse it.
* **audit** (the verifier): each window from the record, the opened advice and its own compiler (`open_advice`, `window_of`;
  a single-GPU or served window's Programs from its own Builds, written into ``--builds``, default ``OUT/builds``),
  audited by a5 on the verifier's coins with the advice negatives beside it (`more_negatives`), then `advice_before_draw` on the
  sample's receipt and the advice-root-after-the-draw negative (`advice_after_draw`); ``OUT/summary.json`` over the windows.
  ``--forged``: the prover runs with its self-check skipped (``FC_ZK_SELF_CHECK=skip``) so it sends proofs of the false
  witness, and a window passes when its registration passed every check before the sample and Lean refused the outcome
  (`forged_row`). ``--plain``: the sessions are plain Flock, not `--zk` (a5's ``--plain``: a stratified sample over a window of
  one member, no registered reads or shared rows, Lean `verify --coins os --stratified K`, no ZK server), and the summary
  names the interim (``plain``, `PLAIN_INTERIM`). Each window's row carries the charge terms of an accepted audit
  (``charge``, `charge_terms`): ``advice_bits``, ``delta``, ``wrong_units``, ``location_bits`` (λ_δ), ``harm_bound`` (β_δ)
  and ``exfiltration_bound`` (β_δ + λ_δ).
* **compiler**: the binding's compiler for a served run (its name, digest and ℓ, the compiler it wraps, and the φ I2 compiles
  under), the one `verity-vllm advice` registers. The inner digest is a single-GPU window's I2's (`I2Compiler.digest`) over
  the tree's code (`source_identity.code_identity`: ``integrations/vllm`` and ``verity`` by content), so the audit runs on a
  tree whose two are the server's; over what the compile reads of ``serve.json`` (its model, revision and that pin's config,
  target, execution and default seed), so the audit runs on the serve.json it was registered under; over ℓ and the transfer
  gates its model's compiler of record lists (`GATES_OF_RECORD`). It is the name and digest ``verity-vllm advice
  --compiler-name N --compiler-digest D`` registers the
  window's advice under.
* **weights-registration** (the committer): a served run's weights registration (`weights_registration_of_serve`), each
  weight committed under the committer's secret salts once its shards check against the checkpoint's manifest, and the
  rotary table at the engine's served words once they are within the analytic table's bound (``--tables``), the
  document ``verity-vllm advice --weights-of-record`` records in each window's registration; its roots are what
  ``window-circuit --pins`` and ``records_audit.py count --pins`` take. The verifier pins them before the first window's
  coins and never recomputes one from the checkpoint.
* **serve-standin**: a served run's dir laid out from a single-GPU Commit of record (one window, ``w0``, under the record's run
  root; the declared run's advice committed under the served compiler), a stand-in for serve-openai's until its sealed windows.
* **circuit**: a served window's registered compiler as a command (`circuit_of_advice`), what ``records_audit.py audit``'s
  compile check runs: the circuit record the advice compiles to, over its own Build of each Program (`own_builds`; it takes
  no row dir), and the length it charges. ``records_audit.py`` audits a window
  from the records its integration hands Verity alone (``circuit.json``, the registration and the values, the advice's
  opening) and imports no `verity_vllm`; this file's ``audit`` holds a window to the vLLM run's own records.
* **window-circuit**: the same advice through I2 as its window class's one Circuit (`window_circuit_of_advice`, V3; the
  class ``--class``, by default the class of record, `window_circuit.CLASS_OF_RECORD`: 8 slots, 4 at each of buckets 0
  and 16, 1,024 prompt ids and 64 outputs, the certifier run of record's envelope at the fixed slots; every slot's
  Program whatever the window's requests, each placed at the window's first boundary): its record (or ``--description``,
  its description) and its input count at the verifier's pins (``--pin NAME=ROOT``, the roots it fixed before the run,
  each of its parameter alone, a weight's ``weights/<label>@<words>``; ``--pins REGISTRATION``, each weight's hiding root
  as the committer registered it before the first window's coins, `window_weights.weights_registration`'s ``{program,
  values}``, never a root computed from a checkpoint; a registered read counts registered only where the window's
  registration, `verity-vllm advice`'s, records the pinned root for its parameter, and any other is free), which
  ``records_audit.py count`` recounts from the record alone. A5 audits it once the committer writes each
  unit's rows as the record's members (V3c).
* **class-size**: that Circuit's padded size at ``--class``, too large to lower there (`class_size_of_advice`): one class
  Program of the window's registration lowered at the small class ``--probe`` and projected with v19's heads
  (`class_size.project`), its units, ANDs and m a Program and at the class's slots (the window's circuit, whatever its
  requests).
* **window-commit** (the prover, V3c): the served window committed as that Circuit's units (`window_commit_of_advice`):
  slot k is the advice's k-th request, its row (occupancy, prompt length, ids) its ingress prompt, its values its sealed
  record's (`window_rows.Served`) and the class Program's where the engine computed none; the weights, composed from the
  checkpoint ``--checkpoint-dir`` (its shards checked against ``--checkpoints`` where given) and opened against the committed weights
  root, committed under the committer's secret key ``--key`` (`window_weights.weight_values`) as
  **weights-registration** committed them under it before the first window's coins (``--registration``, which the
  verifier pins: ``records_audit.py audit --registered``), so a weight served as registered has its registered root and
  any other's reads are free; a stand-in scan state a per-window value under the run root. ``--out`` is the window's records for ``records_audit.py audit --circuit``: the circuit is the
  class's, a function of the class alone, so the verifier's own ``window-circuit`` record is the circuit it holds it to.

`Capped` adds the public cap ℓ on the advice's length to the compiler's registered digest, and `compile_advice` is total on
every byte string: the requests, or a `ValueError` naming why, the same one on every run.

**Transfer gates** (``--transfer-gate T``, repeatable, the same on both sides): the templates the registered compiler lists as
holding transfer gates (`advice_windows`: the I/O shell's units, proved with selection All, never sampled), which its digest
covers (`I2Compiler`, `advice_windows.Traced`). The window's ``work:K`` law then draws every unit of theirs beside K of the rest
(`serving_rows.work_table_of`), and the stream registers with that work table. A served window's I2 lists them beside the
server's code and ``serve.json`` (`I2Compiler.digest`), and its advice is registered under that compiler (``compiler``).
A served window's list is its model's compiler of record's (`GATES_OF_RECORD`, by ``serve.json``'s ``model``) on both sides,
whatever ``--transfer-gate`` says (`served_gates`: a list other than the record's is refused), so the verifier holds a
registration to that list and no other. Until COMPILER lists the I/O shell's units, the run names a stand-in template, at TP1
``RoPEHead_v1{D=64}`` (``llama32_1b_tg_*.json``). Their negatives are a5's `transfer_gate_negatives` and the registrations under
the compiler listing none and under it less each one gate (`more_negatives`).

**Coins**: each window's `service.Stream` draws on the verifier's own coins, or with ``--challenger local|gate:USER@HOST --main
FILE`` on the auditor's challenger's (`a5_coins`: the run root's one session, committed to each registration before any coin is
read; ``local`` is a development session on this machine, pinned on ``--main``, the verifier's copy of main's first-parent
history; ``gate:USER@HOST`` is the auditor host through its SSH gate with the lane's gate key ``--gate-key``, its challenger on
``--peer``'s 127.0.0.1, so the audit runs on that peer, under the deployment's key record ``--challenger-key``). The gate opens
a root once, ever, and a window's honest and forged commits share the run root, so only one of them draws there.
`advice_before_draw` holds the advice to the receipt of the stream that sampled.
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import json
import multiprocessing
import os
import secrets
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "one_stage"))

from records_audit import PORTS, REFUSED, free_ports, pin_arg, pins_of, registration_roots  # noqa: E402, F401
from verity.core.protocols.sampled_proofs.consumers import exfiltration_bound, location_bits  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import audit as AU, partition as P, registration as R  # noqa: E402
from verity.ml.circuits.codec import canonical_json  # noqa: E402
from verity.ml.windows import interface as IF  # noqa: E402
from verity_vllm.pipeline import advice_windows as AW  # noqa: E402
from verity_vllm.pipeline import deferred_rows as DRW  # noqa: E402
from verity_vllm.pipeline.tp import serving_rows as SRT  # noqa: E402

CAPPED = "verity/prove-path/capped-compiler/v0"
I2 = "verity/prove-path/i2-compile-steps/v0"
#: ℓ, the public cap on the advice's encoding in bits: 1 MiB, far above a TP Commit's few hundred requests (I1 spends tens of
#: bits a row)
CAP_BITS = 1 << 23
#: the checkpoint config I2's φ reads when none is given: unsloth/Llama-3.2-1B's (I2's test fixture)
LLAMA32_1B_CONFIG = HERE.parents[1] / "integrations/vllm/tests/program/predict/fixtures/llama32-1b.config.json"
ROWS1 = "prove_path.py commit: a single-GPU Commit's sealed replay bundle -> rows over the registered compiler's Programs"
#: the sampling parameters a request's body carries to I2 (vLLM's `SamplingParams` names the record holds)
BODY_SAMPLING = ("temperature", "top_p", "top_k", "seed", "n")
#: the templates each served model's compiler of record lists as holding transfer gates, by ``serve.json``'s ``model``
#: (`served_gates`): unsloth/Llama-3.2-1B's is the TP1 stand-in, its RoPE heads, until I2 lists the I/O shell's units
GATES_OF_RECORD: Mapping[str, tuple[str, ...]] = {"unsloth/Llama-3.2-1B": ("RoPEHead_v1{D=64}",)}
#: the schema of the request Programs vllm's registered compiler writes into a served window's row dir (``manifest.json``)
SERVED_PROGRAMS = "verity-vllm/served-window-programs/v1"
#: what a plain audit's summary names as its interim (a5's `PLAIN_SCOPE` is the scope it refuses outside of), with the
#: changes it waits on (``sources.json``)
PLAIN_INTERIM: str = json.loads((HERE / "sources.json").read_text())["plain_interim"]


@dataclass(frozen=True)
class Capped:
    """A registered compiler with ℓ, the public cap on the advice's length in bits. Its digest names the inner compiler's and ℓ,
    so the registration's ``compiler`` fixes ℓ before the sample as it fixes the code; `load` and ``transfer_gates`` are the inner
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
    ``result.json`` names it), refused where no Build dir holds it or its files aren't that Program (`build_of`: its digest
    recomputed from its descriptor, its rows held to the descriptor's): the Build's files are the bodies' cache, I2's digests
    are the Programs. What no digest covers (the rows' names, the parameters' names and types, ``supplied``, the
    correspondence and ``prefix_cache.json``) is the Build's word, so the verifier's I2 reads only Builds it wrote itself
    (``own``, `own_builds`) and the prover's reads its row dir's. ``transfer_gates`` are the templates it lists as holding
    transfer gates (module docstring), the run's stand-in until I2 lists the I/O shell's own. Its digest, a single-GPU
    window's and a served one's alike, names its name, its code (``code``: a single-GPU window's I2 runs this tree's,
    `code_identity`, ``verity`` and ``verity_vllm`` as git lists them, so a node's ignored build outputs don't move it; a
    served window's, the server's registered compiler, `served_identity`), everything ``registered`` holds that I2 reads (φ:
    the model, its revision and config, the target and the execution; the default seed; ℓ; the served model name, where it
    names one) and that list. So a served window's ``serve.json`` is bound before the sample with the code: one that differs
    from the one its advice was registered under names another compiler."""

    registered: Any
    ingress: Mapping[str, Mapping[str, Any]]
    programs_root: str
    name: str = I2
    transfer_gates: tuple[str, ...] = ()
    code: str | None = None
    #: the verifier's: its own Build of each Program it compiles is written into ``programs_root`` (`own_builds`)
    own: bool = field(default=False, compare=False)
    compiled: dict = field(default_factory=dict, compare=False, repr=False)
    #: with ``own``, each compiled Program by digest, with its names (`naming.Named`), until `own_builds` writes it
    builds: dict = field(default_factory=dict, compare=False, repr=False)

    @functools.cached_property
    def digest(self) -> str:
        r = self.registered
        served_model = getattr(r, "served_model", None)
        return hashlib.sha512(canonical_json({"name": self.name, "code": self.code or code_identity(), "phi": r.phi.to_json(),
                                              "default_seed": r.default_seed, "cap_bits": r.cap_bits,
                                              **({"served_model": served_model} if served_model is not None else {}),
                                              "transfer_gates": sorted(self.transfer_gates)})).hexdigest()

    def requests(self, data: bytes) -> list[dict]:
        """I2's requests for the advice ``data``: each request's id, first step, prompt length, ``max_tokens`` (its kept outputs)
        and the digest of the Program I2 compiled for it; I2's refusal (`Unexpressible`) is a ValueError naming it."""
        from verity_vllm.program.predict.build import Unexpressible
        from verity_vllm.program.predict.steps import compile_steps

        if data not in self.compiled:
            try:
                c = compile_steps(self.registered, data, self.ingress, named=self.own)
            except Unexpressible as e:
                raise ValueError(f"I2 refuses the advice: {e}") from None
            self.compiled[data] = [{"request_id": u.id, "arrive_step": u.first_step, "prompt_len": u.LP, "max_tokens": u.T + 1,
                                    "digest": u.digest} for u in c.requests]
            self.builds.update({d: (c.programs[d], named) for d, named in c.named.items()})
        return [dict(q) for q in self.compiled[data]]

    def load(self, rank: int, world: int, requests: Any) -> SimpleNamespace:
        """The window's Programs for I2's ``requests`` (`requests`), as `deferred_rows.of_record` holds the Build's: by request id
        its instances, its dir and its digest, and each request's arrival, its first step in the advice. Each Build dir is held to
        I2's digest (`build_of`) before its instances are read, so the rows the window and the audit read are I2's Program's, not
        the Build's word for it; with ``own``, every Build dir is the verifier's own (`own_builds`)."""
        if (int(rank), int(world)) != (0, 1):
            raise ValueError(f"I2 compiles a single-GPU session: rank {rank} of a world of {world}")
        if self.own:
            own_builds(self, requests)
        by = bodies_by_digest(self.programs_root)
        loaded, digs, held = {}, {}, {}
        for q in requests:
            d = by.get(str(q.get("digest")))
            if d is None:
                raise ValueError(f"request {q['request_id']}: no Build dir under {self.programs_root} holds I2's Program "
                                 f"{str(q.get('digest'))[:16]}")
            if (d, q["digest"]) not in held:
                try:
                    held[d, q["digest"]] = build_of(d, q["digest"])
                except ValueError as e:
                    raise ValueError(f"request {q['request_id']}: {e}") from None
            loaded[q["request_id"]], digs[q["request_id"]] = held[d, q["digest"]], q["digest"]
        return SimpleNamespace(progs={k: v["instances"] for k, v in loaded.items()}, dirs={k: v["dir"] for k, v in loaded.items()},
                               digs=digs, arrivals={q["request_id"]: int(q["arrive_step"]) for q in requests})


def own_builds(i2: I2Compiler, requests: Sequence[Mapping[str, Any]]) -> None:
    """The verifier's own Build of each Program I2 compiled for ``requests`` that no Build dir under ``i2.programs_root`` holds
    yet, written from its compile (`naming.write_build`: the Program with the names I2 gives it, as vLLM's trace of it does) into
    ``programs_root`` under `steps.trace_dir`'s name. So the rows' names, the parameters' names and types, ``supplied``, the
    correspondence and ``prefix_cache.json`` the window reads are this verifier's compile's, and no row dir's or trace's."""
    from verity_vllm.program.predict.naming import write_build
    from verity_vllm.program.predict.steps import trace_dir

    Path(i2.programs_root).mkdir(parents=True, exist_ok=True)
    have = bodies_by_digest(i2.programs_root)
    for q in requests:
        if (dg := str(q["digest"])) in have:
            continue
        if dg not in i2.builds:
            raise ValueError(f"request {q['request_id']}: I2 compiled no Program {dg[:16]} here, so the verifier has no Build of "
                             f"it to write")
        program, named = i2.builds.pop(dg)
        have[dg] = str(Path(i2.programs_root) / trace_dir({**named.shape, "policy": named.shape["sampling"]}))
        write_build(have[dg], program, named)


#: the two packages a predicted Program is built from (`score.code_version`'s), by repository path
CODE_DIRS = ("verity", "integrations/vllm/verity_vllm")


@functools.lru_cache(maxsize=1)
def code_identity() -> str:
    """The code the compiler runs, once per process: every file git lists under `CODE_DIRS` (tracked, or untracked and not
    ignored, so a node's build outputs, ``.lake`` under ``verity/``, don't count; docs, ``*.md``, compute nothing), by path and
    content. Refused where the tree isn't a git work tree: a listing without git's ignore rules is another identity."""
    import subprocess

    import verity_vllm

    root = Path(verity_vllm.__file__).resolve().parents[3]
    p = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", *CODE_DIRS], cwd=root,
                       capture_output=True)
    if p.returncode != 0:
        raise ValueError(f"the compiler's code identity reads the tree's git listing, and {root} has none: "
                         f"{p.stderr.decode(errors='replace').strip()[:200]}")
    h = hashlib.sha256()
    for rel in sorted({f for f in p.stdout.decode().split("\0") if f and not f.endswith(".md") and (root / f).is_file()}):
        h.update(rel.encode() + b"\0" + hashlib.sha256((root / rel).read_bytes()).digest())
    return h.hexdigest()


def bodies_by_digest(programs_root: str) -> dict[str, str]:
    """Each Build dir under ``programs_root`` (``build_request*/``) by the Program digest its ``result.json`` names."""
    out: dict[str, str] = {}
    for d in sorted(Path(programs_root).glob("build_request*/result.json")):
        res = json.loads(d.read_text())
        if res.get("ok") and (dig := (res.get("program") or {}).get("digest")):
            out.setdefault(str(dig), str(d.parent))
    return out


def build_of(d: str, digest: str) -> dict[str, Any]:
    """Build dir ``d``'s Program as `MF.load_program` reads it, held to Program ``digest``, each failure a `ValueError` naming it.

    The digest is recomputed from the Build's body (``descriptor.json.gz``, `codec.program_digest`) and must be ``digest``, and
    each digest the Build records (``result.json``'s, the instances head's) must be the recomputed one. The rows the window and
    the audit read (``instances.json.gz``) must be the descriptor's: row k is the root's node k past its Input batches, by spec,
    form and operand leaves, a read of an Input batch as the parameter leaves it binds. That layout is the program function's
    binding record's (I2's ``Derived[...]{SRC}``: the shapes of its parameters but the last, the weights, which binds every
    later Input batch). A word Build only: I2 compiles word Programs. Row names, the head's parameter names and types and its
    ``supplied``, and the Build's correspondence and ``prefix_cache.json`` stay the Build's: no Program digest covers them, so
    the verifier reads them only from Builds it wrote itself (`own_builds`)."""
    import gzip

    from verity.ml.circuits.codec import decode_program, descriptor_id, program_digest
    from verity.ml.circuits.program import is_input
    from verity_vllm.program.compact import expanded_file_args, file_items_from_refs
    from verity.ml.definitions.registry.catalog import load
    from verity_vllm.query.manifest import format as MF

    p = Path(d) / "descriptor.json.gz"
    if not p.is_file():
        raise ValueError(f"{d} has no descriptor.json.gz: its Program's digest can't be recomputed")
    with gzip.open(p, "rt") as f:
        desc = json.load(f)
    got = program_digest(desc)
    if got != digest:
        raise ValueError(f"{p} is Program {got[:16]}, not I2's {str(digest)[:16]}")
    v = MF.load_program(d)
    inst = v["instances"]
    for what, rec in (("result.json", ((v["result"] or {}).get("program") or {}).get("digest")), ("instances.json.gz", inst.get("digest"))):
        if rec != got:
            raise ValueError(f"{d}'s {what} records Program {str(rec)[:16]}, and its descriptor.json.gz is Program {got[:16]}")
    if inst.get("ir") is not None:
        raise ValueError(f"{d}'s instances.json.gz is a Build in IR {inst['ir']!r}, and I2 compiles word Programs")
    nodes = decode_program(desc, load()[0]).root.body.nodes
    n_in = next((k for k, n in enumerate(nodes) if not is_input(n.fn)), len(nodes))
    where = input_layout(desc, [n.n for n in nodes[:n_in]])
    rows = inst.get("rows") or []
    if len(rows) != len(nodes) - n_in:
        raise ValueError(f"{d}'s instances.json.gz has {len(rows)} rows, and its descriptor.json.gz {len(nodes) - n_in} nodes")

    def param(run: Sequence) -> list:
        sp, i, lo, hi = run
        if sp != "n":
            return [sp, i, lo, hi]
        if i < n_in:
            j, off = where[i]
            return ["p", j, off + lo, off + hi]
        return ["n", i - n_in, lo, hi]

    for k, (r, n) in enumerate(zip(rows, nodes[n_in:])):
        want = {"i": k, "spec": descriptor_id(n.fn), "form": n.form,
                "args": [merged([param(run) for run in a]) for a in expanded_file_args([file_items_from_refs(a.refs) for a in n.args])]}
        try:
            have = {"i": r.get("i"), "spec": r.get("spec"), "form": r.get("form"), "args": [merged(a) for a in r.get("args") or ()]}
        except (AttributeError, TypeError, ValueError):
            have = {}
        if have != want:
            bad = ", ".join(x for x in want if have.get(x) != want[x])
            raise ValueError(f"{d}'s instances.json.gz row {k} is not its descriptor.json.gz's node {k}: they differ in {bad}")
    return v


def input_layout(desc: Mapping[str, Any], sizes: Sequence[int]) -> list[tuple[int, int]]:
    """Each root Input batch of leaf count ``sizes[m]`` as (parameter, leaf offset): in order, the parameters the program
    function's ``SRC`` record shapes, then the last parameter, the weights, for every later batch. Refused where the record
    names no shapes or the batches don't fill them."""
    import math

    pf = str(desc.get("program_fn") or "")
    _, sep, rec = pf.partition("{SRC=")
    try:
        shapes = json.loads(rec[:-1])["dict"]["inputs"] if sep and rec.endswith("}") else None
        need = [math.prod(int(x) for x in s) for s in shapes]
    except (ValueError, KeyError, TypeError):
        need = None
    if need is None:
        raise ValueError(f"the program function {pf[:80]} records no SRC input shapes: not one of I2's Programs")
    out, j, off = [], 0, 0
    for m, size in enumerate(sizes):
        while j < len(need) and need[j] == 0:
            j += 1
        out.append((j, off))
        off += int(size)
        if j < len(need) and off >= need[j]:
            if off > need[j]:
                raise ValueError(f"Input batch {m} runs past parameter {j}'s {need[j]} leaves")
            j, off = j + 1, 0
    if any(need[j:]):
        raise ValueError(f"the root's {len(sizes)} Input batches don't fill the {len(need)} parameters SRC shapes")
    return out


def merged(runs: Sequence[Sequence]) -> list[list]:
    """Operand runs ``[space, index, lo, hi]`` with each run that continues the one before folded into it: one leaf sequence,
    one spelling, however the Input batches split a parameter."""
    out: list[list] = []
    for sp, i, lo, hi in runs:
        if out and out[-1][0] == sp and out[-1][1] == i and out[-1][3] == lo:
            out[-1][3] = hi
        else:
            out.append([sp, i, lo, hi])
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
                 transfer_gates: Sequence[str] = (), programs_root: str | None = None, own: bool = False) -> I2Compiler:
    """I2 over a single-GPU Commit: φ from the row's declared inputs (`score.phi_of`: its row id, the workload file the record
    served, refused unless its digest is the record's, and the checkpoint's ``config.json``), the bodies from the record, listing
    ``transfer_gates``; its Programs' Builds under ``programs_root`` (default the row dir; with ``own``, the verifier's own)."""
    from verity_vllm.program.predict.score import phi_of
    from verity_vllm.program.predict.steps import Registered

    r = record1(str(bundle), str(row_dir))
    doc = json.loads(Path(workload).read_text())
    got = hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if got != r.row.get("workload_sha256"):
        raise ValueError(f"{workload}: workload sha256 {got[:16]} is not the record's {str(r.row.get('workload_sha256'))[:16]}")
    phi = phi_of({"row": r.row["label"], "workload": doc}, config=str(hf_config or LLAMA32_1B_CONFIG))
    return I2Compiler(Registered.of(phi, int(doc.get("seed") or 0), cap_bits), ingress_of(r.ctx), str(programs_root or row_dir),
                      transfer_gates=tuple(dict.fromkeys(map(str, transfer_gates))), own=own)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def serve_window(run: str, window: int) -> SimpleNamespace:
    """A serve-openai run's closed Commit window ``window``: ``serve.json`` (the engine), its line of ``windows.jsonl`` (its Commit:
    run id and run root), the requests of ``requests.jsonl`` with rows in it, and `verity-vllm advice`'s ``advice/w<K>/``:
    ``registration.json`` (public: the binding) and the paths of ``private/advice.bin`` and ``private/opening.json``."""
    d, k = Path(run), int(window)
    commits = [w for w in _jsonl(d / "windows.jsonl") if int(w["window"]) == k]
    if len(commits) != 1:
        raise ValueError(f"{d}/windows.jsonl: {len(commits)} lines for window {k}, not one")
    adv = d / "advice" / f"w{k}"
    return SimpleNamespace(dir=d, window=k, serve=json.loads((d / "serve.json").read_text()), commit=commits[0],
                           registration=json.loads((adv / "registration.json").read_text()), requests=serve_requests(d, k),
                           advice=adv / "private" / "advice.bin", opening=adv / "private" / "opening.json")


def serve_requests(run: str | Path, window: int) -> list[dict]:
    """The requests of a serve-openai run's ``requests.jsonl`` with rows in Commit window ``window``."""
    return [q for q in _jsonl(Path(run) / "requests.jsonl") if int(window) in (q.get("windows") or [])]


def serve_id(q: Mapping[str, Any]) -> str:
    """A served request's I1 id: its body's ``user``, else its HTTP id, as `verity-vllm advice` names it (`advice.io_record`)."""
    return str(q.get("user") or q.get("http_id") or q.get("engine_id"))


def serve_ingress(requests: list[Mapping[str, Any]]) -> dict[str, dict]:
    """Each served request's body by its I1 id (`serve_id`), as vllm's registered compiler gives it to I2 (its
    ``advice_compile.served_inputs``): the sampling fields the request brought (its ingress record's ``brought``) with its
    prompt token ids."""
    out: dict[str, dict] = {}
    for q in requests:
        i = serve_id(q)
        if i in out:
            raise ValueError(f"requests.jsonl: request {i} is recorded twice")
        out[i] = {**((q.get("ingress") or {}).get("brought") or {}), "prompt": list(q["prompt_ids"])}
    return out


def phi_of_serve(serve: Mapping[str, Any], hf_config: str | None = None, target: str | None = None) -> Any:
    """φ of a served engine as vllm's registered compiler registers it (the formula its Build compiles the request Programs under):
    the pinned model (``serve.json``'s ``model`` and ``revision``, and the checkpoint's ``config.json``: ``hf_config``, else the
    pin's config of record, as `advice_compile` reads it (`dense_generic.pinned_hf_config`), else for a model no manifest pins
    unsloth/Llama-3.2-1B's), the serving device's target profile (``serve.json``'s ``target``, the int8 scope its layers ran
    included, else the file ``target``) and the engine's execution label (``execution``, else ``engine.execution``); no workload."""
    from verity_vllm.engine.profiles.dense_generic import pinned_hf_config
    from verity_vllm.program.predict.phi import Phi

    prof = json.loads(Path(target).read_text()) if target else serve.get("target")
    if not isinstance(prof, Mapping):
        raise ValueError("serve.json records no target profile (the serving device's TargetProfile.to_json()), so φ can't be read: "
                         "--target F gives it")
    execution = serve.get("execution") or (serve.get("engine") or {}).get("execution")
    if not isinstance(execution, str) or not execution:
        raise ValueError("serve.json records no execution label (execution, engine.execution), so φ can't be read")
    hf = json.loads(Path(hf_config or pinned_hf_config(serve["model"], serve["revision"]) or LLAMA32_1B_CONFIG).read_text())
    return Phi(model={"repo": serve["model"], "revision": serve["revision"], "hf_config": hf}, target=dict(prof), execution=execution)


def registered_seed(serve: Mapping[str, Any]) -> int:
    """The default seed a served window's registration fixes, `engine.served_seed.DEFAULT_SEED` (as vllm's registered compiler takes
    it); a ValueError when ``serve.json`` says the server filled in another (or says none), since a request that sent no
    seed would be compiled under a seed its sample didn't use."""
    from verity_vllm.engine.served_seed import DEFAULT_SEED

    seed = serve.get("default_seed")
    if type(seed) is not int or seed != DEFAULT_SEED:
        raise ValueError(f"serve.json: the run's server filled in default seed {seed!r}, and the registered one is "
                         f"engine.served_seed.DEFAULT_SEED ({DEFAULT_SEED}): serve it under that rule")
    return DEFAULT_SEED


def registered_of_serve(serve: Mapping[str, Any], hf_config: str | None, target: str | None, cap_bits: int) -> Any:
    from verity_vllm.program.predict.steps import Registered

    return Registered.of(phi_of_serve(serve, hf_config, target), registered_seed(serve), cap_bits)


def served_gates(serve: Mapping[str, Any], given: Sequence[str] = ()) -> tuple[str, ...]:
    """The transfer gates a served window's compiler lists: its model's compiler of record's (`GATES_OF_RECORD`, none for a
    model it has no line for). ``given`` (``--transfer-gate``) may only restate them: any other list is refused, so neither side
    builds a served compiler listing another."""
    model, pinned = str(serve.get("model")), GATES_OF_RECORD.get(str(serve.get("model")), ())
    if given and sorted(set(map(str, given))) != sorted(pinned):
        raise ValueError(f"--transfer-gate {sorted(set(map(str, given)))}: {model}'s compiler of record lists {list(pinned)} "
                         "(prove_path.GATES_OF_RECORD), and a served window's compiler lists those and no others")
    return pinned


def i2_of_serve(run: str, window: int, programs_root: str, hf_config: str | None, target: str | None,
                cap_bits: int = CAP_BITS, transfer_gates: Sequence[str] = (), own: bool = False) -> I2Compiler:
    """I2 over a served window: φ of the served engine (`phi_of_serve`), the public default seed (`registered_seed`), the bodies
    the window's requests brought (`serve_ingress`), each Program's body from the Build's files under ``programs_root`` (the
    prover's row dir, or with ``own`` the verifier's own Builds), named as the server registers it and running its code
    (`served_identity`), listing its model's transfer gates of record (`served_gates`, which ``transfer_gates`` may only
    restate). Its digest is a single-GPU window's I2's over that code (`I2Compiler.digest`), so it covers every field of
    ``serve.json`` the compile reads."""
    serve = json.loads((Path(run) / "serve.json").read_text())
    name, code = served_identity()
    return I2Compiler(registered_of_serve(serve, hf_config, target, cap_bits), serve_ingress(serve_requests(run, window)),
                      str(programs_root), name=name, transfer_gates=served_gates(serve, transfer_gates), code=code, own=own)

@functools.lru_cache(maxsize=1)
def served_identity() -> tuple[str, str]:
    """The code a served window's compiler runs, as `verity-vllm advice` names it (``pipeline.advice``'s
    ``registered_compiler``): compile_steps by its module path, its digest over the tree's code identity
    (`source_identity.code_identity`: ``integrations/vllm`` and ``verity``), so the audit's tree holds the server's. The
    window's I2 is that name, with this digest as its code (`i2_of_serve`)."""
    try:
        from verity_vllm.pipeline import advice as ADV
    except ImportError:
        raise ValueError("a served window's registered compiler is `verity-vllm advice`'s (verity_vllm.pipeline.advice), "
                         "which this tree lacks") from None
    inner = getattr(ADV.registered_compiler(), "inner", None)
    if inner is None:
        raise ValueError("verity_vllm.pipeline.advice registers no capped compiler (its registered_compiler wraps none)")
    return str(inner.name), str(inner.digest)


def compiler_of(tp_commit: str, build_root: str, cap_bits: int = CAP_BITS, transfer_gates: Sequence[str] = ()) -> Capped:
    return Capped(AW.Traced.for_commit(tp_commit, build_root, transfer_gates), cap_bits)


def compiler_of_args(a: argparse.Namespace) -> Capped:
    """The run's compiler. The verifier's (``a.builds``, `own_builds`: ``audit``'s and ``circuit``'s) reads I2's Programs from
    Builds it writes itself into ``a.builds`` from its compile; the prover's from its row dir."""
    gates, own = getattr(a, "transfer_gate", None) or (), getattr(a, "builds", None)
    if served(a):
        return Capped(i2_of_serve(a.serve_run, a.window, own or a.row_dir, a.hf_config, a.target, a.cap_bits, gates, bool(own)), a.cap_bits)
    if single(a):
        return Capped(i2_of_record(a.bundle, a.row_dir, a.workload, a.hf_config, a.cap_bits, gates, own, bool(own)), a.cap_bits)
    return compiler_of(a.tp_commit, a.build_root, a.cap_bits, gates)


def single(a: argparse.Namespace) -> bool:
    """A single-GPU Commit's run (``--bundle``, or a served window's ``--serve-run``), not a TP Commit's."""
    return bool(getattr(a, "bundle", None) or served(a))


def served(a: argparse.Namespace) -> bool:
    """A serve-openai run's window (``--serve-run``)."""
    return bool(getattr(a, "serve_run", None))


def serve_check(s: SimpleNamespace, bound: Mapping[str, Any], run_root: str, commitment: str) -> dict[str, Any]:
    """A served window's published records held to the verifier's own, each check by name: its registration
    (``advice/w<K>/registration.json``) names this window (``window``), a compiler whose digest carries the cap ℓ (``cap``: a
    `Capped` one) and is the verifier's own (``compiler``: I2 running the server's code under the run's ``serve.json``,
    `i2_of_serve`, and ℓ, so a ``serve.json`` other than the one the advice was registered under is refused here), and the advice root and length the
    opening opens (``advice``); the root serve-openai published for the window (``windows.jsonl``) is a SHA-512 root of a
    ``rows-sha512`` store (``sha512``), the row dir's run root ``run_root`` (``run_root``), and the Commit the registration names
    (``commit``: its run id and run root)."""
    reg, pub = s.registration, s.commit
    b = reg.get("binding") if isinstance(reg.get("binding"), Mapping) else {}
    c, root = b.get("compiler") or {}, str(pub.get("run_root") or "")
    ok = {"window": reg.get("window") == s.window, "cap": c.get("name") == CAPPED, "compiler": c == bound["compiler"],
          "advice": b.get("advice") == bound["advice"], "sha512": len(root) == 2 * DRW.SERVED_ROOT and commitment == SRT.ROWS_SCHEME,
          "run_root": root == run_root, "commit": reg.get("commit") == {"run_id": pub.get("run_id"), "run_root": pub.get("run_root")}}
    return {"ok": all(ok.values()), "codes": [k for k, v in ok.items() if not v],
            "detail": {"window": s.window, "published_root": root, "commitment": commitment, "compiler": c, "advice": b.get("advice"),
                       "run_id": pub.get("run_id")}}


def serve_check_of(a: argparse.Namespace, bound: Mapping[str, Any]) -> dict[str, Any]:
    r = record1(str(a.bundle), str(a.row_dir))
    return serve_check(serve_window(str(a.serve_run), int(a.window)), bound, bytes(r.run.run_root).hex(), r.store.commit_scheme)


def serve_negatives(s: SimpleNamespace, compiler: Capped, bound: Mapping[str, Any], run_root: str,
                    commitment: str) -> dict[str, dict]:
    """What `serve_check` refuses of a served window's published records: a registration whose compiler carries no cap (the inner
    compiler's name and digest, as `verity-vllm advice` registers by default), one naming another window, the window's root
    published under vllm-v1 (32 bytes), and another Commit's root."""
    reg, pub = s.registration, s.commit
    uncapped = {**reg, "binding": {**reg["binding"], "compiler": {"name": str(compiler.inner.name), "digest": compiler.inner.digest}}}
    cases = {"registration-without-the-cap (serve)": SimpleNamespace(**{**vars(s), "registration": uncapped}),
             "registration-of-another-window (serve)": SimpleNamespace(**{**vars(s), "registration": {**reg, "window": s.window + 1}}),
             "published-root-not-sha512 (serve)": SimpleNamespace(**{**vars(s), "commit": {**pub, "run_root": run_root[:64]}}),
             "published-root-of-another-commit (serve)": SimpleNamespace(**{**vars(s), "commit": {**pub, "run_root": "00" * 64}})}
    out = {}
    for name, x in cases.items():
        v = serve_check(x, bound, run_root, commitment)
        out[name] = {"refused": not v["ok"], "codes": v["codes"]}
    return out


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


def served_advice(a: argparse.Namespace) -> tuple[list[dict], dict, dict]:
    """The prover's side of a served window: the server committed the advice, so the requests and binding are the ones its
    opening opens (`compiled`), refused unless the opening holds the window's ``advice.bin`` and the published records pass
    `serve_check`: the prover commits no window the verifier would refuse before the sample."""
    _compiler, requests, bound, opening = compiled(a)
    s = serve_window(str(a.serve_run), int(a.window))
    if bytes.fromhex(opening["advice"]) != Path(s.advice).read_bytes():
        raise ValueError(f"{a.opening} opens other advice than {s.advice}")
    if not (v := serve_check_of(a, bound))["ok"]:
        raise ValueError(f"serve-openai window {a.window}'s published records are refused ({', '.join(v['codes'])}): "
                         f"{json.dumps(v['detail'], sort_keys=True)}")
    return requests, bound, opening


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


def with_serve(v: dict[str, Any], sv: dict[str, Any]) -> dict[str, Any]:
    """A served window's source verdict: the registration's (`source1`) and the published records' (`serve_check`)."""
    return {"ok": v["ok"] and sv["ok"], "codes": [*v["codes"], *(c for c in sv["codes"] if c not in v["codes"])],
            "detail": {**v["detail"], "serve": sv["detail"]}}


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
    by the manifest's lookup): each request's two digests and whether they are equal. A served window's Programs of record are
    the ones vllm's registered compiler wrote into its row dir by I1 id (``manifest.json``, `SERVED_PROGRAMS`); a stand-in's row
    dir has none, and its record names requests by its own labels, so there the two are compared as multisets of digests."""
    if not single(a):
        return None
    man = Path(a.row_dir) / "manifest.json"
    doc = json.loads(man.read_text()) if served(a) and man.is_file() else {}
    if doc.get("schema") == SERVED_PROGRAMS:
        of = {str(k): str(v["digest"]) for k, v in doc["requests"].items()}
    elif served(a):
        mine, theirs = sorted(q["digest"] for q in requests), sorted(map(str, record1(str(a.bundle), str(a.row_dir)).digests.values()))
        return {"units": {q["request_id"]: {"i2": q["digest"]} for q in requests}, "of_record": theirs, "equal": mine == theirs}
    else:
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
    """The advice committed once, so every window's registration binds one root, then each window. A served window's advice the
    server committed: its windows bind the published root (`served_advice`), and the opening stays the server's."""
    _world, ranks = ranks_of(a)
    t0 = time.time()
    if served(a):
        requests, bound, _opening = served_advice(a)
    else:
        requests, bound, opening = commit_advice(compiler_of_args(a), Path(a.advice).read_bytes())
        Path(a.opening).write_text(json.dumps(opening) + "\n")
    compile_s = round(time.time() - t0, 1)
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
    listing no transfer gates and one under it less each one gate (where it lists some: a window sampling them, as another
    compiler's), and another rank's."""
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
    for g in compiler.transfer_gates:
        less = Capped(replace(compiler.inner, transfer_gates=tuple(t for t in compiler.transfer_gates if t != g)), compiler.cap_bits)
        cases[f"registration-under-the-compiler-less-the-gate-{g} (source)"] = {**w, "compiler": {"name": CAPPED,
                                                                                                  "digest": less.digest}}
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
    ``compiler`` and ``advice`` root) must be the one the sample's receipt names, the receipt the stream wrote when it received the
    registration and before its select read any coin (a5's ``verifier-coins-before-the-registration``: a select on a
    registration the stream never received reads none; on the challenger's coins, ``coins="challenger"``, the coins are also
    committed to that digest before any is read); and that registration must bind the compiler and the root the verifier
    opened. -> {ok, codes, detail}: ``after-the-draw`` where the sample followed another registration, or another stream's,
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
    in a registration binding that root, with its opening. Every check before the sample passes on it (``source`` against the new
    opening: the root is a true commitment of the advice), and `advice_before_draw` refuses it (``after-the-draw``): the sample
    followed the registration received first."""
    data = bytes.fromhex(opening["advice"])
    late = AW.opening(data, AW.commit_advice(compiler, data))
    _reqs, bound = open_advice(compiler, late)
    late_rec = {**rec, "window": {**rec["window"], "advice": bound["advice"]}}
    src = source(late_rec["window"], mine, {**held, **bound})
    v = advice_before_draw(receipt, late_rec, bound, coins)
    return {"refused": not v["ok"], "codes": v["codes"], "before_draw_checks_pass": src["ok"]}


def forged_row(s: dict, forged: dict | None) -> dict:
    """A forged window's verdict: its registration passed every check before the sample, the forged unit was sampled, and the outcome
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
    """Each drawn member's session timing per sampled proof block (the units its one statement proves, ``draw_counts``; Flock's own
    ``units`` count its virtual units): proving (Flock's ``prove_total_s`` and the session's wall), M0's and Lean's verify,
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


def template_bits(mine: Any, sets: Mapping[str, tuple]) -> dict[str, int]:
    """The bits one unit of each of the window's templates commits (`partition._template_bits`), by template."""
    from verity_vllm.pipeline import serving_rows as SRW

    return {m.template: P._template_bits(SRW.template(m.template).fn if m.schemas else sets[m.template][1]) for m in mine.members}


def charge_terms(record: Mapping[str, Any], bits: Mapping[str, int], advice_bits: int, delta: float) -> dict[str, Any] | None:
    """The terms of the leak bound an accepted audit gives (README, "What an accepted audit means"): c, the advice's bits
    (``advice_bits``); δ (``delta``); ``wrong_units``, the profile's bounds at δ; λ_δ, which units are faulty
    (``location_bits``, `consumers.location_bits`); β_δ, the faulty proof blocks' committed bits (``harm_bound``: the profile's
    `harm_bound`, each template's units weighing its committed ``bits``, the most of them where the law's strata aren't the
    templates); and ``exfiltration_bound`` = β_δ + λ_δ (`consumers.exfiltration_bound`). ``record`` is the audit record
    (a5's ``audit.json``); None where it isn't accepted, which bounds nothing."""
    if not record.get("accepted"):
        return None
    p = AU.profile(record["registration"], record["draw"], delta)
    law = p.strata()
    weights: float | dict[str, float] = max(bits.values())
    if law is not None and {s.name for s in law.strata} <= set(bits):
        weights = {s.name: float(bits[s.name]) for s in law.strata}
    return {"advice_bits": advice_bits, "delta": delta, "wrong_units": record.get("wrong_units"), "template_bits": dict(bits),
            "location_bits": location_bits(p), "harm_bound": p.harm_bound(weights), "exfiltration_bound": exfiltration_bound(p, weights)}


def audit_rank(job: tuple[argparse.Namespace, int, int]) -> dict:
    """Rank ``r``'s audit (`rank_audit`). On the challenger's coins, a rank that ends without its sample says so, so the run root's
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
    wdir, out, src = Path(a.windows) / f"rank{r}", Path(a.out) / f"rank{r}", source_of(a)
    more = more_negatives(a, r, world, compiler, mine, held, wdir, opening)
    check, sv = (lambda w: src(w, mine, held)), None
    if served(a):
        rr = record1(str(a.bundle), str(a.row_dir))
        sv = serve_check_of(a, bound)
        more = {**more, **serve_negatives(serve_window(str(a.serve_run), int(a.window)), compiler, bound, bytes(rr.run.run_root).hex(),
                                          rr.store.commit_scheme)}
        check = lambda w: with_serve(src(w, mine, held), sv)
    sets = A5.template_sets(Path(a.template_sets))
    s, ok = A5.audit(a, sets, out, mine, wdir, check, a.port + A5.RANK_PORTS * r,
                     {"rank": r, "tp": held.get("tp"), "binding": bound, "requests": requests, "verifier_window_s": window_s,
                      **({"serve": sv} if sv else {})}, more, rank=r)
    record = json.loads((out / "audit.json").read_text()) if (out / "audit.json").is_file() else {}
    row = {"ok": ok, "seconds": round(time.time() - t0, 1), "verifier_window_s": window_s, "per_unit": per_unit(s),
           "charge": charge_terms(record, template_bits(mine, sets), bound["advice"]["bits"], a.delta),
           **({"serve": sv} if sv else {}), **({"plain": s["plain"]} if "plain" in s else {}),
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
                     **({"serve": {"run": str(a.serve_run), "window": int(a.window), **serve_check_of(a, bound)}} if served(a) else {}),
                     "requests": requests, "rule": a.rule, "delta": a.delta, "forged": a.forged, "coins": coins_of(a), "ranks": {},
                     **({"plain": {"scope": A5.PLAIN_SCOPE, "interim": PLAIN_INTERIM}} if a.plain else {})}
    if a.challenger:
        manager, a.coins, finisher = AC.start(a.challenger, root, ranks, out / "challenger", a.main, ssh_key=a.gate_key,
                                              peer=a.peer, key_record=a.challenger_key)
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
             | {"charge": {x: (s.get("charge") or {}).get(x) for x in ("location_bits", "harm_bound", "exfiltration_bound")}
                | {"wrong_units_bound": ((s.get("charge") or {}).get("wrong_units") or {}).get("bound")}}
             for r, s in summary["ranks"].items()}
    print("PROVE-PATH " + json.dumps({"run_root" if single(a) else "tp_run_root": root, "forged": a.forged, "ok": summary["ok"],
                                      "proof": "plain" if a.plain else "zk", **({"plain": summary["plain"]} if a.plain else {}),
                                      "advice_bits": bound["advice"]["bits"], "delta": a.delta,
                                      "coins": summary["coins"]["source"], "transfer_gates": summary["transfer_gates"],
                                      **({"i2_equal": summary["i2"]["equal"]} if single(a) else {}),
                                      **({"serve": {k: summary["serve"][k] for k in ("window", "ok", "codes")}} if served(a) else {}),
                                      "ranks": shown},
                                     sort_keys=True, default=str))
    return 0 if summary["ok"] else 1


def compiler_doc(serve_run: str, hf_config: str | None, target: str | None, cap_bits: int,
                 transfer_gates: Sequence[str] = ()) -> dict[str, Any]:
    """A served run's registered compiler, `Capped` over I2 running the server's code (`served_identity`) under the run's
    ``serve.json`` (`registered_of_serve`), listing its model's transfer gates of record (`served_gates`, which
    ``transfer_gates`` may only restate): the name and digest its windows' bindings must carry, ℓ and the compiler it wraps
    (``inner``, what `verity-vllm advice --compiler-name --compiler-digest` takes, its digest `I2Compiler.digest`'s), the
    server's code (``served``), and the φ and default seed I2 compiles each window's Programs under."""
    name, digest = served_identity()
    serve = json.loads((Path(serve_run) / "serve.json").read_text())
    reg = registered_of_serve(serve, hf_config, target, cap_bits)
    inner = I2Compiler(reg, {}, "", name=name, transfer_gates=served_gates(serve, transfer_gates), code=digest)
    c = Capped(inner, cap_bits)
    return {"name": c.name, "digest": c.digest, "cap_bits": c.cap_bits, "inner": {"name": name, "digest": inner.digest},
            "served": {"name": name, "digest": digest}, "transfer_gates": list(inner.transfer_gates),
            "phi": reg.phi.to_json(), "default_seed": reg.default_seed}


def committer_key(path: Path) -> bytes:
    """The committer's 32-byte salt key at ``path``, which never leaves the committer: drawn fresh (mode 0600) the first
    time, read thereafter. Refused (ValueError) unless 32 bytes."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as f:
            f.write(secrets.token_bytes(32))
    key = path.read_bytes()
    if len(key) != 32:
        raise ValueError(f"{path}: a committer's salt key is 32 bytes, not {len(key)}")
    return key


def weights_registration_of_serve(a: argparse.Namespace) -> dict[str, Any]:
    """A served run's weights registration, the committer's (`window_weights.weights_registration` under the φ its window
    circuit lowers under, `registered_of_serve`): each weight committed under the committer's salts, drawn under ``--key``
    (`committer_key`, never sent), once the committer has checked ``--checkpoint-dir``'s shards against ``--checkpoints``
    (default the tree's ``manifests/checkpoints.json``), and the rotary table at the words the engine serves
    (``--tables``' ``cos_sin_cache.npy``, serve-openai's, default under ``--serve-run``'s ``private/``) once they are
    within the analytic table's bound; with no such file it stays unrooted. Written to ``--out``: what the server's
    ``verity-vllm advice --weights-of-record`` records in each window's registration, and the roots the verifier pins
    before the first window's coins (``window-circuit --pins``, ``records_audit.py count --pins``); the verifier never
    recomputes one. Refused (ValueError) for a checkpoint not of record or one the manifest pins as another model than
    ``serve.json``'s, a served rotary table outside its bound, and as `advice.registered_weights` refuses it. -> how many
    roots it registers, and the weight parameters it leaves free (``unrooted``)."""
    import numpy as np

    from verity_vllm.pipeline.advice import registered_weights
    from verity_vllm.program.predict.window_weights import weights_registration

    serve = json.loads((Path(a.serve_run) / "serve.json").read_text())
    cos_sin = Path(a.tables or Path(a.serve_run) / "private") / "cos_sin_cache.npy"
    doc = weights_registration(registered_of_serve(serve, a.hf_config, a.target, CAP_BITS), committer_key(Path(a.key)),
                               a.checkpoint_dir, checkpoints_manifest=a.checkpoints,
                               tables={"cos_sin": np.load(cos_sin)} if cos_sin.is_file() else None)
    roots = registered_weights(doc, serve)
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
    return {"roots": len(roots), "unrooted": sorted(doc["unrooted"]), "out": a.out}


def circuit_of_advice(a: argparse.Namespace) -> dict[str, int]:
    """A served window's registered compiler as the command `records_audit.compile_check` runs: the advice ``--advice``
    compiled (`compile_advice`, every refusal a `ValueError`), the window over the compiler's Programs for it
    (`serving_rows.population`, as `window1` makes the verifier's), each Program's Build this command's own, from its compile
    (`own_builds`, into a directory of its own beside ``--out``: it reads no row dir and needs no vLLM), and its circuit record
    (`serving_rows.circuit_record`) written to ``--out``, under no run (its `interface.RUN_FIELDS` empty: the compiler knows
    none). -> the advice's length in bits, the one the registration charges (`advice_windows.of_advice`)."""
    import tempfile

    from verity_vllm.pipeline import serving_rows as SRW

    data = Path(a.advice).read_bytes()
    with tempfile.TemporaryDirectory(prefix="builds-", dir=Path(a.out).resolve().parent) as builds:
        a.builds = builds
        compiler = compiler_of_args(a)
        p = compiler.load(0, 1, compile_advice(compiler, data))
        window, _rows = SRW.population(p.progs, json.loads(Path(a.partition).read_text()), "", p.dirs, arrivals=p.arrivals,
                                       transfer_gates=compiler.transfer_gates)
    Path(a.out).write_bytes(IF.circuit_bytes(SRW.circuit_record(window)))
    c = AW.codec_of()
    return {"bits": int(c.total_bits(c.decode(data)))}


def window_class_of(arg: str | None) -> Any:
    """``--class`` or ``--probe``'s SLOTS,PROMPT,OUTPUTS[,LANES,EVERY] as a `window_circuit.WindowClass`, None the class of
    record."""
    from verity_vllm.program.predict.window_circuit import CLASS_OF_RECORD, WindowClass

    return CLASS_OF_RECORD if arg is None else WindowClass(*(int(x) for x in arg.split(",")))


def window_circuit_of_advice(a: argparse.Namespace) -> dict[str, Any]:
    """A served window's advice through I2 as its window class's one Circuit (V3: `window_circuit.window_circuit` at
    ``--class``, a class Program a slot flattened into units, every input row wired or registered, the weights' or the
    network certifier's scan state rows), written to
    ``--out`` as its circuit record from its description alone (`interface.descriptor_record`, members stated at ``--cap``
    units), or with ``--description`` as the description `interface.input_count` reads, no program built. -> the advice's
    length in bits and the record's input count at the verifier's pins (``--pin NAME=ROOT``, ``--pins FILE``: `pins_of`),
    each held to the root the window's registration (``advice/w<K>/registration.json``) records for its parameter
    (`registration_roots`)."""
    from verity_vllm.program.predict.build import Unexpressible
    from verity_vllm.program.predict.steps import compile_steps
    from verity_vllm.program.predict.window_circuit import window_circuit

    compiler, data, pins = compiler_of_args(a), Path(a.advice).read_bytes(), pins_of(a.pin, a.pins)
    compile_advice(compiler, data)
    registered, cls = compiler.inner.registered, window_class_of(a.window_class)
    t0 = time.time()
    try:
        circuit = window_circuit(registered, compile_steps(registered, data, compiler.inner.ingress), cls)
    except Unexpressible as e:
        raise ValueError(f"I2's window circuit refuses the advice: {e}") from None
    lowered = time.time() - t0
    doc = circuit.description() if a.description else IF.descriptor_record(circuit, a.cap)
    Path(a.out).write_bytes(IF.circuit_bytes(doc))
    c = AW.codec_of()
    roots = registration_roots(json.loads((Path(a.serve_run) / "advice" / f"w{a.window}" / "registration.json").read_text()))
    return {"bits": int(c.total_bits(c.decode(data))), "count": IF.input_count(doc, pins=pins, roots=roots),
            "seconds": {"circuit": round(lowered, 1), "record": round(time.time() - t0 - lowered, 1)}}


def class_size_of_advice(a: argparse.Namespace) -> dict[str, Any]:
    """A served window's padded size at its class ``--class`` (`class_size`): one class Program of the window's registration
    lowered at the small class ``--probe`` (`window_circuit.class_circuit`, exact), projected to ``--class`` with v19's heads
    (`class_size.project`) and priced (`class_size.size`) a Program and at the class's slots, the window's circuit whatever
    its requests. -> the probe's census, the requests its advice compiles to and the two sizes."""
    from dataclasses import asdict

    import class_size as CZ
    from verity_vllm.program.predict.window_circuit import class_circuit

    compiler = compiler_of_args(a)
    requests = len(compile_advice(compiler, Path(a.advice).read_bytes()))
    probe, cls = window_class_of(a.probe), window_class_of(a.window_class)
    t0 = time.time()
    seen = CZ.census(class_circuit(compiler.inner.registered, probe, 1, provable=False))
    lowered = time.time() - t0
    try:
        one = CZ.project(seen, probe, cls)
        sizes = {"program": CZ.size(one), "window": CZ.size(one, cls.slots)}
    except ValueError as e:
        raise ValueError(f"{e} (the probe's census: {json.dumps(seen, sort_keys=True)})") from None
    return {"class": asdict(cls), "positions": cls.positions, "requests": requests,
            "probe": {"class": asdict(probe), "census": seen, "seconds": round(lowered, 1)}, **sizes}


def window_commit_of_advice(a: argparse.Namespace) -> dict[str, Any]:
    """A served window committed as its window class's one Circuit (V3c, module docstring): the class lowered
    (`window_circuit.lower`, refused where the class's slots don't take the window), each slot's values (`window_rows.Slot`,
    the k-th request's served values its sealed record's), every unit's rows (`window_rows.Rows`) recorded into a `Window`
    under the record's run root, the weights it serves committed under the committer's key ``--key`` as the registration
    ``--registration`` (`weights-registration`, before the first window's coins) committed them, and the scan state the
    slots' rows and ``max_tokens`` give (`window_rows.standin_state`) the window's own value, a stand-in for the network
    certifier's, whose reads are free. Refused for a registration of another model. -> the window's run root, its circuit record's SHA-512, its
    population and members, the registered weights whose root the window's is (``held``) and those it isn't, whose reads
    are free (``free``), and the seconds each part took."""
    import numpy as np
    from verity_vllm.check import weights_of_record as WR
    from verity_vllm.check.oracle_compare import OpenedReader
    from verity_vllm.check.replay.index import ProgramIndex
    from verity_vllm.check.replay.opening import CommittedStore
    from verity_vllm.pipeline import cpu_replay as CPU
    from verity_vllm.program.predict import scan_state as SS
    from verity_vllm.program.predict import window_circuit as WC
    from verity_vllm.program.predict import window_rows as WRS
    from verity_vllm.program.predict import window_weights as WW
    from verity_vllm.program.predict.build import Unexpressible
    from verity_vllm.program.predict.steps import compile_steps
    from verity.core.protocols.sampled_proofs.one_stage import draw as DR
    from verity.core.protocols.sampled_proofs.one_stage import registered as CR

    seconds: dict[str, float] = {}
    t = time.time()

    def lap(what: str) -> None:
        nonlocal t
        seconds[what], t = round(time.time() - t, 1), time.time()

    key = committer_key(Path(a.key))
    compiler, data = compiler_of_args(a), Path(a.advice).read_bytes()
    requests = compile_advice(compiler, data)
    registered, cls = compiler.inner.registered, window_class_of(a.window_class)
    try:
        compiled = compile_steps(registered, data, compiler.inner.ingress)
        WC.takes(compiled, cls)
        low = WC.lower(registered, cls)
    except Unexpressible as e:
        raise ValueError(f"I2's window circuit refuses the advice: {e}") from None
    lap("lower")
    r = record1(str(a.bundle), str(a.row_dir))
    run = bytes(r.run.run_root).hex()
    p = compiler.load(0, 1, requests)
    k0 = next(iter(p.progs))
    tree = json.loads((r.bdir / "weights" / "tree.json").read_text())
    model = registered.phi.model
    ck = WR.CheckpointIndex(a.checkpoint_dir, checkpoints_manifest=a.checkpoints, model=model["repo"], revision=model["revision"])
    opened = CPU.OnRead(r.bdir, tree, (r.row["commit"].get("weight_registration") or {}).get("root"), ck)
    read = CPU.provider(opened, ProgramIndex(p.progs[k0], program_dir=p.dirs[k0]).prefix, set())
    sizes = {label: ty.leaves for label, ty in dict(low.fn.signature()[0])[WC.WEIGHTS].fields}

    def weights(label: str, lo: int, hi: int) -> np.ndarray:
        w = read(label, lo, hi)
        if w is None:
            raise ValueError(f"weight {label}: no tensor of the committed weights holds its elements [{lo}, {hi})")
        return w

    bmap = json.loads((r.out / f"binding_map_p{r.pair}.json").read_text())
    store = CommittedStore.of_record(bmap, OpenedReader(r.store, r.run), r.manifest if isinstance(r.manifest, dict) else None)
    units = [u for u in compiled.requests if u is not None]
    slots, rows = [], []
    for k in range(cls.slots):
        u = units[k] if k < len(units) else None
        if u is None:
            row, max_tokens, served_ = [0] * (cls.prompt + 2), 0, None
        else:
            body = compiler.inner.ingress[u.id]
            ids, max_tokens = [int(x) for x in body["prompt"]], int(body["max_tokens"])
            row = [1, len(ids), *ids, *[0] * (cls.prompt - len(ids))]
            pi = ProgramIndex(p.progs[u.id], program_dir=p.dirs[u.id])
            served_ = WRS.Served.of_record(pi, store, u.id, p.arrivals[u.id])
        slots.append(WRS.Slot(low.fn, cls, row, weights, served_))
        rows.append((row, max_tokens))
    lap("slots")
    reg = WW.WeightsRegistration.of(json.loads(Path(a.registration).read_text()), a.registration)
    if (reg.repo, reg.revision) != (model["repo"], model["revision"]):
        raise ValueError(f"{a.registration} registers {reg.repo}@{reg.revision}'s weights, not the served "
                         f"{model['repo']}@{model['revision']}'s")
    served = {label: weights(label, 0, n) for label, n in sizes.items()}
    values = {**WW.weight_values(registered, served, key, keep_rows=True),
              SS.SCAN_STATE: WRS.standin_state(WC.NAME, cls, rows, key, int(a.window), window=run)}
    held = sorted(p for p, e in reg.values.items() if p in values and values[p].entry() == dict(e))
    lap("weights")
    filled = WRS.Rows(low, slots, values)
    lap("rows")
    win = IF.Window(low.circuit, run, DR.parse_law(a.law), a.cap, registered=values)
    win.supply(CR)
    filled.record(win)
    lap("record")
    c = win.close(workers=a.workers or min(16, os.cpu_count() or 1))
    lap("close")
    prov = {"window_circuit": {"class": [cls.slots, cls.prompt, cls.outputs, cls.lanes, cls.every], "window": int(a.window),
                               "requests": [u.id for u in units]},
            "activations": {"program": WC.NAME, "values": {SS.SCAN_STATE: values[SS.SCAN_STATE].entry()}}}
    win.write(c, a.out, prov, registrar=R)
    lap("write")
    sources = {s: int(sum((x == i).sum() for x in filled.source.values())) for i, s in enumerate(WRS.SOURCES)}
    return {"run": run, "circuit": hashlib.sha512((Path(a.out) / "circuit.json").read_bytes()).hexdigest(),
            "population": sum(filled.n.values()), "members": len(win.window.members), "sources": sources,
            "registration": {"held": len(held), "free": sorted(set(reg.values) - set(held)), "unrooted": sorted(reg.unrooted)},
            "seconds": seconds}


def standin_brought(body: Mapping[str, Any], seed: int) -> dict[str, Any]:
    """What a stand-in request's ingress brought: the record's body without its prompt, a stochastic one that names no seed
    bringing the declared run's default ``seed``, since the stand-in's server fills in the public one (`registered_seed`)."""
    out = {k: v for k, v in body.items() if k != "prompt"}
    return {**out, "seed": int(seed)} if out.get("temperature") and out.get("seed") is None else out


def serve_standin(a: argparse.Namespace) -> dict[str, Any]:
    """A served run's dir under ``a.out`` laid out from a single-GPU Commit of record, as serve-openai and `verity-vllm advice` lay
    one out: ``serve.json`` (the record's model and revision, the public default seed `engine.served_seed.DEFAULT_SEED`, and the
    execution and target φ reads from its row, `score.phi_of`), ``windows.jsonl`` (window 0 under the record's run root),
    ``requests.jsonl`` (each request's I1 id as its ``user``, its prompt and the parameters the record served it under as what
    its ingress brought, `standin_brought`) and ``advice/w0/``: the declared run's advice committed under the served compiler
    (registration.json, private/advice.bin, private/opening.json)."""
    from verity_vllm.engine.served_seed import DEFAULT_SEED

    r, rec = record1(str(a.bundle), str(a.row_dir)), i2_of_record(a.bundle, a.row_dir, a.workload, a.hf_config, a.cap_bits)
    phi, out, root = rec.registered.phi, Path(a.out), bytes(record1(str(a.bundle), str(a.row_dir)).run.run_root).hex()
    run_id = f"{r.row.get('label')}/w0"
    out.mkdir(parents=True, exist_ok=True)
    (out / "serve.json").write_text(json.dumps({
        "model": phi.model["repo"], "revision": phi.model["revision"], "default_seed": DEFAULT_SEED,
        "engine": {"execution": phi.execution, "pin": {"world": 1}}, "target": phi.target, "run_id": run_id,
        "stand_in": f"prove_path.py serve-standin: the Commit of record {r.row.get('label')} laid out as a serve-openai run"}, indent=1))
    reqs = [{"http_id": i, "engine_id": i, "user": i, "prompt_ids": body["prompt"],
             "ingress": {"brought": standin_brought(body, rec.registered.default_seed)}, "windows": [0]}
            for i, body in rec.ingress.items()]
    (out / "requests.jsonl").write_text("".join(json.dumps(q) + "\n" for q in reqs))
    (out / "windows.jsonl").write_text(json.dumps({"window": 0, "run_id": run_id, "run_root": root,
                                                   "requests": [q["engine_id"] for q in reqs], "passed": True}) + "\n")
    compiler = Capped(i2_of_serve(str(out), 0, a.row_dir, a.hf_config, None, a.cap_bits), a.cap_bits)
    data = AW.codec_of().encode(AW.advice_of_workload(json.loads(Path(a.workload).read_text())))
    _reqs, bound, reg = AW.of_advice(compiler, data)
    adv = out / "advice" / "w0"
    (adv / "private").mkdir(parents=True, exist_ok=True)
    (adv / "registration.json").write_text(json.dumps({
        "schema": "verity-vllm/advice-registration/v1", "scheme": AW.SCHEME, "binding": bound,
        "compiler": {**bound["compiler"], "cap_bits": compiler.cap_bits, "inner": {"name": compiler.inner.name, "digest": compiler.inner.digest}},
        "rows": len(reg.salts), "row_bits": AW.ROW_BITS, "window": 0, "commit": {"run_id": run_id, "run_root": root}}, indent=1))
    (adv / "private" / "advice.bin").write_bytes(data)
    (adv / "private" / "opening.json").write_text(json.dumps(AW.opening(data, reg)) + "\n")
    return {"run": str(out), "window": 0, "run_root": root, "compiler": bound["compiler"], "advice": bound["advice"],
            "requests": len(reqs)}


def main(argv: list[str] | None = None) -> int:
    from verity_vllm.pipeline.workload_file import workload_file

    def workload_arg(s: str) -> str:
        workload_file(s)
        return s

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("advice", help="a declared run's I1 advice from its workload file")
    p.add_argument("--workload", required=True, type=workload_arg)
    p.add_argument("--out", required=True, help="the advice's file: I1's canonical encoding")
    for name in ("commit", "audit"):
        p = sub.add_parser(name)
        p.add_argument("--tp-commit", help="a TP Commit's record (its --out: runs.jsonl, versions.json)")
        p.add_argument("--build-root", help="with --tp-commit: each rank's Programs of record in rank<r>/ (the stand-in compiler's)")
        p.add_argument("--bundle", help="instead of --tp-commit: a single-GPU Commit's sealed replay bundle (<row dir>/commit/"
                       "replay_bundle_p0), its window's Programs I2's")
        p.add_argument("--row-dir", help="with --bundle or --serve-run: the row dir (the Build's Programs, manifest.json, commit/)")
        p.add_argument("--workload", type=workload_arg, help="with --bundle: the workload file the Commit served (φ's declared inputs, I2's)")
        p.add_argument("--serve-run", help="instead: a serve-openai run dir (serve.json, windows.jsonl, requests.jsonl, advice/w<K>/), "
                       "its window --window from the sealed row dir --row-dir (bundle default <row dir>/commit/replay_bundle_p0)")
        p.add_argument("--window", type=int, help="with --serve-run: the closed Commit window K")
        p.add_argument("--target", help="with --serve-run: the serving device's target profile (JSON), where serve.json records none")
        p.add_argument("--hf-config", help="with --bundle or --serve-run: the checkpoint's config.json (default with --serve-run the "
                                           "served pin's config of record, `pinned_hf_config`; else unsloth/Llama-3.2-1B's)")
        p.add_argument("--opening", help="the advice's opening (advice_windows.opening): commit writes it, the prover's private file; "
                       "audit reads it, the named gap. A served window's is the server's (default advice/w<K>/private/opening.json)")
        p.add_argument("--cap-bits", type=int, default=CAP_BITS, help="ℓ, the public cap on the advice's length in bits, "
                       "registered in the compiler's digest")
        p.add_argument("--partition", required=True, help="the verifier's partition file: {templates, law, scope?}")
        p.add_argument("--ranks", default=None, help="comma-separated (default every rank)")
        p.add_argument("--jobs", type=int, default=1, help="ranks at once, each in its own process")
        p.add_argument("--transfer-gate", action="append", default=[], help="a template the registered compiler lists as holding "
                       "transfer gates, in its digest, every unit of which is proved (repeatable; the partition's law is work:K). A "
                       "served window's are its model's of record (GATES_OF_RECORD), which this may only restate")
        p.add_argument("--out", required=True)
    p = sub.choices["commit"]
    p.add_argument("--advice", help="the advice's file: I1's canonical encoding (the prover's, private); a served window's is the "
                   "server's (advice/w<K>/private/advice.bin)")
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
    p.add_argument("--plain", action="store_true", help="plain Flock sessions, not --zk (a5's --plain): the partition's law "
                   "stratified:K, each window one member with no registered reads or shared rows (a5.PLAIN_SCOPE)")
    p.add_argument("--challenger", help="draw each window on the auditor's challenger's coins, the run root's one session "
                   "(a5_coins), not the verifier's own: local (a development session on this machine), or gate:USER@HOST "
                   "(the auditor host through its SSH gate, with --peer and --challenger-key)")
    p.add_argument("--peer", help="with --challenger gate:USER@HOST: this machine as the gate's /etc/vy/peers names it, whose "
                   "127.0.0.1 the session's challenger listens on")
    p.add_argument("--challenger-key", help="with --challenger gate:USER@HOST: the challenger's key record of its deployment "
                   "(challenger-key/v1), never the session's key.json")
    p.add_argument("--gate-key", help="with --challenger gate:USER@HOST: the lane's own gate key (default ssh's own)")
    p.add_argument("--main", help="with --challenger: the verifier's own copy of main's first-parent history (git rev-list "
                   "--first-parent origin/main), which must hold the commit the session's host pins")
    p = sub.add_parser("compiler", help="a served run's registered compiler: what verity-vllm advice --compiler-name/--compiler-digest takes")
    p.add_argument("--serve-run", required=True)
    p.add_argument("--transfer-gate", action="append", default=[], help="a template the compiler lists as holding transfer gates "
                   "(repeatable; the model's of record, GATES_OF_RECORD, whatever it says: another list is refused)")
    p = sub.add_parser("weights-registration", help="a served run's weights registration, the committer's: what verity-vllm "
                       "advice --weights-of-record takes, and the roots the verifier pins (window-circuit --pins)")
    for flag in ("--serve-run", "--checkpoint-dir", "--out"):
        p.add_argument(flag, required=True)
    p.add_argument("--key", required=True, help="the committer's 32-byte salt key file, drawn there the first time (0600); "
                   "it never leaves the committer")
    p.add_argument("--checkpoints", help="the manifest pinning the checkpoint's shards (default the tree's manifests/checkpoints.json)")
    p.add_argument("--tables", help="the dir holding the engine's served rotary table, cos_sin_cache.npy (serve-openai's; "
                   "default SERVE_RUN/private): registered once within its bound, else unrooted")
    p.add_argument("--hf-config")
    p.add_argument("--target")
    p = sub.add_parser("serve-standin", help="a served run's dir laid out from a single-GPU Commit of record")
    for flag in ("--bundle", "--row-dir", "--workload", "--out"):
        p.add_argument(flag, required=True, **({"type": workload_arg} if flag == "--workload" else {}))
    p = sub.add_parser("circuit", help="a served window's registered compiler as a command: the circuit record it compiles the "
                       "advice to (records_audit.py's compile check)")
    for flag in ("--serve-run", "--partition", "--advice", "--out"):
        p.add_argument(flag, required=True)
    p.add_argument("--window", type=int, required=True)
    p.add_argument("--transfer-gate", action="append", default=[])
    p = sub.add_parser("window-circuit", help="a served window's advice through I2 as one Circuit (V3), its record written to --out")
    for flag in ("--serve-run", "--row-dir", "--advice", "--out"):
        p.add_argument(flag, required=True)
    p.add_argument("--window", type=int, required=True)
    p.add_argument("--cap", type=int, default=64, help="units a member (a statement), the record's stated cap")
    p.add_argument("--description", action="store_true", help="write the circuit's description (templates, wires and "
                   "declarations), not its record: no program is built")
    p.add_argument("--pin", action="append", default=[], type=pin_arg, metavar="NAME=ROOT", help="a registered parameter and "
                   "the root the verifier fixed for it before the run, as `records_audit.py count --pin` takes it (repeatable; "
                   "none: every registered read is free)")
    p.add_argument("--pins", action="append", default=[], metavar="FILE", help="pins as a JSON {param: root}, or a "
                   "registration of values {program, values}, the committer's weights registration "
                   "(window_weights.weights_registration), as `records_audit.py count --pins` takes them (repeatable)")
    p = sub.add_parser("class-size", help="a served window's padded size at its class, projected from a small probe class")
    for flag in ("--serve-run", "--row-dir", "--advice"):
        p.add_argument(flag, required=True)
    p.add_argument("--window", type=int, required=True)
    p.add_argument("--probe", default="1,2,2", help="the class SLOTS,PROMPT,OUTPUTS one Program is lowered at exactly")
    p = sub.add_parser("window-commit", help="a served window committed as its window class's one Circuit's units (V3c)")
    for flag in ("--serve-run", "--row-dir", "--advice", "--out"):
        p.add_argument(flag, required=True)
    p.add_argument("--window", type=int, required=True)
    p.add_argument("--key", required=True, help="the committer's secret key (32 bytes, the prover's private file; made "
                   "0600 where it doesn't exist), the same every window of a period")
    p.add_argument("--registration", required=True, help="the committer's weights registration under --key "
                   "(weights-registration), the verifier's pins (records_audit.py audit --registered)")
    p.add_argument("--checkpoint-dir", required=True, help="the served checkpoint, whose tensors the served weights are "
                   "opened from against the committed weights root")
    p.add_argument("--checkpoints", help="the manifest pinning the checkpoint's shards, checked when given (each tensor is "
                   "held to its committed tensor root either way)")
    p.add_argument("--law", default="subset:1", help="the proof system's law of the window")
    p.add_argument("--cap", type=int, default=64, help="units a member (a statement)")
    p.add_argument("--workers", type=int, default=None, help="commit workers (default min(16, CPUs))")
    for name in ("window-circuit", "class-size", "window-commit"):
        sub.choices[name].add_argument("--class", dest="window_class", help="the window class SLOTS,PROMPT,OUTPUTS[,LANES,"
                                       "EVERY] (`window_circuit.WindowClass`; by default `CLASS_OF_RECORD`, 8,1024,64,4,16)")
    sub.choices["audit"].add_argument("--builds", help="with --bundle or --serve-run: where the verifier writes its own Build of "
                                      "each Program I2 compiles (own_builds; default OUT/builds), never a row dir")
    for name in ("audit", "circuit"):
        sub.choices[name].add_argument("--vllm-python", help="ignored: the verifier writes its own Builds from its compile, "
                                       "with no vLLM (own_builds)")
    for name in ("compiler", "serve-standin", "circuit", "window-circuit", "class-size", "window-commit"):
        sub.choices[name].add_argument("--hf-config")
        sub.choices[name].add_argument("--cap-bits", type=int, default=CAP_BITS)
    for name in ("compiler", "circuit", "window-circuit", "class-size", "window-commit"):
        sub.choices[name].add_argument("--target")
    a = ap.parse_args(argv)
    if a.cmd == "compiler":
        print(json.dumps(compiler_doc(a.serve_run, a.hf_config, a.target, a.cap_bits, a.transfer_gate), sort_keys=True))
        return 0
    if a.cmd == "window-commit":
        a.bundle = str(Path(a.row_dir) / "commit" / "replay_bundle_p0")
    if a.cmd in ("circuit", "window-circuit", "class-size", "window-commit", "weights-registration"):
        commands = {"circuit": circuit_of_advice, "window-circuit": window_circuit_of_advice, "class-size": class_size_of_advice,
                    "window-commit": window_commit_of_advice, "weights-registration": weights_registration_of_serve}
        try:
            print(json.dumps(commands[a.cmd](a), sort_keys=True))
        except ValueError as e:
            print(f"prove_path.py {a.cmd}: {e}", file=sys.stderr)
            return REFUSED
        return 0
    if a.cmd == "serve-standin":
        print(json.dumps(serve_standin(a), sort_keys=True))
        return 0
    if a.cmd == "advice":
        c = AW.codec_of()
        advice = AW.advice_of_workload(json.loads(Path(a.workload).read_text()), c)
        data = c.encode(advice)
        AW.requests_of(advice)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_bytes(data)
        print(json.dumps({"advice": a.out, "requests": len(advice.requests), "steps": len(advice.steps), "bits": c.total_bits(advice)}))
        return 0
    if served(a):
        if a.tp_commit or a.workload or a.window is None or not a.row_dir or getattr(a, "advice", None):
            ap.error("a served window: --serve-run DIR --window K --row-dir DIR, its advice and opening the server's (no --tp-commit, "
                     "--workload or --advice)")
        a.bundle = a.bundle or str(Path(a.row_dir) / "commit" / "replay_bundle_p0")
        a.opening = a.opening or str(Path(a.serve_run) / "advice" / f"w{a.window}" / "private" / "opening.json")
    elif single(a) == bool(a.tp_commit) or (a.tp_commit and not a.build_root) or (single(a) and not (a.row_dir and a.workload)):
        ap.error("one committed run: --tp-commit DIR --build-root DIR, --bundle DIR --row-dir DIR --workload F, or --serve-run DIR "
                 "--window K --row-dir DIR")
    elif not a.opening or (a.cmd == "commit" and not a.advice):
        ap.error("--opening FILE (and commit's --advice FILE)")
    if a.cmd == "audit" and single(a):
        a.builds = os.path.abspath(a.builds or Path(a.out) / "builds")
        if Path(a.builds).resolve().is_relative_to(Path(a.row_dir).resolve()):
            ap.error(f"--builds {a.builds} is under the row dir {a.row_dir}: the verifier reads only Builds it wrote itself")
    if a.cmd == "audit" and a.challenger:
        if a.challenger != "local" and not a.challenger.startswith("gate:"):
            ap.error(f"--challenger is local or gate:USER@HOST, not {a.challenger!r}")
        if a.challenger.startswith("gate:") and not (a.peer and a.challenger_key):
            ap.error("--challenger gate:USER@HOST needs --peer and --challenger-key")
        if not a.main:
            ap.error("--challenger needs --main: the verifier's `git rev-list --first-parent origin/main`, which must hold the "
                     "commit the session's host pins")
    return commit(a) if a.cmd == "commit" else audit(a)


if __name__ == "__main__":
    raise SystemExit(main())
