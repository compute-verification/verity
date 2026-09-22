"""One command: a supported, composed example end to end through the existing machinery (lane INT).

    python -m e2e.run_composed --case B0 --out DIR [--challenge security|diagnostic --seed N]

Stages, each calling an existing module (nothing here is a demo path; the seams added for this command
are ``veritor.plans.verity_ir_adapter``, ``verity_vllm_adapter.obligations``,
``verity_vllm_adapter.coordinate_check.coordinate_openings``, ``verity_vllm.typed_obligation_export.export_composed``
and ``verity_vllm_adapter.weight_sources.checkpoint_row_words``):

1. **program** -- TA1's typed ``Program`` for the case (``Serve{C=B0, LP, STEPS}`` from the workload) is
   constructed and its descriptor digest fixed (``verity.ir``); the layer sub-program ``LayerPost{C=B0}``
   likewise.
2. **plan** -- the layer sub-program is bridged to RA's plan IR and two partitions are validated
   (``veritor.plans.validate``: executability, acyclicity of the quotient, one-producer) with the bound
   (``veritor.plans.bound``): P1 at function-call granularity (the adapter's row VUs), P2 with every GEMM
   refined to coordinate VUs (the units the backend proves).  The full ``Serve`` program is *not* put through
   the validator (its 5e8 gates exceed the validator's design point); this is declared in ``summary.json``.
3. **production** -- vLLM production capture (``verity_vllm_adapter.production``).
4. **preparation** -- ``full replay (selective preparation pending RC)``: two reference replays and one
   commit replay (``verity_vllm_adapter.replay``), noninterference (``compare``), then the committed weights
   converted to capture format v2 (row leaves, ``verity_vllm.weights_v2``) so coordinates are addressable.
5. **verify** -- WB's staged security-mode challenge and the adapter verifier's row over the bundle
   (``verity_vllm_adapter.verify --challenge security``): bindings, certified + coverage samples, the
   same-runtime / independent relations, the negatives.  Every sampled VU is checked here.
6. **obligations** -- the sampled GEMM rows' coordinates (a deterministic sub-selection seeded by the
   challenge digest), sampled ``Input`` equations (checkpoint weight words; the request's boundary tokens)
   and the boundary leaves, exported as one composed ``typed-obligation/v0`` object with shared authenticated
   values; relation check (``veritor.interface.check_relations``) and the transparent backend on the witness.
7. **prove** -- the generic SP1 guest (PW's packed authentication, ``--typed-object-digest h_O``): execute,
   core proof, compressed proof, host verification; ``check_binding`` over the 65-byte public values (IB).
8. **bundle** -- the tarball a second process re-verifies (``verity_vllm.bundle``).

The final verdict is the conjunction of: the adapter verifier's row, the plan validator, every proof
verified, every exported obligation's verdict, every binding check.  ``--negative`` injects one of WA's
labelled mutations into a copy of the committed bundle before stage 5 (``docs/verified-backend/audit-security-baseline.md``):
``wa-m12`` flips one bit of a stored weight value; ``wa-m8b`` drops every node of one layer from
``description.json``.  The verdict must then be FAIL and ``summary.json`` names the stage that caught it.

Environment (pod defaults; override with flags or ``VERITY_*``): the pinned vLLM venv for stages 3-5, the
current interpreter (Python >= 3.12 with ``verity.ir``, ``veritor``, numpy) for the rest, the CUDA SP1 host
for stage 7.  ``--reuse-adapter DIR`` reuses another run's stages 3-4 (labelled as such in ``summary.json``).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from fractions import Fraction
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
POC = HERE.parent
ROOT = POC.parent
for p in (str(POC), str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

SCHEMA = "verity-vllm/e2e-composed/v1"
PREPARATION_LABEL = "full replay (selective preparation pending RC)"  # --preparation full: INT's original stage 4, kept as run
SELECTIVE_LABEL = ("RC selective preparation (research-c-replay.md, decision 3): boundaries of every request + weights (capture format v2) "
                   "committed before r_q; r_q selects the requests J; only J is replayed (its own batch) and only J's interiors are committed; "
                   "prover = verity_vllm_adapter.selective_prover, verifier = verity_vllm_adapter.selective_verifier, file handshake relayed by run_composed")
# the handshake files of RC's protocol (verity_vllm_adapter.selective_prover; not imported here: that module imports torch/vLLM)
HS_READY0, HS_CHALLENGE, HS_READY2, HS_ABORT = "READY_PHASE0", "replay_challenge.json", "READY_PHASE2", "ABORT"
# where a labelled mutation is applied in the selective protocol: after the prover has finalised the artefact it targets and
# before the verifier reads it (phase-0 artefacts -- description, weights -- at READY_PHASE0; interiors at READY_PHASE2).
# wa-m12 targets weights/values.bin, a phase-0 artefact, but the prover re-hashes the weights when it builds the final
# commitments in phase 2 (build_commitments recompute_leaves=True) and would itself refuse; the verifier is the party that must
# catch a value that no longer matches its committed leaf, so the flip is applied once the prover has exited (READY_PHASE2).
NEGATIVE_POINT = {"wa-m12": HS_READY2, "wa-m8b": HS_READY0}
TC_FAMILY = "veritor.tc-ampere-bf16@1"
WORKLOADS = {
    "B0": "B0 (SmolLM2-135M, eager verifiable profile, batch-invariant kernels), the small security row's configuration: the workload's requests, prompt 32 tokens, 16 output tokens",
    "B1": ("B1 (Qwen2.5-1.5B, eager verifiable profile, batch-invariant kernels): e2e/workload-B1-512.json -- the standard generator's "
           "cells {32, 128, 512} prompt tokens x 16 output tokens x 4 prompts = 12 requests, i.e. the cells inside the pinned "
           "Serve{C=B1, LP=512, STEPS=15} shape (the 410 MB descriptor, typed-b1.md); the 2048-token / 64-token cells of the 16-request "
           "standard workload are excluded: Serve{LP=2048, STEPS=63} cannot be built under the pod's 31 GB host cgroup"),
}
# the verifier's cost is recorded, not fixed (int3-brief §3): what the number means and where the leverage is known to be
VERIFIER_COST_NOTE = {
    "recorded_not_fixed": True,
    "known_leverage": "RC's selective verifier memoises Merkle paths (MemoCache); the residual cost is dominated by the per-VU relation recompute in numpy and by "
                      "per-request Merkle authentication of J's interiors; WB's full verifier (verify.py) walks every request. Neither is optimised here.",
}
PROVED_FAMILIES = {"gemm": "GEMM coordinate (BF16, tc-ampere)", "lm_head": "GEMM coordinate (BF16, tc-ampere): the LM-head GEMM is coordinate-addressable too"}
NEGATIVES = {
    "wa-m12": "WA M12: one bit of a stored weight value flipped in commit/weights/values.bin (index untouched)",
    "wa-m8b": "WA M8b: every node of one layer removed from commit/description.json",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(doc: Any) -> str:
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def host_json(stdout: str) -> dict[str, Any]:
    """The host's JSON document: the last stdout line that parses as a JSON object (the CUDA prover's
    ``sp1-gpu-server`` interleaves its own log lines on stdout)."""
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    raise RuntimeError(f"host printed no JSON document: {stdout[-400:]!r}")


PLAN_CHECKS = ["executable (every VU's reads are produced or committed)", "quotient acyclic", "one producer per value"]


def static_configuration(case: str) -> dict[str, Any]:
    """TA1's static configuration for the case (``verity_vllm.registry.b1``): ``B0`` / ``B1``."""
    from verity_vllm.registry import b1

    try:
        return dict(getattr(b1, f"{case}_CONFIG"))
    except AttributeError as e:
        raise SystemExit(f"no static configuration {case!r} in verity_vllm.registry.b1") from e


def unit_program(config: dict[str, Any], unit: str):
    """The verity-ir ``Program`` a plan unit names: ``layer`` = ``LayerPost{C}``; ``gemm:<K>x<N>`` = the bare
    ``Gemm{K,N}`` function (one of the layer's GEMM shapes, validated on its own when the layer does not fit)."""
    from verity.ir import Program, bind
    from verity_vllm.registry import b1

    if unit == "layer":
        return Program(bind(b1.LayerPost, C=config))
    if unit.startswith("gemm:"):
        k, n = unit[5:].lower().split("x")
        return Program(bind(b1.Gemm, K=int(k), N=int(n)))
    raise ValueError(f"unknown plan unit {unit!r} (layer | gemm:<K>x<N>)")


def layer_gemm_shapes(config: dict[str, Any]) -> list[str]:
    """The layer's four GEMM shapes as plan units, in call order: qkv, o, gate_up, down."""
    H, NH, D = int(config["H"]), int(config["NH"]), int(config["D"])
    return [f"gemm:{H}x{int(config['QKV'])}", f"gemm:{NH * D}x{H}", f"gemm:{H}x{int(config['GU'])}", f"gemm:{int(config['I'])}x{H}"]


def unit_plans(config: dict[str, Any], unit: str = "layer"):
    """A plan unit (``unit_program``) bridged to the plan IR, and the two partitions the validator checks.

    Returns ``(Q, circuit, [("P1", plan), ("P2", plan)])``: P1 has one VU per function call of the unit
    (the adapter's row VUs); P2 refines every GEMM call to its coordinate VUs (the units the backend
    proves) and keeps the function-call units elsewhere.  Both share the CUs: the unit's inputs (the
    layer's ``attn``/``res`` and ``weights``; a bare GEMM's ``x`` and ``w``) and every call's outputs (the
    committed rows).  Shared by ``run_composed`` (validation) and ``verify_composed`` (re-validation
    from the case name and the recorded unit alone)."""
    from veritor.plans.boundary import Part, Parts
    from veritor.plans.plan import VerificationPlan
    from veritor.plans.verity_ir_adapter import program_from_verity_ir

    Q = program_from_verity_ir(unit_program(config, unit))
    c = Q.circuit()
    root = Q.root
    inputs = [k for k, n in enumerate(root.nodes) if n.form == "batch" and n.fn.name == "Input"]
    calls = [k for k, n in enumerate(root.nodes) if k not in inputs]
    gemm_calls = [k for k in calls if root.nodes[k].fn.name.startswith("Gemm_")]  # a layer: its Gemm{K,N} calls
    coord_batches = [k for k in calls if root.nodes[k].fn.name.startswith("GemmCoordinate")]  # a bare Gemm: the root IS the batch of coordinates

    def whole(k: int) -> Parts:
        n = root.nodes[k]
        return Parts(c, [Part((), k, 0, n.n if n.form == "batch" else 1)])

    if len(inputs) >= 3:  # the layer: (attn, res) | weights | every call's outputs
        cus = [Parts(c, [Part((), k, 0, root.nodes[k].n) for k in inputs[:2]]), whole(inputs[2]), Parts(c, [Part((), k, 0, 1) for k in calls])]
    else:  # a bare GEMM: x | w | the coordinate batch's outputs
        cus = [whole(k) for k in inputs] + [Parts(c, [Part((), k, 0, root.nodes[k].n if root.nodes[k].form == "batch" else 1) for k in calls])]
    input_vus = [whole(k) for k in inputs]  # a partition covers every gate: the Input batches are (trivial) VUs of their own
    p1 = VerificationPlan("P1 function-call units (the adapter's row VUs)", input_vus + [whole(k) for k in calls], cus)
    vus2: list[Parts] = list(input_vus)
    for k in calls:
        if k in gemm_calls:
            n = root.nodes[k].fn.nodes[0].n
            vus2 += [Parts(c, [Part(((k, 0),), 0, j, j + 1)]) for j in range(n)]
        elif k in coord_batches:
            vus2 += [Parts(c, [Part((), k, j, j + 1)]) for j in range(root.nodes[k].n)]
        else:
            vus2.append(whole(k))
    p2 = VerificationPlan("P2 GEMM coordinate units (the backend's proved units) + function-call units elsewhere", vus2, cus)
    return Q, c, [("P1", p1), ("P2", p2)]


def layer_plans(config: dict[str, Any]):
    """INT's name for ``unit_plans(config, "layer")``."""
    return unit_plans(config, "layer")


def plans_digest(Q, partitions) -> str:
    """The plan digest recorded in ``manifest.json``: the bridged program's name and every partition's
    ``VerificationPlan.to_json()`` (name, every VU and CU as parts) -- a function of the plans' content."""
    return sha256_json({"program": Q.name, "partitions": {tag: plan.to_json() for tag, plan in partitions}})


@dataclass
class Stage:
    name: str
    status: str = "not run"  # ok | FAIL | error | skipped | not run
    seconds: float = 0.0
    detail: str = ""
    artefacts: dict[str, str] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"status": self.status, "seconds": round(self.seconds, 3), "detail": self.detail, "artefacts": self.artefacts, **self.data}


class Run:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.out = Path(args.out).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "logs").mkdir(exist_ok=True)
        self.log_path = self.out / "log.txt"
        self.stages: dict[str, Stage] = {}
        self.manifest: dict[str, Any] = {"schema": SCHEMA, "case": args.case, "utc_start": utc(), "command": " ".join(sys.argv), "inputs": {},
                                         # integration-pad.md §2 item 1: the record names its own archive location
                                         # (fixture copies omit adapter/ + bundle/ + bin/; the archive carries them)
                                         "archive": {"convention": f"~/veritor-artifacts/int/INT-PAD-2/{Path(args.out).name}",
                                                     "carries": ["adapter/ (weights values, production, selective bundle)", "bundle/ (tarball)", "bin/ (host binary + guest elf)"],
                                                     "committed_fixture_omits": ["adapter/", "bundle/"]}}
        self.t0 = time.time()
        self.adapter = Path(args.reuse_adapter).resolve() if args.reuse_adapter else self.out / "adapter"
        self.adapter.mkdir(parents=True, exist_ok=True)
        # the selective bundle is always this run's own (r_q is drawn per run; a negative's mutation touches only its own copy)
        self.selective_dir = self.out / "adapter" / "selective"
        self.selective = args.preparation == "selective"
        self.protocol: dict[str, Any] = {}

    # -- plumbing --------------------------------------------------------------------------------

    def log(self, msg: str) -> None:
        line = f"[{utc()}] {msg}"
        print(line, flush=True)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    def stage(self, name: str) -> Stage:
        s = self.stages.setdefault(name, Stage(name))
        return s

    def run_stage(self, name: str, fn, *, fatal: bool = True) -> Stage:
        s = self.stage(name)
        self.log(f"stage {name}: start")
        t = time.perf_counter()
        try:
            fn(s)
            if s.status == "not run":
                s.status = "ok"
        except Exception as e:  # noqa: BLE001
            s.status = "error"
            s.detail = f"{type(e).__name__}: {e}"
            (self.out / "logs" / f"{name}.traceback.txt").write_text(traceback.format_exc())
            self.log(f"stage {name}: ERROR {s.detail}")
            if fatal:
                self.finish(final_note=f"stage {name} raised; later stages not run")
                raise SystemExit(1)
        s.seconds = time.perf_counter() - t
        self.log(f"stage {name}: {s.status} in {s.seconds:.1f}s {s.detail[:200]}")
        self.write_summary(partial=True)
        return s

    def subprocess(self, name: str, cmd: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None) -> int:
        log = self.out / "logs" / f"{name}.log"
        self.log(f"  $ {' '.join(cmd)}  > {log.relative_to(self.out)}")
        full_env = dict(os.environ)
        full_env.update(env or {})
        with open(log, "ab") as f:
            f.write(f"# {utc()} {' '.join(cmd)}\n".encode())
            f.flush()
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=full_env, cwd=str(cwd or POC))
        return rc

    def popen(self, name: str, cmd: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None) -> tuple[subprocess.Popen, Any]:
        """A concurrent subprocess (the selective protocol's two parties); the caller waits on it."""
        log = self.out / "logs" / f"{name}.log"
        self.log(f"  $ {' '.join(cmd)}  > {log.relative_to(self.out)}  &")
        full_env = dict(os.environ)
        full_env.update(env or {})
        f = open(log, "ab")  # noqa: SIM115 -- closed by the caller after wait()
        f.write(f"# {utc()} {' '.join(cmd)}\n".encode())
        f.flush()
        return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=full_env, cwd=str(cwd or POC)), f

    def vllm_env(self, stage: str) -> dict[str, str]:
        cache = Path(self.args.cache_root) / f"{self.args.case}-{self.out.name}-{stage}"
        (cache / "inductor").mkdir(parents=True, exist_ok=True)
        (cache / "triton").mkdir(parents=True, exist_ok=True)
        env = {"PYTHONPATH": str(POC), "VLLM_CACHE_ROOT": str(cache), "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"),
               "TRITON_CACHE_DIR": str(cache / "triton"), "HF_HUB_OFFLINE": "1"}
        if self.args.hf_home:
            env["HF_HOME"] = self.args.hf_home
        return env

    def rel(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.out))
        except ValueError:
            return str(path)

    # -- stage 1: program -------------------------------------------------------------------------

    def stage_program(self, s: Stage) -> None:
        from verity.ir import Program, bind, descriptor_size, encode_program, program_digest
        from verity_vllm.registry import b1

        workload = json.loads(Path(self.args.workload).read_text())
        lp = max(int(x) for x in workload["prompt_lengths"])
        steps = max(int(x) for x in workload["max_tokens"]) - 1
        config = static_configuration(self.args.case)
        pdir = self.out / "program"
        pdir.mkdir(exist_ok=True)
        padded_block: dict[str, Any] | None = None
        if getattr(self.args, "padded_program", None):
            # [PAD-INT, additive; default off = the unpadded Serve_v1 below] GUIDANCE §51.4: the record's program is the
            # DERIVED padded program (derive_cli serve --padded -> Serve_v3{C3, LP, D_MAX=STEPS}), so the certificate and
            # the pad statements name ONE digest. `derive` runs derive_cli here; `derived:<descriptor.json.gz>` pins a
            # descriptor it already wrote. Either way the hand-built registered oracle serve3.serve_v3 is rebuilt and its
            # digest must equal the derived one (the spec's identity claim), else the stage FAILs.
            program, desc, digest, padded_block = self._padded_program(config, lp, steps, pdir)
        else:
            program = Program(bind(b1.Serve, C=config, LP=lp, STEPS=steps))
            desc = encode_program(program)
            digest = program_digest(desc)
        raw = json.dumps(desc, sort_keys=True).encode()
        with gzip.open(pdir / "descriptor.json.gz", "wb") as f:
            f.write(raw)
        layer = Program(bind(b1.LayerPost, C=config))
        ldesc = encode_program(layer)
        ldigest = program_digest(ldesc)
        (pdir / "layer_descriptor.json").write_text(json.dumps(ldesc, sort_keys=True))
        doc = {
            "schema": "verity-ir/program-summary/v1",
            "program": {"function": program.fn.id, "digest": digest, "gates": program.gates, "input_gates": program.input_gates,
                        "descriptor_bytes": descriptor_size(desc), "descriptor_sha256": hashlib.sha256(raw).hexdigest(),
                        "descriptor": "descriptor.json.gz", "static_configuration": {k: v for k, v in config.items() if k != "profile"},
                        "profile": config.get("profile"), "LP": lp, "STEPS": steps, **(padded_block or {}).get("program", {})},
            "layer_program": {"function": layer.fn.id, "digest": ldigest, "gates": layer.gates, "input_gates": layer.input_gates,
                              "descriptor": "layer_descriptor.json", "role": "the sub-program whose plan the validator checks (stage plan)"},
            "vocabulary": "Program / FunctionCall / Subcircuit / Obligation (GUIDANCE §13.3); the descriptor is TA1's verity.ir codec",
        }
        if padded_block:
            doc["derived"] = padded_block["derived"]
            doc["padded"] = padded_block["padded"]
        write_json(pdir / "program.json", doc)
        s.artefacts = {"program": self.rel(pdir / "program.json"), "descriptor": self.rel(pdir / "descriptor.json.gz")}
        s.data = {"program_digest": digest, "layer_program_digest": ldigest, "gates": program.gates, "input_gates": program.input_gates}
        s.detail = f"{program.fn.id}: {program.gates:,} gates ({program.input_gates:,} Input); digest {digest[:16]}; layer {layer.fn.id} digest {ldigest[:16]}"
        self.manifest["inputs"]["program_digest"] = digest
        self.manifest["inputs"]["layer_program_digest"] = ldigest
        self.manifest["inputs"]["workload_sha256"] = sha256_file(Path(self.args.workload))
        # NOTE 15:20 item 3: the workload is part of the committed record set, not found by convention
        shutil.copyfile(self.args.workload, self.out / "workload.json")
        s.artefacts["workload"] = "workload.json"
        self._config = config

    def _padded_program(self, config: dict, lp: int, steps: int, pdir: Path):
        """[PAD-INT, additive] the DERIVED padded program for `--padded-program derive | derived:<descriptor.json.gz>`.
        Returns `(program_like, desc, digest, block)`; `program_like` has `.fn.id/.gates/.input_gates` like `Program`."""
        from verity.ir import Program, bind, encode_program, program_digest
        from verity.ir.codec import decode_program
        from verity.ir.defs import REGISTRY
        from verity_vllm.registry import b1, serve3  # noqa: F401  (serve3 registers the pad-serve@1 kinds the descriptor uses)

        spec = str(self.args.padded_program)
        eos = int(getattr(self.args, "eos_id", 0) or 0)
        d_max = steps  # lead's decision on serve3-spec §15 Q1: D_MAX := STEPS = max_tokens - 1
        C3 = {**config, "EOS_ID": eos}
        oracle = Program(serve3.serve_v3(C3, lp, d_max))
        oracle_desc = encode_program(oracle)
        oracle_digest = program_digest(oracle_desc)
        provenance: dict[str, Any] = {}
        if spec == "derive":
            ddir = self.out / "program" / "derived"
            cmd = [self.args.derive_python or sys.executable, "-m", "verity_vllm.frontend.derive_cli", "serve", "--config", self.args.case, "--lp", str(lp),
                   "--steps", str(steps), "--padded", "--eos-id", str(eos), "--out", str(ddir), "--expect-digest", oracle_digest]
            rc = self.subprocess("derive-padded", cmd, env={"PYTHONPATH": f"{ROOT}:{ROOT / 'src'}:{POC}"}, cwd=ROOT)
            if rc != 0:
                raise RuntimeError(f"derive_cli serve --padded exited {rc} (see logs/derive-padded.log)")
            dpath = ddir / "program" / "descriptor.json.gz"
            provenance = {"source": "derive_cli serve --padded (run by this stage)", "derived_dir": self.rel(ddir),
                          "program_json": json.loads((ddir / "program" / "program.json").read_text()).get("derived")}
        elif spec.startswith("derived:"):
            dpath = Path(spec.split(":", 1)[1])
            provenance = {"source": f"pinned derive_cli descriptor {dpath}", "descriptor_sha256_gz": sha256_file(dpath)}
            pj = dpath.parent / "program.json"
            if pj.exists():
                provenance["program_json"] = json.loads(pj.read_text()).get("derived")
                shutil.copyfile(pj, pdir / "derived-program.json")
        else:
            raise RuntimeError(f"--padded-program must be `derive` or `derived:<descriptor.json.gz>`, got {spec!r}")
        with gzip.open(dpath, "rb") as f:
            desc = json.loads(f.read())
        digest = program_digest(desc)
        if digest != oracle_digest:
            raise RuntimeError(f"derived padded program digest {digest[:16]} != registered oracle Serve_v3 {oracle_digest[:16]} "
                               f"(serve3.serve_v3({{**{self.args.case}, EOS_ID={eos}}}, LP={lp}, D_MAX={d_max}))")
        decoded = decode_program(desc, REGISTRY)  # the DERIVED descriptor is the record's program; the oracle only certifies its identity
        root_id = decoded.root.id
        fn_id = root_id[len("Root["):-len("]_v1")] if root_id.startswith("Root[") and root_id.endswith("]_v1") else oracle.fn.id
        block = {
            "program": {"padded": True, "D_MAX": d_max, "EOS_ID": eos, "designated_slots": d_max + 1, "slot_positions": list(range(lp - 1, lp + d_max)),
                        "declared_domain": 16, "designation": "token slots only (GUIDANCE §32.6; serve3-spec errata 2026-09-08); the live vector is internal, not designated",
                        "d_max_convention": "D_MAX := STEPS = max_tokens - 1; designated = D_MAX+1 token slots at t = LP-1+k (lead's decision, serve3-spec §15 Q1)"},
            "derived": {"padded": True, **provenance, "oracle": {"function": oracle.fn.id, "digest": oracle_digest, "gates": oracle.gates,
                                                              "identity": "derived descriptor digest == program_digest(Program(serve3.serve_v3(C3, LP, D_MAX))) (checked here)"}},
            "padded": {"evidence_label": "DERIVED padded program (verity_vllm.frontend.derive_cli serve --padded); identity = registered Serve_v3 oracle, digest-equal",
                       "unpadded_counterpart": {"function": f"Serve_v1{{C=...,LP={lp},STEPS={steps}}}",
                                                "digest": program_digest(encode_program(Program(bind(b1.Serve, C=config, LP=lp, STEPS=steps))))},
                       "prefix_property": "live region == Serve_v2{C,LP,STEPS=l,...} gate-for-gate by construction; == Serve_v1{C,LP,STEPS=l} via CF1's identity (serve3-spec §13)"},
        }

        class _Like:  # what the rest of stage_program reads from a `Program`
            fn = type("F", (), {"id": fn_id})()
            gates = decoded.gates
            input_gates = oracle.input_gates

        return _Like(), desc, digest, block

    # -- stage 2: plan ----------------------------------------------------------------------------

    def stage_plan(self, s: Stage) -> None:
        units: list[str] = list(self.args.plan_unit)
        pdir = self.out / "plan"
        pdir.mkdir(exist_ok=True)
        report: dict[str, Any] = {"schema": "veritor-plans/validation-report/v2", "layer_program_digest": self.stages["program"].data["layer_program_digest"],
                                  "units": {}, "unit_order": units,
                                  "unit_note": ("`layer` = LayerPost{C}, INT's unit (P1 row VUs / P2 GEMM coordinate VUs); `gemm:<K>x<N>` = one of the layer's "
                                                "GEMM shapes validated as its own program, used when the layer does not fit the host memory of the pod (the "
                                                "validator holds one interval per gate; the layer's weights are Input gates)")}
        all_valid = True
        details: list[str] = []
        digests: dict[str, str] = {}
        for unit in units:
            # one subprocess per unit (e2e/validate_unit.py): the validator's circuit is released to the OS before the GPU
            # stages, so the runner stays small beside the vLLM prover/verifier under the pod's cgroup (B1's largest GEMM unit peaks ~20 GB)
            ufile = pdir / f"unit-{unit.replace(':', '-')}.json"
            cmd = [sys.executable, "-m", "e2e.validate_unit", "--case", self.args.case, "--unit", unit, "--out", str(ufile),
                   "--bound-steps", str(int(self.args.bound_steps)), "--q", str(self.args.q), "--s", str(self.args.s)]
            self.log("  $ " + " ".join(cmd))
            t = time.perf_counter()
            with (self.out / "logs" / f"validate-unit-{unit.replace(':', '-')}.log").open("a") as f:
                rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONPATH": f"{POC}:{ROOT}:{ROOT / 'src'}"}, cwd=str(POC))
            udoc: dict[str, Any] = json.loads(ufile.read_text()) if ufile.exists() else {"unit": unit, "status": "no record", "partitions": {}}
            udoc["exit_code"], udoc["wall_seconds"] = rc, round(time.perf_counter() - t, 1)
            if rc not in (0, 1, 2, 3):
                udoc["status"] = f"killed (exit {rc}) at phase {udoc.get('status')}"
            unit_valid = bool(udoc.get("valid")) and rc == 0
            all_valid &= unit_valid
            for tag, e in udoc["partitions"].items():
                self.log(f"  plan {unit} {tag}: {e['status']} vus={e['vus']} edges={e['vu_edges']}/{e['cu_edges']} {e['validate_seconds']:.1f}s (unit process peak RSS {udoc.get('maxrss_mb')} MB)")
            if not unit_valid:
                self.log(f"  plan {unit}: {udoc.get('status')} (exit {rc})")
            digests[unit] = udoc.get("plan_digest") or ""
            report["units"][unit] = udoc
            details.append(f"{unit} ({udoc.get('program')}): " + (", ".join(f"{k} {e['status']} ({e['vus']} VUs, bound {(e.get('bound') or {}).get('bits')})" for k, e in udoc["partitions"].items())
                                                                  or str(udoc.get("status"))))
        if units == ["layer"]:  # INT's records: the layer's own digest is the plan digest (unchanged for B0 records)
            plan_digest = digests["layer"]
            report["program"] = report["units"]["layer"]["program"]
            report["partitions"] = report["units"]["layer"]["partitions"]  # v1 readers
        else:
            plan_digest = sha256_json({"units": digests})
        if self.args.plan_probe:
            probe = json.loads(Path(self.args.plan_probe).read_text())
            report["unit_probe"] = probe  # the measured memory finding behind the unit choice (e2e/probe_plan_units.py)
            shutil.copy(self.args.plan_probe, pdir / "unit_probe.json")
        validated_units: list[str] = units
        unsupported_units: list[str] = []  # the layer covers its four GEMMs; a GEMM-shape unit list leaves the layer (and any missing shape) unvalidated
        if "layer" not in units:
            unsupported_units = ["layer"] + [shape for shape in layer_gemm_shapes(self._config) if shape not in units]
        report["validated_units"] = validated_units
        report["not_validated_units"] = unsupported_units
        report["full_program"] = self._validate_full_program(pdir)
        report["plan_digest"] = plan_digest
        report["valid"] = all_valid
        write_json(pdir / "validator_report.json", report)
        s.artefacts = {"validator_report": self.rel(pdir / "validator_report.json"), "full_program": self.rel(pdir / "full_program.json")}
        s.status = "ok" if all_valid else "FAIL"
        fp = report["full_program"]
        first = report["units"][units[0]]
        s.data = {"plan_digest": plan_digest, "valid": all_valid, "units": units, "not_validated_units": unsupported_units,
                  "partitions": {k: {"status": v["status"], "vus": v["vus"], "bound_bits": (v.get("bound") or {}).get("bits")} for k, v in first["partitions"].items()},
                  "units_detail": {u: {k: {"status": v["status"], "vus": v["vus"], "bound_bits": (v.get("bound") or {}).get("bits")} for k, v in ud["partitions"].items()} for u, ud in report["units"].items()},
                  "unit_probe": ({"results": [{k: r.get(k) for k in ("unit", "partition", "status", "maxrss_mb", "validate_seconds", "wall_seconds", "cap_gb")} for r in report["unit_probe"].get("results", [])]}
                                 if "unit_probe" in report else None),
                  "full_program": {k: fp.get(k) for k in ("attempted", "validated", "status", "seconds", "maxrss_mb", "cap_gb", "verdict", "reason", "phases", "exit_code", "plan")}}
        s.detail = "; ".join(details)
        if unsupported_units:
            s.detail += f"; NOT validated: {', '.join(unsupported_units)}" + (" (see unit_probe)" if "unit_probe" in report else "")
        s.detail += f"; full Serve: {fp.get('status')} ({fp.get('seconds')} s, peak RSS {fp.get('maxrss_mb')} MB, cap {fp.get('cap_gb')} GB)"
        self.manifest["inputs"]["plan_digest"] = plan_digest
        self.manifest["inputs"]["plan_units"] = units

    def _validate_full_program(self, pdir: Path) -> dict[str, Any]:
        """RA's validator over the whole ``Serve`` in a memory-capped subprocess (``e2e/validate_full.py``); the
        outcome -- valid, invalid, MemoryError under the cap, killed, timeout -- is the finding, with its numbers."""
        a = self.args
        out = pdir / "full_program.json"
        if a.full_program == "skip":
            doc = {"attempted": False, "validated": False, "status": "skipped (--full-program skip)",
                   "reason": f"the units validated in this run are {list(a.plan_unit)} (plan/validator_report.json); the full program's digest is in program.json"}
            write_json(out, doc)
            return doc
        if out.exists():
            out.unlink()
        cmd = [sys.executable, "-m", "e2e.validate_full", "--case", a.case, "--workload", a.workload, "--out", str(out), "--cap-gb", str(a.full_program_gb)]
        t = time.perf_counter()
        log = self.out / "logs" / "validate-full.log"
        self.log(f"  $ {' '.join(cmd)}  > {log.relative_to(self.out)}  (timeout {a.full_program_timeout}s)")
        timed_out = False
        with open(log, "ab") as f:
            f.write(f"# {utc()} {' '.join(cmd)}\n".encode())
            f.flush()
            try:
                rc: int | None = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONPATH": f"{POC}:{ROOT}:{ROOT / 'src'}"}, cwd=str(POC), timeout=a.full_program_timeout)
            except subprocess.TimeoutExpired:
                rc, timed_out = None, True
        seconds = round(time.perf_counter() - t, 1)
        doc = json.loads(out.read_text()) if out.exists() else {}
        doc.update({"attempted": True, "exit_code": rc, "seconds": seconds, "cap_gb": a.full_program_gb, "log": self.rel(log)})
        if timed_out:
            doc["status"] = f"timeout after {a.full_program_timeout}s (killed; last phase reached: {doc.get('status')})"
            doc["validated"] = False
        elif rc in (2, 3) and doc.get("error"):
            doc["validated"] = False  # the child recorded its own failure (MemoryError under the cap / an exception): keep its status
        elif rc not in (0, 1) or not doc.get("phases"):
            doc["status"] = f"killed (exit {rc}) at phase {doc.get('status')}: the cap of {a.full_program_gb} GB or the pod's cgroup ended the process"
            doc["validated"] = False
        doc.setdefault("validated", False)
        doc["reason"] = ("RA's validator answered over the whole Serve program" if doc.get("validated") else
                         f"the whole Serve program was NOT validated ({doc.get('status')}); the units validated in this run are {list(a.plan_unit)} "
                         "(plan/validator_report.json, summary scope.plan_validation)")
        write_json(out, doc)
        self.log(f"  full program: {doc['status']} in {seconds}s, peak RSS {doc.get('maxrss_mb')} MB")
        return doc

    # -- stages 3-4: production and preparation -------------------------------------------------

    def stage_production(self, s: Stage) -> None:
        a = self.args
        prod = self.adapter / "production"
        if a.reuse_adapter:
            if not (prod / "production.jsonl").exists():
                raise FileNotFoundError(f"{prod}: no production.jsonl to reuse")
            s.status, s.detail = "ok", f"reused from {prod}"
        else:
            if prod.exists():
                shutil.rmtree(prod)
            cmd = [a.py_vllm, "-m", "verity_vllm_adapter.production", "--case", a.case, "--checkpoints", a.checkpoints, "--workload", a.workload,
                   "--out", str(prod), "--adapter-revision", a.adapter_revision, "--wheel-sha256", a.wheel_sha256, "--vllm-commit", a.vllm_commit, "--enforce-eager",
                   "--logprobs", "2"]  # NOTE 15:20 item 1: top-2 margins per emitted token need the runner-up logprob
            rc = self.subprocess("production", cmd, env=self.vllm_env("production"))
            if rc != 0 or not (prod / "production.jsonl").exists():
                s.status, s.detail = "FAIL", f"production exited {rc}"
                return
        manifest = json.loads((prod / "manifest.json").read_text())
        timings = json.loads((prod / "timings.json").read_text()) if (prod / "timings.json").exists() else {}
        s.artefacts = {"production": str(prod), "production_record": str(prod / "production.jsonl"), "manifest": str(prod / "manifest.json")}
        s.data = {"profile_id": manifest.get("profile_id") or (manifest.get("adapter") or {}).get("profile_id"), "requests": sum(1 for l in (prod / "production.jsonl").read_text().splitlines() if l.strip()),
                  "production_record_sha256": sha256_file(prod / "production.jsonl"), "overhead": timings.get("overhead")}
        s.detail = (s.detail + "; " if s.detail else "") + f"{s.data['requests']} requests; production.jsonl sha256 {s.data['production_record_sha256'][:16]}"
        self.manifest["inputs"]["production_record_sha256"] = s.data["production_record_sha256"]
        self.manifest["inputs"]["production_manifest"] = {k: manifest.get(k) for k in ("adapter", "vllm", "model", "compilation", "engine_flags", "dtype", "gpu", "python")}
        # integration-pad.md §2 item 5: T_native (the recording-off passes) is the denominator of the
        # headline cost metric and must not be separable from the committed record
        if (prod / "timings.json").exists():
            shutil.copyfile(prod / "timings.json", self.out / "production_timings.json")
            s.artefacts["production_timings"] = "production_timings.json"
            self.manifest["inputs"]["production_timings_sha256"] = sha256_file(self.out / "production_timings.json")
        # NOTE 15:20 item 2: the declared trusted-replay equivalence class -- what the replay is
        # declared exchangeable with (GPU model, kernel family = the semantic profile's kernels,
        # batch composition); the verifier checks the record against this declaration, not against
        # an unstated convention
        gpu = manifest.get("gpu") or {}
        self.manifest["inputs"]["trusted_replay_equivalence_class"] = {
            "gpu_model": gpu.get("name") or gpu,
            "kernel_family": s.data.get("profile_id"),
            "batch_composition": ("production: the workload as scheduled by the engine; selective replay: the r_q-selected requests J in their own batch; "
                                  "control replay: all requests in one batch -- exchangeability is TESTED by compare_selected (bit-identical RU leaves), not assumed"
                                  if self.selective else "production and full replay: the workload as scheduled"),
            "declared": "replays within this class are declared value-exchangeable under VLLM_BATCH_INVARIANT=1 (trusted-replay.md); outside it, nothing is claimed",
        }

    def stage_preparation(self, s: Stage) -> None:
        a = self.args
        prod = self.adapter / "production"
        commit = self.adapter / "commit"
        commit_v2 = self.adapter / "commit-v2"
        s.data["label"] = PREPARATION_LABEL
        if a.reuse_adapter:
            for d in (commit, commit_v2):
                if not (d / "commitments.json").exists():
                    raise FileNotFoundError(f"{d}: nothing to reuse")
            s.detail = f"{PREPARATION_LABEL}; reused from {self.adapter}"
        else:
            for mode in ("ref1", "ref2", "commit"):
                out = self.adapter / mode
                if out.exists():
                    shutil.rmtree(out)
                flags = ["--mode", "commit", "--weights-values", a.weights_values] if mode == "commit" else ["--mode", "reference"]
                cmd = [a.py_vllm, "-m", "verity_vllm_adapter.replay", "--case", a.case, "--checkpoints", a.checkpoints, "--production", str(prod), "--out", str(out),
                       *flags, "--adapter-revision", a.adapter_revision, "--arena-gb", str(a.arena_gb), "--replay-exec", "eager"]
                t = time.perf_counter()
                rc = self.subprocess(f"replay-{mode}", cmd, env=self.vllm_env(mode))
                s.data[f"{mode}_seconds"] = round(time.perf_counter() - t, 1)
                if rc != 0 or not (out / "commitments.json" if mode == "commit" else out / "capture_summary.json").exists():
                    s.status, s.detail = "FAIL", f"{mode} replay exited {rc}"
                    return
            cmd = [a.py_vllm, "-m", "verity_vllm_adapter.compare", "--commit", str(commit), "--ref", str(self.adapter / "ref1"), "--ref", str(self.adapter / "ref2"), "--out", str(self.adapter / "compare.json")]
            rc = self.subprocess("compare", cmd, env=self.vllm_env("compare"))
            if rc != 0:
                s.status, s.detail = "FAIL", f"compare exited {rc}"
                return
            if commit_v2.exists():
                shutil.rmtree(commit_v2)
            t = time.perf_counter()
            rc = self.subprocess("weights-v2", [sys.executable, "-m", "verity_vllm.weights_v2", str(commit), str(commit_v2), "--rows-per-leaf", "1"], env={"PYTHONPATH": f"{POC}:{ROOT / 'src'}"})
            s.data["weights_v2_seconds"] = round(time.perf_counter() - t, 1)
            if rc != 0:
                s.status, s.detail = "FAIL", f"weights v2 conversion exited {rc}"
                return
            s.detail = PREPARATION_LABEL
        compare = json.loads((self.adapter / "compare.json").read_text())
        commitments = json.loads((commit_v2 / "commitments.json").read_text())
        summary = json.loads((commit_v2 / "capture_summary.json").read_text()) if (commit_v2 / "capture_summary.json").exists() else {}
        conv = json.loads((commit_v2 / "weights" / "conversion.json").read_text()) if (commit_v2 / "weights" / "conversion.json").exists() else {}
        trees = commitments.get("trees") or {}
        s.artefacts = {"commit": str(commit), "commit_v2": str(commit_v2), "compare": str(self.adapter / "compare.json"), "commitments": str(commit_v2 / "commitments.json")}
        s.data.update({
            "noninterference": {k: compare.get(k) for k in ("identical", "compared_scalars", "mismatched_scalars")},
            "trees": len(trees), "leaves": sum(int(t.get("leaf_count", 0)) for t in trees.values()),
            "weights_root_v1": conv.get("v1_root") or conv.get("weights_root_v1"), "weights_root_v2": (trees.get("weights") or {}).get("root"),
            "boundary_roots": {k: v.get("root") for k, v in trees.items() if k.startswith("boundary/")},
            "bytes_captured": summary.get("bytes_captured"), "replay_units": summary.get("replay_units"), "steps": summary.get("steps"),
            "commitments_sha256": sha256_file(commit_v2 / "commitments.json"), "description_sha256": sha256_file(commit_v2 / "description.json"),
        })
        s.detail += f"; noninterference identical={compare.get('identical')} ({compare.get('compared_scalars')} scalars); {len(trees)} trees, {s.data['leaves']} leaves"
        self.manifest["inputs"]["commitments_sha256"] = s.data["commitments_sha256"]
        self.manifest["inputs"]["description_sha256"] = s.data["description_sha256"]
        if s.status == "not run" and not compare.get("identical", False):
            s.status, s.detail = "FAIL", "noninterference: commit and reference replays differ; " + s.detail

    # -- stages 4-5, selective (RC): prover and verifier as two processes, the handshake relayed ---------

    def stage_preparation_selective(self, s: Stage) -> None:
        """RC's protocol as two subprocesses.  The prover writes ``READY_PHASE0`` / ``READY_PHASE2`` and waits for
        ``replay_challenge.json``; the verifier the converse.  Each party gets its own handshake directory and this
        process relays the files -- the man-in-the-middle position a fault-injection test needs: a ``--negative``
        mutation is applied to the prover's finished artefact at its ``NEGATIVE_POINT`` before the verifier is told
        it is ready.  A full commit replay of every request (the *control*) is run first and RC's ``compare_selected``
        checks afterwards that every leaf J committed in its own batch is bit-identical to the control's (batch
        invariance tested, not assumed); the control is a check, not part of the audit's preparation."""
        a = self.args
        prod = self.adapter / "production"
        control = self.adapter / "control"
        sel = self.selective_dir
        s.data["label"] = SELECTIVE_LABEL
        cp, sprof = self._semantic_profile(prod)
        s.data["semantic_profile"] = {"path": str(sprof), "sha256": sha256_file(sprof)}
        self.manifest["inputs"]["semantic_profile"] = s.data["semantic_profile"]
        # milestone req. 4 (GUIDANCE §32.14 / NOTE 15:45): the pre-challenge sampling policy (q, s) is
        # bound BEFORE the protocol draws any challenge; the realized selections (challenge.json,
        # challenge_transcript.json) are recorded separately and are cross-checked against this
        # declaration by stage verify.  q for the fixed-size selector is declared as k/N.
        n_requests = sum(1 for l in (prod / "production.jsonl").read_text().splitlines() if l.strip())
        policy = {
            "schema": "verity-vllm/sampling-policy/v1", "bound_utc": utc(), "bound_before": "READY_PHASE0 (no challenge exists yet)",
            "selector": ("bernoulli" if a.selective_q else "fixed_size"),
            "q_declared": (a.selective_q if a.selective_q else f"{a.k}/{n_requests}"), "k": (None if a.selective_q else a.k),
            "requests_total": n_requests, "s_declared": a.s, "coverage_size": a.coverage_size, "challenge_mode": a.challenge,
            "note": "realized selections (r_q, r_s, the drawn J and VU sample) live in verify/challenge.json + challenge_transcript.json and are never substituted for these declared values",
        }
        write_json(self.out / "sampling_policy.json", policy)
        self.manifest["inputs"]["sampling_policy_sha256"] = sha256_file(self.out / "sampling_policy.json")
        s.artefacts["sampling_policy"] = "sampling_policy.json"
        # [REC 2026-09-09, GAP P2-0 (a)] the pre-challenge protocol instance (agreement 1) and its evidence are copied into the record and
        # digest-pinned in the manifest HERE -- before READY_PHASE0, i.e. before any challenge exists -- and cross-checked against this
        # run's program digest and declared law.  A mismatch is a preparation failure (nothing is drawn).
        if a.instance_dir:
            self._pin_instance(Path(a.instance_dir), policy, s)
        self.commit_dir = sel  # the selective bundle is what the obligations stage and the bundle read (a negative mutates it in place, at its relay point)
        # (a) the control replay (full, every request; RC's batch-invariance reference)
        if a.control == "full":
            if a.reuse_adapter and (control / "commitments.json").exists():
                s.data["control"] = {"reused_from": str(control)}
            else:
                if control.exists():
                    shutil.rmtree(control)
                cmd = [a.py_vllm, "-m", "verity_vllm_adapter.replay", "--case", a.case, "--checkpoints", a.checkpoints, "--production", str(prod), "--out", str(control),
                       "--mode", "commit", "--weights-values", "hash", "--adapter-revision", a.adapter_revision, "--arena-gb", str(a.arena_gb), "--replay-exec", "eager"]
                t = time.perf_counter()
                rc = self.subprocess("replay-control", cmd, env=self.vllm_env("control"))
                s.data["control"] = {"seconds": round(time.perf_counter() - t, 1), "dir": str(control)}
                if rc != 0 or not (control / "commitments.json").exists():
                    s.status, s.detail = "FAIL", f"control replay exited {rc}"
                    return
        else:
            s.data["control"] = {"skipped": "--control none: batch invariance of the selective replay NOT tested this run"}
        # (b) the protocol: prover and verifier concurrently, handshake relayed (and mutated, for a negative)
        if sel.exists():
            shutil.rmtree(sel)
        hs_p, hs_v = self.out / "handshake" / "prover", self.out / "handshake" / "verifier"
        for d in (hs_p, hs_v):
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)
        vdir = self.out / "verify"
        if vdir.exists():
            shutil.rmtree(vdir)
        vdir.mkdir()
        prover_cmd = [a.py_vllm, "-m", "verity_vllm_adapter.selective_prover", "--case", a.case, "--checkpoints", a.checkpoints, "--production", str(prod), "--out", str(sel),
                      "--handshake", str(hs_p), "--adapter-revision", a.adapter_revision, "--semantic-profile", str(sprof), "--arena-gb", str(a.arena_gb),
                      "--weights-format", "2", "--rows-per-leaf", "1"]
        verifier_cmd = [a.py_vllm, "-m", "verity_vllm_adapter.selective_verifier", "--run", str(sel), "--handshake", str(hs_v), "--production", str(prod), "--checkpoints", a.checkpoints,
                        "--case", a.case, "--semantic-profile", str(sprof), "--s", a.s, "--coverage-size", str(a.coverage_size), "--challenge", a.challenge, "--out", str(vdir),
                        "--adapter-revision", a.adapter_revision]
        verifier_cmd += ["--q", a.selective_q] if a.selective_q else ["--k", str(a.k)]
        if a.challenge == "diagnostic":
            verifier_cmd += ["--seed", str(a.seed)]
        t_protocol = time.perf_counter()
        prover, pf = self.popen("selective-prover", prover_cmd, env=self.vllm_env("selective-prover"))
        verifier, vf = self.popen("selective-verifier", verifier_cmd, env=self.vllm_env("selective-verifier"))
        try:
            events = self._relay(hs_p, hs_v, prover, verifier)
        finally:
            for p in (prover, verifier):
                if p.poll() is None:
                    p.kill()
            pf.close()
            vf.close()
        protocol_wall = time.perf_counter() - t_protocol
        relayed_abort = bool(self.protocol.get("relayed_abort_to_prover"))  # set by _relay: the prover's ABORT came from the verifier / from us, not from itself
        self.protocol = {"prover_rc": prover.returncode, "verifier_rc": verifier.returncode, "events": events, "wall_seconds": round(protocol_wall, 1),
                         "handshake": {"prover": str(hs_p), "verifier": str(hs_v)}, "prover_log": self.rel(self.out / "logs" / "selective-prover.log"),
                         "verifier_log": self.rel(self.out / "logs" / "selective-verifier.log"), "relayed_abort_to_prover": relayed_abort}
        for side, hs in (("prover", hs_p), ("verifier", hs_v)):
            if (hs / HS_ABORT).exists():
                self.protocol[f"abort_seen_by_{side}"] = (hs / HS_ABORT).read_text().strip()[:400]
        names = [e["file"] for e in events if e.get("relayed")]
        reached0, reached2 = HS_READY0 in names, HS_READY2 in names
        pj = json.loads((sel / "selective_prover.json").read_text()) if (sel / "selective_prover.json").exists() else None
        p0 = json.loads((sel / "phase0.json").read_text()) if (sel / "phase0.json").exists() else None
        s.artefacts = {"selective_bundle": str(sel), "prover_json": str(sel / "selective_prover.json"), "phase0_json": str(sel / "phase0.json"), "control": str(control),
                       "prover_log": self.protocol["prover_log"], "handshake_events": self.rel(self.out / "handshake" / "events.json")}
        write_json(self.out / "handshake" / "events.json", self.protocol)
        s.data.update({"protocol": {k: v for k, v in self.protocol.items() if k != "events"}, "handshake_events": events, "reached": {"READY_PHASE0": reached0, "READY_PHASE2": reached2}})
        if p0:
            s.data["phase0"] = {k: p0.get(k) for k in ("requests", "boundary_trees", "boundary_leaves", "weights", "description_digest", "required_interfaces", "replay_units_committed", "commitments_phase0_sha256")}
            s.data["phase0"]["timing"] = p0.get("timing")
        # (c) what the prover did: replayed fraction (RC's measured q), bytes, batch, followed trajectories
        if pj:
            total_positions = _positions_processed(prod / "production.jsonl")
            rt = pj["phase2"]["replayed_tokens"]
            s.data.update({
                "k": pj["k"], "requests_total": pj["requests_total"], "selected": pj["selected"], "selector": pj.get("challenge_file"),
                "q_measured": {"positions_replayed": rt["positions_processed"], "positions_total": total_positions,
                               "fraction": round(rt["positions_processed"] / total_positions, 6) if total_positions else None,
                               "requests_fraction": f"{pj['k']}/{pj['requests_total']}",
                               "note": "measured: positions of the r_q-selected requests / positions of the production workload (RC TABLE 1's denominator); not an estimate"},
                "replayed_tokens": rt, "trajectory_followed": pj["phase2"]["trajectory_followed"], "unforced_sampler_agreement": pj["phase2"]["unforced_sampler_agreement"],
                "capture": pj["phase2"]["capture"], "bytes_captured": (pj["phase2"]["capture"] or {}).get("bytes_captured"),
                "bytes": pj["bytes"], "timing": pj["timing"], "replay_units_committed": pj["phase2"]["replay_units_committed"],
                "ru_leaves": pj["phase2"]["ru_leaves"], "architecture_observed_vs_registered": (pj["phase2"].get("architecture_observed_vs_registered") or {}).get("ok"),
                "effective_execution": pj.get("effective_execution"), "note_batch": pj.get("note_batch"),
            })
        if (sel / "commitments.json").exists():
            commitments = json.loads((sel / "commitments.json").read_text())
            trees = commitments.get("trees") or {}
            s.data.update({"trees": len(trees), "leaves": sum(int(t.get("leaf_count", 0)) for t in trees.values()),
                           "weights_root_v2": (trees.get("weights") or {}).get("root"),
                           "boundary_roots": {k: v.get("root") for k, v in trees.items() if k.startswith("boundary/")},
                           "commitments_sha256": sha256_file(sel / "commitments.json"), "description_sha256": sha256_file(sel / "description.json")})
            self.manifest["inputs"]["commitments_sha256"] = s.data["commitments_sha256"]
            self.manifest["inputs"]["description_sha256"] = s.data["description_sha256"]
        # (d) batch invariance: the selective bundle's RU leaves vs the control's (a negative's bundle is mutated -- weights or description --
        # but its RU leaves are the prover's own, so the comparison is still meaningful and still expected bit-identical)
        if a.control == "full" and pj and (sel / "commitments.json").exists():
            out = self.out / "adapter" / "compare_selected.json"
            rc = self.subprocess("compare-selected", [a.py_vllm, "-m", "verity_vllm_adapter.compare_selected", "--selective", str(sel), "--control", str(control), "--out", str(out)],
                                 env=self.vllm_env("compare"))
            cs = json.loads(out.read_text()) if out.exists() else {}
            s.artefacts["compare_selected"] = self.rel(out)
            s.data["batch_invariance"] = {k: cs.get(k) for k in ("bit_identical", "leaves_both", "leaves_only_selective", "compared_scalars", "mismatched_leaves", "mismatched_scalars", "first_divergent_ids", "method")}
            s.data["batch_invariance"]["control"] = str(control)
            if rc != 0 or not cs.get("bit_identical"):
                s.status, s.detail = "FAIL", f"batch invariance: selective replay of J is NOT bit-identical to the full-replay control ({cs.get('mismatched_leaves')} leaves / {cs.get('mismatched_scalars')} scalars differ)"
                return
        # status: the prover did its part when phase 0 is complete and either phase 2 completed or the protocol was ended by the
        # verifier's refusal (a negative's intended outcome); the prover's own abort or crash is a preparation failure
        prover_aborted_itself = (hs_p / HS_ABORT).exists() and not self.protocol.get("relayed_abort_to_prover")
        if not reached0 or prover_aborted_itself or (prover.returncode != 0 and reached2):
            s.status = "FAIL"
            s.detail = f"prover exited {prover.returncode} (READY_PHASE0 {reached0}, READY_PHASE2 {reached2}); {self.protocol.get('abort_seen_by_prover') or ''}"[:300]
            return
        qm = s.data.get("q_measured") or {}
        s.detail = (f"{SELECTIVE_LABEL.split(':')[0]}: k={s.data.get('k')}/{s.data.get('requests_total')} requests replayed"
                    + (f", {qm.get('positions_replayed')}/{qm.get('positions_total')} positions (q measured {qm.get('fraction')})" if qm else "")
                    + (f"; {s.data.get('trees')} trees, {s.data.get('leaves')} leaves" if "trees" in s.data else "")
                    + (f"; batch invariance vs control: bit_identical={s.data['batch_invariance'].get('bit_identical')} over {s.data['batch_invariance'].get('compared_scalars')} scalars" if "batch_invariance" in s.data else "")
                    + ("" if reached2 else f"; protocol ended by the verifier before phase 2 ({self.protocol.get('abort_seen_by_prover', '')[:120]})"))

    def _pin_instance(self, idir: Path, policy: dict[str, Any], s: Stage) -> None:
        """[REC] Copy the pre-challenge protocol instance (``protocol_instance.json``) and its evidence
        (``population.json``, ``compact_validation.json``, optional ``maps.json``, ``weights_authorized.json``) into the
        record root and pin their digests in ``manifest.inputs`` BEFORE the protocol starts (no challenge exists yet).
        Cross-checks: the instance's program digest is this run's; its law is the one declared in ``sampling_policy.json``;
        its status is ``pre-challenge``.  Any mismatch raises (fatal for the preparation stage)."""
        from e2e import protocol_instance as pi

        files = {"protocol_instance.json": True, "population.json": True, "compact_validation.json": True, "maps.json": False, "weights_authorized.json": False}
        pinned: dict[str, str] = {}
        for name, required in files.items():
            src = idir / name
            if not src.exists():
                if required:
                    raise FileNotFoundError(f"--instance-dir {idir}: {name} missing")
                continue
            shutil.copyfile(src, self.out / name)
            pinned[name] = sha256_file(self.out / name)
        inst = pi.load_instance(self.out / "protocol_instance.json")
        problems = pi.validate_instance(inst)
        if problems:
            raise ValueError(f"protocol_instance.json invalid: {problems[:3]}")
        if inst.status != "pre-challenge":
            raise ValueError(f"protocol_instance.json status {inst.status!r}: only a pre-challenge instance can be pinned into a new record")
        pd = self.manifest["inputs"].get("program_digest")
        if inst.program.digest != pd:
            raise ValueError(f"protocol_instance.json program_digest {inst.program.digest[:16]} != this run's {str(pd)[:16]}")
        law = inst.sampler.law
        declared_q = policy["q_declared"]
        if policy["selector"] != "bernoulli" or law.kind != "bernoulli":
            raise ValueError(f"instance law {law.kind!r} vs declared selector {policy['selector']!r}: only Bernoulli x Bernoulli is realised by this recorder")
        if Fraction(str(law.q)) != Fraction(str(declared_q)) or Fraction(str(law.s)) != Fraction(str(policy["s_declared"])):
            raise ValueError(f"instance law q={law.q} s={law.s} != declared q={declared_q} s={policy['s_declared']}")
        pop = json.loads((self.out / "population.json").read_text())
        if pop.get("partition_digest") != inst.partition.partition_digest:
            raise ValueError("population.json partition_digest != instance partition_digest")
        self.manifest["inputs"]["protocol_instance"] = {
            "pinned_utc": utc(), "pinned_before": "READY_PHASE0 (no challenge exists yet)", "files_sha256": pinned,
            "instance_digest": pi.instance_digest(inst), "program_digest": inst.program.digest,
            "partition": {"plan": inst.partition.plan, "partition_digest": inst.partition.partition_digest, "plan_descriptor_digest": inst.partition.plan_descriptor_digest},
            "law": {"kind": law.kind, "q": str(law.q), "s": str(law.s)},
            "first_boundary": {"rule": inst.partition.first_boundary.rule, "committed_in": inst.partition.first_boundary.committed_in},
            "acquisition_granularity": inst.partition.acquisition_granularity,
        }
        self.manifest["inputs"]["protocol_instance_sha256"] = pinned["protocol_instance.json"]  # the pin e2e.instance_check reads
        for name in ("population.json", "compact_validation.json", "weights_authorized.json"):
            if name in pinned:
                self.manifest["inputs"][name.replace(".json", "_sha256")] = pinned[name]
        s.data["protocol_instance"] = self.manifest["inputs"]["protocol_instance"]
        s.artefacts["protocol_instance"] = "protocol_instance.json"
        self.log(f"  protocol instance pinned before READY_PHASE0: {pi.instance_digest(inst)[:16]} ({inst.partition.plan}, law {law.kind} q={law.q} s={law.s})")

    def _semantic_profile(self, prod: Path) -> tuple[dict[str, Any], Path]:
        """The one shipped semantic profile for the case's checkpoint architecture and the production manifest
        (``verify.py --semantic-profile auto``'s rule, ``binding.select_semantic_profile``)."""
        from verity_vllm_adapter import binding as vbinding
        from verity_vllm_adapter import profile as vprofile

        a = self.args
        cp = vprofile.checkpoint_for(a.case, a.checkpoints)
        if a.semantic_profile != "auto":
            return cp, Path(a.semantic_profile).resolve()
        config = json.loads((Path(cp["local_path"]) / "config.json").read_text())
        manifest = json.loads((prod / "manifest.json").read_text())
        return cp, Path(vbinding.select_semantic_profile(config, manifest)).resolve()

    def _relay(self, hs_p: Path, hs_v: Path, prover: subprocess.Popen, verifier: subprocess.Popen) -> list[dict[str, Any]]:
        """Relay the handshake files between the two parties' directories; apply the negative's mutation at its point."""
        a = self.args
        events: list[dict[str, Any]] = []
        relayed: set[str] = set()
        routes = [(HS_READY0, hs_p, hs_v), (HS_CHALLENGE, hs_v, hs_p), (HS_READY2, hs_p, hs_v)]

        def copy(name: str, src: Path, dst: Path) -> None:
            data = (src / name).read_bytes()
            tmp = dst / (name + ".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, dst / name)

        deadline = time.time() + a.protocol_timeout
        while True:
            for name, src, dst in routes:
                if name in relayed or not (src / name).exists():
                    continue
                if name == HS_READY2 and prover.poll() is None:
                    # the prover exits right after writing READY_PHASE2; wait for it so the bundle is final before anyone reads it
                    try:
                        prover.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        pass
                if a.negative and NEGATIVE_POINT[a.negative] == name:
                    ms = self.stage("mutation")
                    t = time.perf_counter()
                    try:
                        self.apply_mutation(self.selective_dir, ms)
                        ms.status = "ok"
                        ms.data["applied_at"] = f"handshake {name} (after the prover finalised the artefact, before the verifier read it)"
                    except Exception as e:  # noqa: BLE001
                        ms.status, ms.detail = "error", f"{type(e).__name__}: {e}"
                    ms.seconds = time.perf_counter() - t
                    self.log(f"stage mutation: {ms.status} at {name}: {ms.detail}")
                    events.append({"utc": utc(), "file": name, "mutation": a.negative, "status": ms.status, "detail": ms.detail})
                copy(name, src, dst)
                relayed.add(name)
                events.append({"utc": utc(), "file": name, "from": src.name, "to": dst.name, "relayed": True})
                self.log(f"  relay {name}: {src.name} -> {dst.name}")
            for src, dst, tag in ((hs_p, hs_v, "prover->verifier"), (hs_v, hs_p, "verifier->prover")):
                if (src / HS_ABORT).exists() and not (dst / HS_ABORT).exists():
                    copy(HS_ABORT, src, dst)
                    if tag == "verifier->prover":
                        self.protocol["relayed_abort_to_prover"] = True
                    events.append({"utc": utc(), "file": HS_ABORT, "direction": tag, "reason": (src / HS_ABORT).read_text().strip()[:300], "relayed": True})
                    self.log(f"  relay ABORT {tag}: {(src / HS_ABORT).read_text().strip()[:200]}")
            prc, vrc = prover.poll(), verifier.poll()
            if vrc is not None and prc is None and HS_READY2 not in relayed and not (hs_p / HS_ABORT).exists():
                (hs_p / HS_ABORT).write_text(f"verifier exited {vrc} before the protocol completed (see logs/selective-verifier.log)\n")
                self.protocol["relayed_abort_to_prover"] = True
                events.append({"utc": utc(), "file": HS_ABORT, "direction": "run_composed->prover", "reason": f"verifier exited {vrc}"})
                self.log(f"  verifier exited {vrc} before READY_PHASE2: aborting the prover")
            if prc is not None and prc != 0 and vrc is None and not (hs_v / HS_ABORT).exists():
                (hs_v / HS_ABORT).write_text(f"prover exited {prc} (see logs/selective-prover.log)\n")
                events.append({"utc": utc(), "file": HS_ABORT, "direction": "run_composed->verifier", "reason": f"prover exited {prc}"})
                self.log(f"  prover exited {prc}: aborting the verifier")
            if prc is not None and vrc is not None:
                return events
            if time.time() > deadline:
                events.append({"utc": utc(), "timeout": a.protocol_timeout})
                self.log("  protocol timeout: killing both parties")
                return events
            time.sleep(0.5)

    def stage_verify_selective(self, s: Stage) -> None:
        """The verifier's row of the selective protocol (``selective_verify.json``; RC's verdict = every checked VU exact,
        every opening authenticated, leaf predicate over J, output linkage of J).  For a refused protocol the first
        blocker is read from the verifier's abort reason or its log."""
        vdir = self.out / "verify"
        sj = vdir / "selective_verify.json"
        s.artefacts = {"verify_dir": self.rel(vdir), "selective_verify": self.rel(sj), "challenge": self.rel(vdir / "challenge.json"),
                       "transcript": self.rel(vdir / "challenge_transcript.json"), "results": self.rel(vdir / "results.jsonl"), "statement": self.rel(vdir / "statement.json"),
                       "verifier_log": self.protocol.get("verifier_log")}
        rc = self.protocol.get("verifier_rc")
        s.data = {"exit_code": rc, "challenge_mode": self.args.challenge, "protocol": "selective (RC decision 3)"}
        if not sj.exists():
            reason = self.protocol.get("abort_seen_by_verifier") or self.protocol.get("abort_seen_by_prover") or ""
            log_tail = ""
            lp = self.out / "logs" / "selective-verifier.log"
            if lp.exists():
                lines = lp.read_text(errors="replace").splitlines()
                log_tail = "\n".join(lines[-12:])
                err = next((l for l in reversed(lines) if "Error" in l or "ABORT" in l), "")
            else:
                err = ""
            s.data.update({"result": "FAIL", "first_blocker": _classify_refusal(reason, err), "abort_reason": reason[:400], "verifier_error": err[:400], "log_tail": log_tail})
            s.status, s.detail = "FAIL", f"verifier exited {rc} without a row; first_blocker={s.data['first_blocker']}; {err[:160]}"
            return
        doc = json.loads(sj.read_text())
        check = doc.get("check") or {}
        statuses = check.get("statuses") or {}
        fams = check.get("families") or {}
        rel: dict[str, dict[str, int]] = {}
        for _op, by_klass in fams.items():
            for klass, fr in (by_klass or {}).items():
                d = rel.setdefault(klass, {"attempted": 0, "exact": 0})
                d["attempted"] += int(fr.get("attempted", 0))
                d["exact"] += int(fr.get("exact", 0))
        linkage_problems = [p for l in (doc.get("output_linkage") or []) for p in (l.get("problems") or [])]
        blocker = None
        if doc.get("verdict") != "PASS":
            if statuses.get("auth_failed"):
                blocker = f"authentication ({statuses['auth_failed']} VU(s) with an opening that did not authenticate)"
            elif statuses.get("mismatch"):
                blocker = f"relation ({statuses['mismatch']} VU(s) mismatched)"
            elif not (doc.get("leaf_predicate") or {}).get("ok", True):
                blocker = "structural: leaf predicate over J"
            elif linkage_problems:
                blocker = f"linkage: {linkage_problems[0]}"
            else:
                blocker = "verifier FAIL: " + ", ".join(f"{k}={v}" for k, v in statuses.items() if k != "exact")
        s.data.update({"result": doc.get("verdict"), "first_blocker": blocker, "authentication": check.get("authentication"), "statuses": statuses,
                       "relation_checks": rel, "bindings": (doc.get("bindings") or {}).get("ok"), "bindings_detail": doc.get("bindings"),
                       "certified_sample": doc.get("certified_sample"), "coverage_sample": doc.get("coverage_sample"), "adjacent": doc.get("adjacent"), "sample_union": doc.get("sample_union"),
                       "population": doc.get("population"), "selected": doc.get("selected"), "k": doc.get("k"), "requests_total": doc.get("requests_total"),
                       "first_stage": doc.get("first_stage"), "leaf_predicate": doc.get("leaf_predicate"), "output_linkage_ok": not linkage_problems,
                       "witness": doc.get("witness"), "timing": doc.get("timing"), "challenge": doc.get("challenge"), "relations_installed": doc.get("relations_installed")})
        s.data["population_counts"] = {"certified": doc.get("certified_sample"), "coverage": doc.get("coverage_sample"), "adjacent": doc.get("adjacent"), "union": doc.get("sample_union")}
        if (vdir / "challenge.json").exists():
            ch = json.loads((vdir / "challenge.json").read_text())
            s.data["challenge_digest"] = ch.get("digest")
            self.manifest["inputs"]["challenge_digest"] = ch.get("digest")
            chal = doc.get("challenge") or {}
            self.manifest["inputs"]["challenge"] = {"mode": chal.get("mode"), "policy": chal.get("policy"), "selector": chal.get("selector"), "r_q": chal.get("r_q"), "r_s": chal.get("r_s"),
                                                    "phase_digests": chal.get("phase_digests"), "seed": self.args.seed if self.args.challenge == "diagnostic" else None,
                                                    "protocol": "selective (T_0 declares no interior; r_q selects requests; T_2 over J's interiors)"}
            if (vdir / "challenge_transcript.json").exists():
                tr = json.loads((vdir / "challenge_transcript.json").read_text())
                self.manifest["inputs"]["challenge"]["seed_commitment"] = (tr.get("source") or {}).get("seed_commitment")
        s.status = "ok" if doc.get("verdict") == "PASS" and rc == 0 else "FAIL"
        s.detail = (f"row {doc.get('verdict')} first_blocker={blocker} auth={check.get('authentication')} statuses={statuses} "
                    f"(certified {doc.get('certified_sample')} + coverage {doc.get('coverage_sample')} = {doc.get('sample_union')} VUs over J={doc.get('k')}/{doc.get('requests_total')}); "
                    f"work {round((doc.get('timing') or {}).get('work_total_s', 0), 1)} s (check {round((doc.get('timing') or {}).get('check_s', 0), 1)} s)")
        # milestone req. 4: the realized selections must match the pre-challenge declaration bit for bit
        # (k, population, mode) -- realized values recorded beside, never substituted for the declaration
        pfile = self.out / "sampling_policy.json"
        if pfile.exists():
            pol = json.loads(pfile.read_text())
            problems = []
            if pol["selector"] == "fixed_size" and doc.get("k") != pol.get("k"):
                problems.append(f"realized |J| = {doc.get('k')} != declared k = {pol.get('k')}")
            if doc.get("requests_total") not in (None, pol.get("requests_total")):
                problems.append(f"population {doc.get('requests_total')} != declared {pol.get('requests_total')}")
            if s.data.get("challenge_mode") != pol.get("challenge_mode"):
                problems.append(f"challenge mode {s.data.get('challenge_mode')} != declared {pol.get('challenge_mode')}")
            s.data["sampling_policy_check"] = {"declared_sha256": self.manifest["inputs"].get("sampling_policy_sha256"),
                                               "declared": {k: pol.get(k) for k in ("selector", "q_declared", "k", "requests_total", "s_declared", "coverage_size", "challenge_mode")},
                                               "realized": {"k": doc.get("k"), "requests_total": doc.get("requests_total"),
                                                            "r_q": (self.manifest["inputs"].get("challenge") or {}).get("r_q"),
                                                            "r_s": (self.manifest["inputs"].get("challenge") or {}).get("r_s")},
                                               "problems": problems}
            if problems and s.status == "ok":
                s.status, s.detail = "FAIL", "sampling policy violated: " + "; ".join(problems) + "; " + s.detail

    # -- negative injection (between preparation and verify) --------------------------------------

    def apply_mutation(self, dst: Path, s: Stage) -> None:
        """WA's labelled mutation applied in place to the bundle at ``dst`` (a copy, or a negative run's own selective bundle)."""
        a = self.args
        if a.negative == "wa-m12":
            wdir = dst / "weights"
            index = json.loads((wdir / "index.json").read_text())
            entry = next(e for e in index["entries"] if "mlp.down_proj" in e["param"] and e.get("offset") is not None)
            pos = int(entry["offset"]) + 1
            with open(wdir / "values.bin", "r+b") as f:
                f.seek(pos)
                b = f.read(1)
                f.seek(pos)
                f.write(bytes([b[0] ^ 1]))
            s.detail = f"{NEGATIVES[a.negative]}: {entry['param']} byte {pos}"
        elif a.negative == "wa-m8b":
            d = json.loads((dst / "description.json").read_text())
            layers = sorted({str(n.get("layer")) for n in d["nodes"] if str(n.get("layer")).isdigit()}, key=int)  # decoder layers only ("lm_head", "embed" are not layers)
            victim = layers[-1]
            d["nodes"] = [n for n in d["nodes"] if n.get("layer") != victim]
            if (dst / "description.json").is_symlink() or os.stat(dst / "description.json").st_nlink > 1:
                os.unlink(dst / "description.json")
            (dst / "description.json").write_text(json.dumps(d))
            s.detail = f"{NEGATIVES[a.negative]}: layer {victim}"
        else:
            raise ValueError(a.negative)
        s.artefacts = {"mutated_bundle": str(dst)}
        s.data = {"mutation": a.negative, "label": NEGATIVES[a.negative], "source": "docs/verified-backend/audit-security-baseline.md", "target": str(dst)}

    def stage_mutation_selective_record(self, s: Stage) -> None:
        """Selective mode: the mutation was applied inside the protocol (``_relay``) at the negative's point; this stage
        only records it (the Stage object was filled by ``apply_mutation`` at that moment)."""
        ms = self.stage("mutation")
        s.data.setdefault("mutation", self.args.negative)
        s.data["point"] = NEGATIVE_POINT[self.args.negative]
        s.data["applied_seconds"] = round(ms.seconds, 3)
        if not ms.data.get("applied_at"):
            s.status, s.detail = "FAIL", f"mutation point {NEGATIVE_POINT[self.args.negative]} never reached: the protocol ended before it (nothing was mutated)"
        # else: status/detail/artefacts already set by apply_mutation via _relay

    def stage_mutation(self, s: Stage) -> None:
        src = self.adapter / "commit-v2"
        dst = self.out / "adapter-mutated" / "commit-v2"
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, copy_function=shutil.copy2)  # a real copy: the mutation must not touch the positive run's bundle
        self.apply_mutation(dst, s)
        self.commit_dir = dst

    # -- stage 5: verify --------------------------------------------------------------------------

    def stage_verify(self, s: Stage) -> None:
        a = self.args
        vdir = self.out / "verify"
        if vdir.exists():
            shutil.rmtree(vdir)
        cmd = [a.py_vllm, "-m", "verity_vllm_adapter.verify", str(self.commit_dir), "--challenge", a.challenge, "--q", a.q, "--s", a.s,
               "--coverage-size", str(a.coverage_size), "--negative-limit", "4", "--production", str(self.adapter / "production"), "--checkpoints", a.checkpoints,
               "--noninterference", str(self.adapter / "compare.json"), "--adapter-revision", a.adapter_revision, "--case", a.case, "--worker", a.worker, "--out", str(vdir)]
        if a.challenge == "diagnostic":
            cmd += ["--seed", str(a.seed)]
        rc = self.subprocess("verify", cmd, env=self.vllm_env("verify"))
        s.artefacts = {"verify_dir": self.rel(vdir), "verify_json": self.rel(vdir / "verify.json"), "table1": self.rel(vdir / "table1.jsonl"),
                       "challenge": self.rel(vdir / "challenge.json"), "transcript": self.rel(vdir / "challenge_transcript.json")}
        row = None
        if (vdir / "table1.jsonl").exists():
            rows = [json.loads(l) for l in (vdir / "table1.jsonl").read_text().splitlines() if l.strip()]
            row = rows[-1] if rows else None
        if row is None:
            s.status, s.detail = "FAIL", f"verifier exited {rc} without a Table 1 row"
            return
        verify_json = json.loads((vdir / "verify.json").read_text()) if (vdir / "verify.json").exists() else {}
        s.data = {"result": row.get("result"), "first_blocker": row.get("first_blocker"), "verdict_label": verify_json.get("verdict_label"),
                  "authentication": row.get("authentication"), "relation_checks": row.get("relation_checks"), "coverage": row.get("coverage"),
                  "negative": row.get("negative"), "structural": row.get("structural"), "qualifications": row.get("qualifications"), "exit_code": rc,
                  "challenge_mode": a.challenge, "bindings": verify_json.get("bindings", {}).get("ok") if isinstance(verify_json.get("bindings"), dict) else None}
        if (vdir / "challenge.json").exists():
            ch = json.loads((vdir / "challenge.json").read_text())
            s.data["challenge_digest"] = ch.get("digest")
            s.data["population"] = {"certified": len(ch.get("certified_sample") or []), "coverage": len(ch.get("coverage_sample") or []), "adjacent": len(ch.get("adjacent_vus") or []), "union": len(ch.get("sample") or [])}
            self.manifest["inputs"]["challenge_digest"] = ch.get("digest")
            chal = ch.get("challenge") or {}
            self.manifest["inputs"]["challenge"] = {"mode": chal.get("mode"), "source": chal.get("source"), "r_q": chal.get("r_q"), "r_s": chal.get("r_s"), "phase_digests": chal.get("phase_digests"), "seed": a.seed if a.challenge == "diagnostic" else None}
            if (vdir / "challenge_transcript.json").exists():
                tr = json.loads((vdir / "challenge_transcript.json").read_text())
                src = tr.get("source") or {}
                self.manifest["inputs"]["challenge"]["seed_commitment"] = src.get("seed_commitment") or src.get("commitment") or {k: v for k, v in src.items() if "commit" in k}
        s.status = "ok" if row.get("result") == "PASS" else "FAIL"
        s.detail = f"row {row.get('result')} first_blocker={row.get('first_blocker')} auth={row.get('authentication')} relations={row.get('relation_checks')}"

    # -- stage 5b: served-token correspondence ------------------------------------------------------

    def stage_correspondence(self, s: Stage) -> None:
        """NOTE 15:20 item 4 / milestone req. 5: the served token ids (what production answered the
        client, ``production.jsonl``) against the committed boundary trees' ``output_token`` records,
        for EVERY request -- not only J.  ``served_tokens.json`` becomes part of the committed record;
        a mismatch is a FAIL (the record would be attesting a different serving than the one served)."""
        from verity_vllm.capture import CaptureReader

        reader = CaptureReader(self.commit_dir)
        prod_rows = [json.loads(l) for l in (self.adapter / "production" / "production.jsonl").read_text().splitlines() if l.strip()]
        doc: dict[str, Any] = {"schema": "verity-vllm/served-tokens/v1", "utc": utc(), "requests": []}
        mismatches: list[str] = []
        for row in prod_rows:
            rid = row["request_id"]
            served = list(row.get("output_token_ids") or [])
            entry: dict[str, Any] = {"request_id": rid, "prompt_len": row.get("prompt_len"), "served_token_ids": served,
                                     "finish_reason": row.get("finish_reason"), "top2_margins": row.get("top2_margins"),
                                     "chosen_logprobs": row.get("chosen_logprobs")}
            if reader.has_boundary(rid):
                b = reader.boundary(rid)
                committed = [r.token_id for r in b.output_records if r.present]
                entry["committed_output_token_ids"] = committed
                entry["prompt_match"] = list(b.prompt_token_ids) == list(row.get("prompt_token_ids") or [])
                entry["output_match"] = committed == served
                if not entry["output_match"]:
                    mismatches.append(f"{rid}: served {served[:4]}... != committed {committed[:4]}...")
                if not entry["prompt_match"]:
                    mismatches.append(f"{rid}: prompt tokens differ")
            else:
                entry["committed_output_token_ids"] = None
                mismatches.append(f"{rid}: no committed boundary")
            doc["requests"].append(entry)
        doc["mismatches"] = mismatches
        doc["ok"] = not mismatches
        write_json(self.out / "served_tokens.json", doc)
        self.manifest["inputs"]["served_tokens_sha256"] = sha256_file(self.out / "served_tokens.json")
        s.artefacts = {"served_tokens": "served_tokens.json"}
        margins = [m for e in doc["requests"] for m in (e.get("top2_margins") or []) if m is not None]
        s.data = {"requests": len(doc["requests"]), "ok": doc["ok"], "mismatches": mismatches[:4],
                  "top2_margins": {"count": len(margins), "min": min(margins) if margins else None,
                                   "note": "chosen - best other logprob per emitted token (production --logprobs 2); a margin near 0 flags a knife-edge argmax (trusted-replay.md complaint 2)"}}
        s.status = "ok" if doc["ok"] else "FAIL"
        s.detail = (f"{len(doc['requests'])} requests: served token ids == committed boundary output tokens" if doc["ok"]
                    else f"{len(mismatches)} mismatch(es): " + "; ".join(mismatches[:3])) + (f"; min top-2 margin {min(margins):.4g}" if margins else "; no margins recorded")

    # -- stage 6: obligations ---------------------------------------------------------------------

    def stage_obligations(self, s: Stage) -> None:
        import numpy as np  # noqa: F401

        from verity_vllm.capture import CaptureReader
        from verity_vllm.commit import TreeCache, load_commitments
        from verity_vllm.description import Description
        from verity_vllm_adapter import obligations as O
        from verity_vllm_adapter import profile as vprofile
        from veritor.interface import TypedObligationSet, to_statement

        a = self.args
        vdir = self.out / "verify"
        odir = self.out / "obligations"
        odir.mkdir(exist_ok=True)
        if not (vdir / "challenge.json").exists():
            s.status, s.detail = "skipped", "no challenge.json (the verifier did not reach the challenge)"
            return
        reader = CaptureReader(self.commit_dir)
        desc = Description(reader.description)
        comm = load_commitments(reader)
        cache = TreeCache(reader)
        ch = json.loads((vdir / "challenge.json").read_text())
        if (vdir / "verify.json").exists():  # WB's verifier (--preparation full)
            ver = json.loads((vdir / "verify.json").read_text())
            results = ((ver.get("report") or {}).get("results")) or []
        else:  # RC's selective verifier writes one VUResult per line
            results = [json.loads(l) for l in (vdir / "results.jsonl").read_text().splitlines() if l.strip()]
        rows = O.sampled_gemm_rows(ch, desc, reader, results)
        # the output-feeds-input chain: sampled GEMM rows whose producer VU (the row its `x` is) is itself in the sample
        producers = O.sampled_producers(rows, desc, reader, ch, results)
        sel = O.select_coordinate_units(rows, ch, rows_sharing_x=a.rows_sharing_x, coords_per_row=a.coords_per_row, ops_sharing_w=a.ops_sharing_w,
                                        chain_candidates=producers if a.chain else None)
        units, problems = O.open_units(desc, reader, comm, cache, sel)
        cp = vprofile.checkpoint_for(a.case, a.checkpoints)
        w_eqs, w_rep = O.input_equations(units, ch, model_dir=cp["local_path"], expected_file_sha256=cp.get("safetensors_sha256"), per_unit=a.weight_pins_per_unit)
        prod = [json.loads(l) for l in (self.adapter / "production" / "production.jsonl").read_text().splitlines() if l.strip()]
        b_eqs, b_rep = O.boundary_input_equations(reader, comm, cache, prod, ch, request_ids=sorted({u.request_id for u in units}))
        export, record = O.build(comm, units, w_eqs + b_eqs, profile_id=reader.profile_id, name=f"INT-{a.case}-composed",
                                 description=f"composed obligations of the {a.case} security row: {len(units)} GEMM coordinates of the sampled rows, {len(w_eqs)} checkpoint weight-word Input equations, {len(b_eqs)} boundary token half-word Input equations",
                                 source=f"{self.commit_dir} + {vdir / 'challenge.json'}")
        chain = O.chain_record(sel, units, producers, results, ch)
        record["provenance"]["chain"] = chain
        write_json(odir / "chain.json", chain)
        obj = TypedObligationSet.from_json(record["object"])
        statement = to_statement(obj)
        from veritor.interface.binding import bound_public_values
        from veritor.interface.lowering import statement_bytes_v3
        from veritor.protocol.proofs.wire import encode_statement, statement_digest
        # GUEST (S12, opt-in): the statement is handed to the guest in the RECORDED format.  Format 2 (default) is
        # the historical record; format 3 (`--statement-format 3`) writes `statement_bytes_v3(obj)` -- the same
        # lowering with uniform position rows as ranges (`Statement.expand` is the format-2 statement).  `h_L`
        # is ALWAYS sha256 of the bytes in statement.bin; a format-3 record also carries the format-2 digest.
        statement_format = int(a.statement_format)
        v2_bytes = encode_statement(statement)
        assert v2_bytes.hex() == record["statement"]  # Vector.to_json() wrote the format-2 column
        record["statement_format"] = statement_format
        if statement_format == 3:
            v3_bytes = statement_bytes_v3(obj)
            assert v3_bytes.hex() == record["statement_v3"]  # and the format-3 column
            (odir / "statement.bin").write_bytes(v3_bytes)
            record["statement_v2_digest"] = record["statement_digest"]
            record["statement_digest"] = statement_digest(v3_bytes).hex()
            assert record["statement_digest"] == record["statement_v3_digest"]
            # the binding column names the public values a correct prover produces for THIS format
            record["binding"] = {**record["binding"], "statement_digest": record["statement_digest"],
                                 "statement_format": 3, "public_values": bound_public_values(record["object"], statement_format=3).hex()}
        else:
            (odir / "statement.bin").write_bytes(v2_bytes)
            record["binding"] = {**record["binding"], "statement_format": 2}
        (odir / "witness.bin").write_bytes(bytes.fromhex(record["witness"]))
        # no plan.bin: a packed (format-3) witness is authenticated by the guest's own packed table (PW ran with
        # plan_bytes 0); the legacy plan encoder refuses a statement that opens several words of one leaf
        assert statement_digest((odir / "statement.bin").read_bytes()).hex() == record["statement_digest"]
        write_json(odir / "composed.json", record)
        write_json(odir / "selection.json", {"sampled_gemm_rows": len(rows), "sampled_gemm_rows_certified": sum("certified" in r.samples for r in rows), "selection": sel.to_json(),
                                              "rows_with_sampled_producer": len(producers), "producers": producers,
                                              "open_problems": problems, "weight_input_equations": w_rep, "boundary_input_equations": b_rep})
        comp = export.composition
        # per-family accounting: sampled (verifier-checked) vs proved
        fam_counts: dict[str, dict[str, int]] = {}
        for r in results:
            fam = _family_of_result(r)
            d = fam_counts.setdefault(fam, {"sampled": 0, "verifier_exact": 0, "proved_units": 0})
            d["sampled"] += 1
            d["verifier_exact"] += int(r.get("status") == "exact")
        proved_rows: dict[str, set[str]] = {}
        for u in units:
            fam_counts.setdefault(u.op_type, {"sampled": 0, "verifier_exact": 0, "proved_units": 0})["proved_units"] += 1
            proved_rows.setdefault(u.op_type, set()).add(u.canonical_id)
        families = {}
        for fam, d in sorted(fam_counts.items()):
            supported = fam in PROVED_FAMILIES
            rows_proved = len(proved_rows.get(fam, ()))
            families[fam] = {**d, "backend_support": PROVED_FAMILIES.get(fam, "unsupported by the generic backend"),
                             "proved_rows": rows_proved, "verifier_checked_not_proved": d["sampled"] - rows_proved,
                             "status": ("proved (coordinates of sampled rows) + verifier-checked rows" if rows_proved else
                                        "supported by the backend; no row of this family was selected for proving this run" if supported else
                                        "checked by the verifier, not proved")}
        unsupported_sampled = sum(d["sampled"] for f, d in fam_counts.items() if f not in PROVED_FAMILIES)
        s.artefacts = {"record": self.rel(odir / "composed.json"), "statement": self.rel(odir / "statement.bin"), "witness": self.rel(odir / "witness.bin"), "selection": self.rel(odir / "selection.json"),
                       "chain": self.rel(odir / "chain.json")}
        s.data = {
            "obligations": len(obj.obligations), "coordinate_units": len(units), "input_equations": {"weights": len(w_eqs), "boundary_halfwords": len(b_eqs)},
            "leaves": len(obj.leaves), "references": len(obj.references), "roots": [r.id for r in obj.roots], "subcircuits": len(obj.subcircuits),
            "h_O": record["digest"], "h_L": record["statement_digest"], "statement_format": statement_format, "statement_v2_digest": record.get("statement_v2_digest"),
            "statement_bytes": (odir / "statement.bin").stat().st_size, "statement_v2_bytes": len(record["statement"]) // 2, "witness_bytes": len(record["witness"]) // 2,
            "relations_verdict": record["relations"], "witness_verdict": record["witness_verdict"], "verdict": record["verdict"],
            "open_problems": problems, "shared_x_rows": comp["shared_x_rows"], "shared_w_rows": comp["shared_w_rows"],
            "input_equation_refs": [e["ref"] for e in comp["input_equations"]], "families": families, "unsupported_sampled": unsupported_sampled,
            "selection_note": sel.note, "authentication": sorted(set(comp["authentication"].values())),
            "chain": chain, "sampled_gemm_rows": len(rows), "rows_with_sampled_producer": len(producers),
        }
        ok = record["verdict"] == "accept" and record["witness_verdict"] == "accept" and not problems and len(units) >= 2 and (comp["shared_x_rows"] or comp["shared_w_rows"])
        if a.chain and chain.get("realised") is False and chain.get("candidates", 0) > 0:
            ok = False  # a chain was available and the composition failed to realise it: that is a composition bug, not a scope limit
        s.status = "ok" if ok else "FAIL"
        s.detail = (f"{len(obj.obligations)} obligations ({len(units)} GEMM coordinates, {len(w_eqs)} weight + {len(b_eqs)} boundary Input eqs) over {len(obj.leaves)} leaves / {len(obj.references)} refs; "
                    f"relations {record['relations']}, witness {record['witness_verdict']}; shared x rows {len(comp['shared_x_rows'])}, shared W rows {len(comp['shared_w_rows'])}; "
                    f"chain {'realised' if chain.get('realised') else 'not realised'} ({len(producers)} of {len(rows)} sampled GEMM rows have a sampled producer)")
        self.manifest["inputs"]["h_O"] = record["digest"]
        self.manifest["inputs"]["h_L"] = record["statement_digest"]
        # [REC] the exported obligation files are digest-pinned so the strict certificate can refuse a tampered value / a dropped obligation offline
        self.manifest["inputs"]["obligations_sha256"] = {name: sha256_file(odir / name) for name in ("composed.json", "statement.bin", "witness.bin", "selection.json") if (odir / name).exists()}
        self.manifest["inputs"]["obligations_count"] = len((record.get("object") or {}).get("obligations") or []) or record.get("obligations")
        self.manifest["inputs"]["statement_format"] = statement_format
        if statement_format == 3:
            self.manifest["inputs"]["statement_v2_digest"] = record["statement_v2_digest"]

    # -- stage 7: prove ---------------------------------------------------------------------------

    def host(self, name: str, *hargs: str, prover: str | None = None) -> dict[str, Any]:
        cmd = [self.args.host, *hargs]
        log = self.out / "logs" / f"host-{name}.log"
        self.log(f"  $ {' '.join(cmd)}")
        env = dict(os.environ)
        if prover is not None:
            env["SP1_PROVER"] = prover  # verify is CPU-only: never require the CUDA server for it
        env.setdefault("SP1_PROVER", self.args.sp1_prover)
        env.setdefault("RUST_LOG", "info")
        res = subprocess.run(cmd, capture_output=True, text=True, env=env)
        log.write_text(res.stderr + "\n" + res.stdout)
        if res.returncode != 0:
            raise RuntimeError(f"host {name} exited {res.returncode}: {res.stderr[-800:]}")
        return host_json(res.stdout)

    def stage_prove(self, s: Stage) -> None:
        from veritor.interface import TypedObligationSet, check_binding

        odir = self.out / "obligations"
        pdir = self.out / "proofs"
        pdir.mkdir(exist_ok=True)
        if not (odir / "statement.bin").exists():
            s.status, s.detail = "skipped", "no statement to prove"
            return
        record = json.loads((odir / "composed.json").read_text())
        h_o = record["digest"]
        st, wt = str(odir / "statement.bin"), str(odir / "witness.bin")
        info = self.host("info", "info", prover="cpu")
        write_json(pdir / "info.json", info)
        common = ["--statement", st, "--witness", wt, "--typed-object-digest", h_o]
        t = time.perf_counter()
        ex = self.host("execute", "execute", *common, prover="cpu")
        ex["seconds"] = round(time.perf_counter() - t, 2)
        write_json(pdir / "execute.json", ex)
        proofs: dict[str, Any] = {}
        binding: dict[str, Any] = {}
        all_ok = bool(ex.get("verdict")) and not ex.get("unsound")
        for mode in ("core", "compressed"):
            out = pdir / f"proof-{mode}.bin"
            t = time.perf_counter()
            # [PAD-INT, additive; default 0 = the old single attempt] on a SHARED GPU the sp1-gpu-server dies now and then
            # ("CudaClientError: ... early eof") when another process holds the card; the prove is deterministic in its
            # inputs, so a retry proves the same statement — the attempt count is recorded in prove-<mode>.json.
            attempts = 0
            while True:
                attempts += 1
                try:
                    pr = self.host(f"prove-{mode}", "prove", *common, "--out", str(out), "--mode", mode)
                    break
                except RuntimeError as e:
                    if attempts > int(getattr(self.args, "prove_retries", 0) or 0):
                        raise
                    self.log(f"  prove-{mode} attempt {attempts} failed ({str(e)[-160:]!r}); retrying in 30 s")
                    time.sleep(30)
            pr["attempts"] = attempts
            pr["wall_seconds"] = round(time.perf_counter() - t, 2)
            write_json(pdir / f"prove-{mode}.json", pr)
            t = time.perf_counter()
            vr = self.host(f"verify-{mode}", "verify", "--proof", str(out), "--statement", st, prover="cpu")
            vr["wall_seconds"] = round(time.perf_counter() - t, 2)
            write_json(pdir / f"verify-{mode}.json", vr)
            pv = bytes.fromhex(vr["public_values"])
            b = check_binding(TypedObligationSet.from_json(record["object"]), pv, statement_format=int(record.get("statement_format", 2)))
            binding[mode] = {"ok": b.ok, "code": str(getattr(b, "code", "")), "detail": getattr(b, "detail", ""), "verdict": getattr(b, "verdict", None), "public_values_len": len(pv)}
            ok = bool(vr.get("ok")) and bool(vr.get("statement_match")) and bool(vr.get("verdict")) and not vr.get("unsound") and vr.get("vk_hash") == info.get("vk_hash") and b.ok
            proofs[mode] = {"ok": ok, "proof_bytes": pr.get("proof_bytes"), "shards": pr.get("shards"), "setup_seconds": pr.get("setup_seconds"), "prove_seconds": pr.get("prove_seconds"),
                            "host_verify_seconds": pr.get("verify_seconds"), "verify_ok": vr.get("ok"), "statement_match": vr.get("statement_match"), "verdict": vr.get("verdict"),
                            "vk_hash": vr.get("vk_hash"), "proof_sha256": sha256_file(out), "public_values": vr.get("public_values")}
            all_ok &= ok
        write_json(pdir / "binding.json", {"layout": "h_O || h_L || verdict (65 bytes; IB check_binding)", "h_O": h_o, "h_L": record["statement_digest"],
                                           "statement_format": int(record.get("statement_format", 2)), "modes": binding})
        s.artefacts = {"proofs_dir": self.rel(pdir), "execute": self.rel(pdir / "execute.json"), "binding": self.rel(pdir / "binding.json")}
        s.data = {"elf_sha256": info.get("elf_sha256"), "vk_hash": info.get("vk_hash"), "sp1_version": info.get("sp1_version"), "guest_features": info.get("guest_features"),
                  "total_cycles": ex.get("total_cycles"), "cycle_tracker": ex.get("cycle_tracker"), "public_values": ex.get("public_values"), "execute_verdict": ex.get("verdict"),
                  "proofs": proofs, "binding": binding, "host_sha256": sha256_file(Path(self.args.host))}
        s.status = "ok" if all_ok else "FAIL"
        s.detail = f"cycles {ex.get('total_cycles')}; " + "; ".join(f"{m}: {p['proof_bytes']} B, prove {p['prove_seconds']}s, verify ok={p['verify_ok']}, binding ok={binding[m]['ok']}" for m, p in proofs.items())
        # GUEST (S12): the host's self-reported features (a CLAIM, recorded as such) and the build pin the host ships
        # with (`<host>.guest_build.json`, written by e2e/guest_build.py after `cargo prove build`; --guest-build overrides).
        # Neither is evidence by itself: the verifier checks the ELF/vk against e2e/approved_guests.json.
        guest_build = None
        pin_file = Path(self.args.guest_build) if self.args.guest_build else Path(str(self.args.host) + ".guest_build.json")
        if pin_file.exists():
            guest_build = json.loads(pin_file.read_text())
            if guest_build.get("elf_sha256") != info.get("elf_sha256") or str(guest_build.get("vk_hash", "")).lower() != str(info.get("vk_hash", "")).lower():
                s.status = "FAIL"
                s.detail += f"; guest_build pin {pin_file.name} names ELF {str(guest_build.get('elf_sha256'))[:12]} / vk {str(guest_build.get('vk_hash'))[:14]}, the host reports {str(info.get('elf_sha256'))[:12]} / {str(info.get('vk_hash'))[:14]}"
        s.data["guest_build"] = guest_build
        s.data["guest_build_source"] = str(pin_file) if guest_build else None
        self.manifest["inputs"].update({"elf_sha256": info.get("elf_sha256"), "vk_hash": info.get("vk_hash"), "sp1_version": info.get("sp1_version"), "host_sha256": s.data["host_sha256"],
                                        "guest_features": info.get("guest_features"), "guest_build": guest_build})

    # -- stage 7b: archive the prover binaries beside the record -----------------------------------

    def stage_binaries(self, s: Stage) -> None:
        """integration-pad.md §2 item 3: true proof re-verification needs the exact guest identity (the vk
        pins the guest ELF; a locally built host is a different program).  The host binary (which embeds
        the guest) is copied INTO the record, so the record carries everything a second process needs on
        any Linux x86_64 host; a standalone guest ELF is copied too when the build tree still has it."""
        bdir = self.out / "bin"
        bdir.mkdir(exist_ok=True)
        host = Path(self.args.host)
        copied: dict[str, str] = {}
        if host.exists():
            shutil.copy2(host, bdir / "veritor-zk-host")
            copied["veritor-zk-host"] = sha256_file(bdir / "veritor-zk-host")
        elves = sorted((ROOT / "zk" / "sp1").glob("**/riscv32im-succinct-zkvm-elf/release/*"), key=lambda p: p.stat().st_mtime, reverse=True)
        for elf in elves[:2]:
            if elf.is_file():
                shutil.copy2(elf, bdir / f"guest-{elf.name}.elf")
                copied[f"guest-{elf.name}.elf"] = sha256_file(bdir / f"guest-{elf.name}.elf")
        info = json.loads((self.out / "proofs" / "info.json").read_text()) if (self.out / "proofs" / "info.json").exists() else {}
        s.artefacts = {name: f"bin/{name}" for name in copied}
        s.data = {"copied_sha256": copied, "host_source": str(host), "elf_sha256_expected": info.get("elf_sha256"), "vk_hash": info.get("vk_hash"),
                  "note": "the host embeds the guest ELF (elf_sha256/vk_hash above); re-verification: bin/veritor-zk-host verify --proof proofs/proof-<mode>.bin --statement obligations/statement.bin"}
        self.manifest["inputs"]["archived_binaries"] = copied
        s.status = "ok" if copied else "FAIL"
        s.detail = ", ".join(f"{k} {v[:12]}" for k, v in copied.items()) or f"host binary not found at {host}"

    # -- stage 8: bundle --------------------------------------------------------------------------

    def stage_bundle(self, s: Stage) -> None:
        from verity_vllm import bundle as vbundle

        bdir = self.out / "bundle"
        bdir.mkdir(exist_ok=True)
        name = f"INT-{self.args.case}-{self.out.name}"
        vdir = self.out / "verify"
        tar = bdir / f"{name}.tar.gz"
        vbundle.pack(self.commit_dir, tar, name=name, production_record=self.adapter / "production" / "production.jsonl",
                     verify_dir=vdir if vdir.is_dir() else None, weights="omit")
        write_json(bdir / "bundle.json", vbundle.read_manifest(tar))
        s.artefacts = {"tarball": self.rel(tar), "bundle_json": self.rel(bdir / "bundle.json"),
                       "weights_values": str(self.commit_dir / "weights" / "values.bin"),
                       # integration-pad.md §2 item 2: the run-relative layout, so a second machine can find
                       # the weight values without the run machine's absolute paths
                       "weights_values_rel": self.rel(self.commit_dir / "weights" / "values.bin")}
        s.data = {"tarball_sha256": sha256_file(tar), "bytes": tar.stat().st_size, "name": name}
        s.detail = f"{tar.name}: {tar.stat().st_size:,} bytes, sha256 {s.data['tarball_sha256'][:16]}; weights referenced by checkpoint sha (values.bin kept beside the run)"

    # -- summary ----------------------------------------------------------------------------------

    def verdict(self) -> tuple[str, list[str]]:
        reasons: list[str] = []
        required = ["program", "plan", "production", "preparation", "verify", "correspondence", "obligations", "prove"]
        for name in required:
            st = self.stages.get(name)
            if st is None or st.status != "ok":
                reasons.append(f"{name}: {st.status if st else 'not run'}" + (f" ({st.detail[:160]})" if st and st.detail else ""))
        for name, st in self.stages.items():
            if name not in required and st.status in ("FAIL", "error"):
                reasons.append(f"{name}: {st.status}")
        return ("PASS" if not reasons else "FAIL"), reasons

    def write_summary(self, *, partial: bool = False, final_note: str = "") -> None:
        verdict, reasons = self.verdict()
        caught = next((r.split(":")[0] for r in reasons), None)
        acct = self.accounting()
        prep = self.stages.get("preparation") or Stage("x")
        if self.selective:
            preparation_label = SELECTIVE_LABEL + (" -- production (and the control replay) reused from another run; the protocol itself is never reused: r_q is drawn per run" if self.args.reuse_adapter else "")
        else:
            preparation_label = PREPARATION_LABEL + (" -- reused from another run" if self.args.reuse_adapter else "")
        summary = {
            "schema": SCHEMA, "case": self.args.case, "out": str(self.out), "utc": utc(), "partial": partial, "wall_seconds": round(time.time() - self.t0, 1),
            "scope": {
                "workload": WORKLOADS.get(self.args.case, f"{self.args.case}: {self.args.workload}"),
                "preparation": preparation_label,
                "selection": ({"selector": "fixed_size k of N (RC: q = k/N realised exactly; the Bound is stated for Bernoulli(q) -- recorded, not substituted)" if not self.args.selective_q
                               else f"Bernoulli q={self.args.selective_q} over requests (WB's selector)", "k": self.args.k if not self.args.selective_q else None,
                               "q_measured": prep.data.get("q_measured"), "selected_requests": prep.data.get("selected"),
                               "batch_invariance": (prep.data.get("batch_invariance") or {}).get("bit_identical"),
                               "batch_invariance_note": "tested: every RU leaf of J (replayed alone) vs the full-replay control (all requests), RC compare_selected"}
                              if self.selective else {"selector": "none: every request replayed (q = 1)"}),
                "challenge": self.args.challenge, "binding": "IB check_binding over the 65-byte h_O || h_L || verdict public values",
                "proved_families": PROVED_FAMILIES, "negative": self.args.negative,
                "negative_point": NEGATIVE_POINT.get(self.args.negative) if (self.args.negative and self.selective) else None,
                "chain": (self.stages.get("obligations") or Stage("x")).data.get("chain"),
                "full_program_validation": (self.stages.get("plan") or Stage("x")).data.get("full_program"),
                "plan_validation": {"validated_units": (self.stages.get("plan") or Stage("x")).data.get("units"),
                                    "not_validated_units": (self.stages.get("plan") or Stage("x")).data.get("not_validated_units"),
                                    "unit_probe": (self.stages.get("plan") or Stage("x")).data.get("unit_probe"),
                                    "note": ("`layer` = LayerPost{C} (INT's unit: the adapter's row VUs and the backend's coordinate VUs of one decoder layer); "
                                             "`gemm:<K>x<N>` = one GEMM shape of the layer validated as its own program. A unit listed under not_validated_units "
                                             "was NOT validated in this run; unit_probe records why (the measured memory need under the pod's cgroup)")},
                "verifier_cost": VERIFIER_COST_NOTE | {"this_run": {"verifier": "selective_verifier (RC; MemoCache)" if self.selective else "verify.py (WB)",
                                                                    "check_seconds": ((self.stages.get("verify") or Stage("x")).data.get("timing") or {}).get("check_s"),
                                                                    "work_seconds": ((self.stages.get("verify") or Stage("x")).data.get("timing") or {}).get("work_total_s"),
                                                                    "vus_checked": (self.stages.get("verify") or Stage("x")).data.get("sample_union")}},
            },
            "stages": {k: v.to_json() for k, v in self.stages.items()},
            "timings_seconds": {k: round(v.seconds, 1) for k, v in self.stages.items()},
            "timings_note": ("selective: preparation = the prover's process (phase 0 + wait + phase 2) + control replay + compare_selected, verify = the verifier's work "
                             "(waits excluded, from selective_verify.json timing.work_total_s); the two run concurrently -- protocol_wall_seconds is the wall of both"
                             if self.selective else "stages run sequentially"),
            "protocol_wall_seconds": self.protocol.get("wall_seconds"),
            "families": (self.stages.get("obligations") or Stage("x")).data.get("families"),
            "composition": {k: (self.stages.get("obligations") or Stage("x")).data.get(k) for k in ("coordinate_units", "shared_x_rows", "shared_w_rows", "input_equation_refs", "input_equations", "roots", "leaves", "references", "chain")},
            "accounting": acct,
            "verdict": verdict, "verdict_reasons": reasons, "caught_by_stage": caught if verdict == "FAIL" else None, "note": final_note,
        }
        write_json(self.out / "summary.json", summary)
        self.manifest["utc_end"] = utc()
        write_json(self.out / "manifest.json", self.manifest)

    def accounting(self) -> dict[str, Any]:
        """The assessment's three counts, where the existing commit/description data give them cheaply."""
        prep = self.stages.get("preparation")
        out: dict[str, Any] = {}
        commit = getattr(self, "commit_dir", None)
        if prep and prep.status == "ok" and commit and (Path(commit) / "required.json").exists():
            try:
                req = json.loads((Path(commit) / "required.json").read_text())
                ifaces = req.get("interfaces") or []
                out["required_distinct_boundary_values"] = {"interfaces": len(ifaces), "input_bytes": sum(int(i.get("input_bytes", 0)) for i in ifaces), "output_bits": sum(int(i.get("output_bits", 0)) for i in ifaces),
                                                            "note": "per required.json (the declared row interfaces of the partition); a per-value count over the whole Program is the plan's business"}
            except Exception as e:  # noqa: BLE001
                out["required_distinct_boundary_values"] = {"missing": f"required.json unreadable: {e}"}
            out["committed_bytes"] = {"bytes_captured": prep.data.get("bytes_captured"), "leaves": prep.data.get("leaves"), "trees": prep.data.get("trees"), "note": "capture_summary.bytes_captured = the committed activation bytes (weights excluded)"}
            out["extra_bytes_packing_or_conservative_capture"] = {"missing": "not computed by the existing commit data: needs the per-value required set of the whole Program against the committed leaves (RC / plan lane)"}
        else:
            out["missing"] = "preparation did not complete"
        return out

    def finish(self, final_note: str = "") -> None:
        self.write_summary(partial=False, final_note=final_note)
        verdict, reasons = self.verdict()
        self.log(f"VERDICT {verdict}" + (f" caught by {reasons[0].split(':')[0]}: {reasons[0]}" if reasons else "") + f" wall {time.time() - self.t0:.0f}s")

    # -- main -------------------------------------------------------------------------------------

    def main(self) -> int:
        a = self.args
        self.manifest["inputs"].update({
            "checkpoints": _checkpoint_digests(a.checkpoints, a.case), "workload": a.workload, "adapter_revision": a.adapter_revision, "wheel_sha256": a.wheel_sha256, "vllm_commit": a.vllm_commit,
            # INT's field: B0's frozen Llama profile. The profile the run actually used (auto-selected per checkpoint architecture,
            # e.g. qwen2-* for B1) is recorded by the preparation stage under inputs.semantic_profile.
            "semantic_profile_sha256": sha256_file(POC / "manifests" / "semantic-profiles" / "llama-eager-batch-invariant-v1.json") if a.case == "B0" else None,
            "python": {"runner": sys.version.split()[0], "vllm_venv": a.py_vllm}, "platform": platform.platform(), "challenge_mode": a.challenge, "seed": a.seed,
            "policy": {"q": a.q, "s": a.s, "coverage_size": a.coverage_size}, "negative": a.negative, "reuse_adapter": a.reuse_adapter,
            "preparation": a.preparation, "selective": ({"k": a.k, "q": a.selective_q, "control": a.control} if self.selective else None),
            "chain": a.chain, "full_program": a.full_program,
        })
        self.commit_dir = self.adapter / "commit-v2"
        self.run_stage("program", self.stage_program)
        self.run_stage("plan", self.stage_plan)
        self.run_stage("production", self.stage_production)
        if self.stages["production"].status != "ok":
            self.finish("production failed; later stages not run")
            return 1
        if self.selective:
            # one protocol run fills both stages (the two parties are concurrent); a negative's mutation is applied inside the
            # protocol at its relay point, so the separate mutation stage records the point rather than acting
            self.run_stage("preparation", self.stage_preparation_selective)
            if self.stages["preparation"].status != "ok":
                self.finish("preparation failed; later stages not run")
                return 1
            if a.negative:
                self.run_stage("mutation", self.stage_mutation_selective_record)
            self.run_stage("verify", self.stage_verify_selective, fatal=False)
            self.stages["verify"].seconds = float(((self.stages["verify"].data.get("timing") or {}).get("work_total_s")) or self.stages["verify"].seconds)
        else:
            self.run_stage("preparation", self.stage_preparation)
            if self.stages["preparation"].status != "ok":
                self.finish("preparation failed; later stages not run")
                return 1
            if a.negative:
                self.run_stage("mutation", self.stage_mutation)
            self.run_stage("verify", self.stage_verify, fatal=False)
        self.run_stage("correspondence", self.stage_correspondence, fatal=False)
        self.run_stage("obligations", self.stage_obligations, fatal=False)
        if self.stages["obligations"].status == "ok" or (self.stages["obligations"].data.get("obligations") or 0) > 0:
            self.run_stage("prove", self.stage_prove, fatal=False)  # a negative's non-empty (rejecting) object is still proved: verdict 0 is the public output
        else:
            self.stage("prove").status, self.stage("prove").detail = "skipped", "no obligation to prove (the obligations stage exported none)"
        self.run_stage("binaries", self.stage_binaries, fatal=False)
        self.run_stage("bundle", self.stage_bundle, fatal=False)
        self.finish()
        return 0 if self.verdict()[0] == "PASS" else 1


def _positions_processed(production_jsonl: Path) -> int:
    """RC TABLE 1's denominator: positions of every production request (prompt + generated - 1, ``selective.replayed_positions``' rule)."""
    total = 0
    for line in production_jsonl.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        prompt_len = int(r.get("prompt_len") or len(r.get("prompt_token_ids") or []))
        total += prompt_len + max(len(r.get("output_token_ids") or []) - 1, 0)
    return total


def _classify_refusal(reason: str, err: str) -> str:
    """The first blocker of a refused selective protocol, from the verifier's abort reason (or its last error line)."""
    text = f"{reason} {err}".lower()
    if "binding" in text:
        return "binding" + (": architecture" if "architecture" in text else ": model" if "model" in text else ": output" if "output" in text else "")
    if "first-stage" in text or "first_stage" in text:
        return "structural: first stage over J"
    if "phase-0 records changed" in text:
        return "structural: phase-0 commitments changed after r_q"
    if "observed description differs" in text:
        return "structural: observed description differs from the registered one"
    if "re-derive" in text or "rederive" in text:
        return "structural: transcript does not re-derive"
    if "not selective" in text:
        return "structural: interiors existed before r_q"
    if "validate" in text or "description" in text:
        return "structural: description"
    return "verifier refused: " + (reason or err)[:120]


def _family_of_result(r: dict[str, Any]) -> str:
    op_type = str(r.get("op_type") or "")
    op = str(r.get("op") or "")
    if op_type in ("gemm",):
        return "gemm"
    if op_type == "lm_head":
        return "lm_head"
    if op_type:
        return op_type
    return op.split(".")[-1] or "unknown"


def _checkpoint_digests(path: str, case: str) -> dict[str, Any]:
    doc = json.loads(Path(path).read_text())
    entries = doc if isinstance(doc, list) else doc.get("checkpoints") or doc.get("models") or doc
    if isinstance(entries, dict):
        entries = list(entries.values()) if all(isinstance(v, dict) for v in entries.values()) else [entries]
    found = [e for e in entries if isinstance(e, dict) and (e.get("role") == case or e.get("case") == case)]
    e = found[0] if found else {}
    return {"file": path, "file_sha256": sha256_file(Path(path)), "repo": e.get("repo"), "revision": e.get("revision"), "safetensors_sha256": e.get("safetensors_sha256"), "config_sha256": e.get("config_sha256")}


def git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"], cwd=str(ROOT), text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return os.environ.get("VERITY_ADAPTER_REVISION", "unknown")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="B0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--challenge", choices=("security", "diagnostic"), default="security")
    ap.add_argument("--seed", type=int, default=7, help="diagnostic mode only (security mode uses the verifier's own randomness)")
    ap.add_argument("--q", default="1")
    ap.add_argument("--s", default="1/4096")
    ap.add_argument("--coverage-size", type=int, default=1000)
    ap.add_argument("--statement-format", type=int, choices=(2, 3), default=2,
                    help="[GUEST, S12 opt-in] the statement format handed to the guest and recorded in obligations/statement.bin: 2 (default, "
                         "the historical record) or 3 (ranged positions; the same lowering, h_L = sha256 of the format-3 bytes; the record "
                         "also carries statement_v2_digest). 3 is NOT the default until the actual-guest smoke test is approved.")
    ap.add_argument("--workload", default=str(HERE / "workload-B0-small.json"))
    ap.add_argument("--checkpoints", default=str(POC / "manifests" / "checkpoints.json"))
    ap.add_argument("--adapter-revision", default=git_revision())
    ap.add_argument("--wheel-sha256", default=os.environ.get("WHEEL", "7aa52ac71500caf74643135a340eb2d03ae0386d4282fc0b23846940029fbbb3"))
    ap.add_argument("--vllm-commit", default=os.environ.get("VLLM_COMMIT", "d9105ea80"))
    ap.add_argument("--py-vllm", default=os.environ.get("VERITY_PY_VLLM", "/workspace/venv-cu129/bin/python"))
    ap.add_argument("--host", default=os.environ.get("VERITY_ZK_HOST", "/workspace/bin/veritor-zk-host-cuda"))
    ap.add_argument("--sp1-prover", default=os.environ.get("SP1_PROVER", "cuda"))
    ap.add_argument("--guest-build", default=None,
                    help="[GUEST] the guest build pin (e2e/guest_build.py output) to record in manifest.inputs.guest_build; default <host>.guest_build.json "
                         "when it exists. Its elf_sha256/vk_hash must match the host's `info`.")
    ap.add_argument("--padded-program", default=None,
                    help="[PAD-INT, additive; GUIDANCE §51.4] stage 1 builds the DERIVED padded program instead of Serve_v1: `derive` (runs "
                         "derive_cli serve --padded) or `derived:<descriptor.json.gz>`; digest must equal the registered Serve_v3 oracle")
    ap.add_argument("--eos-id", type=int, default=0, help="[PAD-INT] EOS id bound into the padded program (B0: 0)")
    ap.add_argument("--derive-python", default=None, help="[PAD-INT] interpreter for derive_cli (needs torch); default: this one")
    ap.add_argument("--prove-retries", type=int, default=0, help="[PAD-INT, additive] retries of a failed `host prove` (shared-GPU sp1-gpu-server crashes); 0 = old behaviour")
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME", "/workspace/hf"))
    ap.add_argument("--cache-root", default="/workspace/cache")
    ap.add_argument("--arena-gb", type=int, default=8)
    ap.add_argument("--weights-values", default="store", choices=("store", "hash", "checkpoint"))
    ap.add_argument("--worker", default="INT")
    ap.add_argument("--reuse-adapter", default=None, help="reuse production/replay/commit from this run's adapter dir")
    ap.add_argument("--negative", choices=sorted(NEGATIVES), default=None)
    ap.add_argument("--rows-sharing-x", type=int, default=2)
    ap.add_argument("--coords-per-row", type=int, default=2)
    ap.add_argument("--ops-sharing-w", type=int, default=1)
    ap.add_argument("--weight-pins-per-unit", type=int, default=2)
    ap.add_argument("--bound-steps", type=int, default=200_000, help="reach-step cap per unit for the bound tables (0 = skip the bound; 2e6 exceeded the pod's 31 GB cgroup on P1)")
    ap.add_argument("--plan-unit", nargs="+", default=["layer"],
                    help="the sub-program(s) the validator checks: `layer` (LayerPost{C}, INT's unit, default) and/or `gemm:<K>x<N>` (one of the layer's GEMM "
                         "shapes as its own program) -- for a case whose layer does not fit the pod's host memory; what was not validated is recorded")
    ap.add_argument("--plan-probe", default=None, help="e2e/probe_plan_units.py output to record beside the plan (the measured memory finding behind --plan-unit)")
    ap.add_argument("--preparation", choices=("full", "selective"), default="selective",
                    help="selective = RC's protocol (boundaries of all requests committed before r_q, only the selected J replayed; INT3); full = INT's original full replay")
    ap.add_argument("--k", type=int, default=2, help="selective: number of requests r_q selects (fixed-size selector; q_measured = k/N)")
    ap.add_argument("--selective-q", default=None, help="selective: use WB's Bernoulli selector with this q instead of --k")
    ap.add_argument("--control", choices=("full", "none"), default="full",
                    help="selective: also run a full-commit replay of every request and compare J's leaves against it (batch invariance of the audit); 'none' skips the control")
    ap.add_argument("--chain", dest="chain", action="store_true", default=True, help="prefer coordinate units whose producer VU is also sampled (output-feeds-input chain)")
    ap.add_argument("--no-chain", dest="chain", action="store_false")
    ap.add_argument("--full-program", choices=("try", "skip"), default="try",
                    help="plan stage: also validate the whole Serve program in a memory-capped subprocess (the finding is recorded either way)")
    ap.add_argument("--full-program-gb", type=int, default=20, help="address-space cap (GiB) for the full-program validation subprocess")
    ap.add_argument("--full-program-timeout", type=float, default=1800.0, help="seconds before the full-program validation subprocess is killed")
    ap.add_argument("--semantic-profile", default="auto", help="selective: the semantic profile both parties use (auto = binding.select_semantic_profile's rule)")
    ap.add_argument("--protocol-timeout", type=float, default=5400.0, help="selective: seconds before both parties are killed")
    ap.add_argument("--instance-dir", default=None,
                    help="[REC] directory with the pre-challenge protocol_instance.json + population.json + compact_validation.json (agreement 1): copied into "
                         "the record and digest-pinned in the manifest before READY_PHASE0; the instance's program digest and law must match this run")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.challenge == "security" and "--seed" in (argv or sys.argv):
        print("--seed is a diagnostic-mode argument; security mode uses the verifier's own randomness", file=sys.stderr)
        return 2
    return Run(args).main()


if __name__ == "__main__":
    raise SystemExit(main())
