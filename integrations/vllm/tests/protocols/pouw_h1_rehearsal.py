"""The first served H1 window end to end against a challenger gate: rehearsed against a local challenger gate on the serving node
(`local_gate.py`), and the same script run live against the auditor host's. Run by hand or as a research run; never collected. ``--red-team RUN_OUT`` reruns the red team's edited copies of a
fetched run on any CPU.

**What it does.** Builds the window's declared Program first, on CPU (`build_program`: `pouw_h1_served.declared` for
`pouw_h1.FILL` over Llama-3.1-8B, no window's prefix; its descriptor's SHA-512 is its `program_digest`, the period's
circuit), unless ``--program DIR`` names a ``--program-only`` run's. With ``--served`` the period's weights are the
auditor's registration of `MODEL` under that circuit: each K's ``b/<K>`` committed from the published
checkpoint under the auditor's own salt key, their root W. Against the local gate the rehearsal registers it first, as
vy-admin does (`register_model`: ``admin model-register --model MODEL --circuit DIGEST --descriptor FILE --checkpoint
DIR``, DIR ``--checkpoint``); on the auditor host vy-admin makes that registration. It builds each stratum's
circuit file with the auditor's lowering (`pouw_h1_circuits.build`: K's template lowered as ``h1_prove/stage.py``
lowers it, bound to the Program and its partition), unless ``--circuits DIR`` names a built directory. Then it opens
the lane's period (`period_open_argv`: ``period-open --name P --cap 1 --circuit DIGEST --draw-k K --model MODEL
--circuit-files FILES --length LENGTH``, which signs the registration's W, K the proof service's committed draw K, `declared_draw_k`, which core's credit reads as the
open's ``draw_k``, FILES ``{"k<K>": SHA-512}``, each stratum's file's, which core's credit verifies statements
against, `credit.committed_files`, and LENGTH `FILL`'s fixed length, `fill_length`, which core's credit holds the
window's served rows to, `credit.committed_length`), reserves the window's challenge phase (``reserve --peer PEER --scheme S --families
F``), and serves
`pouw_h1.FILL` (Llama-3.1-8B int8, 64 in flight, T = 64, S = 18) with ``pouw_h1_gpu.py --workload fill --served
--model MODEL --circuit DIGEST --challenger AT --session RESERVE --gate GATE`` (the window's b/<K> salted with the key the
gate's ``model-key`` gives the lane's key), under ``gpu-lease 1 --wait --max-min MIN`` with ``--gpu-lease MIN``, so
the GPU is held only while it serves (a run without ``--gpus`` sees no GPU, and gets the one its own
``gpu-lease`` hands out): the window opens its challenge phase through ``GATE open --session S --root R`` once round
0 is admitted, R = `salt.window_open_root(W, round 0's request table root, S)`, its salt coin opens after R,
and each call's A and z/y tables are committed as the window Program's registered values, beside the period's ``b/<K>``
(`pouw_h1_served.Served`) under the round roots; the served Program's layout must be the declared one's. The serving process stops at its last
served step: it hands the window and its challenge phase's stream to this process (``--handoff``, `pouw_h1_handoff`) and exits,
ending its lease, and ``pouw_h1_gpu.py --close-from`` closes and releases the window on no GPU, on the same challenge phase
(`serve`; the verdict's ``stages`` has when each phase exited and ``lease_after_served_s``). Once the challenge phase has its ``end`` entry it
closes the period, takes the ledger (``GATE ledger``), the host's pins log (``GATE pins``, the bytes the gate publishes as
``challenger-pins head``), the challenge phase's tree (``GATE fetch S``) and the key's pin (``GATE pubkey``), and runs the
verifier's side on them (`pouw_h1_gpu.check_challenger`): `pouw_h1_challenger.session_reason`
against the ledger's newest head, `trusted_rounds_reason` on the challenge phase's ``out/`` with the window's ``{root}``, each
reading the challenge phase through core's `challenger.pinned_session` under that pins log and the verifier's own first-parent
history of main (``--main``, ``git rev-list --first-parent origin/main``; without it every challenge phase is refused),
`pouw_h1_challenger.manifest_reason` and `period_reason`. ``pouw_h1_served.register``'s outcome (each call's
``--registered`` file) is in the serve record. Then
the red team (`red_team`), CPU only, reruns the verifier's side on edited copies of the record and the ledger, each of
which must be refused: R replaced by the bare weights' root, R replaced by another window's root, the ledger cut before
the challenge phase's end entry, a call's table root changed, a call with no A table root, the round log cut before the last
step, a pin that isn't the gate key's, no pins log, a pins log cut before the lines the open sealed and one with a
line rewritten, no copy of main's first-parent history and one without the challenge phase's pin; and with the local gate, a
period opened with a ``--weights-root`` no registration has (refused at the open), and challenge phases forged through the gate
itself (a period of their own), opened on the bare weights' root and on another window's root with the window's other
coins after its own roots. The verdict is ``$OUT/rehearsal.json``: ``ok`` when the honest window is accepted by
``session_reason``, ``trusted_rounds_reason``, ``manifest_reason`` and ``period_reason``, each of its calls' A tables is
under its round's root (``a_roots``), its served layout is the declared one's, the salt key is kept past the close in
the prover's own directory and published nowhere under ``$OUT`` (``key_custody``), and every red-team case is refused.
Its ``credit`` is the gate-passed rows beside core's (`pouw_h1_challenger.credit_reason`: ``manifest``, ``window``,
``calls``, ``eps``), which credits nothing until each drawn call's statement (its proof files) exists.

**The rehearsal** (a research run on the serving node; the gate's tree is one whose serve.py ends a challenge phase with its record
leaf and refused hellos and seals ``hellos.jsonl``; with ``--served`` the tree must be one with the model registry,
archived with the int8 checkpoints' manifest
its ``model-register`` reads, and ``--checkpoint`` the pinned LLAMA31_8B_INT8 snapshot on the node):

    git archive --format=tar.gz -o gate-tree.tgz COMMIT verity benchmarks/live_challenger \\
      integrations/vllm/manifests/checkpoints-int8.json
    git rev-list --first-parent origin/main > main.txt
    research run --on vy-nebius-2 --project verity --source . --cwd source --send gate-tree.tgz --send main.txt \\
      --mem-gb 300 --mem-wait 600 --declared-output 'out/*' -- bash -lc 'cd integrations/vllm && \\
      PYTHONPATH=$PWD/../..:$PWD HF_HOME=/workspace/hf /workspace/jobs/venv312/bin/python \\
      -m tests.protocols.pouw_h1_rehearsal --out $OUT --gpu-lease 90 --main $RESEARCH_RUN_DIR/inputs/main.txt \\
      --tree $RESEARCH_RUN_DIR/inputs/gate-tree.tgz --commit COMMIT --checkpoint SNAPSHOT_DIR'

Each stratum's circuit file is the long CPU step at the 32 × 32 tile, so build it once as its own run and send it with
``--circuits DIR``: ``python -m tests.protocols.pouw_h1_circuits --program PROGRAM_DIR --out $OUT``.

**The live run** is the same command with the auditor host's gate, over the lane's own gate key (on the serving node),
the auditor's pin, the declared Program's ``window-program.meta.json`` sent (a ``--program-only`` run's, or a
rehearsal's ``out/program/``), its circuit files' directory (``--circuits``), and no ``--tree``:

    research run --on vy-nebius-2 --project verity --source . --cwd source --send window-program.meta.json \\
      --mem-gb 300 --mem-wait 600 --declared-output 'out/*' -- bash -lc 'cd integrations/vllm && \\
      PYTHONPATH=$PWD/../..:$PWD HF_HOME=/workspace/hf /workspace/jobs/venv312/bin/python \\
      -m tests.protocols.pouw_h1_rehearsal --out $OUT --gpu-lease 90 --program $RESEARCH_RUN_DIR/inputs \\
      --circuits $RESEARCH_RUN_DIR/inputs/circuits \\
      --gate "ssh -i $HOME/.local/state/vllm-gate/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes research@204.12.178.84" \\
      --peer vy-nebius-2 --at 127.0.0.1:7811 --period vllm-h1-live-1 \\
      --key-sha512 27271eaf917d1a817198cbcfe7684debd9bdb11b9e386460948bb981642fab3ae065c0a877330a0da68ae76969550eb786dc172319876aaca21fd116e5a193bc'

What changes from the rehearsal: the gate is the auditor host's forced command over ssh (``by`` the lane the key is
registered for, ``vllm``), not ``python -I local_gate.py DIR``; `MODEL`'s registration under the period's circuit is
vy-admin's, made before the run (the run registers nothing); its key is ``lms.DEPLOYED``, pinned by ``--key-sha512``
(the auditor's ``public_sha512``, which the run checks against ``GATE pubkey``), not a test key made in a
temporary directory; serve.py on the auditor host reaches the serving node by its certifier coin link, the peer named
in ``/etc/vy/peers`` (``--peer``, the serving node), whose 127.0.0.1:PORT the window connects to, PORT the one the open
answers (7811..7899, `pouw_h1_window.at_port`), not 27811.. on the node itself; its coins are the Lean verifier's on
the auditor host; the challenge phase's end is read from ``GATE status`` (the live gate has no ``wait``); and no challenge phase is
forged (each would be a live window of the lane's period). The gate checks only that ``--circuit`` is a SHA-512 (128 hex
digits) and ``--circuit-files`` each K's (`auditor.circuit_files_of`), and records them; core's credit compares the
files with each statement's circuit file, and the descriptor with the window's layout. Never run anything on
the auditor host itself.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shlex
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
LOCAL_GATE = HERE / "local_gate.py"
SCHEMA = "verity-vllm/pouw-h1-rehearsal/v1"
RECORD = "rehearsal.json"


#: Llama-3.1-8B's linears in vLLM's module order, a layer's: (module, n, K)
LLAMA_8B = (("self_attn.qkv_proj", 6144, 4096), ("self_attn.o_proj", 4096, 4096), ("mlp.gate_up_proj", 28672, 4096),
            ("mlp.down_proj", 4096, 14336))
PROGRAM, PROGRAM_META = "window-program.json.gz", "window-program.meta.json"
#: the served model's int8 checkpoint role, the one the period opens with (``--model``) and core's credit pins
MODEL = "LLAMA31_8B_INT8"


def build_program(d: Path, layers: int = 32) -> dict[str, Any]:
    """The CPU phase: `pouw_h1.FILL`'s declared window Program over Llama-3.1-8B (`pouw_h1_served.declared`, no window's
    prefix), written as its descriptor's canonical JSON, gzipped (``d/PROGRAM``; the JSON's SHA-512 is its
    `program_digest`: the period's ``--circuit``), and its layout (``d/PROGRAM_META``), which the served window's must
    equal."""
    import gzip
    import resource
    import time
    from verity.ml.circuits.codec import canonical_json, encode_program, program_digest
    from verity_vllm.protocols.pouw import h1 as PH
    from verity_vllm.protocols.pouw.h1 import served as SV
    from tests.protocols.pouw_h1_gpu import layout_of
    d.mkdir(parents=True, exist_ok=True)
    shapes, modules = declaration(layers)
    t0 = time.monotonic()
    prog = SV.declared(shapes, modules, PH.FILL.rows, PH.FILL.steps)
    t1 = time.monotonic()
    desc = encode_program(prog)
    body = canonical_json({k: v for k, v in desc.items() if k != "annotations"})
    t2 = time.monotonic()
    (d / PROGRAM).write_bytes(gzip.compress(body, 6))
    digest = program_digest(desc)
    b = prog.fn.bindings
    meta = {"program_digest": digest, "file_sha512": hashlib.sha512(body).hexdigest(), "bytes": len(body),
            "calls": len(b["CALLS"]), "build_s": round(t1 - t0, 1), "encode_s": round(t2 - t1, 1),
            "peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2),
            "layout": layout_of(b["LINEARS"], b["CALLS"], b["M"], PH.TILE)}
    if meta["file_sha512"] != digest:
        raise ValueError(f"{PROGRAM}'s SHA-512 {meta['file_sha512'][:16]}... is not its program_digest {digest[:16]}...")
    (d / PROGRAM_META).write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def program_bytes(d: Path) -> bytes:
    """The declared window Program's descriptor bytes (``d/PROGRAM``, gunzipped): the period's circuit is their
    SHA-512, and core's credit reads the window's layout from them."""
    import gzip
    return gzip.decompress((d / PROGRAM).read_bytes())


def declaration(layers: int = 32) -> tuple[list[tuple[int, int]], dict[int, str]]:
    """`pouw_h1.FILL`'s window over Llama-3.1-8B as `pouw_h1_served.declared` takes it: its linears' shapes and names."""
    named = [(f"model.layers.{i}.{m}", n, k) for i in range(layers) for m, n, k in LLAMA_8B]
    return [(n, k) for _m, n, k in named], {w: m for w, (m, _n, _k) in enumerate(named)}


def declared_draw_k(layers: int = 32) -> int:
    """The draw K the period's open commits (``--draw-k``, ``draw_k``) for the declared window: the proof service's
    selection, `pouw_h1_challenger.draw_k` over `pouw_h1_served.declared_layout`, which `credit.strata` reads."""
    from verity_vllm.protocols.pouw import h1 as PH
    from verity_vllm.protocols.pouw.h1 import challenger as PHC
    from verity_vllm.protocols.pouw.h1 import served as SV
    shapes, modules = declaration(layers)
    return PHC.draw_k(SV.declared_layout(shapes, modules, PH.FILL.rows, PH.FILL.steps))


def fill_length() -> dict:
    """The fixed length `pouw_h1.FILL` serves (``--length``, `credit.LENGTH_FIELD`): each of its requests' public
    max_tokens, every one admitted at step 0 and run to its cap (``ignore_eos``, the public-cap done source)."""
    from verity_vllm.protocols.pouw import h1 as PH
    return {"max_tokens": [PH.FILL.max_tokens] * PH.FILL.slots, "stop": "fixed"}


def period_open_argv(name: str, cap: int, circuit: str, draw_k: int, model: str | None,
                     circuit_files: dict[str, str]) -> list[str]:
    """The gate's ``period-open`` of period `name`: its cap, its circuit (the declared Program's digest), its draw K,
    the served tables' rows T (``--rows``, `pouw_h1.FILL`'s), its model when it serves the registered weights
    (``--model``, whose registration's weights' root the open signs; never a ``--weights-root`` of the prover's), each
    stratum's circuit file's SHA-512 (``--circuit-files``, `pouw_h1_circuits.build`'s ``files``) and the fixed length
    its window serves (``--length``, `fill_length`)."""
    from verity_vllm.protocols.pouw import h1 as PH
    return ["period-open", "--name", name, "--cap", str(cap), "--circuit", circuit, "--draw-k", str(draw_k),
            "--rows", str(PH.FILL.rows), *(["--model", model] if model else []),
            "--circuit-files", json.dumps(circuit_files, sort_keys=True, separators=(",", ":")),
            "--length", json.dumps(fill_length(), sort_keys=True, separators=(",", ":"))]


class Gate:
    """The gate's verbs as ``argv + VERB``, each call and its answer kept in ``log``."""

    def __init__(self, argv: list[str]) -> None:
        self.argv, self.log = argv, []

    def raw(self, *args: str) -> tuple[int, bytes]:
        from tests.protocols.pouw_h1_gpu import gate_argv
        p = subprocess.run(gate_argv(self.argv, args), capture_output=True)
        return p.returncode, p.stdout + (p.stderr if p.returncode else b"")

    def __call__(self, *args: str) -> dict[str, Any]:
        rc, out = self.raw(*args)
        try:
            doc = json.loads(out)
        except ValueError:
            doc = {"refused": f"exit {rc}: {out[-2000:].decode(errors='replace')}"}
        self.log.append({"verb": list(args), "rc": rc,
                         "answer": {k: v for k, v in doc.items() if k not in ("secret", "key")}})
        return doc


def _poll_end(gate: Gate, session: str) -> dict[str, Any]:
    """The live gate has no ``wait``: read ``status`` until the challenge phase has its end entry (serve.py's exit after the
    window's close), at most an hour."""
    import time
    for _ in range(720):
        s = gate("status", session)
        if s.get("end") is not None or "refused" in s or not s.get("active"):
            return s
        time.sleep(5)
    return gate("status", session)


def register_model(gate: Gate, checkpoint: str, pdir: Path, circuit: str, scratch: Path) -> dict[str, Any]:
    """The local gate's registration of `MODEL` under the period's `circuit`, as vy-admin makes it on the auditor host
    (``admin model-register``, `auditor.model_register`): the pinned checkpoint in `checkpoint` committed under
    a fresh auditor key for the declared Program's descriptor (`program_bytes`, written to `scratch`). ``ok`` when the
    registry has it under the circuit; its answer carries the roots, never the key."""
    desc = scratch / "window-program.json"
    desc.write_bytes(program_bytes(pdir))
    doc = gate("admin", "model-register", "--model", MODEL, "--circuit", circuit, "--descriptor", str(desc),
               "--checkpoint", checkpoint)
    return {**doc, "ok": doc.get("circuit") == circuit and doc.get("model") == MODEL and "refused" not in doc}


def serve(a: argparse.Namespace, gate: Gate, reserved: dict[str, Any], out: Path, circuit: str | None = None
          ) -> dict[str, Any]:
    """The window in two phases, both logged to ``serve.log``: served on the GPU (under ``gpu-lease`` with
    ``--gpu-lease``), which hands its window and its challenge phase's stream to this process (`pouw_h1_handoff.Holder`) and
    exits, ending its lease; then closed and released on no GPU (``pouw_h1_gpu.py --close-from``) on that stream. The
    window's salt key, its saved state and each call's values are the prover's (`pouw_h1_prover`, DRAWN1): kept past the
    close in ``--prover-root``/SESSION, owner-only and outside the run's directory, under a ``research keep`` record."""
    import os
    import shutil
    import time
    from verity_vllm.protocols.pouw.h1 import handoff as HO
    from verity_vllm.protocols.pouw.h1 import prover as PV
    prover = PV.private(Path(a.prover_root) / reserved["session"])
    scratch = Path(tempfile.mkdtemp(prefix="h1-handoff-"))
    holder = HO.Holder(scratch / "stream.sock")
    gpu = [sys.executable, "-m", "tests.protocols.pouw_h1_gpu", "--workload", "fill", "--out", str(out)]
    argv = [*gpu, *(["--served", "--model", MODEL, "--circuit", circuit] if a.served else []),
            "--challenger", a.at_addr, "--session", json.dumps(reserved),
            "--gate", shlex.join(gate.argv), "--at", a.pin_at, "--by", a.pin_by, "--prover-dir", str(prover),
            "--handoff", str(holder.path), "--handoff-state", str(prover / PV.STATE)]
    if a.gpu_lease:
        argv = ["gpu-lease", "1", "--wait", "--max-min", str(a.gpu_lease), "--", *argv]
    doc: dict[str, Any] = {"serving_start_unix": round(time.time(), 3), "prover": {"directory": str(prover)}}
    if shutil.which("research"):
        keep = subprocess.run(["research", "keep", str(prover), "--owner", "@vllm", "--expires", "never", "--priority",
                               "keep", "--why", "an H1 window's salt key and values: a drawn unit opens from them (DRAWN1)"],
                              capture_output=True, text=True)
        doc["prover"]["keep"] = {"rc": keep.returncode, "out": (keep.stdout + keep.stderr)[-1000:]}
    try:
        with open(out.parent / "serve.log", "wb") as log:
            doc["serving_rc"] = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, cwd=HERE.parents[1]).returncode
            doc["serving_exit_unix"] = round(time.time(), 3)
            fd = holder.take()
            doc["stream_taken"] = fd is not None
            if fd is None:
                doc["holder"] = holder.error or "the serving stage handed no stream over"
                return {**doc, "rc": doc["serving_rc"] or 1}
            try:
                doc["close_rc"] = subprocess.run([*gpu, "--close-from", str(prover / PV.STATE), "--stream-fd", str(fd)],
                                                 stdout=log, stderr=subprocess.STDOUT, cwd=HERE.parents[1], pass_fds=(fd,),
                                                 env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}).returncode
            finally:
                os.close(fd)
            doc["close_exit_unix"] = round(time.time(), 3)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return {**doc, "rc": doc["serving_rc"] or doc["close_rc"]}


def key_custody(prover: Path, published: Path) -> dict[str, Any]:
    """The salt key past the close: kept in `prover` (owner-only), and neither its bytes nor its hex in any file under
    `published` (the run's ``$OUT``)."""
    import os
    import stat
    from verity_vllm.protocols.pouw.h1 import prover as PV
    path = prover / PV.KEY
    if not path.is_file():
        return {"kept": False, "ok": False}
    key = path.read_bytes()
    modes = {"directory": oct(stat.S_IMODE(os.stat(prover).st_mode)), "key": oct(stat.S_IMODE(os.stat(path).st_mode))}
    needles, found = (key, key.hex().encode()), []
    for f in sorted(p for p in published.rglob("*") if p.is_file()):
        with open(f, "rb") as fh:
            tail = b""
            while chunk := fh.read(1 << 24):
                if any(n in tail + chunk for n in needles):
                    found.append(str(f.relative_to(published)))
                    break
                tail = chunk[-128:]
    return {"kept": True, "modes": modes, "published_in": found,
            "ok": not found and modes == {"directory": "0o700", "key": "0o600"}}


def evidence(gate: Gate, session: str, d: Path) -> dict[str, Any]:
    """The ledger, the host's pins log, the key's pin and the challenge phase's tree as the gate gives them, in `d`."""
    d.mkdir(parents=True, exist_ok=True)
    rc, ledger = gate.raw("ledger")
    (d / "windows.jsonl").write_bytes(ledger if rc == 0 else b"")
    prc, pins = gate.raw("pins")
    (d / "pins.jsonl").write_bytes(pins if prc == 0 else b"")
    pub = gate("pubkey")
    (d / "pubkey.json").write_text(json.dumps(pub, indent=1) + "\n")
    rc, tgz = gate.raw("fetch", session)
    if rc == 0:
        with tarfile.open(fileobj=io.BytesIO(tgz)) as t:
            t.extractall(d / "fetch", filter="data")
    return {"ledger_lines": ledger.count(b"\n") if rc == 0 else 0, "pins_lines": pins.count(b"\n") if prc == 0 else 0,
            "pubkey": pub, "fetched": rc == 0}


def check(calls: Path, tree: Path, pin: str, ledger: list[Path], out: Path, window: dict | None = None,
          period: str | None = None, descriptor: bytes | None = None, pins: list[Path] | None = None,
          main: list[str] | None = None) -> dict[str, Any]:
    """`pouw_h1_gpu.check_challenger`, the verifier's side: with `window` on the challenge phase's ``out/``
    (`trusted_rounds_reason`), else on its tree (`session_reason`), under the host's pins log's heads `pins` and the
    verifier's own first-parent history of main `main`; its credit over `descriptor`, the period's circuit's bytes
    (`program_bytes`)."""
    from tests.protocols import pouw_h1_gpu as GPU
    wfile = None
    if window is not None:
        out.mkdir(parents=True, exist_ok=True)
        wfile = out / "window.json"
        wfile.write_text(json.dumps(window) + "\n")
    doc = GPU.check_challenger(str(calls), str(tree / "out" if window is not None else tree), pin, str(out),
                               [str(p) for p in ledger], None if wfile is None else str(wfile), period, descriptor,
                               [str(p) for p in pins or ()], main)
    return {k: doc.get(k) for k in ("ok", "reason", "manifest", "period", "credit")}


def _edited(record: dict, edit: Any, path: Path) -> Path:
    doc = json.loads(json.dumps(record))
    edit(doc)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc) + "\n")
    return path


def forge(gate: Gate, honest: dict[str, Any], opened_on: bytes, peer: str, at: str) -> tuple[dict[str, Any], str]:
    """`honest` (the window's commitment) with its coins swapped for a challenge phase the gate reserves and opens on
    `opened_on`, its salt opened after `opened_on` and its round and draw coins after the window's own roots, as
    `test_pouw_h1._salted` forges one in process: (the forged commitment, its challenge phase)."""
    from verity.core.protocols.pouw import window as Wd
    from verity_vllm.protocols.pouw.h1 import challenger as PHC
    from verity_vllm.protocols.pouw.h1 import requests as RQ
    from verity_vllm.protocols.pouw.h1 import window as PHW
    reserved = gate("reserve", "--peer", peer, "--scheme", PHC.SCHEME, "--families", PHC.families(len(honest["rounds"])),
                    "--minutes", "30")
    if "refused" in reserved:
        raise ValueError(f"reserve: {reserved['refused']}")
    answer = gate("open", "--session", reserved["session"], "--root", opened_on.hex())
    if "refused" in answer:
        raise ValueError(f"open: {answer['refused']}")
    if answer.get("commitment") is None:
        answer["commitment"] = gate("status", reserved["session"]).get("commitment")
    client = PHW.connect(PHW.at_port(at, answer.get("port")), answer)
    salt = client.open(Wd.SALT, 0, opened_on)
    log = [{**r, "coin": client.open(Wd.ROUND, s, bytes.fromhex(r["root"])).to_json()} for s, r in enumerate(honest["rounds"])]
    root = RQ.draw_root(bytes.fromhex(honest["calls_root"]), bytes.fromhex(honest["draw"]["receipts"]), log)
    draw = client.open(Wd.DRAW, 0, root)
    client.close()
    coins = client.commitment.to_json()
    return ({**honest, "coins": coins, "round_coins": coins, "salt_opening": salt.to_json(), "rounds": log,
             "session": PHC.carried(answer), "draw": {**honest["draw"], "root": root.hex(), "coin": draw.to_json()}},
            reserved["session"])


def red_team(out: Path, pin: str, period: str, gate: Gate | None, peer: str, at: str, circuit: str = "00" * 64,
             draw_k: int = 1, model: str | None = None, circuit_files: dict[str, str] | None = None,
             main: list[str] | None = None) -> dict[str, Any]:
    """Each case the verifier's side must refuse, on edited copies of the run's record and ledger (CPU only), and with
    `gate` (the local gate's), challenge phases forged through it; `main` is the verifier's first-parent history of main."""
    from verity.core.protocols.pouw import salt as PS
    record_path = out / "serve" / "pouw-h1-calls.json"
    record = json.loads(record_path.read_text())
    c = record["commitment"]
    sid, R = c["session"]["session"], c["session"]["root"]
    weights, table = c["weights_root"], c["requests_root"]
    other = PS.window_open_root(bytes.fromhex(weights), bytes.fromhex(table), "s20261021T000000Z-0000").hex()
    g = out / "gate"
    tree, ledger, pins = g / "fetch" / sid, [g / "windows.jsonl"], [g / "pins.jsonl"]
    rt = out / "red_team"
    cases: dict[str, Any] = {}

    def case(name: str, rec: Path, *, window: dict | None = None, led: list[Path] | None = None, tr: Path | None = None,
             pn: list[Path] | None = None, mn: Any = main) -> None:
        got = check(rec, tr or tree, pin, led or ledger, rt / name, window, period, pins=pins if pn is None else pn,
                    main=mn)
        cases[name] = {**got, "refused": not got["ok"]}

    case("r_bare_weights_root", _edited(record, lambda d: d["commitment"]["session"].update(root=weights), rt / "bare.json"))
    case("r_bare_weights_root_window", record_path, window={"root": weights})
    case("r_other_window_root", _edited(record, lambda d: d["commitment"]["session"].update(root=other), rt / "other.json"))
    case("r_other_window_root_window", record_path, window={"root": other})
    lines = (g / "windows.jsonl").read_bytes().split(b"\n")
    end = next(i for i, ln in enumerate(lines) if ln and json.loads(ln)["kind"] == "end" and json.loads(ln)["session"] == sid)
    cut = rt / "ledger-cut.jsonl"
    rt.mkdir(parents=True, exist_ok=True)
    cut.write_bytes(b"".join(ln + b"\n" for ln in lines[:end]))
    case("ledger_cut_before_end", record_path, led=[cut])
    case("pins_none", record_path, pn=[])
    (rt / "pins-cut.jsonl").write_bytes(b"")
    case("pins_cut", record_path, pn=[rt / "pins-cut.jsonl"])
    logged = [ln for ln in (g / "pins.jsonl").read_text().split("\n") if ln]
    if logged:
        from verity.core.protocols.sampled_proofs import challenger as LC
        e = json.loads(logged[0])
        logged[0] = LC.pin_line(e["commit"], e["main"], "2000-01-01T00:00:00Z")
    (rt / "pins-rewritten.jsonl").write_text("".join(ln + "\n" for ln in logged))
    case("pins_rewritten", record_path, pn=[rt / "pins-rewritten.jsonl"])
    case("main_none", record_path, mn=None)
    case("main_without_pin", record_path, mn=[c for c in main or () if c not in {json.loads(ln)["commit"] for ln in logged}])
    first = next(i for i, x in enumerate(record["calls"]) if x.get("tables"))
    case("table_root_changed", _edited(record, lambda d: d["calls"][first]["tables"].update(a="00" * 64), rt / "table.json"))
    case("table_a_root_unbound", _edited(record, lambda d: d["calls"][first]["tables"].update(a=None), rt / "unbound.json"))
    case("table_rounds_cut", _edited(record, lambda d: d["commitment"]["rounds"].pop(), rt / "rounds-cut.json"))
    got = check(record_path, tree, "ab" * 64, ledger, rt / "other_pin", None, period, pins=pins, main=main)
    cases["other_pin"] = {**got, "refused": not got["ok"]}
    if gate is not None:
        forged_period = f"{period}-forged"
        bare = gate(*period_open_argv(f"{period}-bare", 1, circuit, draw_k, None, circuit_files or {}),
                    "--weights-root", other)
        cases["r_period_open_bare_weights_root"] = {"answer": bare, "refused": "refused" in bare}
        cases["forged_period_open"] = gate(*period_open_argv(forged_period, 2, circuit, draw_k, model,
                                                             circuit_files or {}))
        forged = {}
        for name, root in (("forged_bare_weights_root", bytes.fromhex(weights)), ("forged_other_window_root", bytes.fromhex(other))):
            try:
                forged[name] = forge(gate, c, root, peer, at)
            except Exception as e:  # noqa: BLE001
                cases[name] = {"refused": True, "at": "forge", "reason": f"{type(e).__name__}: {e}"}
        for name, (_c, s) in forged.items():
            gate("wait", s)
        cases["forged_period_close"] = gate("period-close", "--name", forged_period)
        fe = evidence(gate, sid, rt / "forged_gate")
        cases["forged_gate"] = fe
        for name, (commits, s) in forged.items():
            rc, tgz = gate.raw("fetch", s)
            with tarfile.open(fileobj=io.BytesIO(tgz)) as t:
                t.extractall(rt / "forged_gate" / "fetch", filter="data")
            rec = _edited(record, lambda d, commits=commits: d.update(commitment=commits), rt / f"{name}.json")
            case(name, rec, led=[rt / "forged_gate" / "windows.jsonl"], tr=rt / "forged_gate" / "fetch" / s,
                 pn=[rt / "forged_gate" / "pins.jsonl"])
        case("honest_under_later_head", record_path, led=[rt / "forged_gate" / "windows.jsonl"],
             pn=[rt / "forged_gate" / "pins.jsonl"])
        cases["honest_under_later_head"]["refused"] = not cases["honest_under_later_head"]["ok"]
    must = [k for k in cases if k.startswith(("r_", "ledger_", "pins_", "main_", "table_", "other_pin", "forged_b",
                                              "forged_o"))]
    return {"cases": cases, "all_refused": all(cases[k].get("refused") for k in must), "must_refuse": must}


def rehearse(a: argparse.Namespace) -> dict[str, Any]:
    from verity_vllm.protocols.pouw import h1 as PH
    from verity_vllm.protocols.pouw.h1 import challenger as PHC
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    doc: dict[str, Any] = {"schema": SCHEMA, "argv": sys.argv, "period": a.period, "live": a.gate is not None, "served": a.served}
    main = _main_of(a)
    pdir = Path(a.program) if a.program else out / "program"
    doc["program"] = (json.loads((pdir / PROGRAM_META).read_text()) if a.program else build_program(pdir))
    circuit = doc["program"]["program_digest"]
    doc["draw_k"] = draw_k = declared_draw_k()
    if a.gate is None:
        gdir = Path(tempfile.mkdtemp(prefix="h1-gate-")) / "gate"
        gdir.mkdir()
        gate = Gate([sys.executable, "-I", str(LOCAL_GATE), str(gdir)])
        rc, init = gate.raw("init", "--tree", a.tree, "--commit", a.commit, "--peer", a.peer, "--coins", a.coins)
        doc["gate"] = json.loads(init) if rc == 0 else {"init_failed": init[-3000:].decode(errors="replace")}
        if rc:
            return doc
        doc["gate"]["lean_build"] = json.loads((gdir / "gate.json").read_text()).get("lean_build")
        pin = doc["gate"]["pubkey"]["public_sha512"]
        lo = doc["gate"]["ports"][0]
        a.at_addr = a.at or f"127.0.0.1:{lo}"
    else:
        gate = Gate(shlex.split(a.gate))
        pin = a.key_sha512
        a.at_addr = a.at
        if gate("pubkey").get("public_sha512") != pin:
            doc["refused"] = "the gate's pubkey is not the pinned key (--key-sha512)"
            return doc
    doc["key_sha512"] = pin
    model = MODEL if a.served else None
    if a.served and a.gate is None:
        doc["model"] = register_model(gate, a.checkpoint, pdir, circuit, gdir.parent)
        if not doc["model"]["ok"]:
            doc["gate_log"] = gate.log
            return doc
    from tests.protocols import pouw_h1_circuits as PC
    try:
        doc["circuits"] = PC.build(pdir, Path(a.circuits) if a.circuits else out / "circuits")
    except (OSError, ValueError, subprocess.CalledProcessError) as e:
        doc["circuits"] = {"refused": f"{type(e).__name__}: {e}"}
        return doc
    if doc["circuits"]["program"] != circuit:
        doc["circuits"]["refused"] = (f"the circuit files are bound to Program {doc['circuits']['program'][:16]}..., not "
                                      f"{circuit[:16]}..., the period's")
        return doc
    files = doc["circuits"]["files"]
    doc["period_open"] = gate(*period_open_argv(a.period, 1, circuit, draw_k, model, files))
    if a.served and not (doc["period_open"].get("body") or {}).get("weights_root"):
        doc["gate_log"] = gate.log
        doc["refused"] = (f"the period's open signs no weights' root: the gate has no registration of {MODEL} under "
                          f"circuit {circuit[:16]}... (vy-admin's model-register)")
        return doc
    reserved = gate("reserve", "--peer", a.peer, "--scheme", PHC.SCHEME, "--families", PHC.families(PH.FILL.steps),
                    "--minutes", str(a.minutes))
    doc["reserved"] = reserved
    if "refused" in reserved:
        doc["gate_log"] = gate.log
        return doc
    sid = reserved["session"]
    doc["stages"] = serve(a, gate, reserved, out / "serve", circuit)
    doc["serve_rc"] = doc["stages"]["rc"]
    doc["key_custody"] = key_custody(Path(doc["stages"]["prover"]["directory"]), out)
    status = gate("status", sid)
    if status.get("open") is not None:
        doc["unit"] = gate("wait", sid) if a.gate is None else _poll_end(gate, sid)
    doc["period_close"] = gate("period-close", "--name", a.period)
    doc["evidence"] = evidence(gate, sid, out / "gate")
    try:
        serve_doc = json.loads((out / "serve" / "pouw_h1_gpu.json").read_text())
        h = serve_doc["h1"]
        st = doc["stages"]
        if h.get("served_end_unix") and st.get("serving_exit_unix"):
            st["lease_after_served_s"] = round(st["serving_exit_unix"] - h["served_end_unix"], 1)
        doc["serve"] = {"ok": serve_doc["ok"], "u": h.get("u"), "same_tokens": h.get("same_tokens"),
                        "decode_step_s_median": h.get("decode_step_s_median"), "close_s": h.get("close_s"),
                        "handoff": h.get("handoff"), "a_roots": (h.get("requests") or {}).get("a_roots"),
                        "gate": h.get("gate"), "step_rows": h.get("step_rows"),
                        "z_exact": [z.get("y_exact") for z in h.get("z") or []],
                        "challenger": h.get("challenger"), "served": {k: v for k, v in (h.get("served") or {}).items()
                                                                      if k != "entries"},
                        "served_values": [(e["value"], e["leaves"], e["words"]) for e in (h.get("served") or {}).get("entries", [])]}
        if a.served:
            got = (h.get("served") or {}).get("layout")
            doc["serve"]["declared"] = {"program_digest": circuit, "layout_matches": got == doc["program"]["layout"],
                                        **({} if got == doc["program"]["layout"] else {"served_layout": got})}
    except (OSError, ValueError, KeyError) as e:
        doc["serve"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    calls = out / "serve" / "pouw-h1-calls.json"
    if calls.is_file() and doc["evidence"]["fetched"]:
        g = out / "gate"
        tree, ledger, pins = g / "fetch" / sid, [g / "windows.jsonl"], [g / "pins.jsonl"]
        record = json.loads(calls.read_text())
        desc, root = program_bytes(pdir), record["commitment"]["session"]["root"]
        doc["verify"] = {
            "session": check(calls, tree, pin, ledger, out / "verify" / "session", None, a.period, desc, pins, main),
            "trusted": check(calls, tree, pin, ledger, out / "verify" / "trusted", {"root": root}, None, desc, pins,
                             main),
        }
        doc["red_team"] = red_team(out, pin, a.period, gate if a.gate is None else None, a.peer, a.at_addr, circuit,
                                   draw_k, model, files, main)
    doc["gate_log"] = gate.log
    v = doc.get("verify") or {}
    cr = (v.get("trusted") or {}).get("credit") or {}
    doc["credit"] = {"gate_passed": ((doc.get("serve") or {}).get("gate") or {}).get("gate_passed"),
                     **{k: cr.get(k) for k in ("credit", "calls", "manifest", "window", "k", "eps")}}
    doc["ok"] = bool(doc.get("serve", {}).get("ok") and (v.get("session") or {}).get("ok") and (v.get("trusted") or {}).get("ok")
                     and str(doc["serve"].get("a_roots")).startswith("bound") and doc["key_custody"]["ok"]
                     and (doc.get("red_team") or {}).get("all_refused")
                     and (doc["serve"].get("declared") or {}).get("layout_matches", not a.served))
    return doc


def _main_of(a: argparse.Namespace) -> list[str] | None:
    """The verifier's own first-parent history of main at ``--main`` (`remote.main_lines`), or None without one."""
    from verity.core.protocols.sampled_proofs import remote
    return None if a.main is None else remote.main_lines(a.main)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--gate", help="the live gate's command (ssh ... research@vy-challenger-1); default: a local gate of --tree")
    ap.add_argument("--tree", help="infra's tree (git archive COMMIT of local_gate.ARCHIVE's paths, gzipped)")
    ap.add_argument("--commit", help="the commit --tree is (the local gate's pinned commit)")
    ap.add_argument("--coins", choices=("lean", "stub"), default="lean")
    ap.add_argument("--peer", default="local", help="the serving node's name in the gate's peers")
    ap.add_argument("--at", help="the challenger's HOST:PORT before the open answers its port (live: 127.0.0.1:7811)")
    ap.add_argument("--key-sha512", help="live: the auditor's pinned public_sha512")
    ap.add_argument("--main", metavar="FILE", help="the verifier's own `git rev-list --first-parent origin/main`, a commit a "
                                                  "line, which must hold the session's pin; without it every session is "
                                                  "refused")
    ap.add_argument("--period", default="vllm-h1-rehearsal")
    ap.add_argument("--no-served", dest="served", action="store_false",
                    help="serve without pouw_h1_served.Served (no A and B trees, no registration)")
    ap.add_argument("--checkpoint", metavar="DIR", help="the local gate with --served: the pinned LLAMA31_8B_INT8 snapshot "
                                                        "the rehearsal registers as vy-admin (model-register)")
    ap.add_argument("--program-only", action="store_true", help="CPU only: build the declared window Program into --out "
                                                               "(build_program) and stop")
    ap.add_argument("--program", metavar="DIR", help="a --program-only run's out/ (its Program's digest is the period's "
                                                    "--circuit), not built again")
    ap.add_argument("--circuits", metavar="DIR", help="each stratum's circuit file, built (pouw_h1_circuits --program "
                                                     "--out DIR), not built again; default: built into --out/circuits")
    ap.add_argument("--gpu-lease", type=int, metavar="MIN", help="serve under `gpu-lease 1 --wait --max-min MIN` (in a "
                                                                "research run without --gpus: the GPU held only while it serves)")
    ap.add_argument("--minutes", type=int, default=300, help="the reserve's minutes, which also bound the session")
    ap.add_argument("--pin-at", default="2026-10-21T00:00:00Z", help="pouw_h1_gpu.py's --at (resolved_at, if a pin is made)")
    ap.add_argument("--pin-by", default="vllm", help="pouw_h1_gpu.py's --by")
    ap.add_argument("--prover-root", default="/workspace/vllm-h1-prover", help="the prover's private directories, one a "
                                                                               "session (pouw_h1_prover), outside --out")
    ap.add_argument("--red-team", metavar="RUN_OUT", help="CPU only: rerun the red team's edited copies on a run's fetched out/ "
                                                         "(its serve/, gate/ and rehearsal.json), into --out")
    a = ap.parse_args(argv)
    if a.red_team:
        src = Path(a.red_team)
        done = json.loads((src / RECORD).read_text())
        out = Path(a.out)
        if out.resolve() != src.resolve():
            import shutil
            for d in ("serve", "gate"):
                shutil.copytree(src / d, out / d, dirs_exist_ok=True)
        rt = red_team(out, done["key_sha512"], done["period"], None, a.peer, a.at or "127.0.0.1:1",
                      (done.get("program") or {}).get("program_digest", "00" * 64), main=_main_of(a))
        Path(out, "red_team.json").write_text(json.dumps(rt, indent=1, default=str) + "\n")
        print(f"POUW-H1-RED-TEAM all_refused={rt['all_refused']} "
              + " ".join(f"{k}={'refused' if v.get('refused') else 'ACCEPTED'}" for k, v in rt["cases"].items()), flush=True)
        return 0 if rt["all_refused"] else 1
    if a.program_only:
        meta = build_program(Path(a.out))
        print(f"POUW-H1-PROGRAM digest={meta['program_digest']} calls={meta['calls']} bytes={meta['bytes']} "
              f"build_s={meta['build_s']} encode_s={meta['encode_s']} peak_rss_gib={meta['peak_rss_gib']}", flush=True)
        return 0
    if a.gate is None and not (a.tree and a.commit):
        ap.error("the local gate needs --tree and --commit")
    if a.gate is None and a.served and not a.checkpoint:
        ap.error("the local gate with --served needs --checkpoint (the snapshot its model-register commits)")
    if a.gate is not None and not (a.key_sha512 and a.at):
        ap.error("the live gate needs --key-sha512 (the auditor's pin) and --at (127.0.0.1:7811 on the peer)")
    doc = rehearse(a)
    Path(a.out, RECORD).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    rt = doc.get("red_team") or {}
    print(f"POUW-H1-REHEARSAL {'OK' if doc.get('ok') else 'FAIL'} serve={doc.get('serve', {}).get('ok')} "
          f"session={(doc.get('verify') or {}).get('session')} trusted={(doc.get('verify') or {}).get('trusted')} "
          f"red_team_all_refused={rt.get('all_refused')} key_custody={(doc.get('key_custody') or {}).get('ok')} "
          f"credit={json.dumps(doc.get('credit'), default=str)}", flush=True)
    return 0 if doc.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
