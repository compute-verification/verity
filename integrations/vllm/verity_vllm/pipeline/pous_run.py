"""`verity-vllm pous`: POUS end to end on one GPU, each role in its own process.

    verity-vllm pous e2e --out DIR --case QWEN05 --build-dir DIR [--codec NAME] [--audits N] [--drop-fraction F] [--result FILE]

runs the reference lifecycle (`protocols/pous/PROTOCOL.md`) of the reference codec ``--codec`` (`engine/pous/twins.reference`:
``p3/v1``, the default, ``dense-chain/v1`` or ``band-chain/d12/v1``), at the geometry it is served at, in every role:

1. `commit` (server): `verity_pous.setup.commit` of `W`, the checkpoint's data section (`engine/pous/layout.py`);
2. `vk` (verifier): `setup.draw_salt` from its own secret after the commitment, then its own reference encoding of `W`
   (`pipeline/pous_verifier.encode`, per segment) and `vk`; it also decodes layer 0's tensors with the reference
   `layout.decode_tensor`;
3. `plain`: the plaintext model under the integration's engine (`engine.vllm_adapter.build_engine`, case `QWEN05`), greedy, with
   logprobs;
4. `serve`: the same engine with `quantization="pous"` (`engine/pous/quant.py`): `C` encoded on the GPU under the salt, every
   weight decoded per layer at every use; the same generation; the server's leaves and root; then the native responder
   (`engine/pous/worker.py`) and a serving loop until the verifier is done;
5. `verify`: a separate verifier process checks the server's root against `vk` (`setup.check_server_root`), calibrates its round
   trip to the responder, and runs the reference's timed audits (`pipeline/pous_verifier.run`);
6. the negative control: `serve` with a fraction of `C`'s blocks dropped, and `verify` against it.

`pous_e2e.json` in DIR (and `result.json` under `research run`) carries the verdicts.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from verity_vllm.config import option

PROMPTS = ["The capital of France is", "def fibonacci(n):", "Proof of useful space means", "1, 1, 2, 3, 5, 8,"]
GPU_MEMORY_UTILIZATION = 0.6
LAYER0 = "model.layers.0."


def _write(path: Path, doc: Any) -> None:
    path.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str))


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def reference_codec(name: str):
    """The reference codec ``name`` and the geometry it is served at (`engine/pous/twins.reference`, no torch)."""
    from verity_vllm.engine.pous import twins

    return twins.reference(name)


def checkpoint_files(case: str) -> list[str]:
    """The pinned checkpoint's `.safetensors` files, in name order (the shard order)."""
    from verity_vllm import config
    from verity_vllm.engine import engine_profile as prof

    cp = prof.checkpoint_for(case, config.CHECKPOINTS)
    return [str(Path(cp["local_path"]) / name) for name in sorted(cp["safetensors_sha256"])]


def build(case: str, pous_spec: Path | None, max_num_seqs: int = 8, engine_args: dict[str, Any] | None = None):
    from verity_vllm import config
    from verity_vllm.engine import vllm_adapter as va

    args: dict[str, Any] = {"gpu_memory_utilization": GPU_MEMORY_UTILIZATION, **(engine_args or {})}
    if pous_spec is not None:
        from verity_vllm.engine.pous import quant, worker  # noqa: F401  (registers quantization="pous")

        args.update(quantization="pous", worker_extension_cls=worker.EXTENSION,
                    hf_overrides={"quantization_config_file": str(pous_spec)})
    return va.build_engine(config.CHECKPOINTS, max_num_seqs=max_num_seqs, case=case, engine_args=args)


def generate(engine, max_tokens: int) -> list[dict[str, Any]]:
    from vllm import SamplingParams

    t0 = time.perf_counter()
    outs = engine.llm.generate(PROMPTS, SamplingParams(temperature=0.0, max_tokens=max_tokens, logprobs=5), use_tqdm=False)
    wall = time.perf_counter() - t0
    rows = []
    for o in outs:
        c = o.outputs[0]
        lps = [{str(t): lp.logprob.hex() for t, lp in sorted(step.items())} for step in (c.logprobs or [])]
        rows.append({"prompt": o.prompt, "token_ids": list(c.token_ids), "text": c.text, "logprobs": lps})
    return [{"wall_s": wall}] + rows


def cmd_commit(a) -> None:
    from verity_pous import setup

    from verity_vllm.engine.pous import layout

    ref, base = reference_codec(a.codec)
    ck = layout.read(checkpoint_files(a.case), base)
    commitment = setup.commit(ck.params, ck.read_w())
    _write(Path(a.out) / "commit.json", {"commitment": commitment.hex(), "codec": ref.name, "params": ck.params.manifest(),
                                         "checkpoint": [x.path for x in ck.sections]})


def cmd_vk(a) -> None:
    from verity_pous import setup
    from verity_pous.primitives import reference

    from verity_vllm.engine.pous import layout
    from verity_vllm.pipeline import pous_verifier as pv

    out = Path(a.out)
    committed = _read(out / "commit.json")
    commitment = bytes.fromhex(committed["commitment"])
    ref_codec, base = reference_codec(a.codec)
    if committed.get("codec", ref_codec.name) != ref_codec.name:
        raise SystemExit(f"the server committed under {committed['codec']}, not {ref_codec.name}")
    salt = setup.draw_salt(os.urandom(32), commitment)
    ck = layout.read(checkpoint_files(a.case), base)
    w = ck.read_w()
    if setup.commit(ck.params, w) != commitment:
        raise SystemExit("W does not match the server's commitment")
    _write(out / "salt.json", {"salt": salt.hex()})
    t0 = time.perf_counter()
    segments = pv.encode(ck.params, w, salt, a.workers, ref_codec)
    vk = setup.verifier_key(ck.params, reference(ck.params), w, commitment, salt, encoded=segments, codec=ref_codec)
    enc_s = time.perf_counter() - t0
    (out / "vk-leaves.bin").write_bytes(b"".join(vk.leaves))
    t1 = time.perf_counter()
    names = sorted(n for n in ck.dtypes if n.startswith(LAYER0))
    ref = {n: hashlib.sha256(b).hexdigest()
           for n, b in pv.reference_decode(ck.layout, segments, salt, names, a.workers, ref_codec).items()}
    plain = {n: hashlib.sha256(ck.layout.unpack(w, n)).hexdigest() for n in names}
    _write(out / "vk.json", {"salt": salt.hex(), "root": vk.root.hex(), "params": ck.params.manifest(), "codec": ref_codec.name,
                             "encode_s": enc_s,
                             "workers": a.workers, "reference_decode_s": time.perf_counter() - t1,
                             "reference_decode_layer0": ref, "reference_decode_matches_w": ref == plain})


def load_vk(out: Path):
    from verity_pous.verifier import VerifierKey

    doc = _read(out / "vk.json")
    _, base = reference_codec(doc["codec"])
    params = base.with_segments(doc["params"]["segments"])
    if params.manifest() != doc["params"]:
        raise ValueError(f"vk.json's params {doc['params']} are not {doc['codec']}'s served geometry {params.manifest()}")
    raw = (out / "vk-leaves.bin").read_bytes()
    vk = VerifierKey.from_leaves(params, bytes.fromhex(doc["salt"]), [raw[i:i + 32] for i in range(0, len(raw), 32)])
    if vk.root.hex() != doc["root"]:
        raise ValueError("vk-leaves.bin does not have the root vk.json records")
    return doc, vk


def cmd_plain(a) -> None:
    engine = build(a.case, None)
    _write(Path(a.out) / "plain.json", {"outputs": generate(engine, a.max_tokens)})


def serve_spec(a, out: Path) -> dict[str, Any]:
    """The `engine/pous/quant.py` spec of the served store: the codec, the checkpoint, the verifier's salt, the drop."""
    return {"codec": reference_codec(a.codec)[0].name, "safetensors": checkpoint_files(a.case), "build_dir": a.build_dir,
            "salt": _read(out / "salt.json")["salt"], "drop_fraction": a.drop_fraction, "drop_source": os.urandom(32).hex(),
            "layers_per_decode": a.layers_per_decode}


def cmd_serve(a) -> None:
    from verity_pous.verifier import MerkleTree

    reserved = {c for c in (a.responder_cpu, a.verifier_cpu) if c is not None}
    if reserved:
        os.sched_setaffinity(0, set(os.sched_getaffinity(0)) - reserved)
    out = Path(a.out)
    spec = serve_spec(a, out)
    spec_path = out / f"pous-{a.tag}.json"
    _write(spec_path, spec)
    t0 = time.perf_counter()
    engine = build(a.case, spec_path)
    llm = engine.llm
    doc: dict[str, Any] = {"build_s": time.perf_counter() - t0, "spec": spec, "info_after_load": llm.collective_rpc("pous_info")[0]}
    if not a.drop_fraction:
        doc["outputs"] = generate(engine, a.max_tokens)
    raw = llm.collective_rpc("pous_leaves")[0]
    (out / f"leaves-{a.tag}.bin").write_bytes(raw)
    doc["root"] = MerkleTree([raw[i:i + 32] for i in range(0, len(raw), 32)]).root.hex()
    port = llm.collective_rpc("pous_start_responder", args=("127.0.0.1", 0, -1 if a.responder_cpu is None else a.responder_cpu))[0]
    _write(out / f"ready-{a.tag}.json", {"port": port, "pid": os.getpid(), "root": doc["root"]})
    stop = out / f"stop-{a.tag}"
    loops = []
    from vllm import SamplingParams

    while not stop.exists():
        t = time.perf_counter()
        llm.generate(PROMPTS, SamplingParams(temperature=0.0, max_tokens=a.load_tokens), use_tqdm=False)
        loops.append(time.perf_counter() - t)
    doc["responder"] = llm.collective_rpc("pous_stop_responder")[0]
    doc["serving_loops_during_audit_s"] = loops
    doc["info_end"] = llm.collective_rpc("pous_info")[0]
    _write(out / f"serve-{a.tag}.json", doc)


def cmd_verify(a) -> None:
    from verity_pous import setup

    from verity_vllm.pipeline import pous_verifier as pv

    out = Path(a.out)
    _, vk = load_vk(out)
    ready = _read(out / f"ready-{a.tag}.json")
    served = (out / f"leaves-{a.tag}.bin").read_bytes()
    doc: dict[str, Any] = {"server_root_matches_vk": setup.check_server_root(vk, bytes.fromhex(ready["root"])),
                           "blocks_differing": sum(served[32 * i:32 * i + 32] != leaf for i, leaf in enumerate(vk.leaves))}
    try:
        doc.update(pv.run("127.0.0.1", ready["port"], vk, audits=a.audits, pings=a.pings, cpu=a.verifier_cpu, allowance=a.allowance))
    finally:
        (out / f"stop-{a.tag}").write_text("")
    _write(out / f"verify-{a.tag}.json", doc)


def _sub(args: list[str], log: Path) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-m", "verity_vllm.pipeline.cli", "pous", *args], stdout=log.open("w"),
                            stderr=subprocess.STDOUT)


def _step(out: Path, name: str, args: list[str]) -> None:
    rc = _sub(args, out / f"{name}.log").wait()
    if rc != 0:
        raise SystemExit(f"{name} failed with {rc}; see {out / (name + '.log')}")


def _base(a) -> list[str]:
    return ["--out", a.out, "--case", a.case, "--build-dir", a.build_dir, "--codec", a.codec]


def _audit_round(out: Path, tag: str, extra: list[str], a) -> dict[str, Any]:
    common = [*_base(a), "--tag", tag]
    cpus = []
    for flag, cpu in (("--verifier-cpu", a.verifier_cpu), ("--responder-cpu", a.responder_cpu)):
        if cpu is not None:
            cpus += [flag, str(cpu)]
    srv = _sub(["serve", *common, "--max-tokens", str(a.max_tokens), "--load-tokens", str(a.load_tokens),
                "--layers-per-decode", str(a.layers_per_decode), *cpus, *extra], out / f"serve-{tag}.log")
    t0 = time.time()
    while not (out / f"ready-{tag}.json").exists():
        if srv.poll() is not None or time.time() - t0 > a.timeout:
            srv.kill()
            return {"serve_rc": srv.returncode, "verify_rc": None, "serve": None, "verify": None}
        time.sleep(2)
    pin = [] if a.verifier_cpu is None else ["--verifier-cpu", str(a.verifier_cpu)]
    rc_v = _sub(["verify", *common, "--audits", str(a.audits), "--pings", str(a.pings), "--allowance", a.allowance, *pin],
                out / f"verify-{tag}.log").wait()
    rc_s = srv.wait(timeout=a.timeout)
    load = lambda p: _read(p) if p.exists() else None  # noqa: E731
    return {"serve_rc": rc_s, "verify_rc": rc_v, "serve": load(out / f"serve-{tag}.json"), "verify": load(out / f"verify-{tag}.json")}


def compare(plain: list[dict], pous: list[dict]) -> dict[str, Any]:
    p, q = plain[1:], pous[1:]
    same = len(p) == len(q) and len(p) > 0
    return {"requests": len(p), "token_ids_equal": same and all(x["token_ids"] == y["token_ids"] for x, y in zip(p, q)),
            "logprobs_equal": same and all(x["logprobs"] == y["logprobs"] for x, y in zip(p, q)),
            "tokens": [len(x["token_ids"]) for x in q]}


def summarize(out: Path, honest: dict, negative: dict, a) -> dict[str, Any]:
    plain = _read(out / "plain.json")["outputs"]
    vk = _read(out / "vk.json")
    hs, hv, nv = honest["serve"] or {}, honest["verify"] or {}, negative["verify"] or {}
    load_info = hs.get("info_after_load") or {}
    exact = compare(plain, hs.get("outputs") or [{}])
    served = load_info.get("digests") or {}
    ref = vk["reference_decode_layer0"]
    keys = ("audit_of_record", "audits", "audits_accepted", "answers", "late", "wrong", "server_root_matches_vk", "blocks_differing")
    result = {
        "kind": "pous-vllm-e2e/v1", "model": a.case, "codec": load_info.get("codec"),
        "setup": {"commit": _read(out / "commit.json"), "salt": vk["salt"], "vk_root": vk["root"], "verifier_encode_s": vk["encode_s"],
                  "server_encode_s": load_info.get("encode_s"), "server_root": hs.get("root")},
        "bit_exact": {**exact, "load_checks": load_info.get("checks"),
                      "layer0_decoded_by_gpu": sorted(n for n in ref if n in served),
                      "gpu_decode_equals_reference_decode_layer0": bool(set(ref) & set(served))
                      and all(served[n] == d for n, d in ref.items() if n in served),
                      "reference_decode_matches_w": vk["reference_decode_matches_w"]},
        "in_process_responder": {"device_s": hv.get("device_s"), "server_s": hv.get("server_s"),
                                 "device_over_deadline": hv.get("device_over_deadline"), "counts": hs.get("responder")},
        "verifier_round_trip_s": {"before": hv.get("rtt"), "after": hv.get("rtt_after")},
        "deadline": {k: hv.get(k) for k in ("deadline_ns", "allowance", "rtt_allowance_ns", "limit_ns", "k",
                                            "audits_accepted_under_allowance")},
        "honest": {**{k: hv.get(k) for k in keys}, "elapsed_s": hv.get("elapsed_s")},
        "negative": {"drop_fraction": a.drop_fraction or 0.1, "dropped_blocks": ((negative["serve"] or {}).get("info_after_load") or {}).get("dropped"),
                     **{k: nv.get(k) for k in keys}},
        "timing": {"plain_generate_s": plain[0]["wall_s"], "pous_generate_s": (hs.get("outputs") or [{}])[0].get("wall_s"),
                   "max_tokens": a.max_tokens, "decode": {k: (hs.get("info_end") or {}).get(k) for k in ("decodes", "decode_s", "decoded_bytes")},
                   "serving_loops_during_audit_s": hs.get("serving_loops_during_audit_s")},
        "rc": {"honest": [honest["serve_rc"], honest["verify_rc"]], "dropped": [negative["serve_rc"], negative["verify_rc"]]},
    }
    rec = hv.get("audit_of_record") or {}
    result["verdicts"] = {
        "outputs_bit_exact": bool(exact["token_ids_equal"] and exact["logprobs_equal"]),
        "load_checks_passed": bool(load_info.get("checks")) and all(load_info["checks"].values()),
        "gpu_encode_equals_reference": bool(hv.get("server_root_matches_vk")),
        "gpu_decode_equals_reference_layer0": result["bit_exact"]["gpu_decode_equals_reference_decode_layer0"],
        "responder_device_within_deadline": hv.get("device_over_deadline") == 0 and bool(hv.get("answers")),
        "honest_audit_accepted": bool(rec.get("accepted")),
        "negative_audit_rejected": bool(nv.get("audit_of_record")) and not nv["audit_of_record"]["accepted"],
    }
    return result


def research_result(result: dict[str, Any], a, passed: bool) -> dict[str, Any]:
    """`result.json` of `tools/research` (`research/result/v0.1`); the details stay in `pous/pous_e2e.json`."""
    rtt, el = result["verifier_round_trip_s"]["before"] or {}, result["honest"]["elapsed_s"] or {}
    dev = result["in_process_responder"]["device_s"] or {}
    rows = [("rtt_p50", rtt.get("p50"), "s"), ("rtt_p99.9", rtt.get("p99.9"), "s"), ("rtt_max", rtt.get("max"), "s"),
            ("answer_p50", el.get("p50"), "s"), ("answer_p99.9", el.get("p99.9"), "s"), ("answer_max", el.get("max"), "s"),
            ("device_max", dev.get("max"), "s"), ("limit", (result["deadline"]["limit_ns"] or 0) / 1e9, "s"),
            ("server_encode", result["setup"]["server_encode_s"], "s"), ("verifier_encode", result["setup"]["verifier_encode_s"], "s"),
            ("plain_generate", result["timing"]["plain_generate_s"], "s"), ("pous_generate", result["timing"]["pous_generate_s"], "s"),
            ("honest_audits_accepted", result["honest"]["audits_accepted"], "count"),
            ("negative_audits_accepted", result["negative"]["audits_accepted"], "count")]
    failed = sorted(k for k, v in result["verdicts"].items() if not v)
    return {"schema": "research/result/v0.1", "run_id": a.run_id,
            "workload_fingerprint": {"case": a.case, "codec": result["codec"], "salt": result["setup"]["salt"],
                                     "vk_root": result["setup"]["vk_root"], "commitment": result["setup"]["commit"]["commitment"]},
            "validation": {"status": "passed" if passed else "failed", "checker": "verity-vllm pous e2e",
                           "detail": "all verdicts hold" if passed else "failed: " + ", ".join(failed),
                           "evidence": ["pous/pous_e2e.json"]},
            "measurements": [{"name": n, "value": v, "unit": u} for n, v, u in rows if v is not None],
            "measurement_files": ["pous/pous_e2e.json"], "artifacts": ["pous/"]}


def cmd_e2e(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    _step(out, "commit", ["commit", *_base(a)])
    _step(out, "vk", ["vk", *_base(a), "--workers", str(a.workers)])
    _step(out, "plain", ["plain", *_base(a), "--max-tokens", str(a.max_tokens)])
    honest = _audit_round(out, "honest", [], a)
    negative = _audit_round(out, "dropped", ["--drop-fraction", str(a.drop_fraction or 0.1)], a)
    result = summarize(out, honest, negative, a)
    result["wall_s"] = time.time() - t0
    _write(out / "pous_e2e.json", result)
    passed = all(result["verdicts"].values())
    if a.result:
        _write(Path(a.result), research_result(result, a, passed))
    print(json.dumps(result["verdicts"]), flush=True)
    return 0 if passed else 1


@dataclasses.dataclass(frozen=True, kw_only=True)
class _Common:
    out: str = option("--out", required=True, help="the run directory every role reads and writes")
    case: str = option("--case", required=True, help="the checkpoint role in manifests/checkpoints.json")
    build_dir: str = option("--build-dir", required=True, help="parent directory of the pous_native JIT build")
    codec: str = option("--codec", default="p3/v1",
                        help="the reference codec served and audited (engine/pous/twins.reference): p3/v1, dense-chain/v1, band-chain/d12/v1")
    tag: str = option("--tag", default="honest", help="the server round: honest or dropped")
    max_tokens: int = option("--max-tokens", type=int, default=6, help="tokens per prompt of the compared generation")
    load_tokens: int = option("--load-tokens", type=int, default=2, help="tokens per prompt of the serving loop during the audit")
    drop_fraction: float = option("--drop-fraction", type=float, default=0.0, help="the fraction of C the server drops")
    audits: int = option("--audits", type=int, default=20, help="timed audits against each server; the first is of record")
    pings: int = option("--pings", type=int, default=2000, help="round-trip calibration pings before and after the audits")
    workers: int = option("--workers", type=int, default=64, help="processes of the verifier's reference encoding")
    timeout: float = option("--timeout", type=float, default=3600.0, help="seconds a server may take to become ready")
    verifier_cpu: int | None = option("--verifier-cpu", type=int, default=None, help="pin the timed audit to this core")
    responder_cpu: int | None = option("--responder-cpu", type=int, default=None,
                                       help="pin the native responder to this core; the server's other threads avoid it and the verifier's")
    layers_per_decode: int = option("--layers-per-decode", type=int, default=1,
                                    help="transformer layers decoded together at their first use in a forward pass")
    result: str | None = option("--result", default=None, help="also write the research result.json here (e2e)")
    run_id: str | None = option("--run-id", default=None, help="the research run id result.json names (e2e)")
    allowance: str = option("--allowance", default="p99.9", choices=["p50", "p99.9", "max"],
                            help="the calibration quantile of the verifier's round trip added to the 0.5 ms deadline")


def _cmd(name: str, doc: str):
    cls = dataclasses.make_dataclass(f"{name.title()}Options", [("cmd", str, dataclasses.field(default=name, init=False))],
                                     bases=(_Common,), frozen=True, kw_only=True)
    cls.__doc__ = doc
    return cls


COMMANDS = {
    "commit": _cmd("commit", "server: commit to W before any salt exists"),
    "vk": _cmd("vk", "verifier: draw the salt, encode W with the reference, keep vk"),
    "plain": _cmd("plain", "the plaintext model's generation"),
    "serve": _cmd("serve", "server: quantization='pous', the generation, then the native responder under serving load"),
    "verify": _cmd("verify", "verifier: root check, round-trip calibration, timed audits"),
    "e2e": _cmd("e2e", "every role in order, the negative control, and the verdicts"),
}


def main(a) -> int:
    return {"commit": cmd_commit, "vk": cmd_vk, "plain": cmd_plain, "serve": cmd_serve, "verify": cmd_verify, "e2e": cmd_e2e}[a.cmd](a) or 0
