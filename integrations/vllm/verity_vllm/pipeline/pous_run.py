"""`verity-vllm pous`: POUS end to end on one GPU through the protocol option (`protocol_options.pous`), each role in its own
process.

    verity-vllm pous e2e --out DIR --case QWEN05 --build-dir DIR [--scheme ID] [--audits N] [--drop-fraction F] [--result FILE]

runs the protocol's lifecycle (`protocols/pous/PROTOCOL.md`) for the scheme ``--scheme`` (a `verity_pous.schemes` id with a GPU
twin, `engine/pous/twins.py`; ``band-chain/d12/v1`` by default; ``--codec`` is the same flag), in every role:

1. `plain` (the server's plaintext): the model under the integration's engine (`engine.vllm_adapter.build_engine`, case
   `QWEN05`), greedy, with logprobs; and `W` of record as the option lays it out (`pous.weights_of_record`: every linear weight,
   deduplicated), with the commitment to it (`protocol.commit`), before any salt exists;
2. `vk` (verifier): the salt drawn from its own secret after the commitment (`protocol.draw_salt`), its own reference encoding of
   `W` (`pipeline/pous_verifier.encode`, per segment) and `vk` (`protocol.verifier_key`); it also decodes layer 0's tensors with
   the reference;
3. `serve`: the same engine with ``protocols = pous:<scheme>`` and ``codec: gpu``: `C` encoded on the GPU under the verifier's
   salt by the scheme's twin once it reproduced its pinned vector, every weight decoded on the GPU at every use; the same
   generation; the server's leaves and root; the native responder on its own core and port; a serving loop until the verifier
   is done;
4. `verify`: a separate verifier process checks the server's root against `vk` (`setup.check_server_root`), calibrates its round
   trip to the responder, and runs the protocol's timed audits (`pipeline/pous_verifier.run`);
5. the negative control: `serve` with a fraction of `C`'s blocks dropped, and `verify` against it.

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
MAX_BATCHED_TOKENS = 16384      # pous-bench's scheduler budget: a 4 x 2048-token prefill is one forward
DECODE_PROMPT = 128             # pous-bench's decode prompt
PLAIN_DECODE_TOKENS = 64        # the plaintext decode steps averaged, as pous-bench


def _write(path: Path, doc: Any) -> None:
    path.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str))


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def geometry(name: str):
    """The scheme's served geometry before `W` sizes it (`verity_pous.schemes`), the one `protocol_options.pous` lays `W` out at."""
    from verity_pous import schemes

    return schemes.get(name).params


def build(case: str, protocols: str | None = None, protocols_config: dict[str, Any] | None = None, max_num_seqs: int = 8,
          engine_args: dict[str, Any] | None = None):
    """The engine of the case, and the installed composition when ``protocols`` is set (as `verity_vllm.LLM` installs it)."""
    from verity_vllm import config
    from verity_vllm.engine import vllm_adapter as va
    from verity_vllm.pipeline import llm as L

    ctx = L.protocol_context(L.resolve_options({"protocols": protocols, "protocols_config": protocols_config or {}})) if protocols else None
    args: dict[str, Any] = {"gpu_memory_utilization": GPU_MEMORY_UTILIZATION, **(engine_args or {})}
    engine = va.build_engine(config.CHECKPOINTS, max_num_seqs=max_num_seqs, case=case, engine_args=args,
                             execution=va.execution_label_of(True))
    if ctx is None:
        return engine, None
    from verity_vllm import protocol_options as PO
    return engine, PO.install(engine.model, ctx)


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


def record_w(scheme: str, model: Any, out: Path) -> dict[str, Any]:
    """`W` of record (`pous.weights_of_record`) into ``w.bin`` and its commitment into ``commit.json``: what the server commits to
    before any salt exists, and what the verifier encodes."""
    from verity_vllm.protocol_options import pous

    layout, sch, w, commitment = pous.weights_of_record(scheme, model)
    (out / "w.bin").write_bytes(w)
    doc = {"commitment": commitment.hex(), "scheme": sch.name, "params": sch.params.manifest(),
           "tensors": [[t.name, t.size] for t in layout.tensors], "w_sha256": hashlib.sha256(w).hexdigest()}
    _write(out / "commit.json", doc)
    return doc


def committed_w(out: Path) -> tuple[dict[str, Any], Any, Any, bytes]:
    """``commit.json``, the layout it names, the scheme at its size, and ``w.bin`` checked against it."""
    from verity_pous import protocol
    from verity_pous.layout import Layout

    from verity_vllm.pipeline import pous_verifier as pv

    doc = _read(out / "commit.json")
    layout = Layout.of(geometry(doc["scheme"]), [(n, int(size)) for n, size in doc["tensors"]])
    scheme = pv.scheme_at(doc["scheme"], layout.params)
    w = (out / "w.bin").read_bytes()
    if layout.params.manifest() != doc["params"] or protocol.commit(scheme, w).hex() != doc["commitment"]:
        raise SystemExit("w.bin and its layout do not give commit.json's parameters and commitment")
    return doc, layout, scheme, w


def cmd_vk(a) -> None:
    from verity_pous import protocol

    from verity_vllm.pipeline import pous_verifier as pv

    out = Path(a.out)
    doc, layout, scheme, w = committed_w(out)
    if doc["scheme"] != a.scheme:
        raise SystemExit(f"the server committed under {doc['scheme']}, not {a.scheme}")
    commitment = bytes.fromhex(doc["commitment"])
    salt = protocol.draw_salt(os.urandom(32), commitment)
    _write(out / "salt.json", {"salt": salt.hex()})
    t0 = time.perf_counter()
    segments = pv.encode(scheme, w, salt, a.workers)
    vk = pv.verifier_key(scheme, commitment, salt, w, segments)
    enc_s = time.perf_counter() - t0
    (out / "vk-leaves.bin").write_bytes(b"".join(vk.leaves))
    t1 = time.perf_counter()
    names = [t.name for t in layout.tensors if t.name.startswith(LAYER0)]
    ref = {n: hashlib.sha256(b).hexdigest() for n, b in pv.reference_decode(scheme, layout, segments, salt, names, a.workers).items()}
    plain = {n: hashlib.sha256(layout.unpack(w, n)).hexdigest() for n in names}
    _write(out / "vk.json", {"salt": salt.hex(), "root": vk.root.hex(), "params": scheme.params.manifest(), "scheme": scheme.name,
                             "encode_s": enc_s, "workers": a.workers, "reference_decode_s": time.perf_counter() - t1,
                             "reference_decode_layer0": ref, "reference_decode_matches_w": ref == plain})


def load_vk(out: Path):
    from verity_pous.verifier import VerifierKey

    from verity_vllm.pipeline import pous_verifier as pv

    doc = _read(out / "vk.json")
    params = geometry(doc["scheme"]).with_segments(doc["params"]["segments"])
    if params.manifest() != doc["params"]:
        raise ValueError(f"vk.json's params {doc['params']} are not {doc['scheme']}'s served geometry {params.manifest()}")
    raw = (out / "vk-leaves.bin").read_bytes()
    vk = VerifierKey.from_leaves(params, bytes.fromhex(doc["salt"]), [raw[i:i + 32] for i in range(0, len(raw), 32)])
    if vk.root.hex() != doc["root"]:
        raise ValueError("vk-leaves.bin does not have the root vk.json records")
    return doc, pv.scheme_at(doc["scheme"], params), vk


def engine_args(a) -> dict[str, Any] | None:
    """The scheduler budget of the timed configurations, the same for the plain and the served engine."""
    return {"max_num_batched_tokens": MAX_BATCHED_TOKENS} if a.prefill else None


def throughput(engine, a, info) -> list[dict[str, Any]]:
    """pous-bench's prefill configurations (``--prefill``) and decode at batch 1 on this role's engine (`pipeline/pous_timing.py`):
    prefill tokens/s over the time to every first token, decode tokens/s over the steps after it.  A POUS forward pays every
    layer's decode, so the served engine times ``--decode-tokens`` steps and the plain one `PLAIN_DECODE_TOKENS`."""
    from verity_vllm.pipeline import pous_timing as pb

    rows = []
    for item in filter(None, a.prefill.split(",")):
        length, batch = (int(x) for x in item.split("x"))
        rows.append(pb.prefill(engine, length, batch, info))
    if a.decode_tokens:
        rows.append(pb.decode(engine, 1, DECODE_PROMPT, a.decode_tokens if info() is not None else PLAIN_DECODE_TOKENS, info))
    return rows


def slowdowns(plain: list[dict[str, Any]], pous: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Plain tokens/s over POUS tokens/s, per configuration both roles timed."""
    key = lambda r: f"prefill {r['length']}x{r['batch']}" if r["kind"] == "prefill" else f"decode batch {r['batch']}"  # noqa: E731
    served = {key(r): r for r in pous}
    return {key(p): {"plain_tokens_per_s": p["tokens_per_s"], "pous_tokens_per_s": served[key(p)]["tokens_per_s"],
                     "slowdown": p["tokens_per_s"] / served[key(p)]["tokens_per_s"],
                     "pous_decode_s": served[key(p)].get("decode_s"), "pous_decodes": served[key(p)].get("decodes")}
            for p in plain if key(p) in served}


def cmd_plain(a) -> None:
    out = Path(a.out)
    engine, _ = build(a.case, engine_args=engine_args(a))
    record_w(a.scheme, engine.model, out)
    doc = {"outputs": generate(engine, a.max_tokens)}
    if a.prefill or a.decode_tokens:
        doc["throughput"] = throughput(engine, a, lambda: None)
    _write(out / "plain.json", doc)


def serve_config(a, out: Path) -> dict[str, Any]:
    """The option's settings for the served store: the GPU codec, the verifier's salt, the native responder, the drop."""
    own: dict[str, Any] = {"codec": "gpu", "build_dir": a.build_dir, "salt": _read(out / "salt.json")["salt"], "host": "127.0.0.1",
                           "port": 0, "drop_fraction": a.drop_fraction, "drop_source": os.urandom(32).hex()}
    if a.kernels:
        own["kernels"] = json.loads(a.kernels)
    if a.responder_cpu is not None:
        own["responder_cpu"] = a.responder_cpu
    return {"pous": own}


def _stats(opt) -> dict[str, Any]:
    return dict(opt.info()["stats"], dropped=opt.info()["dropped"])


def cmd_serve(a) -> None:
    reserved = {c for c in (a.responder_cpu, a.verifier_cpu) if c is not None}
    if reserved:
        os.sched_setaffinity(0, set(os.sched_getaffinity(0)) - reserved)
    out = Path(a.out)
    settings = serve_config(a, out)
    t0 = time.perf_counter()
    engine, comp = build(a.case, f"pous:{a.scheme}", settings, engine_args=engine_args(a))
    opt = comp.get("pous")
    names = [t.name for t in opt.store.layout.tensors if t.name.startswith(LAYER0)]
    doc: dict[str, Any] = {"build_s": time.perf_counter() - t0, "settings": {k: v for k, v in settings["pous"].items() if k != "drop_source"},
                           "commitment": opt.commitment(), "info_after_load": {**_stats(opt), "scheme": a.scheme,
                           "digests": {n: hashlib.sha256(opt.store.decode(n)).hexdigest() for n in names}}}
    if not a.drop_fraction:
        doc["outputs"] = generate(engine, a.max_tokens)
        if a.prefill or a.decode_tokens:
            doc["throughput"] = throughput(engine, a, lambda: _stats(opt))
    (out / f"leaves-{a.tag}.bin").write_bytes(opt.leaves())
    doc["root"] = opt.commitment()["storage_root"]
    _write(out / f"ready-{a.tag}.json", {"port": opt.info()["responder"]["port"], "pid": os.getpid(), "root": doc["root"]})
    stop = out / f"stop-{a.tag}"
    loops = []
    from vllm import SamplingParams

    while not stop.exists():
        t = time.perf_counter()
        engine.llm.generate(PROMPTS, SamplingParams(temperature=0.0, max_tokens=a.load_tokens), use_tqdm=False)
        loops.append(time.perf_counter() - t)
    doc["info_end"] = _stats(opt)
    comp.close()
    doc["responder"] = opt.responder.result
    doc["serving_loops_during_audit_s"] = loops
    _write(out / f"serve-{a.tag}.json", doc)


def cmd_verify(a) -> None:
    from verity_pous import setup

    from verity_vllm.pipeline import pous_verifier as pv

    out = Path(a.out)
    _, scheme, vk = load_vk(out)
    ready = _read(out / f"ready-{a.tag}.json")
    served = (out / f"leaves-{a.tag}.bin").read_bytes()
    doc: dict[str, Any] = {"server_root_matches_vk": setup.check_server_root(vk, bytes.fromhex(ready["root"])),
                           "blocks_differing": sum(served[32 * i:32 * i + 32] != leaf for i, leaf in enumerate(vk.leaves))}
    try:
        doc.update(pv.run("127.0.0.1", ready["port"], scheme, vk, audits=a.audits, pings=a.pings, cpu=a.verifier_cpu,
                          allowance=a.allowance))
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
    return ["--out", a.out, "--case", a.case, "--build-dir", a.build_dir, "--scheme", a.scheme, "--prefill", a.prefill,
            "--decode-tokens", str(a.decode_tokens), "--kernels", a.kernels]


def _audit_round(out: Path, tag: str, extra: list[str], a) -> dict[str, Any]:
    common = [*_base(a), "--tag", tag]
    cpus = []
    for flag, cpu in (("--verifier-cpu", a.verifier_cpu), ("--responder-cpu", a.responder_cpu)):
        if cpu is not None:
            cpus += [flag, str(cpu)]
    srv = _sub(["serve", *common, "--max-tokens", str(a.max_tokens), "--load-tokens", str(a.load_tokens), *cpus, *extra],
               out / f"serve-{tag}.log")
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
    keys = ("audit_of_record", "audits", "audits_accepted", "answers", "late", "wrong", "server_root_matches_vk", "blocks_differing",
            "k", "research_mode", "gaps", "profile")
    result = {
        "kind": "pous-vllm-e2e/v2", "model": a.case, "scheme": load_info.get("scheme"),
        "setup": {"commit": _read(out / "commit.json"), "salt": vk["salt"], "vk_root": vk["root"], "verifier_encode_s": vk["encode_s"],
                  "server_encode_s": load_info.get("encode_s"), "server_root": hs.get("root"), "gate": load_info.get("gate")},
        "bit_exact": {**exact, "layer0_decoded_by_gpu": sorted(n for n in ref if n in served),
                      "gpu_decode_equals_reference_decode_layer0": bool(ref) and set(ref) <= set(served)
                      and all(served[n] == d for n, d in ref.items()),
                      "reference_decode_matches_w": vk["reference_decode_matches_w"]},
        "native_responder": {"device_s": hv.get("device_s"), "server_s": hv.get("server_s"),
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
        "throughput": slowdowns(_read(out / "plain.json").get("throughput") or [], hs.get("throughput") or []),
        "rc": {"honest": [honest["serve_rc"], honest["verify_rc"]], "dropped": [negative["serve_rc"], negative["verify_rc"]]},
    }
    rec = hv.get("audit_of_record") or {}
    result["verdicts"] = {
        "outputs_bit_exact": bool(exact["token_ids_equal"] and exact["logprobs_equal"]),
        "twin_reproduced_pinned_vector": bool((load_info.get("gate") or {}).get("digest")),
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
    dev = result["native_responder"]["device_s"] or {}
    rows = [("rtt_p50", rtt.get("p50"), "s"), ("rtt_p99.9", rtt.get("p99.9"), "s"), ("rtt_max", rtt.get("max"), "s"),
            ("answer_p50", el.get("p50"), "s"), ("answer_p99.9", el.get("p99.9"), "s"), ("answer_max", el.get("max"), "s"),
            ("device_max", dev.get("max"), "s"), ("limit", (result["deadline"]["limit_ns"] or 0) / 1e9, "s"),
            ("server_encode", result["setup"]["server_encode_s"], "s"), ("verifier_encode", result["setup"]["verifier_encode_s"], "s"),
            ("plain_generate", result["timing"]["plain_generate_s"], "s"), ("pous_generate", result["timing"]["pous_generate_s"], "s"),
            ("honest_audits_accepted", result["honest"]["audits_accepted"], "count"),
            ("negative_audits_accepted", result["negative"]["audits_accepted"], "count")]
    for config, s in (result.get("throughput") or {}).items():
        tag = config.replace(" ", "_")
        rows += [(f"slowdown_{tag}", s["slowdown"], "x"), (f"plain_{tag}", s["plain_tokens_per_s"], "token/s"),
                 (f"pous_{tag}", s["pous_tokens_per_s"], "token/s")]
    failed = sorted(k for k, v in result["verdicts"].items() if not v)
    return {"schema": "research/result/v0.1", "run_id": a.run_id,
            "workload_fingerprint": {"case": a.case, "scheme": result["scheme"], "salt": result["setup"]["salt"],
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
    _step(out, "plain", ["plain", *_base(a), "--max-tokens", str(a.max_tokens)])
    _step(out, "vk", ["vk", *_base(a), "--workers", str(a.workers)])
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
    prefill: str = option("--prefill", default="",
                          help="prefill configurations the plain and honest serve roles time, length x batch, comma separated (e.g. 2048x4)")
    decode_tokens: int = option("--decode-tokens", type=int, default=0,
                                help="decode steps the honest serve role times at batch 1 (plain: 64); 0 times no decode")
    scheme: str = option("--scheme", "--codec", default="band-chain/d12/v1",
                         help="the verity_pous scheme served and audited, one with a GPU twin (engine/pous/twins.py)")
    kernels: str = option("--kernels", default="", help="the served kernels' dispatch settings as JSON (engine/pous/twins.KERNEL_FLAGS), "
                          "e.g. the band's run of record's")
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
    "plain": _cmd("plain", "the server's plaintext: the generation, and W of record with its commitment before any salt"),
    "vk": _cmd("vk", "verifier: draw the salt, encode W with the reference, keep vk"),
    "serve": _cmd("serve", "server: protocols pous:<scheme> on the GPU codec, the generation, then the native responder under serving load"),
    "verify": _cmd("verify", "verifier: root check, round-trip calibration, timed audits"),
    "e2e": _cmd("e2e", "every role in order, the negative control, and the verdicts"),
}


def main(a) -> int:
    return {"plain": cmd_plain, "vk": cmd_vk, "serve": cmd_serve, "verify": cmd_verify, "e2e": cmd_e2e}[a.cmd](a) or 0
