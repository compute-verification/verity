"""GLM-5.3's DSA indexer logits at a draft step's six query rows a request on an H200: DeepGEMM's `fp8_fp4_paged_mqa_logits` as vLLM
d9105ea80 launches it when a decode step carries 1 + 5 tokens a request, against `dsa.DsaMqaLogit_v3{H=32, D=128}`.

At the pin the indexer passes next_n query rows a request to the kernel only where `_supports_native_decode(next_n)` holds
(v1/attention/backends/mla/indexer.py:672-682); on cc 9.0 that is `native_next_n_supported`, next_n in {1, 2, 4}
(utils/deep_gemm.py:560-568), so next_n = 6 is flattened (`_use_flattening`, indexer.py:685-692): `_prepare_decode_tensors`
(indexer.py:874-905) runs `_PREPARE_UNIFORM_DECODE_KERNEL` (indexer.py:72-119), which gives each request's six rows the context
lengths max(L - 6 + j + 1, 0), j = 0..5, its block-table row each, and decode length 1; seq_lens is unsqueezed to [6R, 1]
(indexer.py:1258-1262), the schedule metadata built from it (indexer.py:1266-1275, no `indices`: varlen is cc 10.x only), and
`sparse_attn_indexer` calls `fp8_fp4_paged_mqa_logits((q [6R, 1, 32, 128], None), cache, weights [6R, 32], seq_lens, block_table,
schedule, max_model_len, clean_logits=False, indices=None)` (sparse_attn_indexer.py:563-612).  The kernel so runs at next_n = 1 over
6R rows, the launch `glm53_dsa_capture_gpu`'s decode case runs at 8 rows; here the rows come from vLLM's own flattening.

One GPU process writes `<out>/glm53_mtp_indexer_logits_capture.json`, the property record `glm53_mtp_indexer_logits_capture`
(`properties.record`), and each launch's inputs and compared words under `<out>/words/`.

  gate     `fa2_target_capture_gpu.gate`: driver, cc 9.0, 132 SMs; nothing else runs when it fails (exit 3).
  served   vLLM's switches (`glm53_dsa_capture_gpu.served`): DeepGEMM on and importable, not batch invariant; `_use_flattening`
           true for 5 speculative tokens and `native_next_n_supported(6)` false on this device.
  flatten  `_PREPARE_UNIFORM_DECODE_KERNEL` on each launch's requests: its seq_lens equal max(L - 5 + j, 0) and its block table
           each request's row six times.
  launch   per case of CASES, R requests (a padding request of length 0 among them) over a cache of e4m3 key rows and f32 scales
           (the indexer cache's layout, `glm53_dsa_capture_gpu.slot_bytes`) in shuffled blocks, e4m3 queries and f32 weights;
           twice, every written word equal.
  words    each logit word [0, seq_len) of the FULL rows and of RANDOM_PAIRS random (row, key) pairs against the twin
           (`glm53_dsa_capture_gpu.logit_rows`, of the row's q and w and the key's bytes and scale) and the reference evaluator
           of `DsaMqaLogit_v3{32,128}` on two entries a job.
  bind     DsaMqaLogit_v3{32,128} binds.

GO = all of it, no error, and every compared word resolved (these operands keep every logit finite).

    cd adapters/vllm; PY=/workspace/venv312/bin/python     # PYTHONPATH as pod_bootstrap.sh sets it
    $PY tests/properties/glm53_mtp_indexer_logits_capture_gpu.py --out DIR --scratch /tmp
"""
from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

import argparse
import json
import multiprocessing
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from tests.properties import fa2_target_capture_gpu as C
from tests.properties import fa_tap_exactness_gpu as G
from tests.properties import glm53_dsa_capture_gpu as DS
from verity_vllm.properties.record import seal as seal_record

SCHEMA = "verity-vllm/glm53-mtp-indexer-logits-capture/v1"
RECORD = "glm53_mtp_indexer_logits_capture.json"
H, D, BLOCK, STRIDE = DS.H, DS.D, DS.BLOCK, DS.STRIDE
SPEC, NEXT_N = 5, 6
MAX_MODEL_LEN = 4096
#: (name, request lengths L; 0 a padding request, as cudagraph padding leaves one, and None a random length in [NEXT_N, MAX])
CASES = (("edges", (6, 7, 64, 65, 129, 2048, 2049, MAX_MODEL_LEN, 0)),
         ("batch", (None,) * 63 + (0,)))
FULL_REQUESTS, RANDOM_PAIRS = 9, 50_000
TOTALS = DS.TOTALS


# ------------------------------------------------------------------------------------------------------------------- operands
def lengths(spec: tuple[int | None, ...], rng: np.random.Generator) -> np.ndarray:
    return np.array([int(rng.integers(NEXT_N, MAX_MODEL_LEN + 1)) if L is None else L for L in spec], dtype=np.int64)


def operands(L: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """The cache, block table, queries and weights for requests of lengths L: each request's ceil(L / BLOCK) blocks picked at random
    from the cache's (one spare block first), unused table entries 0; key rows e4m3 of N(0, 1) * 2^U(-2, 2) with scales
    2^U(-10, -6); queries e4m3 of N(0, 1) * 2^U(-2, 2); weights N(0, 1) * 2^-4 in f32."""
    import torch

    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    need = [-(-int(n) // BLOCK) for n in L]
    nb = 1 + sum(need)
    perm = 1 + rng.permutation(nb - 1)
    bt = np.zeros((len(L), MAX_MODEL_LEN // BLOCK), dtype=np.int32)
    i = 0
    for r, n in enumerate(need):
        bt[r, :n] = perm[i:i + n]
        i += n
    vals = torch.randn(nb, BLOCK, D, generator=gen) * torch.exp2(torch.empty(nb, BLOCK, 1).uniform_(-2, 2, generator=gen))
    kq = vals.to(torch.float8_e4m3fn).view(torch.uint8).numpy()
    ks = np.exp2(rng.uniform(-10, -6, (nb, BLOCK))).astype(np.float32).view(np.uint8).reshape(nb, BLOCK * 4)
    cache = np.concatenate([kq.reshape(nb, BLOCK * D), ks], axis=1).reshape(nb, BLOCK, STRIDE)
    rows = NEXT_N * len(L)
    qv = torch.randn(rows, H, D, generator=gen) * torch.exp2(torch.empty(rows, 1, 1).uniform_(-2, 2, generator=gen))
    q = qv.to(torch.float8_e4m3fn).view(torch.uint8).numpy()
    w = (rng.standard_normal((rows, H)) * 2.0 ** -4).astype(np.float32).view(np.uint32)
    return {"L": L, "cache": cache, "bt": bt, "q": q, "w": w}


def flattened(L: np.ndarray, bt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """What `_PREPARE_UNIFORM_DECODE_KERNEL` should give: each row's context length and block-table row."""
    j = np.tile(np.arange(NEXT_N), len(L))
    lens = np.maximum(np.repeat(L, NEXT_N) - NEXT_N + j + 1, 0).astype(np.int32)
    return lens, np.repeat(bt, NEXT_N, axis=0)


def key_table(o: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every request's keys 0..L-1 one after another, read out of the cache bytes by the indexer cache's layout: (key bytes
    [sum L, D], scale words [sum L], each request's first key's index)."""
    flat = np.ascontiguousarray(o["cache"])
    k, s = [], []
    for r, n in enumerate(o["L"]):
        for t in range(int(n)):
            q, w = DS.slot_bytes(flat, int(o["bt"][r, t // BLOCK]) * BLOCK + t % BLOCK)
            k.append(q)
            s.append(w)
    first = np.concatenate([[0], np.cumsum(o["L"])[:-1]]).astype(np.int64)
    return np.asarray(k, dtype=np.uint8).reshape(-1, D), np.asarray(s, dtype=np.uint32), first


# ------------------------------------------------------------------------------------------------------------------------ gpu
def selection() -> dict[str, Any]:
    from vllm.utils.deep_gemm import native_next_n_supported
    from vllm.v1.attention.backends.mla import indexer as IX

    rec = {"native_next_n_6": bool(native_next_n_supported(NEXT_N)), "supports_native_decode_6": bool(IX._supports_native_decode(NEXT_N)),
           "supports_varlen": bool(IX._supports_varlen_paged_mqa_logits())}
    rec["ok"] = not rec["native_next_n_6"] and not rec["supports_native_decode_6"] and not rec["supports_varlen"]
    return rec


def launch(outdir: Path, name: str, o: dict[str, np.ndarray], seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """vLLM's flattening and the paged logits twice; the record and the comparison jobs."""
    import torch
    from vllm.utils.deep_gemm import fp8_fp4_paged_mqa_logits, get_paged_mqa_logits_metadata
    from vllm.utils.platform_utils import num_compute_units
    from vllm.v1.attention.backends.mla import indexer as IX

    dev = "cuda"
    R, rows = len(o["L"]), NEXT_N * len(o["L"])
    seq_lens = torch.from_numpy(o["L"].astype(np.int32)).to(dev)
    bt = torch.from_numpy(o["bt"]).to(dev)
    dec_lens_buf = torch.zeros(rows, dtype=torch.int32, device=dev)
    lens_buf = torch.zeros(rows, dtype=torch.int32, device=dev)
    bt_buf = torch.zeros(rows, bt.shape[1], dtype=torch.int32, device=dev)
    IX._PREPARE_UNIFORM_DECODE_KERNEL(seq_lens, lens_buf, bt, bt_buf, dec_lens_buf, rows, NEXT_N)
    want_lens, want_bt = flattened(o["L"], o["bt"])
    got_lens, got_bt = lens_buf.cpu().numpy(), bt_buf.cpu().numpy()
    lens2 = lens_buf.unsqueeze(-1)
    cache = torch.from_numpy(o["cache"]).to(dev)
    q = torch.from_numpy(o["q"]).to(dev).view(torch.float8_e4m3fn).view(rows, 1, H, D)
    w = torch.from_numpy(o["w"].view(np.float32)).to(dev)
    sched = get_paged_mqa_logits_metadata(lens2, BLOCK, num_compute_units(torch.cuda.current_device()))
    outs = [fp8_fp4_paged_mqa_logits((q, None), cache.unsqueeze(-2), w, lens2, bt_buf, sched, max_model_len=MAX_MODEL_LEN,
                                     clean_logits=False, indices=None) for _ in range(2)]
    torch.cuda.synchronize()
    L0, L1 = (DS._u32(x.view(torch.int32)) for x in outs)
    same = all(np.array_equal(L0[b, :n], L1[b, :n]) for b, n in enumerate(want_lens))
    k, ks, first = key_table(o)
    rng = np.random.default_rng(seed)
    req = np.repeat(np.arange(R), NEXT_N)
    full = [b for b in range(rows) if req[b] < FULL_REQUESTS or req[b] in set(rng.choice(R, min(R, 2), replace=False).tolist())]
    live = np.flatnonzero(want_lens > 0)
    rr_rand = rng.choice(live, RANDOM_PAIRS)
    cc_rand = (rng.random(RANDOM_PAIRS) * want_lens[rr_rand]).astype(np.int64)
    rr = np.concatenate([np.repeat(full, want_lens[full]), rr_rand]).astype(np.int64)
    cc = np.concatenate([np.concatenate([np.arange(want_lens[b]) for b in full]) if full else [], cc_rand]).astype(np.int64)
    got = L0[rr, cc]
    d = outdir / "words" / name
    paths = DS._save(d, q=o["q"], w=o["w"], k=k, ks=ks, r=rr, c=first[req[rr]] + cc, key=cc, got=got)
    DS._save(d, cache=o["cache"], bt=o["bt"], L=o["L"], seq_lens=got_lens, bt_flat=got_bt)
    rec = {"case": name, "requests": R, "rows": rows, "lengths": o["L"].tolist(), "max_model_len": MAX_MODEL_LEN,
           "shape": list(outs[0].shape), "full_rows": len(full), "entries": int(rr.size), "checks": {
               "flatten_lens": bool(np.array_equal(got_lens, want_lens)), "flatten_bt": bool(np.array_equal(got_bt, want_bt)),
               "decode_lens_one": bool((dec_lens_buf == 1).all().item()),
               "shape": list(outs[0].shape) == [rows, MAX_MODEL_LEN] and outs[0].dtype == torch.float32, "deterministic": bool(same)}}
    jobs = [{"phase": "logits", "case": name, "kind": "logits", "paths": paths, "keys": ["r", "c", "got"], "range": [a, b]}
            for a, b in DS._chunks(rr.size, DS.JOB_LOGITS)]
    return rec, jobs


def bind_check() -> dict[str, Any]:
    from verity.circuits.defs import bind
    from verity.definitions.registry import dsa

    try:
        fn = bind(dsa.DsaMqaLogitV3, H=H, D=D)
        return {"bindings": {fn.id: bool(fn.signature())}, "ok": bool(fn.signature())}
    except Exception as e:  # noqa: BLE001
        return {"error": DS._error(e), "ok": False}


# ------------------------------------------------------------------------------------------------------------------------ run
def _verdict(doc: dict[str, Any]) -> bool:
    return bool(doc["gate"]["ok"] and doc["served"].get("ok") is True and doc["selection"].get("ok") is True and not doc["errors"]
                and doc["bind"].get("ok") is True and {x["case"] for x in doc["launches"]} == {n for n, _ in CASES}
                and all(all(v is True for v in x["checks"].values()) for x in doc["launches"])
                and len(doc["cases"]) == len(CASES)
                and all(t["words"] > 0 and t["mismatch"] == 0 and t["unresolved"] == 0 and t["reference"] > 0
                        and t["twin_mismatch"] == 0 for t in doc["cases"].values()))


def run(a: argparse.Namespace) -> dict[str, Any]:
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.TemporaryDirectory(dir=a.scratch)
    work = Path(tmp.name)
    env, cleared = DS.environment(work)
    doc: dict[str, Any] = {"schema": SCHEMA, "argv": sys.argv, "env": env, "env_cleared": cleared, "served": DS.served(),
                           "environment": G._environment(), "gate": C.gate(a.min_driver, a.want_cc, a.want_sms), "errors": {},
                           "launches": [], "selection": {}}
    print(f"[glm53-mtp-indexer-logits] gate {json.dumps(doc['gate'])} served {json.dumps(doc['served'])}", flush=True)
    if not doc["gate"]["ok"]:
        (outdir / RECORD).write_text(json.dumps(seal_record(doc, False), indent=1) + "\n")
        print("GLM53-MTP-INDEXER-LOGITS GATE-FAIL", flush=True)
        raise SystemExit(3)
    jobs: list[dict[str, Any]] = []
    try:
        doc["selection"] = selection()
    except Exception as e:  # noqa: BLE001
        doc["errors"]["selection"] = DS._error(e)
    print(f"[glm53-mtp-indexer-logits] selection {json.dumps(doc['selection'])}", flush=True)
    rng = np.random.default_rng(20261112)
    for i, (name, spec) in enumerate(CASES):
        try:
            rec, js = launch(outdir, name, operands(lengths(spec, rng), 9000 + i), 500 + i)
            doc["launches"].append(rec)
            jobs += js
            print(f"[glm53-mtp-indexer-logits] launch {name} {json.dumps(rec)[:1500]}", flush=True)
        except Exception as e:  # noqa: BLE001
            doc["errors"][f"launch/{name}"] = DS._error(e)
            print(f"[glm53-mtp-indexer-logits] launch {name} ERROR {doc['errors'][f'launch/{name}'][:600]}", flush=True)
    doc["bind"] = bind_check()
    print(f"[glm53-mtp-indexer-logits] bind {json.dumps(doc['bind'])}; comparing {len(jobs)} jobs on {a.workers} workers", flush=True)
    doc["cases"] = {}
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        for res in ex.map(DS.compare, jobs):
            if "error" in res:
                doc["errors"][f"compare/{res['case']}"] = res["error"]
                continue
            t = doc["cases"].setdefault(res["case"], {**{k: 0 for k in TOTALS}, "first_mismatch": None})
            for k in TOTALS:
                t[k] += res[k]
            t["first_mismatch"] = t["first_mismatch"] or res["first_mismatch"]
    for name, t in doc["cases"].items():
        print(f"[glm53-mtp-indexer-logits] words {name:6s} words={t['words']} !={t['mismatch']} unresolved={t['unresolved']} "
              f"ref={t['reference']} twin!={t['twin_mismatch']} first={json.dumps(t['first_mismatch'])}", flush=True)
    doc = seal_record(doc, _verdict(doc))
    (outdir / RECORD).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    tmp.cleanup()
    totals = {k: sum(t[k] for t in doc["cases"].values()) for k in TOTALS}
    print(f"GLM53-MTP-INDEXER-LOGITS {'GO' if doc['ok'] else 'NO-GO'} {doc['digest']} words={json.dumps(totals)} "
          f"selection={doc['selection'].get('ok')} errors={sorted(doc['errors'])}", flush=True)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GLM-5.3's indexer logits at a draft step's six rows a request, as vLLM flattens them "
                                             "on this Hopper GPU, against DsaMqaLogit_v3")
    ap.add_argument("--out", required=True, help=f"directory for {RECORD} and words/")
    ap.add_argument("--scratch", default=None, help="parent of the scratch directory (Triton and DeepGEMM caches)")
    ap.add_argument("--min-driver", type=int, default=575)
    ap.add_argument("--want-cc", default="9.0")
    ap.add_argument("--want-sms", type=int, default=132)
    ap.add_argument("--workers", type=int, default=C.cpu_workers())
    a = ap.parse_args(argv)
    return 0 if run(a)["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
