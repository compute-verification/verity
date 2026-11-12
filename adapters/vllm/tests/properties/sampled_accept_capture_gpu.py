"""The sampled accept and its resample on Hopper against `sampled_accept.SampledAccept_v1{K,V,TILE}`: vLLM d9105ea80's
`rejection_sample` (v1/worker/gpu/spec_decode/rejection_sampler_utils.py) with a one-hot draft and the standard method, one
request of 1 + K rows a launch, at GLM-5.3's vocabulary (V = 154,880, tiles of 8192) and K = 5 draft tokens.

One GPU process writes `<out>/sampled_accept_capture.json`, each case's operands and every word the kernels stored for it
under `<out>/cases/`, and the four kernels' PTX and cubins from Triton's cache under `<out>/ptx/`; it prints one line a case
and a last line `SAMPLED-ACCEPT GO` or `NO-GO`.  The cases are `tests.program.sampled_accept_vectors`' hand vectors and ``--random`` of
its `random_rows`.  No model is loaded: the operands are a request's logits after top-p, its draft tokens, seed, positions
and temperature.

  gate       a cc 9.0 card (without one nothing is launched: exit 3); the installed vLLM's version holding the pin, Triton
             3.7.1, and the two installed files the Definition was read from (`SOURCES`, by sha256).  Where only these
             differ the cases are still launched and kept, and the verdict is NO-GO.
  wrapper    `rejection_sample` as the sampler calls it for such a request (`draft_logits` None, `use_fp64` False, no synthetic
             rates, no block verification), the request alone in its launch, twice: ``sampled[:num_sampled]`` and
             ``num_sampled``, the two launches the same words.
  kernels    the four kernels launched with the wrapper's arguments into buffers of 0xA5 bytes: each draft row's tile maxima
             and sums (`_compute_local_logits_stats_kernel`), the accepted count and the log-sum-exp of the last position
             tested (`_rejection_kernel`), the resample's tiles (`_resample_kernel`), the tokens and the count
             (`_insert_resampled_kernel`), which equal the wrapper's; and which words no kernel stored to (the bonus row's
             statistics, the draft's, the greedy argmax, ``sampled`` past the count).
  batch      every case as one request of a single launch, the requests' state rows shuffled (`idx_mapping`): each request's
             tokens and count are its own launch's.
  reference  `sampled_accept.sampled_accept` on the same operands, word for word: the tile maxima and sums of the K draft
             rows, the accepted count, the appended tokens, lane 0's log-sum-exp of the last position tested, and 0.0 for the
             draft's.
  ptx        Triton's cache holds each kernel's PTX and cubin for these launches: the PTX's instructions (`ptx_body`: the
             text to the entry's closing brace without its `.loc` lines) are the ones the Definition was read from
             (`PTX_BODY`), and the cubin's machine code (`cubin_text`) is the SASS it was read from (`CUBIN_TEXT`).

A case whose reference reports ``halves_split`` is outside the Definition's domain (lane 0 of `_rejection_kernel` accepts a
draft token that is not the last while lanes 16-31 refuse it, and the next position's shuffles read lanes that stopped).  It is
launched and kept like the others and reported apart: its words decide nothing.

GO needs the gate, every case inside the domain compared with 0 differing words, the two wrapper launches, the kernels'
launch and the batch agreeing, and the PTX and the machine code.  The exit status is 0 on GO, 1 on NO-GO, 3 when there is
no cc 9.0 card.

``--replay DIR`` needs no card: it reads a kept `<out>` and compares the stored words with the reference again, and with
``--definition N`` evaluates the registered `SampledAccept_v1{K,V,TILE=8192}` on the first N cases against the stored tokens
and count (about a minute a case).

    PY=/workspace/venv312/bin/python     # from the tree's root, on one idle card
    $PY adapters/vllm/tests/properties/sampled_accept_capture_gpu.py --out DIR [--random N] [--scratch DIR]
    python adapters/vllm/tests/properties/sampled_accept_capture_gpu.py --replay DIR [--definition N]
"""
from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
for _p in (ROOT / "adapters" / "vllm", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tests.program import sampled_accept_vectors as SV  # noqa: E402
from verity.definitions.registry import sampled_accept as A  # noqa: E402

SCHEMA = "verity-vllm/sampled-accept-capture/v1"
RECORD = "sampled_accept_capture.json"
#: GLM-5.3's vocabulary and the draft tokens a step verifies under its served recipe
V, K = 154880, 5
#: the rows of the sampler's per-request state (temperature, seed): `max_num_reqs`
MAX_REQS = 1024
RESAMPLE_TILE = 1024
PIN, TRITON = "d9105ea8", "3.7.1"
F = np.float32
POISON = 0xA5A5A5A5
KERNELS = ("_compute_local_logits_stats_kernel", "_rejection_kernel", "_resample_kernel", "_insert_resampled_kernel")
#: the installed vLLM's files the Definition was read from, by sha256 at d9105ea80
SOURCES = {"v1/worker/gpu/spec_decode/rejection_sampler_utils.py": "20ca2e5ac34e9ef93dca388bed72a00d4ff67d569ec6ec2393b21a92d74a1fae",
           "v1/worker/gpu/sample/gumbel.py": "3ec1df510bdad13e8b0a457b5b67c98affdf6e57369e56cb471246cc8f40dd34"}
#: sha256 of `ptx_body` of each kernel's PTX as Triton 3.7.1 compiles it for sm_90a at these launches' specialization (one
#: request of K = 5 over V = 154,880, a one-hot draft, fp32 noise)
PTX_BODY = {"_compute_local_logits_stats_kernel": "137a82dafc43a5e04d2c64405010033c42499f8d874566db2eae40cc40a4b2f7",
            "_rejection_kernel": "efb21b0aca5e403897a9614bab0bebeee659af248fe323e2153d99414e3bf5ce",
            "_resample_kernel": "0d096db8b8d6c0f0055bdf5ebe735761c635f21edea877ab92ec9b919e55137e",
            "_insert_resampled_kernel": "dbab4c6dc030a2f5b7ee2ff08496d4aeb932eb92a95610f4bd1dbc68d2de89af"}
#: sha256 of each kernel's machine code, the `.text.<kernel>` section of the cubin Triton 3.7.1's ptxas (12.8.93) assembles
#: from that PTX: the SASS the Definition was read from
CUBIN_TEXT = {"_compute_local_logits_stats_kernel": "98c93ee47961e762dcdc20e9a4a901b1bdc7b9f26ac5882deaefa4847bc6d652",
              "_rejection_kernel": "dd215db9893566b7ed02ba53343ec9f2d26542789a090d77774bcf52ba8f00f5",
              "_resample_kernel": "77905dc429852e3d556a02e835f4fb541e9db7fd23210c1d02a831734121c11b",
              "_insert_resampled_kernel": "0dc0a4d86ae47391499eb2c61ca5ee6b6f90a7628cf98c04701145ea1194d572"}
#: what a case keeps of the kernels' launch, and of those what the reference states
STORED = ("target_local_max", "target_local_sumexp", "target_local_argmax", "draft_local_max", "draft_local_sumexp", "sampled",
          "num_sampled", "num_sampled_after_rejection", "target_rejected_logsumexp", "draft_rejected_logsumexp",
          "resampled_local_argmax", "resampled_local_max")


def ptx_body(text: str) -> str:
    """A PTX file's instructions: its lines to the entry's closing brace, without `.loc` and `.file` lines, comments and
    blank lines, so two compiles of one kernel from sources at different paths give one text."""
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith((".loc", ".file", "//")):
            continue
        out.append(re.sub(r"\s*//.*$", "", s))
        if line.startswith("}"):
            break
    return "\n".join(out)


def cubin_text(blob: bytes, name: str) -> bytes | None:
    """A kernel's machine code: the `.text.<name>` section of its cubin (a little-endian ELF64), or None without one."""
    if blob[:6] != b"\x7fELF\x02\x01":
        return None
    (shoff,), (size, count, strings) = struct.unpack_from("<Q", blob, 0x28), struct.unpack_from("<HHH", blob, 0x3A)
    heads = [struct.unpack_from("<IIQQQQ", blob, shoff + i * size) for i in range(count)]
    if strings >= count:
        return None
    table = blob[heads[strings][4]:heads[strings][4] + heads[strings][5]]
    for h in heads:
        if table[h[0]:table.find(b"\0", h[0])] == f".text.{name}".encode():
            return blob[h[4]:h[4] + h[5]]
    return None


def _w32(a) -> np.ndarray:
    return np.ascontiguousarray(a).view(np.uint32)


# ---------------------------------------------------------------------------------------------------------------- the reference
def compare(got: dict[str, np.ndarray], ref: dict, draft: list[int]) -> tuple[int, int, list[str]]:
    """(words compared, words that differ, what differed) of one case's stored words against the reference."""
    k = len(draft)
    n = bad = 0
    notes: list[str] = []

    def eq(what: str, a, b) -> None:
        nonlocal n, bad
        a, b = np.asarray(a).reshape(-1), np.asarray(b).reshape(-1)
        if a.shape != b.shape:
            n, bad = n + max(a.size, b.size), bad + max(a.size, b.size)
            notes.append(f"{what}: {a.size} words against {b.size}")
            return
        d = int((a != b).sum())
        n, bad = n + a.size, bad + d
        if d:
            notes.append(f"{what}: {d} of {a.size} differ")

    r = ref["accepted"]
    eq("tile maxima, draft rows", _w32(got["target_local_max"][:k]), _w32(np.array(ref["max"], F)))
    eq("tile sums, draft rows", _w32(got["target_local_sumexp"][:k]), _w32(np.array(ref["sumexp"], F)))
    eq("accepted count after the rejection kernel", got["num_sampled_after_rejection"], [r])
    eq("accepted count + 1 after the insert", got["num_sampled"], [r + 1])
    eq("appended tokens", got["sampled"].reshape(-1)[:r + 1], np.array(ref["appended"][:r + 1], np.int64))
    tested = [i for i in range(k) if i <= r and all(draft[j] >= 0 for j in range(i + 1))]
    eq("lane 0's log-sum-exp of the last position tested", _w32(got["target_rejected_logsumexp"]),
       [ref["lse"][tested[-1]] if tested else 0])
    eq("the draft's log-sum-exp (a one-hot draft: 0.0)", _w32(got["draft_rejected_logsumexp"]), [0])
    return n, bad, notes


def unwritten(got: dict[str, np.ndarray], k: int) -> dict[str, Any]:
    """Which of the launch's 0xA5 buffers no kernel stored to."""
    count = int(np.asarray(got["num_sampled"]).reshape(-1)[0])
    return {"bonus_row_stats": bool((_w32(got["target_local_max"][k]) == POISON).all()
                                    and (_w32(got["target_local_sumexp"][k]) == POISON).all()),
            "draft_stats": bool((_w32(got["draft_local_max"]) == POISON).all() and (_w32(got["draft_local_sumexp"]) == POISON).all()),
            "target_argmax": bool((_w32(got["target_local_argmax"]) == POISON).all()),
            "sampled_past_count": bool((_w32(got["sampled"].reshape(-1)[max(count, 0):]) == POISON).all()),
            "resample_tiles_written": int((_w32(got["resampled_local_max"]) != POISON).sum())}


def judge(name: str, case: dict, got: dict[str, np.ndarray]) -> dict[str, Any]:
    """One case's entry: the reference's outputs, the kernels', and the comparison."""
    ref, draft = case["ref"], case["draft"]
    n, bad, notes = compare(got, ref, draft)
    count = int(np.asarray(got["num_sampled"]).reshape(-1)[0])
    return {"case": name, "draft": draft, "seed": case["seed"], "pos": case["pos"], "in_domain": ref["halves_split"] is None,
            "halves_split": ref["halves_split"], "verdict": ref["verdict"], "accepted": ref["accepted"],
            "appended": ref["appended"], "kernel_num_sampled": count,
            "kernel_sampled": [int(x) for x in got["sampled"].reshape(-1)[:max(0, min(count, len(draft) + 1))]],
            "lse": [hex(x) for x in ref["lse"]], "lse_upper_half": [hex(x) for x in ref["lse_hi"]],
            "u": [hex(x) for x in ref["u"]], "logprob": [hex(x) for x in ref["logprob"]],
            "words_compared": n, "words_differ": bad, "notes": notes, "unwritten": unwritten(got, len(draft))}


def definition_words(case: dict) -> tuple[list[int], list[int]]:
    """`SampledAccept_v1{K,V,TILE=8192}` evaluated on a case, and the same words of the reference."""
    from verity.circuits.defs import bind
    from verity.circuits.evaluation.reference import evaluate

    logits, draft, ref = case["logits"], case["draft"], case["ref"]
    got = evaluate(bind(A.SampledAccept, K=len(draft), V=logits.shape[1], TILE=A.STATS_TILE),
                   [int(w) for w in _w32(np.ascontiguousarray(logits, F)).reshape(-1)], [d & 0xFFFFFFFF for d in draft],
                   case["seed"], case["pos"], case["temp"])
    return list(got), list(ref["verdict"]) + [ref["accepted"]] + [x & 0xFFFFFFFF for x in ref["appended"]]


def summarize(entries: list[dict]) -> dict[str, Any]:
    inside = [e for e in entries if e["in_domain"]]
    return {"cases": len(entries), "rows": sum(len(e["draft"]) + 1 for e in entries), "cases_in_domain": len(inside),
            "cases_in_domain_that_differ": sum(bool(e["words_differ"]) for e in inside),
            "words_compared": sum(e["words_compared"] for e in inside), "words_differ": sum(e["words_differ"] for e in inside),
            "cases_outside_domain": [{k: e[k] for k in ("case", "halves_split", "verdict", "accepted", "kernel_num_sampled",
                                                         "kernel_sampled", "words_compared", "words_differ", "notes")}
                                     for e in entries if not e["in_domain"]],
            "accepted_counts": {str(c): sum(e["accepted"] == c for e in entries) for c in range(K + 1)},
            "unwritten": {k: sum(bool(e["unwritten"][k]) for e in entries)
                          for k in ("bonus_row_stats", "draft_stats", "target_argmax", "sampled_past_count")}}


def cases(random: int, seed: int) -> list[dict]:
    return SV.vectors(K, V) + SV.random_rows(K, V, random, seed)


# ------------------------------------------------------------------------------------------------------------------- the card
def gate(torch) -> dict[str, Any]:
    import triton
    import vllm

    out: dict[str, Any] = {"cuda": bool(torch.cuda.is_available())}
    if out["cuda"]:
        out["device"] = torch.cuda.get_device_name(0)
        out["capability"] = list(torch.cuda.get_device_capability(0))
    out.update(torch=torch.__version__, triton=triton.__version__, vllm=vllm.__version__, numpy=np.__version__)
    base = Path(vllm.__file__).resolve().parent
    out["sources"] = {f: hashlib.sha256((base / f).read_bytes()).hexdigest() if (base / f).is_file() else None for f in SOURCES}
    out["card"] = bool(out["cuda"] and out.get("capability") == [9, 0])
    out["ok"] = bool(out["card"] and PIN in out["vllm"] and out["triton"] == TRITON and out["sources"] == SOURCES)
    return out


def _poison(torch, shape, dtype, device):
    """A tensor of ``shape`` whose every byte is 0xA5."""
    t = torch.empty(shape, dtype=dtype, device=device)
    t.view(torch.uint8).fill_(0xA5)
    return t


def operands(torch, batch: list[dict], device, order: list[int] | None = None) -> dict[str, Any]:
    """`rejection_sample`'s operands for the requests of ``batch``, each of K + 1 rows: request r's temperature and seed in
    state row ``order[r]`` (its own number when None)."""
    n, rows = len(batch), K + 1
    order = list(range(n)) if order is None else order
    temperature = torch.zeros(MAX_REQS, dtype=torch.float32)
    seed = torch.zeros(MAX_REQS, dtype=torch.int64)
    for r, c in enumerate(batch):
        temperature[order[r]] = float(np.uint32(c["temp"]).view(F))
        seed[order[r]] = c["seed"]
    idx = torch.tensor(order, dtype=torch.int64)
    return {"target_logits": torch.from_numpy(np.concatenate([np.ascontiguousarray(c["logits"], F) for c in batch])).to(device),
            "draft_sampled": torch.tensor([t for c in batch for t in [0] + c["draft"]], dtype=torch.int32, device=device),
            "cu_num_logits": torch.arange(0, rows * n + 1, rows, dtype=torch.int32, device=device),
            "pos": torch.tensor([p for c in batch for p in c["pos"]], dtype=torch.int64, device=device),
            "idx_mapping": idx.to(device), "expanded_idx_mapping": idx.repeat_interleave(rows).to(device),
            "expanded_local_pos": torch.arange(rows, dtype=torch.int32).repeat(n).to(device),
            "temperature": temperature.to(device), "seed": seed.to(device)}


def wrapper(torch, RU, op: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """`rejection_sample` as the sampler calls it for a one-hot draft: (sampled, num_sampled)."""
    sampled, count = RU.rejection_sample(op["target_logits"], None, op["draft_sampled"], op["cu_num_logits"], op["pos"],
                                         op["idx_mapping"], op["expanded_idx_mapping"], op["expanded_local_pos"],
                                         op["temperature"], op["seed"], K, None, use_fp64=False, use_block_verification=False)
    torch.cuda.synchronize()
    return sampled.cpu().numpy(), count.cpu().numpy()


def kernels(torch, triton, RU, op: dict[str, Any]) -> dict[str, np.ndarray]:
    """The four kernels as `rejection_sample` launches them for a one-hot draft under the standard method, its `new_empty`
    buffers filled with 0xA5 bytes instead; every buffer after the last launch, and the count after `_rejection_kernel`."""
    logits, dev = op["target_logits"], op["target_logits"].device
    reqs, (rows, vocab) = op["cu_num_logits"].shape[0] - 1, logits.shape
    tiles = triton.cdiv(vocab, A.STATS_TILE)
    rtiles = triton.cdiv(vocab, RESAMPLE_TILE)
    f32, i64 = torch.float32, torch.int64
    b = {"target_local_argmax": _poison(torch, (rows, tiles), i64, dev), "target_local_max": _poison(torch, (rows, tiles), f32, dev),
         "target_local_sumexp": _poison(torch, (rows, tiles), f32, dev), "draft_local_max": _poison(torch, (rows, tiles), f32, dev),
         "draft_local_sumexp": _poison(torch, (rows, tiles), f32, dev), "sampled": _poison(torch, (reqs, K + 1), i64, dev),
         "num_sampled": _poison(torch, (reqs,), torch.int32, dev), "target_rejected_logsumexp": _poison(torch, (reqs,), f32, dev),
         "draft_rejected_logsumexp": _poison(torch, (reqs,), f32, dev),
         "resampled_local_argmax": _poison(torch, (reqs, rtiles), i64, dev),
         "resampled_local_max": _poison(torch, (reqs, rtiles), f32, dev)}
    RU._compute_local_logits_stats_kernel[(rows, tiles)](
        b["target_local_argmax"], b["target_local_argmax"].stride(0), b["target_local_max"], b["target_local_max"].stride(0),
        b["target_local_sumexp"], b["target_local_sumexp"].stride(0), b["draft_local_max"], b["draft_local_max"].stride(0),
        b["draft_local_sumexp"], b["draft_local_sumexp"].stride(0), logits, logits.stride(0), None, 0, 0,
        op["expanded_idx_mapping"], op["expanded_local_pos"], op["temperature"], vocab, K,
        BLOCK_SIZE=A.STATS_TILE, HAS_DRAFT_LOGITS=False)
    RU._rejection_kernel[(reqs,)](
        b["sampled"], b["sampled"].stride(0), b["num_sampled"], b["target_rejected_logsumexp"], b["draft_rejected_logsumexp"],
        logits, logits.stride(0), b["target_local_argmax"], b["target_local_argmax"].stride(0), b["target_local_max"],
        b["target_local_max"].stride(0), b["target_local_sumexp"], b["target_local_sumexp"].stride(0), op["draft_sampled"],
        None, 0, 0, b["draft_local_max"], b["draft_local_max"].stride(0), b["draft_local_sumexp"],
        b["draft_local_sumexp"].stride(0), op["cu_num_logits"], op["idx_mapping"], op["temperature"], op["seed"], op["pos"],
        None, None, None, 0, tiles,
        PADDED_VOCAB_NUM_BLOCKS=triton.next_power_of_2(tiles), HAS_DRAFT_LOGITS=False, SYNTHETIC_MODE=False,
        USE_BLOCK_VERIFICATION=False, num_warps=1)
    torch.cuda.synchronize()
    after = b["num_sampled"].cpu().numpy().copy()
    RU._resample_kernel[(reqs, rtiles)](
        b["resampled_local_argmax"], b["resampled_local_argmax"].stride(0), b["resampled_local_max"],
        b["resampled_local_max"].stride(0), logits, logits.stride(0), b["target_rejected_logsumexp"], None, 0, 0,
        b["draft_rejected_logsumexp"], b["num_sampled"], op["cu_num_logits"], op["expanded_idx_mapping"], op["draft_sampled"],
        op["temperature"], op["seed"], op["pos"], None, vocab,
        BLOCK_SIZE=RESAMPLE_TILE, HAS_DRAFT_LOGITS=False, USE_FP64=False, USE_BLOCK_VERIFICATION=False)
    RU._insert_resampled_kernel[(reqs,)](
        b["sampled"], b["sampled"].stride(0), b["num_sampled"], b["resampled_local_argmax"],
        b["resampled_local_argmax"].stride(0), b["resampled_local_max"], b["resampled_local_max"].stride(0), rtiles,
        op["cu_num_logits"], op["expanded_idx_mapping"], op["temperature"],
        PADDED_RESAMPLE_NUM_BLOCKS=triton.next_power_of_2(rtiles))
    torch.cuda.synchronize()
    return {**{name: t.cpu().numpy() for name, t in b.items()}, "num_sampled_after_rejection": after}


def keep_ptx(cache: Path, dest: Path) -> dict[str, Any]:
    """Each kernel's PTX and cubin in Triton's cache directory, copied under ``dest``, with whether there is one and every
    one's instructions and machine code are the read ones (`PTX_BODY`, `CUBIN_TEXT`)."""
    out: dict[str, Any] = {}
    for name in KERNELS:
        found = sorted(cache.rglob(f"{name}.ptx"))
        files = []
        for f in found:
            (dest / name).mkdir(parents=True, exist_ok=True)
            kept = dest / name / f"{f.parent.name[:12]}-{f.name}"
            shutil.copy(f, kept)
            text = f.read_text(errors="replace")
            entry = {"file": str(kept.relative_to(dest)), "sha256": hashlib.sha256(text.encode()).hexdigest(),
                     "body_sha256": hashlib.sha256(ptx_body(text).encode()).hexdigest(), "cubin_text_sha256": None}
            if (cubin := f.with_suffix(".cubin")).is_file():
                shutil.copy(cubin, kept.with_suffix(".cubin"))
                code = cubin_text(cubin.read_bytes(), name)
                entry["cubin_text_sha256"] = hashlib.sha256(code).hexdigest() if code is not None else None
            files.append(entry)
        out[name] = {"files": files, "ok": bool(files) and all(f["body_sha256"] == PTX_BODY[name] for f in files),
                     "code_ok": bool(files) and all(f["cubin_text_sha256"] == CUBIN_TEXT[name] for f in files)}
    return out


def save(dest: Path, case: dict, got: dict[str, np.ndarray], first: tuple, second: tuple) -> None:
    np.savez_compressed(dest / f"{case['name']}.npz", logits=np.ascontiguousarray(case["logits"], F),
                        draft=np.array(case["draft"], np.int64), seed=np.int64(case["seed"]), pos=np.array(case["pos"], np.int64),
                        temp=np.uint32(case["temp"]), wrapper_sampled=np.stack([first[0], second[0]]),
                        wrapper_num_sampled=np.stack([first[1], second[1]]), **{k: got[k] for k in STORED})


def capture(a: argparse.Namespace) -> int:
    out = Path(a.out)
    (out / "cases").mkdir(parents=True, exist_ok=True)
    scratch = Path(a.scratch or tempfile.mkdtemp(prefix="sampled-accept-"))
    os.environ["TRITON_CACHE_DIR"] = str(scratch / "triton")
    import torch
    import triton

    record: dict[str, Any] = {"schema": SCHEMA, "statics": {"K": K, "V": V, "TILE": A.STATS_TILE}, "seed": a.seed,
                              "random": a.random, "gate": gate(torch)}
    if not record["gate"]["card"]:
        (out / RECORD).write_text(json.dumps({**record, "ok": False}, indent=1) + "\n")
        print(f"SAMPLED-ACCEPT NO-GO: no cc 9.0 card: {json.dumps(record['gate'])}")
        return 3
    if not record["gate"]["ok"]:
        print(f"the gate fails, so the verdict is NO-GO; launching and keeping the cases: {json.dumps(record['gate'])}", flush=True)
    from vllm.v1.worker.gpu.spec_decode import rejection_sampler_utils as RU

    device = torch.device("cuda")
    todo = cases(a.random, a.seed)
    entries, alone = [], []
    for c in todo:
        op = operands(torch, [c], device)
        first, second = wrapper(torch, RU, op), wrapper(torch, RU, op)
        got = kernels(torch, triton, RU, op)
        save(out / "cases", c, got, first, second)
        e = judge(c["name"], c, got)
        count = int(first[1][0])
        e["launches_agree"] = bool(np.array_equal(first[1], second[1]) and np.array_equal(first[1], got["num_sampled"])
                                   and 1 <= count <= K + 1 and np.array_equal(first[0][0, :count], second[0][0, :count])
                                   and np.array_equal(first[0][0, :count], got["sampled"][0, :count]))
        entries.append(e)
        alone.append((count, first[0][0, :max(count, 0)].tolist()))
        print(f"{'ok  ' if not e['words_differ'] and e['launches_agree'] else 'DIFF'} {c['name']}: {e['words_compared']} words, "
              f"{e['words_differ']} differ; accepted {e['accepted']}, kernel count {e['kernel_num_sampled']}"
              f"{'' if e['in_domain'] else '; outside the domain (halves split)'}{'; ' + '; '.join(e['notes']) if e['notes'] else ''}",
              flush=True)
    order = [int(x) for x in np.random.default_rng(a.seed).permutation(MAX_REQS)[:len(todo)]]
    sampled, counts = wrapper(torch, RU, operands(torch, todo, device, order))
    batch_differ = [c["name"] for r, c in enumerate(todo)
                    if (int(counts[r]), sampled[r, :max(int(counts[r]), 0)].tolist()) != alone[r]]
    np.savez_compressed(out / "batch.npz", order=np.array(order), sampled=sampled, num_sampled=counts)
    record["batch"] = {"requests": len(todo), "differ_from_own_launch": batch_differ}
    record["ptx"] = keep_ptx(scratch / "triton", out / "ptx")
    record["summary"] = s = summarize(entries)
    record["launches_disagree"] = [e["case"] for e in entries if not e["launches_agree"]]
    record["cases"] = entries
    inside_batch = [n for n in batch_differ if n in {e["case"] for e in entries if e["in_domain"]}]
    record["ok"] = bool(record["gate"]["ok"] and not s["words_differ"] and s["cases_in_domain"] and not inside_batch
                        and not [e for e in entries if e["in_domain"] and not e["launches_agree"]]
                        and all(p["ok"] and p["code_ok"] for p in record["ptx"].values()))
    (out / RECORD).write_text(json.dumps(record, indent=1) + "\n")
    if not a.scratch:
        shutil.rmtree(scratch, ignore_errors=True)
    print(report(record))
    return 0 if record["ok"] else 1


def report(record: dict[str, Any]) -> str:
    s = record["summary"]
    lines = [f"cases {s['cases']} (requests of 1 + {K} rows; {s['rows']} rows), inside the domain {s['cases_in_domain']}, of them "
             f"{s['cases_in_domain_that_differ']} differ; words compared {s['words_compared']}, differ {s['words_differ']}",
             f"accepted counts {s['accepted_counts']}"]
    for o in s["cases_outside_domain"]:
        lines.append(f"outside the domain: {o['case']}: lane 0 accepts {o['accepted']}, the kernel's count {o['kernel_num_sampled']} "
                     f"({o['words_differ']} of {o['words_compared']} words differ)")
    if "batch" in record:
        lines.append(f"batch of {record['batch']['requests']}: {len(record['batch']['differ_from_own_launch'])} requests differ from "
                     f"their own launch; launches that disagree: {len(record['launches_disagree'])}")
        lines.append("ptx: " + ", ".join(f"{k} {'the read instructions' if p['ok'] else 'DIFFERS (' + str(len(p['files'])) + ' files)'}"
                                         for k, p in record["ptx"].items()))
        lines.append("machine code: " + ", ".join(f"{k} {'the read SASS' if p.get('code_ok') else 'DIFFERS'}"
                                                  for k, p in record["ptx"].items()))
    if "gate" in record and not record["gate"].get("ok", True):
        lines.append(f"the gate fails: {json.dumps(record['gate'])}")
    if "definition" in record:
        d = record["definition"]
        lines.append(f"definition: {d['cases']} cases, {d['words_compared']} words compared, {d['words_differ']} differ")
    lines.append(f"SAMPLED-ACCEPT {'GO' if record['ok'] else 'NO-GO'}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------------------------ off the card
def load(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """A kept case: its operands with the reference on them, and what the kernels stored."""
    z = np.load(path)
    logits, draft = z["logits"], [int(d) for d in z["draft"]]
    seed, pos, temp = int(z["seed"]), [int(p) for p in z["pos"]], int(z["temp"])
    case = {"name": path.stem, "logits": logits, "draft": draft, "seed": seed, "pos": pos, "temp": temp,
            "ref": A.sampled_accept(logits, draft, seed, pos, temp)}
    return case, {k: z[k] for k in STORED}


def replay(a: argparse.Namespace) -> int:
    src = Path(a.replay)
    kept = json.loads((src / RECORD).read_text())
    names = [e["case"] for e in kept["cases"]]
    entries, dfn = [], {"cases": 0, "words_compared": 0, "words_differ": 0, "differ": []}
    for i, name in enumerate(names):
        case, got = load(src / "cases" / f"{name}.npz")
        entries.append(e := judge(name, case, got))
        if i < a.definition and e["in_domain"]:
            words, want = definition_words(case)
            count = int(got["num_sampled"].reshape(-1)[0])
            card = [int(p < count - 1) for p in range(K)] + [count - 1] + [
                int(x) & 0xFFFFFFFF if j < count else 0xFFFFFFFF for j, x in enumerate(got["sampled"].reshape(-1))]
            bad = sum(int(x != y) for x, y in zip(words, card)) + sum(int(x != y) for x, y in zip(words, want))
            dfn = {"cases": dfn["cases"] + 1, "words_compared": dfn["words_compared"] + len(card) + len(want),
                   "words_differ": dfn["words_differ"] + bad, "differ": dfn["differ"] + ([name] if bad else [])}
        print(f"{'ok  ' if not e['words_differ'] else 'DIFF'} {name}: {e['words_compared']} words, {e['words_differ']} differ",
              flush=True)
    s = summarize(entries)
    record = {"schema": SCHEMA, "replay_of": kept.get("gate", {}), "summary": s, "definition": dfn, "cases": entries,
              "ok": bool(not s["words_differ"] and s["cases_in_domain"] and not dfn["words_differ"])}
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "sampled_accept_replay.json").write_text(json.dumps(record, indent=1) + "\n")
    print(report(record))
    return 0 if record["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", help="the directory the record, the cases and the PTX go to")
    ap.add_argument("--random", type=int, default=200, help="random requests beside the hand vectors (`random_rows`)")
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--scratch", help="Triton's cache goes under it and stays; a temporary directory otherwise")
    ap.add_argument("--replay", metavar="DIR", help="compare a kept capture with the reference again, off the card")
    ap.add_argument("--definition", type=int, default=0, metavar="N", help="with --replay: evaluate the Definition on the first N cases")
    a = ap.parse_args(argv)
    if a.replay:
        return replay(a)
    if not a.out:
        ap.error("--out DIR, or --replay DIR")
    return capture(a)


if __name__ == "__main__":
    raise SystemExit(main())
