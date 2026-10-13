"""sweep/workload.py — the ONE deterministic heterogeneous workload generator of the R12 sweep (BRIEF §12).

    verity-vllm workload --row-id <row-id> --hf-repo <repo> --revision <rev> --batch B --max-input I --max-output O \\
        [--sampling greedy|stoch-t0.8-p0.95|stoch-t0.8-p1] [--arrivals none|staggered] [--tokenizer <snapshot dir or tokenizer.json>] [--case <ROLE>] \\
        [--lengths-only] [--check] --out workloads/<row-id>.json

Output = the existing request-list workload format (`verity-vllm/workload/v1`, see `workloads/workload_cov_*.json`) that
`verity_vllm.engine.vllm_adapter.load_workload` / `bench/commit_delta.py` / `zk/bench/e2e_commit_run.py` / the eager capture consume unchanged,
plus one extra top-level key `sweep` (row id, seed, the length/cap/seed/arrival vectors, the plan) that existing loaders ignore.

Determinism: `seed = SHA256(row_id)`; every draw comes from `numpy.random.default_rng` seeded from that digest and a purpose tag.

Shape plan for batch B over [lo, I] (lo = 8, or 1 when I < 32):
  anchors (in this priority order, truncated to B):  I exactly · lo + {0..3} (near the minimum) · one awkward (odd, non-power-of-two)
  length in each linear quartile of [lo, I] (Q2, Q3, Q1, Q4) — then the remaining B - 6 lengths are log-uniform over [lo, I]
  (stratified: one draw per equal log-width stratum), nudged off powers of two.  Output caps over [2, O]: O exactly · 2 (very short) ·
  one odd cap per quartile · log-uniform remainder; caps are paired with the lengths by a seeded permutation.  For B >= 8 this gives
  >= 6 distinct lengths and caps, one near the min, one at the max, one per quartile, several non-power-of-two values (`--check`).
Prompts: distinct real text from `verity_vllm/engine/profiles/corpus_coverage.json` (three public-domain Gutenberg books), each request
starting at a different seeded paragraph, tokenized with the served model's tokenizer.json (`tokenizers`, add_special_tokens=False like
every earlier workload) and truncated to the exact planned length.  Sampling `stoch-t0.8-p0.95`: temperature 0.8 / top_p 0.95 and an
independent stored seed per request (`requests[i].seed`, read by `load_workload`).  `stoch-t0.8-p1`: the same without the top-p mask (temperature and
Gumbel-max only).  Arrivals `staggered`: request 0 at 0 s, the
others at seeded offsets over the first ~60 % of a full-length run; stored as `arrival_offsets_s` and as the step-indexed
`arrive_step` the loaders consume (`arrive_step = round(offset_s / nominal_step_s)`, nominal_step_s = 0.010, recorded).
`--lengths-only` writes the plan without text/token ids (no tokenizer needed) so lanes can see the shape plan on a laptop.
"""
from __future__ import annotations

import hashlib
import json
import re
import os
import sys
import uuid
from importlib import resources
from pathlib import Path

import numpy as np
from dataclasses import dataclass

from verity_vllm.config import CHECKPOINTS, WORKLOADS
from verity_vllm.config import option
from verity_vllm.engine.profiles.transport import row_custom_all_reduce_off, row_p2p_env

CORPUS = resources.files("verity_vllm") / "engine" / "profiles" / "corpus_coverage.json"
SCHEMA = "verity-vllm/workload/v1"
SWEEP_SCHEMA = "verity-sweep/workload/v1"
SAMPLING = {
    "greedy": {"temperature": 0.0, "top_p": 1.0, "top_k": 0},
    "stoch-t0.8-p0.95": {"temperature": 0.8, "top_p": 0.95, "top_k": 0},
    "stoch-t0.8-p1": {"temperature": 0.8, "top_p": 1.0, "top_k": 0},
}
NOMINAL_STEP_S = 0.010          # a decode step of the 1–2B models on L40S (R11: 7.5–10.9 ms); converts arrival seconds -> arrive_step
MIN_CAP = 2                     # the "very short" output cap (prefill + one decode step)


# ---------------------------------------------------------------------------------------------------------------------------------
# seeding
# ---------------------------------------------------------------------------------------------------------------------------------
def row_seed(row_id: str) -> str:
    """SHA256 of the row id (hex) — the workload seed stored in the manifest."""
    return hashlib.sha256(row_id.encode()).hexdigest()


def rng_for(seed_hex: str, purpose: str) -> np.random.Generator:
    h = hashlib.sha256(bytes.fromhex(seed_hex) + b"/" + purpose.encode()).digest()
    return np.random.default_rng([int.from_bytes(h[i:i + 4], "little") for i in range(0, 32, 4)])


def sampling_seed(seed_hex: str, index: int) -> int:
    h = hashlib.sha256(bytes.fromhex(seed_hex) + b"/sampling/" + str(index).encode()).digest()
    return int.from_bytes(h[:4], "little") & 0x7FFF_FFFF


# ---------------------------------------------------------------------------------------------------------------------------------
# the shape plan
# ---------------------------------------------------------------------------------------------------------------------------------
def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _awkward(n: int, lo: int, hi: int, rng: np.random.Generator) -> int:
    """Nudge `n` into [lo, hi] and off powers of two / multiples of 64 (odd where the range allows)."""
    n = int(min(max(n, lo), hi))
    if hi - lo < 4:
        return n
    for _ in range(8):
        if not _is_pow2(n) and n % 64 != 0 and (n % 2 == 1 or hi - lo < 8):
            return n
        n = int(min(max(n + int(rng.integers(-3, 4)) * 2 + 1, lo), hi))
    return n


def _quartile_bounds(lo: int, hi: int) -> list[tuple[int, int]]:
    span = hi - lo
    qs = [lo + span * k // 4 for k in range(5)]
    return [(qs[k], max(qs[k], qs[k + 1] - (1 if k < 3 else 0))) for k in range(4)]


def _quartile_of(n: int, lo: int, hi: int) -> int:
    if hi <= lo:
        return 0
    return min(3, int((n - lo) * 4 // (hi - lo)))


def plan_values(n: int, lo: int, hi: int, rng: np.random.Generator, *, anchor_near_lo: bool) -> list[int]:
    """`n` values in [lo, hi]: anchors (hi exactly, lo + {0..3}, one awkward value per quartile in order Q2, Q3, Q1, Q4), then a
    stratified log-uniform remainder nudged off powers of two.  Duplicates are avoided while the range allows."""
    if hi <= lo:
        return [hi] * n
    out: list[int] = []
    q = _quartile_bounds(lo, hi)
    near_lo = lo if not anchor_near_lo else int(min(hi, lo + rng.integers(0, 4)))
    anchors = [hi, near_lo]
    for k in (1, 2, 0, 3):
        a, b = q[k]
        anchors.append(_awkward(int(rng.integers(a, b + 1)), a, b, rng))
    for v in anchors:
        if len(out) >= n:
            break
        out.append(int(v))
    rest = n - len(out)
    if rest > 0:
        # stratified log-uniform: one draw per equal-width stratum of log[lo, hi]
        la, lb = np.log(max(lo, 1)), np.log(hi)
        edges = np.linspace(la, lb, rest + 1)
        for k in range(rest):
            v = int(round(float(np.exp(rng.uniform(edges[k], edges[k + 1])))))
            v = _awkward(v, lo, hi, rng)
            tries = 0
            while v in out and tries < 12 and hi - lo + 1 > len(out):
                v = _awkward(int(min(max(v + int(rng.integers(-5, 6)), lo), hi)), lo, hi, rng)
                tries += 1
            out.append(v)
    return out


def plan(row_id: str, batch: int, max_input: int, max_output: int, *, sampling: str, arrivals: str) -> dict:
    """The model-independent shape plan of one row: lengths, caps, sampling seeds, arrival offsets (all deterministic in the row id)."""
    seed = row_seed(row_id)
    lo = 8 if max_input >= 32 else 1
    lengths = plan_values(batch, lo, max_input, rng_for(seed, "lengths"), anchor_near_lo=True)
    caps = plan_values(batch, MIN_CAP if max_output >= MIN_CAP else max_output, max_output, rng_for(seed, "caps"), anchor_near_lo=False)
    if batch > 1:   # pair caps with lengths by a seeded permutation (the longest prompt need not get the largest cap) and shuffle the order
        r = rng_for(seed, "pairing")
        caps = [caps[int(j)] for j in r.permutation(batch)]
        order = [int(j) for j in r.permutation(batch)]
        lengths = [lengths[j] for j in order]
        caps = [caps[j] for j in order]
    if arrivals == "staggered" and batch > 1:
        r = rng_for(seed, "arrivals")
        horizon_s = max(0.1, 0.6 * max_output * NOMINAL_STEP_S)            # arrivals over the first ~60 % of a full-length run
        offs = sorted(round(float(x), 3) for x in r.uniform(0.05, horizon_s, size=batch - 1))
        offsets_s = [0.0] + offs
        steps = [0] + [max(1, int(round(o / NOMINAL_STEP_S))) for o in offs]
    else:
        offsets_s = [0.0] * batch
        steps = [0] * batch
    seeds = [sampling_seed(seed, i) for i in range(batch)] if sampling != "greedy" else []
    return {"schema": SWEEP_SCHEMA, "row_id": row_id, "seed": seed, "batch": batch, "max_input": max_input, "max_output": max_output,
            "min_input": lo, "min_output": MIN_CAP if max_output >= MIN_CAP else max_output,
            "sampling": sampling, "sampling_params": SAMPLING[sampling], "arrivals": arrivals,
            "input_lengths": lengths, "output_caps": caps, "sampling_seeds": seeds, "arrival_offsets_s": offsets_s, "arrive_steps": steps,
            "arrival_rule": {"nominal_step_s": NOMINAL_STEP_S, "arrive_step": "round(offset_s / nominal_step_s)",
                             "schedule": "0 | sorted U[0.05, 0.6*O*nominal_step_s] s (staggered); all 0 (none)"},
            "length_rule": {"input": "anchors I, lo+{0..3}, one awkward per quartile (Q2,Q3,Q1,Q4), then stratified log-uniform over [lo, I] "
                                     "nudged off powers of two; seeded permutation", "output": "same over [2, O]; paired by seeded permutation",
                            "depends_on": "SHA256(row_id) only"},
            "seed_rule": "seed = SHA256(row_id); rng(purpose) = default_rng(SHA256(seed || '/' || purpose)); sampling seed i = SHA256(seed||'/sampling/'||i)[:4]"}


# ---------------------------------------------------------------------------------------------------------------------------------
# the §12 self-check
# ---------------------------------------------------------------------------------------------------------------------------------
def check_plan(p: dict) -> list[str]:
    """BRIEF §12 requirements; returns the list of violations (empty = PASS).  The B >= 8 clauses apply only for B >= 8."""
    B, I, O = int(p["batch"]), int(p["max_input"]), int(p["max_output"])
    L, C = [int(x) for x in p["input_lengths"]], [int(x) for x in p["output_caps"]]
    lo, olo = int(p["min_input"]), int(p["min_output"])
    bad: list[str] = []
    if len(L) != B or len(C) != B:
        bad.append(f"vector lengths {len(L)}/{len(C)} != batch {B}")
    if any(not (lo <= x <= I) for x in L):
        bad.append(f"input length outside [{lo}, {I}]: {L}")
    if any(not (1 <= c <= O) for c in C):
        bad.append(f"output cap outside [1, {O}]: {C}")
    if not any(abs(x - I) <= 3 for x in L):
        bad.append("no request within 3 tokens of max_input")
    if B >= 8:
        if len(set(L)) < 6:
            bad.append(f"only {len(set(L))} distinct input lengths (need >= 6)")
        if not any(x <= lo + 3 for x in L):
            bad.append("no request near the minimum input length")
        qs = {_quartile_of(x, lo, I) for x in L}
        if qs != {0, 1, 2, 3}:
            bad.append(f"input quartiles covered {sorted(qs)} != [0,1,2,3]")
        awk = [x for x in L if not _is_pow2(x) and x % 64 != 0 and x not in (I, lo)]
        if len(awk) < 3:
            bad.append(f"only {len(awk)} awkward non-power-of-two input lengths (need several >= 3)")
        if len(set(C)) < 6:
            bad.append(f"only {len(set(C))} distinct output caps (need >= 6)")
        if not any(c <= max(olo, 4) for c in C):
            bad.append("no very short output cap")
        if not any(abs(c - O) <= 3 for c in C):
            bad.append("no output cap equal/near max_output")
        cq = {_quartile_of(c, olo, O) for c in C}
        if len(cq) < 3:
            bad.append(f"output caps not spread across the range (quartiles {sorted(cq)})")
    if p["sampling"] != "greedy":
        if len(p["sampling_seeds"]) != B or len(set(p["sampling_seeds"])) != B:
            bad.append("stochastic row without B distinct per-request sampling seeds")
    if p["arrivals"] == "staggered" and B > 1:
        offs = p["arrival_offsets_s"]
        if offs[0] != 0.0 or offs != sorted(offs) or len(set(offs)) < min(B, 4):
            bad.append(f"staggered arrival schedule malformed: {offs}")
    return bad


# ---------------------------------------------------------------------------------------------------------------------------------
# text
# ---------------------------------------------------------------------------------------------------------------------------------
class Tok:
    """`tokenizers` tokenizer (tokenizer.json), add_special_tokens=False, like every earlier workload of the repo."""

    def __init__(self, path: str | Path) -> None:
        from tokenizers import Tokenizer

        p = Path(path)
        if p.is_dir():
            p = p / "tokenizer.json"
        if not p.exists():
            raise FileNotFoundError(f"tokenizer.json not found at {p}")
        self.path = p
        self.tk = Tokenizer.from_file(str(p))
        self.sha256 = hashlib.sha256(p.read_bytes()).hexdigest()

    def encode(self, text: str) -> list[int]:
        return list(self.tk.encode(text, add_special_tokens=False).ids)

    def decode(self, ids: list[int]) -> str:
        return self.tk.decode(ids, skip_special_tokens=False)


def hf_home() -> Path:
    env = os.environ.get("HF_HOME")
    if env:
        return Path(env)
    try:
        hh = json.loads(CHECKPOINTS.read_text()).get("hf_home")
        if hh and Path(hh).exists():
            return Path(hh)
    except (OSError, ValueError):
        pass
    return Path.home() / ".cache" / "huggingface"


def find_tokenizer(repo: str, revision: str) -> Path | None:
    base = hf_home() / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots"
    if not base.exists():
        return None
    for d in sorted(base.iterdir()):
        if d.name.startswith(revision) and (d / "tokenizer.json").exists():
            return d / "tokenizer.json"
    return None


def corpus_paragraphs() -> list[dict]:
    doc = json.loads(CORPUS.read_text())
    return [{"text": p, "source": b["source"], "index": i} for b in doc["books"] for i, p in enumerate(b["paragraphs"])]


def build_requests(tok: Tok, p: dict) -> list[dict]:
    paras = corpus_paragraphs()
    B = int(p["batch"])
    rng = rng_for(p["seed"], "prompts")
    starts = [int(s) for s in rng.choice(len(paras), size=min(B, len(paras)), replace=False)]
    while len(starts) < B:                       # B > #paragraphs: reuse starts with a paragraph offset so prefixes still differ
        starts.append((starts[len(starts) - len(paras)] + 7 * (1 + len(starts) // len(paras))) % len(paras))
    reqs = []
    for i in range(B):
        need = int(p["input_lengths"][i])
        ids: list[int] = []
        used: list[str] = []
        j = starts[i]
        while len(ids) < need:
            para = paras[j % len(paras)]
            ids += tok.encode((" " if ids else "") + para["text"])
            used.append(f"{para['source']} ¶{para['index']}")
            j += 1
        ids = ids[:need]
        req = {"request_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"verity-sweep/{p['row_id']}/{i}")), "index": i,
               "prompt_len": need, "max_tokens": int(p["output_caps"][i]), "arrive_step": int(p["arrive_steps"][i]),
               "arrival_offset_s": float(p["arrival_offsets_s"][i]),
               "prompt_token_ids": ids, "prompt_text": tok.decode(ids), "origin": {"corpus": CORPUS.name, "paragraphs": used}}
        if p["sampling_seeds"]:
            req["seed"] = int(p["sampling_seeds"][i])
        reqs.append(req)
    return reqs


def workload_doc(p: dict, *, hf_repo: str, revision: str, case: str | None, tok: Tok | None) -> dict:
    B, I, O = int(p["batch"]), int(p["max_input"]), int(p["max_output"])
    reqs = build_requests(tok, p) if tok is not None else [
        {"request_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"verity-sweep/{p['row_id']}/{i}")), "index": i,
         "prompt_len": int(p["input_lengths"][i]), "max_tokens": int(p["output_caps"][i]), "arrive_step": int(p["arrive_steps"][i]),
         "arrival_offset_s": float(p["arrival_offsets_s"][i]), **({"seed": int(p["sampling_seeds"][i])} if p["sampling_seeds"] else {})}
        for i in range(B)]
    variant = p["sampling"]
    variants = {"greedy": SAMPLING["greedy"]}
    if variant != "greedy":
        variants[variant] = SAMPLING[variant]
    horizon = max(int(r["arrive_step"]) + int(r["max_tokens"]) for r in reqs)
    total_prompt = sum(int(r["prompt_len"]) for r in reqs)
    doc = {
        "schema": SCHEMA, "model": hf_repo, "case": case, "tokenizer_revision": revision, "seed": 0,
        "variants": variants, "seeds": list(p["sampling_seeds"]),
        "sampling": {"variant": variant, "seed": None, "n": 1},
        "serving_profile": f"sweep-B{B}" + ("-staggered" if p["arrivals"] == "staggered" else ""), "context_class": f"I{I}-O{O}",
        "length_seed": p["seed"], "prompt_seed": p["seed"], "replicate": 0,
        "length_rule": p["length_rule"],
        "expected_engine_steps": horizon,
        "engine_args_required": {"max_num_batched_tokens": int(max(2048, total_prompt + B)), "max_model_len": int(I + O)},
        "tokenizer_note": (f"prompt_token_ids from {tok.path.name} sha256 {tok.sha256[:16]} (tokenizers, add_special_tokens=False), natural text "
                           f"from {CORPUS.name}, truncated to prompt_len; prompt_text is the decode of the ids (informational)" if tok else
                           "LENGTHS-ONLY plan: no tokenizer; prompt_token_ids absent (regenerate on a host with the model's tokenizer.json)"),
        "prompt_lengths": sorted({int(r["prompt_len"]) for r in reqs}), "max_tokens": sorted({int(r["max_tokens"]) for r in reqs}),
        "requests": reqs,
        "sweep": {k: p[k] for k in ("schema", "row_id", "seed", "batch", "max_input", "max_output", "min_input", "min_output", "sampling",
                                    "sampling_params", "arrivals", "input_lengths", "output_caps", "sampling_seeds", "arrival_offsets_s",
                                    "arrive_steps", "arrival_rule", "length_rule", "seed_rule")},
        "note": (f"R12 sweep {p['row_id']}: B={B} I<={I} O<={O} {variant} arrivals={p['arrivals']}; prompt lens {p['input_lengths']}, caps "
                 f"{p['output_caps']}, arrive steps {p['arrive_steps']}; every request to max_tokens => {horizon} engine steps; "
                 f"max_num_batched_tokens keeps the whole admitted prefill in one step; max_model_len = I + O.  EOS untouched."),
    }
    doc["sweep"]["lengths_only"] = tok is None
    # [lane vllm-tp-n] a tensor-parallel row (`__tp<N>__` in the row id, N >= 2): the engine's world, the manifests' TP flags of record
    # (#70 / #75: VLLM_BATCH_INVARIANT=1; on a SYS target, where P2P hangs, disable_custom_all_reduce and NCCL_P2P_DISABLE=1) and `sweep.tp`
    world = row_world(p["row_id"])
    if world > 1:
        doc["engine_args_required"].update({"tensor_parallel_size": int(world),
                                            "disable_custom_all_reduce": row_custom_all_reduce_off(p["row_id"])})
        doc["env_required"] = {**row_p2p_env(p["row_id"]), "VLLM_BATCH_INVARIANT": "1"}
        doc["sweep"]["tp"] = int(world)
    return doc


def row_world(row_id: str) -> int:
    """The tensor-parallel world a sweep row id declares (`__tp<N>__`), 1 when absent."""
    m = re.search(r"__tp(\d+)__", str(row_id))
    return int(m.group(1)) if m else 1


def case_for(hf_repo: str, revision: str, explicit: str | None) -> str | None:
    if explicit:
        return explicit
    try:                                    # the adapter's fixed cases first (B0 = SmolLM2, B1 = Qwen2.5-1.5B, ...)
        from verity_vllm.engine.engine_profile import CASES
        for k, v in CASES.items():
            if v.get("repo") == hf_repo:
                return k
    except Exception:  # noqa: BLE001
        pass
    try:
        doc = json.loads(CHECKPOINTS.read_text())
    except (OSError, ValueError):
        return None
    for c in doc.get("checkpoints", []):    # registry roles (`add_checkpoint.py`): a single identifier, e.g. LLAMA32_1B
        role = str(c.get("role") or "")
        if c.get("repo") == hf_repo and str(c.get("revision", "")).startswith(revision) and role.replace("_", "").isalnum():
            return role
    return None


# ---------------------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class Options:
    row_id: str = option("--row-id")
    hf_repo: str = option("--hf-repo")
    revision: str = option("--revision")
    batch: int = option("--batch", type=int)
    max_input: int = option("--max-input", type=int)
    max_output: int = option("--max-output", type=int)
    sampling: str = option("--sampling", default="greedy", choices=sorted(SAMPLING))
    arrivals: str = option("--arrivals", default="none", choices=["none", "staggered"])
    case: str | None = option("--case", default=None,
                              help="checkpoint registry role written to `case` (default: looked up by repo in checkpoints.json)")
    tokenizer: str | None = option("--tokenizer", default=None,
                                   help="snapshot dir or tokenizer.json (default: HF_HOME/hub/models--<repo>/snapshots/<revision>)")
    lengths_only: bool = option("--lengths-only", default=False, action="store_true",
                                help="emit the shape plan without text / token ids (no tokenizer needed)")
    check: bool = option("--check", default=False, action="store_true",
                         help="run the §12 self-check on the plan; exit 1 on a violation")
    out: str | None = option("--out", default=None, help="default workloads/<row-id>.json (relative to the repo)")
    target: str | None = option("--target", default=None,
                                help="[R13 hopper] TargetProfile overrides of the row as JSON (e.g. '{\"flash_attn_version\": 2}' for an `…__bi-eager-fa2` row): written "
                                "to the manifest's `target` block (every GPU stage builds its engine for it; `TargetProfile.engine_args()` is merged into "
                                "`engine_args_required`) and `sweep.execution` = the profile's execution label")


def main(a: Options) -> int:
    p = plan(a.row_id, a.batch, a.max_input, a.max_output, sampling=a.sampling, arrivals=a.arrivals)
    bad = check_plan(p)
    if a.check:
        print(f"[check] {a.row_id}: B={a.batch} lens={p['input_lengths']} caps={p['output_caps']} steps={p['arrive_steps']} -> "
              f"{'PASS' if not bad else 'FAIL: ' + '; '.join(bad)}", file=sys.stderr)
        if bad:
            return 1
    elif bad:
        print(f"[warn] §12 self-check: {'; '.join(bad)}", file=sys.stderr)

    tok = None
    if not a.lengths_only:
        tp = Path(a.tokenizer) if a.tokenizer else find_tokenizer(a.hf_repo, a.revision)
        if tp is None:
            print(f"[error] tokenizer.json for {a.hf_repo}@{a.revision} not found under {hf_home()} — pass --tokenizer or use --lengths-only",
                  file=sys.stderr)
            return 2
        tok = Tok(tp)
    doc = workload_doc(p, hf_repo=a.hf_repo, revision=a.revision, case=case_for(a.hf_repo, a.revision, a.case), tok=tok)
    if a.target:
        from verity_vllm.program.frontend.target_profile import TargetProfile
        t = json.loads(a.target)
        tp = TargetProfile.from_json({**TargetProfile.default().to_json(), **t})   # validates the fields; cc / SMs come from the host at run time
        doc["target"] = t
        doc["engine_args_required"].update(tp.engine_args())
        doc["sweep"]["execution"] = tp.execution_label
        exec_suffix = a.row_id.rsplit("__", 1)[-1]
        if exec_suffix != tp.execution_label:
            print(f"[error] row id execution suffix {exec_suffix!r} != the target's execution label {tp.execution_label!r}", file=sys.stderr)
            return 3
    if tok is not None:
        for r in doc["requests"]:
            assert len(r["prompt_token_ids"]) == r["prompt_len"], (r["index"], len(r["prompt_token_ids"]), r["prompt_len"])
        texts = [tuple(r["prompt_token_ids"]) for r in doc["requests"]]
        assert len(set(texts)) == len(texts), "duplicate prompt token sequences"
    out = Path(a.out) if a.out else WORKLOADS / f"{a.row_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(f"{out}: {doc['note'][:220]}", file=sys.stderr)
    return 0
