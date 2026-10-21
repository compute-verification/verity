"""Fewer SHA-512 compressions without changing the hash: each value that could leave the served Commit, and each change to the
leaf format, priced through the hash lane's commit model as points of a Qwen3-8B B32 run's stock serve time.

    python benchmarks/hash/bytes/bytes_model.py              # the ranked table: points of serve saved per GPU
    python benchmarks/hash/bytes/bytes_model.py --formats    # per format: compressions a leaf, serial chain, opening ANDs
    python benchmarks/hash/bytes/bytes_model.py --commands   # each row as commit_model.py's own command line
    python benchmarks/hash/bytes/bytes_model.py --json
    python benchmarks/hash/bytes/bytes_model.py --check      # the counts reproduce the protocols' and the lane's numbers

The baseline is the card's best SHA-512 cell (note:hash/20261018T0623Z-draft-baseline, Card v2): per-Call units with fused
epilogues, 4 KiB `hm96-sha512/row-seg/v1` leaves in a `frame-v3-sha512` tree, sharaw's prefill rate, salts read ahead, the
decode commit on a second stream: 2.93% / 11.9% / 17.4% of serve on RTX PRO 6000 / H200 / B200.

A byte change removes bytes a token a layer (or the logits) from both phases; `commit_model.set_fused_bytes` takes the new bytes,
so the decode rate follows the smaller window as the model reads it. A format change moves the compressions a committed byte
costs in the data phase (what runs on the step once the salts are read ahead), and the overhead is linear in them:
- prefill is throughput-bound (6.66 GB windows, sharaw at 92.5-95% of raw), so its rate scales by today's compressions a byte
  over the format's;
- decode is latency-bound (a 73 MB window keeps ~6-20% of the GPU busy), so its window time is the 5-40 MB fit's fixed part
  plus the serial chain's change at `T_COMPRESSION` a compression, plus the rest of today's time scaled like prefill
  (note:hash/20261018T0632Z-draft-intpipe sections 5 and 6).
Each priced row goes to `commit_model.fused_cost` as `--fused-bytes` and `--hm96` overrides (`--commands` prints them).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))
import commit_model as cm  # noqa: E402

GPUS = ("RTX PRO 6000", "H200", "B200")
COL, MODE, SECOND_STREAM = "SHA-512 sharaw", "pre", True

# --- bytes -------------------------------------------------------------------------------------------------------------------

LAYERS = 36
#: Bytes a token a layer the fused per-Call units commit (layout_bytes.py's all-three unit on r20261018-031955-d2b3, the model's
#: QWEN_FUSED). `residual` averages 35 of 36 layers' 8,192; `down` is the last layer's, which no residual follows.
KEPT = {"qkv": 12_288, "rope_q": 8_192, "kv_k": 2_048, "attention": 8_192, "residual_attn": 8_192, "residual": 7_964,
        "activation": 24_576, "down": 228}
PREFILL_TOKENS = 2_560
LOGITS = {p.name: p.logits_bytes_per_step for p in cm.qwen("H200").phases}
FUSED = dict(cm.QWEN_FUSED)
#: Bytes a token outside the logits (the layers, the embedding, the inputs), from the prefill step.
TOKEN_BYTES = (FUSED["prefill"] - LOGITS["prefill"]) / PREFILL_TOKENS

# --- leaf formats ------------------------------------------------------------------------------------------------------------

BLOCK, PAD = 128, 17                   # SHA-512's block, and its padding's least (0x80 and the 16-byte length)
ANDS = 58_120                          # a private SHA-512 compression in C-Flock (hm96/PROTOCOL.md section 5)
T_COMPRESSION = 4.8                    # µs, one serial SHA-512 compression at decode occupancy (intpipe section 5)
#: The fixed part of a fit of window time against bytes over 5-40 MB, salts read ahead, eager (intpipe section 5), µs.
DECODE_FIXED_US = {"B200": 218.0, "H200": 208.0, "RTX PRO 6000": 196.0}


def blocks(nbytes: int) -> int:
    """Compressions over `nbytes` of message after a constant-midstate prefix block, SHA-512's padding included."""
    return -(-(nbytes + PAD) // BLOCK)


def node_blocks(arity: int) -> int:
    """A frame-v3-sha512 node's data compressions: its first 128 bytes (frame, tag, domain id, level, the index length's top
    bytes) are one midstate a level, and the rest is the index length's low bytes, an index under 2^24 and the children."""
    return blocks(5 + arity * (8 + 64))


@dataclass(frozen=True)
class Format:
    """A row leaf: `leaf` row bytes; `seg` the row-seg segment (None: `sha512/row/v2`, one message); `leafhash` "hm96+frame"
    (today: the hm96 leaf h over b ‖ c, then the frame leaf over h), "folded" (one hash of a per-domain constant block ‖ rank ‖
    position ‖ b ‖ c) or "none" (b ‖ c straight into the tree's bottom nodes); `arity` the frame tree's."""
    leaf: int = 4096
    seg: int | None = 1024
    leafhash: str = "hm96+frame"
    arity: int = 2

    @property
    def label(self) -> str:
        seg = "row/v2" if self.seg is None else f"row-seg {self.seg // 1024} KiB"
        return f"{self.leaf // 1024} KiB {seg}, {self.leafhash}, {self.arity}-ary"

    def row(self) -> tuple[int, int]:
        """(compressions, serial chain) of the row digest over one leaf."""
        if self.seg is None:
            return blocks(self.leaf), blocks(self.leaf)
        n, per = self.leaf // self.seg, blocks(self.seg)
        outer = blocks(64 * n)
        return n * per + outer, per + outer

    def bottom(self) -> tuple[float, int]:
        """(compressions a leaf, chain) from b ‖ c to the tree's first level of 64-byte digests."""
        if self.leafhash == "hm96+frame":
            return 3, 3                # h: 2 (constant key block, then b ‖ c and padding); the frame leaf's last block: 1
        if self.leafhash == "folded":
            return blocks(8 + 8 + 128), blocks(8 + 8 + 128)
        b = blocks(5 + self.arity * (8 + 128))
        return b / self.arity, b

    def tree_share(self) -> float:
        share = node_blocks(self.arity) / (self.arity - 1)
        return share / self.arity if self.leafhash == "none" else share

    def data(self) -> float:
        """Data-phase compressions a leaf: what runs on the step with the salts (c, M·y) and every position-only block ahead."""
        return self.row()[0] + self.bottom()[0] + self.tree_share()

    def per_4k(self) -> float:
        return self.data() * 4096 / self.leaf

    def chain(self, nbytes: float) -> int:
        """The serial chain of a window of `nbytes`: a segment, the outer digest, the leaf, then one node a level."""
        levels = max(1, math.ceil(math.log(max(nbytes / self.leaf, 2), self.arity)))
        upper = levels - 1 if self.leafhash == "none" else levels
        return self.row()[1] + self.bottom()[1] + upper * node_blocks(self.arity)

    def opening_ands(self) -> int:
        """ANDs to open one leaf in C-Flock: the row digest's private compressions and the salt digest c (2); M·y is XORs, and
        the leaf and tree are native at the verifier."""
        return (self.row()[0] + 2) * ANDS

    def two_rows_ands(self, row_bytes: int = 4096) -> tuple[int, int]:
        """A drawn unit opening two rows of `row_bytes` (compute accounting's two K-byte rows): (each in its own leaves, both
        in one leaf when a leaf holds them)."""
        apart = 2 * math.ceil(row_bytes / self.leaf) * self.opening_ands()
        together = math.ceil(2 * row_bytes / self.leaf) * self.opening_ands()
        return apart, together


TODAY = Format()

# --- changes -----------------------------------------------------------------------------------------------------------------


#: Bytes a token a layer of KV rows: what an attention unit reads one head slice (256 B of K) or one element (of V) of, per key.
KV = 2_048 + 2_048


@dataclass(frozen=True)
class Change:
    key: str
    tier: str                                         # "bytes" (rules kept), "format" (proofs' format), "protocol" (a rule)
    name: str
    owner: str
    rule: str
    saves: int = 0                                    # bytes a token a layer that leave the serve-time Commit
    logits: bool = False                              # the logits leave too
    kv: int = KV                                      # bytes a token a layer of KV rows left in the Commit
    fmt: dict[str, Format] = field(default_factory=lambda: {"prefill": TODAY, "decode": TODAY})
    kv_fmt: dict[str, Format] | None = None           # the KV rows' format where it differs (they're read a slice a key)
    note: str = ""

    def per_4k(self, phase: str) -> float:
        """Data compressions a 4 KiB of the phase's bytes, the KV rows at their own format."""
        share = self.kv / (sum(KEPT.values()) - self.saves)
        kv = (self.kv_fmt or self.fmt)[phase]
        return share * kv.per_4k() + (1 - share) * self.fmt[phase].per_4k()

    def chain(self, phase: str, nbytes: float) -> int:
        """One tree a window: its serial chain is the longer of the two formats' paths."""
        return max(self.fmt[phase].chain(nbytes), (self.kv_fmt or self.fmt)[phase].chain(nbytes))


def both(f: Format) -> dict[str, Format]:
    return {"prefill": f, "decode": f}


Q_ROPE, K_ROPE = KEPT["rope_q"], KEPT["kv_k"]         # recomputed in the attention unit from qkv's q and k
V_STORE = 2_048                                       # V read from the KV store's 4 KiB token-layer block
INT8_ACT = KEPT["activation"] // 2                    # down's operand as s8 with its row scale
INT8_ATT = KEPT["attention"] // 2                     # o_proj's operand as s8 with its row scale
#: Bytes a token a layer at a replay unit's boundary under two-stage sampling: the residual stream and the KV it writes (a
#: decoder layer), and the post-attention residual too (an attention block and an MLP block).
RU_LAYER, RU_BLOCK = 8_192 + KV, 8_192 + 8_192 + KV
DECODE_FMT = Format(4096, 1024, "folded", 4)
PROPOSED = {"prefill": Format(8192, None, "none", 4), "decode": DECODE_FMT}
KV_FMT = both(DECODE_FMT)
TODAY_BYTES = sum(KEPT.values())
CUT = "a unit's cone becomes a whole row: "

CHANGES = (
    Change("today", "-", "today (card v2's best SHA-512 cell)", "-", "-"),
    Change("q-rope", "bytes", "q's per-head norm + RoPE recomputed in the attention unit from qkv's q: rope_q not committed",
           "proofs (units: PRF-62's prologue, one head of 128 a unit); circuits (Commit layout)",
           "none: every value left is committed, and the attention unit's output stays 16 bits", saves=Q_ROPE),
    Change("qk-rope", "bytes", "q's and k's norm + RoPE in the attention unit (k's per key read): rope_q and kv_k not committed",
           "proofs; circuits", "none; the attention unit gains a 128-element norm and rotation a key (about its QK^T dot again)",
           saves=Q_ROPE + K_ROPE,
           note="the epilogue form (the qkv unit outputs rotated q, k) saves the same bytes but makes a q or k unit compute "
                "its head's 128 dots, since each of its units outputs 16 bits"),
    Change("qk-rope+v-store", "bytes", "the above, and V read from memory accounting's KV store (4 KiB token-layer "
           "hm96-sha512/row/v2 blocks), not the Commit", "memory-accounting (the store); proofs (attention opens store "
           "leaves); circuits", "every value committed: V by the store, so only where the store runs; zero knowledge: the "
           "store's openings reveal KV today", saves=Q_ROPE + K_ROPE + V_STORE, kv=2_048,
           note="K stays: the store holds rotated K, the Commit raw k; or rotated K from the store and raw k kept, same bytes"),
    Change("int8-act", "bytes", "down's operand committed as s8 with its row scale (VLLM-18's exact-int8 linear): activation at "
           "half its bf16 bytes", "vllm (VLLM-18 profile for Qwen3-8B); proofs; circuits",
           CUT + "X = round(a / s_a) needs s_a = max|a| / 127 over the row, so the s8 producer reads 12,288 activations (24,576 "
                 "dots), and s_a is 32 bits (two 16-bit units)", saves=INT8_ACT,
           note="as built, VLLM-18 quantizes inside the linear: bf16 in, s8 internal, so today's Commit is unchanged"),
    Change("int8-act+att", "bytes", "down's and o_proj's operands as s8 (activation and attention output)", "vllm; proofs; circuits",
           CUT + "as int8-act, and the attention row's scale reads 32 heads' attention", saves=INT8_ACT + INT8_ATT),
    Change("logits", "bytes", "lm_head and the sampler in one unit: the sampled token committed, no logits", "proofs; circuits",
           CUT + "a token unit reads the 151,936-logit row (622M MACs), and an 18-bit token is two units", logits=True,
           note="logits are already committed only at sampled positions (6.1 MB a prefill step, 7.7 MB a decode step)"),
    Change("leaf-8k", "format", "8 KiB leaves, 1 KiB segments", "proofs (PRF-64 format)", "none", fmt=both(Format(8192))),
    Change("leaf-16k", "format", "16 KiB leaves, 1 KiB segments", "proofs (PRF-64 format)", "none", fmt=both(Format(16384))),
    Change("seg-2k", "format", "2 KiB segments at 4 KiB leaves", "proofs (row-seg, Daniel's card 637f70c6)", "none",
           fmt=both(Format(4096, 2048))),
    Change("row-v2", "format", "sha512/row/v2 rows at 4 KiB (one segment; an existing schema)", "proofs (card 637f70c6)", "none",
           fmt=both(Format(4096, None))),
    Change("row-v2-prefill", "format", "sha512/row/v2 at 4 KiB in prefill windows only; decode keeps row-seg", "proofs", "none",
           fmt={"prefill": Format(4096, None), "decode": TODAY}),
    Change("fold", "format", "hm96 leaf folded into the frame leaf: H(domain block ‖ rank ‖ position ‖ b ‖ c)",
           "proofs (frame-v3, hm96)", "none; RowBinding's reduction loses a step", fmt=both(Format(4096, 1024, "folded"))),
    Change("tree-4ary", "format", "4-ary frame tree (a node 3 compressions over 4 children)", "proofs (frame-v3)", "none",
           fmt=both(Format(4096, 1024, "hm96+frame", 4))),
    Change("no-leaf-hash", "format", "b ‖ c straight into 4-ary bottom nodes (no leaf hash), 4-ary tree", "proofs (frame-v3, hm96)",
           "none; RowBinding's reduction loses two steps", fmt=both(Format(4096, 1024, "none", 4))),
    Change("format", "format", "prefill: 8 KiB row/v2 leaves (a hidden row), b ‖ c into 4-ary nodes; KV rows and decode: 4 KiB "
           "row-seg, folded leaf, 4-ary", "proofs", "none", fmt=PROPOSED, kv_fmt=KV_FMT),
    Change("format-16k", "format", "as format, 16 KiB prefill leaves", "proofs", "none",
           fmt={"prefill": Format(16384, None, "none", 4), "decode": DECODE_FMT}, kv_fmt=KV_FMT),
    Change("qk-rope+format", "bytes", "qk-rope + format: the most that keeps every rule", "proofs; circuits", "none",
           saves=Q_ROPE + K_ROPE, fmt=PROPOSED, kv_fmt=KV_FMT,
           note="qkv's row laid out q | k ‖ v, with k ‖ v one 4 KiB leaf at the KV format: the attention unit's per-key reads"),
    Change("bytes+format", "bytes", "qk-rope + V from the store + format", "proofs; memory-accounting; circuits", "as the rows",
           saves=Q_ROPE + K_ROPE + V_STORE, kv=2_048, fmt=PROPOSED, kv_fmt=KV_FMT),
    Change("two-stage-block", "protocol", "two-stage sampling (decision 43) with a block per replay unit: serve commits the "
           "residual stream, the post-attention residual and KV; interiors only for drawn RUs, after the draw",
           "proofs; Daniel (retired: ASK-25, 18 Oct, commits everything at serve time)",
           "every value committed: an undrawn RU's interiors never are; the profile is over RUs, each charged at its "
           "class's largest", saves=TODAY_BYTES - RU_BLOCK, logits=True),
    Change("two-stage-layer", "protocol", "two-stage sampling with a decoder layer per replay unit: serve commits the residual "
           "stream and KV", "proofs; Daniel", "as two-stage-block", saves=TODAY_BYTES - RU_LAYER, logits=True),
    Change("two-stage-layer+format", "protocol", "two-stage-layer + format", "proofs; Daniel", "as two-stage-block",
           saves=TODAY_BYTES - RU_LAYER, logits=True, fmt=PROPOSED, kv_fmt=KV_FMT),
    Change("two-stage-block+format", "protocol", "two-stage-block + format", "proofs; Daniel", "as two-stage-block",
           saves=TODAY_BYTES - RU_BLOCK, logits=True, fmt=PROPOSED, kv_fmt=KV_FMT),
)


def step_bytes(c: Change) -> dict[str, float]:
    keep = 1 - LAYERS * c.saves / TOKEN_BYTES
    return {ph: (FUSED[ph] - LOGITS[ph]) * keep + (0.0 if c.logits else LOGITS[ph]) for ph in FUSED}


def decode_rate(gpu: str, nbytes: float, gbps: float, c: Change) -> float:
    t = nbytes / (gbps * cm.GB) * 1e6
    fixed = DECODE_FIXED_US[gpu]
    t_new = (fixed + (c.chain("decode", nbytes) - TODAY.chain(nbytes)) * T_COMPRESSION
             + max(t - fixed, 0.0) * c.per_4k("decode") / TODAY.per_4k())
    return nbytes / (t_new * 1e-6) / cm.GB


def price(c: Change, gpu: str, second_stream: bool = SECOND_STREAM) -> dict:
    """The run's overhead under `c` and the overrides that give it, through commit_model."""
    b = step_bytes(c)
    cm.set_fused_bytes(b["prefill"], b["decode"])
    try:
        base = cm.fused_rates(gpu, COL, MODE)
        rates = (base["prefill"] * TODAY.per_4k() / c.per_4k("prefill"), decode_rate(gpu, b["decode"], base["decode"], c))
        cost = cm.fused_cost(gpu, COL, MODE, second_stream, {(gpu, COL, MODE): rates})
    finally:
        cm.set_fused_bytes(FUSED["prefill"], FUSED["decode"])
    return {"bytes": b, "rates": rates, "run": cost["run"], "prefill_ms": cost["prefill_ms"], "decode_ms": cost["decode_ms"]}


def table() -> list[dict]:
    rows = []
    base = {g: price(CHANGES[0], g)["run"] for g in GPUS}
    for c in CHANGES:
        p = {g: price(c, g) for g in GPUS}
        run_bytes = sum(p["H200"]["bytes"][ph.name] * ph.steps for ph in cm.qwen("H200").phases)
        rows.append({**asdict(c), "fmt": {ph: f.label for ph, f in c.fmt.items()},
                     "kv_fmt": c.kv_fmt and {ph: f.label for ph, f in c.kv_fmt.items()},
                     "per_4k": {ph: c.per_4k(ph) for ph in c.fmt}, "run_gb": run_bytes / cm.GB,
                     "points": {g: 100 * (base[g] - p[g]["run"]) for g in GPUS}, "run_pct": {g: 100 * p[g]["run"] for g in GPUS},
                     "rates": {g: p[g]["rates"] for g in GPUS}, "bytes": p["H200"]["bytes"]})
    return rows


def command(c: Change, gpu: str) -> str:
    p = price(c, gpu)
    b, (pre, dec) = p["bytes"], p["rates"]
    return (f"python3 benchmarks/hash/model/commit_model.py --card '{gpu}' --fused-bytes {b['prefill']:.0f},{b['decode']:.0f} "
            f"--hm96 '{gpu}/{COL}/{MODE}={pre:.1f},{dec:.1f}'   # {c.key}: second-stream row, column {COL}, first value")


def formats_table() -> str:
    fs = [("today", TODAY), *((c.key, c.fmt["prefill"]) for c in CHANGES if c.fmt["prefill"] != TODAY and c.key != "format-16k"),
          ("format, decode", PROPOSED["decode"]), ("8 KiB row/v2, hm96+frame", Format(8192, None)),
          ("16 KiB row/v2, none, 4-ary", Format(16384, None, "none", 4))]
    seen, rows = set(), []
    head = ("| format | data compressions a leaf (row + leaf + tree) | a 4 KiB of bytes | x today | chain at 73 MB | "
            "opening ANDs a leaf | two 4 KiB rows apart / together |")
    rows += [head, "|---|---|---|---|---|---|---|"]
    for name, f in fs:
        if f in seen:
            continue
        seen.add(f)
        apart, together = f.two_rows_ands()
        rows.append(f"| {name}: {f.label} | {f.data():.2f} ({f.row()[0]} + {f.bottom()[0]:g} + {f.tree_share():.2f}) | "
                    f"{f.per_4k():.2f} | {TODAY.per_4k() / f.per_4k():.3f} | {f.chain(FUSED['decode'])} | {f.opening_ands() / 1e6:.2f}M | "
                    f"{apart / 1e6:.2f}M / {together / 1e6:.2f}M |")
    return "\n".join(rows)


def ranked() -> str:
    rows = sorted(table(), key=lambda r: -r["points"]["H200"])
    out = ["| change | tier | owner | rule touched | GB a run | compressions a 4 KiB, prefill / decode | points of serve saved: "
           "RTX PRO 6000 / H200 / B200 | then % of serve |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        pts = " / ".join(f"{r['points'][g]:.2f}" for g in GPUS)
        pct = " / ".join(f"{r['run_pct'][g]:.2f}" for g in GPUS)
        out.append(f"| {r['key']}: {r['name']} | {r['tier']} | {r['owner']} | {r['rule']} | {r['run_gb']:.2f} | "
                   f"{r['per_4k']['prefill']:.1f} / {r['per_4k']['decode']:.1f} | {pts} | {pct} |")
    return "\n".join(out)


def check() -> list[str]:
    bad = []
    if sum(KEPT.values()) != 71_680:
        bad.append(f"KEPT sums to {sum(KEPT.values())}, layout_bytes.py's all-three unit keeps 71,680 B a token a layer")
    if (TODAY.row(), TODAY.data(), TODAY.chain(FUSED["decode"])) != ((39, 12), 44, 45):
        bad.append(f"today's leaf: row {TODAY.row()}, data {TODAY.data()}, chain {TODAY.chain(FUSED['decode'])}; intpipe counts "
                   "row-seg 4 x 9 + 3, 44 a leaf with salts ahead and a ~45-compression chain")
    if Format(8192).row()[0] != 77:
        bad.append("frame_v3/PROTOCOL.md section 6: an 8 KiB row-seg row is 8 x 9 + 5 = 77 compressions")
    if Format(4096, None).row()[0] * ANDS + 2 * ANDS != 2_034_200:
        bad.append("hm96/PROTOCOL.md section 6: a 4,096-byte hm96-sha512 row costs 2,034,200 ANDs")
    if (node_blocks(2), node_blocks(8)) != (2, 5):
        bad.append("a binary node is 2 data compressions after its level's midstate, and intpipe's 709-byte 8-ary node 5")
    for g, want in (("RTX PRO 6000", 2.93), ("H200", 11.9), ("B200", 17.4)):
        got = 100 * price(CHANGES[0], g)["run"]
        if abs(got - want) > 0.05:
            bad.append(f"{g}: today prices at {got:.2f}% of serve, card v2's best SHA-512 cell is {want}%")
    if abs(sum(step_bytes(CHANGES[0])[p.name] * p.steps for p in cm.qwen("H200").phases) / 31.331088384e9 - 1) > 1e-9:
        bad.append("today's bytes do not sum to the fused 31.331 GB a run")
    try:
        from verity.ml.commitments import merkle
        from verity.ml.commitments.indexed import RangeIndexedDomain
        d = merkle.CommitmentDomain(bytes(64), -1, RangeIndexedDomain(0, 1 << 20), "sha512")
        msgs = [merkle._message(merkle._NODE, d.domain_id, merkle._uint(3), merkle._uint(i), bytes(64), bytes(64))
                for i in (1, 300, 70_000)]
        if len({m[:BLOCK] for m in msgs}) != 1 or max(blocks(len(m) - BLOCK) for m in msgs) != node_blocks(2):
            bad.append("verity's frame-v3-sha512 node: the first block is not constant a level, or the rest is not 2 compressions")
        lp = d.leaf_prefix(70_000, 70_000, "hm96-sha512/row-seg/v1", 64)
        if blocks(len(lp) + 64 - BLOCK) != 1:
            bad.append("verity's frame-v3-sha512 row leaf is not one position-only block then one data block")
    except ImportError:
        print("check: verity not importable; the frame message checks are skipped (run from the workspace with uv run)")
    return bad


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--formats", action="store_true")
    ap.add_argument("--commands", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    if a.check:
        bad = check()
        print("\n".join(bad) if bad else "ok: layout bytes, leaf counts, opening ANDs, node messages and card v2's cells reproduce")
        return 1 if bad else 0
    if a.json:
        print(json.dumps({"baseline": {"column": COL, "mode": MODE, "second_stream": SECOND_STREAM},
                          "kept_token_layer": KEPT, "rows": table(),
                          "formats": {f.label: {"data": f.data(), "per_4k": f.per_4k(), "chain_73mb": f.chain(FUSED["decode"]),
                                                "opening_ands": f.opening_ands(), "two_rows_ands": f.two_rows_ands()}
                                      for c in CHANGES for fm in (c.fmt, c.kv_fmt or {}) for f in fm.values()}},
                         indent=1, ensure_ascii=False))
        return 0
    if a.commands:
        print("\n".join(command(c, g) for c in CHANGES for g in GPUS))
        return 0
    if a.formats:
        print(formats_table())
        return 0
    print(ranked())
    return 0


if __name__ == "__main__":
    sys.exit(main())
