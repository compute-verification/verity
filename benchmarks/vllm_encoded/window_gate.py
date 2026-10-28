"""`paged.cu`'s decode window (`pg_window`) on one GPU: its gate, then its cost a decode step at Llama-3.1-8B's pool.

The window is what the exact-int8 attention reads (`verity_vllm.engine.int8_attention`: `window_of`, then
`Int8KvQuant` and `Int8LutAttentionRow` on each KV head's [S][D]): a request's tokens 0 … s − 1, K and V as bf16,
[rows][KV heads][head_dim]. The reference is vLLM's own: `int8_attention.window_of` over the plain cache's K and V
views as `Int8AttentionImpl.forward` takes them (`kv_cache.transpose(1, 2).split(head_size, -1)`). For each k of
`--ks` and two caches,

    vllm      vLLM's layout today, [pages][KV heads][slots][K ‖ V], two layers a store (store_base 77 TLW), the codec's
              word order 0 (pg_wi's), the encoded slots appended by `pg_append` under their bound offsets and the
              plaintext slots appended as they are with their flags set (deferred residency)
    planned   [pages][slots][KV heads][K ‖ V], a layer to a region (layers 1, store_base 4099 TLW), word order 1 (each
              head's K ‖ V): the encoded slots `kv_encode_bound` of the flat cache, so codec index = store_base plus
              the physical word

over random block tables (windows of 0, 1, 15, 16, 17, 63, 64, 65, 300 and 511 tokens and random ones across scattered
pages, two padding requests of length 0), each check bit for bit:

    copy        encoded 0 on the plain cache equals the reference
    mixed       encoded 1 on the mixed cache, under the bound offsets and the flags, equals it; without the flags
                (every slot decoded) and under the seed's key it differs
    kv_py       sampled words of the window equal `kv.py`'s decode (kv.cu's reference) of the cache's word at its
                codec index and offset, or the word itself at a plaintext slot
    last_rows   each request's last token from the step's K and V rows (ck, cv, at permuted rows qrow) with its slot
                in the pages overwritten where no other token reads it: equals the reference
    head_major  the buffer as [KV heads][rows][head_dim] (another row and head stride): equals it
    cap         a buffer shorter than the step's rows: the rows it holds equal it, the rest are untouched
    graph       (vllm) offsets and window captured in a CUDA graph, then new lengths and block tables written in
                place and half the plaintext slots encoded in place by `pg_encode` (their flags cleared): the replay
                equals the reference of the new step
    refusals    a misaligned buffer, a row stride not 16 bytes, an unknown word order, k = 0, flags or offsets on a
                plain copy, ck without cv

Timing (`--pages` pages a layer, `--layers` layers, vLLM's layout, random words, every slot under its own offset): a
decode step's windows, all layers, at each `--points` batch × context, each replayed from a CUDA graph: the decode
(`--k`, encoded, no slot plaintext; and half the slots plaintext), the copy (encoded 0), vLLM's indexing (`window_of`'s
gather, over the whole batch at once) and a contiguous copy of the same bytes; the decode also launched eagerly. Writes
OUT/window-gate.json.

    python window_gate.py --lib p1024=paged.so --out OUT [--ks 2,3 --pages 23040 --layers 32 --k 3]
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parents[1] / "integrations" / "vllm")]

import paged as P  # noqa: E402
from paged import K, KB  # noqa: E402
from verity_vllm.engine import int8_attention as IA  # noqa: E402

BLOCKS, BS = 96, 16
LENS = (0, 1, 15, 16, 17, 63, 64, 65, 300, 511)
PADDING = 2
SB_VLLM, SB_PLANNED = 77, 4099
SAMPLES = 96


def eq(torch, a, b) -> bool:
    return bool(torch.equal(a.view(torch.int16), b.view(torch.int16)))


class Case:
    """A cache in one layout and word order, its plain twin, flags, offsets and the views `window_of` reads."""

    def __init__(self, lib, torch, name: str, k: int, key, rng: random.Random) -> None:
        G, D, WB, f = lib.shape["kv_heads"], lib.shape["head_dim"], lib.shape["wb"], lib.field
        self.name, self.k, self.key, self.f = name, k, key, f
        self.nslots, self.W = BLOCKS * BS, K.token_layer_words(f)
        self.order, self.layers = (0, 2) if name == "vllm" else (1, 1)
        self.layer = 1 if name == "vllm" else 0
        self.store_base = (SB_VLLM if name == "vllm" else SB_PLANNED) * self.W
        shape = (BLOCKS, G, BS, 2 * D) if name == "vllm" else (BLOCKS, BS, G, 2 * D)
        self.plain = (torch.randn(*shape, device="cuda") * 2).to(torch.bfloat16)
        self.lay = P.layout(self.plain, G, WB, (0, 1, 2) if name == "vllm" else (0, 2, 1))
        self.offs = [rng.randrange(f.p) for _ in range(self.nslots)]
        self.bj = P.offsets(torch, self.offs, f)
        self.is_plain = [rng.random() < 0.5 for _ in range(self.nslots)]
        want = torch.tensor(self.is_plain, dtype=torch.uint8, device="cuda")
        if name == "vllm":
            keys = self.plain[..., :D].permute(0, 2, 1, 3).reshape(self.nslots, G * D).contiguous()
            vals = self.plain[..., D:].permute(0, 2, 1, 3).reshape(self.nslots, G * D).contiguous()
            self.mix = torch.empty_like(self.plain)
            self.flags = torch.full((self.nslots,), 7, dtype=torch.uint8, device="cuda")
            for encd in (0, 1):
                sl = torch.tensor([s for s in range(self.nslots) if self.is_plain[s] != bool(encd)], dtype=torch.int64,
                                  device="cuda")
                P.append(lib, torch, self.mix, self.lay, self.store_base, self.layers, self.layer, encd, k, keys[sl],
                         vals[sl], sl, bj=self.bj if encd else None, plain=self.flags)
            self.kc, self.vc = self.plain.transpose(1, 2).split(D, dim=-1)
        else:
            if self.W != G * lib.shape["hw"]:
                raise RuntimeError(f"a token-layer is {self.W} words, not its heads' {G * lib.shape['hw']}")
            enc = torch.empty_like(self.plain)
            lib.call("kv_encode_bound", self.plain.data_ptr(), enc.data_ptr(), self.nslots * self.W, self.store_base, k,
                     self.bj.data_ptr(), 0, P.stream(torch))
            self.mix = torch.where(want.bool().view(BLOCKS, BS, 1, 1), self.plain.view(torch.int16),
                                   enc.view(torch.int16)).view(torch.bfloat16)
            self.flags = want.clone()
            self.kc, self.vc = self.plain.split(D, dim=-1)
        torch.cuda.synchronize()
        self.flags_ok = torch.equal(self.flags, want)

    def window(self, lib, torch, src, lens, bt, wo, batch, outk, outv, encoded=1, flags=True, bound=True, **kw) -> None:
        P.window(lib, torch, src, self.lay, self.store_base, self.layers, self.layer, encoded, self.k, lens, bt, wo,
                 batch, outk, outv, order=self.order, bj=self.bj if encoded and bound else None,
                 plain=self.flags if encoded and flags else None, **kw)

    def codec_index(self, slot: int, g: int, j: int, hw: int, wb: int) -> int:
        wi = g * hw + j if self.order else (0 if j < hw // 2 else 2048 // wb - hw // 2) + g * (hw // 2) + j
        return self.store_base + (slot * self.layers + self.layer) * self.W + wi


def step(rng: random.Random, batch_lens: list[int], bstride: int):
    """Lengths (the padding last) and block-table rows over scattered pages, pages shared among requests."""
    lens = list(batch_lens) + [0] * PADDING
    perm = rng.sample(range(BLOCKS), BLOCKS)
    bt = [[0] * bstride for _ in lens]
    for i, n in enumerate(lens):
        for b in range(-(-n // BS)):
            bt[i][b] = perm[(7 * i + 3 * b) % BLOCKS]
    return lens, bt


def reference(torch, case: Case, lens: list[int], bt) -> tuple[list, list]:
    ks, vs = [], []
    for i, n in enumerate(lens):
        k_, v_ = IA.window_of(case.kc, case.vc, bt[i], n)
        ks.append(k_)
        vs.append(v_)
    return ks, vs


def matches(torch, outk, outv, wo: list[int], lens: list[int], ref, rows: int | None = None) -> bool:
    ok = True
    for i, n in enumerate(lens):
        a, b = wo[i], wo[i] + n
        if rows is not None:
            b = min(b, rows)
        if b <= a:
            continue
        ok &= eq(torch, outk[a:b], ref[0][i][:b - a]) and eq(torch, outv[a:b], ref[1][i][:b - a])
    return ok


def gate_case(lib, torch, case: Case, rng: random.Random, log) -> dict:
    G, D, WB, HW = (lib.shape[x] for x in ("kv_heads", "head_dim", "wb", "hw"))
    batch_lens = list(LENS) + [rng.randrange(1, 513) for _ in range(3)]
    rng.shuffle(batch_lens)
    bstride = 32 + 3
    lens, bt = step(rng, batch_lens, bstride)
    batch = len(lens)
    lens_d = torch.tensor(lens, dtype=torch.int32, device="cuda")
    bt_d = torch.tensor(bt, dtype=torch.int32, device="cuda")
    wo_d = torch.full((batch + 1,), -5, dtype=torch.int32, device="cuda")
    P.window_offsets(lib, torch, lens_d, batch, wo_d)
    torch.cuda.synchronize()
    wo = wo_d.cpu().tolist()
    rows = sum(lens)
    out = {"batch": batch, "rows": rows, "flags": case.flags_ok,
           "offsets": wo == [sum(lens[:i]) for i in range(batch + 1)]}
    ref = reference(torch, case, lens, bt_d)

    def buffers(n=rows, fill=3.0):
        return (torch.full((n, G, D), fill, dtype=torch.bfloat16, device="cuda"),
                torch.full((n, G, D), -fill, dtype=torch.bfloat16, device="cuda"))

    ok_, ov_ = buffers()
    case.window(lib, torch, case.plain, lens_d, bt_d, wo_d, batch, ok_, ov_, encoded=0)
    mk, mv = buffers()
    case.window(lib, torch, case.mix, lens_d, bt_d, wo_d, batch, mk, mv)
    nk, nv = buffers()
    case.window(lib, torch, case.mix, lens_d, bt_d, wo_d, batch, nk, nv, flags=False)
    sk_, sv_ = buffers()
    case.window(lib, torch, case.mix, lens_d, bt_d, wo_d, batch, sk_, sv_, bound=False)
    torch.cuda.synchronize()
    read = sorted({int(bt[i][p // BS]) * BS + p % BS for i, n in enumerate(lens) for p in range(n)})
    out["copy"] = matches(torch, ok_, ov_, wo, lens, ref)
    out["mixed"] = matches(torch, mk, mv, wo, lens, ref)
    out["slots_read"] = {"plaintext": sum(case.is_plain[s] for s in read),
                         "encoded": sum(not case.is_plain[s] for s in read)}
    out["unflagged_differs"] = not matches(torch, nk, nv, wo, lens, ref)
    out["seed_key_differs"] = not matches(torch, sk_, sv_, wo, lens, ref)

    w8 = case.mix.view(torch.uint8).view(-1, WB)
    picks = []
    live = [(i, t) for i, n in enumerate(lens) for t in range(n)]
    for i, t in rng.sample(live, min(SAMPLES, len(live))):
        g, j = rng.randrange(G), rng.randrange(HW)
        blk, so = bt[i][t // BS], t % BS
        slot = blk * BS + so
        y = int.from_bytes(bytes(w8[blk * case.lay.sb + g * case.lay.sh + so * case.lay.ss + j].cpu().tolist()), "little")
        want = y if case.is_plain[slot] else K.decode_word(y, case.codec_index(slot, g, j, HW, WB), case.key,
                                                           case.offs[slot])
        half = (mk if j < HW // 2 else mv)[wo[i] + t, g, (j % (HW // 2)) * (WB // 2):(j % (HW // 2) + 1) * (WB // 2)]
        got = int.from_bytes(half.contiguous().view(torch.uint8).cpu().numpy().tobytes(), "little")
        picks.append((got == want, case.is_plain[slot]))
    out["kv_py"] = {"words": len(picks), "plaintext": sum(p for _, p in picks), "ok": all(g for g, _ in picks)}

    nrows = batch + 3
    qrow = torch.tensor(rng.sample(range(nrows), batch), dtype=torch.int32, device="cuda")
    ck = torch.zeros(nrows, G * D, dtype=torch.bfloat16, device="cuda")
    cv = torch.zeros_like(ck)
    mixed_x = case.mix.clone()
    readers: dict[tuple, int] = {}
    for i, n in enumerate(lens):
        for p in range(n):
            readers[(bt[i][p // BS], p % BS)] = readers.get((bt[i][p // BS], p % BS), 0) + 1
    overwritten = 0
    qr = qrow.cpu().tolist()
    for i, n in enumerate(lens):
        if not n:
            continue
        blk, so = bt[i][(n - 1) // BS], (n - 1) % BS
        ck[qr[i]] = ref[0][i][n - 1].reshape(-1)
        cv[qr[i]] = ref[1][i][n - 1].reshape(-1)
        if readers[(blk, so)] == 1:
            junk = (torch.randn(G, 2 * D, device="cuda") * 9).to(torch.bfloat16)
            if case.name == "vllm":
                mixed_x[blk, :, so] = junk
            else:
                mixed_x[blk, so] = junk
            overwritten += 1
    lk, lv = buffers()
    case.window(lib, torch, mixed_x, lens_d, bt_d, wo_d, batch, lk, lv, ck=ck, cv=cv, qrow=qrow)
    hk, hv = (torch.full((G, rows, D), c, dtype=torch.bfloat16, device="cuda") for c in (1.0, -1.0))
    case.window(lib, torch, case.mix, lens_d, bt_d, wo_d, batch, hk.permute(1, 0, 2), hv.permute(1, 0, 2))
    cap = rows - 37
    ck_, cv_ = buffers()
    case.window(lib, torch, case.mix, lens_d, bt_d, wo_d, batch, ck_[:cap], cv_[:cap])
    torch.cuda.synchronize()
    out["last_rows"] = {"overwritten": overwritten, "ok": matches(torch, lk, lv, wo, lens, ref)}
    out["head_major"] = matches(torch, hk.permute(1, 0, 2), hv.permute(1, 0, 2), wo, lens, ref)
    out["cap"] = {"rows": cap, "ok": matches(torch, ck_, cv_, wo, lens, ref, rows=cap)
                  and bool((ck_[cap:] == 3.0).all()) and bool((cv_[cap:] == -3.0).all())}
    out["ok"] = bool(out["flags"] and out["offsets"] and out["copy"] and out["mixed"] and out["unflagged_differs"]
                     and out["seed_key_differs"] and out["kv_py"]["ok"] and out["last_rows"]["ok"]
                     and out["head_major"] and out["cap"]["ok"])
    if case.name == "vllm":
        out["graph"] = graph_check(lib, torch, case, rng, bstride)
        out["ok"] &= out["graph"]["ok"]
    return out


def graph_check(lib, torch, case: Case, rng: random.Random, bstride: int) -> dict:
    """Offsets and window in one CUDA graph; a new step written in place (lengths, block tables) and half the plaintext
    slots encoded in place (`pg_encode`, flags cleared) before the replay."""
    G, D = lib.shape["kv_heads"], lib.shape["head_dim"]
    first = [rng.randrange(1, 300) for _ in range(9)]
    lens, bt = step(rng, first, bstride)
    batch = len(lens)
    lens_d = torch.tensor(lens, dtype=torch.int32, device="cuda")
    bt_d = torch.tensor(bt, dtype=torch.int32, device="cuda")
    wo_d = torch.zeros(batch + 1, dtype=torch.int32, device="cuda")
    cap = batch * 512
    gk = torch.full((cap, G, D), 3.0, dtype=torch.bfloat16, device="cuda")
    gv = torch.full_like(gk, -3.0)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        P.window_offsets(lib, torch, lens_d, batch, wo_d)
        case.window(lib, torch, case.mix, lens_d, bt_d, wo_d, batch, gk, gv)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        P.window_offsets(lib, torch, lens_d, batch, wo_d)
        case.window(lib, torch, case.mix, lens_d, bt_d, wo_d, batch, gk, gv)
    res = {"steps": []}
    ok = True
    for n in range(2):
        if n:
            lens, bt = step(rng, [rng.randrange(1, 513) for _ in range(9)], bstride)
            lens_d.copy_(torch.tensor(lens, dtype=torch.int32))
            bt_d.copy_(torch.tensor(bt, dtype=torch.int32))
            flagged = [s_ for s_ in range(case.nslots) if case.is_plain[s_]]
            now = sorted(rng.sample(flagged, len(flagged) // 2))
            sl = torch.tensor(now, dtype=torch.int64, device="cuda")
            tab = P.encode_table(torch, [(sl, len(now), case.mix, case.store_base, case.layers, case.layer, case.bj,
                                          case.flags)]).cuda()
            P.encode(lib, torch, tab, 1, len(now), case.lay, case.k)
            for s_ in now:
                case.is_plain[s_] = False
            torch.cuda.synchronize()
            res["encoded_in_place"] = len(now)
            res["flags_cleared"] = not bool(case.flags[sl].any())
            ok &= res["flags_cleared"]
        gk.fill_(3.0)
        gv.fill_(-3.0)
        graph.replay()
        torch.cuda.synchronize()
        wo = wo_d.cpu().tolist()
        good = wo == [sum(lens[:i]) for i in range(batch + 1)] and matches(torch, gk, gv, wo, lens,
                                                                            reference(torch, case, lens, bt_d))
        res["steps"].append({"rows": sum(lens), "ok": good})
        ok &= good
    res["ok"] = ok
    return res


def refusals(lib, torch, case: Case) -> dict:
    G, D = lib.shape["kv_heads"], lib.shape["head_dim"]
    lens = torch.tensor([5], dtype=torch.int32, device="cuda")
    bt = torch.zeros(1, 4, dtype=torch.int32, device="cuda")
    wo = torch.tensor([0, 5], dtype=torch.int32, device="cuda")
    buf = torch.zeros(8 + 1, G, D, dtype=torch.bfloat16, device="cuda")
    row = torch.zeros(1, G * D, dtype=torch.bfloat16, device="cuda")
    lay = case.lay
    st = P.stream(torch)

    def call(**kw) -> bool:
        a = dict(cache=case.mix.data_ptr(), encoded=1, order=case.order, k=case.k, bj=case.bj.data_ptr(),
                 plain=case.flags.data_ptr(), ck=0, cv=0, outk=buf.data_ptr(), outv=buf.data_ptr(), ots=G * D, ogs=D)
        a.update(kw)
        try:
            lib.call("pg_window", a["cache"], a["encoded"], a["order"], lens.data_ptr(), bt.data_ptr(), 4, lay.bs,
                     lay.sb, lay.sh, lay.ss, case.store_base, case.layers, case.layer, a["k"], a["bj"], a["plain"],
                     a["ck"], a["cv"], G * D, G * D, 0, wo.data_ptr(), 1, 8, a["outk"], a["outv"], a["ots"], a["ogs"],
                     0, st)
        except RuntimeError:
            return True
        return False

    torch.cuda.synchronize()
    res = {"valid_accepted": not call(),
           "misaligned_buffer": call(outk=buf.data_ptr() + 2),
           "row_stride": call(ots=G * D + 4),
           "order": call(order=2),
           "k_zero": call(k=0),
           "flags_on_copy": call(encoded=0, bj=0),
           "offsets_on_copy": call(encoded=0, plain=0),
           "ck_without_cv": call(ck=row.data_ptr())}
    torch.cuda.synchronize()
    res["ok"] = all(res.values())
    return res


def timing(lib, torch, a, log) -> dict:
    G, D, WB, NL = lib.shape["kv_heads"], lib.shape["head_dim"], lib.shape["wb"], lib.field.limbs
    pages, L, k = a.pages, a.layers, a.k
    nslots = pages * BS
    KB.set_key(lib, KB.lib_key(lib, k))
    caches = [torch.randint(-2 ** 15, 2 ** 15 - 1, (pages, G, BS, 2 * D), dtype=torch.int16, device="cuda")
              .view(torch.bfloat16) for _ in range(L)]
    lay = P.layout(caches[0], G, WB)
    bjs = []
    for _ in range(L):
        b = torch.randint(-2 ** 31, 2 ** 31 - 1, (nslots, NL), dtype=torch.int32, device="cuda")
        b[:, -1] = 0
        bjs.append(b)
    none = torch.zeros(nslots, dtype=torch.uint8, device="cuda")
    half = (torch.rand(nslots, device="cuda") < 0.5).to(torch.uint8)
    kvs = [c.transpose(1, 2).split(D, dim=-1) for c in caches]
    res = {"pages": pages, "layers": L, "k": k, "pool_gb": L * nslots * 4096 / 1e9,
           "ctas": {"copy": lib.so.pg_window_ctas(0), "decode": lib.so.pg_window_ctas(1)}, "points": []}
    log(f"timing: {pages} pages x {L} layers ({res['pool_gb']:.1f} GB), k={k}, grid {res['ctas']}")
    rng = random.Random(5)
    for spec in a.points.split(","):
        B, T = (int(v) for v in spec.split("x"))
        npr = -(-T // BS)
        if B * npr > pages:
            log(f"{spec}: {B * npr} pages past the pool's {pages}, skipped")
            continue
        perm = rng.sample(range(pages), B * npr)
        bt = torch.tensor([perm[r * npr:(r + 1) * npr] for r in range(B)], dtype=torch.int32, device="cuda")
        lens = torch.full((B,), T, dtype=torch.int32, device="cuda")
        wo = torch.zeros(B + 1, dtype=torch.int32, device="cuda")
        rows = B * T
        outk = torch.empty(rows, G, D, dtype=torch.bfloat16, device="cuda")
        outv = torch.empty_like(outk)
        pos = torch.arange(T, device="cuda")
        blk = bt[:, pos // BS].reshape(-1).long()
        off = (pos % BS).repeat(B)
        flat = torch.empty(rows * 2 * G * D, dtype=torch.int16, device="cuda")
        src = [c.view(torch.int16).reshape(-1)[:rows * 2 * G * D] for c in caches]

        def windows(encoded, flags=None):
            P.window_offsets(lib, torch, lens, B, wo)
            for l in range(L):
                P.window(lib, torch, caches[l], lay, 0, L, l, encoded, k, lens, bt, wo, B, outk, outv,
                         bj=bjs[l] if encoded else None, plain=flags if encoded else None)

        def index():
            for l in range(L):
                kc, vc = kvs[l]
                _ = kc[blk, off], vc[blk, off]

        def contiguous():
            for l in range(L):
                flat.copy_(src[l])

        arms = {"decode": lambda: windows(1, none), "decode_half_plaintext": lambda: windows(1, half),
                "copy": lambda: windows(0), "vllm_index": index, "contiguous_copy": contiguous}
        pt = {"batch": B, "context": T, "rows": rows, "gb_read": L * rows * 4096 / 1e9}

        def put(name, ms):
            pt[name] = {"step_ms": round(ms, 3), "layer_us": round(ms * 1e3 / L, 1),
                        "gbps_read": round(pt["gb_read"] / ms * 1e3, 1)}

        for name, fn in arms.items():
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                fn()
            torch.cuda.current_stream().wait_stream(s)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn()
            put(name, statistics.median(KB.timed(torch, graph.replay, a.reps, warm=2)) * 1e3)
            del graph
        put("decode_eager", statistics.median(KB.timed(torch, arms["decode"], a.reps, warm=2)) * 1e3)
        pt["decode_over_copy"] = round(pt["decode"]["step_ms"] / pt["copy"]["step_ms"], 3)
        res["points"].append(pt)
        log(f"{spec}: {rows} rows, {pt['gb_read']:.2f} GB a step: " + ", ".join(
            f"{n} {pt[n]['step_ms']:.2f} ms ({pt[n]['gbps_read']:.0f} GB/s)"
            for n in ("decode", "decode_half_plaintext", "copy", "vllm_index", "contiguous_copy", "decode_eager")))
        del outk, outv, flat
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", required=True, help="NAME=PATH, paged.cu built at field NAME")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ks", default="2,3")
    ap.add_argument("--pages", type=int, default=23040)
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--points", default="1x4096,8x4096,32x1024,32x4096,64x5760")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--no-timing", action="store_true")
    a = ap.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    logf = (out_dir / "window-gate.log").open("a")

    def log(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S', time.gmtime())}Z {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    import torch

    lib = P.PagedLib(*a.lib.split("=", 1))
    log(f"{lib.name}: shape {lib.shape}, mask {lib.mask}, fold6 {lib.fold6}")
    torch.manual_seed(0)
    rng = random.Random(3)
    res = {"lib": lib.name, "shape": lib.shape, "mask": lib.mask, "ks": {}, "ok": True}
    for k in (int(v) for v in a.ks.split(",")):
        key = KB.lib_key(lib, k)
        KB.set_key(lib, key)
        cells = {}
        for name in ("vllm", "planned"):
            case = Case(lib, torch, name, k, key, rng)
            cells[name] = gate_case(lib, torch, case, rng, log)
            if name == "vllm":
                cells["refusals"] = refusals(lib, torch, case)
            c = cells[name]
            log(f"k={k} {name}: {'ok' if c['ok'] else 'FAIL'} ({c['rows']} rows, slots read {c['slots_read']}, copy "
                f"{c['copy']}, mixed {c['mixed']}, unflagged differs {c['unflagged_differs']}, seed key differs "
                f"{c['seed_key_differs']}, kv.py {c['kv_py']}, last rows {c['last_rows']}, head-major {c['head_major']}, "
                f"cap {c['cap']}" + (f", graph {c['graph']}" if "graph" in c else "") + ")")
        log(f"k={k} refusals: {cells['refusals']}")
        cells["ok"] = all(cells[n]["ok"] for n in ("vllm", "planned", "refusals"))
        res["ks"][k] = cells
        res["ok"] &= cells["ok"]
    res["gate_ok"] = res["ok"]
    if not a.no_timing:
        res["timing"] = timing(lib, torch, a, log)
    (out_dir / "window-gate.json").write_text(json.dumps(res, indent=1))
    log("window gate " + ("passed" if res["ok"] else "FAILED"))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
