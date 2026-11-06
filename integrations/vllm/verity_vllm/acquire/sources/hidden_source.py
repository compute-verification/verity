"""Hidden rows R5–R8 (FA2 fp32 scores, softmax weights, bf16 P fragments, running max / rescale / lane sums / accumulators / epilogue)
as an acquisition source of `native_host`: the **M1 export tap** (the tap-enabled FA2 extension `verity_fa2_matAll.so`, pinned FA2 tree 506341a1 + `verity_tap_v7.patch`, `VERITY_TAP=3`, `VERITY_MAT_SRC=63`).

Mechanism: vLLM's FA2 backend calls `torch.ops._vllm_fa2_C.varlen_fwd(...)`; we replace that attribute with a wrapper that (i) derives the
kernel geometry (HB = b·h slabs, M query rows, NB key-block cap) exactly as `hidden_tap.kernel_geometry`, (ii) points the tap at a
preallocated per-geometry i32 stream buffer (padded to whole 64-word chunks) with the per-launch tag, (iii) calls the TAPPED kernel
(`torch.ops.verity_fa2_matAll.varlen_fwd`, same signature; out/lse bit-identical to the installed kernel) and (iv) hands the stream buffer to the committer's arena acquisition on the compute stream (a D2D snapshot
ordered after the kernel, before the buffer is reused by the next layer).  The stream is then part of the SAME packed step stream / step
root / run root as the native rows, in `chunk`-byte position-identified leaves (identity = (step, byte range) → tensor `fa2.m1/L<k>`
with its geometry in the layout digest).

Stream layout (`StreamLayout`): S, P [HB, M, N] f32; RP [HB, M, N/2] u32 (bf16 pairs); ROW [HB, M, NB, 8] f32; O [HB, M, NB, D]
f32; FIN [HB, M, 4], with N = NB * kBlockN.  Rows ≥ seqlen_q / blocks ≥ n_block_max are never written: the buffer is reused
deterministically (same launch sequence ⇒ same bytes); unvisited words carry the previous launch's content of
the same geometry — bound but outside the layout (the verifier's queries never address them).

Tap layout v8 (C-FA2-HDIM): the layout is arithmetic in the kernel tile (kBlockN, head_dim) -- 128/64 for SmolLM2/Llama,
64/96 for Phi-3, 64/128 for Qwen2.5/Mistral/OLMoE (num_splits == 1 "align" kernel, `fa2_kblock_n`).  The wrapper declares the tile per
launch (`verity_set_tap_v8`); the kernel arms only if its compile-time tile equals the declaration, otherwise it counts the launch in
`verity_tap_skipped` and writes nothing -- which the wrapper turns into a collector error (X-03: expected != committed -> COMMIT FAIL).
"""
from __future__ import annotations

import time
from typing import Any, Mapping

from verity_vllm.engine import hooks

KBLOCK_N = 128          # hd 64 (legacy constant; use fa2_kblock_n)
KBLOCK_M = 64           # every split-KV kernel (flash_fwd_launch_template.h: kBlockM = 64 fixed for all head dims)
CHUNK_WORDS = 64
LEGACY_TILE = (128, 64)   # (kBlockN, head_dim) of the v7 hd-64 tap: v8 keeps its stream layout, names and roots


def fa2_kblock_n(head_dim: int, num_splits: int) -> int:
    """kBlockN of the pinned FA2 (506341a1) split-KV kernel vLLM launches for this head dim (`run_mha_fwd_splitkv_dispatch`,
    flash_fwd_launch_template.h:172-191).  Batch-invariant serving passes num_splits == 1 -> the "align" kernel with
    kBlockN_standard = 128 for hd <= 64 and 64 otherwise (hd 96/128/256).  Any other num_splits selects the split/combine kernels
    (kBlockN 256/128/64 + a combine pass) which the M1 tap does not cover: refused here (the kernel-side arm check would skip the
    launch anyway; the tap layout is arithmetic in (kBlockN, head_dim) and the kernel only arms when its tile equals the declared one)."""
    if num_splits != 1:
        raise RuntimeError(f"fa2 tap: num_splits={num_splits} (only the num_splits==1 batch-invariant kernel is tapped)")
    if head_dim not in (64, 96, 128, 256):
        raise RuntimeError(f"fa2 tap: head_dim {head_dim} has no pinned bf16 split-KV kernel")
    return 128 if head_dim <= 64 else 64


def fa2_swapped(max_seqlen_q: int, H: int, Hk: int, head_dim: int, window_left: int, window_right: int, dropout_p: float = 0.0,
                alibi_slopes=None) -> bool:
    """`seqlenq_ngroups_swapped` of the pinned FA2 varlen forward (flash_api.cpp:646): a decode launch (max_seqlen_q == 1)
    of a GQA model folds the H/Hk query heads of each KV group into the row dimension (rows = groups, heads = Hk) ONLY when no sliding
    window is set (window_left < 0 and window_right < 0), dropout is 0, head_dim % 8 == 0 and there are no alibi slopes.  vLLM passes
    window_size (sliding_window - 1, 0) on the sliding layers of Gemma-2 (every other layer) / Phi-3, so their decode launches run the
    UN-swapped kernel (b*H slabs of 1 row); a mirror of `max_seqlen_q == 1 and H > Hk` alone would declare the swapped geometry there and
    the kernel would refuse the launch (arm check b*h <= vt.H fails: verity_tap_skipped++ -> 'skipped by the kernel' -> X-03 COMMIT FAIL, never
    a wrong binding).  This mirrors the kernel's own rule so the declared (HB, M) is the launched one."""
    return (int(max_seqlen_q) == 1 and H > Hk and int(window_left) < 0 and int(window_right) < 0 and float(dropout_p) == 0.0
            and head_dim % 8 == 0 and alibi_slopes is None)


def mat_total_words(HB: int, M: int, NB: int, BN: int = 128, D: int = 64, ms: bool = False) -> int:
    """Words of the M1 stream (mirror of `Verity_mat_layout(HB, M, NB, BN, D)` / `hidden_stream.StreamLayout`); `ms`: a build that also
    writes the MS class (`verity_tap_src() & 64`), one word per (slab, row, key block) after FIN."""
    N = NB * BN
    return HB * M * N * 2 + HB * M * (N // 2) + HB * M * NB * 8 + HB * M * NB * D + HB * M * 4 + (HB * M * NB if ms else 0)


def fa2_tap_words_per_seq(h: int, M: int, NB: int, BN: int, D: int, ms: bool = False) -> int:
    """Words of the padded M1 tap buffer ONE sequence occupies in a multi-sequence launch: its h slabs x the launch's
    row pitch M (= max_seqlen_q) x the six plane widths -- `mat_total_words(HB, M, NB, BN, D) / HB`.  Independent of the sequence's own
    length (the buffer is pitched to the batch maximum); the STAGED bytes are the compacted rows < seqlen_q only (native_collect.cpp p0b)."""
    return mat_total_words(1, M, NB, BN, D, ms) * h


def fa2_tap_buffer_words(b: int, h: int, M: int, NB: int, BN: int, D: int, ms: bool = False) -> int:
    """Padded (whole 64-word chunks) tap buffer of a launch of `b` sequences (B64 llama 1024: 6,585,057,280 words =
    26,340,229,120 B = 24.53 GiB, an allocation that ran out of device memory; `plan_fa2_tap_groups` bounds it)."""
    return -(-(b * fa2_tap_words_per_seq(h, M, NB, BN, D, ms)) // CHUNK_WORDS) * CHUNK_WORDS


def plan_fa2_tap_groups(b: int, words_per_seq: int, cap_words: int) -> list[tuple[int, int]]:
    """Consecutive sequence groups [g0, g1) of a `b`-sequence launch such that every group's padded tap buffer
    (g1 - g0) * words_per_seq (rounded to whole chunks) fits `cap_words`; order preserved, every sequence in exactly one group.  A single
    sequence larger than the cap forms its own group (the driver records `cap_exceeded` -- the per-sequence cost h*M*W is the residual bound
    the split cannot lower; it is quadratic in the sequence length, not in the batch).  cap_words <= 0 -> one group (bounded path off)."""
    if b <= 0:
        return []
    if cap_words <= 0 or words_per_seq <= 0:
        return [(0, b)]
    per = max(1, cap_words // words_per_seq)
    while per > 1 and -(-(per * words_per_seq) // CHUNK_WORDS) * CHUNK_WORDS > cap_words:
        per -= 1
    return [(g0, min(b, g0 + per)) for g0 in range(0, b, per)]


def tap_row3(ns: Any) -> int:
    """What word 3 of the build's ROW entries holds (`verity_tap_row3`): 0 = never written (builds before the op included), 1 = the guarded
    max (`TAP_ROW3`)."""
    return int(ns.verity_tap_row3()) if hasattr(ns, "verity_tap_row3") else 0


#: `verity_tap_row3()` of the tap build a manifest built under the guarded-max policy needs (ROW word 3 the guarded max of each key
#: block after a row's first, and the MS class after FIN); no manifest is built under it since the M1 stream left the manifest
TAP_ROW3 = 1


def guarded_max_policy(manifest: Mapping[str, Any] | None) -> bool:
    """Whether a manifest was built under the guarded-max policy (its query header has a `guarded_max` record)."""
    q = (manifest or {}).get("query")
    return isinstance(q, dict) and isinstance(q.get("guarded_max"), dict)


def tap_ms(src: int) -> bool:
    """Whether a build with source mask `src` (`verity_tap_src`) writes the MS class after FIN (`hidden_stream.SRC_BITS["MS"]`; the
    guarded-max builds)."""
    return bool(int(src) & 64)


def stream_name(k: int, HB: int, M: int, NB: int, BN: int, D: int) -> str:
    """Identity of launch k's stream in the step layout (the geometry is part of the layout digest).  The hd-64 tile
    (`LEGACY_TILE`) keeps its stream's original name so hd-64 roots are unchanged; other tiles carry (kBlockN, head_dim) explicitly."""
    base = f"fa2.m1/L{k}/g{HB}x{M}x{NB}"
    return base if (BN, D) == LEGACY_TILE else f"{base}x{BN}x{D}"


class HiddenM1Source:
    INSTALLED_NS = "_vllm_fa2_C"
    _tap: hooks.Patch | None = None

    def __init__(self, committer, so_path: str, ns: str = "verity_fa2_matAll") -> None:
        import torch

        self.torch = torch
        self.c = committer
        self.so_path, self.ns_name = so_path, ns
        torch.ops.load_library(so_path)
        self.ns = getattr(torch.ops, ns)
        self.mode = int(self.ns.verity_tap_mode())
        self.src = int(self.ns.verity_tap_src())
        if self.mode != 3:
            raise RuntimeError(f"{ns}: compiled tap mode {self.mode} != 3 (M1)")
        self.tapped_op = self.ns.varlen_fwd
        # the head-dim-portable v8 build exposes verity_set_tap_v8 (declares (kBlockN, head_dim)) and verity_tap_hdims (compiled hd mask);
        # the v7 build has neither and covers hd 64 only.
        self.v8 = hasattr(self.ns, "verity_set_tap_v8")
        self.hdims = int(self.ns.verity_tap_hdims()) if self.v8 else (1 << 2)
        self.installed_ns = getattr(torch.ops, self.INSTALLED_NS)
        self.orig_op = self.installed_ns.varlen_fwd
        self.bufs: dict[tuple[int, int, int, int, int], Any] = {}
        self.launch = 0
        self.launches_in_step = 0
        self.stats = {"launches": 0, "bytes": 0, "geometries": {}, "set_tap_s": 0.0, "skipped": None, "so_sha256": None,
                      "tap_layout": 8 if self.v8 else 7, "hdims": [hd for hd in (64, 96, 128, 256) if self.hdims & (1 << (hd // 32))],
                      "row3": tap_row3(self.ns), "ms": tap_ms(self.src)}
        self.ms = self.stats["ms"]
        self.active = False
        self.time_launch = None

    def require_row3(self, guarded_max: bool) -> None:
        """Refuse a build whose ROW word 3 or MS class is not the manifest's: a manifest built under the guarded-max policy
        (`guarded_max_policy`) needs a build that writes the guarded max and the MS class, any other manifest one that writes neither
        (the stream bytes of record)."""
        want = TAP_ROW3 if guarded_max else 0
        policy = "is under the guarded-max policy" if guarded_max else "has no guarded-max policy"
        if self.stats["row3"] != want:
            raise RuntimeError(f"hidden tap {self.ns_name}: ROW word 3 mode {self.stats['row3']} (verity_tap_row3), the manifest "
                               f"{policy} and needs mode {want}")
        if bool(self.stats.get("ms")) != bool(guarded_max):
            raise RuntimeError(f"hidden tap {self.ns_name}: source mask {self.src} {'writes' if self.stats.get('ms') else 'lacks'} the MS "
                               f"class (verity_tap_src & 64), the manifest {policy}")

    def extra_step_bytes(self, envelope: dict[str, Any], model: Any) -> dict[str, int]:
        """The MS class on top of the arena bound's six classes (`native_host.step_bound_bytes`): a build that writes it adds, per token
        row of a launch, one word per head and key block (at most max_model_len / 64) of every attention layer, and one tensor per
        sequence and layer on the compacted path."""
        if not self.ms:
            return {"per_token_bytes": 0, "per_seq_bytes": 0, "tensors_per_step": 0}
        cfg = getattr(model, "config", None)
        heads, layers = int(getattr(cfg, "num_attention_heads", 0) or 0), int(getattr(cfg, "num_hidden_layers", 0) or 0)
        T = int(envelope.get("max_model_len") or int(envelope.get("max_prompt") or 1) + int(envelope.get("max_new") or 1))
        return {"per_token_bytes": 4 * layers * heads * -(-T // 64), "per_seq_bytes": 0,
                "tensors_per_step": layers * max(1, int(envelope.get("batch") or 1))}

    def tile(self, head_dim: int, num_splits: int) -> tuple[int, int]:
        """(kBlockN, head_dim) the tap must be armed with for this launch; refuses head dims this build has no kernel for."""
        if not self.hdims & (1 << (head_dim // 32)):
            raise RuntimeError(f"fa2 tap {self.ns_name}: head_dim {head_dim} not compiled (hdims {self.stats['hdims']}); rebuild with pod_fa2_tap.sh")
        BN = fa2_kblock_n(head_dim, num_splits)
        if not self.v8 and (BN, head_dim) != LEGACY_TILE:
            raise RuntimeError(f"fa2 tap {self.ns_name}: the v7 build declares the hd-64 layout only; tile ({BN}, {head_dim}) needs the v8 build")
        return BN, head_dim

    def arm(self, buf, HB: int, M: int, NB: int, BN: int, D: int, tag: int) -> None:
        if self.v8:
            self.ns.verity_set_tap_v8(3, buf, HB, M, NB, BN, D, tag)
        else:
            self.ns.verity_set_tap(3, None, buf, HB, M, 0, NB)
            self.ns.verity_set_tap_tag(tag)

    def install(self) -> None:
        self._tap = hooks.patch(self.installed_ns, "varlen_fwd", self._wrapper())
        self.active = True

    def uninstall(self) -> None:
        if self._tap is not None:
            self._tap.uninstall()
            self._tap = None
        try:
            self.stats["skipped"] = int(self.ns.verity_tap_skipped())
        except Exception as e:  # noqa: BLE001
            self.c.errors.append(f"fa2 tap skipped count: {e!r}"[:200])
        self.ns.verity_set_tap(0, None, None, 0, 0, 0, 0)
        self.active = False

    def _buffer(self, HB: int, M: int, NB: int, BN: int = 128, D: int = 64):
        key = (HB, M, NB, BN, D)
        b = self.bufs.get(key)
        if b is None:
            words = -(-mat_total_words(HB, M, NB, BN, D, self.ms) // CHUNK_WORDS) * CHUNK_WORDS
            b = self.torch.zeros((words,), dtype=self.torch.int32, device="cuda")
            self.bufs[key] = b
            self.stats["geometries"][str(key if (BN, D) != LEGACY_TILE else (HB, M, NB))] = words * 4
        return b

    def _wrapper(self):
        me = self
        torch = self.torch

        def varlen_fwd(q, k, v, out, cu_seqlens_q, cu_seqlens_k, seqused_k, leftpad_k, block_table, alibi_slopes, max_seqlen_q, max_seqlen_k,
                       dropout_p, softmax_scale, zero_tensors, causal, window_left, window_right, softcap, return_softmax, num_splits, gen):
            job = me.c._cur
            if job is None:                                   # not inside a committed step (warm-up, dummy run): the installed kernel
                return me.orig_op(q, k, v, out, cu_seqlens_q, cu_seqlens_k, seqused_k, leftpad_k, block_table, alibi_slopes, max_seqlen_q,
                                  max_seqlen_k, dropout_p, softmax_scale, zero_tensors, causal, window_left, window_right, softcap,
                                  return_softmax, num_splits, gen)
            t0 = time.perf_counter()
            b = int(cu_seqlens_q.numel()) - 1
            H, Hk, D = int(q.shape[1]), int(k.shape[-2]), int(q.shape[-1])
            swapped = fa2_swapped(max_seqlen_q, H, Hk, D, window_left, window_right, dropout_p, alibi_slopes)
            h = Hk if swapped else H
            M = (H // Hk) if swapped else int(max_seqlen_q)
            HB = b * h
            try:
                BN, D = me.tile(D, int(num_splits))
            except RuntimeError as e:                          # uncovered tile: the launch is expected (manifest) but never bound -> X-03 fails the run
                me.c.x03_expected = getattr(me.c, "x03_expected", 0) + 1
                me.c.errors.append(f"fa2 launch {me.launch}: {e}"[:300])
                return me.orig_op(q, k, v, out, cu_seqlens_q, cu_seqlens_k, seqused_k, leftpad_k, block_table, alibi_slopes, max_seqlen_q,
                                  max_seqlen_k, dropout_p, softmax_scale, zero_tensors, causal, window_left, window_right, softcap,
                                  return_softmax, num_splits, gen)
            NB = (int(max_seqlen_k) + BN - 1) // BN
            buf = me._buffer(HB, M, NB, BN, D)
            skipped0 = int(me.ns.verity_tap_skipped())
            me.arm(buf, HB, M, NB, BN, D, me.launch)
            t1 = time.perf_counter()
            res = me.tapped_op(q, k, v, out, cu_seqlens_q, cu_seqlens_k, seqused_k, leftpad_k, block_table, alibi_slopes, max_seqlen_q,
                               max_seqlen_k, dropout_p, softmax_scale, zero_tensors, causal, window_left, window_right, softcap,
                               return_softmax, num_splits, gen)
            if int(me.ns.verity_tap_skipped()) != skipped0:   # the kernel refused the declared tile/caps: nothing was written -> fail closed
                me.c.x03_expected = getattr(me.c, "x03_expected", 0) + 1
                me.c.errors.append(f"fa2 launch {me.launch}: skipped by the kernel (declared tile BN={BN} D={D}, HB={HB} M={M} NB={NB}, "
                                   f"num_splits={int(num_splits)}); stream not bound"[:300])
                me.launch += 1; me.launches_in_step += 1
                return res
            # acquire the stream (D2D snapshot into the arena on the compute stream, ordered after the kernel, before the next launch reuses buf)
            me.c._acquire(stream_name(me.launches_in_step, HB, M, NB, BN, D), buf, job)
            me.launch += 1
            me.launches_in_step += 1
            me.stats["launches"] += 1
            me.stats["bytes"] += buf.numel() * 4
            me.stats["set_tap_s"] += t1 - t0
            me.c.spans.add("acquisition", time.perf_counter() - t0, step=job.step, nbytes=buf.numel() * 4)
            return res

        return varlen_fwd

    def on_step_begin(self) -> None:
        self.launches_in_step = 0

    def declaration_classes(self) -> list[dict]:
        return [{"class": "fa2_hidden_m1_stream", "how": "bound-as-leaf", "covers_rows": ["R5", "R6", "R7", "R8"],
                 "rule": None, "count": None, "bytes": self.stats["bytes"], "tensors": self.stats["launches"],
                 "tap_layout": self.stats["tap_layout"], "hdims": self.stats["hdims"], **({"row3": "guarded max"} if self.stats["row3"] else {}),
                 **({"ms": "max * scale per key block"} if self.ms else {}),
                 "evidence": (f"FA2 M1 export tap {self.ns_name} (mode {self.mode}, src {self.src}, layout v{self.stats['tap_layout']}, hdims "
                              f"{self.stats['hdims']}) in place of _vllm_fa2_C.varlen_fwd; stream buffer per launch acquired into the packed step "
                              f"stream (same run root); geometries (HB, M, NB[, kBlockN, head_dim]) {self.stats['geometries']}; tap_skipped "
                              f"{self.stats['skipped']}; out/lse "
                              f"bit-identical to the installed kernel: verity_vllm.properties.fa_tap_exactness")}]


# ----------------------------------------------------------------------------------------------------------------------------------
# the same M1 export-and-commit source for FlashAttention-3 (sm90, `_vllm_fa3_C.fwd`), tap layout v9 = the FA2 v8 word
# layout over PackGQA-packed rows (`verity_vllm/acquire/fa3_tap_src/verity_tap_fa3.h`; exactness: verity_vllm.properties.fa_tap_exactness).
# ----------------------------------------------------------------------------------------------------------------------------------
def fa3_tile(head_dim: int, max_seqlen_q: int, qheads_per_kvhead: int, num_splits: int) -> tuple[int, int]:
    """(kBlockM, kBlockN) of the pinned FA3 (506341a1 hopper/) forward vLLM launches: `tile_size_fwd_sm90` (bf16, causal, paged KV without TMA)
    under `use_one_mma_wg` = max_seqlen_q * qheads_per_kvhead <= 64 (heuristics.h; varlen -> params.seqlen_q is max_seqlen_q): hdim 64 ->
    64 x 192 (one warpgroup) / 192 x 128; hdim 128 -> 64 x 128 / 128 x 128.  num_splits must be 1 (the kernel disarms the tap on the
    split/combine path; batch-invariant serving passes 1).  The same rule as `verity_vllm.program.frontend.targets.fa3_kblock_n`."""
    if num_splits != 1:
        raise RuntimeError(f"fa3 tap: num_splits={num_splits} (only the num_splits==1 batch-invariant kernel is tapped)")
    one_wg = max_seqlen_q * qheads_per_kvhead <= 64
    if head_dim == 64:
        return (64, 192) if one_wg else (192, 128)
    if head_dim == 128:
        return (64, 128) if one_wg else (128, 128)
    raise RuntimeError(f"fa3 tap: head_dim {head_dim} has no tapped bf16 paged PackGQA kernel (64, 128)")


def fa3_stream_name(k: int, HB: int, M: int, NB: int, BN: int, D: int) -> str:
    """Identity of FA3 launch k's stream in the step layout: slabs HB = b * h_k, M = PACKED rows (max_seqlen_q * qheads_per_kvhead), NB key
    blocks, and the tile (kBlockN, head_dim) the layout is arithmetic in -- always explicit (no legacy name to preserve)."""
    return f"fa3.m1/L{k}/g{HB}x{M}x{NB}x{BN}x{D}"


class HiddenM1SourceFA3(HiddenM1Source):
    """`HiddenM1Source` for the FA3 backend: replaces `torch.ops._vllm_fa3_C.fwd` with a wrapper that declares the launch's geometry to the
    v9 tap (`verity_set_tap_v9(mode, buf, HB, M, NB, BN, D, tag)`), calls the tapped `fwd` (`verity_fa3_matReq.so`) and acquires the stream
    buffer into the arena.  Geometry: slab = (sequence, KV head) -- FA3's paged varlen path is always PackGQA, the kernel's bidh is the KV
    head and the tile rows are (query row, head-in-group) pairs -- so HB = b * h_k and M = max_seqlen_q * (h / h_k); NB = ceil(max_seqlen_k
    / kBlockN).  The kernel arms only when the selected tile is exactly the declared (kBlockN, head_dim), PackGQA, num_splits 1, no
    local/softcap/append-KV/Qv/FP8 and the caps cover the launch; otherwise `verity_tap_skipped` increments and nothing is written, which
    the wrapper turns into a collector error (X-03 fails the run closed)."""
    INSTALLED_NS = "_vllm_fa3_C"
    # positional layout of `_vllm_fa3_C.fwd` (vllm_flash_attn/flash_attn_interface.py, fa_version 3)
    _Q, _K, _V, _CU_Q, _SEQUSED_K, _MAXQ, _MAXK, _PAGE_TABLE, _NUM_SPLITS = 0, 1, 2, 7, 11, 12, 13, 14, 30

    def __init__(self, committer, so_path: str, ns: str = "verity_fa3_matReq") -> None:
        import torch

        self.torch = torch
        self.c = committer
        self.so_path, self.ns_name = so_path, ns
        torch.ops.load_library(so_path)
        self.ns = getattr(torch.ops, ns)
        if not hasattr(self.ns, "verity_set_tap_v9"):
            raise RuntimeError(f"{ns}: not an FA3 (v9) tap build")
        self.mode = int(self.ns.verity_tap_mode())
        self.src = int(self.ns.verity_tap_src())
        if self.mode != 3:
            raise RuntimeError(f"{ns}: compiled tap mode {self.mode} != 3 (M1)")
        self.tapped_op = self.ns.fwd
        self.v8 = True
        self.hdims = int(self.ns.verity_tap_hdims())
        self.installed_ns = getattr(torch.ops, self.INSTALLED_NS)
        self.orig_op = self.installed_ns.fwd
        self.bufs = {}
        self.launch = 0
        self.launches_in_step = 0
        self.stats = {"launches": 0, "bytes": 0, "geometries": {}, "set_tap_s": 0.0, "skipped": None, "so_sha256": None,
                      "tap_layout": int(self.ns.verity_tap_layout_version()), "hdims": [hd for hd in (64, 128) if self.hdims & (1 << (hd // 32))],
                      "backend": "fa3", "tiles": {}, "row3": tap_row3(self.ns), "ms": tap_ms(self.src)}
        self.ms = self.stats["ms"]
        self.active = False
        self.time_launch = None

    def arm(self, buf, HB: int, M: int, NB: int, BN: int, D: int, tag: int) -> None:
        self.ns.verity_set_tap_v9(3, buf, HB, M, NB, BN, D, tag)

    def install(self) -> None:
        self._tap = hooks.patch(self.installed_ns, "fwd", self._wrapper())
        self.active = True

    def uninstall(self) -> None:
        if self._tap is not None:
            self._tap.uninstall()
            self._tap = None
        try:
            self.stats["skipped"] = int(self.ns.verity_tap_skipped())
        except Exception as e:  # noqa: BLE001
            self.c.errors.append(f"fa3 tap skipped count: {e!r}"[:200])
        self.ns.verity_set_tap_off()
        self.active = False

    def _buffer(self, HB: int, M: int, NB: int, BN: int = 128, D: int = 64):
        key = (HB, M, NB, BN, D)
        b = self.bufs.get(key)
        if b is None:
            words = -(-mat_total_words(HB, M, NB, BN, D, self.ms) // CHUNK_WORDS) * CHUNK_WORDS
            b = self.torch.zeros((words,), dtype=self.torch.int32, device="cuda")
            self.bufs[key] = b
            self.stats["geometries"][str(key)] = words * 4
        return b

    def _wrapper(self):
        me = self

        def fwd(*args):
            job = me.c._cur
            if job is None:                                   # not inside a committed step (warm-up, dummy run): the installed kernel
                return me.orig_op(*args)
            t0 = time.perf_counter()
            q, k, cu_q, max_q, max_k = args[me._Q], args[me._K], args[me._CU_Q], args[me._MAXQ], args[me._MAXK]
            num_splits = int(args[me._NUM_SPLITS]) if len(args) > me._NUM_SPLITS and args[me._NUM_SPLITS] is not None else 1
            if cu_q is None or args[me._PAGE_TABLE] is None or max_q is None or max_k is None:
                me.c.x03_expected = getattr(me.c, "x03_expected", 0) + 1
                me.c.errors.append(f"fa3 launch {me.launch}: not the varlen paged-KV launch the tap covers (cu_seqlens_q/page_table/max_seqlen missing)"[:300])
                return me.orig_op(*args)
            b = int(cu_q.numel()) - 1
            H, Hk, D = int(q.shape[1]), int(k.shape[-2]), int(q.shape[-1])
            g = H // Hk
            try:
                if not me.hdims & (1 << (D // 32)):
                    raise RuntimeError(f"fa3 tap {me.ns_name}: head_dim {D} not compiled (hdims {me.stats['hdims']}); rebuild with pod_fa3_tap.sh")
                kBM, BN = fa3_tile(D, int(max_q), g, num_splits)
            except RuntimeError as e:                          # uncovered launch: expected (manifest) but never bound -> X-03 fails the run
                me.c.x03_expected = getattr(me.c, "x03_expected", 0) + 1
                me.c.errors.append(f"fa3 launch {me.launch}: {e}"[:300])
                return me.orig_op(*args)
            HB, M = b * Hk, int(max_q) * g
            NB = (int(max_k) + BN - 1) // BN
            buf = me._buffer(HB, M, NB, BN, D)
            skipped0 = int(me.ns.verity_tap_skipped())
            me.arm(buf, HB, M, NB, BN, D, me.launch)
            t1 = time.perf_counter()
            res = me.tapped_op(*args)
            if int(me.ns.verity_tap_skipped()) != skipped0:   # the kernel refused the declared tile/caps: nothing was written -> fail closed
                me.c.x03_expected = getattr(me.c, "x03_expected", 0) + 1
                me.c.errors.append(f"fa3 launch {me.launch}: skipped by the kernel (declared tile kBlockM={kBM} BN={BN} D={D}, HB={HB} M={M} NB={NB}, "
                                   f"num_splits={num_splits}); stream not bound"[:300])
                me.launch += 1; me.launches_in_step += 1
                return res
            me.c._acquire(fa3_stream_name(me.launches_in_step, HB, M, NB, BN, D), buf, job)
            me.launch += 1
            me.launches_in_step += 1
            me.stats["launches"] += 1
            me.stats["bytes"] += buf.numel() * 4
            me.stats["set_tap_s"] += t1 - t0
            tile = f"{kBM}x{BN}"
            me.stats["tiles"][tile] = me.stats["tiles"].get(tile, 0) + 1
            me.c.spans.add("acquisition", time.perf_counter() - t0, step=job.step, nbytes=buf.numel() * 4)
            return res

        return fwd

    def declaration_classes(self) -> list[dict]:
        return [{"class": "fa3_hidden_m1_stream", "how": "bound-as-leaf", "covers_rows": ["R5", "R6", "R7", "R8"],
                 "rule": None, "count": None, "bytes": self.stats["bytes"], "tensors": self.stats["launches"],
                 "tap_layout": self.stats["tap_layout"], "hdims": self.stats["hdims"], "tiles": dict(self.stats["tiles"]),
                 **({"row3": "guarded max"} if self.stats["row3"] else {}), **({"ms": "max * scale per key block"} if self.ms else {}),
                 "evidence": (f"FA3 M1 export tap {self.ns_name} (mode {self.mode}, src {self.src}, layout v{self.stats['tap_layout']}, hdims "
                              f"{self.stats['hdims']}) in place of _vllm_fa3_C.fwd; stream buffer per launch acquired into the packed step stream "
                              f"(same run root); geometries (HB=b*h_k, M=packed rows, NB, kBlockN, head_dim) {self.stats['geometries']}; tiles {self.stats['tiles']}; "
                              f"tap_skipped {self.stats['skipped']}; out/lse bit-identical to the installed _vllm_fa3_C.fwd (hd 64/128, tiles 192x128 / "
                              f"64x192 / 128x128 / 64x128): verity_vllm.properties.fa_tap_exactness")}]


HIDDEN_STREAM_BY_FA = {2: "fa2_hidden_m1_stream", 3: "fa3_hidden_m1_stream"}


def hidden_class_for(flash_attn_version: int | None) -> str:
    """The attention hidden-value class the run's TargetProfile (`effective_flash_attn_version`) requires: FA2 -> fa2_hidden_m1_stream,
    FA3 -> fa3_hidden_m1_stream.  ONE source: the same fact selects the Bet A attention Definition, the Match resolver's pattern and this."""
    return HIDDEN_STREAM_BY_FA[3 if flash_attn_version == 3 else 2]


def attention_hidden(manifest: Any, model: Any) -> dict[str, Any]:
    """The attention hidden class this Commit binds, `{"family": class | None, "why": ...}`: None when the model instantiated no
    FlashAttention impl (`engine.build.flash_attn_versions`; MLA / DSA sparse MLA, GLM-5.3: no `varlen_fwd`, no M1 tap fires) or the
    required-value `manifest` names none (`query.required`: the M1 stream is no Value of the Program); else the hidden family its
    identities name, or without a manifest the impls' FA version's class."""
    from verity_vllm.engine.build import flash_attn_versions

    vs = flash_attn_versions(model)
    if not vs:
        return {"family": None, "why": "no attention impl carries vllm_flash_attn_version (MLA)"}
    fams = sorted({r.get("family") for r in (manifest or {}).get("identities") or ()} & set(HIDDEN_STREAM_BY_FA.values()))
    if len(fams) > 1:
        raise ValueError(f"required manifest names two attention hidden families {fams}")
    if fams:
        return {"family": fams[0], "why": "required manifest identities"}
    if manifest is not None:
        return {"family": None, "why": "the required manifest names no attention hidden stream"}
    return {"family": hidden_class_for(int(next(iter(vs))) if len(vs) == 1 and None not in vs else None),
            "why": f"the attention impls' vllm_flash_attn_version {sorted(vs, key=str)}"}


def with_hidden_class(classes: tuple[str, ...], family: str | None) -> tuple[str, ...]:
    """`classes` with its attention hidden class replaced by `family` in place, or dropped when `family` is None."""
    hidden = set(HIDDEN_STREAM_BY_FA.values())
    return tuple(family if c in hidden else c for c in classes if family or c not in hidden)


def require_hidden_on_every_rank(made: list[dict[str, Any]], required_classes: tuple[str, ...], versions: dict[str, Any], *,
                                 explicit_classes: bool) -> tuple[str, ...]:
    """The TP driver's side of `attention_hidden`: every rank reports one family (else SystemExit, fail closed), recorded in `versions`;
    unless the classes were given, the default's hidden class becomes that family (dropped when None)."""
    states = {int(m["rank"]): m["attention_hidden"] for m in made if "attention_hidden" in m}
    if not states:
        return required_classes
    fams = {s.get("family") for s in states.values()}
    if len(fams) > 1 or len(states) != len(made):
        raise SystemExit(f"fail-closed: the ranks disagree on the attention hidden class: {states} of {len(made)} ranks")
    versions["attention_hidden"] = states
    if explicit_classes:
        return required_classes
    fam = fams.pop()
    out = with_hidden_class(required_classes, fam)
    if out != required_classes:
        versions["required_classes"] = list(out)
        versions["required_classes_source"] = f"{versions.get('required_classes_source', '')} + attention hidden class {fam or 'none'} ({states[min(states)]['why']})"
        print(f"[tp2commit] attention hidden class {fam or 'none'} on every rank ({states[min(states)]['why']}); required classes {list(out)}", flush=True)
    return out


def make_hidden_source(committer, so_path: str, ns: str | None = None, flash_attn_version: int | None = None, fa2_cls=None, fa3_cls=None,
                       manifest=None):
    """The M1 source matching the tap build at `so_path` (an FA3 v9 build exposes `verity_set_tap_v9`; FA2 v7/v8 builds `verity_set_tap`).
    Refuses a tap of the other attention implementation than the engine's `flash_attn_version` (when given): the FA2 tap loaded under
    FA3 never fires -- that is a configuration error to report at attach time, not a zero to discover at the verdict.  With the
    required-value `manifest`, also refuses a build whose ROW word 3 or MS class is not the manifest's (`require_row3`)."""
    import torch

    ns = ns or __import__("pathlib").Path(so_path).stem
    torch.ops.load_library(so_path)
    is_fa3 = hasattr(getattr(torch.ops, ns), "verity_set_tap_v9")
    if flash_attn_version is not None and (flash_attn_version == 3) != is_fa3:
        raise RuntimeError(f"hidden tap {so_path} is an {'FA3' if is_fa3 else 'FA2'} build but the engine executes FA{flash_attn_version}: "
                           f"use {'verity_fa3_matReq.so (pod_fa3_tap.sh)' if flash_attn_version == 3 else 'verity_fa2_matReq.so (pod_fa2_tap.sh)'}")
    cls = (fa3_cls or HiddenM1SourceFA3) if is_fa3 else (fa2_cls or HiddenM1Source)     # a committer may supply its native (C++) twins
    src = cls(committer, so_path, ns=ns)
    if manifest is not None:
        src.require_row3(guarded_max_policy(manifest))
    return src
