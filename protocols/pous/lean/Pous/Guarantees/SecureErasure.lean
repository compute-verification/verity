import Pous.Protocol.Model.Digest
import Pous.Protocol.Model.BandChain
import Pous.Protocol.Accounting.EraseMeets

/-!
# Secure erasure: volatile memory in full use through the spot reads

Added on 4 Oct 2026, under the model Daniel set that day: the disks are unplugged, so the prover's memory is volatile
only (HBM and host DRAM), and the audit is a proof of memory utilization. The game is
`Pous/Protocol/Model/SecureErasure.lean`, the requirement `EraseMeets` (`Pous/Protocol/Accounting/EraseMeets.lean`).
Proofs: `Pous/SecurityProofs/SecureErasure/`.

**What they rest on.** The ideal-primitive model and the `(D, Q)` game the band family is proved in, and the
incompressibility of the keyed fill there (the band's `TimedINC`, proved, `PousSpongeChain.chainExPostFactoG_v6`): a
timed spot read within `(D, Q)` needs the block, or what it takes to rebuild it within `D` rounds, in memory. No named
assumption. **No bandwidth-separation premise:** none of them reads or assumes the host-link bound `TwoTierBandwidth`
(PR #1010, dropped on 3 Oct 2026); HBM and host DRAM count alike.
-/

namespace Pous.Guarantees

open scoped ENNReal

open Pous.Digest in
/-- **The digest is almost universal.** Two different lists of `R` blocks of `W` words over a finite field `F` have the
same digest at a uniform key `(g, z)` with probability at most `((W − 1) + (R − 1)) / |F|`: either some block hash
differs, and the digests agree at at most `R − 1` points `z`, or every block hash agrees, which a block that differs
allows at at most `W − 1` points `g`. A word enters as an element of `F`: two `uint32` words that agree mod `p` count as
equal. Unconditional. This is what makes the fill check's digest catch a region that differs from the fill; the
secure-erasure bounds below hold for every digest and do not use it. Ported from PR #1010, unchanged but for the
namespace. Proof: `PousDigest.digestAU`. -/
def DigestAU : Prop :=
  ∀ (R W : ℕ) (F : Type) [Field F] [Fintype F] (xs ys : Fin R → Blk W F), xs ≠ ys →
    pr (fun k : Key F => digest k.1 k.2 xs = digest k.1 k.2 ys) ≤
      ((W - 1 + (R - 1) : ℕ) : ℝ≥0∞) / Fintype.card F

/-- **Secure erasure from the timed audit (general).** For every scheme whose sequential audit accepts every prover with
`S'` bits of state at each arrival with probability at most `δ + ε`, the secure-erasure audit over it, with any digest
`dig`, accepts a prover with `S` bits of volatile memory that keeps `S_φ` bits of foreign content at every spot read's
arrival with probability at most `δ + ε + s · 2^S / (2^S' · 2^S_φ)`. So at `S_φ ≥ S − S' + λ + log₂ s` the foreign
content costs the prover `2^-λ`: whatever it keeps of `z` takes the place of the fill.

The proof counts. For each round `j`, the memories from which `keep j` reads `z` are disjoint as `z` varies, so fewer
than `2^S / 2^S'` values of `z` have more than `2^S'` of them, and a uniform `z` is such a value at some round with
probability at most `s · 2^S / (2^S' · 2^S_φ)`. For every other `z`, the memories at the arrivals that keep it have an
`S'`-bit code, and the prover run on the code is an `S'`-bit sequential responder (same programs, same `(D, Q)`), which
the audit's bound covers.

It rests on what `AuditSecure` rests on for the scheme (for the band: the ideal-primitive model, the keyed fill's
incompressibility and the `(D, Q)` game), and on nothing else: no named assumption and no bandwidth-separation premise
(`TwoTierBandwidth`, dropped on 3 Oct 2026, is not used). The memory is bounded at every spot read's arrival; the fill
and the digest are unbounded. Proof: `PousErase.secure_erasure_meets`. -/
def SecureErasureMeets : Prop :=
  ∀ {M : IdealModel} (sch : Scheme M) (K V : Type) [Fintype K] [Nonempty K] (dig : K → sch.Code → V)
    (S S' S_φ D Q s : ℕ) (δ ε : ℝ≥0∞),
    AuditSecure sch .sequential S' D Q s δ ε →
      EraseSecure sch dig S S_φ D Q s (δ + ε + (s : ℝ≥0∞) * 2 ^ S / (2 ^ S' * 2 ^ S_φ))

/-- **Secure erasure on the band, at every memory size from 32 MiB to 2^64 segments.** For every `N` with
`1 ≤ N ≤ 2^64` and every in-degree `d ≥ 5`, the `N`-segment band `chainBandSeg N (2^9) d` (`2^9` blocks of 64 KiB per
segment, so a fill of `M = N · 32 MiB`), keyed by the verifier's fresh key as its salt (per-segment tags `segDom salt`,
any 192-bit salt, or one global 256-bit salt), meets `EraseMeets` at `f = 1 − ρ = 1/19`, `D = 5`, `Q = 2^20`,
`s = 2152` spot reads and `ε = 2^-128`:

- every prover whose volatile memory at each spot read's arrival is at most `⌊(18/19)·|C|⌋` bits is accepted with
  probability at most `2^-128`;
- every prover with `M` bytes of volatile memory that keeps foreign content of at least `M/19` bytes and 32 bytes
  (`f·|C| + 2λ` bits) through the spot reads wins with probability at most `2^-128`. So no prover is accepted and still
  holds that foreign content at every spot read's arrival, except with probability `2^-128`. This bounds what survives
  the reads, not what is held at one instant: a prover can hold about `M/10` bytes of foreign content at the first spot
  read's arrival and be accepted with probability about 0.9, erasing it while it rebuilds, during the first reads, the
  blocks it dropped.

Cite it as "cannot keep it through all the reads", never as "holds at the challenge", and for a verifier that draws
2152 sequential reads under a deadline bounding `(D, Q) = (5, 2^20)`.

`M` must be the prover's whole volatile memory, on-chip SRAM included; with host DRAM, `N` is the total over 32 MiB.
`N = 3072` is 96 GiB (`SecureErasureMeets96GiB`), the GDDR7 an RTX PRO 6000 has, but no fill covers all of it: software
sees about 95 GiB, and the 4 Oct 2026 calibration filled 93.86 GiB (`N = 3003`). On one device the guarantee holds at
the `N` the fill reaches, and the memory outside the fill adds to what a prover can keep. The read deadline `T_read` is a symbol: any
deadline within which the prover completes at most 5 rounds of `Π₂` calls and `2^20` calls per answer; its hardware
calibration is separate. The fill deadline does not enter.

The numbers: `9593/10000` bounds the band's certified per-read success `p₀` at `⌊ρ·|C|⌋` bits for every `N` (one
segment is the worst, `p₀ ≈ 0.959292`), and `2152` is the least `s` with `(9593/10000)^s ≤ 2^-129`. That is more than the
`1641` reads that `(18/19)^s ≤ 2^-128` alone would need, because the band's incompressibility bound `TimedINC` has slack
beyond `ρ`. The error splits as `2^-129` (sampling) `+ 2^-130` (INC's union over histories) `+ 2^-130` (the foreign
content's count, at most `2152 · 2^-256`).

It rests on the ideal-primitive model, the keyed fill's incompressibility (the band's `TimedINC`, proved) and the
`(D, Q)` game, as `BandMultiMeetsFamily` does; no named assumption and no bandwidth-separation premise
(`TwoTierBandwidth`, dropped on 3 Oct 2026, is not used). Proof: `PousErase.secure_erasure_meets_family`. -/
def SecureErasureMeetsFamily : Prop :=
  ∀ (N : ℕ) [NeZero N], N ≤ 2 ^ 64 → ∀ d : ℕ, 5 ≤ d →
    (∀ salt : Bits 192,
      EraseMeets (Pous.BandMulti.chainBandSeg N (2 ^ 9) d 524288 512 (Pous.BandMulti.segDom salt (2 ^ 9)))
        (1 - Params.ρ) 5 (2 ^ 20) 2152 ((2⁻¹ : ℝ≥0∞) ^ 128)) ∧
    (∀ salt : Bits 256,
      EraseMeets (Pous.BandMulti.chainBandSeg N (2 ^ 9) d 524288 512 (Pous.Sponge.dom salt))
        (1 - Params.ρ) 5 (2 ^ 20) 2152 ((2⁻¹ : ℝ≥0∞) ^ 128))

/-- **Secure erasure of 96 GiB.** `SecureErasureMeetsFamily` at `N = 3072` segments of 32 MiB, with the implementation's
per-segment tags. For every `d ≥ 5` and salt, and a prover whose whole volatile memory is the 96 GiB: if its memory at
each spot read's arrival is at most `⌊(18/19)·|C|⌋` bits it is accepted with probability at most `2^-128`, and if it
keeps at least `96 GiB / 19 + 32` bytes (about 5.05 GiB) of foreign content through the 2152 spot reads it wins with
probability at most `2^-128`. 96 GiB is an RTX PRO 6000's GDDR7, but one device cannot hold this fill (about 95 GiB is
visible to software); for one device, cite the family at the `N` its fill reaches. Same footing as the family: no named
assumption, no bandwidth-separation premise. Proof: `PousErase.secure_erasure_meets_96GiB`. -/
def SecureErasureMeets96GiB : Prop :=
  ∀ d : ℕ, 5 ≤ d → ∀ salt : Bits 192,
    EraseMeets (Pous.BandMulti.chainBandSeg 3072 (2 ^ 9) d 524288 512 (Pous.BandMulti.segDom salt (2 ^ 9)))
      (1 - Params.ρ) 5 (2 ^ 20) 2152 ((2⁻¹ : ℝ≥0∞) ^ 128)

end Pous.Guarantees
