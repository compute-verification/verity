import Pous.Protocol.Model.PageFill
import Pous.Protocol.Accounting.EraseMeetsDistinct

/-!
# Secure erasure of the keyed page fill, with distinct byte reads

The fill the deployed verifier uses (`verity_pous.erasure.KeyedFill`, audited by `ErasureVerifier`): pages of 1 MiB,
each a keyed `Feistel10` permutation of the zero page, read back a byte at a time at `s` distinct indices, each read
within a deadline `T_read` below one page's recomputation. The model is `Pous/Protocol/Model/PageFill.lean` (each page's
permutation ideal, as P3 idealizes `Π₂`), the read draw `Pous/Protocol/Model/ReadDraw.lean`, the requirement
`EraseMeetsDistinct` (`Pous/Protocol/Accounting/EraseMeetsDistinct.lean`). Proofs:
`Pous/SecurityProofs/SecureErasure/{ReductionDraw,PagesAudit,Pages}.lean`.

**What they rest on.** The ideal-permutation model of the pages, and the deadline: a read within `T_read` makes no
permutation query (`D = 0`). No named assumption, no bandwidth-separation premise (`TwoTierBandwidth` is not used),
and no idealization beyond P3's.

**What they say.** Every guarantee about erasure here is about the foreign content kept at every spot read's arrival:
an accepted prover cannot keep it through all the reads. None says what a prover holds at any single arrival.
-/

namespace Pous.Guarantees

open scoped ENNReal

/-- **Secure erasure from the sequential audit, over any read draw.** `SecureErasureMeets` with the spot reads' indices
`rd r` for a uniform `r : R` (`Pous/Protocol/Model/ReadDraw.lean`), by the same counting: if the sequential audit over
`rd` accepts every prover with `S'` bits of state at each arrival with probability at most `δ + ε`, then the
secure-erasure audit over `rd`, with any digest `dig`, accepts a prover with `S` bits of volatile memory that keeps
`S_φ` bits of foreign content at every spot read's arrival with probability at most
`δ + ε + s · 2^S / (2^S' · 2^S_φ)`. With `rd = id` on `Fin s → Fin B` this is `SecureErasureMeets`; with `distinctReads`
reads it is what the page fill needs. It rests on whatever the audit bound rests on, and on nothing else. Proof:
`PousErase.secure_erasure_meets_draw`. -/
def SecureErasureMeetsDraw : Prop :=
  ∀ {M : IdealModel} (sch : Scheme M) (K V : Type) [Fintype K] [Nonempty K] (dig : K → sch.Code → V)
    (R : Type) [Fintype R] [Nonempty R] {s : ℕ} (rd : R → Fin s → Fin sch.B) (S S' S_φ D Q : ℕ) (δ ε : ℝ≥0∞),
    AuditSecureDraw sch rd S' D Q δ ε →
      EraseSecureDraw sch rd dig S S_φ D Q (δ + ε + (s : ℝ≥0∞) * 2 ^ S / (2 ^ S' * 2 ^ S_φ))

/-- **The page fill's sequential audit at `D = 0`, with distinct byte reads.** For every page size `pb`, page count
`1 ≤ P ≤ 2^64`, key `K`, state size `S`, query budget `Q`, number of reads `s ≤ B` and slack `w ≤ B` (`B = P·pb`
bytes), every prover with `S` bits of state at each spot read's arrival, whose reads make no query to the pages'
permutations (`D = 0`: a deadline below one page's recomputation), is accepted by `s` reads at distinct uniform byte
indices with probability at most

  `s · 2^S · C(B, w) / 2^(8·(B − w)) + (B − w)^s / (B·(B − 1)⋯(B − s + 1))`.

The first term: some round `j < s` and some `S`-bit state predict at least `B − w` bytes of the fill right. The fill is
uniform (the pages are independent ideal permutations of the zero page under distinct tweaks), so a fixed prediction
agrees with it on a fixed set of `B − w` bytes with probability `2^(-8·(B − w))`, guesses included. The second: in
every other case each read is right with probability at most `(B − w)/(B − j)` given the reads before it, since a
round's answer depends only on the round, its state and its index. The preprocessing is unbounded. Proof:
`PousErase.page_audit_secure`. -/
def PageFillAuditSecure : Prop :=
  ∀ (pb P : ℕ) [NeZero pb] [NeZero P], P ≤ 2 ^ 64 → ∀ (K : Bits 256) (S Q s w : ℕ), w ≤ P * pb → s ≤ P * pb →
    AuditSecureDraw (PageFill.pageScheme pb P K) (distinctReads (PageFill.pageScheme pb P K) s) S 0 Q 0
      ((s : ℝ≥0∞) * 2 ^ S * ((P * pb).choose w : ℝ≥0∞) / 2 ^ (8 * (P * pb - w)) +
        ((P * pb - w : ℕ) : ℝ≥0∞) ^ s / ((P * pb).descFactorial s : ℝ≥0∞))

/-- **Secure erasure of the keyed page fill, at every fill from 1 page to 2^64 pages.** For every `P` with
`1 ≤ P ≤ 2^64` and every key `K`, the fill of `P` pages of 1 MiB (`pageScheme (2^20) P K`, `|C| = P · 2^23` bits)
meets `EraseMeetsDistinct` at `f = 1/19`, `D = 0` (any `Q`), `s = 3430` spot reads at distinct byte indices and
`ε = 2^-128`:

- every prover whose volatile memory at each spot read's arrival is at most `⌊(18/19)·|C|⌋` bits is accepted with
  probability at most `2^-128`;
- every prover with `P` MiB of volatile memory that keeps uniform foreign content of at least `P MiB / 19 + 32` bytes
  (`f·|C| + 2λ` bits; independent of the fill, the pages' permutations and the reads, and read from its memory alone)
  at every spot read's arrival wins with probability at most `2^-128`: an accepted prover cannot keep that foreign
  content through all the reads, except with probability `2^-128`.

Cite it as "cannot keep it through all the reads", for a verifier that draws 3430 distinct byte reads, each within a
`T_read` below one page's recomputation (`KeyedFill.recompute_ns`, 1.7352 ms for 1 MiB pages), which is what `D = 0`
models. The fill and its deadline `T_fill` do not enter: the fill is unbounded. `|C|` must be the prover's whole
volatile memory, on-chip SRAM and host DRAM included: memory outside the fill adds to what a prover can keep.

The numbers: the union term is at most `2^-130` once a prediction may be wrong on `w = 30324·P` bytes
(`94^30324 · 2^(7947103 + 8·30324 + 142) ≤ 2^(2^23)` per page, with `C(B, w) ≤ (e·B/w)^w`), and the sampling term
`((2^20 − 30324)/(2^20 − 3429))^3430 ≤ 2^-129` at the worst case `P = 1`; the foreign content's count is at most
`3430 · 2^-256 ≤ 2^-130`. The bound charges each read up to about `0.9711` for large `P`, not `18/19`: it allows for a
byte not held being guessed with probability `2^-8`, and for a memory of `18/19` of the fill holding a lossy summary of
all of it.

It rests on the ideal-permutation model of the pages (P3's idealization) and the `D = 0` deadline; no named assumption
and no bandwidth-separation premise. Proof: `PousErase.secure_erasure_pages_family`. -/
def SecureErasurePagesFamily : Prop :=
  ∀ (P : ℕ) [NeZero P], P ≤ 2 ^ 64 → ∀ (K : Bits 256) (Q : ℕ),
    EraseMeetsDistinct (PageFill.pageScheme (2 ^ 20) P K) (1 - Params.ρ) 0 Q 3430 ((2⁻¹ : ℝ≥0∞) ^ 128)

/-- **Secure erasure of the calibrated fill: 96112 pages of 1 MiB (93.86 GiB).** For every key `K`, the fill of
`P = 96112` pages of 1 MiB meets `EraseMeetsDistinct` at `f = 1/19`, `D = 0` (any `Q`), `s = 3046` spot reads at
distinct byte indices and `ε = 2^-128`: a prover whose memory at each spot read's arrival is at most `⌊(18/19)·|C|⌋`
bits is accepted with probability at most `2^-128`, and a prover with 93.86 GiB of volatile memory that keeps at least
`93.86 GiB / 19 + 32` bytes (about 4.94 GiB) of uniform foreign content (independent of the fill, the pages'
permutations and the reads, and read from its memory alone) at every spot read's arrival wins with probability at most
`2^-128`: an accepted prover cannot keep it through all the reads.

`P = 96112` is the fill the 4 Oct 2026 node-2 calibration reached on an RTX PRO 6000
(`art:3340ecbae35eb5e3732bec54c72f4196a94318f04d8d561f8dc83b6dbfc9c2cf`, 93.86 GiB), which shows software about 95 GiB of
its 96 GiB. The guarantee is about the 93.86 GiB the fill covers: memory outside the fill, the rest of the device's
memory, its on-chip SRAM and any host DRAM, adds to what a prover can keep.

Cite it as "cannot keep it through all the reads", for a verifier that draws 3046 distinct byte reads, each within a
`T_read` below one page's recomputation (1.7352 ms for 1 MiB pages; `D = 0`). The numbers: `w = 30335·P` wrong bytes
(`(9397/100)^30335 · 2^(7947103 + 8·30335 + 2) ≤ 2^(2^23)` per page, so the union term is at most `3046 · 2^(-2P) ≤ 2^-130`),
`((2^20 − 30335)·96112 / (2^20·96112 − 3045))^3046 ≤ 2^-129`, and the foreign content's count `3046 · 2^-256 ≤ 2^-130`.
Same footing as the family: the ideal-permutation model of the pages and the `D = 0` deadline, no named assumption.
Proof: `PousErase.secure_erasure_pages_96112`. -/
def SecureErasurePages96112 : Prop :=
  ∀ (K : Bits 256) (Q : ℕ),
    EraseMeetsDistinct (PageFill.pageScheme (2 ^ 20) 96112 K) (1 - Params.ρ) 0 Q 3046 ((2⁻¹ : ℝ≥0∞) ^ 128)

end Pous.Guarantees
