import Pous.Protocol.Game.Prob
import Pous.Protocol.Model.Bandwidth
import Pous.Assumptions.Bandwidth

/-!
# The bulk read's guarantees

The digest's almost-universality, and the placement guarantee in two forms: for any deadline `T`, and at a deadline
tuned to a device rate `BWdev`, where the threshold is `f* = BWpcie / BWdev` (`Pous/Protocol/Model/Bandwidth.lean`).
Proofs: `Pous/SecurityProofs/Bandwidth/`.
-/

namespace Pous.Guarantees

open Pous.Bandwidth Pous.Assumptions
open scoped ENNReal NNReal

/-- **The digest is almost universal.** Two different lists of `R` blocks of `W` words over a finite field `F` have the
same digest at a uniform key `(g, z)` with probability at most `((W − 1) + (R − 1)) / |F|`: either some block hash
differs, and the digests agree at at most `R − 1` points `z`, or every block hash agrees, which a block that differs
allows at at most `W − 1` points `g`. At the benchmark's point (`W = 16,384`, `R = 8,192`, `|F| = 2^31 − 1`) that is
about `2^-16.4`. A word enters as an element of `F`: two `uint32` words that agree mod `p` count as equal. -/
def DigestAU : Prop :=
  ∀ (R W : ℕ) (F : Type) [Field F] [Fintype F] (xs ys : Fin R → Blk W F), xs ≠ ys →
    pr (fun k : Key F => digest k.1 k.2 xs = digest k.1 k.2 ys) ≤
      ((W - 1 + (R - 1) : ℕ) : ℝ≥0∞) / Fintype.card F

/-- **The bulk read pins the challenged blocks to device memory.** Under `TwoTierBandwidth`, if the challenged blocks
that only host memory holds are more than the host link carries by the deadline (`BWpcie · T < b · h`, with `h` of
them), the verifier accepts with probability at most `(h + 1) / |F|`, over a uniform store and a uniform key. So a
server that passes holds all but at most `BWpcie · T / b` of the challenged blocks in device memory.

The challenge `S` is any `R` distinct blocks; `h` counts those of them in `host`. The responder reads the store only
through its program (`BulkResponder`), the store is uniform, and the responder holds nothing else about it: the model
of `C`'s pseudorandom labels. Proof: a response within `T` reads fewer than `h` host-resident blocks
(`TwoTierBandwidth`), so it leaves a challenged block `j` unread, and its answer does not change with `j`'s content. The
last word of `j` enters the digest with the coefficient `z^r`, so the answer is right for at most one value of that
word when `z ≠ 0`.

What it does not say: anything about a challenge the verifier did not send (drawing `S` and the fraction of `C` in
host memory are the sampling step, not proved here); anything about a responder that keeps summaries of host blocks
in device memory, or recomputes a block from others within `T` (outside the model; the fresh key is what defeats
summaries, see `DigestAU`); or that the honest server meets `T` (a measurement, #935). -/
def HbmPlacement : Prop :=
  ∀ (N R W : ℕ) (F : Type) [Field F] [Fintype F], 0 < W →
    ∀ (host : Finset (Fin N)) (b BWpcie T : ℝ≥0) (A : BulkResponder N R W F),
      TwoTierBandwidth host b BWpcie A →
      ∀ S : Fin R → Fin N, Function.Injective S →
        BWpcie * T < b * ((Finset.univ.image S ∩ host).card : ℝ≥0) →
        pr (fun x : Store N W F × Key F => Accepts A T S x.1 x.2) ≤
          (((Finset.univ.image S ∩ host).card + 1 : ℕ) : ℝ≥0∞) / Fintype.card F

/-- **The threshold `f* = BWpcie / BWdev`.** `HbmPlacement` at a deadline tuned to a device rate `BWdev`: if `T` is at
most the time to read the `R` challenged blocks at `BWdev` (`T · BWdev ≤ R · b`), and the challenged fraction that only
host memory holds, `f = h / R`, exceeds `f* = BWpcie / BWdev`, the verifier accepts with probability at most
`(h + 1) / |F|`. `BWdev` is the verifier's parameter, the rate the honest server reaches; it is not assumed to bound
anything. At #935's numbers (`BWpcie ≈ 56 GB/s`, `BWdev ≈ 1.54 TB/s`) `f* ≈ 3.6%`; its recommended deadline,
`T = 0.50 ms` for `R = 8,192` blocks of 64 KiB, is `BWdev ≈ 1.07 TB/s` here, where `f* ≈ 5.2%`. -/
def HbmPlacementFStar : Prop :=
  ∀ (N R W : ℕ) (F : Type) [Field F] [Fintype F], 0 < W →
    ∀ (host : Finset (Fin N)) (b BWpcie BWdev T : ℝ≥0) (A : BulkResponder N R W F),
      TwoTierBandwidth host b BWpcie A → 0 < b → 0 < BWdev → T * BWdev ≤ R * b →
      ∀ S : Fin R → Fin N, Function.Injective S →
        BWpcie / BWdev < ((Finset.univ.image S ∩ host).card : ℝ≥0) / R →
        pr (fun x : Store N W F × Key F => Accepts A T S x.1 x.2) ≤
          (((Finset.univ.image S ∩ host).card + 1 : ℕ) : ℝ≥0∞) / Fintype.card F

end Pous.Guarantees
