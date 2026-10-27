import Definitions.Accounting.BoundSketch
import Mathlib.Analysis.Real.Sqrt

/-!
# The deployed bound receipt's sketch: what a relay must exchange (trusted)

The game is `Security/Definitions/Accounting/BoundSketch.lean`; proofs are in
`Security/Proofs/Accounting/BoundReceipt/Sketch.lean`. No named assumption of its own: the two sides' inputs are
independent with given guessing probabilities, which is the ideal tile and the ideal store, read in this model.

* `InnerProductRelay`: a protocol of `c` bits that ends with either side knowing `⟨x, y⟩` over `𝔽_p^n`, for
  independent inputs whose guessing probabilities are at most `κ_E` and `κ_R`, succeeds with probability at most
  `1/p + 2^c·√(κ_E·κ_R·p^n)` (Lindsey's lemma, by Cauchy–Schwarz and the kernels of `𝔽_p^n`'s linear forms).
* `BoundSketchRelay`: the same for the deployed sketch of a tile, `r` vectors of `m` bytes against `r` columns of
  `m` bytes: the sketch's trace is that inner product, at `n = r·m`.
-/

namespace Accounting.Guarantees

open Accounting.BoundSketch

/-- **A relay that ends knowing an inner product over `𝔽_p^n` exchanges about half the inputs' joint entropy beyond
`n·log₂ p`.** For every prime `p`, finite index set `ι` (`n = |ι|`), distributions `μ` of the block side's input and
`ν` of the tile side's (independent), with every value at most `κ_E` and `κ_R` likely, and every deterministic
protocol `π`: the probability that `π`'s output is `⟨x, y⟩` is at most `1/p + 2^bits(π)·√(κ_E·κ_R·p^n)`.

**The reading.** With `κ_E = 2^{−h_E}` and `κ_R = 2^{−h_R}` (min-entropies), success `1/p + ε` takes
`bits ≥ ½(h_E + h_R − n·log₂ p) − log₂(1/ε)` bits, both directions together. There is no amortization: `t` relayed
tiles with independent inputs are one inner product over `𝔽_p^{t·n}`, whose joint min-entropies add. Both sides may
compute without bound; a randomized protocol, public or private coins, is an average of deterministic ones.

Proof: `Accounting.SecurityProofs.InnerProductRelay`: Cauchy–Schwarz over the side that outputs, then
`Σ_c ([c = ⟨x, y⟩] − 1/p)([c = ⟨x, y'⟩] − 1/p)` summed over `x` is `p^n(1 − 1/p)` at `y = y'` and `0` otherwise,
since a nonzero linear form on `𝔽_p^n` has `p^{n−1}` zeros; each bit sent splits one side's weight in two.

Frozen (no-function): an information-theoretic bound over abstract protocols, about no function of the remote
protocol's code. -/
def InnerProductRelay : Prop :=
  ∀ (p : ℕ) [Fact p.Prime] (ι : Type) [Fintype ι] [DecidableEq ι] (μ ν : (ι → ZMod p) → ℝ) (κE κR : ℝ)
    (π : Protocol (ι → ZMod p) (ι → ZMod p) (ZMod p)),
    (∀ x, 0 ≤ μ x) → (∀ y, 0 ≤ ν y) → ∑ x, μ x = 1 → ∑ y, ν y = 1 → (∀ x, μ x ≤ κE) → (∀ y, ν y ≤ κR) →
      ∑ x, ∑ y, μ x * ν y * (if π.eval x y = ip x y then (1 : ℝ) else 0) ≤
        1 / p + 2 ^ π.bits * Real.sqrt (κE * κR * (p : ℝ) ^ Fintype.card ι)

/-- **A relay that ends knowing a bound receipt's sketch exchanges about half of `min(H(R), 8|E|) − 0.006|E|` bits.**
For every prime `p`, tile of `r` vectors of `m` bytes, distributions `μ` of the tile's bytes `E` and `ν` of the
multiplier matrix `B` (independent), with every value at most `κ_E` and `κ_R` likely, and every deterministic
protocol `π` between the side holding `E` and the side holding `B`: the probability that `π`'s output is the sketch
`v_i[c] = Σ_q E(i, q)·B(c, q)` is at most `1/p + 2^bits(π)·√(κ_E·κ_R·p^{r·m})`.

**The reading** (`ro-bound-sketch-blake3/v0`, at `p = 257`). The block side holds the tile's bytes `E` (`|E| = r·m`
bytes, int8, embedded in `𝔽₂₅₇`), the tile side the bytes `R` of the matmul's values. Both know the pass's mask `M`
(public, fresh each pass), so `B = R ⊕ M` is a bijection of `R` and `κ_R` is `R`'s guessing probability: the entropy
that counts is `R`'s, after slice mode's XOR fold, not `z`'s. For any random oracle `H` (BLAKE3 in the receipt, SHA-512
in the PoUW leaf), a receipt `H(σ_p, j, v)` is right, but for a guess, only if one side queried `H` at `v`, so a relay
that produces it is such a protocol. With `E` uniform (`κ_E = 2^{−8|E|}`) and `κ_R = 2^{−h_R}`, `h_R ≤ 8|E|`, success
`1/257 + ε` takes `bits ≥ ½(h_R − (log₂ 257 − 8)·|E|) − log₂(1/ε) ≥ ½(h_R − 0.006·|E|) − log₂(1/ε)`: about half of
`min(H(R), 8|E|)` a relayed tile, where `BoundReceiptLink`'s generic mix gives `min(|z|, |E|)`. One buffer `M` for every
tile of a pass costs nothing, since `M` is public either way; two receipts whose `R` is one `z` count it once, since
`κ_R` is the joint guessing probability of the inputs the protocol relays.

**What the game takes.** `E` and `R` independent, `R`'s min-entropy against the block side being the ideal tile's
(`TileUnpredictable` below its work: against a side that can compute `R` from the activations, `R` has none), and `E`'s
against the tile side the store's (the codec's work, charged as rebuilds). `M` is fixed before the protocol: a mask
fixed for the window is a public matrix, and leaves residency to the generic-mix game's fresh mix.

Proof: `Accounting.SecurityProofs.BoundSketchRelay`, from `InnerProductRelay` on the sketch's trace
`Σ_i v_i[i] = ⟨E, B⟩`, at `n = r·m`.

Frozen (no-function): as `InnerProductRelay`. -/
def BoundSketchRelay : Prop :=
  ∀ (p : ℕ) [Fact p.Prime] (r m : ℕ) (μ ν : (Fin r × Fin m → ZMod p) → ℝ) (κE κR : ℝ)
    (π : Protocol (Fin r × Fin m → ZMod p) (Fin r × Fin m → ZMod p) (Fin r → Fin r → ZMod p)),
    (∀ E, 0 ≤ μ E) → (∀ B, 0 ≤ ν B) → ∑ E, μ E = 1 → ∑ B, ν B = 1 → (∀ E, μ E ≤ κE) → (∀ B, ν B ≤ κR) →
      ∑ E, ∑ B, μ E * ν B * (if π.eval E B = sketch E B then (1 : ℝ) else 0) ≤
        1 / p + 2 ^ π.bits * Real.sqrt (κE * κR * (p : ℝ) ^ (r * m))

end Accounting.Guarantees
