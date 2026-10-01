import Pous.P2.Cost

/-!
# P2's open steps in the ring model (named assumptions)

Each is a `Prop` taken as a hypothesis by the statements that use it, never an `axiom`: the joint bound for P2 in the
cost model of `Pous/P2/Cost.lean` (`RingLemmaA`, and `RingLemmaAP2` at P2's parameters), and the primality of P2's
modulus (`PrimeP16448`).
-/

namespace Pous.P2.Assumptions

open scoped ENNReal

variable {B w p : ℕ} [NeZero p]

/-- **Lemma A over the ring bound** (the per-run form, good event `True`): every `S`-bit state and every per-block
responder within `(D, Q, L)` answers at least `g` blocks correctly with probability at most `Γ`. -/
def LemmaAOn (sch : Scheme (ringRO B w p)) (S D Q L g : ℕ) (Γ : ℝ≥0∞) : Prop :=
  ∀ (W : Bits sch.n) (A₁ : Preprocessor sch S) (A₂ : ProdResponderOn sch S D Q L),
    pr (fun x : (ringRO B w p).Ω × sch.Coins =>
      let pp := (sch.enc W x.1 x.2).1
      let C := (sch.enc W x.1 x.2).2
      g ≤ (Finset.univ.filter fun i =>
        (A₂.prog W pp (A₁ W pp C x.1) i).run ((ringRO B w p).answer x.1) = C i).card)
      ≤ Γ

/-- **The joint bound (open)**: Lemma A for `p2ring` over the ring bound, at an effective domain of `⌊p/2^lam⌋`
points. It is the conjecture that a block not stored costs the responder two sequential roots: a fixed program
within `(D, Q, L)` answers it with probability at most `(c + (D + 1)·L²·2^(D+2)/p)/p`, and these per-block bounds
multiply over `g` given blocks as M1's do. The term `c/p` covers answers that skip a root (a guessed or unrelated
label), `L·2^(d+1)/p` bounds one root found within `d` multiplication rounds among `L` candidates (a nonzero
polynomial of degree at most `2^(d+1)` has at most that many roots), and the mask `σ_K` makes the second root's input
a fresh element, so the two roots' depths add up to `D`. Whenever `c·p + (D + 1)·L²·2^(D+2) ≤ 2^lam·p`, the
per-block factor is at most `2^lam/p ≤ 1/⌊p/2^lam⌋`, and Lemma A holds with M1's bound `gammaM1` at that domain,
keeping its hit term with `Q` key queries.

Lean proves the rest of the route (`PousP2Ring.meetsOn_of_row`). This step is open: the runs' polynomials for one block
can involve other blocks' elements, so conditioning does not give the product, and the proof needs a compression or
presampling argument over the labeling and the keys (as in Corrigan-Gibbs–Kogan, EUROCRYPT 2018, and
Coretti–Dodis–Guo, CRYPTO 2018). It is satisfiable (`PousP2Ring.ringLemmaA_sat`, at `c = p`). -/
def RingLemmaA (B w p b c : ℕ) [NeZero B] [NeZero p] : Prop :=
  ∀ S D Q L g lam : ℕ, c * p + (D + 1) * L ^ 2 * 2 ^ (D + 2) ≤ 2 ^ lam * p →
    LemmaAOn (p2ring B w p b) S D Q L g (gammaM1 B S Q g (p / 2 ^ lam))

/-- **The joint bound at P2's parameters**: `RingLemmaA` with `c = 2` at `2^19` blocks of P2's modulus
`p = 2^16448 − 21065`, for every payload width in range (`20·16448 ≤ 21·b`, `b ≤ 16447`). This is the instance the
`k = 88` row uses (`PousP2Ring.meetsOn_b19_k88`); `ringLemmaA_sat`'s witness is at `c = p`, not at `c = 2`. -/
def RingLemmaAP2 : Prop :=
  ∀ b : ℕ, 20 * 16448 ≤ 21 * b → b ≤ 16447 → RingLemmaA (2 ^ 19) 16448 (2 ^ 16448 - 21065) b 2

/-- **P2's registered modulus is prime (open in Lean)**: `p = 2^16448 − 21065`. Its primality certificate is ECPP, and
Lean checks none here. `p ≡ 3 mod 4` is proved (`PousP2Ring.p16448_mod_four`). -/
def PrimeP16448 : Prop := Nat.Prime (2 ^ 16448 - 21065)

end Pous.P2.Assumptions
