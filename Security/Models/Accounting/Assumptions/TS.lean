import Definitions.Accounting.TS

/-!
# The tensor–scalar model's named assumptions (trusted)

The premises of EPS1-SPEC's Theorem D-ideal ((I1)–(I8), §2.7) and of the compiler lemma (Lemma C), each a `Prop` over
`Accounting.TS`'s vocabulary, taken as a hypothesis, never an `axiom`. Each docstring opens with its claim id
(`verity.core.claims`). An assumption over an `Anatomy` is claimed at §0.4's and §1.3's labelling of the cheater's
instructions (heads, ownership, copies); an assumption over a run is claimed for the stated device.

Which form each takes:

* (I1) `FacetCeilings`: a device-model assumption, scoped by device (sm_120 RTX PRO 6000, rows VF3p-16 on VF3o-16),
  checked by measurement (Daniel, 8 Oct): a corpus program that runs faster than a facet falsifies it.
* (I2) and the uniform part of (I4): `WordsWritten`, the ISA's `ω` (P12). Lemma 2 (`Accounting.Guarantees.TS.OutputCount`)
  is its counting; the ISA fact that an instruction writes at most `ω` words is the assumption.
* (I3) `ScalarRoute`: SCALAR-ROUTE as restated at 06:15Z, at its floors `(q, q_T, b_T) = (2.03, 1.22, 3.5)`. QSTAR
  derives it from class P and ISA-CAP (each sm_120 ALU instruction creates at most 32/7 incidences); class P is a
  modeling restriction and ISA-CAP is likely false as stated (QSTAR r3: IMAD.WIDE 8, IDP 5), so it stays an
  assumption.
* (I5) `SideUnits` (measured per-word vectors, P11) and `NoSum` (NOSUM, P13), a measured statement: no unit outside
  the menu heads a fresh word.
* (I6) `FoldOpt` and `FillOpt` in the price form D-ideal's proof reads. FOLD-OPT's per-word kernel (the duo step
  `u = y ⊕ d ⊕ k`, `y' = u·a + z`, `z' = rotl(u, r) + y'` needs three instructions over the modeled ISA,
  `benchmarks/pouw/theory/fold_opt.py`, #2240) is a mechanical fact; the multi-round residual, that no program folds
  below the menu's price across rounds, is what stays assumed.
* (I7) `LuckAt`: (L) at `p` with freshness (F1)–(F2), one bet a round.
* (I8) `ProofsSound`: deterministic computation, the circuit and inputs committed before σ, Theorem S's order, binding
  `δ_bind` and proof soundness `δ_ZK`, with Lemma E1(c) (every adaptive word lies below a committed value wrong for its
  own derivation). E1(c) is proved on paper; it sits inside this premise until the adaptive model is in Lean.
* `LinSteer`: SAMPLING's LIN-STEER, for the compiler's closure.

The joint-luck premise JL is not used: D-ideal keeps the proved `ε_jl`.
-/

namespace Accounting.Assumptions.TS

open Accounting.TS Pouw.RelativeSecurity

variable {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U] [Fintype F]
  [Fintype X]

/-- `accounting-ts-facet-ceilings` (measured, device-scoped). **The measured facets are ceilings** for the run `P` on
the stated device (sm_120 RTX PRO 6000; Π_G = VF3p-16 on VF3o-16, decision 73): the run's counts and cycles are
`≥ 0` and its clock is in `(0, f_max]`, every SM-clocked facet's load is at most the run's cycles (S_r), and every
fixed-clock facet's load, priced at `f_max`, is at most `f_max` times the run's time. Rows take the best configuration
measured plus a 2% margin; a joint row is a facet on the mixes measured, and beyond them it is a red-team
obligation. -/
def FacetCeilings (G : Machine K R) (P : Run K) : Prop :=
  (∀ k, 0 ≤ P.n k) ∧ 0 ≤ P.cycles ∧ 0 < P.clock ∧ P.clock ≤ G.fmax ∧
    ∀ r, (G.smClocked r = true → G.load P.n r ≤ P.cycles) ∧
      (G.smClocked r = false → G.load P.n r ≤ G.fmax * P.time)

/-- `accounting-ts-words-written` (ISA, P12). **An instruction writes at most `ω` words**: the owned right fixed words
headed by writer kind `t` are at most `ω_t` times the instructions of `t`'s class the labelling gives them. -/
def WordsWritten (Z : Model K R T L U F X) (A : Anatomy K T L U) : Prop :=
  ∀ t, A.words (.writer t) ≤ Z.omega t * A.nword (.writer t) (Z.cls t)

/-- `accounting-ts-scalar-route` (assumed; QSTAR: from class P and ISA-CAP). **SCALAR-ROUTE**, in §0.6's units
(lane-instructions a word, net of the mandatory fold and of operand delivery): the gather-free vector-headed words
cost at least `q` lane-instructions each (2.03), and the table words served from level `l` at least `q_T`
lane-instructions (1.22) and `b_T` bytes read from `l` (3.5) each. -/
def ScalarRoute (Z : Model K R T L U F X) (A : Anatomy K T L U) : Prop :=
  Z.q * A.words .vector ≤ 32 * ∑ k ∈ Z.vec, A.nword .vector k ∧
    ∀ l, Z.qT * A.words (.table l) ≤ 32 * ∑ k ∈ Z.vec, A.nword (.table l) k ∧
      Z.bT * A.words (.table l) ≤ A.nword (.table l) (Z.bytes l)

/-- `accounting-ts-side-units` (measured, P11). **A side unit's words cost its per-word vector**: the instructions the
labelling gives side unit `u`'s words are at least its per-word vector times its words, class by class. -/
def SideUnits (Z : Model K R T L U F X) (A : Anatomy K T L U) : Prop :=
  ∀ u k, A.words (.side u) * Z.side u k ≤ A.nword (.side u) k

/-- `accounting-ts-nosum` (measured, P13). **NOSUM**: no unit outside the menu heads a fresh committed word, so every
owned right fixed word has a head of the menu. -/
def NoSum (A : Anatomy K T L U) : Prop := A.fixed ≤ ∑ h, A.words h

/-- `accounting-ts-fold-opt` (assumed; its per-word kernel a mechanical fact). **FOLD-OPT_μ**: at every certificate,
the run's fold is priced at least `(1 − μ_f)·N_w·φ*(π)`. -/
def FoldOpt (Z : Model K R T L U F X) (A : Anatomy K T L U) : Prop :=
  ∀ lam : R → ℝ, IsCert lam → (1 - Z.muF) * Z.Nw * Z.phiStar lam ≤ Z.G.price lam A.nfold

/-- `accounting-ts-fill-opt` (assumed). **FILL-OPT_μ**: at every certificate, the run's filler is priced at least
`(1 − μ_x)·χ*(π)`. -/
def FillOpt (Z : Model K R T L U F X) (A : Anatomy K T L U) : Prop :=
  ∀ lam : R → ℝ, IsCert lam → (1 - Z.muX) * Z.chiStar lam ≤ Z.G.price lam A.nfill

/-- `accounting-ts-luck` ([P+R] per noise; ¾ in Lean for affine operands). **(L) at `p`, with freshness (F1)–(F2)**:
the ideal game is one bet a round, a deficient round passing with probability at most `p` given the past. -/
def LuckAt {Ω : Type} [Fintype Ω] {S : ℕ} (p : ℝ) (g : RoundGame Ω S) : Prop := RoundGame.LuckAt p g

/-- `accounting-ts-proofs-sound` (binding H-μ⁺, proof soundness, the protocol's order, Lemma E1(c)). **The proofs are
sound up to `S·(δ_bind + n_π·δ_ZK)`**: outside an event of that probability, an accepted window is accepted by the
ideal game, and its owned right fixed words per unit are at least `(1 − Σ_w d_w / S)·N_w` (no committed value is wrong
for its derivation, so every word that isn't fixed is in its round's share; the rounds carry equal shares of the
window's words). -/
def ProofsSound {Ω : Type} [Fintype Ω] {S : ℕ} (P : Play Ω S K T L U) (Nw δbind δzk : ℝ) (nπ : ℕ) : Prop :=
  ∃ E : Paths Ω S → Prop, prob E ≤ S * (δbind + nπ * δzk) ∧
    ∀ x, P.accepts x → ¬ E x → P.game.accepts x ∧ (1 - P.game.deficiency x / S) * Nw ≤ (P.run x).fixed

/-- `accounting-ts-lin-steer` (assumed; SAMPLING). **LIN-STEER**: zeroing a fraction `φ = |Z|/|act|` of the row's
noise-active coordinates saves at most `φ` of the consumer's work. -/
def LinSteer {m : ℕ} (c : Consumer m) : Prop :=
  ∀ Z ⊆ c.act, c.cost ∅ - c.cost Z ≤ (Z.card : ℝ) / c.act.card * c.cost ∅

end Accounting.Assumptions.TS
