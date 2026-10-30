import Proofs.Flock.Claims.Recursion.KSLin2
import Proofs.Flock.Claims.Bridge.CommittedNil

/-!
# `KSLin2On` at Hello-committed coins

`KSLin2On` reads the compiled recursive verifier's run at coins drawn after each message. Deployed, the coins are
committed at Hello in an hm96 coin tree and each opened with its round's answer, except where that answer is `[]` (a
V\* session before the last one, accepted). The prover reads every root at Hello, so a deployment that sends a tree's
root later gives it less. `KSLin2OnC` is `KSLin2On`'s bound at that run: every hm96 tree whose leaves hide at `δ₁ ≠ ∞`
(`Hm96Hiding`) has a simulator `S` such that, for every prover that reads the root and the openings, the committed run
accepts at most as often as `KSLin2On`'s extractor, run on the prover with `S`'s root and openings
(`Bridge.simProverM`), recovers a witness, plus `KSLin2On`'s bound at that prover (`lin2Rhs`) and `2 · k(x) · δ₁`.
`ksLin2OnC_of_ksLin2On` derives it from `KSLin2On` (`Bridge.Pr_acc_runCM_le`).
-/

namespace Claims.Recursion

open Claims
open scoped ENNReal

section Scoped

variable {Coin Col T Col' T' : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
  [DecidableEq Col']

/-- **`KSLin2On`'s bound at the prover `P`**: `κ + √(γ · Pr[B])` and the inner verifier's bad-abort, forked-coin and
first-coin terms, at `P` and the counts `M`, `R`. -/
noncomputable def lin2Rhs [Zero Col] [Zero Col'] (H : Msg → Msg) (D : RecAt Coin) (Bd : Bound Coin Col T)
    (DO : Flock.Deploy Coin Col' T') (Cp : Caps Coin) (decI : List (ℕ × Col) → List T)
    (decO : List (ℕ × Col') → List T') (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (M R : ℕ) : ℝ :=
  κ (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) x +
    Real.sqrt (γ (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) x * PrB H (D.record Bd.IVStar).V P x) +
    (4 * (Inner.ofFlock Bd.deploy decI).V.k (xI (D.record Bd.IVStar) x) + 1) *
      PrBinnerV H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO P x +
    4 * Cp.width (xI (D.record Bd.IVStar) x) *
      (PrFCI H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO Cp P x M +
        1 / ((M : ℝ) + 1)) +
    4 * (PrF0I H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO P x R +
      (((Inner.ofFlock Bd.deploy decI).k (xI (D.record Bd.IVStar) x) : ℝ) + 1) / ((R : ℝ) + 1)) +
    βdev2 (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) Cp x *
      (PrFdevV H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO P x M +
        1 / ((M : ℝ) + 1))

/-- **`KSLin2On` at Hello-committed coins**, at a leaf hiding bound `δ₁ ≠ ∞`: at every statement `G` admits, every inner
and outer decoder meeting the spec and every hm96 coin tree whose leaves hide at `δ₁`, a simulator `S` of the tree such
that, for every prover that reads its root and openings (`Bridge.runCM`), the committed run's acceptance is at most the
probability
that `KSLin2On`'s extractor, run on the prover with `S`'s root and openings (`Bridge.simProverM`), recovers a witness,
plus `KSLin2On`'s bound at that prover (`lin2Rhs`) and `2 · k(x) · δ₁`. -/
def KSLin2OnC [Zero Col] [Zero Col'] (H : Msg → Msg) (D : RecAt Coin) (Bd : Bound Coin Col T)
    (DO : Flock.Deploy Coin Col' T') (Cp : Caps Coin) (G : Msg → Prop) (δ₁ : ℝ≥0∞) : Prop :=
  δ₁ ≠ ⊤ → ∀ decI, decOK Bd.deploy decI → ∀ decO, decOK DO decO → ∀ (x : Msg), G x →
    ∀ (B Y C T₀ : Type) [AddGroup B] [Fintype B] [Fintype Y] [Nonempty Y] [Fintype T₀] [Nonempty T₀]
      (Tr : Bridge.Hm96Tree Coin B Y C T₀ ((D.record Bd.IVStar).V.k x)), Hm96Hiding Tr.My Tr.cs δ₁ →
    ∃ S : Bridge.OnlineSim, ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Bridge.CProver Ω) (M R : ℕ),
      Pr (fun p : Ω × (Fin ((D.record Bd.IVStar).V.k x) → Coin) × (Fin ((D.record Bd.IVStar).V.k x) → Y) × T₀ =>
          (Bridge.runCM H P (D.record Bd.IVStar).V x Tr.tree p.1 p.2.1 p.2.2).out = .acc) ≤
        Pr (fun s : (Ω × S.Tape) × (Fin ((D.record Bd.IVStar).V.k x) → Coin) ×
              (E H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO).C x =>
            (Recursion.R (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI)).ext x
              ((E H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO).eval H
                (Bridge.simProverM S (D.record Bd.IVStar).V.k P) x s.1 s.2.1 s.2.2).1) +
          lin2Rhs H D Bd DO Cp decI decO x (Bridge.simProverM S (D.record Bd.IVStar).V.k P) M R +
          2 * (D.record Bd.IVStar).V.k x * δ₁.toReal

/-- **`KSLin2On` gives `KSLin2OnC`** wherever the compiled record's verifier answers each round with its coin, through
an injective `ans`, or with nothing: `Bridge.Pr_acc_runCM_le` at `KSLin2On`'s extractor, with `PrBad` at the simulating
prover bounded by `KSLin2On` there. -/
theorem ksLin2OnC_of_ksLin2On [Zero Col] [Zero Col'] {H : Msg → Msg} {D : RecAt Coin} {Bd : Bound Coin Col T}
    {DO : Flock.Deploy Coin Col' T'} {Cp : Caps Coin} {G : Msg → Prop} (h : KSLin2On H D Bd DO Cp G)
    (ans : Coin → Msg) (hans : Function.Injective ans)
    (hV : ∀ x, Bridge.AnswersCoinOrNil H (D.record Bd.IVStar).V x ans) (δ₁ : ℝ≥0∞) :
    KSLin2OnC H D Bd DO Cp G δ₁ := by
  intro hδ decI hdI decO hdO x hx B Y C T₀ _ _ _ _ _ _ Tr hH
  obtain ⟨S, hS⟩ := Bridge.Pr_acc_runCM_le H (D.record Bd.IVStar).V
    (Recursion.R (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI))
    (E H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO) x Tr hH hδ ans hans (hV x)
  refine ⟨S, ?_⟩
  intro Ω _ _ P M R'
  have hA := hS Ω P
  have hK : PrBad H (D.record Bd.IVStar).V (Recursion.R (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI))
      (E H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO) x
      (Bridge.simProverM S (D.record Bd.IVStar).V.k P) ≤
      lin2Rhs H D Bd DO Cp decI decO x (Bridge.simProverM S (D.record Bd.IVStar).V.k P) M R' :=
    h decI hdI decO hdO x hx (Ω × S.Tape) (Bridge.simProverM S (D.record Bd.IVStar).V.k P) M R'
  linarith

end Scoped

end Claims.Recursion
