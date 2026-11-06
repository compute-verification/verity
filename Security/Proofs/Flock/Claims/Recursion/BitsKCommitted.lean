module

public import Proofs.Flock.Claims.Recursion.BitsLin2
public import Proofs.Flock.Claims.Recursion.KSLin2Committed

/-!
# `BitsKAt` at Hello-committed coins

`BitsKAt` bounds `Pr[acc ∧ ¬R]` by `a + t·e` at coins drawn after each message. Deployed, the coins are committed at
Hello in an hm96 coin tree and each opened with its round's answer, except where that answer is `[]` (`KSLin2OnC`).
`BitsKAtC` is `BitsKAt`'s bound at that run: every hm96 tree whose leaves hide at `δ₁ ≠ ∞` (`Hm96Hiding`) has a
simulator `S` such that, for every prover that reads the root and the openings and whose run with `S`'s root and
openings (`Bridge.simProverM`) meets `BitsKAt`'s hypotheses, the committed run accepts at most as often as the
extractor, run on that simulating prover, recovers a witness, plus `a + t·e` and `2 · k(x) · δ₁`. `S`'s root and
openings are functions of its tape, so the simulating prover's SHA-512 calls are the prover's own.
`bitsKAtC_of_bitsKAt` derives it from `BitsKAt` (`Bridge.Pr_acc_runCM_le`).
-/

@[expose] public section

namespace Claims.Recursion

open Claims
open scoped ENNReal

section Scoped

variable {Coin Col T Col' T' : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
  [DecidableEq Col']

/-- **`BitsKAt` at Hello-committed coins**, at a leaf hiding bound `δ₁ ≠ ∞`: at every statement `G` admits, every inner
and outer decoder meeting the spec and every hm96 coin tree whose leaves hide at `δ₁`, a simulator `S` of the tree such
that, for every prover that reads its root and openings (`Bridge.runCM`), at call bounds `n ≤ N` and `v ≤ V` and under
`BitsKAt`'s hypotheses at the prover with `S`'s root and openings (`Bridge.simProverM`), the committed run's acceptance
is at most the probability that the extractor, run on that prover, recovers a witness, plus `a + t·e` and
`2 · k(x) · δ₁`. -/
def BitsKAtC [Zero Col] [Zero Col'] (H : Msg → Msg) (D : RecAt Coin) (Bd : Bound Coin Col T)
    (DO : Flock.Deploy Coin Col' T') (Cp : Caps Coin) (G : Msg → Prop) (N V : ℕ) (a e : ℝ) (δ₁ : ℝ≥0∞) : Prop :=
  δ₁ ≠ ⊤ → ∀ decI, decOK Bd.deploy decI → ∀ decO, decOK DO decO → ∀ (x : Msg), G x →
    ∀ (Gp Lf Cm T₀ : Type) [AddGroup Gp] [Fintype Gp] [Fintype Lf] [Nonempty Lf] [Fintype T₀] [Nonempty T₀]
      (Tr : Bridge.Hm96Tree Coin Gp Lf Cm T₀ ((D.record Bd.IVStar).V.k x)), Hm96Hiding Tr.My Tr.cs δ₁ →
    ∃ S : Bridge.OnlineSim, ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Bridge.CProver Ω) (t : ℕ), 1 ≤ t →
      (∀ ω r, (run H (Bridge.simProverM S (D.record Bd.IVStar).V.k P) (D.record Bd.IVStar).V x ω r).pLog.length +
        (run H (Bridge.simProverM S (D.record Bd.IVStar).V.k P) (D.record Bd.IVStar).V x ω r).vLog.length ≤ t) →
      ∀ n : ℕ, n ≤ N → (∀ cs ws, ((D.record Bd.IVStar).innerCalls H x cs ws).length ≤ n) →
      ∀ v : ℕ, v ≤ V → (∀ ps, (setupCalls H (D.record Bd.IVStar) ((D.record Bd.IVStar).vstar x ps)).length ≤ v) →
      (∀ hk : 0 < (D.record Bd.IVStar).V.k x, StrictBirthday H
        (B H (D.record Bd.IVStar).V (Bridge.simProverM S (D.record Bd.IVStar).V.k P) x hk)) →
      (∀ hk : 0 < (Inner.ofFlock Bd.deploy decI).V.k (xI (D.record Bd.IVStar) x),
        ExpectedCR H ((BinnerV H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO
          (Bridge.simProverM S (D.record Bd.IVStar).V.k P) x hk).charge
            (setupLen (D.record Bd.IVStar) x + 2 * (n + v)))) →
      (∀ (M : ℕ) (hk : 0 < (Inner.ofFlock Bd.deploy decI).V.k (xI (D.record Bd.IVStar) x)),
        ExpectedCR H ((FdevV H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO
          (Bridge.simProverM S (D.record Bd.IVStar).V.k P) x M hk).charge
            (setupLen (D.record Bd.IVStar) x + 2 * (n + v)))) →
      (∀ (M : ℕ) (hW : 0 < Cp.width (xI (D.record Bd.IVStar) x)),
        ExpectedCR H ((FCI H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO Cp
          (Bridge.simProverM S (D.record Bd.IVStar).V.k P) x M hW).charge
            (setupLen (D.record Bd.IVStar) x + 2 * (n + v)))) →
      (∀ R : ℕ, ExpectedCR H ((F0I H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO
        (Bridge.simProverM S (D.record Bd.IVStar).V.k P) x R).charge (setupLen (D.record Bd.IVStar) x +
          ((Inner.ofFlock Bd.deploy decI).k (xI (D.record Bd.IVStar) x) + 1) * (n + v)))) →
      Pr (fun p : Ω × (Fin ((D.record Bd.IVStar).V.k x) → Coin) × (Fin ((D.record Bd.IVStar).V.k x) → Lf) × T₀ =>
          (Bridge.runCM H P (D.record Bd.IVStar).V x Tr.tree p.1 p.2.1 p.2.2).out = .acc) ≤
        Pr (fun s : (Ω × S.Tape) × (Fin ((D.record Bd.IVStar).V.k x) → Coin) ×
              (E H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO).C x =>
            (Recursion.R (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI)).ext x
              ((E H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO).eval H
                (Bridge.simProverM S (D.record Bd.IVStar).V.k P) x s.1 s.2.1 s.2.2).1) +
          (a + t * e) + 2 * (D.record Bd.IVStar).V.k x * δ₁.toReal

/-- **`BitsKAt` gives `BitsKAtC`** wherever the compiled record's verifier answers each round with its coin, through an
injective `ans`, or with nothing: `Bridge.Pr_acc_runCM_le` at the extractor, with `PrBad` at the simulating prover
bounded by `BitsKAt` there. -/
theorem bitsKAtC_of_bitsKAt [Zero Col] [Zero Col'] {H : Msg → Msg} {D : RecAt Coin} {Bd : Bound Coin Col T}
    {DO : Flock.Deploy Coin Col' T'} {Cp : Caps Coin} {G : Msg → Prop} {N V : ℕ} {a e : ℝ}
    (h : BitsKAt H D Bd DO Cp G N V a e) (ans : Coin → Msg) (hans : Function.Injective ans)
    (hV : ∀ x, Bridge.AnswersCoinOrNil H (D.record Bd.IVStar).V x ans) (δ₁ : ℝ≥0∞) :
    BitsKAtC H D Bd DO Cp G N V a e δ₁ := by
  intro hδ decI hdI decO hdO x hx Gp Lf Cm T₀ _ _ _ _ _ _ Tr hH
  obtain ⟨S, hS⟩ := Bridge.Pr_acc_runCM_le H (D.record Bd.IVStar).V
    (Recursion.R (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI))
    (E H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO) x Tr hH hδ ans hans (hV x)
  refine ⟨S, ?_⟩
  intro Ω _ _ P t ht1 hrun n hnN hn v hvV hv hSB hBI hFdev hFCI hF0I
  have hA := hS Ω P
  have hK := h decI hdI decO hdO x hx (Ω × S.Tape) (Bridge.simProverM S (D.record Bd.IVStar).V.k P) t ht1 hrun
    n hnN hn v hvV hv hSB hBI hFdev hFCI hF0I
  linarith

end Scoped

end Claims.Recursion
