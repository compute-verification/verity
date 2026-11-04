import Definitions.Claims.SamplingGeneric
import Proofs.Flock.Claims.Sampling.Bext
import Proofs.Flock.SamplingGeneric.Fails

/-!
# The sampled compiler's bits

`Bits` from `KS` at any compiled sampled verifier whose extractor's expected reruns and own SHA-512 calls are
`Efficient`'s (`Compiled.bits_of_ks`, `Sampling.Bits_of_KS` generic, through `Sampling.Bits_of_KSLin`). Those two
bounds read no strict one (`ExpOwnOf`, `CompilerEfficient`'s second and third conjuncts), so `CompilerKS` and
`ExpOwnOf` give `CompilerBits` (`compilerBits_of`).
-/

namespace Claims.SamplingGeneric

namespace Compiled

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem κ_pos (L : Compiled Coin A) {δ : ℝ} (hδ : 0 ≤ δ) (x : Msg) : 0 < L.κ δ x := by
  have hs : 0 < Sampling.slack := by unfold Sampling.slack; positivity
  have h1 : 0 ≤ Sampling.slack ^ 2 * (δ + L.κs x) :=
    mul_nonneg (pow_nonneg hs.le 2) (add_nonneg hδ (maxBelow_nonneg _ _))
  have h2 : 0 < (1 + Sampling.slack) * (2 * L.k x / L.cap x + Real.exp (-(L.k x : ℝ) / 8)) :=
    mul_pos (by linarith) (by positivity)
  unfold Compiled.κ
  linarith

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem γ_nonneg (L : Compiled Coin A) (x : Msg) : 0 ≤ L.γ x := by
  have hs : 0 < Sampling.slack := by unfold Sampling.slack; positivity
  unfold Compiled.γ
  exact mul_nonneg (pow_nonneg hs.le 4) (mul_nonneg (Nat.cast_nonneg _) (maxBelow_nonneg _ _))

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem work_pos (L : Compiled Coin A) (e : A.Sh → ℝ) (h : A.Sh → ℕ) (x : Msg) : 0 < L.work e h x := by
  have h1 : 0 ≤ (L.k x : ℝ) ^ 2 * (1 + L.nPU x * L.es e x) :=
    mul_nonneg (sq_nonneg _) (add_nonneg zero_le_one (mul_nonneg (Nat.cast_nonneg _) (maxBelow_nonneg _ _)))
  have h2 : 0 ≤ (L.k x : ℝ) ^ 2 * L.nPU x * (L.openCalls x + L.hs h x) := by positivity
  have h3 : (0 : ℝ) ≤ L.k x := Nat.cast_nonneg _
  unfold Compiled.work
  linarith

theorem bits_of_ks {H : Msg → Msg} {L : Compiled Coin A} {e : A.Sh → ℝ} {h : A.Sh → ℕ} (hks : L.KS H)
    (hruns : ExtractRunsExpLe H L.E (fun x => 1 + L.k x + (L.k x : ℝ) ^ 2 * (1 + L.nPU x * L.es e x)))
    (hown : Sampling.ExtractOwnLe H L.E (fun x => L.k x ^ 2 * L.nPU x * (L.openCalls x + L.hs h x))) :
    L.Bits H e h := fun δ hδ =>
  Sampling.Bits_of_KSLin (hks δ hδ) hruns hown (fun x => L.κ_pos hδ x) L.γ_nonneg (L.work_pos e h)
    fun x => le_of_eq (by unfold Compiled.work; push_cast; ring)

end Compiled

/-- **The compiled extractor's expected reruns and own calls, from the proof system's** (`CompilerEfficient`'s second
and third conjuncts, which read no strict bound on the proof system's reruns). -/
def ExpOwnOf : Prop :=
  ∀ (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (A : Arg Coin) (L : Compiled Coin A)
    (H : Msg → Msg) (e : A.Sh → ℝ) (h : A.Sh → ℕ), L.Sessions H → A.RerunsExpLe H e → A.OwnLe H h →
      L.OpenCallsLe H → ExtractRunsExpLe H L.E (fun x => 1 + L.k x + (L.k x : ℝ) ^ 2 * (1 + L.nPU x * L.es e x)) ∧
        Sampling.ExtractOwnLe H L.E (fun x => L.k x ^ 2 * L.nPU x * (L.openCalls x + L.hs h x))

theorem compilerBits_of (hks : CompilerKS) (heo : ExpOwnOf) : CompilerBits := by
  intro Coin _ _ _ A L H e h hA hB hS hexp hown hopen
  obtain ⟨hr, ho⟩ := heo Coin A L H e h hS hexp hown hopen
  exact Compiled.bits_of_ks (hks Coin A L H hA hB hS) hr ho

end Claims.SamplingGeneric
