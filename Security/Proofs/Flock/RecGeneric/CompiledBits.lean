module

public import Proofs.Flock.RecGeneric.CompiledCost
public import Proofs.Flock.RecGeneric.CostProof
public import Proofs.Flock.RecGeneric.AtFlockCompiledNontrivial
public import Proofs.Flock.Claims.KS.Flock

/-!
# `KSCompiledOn`'s bits at the Flock instance

`KSCompiledOn`'s bound at the loop bounds `(·, 2^256, 2^257)` and the link's `M = 2^256` is at most `κ + t·ε`
(`ksCompiledOn_bits`), `κ` route 2C's (`compiledAt_κ_eq`) and `ε` five per-call terms, each as `2^{−T}`, at
`c_S = c_Be + 2·n_S + |S(x_I)| + 2(n + v)`. Where every statement a run accepts has `κ ≤ 2^{−167.7}` and
`ε ≤ 2^{−174.95}`, it is at most `2^{−167.7} + t·2^{−174.95}` (`ksCompiledOn_tight`); where no run accepts,
`Pr[acc ∧ ¬R] = 0`.

* A rerun of the composite costs at most `t·(c_Be + 2·n_S)/2 + L` calls on average (`ofRec_Ex_cost_le`): its compiled
  run makes at most `t`, and `E_O`, which makes no call of its own (`Compiled.noHash_EO`), at most `t` per rerun of the
  prover, which are at most `1 + Σ_s (1 + m′_s·k_s)` in expectation (`ofRec_EO_runs_exp`), one more per session than
  `Claims.Recursion.EO`'s; `L` bounds the rest, at most `n_V + w + |S(x_I)| + n + v` (`ofRec_extras_le`).
* No rerun's witnesses fall outside `R_O` (`compiled_PrOut_eq_zero`): every witness `E_O` gives passes Flock's check at
  its statement (`Compiled.eo_some_staged`), which implies Flock's relation there (`Claims.Flock.check_sound`).
* Each run finder of `Comp.err` finds a collision with probability at most its plan's count of reruns times a rerun's
  average cost over `2^256` (`RunFinder.PrAt_le_costLe`): two for `B`'s, `FC`'s, `B_I`'s and the link's plans, `k + 1`
  for `F0`'s.

`B`'s and `B_I`'s terms together are at `255 − log₂((4k_I + 1)·c_S)`, one bit below route 2C's: each of their plans
names two reruns, each charged the calls beyond the compiled run's and `E_O`'s, where route 2C charges `B_I` those
once.
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion
open Claims.SamplingGeneric (Arg)

/-! ## A rerun's cost -/

namespace Compiled

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- **`E_O` makes no SHA-512 call of its own**: it reruns the prover and serves `A`'s extractors, which make none. -/
theorem noHash_EO (L : Compiled Coin A) (x : Msg) (r₀ : Fin (L.V.k x) → Coin) (c : L.EO.C x) :
    Claims.Sampling.NoHash (L.EO.ext x r₀ c) := by
  refine Claims.Sampling.NoHash.rerun r₀ _ fun v₀ => ?_
  split
  · refine Claims.Recursion.noHash_collect _ fun p hp => ?_
    obtain ⟨s, -, rfl⟩ := List.mem_map.1 hp
    unfold Compiled.sessionProg
    split
    · exact Claims.Sampling.NoHash.bind (Claims.Recursion.noHash_serve
        (fun _ => Claims.Recursion.noHash_viewing _ _) (Claims.SamplingGeneric.noHash_ext _ _ _ _))
        fun _ => Claims.Sampling.NoHash.ret _
    · exact Claims.Sampling.NoHash.ret _
  · exact Claims.Sampling.NoHash.ret _

/-- **No rerun's witnesses fall outside `R_O`** where `A`'s check implies its relation: every witness `E_O` gives is
accepted by the check at its statement (`eo_some_staged`). -/
theorem PrOut_eq_zero (L : Compiled Coin A) (hc : ∀ x w, A.check x w = true → A.R x w) (H : Msg → Msg)
    {Ω : Type} [Fintype Ω] (P : Prover Ω) (x : Msg) : L.toComp.PrOut H P x = 0 := by
  unfold Comp.PrOut
  refine (Pr_congr fun t => iff_false_intro ?_).trans Pr_false
  rintro ⟨ws, hws, hn⟩
  obtain ⟨-, -, hs⟩ := L.eo_some_staged H P x t.1 t.2.1 t.2.2 (ws := ws) hws
  refine hn fun p hp => ?_
  obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem hp
  exact hc _ _ (hs i hi).2

/-- `Compiled.noHash_EO` at the composite's `E_O`, which is `Compiled.EO`. -/
theorem toComp_noHash_EO (L : Compiled Coin A) (x : Msg) :
    ∀ r c, Claims.Sampling.NoHash (L.toComp.EO.ext x r c) :=
  L.noHash_EO x

/-- An expected-reruns bound on `Compiled.EO` at the composite's `E_O`, which is `Compiled.EO`. -/
theorem toComp_EO_runs_exp (L : Compiled Coin A) (H : Msg → Msg) {N : Msg → ℝ} (h : ExtractRunsExpLe H L.EO N) :
    ExtractRunsExpLe H L.toComp.EO N :=
  h

end Compiled

section Cost

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **A rerun costs at most `t·(1 + N(x)) + L` calls on average**, where each run of the prover makes at most `t`,
`E_O` makes no call of its own and reruns the prover at most `N(x)` times in expectation, and the rerun's inner view's
run's, searched and own calls are at most `L`. -/
theorem Comp.Ex_rerunCost_le (K : Comp Coin) (IV : Verifier Coin) (H : Msg → Msg) (x : Msg) {Ω : Type}
    (P : Prover Ω) (t : ℕ) (ht : ∀ ω r, (run H P K.V x ω r).pLog.length + (run H P K.V x ω r).vLog.length ≤ t)
    (hNH : ∀ r c, Claims.Sampling.NoHash (K.EO.ext x r c)) (N : Msg → ℝ) (he : ExtractRunsExpLe H K.EO N)
    (L : ℕ)
    (hL : ∀ ω u, (K.viewCalls IV H P x ω u).length + (K.rerunCalls H P x ω u).length + (K.ownAt H P x ω u).length ≤ L)
    (ω : Ω) :
    Ex (fun u : (Fin (K.V.k x) → Coin) × K.EO.C x => (K.rerunCost IV H P x ω u : ℝ)) ≤ t * (1 + N x) + L := by
  have hpt : ∀ u : (Fin (K.V.k x) → Coin) × K.EO.C x,
      (K.rerunCost IV H P x ω u : ℝ) ≤ ((t + L : ℕ) : ℝ) + t * ((K.EO.eval H P x ω u.1 u.2).2.2 : ℝ) := by
    intro u
    have h1 := ht ω u.1
    have h2 : (K.EO.eval H P x ω u.1 u.2).2.1 ≤ t * (K.EO.eval H P x ω u.1 u.2).2.2 :=
      Claims.Sampling.calls_le_of_noHash H (run H P K.V x ω) t (ht ω) (hNH u.1 u.2)
    have h3 := hL ω u
    have h : K.rerunCost IV H P x ω u ≤ (t + L) + t * (K.EO.eval H P x ω u.1 u.2).2.2 := by
      unfold Comp.rerunCost
      omega
    exact_mod_cast h
  have ht0 : (0 : ℝ) ≤ t := Nat.cast_nonneg _
  calc Ex (fun u : (Fin (K.V.k x) → Coin) × K.EO.C x => (K.rerunCost IV H P x ω u : ℝ))
      ≤ Ex (fun u : (Fin (K.V.k x) → Coin) × K.EO.C x =>
          ((t + L : ℕ) : ℝ) + t * ((K.EO.eval H P x ω u.1 u.2).2.2 : ℝ)) := Ex_mono hpt
    _ = ((t + L : ℕ) : ℝ) +
          t * Ex (fun u : (Fin (K.V.k x) → Coin) × K.EO.C x => ((K.EO.eval H P x ω u.1 u.2).2.2 : ℝ)) := by
        rw [Ex_add, Ex_const, Ex_mul_left]
    _ ≤ ((t + L : ℕ) : ℝ) + t * N x := add_le_add le_rfl (mul_le_mul_of_nonneg_left (he x Ω P ω) ht0)
    _ = t * (1 + N x) + L := by push_cast; ring

variable {Col' T' : Type} [DecidableEq Col']

omit [DecidableEq Col'] in
/-- **A rerun's calls beyond its compiled run's and `E_O`'s are at most `n_V + w + |S(x_I)| + n + v`** at a compiled
record: none in the inner view's run or searched where `E_O` gives up; on its witnesses, the inner view's run's at most
`n_V`, and the searched ones the calls the witnesses' second components record at most `w`, the inner setup's and the
inner verifier's at most `n` (`Deploy.callsAt`); and the compiled verifier's own at the run's V\* statements at most
`v`. -/
theorem ofRec_extras_le (H : Msg → Msg) (D : RecAt Coin) (DO : Claims.Flock.Deploy Coin Col' T')
    (IV IV' : Verifier Coin) (x : Msg) {Ω : Type} (P : Prover Ω) (nV w n v : ℕ)
    (hn : ∀ cs ws, ((D.record IV).innerCalls H x cs ws).length ≤ n)
    (hv : ∀ ps, (setupCalls H (D.record IV) ((D.record IV).vstar x ps)).length ≤ v) (ω : Ω)
    (u : (Fin ((Compiled.ofRec D DO IV).toComp.V.k x) → Coin) × (Compiled.ofRec D DO IV).toComp.EO.C x)
    (hV : ∀ ws, (Compiled.ofRec D DO IV).toComp.wits H P x ω u = some ws →
      ((Compiled.ofRec D DO IV).toComp.innerLog IV' H x u.1 ws).length ≤ nV)
    (hW : ∀ ws, (Compiled.ofRec D DO IV).toComp.wits H P x ω u = some ws →
      ((D.record IV).calls x (ws.map Prod.snd)).length ≤ w) :
    ((Compiled.ofRec D DO IV).toComp.viewCalls IV' H P x ω u).length +
        ((Compiled.ofRec D DO IV).toComp.rerunCalls H P x ω u).length +
        ((Compiled.ofRec D DO IV).toComp.ownAt H P x ω u).length ≤
      nV + w + setupLen (D.record IV) x + n + v := by
  have ho : ((Compiled.ofRec D DO IV).toComp.ownAt H P x ω u).length ≤ v := hv _
  rcases hw : (Compiled.ofRec D DO IV).toComp.wits H P x ω u with _ | ws
  · have h1 : (Compiled.ofRec D DO IV).toComp.viewCalls IV' H P x ω u = [] := by
      unfold Comp.viewCalls
      rw [hw]
      rfl
    have h2 : (Compiled.ofRec D DO IV).toComp.rerunCalls H P x ω u = [] := by
      unfold Comp.rerunCalls
      rw [hw]
      rfl
    rw [h1, h2, List.length_nil]
    omega
  · have h1 : (Compiled.ofRec D DO IV).toComp.viewCalls IV' H P x ω u =
        (Compiled.ofRec D DO IV).toComp.innerLog IV' H x u.1 ws := by
      unfold Comp.viewCalls
      rw [hw]
      rfl
    have e : (Compiled.ofRec D DO IV).toComp.rerunCalls H P x ω u =
        (D.record IV).callsAt H x u.1 (ws.map Prod.snd) := by
      unfold Comp.rerunCalls
      rw [hw]
      rfl
    have h2 : ((Compiled.ofRec D DO IV).toComp.rerunCalls H P x ω u).length ≤
        w + (setupLen (D.record IV) x + n) := by
      rw [e]
      unfold Claims.Recursion.Deploy.callsAt
      rw [List.length_append, List.length_map, List.length_append]
      exact Nat.add_le_add (hW ws hw) (Nat.add_le_add_left (hn _ _) _)
    have h3 := hV ws hw
    rw [h1]
    omega

/-- **A rerun at a compiled record costs at most `t·(c_Be + 2·n_S)/2 + L` calls on average**, at every tape, where
each session's V\* statement has the session's numbers (`SessionNums`): `E_O` reruns the prover at most
`1 + Σ_s (1 + m′_s·k_s)` times in expectation (`ofRec_EO_runs_exp`), which is `(c_Be + 2·n_S)/2 − 1`, and makes no
call of its own (`Compiled.noHash_EO`), and `L` bounds the rest (`Comp.Ex_rerunCost_le`). -/
theorem ofRec_Ex_cost_le (H : Msg → Msg) (D : RecAt Coin) (DO : Claims.Flock.Deploy Coin Col' T')
    (IV IV' : Verifier Coin) (hN : SessionNums (D.record IV) DO) (x : Msg) {Ω : Type} (P : Prover Ω) (t : ℕ)
    (hrun : ∀ ω r, (run H P (D.record IV).V x ω r).pLog.length + (run H P (D.record IV).V x ω r).vLog.length ≤ t)
    (L : ℕ)
    (hL : ∀ (ω : Ω)
      (u : (Fin ((Compiled.ofRec D DO IV).toComp.V.k x) → Coin) × (Compiled.ofRec D DO IV).toComp.EO.C x),
      ((Compiled.ofRec D DO IV).toComp.viewCalls IV' H P x ω u).length +
        ((Compiled.ofRec D DO IV).toComp.rerunCalls H P x ω u).length +
        ((Compiled.ofRec D DO IV).toComp.ownAt H P x ω u).length ≤ L) (ω : Ω) :
    Ex (fun u : (Fin ((Compiled.ofRec D DO IV).toComp.V.k x) → Coin) × (Compiled.ofRec D DO IV).toComp.EO.C x =>
      ((Compiled.ofRec D DO IV).toComp.rerunCost IV' H P x ω u : ℝ)) ≤
      t * ((cBe (D.record IV) x : ℝ) + 2 * (D.record IV).nS x) / 2 + L := by
  refine (Comp.Ex_rerunCost_le (Compiled.ofRec D DO IV).toComp IV' H x P t hrun
    (Compiled.toComp_noHash_EO (Compiled.ofRec D DO IV) x) _
    (Compiled.toComp_EO_runs_exp (Compiled.ofRec D DO IV) H (ofRec_EO_runs_exp H D DO IV hN)) L hL ω).trans
    (le_of_eq ?_)
  unfold Claims.Recursion.cBe
  push_cast
  simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one]
  ring

end Cost

/-! ## The terms -/

section Terms

/-- **`B`'s term is at most `t·2^{−(255.5 − ½ log₂ γ)}`** where `Pr[B]` is within the birthday bound at `2t` calls. -/
theorem birthday_term_le {γ PB t : ℝ} (hγ : 0 ≤ γ) (ht : 0 ≤ t) (hB : PB ≤ (2 * t) ^ 2 / 2 ^ (513 : ℝ)) :
    Real.sqrt (γ * PB) ≤ t * (2 : ℝ) ^ (-(255.5 - Real.logb 2 γ / 2)) :=
  calc Real.sqrt (γ * PB) ≤ Real.sqrt (γ * ((2 * t) ^ 2 / 2 ^ (513 : ℝ))) :=
        Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left hB hγ)
    _ = t * (Real.sqrt γ * 2 ^ (-(255.5 : ℝ))) := Claims.Sampling.sqrt_birthday hγ ht
    _ ≤ _ := mul_le_mul_of_nonneg_left (Claims.Recursion.sqrt_mul_two_rpow_le hγ) ht

/-- **The two fork finders' terms are at most `t·2^{−(255 − log₂((4k + 1)·c_S))}`**, each finding a collision with
probability at most `2c/2^256`, at `2c + 1 ≤ 2t·c_S`. -/
theorem fork_terms_le {k P PI c cS t : ℝ} (hk : 0 ≤ k) (ht : 0 ≤ t) (hP : P ≤ 2 * c / 2 ^ (256 : ℝ))
    (hPI : PI ≤ 2 * c / 2 ^ (256 : ℝ)) (hc : 2 * c + 1 ≤ 2 * t * cS) (hc0 : 0 ≤ c) :
    4 * k * (P + 0) + PI ≤ t * (2 : ℝ) ^ (-(255 - Real.logb 2 ((4 * k + 1) * cS))) := by
  have hp256 : (0 : ℝ) < 2 ^ (256 : ℝ) := by positivity
  have h256 : (2 : ℝ) ^ (256 : ℝ) = 2 * 2 ^ (255 : ℝ) := by
    rw [show (256 : ℝ) = 1 + 255 by norm_num, Real.rpow_add (by norm_num), Real.rpow_one]
  have hcS : 0 ≤ cS := by nlinarith
  have hk4 : 0 ≤ 4 * k + 1 := by linarith
  calc 4 * k * (P + 0) + PI ≤ 4 * k * (2 * c / 2 ^ (256 : ℝ)) + 2 * c / 2 ^ (256 : ℝ) := by
        rw [add_zero]
        exact add_le_add (mul_le_mul_of_nonneg_left hP (by linarith)) hPI
    _ = (4 * k + 1) * (2 * c) / 2 ^ (256 : ℝ) := by ring
    _ ≤ (4 * k + 1) * (2 * t * cS) / 2 ^ (256 : ℝ) :=
        div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left (by linarith) hk4) hp256.le
    _ = t * ((4 * k + 1) * cS) / 2 ^ (255 : ℝ) := by rw [h256]; ring
    _ ≤ _ := Claims.Recursion.le_two_rpow_of_div ht (mul_nonneg hk4 hcS)

/-- **A two-rerun finder's term with its tail `1/(2^256 + 1)` is at most `t·2^{−(255 − log₂(w·c_S))}`** at the
weight `w`, the finder finding a collision with probability at most `2c/2^256`, at `2c + 1 ≤ 2t·c_S`. -/
theorem two_run_term_le {w P c cS t : ℝ} (hw : 0 ≤ w) (ht : 0 ≤ t) (hP : P ≤ 2 * c / 2 ^ (256 : ℝ))
    (hc : 2 * c + 1 ≤ 2 * t * cS) (hc0 : 0 ≤ c) :
    w * (P + 1 / (((2 ^ 256 : ℕ) : ℝ) + 1)) ≤ t * (2 : ℝ) ^ (-(255 - Real.logb 2 (w * cS))) := by
  have hp256 : (0 : ℝ) < 2 ^ (256 : ℝ) := by positivity
  have h256 : (2 : ℝ) ^ (256 : ℝ) = 2 * 2 ^ (255 : ℝ) := by
    rw [show (256 : ℝ) = 1 + 255 by norm_num, Real.rpow_add (by norm_num), Real.rpow_one]
  have hMc : (((2 ^ 256 : ℕ) : ℝ)) = (2 : ℝ) ^ (256 : ℝ) := by
    rw [show (256 : ℝ) = ((256 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
    norm_num
  have hM : 1 / (((2 ^ 256 : ℕ) : ℝ) + 1) ≤ 1 / 2 ^ (256 : ℝ) := by
    rw [hMc]
    exact one_div_le_one_div_of_le hp256 (by linarith)
  have hcS : 0 ≤ cS := by nlinarith
  calc w * (P + 1 / (((2 ^ 256 : ℕ) : ℝ) + 1)) ≤ w * (2 * c / 2 ^ (256 : ℝ) + 1 / 2 ^ (256 : ℝ)) :=
        mul_le_mul_of_nonneg_left (add_le_add hP hM) hw
    _ = w * (2 * c + 1) / 2 ^ (256 : ℝ) := by ring
    _ ≤ w * (2 * t * cS) / 2 ^ (256 : ℝ) :=
        div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left hc hw) hp256.le
    _ = t * (w * cS) / 2 ^ (255 : ℝ) := by rw [h256]; ring
    _ ≤ _ := Claims.Recursion.le_two_rpow_of_div ht (mul_nonneg hw hcS)

/-- **`F0`'s term with its tail `(k + 1)/(2^257 + 1)` is at most `t·2^{−(256 − log₂(4(k + 1)·c_S))}`**, its finder
finding a collision with probability at most `(k + 1)·c/2^256`, at `2c + 1 ≤ 2t·c_S`. -/
theorem conflict_term_le {k P c cS t : ℝ} (hk : 0 ≤ k) (ht : 0 ≤ t) (hP : P ≤ (k + 1) * c / 2 ^ (256 : ℝ))
    (hc : 2 * c + 1 ≤ 2 * t * cS) (hc0 : 0 ≤ c) :
    4 * (P + (k + 1) / (((2 ^ 257 : ℕ) : ℝ) + 1)) ≤
      t * (2 : ℝ) ^ (-(256 - Real.logb 2 (4 * (k + 1) * cS))) := by
  have hp257 : (0 : ℝ) < 2 ^ (257 : ℝ) := by positivity
  have h257 : (2 : ℝ) ^ (257 : ℝ) = 2 * 2 ^ (256 : ℝ) := by
    rw [show (257 : ℝ) = 1 + 256 by norm_num, Real.rpow_add (by norm_num), Real.rpow_one]
  have hRc : (((2 ^ 257 : ℕ) : ℝ)) = (2 : ℝ) ^ (257 : ℝ) := by
    rw [show (257 : ℝ) = ((257 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
    norm_num
  have hk1 : 0 ≤ k + 1 := by linarith
  have hcS : 0 ≤ cS := by nlinarith
  have hR : (k + 1) / (((2 ^ 257 : ℕ) : ℝ) + 1) ≤ (k + 1) / 2 ^ (257 : ℝ) := by
    rw [hRc]
    exact div_le_div_of_nonneg_left hk1 hp257 (by linarith)
  calc 4 * (P + (k + 1) / (((2 ^ 257 : ℕ) : ℝ) + 1))
      ≤ 4 * ((k + 1) * c / 2 ^ (256 : ℝ) + (k + 1) / 2 ^ (257 : ℝ)) :=
        mul_le_mul_of_nonneg_left (add_le_add hP hR) (by norm_num)
    _ = 4 * (k + 1) * (2 * c + 1) / 2 ^ (257 : ℝ) := by rw [h257]; ring
    _ ≤ 4 * (k + 1) * (2 * t * cS) / 2 ^ (257 : ℝ) :=
        div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left hc (by linarith)) hp257.le
    _ = t * (4 * (k + 1) * cS) / 2 ^ (256 : ℝ) := by rw [h257]; ring
    _ ≤ _ := Claims.Recursion.le_two_rpow_of_div ht (mul_nonneg (by linarith) hcS)

end Terms

/-! ## `Comp.err` at the element-level Flock -/

section Err

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **`Comp.err` at the element-level Flock, written out**: its three inner terms at the loop bounds `(·, M, R)`. -/
theorem Comp.err_ofFlock_eq (K : Comp Coin) (Bd : Bound Coin Col T) (decI : List (ℕ × Col) → List T)
    (Cp : Caps Coin) (S : Server) (H : Msg → Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg)
    (M R : ℕ) :
    K.err (Proto.ofFlock Bd decI Cp) S H P x (![0, M, R] : Fin 3 → ℕ) M =
      K.κ (Proto.ofFlock Bd decI Cp) x + Real.sqrt (K.γ (Proto.ofFlock Bd decI Cp) x * PrB H K.V P x) +
      (4 * (Bd.DE.V.k (K.xI x) : ℝ) *
          ((forkPlan (Coin := Coin) Bd.DE.V.k).PrAt H (K.innerModel (S.IVStar Bd.DE.V Bd.ans) H P x) (K.xI x) + 0) +
        4 * (Cp.width (K.xI x) : ℝ) *
          ((capPlan Cp M).PrAt H (K.innerModel (S.IVStar Bd.DE.V Bd.ans) H P x) (K.xI x) + 1 / ((M : ℝ) + 1)) +
        4 * ((conflictPlan (Coin := Coin) Bd.DE.k R).PrAt H (K.innerModel (S.IVStar Bd.DE.V Bd.ans) H P x)
            (K.xI x) + ((Bd.DE.k (K.xI x) : ℝ) + 1) / ((R : ℝ) + 1))) +
      K.reads (Proto.ofFlock Bd decI Cp) x * K.PrOut H P x +
      (forkPlan (Coin := Coin) Bd.DE.V.k).PrAt H (K.verModel (S.IVStar Bd.DE.V Bd.ans) H P x) (K.xI x) +
      K.β (Proto.ofFlock Bd decI Cp) x *
        ((linkPlan (Coin := Coin) Bd.DE.V.k M).PrAt H (K.verModel (S.IVStar Bd.DE.V Bd.ans) H P x) (K.xI x) +
          1 / ((M : ℝ) + 1) + K.PrOut H P x) := by
  show K.κ (Proto.ofFlock Bd decI Cp) x + Real.sqrt (K.γ (Proto.ofFlock Bd decI Cp) x * PrB H K.V P x) +
      ∑ i : Fin 3, (![termB Bd.DE.V.k, termFC Cp, termF0 Bd.DE.k] i).w (K.xI x) *
        (((![termB Bd.DE.V.k, termFC Cp, termF0 Bd.DE.k] i).F (![0, M, R] i)).PrAt H
            (K.innerModel (S.IVStar Bd.DE.V Bd.ans) H P x) (K.xI x) +
          (![termB Bd.DE.V.k, termFC Cp, termF0 Bd.DE.k] i).tail (K.xI x) (![0, M, R] i)) +
      K.reads (Proto.ofFlock Bd decI Cp) x * K.PrOut H P x +
      (forkPlan (Coin := Coin) Bd.DE.V.k).PrAt H (K.verModel (S.IVStar Bd.DE.V Bd.ans) H P x) (K.xI x) +
      K.β (Proto.ofFlock Bd decI Cp) x *
        ((linkPlan (Coin := Coin) Bd.DE.V.k M).PrAt H (K.verModel (S.IVStar Bd.DE.V Bd.ans) H P x) (K.xI x) +
          1 / ((M : ℝ) + 1) + K.PrOut H P x) = _
  rw [Fin.sum_univ_three]
  rfl

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- **The deviation finder's weight at the element-level Flock**: `β = (N_I + 8W + 4(k + 1))·k_I`, the terms' reads
`4W·2 + 4·(k + 1)`. -/
theorem Comp.β_ofFlock_eq (K : Comp Coin) (Bd : Bound Coin Col T) (decI : List (ℕ × Col) → List T)
    (Cp : Caps Coin) (x : Msg) :
    K.β (Proto.ofFlock Bd decI Cp) x =
      ((K.NI (Proto.ofFlock Bd decI Cp) x : ℝ) + 8 * Cp.width (K.xI x) + 4 * ((Bd.DE.k (K.xI x) : ℝ) + 1)) *
        Bd.DE.V.k (K.xI x) := by
  show ((K.NI (Proto.ofFlock Bd decI Cp) x : ℝ) +
      ∑ i : Fin 3, (![termB Bd.DE.V.k, termFC Cp, termF0 Bd.DE.k] i).w (K.xI x) *
        (((![termB Bd.DE.V.k, termFC Cp, termF0 Bd.DE.k] i).d (K.xI x) : ℕ) : ℝ)) * Bd.DE.V.k (K.xI x) = _
  rw [Fin.sum_univ_three]
  show ((K.NI (Proto.ofFlock Bd decI Cp) x : ℝ) +
      (4 * (Bd.DE.V.k (K.xI x) : ℝ) * ((0 : ℕ) : ℝ) + 4 * (Cp.width (K.xI x) : ℝ) * ((2 : ℕ) : ℝ) +
        4 * ((Bd.DE.k (K.xI x) + 1 : ℕ) : ℝ))) * Bd.DE.V.k (K.xI x) = _
  push_cast
  ring

/-- **`Comp.err` at the element-level Flock, at the loop bounds `(·, 2^256, 2^257)` and `M = 2^256`, is at most
`κ + t·ε`**, `ε` five per-call terms at `c_S`, where no rerun's witnesses fall outside `R_O`, a rerun costs at most `c`
calls on average with `2c + 1 ≤ 2t·c_S`, and SHA-512's named collision assumptions hold at `B` (`StrictBirthday`) and
at the five run finders (`ExpectedCR`): each finds a collision with probability at most its plan's count of reruns
times `c` over `2^256` (`RunFinder.PrAt_le_costLe`). -/
theorem Comp.err_ofFlock_le (K : Comp Coin) (Bd : Bound Coin Col T) (decI : List (ℕ × Col) → List T)
    (Cp : Caps Coin) (S : Server) (H : Msg → Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg)
    (t : ℕ) (hrun : ∀ ω r, (run H P K.V x ω r).pLog.length + (run H P K.V x ω r).vLog.length ≤ t)
    (hγO : 0 ≤ K.γO x) (hPO : K.PrOut H P x = 0) (c cS : ℝ)
    (hrr : ∀ ω, Ex (fun u : (Fin (K.V.k x) → Coin) × K.EO.C x =>
      (K.rerunCost (S.IVStar Bd.DE.V Bd.ans) H P x ω u : ℝ)) ≤ c)
    (hc : 2 * c + 1 ≤ 2 * t * cS)
    (hSB : ∀ hk : 0 < K.V.k x, StrictBirthday H (B H K.V P x hk))
    (hB : ∀ h : 0 < (forkPlan (Coin := Coin) Bd.DE.V.k).sel (K.xI x),
      ExpectedCR H ((forkPlan (Coin := Coin) Bd.DE.V.k).finder (K.innerModel (S.IVStar Bd.DE.V Bd.ans) H P x)
        (K.xI x) h))
    (hFC : ∀ h : 0 < (capPlan Cp (2 ^ 256)).sel (K.xI x),
      ExpectedCR H ((capPlan Cp (2 ^ 256)).finder (K.innerModel (S.IVStar Bd.DE.V Bd.ans) H P x) (K.xI x) h))
    (hF0 : ∀ h : 0 < (conflictPlan (Coin := Coin) Bd.DE.k (2 ^ 257)).sel (K.xI x),
      ExpectedCR H ((conflictPlan (Coin := Coin) Bd.DE.k (2 ^ 257)).finder
        (K.innerModel (S.IVStar Bd.DE.V Bd.ans) H P x) (K.xI x) h))
    (hBI : ∀ h : 0 < (forkPlan (Coin := Coin) Bd.DE.V.k).sel (K.xI x),
      ExpectedCR H ((forkPlan (Coin := Coin) Bd.DE.V.k).finder (K.verModel (S.IVStar Bd.DE.V Bd.ans) H P x)
        (K.xI x) h))
    (hLink : ∀ h : 0 < (linkPlan (Coin := Coin) Bd.DE.V.k (2 ^ 256)).sel (K.xI x),
      ExpectedCR H ((linkPlan (Coin := Coin) Bd.DE.V.k (2 ^ 256)).finder
        (K.verModel (S.IVStar Bd.DE.V Bd.ans) H P x) (K.xI x) h)) :
    K.err (Proto.ofFlock Bd decI Cp) S H P x (![0, 2 ^ 256, 2 ^ 257] : Fin 3 → ℕ) (2 ^ 256) ≤
      K.κ (Proto.ofFlock Bd decI Cp) x +
        t * ((2 : ℝ) ^ (-(255.5 - Real.logb 2 (K.γ (Proto.ofFlock Bd decI Cp) x) / 2)) +
          (2 : ℝ) ^ (-(255 - Real.logb 2 ((4 * Bd.DE.V.k (K.xI x) + 1) * cS))) +
          (2 : ℝ) ^ (-(255 - Real.logb 2 (4 * Cp.width (K.xI x) * cS))) +
          (2 : ℝ) ^ (-(256 - Real.logb 2 (4 * ((Bd.DE.k (K.xI x) : ℝ) + 1) * cS))) +
          (2 : ℝ) ^ (-(255 - Real.logb 2 (K.β (Proto.ofFlock Bd decI Cp) x * cS)))) := by
  rw [Comp.err_ofFlock_eq, hPO]
  have ht0 : (0 : ℝ) ≤ t := Nat.cast_nonneg _
  have hc0 : 0 ≤ c := (Ex_nonneg fun _ => Nat.cast_nonneg _).trans (hrr (Classical.arbitrary Ω))
  have hγ0 : 0 ≤ K.γ (Proto.ofFlock Bd decI Cp) x := by
    unfold Comp.γ
    exact mul_nonneg (sq_nonneg _) hγO
  have hβ0 : 0 ≤ K.β (Proto.ofFlock Bd decI Cp) x := by
    rw [Comp.β_ofFlock_eq]
    positivity
  have hPB : PrB H K.V P x ≤ (2 * (t : ℝ)) ^ 2 / 2 ^ (513 : ℝ) := by
    unfold PrB
    split
    · rename_i hk
      refine hSB hk (2 * t) fun s => ?_
      exact_mod_cast B_cost_le H K.V P x hk t hrun s
    · positivity
  have h1 := RunFinder.PrAt_le_costLe H _ _ (forkPlanCost (Coin := Coin) Bd.DE.V.k)
    (K.innerModel (S.IVStar Bd.DE.V Bd.ans) H P x) (K.xI x) c hrr hB
  have h2 := RunFinder.PrAt_le_costLe H _ _ (capPlanCost (Coin := Coin) Cp (2 ^ 256))
    (K.innerModel (S.IVStar Bd.DE.V Bd.ans) H P x) (K.xI x) c hrr hFC
  have h3 := RunFinder.PrAt_le_costLe H _ _ (conflictPlanCost (Coin := Coin) Bd.DE.k (2 ^ 257))
    (K.innerModel (S.IVStar Bd.DE.V Bd.ans) H P x) (K.xI x) c hrr hF0
  have h4 := RunFinder.PrAt_le_costLe H _ _ (forkPlanCost (Coin := Coin) Bd.DE.V.k)
    (K.verModel (S.IVStar Bd.DE.V Bd.ans) H P x) (K.xI x) c hrr hBI
  have h5 := RunFinder.PrAt_le_costLe H _ _ (linkPlanCost (Coin := Coin) Bd.DE.V.k (2 ^ 256))
    (K.verModel (S.IVStar Bd.DE.V Bd.ans) H P x) (K.xI x) c hrr hLink
  simp only [Nat.cast_ofNat, Nat.cast_add, Nat.cast_one] at h1 h2 h3 h4 h5
  have e1 := birthday_term_le hγ0 ht0 hPB
  have e2 := fork_terms_le (k := (Bd.DE.V.k (K.xI x) : ℝ)) (Nat.cast_nonneg _) ht0 h1 h4 hc hc0
  have e3 := two_run_term_le (w := 4 * (Cp.width (K.xI x) : ℝ)) (by positivity) ht0 h2 hc hc0
  have e4 := conflict_term_le (k := (Bd.DE.k (K.xI x) : ℝ)) (Nat.cast_nonneg _) ht0 h3 hc hc0
  have e5 := two_run_term_le hβ0 ht0 h5 hc hc0
  linarith

end Err

/-! ## At today's compiled record -/

section Instance

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] {Col' T' : Type}
  [DecidableEq Col']

/-- **The composite's `κ` at `compiledAt` is route 2C's** at the compiled record over `Open(DE, hm96)`
(`Compiled.κ_ofFlock`, `κ_ofFlock_deploy`). -/
theorem compiledAt_κ_eq (D : RecAt Coin) (Bd : Bound Coin Col T) (DO : Claims.Flock.Deploy Coin Col' T')
    (Cp : Caps Coin) (x : Msg) :
    (compiledAt D Bd DO Cp).toComp.κ (Proto.ofFlock Bd Bd.deploy.decode Cp) x =
      Claims.Recursion.κ (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans))
        (Inner.ofFlock Bd.deploy Bd.deploy.decode) x := by
  rw [Compiled.κ_ofFlock, κ_ofFlock_deploy]
  rfl

omit [DecidableEq Col] [DecidableEq Col'] in
/-- **The composite's `B` weight at `compiledAt` is route 2C's**: `N_I²·(Σ_s √(γ_s·k))² = (N_I·Σ_s √(γ_s·k))²`. -/
theorem compiledAt_γ_eq (D : RecAt Coin) (Bd : Bound Coin Col T) (DO : Claims.Flock.Deploy Coin Col' T')
    (Cp : Caps Coin) (x : Msg) :
    (compiledAt D Bd DO Cp).toComp.γ (Proto.ofFlock Bd Bd.deploy.decode Cp) x =
      Claims.Recursion.γ (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans))
        (Inner.ofFlock Bd.deploy Bd.deploy.decode) x := by
  unfold Comp.γ Claims.Recursion.γ
  rw [mul_pow]
  rfl

omit [DecidableEq Col] [DecidableEq Col'] in
/-- **The deviation finder's weight at `compiledAt` is route 2C's `β₂`** (`Comp.β_ofFlock_eq`). -/
theorem compiledAt_β_eq (D : RecAt Coin) (Bd : Bound Coin Col T) (DO : Claims.Flock.Deploy Coin Col' T')
    (Cp : Caps Coin) (x : Msg) :
    (compiledAt D Bd DO Cp).toComp.β (Proto.ofFlock Bd Bd.deploy.decode Cp) x =
      βdev2 (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)) (Inner.ofFlock Bd.deploy Bd.deploy.decode) Cp x := by
  rw [Comp.β_ofFlock_eq]
  rfl

omit [DecidableEq Col] [DecidableEq Col'] in
/-- **No rerun's witnesses fall outside `R_O` at `compiledAt`**: Flock's check implies its relation
(`Claims.Flock.check_sound`, `Compiled.PrOut_eq_zero`). -/
theorem compiled_PrOut_eq_zero (D : RecAt Coin) (Bd : Bound Coin Col T) (DO : Claims.Flock.Deploy Coin Col' T')
    (Cp : Caps Coin) (H : Msg → Msg) {Ω : Type} [Fintype Ω] (P : Prover Ω) (x : Msg) :
    (compiledAt D Bd DO Cp).toComp.PrOut H P x = 0 :=
  Compiled.PrOut_eq_zero _ (Claims.Flock.check_sound DO.parse) H P x

/-- **`KSCompiledOn`'s bound at the loop bounds `(·, 2^256, 2^257)` and `M = 2^256` is at most `κ + t·ε`** at a
statement `G` admits, `κ` route 2C's and `ε` five per-call terms at `c_S = c_Be + 2·n_S + |S(x_I)| + 2(n + v)`
(`Comp.err_ofFlock_le`), where each session's V\* statement has the session's numbers (`SessionNums`), under SHA-512's
named collision assumptions at `B` and at the five run finders over the composite's reruns. `n` and `v` are route
2C's call bounds, the inner verifier's calls on the inner messages (`hn`) and the compiled verifier's own at the V\*
statements (`hv`); a rerun's inner view's run makes at most `n` (`hV`) and the calls `E_O`'s witnesses record number
at most `v` (`hW`), so a rerun costs at most `t·(c_Be + 2·n_S)/2 + |S(x_I)| + 2(n + v)` calls on average
(`ofRec_Ex_cost_le`, `ofRec_extras_le`). -/
theorem ksCompiledOn_bits {H : Msg → Msg} {D : RecAt Coin} {Bd : Bound Coin Col T}
    {DO : Claims.Flock.Deploy Coin Col' T'} {Cp : Caps Coin} {G : Msg → Prop} (hK : KSCompiledOn H D Bd DO Cp G)
    (hN : SessionNums (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)) DO) {x : Msg} (hx : G x) {Ω : Type}
    [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (t : ℕ) (ht1 : 1 ≤ t)
    (hrun : ∀ ω r, (run H P (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)).V x ω r).pLog.length +
      (run H P (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)).V x ω r).vLog.length ≤ t) (n v : ℕ)
    (hn : ∀ cs ws, ((D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)).innerCalls H x cs ws).length ≤ n)
    (hv : ∀ ps, (setupCalls H (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans))
      ((D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)).vstar x ps)).length ≤ v)
    (hV : ∀ (r : Fin ((compiledAt D Bd DO Cp).toComp.V.k x) → Coin) ws,
      ((compiledAt D Bd DO Cp).toComp.innerLog ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans) H x r ws).length ≤ n)
    (hW : ∀ (ω : Ω)
      (u : (Fin ((compiledAt D Bd DO Cp).toComp.V.k x) → Coin) × (compiledAt D Bd DO Cp).toComp.EO.C x) ws,
      (compiledAt D Bd DO Cp).toComp.wits H P x ω u = some ws →
        ((D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)).calls x (ws.map Prod.snd)).length ≤ v)
    (hSB : ∀ hk : 0 < (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)).V.k x,
      StrictBirthday H (B H (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)).V P x hk))
    (hB : ∀ h : 0 < (forkPlan (Coin := Coin) Bd.DE.V.k).sel ((compiledAt D Bd DO Cp).toComp.xI x),
      ExpectedCR H ((forkPlan (Coin := Coin) Bd.DE.V.k).finder
        ((compiledAt D Bd DO Cp).toComp.innerModel ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans) H P x)
        ((compiledAt D Bd DO Cp).toComp.xI x) h))
    (hFC : ∀ h : 0 < (capPlan Cp (2 ^ 256)).sel ((compiledAt D Bd DO Cp).toComp.xI x),
      ExpectedCR H ((capPlan Cp (2 ^ 256)).finder
        ((compiledAt D Bd DO Cp).toComp.innerModel ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans) H P x)
        ((compiledAt D Bd DO Cp).toComp.xI x) h))
    (hF0 : ∀ h : 0 < (conflictPlan (Coin := Coin) Bd.DE.k (2 ^ 257)).sel ((compiledAt D Bd DO Cp).toComp.xI x),
      ExpectedCR H ((conflictPlan (Coin := Coin) Bd.DE.k (2 ^ 257)).finder
        ((compiledAt D Bd DO Cp).toComp.innerModel ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans) H P x)
        ((compiledAt D Bd DO Cp).toComp.xI x) h))
    (hBI : ∀ h : 0 < (forkPlan (Coin := Coin) Bd.DE.V.k).sel ((compiledAt D Bd DO Cp).toComp.xI x),
      ExpectedCR H ((forkPlan (Coin := Coin) Bd.DE.V.k).finder
        ((compiledAt D Bd DO Cp).toComp.verModel ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans) H P x)
        ((compiledAt D Bd DO Cp).toComp.xI x) h))
    (hLink : ∀ h : 0 < (linkPlan (Coin := Coin) Bd.DE.V.k (2 ^ 256)).sel ((compiledAt D Bd DO Cp).toComp.xI x),
      ExpectedCR H ((linkPlan (Coin := Coin) Bd.DE.V.k (2 ^ 256)).finder
        ((compiledAt D Bd DO Cp).toComp.verModel ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans) H P x)
        ((compiledAt D Bd DO Cp).toComp.xI x) h)) :
    PrBad H (compiledAt D Bd DO Cp).V ((compiledAt D Bd DO Cp).toComp.R (Proto.ofFlock Bd Bd.deploy.decode Cp))
        ((compiledAt D Bd DO Cp).toComp.E (Proto.ofFlock Bd Bd.deploy.decode Cp) (Server.ofBound Bd)) x P ≤
      Claims.Recursion.κ (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans))
          (Inner.ofFlock Bd.deploy Bd.deploy.decode) x +
        t * ((2 : ℝ) ^ (-(255.5 - Real.logb 2 (Claims.Recursion.γ (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans))
              (Inner.ofFlock Bd.deploy Bd.deploy.decode) x) / 2)) +
          (2 : ℝ) ^ (-(255 - Real.logb 2 ((4 * (Inner.ofFlock Bd.deploy Bd.deploy.decode).V.k
              (xI (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)) x) + 1) *
            (cBe (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)) x +
              2 * (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)).nS x +
              setupLen (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)) x + 2 * ((n + v : ℕ) : ℝ))))) +
          (2 : ℝ) ^ (-(255 - Real.logb 2 (4 * Cp.width (xI (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)) x) *
            (cBe (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)) x +
              2 * (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)).nS x +
              setupLen (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)) x + 2 * ((n + v : ℕ) : ℝ))))) +
          (2 : ℝ) ^ (-(256 - Real.logb 2 (4 * (((Inner.ofFlock Bd.deploy Bd.deploy.decode).k
              (xI (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)) x) : ℝ) + 1) *
            (cBe (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)) x +
              2 * (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)).nS x +
              setupLen (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)) x + 2 * ((n + v : ℕ) : ℝ))))) +
          (2 : ℝ) ^ (-(255 - Real.logb 2 (βdev2 (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans))
              (Inner.ofFlock Bd.deploy Bd.deploy.decode) Cp x *
            (cBe (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)) x +
              2 * (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)).nS x +
              setupLen (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)) x + 2 * ((n + v : ℕ) : ℝ)))))) := by
  have ht1' : (1 : ℝ) ≤ t := by exact_mod_cast ht1
  have hcB : (4 : ℝ) ≤ cBe (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)) x := by
    have : 4 ≤ cBe (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)) x := by
      unfold Claims.Recursion.cBe
      omega
    exact_mod_cast this
  have p1 : 0 ≤ ((t : ℝ) - 1) * cBe (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)) x :=
    mul_nonneg (sub_nonneg.2 ht1') (Nat.cast_nonneg _)
  have p2 : 0 ≤ (t : ℝ) * (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)).nS x :=
    mul_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)
  have p3 : 0 ≤ ((t : ℝ) - 1) * setupLen (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)) x :=
    mul_nonneg (sub_nonneg.2 ht1') (Nat.cast_nonneg _)
  have p4 : 0 ≤ ((t : ℝ) - 1) * n := mul_nonneg (sub_nonneg.2 ht1') (Nat.cast_nonneg _)
  have p5 : 0 ≤ ((t : ℝ) - 1) * v := mul_nonneg (sub_nonneg.2 ht1') (Nat.cast_nonneg _)
  have hc : 2 * (t * ((cBe (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)) x : ℝ) +
        2 * (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)).nS x) / 2 +
        ((n + v + setupLen (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)) x + n + v : ℕ) : ℝ)) + 1 ≤
      2 * t * ((cBe (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)) x : ℝ) +
        2 * (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)).nS x +
        setupLen (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)) x + 2 * ((n + v : ℕ) : ℝ)) := by
    push_cast
    linarith
  rw [← compiledAt_κ_eq D Bd DO Cp x, ← compiledAt_γ_eq D Bd DO Cp x, ← compiledAt_β_eq D Bd DO Cp x]
  exact (hK x hx Ω P (![0, 2 ^ 256, 2 ^ 257] : Fin 3 → ℕ) (2 ^ 256)).trans
    (Comp.err_ofFlock_le (compiledAt D Bd DO Cp).toComp Bd Bd.deploy.decode Cp (Server.ofBound Bd) H P x t hrun
      (by unfold Compiled.toComp; exact sq_nonneg _) (compiled_PrOut_eq_zero D Bd DO Cp H P x)
      (t * ((cBe (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)) x : ℝ) +
        2 * (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)).nS x) / 2 +
        ((n + v + setupLen (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)) x + n + v : ℕ) : ℝ))
      ((cBe (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)) x : ℝ) +
        2 * (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)).nS x +
        setupLen (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)) x + 2 * ((n + v : ℕ) : ℝ))
      (fun ω => ofRec_Ex_cost_le H D DO _ _ hN x P t hrun _
        (fun ω u => ofRec_extras_le H D DO _ _ x P n v n v hn hv ω u (fun ws _ => hV u.1 ws) (hW ω u)) ω)
      hc hSB hB hFC hF0 hBI hLink)

/-- **`Pr[acc ∧ ¬R] ≤ 2^{−167.7} + t·2^{−174.95}` at every statement** at `compiledAt`, where every statement a run
of the compiled verifier accepts is one `G` admits with `κ ≤ 2^{−167.7}` and `ε ≤ 2^{−174.95}` (`hb`):
`ksCompiledOn_bits` there, and where no run accepts, `Pr[acc ∧ ¬R] = 0`. So `w ≤ v` (`hW`) is needed only where a
run accepts. -/
theorem ksCompiledOn_tight {H : Msg → Msg} {D : RecAt Coin} {Bd : Bound Coin Col T}
    {DO : Claims.Flock.Deploy Coin Col' T'} {Cp : Caps Coin} {G : Msg → Prop} (hK : KSCompiledOn H D Bd DO Cp G)
    (hN : SessionNums (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)) DO) (x : Msg) {Ω : Type}
    [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (t : ℕ) (ht1 : 1 ≤ t)
    (hrun : ∀ ω r, (run H P (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)).V x ω r).pLog.length +
      (run H P (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)).V x ω r).vLog.length ≤ t) (n v : ℕ)
    (hb : (∃ ω r, (run H P (compiledAt D Bd DO Cp).V x ω r).out = .acc) → G x ∧
      Claims.Recursion.κ (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans))
          (Inner.ofFlock Bd.deploy Bd.deploy.decode) x ≤ 2 ^ (-(167.7 : ℝ)) ∧
      (2 : ℝ) ^ (-(255.5 - Real.logb 2 (Claims.Recursion.γ (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans))
              (Inner.ofFlock Bd.deploy Bd.deploy.decode) x) / 2)) +
          (2 : ℝ) ^ (-(255 - Real.logb 2 ((4 * (Inner.ofFlock Bd.deploy Bd.deploy.decode).V.k
              (xI (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)) x) + 1) *
            (cBe (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)) x +
              2 * (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)).nS x +
              setupLen (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)) x + 2 * ((n + v : ℕ) : ℝ))))) +
          (2 : ℝ) ^ (-(255 - Real.logb 2 (4 * Cp.width (xI (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)) x) *
            (cBe (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)) x +
              2 * (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)).nS x +
              setupLen (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)) x + 2 * ((n + v : ℕ) : ℝ))))) +
          (2 : ℝ) ^ (-(256 - Real.logb 2 (4 * (((Inner.ofFlock Bd.deploy Bd.deploy.decode).k
              (xI (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)) x) : ℝ) + 1) *
            (cBe (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)) x +
              2 * (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)).nS x +
              setupLen (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)) x + 2 * ((n + v : ℕ) : ℝ))))) +
          (2 : ℝ) ^ (-(255 - Real.logb 2 (βdev2 (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans))
              (Inner.ofFlock Bd.deploy Bd.deploy.decode) Cp x *
            (cBe (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)) x +
              2 * (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)).nS x +
              setupLen (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)) x + 2 * ((n + v : ℕ) : ℝ))))) ≤
        2 ^ (-(174.95 : ℝ)))
    (hn : ∀ cs ws, ((D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)).innerCalls H x cs ws).length ≤ n)
    (hv : ∀ ps, (setupCalls H (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans))
      ((D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)).vstar x ps)).length ≤ v)
    (hV : ∀ (r : Fin ((compiledAt D Bd DO Cp).toComp.V.k x) → Coin) ws,
      ((compiledAt D Bd DO Cp).toComp.innerLog ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans) H x r ws).length ≤ n)
    (hW : (∃ ω r, (run H P (compiledAt D Bd DO Cp).V x ω r).out = .acc) → ∀ (ω : Ω)
      (u : (Fin ((compiledAt D Bd DO Cp).toComp.V.k x) → Coin) × (compiledAt D Bd DO Cp).toComp.EO.C x) ws,
      (compiledAt D Bd DO Cp).toComp.wits H P x ω u = some ws →
        ((D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)).calls x (ws.map Prod.snd)).length ≤ v)
    (hSB : ∀ hk : 0 < (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)).V.k x,
      StrictBirthday H (B H (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)).V P x hk))
    (hB : ∀ h : 0 < (forkPlan (Coin := Coin) Bd.DE.V.k).sel ((compiledAt D Bd DO Cp).toComp.xI x),
      ExpectedCR H ((forkPlan (Coin := Coin) Bd.DE.V.k).finder
        ((compiledAt D Bd DO Cp).toComp.innerModel ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans) H P x)
        ((compiledAt D Bd DO Cp).toComp.xI x) h))
    (hFC : ∀ h : 0 < (capPlan Cp (2 ^ 256)).sel ((compiledAt D Bd DO Cp).toComp.xI x),
      ExpectedCR H ((capPlan Cp (2 ^ 256)).finder
        ((compiledAt D Bd DO Cp).toComp.innerModel ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans) H P x)
        ((compiledAt D Bd DO Cp).toComp.xI x) h))
    (hF0 : ∀ h : 0 < (conflictPlan (Coin := Coin) Bd.DE.k (2 ^ 257)).sel ((compiledAt D Bd DO Cp).toComp.xI x),
      ExpectedCR H ((conflictPlan (Coin := Coin) Bd.DE.k (2 ^ 257)).finder
        ((compiledAt D Bd DO Cp).toComp.innerModel ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans) H P x)
        ((compiledAt D Bd DO Cp).toComp.xI x) h))
    (hBI : ∀ h : 0 < (forkPlan (Coin := Coin) Bd.DE.V.k).sel ((compiledAt D Bd DO Cp).toComp.xI x),
      ExpectedCR H ((forkPlan (Coin := Coin) Bd.DE.V.k).finder
        ((compiledAt D Bd DO Cp).toComp.verModel ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans) H P x)
        ((compiledAt D Bd DO Cp).toComp.xI x) h))
    (hLink : ∀ h : 0 < (linkPlan (Coin := Coin) Bd.DE.V.k (2 ^ 256)).sel ((compiledAt D Bd DO Cp).toComp.xI x),
      ExpectedCR H ((linkPlan (Coin := Coin) Bd.DE.V.k (2 ^ 256)).finder
        ((compiledAt D Bd DO Cp).toComp.verModel ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans) H P x)
        ((compiledAt D Bd DO Cp).toComp.xI x) h)) :
    PrBad H (compiledAt D Bd DO Cp).V ((compiledAt D Bd DO Cp).toComp.R (Proto.ofFlock Bd Bd.deploy.decode Cp))
        ((compiledAt D Bd DO Cp).toComp.E (Proto.ofFlock Bd Bd.deploy.decode Cp) (Server.ofBound Bd)) x P ≤
      2 ^ (-(167.7 : ℝ)) + t * 2 ^ (-(174.95 : ℝ)) := by
  have hpos : (0 : ℝ) ≤ 2 ^ (-(167.7 : ℝ)) + t * 2 ^ (-(174.95 : ℝ)) := by positivity
  by_cases hex : ∃ ω r, (run H P (compiledAt D Bd DO Cp).V x ω r).out = .acc
  · obtain ⟨hx, hκ, hε⟩ := hb hex
    exact (ksCompiledOn_bits hK hN hx P t ht1 hrun n v hn hv hV (hW hex) hSB hB hFC hF0 hBI hLink).trans
      (add_le_add hκ (mul_le_mul_of_nonneg_left hε (Nat.cast_nonneg t)))
  · unfold PrBad
    refine le_trans (Pr_mono (q := fun _ => False) fun q hq => ?_) (Pr_false.le.trans hpos)
    exact hex ⟨q.1, q.2.1, hq.1⟩

end Instance

end Claims.RecGeneric
