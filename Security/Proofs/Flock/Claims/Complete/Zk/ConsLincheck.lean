import Proofs.Flock.Claims.Complete.Zk.ConsZerocheck
import Proofs.Flock.Claims.Complete.Replay.LincheckAlgebra

/-!
# The lincheck's constraint vanishes at the masked prover's pads

The masked completeness is the plain one transferred through the pads, phase by phase, in three steps:

1. **masks**: the masked prover's proof is the plain prover's with each sent value plus its pad (`zcZ_masks`,
   `lcZ_masks`);
2. **masked evaluation**: the replay's affine form on a masked proof, at the pads, is the plain replay's value on the
   plain proof (`ZerocheckMasked.final_eval`; here `lcRunning_eval`, `lcFinal_eval`), at any coins, for any plain proof;
3. **plain algebra**: the plain replay's two sides agree on the plain prover's proof (`Zc.fold_final`; here
   `Lc.lcAlgebra_of`), which the plain completeness proves.

For the lincheck (`Zk.replay`'s "lincheck final", restated by `Lincheck.lcFinal`):

* `lcZ_masks`: `LcZ.proof` is `Lc.proof` masked at `lay.lc` and `lay.zp`;
* `lcFinal_eval`: on a masked lincheck proof and masked `final_a`, `final_b`, the row evaluates to the plain verifier's
  round fold (`RecAlgebra.lcFold`) from `α·final_a + final_b + β`, plus `⟨comb, z_partial⟩`;
* `lcFinal_zero`: at the masked prover's proofs the row vanishes, at any pads and every coin where the circuit fold
  returns, for a trace of a witness (`Lc.lcAlgebra_of`'s hypotheses).
-/

namespace FlockComplete.Replay

open Flock FlockLevel3 FlockSoundness FlockSoundness.Refine FlockSoundness.Discharge.Composed
open ZerocheckMasked (evalL)

local notation "𝔽" => FlockLevel3.GF128

/-- `lc'` is the clear `lc` masked by the pads `h` at `lay`'s indices, on its first `n` rounds and its `z_partial`. -/
structure LcMasks (lay : Zk.PadLayout) (h : Array F128) (lc lc' : LincheckProof) (n : ℕ) : Prop where
  rounds : ∀ j < n, lc'.rounds[j]! = (lc.rounds[j]!.1 + h[lay.lc j 0]!, lc.rounds[j]!.2 + h[lay.lc j 1]!)
  zp : ∀ i < 64, lc'.zPartial[i]! = lc.zPartial[i]! + h[lay.zp i]!

/-- **The masked lincheck proof is the plain one masked by the pads**, on all `k_log − 6` rounds and `z_partial`. -/
theorem lcZ_masks (lay : Zk.PadLayout) (h : Array F128) (c : CircuitFold) (m : ℕ) (tr : Trace) (zc : ZcOut)
    (full : Array F128) (cs : Lc.Challenges) :
    LcMasks lay h (Lc.proof c m tr zc full cs) (LcZ.proof lay h c m tr zc full cs) (c.kLog - 6) where
  rounds j hj := by
    simp only [LcZ.proof, Lc.proof]
    rw [pushes_get _ _ j hj, pushes_get _ _ j hj]
    rfl
  zp i hi := by
    simp only [LcZ.proof, LcZ.zp, Lc.proof]
    rw [getElem!_pos _ i (by simpa using hi)]
    simp

/-! ## The masked evaluation -/

theorem lc_lt_len (lay : Zk.PadLayout) (j t : ℕ) (hj : j < lay.nLc) (ht : t < 2) : lay.lc j t < lay.len := by
  simp only [Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.hab]
  omega

theorem zp_lt_len (lay : Zk.PadLayout) (i : ℕ) (hi : i < 64) : lay.zp i < lay.len := by
  simp only [Zk.PadLayout.zp, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.hab]
  omega

theorem evalL_addConst (x : Zk.Lin) (k : F128) (h : Array F128) : evalL (x.addConst k) h = evalL x h + k := by
  apply Zerocheck.eq_of_of
  simp only [evalL, Zk.Lin.addConst, of_add]
  ring

theorem lcRunning_succ (lay : Zk.PadLayout) (fA fB alpha beta : F128) (rounds : Array (F128 × F128))
    (ts : Array F128) (j : ℕ) :
    Lincheck.lcRunning lay fA fB alpha beta rounds ts (j + 1) =
      Lincheck.lcStep lay rounds ts (Lincheck.lcRunning lay fA fB alpha beta rounds ts j) j := by
  simp only [Lincheck.lcRunning, List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]

theorem lcRunning_size (lay : Zk.PadLayout) (fA fB alpha beta : F128) (rounds : Array (F128 × F128))
    (ts : Array F128) (j : ℕ) : (Lincheck.lcRunning lay fA fB alpha beta rounds ts j).a.size = lay.len := by
  unfold Lincheck.lcRunning
  dsimp only
  exact Lincheck.foldl_lcStep_size _ _ _ _ (by simp [Zk.Lin.addConst]) j

/-- **The masked lincheck's running form evaluates to the plain verifier's claim**: after `j ≤ n` rounds, on masked
`final_a`, `final_b` and a masked lincheck proof, it is the fold of the plain step `lStep` on the clear messages from
`α·final_a + final_b + β`, at the same coins. -/
theorem lcRunning_eval (lay : Zk.PadLayout) (h : Array F128) (lc lc' : LincheckProof) (n : ℕ) (hn : n ≤ lay.nLc)
    (hm : LcMasks lay h lc lc' n) (fA fB fA' fB' alpha beta : F128) (hA : fA' = fA + h[lay.fa]!)
    (hB : fB' = fB + h[lay.fb]!) (ts : Array F128) :
    ∀ j ≤ n, evalL (Lincheck.lcRunning lay fA' fB' alpha beta lc'.rounds ts j) h =
      (Flock.RecAlgebra.lcFold lc (fun j => #[ts[j]!]) j (alpha * fA + fB + beta)).1
  | 0, _ => by
      have ha := ZerocheckMasked.fa_lt_len lay
      have hb := ZerocheckMasked.fb_lt_len lay
      simp only [Lincheck.lcRunning, List.range_zero, List.foldl_nil, Flock.RecAlgebra.lcFold, List.range'_zero]
      rw [evalL_addConst, ZerocheckMasked.evalL_add _ _ h (by simp), ZerocheckMasked.evalL_scale,
        ZerocheckMasked.evalL_masked _ _ _ ha, ZerocheckMasked.evalL_masked _ _ _ hb, hA, hB,
        ZerocheckMasked.add_add_self, ZerocheckMasked.add_add_self]
      apply Zerocheck.eq_of_of
      simp only [of_add, of_mul]
      ring
  | j + 1, hj => by
      have ih := lcRunning_eval lay h lc lc' n hn hm fA fB fA' fB' alpha beta hA hB ts j (by omega)
      have h0 := lc_lt_len lay j 0 (by omega) (by omega)
      have h1 := lc_lt_len lay j 1 (by omega) (by omega)
      have hsz := lcRunning_size lay fA' fB' alpha beta lc'.rounds ts j
      have hg : (#[ts[j]!] : Array F128).getD 0 0 = ts[j]! := rfl
      rw [lcRunning_succ, Lc.lcFold_succ]
      simp only [Lincheck.lcStep, lStep, hm.rounds j (by omega), hg]
      simp (disch := simp [hsz, h0, h1]) only [ZerocheckMasked.evalL_add, ZerocheckMasked.evalL_scale,
        ZerocheckMasked.evalL_masked]
      rw [ih]
      simp only [ZerocheckMasked.add_add_self]
      apply Zerocheck.eq_of_of
      simp only [of_add, of_mul]
      ring

/-- **`lincheck final` evaluates to its two plain sides**: the plain verifier's round fold from
`α·final_a + final_b + β` and `⟨comb, z_partial⟩`, on the clear values, at any coins. -/
theorem lcFinal_eval (lay : Zk.PadLayout) (h : Array F128) (lc lc' : LincheckProof) (n : ℕ) (hn : n ≤ lay.nLc)
    (hm : LcMasks lay h lc lc' n) (fA fB fA' fB' alpha beta : F128) (hA : fA' = fA + h[lay.fa]!)
    (hB : fB' = fB + h[lay.fb]!) (ts combv : Array F128) (hc : combv.size = 64) :
    evalL (Lincheck.lcFinal lay fA' fB' alpha beta lc'.rounds lc'.zPartial ts n combv) h =
      (Flock.RecAlgebra.lcFold lc (fun j => #[ts[j]!]) n (alpha * fA + fB + beta)).1 + dot combv lc.zPartial := by
  unfold Lincheck.lcFinal
  rw [ZerocheckMasked.evalL_add _ _ h (by rw [lcRunning_size, ZerocheckMasked.size_combo]),
    lcRunning_eval lay h lc lc' n hn hm fA fB fA' fB' alpha beta hA hB ts n le_rfl,
    ZerocheckMasked.evalL_combo _ _ lay.len h fun l hl => ?_]
  · congr 1
    refine ZerocheckMasked.dot_congr _ _ _ 64 hc fun i hi => ?_
    rw [ZerocheckMasked.get_map' _ _ i (by simpa [Lincheck.lcZp] using hi), Lincheck.lcZp,
      ZerocheckMasked.get_map_lin _ hi, ZerocheckMasked.evalL_masked _ _ _ (zp_lt_len lay i hi), hm.zp i hi,
      ZerocheckMasked.add_add_self]
  · obtain ⟨i, _, rfl⟩ := ZerocheckMasked.mem_map_range hl
    simp

/-! ## At the masked prover -/

/-- **`lincheck final` vanishes at the masked prover's pads**, at any pads and every coin where the circuit fold
returns, for a trace of a witness of a statement the circuit fold realizes (`Lc.lcAlgebra_of`'s hypotheses, verbatim):
the replay's folded circuit `comb` returns, and the row on the masked zerocheck's `final_a`, `final_b` and the masked
lincheck proof evaluates to `0`. -/
theorem lcFinal_zero (S : Model.Statement) (c : CircuitFold) (hc : FoldRealizes c S)
    (hfd : ∀ (α β : F128) (lw ρ ts v : Array F128), c.fold α lw ρ = .ok v → ∃ w, c.folded α β lw ρ ts = .ok w)
    (m : ℕ) (hm : m = S.m) (w : Model.Witness S.m) (hpin : ∀ o, S.block w o S.pin = 1) (tr : Trace)
    (hz : ∀ a : Fin (2 ^ S.m), Zc.bitF tr.z a.val = ((w a).val : 𝔽))
    (ha : ∀ a : Fin (2 ^ S.m), Zc.bitF tr.az a.val = ((aVec S w a).val : 𝔽))
    (hb : ∀ a : Fin (2 ^ S.m), Zc.bitF tr.bz a.val = ((bVec S w a).val : 𝔽))
    (zcs : Zc.Challenges) (cs : Lc.Challenges) (full : Array F128)
    (hfold : c.fold (cs.ca 0) (skipWeights (Zc.out m zcs tr).z) ((Zc.out m zcs tr).rho.extract 0 (c.kLog - 6)) =
      .ok full)
    (lay : Zk.PadLayout) (h : Array F128) (hn : c.kLog - 6 ≤ lay.nLc) :
    ∃ comb, c.folded (cs.ca 0) (cs.cb 0) (skipWeights (Zc.out m zcs tr).z)
        ((Zc.out m zcs tr).rho.extract 0 (c.kLog - 6)) (pushes cs.τ (c.kLog - 6)) = .ok comb ∧
      evalL (Lincheck.lcFinal lay (ZcZ.proof lay h m zcs tr).finalA (ZcZ.proof lay h m zcs tr).finalB (cs.ca 0)
        (cs.cb 0) (LcZ.proof lay h c m tr (Zc.out m zcs tr) full cs).rounds
        (LcZ.proof lay h c m tr (Zc.out m zcs tr) full cs).zPartial (pushes cs.τ (c.kLog - 6)) (c.kLog - 6) comb) h =
          0 := by
  obtain ⟨comb, hcomb, hsz, halg⟩ := Lc.lcAlgebra_of S c hc hfd m hm w hpin tr hz ha hb zcs cs full hfold
  refine ⟨comb, hcomb, ?_⟩
  rw [lcFinal_eval lay h _ _ (c.kLog - 6) hn (lcZ_masks lay h c m tr (Zc.out m zcs tr) full cs)
      (Zc.proof m zcs tr).finalA (Zc.proof m zcs tr).finalB (ZcZ.proof lay h m zcs tr).finalA
      (ZcZ.proof lay h m zcs tr).finalB _ _ rfl rfl _ _ hsz,
    Lc.lcFold_congr (C := fun j => #[(pushes cs.τ (c.kLog - 6))[j]!]) (C' := fun j => #[cs.τ j]) _ _ _ fun i hi =>
      show (pushes cs.τ (c.kLog - 6))[i]! = cs.τ i from pushes_get cs.τ (c.kLog - 6) i hi]
  exact (congrArg (· + dot comb _) halg).trans (ZerocheckMasked.add_self' _)

end FlockComplete.Replay
