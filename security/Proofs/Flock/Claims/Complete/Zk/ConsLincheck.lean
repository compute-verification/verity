module

public import Proofs.Flock.Claims.Complete.Replay.ZerocheckAlgebra
public import Proofs.Flock.Claims.Complete.Zk.ConsZerocheck
public import Proofs.Flock.Claims.Complete.Zk.PiopGo
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.Complete.Replay.LincheckAlgebra

/-! The lincheck's constraint vanishes at the masked prover's pads, by masks, masked evaluation and the plain algebra
(`lcFinal_zero`). -/

@[expose] public section

namespace FlockComplete.Replay

open Flock FlockLevel3 FlockSoundness FlockSoundness.Refine FlockSoundness.Discharge.Composed
open ZerocheckMasked (evalL)

local notation "𝔽" => FlockLevel3.GF128

structure LcMasks (lay : Zk.PadLayout) (h : Array F128) (lc lc' : LincheckProof) (n : ℕ) : Prop where
  rounds : ∀ j < n, lc'.rounds[j]! = (lc.rounds[j]!.1 + h[lay.lc j 0]!, lc.rounds[j]!.2 + h[lay.lc j 1]!)
  zp : ∀ i < 64, lc'.zPartial[i]! = lc.zPartial[i]! + h[lay.zp i]!

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
