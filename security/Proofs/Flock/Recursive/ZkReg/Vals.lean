module

public import Proofs.Flock.Recursive.ZkReg.DrawnRefines
public import Proofs.Flock.Recursive.ZkReg.RegLink
public import Proofs.Flock.Soundness.Registered.ReadsX
public import Proofs.Flock.Soundness.Discharge.ZkReg.BindHJ
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
import Proofs.Flock.Soundness.Discharge.ZkReg.Checked
import Proofs.Flock.Soundness.Game

/-! The committed values of a `--zk --registered` session at its registered reads. -/

@[expose] public section

namespace FlockSoundness.Discharge.ZkReg

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR DecodeZL
open Composed
open ZkHidden (planZHJ tabZHJ SampleSetupZHJ TagsOkZHJ SampleOkZKHJ layShHJ placeZHJ)
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

variable {mPts n : ℕ} {Reg : Type}

/-- **The value binding of a hidden-output `--zk` session of `J` tables** (`rsZHJS`'s). -/
noncomputable abbrev vbHJ {I : Inputs} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : SampleSetupZHJ (inputsJ I 0) mPts S₀ d₀) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hscope : Layout.scopeOkH c = true)
    (hU : Built c.unit) {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub) :=
  (rsZHJS (Reg := Reg) dj y₀ ht hc hscope hU hpub).bindingZ (rsZHJS_computes dj y₀ ht hc hscope hU hpub)

/-- **The registrant's opening** of each commit string of `vbHJ`: the row and the salt `rd` reads there. -/
def regOpenZ {I : Inputs} (ht : I.tags.typed = false) (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    {d₀ : Lean.Json} (y₀ : SampleSetupZHJ (inputsJ I 0) mPts S₀ d₀) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit) {pub : Public}
    (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub) (hscope : Layout.scopeOkH c = true)
    (rd : FlockSoundness.Registered.ReadsX (layShHJ (Reg := Reg) dj y₀ ht hc hscope hU (pub := pub))
      (rsShHJS dj y₀ ht hc hscope hU hpub)) :
    (vbHJ (Reg := Reg) dj y₀ ht hc hscope hU hpub).Pos → (vbHJ (Reg := Reg) dj y₀ ht hc hscope hU hpub).Opening :=
  fun p => (p, rd.wv p, rd.ws p)

theorem zk_session_regValsR
    (I : Inputs) (ht : I.tags.typed = false) (own : Array Flock.Registered.Port)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : SampleSetupZHJ (inputsJ I 0) mPts S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length mPts → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)
    (hscope : Layout.scopeOkH c = true)
    (Lv : SamplingDesign n)
    (hDraw : ∀ ω, SampleOkZKHJ I dj mPts (Lv.draw ω))
    (hTags : TagsOkZHJ I) (hRec : RecordsLiveZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t')
    (rd : FlockSoundness.Registered.ReadsX (layShHJ (Reg := Reg) dj y₀ ht hc hscope hU (pub := pub))
      (rsShHJS dj y₀ ht hc hscope hU hpub))
    (τ : Strategy (auditLive Lv Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ execArith hs512
      (planZHJ dj y₀ (Lv.draw ω) (reg (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ))) j
      (zcStrat execArith hs512 (planZHJ dj y₀) (cont (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ) ω)) qF qS)
    {qW : ℝ}
    (hW : RegCRZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
      (vbHJ dj y₀ ht hc hscope hU hpub) (regOpenZ ht dj y₀ hc hU hpub hscope rd)
      (reg (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ))
      (cont (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ)) k Rw qW) :
    prob (fun o => LiveAcceptsZKR I own o ∧
        ∃ u ∈ o.2.1, ∃ g ∈ ((keyProg c hU n).partition (Layout.outsOf c hU n)).io u,
          rd.isRead ((layShHJ dj y₀ ht hc hscope hU (pub := pub)).pos g) ∧
          (layShHJ dj y₀ ht hc hscope hU (pub := pub)).regLeaf o.1
              ((layShHJ dj y₀ ht hc hscope hU (pub := pub)).pos g) =
            rd.leaf ((layShHJ dj y₀ ht hc hscope hU (pub := pub)).pos g) ∧
          XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
              (vbHJ dj y₀ ht hc hscope hU hpub) (reg (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ))
              (cont (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ)) k Rw g ≠
            (layShHJ dj y₀ ht hc hscope hU (pub := pub)).bit
              (rd.wv ((layShHJ dj y₀ ht hc hscope hU (pub := pub)).pos g)) g)
        (auditLive Lv Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      ksAvgStrictZ (planZHJ dj y₀) Lv k qF qS (reg (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ)) +
        linkBoundZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          (vbHJ dj y₀ ht hc hscope hU hpub) k Rw M ρ t'
          (reg (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ))
          (cont (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ)) +
        regTermZC execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ (Reg := Reg) dj y₀)
          (tabZHJ dj y₀) (vbHJ dj y₀ ht hc hscope hU hpub) Lv k Rw ρ qW := by
  have hdec := decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
    (ZkHidden.runsZCJH_live I ht hc (Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr Lv hTags hRec hDraw)
  have hc' := zk_flock_regValsC_cr execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀)
    (tabZHJ dj y₀) execArith_correct (placeZHJ dj y₀ ht hc hscope hU) (vbHJ dj y₀ ht hc hscope hU hpub)
    (regOpenZ ht dj y₀ hc hU hpub hscope rd) hk hRw hM hρ hr ht'
    (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ) hCR hW
  refine le_trans (prob_mono (fun o h => ⟨acceptsZK_of_R h.1, h.2⟩) _ _)
    (zk_headline_exec_sampledJ execArith hs512 _ I wr _ dg _ _ _ _ N hdec τ (fun R S => ∃ u ∈ S,
      ∃ g ∈ ((keyProg c hU n).partition (Layout.outsOf c hU n)).io u,
        rd.isRead ((layShHJ dj y₀ ht hc hscope hU (pub := pub)).pos g) ∧
        (layShHJ dj y₀ ht hc hscope hU (pub := pub)).regLeaf R ((layShHJ dj y₀ ht hc hscope hU (pub := pub)).pos g) =
          rd.leaf ((layShHJ dj y₀ ht hc hscope hU (pub := pub)).pos g) ∧
        XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
            (vbHJ dj y₀ ht hc hscope hU hpub) (reg (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ))
            (cont (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ)) k Rw g ≠
          (layShHJ dj y₀ ht hc hscope hU (pub := pub)).bit
            (rd.wv ((layShHJ dj y₀ ht hc hscope hU (pub := pub)).pos g)) g) _
      (le_trans (prob_mono ?_ _ _) hc'))
  rintro o ⟨hacc, u, hu, g, hg, hread, hleaf, hne⟩
  refine ⟨hacc, u, hu, g, hg, ?_, hne⟩
  refine Prod.ext rfl ?_
  show ((rsZHJS dj y₀ ht hc hscope hU hpub).scheme _).leaf (rd.wv _) (rd.ws _) = _
  unfold HmRowsZX.scheme
  split_ifs with hrow
  · exact hleaf.symm
  · exact absurd (rd.read_row _ hread) hrow

theorem regBadZ_of_leaf {I : Inputs} (ht : I.tags.typed = false) (dj : Finset (Fin n) → Lean.Json)
    {S₀ : Finset (Fin n)} {d₀ : Lean.Json} (y₀ : SampleSetupZHJ (inputsJ I 0) mPts S₀ d₀) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit) {pub : Public}
    (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub) (hscope : Layout.scopeOkH c = true)
    (rd : FlockSoundness.Registered.ReadsX (layShHJ (Reg := Reg) dj y₀ ht hc hscope hU (pub := pub))
      (rsShHJS dj y₀ ht hc hscope hU hpub))
    (R : Reg) (Xt : Fin (Layout.keyN c hU n) → Bool) (g : Fin ((keyProg c hU n).circuit (Layout.outsOf c hU n)).N)
    (hread : rd.isRead ((layShHJ dj y₀ ht hc hscope hU (pub := pub)).pos g))
    (hleaf : (layShHJ dj y₀ ht hc hscope hU (pub := pub)).regLeaf R ((layShHJ dj y₀ ht hc hscope hU (pub := pub)).pos g) =
      rd.leaf ((layShHJ dj y₀ ht hc hscope hU (pub := pub)).pos g))
    (hne : Xt g ≠ (layShHJ dj y₀ ht hc hscope hU (pub := pub)).bit
      (rd.wv ((layShHJ dj y₀ ht hc hscope hU (pub := pub)).pos g)) g) :
    RegBadZ execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
      (vbHJ dj y₀ ht hc hscope hU hpub) (regOpenZ ht dj y₀ hc hU hpub hscope rd) R Xt g := by
  refine ⟨Prod.ext rfl ?_, hne⟩
  show ((rsZHJS dj y₀ ht hc hscope hU hpub).scheme _).leaf (rd.wv _) (rd.ws _) = _
  unfold HmRowsZX.scheme
  split_ifs with hrow
  · exact hleaf.symm
  · exact absurd (rd.read_row _ hread) hrow

theorem zk_session_regSampledR
    (I : Inputs) (ht : I.tags.typed = false) (own : Array Flock.Registered.Port)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : SampleSetupZHJ (inputsJ I 0) mPts S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length mPts → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)
    (hscope : Layout.scopeOkH c = true)
    (Lv : SamplingDesign n)
    (hDraw : ∀ ω, SampleOkZKHJ I dj mPts (Lv.draw ω))
    (hTags : TagsOkZHJ I) (hRec : RecordsLiveZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t')
    (rd : FlockSoundness.Registered.ReadsX (layShHJ (Reg := Reg) dj y₀ ht hc hscope hU (pub := pub))
      (rsShHJS dj y₀ ht hc hscope hU hpub))
    (τ : Strategy (auditLive Lv Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ execArith hs512
      (planZHJ dj y₀ (Lv.draw ω) (reg (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ))) j
      (zcStrat execArith hs512 (planZHJ dj y₀) (cont (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ) ω)) qF qS)
    {qW : ℝ}
    (hW : RegCRZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
      (vbHJ dj y₀ ht hc hscope hU hpub) (regOpenZ ht dj y₀ hc hU hpub hscope rd)
      (reg (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ))
      (cont (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ)) k Rw qW) :
    prob (fun o => LiveAcceptsZKR I own o ∧
        ¬ Disjoint (((keyProg c hU n).partition (Layout.outsOf c hU n)).incorrect
            (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
              (vbHJ dj y₀ ht hc hscope hU hpub) (reg (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ))
              (cont (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ)) k Rw) ∪
          regWrongZ execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
            (vbHJ dj y₀ ht hc hscope hU hpub) (regOpenZ ht dj y₀ hc hU hpub hscope rd)
            (reg (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ))
            (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
              (vbHJ dj y₀ ht hc hscope hU hpub) (reg (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ))
              (cont (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ)) k Rw))
          o.2.1)
        (auditLive Lv Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      ksAvgStrictZ (planZHJ dj y₀) Lv k qF qS (reg (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ)) +
        linkBoundZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          (vbHJ dj y₀ ht hc hscope hU hpub) k Rw M ρ t'
          (reg (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ))
          (cont (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ)) +
        regTermZC execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ (Reg := Reg) dj y₀)
          (tabZHJ dj y₀) (vbHJ dj y₀ ht hc hscope hU hpub) Lv k Rw ρ qW := by
  have hdec := decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
    (ZkHidden.runsZCJH_live I ht hc (Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr Lv hTags hRec hDraw)
  have hc' := zk_flock_regSampledC_cr execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀)
    (tabZHJ dj y₀) execArith_correct (placeZHJ dj y₀ ht hc hscope hU) (vbHJ dj y₀ ht hc hscope hU hpub)
    (regOpenZ ht dj y₀ hc hU hpub hscope rd) hk hRw hM hρ hr ht'
    (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ) hCR hW
  exact le_trans (prob_mono (fun o h => ⟨acceptsZK_of_R h.1, h.2⟩) _ _)
    (zk_headline_exec_sampledJ execArith hs512 _ I wr _ dg _ _ _ _ N hdec τ (fun _ S => ¬ Disjoint
      (((keyProg c hU n).partition (Layout.outsOf c hU n)).incorrect
          (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
            (vbHJ dj y₀ ht hc hscope hU hpub) (reg (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ))
            (cont (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ)) k Rw) ∪
        regWrongZ execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          (vbHJ dj y₀ ht hc hscope hU hpub) (regOpenZ ht dj y₀ hc hU hpub hscope rd)
          (reg (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ))
          (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
            (vbHJ dj y₀ ht hc hscope hU hpub) (reg (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ))
            (cont (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ)) k Rw)) S) _ hc')

end FlockSoundness.Discharge.ZkReg
