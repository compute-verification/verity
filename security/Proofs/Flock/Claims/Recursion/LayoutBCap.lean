module

public import Proofs.Flock.C3.Reps.AtList
public import Proofs.Flock.Claims.KS.PlainCapsDep
public import Proofs.Flock.Claims.Recursion.BindsDeployed
public import Proofs.Flock.Claims.Recursion.BoundRouteKSLin2
public import Proofs.Flock.Claims.Recursion.DeployScope
public import Proofs.Flock.Claims.Recursion.KSLin2PlainClosed
public import Proofs.Flock.Claims.Recursion.LayoutBDeployed
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
import Std.Tactic.BVDecide.Normalize.Bool

/-! V*'s bound layout with round 0 held at RecCap (`layoutBCap`), and `HeldB` there from named facts, one of which, `TopsHeld`, no part meets. -/

@[expose] public section

namespace RecDeploy

open Flock
open Claims (Msg)
open FlockDeploy (Coin)

/-- **RecCap in V\*'s list** at the verifier's own inner shape: its index, the list's last, and its node count `N`. -/
def capAt (r : XR) : Option (ℕ × ℕ) :=
  (RecVStar.statements r.shape).toOption.bind fun ss =>
    match ss.back? with
    | some (VStar.Stmt.cap 0 N) => some (ss.size - 1, N)
    | _ => none

/-- RecCap's port `top` among a circuit's ports. -/
def capPort (c : Circuit) : Option ℕ := c.ports.findIdx? (·.name == "top")

/-- **What RecCap's part holds of round 0**: at RecCap's index, the commit strings of its `N` instances in order, each
port `top`'s hm96 slot's output `b ‖ c` at that instance (`hmCommit`). None at any other part. -/
def capHold (r : XR) (s : ℕ) : Option (List (Msg × List HmNets.Lin)) :=
  (capAt r).bind fun p =>
    if s = p.1 then
      (r.w.parts[s]?.bind partCircuit).bind fun c => (capPort c).map fun q =>
        (List.range p.2).map fun n => (([] : Msg), hmCommit c q n)
    else none

def capCalls (r : XR) (s : ℕ) : List (Msg × List HmNets.Lin × List HmNets.Lin) :=
  ((capAt r).bind fun p =>
    if s = p.1 then
      (r.w.parts[s]?.bind partCircuit).bind fun c => (capPort c).map fun q =>
        (List.range p.2).flatMap fun n => hmCalls c "top" q n
    else none).getD []

theorem readX_statements {x : Msg} {r : XR} (h : readX x = some r) :
    ∃ vs, RecVStar.statements r.shape = .ok vs ∧ r.w.parts.length = vs.size := by
  replace h := readXPre_of_readX h
  unfold readXPre at h
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨w, -, ri, -, sh, -, pa, -, dims, -, vs, hvs, h⟩ := h
  split_ifs at h with hl
  · simp at h
  have h' : (⟨w, ri, pa, sh, dims⟩ : XR) = r := by simpa using h
  subst h'
  exact ⟨vs, toOption_eq_some hvs, not_not.1 (not_or.1 hl).1⟩

theorem capAt_readX {x : Msg} {r : XR} (h : readX x = some r) :
    ∃ N, capAt r = some (nS x - 1, N) ∧ 0 < nS x := by
  obtain ⟨vs, hvs, hlen⟩ := readX_statements h
  obtain ⟨_, _, _, _, _, _, A, C, _, _, _, _, _, _, _, K, L, N, -, -, -, rfl⟩ := FlockC3.Reps.list_ok hvs
  have hn : nS x = r.w.parts.length := by simp [nS, h]
  refine ⟨N, ?_, ?_⟩
  · rw [hn, hlen]
    simp [capAt, hvs, Except.toOption]
  · rw [hn, hlen]
    simp

def lastPieces (fin : Msg → List (ℕ × List (Msg × List HmNets.Lin))) (x : Msg) (i : ℕ) :
    List (ℕ × List (Msg × List HmNets.Lin)) :=
  if i + 1 = layoutOf.rounds x then fin x else []

def layoutBCap (topsAt : Msg → ℕ → ℕ → Option (List (Msg × List HmNets.Lin)))
    (fin : Msg → List (ℕ × List (Msg × List HmNets.Lin))) : Claims.Recursion.VStarB.LayoutB where
  rounds := layoutOf.rounds
  msgAt x s i := match i with
    | 0 => (readX x).bind fun r => capHold r s
    | j + 1 => (streamRound x j).bind fun sr =>
      if sr.2.1.isEmpty then
        if sr.2.2.2 then
          (readX x).bind fun r => (r.w.parts[s]?.bind partCircuit).bind fun c =>
            ((heldRows r s).find? (·.1 == j)).map fun h => [(([] : Msg), hmCommit c h.2.2.1 h.2.2.2)]
        else none
      else topsAt x s (j + 1)
  shaAt x s := layoutBDeployed.shaAt x s ++ ((readX x).map fun r => capCalls r s).getD []
  fixedAt x i := match i with
    | 0 => none
    | j + 1 => (streamRound x j).bind fun sr => if sr.2.1.isEmpty && !sr.2.2.2 then some [] else none
  pieceAt x i := lastPieces fin x i

section Layout

variable (topsAt : Msg → ℕ → ℕ → Option (List (Msg × List HmNets.Lin)))
  (fin : Msg → List (ℕ × List (Msg × List HmNets.Lin)))

theorem layoutBCap_msgAt_zero (x : Msg) (s : ℕ) :
    (layoutBCap topsAt fin).msgAt x s 0 = (readX x).bind fun r => capHold r s := rfl

theorem layoutBCap_fixedAt_zero (x : Msg) : (layoutBCap topsAt fin).fixedAt x 0 = none := rfl

end Layout

theorem heldRows_circuit {r : XR} {s : ℕ} {h : ℕ × String × ℕ × ℕ} (hm : h ∈ heldRows r s) :
    ∃ c, r.w.parts[s]?.bind partCircuit = some c := by
  cases h2 : r.w.parts[s]?.bind partCircuit with
  | some c => exact ⟨c, rfl⟩
  | none =>
    exfalso
    unfold heldRows at hm
    split at hm <;> simp_all

/-- **RecCap's part has its port `top`** at the compiled statements of the shape: the last part's circuit, as the setup
parses it, has the port RecCap reads node `n` through at instance `n`. -/
def CapPorts (shape : Claims.Recursion.Form → Prop) : Prop :=
  ∀ x st, parse x = some st → shape st.form → ∀ r, readX x = some r →
    ∃ c, r.w.parts[nS x - 1]?.bind partCircuit = some c ∧ (capPort c).isSome

def RowsCover (shape : Claims.Recursion.Form → Prop) : Prop :=
  ∀ x st, parse x = some st → shape st.form → ∀ r j sr, readX x = some r → streamRound x j = some sr →
    sr.2.1.isEmpty = true → sr.2.2.2 = true → ∃ s < nS x, ∃ h ∈ heldRows r s, h.1 = j

/-- No part holds such a round whole (its row is another part's), and at `topsAt` none `HeldB` fails
(`not_heldB_layoutBCap_none`); `layoutBTops` holds those rounds as pieces. -/
def TopsHeld (topsAt : Msg → ℕ → ℕ → Option (List (Msg × List HmNets.Lin))) (shape : Claims.Recursion.Form → Prop) :
    Prop :=
  ∀ x st, parse x = some st → shape st.form → ∀ j sr, streamRound x j = some sr → sr.2.1.isEmpty = false →
    ∃ s < nS x, (topsAt x s (j + 1)).isSome

def FinPieces (fin : Msg → List (ℕ × List (Msg × List HmNets.Lin))) (shape : Claims.Recursion.Form → Prop) : Prop :=
  ∀ x st, parse x = some st → shape st.form → fin x ≠ [] ∧ ∀ p ∈ fin x, p.1 < nS x

theorem heldB_layoutBCap {W : Type} (IV : Claims.Verifier Coin) (ivScope : Msg → Bool) (b : Brings)
    (I : Claims.Recursion.Inner Coin W) (hk : ∀ x, I.V.k x = (FlockDeploy.V false).k x)
    (topsAt : Msg → ℕ → ℕ → Option (List (Msg × List HmNets.Lin)))
    (fin : Msg → List (ℕ × List (Msg × List HmNets.Lin))) (shape : Claims.Recursion.Form → Prop)
    (hM : SchedMatches shape) (hP : CapPorts shape) (hR : RowsCover shape) (hT : TopsHeld topsAt shape)
    (hF : FinPieces fin shape) :
    Claims.Recursion.VStarB.HeldB (deployRec IV ivScope b) I (layoutBCap topsAt fin) shape := by
  intro x st hst hf
  obtain ⟨r, rs, hr, hrs, hK⟩ := hM x st hst hf
  have hkx : I.V.k (Claims.Recursion.xI (deployRec IV ivScope b) x) = 2 + 2 * rs.size := by
    rw [hk]
    exact hK
  have hL : layoutOf.rounds x = 2 + 2 * rs.size := hK
  have hsr : ∀ j, j < 2 * rs.size → streamRound x j = some (rs.getD (j % rs.size) default) := by
    intro j hj
    simp only [streamRound, hr, Option.bind_some, hrs, hj, ↓reduceIte]
  have hsr' : streamRound x (2 * rs.size) = none := by
    simp [streamRound, hr, hrs]
  refine ⟨hkx.trans hL.symm, fun i hi => ?_, fun s => ?_⟩
  · rw [hkx] at hi
    cases i with
    | zero =>
      obtain ⟨N, hcap, hn0⟩ := capAt_readX hr
      obtain ⟨c, hc, hq⟩ := hP x st hst hf r hr
      obtain ⟨q, hq'⟩ := Option.isSome_iff_exists.1 hq
      refine ⟨Or.inr (Or.inl ⟨nS x - 1, show nS x - 1 < nS x by omega, ?_⟩), fun h0 => ?_⟩
      · rw [layoutBCap_msgAt_zero]
        simp [hr, capHold, hcap, hc, hq']
      · rw [layoutBCap_fixedAt_zero] at h0
        simp at h0
    | succ j =>
      rcases Nat.lt_or_eq_of_le (show j ≤ 2 * rs.size by omega) with hj | rfl
      · obtain ⟨sr, hs⟩ : ∃ sr, streamRound x j = some sr := ⟨_, hsr j hj⟩
        have hpc : (layoutBCap topsAt fin).pieceAt x (j + 1) = [] := by
          have hc : ¬ (j + 1 + 1 = layoutOf.rounds x) := by
            rw [hL]
            omega
          simp only [layoutBCap, lastPieces, if_neg hc]
        refine ⟨?_, fun _ => hpc⟩
        cases ht : sr.2.1.isEmpty with
        | false =>
          have ht' : ¬ sr.2.1 = #[] := by simpa using ht
          right; left
          obtain ⟨s, hs', hsome⟩ := hT x st hst hf j sr hs ht
          refine ⟨s, hs', ?_⟩
          have hm : (layoutBCap topsAt fin).msgAt x s (j + 1) = topsAt x s (j + 1) := by
            simp [layoutBCap, hs, ht']
          rw [hm]
          exact hsome
        | true =>
          have ht' : sr.2.1 = #[] := by simpa using ht
          cases ha : sr.2.2.2 with
          | false =>
            left
            simp [layoutBCap, hs, ht', ha]
          | true =>
            right; left
            obtain ⟨s, hs', h', hmem, hj'⟩ := hR x st hst hf r j sr hr hs ht ha
            obtain ⟨c, hc⟩ := heldRows_circuit hmem
            refine ⟨s, hs', ?_⟩
            cases hfind : (heldRows r s).find? (·.1 == j) with
            | none => exact absurd (List.find?_eq_none.1 hfind _ hmem) (by simp [hj'])
            | some p => simp [layoutBCap, hs, ht', ha, hr, hc, hfind]
      · refine ⟨Or.inr (Or.inr ?_), fun h0 => ?_⟩
        · have hc : 2 * rs.size + 1 + 1 = layoutOf.rounds x := by
            rw [hL]
            omega
          simp only [layoutBCap, lastPieces, if_pos hc]
          exact hF x st hst hf
        · exfalso
          simp [layoutBCap, hsr'] at h0
  · rw [hkx, show 2 + 2 * rs.size - 1 = 2 * rs.size + 1 by omega]
    simp [layoutBCap, hsr']

theorem not_heldB_layoutBCap_none {W : Type} (IV : Claims.Verifier Coin) (ivScope : Msg → Bool) (b : Brings)
    (I : Claims.Recursion.Inner Coin W) (hk : ∀ x, I.V.k x = (FlockDeploy.V false).k x)
    (fin : Msg → List (ℕ × List (Msg × List HmNets.Lin))) (shape : Claims.Recursion.Form → Prop)
    (hM : SchedMatches shape) {x : Msg} {st : Claims.Recursion.CompiledStatement} (hst : parse x = some st) (hf : shape st.form)
    {j : ℕ} {sr : RecVStar.SchedRound} (hs : streamRound x j = some sr) (ht : sr.2.1.isEmpty = false) :
    ¬ Claims.Recursion.VStarB.HeldB (deployRec IV ivScope b) I (layoutBCap (fun _ _ _ => none) fin) shape := by
  intro hH
  obtain ⟨r, rs, hr, hrs, hK⟩ := hM x st hst hf
  have hj : j < 2 * rs.size := by
    by_contra hj
    simp [streamRound, hr, hrs, hj] at hs
  have ht' : ¬ sr.2.1 = #[] := by simpa using ht
  obtain ⟨-, hcl, -⟩ := hH x st hst hf
  have hkx : I.V.k (Claims.Recursion.xI (deployRec IV ivScope b) x) = 2 + 2 * rs.size := by
    rw [hk]
    exact hK
  rcases (hcl (j + 1) (by rw [hkx]; omega)).1 with h0 | ⟨s, -, h0⟩ | ⟨h0, -⟩
  · simp [layoutBCap, hs, ht'] at h0
  · simp [layoutBCap, hs, ht'] at h0
  · apply h0
    have hc : ¬ (j + 1 + 1 = layoutOf.rounds x) := by
      show ¬ (j + 1 + 1 = (FlockDeploy.V false).k (xIOf x))
      rw [hK]
      omega
    simp only [layoutBCap, lastPieces, if_neg hc]

/-- `TopsHeld` asks for a part that holds a tops round whole, which none does; `ksLin2_deployed_tops` takes the
same facts but that one. -/
theorem ksLin2_deployed_cap (b : Brings) (topsAt : Msg → ℕ → ℕ → Option (List (Msg × List HmNets.Lin)))
    (fin : Msg → List (ℕ × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ) (join : List Msg → Msg)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) :
    let L := layoutBCap topsAt fin
    let Bd := bdDeployed _root_.Flock.Readable.relation (FlockDeploy.V false).next
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation b L Bd.IVStar
    let bL : Brings := { b with rd := L.readers }
    (∃ decC : Msg → Msg → Coin,
      (∀ decI, Claims.Recursion.VStar.RunPairs FlockDeploy.sha512 D
        (Claims.Recursion.Inner.ofFlock Bd.deploy decI) decC) ∧
      ∀ shape ∈ [Claims.Recursion.Form.Dense],
        SchedMatches shape ∧ CapPorts shape ∧ RowsCover shape ∧ TopsHeld topsAt shape ∧ FinPieces fin shape ∧
        ∃ (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop)
          (cmt : Msg → ℕ → Msg → Msg),
          (∀ decI, Claims.Recursion.VStarB.VstarShowsB D (Claims.Recursion.Inner.ofFlock Bd.deploy decI) L shape
            decC shows cmt) ∧
          (∀ s, Claims.Recursion.VStarB.PartChecksBV FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
            L shape shows cmt chk s) ∧
          (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
            L shape shows cmt s) ∧
          Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
    ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    ReadersRunPlayed FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation b L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt := by
  intro L Bd D bL hP hRT hRP
  refine ksLin2_deployed_plain_closed b L (Claims.Recursion.VStarB.LayoutB.readsRows_of_rowIn fun _ _ => rfl) wE join
    final w ?_ hRT hRP
  obtain ⟨decC, hRun, hS⟩ := hP
  refine ⟨decC, hRun, fun shape hsh => ?_⟩
  obtain ⟨hM, hC, hRw, hTp, hF, chk, shows, cmt, hV, hPC, hPO, hD⟩ := hS shape hsh
  exact ⟨chk, shows, cmt,
    fun decI => heldB_layoutBCap _ _ _ _ (fun _ => rfl) topsAt fin shape hM hC hRw hTp hF, hV, hPC, hPO, hD⟩

end RecDeploy
