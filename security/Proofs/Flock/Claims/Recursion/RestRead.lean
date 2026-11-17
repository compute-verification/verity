module

public import Proofs.Flock.Claims.Recursion.NextR
public import Proofs.Flock.Claims.Recursion.VerdictPieces
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
import Proofs.Flock.Claims.Recursion.SchedDense

/-! The statement's share of `hrest` at `nextR`: the last message's reading, the rounds' lengths, and the record's reading. -/

@[expose] public section

namespace RecDeploy.RestRead

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP nextR nextR_coin ReadOkR)
open Claims.Recursion.VStarB

noncomputable section

def DenseX (x : Msg) : Prop := ∃ st, parse x = some st ∧ Claims.Recursion.Form.Dense st.form

/-- The inner rounds at a compiled statement: the deployed plain verifier's at its inner statement. -/
abbrev kI (x : Msg) : ℕ := (FlockDeploy.V false).k (xIOf x)

abbrev bdAt (rd : Msg → Option FinalP) (wE : Msg → ℝ) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) :=
  bdDeployed _root_.Flock.Readable.relation (nextR rd) (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols
    wE FlockDeploy.joinR rowInDeployed final w

abbrev layAt (fin : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) : LayoutB :=
  layoutBTops (finOpen fin)

abbrev splitAt (rd : Msg → Option FinalP) (wE : Msg → ℝ) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) {R : Type} (LS : LastSplitR FlockDeploy.sha512 rd R) :=
  deploySplit FlockDeploy.sha512 _root_.Flock.Readable.relation (nextR rd)
    (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE FlockDeploy.joinR rowInDeployed final w
    (nextR_coin rd FlockDeploy.sha512) LS.toLast

def lastAt (fin : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (x : Msg)
    (ws : List Claims.Flock.Wit) : Msg :=
  ((layAt fin).msgs x ws).getD (kI x - 1) []

def FinalReads (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (fin : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) : Prop :=
  ∀ x, DenseX x → ∀ ws : List Claims.Flock.Wit, ws.length = nS x →
    ∃ os last, final (lastAt fin x ws) = some (os, last) ∧ os.length + 1 = kI x ∧
      ∀ j < os.length, (os.getD j []).length = nAtSched (xIOf x) j

def RoundLens (fin : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) : Prop :=
  ∀ x, DenseX x → ∀ ws : List Claims.Flock.Wit, ws.length = nS x → ∀ j < kI x - 1,
    (((layAt fin).msgs x ws).getD j []).length = 128 * nAtSched (xIOf x) j

def ReadsAt (rd : Msg → Option FinalP) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (fin : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) : Prop :=
  ∀ x, DenseX x → ∀ (c : ℕ → Coin) (ws : List Claims.Flock.Wit), ws.length = nS x →
    (∀ e ∈ (layAt fin).allCalls x ws, FlockDeploy.sha512 e.1 = e.2) →
    ∀ os last, final (lastAt fin x ws) = some (os, last) → os.length + 1 = kI x →
      ReadOkR rd (xIOf x) (os.map (fun o => FlockDeploy.joinR (o.map Prod.fst)) ++ [last])
        ((List.range (kI x)).map c)

theorem kI_pos (x : Msg) : 0 < kI x := by
  unfold kI FlockDeploy.V
  dsimp only
  split <;> omega

section Checks

variable (rd : Msg → Option FinalP) (wE : Msg → ℝ) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
  (w : Msg → ℝ) (fin : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) {R : Type}
  (LS : LastSplitR FlockDeploy.sha512 rd R)

theorem hrest_shape (hF : FinalReads final fin) {x : Msg} (hx : DenseX x) (c : ℕ → Coin)
    {ws : List Claims.Flock.Wit} (hl : ws.length = nS x) :
    (IVCheck.shape : IVCheck (DeployCheck (ReadCheck R))).Holds FlockDeploy.sha512 (bdAt rd wE final w)
      (splitAt rd wE final w LS) (xIOf x) c ((layAt fin).msgs x ws) := by
  obtain ⟨os, last, hf, hos, hcnt⟩ := hF x hx ws hl
  exact ⟨kI_pos x, os, last, hf, hos, hcnt⟩

theorem hrest_len (hF : FinalReads final fin) (hL : RoundLens fin) {x : Msg} (hx : DenseX x) (c : ℕ → Coin)
    {ws : List Claims.Flock.Wit} (hl : ws.length = nS x) {j : ℕ} (hj : j < kI x - 1) :
    (IVCheck.len j : IVCheck (DeployCheck (ReadCheck R))).Holds FlockDeploy.sha512 (bdAt rd wE final w)
      (splitAt rd wE final w LS) (xIOf x) c ((layAt fin).msgs x ws) := by
  obtain ⟨os, last, hf, hos, hcnt⟩ := hF x hx ws hl
  intro os' last' hf'
  have hf'' : final (lastAt fin x ws) = some (os', last') := hf'
  rw [hf] at hf''
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj hf'')
  rw [hL x hx ws hl j hj]
  exact congrArg (128 * ·) (hcnt j (by omega)).symm

theorem hrest_read (hR : ReadsAt rd final fin) {x : Msg} (hx : DenseX x) (c : ℕ → Coin)
    {ws : List Claims.Flock.Wit} (hl : ws.length = nS x)
    (hT : ∀ e ∈ (layAt fin).allCalls x ws, FlockDeploy.sha512 e.1 = e.2) :
    (IVCheck.elem (DeployCheck.last ReadCheck.read) : IVCheck (DeployCheck (ReadCheck R))).Holds FlockDeploy.sha512
      (bdAt rd wE final w) (splitAt rd wE final w LS) (xIOf x) c ((layAt fin).msgs x ws) := by
  intro os last hf hos
  exact hR x hx c ws hl hT os last hf hos

theorem hrest_part (bB : Brings) (hF : FinalReads final fin) (hL : RoundLens fin) (hR : ReadsAt rd final fin) :
    ∀ (x : Msg) (st : Claims.Recursion.CompiledStatement),
      (recBound _root_.Flock.Readable.relation bB (layAt fin) (bdAt rd wE final w).IVStar).parse x = some st →
      Claims.Recursion.Form.Dense st.form →
      ∀ q ∈ ivChecks (bdAt rd wE final w) (splitAt rd wE final w LS)
          (Claims.Recursion.xI (recBound _root_.Flock.Readable.relation bB (layAt fin) (bdAt rd wE final w).IVStar) x),
        (q = .shape ∨ (∃ j, q = .len j) ∨ q = .elem (.last .read)) →
        ∀ (c : ℕ → Coin) (ws : List Claims.Flock.Wit),
          ws.length = (recBound _root_.Flock.Readable.relation bB (layAt fin) (bdAt rd wE final w).IVStar).nS x →
          (∀ e ∈ (layAt fin).allCalls x ws, FlockDeploy.sha512 e.1 = e.2) →
          q.Holds FlockDeploy.sha512 (bdAt rd wE final w) (splitAt rd wE final w LS)
            (Claims.Recursion.xI (recBound _root_.Flock.Readable.relation bB (layAt fin) (bdAt rd wE final w).IVStar) x)
            c ((layAt fin).msgs x ws) := by
  intro x st hx hd q hq hk c ws hl hT
  have hX : DenseX x := ⟨st, hx, hd⟩
  rcases hk with rfl | ⟨j, rfl⟩ | rfl
  · exact hrest_shape rd wE final w fin LS hF hX c hl
  · have hj : j < kI x - 1 := by
      rcases mem_ivChecks _ _ _ hq with h | ⟨j', hj', h⟩ | ⟨_, _, _, _, h | h⟩ | ⟨_, _, h⟩
      · cases h
      · cases h
        exact hj'
      · cases h
      · cases h
      · cases h
    exact hrest_len rd wE final w fin LS hF hL hX c hl hj
  · exact hrest_read rd wE final w fin LS hR hX c hl hT

end Checks

theorem firstHolder_last (D : Claims.Recursion.Deploy Coin)
    (fin : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) {x : Msg} (hx : DenseX x) :
    firstHolder D (layAt fin) x (kI x - 1) = none := by
  obtain ⟨st, hst, hd⟩ := hx
  obtain ⟨r, rs, hr, hrs, hK⟩ := schedMatches_dense x st hst hd
  have hk : kI x - 1 = 2 * rs.size + 1 := by
    show (FlockDeploy.V false).k (xIOf x) - 1 = _
    omega
  unfold firstHolder
  rw [hk, List.find?_eq_none]
  intro s _
  simp [layAt, layoutBTops, streamRound, hr, hrs]

end

end RecDeploy.RestRead
