module

public import Spot.Docket
public import Proofs.Flock.Claims.Sampling.Reference.Deployed
public import Proofs.Flock.Claims.Sampling.Reg
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

/-! The sampled verifier's code with its checked units the Lean sampled verifier's sample of record and one session per registration. -/

@[expose] public section

namespace SamplingDeploy

open Claims Claims.Sampling Claims.Sampling.Reference Properties.Spot
open Lean (Json)

def code (c : Code) (reg : Msg → Option Reg) (L : Msg → ℕ)
    (bytes : (x : Msg) → (Fin (c.V.k x) → FlockDeploy.Coin) → Fin (L x) → Fin 256)
    (sr : (x : Msg) → Finset (Fin (c.V.k x))) (sess : Msg → Seen → Option (Msg × Seen)) : Code :=
  { c with
    checked := fun x r => ((reg x).map fun g => g.checked (bytesOf (bytes x r))).getD []
    rounds := fun x _ => sr x
    sessions := fun _ => 1
    session := fun x _ v => sess x v
    unitN₀ := fun x _ => c.innerN₀ x }

theorem sessionSlots_of_one (H : Msg → Msg) (d : Code) (hn : ∀ x, 1 ≤ d.sessions x)
    (hs : ∀ x u u' v, d.session x u v = d.session x u' v) : SessionSlots d H := by
  intro x _ Ω P ω r u hu
  cases hc : d.checked x r with
  | nil => rw [hc] at hu; simp at hu
  | cons u' l => exact ⟨0, hn x, u', rfl, hs x u u' _⟩

variable (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (c : Code) (reg : Msg → Option Reg)
  (L : Msg → ℕ) (bytes : (x : Msg) → (Fin (c.V.k x) → FlockDeploy.Coin) → Fin (L x) → Fin 256)
  (sr : (x : Msg) → Finset (Fin (c.V.k x))) (sess : Msg → Seen → Option (Msg × Seen))

theorem code_checked (x : Msg) (r : Fin (c.V.k x) → FlockDeploy.Coin) :
    (code c reg L bytes sr sess).checked x r = ((reg x).map fun g => g.checked (bytesOf (bytes x r))).getD [] :=
  rfl

theorem code_session (x : Msg) (u : ℕ) (v : Seen) : (code c reg L bytes sr sess).session x u v = sess x v := rfl

theorem sessionSlots_code (H : Msg → Msg) : SessionSlots (code c reg L bytes sr sess) H :=
  sessionSlots_of_one H _ (fun _ => le_rfl) fun _ _ _ _ => rfl

theorem δc_code (H : Msg → Msg) (x : Msg) :
    (deploy rel (code c reg L bytes sr sess)).δc H x =
      sessErrAt H (Rhon H (commit c) c.parse c.partition)
        (refP FlockComplete.ZkTables.honestPR c.step c.parse c.reads (commit c) c.wit) c.V
        (code c reg L bytes sr sess).checked (fun x _ v => sess x v) (FlockComplete.ZkTables.δc H) x := by
  show ((1 : ℕ) : ℝ) * _ = _
  rw [Nat.cast_one, one_mul]
  rfl

def RegMatches : Prop :=
  ∀ x s Pt, c.parse x = some s → c.partition s = some Pt →
    ∃ g l, reg x = some g ∧ Spot.Program.parse ⟨s.desc.toArray⟩ = .ok g.program ∧
      Spot.Slots.layout g.program g.query = .ok l ∧ Pt = slotPartition l ∧
      slotDesign g.rule g.program g.query g.table = .ok s.design

def BytesUniform : Prop :=
  ∀ x (E : (Fin (L x) → Fin 256) → Prop), Pr (fun r : Fin (c.V.k x) → FlockDeploy.Coin => E (bytes x r)) = Pr E

def BytesFixed : Prop :=
  ∀ x, sr x ⊆ c.secondStage x ∧ ∀ r d, bytes x (forkOn (sr x) r d) = bytes x r

theorem sampleSlots_code (hreg : RegMatches c reg) (hunif : BytesUniform c L bytes) :
    DrawSlots (deploy rel (code c reg L bytes sr sess)) := by
  intro x s Pt hx hP
  obtain ⟨g, l, hg, hd, hl, hPt, hds⟩ := hreg x s Pt hx hP
  refine ⟨g.rule, g.program, g.query, g.table, g.closure, l, L x, bytes x, hd, hl, hPt ▸ rfl, hds, hunif x,
    fun r => ?_⟩
  show ((reg x).map fun g => g.checked (bytesOf (bytes x r))).getD [] = _
  rw [hg]
  simp only [Option.map_some, Option.getD_some, Reg.checked, Reg.layout, hl]

theorem sampleFixed_code (h : BytesFixed c L bytes sr) : SampleFixed (deploy rel (code c reg L bytes sr sess)) := by
  intro x r u _
  refine ⟨(h x).1, fun d => ?_⟩
  show ((reg x).map fun g => g.checked (bytesOf (bytes x (forkOn (sr x) r d)))).getD [] =
    ((reg x).map fun g => g.checked (bytesOf (bytes x r))).getD []
  rw [(h x).2 r d]

def FramedOne (H : Msg → Msg) : Prop :=
  ∀ x {Ω : Type} (P : Prover Ω) (ω : Ω) (r : Fin (c.V.k x) → FlockDeploy.Coin),
    (code c reg L bytes sr sess).checked x r ≠ [] →
    ∃ (xS : Msg) (emb : Fin ((FlockDeploy.V true).k xS) → Fin (c.V.k x)), Function.Injective emb ∧
      (∀ i, emb i ∈ sr x) ∧
      ∀ {ΩF : Type} (PF : Prover ΩF) (ωF : ΩF),
        (∀ hist hS, c.step x hist = .session xS hS →
          ((P.next x hist ω).eval H).1 = ((PF.next xS hS ωF).eval H).1) →
        ∀ r' : Fin (c.V.k x) → FlockDeploy.Coin, (∀ t ∉ sr x, r' t = r t) →
          sess x (run H P c.V x ω r').seen =
            some (xS, (run H PF (FlockDeploy.V true) xS ωF fun i => r' (emb i)).seen)

def AcceptsOne (H : Msg → Msg) : Prop :=
  ∀ x {Ω : Type} (P : Prover Ω) (ω : Ω) (r : Fin (c.V.k x) → FlockDeploy.Coin),
    (∃ s Pt, c.parse x = some s ∧ c.partition s = some Pt ∧ InScope Pt s) →
    (∀ hist m, c.step x hist = .say m → ((P.next x hist ω).eval H).1 = m) →
    ((code c reg L bytes sr sess).checked x r ≠ [] →
      ∀ xS vS, sess x (run H P c.V x ω r).seen = some (xS, vS) → vS.out = .acc) →
      (run H P c.V x ω r).out = .acc

def SessScope (H : Msg → Msg) : Prop :=
  ∀ x s Pt, c.parse x = some s → c.partition s = some Pt → InScope Pt s →
    ∀ {Ω : Type} (P : Prover Ω) (ω : Ω) (r : Fin (c.V.k x) → FlockDeploy.Coin) xS vS,
      (code c reg L bytes sr sess).checked x r ≠ [] →
      sess x (run H P c.V x ω r).seen = some (xS, vS) → FlockDeploy.scopeCheck true xS = true

def SessWit (H : Msg → Msg) : Prop :=
  ∀ x s Pt w, c.parse x = some s → c.partition s = some Pt → InScope Pt s →
    faulty H (commit c) s Pt w = ∅ →
    ∀ {Ω : Type} (P : Prover Ω) (ω : Ω) (r : Fin (c.V.k x) → FlockDeploy.Coin) xS vS,
      (code c reg L bytes sr sess).checked x r ≠ [] →
      sess x (run H P c.V x ω r).seen = some (xS, vS) →
      ∀ os : List (Opening Bool),
        (∀ o ∈ os, (commit c).Opens H s.roots o = true ∧ w.val H (commit c) s o.pos = some o.val) →
        (∀ g ∈ c.reads s xS, ∀ b, w.val H (commit c) s g = some b → ∃ o ∈ os, o.pos = g) →
          Claims.Flock.R (FlockDeploy.deployZk rel).parse xS (c.wit s xS os)

def SessIn : Prop :=
  ∀ x v xS vS, sess x v = some (xS, vS) →
    (∃ s spec sts, FlockDeploy.ZkIn xS s spec sts ∧ sts.size ≤ c.tables x) ∧ FlockDeploy.N₀ true xS = c.innerN₀ x

theorem framed_code_iff (H : Msg → Msg) : Framed (code c reg L bytes sr sess) H ↔ FramedOne c reg L bytes sr sess H := by
  constructor
  · intro h x Ω P ω r hne
    obtain ⟨u, hu⟩ := List.exists_mem_of_ne_nil _ hne
    exact h x P ω r u hu
  · intro h x Ω P ω r u hu
    exact h x P ω r (List.ne_nil_of_mem hu)

theorem acceptsFramed_code_iff (H : Msg → Msg) :
    AcceptsFramed (code c reg L bytes sr sess) H ↔ AcceptsOne c reg L bytes sr sess H := by
  constructor
  · intro h x Ω P ω r hx hsay hs
    refine h x P ω r hx hsay fun u hu xS vS hv => hs (List.ne_nil_of_mem hu) xS vS hv
  · intro h x Ω P ω r hx hsay hs
    refine h x P ω r hx hsay fun hne xS vS hv => ?_
    obtain ⟨u, hu⟩ := List.exists_mem_of_ne_nil _ hne
    exact hs u hu xS vS hv

theorem sessionsScope_code_iff (H : Msg → Msg) :
    SessionsScope (code c reg L bytes sr sess) H ↔ SessScope c reg L bytes sr sess H := by
  constructor
  · intro h x s Pt hx hP hin Ω P ω r xS vS hne hv
    obtain ⟨u, hu⟩ := List.exists_mem_of_ne_nil _ hne
    exact h x s Pt hx hP hin P ω r u xS vS hu hv
  · intro h x s Pt hx hP hin Ω P ω r u xS vS hu hv
    exact h x s Pt hx hP hin P ω r xS vS (List.ne_nil_of_mem hu) hv

theorem sessionWit_code_iff (H : Msg → Msg) :
    SessionWit rel (code c reg L bytes sr sess) H ↔ SessWit rel c reg L bytes sr sess H := by
  constructor
  · intro h x s Pt w hx hP hin hf Ω P ω r xS vS hne hv
    obtain ⟨u, hu⟩ := List.exists_mem_of_ne_nil _ hne
    exact h x s Pt w hx hP hin hf P ω r u xS vS hu hv
  · intro h x s Pt w hx hP hin hf Ω P ω r u xS vS hu hv
    exact h x s Pt w hx hP hin hf P ω r xS vS (List.ne_nil_of_mem hu) hv

theorem sessionsIn_code_iff : SessionsIn (code c reg L bytes sr sess) ↔ SessIn c sess := by
  constructor
  · intro h x v xS vS hv
    exact h x 0 v xS vS hv
  · intro h x _ v xS vS hv
    exact h x v xS vS hv

/-- Nothing in the Lean verifier establishes this at `sess`: its registration refusal fixes the sampled count only, and its
outcome step never reads the session's tables; at the session the shape check lets through it holds (`shapesFixed_checked`). -/
def ShapesFixed (shape : Msg → List (Bool × ℕ)) : Prop :=
  ∀ x v xS vS, sess x v = some (xS, vS) → (FlockDeploy.tables true xS).map (fun p => (p.2.wideLeaf, p.2.m)) = shape x

noncomputable def shapeN₀ (ms : List (Bool × ℕ)) : ℝ :=
  (ms.map fun p => (2 : ℝ) ^ (Claims.Flock.zkLevel0 p.1 p.2).d).sum

theorem N₀_of_shapes (shape : Msg → List (Bool × ℕ)) (h : ShapesFixed sess shape) (x : Msg) (v : Seen) (xS : Msg) (vS : Seen)
    (hv : sess x v = some (xS, vS)) : FlockDeploy.N₀ true xS = shapeN₀ (shape x) := by
  rw [← h x v xS vS hv, shapeN₀, List.map_map]
  rfl

theorem sessIn_of_shapes (shape : Msg → List (Bool × ℕ))
    (hin : ∀ x v xS vS, sess x v = some (xS, vS) → ∃ s spec sts, FlockDeploy.ZkIn xS s spec sts ∧ sts.size ≤ c.tables x)
    (h : ShapesFixed sess shape) (hN : ∀ x, c.innerN₀ x = shapeN₀ (shape x)) : SessIn c sess :=
  fun x v xS vS hv => ⟨hin x v xS vS hv, (N₀_of_shapes sess shape h x v xS vS hv).trans (hN x).symm⟩

def shapeChecked (shape : Msg → List (Bool × ℕ)) (x : Msg) (v : Seen) : Option (Msg × Seen) :=
  (sess x v).filter fun p => decide ((FlockDeploy.tables true p.1).map (fun q => (q.2.wideLeaf, q.2.m)) = shape x)

theorem shapeChecked_some (shape : Msg → List (Bool × ℕ)) {x : Msg} {v : Seen} {p : Msg × Seen}
    (h : shapeChecked sess shape x v = some p) :
    sess x v = some p ∧ (FlockDeploy.tables true p.1).map (fun q => (q.2.wideLeaf, q.2.m)) = shape x := by
  unfold shapeChecked at h
  cases hs : sess x v with
  | none => rw [hs] at h; cases h
  | some q =>
    simp only [hs, Option.filter_some] at h
    split at h
    · rename_i hq
      cases h
      exact ⟨rfl, of_decide_eq_true hq⟩
    · cases h

theorem shapesFixed_checked (shape : Msg → List (Bool × ℕ)) : ShapesFixed (shapeChecked sess shape) shape :=
  fun _ _ _ _ hv => (shapeChecked_some sess shape hv).2

theorem shapesFixed_iff_checked (shape : Msg → List (Bool × ℕ)) :
    ShapesFixed sess shape ↔ shapeChecked sess shape = sess := by
  constructor
  · intro h
    funext x v
    unfold shapeChecked
    cases hs : sess x v with
    | none => rfl
    | some p =>
      obtain ⟨xS, vS⟩ := p
      simp only [Option.filter_some, decide_eq_true (h x v xS vS hs), ite_true]
  · intro h
    rw [← h]
    exact shapesFixed_checked sess shape

theorem sessIn_checked (shape : Msg → List (Bool × ℕ))
    (hin : ∀ x v xS vS, sess x v = some (xS, vS) → ∃ s spec sts, FlockDeploy.ZkIn xS s spec sts ∧ sts.size ≤ c.tables x)
    (hN : ∀ x, c.innerN₀ x = shapeN₀ (shape x)) : SessIn c (shapeChecked sess shape) :=
  sessIn_of_shapes c (shapeChecked sess shape) shape
    (fun x v xS vS hv => hin x v xS vS (shapeChecked_some sess shape hv).1) (shapesFixed_checked sess shape) hN

set_option synthInstance.maxSize 1024 in
theorem complete_code (H : Msg → Msg) (hF : Claims.Flock.Complete H (FlockDeploy.deployZk rel))
    (hfixed : BytesFixed c L bytes sr) (hreg : RegMatches c reg) (hunif : BytesUniform c L bytes)
    (hframe : FramedOne c reg L bytes sr sess H) (haccept : AcceptsOne c reg L bytes sr sess H)
    (hscope : SessScope c reg L bytes sr sess H) (hwit : SessWit rel c reg L bytes sr sess H) :
    Complete H (deploy rel (code c reg L bytes sr sess)) :=
  complete_deploy rel _ H hF (sampleFixed_code rel c reg L bytes sr sess hfixed)
    (sampleSlots_code rel c reg L bytes sr sess hreg hunif) ((framed_code_iff c reg L bytes sr sess H).2 hframe)
    ((acceptsFramed_code_iff c reg L bytes sr sess H).2 haccept)
    ((sessionsScope_code_iff c reg L bytes sr sess H).2 hscope)
    ((sessionWit_code_iff rel c reg L bytes sr sess H).2 hwit) (sessionSlots_code c reg L bytes sr sess H)

end SamplingDeploy
