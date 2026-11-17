module

public import Proofs.Properties.Verity.Sampled.Window.Read
public import Proofs.Properties.Verity.Sampled.Window.UnboundLaid
public import Proofs.Properties.Verity.Sampled.Instance
public import Proofs.Properties.Verity.Sampled.MemberForm.SetsUp

/-! SPOT's vocabulary as deployed over windows of several members. -/

@[expose] public section

namespace Verity.Sampled

open Verity.Game
open scoped ENNReal

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

theorem ExecutionPhase.delta_extend (W : ExecutionPhase) (S : Widening) (t : ℕ) (cr : ℝ) : (W.extend S).delta t cr = W.delta t cr := by
  unfold ExecutionPhase.delta
  rw [W.sessionDelta_extend S t]
  rfl

theorem ExecutionPhase.ofReal_le_delta (W : ExecutionPhase) (t : ℕ) (cr : ℝ) : ENNReal.ofReal cr ≤ W.delta t cr := by
  unfold ExecutionPhase.delta
  exact le_add_left le_rfl

theorem get_map_of {α β : Type*} (f : α → β) (P : ℕ → β → Prop) :
    ∀ (l : List α), (∀ j : Fin l.length, P j.1 (f (l.get j))) → ∀ k : Fin (l.map f).length, P k.1 ((l.map f).get k)
  | [], _, k => absurd k.2 (Nat.not_lt_zero _)
  | _ :: _, h, ⟨0, _⟩ => h ⟨0, Nat.succ_pos _⟩
  | _ :: l, h, ⟨k + 1, hk⟩ =>
    get_map_of f (fun j => P (j + 1)) l (fun j => h ⟨j.1 + 1, Nat.succ_lt_succ j.2⟩) ⟨k, Nat.lt_of_succ_lt_succ hk⟩

theorem Member.map_n_toV5 {nJ : ℕ} (t : ℕ) : ∀ l : List (Member nJ),
    ((l.map fun m => m.toV5 t).map fun m' => m'.n) = l.map fun m => m.n
  | [] => rfl
  | m :: l => congrArg (m.n :: ·) (Member.map_n_toV5 t l)

theorem get_of_map {α β : Type*} (f : α → β) (P : ℕ → β → Prop) :
    ∀ (l : List α), (∀ k : Fin (l.map f).length, P k.1 ((l.map f).get k)) → ∀ j : Fin l.length, P j.1 (f (l.get j))
  | [], _, j => absurd j.2 (Nat.not_lt_zero _)
  | _ :: _, h, ⟨0, _⟩ => h ⟨0, Nat.succ_pos _⟩
  | _ :: l, h, ⟨j + 1, hj⟩ =>
    get_of_map f (fun j => P (j + 1)) l (fun k => h ⟨k.1 + 1, Nat.succ_lt_succ k.2⟩) ⟨j, Nat.lt_of_succ_lt_succ hj⟩

theorem ExecutionPhase.counts_joint (W : ExecutionPhase) (t : ℕ) : (W.joint t).counts = W.ms.map fun m => m.n :=
  Member.map_n_toV5 t W.ms

theorem ExecutionPhase.Laid.joint {W : ExecutionPhase} (h : W.Laid) (t : ℕ) : (W.joint t).Laid where
  total := by
    rw [W.counts_joint t, ← W.counts_joint 0]
    exact h.total
  emb k v := by
    have hb : ∀ (s : ℕ) (j : Fin (W.joint s).ms.length),
        (W.joint s).base j = MemberLayout.base (W.ms.map fun m => m.n) j.1 := fun s j =>
      congrArg (fun cnt => MemberLayout.base cnt j.1) (W.counts_joint s)
    have h0 : ∀ (j : Fin W.ms.length) (u : Fin (W.ms.get j).n),
        ((W.ms.get j).emb u).val = MemberLayout.base (W.ms.map fun m => m.n) j.1 + u.val :=
      get_of_map (fun m : Member W.nJ => m.toV5 0)
        (fun j m' => ∀ u : Fin m'.n, (m'.emb u).val = MemberLayout.base (W.ms.map fun m => m.n) j + u.val) W.ms
        fun j u => (h.emb j u).trans (congrArg (· + u.val) (hb 0 j))
    rw [hb t k]
    exact get_map_of (fun m : Member W.nJ => m.toV5 t)
      (fun j m' => ∀ u : Fin m'.n, (m'.emb u).val = MemberLayout.base (W.ms.map fun m => m.n) j + u.val)
      W.ms h0 k v

theorem ExecutionPhase.Laid.extend {W : ExecutionPhase} (h : W.Laid) (S : Widening) : (W.extend S).Laid :=
  ⟨h.total, h.emb⟩

/-- A deployed instance fills `signer`, `read` and `place` at its construction; no theorem takes any of them as
a hypothesis of its own. -/
structure ExecutionPhaseInstance (d : ResourceAccounting.Device ι) (α : Type) [Fintype α] where
  base : Vocab d α
  deployment : Device.Deployment
  signer : ∀ θ, base.signer θ = deployment.signer base.families θ
  /-- Every pinned table file the deployment can register at one of its sessions. -/
  tables : Array ByteArray
  read : ∀ θ F, base.read θ F = MemberForm.membersRead (PinnedTables.ofFiles tables) (base.registered θ) θ
  place : ∀ θ F c g, base.place θ F c g = placeOfFiles (base.read θ F) g
  costs : CostsRead base (Device.dev (deployment.setupOf base.families))
  window : ByteArray → Option ExecutionPhase

namespace ExecutionPhaseInstance

variable (I : ExecutionPhaseInstance d α)

/-- The deployment's pinned table lookup, derived from the table bytes it registers. -/
def pinned : PinnedTables := .ofFiles I.tables

def SetsUp (θ : ByteArray) (F : Array ByteArray) (W : ExecutionPhase) : Prop := MemberForm.MembersSetUp I.base I.pinned θ F W

theorem lb (θ : ByteArray) (F : Array ByteArray) (W : ExecutionPhase) (h : I.SetsUp θ F W) :
    W.Lb 0 = _root_.Spot.Challenger.MAX_COINS * 64 :=
  (show MemberForm.MembersSetUp I.base I.pinned θ F W from h).lb

theorem members (θ : ByteArray) (F : Array ByteArray) (W : ExecutionPhase) (h : I.SetsUp θ F W) : MembersAt W W.table :=
  (show MemberForm.MembersSetUp I.base I.pinned θ F W from h).toExecutionPhaseOn.membersAt

noncomputable abbrev dev : Exchange → Properties.Challenger.History :=
  Device.dev (I.deployment.setupOf I.base.families)

noncomputable def coinsReadOf : Subprotocol → Sent → Option (List Release)
  | .sampled => Verity.Sampled.coinsRead
  | a => I.base.coinsRead a

noncomputable def finderOf : Subprotocol → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) →
    Verity.Game.Strategy (I.base.game θ F c) → Claims.Finder
  | .sampled => executionPhaseReadFinder I.base I.pinned I.dev
  | a => I.base.finder a

noncomputable def forgerOf : Subprotocol → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) →
    Verity.Game.Strategy (I.base.game θ F c) → Hash.Assumptions.LmsForger
  | .sampled => Binding.forger I.base I.dev
  | a => I.base.forger a

def callsMulOf (θ : ByteArray) : Subprotocol → ℕ
  | .sampled => 2
  | a => I.base.callsMul θ a

def callsAddOf (θ : ByteArray) : Subprotocol → ℕ
  | .sampled => 0
  | a => I.base.callsAdd θ a

open Classical in
noncomputable def executionPhases (θ : ByteArray) (F : Array ByteArray) : Option ExecutionPhase :=
  (I.window θ).bind fun W => if I.SetsUp θ F W then some W else none

theorem executionPhases_some {θ : ByteArray} {F : Array ByteArray} {W : ExecutionPhase} (h : I.executionPhases θ F = some W) :
    I.window θ = some W ∧ I.SetsUp θ F W := by
  unfold executionPhases at h
  cases hW : I.window θ with
  | none => rw [hW] at h; cases h
  | some B =>
    rw [hW] at h
    dsimp only [Option.bind] at h
    split_ifs at h with hs
    cases h
    exact ⟨rfl, hs⟩

theorem executionPhases_of {θ : ByteArray} {F : Array ByteArray} {W : ExecutionPhase} (hW : I.window θ = some W)
    (hs : I.SetsUp θ F W) : I.executionPhases θ F = some W := by
  unfold executionPhases
  rw [hW]
  dsimp only [Option.bind]
  rw [if_pos hs]

variable (K : ExecutionPhaseConstructions I.base)

noncomputable def msgSpec (W : ExecutionPhase) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (j : Fin W.rounds) :
    FinderSpec (I.base.game θ F c) (I.base.cost θ F c) where
  finder σ := K.msgFinder W θ F c σ j
  budget t := 2 * t + 2 * W.h
  calls σ x := by
    have h := K.msg_calls W θ F c σ j x
    unfold ExecutionPhase.qM at h
    exact_mod_cast h
  claim := .cr

noncomputable def recordSpec (W : ExecutionPhase) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) :
    FinderSpec (I.base.game θ F c) (I.base.cost θ F c) where
  finder σ := K.recordFinder W θ F c σ
  budget t := 2 * t + 2 * W.h
  calls σ x := by
    have h := K.record_calls W θ F c σ x
    unfold ExecutionPhase.qM at h
    exact_mod_cast h
  claim := .cr

noncomputable def bindSpec (W : ExecutionPhase) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) :
    FinderSpec (I.base.game θ F c) (I.base.cost θ F c) where
  finder σ := K.bindFinder W θ F c σ
  budget t := 2 * t + 2 * W.h
  calls σ x := by
    have h := K.bind_calls W θ F c σ x
    unfold ExecutionPhase.qM at h
    exact_mod_cast h
  claim := .cr

noncomputable def strictSpecs (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) :
    List (FinderSpec (I.base.game θ F c) (I.base.cost θ F c)) :=
  match I.executionPhases θ F with
  | some W => (List.finRange W.rounds).map (I.msgSpec K W θ F c) ++ [I.recordSpec K W θ F c, I.bindSpec K W θ F c]
  | none => []

noncomputable def rootsSpec {nJ : ℕ} (m : Member nJ) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) :
    FinderSpec (I.base.game θ F c) (I.base.cost θ F c) where
  finder _ := m.rootsFinder
  budget _ := m.rootsCalls
  calls _ _ := Nat.le_refl _

noncomputable def rootsSpecs (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) :
    List (FinderSpec (I.base.game θ F c) (I.base.cost θ F c)) :=
  match I.executionPhases θ F with
  | some W => W.ms.map fun m => I.rootsSpec m θ F c
  | none => []

noncomputable def findersOf : Subprotocol → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) →
    List (FinderSpec (I.base.game θ F c) (I.base.cost θ F c))
  | .sampled => fun θ F c => I.strictSpecs K θ F c ++ I.rootsSpecs θ F c
  | a => I.base.finders a

def premisesOf : Subprotocol → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) →
    Verity.Game.Strategy (I.base.game θ F c) → Prop
  | .sampled => fun θ F c σ =>
    ∀ W, I.executionPhases θ F = some W → ∀ _ : Nonempty ((K.live W θ F c σ).joint (I.base.cost θ F c σ)).EndΩ,
      JointFinderCR ((K.live W θ F c σ).joint (I.base.cost θ F c σ)) (K.project W θ F c σ)
  | a => I.base.premises a

noncomputable def sampledBound (θ : ByteArray) (t : ℕ) : ℝ :=
  ((I.callsMulOf θ .sampled * t + I.callsAddOf θ .sampled : ℕ) : ℝ) / 2 ^ (256 : ℝ)

noncomputable def deltaOf (θ : ByteArray) : Subprotocol → ℕ → ℝ≥0∞
  | .sampled, t => ((I.window θ).map fun W => W.delta t (I.sampledBound θ t)).getD ⊤
  | a, t => I.base.delta θ a t

theorem deltaOf_some {θ : ByteArray} {W : ExecutionPhase} (h : I.window θ = some W) (t : ℕ) :
    I.deltaOf θ .sampled t = W.delta t (I.sampledBound θ t) := by
  simp [deltaOf, h]

theorem deltaOf_none {θ : ByteArray} (h : I.window θ = none) (t : ℕ) : I.deltaOf θ .sampled t = ⊤ := by
  simp [deltaOf, h]

noncomputable def lossOf : Subprotocol → ByteArray → ℝ≥0∞ → ℝ≥0∞
  | .sampled => fun _ _ => 0
  | a => I.base.loss a

theorem finderOf_calls : ∀ a θ F c σ (x : (I.finderOf a θ F c σ).S),
    (I.finderOf a θ F c σ).cost x ≤ I.callsMulOf θ a * I.base.cost θ F c σ + I.callsAddOf θ a
  | .sampled, θ, F, c, σ, x =>
    calc (I.finderOf .sampled θ F c σ).cost x
        = I.base.cost θ F c σ + (Binding.breakCalls (I.dev (Verity.Game.Tape.out _ σ x))
            (Verity.Game.Tape.out _ σ x).sent + (Verity.Game.Tape.out _ σ x).sent.blobs.size) := Nat.add_assoc _ _ _
      _ ≤ I.base.cost θ F c σ + I.base.cost θ F c σ := Nat.add_le_add_left (I.costs θ F c σ x) _
      _ = 2 * I.base.cost θ F c σ + 0 := by omega
  | .pouw, θ, F, c, σ, x => I.base.finder_calls .pouw θ F c σ x
  | .inferenceOnly, θ, F, c, σ, x => I.base.finder_calls .inferenceOnly θ F c σ x
  | .pous, θ, F, c, σ, x => I.base.finder_calls .pous θ F c σ x
  | .poub, θ, F, c, σ, x => I.base.finder_calls .poub θ F c σ x
  | .network, θ, F, c, σ, x => I.base.finder_calls .network θ F c σ x
  | .wipe, θ, F, c, σ, x => I.base.finder_calls .wipe θ F c σ x

theorem forgerOf_calls : ∀ a θ F c σ (x : (I.forgerOf a θ F c σ).S),
    (I.forgerOf a θ F c σ).cost x ≤ I.callsMulOf θ a * I.base.cost θ F c σ + I.callsAddOf θ a
  | .sampled, θ, F, c, σ, x =>
    calc (I.forgerOf .sampled θ F c σ).cost x
        = I.base.cost θ F c σ + Binding.breakCalls (I.dev (Verity.Game.Tape.out _ σ x))
            (Verity.Game.Tape.out _ σ x).sent := rfl
      _ ≤ I.base.cost θ F c σ + I.base.cost θ F c σ :=
          Nat.add_le_add_left ((Nat.le_add_right _ _).trans (I.costs θ F c σ x)) _
      _ = 2 * I.base.cost θ F c σ + 0 := by omega
  | .pouw, θ, F, c, σ, x => I.base.forger_calls .pouw θ F c σ x
  | .inferenceOnly, θ, F, c, σ, x => I.base.forger_calls .inferenceOnly θ F c σ x
  | .pous, θ, F, c, σ, x => I.base.forger_calls .pous θ F c σ x
  | .poub, θ, F, c, σ, x => I.base.forger_calls .poub θ F c σ x
  | .network, θ, F, c, σ, x => I.base.forger_calls .network θ F c σ x
  | .wipe, θ, F, c, σ, x => I.base.forger_calls .wipe θ F c σ x

theorem forgerOf_signed : ∀ a θ F c σ (x : (I.forgerOf a θ F c σ).S),
    ((I.forgerOf a θ F c σ).signed x).length ≤ I.base.signatures θ
  | .sampled, θ, F, c, σ, x => Binding.forger_signed_le I.base I.dev θ F c σ x
  | .pouw, θ, F, c, σ, x => I.base.forger_signed .pouw θ F c σ x
  | .inferenceOnly, θ, F, c, σ, x => I.base.forger_signed .inferenceOnly θ F c σ x
  | .pous, θ, F, c, σ, x => I.base.forger_signed .pous θ F c σ x
  | .poub, θ, F, c, σ, x => I.base.forger_signed .poub θ F c σ x
  | .network, θ, F, c, σ, x => I.base.forger_signed .network θ F c σ x
  | .wipe, θ, F, c, σ, x => I.base.forger_signed .wipe θ F c σ x

noncomputable def extractOf (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (I.base.game θ F c)) : (I.base.read θ F).circuit.Trace :=
  open Classical in
  match I.executionPhases θ F with
  | some W =>
    if _ : Nonempty ((K.live W θ F c σ).joint (I.base.cost θ F c σ)).EndΩ then
      Flock.Guarantees.tracesWith (I.base.read θ F).first (I.base.read θ F).rest
        (((K.live W θ F c σ).joint (I.base.cost θ F c σ)).seats (K.project W θ F c σ))
    else fun _ => 0
  | none => I.base.extract θ F c σ

theorem extractOf_some {θ : ByteArray} {F : Array ByteArray} {W : ExecutionPhase} (hW : I.executionPhases θ F = some W)
    (c : ByteArray) (σ : Verity.Game.Strategy (I.base.game θ F c))
    [he : Nonempty ((K.live W θ F c σ).joint (I.base.cost θ F c σ)).EndΩ] :
    I.extractOf K θ F c σ = Flock.Guarantees.tracesWith (I.base.read θ F).first (I.base.read θ F).rest
      (((K.live W θ F c σ).joint (I.base.cost θ F c σ)).seats (K.project W θ F c σ)) := by
  unfold extractOf
  rw [hW]
  exact dite_eq_left he

noncomputable def vocab : Vocab d α :=
  { I.base with
    coinsRead := I.coinsReadOf
    extract := I.extractOf K
    finder := I.finderOf
    callsMul := I.callsMulOf
    callsAdd := I.callsAddOf
    finder_calls := I.finderOf_calls
    finders := I.findersOf K
    premises := I.premisesOf K
    forger := I.forgerOf
    forger_calls := I.forgerOf_calls
    forger_signed := I.forgerOf_signed
    delta := I.deltaOf
    loss := I.lossOf }

noncomputable def liftOf (K' : ExecutionPhaseConstructions I.base) : ExecutionPhaseConstructions (I.vocab K) where
  coins := K'.coins
  project := K'.project
  readFinder := K'.readFinder
  forger := K'.forger
  msgFinder := K'.msgFinder
  recordFinder := K'.recordFinder
  bindFinder := K'.bindFinder
  msg_calls := K'.msg_calls
  record_calls := K'.record_calls
  bind_calls := K'.bind_calls

theorem strictCR {θ : ByteArray} {F : Array ByteArray} {c : ByteArray} {W : ExecutionPhase}
    {σ : Verity.Game.Strategy ((I.vocab K).game θ F c)} (hW : I.executionPhases θ F = some W)
    (h : (I.vocab K).finderCR .sampled θ F c σ) :
    (∀ j, Claims.StrictBirthday Verity.sha512Msg (K.msgFinder W θ F c σ j)) ∧
      Claims.StrictBirthday Verity.sha512Msg (K.recordFinder W θ F c σ) ∧
      Claims.StrictBirthday Verity.sha512Msg (K.bindFinder W θ F c σ) := by
  have hall := List.forall_iff_forall_mem.1
    (show finderCROf (I.strictSpecs K θ F c ++ I.rootsSpecs θ F c) σ from h)
  have hs : I.strictSpecs K θ F c =
      (List.finRange W.rounds).map (I.msgSpec K W θ F c) ++ [I.recordSpec K W θ F c, I.bindSpec K W θ F c] := by
    unfold strictSpecs
    rw [hW]
  refine ⟨fun j => hall (I.msgSpec K W θ F c j) (List.mem_append_left _ ?_),
    hall (I.recordSpec K W θ F c) (List.mem_append_left _ ?_),
    hall (I.bindSpec K W θ F c) (List.mem_append_left _ ?_)⟩
  · rw [hs]
    exact List.mem_append_left _ (List.mem_map.2 ⟨j, List.mem_finRange j, rfl⟩)
  · rw [hs]
    exact List.mem_append_right _ (by simp)
  · rw [hs]
    exact List.mem_append_right _ (by simp)

theorem rootsCR {θ : ByteArray} {F : Array ByteArray} {c : ByteArray} {W : ExecutionPhase}
    {σ : Verity.Game.Strategy ((I.vocab K).game θ F c)} (hW : I.executionPhases θ F = some W)
    (h : (I.vocab K).finderCR .sampled θ F c σ) : W.RootsCR := by
  have hall := List.forall_iff_forall_mem.1
    (show finderCROf (I.strictSpecs K θ F c ++ I.rootsSpecs θ F c) σ from h)
  have hs : I.rootsSpecs θ F c = W.ms.map fun m => I.rootsSpec m θ F c := by
    unfold rootsSpecs
    rw [hW]
  intro m hm
  exact hall (I.rootsSpec m θ F c) (List.mem_append_right _ (by rw [hs]; exact List.mem_map.2 ⟨m, hm, rfl⟩))

def ClaimTransfers : Prop :=
  ∀ θ F W, I.executionPhases θ F = some W → W.RootsFix → ∀ (c : ByteArray) (σ : Verity.Game.Strategy (I.base.game θ F c)),
    ¬ ResourceAccounting.IntegrityHolds (I.base.partsOf θ F) (I.base.Opens θ F c) (I.extractOf K θ F c σ)
        (I.base.certOf θ).ε → ∀ _ : Nonempty ((K.live W θ F c σ).joint (I.base.cost θ F c σ)).EndΩ,
      (K.live W θ F c σ).ε * (K.live W θ F c σ).nJ <
        ((((K.live W θ F c σ).joint (I.base.cost θ F c σ)).incorrectSet (K.project W θ F c σ)).card : ℝ)

structure ClaimReads : Prop where
  laid : ∀ θ F W, I.executionPhases θ F = some W → W.Laid
  share : ∀ θ F W, I.executionPhases θ F = some W → (I.base.certOf θ).ε = W.ε
  units : ∀ θ F W, I.executionPhases θ F = some W → I.base.units θ F = W.nJ
  reads : ∀ θ F W, I.executionPhases θ F = some W → W.RootsFix →
    ∀ (c : ByteArray) (σ : Verity.Game.Strategy (I.base.game θ F c))
      (_ : Nonempty ((K.live W θ F c σ).joint (I.base.cost θ F c σ)).EndΩ),
    ((K.live W θ F c σ).joint (I.base.cost θ F c σ)).ReadsAt (I.base.read θ F) (I.base.Opens θ F c)
      (Flock.Guarantees.tracesWith (I.base.read θ F).first (I.base.read θ F).rest
        (((K.live W θ F c σ).joint (I.base.cost θ F c σ)).seats (K.project W θ F c σ)))
      (K.project W θ F c σ)

theorem claimTransfers (h : I.ClaimReads K) : I.ClaimTransfers K := by
  intro θ F W hW hR c σ hI he
  rw [I.extractOf_some K hW c σ] at hI
  have hcard : (ResourceAccounting.unbound (I.base.partsOf θ F) (I.base.Opens θ F c)
        (Flock.Guarantees.tracesWith (I.base.read θ F).first (I.base.read θ F).rest
          (((K.live W θ F c σ).joint (I.base.cost θ F c σ)).seats (K.project W θ F c σ)))).card ≤
      (((K.live W θ F c σ).joint (I.base.cost θ F c σ)).incorrectSet (K.project W θ F c σ)).card :=
    Flock.Guarantees.JointAuditV5.UnboundIncorrect.card_le
      (ExecutionPhase.unboundIncorrect (K.live W θ F c σ) (I.base.cost θ F c σ))
      (((h.laid θ F W hW).extend (K.coins W θ F c σ)).joint (I.base.cost θ F c σ))
      (I.base.read θ F) (I.base.Opens θ F c) _ (K.project W θ F c σ) (h.reads θ F W hW hR c σ he)
  have hlt : (I.base.certOf θ).ε * ((I.base.units θ F : ℕ) : ℝ) <
      ((ResourceAccounting.unbound (I.base.partsOf θ F) (I.base.Opens θ F c)
        (Flock.Guarantees.tracesWith (I.base.read θ F).first (I.base.read θ F).rest
          (((K.live W θ F c σ).joint (I.base.cost θ F c σ)).seats (K.project W θ F c σ)))).card : ℝ) :=
    not_le.1 hI
  have he : (K.live W θ F c σ).ε * ((K.live W θ F c σ).nJ : ℝ) =
      (I.base.certOf θ).ε * ((I.base.units θ F : ℕ) : ℝ) :=
    congrArg₂ (fun (e : ℝ) (n : ℕ) => e * (n : ℝ)) (h.share θ F W hW).symm (h.units θ F W hW).symm
  exact he.trans_lt (lt_of_lt_of_le hlt (Nat.cast_le.2 hcard))

theorem deployed (hKr : ∀ θ F c σ, K.readFinder θ F c σ = executionPhaseReadFinder I.base I.pinned I.dev θ F c σ)
    (hKf : ∀ θ F c σ, K.forger θ F c σ = Binding.forger I.base I.dev θ F c σ) (hC : I.ClaimReads K) :
    ExecutionPhaseDeployed (I.vocab K) (I.liftOf K K) I.executionPhases where
  finder θ F c σ := (hKr θ F c σ).symm
  forger θ F c σ := (hKf θ F c σ).symm
  loss _ _ := rfl
  claim θ F W h hR c σ := I.claimTransfers K hC θ F W h hR c σ
  finderCR θ F W h c σ hcr hpr := by
    obtain ⟨hm, hr, hb⟩ := I.strictCR K h hcr
    exact ⟨hpr W h, hm, hr, hb, I.rootsCR K h hcr⟩
  delta θ F W h c σ :=
    le_of_eq ((ExecutionPhase.delta_extend W _ _ _).trans (I.deltaOf_some (I.executionPhases_some h).1 _).symm)

def ExecutionPhaseAt (k : Verity.Registration.Composite) : Prop :=
  ∀ θ F W, I.window θ = some W → (∃ p, ParamsAt θ p) →
    (∃ c o, Verity.Game.Reaches (I.base.game θ F c) o ∧ I.base.Verdict k .sampled o ∧ TablesSent I.pinned o.sent ∧
      FilesSent I.base o) → I.SetsUp θ F W

theorem uncovered_le (k : Verity.Registration.Composite) (hWa : I.ExecutionPhaseAt k) {θ : ByteArray}
    {F : Array ByteArray} {W : ExecutionPhase} (hA : I.window θ = some W) (hP : ∃ p, ParamsAt θ p)
    (hS : I.executionPhases θ F = none) (c : ByteArray) (σ : Verity.Game.Strategy (I.base.game θ F c)) :
    Verity.Game.prob (fun o => I.base.Verdict k .sampled o) (I.base.game θ F c) σ ≤
      ENNReal.ofReal ((executionPhaseReadFinder I.base I.pinned I.dev θ F c σ).Pr Verity.sha512Msg) :=
  verdict_le_files I.base I.pinned k I.dev θ F c σ fun t hv ht hf => by
    have hs := hWa θ F W hA hP ⟨c, _, Verity.Game.Tape.reaches_out _ σ t, hv, ht, hf⟩
    rw [I.executionPhases_of hA hs] at hS
    cases hS

theorem vocab_verdict_le (k : Verity.Registration.Composite) {θ : ByteArray} {F : Array ByteArray}
    {c : ByteArray} (σ : Verity.Game.Strategy ((I.vocab K).game θ F c)) :
    Verity.Game.prob (fun o => (I.vocab K).Verdict k .sampled o) ((I.vocab K).game θ F c) σ ≤
      Verity.Game.prob (fun o => I.base.Verdict k .sampled o) (I.base.game θ F c) σ :=
  Verity.Game.prob_mono (p := fun o => (I.vocab K).Verdict k .sampled o)
    (q := fun o => I.base.Verdict k .sampled o) (fun _ ho => ho) _ σ

theorem vocab_finder_le {θ : ByteArray} {F : Array ByteArray} {c : ByteArray}
    {σ : Verity.Game.Strategy ((I.vocab K).game θ F c)} (h : (I.vocab K).FinderCR .sampled θ F c σ) :
    (executionPhaseReadFinder I.base I.pinned I.dev θ F c σ).Pr Verity.sha512Msg ≤
      I.sampledBound θ ((I.vocab K).cost θ F c σ) := by
  have h' := (I.vocab K).finderPr_le_crBound h
  exact h'

theorem vocab_delta_ge {θ : ByteArray} {W : ExecutionPhase} (hA : I.window θ = some W) (t : ℕ) :
    ENNReal.ofReal (I.sampledBound θ t) ≤ (I.vocab K).delta θ .sampled t := by
  rw [show (I.vocab K).delta θ .sampled t = I.deltaOf θ .sampled t from rfl, I.deltaOf_some hA]
  exact W.ofReal_le_delta t _

theorem covers (k : Verity.Registration.Composite) (hWa : I.ExecutionPhaseAt k) :
    ExecutionPhaseCovers (I.vocab K) k I.executionPhases := by
  intro θ F hS hP c σ _ hFC
  rcases hA : I.window θ with _ | W
  · exact le_top.trans_eq (I.deltaOf_none hA _).symm
  exact (I.vocab_verdict_le K k σ).trans ((I.uncovered_le k hWa hA hP hS c σ).trans
    ((ENNReal.ofReal_le_ofReal (I.vocab_finder_le K hFC)).trans (I.vocab_delta_ge K hA _)))

def ExecutionPhaseRegistered : Prop :=
  ∀ θ W, I.window θ = some W → (∃ p, ParamsAt θ p) → MemberForm.ExecutionPhaseOn I.base I.pinned θ (I.base.registered θ) W

def ExecutionPhasePasses (k : Verity.Registration.Composite) : Prop :=
  ∀ θ F W, I.window θ = some W → MemberForm.ExecutionPhaseOn I.base I.pinned θ F W →
    (∃ c o, Verity.Game.Reaches (I.base.game θ F c) o ∧ I.base.Verdict k .sampled o ∧ TablesSent I.pinned o.sent ∧
      FilesSent I.base o) →
    W.Laid ∧ ∀ m ∈ W.ms, MemberForm.FilesPass m

theorem executionPhaseAt_members (k : Verity.Registration.Composite) (hW : I.ExecutionPhaseRegistered)
    (hpass : I.ExecutionPhasePasses k) : I.ExecutionPhaseAt k := by
  intro θ F W hw hP hplay
  have hon : MemberForm.ExecutionPhaseOn I.base I.pinned θ F W :=
    (hW θ W hw hP).of_read ((I.read θ _).trans (I.read θ F).symm)
  obtain ⟨hlaid, hpassed⟩ := hpass θ F W hw hon hplay
  exact (⟨hon, hlaid, hpassed⟩ : MemberForm.MembersSetUp I.base I.pinned θ F W)

noncomputable def constructionsAt (K₀ : ExecutionPhaseConstructions I.base) (k : Verity.Registration.Composite)
    (reg : Option Claims.Msg → ByteArray) (ev : ℕ → Exchange → Option Claims.Msg)
    (filesDec : Claims.Msg → Array ByteArray)
    (record₀ : ExecutionPhase → ByteArray → Array ByteArray → ByteArray → Exchange → Option Claims.Msg)
    (recordDigest : ExecutionPhase → ByteArray → Array ByteArray → ByteArray → Claims.Msg) : ExecutionPhaseConstructions I.base :=
  (K₀.withBinding I.pinned I.dev k).withProject I.base (sessionReadings ExecutionPhase.table ev record₀ recordDigest)
    (fun W _ _ _ _ _ => sessionDecoders reg W.table filesDec)
    (executionPhaseSampleMaps fieldEnc I.base (sessionReadings ExecutionPhase.table ev record₀ recordDigest)
      (fun θ _ _ _ => labelAt θ) (afterAt I.base k) fun W => W.srcCoinsAt _root_.Spot.Challenger.MAX_COINS)

def ExecutionPhaseReads (k : Verity.Registration.Composite) (ev : ℕ → Exchange → Option Claims.Msg)
    (filesDec : Claims.Msg → Array ByteArray) : Prop :=
  ∀ θ F W, I.executionPhases θ F = some W →
  ∀ (c : ByteArray) (σ : Strategy (I.base.game θ F c)) (z : CoinTape (I.base.game θ F c) σ),
    I.base.Verdict k .sampled (CoinTape.run (I.base.game θ F c) σ z) →
    Binding.Logged Binding.readsV2 (CoinTape.run (I.base.game θ F c) σ z) →
    TablesSent I.pinned (CoinTape.run (I.base.game θ F c) σ z).sent →
    FilesSent I.base (CoinTape.run (I.base.game θ F c) σ z) →
    SampledDigest k (CoinTape.run (I.base.game θ F c) σ z).sent (afterAt I.base k θ F c σ) →
    ExecutionPhaseRead W W.table ev filesDec (labelAt θ) (afterAt I.base k θ F c σ) _root_.Spot.Challenger.MAX_COINS
      (CoinTape.run (I.base.game θ F c) σ z)

/-- What is left is what the claim reads of the window (`ClaimReads`), the verifier's window at an accepted
play (`ExecutionPhaseAt`) and what the verifier's acceptance gives of each play (`ExecutionPhaseReads`). -/
theorem spotSound (K₀ : ExecutionPhaseConstructions I.base) (k : Verity.Registration.Composite)
    (reg : Option Claims.Msg → ByteArray) (ev : ℕ → Exchange → Option Claims.Msg)
    (filesDec : Claims.Msg → Array ByteArray)
    (record₀ : ExecutionPhase → ByteArray → Array ByteArray → ByteArray → Exchange → Option Claims.Msg)
    (recordDigest : ExecutionPhase → ByteArray → Array ByteArray → ByteArray → Claims.Msg)
    (hC : I.ClaimReads (I.constructionsAt K₀ k reg ev filesDec record₀ recordDigest))
    (hWa : I.ExecutionPhaseAt k) (hWR : I.ExecutionPhaseReads k ev filesDec) :
    (I.vocab (I.constructionsAt K₀ k reg ev filesDec record₀ recordDigest)).SpotSound k :=
  executionPhaseSpotSound_sessions (I.vocab (I.constructionsAt K₀ k reg ev filesDec record₀ recordDigest)) k
    (I.liftOf (I.constructionsAt K₀ k reg ev filesDec record₀ recordDigest) K₀) I.pinned I.dev reg ExecutionPhase.table ev filesDec
    record₀ recordDigest (fun θ _ _ _ => labelAt θ) _root_.Spot.Challenger.MAX_COINS
    (I.deployed (I.constructionsAt K₀ k reg ev filesDec record₀ recordDigest) (fun _ _ _ _ => rfl)
      (fun _ _ _ _ => rfl) hC)
    (I.covers (I.constructionsAt K₀ k reg ev filesDec record₀ recordDigest) k hWa)
    (fun _ _ _ _ o _ => Device.printsLog_dev _ o)
    (fun θ F _ _ c => Device.signsPlay_dev (I.vocab (I.constructionsAt K₀ k reg ev filesDec record₀ recordDigest))
      I.deployment θ F c (I.signer θ))
    (fun θ F W h => I.lb θ F W (I.executionPhases_some h).2)
    (fun θ F W h => I.members θ F W (I.executionPhases_some h).2)
    (fun θ F W h c σ z hv hl ht hf hd => hWR θ F W h c σ z hv hl ht hf hd)

end ExecutionPhaseInstance

theorem remote_soundness_executionPhase (I : ExecutionPhaseInstance d α) (K₀ : ExecutionPhaseConstructions I.base)
    (reg : Option Claims.Msg → ByteArray) (ev : ℕ → Exchange → Option Claims.Msg)
    (filesDec : Claims.Msg → Array ByteArray)
    (record₀ : ExecutionPhase → ByteArray → Array ByteArray → ByteArray → Exchange → Option Claims.Msg)
    (recordDigest : ExecutionPhase → ByteArray → Array ByteArray → ByteArray → Claims.Msg)
    (hC : I.ClaimReads (I.constructionsAt K₀ .remote reg ev filesDec record₀ recordDigest))
    (hWa : I.ExecutionPhaseAt .remote) (hWR : I.ExecutionPhaseReads .remote ev filesDec)
    (hW : I.base.PouwSound .remote)
    (hWi : (I.vocab (I.constructionsAt K₀ .remote reg ev filesDec record₀ recordDigest)).WiresSound .remote)
    (hG : GateARow) :
    (I.vocab (I.constructionsAt K₀ .remote reg ev filesDec record₀ recordDigest)).SoundnessAt .remote :=
  Verity.remote_soundness_of_subprotocols ι α d _
    (I.spotSound K₀ .remote reg ev filesDec record₀ recordDigest hC hWa hWR) hW hWi hG

end Verity.Sampled
