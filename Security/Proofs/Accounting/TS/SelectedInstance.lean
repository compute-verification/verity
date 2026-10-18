import Definitions.Accounting.TS.Selected

/-!
# A padded window's structure is satisfiable (completeness instance)

note:20261018T2200Z-draft-padded-credit-rule §8.8, §9.5 and §10.7, round 4: one round, two calls `c`, each with an ok
unit `OkA c` reading its raw activation row, an A forming `FA c` and a B forming `FB c` (rowk's pieces collapsed to one
unit: their chain is `FormEdge`'s), the noise units `NA` (E_A's lines) and `NB` (E_B's lines and `F = F_A ‖ F_B`),
each deriving its noise in its own body (`noiseSelf`), a tile `T c` gated by `ok_A,c` and the registered `ok_B,c`, and
the tally `L`. Every unit but the tiles is always proved; draw slots 0–8 name them and slot 9 the tile the draw coin
picks. `structure_holds`: this wiring meets every protocol-structure condition of `ProofsSoundOfSelected`
(`WriterReads`, `SelectionsSourced`, `SelectionsProved`, `CertifiedOperands`, `FormingClosed`, `NoLateSelection`,
`NoiseProved`, `AnchorsPinned`, `AlwaysProved`, `TallyIn`, `GatesTallied`, `TallyReads`, `TallyNoAdvice`), with
`SelFan` at `F = 1`, while both tiles weigh 1 and are gated. The path-dependent hypotheses (`Of`, `JointSound`,
`SelCovers`, `SelectedFixed`) need a play whose fixed words are defined, which the statements don't have yet.
-/

namespace Accounting.SecurityProofs.TS.SelectedInstance

open Accounting.TS Pouw.RelativeSecurity

inductive U
  | okA (c : Bool)
  | fA (c : Bool)
  | fB (c : Bool)
  | nA
  | nB
  | tile (c : Bool)
  | L
  deriving DecidableEq, Fintype

inductive P
  | x (c : Bool)
  | okA (c : Bool)
  | b (c : Bool)
  | okB (c : Bool)
  | eA (c : Bool)
  | eB (c : Bool)
  | f
  | ta (c : Bool)
  | tb (c : Bool)
  | y (c : Bool)
  | out
  deriving DecidableEq, Fintype

/-- Each unit's five slots, its output last (repeated to five). -/
def slots : U → Fin 5 → P
  | .okA c, k => if k.val < 4 then .x c else .okA c
  | .fA c, k => (![P.x c, P.eA c, P.f, P.ta c, P.ta c] : Fin 5 → P) k
  | .fB c, k => (![P.b c, P.eB c, P.f, P.tb c, P.tb c] : Fin 5 → P) k
  | .nA, k => (![P.eA false, P.eA true, P.eA true, P.eA true, P.eA true] : Fin 5 → P) k
  | .nB, k => (![P.eB false, P.eB true, P.f, P.f, P.f] : Fin 5 → P) k
  | .tile c, k => (![P.ta c, P.tb c, P.okA c, P.okB c, P.y c] : Fin 5 → P) k
  | .L, k => (![P.okA false, P.okA true, P.okB false, P.okB true, P.out] : Fin 5 → P) k

def srcP : P → Option (Fin 1 × U)
  | .okA c => some (0, .okA c)
  | .eA _ => some (0, .nA)
  | .eB _ => some (0, .nB)
  | .f => some (0, .nB)
  | .ta c => some (0, .fA c)
  | .tb c => some (0, .fB c)
  | .y c => some (0, .tile c)
  | .out => some (0, .L)
  | _ => none

/-- The raw rows and the oks are cut before coin 0 (the noise's); everything else at the draw. -/
def cutP : P → Fin 1 × Fin 2
  | .x _ => (0, 0)
  | .okA _ => (0, 0)
  | .b _ => (0, 0)
  | .okB _ => (0, 0)
  | _ => (0, 1)

def provedList : Fin 9 → U :=
  ![.okA false, .okA true, .fA false, .fA true, .fB false, .fB true, .nA, .nB, .L]

def drawB (b : Bool) (i : Fin 10) : U := if h : i.val < 9 then provedList ⟨i.val, h⟩ else .tile b

def gatesU : U → Finset P
  | .tile c => {.okA c, .okB c}
  | _ => ∅

def floorU : U → Finset (Fin 5)
  | .tile _ => {0, 1}
  | _ => ∅

def noiseU : U → Finset (Fin 5)
  | .fA _ => {1, 2}
  | .fB _ => {1, 2}
  | .tile _ => {0, 1}
  | _ => ∅

def certRowP : P → P
  | .okA c => .x c
  | .okB c => .b c
  | p => p

def regdP : P → Option ℕ
  | .b c => some c.toNat
  | .okB c => some c.toNat
  | _ => none

noncomputable def wt (_ : Fin 1) : U → ℝ
  | .tile _ => 1
  | _ => 0

/-- The instance: §9.5's wiring, unit noise, no anchors, nothing weighted, no epoch record. -/
noncomputable def sl : SelectLayer Unit Bool 1 1 10 U (Fin 5) P Bool where
  draw := drawB
  pos := fun _ u k => slots u k
  coin := fun _ _ => 0
  cut := cutP
  src := srcP
  accepted := fun _ _ _ => True
  opened := fun _ _ _ _ => true
  rel := fun _ _ _ _ => True
  right := fun _ _ _ _ => True
  gates := fun _ => gatesU
  floor := fun _ => floorU
  certRow := certRowP
  forms := {(0, .fA false), (0, .fA true), (0, .fB false), (0, .fB true)}
  one := fun v => v = true
  proved := Finset.univ.image (fun i => ((0 : Fin 1), provedList i))
  tally := {(0, .L)}
  regd := regdP
  regRight := fun _ => True
  out := .out
  credOf := fun _ => 0
  noise := fun _ => noiseU
  anchor := fun _ => False
  noiseSelf := {(0, .nA), (0, .nB)}
  derive := fun _ _ => true
  weighted := ∅
  dw := fun _ _ => 0
  epoch := fun _ => none
  epochRow := fun _ => .out
  epochRight := fun _ => True
  advised := ∅

theorem first_step {α : Type} {r : α → α → Prop} {a b : α} (h : Relation.TransGen r a b) : ∃ c, r a c := by
  induction h with
  | single h => exact ⟨_, h⟩
  | tail _ _ ih => exact ih

theorem writerReads : sl.toDrawnLayer.WriterReads := by
  unfold DrawnLayer.WriterReads; decide

theorem selectionsSourced : sl.SelectionsSourced := by
  have h : ∀ w j, ∀ ℓ ∈ sl.gates w j, (∃ g, sl.src ℓ = some g) ∨ (sl.src ℓ = none ∧ (sl.regd ℓ).isSome) := by
    decide
  intro w j ℓ hℓ
  rcases h w j ℓ hℓ with h1 | ⟨h1, h2⟩
  · exact Or.inl h1
  · exact Or.inr ⟨h1, Option.isSome_iff_exists.mp h2⟩

theorem selectionsProved : sl.SelectionsProved := by
  have h : ∀ w j, ∀ ℓ ∈ sl.gates w j,
      (∃ g ∈ sl.proved, sl.src ℓ = some g) ∨ (sl.src ℓ = none ∧ (sl.regd ℓ).isSome) := by
    decide
  intro w j ℓ hℓ
  rcases h w j ℓ hℓ with h1 | ⟨h1, h2⟩
  · exact Or.inl h1
  · exact Or.inr ⟨h1, Option.isSome_iff_exists.mp h2⟩

theorem certifiedOperands : sl.CertifiedOperands := by
  have h : ∀ w j, ∀ k ∈ sl.floor w j, ∃ ℓ ∈ sl.gates w j,
      (∃ g, g ∈ sl.forms ∧ (g ∈ sl.proved ∨ g ∈ sl.weighted) ∧ sl.src (sl.pos w j k) = some g ∧
        sl.ReadsPos g (sl.certRow ℓ)) ∧
      ((∃ c, sl.src ℓ = some c ∧ sl.ReadsPos c (sl.certRow ℓ)) ∨
        (∃ r, sl.regd ℓ = some r ∧ sl.regd (sl.certRow ℓ) = some r)) := by
    unfold SelectLayer.ReadsPos; decide
  intro w j k hk
  obtain ⟨ℓ, hℓ, ⟨g, h1, h2, h3, h4⟩, h5⟩ := h w j k hk
  exact ⟨ℓ, hℓ, Or.inl ⟨g, h1, h2, h3, g, Relation.ReflTransGen.refl, h4⟩, h5⟩

theorem formingClosed : sl.FormingClosed := by
  unfold SelectLayer.FormingClosed; decide

theorem noLateSelection : sl.NoLateSelection := by
  unfold SelectLayer.NoLateSelection SelectLayer.AfterCoin; decide

theorem noiseProved : sl.NoiseProved wt := by
  have h1 : ∀ w j, ∀ k ∈ sl.noise w j, (∃ g, sl.src (sl.pos w j k) = some g) ∨
      (sl.src (sl.pos w j k) = none ∧ sl.anchor (sl.pos w j k)) := by
    intro w j k hk
    left
    revert w j k hk
    decide
  have h2 : ∀ w j, (sl.floor w j).Nonempty → (sl.noise w j).Nonempty ∨ (w, j) ∈ sl.noiseSelf := by
    decide
  have h3 : ∀ g y, sl.NoiseIn g y → g ∈ sl.proved := by
    unfold SelectLayer.NoiseIn; decide
  refine ⟨h1, fun w j _ hf => h2 w j hf, fun g v _ _ _ _ hv _ => Or.inl ?_⟩
  obtain ⟨y, hy⟩ := first_step hv
  exact h3 g y hy

theorem anchorsPinned : sl.AnchorsPinned :=
  ⟨fun _ _ _ h => h.elim, fun _ _ _ _ _ h => h.elim⟩

theorem alwaysProved : sl.AlwaysProved := by
  unfold SelectLayer.AlwaysProved; decide

theorem tallyIn : sl.TallyIn := by
  unfold SelectLayer.TallyIn; decide

theorem gatesTallied : sl.GatesTallied := by
  unfold SelectLayer.GatesTallied; decide

theorem tallyReads : sl.TallyReads := by
  unfold SelectLayer.TallyReads; decide

theorem tallyNoAdvice : sl.TallyNoAdvice := by
  unfold SelectLayer.TallyNoAdvice; decide

theorem selFan : sl.SelFan wt 1 := by
  have h : ∀ g : Fin 1 × U, ¬ ((g ∉ sl.proved ∧ ∃ v, sl.Gates g v) ∨ g ∈ sl.weighted) := by
    unfold SelectLayer.Gates; decide
  intro g hg
  exact absurd hg (h g)

/-- **The structure is satisfiable**: §9.5's wiring meets every protocol-structure condition of
`ProofsSoundOfSelected`, the registration's and the epoch's records are right, and `SelFan` holds at `F = 1`, with two
gated tiles of weight 1. -/
theorem structure_holds :
    sl.toDrawnLayer.WriterReads ∧ sl.SelectionsSourced ∧ sl.SelectionsProved ∧ sl.CertifiedOperands ∧
      sl.FormingClosed ∧ sl.NoLateSelection ∧ sl.NoiseProved wt ∧ sl.AnchorsPinned ∧ sl.AlwaysProved ∧
      sl.TallyIn ∧ sl.GatesTallied ∧ sl.TallyReads ∧ sl.TallyNoAdvice ∧ sl.RegisteredRight ∧
      sl.EpochFormedRight ∧ sl.SelFan wt 1 ∧ (∀ w j, 0 ≤ wt w j) ∧
      (∀ c, 0 < wt 0 (.tile c) ∧ (sl.gates 0 (.tile c)).Nonempty) :=
  ⟨writerReads, selectionsSourced, selectionsProved, certifiedOperands, formingClosed, noLateSelection,
    noiseProved, anchorsPinned, alwaysProved, tallyIn, gatesTallied, tallyReads, tallyNoAdvice,
    fun _ => trivial, fun _ => trivial, selFan,
    fun _ j => by cases j <;> simp [wt],
    fun c => ⟨by simp [wt], ⟨.okA c, by simp [sl, gatesU]⟩⟩⟩

end Accounting.SecurityProofs.TS.SelectedInstance
