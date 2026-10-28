import Mathlib.Data.Fintype.Card
import Proofs.Flock.Sampled.Statements

/-!
# The sampled audit's statements' hypotheses hold together

One instance for all of them: one padded index, one type, nothing in the values (`Ty₁`, `X₁`), a one-leaf root holding
the unit's record (`onePort`: its root is the climb of the record's leaf, so the opening holds without computing SHA-512),
and one session of one slot holding the unit (`batch₁`, its slots routed in session order, `route₁`).

* `coverage_nonvacuous`, `slotBinding_nonvacuous`, `slotsSound_nonvacuous`: the hypotheses of `Coverage`, `SlotBinding`
  and `SlotsSound` at that instance, the draw being the one unit.
* `menu₀_shapeMenu`, `menu₀_inRange`, `menu₀_room`: `menu₀`, one-slot sessions, one per drawn unit (`batchOf`), one
  type per shape, each staged over the inner shape `m = 22, k_log = 6`, two claims. `ShapeMenu menu₀`, `Menu.InRange`
  and `Menu.Room` by proof; that V* stages it (`menu₀.staged`, so `Menu.Staged`) is checked by `#guard`, by evaluation
  and not by proof: building V*'s algebra is beyond the kernel's evaluation.
* `Menu.Room.fits`, `Menu.Room.roomAt`: the deterministic room fits every draw, so it gives `Menu.RoomAt` at δ = 0
  under every law; `menu₀_fits`, `menu₀_roomAt` there. `menuComplete_nonvacuous`: `MenuComplete`'s and
  `MenuCompleteAt`'s hypotheses at `menu₀`, given that V* stages it.
* `composition_nonvacuous`: `Composition`'s hypotheses at that instance, under a one-point law and a session that
  verifies at once, with an instance of the recursion's interface (`I₁`) whose break fires at no draw.
* `zeroKnowledge_nonvacuous`, `zeroKnowledgeAt_nonvacuous`: `ZeroKnowledge`'s hypotheses at that instance, `menu₀` and
  the one-unit draw; `ZeroKnowledgeAt`'s, under the one-point law.
-/

namespace FlockSampled

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit
open scoped ENNReal

/-- One type, whose relation holds everywhere, whose matrix has no nonzeros and whose unit fills one instance. -/
def Ty₁ : Types Unit Unit := { holds := fun _ _ => True, nnz := fun _ => 0, instances := fun _ => 1 }

/-- The one unit's record, everywhere. -/
def X₁ : Trace 1 Unit Unit := fun _ => some ((), ())

/-- `k` sessions of one slot each, session `b` holding unit `b`. -/
def batch₁ (k : ℕ) : Batching k Unit Unit where
  sessions := k
  shape := fun _ => ⟨1, 0, 1⟩
  typ := fun _ _ => ()
  idx := fun b _ => some b
  val := fun _ _ => ()

/-- `batch₁ k`'s slots in session order: position `j` is session `j`'s one slot. -/
def route₁ (k : ℕ) : Fin k ≃ (batch₁ k).Pos where
  toFun j := ⟨j, ⟨0, Nat.zero_lt_one⟩⟩
  invFun p := p.1
  left_inv _ := rfl
  right_inv := by
    rintro ⟨b, c⟩
    have hc : (c : ℕ) < 1 := c.isLt
    have e : c = ⟨0, Nat.zero_lt_one⟩ := Fin.ext (show (c : ℕ) = 0 by omega)
    subst e
    rfl

/-- The draw of every unit, in order, is the identity. -/
theorem drawOrder_univ {k : ℕ} (j : Fin k) : drawOrder (Finset.univ : Finset (Fin k)) j = some j := by
  have hj : (j : ℕ) < (Finset.univ : Finset (Fin k)).card := lt_of_lt_of_eq j.isLt (Finset.card_fin k).symm
  have hu := Finset.orderEmbOfFin_unique (s := (Finset.univ : Finset (Fin k))) rfl
    (f := fun x => (⟨x.val, lt_of_lt_of_eq x.isLt (Finset.card_fin k)⟩ : Fin k)) (fun _ => Finset.mem_univ _)
    (fun _ _ h => Fin.lt_def.2 (Fin.lt_def.1 h))
  unfold drawOrder
  rw [dite_eq_left hj]
  exact congrArg some (congrFun hu ⟨j.val, hj⟩).symm

theorem batch₁_cover (k : ℕ) : CoverCheck Finset.univ (batch₁ k) :=
  ⟨k, route₁ k, (Finset.card_fin k).le, fun j => (drawOrder_univ j).symm⟩

/-- The draw `S`'s units, one per one-slot session, in the draw's order. -/
def batchOf {N : ℕ} (S : Finset (Fin N)) : Batching N Unit Unit where
  sessions := S.card
  shape := fun _ => ⟨1, 0, 1⟩
  typ := fun _ _ => ()
  idx := fun b _ => some (S.orderEmbOfFin rfl b)
  val := fun _ _ => ()

theorem batchOf_cover {N : ℕ} (S : Finset (Fin N)) : CoverCheck S (batchOf S) := by
  refine ⟨S.card, route₁ S.card, le_rfl, fun j => ?_⟩
  have h : drawOrder S j = some (S.orderEmbOfFin rfl j) := by
    unfold drawOrder
    exact dite_eq_left j.isLt
  exact h.symm

/-- A one-leaf root whose leaf is the record `bc`'s. -/
def onePort (bc : ByteArray) : Flock.Registered.Port where
  name := ""
  value := ""
  root := Flock.Registered.rowLeaf ByteArray.empty 0 (Flock.Hm96.Default512.treeLeaf bc)
  leaves := 1
  words := none
  wordBits := 0
  domain := ByteArray.empty
  readsRel := #[]

theorem climb_one (dom leaf path : ByteArray) (i : ℕ) :
    Flock.Registered.climb dom leaf i (Flock.log2ceil 1) path = leaf := by
  rfl

theorem onePort_opens (bc : ByteArray) : Flock.Registered.opens (onePort bc) bc 0 ByteArray.empty = true := by
  simp only [Flock.Registered.opens, Flock.Registered.opensLeaf, Bool.and_eq_true, decide_eq_true_eq]
  exact ⟨⟨Nat.zero_lt_one, rfl⟩, climb_one _ _ _ _⟩

/-- The commitment: the one-leaf root, every record empty, every unit at position 0. -/
def C₁ : Commit 1 Unit Unit := { rt := onePort ByteArray.empty, enc := fun _ _ => ByteArray.empty, pos := fun _ => 0 }

/-- The committed trace's openings. -/
def path₀₁ : Fin 1 → ByteArray := fun _ => ByteArray.empty

/-- One session of one slot holding the unit, and the slot's opening. -/
def h₁ : Hidden 1 Unit Unit := { β := batch₁ 1, path := fun _ => ByteArray.empty }

theorem C₁_injective : C₁.Injective := fun _ _ _ _ _ => ⟨rfl, rfl⟩

theorem C₁_committed : C₁.Committed X₁ path₀₁ where
  opens := fun _ _ _ _ => onePort_opens _
  full := fun _ _ _ _ _ => rfl

theorem h₁_opens : C₁.SlotOpens h₁ := fun _ _ _ => onePort_opens _

theorem h₁_bound : Bound X₁ h₁.β := fun _ _ _ => rfl

theorem h₁_proves : Proves Ty₁ C₁ Finset.univ h₁ where
  cover := batch₁_cover 1
  opens := h₁_opens
  fits := fun _ _ _ => ⟨Nat.zero_le _, le_rfl⟩
  holds := fun _ _ _ => trivial

/-- **`Coverage`'s hypotheses hold together**, at a nonempty draw. -/
theorem coverage_nonvacuous :
    (Finset.univ : Finset (Fin 1)).Nonempty ∧ CoverCheck Finset.univ (batch₁ 1) ∧ Bound X₁ (batch₁ 1) :=
  ⟨Finset.univ_nonempty, batch₁_cover 1, h₁_bound⟩

/-- **`SlotBinding`'s hypotheses hold together**, at a slot that holds a unit. -/
theorem slotBinding_nonvacuous :
    C₁.Injective ∧ C₁.Committed X₁ path₀₁ ∧ C₁.SlotOpens h₁ ∧ ∃ p i, h₁.β.held p = some i :=
  ⟨C₁_injective, C₁_committed, h₁_opens, ⟨⟨⟨0, Nat.zero_lt_one⟩, ⟨0, Nat.zero_lt_one⟩⟩, 0, rfl⟩⟩

/-- **`SlotsSound`'s hypotheses hold together**, at a nonempty draw. -/
theorem slotsSound_nonvacuous :
    (Finset.univ : Finset (Fin 1)).Nonempty ∧ C₁.Injective ∧ C₁.Committed X₁ path₀₁ ∧ Proves Ty₁ C₁ Finset.univ h₁ :=
  ⟨Finset.univ_nonempty, C₁_injective, C₁_committed, h₁_proves⟩

/-- One-slot sessions of no nonzeros and one unit position, one per drawn unit, one type per shape, over the inner shape
`m = 22, k_log = 6`, two claims. -/
def menu₀ : Menu where
  shapes := [⟨1, 0, 1⟩]
  inner := fun _ => ⟨22, 6, 2⟩
  types := fun _ => 1
  padded := fun k => (List.finRange k).map fun _ => ⟨1, 0, 1⟩

#guard menu₀.staged

theorem isOk_iff {ε α : Type} (x : Except ε α) : isOk x = true ↔ ∃ a, x = .ok a := by
  cases x <;> simp [isOk]

theorem Menu.staged_ok {M : Menu} (h : M.staged = true) : M.Staged :=
  fun s hs => (isOk_iff _).1 (List.all_eq_true.1 h s hs)

/-- **The live verifier accepts `menu₀`**: its inner session has `k_log` 6, no regions and 24 + 16 link-point
coordinates. -/
theorem menu₀_inRange : menu₀.InRange :=
  fun _ _ => (Flock.checkInRange_ok _ _ _).2 (by simp [menu₀, Flock.PT_LOCAL])

/-- **`menu₀` is a shape menu.** -/
theorem menu₀_shapeMenu : ShapeMenu menu₀ := by
  intro k s hs
  obtain ⟨_, _, e⟩ := List.mem_map.1 hs
  cases e
  exact List.mem_singleton_self _

/-- **`menu₀`'s padded sessions have room**, for `Ty₁`. -/
theorem menu₀_room : menu₀.Room Ty₁ := by
  intro N S ty U place hU hW
  refine ⟨batchOf S, rfl, batchOf_cover S, fun p i hp => ⟨rfl, ?_⟩,
    fun _ => le_of_le_of_eq (Finset.card_le_univ _) Fintype.card_unit⟩
  have hi : i ∈ S := by
    obtain rfl := Option.some.inj hp
    exact Finset.orderEmbOfFin_mem S rfl _
  exact (List.mem_singleton.1 (hW.1 _ (hU i hi)).1).symm

/-- The one type's placement, at `menu₀`'s one shape. -/
def place₁ : Unit → Shape := fun _ => ⟨1, 0, 1⟩

theorem menu₀_within : menu₀.Within Ty₁ Finset.univ place₁ :=
  ⟨fun _ _ => ⟨List.mem_singleton_self _, Nat.zero_le _, le_rfl⟩,
    fun _ => le_of_le_of_eq (Finset.card_le_univ _) Fintype.card_unit⟩

/-- **`Menu.Room` fits every draw** of a trace whose units all hold records of types in `U`, placed within the menu's
bounds. -/
theorem Menu.Room.fits {N : ℕ} {T V : Type} {M : Menu} {Ty : Types T V} (hR : M.Room Ty) {X : Trace N T V}
    {U : Finset T} {place : T → Shape} (hX : ∀ i, ∃ t v, X i = some (t, v) ∧ t ∈ U) (hW : M.Within Ty U place)
    (S : Finset (Fin N)) : M.Fits Ty X S := by
  let ty : Fin N → T := fun i => Classical.choose (hX i)
  have hty : ∀ i, ∃ v, X i = some (ty i, v) ∧ ty i ∈ U := fun i => Classical.choose_spec (hX i)
  obtain ⟨β, hv, hc, hp, ht⟩ := hR S ty U place (fun i _ => (hty i).elim fun _ h => h.2) hW
  refine ⟨β, hv, hc, fun p i hi => ?_, ht⟩
  obtain ⟨hT, hS⟩ := hp p i hi
  obtain ⟨w, hw, hU⟩ := hty i
  refine ⟨⟨w, hw.trans (congrArg (fun t => some (t, w)) hT.symm)⟩, ?_⟩
  rw [hT, hS]
  exact (hW.1 _ hU).2

/-- **`Menu.Room` gives `Menu.RoomAt` at δ = 0**, under every law, for a trace whose units all hold records of types in
`U`, placed within the menu's bounds. -/
theorem Menu.Room.roomAt {N : ℕ} {T V : Type} {M : Menu} {Ty : Types T V} (hR : M.Room Ty) (L : Law N)
    {X : Trace N T V} {U : Finset T} {place : T → Shape} (hX : ∀ i, ∃ t v, X i = some (t, v) ∧ t ∈ U)
    (hW : M.Within Ty U place) : M.RoomAt Ty L X 0 := by
  unfold Menu.RoomAt
  refine le_of_le_of_eq (prCoin_mono (q := fun _ => False) fun ω h => ?_) prCoin_false
  exact h (Menu.Room.fits hR hX hW (L.draw ω))

theorem X₁_placed : ∀ i, ∃ t v, X₁ i = some (t, v) ∧ t ∈ (Finset.univ : Finset Unit) :=
  fun _ => ⟨(), (), rfl, Finset.mem_univ _⟩

/-- The one-point law over the one unit. -/
def L₁ : Law 1 := { Ω := Unit, draw := fun _ => Finset.univ }

theorem menu₀_fits : menu₀.Fits Ty₁ X₁ Finset.univ := Menu.Room.fits menu₀_room X₁_placed menu₀_within _

theorem menu₀_roomAt : menu₀.RoomAt Ty₁ L₁ X₁ 0 := Menu.Room.roomAt menu₀_room L₁ X₁_placed menu₀_within

/-- **`MenuComplete`'s and `MenuCompleteAt`'s hypotheses hold together**, at `menu₀` given that V* stages it: a
nonempty draw of `X₁` fits, and under the one-point law every draw does (δ = 0, from `menu₀_room`). -/
theorem menuComplete_nonvacuous (h : menu₀.staged = true) :
    menu₀.Staged ∧ menu₀.InRange ∧ ShapeMenu menu₀ ∧ (Finset.univ : Finset (Fin 1)).Nonempty ∧
      menu₀.Fits Ty₁ X₁ Finset.univ ∧ menu₀.RoomAt Ty₁ L₁ X₁ 0 :=
  ⟨Menu.staged_ok h, menu₀_inRange, menu₀_shapeMenu, Finset.univ_nonempty, menu₀_fits, menu₀_roomAt⟩

/-- Sessions that end at once. -/
def session₁ : Finset (Fin 1) → Unit → Game Unit := fun _ _ => .ret ()

/-- **An instance of the recursion's interface**: the sessions always verify, hold `h₁`, and break only away from the
one draw; no owed outputs, and one view. -/
noncomputable def I₁ : Recursion session₁ Ty₁ (fun _ => C₁) where
  verifies := fun _ => true
  hid := fun _ _ _ => h₁
  Break := fun S _ _ => S ≠ Finset.univ
  ε := fun _ _ _ => 0
  sound := fun S _ _ => by
    by_cases hS : S = Finset.univ
    · subst hS
      refine Or.inr (le_of_le_of_eq (prob_mono (q := fun _ => False) ?_ _ _) (prob_false _ _))
      intro _ h
      exact h.2 h₁_proves
    · exact Or.inl hS
  Out := Unit
  out := fun _ => ()
  View := Unit
  real := fun _ _ _ => ()
  sim := fun _ => ()
  Close := fun _ a b => a = b
  hides := fun _ _ _ _ _ _ => rfl

/-- **`Composition`'s hypotheses hold together**: at the one registration, the one-point law, `session₁` and `I₁`, the
records determine their contents, `X₁` is the trace committed, and the break of the recursion's interface fires at no
draw of any prover. -/
theorem composition_nonvacuous :
    C₁.Injective ∧ C₁.Committed X₁ path₀₁ ∧
      ∀ σ : Strategy (sampledAudit L₁ session₁ ()), ¬ ∃ ω, I₁.Break (L₁.draw ω) () (contOf σ ω) :=
  ⟨C₁_injective, C₁_committed, fun _ ⟨_, h⟩ => h rfl⟩

/-- **`ZeroKnowledge`'s hypotheses hold together**: `h₁`'s one session is `menu₀`'s padded sessions for the one-unit
draw, and it proves the statement there. -/
theorem zeroKnowledge_nonvacuous :
    h₁.β.view = menu₀.padded (Finset.univ : Finset (Fin 1)).card ∧ Proves Ty₁ C₁ Finset.univ h₁ :=
  ⟨(rfl : h₁.β.view = menu₀.padded 1).trans (congrArg menu₀.padded (Finset.card_fin 1).symm), h₁_proves⟩

/-- **`ZeroKnowledgeAt`'s hypotheses hold together**: at `menu₀` given that V* stages it, `X₁` committed under `C₁`, its
one record in its type's relation, and room at every draw of the one-point law. -/
theorem zeroKnowledgeAt_nonvacuous (h : menu₀.staged = true) :
    menu₀.Staged ∧ menu₀.InRange ∧ ShapeMenu menu₀ ∧ C₁.Committed X₁ path₀₁ ∧
      (∀ i t v, X₁ i = some (t, v) → Ty₁.holds t v) ∧ menu₀.RoomAt Ty₁ L₁ X₁ 0 :=
  ⟨Menu.staged_ok h, menu₀_inRange, menu₀_shapeMenu, C₁_committed, fun _ _ _ _ => trivial, menu₀_roomAt⟩

end FlockSampled
