module

public import Proofs.Flock.Soundness.Refine.OkLe

/-!
# The setup reads the lookup tables only by name

flock-verify reads a lookup table only by `find?` at a name its circuit lists (`HmRow.parse` at each of META's lookups,
`Typed.hold` at each table a type reads), and refuses the circuit where the table is missing. So more tables change
nothing: where `T'` finds by name whatever `T` finds (`TablesExtend T T'`), every circuit parsed with `T` is parsed with
`T'` (`parse_okLe`, `hmOutParse_okLe`, `parseCommit_okLe`), and where the circuit file parses with `T`, a one-table
session `T'` sets up is the one `T` sets up (`setupTables_back`). Tables found among the pinned MUFU tables, whose
names are distinct (`mufu_nodup`), extend those found where more are found alike (`extend_filterMap`).

The proofs walk both runs at once with `ok_le_walk` (`Refine/OkLe.lean`), which leaves the table reads.
-/

@[expose] public section

namespace FlockSoundness.Refine

open Flock

/-! ## More tables change nothing -/

/-- **`T'` extends `T`** as flock-verify reads lookup tables: by name, `T'` finds whatever `T` finds. -/
def TablesExtend (T T' : List (String × ByteArray)) : Prop :=
  ∀ n x, T.find? (·.1 == n) = some x → T'.find? (·.1 == n) = some x

variable {T T' : List (String × ByteArray)}

/-- **A circuit `HmRow.parse` reads with `T` it reads with `T'`.** -/
theorem parse_okLe (hext : TablesExtend T T') (tags : Tags) (bytes : ByteArray) (tmpl : Option Typed.Template)
    (pre : Array (String × Net)) : OkLe (HmRow.parse tags bytes T tmpl pre) (HmRow.parse tags bytes T' tmpl pre) := by
  unfold HmRow.parse
  ok_le_walk
  split
  · rename_i heq
    simp only [hext _ _ heq]
    ok_le_walk
  · ok_le_walk

theorem hold_okLe (hext : TablesExtend T T') (types : Array (String × CircuitType)) :
    OkLe (Typed.hold types T) (Typed.hold types T') := by
  unfold Typed.hold
  ok_le_walk
  split
  · rename_i heq
    simp only [hext _ _ heq]
    ok_le_walk
  · ok_le_walk

theorem classOf_okLe (hext : TablesExtend T T') (mj : Lean.Json) :
    OkLe (Typed.classOf mj T) (Typed.classOf mj T') := by
  unfold Typed.classOf
  ok_le_walk
  exact hold_okLe hext _

theorem read_okLe (hext : TablesExtend T T') (tags : Tags) (bytes : ByteArray) :
    OkLe (Typed.read tags bytes T) (Typed.read tags bytes T') := by
  unfold Typed.read
  ok_le_walk
  exact classOf_okLe hext _

/-- **A circuit `HmOut.parse` reads with `T` it reads with `T'`.** -/
theorem hmOutParse_okLe (hext : TablesExtend T T') (tags : Tags) (bytes : ByteArray) :
    OkLe (HmOut.parse tags bytes T) (HmOut.parse tags bytes T') := by
  unfold HmOut.parse
  ok_le_walk
  all_goals first | exact parse_okLe hext _ _ _ _ | exact read_okLe hext _ _

theorem parseTmpl_okLe (hext : TablesExtend T T') (tags : Tags) (bytes : ByteArray) :
    OkLe (HmOut.parseTmpl tags bytes T) (HmOut.parseTmpl tags bytes T') := by
  unfold HmOut.parseTmpl
  ok_le_walk
  exact parse_okLe hext _ _ _ _

/-- **A circuit `HmOut.parseCommit` reads with `T` it reads with `T'`.** -/
theorem parseCommit_okLe (hext : TablesExtend T T') (tags : Tags) (bytes : ByteArray) :
    OkLe (HmOut.parseCommit tags bytes T) (HmOut.parseCommit tags bytes T') := by
  unfold HmOut.parseCommit
  ok_le_walk
  · exact parseTmpl_okLe hext _ _
  · exact hmOutParse_okLe hext _ _

/-- An untyped statement's `setupH` with `T'` sets up what it sets up with `T`, where its circuit parses with `T`. -/
theorem setupH_back (hext : TablesExtend T T') {tags : Tags} {cf : ByteArray} {c : Circuit}
    (ht : tags.typed = false) (hp : HmRow.parse tags cf T = .ok c) (pf : ByteArray) (coins : Coins)
    (draw partition : Option Lean.Json) (program : Option ByteArray) :
    OkLe (Stmt.setupH tags cf pf coins T' draw partition program) (Stmt.setupH tags cf pf coins T draw partition program) := by
  unfold Stmt.setupH
  ok_le_walk
  · rename_i hc
    rw [ht] at hc
    cases hc
  · exact okLe_back hp (parse_okLe hext _ _ _ _)

/-- `setupHidden` with `T'` sets up what it sets up with `T`, where its circuit parses with `T`. -/
theorem setupHidden_back (hext : TablesExtend T T') {tags : Tags} {cf : ByteArray} {c : Circuit}
    (hp : HmOut.parseCommit tags cf T = .ok c) (pf : ByteArray) (coins : Coins) (draw partition : Option Lean.Json)
    (program own : Option ByteArray) :
    OkLe (Stmt.setupHidden tags cf pf coins T' draw partition program own)
      (Stmt.setupHidden tags cf pf coins T draw partition program own) := by
  rw [Stmt.setupHidden_eq, Stmt.setupHidden_eq, hp, parseCommit_okLe hext _ _ c hp]
  exact okLe_refl _

/-- **A one-table session `Stmt.setupTables` sets up with tables `T'` that extend `T` is the one it sets up with `T`**,
where the circuit file `HmOut.parse`s with `T` and its statement is untyped. -/
theorem setupTables_back (hext : TablesExtend T T') {tags : Tags} {cf : ByteArray} {count : ℕ} {c : Circuit}
    (hcount : count ≤ 1) (ht : tags.typed = false) (hp : HmOut.parse tags cf T = .ok c) (pf : ByteArray)
    (coins : Coins) (draw partition : Option Lean.Json) (program : Option ByteArray)
    (work : Option (ℕ × Array (String × ℕ × ℕ))) (stratified : Option ℕ)
    (closure : Option (Std.HashMap ℕ (Array ℕ))) (own : Option ByteArray) :
    OkLe (Stmt.setupTables tags cf pf coins count T' draw partition program work stratified closure own)
      (Stmt.setupTables tags cf pf coins count T draw partition program work stratified closure own) := by
  have hi := HmOut.parse_isTmpl hp
  have hpc : HmOut.parseCommit tags cf T = .ok c := by
    rw [HmOut.parseCommit, ite_eq_right (by simp [hi])]
    exact hp
  have hr : HmRow.parse tags cf T = .ok c := by
    unfold HmOut.parse at hp
    rw [ite_eq_right (by simp [hi])] at hp
    dsimp only at hp
    rw [ite_eq_left (by simp [ht])] at hp
    obtain ⟨c', hc', hp⟩ := ebind_ok hp
    obtain ⟨_, -, hp⟩ := ebind_ok hp
    cases hp
    exact hc'
  unfold Stmt.setupTables
  ok_le_walk
  · exact setupHidden_back hext hpc _ _ _ _ _ _
  · exact setupH_back hext ht hr _ _ _ _ _
  all_goals exact absurd hcount ‹_›

/-! ## Tables found among the pinned MUFU tables -/

theorem mufu_nodup : (MUFU_TABLES.map (·.name)).Nodup := by decide

theorem name_of_mem {L : List MufuTable} {f : MufuTable → Option (String × ByteArray)}
    (hn : ∀ t x, f t = some x → x.1 = t.name) {x : String × ByteArray} (hx : x ∈ L.filterMap f) :
    x.1 ∈ L.map (·.name) := by
  obtain ⟨t, ht, hft⟩ := List.mem_filterMap.1 hx
  exact List.mem_map.2 ⟨t, ht, (hn t x hft).symm⟩

/-- **Finding more of tables with distinct names extends what is found**: where `f` and `f'` each find a table under
its name, and `f'` finds whatever `f` finds, the tables `f'` finds extend those `f` finds. -/
theorem extend_filterMap : ∀ {L : List MufuTable}, (L.map (·.name)).Nodup →
    ∀ {f f' : MufuTable → Option (String × ByteArray)}, (∀ t x, f t = some x → x.1 = t.name) →
      (∀ t x, f' t = some x → x.1 = t.name) → (∀ t x, f t = some x → f' t = some x) →
      TablesExtend (L.filterMap f) (L.filterMap f')
  | [], _, _, _, _, _, _, _, _, h => by simp at h
  | t :: L, hL, f, f', hn, hn', hf, n, x, h => by
    rw [List.map_cons, List.nodup_cons] at hL
    have ih := extend_filterMap hL.2 hn hn' hf n x
    rw [List.filterMap_cons] at h ⊢
    cases hft : f t with
    | some y =>
      rw [hft] at h
      rw [hf t y hft]
      simp only [List.find?_cons] at h ⊢
      cases hy : y.1 == n
      · rw [hy] at h; exact ih h
      · rw [hy] at h; exact h
    | none =>
      rw [hft] at h
      have hx := List.mem_of_find?_eq_some h
      have hxn : x.1 = n := by simpa using List.find?_some h
      have hne : t.name ≠ n := fun e => hL.1 (e ▸ hxn ▸ name_of_mem hn hx)
      cases hft' : f' t with
      | none => exact ih h
      | some y' =>
        simp only [List.find?_cons]
        have : (y'.1 == n) = false := by rw [hn' t y' hft']; simpa using hne
        rw [this]
        exact ih h

end FlockSoundness.Refine
