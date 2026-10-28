/-!
# Linking's units, read from the description (PROTOCOL.md §16.13)

A template query's units are its root body's members: per root node that is not an input node, `(k, members, offset,
gates)` (`HmLinks.nodes`), member `m` holding gates `[offset + m · gates, offset + (m + 1) · gates)`. `unitAt` and `ownerAt`
locate a unit and a gate's unit from that list, in work proportional to the root body's nodes, never to its members.
`unitsRef` unrolls every member; `unitRef` and `ownerRef`, the lookups in it (the first unit whose gates hold a gate), are
the reference they are proved equal to (`unitAt_eq`, `ownerAt_eq`, `count_eq`).
-/

namespace Flock.HmLinks

/-- A root node's units: its index, members, first gate and gates per member. -/
abbrev Node := Nat × Nat × Nat × Nat

/-- One node's units, unrolled: per member `(k, m, lo, hi)`. -/
def block : Node → List (Nat × Nat × Nat × Nat)
  | (k, mm, off, g) => (List.range mm).map fun m => (k, m, off + m * g, off + m * g + g)

/-- Every unit, unrolled, in node order. -/
def unitsRef (ns : List Node) : List (Nat × Nat × Nat × Nat) := ns.flatMap block

/-- Unit `u`'s node and member in the unrolled units. -/
def unitRef (ns : List Node) (u : Nat) : Option (Nat × Nat) := (unitsRef ns)[u]?.map fun (k, m, _, _) => (k, m)

/-- The first unrolled unit whose gates hold gate `x`. -/
def ownerRef (ns : List Node) (x : Nat) : Option Nat := (unitsRef ns).findIdx? fun (_, _, a, b) => a ≤ x && x < b

/-- The number of units. -/
def count (ns : List Node) : Nat := (ns.map (·.2.1)).sum

/-- Unit `u`'s node and member. -/
def unitAt : List Node → Nat → Option (Nat × Nat)
  | [], _ => none
  | (k, mm, _, _) :: ns, u => if u < mm then some (k, u) else unitAt ns (u - mm)

/-- The unit whose gates hold gate `x`: the first node whose members' gates hold it, and its member there. -/
def ownerAt : List Node → Nat → Option Nat
  | [], _ => none
  | (_, mm, off, g) :: ns, x => if off ≤ x && x < off + mm * g then some ((x - off) / g) else (ownerAt ns x).map (· + mm)

theorem block_length (n : Node) : (block n).length = n.2.1 := by
  obtain ⟨k, mm, off, g⟩ := n
  simp [block]

theorem count_eq (ns : List Node) : count ns = (unitsRef ns).length := by
  induction ns with
  | nil => rfl
  | cons n ns ih => simp [count, unitsRef, block_length] at ih ⊢

theorem unitAt_eq (ns : List Node) : unitAt ns = unitRef ns := by
  funext u
  induction ns generalizing u with
  | nil => rfl
  | cons n ns ih =>
    obtain ⟨k, mm, off, g⟩ := n
    have hl : (block (k, mm, off, g)).length = mm := block_length _
    simp only [unitAt, unitRef, unitsRef, List.flatMap_cons]
    by_cases h : u < mm
    · simp only [h, ↓reduceIte]
      rw [List.getElem?_append_left (by omega)]
      simp [block, h]
    · simp only [h, ↓reduceIte]
      rw [List.getElem?_append_right (by omega), hl, ih]
      rfl

theorem block_findIdx (k mm off g x : Nat) :
    (block (k, mm, off, g)).findIdx? (fun (_, _, a, b) => a ≤ x && x < b)
      = if off ≤ x && x < off + mm * g then some ((x - off) / g) else none := by
  induction mm with
  | zero => simp [block]
  | succ mm ih =>
    have hb : block (k, mm + 1, off, g) = block (k, mm, off, g) ++ [(k, mm, off + mm * g, off + mm * g + g)] := by
      simp [block, List.range_succ]
    rw [hb, List.findIdx?_append, ih, block_length]
    simp only [List.findIdx?_cons, List.findIdx?_nil]
    have hs : (mm + 1) * g = mm * g + g := Nat.succ_mul mm g
    by_cases h1 : off ≤ x ∧ x < off + mm * g
    · have h2 : off ≤ x ∧ x < off + (mm + 1) * g := by rw [hs]; omega
      simp [h1, h2]
    · by_cases h3 : off + mm * g ≤ x ∧ x < off + mm * g + g
      · have h2 : off ≤ x ∧ x < off + (mm + 1) * g := by rw [hs]; omega
        have hq : (x - off) / g = mm := Nat.div_eq_of_lt_le (k := mm) (n := g) (by omega) (by rw [hs]; omega)
        have h4 : ¬ x < off + mm * g := by omega
        simp [h2, h3, h4, hq]
      · have h2 : ¬(off ≤ x ∧ x < off + (mm + 1) * g) := by rw [hs]; omega
        simp [h2, h3]
        omega

theorem ownerAt_eq (ns : List Node) : ownerAt ns = ownerRef ns := by
  funext x
  induction ns with
  | nil => rfl
  | cons n ns ih =>
    obtain ⟨k, mm, off, g⟩ := n
    simp only [ownerAt, ownerRef, unitsRef, List.flatMap_cons]
    rw [List.findIdx?_append, block_findIdx, block_length]
    simp only [ownerRef, unitsRef] at ih
    rw [← ih]
    by_cases h : off ≤ x ∧ x < off + mm * g <;> simp [h, Nat.add_comm]

end Flock.HmLinks
