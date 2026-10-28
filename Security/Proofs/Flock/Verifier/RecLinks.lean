/-
The link check's binding (PRF-282 G4). A link row of a recursive session's table that V*'s `RecLink_v1` accepts, its outputs
the two rows trees' roots the verifier builds (`RecLinkTree.rowsRoot`), opens to the reader's input row and the writer's
output row the inner statement publishes, and the two rows commit one digest, unless SHA-512 collides: `link_binding`
gives the explicit colliding pair (`treePair`), as `climb_binding` does for a Flock opening. One way: it says what an
accepted row's outputs mean, not that every honest table is accepted. Collision-tolerant: a collision voids the conclusion
for that row only, and names the two inputs.

A leaf no row fills is `EMPTY = SHA-512("")`, no row leaf's input, and a root of 64 zero bytes (a failed row's output) is
refused, so neither a missing row nor a failed one opens without a collision: no assumption beyond SHA-512's collision
resistance, which a reader of this lemma takes as the premise its pair refutes.

What is proved is about the unit's reference (`RecLinkRef.linkRef`); the netlist (`VStar.linkUnit`) is held to it by the
`link_unit` and `link_ref` difftests, not by a proof.
-/
import Proofs.Flock.Verifier.Merkle
import Flock.RecLinkRef

namespace Flock.RecLinkBind

open RecLinkTree RecLinkRef RecLinks

/-! ### A climb, from the top -/

theorem climb_append (ms : MerkleScheme) (s : ByteArray) :
    ∀ (l : List ByteArray) (acc : ByteArray) (idx : Nat),
      climb ms acc idx (l ++ [s]) =
        if idx / 2 ^ l.length % 2 = 0 then ms.node (climb ms acc idx l) s else ms.node s (climb ms acc idx l)
  | [], acc, idx => by simp [climb]
  | t :: l, acc, idx => by
    have hd : idx / 2 / 2 ^ l.length = idx / 2 ^ (l.length + 1) := by
      rw [Nat.div_div_eq_div_mul, Nat.pow_succ']
    simp only [List.cons_append, climb, List.length_cons]
    rw [climb_append ms s l, hd]

/-- A climb reads only its position's low bits, one a sibling. -/
theorem climb_mod (ms : MerkleScheme) :
    ∀ (l : List ByteArray) (acc : ByteArray) (idx : Nat), climb ms acc idx l = climb ms acc (idx % 2 ^ l.length) l
  | [], _, _ => rfl
  | s :: l, acc, idx => by
    have h2 : idx % 2 ^ (l.length + 1) % 2 = idx % 2 := Nat.mod_mod_of_dvd idx ⟨2 ^ l.length, Nat.pow_succ'⟩
    have hd : idx % 2 ^ (l.length + 1) / 2 % 2 ^ l.length = idx / 2 % 2 ^ l.length := by
      rw [Nat.pow_succ', Nat.mod_mul_right_div_self, Nat.mod_mod]
    simp only [climb, List.length_cons, h2]
    rw [climb_mod ms l _ (idx / 2), climb_mod ms l _ (idx % 2 ^ (l.length + 1) / 2), hd]

/-! ### The tree's leaf at a position, and its path -/

/-- The leaf at `idx` of `sub e h leaves`: the first leaf the halves lead to, or `e 0`. -/
def atLeaf (e : Nat → ByteArray) : Nat → List (Nat × ByteArray) → Nat → ByteArray
  | 0, [], _ => e 0
  | 0, (_, v) :: _, _ => v
  | h + 1, leaves, idx =>
    let p := leaves.partition (·.1 < 2 ^ h)
    if idx < 2 ^ h then atLeaf e h p.1 idx else atLeaf e h (p.2.map (shift h)) (idx - 2 ^ h)

/-- `e` is a tree of empty subtrees to height `H`. -/
def Empties (e : Nat → ByteArray) (H : Nat) : Prop := ∀ k, k < H → e (k + 1) = node (e k) (e k)

theorem sub_succ {e : Nat → ByteArray} {h : Nat} (he : e (h + 1) = node (e h) (e h)) (leaves : List (Nat × ByteArray)) :
    sub e (h + 1) leaves =
      node (sub e h (leaves.partition (·.1 < 2 ^ h)).1) (sub e h ((leaves.partition (·.1 < 2 ^ h)).2.map (shift h))) := by
  cases leaves with
  | nil => simp [sub, he]
  | cons x xs => rfl

theorem subPath_length (e : Nat → ByteArray) : ∀ (h : Nat) (leaves : List (Nat × ByteArray)) (idx : Nat),
    (subPath e h leaves idx).length = h
  | 0, _, _ => rfl
  | h + 1, leaves, idx => by
    simp only [subPath]
    split <;> simp [subPath_length e h]

/-- **The path climbs to the root**: leaf `idx`'s siblings take `atLeaf` to `sub`. -/
theorem climb_subPath (e : Nat → ByteArray) : ∀ (h : Nat) (leaves : List (Nat × ByteArray)) (idx : Nat),
    Empties e h → idx < 2 ^ h → climb scheme (atLeaf e h leaves idx) idx (subPath e h leaves idx) = sub e h leaves
  | 0, [], _, _, _ => rfl
  | 0, _ :: _, _, _, _ => rfl
  | h + 1, leaves, idx, he, hi => by
    have he' : Empties e h := fun k hk => he k (Nat.lt_succ_of_lt hk)
    rw [sub_succ (he h (Nat.lt_succ_self h))]
    simp only [atLeaf, subPath]
    by_cases hlo : idx < 2 ^ h
    · simp only [hlo, ↓reduceIte]
      rw [climb_append, subPath_length, Nat.div_eq_of_lt hlo, climb_subPath e h _ idx he' hlo]
      rfl
    · simp only [hlo, ↓reduceIte]
      have hp : 2 ^ (h + 1) = 2 * 2 ^ h := Nat.pow_succ'
      have hlt : idx - 2 ^ h < 2 ^ h := by omega
      have hm : idx % 2 ^ h = idx - 2 ^ h := by
        rw [Nat.mod_eq_sub_mod (Nat.le_of_not_lt hlo), Nat.mod_eq_of_lt hlt]
      have hb : idx / 2 ^ h = 1 := Nat.div_eq_of_lt_le (by omega) (by omega)
      rw [climb_append, subPath_length, hb, climb_mod, subPath_length, hm, climb_subPath e h _ _ he' hlt]
      rfl

/-! ### The leaf at a position is a leaf there, or `e 0` -/

/-- Every leaf is 64 bytes and below `2^h`. -/
def Good (h : Nat) (leaves : List (Nat × ByteArray)) : Prop := ∀ x ∈ leaves, x.2.size = 64 ∧ x.1 < 2 ^ h

theorem good_lo {h : Nat} {leaves : List (Nat × ByteArray)} (hg : Good (h + 1) leaves) :
    Good h (leaves.partition (·.1 < 2 ^ h)).1 := by
  intro x hx
  rw [List.partition_eq_filter_filter] at hx
  simp only [List.mem_filter, decide_eq_true_eq] at hx
  exact ⟨(hg x hx.1).1, hx.2⟩

theorem good_hi {h : Nat} {leaves : List (Nat × ByteArray)} (hg : Good (h + 1) leaves) :
    Good h ((leaves.partition (·.1 < 2 ^ h)).2.map (shift h)) := by
  intro x hx
  rw [List.partition_eq_filter_filter] at hx
  simp only [List.mem_map, List.mem_filter] at hx
  obtain ⟨y, ⟨hy, -⟩, rfl⟩ := hx
  have hp : 2 ^ (h + 1) = 2 * 2 ^ h := Nat.pow_succ'
  have := (hg y hy).2
  exact ⟨(hg y hy).1, by simp only [shift]; omega⟩

theorem atLeaf_mem (e : Nat → ByteArray) : ∀ (h : Nat) (leaves : List (Nat × ByteArray)) (idx : Nat),
    Good h leaves → idx < 2 ^ h → atLeaf e h leaves idx = e 0 ∨ (idx, atLeaf e h leaves idx) ∈ leaves
  | 0, [], _, _, _ => Or.inl rfl
  | 0, (i, v) :: tl, idx, hg, hi => by
    have hi0 := (hg (i, v) (by simp)).2
    simp only [Nat.pow_zero] at hi0 hi
    have : i = idx := by omega
    subst this
    exact Or.inr (by simp [atLeaf])
  | h + 1, leaves, idx, hg, hi => by
    simp only [atLeaf]
    by_cases hlo : idx < 2 ^ h
    · simp only [hlo, ↓reduceIte]
      rcases atLeaf_mem e h _ idx (good_lo hg) hlo with h1 | h1
      · exact Or.inl h1
      · rw [List.partition_eq_filter_filter] at h1 ⊢
        exact Or.inr (List.mem_filter.mp h1).1
    · simp only [hlo, ↓reduceIte]
      have hp : 2 ^ (h + 1) = 2 * 2 ^ h := Nat.pow_succ'
      rcases atLeaf_mem e h _ (idx - 2 ^ h) (good_hi hg) (by omega) with h1 | h1
      · exact Or.inl h1
      · right
        rw [List.partition_eq_filter_filter] at h1
        simp only [List.mem_map, List.mem_filter] at h1
        obtain ⟨y, ⟨hy, hny⟩, hxy⟩ := h1
        have hny : ¬ y.1 < 2 ^ h := by simpa using hny
        simp only [shift, Prod.mk.injEq] at hxy
        have hy1 : y.1 = idx := by omega
        rw [List.partition_eq_filter_filter]
        rw [← hxy.2, ← hy1]
        exact hy

theorem atLeaf_size (e : Nat → ByteArray) (h : Nat) (leaves : List (Nat × ByteArray)) (idx : Nat)
    (hg : Good h leaves) (hi : idx < 2 ^ h) (h0 : (e 0).size = 64) : (atLeaf e h leaves idx).size = 64 := by
  rcases atLeaf_mem e h leaves idx hg hi with h1 | h1
  · rw [h1]; exact h0
  · exact (hg _ h1).1

theorem sub_size (e : Nat → ByteArray) (h : Nat) (leaves : List (Nat × ByteArray)) (hg : Good h leaves)
    (hes : (e h).size = 64) : (sub e h leaves).size = 64 := by
  cases leaves with
  | nil => cases h <;> exact hes
  | cons x xs =>
    obtain ⟨i, v⟩ := x
    cases h with
    | zero => exact (hg (i, v) (by simp)).1
    | succ h => exact Sha512.hash_size _

theorem subPath_size (e : Nat → ByteArray) : ∀ (h : Nat) (leaves : List (Nat × ByteArray)) (idx : Nat),
    Good h leaves → (∀ k, k ≤ h → (e k).size = 64) → ∀ s ∈ subPath e h leaves idx, s.size = 64
  | 0, _, _, _, _ => by simp [subPath]
  | h + 1, leaves, idx, hg, hes => by
    have hes' : ∀ k, k ≤ h → (e k).size = 64 := fun k hk => hes k (Nat.le_succ_of_le hk)
    simp only [subPath]
    split
    · intro s hs
      rcases List.mem_append.mp hs with hs | hs
      · exact subPath_size e h _ idx (good_lo hg) hes' s hs
      · simp only [List.mem_singleton] at hs; subst hs
        exact sub_size e h _ (good_hi hg) (hes' h (Nat.le_refl h))
    · intro s hs
      rcases List.mem_append.mp hs with hs | hs
      · exact subPath_size e h _ _ (good_hi hg) hes' s hs
      · simp only [List.mem_singleton] at hs; subst hs
        exact sub_size e h _ (good_lo hg) (hes' h (Nat.le_refl h))

/-! ### The extractor for one tree -/

/-- Two SHA-512 inputs that collide: different, with one digest. -/
def Collides (p : ByteArray × ByteArray) : Prop := p.1 ≠ p.2 ∧ Sha512.hash p.1 = Sha512.hash p.2

theorem Collides.collision {p : ByteArray × ByteArray} (h : Collides p) : Sha512.Collision := ⟨p.1, p.2, h.1, h.2⟩

/-- A rows tree's leaves, `rowLeaves`'s once its rows fit. -/
def leavesOf (PB : Nat) (rows : List ((Nat × Nat) × ByteArray)) : List (Nat × ByteArray) :=
  rows.map fun ((u, r), bc) => (index u r PB, leaf bc)

/-- **The extractor for an opening of a rows tree**: commit string `bc`'s leaf climbed by `sibs` at `idx`. If its leaf is
not the tree's leaf there, the climb's colliding node inputs; if it is, a row there with that leaf and a different commit
string gives the two leaf inputs, and no row there (the leaf is `EMPTY`) gives the leaf input and the empty string. -/
def treePair (e : Nat → ByteArray) (h PB : Nat) (rows : List ((Nat × Nat) × ByteArray)) (idx : Nat) (bc : ByteArray)
    (sibs : List ByteArray) : ByteArray × ByteArray :=
  let L := atLeaf e h (leavesOf PB rows) idx
  if leaf bc = L then
    match rows.find? (fun x => index x.1.1 x.1.2 PB == idx && decide (leaf x.2 = L)) with
    | some x => (Hm96.Default512.leafPrefix ++ bc, Hm96.Default512.leafPrefix ++ x.2)
    | none => (Hm96.Default512.leafPrefix ++ bc, ByteArray.empty)
  else (climbPair scheme sibs (subPath e h (leavesOf PB rows) idx) (leaf bc) L idx).shaInputs

/-- **An opening of a rows tree binds**: commit string `bc`'s leaf, climbed by 64-byte siblings at `idx` to the tree's
root, is a row's at `idx`, or `treePair` collides. -/
theorem tree_binding {e : Nat → ByteArray} {h PB : Nat} {rows : List ((Nat × Nat) × ByteArray)} {idx : Nat}
    {bc : ByteArray} {sibs : List ByteArray} (he : Empties e h) (hes : ∀ k, k ≤ h → (e k).size = 64) (h0 : e 0 = EMPTY)
    (hg : Good h (leavesOf PB rows)) (hi : idx < 2 ^ h) (hbc : 0 < bc.size) (hs : ∀ s ∈ sibs, s.size = 64)
    (hl : sibs.length = h) (hc : climb scheme (leaf bc) idx sibs = sub e h (leavesOf PB rows)) :
    (∃ x ∈ rows, index x.1.1 x.1.2 PB = idx ∧ x.2 = bc) ∨ Collides (treePair e h PB rows idx bc sibs) := by
  unfold treePair
  by_cases hL : leaf bc = atLeaf e h (leavesOf PB rows) idx
  · simp only [hL, ↓reduceIte]
    cases hf : rows.find? (fun x => index x.1.1 x.1.2 PB == idx && decide (leaf x.2 = atLeaf e h (leavesOf PB rows) idx)) with
    | some x =>
      have hp := List.find?_some hf
      simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hp
      by_cases hx : x.2 = bc
      · exact Or.inl ⟨x, List.mem_of_find?_eq_some hf, hp.1, hx⟩
      · refine Or.inr ⟨fun hh => hx ((ByteArray.append_right_inj _).mp hh).symm, ?_⟩
        show leaf bc = leaf x.2
        rw [hL, hp.2]
    | none =>
      rcases atLeaf_mem e h _ idx hg hi with h1 | h1
      · refine Or.inr ⟨fun hh => ?_, ?_⟩
        · have hsz := congrArg ByteArray.size hh
          have he0 : ByteArray.empty.size = 0 := rfl
          rw [ByteArray.size_append, he0] at hsz
          omega
        · show leaf bc = Sha512.hash ByteArray.empty
          rw [hL, h1, h0]
          rfl
      · exfalso
        obtain ⟨x, hx, hfx⟩ := List.mem_map.mp h1
        have := List.find?_eq_none.mp hf x hx
        apply this
        obtain ⟨⟨u, r⟩, b⟩ := x
        simp only [Prod.mk.injEq] at hfx
        simp [hfx.1, hfx.2]
  · simp only [hL, ↓reduceIte]
    right
    apply MerklePair.sha512_inputs
    have hl' : sibs.length = (subPath e h (leavesOf PB rows) idx).length := by rw [hl, subPath_length]
    exact climb_binding scheme (MerkleScheme.sha512_sized _) sibs _ _ _ idx hl' hs
      (subPath_size e h _ idx hg hes) (Sha512.hash_size _)
      (atLeaf_size e h _ idx hg hi (by rw [h0]; exact Sha512.hash_size _)) hL
      (by rw [hc, climb_subPath e h _ idx he hi])

/-! ### The verifier's trees -/

/-- The empty subtrees' roots `RecLinkTree.root` reads at depth `D`. -/
def eOf (D : Nat) (k : Nat) : ByteArray := (empties D).getD k EMPTY

/-- **What a rows tree's root is**: `sub` over the rows' leaves, every row in range, and not 64 zero bytes. -/
theorem rowsRoot_ok {UB PB : Nat} {rows : List ((Nat × Nat) × ByteArray)} {R : ByteArray}
    (h : rowsRoot UB PB rows = .ok R) :
    R = sub (eOf (UB + PB)) (UB + PB) (leavesOf PB rows) ∧ (∀ x ∈ rows, x.1.1 < 2 ^ UB ∧ x.1.2 < 2 ^ PB) ∧
      R.data.all (· == 0) = false := by
  unfold rowsRoot rowLeaves root at h
  cases hs : checkShape UB PB with
  | error _ => simp [hs, bind, Except.bind] at h
  | ok _ =>
    simp [hs, bind, Except.bind, pure, Except.pure] at h
    cases hf : List.find? (fun x => !decide (x.1.fst < 2 ^ UB) || !decide (x.1.snd < 2 ^ PB)) rows with
    | some x =>
      obtain ⟨⟨u, r⟩, b⟩ := x
      simp [hf, throw, throwThe, MonadExceptOf.throw] at h
    | none =>
      have hfit : ∀ x ∈ rows, x.1.1 < 2 ^ UB ∧ x.1.2 < 2 ^ PB := by
        intro x hx
        have := List.find?_eq_none.mp hf x hx
        simpa using this
      simp only [hf] at h
      cases hc : checkLeaves (UB + PB) (List.map (fun x => (index x.1.fst x.1.snd PB, leaf x.snd)) rows) with
      | error _ => simp [hc] at h
      | ok _ =>
        simp only [hc] at h
        split at h
        · simp [throw, throwThe, MonadExceptOf.throw] at h
        · rename_i hz
          injection h with h
          subst h
          refine ⟨?_, hfit, by simpa using hz⟩
          unfold leavesOf
          congr 1
          funext k
          simp [eOf, Array.getD_eq_getD_getElem?]

theorem empty_size : ∀ k, (empty k).size = 64
  | 0 => Sha512.hash_size _
  | _ + 1 => Sha512.hash_size _

theorem eOf_eq {D k : Nat} (hk : k ≤ D) : eOf D k = empty k := by
  simp [eOf, empties, Array.getD_eq_getD_getElem?, Nat.lt_succ_of_le hk]

theorem eOf_empties (D : Nat) : Empties (eOf D) D := fun k hk => by
  rw [eOf_eq hk, eOf_eq (Nat.le_of_lt hk)]; rfl

theorem eOf_size {D k : Nat} (hk : k ≤ D) : (eOf D k).size = 64 := by rw [eOf_eq hk]; exact empty_size k

theorem eOf_zero (D : Nat) : eOf D 0 = EMPTY := eOf_eq (Nat.zero_le D)

theorem index_lt {u r UB PB : Nat} (hu : u < 2 ^ UB) (hr : r < 2 ^ PB) : index u r PB < 2 ^ (UB + PB) := by
  unfold index
  rw [Nat.pow_add]
  calc u * 2 ^ PB + r < u * 2 ^ PB + 2 ^ PB := Nat.add_lt_add_left hr _
    _ = (u + 1) * 2 ^ PB := (Nat.succ_mul _ _).symm
    _ ≤ 2 ^ UB * 2 ^ PB := Nat.mul_le_mul_right _ hu

theorem index_inj {u r u' r' PB : Nat} (hr : r < 2 ^ PB) (hr' : r' < 2 ^ PB) (h : index u r PB = index u' r' PB) :
    u = u' ∧ r = r' := by
  unfold index at h
  have hp : 0 < 2 ^ PB := Nat.two_pow_pos PB
  rw [Nat.mul_comm u, Nat.mul_comm u'] at h
  have hd := congrArg (· / 2 ^ PB) h
  have hm := congrArg (· % 2 ^ PB) h
  simp only [Nat.mul_add_div hp, Nat.div_eq_of_lt hr, Nat.div_eq_of_lt hr', Nat.mul_add_mod, Nat.mod_eq_of_lt hr,
    Nat.mod_eq_of_lt hr', Nat.add_zero] at hd hm
  exact ⟨hd, hm⟩

theorem good_leavesOf {UB PB : Nat} {rows : List ((Nat × Nat) × ByteArray)}
    (hfit : ∀ x ∈ rows, x.1.1 < 2 ^ UB ∧ x.1.2 < 2 ^ PB) : Good (UB + PB) (leavesOf PB rows) := by
  intro x hx
  obtain ⟨y, hy, rfl⟩ := List.mem_map.mp hx
  exact ⟨Sha512.hash_size _, index_lt (hfit y hy).1 (hfit y hy).2⟩

theorem wordsOf_get {a : ByteArray} (ha : a.size = 64) {j : Nat} (hj : j < 32) :
    (wordsOf a)[j]? = some ((a.get! (2 * j)).toNat + 256 * (a.get! (2 * j + 1)).toNat) := by
  simp [wordsOf, ha, hj]

/-- A 64-byte root is its 32 words. -/
theorem wordsOf_inj {a b : ByteArray} (ha : a.size = 64) (hb : b.size = 64) (h : wordsOf a = wordsOf b) : a = b := by
  apply ByteArray.eq_of_get!_64 ha hb
  intro i hi
  have hj : i / 2 < 32 := by omega
  have hw := congrArg (·[i / 2]?) h
  simp only [wordsOf_get ha hj, wordsOf_get hb hj, Option.some.injEq] at hw
  have h1 := (a.get! (2 * (i / 2))).toNat_lt
  have h2 := (a.get! (2 * (i / 2) + 1)).toNat_lt
  have h3 := (b.get! (2 * (i / 2))).toNat_lt
  have h4 := (b.get! (2 * (i / 2) + 1)).toNat_lt
  rcases Nat.mod_two_eq_zero_or_one i with hi2 | hi2
  · have hi' : i = 2 * (i / 2) := by omega
    rw [hi']
    exact UInt8.toNat_inj.mp (by omega)
  · have hi' : i = 2 * (i / 2) + 1 := by omega
    rw [hi']
    exact UInt8.toNat_inj.mp (by omega)

theorem expected_zero : zeros = expected ⟨Array.replicate 64 0⟩ := by decide

theorem expected_inj {a b : ByteArray} (ha : a.size = 64) (hb : b.size = 64) (h : expected a = expected b) : a = b :=
  wordsOf_inj ha hb (List.append_cancel_right h)

/-- A root `rowsRoot` returns is no failed row's output. -/
theorem expected_ne_zeros {R : ByteArray} (hR : R.size = 64) (hz : R.data.all (· == 0) = false) : expected R ≠ zeros := by
  intro h
  rw [expected_zero] at h
  have := expected_inj hR rfl h
  subst this
  have ht : (Array.replicate 64 (0 : UInt8)).all (· == 0) = true := by
    rw [Array.all_eq_true]; intro i hi; simp
  dsimp only at hz
  rw [ht] at hz
  exact Bool.noConfusion hz

theorem bytesOf_size (ws : List Nat) : (bytesOf ws).size = 2 * ws.length := by
  induction ws with
  | nil => rfl
  | cons w ws ih =>
    simp only [bytesOf, List.flatMap_cons] at ih ⊢
    simp [ByteArray.size, Array.size] at ih ⊢
    omega

theorem sibsOf_spec {aux : List Nat} {D : Nat} (ha : aux.length = auxWords D) :
    (RecLinkRef.sibsOf aux D).length = D ∧ ∀ s ∈ RecLinkRef.sibsOf aux D, s.size = 64 := by
  refine ⟨by simp [RecLinkRef.sibsOf], ?_⟩
  intro s hs
  simp only [RecLinkRef.sibsOf, List.mem_map, List.mem_range] at hs
  obtain ⟨h, hh, rfl⟩ := hs
  have hb : 32 * D ≤ auxWords D := by
    unfold auxWords NODE_WORDS
    have := Nat.le_max_right 1 ((32 * D + 63) / 64)
    have : 32 * D ≤ (32 * D + 63) / 64 * 64 := by omega
    exact Nat.le_trans this (Nat.mul_le_mul_right _ (by assumption))
  rw [bytesOf_size, List.length_take, List.length_drop, ha]
  unfold NODE_WORDS
  have : 32 * h + 32 ≤ 32 * D := by omega
  omega

theorem climb_size (ms : MerkleScheme) (hs : ms.Sized) : ∀ (l : List ByteArray) (acc : ByteArray) (idx : Nat),
    acc.size = ms.digestLen → (climb ms acc idx l).size = ms.digestLen
  | [], _, _, h => h
  | s :: l, acc, idx, _ => by
    simp only [climb]
    apply climb_size ms hs l
    split <;> exact hs.1 _ _

/-- An output `linkRef` returns is `linkOut`'s, its ports of `portWords`' sizes. -/
theorem linkRef_ok {UB PB : Nat} {trow xr yr raux xw yw waux : List Nat} {out : List Nat × List Nat}
    (h : linkRef UB PB trow xr yr raux xw yw waux = .ok out) :
    [trow, xr, yr, raux, xw, yw, waux].map (·.length) = portWords UB PB ∧
      out = linkOut UB PB trow xr yr raux xw yw waux := by
  unfold linkRef at h
  cases hs : checkShape UB PB with
  | error _ => simp [hs, bind, Except.bind] at h
  | ok _ =>
    simp [hs, bind, Except.bind, pure, Except.pure] at h
    split at h
    · rename_i hl
      split at h
      · injection h with h
        exact ⟨by simpa using hl, h.symm⟩
      · simp [throw, throwThe, MonadExceptOf.throw] at h
    · simp [throw, throwThe, MonadExceptOf.throw] at h

theorem commitString_pos (x y : ByteArray) : 0 < (Hm96.Default512.commitString x y).size := by
  unfold Hm96.Default512.commitString
  rw [ByteArray.size_append, Sha512.hash_size]
  omega

/-- **One side's opening**: a climb to a rows tree's root from the leaf of `commitString(x, y)` at row `f2` of unit `f1`
is that row's, or its `treePair` collides. -/
theorem opens {UB PB : Nat} {rows : List ((Nat × Nat) × ByteArray)} {R : ByteArray} {f1 f2 : Nat} {x y aux : List Nat}
    (hR : rowsRoot UB PB rows = .ok R) (hf1 : f1 < 2 ^ UB) (hf2 : f2 < 2 ^ PB) (haux : aux.length = auxWords (UB + PB))
    (hc : climbRef (leafOf x y) (RecLinkRef.sibsOf aux (UB + PB)) (index f1 f2 PB) = R) :
    ((f1, f2), Hm96.Default512.commitString (bytesOf x) (bytesOf y)) ∈ rows ∨
      Collides (treePair (eOf (UB + PB)) (UB + PB) PB rows (index f1 f2 PB)
        (Hm96.Default512.commitString (bytesOf x) (bytesOf y)) (RecLinkRef.sibsOf aux (UB + PB))) := by
  obtain ⟨hRe, hfit, -⟩ := rowsRoot_ok hR
  obtain ⟨hl, hs⟩ := sibsOf_spec haux
  rcases tree_binding (eOf_empties _) (fun k hk => eOf_size hk) (eOf_zero _) (good_leavesOf hfit) (index_lt hf1 hf2)
      (commitString_pos _ _) hs hl (by rw [← hRe]; exact hc) with ⟨z, hz, hidx, hbc⟩ | hcol
  · left
    obtain ⟨hu, hr⟩ := index_inj (hfit z hz).2 hf2 hidx
    obtain ⟨⟨u, r⟩, b⟩ := z
    simp only at hu hr hbc
    subst hu hr hbc
    exact hz
  · exact Or.inr hcol

/-- The reader's extractor: its opening's `treePair` in the input rows' tree, at the unit and port the row names. -/
def readerPair (UB PB : Nat) (rowsIn : List ((Nat × Nat) × ByteArray)) (trow xr yr raux : List Nat) :
    ByteArray × ByteArray :=
  treePair (eOf (UB + PB)) (UB + PB) PB rowsIn (index (field (bytesOf trow) 1) (field (bytesOf trow) 2) PB)
    (Hm96.Default512.commitString (bytesOf xr) (bytesOf yr)) (RecLinkRef.sibsOf raux (UB + PB))

/-- The writer's extractor: its opening's `treePair` in the output rows' tree, at the unit and row the row names. -/
def writerPair (UB PB : Nat) (rowsOut : List ((Nat × Nat) × ByteArray)) (trow xw yw waux : List Nat) :
    ByteArray × ByteArray :=
  treePair (eOf (UB + PB)) (UB + PB) PB rowsOut (index (field (bytesOf trow) 3) (field (bytesOf trow) 4) PB)
    (Hm96.Default512.commitString (bytesOf xw) (bytesOf yw)) (RecLinkRef.sibsOf waux (UB + PB))

theorem root_size {UB PB : Nat} {rows : List ((Nat × Nat) × ByteArray)} {R : ByteArray}
    (hR : rowsRoot UB PB rows = .ok R) : R.size = 64 := by
  obtain ⟨hRe, hfit, -⟩ := rowsRoot_ok hR
  rw [hRe]
  exact sub_size _ _ _ (good_leavesOf hfit) (eOf_size (Nat.le_refl _))

theorem climbRef_size (start : ByteArray) (sibs : List ByteArray) (pos : Nat) (h : start.size = 64) :
    (climbRef start sibs pos).size = 64 :=
  climb_size scheme (MerkleScheme.sha512_sized _) sibs start pos h

/-- **An accepted link row binds its reader to its writer.** The verifier's two rows trees over the inner statement's
rows (`rowsIn` the units' input rows, `rowsOut` their output rows, each `((unit, row), b ‖ c)`), and a link row whose unit
`RecLink_v1{UB, PB}` (at its reference) outputs the two roots as registered rows: the input row at the reader's unit and
port publishes `commitString(xr, yr)`, the output row at the writer's unit and row publishes `commitString(xw, yw)`, and
`xr = xw`, the one digest both commit; or the reader's or the writer's extractor names a SHA-512 collision. -/
theorem link_binding {UB PB : Nat} {rowsIn rowsOut : List ((Nat × Nat) × ByteArray)} {R W : ByteArray}
    {trow xr yr raux xw yw waux : List Nat}
    (hR : rowsRoot UB PB rowsIn = .ok R) (hW : rowsRoot UB PB rowsOut = .ok W)
    (hout : linkRef UB PB trow xr yr raux xw yw waux = .ok (expected R, expected W))
    (hk : field (bytesOf trow) 0 = KIND_LINK) :
    (((field (bytesOf trow) 1, field (bytesOf trow) 2), Hm96.Default512.commitString (bytesOf xr) (bytesOf yr)) ∈
        rowsIn ∧
      ((field (bytesOf trow) 3, field (bytesOf trow) 4), Hm96.Default512.commitString (bytesOf xw) (bytesOf yw)) ∈
        rowsOut ∧
      bytesOf xr = bytesOf xw) ∨
    Collides (readerPair UB PB rowsIn trow xr yr raux) ∨ Collides (writerPair UB PB rowsOut trow xw yw waux) := by
  obtain ⟨hlen, hlo⟩ := linkRef_ok hout
  simp only [List.map_cons, List.map_nil, portWords, List.cons.injEq] at hlen
  obtain ⟨-, -, -, hraux, -, -, hwaux, -⟩ := hlen
  have hRs := root_size hR
  have hWs := root_size hW
  have hne : (linkOut UB PB trow xr yr raux xw yw waux).1 ≠ zeros := by
    rw [← hlo]
    exact expected_ne_zeros hRs (rowsRoot_ok hR).2.2
  obtain ⟨hok, heq⟩ := linkOut_meaning (Or.inl hne)
  rw [← hlo, Prod.mk.injEq] at heq
  have hrR : rootR UB PB trow xr yr raux = R :=
    (expected_inj hRs (climbRef_size _ _ _ (Sha512.hash_size _)) heq.1).symm
  have hrW : rootW UB PB trow xw yw waux = W :=
    (expected_inj hWs (climbRef_size _ _ _ (Sha512.hash_size _)) heq.2).symm
  obtain ⟨-, h1, h3, h2, h4⟩ := okRef_fits hok
  have hx := okRef_link hok hk
  unfold rootR at hrR
  unfold rootW at hrW
  rw [posOf_fits h1 h2] at hrR
  rw [posOf_fits h3 h4] at hrW
  rcases opens hR h1 h2 hraux hrR with hin | hcr
  · rcases opens hW h3 h4 hwaux hrW with hon | hcw
    · exact Or.inl ⟨hin, hon, hx⟩
    · exact Or.inr (Or.inr hcw)
  · exact Or.inr (Or.inl hcr)

/-- `link_binding` as a proposition: the rows bind, or SHA-512 has a collision. -/
theorem link_binding_collision {UB PB : Nat} {rowsIn rowsOut : List ((Nat × Nat) × ByteArray)} {R W : ByteArray}
    {trow xr yr raux xw yw waux : List Nat}
    (hR : rowsRoot UB PB rowsIn = .ok R) (hW : rowsRoot UB PB rowsOut = .ok W)
    (hout : linkRef UB PB trow xr yr raux xw yw waux = .ok (expected R, expected W))
    (hk : field (bytesOf trow) 0 = KIND_LINK) :
    (((field (bytesOf trow) 1, field (bytesOf trow) 2), Hm96.Default512.commitString (bytesOf xr) (bytesOf yr)) ∈
        rowsIn ∧
      ((field (bytesOf trow) 3, field (bytesOf trow) 4), Hm96.Default512.commitString (bytesOf xw) (bytesOf yw)) ∈
        rowsOut ∧
      bytesOf xr = bytesOf xw) ∨ Sha512.Collision := by
  rcases link_binding hR hW hout hk with h | h | h
  · exact Or.inl h
  · exact Or.inr h.collision
  · exact Or.inr h.collision

end Flock.RecLinkBind
