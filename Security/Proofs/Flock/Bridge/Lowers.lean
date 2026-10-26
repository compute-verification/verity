import Proofs.Flock.Bridge.Defs
import Proofs.Flock.Claims.Bridge.WrongSet
import Proofs.Flock.SamplingBoundary
import Proofs.Flock.Soundness.Audit.Circuit

/-!
# The lowering is correct, unit by unit

`lowers`: for every circuit `C`, constants `κ`, partition `P` and unit `u`, `WrongSet.Lowers (progOf C κ)
[unitGates P u] φ Xv`, at any `φ` that inverts `posOf` (`wireAt C d` does, `wireAt_posOf`) and any evaluation `Xv` that
is `κ`'s constant at each constant wire of `P.io u`. So the claims program of a statement is `progOf C κ`, defined from
the parsed circuit, and its unit `u` computes what `C` computes there; `faulty_card_le_wrongRegZ` reads it at
`r u := [unitGates P u]`, slot `u` of `partOf P` (`rus_partOf_getElem?`).

* Where the lowered gates sit: wire `g`'s `gateSize` gates are at `off C g` onward (`gates_off`), every gate of
  `progOf C κ` is one of them (`exists_block`), and two wires' blocks don't meet (`block_inj`).
* An input `κ` fixes to `b` is lowered to `.const b` at its position, and no other gate is a constant (`posOf_const`,
  `unitGates_ne_const`). A unit reads such an input with no opening, as `b`, which is its value in `C.eval` on any
  evaluation that agrees with `κ`.
* A row `(⊕ a) ∧ (⊕ b)` is lowered to the XOR of `a`'s positions, the XOR of `b`'s and their AND, at `posOf C g`. The
  two XORs are interior to the unit (`boundary_inner`): no output, input or gate of another unit reads them, so `Correct`
  asks nothing of their committed values, and the AND reads them as the unit computes them.
* `runIn_posOf`: by induction over the wires, each wire's position computes its value in `C.eval`, reading from outside
  the unit only positions of wires of `P.io u`: a constant's value is `κ`'s, any other's committed value `Lowers`
  gives.
-/

namespace Flock.Bridge

open FlockSoundness.Audit Claims.Sampling

/-! ## Indexing a `flatMap` -/

theorem getElem?_flatMap_take {α β : Type} (f : α → List β) :
    ∀ (l : List α) (k : ℕ) (hk : k < l.length) (j : ℕ), j < (f l[k]).length →
      (l.flatMap f)[((l.take k).flatMap f).length + j]? = (f l[k])[j]?
  | [], _, hk, _, _ => absurd hk (by simp)
  | a :: t, 0, _, j, hj => by
    simp only [List.getElem_cons_zero] at hj
    simp only [List.take_zero, List.flatMap_nil, List.length_nil, Nat.zero_add, List.flatMap_cons,
      List.getElem_cons_zero]
    exact List.getElem?_append_left hj
  | a :: t, k + 1, hk, j, hj => by
    simp only [List.flatMap_cons, List.take_succ_cons, List.length_append, List.getElem_cons_succ]
    rw [Nat.add_assoc, List.getElem?_append_right (by omega), Nat.add_sub_cancel_left]
    exact getElem?_flatMap_take f t k (by simpa using hk) j hj

theorem exists_of_lt_flatMap {α β : Type} (f : α → List β) :
    ∀ (l : List α) (i : ℕ), i < (l.flatMap f).length →
      ∃ k, ∃ hk : k < l.length, ∃ j < (f l[k]).length, i = ((l.take k).flatMap f).length + j
  | [], i, hi => by simp at hi
  | a :: t, i, hi => by
    by_cases h : i < (f a).length
    · exact ⟨0, by simp, i, by simpa using h, by simp⟩
    · simp only [List.flatMap_cons, List.length_append] at hi
      obtain ⟨k, hk, j, hj, hi'⟩ := exists_of_lt_flatMap f t (i - (f a).length) (by omega)
      refine ⟨k + 1, by simpa using hk, j, by simpa using hj, ?_⟩
      simp only [List.take_succ_cons, List.flatMap_cons, List.length_append]
      omega

/-! ## Where each wire's lowered gates sit -/

variable {C : FlockSoundness.Audit.Circuit} {κ : Fin C.N → Option Bool}

theorem one_le_gateSize {N : ℕ} (o : Op N) : 1 ≤ gateSize o := by
  cases o <;> simp [gateSize]

theorem length_lower (g : Fin C.N) : (lower C κ g).length = gateSize (C.op g) := by
  unfold lower
  split <;> (try split) <;> simp_all [gateSize]

theorem getElem_finRange_val (g : Fin C.N) :
    (List.finRange C.N)[g.val]'(by simp) = g := by
  simp

theorem mem_take_finRange {k : ℕ} {h : Fin C.N} : h ∈ (List.finRange C.N).take k ↔ h.val < k := by
  rw [List.mem_take_iff_getElem]
  constructor
  · rintro ⟨j, hj, rfl⟩
    simp only [List.getElem_finRange, Fin.cast_mk]
    simp at hj
    omega
  · intro hh
    exact ⟨h.val, by simp; omega, by simp⟩

theorem off_eq_sum (k : ℕ) :
    off C k = ∑ h ∈ Finset.univ.filter (fun h : Fin C.N => h.val < k), gateSize (C.op h) := by
  classical
  unfold off
  rw [← List.sum_toFinset _ ((List.nodup_finRange C.N).filter _)]
  congr 1
  ext h
  simp

theorem off_eq_length (k : ℕ) : off C k = (((List.finRange C.N).take k).flatMap (lower C κ)).length := by
  classical
  rw [List.length_flatMap, off_eq_sum]
  simp only [length_lower (κ := κ)]
  rw [← List.sum_toFinset _ ((List.nodup_finRange C.N).sublist (List.take_sublist _ _))]
  congr 1
  ext h
  simp [mem_take_finRange]

theorem off_add_le {g h : Fin C.N} (hgh : g < h) : off C g.val + gateSize (C.op g) ≤ off C h.val := by
  classical
  have hs : ∑ x ∈ insert g (Finset.univ.filter fun h : Fin C.N => h.val < g.val), gateSize (C.op x) =
      gateSize (C.op g) + ∑ x ∈ Finset.univ.filter (fun h : Fin C.N => h.val < g.val), gateSize (C.op x) :=
    Finset.sum_insert (by simp)
  rw [off_eq_sum, off_eq_sum, add_comm, ← hs]
  apply Finset.sum_le_sum_of_subset
  intro x hx
  simp only [Finset.mem_insert, Finset.mem_filter, Finset.mem_univ, true_and] at hx ⊢
  rcases hx with rfl | hx
  · exact hgh
  · exact lt_trans hx hgh

/-- Two wires' blocks of lowered gates don't meet. -/
theorem block_inj {g h : Fin C.N} {j k : ℕ} (hj : j < gateSize (C.op g)) (hk : k < gateSize (C.op h))
    (he : off C g.val + j = off C h.val + k) : g = h ∧ j = k := by
  rcases lt_trichotomy g h with hlt | rfl | hlt
  · have := off_add_le hlt; omega
  · exact ⟨rfl, by omega⟩
  · have := off_add_le hlt; omega

theorem posOf_eq (g : Fin C.N) : posOf C g = off C g.val + (gateSize (C.op g) - 1) := by
  have := one_le_gateSize (C.op g)
  unfold posOf
  omega

theorem posOf_lt_off {a g : Fin C.N} (h : a < g) : posOf C a < off C g.val := by
  have := off_add_le h
  have := one_le_gateSize (C.op a)
  rw [posOf_eq]
  omega

/-- The lowered program's gate at `off g + j` is `lower g`'s `j`th. -/
theorem gates_off (g : Fin C.N) {j : ℕ} (hj : j < gateSize (C.op g)) :
    (progOf C κ).gates[off C g.val + j]? = (lower C κ g)[j]? := by
  have h := getElem?_flatMap_take (lower C κ) (List.finRange C.N) g.val (by simp) j
    (by rw [getElem_finRange_val, length_lower]; exact hj)
  rw [getElem_finRange_val] at h
  rw [off_eq_length (κ := κ)]
  exact h

/-- Every gate of the lowered program is some wire's `j`th lowered gate. -/
theorem exists_block {i : ℕ} {x : BGate} (hx : (progOf C κ).gates[i]? = some x) :
    ∃ g : Fin C.N, ∃ j < gateSize (C.op g), i = off C g.val + j ∧ (lower C κ g)[j]? = some x := by
  have hi : i < (progOf C κ).gates.length := (List.getElem?_eq_some_iff.1 hx).1
  obtain ⟨k, hk, j, hj, rfl⟩ := exists_of_lt_flatMap (lower C κ) (List.finRange C.N) i hi
  have hkN : k < C.N := by simpa using hk
  have hg : (List.finRange C.N)[k] = (⟨k, hkN⟩ : Fin C.N) := by simp
  rw [hg, length_lower] at hj
  refine ⟨⟨k, hkN⟩, j, hj, by rw [off_eq_length (κ := κ)], ?_⟩
  rw [← gates_off _ hj, off_eq_length (κ := κ)]
  exact hx

theorem mem_unitGates {n : ℕ} {P : Partition C n} {u : Fin n} {i : ℕ} :
    i ∈ unitGates P u ↔ ∃ g : Fin C.N, P.unit g = some u ∧ ∃ j < gateSize (C.op g), i = off C g.val + j := by
  simp only [unitGates, List.mem_flatMap, List.mem_finRange, true_and]
  constructor
  · rintro ⟨g, hg⟩
    split_ifs at hg with hu
    · obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hg
      exact ⟨g, hu, j, List.mem_range.1 hj, rfl⟩
    · simp at hg
  · rintro ⟨g, hu, j, hj, rfl⟩
    exact ⟨g, by simp only [hu, ↓reduceIte]; exact List.mem_map.2 ⟨j, List.mem_range.2 hj, rfl⟩⟩

theorem posOf_mem_unitGates {n : ℕ} {P : Partition C n} {u : Fin n} {a : Fin C.N} :
    posOf C a ∈ unitGates P u ↔ P.unit a = some u := by
  have h1 := one_le_gateSize (C.op a)
  rw [mem_unitGates]
  constructor
  · rintro ⟨g, hu, j, hj, he⟩
    rw [posOf_eq] at he
    obtain ⟨rfl, -⟩ := block_inj (by omega) hj he
    exact hu
  · intro hu
    exact ⟨a, hu, _, by omega, posOf_eq a⟩

/-! ## The lowered gates' reads and values -/

/-- A gate other than an input is lowered to no constant. -/
theorem lower_ne_const {g : Fin C.N} (hg : (C.op g).isInput = false) (b : Bool) :
    ∀ x ∈ lower C κ g, x ≠ .const b := by
  unfold lower
  split <;> simp_all [Op.isInput]

/-- **The lowered program's constants** are the inputs `κ` fixes, each at its wire's position. -/
theorem posOf_const {a : Fin C.N} {b : Bool} (h : (progOf C κ).gates[posOf C a]? = some (.const b)) :
    C.op a = .input ∧ κ a = some b := by
  have := one_le_gateSize (C.op a)
  rw [posOf_eq, gates_off a (by omega)] at h
  cases hop : C.op a with
  | input =>
    refine ⟨rfl, ?_⟩
    simp only [lower, hop, gateSize] at h
    cases hk : κ a <;> simp_all
  | _ => simp [lower, hop, gateSize] at h

/-- A unit's gates are none of them constants: the input gates are in no unit. -/
theorem unitGates_ne_const {n : ℕ} {P : Partition C n} {u : Fin n} {p : ℕ} (hp : p ∈ unitGates P u) (b : Bool) :
    (progOf C κ).gates[p]? ≠ some (.const b) := by
  intro h
  obtain ⟨g, hu, j, hj, rfl⟩ := mem_unitGates.1 hp
  rw [gates_off g hj] at h
  have hni : (C.op g).isInput = false := by
    cases hi : (C.op g).isInput
    · rfl
    · exact absurd ((P.input_iff g).2 hi) (by simp [hu])
  exact lower_ne_const hni b _ (List.mem_of_getElem? h) rfl

/-- A wire a gate of unit `u` reads from outside `u` is one of `u`'s committed inputs. -/
theorem mem_io_of_arg {n : ℕ} {P : Partition C n} {u : Fin n} {g a : Fin C.N} (hu : P.unit g = some u)
    (ha : a ∈ (C.op g).args) (hout : posOf C a ∉ unitGates P u) : a ∈ P.io u := by
  classical
  have hne : P.unit g ≠ P.unit a := by
    rw [hu]
    exact fun h => hout (posOf_mem_unitGates.2 h.symm)
  simp only [FlockSoundness.Audit.Partition.io, Finset.mem_filter]
  exact ⟨P.mem_committed_of_read ha hne, Or.inr ⟨g, hu, ha⟩⟩

/-- A lowered gate reads the positions of the wires its gate reads, except a row's AND, which reads its own two XORs. -/
theorem reads_lower {g : Fin C.N} {j : ℕ} {x : BGate} (hx : (lower C κ g)[j]? = some x) {i : ℕ} (hi : i ∈ x.reads) :
    (∃ a ∈ (C.op g).args, i = posOf C a) ∨ (j = 2 ∧ (i = off C g.val ∨ i = off C g.val + 1)) := by
  cases hop : C.op g with
  | input =>
    simp only [lower, hop] at hx
    cases hk : κ g <;> simp only [hk] at hx <;> rcases j with _ | j <;> simp at hx <;> subst hx <;>
      simp [BGate.reads] at hi
  | and a b =>
    simp only [lower, hop] at hx
    rcases j with _ | j <;> simp at hx
    subst hx
    simp only [BGate.reads, List.mem_cons, List.not_mem_nil, or_false] at hi
    rcases hi with rfl | rfl <;> exact Or.inl ⟨_, by simp [Op.args], rfl⟩
  | xor a b =>
    simp only [lower, hop] at hx
    rcases j with _ | j <;> simp at hx
    subst hx
    simp only [BGate.reads, List.mem_cons, List.not_mem_nil, or_false] at hi
    rcases hi with rfl | rfl <;> exact Or.inl ⟨_, by simp [Op.args], rfl⟩
  | not a =>
    simp only [lower, hop] at hx
    rcases j with _ | j <;> simp at hx
    subst hx
    simp only [BGate.reads, List.mem_cons, List.not_mem_nil, or_false] at hi
    exact Or.inl ⟨a, by simp [Op.args], hi⟩
  | row a b =>
    simp only [lower, hop] at hx
    rcases j with _ | _ | _ | j <;> simp at hx <;> subst hx <;> simp only [BGate.reads, List.mem_map,
      List.mem_cons, List.not_mem_nil, or_false] at hi
    · obtain ⟨a', ha', rfl⟩ := hi
      exact Or.inl ⟨a', by simp [Op.args, ha'], rfl⟩
    · obtain ⟨a', ha', rfl⟩ := hi
      exact Or.inl ⟨a', by simp [Op.args, ha'], rfl⟩
    · exact Or.inr ⟨rfl, hi⟩

theorem off_ne_posOf {g a : Fin C.N} {j : ℕ} (hj : j + 1 < gateSize (C.op g)) : off C g.val + j ≠ posOf C a := by
  intro he
  have := one_le_gateSize (C.op a)
  rw [posOf_eq] at he
  obtain ⟨rfl, -⟩ := block_inj (by omega) (by omega) he
  omega

/-- **A row's two XORs are interior**: no output, input or gate outside the unit reads them. -/
theorem boundary_inner {n : ℕ} {P : Partition C n} {u : Fin n} {g : Fin C.N} (hu : P.unit g = some u) {j : ℕ}
    (hj : j + 1 < gateSize (C.op g)) : (progOf C κ).Boundary (unitGates P u) (off C g.val + j) = false := by
  obtain ⟨a, b, hab⟩ : ∃ a b, C.op g = .row a b := by
    cases h : C.op g <;> simp only [h, gateSize] at hj <;> try omega
    exact ⟨_, _, rfl⟩
  have hj2 : j < 2 := by simp only [hab, gateSize] at hj; omega
  cases hB : (progOf C κ).Boundary (unitGates P u) (off C g.val + j)
  · rfl
  exfalso
  simp only [Prog.Boundary, Bool.or_eq_true, List.contains_iff_mem, decide_eq_true_eq, List.any_eq_true,
    List.mem_range, Bool.and_eq_true, decide_not, Bool.not_eq_true', decide_eq_false_iff_not] at hB
  rcases hB with (hout | hin) | ⟨g', -, hg'u, hrd⟩
  · simp only [progOf, List.mem_map] at hout
    obtain ⟨a', -, he⟩ := hout
    exact off_ne_posOf hj he.symm
  · rw [gates_off g (by omega)] at hin
    simp only [lower, hab] at hin
    rcases j with _ | _ | j <;> simp at hin
    omega
  · cases hx : (progOf C κ).gates[g']? with
    | none => simp [hx] at hrd
    | some x =>
      simp only [hx, Option.map_some, Option.getD_some] at hrd
      obtain ⟨h, k, hk, rfl, hxk⟩ := exists_block hx
      rcases reads_lower hxk hrd with ⟨a', -, ha'⟩ | ⟨rfl, hi⟩
      · exact off_ne_posOf hj ha'
      · have hh : g = h := by
          rcases hi with hi | hi
          · exact (block_inj (j := j) (k := 0) (by omega) (by omega) (by simpa using hi)).1
          · exact (block_inj (j := j) (k := 1) (by omega) (by omega) hi).1
        subst hh
        exact hg'u (mem_unitGates.2 ⟨g, hu, 2, hk, rfl⟩)

/-- What a gate at `p` of the set of gates `U` reads at `i`, as `Prog.runIn` reads it. -/
def feed (Cp : Claims.Sampling.Prog) (ref : ℕ → Option Bool) (U : List ℕ) (p i : ℕ) : Option Bool :=
  if i ∈ U then (if i < p then Cp.runIn ref U i else none) else Cp.read ref i

theorem runIn_feed (Cp : Claims.Sampling.Prog) (ref : ℕ → Option Bool) (U : List ℕ) (p : ℕ) :
    Cp.runIn ref U p = Cp.gates[p]?.bind fun x => x.val (feed Cp ref U p) (ref p) :=
  Boundary.runIn_eq Cp ref U p

theorem mapM_posOf {rd : ℕ → Option Bool} {v : Fin C.N → Bool} :
    ∀ ws : List (Fin C.N), (∀ a ∈ ws, rd (posOf C a) = some (v a)) → (ws.map (posOf C)).mapM rd = some (ws.map v)
  | [], _ => rfl
  | a :: ws, h => by
    simp [List.mapM_cons, h a List.mem_cons_self, mapM_posOf ws fun b hb => h b (List.mem_cons_of_mem _ hb)]

theorem foldl_xor (l : List Bool) (b : Bool) : l.foldl Bool.xor b = Bool.xor b (l.foldr Bool.xor false) := by
  induction l generalizing b with
  | nil => simp
  | cons x l ih => simp [ih]

theorem val_xorList {rd : ℕ → Option Bool} {v : Fin C.N → Bool} (ws : List (Fin C.N)) (own : Option Bool)
    (h : ∀ a ∈ ws, rd (posOf C a) = some (v a)) :
    (BGate.xor (ws.map (posOf C))).val rd own = some (xorSum ws v) := by
  simp [BGate.val, mapM_posOf ws h, xorSum, foldl_xor]

/-- **Each wire's lowered gates compute its value** in the unit holding it, on committed values `ref` that carry the
circuit's evaluation at every position of a committed input the unit reads from outside. -/
theorem runIn_posOf {n : ℕ} (P : Partition C n) (κ : Fin C.N → Option Bool) (u : Fin n) (Xv : Fin C.N → Bool) (ref : ℕ → Option Bool)
    (hread : ∀ p ∈ unitGates P u, ∀ x, (progOf C κ).gates[p]? = some x → ∀ a : Fin C.N, posOf C a ∈ x.reads →
      posOf C a ∉ unitGates P u → a ∈ P.io u → (progOf C κ).read ref (posOf C a) = some (C.eval Xv a)) :
    ∀ m (g : Fin C.N), g.val = m → P.unit g = some u →
      (progOf C κ).runIn ref (unitGates P u) (posOf C g) = some (C.eval Xv g) := by
  intro m
  induction m using Nat.strong_induction_on with
  | _ m ih =>
  intro g hgm hu
  have hfeed : ∀ a : Fin C.N, a ∈ (C.op g).args → ∀ p ∈ unitGates P u, off C g.val ≤ p →
      ∀ x, (progOf C κ).gates[p]? = some x → posOf C a ∈ x.reads →
      feed (progOf C κ) ref (unitGates P u) p (posOf C a) = some (C.eval Xv a) := by
    intro a ha p hp hp' x hx hr
    have hag := C.topo g a ha
    have hlt : posOf C a < p := lt_of_lt_of_le (posOf_lt_off hag) hp'
    unfold feed
    by_cases hin : posOf C a ∈ unitGates P u
    · simp only [hin, hlt, ↓reduceIte]
      exact ih a.val (hgm ▸ hag) a rfl (posOf_mem_unitGates.1 hin)
    · simp only [hin, ↓reduceIte]
      exact hread p hp x hx a hr hin (mem_io_of_arg hu ha hin)
  have hni : (C.op g).isInput = false := by
    cases h : (C.op g).isInput
    · rfl
    · exact absurd ((P.input_iff g).2 h) (by simp [hu])
  have hmem : ∀ j < gateSize (C.op g), off C g.val + j ∈ unitGates P u :=
    fun j hj => mem_unitGates.2 ⟨g, hu, j, hj, rfl⟩
  have hgate : ∀ j (hj : j < gateSize (C.op g)), (progOf C κ).gates[off C g.val + j]? = (lower C κ g)[j]? :=
    fun j hj => gates_off g hj
  have hin : ∀ j < gateSize (C.op g), ∀ x, (progOf C κ).gates[off C g.val + j]? = some x →
      ∀ a ∈ (C.op g).args, posOf C a ∈ x.reads →
      feed (progOf C κ) ref (unitGates P u) (off C g.val + j) (posOf C a) = some (C.eval Xv a) :=
    fun j hj x hx a ha hr => hfeed a ha _ (hmem j hj) (Nat.le_add_right _ _) x hx hr
  rw [C.eval_op Xv hni, runIn_feed, posOf_eq]
  cases hop : C.op g with
  | input => simp [hop, Op.isInput] at hni
  | and a b =>
    have h0 : (progOf C κ).gates[off C g.val + 0]? = some (.and [posOf C a, posOf C b]) := by
      simpa [lower, hop] using hgate 0 (by simp [hop, gateSize])
    have hs : gateSize (Op.and a b : Op C.N) - 1 = 0 := rfl
    have h1 : 0 < gateSize (C.op g) := by simp [hop, gateSize]
    rw [hs, h0, Option.bind_some]
    have fa := hin 0 h1 _ h0 a (by simp [hop, Op.args]) (by simp [BGate.reads])
    have fb := hin 0 h1 _ h0 b (by simp [hop, Op.args]) (by simp [BGate.reads])
    simp only [Nat.add_zero] at fa fb
    simp [BGate.val, List.mapM_cons, fa, fb, Op.apply]
  | xor a b =>
    have h0 : (progOf C κ).gates[off C g.val + 0]? = some (.xor [posOf C a, posOf C b]) := by
      simpa [lower, hop] using hgate 0 (by simp [hop, gateSize])
    have hs : gateSize (Op.xor a b : Op C.N) - 1 = 0 := rfl
    have h1 : 0 < gateSize (C.op g) := by simp [hop, gateSize]
    rw [hs, h0, Option.bind_some]
    have fa := hin 0 h1 _ h0 a (by simp [hop, Op.args]) (by simp [BGate.reads])
    have fb := hin 0 h1 _ h0 b (by simp [hop, Op.args]) (by simp [BGate.reads])
    simp only [Nat.add_zero] at fa fb
    simp [BGate.val, List.mapM_cons, fa, fb, Op.apply]
  | not a =>
    have h0 : (progOf C κ).gates[off C g.val + 0]? = some (.not (posOf C a)) := by
      simpa [lower, hop] using hgate 0 (by simp [hop, gateSize])
    have hs : gateSize (Op.not a : Op C.N) - 1 = 0 := rfl
    have h1 : 0 < gateSize (C.op g) := by simp [hop, gateSize]
    rw [hs, h0, Option.bind_some]
    have fa := hin 0 h1 _ h0 a (by simp [hop, Op.args]) (by simp [BGate.reads])
    simp only [Nat.add_zero] at fa
    simp [BGate.val, fa, Op.apply]
  | row a b =>
    have h3 : gateSize (C.op g) = 3 := by simp [hop, gateSize]
    have h0 : (progOf C κ).gates[off C g.val + 0]? = some (.xor (a.map (posOf C))) := by
      simpa [lower, hop] using hgate 0 (by omega)
    have h1 : (progOf C κ).gates[off C g.val + 1]? = some (.xor (b.map (posOf C))) := by
      simpa [lower, hop] using hgate 1 (by omega)
    have h2 : (progOf C κ).gates[off C g.val + 2]? = some (.and [off C g.val, off C g.val + 1]) := by
      simpa [lower, hop] using hgate 2 (by omega)
    have hs : gateSize (Op.row a b : Op C.N) - 1 = 2 := rfl
    rw [hs, h2, Option.bind_some]
    have r0 : feed (progOf C κ) ref (unitGates P u) (off C g.val + 2) (off C g.val) =
        some (xorSum a (C.eval Xv)) := by
      have hm := hmem 0 (by omega)
      rw [Nat.add_zero] at hm
      unfold feed
      simp only [hm, show off C g.val < off C g.val + 2 by omega, ↓reduceIte]
      rw [runIn_feed, ← Nat.add_zero (off C g.val), h0, Option.bind_some]
      exact val_xorList a _ fun a' ha' =>
        hin 0 (by omega) _ h0 a' (by simp [hop, Op.args, ha']) (by simp [BGate.reads]; exact ⟨a', ha', rfl⟩)
    have r1 : feed (progOf C κ) ref (unitGates P u) (off C g.val + 2) (off C g.val + 1) =
        some (xorSum b (C.eval Xv)) := by
      unfold feed
      simp only [hmem 1 (by omega), show off C g.val + 1 < off C g.val + 2 by omega, ↓reduceIte]
      rw [runIn_feed, h1, Option.bind_some]
      exact val_xorList b _ fun b' hb' =>
        hin 1 (by omega) _ h1 b' (by simp [hop, Op.args, hb']) (by simp [BGate.reads]; exact ⟨b', hb', rfl⟩)
    simp [BGate.val, List.mapM_cons, r0, r1, Op.apply]

/-! ## The lowering is correct, unit by unit -/

theorem posOf_injective : Function.Injective (posOf C) := by
  intro a b h
  have := one_le_gateSize (C.op a)
  have := one_le_gateSize (C.op b)
  rw [posOf_eq, posOf_eq] at h
  exact (block_inj (by omega) (by omega) h).1

open FlockSoundness.Discharge.ZkLink in
/-- **`Lowers` for every circuit**: unit `u`'s replay unit of `partOf P`, the one proof unit `unitGates P u`, is
correct in `progOf C κ` on every committed values that carry `C`'s evaluation `Xv` at the wires `φ` maps the committed
positions it reads or computes to, for any `φ` that inverts `posOf` and any `Xv` that is `κ`'s constant at each constant
wire of `P.io u` (the unit reads a constant with no opening). -/
theorem lowers {n : ℕ} (P : Partition C n) (κ : Fin C.N → Option Bool) (u : Fin n) (φ : ℕ → Fin C.N)
    (hφ : ∀ g, φ (posOf C g) = g) (Xv : Fin C.N → Bool)
    (hκ : ∀ g ∈ P.io u, ∀ b, κ g = some b → C.eval Xv g = b) :
    WrongSet.Lowers (progOf C κ) [unitGates P u] φ Xv := by
  intro ref href pu hpu
  rw [List.mem_singleton] at hpu
  subst hpu
  have hgates : RU.gates [unitGates P u] = unitGates P u := by simp [RU.gates]
  have hneed : ∀ p ∈ unitGates P u, ∀ x, (progOf C κ).gates[p]? = some x → ∀ i ∈ x.reads, i ∉ unitGates P u →
      ∃ a : Fin C.N, i = posOf C a ∧ (progOf C κ).read ref i = some (C.eval Xv a) := by
    intro p hp x hx i hi hout
    obtain ⟨g, hu, j, hj, hpj⟩ := mem_unitGates.1 hp
    have hx' := hx
    rw [hpj, gates_off g hj] at hx'
    rcases reads_lower hx' hi with ⟨a, ha, rfl⟩ | ⟨rfl, hi'⟩
    · refine ⟨a, rfl, ?_⟩
      by_cases hc : ∃ b, (progOf C κ).gates[posOf C a]? = some (.const b)
      · obtain ⟨b, hb⟩ := hc
        simp only [Prog.read, hb]
        rw [hκ a (mem_io_of_arg hu ha hout) b (posOf_const hb).2]
      · have hc' : ∀ b, (progOf C κ).gates[posOf C a]? ≠ some (.const b) := fun b h => hc ⟨b, h⟩
        rw [Boundary.read_eq _ hc']
        refine (href _ ?_ hc').trans (by rw [hφ])
        simp only [WrongSet.need, hgates, List.mem_append, List.mem_flatMap]
        exact Or.inr ⟨p, hp, by simp [hx, hi]⟩
    · exfalso
      apply hout
      rcases hi' with rfl | rfl
      · exact mem_unitGates.2 ⟨g, hu, 0, by omega, rfl⟩
      · exact mem_unitGates.2 ⟨g, hu, 1, by omega, rfl⟩
  have hself : ∀ p ∈ unitGates P u, ref p = some (C.eval Xv (φ p)) := fun p hp =>
    href p (by simp [WrongSet.need, hgates, hp]) (unitGates_ne_const hp)
  have hrun := runIn_posOf P κ u Xv ref fun p hp x hx a hr hout _ => by
    obtain ⟨a', he, h⟩ := hneed p hp x hx _ hr hout
    rw [h, posOf_injective he]
  intro p hp
  refine ⟨fun x hx i hi hiu => ?_, fun hb => ?_⟩
  · obtain ⟨a, -, h⟩ := hneed p hp x hx i hi hiu
    rw [h]
    simp
  obtain ⟨g, hu, j, hj, rfl⟩ := mem_unitGates.1 hp
  by_cases hjl : j + 1 < gateSize (C.op g)
  · rw [boundary_inner hu hjl] at hb
    exact absurd hb (by simp)
  · have hpos : off C g.val + j = posOf C g := by rw [posOf_eq]; omega
    rw [hpos] at hp ⊢
    exact ⟨_, by rw [hself _ hp, hφ], hrun g.val g rfl hu⟩

/-- **The wire at a position** of the lowered program: the wire it is the position of, `d` at any other. -/
def wireAt (C : FlockSoundness.Audit.Circuit) (d : Fin C.N) (i : ℕ) : Fin C.N :=
  ((List.finRange C.N).find? fun g => posOf C g = i).getD d

theorem wireAt_posOf (d g : Fin C.N) : wireAt C d (posOf C g) = g := by
  unfold wireAt
  obtain ⟨g', hg'⟩ : ∃ g', (List.finRange C.N).find? (fun h => posOf C h = posOf C g) = some g' :=
    Option.isSome_iff_exists.1 (List.find?_isSome.2 ⟨g, by simp, by simp⟩)
  rw [hg', Option.getD_some]
  exact posOf_injective (by simpa using List.find?_some hg')

theorem rus_partOf {n : ℕ} (P : Partition C n) :
    (partOf P).rus = (List.finRange n).map fun u => [unitGates P u] := by
  simp [partOf, Claims.Sampling.Partition.rus]

/-- Slot `u` of `partOf P` is unit `u`'s replay unit. -/
theorem rus_partOf_getElem? {n : ℕ} (P : Partition C n) (u : Fin n) :
    (partOf P).rus[u.val]? = some [unitGates P u] := by
  simp [rus_partOf]

end Flock.Bridge
