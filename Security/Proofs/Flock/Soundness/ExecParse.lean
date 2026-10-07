import Proofs.Flock.Soundness.ExecPlacement

/-!
# What the parsers accepted, and Δ's first loop

- `built_spec`: a net `Net.ofRows` built (`Built`: `Net.parse`'s text nets, or a typed class's `Typed.netOfD`) has its
  constant the last useful row, reading itself in `A` and in `B`, its input rows before the constant, and port groups
  inside its input rows (`portCols_le`).
- `delta_split`: `HmRow.delta`'s first loop writes exactly `constPairs` (every net slot's constant other than the pin,
  `(k, k)` and `(k, pin)`), and every later pair's row has one of the input-bit forms (`DeltaIn`).
-/

open Flock FlockLevel3

namespace FlockSoundness

/-! ## Walking `do` blocks -/

theorem ite_throw_ok' {α β : Type} {c : Prop} [Decidable c] {e : String} {f : α → Except String β}
    {g : Except String β} {v : β} (h : (if c then (throw e >>= f) else g) = .ok v) : ¬ c ∧ g = .ok v := by
  by_cases hc : c
  · rw [ite_eq_left hc] at h; cases h
  · exact ⟨hc, by rwa [ite_eq_right hc] at h⟩

/-- A loop in `Id` whose body continues, with an invariant on the elements still to come. -/
theorem forIn_id_inv_tail {α β : Type} (P : List α → β → Prop) {f : α → β → Id (ForInStep β)}
    (hf : ∀ a l b, P (a :: l) b → ∃ b', f a b = ForInStep.yield b' ∧ P l b') :
    ∀ (l : List α) (b : β), P l b → P [] (forIn l b f)
  | [], _, hb => hb
  | a :: l, b, hb => by
    rw [List.forIn_cons]
    obtain ⟨b', hab, hb'⟩ := hf a l b hb
    rw [hab]
    exact forIn_id_inv_tail P hf l b' hb'

/-- A loop in `Id` whose body appends `G a` to an array. -/
theorem forIn_id_append {α γ : Type} (G : α → List γ) {f : α → Array γ → Id (ForInStep (Array γ))} :
    ∀ (l : List α), (∀ a ∈ l, ∀ d, ∃ d', f a d = ForInStep.yield d' ∧ d'.toList = d.toList ++ G a) →
      ∀ d : Array γ, (forIn l d f).toList = d.toList ++ l.flatMap G
  | [], _, d => by simp only [List.forIn_nil, List.flatMap_nil, List.append_nil]; rfl
  | a :: l, hf, d => by
    rw [List.forIn_cons]
    obtain ⟨d', hab, hd'⟩ := hf a List.mem_cons_self d
    rw [hab]
    show (forIn l d' f).toList = _
    rw [forIn_id_append G l (fun a ha d => hf a (List.mem_cons_of_mem _ ha) d) d', hd']
    simp

/-! ## Port groups -/

/-- Every port `portCols` lists ends within its group's bits. -/
theorem portCols_le (gs : Array Group) :
    ∀ q ∈ (portCols gs).toList, ∃ g ∈ gs.toList, q.1 + q.2 ≤ g.col * WORD + g.bits.toList.sum := by
  unfold portCols
  simp only
  refine forIn_id_inv_array (fun out : Array (ℕ × ℕ) =>
    ∀ q ∈ out.toList, ∃ g ∈ gs.toList, q.1 + q.2 ≤ g.col * WORD + g.bits.toList.sum)
    (fun g hg out hout => ⟨_, rfl, ?_⟩) (by simp)
  rw [← Array.forIn_toList]
  refine (forIn_id_inv_tail (fun (L : List ℕ) (r : Array (ℕ × ℕ) × ℕ) =>
      (∀ q ∈ r.1.toList, ∃ g ∈ gs.toList, q.1 + q.2 ≤ g.col * WORD + g.bits.toList.sum) ∧
        r.2 + L.sum ≤ g.col * WORD + g.bits.toList.sum)
    (fun w L r hr => ⟨_, rfl, ?_, ?_⟩) g.bits.toList (out, g.col * WORD) ⟨hout, le_rfl⟩).1
  · intro q hq
    simp only [Array.toList_push, List.mem_append, List.mem_singleton] at hq
    rcases hq with hq | rfl
    · exact hr.1 q hq
    · exact ⟨g, hg, by have := hr.2; simp only [List.sum_cons] at this; simp only; omega⟩
  · have := hr.2; simp only [List.sum_cons] at this; simp only; omega

/-! ## What `Net.ofRows` built -/

/-- `Net.ofRows`' checks on its size and input groups: the constant is the last useful row, and each input group's ports
fit its words, inside the input rows. -/
theorem ofRows_groups {name : String} {useful constPos : ℕ} {ig og : Array Group} {ra rb : Array (Array ℕ)} {n : Net}
    (h : Net.ofRows name useful constPos ig og ra rb = .ok n) :
    constPos + 1 = useful ∧ ∀ g ∈ ig.toList, g.bits.foldl (· + ·) 0 ≤ g.words * WORD ∧ g.col + g.words ≤ n.inWords := by
  have hiw := (Net.ofRows_ok h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  unfold Net.ofRows at h
  simp only at h
  replace h := ite_throw_ok (ite_throw_ok h)
  obtain ⟨h4, h⟩ := ite_throw_ok' h
  obtain ⟨h5, -⟩ := ite_throw_ok' h
  simp only [Bool.or_eq_true, Array.any_eq_true, not_or, not_exists, decide_eq_true_eq] at h5
  refine ⟨by simpa using h4, fun g hg => ?_⟩
  obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 (Array.mem_def.2 hg)
  have := h5.1 i hi
  rw [hiw]
  omega

/-- **What `Net.ofRows` built** (`Built`: a text net, or a typed class's derived net): no lookup table, its constant the
last useful row, its slot `2^unitLog` rows, its input rows before the constant and its input groups inside them, and its
constant row `[constPos]` in the rows it stored as `A` and `B`. -/
theorem built_spec {net : Net} (h : Built net) :
    net.lookup = none ∧ net.useful = net.constPos + 1 ∧ net.unitLog = max 7 (log2ceil net.useful) ∧
      net.inWords * WORD ≤ net.constPos ∧
      (∀ g ∈ net.inGroups.toList, g.bits.foldl (· + ·) 0 ≤ g.words * WORD ∧ g.col + g.words ≤ net.inWords) ∧
      ∃ ra rb : Array (Array ℕ), ra.getD net.constPos #[] = #[net.constPos] ∧
        rb.getD net.constPos #[] = #[net.constPos] ∧
        Sparse.ofRows ra (2 ^ net.unitLog) = .ok net.a ∧ Sparse.ofRows rb (2 ^ net.unitLog) = .ok net.b := by
  obtain ⟨_, useful, constPos, ig, og, ra, rb, ho⟩ := h
  obtain ⟨hul, ha, hb, huse, hc, hig, -, hle, hca, hcb, -, -, hlk, -⟩ := Net.ofRows_ok ho
  obtain ⟨hcu, hgr⟩ := ofRows_groups ho
  exact ⟨hlk, by rw [huse, hc]; exact hcu.symm, by rw [hul, huse], by rw [hc]; exact hle, by rw [hig]; exact hgr,
    ra, rb, by rw [hc]; exact hca, by rw [hc]; exact hcb, ha, hb⟩

/-! ## Built nets -/

theorem le_two_pow_log2ceil (n : ℕ) : n ≤ 2 ^ log2ceil n := le_nextPow2 n

variable {net : Net}

/-- A built net's constant and input rows fit its `2^unitLog`-row slot, which is at least a word. -/
theorem built_fit (h : Built net) :
    net.constPos < 2 ^ net.unitLog ∧ net.inWords * WORD ≤ 2 ^ net.unitLog ∧ 7 ≤ net.unitLog := by
  obtain ⟨-, hu, hl, hin, -⟩ := built_spec h
  have h1 : 2 ^ log2ceil net.useful ≤ 2 ^ net.unitLog :=
    Nat.pow_le_pow_right (by norm_num) (by rw [hl]; exact le_max_right _ _)
  have := le_two_pow_log2ceil net.useful
  refine ⟨by omega, by omega, by rw [hl]; exact le_max_left _ _⟩

theorem foldl_add_eq_sum (l : List ℕ) (a : ℕ) : l.foldl (· + ·) a = a + l.sum := by
  induction l generalizing a with
  | nil => simp
  | cons x l ih => rw [List.foldl_cons, ih, List.sum_cons]; omega

/-- A built net's input ports lie inside its input rows. -/
theorem built_ports (h : Built net) :
    ∀ q ∈ (portCols net.inGroups).toList, q.1 + q.2 ≤ net.inWords * WORD := by
  obtain ⟨-, -, -, -, hg, -⟩ := built_spec h
  intro q hq
  obtain ⟨g, hgm, hle⟩ := portCols_le _ q hq
  obtain ⟨h1, h2⟩ := hg g hgm
  rw [← Array.foldl_toList, foldl_add_eq_sum] at h1
  have : (g.col + g.words) * WORD ≤ net.inWords * WORD := Nat.mul_le_mul_right _ h2
  rw [add_mul] at this
  omega

/-- **A built net's constant row reads itself**, and its input rows end before it. -/
theorem built_constRow (h : Built net) (hu : net.unitLog ≤ 32) : ConstRow net := by
  obtain ⟨hlk, -, -, hin, -, ra, rb, hra, hrb, ha, hb⟩ := built_spec h
  have hw : 2 ^ net.unitLog ≤ 2 ^ 32 := Nat.pow_le_pow_right (by norm_num) hu
  refine ⟨hin, fun j => ?_, fun j => ?_⟩
  · rw [sparseEntry_ofRows ha, ofRows_row ha hw, hra]
    by_cases hj : j = net.constPos
    · subst hj; simp
    · simp [hj, Ne.symm hj]
  · rw [sparseEntry_ofRows hb, ofRows_row hb hw, hrb, hlk]
    show _ + 0 = _
    by_cases hj : j = net.constPos
    · subst hj; simp
    · simp [hj, Ne.symm hj]

/-! ## Δ's first loop -/

/-- Every pair of an array past a prefix `L` has its row in `Q`. -/
def ExtL (L : List (ℕ × ℕ)) (Q : ℕ → Prop) (d : Array (ℕ × ℕ)) : Prop :=
  ∃ rest, d.toList = L ++ rest ∧ ∀ p ∈ rest, Q p.1

theorem ExtL.push2 {L : List (ℕ × ℕ)} {Q : ℕ → Prop} {d : Array (ℕ × ℕ)} (h : ExtL L Q d) {i a b : ℕ}
    (hi : Q i) : ExtL L Q ((d.push (i, a)).push (i, b)) := by
  obtain ⟨rest, hd, hr⟩ := h
  refine ⟨rest ++ [(i, a), (i, b)], by simp [hd], ?_⟩
  intro p hp
  simp only [List.mem_append, List.mem_cons, List.not_mem_nil, or_false] at hp
  rcases hp with hp | rfl | rfl
  · exact hr p hp
  · exact hi
  · exact hi

theorem run_bind_fst {P Q : Array (ℕ × ℕ) → Prop} {m : Id (Array (ℕ × ℕ))}
    {k : Array (ℕ × ℕ) → Id (Array (ℕ × ℕ) × Array (ℕ × ℕ))} (hm : P (Id.run m))
    (hk : ∀ d, P d → Q (Id.run (k d)).1) : Q (Id.run (m >>= k)).1 := hk _ hm

/-- **The rows Δ writes after its first loop** (`DeltaRow`'s other forms): a bit of a row's chain, a bit of a row's hm96
slot, a unit's leaf bit, a unit's leaf-cut bit, or a wire's destination bit. -/
def DeltaIn (c : Circuit) (x : ℕ) : Prop :=
  ShaRowBit c x ∨ HmRowBit c x ∨
  (∃ k < c.unitRange.count, ∃ j < (c.leavesIn.getD (k % c.upv) #[]).size, ∃ t < 16,
      x = c.unitRange.slot k + ((portCols #[c.unit.inGroups.getD 0 default]).map (·.1)).getD j 0 + t) ∨
  (∃ g < c.g, ∃ e ∈ c.leafCuts.toList, ∃ t < (portCols c.unit.inGroups)[e.2.1]!.2,
      x = c.slotCol c.unitNet g e.1 + (portCols c.unit.inGroups)[e.2.1]!.1 + t) ∨
  (∃ g < c.g, ∃ w ∈ c.wires.toList,
      ∃ t < ((portCols (c.nets.getD w.dst.1 default).2.inGroups).getD w.dst.2.2 default).2,
      x = c.slotCol w.dst.1 g w.dst.2.1 +
        ((portCols (c.nets.getD w.dst.1 default).2.inGroups).getD w.dst.2.2 default).1 + t)

/-- **Δ is its first loop's pairs, then input bits** (`HmRow.delta`): the first loop writes exactly `constPairs`, and
every later pair's row is one of `DeltaIn`'s forms. -/
theorem delta_split (c : Circuit) (pin : ℕ) (ht : c.typed = none) :
    ExtL (constPairs c pin) (DeltaIn c) (HmRow.delta c pin).1 := by
  unfold HmRow.delta
  simp only
  refine run_bind_fst (P := fun d => d.toList = constPairs c pin) ?_ ?_
  · rw [← Array.forIn_toList]
    refine (forIn_id_append (slotPairs c pin) _ (fun r _ d => ?_) #[]).trans ?_
    rotate_left
    · simp [constPairs]
    · split
      · rename_i i hk
        refine ⟨_, rfl, ?_⟩
        simp only [slotPairs, hk]
        rw [Std.Legacy.Range.forIn_eq_forIn_range', forIn_id_append (fun s =>
          if (r.slot s + (c.nets.getD i default).2.constPos != pin) = true then
            [(r.slot s + (c.nets.getD i default).2.constPos, r.slot s + (c.nets.getD i default).2.constPos),
              (r.slot s + (c.nets.getD i default).2.constPos, pin)] else []) _ (fun s _ d => ?_) d]
        · simp [List.range_eq_range']
        · split
          · exact ⟨_, rfl, by simp⟩
          · exact ⟨_, rfl, by simp⟩
      · rename_i hk
        refine ⟨_, rfl, ?_⟩
        unfold slotPairs
        split
        · rename_i i hi; exact absurd hi (hk i)
        · simp
  intro d hd
  have h0 : ExtL (constPairs c pin) (DeltaIn c) d := ⟨[], by simp [hd], by simp⟩
  -- the rows' constant bits
  refine run_bind_fst (forIn_id_inv_range _ (fun g _ hg d hd => ⟨_, rfl,
    forIn_id_inv_range _ (fun p _ hp d hd => ?_) hd⟩) h0) ?_
  · split
    · -- a segmented row: each chain start's midstate, then every constant or digest-copied message bit
      refine ⟨_, rfl, forIn_id_inv_range _ (fun k _ hk d hd => ?_) hd⟩
      split
      · refine ⟨_, rfl, forIn_id_inv_range _ (fun t _ ht d hd => ?_)
          (forIn_id_inv_range _ (fun t _ ht d hd => ⟨_, rfl, hd.push2 ?_⟩) hd)⟩
        · split
          · refine ⟨_, rfl, hd.push2 ?_⟩
            split
            · next hk' => exact Or.inl ⟨g, hg, p, hp, k, hk', 1, by omega, t, by omega, rfl⟩
            · exact Or.inr (Or.inl ⟨g, hg, p, hp, 1, by omega, t, by omega, rfl⟩)
          · refine ⟨_, rfl, hd.push2 ?_⟩
            split
            · next hk' => exact Or.inl ⟨g, hg, p, hp, k, hk', 1, by omega, t, by omega, rfl⟩
            · exact Or.inr (Or.inl ⟨g, hg, p, hp, 1, by omega, t, by omega, rfl⟩)
          · exact ⟨_, rfl, hd⟩
        · split
          · next hk' => exact Or.inl ⟨g, hg, p, hp, k, hk', 0, by omega, t, by omega, rfl⟩
          · exact Or.inr (Or.inl ⟨g, hg, p, hp, 0, by omega, t, by omega, rfl⟩)
      · refine ⟨_, rfl, forIn_id_inv_range _ (fun t _ ht d hd => ?_) hd⟩
        split
        · refine ⟨_, rfl, hd.push2 ?_⟩
          split
          · next hk' => exact Or.inl ⟨g, hg, p, hp, k, hk', 1, by omega, t, by omega, rfl⟩
          · exact Or.inr (Or.inl ⟨g, hg, p, hp, 1, by omega, t, by omega, rfl⟩)
        · refine ⟨_, rfl, hd.push2 ?_⟩
          split
          · next hk' => exact Or.inl ⟨g, hg, p, hp, k, hk', 1, by omega, t, by omega, rfl⟩
          · exact Or.inr (Or.inl ⟨g, hg, p, hp, 1, by omega, t, by omega, rfl⟩)
        · exact ⟨_, rfl, hd⟩
    · refine ⟨_, rfl, ?_⟩
      refine forIn_id_inv_range _ (fun k _ hk d hd => ⟨_, rfl, forIn_id_inv_range _ (fun t _ ht d hd => ?_) hd⟩)
        (forIn_id_inv_range _ (fun t _ ht d hd => ?_)
          (forIn_id_inv_range _ (fun t _ ht d hd => ⟨_, rfl, hd.push2 ?_⟩) hd))
      · split
        · exact ⟨_, rfl, hd.push2 (Or.inl ⟨g, hg, p, hp, k, hk, 1, by omega, t, by omega, rfl⟩)⟩
        · exact ⟨_, rfl, hd⟩
      · split
        · exact ⟨_, rfl, hd.push2 (Or.inr (Or.inl ⟨g, hg, p, hp, 1, by omega, t, by omega, rfl⟩))⟩
        · exact ⟨_, rfl, hd⟩
      · split
        · exact Or.inl ⟨g, hg, p, hp, 0, by assumption, 0, by omega, t, by omega, rfl⟩
        · exact Or.inr (Or.inl ⟨g, hg, p, hp, 0, by omega, t, by omega, rfl⟩)
  intro d hd
  -- the units' leaf bits
  refine run_bind_fst (forIn_id_inv_range _ (fun k _ hk d hd => ⟨_, rfl,
    forIn_id_inv_range _ (fun j _ hj d hd => ⟨_, rfl,
      forIn_id_inv_range _ (fun t _ ht d hd => ⟨_, rfl, hd.push2
        (Or.inr (Or.inr (Or.inl ⟨k, hk, j, hj, t, ht, rfl⟩)))⟩) hd⟩) hd⟩) hd) ?_
  intro d hd
  -- the units' leaf cuts
  refine run_bind_fst (forIn_id_inv_range _ (fun g _ hg d hd => ⟨_, rfl,
    forIn_id_inv_array _ (fun e he d hd => ⟨_, rfl,
      forIn_id_inv_range _ (fun t _ ht d hd => ⟨_, rfl, hd.push2
        (Or.inr (Or.inr (Or.inr (Or.inl ⟨g, hg, e, he, t, ht, rfl⟩))))⟩) hd⟩) hd⟩) hd) ?_
  intro d hd
  -- the wires
  refine run_bind_fst (forIn_id_inv_range _ (fun g _ hg d hd => ⟨_, rfl,
    forIn_id_inv_array _ (fun w hw d hd => ⟨_, rfl,
      forIn_id_inv_range _ (fun t _ ht d hd => ⟨_, rfl, hd.push2
        (Or.inr (Or.inr (Or.inr (Or.inr ⟨g, hg, w, hw, t, ht, rfl⟩))))⟩) hd⟩) hd⟩) hd) ?_
  intro d hd
  -- no typed tail: a flat statement's Δ ends with its wires
  simp only [ht]
  exact hd

theorem delta_snd (c : Circuit) (pin : ℕ) (ht : c.typed = none) : (HmRow.delta c pin).2 = (HmRow.delta c pin).1 := by
  unfold HmRow.delta
  simp only [ht]
  rfl

end FlockSoundness
