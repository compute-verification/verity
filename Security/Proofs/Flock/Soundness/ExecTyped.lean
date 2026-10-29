import Proofs.Flock.Soundness.ExecParse
import Properties.Flock.Circuit.Keys

/-!
# A typed template's Δ, and the rows it makes copies

`HmRow.delta` writes a flat statement's entries (`delta_split`'s forms), then, for a typed template
(`c.typed = some tb`), per template instance `g`:
1. each input bit `w`'s copy of its message bit: `(i, i)` and `(i, src)`, in `A` and `B`;
2. each part's bound input row `r`: `(i, i)` and `(i, col g x)` for its sources `x`, in `A` and `B`;
3. each root row's entries at its parts' columns: `A`'s in Δ_A, `B`'s in Δ_B;

with `i = col g r` and `col` the unit column's block position (`Typed.col`).

- `delta_typed`: Δ_A and Δ_B are the flat entries followed by `Typed.tail c tb true` and `Typed.tail c tb false`, the
  lists those loops write.
- `CopyRow S r src` and `ZeroRow S r`: the block rows at `r` read exactly `src`, or nothing, in `A` and `B`. A
  satisfying witness then carries at `r` what it carries at `src` (`CopyRow.block_eq`), or 0 (`ZeroRow.block_eq`).
-/

open Flock FlockLevel3

namespace FlockSoundness

/-! ## Copy and zero rows -/

section rows

variable {S : Model.Statement}

/-- **The block rows at `r` copy `src`**: in `A` and in `B` they read exactly `src`. -/
def CopyRow (S : Model.Statement) (r src : Fin (2 ^ S.kLog)) : Prop :=
  (∀ j, S.A₀ r j = if j = src then 1 else 0) ∧ ∀ j, S.B₀ r j = if j = src then 1 else 0

/-- **The block rows at `r` are zero**, in `A` and in `B`. -/
def ZeroRow (S : Model.Statement) (r : Fin (2 ^ S.kLog)) : Prop :=
  (∀ j, S.A₀ r j = 0) ∧ ∀ j, S.B₀ r j = 0

theorem mulVec_single (M : Matrix (Fin (2 ^ S.kLog)) (Fin (2 ^ S.kLog)) (ZMod 2)) {r src : Fin (2 ^ S.kLog)}
    (h : ∀ j, M r j = if j = src then 1 else 0) (v : Fin (2 ^ S.kLog) → ZMod 2) : M.mulVec v r = v src := by
  simp only [Matrix.mulVec, dotProduct, h, ite_mul, one_mul, zero_mul, Finset.sum_ite_eq', Finset.mem_univ, ite_true]

theorem mulVec_zero_row (M : Matrix (Fin (2 ^ S.kLog)) (Fin (2 ^ S.kLog)) (ZMod 2)) {r : Fin (2 ^ S.kLog)}
    (h : ∀ j, M r j = 0) (v : Fin (2 ^ S.kLog) → ZMod 2) : M.mulVec v r = 0 := by
  simp [Matrix.mulVec, dotProduct, h]

/-- **A copy row carries its source**, in every block of a satisfying witness. -/
theorem CopyRow.block_eq {r src : Fin (2 ^ S.kLog)} (h : CopyRow S r src) {z : Model.Witness S.m}
    (hz : S.Satisfies z) (o : Fin (2 ^ S.nbl)) : S.block z o r = S.block z o src := by
  have hr := hz.1 o r
  rw [mulVec_single S.A₀ h.1, mulVec_single S.B₀ h.2] at hr
  rw [← hr]
  generalize S.block z o src = x
  fin_cases x <;> rfl

/-- **A zero row carries 0**, in every block of a satisfying witness. -/
theorem ZeroRow.block_eq {r : Fin (2 ^ S.kLog)} (h : ZeroRow S r) {z : Model.Witness S.m} (hz : S.Satisfies z)
    (o : Fin (2 ^ S.nbl)) : S.block z o r = 0 := by
  have hr := hz.1 o r
  rw [mulVec_zero_row S.A₀ h.1, mulVec_zero_row S.B₀ h.2, zero_mul] at hr
  exact hr.symm

end rows

/-! ## The typed tail, as lists -/

namespace Typed

/-- Message bit `t` of byte `byte` of row `p`'s message in template instance `g` (`HmRow.delta`'s `msgCol`): in its
chain's `sha512x3` slot, or past the chain in its hm96 slot's `pad`. -/
def msgCol (c : Circuit) (g p byte t : ℕ) : ℕ :=
  let (_, where_) := HmRow.rowSlots c.ports
  let sin := portCols (c.nets.getD c.shaNet default).2.inGroups
  let hin := portCols (c.nets.getD c.hmNet default).2.inGroups
  let (k, o) := HmRow.byteAt c.ports[p]! byte
  if k < (where_[p]!).size then
    let (slot, sub) := (where_[p]!)[k]!
    c.slotCol c.shaNet g slot + sin[2 * sub + 1]!.1 + 64 * (o / 8) + 56 - 8 * (o % 8) + t
  else c.slotCol c.hmNet g p + hin[1]!.1 + 64 * (o / 8) + 56 - 8 * (o % 8) + t

/-- A row byte's offset in its block (`HmRow.byteAt`), segmented or not. -/
theorem byteAt_snd (p : Port) (byte : ℕ) : (HmRow.byteAt p byte).2 = byte % 128 := by
  unfold HmRow.byteAt
  split <;> rfl

/-- An unsegmented row's byte `byte` is in its compression `byte / 128`. -/
theorem byteAt_none {p : Port} (hs : p.segs = none) (byte : ℕ) : HmRow.byteAt p byte = (byte / 128, byte % 128) := by
  unfold HmRow.byteAt
  rw [hs]

/-- **A unit column's block position** in template instance `g` (`HmRow.delta`'s `col`): the root's own columns in its
slot, a part's in its layout's slot. -/
def col (c : Circuit) (tb : TypedBlock) (g u : ℕ) : ℕ :=
  if u < tb.own then c.slotCol c.unitNet g 0 + u
  else match tb.parts.toList.reverse.find? (·.2.2 ≤ u) with
    | some (n, q, b) => c.slotCol n g q + (u - b)
    | none => 0

/-- **The message bit input `w` of template instance `g` copies**: bit `w % 16` of row word `w / 16`. -/
def copySrc (c : Circuit) (g w : ℕ) : ℕ :=
  let (leaf, bit) := (w / 16, w % 16)
  let p := ((List.range (starts c).size).reverse.find? (fun q => (starts c)[q]! ≤ leaf)).getD 0
  msgCol c g p (2 * (leaf - (starts c)[p]!) + bit / 8) (bit % 8)

/-- Template instance `g`'s typed entries, in `A` (`side = true`) or `B`. -/
def tailVU (c : Circuit) (tb : TypedBlock) (side : Bool) (g : ℕ) : List (ℕ × ℕ) :=
  ((List.range tb.inCols.size).flatMap fun w =>
      [(col c tb g tb.inCols[w]!, col c tb g tb.inCols[w]!), (col c tb g tb.inCols[w]!, copySrc c g w)]) ++
    (tb.bind.toList.flatMap fun rs =>
      (col c tb g rs.1, col c tb g rs.1) :: rs.2.toList.map fun x => (col c tb g rs.1, col c tb g x)) ++
    tb.cross.toList.flatMap fun e => (if side then e.2.1 else e.2.2).toList.map fun x => (col c tb g e.1, col c tb g x)

/-- **The typed tail**: every template instance's typed entries, in `A` (`side = true`) or `B`. -/
def tail (c : Circuit) (tb : TypedBlock) (side : Bool) : List (ℕ × ℕ) := (List.range c.g).flatMap (tailVU c tb side)

end Typed

/-! ## `HmRow.delta` for a typed template -/

/-- A loop in `Id` over a pair state whose body appends `GA a` to one side and `GB a` to the other. -/
theorem forIn_id_append2 {α γ δ : Type} (GA : α → List γ) (GB : α → List δ)
    {f : α → Array γ × Array δ → Id (ForInStep (Array γ × Array δ))} :
    ∀ (l : List α), (∀ a ∈ l, ∀ s, ∃ s', f a s = ForInStep.yield s' ∧ s'.1.toList = s.1.toList ++ GA a ∧
      s'.2.toList = s.2.toList ++ GB a) →
      ∀ s, (forIn l s f).1.toList = s.1.toList ++ l.flatMap GA ∧ (forIn l s f).2.toList = s.2.toList ++ l.flatMap GB
  | [], _, s => by simp only [List.forIn_nil, List.flatMap_nil, List.append_nil]; exact ⟨rfl, rfl⟩
  | a :: l, hf, s => by
    rw [List.forIn_cons]
    obtain ⟨s', hab, h1, h2⟩ := hf a List.mem_cons_self s
    rw [hab]
    show (forIn l s' f).1.toList = _ ∧ (forIn l s' f).2.toList = _
    obtain ⟨k1, k2⟩ := forIn_id_append2 GA GB l (fun a ha s => hf a (List.mem_cons_of_mem _ ha) s) s'
    rw [k1, k2, h1, h2]
    simp

theorem forIn_range_append2 {γ δ : Type} (GA : ℕ → List γ) (GB : ℕ → List δ) (n : ℕ)
    {f : ℕ → Array γ × Array δ → Id (ForInStep (Array γ × Array δ))}
    (hf : ∀ a < n, ∀ s, ∃ s', f a s = ForInStep.yield s' ∧ s'.1.toList = s.1.toList ++ GA a ∧
      s'.2.toList = s.2.toList ++ GB a) (s : Array γ × Array δ) :
    (forIn [:n] s f).1.toList = s.1.toList ++ (List.range n).flatMap GA ∧
      (forIn [:n] s f).2.toList = s.2.toList ++ (List.range n).flatMap GB := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  have := forIn_id_append2 GA GB (List.range' 0 [:n].size) (fun a ha s => hf a (by
    simp only [Std.Legacy.Range.size, List.mem_range'_1] at ha; omega) s) s
  simpa [Std.Legacy.Range.size, List.range_eq_range'] using this

theorem forIn_array_append2 {α γ δ : Type} (GA : α → List γ) (GB : α → List δ) (xs : Array α)
    {f : α → Array γ × Array δ → Id (ForInStep (Array γ × Array δ))}
    (hf : ∀ a ∈ xs.toList, ∀ s, ∃ s', f a s = ForInStep.yield s' ∧ s'.1.toList = s.1.toList ++ GA a ∧
      s'.2.toList = s.2.toList ++ GB a) (s : Array γ × Array δ) :
    (forIn xs s f).1.toList = s.1.toList ++ xs.toList.flatMap GA ∧
      (forIn xs s f).2.toList = s.2.toList ++ xs.toList.flatMap GB := by
  rw [← Array.forIn_toList]
  exact forIn_id_append2 GA GB xs.toList hf s

theorem run_bind_pair {P : Array (ℕ × ℕ) → Prop} {Q : Array (ℕ × ℕ) × Array (ℕ × ℕ) → Prop} {m : Id (Array (ℕ × ℕ))}
    {k : Array (ℕ × ℕ) → Id (Array (ℕ × ℕ) × Array (ℕ × ℕ))} (hm : P (Id.run m))
    (hk : ∀ d, P d → Q (Id.run (k d))) : Q (Id.run (m >>= k)) := hk _ hm

/-- Δ's pair is the flat entries `d` (`delta_split`'s forms), then the typed tail on each side. -/
def TailQ (c : Circuit) (pin : ℕ) (tb : TypedBlock) (r : Array (ℕ × ℕ) × Array (ℕ × ℕ)) : Prop :=
  ∃ d : Array (ℕ × ℕ), ExtL (constPairs c pin) (DeltaIn c) d ∧
    r.1.toList = d.toList ++ Typed.tail c tb true ∧ r.2.toList = d.toList ++ Typed.tail c tb false

/-- **A typed template's Δ**: the flat statement's entries, then the typed tail, in Δ_A and in Δ_B. -/
theorem delta_typed (c : Circuit) (pin : ℕ) {tb : TypedBlock} (ht : c.typed = some tb) :
    TailQ c pin tb (HmRow.delta c pin) := by
  unfold HmRow.delta
  simp only
  refine run_bind_pair (Q := TailQ c pin tb) (P := fun d => d.toList = constPairs c pin) ?_ ?_
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
  refine run_bind_pair (Q := TailQ c pin tb) (forIn_id_inv_range _ (fun g _ hg d hd => ⟨_, rfl,
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
  refine run_bind_pair (Q := TailQ c pin tb) (forIn_id_inv_range _ (fun k _ hk d hd => ⟨_, rfl,
    forIn_id_inv_range _ (fun j _ hj d hd => ⟨_, rfl,
      forIn_id_inv_range _ (fun t _ ht d hd => ⟨_, rfl, hd.push2
        (Or.inr (Or.inr (Or.inl ⟨k, hk, j, hj, t, ht, rfl⟩)))⟩) hd⟩) hd⟩) hd) ?_
  intro d hd
  -- the units' leaf cuts
  refine run_bind_pair (Q := TailQ c pin tb) (forIn_id_inv_range _ (fun g _ hg d hd => ⟨_, rfl,
    forIn_id_inv_array _ (fun e he d hd => ⟨_, rfl,
      forIn_id_inv_range _ (fun t _ ht d hd => ⟨_, rfl, hd.push2
        (Or.inr (Or.inr (Or.inr (Or.inl ⟨g, hg, e, he, t, ht, rfl⟩))))⟩) hd⟩) hd⟩) hd) ?_
  intro d hd
  -- the wires
  refine run_bind_pair (Q := TailQ c pin tb) (forIn_id_inv_range _ (fun g _ hg d hd => ⟨_, rfl,
    forIn_id_inv_array _ (fun w hw d hd => ⟨_, rfl,
      forIn_id_inv_range _ (fun t _ ht d hd => ⟨_, rfl, hd.push2
        (Or.inr (Or.inr (Or.inr (Or.inr ⟨g, hg, w, hw, t, ht, rfl⟩))))⟩) hd⟩) hd⟩) hd) ?_
  intro d hd
  -- the typed tail
  simp only [ht]
  refine ⟨d, hd, ?_⟩
  have H := fun f hf => forIn_range_append2 (f := f) (Typed.tailVU c tb true) (Typed.tailVU c tb false) c.g hf (d, d)
  refine ⟨(H _ ?hf).1, (H _ ?hf).2⟩
  intro g _ s
  refine ⟨_, rfl, ?_⟩
  have HW := fun f hf => forIn_range_append2 (f := f)
    (fun w => [(Typed.col c tb g tb.inCols[w]!, Typed.col c tb g tb.inCols[w]!),
      (Typed.col c tb g tb.inCols[w]!, Typed.copySrc c g w)])
    (fun w => [(Typed.col c tb g tb.inCols[w]!, Typed.col c tb g tb.inCols[w]!),
      (Typed.col c tb g tb.inCols[w]!, Typed.copySrc c g w)]) tb.inCols.size hf
  have HB := fun f hf => forIn_array_append2 (f := f)
    (fun rs : ℕ × Array ℕ => (Typed.col c tb g rs.1, Typed.col c tb g rs.1) ::
      rs.2.toList.map fun x => (Typed.col c tb g rs.1, Typed.col c tb g x))
    (fun rs : ℕ × Array ℕ => (Typed.col c tb g rs.1, Typed.col c tb g rs.1) ::
      rs.2.toList.map fun x => (Typed.col c tb g rs.1, Typed.col c tb g x)) tb.bind hf
  have HC := fun f hf => forIn_array_append2 (f := f)
    (fun e : ℕ × Array ℕ × Array ℕ => e.2.1.toList.map fun x => (Typed.col c tb g e.1, Typed.col c tb g x))
    (fun e : ℕ × Array ℕ × Array ℕ => e.2.2.toList.map fun x => (Typed.col c tb g e.1, Typed.col c tb g x)) tb.cross hf
  constructor
  all_goals first
    | refine (HC _ ?_ _).1.trans ?_
    | refine (HC _ ?_ _).2.trans ?_
  all_goals first
    | (intro e _ t
       exact ⟨_, rfl, by (simp only [Array.toList_append, Array.toList_map]; rfl),
         by (simp only [Array.toList_append, Array.toList_map]; rfl)⟩)
    | skip
  all_goals first
    | (dsimp only; rw [(HB _ ?_ _).1, (HW _ ?_ _).1])
    | (dsimp only; rw [(HB _ ?_ _).2, (HW _ ?_ _).2])
  all_goals first
    | (simp [Typed.tailVU]; done)
    | (intro rs _ t
       exact ⟨_, rfl,
         by (simp only [Array.toList_append, Array.toList_push, Array.toList_map, List.append_assoc]; rfl),
         by (simp only [Array.toList_append, Array.toList_push, Array.toList_map, List.append_assoc]; rfl)⟩)

end FlockSoundness
