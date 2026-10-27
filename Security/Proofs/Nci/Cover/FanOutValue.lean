import Proofs.Nci.Cover.FanOut
import Proofs.Nci.Cover.FanOutChain
import Definitions.Nci.Guarantees.CoverFanOutValue

/-!
# The fan-out read tier from wiring

`FanOutReadIntensityStructural` is `FanOutReadIntensity` with the values priced by the circuit. A token is *heavy* in a
block if the block holds one of its roots, or computes one of its products whole while handed fewer than `f` gates of
the activation's key path. A heavy token reads `d − E − (g − 1)` of its values (`rootCost`, or `cut_legs` and
`heavy_cost`): by `PerValueCut`, all but `g − 1` of the root's legs have an exit in the block, `walk` takes each to a
value the block reads or a lookup it holds, and the legs' key paths are disjoint. A light token's whole products are
charged to spent values, at most `aOut` each (`charge_wide`), and each is handed `f` gates of its own key path, which
no other spent value's meets (`pathGates_disjoint`, from `spent_eq_of_follows`). Tokens' values are disjoint, so
`f · #light + #heavy reads` is at most the block's inputs among the values' gates (`fanOut_value_split`), and
`ConditionalUpdateHardness` pays the parameters outside them (`valueReadHardness_of_structure`). The arithmetic is
`FanOutReadIntensity`'s.

`Keyed.keyDisjoint` proves `KeyDisjoint` from `Keyed`'s (I) and (F).
-/

namespace Nci.Cover

namespace Follows

variable {Val : Type*} {key : Val → Option Val}

/-- **The first value of a stop set**: if the key path from `v` through `T` ends at `u`, a value of `T` that the key
path from `v` through `S` reaches lies on the key path from `u` through `S`. -/
theorem of_first {S T : Finset Val} {v u w : Val} (hu : Follows key T v u) (hw : Follows key S v w)
    (hwT : w ∈ T) : Follows key S u w := by
  induction hu with
  | refl => exact hw
  | step hv hk _ ih =>
    cases hw with
    | refl => exact absurd hwT hv
    | step _ hk' hw' =>
      rw [Option.some.inj (hk.symm.trans hk')] at ih
      exact ih hw'

end Follows

namespace Keyed

open SecurityProofs

variable {C : Circuit} {J L Tok Val : Type} [DecidableEq Val] {M : MulBlocks C J L} {d aOut c V f : ℕ}

/-- **Distinct activations have disjoint key paths** ((I) and (F)): two key paths into one value are one path, so one
activation would be a key ancestor of the other, and no value is keyed to an activation. -/
theorem keyDisjoint (K : Keyed M Tok Val d aOut c V f) : K.KeyDisjoint := by
  intro p hp p' hp' hne _ u hu hu'
  have hv := K.act_mem p hp
  have hv' := K.act_mem p' hp'
  have hstart : ∀ {a q}, q ∈ M.P → a ∈ K.vals → a ≠ K.act q → ¬ Follows K.key K.stops a (K.act q) := by
    intro a q hq ha haq h
    obtain ⟨w, hw, -, hwk⟩ := Follows.last_step h haq
    exact (K.key_fresh w (K.follows_mem subset_rfl ha hw) _ hwk).1 q hq rfl
  rcases K.follows_suffix subset_rfl subset_rfl hv hv' hu' hu with h | h
  · exact hstart hp hv' (Ne.symm hne) h
  · exact hstart hp' hv hne h

/-- Following keys stays in a token. -/
theorem follows_tok (K : Keyed M Tok Val d aOut c V f) {T : Finset Val} (hT : K.stops ⊆ T) {a w : Val}
    (ha : a ∈ K.vals) (h : Follows K.key T a w) : K.tok w = K.tok a := by
  revert ha
  induction h with
  | refl => exact fun _ => rfl
  | @step v κ u hv hk _ ih =>
    intro ha
    obtain ⟨κ', hκ', hk', htok, -⟩ := K.keyed v ha fun hs => hv (hT hs)
    rw [Option.some.inj (hk.symm.trans hk')] at ih
    exact (ih hκ').trans htok

theorem mem_pathGates {K : Keyed M Tok Val d aOut c V f} {v : Val} {z : Fin C.N} :
    z ∈ K.pathGates v ↔ ∃ w ∈ K.vals, Follows K.key K.stops v w ∧ z ∈ K.gates w := by
  simp only [pathGates, Finset.mem_biUnion, Finset.mem_filter, and_assoc]

/-- **Spent values' key paths are disjoint** ((I) and (F), through `spent_eq_of_follows`). -/
theorem pathGates_disjoint [DecidableEq J] (K : Keyed M Tok Val d aOut c V f) {I : Finset (Fin C.N)} {u u' : Val}
    (hu : K.Spent I u) (hu' : K.Spent I u') (hne : u ≠ u') : Disjoint (K.pathGates u) (K.pathGates u') := by
  rw [Finset.disjoint_left]
  intro z hz hz'
  obtain ⟨w, hw, hfw, hzw⟩ := mem_pathGates.1 hz
  obtain ⟨w', hw', hfw', hzw'⟩ := mem_pathGates.1 hz'
  obtain rfl : w = w' := by
    by_contra hww
    have hd : Disjoint (K.gates w) (K.gates w') := K.gates_disjoint hw hw' hww
    exact Finset.disjoint_left.1 hd hzw hzw'
  have huv : u ∈ K.vals := (mem_readSet.1 hu.1).1
  have hu'v : u' ∈ K.vals := (mem_readSet.1 hu'.1).1
  rcases K.follows_suffix subset_rfl subset_rfl huv hu'v hfw' hfw with h | h
  · exact hne (K.spent_eq_of_follows hu' hu h).symm
  · exact hne (K.spent_eq_of_follows hu hu' h)

open Classical in
/-- The gates of a token's values. -/
noncomputable def tokGates (K : Keyed M Tok Val d aOut c V f) (i : Tok) : Finset (Fin C.N) :=
  (K.vals.filter fun v => K.tok v = i).biUnion K.gates

theorem mem_tokGates {K : Keyed M Tok Val d aOut c V f} {i : Tok} {z : Fin C.N} :
    z ∈ K.tokGates i ↔ ∃ v ∈ K.vals, K.tok v = i ∧ z ∈ K.gates v := by
  simp only [tokGates, Finset.mem_biUnion, Finset.mem_filter, and_assoc]

theorem pathGates_subset_tokGates (K : Keyed M Tok Val d aOut c V f) {u : Val} (hu : u ∈ K.vals) :
    K.pathGates u ⊆ K.tokGates (K.tok u) := by
  intro z hz
  obtain ⟨w, hw, hfw, hzw⟩ := mem_pathGates.1 hz
  exact mem_tokGates.2 ⟨w, hw, K.follows_tok subset_rfl hu hfw, hzw⟩

theorem tokGates_disjoint (K : Keyed M Tok Val d aOut c V f) {i j : Tok} (hij : i ≠ j) :
    Disjoint (K.tokGates i) (K.tokGates j) := by
  rw [Finset.disjoint_left]
  intro z hz hz'
  obtain ⟨v, hv, hvi, hzv⟩ := mem_tokGates.1 hz
  obtain ⟨v', hv', hvj, hzv'⟩ := mem_tokGates.1 hz'
  have hvv : v ≠ v' := fun h => hij (hvi.symm.trans (h ▸ hvj))
  have hd : Disjoint (K.gates v) (K.gates v') := K.gates_disjoint hv hv' hvv
  exact Finset.disjoint_left.1 hd hzv hzv'

theorem tokGates_subset (K : Keyed M Tok Val d aOut c V f) (i : Tok) : K.tokGates i ⊆ K.vals.biUnion K.gates := by
  intro z hz
  obtain ⟨v, hv, -, hzv⟩ := mem_tokGates.1 hz
  exact Finset.mem_biUnion.2 ⟨v, hv, hzv⟩

/-- **A token's reads**: a block reads at most its inputs among a token's gates' worth of the token's values. -/
theorem read_cost (K : Keyed M Tok Val d aOut c V f) {I : Finset (Fin C.N)} {i : Tok} {Y : Finset Val}
    (hY : ∀ u ∈ Y, u ∈ K.readSet I ∧ K.tok u = i) : Y.card ≤ (C.inputs I ∩ K.tokGates i).card := by
  have hne : ∀ u ∈ Y, (K.gates u ∩ C.inputs I).Nonempty := fun u hu => (mem_readSet.1 (hY u hu).1).2
  have hdisj : (Y : Set Val).PairwiseDisjoint fun u => K.gates u ∩ C.inputs I := by
    intro u hu u' hu' huu
    have hd : Disjoint (K.gates u) (K.gates u') :=
      K.gates_disjoint (mem_readSet.1 (hY u hu).1).1 (mem_readSet.1 (hY u' hu').1).1 huu
    exact hd.mono Finset.inter_subset_left Finset.inter_subset_left
  refine (Finset.card_le_card_biUnion hdisj hne).trans (Finset.card_le_card ?_)
  intro z hz
  obtain ⟨u, hu, hz⟩ := Finset.mem_biUnion.1 hz
  obtain ⟨hzg, hzI⟩ := Finset.mem_inter.1 hz
  exact Finset.mem_inter.2 ⟨hzI, mem_tokGates.2 ⟨u, (mem_readSet.1 (hY u hu).1).1, (hY u hu).2, hzg⟩⟩

/-- **A light token's spent values cost `f` gates each**: each is handed `f` gates of its own key path, and spent
values' key paths are disjoint. -/
theorem light_cost [DecidableEq J] (K : Keyed M Tok Val d aOut c V f) {I : Finset (Fin C.N)} {i : Tok}
    {X : Finset Val} (hX : ∀ u ∈ X, K.Spent I u ∧ K.tok u = i ∧ f ≤ (C.inputs I ∩ K.pathGates u).card) :
    f * X.card ≤ (C.inputs I ∩ K.tokGates i).card := by
  have hdisj : (X : Set Val).PairwiseDisjoint fun u => C.inputs I ∩ K.pathGates u := by
    intro u hu u' hu' huu
    exact (K.pathGates_disjoint (hX u hu).1 (hX u' hu').1 huu).mono Finset.inter_subset_right
      Finset.inter_subset_right
  calc f * X.card = ∑ _u ∈ X, f := by rw [Finset.sum_const, smul_eq_mul, mul_comm]
    _ ≤ ∑ u ∈ X, (C.inputs I ∩ K.pathGates u).card := Finset.sum_le_sum fun u hu => (hX u hu).2.2
    _ = (X.biUnion fun u => C.inputs I ∩ K.pathGates u).card := (Finset.card_biUnion hdisj).symm
    _ ≤ (C.inputs I ∩ K.tokGates i).card := Finset.card_le_card fun z hz => by
        obtain ⟨u, hu, hz⟩ := Finset.mem_biUnion.1 hz
        obtain ⟨hzI, hzp⟩ := Finset.mem_inter.1 hz
        have huv : u ∈ K.vals := (mem_readSet.1 (hX u hu).1.1).1
        have hzt := K.pathGates_subset_tokGates huv hzp
        rw [(hX u hu).2.1] at hzt
        exact Finset.mem_inter.2 ⟨hzI, hzt⟩

/-- **The cut**: if a block computes a counted product whole and is handed fewer than `f` gates of its activation's
key path, fewer than `g` of the legs of the root that path ends at have no exit in the block. A path from such a leg's
exit to the product's output gate enters the block at an input (`entry`), which is a gate of the key path, since the
product's gates are computed. -/
theorem cut_legs [DecidableEq J] (K : Keyed M Tok Val d aOut c V f) {g : ℕ} {I : Finset (Fin C.N)} {p : J × L}
    (hp : p ∈ M.whole I) {r : Val}
    (hcut : ∀ G ⊆ K.legs r, g ≤ G.card → ∀ Z ⊆ K.pathGates (K.act p),
      (∀ q ∈ G, ∀ h ∈ C.exits (K.gates q), ∀ π, C.IsPath π h (M.o p) →
        (∀ x ∈ π.tail, x ∈ K.pathGates (K.act p) ∪ M.m p) → ∃ z ∈ Z, z ∈ π) → f ≤ Z.card)
    (hZ : (C.inputs I ∩ K.pathGates (K.act p)).card < f) :
    ((K.legs r).filter fun q => ∀ h ∈ C.exits (K.gates q), h ∉ I).card < g := by
  obtain ⟨hpP, hpI⟩ := Finset.mem_filter.1 hp
  have hoI : M.o p ∈ I := computed_subset C I (hpI (o_mem M (M.P_subset hpP)))
  by_contra hlt
  refine absurd (hcut _ (Finset.filter_subset _ _) (not_lt.1 hlt) _ Finset.inter_subset_right ?_) (not_le.2 hZ)
  intro q hq h hh π hπ hπt
  have hhI : h ∉ I := (Finset.mem_filter.1 hq).2 h hh
  obtain ⟨x, hxπ, hxI⟩ := SecurityProofs.entry C I π h (M.o p) hπ hoI fun hc => hhI (computed_subset C I hc)
  have hxh : x ≠ h := fun hxh => hhI (hxh ▸ (mem_inputs.1 hxI).1)
  have hxt : x ∈ π.tail := by
    obtain ⟨hhd, -, -⟩ := hπ
    cases π with
    | nil => simp at hhd
    | cons a t =>
      simp only [List.head?_cons, Option.some.injEq] at hhd
      subst hhd
      exact (List.mem_cons.1 hxπ).resolve_left hxh
  rcases Finset.mem_union.1 (hπt x hxt) with hxp | hxm
  · exact ⟨x, Finset.mem_inter.2 ⟨hxI, hxp⟩, hxπ⟩
  · exact absurd hxI (mem_computed.1 (hpI hxm)).2

/-- **A heavy token without a held root costs `d − E − (g − 1)` reads**: if all but `g − 1` of a root's legs have an
exit in an inference block that holds no root of its token, `walk` takes each such leg to a value the block reads or a
lookup it holds, distinct ends for distinct legs (`legs_disjoint`), and the block holds at most `E` lookups of a token
(`lookupCost`). -/
theorem heavy_cost [DecidableEq L] (K : Keyed M Tok Val d aOut c V f) {s k E g : ℕ} {bcol : ℝ}
    (hCI : Assumptions.ColumnsIncompressible K s bcol) (hk : (k : ℝ) < bcol * V * (E + 1))
    {I : Finset (Fin C.N)} (hI : C.InferenceBlock s k I) {r : Val} (hr : r ∈ K.roots)
    (hnoroot : ∀ r' ∈ K.roots ∩ K.heldSet I, K.tok r' ≠ K.tok r)
    (hG : ((K.legs r).filter fun q => ∀ h ∈ C.exits (K.gates q), h ∉ I).card < g) :
    ∃ Y : Finset Val, (∀ u ∈ Y, u ∈ K.readSet I ∧ K.tok u = K.tok r) ∧ d - E - (g - 1) ≤ Y.card := by
  classical
  set G := (K.legs r).filter fun q => ∀ h ∈ C.exits (K.gates q), h ∉ I with hGdef
  set Lg := K.legs r \ G
  have hleg : ∀ q, ∃ u, q ∈ Lg → K.tok u = K.tok r ∧ Follows K.key (K.stops ∪ K.readSet I) q u ∧
      (u ∈ K.readSet I ∨ u ∈ K.lookups ∩ K.heldSet I) := by
    intro q
    by_cases hq : q ∈ Lg
    swap
    · exact ⟨q, fun h => absurd h hq⟩
    obtain ⟨hqr, hqG⟩ := Finset.mem_sdiff.1 hq
    have hex : ∃ h ∈ C.exits (K.gates q), h ∈ I := by
      by_contra hno
      exact hqG (Finset.mem_filter.2 ⟨hqr, fun h hh hhI => hno ⟨h, hh, hhI⟩⟩)
    obtain ⟨h, hh, hhI⟩ := hex
    obtain ⟨hqv, hqt⟩ := K.legs_mem r hr q hqr
    obtain ⟨u, -, hut, hfu, hend⟩ := walk C M d aOut c V f K I q hqv h hh hhI
    refine ⟨u, fun _ => ⟨hut.trans hqt, hfu, ?_⟩⟩
    rcases hend with hend | ⟨hus, -⟩
    · exact Or.inl hend
    · obtain ⟨hus, huh⟩ := Finset.mem_inter.1 hus
      rcases Finset.mem_union.1 hus with huR | huL
      · exact absurd (hut.trans hqt) (hnoroot u (Finset.mem_inter.2 ⟨huR, huh⟩))
      · exact Or.inr (Finset.mem_inter.2 ⟨huL, huh⟩)
  choose φ hφ using hleg
  have hinj : Set.InjOn φ Lg := by
    intro q hq q' hq' hqq
    by_contra hne
    have h1 : Follows K.key (K.roots ∪ K.lookups) q (φ q') :=
      hqq ▸ Follows.mono Finset.subset_union_left (hφ q hq).2.1
    have h2 : Follows K.key (K.roots ∪ K.lookups) q' (φ q') :=
      Follows.mono Finset.subset_union_left (hφ q' hq').2.1
    exact K.legs_disjoint r hr q (Finset.mem_sdiff.1 hq).1 q' (Finset.mem_sdiff.1 hq').1 hne (φ q') h1 h2
  set Yr := (Lg.image φ).filter (· ∈ K.readSet I)
  set Yl := (Lg.image φ).filter (· ∉ K.readSet I)
  have hYl : Yl.card ≤ E := by
    refine lookupCost C M d aOut c V f K s k E bcol hCI hk I hI _ (fun u hu => ?_) fun u hu u' hu' => ?_
    · obtain ⟨hu, hur⟩ := Finset.mem_filter.1 hu
      obtain ⟨q, hq, rfl⟩ := Finset.mem_image.1 hu
      exact (hφ q hq).2.2.resolve_left hur
    · obtain ⟨q, hq, rfl⟩ := Finset.mem_image.1 (Finset.mem_filter.1 hu).1
      obtain ⟨q', hq', rfl⟩ := Finset.mem_image.1 (Finset.mem_filter.1 hu').1
      exact (hφ q hq).1.trans (hφ q' hq').1.symm
  have hsum : Yr.card + Yl.card = Lg.card := by
    rw [Finset.card_filter_add_card_filter_not, Finset.card_image_of_injOn hinj]
  have hLg : Lg.card + G.card = (K.legs r).card := Finset.card_sdiff_add_card_eq_card (Finset.filter_subset _ _)
  have hd := K.legs_card r hr
  refine ⟨Yr, fun u hu => ?_, by omega⟩
  obtain ⟨hu, hur⟩ := Finset.mem_filter.1 hu
  obtain ⟨q, hq, rfl⟩ := Finset.mem_image.1 hu
  exact ⟨hur, (hφ q hq).1⟩

/-- **A light token's charge** (`charge`, with widths): if a block holds no root of token `i` and is handed `f` gates
of each of its whole products' activations' key paths, those products number at most `aOut` per value it spends, and
each such value is handed `f` gates of its own key path. The value is the first the block reads on the activation's
key path, so the key path's inputs all lie on the value's (`Follows.of_first`). -/
theorem charge_wide [DecidableEq J] (K : Keyed M Tok Val d aOut c V f) {I : Finset (Fin C.N)} {i : Tok}
    (hnoroot : ∀ r ∈ K.roots ∩ K.heldSet I, K.tok r ≠ i) (B : Finset (J × L)) (hB : B ⊆ M.whole I)
    (htokB : ∀ p ∈ B, K.tok (K.act p) = i) (hwide : ∀ p ∈ B, f ≤ (C.inputs I ∩ K.pathGates (K.act p)).card) :
    ∃ X : Finset Val, (∀ u ∈ X, K.Spent I u ∧ K.tok u = i ∧ f ≤ (C.inputs I ∩ K.pathGates u).card) ∧
      B.card ≤ aOut * X.card := by
  classical
  have hpick : ∀ p ∈ B, ∃ u, (K.Spent I u ∧ K.tok u = i ∧ f ≤ (C.inputs I ∩ K.pathGates u).card) ∧
      Follows K.key (K.roots ∪ K.lookups) (K.act p) u := by
    intro p hp
    have hpw := hB hp
    obtain ⟨hpP, hpI⟩ := Finset.mem_filter.1 hpw
    obtain ⟨h, hh, x, hx, hhx, -⟩ := K.act_link p hpP
    have hhI : h ∈ I := pred_subset_of_mem_computed (hpI hx) hhx
    obtain ⟨u, -, hut, hfu, hend⟩ := walk C M d aOut c V f K I (K.act p) (K.act_mem p hpP) h hh hhI
    have hfu' : Follows K.key (K.roots ∪ K.lookups) (K.act p) u := hfu.mono Finset.subset_union_left
    rcases hend with hur | ⟨hus, -⟩
    · refine ⟨u, ⟨⟨hur, K.act p, Or.inl ⟨p, hpw, rfl⟩, hfu⟩, hut.trans (htokB p hp), ?_⟩, hfu'⟩
      refine (hwide p hp).trans (Finset.card_le_card fun z hz => ?_)
      obtain ⟨hzI, hzp⟩ := Finset.mem_inter.1 hz
      obtain ⟨w, hw, hfw, hzw⟩ := mem_pathGates.1 hzp
      have hwr : w ∈ K.readSet I := mem_readSet.2 ⟨hw, z, Finset.mem_inter.2 ⟨hzw, hzI⟩⟩
      exact Finset.mem_inter.2 ⟨hzI, mem_pathGates.2 ⟨w, hw,
        Follows.of_first hfu hfw (Finset.mem_union_right _ hwr), hzw⟩⟩
    · exfalso
      obtain ⟨hus, huh⟩ := Finset.mem_inter.1 hus
      rcases Finset.mem_union.1 hus with hur | hul
      · exact hnoroot u (Finset.mem_inter.2 ⟨hur, huh⟩) (hut.trans (htokB p hp))
      · exact K.reach_lookups p hpP u hul hfu'
  rcases B.eq_empty_or_nonempty with rfl | ⟨p₀, -⟩
  · exact ⟨∅, fun u hu => absurd hu (Finset.notMem_empty u), by simp⟩
  have : Nonempty Val := ⟨K.act p₀⟩
  choose! φ hφ using hpick
  refine ⟨B.image φ, fun u hu => ?_, ?_⟩
  · obtain ⟨p, hp, rfl⟩ := Finset.mem_image.1 hu
    exact (hφ p hp).1
  · refine Finset.card_le_mul_card_image B aOut fun u _ => K.reach u _ (fun p hp => ?_) fun p hp => ?_
    · exact (Finset.mem_filter.1 (hB (Finset.mem_filter.1 hp).1)).1
    · obtain ⟨hpB, rfl⟩ := Finset.mem_filter.1 hp
      exact (hφ p hpB).2

end Keyed

namespace SecurityProofs

/-- **The split by token, from wiring**: an inference block spends values `X`, of tokens it is light on, and reads `y`
values of `h` heavy tokens, with `f · #X + y` at most its inputs among the values' gates, `(d − E − (g − 1)) h ≤ y` and
`#whole ≤ aOut · #X + c · h · #(updated parameters of whole)`. -/
theorem fanOut_value_split {C : Circuit} {J L Tok Val : Type} [DecidableEq J] [DecidableEq L] [DecidableEq Val]
    {M : MulBlocks C J L} {d aOut c V f : ℕ} (K : Keyed M Tok Val d aOut c V f) {s k E g : ℕ} {bcol : ℝ}
    (hCI : Assumptions.ColumnsIncompressible K s bcol) (hk : (k : ℝ) < bcol * V * (E + 1)) (hcut : K.PerValueCut g)
    {I : Finset (Fin C.N)} (hI : C.InferenceBlock s k I) :
    ∃ X : Finset Val, (∀ u ∈ X, K.Spent I u) ∧ ∃ y h : ℕ,
      f * X.card + y ≤ (C.inputs I ∩ K.vals.biUnion K.gates).card ∧ (d - E - (g - 1)) * h ≤ y ∧
      (M.whole I).card ≤ aOut * X.card + c * h * ((M.whole I).image Prod.snd).card := by
  classical
  set W := M.whole I
  set Lw := W.image Prod.snd
  set Th : Finset Tok := (K.roots ∩ K.heldSet I).image K.tok ∪
    (W.filter fun p => (C.inputs I ∩ K.pathGates (K.act p)).card < f).image fun p => K.tok (K.act p)
  have hX : ∀ i : Tok, ∃ Xi Yi : Finset Val, (∀ u ∈ Xi, K.Spent I u ∧ K.tok u = i) ∧
      f * Xi.card + Yi.card ≤ (C.inputs I ∩ K.tokGates i).card ∧ (i ∈ Th → d - E - (g - 1) ≤ Yi.card) ∧
      (i ∉ Th → (W.filter fun p => K.tok (K.act p) = i).card ≤ aOut * Xi.card) := by
    intro i
    by_cases hi : i ∈ Th
    · suffices H : ∃ Yi : Finset Val, (∀ u ∈ Yi, u ∈ K.readSet I ∧ K.tok u = i) ∧ d - E - (g - 1) ≤ Yi.card by
        obtain ⟨Yi, hYr, hYc⟩ := H
        exact ⟨∅, Yi, fun u hu => absurd hu (Finset.notMem_empty u), by simpa using K.read_cost hYr,
          fun _ => hYc, fun h => absurd hi h⟩
      by_cases hroot : ∃ r ∈ K.roots ∩ K.heldSet I, K.tok r = i
      · obtain ⟨r, hr, rfl⟩ := hroot
        obtain ⟨X, Y, hXs, hYs, hd⟩ := rootCost C M d aOut c V f K I r hr
        have hY := lookupCost C M d aOut c V f K s k E bcol hCI hk I hI Y (fun u hu => (hYs u hu).1)
          fun u hu u' hu' => (hYs u hu).2.trans (hYs u' hu').2.symm
        exact ⟨X, fun u hu => ⟨(hXs u hu).1.1, (hXs u hu).2⟩, by omega⟩
      · have hnr : ∀ r ∈ K.roots ∩ K.heldSet I, K.tok r ≠ i := fun r hr hri => hroot ⟨r, hr, hri⟩
        obtain ⟨p, hp, hpf, hpi⟩ : ∃ p ∈ W, (C.inputs I ∩ K.pathGates (K.act p)).card < f ∧ K.tok (K.act p) = i := by
          rcases Finset.mem_union.1 hi with hi | hi
          · obtain ⟨r, hr, hri⟩ := Finset.mem_image.1 hi
            exact absurd ⟨r, hr, hri⟩ hroot
          · obtain ⟨p, hp, hpi⟩ := Finset.mem_image.1 hi
            exact ⟨p, (Finset.mem_filter.1 hp).1, (Finset.mem_filter.1 hp).2, hpi⟩
        have hpP : p ∈ M.P := (Finset.mem_filter.1 hp).1
        obtain ⟨r, hr, hfr, hcr⟩ := hcut p hpP
        have hrt : K.tok r = i := (K.follows_tok subset_rfl (K.act_mem p hpP) hfr).trans hpi
        obtain ⟨Y, hY, hYc⟩ := K.heavy_cost hCI hk hI hr (fun r' hr' => hrt ▸ hnr r' hr') (K.cut_legs hp hcr hpf)
        exact ⟨Y, fun u hu => ⟨(hY u hu).1, (hY u hu).2.trans hrt⟩, hYc⟩
    · have hnr : ∀ r ∈ K.roots ∩ K.heldSet I, K.tok r ≠ i := fun r hr hri =>
        hi (Finset.mem_union_left _ (hri ▸ Finset.mem_image_of_mem _ hr))
      have hwide : ∀ p ∈ W.filter (fun p => K.tok (K.act p) = i),
          f ≤ (C.inputs I ∩ K.pathGates (K.act p)).card := by
        intro p hp
        obtain ⟨hpW, hpi⟩ := Finset.mem_filter.1 hp
        by_contra hlt
        exact hi (Finset.mem_union_right _ (Finset.mem_image.2 ⟨p, Finset.mem_filter.2 ⟨hpW, not_le.1 hlt⟩, hpi⟩))
      obtain ⟨X, hXs, hcard⟩ := K.charge_wide hnr (W.filter fun p => K.tok (K.act p) = i)
        (Finset.filter_subset _ _) (fun p hp => (Finset.mem_filter.1 hp).2) hwide
      exact ⟨X, ∅, fun u hu => ⟨(hXs u hu).1, (hXs u hu).2.1⟩, by simpa using K.light_cost hXs,
        fun h => absurd h hi, fun _ => hcard⟩
  choose Xt Yt hXt using hX
  set Tl : Finset Tok := (W.image fun p => K.tok (K.act p)).filter (· ∉ Th)
  have hTdisj : Disjoint Tl Th := Finset.disjoint_left.2 fun i hi hi' => (Finset.mem_filter.1 hi).2 hi'
  have hXdisj : (Tl : Set Tok).PairwiseDisjoint Xt := by
    intro i _ j _ hij
    refine Finset.disjoint_left.2 fun u hu hu' => hij ?_
    exact ((hXt i).1 u hu).2.symm.trans ((hXt j).1 u hu').2
  refine ⟨Tl.biUnion Xt, fun u hu => ?_, ∑ i ∈ Th, (Yt i).card, Th.card, ?_, ?_, ?_⟩
  · obtain ⟨i, -, hu⟩ := Finset.mem_biUnion.1 hu
    exact ((hXt i).1 u hu).1
  · rw [Finset.card_biUnion hXdisj, Finset.mul_sum]
    have hcost : ∀ i, f * (Xt i).card + (Yt i).card ≤ (C.inputs I ∩ K.tokGates i).card := fun i => (hXt i).2.1
    have htok : ((Tl ∪ Th : Finset Tok) : Set Tok).PairwiseDisjoint fun i => C.inputs I ∩ K.tokGates i := by
      intro i _ j _ hij
      exact (K.tokGates_disjoint hij).mono Finset.inter_subset_right Finset.inter_subset_right
    calc ∑ i ∈ Tl, f * (Xt i).card + ∑ i ∈ Th, (Yt i).card
        ≤ ∑ i ∈ Tl, (f * (Xt i).card + (Yt i).card) + ∑ i ∈ Th, (f * (Xt i).card + (Yt i).card) :=
          Nat.add_le_add (Finset.sum_le_sum fun i _ => Nat.le_add_right _ _)
            (Finset.sum_le_sum fun i _ => Nat.le_add_left _ _)
      _ = ∑ i ∈ Tl ∪ Th, (f * (Xt i).card + (Yt i).card) := (Finset.sum_union hTdisj).symm
      _ ≤ ∑ i ∈ Tl ∪ Th, (C.inputs I ∩ K.tokGates i).card := Finset.sum_le_sum fun i _ => hcost i
      _ = ((Tl ∪ Th).biUnion fun i => C.inputs I ∩ K.tokGates i).card := (Finset.card_biUnion htok).symm
      _ ≤ (C.inputs I ∩ K.vals.biUnion K.gates).card := Finset.card_le_card fun z hz => by
          obtain ⟨i, -, hz⟩ := Finset.mem_biUnion.1 hz
          obtain ⟨hzI, hzt⟩ := Finset.mem_inter.1 hz
          exact Finset.mem_inter.2 ⟨hzI, K.tokGates_subset i hzt⟩
  · calc (d - E - (g - 1)) * Th.card = ∑ _i ∈ Th, (d - E - (g - 1)) := by
          rw [Finset.sum_const, smul_eq_mul, mul_comm]
      _ ≤ ∑ i ∈ Th, (Yt i).card := Finset.sum_le_sum fun i hi => (hXt i).2.2.1 hi
  · have hsplit := Finset.card_filter_add_card_filter_not (s := W) fun p => K.tok (K.act p) ∈ Th
    have hlow : (W.filter fun p => ¬ K.tok (K.act p) ∈ Th).card ≤ aOut * (Tl.biUnion Xt).card := by
      have hsub : (W.filter fun p => ¬ K.tok (K.act p) ∈ Th) ⊆
          Tl.biUnion fun i => W.filter fun p => K.tok (K.act p) = i := by
        intro p hp
        obtain ⟨hpW, hpT⟩ := Finset.mem_filter.1 hp
        exact Finset.mem_biUnion.2 ⟨K.tok (K.act p), Finset.mem_filter.2 ⟨Finset.mem_image_of_mem _ hpW, hpT⟩,
          Finset.mem_filter.2 ⟨hpW, rfl⟩⟩
      rw [Finset.card_biUnion hXdisj]
      calc _ ≤ (Tl.biUnion fun i => W.filter fun p => K.tok (K.act p) = i).card := Finset.card_le_card hsub
        _ ≤ ∑ i ∈ Tl, (W.filter fun p => K.tok (K.act p) = i).card := Finset.card_biUnion_le
        _ ≤ ∑ i ∈ Tl, aOut * (Xt i).card :=
          Finset.sum_le_sum fun i hi => (hXt i).2.2.2 (Finset.mem_filter.1 hi).2
        _ = aOut * ∑ i ∈ Tl, (Xt i).card := (Finset.mul_sum _ _ _).symm
    have hhigh : (W.filter fun p => K.tok (K.act p) ∈ Th).card ≤ c * Th.card * Lw.card := by
      set Wh := W.filter fun p => K.tok (K.act p) ∈ Th
      have h1 := Finset.card_le_mul_card_image (f := fun p => (K.tok (K.act p), p.2)) Wh c fun q _ => by
        refine K.mult_le q.1 q.2 _ (fun p hp => ?_) fun p hp => ?_
        · exact (Finset.mem_filter.1 (Finset.mem_filter.1 (Finset.mem_filter.1 hp).1).1).1
        · have hq := (Finset.mem_filter.1 hp).2
          exact ⟨congrArg Prod.fst hq, congrArg Prod.snd hq⟩
      have h2 : (Wh.image fun p => (K.tok (K.act p), p.2)) ⊆ Th ×ˢ Lw := by
        intro q hq
        obtain ⟨p, hp, rfl⟩ := Finset.mem_image.1 hq
        exact Finset.mem_product.2 ⟨(Finset.mem_filter.1 hp).2, Finset.mem_image_of_mem _ (Finset.mem_filter.1 hp).1⟩
      calc Wh.card ≤ c * (Wh.image fun p => (K.tok (K.act p), p.2)).card := h1
        _ ≤ c * (Th ×ˢ Lw).card := Nat.mul_le_mul_left _ (Finset.card_le_card h2)
        _ = c * Th.card * Lw.card := by rw [Finset.card_product, mul_assoc]
    rw [← hsplit]
    exact (Nat.add_le_add hhigh hlow).trans_eq (Nat.add_comm _ _)

/-- **The read premise's instance, from wiring** (restated): an inference block spends values `X` and reads `y` values
of `h` heavy tokens, paying `f` gates for each of `X`, one for each of the `y` and `b` per parameter, all from its
inputs outside `θ`, with `(d − E − (g − 1)) h ≤ y` and `#whole ≤ aOut · #X + c · h · #(updated parameters of whole)`.
`ValueReadHardness`'s instance in `FanOutReadIntensity` pays `f` for the heavy tokens' values too, which the wiring
doesn't: a block that holds a root and cuts the word of a value keyed to it spends that value at one gate. -/
theorem valueReadHardness_of_structure {C : Circuit} {J L Tok Val : Type} [DecidableEq J] [DecidableEq L]
    [DecidableEq Val] {M : MulBlocks C J L} {d aOut c V f : ℕ} (K : Keyed M Tok Val d aOut c V f) {s k E g : ℕ}
    {b bcol : ℝ} (hCH : Assumptions.ConditionalUpdateHardness M (fun _ => K.vals.biUnion K.gates) s b)
    (hCI : Assumptions.ColumnsIncompressible K s bcol) (hk : (k : ℝ) < bcol * V * (E + 1)) (hcut : K.PerValueCut g)
    {I : Finset (Fin C.N)} (hI : C.InferenceBlock s k I) :
    ∃ X : Finset Val, (∀ u ∈ X, K.Spent I u) ∧ ∃ y h : ℕ,
      ((f * X.card + y : ℕ) : ℝ) + b * ((M.whole I).image Prod.snd).card ≤ (C.inputs I \ C.θ).card ∧
      (d - E - (g - 1)) * h ≤ y ∧ (M.whole I).card ≤ aOut * X.card + c * h * ((M.whole I).image Prod.snd).card := by
  obtain ⟨X, hXs, y, h, hval, hy, hW⟩ := fanOut_value_split K hCI hk hcut hI
  refine ⟨X, hXs, y, h, ?_, hy, hW⟩
  set R := K.vals.biUnion K.gates
  have hL : b * ((M.whole I).image Prod.snd).card ≤ (C.inputs I \ (C.θ ∪ R)).card := by
    rcases (M.whole I).eq_empty_or_nonempty with hW0 | ⟨p, hp⟩
    · rw [hW0, Finset.image_empty, Finset.card_empty, Nat.cast_zero, mul_zero]
      exact Nat.cast_nonneg _
    · refine (hCH I hI.1 (M.whole I) (fun p hp => M.P_subset (Finset.mem_filter.1 hp).1)
        fun p hp => (Finset.mem_filter.1 hp).2).trans ?_
      have hR : R ⊆ M.uses.biUnion fun _ => R := fun z hz =>
        Finset.mem_biUnion.2 ⟨p.1, Finset.mem_image_of_mem _ (Finset.mem_filter.1 hp).1, hz⟩
      exact_mod_cast Finset.card_le_card (Finset.sdiff_subset_sdiff le_rfl (Finset.union_subset_union le_rfl hR))
  have hRθ : C.inputs I ∩ R ⊆ (C.inputs I \ C.θ) ∩ R := by
    intro z hz
    obtain ⟨hzI, hzR⟩ := Finset.mem_inter.1 hz
    obtain ⟨v, hv, hzv⟩ := Finset.mem_biUnion.1 hzR
    exact Finset.mem_inter.2 ⟨Finset.mem_sdiff.2 ⟨hzI, Finset.disjoint_left.1 (K.gates_θ v hv) hzv⟩, hzR⟩
  have hsplit : C.inputs I \ (C.θ ∪ R) = (C.inputs I \ C.θ) \ R := by
    ext x
    simp only [Finset.mem_sdiff, Finset.mem_union, not_or]
    tauto
  have hsum : (((C.inputs I \ C.θ) ∩ R).card : ℝ) + ((C.inputs I \ (C.θ ∪ R)).card : ℝ) =
      (C.inputs I \ C.θ).card := by
    rw [hsplit]
    exact_mod_cast Finset.card_inter_add_card_sdiff _ _
  have hval' : ((f * X.card + y : ℕ) : ℝ) ≤ ((C.inputs I \ C.θ) ∩ R).card := by
    exact_mod_cast hval.trans (Finset.card_le_card hRθ)
  linarith

theorem FanOutReadIntensityStructural : Guarantees.FanOutReadIntensityStructural := by
  intro C J L Tok Val _ _ _ M d aOut c V f K s k E g b bcol hb hCH hCI hcut hk I hI
  obtain ⟨X, -, y, h, hpay, hy, hW⟩ := valueReadHardness_of_structure K hCH hCI hk hcut hI
  have hcap : ((C.inputs I \ C.θ).card : ℝ) ≤ k := by exact_mod_cast hI.2
  have hpay' := hpay.trans hcap
  push_cast at hpay'
  have hrest : ((M.finished I \ M.whole I).card : ℝ) ≤ k := by
    exact_mod_cast (finished_sdiff_whole_card_le M).trans hI.2
  have hfin : ((M.finished I).card : ℝ) ≤ (M.finished I \ M.whole I).card + (M.whole I).card := by
    exact_mod_cast Finset.card_le_card_sdiff_add_card
  have hW' : ((M.whole I).card : ℝ) ≤ aOut * X.card + c * h * ((M.whole I).image Prod.snd).card := by
    exact_mod_cast hW
  generalize d - E - (g - 1) = δ at hy ⊢
  have hy' : (δ : ℝ) * h ≤ y := by exact_mod_cast hy
  set x : ℝ := (X.card : ℝ)
  set w : ℝ := (((M.whole I).image Prod.snd).card : ℝ)
  set m := max (4 * (δ : ℝ) * b * aOut) ((c : ℝ) * f * k)
  have hx0 : (0 : ℝ) ≤ x := Nat.cast_nonneg _
  have hy0 : (0 : ℝ) ≤ y := Nat.cast_nonneg _
  have hc0 : (0 : ℝ) ≤ c := Nat.cast_nonneg _
  have hf0 : (0 : ℝ) ≤ f := Nat.cast_nonneg _
  have hw0 : 0 ≤ w := Nat.cast_nonneg _
  have hbw : 0 ≤ b * w := by positivity
  have hcoef : 0 ≤ 4 * (δ : ℝ) * f * b := by positivity
  have e1 : 4 * (δ : ℝ) * f * b * (M.finished I).card ≤ 4 * δ * f * b * (k + (aOut * x + c * h * w)) :=
    mul_le_mul_of_nonneg_left (hfin.trans (add_le_add hrest hW')) hcoef
  have e2 : 4 * (y : ℝ) * (b * w) ≤ (k - f * x) * (k - f * x) := four_mul_le_sq hy0 hbw (by linarith)
  have e3 : 4 * ((δ : ℝ) * h) * (b * w) ≤ 4 * y * (b * w) :=
    mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hy' (by norm_num)) hbw
  have e4 : ((k : ℝ) - f * x) * (k - f * x) ≤ k * (k - f * x) :=
    mul_le_mul_of_nonneg_right (by linarith [mul_nonneg hf0 hx0]) (by linarith)
  have e5 : (c : ℝ) * f * (4 * ((δ : ℝ) * h) * (b * w)) ≤ c * f * (k * (k - f * x)) :=
    mul_le_mul_of_nonneg_left (e3.trans (e2.trans e4)) (mul_nonneg hc0 hf0)
  have e6 : 4 * (δ : ℝ) * b * aOut * (f * x) ≤ m * (f * x) :=
    mul_le_mul_of_nonneg_right (le_max_left _ _) (mul_nonneg hf0 hx0)
  have e7 : (c : ℝ) * f * k * (k - f * x) ≤ m * (k - f * x) :=
    mul_le_mul_of_nonneg_right (le_max_right _ _) (by linarith)
  linarith

theorem FanOutReadCoverBoundStructural : Guarantees.FanOutReadCoverBoundStructural := by
  intro C J L Tok Val _ _ _ M d aOut c V f K s k E g b bcol 𝓘 hb hCH hCI hcut hk h𝓘 hcov
  exact cover_sum M hcov (by positivity) fun I hI =>
    FanOutReadIntensityStructural C M d aOut c V f K s k E g b bcol hb hCH hCI hcut hk I (h𝓘 I hI)

end SecurityProofs

end Nci.Cover
