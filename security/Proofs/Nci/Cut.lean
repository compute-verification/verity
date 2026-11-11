module

public import Definitions.Nci.Guarantees.Cut
public import Proofs.Core.Circuit
public import Proofs.Nci.Covering
public import Proofs.Nci.LoomisWhitneyBase
public import Mathlib.Algebra.BigOperators.Field
public import Mathlib.Algebra.BigOperators.Fin
public import Mathlib.Analysis.SpecificLimits.Basic
public import Mathlib.Combinatorics.Enumerative.DoubleCounting
public import Mathlib.Combinatorics.SimpleGraph.Extremal.Turan
public import Mathlib.Data.Fintype.BigOperators

/-!
# The min-cut bound, its slowdown, and its corollaries

Proofs of the guarantees of `Definitions.Nci.Guarantees.Cut`. Fix a partition into `m` blocks, each with at most `k`
inputs outside `θ` (`ReadsAtMost`); a block's **entries** (`entries`) are those inputs, at most `m k` in all. A MAC is
**split** when no block holds all its gates.

1. **Split MACs** (`card_split_le`): a split MAC is connected, so one of its wires runs between two blocks
   (`exists_cross`); that wire is a computed gate the other block reads, so it is one of that block's entries, and
   disjoint MACs give distinct entries. At most `m k` MACs are split.
2. **A block's words** (`kappaWords_readWords_le`): every path from an unregistered input to a wire a block's MACs read
   enters the block through one of its entries (`meets_into`), so the words its MACs read have `κ ≤ k`, and their `π`
   is at most `Π(k)` (`pi_readWords_le`).
3. **Loomis–Whitney** (`card_inBlock_sq_le`): a block's MACs of one matmul are index triples `(i, j, l)`, whose
   projections are the outputs `(i, j)` they feed, the `w`-words `(i, l)` and the `x`-words `(l, j)` they read. The
   discrete Loomis–Whitney inequality (`Nci.SecurityProofs.loomis_whitney`) and Cauchy–Schwarz over the matmuls give
   `(MACs in the block)² ≤ π(its words) · (outputs it feeds)`, and Cauchy–Schwarz over the blocks gives
   `V² ≤ m Π(k) F` for the `V` unsplit MACs and the `F` pairs of a block and an output it feeds (`counts`).
4. **Outputs** (`card_fedBy_le`, `outOn_le`, `bY_mul_card_clear_le`): an output a block feeds has a wire in the
   block, at most `O` such pairs in all, or lies in `E(block)`. The block's gates that another block reads (`leaving`)
   separate it from `E(block)` (`meets_exits`), so `b_y |E(block)| ≤ |leaving|`, and each leaving gate is an entry of
   the block reading it (`sum_card_leaving_le`): `b_y ∑ |E| ≤ m k`.
5. **Algebra** (`real_step`): `N ≤ m k + V`, `V ≤ √(m Π O) + √(m Π ∑ |E|)` and `∑ |E| ≤ m k / b_y` give `Bound`, and
   `Slowdown` follows by cases on `m` against `N / (k + α)` (`slow_step`).

Under (H) at the arrays `W` and `X`, `onePremise` bounds `π(S) ≤ |val(S ∩ W)| |val(S ∩ X)|` (`pi_le_card_mul`: a MAC's
pair of words goes to its pair of values, one-to-one since `val` is one-to-one on each matmul's words and no two matmuls
share a pair of values) and applies AM–GM (`amgm_bound`). With shared arrays, `|S ∩ W_a| ≤ |val(S ∩ A r)|`
(`card_inter_le_image`), and `u_r = β_r |val(S ∩ A r)|` and `q ≤ β_r β_s` on each matmul's arrays give
`q π(S) ≤ ∑ u_r u_s` over those pairs, which `sharedArraysClique` bounds by the weighted Turán inequality
(`weighted_turan_real`), the limit of Turán's theorem on the graph that replaces each array `r` by `⌊n u_r⌋` copies
(`blowUp`, `weighted_turan`).

`HE` and `HEShared` put those bounds on `Π(k)`, and `b ≤ b_y` from (E) (`le_bY`, `alpha_le_of_exports`), into
`Bound` (`bound_of_le`), and take the slowdown from the bound so obtained (`slowdown_of_bound`).

**Over a set of blocks.** `counts_on` is steps 1 to 4 with the fresh-input budget on a set `G` of blocks only, each
block of `G` under its own, and the entries of all the blocks together at most a total `T`. The split MACs and the
leaving gates are entries, whichever block reads them, so they are counted against `T`; steps 2 and 3 are per block
(`card_held_sq_le`), so they need the budget on `G` alone and count only the pairs of a block of `G` and an output with
a wire in it (`outOn`). An output with wires in two blocks of `G` counts once for each: nothing here bounds `N` with the
distinct outputs in the pairs' place. The pairs are at most the output wires in the blocks of `G` (`wiresOn`),
themselves at most `O` (`outOn_le_wiresOn`, `wiresOn_le`). The MACs a block outside `G` holds are counted apart, at most
its gates (`sum_card_inBlock_le`), and `T` is over every block, those outside `G` too. Those are the MACs each of whose
gates is in the block (`card_whole_eq`). `counts` is `counts_on` at every block. `bound_on_budgets` is step 5 on those
counts, with `S = ∑ Π(κ u)` over `G` for `m Π(k)` (`real_step_on`). Under (H) at each block's own budget and its own
`β_x`, and (E), `bound_on_rows` puts `κ u² / (4 β_w β_x u)` for each `Π(κ u)`; `bound_on_budgets_he` is it at one `β_x`,
and `bound_on` and `bound_on_he` are `Bound` and `HE`'s bound over `G` at one budget.
-/

@[expose] public section

namespace Nci.Cut.SecurityProofs

open Verity Finset Nci.Covering

variable {C : Circuit}

/-! ### Paths -/

theorem feedsWithin_univ {g t : Fin C.N} (h : C.Feeds g t) : FeedsWithin C univ g t := by
  induction h with
  | refl t => exact .refl (mem_univ t)
  | step hgh _ ih => exact .step (mem_univ _) hgh ih

theorem feedsWithin_snoc {R : Finset (Fin C.N)} {g y h : Fin C.N} (hp : FeedsWithin C R g y) :
    y ∈ C.reads h → h ∈ R → FeedsWithin C R g h := by
  induction hp with
  | refl hy' => exact fun hy hh => .step hy' hy (.refl hh)
  | step hg hgh _ ih => exact fun hy hh => .step hg hgh (ih hy hh)

theorem mem_computed {g : Fin C.N} : g ∈ C.computed ↔ C.isInput g = false := by
  simp [Circuit.computed]

theorem reader_computed {a h : Fin C.N} (hah : a ∈ C.reads h) : C.isInput h = false := by
  cases hi : C.isInput h
  · rfl
  · rw [C.input_reads h hi] at hah
    exact absurd hah (notMem_empty a)

theorem mem_into {S : Finset (Fin C.N)} {a : Fin C.N} : a ∈ C.into S ↔ a ∉ S ∧ ∃ g ∈ S, a ∈ C.reads g := by
  simp only [Circuit.into, mem_filter, mem_univ, true_and]

/-- A path from outside `S` into `S` passes a gate outside `S` that a gate of `S` reads: its start, or a computed
gate. -/
theorem enters {R S : Finset (Fin C.N)} {g t : Fin C.N} (hp : FeedsWithin C R g t) :
    g ∉ S → t ∈ S → ∃ v ∈ R, v ∉ S ∧ (∃ h ∈ S, v ∈ C.reads h) ∧ (v = g ∨ C.isInput v = false) := by
  induction hp with
  | refl _ => exact fun hg ht => absurd ht hg
  | @step g h t hg hgh _ ih =>
    intro hgS htS
    by_cases hh : h ∈ S
    · exact ⟨g, hg, hgS, ⟨h, hh, hgh⟩, Or.inl rfl⟩
    · obtain ⟨v, hv, hvS, hvh, hvg⟩ := ih hh htS
      refine ⟨v, hv, hvS, hvh, Or.inr ?_⟩
      rcases hvg with rfl | hvg
      · exact reader_computed hgh
      · exact hvg

/-- A block's inputs outside `θ` separate the unregistered inputs from every wire the block reads. -/
theorem meets_into (d : RegisteredCircuit) {B Y : Finset (Fin d.D.N)} (hB : B ⊆ d.D.computed)
    (hY : ∀ y ∈ Y, ∃ h ∈ B, y ∈ d.D.reads h) : Separates d.D (d.D.into B \ d.θ) (unregisteredInputs d) Y := by
  intro x hx y hy hp
  obtain ⟨h, hh, hyh⟩ := hY y hy
  have hhR : h ∈ univ \ (d.D.into B \ d.θ) :=
    mem_sdiff.2 ⟨mem_univ _, fun hi => (mem_into.1 (mem_sdiff.1 hi).1).1 hh⟩
  have hxi : d.D.isInput x = true := (mem_filter.1 hx).2.1
  have hxB : x ∉ B := fun hxB => by
    have := mem_computed.1 (hB hxB)
    rw [hxi] at this
    cases this
  obtain ⟨v, hv, hvB, hvh, hvx⟩ := enters (feedsWithin_snoc hp hyh hhR) hxB hh
  refine (mem_sdiff.1 hv).2 (mem_sdiff.2 ⟨mem_into.2 ⟨hvB, hvh⟩, fun hvθ => ?_⟩)
  rcases hvx with rfl | hvx
  · exact (mem_filter.1 hx).2.2 hvθ
  · rw [d.θ_input v hvθ] at hvx
    cases hvx

/-- The gates of `B` that a gate outside `B` reads: `out(B)` (`Circuit.exits`) without its output gates. -/
def leaving (B : Finset (Fin C.N)) : Finset (Fin C.N) := B.filter fun v => ∃ h, h ∉ B ∧ v ∈ C.reads h

theorem leaves {R B : Finset (Fin C.N)} {g t : Fin C.N} (hp : FeedsWithin C R g t) :
    g ∈ B → t ∉ B → ∃ v ∈ R, v ∈ leaving B := by
  induction hp with
  | refl _ => exact fun hg ht => absurd hg ht
  | @step g h t hg hgh _ ih =>
    intro hgB htB
    by_cases hh : h ∈ B
    · exact ih hh htB
    · exact ⟨g, hg, mem_filter.2 ⟨hgB, h, hh, hgh⟩⟩

/-- The gates leaving `B` separate `B` from every set of gates outside it. -/
theorem meets_exits {B Y : Finset (Fin C.N)} (hY : Disjoint Y B) : Separates C (leaving B) B Y := by
  intro x hx y hy hp
  obtain ⟨v, hv, hvl⟩ := leaves hp hx (disjoint_left.1 hY hy)
  exact (mem_sdiff.1 hv).2 hvl

theorem kappa_le {X Y Z : Finset (Fin C.N)} (hZ : Separates C Z X Y) : kappa C X Y ≤ Z.card := by
  unfold kappa
  exact Nat.sInf_le ⟨Z, rfl, hZ⟩

/-- A least separator exists: all the gates separate `X` from `Y`. -/
theorem exists_separator (X Y : Finset (Fin C.N)) :
    ∃ Z : Finset (Fin C.N), Z.card = kappa C X Y ∧ Separates C Z X Y := by
  have hne : {n | ∃ Z : Finset (Fin C.N), Z.card = n ∧ Separates C Z X Y}.Nonempty := by
    refine ⟨_, univ, rfl, ?_⟩
    intro a _ b _ hab
    simpa using Nci.SecurityProofs.Covering.FeedsWithin.mem_right hab
  exact Nat.sInf_mem hne

theorem kappa_pos {X Y : Finset (Fin C.N)} {x y : Fin C.N} (hx : x ∈ X) (hy : y ∈ Y) (hxy : C.Feeds x y) :
    0 < kappa C X Y := by
  obtain ⟨Z, hZ, hsep⟩ := exists_separator X Y
  rcases Z.eq_empty_or_nonempty with rfl | hZne
  · exact (hsep x hx y hy (by rw [sdiff_empty]; exact feedsWithin_univ hxy)).elim
  · exact lt_of_lt_of_eq (card_pos.2 hZne) hZ

/-- Every separator holds `X ∩ Y`: a gate of both is a path from `X` to `Y` by itself. -/
theorem card_inter_le_kappa (X Y : Finset (Fin C.N)) : (X ∩ Y).card ≤ kappa C X Y := by
  obtain ⟨Z, hZ, hsep⟩ := exists_separator X Y
  rw [← hZ]
  refine card_le_card fun g hg => ?_
  by_contra hgZ
  exact hsep g (mem_inter.1 hg).1 g (mem_inter.1 hg).2 (.refl (mem_sdiff.2 ⟨mem_univ g, hgZ⟩))

/-! ### Blocks -/

variable {m : ℕ}

theorem exists_unit (P : Partition C m) {g : Fin C.N} (hg : C.isInput g = false) : ∃ u, P.unit g = some u :=
  Option.ne_none_iff_exists'.1 fun h => by simp [(P.input_iff g).1 h] at hg

theorem gate_computed (P : Partition C m) {u : Fin m} {g : Fin C.N} (hg : g ∈ P.gates u) : C.isInput g = false :=
  mem_computed.1 (P.gates_subset_computed u hg)

theorem sum_card_inter_gates_le (P : Partition C m) (Y : Finset (Fin C.N)) : ∑ u, (Y ∩ P.gates u).card ≤ Y.card := by
  rw [← card_biUnion]
  · exact card_le_card (biUnion_subset.2 fun u _ => inter_subset_left)
  · intro u _ v _ huv
    refine disjoint_left.2 fun g hu hv => huv ?_
    exact Option.some_injective _
      ((P.mem_gates.1 (mem_inter.1 hu).2).symm.trans (P.mem_gates.1 (mem_inter.1 hv).2))

/-- The blocks' entries: a block and one of its inputs outside `θ`. -/
def entries {d : RegisteredCircuit} (P : Partition d.D m) : Finset (Σ _ : Fin m, Fin d.D.N) :=
  univ.sigma fun u => d.D.into (P.gates u) \ d.θ

theorem card_entries_le {d : RegisteredCircuit} {T : ℕ} (P : Partition d.D m)
    (hT : ∑ u, (d.unregistered (P.gates u)).card ≤ T) : (entries P).card ≤ T := by
  rw [entries, card_sigma]
  exact hT

/-- A computed gate that a gate of block `u` reads from outside `u` is an entry of `u`. -/
theorem mem_entries {d : RegisteredCircuit} (P : Partition d.D m) {a h : Fin d.D.N} {u : Fin m}
    (ha : d.D.isInput a = false) (hah : a ∈ d.D.reads h) (hu : P.unit h = some u) (hau : P.unit a ≠ some u) :
    (⟨u, a⟩ : Σ _ : Fin m, Fin d.D.N) ∈ entries P := by
  refine mem_sigma.2 ⟨mem_univ _, mem_sdiff.2 ⟨mem_into.2 ⟨fun h' => hau (P.mem_gates.1 h'), h, P.mem_gates.2 hu, hah⟩,
    fun hθ => ?_⟩⟩
  rw [d.θ_input a hθ] at ha
  cases ha

theorem false_of_iff_not (p : Prop) (h : p ↔ ¬p) : False :=
  have hn : ¬p := fun hp => h.1 hp hp
  hn (h.2 hn)

/-- A connected set of computed gates that no block holds has a wire between two blocks. -/
theorem exists_cross (P : Partition C m) {A : Finset (Fin C.N)} (hA : Connected C A)
    (hc : ∀ a ∈ A, C.isInput a = false) (hsplit : ∀ u, ¬ A ⊆ P.gates u) :
    ∃ a ∈ A, ∃ h ∈ A, a ∈ C.reads h ∧ P.unit a ≠ P.unit h := by
  obtain ⟨g₀, hg₀⟩ := hA.1
  obtain ⟨u₀, hu₀⟩ := exists_unit P (hc g₀ hg₀)
  have hT : A.filter (fun g => P.unit g = some u₀) ≠ A := by
    intro heq
    refine hsplit u₀ fun g hg => ?_
    rw [← heq] at hg
    exact P.mem_gates.2 (mem_filter.1 hg).2
  obtain ⟨a, ha, h, hh, hah, hcross⟩ :=
    hA.2 (A.filter fun g => P.unit g = some u₀) (filter_subset _ _) ⟨g₀, mem_filter.2 ⟨hg₀, hu₀⟩⟩ hT
  refine ⟨a, ha, h, hh, hah, fun heq => ?_⟩
  have hfa : a ∈ A.filter (fun g => P.unit g = some u₀) ↔ P.unit a = some u₀ := mem_filter.trans (and_iff_right ha)
  have hfh : h ∈ A.filter (fun g => P.unit g = some u₀) ↔ P.unit a = some u₀ := by
    rw [heq]
    exact mem_filter.trans (and_iff_right hh)
  exact false_of_iff_not _ (hfa.symm.trans (hcross.trans (not_congr hfh)))

/-- **Step 1.** At most `m k` MACs are split. -/
theorem card_split_le {d : RegisteredCircuit} {ι : Type} [Fintype ι] (P : Partition d.D m) (W : Work d.D ι) :
    (univ.filter fun s : W.Mac => ∀ u, ¬ W.mac s ⊆ P.gates u).card ≤ (entries P).card := by
  refine card_le_card_of_forall_subsingleton (fun s e => e.2 ∈ W.mac s) (fun s hs => ?_) fun e _ s hs t ht => ?_
  · obtain ⟨a, ha, h, hh, hah, hne⟩ :=
      exists_cross P ((W.M s.1).mac_connected s.2) ((W.M s.1).mac_computed s.2) (mem_filter.1 hs).2
    obtain ⟨uh, huh⟩ := exists_unit P (reader_computed hah)
    exact ⟨⟨uh, a⟩, mem_entries P ((W.M s.1).mac_computed s.2 a ha) hah huh fun hau => hne (hau.trans huh.symm), ha⟩
  · by_contra hst
    exact disjoint_left.1 (W.mac_disjoint s t hst) hs.2 ht.2

/-- Block `u`'s MACs of matmul `a`. -/
def inBlock {ι : Type} (P : Partition C m) (W : Work C ι) (u : Fin m) (a : ι) :
    Finset (Fin (W.M a).ni × Fin (W.M a).nj × Fin (W.M a).nl) :=
  univ.filter fun t => (W.M a).mac t ⊆ P.gates u

/-- A block holds at most as many MACs as it has gates: its MACs are disjoint and none is empty. -/
theorem sum_card_inBlock_le {ι : Type} [Fintype ι] (P : Partition C m) (W : Work C ι) (u : Fin m) :
    ∑ a, (inBlock P W u a).card ≤ (P.gates u).card := by
  rw [← card_sigma]
  refine card_le_card_of_forall_subsingleton (fun s g => g ∈ W.mac s) (fun s hs => ?_) fun g _ s hs t ht => ?_
  · obtain ⟨g, hg⟩ := ((W.M s.1).mac_connected s.2).1
    exact ⟨g, (mem_filter.1 (mem_sigma.1 hs).2).2 hg, hg⟩
  · by_contra hst
    exact disjoint_left.1 (W.mac_disjoint s t hst) hs.2 ht.2

/-- The MACs block `u` holds, over every matmul, are the MACs each of whose gates is in the block. -/
theorem card_whole_eq {ι : Type} [Fintype ι] (P : Partition C m) (W : Work C ι) (u : Fin m) :
    (univ.filter fun s : W.Mac => W.mac s ⊆ P.gates u).card = ∑ a, (inBlock P W u a).card :=
  calc (univ.filter fun s : W.Mac => W.mac s ⊆ P.gates u).card
      = (univ.sigma (inBlock P W u)).card := congrArg Finset.card (Finset.ext fun _ =>
        ⟨fun h => mem_sigma.2 ⟨mem_univ _, mem_filter.2 ⟨mem_univ _, (mem_filter.1 h).2⟩⟩,
          fun h => mem_filter.2 ⟨mem_univ _, (mem_filter.1 (mem_sigma.1 h).2).2⟩⟩)
    _ = ∑ a, (inBlock P W u a).card := card_sigma _ _

/-- The words block `u`'s MACs read. -/
def readWords {ι : Type} [Fintype ι] (P : Partition C m) (W : Work C ι) (u : Fin m) : Finset (Finset (Fin C.N)) :=
  (univ.sigma (inBlock P W u)).biUnion fun s => {(W.M s.1).w (s.2.1, s.2.2.2), (W.M s.1).x (s.2.2.2, s.2.2.1)}

/-- The outputs block `u`'s MACs feed. -/
def fedBy {ι : Type} [Fintype ι] (P : Partition C m) (W : Work C ι) (u : Fin m) : Finset W.Out :=
  univ.sigma fun a => (inBlock P W u a).image fun t => (t.1, t.2.1)

/-- The outputs with a wire in block `u`. -/
def hit {ι : Type} [Fintype ι] (P : Partition C m) (W : Work C ι) (u : Fin m) : Finset W.Out :=
  univ.filter fun o => ¬ Disjoint (W.y o) (P.gates u)

/-- `O` over the blocks of `G`: the pairs of a block of `G` and an output with a wire in it. An output with a wire in
each of two blocks of `G` counts twice: the count over `G` is in these pairs, not in the distinct outputs. -/
def outOn {ι : Type} [Fintype ι] (P : Partition C m) (W : Work C ι) (G : Finset (Fin m)) : ℕ :=
  ∑ u ∈ G, (hit P W u).card

theorem mem_readWords {ι : Type} [Fintype ι] (P : Partition C m) (W : Work C ι) (u : Fin m) {a : Fin C.N}
    (ha : a ∈ (readWords P W u).biUnion id) : ∃ h ∈ P.gates u, a ∈ C.reads h := by
  simp only [readWords, mem_biUnion, mem_sigma, mem_univ, true_and, id, mem_insert, mem_singleton] at ha
  obtain ⟨S, ⟨s, hs, hS⟩, haS⟩ := ha
  rcases hS with rfl | rfl
  · obtain ⟨h, hh, hah⟩ := (W.M s.1).reads_w s.2 a haS
    exact ⟨h, (mem_filter.1 hs).2 hh, hah⟩
  · obtain ⟨h, hh, hah⟩ := (W.M s.1).reads_x s.2 a haS
    exact ⟨h, (mem_filter.1 hs).2 hh, hah⟩

/-- **Step 2.** The words a block's MACs read have `κ` at most the block's inputs outside `θ`. -/
theorem kappaWords_readWords_le {d : RegisteredCircuit} {ι : Type} [Fintype ι] (P : Partition d.D m) (W : Work d.D ι)
    (u : Fin m) : kappaWords d (readWords P W u) ≤ (d.D.into (P.gates u) \ d.θ).card := by
  unfold kappaWords
  exact kappa_le (meets_into d (P.gates_subset_computed u) fun _ ha => mem_readWords P W u ha)

theorem pi_readWords_le {d : RegisteredCircuit} {ι : Type} [Fintype ι] {k : ℕ} (P : Partition d.D m) (W : Work d.D ι)
    {u : Fin m} (hk : (d.unregistered (P.gates u)).card ≤ k) : W.pi (readWords P W u) ≤ piMax d W k := by
  unfold piMax
  exact le_sup (f := W.pi) (mem_filter.2 ⟨mem_univ _, (kappaWords_readWords_le P W u).trans hk⟩)

/-- **Step 3.** Loomis–Whitney at each matmul and Cauchy–Schwarz over the matmuls: a block's MACs, squared, are at
most `π` of the words they read times the outputs they feed. -/
theorem card_inBlock_sq_le {ι : Type} [Fintype ι] (P : Partition C m) (W : Work C ι) (u : Fin m) :
    (∑ a, (inBlock P W u a).card) ^ 2 ≤ W.pi (readWords P W u) * (fedBy P W u).card := by
  rw [fedBy, card_sigma, Work.pi]
  refine sum_sq_le_sum_mul_sum_of_sq_le_mul _ (fun _ _ => Nat.zero_le _) (fun _ _ => Nat.zero_le _) fun a _ => ?_
  have hw : ((inBlock P W u a).image fun t => (t.1, t.2.2)).card ≤ (readWords P W u ∩ W.wWords a).card := by
    refine card_le_card_of_injOn (W.M a).w (fun p hp => ?_) fun p _ q _ hpq => (W.M a).w_inj hpq
    obtain ⟨t, ht, rfl⟩ := mem_image.1 (mem_coe.1 hp)
    refine mem_coe.2 (mem_inter.2 ⟨mem_biUnion.2 ⟨⟨a, t⟩, mem_sigma.2 ⟨mem_univ _, ht⟩, ?_⟩,
      mem_image_of_mem (W.M a).w (mem_univ _)⟩)
    exact mem_insert.2 (Or.inl rfl)
  have hx : ((inBlock P W u a).image fun t => (t.2.1, t.2.2)).card ≤ (readWords P W u ∩ W.xWords a).card := by
    refine card_le_card_of_injOn (fun p => (W.M a).x (p.2, p.1)) (fun p hp => ?_) fun p _ q _ hpq => ?_
    · obtain ⟨t, ht, rfl⟩ := mem_image.1 (mem_coe.1 hp)
      refine mem_coe.2 (mem_inter.2 ⟨mem_biUnion.2 ⟨⟨a, t⟩, mem_sigma.2 ⟨mem_univ _, ht⟩, ?_⟩,
        mem_image_of_mem (W.M a).x (mem_univ _)⟩)
      exact mem_insert.2 (Or.inr (mem_singleton.2 rfl))
    · have h := (W.M a).x_inj hpq
      exact Prod.ext (congrArg Prod.snd h) (congrArg Prod.fst h)
  calc (inBlock P W u a).card ^ 2
      ≤ ((inBlock P W u a).image fun t => (t.1, t.2.1)).card * ((inBlock P W u a).image fun t => (t.1, t.2.2)).card *
          ((inBlock P W u a).image fun t => (t.2.1, t.2.2)).card := Nci.SecurityProofs.loomis_whitney _
    _ ≤ ((inBlock P W u a).image fun t => (t.1, t.2.1)).card * (readWords P W u ∩ W.wWords a).card *
          (readWords P W u ∩ W.xWords a).card := Nat.mul_le_mul (Nat.mul_le_mul le_rfl hw) hx
    _ = (readWords P W u ∩ W.wWords a).card * (readWords P W u ∩ W.xWords a).card *
          ((inBlock P W u a).image fun t => (t.1, t.2.1)).card := by ring

/-- An output a block feeds has a wire in the block or lies in `E(block)`. -/
theorem card_fedBy_le {ι : Type} [Fintype ι] [DecidableEq ι] (P : Partition C m) (W : Work C ι) (u : Fin m) :
    (fedBy P W u).card ≤ (hit P W u).card + (W.clear (P.gates u)).card := by
  calc (fedBy P W u).card
      ≤ ((univ.filter fun o : W.Out => ¬ Disjoint (W.y o) (P.gates u)) ∪ W.clear (P.gates u)).card := by
        refine card_le_card fun o ho => ?_
        obtain ⟨a, p⟩ := o
        simp only [fedBy, mem_sigma, mem_univ, true_and, mem_image] at ho
        obtain ⟨t, ht, rfl⟩ := ho
        by_cases h : Disjoint (W.y ⟨a, (t.1, t.2.1)⟩) (P.gates u)
        · refine mem_union_right _ ?_
          simp only [Work.clear, mem_filter, mem_univ, true_and]
          exact ⟨⟨⟨a, t⟩, rfl, (mem_filter.1 ht).2⟩, h⟩
        · exact mem_union_left _ (mem_filter.2 ⟨mem_univ _, h⟩)
    _ ≤ _ := card_union_le _ _

/-- A block has at most as many outputs with a wire in it as output wires in it. -/
theorem card_hit_le {ι : Type} [Fintype ι] (P : Partition C m) (W : Work C ι) (u : Fin m) :
    (hit P W u).card ≤ ∑ o : W.Out, (W.y o ∩ P.gates u).card := by
  calc (univ.filter fun o : W.Out => ¬ Disjoint (W.y o) (P.gates u)).card
      = ∑ o : W.Out, if ¬ Disjoint (W.y o) (P.gates u) then 1 else 0 := card_filter _ _
    _ ≤ ∑ o : W.Out, (W.y o ∩ P.gates u).card := by
        refine sum_le_sum fun o _ => ?_
        by_cases h : Disjoint (W.y o) (P.gates u)
        · rw [ite_eq_right (not_not.2 h)]
          exact Nat.zero_le _
        · rw [ite_eq_left h]
          exact card_pos.2 (not_disjoint_iff_nonempty_inter.1 h)

/-- `O` over the blocks of `G`, in wires: the output wires in the blocks of `G`, as `O` counts all of them. -/
def wiresOn {ι : Type} [Fintype ι] (P : Partition C m) (W : Work C ι) (G : Finset (Fin m)) : ℕ :=
  ∑ u ∈ G, ∑ o : W.Out, (W.y o ∩ P.gates u).card

/-- The pairs of a block of `G` and an output with a wire in it are at most the output wires in the blocks of `G`. -/
theorem outOn_le_wiresOn {ι : Type} [Fintype ι] (P : Partition C m) (W : Work C ι) (G : Finset (Fin m)) :
    outOn P W G ≤ wiresOn P W G :=
  sum_le_sum fun u _ => card_hit_le P W u

/-- Each output wire lies in at most one block: the blocks of `G` hold at most `O` output wires between them. -/
theorem wiresOn_le {ι : Type} [Fintype ι] (P : Partition C m) (W : Work C ι) (G : Finset (Fin m)) :
    wiresOn P W G ≤ W.O := by
  calc ∑ u ∈ G, ∑ o : W.Out, (W.y o ∩ P.gates u).card
      ≤ ∑ u, ∑ o : W.Out, (W.y o ∩ P.gates u).card := sum_le_sum_of_subset (subset_univ G)
    _ = ∑ o : W.Out, ∑ u, (W.y o ∩ P.gates u).card := sum_comm
    _ ≤ ∑ o : W.Out, (W.y o).card := sum_le_sum fun o _ => sum_card_inter_gates_le P (W.y o)
    _ = W.O := rfl

/-- The blocks of `G` have at most `O` pairs of a block and an output with a wire in it between them. -/
theorem outOn_le {ι : Type} [Fintype ι] (P : Partition C m) (W : Work C ι) (G : Finset (Fin m)) :
    outOn P W G ≤ W.O :=
  (outOn_le_wiresOn P W G).trans (wiresOn_le P W G)

/-- **Steps 2 and 3 at one block.** The MACs a block reading at most `k` gates outside `θ` holds, squared, are at most
`Π(k)` times the outputs with a wire in the block and those in `E(block)`. -/
theorem card_held_sq_le {d : RegisteredCircuit} {ι : Type} [Fintype ι] [DecidableEq ι] {k : ℕ} (P : Partition d.D m)
    (W : Work d.D ι) {u : Fin m} (hk : (d.unregistered (P.gates u)).card ≤ k) :
    (∑ a, (inBlock P W u a).card) ^ 2 ≤ piMax d W k * ((hit P W u).card + (W.clear (P.gates u)).card) :=
  ((card_inBlock_sq_le P W u).trans (Nat.mul_le_mul (pi_readWords_le P W hk) le_rfl)).trans
    (Nat.mul_le_mul le_rfl (card_fedBy_le P W u))

/-- **Step 4.** `b_y |E(A)| ≤ |leaving A|`. -/
theorem bY_mul_card_clear_le {ι : Type} [Fintype ι] [DecidableEq ι] (W : Work C ι) {A : Finset (Fin C.N)}
    (hA : A ⊆ C.computed) : W.bY * (W.clear A).card ≤ (leaving A).card := by
  rcases (W.clear A).eq_empty_or_nonempty with h | h
  · rw [h, card_empty, Nat.cast_zero, mul_zero]
    exact Nat.cast_nonneg _
  · have hfed : A ∈ W.fed := mem_filter.2 ⟨mem_univ _, hA, h⟩
    have hne : W.fed.Nonempty := ⟨_, hfed⟩
    have hc : (0 : ℝ) < (W.clear A).card := Nat.cast_pos.2 (card_pos.2 h)
    have hb : W.bY ≤ (kappa C A ((W.clear A).biUnion W.y) : ℝ) / (W.clear A).card := by
      rw [Work.bY, dite_eq_left hne]
      exact inf'_le _ hfed
    calc W.bY * (W.clear A).card
        ≤ (kappa C A ((W.clear A).biUnion W.y) : ℝ) / (W.clear A).card * (W.clear A).card :=
          mul_le_mul_of_nonneg_right hb hc.le
      _ = ((kappa C A ((W.clear A).biUnion W.y) : ℕ) : ℝ) := div_mul_cancel₀ _ hc.ne'
      _ ≤ (leaving A).card := by
          have h := kappa_le
            (meets_exits ((disjoint_biUnion_left (W.clear A) W.y A).2 fun o ho => (mem_filter.1 ho).2.2))
          exact_mod_cast h

/-- Each gate leaving a block is an entry of the block reading it. -/
theorem sum_card_leaving_le {d : RegisteredCircuit} (P : Partition d.D m) :
    ∑ u, (leaving (P.gates u)).card ≤ (entries P).card := by
  rw [← card_sigma]
  refine card_le_card_of_forall_subsingleton (fun s e => e.2 = s.2) (fun s hs => ?_) fun e _ s hs t ht => ?_
  · obtain ⟨-, hs'⟩ := mem_sigma.1 hs
    obtain ⟨hv, h, hh, hvh⟩ := mem_filter.1 hs'
    obtain ⟨uh, huh⟩ := exists_unit P (reader_computed hvh)
    exact ⟨⟨uh, s.2⟩, mem_entries P (gate_computed P hv) hvh huh fun hsu =>
      hh (P.mem_gates.2 (huh.trans (hsu.symm.trans (P.mem_gates.1 hv)))), rfl⟩
  · have h2 : s.2 = t.2 := hs.2.symm.trans ht.2
    have hs' : P.unit s.2 = some s.1 := P.mem_gates.1 (mem_filter.1 (mem_sigma.1 hs.1).2).1
    have ht' : P.unit t.2 = some t.1 := P.mem_gates.1 (mem_filter.1 (mem_sigma.1 ht.1).2).1
    rw [h2] at hs'
    exact Sigma.ext (Option.some_injective _ (hs'.symm.trans ht')) (heq_of_eq h2)

theorem bY_pos {ι : Type} [Fintype ι] [DecidableEq ι] (W : Work C ι) (h : W.fed.Nonempty) : 0 < W.bY := by
  rw [Work.bY, dite_eq_left h, lt_inf'_iff]
  intro A hA
  obtain ⟨-, -, o, ho⟩ := mem_filter.1 hA
  obtain ⟨⟨s, rfl, hsA⟩, -⟩ := (mem_filter.1 ho).2
  obtain ⟨g, hg, a, ha, hga⟩ := (W.M s.1).feeds_y s.2
  have hy : a ∈ (W.clear A).biUnion W.y := mem_biUnion.2 ⟨_, ho, ha⟩
  have hk : 0 < kappa C A ((W.clear A).biUnion W.y) := kappa_pos (hsA hg) hy hga
  exact div_pos (Nat.cast_pos.2 hk) (Nat.cast_pos.2 (card_pos.2 ⟨_, ho⟩))

theorem bY_nonneg {ι : Type} [Fintype ι] [DecidableEq ι] (W : Work C ι) : 0 ≤ W.bY := by
  by_cases h : W.fed.Nonempty
  · exact (bY_pos W h).le
  · exact le_of_eq (by rw [Work.bY, dite_eq_right h])

/-- Steps 1 to 4 over a set `G` of blocks, as counts. Each block of `G` reads at most its own `κ u` gates outside `θ`,
and all the blocks together at most `T`. `V` counts the MACs a block of `G` holds, `F` the pairs of a block of `G` and
an output it feeds, and `E` the pairs of a block of `G` and an output in its `E(block)`. A split MAC and a gate leaving
a block are entries of the block reading them, in `G` or not, so both are counted against `T`; the MACs a block outside
`G` holds are counted apart, with no budget on what it reads. -/
theorem counts_on {ι : Type} [Fintype ι] [DecidableEq ι] {d : RegisteredCircuit} {T : ℕ} (W : Work d.D ι)
    (P : Partition d.D m) (G : Finset (Fin m)) (κ : Fin m → ℕ)
    (hκ : ∀ u ∈ G, (d.unregistered (P.gates u)).card ≤ κ u)
    (hT : ∑ u, (d.unregistered (P.gates u)).card ≤ T) :
    ∃ V F E : ℕ, W.N ≤ T + V + ∑ u ∈ Gᶜ, ∑ a, (inBlock P W u a).card ∧
      V ^ 2 ≤ (∑ u ∈ G, piMax d W (κ u)) * F ∧ F ≤ outOn P W G + E ∧ W.bY * E ≤ T ∧ (E = 0 ∨ 0 < W.bY) := by
  have hent : (entries P).card ≤ T := card_entries_le P hT
  have hleave : ∑ u ∈ G, (leaving (P.gates u)).card ≤ T :=
    ((sum_le_sum_of_subset (subset_univ G)).trans (sum_card_leaving_le P)).trans hent
  refine ⟨∑ u ∈ G, ∑ a, (inBlock P W u a).card, ∑ u ∈ G, (fedBy P W u).card,
    ∑ u ∈ G, (W.clear (P.gates u)).card, ?_, ?_, ?_, ?_, ?_⟩
  · calc W.N = (univ : Finset W.Mac).card := card_univ.symm
      _ ≤ ((univ.filter fun s : W.Mac => ∀ u, ¬ W.mac s ⊆ P.gates u) ∪
            univ.biUnion fun u => univ.sigma (inBlock P W u)).card := by
          refine card_le_card fun s _ => ?_
          by_cases h : ∀ u, ¬ W.mac s ⊆ P.gates u
          · exact mem_union_left _ (mem_filter.2 ⟨mem_univ _, h⟩)
          · obtain ⟨u, hu⟩ := not_forall.1 h
            exact mem_union_right _ (mem_biUnion.2 ⟨u, mem_univ _,
              mem_sigma.2 ⟨mem_univ _, mem_filter.2 ⟨mem_univ _, not_not.1 hu⟩⟩⟩)
      _ ≤ (entries P).card + ∑ u, (univ.sigma (inBlock P W u)).card :=
          (card_union_le _ _).trans (add_le_add (card_split_le P W) card_biUnion_le)
      _ ≤ T + ∑ u, ∑ a, (inBlock P W u a).card :=
          add_le_add hent (le_of_eq (sum_congr rfl fun u _ => card_sigma _ _))
      _ = T + ∑ u ∈ G, ∑ a, (inBlock P W u a).card + ∑ u ∈ Gᶜ, ∑ a, (inBlock P W u a).card := by
          rw [add_assoc, sum_add_sum_compl]
  · exact sum_sq_le_sum_mul_sum_of_sq_le_mul _ (fun _ _ => Nat.zero_le _) (fun _ _ => Nat.zero_le _) fun u hu =>
      (card_inBlock_sq_le P W u).trans (Nat.mul_le_mul (pi_readWords_le P W (hκ u hu)) le_rfl)
  · calc ∑ u ∈ G, (fedBy P W u).card
        ≤ ∑ u ∈ G, ((hit P W u).card + (W.clear (P.gates u)).card) := sum_le_sum fun u _ => card_fedBy_le P W u
      _ = outOn P W G + ∑ u ∈ G, (W.clear (P.gates u)).card := sum_add_distrib
  · calc W.bY * ((∑ u ∈ G, (W.clear (P.gates u)).card : ℕ) : ℝ)
        = ∑ u ∈ G, W.bY * (W.clear (P.gates u)).card := by rw [Nat.cast_sum, mul_sum]
      _ ≤ ∑ u ∈ G, ((leaving (P.gates u)).card : ℝ) :=
          sum_le_sum fun u _ => bY_mul_card_clear_le W (P.gates_subset_computed u)
      _ ≤ T := by exact_mod_cast hleave
  · by_cases h : W.fed.Nonempty
    · exact Or.inr (bY_pos W h)
    · refine Or.inl (sum_eq_zero fun u _ => ?_)
      by_contra hu
      exact h ⟨P.gates u, mem_filter.2 ⟨mem_univ _, P.gates_subset_computed u, card_pos.1 (Nat.pos_of_ne_zero hu)⟩⟩

/-- Steps 1 to 4, as counts: `V` unsplit MACs, `F` pairs of a block and an output it feeds, and `E` pairs of a block
and an output in its `E(block)`. `counts_on` at every block, each under the one budget `k`, `m k` entries in all. -/
theorem counts {ι : Type} [Fintype ι] [DecidableEq ι] {d : RegisteredCircuit} {k : ℕ} (W : Work d.D ι)
    (P : Partition d.D m) (hk : ReadsAtMost d P k) :
    ∃ V F E : ℕ, W.N ≤ m * k + V ∧ V ^ 2 ≤ m * piMax d W k * F ∧ F ≤ W.O + E ∧ W.bY * E ≤ m * k ∧
      (E = 0 ∨ 0 < W.bY) := by
  have hT : ∑ u, (d.unregistered (P.gates u)).card ≤ m * k :=
    calc ∑ u, (d.unregistered (P.gates u)).card ≤ ∑ _u : Fin m, k := sum_le_sum fun u _ => hk u
      _ = m * k := by rw [sum_const, card_univ, Fintype.card_fin, smul_eq_mul]
  obtain ⟨V, F, E, hN, hV, hF, hbE, hb⟩ := counts_on W P univ (fun _ => k) (fun u _ => hk u) hT
  rw [compl_univ, sum_empty, add_zero] at hN
  simp only [sum_const, card_univ, Fintype.card_fin, smul_eq_mul] at hV
  exact ⟨V, F, E, hN, hV, hF.trans (Nat.add_le_add_right (outOn_le P W univ) E), (by exact_mod_cast hbE), hb⟩

/-! ### The bound and the slowdown -/

theorem sqrt_add_le {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) : √(a + b) ≤ √a + √b := by
  rw [Real.sqrt_le_left (add_nonneg (Real.sqrt_nonneg a) (Real.sqrt_nonneg b))]
  nlinarith [Real.sq_sqrt ha, Real.sq_sqrt hb, mul_nonneg (Real.sqrt_nonneg a) (Real.sqrt_nonneg b)]

/-- **Step 5.** The counts give the bound: `g` blocks under the budget, `T` entries in all. -/
theorem real_step_on {N V g T P O E b : ℝ} (hg : 0 ≤ g) (hP : 0 ≤ P) (hO : 0 ≤ O) (hE : 0 ≤ E) (hN : N ≤ T + V)
    (hV : V ^ 2 ≤ g * P * (O + E)) (hbE : b * E ≤ T) (hb : E = 0 ∨ 0 < b) :
    N ≤ T + √(P * g * O) + √(P * g * (T / b)) := by
  have hgP : 0 ≤ g * P := mul_nonneg hg hP
  have h1 : V ≤ √(g * P * O) + √(g * P * E) :=
    calc V ≤ √(g * P * (O + E)) := Real.le_sqrt_of_sq_le hV
      _ = √(g * P * O + g * P * E) := by rw [mul_add]
      _ ≤ √(g * P * O) + √(g * P * E) := sqrt_add_le (mul_nonneg hgP hO) (mul_nonneg hgP hE)
  have h2 : √(g * P * E) ≤ √(P * g * (T / b)) := by
    rcases hb with rfl | hb
    · rw [mul_zero, Real.sqrt_zero]
      exact Real.sqrt_nonneg _
    · have hE' : E ≤ T / b := (le_div_iff₀ hb).2 ((mul_comm E b).trans_le hbE)
      refine Real.sqrt_le_sqrt ?_
      calc g * P * E ≤ g * P * (T / b) := mul_le_mul_of_nonneg_left hE' hgP
        _ = P * g * (T / b) := by ring
  have h3 : √(g * P * O) = √(P * g * O) := by rw [mul_comm g P]
  linarith

/-- **Step 5** at every block: `m` blocks under the budget `k`, `m k` entries in all. -/
theorem real_step {N V m k P O E b : ℝ} (hm : 0 ≤ m) (hP : 0 ≤ P) (hO : 0 ≤ O) (hE : 0 ≤ E)
    (hN : N ≤ m * k + V) (hV : V ^ 2 ≤ m * P * (O + E)) (hbE : b * E ≤ m * k) (hb : E = 0 ∨ 0 < b) :
    N ≤ m * (k + √(k * P / b)) + √(P * m * O) := by
  have h := real_step_on hm hP hO hE hN hV hbE hb
  rw [show P * m * (m * k / b) = m ^ 2 * (k * P / b) by ring, Real.sqrt_mul (sq_nonneg m), Real.sqrt_sq hm] at h
  linarith

theorem Bound : Guarantees.Bound := by
  intro ι _ _ d W k m P hk
  obtain ⟨V, F, E, hN, hV, hF, hbE, hb⟩ := counts W P hk
  have hV' : (V : ℝ) ^ 2 ≤ m * piMax d W k * (W.O + E) := by
    have h : V ^ 2 ≤ m * piMax d W k * (W.O + E) := hV.trans (Nat.mul_le_mul le_rfl hF)
    exact_mod_cast h
  have hN' : (W.N : ℝ) ≤ m * k + V := by exact_mod_cast hN
  have hE' : (E : ℝ) = 0 ∨ 0 < W.bY := hb.imp_left fun h => by rw [h, Nat.cast_zero]
  unfold alpha
  exact real_step (Nat.cast_nonneg m) (Nat.cast_nonneg _) (Nat.cast_nonneg _) (Nat.cast_nonneg E) hN' hV' hbE hE'

/-- **The bound over a set of blocks, each under its own budget.** Block `u` of `G` reads at most `κ u` gates outside
`θ` and all the blocks together, those outside `G` too, at most `T`. With `S = ∑_{u ∈ G} Π(κ u)`: `N` is at most the
MACs the blocks outside `G` hold, plus `T + √(S O_G) + √(S T / b_y)`, with `O_G` the pairs of a block of `G` and an
output with a wire in it (`outOn`), at most `O`. -/
theorem bound_on_budgets {ι : Type} [Fintype ι] [DecidableEq ι] {d : RegisteredCircuit} {T : ℕ} (W : Work d.D ι)
    (P : Partition d.D m) (G : Finset (Fin m)) (κ : Fin m → ℕ)
    (hκ : ∀ u ∈ G, (d.unregistered (P.gates u)).card ≤ κ u)
    (hT : ∑ u, (d.unregistered (P.gates u)).card ≤ T) :
    (W.N : ℝ) ≤ (∑ u ∈ Gᶜ, ∑ a, (inBlock P W u a).card : ℕ) + T +
      √((∑ u ∈ G, piMax d W (κ u) : ℕ) * outOn P W G) + √((∑ u ∈ G, piMax d W (κ u) : ℕ) * (T / W.bY)) := by
  obtain ⟨V, F, E, hN, hV, hF, hbE, hb⟩ := counts_on W P G κ hκ hT
  have hV' : (V : ℝ) ^ 2 ≤ 1 * ((∑ u ∈ G, piMax d W (κ u) : ℕ) : ℝ) * (outOn P W G + E) := by
    have h : V ^ 2 ≤ (∑ u ∈ G, piMax d W (κ u)) * (outOn P W G + E) := hV.trans (Nat.mul_le_mul le_rfl hF)
    rw [one_mul]
    exact_mod_cast h
  have hN' : (W.N : ℝ) - (∑ u ∈ Gᶜ, ∑ a, (inBlock P W u a).card : ℕ) ≤ T + V := by
    have h : (W.N : ℝ) ≤ ((T + V + ∑ u ∈ Gᶜ, ∑ a, (inBlock P W u a).card : ℕ) : ℝ) := by exact_mod_cast hN
    rw [Nat.cast_add, Nat.cast_add] at h
    linarith
  have hE' : (E : ℝ) = 0 ∨ 0 < W.bY := hb.imp_left fun h => by rw [h, Nat.cast_zero]
  have h := real_step_on (g := 1) zero_le_one (Nat.cast_nonneg (∑ u ∈ G, piMax d W (κ u)))
    (Nat.cast_nonneg (outOn P W G)) (Nat.cast_nonneg E) hN' hV' hbE hE'
  rw [mul_one] at h
  linarith

/-- **The bound over a set of blocks.** Each block of `G` reads at most `k` gates outside `θ` and all the blocks
together at most `T`: `N` is at most the MACs the blocks outside `G` hold, plus
`T + √(Π(k) |G| O_G) + √(Π(k) |G| T / b_y)`, with `O_G` the pairs of a block of `G` and an output with a wire in it
(`outOn`), at most `O`. At every block and `T = m k` it gives `Bound`. -/
theorem bound_on {ι : Type} [Fintype ι] [DecidableEq ι] {d : RegisteredCircuit} {k T : ℕ} (W : Work d.D ι)
    (P : Partition d.D m) (G : Finset (Fin m)) (hk : ∀ u ∈ G, (d.unregistered (P.gates u)).card ≤ k)
    (hT : ∑ u, (d.unregistered (P.gates u)).card ≤ T) :
    (W.N : ℝ) ≤ (∑ u ∈ Gᶜ, ∑ a, (inBlock P W u a).card : ℕ) + T + √(piMax d W k * G.card * outOn P W G) +
      √(piMax d W k * G.card * (T / W.bY)) := by
  have h : (W.N : ℝ) ≤ (∑ u ∈ Gᶜ, ∑ a, (inBlock P W u a).card : ℕ) + T +
      √((∑ _u ∈ G, piMax d W k : ℕ) * outOn P W G) + √((∑ _u ∈ G, piMax d W k : ℕ) * (T / W.bY)) :=
    bound_on_budgets W P G (fun _ => k) hk hT
  have hs : ((∑ _u ∈ G, piMax d W k : ℕ) : ℝ) = piMax d W k * G.card := by
    rw [sum_const, smul_eq_mul, Nat.cast_mul]
    exact mul_comm _ _
  rw [hs] at h
  exact h

theorem alpha_nonneg {ι : Type} [Fintype ι] [DecidableEq ι] (d : RegisteredCircuit) (W : Work d.D ι) (k : ℕ) :
    0 ≤ alpha d W k :=
  Real.sqrt_nonneg _

/-- `N ≤ m L + √(P m O)` gives `m ≥ N / L - √(P O N) / L^{3/2}`: when `m < N / L`, `√(P m O) ≤ √(P O N / L)`. -/
theorem slow_step {N m L P O : ℝ} (hm : 0 ≤ m) (hL : 0 ≤ L) (hP : 0 ≤ P) (hO : 0 ≤ O)
    (h : N ≤ m * L + √(P * m * O)) : N / L - √(P * O * N) / L ^ (3 / 2 : ℝ) ≤ m := by
  rcases hL.eq_or_lt with hL0 | hL0
  · rw [← hL0, div_zero, Real.zero_rpow (show (3 / 2 : ℝ) ≠ 0 by norm_num), div_zero, sub_zero]
    exact hm
  · have hs : 0 < √L := Real.sqrt_pos.2 hL0
    have hpow : L ^ (3 / 2 : ℝ) = L * √L := by
      rw [show (3 / 2 : ℝ) = 1 + 1 / 2 by norm_num, Real.rpow_add hL0, Real.rpow_one, ← Real.sqrt_eq_rpow]
    rw [hpow]
    rcases le_total (N / L) m with hNm | hmN
    · have : 0 ≤ √(P * O * N) / (L * √L) := div_nonneg (Real.sqrt_nonneg _) (mul_pos hL0 hs).le
      linarith
    · have hmL : m * L ≤ N := (le_div_iff₀ hL0).1 hmN
      have h1 : √(P * m * O) ≤ √(P * O * N) / √L := by
        rw [← Real.sqrt_div' _ hL0.le]
        refine Real.sqrt_le_sqrt ?_
        rw [le_div_iff₀ hL0]
        calc P * m * O * L = P * O * (m * L) := by ring
          _ ≤ P * O * N := mul_le_mul_of_nonneg_left hmL (mul_nonneg hP hO)
      have h2 : N ≤ m * L + √(P * O * N) / √L := by linarith
      have h3 : N * √L ≤ m * L * √L + √(P * O * N) := by
        have := mul_le_mul_of_nonneg_right h2 hs.le
        rwa [add_mul, div_mul_cancel₀ _ hs.ne'] at this
      have h4 : N / L - √(P * O * N) / (L * √L) = (N * √L - √(P * O * N)) / (L * √L) := by
        rw [sub_div, mul_div_mul_right _ _ hs.ne']
      rw [h4, div_le_iff₀ (mul_pos hL0 hs)]
      linarith

/-- A bound `N ≤ m L + √(P m O)` gives `(c / n) (N / L - √(P O N) / L^{3/2}) ≤ m c / n`. -/
theorem slowdown_of_bound {N m L P O c n : ℝ} (hm : 0 ≤ m) (hL : 0 ≤ L) (hP : 0 ≤ P) (hO : 0 ≤ O) (hc : 0 ≤ c)
    (hn : 0 ≤ n) (h : N ≤ m * L + √(P * m * O)) : c / n * (N / L - √(P * O * N) / L ^ (3 / 2 : ℝ)) ≤ m * c / n :=
  (mul_le_mul_of_nonneg_left (slow_step hm hL hP hO h) (div_nonneg hc hn)).trans_eq (by ring)

theorem Slowdown : Guarantees.Slowdown := by
  intro ι _ _ d W k m c P hk
  exact slowdown_of_bound (Nat.cast_nonneg m) (add_nonneg (Nat.cast_nonneg k) (alpha_nonneg d W k))
    (Nat.cast_nonneg _) (Nat.cast_nonneg _) (Nat.cast_nonneg c) (Nat.cast_nonneg _) (Bound ι d W k m P hk)

/-- `Bound` with upper bounds on `α` and `Π(k)` in their places. -/
theorem bound_of_le {ι : Type} [Fintype ι] [DecidableEq ι] {d : RegisteredCircuit} {W : Work d.D ι} {k m : ℕ}
    {P : Partition d.D m} (hk : ReadsAtMost d P k) {a p : ℝ} (ha : alpha d W k ≤ a)
    (hp : (piMax d W k : ℝ) ≤ p) : (W.N : ℝ) ≤ m * (k + a) + √(p * m * W.O) := by
  have h1 : (m : ℝ) * (k + alpha d W k) ≤ m * (k + a) := mul_le_mul_of_nonneg_left (by linarith) (Nat.cast_nonneg m)
  have h2 : √(piMax d W k * m * W.O) ≤ √(p * m * W.O) := Real.sqrt_le_sqrt
    (mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_right hp (Nat.cast_nonneg m)) (Nat.cast_nonneg _))
  linarith [Bound ι d W k m P hk]

/-! ### Bounds on `Π(k)` -/

theorem piMax_le {ι : Type} [Fintype ι] {d : RegisteredCircuit} (W : Work d.D ι) {k : ℕ} {B : ℝ} (hB : 0 ≤ B)
    (h : ∀ S, kappaWords d S ≤ k → (W.pi S : ℝ) ≤ B) : (piMax d W k : ℝ) ≤ B := by
  unfold piMax
  refine sup_induction (p := fun n : ℕ => (n : ℝ) ≤ B) ?_ ?_ ?_
  · show ((⊥ : ℕ) : ℝ) ≤ B
    rw [show (⊥ : ℕ) = 0 from rfl, Nat.cast_zero]
    exact hB
  · intro a ha b hb
    show ((a ⊔ b : ℕ) : ℝ) ≤ B
    rcases le_total a b with hab | hab
    · rw [sup_eq_right.2 hab]
      exact hb
    · rw [sup_eq_left.2 hab]
      exact ha
  · intro S hS
    exact h S (mem_filter.1 hS).2

/-- When `val` is one-to-one on each matmul's `w`-words and on its `x`-words, and no two matmuls share both a `w`- and
an `x`-value, `π(S) ≤ |val(S ∩ W)| |val(S ∩ X)|`. -/
theorem pi_le_card_mul {ι V : Type} [Fintype ι] [DecidableEq V] (W : Work C ι) (val : Finset (Fin C.N) → V)
    (hval : ValuesInjective W val) (h : NoSharedPair W val) (S : Finset (Finset (Fin C.N))) :
    W.pi S ≤ ((S ∩ W.allW).image val).card * ((S ∩ W.allX).image val).card := by
  rw [← card_product]
  calc W.pi S = (univ.sigma fun a => (S ∩ W.wWords a) ×ˢ (S ∩ W.xWords a)).card := by
        rw [Work.pi, card_sigma]
        exact sum_congr rfl fun a _ => (card_product _ _).symm
    _ ≤ (((S ∩ W.allW).image val) ×ˢ ((S ∩ W.allX).image val)).card := by
        refine card_le_card_of_injOn (fun e => (val e.2.1, val e.2.2)) (fun e he => ?_) fun e he e' he' hee => ?_
        · obtain ⟨-, he2⟩ := mem_sigma.1 (mem_coe.1 he)
          obtain ⟨hw, hx⟩ := mem_product.1 he2
          have hW : e.2.1 ∈ S ∩ W.allW :=
            mem_inter.2 ⟨(mem_inter.1 hw).1, mem_biUnion.2 ⟨e.1, mem_univ _, (mem_inter.1 hw).2⟩⟩
          have hX : e.2.2 ∈ S ∩ W.allX :=
            mem_inter.2 ⟨(mem_inter.1 hx).1, mem_biUnion.2 ⟨e.1, mem_univ _, (mem_inter.1 hx).2⟩⟩
          exact mem_coe.2 (mem_product.2 ⟨mem_image_of_mem val hW, mem_image_of_mem val hX⟩)
        · obtain ⟨-, he2⟩ := mem_sigma.1 (mem_coe.1 he)
          obtain ⟨-, he2'⟩ := mem_sigma.1 (mem_coe.1 he')
          obtain ⟨hw, hx⟩ := mem_product.1 he2
          obtain ⟨hw', hx'⟩ := mem_product.1 he2'
          have hv : (val e.2.1, val e.2.2) = (val e'.2.1, val e'.2.2) := hee
          obtain ⟨hv1, hv2⟩ := Prod.mk.inj hv
          have h1 : e.1 = e'.1 := by
            by_contra hne
            rcases h e.1 e'.1 hne with hd | hd
            · refine disjoint_left.1 hd (mem_image_of_mem val (mem_inter.1 hw).2) ?_
              rw [hv1]
              exact mem_image_of_mem val (mem_inter.1 hw').2
            · refine disjoint_left.1 hd (mem_image_of_mem val (mem_inter.1 hx).2) ?_
              rw [hv2]
              exact mem_image_of_mem val (mem_inter.1 hx').2
          have hw'' : e'.2.1 ∈ W.wWords e.1 := by
            rw [h1]
            exact (mem_inter.1 hw').2
          have hx'' : e'.2.2 ∈ W.xWords e.1 := by
            rw [h1]
            exact (mem_inter.1 hx').2
          exact Sigma.ext h1 (heq_of_eq (Prod.ext ((hval e.1).1 (mem_coe.2 (mem_inter.1 hw).2) (mem_coe.2 hw'') hv1)
            ((hval e.1).2 (mem_coe.2 (mem_inter.1 hx).2) (mem_coe.2 hx'') hv2)))

/-- With `val` one-to-one on `T ⊆ A`, `|S ∩ T| ≤ |val(S ∩ A)|`. -/
theorem card_inter_le_image {V : Type} [DecidableEq V] {val : Finset (Fin C.N) → V}
    {T A : Finset (Finset (Fin C.N))} (hval : Set.InjOn val T) (hTA : T ⊆ A) (S : Finset (Finset (Fin C.N))) :
    (S ∩ T).card ≤ ((S ∩ A).image val).card :=
  (card_image_of_injOn (hval.mono (coe_subset.2 inter_subset_right))).symm.le.trans
    (card_le_card (image_subset_image (inter_subset_inter_left hTA)))

theorem amgm_bound {p q βw βx κ : ℝ} (hw : 0 < βw) (hx : 0 < βx) (hp : 0 ≤ p) (hq : 0 ≤ q)
    (h : βw * p + βx * q ≤ κ) : p * q ≤ κ ^ 2 / (4 * βw * βx) := by
  have hs : 0 ≤ βw * p + βx * q := add_nonneg (mul_nonneg hw.le hp) (mul_nonneg hx.le hq)
  rw [le_div_iff₀ (mul_pos (mul_pos zero_lt_four hw) hx)]
  nlinarith [sq_nonneg (βw * p - βx * q), mul_nonneg (sub_nonneg.2 h) (add_nonneg (hs.trans h) hs)]

theorem sqrt_quarter {k βw βx b : ℝ} : √(k * (k ^ 2 / (4 * βw * βx)) / b) = √(k ^ 3 / (βw * βx * b)) / 2 := by
  rw [show k * (k ^ 2 / (4 * βw * βx)) / b = k ^ 3 / (βw * βx * b) / 4 by ring,
    Real.sqrt_div' _ (by norm_num : (0 : ℝ) ≤ 4), show (4 : ℝ) = 2 ^ 2 by norm_num,
    Real.sqrt_sq (by norm_num : (0 : ℝ) ≤ 2)]

theorem alpha_bound {ι : Type} [Fintype ι] [DecidableEq ι] {d : RegisteredCircuit} (W : Work d.D ι) {k : ℕ}
    {βw βx : ℝ} (hpi : (piMax d W k : ℝ) ≤ k ^ 2 / (4 * βw * βx)) :
    alpha d W k ≤ √(k ^ 3 / (βw * βx * W.bY)) / 2 := by
  unfold alpha
  exact (Real.sqrt_le_sqrt (div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left hpi (Nat.cast_nonneg k))
    (bY_nonneg W))).trans_eq sqrt_quarter

/-- **One pair of arrays.** With `val` one-to-one on each matmul's words (`ValuesInjective`), no two matmuls sharing
both a `w`- and an `x`-value (`NoSharedPair`), and (H) at the arrays `W` and `X`, a value of each costing `β_w` and
`β_x`: `Π(k) ≤ k² / (4 β_w β_x)` and `α ≤ ½ √(k³ / (β_w β_x b_y))`. -/
theorem onePremise {ι V : Type} [Fintype ι] [DecidableEq ι] [DecidableEq V] {d : RegisteredCircuit} {W : Work d.D ι}
    {val : Finset (Fin d.D.N) → V} {k : ℕ} {βw βx : ℝ} (hw : 0 < βw) (hx : 0 < βx) (hval : ValuesInjective W val)
    (hshare : NoSharedPair W val) (hH : Assumptions.Hardness d ![W.allW, W.allX] val ![βw, βx] k) :
    (piMax d W k : ℝ) ≤ k ^ 2 / (4 * βw * βx) ∧ alpha d W k ≤ √(k ^ 3 / (βw * βx * W.bY)) / 2 := by
  have hpi : (piMax d W k : ℝ) ≤ k ^ 2 / (4 * βw * βx) := by
    refine piMax_le W (div_nonneg (sq_nonneg _) (mul_pos (mul_pos zero_lt_four hw) hx).le) fun S hS => ?_
    have h1 : (W.pi S : ℝ) ≤ ((S ∩ W.allW).image val).card * ((S ∩ W.allX).image val).card := by
      exact_mod_cast pi_le_card_mul W val hval hshare S
    have h2 : βw * ((S ∩ W.allW).image val).card + βx * ((S ∩ W.allX).image val).card ≤ kappaWords d S := by
      simpa only [Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one] using hH S hS
    exact h1.trans (amgm_bound hw hx (Nat.cast_nonneg _) (Nat.cast_nonneg _) (h2.trans (Nat.cast_le.2 hS)))
  exact ⟨hpi, alpha_bound W hpi⟩

/-- The blow-up of `G`: each vertex `r` replaced by `z r` copies, copies adjacent when their vertices are. -/
def blowUp {ρ : Type} (G : SimpleGraph ρ) (z : ρ → ℕ) : SimpleGraph (Σ r, Fin (z r)) := G.comap Sigma.fst

/-- **The weighted Turán inequality.** For a graph with no clique of `ω + 1`, weights `z` and distinct edges `e a`,
`2 ω ∑_a z(e_a) z(e'_a) ≤ (ω - 1) (∑ z)²`: Turán's theorem on the blow-up, which has no clique of `ω + 1` either. -/
theorem weighted_turan {ι ρ : Type} [Fintype ι] [Fintype ρ] {G : SimpleGraph ρ} {ω : ℕ} (hG : G.CliqueFree (ω + 1))
    (e : ι → ρ × ρ) (he : ∀ a, G.Adj (e a).1 (e a).2) (hinj : ∀ a b, s((e a).1, (e a).2) = s((e b).1, (e b).2) → a = b)
    (z : ρ → ℕ) : 2 * ω * ∑ a, z (e a).1 * z (e a).2 ≤ (ω - 1) * (∑ r, z r) ^ 2 := by
  classical
  have hH : (blowUp G z).CliqueFree (ω + 1) := by
    intro t ht
    have hinjOn : Set.InjOn Sigma.fst (t : Set (Σ r, Fin (z r))) := by
      intro x hx y hy hxy
      by_contra hne
      have hadj : G.Adj x.1 y.1 := ht.isClique hx hy hne
      exact hadj.ne hxy
    refine hG (t.image Sigma.fst) ⟨?_, ?_⟩
    · intro r hr r' hr' hne
      obtain ⟨x, hx, rfl⟩ := mem_image.1 (mem_coe.1 hr)
      obtain ⟨y, hy, rfl⟩ := mem_image.1 (mem_coe.1 hr')
      exact ht.isClique (mem_coe.2 hx) (mem_coe.2 hy) fun hxy => hne (congrArg Sigma.fst hxy)
    · rw [card_image_of_injOn hinjOn, ht.card_eq]
  refine le_trans (Nat.mul_le_mul le_rfl (le_trans ?_ hH.card_edgeFinset_le)) ?_
  · rw [show ∑ a, z (e a).1 * z (e a).2 = (univ : Finset (Σ a : ι, Fin (z (e a).1) × Fin (z (e a).2))).card by
      simp only [card_univ, Fintype.card_sigma, Fintype.card_prod, Fintype.card_fin]]
    refine card_le_card_of_injOn (fun x => s((⟨(e x.1).1, x.2.1⟩ : Σ r, Fin (z r)), ⟨(e x.1).2, x.2.2⟩))
      (fun x _ => ?_) ?_
    · exact mem_coe.2 (SimpleGraph.mem_edgeFinset.2 ((SimpleGraph.mem_edgeSet _).2 (he x.1)))
    · rintro ⟨a, i, j⟩ - ⟨b, i', j'⟩ - hxy
      simp only [Sym2.eq_iff, Sigma.mk.injEq] at hxy
      rcases hxy with ⟨⟨h1, h1'⟩, h2, h2'⟩ | ⟨⟨h1, -⟩, h2, -⟩
      · have hab := hinj a b (by rw [h1, h2])
        subst hab
        obtain rfl := eq_of_heq h1'
        obtain rfl := eq_of_heq h2'
        rfl
      · have hab := hinj a b (by rw [h1, h2]; exact Sym2.eq_swap)
        subst hab
        exact absurd h1 (he _).ne
  · exact SimpleGraph.mul_turanNumber_le.trans_eq (by rw [Fintype.card_sigma]; simp only [Fintype.card_fin])

open Filter Topology in
/-- **The weighted Turán inequality, real weights.** The same for weights `u ≥ 0`: the limit of `weighted_turan` at
`⌊n u⌋`, divided by `n²`. -/
theorem weighted_turan_real {ι ρ : Type} [Fintype ι] [Fintype ρ] {G : SimpleGraph ρ} {ω : ℕ} (hω : 0 < ω)
    (hG : G.CliqueFree (ω + 1)) (e : ι → ρ × ρ) (he : ∀ a, G.Adj (e a).1 (e a).2)
    (hinj : ∀ a b, s((e a).1, (e a).2) = s((e b).1, (e b).2) → a = b) (u : ρ → ℝ) (hu : ∀ r, 0 ≤ u r) :
    2 * ω * ∑ a, u (e a).1 * u (e a).2 ≤ (ω - 1) * (∑ r, u r) ^ 2 := by
  have hω' : 1 ≤ ω := hω
  have h1 : ((ω - 1 : ℕ) : ℝ) = ω - 1 := by rw [Nat.cast_sub hω', Nat.cast_one]
  have hle : ∀ n : ℕ, 2 * ω * ∑ a, (⌊u (e a).1 * n⌋₊ / n : ℝ) * (⌊u (e a).2 * n⌋₊ / n) ≤
      (ω - 1) * (∑ r, (⌊u r * n⌋₊ / n : ℝ)) ^ 2 := by
    intro n
    have hz : 2 * ω * ∑ a, (⌊u (e a).1 * n⌋₊ : ℝ) * ⌊u (e a).2 * n⌋₊ ≤ (ω - 1) * (∑ r, (⌊u r * n⌋₊ : ℝ)) ^ 2 := by
      rw [← h1]
      exact_mod_cast weighted_turan hG e he hinj fun r => ⌊u r * n⌋₊
    have s1 : ∑ a, (⌊u (e a).1 * n⌋₊ / n : ℝ) * (⌊u (e a).2 * n⌋₊ / n) =
        (∑ a, (⌊u (e a).1 * n⌋₊ : ℝ) * ⌊u (e a).2 * n⌋₊) / n ^ 2 := by
      rw [sum_div]
      exact sum_congr rfl fun a _ => by ring
    have s2 : ∑ r, (⌊u r * n⌋₊ / n : ℝ) = (∑ r, (⌊u r * n⌋₊ : ℝ)) / n := by rw [sum_div]
    rw [s1, s2, div_pow, ← mul_div_assoc, ← mul_div_assoc]
    exact div_le_div_of_nonneg_right hz (sq_nonneg _)
  have hv : ∀ r, Tendsto (fun n : ℕ => (⌊u r * n⌋₊ / n : ℝ)) atTop (𝓝 (u r)) := fun r =>
    (tendsto_nat_floor_mul_div_atTop (hu r)).comp tendsto_natCast_atTop_atTop
  exact le_of_tendsto_of_tendsto'
    ((tendsto_finsetSum univ fun a _ => (hv (e a).1).mul (hv (e a).2)).const_mul (2 * (ω : ℝ)))
    (((tendsto_finsetSum univ fun r _ => hv r).pow 2).const_mul ((ω : ℝ) - 1)) hle

theorem clique_real {t T U σ k q ω : ℝ} (hq : 0 < q) (hω : 1 ≤ ω) (hσ : 0 ≤ σ) (ht : t ≤ T) (hqT : q * T ≤ U)
    (hU : 2 * ω * U ≤ (ω - 1) * σ ^ 2) (hk : σ ≤ k) : t ≤ 2 * (1 - 1 / ω) * (k ^ 2 / (4 * q)) := by
  have hω0 : 0 < ω := by linarith
  have e1 : 1 - 1 / ω = (ω - 1) / ω := by rw [sub_div, div_self hω0.ne']
  have e2 : 2 * ((ω - 1) / ω) * (k ^ 2 / (4 * q)) = (ω - 1) * k ^ 2 / (2 * ω * q) := by ring
  have h2ωq : 0 ≤ 2 * ω * q := (mul_pos (mul_pos two_pos hω0) hq).le
  rw [e1, e2, le_div_iff₀ (mul_pos (mul_pos two_pos hω0) hq)]
  have h3 : 0 ≤ ω - 1 := by linarith
  nlinarith [mul_le_mul_of_nonneg_left (mul_self_le_mul_self hσ hk) h3, mul_le_mul_of_nonneg_left ht h2ωq,
    mul_le_mul_of_nonneg_left hqT (mul_pos two_pos hω0).le]

/-- **Shared arrays.** Matmul `a` reads arrays `(e a).1` and `(e a).2`, an edge of a graph with no clique of `ω + 1`
(`Arrays`), `val` is one-to-one on each matmul's words (`ValuesInjective`), array `r` costs `β_r ≥ 0` a value and each
matmul's pair at least `q > 0` (`CostsAtLeast`). (H) at those arrays gives `Π(k) ≤ 2 (1 - 1/ω) k² / (4 q)`. -/
theorem sharedArraysClique {ι ρ V : Type} [Fintype ι] [DecidableEq ι] [Fintype ρ] [DecidableEq V]
    {d : RegisteredCircuit} {W : Work d.D ι} {val : Finset (Fin d.D.N) → V} {k ω : ℕ} {G : SimpleGraph ρ}
    {A : ρ → Finset (Finset (Fin d.D.N))} {e : ι → ρ × ρ} {β : ρ → ℝ} {q : ℝ} (hω : 0 < ω) (hβ : ∀ r, 0 ≤ β r)
    (hq : 0 < q) (hA : Arrays W G A e ω) (hval : ValuesInjective W val) (hqe : CostsAtLeast e β q)
    (hprem : Assumptions.Hardness d A val β k) : (piMax d W k : ℝ) ≤ 2 * (1 - 1 / ω) * (k ^ 2 / (4 * q)) := by
  obtain ⟨hG, hadj, hinj, hsub⟩ := hA
  have hω' : 1 ≤ ω := hω
  have hω1 : (1 : ℝ) ≤ ω := Nat.one_le_cast.2 hω'
  have hB : (0 : ℝ) ≤ 2 * (1 - 1 / ω) * (k ^ 2 / (4 * q)) := by
    have h1 : (0 : ℝ) ≤ 1 - 1 / ω := by
      rw [sub_nonneg, div_le_iff₀ (by linarith : (0 : ℝ) < ω)]
      linarith
    exact mul_nonneg (mul_nonneg zero_le_two h1) (div_nonneg (sq_nonneg _) (mul_pos four_pos hq).le)
  refine piMax_le W hB fun S hS => ?_
  have key : ∀ n : ρ → ℕ, (∀ a, (S ∩ W.wWords a).card ≤ n (e a).1 ∧ (S ∩ W.xWords a).card ≤ n (e a).2) →
      ∑ r, β r * (n r : ℝ) ≤ k → (W.pi S : ℝ) ≤ 2 * (1 - 1 / ω) * (k ^ 2 / (4 * q)) := by
    intro n hn hk
    have hpi : W.pi S ≤ ∑ a, n (e a).1 * n (e a).2 :=
      sum_le_sum fun a _ => Nat.mul_le_mul (hn a).1 (hn a).2
    have hpi' : (W.pi S : ℝ) ≤ ∑ a, (n (e a).1 : ℝ) * n (e a).2 := by
      exact_mod_cast hpi
    have hu : ∀ r, 0 ≤ β r * n r := fun r => mul_nonneg (hβ r) (Nat.cast_nonneg _)
    have hT : 2 * (ω : ℝ) * ∑ a, β (e a).1 * n (e a).1 * (β (e a).2 * n (e a).2) ≤
        (ω - 1) * (∑ r, β r * n r) ^ 2 :=
      weighted_turan_real hω hG e hadj hinj (fun r => β r * n r) hu
    have hqT : q * ∑ a, (n (e a).1 : ℝ) * n (e a).2 ≤ ∑ a, β (e a).1 * n (e a).1 * (β (e a).2 * n (e a).2) := by
      rw [mul_sum]
      refine sum_le_sum fun a _ => ?_
      calc q * ((n (e a).1 : ℝ) * n (e a).2) ≤ β (e a).1 * β (e a).2 * ((n (e a).1 : ℝ) * n (e a).2) :=
            mul_le_mul_of_nonneg_right (hqe a) (mul_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _))
        _ = β (e a).1 * n (e a).1 * (β (e a).2 * n (e a).2) := by ring
    exact clique_real hq hω1 (sum_nonneg fun r _ => hu r) hpi' hqT hT hk
  exact key (fun r => ((S ∩ A r).image val).card)
    (fun a => ⟨card_inter_le_image (hval a).1 (hsub a).1 S, card_inter_le_image (hval a).2 (hsub a).2 S⟩)
    ((hprem S hS).trans (Nat.cast_le.2 hS))

/-! ### The theorem of record -/

/-- (E) at `b` gives `b ≤ b_y` once some `E(A)` is nonempty. -/
theorem le_bY {ι : Type} [Fintype ι] [DecidableEq ι] (W : Work C ι) {b : ℝ} (hE : Assumptions.Exports W b)
    (h : W.fed.Nonempty) : b ≤ W.bY := by
  rw [Work.bY, dite_eq_left h]
  refine le_inf' h _ fun B hB => ?_
  obtain ⟨-, hBc, hne⟩ := mem_filter.1 hB
  exact (le_div_iff₀ (Nat.cast_pos.2 (card_pos.2 hne))).2 (hE B hBc)

/-- Under (E) at `b > 0`, dividing by `b_y` gives at most dividing by `b`. -/
theorem div_bY_le {ι : Type} [Fintype ι] [DecidableEq ι] (W : Work C ι) {b T : ℝ} (hb : 0 < b) (hT : 0 ≤ T)
    (hE : Assumptions.Exports W b) : T / W.bY ≤ T / b := by
  by_cases h : W.fed.Nonempty
  · exact div_le_div_of_nonneg_left hT hb (le_bY W hE h)
  · rw [Work.bY, dite_eq_right h, div_zero]
    exact div_nonneg hT hb.le

/-- With (E) at `b > 0`, `α ≤ √(k P / b)` for every `P ≥ Π(k)`. -/
theorem alpha_le_of_exports {ι : Type} [Fintype ι] [DecidableEq ι] {d : RegisteredCircuit} (W : Work d.D ι) {k : ℕ}
    {b p : ℝ} (hb : 0 < b) (hE : Assumptions.Exports W b) (hp : (piMax d W k : ℝ) ≤ p) :
    alpha d W k ≤ √(k * p / b) := by
  unfold alpha
  by_cases h : W.fed.Nonempty
  · have hbY := le_bY W hE h
    have hp0 : (0 : ℝ) ≤ p := le_trans (Nat.cast_nonneg _) hp
    refine Real.sqrt_le_sqrt ?_
    calc (k : ℝ) * piMax d W k / W.bY ≤ k * p / W.bY :=
          div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left hp (Nat.cast_nonneg k)) (hb.le.trans hbY)
      _ ≤ k * p / b := div_le_div_of_nonneg_left (mul_nonneg (Nat.cast_nonneg k) hp0) hb hbY
  · rw [Work.bY, dite_eq_right h, div_zero, Real.sqrt_zero]
    exact Real.sqrt_nonneg _

theorem sqrt_scaled {s X : ℝ} (hs : 0 ≤ s) : √(s ^ 2 * X) = s * √X := by
  rw [Real.sqrt_mul (sq_nonneg s), Real.sqrt_sq hs]

theorem alphaHE_nonneg (k : ℕ) (q b : ℝ) : 0 ≤ alphaHE k q b :=
  div_nonneg (Real.sqrt_nonneg _) zero_le_two

theorem shareF_nonneg (ω : ℕ) : 0 ≤ shareF ω := Real.sqrt_nonneg _

theorem HE : Guarantees.HE := by
  intro ι V _ _ _ d W val k m c P βw βx b hw hx hb hval hshare hH hE hk
  obtain ⟨hpi, -⟩ := onePremise hw hx hval hshare hH
  have hα : alpha d W k ≤ alphaHE k (βw * βx) b := (alpha_le_of_exports W hb hE hpi).trans_eq sqrt_quarter
  have hp : (0 : ℝ) ≤ k ^ 2 / (4 * βw * βx) := div_nonneg (sq_nonneg _) (mul_pos (mul_pos four_pos hw) hx).le
  have hk2 : (0 : ℝ) ≤ k / 2 := div_nonneg (Nat.cast_nonneg k) zero_le_two
  have hN := bound_of_le hk hα hpi
  have hs := slowdown_of_bound (Nat.cast_nonneg m) (add_nonneg (Nat.cast_nonneg k) (alphaHE_nonneg k (βw * βx) b))
    hp (Nat.cast_nonneg W.O) (Nat.cast_nonneg c) (Nat.cast_nonneg (d.D.size d.D.computed)) hN
  rw [show (k : ℝ) ^ 2 / (4 * βw * βx) * m * W.O = (k / 2) ^ 2 * (m * W.O / (βw * βx)) by ring,
    sqrt_scaled hk2] at hN
  rw [show (k : ℝ) ^ 2 / (4 * βw * βx) * W.O * W.N = (k / 2) ^ 2 * (W.O * W.N / (βw * βx)) by ring,
    sqrt_scaled hk2] at hs
  exact ⟨hN, hs⟩

/-- `CountHE` is `HE` at the verdict's partition into `B ≤ n` blocks, whose fresh-input budget is `IsCK.reads`; both
bounds grow with the number of blocks. -/
theorem CountHE : Guarantees.CountHE := by
  intro ι V _ _ _ d W val c k n βw βx b hw hx hb hval hshare hH hE hcomp
  obtain ⟨B, hBn, P, hP⟩ := hcomp
  obtain ⟨hN, hs⟩ := HE ι V d W val k B c P βw βx b hw hx hb hval hshare hH hE hP.reads
  have hq : 0 < βw * βx := mul_pos hw hx
  have hB : (B : ℝ) ≤ n := Nat.cast_le.2 hBn
  have hα : 0 ≤ (k : ℝ) + alphaHE k (βw * βx) b := add_nonneg (Nat.cast_nonneg k) (alphaHE_nonneg k _ b)
  refine ⟨hN.trans ?_, hs.trans ?_⟩
  · unfold workHE
    have hO : (B : ℝ) * W.O / (βw * βx) ≤ n * W.O / (βw * βx) :=
      div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_right hB (Nat.cast_nonneg _)) hq.le
    exact add_le_add (mul_le_mul_of_nonneg_right hB hα)
      (mul_le_mul_of_nonneg_left (Real.sqrt_le_sqrt hO) (div_nonneg (Nat.cast_nonneg k) zero_le_two))
  · exact div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_right hB (Nat.cast_nonneg c)) (Nat.cast_nonneg _)

theorem HEShared : Guarantees.HEShared := by
  intro ι ρ V _ _ _ _ d W val k m c ω P G A e β q b hω hβ hq hb hA hval hqe hH hE hk
  have hω' : 1 ≤ ω := hω
  have hω1 : (1 : ℝ) ≤ ω := Nat.one_le_cast.2 hω'
  have hF : (0 : ℝ) ≤ 2 * (1 - 1 / ω) := by
    have h1 : (0 : ℝ) ≤ 1 - 1 / ω := by
      rw [sub_nonneg, div_le_iff₀ (by linarith : (0 : ℝ) < ω)]
      linarith
    linarith
  have hpi := sharedArraysClique hω hβ hq hA hval hqe hH
  have hα : alpha d W k ≤ shareF ω * alphaHE k q b := by
    refine (alpha_le_of_exports W hb hE hpi).trans_eq ?_
    rw [show (k : ℝ) * (2 * (1 - 1 / ω) * (k ^ 2 / (4 * q))) / b = 2 * (1 - 1 / ω) * (k ^ 3 / (q * b) / 2 ^ 2) by
      ring, Real.sqrt_mul hF, Real.sqrt_div' _ (sq_nonneg (2 : ℝ)), Real.sqrt_sq zero_le_two, shareF, alphaHE]
  have hp : (0 : ℝ) ≤ 2 * (1 - 1 / ω) * (k ^ 2 / (4 * q)) :=
    mul_nonneg hF (div_nonneg (sq_nonneg _) (mul_pos four_pos hq).le)
  have hk2 : (0 : ℝ) ≤ k / 2 := div_nonneg (Nat.cast_nonneg k) zero_le_two
  have hN := bound_of_le hk hα hpi
  have hs := slowdown_of_bound (Nat.cast_nonneg m) (add_nonneg (Nat.cast_nonneg k)
    (mul_nonneg (shareF_nonneg ω) (alphaHE_nonneg k q b))) hp (Nat.cast_nonneg W.O) (Nat.cast_nonneg c)
    (Nat.cast_nonneg (d.D.size d.D.computed)) hN
  rw [show (2 : ℝ) * (1 - 1 / ω) * (k ^ 2 / (4 * q)) * m * W.O = 2 * (1 - 1 / ω) * ((k / 2) ^ 2 * (m * W.O / q)) by
    ring, Real.sqrt_mul hF, sqrt_scaled hk2, ← shareF] at hN
  rw [show (2 : ℝ) * (1 - 1 / ω) * (k ^ 2 / (4 * q)) * W.O * W.N =
    2 * (1 - 1 / ω) * ((k / 2) ^ 2 * (W.O * W.N / q)) by ring, Real.sqrt_mul hF, sqrt_scaled hk2, ← shareF] at hs
  exact ⟨hN, hs⟩

/-- **The bound over a set of blocks, under (H) and (E).** `HE`'s premises on the values, with (H) at the budget `k` of
the blocks of `G` and (E) at `b > 0`; all the blocks together read at most `T` gates outside `θ`. `N` is at most the
MACs the blocks outside `G` hold, plus `T + (k/2) √(|G| O_G / (β_w β_x)) + (k/2) √(|G| T / (β_w β_x b))`, with `O_G`
the pairs of a block of `G` and an output with a wire in it (`outOn`), at most `O`. At every block and `T = m k` it
gives `HE`'s bound. -/
theorem bound_on_he {ι V : Type} [Fintype ι] [DecidableEq ι] [DecidableEq V] {d : RegisteredCircuit} {W : Work d.D ι}
    {val : Finset (Fin d.D.N) → V} {k T : ℕ} {βw βx b : ℝ} (P : Partition d.D m) (G : Finset (Fin m))
    (hw : 0 < βw) (hx : 0 < βx) (hb : 0 < b) (hval : ValuesInjective W val) (hshare : NoSharedPair W val)
    (hH : Assumptions.Hardness d ![W.allW, W.allX] val ![βw, βx] k) (hE : Assumptions.Exports W b)
    (hk : ∀ u ∈ G, (d.unregistered (P.gates u)).card ≤ k) (hT : ∑ u, (d.unregistered (P.gates u)).card ≤ T) :
    (W.N : ℝ) ≤ (∑ u ∈ Gᶜ, ∑ a, (inBlock P W u a).card : ℕ) + T + k / 2 * √(G.card * outOn P W G / (βw * βx)) +
      k / 2 * √(G.card * T / (βw * βx * b)) := by
  obtain ⟨hpi, -⟩ := onePremise hw hx hval hshare hH
  have hp : (0 : ℝ) ≤ k ^ 2 / (4 * βw * βx) := div_nonneg (sq_nonneg _) (mul_pos (mul_pos four_pos hw) hx).le
  have hk2 : (0 : ℝ) ≤ k / 2 := div_nonneg (Nat.cast_nonneg k) zero_le_two
  have hg : (0 : ℝ) ≤ G.card := Nat.cast_nonneg _
  have hT0 : (0 : ℝ) ≤ T := Nat.cast_nonneg _
  have hdiv : (T : ℝ) / W.bY ≤ T / b := div_bY_le W hb hT0 hE
  have h1 : √(piMax d W k * G.card * outOn P W G) ≤ k / 2 * √(G.card * outOn P W G / (βw * βx)) := by
    rw [← sqrt_scaled hk2]
    refine Real.sqrt_le_sqrt ?_
    calc (piMax d W k : ℝ) * G.card * outOn P W G ≤ k ^ 2 / (4 * βw * βx) * G.card * outOn P W G :=
          mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_right hpi hg) (Nat.cast_nonneg _)
      _ = (k / 2) ^ 2 * (G.card * outOn P W G / (βw * βx)) := by ring
  have h2 : √(piMax d W k * G.card * (T / W.bY)) ≤ k / 2 * √(G.card * T / (βw * βx * b)) := by
    rw [← sqrt_scaled hk2]
    refine Real.sqrt_le_sqrt ?_
    calc (piMax d W k : ℝ) * G.card * (T / W.bY) ≤ k ^ 2 / (4 * βw * βx) * G.card * (T / b) :=
          mul_le_mul (mul_le_mul_of_nonneg_right hpi hg) hdiv (div_nonneg hT0 (bY_nonneg W)) (mul_nonneg hp hg)
      _ = (k / 2) ^ 2 * (G.card * T / (βw * βx * b)) := by ring
  linarith [bound_on W P G hk hT]

/-- **The bound over a set of blocks, (H) at each block's own budget and its own `β_x`.** `HE`'s premises on the values
and (E) at `b > 0`; each block `u` of `G` takes (H) at its own budget `κ u` and its own `β_x u`, so
`Π(κ u) ≤ κ u² / (4 β_w β_x u)`; all the blocks together, those outside `G` too, read at most `T` gates outside `θ`.
With `Q` at least the sum over `G` of `κ u² / (β_w β_x u)`: `N` is at most the MACs the blocks outside `G` hold, plus
`T + ½ √(Q O_G) + ½ √(Q T / b)`, with `O_G` the pairs of a block of `G` and an output with a wire in it (`outOn`), at
most `O`. No block's budget or cost is another's. -/
theorem bound_on_rows {ι V : Type} [Fintype ι] [DecidableEq ι] [DecidableEq V] {d : RegisteredCircuit}
    {W : Work d.D ι} {val : Finset (Fin d.D.N) → V} {T : ℕ} {βw b Q : ℝ} (P : Partition d.D m)
    (G : Finset (Fin m)) (κ : Fin m → ℕ) (βx : Fin m → ℝ) (hw : 0 < βw) (hx : ∀ u ∈ G, 0 < βx u) (hb : 0 < b)
    (hval : ValuesInjective W val) (hshare : NoSharedPair W val)
    (hH : ∀ u ∈ G, Assumptions.Hardness d ![W.allW, W.allX] val ![βw, βx u] (κ u))
    (hE : Assumptions.Exports W b) (hκ : ∀ u ∈ G, (d.unregistered (P.gates u)).card ≤ κ u)
    (hT : ∑ u, (d.unregistered (P.gates u)).card ≤ T) (hQ : ∑ u ∈ G, (κ u : ℝ) ^ 2 / (βw * βx u) ≤ Q) :
    (W.N : ℝ) ≤ (∑ u ∈ Gᶜ, ∑ a, (inBlock P W u a).card : ℕ) + T +
      1 / 2 * √(Q * outOn P W G) + 1 / 2 * √(Q * T / b) := by
  have hS : ((∑ u ∈ G, piMax d W (κ u) : ℕ) : ℝ) ≤ Q / 4 :=
    calc ((∑ u ∈ G, piMax d W (κ u) : ℕ) : ℝ) = ∑ u ∈ G, (piMax d W (κ u) : ℝ) := Nat.cast_sum _ _
      _ ≤ ∑ u ∈ G, (κ u : ℝ) ^ 2 / (βw * βx u) / 4 := sum_le_sum fun u hu =>
          ((onePremise hw (hx u hu) hval hshare (hH u hu)).1).trans_eq (by ring)
      _ = (∑ u ∈ G, (κ u : ℝ) ^ 2 / (βw * βx u)) / 4 := (sum_div _ _ _).symm
      _ ≤ Q / 4 := div_le_div_of_nonneg_right hQ (by norm_num)
  have hQ0 : (0 : ℝ) ≤ Q / 4 := (Nat.cast_nonneg _).trans hS
  have hT0 : (0 : ℝ) ≤ T := Nat.cast_nonneg _
  have hhalf : (0 : ℝ) ≤ 1 / 2 := by norm_num
  have h1 : √((∑ u ∈ G, piMax d W (κ u) : ℕ) * outOn P W G) ≤ 1 / 2 * √(Q * outOn P W G) := by
    rw [← sqrt_scaled hhalf]
    refine Real.sqrt_le_sqrt ?_
    calc ((∑ u ∈ G, piMax d W (κ u) : ℕ) : ℝ) * outOn P W G
        ≤ Q / 4 * outOn P W G := mul_le_mul_of_nonneg_right hS (Nat.cast_nonneg _)
      _ = (1 / 2) ^ 2 * (Q * outOn P W G) := by ring
  have h2 : √((∑ u ∈ G, piMax d W (κ u) : ℕ) * (T / W.bY)) ≤ 1 / 2 * √(Q * T / b) := by
    rw [← sqrt_scaled hhalf]
    refine Real.sqrt_le_sqrt ?_
    calc ((∑ u ∈ G, piMax d W (κ u) : ℕ) : ℝ) * (T / W.bY)
        ≤ Q / 4 * (T / b) := mul_le_mul hS (div_bY_le W hb hT0 hE) (div_nonneg hT0 (bY_nonneg W)) hQ0
      _ = (1 / 2) ^ 2 * (Q * T / b) := by ring
  linarith [bound_on_budgets W P G κ hκ hT]

/-- **The bound over a set of blocks, each under its own budget, under (H) and (E).** `bound_on_rows` at one `β_x`: with
`Q = ∑_{u ∈ G} (κ u)²`, `N` is at most the MACs the blocks outside `G` hold, plus
`T + ½ √(Q O_G / (β_w β_x)) + ½ √(Q T / (β_w β_x b))`. (H) at a budget gives it at every smaller one, so (H) at the
largest `κ u` is enough; at one budget `k` on `G`, `Q = |G| k²` and it is `bound_on_he`. -/
theorem bound_on_budgets_he {ι V : Type} [Fintype ι] [DecidableEq ι] [DecidableEq V] {d : RegisteredCircuit}
    {W : Work d.D ι} {val : Finset (Fin d.D.N) → V} {T : ℕ} {βw βx b : ℝ} (P : Partition d.D m)
    (G : Finset (Fin m)) (κ : Fin m → ℕ) (hw : 0 < βw) (hx : 0 < βx) (hb : 0 < b) (hval : ValuesInjective W val)
    (hshare : NoSharedPair W val) (hH : ∀ u ∈ G, Assumptions.Hardness d ![W.allW, W.allX] val ![βw, βx] (κ u))
    (hE : Assumptions.Exports W b) (hκ : ∀ u ∈ G, (d.unregistered (P.gates u)).card ≤ κ u)
    (hT : ∑ u, (d.unregistered (P.gates u)).card ≤ T) :
    (W.N : ℝ) ≤ (∑ u ∈ Gᶜ, ∑ a, (inBlock P W u a).card : ℕ) + T +
      1 / 2 * √((∑ u ∈ G, κ u ^ 2 : ℕ) * outOn P W G / (βw * βx)) +
      1 / 2 * √((∑ u ∈ G, κ u ^ 2 : ℕ) * T / (βw * βx * b)) := by
  have hQ : ∑ u ∈ G, (κ u : ℝ) ^ 2 / (βw * βx) ≤ ((∑ u ∈ G, κ u ^ 2 : ℕ) : ℝ) / (βw * βx) := by
    rw [Nat.cast_sum, sum_div]
    exact sum_le_sum fun u _ => le_of_eq (by rw [Nat.cast_pow])
  have h := bound_on_rows (W := W) P G κ (fun _ => βx) hw (fun _ _ => hx) hb hval hshare hH hE hκ hT hQ
  have e1 : ((∑ u ∈ G, κ u ^ 2 : ℕ) : ℝ) / (βw * βx) * outOn P W G =
      (∑ u ∈ G, κ u ^ 2 : ℕ) * outOn P W G / (βw * βx) := by ring
  have e2 : ((∑ u ∈ G, κ u ^ 2 : ℕ) : ℝ) / (βw * βx) * T / b = (∑ u ∈ G, κ u ^ 2 : ℕ) * T / (βw * βx * b) := by
    ring
  rw [e1, e2] at h
  exact h

end Nci.Cut.SecurityProofs
