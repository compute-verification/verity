module

public import Proofs.Nci.Untied
public import Proofs.Nci.CutWitness

/-!
# Instances against the held family and the two untied bounds

A scratch file of instances, never merged: what the lemmas for computed gates taken as inputs and the two bounds
under (H) and (E) say on small circuits, and where they are silent.

* **Two blocks hold gates of one word.** On `CH` a word of two computed gates has one gate in each of two blocks.
  Each block read 3 gates outside `θ` (`toyH_tied`) and, with the word as inputs, reads 4 (`toyH_asInputs`): its own
  3 and the one gate of the word it held (`toyH_held`), so the per-block lemma is met with equality (`toyH_tight`),
  and so is the total `m k + |T|` (`toyH_sum`). No cap below `k + 1` holds for either block (`toyH_not_reads`), so
  no one block pays for the word: the lemmas charge a gate, never a word, to the block that held it.
* **The premises hold together** with a word as inputs (`toyH_premises`) and with an untied registered word
  (`toyU_premises`), on works none of whose words is the untied one.
* **(H) at `d` and a word that is a function of `θ`.** A set of words inside a set of gates that is closed under
  reads and holds no unregistered input has `κ = 0` (`kappaWords_eq_zero_of_closed`): words of registered inputs
  (`kappaWords_eq_zero_of_registered`), and words computed from them. (H) then fails at every cap, at any positive
  cost for an array that holds such a word (`not_hardness_of_kappa_zero_w`, `not_hardness_of_kappa_zero_x`). So (H)
  at `d` fails for a work that multiplies by an untied word (`untie_hH_false`, `untie_premises_false`), and for a
  work of the circuit with gates as inputs that multiplies by a word `d` computes from `θ` alone
  (`asInputs_hH_false`). On the in-tree toy with its `w` gate registered and then untied, (H) holds at the untied
  circuit with (E) and the values, and fails at `d` at every cap and all positive costs (`toyR_premises`). On `CZ`,
  with a gate computed from `θ` as an input, the same (`toyZ_premises`).
* **`Π` at `d` against `Π` untied.** On that toy `Π(1)` is at least 1 at `d`, where the registered `w` gate costs
  nothing, and 0 untied (`toyR_piMax_lt`).
-/

@[expose] public section

namespace Nci.Cut.Attack

open Verity Finset Nci.Cut.SecurityProofs Nci.Cut.SecurityProofs.Untied

/-! ### (H) from words that are unregistered inputs -/

/-- (H) holds at unit costs and every cap where each word of the two arrays is one unregistered input and no word is
in both: every separator from the unregistered inputs holds each such gate. -/
theorem hardness_of_input_words {V : Type} [DecidableEq V] (d : RegisteredCircuit)
    (Aw Ax : Finset (Finset (Fin d.D.N))) (val : Finset (Fin d.D.N) → V) (hdisj : Disjoint Aw Ax)
    (hw : ∀ E ∈ Aw ∪ Ax, ∃ g ∈ unregisteredInputs d, E = {g}) (k : ℕ) :
    Assumptions.Hardness d ![Aw, Ax] val ![1, 1] k := by
  intro S _
  have hd : Disjoint (S ∩ Aw) (S ∩ Ax) :=
    disjoint_of_subset_left inter_subset_right (disjoint_of_subset_right inter_subset_right hdisj)
  have hT : (S ∩ Aw) ∪ (S ∩ Ax) ⊆
      (unregisteredInputs d ∩ S.biUnion id).image fun g => ({g} : Finset (Fin d.D.N)) := by
    intro s hs
    have hsS : s ∈ S := by
      rcases mem_union.1 hs with h | h
      · exact (mem_inter.1 h).1
      · exact (mem_inter.1 h).1
    have hsA : s ∈ Aw ∪ Ax := by
      rcases mem_union.1 hs with h | h
      · exact mem_union_left _ (mem_inter.1 h).2
      · exact mem_union_right _ (mem_inter.1 h).2
    obtain ⟨g, hg, rfl⟩ := hw s hsA
    exact mem_image.2 ⟨g, mem_inter.2 ⟨hg, mem_biUnion.2 ⟨_, hsS, mem_singleton_self _⟩⟩, rfl⟩
  have hcard : (S ∩ Aw).card + (S ∩ Ax).card ≤ kappaWords d S :=
    calc (S ∩ Aw).card + (S ∩ Ax).card
        = ((S ∩ Aw) ∪ (S ∩ Ax)).card := (card_union_of_disjoint hd).symm
      _ ≤ ((unregisteredInputs d ∩ S.biUnion id).image fun g => ({g} : Finset (Fin d.D.N))).card :=
        card_le_card hT
      _ ≤ (unregisteredInputs d ∩ S.biUnion id).card := card_image_le
      _ ≤ kappaWords d S := card_inter_le_kappa (unregisteredInputs d) (S.biUnion id)
  have h1 : ((S ∩ Aw).image val).card ≤ (S ∩ Aw).card := card_image_le
  have h2 : ((S ∩ Ax).image val).card ≤ (S ∩ Ax).card := card_image_le
  have h3 : ((S ∩ Aw).image val).card + ((S ∩ Ax).image val).card ≤ kappaWords d S :=
    (Nat.add_le_add h1 h2).trans hcard
  simp only [Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one, one_mul]
  exact_mod_cast h3

/-! ### Two blocks hold gates of one word -/

/-- The gates each gate of `CH` reads. -/
def readsH : Fin 6 → Finset (Fin 6) := ![∅, ∅, {0}, {1}, {0, 1, 2, 3}, {0, 1, 2, 3}]

/-- Six one-bit gates: unregistered inputs `a`, `c` (gates 0, 1); the word's two gates, 2 computed from `a` and 3
from `c`; and gates 4 and 5, each reading both inputs and both gates of the word. -/
abbrev CH : Circuit where
  N := 6
  w := fun _ => 1
  isInput := fun g => decide (g.val < 2)
  reads := readsH
  reads_lt := by decide
  input_reads := by decide
  op := fun _ _ => 0
  op_congr := fun _ _ _ _ => rfl
  outputs := {4, 5}

/-- `CH`, nothing registered. -/
abbrev dH : RegisteredCircuit := .ofCircuit CH ∅ (by decide)

/-- Two blocks: gates 2 and 4, and gates 3 and 5. Each holds one gate of the word. -/
def PH : Partition CH 2 where
  unit := fun g => if g.val < 2 then none else if g.val % 2 = 0 then some 0 else some 1
  input_iff := by decide

/-- The word: gates 2 and 3. -/
abbrev TH : Finset (Fin CH.N) := {2, 3}

/-- The values: each word carries its own. -/
abbrev valH : Finset (Fin CH.N) → Finset (Fin CH.N) := id

theorem toyH_computed : ∀ g ∈ TH, CH.isInput g = false := by decide

/-- Each block holds one gate of the word. -/
theorem toyH_held : ∀ u, (TH ∩ PH.gates u).card = 1 := by decide

/-- Each block reads 3 gates outside `θ`: an input twice over, the other input and the other block's gate of the
word. -/
theorem toyH_tied : ∀ u, (dH.unregistered (PH.gates u)).card = 3 := by decide

/-- With the word as inputs each block reads 4. -/
theorem toyH_asInputs : ∀ u, ((dH.asInputs TH).unregistered ((PH.asInputs TH).gates u)).card = 4 := by decide

/-- The per-block lemma is met with equality: a block newly reads the one gate of the word it held. -/
theorem toyH_tight (u : Fin 2) :
    ((dH.asInputs TH).unregistered ((PH.asInputs TH).gates u)).card =
      (dH.unregistered (PH.gates u)).card + (TH ∩ PH.gates u).card := by
  rw [toyH_asInputs u, toyH_tied u, toyH_held u]

/-- The total is met with equality: `m k + |T|` at `m = 2`, `k = 3`. -/
theorem toyH_sum :
    ∑ u, ((dH.asInputs TH).unregistered ((PH.asInputs TH).gates u)).card = 2 * 3 + TH.card := by decide

theorem toyH_reads : ReadsAtMost dH PH 3 := fun u => (toyH_tied u).le

/-- `readsAtMost_asInputs_held` at the toy: `k + h` at `k = 3`, `h = 1`. -/
theorem toyH_reads_asInputs : ReadsAtMost (dH.asInputs TH) (PH.asInputs TH) (3 + 1) :=
  readsAtMost_asInputs_held dH PH TH toyH_computed toyH_reads fun u => (toyH_held u).le

/-- No smaller cap holds, for either block: neither is left at `k = 3`, so no one block pays for the word. -/
theorem toyH_not_reads {k : ℕ} (hk : k < 4) (u : Fin 2) :
    ¬ ((dH.asInputs TH).unregistered ((PH.asInputs TH).gates u)).card ≤ k := by
  rw [toyH_asInputs u]
  omega

/-- With the word as inputs, one matmul of one MAC: `w = {0}`, `x = {1}`, the MAC and its output gate 4. -/
def MH : Matmul (CH.asInputs TH) where
  ni := 1
  nj := 1
  nl := 1
  w := fun _ => ({0} : Finset (Fin 6))
  x := fun _ => ({1} : Finset (Fin 6))
  y := fun _ => ({4} : Finset (Fin 6))
  w_inj := fun a b _ => Subsingleton.elim a b
  x_inj := fun a b _ => Subsingleton.elim a b
  mac := fun _ => ({4} : Finset (Fin 6))
  mac_computed := by decide
  mac_connected := fun _ => ⟨singleton_nonempty _, fun _ hT hne hT' =>
    absurd ((subset_singleton_iff.1 hT).resolve_left hne.ne_empty) hT'⟩
  reads_w := by decide
  reads_x := by decide
  feeds_y := fun _ => ⟨_, mem_singleton_self _, _, mem_singleton_self _, .refl _⟩

/-- The charged work with the word as inputs: `MH` alone. -/
def WH : Work (CH.asInputs TH) Unit where
  M := fun _ => MH
  mac_disjoint := by decide

theorem toyH_N : WH.N = 1 := by decide

theorem toyH_clear (B : Finset (Fin (CH.asInputs TH).N)) : WH.clear B = ∅ := by
  refine filter_false_of_mem fun o _ h => ?_
  obtain ⟨⟨s, rfl, hsB⟩, hdis⟩ := h
  exact disjoint_left.1 hdis (mem_singleton_self _) (hsB (mem_singleton_self _))

theorem toyH_exports (b : ℝ) : Assumptions.Exports WH b := by
  intro B _
  rw [toyH_clear, card_empty, Nat.cast_zero, mul_zero]
  exact Nat.cast_nonneg _

theorem toyH_hardness (k : ℕ) : Assumptions.Hardness dH ![WH.allW, WH.allX] valH ![1, 1] k :=
  hardness_of_input_words dH WH.allW WH.allX valH (by decide) (by decide) k

/-- **The premises of `bound_asInputs_he` hold together** with a word of two computed gates as inputs, a gate in each
of two blocks, every block in `G`: the word's gates are computed, each block read at most 3 gates outside `θ`, (H)
holds at `d` at each block's cap `3 + 1`, (E) holds with the word as inputs, the values are distinct, and the work
has a MAC. -/
theorem toyH_premises :
    (∀ g ∈ TH, CH.isInput g = false) ∧ ReadsAtMost dH PH 3 ∧
      (∀ u, Assumptions.Hardness dH ![WH.allW, WH.allX] valH ![1, 1] (3 + (TH ∩ PH.gates u).card)) ∧
      Assumptions.Exports WH 1 ∧ ValuesInjective WH valH ∧ NoSharedPair WH valH ∧ WH.N = 1 :=
  ⟨toyH_computed, toyH_reads, fun _ => toyH_hardness _, toyH_exports 1,
    fun _ => ⟨Set.injOn_id _, Set.injOn_id _⟩, fun a a' h => absurd (Subsingleton.elim a a') h, toyH_N⟩

/-! ### An untied registered word that the work does not multiply by -/

/-- The gates each gate of `CU` reads. -/
def readsU : Fin 4 → Finset (Fin 4) := ![∅, ∅, ∅, {0, 1, 2}]

/-- Four one-bit gates: unregistered inputs `w`, `x` (gates 0, 1), a registered input (gate 2), and the MAC (gate 3),
which reads all three. -/
abbrev CU : Circuit where
  N := 4
  w := fun _ => 1
  isInput := fun g => decide (g.val < 3)
  reads := readsU
  reads_lt := by decide
  input_reads := by decide
  op := fun _ _ => 0
  op_congr := fun _ _ _ _ => rfl
  outputs := {3}

/-- `CU` with gate 2 registered. -/
abbrev dU : RegisteredCircuit := .ofCircuit CU {2} (by decide)

/-- One matmul of one MAC: `w = {0}`, `x = {1}`, the MAC and its output gate 3. -/
def MU : Matmul CU where
  ni := 1
  nj := 1
  nl := 1
  w := fun _ => {0}
  x := fun _ => {1}
  y := fun _ => {3}
  w_inj := fun a b _ => Subsingleton.elim a b
  x_inj := fun a b _ => Subsingleton.elim a b
  mac := fun _ => {3}
  mac_computed := by decide
  mac_connected := fun _ => ⟨singleton_nonempty _, fun _ hT hne hT' =>
    absurd ((subset_singleton_iff.1 hT).resolve_left hne.ne_empty) hT'⟩
  reads_w := by decide
  reads_x := by decide
  feeds_y := fun _ => ⟨_, mem_singleton_self _, _, mem_singleton_self _, .refl _⟩

/-- The charged work: `MU` alone. -/
def WU : Work CU Unit where
  M := fun _ => MU
  mac_disjoint := by decide

/-- One block holding the MAC. -/
def PU : Partition CU 1 where
  unit := fun g => if g.val < 3 then none else some 0
  input_iff := by decide

/-- The values: each word carries its own. -/
abbrev valU : Finset (Fin CU.N) → Finset (Fin CU.N) := id

/-- The untied words: the registered gate alone. -/
abbrev wordsU : Finset (Finset (Fin CU.N)) := {{2}}

theorem toyU_reads : ReadsAtMost dU PU 2 := by
  unfold ReadsAtMost
  decide

/-- Untied, the block reads `k + b` gates outside `θ`, at `k = 2` and `b = 1`. -/
theorem toyU_untied : ((dU.untie (wordsU.biUnion id)).unregistered (PU.gates 0)).card = 2 + 1 := by decide

theorem toyU_clear (B : Finset (Fin CU.N)) : WU.clear B = ∅ := by
  refine filter_false_of_mem fun o _ h => ?_
  obtain ⟨⟨s, rfl, hsB⟩, hdis⟩ := h
  exact disjoint_left.1 hdis (mem_singleton_self _) (hsB (mem_singleton_self _))

theorem toyU_exports (b : ℝ) : Assumptions.Exports WU b := by
  intro B _
  rw [toyU_clear, card_empty, Nat.cast_zero, mul_zero]
  exact Nat.cast_nonneg _

theorem toyU_hardness (k : ℕ) : Assumptions.Hardness dU ![WU.allW, WU.allX] valU ![1, 1] k :=
  hardness_of_input_words dU WU.allW WU.allX valU (by decide) (by decide) k

/-- **The premises of `bound_untie_he` hold together** with one untied word, a registered input that the work does
not multiply by: the word is one gate, the block read at most 2 gates outside `θ`, the word has one reader, (H)
holds at `d` at the block's cap, (E) holds and the values are distinct. -/
theorem toyU_premises :
    (∀ S ∈ wordsU, S.card ≤ 1) ∧ ReadsAtMost dU PU 2 ∧ OneReader PU wordsU ∧
      (∀ u, Assumptions.Hardness dU ![WU.allW, WU.allX] valU ![1, 1]
        (2 + 1 * (wordsRead dU.D wordsU (PU.gates u)).card)) ∧
      Assumptions.Exports WU 1 ∧ ValuesInjective WU valU ∧ NoSharedPair WU valU ∧ WU.N = 1 :=
  ⟨by decide, toyU_reads, fun _ _ u v _ _ => Subsingleton.elim u v, fun _ => toyU_hardness _, toyU_exports 1,
    fun _ => ⟨Set.injOn_id _, Set.injOn_id _⟩, fun a a' h => absurd (Subsingleton.elim a a') h, by decide⟩

/-! ### (H) at `d` and a word that is a function of `θ` -/

/-- A path that ends in a set of gates closed under reads starts in it. -/
theorem mem_of_feedsWithin_closed {C : Circuit} {R A : Finset (Fin C.N)} (hA : ∀ h ∈ A, C.reads h ⊆ A)
    {g t : Fin C.N} (hp : Nci.Covering.FeedsWithin C R g t) : t ∈ A → g ∈ A := by
  induction hp with
  | refl _ => exact fun h => h
  | step _ hgh _ ih => exact fun ht => hA _ (ih ht) hgh

/-- **Words that are functions of `θ` alone have `κ = 0`.** Words inside a set `A` of gates that is closed under reads
and holds no unregistered input are reached from no unregistered input, so the empty set separates them. -/
theorem kappaWords_eq_zero_of_closed (d : RegisteredCircuit) (S : Finset (Finset (Fin d.D.N)))
    (A : Finset (Fin d.D.N)) (hA : ∀ h ∈ A, d.D.reads h ⊆ A) (hU : ∀ g ∈ A, g ∉ unregisteredInputs d)
    (hS : S.biUnion id ⊆ A) : kappaWords d S = 0 := by
  have hsep : Separates d.D ∅ (unregisteredInputs d) (S.biUnion id) := fun x hx y hy hp =>
    hU x (mem_of_feedsWithin_closed hA hp (hS hy)) hx
  have h : kappa d.D (unregisteredInputs d) (S.biUnion id) ≤ (∅ : Finset (Fin d.D.N)).card := kappa_le hsep
  rw [card_empty] at h
  exact Nat.le_zero.1 h

/-- Words of registered inputs have `κ = 0`. -/
theorem kappaWords_eq_zero_of_registered (d : RegisteredCircuit) (S : Finset (Finset (Fin d.D.N)))
    (hS : ∀ g ∈ S.biUnion id, g ∈ d.θ) : kappaWords d S = 0 :=
  kappaWords_eq_zero_of_closed d S (S.biUnion id)
    (fun h hh => by
      rw [d.D.input_reads h (d.θ_input h (hS h hh))]
      exact empty_subset _)
    (fun g hg hu => (mem_filter.1 hu).2.2 (hS g hg)) (Subset.refl _)

/-- (H) fails at every cap, at a positive cost of a `w`-value, where a `w`-word has `κ = 0`. -/
theorem not_hardness_of_kappa_zero_w {V : Type} [DecidableEq V] {d : RegisteredCircuit}
    {Aw Ax : Finset (Finset (Fin d.D.N))} {val : Finset (Fin d.D.N) → V} {βw βx : ℝ} (hw : 0 < βw) (hx : 0 ≤ βx)
    {E : Finset (Fin d.D.N)} (hE : E ∈ Aw) (hκ : kappaWords d {E} = 0) (k : ℕ) :
    ¬ Assumptions.Hardness d ![Aw, Ax] val ![βw, βx] k := by
  intro hH
  have h := hH {E} (by rw [hκ]; exact Nat.zero_le k)
  simp only [Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one, singleton_inter_of_mem hE, image_singleton,
    card_singleton, Nat.cast_one, mul_one] at h
  rw [hκ, Nat.cast_zero] at h
  have h0 : ∀ t : ℝ, 0 ≤ t → βw + βx * t ≤ 0 → False := fun t ht h' => by linarith [mul_nonneg hx ht]
  exact h0 _ (Nat.cast_nonneg _) h

/-- (H) fails at every cap, at a positive cost of an `x`-value, where an `x`-word has `κ = 0`. -/
theorem not_hardness_of_kappa_zero_x {V : Type} [DecidableEq V] {d : RegisteredCircuit}
    {Aw Ax : Finset (Finset (Fin d.D.N))} {val : Finset (Fin d.D.N) → V} {βw βx : ℝ} (hw : 0 ≤ βw) (hx : 0 < βx)
    {E : Finset (Fin d.D.N)} (hE : E ∈ Ax) (hκ : kappaWords d {E} = 0) (k : ℕ) :
    ¬ Assumptions.Hardness d ![Aw, Ax] val ![βw, βx] k := by
  intro hH
  have h := hH {E} (by rw [hκ]; exact Nat.zero_le k)
  simp only [Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one, singleton_inter_of_mem hE, image_singleton,
    card_singleton, Nat.cast_one, mul_one] at h
  rw [hκ, Nat.cast_zero] at h
  have h0 : ∀ t : ℝ, 0 ≤ t → βw * t + βx ≤ 0 → False := fun t ht h' => by linarith [mul_nonneg hw ht]
  exact h0 _ (Nat.cast_nonneg _) h

/-- **(H) at `d` fails for a work that multiplies by a word of registered inputs**, at every cap and all positive
costs. -/
theorem untie_hH_false {ι V : Type} [Fintype ι] [DecidableEq V] (d : RegisteredCircuit) (W : Work d.D ι)
    (val : Finset (Fin d.D.N) → V) {βw βx : ℝ} (hw : 0 < βw) (hx : 0 < βx) {E : Finset (Fin d.D.N)}
    (hE : E ∈ W.allW ∨ E ∈ W.allX) (hθ : ∀ g ∈ E, g ∈ d.θ) (k : ℕ) :
    ¬ Assumptions.Hardness d ![W.allW, W.allX] val ![βw, βx] k := by
  have hκ : kappaWords d {E} = 0 :=
    kappaWords_eq_zero_of_registered d {E} (by rw [singleton_biUnion]; exact hθ)
  rcases hE with hE | hE
  · exact not_hardness_of_kappa_zero_w hw hx.le hE hκ k
  · exact not_hardness_of_kappa_zero_x hw.le hx hE hκ k

/-- **The premises of `bound_untie_he` do not hold together for a work that multiplies by an untied word**: with a
block in `G`, positive costs, and a `w`- or `x`-word of the work among the untied words, each a word of registered
inputs, (H) at `d` at the block's cap is false. -/
theorem untie_premises_false {ι V : Type} [Fintype ι] [DecidableEq V] {m : ℕ} (d : RegisteredCircuit)
    (P : Partition d.D m) (W : Work d.D ι) (val : Finset (Fin d.D.N) → V) {k b : ℕ} {βw βx : ℝ}
    (𝒲 : Finset (Finset (Fin d.D.N))) (h𝒲 : ∀ S ∈ 𝒲, ∀ g ∈ S, g ∈ d.θ) (G : Finset (Fin m)) (hG : G.Nonempty)
    (hw : 0 < βw) (hx : 0 < βx) {E : Finset (Fin d.D.N)} (hE𝒲 : E ∈ 𝒲) (hE : E ∈ W.allW ∨ E ∈ W.allX)
    (hH : ∀ u ∈ G, Assumptions.Hardness d ![W.allW, W.allX] val ![βw, βx]
      (k + b * (wordsRead d.D 𝒲 (P.gates u)).card)) : False := by
  obtain ⟨u, hu⟩ := hG
  exact untie_hH_false d W val hw hx hE (h𝒲 E hE𝒲) _ (hH u hu)

/-- **(H) at `d` fails for a work of the circuit with `T` as inputs that multiplies by a word `d` computes from `θ`
alone**, at every cap and all positive costs. -/
theorem asInputs_hH_false {ι V : Type} [Fintype ι] [DecidableEq V] (d : RegisteredCircuit) (T : Finset (Fin d.D.N))
    (W : Work (d.D.asInputs T) ι) (val : Finset (Fin d.D.N) → V) {βw βx : ℝ} (hw : 0 < βw) (hx : 0 < βx)
    {E : Finset (Fin d.D.N)} (hE : E ∈ W.allW ∨ E ∈ W.allX) (hκ : kappaWords d {E} = 0) (k : ℕ) :
    ¬ Assumptions.Hardness d ![W.allW, W.allX] val ![βw, βx] k := by
  rcases hE with hE | hE
  · exact not_hardness_of_kappa_zero_w hw hx.le hE hκ k
  · exact not_hardness_of_kappa_zero_x hw.le hx hE hκ k

/-! ### The in-tree toy with its `w` gate registered, then untied -/

/-- The in-tree toy with its `w` gate, gate 0, registered. -/
abbrev dR : RegisteredCircuit := .ofCircuit Toy.C {0} (by decide)

/-- The untied word: the `w` gate. -/
abbrev wordR : Finset (Fin Toy.C.N) := {0}

theorem toyR_word : wordR ∈ Toy.W.allW := by
  rw [toy_allW]
  exact mem_singleton_self _

/-- Untied at its `w` gate, the toy has the unregistered inputs of the toy with nothing registered. -/
theorem toyR_unreg : unregisteredInputs (dR.untie wordR) = unregisteredInputs Toy.d := by decide

theorem toyR_kappa (S : Finset (Finset (Fin Toy.C.N))) : kappaWords (dR.untie wordR) S = kappaWords Toy.d S :=
  congrArg (fun X : Finset (Fin Toy.C.N) => kappa Toy.C X (S.biUnion id)) toyR_unreg

/-- (H) holds at the untied circuit, at unit costs and every cap. -/
theorem toyR_hardness_untied (k : ℕ) :
    Assumptions.Hardness (dR.untie wordR) Toy.arrays Toy.val ![1, 1] k := by
  intro S hS
  have hκ : kappaWords (dR.untie wordR) S = kappaWords Toy.d S := toyR_kappa S
  exact (toy_hardness k S ((le_of_eq hκ.symm).trans hS)).trans (Nat.cast_le.2 (le_of_eq hκ.symm))

/-- One block reads 2 gates outside `θ`, the two `x` gates. -/
theorem toyR_reads : ReadsAtMost dR Toy.P1 2 := by
  unfold ReadsAtMost
  decide

/-- Untied, it reads 3: `k + b` for the one word, at `b = 1`. -/
theorem toyR_untied : ((dR.untie wordR).unregistered (Toy.P1.gates 0)).card = 2 + 1 := by decide

/-- **Where `bound_untie_he` is silent.** The in-tree toy with its `w` gate registered, untied at that gate, one
block. The block read 2 gates outside `θ` and the untied word has one reader. At the untied circuit (H) holds at unit
costs and every cap, (E) holds and the values are distinct, which is what the bound with a cap for each block asks
there. (H) at `d`, which `bound_untie_he` asks, fails at every cap and all positive costs: the work multiplies by the
untied word, whose `κ` at `d` is 0. -/
theorem toyR_premises :
    ReadsAtMost dR Toy.P1 2 ∧ OneReader Toy.P1 ({wordR} : Finset (Finset (Fin Toy.C.N))) ∧
      (∀ k, Assumptions.Hardness (dR.untie wordR) Toy.arrays Toy.val ![1, 1] k) ∧
      Assumptions.Exports Toy.W 1 ∧ ValuesInjective Toy.W Toy.val ∧ NoSharedPair Toy.W Toy.val ∧
      ∀ βw βx : ℝ, 0 < βw → 0 < βx → ∀ k, ¬ Assumptions.Hardness dR Toy.arrays Toy.val ![βw, βx] k :=
  ⟨toyR_reads, fun _ _ u v _ _ => Subsingleton.elim u v, toyR_hardness_untied, toy_exports 1, toy_values,
    toy_noSharedPair,
    fun _ _ hw hx k => untie_hH_false dR Toy.W Toy.val hw hx (Or.inl toyR_word) (by decide) k⟩

/-! ### `Π` at `d` and `Π` untied, at that toy -/

/-- A path into a gate that reads nothing is that gate alone. -/
theorem eq_of_feedsWithin_of_reads_empty {C : Circuit} {R : Finset (Fin C.N)} {g t : Fin C.N}
    (hp : Nci.Covering.FeedsWithin C R g t) (ht : C.reads t = ∅) : g = t := by
  induction hp with
  | refl _ => rfl
  | step _ hgh _ ih =>
    rw [ih ht, ht] at hgh
    exact absurd hgh (notMem_empty _)

/-- A path inside `R` starts in `R`. -/
theorem mem_of_feedsWithin {C : Circuit} {R : Finset (Fin C.N)} {g t : Fin C.N}
    (hp : Nci.Covering.FeedsWithin C R g t) : g ∈ R := by
  cases hp with
  | refl h => exact h
  | step h _ _ => exact h

/-- The toy's `w`-word and one of its `x`-words. -/
abbrev pairR : Finset (Finset (Fin Toy.C.N)) := {{0}, {1}}

theorem toyR_pair_reads : ∀ y ∈ pairR.biUnion id, Toy.C.reads y = ∅ := by decide

theorem toyR_pair_starts :
    ∀ x ∈ unregisteredInputs dR, x ∈ pairR.biUnion id → x ∉ (univ \ {1} : Finset (Fin Toy.C.N)) := by decide

/-- At `d` the pair is behind one gate: the registered `w` gate costs nothing. -/
theorem toyR_kappa_tied : kappaWords dR pairR ≤ 1 := by
  have hsep : Separates Toy.C ({1} : Finset (Fin Toy.C.N)) (unregisteredInputs dR) (pairR.biUnion id) := by
    intro x hx y hy hp
    have hxy : x = y := eq_of_feedsWithin_of_reads_empty hp (toyR_pair_reads y hy)
    rw [← hxy] at hy
    exact toyR_pair_starts x hx hy (mem_of_feedsWithin hp)
  have h : kappa Toy.C (unregisteredInputs dR) (pairR.biUnion id) ≤ ({1} : Finset (Fin Toy.C.N)).card :=
    kappa_le hsep
  rw [card_singleton] at h
  exact h

theorem toyR_pi : Toy.W.pi pairR = 1 := by
  first
    | decide
    | (simp only [Work.pi, Fintype.univ_unit, sum_singleton, toy_wWords, toy_xWords]; decide)

/-- `Π(1)` at `d` is at least 1: the pair. -/
theorem toyR_piMax_tied : 1 ≤ piMax dR Toy.W 1 := by
  have h : Toy.W.pi pairR ≤ piMax dR Toy.W 1 := by
    first
      | exact Finset.le_sup (f := Toy.W.pi) (mem_filter.2 ⟨mem_univ _, toyR_kappa_tied⟩)
      | (unfold piMax; exact Finset.le_sup (f := Toy.W.pi) (mem_filter.2 ⟨mem_univ pairR, toyR_kappa_tied⟩))
  rw [toyR_pi] at h
  exact h

/-- `Π(1)` untied is 0: a `w`-word and an `x`-word are then two unregistered inputs. -/
theorem toyR_piMax_untied : piMax (dR.untie wordR) Toy.W 1 = 0 := by
  have h : piMax (dR.untie wordR) Toy.W 1 = piMax Toy.d Toy.W 1 := by
    first
      | exact congrArg (fun f : Finset (Finset (Fin Toy.C.N)) → ℕ =>
          ((univ : Finset (Finset (Finset (Fin Toy.C.N)))).filter fun S => f S ≤ 1).sup Toy.W.pi)
          (funext toyR_kappa)
      | (unfold piMax; simp only [toyR_kappa])
      | (unfold piMax; simp only [toyR_kappa]; rfl)
  rw [h]
  exact toy_piMax le_rfl

/-- **`Π` at `d` is larger than `Π` untied**, at the cap 1: a count that takes `Π` at `d` for a work that multiplies
by an untied word counts that word as free. -/
theorem toyR_piMax_lt : piMax (dR.untie wordR) Toy.W 1 < piMax dR Toy.W 1 := by
  rw [toyR_piMax_untied]
  exact toyR_piMax_tied

/-! ### A gate computed from `θ`, as an input -/

/-- The gates each gate of `CZ` reads. -/
def readsZ : Fin 4 → Finset (Fin 4) := ![∅, ∅, {0}, {1, 2}]

/-- Four one-bit gates: a registered input (gate 0), an unregistered input `x` (gate 1), gate 2 computed from the
registered input alone, and the MAC (gate 3), which reads gates 1 and 2. -/
abbrev CZ : Circuit where
  N := 4
  w := fun _ => 1
  isInput := fun g => decide (g.val < 2)
  reads := readsZ
  reads_lt := by decide
  input_reads := by decide
  op := fun _ _ => 0
  op_congr := fun _ _ _ _ => rfl
  outputs := {3}

/-- `CZ` with gate 0 registered. -/
abbrev dZ : RegisteredCircuit := .ofCircuit CZ {0} (by decide)

/-- The gate computed from `θ`. -/
abbrev TZ : Finset (Fin CZ.N) := {2}

/-- The values: each word carries its own. -/
abbrev valZ : Finset (Fin CZ.N) → Finset (Fin CZ.N) := id

/-- With gate 2 as an input, one matmul of one MAC: `w = {2}`, `x = {1}`, the MAC and its output gate 3. -/
def MZ : Matmul (CZ.asInputs TZ) where
  ni := 1
  nj := 1
  nl := 1
  w := fun _ => ({2} : Finset (Fin 4))
  x := fun _ => ({1} : Finset (Fin 4))
  y := fun _ => ({3} : Finset (Fin 4))
  w_inj := fun a b _ => Subsingleton.elim a b
  x_inj := fun a b _ => Subsingleton.elim a b
  mac := fun _ => ({3} : Finset (Fin 4))
  mac_computed := by decide
  mac_connected := fun _ => ⟨singleton_nonempty _, fun _ hT hne hT' =>
    absurd ((subset_singleton_iff.1 hT).resolve_left hne.ne_empty) hT'⟩
  reads_w := by decide
  reads_x := by decide
  feeds_y := fun _ => ⟨_, mem_singleton_self _, _, mem_singleton_self _, .refl _⟩

/-- The charged work with gate 2 as an input: `MZ` alone. -/
def WZ : Work (CZ.asInputs TZ) Unit where
  M := fun _ => MZ
  mac_disjoint := by decide

theorem toyZ_allW : WZ.allW = {TZ} := by decide

theorem toyZ_word : TZ ∈ WZ.allW := by
  rw [toyZ_allW]
  exact mem_singleton_self _

/-- At `d` the word is a function of `θ` alone: gates 0 and 2 are closed under reads and hold no unregistered
input. -/
theorem toyZ_kappa : kappaWords dZ ({TZ} : Finset (Finset (Fin CZ.N))) = 0 :=
  kappaWords_eq_zero_of_closed dZ {TZ} ({0, 2} : Finset (Fin CZ.N)) (by decide) (by decide) (by decide)

/-- With the gate as an input, (H) holds at unit costs and every cap: both words are unregistered inputs there. -/
theorem toyZ_hardness_asInputs (k : ℕ) :
    Assumptions.Hardness (dZ.asInputs TZ) ![WZ.allW, WZ.allX] valZ ![1, 1] k :=
  hardness_of_input_words (dZ.asInputs TZ) WZ.allW WZ.allX valZ (by decide) (by decide) k

/-- **Where `bound_asInputs_he` is silent.** With a gate that `d` computes from `θ` alone as an input, and a work that
multiplies by it: (H) holds with the gate as an input, at unit costs and every cap; (H) at `d`, which
`bound_asInputs_he` asks, fails at every cap and all positive costs. -/
theorem toyZ_premises :
    (∀ g ∈ TZ, CZ.isInput g = false) ∧
      (∀ k, Assumptions.Hardness (dZ.asInputs TZ) ![WZ.allW, WZ.allX] valZ ![1, 1] k) ∧
      ∀ βw βx : ℝ, 0 < βw → 0 < βx → ∀ k, ¬ Assumptions.Hardness dZ ![WZ.allW, WZ.allX] valZ ![βw, βx] k :=
  ⟨by decide, toyZ_hardness_asInputs,
    fun _ _ hw hx k => asInputs_hH_false dZ TZ WZ valZ hw hx (Or.inl toyZ_word) toyZ_kappa k⟩

end Nci.Cut.Attack
