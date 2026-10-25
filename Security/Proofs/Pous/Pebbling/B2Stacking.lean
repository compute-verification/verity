import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Basic.Real.Basic
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Data.List.Chain
import Mathlib.Tactic.Linarith
import Definitions.Pous
import Models.Pous.Assumptions
import Proofs.Pous.Statements
import Proofs.Pous.Sanity
import Proofs.Pous.Accounting.Numbers

/-!
# B2: Fisch's stacking lemma, proved

Source: B. Fisch, ePrint 2018/702, Appendix B, Claim 12. The definitions and the statement of `b2_stacking` are
copied verbatim from `B2StackingDraft.lean` (the statement under red-team review); its header lists the modelling
choices. Only the `sorry` is replaced.

The combinatorial core is kept in general lemmas, so it survives a restatement:

* paths as lists (`IsPathTo`, joining, suffixes, simple paths from a DAG rank);
* `exists_path_avoiding`: pairwise vertex-disjoint paths cannot all be blocked by fewer vertices than there are
  paths;
* `card_reach_ge`: one connector step. If more than `|K|` nodes of `V_{i+1}` are good, all but `|K|` nodes of `V_i`
  reach one by an unpebbled green path, where `K` holds the pebbles of the level `V_i ∪ G_i`.

The proof of `b2_stacking` follows Fisch's. Let `G_i` be the nodes of `V_i` with an unpebbled green path to the last
layer `V_t`. Downward induction gives `|G_i| ≥ n − ρ_i − δn`, where `ρ_i` counts the black pebbles of level `i`. If
every `|G_i| < αn`, the `ρ_i` sum past `ℓ(1 − α − δ)n ≥ n`. So some `|G_i| ≥ αn`, and depth robustness gives a path
of `βn` nodes in `G_i`, which the green path from its end extends to `V_t`.
-/

namespace PousB2

/-- Fisch §2.6: a DAG on `n` nodes is `(n, α, β)`-depth-robust if every subgraph on (at least) `αn` nodes contains a
path of at least `βn` nodes. -/
def DepthRobust (n : ℕ) (E : Fin n → Fin n → Prop) (α β : ℝ) : Prop :=
  ∀ S : Finset (Fin n), α * n ≤ S.card →
    ∃ p : List (Fin n), p.IsChain E ∧ p.Nodup ∧ (∀ v ∈ p, v ∈ S) ∧ β * n ≤ p.length

variable {ℓ n : ℕ} {V : Type}

/-- The edges of the stacked graph: the white (DRG) edges inside each layer and the green (connector) edges. -/
def StackedEdge (node : Fin ℓ → Fin n → V) (drg : Fin ℓ → Fin n → Fin n → Prop) (green : V → V → Prop)
    (u v : V) : Prop :=
  green u v ∨ ∃ i a b, drg i a b ∧ u = node i a ∧ v = node i b

/-- The connector from layer `i` to layer `j`, as the disjoint-path property of an `n`-superconcentrator with
sources `V_i` and sinks `V_j`: any `k` nodes `A` of `V_i` and `k` nodes `B` of `V_j` are joined by `k` pairwise
vertex-disjoint green paths, the one from `a ∈ A` running inside the level `V_i ∪ G_i` until it ends in `B`. -/
def DisjointPaths (node : Fin ℓ → Fin n → V) (level : V → Fin ℓ) (green : V → V → Prop) (i j : Fin ℓ) : Prop :=
  ∀ A B : Finset (Fin n), A.card = B.card →
    ∃ path : Fin n → List V, ∀ a ∈ A,
      (path a).head? = some (node i a) ∧
      (∃ b ∈ B, (path a).getLast? = some (node j b)) ∧
      (path a).IsChain green ∧
      (∀ v ∈ (path a).dropLast, level v = i) ∧
      ∀ a' ∈ A, a' ≠ a → ∀ v ∈ path a, v ∉ path a'

/-- A vertex carries no pebble. -/
def Unpebbled (black red : Finset V) (v : V) : Prop := v ∉ black ∧ v ∉ red

/-! ### Paths as lists

A path is a list whose consecutive vertices are joined by edges (`List.IsChain`). The lemmas below are about lists
and relations only, so they do not depend on the shape of the graph. -/

section Paths

variable {α : Type} {R : α → α → Prop}

/-- A path from `x` to `z` all of whose vertices satisfy `Q`. -/
def IsPathTo (R : α → α → Prop) (Q : α → Prop) (x z : α) (g : List α) : Prop :=
  g.head? = some x ∧ g.IsChain R ∧ (∀ w ∈ g, Q w) ∧ g.getLast? = some z

/-- A vertex of a list ending at `z` lies before the end or is `z`. -/
theorem mem_dropLast_or_eq {l : List α} {z w : α} (hz : l.getLast? = some z) (hw : w ∈ l) :
    w ∈ l.dropLast ∨ w = z := by
  rw [← List.dropLast_append_getLast? z hz] at hw
  simpa using hw

theorem eq_cons_of_head? {l : List α} {x : α} (h : l.head? = some x) : l = x :: l.tail := by
  cases l with
  | nil => simp at h
  | cons y t => simp only [List.head?_cons, Option.some.injEq] at h; rw [h]; rfl

/-- Joining a path ending at `x` to a path starting at `x`. -/
theorem isChain_append_tail {l₁ l₂ : List α} {x : α} (h₁ : l₁.IsChain R) (h₂ : l₂.IsChain R)
    (hx₁ : l₁.getLast? = some x) (hx₂ : l₂.head? = some x) : (l₁ ++ l₂.tail).IsChain R := by
  rw [eq_cons_of_head? hx₂] at h₂
  rw [List.isChain_append]
  refine ⟨h₁, h₂.tail, fun a ha b hb => ?_⟩
  rw [hx₁, Option.mem_some_iff] at ha
  subst ha
  cases ht : l₂.tail with
  | nil => simp [ht] at hb
  | cons c t =>
    rw [ht, List.head?_cons, Option.mem_some_iff] at hb
    subst hb
    rw [ht] at h₂
    exact (List.isChain_cons_cons.1 h₂).1

theorem getLast?_append_tail {l₁ l₂ : List α} {x : α} (hx₁ : l₁.getLast? = some x) (hx₂ : l₂.head? = some x) :
    (l₁ ++ l₂.tail).getLast? = l₂.getLast? := by
  rw [eq_cons_of_head? hx₂]
  cases ht : l₂.tail with
  | nil => simpa using hx₁
  | cons c t => rw [List.getLast?_cons_cons, List.getLast?_append]; simp [List.getLast?_cons]

theorem head?_append_tail {l₁ l₂ : List α} {x : α} (hx₁ : l₁.getLast? = some x) :
    (l₁ ++ l₂.tail).head? = l₁.head? := by
  cases l₁ with
  | nil => simp at hx₁
  | cons y t => rfl

/-- Paths compose. -/
theorem IsPathTo.trans {Q : α → Prop} {x y z : α} {g₁ g₂ : List α} (h₁ : IsPathTo R Q x y g₁)
    (h₂ : IsPathTo R Q y z g₂) : IsPathTo R Q x z (g₁ ++ g₂.tail) :=
  ⟨(head?_append_tail h₁.2.2.2).trans h₁.1, isChain_append_tail h₁.2.1 h₂.2.1 h₁.2.2.2 h₂.1,
    fun w hw => (List.mem_append.1 hw).elim (h₁.2.2.1 w) fun hw => h₂.2.2.1 w (List.mem_of_mem_tail hw),
    (getLast?_append_tail h₁.2.2.2 h₂.1).trans h₂.2.2.2⟩

/-- The part of a path from any of its vertices on. -/
theorem IsPathTo.suffix {Q : α → Prop} {x z v : α} {g : List α} (h : IsPathTo R Q x z g) (hv : v ∈ g) :
    ∃ g', IsPathTo R Q v z g' := by
  obtain ⟨s, t, rfl⟩ := List.mem_iff_append.1 hv
  refine ⟨v :: t, rfl, h.2.1.right_of_append, fun w hw => h.2.2.1 w (List.mem_append_right _ hw), ?_⟩
  have := h.2.2.2
  rw [List.getLast?_append] at this
  simpa [List.getLast?_cons] using this

/-- A path along which a rank strictly increases is simple. -/
theorem nodup_of_rank {l : List α} (r : α → ℕ) (hr : ∀ u v, R u v → r u < r v) (h : l.IsChain R) : l.Nodup := by
  have hmap : (l.map r).IsChain (· < ·) := (List.isChain_map r).2 (h.imp fun u v huv => hr u v huv)
  exact List.Nodup.of_map r (hmap.pairwise.imp fun hlt => ne_of_lt hlt)

end Paths

/-! ### Disjoint-path counting -/

/-- Pairwise vertex-disjoint paths indexed by `W` cannot all be blocked by fewer than `|W|` vertices. -/
theorem exists_path_avoiding {ι W' : Type} [DecidableEq W'] (W : Finset ι) (path : ι → List W') (K : Finset W')
    (hdisj : ∀ a ∈ W, ∀ a' ∈ W, a' ≠ a → ∀ v ∈ path a, v ∉ path a') (hK : K.card < W.card) :
    ∃ a ∈ W, ∀ v ∈ path a, v ∉ K := by
  by_contra hno
  push Not at hno
  have hpd : (W : Set ι).PairwiseDisjoint fun a => K.filter (· ∈ path a) := by
    intro a ha a' ha' hne
    exact Finset.disjoint_filter.2 fun v _ h1 h2 => hdisj a ha a' ha' (Ne.symm hne) v h1 h2
  have h1 : W.card ≤ ∑ a ∈ W, (K.filter (· ∈ path a)).card := by
    calc W.card = ∑ _a ∈ W, 1 := by simp
      _ ≤ _ := Finset.sum_le_sum fun a ha => by
          obtain ⟨v, hv, hvK⟩ := hno a ha
          exact Finset.card_pos.2 ⟨v, Finset.mem_filter.2 ⟨hvK, hv⟩⟩
  have h2 : ∑ a ∈ W, (K.filter (· ∈ path a)).card ≤ K.card := by
    rw [← Finset.card_biUnion hpd]
    exact Finset.card_le_card (Finset.biUnion_subset.2 fun a _ => Finset.filter_subset _ _)
  omega

/-! ### Pebbling-path extension across one connector -/

open Classical in
/-- One connector step (the first paragraph of Fisch's proof). `Good` is a set of more than `|K|` nodes of `V_j`,
all satisfying `Q`, and `K` contains every vertex of the level `V_i ∪ G_i` failing `Q`. Then all but at most `|K|`
nodes `a` of `V_i` have a green `Q`-path from `node i a` to `node j b` for some `b ∈ Good`. -/
theorem card_reach_ge [DecidableEq V] (node : Fin ℓ → Fin n → V) (level : V → Fin ℓ) (green : V → V → Prop)
    {i j : Fin ℓ} (hsc : DisjointPaths node level green i j) (Q : V → Prop) (K : Finset V)
    (hK : ∀ v, level v = i → ¬ Q v → v ∈ K) (Good : Finset (Fin n)) (hGood : ∀ b ∈ Good, Q (node j b))
    (hcard : K.card < Good.card) :
    n ≤ (Finset.univ.filter fun a =>
      ∃ b ∈ Good, ∃ g : List V, IsPathTo green Q (node i a) (node j b) g).card + K.card := by
  classical
  set U := Finset.univ.filter fun a => ∃ b ∈ Good, ∃ g : List V, IsPathTo green Q (node i a) (node j b) g
    with hU
  by_contra hlt
  push Not at hlt
  have hX : K.card + 1 ≤ Uᶜ.card := by
    rw [Finset.card_compl, Fintype.card_fin]
    omega
  obtain ⟨W, hWX, hWc⟩ := Finset.exists_subset_card_eq hX
  obtain ⟨B, hBG, hBc⟩ := Finset.exists_subset_card_eq (show K.card + 1 ≤ Good.card by omega)
  obtain ⟨path, hpath⟩ := hsc W B (hWc.trans hBc.symm)
  obtain ⟨a, haW, hav⟩ :=
    exists_path_avoiding W path K (fun a ha a' ha' hne => (hpath a ha).2.2.2.2 a' ha' hne) (by omega)
  obtain ⟨hhead, ⟨b, hbB, hlast⟩, hchain, hdrop, -⟩ := hpath a haW
  have haU : a ∈ U := by
    rw [hU, Finset.mem_filter]
    refine ⟨Finset.mem_univ _, b, hBG hbB, path a, hhead, hchain, fun w hw => ?_, hlast⟩
    rcases mem_dropLast_or_eq hlast hw with hw' | rfl
    · by_contra hq
      exact hav w hw (hK w (hdrop w hw') hq)
    · exact hGood b (hBG hbB)
  exact Finset.mem_compl.1 (hWX haW) haU

/-- **B2** (Fisch 2018/702, Claim 12). In `ℓ ≥ 1/(1 − α − δ)` stacked `(n, α, β)`-depth-robust layers joined by
superconcentrators, any `(1 − ε)n` black pebbles plus at most `δn` red pebbles per level, with `δ < ε/2`, leave an
unpebbled path of at least `βn` nodes ending in the last layer `V_t`, each of whose nodes has an unpebbled green path
to a node of `V_t`. -/
theorem b2_stacking [Fintype V] [DecidableEq V] (α β ε δ : ℝ)
    (node : Fin ℓ → Fin n → V) (level : V → Fin ℓ) (drg : Fin ℓ → Fin n → Fin n → Prop) (green : V → V → Prop)
    (hnode : ∀ i, Function.Injective (node i)) (hlevel : ∀ i a, level (node i a) = i)
    (hdag : ∃ r : V → ℕ, ∀ u v, StackedEdge node drg green u v → r u < r v)
    (hdr : ∀ i, DepthRobust n (drg i) α β)
    (hsc : ∀ i j : Fin ℓ, (j : ℕ) = i + 1 → DisjointPaths node level green i j)
    (hn : 0 < n) (hε : 0 < ε) (hδ : δ < ε / 2) (hℓ : 1 ≤ (1 - α - δ) * ℓ)
    (black red : Finset V) (hblack : (black.card : ℝ) ≤ (1 - ε) * n)
    (hred : ∀ i, ((red.filter fun v => level v = i).card : ℝ) ≤ δ * n) :
    ∀ t : Fin ℓ, (t : ℕ) + 1 = ℓ →
      ∃ P : List V,
        P.IsChain (StackedEdge node drg green) ∧ P.Nodup ∧ β * n ≤ P.length ∧
        (∀ v ∈ P, Unpebbled black red v) ∧
        (∃ b, P.getLast? = some (node t b)) ∧
        ∀ v ∈ P, ∃ g : List V, g.head? = some v ∧ g.IsChain green ∧ (∀ w ∈ g, Unpebbled black red w) ∧
          ∃ b, g.getLast? = some (node t b) := by
  classical
  intro t ht
  obtain ⟨r, hr⟩ := hdag
  have hnpos : (0 : ℝ) < n := by exact_mod_cast hn
  have hδε : 2 * (δ * n) < ε * n := by
    have := mul_lt_mul_of_pos_right (show 2 * δ < ε by linarith) hnpos
    linarith
  -- `G i`: the nodes of `V_i` with an unpebbled green path to `V_t`
  let G : Fin ℓ → Finset (Fin n) := fun i => Finset.univ.filter fun a =>
    ∃ b, ∃ g : List V, IsPathTo green (Unpebbled black red) (node i a) (node t b) g
  -- `K i`: the pebbled vertices of the level `V_i ∪ G_i`; `ρ i`: its black pebbles
  let K : Fin ℓ → Finset V := fun i => (black ∪ red).filter fun v => level v = i
  let ρ : Fin ℓ → ℕ := fun i => (black.filter fun v => level v = i).card
  have hG : ∀ i a, a ∈ G i ↔ ∃ b, ∃ g : List V, IsPathTo green (Unpebbled black red) (node i a) (node t b) g := by
    intro i a
    simp only [G, Finset.mem_filter, Finset.mem_univ, true_and]
  have hGunp : ∀ i a, a ∈ G i → Unpebbled black red (node i a) := by
    intro i a ha
    obtain ⟨_, g, hg⟩ := (hG i a).1 ha
    exact hg.2.2.1 _ (List.mem_of_mem_head? hg.1)
  have hKmem : ∀ i v, level v = i → ¬ Unpebbled black red v → v ∈ K i := by
    intro i v hv hu
    simp only [K, Finset.mem_filter, Finset.mem_union]
    refine ⟨?_, hv⟩
    by_contra h
    push Not at h
    exact hu h
  have hKρ : ∀ i, ((K i).card : ℝ) ≤ ρ i + δ * n := by
    intro i
    have h1 : (K i).card ≤ ρ i + (red.filter fun v => level v = i).card := by
      simp only [K, ρ, Finset.filter_union]
      exact Finset.card_union_le _ _
    have h2 : ((K i).card : ℝ) ≤ (ρ i : ℝ) + ((red.filter fun v => level v = i).card : ℝ) := by
      exact_mod_cast h1
    linarith [hred i]
  have hρpair : ∀ i j, i ≠ j → (ρ i : ℝ) + ρ j ≤ (1 - ε) * n := by
    intro i j hij
    have h1 : ρ i + ρ j ≤ black.card := by
      simp only [ρ]
      rw [← Finset.card_union_of_disjoint (Finset.disjoint_filter.2 fun v _ h1 h2 => hij (h1.symm.trans h2))]
      exact Finset.card_le_card (Finset.union_subset (Finset.filter_subset _ _) (Finset.filter_subset _ _))
    have h2 : (ρ i : ℝ) + ρ j ≤ black.card := by exact_mod_cast h1
    linarith
  have hρle : ∀ i, (ρ i : ℝ) ≤ (1 - ε) * n := by
    intro i
    have : ρ i ≤ black.card := Finset.card_le_card (Finset.filter_subset _ _)
    have : (ρ i : ℝ) ≤ black.card := by exact_mod_cast this
    linarith
  -- the last layer: every unpebbled node is good
  have htop : n ≤ (G t).card + (K t).card := by
    have hsub : (Finset.univ : Finset (Fin n)) ⊆ G t ∪ Finset.univ.filter (fun b => node t b ∈ K t) := by
      intro b _
      by_cases hb : Unpebbled black red (node t b)
      · refine Finset.mem_union_left _ ((hG t b).2 ⟨b, [node t b], rfl, List.isChain_singleton _, ?_, rfl⟩)
        simpa using hb
      · exact Finset.mem_union_right _ (Finset.mem_filter.2 ⟨Finset.mem_univ _, hKmem t _ (hlevel t b) hb⟩)
    have h1 := Finset.card_le_card hsub
    have h2 := Finset.card_union_le (G t) (Finset.univ.filter (fun b => node t b ∈ K t))
    have h3 : (Finset.univ.filter (fun b => node t b ∈ K t)).card ≤ (K t).card :=
      Finset.card_le_card_of_injOn (node t) (fun b hb => (Finset.mem_filter.1 hb).2) (hnode t).injOn
    rw [Finset.card_univ, Fintype.card_fin] at h1
    omega
  -- one level down: more good nodes above than pebbles here leaves all but the pebbles here good
  have hstep : ∀ i j : Fin ℓ, (j : ℕ) = i + 1 → (K i).card < (G j).card → n ≤ (G i).card + (K i).card := by
    intro i j hij hlt
    refine (card_reach_ge node level green (hsc i j hij) (Unpebbled black red) (K i) (hKmem i) (G j)
      (hGunp j) hlt).trans (Nat.add_le_add_right (Finset.card_le_card fun a ha => ?_) _)
    obtain ⟨b, hb, g₁, hg₁⟩ := (Finset.mem_filter.1 ha).2
    obtain ⟨c, g₂, hg₂⟩ := (hG j b).1 hb
    exact (hG i a).2 ⟨c, _, hg₁.trans hg₂⟩
  -- downward induction: `|G i| ≥ n − ρ_i − δn` on every layer
  have hbound : ∀ k (i : Fin ℓ), (i : ℕ) + k = t → (n : ℝ) ≤ (G i).card + ρ i + δ * n := by
    intro k
    induction k with
    | zero =>
      intro i hi
      obtain rfl : i = t := Fin.ext (by omega)
      have h1 : (n : ℝ) ≤ (G i).card + (K i).card := by exact_mod_cast htop
      linarith [hKρ i]
    | succ k ih =>
      intro i hi
      let j : Fin ℓ := ⟨i + 1, by omega⟩
      have hIH := ih j (by simp only [j]; omega)
      have hne : i ≠ j := fun h => by
        have := congrArg Fin.val h
        simp only [j] at this
        omega
      have hlt : (K i).card < (G j).card := by
        have : ((K i).card : ℝ) < (G j).card := by linarith [hρpair i j hne, hKρ i]
        exact_mod_cast this
      have h1 : (n : ℝ) ≤ (G i).card + (K i).card := by exact_mod_cast hstep i j rfl hlt
      linarith [hKρ i]
  -- some layer has at least `αn` good nodes, else the black pebbles exceed `(1 − ε)n`
  have hex : ∃ i, α * n ≤ (G i).card := by
    by_contra hno
    push Not at hno
    have hlt : ∀ i ∈ (Finset.univ : Finset (Fin ℓ)), (1 - α - δ) * n < (ρ i : ℝ) := by
      intro i _
      have h1 := hbound (t - i) i (by omega)
      linarith [hno i]
    have hsum := Finset.sum_lt_sum_of_nonempty ⟨t, Finset.mem_univ _⟩ hlt
    rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at hsum
    have htot : ∑ i, (ρ i : ℝ) = black.card := by
      rw [Finset.card_eq_sum_card_fiberwise (f := level) (t := Finset.univ) fun _ _ => Finset.mem_coe.2
        (Finset.mem_univ _)]
      rw [Nat.cast_sum]
    have hℓn : (n : ℝ) ≤ ℓ * ((1 - α - δ) * n) := by
      have := mul_le_mul_of_nonneg_right hℓ hnpos.le
      linarith
    have hεn : 0 < ε * n := mul_pos hε hnpos
    linarith
  -- `V_t` has a good node
  have hGt : (G t).Nonempty := by
    rw [Finset.nonempty_iff_ne_empty]
    intro hempty
    have h1 := hbound 0 t (by simp)
    rw [hempty, Finset.card_empty] at h1
    push_cast at h1
    linarith [hρle t, mul_pos hε hnpos]
  obtain ⟨i, hi⟩ := hex
  obtain ⟨p, hpc, -, hpG, hplen⟩ := hdr i (G i) hi
  have hwhite : ((p.map (node i)).IsChain (StackedEdge node drg green)) :=
    (List.isChain_map (node i)).2 (hpc.imp fun a b hab => Or.inr ⟨i, a, b, hab, rfl, rfl⟩)
  cases hlast : p.getLast? with
  | none =>
    -- `βn ≤ 0`: a single good node of `V_t` will do
    rw [List.getLast?_eq_none_iff] at hlast
    subst hlast
    obtain ⟨c, hc⟩ := hGt
    refine ⟨[node t c], List.isChain_singleton _, List.nodup_singleton _, ?_, ?_, ⟨c, rfl⟩, ?_⟩
    · simp only [List.length_nil, Nat.cast_zero] at hplen
      simp only [List.length_singleton, Nat.cast_one]
      linarith
    · simpa using hGunp t c hc
    · intro v hv
      rw [List.mem_singleton] at hv
      subst hv
      exact ⟨[node t c], rfl, List.isChain_singleton _, by simpa using hGunp t c hc, c, rfl⟩
  | some a =>
    -- the depth-robust path in `V_i`, then the green path from its end to `V_t`
    obtain ⟨b, g, hg⟩ := (hG i a).1 (hpG a (List.mem_of_getLast? hlast))
    have hmlast : (p.map (node i)).getLast? = some (node i a) := by rw [List.getLast?_map, hlast]; rfl
    have hPc : (p.map (node i) ++ g.tail).IsChain (StackedEdge node drg green) :=
      isChain_append_tail hwhite (hg.2.1.imp fun u v h => Or.inl h) hmlast hg.1
    refine ⟨p.map (node i) ++ g.tail, hPc, nodup_of_rank r hr hPc, ?_, ?_, ⟨b, ?_⟩, ?_⟩
    · rw [List.length_append, List.length_map]
      push_cast
      have : (0 : ℝ) ≤ g.tail.length := Nat.cast_nonneg _
      linarith
    · intro v hv
      rcases List.mem_append.1 hv with hv | hv
      · obtain ⟨a', ha', rfl⟩ := List.mem_map.1 hv
        exact hGunp i a' (hpG a' ha')
      · exact hg.2.2.1 v (List.mem_of_mem_tail hv)
    · exact (getLast?_append_tail hmlast hg.1).trans hg.2.2.2
    · intro v hv
      rcases List.mem_append.1 hv with hv | hv
      · obtain ⟨a', ha', rfl⟩ := List.mem_map.1 hv
        obtain ⟨c, g', hg'⟩ := (hG i a').1 (hpG a' ha')
        exact ⟨g', hg'.1, hg'.2.1, hg'.2.2.1, c, hg'.2.2.2⟩
      · obtain ⟨g', hg'⟩ := hg.suffix (List.mem_of_mem_tail hv)
        exact ⟨g', hg'.1, hg'.2.1, hg'.2.2.1, b, hg'.2.2.2⟩

end PousB2
