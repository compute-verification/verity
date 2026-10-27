import Proofs.Pous.Pebbling.B2Stacking
import Definitions.Pous
import Definitions.Pous.Assumptions
import Proofs.Pous.Statements
import Proofs.Pous.Sanity
import Proofs.Pous.Accounting.Numbers

/-!
# Fisch 2018/702, Claims 13–14, proved

The definitions and statements are copies of the trusted ones (`Security/Definitions/Pous/Model/Pebbling.lean`,
`Pous.Guarantees.FischClaim13` and `FischClaim14`).

* **Claim 13** from Claim 12 (`PousB2.b2_stacking`). Let `E` be the set of endpoints. If `|E| < εn/2`, pebble
  every node of `E` black as well. That is at most `(1 − ε/2)n` black pebbles, and `δ < (ε/2)/2`, so Claim 12 gives
  a new path, which is also a path for the old configuration. Its end is then in `E`, yet it carries no new pebble.
* **Claim 14** from three general lemmas about the game:
  - `exists_first`: a vertex first receives a pebble by a legal move;
  - `green_before`: a green path to a vertex with no green child is pebbled in order;
  - `lt_of_isChain_before`: pebbling in order puts the `k`-th vertex at round `k + 1` or later.

  The reverse rule at a node of `P` would need the next node of its green path pebbled first, which
  `green_before` forbids. So each node of `P` is pebbled forward, after its predecessor.
-/

namespace PousFisch

open PousB2

variable {ℓ n : ℕ} {V : Type}

/-- Fisch App. B, the white/green rule: `v` may receive a pebble if all its parents carry pebbles, or if `v` has a
green child and every green child `u` of `v` carries a pebble, as do all white parents of `u`. -/
def Available (node : Fin ℓ → Fin n → V) (drg : Fin ℓ → Fin n → Fin n → Prop) (green : V → V → Prop)
    (C : Finset V) (v : V) : Prop :=
  (∀ u, StackedEdge node drg green u v → u ∈ C) ∨
    ((∃ u, green v u) ∧
      ∀ u, green v u → u ∈ C ∧ ∀ w, (∃ i a b, drg i a b ∧ w = node i a ∧ u = node i b) → w ∈ C)

/-- A parallel pebbling: round `r + 1` places pebbles only on vertices available in `C r`, and may remove any. -/
def IsPebbling (node : Fin ℓ → Fin n → V) (drg : Fin ℓ → Fin n → Fin n → Prop) (green : V → V → Prop)
    (C : ℕ → Finset V) : Prop :=
  ∀ r v, v ∈ C (r + 1) → v ∉ C r → Available node drg green (C r) v

/-! ### The pebbling game: general lemmas

Only the two move rules matter here, so these lemmas hold for any graph in B2's model. -/

section Game

variable {node : Fin ℓ → Fin n → V} {drg : Fin ℓ → Fin n → Fin n → Prop} {green : V → V → Prop}

/-- `x` carries a pebble strictly before every round in which `y` does. -/
def Before (C : ℕ → Finset V) (x y : V) : Prop := ∀ r, y ∈ C r → ∃ r' < r, x ∈ C r'

/-- A vertex without a pebble at the start first receives one by a legal move, in some round before any round in
which it carries one. -/
theorem exists_first {C : ℕ → Finset V} (hC : IsPebbling node drg green C) {y : V} (hy0 : y ∉ C 0) {r : ℕ}
    (hy : y ∈ C r) : ∃ r₀, r₀ < r ∧ (∀ r' ≤ r₀, y ∉ C r') ∧ Available node drg green (C r₀) y := by
  classical
  have hex : ∃ r, y ∈ C r := ⟨r, hy⟩
  have h1 : y ∈ C (Nat.find hex) := Nat.find_spec hex
  have hle : Nat.find hex ≤ r := Nat.find_min' hex hy
  obtain ⟨r₀, hr₀⟩ : ∃ r₀, Nat.find hex = r₀ + 1 :=
    Nat.exists_eq_succ_of_ne_zero fun h => hy0 (h ▸ h1)
  have hmin : ∀ r' ≤ r₀, y ∉ C r' := fun r' hr' => Nat.find_min hex (by omega)
  exact ⟨r₀, by omega, hmin, hC r₀ y (hr₀ ▸ h1) (hmin r₀ le_rfl)⟩

/-- A green path of vertices without pebbles at the start, whose last vertex has no green child, is pebbled in
order (the green-section half of Fisch's proof of Claim 14). -/
theorem green_before {C : ℕ → Finset V} (hC : IsPebbling node drg green C) :
    ∀ g : List V, g.IsChain green → (∀ w ∈ g, w ∉ C 0) → (∀ z ∈ g.getLast?, ∀ u, ¬ green z u) →
      g.IsChain (Before C)
  | [], _, _, _ => List.IsChain.nil
  | [x], _, _, _ => List.isChain_singleton x
  | x :: y :: rest, hg, h0, hlast => by
    have hxy := (List.isChain_cons_cons.1 hg).1
    have ih := green_before hC (y :: rest) (List.isChain_cons_cons.1 hg).2
      (fun w hw => h0 w (List.mem_cons_of_mem _ hw)) (by simpa [List.getLast?_cons_cons] using hlast)
    refine List.isChain_cons_cons.2 ⟨fun r hy => ?_, ih⟩
    obtain ⟨r₀, hr₀, hmin, havail⟩ := exists_first hC (h0 y (by simp)) hy
    rcases havail with hfwd | ⟨⟨u, hu⟩, hrev⟩
    · exact ⟨r₀, hr₀, hfwd x (Or.inl hxy)⟩
    · exfalso
      cases rest with
      | nil => exact hlast y rfl u hu
      | cons y' rest' =>
        have hyy' := (List.isChain_cons_cons.1 (List.isChain_cons_cons.1 hg).2).1
        obtain ⟨r', hr', hyr'⟩ := (List.isChain_cons_cons.1 ih).1 r₀ (hrev y' hyy').1
        exact hmin r' hr'.le hyr'

/-- Along a chain of `Before`, starting without pebbles, the `k`-th vertex has no pebble before round `k + 1`. -/
theorem lt_of_isChain_before {C : ℕ → Finset V} {l : List V} (hl : l.IsChain (Before C))
    (h0 : ∀ x ∈ l, x ∉ C 0) : ∀ (k : ℕ) (hk : k < l.length) (r : ℕ), l[k] ∈ C r → k < r := by
  intro k
  induction k with
  | zero =>
    intro hk r hr
    rcases Nat.eq_zero_or_pos r with rfl | h
    · exact absurd hr (h0 _ (List.getElem_mem hk))
    · exact h
  | succ k ih =>
    intro hk r hr
    obtain ⟨r', hr', hx⟩ := hl.getElem k hk r hr
    have := ih (by omega) r' hx
    omega

end Game

/-- **Claim 13** (Fisch 2018/702). Under B2's hypotheses with `δ < ε/4`, at least `εn/2` nodes of the last layer
`V_t` end an unpebbled path of at least `βn` nodes, each of whose nodes has an unpebbled green path to `V_t`. -/
theorem claim13 [Fintype V] [DecidableEq V] (α β ε δ : ℝ)
    (node : Fin ℓ → Fin n → V) (level : V → Fin ℓ) (drg : Fin ℓ → Fin n → Fin n → Prop) (green : V → V → Prop)
    (hnode : ∀ i, Function.Injective (node i)) (hlevel : ∀ i a, level (node i a) = i)
    (hdag : ∃ r : V → ℕ, ∀ u v, StackedEdge node drg green u v → r u < r v)
    (hdr : ∀ i, DepthRobust n (drg i) α β)
    (hsc : ∀ i j : Fin ℓ, (j : ℕ) = i + 1 → DisjointPaths node level green i j)
    (hn : 0 < n) (hε : 0 < ε) (hδ : δ < ε / 4) (hℓ : 1 ≤ (1 - α - δ) * ℓ)
    (black red : Finset V) (hblack : (black.card : ℝ) ≤ (1 - ε) * n)
    (hred : ∀ i, ((red.filter fun v => level v = i).card : ℝ) ≤ δ * n) :
    ∀ t : Fin ℓ, (t : ℕ) + 1 = ℓ →
      ∃ E : Finset (Fin n), ε / 2 * n ≤ E.card ∧ ∀ b ∈ E,
        ∃ P : List V,
          P.IsChain (StackedEdge node drg green) ∧ P.Nodup ∧ β * n ≤ P.length ∧
          (∀ v ∈ P, Unpebbled black red v) ∧ P.getLast? = some (node t b) ∧
          ∀ v ∈ P, ∃ g : List V, g.head? = some v ∧ g.IsChain green ∧ (∀ w ∈ g, Unpebbled black red w) ∧
            ∃ b', g.getLast? = some (node t b') := by
  classical
  intro t ht
  set E := Finset.univ.filter fun b : Fin n =>
    ∃ P : List V,
      P.IsChain (StackedEdge node drg green) ∧ P.Nodup ∧ β * n ≤ P.length ∧
      (∀ v ∈ P, Unpebbled black red v) ∧ P.getLast? = some (node t b) ∧
      ∀ v ∈ P, ∃ g : List V, g.head? = some v ∧ g.IsChain green ∧ (∀ w ∈ g, Unpebbled black red w) ∧
        ∃ b', g.getLast? = some (node t b') with hE
  refine ⟨E, ?_, fun b hb => (Finset.mem_filter.1 hb).2⟩
  -- if fewer, pebble every endpoint black as well and apply Claim 12 with `ε / 2`
  by_contra hlt
  push Not at hlt
  have hblack' : ((black ∪ E.image (node t)).card : ℝ) ≤ (1 - ε / 2) * n := by
    have h1 : (black ∪ E.image (node t)).card ≤ black.card + E.card :=
      (Finset.card_union_le _ _).trans (Nat.add_le_add_left Finset.card_image_le _)
    have h2 : ((black ∪ E.image (node t)).card : ℝ) ≤ black.card + E.card := by exact_mod_cast h1
    linarith
  obtain ⟨P, hPc, hPnd, hPlen, hPunp, ⟨b, hPlast⟩, hPg⟩ :=
    b2_stacking α β (ε / 2) δ node level drg green hnode hlevel hdag hdr hsc hn (by linarith) (by linarith) hℓ
      (black ∪ E.image (node t)) red hblack' hred t ht
  have hmono : ∀ v, Unpebbled (black ∪ E.image (node t)) red v → Unpebbled black red v :=
    fun v hv => ⟨fun h => hv.1 (Finset.mem_union_left _ h), hv.2⟩
  -- the new path's end is an endpoint for the old configuration, yet it carries no new pebble
  have hbE : b ∈ E := by
    rw [hE, Finset.mem_filter]
    refine ⟨Finset.mem_univ _, P, hPc, hPnd, hPlen, fun v hv => hmono v (hPunp v hv), hPlast, fun v hv => ?_⟩
    obtain ⟨g, h1, h2, h3, b', h4⟩ := hPg v hv
    exact ⟨g, h1, h2, fun w hw => hmono w (h3 w hw), b', h4⟩
  exact (hPunp _ (List.mem_of_getLast? hPlast)).1 (Finset.mem_union_right _ (Finset.mem_image_of_mem _ hbE))

/-- **Claim 14** (Fisch 2018/702). In the white/green pebbling game, an unpebbled path each of whose nodes has an
unpebbled green path to the last layer `V_t` is pebbled in order: its `k`-th node carries no pebble before round
`k + 1`. -/
theorem claim14 (node : Fin ℓ → Fin n → V) (drg : Fin ℓ → Fin n → Fin n → Prop) (green : V → V → Prop)
    (t : Fin ℓ) (htop : ∀ b u, ¬ green (node t b) u)
    (black red : Finset V) (C : ℕ → Finset V) (hC0 : ∀ v, Unpebbled black red v → v ∉ C 0)
    (hC : IsPebbling node drg green C)
    (P : List V) (hP : P.IsChain (StackedEdge node drg green)) (hunp : ∀ v ∈ P, Unpebbled black red v)
    (hgreen : ∀ v ∈ P, ∃ g : List V, g.head? = some v ∧ g.IsChain green ∧ (∀ w ∈ g, Unpebbled black red w) ∧
      ∃ b, g.getLast? = some (node t b)) :
    ∀ (k : ℕ) (hk : k < P.length) (r : ℕ), P[k] ∈ C r → k < r := by
  have h0 : ∀ v ∈ P, v ∉ C 0 := fun v hv => hC0 v (hunp v hv)
  refine lt_of_isChain_before (List.isChain_iff_getElem.2 fun i hi => ?_) h0
  -- `P[i]` is pebbled before `P[i+1]`: the reverse rule at `P[i+1]` would need its green path pebbled first
  intro r₁ hy
  obtain ⟨r₀, hr₀, hmin, havail⟩ := exists_first hC (h0 _ (List.getElem_mem hi)) hy
  rcases havail with hfwd | ⟨⟨u, hu⟩, hrev⟩
  · exact ⟨r₀, hr₀, hfwd _ (hP.getElem i hi)⟩
  · exfalso
    obtain ⟨g, hgh, hgc, hgu, b, hgl⟩ := hgreen _ (List.getElem_mem hi)
    have hglast : ∀ z ∈ g.getLast?, ∀ u, ¬ green z u := by
      intro z hz u
      rw [hgl, Option.mem_some_iff] at hz
      subst hz
      exact htop b u
    have hbefore := green_before hC g hgc (fun w hw => hC0 w (hgu w hw)) hglast
    cases g with
    | nil => simp at hgh
    | cons y0 gt =>
      simp only [List.head?_cons, Option.some.injEq] at hgh
      subst hgh
      cases gt with
      | nil => exact hglast _ rfl u hu
      | cons y' rest =>
        obtain ⟨r', hr', hyr'⟩ :=
          (List.isChain_cons_cons.1 hbefore).1 r₀ (hrev y' (List.isChain_cons_cons.1 hgc).1).1
        exact hmin r' hr'.le hyr'

end PousFisch
