import Mathlib
import Models.Accounting.Guarantees.RateAuditSketch
import Proofs.Accounting.Bandwidth.AuditRO
import Proofs.Accounting.RateAudit.Chip

/-!
# The rate audit with a sketch-then-hash tag: the proof

An accepted run whose compute is within `F` and whose footprint is short of the floor falls in one of four events:

- the signs' fiber exceeds `g`: the signs' chance, `Pr_σ[g < fib_σ]`;
- a correct query for a tile one of whose sectors, or one of the stream's, the run never obtained: at most `g/|X|` per
  query index (`card_bad_le`), because resampling the unobtained sector leaves the run as it is
  (`Device.obtained_congr`) and at most `g` of its values give the query's digest (`card_resample`);
- fewer than `k` tiles queried, and every sampled tile among them: `((k − 1)/N)^s`;
- a sampled tile never queried, with the right tag anyway: `2^-h` per sampled tile (`card_guess_le`).

Otherwise at least `k` tiles were queried and each, with the stream, was obtained whole, and counting the sectors
obtained puts the footprint at the floor (`floor_le_of_complete`).
-/

namespace Accounting.RateSketch

open Pous (Bits pr)
open scoped ENNReal
open Bandwidth (card_line_le)

namespace Chip

variable {N m ρ h : ℕ} {X Dg α : Type}

/-- The positions a run's reads were answered at: device memory and rebuilding at store sectors, the link at any. -/
def reads (O : Oracle N ρ h X Dg) (D : Data N m ρ X) : Chip N m ρ h X Dg α → Finset (Pos N m ρ)
  | done _ => ∅
  | tag x k => (k (O x)).reads O D
  | mem p k => insert (.inl p) ((k (D (.inl p))).reads O D)
  | link p k => insert p ((k (D p)).reads O D)
  | rebuild p k => insert (.inl p) ((k (D (.inl p))).reads O D)

/-- Changing the oracle off the points a run queries, and the data off the positions it read, changes neither its
value, nor its queries, nor what it read. -/
theorem congr (O O' : Oracle N ρ h X Dg) (D D' : Data N m ρ X) (c : Chip N m ρ h X Dg α)
    (hO : ∀ x ∈ c.queries O D, O' x = O x) (hD : ∀ p ∈ c.reads O D, D' p = D p) :
    c.run O' D' = c.run O D ∧ c.queries O' D' = c.queries O D ∧ c.reads O' D' = c.reads O D := by
  induction c with
  | done a => exact ⟨rfl, rfl, rfl⟩
  | tag x k ih =>
    simp only [queries, List.mem_cons, forall_eq_or_imp] at hO
    simp only [reads] at hD
    obtain ⟨h1, h2, h3⟩ := ih (O x) hO.2 hD
    simp only [run, queries, reads, hO.1]
    exact ⟨h1, by rw [h2], h3⟩
  | mem p k ih =>
    simp only [queries] at hO
    simp only [reads, Finset.mem_insert, forall_eq_or_imp] at hD
    obtain ⟨h1, h2, h3⟩ := ih _ hO hD.2
    simp only [run, queries, reads, hD.1]
    exact ⟨h1, h2, by rw [h3]⟩
  | link p k ih =>
    simp only [queries] at hO
    simp only [reads, Finset.mem_insert, forall_eq_or_imp] at hD
    obtain ⟨h1, h2, h3⟩ := ih _ hO hD.2
    simp only [run, queries, reads, hD.1]
    exact ⟨h1, h2, by rw [h3]⟩
  | rebuild p k ih =>
    simp only [queries] at hO
    simp only [reads, Finset.mem_insert, forall_eq_or_imp] at hD
    obtain ⟨h1, h2, h3⟩ := ih _ hO hD.2
    simp only [run, queries, reads, hD.1]
    exact ⟨h1, h2, by rw [h3]⟩

/-- The positions read are at most the distinct sectors read from device memory, the link's sectors and the rebuilt
sectors (with any set `M` of store sectors on both sides). -/
theorem card_reads_le (O : Oracle N ρ h X Dg) (D : Data N m ρ X) (c : Chip N m ρ h X Dg α)
    (M : Finset (Fin N × Fin m)) :
    (c.reads O D ∪ M.map Function.Embedding.inl).card ≤ (c.mems O D ∪ M).card + c.linkSteps O D + c.rebuilds O D := by
  induction c generalizing M with
  | done a => simp [reads, mems, linkSteps, rebuilds]
  | tag x k ih => simpa [reads, mems, linkSteps, rebuilds] using ih (O x) M
  | mem p k ih =>
    have := ih (D (.inl p)) (insert p M)
    simp only [reads, mems, linkSteps, rebuilds, Finset.insert_union, Finset.union_insert, Finset.map_insert,
      Function.Embedding.inl_apply] at this ⊢
    exact this
  | link p k ih =>
    have := ih (D p) M
    simp only [reads, mems, linkSteps, rebuilds, Finset.insert_union]
    have := Finset.card_insert_le p ((k (D p)).reads O D ∪ M.map Function.Embedding.inl)
    omega
  | rebuild p k ih =>
    have := ih (D (.inl p)) M
    simp only [reads, mems, linkSteps, rebuilds, Finset.insert_union]
    have := Finset.card_insert_le (Sum.inl p) ((k (D (.inl p))).reads O D ∪ M.map Function.Embedding.inl)
    omega

end Chip

namespace Device

variable {N m ρ h S L q w : ℕ} {X Sig Dg : Type}

/-- The positions a run obtained: the sectors held on chip and every position it read. -/
def obtained (A : Device N m ρ h S L q w X Sig Dg) (σ : Sig) (O : Oracle N ρ h X Dg) (D : Data N m ρ X) :
    Finset (Pos N m ρ) :=
  A.held.map Function.Embedding.inl ∪ (A.prog σ D).reads O D

/-- Changing the data off the positions a run obtained changes neither its queries nor what it obtained. -/
theorem obtained_congr (A : Device N m ρ h S L q w X Sig Dg) (σ : Sig) (O : Oracle N ρ h X Dg)
    (D D' : Data N m ρ X) (hD : ∀ p ∈ A.obtained σ O D, D' p = D p) :
    (A.prog σ D').queries O D' = (A.prog σ D).queries O D ∧ A.obtained σ O D' = A.obtained σ O D := by
  have hp : A.prog σ D' = A.prog σ D := by
    unfold prog
    congr 1
    funext p
    exact hD _ (Finset.mem_union_left _ (Finset.mem_map_of_mem _ p.2))
  obtain ⟨-, h2, h3⟩ := Chip.congr O O D D' (A.prog σ D) (fun _ _ => rfl)
    (fun p hp' => hD p (Finset.mem_union_right _ hp'))
  exact ⟨by rw [hp]; exact h2, by unfold obtained; rw [hp, h3]⟩

/-- What a run obtained, in bits: at most its on-chip state, its footprint, the link and its rebuilt sectors. -/
theorem card_obtained (A : Device N m ρ h S L q w X Sig Dg) (σ : Sig) (O : Oracle N ρ h X Dg) (D : Data N m ρ X) :
    (A.obtained σ O D).card * w ≤ S + A.footprint σ O D + L + (A.prog σ D).rebuilds O D * w := by
  have h1 := Chip.card_reads_le O D (A.prog σ D) ∅
  simp only [Finset.map_empty, Finset.union_empty] at h1
  have h2 : (A.obtained σ O D).card ≤ A.held.card + ((A.prog σ D).reads O D).card := by
    unfold obtained
    exact (Finset.card_union_le _ _).trans (by rw [Finset.card_map])
  have h3 := Nat.mul_le_mul_right w (h2.trans (Nat.add_le_add_left h1 _))
  have h4 := A.held_le
  have h5 := (A.bounded σ (fun p => D (.inl p.1)) O D).1
  unfold footprint
  simp only [add_mul] at h3
  change (A.prog σ D).linkSteps O D * w ≤ L at h5
  omega

end Device

variable {N m ρ h S L q w : ℕ} {X Sig Dg : Type}

/-- The positions a tile's tag reads: the tile's sectors and the stream's. -/
def relevant (t : Fin N) : Pos N m ρ → Prop
  | .inl p => p.1 = t
  | .inr _ => True

open Classical in
/-- The tiles whose digest the device queried the oracle at, under the certified stream. -/
noncomputable def queried (A : Device N m ρ h S L q w X Sig Dg) (dig : Sig → (Fin m → X) → Dg) (σ : Sig)
    (O : Oracle N ρ h X Dg) (D : Data N m ρ X) : Finset (Fin N) :=
  Finset.univ.filter (fun t => (D.stream, t, dig σ (D.tile t)) ∈ (A.prog σ D).queries O D)

open Classical in
/-- The positions the `i`-th query's tile reads that the run never obtained. -/
noncomputable def unobtained (A : Device N m ρ h S L q w X Sig Dg) (σ : Sig) (O : Oracle N ρ h X Dg)
    (D : Data N m ρ X) (i : ℕ) : Finset (Pos N m ρ) :=
  Finset.univ.filter (fun p => p ∉ A.obtained σ O D ∧
    ∃ y, ((A.prog σ D).queries O D)[i]? = some y ∧ relevant y.2.1 p)

/-- The `i`-th query is correct for its tile, a position that tile's tag reads was never obtained, and the signs'
fiber is at most `g`. -/
def Bad [Fintype X] [DecidableEq X] [Fintype Dg] [DecidableEq Dg] (A : Device N m ρ h S L q w X Sig Dg)
    (dig : Sig → (Fin m → X) → Dg) (σ : Sig) (O : Oracle N ρ h X Dg) (D : Data N m ρ X) (i g : ℕ) : Prop :=
  fib dig σ ≤ g ∧ (unobtained A σ O D i).Nonempty ∧
    ∃ y, ((A.prog σ D).queries O D)[i]? = some y ∧ y = (D.stream, y.2.1, dig σ (D.tile y.2.1))

theorem tile_update (D : Data N m ρ X) (t : Fin N) (u : Fin m) (a : X) :
    Data.tile (Function.update D (.inl (t, u)) a) t = Function.update (D.tile t) u a := by
  funext u'
  simp [Data.tile, Function.update_apply]

theorem stream_update (D : Data N m ρ X) (r : Fin ρ) (a : X) :
    Data.stream (Function.update D (.inr r) a) = Function.update D.stream r a := by
  funext r'
  simp [Data.stream, Function.update_apply]

/-! ## Resampling an unobtained sector -/

/-- **Resampling.** If every point `D` of `G` has a direction `dir D` along which `G` and `dir` are constant, and each
such line meets `B` in at most `g` points, then `B` is at most a `g/|X|` fraction of the functions. -/
theorem card_resample {P X : Type} [Fintype P] [DecidableEq P] [Fintype X] [DecidableEq X]
    (G B : Finset (P → X)) (dir : (P → X) → P) (g : ℕ) (hBG : B ⊆ G)
    (hG : ∀ D ∈ G, ∀ a, Function.update D (dir D) a ∈ G ∧ dir (Function.update D (dir D) a) = dir D)
    (hB : ∀ D ∈ G, (Finset.univ.filter (fun a => Function.update D (dir D) a ∈ B)).card ≤ g) :
    B.card * Fintype.card X ≤ g * Fintype.card (P → X) := by
  classical
  have hback : ∀ z : (P → X) × X, z.1 ∈ B →
      dir (Function.update z.1 (dir z.1) z.2) = dir z.1 ∧
        Function.update (Function.update z.1 (dir z.1) z.2) (dir z.1) (z.1 (dir z.1)) = z.1 := fun z hz =>
    ⟨(hG z.1 (hBG hz) z.2).2, by rw [Function.update_idem, Function.update_eq_self]⟩
  have h1 : (B ×ˢ (Finset.univ : Finset X)).card ≤ g * G.card := by
    refine Finset.card_le_mul_card_image_of_maps_to (f := fun z => Function.update z.1 (dir z.1) z.2)
      (fun z hz => (hG z.1 (hBG (Finset.mem_product.1 hz).1) z.2).1) g (fun D' hD' => ?_)
    refine (Finset.card_le_card_of_injOn (fun z => z.1 (dir z.1)) (fun z hz => ?_) ?_).trans (hB D' hD')
    · simp only [Finset.coe_filter, Finset.mem_product, Finset.mem_univ, and_true, Set.mem_ofPred_eq] at hz
      obtain ⟨hzB, rfl⟩ := hz
      obtain ⟨hd, he⟩ := hback z hzB
      simp only [Finset.coe_filter, Finset.mem_univ, true_and, Set.mem_ofPred_eq, hd, he]
      exact hzB
    · intro z₁ h₁ z₂ h₂ he
      simp only [Finset.coe_filter, Finset.mem_product, Finset.mem_univ, and_true, Set.mem_ofPred_eq] at h₁ h₂
      obtain ⟨hz₁, e₁⟩ := h₁
      obtain ⟨hz₂, e₂⟩ := h₂
      obtain ⟨hd₁, hb₁⟩ := hback z₁ hz₁
      obtain ⟨hd₂, hb₂⟩ := hback z₂ hz₂
      rw [e₁] at hd₁ hb₁
      rw [e₂] at hd₂ hb₂
      have hdir : dir z₁.1 = dir z₂.1 := hd₁.symm.trans hd₂
      simp only at he
      have h1' : z₁.1 = z₂.1 := by rw [← hb₁, ← hb₂, he, hdir]
      refine Prod.ext h1' ?_
      have := congrFun (e₁.trans e₂.symm) (dir z₁.1)
      rw [Function.update_self, hdir, Function.update_self] at this
      exact this
  calc B.card * Fintype.card X = (B ×ˢ (Finset.univ : Finset X)).card := by
        rw [Finset.card_product, Finset.card_univ]
    _ ≤ g * G.card := h1
    _ ≤ g * Fintype.card (P → X) := Nat.mul_le_mul_left g (Finset.card_le_univ G)

theorem card_bad_le [Fintype X] [DecidableEq X] [Fintype Dg] [DecidableEq Dg]
    (A : Device N m ρ h S L q w X Sig Dg) (dig : Sig → (Fin m → X) → Dg) (σ : Sig) (O : Oracle N ρ h X Dg)
    (i g : ℕ) [DecidablePred (fun D : Data N m ρ X => Bad A dig σ O D i g)] :
    (Finset.univ.filter (fun D : Data N m ρ X => Bad A dig σ O D i g)).card * Fintype.card X ≤
      g * Fintype.card (Data N m ρ X) := by
  classical
  by_cases hf : fib dig σ ≤ g
  swap
  · rw [Finset.filter_false_of_mem (fun D _ hb => hf hb.1)]
    simp
  rcases isEmpty_or_nonempty (Pos N m ρ) with hP | hP
  · rw [Finset.filter_false_of_mem (fun D _ hb => by
      obtain ⟨p, -⟩ := hb.2.1
      exact IsEmpty.false p)]
    simp
  let dir : Data N m ρ X → Pos N m ρ := fun D => Classical.epsilon (· ∈ unobtained A σ O D i)
  have hdir : ∀ D, (unobtained A σ O D i).Nonempty → dir D ∈ unobtained A σ O D i := fun D hne =>
    Classical.epsilon_spec (p := (· ∈ unobtained A σ O D i)) hne
  have hsame : ∀ D, (unobtained A σ O D i).Nonempty → ∀ a,
      (A.prog σ (Function.update D (dir D) a)).queries O (Function.update D (dir D) a) =
          (A.prog σ D).queries O D ∧
        unobtained A σ O (Function.update D (dir D) a) i = unobtained A σ O D i := by
    intro D hne a
    have hp := hdir D hne
    simp only [unobtained, Finset.mem_filter, Finset.mem_univ, true_and] at hp
    obtain ⟨hq, ho⟩ := Device.obtained_congr A σ O D (Function.update D (dir D) a)
      (fun p hp' => by rw [Function.update_of_ne]; rintro rfl; exact hp.1 hp')
    refine ⟨hq, ?_⟩
    unfold unobtained
    rw [hq, ho]
  refine card_resample (Finset.univ.filter (fun D => (unobtained A σ O D i).Nonempty)) _ dir g
    (fun D hD => by
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hD ⊢
      exact hD.2.1) (fun D hD a => ?_) (fun D hD => ?_)
  · simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hD ⊢
    have := (hsame D hD a).2
    exact ⟨this ▸ hD, by simp only [dir, this]⟩
  · simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hD
    have hp := hdir D hD
    have hq := fun a => (hsame D hD a).1
    simp only [unobtained, Finset.mem_filter, Finset.mem_univ, true_and] at hp
    obtain ⟨-, y, hy, hrel⟩ := hp
    have hmem : ∀ a, Bad A dig σ O (Function.update D (dir D) a) i g →
        y = (Data.stream (Function.update D (dir D) a), y.2.1,
          dig σ (Data.tile (Function.update D (dir D) a) y.2.1)) := by
      intro a hb
      obtain ⟨-, -, y', hy', he⟩ := hb
      rw [hq a, hy] at hy'
      cases hy'
      exact he
    have h1 : 1 ≤ g := (le_max_left _ _).trans hf
    generalize dir D = p at hmem hrel
    rcases p with ⟨t, u⟩ | r
    · simp only [relevant] at hrel
      subst hrel
      calc (Finset.univ.filter (fun a => Function.update D (.inl (y.2.1, u)) a ∈
              Finset.univ.filter (fun D => Bad A dig σ O D i g))).card
          ≤ (Finset.univ.filter (fun a => dig σ (Function.update (Data.tile D y.2.1) u a) = y.2.2)).card := by
            refine Finset.card_le_card (fun a ha => ?_)
            simp only [Finset.mem_filter, Finset.mem_univ, true_and] at ha ⊢
            have := congrArg (fun z => z.2.2) (hmem a ha)
            simp only [tile_update] at this
            exact this.symm
        _ ≤ fib dig σ := (Finset.le_sup (f := fun p : (Fin m → X) × Fin m × Dg =>
              (Finset.univ.filter fun a : X => dig σ (Function.update p.1 p.2.1 a) = p.2.2).card)
              (Finset.mem_univ (Data.tile D y.2.1, u, y.2.2))).trans (le_max_right _ _)
        _ ≤ g := hf
    · calc (Finset.univ.filter (fun a => Function.update D (.inr r) a ∈
              Finset.univ.filter (fun D => Bad A dig σ O D i g))).card
          ≤ ({y.1 r} : Finset X).card := by
            refine Finset.card_le_card (fun a ha => ?_)
            simp only [Finset.mem_filter, Finset.mem_univ, true_and] at ha
            have := congrFun (congrArg (fun z => z.1) (hmem a ha)) r
            simp only [stream_update, Function.update_self] at this
            simp [this]
        _ ≤ g := by simpa using h1

/-! ## A guessed tag -/

theorem card_guess_le (A : Device N m ρ h S L q w X Sig Dg) (dig : Sig → (Fin m → X) → Dg) (σ : Sig)
    (D : Data N m ρ X) (j : Fin N) [Fintype X] [DecidableEq X] [Fintype Dg] [DecidableEq Dg]
    [DecidablePred (fun O : Oracle N ρ h X Dg =>
      j ∉ queried A dig σ O D ∧ A.tags σ O D j = O (D.stream, j, dig σ (D.tile j)))] :
    (Finset.univ.filter (fun O : Oracle N ρ h X Dg =>
        j ∉ queried A dig σ O D ∧ A.tags σ O D j = O (D.stream, j, dig σ (D.tile j)))).card * 2 ^ h ≤
      Fintype.card (Oracle N ρ h X Dg) := by
  classical
  have hY : Fintype.card (Bits h) = 2 ^ h := by simp
  rw [← hY]
  refine card_line_le (D.stream, j, dig σ (D.tile j)) _ (fun O₁ h₁ O₂ h₂ hoff => ?_)
  simp only [Finset.mem_filter, Finset.mem_univ, true_and, queried] at h₁ h₂
  have hq : ∀ x ∈ (A.prog σ D).queries O₁ D, O₂ x = O₁ x :=
    fun x hx => (hoff x (fun he => h₁.1 (he ▸ hx))).symm
  have ht : A.tags σ O₂ D j = A.tags σ O₁ D j :=
    congrFun (Chip.congr O₁ O₂ D D (A.prog σ D) hq (fun _ _ => rfl)).1 j
  funext x
  by_cases hx : x = (D.stream, j, dig σ (D.tile j))
  · subst hx
    rw [← h₁.2, ← h₂.2, ht]
  · exact hoff x hx

/-! ## Every queried tile obtained whole -/

theorem floor_le_of_complete (A : Device N m ρ h S L q w X Sig Dg) (dig : Sig → (Fin m → X) → Dg) (σ : Sig)
    (O : Oracle N ρ h X Dg) (D : Data N m ρ X) {k F cTag cEnc : ℕ} (hk : 0 < k)
    (hkq : k ≤ (queried A dig σ O D).card)
    (hcomp : ∀ t ∈ queried A dig σ O D, ∀ p, relevant t p → p ∈ A.obtained σ O D)
    (hc : A.compute cTag cEnc σ O D ≤ F) :
    floor m w ρ S L F cTag cEnc k ≤ A.footprint σ O D * cEnc := by
  classical
  obtain ⟨K, hK, hKc⟩ := Finset.exists_subset_card_eq hkq
  obtain ⟨t₀, ht₀⟩ : K.Nonempty := Finset.card_pos.1 (hKc ▸ hk)
  have hsub : (K ×ˢ (Finset.univ : Finset (Fin m))).disjSum (Finset.univ : Finset (Fin ρ)) ⊆
      A.obtained σ O D := by
    rintro (⟨t, u⟩ | r) hp
    · simp only [Finset.inl_mem_disjSum, Finset.mem_product, Finset.mem_univ, and_true] at hp
      exact hcomp t (hK hp) _ rfl
    · exact hcomp t₀ (hK ht₀) _ trivial
  have hcard := Finset.card_le_card hsub
  rw [Finset.card_disjSum, Finset.card_product, Finset.card_univ, Fintype.card_fin, Finset.card_univ,
    Fintype.card_fin, hKc] at hcard
  have hob := A.card_obtained σ O D
  have hQ : k ≤ ((A.prog σ D).queries O D).length := by
    rw [← hKc]
    refine (Finset.card_le_card_of_injOn (fun t => (D.stream, t, dig σ (D.tile t))) (fun t ht => ?_) ?_).trans
      (List.toFinset_card_le _)
    · have := hK ht
      simp only [queried, Finset.mem_filter, Finset.mem_univ, true_and] at this
      simpa using this
    · intro t₁ _ t₂ _ he
      exact (Prod.mk.inj (Prod.mk.inj he).2).1
  unfold Device.compute at hc
  unfold floor
  generalize ((A.prog σ D).queries O D).length = Qn at hQ hc
  generalize (A.prog σ D).rebuilds O D = G at hc hob
  generalize A.footprint σ O D = fp at hob ⊢
  generalize (A.obtained σ O D).card = ob at hcard hob
  have h1 : G * cEnc ≤ F - k * cTag := by
    have := Nat.mul_le_mul_right cTag hQ
    omega
  have h2 : G * w * cEnc ≤ (F - k * cTag) * w :=
    calc G * w * cEnc = G * cEnc * w := by ring
      _ ≤ (F - k * cTag) * w := Nat.mul_le_mul_right w h1
  have h3 : k * m * w + ρ * w ≤ S + fp + L + G * w := by
    have := Nat.mul_le_mul_right w hcard
    rw [add_mul] at this
    omega
  have h4 : k * (m * w) ≤ S + (L - ρ * w) + fp + G * w := by
    rw [← mul_assoc]
    omega
  have h5 := Nat.mul_le_mul_right cEnc h4
  have e : (S + (L - ρ * w) + fp + G * w) * cEnc = (S + (L - ρ * w)) * cEnc + fp * cEnc + G * w * cEnc := by
    ring
  omega

end Accounting.RateSketch

namespace Accounting.SecurityProofs

open Accounting.RateSketch
open Pous (Bits pr)
open scoped ENNReal
open Bandwidth (card_filter_prod_le card_filter_prod_le' pr_le_of_card pr_mono pr_or_le pr_exists_le
  pr_eq_zero_of_card)
open Bandwidth.AuditRO (card_sample)

/-- **The rate audit with a sketch-then-hash tag** (`Accounting.Guarantees.RateAuditSketch`). -/
theorem RateAuditSketch : Accounting.Guarantees.RateAuditSketch := by
  classical
  intro N m ρ h S L q w s k F cTag cEnc g X Sig Dg _ _ _ _ _ dig A
  rcases Nat.eq_zero_or_pos k with rfl | hk0
  · refine (pr_le_of_card _ 1 0 one_ne_zero ?_).trans (by simp)
    rw [Finset.filter_false_of_mem (fun x _ hx => by simp [floor] at hx), Finset.card_empty, zero_mul, zero_mul]
  have hsplit : ∀ x : Data N m ρ X × Coins N ρ h s X Sig Dg,
      (A.Accepts dig x.2.1 x.2.2.1 x.1 x.2.2.2 ∧ A.compute cTag cEnc x.2.1 x.2.2.1 x.1 ≤ F ∧
        A.footprint x.2.1 x.2.2.1 x.1 * cEnc < floor m w ρ S L F cTag cEnc k) →
      ((((queried A dig x.2.1 x.2.2.1 x.1).card < k ∧ ∀ i, x.2.2.2 i ∈ queried A dig x.2.1 x.2.2.1 x.1) ∨
        (∃ i, x.2.2.2 i ∉ queried A dig x.2.1 x.2.2.1 x.1 ∧
          A.tags x.2.1 x.2.2.1 x.1 (x.2.2.2 i) =
            x.2.2.1 (x.1.stream, x.2.2.2 i, dig x.2.1 (x.1.tile (x.2.2.2 i))))) ∨
        g < fib dig x.2.1) ∨
      ∃ i : Fin q, Bad A dig x.2.1 x.2.2.1 x.1 i g := by
    rintro ⟨D, σ, O, I⟩ ⟨hacc, hc, hf⟩
    dsimp only at hacc hc hf ⊢
    by_cases hfib : fib dig σ ≤ g
    swap
    · exact Or.inl (Or.inr (not_le.1 hfib))
    by_cases hcomp : ∀ t ∈ queried A dig σ O D, ∀ p, relevant t p → p ∈ A.obtained σ O D
    · by_cases hk : k ≤ (queried A dig σ O D).card
      · exact absurd (floor_le_of_complete A dig σ O D hk0 hk hcomp hc) (not_le.2 hf)
      · by_cases hall : ∀ i, I i ∈ queried A dig σ O D
        · exact Or.inl (Or.inl (Or.inl ⟨by omega, hall⟩))
        · push Not at hall
          obtain ⟨i, hi⟩ := hall
          exact Or.inl (Or.inl (Or.inr ⟨i, hi, hacc i⟩))
    · push Not at hcomp
      obtain ⟨t, ht, p, hrel, hp⟩ := hcomp
      have hmem : (D.stream, t, dig σ (D.tile t)) ∈ (A.prog σ D).queries O D := by
        simpa [queried] using ht
      have hlt := List.idxOf_lt_length_of_mem hmem
      have hget : ((A.prog σ D).queries O D)[((A.prog σ D).queries O D).idxOf (D.stream, t, dig σ (D.tile t))]? =
          some (D.stream, t, dig σ (D.tile t)) := by
        rw [List.getElem?_eq_getElem hlt, List.getElem_idxOf hlt]
      refine Or.inr ⟨⟨_, hlt.trans_le (A.bounded σ _ O D).2⟩, hfib, ⟨p, ?_⟩, _, hget, rfl⟩
      simp only [unobtained, Finset.mem_filter, Finset.mem_univ, true_and]
      exact ⟨hp, _, hget, hrel⟩
  have hC : Fintype.card (Coins N ρ h s X Sig Dg) =
      Fintype.card Sig * (Fintype.card (Oracle N ρ h X Dg) * N ^ s) := by
    rw [Fintype.card_prod, Fintype.card_prod]
    simp
  have hΩ : Fintype.card (Data N m ρ X × Coins N ρ h s X Sig Dg) =
      Fintype.card (Data N m ρ X) * (Fintype.card Sig * (Fintype.card (Oracle N ρ h X Dg) * N ^ s)) := by
    rw [Fintype.card_prod, hC]
  by_cases hN : N ^ s = 0
  · rw [pr_eq_zero_of_card _ (by rw [hΩ, hN]; simp)]
    exact bot_le
  refine (pr_mono hsplit).trans ?_
  refine (pr_or_le _ _).trans (add_le_add ((pr_or_le _ _).trans
    (add_le_add ((pr_or_le _ _).trans (add_le_add ?_ ?_)) ?_)) ?_)
  · -- the sample misses the tiles never queried
    have hc := pr_le_of_card (fun x : Data N m ρ X × Coins N ρ h s X Sig Dg =>
        (queried A dig x.2.1 x.2.2.1 x.1).card < k ∧ ∀ i, x.2.2.2 i ∈ queried A dig x.2.1 x.2.2.1 x.1)
        (N ^ s) ((k - 1) ^ s) hN (by
      rw [hΩ]
      refine (card_filter_prod_le _ _
        (Fintype.card Sig * (Fintype.card (Oracle N ρ h X Dg) * ((k - 1) ^ s * N ^ s)))
        (fun D => ?_)).trans (le_of_eq (by ring))
      refine card_filter_prod_le _ _ _ (fun σ => ?_)
      refine card_filter_prod_le _ _ _ (fun O => ?_)
      by_cases hk : (queried A dig σ O D).card < k
      · simp only [hk, true_and]
        rw [card_sample]
        exact Nat.mul_le_mul_right _ (Nat.pow_le_pow_left (by omega) _)
      · simp [hk])
    refine hc.trans (le_of_eq ?_)
    rw [Nat.cast_pow, Nat.cast_pow, div_eq_mul_inv, div_eq_mul_inv, mul_pow, ENNReal.inv_pow]
  · -- a guessed tag
    refine (pr_exists_le _).trans ?_
    have hi : ∀ i : Fin s, pr (fun x : Data N m ρ X × Coins N ρ h s X Sig Dg =>
        x.2.2.2 i ∉ queried A dig x.2.1 x.2.2.1 x.1 ∧
          A.tags x.2.1 x.2.2.1 x.1 (x.2.2.2 i) =
            x.2.2.1 (x.1.stream, x.2.2.2 i, dig x.2.1 (x.1.tile (x.2.2.2 i)))) ≤
        ((1 : ℕ) : ℝ≥0∞) / ((2 ^ h : ℕ) : ℝ≥0∞) := fun i =>
      pr_le_of_card _ (2 ^ h) 1 (by positivity) (by
        rw [hΩ, one_mul]
        refine (card_filter_prod_le _ _
          (Fintype.card Sig * (Fintype.card (Oracle N ρ h X Dg) * Fintype.card (Fin s → Fin N)))
          (fun D => ?_)).trans (le_of_eq (by simp))
        refine card_filter_prod_le _ _ _ (fun σ => ?_)
        refine card_filter_prod_le' _ _ _ (fun I => ?_)
        exact card_guess_le A dig σ D (I i))
    refine (Finset.sum_le_sum (fun i _ => hi i)).trans (le_of_eq ?_)
    simp [div_eq_mul_inv]
  · -- the signs' fiber
    by_cases hSig : Fintype.card Sig = 0
    · rw [pr_eq_zero_of_card _ (by rw [hΩ, hSig]; simp)]
      exact bot_le
    refine (pr_le_of_card _ (Fintype.card Sig) (Finset.univ.filter (fun σ : Sig => g < fib dig σ)).card hSig
      ?_).trans (le_of_eq ?_)
    · rw [hΩ]
      refine (card_filter_prod_le _ _
        ((Finset.univ.filter (fun σ : Sig => g < fib dig σ)).card * Fintype.card Sig *
          Fintype.card (Oracle N ρ h X Dg × (Fin s → Fin N))) (fun D => ?_)).trans
        (le_of_eq (by simp only [Fintype.card_prod, Fintype.card_fun, Fintype.card_fin]; ring))
      exact card_filter_prod_le' _ _ _ (fun y => le_rfl)
    · unfold pr
      convert rfl
  · -- a correct query for a tile not obtained whole
    refine (pr_exists_le _).trans ?_
    have hi : ∀ i : Fin q, pr (fun x : Data N m ρ X × Coins N ρ h s X Sig Dg =>
        Bad A dig x.2.1 x.2.2.1 x.1 i g) ≤ ((g : ℕ) : ℝ≥0∞) / ((Fintype.card X : ℕ) : ℝ≥0∞) := by
      intro i
      by_cases hX : Fintype.card X = 0
      · refine (pr_le_of_card _ 1 0 one_ne_zero ?_).trans (by simp)
        rw [Finset.filter_false_of_mem (fun x _ hb => by
          obtain ⟨p, -⟩ := hb.2.1
          exact (@Fintype.card_ne_zero X _ ⟨x.1 p⟩) hX), Finset.card_empty, zero_mul, zero_mul]
      refine pr_le_of_card _ (Fintype.card X) g hX ?_
      rw [hΩ]
      refine (card_filter_prod_le' _ _ (g * Fintype.card (Data N m ρ X)) (fun y => ?_)).trans
        (le_of_eq (by rw [hC]; ring))
      exact card_bad_le A dig y.1 y.2.1 i g
    refine (Finset.sum_le_sum (fun i _ => hi i)).trans (le_of_eq ?_)
    simp [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, mul_div_assoc]

/-- An INT8 is at most `128` in absolute value. -/
theorem abs_int8_le (a : Fin 256) : |int8 a| ≤ 128 := by
  unfold int8
  have := a.isLt
  split_ifs <;> rw [abs_le] <;> constructor <;> omega

/-- One coordinate of a sector's projection is at most `32·128` in absolute value. -/
theorem abs_project_le {k : ℕ} (T : Accounting.Tag.Signs k 32) (a : Sector) (i : Fin k) :
    |Accounting.Tag.project T (fun l => int8 (a l)) i| ≤ 4096 := by
  unfold Accounting.Tag.project
  calc |∑ j, Accounting.Tag.sgn (T i j) * int8 (a j)| ≤ ∑ j, |Accounting.Tag.sgn (T i j) * int8 (a j)| :=
        Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _j : Fin 32, (128 : ℤ) := Finset.sum_le_sum fun j _ => by
        rw [abs_mul]
        have hs : |Accounting.Tag.sgn (T i j)| = 1 := by
          unfold Accounting.Tag.sgn
          split_ifs <;> simp
        rw [hs, one_mul]
        exact abs_int8_le _
    _ = 4096 := by simp

set_option linter.constructorNameAsVariable false in
/-- **Binding signs make the tensor-core sketch's fiber `1`** (`Accounting.Guarantees.SketchBinds`). -/
theorem SketchBinds : Accounting.Guarantees.SketchBinds := by
  intro k m σ hσ
  unfold fib
  refine max_le le_rfl (Finset.sup_le fun p _ => ?_)
  refine Finset.card_le_one.2 fun a ha b hb => ?_
  simp only [Finset.mem_filter, Finset.mem_univ, true_and] at ha hb
  obtain ⟨x, u, d⟩ := p
  dsimp only at ha hb
  apply hσ u
  refine funext fun i => ?_
  have := congrFun (ha.trans hb.symm) i
  simp only [sketch] at this
  have hrest : ∀ c : Sector,
      ∑ u' ∈ Finset.univ.erase u, Accounting.Tag.project (σ.block u') (fun l => int8 (Function.update x u c u' l)) i =
        ∑ u' ∈ Finset.univ.erase u, Accounting.Tag.project (σ.block u') (fun l => int8 (x u' l)) i := fun c =>
    Finset.sum_congr rfl (fun u' hu' => by rw [Function.update_of_ne (Finset.ne_of_mem_erase hu')])
  rw [← Finset.add_sum_erase _ _ (Finset.mem_univ u), ← Finset.add_sum_erase _ _ (Finset.mem_univ u), hrest,
    hrest, Function.update_self, Function.update_self] at this
  push_cast at this
  have h2 := add_right_cancel this
  rw [ZMod.intCast_eq_intCast_iff_dvd_sub] at h2
  have ha' := abs_project_le (σ.block u) a i
  have hb' := abs_project_le (σ.block u) b i
  have hlt : |Accounting.Tag.project (σ.block u) (fun l => int8 (b l)) i -
      Accounting.Tag.project (σ.block u) (fun l => int8 (a l)) i| < ((2 ^ 32 : ℕ) : ℤ) := by
    rw [abs_lt]
    rw [abs_le] at ha' hb'
    constructor <;> push_cast <;> linarith [ha'.1, ha'.2, hb'.1, hb'.2]
  exact (sub_eq_zero.1 (Int.eq_zero_of_abs_lt_dvd h2 hlt)).symm

/-- **The rate audit at the deployed sketch-then-hash tag** (`Accounting.Guarantees.RateAuditTensorSketch`). -/
theorem RateAuditTensorSketch : Accounting.Guarantees.RateAuditTensorSketch := by
  intro N m ρ h S L q s k F cTag cEnc ks A
  refine (RateAuditSketch N m ρ h S L q 256 s k F cTag cEnc 1 Sector (SketchSigns ks m)
    (Fin ks → ZMod (2 ^ 32)) sketch A).trans ?_
  refine add_le_add (add_le_add le_rfl (pr_mono fun σ hlt hb => absurd (SketchBinds ks m σ hb) (not_le.2 hlt)))
    (le_of_eq ?_)
  have : Fintype.card Sector = 2 ^ 256 := by simp [Sector]
  rw [this]
  norm_num

end Accounting.SecurityProofs
