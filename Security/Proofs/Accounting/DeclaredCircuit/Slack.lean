import Mathlib
import Proofs.Accounting.DeclaredCircuit.Tiles

/-!
# The slack's price: `k + 1` unobtained positions behind correct digests

The pieces of `Accounting.SecurityProofs.DeclaredCircuitRequirement` that price `BadK`. Resampling the `K` unobtained
positions of a run together leaves its queries and what it obtained unchanged, so each correct digest query pins its
position to the digest's fiber, at most `fib` values, and a stream position to one value (`card_resample_planes`,
`Prover.card_badK_ground_le`). Summed over the candidates of a pass and their `K`-tuples of query indices, a shared
budget of `qtot` queries pays `(fib·qtot/|X|)^K` (`Prover.card_badK_family_le`), which on the sketch at rank `r` is
`(qtot/2^(8r))^K` (`Prover.pr_badK_family_le`).
-/

namespace Accounting.RateSketch

open Pous (pr)
open scoped ENNReal

/-- Extending twice along an injection keeps only the last values on its image. -/
theorem extend_extend {K P X : Type} {d : K → P} (hd : Function.Injective d) (a a' : K → X) (D : P → X) :
    Function.extend d a' (Function.extend d a D) = Function.extend d a' D := by
  funext p
  by_cases hp : ∃ i, d i = p
  · obtain ⟨i, rfl⟩ := hp
    rw [hd.extend_apply a' (Function.extend d a D) i, hd.extend_apply a' D i]
  · rw [Function.extend_apply' a' (Function.extend d a D) p hp, Function.extend_apply' a D p hp,
      Function.extend_apply' a' D p hp]

/-- Extending a function along an injection by its own values is the function. -/
theorem extend_comp_self {K P X : Type} {d : K → P} (hd : Function.Injective d) (D : P → X) :
    Function.extend d (D ∘ d) D = D := by
  funext p
  by_cases hp : ∃ i, d i = p
  · obtain ⟨i, rfl⟩ := hp
    rw [hd.extend_apply (D ∘ d) D i, Function.comp_apply]
  · rw [Function.extend_apply' (D ∘ d) D p hp]

/-- **Resampling `K` coordinates at once, counted against its ground set.** `card_resample_ground` with a plane for a
line: if every point of `G` has `K` distinct directions along which `G` and the directions are constant, and each such
plane meets `B` in at most `b` points, then `|B|·|X|^K ≤ b·|G|`. -/
theorem card_resample_planes {K P X : Type} [Fintype K] [DecidableEq K] [Fintype P] [DecidableEq P] [Fintype X]
    [DecidableEq X] (G B : Finset (P → X)) (dir : (P → X) → K → P) (b : ℕ) (hBG : B ⊆ G)
    (hinj : ∀ D ∈ G, Function.Injective (dir D))
    (hG : ∀ D ∈ G, ∀ a : K → X, Function.extend (dir D) a D ∈ G ∧ dir (Function.extend (dir D) a D) = dir D)
    (hB : ∀ D ∈ G, (Finset.univ.filter fun a : K → X => Function.extend (dir D) a D ∈ B).card ≤ b) :
    B.card * Fintype.card (K → X) ≤ b * G.card := by
  classical
  have hback : ∀ z : (P → X) × (K → X), z.1 ∈ B →
      dir (Function.extend (dir z.1) z.2 z.1) = dir z.1 ∧
        Function.extend (dir z.1) (z.1 ∘ dir z.1) (Function.extend (dir z.1) z.2 z.1) = z.1 ∧
        Function.extend (dir z.1) z.2 z.1 ∘ dir z.1 = z.2 := fun z hz =>
    ⟨(hG z.1 (hBG hz) z.2).2,
      by rw [extend_extend (hinj z.1 (hBG hz)), extend_comp_self (hinj z.1 (hBG hz))],
      Function.extend_comp (hinj z.1 (hBG hz)) z.2 z.1⟩
  have h1 : (B ×ˢ (Finset.univ : Finset (K → X))).card ≤ b * G.card := by
    refine Finset.card_le_mul_card_image_of_maps_to (f := fun z => Function.extend (dir z.1) z.2 z.1)
      (fun z hz => (hG z.1 (hBG (Finset.mem_product.1 hz).1) z.2).1) b (fun D' hD' => ?_)
    refine (Finset.card_le_card_of_injOn (fun z => z.1 ∘ dir z.1) (fun z hz => ?_) ?_).trans (hB D' hD')
    · simp only [Finset.coe_filter, Finset.mem_product, Finset.mem_univ, and_true, Set.mem_ofPred_eq] at hz
      obtain ⟨hzB, rfl⟩ := hz
      obtain ⟨hd, he, -⟩ := hback z hzB
      simp only [Finset.coe_filter, Finset.mem_univ, true_and, Set.mem_ofPred_eq, hd, he]
      exact hzB
    · intro z₁ h₁ z₂ h₂ he
      simp only [Finset.coe_filter, Finset.mem_product, Finset.mem_univ, and_true, Set.mem_ofPred_eq] at h₁ h₂
      obtain ⟨hz₁, e₁⟩ := h₁
      obtain ⟨hz₂, e₂⟩ := h₂
      obtain ⟨hd₁, hb₁, hc₁⟩ := hback z₁ hz₁
      obtain ⟨hd₂, hb₂, hc₂⟩ := hback z₂ hz₂
      rw [e₁] at hd₁ hb₁ hc₁
      rw [e₂] at hd₂ hb₂ hc₂
      have hdir : dir z₁.1 = dir z₂.1 := hd₁.symm.trans hd₂
      have he' : z₁.1 ∘ dir z₁.1 = z₂.1 ∘ dir z₂.1 := he
      have h1' : z₁.1 = z₂.1 := by rw [← hb₁, ← hb₂, he', hdir]
      refine Prod.ext h1' ?_
      rw [← hc₁, ← hc₂, hdir]
  calc B.card * Fintype.card (K → X) = (B ×ˢ (Finset.univ : Finset (K → X))).card := by
        rw [Finset.card_product, Finset.card_univ]
    _ ≤ b * G.card := h1

variable {N m ρ h S L q w : ℕ} {X Sig Dg : Type}

/-- Tile `t` of data extended at spread positions, one of them `(t, u)`: the tile with sector `u` replaced. -/
theorem tile_extend {K : ℕ} {d : Fin K → Pos N m ρ} (hd : Spread d) (a : Fin K → X) (D : Data N m ρ X)
    {i : Fin K} {t : Fin N} {u : Fin m} (hi : d i = .inl (t, u)) :
    Data.tile (Function.extend d a D) t = Function.update (D.tile t) u (a i) := by
  funext u'
  show Function.extend d a D (.inl (t, u')) = Function.update (D.tile t) u (a i) u'
  by_cases hu : u' = u
  · subst hu
    rw [Function.update_self, ← hi, hd.1.extend_apply a D i]
  · rw [Function.update_of_ne hu]
    refine Function.extend_apply' _ _ _ ?_
    rintro ⟨i', hi'⟩
    have hii := hd.2 i i' (t, u) (t, u') hi hi' rfl
    subst hii
    rw [hi] at hi'
    exact hu (congrArg Prod.snd (Sum.inl_injective hi')).symm

/-- The stream of data extended at injective positions, one of them stream sector `r`. -/
theorem stream_extend {K : ℕ} {d : Fin K → Pos N m ρ} (hd : Function.Injective d) (a : Fin K → X)
    (D : Data N m ρ X) {i : Fin K} {r : Fin ρ} (hi : d i = .inr r) :
    Data.stream (Function.extend d a D) r = a i := by
  show Function.extend d a D (.inr r) = a i
  rw [← hi]
  exact hd.extend_apply a D i

/-- The values a position may take for a correct digest query `y` of its tile: the sectors keeping the tile at `y`'s
digest, or the stream's sector itself. -/
def coordSet [Fintype X] [DecidableEq Dg] (dig : Sig → (Fin m → X) → Dg) (σ : Sig) (D : Data N m ρ X)
    (y : (Fin ρ → X) × Fin N × Dg) : Pos N m ρ → Finset X
  | .inl (t, u) => Finset.univ.filter fun v => dig σ (Function.update (D.tile t) u v) = y.2.2
  | .inr r => {y.1 r}

theorem card_coordSet_le [Fintype X] [DecidableEq X] [Fintype Dg] [DecidableEq Dg]
    (dig : Sig → (Fin m → X) → Dg) (σ : Sig) (D : Data N m ρ X) (y : (Fin ρ → X) × Fin N × Dg)
    (p : Pos N m ρ) : (coordSet dig σ D y p).card ≤ fib dig σ := by
  rcases p with ⟨t, u⟩ | r
  · simp only [coordSet]
    exact (Finset.le_sup (f := fun p : (Fin m → X) × Fin m × Dg =>
      (Finset.univ.filter fun a : X => dig σ (Function.update p.1 p.2.1 a) = p.2.2).card)
      (Finset.mem_univ (D.tile t, u, y.2.2))).trans (le_max_right _ _)
  · simp only [coordSet, Finset.card_singleton]
    exact le_max_left _ _

/-- The query-index tuples below a length: at most `len^K`. -/
theorem card_idx_le (q len K : ℕ) :
    (Finset.univ.filter fun ι : Fin K → Fin q => ∀ i, (ι i : ℕ) < len).card ≤ len ^ K := by
  classical
  have hsub : (Finset.univ.filter fun ι : Fin K → Fin q => ∀ i, (ι i : ℕ) < len) ⊆
      Fintype.piFinset fun _ : Fin K => Finset.univ.filter fun j : Fin q => (j : ℕ) < len := by
    intro ι hι
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hι
    simp only [Fintype.mem_piFinset, Finset.mem_filter, Finset.mem_univ, true_and]
    exact hι
  refine (Finset.card_le_card hsub).trans ?_
  rw [Fintype.card_piFinset_const]
  refine Nat.pow_le_pow_left ?_ K
  calc (Finset.univ.filter fun j : Fin q => (j : ℕ) < len).card ≤ (Finset.range len).card :=
        Finset.card_le_card_of_injOn (fun j => (j : ℕ)) (fun j hj => by
          rw [Finset.mem_coe, Finset.mem_filter] at hj
          rw [Finset.mem_coe, Finset.mem_range]
          exact hj.2) Fin.val_injective.injOn
    _ = len := Finset.card_range _

/-- A sum of `(k+1)`-th powers is at most the `(k+1)`-th power of the sum. -/
theorem sum_pow_succ_le {ι : Type} (s : Finset ι) (f : ι → ℕ) (k : ℕ) :
    ∑ i ∈ s, f i ^ (k + 1) ≤ (∑ i ∈ s, f i) ^ (k + 1) := by
  calc ∑ i ∈ s, f i ^ (k + 1) = ∑ i ∈ s, f i ^ k * f i := by simp only [pow_succ]
    _ ≤ ∑ i ∈ s, (∑ j ∈ s, f j) ^ k * f i := Finset.sum_le_sum fun i hi =>
        Nat.mul_le_mul_right _ (Nat.pow_le_pow_left (Finset.single_le_sum (fun j _ => Nat.zero_le (f j)) hi) k)
    _ = (∑ j ∈ s, f j) ^ k * ∑ i ∈ s, f i := by rw [Finset.mul_sum]
    _ = (∑ i ∈ s, f i) ^ (k + 1) := by rw [pow_succ]

namespace Prover

/-- **One tuple of query indices, counted against the data where they exist.** The data on which a run made correct
digest queries at the indices `ι` behind `K` unobtained spread positions are at most a `(g/|X|)^K` fraction of the data
on which all `K` indices exist. -/
theorem card_badK_ground_le [Fintype X] [DecidableEq X] [Fintype Dg] [DecidableEq Dg] {K : ℕ}
    (A : Prover N m ρ h S L q w X Sig Dg) (dig : Sig → (Fin m → X) → Dg) (σ : Sig) (O : Oracle N ρ h X Dg)
    (ι : Fin K → ℕ) (g : ℕ) (hf : fib dig σ ≤ g)
    [DecidablePred (fun D : Data N m ρ X => BadK A dig σ O D ι)] :
    (Finset.univ.filter fun D : Data N m ρ X => BadK A dig σ O D ι).card * Fintype.card (Fin K → X) ≤
      g ^ K * (Finset.univ.filter fun D : Data N m ρ X =>
        ∀ i, ι i < ((A.prog σ D).queries O D).length).card := by
  classical
  rcases isEmpty_or_nonempty (Fin K → Pos N m ρ) with hP | hP
  · rw [Finset.filter_false_of_mem (fun D _ hb => by
      obtain ⟨p, -⟩ := hb.1
      exact IsEmpty.false p)]
    simp
  obtain ⟨dir, hdirdef⟩ : ∃ dir : Data N m ρ X → Fin K → Pos N m ρ, ∀ D, dir D =
      Classical.epsilon (Unobtained ((A.prog σ D).queries O D) (A.obtained σ O D) ι) := ⟨_, fun _ => rfl⟩
  have hdir : ∀ D, (∃ p, Unobtained ((A.prog σ D).queries O D) (A.obtained σ O D) ι p) →
      Unobtained ((A.prog σ D).queries O D) (A.obtained σ O D) ι (dir D) := fun D hne => by
    rw [hdirdef]
    exact Classical.epsilon_spec hne
  have hsame : ∀ D, (∃ p, Unobtained ((A.prog σ D).queries O D) (A.obtained σ O D) ι p) → ∀ a : Fin K → X,
      (A.prog σ (Function.extend (dir D) a D)).queries O (Function.extend (dir D) a D) =
          (A.prog σ D).queries O D ∧
        A.obtained σ O (Function.extend (dir D) a D) = A.obtained σ O D := by
    intro D hne a
    have hp := hdir D hne
    refine Prover.obtained_congr A σ O D (Function.extend (dir D) a D) (fun p hp' => ?_)
    refine Function.extend_apply' _ _ _ ?_
    rintro ⟨i, rfl⟩
    exact (hp.2 i).1 hp'
  have hground : (Finset.univ.filter fun D : Data N m ρ X =>
        ∃ p, Unobtained ((A.prog σ D).queries O D) (A.obtained σ O D) ι p) ⊆
      Finset.univ.filter fun D : Data N m ρ X => ∀ i, ι i < ((A.prog σ D).queries O D).length := by
    intro D hD
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hD ⊢
    obtain ⟨p, hp⟩ := hD
    intro i
    obtain ⟨-, y, hy, -⟩ := hp.2 i
    exact (List.getElem?_eq_some_iff.1 hy).1
  refine le_trans (card_resample_planes
    (Finset.univ.filter fun D : Data N m ρ X =>
      ∃ p, Unobtained ((A.prog σ D).queries O D) (A.obtained σ O D) ι p)
    (Finset.univ.filter fun D : Data N m ρ X => BadK A dig σ O D ι) dir (g ^ K)
    (fun D hD => by
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hD ⊢
      exact hD.1)
    (fun D hD => by
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hD
      exact (hdir D hD).1.1)
    (fun D hD a => ?_) (fun D hD => ?_))
    (Nat.mul_le_mul_left _ (Finset.card_le_card hground))
  · simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hD ⊢
    obtain ⟨hq, ho⟩ := hsame D hD a
    refine ⟨?_, ?_⟩
    · rw [hq, ho]
      exact hD
    · rw [hdirdef (Function.extend (dir D) a D), hq, ho, ← hdirdef D]
  · simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hD
    have hp := hdir D hD
    choose y hy hrel using fun i => (hp.2 i).2
    refine (Finset.card_le_card (t := Fintype.piFinset fun i => coordSet dig σ D (y i) (dir D i))
      fun a ha => ?_).trans ?_
    · simp only [Finset.mem_filter, Finset.mem_univ, true_and] at ha
      obtain ⟨hq, -⟩ := hsame D hD a
      rw [Fintype.mem_piFinset]
      intro i
      obtain ⟨y', hy', he⟩ := ha.2 i
      rw [hq, hy i] at hy'
      obtain rfl := Option.some.inj hy'
      have hr := hrel i
      rcases hdi : dir D i with ⟨t, u⟩ | r
      · rw [hdi] at hr
        have ht : t = (y i).2.1 := hr
        have h2 := congrArg (fun z => z.2.2) he
        simp only at h2
        rw [← ht, tile_extend hp.1 a D hdi] at h2
        simp only [coordSet, Finset.mem_filter, Finset.mem_univ, true_and]
        exact h2.symm
      · have h1 := congrFun (congrArg (fun z => z.1) he) r
        simp only at h1
        rw [stream_extend hp.1.1 a D hdi] at h1
        simp only [coordSet, Finset.mem_singleton]
        exact h1.symm
    · rw [Fintype.card_piFinset]
      refine (Finset.prod_le_pow_card _ _ g fun i _ =>
        (card_coordSet_le dig σ D (y i) (dir D i)).trans hf).trans (le_of_eq ?_)
      rw [Finset.card_univ, Fintype.card_fin]

/-- **The budgeted union of `K`-tuples.** A family of `C` candidate devices on shared data, each with its own oracle:
if across the family the queries total at most `qtot` on every data point, then the data on which some candidate made
correct digest queries behind `k + 1` unobtained spread positions are at most a `(g·qtot/|X|)^(k+1)` fraction. Each
candidate's `(k+1)`-tuples of its queries are at most its query count to the `k + 1`, and those add to at most
`qtot^(k+1)`. -/
theorem card_badK_family_le [Fintype X] [DecidableEq X] [Fintype Dg] [DecidableEq Dg] {C k : ℕ}
    (A : Fin C → Prover N m ρ h S L q w X Sig Dg) (dig : Sig → (Fin m → X) → Dg) (σ : Sig)
    (O : Fin C → Oracle N ρ h X Dg) (g qtot : ℕ) (hf : fib dig σ ≤ g)
    (hq : ∀ D : Data N m ρ X, ∑ c, (((A c).prog σ D).queries (O c) D).length ≤ qtot)
    [DecidablePred (fun D : Data N m ρ X =>
      ∃ c, ∃ ι : Fin (k + 1) → Fin q, BadK (A c) dig σ (O c) D fun i => (ι i : ℕ))] :
    (Finset.univ.filter fun D : Data N m ρ X =>
        ∃ c, ∃ ι : Fin (k + 1) → Fin q, BadK (A c) dig σ (O c) D fun i => (ι i : ℕ)).card *
        Fintype.card (Fin (k + 1) → X) ≤ (g * qtot) ^ (k + 1) * Fintype.card (Data N m ρ X) := by
  classical
  have hsub : (Finset.univ.filter fun D : Data N m ρ X =>
        ∃ c, ∃ ι : Fin (k + 1) → Fin q, BadK (A c) dig σ (O c) D fun i => (ι i : ℕ)).card ≤
      ∑ cι : Fin C × (Fin (k + 1) → Fin q),
        (Finset.univ.filter fun D : Data N m ρ X =>
          BadK (A cι.1) dig σ (O cι.1) D fun i => (cι.2 i : ℕ)).card := by
    refine (Finset.card_le_card (fun D hD => ?_)).trans Finset.card_biUnion_le
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hD
    obtain ⟨c, ι, hci⟩ := hD
    exact Finset.mem_biUnion.2 ⟨(c, ι), Finset.mem_univ _, Finset.mem_filter.2 ⟨Finset.mem_univ _, hci⟩⟩
  calc _ ≤ (∑ cι : Fin C × (Fin (k + 1) → Fin q),
          (Finset.univ.filter fun D : Data N m ρ X =>
            BadK (A cι.1) dig σ (O cι.1) D fun i => (cι.2 i : ℕ)).card) * Fintype.card (Fin (k + 1) → X) :=
        Nat.mul_le_mul_right _ hsub
    _ = ∑ cι : Fin C × (Fin (k + 1) → Fin q),
          (Finset.univ.filter fun D : Data N m ρ X =>
            BadK (A cι.1) dig σ (O cι.1) D fun i => (cι.2 i : ℕ)).card * Fintype.card (Fin (k + 1) → X) := by
        rw [Finset.sum_mul]
    _ ≤ ∑ cι : Fin C × (Fin (k + 1) → Fin q), g ^ (k + 1) *
          (Finset.univ.filter fun D : Data N m ρ X =>
            ∀ i, (cι.2 i : ℕ) < (((A cι.1).prog σ D).queries (O cι.1) D).length).card :=
        Finset.sum_le_sum fun cι _ => card_badK_ground_le (A cι.1) dig σ (O cι.1) (fun i => (cι.2 i : ℕ)) g hf
    _ = g ^ (k + 1) * ∑ c, ∑ ι : Fin (k + 1) → Fin q,
          (Finset.univ.filter fun D : Data N m ρ X =>
            ∀ i, (ι i : ℕ) < (((A c).prog σ D).queries (O c) D).length).card := by
        rw [← Finset.mul_sum, Fintype.sum_prod_type]
    _ ≤ g ^ (k + 1) * ∑ c, ∑ D : Data N m ρ X, (((A c).prog σ D).queries (O c) D).length ^ (k + 1) := by
        refine Nat.mul_le_mul_left _ (Finset.sum_le_sum fun c _ => ?_)
        calc (∑ ι : Fin (k + 1) → Fin q, (Finset.univ.filter fun D : Data N m ρ X =>
                ∀ i, (ι i : ℕ) < (((A c).prog σ D).queries (O c) D).length).card)
            = ∑ ι : Fin (k + 1) → Fin q, ∑ D : Data N m ρ X,
                if ∀ i, (ι i : ℕ) < (((A c).prog σ D).queries (O c) D).length then 1 else 0 :=
              Finset.sum_congr rfl fun ι _ => by rw [Finset.card_filter]
          _ = ∑ D : Data N m ρ X, ∑ ι : Fin (k + 1) → Fin q,
                if ∀ i, (ι i : ℕ) < (((A c).prog σ D).queries (O c) D).length then 1 else 0 :=
              Finset.sum_comm
          _ ≤ ∑ D : Data N m ρ X, (((A c).prog σ D).queries (O c) D).length ^ (k + 1) := by
              refine Finset.sum_le_sum fun D _ => ?_
              calc (∑ ι : Fin (k + 1) → Fin q,
                      if ∀ i, (ι i : ℕ) < (((A c).prog σ D).queries (O c) D).length then 1 else 0)
                  = (Finset.univ.filter fun ι : Fin (k + 1) → Fin q =>
                      ∀ i, (ι i : ℕ) < (((A c).prog σ D).queries (O c) D).length).card :=
                    (Finset.card_filter _ _).symm
                _ ≤ _ := card_idx_le q _ (k + 1)
    _ = g ^ (k + 1) * ∑ D : Data N m ρ X, ∑ c, (((A c).prog σ D).queries (O c) D).length ^ (k + 1) := by
        rw [Finset.sum_comm]
    _ ≤ g ^ (k + 1) * ∑ _D : Data N m ρ X, qtot ^ (k + 1) := by
        refine Nat.mul_le_mul_left _ (Finset.sum_le_sum fun D _ => ?_)
        exact (sum_pow_succ_le _ _ k).trans (Nat.pow_le_pow_left (hq D) _)
    _ = (g * qtot) ^ (k + 1) * Fintype.card (Data N m ρ X) := by
        rw [Finset.sum_const, Finset.card_univ, smul_eq_mul]
        ring

variable {ks : ℕ}

/-- **The slack's price on the sketch.** At signs whose sector blocks have rank at least `r`, if a pass's candidates
query at most `qtot` times in all, then on uniform data some candidate made correct digest queries behind `k + 1`
unobtained spread positions with probability at most `(qtot/2^(8r))^(k+1)`. -/
theorem pr_badK_family_le {C k : ℕ}
    (A : Fin C → Prover N m ρ h S L q w Sector (SketchSigns ks m) (Fin ks → ZMod (2 ^ 32)))
    (σ : SketchSigns ks m) (O : Fin C → Oracle N ρ h Sector (Fin ks → ZMod (2 ^ 32))) (r qtot : ℕ)
    (hr32 : r ≤ 32)
    (hr : ∀ u, r ≤ ((Accounting.Tag.signMatrix (σ.block u)).map (Int.cast : ℤ → ℚ)).rank)
    (hq : ∀ D : Data N m ρ Sector, ∑ c, (((A c).prog σ D).queries (O c) D).length ≤ qtot) :
    pr (fun D : Data N m ρ Sector =>
        ∃ c, ∃ ι : Fin (k + 1) → Fin q, BadK (A c) sketch σ (O c) D fun i => (ι i : ℕ)) ≤
      ((qtot : ℝ≥0∞) / 2 ^ (8 * r)) ^ (k + 1) := by
  classical
  obtain ⟨g, hgdef⟩ : ∃ g, g = 256 ^ (32 - r) := ⟨_, rfl⟩
  have hfib : fib (sketch (k := ks) (m := m)) σ ≤ g := by
    rw [hgdef]
    exact Accounting.SecurityProofs.SketchRankFiber ks m r σ hr
  have hcard := card_badK_family_le (k := k) A sketch σ O g qtot hfib hq
  have hbound := Bandwidth.pr_le_of_card
    (fun D : Data N m ρ Sector =>
      ∃ c, ∃ ι : Fin (k + 1) → Fin q, BadK (A c) sketch σ (O c) D fun i => (ι i : ℕ))
    (Fintype.card (Fin (k + 1) → Sector)) ((g * qtot) ^ (k + 1)) Fintype.card_ne_zero hcard
  refine hbound.trans (le_of_eq ?_)
  have hXc : (Fintype.card Sector : ℝ≥0∞) = 2 ^ 256 := by
    norm_num [Sector]
  have hg : (g : ℝ≥0∞) = 2 ^ (256 - 8 * r) := by
    rw [hgdef]
    push_cast
    rw [show (256 : ℝ≥0∞) = 2 ^ 8 by norm_num, ← pow_mul, show 8 * (32 - r) = 256 - 8 * r by omega]
  have hbase : (g : ℝ≥0∞) * qtot / 2 ^ 256 = (qtot : ℝ≥0∞) / 2 ^ (8 * r) := by
    rw [hg, show (2 : ℝ≥0∞) ^ 256 = 2 ^ (256 - 8 * r) * 2 ^ (8 * r) by
      rw [← pow_add, Nat.sub_add_cancel (by omega)]]
    exact ENNReal.mul_div_mul_left _ _ (by positivity) (ENNReal.pow_ne_top ENNReal.ofNat_ne_top)
  have hdivpow : ∀ (a b : ℝ≥0∞) (n : ℕ), (a / b) ^ n = a ^ n / b ^ n := fun a b n => by
    rw [div_eq_mul_inv, mul_pow, ← ENNReal.inv_pow, ← div_eq_mul_inv]
  rw [← hbase, hdivpow, Fintype.card_fun, Fintype.card_fin, Nat.cast_pow, Nat.cast_pow, Nat.cast_mul, hXc]

end Prover

end Accounting.RateSketch
