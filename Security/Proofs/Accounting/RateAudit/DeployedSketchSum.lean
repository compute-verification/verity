import Mathlib
import Models.Accounting.Guarantees.RateAuditDeployed
import Proofs.Accounting.RateAudit.Deployed
import Proofs.Accounting.RateAudit.DeployedSketch

/-!
# The deployed verbatim-sketch session with a shared query budget: queries add, grinding does not multiply them

`RemoteSessionConservationSketch`'s receipt term is `n·P·G·(N/2^h + q/2^(8r))`: the unknown-sector query term
`q/2^(8r)` is charged once per grinding candidate, because the model gives each of a pass's `G` candidate devices its
own cap of `q` oracle queries (`Device.bounded`), and `pr_unpaid_good_le` unions `Device.pr_unpaid_rank_le`'s
per-device `Fin q` union over all `P·G` candidates. When the deployed cap reads `q` as the prover's total hash budget,
that charges the same budget `G` times over, and at `q = G = 2^55`, `r = 12` the term is `2^14`: vacuous.

A pass's `G` candidates share their data: the store and the pass's certified stream are drawn once, and the candidates
differ only in their oracles (`SketchPassCoins`). So for fixed oracles one resampling argument prices every
candidate's correct-digest queries together: each query of each candidate is one guess at the shared unknown sector,
and a budget of `qtot` queries summed across the candidates pays `qtot·fib/|X|` once. Queries add; grinding multiplies
only the tag-guess term `N/2^h`. The composed bound is

  `(1 − d̄)^(d·n) + (1 − d_w/n)^(d·n) + n·ε + (n·P·G)·N/2^h + (n·P)·qtot/2^(8r)`.

At the deployed parameters (`h = 512`, `r = 12`) and the same caps `G = qtot = 2^55` the query term is
`n·P·2^-41` (`sketchQueryTermCapsSum`), non-vacuous where the product form's `2^14` was vacuous, and the guess term
`n·P·G·N/2^512` stays negligible.

The chain: `card_resample_ground` (the resampling count against its ground set), `Device.card_bad_ground_le` (one
query index's bad data, counted against the data where that index exists), `Device.card_bad_family_le` (the budgeted
union over a candidate family's query indices), `Device.pr_unpaid_rank_family_le` (one pass, all candidates),
`SketchDeployedWindow.pr_unpaid_good_sum_le` and `pr_unpaid_sum_le` (the window), and
`Accounting.SecurityProofs.RemoteSessionConservationSketchSum` (the session). The budget hypothesis is new: it is
stated inline here, beside the frozen guarantees, and `budget_of_bounded` shows `qtot = G·q` always satisfies it, so
the sum form at the worst budget recovers the product form. Adopting the sum form as the deployed guarantee (and
giving the budget a named home) is a statement change, which goes to Daniel.
-/

namespace Accounting.RateSketch

open Pous (pr)
open scoped ENNReal
open Bandwidth (pr_exists_le pr_le_of_card pr_mono pr_or_le)

/-- A uniform finite event whose count times `m` is at most `g` times the sample-space size has probability at most
`g/m` (restating `DeployedSketch.lean`'s private lemma). -/
private theorem pr_le_div_of_card_mul_le {α : Type} [Fintype α] (P : α → Prop) [DecidablePred P] (g m : ℕ)
    (hα : Fintype.card α ≠ 0) (hm : m ≠ 0)
    (h : (Finset.univ.filter P).card * m ≤ g * Fintype.card α) :
    pr P ≤ (g : ℝ≥0∞) / m := by
  unfold pr
  rw [ENNReal.div_le_iff (by exact_mod_cast hα) (ENNReal.natCast_ne_top _)]
  rw [show (g : ℝ≥0∞) / m * Fintype.card α =
      ((g : ℝ≥0∞) * Fintype.card α) / m by
        simp only [ENNReal.div_eq_inv_mul]
        ac_rfl]
  rw [ENNReal.le_div_iff_mul_le (Or.inl (by exact_mod_cast hm)) (Or.inl (ENNReal.natCast_ne_top _))]
  rw [Finset.filter_congr_decidable]
  exact_mod_cast h

/-- **Resampling, counted against its ground set.** `card_resample` with the image kept in `G`: if every point of `G`
has a direction along which `G` and `dir` are constant, and each such line meets `B` in at most `g` points, then
`|B|·|X| ≤ g·|G|`. The budgeted union needs the `g·|G|` reading, which `card_resample` weakens to `g·|X|^|P|`. -/
theorem card_resample_ground {P X : Type} [Fintype P] [DecidableEq P] [Fintype X] [DecidableEq X]
    (G B : Finset (P → X)) (dir : (P → X) → P) (g : ℕ) (hBG : B ⊆ G)
    (hG : ∀ D ∈ G, ∀ a, Function.update D (dir D) a ∈ G ∧ dir (Function.update D (dir D) a) = dir D)
    (hB : ∀ D ∈ G, (Finset.univ.filter (fun a => Function.update D (dir D) a ∈ B)).card ≤ g) :
    B.card * Fintype.card X ≤ g * G.card := by
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

namespace Device

variable {N m ρ h S L q w : ℕ} {X Sig Dg : Type}

/-- `card_bad_le`, counted against the data where query index `i` exists: one query index's bad data is at most a
`g/|X|` fraction of the data on which the device makes more than `i` queries. Summed over indices and candidates,
each data point is charged once per query it actually answers, which is what lets a shared budget replace the
per-candidate cap. -/
theorem card_bad_ground_le [Fintype X] [DecidableEq X] [Fintype Dg] [DecidableEq Dg]
    (A : Device N m ρ h S L q w X Sig Dg) (dig : Sig → (Fin m → X) → Dg) (σ : Sig) (O : Oracle N ρ h X Dg)
    (i g : ℕ) [DecidablePred (fun D : Data N m ρ X => Bad A dig σ O D i g)] :
    (Finset.univ.filter (fun D : Data N m ρ X => Bad A dig σ O D i g)).card * Fintype.card X ≤
      g * (Finset.univ.filter (fun D : Data N m ρ X => i < ((A.prog σ D).queries O D).length)).card := by
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
  have hground : (Finset.univ.filter (fun D : Data N m ρ X => (unobtained A σ O D i).Nonempty)) ⊆
      Finset.univ.filter (fun D : Data N m ρ X => i < ((A.prog σ D).queries O D).length) := by
    intro D hD
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hD ⊢
    obtain ⟨p, hp⟩ := hD
    simp only [unobtained, Finset.mem_filter, Finset.mem_univ, true_and] at hp
    obtain ⟨-, y, hy, -⟩ := hp
    exact (List.getElem?_eq_some_iff.1 hy).1
  refine le_trans (card_resample_ground
    (Finset.univ.filter (fun D : Data N m ρ X => (unobtained A σ O D i).Nonempty))
    (Finset.univ.filter (fun D : Data N m ρ X => Bad A dig σ O D i g)) dir g
    (fun D hD => by
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hD ⊢
      exact hD.2.1) (fun D hD a => ?_) (fun D hD => ?_))
    (Nat.mul_le_mul_left g (Finset.card_le_card hground))
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

/-- **The budgeted union.** A family of `C` candidate devices on shared data, each with its own oracle: if across the
family the queries total at most `qtot` on every data point, then the data on which some candidate makes a correct
digest query without an obtained sector is at most a `qtot·g/|X|` fraction. The union is over the queries actually
made, not over `C` per-candidate caps. -/
theorem card_bad_family_le [Fintype X] [DecidableEq X] [Fintype Dg] [DecidableEq Dg] {C : ℕ}
    (A : Fin C → Device N m ρ h S L q w X Sig Dg) (dig : Sig → (Fin m → X) → Dg) (σ : Sig)
    (O : Fin C → Oracle N ρ h X Dg) (g qtot : ℕ)
    (hq : ∀ D : Data N m ρ X, ∑ c, (((A c).prog σ D).queries (O c) D).length ≤ qtot)
    [DecidablePred (fun D : Data N m ρ X => ∃ c, ∃ i : Fin q, Bad (A c) dig σ (O c) D i g)] :
    (Finset.univ.filter (fun D : Data N m ρ X => ∃ c, ∃ i : Fin q, Bad (A c) dig σ (O c) D i g)).card *
        Fintype.card X ≤ g * qtot * Fintype.card (Data N m ρ X) := by
  classical
  have hsub : (Finset.univ.filter (fun D : Data N m ρ X => ∃ c, ∃ i : Fin q, Bad (A c) dig σ (O c) D i g)).card ≤
      ∑ ci : Fin C × Fin q,
        (Finset.univ.filter (fun D : Data N m ρ X => Bad (A ci.1) dig σ (O ci.1) D ci.2 g)).card := by
    refine (Finset.card_le_card (fun D hD => ?_)).trans Finset.card_biUnion_le
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hD
    obtain ⟨c, i, hci⟩ := hD
    exact Finset.mem_biUnion.2 ⟨(c, i), Finset.mem_univ _,
      Finset.mem_filter.2 ⟨Finset.mem_univ _, hci⟩⟩
  calc (Finset.univ.filter
          (fun D : Data N m ρ X => ∃ c, ∃ i : Fin q, Bad (A c) dig σ (O c) D i g)).card * Fintype.card X
      ≤ (∑ ci : Fin C × Fin q,
          (Finset.univ.filter (fun D : Data N m ρ X => Bad (A ci.1) dig σ (O ci.1) D ci.2 g)).card) *
            Fintype.card X := Nat.mul_le_mul_right _ hsub
    _ = ∑ ci : Fin C × Fin q,
          (Finset.univ.filter (fun D : Data N m ρ X => Bad (A ci.1) dig σ (O ci.1) D ci.2 g)).card *
            Fintype.card X := by rw [Finset.sum_mul]
    _ ≤ ∑ ci : Fin C × Fin q, g * (Finset.univ.filter
          (fun D : Data N m ρ X =>
            (ci.2 : ℕ) < (((A ci.1).prog σ D).queries (O ci.1) D).length)).card :=
        Finset.sum_le_sum fun ci _ => card_bad_ground_le (A ci.1) dig σ (O ci.1) ci.2 g
    _ = g * ∑ c, ∑ i : Fin q, (Finset.univ.filter
          (fun D : Data N m ρ X => (i : ℕ) < (((A c).prog σ D).queries (O c) D).length)).card := by
        rw [← Finset.mul_sum, Fintype.sum_prod_type]
    _ ≤ g * ∑ c, ∑ D : Data N m ρ X, (((A c).prog σ D).queries (O c) D).length := by
        refine Nat.mul_le_mul_left g (Finset.sum_le_sum fun c _ => ?_)
        calc (∑ i : Fin q, (Finset.univ.filter
                (fun D : Data N m ρ X =>
                  (i : ℕ) < (((A c).prog σ D).queries (O c) D).length)).card)
            = ∑ i : Fin q, ∑ D : Data N m ρ X,
                if (i : ℕ) < (((A c).prog σ D).queries (O c) D).length then 1 else 0 :=
              Finset.sum_congr rfl fun i _ => by rw [Finset.card_filter]
          _ = ∑ D : Data N m ρ X, ∑ i : Fin q,
                if (i : ℕ) < (((A c).prog σ D).queries (O c) D).length then 1 else 0 :=
              Finset.sum_comm
          _ ≤ ∑ D : Data N m ρ X, (((A c).prog σ D).queries (O c) D).length := by
              refine Finset.sum_le_sum fun D _ => ?_
              calc (∑ i : Fin q,
                      if (i : ℕ) < (((A c).prog σ D).queries (O c) D).length then 1 else 0)
                  = (Finset.univ.filter
                      (fun i : Fin q =>
                        (i : ℕ) < (((A c).prog σ D).queries (O c) D).length)).card :=
                    (Finset.card_filter _ _).symm
                _ ≤ (Finset.range ((((A c).prog σ D).queries (O c) D).length)).card := by
                    refine Finset.card_le_card_of_injOn (fun i => (i : ℕ)) (fun i hi => ?_)
                      (Fin.val_injective.injOn)
                    simp only [Finset.coe_filter, Finset.mem_univ, true_and,
                      Set.mem_ofPred_eq] at hi
                    simpa using hi
                _ = (((A c).prog σ D).queries (O c) D).length := Finset.card_range _
    _ = g * ∑ D : Data N m ρ X, ∑ c, (((A c).prog σ D).queries (O c) D).length := by
        rw [Finset.sum_comm]
    _ ≤ g * ∑ _D : Data N m ρ X, qtot :=
        Nat.mul_le_mul_left g (Finset.sum_le_sum fun D _ => hq D)
    _ = g * qtot * Fintype.card (Data N m ρ X) := by
        rw [Finset.sum_const, Finset.card_univ, smul_eq_mul, mul_assoc, mul_comm qtot]

variable {ks : ℕ}

/-- The pointwise split behind `pr_unpaid_rank_le`: an unpaid verbatim pass either holds a correct unqueried tag or
made a correct digest query without an obtained sector. Factored out of `pr_unpaid_rank_le` so the family bound can
reuse it per candidate. -/
theorem unpaid_split
    (A : Device N m ρ h S L q 256 Sector (SketchSigns ks m) (Fin ks → ZMod (2 ^ 32)))
    (σ : SketchSigns ks m) (g X : ℕ)
    (hfib : fib (sketch (k := ks) (m := m)) σ ≤ g)
    (hX : ∀ O D, (A.prog σ D).rebuilds O D * 256 ≤ X)
    (O : Oracle N ρ h Sector (Fin ks → ZMod (2 ^ 32))) (D : Data N m ρ Sector)
    (hunpaid : A.footprint σ O D + (S + (L - ρ * 256) + X) <
      (A.correct sketch σ O D).card * (m * 256)) :
    (∃ t, t ∈ A.correct sketch σ O D ∧ t ∉ queried A sketch σ O D) ∨
      ∃ i : Fin q, Bad A sketch σ O D i g := by
  classical
  by_cases hall : ∀ t ∈ A.correct sketch σ O D, t ∈ queried A sketch σ O D
  swap
  · push Not at hall
    exact Or.inl hall
  by_cases hcomp : ∀ t ∈ A.correct sketch σ O D, ∀ p, relevant t p → p ∈ A.obtained σ O D
  swap
  · push Not at hcomp
    obtain ⟨t, ht, p, hrel, hp⟩ := hcomp
    have hqmem : (D.stream, t, sketch σ (D.tile t)) ∈ (A.prog σ D).queries O D := by
      simpa [queried] using hall t ht
    have hlt := List.idxOf_lt_length_of_mem hqmem
    have hget : ((A.prog σ D).queries O D)[
        ((A.prog σ D).queries O D).idxOf (D.stream, t, sketch σ (D.tile t))]? =
        some (D.stream, t, sketch σ (D.tile t)) := by
      rw [List.getElem?_eq_getElem hlt, List.getElem_idxOf hlt]
    refine Or.inr ⟨⟨_, hlt.trans_le (A.bounded σ _ O D).2⟩, hfib, ⟨p, ?_⟩, _, hget, rfl⟩
    simp only [unobtained, Finset.mem_filter, Finset.mem_univ, true_and]
    exact ⟨hp, _, hget, hrel⟩
  have hne : (A.correct sketch σ O D).Nonempty := by
    by_contra hn
    rw [Finset.not_nonempty_iff_eq_empty.mp hn, Finset.card_empty, zero_mul] at hunpaid
    omega
  obtain ⟨t₀, ht₀⟩ := hne
  have hsub :
      ((A.correct sketch σ O D) ×ˢ (Finset.univ : Finset (Fin m))).disjSum
          (Finset.univ : Finset (Fin ρ)) ⊆ A.obtained σ O D := by
    rintro (⟨t, u⟩ | i) hp'
    · simp only [Finset.inl_mem_disjSum, Finset.mem_product, Finset.mem_univ, and_true] at hp'
      exact hcomp t hp' _ rfl
    · exact hcomp t₀ ht₀ _ trivial
  have hc := Finset.card_le_card hsub
  rw [Finset.card_disjSum, Finset.card_product, Finset.card_univ, Fintype.card_fin, Finset.card_univ,
    Fintype.card_fin] at hc
  have hob := A.card_obtained σ O D
  have hrebuild := hX O D
  have hpay :
      (A.correct sketch σ O D).card * (m * 256) + ρ * 256 ≤
        S + A.footprint σ O D + L + X := by
    calc
      _ = ((A.correct sketch σ O D).card * m + ρ) * 256 := by ring
      _ ≤ (A.obtained σ O D).card * 256 := Nat.mul_le_mul_right 256 hc
      _ ≤ S + A.footprint σ O D + L + (A.prog σ D).rebuilds O D * 256 := hob
      _ ≤ S + A.footprint σ O D + L + X := by omega
  omega

/-- The per-candidate cap `Device.bounded` gives the trivial budget `G·q`: the sum form at the worst budget is the
product form, so the budgeted bound generalizes `pr_unpaid_good_le`'s. -/
theorem budget_of_bounded {G : ℕ}
    (A : Fin G → Device N m ρ h S L q w X Sig Dg) (σ : Sig)
    (Os : Fin G → Oracle N ρ h X Dg) (D : Data N m ρ X) :
    ∑ c, (((A c).prog σ D).queries (Os c) D).length ≤ G * q :=
  (Finset.sum_le_sum fun c _ => ((A c).bounded σ _ (Os c) D).2).trans
    (by simp [Finset.sum_const, Finset.card_univ])

/-- An event of the shared data and one candidate's oracle has at most its probability over one data–oracle pair:
grinding candidates beyond `c` are a uniform fiber. -/
theorem pr_component_le {α β : Type} [Fintype α] [Fintype β] {G : ℕ} (c : Fin G) (Q : α × β → Prop)
    (ε : ℝ≥0∞) (hQ : pr Q ≤ ε) :
    pr (fun x : α × (Fin G → β) => Q (x.1, x.2 c)) ≤ ε := by
  classical
  let e : α × (Fin G → β) ≃ (α × β) × ({z : Fin G // z ≠ c} → β) :=
    ((Equiv.refl α).prodCongr (Equiv.funSplitAt c β)).trans (Equiv.prodAssoc α β _).symm
  rw [Rate.pr_equiv e]
  refine Rate.pr_le_of_fiber _ _ (fun rest => ?_)
  refine (pr_mono (fun z hz => ?_)).trans hQ
  have h1 : (e.symm (z, rest)).1 = z.1 := by simp [e]
  have h2 : (e.symm (z, rest)).2 c = z.2 := by simp [e]
  rw [h1, h2] at hz
  exact hz

variable {ks : ℕ} in
/-- A correct receipt whose tag was never queried: `1/2^h` per tile, over one pass's data and oracle. The `ht` step of
`pr_unpaid_rank_le`, factored out so the family bound can take it per candidate. -/
theorem pr_guess_le
    (A : Device N m ρ h S L q 256 Sector (SketchSigns ks m) (Fin ks → ZMod (2 ^ 32)))
    (σ : SketchSigns ks m) (t : Fin N) :
    pr (fun z : Data N m ρ Sector × Oracle N ρ h Sector (Fin ks → ZMod (2 ^ 32)) =>
      t ∈ A.correct sketch σ z.2 z.1 ∧ t ∉ queried A sketch σ z.2 z.1)
      ≤ (1 : ℝ≥0∞) / 2 ^ h := by
  classical
  rw [Rate.pr_equiv (Equiv.prodComm (Data N m ρ Sector)
    (Oracle N ρ h Sector (Fin ks → ZMod (2 ^ 32))))]
  change pr (fun x : Oracle N ρ h Sector (Fin ks → ZMod (2 ^ 32)) × Data N m ρ Sector =>
    t ∈ A.correct sketch σ x.1 x.2 ∧ t ∉ queried A sketch σ x.1 x.2) ≤ (1 : ℝ≥0∞) / 2 ^ h
  refine Rate.pr_le_of_fiber _ _ (fun D => ?_)
  change pr (fun O : Oracle N ρ h Sector (Fin ks → ZMod (2 ^ 32)) =>
    t ∈ A.correct sketch σ O D ∧ t ∉ queried A sketch σ O D) ≤ (1 : ℝ≥0∞) / 2 ^ h
  have hbound := pr_le_div_of_card_mul_le
    (α := Oracle N ρ h Sector (Fin ks → ZMod (2 ^ 32)))
    (fun O => t ∈ A.correct sketch σ O D ∧ t ∉ queried A sketch σ O D)
    1 (2 ^ h) Fintype.card_ne_zero (by positivity)
    (by simpa [Device.correct, and_comm] using card_guess_le A sketch σ D t)
  push_cast at hbound
  simpa using hbound

/-- The oracle family of a pass's `G` grinding candidates is finite (unfolded in two steps, as
`instFintypeSketchPassCoins` does). -/
local instance instFintypeOracleFam (N ρ h ks G : ℕ) :
    Fintype (Fin G → Oracle N ρ h Sector (Fin ks → ZMod (2 ^ 32))) := by
  letI : Fintype (Oracle N ρ h Sector (Fin ks → ZMod (2 ^ 32))) := inferInstance
  exact inferInstance

/-- **One pass, all its grinding candidates.** Outside the low-rank event, some candidate of a verbatim pass is
unpaid with probability at most `G·N/2^h + qtot/2^(8r)`: grinding pays the tag-guess term per candidate, and the
query term once for the shared budget `qtot`. Compare `pr_unpaid_rank_le`, whose union over candidates pays
`G·(N/2^h + q/2^(8r))`. -/
theorem pr_unpaid_rank_family_le {G : ℕ}
    (A : Fin G → Device N m ρ h S L q 256 Sector (SketchSigns ks m) (Fin ks → ZMod (2 ^ 32)))
    (σ : SketchSigns ks m) (r X qtot : ℕ)
    (hr32 : r ≤ 32)
    (hr : ∀ u, r ≤ ((Accounting.Tag.signMatrix (σ.block u)).map (Int.cast : ℤ → ℚ)).rank)
    (hX : ∀ c O D, ((A c).prog σ D).rebuilds O D * 256 ≤ X)
    (hq : ∀ (Os : Fin G → Oracle N ρ h Sector (Fin ks → ZMod (2 ^ 32))) (D : Data N m ρ Sector),
      ∑ c, (((A c).prog σ D).queries (Os c) D).length ≤ qtot) :
    pr (fun x : Data N m ρ Sector × (Fin G → Oracle N ρ h Sector (Fin ks → ZMod (2 ^ 32))) =>
      ∃ c, (A c).footprint σ (x.2 c) x.1 + (S + (L - ρ * 256) + X) <
        ((A c).correct sketch σ (x.2 c) x.1).card * (m * 256))
      ≤ (G : ℝ≥0∞) * ((N : ℝ≥0∞) / 2 ^ h) + (qtot : ℝ≥0∞) / 2 ^ (8 * r) := by
  classical
  let g := 256 ^ (32 - r)
  have hfib : fib (sketch (k := ks) (m := m)) σ ≤ g :=
    Accounting.SecurityProofs.SketchRankFiber ks m r σ hr
  have hsplit : ∀ x : Data N m ρ Sector × (Fin G → Oracle N ρ h Sector (Fin ks → ZMod (2 ^ 32))),
      (∃ c, (A c).footprint σ (x.2 c) x.1 + (S + (L - ρ * 256) + X) <
          ((A c).correct sketch σ (x.2 c) x.1).card * (m * 256)) →
        (∃ c, ∃ t, t ∈ (A c).correct sketch σ (x.2 c) x.1 ∧
            t ∉ queried (A c) sketch σ (x.2 c) x.1) ∨
          ∃ c, ∃ i : Fin q, Bad (A c) sketch σ (x.2 c) x.1 i g := by
    rintro ⟨D, Os⟩ ⟨c, hc⟩
    rcases unpaid_split (A c) σ g X hfib (hX c) (Os c) D hc with hguess | hbad
    · exact Or.inl ⟨c, hguess⟩
    · exact Or.inr ⟨c, hbad⟩
  refine (pr_mono hsplit).trans ((pr_or_le _ _).trans (add_le_add ?_ ?_))
  · -- the tag-guess term: one union per candidate and tile, as in `pr_unpaid_rank_le`
    refine (pr_exists_le _).trans ?_
    have hc : ∀ c : Fin G,
        pr (fun x : Data N m ρ Sector × (Fin G → Oracle N ρ h Sector (Fin ks → ZMod (2 ^ 32))) =>
          ∃ t, t ∈ (A c).correct sketch σ (x.2 c) x.1 ∧ t ∉ queried (A c) sketch σ (x.2 c) x.1)
          ≤ (N : ℝ≥0∞) / 2 ^ h := by
      intro c
      refine (pr_exists_le _).trans ?_
      have ht : ∀ t : Fin N,
          pr (fun x : Data N m ρ Sector × (Fin G → Oracle N ρ h Sector (Fin ks → ZMod (2 ^ 32))) =>
            t ∈ (A c).correct sketch σ (x.2 c) x.1 ∧ t ∉ queried (A c) sketch σ (x.2 c) x.1)
            ≤ (1 : ℝ≥0∞) / 2 ^ h := fun t =>
        pr_component_le c
          (fun z : Data N m ρ Sector × Oracle N ρ h Sector (Fin ks → ZMod (2 ^ 32)) =>
            t ∈ (A c).correct sketch σ z.2 z.1 ∧ t ∉ queried (A c) sketch σ z.2 z.1)
          _ (pr_guess_le (A c) σ t)
      refine (Finset.sum_le_sum (fun t _ => ht t)).trans (le_of_eq ?_)
      simp [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, div_eq_mul_inv]
    refine (Finset.sum_le_sum (fun c _ => hc c)).trans (le_of_eq ?_)
    simp [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  · -- the query term: one budgeted union over the shared data, fibered at each oracle family
    refine Rate.pr_le_of_fiber _ _ (fun Os => ?_)
    have hcard := card_bad_family_le A sketch σ Os g qtot (fun D => hq Os D)
    have hbound := pr_le_div_of_card_mul_le
      (fun D : Data N m ρ Sector => ∃ c, ∃ i : Fin q, Bad (A c) sketch σ (Os c) D i g)
      (g * qtot) (Fintype.card Sector) Fintype.card_ne_zero (by simp [Sector]) (by
        simpa using hcard)
    have hXc : (Fintype.card Sector : ℝ≥0∞) = 2 ^ 256 := by
      norm_num [Sector]
    rw [hXc] at hbound
    refine hbound.trans (le_of_eq ?_)
    have hg : ((g : ℕ) : ℝ≥0∞) = 2 ^ (256 - 8 * r) := by
      dsimp [g]
      push_cast
      rw [show (256 : ℝ≥0∞) = 2 ^ 8 by norm_num, ← pow_mul,
        show 8 * (32 - r) = 256 - 8 * r by omega]
    have hsplitPow : (2 : ℝ≥0∞) ^ 256 = 2 ^ (256 - 8 * r) * 2 ^ (8 * r) := by
      rw [← pow_add, Nat.sub_add_cancel (by omega)]
    push_cast
    rw [hg, hsplitPow, mul_comm ((2 : ℝ≥0∞) ^ (256 - 8 * r)) ((qtot : ℝ≥0∞))]
    rw [show (qtot : ℝ≥0∞) * 2 ^ (256 - 8 * r) / (2 ^ (256 - 8 * r) * 2 ^ (8 * r)) =
        2 ^ (256 - 8 * r) * (qtot : ℝ≥0∞) / (2 ^ (256 - 8 * r) * 2 ^ (8 * r)) by ring_nf]
    exact ENNReal.mul_div_mul_left _ _ (by positivity) (ENNReal.pow_ne_top ENNReal.ofNat_ne_top)

end Device

end Accounting.RateSketch
