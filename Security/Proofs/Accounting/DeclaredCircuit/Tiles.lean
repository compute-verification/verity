import Mathlib
import Definitions.Accounting.Carrier
import Proofs.Accounting.RateAudit.DeployedSketchSum

/-!
# A pass on some of its tiles: what it obtained there

The pieces of `Accounting.SecurityProofs.DeclaredCircuitRequirement` that count one pass on a set `U` of its tiles. The
positions a run read in `U`'s tiles and the stream are at most its distinct memory reads there, its link steps and its
rebuilds there (`Chip.card_reads_tiles_le`), so what a device obtained there is at most its on-chip state, those reads,
the link and those rebuilds (`Prover.card_obtained_tiles`). A pass whose memory reads in `U` plus the slack fall short of
`U`'s correct receipts by `k` tiles more than the slack pays has a correct tag it never queried, or `k + 1` positions it
never obtained behind correct digest queries, at most one sector a tile (`Prover.unpaid_tiles_split`, `BadK`): each
tile short of a sector gives one, and each stream sector short gives one.
-/

namespace Accounting.RateSketch

/-- The positions of the tiles `U` and of the stream. -/
def tilePos {N : ℕ} (U : Finset (Fin N)) (m ρ : ℕ) : Finset (Pos N m ρ) :=
  (U ×ˢ (Finset.univ : Finset (Fin m))).disjSum (Finset.univ : Finset (Fin ρ))

@[simp] theorem inl_mem_tilePos {N m ρ : ℕ} {U : Finset (Fin N)} {p : Fin N × Fin m} :
    (Sum.inl p : Pos N m ρ) ∈ tilePos U m ρ ↔ p.1 ∈ U := by
  simp [tilePos]

@[simp] theorem inr_mem_tilePos {N m ρ : ℕ} {U : Finset (Fin N)} {r : Fin ρ} :
    (Sum.inr r : Pos N m ρ) ∈ tilePos U m ρ := by
  simp [tilePos]

/-- Inserting a point adds one to a set's part in `t` when the point is in `t`, and nothing otherwise. -/
theorem card_insert_inter_le {β : Type} [DecidableEq β] (a : β) (s t : Finset β) :
    (insert a s ∩ t).card ≤ (s ∩ t).card + if a ∈ t then 1 else 0 := by
  by_cases ha : a ∈ t
  · simp only [ha, ↓reduceIte]
    refine (Finset.card_le_card fun x hx => ?_).trans (Finset.card_insert_le a (s ∩ t))
    simp only [Finset.mem_inter, Finset.mem_insert] at hx ⊢
    tauto
  · simp only [ha, ↓reduceIte, add_zero]
    refine Finset.card_le_card fun x hx => ?_
    simp only [Finset.mem_inter, Finset.mem_insert] at hx ⊢
    rcases hx with ⟨rfl | hx, hz⟩
    · exact absurd hz ha
    · exact ⟨hx, hz⟩

namespace Chip

variable {N m ρ h : ℕ} {X Dg α : Type}

theorem card_map_inl_inter_tilePos (U : Finset (Fin N)) (M : Finset (Fin N × Fin m)) :
    (M.map Function.Embedding.inl ∩ tilePos U m ρ).card = (M.filter fun p => p.1 ∈ U).card := by
  have e : M.map Function.Embedding.inl ∩ tilePos U m ρ =
      (M.filter fun p => p.1 ∈ U).map (Function.Embedding.inl : (Fin N × Fin m) ↪ Pos N m ρ) := by
    ext x
    rcases x with p | r <;> simp
  rw [e, Finset.card_map]

/-- **The positions read in the tiles `U` and the stream** are at most the distinct sectors of `U` read from device
memory, the link's sectors and the sectors of `U` rebuilt (with any set `M` of store sectors on both sides). -/
theorem card_reads_tiles_le (O : Oracle N ρ h X Dg) (D : Data N m ρ X) (U : Finset (Fin N))
    (c : Chip N m ρ h X Dg α) (M : Finset (Fin N × Fin m)) :
    ((c.reads O D ∪ M.map Function.Embedding.inl) ∩ tilePos U m ρ).card ≤
      ((c.mems O D ∪ M).filter fun p => p.1 ∈ U).card + c.linkSteps O D + c.rebuildsIn U O D := by
  induction c generalizing M with
  | done a =>
    simp only [reads, mems, linkSteps, rebuildsIn, Finset.empty_union, add_zero]
    exact (card_map_inl_inter_tilePos U M).le
  | tag x k ih =>
    simp only [reads, mems, linkSteps, rebuildsIn]
    exact ih (O x) M
  | mem p k ih =>
    have := ih (D (.inl p)) (insert p M)
    simp only [reads, mems, linkSteps, rebuildsIn, Finset.insert_union, Finset.union_insert, Finset.map_insert,
      Function.Embedding.inl_apply] at this ⊢
    exact this
  | link p k ih =>
    have h1 := ih (D p) M
    simp only [reads, mems, linkSteps, rebuildsIn, Finset.insert_union]
    have h2 := card_insert_inter_le p ((k (D p)).reads O D ∪ M.map Function.Embedding.inl) (tilePos U m ρ)
    have h3 : (if p ∈ tilePos U m ρ then 1 else 0) ≤ 1 := by split <;> omega
    omega
  | rebuild p k ih =>
    have h1 := ih (D (.inl p)) M
    simp only [reads, mems, linkSteps, rebuildsIn, Finset.insert_union]
    have h2 := card_insert_inter_le (Sum.inl p : Pos N m ρ)
      ((k (D (.inl p))).reads O D ∪ M.map Function.Embedding.inl) (tilePos U m ρ)
    simp only [inl_mem_tilePos] at h2
    omega

/-- Rebuilds in more tiles are more. -/
theorem rebuildsIn_mono {U V : Finset (Fin N)} (hUV : U ⊆ V) (O : Oracle N ρ h X Dg) (D : Data N m ρ X)
    (c : Chip N m ρ h X Dg α) : c.rebuildsIn U O D ≤ c.rebuildsIn V O D := by
  induction c with
  | done a => simp only [rebuildsIn, le_refl]
  | tag x k ih =>
    simp only [rebuildsIn]
    exact ih (O x)
  | mem p k ih =>
    simp only [rebuildsIn]
    exact ih (D (.inl p))
  | link p k ih =>
    simp only [rebuildsIn]
    exact ih (D p)
  | rebuild p k ih =>
    have h1 := ih (D (.inl p))
    simp only [rebuildsIn]
    by_cases hp : p.1 ∈ U
    · simp only [hp, hUV hp, ↓reduceIte]
      omega
    · simp only [hp, ↓reduceIte, add_zero]
      split <;> omega

end Chip

namespace Prover

variable {N m ρ h S L q w : ℕ} {X Sig Dg : Type}

/-- **What a run obtained in the tiles `U` and the stream**, in bits: at most its on-chip state, its distinct memory
reads in `U`, the link and its rebuilt sectors of `U`. -/
theorem card_obtained_tiles (A : Prover N m ρ h S L q w X Sig Dg) (σ : Sig) (O : Oracle N ρ h X Dg)
    (D : Data N m ρ X) (U : Finset (Fin N)) :
    (A.obtained σ O D ∩ tilePos U m ρ).card * w ≤
      S + (((A.prog σ D).mems O D).filter fun p => p.1 ∈ U).card * w + L +
        (A.prog σ D).rebuildsIn U O D * w := by
  have h1 := Chip.card_reads_tiles_le O D U (A.prog σ D) ∅
  simp only [Finset.map_empty, Finset.union_empty] at h1
  have h2 : (A.obtained σ O D ∩ tilePos U m ρ).card ≤
      A.held.card + ((A.prog σ D).reads O D ∩ tilePos U m ρ).card := by
    unfold obtained
    calc ((A.held.map Function.Embedding.inl ∪ (A.prog σ D).reads O D) ∩ tilePos U m ρ).card
        ≤ (A.held.map Function.Embedding.inl ∪ ((A.prog σ D).reads O D ∩ tilePos U m ρ)).card := by
          refine Finset.card_le_card fun x hx => ?_
          simp only [Finset.mem_inter, Finset.mem_union] at hx ⊢
          tauto
      _ ≤ (A.held.map Function.Embedding.inl).card + ((A.prog σ D).reads O D ∩ tilePos U m ρ).card :=
          Finset.card_union_le _ _
      _ = _ := by rw [Finset.card_map]
  have h3 := Nat.mul_le_mul_right w (h2.trans (Nat.add_le_add_left h1 _))
  have h4 := A.held_le
  have h5 := (A.bounded σ (fun p => D (.inl p.1)) O D).1
  change (A.prog σ D).linkSteps O D * w ≤ L at h5
  simp only [add_mul] at h3
  omega

end Prover

variable {N m ρ h S L q w : ℕ} {X Sig Dg : Type}

/-- `K` distinct positions, at most one sector of each tile. -/
def Spread {K : ℕ} (p : Fin K → Pos N m ρ) : Prop :=
  Function.Injective p ∧
    ∀ i i' (a a' : Fin N × Fin m), p i = .inl a → p i' = .inl a' → a.1 = a'.1 → i = i'

/-- **`K` unobtained positions behind the queries at `ι`**: for a run whose queries are `Q` and whose obtained positions
are `Ob`, `p` are `K` distinct positions outside `Ob`, at most one sector of each tile, each read by the tag of its
query's tile. -/
def Unobtained {K : ℕ} (Q : List ((Fin ρ → X) × Fin N × Dg)) (Ob : Finset (Pos N m ρ)) (ι : Fin K → ℕ)
    (p : Fin K → Pos N m ρ) : Prop :=
  Spread p ∧ ∀ i, p i ∉ Ob ∧ ∃ y, Q[ι i]? = some y ∧ relevant y.2.1 (p i)

/-- **`K` unobtained positions behind correct digest queries**: the device's queries at the indices `ι` are at the
digests of their tiles, and `K` positions their tiles' tags read were never obtained, at most one sector of each tile.
At `K = 1` it is `Bad` with the signs' fiber left out. -/
def BadK {K : ℕ} (A : Prover N m ρ h S L q w X Sig Dg) (dig : Sig → (Fin m → X) → Dg) (σ : Sig)
    (O : Oracle N ρ h X Dg) (D : Data N m ρ X) (ι : Fin K → ℕ) : Prop :=
  (∃ p, Unobtained ((A.prog σ D).queries O D) (A.obtained σ O D) ι p) ∧
    ∀ i, ∃ y, ((A.prog σ D).queries O D)[ι i]? = some y ∧ y = (D.stream, y.2.1, dig σ (D.tile y.2.1))

/-- **An unpaid pass on the tiles `U`**: if a verbatim pass's distinct memory reads in `U` plus the slack, with `k` tiles
more, fall short of its correct receipts in `U`, and it rebuilt at most `X` bits of `U`'s sectors, then it holds a
correct tag it never queried, or it made correct digest queries behind `k + 1` positions it never obtained, at most one
sector a tile. The positions it obtained in `U`'s tiles and the stream are at most its state, those reads, the link and
those rebuilds, so more than `k·m` of them are missing; at most `m` of them in each tile, so `k + 1` tiles or stream
sectors are short. -/
theorem Prover.unpaid_tiles_split [Fintype X] [DecidableEq X] [Fintype Dg] [DecidableEq Dg]
    (A : Prover N m ρ h S L q 256 X Sig Dg) (dig : Sig → (Fin m → X) → Dg) (σ : Sig) (O : Oracle N ρ h X Dg)
    (D : Data N m ρ X) (U : Finset (Fin N)) (Xr k : ℕ) (hm : 0 < m)
    (hX : (A.prog σ D).rebuildsIn U O D * 256 ≤ Xr)
    (hunpaid : (((A.prog σ D).mems O D).filter fun p => p.1 ∈ U).card * 256 +
        (S + (L - ρ * 256) + (Xr + k * (m * 256))) < (A.correct dig σ O D ∩ U).card * (m * 256)) :
    (∃ t, t ∈ A.correct dig σ O D ∧ t ∉ queried A dig σ O D) ∨
      ∃ ι : Fin (k + 1) → Fin q, BadK A dig σ O D fun i => (ι i : ℕ) := by
  classical
  by_cases hall : ∀ t ∈ A.correct dig σ O D, t ∈ queried A dig σ O D
  swap
  · push Not at hall
    exact Or.inl hall
  right
  have hob := A.card_obtained_tiles σ O D U
  set Q := A.correct dig σ O D ∩ U with hQ
  set Z : Finset (Pos N m ρ) := (Q ×ˢ (Finset.univ : Finset (Fin m))).disjSum (Finset.univ : Finset (Fin ρ))
    with hZ
  set Ob := A.obtained σ O D with hOb
  set Mis := Z \ Ob with hMis
  have hZcard : Z.card = Q.card * m + ρ := by
    simp only [hZ, Finset.card_disjSum, Finset.card_product, Finset.card_univ, Fintype.card_fin]
  have hZU : Z ∩ Ob ⊆ Ob ∩ tilePos U m ρ := by
    intro x hx
    simp only [Finset.mem_inter] at hx ⊢
    refine ⟨hx.2, ?_⟩
    rcases x with ⟨t, u⟩ | r
    · simp only [hZ, Finset.inl_mem_disjSum, Finset.mem_product, Finset.mem_univ, and_true, hQ,
        Finset.mem_inter] at hx
      simpa using hx.1.2
    · simp
  have hsplit : Mis.card + (Z ∩ Ob).card = Z.card := Finset.card_sdiff_add_card_inter Z Ob
  have hzo := Finset.card_le_card hZU
  have hmis : k * m < Mis.card := by
    have e1 : Q.card * (m * 256) = Q.card * m * 256 := by ring
    have e2 : k * (m * 256) = k * m * 256 := by ring
    rw [e1, e2] at hunpaid
    rw [hZcard] at hsplit
    generalize Q.card * m = Y at *
    generalize k * m = Km at *
    omega
  -- the tiles short of a sector, and the stream sectors short
  set Tm := Q.filter fun t => ∃ u, (Sum.inl (t, u) : Pos N m ρ) ∈ Mis with hTm
  set Sm := (Finset.univ : Finset (Fin ρ)).filter fun r => (Sum.inr r : Pos N m ρ) ∈ Mis with hSm
  have hMisSub : Mis ⊆ (Tm ×ˢ (Finset.univ : Finset (Fin m))).disjSum Sm := by
    intro x hx
    rcases x with ⟨t, u⟩ | r
    · have hxZ := (Finset.mem_sdiff.mp hx).1
      simp only [hZ, Finset.inl_mem_disjSum, Finset.mem_product, Finset.mem_univ, and_true] at hxZ
      simp only [Finset.inl_mem_disjSum, Finset.mem_product, Finset.mem_univ, and_true, hTm, Finset.mem_filter]
      exact ⟨hxZ, u, hx⟩
    · simp only [Finset.inr_mem_disjSum, hSm, Finset.mem_filter, Finset.mem_univ, true_and]
      exact hx
  have hMisCard : Mis.card ≤ Tm.card * m + Sm.card := by
    have := Finset.card_le_card hMisSub
    simpa [Finset.card_disjSum, Finset.card_product] using this
  have hK : k + 1 ≤ Tm.card + Sm.card := by
    by_contra hlt
    push Not at hlt
    have : Tm.card * m + Sm.card ≤ k * m :=
      calc Tm.card * m + Sm.card ≤ Tm.card * m + Sm.card * m :=
            Nat.add_le_add_left (Nat.le_mul_of_pos_right _ hm) _
        _ = (Tm.card + Sm.card) * m := by ring
        _ ≤ k * m := Nat.mul_le_mul_right m (by omega)
    omega
  -- one missing sector of each short tile
  let uf : Fin N → Fin m := fun t =>
    if hu : ∃ u, (Sum.inl (t, u) : Pos N m ρ) ∈ Mis then hu.choose else ⟨0, hm⟩
  have huf : ∀ t ∈ Tm, (Sum.inl (t, uf t) : Pos N m ρ) ∈ Mis := by
    intro t ht
    have hu : ∃ u, (Sum.inl (t, u) : Pos N m ρ) ∈ Mis := (Finset.mem_filter.mp ht).2
    simp only [uf, hu, ↓reduceDIte]
    exact hu.choose_spec
  set Ps : Finset (Pos N m ρ) := (Tm.image fun t => (t, uf t)).disjSum Sm with hPs
  have hPsCard : Ps.card = Tm.card + Sm.card := by
    rw [hPs, Finset.card_disjSum, Finset.card_image_of_injective]
    intro t t' he
    exact (Prod.mk.inj he).1
  have hPsMis : Ps ⊆ Mis := by
    intro x hx
    rcases x with a | r
    · simp only [hPs, Finset.inl_mem_disjSum, Finset.mem_image] at hx
      obtain ⟨t, ht, rfl⟩ := hx
      exact huf t ht
    · simp only [hPs, Finset.inr_mem_disjSum] at hx
      exact (Finset.mem_filter.mp hx).2
  have hPsTile : ∀ a a' : Fin N × Fin m, (Sum.inl a : Pos N m ρ) ∈ Ps → (Sum.inl a' : Pos N m ρ) ∈ Ps →
      a.1 = a'.1 → a = a' := by
    intro a a' ha ha' he
    simp only [hPs, Finset.inl_mem_disjSum, Finset.mem_image] at ha ha'
    obtain ⟨t, -, rfl⟩ := ha
    obtain ⟨t', -, rfl⟩ := ha'
    simp only at he
    rw [he]
  obtain ⟨Ps', hPs', hPs'card⟩ := Finset.exists_subset_card_eq (hK.trans hPsCard.ge)
  let p : Fin (k + 1) → Pos N m ρ := fun i => (Ps'.equivFin.symm (Fin.cast hPs'card.symm i)).1
  have hpmem : ∀ i, p i ∈ Ps := fun i => hPs' (Ps'.equivFin.symm (Fin.cast hPs'card.symm i)).2
  have hpinj : Function.Injective p := by
    intro i i' he
    have := Subtype.ext he
    simpa using this
  -- a correct query of a tile of `Q`, for the stream's sectors
  have hQne : Q.Nonempty := by
    rw [← Finset.card_pos]
    by_contra h0
    push Not at h0
    have : Q.card = 0 := by omega
    rw [this, zero_mul] at hunpaid
    omega
  obtain ⟨t₀, ht₀⟩ := hQne
  let tileOf : Pos N m ρ → Fin N := Sum.elim Prod.fst fun _ => t₀
  have htileQ : ∀ x ∈ Ps, tileOf x ∈ Q := by
    intro x hx
    rcases x with a | r
    · have hxZ := (Finset.mem_sdiff.mp (hPsMis hx)).1
      simp only [hZ, Finset.inl_mem_disjSum, Finset.mem_product, Finset.mem_univ, and_true] at hxZ
      exact hxZ
    · exact ht₀
  let qs := (A.prog σ D).queries O D
  let yq : Fin N → (Fin ρ → X) × Fin N × Dg := fun t => (D.stream, t, dig σ (D.tile t))
  have hyq : ∀ t ∈ Q, yq t ∈ qs := by
    intro t ht
    have := hall t (Finset.mem_inter.mp ht).1
    simpa [queried, yq, qs] using this
  have hlen : qs.length ≤ q := (A.bounded σ (fun p => D (.inl p.1)) O D).2
  have hidx : ∀ i, qs.idxOf (yq (tileOf (p i))) < q := fun i =>
    (List.idxOf_lt_length_of_mem (hyq _ (htileQ _ (hpmem i)))).trans_le hlen
  have hget : ∀ i, qs[qs.idxOf (yq (tileOf (p i)))]? = some (yq (tileOf (p i))) := by
    intro i
    have hlt := List.idxOf_lt_length_of_mem (hyq _ (htileQ _ (hpmem i)))
    rw [List.getElem?_eq_getElem hlt, List.getElem_idxOf hlt]
  refine ⟨fun i => ⟨_, hidx i⟩, ⟨p, ⟨hpinj, ?_⟩, fun i => ⟨?_, yq (tileOf (p i)), hget i, ?_⟩⟩,
    fun i => ⟨yq (tileOf (p i)), hget i, rfl⟩⟩
  · intro i i' a a' hi hi' he
    apply hpinj
    rw [hi, hi', hPsTile a a' (hi ▸ hpmem i) (hi' ▸ hpmem i') he]
  · exact (Finset.mem_sdiff.mp (hPsMis (hpmem i))).2
  · rcases hpi : p i with a | r
    · simp [relevant, yq, tileOf]
    · simp [relevant]

end Accounting.RateSketch
