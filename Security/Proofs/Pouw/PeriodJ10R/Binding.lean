module

public import Properties.Pouw.PeriodJ10R
public import Definitions.Accounting.TS.Hashed
public import Proofs.Accounting.Compose.BindingOfECR
public import Proofs.Accounting.RemoteHeld.Binding
public import Proofs.Hash.RowBinding

/-!
# A drawn layer's binding from its hashed openings

The reduction behind binding's premise at a sampled window's proofs whose openings are frame-v3-sha512 paths
(`Accounting.TS.HashedOpenings`), for any `DrawnLayer`, and at J10-R's (`tileLayer`). Lemmas, not guarantees: the
open terms are explicit.

* **A fork is a break, an unverified opening or a split** (`fork_cases`). Two accepted openings of one position
  `ℓ ∈ L` to different values on a run `x` and on its continuation from `ℓ`'s cut (`DrawnLayer.Forked`), each
  `Verified` at one verifier, root and rank (no `Split`), have different rows, since a row decodes to the value it
  opens; so the two checks hashed a SHA-512 collision (`Hash.SecurityProofs.RowBinding`), which the fork finder
  outputs (`HashedOpenings.Finds`).
* **The fork's probability** (`forked_le`): at most the finder's success (`Claims.Finder.Pr`) plus
  `2·Σ_{ℓ ∈ L} Pr[Unverified at ℓ]` plus `Pr[some ℓ ∈ L splits]`. A continuation from `ℓ`'s cut is uniform
  (`prob_cont`: `(x, z) ↦ (cont x z, cont z x)` is an involution), so the unverified opening on either run is
  `Pr[Unverified at ℓ]` once per position.
* **Binding** (`movedIn_le`, `movedIn_le_sqrt`). With `κ` the last two terms, `BindingOfECR` gives
  `Pr[MovedIn w] ≤ δ' + |L|²/(4δ')·(ecr(finder) + κ)`, and under `ecr/sha-512` at the finder's `q` calls
  `DrawnBindingOfFork` gives `Pr[MovedIn w] ≤ |L|·√(q/2^256 + κ)`.
* **At J10-R** (`tile_binding`): `tileLayer`'s `BindingAt` at that bound, the binding premise of
  `H1J10RCheckedCreditPeriod`, for any relation and rightness, since `MovedIn` reads neither.

Neither open term is bounded here: no premise ties an accepted proof's `opened` to a verified opening of its
position's leaf, and none fixes a position's verifier, root and rank by its cut.
-/

@[expose] public section

namespace Properties.Pouw.PeriodJ10R.SecurityProofs.Binding

open Accounting.TS Accounting.Compose Pouw.RelativeSecurity

section Layer

variable {C B : Type} [Fintype C] [Fintype B] {S m s : ℕ} {J Q Pos V : Type}
  (hl : HashedOpenings C B S m s Q Pos V) (pf : DrawnLayer C B S m s J Q Pos V)

omit [Fintype C] [Fintype B] in
/-- **A fork at `ℓ ∈ L` is a break the fork finder outputs, an unverified opening on either run, or a split.** -/
theorem fork_cases (H : Hash.Msg → Hash.Digest) {L : Finset Pos} {ℓ : Pos} (hℓ : ℓ ∈ L)
    {x z : Paths (Paths C m × B) S} (h : pf.Forked ℓ x z) :
    (∃ uv, hl.Finds pf H L x z uv) ∨ hl.Unverified pf H x ℓ ∨ hl.Unverified pf H (pf.cont ℓ x z) ℓ ∨
      hl.Split pf x z ℓ := by
  obtain ⟨_, _, ⟨w, i, k, hp, hA, rfl⟩, ⟨w', i', k', hp', hA', rfl⟩, hne⟩ := h
  by_cases hv : hl.Verified pf H x w i k
  swap
  · exact Or.inr (Or.inl ⟨w, i, k, hp, hA, hv⟩)
  by_cases hv' : hl.Verified pf H (pf.cont ℓ x z) w' i' k'
  swap
  · exact Or.inr (Or.inr (Or.inl ⟨w', i', k', hp', hA', hv'⟩))
  by_cases hs : hl.Split pf x z ℓ
  · exact Or.inr (Or.inr (Or.inr hs))
  left
  have hfix : hl.leaf (pf.cont ℓ x z) ℓ = hl.leaf x ℓ ∧ hl.root (pf.cont ℓ x z) ℓ = hl.root x ℓ ∧
      hl.rank (pf.cont ℓ x z) ℓ = hl.rank x ℓ := by
    by_contra hn
    exact hs hn
  obtain ⟨hleaf, hroot, hrank⟩ := hfix
  obtain ⟨hac, hdec⟩ := hv
  obtain ⟨hac', hdec'⟩ := hv'
  rw [hp] at hac
  rw [hp', hleaf, hroot, hrank] at hac'
  have hmem : ∀ a,
      (a ∈ Hash.RowLeaf.outerHashed H H (hl.leaf x ℓ) (hl.rank x ℓ) (hl.opening x w i k) ++
          Hash.RowLeaf.outerHashed H H (hl.leaf x ℓ) (hl.rank x ℓ) (hl.opening (pf.cont ℓ x z) w' i' k') ∨
        a ∈ Hash.RowLeaf.innerHashed H (hl.leaf x ℓ) (hl.opening x w i k) ++
          Hash.RowLeaf.innerHashed H (hl.leaf x ℓ) (hl.opening (pf.cont ℓ x z) w' i' k')) →
      a ∈ hl.checked H x ℓ (hl.opening x w i k) ++
        hl.checked H (pf.cont ℓ x z) ℓ (hl.opening (pf.cont ℓ x z) w' i' k') := by
    intro a ha
    simp only [HashedOpenings.checked, hleaf, hrank, List.mem_append] at ha ⊢
    tauto
  rcases Hash.SecurityProofs.RowBinding H H _ _ _ _ _ hac hac' with hrow | ⟨a, ha, b, hb, hab⟩ |
      ⟨a, ha, b, hb, hab⟩
  · exact (hne (by rw [← hdec, ← hdec', hrow])).elim
  · exact ⟨(a, b), hab, ℓ, hℓ, w, i, k, w', i', k', hp, hA, hp', hA', hmem a (Or.inl ha), hmem b (Or.inl hb)⟩
  · exact ⟨(a, b), hab, ℓ, hℓ, w, i, k, w', i', k', hp, hA, hp', hA', hmem a (Or.inr ha), hmem b (Or.inr hb)⟩

/-- A continuation from `ℓ`'s cut on fresh coins is a uniform path. -/
theorem prob_cont (ℓ : Pos) (P : Paths (Paths C m × B) S → Prop) :
    prob (fun p : Paths (Paths C m × B) S × Paths (Paths C m × B) S => P (pf.cont ℓ p.1 p.2)) =
      prob (fun p : Paths (Paths C m × B) S × Paths (Paths C m × B) S => P p.1) := by
  have hs : ∀ x z, pf.cont ℓ (pf.cont ℓ x z) (pf.cont ℓ z x) = x :=
    Accounting.SecurityProofs.TS.spliceAt_swap (pf.cut ℓ).1 (pf.cut ℓ).2
  let Φ : Paths (Paths C m × B) S × Paths (Paths C m × B) S → Paths (Paths C m × B) S × Paths (Paths C m × B) S :=
    fun p => (pf.cont ℓ p.1 p.2, pf.cont ℓ p.2 p.1)
  have hΦ : Function.Involutive Φ := fun p => by simp only [Φ, hs]
  rw [prob_equiv (hΦ.toPerm Φ) (fun p => P p.1)]
  rfl

/-- What the fork finder finds is at most its success. -/
theorem finds_le [Nonempty (Paths (Paths C m × B) S)] (H : Hash.Msg → Hash.Digest) (L : Finset Pos) (q : ℕ) :
    prob (fun p : Paths (Paths C m × B) S × Paths (Paths C m × B) S => ∃ uv, hl.Finds pf H L p.1 p.2 uv) ≤
      (hl.finder pf H L q).Pr (fun u => (H u).bytes) := by
  rw [Claims.Finder.Pr, Accounting.SecurityProofs.Remote.Binding.claims_pr]
  refine prob_mono (Ω := Paths (Paths C m × B) S × Paths (Paths C m × B) S) fun y hy => ?_
  refine ⟨hy.choose.1, hy.choose.2, ?_, hy.choose_spec.1⟩
  simp [HashedOpenings.finder, hy]

/-- **The fork form's probability at `L`**: at most the fork finder's success, plus twice the unverified openings'
probability summed over `L`, plus the probability that some position of `L` splits. -/
theorem forked_le [Nonempty (Paths (Paths C m × B) S)] (H : Hash.Msg → Hash.Digest) (L : Finset Pos) (q : ℕ) :
    prob (fun p : Paths (Paths C m × B) S × Paths (Paths C m × B) S => ∃ ℓ ∈ L, pf.Forked ℓ p.1 p.2) ≤
      (hl.finder pf H L q).Pr (fun u => (H u).bytes) +
        (2 * ∑ ℓ ∈ L, prob (fun x => hl.Unverified pf H x ℓ) +
          prob (fun p : Paths (Paths C m × B) S × Paths (Paths C m × B) S => ∃ ℓ ∈ L, hl.Split pf p.1 p.2 ℓ)) := by
  have hU : prob (fun p : Paths (Paths C m × B) S × Paths (Paths C m × B) S =>
      (∃ ℓ ∈ L, hl.Unverified pf H p.1 ℓ) ∨ ∃ ℓ ∈ L, hl.Unverified pf H (pf.cont ℓ p.1 p.2) ℓ) ≤
      2 * ∑ ℓ ∈ L, prob (fun x => hl.Unverified pf H x ℓ) := by
    refine (prob_or_le _ _).trans ?_
    rw [two_mul]
    refine add_le_add ((prob_exists_le L _).trans_eq (Finset.sum_congr rfl fun ℓ _ => ?_))
      ((prob_exists_le L _).trans_eq (Finset.sum_congr rfl fun ℓ _ => ?_))
    · exact prob_fst (fun x => hl.Unverified pf H x ℓ)
    · exact (prob_cont pf ℓ (fun x => hl.Unverified pf H x ℓ)).trans (prob_fst (fun x => hl.Unverified pf H x ℓ))
  calc prob (fun p : Paths (Paths C m × B) S × Paths (Paths C m × B) S => ∃ ℓ ∈ L, pf.Forked ℓ p.1 p.2)
      ≤ prob (fun p : Paths (Paths C m × B) S × Paths (Paths C m × B) S => (∃ uv, hl.Finds pf H L p.1 p.2 uv) ∨
          (((∃ ℓ ∈ L, hl.Unverified pf H p.1 ℓ) ∨ ∃ ℓ ∈ L, hl.Unverified pf H (pf.cont ℓ p.1 p.2) ℓ) ∨
            ∃ ℓ ∈ L, hl.Split pf p.1 p.2 ℓ)) := by
        refine prob_mono fun p hp => ?_
        obtain ⟨ℓ, hℓ, hf⟩ := hp
        rcases fork_cases hl pf H hℓ hf with h | h | h | h
        · exact Or.inl h
        · exact Or.inr (Or.inl (Or.inl ⟨ℓ, hℓ, h⟩))
        · exact Or.inr (Or.inl (Or.inr ⟨ℓ, hℓ, h⟩))
        · exact Or.inr (Or.inr ⟨ℓ, hℓ, h⟩)
    _ ≤ _ := (prob_or_le _ _).trans (add_le_add (finds_le hl pf H L q)
        ((prob_or_le _ _).trans (add_le_add hU le_rfl)))

open Classical in
/-- **Binding through `ecr/sha-512` at the fork finder** (`Accounting.Guarantees.Compose.BindingOfECR` at
`forked_le`): for every `δ' > 0`, `Pr[MovedIn w] ≤ δ' + |L|²/(4δ')·(ecr(finder) + κ)`, `κ` the unverified and split
terms. -/
theorem movedIn_le [Nonempty V] [Nonempty (Paths (Paths C m × B) S)] (H : Hash.Msg → Hash.Digest) (L : Finset Pos)
    (q : ℕ) (w : Fin S) (hL : ∀ j k, pf.pos w j k ∈ L) (δ' : ℝ) (hδ : 0 < δ') :
    ENNReal.ofReal (prob (fun x => pf.MovedIn x w)) ≤
      ENNReal.ofReal δ' + ENNReal.ofReal ((L.card : ℝ) ^ 2 / (4 * δ')) *
        ((if Claims.ExpectedCR (fun u => (H u).bytes) (hl.finder pf H L q) then
            ENNReal.ofReal (Claims.Ex (fun c => ((hl.finder pf H L q).cost c : ℝ)) / 2 ^ (256 : ℝ))
          else (⊤ : ENNReal)) +
          ENNReal.ofReal (2 * ∑ ℓ ∈ L, prob (fun x => hl.Unverified pf H x ℓ) +
            prob (fun p : Paths (Paths C m × B) S × Paths (Paths C m × B) S => ∃ ℓ ∈ L, hl.Split pf p.1 p.2 ℓ))) :=
  Accounting.SecurityProofs.Compose.Binding.bindingOfECR C B S m s pf w L (fun u => (H u).bytes)
    (hl.finder pf H L q) _ δ' hL hδ (forked_le hl pf H L q)

/-- **Binding under `ecr/sha-512` at the fork finder's `q` calls**: `Pr[MovedIn w] ≤ |L|·√(q/2^256 + κ)`, `κ` the
unverified and split terms (`DrawnBindingOfFork` at `forked_le`). -/
theorem movedIn_le_sqrt [Nonempty V] [Nonempty (Paths (Paths C m × B) S)] (H : Hash.Msg → Hash.Digest)
    (L : Finset Pos) (q : ℕ) (w : Fin S) (hL : ∀ j k, pf.pos w j k ∈ L)
    (hcr : Claims.ExpectedCR (fun u => (H u).bytes) (hl.finder pf H L q)) :
    prob (fun x => pf.MovedIn x w) ≤
      L.card * √(q / 2 ^ (256 : ℝ) + (2 * ∑ ℓ ∈ L, prob (fun x => hl.Unverified pf H x ℓ) +
        prob (fun p : Paths (Paths C m × B) S × Paths (Paths C m × B) S => ∃ ℓ ∈ L, hl.Split pf p.1 p.2 ℓ))) := by
  have hEx : Claims.Ex (fun c => ((hl.finder pf H L q).cost c : ℝ)) = q := by
    have hcost : ∀ c, ((hl.finder pf H L q).cost c : ℝ) = q := fun _ => rfl
    have hc : (Fintype.card (hl.finder pf H L q).S : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr Fintype.card_ne_zero
    unfold Claims.Ex
    rw [div_eq_iff hc]
    simp only [hcost, Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
    ring
  have hP : (hl.finder pf H L q).Pr (fun u => (H u).bytes) ≤ q / 2 ^ (256 : ℝ) := by
    have := hcr
    unfold Claims.ExpectedCR at this
    rwa [hEx] at this
  have hsq := Accounting.SecurityProofs.TS.DrawnBindingOfFork C B S m s pf w L hL
  have hΦ := (forked_le hl pf H L q).trans (add_le_add hP le_rfl)
  calc prob (fun x => pf.MovedIn x w) = √(prob (fun x => pf.MovedIn x w) ^ 2) := (Real.sqrt_sq (prob_nonneg _)).symm
    _ ≤ √((L.card : ℝ) ^ 2 * (q / 2 ^ (256 : ℝ) + (2 * ∑ ℓ ∈ L, prob (fun x => hl.Unverified pf H x ℓ) +
          prob (fun p : Paths (Paths C m × B) S × Paths (Paths C m × B) S => ∃ ℓ ∈ L, hl.Split pf p.1 p.2 ℓ)))) :=
        Real.sqrt_le_sqrt (hsq.trans (mul_le_mul_of_nonneg_left hΦ (sq_nonneg _)))
    _ = _ := by rw [Real.sqrt_mul (sq_nonneg _), Real.sqrt_sq (Nat.cast_nonneg _)]

end Layer

/-- **J10-R's binding premise from its hashed openings**: for any relation and rightness, the period's proofs under
`Credit.window` (`tileLayer`) bind at `δ_bind = |L|·√(q/2^256 + κ)` (`BindingAt`, the binding premise of
`H1J10RCheckedCreditPeriod`) when `L` holds every round's slots and the fork finder at `L` and `q` calls meets
`ecr/sha-512`, `κ` the unverified and split terms. -/
theorem tile_binding {ι C Ξ V : Type} [Fintype C] [Fintype Ξ] [Nonempty V] (Cd : Declared ι) {nB mB n : ℕ}
    (hr : 0 < Cd.rounds) (σ : Fin nB → Fin mB) (k : Fin mB → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card) (sT : Fin mB)
    (unitOf : Fin nB → ℕ) (ws : CreditWindows (Coin C Ξ Cd.rounds σ sT) n)
    (opened : Paths (Coin C Ξ Cd.rounds σ sT) n → Fin n → ℕ → ℕ → V)
    (rel right : Paths (Coin C Ξ Cd.rounds σ sT) n → Fin n → Fin (tileCount σ sT) → (ℕ → V) → Prop)
    [Nonempty (Paths (Coin C Ξ Cd.rounds σ sT) n)]
    (hl : HashedOpenings C (Equiv.Perm (Fin (tileCount σ sT)) × Ξ) n Cd.rounds (k sT) ℕ (Fin n × ℕ) V)
    (H : Hash.Msg → Hash.Digest) (L : Finset (Fin n × ℕ)) (q : ℕ)
    (hL : ∀ w j i, (tileLayer Cd hr σ k hk sT unitOf ws opened rel right).pos w j i ∈ L)
    (hcr : Claims.ExpectedCR (fun u => (H u).bytes)
      (hl.finder (tileLayer Cd hr σ k hk sT unitOf ws opened rel right) H L q)) :
    (tileLayer Cd hr σ k hk sT unitOf ws opened rel right).BindingAt
      (L.card * √(q / 2 ^ (256 : ℝ) +
        (2 * ∑ ℓ ∈ L, prob (fun x => hl.Unverified (tileLayer Cd hr σ k hk sT unitOf ws opened rel right) H x ℓ) +
          prob (fun p : Paths (Coin C Ξ Cd.rounds σ sT) n × Paths (Coin C Ξ Cd.rounds σ sT) n =>
            ∃ ℓ ∈ L, hl.Split (tileLayer Cd hr σ k hk sT unitOf ws opened rel right) p.1 p.2 ℓ)))) := by
  intro w
  exact movedIn_le_sqrt hl _ H L q w (hL w) hcr

end Properties.Pouw.PeriodJ10R.SecurityProofs.Binding
