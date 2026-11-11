module

public import Properties.Pouw.PeriodJ10R
public import Definitions.Accounting.TS.Hashed
public import Proofs.Accounting.Compose.BindingOfECR
public import Proofs.Accounting.RemoteHeld.Binding
public import Proofs.Hash.RowBinding
public import Proofs.Accounting.TS.FixedMeets
public import Definitions.Accounting.Guarantees.PeriodBinding

/-!
# A sampled layer's binding from its hashed openings

The reduction behind binding's premise at a sampled window's proofs whose openings are frame-v3-sha512 paths
(`Accounting.TS.HashedOpenings`), for any `SampledLayer`, at J10-R's (`tileLayer`), and at every period's
(`Declared.periodLayer`) as the guarantee `Pouw.SecurityProofs.PeriodBindingPerProver`, with its two witnesses. The
open terms are explicit.

* **A fork is a break, an unverified opening or a split** (`fork_cases`). Two accepted openings of one position
  `ℓ ∈ L` to different values on a run `x` and on its continuation from `ℓ`'s cut (`SampledLayer.Forked`), each
  `Verified` at one verifier, root and rank (no `Split`), have different rows, since a row decodes to the value it
  opens; so the two checks hashed a SHA-512 collision (`Hash.SecurityProofs.RowBinding`), which the fork finder
  outputs (`HashedOpenings.Finds`).
* **The fork's probability** (`forked_le`): at most the finder's success (`Claims.Finder.Pr`) plus
  `2·Σ_{ℓ ∈ L} Pr[Unverified at ℓ]` plus `Pr[some ℓ ∈ L splits]`. A continuation from `ℓ`'s cut is uniform
  (`prob_cont`: `(x, z) ↦ (cont x z, cont z x)` is an involution), so the unverified opening on either run is
  `Pr[Unverified at ℓ]` once per position.
* **Binding** (`movedIn_le`, `movedIn_le_sqrt`). With `κ` the last two terms, `BindingOfECR` gives
  `Pr[MovedIn w] ≤ δ' + |L|²/(4δ')·(ecr(finder) + κ)`, and under `ecr/sha-512` at the finder's `q` calls
  `SampledBindingOfFork` gives `Pr[MovedIn w] ≤ |L|·√(q/2^256 + κ)`.
* **At J10-R** (`tile_binding`): `tileLayer`'s `BindingAt` at that bound, the binding premise of
  `H1J10RCheckedCreditPeriod`, for any relation and rightness, since `MovedIn` reads neither.
* **At every period** (`Pouw.SecurityProofs.PeriodBindingPerProver`, `Pouw.Guarantees.PeriodBindingPerProver`): the
  same bound at `Declared.periodLayer` for any circuit and outcomes, at `L` the period's positions
  (`Declared.periodPositions`, which hold every slot: `periodPositions_mem`). Its honest witness is `onePeriod` under
  `honestOutcomes`, where `κ = 0` and binding never fails; its nontrivial one is `onePeriod` under
  `acceptAllOutcomes`, where `κ = 0` and the opened coin bit moves with every fresh sample (`acceptAll_moved`).

Neither open term is bounded here: no premise ties an accepted proof's `opened` to a verified opening of its
position's leaf, and none fixes a position's verifier, root and rank by its cut.
-/

@[expose] public section

namespace Properties.Pouw.PeriodJ10R.SecurityProofs.Binding

open Accounting.TS Accounting.Compose Pouw.RelativeSecurity

section Layer

variable {C B : Type} [Fintype C] [Fintype B] {S m s : ℕ} {J Q Pos V : Type}
  (hl : HashedOpenings C B S m s Q Pos V) (pf : SampledLayer C B S m s J Q Pos V)

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
unverified and split terms (`SampledBindingOfFork` at `forked_le`). -/
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
  have hsq := Accounting.SecurityProofs.TS.SampledBindingOfFork C B S m s pf w L hL
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

/-- Every slot of a period's tile is at one of the period's positions. -/
theorem periodPositions_mem {ι C Ξ V : Type} (Cd : Declared ι) {n Np Kd : ℕ} (hr : 0 < Cd.rounds) (hK : Kd ≤ Np)
    (pr : Outcomes C (Equiv.Perm (Fin Np) × Ξ) Cd.rounds Kd Np n V) (w : Fin n) (t : Fin Np) (k : ℕ) :
    (Cd.periodLayer hr hK pr).pos w t k ∈ Cd.periodPositions n Np := by
  show (w, (Cd.slots t.val)[k]?.getD ((Cd.slots t.val).headD Cd.dag.length)) ∈ _
  refine Finset.mem_product.2 ⟨Finset.mem_univ _, Finset.mem_biUnion.2 ⟨t, Finset.mem_univ _, ?_⟩⟩
  cases h : (Cd.slots t.val)[k]? with
  | none => exact Finset.mem_insert_self _ _
  | some g => exact Finset.mem_insert_of_mem (List.mem_toFinset.2 (List.mem_of_getElem? h))

section OnePeriod

open Accounting.Compose.PeriodBinding

theorem oneProb_zero {Ω : Type} [Fintype Ω] {p : Ω → Prop} (h : ∀ ω, ¬ p ω) : prob p = 0 := by
  classical
  unfold prob Claims.prUniform
  rw [Finset.filter_false_of_mem fun ω _ => h ω, Finset.card_empty, Nat.cast_zero, zero_div]

theorem oneProb_one {Ω : Type} [Fintype Ω] [Nonempty Ω] {p : Ω → Prop} (h : ∀ ω, p ω) : prob p = 1 := by
  classical
  unfold prob Claims.prUniform
  rw [Finset.filter_true_of_mem fun ω _ => h ω, Finset.card_univ]
  exact div_self (Nat.cast_ne_zero.mpr Fintype.card_ne_zero)

theorem oneProb_pos {Ω : Type} [Fintype Ω] {p : Ω → Prop} (ω : Ω) (h : p ω) : 0 < prob p := by
  classical
  unfold prob Claims.prUniform
  exact div_pos (Nat.cast_pos.mpr (Finset.card_pos.mpr ⟨ω, Finset.mem_filter.mpr ⟨Finset.mem_univ _, h⟩⟩))
    (Nat.cast_pos.mpr (Fintype.card_pos_iff.mpr ⟨ω⟩))

/-- A plurality reads its continuations only. -/
theorem plural_of_sp {X W : Type} [Fintype X] [Nonempty W] (sp : X → X → X) (opens : X → W → Prop) {x x' : X}
    (h : sp x = sp x') : plural sp opens x = plural sp opens x' := by
  unfold plural tally
  rw [h]

/-- A path of `onePeriod`'s coins with coin bit `b`. -/
def oneCoins (b : Bool) : OneCoins := (((b, ()), (1, ())), ())

instance instNonemptyOneCoins_proofs : Nonempty OneCoins := ⟨oneCoins false⟩

variable (o : Outcomes Bool (Equiv.Perm (Fin 1) × Unit) 1 1 1 1 Bool)

/-- `onePeriod`'s tile reads the gate `dag.length = 0` at every slot. -/
theorem onePeriod_pos (w t : Fin 1) (k : ℕ) : (onePeriod.periodLayer Nat.one_pos le_rfl o).pos w t k = (w, 0) := by
  simp [Declared.periodLayer, Declared.slots, unitGates, onePeriod]

/-- Every position of `onePeriod` is cut before the coin bit, so its continuations are the fresh coins. -/
theorem onePeriod_cont (ℓ : Fin 1 × ℕ) (x z : OneCoins) :
    (onePeriod.periodLayer Nat.one_pos le_rfl o).cont ℓ x z = z := by
  obtain ⟨w, g⟩ := ℓ
  obtain rfl : w = 0 := Subsingleton.elim _ _
  have hc : (((onePeriod.periodLayer Nat.one_pos le_rfl o).cut (0, g)).2 : ℕ) = 0 := by
    simp [Declared.periodLayer, Declared.cutOf, Declared.binding, coinMark, onePeriod]
  show spliceAt (((onePeriod.periodLayer Nat.one_pos le_rfl o).cut (0, g)).1 : ℕ)
    (((onePeriod.periodLayer Nat.one_pos le_rfl o).cut (0, g)).2 : ℕ) x z = z
  rw [hc]
  rfl

/-- An opening of the value `v x` by `oneOpenings v` is verified. -/
theorem one_verified (v : OneCoins → Bool) (x : OneCoins) (w i : Fin 1) (k : ℕ) (hv : o.opened x w i k = v x) :
    (oneOpenings v).Verified (onePeriod.periodLayer Nat.one_pos le_rfl o) zeroHash x w i k := by
  refine ⟨⟨?_, ?_, ?_⟩, hv.symm⟩
  · show 1 < 2 ^ 64
    norm_num
  · simp [oneOpenings]
  · exact ⟨Nat.one_pos, rfl, rfl⟩

theorem one_unverified (v : OneCoins → Bool) (hv : ∀ x w i k, o.opened x w i k = v x) (x : OneCoins)
    (ℓ : Fin 1 × ℕ) : ¬ (oneOpenings v).Unverified (onePeriod.periodLayer Nat.one_pos le_rfl o) zeroHash x ℓ := by
  rintro ⟨w, i, k, -, -, hn⟩
  exact hn (one_verified o v x w i k (hv x w i k))

theorem one_split (v : OneCoins → Bool) (p : OneCoins × OneCoins) :
    ¬ ∃ ℓ ∈ onePeriod.periodPositions 1 1,
      (oneOpenings v).Split (onePeriod.periodLayer Nat.one_pos le_rfl o) p.1 p.2 ℓ := by
  rintro ⟨_, -, hs⟩
  exact hs ⟨rfl, rfl, rfl⟩

/-- Under `acceptAllOutcomes` a position's committed value is the same on every path. -/
theorem acceptAll_committed (x x' : OneCoins) (ℓ : Fin 1 × ℕ) :
    acceptAllLayer.committed x ℓ = acceptAllLayer.committed x' ℓ := by
  unfold SampledLayer.committed
  exact plural_of_sp _ _ (funext fun z => (onePeriod_cont acceptAllOutcomes ℓ x z).trans
    (onePeriod_cont acceptAllOutcomes ℓ x' z).symm)

/-- Under `acceptAllOutcomes` binding fails on the path whose coin bit isn't the committed value. -/
theorem acceptAll_moved : ∃ x, acceptAllLayer.MovedIn x 0 := by
  refine ⟨oneCoins (!acceptAllLayer.committed (oneCoins false) (0, 0)), 0, trivial, fun h => ?_⟩
  have h0 := congrFun h 0
  change _ = acceptAllLayer.committed _ ((onePeriod.periodLayer Nat.one_pos le_rfl acceptAllOutcomes).pos 0 _ 0) at h0
  rw [onePeriod_pos, acceptAll_committed _ (oneCoins false)] at h0
  have h1 : (!acceptAllLayer.committed (oneCoins false) (0, 0)) = acceptAllLayer.committed (oneCoins false) (0, 0) :=
    h0
  revert h1
  cases acceptAllLayer.committed (oneCoins false) (0, 0) <;> decide

/-- Under `honestOutcomes` every position a tile reads is committed to `false`. -/
theorem honest_committed (x : OneCoins) (w : Fin 1) : honestLayer.committed x (w, 0) = false := by
  unfold SampledLayer.committed
  refine Accounting.SecurityProofs.TS.FixedMeets.plural_of_val _ _ (fun _ => false) (fun y v => ⟨?_, ?_⟩) x
    fun _ => rfl
  · rintro ⟨_, _, _, -, -, hv⟩
    exact hv.symm
  · intro hv
    exact ⟨w, 0, 0, onePeriod_pos honestOutcomes w _ 0, trivial, hv.symm⟩

end OnePeriod

end Properties.Pouw.PeriodJ10R.SecurityProofs.Binding

namespace Pouw.SecurityProofs

open Accounting.Compose Accounting.Compose.PeriodBinding Accounting.TS Pouw.RelativeSecurity
  Properties.Pouw.PeriodJ10R.SecurityProofs.Binding

/-- **A period's binding per prover** (`movedIn_le_sqrt` at the period's positions). -/
theorem PeriodBindingPerProver : Pouw.Guarantees.PeriodBindingPerProver := by
  intro ι C Ξ V _ _ _ Cd _ n Np Kd hK pr _ hl H q hcr w
  exact movedIn_le_sqrt hl _ H _ q w (fun j k => periodPositions_mem Cd Fact.out hK pr w j k) hcr

theorem PeriodBindingPerProver_honest : Pouw.Guarantees.PeriodBindingPerProver_honest := by
  have hU : ∀ ℓ, prob (fun x => (oneOpenings fun _ => false).Unverified honestLayer zeroHash x ℓ) = 0 :=
    fun ℓ => oneProb_zero fun x => one_unverified honestOutcomes _ (fun _ _ _ _ => rfl) x ℓ
  have hS : prob (fun p : OneCoins × OneCoins =>
      ∃ ℓ ∈ onePeriod.periodPositions 1 1, (oneOpenings fun _ => false).Split honestLayer p.1 p.2 ℓ) = 0 :=
    oneProb_zero (one_split honestOutcomes _)
  refine ⟨oneProb_one fun _ _ => trivial, ?_, ?_⟩
  · simp only [hU, hS, Finset.sum_const_zero, mul_zero, add_zero]
  · intro w
    refine le_of_eq (oneProb_zero fun x hx => ?_)
    obtain ⟨i, -, hi⟩ := hx
    refine hi (funext fun k => ?_)
    show false = honestLayer.committed x ((onePeriod.periodLayer Nat.one_pos le_rfl honestOutcomes).pos w _ k)
    rw [onePeriod_pos]
    exact (honest_committed x w).symm

theorem PeriodBindingPerProver_nontrivial : Pouw.Guarantees.PeriodBindingPerProver_nontrivial := by
  intro h
  have hU : ∀ ℓ, prob (fun x => (oneOpenings coinBit).Unverified acceptAllLayer zeroHash x ℓ) = 0 :=
    fun ℓ => oneProb_zero fun x => one_unverified acceptAllOutcomes _ (fun _ _ _ _ => rfl) x ℓ
  have hS : prob (fun p : OneCoins × OneCoins =>
      ∃ ℓ ∈ onePeriod.periodPositions 1 1, (oneOpenings coinBit).Split acceptAllLayer p.1 p.2 ℓ) = 0 :=
    oneProb_zero (one_split acceptAllOutcomes _)
  have h0 := h 0
  simp only [hU, hS, Finset.sum_const_zero, mul_zero, add_zero, Nat.cast_zero, zero_div, Real.sqrt_zero] at h0
  obtain ⟨x, hx⟩ := acceptAll_moved
  exact absurd h0 (not_le.2 (oneProb_pos x hx))

end Pouw.SecurityProofs
