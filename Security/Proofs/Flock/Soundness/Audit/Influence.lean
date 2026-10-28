import Proofs.Flock.Soundness.Audit.TwoStage

/-!
# Influence through a separator

An accepted audit bounds which units are wrong; this file bounds what the wrong units can make the audited computation
deliver. `D` is the set of delivered wires: what the consumer receives (the served tokens). It can be smaller than the
circuit's outputs, since commitments and leaves are outputs that the verifier sees and nobody is delivered.

* `Circuit.FeedsAvoiding K g t`: `g` feeds `t` along a path none of whose gates is in `K`.
* `Partition.Separates K B D` (a downstream cut): `K` meets every path from a gate of a unit of `B` to a wire of `D`.
* `outputs_eq_of_separates`, **the influence cap**: two transcripts with the anchors' inputs, whose wrong units lie in
  `B`, and that agree on a committed separator of `B`, deliver the same values on `D`. So they deliver at most `2^|K|`
  values on `D` (`card_delivered_le`): in a Boolean circuit a gate is one bit, and `|K|` is the separator's width.
* `separates_exits`: the committed wires of the units of `B` separate `B`. For a replay unit, those are its exports.
  `Separates.mono` and `Separates.union` build separators for larger sets from smaller ones.
* `influenceSet D e a δ`: what the transcripts with the anchors' inputs deliver on `D` when their wrong units form a set
  whose escape `e B` is at least `δ`. It is fixed by the circuit, the partition, the anchors, the law and `δ`, never by
  the prover, and `card_influenceSet_le` bounds its size by `∑_{B : δ ≤ e B} 2^|K B|`.
* `audit_influence` (one-stage) and `Refines.twoStage_influence` (two-stage, over replay units, with `effEscape`): the
  audit accepts while the delivered values lie outside the influence set with probability at most
  `δ + ε_ks + δ_link + δ_in`. The influence in bits is `log₂ |influenceSet|`.
* `card_influenceSet_le_harm` and `card_admissible_le`: the influence is at most `H + log₂ ∑_{j ≤ K} C(n, j)` bits. `H` is
  a harm bound (`IsHarmBound`) with each unit's harm the width it adds to a separator, for example its exits
  (`card_exits_le`): the value bits. The second term is the choice of which units are wrong, when `miss (K + 1) < δ`.
* `audit_exfiltration`: a prover that picks its strategy from a uniform message `m`, and a receiver that sees only what
  is delivered, recover `m` from an accepted audit with probability at most `2^H · ∑_{j ≤ K} C(n, j) / |M|` plus the
  audit's error: the value bits and the location term together. `card_guess_le` is its counting step.
-/

namespace FlockSoundness.Audit

open Game Finset
open scoped ENNReal

namespace Circuit

variable (C : Circuit)

/-- `g` feeds `t` along a path none of whose gates is in `K`, both ends included. -/
inductive FeedsAvoiding (K : Finset (Fin C.N)) : Fin C.N → Fin C.N → Prop
  | refl {t : Fin C.N} : t ∉ K → FeedsAvoiding K t t
  | step {g h t : Fin C.N} : g ∉ K → g ∈ (C.op h).args → FeedsAvoiding K h t → FeedsAvoiding K g t

end Circuit

theorem Circuit.FeedsAvoiding.not_mem {C : Circuit} {K : Finset (Fin C.N)} {g t : Fin C.N}
    (h : C.FeedsAvoiding K g t) : g ∉ K := by
  cases h with
  | refl hK => exact hK
  | step hg _ _ => exact hg

/-- **Guessing through a small set of outputs**: a receiver `g` that sees only the delivered output `y m` recovers the
message `m`, with `y m ∈ Y`, for at most `|Y|` messages. -/
theorem card_guess_le {M O : Type} [Fintype M] [DecidableEq M] [DecidableEq O] (Y : Finset O) (y : M → O) (g : O → M) :
    (univ.filter fun m => y m ∈ Y ∧ g (y m) = m).card ≤ Y.card :=
  card_le_card_of_injOn y (fun _ hm => (mem_filter.1 hm).2.1) fun m hm m' hm' h =>
    calc m = g (y m) := (mem_filter.1 hm).2.2.symm
      _ = g (y m') := by rw [h]
      _ = m' := (mem_filter.1 hm').2.2

namespace Partition

variable {C : Circuit} {n : ℕ} (P : Partition C n)

/-- **A separator of the units `B` from the delivered wires `D`** (a downstream cut): every path from a gate of a unit
of `B` to a wire of `D` meets `K`. -/
def Separates (K : Finset (Fin C.N)) (B : Finset (Fin n)) (D : Finset (Fin C.N)) : Prop :=
  ∀ g t u, P.unit g = some u → u ∈ B → t ∈ D → ¬ C.FeedsAvoiding K g t

variable {P}

theorem Separates.mono {K K' : Finset (Fin C.N)} {B B' : Finset (Fin n)} {D : Finset (Fin C.N)}
    (h : P.Separates K B D) (hK : K ⊆ K') (hB : B' ⊆ B) : P.Separates K' B' D := by
  intro g t u hu huB ht hgt
  refine h g t u hu (hB huB) ht ?_
  clear hu huB ht
  induction hgt with
  | refl hn => exact .refl fun hk => hn (hK hk)
  | step hg hx _ ih => exact .step (fun hk => hg (hK hk)) hx ih

theorem Separates.union {K₁ K₂ : Finset (Fin C.N)} {B₁ B₂ : Finset (Fin n)} {D : Finset (Fin C.N)}
    (h₁ : P.Separates K₁ B₁ D) (h₂ : P.Separates K₂ B₂ D) : P.Separates (K₁ ∪ K₂) (B₁ ∪ B₂) D := by
  intro g t u hu huB ht hgt
  rcases mem_union.1 huB with hB | hB
  · exact (h₁.mono subset_union_left subset_rfl) g t u hu hB ht hgt
  · exact (h₂.mono subset_union_right subset_rfl) g t u hu hB ht hgt

variable (P)

open Classical in
/-- The committed wires of the units of `B`: what they export. -/
noncomputable def exits (B : Finset (Fin n)) : Finset (Fin C.N) :=
  P.committed.filter fun g => ∃ u ∈ B, P.unit g = some u

/-- **The exits separate**: every path from a unit of `B` to a delivered wire leaves `B` through a committed wire, or
ends at a delivered wire of `B`, which is committed. -/
theorem separates_exits (B : Finset (Fin n)) {D : Finset (Fin C.N)} (hD : D ⊆ P.committed) :
    P.Separates (P.exits B) B D := by
  classical
  intro g t u hu huB ht hgt
  induction hgt generalizing u with
  | @refl t hn =>
      exact hn (mem_filter.2 ⟨hD ht, u, huB, hu⟩)
  | @step g h t hg hx _ ih =>
      by_cases hh : P.unit h = some u
      · exact ih u hh huB ht
      · exact hg (mem_filter.2 ⟨P.mem_committed_of_read hx (by rw [hu]; exact hh), u, huB, hu⟩)

/-- The induction behind `outputs_eq_of_separates`: on a gate with a path to a delivered wire avoiding `K`, the two
transcripts' local evaluations agree, and so do their committed values. -/
theorem separator_induction {K : Finset (Fin C.N)} {B : Finset (Fin n)} {D : Finset (Fin C.N)}
    (hK : K ⊆ P.committed) (hsep : P.Separates K B D) {X Y a : Fin C.N → Bool}
    (hX : C.InputsAgree X a) (hY : C.InputsAgree Y a) (hwX : P.wrong X ⊆ B) (hwY : P.wrong Y ⊆ B)
    (hXY : ∀ g ∈ K, X g = Y g) :
    ∀ m (g : Fin C.N), g.val = m → (∃ t ∈ D, C.FeedsAvoiding K g t) →
      (∀ u, P.unit g = some u → P.localEval X u g = P.localEval Y u g) ∧
      (g ∈ P.committed → X g = Y g) := by
  intro m
  induction m using Nat.strong_induction_on with
  | _ m ih =>
  intro g hgm ⟨t, ht, hgt⟩
  have hcor : ∀ u, P.unit g = some u → P.Correct X u ∧ P.Correct Y u := by
    intro u hu
    have huB : u ∉ B := fun huB => hsep g t u hu huB ht hgt
    exact ⟨not_not.1 fun hc => huB (hwX (P.mem_wrong.2 hc)),
      not_not.1 fun hc => huB (hwY (P.mem_wrong.2 hc))⟩
  have hB : ∀ u, P.unit g = some u → P.localEval X u g = P.localEval Y u g := by
    intro u hu
    rw [P.localEval_of_mem X hu, P.localEval_of_mem Y hu]
    refine Op.apply_congr _ fun x hx => ?_
    have hxg : x.val < m := hgm ▸ C.topo g x hx
    by_cases hxK : x ∈ K
    · by_cases hxu : P.unit x = some u
      · rw [← (hcor u hu).1 x (hK hxK) hxu, ← (hcor u hu).2 x (hK hxK) hxu]
        exact hXY x hxK
      · rw [P.localEval_of_not_mem X hxu, P.localEval_of_not_mem Y hxu]
        exact hXY x hxK
    · obtain ⟨ihB, ihA⟩ := ih x.val hxg x rfl ⟨t, ht, .step hxK hx hgt⟩
      by_cases hxu : P.unit x = some u
      · exact ihB u hxu
      · rw [P.localEval_of_not_mem X hxu, P.localEval_of_not_mem Y hxu]
        exact ihA (P.mem_committed_of_read hx (by rw [hu]; exact Ne.symm hxu))
  refine ⟨hB, fun hg => ?_⟩
  cases hi : (C.op g).isInput
  · obtain ⟨u, hu⟩ := Option.ne_none_iff_exists'.1 fun h => by
      simp [(P.input_iff g).1 h] at hi
    rw [(hcor u hu).1 g hg hu, (hcor u hu).2 g hg hu]
    exact hB u hu
  · rw [hX g hi, hY g hi]

/-- **The influence cap**: two transcripts with the anchors' inputs, whose wrong units lie in `B`, and that agree on a
committed separator `K` of `B` from the committed delivered wires `D`, deliver the same values on `D`. -/
theorem outputs_eq_of_separates {K : Finset (Fin C.N)} {B : Finset (Fin n)} {D : Finset (Fin C.N)}
    (hK : K ⊆ P.committed) (hD : D ⊆ P.committed) (hsep : P.Separates K B D) {X Y a : Fin C.N → Bool}
    (hX : C.InputsAgree X a) (hY : C.InputsAgree Y a) (hwX : P.wrong X ⊆ B) (hwY : P.wrong Y ⊆ B)
    (hXY : ∀ g ∈ K, X g = Y g) : ∀ t ∈ D, X t = Y t := by
  intro t ht
  by_cases htK : t ∈ K
  · exact hXY t htK
  · exact (P.separator_induction hK hsep hX hY hwX hwY hXY t.val t rfl ⟨t, ht, .refl htK⟩).2 (hD ht)

/-- What a transcript delivers on the wires `D`. -/
def out (D : Finset (Fin C.N)) (X : Fin C.N → Bool) : D → Bool := fun t => X t

open Classical in
/-- What the transcripts with the anchors' inputs whose wrong units lie in `B` deliver on `D`. -/
noncomputable def delivered (D : Finset (Fin C.N)) (a : Fin C.N → Bool) (B : Finset (Fin n)) : Finset (D → Bool) :=
  univ.filter fun y => ∃ X, C.InputsAgree X a ∧ P.wrong X ⊆ B ∧ out D X = y

/-- **At most `2^|K|` delivered values** behind a committed separator `K` of `B`. -/
theorem card_delivered_le {K : Finset (Fin C.N)} {B : Finset (Fin n)} {D : Finset (Fin C.N)}
    (hK : K ⊆ P.committed) (hD : D ⊆ P.committed) (hsep : P.Separates K B D) (a : Fin C.N → Bool) :
    (P.delivered D a B).card ≤ 2 ^ K.card := by
  classical
  have hspec : ∀ y ∈ P.delivered D a B, ∃ X, C.InputsAgree X a ∧ P.wrong X ⊆ B ∧ out D X = y :=
    fun y hy => (mem_filter.1 hy).2
  let f : (D → Bool) → (K → Bool) := fun y =>
    if h : ∃ X, C.InputsAgree X a ∧ P.wrong X ⊆ B ∧ out D X = y then fun g => Classical.choose h g
    else fun _ => false
  calc (P.delivered D a B).card ≤ (univ : Finset (K → Bool)).card := by
        refine card_le_card_of_injOn f (fun _ _ => mem_coe.2 (mem_univ _)) ?_
        intro y hy y' hy' hf
        have h := hspec y hy
        have h' := hspec y' hy'
        simp only [f, h, h', ↓reduceDIte] at hf
        obtain ⟨hX, hwX, hyX⟩ := Classical.choose_spec h
        obtain ⟨hY, hwY, hyY⟩ := Classical.choose_spec h'
        rw [← hyX, ← hyY]
        funext t
        exact P.outputs_eq_of_separates hK hD hsep hX hY hwX hwY (fun g hg => congrFun hf ⟨g, hg⟩) t t.2
    _ = 2 ^ K.card := by rw [card_univ, Fintype.card_fun, Fintype.card_bool, Fintype.card_coe]

open Classical in
/-- **The influence set at `δ`**: what is delivered on `D` with the wrong units of some set whose escape `e B` is at
least `δ`. -/
noncomputable def influenceSet (D : Finset (Fin C.N)) (e : Finset (Fin n) → ℝ≥0∞) (a : Fin C.N → Bool)
    (δ : ℝ≥0∞) : Finset (D → Bool) :=
  (univ.filter fun B => δ ≤ e B).biUnion fun B => P.delivered D a B

/-- **The influence bound**: with a committed separator `K B` of every set `B`,
`|influenceSet| ≤ ∑_{B : δ ≤ e B} 2^|K B|`. -/
theorem card_influenceSet_le {D : Finset (Fin C.N)} (hD : D ⊆ P.committed) (e : Finset (Fin n) → ℝ≥0∞)
    (a : Fin C.N → Bool) (δ : ℝ≥0∞) (K : Finset (Fin n) → Finset (Fin C.N)) (hK : ∀ B, K B ⊆ P.committed)
    (hsep : ∀ B, P.Separates (K B) B D) :
    (P.influenceSet D e a δ).card ≤ ∑ B ∈ univ.filter (fun B => δ ≤ e B), 2 ^ (K B).card :=
  card_biUnion_le.trans (sum_le_sum fun B _ => P.card_delivered_le (hK B) hD (hsep B) a)

theorem out_mem_influenceSet {D : Finset (Fin C.N)} {e : Finset (Fin n) → ℝ≥0∞} {a X : Fin C.N → Bool}
    {δ : ℝ≥0∞} (hia : C.InputsAgree X a) (he : δ ≤ e (P.wrong X)) : out D X ∈ P.influenceSet D e a δ := by
  classical
  simp only [influenceSet, mem_biUnion, mem_filter, mem_univ, true_and]
  exact ⟨P.wrong X, he, mem_filter.2 ⟨mem_univ _, X, hia, subset_rfl, rfl⟩⟩

theorem exits_subset_biUnion (B : Finset (Fin n)) : P.exits B ⊆ B.biUnion fun u => P.exits {u} := by
  intro g hg
  obtain ⟨hc, u, hu, hgu⟩ := mem_filter.1 hg
  exact mem_biUnion.2 ⟨u, hu, mem_filter.2 ⟨hc, u, mem_singleton_self u, hgu⟩⟩

/-- A set's exits are at most the sum of its units' exits: each unit's exported width is a harm the exits separator
is covered by. -/
theorem card_exits_le (B : Finset (Fin n)) : (P.exits B).card ≤ ∑ u ∈ B, (P.exits {u}).card :=
  (card_le_card (P.exits_subset_biUnion B)).trans card_biUnion_le

section Harm

variable {L : Law n} {a : Fin C.N → Bool} {D : Finset (Fin C.N)}

/-- **Influence from a harm bound**: when every set's separator is no wider than the harm `hb` of its units and `H` is a
harm bound at `δ`, `|influenceSet| ≤ 2^H · #{B : δ ≤ escape B}`. -/
theorem card_influenceSet_le_harm (hD : D ⊆ P.committed) (hb : Fin n → ℕ) (K : Finset (Fin n) → Finset (Fin C.N))
    (hK : ∀ B, K B ⊆ P.committed) (hsep : ∀ B, P.Separates (K B) B D) (hw : ∀ B, (K B).card ≤ ∑ u ∈ B, hb u)
    {δ : ℝ≥0∞} {H : ℕ} (hH : IsHarmBound L (fun u => (hb u : ℝ)) δ H) :
    (P.influenceSet D L.escape a δ).card ≤ 2 ^ H * (univ.filter fun B => δ ≤ L.escape B).card := by
  refine (P.card_influenceSet_le hD L.escape a δ K hK hsep).trans ?_
  calc ∑ B ∈ univ.filter (fun B => δ ≤ L.escape B), 2 ^ (K B).card
      ≤ ∑ B ∈ univ.filter (fun B => δ ≤ L.escape B), 2 ^ H := by
        refine sum_le_sum fun B hB => pow_le_pow_right₀ (by norm_num) ((hw B).trans ?_)
        have h1 := hH B (mem_filter.1 hB).2
        have h2 : ((∑ u ∈ B, hb u : ℕ) : ℝ) ≤ (H : ℝ) := by push_cast; exact h1
        exact_mod_cast h2
    _ = 2 ^ H * (univ.filter fun B => δ ≤ L.escape B).card := by rw [sum_const, smul_eq_mul, mul_comm]

/-- **The location term**: when `miss (K + 1) < δ`, the sets escaping with probability at least `δ` hold at most `K`
units, so there are at most `∑_{j ≤ K} C(n, j)` of them. -/
theorem card_admissible_le {δ : ℝ≥0∞} {K : ℕ} (hK : L.miss (K + 1) < δ) :
    (univ.filter fun B : Finset (Fin n) => δ ≤ L.escape B).card ≤ ∑ j ∈ range (K + 1), n.choose j := by
  classical
  have hsub : (univ.filter fun B : Finset (Fin n) => δ ≤ L.escape B) ⊆
      (range (K + 1)).biUnion fun j => (univ : Finset (Fin n)).powersetCard j := by
    intro B hB
    have hle : B.card ≤ K := by
      by_contra hc
      exact absurd ((mem_filter.1 hB).2.trans (L.escape_le_miss (by omega))) (not_le.2 hK)
    exact mem_biUnion.2 ⟨B.card, mem_range.2 (by omega), mem_powersetCard.2 ⟨subset_univ _, rfl⟩⟩
  refine (card_le_card hsub).trans (card_biUnion_le.trans (le_of_eq (sum_congr rfl fun j _ => ?_)))
  rw [card_powersetCard, card_univ, Fintype.card_fin]

end Harm

section OneStage

variable {L : Law n} {Reg : Type} {session : Finset (Fin n) → Reg → Game Bool}
  {X : (R : Reg) → Cont L Reg session R → (Fin C.N → Bool)}
  {ext : (S : Finset (Fin n)) → (R : Reg) → Strategy (session S R) → Fin n → Option (Fin C.N → Bool)}
  {a : Fin C.N → Bool} {εks δlink δin : ℝ≥0∞} {D : Finset (Fin C.N)}

/-- **The influence of an accepted audit** (one-stage): the audit accepts while the committed transcript delivers on
`D` a value outside the influence set at `δ` with probability at most `δ + ε_ks + δ_link + δ_in`. -/
theorem audit_influence (hks : (P.analysis X ext).KnowledgeSound εks)
    (hlink : (P.analysis X ext).LinkSound δlink) (hin : AnchorsSound X a δin) (δ : ℝ≥0∞)
    (σ : Strategy (audit L Reg session)) :
    prob (fun o => o.1 = true ∧ out D (X (reg σ) (cont σ)) ∉ P.influenceSet D L.escape a δ)
        (audit L Reg session) σ ≤ δ + εks + δlink + δin := by
  refine (prob_split_inputs X a σ (fun _ => out D (X (reg σ) (cont σ)) ∉ P.influenceSet D L.escape a δ)).trans ?_
  rw [add_comm]
  refine add_le_add ?_ (hin σ)
  have key : ∀ o : Bool × Finset (Fin n), (o.1 = true ∧ (C.InputsAgree (X (reg σ) (cont σ)) a ∧
      out D (X (reg σ) (cont σ)) ∉ P.influenceSet D L.escape a δ)) →
      o.1 = true ∧ L.escape (P.wrong (X (reg σ) (cont σ))) < δ := by
    rintro o ⟨hacc, hia, hout⟩
    exact ⟨hacc, lt_of_not_ge fun hge => hout (P.out_mem_influenceSet hia hge)⟩
  refine (prob_mono key _ _).trans ?_
  refine (audit_profile (P.analysis X ext) hks hlink (fun B => L.escape B < δ) σ).trans ?_
  gcongr
  exact iSup₂_le fun B hB => hB.le

/-- The influence-set form of `audit_exfiltration`: a prover that picks its strategy `σ m` from a uniform message `m`,
and a receiver `g` that sees only what is delivered on `D`, succeed together (the audit accepts and `g` returns `m`) with
probability at most `|influenceSet|/|M| + δ + ε_ks + δ_link + δ_in`. -/
theorem audit_exfiltration_influenceSet {M : Type} [Fintype M] [Nonempty M] [DecidableEq M]
    (hks : (P.analysis X ext).KnowledgeSound εks) (hlink : (P.analysis X ext).LinkSound δlink)
    (hin : AnchorsSound X a δin) (δ : ℝ≥0∞) (σ : M → Strategy (audit L Reg session)) (g : (D → Bool) → M) :
    avg (fun m => prob (fun o => o.1 = true ∧ g (out D (X (reg (σ m)) (cont (σ m)))) = m)
        (audit L Reg session) (σ m)) ≤
      ((P.influenceSet D L.escape a δ).card : ℝ≥0∞) / (Fintype.card M : ℝ≥0∞) + (δ + εks + δlink + δin) := by
  classical
  let Y := P.influenceSet D L.escape a δ
  let y : M → (D → Bool) := fun m => out D (X (reg (σ m)) (cont (σ m)))
  have step : ∀ m, prob (fun o => o.1 = true ∧ g (y m) = m) (audit L Reg session) (σ m) ≤
      (if y m ∈ Y ∧ g (y m) = m then 1 else 0) +
        prob (fun o => o.1 = true ∧ y m ∉ Y) (audit L Reg session) (σ m) := by
    intro m
    refine (prob_mono (q := fun o => (o.1 = true ∧ (y m ∈ Y ∧ g (y m) = m)) ∨ (o.1 = true ∧ y m ∉ Y))
      (fun o h => ?_) _ _).trans ((prob_or_le _ _ _ _).trans (add_le_add ?_ le_rfl))
    · by_cases hy : y m ∈ Y
      · exact Or.inl ⟨h.1, hy, h.2⟩
      · exact Or.inr ⟨h.1, hy⟩
    · rw [prob_and_const]
      split_ifs
      · exact prob_le_one _ _ _
      · exact le_rfl
  calc avg (fun m => prob (fun o => o.1 = true ∧ g (y m) = m) (audit L Reg session) (σ m))
      ≤ avg (fun m => (if y m ∈ Y ∧ g (y m) = m then (1 : ℝ≥0∞) else 0) +
          prob (fun o => o.1 = true ∧ y m ∉ Y) (audit L Reg session) (σ m)) := avg_mono step
    _ = prCoin (fun m => y m ∈ Y ∧ g (y m) = m) +
          avg (fun m => prob (fun o => o.1 = true ∧ y m ∉ Y) (audit L Reg session) (σ m)) := by
        rw [avg_add, avg_ite_eq_prCoin]
    _ ≤ (Y.card : ℝ≥0∞) / (Fintype.card M : ℝ≥0∞) + (δ + εks + δlink + δin) := by
        refine add_le_add ?_ ((avg_mono fun m => P.audit_influence hks hlink hin δ (σ m)).trans
          (avg_const _).le)
        rw [prCoin_eq_card]
        gcongr
        exact_mod_cast card_guess_le Y y g

/-- **Exfiltration, with the location term**: a prover that picks its strategy `σ m` from a uniform message `m`, and a
receiver `g` that sees only what is delivered on `D`, succeed together with probability at most
`2^H · ∑_{j ≤ k} C(n, j) / |M| + δ + ε_ks + δ_link + δ_in`. Here `H` is a harm bound at `δ` of each unit's width `hb` on
a separator (the value bits) and `miss (k + 1) < δ` (which units are wrong). With `|M| = 2^ℓ`, the first term is
`2^(H + log₂ ∑_{j ≤ k} C(n, j) − ℓ)`. -/
theorem audit_exfiltration {M : Type} [Fintype M] [Nonempty M] [DecidableEq M]
    (hks : (P.analysis X ext).KnowledgeSound εks) (hlink : (P.analysis X ext).LinkSound δlink)
    (hin : AnchorsSound X a δin) (hD : D ⊆ P.committed) (hb : Fin n → ℕ) (K : Finset (Fin n) → Finset (Fin C.N))
    (hK : ∀ B, K B ⊆ P.committed) (hsep : ∀ B, P.Separates (K B) B D) (hw : ∀ B, (K B).card ≤ ∑ u ∈ B, hb u)
    {δ : ℝ≥0∞} {H k : ℕ} (hH : IsHarmBound L (fun u => (hb u : ℝ)) δ H) (hk : L.miss (k + 1) < δ)
    (σ : M → Strategy (audit L Reg session)) (g : (D → Bool) → M) :
    avg (fun m => prob (fun o => o.1 = true ∧ g (out D (X (reg (σ m)) (cont (σ m)))) = m)
        (audit L Reg session) (σ m)) ≤
      ((2 ^ H * ∑ j ∈ range (k + 1), n.choose j : ℕ) : ℝ≥0∞) / (Fintype.card M : ℝ≥0∞) +
        (δ + εks + δlink + δin) := by
  refine (P.audit_exfiltration_influenceSet hks hlink hin δ σ g).trans ?_
  have hcard := (P.card_influenceSet_le_harm (a := a) hD hb K hK hsep hw hH).trans
    (Nat.mul_le_mul_left _ (card_admissible_le hk))
  gcongr

end OneStage

section TwoStage

variable {nf nc : ℕ} {Pf : Partition C nf} {Pc : Partition C nc} {a : Fin C.N → Bool} {εks δlink δin : ℝ≥0∞}
  {D : Finset (Fin C.N)}

/-- **The influence of an accepted two-stage audit**, variant (b), over replay units: the committed coarse transcript
delivers on `D` a value outside the coarse partition's influence set, with the effective escape, with probability at
most `δ + ε_ks + δ_link + δ_in`, given the two-stage input hypothesis `AnchorsSound₂`. A replay unit's separator can be
its exits (`separates_exits`). -/
theorem Refines.twoStage_influence (r : Refines Pf Pc) {L₁ : Law nc} {L₂ : Fin nc → Law nf} {Reg : Type}
    {Int : Finset (Fin nc) → Type}
    {session : Finset (Fin nf) → Reg → (S₁ : Finset (Fin nc)) → Int S₁ → Game Bool}
    (X : (R : Reg) → Cont₁ L₁ L₂ Reg Int session R → (Fin C.N → Bool))
    (Yint : (R : Reg) → (ω₁ : L₁.Ω) → (I : Int (L₁.draw ω₁)) → Cont₂ L₁ L₂ Reg Int session R ω₁ I →
      (Fin C.N → Bool))
    (ext : (S : Finset (Fin nf)) → (R : Reg) → (S₁ : Finset (Fin nc)) → (I : Int S₁) →
      Strategy (session S R S₁ I) → Fin nf → Option (Fin C.N → Bool))
    (hks : (r.analysis₂ X Yint ext).KnowledgeSound εks) (hlink : (r.analysis₂ X Yint ext).LinkSound δlink)
    (hin : AnchorsSound₂ X a δin) (δ : ℝ≥0∞) (σ : Strategy (twoStage L₁ L₂ Reg Int session)) :
    prob (fun o => o.1 = true ∧ out D (X (reg₂ σ) (cont₁ σ)) ∉ Pc.influenceSet D (effEscape L₁ L₂ r.parent) a δ)
        (twoStage L₁ L₂ Reg Int session) σ ≤ δ + εks + δlink + δin := by
  let W := Pc.wrong (X (reg₂ σ) (cont₁ σ))
  let good : Bool × Finset (Fin nc) × Finset (Fin nf) → Prop := fun o =>
    o.1 = true ∧ effEscape L₁ L₂ r.parent W < δ
  let bad : Bool × Finset (Fin nc) × Finset (Fin nf) → Prop := fun o =>
    o.1 = true ∧ ¬ C.InputsAgree (X (reg₂ σ) (cont₁ σ)) a
  have hsplit : ∀ o : Bool × Finset (Fin nc) × Finset (Fin nf),
      (o.1 = true ∧ out D (X (reg₂ σ) (cont₁ σ)) ∉ Pc.influenceSet D (effEscape L₁ L₂ r.parent) a δ) →
      good o ∨ bad o := by
    intro o h
    by_cases hia : C.InputsAgree (X (reg₂ σ) (cont₁ σ)) a
    · exact Or.inl ⟨h.1, lt_of_not_ge fun hge => h.2 (Pc.out_mem_influenceSet hia hge)⟩
    · exact Or.inr ⟨h.1, hia⟩
  have hgood : prob good (twoStage L₁ L₂ Reg Int session) σ ≤ δ + εks + δlink := by
    refine (twoStage_profile (r.analysis₂ X Yint ext) hks hlink
      (fun B => effEscape L₁ L₂ r.parent B < δ) σ).trans ?_
    gcongr
    exact iSup₂_le fun B hB => hB.le
  calc _ ≤ prob (fun o => good o ∨ bad o) (twoStage L₁ L₂ Reg Int session) σ := prob_mono hsplit _ _
    _ ≤ prob good (twoStage L₁ L₂ Reg Int session) σ + prob bad (twoStage L₁ L₂ Reg Int session) σ :=
        prob_or_le good bad _ _
    _ ≤ δ + εks + δlink + δin := add_le_add hgood (hin σ)

end TwoStage

end Partition

end FlockSoundness.Audit
