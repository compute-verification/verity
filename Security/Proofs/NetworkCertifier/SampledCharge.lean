import Definitions.NetworkCertifier.Guarantees.SampledCharge
import Proofs.NetworkCertifier.Advice
import Mathlib.Tactic.Ring

/-!
# The sampled audit's charge (`Guarantees.SampledCharge`)

A gate's value under held gates reads only the gates it depends on (`valOver_congr_inp`). The layout fits the program
(`Layout.Fits`), so step `t`'s egress gates depend on no later step's ingress, and holding every committed gate of the
faulty units at the value it takes over the whole horizon leaves each step's egress rows as they were (`valOver_hold`,
`rows_holdAll`). So an accepted sequence is accepted with advice that holds each of those gates at one bit, and the count
is Theorem 3's with that advice (`encard_acceptedFaulty_le`): the inputs besides the ingress, one bit per committed gate
of the faulty units, and the jitter code, summed over the possible faulty sets (`encard_accSeqsSampled_le`). At
`δ ≥ miss(1)` the only possible faulty set is empty (`eq_empty_of_mem_faultySets`).
-/

namespace NetTiming

open Finset Claims.Sampling

variable {L K S : ℕ}

/-! ### A gate's value reads only what it depends on -/

theorem mapM_congr_option {ws : List ℕ} {rd rd' : ℕ → Option Bool} (h : ∀ i ∈ ws, rd i = rd' i) :
    ws.mapM rd = ws.mapM rd' := by
  induction ws with
  | nil => rfl
  | cons w ws ih =>
    simp only [List.mapM_cons]
    rw [h w List.mem_cons_self, ih fun i hi => h i (List.mem_cons_of_mem w hi)]

theorem bgate_val_congr (x : BGate) {rd rd' : ℕ → Option Bool} (own : Option Bool)
    (h : ∀ i ∈ x.reads, rd i = rd' i) : x.val rd own = x.val rd' own := by
  cases x with
  | input => rfl
  | const b => rfl
  | not i => simp only [BGate.val, h i (by simp [BGate.reads])]
  | and ws => simp only [BGate.val, mapM_congr_option (ws := ws) fun i hi => h i (by simpa [BGate.reads] using hi)]
  | xor ws => simp only [BGate.val, mapM_congr_option (ws := ws) fun i hi => h i (by simpa [BGate.reads] using hi)]

theorem depends_of_read {C : Prog} {g : ℕ} {x : BGate} (hx : C.gates[g]? = some x) {i : ℕ} (hi : i ∈ x.reads) :
    Depends C i g :=
  Relation.ReflTransGen.single (by simp [hx, hi])

theorem valOver_of_some (C : Prog) (inp : ℕ → Bool) {free : ℕ → Option Bool} {g : ℕ} {b : Bool}
    (hf : free g = some b) : valOver C inp free g = b := by
  rw [valOver]
  simp only [hf]

theorem valOver_of_none_none (C : Prog) (inp : ℕ → Bool) {free : ℕ → Option Bool} {g : ℕ}
    (hf : free g = none) (hx : C.gates[g]? = none) : valOver C inp free g = false := by
  rw [valOver]
  simp only [hf, hx]

theorem valOver_of_none_some (C : Prog) (inp : ℕ → Bool) {free : ℕ → Option Bool} {g : ℕ} {x : BGate}
    (hf : free g = none) (hx : C.gates[g]? = some x) :
    valOver C inp free g =
      (x.val (fun i => if _h : i < g then some (valOver C inp free i) else some false) (some (inp g))).getD false := by
  rw [valOver]
  simp only [hf, hx]

/-- Inputs that agree on every gate `g` depends on give `g` one value. -/
theorem valOver_congr_inp (C : Prog) {inp inp' : ℕ → Bool} (free : ℕ → Option Bool) (g : ℕ)
    (h : ∀ i, Depends C i g → inp i = inp' i) : valOver C inp free g = valOver C inp' free g := by
  induction g using Nat.strong_induction_on with
  | _ g ih =>
    rcases hf : free g with _ | b
    · rcases hx : C.gates[g]? with _ | x
      · rw [valOver_of_none_none C inp hf hx, valOver_of_none_none C inp' hf hx]
      · rw [valOver_of_none_some C inp hf hx, valOver_of_none_some C inp' hf hx, h g Relation.ReflTransGen.refl]
        congr 1
        apply bgate_val_congr x
        intro i hi
        split_ifs with hig
        · rw [ih i hig fun j hj => h j (hj.trans (depends_of_read hx hi))]
        · rfl
    · rw [valOver_of_some C inp hf, valOver_of_some C inp' hf]

/-! ### Causality: a step's egress reads no later ingress -/

theorem findSome?_range_eq {β : Type} (f : ℕ → Option β) (k : ℕ) :
    ∀ m, k ≤ m → (∀ j, k ≤ j → j < m → f j = none) →
      (List.range m).findSome? f = (List.range k).findSome? f := by
  intro m
  induction m with
  | zero => intro hk _; rw [Nat.le_zero.mp hk]
  | succ m ih =>
    intro hk h
    rcases Nat.lt_or_ge m k with hmk | hmk
    · rw [show k = m + 1 by omega]
    · rw [List.range_succ, List.findSome?_append, ih hmk fun j h1 h2 => h j h1 (by omega)]
      simp [h m hmk (by omega)]

theorem ingressUpTo_getD {In : Type} (op : ℕ → List (Obs L K S) → In) (o : List (Obs L K S)) (dflt : In)
    {m j : ℕ} (h : j ≤ m) : (ingressUpTo op o m).getD j dflt = op j (o.take j) := by
  simp [ingressUpTo, List.getD_eq_getElem?_getD, show j < m + 1 by omega]

theorem length_ingressUpTo {In : Type} (op : ℕ → List (Obs L K S) → In) (o : List (Obs L K S)) (m : ℕ) :
    (ingressUpTo op o m).length = m + 1 := by
  simp [ingressUpTo]

/-- A gate that an egress gate of step `t` depends on reads the same input whether the ingress runs to `t` or further:
by causality it is no ingress gate of a later step. -/
theorem inputOf_eq_of_causal {lay : Layout L S} {C : Prog} {n t : ℕ} (hfits : lay.Fits C n) (ht : t < n)
    {ℓ : Fin L} {σ : Fin S} {e : ℕ} (he : e ∈ slotGates (lay.egress t ℓ σ)) {ins ins' : List (List Bool)}
    (hlen : ins.length = t + 1) (hlen' : t + 1 ≤ ins'.length) (hlen'' : ins'.length ≤ n)
    (hpre : ∀ j ≤ t, ins.getD j [] = ins'.getD j []) (a : ℕ → Bool) {i : ℕ} (hi : Depends C i e) :
    lay.inputOf n ins a i = lay.inputOf n ins' a i := by
  unfold Layout.inputOf
  have hnone : ∀ j, t + 1 ≤ j → j < ins'.length →
      ((lay.ingress j).findIdx? (· == i)).map (j, ·) = none := by
    intro j h1 h2
    have hm : i ∉ lay.ingress j := fun hm => hfits.2.2 t ht j (by omega) (by omega) ℓ σ e he i hm hi
    rw [Option.map_eq_none_iff, List.findIdx?_eq_none_iff]
    intro x hx
    simp only [beq_eq_false_iff_ne, ne_eq]
    rintro rfl
    exact hm hx
  rw [hlen, findSome?_range_eq _ (t + 1) ins'.length hlen' hnone]
  rcases hR : (List.range (t + 1)).findSome? (fun j => ((lay.ingress j).findIdx? (· == i)).map (j, ·)) with
    _ | ⟨j, idx⟩
  · rfl
  · obtain ⟨j', hj', hfj⟩ := List.exists_of_findSome?_eq_some hR
    obtain ⟨idx', -, hp⟩ := Option.map_eq_some_iff.mp hfj
    simp only [Prod.mk.injEq] at hp
    obtain ⟨rfl, rfl⟩ := hp
    simp only
    rw [hpre j' (by rw [List.mem_range] at hj'; omega)]

/-- Holding every gate of `G` at its value on the inputs `inpF` changes no gate that `e` depends on, when `inpF` agrees
with the inputs on every gate `e` depends on. -/
theorem valOver_hold (C : Prog) (inp inpF : ℕ → Bool) (free : ℕ → Option Bool) (G : Finset ℕ)
    (hG : ∀ g b, free g = some b → g ∈ G) (e : ℕ) (hagree : ∀ i, Depends C i e → inp i = inpF i) (g : ℕ)
    (hg : Depends C g e) :
    valOver C inp free g = valOver C inp (fun g => if g ∈ G then some (valOver C inpF free g) else none) g := by
  set free' : ℕ → Option Bool := fun g => if g ∈ G then some (valOver C inpF free g) else none with hfree'
  induction g using Nat.strong_induction_on with
  | _ g ih =>
    by_cases hgG : g ∈ G
    · have hf' : free' g = some (valOver C inpF free g) := by simp only [hfree', hgG, ↓reduceIte]
      rw [valOver_of_some C inp hf']
      exact valOver_congr_inp C free g fun i hi => hagree i (hi.trans hg)
    · have hf : free g = none := by
        rcases hfg : free g with _ | b
        · rfl
        · exact absurd (hG g b hfg) hgG
      have hf' : free' g = none := by simp only [hfree', hgG, ↓reduceIte]
      rcases hx : C.gates[g]? with _ | x
      · rw [valOver_of_none_none C inp hf hx, valOver_of_none_none C inp hf' hx]
      · rw [valOver_of_none_some C inp hf hx, valOver_of_none_some C inp hf' hx]
        congr 1
        apply bgate_val_congr x
        intro i hi
        split_ifs with hig
        · rw [ih i hig (Relation.ReflTransGen.trans (depends_of_read hx hi) hg)]
        · rfl

theorem rowOf_congr {val val' : ℕ → Bool} {p : Fin S → ℕ × List ℕ}
    (h : ∀ σ, ∀ g ∈ slotGates (p σ), val g = val' g) : rowOf val p = rowOf val' p := by
  funext σ
  simp only [rowOf]
  rw [h σ (p σ).1 (by simp [slotGates]),
    List.map_congr_left fun g hg => h σ g (by simp [slotGates, hg])]

/-- **The advice that holds each gate of `G` at one bit**: its value over the whole horizon, on the ingress `o`'s
operator moves give. -/
noncomputable def holdAll (C : Prog) (lay : Layout L S) (n : ℕ) (op : ℕ → List (Obs L K S) → List Bool)
    (o : List (Obs L K S)) (a : ℕ → Bool) (free : ℕ → Option Bool) (G : Finset ℕ) : ℕ → Option Bool :=
  fun g => if g ∈ G then some (valOver C (lay.inputOf n (ingressUpTo op o (n - 1)) a) free g) else none

theorem rows_holdAll (C : Prog) (lay : Layout L S) (n : ℕ) (hfits : lay.Fits C n)
    (op : ℕ → List (Obs L K S) → List Bool) (o : List (Obs L K S)) (p : (ℕ → Bool) × (ℕ → Option Bool))
    (G : Finset ℕ) (hG : ∀ g b, p.2 g = some b → g ∈ G) {t : ℕ} (ht : t < n) :
    (circuitProgram C lay n).rows t (ingressUpTo op o t) p =
      (circuitProgram C lay n).rows t (ingressUpTo op o t) (p.1, holdAll C lay n op o p.1 p.2 G) := by
  funext ℓ
  simp only [circuitProgram]
  apply rowOf_congr
  intro σ e he
  apply valOver_hold C _ _ p.2 G hG e _ e Relation.ReflTransGen.refl
  intro i hi
  apply inputOf_eq_of_causal hfits ht he (length_ingressUpTo op o t)
    (by rw [length_ingressUpTo]; omega) (by rw [length_ingressUpTo]; omega) _ p.1 hi
  intro j hj
  rw [ingressUpTo_getD op o [] hj, ingressUpTo_getD op o [] (by omega)]

/-! ### The count -/

/-- The committed gates of the units at the slots `F`. -/
def faultyGateSet (C : Prog) (P : Partition) (F : Finset ℕ) : Finset ℕ :=
  F.biUnion fun v => ((P.blocks.getD v []).filter fun g => C.Boundary (P.blocks.getD v []) g).toFinset

theorem mem_faultyGateSet {C : Prog} {P : Partition} {F : Finset ℕ} {g : ℕ} (h : FaultyGate C P F g) :
    g ∈ faultyGateSet C P F := by
  obtain ⟨v, hv, r, hr, hg, hb⟩ := h
  have hr' : P.blocks.getD v [] = r := by rw [List.getD_eq_getElem?_getD, hr, Option.getD_some]
  simp only [faultyGateSet, Finset.mem_biUnion, List.mem_toFinset, List.mem_filter]
  exact ⟨v, hv, by rw [hr']; exact ⟨hg, hb⟩⟩

theorem card_faultyGateSet_le (C : Prog) (P : Partition) (F : Finset ℕ) :
    (faultyGateSet C P F).card ≤ faultyBits C P F :=
  Finset.card_biUnion_le.trans (Finset.sum_le_sum fun _ _ => List.toFinset_card_le _)

theorem encard_biUnion_le {α ι : Type} [DecidableEq ι] (s : Finset ι) (f : ι → Set α) :
    (⋃ i ∈ s, f i).encard ≤ ∑ i ∈ s, (f i).encard := by
  induction s using Finset.induction_on with
  | empty => simp
  | insert i s hi ih =>
    rw [Finset.set_biUnion_insert, Finset.sum_insert hi]
    exact (Set.encard_union_le _ _).trans (add_le_add le_rfl ih)

/-- **One faulty set's count.** The sequences accepted with the faulty set `F` number at most
`#As · 2^(faultyBits F) · D^n`: each maps injectively to its inputs besides the ingress, the bit at which `holdAll` holds
each committed gate of `F`'s units, and its jitter code. -/
theorem encard_acceptedFaulty_le (C : Prog) (P : Partition) (lay : Layout L S) (n : ℕ) (hfits : lay.Fits C n)
    (As : Set (ℕ → Bool)) (F : Finset ℕ) (op : ℕ → List (Obs L K S) → List Bool) (D : ℕ) :
    {o : List (Obs L K S) | o.length = n ∧ (∀ x ∈ o, x.jitter < D) ∧ AcceptedFaulty C P lay n As F op o}.encard ≤
      As.encard * ((2 ^ faultyBits C P F * D ^ n : ℕ) : ℕ∞) := by
  classical
  set Sq := {o : List (Obs L K S) | o.length = n ∧ (∀ x ∈ o, x.jitter < D) ∧ AcceptedFaulty C P lay n As F op o}
    with hSq
  set G : Finset ℕ := faultyGateSet C P F with hGdef
  let adv : List (Obs L K S) → (ℕ → Bool) × (ℕ → Option Bool) := fun o =>
    if h : AcceptedFaulty C P lay n As F op o then (h.choose, h.choose_spec.2.choose) else (fun _ => false, fun _ => none)
  have adv_spec : ∀ o ∈ Sq, (adv o).1 ∈ As ∧ (∀ g b, (adv o).2 g = some b → g ∈ G) ∧
      Accepted (circuitProgram C lay n) op n o (adv o) := by
    intro o ho
    have hacc : AcceptedFaulty C P lay n As F op o := ho.2.2
    simp only [adv, hacc, ↓reduceDIte]
    refine ⟨hacc.choose_spec.1, fun g b hgb => mem_faultyGateSet (hacc.choose_spec.2.choose_spec.1 g b hgb),
      hacc.choose_spec.2.choose_spec.2⟩
  let code : List (Obs L K S) → (ℕ → Bool) × (↥G → Bool) × (Fin n → ℕ) := fun o =>
    ((adv o).1, (fun g => valOver C (lay.inputOf n (ingressUpTo op o (n - 1)) (adv o).1) (adv o).2 g), jitterCode o)
  have hinj : Set.InjOn code Sq := by
    intro o ho o' ho' h
    simp only [code, Prod.mk.injEq] at h
    obtain ⟨ha, hv, hj⟩ := h
    obtain ⟨-, hG1, acc⟩ := adv_spec o ho
    obtain ⟨-, hG2, acc'⟩ := adv_spec o' ho'
    have hhold : holdAll C lay n op o (adv o).1 (adv o).2 G = holdAll C lay n op o' (adv o').1 (adv o').2 G := by
      funext g
      simp only [holdAll]
      split_ifs with hg
      · exact congrArg some (congrFun hv ⟨g, hg⟩)
      · rfl
    apply eq_of_stepwise ho.1 ho'.1
    intro t ht hpre
    obtain ⟨h1, hr, hs⟩ := acc t ht
    obtain ⟨h2, hr', hs'⟩ := acc' t ht
    apply Obs.ext'
    · calc o[t].rows = (circuitProgram C lay n).rows t (ingressUpTo op o t) (adv o) := hr
        _ = (circuitProgram C lay n).rows t (ingressUpTo op o t) ((adv o).1, holdAll C lay n op o (adv o).1 (adv o).2 G) :=
            rows_holdAll C lay n hfits op o (adv o) G hG1 ht
        _ = (circuitProgram C lay n).rows t (ingressUpTo op o' t)
              ((adv o').1, holdAll C lay n op o' (adv o').1 (adv o').2 G) := by
            rw [ingressUpTo_congr op hpre, hhold, ha]
        _ = (circuitProgram C lay n).rows t (ingressUpTo op o' t) (adv o') :=
            (rows_holdAll C lay n hfits op o' (adv o') G hG2 ht).symm
        _ = o'[t].rows := hr'.symm
    · exact jitter_eq_of_jitterCode hj ht h1 h2
    · funext κ; rw [hs κ, hs' κ]
  let T : Finset ((↥G → Bool) × (Fin n → ℕ)) := (univ : Finset (↥G → Bool)) ×ˢ Fintype.piFinset fun _ : Fin n => range D
  have hsub : code '' Sq ⊆ As ×ˢ (T : Set ((↥G → Bool) × (Fin n → ℕ))) := by
    rintro _ ⟨o, ho, rfl⟩
    refine ⟨(adv_spec o ho).1, ?_⟩
    simp only [T, Finset.coe_product, Finset.coe_univ, Set.mem_prod, Set.mem_univ, true_and, Finset.mem_coe]
    exact jitterCode_mem o ho.1 ho.2.1
  calc Sq.encard = (code '' Sq).encard := hinj.encard_image.symm
    _ ≤ (As ×ˢ (T : Set ((↥G → Bool) × (Fin n → ℕ)))).encard := Set.encard_le_encard hsub
    _ = As.encard * ((2 ^ G.card * D ^ n : ℕ) : ℕ∞) := by
        rw [Set.encard_prod, Set.encard_coe_eq_coe_finsetCard]
        congr 2
        simp [T, Fintype.card_piFinset]
    _ ≤ As.encard * ((2 ^ faultyBits C P F * D ^ n : ℕ) : ℕ∞) := by
        have h := Nat.mul_le_mul_right (D ^ n) (Nat.pow_le_pow_right (n := 2) (by norm_num) (card_faultyGateSet_le C P F))
        exact mul_le_mul' le_rfl (by exact_mod_cast h)

/-- At `δ ≥ miss(1)` the only possible faulty set is empty: a non-empty set of slots is accepted with probability at
most `miss(1)`, so it is not in `M_δ`. -/
theorem eq_empty_of_mem_faultySets (d : Design) (P : Partition) {δ : ℝ} (hδ : d.miss P 1 ≤ δ) {F : Finset ℕ}
    (hF : F ∈ faultySets d P δ) : F = ∅ := by
  classical
  simp only [faultySets, Finset.mem_filter, Finset.mem_powerset] at hF
  obtain ⟨hsub, hM, -⟩ := hF
  by_contra hne
  have hcard : 1 ≤ F.card := Finset.card_pos.mpr (Finset.nonempty_iff_ne_empty.mpr hne)
  have : Finite {B : Finset ℕ // B ⊆ Finset.range P.blocks.length ∧ 1 ≤ B.card} :=
    Finite.of_injective (β := ↥(Finset.range P.blocks.length).powerset)
      (fun B => ⟨B.1, Finset.mem_powerset.mpr B.2.1⟩) fun B B' h => Subtype.ext (by simpa using h)
  have hle : d.Pa P F ≤ d.miss P 1 :=
    le_ciSup (f := fun B : {B : Finset ℕ // B ⊆ Finset.range P.blocks.length ∧ 1 ≤ B.card} => d.Pa P B.1)
      (Set.finite_range _).bddAbove ⟨F, hsub, hcard⟩
  exact absurd hM (not_lt.mpr (hle.trans hδ))

theorem encard_accSeqsSampled_le (C : Prog) (P : Partition) (d : Design) (lay : Layout L S) (n : ℕ)
    (hfits : lay.Fits C n) (δ : ℝ) (As : Set (ℕ → Bool)) (c : ℕ) (hAs : As.encard ≤ ((2 ^ c : ℕ) : ℕ∞)) (β : ℕ)
    (hβ : ∀ F ∈ faultySets d P δ, faultyBits C P F ≤ β) (op : ℕ → List (Obs L K S) → List Bool) (D : ℕ) :
    (AccSeqsSampled C P d lay n As δ op D).encard ≤
      ((2 ^ c * (faultySets d P δ).card * 2 ^ β * D ^ n : ℕ) : ℕ∞) := by
  classical
  let Sq : Finset ℕ → Set (List (Obs L K S)) := fun F =>
    {o | o.length = n ∧ (∀ x ∈ o, x.jitter < D) ∧ AcceptedFaulty C P lay n As F op o}
  have hsub : AccSeqsSampled C P d lay n As δ op D ⊆ ⋃ F ∈ faultySets d P δ, Sq F := by
    rintro o ⟨hl, hD, F, hF, hacc⟩
    exact Set.mem_biUnion hF ⟨hl, hD, hacc⟩
  calc _ ≤ (⋃ F ∈ faultySets d P δ, Sq F).encard := Set.encard_le_encard hsub
    _ ≤ ∑ F ∈ faultySets d P δ, (Sq F).encard := encard_biUnion_le _ _
    _ ≤ ∑ _F ∈ faultySets d P δ, ((2 ^ c * (2 ^ β * D ^ n) : ℕ) : ℕ∞) := by
        apply Finset.sum_le_sum
        intro F hF
        calc (Sq F).encard ≤ As.encard * ((2 ^ faultyBits C P F * D ^ n : ℕ) : ℕ∞) :=
              encard_acceptedFaulty_le C P lay n hfits As F op D
          _ ≤ ((2 ^ c : ℕ) : ℕ∞) * ((2 ^ β * D ^ n : ℕ) : ℕ∞) := by
              have h := Nat.mul_le_mul_right (D ^ n) (Nat.pow_le_pow_right (n := 2) (by norm_num) (hβ F hF))
              exact mul_le_mul' hAs (by exact_mod_cast h)
          _ = ((2 ^ c * (2 ^ β * D ^ n) : ℕ) : ℕ∞) := by push_cast; rfl
    _ = _ := by
        rw [Finset.sum_const, nsmul_eq_mul]
        norm_cast
        ring

end NetTiming
