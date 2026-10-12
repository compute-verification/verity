import Proofs.NetworkCertifier.Advice

/-!
# The ingress timing bound

`docs/network-transparency/timing-channel.md` §2.5 ("Ingress content") and §3.2. The interaction is read inward
(`NetTiming.Protocol.Ingress`): the outside holds the message and reacts to every record, and a decoder inside X reads
the wire. On runs whose statuses are complete and whose frames follow the prescribed content, a record sequence is
determined by its occupancy counts and its jitter classes.
-/

namespace NetTiming

open Finset

variable {L K S : ℕ} {Msg In Act Wire : Type}

/-! ### A left-packed row is its frames -/

theorem getElem_eq_getElem?_filterMap_id {F : Type} : ∀ (l : List (Option F)),
    (∀ i j (hj : j < l.length) (_ : i ≤ j), l[j].isSome → (l[i]'(by omega)).isSome) →
    ∀ i (hi : i < l.length), l[i] = (l.filterMap id)[i]?
  | [], _, i, hi => absurd hi (by simp)
  | some x :: l, _, 0, _ => by simp
  | some x :: l, h, i + 1, hi => by
      simp only [List.getElem_cons_succ, List.filterMap_cons, id, List.getElem?_cons_succ]
      exact getElem_eq_getElem?_filterMap_id l
        (fun i j hj hij hs => h (i + 1) (j + 1) (by simpa using hj) (by omega) hs) i (by simpa using hi)
  | none :: l, h, i, hi => by
      have hnone : ∀ j (hj : j < (none :: l).length), (none :: l)[j] = none := by
        intro j hj
        by_contra hne
        have := h 0 j hj (Nat.zero_le _) (Option.isSome_iff_ne_none.mpr hne)
        simp at this
      have hnil : (none :: l).filterMap id = [] := by
        rw [List.filterMap_eq_nil_iff]; intro a ha
        obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem ha
        exact hnone j hj
      rw [hnil, hnone i hi]; rfl

/-- Under W4 a row is determined by its frames: slot `s` holds the `s`-th of them. -/
theorem row_eq_getElem?_rowFrames {F : Type} (R : Row S F) (h : RowLeftPacked fun s => (R s).isSome) (s : Fin S) :
    R s = (rowFrames R)[(s : ℕ)]? := by
  have hl : (s : ℕ) < (List.ofFn R).length := by simp
  have := getElem_eq_getElem?_filterMap_id (List.ofFn R) (fun i j hj hij hs => by
      simp only [List.getElem_ofFn] at hs ⊢
      exact h ⟨i, by simp at hj; omega⟩ ⟨j, by simpa using hj⟩ (Fin.mk_le_mk.mpr hij) hs) s hl
  simpa [rowFrames] using this

theorem length_rowFrames_congr {F : Type} {R R' : Row S F}
    (h : (fun s => (R s).isSome) = fun s => (R' s).isSome) : (rowFrames R).length = (rowFrames R').length := by
  simp only [rowFrames, List.length_filterMap_eq_countP, id, List.ofFn_eq_map, List.countP_map]
  show List.countP (fun s => (R s).isSome) _ = List.countP (fun s => (R' s).isSome) _
  rw [h]

theorem occ_eq_of_rowCount {F : Type} {R R' : Row S F} (h : RowLeftPacked fun s => (R s).isSome)
    (h' : RowLeftPacked fun s => (R' s).isSome) (hc : (rowCount R : ℕ) = rowCount R') :
    (fun s => (R s).isSome) = fun s => (R' s).isSome :=
  (eq_prefixRow_of_rowLeftPacked _ h).trans ((congrArg (prefixRow S) hc).trans (eq_prefixRow_of_rowLeftPacked _ h').symm)

/-! ### Frames over a prefix -/

theorem linkFrames_append (o o' : List (Obs L K S)) (ℓ : Fin L) :
    linkFrames (o ++ o') ℓ = linkFrames o ℓ ++ linkFrames o' ℓ := by
  simp only [linkFrames, List.flatMap_append]

theorem linkFrames_take_succ (o : List (Obs L K S)) (ℓ : Fin L) {t : ℕ} (ht : t < o.length) :
    linkFrames (o.take (t + 1)) ℓ = linkFrames (o.take t) ℓ ++ rowFrames (o[t].rows ℓ) := by
  rw [← List.take_concat_get' o t ht, linkFrames_append]
  simp [linkFrames]

theorem linkFrames_take_prefix (o : List (Obs L K S)) (ℓ : Fin L) (k : ℕ) :
    linkFrames (o.take k) ℓ <+: linkFrames o ℓ := by
  conv_rhs => rw [← List.take_append_drop k o]
  rw [linkFrames_append]; exact List.prefix_append _ _

theorem eq_of_append_prefix {α : Type} {p F F' c : List α} (h : p ++ F <+: c) (h' : p ++ F' <+: c)
    (hl : F.length = F'.length) : F = F' :=
  List.append_cancel_left
    ((List.prefix_of_prefix_length_le h h' (by simp [hl])).eq_of_length (by simp [hl]))

/-! ### The code: occupancy counts and jitter classes -/

/-- The count code of a sequence: each step's real-frame count on each link, `0` past its end. -/
def countCode {n : ℕ} (o : List (Obs L K S)) : Fin n → Fin L → ℕ :=
  fun t ℓ => (o[(t : ℕ)]?.map fun x => (rowCount (x.rows ℓ) : ℕ)).getD 0

theorem rowCount_eq_of_countCode {n : ℕ} {o o' : List (Obs L K S)} (h : countCode (n := n) o = countCode o')
    {t : ℕ} (ht : t < n) (h1 : t < o.length) (h2 : t < o'.length) (ℓ : Fin L) :
    (rowCount (o[t].rows ℓ) : ℕ) = rowCount (o'[t].rows ℓ) := by
  have := congrFun (congrFun h ⟨t, ht⟩) ℓ
  simpa [countCode, List.getElem?_eq_getElem h1, List.getElem?_eq_getElem h2] using this

/-- Two left-packed sequences of length `n` with complete statuses, whose frames on each link are prefixes of the same
content, are equal once their counts and jitter classes are. -/
theorem eq_of_ingressCode {n : ℕ} (c : Fin L → List (List Bool)) {o o' : List (Obs L K S)}
    (hl : o.length = n) (hl' : o'.length = n)
    (hp : ∀ x ∈ o, ∀ ℓ, RowLeftPacked fun s => (x.rows ℓ s).isSome)
    (hp' : ∀ x ∈ o', ∀ ℓ, RowLeftPacked fun s => (x.rows ℓ s).isSome)
    (hs : StatusComplete n o) (hs' : StatusComplete n o')
    (hc : ∀ ℓ, linkFrames o ℓ <+: c ℓ) (hc' : ∀ ℓ, linkFrames o' ℓ <+: c ℓ)
    (hk : countCode (n := n) o = countCode o') (hj : jitterCode (n := n) o = jitterCode o') : o = o' := by
  apply eq_of_stepwise hl hl'
  intro t ht hpre
  have h1 : t < o.length := by omega
  have h2 : t < o'.length := by omega
  apply Obs.ext'
  · funext ℓ
    have hP := hp _ (List.getElem_mem h1) ℓ
    have hP' := hp' _ (List.getElem_mem h2) ℓ
    have hocc := occ_eq_of_rowCount hP hP' (rowCount_eq_of_countCode hk ht h1 h2 ℓ)
    have hF : rowFrames (o[t].rows ℓ) = rowFrames (o'[t].rows ℓ) := by
      apply eq_of_append_prefix (p := linkFrames (o.take t) ℓ) (c := c ℓ)
      · rw [← linkFrames_take_succ o ℓ h1]; exact (linkFrames_take_prefix o ℓ _).trans (hc ℓ)
      · rw [hpre, ← linkFrames_take_succ o' ℓ h2]; exact (linkFrames_take_prefix o' ℓ _).trans (hc' ℓ)
      · exact length_rowFrames_congr hocc
    funext s
    rw [row_eq_getElem?_rowFrames _ hP s, row_eq_getElem?_rowFrames _ hP' s, hF]
  · exact jitter_eq_of_jitterCode hj ht h1 h2
  · obtain ⟨_, hs1⟩ := hs t ht
    obtain ⟨_, hs2⟩ := hs' t ht
    funext κ; rw [hs1 κ, hs2 κ]

/-! ### The bounds -/

/-- **Content plus timing, from the records.** When each message's frames follow some content assignment in `Cs`, a
decoder inside X that reads the records tells apart at most `#Cs · (∏_ℓ (r ℓ + 1))^n · D^n` messages. -/
theorem encard_ingress_obsDecodable_le_contents (I : Interaction L K S Msg In Act Wire) (n D : ℕ) (r : Fin L → ℕ)
    (hW4 : Packed I.certifier) (hr : Allocated I.certifier r) (hJ : JitterBounded I.certifier D)
    (Cs : Set (Fin L → List (List Bool))) (Ms : Set Msg)
    (hacc : ∀ m ∈ Ms, StatusComplete n (I.obsRun m n))
    (hc : ∀ m ∈ Ms, ∃ c ∈ Cs, ∀ ℓ, linkFrames (I.obsRun m n) ℓ <+: c ℓ)
    (dec : List (Obs L K S) → Msg) (hdec : ∀ m ∈ Ms, dec (I.obsRun m n) = m) :
    Ms.encard ≤ Cs.encard * (((∏ ℓ, (r ℓ + 1)) ^ n * D ^ n : ℕ) : ℕ∞) := by
  classical
  have hpk : ∀ m, ∀ x ∈ I.obsRun m n, ∀ ℓ, RowLeftPacked fun s => (x.rows ℓ s).isSome := by
    intro m x hx ℓ
    obtain ⟨k, xs, as, rfl⟩ := I.mem_obsRun m n x hx
    exact hW4 k xs as ℓ
  let T : Finset ((Fin n → Fin L → ℕ) × (Fin n → ℕ)) :=
    (Fintype.piFinset fun _ : Fin n => Fintype.piFinset fun ℓ => range (r ℓ + 1)) ×ˢ
      Fintype.piFinset fun _ : Fin n => range D
  have hT : ∀ m, (countCode (I.obsRun m n), jitterCode (I.obsRun m n)) ∈ T := by
    intro m
    refine mem_product.mpr ⟨?_, jitterCode_mem _ (I.length_obsRun m n) ?_⟩
    · show countCode (I.obsRun m n) ∈ _
      rw [Fintype.mem_piFinset]; intro t; rw [Fintype.mem_piFinset]; intro ℓ
      have ht : (t : ℕ) < (I.obsRun m n).length := by rw [I.length_obsRun]; exact t.2
      simp only [countCode, List.getElem?_eq_getElem ht, Option.map_some, Option.getD_some, mem_range]
      obtain ⟨k, xs, as, hx⟩ := I.mem_obsRun m n _ (List.getElem_mem ht)
      rw [hx]; exact Nat.lt_succ_of_le (hr k xs as ℓ)
    · intro x hx
      obtain ⟨k, xs, as, rfl⟩ := I.mem_obsRun m n x hx
      exact hJ k xs as
  have hcard : T.card = (∏ ℓ, (r ℓ + 1)) ^ n * D ^ n := by
    rw [card_product, Fintype.card_piFinset, Fintype.card_piFinset]
    simp only [Fintype.card_piFinset, card_range, prod_const, card_univ, Fintype.card_fin]
  let cOf : Msg → Fin L → List (List Bool) := fun m => if h : m ∈ Ms then (hc m h).choose else fun _ => []
  have hcOf : ∀ m ∈ Ms, cOf m ∈ Cs ∧ ∀ ℓ, linkFrames (I.obsRun m n) ℓ <+: cOf m ℓ := by
    intro m hm; simp only [cOf, hm, ↓reduceDIte]; exact (hc m hm).choose_spec
  let code : Msg → (Fin L → List (List Bool)) × (Fin n → Fin L → ℕ) × (Fin n → ℕ) := fun m =>
    (cOf m, countCode (I.obsRun m n), jitterCode (I.obsRun m n))
  have hinj : Set.InjOn code Ms := by
    intro m hm m' hm' h
    simp only [code, Prod.mk.injEq] at h
    obtain ⟨hcm, hk, hj⟩ := h
    have h1 := (hcOf m hm).2
    rw [hcm] at h1
    have ho := eq_of_ingressCode (cOf m') (I.length_obsRun m n) (I.length_obsRun m' n) (hpk m) (hpk m')
      (hacc m hm) (hacc m' hm') h1 (hcOf m' hm').2 hk hj
    rw [← hdec m hm, ← hdec m' hm', ho]
  calc Ms.encard = (code '' Ms).encard := hinj.encard_image.symm
    _ ≤ (Cs ×ˢ (T : Set ((Fin n → Fin L → ℕ) × (Fin n → ℕ)))).encard := by
        apply Set.encard_le_encard
        rintro _ ⟨m, hm, rfl⟩
        exact Set.mk_mem_prod (hcOf m hm).1 (Finset.mem_coe.mpr (hT m))
    _ = Cs.encard * (((∏ ℓ, (r ℓ + 1)) ^ n * D ^ n : ℕ) : ℕ∞) := by
        rw [Set.encard_prod, Set.encard_coe_eq_coe_finsetCard, hcard]

/-- **The ingress timing bound, from the records.** On runs whose statuses are complete up to `n` and whose frames on
each link are a prefix of the prescribed content, a decoder inside X that reads the records tells apart at most
`(∏_ℓ (r ℓ + 1))^n · D^n` messages. -/
theorem encard_ingress_obsDecodable_le (I : Interaction L K S Msg In Act Wire) (n D : ℕ) (r : Fin L → ℕ)
    (hW4 : Packed I.certifier) (hr : Allocated I.certifier r) (hJ : JitterBounded I.certifier D)
    (c : Fin L → List (List Bool)) (Ms : Set Msg)
    (hacc : ∀ m ∈ Ms, StatusComplete n (I.obsRun m n))
    (hc : ∀ m ∈ Ms, ∀ ℓ, linkFrames (I.obsRun m n) ℓ <+: c ℓ)
    (dec : List (Obs L K S) → Msg) (hdec : ∀ m ∈ Ms, dec (I.obsRun m n) = m) :
    Ms.encard ≤ (((∏ ℓ, (r ℓ + 1)) ^ n * D ^ n : ℕ) : ℕ∞) := by
  have := encard_ingress_obsDecodable_le_contents I n D r hW4 hr hJ {c} Ms hacc
    (fun m hm => ⟨c, rfl, hc m hm⟩) dec hdec
  rwa [Set.encard_singleton, one_mul] at this

/-- **The ingress timing bound.** On runs whose statuses are complete up to `n` and whose frames on each link are a
prefix of the prescribed content, a decoder inside X that reads the wire tells apart at most
`(∏_ℓ (r ℓ + 1))^n · D^n` messages. -/
theorem encard_ingress_decodable_le (I : Interaction L K S Msg In Act Wire) (n D : ℕ) (r : Fin L → ℕ)
    (view : ℕ → Obs L K S → Wire) (hW2 : Regenerates I.certifier view) (hW4 : Packed I.certifier)
    (hr : Allocated I.certifier r) (hJ : JitterBounded I.certifier D)
    (c : Fin L → List (List Bool)) (Ms : Set Msg)
    (hacc : ∀ m ∈ Ms, StatusComplete n (I.obsRun m n))
    (hc : ∀ m ∈ Ms, ∀ ℓ, linkFrames (I.obsRun m n) ℓ <+: c ℓ)
    (dec : List Wire → Msg) (hdec : ∀ m ∈ Ms, dec (I.wireRun m n) = m) :
    Ms.encard ≤ (((∏ ℓ, (r ℓ + 1)) ^ n * D ^ n : ℕ) : ℕ∞) :=
  encard_ingress_obsDecodable_le I n D r hW4 hr hJ c Ms hacc hc (fun o => dec (o.mapIdx view))
    (fun m hm => by rw [← I.wireRun_eq view hW2]; exact hdec m hm)

/-- **Content plus timing.** When each message's frames follow some content assignment in `Cs`, a decoder inside X
that reads the wire tells apart at most `#Cs · (∏_ℓ (r ℓ + 1))^n · D^n` messages. -/
theorem encard_ingress_decodable_le_contents (I : Interaction L K S Msg In Act Wire) (n D : ℕ) (r : Fin L → ℕ)
    (view : ℕ → Obs L K S → Wire) (hW2 : Regenerates I.certifier view) (hW4 : Packed I.certifier)
    (hr : Allocated I.certifier r) (hJ : JitterBounded I.certifier D)
    (Cs : Set (Fin L → List (List Bool))) (Ms : Set Msg)
    (hacc : ∀ m ∈ Ms, StatusComplete n (I.obsRun m n))
    (hc : ∀ m ∈ Ms, ∃ c ∈ Cs, ∀ ℓ, linkFrames (I.obsRun m n) ℓ <+: c ℓ)
    (dec : List Wire → Msg) (hdec : ∀ m ∈ Ms, dec (I.wireRun m n) = m) :
    Ms.encard ≤ Cs.encard * (((∏ ℓ, (r ℓ + 1)) ^ n * D ^ n : ℕ) : ℕ∞) :=
  encard_ingress_obsDecodable_le_contents I n D r hW4 hr hJ Cs Ms hacc hc (fun o => dec (o.mapIdx view))
    (fun m hm => by rw [← I.wireRun_eq view hW2]; exact hdec m hm)

end NetTiming
