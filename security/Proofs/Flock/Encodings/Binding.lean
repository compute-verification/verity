module

public import Definitions.Claims.Encodings
public import Proofs.Flock.Claims.Sampling.Cost

/-!
# The registered encodings bind: the sum by position, the encoding, the link

* `byPosBinds`: two openings at one position are checked by one part, the encoding's where the roots encode the
  position and the record's elsewhere, so the part's binding is the sum's.
* `codeBinds`: two openings at one position of different values parse to messages whose bits there differ, so by the
  distance their codewords differ at `d` or more columns. Each opening's columns agree with its message's codeword at
  `agree` or more distinct positions below `len`, so at most `len − agree` positions miss each, and since
  `2·(len − agree) < d`, some position where the codewords differ is opened, and agrees, in both (`exists_common`).
  There the two opened columns differ and both pass their checks, which collide (`ColBinds`).
* `linkBinds`: the pair the link holds at the position opens one value under both schemes, which differs from the
  record opening's or from the encoding opening's, so that scheme's binding collides on that opening and the pair's.
-/

@[expose] public section

namespace Claims.Encodings

open Claims
open Claims.Sampling (CommitmentScheme Opening LogColl)
open Claims.Recursion (allOC bindOC)

/-! ## Oracle programs -/

theorem bind_eval (H : Msg → Msg) {α β : Type} (p : OC α) (f : α → OC β) :
    (p.bind f).eval H = (((f (p.eval H).1).eval H).1, (p.eval H).2 ++ ((f (p.eval H).1).eval H).2) := by
  induction p with
  | ret a => simp [OC.bind, OC.eval]
  | query q k ih => simp [OC.bind, OC.eval, ih]

theorem bindOC_eval (H : Msg → Msg) {α β : Type} (p : OC α) (f : α → OC β) :
    (bindOC p f).eval H = (((f (p.eval H).1).eval H).1, (p.eval H).2 ++ ((f (p.eval H).1).eval H).2) := by
  induction p with
  | ret a => simp [bindOC, OC.eval]
  | query q k ih => simp [bindOC, OC.eval, ih]

theorem allOC_cons_eval (H : Msg → Msg) (p : OC Bool) (ps : List (OC Bool)) (hp : (p.eval H).1 = true) :
    (allOC (p :: ps)).eval H = (((allOC ps).eval H).1, (p.eval H).2 ++ ((allOC ps).eval H).2) := by
  show (bindOC p fun b => if b then allOC ps else OC.ret false).eval H = _
  rw [bindOC_eval, hp]
  rfl

/-- Where `allOC` returns `true`, so does each of its programs, and each one's calls are among its calls. -/
theorem allOC_true (H : Msg → Msg) : ∀ ps : List (OC Bool), ((allOC ps).eval H).1 = true →
    ∀ p ∈ ps, (p.eval H).1 = true ∧ ∀ e ∈ (p.eval H).2, e ∈ ((allOC ps).eval H).2
  | [], _, _, hp => absurd hp List.not_mem_nil
  | q :: qs, h, p, hp => by
    cases hq : (q.eval H).1 with
    | false =>
      have hf : ((allOC (q :: qs)).eval H).1 = false := by
        show ((bindOC q fun b => if b then allOC qs else OC.ret false).eval H).1 = false
        rw [bindOC_eval, hq]
        rfl
      rw [hf] at h
      cases h
    | true =>
      rw [allOC_cons_eval H q qs hq] at h ⊢
      rcases List.mem_cons.1 hp with rfl | hp
      · exact ⟨hq, fun e he => List.mem_append_left _ he⟩
      · obtain ⟨h1, h2⟩ := allOC_true H qs h p hp
        exact ⟨h1, fun e he => List.mem_append_right _ (h2 e he)⟩

/-! ## 1. The sum by position -/

theorem byPos_opens_true {α : Type} {enc : List Msg → ℕ → Bool} {Ce Cm : CommitmentScheme α} {roots : List Msg}
    {o : Opening α} (h : enc roots o.pos = true) : (byPos enc Ce Cm).opens roots o = Ce.opens roots o :=
  ite_eq_left h

theorem byPos_opens_false {α : Type} {enc : List Msg → ℕ → Bool} {Ce Cm : CommitmentScheme α} {roots : List Msg}
    {o : Opening α} (h : enc roots o.pos = false) : (byPos enc Ce Cm).opens roots o = Cm.opens roots o :=
  ite_eq_right (by simp [h])

theorem byPos_Opens_true {α : Type} {H : Msg → Msg} {enc : List Msg → ℕ → Bool} {Ce Cm : CommitmentScheme α}
    {roots : List Msg} {o : Opening α} (h : enc roots o.pos = true) :
    (byPos enc Ce Cm).Opens H roots o = Ce.Opens H roots o := by
  simp only [CommitmentScheme.Opens, byPos_opens_true h]

theorem byPos_Opens_false {α : Type} {H : Msg → Msg} {enc : List Msg → ℕ → Bool} {Ce Cm : CommitmentScheme α}
    {roots : List Msg} {o : Opening α} (h : enc roots o.pos = false) :
    (byPos enc Ce Cm).Opens H roots o = Cm.Opens H roots o := by
  simp only [CommitmentScheme.Opens, byPos_opens_false h]

/-- **The sum binds when its parts do.** -/
theorem byPosBinds : ByPosBinds := by
  intro α H enc Ce Cm hCe hCm roots o o' hpos hval ho ho'
  cases he : enc roots o.pos with
  | true =>
    have he' : enc roots o'.pos = true := by rw [← hpos]; exact he
    rw [byPos_Opens_true he] at ho
    rw [byPos_Opens_true he'] at ho'
    rw [byPos_opens_true he, byPos_opens_true he']
    exact hCe roots o o' hpos hval ho ho'
  | false =>
    have he' : enc roots o'.pos = false := by rw [← hpos]; exact he
    rw [byPos_Opens_false he] at ho
    rw [byPos_Opens_false he'] at ho'
    rw [byPos_opens_false he, byPos_opens_false he']
    exact hCm roots o o' hpos hval ho ho'

/-! ## 2. The encoding -/

namespace CodeScheme

variable {M Col Pf : Type} [DecidableEq Col]

theorem scheme_opens_none {C : CodeScheme M Col Pf} {roots : List Msg} {o : Opening Bool}
    (h : C.parse o.data = none) : C.scheme.opens roots o = .ret false := by
  simp [CodeScheme.scheme, h]

theorem scheme_opens_some {C : CodeScheme M Col Pf} {roots : List Msg} {o : Opening Bool} {m : M}
    {cs : List (ℕ × Col × Pf)} (h : C.parse o.data = some (m, cs)) :
    C.scheme.opens roots o = OC.bind (C.colsOK roots cs) fun ok => .ret (ok && C.fits roots m cs o) := by
  simp only [CodeScheme.scheme, h]

/-- An opening that passes parses to a message and columns that pass their checks and fit, and its calls are the
columns' checks'. -/
theorem scheme_passes {C : CodeScheme M Col Pf} {H : Msg → Msg} {roots : List Msg} {o : Opening Bool}
    (h : C.scheme.Opens H roots o = true) :
    ∃ m cs, C.parse o.data = some (m, cs) ∧ ((C.colsOK roots cs).eval H).1 = true ∧ C.fits roots m cs o = true ∧
      ((C.scheme.opens roots o).eval H).2 = ((C.colsOK roots cs).eval H).2 := by
  unfold CommitmentScheme.Opens at h
  obtain ⟨⟨m, cs⟩, hp⟩ : ∃ mc, C.parse o.data = some mc := by
    cases hp : C.parse o.data with
    | none =>
      rw [scheme_opens_none hp] at h
      simp [OC.eval] at h
    | some mc => exact ⟨mc, rfl⟩
  rw [scheme_opens_some hp, bind_eval] at h
  simp only [OC.eval, Bool.and_eq_true] at h
  refine ⟨m, cs, hp, h.1, h.2, ?_⟩
  rw [scheme_opens_some hp, bind_eval]
  simp [OC.eval]

theorem fits_true {C : CodeScheme M Col Pf} {roots : List Msg} {m : M} {cs : List (ℕ × Col × Pf)}
    {o : Opening Bool} (h : C.fits roots m cs o = true) :
    (cs.map (·.1)).Nodup ∧ (∀ q ∈ cs, q.1 < C.len roots) ∧
      C.agree roots ≤ (cs.filter fun q => decide (C.encode roots m q.1 = q.2.1)).length ∧
      C.bit roots m o.pos = some o.val := by
  unfold CodeScheme.fits at h
  rw [Bool.and_eq_true, Bool.and_eq_true, Bool.and_eq_true] at h
  obtain ⟨⟨⟨h1, h2⟩, h3⟩, h4⟩ := h
  exact ⟨of_decide_eq_true h1, fun q hq => of_decide_eq_true (List.all_eq_true.1 h2 q hq), of_decide_eq_true h3,
    of_decide_eq_true h4⟩

/-- The positions of the opened columns `cs` equal to `m`'s codeword there. -/
def agreeSet (C : CodeScheme M Col Pf) (roots : List Msg) (m : M) (cs : List (ℕ × Col × Pf)) : Finset ℕ :=
  ((cs.filter fun q => decide (C.encode roots m q.1 = q.2.1)).map (·.1)).toFinset

theorem agreeSet_card {C : CodeScheme M Col Pf} {roots : List Msg} {m : M} {cs : List (ℕ × Col × Pf)}
    (hnd : (cs.map (·.1)).Nodup) :
    (C.agreeSet roots m cs).card = (cs.filter fun q => decide (C.encode roots m q.1 = q.2.1)).length := by
  unfold agreeSet
  rw [List.toFinset_card_of_nodup (hnd.sublist (List.filter_sublist.map _)), List.length_map]

theorem agreeSet_sub {C : CodeScheme M Col Pf} {roots : List Msg} {m : M} {cs : List (ℕ × Col × Pf)}
    (hlt : ∀ q ∈ cs, q.1 < C.len roots) : C.agreeSet roots m cs ⊆ Finset.range (C.len roots) := by
  intro i hi
  simp only [agreeSet, List.mem_toFinset, List.mem_map, List.mem_filter] at hi
  obtain ⟨q, ⟨hq, _⟩, rfl⟩ := hi
  exact Finset.mem_range.2 (hlt q hq)

theorem mem_agreeSet {C : CodeScheme M Col Pf} {roots : List Msg} {m : M} {cs : List (ℕ × Col × Pf)} {i : ℕ}
    (hi : i ∈ C.agreeSet roots m cs) : ∃ q ∈ cs, q.1 = i ∧ C.encode roots m i = q.2.1 := by
  simp only [agreeSet, List.mem_toFinset, List.mem_map, List.mem_filter, decide_eq_true_eq] at hi
  obtain ⟨q, ⟨hq, he⟩, rfl⟩ := hi
  exact ⟨q, hq, rfl, he⟩

end CodeScheme

/-- **Two sets of `a` or more of `n` positions meet in each set of more than `2·(n − a)` of them.** -/
theorem exists_common {n a d : ℕ} {A A' D : Finset ℕ} (hA : A ⊆ Finset.range n) (hA' : A' ⊆ Finset.range n)
    (ha : a ≤ A.card) (ha' : a ≤ A'.card) (hD : D ⊆ Finset.range n) (hd : 2 * (n - a) < d) (hDd : d ≤ D.card) :
    ∃ i ∈ D, i ∈ A ∧ i ∈ A' := by
  by_contra hne
  have hno : ∀ i ∈ D, i ∈ A → i ∉ A' := fun i hi hiA hiA' => hne ⟨i, hi, hiA, hiA'⟩
  have hsub : D ⊆ (Finset.range n \ A) ∪ (Finset.range n \ A') := by
    intro i hi
    by_cases hiA : i ∈ A
    · exact Finset.mem_union_right _ (Finset.mem_sdiff.2 ⟨hD hi, hno i hi hiA⟩)
    · exact Finset.mem_union_left _ (Finset.mem_sdiff.2 ⟨hD hi, hiA⟩)
  have h1 := Finset.card_le_card hsub
  have h2 := Finset.card_union_le (Finset.range n \ A) (Finset.range n \ A')
  rw [Finset.card_sdiff_of_subset hA, Finset.card_sdiff_of_subset hA', Finset.card_range] at h2
  omega

/-- **The encoding binds**, within the unique decoding radius, at a distance and a binding column check. -/
theorem codeBinds : CodeBinds := by
  intro M Col Pf _ H C d hdist hrad hcol roots o o' hpos hval ho ho'
  obtain ⟨m, cs, -, hok, hfit, hlog⟩ := CodeScheme.scheme_passes ho
  obtain ⟨m', cs', -, hok', hfit', hlog'⟩ := CodeScheme.scheme_passes ho'
  obtain ⟨hnd, hlt, hag, hbit⟩ := CodeScheme.fits_true hfit
  obtain ⟨hnd', hlt', hag', hbit'⟩ := CodeScheme.fits_true hfit'
  have hDd : d roots ≤ ((Finset.range (C.len roots)).filter fun i => C.encode roots m i ≠ C.encode roots m' i).card := by
    by_contra hlt0
    have hb := hdist roots m m' (not_le.1 hlt0) o.pos
    rw [hbit, hpos, hbit'] at hb
    exact hval (Option.some.inj hb)
  obtain ⟨i, hiD, hiA, hiA'⟩ := exists_common (CodeScheme.agreeSet_sub (m := m) hlt)
    (CodeScheme.agreeSet_sub (m := m') hlt') (by rw [CodeScheme.agreeSet_card hnd]; exact hag)
    (by rw [CodeScheme.agreeSet_card hnd']; exact hag') (Finset.filter_subset _ _) (hrad roots) hDd
  obtain ⟨q, hq, hqi, hqe⟩ := CodeScheme.mem_agreeSet hiA
  obtain ⟨q', hq', hqi', hqe'⟩ := CodeScheme.mem_agreeSet hiA'
  have hcne : q.2.1 ≠ q'.2.1 := by
    rw [← hqe, ← hqe']
    exact (Finset.mem_filter.1 hiD).2
  have hall : ((allOC (cs.map fun q => C.colOK roots q.1 q.2.1 q.2.2)).eval H).1 = true := hok
  have hall' : ((allOC (cs'.map fun q => C.colOK roots q.1 q.2.1 q.2.2)).eval H).1 = true := hok'
  obtain ⟨hqok, hqlog⟩ := allOC_true H _ hall _ (List.mem_map.2 ⟨q, hq, rfl⟩)
  obtain ⟨hqok', hqlog'⟩ := allOC_true H _ hall' _ (List.mem_map.2 ⟨q', hq', rfl⟩)
  rw [hqi] at hqok hqlog
  rw [hqi'] at hqok' hqlog'
  refine (hcol roots i q.2.1 q'.2.1 q.2.2 q'.2.2 hcne hqok hqok').mono fun e he => ?_
  rw [hlog, hlog']
  rcases List.mem_append.1 he with he | he
  · exact List.mem_append_left _ (hqlog e he)
  · exact List.mem_append_right _ (hqlog' e he)

/-! ## 3. The link -/

/-- Where the link holds a value, it holds a pair at the position whose openings pass and agree. -/
theorem linkedAt_some {H : Msg → Msg} {Cm Ce : CommitmentScheme Bool} {roots : List Msg}
    {ℓ : List (Opening Bool × Opening Bool)} {g : ℕ} (h : linkedAt H Cm Ce roots ℓ g ≠ none) :
    ∃ p ∈ ℓ, p.1.pos = g ∧ p.2.pos = g ∧ p.1.val = p.2.val ∧ Cm.Opens H roots p.1 = true ∧
      Ce.Opens H roots p.2 = true := by
  obtain ⟨b, hb⟩ := Option.ne_none_iff_exists'.1 h
  unfold linkedAt at hb
  obtain ⟨p, hp, -⟩ := Option.map_eq_some_iff.1 hb
  have hpred := List.find?_some hp
  simp only [Bool.and_eq_true, beq_iff_eq] at hpred
  obtain ⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, h5⟩ := hpred
  exact ⟨p, List.mem_of_find?_eq_some hp, h1, h2, h3, h4, h5⟩

theorem linkLog_mem {H : Msg → Msg} {Cm Ce : CommitmentScheme Bool} {roots : List Msg}
    {ℓ : List (Opening Bool × Opening Bool)} {p : Opening Bool × Opening Bool} (hp : p ∈ ℓ) {x : Msg × Msg}
    (hx : x ∈ ((Cm.opens roots p.1).eval H).2 ∨ x ∈ ((Ce.opens roots p.2).eval H).2) :
    x ∈ linkLog H Cm Ce roots ℓ :=
  List.mem_flatMap.2 ⟨p, hp, List.mem_append.2 hx⟩

/-- **The link binds the record to the encoding.** -/
theorem linkBinds : LinkBinds := by
  intro H Cm Ce roots ℓ a e hCm hCe hpos hlk ha he hne
  obtain ⟨p, hp, hp1, hp2, hpv, hpm, hpe⟩ := linkedAt_some hlk
  by_cases hav : a.val = p.1.val
  · have hev : e.val ≠ p.2.val := fun h => hne (by rw [hav, hpv, h])
    refine (hCe roots e p.2 (by rw [hp2, hpos]) hev he hpe).mono fun x hx => ?_
    rcases List.mem_append.1 hx with hx | hx
    · exact List.mem_append_left _ (List.mem_append_right _ hx)
    · exact List.mem_append_right _ (linkLog_mem hp (Or.inr hx))
  · refine (hCm roots a p.1 hp1.symm hav ha hpm).mono fun x hx => ?_
    rcases List.mem_append.1 hx with hx | hx
    · exact List.mem_append_left _ (List.mem_append_left _ hx)
    · exact List.mem_append_right _ (linkLog_mem hp (Or.inl hx))

end Claims.Encodings
