import Flock.RecVStar
import Batteries.Tactic.OpenPrivate

/-!
# R's stream's first top makes R's registered entry

`RecVStar.candidate3` holds a stream only with a round `k = 0` among its rounds (its count check), and walks the stream's
rounds sorted by `k` (`Array.qsort`), so round 0, R's binding round, comes first (`qsort_min`, through core's private
`qsort.sort` and `qpartition.loop`: a least key is first). R's stream starts with no tops, its binding round commits one,
and the session refuses a binding round whose first top does not make `regR` (`Made`). So R's stream's first top makes
`regR` (`candidate3_top`), which `RecVStar.innerCommits` takes from the registration (`innerCommits_top`).
-/

open private Array.qsort.sort Array.qpartition.loop from Init.Data.Array.QSort.Basic

namespace FlockC3.RegLevels.TopHold

section QSort

variable {α : Type} {n : Nat}

/-- `bs` holds the elements `as` holds on `[lo, hi]` there, as a set, and `as`'s own elements outside it. -/
def SameOn (lo hi : Nat) (as bs : Vector α n) : Prop :=
  (∀ j (hj : j < n), (j < lo ∨ hi < j) → bs[j] = as[j]) ∧
    ∀ x, (∃ j, ∃ hj : j < n, lo ≤ j ∧ j ≤ hi ∧ as[j] = x) ↔ (∃ j, ∃ hj : j < n, lo ≤ j ∧ j ≤ hi ∧ bs[j] = x)

theorem SameOn.trans {lo hi : Nat} {as bs cs : Vector α n} (h₁ : SameOn lo hi as bs) (h₂ : SameOn lo hi bs cs) :
    SameOn lo hi as cs :=
  ⟨fun j hj o => (h₂.1 j hj o).trans (h₁.1 j hj o), fun x => (h₁.2 x).trans (h₂.2 x)⟩

theorem SameOn.mono {lo hi lo' hi' : Nat} {as bs : Vector α n} (h : SameOn lo' hi' as bs) (hl : lo ≤ lo')
    (hh : hi' ≤ hi) : SameOn lo hi as bs := by
  refine ⟨fun j hj o => h.1 j hj (by omega), fun x => ⟨?_, ?_⟩⟩
  · rintro ⟨j, hj, h1, h2, rfl⟩
    by_cases hin : lo' ≤ j ∧ j ≤ hi'
    · obtain ⟨j', hj', _, _, e⟩ := (h.2 as[j]).1 ⟨j, hj, hin.1, hin.2, rfl⟩
      exact ⟨j', hj', by omega, by omega, e⟩
    · exact ⟨j, hj, h1, h2, h.1 j hj (by omega)⟩
  · rintro ⟨j, hj, h1, h2, rfl⟩
    by_cases hin : lo' ≤ j ∧ j ≤ hi'
    · obtain ⟨j', hj', _, _, e⟩ := (h.2 bs[j]).2 ⟨j, hj, hin.1, hin.2, rfl⟩
      exact ⟨j', hj', by omega, by omega, e⟩
    · exact ⟨j, hj, h1, h2, (h.1 j hj (by omega)).symm⟩

theorem sameOn_swap {lo hi : Nat} (as : Vector α n) {i k : Nat} (hi' : i < n) (hk : k < n) (h1 : lo ≤ i)
    (h2 : i ≤ hi) (h3 : lo ≤ k) (h4 : k ≤ hi) : SameOn lo hi as (as.swap i k hi' hk) := by
  refine ⟨fun j hj o => Vector.getElem_swap_of_ne (by omega) (by omega), fun x => ⟨?_, ?_⟩⟩
  · rintro ⟨j, hj, h5, h6, rfl⟩
    by_cases e1 : j = i
    · subst e1; exact ⟨k, hk, h3, h4, Vector.getElem_swap_right _ _⟩
    by_cases e2 : j = k
    · subst e2; exact ⟨i, hi', h1, h2, Vector.getElem_swap_left _ _⟩
    exact ⟨j, hj, h5, h6, Vector.getElem_swap_of_ne e1 e2⟩
  · rintro ⟨j, hj, h5, h6, rfl⟩
    rw [Vector.getElem_swap]
    split
    · exact ⟨k, hk, h3, h4, rfl⟩
    split
    · exact ⟨i, hi', h1, h2, rfl⟩
    exact ⟨j, hj, h5, h6, rfl⟩

theorem sameOn_ite_swap {lo hi : Nat} (as : Vector α n) (c : Bool) {i k : Nat} (hi' : i < n) (hk : k < n)
    (h1 : lo ≤ i) (h2 : i ≤ hi) (h3 : lo ≤ k) (h4 : k ≤ hi) :
    SameOn lo hi as (if c = true then as.swap i k hi' hk else as) := by
  split
  · exact sameOn_swap as hi' hk h1 h2 h3 h4
  · exact ⟨fun _ _ _ => rfl, fun _ => Iff.rfl⟩

section Loop

variable (lt : α → α → Bool) {lo hi : Nat} (hhi : hi < n) (pivot : α)

theorem loop_sameOn :
    ∀ (d : Nat) (as : Vector α n) (i k : Nat) (ilo : lo ≤ i) (ik : i ≤ k) (w : k ≤ hi), hi - k = d →
      SameOn lo hi as (Array.qpartition.loop lt lo hi hhi pivot as i k ilo ik w).2 := by
  intro d
  induction d with
  | zero =>
    intro as i k ilo ik w hd
    rw [Array.qpartition.loop.eq_def]
    have : ¬ k < hi := by omega
    simp only [this, dite_false]
    exact sameOn_swap as _ _ ilo (by omega) (by omega) (Nat.le_refl _)
  | succ d ih =>
    intro as i k ilo ik w hd
    rw [Array.qpartition.loop.eq_def]
    have hk : k < hi := by omega
    simp only [hk, dite_true]
    split
    · exact (sameOn_swap as _ _ ilo (by omega) (by omega) (by omega)).trans (ih _ _ _ _ _ _ (by omega))
    · exact ih _ _ _ _ _ _ (by omega)

/-- The loop's split: past its pivot's place, no element is below the pivot, which it puts there. -/
theorem loop_split :
    ∀ (d : Nat) (as : Vector α n) (i k : Nat) (ilo : lo ≤ i) (ik : i ≤ k) (w : k ≤ hi), hi - k = d →
      as[hi] = pivot → (∀ j (hj : j < n), i ≤ j → j < k → lt as[j] pivot = false) →
      ∀ m as', Array.qpartition.loop lt lo hi hhi pivot as i k ilo ik w = (m, as') →
        as'[m.1]'(Nat.lt_of_le_of_lt m.2.2 hhi) = pivot ∧
          ∀ j (hj : j < n), m.1 < j → j ≤ hi → lt as'[j] pivot = false := by
  intro d
  induction d with
  | zero =>
    intro as i k ilo ik w hd hp hb m as' he
    rw [Array.qpartition.loop.eq_def] at he
    have : ¬ k < hi := by omega
    simp only [this, dite_false, Prod.mk.injEq] at he
    obtain ⟨rfl, rfl⟩ := he
    refine ⟨by simp [hp], fun j hj h1 h2 => ?_⟩
    dsimp only at h1
    rw [Vector.getElem_swap]
    have hji : j ≠ i := by omega
    simp only [hji, ite_false]
    split
    · exact hb i (by omega) (Nat.le_refl _) (by omega)
    · exact hb j hj (by omega) (by omega)
  | succ d ih =>
    intro as i k ilo ik w hd hp hb m as' he
    rw [Array.qpartition.loop.eq_def] at he
    have hk : k < hi := by omega
    simp only [hk, dite_true] at he
    split at he
    · rename_i hlt
      refine ih _ _ _ _ _ _ (by omega) ?_ ?_ m as' he
      · rw [Vector.getElem_swap_of_ne (by omega) (by omega)]; exact hp
      · intro j hj h1 h2
        rw [Vector.getElem_swap]
        have hji : j ≠ i := by omega
        simp only [hji, ite_false]
        split
        · exact hb i (by omega) (Nat.le_refl _) (by omega)
        · exact hb j hj (by omega) (by omega)
    · rename_i hlt
      refine ih _ _ _ _ _ _ (by omega) hp ?_ m as' he
      intro j hj h1 h2
      by_cases hjk : j = k
      · subst hjk; simpa using hlt
      · exact hb j hj h1 (by omega)

/-- The loop puts its pivot before `hi` once it has passed a place `mid < hi` whose element is not below it. -/
theorem loop_lt {mid : Nat} (hmid : mid < hi) :
    ∀ (d : Nat) (as : Vector α n) (i k : Nat) (ilo : lo ≤ i) (ik : i ≤ k) (w : k ≤ hi), hi - k = d →
      (i < k ∨ (k ≤ mid ∧ lt (as[mid]'(by omega)) pivot = false)) →
      (Array.qpartition.loop lt lo hi hhi pivot as i k ilo ik w).1.1 < hi := by
  intro d
  induction d with
  | zero =>
    intro as i k ilo ik w hd hinv
    rw [Array.qpartition.loop.eq_def]
    have : ¬ k < hi := by omega
    simp only [this, dite_false]
    omega
  | succ d ih =>
    intro as i k ilo ik w hd hinv
    rw [Array.qpartition.loop.eq_def]
    have hk : k < hi := by omega
    simp only [hk, dite_true]
    split
    · rename_i hlt
      refine ih _ _ _ _ _ _ (by omega) ?_
      rcases hinv with hinv | ⟨h1, h2⟩
      · exact .inl (by omega)
      · by_cases hkm : k = mid
        · subst hkm; rw [h2] at hlt; cases hlt
        · right
          refine ⟨by omega, ?_⟩
          rw [Vector.getElem_swap_of_ne (by omega) (by omega)]; exact h2
    · refine ih _ _ _ _ _ _ (by omega) ?_
      rcases hinv with hinv | ⟨h1, h2⟩
      · exact .inl (by omega)
      · by_cases hkm : k = mid
        · exact .inl (by omega)
        · exact .inr ⟨by omega, h2⟩

end Loop

theorem qsort_size (as : Array α) (lt : α → α → Bool) : (as.qsort lt).size = as.size := by
  unfold Array.qsort
  split
  · rfl
  · simp

theorem ite_swap_lt (lt : α → α → Bool) (hasym : ∀ a b, lt a b = true → lt b a = false) (b : Vector α n) {i k : Nat}
    (hi' : i < n) (hk : k < n) :
    lt (if lt b[i] b[k] = true then b.swap i k hi' hk else b)[i]
      (if lt b[i] b[k] = true then b.swap i k hi' hk else b)[k] = false := by
  split
  · rename_i h; simp only [Vector.getElem_swap_left, Vector.getElem_swap_right]; exact hasym _ _ h
  · rename_i h; simpa using h

theorem loop_spec (lt : α → α → Bool) {lo hi : Nat} (hhi : hi < n) {as as₃ : Vector α n} (w : lo ≤ hi) {mid : Nat}
    (hm1 : lo ≤ mid) (hm2 : mid ≤ hi) (hmid : lo < hi → mid < hi) (hS : SameOn lo hi as as₃)
    (H : lt (as₃[mid]'(by omega)) as₃[hi] = false) :
    ∀ m as', Array.qpartition.loop lt lo hi hhi as₃[hi] as₃ lo lo (Nat.le_refl _) (Nat.le_refl _) w = (m, as') →
      SameOn lo hi as as' ∧ (lo < hi → m.1 < hi) ∧
        ∀ j (hj : j < n), m.1 < j → j ≤ hi → lt as'[j] (as'[m.1]'(Nat.lt_of_le_of_lt m.2.2 hhi)) = false := by
  intro m as' he
  refine ⟨hS.trans ?_, fun h => ?_, ?_⟩
  · have := loop_sameOn lt (lo := lo) hhi as₃[hi] _ as₃ lo lo (Nat.le_refl _) (Nat.le_refl _) w rfl; rw [he] at this; exact this
  · have := loop_lt lt (lo := lo) hhi as₃[hi] (hmid h) _ as₃ lo lo (Nat.le_refl _) (Nat.le_refl _) w rfl (.inr ⟨hm1, H⟩); rw [he] at this; exact this
  · obtain ⟨h1, h2⟩ := loop_split lt (lo := lo) hhi as₃[hi] _ as₃ lo lo (Nat.le_refl _) (Nat.le_refl _) w rfl rfl
      (fun j _ h1 h2 => absurd h2 (by omega)) m as' he
    intro j hj h3 h4; rw [h1]; exact h2 j hj h3 h4

theorem qpartition_spec (lt : α → α → Bool) (hasym : ∀ a b, lt a b = true → lt b a = false) (as : Vector α n)
    {lo hi : Nat} (w : lo ≤ hi) (hlo : lo < n) (hhi : hi < n) :
    ∀ m as', Array.qpartition as lt lo hi w hlo hhi = (m, as') →
      SameOn lo hi as as' ∧ (lo < hi → m.1 < hi) ∧
        ∀ j (hj : j < n), m.1 < j → j ≤ hi → lt as'[j] (as'[m.1]'(Nat.lt_of_le_of_lt m.2.2 hhi)) = false := by
  intro m as' he
  unfold Array.qpartition at he
  refine loop_spec lt hhi w (mid := (lo + hi) / 2) (by omega) (by omega) (fun _ => by omega) ?_ ?_ m as' he
  · exact (sameOn_ite_swap _ _ _ _ (by omega) (by omega) (by omega) (by omega)).trans
      ((sameOn_ite_swap _ _ _ _ (by omega) (by omega) (by omega) (by omega)).trans
        (sameOn_ite_swap _ _ _ _ (by omega) (by omega) (by omega) (by omega)))
  · exact ite_swap_lt lt hasym _ _ _

theorem sort_spec (lt : α → α → Bool) (hasym : ∀ a b, lt a b = true → lt b a = false)
    (htr : ∀ a b c, lt a b = false → lt b c = false → lt a c = false) :
    ∀ (d : Nat) (as : Vector α n) (lo hi : Nat) (w : lo ≤ hi) (hlo : lo < n) (hhi : hi < n), hi - lo < d →
      SameOn lo hi as (Array.qsort.sort lt as lo hi w hlo hhi) ∧
        ∀ j (hj : j < n), lo ≤ j → j ≤ hi → lt as[j] (Array.qsort.sort lt as lo hi w hlo hhi)[lo] = false := by
  intro d
  induction d with
  | zero => intro _ _ _ _ _ _ hd; omega
  | succ d ih =>
    intro as lo hi w hlo hhi hd
    rw [Array.qsort.sort.eq_def]
    split
    · rename_i hlt
      generalize hq : Array.qpartition as lt lo hi w hlo hhi = q
      obtain ⟨⟨mid, hmid⟩, as'⟩ := q
      obtain ⟨hS, hlt', hsp⟩ := qpartition_spec lt hasym as w hlo hhi _ _ hq
      have hmh : mid < hi := hlt' hlt
      simp only
      split
      · omega
      · obtain ⟨hS₁, hm₁⟩ := ih as' lo mid hmid.1 hlo (by omega) (by omega)
        obtain ⟨hS₂, _⟩ := ih (Array.qsort.sort lt as' lo mid hmid.1 hlo (by omega)) (mid + 1) hi (by omega)
          (by omega) hhi (by omega)
        refine ⟨hS.trans ((hS₁.mono (Nat.le_refl _) (by omega)).trans (hS₂.mono (by omega) (Nat.le_refl _))), ?_⟩
        intro j hj h1 h2
        rw [hS₂.1 lo hlo (.inl (by omega))]
        obtain ⟨j', hj', h3, h4, e⟩ := (hS.2 as[j]).1 ⟨j, hj, h1, h2, rfl⟩
        rw [← e]
        by_cases hjm : j' ≤ mid
        · exact hm₁ j' hj' h3 hjm
        · exact htr _ _ _ (hsp j' hj' (by show mid < j'; omega) h4) (hm₁ mid (by omega) hmid.1 (Nat.le_refl _))
    · refine ⟨⟨fun _ _ _ => rfl, fun _ => Iff.rfl⟩, fun j hj h1 h2 => ?_⟩
      have : j = lo := by omega
      subst this
      cases h : lt as[j] as[j]
      · rfl
      · have := hasym _ _ h; rw [h] at this; cases this

/-- **`Array.qsort` by a key puts a least key first**. -/
theorem qsort_min (as : Array α) (f : α → Nat) {x y : α} (hx : x ∈ as)
    (hy : (as.qsort fun a b => decide (f a < f b))[0]? = some y) : f y ≤ f x := by
  have hasym : ∀ a b, decide (f a < f b) = true → decide (f b < f a) = false := by
    intro a b h; simp only [decide_eq_true_eq, decide_eq_false_iff_not] at h ⊢; omega
  have htr : ∀ a b c, decide (f a < f b) = false → decide (f b < f c) = false → decide (f a < f c) = false := by
    intro a b c h1 h2; simp only [decide_eq_false_iff_not] at h1 h2 ⊢; omega
  obtain ⟨j, hj, rfl⟩ := Array.getElem_of_mem hx
  unfold Array.qsort at hy
  split at hy
  · omega
  · simp only at hy
    have key : ∀ (r : Vector α as.size) (lo : Nat) (hlo : lo < as.size), lo = 0 → r.toArray[0]? = some y → r[lo] = y := by
      intro r lo hlo e h; subst e
      rw [Vector.getElem?_toArray, Vector.getElem?_eq_getElem hlo] at h
      exact Option.some.inj h
    obtain ⟨_, hm⟩ := sort_spec _ hasym htr _ as.toVector (min 0 (as.size - 1))
      (max (min 0 (as.size - 1)) (min (as.size - 1) (as.size - 1))) (by omega) (by omega) (by omega)
      (Nat.lt_succ_self _)
    have h := hm j hj (by omega) (by omega)
    rw [key _ _ _ (Nat.zero_min _) hy] at h
    simp at h
    exact Nat.le_of_not_lt (of_decide_eq_false h)

end QSort


section Session

open Lean Flock.RecVStar

open String in
theorem tbl_ne_stacked {n : String} (hn : n.startsWith "stacked/" = false) :
    (((n.dropEndWhile (· != '/')).toString.dropEnd 1).toString) ≠ "stacked" := by
  intro h
  have hl := congrArg String.toList h
  have he := endsWith_dropEndWhile (s := n) (pat := (· != '/'))
  rw [Slice.endsWith_bool_eq_getLast?] at he
  have ha := dropEndWhile_append_takeEndWhile (s := n) (pat := (· != '/'))
  rw [startsWith_string_eq_false_iff] at hn
  apply hn
  generalize (n.dropEndWhile (· != '/')) = d at he ha hl
  simp only [Slice.toString_eq, toList_copy_dropEnd] at hl
  rcases hd : d.copy.toList.eq_nil_or_concat with h0 | ⟨L, c, hc⟩
  · rw [h0] at hl; simp at hl
  · rw [hc] at hl he
    simp at hl he
    subst hl he
    rw [← ha, String.toList_append, hc]
    exact ⟨_, by simp; rfl⟩

theorem eok_bind {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : (x >>= f) = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error e => cases h
  | ok a => exact ⟨a, rfl, h⟩

/-- `bcs` makes R's registered entry: `regR` is the registration's `program` and `stacked` entry, and `topEntry` at
`bcs` has the entry's canonical form. -/
def Made (regR : Option (String × Json)) (bcs : Array ByteArray) : Prop :=
  ∃ prog e, regR = some (prog, e) ∧ topEntryFields prog bcs e = .ok true ∧
    ∃ e' str, topEntry prog bcs = .ok e' ∧ Flock.canon e' = .ok str ∧ Flock.canon e = .ok str

theorem forIn_all {α ε : Type} (P : α → Prop) {f : α → Unit → Except ε (ForInStep Unit)}
    (hf : ∀ a u r, f a u = .ok r → P a ∧ ∃ u', r = .yield u') :
    ∀ (l : List α) {u u' : Unit}, forIn l u f = .ok u' → ∀ a ∈ l, P a
  | [], _, _, _, a, ha => by cases ha
  | b :: l, u, u', h, a, ha => by
    rw [List.forIn_cons] at h
    obtain ⟨st, hst, h⟩ := eok_bind h
    obtain ⟨hb, u'', rfl⟩ := hf b u st hst
    cases ha with
    | head => exact hb
    | tail _ ha => exact forIn_all P hf l h a ha

theorem forIn_inv {α σ ε : Type} (P : σ → Prop) {f : α → σ → Except ε (ForInStep σ)} :
    ∀ (l : List α) {s s' : σ}, (∀ a ∈ l, ∀ s r, P s → f a s = .ok r → ∃ s', r = .yield s' ∧ P s') → P s →
      forIn l s f = .ok s' → P s'
  | [], s, s', _, hs, h => by
    rw [List.forIn_nil] at h
    cases h; exact hs
  | b :: l, s, s', hf, hs, h => by
    rw [List.forIn_cons] at h
    obtain ⟨st, hst, h⟩ := eok_bind h
    obtain ⟨s'', rfl, hs''⟩ := hf b (.head _) s st hs hst
    exact forIn_inv P l (fun a ha => hf a (.tail _ ha)) hs'' h

theorem exists_fn {α σ ε : Type} {l : List α} {s s' : σ} {f : α → σ → Except ε (ForInStep σ)}
    (h : forIn l s f = .ok s') : ∃ g, g = f ∧ forIn l s g = .ok s' := ⟨f, rfl, h⟩

theorem forIn_head {α σ ε : Type} (R : α → σ → σ → Prop) (Q : σ → Prop) {f : α → σ → Except ε (ForInStep σ)}
    {a₀ : α} {l : List α} {s s' : σ}
    (hf : ∀ a t st, f a t = .ok st → ∃ t', st = .yield t' ∧ R a t t')
    (h0 : ∀ t', R a₀ s t' → Q t') (hQ : ∀ a t t', Q t → R a t t' → Q t')
    (h : forIn (a₀ :: l) s f = .ok s') : Q s' := by
  rw [List.forIn_cons] at h
  obtain ⟨st, hst, h⟩ := eok_bind h
  obtain ⟨t₁, rfl, hr⟩ := hf a₀ s st hst
  exact forIn_inv Q l (fun a _ t st hQt hst => let ⟨t', e, hr'⟩ := hf a t st hst; ⟨t', e, hQ a t t' hQt hr'⟩)
    (h0 t₁ hr) h

theorem push_keep {β : Type} {t : Array β} {b x : β} (h : t[0]? = some b) : (t.push x)[0]? = some b := by
  rw [Array.getElem?_push]
  have : 0 < t.size := by
    rcases Nat.eq_zero_or_pos t.size with e | e
    · simp [Array.getElem?_eq_none (Nat.le_of_eq e)] at h
    · exact e
  simp [Nat.ne_of_lt this, h]

/-- **Candidate 3's session holds R's binding top**: the first top of the session's `stacked/rep0` stream makes `regR`. -/
theorem candidate3_top {tr : Json} {rounds : Array Json} {t : String} {names : List String}
    {msgs : Array (String × Nat × ByteArray)} {regR : Option (String × Json)} {shape : Option StackedShape}
    {out : Array InnerStream} (ht : (t.dropEnd 1).toString ≠ "stacked")
    (h : candidate3 tr rounds t names msgs regR shape = .ok out) :
    ∀ x ∈ out, x.name = "stacked/rep0" → ∀ bcs, x.tops[0]? = some bcs → Made regR bcs := by
  unfold candidate3 at h
  extract_lets stacked want tbl same tables wantTables rootOf0 out0 seen0 jpRoots jpTables jpDecl jpStart at h
  split at h
  · cases h
  simp only [jpStart] at h
  split at h
  rotate_left
  · cases h
  obtain ⟨declared, -, h⟩ := eok_bind h
  simp (config := { zeta := false }) only [jpDecl] at h
  extract_lets decl count at h
  split at h
  · cases h
  obtain ⟨u, h₁, h⟩ := eok_bind h
  have hP : ∀ s ∈ want, (rounds.filter fun r => (r.getObjValAs? String "stream").toOption == some s).size ≠ 0 →
      ∃ r ∈ rounds.filter (fun r => (r.getObjValAs? String "stream").toOption == some s),
        (r.getObjValAs? Nat "k").toOption = some 0 := by
    refine forIn_all _ ?_ want h₁
    intro s _ r hr
    extract_lets parent at_ mine jpRange jpSize at hr
    split at hr
    · cases hr
    simp (config := { zeta := false }) only [jpSize] at hr
    split at hr
    · cases hr
    rename_i hsz
    obtain ⟨u, hu, hr⟩ := eok_bind hr
    refine ⟨fun hne => ?_, _, (Except.ok.inj hr).symm⟩
    rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hu
    simp only [mine, bne_iff_ne, ne_eq, Decidable.not_not, Array.size_map] at hsz
    obtain ⟨c, hc⟩ : ∃ c, count s = c + 1 := ⟨count s - 1, by omega⟩
    have hl : List.range' [:count s].start [:count s].size [:count s].step = 0 :: List.range' 1 c 1 := by
      simp [Std.Legacy.Range.size, hc, List.range'_succ]
    rw [hl, List.forIn_cons] at hu
    obtain ⟨st, hst, -⟩ := eok_bind hu
    split at hst
    · cases hst
    rename_i hany
    simp only [Bool.not_eq_true', Bool.not_eq_false, mine, Array.any_map] at hany
    obtain ⟨r, hr, e⟩ := Array.any_eq_true'.1 hany
    exact ⟨r, hr, by simpa using e⟩
  split at h
  · cases h
  simp (config := { zeta := false }) only [jpTables] at h
  split at h
  rotate_left
  · cases h
  obtain ⟨roots, -, h⟩ := eok_bind h
  simp (config := { zeta := false }) only [jpRoots] at h
  extract_lets jpR at h
  split at h
  · cases h
  simp (config := { zeta := false }) only [jpR] at h
  obtain ⟨rootOf, -, h⟩ := eok_bind h
  extract_lets rootOf' out1 at h
  obtain ⟨st, hmain, h⟩ := eok_bind h
  extract_lets o a at h
  obtain ⟨_, -, h⟩ := eok_bind h
  have hout : o = out := Except.ok.inj h
  subst hout
  refine forIn_inv (fun st : Array InnerStream × Array ByteArray => ∀ x ∈ st.1, x.name = "stacked/rep0" →
    ∀ bcs, x.tops[0]? = some bcs → Made regR bcs) want ?_ (fun x hx => absurd hx (by simp [out0])) hmain
  intro s hs ⟨o, a⟩ r hinv hr
  extract_lets o' a' table committed tops rs cs at hr
  obtain ⟨topsF, hrs, hr⟩ := eok_bind hr
  extract_lets topsF' allN jpEnd at hr
  have hend : jpEnd () = .ok r := by
    split at hr
    · split at hr
      · obtain ⟨u, -, hr⟩ := eok_bind hr
        exact hr
      · cases hr
    · exact hr
  obtain ⟨coins, -, hend⟩ := eok_bind hend
  refine ⟨_, (Except.ok.inj hend).symm, ?_⟩
  intro y hy hname bcs hbcs
  rcases Array.mem_push.1 hy with hy | rfl
  · exact hinv y hy hname bcs hbcs
  · simp only at hname hbcs
    subst hname
    have htable : table = some STACKED := by
      have h1 : "stacked/rep0".startsWith (toString SPARSE ++ toString "/") = false := by decide +kernel
      have h2 : "stacked/rep0".startsWith (toString STACKED ++ toString "/") = true := by decide +kernel
      simp only [table, h1, h2]
      rfl
    rw [htable] at hbcs
    simp only [Option.isSome_some, ite_true] at hbcs
    have hcomm : committed = false := by
      simp only [committed, htable, tbl]
      cases e : (some STACKED == some (t.dropEnd 1).toString)
      · rfl
      · exact absurd (Option.some.inj (beq_iff_eq.1 e)).symm ht
    have htops : tops = #[] := by simp [tops, hcomm]
    rw [← Array.forIn_toList] at hrs
    obtain ⟨g, hg, hrs⟩ := exists_fn hrs
    have hG : ∀ a tp st, g a tp = .ok st → ∃ tp', st = .yield tp' ∧
        ((∀ b, tp[0]? = some b → tp'[0]? = some b) ∧
          ((a.getObjValAs? Nat "k").toOption.getD 0 = 0 → tp = #[] → ∃ b, tp'[0]? = some b ∧ Made regR b)) := by
      intro a tp st h
      rw [hg] at h
      dsimp (config := { zeta := false }) only at h
      extract_lets tops' k at h
      split at h
      rotate_left
      · cases h
      rename_i ts _
      extract_lets jpZ jpB at h
      split at h
      · cases h
      simp (config := { zeta := false }) only [jpB] at h
      split at h
      · cases h
      rename_i hne
      obtain ⟨tF, hz, h⟩ := eok_bind h
      refine ⟨tF, (Except.ok.inj h).symm, ?_⟩
      rw [← Array.forIn_toList] at hz
      obtain ⟨inner, hi, hz⟩ := exists_fn hz
      have hI : ∀ x tq st, inner x tq = .ok st → ∃ bcs, st = .yield (tq.push bcs) ∧
          ((table == some STACKED && k == 0 && x.2 == 0) = true → Made regR bcs) := by
        intro x tq st h
        rw [hi] at h
        dsimp (config := { zeta := false }) only at h
        extract_lets tq' jpT at h
        split at h
        · cases h
        obtain ⟨bcs, -, h⟩ := eok_bind h
        extract_lets pushed jpPush jpMade at h
        split at h
        · rename_i hc
          rcases hR : regR with _ | ⟨program, e⟩
          rotate_left
          · rw [hR] at h
            obtain ⟨structural, hfields, h⟩ := eok_bind h
            obtain ⟨e', h1, hA⟩ := eok_bind h
            obtain ⟨s1, h2, hB⟩ := eok_bind hA
            obtain ⟨s2, h3, hC⟩ := eok_bind hB
            obtain ⟨made, hm, hD⟩ := eok_bind hC
            simp (config := { zeta := false }) only [jpMade] at hD
            split at hD
            · cases hD
            rename_i hmade
            have hm' : (structural && (s1 == s2)) = made := Except.ok.inj hm
            have hmade' : made = true := by simpa using hmade
            have hboth : structural && (s1 == s2) = true := by
              exact hm'.trans hmade'
            simp only [Bool.and_eq_true] at hboth
            have hstruct : structural = true := hboth.1
            have hs : s1 = s2 := beq_iff_eq.mp hboth.2
            rw [hstruct] at hfields
            refine ⟨bcs, (Except.ok.inj hD).symm,
              fun _ => ⟨program, e, rfl, hfields, e', s1, h1, h2, ?_⟩⟩
            rw [hs]; exact h3
          · rw [hR] at h
            obtain ⟨made, hm, hD⟩ := eok_bind h
            simp (config := { zeta := false }) only [jpMade] at hD
            have hm' : false = made := Except.ok.inj hm
            subst hm'
            cases hD
        · exact ⟨bcs, (Except.ok.inj h).symm, fun hc => absurd hc ‹_›⟩
      refine ⟨fun b hb => forIn_inv (fun tq => tq[0]? = some b) _ (fun x _ tq st hq hst => ?_) hb hz,
        fun hk htp => ?_⟩
      · obtain ⟨bcs, e, -⟩ := hI x tq st hst
        exact ⟨_, e, push_keep hq⟩
      · have hk' : k = 0 := hk
        rw [htable, hk'] at hne
        have hts : ts.toList ≠ [] := by
          intro e
          apply hne
          simp [Array.isEmpty_iff, ← Array.toList_eq_nil_iff, e]
        obtain ⟨top₀, rest, hl⟩ := List.exists_cons_of_ne_nil hts
        rw [Array.toList_zipIdx, hl, List.zipIdx_cons] at hz
        refine forIn_head (fun x tq tq' => ∃ bcs, tq' = tq.push bcs ∧
            ((table == some STACKED && k == 0 && x.2 == 0) = true → Made regR bcs))
          (fun tq => ∃ b, tq[0]? = some b ∧ Made regR b)
          (fun x tq st hst => let ⟨bcs, e, hm⟩ := hI x tq st hst; ⟨_, e, bcs, rfl, hm⟩) ?_ ?_ hz
        · rintro _ ⟨bcs, rfl, hm⟩
          refine ⟨bcs, ?_, hm ?_⟩
          · simp [tops', htp]
          · simp [htable, hk']
        · rintro x tq _ ⟨b, hb, hm⟩ ⟨bcs, rfl, -⟩
          exact ⟨b, push_keep hb, hm⟩
    have hsz : rs.size = (rounds.filter fun r => (r.getObjValAs? String "stream").toOption == some "stacked/rep0").size :=
      qsort_size _ _
    cases hl : rs.toList with
    | nil =>
      rw [hl, List.forIn_nil] at hrs
      have e : tops = topsF := Except.ok.inj hrs
      have : topsF'[0]? = (#[] : Array (Array ByteArray))[0]? := by
        show topsF[0]? = _
        rw [← e, htops]
      rw [this] at hbcs
      simp at hbcs
    | cons r₀ rest =>
      have hne : (rounds.filter fun r => (r.getObjValAs? String "stream").toOption == some "stacked/rep0").size ≠ 0 := by
        rw [← hsz, ← Array.length_toList, hl]
        simp
      obtain ⟨x, hx, hx0⟩ := hP _ hs hne
      have hr₀ : rs[0]? = some r₀ := by rw [← Array.getElem?_toList, hl]; rfl
      have hk : (r₀.getObjValAs? Nat "k").toOption.getD 0 = 0 := by
        have := qsort_min _ (fun r => (r.getObjValAs? Nat "k").toOption.getD 0) hx hr₀
        simp only [hx0, Option.getD_some] at this
        omega
      rw [hl] at hrs
      obtain ⟨b, hb, hm⟩ := forIn_head (fun a tp tp' => (∀ b, tp[0]? = some b → tp'[0]? = some b) ∧
          ((a.getObjValAs? Nat "k").toOption.getD 0 = 0 → tp = #[] → ∃ b, tp'[0]? = some b ∧ Made regR b))
        (fun tp => ∃ b, tp[0]? = some b ∧ Made regR b) hG (fun _ h => h.2 hk htops)
        (fun _ _ _ ⟨b, hb, hm⟩ h => ⟨b, h.1 b hb, hm⟩) hrs
      have : topsF'[0]? = some b := hb
      rw [this] at hbcs
      cases hbcs
      exact hm

/-- **The verifier's hold of the inner transcript holds R's registered top**: R's stream's first top makes the `stacked`
entry of the registration `reg` binds. -/
theorem innerCommits_top {tr : Json} {record : String} {reg : Json} {bound : Option String}
    {shape : Option StackedShape} {cs : InnerCommits} {s : InnerStream} {bcs : Array ByteArray}
    (h : innerCommits tr record (some reg) bound shape = .ok cs) (hs : cs.stream? "stacked/rep0" = some s)
    (htop : s.tops[0]? = some bcs) :
    ∃ prog e, registration (some reg) bound = .ok (some (prog, some e)) ∧
      topEntryFields prog bcs e = .ok true ∧
      ∃ e' str, topEntry prog bcs = .ok e' ∧ Flock.canon e' = .ok str ∧ Flock.canon e = .ok str := by
  unfold innerCommits at h
  extract_lets msgs0 extra stackedS declaredR jpMain at h
  split at h
  · cases h
  simp (config := { zeta := false }) only [jpMain] at h
  obtain ⟨held, hheld, h⟩ := eok_bind h
  split at h
  rotate_left
  · cases h
  rename_i rounds _
  extract_lets regR jpRec at h
  split at h
  rotate_left
  · cases h
  obtain ⟨j, -, h⟩ := eok_bind h
  simp (config := { zeta := false }) only [jpRec] at h
  split at h
  rotate_left
  · cases h
  rename_i streams _
  obtain ⟨msgs, -, h⟩ := eok_bind h
  extract_lets msgs' names c3 recNames jpT at h
  split at h
  rotate_left
  · cases h
  rename_i t0 _
  have key : ∀ t', jpT t' = .ok cs → c3 = true ∧
      ∃ out, candidate3 tr rounds t' names msgs' regR shape = .ok out ∧ cs.streams = out := by
    intro t' hT
    simp (config := { zeta := false }) only [jpT] at hT
    extract_lets repNames reps jpC3 jpShape jpLoop at hT
    split at hT
    · cases hT
    obtain ⟨u, -, hT⟩ := eok_bind hT
    have hS : jpShape () = .ok cs := by
      split at hT
      · cases hT
      · cases hT
      · split at hT
        · cases hT
        · exact hT
      · exact hT
    simp (config := { zeta := false }) only [jpShape] at hS
    split at hS
    · cases hS
    simp (config := { zeta := false }) only [jpC3] at hS
    split at hS
    · rename_i hc3
      obtain ⟨out, hout, hS⟩ := eok_bind hS
      have := Except.ok.inj hS
      subst this
      exact ⟨hc3, out, hout, rfl⟩
    · have := Except.ok.inj hS
      subst this
      simp [InnerCommits.stream?] at hs
  split at h
  · split at h
    · rename_i t' ht'
      obtain ⟨w, hw, h⟩ := eok_bind h
      have : t' = w := Except.ok.inj hw
      subst this
      obtain ⟨-, out, hout, hstreams⟩ := key t' h
      obtain ⟨n, hn, rfl⟩ := Option.map_eq_some_iff.1 ht'
      have hx := List.find?_some hn
      have hnS : n.startsWith "stacked/" = false := by
        simp only [extra, Bool.not_eq_true', Bool.or_eq_false_iff] at hx
        exact hx.2
      have hs' : out.find? (·.name == "stacked/rep0") = some s := hstreams ▸ hs
      have hq := Array.find?_some hs'
      obtain ⟨prog, e, hreg, hfields, e', str, h1, h2, h3⟩ := candidate3_top (tbl_ne_stacked hnS) hout s
        (Array.mem_of_find?_eq_some hs') (beq_iff_eq.1 hq) bcs htop
      refine ⟨prog, e, ?_, hfields, e', str, h1, h2, h3⟩
      rw [hheld]
      rcases held with _ | ⟨p, _ | e0⟩
      · simp [regR] at hreg
      · simp [regR] at hreg
      · simp only [regR, Option.bind_some, Option.map_some, Option.some.injEq, Prod.mk.injEq] at hreg
        obtain ⟨rfl, rfl⟩ := hreg
        rfl
    · cases h
  · split at h
    · cases h
    · obtain ⟨_, -, h⟩ := eok_bind h
      exact absurd (key _ h).1 ‹_›

end Session

end FlockC3.RegLevels.TopHold

#print axioms FlockC3.RegLevels.TopHold.candidate3_top
#print axioms FlockC3.RegLevels.TopHold.innerCommits_top
