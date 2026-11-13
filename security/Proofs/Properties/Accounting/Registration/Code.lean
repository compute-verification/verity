module

public import Properties.Accounting.Registration.Code
public import Proofs.Properties.Core.Site
public import Proofs.Accounting.Registration.Sound

/-!
# The registration log's check is the spec's: the proofs

The code's check folds a `State`. After a prefix of the log, the State's Site, items, declarations, head and prefix
heads are the spec's (`Inv`). Each entry's refusal in code is then the spec's refusal of it against the prefix
(`refusal_eq`): a site entry's by `Properties.Site.refuses`, read at the spec's digests (`entry_refusal_sheet`, since a
site entry's checks read only sheet digests), an information entry's and a declaration's check for check. The check of
the rest then follows the spec one entry at a time (`go_says`), by
`Verity.Registration.valid_append_singleton_iff_refusal`.

A prefix's view read under the whole log's names is its own (`view_take_names`): the names agree on every entry of
the prefix (`names_take`), and a view reads its digests only on its log's entries (`view_congr`).

An accepted log is chained (`check_chained`): each entry passed `previous` against the head of the log before it. So
two chained logs with one head share their last entry, which names one head before it, or differ there
(`search_binds`, by induction from the heads back), and two prefixes of accepted logs with one head are one prefix or a
found collision (`headBinds`); the views a head names are then one (`view_binds`). The view's JSON carries its heads
as they are (`publishesHeads`, by unfolding).
-/

@[expose] public section

namespace Properties.Registration

open Properties.Site (regEntry composite auditor)

/-- The spec's refusal of a code refusal. -/
def refusal : Core.Registration.Refusal → Verity.Registration.Refusal
  | .site r => Properties.Site.refusal r
  | .duplicateInformation => .duplicateInformation
  | .duplicateDeclaration => .duplicateDeclaration

/-- The check's step: an entry refused against the State, or the State with it. -/
def step (A : Core.Site.Registration.Auditor) (c : Core.Site.Registration.Composite) (s : Core.Registration.State)
    (e : Core.Registration.Entry) : Except (Nat × Core.Registration.Refusal) Core.Registration.State :=
  match e.refusal A c s with
  | some r => .error (s.entries, r)
  | none => .ok (s.add e)

theorem check_eq (A : Core.Site.Registration.Auditor) (c : Core.Site.Registration.Composite)
    (l : Core.Registration.Log) :
    Core.Registration.check A c l = l.foldlM (step A c) (Core.Registration.State.empty c) :=
  rfl

theorem nodup_iff : ∀ l : List String, Core.Registration.nodup l = true ↔ l.Nodup
  | [] => by simp [Core.Registration.nodup]
  | x :: xs => by simp [Core.Registration.nodup, nodup_iff xs]

variable (B : String → Verity.Compiler) (E : Verity.Registration.Entry → String)

theorem key_item (i : Core.Registration.Item) : (item B i).key = i.key := by cases i <;> rfl

/-- A site entry's refusal reads the digests only through the sheets'. -/
theorem entry_refusal_sheet {D₁ D₂ : Verity.Registration.Digests} (h : D₁.sheet = D₂.sheet)
    (A : Verity.Registration.Auditor) (c : Verity.Registration.Composite) (S : Verity.Site) (head : String)
    (e : Verity.Registration.Entry) : e.refusal D₁ A c S head = e.refusal D₂ A c S head := by
  obtain ⟨s₁, e₁, g₁⟩ := D₁
  obtain ⟨s₂, e₂, g₂⟩ := D₂
  dsimp only at h
  subst h
  cases e <;> rfl

/-- **A valid log registers each item once.** -/
theorem valid_items_nodup {D : Verity.Registration.Digests} {A : Verity.Registration.Auditor}
    {c : Verity.Registration.Composite} {l : Verity.Registration} (h : Verity.Registration.Valid D A c l) :
    ((Verity.Registration.items l).map Verity.Registration.Item.key).Nodup := by
  induction l using List.reverseRecOn with
  | nil => simp [Verity.Registration.items]
  | append_singleton l e _ =>
    have ho : e.ItemsOnce l := (Verity.Registration.valid_append_singleton.mp h).2.2.1
    simpa [Verity.Registration.Entry.ItemsOnce, Verity.Registration.items, List.flatMap_append] using ho

/-- What the check's State after the prefix `p` holds of it: the spec's Site, items, declarations, head, length and
prefix heads. -/
def Inv (c : Core.Site.Registration.Composite) (p : Core.Registration.Log) (s : Core.Registration.State) : Prop :=
  Properties.Site.site s.site = Verity.Site.ofLog (log B p) ∧
    Verity.Registration.items (log B p) = s.items.map (item B) ∧
    (log B p).filterMap Verity.Registration.Entry.declared? = s.declared.map declared ∧
    Verity.Registration.head (digests E) (composite c) (log B p) = s.head ∧
    s.entries = p.length ∧ Verity.Registration.heads (digests E) (composite c) (log B p) = s.heads

theorem log_append (p q : Core.Registration.Log) : log B (p ++ q) = log B p ++ log B q := List.map_append

/-- The heads of a log with one more entry: the log's, then the entry's digest at the longer length. -/
theorem heads_concat (D : Verity.Registration.Digests) (c : Verity.Registration.Composite) (L : Verity.Registration)
    (x : Verity.Registration.Entry) :
    Verity.Registration.heads D c (L ++ [x]) = Verity.Registration.heads D c L ++ [(L.length + 1, D.entry x)] := by
  unfold Verity.Registration.heads
  rw [List.length_append, List.length_singleton, List.range_succ, List.map_append]
  congr 1
  · apply List.map_congr_left
    intro n hn
    rw [List.take_append_of_le_length (by simpa [List.mem_range, Nat.lt_succ_iff] using hn)]
  · simp [List.take_of_length_le]

theorem inv_empty (c : Core.Site.Registration.Composite) : Inv B E c [] (Core.Registration.State.empty c) := by
  refine ⟨rfl, rfl, rfl, ?_, rfl, ?_⟩
  · simp [log, digests, Core.Registration.State.empty, Properties.Site.ofComposite_composite]
  · simp [Verity.Registration.heads, log, digests, Core.Registration.State.empty,
      Properties.Site.ofComposite_composite]

theorem inv_add (c : Core.Site.Registration.Composite) {p : Core.Registration.Log} {s : Core.Registration.State}
    (e : Core.Registration.Entry) (hN : E (entry B e) = e.digest) (hi : Inv B E c p s) :
    Inv B E c (p ++ [e]) (s.add e) := by
  obtain ⟨hS, hI, hD, hH, hL, hHs⟩ := hi
  rw [Inv, log_append]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [show log B [e] = [entry B e] from rfl, Verity.Site.ofLog_concat, ← hS]
    cases e with
    | site x =>
      simp only [Core.Registration.State.add, entry]
      rw [Properties.Site.site_add, Properties.Site.regEntry_added]
    | information p items =>
      simp [Core.Registration.State.add, entry, Verity.Registration.Entry.added, Verity.Site.add, Properties.Site.site]
    | declaration p d =>
      simp [Core.Registration.State.add, entry, Verity.Registration.Entry.added, Verity.Site.add, Properties.Site.site]
  · rw [show log B [e] = [entry B e] from rfl, Verity.Registration.items, List.flatMap_append,
      ← Verity.Registration.items, hI]
    cases e with
    | site x => cases x; simp [Core.Registration.State.add, entry, regEntry, Verity.Registration.Entry.items]
    | information p items =>
      simp [Core.Registration.State.add, entry, Verity.Registration.Entry.items]
    | declaration p d => simp [Core.Registration.State.add, entry, Verity.Registration.Entry.items]
  · rw [show log B [e] = [entry B e] from rfl, List.filterMap_append, hD]
    cases e with
    | site x => cases x; simp [Core.Registration.State.add, entry, regEntry, Verity.Registration.Entry.declared?]
    | information p items =>
      simp [Core.Registration.State.add, entry, Verity.Registration.Entry.declared?]
    | declaration p d => simp [Core.Registration.State.add, entry, Verity.Registration.Entry.declared?]
  · rw [show log B [e] = [entry B e] from rfl, Verity.Registration.head_concat]
    cases e <;> exact hN
  · cases e <;> simp [Core.Registration.State.add, hL]
  · rw [show log B [e] = [entry B e] from rfl, heads_concat, hHs]
    have hlen : (log B p).length = s.entries := by simp [log, hL]
    rw [hlen, show (digests E).entry (entry B e) = e.digest from hN]
    cases e <;> rfl

/-- **An entry's refusal in code is the spec's** against the prefix the State holds, when that prefix is valid. -/
theorem refusal_eq (A : Core.Site.Registration.Auditor) (c : Core.Site.Registration.Composite)
    {p : Core.Registration.Log} {s : Core.Registration.State} (hi : Inv B E c p s)
    (hv : Verity.Registration.Valid (digests E) (auditor A) (composite c) (log B p)) (e : Core.Registration.Entry) :
    (e.refusal A c s).map refusal =
      Verity.Registration.refusal (digests E) (auditor A) (composite c) (log B p) (entry B e) := by
  obtain ⟨hS, hI, hD, hH, -, -⟩ := hi
  have hn := valid_items_nodup hv
  rw [hI] at hn
  have hdays : Verity.Registration.days (log B p) = s.declared.map (·.day) := by
    rw [Verity.Registration.days, ← List.map_filterMap, hD, List.map_map]
    rfl
  cases e with
  | site x =>
    have hr := Properties.Site.refuses Core.Site.Registration.sha512 A c s.site s.head x
    have hf : refusal ∘ Core.Registration.Refusal.site = Properties.Site.refusal := rfl
    rw [Core.Registration.Entry.refusal, Option.map_map, hf, hr,
      entry_refusal_sheet (D₁ := Properties.Site.digests Core.Site.Registration.sha512) (D₂ := digests E) rfl, hS,
      ← hH, Verity.Registration.refusal]
    have hn' : (s.items.map (Verity.Registration.Item.key ∘ item B)).Nodup := by simpa using hn
    cases x
    simp [entry, regEntry, Verity.Registration.Entry.ItemsOnce, Verity.Registration.Entry.DayOnce,
      Verity.Registration.Entry.items, Verity.Registration.Entry.declared?, hI, hn']
  | information q items =>
    have hk : ((s.items.map (item B) ++ items.map (item B)).map Verity.Registration.Item.key) =
        (s.items ++ items).map Core.Registration.Item.key := by
      simp [List.map_map, Function.comp_def, key_item]
    simp only [Core.Registration.Entry.refusal, Verity.Registration.refusal, Verity.Registration.Entry.refusal,
      entry, Verity.Registration.Entry.Follows, Verity.Registration.Entry.previous,
      Verity.Registration.Entry.kindRefusal, Verity.Registration.Entry.ItemsOnce, Verity.Registration.Entry.DayOnce,
      Verity.Registration.Entry.items, Verity.Registration.Entry.declared?, hH, hI, hk, ← nodup_iff]
    by_cases hq : q = s.head
    · cases h : Core.Registration.nodup ((s.items ++ items).map Core.Registration.Item.key) <;>
        simp [hq, refusal]
    · simp [hq, refusal, Properties.Site.refusal]
  | declaration q d =>
    simp only [Core.Registration.Entry.refusal, Verity.Registration.refusal, Verity.Registration.Entry.refusal,
      entry, Verity.Registration.Entry.Follows, Verity.Registration.Entry.previous,
      Verity.Registration.Entry.kindRefusal, Verity.Registration.Entry.ItemsOnce, Verity.Registration.Entry.DayOnce,
      Verity.Registration.Entry.items, Verity.Registration.Entry.declared?, hH, hI, hdays, List.append_nil, hn]
    by_cases hq : q = s.head
    · by_cases hd : d.day ∈ s.declared.map (·.day)
      · have ha : s.declared.any (·.day == d.day) = true := by
          obtain ⟨x, hx, hxd⟩ := List.mem_map.mp hd
          exact List.any_eq_true.mpr ⟨x, hx, by simpa using hxd⟩
        simp [hq, ha, refusal, declared]
        exact List.mem_map.mp hd
      · have ha : s.declared.any (·.day == d.day) = false := by
          rw [Bool.eq_false_iff]
          intro ha
          obtain ⟨x, hx, hxd⟩ := List.any_eq_true.mp ha
          exact hd (List.mem_map.mpr ⟨x, hx, by simpa using hxd⟩)
        simp [hq, ha, declared]
        intro x hx hxd
        exact hd (List.mem_map.mpr ⟨x, hx, hxd⟩)
    · simp [hq, refusal, Properties.Site.refusal]

/-- What a check's result says of the log it checked: an accepted log is valid, with the State the spec's of it; a
refused one is invalid. -/
def Says (A : Core.Site.Registration.Auditor) (c : Core.Site.Registration.Composite) (l : Core.Registration.Log) :
    Except (Nat × Core.Registration.Refusal) Core.Registration.State → Prop
  | .ok s => Verity.Registration.Valid (digests E) (auditor A) (composite c) (log B l) ∧ Inv B E c l s
  | .error _ => ¬Verity.Registration.Valid (digests E) (auditor A) (composite c) (log B l)

/-- **The check follows the spec one entry at a time**: from a valid prefix `p` and its State, the check of the rest
says what the spec says of the whole log. -/
theorem go_says (A : Core.Site.Registration.Auditor) (c : Core.Site.Registration.Composite) :
    ∀ (rest p : Core.Registration.Log) (s : Core.Registration.State),
      (∀ q e t, q ++ [e] ++ t = p ++ rest →
        Verity.Registration.Valid (digests E) (auditor A) (composite c) (log B (q ++ [e])) →
        E (entry B e) = e.digest) → Inv B E c p s →
      Verity.Registration.Valid (digests E) (auditor A) (composite c) (log B p) →
      Says B E A c (p ++ rest) (rest.foldlM (step A c) s) := by
  intro rest
  induction rest with
  | nil =>
    intro p s _ hi hv
    simpa [Says] using ⟨hv, hi⟩
  | cons e rest ih =>
    intro p s hN hi hv
    have hr := refusal_eq B E A c hi hv e
    rw [List.foldlM_cons]
    cases h : e.refusal A c s with
    | some r =>
      simp only [step, h]
      show ¬_
      intro hall
      have hpre : Verity.Registration.Valid (digests E) (auditor A) (composite c) (log B p ++ [entry B e]) :=
        Verity.Registration.Valid.of_prefix ⟨log B rest, by simp [log]⟩ hall
      rw [Verity.Registration.valid_append_singleton_iff_refusal, ← hr, h] at hpre
      simp at hpre
    | none =>
      simp only [step, h]
      show Says B E A c (p ++ e :: rest) (rest.foldlM (step A c) (s.add e))
      have hv' : Verity.Registration.Valid (digests E) (auditor A) (composite c) (log B (p ++ [e])) := by
        rw [log_append]
        exact Verity.Registration.valid_append_singleton_iff_refusal.mpr ⟨hv, by rw [← hr, h]; rfl⟩
      have hNe : E (entry B e) = e.digest := hN p e rest (by simp) hv'
      have := ih (p ++ [e]) (s.add e) (fun q e' t h => hN q e' t (by simpa using h)) (inv_add B E c e hNe hi) hv'
      simpa using this

theorem view_eq (c : Core.Site.Registration.Composite) {l : Core.Registration.Log} {s : Core.Registration.State}
    (hi : Inv B E c l s) : view B s.view = Verity.Registration.view (digests E) (composite c) (log B l) := by
  obtain ⟨hS, hI, hD, hH, -, hHs⟩ := hi
  unfold view Verity.Registration.view Core.Registration.State.view Verity.Registration.theta
    Verity.Registration.builder?
  rw [hI, hS, hD, hH, hHs]
  congr 1
  · generalize s.items = is
    induction is with
    | nil => rfl
    | cons i is ih => cases i <;> simp [item, theta, ih]
  · generalize s.items = is
    induction is with
    | nil => rfl
    | cons i is ih => cases i <;> simp [item, builder, ih]

/-- **Two entries of a log denoting one spec entry are one**, when the log up to the later is valid: a declaration
would declare its day twice and an information entry with items would register them twice, and a site entry or an
information entry without items is all its denotation says. -/
theorem denotes_once {D : Verity.Registration.Digests} {A' : Verity.Registration.Auditor}
    {c' : Verity.Registration.Composite} {q : Core.Registration.Log} {e y : Core.Registration.Entry}
    (hv : Verity.Registration.Valid D A' c' (log B (q ++ [e]))) (hy : y ∈ q) (h : entry B y = entry B e) : y = e := by
  rw [log_append] at hv
  obtain ⟨-, -, hI, hD⟩ := Verity.Registration.valid_append_singleton.mp hv
  have hmem : entry B e ∈ log B q := h ▸ List.mem_map_of_mem hy
  cases e with
  | site x =>
    cases y with
    | site z =>
      have := congrArg Properties.Site.ofRegEntry (show regEntry z = regEntry x from h)
      rw [Properties.Site.ofRegEntry_regEntry, Properties.Site.ofRegEntry_regEntry] at this
      rw [this]
    | information _ _ => cases x; simp [entry, regEntry] at h
    | declaration _ _ => cases x; simp [entry, regEntry] at h
  | information p is =>
    cases y with
    | site z => cases z; simp [entry, regEntry] at h
    | declaration _ _ => simp [entry] at h
    | information p' is' =>
      simp only [entry, Verity.Registration.Entry.information.injEq] at h
      obtain ⟨rfl, hm⟩ := h
      cases is with
      | nil => simp at hm; rw [hm]
      | cons i is =>
        exfalso
        have hin : item B i ∈ Verity.Registration.items (log B q) :=
          List.mem_flatMap.mpr ⟨_, hmem, by simp [entry, Verity.Registration.Entry.items]⟩
        unfold Verity.Registration.Entry.ItemsOnce at hI
        rw [List.map_append] at hI
        have hie : item B i ∈ (entry B (.information p' (i :: is))).items := by
          simp [entry, Verity.Registration.Entry.items]
        exact (List.nodup_append.mp hI).2.2 _ (List.mem_map_of_mem hin) _ (List.mem_map_of_mem hie) rfl
  | declaration p d =>
    exfalso
    have hday : (declared d).day ∈ Verity.Registration.days (log B q) :=
      List.mem_filterMap.mpr ⟨_, hmem, by simp [entry, Verity.Registration.Entry.declared?]⟩
    exact hD _ (by simp [entry, Verity.Registration.Entry.declared?]) hday

/-- **`names` names each accepted entry as the code does**: an entry after which the log is valid is named by its own
digest. -/
theorem names_accepted {D : Verity.Registration.Digests} {A' : Verity.Registration.Auditor}
    {c' : Verity.Registration.Composite} (q t : Core.Registration.Log) (e : Core.Registration.Entry)
    (hv : Verity.Registration.Valid D A' c' (log B (q ++ [e]))) :
    names B (q ++ [e] ++ t) (entry B e) = e.digest := by
  classical
  unfold names
  rw [List.find?_append, List.find?_append]
  cases hq : q.find? (fun x => decide (entry B x = entry B e)) with
  | some y =>
    have hy := List.mem_of_find?_eq_some hq
    have hp := List.find?_some hq
    rw [denotes_once B hv hy (of_decide_eq_true hp)]
    rfl
  | none => simp

/-- What the check says of a log, its entries named as the code names them. -/
theorem says_check (A : Core.Site.Registration.Auditor) (c : Core.Site.Registration.Composite)
    (l : Core.Registration.Log) : Says B (names B l) A c l (Core.Registration.check A c l) := by
  have hs := go_says B (names B l) A c l [] (Core.Registration.State.empty c)
    (fun q e t h hv => by rw [List.nil_append] at h; rw [← h]; exact names_accepted B q t e hv)
    (inv_empty B (names B l) c) (by simp [log])
  rw [check_eq]
  simpa only [List.nil_append] using hs

theorem checks : Checks := by
  intro B A c l _
  have hs := says_check B A c l
  revert hs
  cases Core.Registration.check A c l with
  | ok s =>
    rintro ⟨hv, hi⟩
    refine ⟨by simp [Except.toOption, hv], fun s' hs' => ?_⟩
    cases hs'
    exact view_eq B (names B l) c hi
  | error ir =>
    intro hnv
    have hnv' : ¬Verity.Registration.Valid (digests (names B l)) (auditor A) (composite c) (log B l) := hnv
    exact ⟨by simp [Except.toOption, hnv'], fun s' hs' => by cases hs'⟩

/-! ## A prefix's view under the whole log's names -/

/-- **The whole log names each entry of a prefix as the prefix does**: the first entry denoting it is in the prefix. -/
theorem names_take (l : Core.Registration.Log) (n : ℕ) {x : Verity.Registration.Entry} (hx : x ∈ log B (l.take n)) :
    names B l x = names B (l.take n) x := by
  classical
  unfold names
  obtain ⟨y, hy, rfl⟩ := List.mem_map.mp hx
  conv_lhs => rw [← List.take_append_drop n l]
  rw [List.find?_append]
  cases hq : (l.take n).find? (fun z => decide (entry B z = entry B y)) with
  | some z => rfl
  | none =>
    exfalso
    have := List.find?_eq_none.mp hq y hy
    simp at this

/-- **A view reads its digests only at its log's entries and the genesis.** -/
theorem view_congr {D D' : Verity.Registration.Digests} (hg : D.genesis = D'.genesis) (c : Verity.Registration.Composite)
    {L : Verity.Registration} (he : ∀ x ∈ L, D.entry x = D'.entry x) :
    Verity.Registration.view D c L = Verity.Registration.view D' c L := by
  have hh : ∀ k, Verity.Registration.head D c (L.take k) = Verity.Registration.head D' c (L.take k) := by
    intro k
    rcases (L.take k).eq_nil_or_concat with hk | ⟨q, x, hk⟩
    · rw [hk, Verity.Registration.head_nil, Verity.Registration.head_nil, hg]
    · rw [hk, Verity.Registration.head_concat, Verity.Registration.head_concat]
      exact he x (List.mem_of_mem_take (by rw [hk]; simp))
  have h0 : Verity.Registration.head D c L = Verity.Registration.head D' c L := by simpa using hh L.length
  simp only [Verity.Registration.view, Verity.Registration.heads, h0, hh]

/-- **A prefix's view read under the whole log's names is its own**: its Site, θ, builder, declarations, head and prefix
heads, so what an answer reads at a prefix's head does not depend on the entries after it. -/
theorem view_take_names (c : Core.Site.Registration.Composite) (l : Core.Registration.Log) (n : ℕ) :
    Verity.Registration.view (digests (names B l)) (composite c) (log B (l.take n)) =
      Verity.Registration.view (digests (names B (l.take n))) (composite c) (log B (l.take n)) :=
  view_congr rfl _ fun _ hx => names_take B l n hx

theorem publishesHeads : PublishesHeads := fun _ _ => rfl

/-! ## A head binds the prefix it heads -/

/-- **A code log is chained**: each entry names the head of the log before it. -/
def Chained (c : Core.Site.Registration.Composite) (l : Core.Registration.Log) : Prop :=
  ∀ q e t, q ++ e :: t = l → e.previous = Core.Registration.head c q

theorem head_nil_code (c : Core.Site.Registration.Composite) :
    Core.Registration.head c [] = Core.Registration.genesis c := rfl

theorem head_concat_code (c : Core.Site.Registration.Composite) (q : Core.Registration.Log)
    (e : Core.Registration.Entry) : Core.Registration.head c (q ++ [e]) = e.digest := by
  simp [Core.Registration.head]

theorem state_head_add (s : Core.Registration.State) (e : Core.Registration.Entry) : (s.add e).head = e.digest := by
  cases e <;> rfl

/-- An entry the check doesn't refuse names the head of the log before it. -/
theorem previous_of_refusal_none {A : Core.Site.Registration.Auditor} {c : Core.Site.Registration.Composite}
    {s : Core.Registration.State} {e : Core.Registration.Entry} (h : e.refusal A c s = none) : e.previous = s.head := by
  by_contra hne
  cases e with
  | site x =>
    simp only [Core.Registration.Entry.previous] at hne
    simp [Core.Registration.Entry.refusal, Core.Site.Registration.Entry.refusal, hne] at h
  | information p items =>
    simp only [Core.Registration.Entry.previous] at hne
    simp [Core.Registration.Entry.refusal, hne] at h
  | declaration p d =>
    simp only [Core.Registration.Entry.previous] at hne
    simp [Core.Registration.Entry.refusal, hne] at h

theorem chained_concat {c : Core.Site.Registration.Composite} {q : Core.Registration.Log} {e : Core.Registration.Entry}
    (hq : Chained c q) (he : e.previous = Core.Registration.head c q) : Chained c (q ++ [e]) := by
  intro q' e' t h
  rcases t.eq_nil_or_concat with rfl | ⟨t', x, rfl⟩
  · obtain ⟨rfl, h2⟩ := List.append_inj' h rfl
    obtain rfl : e' = e := List.singleton_inj.mp h2
    exact he
  · have h' : (q' ++ e' :: t') ++ [x] = q ++ [e] := by rw [← h]; simp
    exact hq q' e' t' (List.append_inj' h' rfl).1

theorem chained_last {c : Core.Site.Registration.Composite} {q : Core.Registration.Log} {e : Core.Registration.Entry}
    (h : Chained c (q ++ [e])) : e.previous = Core.Registration.head c q ∧ Chained c q :=
  ⟨h q e [] rfl, fun q' e' t hq => h q' e' (t ++ [e]) (by rw [← hq]; simp)⟩

theorem chained_take {c : Core.Site.Registration.Composite} {l : Core.Registration.Log} (h : Chained c l) (n : ℕ) :
    Chained c (l.take n) := fun q e t hq =>
  h q e (t ++ l.drop n) (calc
    q ++ e :: (t ++ l.drop n) = (q ++ e :: t) ++ l.drop n := by simp
    _ = l.take n ++ l.drop n := by rw [hq]
    _ = l := List.take_append_drop n l)

theorem fold_chained (A : Core.Site.Registration.Auditor) (c : Core.Site.Registration.Composite) :
    ∀ (l q : Core.Registration.Log) (s s' : Core.Registration.State), Chained c q →
      s.head = Core.Registration.head c q → l.foldlM (step A c) s = .ok s' → Chained c (q ++ l)
  | [], q, _, _, hq, _, _ => by simpa using hq
  | e :: l, q, s, s', hq, hh, hf => by
    rw [List.foldlM_cons] at hf
    cases hr : e.refusal A c s with
    | some r =>
      simp only [step, hr] at hf
      cases hf
    | none =>
      simp only [step, hr] at hf
      have hq' : Chained c (q ++ [e]) := chained_concat hq ((previous_of_refusal_none hr).trans hh)
      have := fold_chained A c l (q ++ [e]) (s.add e) s' hq' (by rw [state_head_add, head_concat_code]) hf
      simpa using this

/-- **An accepted log is chained.** -/
theorem check_chained {A : Core.Site.Registration.Auditor} {c : Core.Site.Registration.Composite}
    {l : Core.Registration.Log} {s : Core.Registration.State} (h : Core.Registration.check A c l = .ok s) :
    Chained c l := by
  have := fold_chained A c l [] (Core.Registration.State.empty c) s (fun q e t h => by simp at h) rfl
    (by rw [← check_eq]; exact h)
  simpa using this

theorem search_nil_nil (c : Core.Site.Registration.Composite) : search c [] [] = none := rfl

theorem search_cons_nil (c : Core.Site.Registration.Composite) (e : Core.Registration.Entry)
    (r : Core.Registration.Log) : search c (e :: r) [] = some (.genesis c e) := rfl

theorem search_nil_cons (c : Core.Site.Registration.Composite) (e : Core.Registration.Entry)
    (r : Core.Registration.Log) : search c [] (e :: r) = some (.genesis c e) := rfl

theorem search_cons_self (c : Core.Site.Registration.Composite) (e : Core.Registration.Entry)
    (r r' : Core.Registration.Log) : search c (e :: r) (e :: r') = search c r r' := by
  simp [search]

theorem search_cons_ne (c : Core.Site.Registration.Composite) {e e' : Core.Registration.Entry} (h : e ≠ e')
    (r r' : Core.Registration.Log) : search c (e :: r) (e' :: r') = some (.entries e e') := by
  simp [search, h]

/-- **Two chained logs with one head are one, or the search finds a collision**, the logs given newest entry first: one
head is one last entry's digest, or the genesis; one last entry names one head before it. -/
theorem search_binds (c : Core.Site.Registration.Composite) :
    ∀ r r' : Core.Registration.Log, Chained c r.reverse → Chained c r'.reverse →
      Core.Registration.head c r.reverse = Core.Registration.head c r'.reverse →
      Resolves (search c r r') r.reverse r'.reverse
  | [], [], _, _, _ => by rw [search_nil_nil]; rfl
  | e :: r, [], _, _, hh => by
    rw [search_cons_nil]
    rw [List.reverse_cons, List.reverse_nil, head_concat_code, head_nil_code] at hh
    exact hh
  | [], e :: r, _, _, hh => by
    rw [search_nil_cons]
    rw [List.reverse_cons, List.reverse_nil, head_concat_code, head_nil_code] at hh
    exact hh.symm
  | e :: r, e' :: r', hc, hc', hh => by
    rw [List.reverse_cons] at hc hc' hh
    rw [List.reverse_cons, head_concat_code, head_concat_code] at hh
    by_cases he : e = e'
    · subst he
      rw [search_cons_self]
      obtain ⟨hp, hr⟩ := chained_last hc
      obtain ⟨hp', hr'⟩ := chained_last hc'
      have ih := search_binds c r r' hr hr' (hp.symm.trans hp')
      revert ih
      cases search c r r' with
      | none =>
        intro ih
        change r.reverse = r'.reverse at ih
        change (e :: r).reverse = (e :: r').reverse
        rw [List.reverse_cons, List.reverse_cons, ih]
      | some x => exact id
    · rw [search_cons_ne c he]
      exact ⟨he, hh⟩

theorem headBinds : HeadBinds := by
  intro A A' c l l' s s' hl hl' n n' hh
  have h := search_binds c (l.take n).reverse (l'.take n').reverse
  rw [List.reverse_reverse, List.reverse_reverse] at h
  exact h (chained_take (check_chained hl) n) (chained_take (check_chained hl') n') hh

/-- **The views a head names are one**: at two accepted logs, the views of two prefixes with one head, each read under
its own log's names, are one view, or the search of the two prefixes' entries finds a pair that collides. So a tie read
at a head (`Verity.Registration.AtHead`) reads what the prefix that head names registers, whichever accepted log holds
it. -/
theorem view_binds {A A' : Core.Site.Registration.Auditor} {c : Core.Site.Registration.Composite}
    {l l' : Core.Registration.Log} {s s' : Core.Registration.State} (hl : Core.Registration.check A c l = .ok s)
    (hl' : Core.Registration.check A' c l' = .ok s') (n n' : ℕ)
    (hh : Core.Registration.head c (l.take n) = Core.Registration.head c (l'.take n')) :
    Verity.Registration.view (digests (names B l)) (composite c) (log B (l.take n)) =
        Verity.Registration.view (digests (names B l')) (composite c) (log B (l'.take n')) ∨
      ∃ x, search c (l.take n).reverse (l'.take n').reverse = some x ∧ x.Holds := by
  have h := headBinds A A' c l l' s s' hl hl' n n' hh
  cases hs : search c (l.take n).reverse (l'.take n').reverse with
  | none =>
    rw [hs] at h
    have he : l.take n = l'.take n' := h
    left
    rw [view_take_names B c l n, view_take_names B c l' n', he]
  | some x =>
    rw [hs] at h
    exact Or.inr ⟨x, rfl, h⟩

/-! ## Witnesses

The guarantee holds at `honestLog`, the code's example site as an onsite site entry (a device, its memory leaf, a
certified uplink and a memory challenger), an information entry registering a value and a builder, and a declaration,
which the check accepts at the code's example auditor, whose denotation is valid and whose builder's compiler has its
cap; and its conclusion fails at the check that accepts everything, at a log that registers one value twice. -/

open Core.Registration.Example in
/-- The example site passes every check of a site entry onsite, at the deployed digests: its device names its sheet's
digest, its memory fits the sheet, its figures are the auditor's within its tolerance, its certifier splits the uplink and
its challenger reaches the host. -/
theorem example_refusal :
    Core.Site.Entry.refusal Core.Site.Registration.sha512 auditor .onsite ⟨[], [], []⟩ ⟨[sheet], locs, links⟩ = none := by
  have hFresh : Core.Site.Entry.Fresh Core.Site.Registration.sha512 ⟨[], [], []⟩ ⟨[sheet], locs, links⟩ :=
    ⟨by decide +kernel, by simp [Core.Site.add]⟩
  have hRefs : Core.Site.Entry.Refs ⟨[], [], []⟩ ⟨[sheet], locs, links⟩ := by decide +kernel
  have hKinds : Core.Site.Entry.Kinds ⟨[], [], []⟩ ⟨[sheet], locs, links⟩ := by decide +kernel
  have hSheets : Core.Site.Entry.Sheets Core.Site.Registration.sha512 auditor ⟨[], [], []⟩ ⟨[sheet], locs, links⟩ := by
    refine ⟨by decide +kernel, fun x hx hd => ?_⟩
    simp only [locs, List.mem_cons, List.mem_nil_iff, or_false] at hx
    rcases hx with rfl | rfl | rfl | rfl | rfl | rfl <;> (try simp at hd) <;> simp [Core.Site.add]
  have hFigures : Core.Site.Entry.Figures auditor ⟨[sheet], locs, links⟩ := by decide +kernel
  have hCapacity : Core.Site.Entry.Capacity Core.Site.Registration.sha512 ⟨[], [], []⟩ ⟨[sheet], locs, links⟩ := by
    refine ⟨by decide +kernel, fun x hx hd k hk => ?_⟩
    simp only [Core.Site.add, List.nil_append, locs, List.mem_cons, List.mem_nil_iff, or_false] at hx
    rcases hx with rfl | rfl | rfl | rfl | rfl | rfl <;> (try simp at hd)
    simp [Core.Site.sheet?, Core.Site.add] at hk
    subst hk
    decide +kernel
  have hOwners : Core.Site.Entry.Owners auditor ⟨[sheet], locs, links⟩ := by decide +kernel
  have hCovered : Core.Site.Entry.Covered .onsite ⟨[], [], []⟩ ⟨[sheet], locs, links⟩ := by decide +kernel
  have hReached : Core.Site.Entry.Reached .onsite ⟨[], [], []⟩ ⟨[sheet], locs, links⟩ := by decide +kernel
  unfold Core.Site.Entry.refusal
  simp only [hFresh, hRefs, hKinds, hSheets, hFigures, hCapacity, hOwners, hCovered, hReached, not_true_eq_false,
    if_false]

theorem siteEntry_refusal :
    Core.Registration.Entry.refusal Core.Registration.Example.auditor .onsite (Core.Registration.State.empty .onsite)
      siteEntry = none := by
  simp only [siteEntry, Core.Registration.Entry.refusal, Core.Registration.State.empty,
    Core.Site.Registration.Entry.refusal, Core.Site.Registration.Entry.kindRefusal, Core.Site.Registration.Entry.previous,
    ne_eq, not_true_eq_false, if_false, example_refusal, Option.map_none]

theorem check_honestLog :
    (Core.Registration.check Core.Registration.Example.auditor .onsite honestLog).toOption.isSome := by
  simp only [honestLog, Core.Registration.check, List.foldlM_cons, siteEntry_refusal]
  simp [siteEntry, infoEntry, dayEntry, Core.Registration.Entry.refusal,
    Core.Registration.State.empty, Core.Registration.State.add, Core.Registration.nodup, Core.Registration.Item.key,
    Except.toOption, bind, Except.bind, pure, Except.pure]

/-- The witness log's one builder names the compiler with no gates, at its cap 0. -/
theorem caps_honestLog : Caps noGatesBuilder honestLog := by
  intro p items h b hb
  simp [honestLog, siteEntry, infoEntry, dayEntry] at h
  obtain ⟨-, rfl⟩ := h
  simp at hb
  subst hb
  rfl

theorem checks_honest :
    (Core.Registration.check Core.Registration.Example.auditor .onsite honestLog).toOption.isSome = true ∧
      Verity.Registration.Valid (digests (names noGatesBuilder honestLog))
        (auditor Core.Registration.Example.auditor) (composite .onsite) (log noGatesBuilder honestLog) ∧
      Caps noGatesBuilder honestLog :=
  ⟨check_honestLog,
    (checks noGatesBuilder Core.Registration.Example.auditor .onsite honestLog caps_honestLog).1.mp check_honestLog,
    caps_honestLog⟩

theorem twiceLog_invalid (B : String → Verity.Compiler)
    (E : Verity.Registration.Entry → String) :
    ¬Verity.Registration.Valid (digests E) (auditor Properties.Site.witnessAuditor) (composite .remote)
      (log B twiceLog) := by
  intro h
  have := (h 0 (by simp [log, twiceLog])).2.1
  simp [log, twiceLog, entry, item, theta, Verity.Registration.Entry.ItemsOnce, Verity.Registration.items,
    Verity.Registration.Entry.items, Verity.Registration.Item.key] at this

theorem checks_nontrivial :
    ¬(((acceptAllCheck .remote twiceLog).toOption.isSome = true ↔
        Verity.Registration.Valid (digests (names noGatesBuilder twiceLog))
          (auditor Properties.Site.witnessAuditor) (composite .remote) (log noGatesBuilder twiceLog)) ∧
      ∀ s, acceptAllCheck .remote twiceLog = .ok s →
        view noGatesBuilder s.view = Verity.Registration.view (digests (names noGatesBuilder twiceLog))
          (composite .remote) (log noGatesBuilder twiceLog)) := by
  rintro ⟨h, -⟩
  exact twiceLog_invalid noGatesBuilder _ (h.mp rfl)

end Properties.Registration
