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

variable (B : Core.Registration.Builder → Verity.Compiler) (E : Verity.Registration.Entry → String)

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

theorem checks : Checks := by
  intro B A c l
  have hs := go_says B (names B l) A c l [] (Core.Registration.State.empty c)
    (fun q e t h hv => by rw [List.nil_append] at h; rw [← h]; exact names_accepted B q t e hv)
    (inv_empty B (names B l) c) (by simp [log])
  simp only [List.nil_append] at hs
  rw [check_eq]
  revert hs
  cases List.foldlM (step A c) (Core.Registration.State.empty c) l with
  | ok s => rintro ⟨hv, hi⟩; simp [Except.toOption, hv, view_eq B (names B l) c hi]
  | error ir =>
    intro hnv
    have hnv' : ¬Verity.Registration.Valid (digests (names B l)) (auditor A) (composite c) (log B l) := hnv
    simp [Except.toOption, hnv']

/-! ## Witnesses

The guarantee holds at `honestLog`, an information entry and a declaration for the remote composite, which the check
accepts and whose denotation is valid; and its conclusion fails at the check that accepts everything, at a log that
registers one value twice. -/

theorem check_honestLog :
    (Core.Registration.check Properties.Site.witnessAuditor .remote honestLog).toOption.isSome := by
  simp [honestLog, infoEntry, dayEntry, Core.Registration.check, Core.Registration.Entry.refusal,
    Core.Registration.State.empty, Core.Registration.State.add, Core.Registration.nodup, Core.Registration.Item.key,
    Except.toOption, bind, Except.bind, pure, Except.pure]

theorem checks_honest :
    (Core.Registration.check Properties.Site.witnessAuditor .remote honestLog).toOption.isSome = true ∧
      Verity.Registration.Valid (digests (names noGatesBuilder honestLog)) (auditor Properties.Site.witnessAuditor)
        (composite .remote) (log noGatesBuilder honestLog) := by
  refine ⟨check_honestLog, ?_⟩
  have h := checks noGatesBuilder Properties.Site.witnessAuditor .remote honestLog
  have hc := check_honestLog
  revert h hc
  cases Core.Registration.check Properties.Site.witnessAuditor .remote honestLog with
  | error _ => simp [Except.toOption]
  | ok s =>
    intro h _
    by_contra hv
    simp [Except.toOption, hv] at h

theorem twiceLog_invalid (B : Core.Registration.Builder → Verity.Compiler)
    (E : Verity.Registration.Entry → String) :
    ¬Verity.Registration.Valid (digests E) (auditor Properties.Site.witnessAuditor) (composite .remote)
      (log B twiceLog) := by
  intro h
  have := (h 0 (by simp [log, twiceLog])).2.1
  simp [log, twiceLog, entry, item, theta, Verity.Registration.Entry.ItemsOnce, Verity.Registration.items,
    Verity.Registration.Entry.items, Verity.Registration.Item.key] at this

theorem checks_nontrivial :
    ¬((acceptAllCheck .remote twiceLog).toOption.map (fun s => view noGatesBuilder s.view) =
      if Verity.Registration.Valid (digests (names noGatesBuilder twiceLog))
          (auditor Properties.Site.witnessAuditor) (composite .remote) (log noGatesBuilder twiceLog) then
        some (Verity.Registration.view (digests (names noGatesBuilder twiceLog)) (composite .remote)
          (log noGatesBuilder twiceLog))
      else none) := by
  simp [twiceLog_invalid, acceptAllCheck, Except.toOption]

end Properties.Registration
