import Batteries.Tactic.OpenPrivate
import Properties.Pouw.LiveSalt
import Std.Tactic.Do

/-!
# The challenger's ledger, for `Properties.Pouw.LiveSalt.LedgerShows`

`verifyLedger` folds the ledger's entries into its windows, one per `open`. `LedgerInv` is what the fold has established
once it has read a prefix of the entries: each window is an `open` of the prefix, named by its open's root; no two
windows share a root or a session; a window's `end` is in the prefix, in its session and after its open, and a v2 `end`
that records a leaf records one inside the window; every `open` and every `end` has its window; and a `reserve` is v2
and comes before every `open` of its session. Each kind of entry keeps it (`inv_open`, `inv_end`, `inv_keep`), so it
holds of the whole ledger (`verifyLedger_inv`).

`checked` parses the ledger's lines into the entries `LiveSalt.entries` reads (`checked_ok`); `auditedOpen` and
`pinnedSession` take the window whose root the caller names, closed by its v2 `end` (`auditedOpen_ok`,
`pinnedSession_ok`). So the ledger of a session pinned on root `R` shows its reserve, its one open on `R` and its end
(`ledgerShows_of`). `sessionNamed` returns the commitment whose root's hex the ledger's `end` recorded
(`sessionNamed_root`).

The challenger's functions are monadic. Each fact about one is a Hoare triple proved by `mvcgen`, and `of_triple` reads
it back at an `.ok` result.
-/

set_option mvcgen.warning false

open private strOf join jsonTrue from SampledProofs.ChallengerSession

namespace Properties.Pouw.LiveSalt.Ledger

open Lean (Json)
open Std.Do
open SampledProofs.Challenger

theorem bind_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} :
    (x >>= f) = .ok b ↔ ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x <;> simp [bind, Except.bind]

theorem forIn_zipIdx_ok {α σ ε : Type} (f : α × Nat → σ → Except ε (ForInStep σ)) (P : List α → σ → Prop)
    (hstep : ∀ pre x s t, P pre s → f (x, pre.length) s = .ok t → ∃ s', t = .yield s' ∧ P (pre ++ [x]) s') :
    ∀ (l pre : List α) (s r : σ), P pre s → forIn (l.zipIdx pre.length) s f = .ok r → P (pre ++ l) r := by
  intro l
  induction l with
  | nil =>
    intro pre s r hP h
    simp only [List.zipIdx_nil, List.forIn_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h; simpa using hP
  | cons x xs ih =>
    intro pre s r hP h
    rw [List.zipIdx_cons, List.forIn_cons] at h
    cases hf : f (x, pre.length) s with
    | error e => simp [hf, bind, Except.bind] at h
    | ok t =>
      obtain ⟨s', rfl, hP'⟩ := hstep pre x s t hP hf
      simp only [hf, bind, Except.bind] at h
      have := ih (pre ++ [x]) s' r hP' (by simpa [List.length_append] using h)
      simpa using this

/-- A loop over `xs.zipIdx` that only yields keeps `P` of the prefix it has read. -/
theorem forIn_array_zipIdx_ok {α σ ε : Type} (xs : Array α) (init : σ) (f : α × Nat → σ → Except ε (ForInStep σ))
    (P : List α → σ → Prop) (h0 : P [] init)
    (hstep : ∀ pre x s t, P pre s → f (x, pre.length) s = .ok t → ∃ s', t = .yield s' ∧ P (pre ++ [x]) s')
    {r : σ} (h : forIn xs.zipIdx init f = .ok r) : P xs.toList r := by
  rw [← Array.forIn_toList, Array.toList_zipIdx] at h
  simpa using forIn_zipIdx_ok f P hstep xs.toList [] init r h0 h

/-- What `verifyLedger` has established once it has read the entries `pre`, of its windows `wins` and its flag `v2`. -/
structure LedgerInv (pre : List LedgerEntry) (wins : Array Win) (v2 : Bool) : Prop where
  seq : ∀ x ∈ pre, x.seq < pre.length
  v2 : ∀ x ∈ pre, x.schema = LEDGER_SCHEMA → v2 = true
  win : ∀ i (h : i < wins.size), wins[i].opened ∈ pre ∧ wins[i].opened.kind = "open" ∧
    wins[i].root = strOf (openRoot wins[i].opened) ∧ wins[i].leaf = leafOf wins[i].opened.sig
  opens : ∀ o ∈ pre, o.kind = "open" → ∃ i, ∃ h : i < wins.size, wins[i].opened = o
  distinct : ∀ i j (hi : i < wins.size) (hj : j < wins.size), i ≠ j →
    wins[i].root ≠ wins[j].root ∧ wins[i].opened.session ≠ wins[j].opened.session
  ended : ∀ i (h : i < wins.size) e, wins[i].ended = some e → e ∈ pre ∧ e.kind = "end" ∧
    e.session = wins[i].opened.session ∧ wins[i].opened.seq < e.seq ∧
    (wins[i].opened.schema = LEDGER_SCHEMA → e.schema ≠ LEDGER_SCHEMA_V1) ∧
    (e.schema = LEDGER_SCHEMA → (SampledProofs.getD e.body "record_leaf").isNull = false →
      ∃ r, SampledProofs.nat? (SampledProofs.getD e.body "record_leaf") = some r ∧ wins[i].leaf < r ∧
        r < leafOf e.sig)
  ends : ∀ e ∈ pre, e.kind = "end" → ∃ i, ∃ h : i < wins.size, wins[i].opened.session = e.session ∧
    wins[i].ended = some e
  reserves : ∀ rv ∈ pre, rv.kind = "reserve" → rv.schema = LEDGER_SCHEMA ∧
    ∀ o ∈ pre, o.kind = "open" → o.session = rv.session → rv.seq < o.seq

/-- A triple whose postcondition is `Q` gives `Q` of an `.ok` result. -/
theorem of_triple {ε α : Type} {x : Except ε α} {Q : α → Prop} {r : α}
    (hT : ⦃⌜True⌝⦄ x ⦃⇓? a => ⌜Q a⌝⦄) (h : x = .ok r) : Q r := by
  subst h; exact hT True.intro

theorem except_spec {ε α : Type} (x : Except ε α) : ⦃⌜True⌝⦄ x ⦃⇓? _ => ⌜True⌝⦄ := by
  cases x <;> exact fun _ => True.intro

theorem throw_spec {ε α : Type} (e : ε) (Q : PostCond α (.except ε .pure)) :
    ⦃Q.2.1 e⦄ (MonadExceptOf.throw e : Except ε α) ⦃Q⦄ := fun h => h

theorem restart {ε α : Type} {x : Except ε α} {Q : PostCond α (.except ε .pure)} (h : ⦃⌜True⌝⦄ x ⦃Q⦄) :
    (wp⟦x⟧ Q).down := h True.intro

theorem wp_ite {ε α : Type} {c : Prop} [Decidable c] {x y : Except ε α} {Q : PostCond α (.except ε .pure)}
    (hx : c → (wp⟦x⟧ Q).down) (hy : ¬c → (wp⟦y⟧ Q).down) : (wp⟦if c then x else y⟧ Q).down := by
  by_cases hc : c
  · rw [ite_eq_left hc]; exact hx hc
  · rw [ite_eq_right hc]; exact hy hc

theorem ledgerMessage_spec (I : ByteArray) (m : Json) : ⦃⌜True⌝⦄ ledgerMessage I m ⦃⇓? _ => ⌜True⌝⦄ := except_spec _

theorem join_spec (ps : Array MonitoringPeriod) (cur : Array (String × String)) (i : Nat) (lane s : String) :
    ⦃⌜True⌝⦄ join ps cur i lane s ⦃⇓? _ => ⌜True⌝⦄ := except_spec _

theorem inv_nil : LedgerInv [] #[] false where
  seq := by simp
  v2 := by simp
  win := by simp
  opens := by simp
  distinct := by simp
  ended := by simp
  ends := by simp
  reserves := by simp

theorem chain {v2 : Bool} {e : LedgerEntry} {n : Nat} {prev : String}
    (h : (!(v2 && e.schema == LEDGER_SCHEMA_V1) && e.seq == n && e.prev == prev) = true) :
    e.seq = n ∧ (v2 = true → e.schema ≠ LEDGER_SCHEMA_V1) := by
  simp only [Bool.and_eq_true, Bool.not_eq_true', Bool.and_eq_false_iff, beq_iff_eq, beq_eq_false_iff_ne] at h
  obtain ⟨⟨h1, h2⟩, -⟩ := h
  refine ⟨h2, fun hv => ?_⟩
  rcases h1 with h1 | h1
  · rw [hv] at h1; cases h1
  · exact h1

theorem reserve_facts {e : LedgerEntry} {wins : Array Win} {reserves : Array Reserve}
    (h : ((e.schema == LEDGER_SCHEMA && isReservedSession e.session && !reserves.any fun x => x.session == e.session) &&
      !wins.any fun x => x.opened.session == e.session) = true) :
    e.schema = LEDGER_SCHEMA ∧ ∀ i (hi : i < wins.size), wins[i].opened.session ≠ e.session := by
  simp only [Bool.and_eq_true, Bool.not_eq_true', beq_iff_eq] at h
  obtain ⟨⟨⟨h1, -⟩, -⟩, h2⟩ := h
  refine ⟨h1, fun i hi he => ?_⟩
  have : (wins.any fun x => x.opened.session == e.session) = true := Array.any_eq_true.2 ⟨i, hi, by simp [he]⟩
  rw [this] at h2; cases h2

theorem mem_snoc {α : Type} {x e : α} {pre : List α} : x ∈ pre ++ [e] ↔ x ∈ pre ∨ x = e := by simp

theorem seq_snoc {pre : List LedgerEntry} {e : LedgerEntry} (hI : ∀ x ∈ pre, x.seq < pre.length)
    (hseq : e.seq = pre.length) : ∀ x ∈ pre ++ [e], x.seq < (pre ++ [e]).length := by
  intro x hx
  rw [List.length_append, List.length_singleton]
  rcases mem_snoc.1 hx with hx | rfl
  · have := hI x hx; omega
  · omega

theorem v2_snoc {pre : List LedgerEntry} {e : LedgerEntry} {v2 : Bool}
    (hI : ∀ x ∈ pre, x.schema = LEDGER_SCHEMA → v2 = true) :
    ∀ x ∈ pre ++ [e], x.schema = LEDGER_SCHEMA → (v2 || e.schema == LEDGER_SCHEMA) = true := by
  intro x hx hs
  rcases mem_snoc.1 hx with hx | rfl
  · simp [hI x hx hs]
  · simp [hs]

theorem kind_ne {e : LedgerEntry} {k k' : String} (h : e.kind = k) (hne : k ≠ k') : e.kind ≠ k' :=
  fun h' => hne (h.symm.trans h')

theorem not_reserve {e : LedgerEntry} {P : Prop} (h : e.kind ≠ "reserve") : e.kind = "reserve" → P :=
  fun hk => absurd hk h

/-- An entry that is neither an `open` nor an `end` keeps the invariant. -/
theorem inv_keep {pre : List LedgerEntry} {wins : Array Win} {v2 : Bool} {e : LedgerEntry}
    (hI : LedgerInv pre wins v2) (hseq : e.seq = pre.length)
    (hko : e.kind ≠ "open") (hke : e.kind ≠ "end")
    (hkr : e.kind = "reserve" → e.schema = LEDGER_SCHEMA ∧ ∀ i (h : i < wins.size), wins[i].opened.session ≠ e.session) :
    LedgerInv (pre ++ [e]) wins (v2 || e.schema == LEDGER_SCHEMA) where
  seq := seq_snoc hI.seq hseq
  v2 := v2_snoc hI.v2
  win := fun i h => by
    obtain ⟨h1, h2⟩ := hI.win i h
    exact ⟨List.mem_append_left _ h1, h2⟩
  opens := by
    intro o ho hk
    rcases mem_snoc.1 ho with ho | rfl
    · exact hI.opens o ho hk
    · exact absurd hk hko
  distinct := hI.distinct
  ended := fun i h x hx => by
    obtain ⟨h1, h2⟩ := hI.ended i h x hx
    exact ⟨List.mem_append_left _ h1, h2⟩
  ends := by
    intro x hx hk
    rcases mem_snoc.1 hx with hx | rfl
    · exact hI.ends x hx hk
    · exact absurd hk hke
  reserves := by
    intro rv hrv hk
    rcases mem_snoc.1 hrv with hrv | rfl
    · obtain ⟨hs, ho⟩ := hI.reserves rv hrv hk
      refine ⟨hs, fun o ho' hko' hses => ?_⟩
      rcases mem_snoc.1 ho' with ho' | rfl
      · exact ho o ho' hko' hses
      · exact absurd hko' hko
    · obtain ⟨hs, hw⟩ := hkr hk
      refine ⟨hs, fun o ho' hko' hses => ?_⟩
      rcases mem_snoc.1 ho' with ho' | rfl
      · obtain ⟨i, hi, rfl⟩ := hI.opens o ho' hko'
        exact absurd hses (hw i hi)
      · exact absurd (hk.symm.trans hko') (by decide)

/-- An `open` on a new root and a new session pushes its window. -/
theorem inv_open {pre : List LedgerEntry} {wins : Array Win} {v2 : Bool} {e : LedgerEntry}
    (hI : LedgerInv pre wins v2) (hseq : e.seq = pre.length) (hk : e.kind = "open")
    (hn : ¬(((wins.any fun x => x.root == strOf (openRoot e)) || wins.any fun x => x.opened.session == e.session) = true)) :
    LedgerInv (pre ++ [e]) (wins.push { root := strOf (openRoot e), opened := e, leaf := leafOf e.sig })
      (v2 || e.schema == LEDGER_SCHEMA) := by
  simp only [Bool.or_eq_true, not_or, Array.any_eq_true, not_exists, beq_iff_eq] at hn
  obtain ⟨hnr, hns⟩ := hn
  have last : ∀ i (h : i < (wins.push { root := strOf (openRoot e), opened := e, leaf := leafOf e.sig }).size),
      i < wins.size ∨ i = wins.size := by
    intro i h; rw [Array.size_push] at h; omega
  refine ⟨seq_snoc hI.seq hseq, v2_snoc hI.v2, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro i h
    rcases last i h with h' | rfl
    · rw [Array.getElem_push_lt h']
      obtain ⟨h1, h2⟩ := hI.win i h'
      exact ⟨List.mem_append_left _ h1, h2⟩
    · rw [Array.getElem_push_eq]
      exact ⟨by simp, hk, rfl, rfl⟩
  · intro o ho hko
    rcases mem_snoc.1 ho with ho | rfl
    · obtain ⟨i, hi, hw⟩ := hI.opens o ho hko
      exact ⟨i, by rw [Array.size_push]; omega, by rw [Array.getElem_push_lt hi]; exact hw⟩
    · exact ⟨wins.size, by rw [Array.size_push]; omega, by rw [Array.getElem_push_eq]⟩
  · intro i j hi hj hij
    rcases last i hi with hi' | rfl <;> rcases last j hj with hj' | rfl
    · rw [Array.getElem_push_lt hi', Array.getElem_push_lt hj']
      exact hI.distinct i j hi' hj' hij
    · rw [Array.getElem_push_lt hi', Array.getElem_push_eq]
      exact ⟨hnr i hi', hns i hi'⟩
    · rw [Array.getElem_push_lt hj', Array.getElem_push_eq]
      exact ⟨fun h => hnr j hj' h.symm, fun h => hns j hj' h.symm⟩
    · exact absurd rfl hij
  · intro i h x hx
    rcases last i h with h' | rfl
    · rw [Array.getElem_push_lt h'] at hx ⊢
      obtain ⟨h1, h2⟩ := hI.ended i h' x hx
      exact ⟨List.mem_append_left _ h1, h2⟩
    · rw [Array.getElem_push_eq] at hx; cases hx
  · intro x hx hke
    rcases mem_snoc.1 hx with hx | rfl
    · obtain ⟨i, hi, h1, h2⟩ := hI.ends x hx hke
      exact ⟨i, by rw [Array.size_push]; omega, by rw [Array.getElem_push_lt hi]; exact ⟨h1, h2⟩⟩
    · exact absurd (hk.symm.trans hke) (by decide)
  · intro rv hrv hkr
    rcases mem_snoc.1 hrv with hrv' | hrve
    · obtain ⟨hs, ho⟩ := hI.reserves rv hrv' hkr
      refine ⟨hs, fun o ho' hko' hses => ?_⟩
      rcases mem_snoc.1 ho' with ho'' | hoe
      · exact ho o ho'' hko' hses
      · rw [hoe, hseq]; exact hI.seq rv hrv'
    · rw [hrve] at hkr; exact absurd (hk.symm.trans hkr) (by decide)

theorem end_rec_some {e : LedgerEntry} {w : Win} {r : Nat} {records : Array (Nat × Nat)}
    (hn : SampledProofs.nat? (SampledProofs.getD e.body "record_leaf") = some r)
    (hb : (decide (w.leaf < r) && decide (r < leafOf e.sig) && !records.any fun x => x.1 == r) = true) :
    e.schema = LEDGER_SCHEMA → (SampledProofs.getD e.body "record_leaf").isNull = false →
      ∃ r, SampledProofs.nat? (SampledProofs.getD e.body "record_leaf") = some r ∧ w.leaf < r ∧ r < leafOf e.sig := by
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hb
  exact fun _ _ => ⟨r, hn, hb.1.1, hb.1.2⟩

theorem end_rec_none {e : LedgerEntry} {w : Win}
    (h : ¬((e.schema == LEDGER_SCHEMA && !(SampledProofs.getD e.body "record_leaf").isNull) = true)) :
    e.schema = LEDGER_SCHEMA → (SampledProofs.getD e.body "record_leaf").isNull = false →
      ∃ r, SampledProofs.nat? (SampledProofs.getD e.body "record_leaf") = some r ∧ w.leaf < r ∧ r < leafOf e.sig := by
  intro hs hn
  exact absurd (by simp [hs, hn]) h

/-- An `end` of an open window that hasn't ended closes it. -/
theorem inv_end {pre : List LedgerEntry} {wins : Array Win} {v2 : Bool} {e : LedgerEntry} {k : Nat}
    (hI : LedgerInv pre wins v2) (hseq : e.seq = pre.length) (hv1 : v2 = true → e.schema ≠ LEDGER_SCHEMA_V1)
    (hk : e.kind = "end")
    (hf : Array.findIdx? (fun x => x.opened.session == e.session) wins = some k)
    (hn : ¬(wins[k]!.ended.isSome = true))
    (hr : e.schema = LEDGER_SCHEMA → (SampledProofs.getD e.body "record_leaf").isNull = false →
      ∃ r, SampledProofs.nat? (SampledProofs.getD e.body "record_leaf") = some r ∧ wins[k]!.leaf < r ∧
        r < leafOf e.sig) :
    LedgerInv (pre ++ [e]) (wins.modify k fun w => { w with ended := some e }) (v2 || e.schema == LEDGER_SCHEMA) := by
  obtain ⟨hk', hks, -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hf
  rw [getElem!_pos wins k hk'] at hn hr
  simp only [beq_iff_eq] at hks
  have hnone : wins[k].ended = none := by
    cases h : wins[k].ended with
    | none => rfl
    | some _ => rw [h] at hn; exact absurd rfl hn
  have hsz : (wins.modify k fun w => { w with ended := some e }).size = wins.size := Array.size_modify
  have self : ∀ h, (wins.modify k fun w => { w with ended := some e })[k]'h = { wins[k] with ended := some e } :=
    fun h => Array.getElem_modify_self _ h
  have other : ∀ i (h : i < (wins.modify k fun w => { w with ended := some e }).size), k ≠ i →
      (wins.modify k fun w => { w with ended := some e })[i]'h = wins[i]'(hsz ▸ h) :=
    fun i h hi => Array.getElem_modify_of_ne hi _ h
  have same : ∀ i (h : i < (wins.modify k fun w => { w with ended := some e }).size),
      ((wins.modify k fun w => { w with ended := some e })[i]'h).opened = (wins[i]'(hsz ▸ h)).opened ∧
      ((wins.modify k fun w => { w with ended := some e })[i]'h).root = (wins[i]'(hsz ▸ h)).root ∧
      ((wins.modify k fun w => { w with ended := some e })[i]'h).leaf = (wins[i]'(hsz ▸ h)).leaf := by
    intro i h
    by_cases hki : k = i
    · subst hki; rw [self]; exact ⟨rfl, rfl, rfl⟩
    · rw [other i h hki]; exact ⟨rfl, rfl, rfl⟩
  refine ⟨seq_snoc hI.seq hseq, v2_snoc hI.v2, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro i h
    obtain ⟨h1, h2, h3⟩ := same i h
    obtain ⟨w1, w2, w3, w4⟩ := hI.win i (hsz ▸ h)
    rw [h1, h2, h3]
    exact ⟨List.mem_append_left _ w1, w2, w3, w4⟩
  · intro o ho hko
    rcases mem_snoc.1 ho with ho' | hoe
    · obtain ⟨i, hi, hw⟩ := hI.opens o ho' hko
      exact ⟨i, hsz ▸ hi, by rw [(same i (hsz ▸ hi)).1]; exact hw⟩
    · rw [hoe] at hko; exact absurd (hk.symm.trans hko) (by decide)
  · intro i j hi hj hij
    rw [(same i hi).1, (same i hi).2.1, (same j hj).1, (same j hj).2.1]
    exact hI.distinct i j (hsz ▸ hi) (hsz ▸ hj) hij
  · intro i h x hx
    by_cases hki : k = i
    · subst hki
      rw [self] at hx ⊢
      cases hx
      have ho := (hI.win k hk').1
      refine ⟨by simp, hk, hks.symm, ?_, ?_, hr⟩
      · rw [hseq]; exact hI.seq _ ho
      · intro hs; exact hv1 (hI.v2 _ ho hs)
    · rw [other i h hki] at hx ⊢
      obtain ⟨h1, h2⟩ := hI.ended i (hsz ▸ h) x hx
      exact ⟨List.mem_append_left _ h1, h2⟩
  · intro x hx hke
    rcases mem_snoc.1 hx with hx' | hxe
    · obtain ⟨i, hi, h1, h2⟩ := hI.ends x hx' hke
      have hik : k ≠ i := by
        rintro rfl; rw [hnone] at h2; cases h2
      exact ⟨i, hsz ▸ hi, by rw [other i (hsz ▸ hi) hik]; exact ⟨h1, h2⟩⟩
    · subst hxe
      exact ⟨k, hsz ▸ hk', by rw [self]; exact ⟨hks, rfl⟩⟩
  · intro rv hrv hkr
    rcases mem_snoc.1 hrv with hrv' | hrve
    · obtain ⟨hs, ho⟩ := hI.reserves rv hrv' hkr
      refine ⟨hs, fun o ho' hko' hses => ?_⟩
      rcases mem_snoc.1 ho' with ho'' | hoe
      · exact ho o ho'' hko' hses
      · rw [hoe] at hko'; exact absurd (hk.symm.trans hko') (by decide)
    · rw [hrve] at hkr; exact absurd (hk.symm.trans hkr) (by decide)

-- `mvcgen` splits `verifyLedger`'s loop body into one goal per path through its checks
set_option maxHeartbeats 4000000 in
/-- The ledger's invariant holds of every ledger `verifyLedger` accepts. -/
theorem verifyLedger_inv {es : Array LedgerEntry} {key : ChallengerKey} {r : Array Win × Array MonitoringPeriod}
    (h : verifyLedger es key = .ok r) : ∃ v2, LedgerInv es.toList r.1 v2 := by
  unfold verifyLedger at h
  rw [bind_ok] at h
  obtain ⟨s, hs, h⟩ := h
  have hI := forIn_array_zipIdx_ok es _ _ (fun pre s => LedgerInv pre s.1 s.2.2.2.2.2.2.2.2.1) inv_nil ?step hs
  · have hr : r.1 = s.1 := by
      refine of_triple (Q := fun r => r.1 = s.1) ?_ h
      clear hI hs h
      mvcgen [-Spec.throw_Except, throw_spec]
    rw [hr]; exact ⟨_, hI⟩
  · intro pre e s t hI hb
    obtain ⟨wins, leaves, records, periods, current, reserves, prev, last, v2, twoStep⟩ := s
    refine of_triple (Q := fun t => ∃ s', t = .yield s' ∧ LedgerInv (pre ++ [e]) s'.1 s'.2.2.2.2.2.2.2.2.1) ?_ hb
    clear hb hs h
    mvcgen [-Spec.throw_Except, throw_spec, ledgerMessage_spec, join_spec]
    all_goals (try (refine wp_ite (fun _ => restart ?_) (fun _ => restart ?_) <;>
      mvcgen [-Spec.throw_Except, throw_spec, ledgerMessage_spec, join_spec]))
    all_goals refine ⟨_, rfl, ?_⟩
    all_goals first
      | exact inv_open hI (chain ‹_›).1 ‹_› ‹_›
      | exact inv_end hI (chain ‹_›).1 (chain ‹_›).2 ‹_› ‹_› ‹_› (end_rec_some ‹_› ‹_›)
      | exact inv_end hI (chain ‹_›).1 (chain ‹_›).2 ‹_› ‹_› ‹_› (end_rec_none ‹_›)
      | exact inv_keep hI (chain ‹_›).1 (kind_ne ‹_› (by decide)) (kind_ne ‹_› (by decide))
          (not_reserve (kind_ne ‹_› (by decide)))
      | exact inv_keep hI (chain ‹_›).1 (kind_ne ‹_› (by decide)) (kind_ne ‹_› (by decide))
          (fun _ => reserve_facts ‹_›)

/-- The specification of a callee whose result `mvcgen` should keep as an equation. -/
theorem ok_spec {ε α : Type} (x : Except ε α) : ⦃⌜True⌝⦄ x ⦃⇓? a => ⌜x = .ok a⌝⦄ := by
  cases x
  · exact fun _ => True.intro
  · exact fun _ => rfl

theorem checked_spec (ledger : Json) (key : ChallengerKey) :
    ⦃⌜True⌝⦄ checked ledger key ⦃⇓? a => ⌜checked ledger key = .ok a⌝⦄ := ok_spec _

theorem auditedOpen_spec (ledger window : Json) (key : ChallengerKey) :
    ⦃⌜True⌝⦄ auditedOpen ledger window key ⦃⇓? a => ⌜auditedOpen ledger window key = .ok a⌝⦄ := ok_spec _

theorem of_triple_opt {α : Type} {x : Option α} {Q : α → Prop} {r : α}
    (hT : ⦃⌜True⌝⦄ x ⦃⇓? a => ⌜Q a⌝⦄) (h : x = some r) : Q r := by
  subst h; exact hT True.intro

theorem none_spec {α : Type} (Q : PostCond α (.except PUnit .pure)) : ⦃Q.2.1 ⟨⟩⦄ (none : Option α) ⦃Q⦄ := fun h => h

theorem failure_spec {α : Type} (Q : PostCond α (.except PUnit .pure)) : ⦃Q.2.1 ⟨⟩⦄ (failure : Option α) ⦃Q⦄ :=
  fun h => h

theorem some_pure_spec {α : Type} (a : α) (Q : PostCond α (.except PUnit .pure)) :
    ⦃Q.1 a⦄ (some a : Option α) ⦃Q⦄ := fun h => h

theorem some_spec {α : Type} (x : Option α) : ⦃⌜True⌝⦄ x ⦃⇓? a => ⌜x = some a⌝⦄ := by
  cases x
  · exact fun _ => True.intro
  · exact fun _ => rfl

theorem toOption_spec {ε α : Type} (x : Except ε α) : ⦃⌜True⌝⦄ x.toOption ⦃⇓? a => ⌜x.toOption = some a⌝⦄ :=
  some_spec _

theorem ledgerTag_spec (s : String) : ⦃⌜True⌝⦄ ledgerTag? s ⦃⇓? a => ⌜ledgerTag? s = some a⌝⦄ := some_spec _

theorem nat_spec (j : Json) : ⦃⌜True⌝⦄ SampledProofs.nat? j ⦃⇓? a => ⌜SampledProofs.nat? j = some a⌝⦄ := some_spec _

theorem ledgerTag_some {s t : String} (h : ledgerTag? s = some t) : s = LEDGER_SCHEMA_V1 ∨ s = LEDGER_SCHEMA := by
  unfold ledgerTag? at h
  split at h
  · exact .inl rfl
  · exact .inr rfl
  · cases h

-- `mvcgen` splits `ledgerEntry?` into one goal per field it parses
set_option maxHeartbeats 1000000 in
/-- A parsed entry is v1 or v2. -/
theorem ledgerEntry_schema {line : Json} {e : LedgerEntry} (h : ledgerEntry? line = some e) :
    e.schema = LEDGER_SCHEMA_V1 ∨ e.schema = LEDGER_SCHEMA := by
  refine of_triple_opt (Q := fun e => e.schema = LEDGER_SCHEMA_V1 ∨ e.schema = LEDGER_SCHEMA) ?_ h
  clear h
  mvcgen [ledgerEntry?, guard, -Spec.throw_Option, none_spec, failure_spec, toOption_spec, ledgerTag_spec, nat_spec,
    some_pure_spec]
  all_goals exact ledgerTag_some ‹_›

theorem step_push {b : Array LedgerEntry} {pre : List (Json × Nat)} {cur : Json × Nat} {e : LedgerEntry}
    (h : b.toList = (pre.map Prod.fst).filterMap ledgerEntry?) (he : ledgerEntry? cur.1 = some e) :
    (b.push e).toList = ((pre ++ [cur]).map Prod.fst).filterMap ledgerEntry? := by
  simp [h, he, List.filterMap_append]

theorem entries_arr {ls : Array Json} {es : Array LedgerEntry}
    (h : es.toList = (ls.zipIdx.toList.map Prod.fst).filterMap ledgerEntry?) :
    entries (Json.arr ls) = es.toList := by
  rw [h, Array.toList_zipIdx, List.zipIdx_map_fst]; rfl

-- the closers must not unfold `verifyLedger` or the key's hash while they search the context
set_option maxHeartbeats 1000000 in
/-- `checked` runs `verifyLedger` on the entries `LiveSalt.entries` reads. -/
theorem checked_ok {ledger : Json} {key : ChallengerKey} {r : Array Win × Array MonitoringPeriod}
    (h : checked ledger key = .ok r) :
    ∃ es : Array LedgerEntry, entries ledger = es.toList ∧ verifyLedger es key = .ok r := by
  refine of_triple (Q := fun r => ∃ es : Array LedgerEntry, entries ledger = es.toList ∧ verifyLedger es key = .ok r) ?_ h
  clear h
  mvcgen -trivial [checked, -Spec.throw_Except, throw_spec] invariants
  · ⇓? ⟨cur, es⟩ => ⌜es.toList = (cur.prefix.map Prod.fst).filterMap ledgerEntry?⌝
  all_goals first
    | exact True.intro
    | (apply step_push <;> with_reducible assumption)
    | exact ⟨_, entries_arr ‹_›, ‹_›⟩
    | exact (rfl : ([] : List LedgerEntry) = [])

theorem pinnedSession_ok {ledger window : Json} {key : ChallengerKey} {pins main : Json} {a : Audited}
    (h : pinnedSession ledger window key pins main = .ok a) : ∃ ob, auditedOpen ledger window key = .ok (a, ob) := by
  refine of_triple (Q := fun a => ∃ ob, auditedOpen ledger window key = .ok (a, ob)) ?_ h
  clear h
  mvcgen [pinnedSession, -Spec.throw_Except, throw_spec, auditedOpen_spec]
  all_goals exact ⟨_, by assumption⟩

-- `mvcgen` splits `auditedOpen` into one goal per check it makes of the window and its `end`
set_option maxHeartbeats 4000000 in
/-- `auditedOpen` takes the v2 window named by the caller's root, closed by its `end`. -/
theorem auditedOpen_ok {ledger window : Json} {key : ChallengerKey} {r : Audited × Json}
    (h : auditedOpen ledger window key = .ok r) :
    ∃ (wp : Array Win × Array MonitoringPeriod) (win : Win) (e : LedgerEntry), checked ledger key = .ok wp ∧
      wp.1.find? (fun w => w.root == strOf (SampledProofs.getD window "root")) = some win ∧
      (win.opened.schema == LEDGER_SCHEMA) = true ∧ win.ended = some e ∧
      jsonTrue (SampledProofs.getD e.body "closed") = true ∧
      SampledProofs.nat? (SampledProofs.getD e.body "record_leaf") = some r.1.recordLeaf ∧
      r.1.session = win.opened.session ∧ r.1.commitment = strOf (SampledProofs.getD e.body "commitment") := by
  refine of_triple (Q := fun r => ∃ (wp : Array Win × Array MonitoringPeriod) (win : Win) (e : LedgerEntry),
      checked ledger key = .ok wp ∧
      wp.1.find? (fun w => w.root == strOf (SampledProofs.getD window "root")) = some win ∧
      (win.opened.schema == LEDGER_SCHEMA) = true ∧ win.ended = some e ∧
      jsonTrue (SampledProofs.getD e.body "closed") = true ∧
      SampledProofs.nat? (SampledProofs.getD e.body "record_leaf") = some r.1.recordLeaf ∧
      r.1.session = win.opened.session ∧ r.1.commitment = strOf (SampledProofs.getD e.body "commitment")) ?_ h
  clear h
  mvcgen [auditedOpen, -Spec.throw_Except, throw_spec, checked_spec]
  all_goals
    exact ⟨_, _, _, by assumption, by assumption, by assumption, by assumption, by assumption, by assumption, rfl, rfl⟩

theorem jsonTrue_eq {j : Json} (h : PouwCheck.jsonTrue j = true) : j = .bool true := by
  cases j <;> (try cases ‹Bool›) <;> first | rfl | cases h

theorem jsonTrue_eq' {j : Json} (h : jsonTrue j = true) : j = .bool true := by
  cases j <;> (try cases ‹Bool›) <;> first | rfl | cases h

theorem nat_nonnull {j : Json} {n : Nat} (h : SampledProofs.nat? j = some n) : j.isNull = false := by
  cases j <;> first | rfl | simp [SampledProofs.nat?, SampledProofs.int?] at h

theorem strOf_eq {j : Json} {s : String} (h : strOf j = s) (hs : s ≠ "") : j = .str s := by
  cases j <;> first | (subst h; rfl) | exact absurd h.symm hs

theorem getD_root (x : Json) : SampledProofs.getD (Json.mkObj [("root", x)]) "root" = x := rfl

theorem idx_session {pre : List LedgerEntry} {wins : Array Win} {v2 : Bool} (hI : LedgerInv pre wins v2)
    {i j : Nat} (hi : i < wins.size) (hj : j < wins.size)
    (h : wins[i].opened.session = wins[j].opened.session) : i = j := by
  by_cases hij : i = j
  · exact hij
  · exact absurd h (hI.distinct i j hi hj hij).2

theorem idx_root {pre : List LedgerEntry} {wins : Array Win} {v2 : Bool} (hI : LedgerInv pre wins v2)
    {i j : Nat} (hi : i < wins.size) (hj : j < wins.size) (h : wins[i].root = wins[j].root) : i = j := by
  by_cases hij : i = j
  · exact hij
  · exact absurd h (hI.distinct i j hi hj hij).1

theorem find_mem {ledger : Json} {sid : String} {p : LedgerEntry → Bool} {x : LedgerEntry}
    (h : List.find? p (PouwCheck.sessionEntries ledger sid) = some x) :
    x ∈ entries ledger ∧ x.session = sid ∧ p x = true := by
  have hm := List.mem_of_find?_eq_some h
  simp only [PouwCheck.sessionEntries, List.mem_filter, beq_iff_eq] at hm
  exact ⟨hm.1, hm.2, List.find?_some h⟩

/-- The ledger of a session pinned on root `R` shows its reserve, its one open on `R` and its end, from the entries
`PouwCheck.liveSaltCheck` finds and checks. -/
theorem ledgerShows_of {ledger : Json} {ck : ChallengerKey} {pins main : Json} {a : Audited} {R : ByteArray}
    {rv op en : LedgerEntry} (hR : Flock.hexOf R ≠ "")
    (hpin : pinnedSession ledger (Json.mkObj [("root", .str (Flock.hexOf R))]) ck pins main = .ok a)
    (hfind : (List.find? (fun x => x.kind == "reserve") (PouwCheck.sessionEntries ledger a.session),
      List.find? (fun x => x.kind == "open") (PouwCheck.sessionEntries ledger a.session),
      List.find? (fun x => x.kind == "end") (PouwCheck.sessionEntries ledger a.session)) = (some rv, some op, some en))
    (hfam : SampledProofs.jeq (SampledProofs.getD rv.body "families") (familiesJson PouwWindow.SALT_FAMILIES) = true)
    (hres : (SampledProofs.int? (SampledProofs.getD op.body "reserve") == some (rv.seq : Int)) = true)
    (hcomp : (PouwCheck.jsonTrue (SampledProofs.getD en.body "complete") &&
      SampledProofs.jeq (SampledProofs.getD en.body "released")
        (Json.mkObj [(PouwWindow.SALT, Lean.toJson (1 : Nat))])) = true) :
    LedgerShows ledger a.session R a.recordLeaf op.seq := by
  obtain ⟨ob, hao⟩ := pinnedSession_ok hpin
  obtain ⟨wp, win, en', hch, hfw, hsch, hend, hclosed, hrl, hsid, -⟩ := auditedOpen_ok hao
  obtain ⟨es, hes, hvl⟩ := checked_ok hch
  obtain ⟨v2, hI⟩ := verifyLedger_inv hvl
  obtain ⟨iw, hiw, hwin⟩ := Array.mem_iff_getElem.1 (Array.mem_of_find?_eq_some hfw)
  have hroot : win.root = Flock.hexOf R := by
    have := Array.find?_some hfw
    rw [getD_root] at this
    exact beq_iff_eq.1 this
  have hsch' : win.opened.schema = LEDGER_SCHEMA := beq_iff_eq.1 hsch
  simp only [Prod.mk.injEq] at hfind
  obtain ⟨hfr, hfo, hfe⟩ := hfind
  obtain ⟨rvE, rvS, rvK⟩ := find_mem hfr
  obtain ⟨opE, opS, opK⟩ := find_mem hfo
  obtain ⟨enE, enS, enK⟩ := find_mem hfe
  simp only [beq_iff_eq] at rvK opK enK
  have mem : ∀ {x}, x ∈ entries ledger → x ∈ es.toList := fun h => hes ▸ h
  have hwS : wp.1[iw].opened.session = a.session := by rw [hwin, hsid]
  have hop : op = win.opened := by
    obtain ⟨j, hj, hjo⟩ := hI.opens op (mem opE) opK
    have : j = iw := idx_session hI hj hiw (by rw [hjo, opS, hwS])
    subst this
    rw [← hjo, hwin]
  have hen : win.ended = some en := by
    obtain ⟨i, hi, his, hie⟩ := hI.ends en (mem enE) enK
    have : i = iw := idx_session hI hi hiw (by rw [his, enS, hwS])
    subst this
    rw [← hwin]; exact hie
  have heq : en' = en := Option.some.inj (hend.symm.trans hen)
  rw [heq] at hclosed hrl
  obtain ⟨-, -, -, hlt, hv1, hrec⟩ := hI.ended iw hiw en (by rw [hwin]; exact hen)
  obtain ⟨-, -, hwr, hwl⟩ := hI.win iw hiw
  rw [hwin] at hlt hv1 hrec hwr hwl
  obtain ⟨line, -, hline⟩ := List.mem_filterMap.1 enE
  have hes2 : en.schema = LEDGER_SCHEMA := by
    rcases ledgerEntry_schema hline with h1 | h1
    · exact absurd h1 (hv1 hsch')
    · exact h1
  obtain ⟨r', hr', hl1, hl2⟩ := hrec hes2 (nat_nonnull hrl)
  obtain rfl : r' = a.recordLeaf := Option.some.inj (hr'.symm.trans hrl)
  have hor : openRoot win.opened = .str (Flock.hexOf R) := strOf_eq (hwr.symm.trans hroot) hR
  obtain ⟨rvSch, rvLt⟩ := hI.reserves rv (mem rvE) rvK
  simp only [Bool.and_eq_true] at hcomp
  obtain ⟨hcomp1, hcomp2⟩ := hcomp
  have huniq : ∀ o' ∈ entries ledger, o'.kind = "open" →
      (openRoot o' = .str (Flock.hexOf R) ∨ o'.session = a.session) → o' = op := by
    intro o' ho' hk' hor'
    obtain ⟨j, hj, hjo⟩ := hI.opens o' (mem ho') hk'
    have : j = iw := by
      rcases hor' with hr | hs
      · apply idx_root hI hj hiw
        rw [(hI.win j hj).2.2.1, hjo, hr, hwin, hroot]; rfl
      · exact idx_session hI hj hiw (by rw [hjo, hs, hwS])
    subst this
    rw [← hjo, hwin, hop]
  refine ⟨rv, op, en, rvE, opE, enE, ⟨rvK, rvS, rvSch, hfam⟩,
    ⟨opK, opS, by rw [hop]; exact hsch', rvLt op (mem opE) opK (opS.trans rvS.symm), rfl, beq_iff_eq.1 hres,
      by rw [hop]; exact hor⟩,
    ⟨enK, enS, by rw [hop]; exact hlt, jsonTrue_eq' hclosed, jsonTrue_eq hcomp1, hcomp2, hrl,
      by rw [hop, ← hwl]; exact hl1, hl2⟩, huniq⟩

theorem sessionOf_spec (key : ChallengerKey) (record log close : Json) :
    ⦃⌜True⌝⦄ sessionOf key record log close ⦃⇓? _ => ⌜True⌝⦄ := except_spec _

theorem recordOfJson_spec (j : Json) : ⦃⌜True⌝⦄ ChallengerRecord.ofJson j ⦃⇓? _ => ⌜True⌝⦄ := except_spec _

theorem mapM_spec {α β : Type} (f : α → Flock.V β) (l : List α) : ⦃⌜True⌝⦄ l.mapM f ⦃⇓? _ => ⌜True⌝⦄ :=
  except_spec _

theorem commitment_of {q l : Nat} {s t : String} (h : (q == l && s == t) = true) : s = t := by
  simp only [Bool.and_eq_true, beq_iff_eq] at h
  exact h.2

-- `mvcgen` splits `sessionNamed` into one goal per check it makes of the session's record
set_option maxHeartbeats 1000000 in
/-- `sessionNamed` returns the commitment whose root's hex the ledger's `end` recorded. -/
theorem sessionNamed_root {a : Audited} {key : ChallengerKey} {session : Json} {r : LiveCommitment × Roots}
    (h : sessionNamed a key session = .ok r) : Flock.hexOf r.1.root = a.commitment := by
  refine of_triple (Q := fun r => Flock.hexOf r.1.root = a.commitment) ?_ h
  clear h
  mvcgen [sessionNamed, -Spec.throw_Except, throw_spec, sessionOf_spec, recordOfJson_spec, mapM_spec]
  apply commitment_of <;> with_reducible assumption

end Properties.Pouw.LiveSalt.Ledger
