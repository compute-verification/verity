module

public import Proofs.Flock.Recursive.FirewallRun

/-!
# The ZK server sends the schedule's transcript

`run_blank_tables`: a session the ZK server (`Flock.Firewall.run`) finishes sends, but for its commit strings,
`tmpl s s.tables`, the pinned schedule `s`'s transcript with Commit's tables in the schedule's order
(`Schedule.tables`), the only order `step` takes, so the tables its Commit names (`namesOf`) are the schedule's
(`namesOf_run`). Every count, name, order, stream, parent and position it sends is the schedule's, through the
field-by-field pins beside `step`'s canonical compares (`roundSpec`, Commit's node counts and order, Finish's round
count). With `FirewallRun.run_of_one`, a session that finishes at one salt list, its Commit naming the tables `o`
(`s.tables`, by `namesOf_run`), its commit strings those of 64-byte digests `D`, sends `tmpl s o` filled with `D`'s
commit strings at every salt list of that length (`sends_of_one`).

The proof reads `step` one case at a time from states equal but for their salts (`H`, `Like`), and carries `Inv` through
`run`'s loop: what was sent, blanked, is what the state's flags say the schedule has had so far (`sofar`).
-/

@[expose] public section

namespace FlockSoundness.Discharge.Recursion.FirewallTemplate

open Flock Flock.Firewall FlockSoundness.Discharge.Recursion.FirewallRun

/-! ## The schedule's transcript, blank -/

/-- `n` blank commit strings. -/
def blanks (n : Nat) : List ByteArray := List.replicate n ByteArray.empty

/-- A stream's open, at the parent and position the schedule gives it. -/
def openT (s : Schedule) (stream : String) : List Sent :=
  match s.streams.find? (·.1 == stream) with
  | some (_, parent, pos, _) => [.openStream stream parent pos]
  | none => []

/-- Round `i` of the schedule, blank: its stream's open before the stream's first round, then the round, each count the
schedule's (`roundSpec`). -/
def roundT (s : Schedule) (i : Nat) : List Sent :=
  match roundSpec (s.rounds.getD i .null) with
  | some (stream, k, n, tops, level0, commits) =>
    (if k == 0 then openT s stream else []) ++ [.round stream (tops.map blanks) level0 (blanks commits) n]
  | none => []

/-- The schedule's first `r` rounds, blank. -/
def roundsT (s : Schedule) (r : Nat) : List Sent := (List.range r).flatMap (roundT s)

/-- Commit, blank, naming the tables `o` in order, each with the schedule's node count. -/
def commitT (s : Schedule) (o : List String) : Sent :=
  .commit (o.map fun t => (t, blanks ((natOf? (s.roots.getObjValD t)).getD 0)))

/-- **The schedule's transcript**, blank, with Commit's tables in the order `o`. -/
def tmpl (s : Schedule) (o : List String) : List Sent :=
  [.hello s.hello, commitT s o, .link s.y] ++ roundsT s s.rounds.size ++ [.finish true]

/-- The tables a transcript's Commit names, in order. -/
def namesOf : List Sent → List String
  | _ :: .commit roots :: _ => roots.map (·.1)
  | _ => []

theorem namesOf_tmpl (s : Schedule) (o : List String) : namesOf (tmpl s o) = o := by
  simp only [tmpl, commitT, List.cons_append, namesOf, List.map_map]
  exact List.map_id' o

theorem namesOf_blank (l : List Sent) : namesOf (l.map blank) = namesOf l := by
  match l with
  | [] => rfl
  | [_] => rfl
  | _ :: m :: _ =>
    cases m <;> simp only [List.map_cons, blank, namesOf, List.map_map]
    rfl

/-- What the schedule has had so far at a state: its flags and its round count. -/
def sofar (s : Schedule) (st : State) : List Sent :=
  (if st.hello then [.hello s.hello] else []) ++ (if st.committed then [commitT s s.tables] else []) ++
    (if st.linked then [.link s.y] else []) ++ roundsT s st.rounds ++ (if st.finished then [.finish true] else [])

/-- The ZK server's invariant: what it sent, blanked, is what the schedule has had so far, and its flags come in order. -/
structure Inv (s : Schedule) (st : State) : Prop where
  sent : st.sent.toList.map blank = sofar s st
  hello : st.committed = true → st.hello = true
  committed : st.linked = true → st.committed = true
  linked : 0 < st.rounds → st.linked = true
  finished : st.finished = true → st.linked = true ∧ st.rounds = s.rounds.size

/-! ## One run of the ZK server's state monad -/

/-- An outcome of the ZK server's state monad: a refusal, or a result `a` at a state `t` with `Q a t`. -/
def OK {α : Type} (Q : α → State → Prop) : V (α × State) → Prop
  | .ok (a, t) => Q a t
  | .error _ => True

/-- From every state `P` holds of, `m` refuses or returns at a state `Q` holds of. -/
def H {α : Type} (P : State → Prop) (m : Fw α) (Q : α → State → Prop) : Prop :=
  ∀ st, P st → OK Q (m.run st)

/-- A state that is `v` but for its salts. -/
def Like (v t : State) : Prop := ∃ z, t = { v with salts := z }

theorem like_refl (v : State) : Like v v := ⟨v.salts, rfl⟩

theorem inv_like {s : Schedule} {v t : State} (h : Like v t) (hv : Inv s v) : Inv s t := by
  obtain ⟨z, rfl⟩ := h
  exact ⟨hv.sent, hv.hello, hv.committed, hv.linked, hv.finished⟩

theorem H.bind {α β : Type} {P : State → Prop} {m : Fw α} {f : α → Fw β} {Q : α → State → Prop}
    {R : β → State → Prop} (h : H P m Q) (hf : ∀ a, H (Q a) (f a) R) : H P (m >>= f) R := by
  intro st hP
  have h1 := h st hP
  rw [StateT.run_bind]
  rcases hm : m.run st with e | ⟨a, t⟩ <;> rw [hm] at h1
  · exact True.intro
  · exact hf a t h1

theorem H.conseq {α : Type} {P : State → Prop} {m : Fw α} {Q Q' : α → State → Prop} (h : H P m Q)
    (hQ : ∀ a t, Q a t → Q' a t) : H P m Q' := by
  intro st hP
  have h1 := h st hP
  revert h1
  rcases m.run st with e | ⟨a, t⟩
  · exact fun _ => True.intro
  · exact hQ a t

theorem H.intro {α : Type} {c : Prop} {P : State → Prop} {m : Fw α} {Q : α → State → Prop}
    (h : c → H P m Q) : H (fun t => c ∧ P t) m Q :=
  fun st hst => h hst.1 st hst.2

theorem H.pure {α : Type} {P : State → Prop} {Q : α → State → Prop} {a : α} (h : ∀ st, P st → Q a st) :
    H P (Pure.pure a) Q :=
  fun st hP => h st hP

theorem H.throw {α : Type} {P : State → Prop} {Q : α → State → Prop} {e : String} :
    H P (throw e : Fw α) Q :=
  fun _ _ => True.intro

theorem H.throw_bind {α β : Type} {P : State → Prop} {Q : β → State → Prop} {e : String} {f : α → Fw β} :
    H P ((MonadExcept.throw e : Fw α) >>= f) Q :=
  fun _ _ => True.intro

theorem H.pure_bind {α β : Type} {P : State → Prop} {R : β → State → Prop} {a : α} {f : α → Fw β}
    (h : H P (f a) R) : H P (Pure.pure a >>= f) R :=
  fun st hP => h st hP

theorem H.ite {α : Type} {P : State → Prop} {Q : α → State → Prop} {c : Prop} [Decidable c] {a b : Fw α}
    (ha : c → H P a Q) (hb : ¬c → H P b Q) : H P (if c then a else b) Q := by
  by_cases h : c
  · rw [ite_eq_left h]; exact ha h
  · rw [ite_eq_right h]; exact hb h

/-- A `do` block's `if` with a join point `b` after it, shown once for every state `J` holds of: `v`'s when the `if`'s
branch is skipped, and the branch `a` may end in it. -/
theorem H.ite_jp {α : Type} {v : State} {Q : α → State → Prop} {c : Prop} [Decidable c] {a b : Fw α}
    (J : State → Prop) (hb : ∀ w, J w → H (Like w) b Q) (hv : ¬c → J v)
    (ha : c → (∀ w, J w → H (Like w) b Q) → H (Like v) a Q) : H (Like v) (if c then a else b) Q :=
  H.ite (fun h => ha h hb) fun h => hb v (hv h)

theorem H.get_like {β : Type} {v : State} {f : State → Fw β} {R : β → State → Prop}
    (hf : ∀ z, H (Like v) (f { v with salts := z }) R) : H (Like v) (MonadState.get >>= f) R := by
  rintro _ ⟨z, rfl⟩
  exact hf z _ ⟨z, rfl⟩

theorem H.set {P : State → Prop} {s' w : State} (h : Like w s') :
    H P (MonadStateOf.set s' : Fw PUnit) (fun _ t => Like w t) :=
  fun _ _ => h

theorem H.modify {v : State} {f : State → State} (w : State) (hf : ∀ z, Like w (f { v with salts := z })) :
    H (Like v) (modify f : Fw PUnit) (fun _ t => Like w t) := by
  rintro _ ⟨z, rfl⟩
  exact hf z

theorem H.send {v : State} (m : Sent) :
    H (Like v) (send m) (fun _ t => Like { v with sent := v.sent.push m } t) :=
  H.modify _ fun z => ⟨z, rfl⟩

/-- A computation the ZK server's state doesn't enter (`decode`, `split`, `nodes`, `pinned`), lifted: its result is its
value. -/
theorem H.lift {α : Type} {P : State → Prop} {x : V α} :
    H P (liftM x : Fw α) (fun a t => x = .ok a ∧ P t) := by
  intro st hP
  cases x with
  | error e => exact True.intro
  | ok a => exact ⟨rfl, hP⟩

theorem H.salted {v : State} (row : Array Nat) : H (Like v) (salted row) (fun _ t => Like v t) := by
  rintro _ ⟨z, rfl⟩
  rw [run_salted]
  cases z with
  | nil => exact True.intro
  | cons y rest => exact ⟨rest, rfl⟩

/-- `l.mapM f` from states `v` but for their salts, where each `f a` returns `b` with `G b = F a`. -/
theorem H.mapM {α β γ : Type} {v : State} (f : α → Fw β) (G : β → γ) (F : α → γ)
    (hf : ∀ a, H (Like v) (f a) (fun b t => G b = F a ∧ Like v t)) :
    ∀ l : List α, H (Like v) (l.mapM f) (fun bs t => bs.map G = l.map F ∧ Like v t)
  | [] => by
    rw [List.mapM_nil]
    exact H.pure fun st h => ⟨rfl, h⟩
  | a :: l => by
    rw [List.mapM_cons]
    refine H.bind (hf a) fun b => H.intro fun hb => ?_
    refine H.bind (H.mapM f G F hf l) fun bs => H.intro fun hbs => ?_
    exact H.pure fun st h => ⟨by rw [List.map_cons, List.map_cons, hb, hbs], h⟩

/-- `l.mapM f` from states `v` but for their salts, where `f` leaves them so. -/
theorem H.mapM_like {α β : Type} {v : State} (f : α → Fw β) (hf : ∀ a, H (Like v) (f a) (fun _ t => Like v t))
    (l : List α) : H (Like v) (l.mapM f) (fun _ t => Like v t) :=
  H.conseq (H.mapM f (fun _ => ()) (fun _ => ()) (fun a => H.conseq (hf a) fun _ _ h => ⟨rfl, h⟩) l)
    fun _ _ h => h.2

theorem blankTop_eq (l : List ByteArray) : blankTop l = blanks l.length := by
  simp only [blankTop, blanks, List.map_const']

/-- `saltedTop`: one commit string per node of the tree top. -/
theorem H.saltedTop {v : State} (top : ByteArray) :
    H (Like v) (saltedTop top) (fun l t => blankTop l = blanks (top.size / NODE_BYTES) ∧ Like v t) := by
  refine H.conseq (H.mapM Firewall.salted (fun _ => ByteArray.empty) (fun _ => ByteArray.empty)
    (fun row => H.conseq (H.salted row) fun _ _ h => ⟨rfl, h⟩) (topRows top)) ?_
  rintro l t ⟨hl, ht⟩
  refine ⟨?_, ht⟩
  have := congrArg List.length hl
  simp only [List.length_map, topRows, List.length_range] at this
  rw [blankTop_eq, this]

/-! ## The invariant through `step` -/

theorem Inv.unlinked {s : Schedule} {v : State} (hv : Inv s v) (h : v.linked = false) :
    v.rounds = 0 ∧ v.finished = false := by
  refine ⟨Nat.eq_zero_of_not_pos fun hp => (nomatch h.symm.trans (hv.linked hp)), ?_⟩
  cases hf : v.finished
  · rfl
  · exact nomatch h.symm.trans (hv.finished hf).1

theorem Inv.uncommitted {s : Schedule} {v : State} (hv : Inv s v) (h : v.committed = false) :
    v.linked = false ∧ v.rounds = 0 ∧ v.finished = false := by
  have hl : v.linked = false := by
    cases hl : v.linked
    · rfl
    · exact nomatch h.symm.trans (hv.committed hl)
  exact ⟨hl, hv.unlinked hl⟩

theorem Inv.fresh {s : Schedule} {v : State} (hv : Inv s v) (h : v.hello = false) :
    v.committed = false ∧ v.linked = false ∧ v.rounds = 0 ∧ v.finished = false := by
  have hc : v.committed = false := by
    cases hc : v.committed
    · rfl
    · exact nomatch h.symm.trans (hv.hello hc)
  exact ⟨hc, hv.uncommitted hc⟩

/-- A message sent, where the schedule's transcript so far grows by it, blanked. -/
theorem Inv.sent_push {s : Schedule} {v : State} (hv : Inv s v) {t : State} {m : Sent} (ht : t.sent = v.sent.push m)
    (h : sofar s t = sofar s v ++ [blank m]) : t.sent.toList.map blank = sofar s t := by
  rw [ht, Array.toList_push, List.map_append, hv.sent, h]
  rfl

theorem step_ping (s : Schedule) (v : State) (hv : Inv s v) : H (Like v) (step s .ping) (fun _ t => Inv s t) := by
  rw [step]
  exact H.pure fun _ h => inv_like h hv

theorem step_prime (s : Schedule) (v : State) : H (Like v) (step s .prime) (fun _ t => Inv s t) := by
  rw [step]
  exact H.throw

theorem step_register (s : Schedule) (v : State) (hv : Inv s v) :
    H (Like v) (step s .register) (fun _ t => Inv s t) := by
  rw [step]
  refine H.get_like fun _ => ?_
  exact H.ite (fun _ => H.throw) fun _ => H.pure fun _ h => inv_like h hv

theorem step_proof (s : Schedule) (v : State) (hv : Inv s v) : H (Like v) (step s .proof) (fun _ t => Inv s t) := by
  rw [step]
  refine H.get_like fun _ => ?_
  exact H.ite (fun _ => H.throw) fun _ => H.pure fun _ h => inv_like h hv

theorem step_hello (s : Schedule) (text : String) (v : State) (hv : Inv s v) :
    H (Like v) (step s (.hello text)) (fun _ t => Inv s t) := by
  rw [step]
  refine H.get_like fun _ => ?_
  refine H.ite (fun _ => H.throw_bind) fun _ => ?_
  refine H.get_like fun _ => ?_
  refine H.ite (fun _ => H.throw_bind) fun hh => ?_
  rcases hj : Flock.JsonParse.parse text with e | j
  · exact H.throw_bind
  refine H.pure_bind ?_
  refine H.ite (fun _ => H.throw_bind) fun _ => ?_
  refine H.ite (fun _ => H.throw_bind) fun _ => ?_
  refine H.ite (fun _ => H.throw_bind) fun _ => ?_
  refine H.bind H.lift fun _ => H.intro fun _ => ?_
  refine H.bind (H.modify { v with hello := true } fun z => ⟨z, rfl⟩) fun _ => ?_
  refine H.conseq (H.send _) fun _ t ht => inv_like ht ?_
  have hh : v.hello = false := by simpa using hh
  obtain ⟨hc, hl, hr, hf⟩ := hv.fresh hh
  refine ⟨hv.sent_push rfl ?_, fun _ => rfl, hv.committed, hv.linked, hv.finished⟩
  simp [sofar, hh, hc, hl, hr, hf, roundsT, blank]

theorem step_commit (s : Schedule) (rootF : ByteArray) (roots publics : List (String × ByteArray)) (v : State)
    (hv : Inv s v) : H (Like v) (step s (.commit rootF roots publics)) (fun _ t => Inv s t) := by
  rw [step]
  refine H.get_like fun _ => ?_
  refine H.ite (fun _ => H.throw_bind) fun _ => ?_
  refine H.ite (fun _ => H.throw_bind) fun _ => ?_
  refine H.ite (fun _ => H.throw_bind) fun _ => ?_
  refine H.bind (H.mapM_like _ (fun e => ?_) _) fun _ => ?_
  · obtain ⟨n, c⟩ := e
    refine H.bind H.lift fun _ => H.intro fun _ => ?_
    exact H.pure fun _ h => h
  refine H.bind H.lift fun _ => H.intro fun _ => ?_
  refine H.ite (fun _ => H.throw_bind) fun hpin => ?_
  refine H.ite (fun _ => H.throw_bind) fun hord => ?_
  refine H.bind (H.mapM _ blankRoot (fun e => (e.1, blanks (e.2.size / NODE_BYTES))) (fun e => ?_) _)
    fun tops => H.intro fun hT => ?_
  · obtain ⟨n, c⟩ := e
    refine H.bind (H.saltedTop c) fun l => H.intro fun hl => ?_
    exact H.pure fun _ h => ⟨by simp only [blankRoot, hl], h⟩
  refine H.get_like fun z => ?_
  refine H.ite (fun _ => H.throw_bind) fun hok => ?_
  refine H.bind (H.set (w := { v with committed := true, roots := roots }) ⟨z, rfl⟩) fun _ => ?_
  refine H.conseq (H.send _) fun _ t ht => inv_like ht ?_
  have hok : v.hello = true ∧ v.committed = false := by simpa using hok
  obtain ⟨hl, hr, hf⟩ := hv.uncommitted hok.2
  have hpin : ∀ e ∈ roots, natOf? (s.roots.getObjValD e.1) = some (e.2.size / NODE_BYTES) := by
    rintro ⟨n, c⟩ he
    simp only [List.any_eq_true, not_exists, not_and] at hpin
    simpa using hpin (n, c) he
  have hord : roots.map (·.1) = s.tables := by simpa using hord
  have hC : commitT s s.tables = blank (.commit tops) := by
    rw [← hord]
    simp only [commitT, blank, hT, List.map_map]
    congr 1
    refine List.map_congr_left fun e he => ?_
    simp [hpin e he]
  refine ⟨hv.sent_push rfl ?_, fun _ => hok.1, fun h => (nomatch hl.symm.trans h), hv.linked, hv.finished⟩
  simp [sofar, hok.1, hok.2, hl, hr, hf, roundsT, hC]

theorem step_link (s : Schedule) (y : ByteArray) (v : State) (hv : Inv s v) :
    H (Like v) (step s (.link y)) (fun _ t => Inv s t) := by
  rw [step]
  refine H.get_like fun _ => ?_
  refine H.ite (fun _ => H.throw_bind) fun _ => ?_
  refine H.ite (fun _ => H.throw_bind) fun _ => ?_
  refine H.ite (fun _ => H.throw_bind) fun _ => ?_
  refine H.get_like fun z => ?_
  refine H.ite (fun _ => H.throw_bind) fun hok => ?_
  refine H.bind (H.set (w := { v with linked := true }) ⟨z, rfl⟩) fun _ => ?_
  refine H.conseq (H.send _) fun _ t ht => inv_like ht ?_
  have hok : v.committed = true ∧ v.linked = false := by simpa using hok
  obtain ⟨hr, hf⟩ := hv.unlinked hok.2
  refine ⟨hv.sent_push rfl ?_, hv.hello, fun _ => hok.1, fun _ => rfl, fun h => (nomatch hf.symm.trans h)⟩
  simp [sofar, hv.hello hok.1, hok.1, hok.2, hr, hf, roundsT, blank]

theorem step_openStream (s : Schedule) (stream parent : String) (label : ByteArray) (pos : Nat) (v : State)
    (hv : Inv s v) : H (Like v) (step s (.openStream stream parent label pos)) (fun _ t => Inv s t) := by
  rw [step]
  refine H.get_like fun _ => ?_
  refine H.ite (fun _ => H.throw_bind) fun _ => ?_
  generalize List.find? _ s.streams = o
  rcases o with _ | ⟨w₁, wp, wa, w₂⟩
  · exact H.throw
  refine H.ite (fun _ => H.throw_bind) fun _ => ?_
  refine H.get_like fun z => ?_
  refine H.ite (fun _ => H.throw_bind) fun _ => ?_
  refine H.conseq (H.set (w := { v with streams := v.streams ++ [(stream, parent, 0)] }) ⟨z, rfl⟩)
    fun _ t ht => inv_like ht ?_
  exact ⟨hv.sent, hv.hello, hv.committed, hv.linked, hv.finished⟩

theorem step_finish (s : Schedule) (v : State) (hv : Inv s v) : H (Like v) (step s .finish) (fun _ t => Inv s t) := by
  rw [step]
  refine H.get_like fun z => ?_
  refine H.ite (fun _ => H.throw_bind) fun hf => ?_
  refine H.ite (fun _ => H.throw_bind) fun hok => ?_
  refine H.bind H.lift fun _ => H.intro fun _ => ?_
  refine H.ite (fun _ => H.throw_bind) fun hr => ?_
  refine H.bind H.lift fun _ => H.intro fun _ => ?_
  refine H.bind (H.set (w := { v with finished := true }) ⟨z, rfl⟩) fun _ => ?_
  refine H.conseq (H.send _) fun _ t ht => inv_like ht ?_
  have hf : v.finished = false := by simpa using hf
  have hok : v.committed = true ∧ v.linked = true := by simpa using hok
  have hr : v.rounds = s.rounds.size := by simpa using hr
  refine ⟨hv.sent_push rfl ?_, hv.hello, hv.committed, hv.linked, fun _ => ⟨hok.2, hr⟩⟩
  simp [sofar, hv.hello hok.1, hok.1, hok.2, hf, blank]

theorem roundsT_succ (s : Schedule) (r : Nat) : roundsT s (r + 1) = roundsT s r ++ roundT s r := by
  simp only [roundsT, List.range_succ, List.flatMap_append, List.flatMap_cons, List.flatMap_nil, List.append_nil]

theorem step_round (s : Schedule) (stream : String) (content : ByteArray) (n : Nat) (v : State) (hv : Inv s v) :
    H (Like v) (step s (.round stream content n)) (fun _ t => Inv s t) := by
  rw [step]
  refine H.get_like fun _ => ?_
  refine H.ite (fun _ => H.throw_bind) fun hf => ?_
  refine H.get_like fun z => ?_
  rw [show ({ v with salts := z } : State).streams = v.streams from rfl]
  generalize List.find? _ v.streams = o
  rcases o with _ | ⟨w, parent, k⟩
  · exact H.throw
  refine H.bind H.lift fun sp => H.intro fun _ => ?_
  refine H.ite (fun _ => H.throw_bind) fun _ => ?_
  refine H.get_like fun z => ?_
  rw [show ({ v with salts := z } : State).roots = v.roots from rfl]
  refine H.ite_jp (fun w => w = v) (fun w hw => ?_) (fun _ => rfl) fun _ hJ => ?_
  · subst hw
    refine H.get_like fun z => ?_
    rw [show ({ w with salts := z } : State).rounds = w.rounds from rfl]
    refine H.ite (fun _ => H.throw_bind) fun _ => ?_
    refine H.bind (H.mapM Firewall.saltedTop blankTop (fun c => blanks (c.size / NODE_BYTES)) H.saltedTop _)
      fun tops => H.intro fun hT => ?_
    refine H.bind (H.mapM_like _ (fun c => ?_) _) fun _ => ?_
    · refine H.bind H.lift fun _ => H.intro fun _ => ?_
      exact H.pure fun _ h => h
    refine H.bind H.lift fun _ => H.intro fun _ => ?_
    refine H.ite (fun _ => H.throw_bind) fun hpin => ?_
    refine H.bind (H.mapM Firewall.salted (fun _ => ByteArray.empty) (fun _ => ByteArray.empty)
      (fun row => H.conseq (H.salted row) fun _ _ h => ⟨rfl, h⟩) _) fun bcs => H.intro fun hB => ?_
    refine H.get_like fun z => ?_
    refine H.ite (fun _ => H.throw_bind) fun hl => ?_
    have hf : w.finished = false := by simpa using hf
    have hl : w.linked = true := by simpa using hl
    have hpin := Decidable.of_not_not hpin
    have hB : blankTop bcs = blanks (messageRows sp.free).length := by
      have := congrArg List.length hB
      simp only [List.length_map] at this
      rw [blankTop_eq, this]
    refine H.ite_jp (fun u => ∃ x, u = { w with sent := x } ∧
        x.toList.map blank = w.sent.toList.map blank ++ (if k == 0 then openT s stream else []))
      (fun u hu => ?_) (fun hk => ⟨w.sent, rfl, by simp [hk]⟩) fun hk hJ => ?_
    · obtain ⟨x, rfl, hx⟩ := hu
      refine H.bind (H.modify
        { w with sent := x, rounds := w.rounds + 1,
                 streams := w.streams.map fun e => if e.1 == stream then (e.1, e.2.1, e.2.2 + 1) else e }
        fun z => ⟨z, rfl⟩) fun _ => ?_
      refine H.conseq (H.send _) fun _ t ht => inv_like ht ?_
      refine ⟨?_, hv.hello, hv.committed, fun _ => hl, fun h => (nomatch hf.symm.trans h)⟩
      show (x.push _).toList.map blank = sofar s { w with rounds := w.rounds + 1 }
      rw [Array.toList_push, List.map_append, hx, hv.sent]
      simp only [sofar, roundsT_succ, roundT, hpin, hf, Bool.false_eq_true, ↓reduceIte, List.append_nil,
        List.append_assoc, List.map_cons, List.map_nil, blank, hB, hT, List.map_map]
      rfl
    · generalize ho : List.find? _ s.streams = o
      rcases o with _ | ⟨w₁, wp, pos, w₂⟩
      · exact H.throw_bind
      refine H.bind (H.send _) fun _ => hJ _ ⟨_, rfl, ?_⟩
      simp [Array.toList_push, openT, ho, hk, blank]
  · generalize List.find? _ v.roots = o₁ at hJ ⊢
    generalize sp.caps[0]? = o₂
    rcases o₁ with _ | ⟨w₁, rootB⟩ <;> rcases o₂ with _ | cap
    · exact H.throw_bind
    · exact H.throw_bind
    · exact H.throw_bind
    · exact H.ite (fun _ => H.throw_bind) fun _ => hJ v rfl

/-- **`step` keeps the invariant**, from every state equal to one it holds at but for its salts. -/
theorem step_inv (s : Schedule) (req : Req) (v : State) (hv : Inv s v) :
    H (Like v) (step s req) (fun _ t => Inv s t) := by
  cases req with
  | hello text => exact step_hello s text v hv
  | link y => exact step_link s y v hv
  | ping => exact step_ping s v hv
  | openStream stream parent label pos => exact step_openStream s stream parent label pos v hv
  | round stream content n => exact step_round s stream content n v hv
  | proof => exact step_proof s v hv
  | finish => exact step_finish s v hv
  | register => exact step_register s v hv
  | commit rootF roots publics => exact step_commit s rootF roots publics v hv
  | prime => exact step_prime s v

theorem stepOf_inv (s : Schedule) (b : ByteArray) (v : State) (hv : Inv s v) :
    H (Like v) (stepOf s b) (fun _ t => Inv s t) := by
  unfold stepOf
  exact H.bind H.lift fun req => H.intro fun _ => step_inv s req v hv

/-! ## The loop -/

theorem inv_init (s : Schedule) (salts : List ByteArray) : Inv s ({ salts } : State) where
  sent := by simp [sofar, roundsT]
  hello := nofun
  committed := nofun
  linked := nofun
  finished := nofun

theorem loop_inv (s : Schedule) (reqs salts : List ByteArray) : Inv s (loop s reqs salts).1 := by
  refine (forIn_yield (fun r r' => r = r' ∧ Inv s r.1) (body s) ?_ reqs (({ salts } : State), false) _
    ⟨rfl, inv_init s salts⟩).2
  rintro b ⟨st, stp⟩ _ ⟨rfl, hr⟩
  cases stp
  · have H1 := stepOf_inv s b st hr st (like_refl st)
    simp only [body, Bool.false_eq_true, ↓reduceIte]
    revert H1
    rcases (stepOf s b).run st with e | ⟨⟨⟩, t⟩
    · exact fun _ => ⟨_, _, rfl, rfl, rfl, hr⟩
    · exact fun ht => ⟨_, _, rfl, rfl, rfl, ht⟩
  · exact ⟨_, _, rfl, rfl, rfl, hr⟩

/-- **A finished session sends the schedule's transcript**: blanked, what the ZK server sent on a session it finished is
`tmpl s s.tables`, Commit's tables in the schedule's order, the only order `step` takes them in. -/
theorem run_blank_tables (s : Schedule) (reqs ys : List ByteArray) (h : (run s reqs ys).2 = true) :
    (run s reqs ys).1.toList.map blank = tmpl s s.tables := by
  have hI := loop_inv s reqs ys
  rw [run_eq] at h ⊢
  have hf : (loop s reqs ys).1.finished = true := by
    simp only [Bool.and_eq_true] at h
    exact h.1
  obtain ⟨hl, hr⟩ := hI.finished hf
  have hc := hI.committed hl
  have e : (loop s reqs ys).1.sent.toList.map blank = tmpl s s.tables := by
    rw [hI.sent]
    simp [sofar, tmpl, hI.hello hc, hc, hl, hr, hf]
  exact e

/-- **A finished session's Commit names the schedule's tables, in its order**: the order is never the worker's. -/
theorem namesOf_run (s : Schedule) (reqs ys : List ByteArray) (h : (run s reqs ys).2 = true) :
    namesOf (run s reqs ys).1.toList = s.tables := by
  rw [← namesOf_blank, run_blank_tables s reqs ys h, namesOf_tmpl]

/-- `run_blank_tables` at the tables the session's Commit named, which are the schedule's (`namesOf_run`). -/
theorem run_blank (s : Schedule) (reqs ys : List ByteArray) (h : (run s reqs ys).2 = true) :
    (run s reqs ys).1.toList.map blank = tmpl s (namesOf (run s reqs ys).1.toList) := by
  rw [namesOf_run s reqs ys h]
  exact run_blank_tables s reqs ys h

/-! ## What the ZK server sends, at every salt list -/

/-- **One run fixes them all, from the schedule**: if at the salts `ys₀` the ZK server finishes a session whose Commit names
the tables `o` in order, its commit strings those of the 64-byte digests `D` under `ys₀`, then at every salt list `ys` of
that length it finishes and sends the schedule's transcript `tmpl s o` filled with `D`'s commit strings under `ys`. -/
theorem sends_of_one (s : Schedule) (reqs : List ByteArray) {nh : Nat} (o : List String) (D ys₀ : Fin nh → ByteArray)
    (hD : ∀ i, (D i).size = 64) (hfin : (run s reqs (List.ofFn ys₀)).2 = true)
    (ho : namesOf (run s reqs (List.ofFn ys₀)).1.toList = o)
    (hh : holesOf (run s reqs (List.ofFn ys₀)).1 = List.ofFn fun i => Hm96.Default512.commitString (D i) (ys₀ i))
    (ys : Fin nh → ByteArray) :
    run s reqs (List.ofFn ys) =
      ((fillAll (tmpl s o) (List.ofFn fun i => Hm96.Default512.commitString (D i) (ys i))).toArray, true) :=
  run_of_one s reqs (tmpl s o) D ys₀ hD hfin (by rw [run_blank s reqs _ hfin, ho]) hh ys

end FlockSoundness.Discharge.Recursion.FirewallTemplate
