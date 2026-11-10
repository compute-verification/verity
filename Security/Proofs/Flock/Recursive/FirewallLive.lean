module

public import Proofs.Flock.Recursive.FirewallRun

/-!
# `flock-firewall live` is the ZK server's run

`flock-firewall live` (`Flock.Firewall.liveWith`) is a live session's send check: `rec_live --lean` relays each request
to it, and the challenger's live-coin part receives only what it sends. Its loop takes each request with the fresh salts read for it
(`liveStep`), keeps the salts a request didn't sample for the next, and stops at the first refusal; `liveRun` is that loop.

* `run_live`: a live session that doesn't stop sends what `Flock.Firewall.run` sends on its requests at the salts it
  read, in order, and finishes where `run` does.
* `run_append`: a run that finishes is the same run at more salts after its own.

`Flock.SecurityProofs.liveSends` puts them together with `FirewallSends`. The proof is one frame lemma, `stepOf_grow`:
a request that returns at a state returns at it with more salts after its own and more messages before its own,
sampling and sending the same (`Grow`).
-/

@[expose] public section

namespace FlockSoundness.Discharge.Recursion.FirewallLive

open Flock Flock.Firewall FirewallRun

/-- The state `u` with the salts `r` after its own and the messages `acc` before its own. -/
abbrev Ext (r : List ByteArray) (acc : Array Sent) (u : State) : State :=
  { u with salts := u.salts ++ r, sent := acc ++ u.sent }

/-! ## A request at more salts after and more messages before -/

section Grow

variable (r : List ByteArray) (acc : Array Sent)

/-- **`m'` grows `m`**: where `m` returns at a state, `m'` returns at that state extended by `r` and `acc`, with a result
`Q` relates to `m`'s. -/
def Grow {α α' : Type} (m : Fw α) (m' : Fw α') (Q : α → α' → Prop) : Prop :=
  ∀ u a t, m.run u = .ok (a, t) → ∃ a', m'.run (Ext r acc u) = .ok (a', Ext r acc t) ∧ Q a a'

variable {r acc}

theorem Grow.bind {α α' β β' : Type} {m : Fw α} {m' : Fw α'} {f : α → Fw β} {f' : α' → Fw β'}
    {Q : α → α' → Prop} {R : β → β' → Prop}
    (h : Grow r acc m m' Q) (hf : ∀ a a', Q a a' → Grow r acc (f a) (f' a') R) :
    Grow r acc (m >>= f) (m' >>= f') R := by
  intro u b t hm
  rw [StateT.run_bind] at hm
  rcases h1 : m.run u with e | ⟨a, t₁⟩ <;> rw [h1] at hm
  · cases hm
  · obtain ⟨a', h1', hQ⟩ := h u a t₁ h1
    obtain ⟨b', h2', hR⟩ := hf a a' hQ t₁ b t hm
    exact ⟨b', by rw [StateT.run_bind, h1']; exact h2', hR⟩

theorem Grow.pure {α α' : Type} {Q : α → α' → Prop} {a : α} {a' : α'} (h : Q a a') :
    Grow r acc (Pure.pure a) (Pure.pure a') Q := by
  intro u b t hm
  cases hm
  exact ⟨a', rfl, h⟩

theorem Grow.throw {α α' : Type} {Q : α → α' → Prop} {e e' : String} :
    Grow r acc (throw e : Fw α) (throw e' : Fw α') Q := by
  intro u b t hm
  cases hm

theorem Grow.throw_bind {α α' β β' : Type} {R : β → β' → Prop} {e e' : String} {f : α → Fw β} {f' : α' → Fw β'} :
    Grow r acc ((MonadExcept.throw e : Fw α) >>= f) ((MonadExcept.throw e' : Fw α') >>= f') R := by
  intro u b t hm
  cases hm

theorem Grow.pure_bind {α α' β β' : Type} {R : β → β' → Prop} {a : α} {a' : α'} {f : α → Fw β} {f' : α' → Fw β'}
    (h : Grow r acc (f a) (f' a') R) : Grow r acc (Pure.pure a >>= f) (Pure.pure a' >>= f') R :=
  Grow.bind (Grow.pure (Q := fun b b' => b = a ∧ b' = a') ⟨rfl, rfl⟩) fun _ _ hb => by
    obtain ⟨rfl, rfl⟩ := hb
    exact h

theorem Grow.ite {α α' : Type} {Q : α → α' → Prop} {c c' : Prop} [Decidable c] [Decidable c'] {a b : Fw α}
    {a' b' : Fw α'} (hc : c ↔ c') (ha : c → Grow r acc a a' Q) (hb : ¬c → Grow r acc b b' Q) :
    Grow r acc (if c then a else b) (if c' then a' else b') Q := by
  by_cases h : c
  · rw [ite_eq_left h, ite_eq_left (hc.mp h)]; exact ha h
  · rw [ite_eq_right h, ite_eq_right (mt hc.mpr h)]; exact hb h

/-- A `do` block's `if` with a join point `b` after it: `b` is shown once, and the branch `a` may use it. -/
theorem Grow.ite_jp {α α' : Type} {Q : α → α' → Prop} {c c' : Prop} [Decidable c] [Decidable c'] {a b : Fw α}
    {a' b' : Fw α'} (hc : c ↔ c') (hb : Grow r acc b b' Q) (ha : c → Grow r acc b b' Q → Grow r acc a a' Q) :
    Grow r acc (if c then a else b) (if c' then a' else b') Q :=
  Grow.ite hc (fun h => ha h hb) fun _ => hb

theorem Grow.get : Grow r acc (MonadState.get : Fw State) MonadState.get fun a a' => a' = Ext r acc a := by
  intro u a t hm
  cases hm
  exact ⟨_, rfl, rfl⟩

theorem Grow.set {s s' : State} (h : s' = Ext r acc s) :
    Grow r acc (MonadStateOf.set s : Fw PUnit) (MonadStateOf.set s') fun _ _ => True := by
  intro u a t hm
  cases hm
  exact ⟨_, by rw [h]; rfl, trivial⟩

theorem Grow.modify {f f' : State → State} (h : ∀ u, f' (Ext r acc u) = Ext r acc (f u)) :
    Grow r acc (modify f : Fw PUnit) (modify f') fun _ _ => True := by
  intro u a t hm
  cases hm
  exact ⟨_, by show Except.ok (_, f' (Ext r acc u)) = _; rw [h], trivial⟩

theorem Grow.send (m : Sent) : Grow r acc (Firewall.send m) (Firewall.send m) fun _ _ => True :=
  Grow.modify fun u => by
    show ({ u with salts := u.salts ++ r, sent := (acc ++ u.sent).push m } : State) =
      { u with salts := u.salts ++ r, sent := acc ++ u.sent.push m }
    rw [Array.push_append]

/-- A computation the ZK server's state doesn't enter (`decode`, `split`, `nodes`, `pinned`), lifted. -/
theorem Grow.lift {α : Type} {x : V α} : Grow r acc (liftM x : Fw α) (liftM x) Eq := by
  intro u a t hm
  cases x with
  | error e => cases hm
  | ok b =>
    cases hm
    exact ⟨_, rfl, rfl⟩

/-- **Salting a row**: at more salts after, the same salt is sampled, so the same commit string. -/
theorem Grow.salted (row : Array Nat) : Grow r acc (Firewall.salted row) (Firewall.salted row) Eq := by
  intro u a t hm
  rw [run_salted] at hm
  rw [run_salted]
  cases u with
  | mk sent hello committed roots linked streams rounds stopped finished salts =>
    cases salts with
    | nil => cases hm
    | cons y rest =>
      cases hm
      exact ⟨_, rfl, rfl⟩

theorem Grow.mapM {α β : Type} (f : α → Fw β) (hf : ∀ a, Grow r acc (f a) (f a) Eq) :
    ∀ l : List α, Grow r acc (l.mapM f) (l.mapM f) Eq
  | [] => by
    rw [List.mapM_nil]
    exact Grow.pure rfl
  | a :: l => by
    rw [List.mapM_cons]
    refine Grow.bind (hf a) fun b b' hb => ?_
    obtain rfl := hb
    refine Grow.bind (Grow.mapM f hf l) fun bs bs' hbs => ?_
    obtain rfl := hbs
    exact Grow.pure rfl

theorem Grow.saltedTop (top : ByteArray) : Grow r acc (Firewall.saltedTop top) (Firewall.saltedTop top) Eq :=
  Grow.mapM _ Grow.salted (topRows top)

/-! ## `step` at more salts after and more messages before -/

/-- What a request returns: nothing. -/
abbrev Done : Unit → Unit → Prop := fun _ _ => True

theorem step_ping (s : Schedule) : Grow r acc (step s .ping) (step s .ping) Done := by
  rw [step]
  exact Grow.pure trivial

theorem step_prime (s : Schedule) : Grow r acc (step s .prime) (step s .prime) Done := by
  rw [step]
  exact Grow.throw

theorem step_register (s : Schedule) : Grow r acc (step s .register) (step s .register) Done := by
  rw [step]
  refine Grow.bind Grow.get fun a a' h => ?_
  obtain rfl := h
  exact Grow.ite Iff.rfl (fun _ => Grow.throw) fun _ => Grow.pure trivial

theorem step_proof (s : Schedule) : Grow r acc (step s .proof) (step s .proof) Done := by
  rw [step]
  refine Grow.bind Grow.get fun a a' h => ?_
  obtain rfl := h
  exact Grow.ite Iff.rfl (fun _ => Grow.throw) fun _ => Grow.pure trivial

theorem step_hello (s : Schedule) (text : String) :
    Grow r acc (step s (.hello text)) (step s (.hello text)) Done := by
  rw [step]
  refine Grow.bind Grow.get fun a a' h => ?_
  obtain rfl := h
  refine Grow.ite Iff.rfl (fun _ => Grow.throw_bind) fun _ => ?_
  refine Grow.bind Grow.get fun a a' h => ?_
  obtain rfl := h
  refine Grow.ite Iff.rfl (fun _ => Grow.throw_bind) fun _ => ?_
  rcases hj : Flock.JsonParse.parse text with e | v
  · exact Grow.throw_bind
  refine Grow.pure_bind ?_
  refine Grow.ite Iff.rfl (fun _ => Grow.throw_bind) fun _ => ?_
  refine Grow.ite Iff.rfl (fun _ => Grow.throw_bind) fun _ => ?_
  refine Grow.ite Iff.rfl (fun _ => Grow.throw_bind) fun _ => ?_
  refine Grow.bind Grow.lift fun _ _ h => ?_
  obtain rfl := h
  refine Grow.bind (Grow.modify fun _ => rfl) fun _ _ _ => ?_
  exact Grow.send _

theorem step_commit (s : Schedule) (rootF : ByteArray) (roots publics : List (String × ByteArray)) :
    Grow r acc (step s (.commit rootF roots publics)) (step s (.commit rootF roots publics)) Done := by
  rw [step]
  refine Grow.bind Grow.get fun a a' h => ?_
  obtain rfl := h
  refine Grow.ite Iff.rfl (fun _ => Grow.throw_bind) fun _ => ?_
  refine Grow.ite Iff.rfl (fun _ => Grow.throw_bind) fun _ => ?_
  refine Grow.ite Iff.rfl (fun _ => Grow.throw_bind) fun _ => ?_
  refine Grow.bind (Grow.mapM _ (fun e => ?_) _) fun c c' h => ?_
  · obtain ⟨n, v⟩ := e
    refine Grow.bind Grow.lift fun k k' h => ?_
    obtain rfl := h
    exact Grow.pure rfl
  obtain rfl := h
  refine Grow.bind Grow.lift fun _ _ h => ?_
  obtain rfl := h
  refine Grow.ite Iff.rfl (fun _ => Grow.throw_bind) fun _ => ?_
  refine Grow.ite Iff.rfl (fun _ => Grow.throw_bind) fun _ => ?_
  refine Grow.bind (Grow.mapM _ (fun e => ?_) _) fun t t' h => ?_
  · obtain ⟨n, v⟩ := e
    refine Grow.bind (Grow.saltedTop v) fun l l' h => ?_
    obtain rfl := h
    exact Grow.pure rfl
  obtain rfl := h
  refine Grow.bind Grow.get fun b b' h => ?_
  obtain rfl := h
  refine Grow.ite Iff.rfl (fun _ => Grow.throw_bind) fun _ => ?_
  refine Grow.bind (Grow.set rfl) fun _ _ _ => ?_
  exact Grow.send _

theorem step_link (s : Schedule) (y : ByteArray) : Grow r acc (step s (.link y)) (step s (.link y)) Done := by
  rw [step]
  refine Grow.bind Grow.get fun a a' h => ?_
  obtain rfl := h
  refine Grow.ite Iff.rfl (fun _ => Grow.throw_bind) fun _ => ?_
  refine Grow.ite Iff.rfl (fun _ => Grow.throw_bind) fun _ => ?_
  refine Grow.ite Iff.rfl (fun _ => Grow.throw_bind) fun _ => ?_
  refine Grow.bind Grow.get fun b b' h => ?_
  obtain rfl := h
  refine Grow.ite Iff.rfl (fun _ => Grow.throw_bind) fun _ => ?_
  refine Grow.bind (Grow.set rfl) fun _ _ _ => ?_
  exact Grow.send _

theorem step_openStream (s : Schedule) (stream parent : String) (label : ByteArray) (pos : Nat) :
    Grow r acc (step s (.openStream stream parent label pos)) (step s (.openStream stream parent label pos)) Done := by
  rw [step]
  refine Grow.bind Grow.get fun a a' h => ?_
  obtain rfl := h
  refine Grow.ite Iff.rfl (fun _ => Grow.throw_bind) fun _ => ?_
  generalize List.find? _ s.streams = o
  rcases o with _ | ⟨w₁, wp, wa, w₂⟩
  · exact Grow.throw
  refine Grow.ite Iff.rfl (fun _ => Grow.throw_bind) fun _ => ?_
  refine Grow.bind Grow.get fun b b' h => ?_
  obtain rfl := h
  refine Grow.ite Iff.rfl (fun _ => Grow.throw_bind) fun _ => ?_
  exact Grow.set rfl

theorem step_finish (s : Schedule) : Grow r acc (step s .finish) (step s .finish) Done := by
  rw [step]
  refine Grow.bind Grow.get fun a a' h => ?_
  obtain rfl := h
  refine Grow.ite Iff.rfl (fun _ => Grow.throw_bind) fun _ => ?_
  refine Grow.ite Iff.rfl (fun _ => Grow.throw_bind) fun _ => ?_
  refine Grow.bind Grow.lift fun _ _ h => ?_
  obtain rfl := h
  refine Grow.ite Iff.rfl (fun _ => Grow.throw_bind) fun _ => ?_
  refine Grow.bind Grow.lift fun _ _ h => ?_
  obtain rfl := h
  refine Grow.bind (Grow.set rfl) fun _ _ _ => ?_
  exact Grow.send _

theorem step_round (s : Schedule) (stream : String) (content : ByteArray) (n : Nat) :
    Grow r acc (step s (.round stream content n)) (step s (.round stream content n)) Done := by
  rw [step]
  refine Grow.bind Grow.get fun a a' h => ?_
  obtain rfl := h
  refine Grow.ite Iff.rfl (fun _ => Grow.throw_bind) fun _ => ?_
  refine Grow.bind Grow.get fun b b' h => ?_
  obtain rfl := h
  rw [show (Ext r acc b).streams = b.streams from rfl]
  generalize List.find? _ b.streams = o
  rcases o with _ | ⟨w, parent, k⟩
  · exact Grow.throw
  refine Grow.bind Grow.lift fun sp sp' h => ?_
  obtain rfl := h
  refine Grow.ite Iff.rfl (fun _ => Grow.throw_bind) fun _ => ?_
  refine Grow.bind Grow.get fun c c' h => ?_
  obtain rfl := h
  rw [show (Ext r acc c).roots = c.roots from rfl]
  refine Grow.ite_jp Iff.rfl ?_ fun _ hJ => ?_
  · refine Grow.bind Grow.get fun d d' h => ?_
    obtain rfl := h
    refine Grow.ite Iff.rfl (fun _ => Grow.throw_bind) fun _ => ?_
    refine Grow.bind (Grow.mapM Firewall.saltedTop Grow.saltedTop _) fun tops tops' h => ?_
    obtain rfl := h
    refine Grow.bind (Grow.mapM _ (fun e => ?_) _) fun tn tn' h => ?_
    · refine Grow.bind Grow.lift fun k k' h => ?_
      obtain rfl := h
      exact Grow.pure rfl
    obtain rfl := h
    refine Grow.bind Grow.lift fun _ _ h => ?_
    obtain rfl := h
    refine Grow.ite Iff.rfl (fun _ => Grow.throw_bind) fun _ => ?_
    refine Grow.bind (Grow.mapM Firewall.salted Grow.salted _) fun bcs bcs' h => ?_
    obtain rfl := h
    refine Grow.bind Grow.get fun e e' h => ?_
    obtain rfl := h
    refine Grow.ite Iff.rfl (fun _ => Grow.throw_bind) fun _ => ?_
    refine Grow.ite_jp Iff.rfl ?_ fun _ hJ' => ?_
    · refine Grow.bind (Grow.modify fun _ => rfl) fun _ _ _ => ?_
      exact Grow.send _
    · generalize List.find? _ s.streams = o
      rcases o with _ | ⟨w₁, w₂, pos, w₃⟩
      · exact Grow.throw_bind
      refine Grow.bind (Grow.send _) fun _ _ _ => ?_
      exact hJ'
  · generalize List.find? _ c.roots = o₁ at hJ ⊢
    generalize sp.caps[0]? = o₂
    rcases o₁ with _ | ⟨w₁, rootB⟩ <;> rcases o₂ with _ | cap
    · exact Grow.throw_bind
    · exact Grow.throw_bind
    · exact Grow.throw_bind
    · exact Grow.ite Iff.rfl (fun _ => Grow.throw_bind) fun _ => hJ

theorem step_grow (s : Schedule) (req : Req) : Grow r acc (step s req) (step s req) Done := by
  cases req with
  | hello text => exact step_hello s text
  | link y => exact step_link s y
  | ping => exact step_ping s
  | openStream stream parent label pos => exact step_openStream s stream parent label pos
  | round stream content n => exact step_round s stream content n
  | proof => exact step_proof s
  | finish => exact step_finish s
  | register => exact step_register s
  | commit rootF roots publics => exact step_commit s rootF roots publics
  | prime => exact step_prime s

/-- **A request grows**: one that returns at a state returns at it with more salts after its own and more messages
before its own, sampling and sending the same. -/
theorem stepOf_grow (s : Schedule) (b : ByteArray) : Grow r acc (stepOf s b) (stepOf s b) Done := by
  unfold stepOf
  refine Grow.bind Grow.lift fun req req' h => ?_
  obtain rfl := h
  exact step_grow s req

end Grow

/-! ## `run` at more salts -/

theorem body_stopped (s : Schedule) (b : ByteArray) (st : State) : body s b (st, true) = .yield (st, true) := rfl

theorem body_ok {s : Schedule} {b : ByteArray} {st t : State} (h : (stepOf s b).run st = .ok ((), t)) :
    body s b (st, false) = .yield (t, false) := by
  simp only [body, Bool.false_eq_true, ↓reduceIte, h]
  rfl

theorem body_error {s : Schedule} {b : ByteArray} {st : State} {e : String} (h : (stepOf s b).run st = .error e) :
    body s b (st, false) = .yield (st, true) := by
  simp only [body, Bool.false_eq_true, ↓reduceIte, h]
  rfl

theorem forIn_stopped (s : Schedule) : ∀ (reqs : List ByteArray) (st : State),
    Id.run (forIn reqs (st, true) (body s)) = (st, true)
  | [], _ => rfl
  | b :: l, st => by
    rw [List.forIn_cons, body_stopped]
    exact forIn_stopped s l st

/-- **A loop that doesn't stop grows**: from its state extended by `r` and `acc`, it runs to its own end extended. -/
theorem loop_grow (s : Schedule) (r : List ByteArray) : ∀ (reqs : List ByteArray) (st : State) (acc : Array Sent),
    (Id.run (forIn reqs (st, false) (body s))).2 = false →
    Id.run (forIn reqs (Ext r acc st, false) (body s)) = (Ext r acc (Id.run (forIn reqs (st, false) (body s))).1, false)
  | [], _, _, _ => rfl
  | b :: l, st, acc, h => by
    rw [List.forIn_cons] at h
    rw [List.forIn_cons, List.forIn_cons]
    rcases hm : (stepOf s b).run st with e | ⟨⟨⟩, t⟩
    · rw [body_error hm] at h
      exact absurd ((congrArg Prod.snd (forIn_stopped s l st)).symm.trans h) Bool.noConfusion
    · obtain ⟨⟨⟩, hm', -⟩ := stepOf_grow (r := r) (acc := acc) s b st () t hm
      rw [body_ok hm] at h
      rw [body_ok hm, body_ok hm']
      exact loop_grow s r l t acc h

/-- **A finished run samples nothing past its salts**: at the salts `ys` then any `zs`, it is the run at `ys`. -/
theorem run_append (s : Schedule) (reqs ys zs : List ByteArray) (h : (run s reqs ys).2 = true) :
    run s reqs (ys ++ zs) = run s reqs ys := by
  rw [run_eq] at h
  rw [run_eq, run_eq]
  have hs : (loop s reqs ys).2 = false := by
    revert h
    cases (loop s reqs ys).2 <;> simp
  have e : (loop s reqs (ys ++ zs)) = (Ext zs #[] (loop s reqs ys).1, false) := by
    have e0 : ({ salts := ys ++ zs } : State) = Ext zs #[] { salts := ys } := by
      simp only [Ext, Array.append_empty]
    unfold loop
    rw [e0]
    exact loop_grow s zs reqs { salts := ys } #[] hs
  rw [e, hs]
  simp only [Array.empty_append]

/-! ## The live loop -/

/-- **A live session's loop** (`liveWith`'s): from the state `st`, each request with the fresh salts read for it, through
`liveStep`, stopping at the first refusal: what it sent, in order, and the state it ends at, or the refusal. -/
def liveRun (s : Schedule) : State → List (ByteArray × List ByteArray) → Except String (List Sent × State)
  | st, [] => .ok ([], st)
  | st, (b, fresh) :: rest =>
    match liveStep s st b fresh with
    | .ok (sent, _, st') =>
      match liveRun s st' rest with
      | .ok (more, t) => .ok (sent ++ more, t)
      | .error e => .error e
    | .error e => .error e

theorem liveStep_ok {s : Schedule} {st : State} {b : ByteArray} {fresh : List ByteArray} {sent : List Sent}
    {used : List ByteArray} {st' : State} (h : liveStep s st b fresh = .ok (sent, used, st')) :
    ∃ t, (stepOf s b).run { st with salts := st.salts ++ fresh, sent := #[] } = .ok ((), t) ∧ sent = t.sent.toList ∧
      st' = { t with sent := #[] } := by
  unfold liveStep at h
  change (match (stepOf s b).run { st with salts := st.salts ++ fresh, sent := #[] } with
    | .ok ((), t) => Except.ok (t.sent.toList, (st.salts ++ fresh).take ((st.salts ++ fresh).length - t.salts.length),
        { t with sent := #[] })
    | .error e => .error e) = _ at h
  rcases hm : (stepOf s b).run { st with salts := st.salts ++ fresh, sent := #[] } with e | ⟨⟨⟩, t⟩ <;> rw [hm] at h
  · cases h
  · cases h
    exact ⟨t, rfl, rfl, rfl⟩

theorem liveRun_cons {s : Schedule} {st : State} {b : ByteArray} {fresh : List ByteArray}
    {rest : List (ByteArray × List ByteArray)} {sent : List Sent} {t : State}
    (h : liveRun s st ((b, fresh) :: rest) = .ok (sent, t)) :
    ∃ sent₁ used st' more, liveStep s st b fresh = .ok (sent₁, used, st') ∧ liveRun s st' rest = .ok (more, t) ∧
      sent = sent₁ ++ more := by
  simp only [liveRun] at h
  split at h
  · rename_i sent₁ used st' h1
    split at h
    · rename_i more t' h2
      cases h
      exact ⟨sent₁, used, st', more, h1, h2, rfl⟩
    · cases h
  · cases h

theorem liveRun_sent : ∀ {s : Schedule} {st : State} {L : List (ByteArray × List ByteArray)} {sent : List Sent}
    {t : State}, st.sent = #[] → liveRun s st L = .ok (sent, t) → t.sent = #[]
  | _, _, [], _, _, h0, h => by
    cases h
    exact h0
  | _, _, (_, _) :: _, _, _, _, h => by
    obtain ⟨_, _, _, _, h1, h2, -⟩ := liveRun_cons h
    obtain ⟨_, -, -, rfl⟩ := liveStep_ok h1
    exact liveRun_sent rfl h2

/-- `run`'s loop at the salts a live session read, from the live loop's state with the salts read after it and the
messages it sent before it: it ends at the live loop's end, with the messages it sent after them. -/
theorem loop_live (s : Schedule) : ∀ (L : List (ByteArray × List ByteArray)) (st : State) (acc : Array Sent)
    (sent : List Sent) (t : State), st.sent = #[] → liveRun s st L = .ok (sent, t) →
    Id.run (forIn (L.map (·.1)) (Ext (L.flatMap (·.2)) acc st, false) (body s)) =
      (Ext [] (acc ++ sent.toArray) t, false)
  | [], st, acc, sent, t, _, h => by
    cases h
    rw [show ([] : List Sent).toArray = #[] from rfl, Array.append_empty]
    rfl
  | (b, fresh) :: rest, st, acc, sent, t, h0, h => by
    obtain ⟨sent₁, used, st', more, h1, h2, rfl⟩ := liveRun_cons h
    obtain ⟨t₁, hm, rfl, rfl⟩ := liveStep_ok h1
    obtain ⟨⟨⟩, hm', -⟩ := stepOf_grow (r := rest.flatMap (·.2)) (acc := acc) s b _ () t₁ hm
    have e0 : Ext (((b, fresh) :: rest).flatMap (·.2)) acc st =
        Ext (rest.flatMap (·.2)) acc { st with salts := st.salts ++ fresh, sent := #[] } := by
      simp only [Ext, List.flatMap_cons, List.append_assoc, h0, Array.append_empty]
    have e1 : Ext (rest.flatMap (·.2)) acc t₁ = Ext (rest.flatMap (·.2)) (acc ++ t₁.sent) { t₁ with sent := #[] } := by
      simp only [Ext, Array.append_empty]
    rw [List.map_cons, List.forIn_cons, e0, body_ok hm', e1]
    refine (loop_live s rest { t₁ with sent := #[] } (acc ++ t₁.sent) more t rfl h2).trans ?_
    rw [← List.append_toArray, Array.toArray_toList, Array.append_assoc]

/-- **`flock-firewall live` is the ZK server's run**: a live session that doesn't stop, its requests each with the fresh
salts read for it, sends what `run` sends on its requests at those salts in order, and finishes where `run` does. -/
theorem run_live (s : Schedule) (L : List (ByteArray × List ByteArray)) (sent : List Sent) (t : State)
    (h : liveRun s { salts := [] } L = .ok (sent, t)) :
    run s (L.map (·.1)) (L.flatMap (·.2)) = (sent.toArray, t.finished) := by
  have e := loop_live s L { salts := [] } #[] sent t rfl h
  have e0 : ({ salts := L.flatMap (·.2) } : State) = Ext (L.flatMap (·.2)) #[] { salts := [] } := by
    simp only [Ext, List.nil_append, Array.append_empty]
  rw [run_eq]
  unfold loop
  rw [e0, e]
  simp only [liveRun_sent rfl h, Array.append_empty, Array.empty_append, Bool.not_false, Bool.and_true]

end FlockSoundness.Discharge.Recursion.FirewallLive
