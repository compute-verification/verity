module

public import Proofs.Flock.ZkAny.RelayReg

/-!
# What a round reads of the loop's frames besides its rep's frame stays put

A round reads, besides its rep's frame (`RelayReg`), its session's entry (`Frames.roundPlace`'s, with the `Commit`'s
coins, `Sessions.link`'s) and the rep's salts for levels ≥ 1 (`Frames.placeOf`'s). Neither changes between the
session's requests:

* `StayStep`, `Stay`: the updates the loop's other tasks make between session `sid`'s requests, where the rep's table's
  level-0 root is `root`: any of `RegStep`'s but a `link` of `sid` (its own `Commit`'s) and an end of `sid` or of a
  session whose tables' level-0 roots contain `root`; and any run of them.
* `sessAt`, `sessAt_stay`: session `sid`'s entry, as `Frames.roundPlace` finds it, where it has one, is kept: a `begin`
  pushes behind it, and another session's `link` and end leave it.
* `saltsAt`, `saltsAt_stay`: the salts for levels ≥ 1 of a rep on the table of level-0 root `root`, once kept, are
  kept: `keepOf` keeps the first.

Two sessions' tables share no level-0 root where the roots bind their samples (hash collision resistance), so another
session's end leaves the rep's salts.
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock (SessionSpec Stmt F128)
open _root_.Flock.Firewall.Frames (Sessions Rep Release)

local instance lawfulBEqByteArrayStay : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := by show (a.data == a.data) = true; exact beq_self_eq_true _

/-- **Session `sid`'s entry** in the loop's frames `R`, as `Frames.roundPlace` finds it. -/
def sessAt (R : Sessions) (sid : Nat) : Option Release := R.sessions.find? (·.sid == sid)

/-- **The salts for levels ≥ 1 of rep `r` on the table of level-0 root `root`** in the loop's frames `R`, as
`Frames.placeOf` finds them. -/
def saltsAt (R : Sessions) (root : ByteArray) (r : Nat) : Option (Array ByteArray) :=
  (R.salts.find? fun x => x.1 == root && x.2.1 == r).map (·.2.2)

/-- **An update the loop's other tasks make between session `sid`'s requests**, the rep's table's level-0 root being
`root`: any of `RegStep`'s but `sid`'s own `link` and an end of `sid` or of a session with `root` among its roots. -/
inductive StayStep (sid : Nat) (root : ByteArray) : Sessions → Sessions → Prop
  | begin (R : Sessions) (sid' : Nat) (spec : SessionSpec) (sts : Array Stmt) :
      StayStep sid root R (R.begin sid' spec sts)
  | link (R : Sessions) (sid' : Nat) (c : Array F128) : sid' ≠ sid → StayStep sid root R (R.link sid' c)
  | start (R : Sessions) (q : Rep) : StayStep sid root R (startAt R q)
  | put (R : Sessions) (p : Rep) : StayStep sid root R (R.put p)
  | keep (R : Sessions) (root' : ByteArray) (r' : Nat) (s : Array ByteArray) :
      StayStep sid root R (keepOf R root' r' s)
  | file {R R' : Sessions} (root' : ByteArray) (r' : Nat) (b : ByteArray) : R.file root' r' b = .ok R' →
      StayStep sid root R R'
  | inner {R R' : Sessions} (root' : ByteArray) (r' j : Nat) (b : ByteArray) : R.innerRound root' r' j b = .ok R' →
      StayStep sid root R R'
  | close (R : Sessions) (sid' : Nat) (roots : Array ByteArray) : sid' ≠ sid → roots.contains root = false →
      StayStep sid root R (closeOf R sid' roots)

/-- Any run of those updates. -/
inductive Stay (sid : Nat) (root : ByteArray) : Sessions → Sessions → Prop
  | refl (R : Sessions) : Stay sid root R R
  | step {R R' R'' : Sessions} : Stay sid root R R' → StayStep sid root R' R'' → Stay sid root R R''

/-- Each `StayStep` is one of the loop's frames' updates. -/
theorem StayStep.regStep {sid : Nat} {root : ByteArray} {R R' : Sessions} (h : StayStep sid root R R') :
    RegStep R R' := by
  cases h with
  | begin => exact .begin _ _ _ _
  | link => exact .link _ _ _
  | start => exact .start _ _
  | put => exact .put _ _
  | keep => exact .keep _ _ _ _
  | file root' r' b hf => exact .file root' r' b hf
  | inner root' r' j b hf => exact .inner root' r' j b hf
  | close sid' roots _ _ => exact .close _ sid' roots

theorem find?_push_of_isSome {α : Type} {xs : Array α} {p : α → Bool} (a : α) (h : (xs.find? p).isSome) :
    (xs.push a).find? p = xs.find? p := by
  rw [Array.find?_push]
  rcases hf : xs.find? p with _ | x
  · rw [hf] at h
    cases h
  · rfl

theorem map_find?_self {α : Type} {xs : Array α} {p : α → Bool} {f : α → α}
    (h : ∀ x, p x = true → f x = x) : (xs.find? p).map f = xs.find? p := by
  rcases hf : xs.find? p with _ | x
  · rfl
  · simp [h x (Array.find?_some hf)]

theorem sessions_file {R R' : Sessions} {root : ByteArray} {r : Nat} {b : ByteArray} (h : R.file root r b = .ok R') :
    R'.sessions = R.sessions ∧ R'.salts = R.salts := by
  unfold _root_.Flock.Firewall.Frames.Sessions.file at h
  split at h
  · cases h
  · cases h
    exact ⟨rfl, rfl⟩

theorem sessions_innerRound {R R' : Sessions} {root : ByteArray} {r j : Nat} {b : ByteArray}
    (h : R.innerRound root r j b = .ok R') : R'.sessions = R.sessions ∧ R'.salts = R.salts := by
  unfold _root_.Flock.Firewall.Frames.Sessions.innerRound at h
  simp only at h
  split at h
  · cases h
  · split at h <;> (cases h; exact ⟨rfl, rfl⟩)

/-- **An update between the session's requests keeps its entry**, where it has one. -/
theorem sessAt_stayStep {sid : Nat} {root : ByteArray} {R R' : Sessions} (h : StayStep sid root R R')
    (hs : (sessAt R sid).isSome) : sessAt R' sid = sessAt R sid := by
  cases h with
  | begin sid' spec sts =>
    unfold sessAt at *
    show (R.sessions.push _).find? (·.sid == sid) = _
    exact find?_push_of_isSome _ hs
  | link sid' c hne =>
    unfold sessAt
    show (R.sessions.map fun (s : Release) => if s.sid == sid' then { s with link := some c } else s).find?
      (·.sid == sid) = _
    rw [Array.find?_map]
    have hc : ((·.sid == sid) ∘ fun (s : Release) => if s.sid == sid' then { s with link := some c } else s) =
        (·.sid == sid) := by
      funext s
      simp only [Function.comp]
      split <;> rfl
    rw [hc]
    refine map_find?_self fun x hx => ?_
    have hx' : x.sid = sid := by simpa using hx
    simp [hx', Ne.symm hne]
  | start q =>
    unfold startAt
    split <;> rfl
  | put => rfl
  | keep =>
    unfold keepOf
    split <;> rfl
  | file root' r' b hf =>
    unfold sessAt
    rw [(sessions_file hf).1]
  | inner root' r' j b hf =>
    unfold sessAt
    rw [(sessions_innerRound hf).1]
  | close sid' roots hne _ =>
    unfold sessAt closeOf
    simp only
    rw [Array.find?_filter]
    congr 1
    funext s
    show decide ((s.sid != sid') = true ∧ (s.sid == sid) = true) = (s.sid == sid)
    rcases hk : s.sid == sid with _ | _
    · simp
    · have hs' : s.sid = sid := by simpa using hk
      have hb : (s.sid != sid') = true := by simp [hs', Ne.symm hne]
      simp [hb]

/-- **Updates between the session's requests keep its entry**, where it has one. -/
theorem sessAt_stay {sid : Nat} {root : ByteArray} {R R' : Sessions} (h : Stay sid root R R')
    (hs : (sessAt R sid).isSome) : sessAt R' sid = sessAt R sid := by
  induction h with
  | refl => rfl
  | step _ hst ih =>
    rw [sessAt_stayStep hst (by rw [ih]; exact hs), ih]

/-- **An update between the session's requests keeps the rep's salts**, once kept. -/
theorem saltsAt_stayStep {sid : Nat} {root : ByteArray} {R R' : Sessions} (h : StayStep sid root R R') (r : Nat)
    (hs : (saltsAt R root r).isSome) : saltsAt R' root r = saltsAt R root r := by
  cases h with
  | begin => rfl
  | link => rfl
  | start q =>
    unfold startAt
    split <;> rfl
  | put => rfl
  | keep root' r' s =>
    unfold keepOf
    split
    · rfl
    · unfold saltsAt at *
      simp only
      rw [find?_push_of_isSome _ (by simpa using hs)]
  | file root' r' b hf =>
    unfold saltsAt
    rw [(sessions_file hf).2]
  | inner root' r' j b hf =>
    unfold saltsAt
    rw [(sessions_innerRound hf).2]
  | close sid' roots _ hroot =>
    unfold saltsAt closeOf
    simp only
    rw [Array.find?_filter]
    congr 2
    funext e
    rcases e with ⟨x, q, s⟩
    show decide ((!roots.contains x) = true ∧ (x == root && q == r) = true) = (x == root && q == r)
    rcases hk : (x == root && q == r) with _ | _
    · simp
    · have hx : x = root := by
        simp only [Bool.and_eq_true] at hk
        exact eq_of_beq hk.1
      rw [hx, hroot]
      rfl

/-- **Updates between the session's requests keep the rep's salts**, once kept. -/
theorem saltsAt_stay {sid : Nat} {root : ByteArray} {R R' : Sessions} (h : Stay sid root R R') (r : Nat)
    (hs : (saltsAt R root r).isSome) : saltsAt R' root r = saltsAt R root r := by
  induction h with
  | refl => rfl
  | step _ hst ih =>
    rw [saltsAt_stayStep hst r (by rw [ih]; exact hs), ih]

end Claims.ZKServer.Relay
