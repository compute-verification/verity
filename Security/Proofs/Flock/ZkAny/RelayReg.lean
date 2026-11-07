module

public import Flock.Firewall.Frames

/-!
# A rep's frame has one writer: its own session's rounds and coins

The loop keeps every rep's frame in one registry, the loop's frames (`Frames.Sessions`), which several tasks write: each
session's own task (its `Hello`, its `Commit`'s coins, each round and its coins, its end) and the frames port's
connections (a rep's start, and the finish's salts for levels ≥ 1, inner rounds and proof files). The frames port's
`place` and `vals` only read it (`Frames.askOf`). A round and its coins read the rep's frame, step it and put it back
(`Frames.round`, `Frames.coins`), and the loop's other tasks may write the registry between that read and that put.

* `RegStep`: the registry's updates, each as the deployed code computes it: `Sessions.begin` at a `Hello`,
  `Sessions.link` at the `Commit`'s coins, a rep's start where it has none (`startAt`, `Frames.startRep`'s), a round's
  or its coins' put (`Sessions.put`), the finish's salts (`keepOf`, `Session.frames`'s `keep`), inner round
  (`Sessions.innerRound`) and proof file (`Sessions.file`), and a session's end (`closeOf`, `Frames.close`'s).
* `frameAt`, `find_frameAt`: a rep's frame in the registry, as `Sessions.find` finds it; `find_key`: the frame found at
  a rep's key has that key.
* `FreshRep`: a frame as `Frames.startRep` pushes it: its configuration and samples the start's, the rest unset.
* `KeptStep`, `Kept`: every update but a put at the rep's key and its own session's end, with a start's frame fresh,
  and any run of them. `frameAt_kept`: those leave a rep's frame as it was. `frameAt_put_self`: a put at the rep's key
  leaves its frame that put.
* `frameAt_own`, `find_own`: so where only `Kept` updates come between a request's read of the rep's frame and its put,
  and between that put and the next request's read, the next request reads the frame the first put back.
* `coinsStep_key`, `roundStep_key`: a round's coins, and a round, put back a frame at the rep's key (through
  `withAlg_key`, `placeOf_key` and `roundPlace_key`, which keep it).

Every write of the loop's frames in `Session.lean` and `Frames.lean` is one of `RegStep`'s (held by reading, as
`SlotsReach` holds the level-0 slots' writes), and only the session's own task puts a rep's frame, one request at a
time: so between that session's requests only `Kept` updates of its reps come.
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock (SessionSpec Stmt F128)
open _root_.Flock.Firewall.Frames (Sessions Rep)

/-- A rep's key in the loop's frames: its session, table and rep. -/
def keyOf (p : Rep) : Nat × Nat × Nat := (p.sid, p.table, p.rep)

/-- Whether `p` is the frame of session `sid`'s table `i`'s rep `r`, as `Sessions.find` tests it. -/
def keyIs (sid i r : Nat) (p : Rep) : Bool := p.sid == sid && p.table == i && p.rep == r

/-- **Session `sid`'s table `i`'s rep `r`'s frame** in the loop's frames `R`, if it has one. -/
def frameAt (R : Sessions) (sid i r : Nat) : Option Rep := R.reps.find? (keyIs sid i r)

/-- `Sessions.find` finds a rep's frame at `frameAt`. -/
theorem find_frameAt (R : Sessions) (sid i r : Nat) : (R.find sid i r).map (·.2) = frameAt R sid i r := by
  unfold _root_.Flock.Firewall.Frames.Sessions.find frameAt
  rcases h : R.reps.findIdx? (keyIs sid i r) with _ | j
  · have h' : R.reps.findIdx? (fun p => p.sid == sid && p.table == i && p.rep == r) = none := h
    have hn : R.reps.find? (keyIs sid i r) = none :=
      Array.find?_eq_none.mpr fun x hx => by simp [Array.findIdx?_eq_none_iff.mp h x hx]
    rw [hn, h']
    rfl
  · have h' : R.reps.findIdx? (fun p => p.sid == sid && p.table == i && p.rep == r) = some j := h
    obtain ⟨hj, hp, hlt⟩ := Array.findIdx?_eq_some_iff_getElem.mp h
    have hf : R.reps.find? (keyIs sid i r) = some R.reps[j] :=
      Array.find?_eq_some_iff_getElem.mpr ⟨hp, j, hj, rfl, fun k hk => by simpa using hlt k hk⟩
    rw [hf, h']
    simp [Array.getElem?_eq_getElem hj]

theorem keyOf_of_keyIs {sid i r : Nat} {p : Rep} (h : keyIs sid i r p = true) : keyOf p = (sid, i, r) := by
  simp only [keyIs, Bool.and_eq_true, beq_iff_eq] at h
  obtain ⟨⟨h1, h2⟩, h3⟩ := h
  simp [keyOf, h1, h2, h3]

/-- **The frame `Sessions.find` finds at a rep's key has that key.** -/
theorem find_key {R : Sessions} {sid i r j : Nat} {p : Rep} (h : R.find sid i r = some (j, p)) :
    keyOf p = (sid, i, r) := by
  have hf : frameAt R sid i r = some p := by rw [← find_frameAt, h]; rfl
  exact keyOf_of_keyIs (Array.find?_some hf)

/-- **A rep's start** where it has no frame (`Frames.startRep`'s update): the frame `q` pushed. -/
def startAt (R : Sessions) (q : Rep) : Sessions :=
  if (R.find q.sid q.table q.rep).isSome then R else { R with reps := R.reps.push q }

/-- **The finish's salts for levels ≥ 1** of rep `r` on the table of level-0 root `root` (`Session.frames`'s `keep`):
kept once. -/
def keepOf (R : Sessions) (root : ByteArray) (r : Nat) (s : Array ByteArray) : Sessions :=
  if R.salts.any fun (x, q, _) => x == root && q == r then R else { R with salts := R.salts.push (root, r, s) }

/-- **Session `sid`'s end** (`Frames.close`'s update), its tables' level-0 roots being `roots`. -/
def closeOf (R : Sessions) (sid : Nat) (roots : Array ByteArray) : Sessions :=
  { sessions := R.sessions.filter (·.sid != sid), reps := R.reps.filter (·.sid != sid),
    salts := R.salts.filter fun (root, _, _) => !roots.contains root,
    files := R.files.filter fun (root, _, _) => !roots.contains root,
    inner := R.inner.filter fun (root, _, _) => !roots.contains root }

/-- **A frame as `Frames.startRep` pushes it**: its configuration and samples the start's, its state the frame's start,
no scheduled rounds, coins or opening yet. -/
structure FreshRep (q : Rep) : Prop where
  st : q.st = {}
  coins : q.coins = #[]
  alg : q.alg = #[]
  opening : q.opening = none
  opened : q.opened = false
  toOpening : q.toOpening = false

/-- **The loop's frames' updates**, each as the deployed code computes it; an update that fails leaves them as they
were. -/
inductive RegStep : Sessions → Sessions → Prop
  | begin (R : Sessions) (sid : Nat) (spec : SessionSpec) (sts : Array Stmt) : RegStep R (R.begin sid spec sts)
  | link (R : Sessions) (sid : Nat) (c : Array F128) : RegStep R (R.link sid c)
  | start (R : Sessions) (q : Rep) : RegStep R (startAt R q)
  | put (R : Sessions) (p : Rep) : RegStep R (R.put p)
  | keep (R : Sessions) (root : ByteArray) (r : Nat) (s : Array ByteArray) : RegStep R (keepOf R root r s)
  | file {R R' : Sessions} (root : ByteArray) (r : Nat) (b : ByteArray) : R.file root r b = .ok R' → RegStep R R'
  | inner {R R' : Sessions} (root : ByteArray) (r j : Nat) (b : ByteArray) : R.innerRound root r j b = .ok R' →
      RegStep R R'
  | close (R : Sessions) (sid : Nat) (roots : Array ByteArray) : RegStep R (closeOf R sid roots)

/-- **An update that keeps session `sid`'s table `i`'s rep `r`'s frame**: any of `RegStep`'s but a put at the rep's key
and its own session's end, with a start's frame fresh (`FreshRep`, as `Frames.startRep` pushes one). -/
inductive KeptStep (sid i r : Nat) : Sessions → Sessions → Prop
  | begin (R : Sessions) (sid' : Nat) (spec : SessionSpec) (sts : Array Stmt) :
      KeptStep sid i r R (R.begin sid' spec sts)
  | link (R : Sessions) (sid' : Nat) (c : Array F128) : KeptStep sid i r R (R.link sid' c)
  | start (R : Sessions) (q : Rep) : FreshRep q → KeptStep sid i r R (startAt R q)
  | put (R : Sessions) (p : Rep) : keyOf p ≠ (sid, i, r) → KeptStep sid i r R (R.put p)
  | keep (R : Sessions) (root : ByteArray) (r' : Nat) (s : Array ByteArray) : KeptStep sid i r R (keepOf R root r' s)
  | file {R R' : Sessions} (root : ByteArray) (r' : Nat) (b : ByteArray) : R.file root r' b = .ok R' →
      KeptStep sid i r R R'
  | inner {R R' : Sessions} (root : ByteArray) (r' j : Nat) (b : ByteArray) : R.innerRound root r' j b = .ok R' →
      KeptStep sid i r R R'
  | close (R : Sessions) (sid' : Nat) (roots : Array ByteArray) : sid' ≠ sid →
      KeptStep sid i r R (closeOf R sid' roots)

/-- Any run of updates that keep the rep's frame. -/
inductive Kept (sid i r : Nat) : Sessions → Sessions → Prop
  | refl (R : Sessions) : Kept sid i r R R
  | step {R R' R'' : Sessions} : Kept sid i r R R' → KeptStep sid i r R' R'' → Kept sid i r R R''

/-- Each `KeptStep` is one of the loop's frames' updates. -/
theorem KeptStep.regStep {sid i r : Nat} {R R' : Sessions} (h : KeptStep sid i r R R') : RegStep R R' := by
  cases h with
  | begin => exact .begin _ _ _ _
  | link => exact .link _ _ _
  | start => exact .start _ _
  | put p _ => exact .put _ p
  | keep => exact .keep _ _ _ _
  | file root r' b hf => exact .file root r' b hf
  | inner root r' j b hf => exact .inner root r' j b hf
  | close sid' roots _ => exact .close _ sid' roots

theorem keyIs_put {p q : Rep} (h : (q.sid == p.sid && q.table == p.table && q.rep == p.rep) = true) (sid i r : Nat) :
    keyIs sid i r p = keyIs sid i r q := by
  simp only [Bool.and_eq_true, beq_iff_eq] at h
  obtain ⟨⟨h1, h2⟩, h3⟩ := h
  simp [keyIs, h1, h2, h3]

theorem keyIs_comp_put (p : Rep) (sid i r : Nat) :
    (keyIs sid i r ∘ fun q => if (q.sid == p.sid && q.table == p.table && q.rep == p.rep) = true then p else q) =
      keyIs sid i r := by
  funext q
  simp only [Function.comp]
  split
  · exact keyIs_put ‹_› sid i r
  · rfl

/-- **A put at another key keeps the rep's frame.** -/
theorem frameAt_put_other (R : Sessions) (p : Rep) {sid i r : Nat} (hk : keyOf p ≠ (sid, i, r)) :
    frameAt (R.put p) sid i r = frameAt R sid i r := by
  unfold frameAt _root_.Flock.Firewall.Frames.Sessions.put
  simp only
  rw [Array.find?_map, keyIs_comp_put]
  rcases hf : R.reps.find? (keyIs sid i r) with _ | q
  · rfl
  · have hq := Array.find?_some hf
    simp only [Option.map_some, Option.some.injEq]
    split
    · rename_i h
      exfalso
      apply hk
      simp only [keyIs, Bool.and_eq_true, beq_iff_eq] at hq h
      obtain ⟨⟨h1, h2⟩, h3⟩ := h
      obtain ⟨⟨h4, h5⟩, h6⟩ := hq
      simp [keyOf, ← h1, ← h2, ← h3, h4, h5, h6]
    · rfl

/-- **A put at the rep's key leaves its frame that put**, where it had one. -/
theorem frameAt_put_self (R : Sessions) (p : Rep) {sid i r : Nat} (hk : keyOf p = (sid, i, r))
    (hs : (frameAt R sid i r).isSome) : frameAt (R.put p) sid i r = some p := by
  unfold frameAt _root_.Flock.Firewall.Frames.Sessions.put at *
  simp only
  rw [Array.find?_map, keyIs_comp_put]
  rcases hf : R.reps.find? (keyIs sid i r) with _ | q
  · rw [hf] at hs
    cases hs
  · have hq := Array.find?_some hf
    simp only [keyOf, Prod.mk.injEq] at hk
    obtain ⟨h1, h2, h3⟩ := hk
    simp only [keyIs, Bool.and_eq_true, beq_iff_eq] at hq
    obtain ⟨⟨h4, h5⟩, h6⟩ := hq
    simp [h1, h2, h3, h4, h5, h6]

/-- **An update that keeps the rep's frame keeps it**, where it has one. -/
theorem frameAt_keptStep {sid i r : Nat} {R R' : Sessions} (h : KeptStep sid i r R R')
    (hs : (frameAt R sid i r).isSome) : frameAt R' sid i r = frameAt R sid i r := by
  cases h with
  | begin => rfl
  | link => rfl
  | start q =>
    unfold startAt
    split
    · rfl
    · unfold frameAt at *
      simp only
      rw [Array.find?_push]
      rcases hf : R.reps.find? (keyIs sid i r) with _ | x
      · rw [hf] at hs
        cases hs
      · rfl
  | put p hk => exact frameAt_put_other R p hk
  | keep =>
    unfold keepOf
    split <;> rfl
  | file root r' b hf =>
    unfold _root_.Flock.Firewall.Frames.Sessions.file at hf
    split at hf
    · cases hf
    · cases hf
      rfl
  | inner root r' j b hf =>
    unfold _root_.Flock.Firewall.Frames.Sessions.innerRound at hf
    simp only at hf
    split at hf
    · cases hf
    · split at hf <;> (cases hf; rfl)
  | close sid' roots hne =>
    unfold frameAt closeOf
    simp only
    rw [Array.find?_filter]
    congr 1
    funext q
    rcases hk : keyIs sid i r q with _ | _
    · simp
    · have hq : q.sid = sid := by
        simp only [keyIs, Bool.and_eq_true, beq_iff_eq] at hk
        exact hk.1.1
      have hb : (q.sid != sid') = true := by simp [hq, Ne.symm hne]
      simp [hb]

/-- **Updates that keep the rep's frame keep it**, where it has one. -/
theorem frameAt_kept {sid i r : Nat} {R R' : Sessions} (h : Kept sid i r R R') (hs : (frameAt R sid i r).isSome) :
    frameAt R' sid i r = frameAt R sid i r := by
  induction h with
  | refl => rfl
  | step _ hst ih =>
    rw [frameAt_keptStep hst (by rw [ih]; exact hs), ih]

/-- **The rep's next request reads what its last one put back**: where the rep has a frame when a request reads the
loop's frames (`R₁`), only updates that keep it come before the request puts back `p` at the rep's key (in `R₂`) and
after (to `R₃`), the rep's frame in `R₃` is `p`. -/
theorem frameAt_own {sid i r : Nat} {R₁ R₂ R₃ : Sessions} {p : Rep} (hs : (frameAt R₁ sid i r).isSome)
    (hk : keyOf p = (sid, i, r)) (h₁₂ : Kept sid i r R₁ R₂) (h₂₃ : Kept sid i r (R₂.put p) R₃) :
    frameAt R₃ sid i r = some p := by
  have h2 : frameAt R₂ sid i r = frameAt R₁ sid i r := frameAt_kept h₁₂ hs
  have hp : frameAt (R₂.put p) sid i r = some p := frameAt_put_self R₂ p hk (by rw [h2]; exact hs)
  rw [frameAt_kept h₂₃ (by rw [hp]; rfl), hp]

/-- `frameAt_own` as `Sessions.find` finds the frame. -/
theorem find_own {sid i r : Nat} {R₁ R₂ R₃ : Sessions} {p : Rep} (hs : (frameAt R₁ sid i r).isSome)
    (hk : keyOf p = (sid, i, r)) (h₁₂ : Kept sid i r R₁ R₂) (h₂₃ : Kept sid i r (R₂.put p) R₃) :
    (R₃.find sid i r).map (·.2) = some p := by
  rw [find_frameAt]
  exact frameAt_own hs hk h₁₂ h₂₃

/-- **A round's coins put back a frame at the rep's key.** -/
theorem coinsStep_key {R : Sessions} {sid i r : Nat} {cs : Array F128} {p' : Rep}
    (h : _root_.Flock.Firewall.Frames.coinsStep R sid i r cs = .ok p') : keyOf p' = (sid, i, r) := by
  unfold _root_.Flock.Firewall.Frames.coinsStep at h
  rcases hf : R.find sid i r with _ | ⟨j, p⟩
  · simp only [hf] at h
    cases h
  · have hk := find_key hf
    simp only [hf] at h
    split at h
    · cases h
      exact hk
    all_goals cases h

/-- `withAlg` keeps a rep's key. -/
theorem withAlg_key {s : _root_.Flock.Firewall.Frames.Release} {l0 : _root_.Flock.Firewall.Level0.Level0}
    {name : String} {p q : Rep} (h : _root_.Flock.Firewall.Frames.withAlg s l0 name p = .ok q) :
    keyOf q = keyOf p := by
  unfold _root_.Flock.Firewall.Frames.withAlg at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first
    | (cases h; done)
    | (cases h; rfl)

/-- `placeOf` keeps a rep's key. -/
theorem placeOf_key {salts : Array (ByteArray × Nat × Array ByteArray)} {l0 : _root_.Flock.Firewall.Level0.Level0}
    {s : _root_.Flock.Firewall.Frames.Release} {p : Rep} {dom : ByteArray} {pl : _root_.Flock.Firewall.Frame.Place}
    {n : Nat} {q : Rep} (h : _root_.Flock.Firewall.Frames.placeOf salts l0 s p dom = .ok (pl, n, q)) :
    keyOf q = keyOf p := by
  unfold _root_.Flock.Firewall.Frames.placeOf at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first
    | (cases h; done)
    | (cases h; rfl)

theorem placeOf_key' {salts : Array (ByteArray × Nat × Array ByteArray)} {l0 : _root_.Flock.Firewall.Level0.Level0}
    {s : _root_.Flock.Firewall.Frames.Release} {p : Rep} {dom : ByteArray}
    {v : _root_.Flock.Firewall.Frame.Place × Nat × Rep}
    (h : _root_.Flock.Firewall.Frames.placeOf salts l0 s p dom = .ok v) : keyOf v.2.2 = keyOf p :=
  placeOf_key (pl := v.1) (n := v.2.1) h

theorem throw_ne_ok {α : Type} {e : String} {v : α} : (throw e : Except String α) ≠ .ok v := fun h => by cases h

set_option maxHeartbeats 1000000 in
/-- **A round's frame is at its rep's key**: the frame `roundPlace` readies for session `sid`'s table `i`'s rep `r`'s
round. -/
theorem roundPlace_key {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i r : Nat}
    {dom content : ByteArray} {n : Nat} {q : Rep} {pl : _root_.Flock.Firewall.Frame.Place} {f : Bool}
    {nd : Option String}
    (h : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n = .ok (q, pl, f, nd)) :
    keyOf q = (sid, i, r) := by
  unfold _root_.Flock.Firewall.Frames.roundPlace at h
  rcases hf : R.find sid i r with _ | ⟨j, p⟩
  · simp only [hf] at h
    cases h
  have hk0 := find_key hf
  simp only [hf, bind, Except.bind, pure, Except.pure] at h
  by_cases hfst : (p.coins.isEmpty && !p.st.begun) = true
  all_goals repeat' (first
    | split at h
    | simp only [eq_true hfst, ↓reduceIte] at h
    | simp only [eq_false hfst, ↓reduceIte] at h
    | simp only at h)
  all_goals first
    | (cases h; done)
    | exact absurd (by assumption) throw_ne_ok
    | (cases h
       have e1 := placeOf_key' (by assumption)
       first
         | exact e1.trans hk0
         | (have e2 := withAlg_key (by assumption)
            exact e1.trans (e2.trans hk0)))

/-- **A round puts back a frame at its rep's key** (`Frames.round`'s put of `roundStep`'s frame). -/
theorem roundStep_key {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i r : Nat}
    {dom content : ByteArray} {n : Nat} {p' : Rep}
    (h : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom content n).put = some p') :
    keyOf p' = (sid, i, r) := by
  unfold _root_.Flock.Firewall.Frames.roundStep at h
  repeat' (first | split at h | simp only at h)
  all_goals first
    | (cases h; done)
    | (cases h
       have hk := roundPlace_key (by assumption)
       first
         | exact hk
         | simpa [keyOf] using hk)

end Claims.ZKServer.Relay
