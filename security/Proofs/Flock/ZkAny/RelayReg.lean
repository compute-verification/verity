module

public import Flock.ZkServer.Frames

/-! A rep's frame has one writer, its own session's rounds and coins, so between that session's requests only `Kept` updates of its reps come (`frameAt_own`). -/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock (SessionSpec Stmt F128)
open _root_.Flock.ZkServer.Frames (Sessions Rep)

def keyOf (p : Rep) : Nat × Nat × Nat := (p.sid, p.table, p.rep)

def keyIs (sid i r : Nat) (p : Rep) : Bool := p.sid == sid && p.table == i && p.rep == r

def frameAt (R : Sessions) (sid i r : Nat) : Option Rep := R.reps.find? (keyIs sid i r)

theorem find_frameAt (R : Sessions) (sid i r : Nat) : (R.find sid i r).map (·.2) = frameAt R sid i r := by
  unfold _root_.Flock.ZkServer.Frames.Sessions.find frameAt
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

theorem find_key {R : Sessions} {sid i r j : Nat} {p : Rep} (h : R.find sid i r = some (j, p)) :
    keyOf p = (sid, i, r) := by
  have hf : frameAt R sid i r = some p := by rw [← find_frameAt, h]; rfl
  exact keyOf_of_keyIs (Array.find?_some hf)

def startAt (R : Sessions) (q : Rep) : Sessions :=
  if (R.find q.sid q.table q.rep).isSome then R else { R with reps := R.reps.push q }

def keepOf (R : Sessions) (root : ByteArray) (r : Nat) (s : Array ByteArray) : Sessions :=
  if R.salts.any fun (x, q, _) => x == root && q == r then R else { R with salts := R.salts.push (root, r, s) }

def closeOf (R : Sessions) (sid : Nat) (roots : Array ByteArray) : Sessions :=
  { sessions := R.sessions.filter (·.sid != sid), reps := R.reps.filter (·.sid != sid),
    salts := R.salts.filter fun (root, _, _) => !roots.contains root,
    files := R.files.filter fun (root, _, _) => !roots.contains root,
    inner := R.inner.filter fun (root, _, _) => !roots.contains root }

structure FreshRep (q : Rep) : Prop where
  st : q.st = {}
  coins : q.coins = #[]
  alg : q.alg = #[]
  opening : q.opening = none
  opened : q.opened = false
  toOpening : q.toOpening = false

/-- An update that fails leaves the frames as they were. That every write of the loop's frames in `Session.lean` and
`Frames.lean` is one of these is held by reading the code, not proved. -/
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

inductive Kept (sid i r : Nat) : Sessions → Sessions → Prop
  | refl (R : Sessions) : Kept sid i r R R
  | step {R R' R'' : Sessions} : Kept sid i r R R' → KeptStep sid i r R' R'' → Kept sid i r R R''

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

theorem frameAt_put_other (R : Sessions) (p : Rep) {sid i r : Nat} (hk : keyOf p ≠ (sid, i, r)) :
    frameAt (R.put p) sid i r = frameAt R sid i r := by
  unfold frameAt _root_.Flock.ZkServer.Frames.Sessions.put
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

theorem frameAt_put_self (R : Sessions) (p : Rep) {sid i r : Nat} (hk : keyOf p = (sid, i, r))
    (hs : (frameAt R sid i r).isSome) : frameAt (R.put p) sid i r = some p := by
  unfold frameAt _root_.Flock.ZkServer.Frames.Sessions.put at *
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
    unfold _root_.Flock.ZkServer.Frames.Sessions.file at hf
    split at hf
    · cases hf
    · cases hf
      rfl
  | inner root r' j b hf =>
    unfold _root_.Flock.ZkServer.Frames.Sessions.innerRound at hf
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

theorem frameAt_kept {sid i r : Nat} {R R' : Sessions} (h : Kept sid i r R R') (hs : (frameAt R sid i r).isSome) :
    frameAt R' sid i r = frameAt R sid i r := by
  induction h with
  | refl => rfl
  | step _ hst ih =>
    rw [frameAt_keptStep hst (by rw [ih]; exact hs), ih]

theorem frameAt_own {sid i r : Nat} {R₁ R₂ R₃ : Sessions} {p : Rep} (hs : (frameAt R₁ sid i r).isSome)
    (hk : keyOf p = (sid, i, r)) (h₁₂ : Kept sid i r R₁ R₂) (h₂₃ : Kept sid i r (R₂.put p) R₃) :
    frameAt R₃ sid i r = some p := by
  have h2 : frameAt R₂ sid i r = frameAt R₁ sid i r := frameAt_kept h₁₂ hs
  have hp : frameAt (R₂.put p) sid i r = some p := frameAt_put_self R₂ p hk (by rw [h2]; exact hs)
  rw [frameAt_kept h₂₃ (by rw [hp]; rfl), hp]

theorem find_own {sid i r : Nat} {R₁ R₂ R₃ : Sessions} {p : Rep} (hs : (frameAt R₁ sid i r).isSome)
    (hk : keyOf p = (sid, i, r)) (h₁₂ : Kept sid i r R₁ R₂) (h₂₃ : Kept sid i r (R₂.put p) R₃) :
    (R₃.find sid i r).map (·.2) = some p := by
  rw [find_frameAt]
  exact frameAt_own hs hk h₁₂ h₂₃

theorem coinsStep_key {R : Sessions} {sid i r : Nat} {cs : Array F128} {p' : Rep}
    (h : _root_.Flock.ZkServer.Frames.coinsStep R sid i r cs = .ok p') : keyOf p' = (sid, i, r) := by
  unfold _root_.Flock.ZkServer.Frames.coinsStep at h
  rcases hf : R.find sid i r with _ | ⟨j, p⟩
  · simp only [hf] at h
    cases h
  · have hk := find_key hf
    simp only [hf] at h
    split at h
    · cases h
      exact hk
    all_goals cases h

theorem withAlg_key {s : _root_.Flock.ZkServer.Frames.Release} {l0 : _root_.Flock.ZkServer.Level0.Level0}
    {name : String} {p q : Rep} (h : _root_.Flock.ZkServer.Frames.withAlg s l0 name p = .ok q) :
    keyOf q = keyOf p := by
  unfold _root_.Flock.ZkServer.Frames.withAlg at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first
    | (cases h; done)
    | (cases h; rfl)

theorem placeOf_key {salts : Array (ByteArray × Nat × Array ByteArray)} {l0 : _root_.Flock.ZkServer.Level0.Level0}
    {s : _root_.Flock.ZkServer.Frames.Release} {p : Rep} {dom : ByteArray} {pl : _root_.Flock.ZkServer.Frame.Place}
    {n : Nat} {q : Rep} (h : _root_.Flock.ZkServer.Frames.placeOf salts l0 s p dom = .ok (pl, n, q)) :
    keyOf q = keyOf p := by
  unfold _root_.Flock.ZkServer.Frames.placeOf at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first
    | (cases h; done)
    | (cases h; rfl)

theorem placeOf_key' {salts : Array (ByteArray × Nat × Array ByteArray)} {l0 : _root_.Flock.ZkServer.Level0.Level0}
    {s : _root_.Flock.ZkServer.Frames.Release} {p : Rep} {dom : ByteArray}
    {v : _root_.Flock.ZkServer.Frame.Place × Nat × Rep}
    (h : _root_.Flock.ZkServer.Frames.placeOf salts l0 s p dom = .ok v) : keyOf v.2.2 = keyOf p :=
  placeOf_key (pl := v.1) (n := v.2.1) h

theorem throw_ne_ok {α : Type} {e : String} {v : α} : (throw e : Except String α) ≠ .ok v := fun h => by cases h

set_option maxHeartbeats 1000000 in
theorem roundPlace_key {R : Sessions} {L0 : _root_.Flock.ZkServer.Level0.Slots} {sid i r : Nat}
    {dom content : ByteArray} {n : Nat} {q : Rep} {pl : _root_.Flock.ZkServer.Frame.Place} {f : Bool}
    {nd : Option String}
    (h : _root_.Flock.ZkServer.Frames.roundPlace R L0 sid i r dom content n = .ok (q, pl, f, nd)) :
    keyOf q = (sid, i, r) := by
  unfold _root_.Flock.ZkServer.Frames.roundPlace at h
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

theorem roundStep_key {R : Sessions} {L0 : _root_.Flock.ZkServer.Level0.Slots} {sid i r : Nat}
    {dom content : ByteArray} {n : Nat} {p' : Rep}
    (h : (_root_.Flock.ZkServer.Frames.roundStep R L0 sid i r dom content n).put = some p') :
    keyOf p' = (sid, i, r) := by
  unfold _root_.Flock.ZkServer.Frames.roundStep at h
  repeat' (first | split at h | simp only at h)
  all_goals first
    | (cases h; done)
    | (cases h
       have hk := roundPlace_key (by assumption)
       first
         | exact hk
         | simpa [keyOf] using hk)

end Claims.ZKServer.Relay
