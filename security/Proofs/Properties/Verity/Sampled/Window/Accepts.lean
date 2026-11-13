module

public import Proofs.Properties.Verity.Sampled.Record.Accepts
public import Proofs.Properties.Verity.Sampled.Moves

/-!
# One-stage v2's acceptance over several sessions is their verdicts, in order

`Spot.OneStageZk.run` folds `Spot.OneStageZk.step` over the evidence's sessions, each a `flock-verify verify --zk` at
the session's own record's sample, carrying a `Run` from one to the next: the coin-tree keys the earlier sessions
used, the verdicts so far, the units the accepted ones proved and whether every one accepted. The last run accepted
every session exactly where each session's verdict was `.ok`, in order:

* **What one session's `step` reads of its statement** (`StepAt`): the inputs, the verifier's registered reads, the
  scope, files and population checks, and the program's digest against the partition object's. A window whose members
  have their own statements has one per member; v2's `run` on one statement has the same one at every session.
* **One session's verdict after the run `r`** (`SessionVerdict`): its record is a blob, UTF-8, whose draw fits the
  statement's length words; it is set up at its record's own sample; its proof files are the blobs it names; its
  coin-tree key is fresh among the keys the earlier sessions used (`r.coinKeys`); `Flock.Zk.verify` accepts them; the
  whole of `verdictOf` is `.ok`; the verdict names the partition object's program; and its record proves its units.
* **The sessions' verdicts in order** (`Verdicts`): the run after each session is the run before the next, its coin
  keys and proved units grown by the session's.
* **`step_iff`, `steps_iff`**: a session's `step`, and a fold of `step` over the sessions, ends accepting every
  session exactly where the run before did and the verdicts hold; `verdicts_of_steps` is the direction a guarantee
  reads. **`run_verdicts_at`, `run_verdicts`**: so is v2's `run` where it accepted, at the one statement its
  parameters fix (`StepAt.ofOwn`, which reads no session's record, coins or sample), with the proved units held to
  the evidence's (`provedHeld`).
* **Each session's verdict is its own** (`Verdicts.getElem`): session `i`'s verdict is at its own statement, after the
  keys of the sessions before it. **Each session's coins are its own** (`Verdicts.keys_nodup`, `Verdicts.coins_own`):
  no two sessions' records name one coin-tree key.
-/

@[expose] public section

namespace Verity.Sampled

open Lean (Json)
open FlockSoundness.Refine (ebind_ok)

/-- An `Except` bind is `.ok` where its first step is and its continuation is at that step's value. -/
theorem ok_bind_of {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {a : α} {b : β} (hx : x = .ok a)
    (hf : f a = .ok b) : (x >>= f) = .ok b := by
  rw [hx]
  exact hf

/-! ## One session -/

/-- **What `Spot.OneStageZk.step` reads of one session's statement**: its inputs, the verifier's registered reads, the
scope, files and population checks its verdict runs, and its program's digest against the partition object's
program. -/
structure StepAt where
  I : Flock.Witness.Inputs
  rd : _root_.Spot.OneStageZk.Reads
  scope : Flock.V Unit
  ksFiles : Flock.V Unit
  population : Flock.V Unit
  programSha : Option String
  want : Json

/-- One session's `step` at the statement `a`, on the blobs `bs`, after the run `r`. -/
def StepAt.step (a : StepAt) (bs : _root_.Core.Blobs) (r : _root_.Spot.OneStageZk.Run)
    (s : _root_.Spot.OneStageZk.RunFiles) : Flock.V _root_.Spot.OneStageZk.Run :=
  _root_.Spot.OneStageZk.step a.I a.rd bs a.scope a.ksFiles a.population a.programSha a.want r s

/-- The coin-tree key the record `record` names at the statement `a`, set up at the record's own sample
(`Spot.OneStageZk.coinKeyOf` at `setupAt`): the key `step` holds fresh. -/
def StepAt.coinKey (a : StepAt) (record : String) : Option String :=
  _root_.Spot.OneStageZk.coinKeyOf record
    (_root_.Spot.OneStageZk.setupAt a.I a.rd (Flock.Witness.recordDraw record))

/-- **One session's verdict at the statement `a`, after the earlier sessions' coin-tree keys `keys`**: the session's
record is a blob, as UTF-8 `record`, whose draw fits the statement's length words; v2 sets it up at its record's own
sample as `(spec, tabs, rest)`; its proof files are the blobs it names, `proofs`; its coin-tree key, where its setup
flips coins from a coin tree, is none of `keys`; `Flock.Zk.verify` accepts the record and the proof files; `verdictOf`
is `.ok` on them; the verdict's program is the partition object's, as `push` compares them; and its record proves the
units `units`. -/
structure SessionVerdict (a : StepAt) (bs : _root_.Core.Blobs) (keys : Array String)
    (s : _root_.Spot.OneStageZk.RunFiles) (record : String) (spec : Flock.SessionSpec) (tabs : Array Flock.Setup)
    (rest : Json × Json) (proofs : Array ByteArray) (units : Array ℕ) : Prop where
  blob : ∃ B, bs.get s.record s!"session {s.name}'s record" = .ok B ∧ String.fromUTF8? B = some record
  fits : _root_.Spot.OneStageZk.drawFitsWhy _root_.Spot.OneStageZk.WORD (Flock.Witness.recordDraw record) = none
  setup : _root_.Spot.OneStageZk.setupAt a.I a.rd (Flock.Witness.recordDraw record) = .ok (spec, tabs, rest)
  files : _root_.Spot.OneStageZk.filesOf bs s (.ok (spec, tabs, rest)) = .ok proofs
  fresh : ∀ key, _root_.Spot.OneStageZk.coinKeyOf record (.ok (spec, tabs, rest)) = some key →
    Flock.CoinTree.checkFresh keys key = .ok ()
  verify : Flock.Zk.verify spec tabs record proofs = .ok ()
  verdict : _root_.Spot.OneStageZk.verdictOf a.I a.scope a.ksFiles a.population keys record
    (Flock.Witness.recordDraw record) (.ok (spec, tabs, rest)) (.ok proofs)
    (_root_.Spot.OneStageZk.coinKeyOf record (.ok (spec, tabs, rest))) = .ok ()
  program : (a.programSha.map Json.str != some a.want) = false
  units : _root_.Spot.OneStageZk.provedUnits a.I.publicFile record = .ok units

/-- The run after an accepted session: the session's coin-tree key among the keys, its verdict of record pushed, and
the units its record proved appended (`Spot.OneStageZk.push` at an `.ok` verdict). -/
def ranOn (a : StepAt) (r : _root_.Spot.OneStageZk.Run) (s : _root_.Spot.OneStageZk.RunFiles) (record : String)
    (spec : Flock.SessionSpec) (tabs : Array Flock.Setup) (rest : Json × Json) (units : Array ℕ) :
    _root_.Spot.OneStageZk.Run where
  coinKeys := match _root_.Spot.OneStageZk.coinKeyOf record (.ok (spec, tabs, rest)) with
    | some key => r.coinKeys.push key
    | none => r.coinKeys
  out := r.out.push (_root_.Spot.OneStageZk.verdictJson a.rd a.programSha s (.ok (spec, tabs, rest)) true "" units)
  proved := r.proved ++ units
  all := r.all

/-- **The sessions' verdicts, in order**, from the run `r` to the run `r'` on the blobs `bs`: each session, at its
statement, has its verdict after the keys of the run before it (`SessionVerdict`), and the run after it (`ranOn`) is
the run before the next. -/
inductive Verdicts (bs : _root_.Core.Blobs) :
    _root_.Spot.OneStageZk.Run → List (StepAt × _root_.Spot.OneStageZk.RunFiles) → _root_.Spot.OneStageZk.Run → Prop
  | nil (r : _root_.Spot.OneStageZk.Run) : Verdicts bs r [] r
  | cons {r r' : _root_.Spot.OneStageZk.Run} {a : StepAt} {s : _root_.Spot.OneStageZk.RunFiles}
      {l : List (StepAt × _root_.Spot.OneStageZk.RunFiles)} (record : String) (spec : Flock.SessionSpec)
      (tabs : Array Flock.Setup) (rest : Json × Json) (proofs : Array ByteArray) (units : Array ℕ) :
      SessionVerdict a bs r.coinKeys s record spec tabs rest proofs units →
      Verdicts bs (ranOn a r s record spec tabs rest units) l r' → Verdicts bs r ((a, s) :: l) r'

/-- **`push` keeps the run accepting exactly on an accepting verdict**: from a run that accepted so far, on a verdict
`.ok ()` naming the partition object's program, with the units the record proves. -/
theorem push_iff {I : Flock.Witness.Inputs} {rd : _root_.Spot.OneStageZk.Reads} {programSha : Option String}
    {want : Json} {r r' : _root_.Spot.OneStageZk.Run} {s : _root_.Spot.OneStageZk.RunFiles} {record : String}
    {st : _root_.Spot.OneStageZk.SetUp} {coinKey : Option String} {verdict : Flock.V Unit} :
    (_root_.Spot.OneStageZk.push I rd programSha want r s record st coinKey verdict = .ok r' ∧ r'.all = true) ↔
      r.all = true ∧ verdict = .ok () ∧ (programSha.map Json.str != some want) = false ∧
        ∃ units, _root_.Spot.OneStageZk.provedUnits I.publicFile record = .ok units ∧
          r' = { coinKeys := match coinKey with | some k => r.coinKeys.push k | none => r.coinKeys,
                 all := r.all, proved := r.proved ++ units,
                 out := r.out.push (_root_.Spot.OneStageZk.verdictJson rd programSha s st true "" units) } := by
  constructor
  · rintro ⟨h, hall⟩
    unfold _root_.Spot.OneStageZk.push at h
    cases verdict with
    | error why =>
      dsimp only at h
      simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      cases hall
    | ok u =>
      cases u
      dsimp only at h
      split at h
      · cases h
      · rename_i hne
        obtain ⟨units, hus, h⟩ := ebind_ok h
        obtain rfl := Except.ok.inj h
        exact ⟨hall, rfl, Bool.eq_false_iff.2 hne, units, hus, rfl⟩
  · rintro ⟨hr, hv, hp, units, hus, rfl⟩
    subst hv
    refine ⟨?_, hr⟩
    unfold _root_.Spot.OneStageZk.push
    dsimp only
    rw [ite_eq_right (Bool.eq_false_iff.1 hp)]
    exact ok_bind_of hus rfl

/-- **An `.ok` verdict held the session's coin-tree key fresh**: `verdictOf` checks the key against the earlier
sessions' before it verifies. -/
theorem verdictOf_fresh {I : Flock.Witness.Inputs} {scope ksFiles population : Flock.V Unit} {coinKeys : Array String}
    {record : String} {draw : Option Json} {st : _root_.Spot.OneStageZk.SetUp}
    {files : Flock.V (Array ByteArray)} {coinKey : Option String}
    (h : _root_.Spot.OneStageZk.verdictOf I scope ksFiles population coinKeys record draw st files coinKey
      = .ok ()) {key : String} (hk : coinKey = some key) : Flock.CoinTree.checkFresh coinKeys key = .ok () := by
  subst hk
  unfold _root_.Spot.OneStageZk.verdictOf at h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨⟨⟩, hf, -⟩ := ebind_ok h
  exact hf

/-- A key `checkFresh` passes is none of the keys it was checked against. -/
theorem not_contains_of_fresh {keys : Array String} {k : String} (h : Flock.CoinTree.checkFresh keys k = .ok ()) :
    keys.contains k = false := by
  unfold Flock.CoinTree.checkFresh at h
  cases hc : keys.contains k with
  | false => rfl
  | true => rw [hc] at h; cases h

/-- The run after an accepted session, with the session's setup as `step` computes it. -/
theorem SessionVerdict.ranOn_eq {a : StepAt} {bs : _root_.Core.Blobs} {keys : Array String}
    {s : _root_.Spot.OneStageZk.RunFiles} {record : String} {spec : Flock.SessionSpec} {tabs : Array Flock.Setup}
    {rest : Json × Json} {proofs : Array ByteArray} {units : Array ℕ}
    (h : SessionVerdict a bs keys s record spec tabs rest proofs units) (r : _root_.Spot.OneStageZk.Run) :
    ranOn a r s record spec tabs rest units =
      { coinKeys := match a.coinKey record with | some k => r.coinKeys.push k | none => r.coinKeys,
        all := r.all, proved := r.proved ++ units,
        out := r.out.push (_root_.Spot.OneStageZk.verdictJson a.rd a.programSha s
          (_root_.Spot.OneStageZk.setupAt a.I a.rd (Flock.Witness.recordDraw record)) true "" units) } := by
  unfold StepAt.coinKey
  rw [h.setup]
  rfl

/-- The session's coin-tree key at its statement is the key its setup names. -/
theorem SessionVerdict.coinKey_eq {a : StepAt} {bs : _root_.Core.Blobs} {keys : Array String}
    {s : _root_.Spot.OneStageZk.RunFiles} {record : String} {spec : Flock.SessionSpec} {tabs : Array Flock.Setup}
    {rest : Json × Json} {proofs : Array ByteArray} {units : Array ℕ}
    (h : SessionVerdict a bs keys s record spec tabs rest proofs units) :
    a.coinKey record = _root_.Spot.OneStageZk.coinKeyOf record (.ok (spec, tabs, rest)) := by
  unfold StepAt.coinKey
  rw [h.setup]

/-- **One session's `step` ends accepting every session exactly where the run before it did and the session has its
verdict**, its run then the run after it (`ranOn`). -/
theorem step_iff {a : StepAt} {bs : _root_.Core.Blobs} {r r' : _root_.Spot.OneStageZk.Run}
    {s : _root_.Spot.OneStageZk.RunFiles} :
    (a.step bs r s = .ok r' ∧ r'.all = true) ↔
      r.all = true ∧ ∃ record spec tabs rest proofs units,
        SessionVerdict a bs r.coinKeys s record spec tabs rest proofs units ∧
          r' = ranOn a r s record spec tabs rest units := by
  constructor
  · rintro ⟨h, hall⟩
    unfold StepAt.step _root_.Spot.OneStageZk.step at h
    obtain ⟨B, hB, h⟩ := ebind_ok h
    cases hu : String.fromUTF8? B with
    | none => rw [hu] at h; cases h
    | some record =>
      rw [hu] at h
      dsimp only at h
      obtain ⟨_, hfit, h⟩ := ebind_ok h
      try dsimp only at h
      obtain ⟨hr, hv, hp, units, hus, hr'⟩ := push_iff.1 ⟨h, hall⟩
      obtain ⟨proofs, spec, tabs, rest, hf, hst, hver⟩ := verdictOf_ok hv
      rw [hst] at hf hv hr'
      rw [hf] at hv
      refine ⟨hr, record, spec, tabs, rest, proofs, units,
        ⟨⟨B, hB, hu⟩, ?_, hst, hf, fun key hk => verdictOf_fresh hv hk, hver, hv, hp, hus⟩, hr'⟩
      cases hd : _root_.Spot.OneStageZk.drawFitsWhy _root_.Spot.OneStageZk.WORD
          (Flock.Witness.recordDraw record) with
      | none => rfl
      | some w => rw [hd] at hfit; cases hfit
  · rintro ⟨hr, record, spec, tabs, rest, proofs, units, hsv, rfl⟩
    obtain ⟨B, hB, hu⟩ := hsv.blob
    have hv : _root_.Spot.OneStageZk.verdictOf a.I a.scope a.ksFiles a.population r.coinKeys record
        (Flock.Witness.recordDraw record)
        (_root_.Spot.OneStageZk.setupAt a.I a.rd (Flock.Witness.recordDraw record))
        (_root_.Spot.OneStageZk.filesOf bs s
          (_root_.Spot.OneStageZk.setupAt a.I a.rd (Flock.Witness.recordDraw record)))
        (a.coinKey record) = .ok () := by
      unfold StepAt.coinKey
      rw [hsv.setup, hsv.files]
      exact hsv.verdict
    have hpush := (push_iff (I := a.I) (rd := a.rd) (programSha := a.programSha) (want := a.want) (r := r) (s := s)
      (record := record)
      (st := _root_.Spot.OneStageZk.setupAt a.I a.rd (Flock.Witness.recordDraw record))
      (coinKey := a.coinKey record)).2 ⟨hr, hv, hsv.program, units, hsv.units, rfl⟩
    rw [hsv.ranOn_eq r]
    refine ⟨?_, hpush.2⟩
    unfold StepAt.step _root_.Spot.OneStageZk.step
    refine ok_bind_of hB ?_
    try dsimp only
    rw [hu]
    try dsimp only
    refine ok_bind_of (a := ()) ?_ hpush.1
    rw [hsv.fits]
    try rfl

/-- A session's `step` keeps a run that didn't accept every session so: a step's run accepts every session only where
the run before it did. -/
theorem step_all {a : StepAt} {bs : _root_.Core.Blobs} {r r' : _root_.Spot.OneStageZk.Run}
    {s : _root_.Spot.OneStageZk.RunFiles} (h : a.step bs r s = .ok r') (hall : r'.all = true) : r.all = true :=
  (step_iff.1 ⟨h, hall⟩).1

/-- **One session's `step` that ends accepting every session is the session's verdict**, and its run the run after
it. -/
theorem verdict_of_step {a : StepAt} {bs : _root_.Core.Blobs} {r r' : _root_.Spot.OneStageZk.Run}
    {s : _root_.Spot.OneStageZk.RunFiles} (h : a.step bs r s = .ok r') (hall : r'.all = true) :
    ∃ record spec tabs rest proofs units, SessionVerdict a bs r.coinKeys s record spec tabs rest proofs units ∧
      r' = ranOn a r s record spec tabs rest units :=
  (step_iff.1 ⟨h, hall⟩).2

/-- **A session with its verdict passes `step`**, from a run that accepted so far to the run after it. -/
theorem step_of_verdict {a : StepAt} {bs : _root_.Core.Blobs} {r : _root_.Spot.OneStageZk.Run}
    {s : _root_.Spot.OneStageZk.RunFiles} {record : String} {spec : Flock.SessionSpec} {tabs : Array Flock.Setup}
    {rest : Json × Json} {proofs : Array ByteArray} {units : Array ℕ} (hr : r.all = true)
    (h : SessionVerdict a bs r.coinKeys s record spec tabs rest proofs units) :
    a.step bs r s = .ok (ranOn a r s record spec tabs rest units) :=
  (step_iff.2 ⟨hr, record, spec, tabs, rest, proofs, units, h, rfl⟩).1

/-! ## The sessions, in order -/

/-- **A fold of `step` over the sessions ends accepting every session exactly where the run it starts from did and
the sessions have their verdicts in order.** -/
theorem steps_iff {bs : _root_.Core.Blobs} :
    ∀ (l : List (StepAt × _root_.Spot.OneStageZk.RunFiles)) {r r' : _root_.Spot.OneStageZk.Run},
      (l.foldlM (fun r x => x.1.step bs r x.2) r = .ok r' ∧ r'.all = true) ↔ r.all = true ∧ Verdicts bs r l r'
  | [], r, r' => by
    constructor
    · rintro ⟨h, hall⟩
      obtain rfl : r = r' := Except.ok.inj h
      exact ⟨hall, .nil r⟩
    · rintro ⟨hall, h⟩
      cases h
      exact ⟨rfl, hall⟩
  | (a, s) :: l, r, r' => by
    constructor
    · rintro ⟨h, hall⟩
      rw [List.foldlM_cons] at h
      obtain ⟨r1, h1, h⟩ := ebind_ok h
      obtain ⟨hall1, hrest⟩ := (steps_iff l).1 ⟨h, hall⟩
      obtain ⟨hr, record, spec, tabs, rest, proofs, units, hsv, rfl⟩ := step_iff.1 ⟨h1, hall1⟩
      exact ⟨hr, .cons record spec tabs rest proofs units hsv hrest⟩
    · rintro ⟨hr, h⟩
      cases h with
      | cons record spec tabs rest proofs units hsv hrest =>
        have h1 := step_iff.2 ⟨hr, record, spec, tabs, rest, proofs, units, hsv, rfl⟩
        have h2 := (steps_iff l).2 ⟨h1.2, hrest⟩
        refine ⟨?_, h2.2⟩
        rw [List.foldlM_cons]
        exact ok_bind_of h1.1 h2.1

/-- **v2's acceptance over several sessions is their verdicts in order**: a fold of `step` over the sessions, each at
its statement, from the run `r`, that ends at a run accepting every session, is the sessions' verdicts in order. -/
theorem verdicts_of_steps {bs : _root_.Core.Blobs} {l : List (StepAt × _root_.Spot.OneStageZk.RunFiles)}
    {r r' : _root_.Spot.OneStageZk.Run} (h : l.foldlM (fun r x => x.1.step bs r x.2) r = .ok r')
    (hall : r'.all = true) : Verdicts bs r l r' :=
  ((steps_iff l).1 ⟨h, hall⟩).2

/-- **Sessions with their verdicts in order pass the fold of `step`**, from a run that accepted so far. -/
theorem steps_of_verdicts {bs : _root_.Core.Blobs} {l : List (StepAt × _root_.Spot.OneStageZk.RunFiles)}
    {r r' : _root_.Spot.OneStageZk.Run} (hr : r.all = true) (h : Verdicts bs r l r') :
    l.foldlM (fun r x => x.1.step bs r x.2) r = .ok r' ∧ r'.all = true :=
  (steps_iff l).2 ⟨hr, h⟩

/-- **The first session's verdict, then the rest's**: `Verdicts` at a first session, whatever its statement. -/
theorem verdicts_cons {bs : _root_.Core.Blobs} {x : StepAt × _root_.Spot.OneStageZk.RunFiles}
    {l : List (StepAt × _root_.Spot.OneStageZk.RunFiles)} {r r' : _root_.Spot.OneStageZk.Run} :
    Verdicts bs r (x :: l) r' ↔ ∃ record spec tabs rest proofs units,
      SessionVerdict x.1 bs r.coinKeys x.2 record spec tabs rest proofs units ∧
        Verdicts bs (ranOn x.1 r x.2 record spec tabs rest units) l r' := by
  obtain ⟨a, s⟩ := x
  constructor
  · intro h
    cases h with
    | cons record spec tabs rest proofs units hsv hrest =>
      exact ⟨record, spec, tabs, rest, proofs, units, hsv, hrest⟩
  · rintro ⟨record, spec, tabs, rest, proofs, units, hsv, hrest⟩
    exact .cons record spec tabs rest proofs units hsv hrest

/-- **Verdicts in order split at any point**: `l` then `l'` have their verdicts in order from `r` to `r'` exactly
where `l` has from `r` to some run and `l'` from it to `r'`. -/
theorem verdicts_append {bs : _root_.Core.Blobs} :
    ∀ (l : List (StepAt × _root_.Spot.OneStageZk.RunFiles)) {l' : List (StepAt × _root_.Spot.OneStageZk.RunFiles)}
      {r r' : _root_.Spot.OneStageZk.Run},
      Verdicts bs r (l ++ l') r' ↔ ∃ m, Verdicts bs r l m ∧ Verdicts bs m l' r'
  | [], l', r, r' => by
    constructor
    · intro h
      exact ⟨r, .nil r, h⟩
    · rintro ⟨m, hm, h⟩
      cases hm
      exact h
  | (a, s) :: l, l', r, r' => by
    show Verdicts bs r ((a, s) :: (l ++ l')) r' ↔ _
    constructor
    · intro h
      cases h with
      | cons record spec tabs rest proofs units hsv hrest =>
        obtain ⟨m, hm, hl'⟩ := (verdicts_append l).1 hrest
        exact ⟨m, .cons record spec tabs rest proofs units hsv hm, hl'⟩
    · rintro ⟨m, hm, hl'⟩
      cases hm with
      | cons record spec tabs rest proofs units hsv hrest =>
        exact .cons record spec tabs rest proofs units hsv ((verdicts_append l).2 ⟨m, hrest, hl'⟩)

/-- **Each session's verdict is its own**: of sessions with their verdicts in order, session `i` has its verdict at
its own statement, after the coin-tree keys of the run `ri` the `i` sessions before it end at, and the sessions after
it have theirs from the run after it. -/
theorem Verdicts.getElem {bs : _root_.Core.Blobs} {l : List (StepAt × _root_.Spot.OneStageZk.RunFiles)}
    {r r' : _root_.Spot.OneStageZk.Run} (h : Verdicts bs r l r') {i : ℕ} (hi : i < l.length) :
    ∃ ri record spec tabs rest proofs units, Verdicts bs r (l.take i) ri ∧
      SessionVerdict l[i].1 bs ri.coinKeys l[i].2 record spec tabs rest proofs units ∧
      Verdicts bs (ranOn l[i].1 ri l[i].2 record spec tabs rest units) (l.drop (i + 1)) r' := by
  have hsplit : l.take i ++ l[i] :: l.drop (i + 1) = l := by
    rw [← List.drop_eq_getElem_cons hi, List.take_append_drop]
  have h' : Verdicts bs r (l.take i ++ l[i] :: l.drop (i + 1)) r' := by
    rw [hsplit]
    exact h
  obtain ⟨ri, hpre, hrest⟩ := (verdicts_append (l.take i)).1 h'
  obtain ⟨record, spec, tabs, rest, proofs, units, hsv, hpost⟩ := verdicts_cons.1 hrest
  exact ⟨ri, record, spec, tabs, rest, proofs, units, hpre, hsv, hpost⟩

/-- **Verdicts in order keep the run's acceptance and only add coin-tree keys**: the run they end at accepted every
session where the run they start from did, and each key of the run before is a key of the run after. -/
theorem Verdicts.keys {bs : _root_.Core.Blobs} {l : List (StepAt × _root_.Spot.OneStageZk.RunFiles)}
    {r r' : _root_.Spot.OneStageZk.Run} (h : Verdicts bs r l r') :
    r'.all = r.all ∧ ∀ k, r.coinKeys.contains k = true → r'.coinKeys.contains k = true := by
  induction h with
  | nil r => exact ⟨rfl, fun _ hk => hk⟩
  | @cons r r' a s l record spec tabs rest proofs units _ _ ih =>
    refine ⟨ih.1, fun k hk => ih.2 k ?_⟩
    show (match _root_.Spot.OneStageZk.coinKeyOf record (.ok (spec, tabs, rest)) with
      | some key => r.coinKeys.push key
      | none => r.coinKeys).contains k = true
    cases _root_.Spot.OneStageZk.coinKeyOf record (.ok (spec, tabs, rest)) with
    | none => exact hk
    | some k' =>
      show (r.coinKeys.push k').contains k = true
      rw [Array.contains_push, hk]
      rfl

/-- **No two sessions flip their coins from one coin tree**: verdicts in order from a run whose coin-tree keys are
distinct end at a run whose keys are, each session's key held fresh among the earlier sessions'. -/
theorem Verdicts.keys_nodup {bs : _root_.Core.Blobs} {l : List (StepAt × _root_.Spot.OneStageZk.RunFiles)}
    {r r' : _root_.Spot.OneStageZk.Run} (h : Verdicts bs r l r') (hr : r.coinKeys.toList.Nodup) :
    r'.coinKeys.toList.Nodup := by
  induction h with
  | nil r => exact hr
  | @cons r r' a s l record spec tabs rest proofs units hsv _ ih =>
    apply ih
    show (match _root_.Spot.OneStageZk.coinKeyOf record (.ok (spec, tabs, rest)) with
      | some key => r.coinKeys.push key
      | none => r.coinKeys).toList.Nodup
    cases hk : _root_.Spot.OneStageZk.coinKeyOf record (.ok (spec, tabs, rest)) with
    | none => exact hr
    | some k =>
      show (r.coinKeys.push k).toList.Nodup
      have hnot : r.coinKeys.contains k = false := not_contains_of_fresh (hsv.fresh k hk)
      rw [Array.toList_push]
      refine List.pairwise_append.2 ⟨hr, List.pairwise_singleton _ _, fun x hx y hy => ?_⟩
      obtain rfl : y = k := List.mem_singleton.1 hy
      rintro rfl
      have hin : r.coinKeys.contains x = true := Array.contains_iff_mem.2 (Array.mem_toList_iff.1 hx)
      rw [hnot] at hin
      cases hin

/-- **A session's record is the blobs'**: a session's verdict is on the one record its files name. -/
theorem SessionVerdict.record_eq {a : StepAt} {bs : _root_.Core.Blobs} {keys : Array String}
    {s : _root_.Spot.OneStageZk.RunFiles} {record : String} {spec : Flock.SessionSpec} {tabs : Array Flock.Setup}
    {rest : Json × Json} {proofs : Array ByteArray} {units : Array ℕ}
    (h : SessionVerdict a bs keys s record spec tabs rest proofs units) {B : ByteArray} {record' : String}
    (hB : bs.get s.record s!"session {s.name}'s record" = .ok B) (hu : String.fromUTF8? B = some record') :
    record = record' := by
  obtain ⟨B', hB', hu'⟩ := h.blob
  rw [hB] at hB'
  obtain rfl : B = B' := Except.ok.inj hB'
  rw [hu] at hu'
  exact (Option.some.inj hu').symm

/-- **A session's coin-tree key stays in the run**: of sessions with their verdicts in order, the key session `i`'s
record names at its statement is a key of the run they end at. -/
theorem Verdicts.key_kept {bs : _root_.Core.Blobs} {l : List (StepAt × _root_.Spot.OneStageZk.RunFiles)}
    {r r' : _root_.Spot.OneStageZk.Run} (h : Verdicts bs r l r') {i : ℕ} (hi : i < l.length) {B : ByteArray}
    {record k : String} (hB : bs.get l[i].2.record s!"session {l[i].2.name}'s record" = .ok B)
    (hu : String.fromUTF8? B = some record) (hk : l[i].1.coinKey record = some k) :
    r'.coinKeys.contains k = true := by
  obtain ⟨ri, record', spec, tabs, rest, proofs, units, -, hsv, hpost⟩ := h.getElem hi
  obtain rfl := hsv.record_eq hB hu
  refine hpost.keys.2 k ?_
  rw [hsv.coinKey_eq] at hk
  show (match _root_.Spot.OneStageZk.coinKeyOf record' (.ok (spec, tabs, rest)) with
    | some key => ri.coinKeys.push key
    | none => ri.coinKeys).contains k = true
  rw [hk]
  show (ri.coinKeys.push k).contains k = true
  rw [Array.contains_push, beq_self_eq_true, Bool.or_true]

/-- **Each session's coins are its own**: of sessions with their verdicts in order, no later session's record names
the coin-tree key an earlier one's names, each at its own statement, so no two of them flip their coins from one coin
tree. -/
theorem Verdicts.coins_own {bs : _root_.Core.Blobs} {l : List (StepAt × _root_.Spot.OneStageZk.RunFiles)}
    {r r' : _root_.Spot.OneStageZk.Run} (h : Verdicts bs r l r') {i j : ℕ} (hij : i < j) (hj : j < l.length)
    {Bi Bj : ByteArray} {ri rj k : String}
    (hBi : bs.get (l[i]'(Nat.lt_trans hij hj)).2.record
      s!"session {(l[i]'(Nat.lt_trans hij hj)).2.name}'s record" = .ok Bi)
    (hui : String.fromUTF8? Bi = some ri)
    (hBj : bs.get l[j].2.record s!"session {l[j].2.name}'s record" = .ok Bj)
    (huj : String.fromUTF8? Bj = some rj) (hk : (l[i]'(Nat.lt_trans hij hj)).1.coinKey ri = some k) :
    l[j].1.coinKey rj ≠ some k := by
  obtain ⟨rj', record, spec, tabs, rest, proofs, units, hpre, hsv, -⟩ := h.getElem hj
  obtain rfl := hsv.record_eq hBj huj
  intro hkj
  have hi' : i < (l.take j).length := by
    rw [List.length_take]
    exact Nat.lt_min.2 ⟨hij, Nat.lt_trans hij hj⟩
  have e : (l.take j)[i]'hi' = l[i]'(Nat.lt_trans hij hj) := List.getElem_take
  have hin : rj'.coinKeys.contains k = true :=
    hpre.key_kept hi' (by rw [e]; exact hBi) hui (by rw [e]; exact hk)
  rw [hsv.coinKey_eq] at hkj
  rw [not_contains_of_fresh (hsv.fresh k hkj)] at hin
  cases hin

/-! ## v2's run -/

/-- **The statement v2's `run` holds every session to**, at the parameters `p` on the blobs `bs`: the statement's tags
`tags0`, the circuit and public files, the blobs' lookup tables and the verifier's partition object and program
(`inputsOf`), the registered reads `rd`, and the checks `run` builds from them: the scope (`Record.scopeOf`), the files'
ports and roots, the population (`FilePart.check` on the public file) and the program's digest against the partition
object's. Nothing of it reads a session's record, coins or sample. -/
def StepAt.ofOwn (p : _root_.Spot.OneStageZk.Own) (tags0 : Flock.Tags) (circuitB pubB : ByteArray)
    (bs : _root_.Core.Blobs) (rd : _root_.Spot.OneStageZk.Reads) : StepAt :=
  let I := _root_.Spot.OneStageZk.inputsOf tags0 p circuitB pubB bs
  { I, rd, scope := Record.scopeOf I rd
    ksFiles := Flock.PortRoots.checkOf (Flock.ProvedScope.circuitOf (Flock.Witness.zkTags I) I.circuitFile I.tables)
      (Flock.Witness.zkTags I) pubB
    population := _root_.Spot.FilePart.check p.program (_root_.Spot.getD p.partition "query") pubB
    programSha := _root_.Spot.OneStageZk.programDigest p.program
    want := _root_.Spot.getD p.partition "program" }

/-- **v2's `run` where it accepted every session is the sessions' verdicts in order, at the one statement its
parameters fix** (`StepAt.ofOwn`): what `run` checks before any session (the blobs' lookup tables, the statement's
tags, the registered reads, the circuit and public files and their lengths); the sessions' verdicts from the empty run;
and the proved units held to the evidence's (`provedHeld`). -/
theorem run_verdicts_at {p : _root_.Spot.OneStageZk.Own} {sessions : Array _root_.Spot.OneStageZk.RunFiles}
    {pu : Option (Array ℕ)} {bs : _root_.Core.Blobs} {r : _root_.Spot.OneStageZk.Run}
    (h : _root_.Spot.OneStageZk.run p sessions pu bs = .ok r) (hall : r.all = true) :
    ∃ (tags0 : Flock.Tags) (circuitB pubB : ByteArray) (rd : _root_.Spot.OneStageZk.Reads),
      _root_.Spot.OneStageZk.tablesHeld p bs = .ok () ∧
      Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT = some tags0 ∧ tags0.plainLeaves = false ∧
      (p.registered.isNone && _root_.Spot.OneStageZk.readsRegistered p.programJ) = false ∧
      bs.get p.circuit "circuit" = .ok circuitB ∧ bs.get p.pub "public file" = .ok pubB ∧
      _root_.Spot.OneStageZk.fitsWhy _root_.Spot.OneStageZk.WORD circuitB pubB (_root_.Spot.OneStageZk.tablesOf bs)
        (some p.program) (some p.partition) p.registered = none ∧
      _root_.Spot.OneStageZk.readsOf p.registered = .ok rd ∧
      Verdicts bs ⟨#[], #[], #[], true⟩
        (sessions.toList.map fun s => (StepAt.ofOwn p tags0 circuitB pubB bs rd, s)) r ∧
      ∀ want, pu = some want → _root_.Spot.OneStageZk.provedHeld r.proved want = .ok () := by
  unfold _root_.Spot.OneStageZk.run at h
  dsimp only at h
  obtain ⟨⟨⟩, htab, h⟩ := ebind_ok h
  split at h
  · rename_i tags0 htags
    obtain ⟨_, hpl, h⟩ := ebind_ok h
    obtain ⟨_, hreg, h⟩ := ebind_ok h
    obtain ⟨circuitB, hc, h⟩ := ebind_ok h
    obtain ⟨pubB, hp, h⟩ := ebind_ok h
    obtain ⟨_, hfits, h⟩ := ebind_ok h
    obtain ⟨rd, hrd, h⟩ := ebind_ok h
    obtain ⟨r', hr', h⟩ := ebind_ok h
    have e : r' = r ∧ ∀ want', pu = some want' → r'.all = true →
        _root_.Spot.OneStageZk.provedHeld r'.proved want' = .ok () := by
      cases pu with
      | none =>
        refine ⟨?_, fun _ hw => by cases hw⟩
        simpa only [pure, Except.pure, Except.ok.injEq] using h
      | some want =>
        refine ⟨?_, fun want' hw hall => ?_⟩
        · repeat' (first | (obtain ⟨_, _, h⟩ := ebind_ok h) | (split at h))
          all_goals simpa only [pure, Except.pure, Except.ok.injEq] using h
        · cases hw
          simp only [hall, ↓reduceIte] at h
          obtain ⟨⟨⟩, hu, -⟩ := ebind_ok h
          exact hu
    obtain ⟨rfl, hw⟩ := e
    have hpl' : tags0.plainLeaves = false := by
      cases hb : tags0.plainLeaves with
      | false => rfl
      | true => rw [hb] at hpl; cases hpl
    have hreg' : (p.registered.isNone && _root_.Spot.OneStageZk.readsRegistered p.programJ) = false := by
      cases hb : (p.registered.isNone && _root_.Spot.OneStageZk.readsRegistered p.programJ) with
      | false => rfl
      | true => rw [hb] at hreg; cases hreg
    have hfits' : _root_.Spot.OneStageZk.fitsWhy _root_.Spot.OneStageZk.WORD circuitB pubB
        (_root_.Spot.OneStageZk.tablesOf bs) (some p.program) (some p.partition) p.registered = none := by
      split at hfits
      · cases hfits
      · rename_i hnone
        exact hnone
    refine ⟨tags0, circuitB, pubB, rd, htab, htags, hpl', hreg', hc, hp, hfits', hrd, ?_,
      fun want hpu => hw want hpu hall⟩
    refine verdicts_of_steps ?_ hall
    rw [List.foldlM_map, Array.foldlM_toList]
    exact hr'
  · cases h

/-- **v2's `run` where it accepted every session is the sessions' verdicts in order, at the parameters' one
statement**: the statement's tags, the circuit and public files and the registered reads; the sessions' verdicts from
the empty run, each at the statement's inputs, scope and population check (`FilePart.check` on the public file); and
the proved units held to the evidence's (`provedHeld`). -/
theorem run_verdicts {p : _root_.Spot.OneStageZk.Own} {sessions : Array _root_.Spot.OneStageZk.RunFiles}
    {pu : Option (Array ℕ)} {bs : _root_.Core.Blobs} {r : _root_.Spot.OneStageZk.Run}
    (h : _root_.Spot.OneStageZk.run p sessions pu bs = .ok r) (hall : r.all = true) :
    ∃ (a : StepAt) (tags0 : Flock.Tags) (circuitB pubB : ByteArray),
      Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT = some tags0 ∧ tags0.plainLeaves = false ∧
      bs.get p.circuit "circuit" = .ok circuitB ∧ bs.get p.pub "public file" = .ok pubB ∧
      a.I = _root_.Spot.OneStageZk.inputsOf tags0 p circuitB pubB bs ∧
      _root_.Spot.OneStageZk.readsOf p.registered = .ok a.rd ∧ a.scope = Record.scopeOf a.I a.rd ∧
      a.population = _root_.Spot.FilePart.check p.program (_root_.Spot.getD p.partition "query") pubB ∧
      Verdicts bs ⟨#[], #[], #[], true⟩ (sessions.toList.map fun s => (a, s)) r ∧
      ∀ want, pu = some want → _root_.Spot.OneStageZk.provedHeld r.proved want = .ok () := by
  obtain ⟨tags0, circuitB, pubB, rd, -, htags, hpl, -, hc, hp, -, hrd, hv, hw⟩ := run_verdicts_at h hall
  exact ⟨StepAt.ofOwn p tags0 circuitB pubB bs rd, tags0, circuitB, pubB, htags, hpl, hc, hp, rfl, hrd, rfl, rfl, hv,
    hw⟩

end Verity.Sampled
