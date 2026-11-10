module

public import Proofs.Properties.Verity.Sampled.Record.Accepts

/-!
# v2's sessions' verdicts, in order

One-stage v2 (`Spot.OneStageZk`) verifies its sessions one after another: `step` on each, a fold that carries a `Run`
(the coin-tree keys the earlier sessions used, the verdicts so far, the units proved and whether every session
accepted). This module reads that fold as the sessions' verdicts in order, whatever each session's own inputs are:

* **what a session's `step` reads** (`Ctx`): its statement's inputs, the verifier's registered reads, the scope, file
  and population checks, and the program's digest against the partition object's;
* **one session passes** (`Passes`): its record is on the blobs as UTF-8 and its draw fits the statement's length
  words, `verdictOf` on it, set up at its own record's sample and at the coin-tree keys of the sessions before it, is
  `.ok ()`, its verdict names the partition object's program, and its record's drawn units read off the public file;
* **`step` accepts exactly where the session passes** (`step_all_iff`): from a run that accepted so far to the run with
  that session's key, verdict and units (`Ctx.after`), and a session that fails ends every later acceptance;
* **the fold accepts exactly where each session passes in order** (`PassAll`, `stepAll_all_iff`), each at the keys of
  the sessions before it (`passAll_getElem`), so no two sessions' coin-tree keys are one (`Passes.fresh`,
  `passAll_keys`).
-/

@[expose] public section

namespace Verity.Sampled.Verdicts

open Lean (Json)
open Verity.Recursive.BaseSetup (bind_ok)

/-- An `Except` bind is `.ok` where its first step is and its continuation is at that step's value. -/
theorem ok_bind_of {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {a : α} {b : β} (hx : x = .ok a)
    (hf : f a = .ok b) : (x >>= f) = .ok b := by
  rw [hx]
  exact hf

/-! ## One session -/

/-- **What one session's `step` reads** besides the run so far, the blobs and the session's files: the statement's
inputs `I`, the verifier's registered reads `rd`, the scope check, the files' ports and roots check and the population
check its verdict runs, and the program's digest `programSha` held to the partition object's `want`. -/
structure Ctx where
  I : Flock.Witness.Inputs
  rd : _root_.Spot.OneStageZk.Reads
  scope : Flock.V Unit
  ksFiles : Flock.V Unit
  population : Flock.V Unit
  programSha : Option String
  want : Json

/-- v2's `step` at `c`, on the blobs `bs`: one more session `s` after the run `r`. -/
def Ctx.step (c : Ctx) (bs : Core.Blobs) (r : _root_.Spot.OneStageZk.Run) (s : _root_.Spot.OneStageZk.RunFiles) :
    Flock.V _root_.Spot.OneStageZk.Run :=
  _root_.Spot.OneStageZk.step c.I c.rd bs c.scope c.ksFiles c.population c.programSha c.want r s

/-- The session `c` sets up at the record `record`'s own sample (`Spot.OneStageZk.setupAt`). -/
def Ctx.setup (c : Ctx) (record : String) : _root_.Spot.OneStageZk.SetUp :=
  _root_.Spot.OneStageZk.setupAt c.I c.rd (Flock.Witness.recordDraw record)

/-- The session's coin-tree key, as its record names it at the session `c` sets up (`Spot.OneStageZk.coinKeyOf`). -/
def Ctx.coinKey (c : Ctx) (record : String) : Option String :=
  _root_.Spot.OneStageZk.coinKeyOf record (c.setup record)

/-- **`c`'s verdict on the session `s`** whose record is `record`, after sessions that used the coin-tree keys `keys`
(`Spot.OneStageZk.verdictOf`): set up at the record's own sample, its rep files the blobs `s` names. -/
def Ctx.verdict (c : Ctx) (bs : Core.Blobs) (keys : Array String) (s : _root_.Spot.OneStageZk.RunFiles)
    (record : String) : Flock.V Unit :=
  _root_.Spot.OneStageZk.verdictOf c.I c.scope c.ksFiles c.population keys record (Flock.Witness.recordDraw record)
    (c.setup record) (_root_.Spot.OneStageZk.filesOf bs s (c.setup record)) (c.coinKey record)

/-- The coin-tree keys `keys` with one more session's `k`, when it has one. -/
def keysWith (keys : Array String) (k : Option String) : Array String :=
  match k with
  | some k => keys.push k
  | none => keys

/-- **The session `s` passes at `c`** on the blobs `bs`, after sessions that used the coin-tree keys `keys`: its record
is a blob, UTF-8 (`record`), whose draw fits the statement's length words; `c`'s verdict on it is `.ok ()`; that verdict
names the partition object's program; and the units its record draws read off the public file (`units`). -/
structure Passes (c : Ctx) (bs : Core.Blobs) (keys : Array String) (s : _root_.Spot.OneStageZk.RunFiles)
    (record : String) (units : Array ℕ) : Prop where
  blob : ∃ B, bs.get s.record s!"session {s.name}'s record" = .ok B ∧ String.fromUTF8? B = some record
  fits : _root_.Spot.OneStageZk.drawFitsWhy _root_.Spot.OneStageZk.WORD (Flock.Witness.recordDraw record) = none
  verdict : c.verdict bs keys s record = .ok ()
  program : (c.programSha.map Json.str != some c.want) = false
  units : _root_.Spot.OneStageZk.provedUnits c.I.publicFile record = .ok units

/-- **The run after a session that passed**: the run `r` with the session's coin-tree key, its accepting verdict and
the units it proved. -/
def Ctx.after (c : Ctx) (r : _root_.Spot.OneStageZk.Run) (s : _root_.Spot.OneStageZk.RunFiles) (record : String)
    (units : Array ℕ) : _root_.Spot.OneStageZk.Run :=
  { coinKeys := keysWith r.coinKeys (c.coinKey record), all := r.all, proved := r.proved ++ units,
    out := r.out.push (_root_.Spot.OneStageZk.verdictJson c.rd c.programSha s (c.setup record) true "" units) }

/-- **`push` keeps the run accepting exactly on an accepting verdict**: from a run that accepted so far, on a verdict
`.ok ()` naming the partition object's program, with the units the record draws. -/
theorem push_all_iff {I : Flock.Witness.Inputs} {rd : _root_.Spot.OneStageZk.Reads} {programSha : Option String}
    {want : Json} {r r' : _root_.Spot.OneStageZk.Run} {s : _root_.Spot.OneStageZk.RunFiles} {record : String}
    {st : _root_.Spot.OneStageZk.SetUp} {coinKey : Option String} {verdict : Flock.V Unit} :
    (_root_.Spot.OneStageZk.push I rd programSha want r s record st coinKey verdict = .ok r' ∧ r'.all = true) ↔
      r.all = true ∧ verdict = .ok () ∧ (programSha.map Json.str != some want) = false ∧
        ∃ units, _root_.Spot.OneStageZk.provedUnits I.publicFile record = .ok units ∧
          r' = { coinKeys := keysWith r.coinKeys coinKey, all := r.all, proved := r.proved ++ units,
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
        obtain ⟨units, hus, h⟩ := bind_ok h
        obtain rfl := Except.ok.inj h
        exact ⟨hall, rfl, Bool.eq_false_iff.2 hne, units, hus, rfl⟩
  · rintro ⟨hr, hv, hp, units, hus, rfl⟩
    subst hv
    refine ⟨?_, hr⟩
    unfold _root_.Spot.OneStageZk.push
    dsimp only
    rw [ite_eq_right (Bool.eq_false_iff.1 hp)]
    exact ok_bind_of hus rfl

/-- **`step` keeps the run accepting exactly where the session passes**: from a run `r` that accepted so far, at the
coin-tree keys of `r`'s sessions, to the run after it (`Ctx.after`). -/
theorem step_all_iff {c : Ctx} {bs : Core.Blobs} {r r' : _root_.Spot.OneStageZk.Run}
    {s : _root_.Spot.OneStageZk.RunFiles} :
    (c.step bs r s = .ok r' ∧ r'.all = true) ↔
      r.all = true ∧ ∃ record units, Passes c bs r.coinKeys s record units ∧ r' = c.after r s record units := by
  constructor
  · rintro ⟨h, hall⟩
    unfold Ctx.step _root_.Spot.OneStageZk.step at h
    obtain ⟨B, hB, h⟩ := bind_ok h
    cases hu : String.fromUTF8? B with
    | none => rw [hu] at h; cases h
    | some record =>
      rw [hu] at h
      dsimp only at h
      obtain ⟨_, hfit, h⟩ := bind_ok h
      try dsimp only at h
      obtain ⟨hr, hv, hp, units, hus, rfl⟩ := push_all_iff.1 ⟨h, hall⟩
      refine ⟨hr, record, units, ⟨⟨B, hB, hu⟩, ?_, hv, hp, hus⟩, rfl⟩
      cases hd : _root_.Spot.OneStageZk.drawFitsWhy _root_.Spot.OneStageZk.WORD
          (Flock.Witness.recordDraw record) with
      | none => rfl
      | some w => rw [hd] at hfit; cases hfit
  · rintro ⟨hr, record, units, ⟨⟨B, hB, hu⟩, hfit, hv, hp, hus⟩, rfl⟩
    have hpush : _root_.Spot.OneStageZk.push c.I c.rd c.programSha c.want r s record (c.setup record)
        (c.coinKey record) (c.verdict bs r.coinKeys s record) = .ok (c.after r s record units) ∧
          (c.after r s record units).all = true :=
      push_all_iff.2 ⟨hr, hv, hp, units, hus, rfl⟩
    refine ⟨?_, hpush.2⟩
    unfold Ctx.step _root_.Spot.OneStageZk.step
    refine ok_bind_of hB ?_
    try dsimp only
    rw [hu]
    try dsimp only
    refine ok_bind_of (a := ()) ?_ hpush.1
    rw [hfit]
    try rfl

/-- **A session that passes used a coin-tree key no earlier session did.** -/
theorem Passes.fresh {c : Ctx} {bs : Core.Blobs} {keys : Array String} {s : _root_.Spot.OneStageZk.RunFiles}
    {record : String} {units : Array ℕ} (h : Passes c bs keys s record units) {k : String}
    (hk : c.coinKey record = some k) : keys.contains k = false := by
  have hv := h.verdict
  unfold Ctx.verdict _root_.Spot.OneStageZk.verdictOf at hv
  obtain ⟨_, -, hv⟩ := bind_ok hv
  obtain ⟨_, hf, -⟩ := bind_ok hv
  rw [hk] at hf
  dsimp only at hf
  unfold Flock.CoinTree.checkFresh at hf
  cases hc : keys.contains k with
  | false => rfl
  | true => rw [hc] at hf; cases hf

/-! ## The sessions, in order -/

/-- **The sessions `xs` pass in order** on the blobs `bs`, from the run `r` to the run `r'`: each, at its own context
and at the coin-tree keys of the run before it, passes (`Passes`), and the run after it (`Ctx.after`) is the next one's;
`r'` is the run after the last. -/
def PassAll (bs : Core.Blobs) :
    _root_.Spot.OneStageZk.Run → List (Ctx × _root_.Spot.OneStageZk.RunFiles) → _root_.Spot.OneStageZk.Run → Prop
  | r, [], r' => r' = r
  | r, x :: xs, r' => ∃ record units, Passes x.1 bs r.coinKeys x.2 record units ∧
      PassAll bs (x.1.after r x.2 record units) xs r'

theorem passAll_nil {bs : Core.Blobs} {r r' : _root_.Spot.OneStageZk.Run} : PassAll bs r [] r' ↔ r' = r :=
  Iff.rfl

theorem passAll_cons {bs : Core.Blobs} {r r' : _root_.Spot.OneStageZk.Run}
    {x : Ctx × _root_.Spot.OneStageZk.RunFiles} {xs : List (Ctx × _root_.Spot.OneStageZk.RunFiles)} :
    PassAll bs r (x :: xs) r' ↔ ∃ record units, Passes x.1 bs r.coinKeys x.2 record units ∧
      PassAll bs (x.1.after r x.2 record units) xs r' :=
  Iff.rfl

/-- v2's fold of `step` over the sessions `xs`, each at its own context, from the run `r`. -/
def stepAll (bs : Core.Blobs) (r : _root_.Spot.OneStageZk.Run)
    (xs : List (Ctx × _root_.Spot.OneStageZk.RunFiles)) : Flock.V _root_.Spot.OneStageZk.Run :=
  xs.foldlM (fun r x => x.1.step bs r x.2) r

/-- **The fold accepts exactly where each session passes, in order**: `stepAll` from a run `r` ends at a run `r'` that
accepted every session exactly where `r` accepted so far and the sessions pass in order from `r` to `r'`. -/
theorem stepAll_all_iff {bs : Core.Blobs} :
    ∀ (xs : List (Ctx × _root_.Spot.OneStageZk.RunFiles)) {r r' : _root_.Spot.OneStageZk.Run},
      (stepAll bs r xs = .ok r' ∧ r'.all = true) ↔ r.all = true ∧ PassAll bs r xs r'
  | [], r, r' => by
    constructor
    · rintro ⟨h, hall⟩
      obtain rfl : r = r' := Except.ok.inj h
      exact ⟨hall, passAll_nil.2 rfl⟩
    · rintro ⟨hall, h⟩
      obtain rfl : r' = r := passAll_nil.1 h
      exact ⟨rfl, hall⟩
  | x :: xs, r, r' => by
    constructor
    · rintro ⟨h, hall⟩
      unfold stepAll at h
      rw [List.foldlM_cons] at h
      obtain ⟨r1, h1, h⟩ := bind_ok h
      have h1' : x.1.step bs r x.2 = .ok r1 := h1
      have h' : stepAll bs r1 xs = .ok r' := h
      obtain ⟨hall1, hrest⟩ := (stepAll_all_iff xs).1 ⟨h', hall⟩
      obtain ⟨hr, record, units, hp, rfl⟩ := step_all_iff.1 ⟨h1', hall1⟩
      exact ⟨hr, passAll_cons.2 ⟨record, units, hp, hrest⟩⟩
    · rintro ⟨hr, h⟩
      obtain ⟨record, units, hp, hrest⟩ := passAll_cons.1 h
      have h1 : x.1.step bs r x.2 = .ok (x.1.after r x.2 record units) ∧
          (x.1.after r x.2 record units).all = true :=
        step_all_iff.2 ⟨hr, record, units, hp, rfl⟩
      have h2 := (stepAll_all_iff xs).2 ⟨h1.2, hrest⟩
      refine ⟨?_, h2.2⟩
      unfold stepAll
      rw [List.foldlM_cons]
      exact ok_bind_of h1.1 h2.1

/-- **Sessions in order split at any point**: `xs` then `ys` pass in order from `r` to `r'` exactly where `xs` pass to
some run and `ys` from it to `r'`. -/
theorem passAll_append {bs : Core.Blobs} :
    ∀ (xs : List (Ctx × _root_.Spot.OneStageZk.RunFiles)) {ys : List (Ctx × _root_.Spot.OneStageZk.RunFiles)}
      {r r' : _root_.Spot.OneStageZk.Run},
      PassAll bs r (xs ++ ys) r' ↔ ∃ m, PassAll bs r xs m ∧ PassAll bs m ys r'
  | [], ys, r, r' => by
    constructor
    · intro h
      exact ⟨r, passAll_nil.2 rfl, h⟩
    · rintro ⟨m, hm, h⟩
      obtain rfl : m = r := passAll_nil.1 hm
      exact h
  | x :: xs, ys, r, r' => by
    show PassAll bs r (x :: (xs ++ ys)) r' ↔ _
    constructor
    · intro h
      obtain ⟨record, units, hp, hrest⟩ := passAll_cons.1 h
      obtain ⟨m, hm, hys⟩ := (passAll_append xs).1 hrest
      exact ⟨m, passAll_cons.2 ⟨record, units, hp, hm⟩, hys⟩
    · rintro ⟨m, hm, hys⟩
      obtain ⟨record, units, hp, hrest⟩ := passAll_cons.1 hm
      exact passAll_cons.2 ⟨record, units, hp, (passAll_append xs).2 ⟨m, hrest, hys⟩⟩

/-- **Each session of sessions that pass in order has its own accepting verdict**: session `i` passes at its own
context, at the coin-tree keys of the run `ri` the `i` sessions before it pass to, and the sessions after it pass from
the run after it. -/
theorem passAll_getElem {bs : Core.Blobs} {xs : List (Ctx × _root_.Spot.OneStageZk.RunFiles)}
    {r r' : _root_.Spot.OneStageZk.Run} (h : PassAll bs r xs r') {i : ℕ} (hi : i < xs.length) :
    ∃ ri record units, PassAll bs r (xs.take i) ri ∧ Passes xs[i].1 bs ri.coinKeys xs[i].2 record units ∧
      PassAll bs (xs[i].1.after ri xs[i].2 record units) (xs.drop (i + 1)) r' := by
  have hsplit : xs.take i ++ xs[i] :: xs.drop (i + 1) = xs := by
    rw [← List.drop_eq_getElem_cons hi, List.take_append_drop]
  have h' : PassAll bs r (xs.take i ++ xs[i] :: xs.drop (i + 1)) r' := by
    rw [hsplit]
    exact h
  obtain ⟨ri, hpre, hrest⟩ := (passAll_append (xs.take i)).1 h'
  obtain ⟨record, units, hp, hpost⟩ := passAll_cons.1 hrest
  exact ⟨ri, record, units, hpre, hp, hpost⟩

/-- **Sessions that pass in order keep the run's acceptance and only add coin-tree keys**: the run they pass to
accepted every session where the run they pass from did, and each key of the run before is a key of the run after. -/
theorem passAll_keys {bs : Core.Blobs} :
    ∀ (xs : List (Ctx × _root_.Spot.OneStageZk.RunFiles)) {r r' : _root_.Spot.OneStageZk.Run},
      PassAll bs r xs r' → r'.all = r.all ∧ ∀ k, r.coinKeys.contains k = true → r'.coinKeys.contains k = true
  | [], r, r', h => by
    obtain rfl : r' = r := passAll_nil.1 h
    exact ⟨rfl, fun _ hk => hk⟩
  | x :: xs, r, r', h => by
    obtain ⟨record, units, hp, hrest⟩ := passAll_cons.1 h
    obtain ⟨hall, hkeys⟩ := passAll_keys xs hrest
    refine ⟨hall, fun k hk => hkeys k ?_⟩
    show (keysWith r.coinKeys (x.1.coinKey record)).contains k = true
    cases x.1.coinKey record with
    | none => exact hk
    | some k' =>
      show (r.coinKeys.push k').contains k = true
      rw [Array.contains_push, hk]
      rfl

end Verity.Sampled.Verdicts
