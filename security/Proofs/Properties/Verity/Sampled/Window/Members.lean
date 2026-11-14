module

public import Proofs.Properties.Verity.Sampled.Window.Accepts
public import Proofs.Properties.Verity.Sampled.Held

/-!
# The per-member form's acceptance is its members' verdicts, in member order

One-stage v2 in this tree runs every session at one statement (`Spot.OneStageZk.run`) and refuses evidence of more
than one session. The per-member form gives each sampled member of a window a statement of its own, with one session.
This module is a stand-in for that form's run over this tree's `step`, as it would be once the verifier holds it and no
longer refuses it, and reads its acceptance; nothing here is the verifier's, and the verifier's gate is not lifted.

* **The statement a member's sessions are held to** (`statementOf`): what `run` checks of its parameters before any
  session, ending in `StepAt.ofOwn`. It reads the member's own inputs and the blobs they name, and no session, record
  or coin. v2's `run` where it accepted read exactly it (`run_statement`), so the one-statement path is the
  per-member run at one member (`runFrom_of_run`).
* **The statements' structure** (`membersHeld`): each statement names a member of the parameters no earlier statement
  named, in the members' order (by first unit), with exactly one session. `membersHeld_ok_iff` reads it as `HeldFrom`,
  `membersHeld_zip` as one member a statement and `membersHeld_inOrder` as the members' order, none twice.
* **The run** (`runFrom`, `runMembers`): each statement's sessions at its member's statement, after the statements
  before it, so a coin-tree key is fresh across members; then the proved units held to the evidence's.
* **`runMembers_iff`**: the per-member run accepts every session exactly where the statements' structure holds, each
  statement is exactly one session at its member's statement (`Stated`), the sessions have their verdicts in order
  (`Verdicts`, each after the coin-tree keys of the members before it) and the proved units are the evidence's.
* **What the registration fixes of a member before the sample** (`SliceHeld`, `Member.Fixed`): the member's public file
  is held to the member's slice of the registration (its scheme, its ports' roots and domains, its number of proof
  blocks) and to the member's units, as `publicHeld` holds the one statement's to the whole registration.
  `publicHeld_iff` is that reading of `publicHeld`: the one-statement path's check is the slice check at the whole
  registration and the population's units. `Accepts` and `accepts_iff` put the two together.
-/

@[expose] public section

namespace Verity.Sampled.PerMember

open Lean (Json)
open FlockSoundness.Refine (ebind_ok)

/-- An `Except` with its refusals rewritten is `.ok` exactly where it was. -/
theorem mapError_ok_iff {ε ε' α : Type} {f : ε → ε'} {x : Except ε α} {a : α} :
    x.mapError f = .ok a ↔ x = .ok a := by
  cases x with
  | error e => exact ⟨fun h => (nomatch h), fun h => (nomatch h)⟩
  | ok b =>
    have e : Except.mapError f (Except.ok b : Except ε α) = Except.ok b := rfl
    constructor
    · intro h
      rw [e] at h
      exact congrArg Except.ok (Except.ok.inj h)
    · intro h
      rw [e]
      exact congrArg Except.ok (Except.ok.inj h)

/-! ## The statement a member's sessions are held to -/

/-- **The statement v2 holds sessions to at the parameters `p`, read off the blobs before any session**: `run`'s own
checks in `run`'s order (the blobs' lookup tables the circuit's, the statement's tags with hiding leaves, registered
values read only with the verifier's registered file, the circuit and public files, each length within a statement's
word, the registered reads), then `StepAt.ofOwn`. A refusal where `run` refuses. -/
def statementOf (p : _root_.Spot.OneStageZk.Own) (bs : _root_.Core.Blobs) : Flock.V StepAt :=
  _root_.Spot.OneStageZk.tablesHeld p bs >>= fun _ =>
  match Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT with
  | none => throw s!"the statement {_root_.Spot.OneStageZk.STATEMENT} is not known"
  | some tags0 =>
    match tags0.plainLeaves with
    | true => throw s!"{tags0.name}: a statement with plain SHA-512 leaves: --zk's hiding needs hm96 leaves"
    | false =>
      match p.registered.isNone && _root_.Spot.OneStageZk.readsRegistered p.programJ with
      | true => throw "registered: the program reads registered values and the verifier holds no registered reads"
      | false =>
        bs.get p.circuit "circuit" >>= fun circuitB =>
        bs.get p.pub "public file" >>= fun pubB =>
        match _root_.Spot.OneStageZk.fitsWhy _root_.Spot.OneStageZk.WORD circuitB pubB
            (_root_.Spot.OneStageZk.tablesOf bs) (some p.program) (some p.partition) p.registered with
        | some what => throw s!"{what}: {_root_.Spot.OneStageZk.FITS}"
        | none =>
          _root_.Spot.OneStageZk.readsOf p.registered >>= fun rd =>
          pure (StepAt.ofOwn p tags0 circuitB pubB bs rd)

/-- **`statementOf` reads a statement exactly where `run`'s checks before its sessions pass**, and the statement is
`StepAt.ofOwn` at what they read. -/
theorem statementOf_ok_iff {p : _root_.Spot.OneStageZk.Own} {bs : _root_.Core.Blobs} {a : StepAt} :
    statementOf p bs = .ok a ↔
      ∃ (tags0 : Flock.Tags) (circuitB pubB : ByteArray) (rd : _root_.Spot.OneStageZk.Reads),
        _root_.Spot.OneStageZk.tablesHeld p bs = .ok () ∧
        Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT = some tags0 ∧ tags0.plainLeaves = false ∧
        (p.registered.isNone && _root_.Spot.OneStageZk.readsRegistered p.programJ) = false ∧
        bs.get p.circuit "circuit" = .ok circuitB ∧ bs.get p.pub "public file" = .ok pubB ∧
        _root_.Spot.OneStageZk.fitsWhy _root_.Spot.OneStageZk.WORD circuitB pubB
          (_root_.Spot.OneStageZk.tablesOf bs) (some p.program) (some p.partition) p.registered = none ∧
        _root_.Spot.OneStageZk.readsOf p.registered = .ok rd ∧ a = StepAt.ofOwn p tags0 circuitB pubB bs rd := by
  constructor
  · intro h
    unfold statementOf at h
    obtain ⟨⟨⟩, htab, h⟩ := ebind_ok h
    cases htags : Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT with
    | none => rw [htags] at h; cases h
    | some tags0 =>
      rw [htags] at h
      dsimp only at h
      cases hpl : tags0.plainLeaves with
      | true => rw [hpl] at h; cases h
      | false =>
        rw [hpl] at h
        dsimp only at h
        cases hreg : (p.registered.isNone && _root_.Spot.OneStageZk.readsRegistered p.programJ) with
        | true => rw [hreg] at h; cases h
        | false =>
          rw [hreg] at h
          dsimp only at h
          obtain ⟨circuitB, hc, h⟩ := ebind_ok h
          obtain ⟨pubB, hp, h⟩ := ebind_ok h
          cases hfits : _root_.Spot.OneStageZk.fitsWhy _root_.Spot.OneStageZk.WORD circuitB pubB
              (_root_.Spot.OneStageZk.tablesOf bs) (some p.program) (some p.partition) p.registered with
          | some what => rw [hfits] at h; cases h
          | none =>
            rw [hfits] at h
            dsimp only at h
            obtain ⟨rd, hrd, h⟩ := ebind_ok h
            exact ⟨tags0, circuitB, pubB, rd, htab, by first | exact htags | rfl, hpl, by first | exact hreg | rfl, hc,
              hp, by first | exact hfits | rfl, hrd, (Except.ok.inj h).symm⟩
  · rintro ⟨tags0, circuitB, pubB, rd, htab, htags, hpl, hreg, hc, hp, hfits, hrd, rfl⟩
    unfold statementOf
    refine ok_bind_of htab ?_
    rw [htags]
    dsimp only
    rw [hpl]
    dsimp only
    rw [hreg]
    dsimp only
    refine ok_bind_of hc ?_
    refine ok_bind_of hp ?_
    rw [hfits]
    exact ok_bind_of hrd rfl

/-- **The statement's public file is the blob the parameters name**: nothing a session sends chooses it. -/
theorem statementOf_public {p : _root_.Spot.OneStageZk.Own} {bs : _root_.Core.Blobs} {a : StepAt}
    (h : statementOf p bs = .ok a) :
    ∃ pubB, bs.get p.pub "public file" = .ok pubB ∧ a.I.publicFile = pubB := by
  obtain ⟨tags0, circuitB, pubB, rd, -, -, -, -, -, hp, -, -, rfl⟩ := statementOf_ok_iff.1 h
  exact ⟨pubB, hp, rfl⟩

/-- **v2's `run` where it accepted every session read its statement as `statementOf` does**, and is the sessions'
verdicts in order at it, the proved units held to the evidence's. -/
theorem run_statement {p : _root_.Spot.OneStageZk.Own} {sessions : Array _root_.Spot.OneStageZk.RunFiles}
    {pu : Option (Array ℕ)} {bs : _root_.Core.Blobs} {r : _root_.Spot.OneStageZk.Run}
    (h : _root_.Spot.OneStageZk.run p sessions pu bs = .ok r) (hall : r.all = true) :
    ∃ a, statementOf p bs = .ok a ∧
      Verdicts bs ⟨#[], #[], #[], true⟩ (sessions.toList.map fun s => (a, s)) r ∧
      ∀ want, pu = some want → _root_.Spot.OneStageZk.provedHeld r.proved want = .ok () := by
  obtain ⟨tags0, circuitB, pubB, rd, htab, htags, hpl, hreg, hc, hp, hfits, hrd, hv, hw⟩ := run_verdicts_at h hall
  exact ⟨_, statementOf_ok_iff.2 ⟨tags0, circuitB, pubB, rd, htab, htags, hpl, hreg, hc, hp, hfits, hrd, rfl⟩, hv, hw⟩

/-! ## The statements' structure -/

/-- A member of the per-member form: its name, its units as `[lo, hi)` ranges and its statement's own inputs. -/
structure Member where
  name : String
  units : Array (ℕ × ℕ)
  own : _root_.Spot.OneStageZk.Own

/-- One statement of the evidence's proofs: the member it proves (none in the one-statement form) and its sessions. -/
structure StatementFiles where
  member : Option String
  sessions : Array _root_.Spot.OneStageZk.RunFiles

/-- A member's first unit, its lowest. Members' units are disjoint, so it orders them, in the registration's order. -/
def Member.first (m : Member) : ℕ := m.units.foldl (fun a x => min a x.1) ((m.units[0]?.map (·.1)).getD 0)

/-- A member's units, each range's in order. -/
def Member.indices (m : Member) : Array ℕ := m.units.flatMap fun x => (Array.range (x.2 - x.1)).map (x.1 + ·)

/-- Whether the member `m`'s statement may follow the members `held`'s: its first unit is not before the last one's. -/
def follows (held : Array Member) (m : Member) : Bool :=
  match held.back? with
  | some prev => !decide (m.first < prev.first)
  | none => true

/-- One statement held, after the members `held` of the statements before it: it names a member of the parameters
`ms` that no earlier statement named, in the members' order (`follows`), and has exactly one session (one pass on live
coins, never retried; an unsampled member has no statement). The members so far, with its own. -/
def heldStep (ms : Array Member) (held : Array Member) (st : StatementFiles) : Flock.V (Array Member) :=
  match st.member with
  | none => throw "the evidence's proofs are one statement's and the parameters' per member"
  | some name =>
    match ms.find? (·.name == name) with
    | none => throw s!"member {name}: the parameters name no such member"
    | some m =>
      match held.any (·.name == name) with
      | true => throw s!"member {name}: a second statement: a member's units are one statement's"
      | false =>
        match follows held m with
        | false => throw s!"member {name}: its statement follows a member's whose units start after its own"
        | true =>
          match st.sessions.size != 1 with
          | true => throw s!"member {name}: its statement has {st.sessions.size} sessions, not exactly one"
          | false => pure (held.push m)

/-- **The per-member form's statements as public structure**, before any session runs (`heldStep` on each, in order):
their members, statement by statement. -/
def membersHeld (ms : Array Member) (sts : Array StatementFiles) : Flock.V (Array Member) :=
  sts.foldlM (heldStep ms) #[]

/-- **The statement `st` is the member `m`'s, after the members `held`'s.** -/
structure HeldAfter (ms held : Array Member) (st : StatementFiles) (m : Member) : Prop where
  named : st.member = some m.name
  member : ms.find? (·.name == m.name) = some m
  fresh : held.any (·.name == m.name) = false
  ordered : follows held m = true
  one : st.sessions.size = 1

theorem heldStep_ok_iff {ms held held' : Array Member} {st : StatementFiles} :
    heldStep ms held st = .ok held' ↔ ∃ m, HeldAfter ms held st m ∧ held' = held.push m := by
  constructor
  · intro h
    unfold heldStep at h
    cases hmem : st.member with
    | none => rw [hmem] at h; cases h
    | some name =>
      rw [hmem] at h
      dsimp only at h
      cases hfind : ms.find? (·.name == name) with
      | none => rw [hfind] at h; cases h
      | some m =>
        rw [hfind] at h
        dsimp only at h
        have hb : (m.name == name) = true := Array.find?_some (p := fun x : Member => x.name == name) hfind
        have hname : m.name = name := eq_of_beq hb
        subst hname
        cases hany : held.any (·.name == m.name) with
        | true => rw [hany] at h; cases h
        | false =>
          rw [hany] at h
          dsimp only at h
          cases hfol : follows held m with
          | false => rw [hfol] at h; cases h
          | true =>
            rw [hfol] at h
            dsimp only at h
            cases hone : (st.sessions.size != 1) with
            | true => rw [hone] at h; cases h
            | false =>
              rw [hone] at h
              refine ⟨m, ⟨hmem, hfind, hany, hfol, ?_⟩, (Except.ok.inj h).symm⟩
              cases hd : decide (st.sessions.size = 1) with
              | true => exact of_decide_eq_true hd
              | false =>
                have hne : (st.sessions.size != 1) = true := bne_iff_ne.2 (of_decide_eq_false hd)
                rw [hone] at hne
                cases hne
  · rintro ⟨m, ⟨hmem, hfind, hany, hfol, hone⟩, rfl⟩
    have hone' : (st.sessions.size != 1) = false := by
      rw [hone]
      rfl
    unfold heldStep
    rw [hmem]
    dsimp only
    rw [hfind]
    dsimp only
    rw [hany]
    dsimp only
    rw [hfol]
    dsimp only
    rw [hone']
    try rfl

/-- **The statements `sts` are members' in order, after the members `held`'s**, to the members `held'`. -/
def HeldFrom (ms : Array Member) : Array Member → List StatementFiles → Array Member → Prop
  | held, [], held' => held' = held
  | held, st :: sts, held' => ∃ m, HeldAfter ms held st m ∧ HeldFrom ms (held.push m) sts held'

theorem foldlM_heldStep_iff {ms : Array Member} :
    ∀ (sts : List StatementFiles) {held held' : Array Member},
      sts.foldlM (heldStep ms) held = .ok held' ↔ HeldFrom ms held sts held'
  | [], held, held' => ⟨fun h => (Except.ok.inj h).symm, fun h => congrArg Except.ok (Eq.symm h)⟩
  | st :: sts, held, held' => by
    rw [List.foldlM_cons]
    constructor
    · intro h
      obtain ⟨held1, h1, h⟩ := ebind_ok h
      obtain ⟨m, hm, rfl⟩ := heldStep_ok_iff.1 h1
      exact ⟨m, hm, (foldlM_heldStep_iff sts).1 h⟩
    · rintro ⟨m, hm, hrest⟩
      exact ok_bind_of (heldStep_ok_iff.2 ⟨m, hm, rfl⟩) ((foldlM_heldStep_iff sts).2 hrest)

/-- **`membersHeld` passes exactly on statements that are members' in order.** -/
theorem membersHeld_ok_iff {ms : Array Member} {sts : Array StatementFiles} {held : Array Member} :
    membersHeld ms sts = .ok held ↔ HeldFrom ms #[] sts.toList held := by
  unfold membersHeld
  rw [← Array.foldlM_toList]
  exact foldlM_heldStep_iff sts.toList

/-- **Statements that are members' in order add one member each**: the members after are the members before, then
one member a statement, the statement naming it, a member of the parameters, with exactly one session. -/
theorem heldFrom_new {ms : Array Member} :
    ∀ (sts : List StatementFiles) {held held' : Array Member}, HeldFrom ms held sts held' →
      ∃ new, held'.toList = held.toList ++ new ∧ sts.length = new.length ∧
        ∀ z ∈ sts.zip new, z.1.member = some z.2.name ∧ z.2 ∈ ms ∧ z.1.sessions.size = 1
  | [], held, held', h => by
    obtain rfl : held' = held := h
    exact ⟨[], (List.append_nil _).symm, rfl, fun _ hz => (nomatch hz)⟩
  | st :: sts, held, held', h => by
    obtain ⟨m, hm, hrest⟩ := h
    obtain ⟨new, hnew, hlen, hall⟩ := heldFrom_new sts hrest
    refine ⟨m :: new, ?_, congrArg (· + 1) hlen, fun z hz => ?_⟩
    · rw [hnew, Array.toList_push, List.append_assoc]
      rfl
    · rw [List.zip_cons_cons, List.mem_cons] at hz
      rcases hz with rfl | hz
      · exact ⟨hm.named, Array.mem_of_find?_eq_some hm.member, hm.one⟩
      · exact hall z hz

/-- **Statements `membersHeld` passes are one a member, of one session**: as many members as statements, and each
statement with its member names it, a member of the parameters, and has exactly one session. -/
theorem membersHeld_zip {ms : Array Member} {sts : Array StatementFiles} {held : Array Member}
    (h : membersHeld ms sts = .ok held) :
    sts.size = held.size ∧ ∀ z ∈ (sts.zip held).toList,
      z.1.member = some z.2.name ∧ z.2 ∈ ms ∧ z.1.sessions.size = 1 := by
  obtain ⟨new, hnew, hlen, hall⟩ := heldFrom_new _ (membersHeld_ok_iff.1 h)
  have hnew' : held.toList = new := hnew
  subst hnew'
  refine ⟨hlen, fun z hz => hall z ?_⟩
  rw [Array.toList_zip] at hz
  exact hz

/-- Two members' statements in order: other names, and the first's first unit not after the second's. -/
def InOrder (a b : Member) : Prop := a.name ≠ b.name ∧ a.first ≤ b.first

theorem heldAfter_inOrder {ms held : Array Member} {st : StatementFiles} {m : Member}
    (hm : HeldAfter ms held st m) (hp : held.toList.Pairwise InOrder) : (held.push m).toList.Pairwise InOrder := by
  rw [Array.toList_push, List.pairwise_append]
  refine ⟨hp, List.pairwise_singleton _ _, fun a ha b hb => ?_⟩
  obtain rfl : b = m := List.mem_singleton.1 hb
  constructor
  · have hfresh := hm.fresh
    rw [← Array.any_toList, List.any_eq_false] at hfresh
    exact fun e => hfresh a ha (beq_iff_eq.2 e)
  · have hfol := hm.ordered
    unfold follows at hfol
    cases hb : held.back? with
    | none =>
      rw [Array.back?_eq_none_iff] at hb
      subst hb
      cases ha
    | some prev =>
      rw [hb] at hfol
      dsimp only at hfol
      have hle : prev.first ≤ b.first := Nat.le_of_not_lt fun hlt => by
        rw [decide_eq_true hlt] at hfol
        cases hfol
      obtain ⟨ys, rfl⟩ := Array.back?_eq_some_iff.1 hb
      rw [Array.toList_push] at ha hp
      rcases List.mem_append.1 ha with ha | ha
      · exact Nat.le_trans ((List.pairwise_append.1 hp).2.2 a ha prev (List.mem_singleton.2 rfl)).2 hle
      · obtain rfl : a = prev := List.mem_singleton.1 ha
        exact hle

theorem heldFrom_inOrder {ms : Array Member} :
    ∀ (sts : List StatementFiles) {held held' : Array Member}, HeldFrom ms held sts held' →
      held.toList.Pairwise InOrder → held'.toList.Pairwise InOrder
  | [], held, held', h, hp => by
    obtain rfl : held' = held := h
    exact hp
  | st :: sts, held, held', h, hp => by
    obtain ⟨m, hm, hrest⟩ := h
    exact heldFrom_inOrder sts hrest (heldAfter_inOrder hm hp)

/-- **The members `membersHeld` passes are in the members' order, none twice.** -/
theorem membersHeld_inOrder {ms : Array Member} {sts : Array StatementFiles} {held : Array Member}
    (h : membersHeld ms sts = .ok held) : held.toList.Pairwise InOrder :=
  heldFrom_inOrder _ (membersHeld_ok_iff.1 h) List.Pairwise.nil

/-! ## The run -/

/-- **One statement's sessions at the parameters `p`, after the sessions the run `r` holds**: the statement read
before any session (`statementOf`), then `step` on each session in order. v2's `run` is this from the empty run, then
its proved units held (`runFrom_of_run`). -/
def runFrom (p : _root_.Spot.OneStageZk.Own) (sessions : Array _root_.Spot.OneStageZk.RunFiles)
    (bs : _root_.Core.Blobs) (r : _root_.Spot.OneStageZk.Run) : Flock.V _root_.Spot.OneStageZk.Run :=
  statementOf p bs >>= fun a => sessions.foldlM (fun r s => a.step bs r s) r

/-- **`runFrom` ends accepting every session exactly where the run before did, the statement reads and the sessions
have their verdicts in order at it.** -/
theorem runFrom_iff {p : _root_.Spot.OneStageZk.Own} {sessions : Array _root_.Spot.OneStageZk.RunFiles}
    {bs : _root_.Core.Blobs} {r r' : _root_.Spot.OneStageZk.Run} :
    (runFrom p sessions bs r = .ok r' ∧ r'.all = true) ↔
      r.all = true ∧ ∃ a, statementOf p bs = .ok a ∧ Verdicts bs r (sessions.toList.map fun s => (a, s)) r' := by
  constructor
  · rintro ⟨h, hall⟩
    unfold runFrom at h
    obtain ⟨a, ha, h⟩ := ebind_ok h
    have hl : (sessions.toList.map fun s => (a, s)).foldlM (fun r x => x.1.step bs r x.2) r = .ok r' := by
      rw [List.foldlM_map, Array.foldlM_toList]
      exact h
    obtain ⟨hr, hv⟩ := (steps_iff _).1 ⟨hl, hall⟩
    exact ⟨hr, a, ha, hv⟩
  · rintro ⟨hr, a, ha, hv⟩
    obtain ⟨hl, hall⟩ := (steps_iff _).2 ⟨hr, hv⟩
    rw [List.foldlM_map, Array.foldlM_toList] at hl
    exact ⟨ok_bind_of ha hl, hall⟩

/-- **The one-statement path is the per-member run at one member**: v2's `run` where it accepted every session is
`runFrom` on its sessions from the empty run. -/
theorem runFrom_of_run {p : _root_.Spot.OneStageZk.Own} {sessions : Array _root_.Spot.OneStageZk.RunFiles}
    {pu : Option (Array ℕ)} {bs : _root_.Core.Blobs} {r : _root_.Spot.OneStageZk.Run}
    (h : _root_.Spot.OneStageZk.run p sessions pu bs = .ok r) (hall : r.all = true) :
    runFrom p sessions bs ⟨#[], #[], #[], true⟩ = .ok r := by
  obtain ⟨a, ha, hv, -⟩ := run_statement h hall
  exact (runFrom_iff.2 ⟨rfl, a, ha, hv⟩).1

/-- One statement's run after the run `r`: its sessions at its member's own inputs (`runFrom`), a refusal naming the
member. -/
def stepMember (bs : _root_.Core.Blobs) (r : _root_.Spot.OneStageZk.Run) (z : StatementFiles × Member) :
    Flock.V _root_.Spot.OneStageZk.Run :=
  (runFrom z.2.own z.1.sessions bs r).mapError (s!"member {z.2.name}: {·}")

/-- **The per-member form's run**: the statements' structure held (`membersHeld`), then each statement on its
member's own inputs after the statements before it (`stepMember`), so `step`'s coin-tree keys are fresh across
statements; then, where every session accepted, the union of the statements' proved units held to the evidence's
(`provedHeld`). -/
def runMembers (ms : Array Member) (sts : Array StatementFiles) (want : Array ℕ) (bs : _root_.Core.Blobs) :
    Flock.V _root_.Spot.OneStageZk.Run :=
  membersHeld ms sts >>= fun held =>
  (sts.zip held).foldlM (stepMember bs) ⟨#[], #[], #[], true⟩ >>= fun r =>
  (match r.all with
    | true => _root_.Spot.OneStageZk.provedHeld r.proved want
    | false => pure ()) >>= fun _ => pure r

/-- **The members' sessions, each at its member's statement**, statement by statement: the statement of each member
read off its own inputs (`statementOf`), with each of its statement's sessions. -/
inductive Sessions (bs : _root_.Core.Blobs) :
    List (StatementFiles × Member) → List (StepAt × _root_.Spot.OneStageZk.RunFiles) → Prop
  | nil : Sessions bs [] []
  | cons {z : StatementFiles × Member} {zs : List (StatementFiles × Member)} {a : StepAt}
      {l : List (StepAt × _root_.Spot.OneStageZk.RunFiles)} :
      statementOf z.2.own bs = .ok a → Sessions bs zs l →
      Sessions bs (z :: zs) ((z.1.sessions.toList.map fun s => (a, s)) ++ l)

/-- **The members' runs, one after another, end accepting every session exactly where the run before did and the
members' sessions, each at its member's statement, have their verdicts in order.** -/
theorem members_iff {bs : _root_.Core.Blobs} :
    ∀ (zs : List (StatementFiles × Member)) {r r' : _root_.Spot.OneStageZk.Run},
      (zs.foldlM (stepMember bs) r = .ok r' ∧ r'.all = true) ↔
        r.all = true ∧ ∃ l, Sessions bs zs l ∧ Verdicts bs r l r'
  | [], r, r' => by
    constructor
    · rintro ⟨h, hall⟩
      obtain rfl : r = r' := Except.ok.inj h
      exact ⟨hall, [], .nil, .nil r⟩
    · rintro ⟨hall, l, hs, hv⟩
      cases hs
      cases hv
      exact ⟨rfl, hall⟩
  | z :: zs, r, r' => by
    constructor
    · rintro ⟨h, hall⟩
      rw [List.foldlM_cons] at h
      obtain ⟨r1, h1, h⟩ := ebind_ok h
      obtain ⟨hall1, l, hs, hv⟩ := (members_iff zs).1 ⟨h, hall⟩
      obtain ⟨hr, a, ha, hv1⟩ := runFrom_iff.1 ⟨mapError_ok_iff.1 h1, hall1⟩
      exact ⟨hr, _, .cons ha hs, (verdicts_append _).2 ⟨r1, hv1, hv⟩⟩
    · rintro ⟨hr, l, hs, hv⟩
      cases hs with
      | cons ha hs' =>
        obtain ⟨r1, hv1, hv2⟩ := (verdicts_append _).1 hv
        obtain ⟨h1, hall1⟩ := runFrom_iff.2 ⟨hr, _, ha, hv1⟩
        obtain ⟨h2, hall⟩ := (members_iff zs).2 ⟨hall1, _, hs', hv2⟩
        refine ⟨?_, hall⟩
        rw [List.foldlM_cons]
        exact ok_bind_of (mapError_ok_iff.2 h1) h2

/-- **The per-member run accepts every session exactly where the statements' structure holds, the members' sessions,
each at its member's statement, have their verdicts in order, and the proved units are the evidence's.** -/
theorem runMembers_sessions_iff {ms : Array Member} {sts : Array StatementFiles} {want : Array ℕ}
    {bs : _root_.Core.Blobs} {r : _root_.Spot.OneStageZk.Run} :
    (runMembers ms sts want bs = .ok r ∧ r.all = true) ↔
      ∃ held l, membersHeld ms sts = .ok held ∧ Sessions bs (sts.zip held).toList l ∧
        Verdicts bs ⟨#[], #[], #[], true⟩ l r ∧ _root_.Spot.OneStageZk.provedHeld r.proved want = .ok () := by
  constructor
  · rintro ⟨h, hall⟩
    unfold runMembers at h
    obtain ⟨held, hheld, h⟩ := ebind_ok h
    obtain ⟨r1, hfold, h⟩ := ebind_ok h
    obtain ⟨⟨⟩, hprov, h⟩ := ebind_ok h
    obtain rfl : r1 = r := Except.ok.inj h
    rw [hall] at hprov
    rw [← Array.foldlM_toList] at hfold
    obtain ⟨-, l, hs, hv⟩ := (members_iff _).1 ⟨hfold, hall⟩
    exact ⟨held, l, hheld, hs, hv, hprov⟩
  · rintro ⟨held, l, hheld, hs, hv, hprov⟩
    obtain ⟨hfold, hall⟩ := (members_iff _).2 ⟨rfl, l, hs, hv⟩
    rw [Array.foldlM_toList] at hfold
    refine ⟨?_, hall⟩
    unfold runMembers
    refine ok_bind_of hheld (ok_bind_of hfold (ok_bind_of (a := ()) ?_ rfl))
    rw [hall]
    exact hprov

/-- **The statement `z.1` is exactly the session `x.2`, at its member `z.2`'s statement `x.1`.** -/
def Stated (bs : _root_.Core.Blobs) (z : StatementFiles × Member)
    (x : StepAt × _root_.Spot.OneStageZk.RunFiles) : Prop :=
  statementOf z.2.own bs = .ok x.1 ∧ z.1.sessions = #[x.2]

/-- Statements of exactly one session each have one session a member, each at its member's statement. -/
theorem stated_of_sessions {bs : _root_.Core.Blobs} {zs : List (StatementFiles × Member)}
    {l : List (StepAt × _root_.Spot.OneStageZk.RunFiles)} (h : Sessions bs zs l) :
    (∀ z ∈ zs, z.1.sessions.size = 1) → List.Forall₂ (Stated bs) zs l := by
  induction h with
  | nil => exact fun _ => .nil
  | @cons z zs a l ha _ ih =>
    intro hone
    obtain ⟨s, hs⟩ : ∃ s, z.1.sessions = #[s] := ⟨_, eq_single (hone z List.mem_cons_self)⟩
    rw [hs]
    exact .cons ⟨ha, hs⟩ (ih fun z' hz' => hone z' (List.mem_cons_of_mem _ hz'))

/-- One session a member, each at its member's statement, is the members' sessions. -/
theorem sessions_of_stated {bs : _root_.Core.Blobs} {zs : List (StatementFiles × Member)}
    {l : List (StepAt × _root_.Spot.OneStageZk.RunFiles)} (h : List.Forall₂ (Stated bs) zs l) :
    Sessions bs zs l := by
  induction h with
  | nil => exact .nil
  | @cons z x zs l hzx _ ih =>
    have h := Sessions.cons (z := z) hzx.1 ih
    rw [hzx.2] at h
    exact h

/-- **The per-member form accepts exactly where each member's session, in member order, has an accepting verdict on
its own statement, each session's coins its own**: `runMembers` ends accepting every session exactly where the
statements are members' in order (`membersHeld`: each a member of the parameters, none twice, in the members' order,
of exactly one session); each statement is that one session at the statement read off its member's own inputs
(`Stated`, `statementOf`); the sessions have their verdicts in order (`Verdicts`: each `SessionVerdict` at its own
member's statement, its coin-tree key fresh among the keys of the members' sessions before it, so no two share one,
`Verdicts.keys_nodup`); and the units they proved are the evidence's (`provedHeld`). -/
theorem runMembers_iff {ms : Array Member} {sts : Array StatementFiles} {want : Array ℕ}
    {bs : _root_.Core.Blobs} {r : _root_.Spot.OneStageZk.Run} :
    (runMembers ms sts want bs = .ok r ∧ r.all = true) ↔
      ∃ held l, membersHeld ms sts = .ok held ∧ List.Forall₂ (Stated bs) (sts.zip held).toList l ∧
        Verdicts bs ⟨#[], #[], #[], true⟩ l r ∧ _root_.Spot.OneStageZk.provedHeld r.proved want = .ok () := by
  rw [runMembers_sessions_iff]
  constructor
  · rintro ⟨held, l, hheld, hs, hv, hprov⟩
    exact ⟨held, l, hheld, stated_of_sessions hs fun z hz => ((membersHeld_zip hheld).2 z hz).2.2, hv, hprov⟩
  · rintro ⟨held, l, hheld, hs, hv, hprov⟩
    exact ⟨held, l, hheld, sessions_of_stated hs, hv, hprov⟩

/-- **The per-member form's sessions are one a statement**: as many as the statements, so as many as the sampled
members. -/
theorem stated_length {bs : _root_.Core.Blobs} {ms : Array Member} {sts : Array StatementFiles}
    {held : Array Member} {l : List (StepAt × _root_.Spot.OneStageZk.RunFiles)}
    (hheld : membersHeld ms sts = .ok held) (h : List.Forall₂ (Stated bs) (sts.zip held).toList l) :
    l.length = sts.size := by
  rw [← h.length_eq, Array.length_toList, Array.size_zip, ← (membersHeld_zip hheld).1, Nat.min_self]

/-! ## What the registration fixes of a member before the sample -/

/-- The header `h`'s `units.indices` are exactly the units `us`, in order: `Spot.OneStageZk.indicesHeld` at a
member's units in place of the population's. -/
def unitsHeld (us : Array ℕ) (h : Json) : Bool :=
  match _root_.Spot.getD (_root_.Spot.getD h "units") "indices" with
  | .arr is => is.size == us.size && (List.range us.size).all fun i => (is[i]? >>= _root_.Spot.nat?) == us[i]?
  | _ => false

/-- `indicesHeld` is `unitsHeld` at the population's units in order. -/
theorem indicesHeld_eq {rec h : Json} {n : ℕ} (hn : _root_.Spot.nat? (_root_.Spot.getD rec "population") = some n) :
    _root_.Spot.OneStageZk.indicesHeld rec h = unitsHeld (Array.range n) h := by
  unfold _root_.Spot.OneStageZk.indicesHeld unitsHeld
  rw [hn]
  cases _root_.Spot.getD (_root_.Spot.getD h "units") "indices" with
  | arr is =>
    dsimp only
    rw [Array.size_range]
    refine congrArg (fun b => is.size == n && b) ?_
    refine Bool.eq_iff_iff.2 ?_
    rw [List.all_eq_true, List.all_eq_true]
    refine forall_congr' fun i => forall_congr' fun hi => ?_
    have hlt : i < (Array.range n).size := by
      rw [Array.size_range]
      exact List.mem_range.1 hi
    rw [Array.getElem?_eq_getElem hlt, Array.getElem_range]
  | _ => rfl

/-- **A public file is a slice of the registration's**: its header's scheme, `frame_v3` roots and their domains, and
instances are the slice `slice`'s (`Record.matchesPublic`, with no failing code: the slice is a registration record
of the member's own, its scheme, its ports' roots and domains under the member's key and its number of proof blocks
as the population), and its `units.indices` are exactly the units `us` in order (`unitsHeld`). What `publicHeld` holds
the one statement's public file to, at a slice and its units in place of the whole registration and its population
(`publicHeld_iff`). -/
def SliceHeld (slice : Json) (us : Array ℕ) (pubB : ByteArray) : Prop :=
  ∃ h, _root_.Spot.FilePart.header pubB = .ok h ∧
    (_root_.Spot.Record.matchesPublic slice (_root_.Spot.getD h "scheme")
      (_root_.Spot.getD (_root_.Spot.getD h "frame_v3") "roots")
      (_root_.Spot.getD (_root_.Spot.getD h "frame_v3") "domain_ids")
      (_root_.Spot.getD h "instances")).codes.isEmpty = true ∧
    unitsHeld us h = true

/-- A check that refuses on `!b` passed only at `b`. -/
theorem of_not_bnot {b : Bool} (h : ¬((!b) = true)) : b = true := by
  cases b with
  | true => rfl
  | false => exact absurd rfl h

/-- **`publicHeld`'s whole-registration check is the slice check at the whole registration**: the one-statement
path's public file is held to the registration record itself and the population's units in order. -/
theorem publicHeld_iff {rec : Json} {n : ℕ} (hn : _root_.Spot.nat? (_root_.Spot.getD rec "population") = some n)
    {pubB : ByteArray} :
    _root_.Spot.OneStageZk.publicHeld rec pubB = .ok () ↔ SliceHeld rec (Array.range n) pubB := by
  constructor
  · intro h
    unfold _root_.Spot.OneStageZk.publicHeld at h
    obtain ⟨hd, hh, h⟩ := ebind_ok h
    dsimp only at h
    split at h
    · cases h
    · rename_i hc
      split at h
      · cases h
      · rename_i hi
        exact ⟨hd, hh, of_not_bnot hc, by rw [← indicesHeld_eq hn]; exact of_not_bnot hi⟩
  · rintro ⟨hd, hh, hc, hi⟩
    rw [← indicesHeld_eq hn] at hi
    unfold _root_.Spot.OneStageZk.publicHeld
    refine ok_bind_of hh ?_
    dsimp only
    rw [hc, hi]
    try rfl

/-- **A slice holds the public file's instances to the slice's population**: a member's public file has the member's
number of proof blocks. -/
theorem SliceHeld.population {slice : Json} {us : Array ℕ} {pubB : ByteArray} (h : SliceHeld slice us pubB) :
    ∃ hd, _root_.Spot.FilePart.header pubB = .ok hd ∧
      _root_.Spot.jeq (_root_.Spot.getD hd "instances") (_root_.Spot.getD slice "population") = true := by
  obtain ⟨hd, hh, hc, -⟩ := h
  exact ⟨hd, hh, matchesPublic_population hc⟩

/-- **A member's statement is fixed by the registration before the sample**: the public file its own inputs name is a
blob held to the member's slice `slice` of the registration and to the member's units (`SliceHeld`). The slice is the
registration's, recorded before any coin; the member's inputs and units are the parameters'. -/
def Member.Fixed (m : Member) (slice : Json) (bs : _root_.Core.Blobs) : Prop :=
  ∃ pubB, bs.get m.own.pub "public file" = .ok pubB ∧ SliceHeld slice m.indices pubB

/-- **The statement a fixed member's session is held to reads the registration's slice**: its inputs' public file is
the one held to the member's slice and units. -/
theorem Member.Fixed.statement {m : Member} {slice : Json} {bs : _root_.Core.Blobs} (h : m.Fixed slice bs)
    {a : StepAt} (ha : statementOf m.own bs = .ok a) : SliceHeld slice m.indices a.I.publicFile := by
  obtain ⟨pubB, hp, hs⟩ := h
  obtain ⟨pubB', hp', hI⟩ := statementOf_public ha
  rw [hp] at hp'
  obtain rfl : pubB = pubB' := Except.ok.inj hp'
  rw [hI]
  exact hs

/-- **The per-member form's acceptance on the proofs, as it would be with the gate lifted**: the run accepts every
session (`runMembers`), and each member with a statement is fixed by its slice of the registration
(`Member.Fixed`, at `slice m`). -/
def Accepts (slice : Member → Json) (ms : Array Member) (sts : Array StatementFiles) (want : Array ℕ)
    (bs : _root_.Core.Blobs) (r : _root_.Spot.OneStageZk.Run) : Prop :=
  (runMembers ms sts want bs = .ok r ∧ r.all = true) ∧
    ∀ held, membersHeld ms sts = .ok held → ∀ m ∈ held, m.Fixed (slice m) bs

/-- **The per-member form accepts exactly where each sampled member's session, in member order, has an accepting
verdict on its own registered statement, each session's coins its own**: `runMembers_iff`, with each member's
statement fixed by its slice of the registration before the sample. -/
theorem accepts_iff {slice : Member → Json} {ms : Array Member} {sts : Array StatementFiles} {want : Array ℕ}
    {bs : _root_.Core.Blobs} {r : _root_.Spot.OneStageZk.Run} :
    Accepts slice ms sts want bs r ↔
      ∃ held l, membersHeld ms sts = .ok held ∧ (∀ m ∈ held, m.Fixed (slice m) bs) ∧
        List.Forall₂ (Stated bs) (sts.zip held).toList l ∧
        Verdicts bs ⟨#[], #[], #[], true⟩ l r ∧ _root_.Spot.OneStageZk.provedHeld r.proved want = .ok () := by
  unfold Accepts
  rw [runMembers_iff]
  constructor
  · rintro ⟨⟨held, l, hheld, hs, hv, hprov⟩, hfix⟩
    exact ⟨held, l, hheld, hfix held hheld, hs, hv, hprov⟩
  · rintro ⟨held, l, hheld, hfix, hs, hv, hprov⟩
    refine ⟨⟨held, l, hheld, hs, hv, hprov⟩, fun held' hheld' => ?_⟩
    rw [hheld] at hheld'
    obtain rfl : held = held' := Except.ok.inj hheld'
    exact hfix

/-- **Each session's statement reads its member's slice of the registration**: where every member of the statements
is fixed, the public file of the statement each session is held to is the one held to its member's slice and units. -/
theorem stated_fixed {slice : Member → Json} {bs : _root_.Core.Blobs} {zs : List (StatementFiles × Member)}
    {l : List (StepAt × _root_.Spot.OneStageZk.RunFiles)} (h : List.Forall₂ (Stated bs) zs l) :
    (∀ z ∈ zs, z.2.Fixed (slice z.2) bs) →
      List.Forall₂ (fun z x => SliceHeld (slice z.2) z.2.indices x.1.I.publicFile) zs l := by
  induction h with
  | nil => exact fun _ => .nil
  | @cons z x zs l hzx _ ih =>
    intro hfix
    exact .cons ((hfix z List.mem_cons_self).statement hzx.1)
      (ih fun z' hz' => hfix z' (List.mem_cons_of_mem _ hz'))

/-- **An accepted per-member run is, session by session in member order, a verdict on the member's registered
statement**: each statement is one session at its member's statement, whose public file is held to the member's
slice of the registration and its units, and the sessions have their verdicts in order. -/
theorem accepts_fixed {slice : Member → Json} {ms : Array Member} {sts : Array StatementFiles} {want : Array ℕ}
    {bs : _root_.Core.Blobs} {r : _root_.Spot.OneStageZk.Run} (h : Accepts slice ms sts want bs r) :
    ∃ held l, membersHeld ms sts = .ok held ∧ List.Forall₂ (Stated bs) (sts.zip held).toList l ∧
      List.Forall₂ (fun z x => SliceHeld (slice z.2) z.2.indices x.1.I.publicFile) (sts.zip held).toList l ∧
      Verdicts bs ⟨#[], #[], #[], true⟩ l r := by
  obtain ⟨held, l, hheld, hfix, hs, hv, -⟩ := accepts_iff.1 h
  refine ⟨held, l, hheld, hs, stated_fixed hs fun z hz => hfix z.2 ?_, hv⟩
  rw [Array.toList_zip] at hz
  exact Array.mem_toList_iff.1 (List.of_mem_zip hz).2

/-! ## The statements' structure, on stand-in members -/

private def ownT : _root_.Spot.OneStageZk.Own := ⟨Json.null, ByteArray.empty, Json.null, "", "", none⟩

private def memberT (name : String) (units : Array (ℕ × ℕ)) : Member := ⟨name, units, ownT⟩

private def statementT (member : String) (sessions : ℕ := 1) : StatementFiles :=
  ⟨some member, (Array.range sessions).map fun i => ⟨s!"{member}/{i}", "", #[]⟩⟩

private def membersT : Array Member :=
  #[memberT "m2" #[(2, 3)], memberT "m10" #[(10, 11)], memberT "m0" #[(5, 6), (0, 1)]]

private def namesT (r : Flock.V (Array Member)) : Option (Array String) := r.toOption.map (·.map (·.name))

#guard (memberT "m0" #[(5, 6), (0, 1)]).first == 0
#guard (memberT "m0" #[(5, 7), (0, 1)]).indices == #[5, 6, 0]
#guard namesT (membersHeld membersT #[statementT "m0", statementT "m2", statementT "m10"]) == some #["m0", "m2", "m10"]
#guard namesT (membersHeld membersT #[statementT "m2", statementT "m10"]) == some #["m2", "m10"]
#guard namesT (membersHeld membersT #[]) == some #[]
#guard namesT (membersHeld membersT #[statementT "m2" 0, statementT "m10"]) == none
#guard namesT (membersHeld membersT #[statementT "m2", statementT "m0"]) == none
#guard namesT (membersHeld membersT #[statementT "m0", statementT "m10", statementT "m2"]) == none
#guard namesT (membersHeld membersT #[statementT "m0", statementT "m0"]) == none
#guard namesT (membersHeld membersT #[statementT "m0" 2]) == none
#guard namesT (membersHeld membersT #[statementT "m3"]) == none
#guard namesT (membersHeld membersT #[⟨none, #[]⟩]) == none

end Verity.Sampled.PerMember
