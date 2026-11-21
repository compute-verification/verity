module

public import CoreProtocol.Blobs
public import Spot.OneStage
public import Spot.ChallengerTrees
public import Spot.Slots
public import Spot.FilePart
public import Spot.CircuitKept
public import Spot.OneStageZkStatement
public import Spot.OneStageZkKept
public import Spot.MembersWires
public import Flock
public import Flock.JsonParse
public import Std.Data.HashMap
meta import CoreProtocol.Blobs
meta import Spot.OneStage
meta import Spot.ChallengerTrees
meta import Spot.Slots
meta import Spot.FilePart
meta import Spot.OneStageZkStatement
meta import Spot.OneStageZkKept
meta import Spot.MembersWires
meta import Flock
meta import Flock.JsonParse
meta import Std.Data.HashMap

/-!
# One-stage v2: the run over the members and the evidence's parts

The run over the members (`runMembers`), the evidence's parts (`evidenceParts`, `sessionEvidence`), the members' trees
and layout (`membersTreesHeld`, `laidHeld`) and one member's registered ports (`classStep`), in the namespace
`Spot.OneStageZk`; `Spot/OneStageZk.lean` imports this module.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Spot.OneStageZk

open Lean (Json)
open Flock (V)

/-- The statement `membersFiles` read for the member `m`, the first under its name, as `heldStep` finds a statement's
member; a refusal where it read none. -/
def statementAt (files : List (String × StepAt)) (m : Member) : V StepAt :=
  match files.lookup m.name with
  | some a => pure a
  | none => throw "the parameters name no such member"

/-- **One statement's sessions, after the sessions the run `r` holds**: the statement `st`, read before any session,
then `step` on each session in order. -/
def runFrom (st : V StepAt) (sessions : Array RunFiles) (bs : Core.Blobs) (r : Run) : V Run :=
  st >>= fun a => sessions.foldlM (fun r s => a.step bs r s) r

/-- One statement's run after the run `r`: its sessions at its member's statement `st` (`runFrom`), a refusal naming
the member. -/
def stepMember (st : Member → V StepAt) (bs : Core.Blobs) (r : Run) (z : StatementFiles × Member) : V Run :=
  (runFrom (st z.2) z.1.sessions bs r).mapError (s!"member {z.2.name}: {·}")

/-- Whether the sample `want`, the evidence's `proved_units`, holds one of the member's units. -/
def Member.sampled (m : Member) (want : Array Nat) : Bool :=
  want.any fun u => m.units.any fun x => decide (x.1 ≤ u) && decide (u < x.2)

/-- The refusal of a statement of the member `name`, none of whose units the sample holds (`sampledHeld`). -/
def unsampled (name : String) : String :=
  s!"member {name}: the sample holds none of its units, and a member the sample missed has no statement"

/-- **Every statement's member is one the sample holds**: each of `held`, the statements' members in order
(`membersHeld`), has a unit among the evidence's `proved_units` (`want`); the first that has none refuses, by name
(`unsampled`). A session proves the units of the sample that are its member's, so a statement of a member the sample
missed is a session that proves no unit: the units proved would still be the sample (`provedHeld`), with one more
session than the sample's members. -/
def sampledHeld (held : Array Member) (want : Array Nat) : V Unit :=
  match held.find? fun m => !m.sampled want with
  | some m => throw (unsampled m.name)
  | none => pure ()

/-- **The per-member form's run**, at each member's `slice` of the registration: the statements' structure held
(`membersHeld`), each statement's member one the sample holds (`sampledHeld`), every member's files held, sampled or
not (`membersFiles`), then each statement at its member's statement as that pass read it, after the statements before
it (`stepMember`), so `step`'s coin-tree keys are fresh across statements; then, where every session accepted, the
union of the statements' proved units held to the evidence's `want` (`provedHeld`). Every member's files are read
once, whichever members the sample holds. -/
def runMembers (slice : Member → Json) (ms : Array Member) (sts : Array StatementFiles) (want : Array Nat)
    (bs : Core.Blobs) : V Run :=
  membersHeld ms sts >>= fun held =>
  sampledHeld held want >>= fun _ =>
  membersFiles slice bs ms.toList >>= fun files =>
  (sts.zip held).foldlM (stepMember (statementAt files) bs) ⟨#[], #[], #[], true⟩ >>= fun r =>
  (match r.all with
    | true => provedHeld r.proved want
    | false => pure ()) >>= fun _ => pure r

/-- **What the per-member form's run holds of a member's files**, at its `slice` of the registration on the blobs
`bs`: its statement reads (`Member.statement`), and at that statement the scope check, the files' ports and roots, the
form the sampled audit's knowledge soundness covers (`Flock.KsScope.filesOf`) and the population check pass. Nothing of
it reads a session, a coin or the sample. -/
def Member.FilesHeld (m : Member) (slice : Json) (bs : Core.Blobs) : Prop :=
  ∃ a : StepAt, m.statement slice bs = .ok a ∧ a.scope = .ok () ∧ a.ksFiles = .ok () ∧
    Flock.KsScope.filesOf (Flock.ProvedScope.circuitOf (Flock.Witness.zkTags a.I) a.I.circuitFile a.I.tables)
      (Flock.Witness.zkTags a.I) a.I.publicFile a.I.count = .ok () ∧ a.population = .ok ()

/-- A refusal rewritten is no acceptance: what `mapError` accepts, its argument did. -/
theorem mapError_ok {α : Type} {f : String → String} {x : V α} {a : α} (h : x.mapError f = .ok a) : x = .ok a := by
  cases x with
  | error e => cases h
  | ok b => exact h

/-- A statement `statementOf` reads is `StepAt.ofOwn`'s, at the blobs' circuit and public files and the lookup tables
it is handed. -/
theorem statementOf_ofOwn {p : Own} {population : ByteArray → V Unit} {tables : List (String × ByteArray)}
    {bs : Core.Blobs} {a : StepAt} (h : statementOf p population tables bs = .ok a) :
    ∃ tags0 circuitB pubB rd, a = StepAt.ofOwn p tags0 circuitB pubB tables rd (population pubB) := by
  unfold statementOf at h
  obtain ⟨_, _, h⟩ := Flock.HmNets.bind_ok h
  split at h
  · cases h
  · rename_i tags0 _
    split at h
    · cases h
    · split at h
      · cases h
      · obtain ⟨circuitB, _, h⟩ := Flock.HmNets.bind_ok h
        obtain ⟨pubB, _, h⟩ := Flock.HmNets.bind_ok h
        split at h
        · cases h
        · obtain ⟨rd, _, h⟩ := Flock.HmNets.bind_ok h
          exact ⟨tags0, circuitB, pubB, rd, (Except.ok.inj h).symm⟩

/-- **A member's statement holds its circuit file's checks at the parse of its own inputs' files**: its form check's
result (`ksForm`) is `Flock.KsScope.filesOf`'s on the circuit those files parse to, and where they parse, the
circuit's units a proof block are the parameters' (`Member.blockHeld`). -/
theorem Member.statement_parse {m : Member} {slice : Json} {bs : Core.Blobs} {a : StepAt}
    (h : m.statement slice bs = .ok a) :
    a.ksForm = Flock.KsScope.filesOf
        (Flock.ProvedScope.circuitOf (Flock.Witness.zkTags a.I) a.I.circuitFile a.I.tables)
        (Flock.Witness.zkTags a.I) a.I.publicFile a.I.count ∧
      ∀ c, Flock.ProvedScope.circuitOf (Flock.Witness.zkTags a.I) a.I.circuitFile a.I.tables = .ok c → c.g = m.g := by
  unfold Member.statement at h
  obtain ⟨a', hs, h⟩ := Flock.HmNets.bind_ok h
  obtain ⟨⟨⟩, hb, h⟩ := Flock.HmNets.bind_ok h
  have ha : a' = a := Except.ok.inj h
  subst ha
  obtain ⟨tables, -, hs⟩ := Flock.HmNets.bind_ok hs
  obtain ⟨tags0, circuitB, pubB, rd, rfl⟩ := statementOf_ofOwn hs
  refine ⟨rfl, fun c hc => ?_⟩
  have hb' : m.blockHeld (some c.g) = .ok () :=
    (congrArg (fun r : V Flock.Circuit => m.blockHeld (r.toOption.map fun x : Flock.Circuit => x.g)) hc).symm.trans hb
  simp only [Member.blockHeld] at hb'
  split at hb'
  · cases hb'
  · rename_i hne
    simpa using hne

/-- A member whose statement `Member.files` returns has its files held. -/
theorem Member.files_ok {m : Member} {slice : Json} {bs : Core.Blobs} {a : StepAt} (h : m.files slice bs = .ok a) :
    m.FilesHeld slice bs := by
  unfold Member.files at h
  obtain ⟨a', hs, h⟩ := Flock.HmNets.bind_ok (mapError_ok h)
  obtain ⟨⟨⟩, hf, h⟩ := Flock.HmNets.bind_ok h
  have ha : a' = a := Except.ok.inj h
  subst ha
  unfold StepAt.filesHeld at hf
  obtain ⟨⟨⟩, hsc, hf⟩ := Flock.HmNets.bind_ok hf
  obtain ⟨⟨⟩, hk, hf⟩ := Flock.HmNets.bind_ok hf
  obtain ⟨⟨⟩, hks, hf⟩ := Flock.HmNets.bind_ok hf
  obtain ⟨⟨⟩, -, hp⟩ := Flock.HmNets.bind_ok hf
  exact ⟨a', hs, hsc, hk, (Member.statement_parse hs).1.symm.trans hks, hp⟩

/-- **A member whose statement `Member.files` returns has its public file's header and rows as a session's setup
takes them**: `Flock.PortRoots.headerOf` passes on the circuit its own inputs' files parse to, at no session. -/
theorem Member.files_header {m : Member} {slice : Json} {bs : Core.Blobs} {a : StepAt}
    (h : m.files slice bs = .ok a) :
    Flock.PortRoots.headerOf (Flock.ProvedScope.circuitOf (Flock.Witness.zkTags a.I) a.I.circuitFile a.I.tables)
      (Flock.Witness.zkTags a.I) a.I.publicFile = .ok () := by
  unfold Member.files at h
  obtain ⟨a', hs, h⟩ := Flock.HmNets.bind_ok (mapError_ok h)
  obtain ⟨⟨⟩, hf, h⟩ := Flock.HmNets.bind_ok h
  have ha : a' = a := Except.ok.inj h
  subst ha
  unfold StepAt.filesHeld at hf
  obtain ⟨⟨⟩, -, hf⟩ := Flock.HmNets.bind_ok hf
  obtain ⟨⟨⟩, -, hf⟩ := Flock.HmNets.bind_ok hf
  obtain ⟨⟨⟩, -, hf⟩ := Flock.HmNets.bind_ok hf
  obtain ⟨⟨⟩, hh, -⟩ := Flock.HmNets.bind_ok hf
  unfold Member.statement at hs
  obtain ⟨a'', hst, hb⟩ := Flock.HmNets.bind_ok hs
  obtain ⟨⟨⟩, -, hb⟩ := Flock.HmNets.bind_ok hb
  have ha : a'' = a' := Except.ok.inj hb
  subst ha
  obtain ⟨_, -, hst⟩ := Flock.HmNets.bind_ok hst
  obtain ⟨tags0, circuitB, pubB, rd, rfl⟩ := statementOf_ofOwn hst
  exact hh

/-- Where `membersFiles` reads the members `ms`, each of them has its files held. -/
theorem membersFiles_ok {slice : Member → Json} {bs : Core.Blobs} {ms : List Member}
    {files : List (String × StepAt)} (h : membersFiles slice bs ms = .ok files) :
    ∀ m ∈ ms, m.FilesHeld (slice m) bs := by
  induction ms generalizing files with
  | nil => intro m hm; cases hm
  | cons m0 ms ih =>
    intro m hm
    rw [membersFiles] at h
    obtain ⟨a, ha, h⟩ := Flock.HmNets.bind_ok h
    obtain ⟨as, hrest, _⟩ := Flock.HmNets.bind_ok h
    rcases List.mem_cons.1 hm with rfl | hm
    · exact Member.files_ok ha
    · exact ih hrest m hm

/-- **The per-member form's run holds every member's files, sampled or not**: where `runMembers` returns a run, each
member of the parameters has its files held (`Member.FilesHeld`) at its slice, whatever the statements name. -/
theorem runMembers_files {slice : Member → Json} {ms : Array Member} {sts : Array StatementFiles} {want : Array Nat}
    {bs : Core.Blobs} {r : Run} (h : runMembers slice ms sts want bs = .ok r) :
    ∀ m ∈ ms, m.FilesHeld (slice m) bs := by
  unfold runMembers at h
  obtain ⟨_, _, h⟩ := Flock.HmNets.bind_ok h
  obtain ⟨_, _, h⟩ := Flock.HmNets.bind_ok h
  obtain ⟨files, hf, _⟩ := Flock.HmNets.bind_ok h
  exact fun m hm => membersFiles_ok hf m (Array.mem_def.1 hm)

/-- Where `sampledHeld` passes, the sample holds a unit of each of the members. -/
theorem sampledHeld_ok {held : Array Member} {want : Array Nat} (h : sampledHeld held want = .ok ()) :
    ∀ m ∈ held, m.sampled want = true := by
  intro m hm
  unfold sampledHeld at h
  cases hf : held.find? (fun m => !m.sampled want) with
  | some m' =>
    rw [hf] at h
    cases h
  | none =>
    have hn := Array.find?_eq_none.1 hf m hm
    cases hd : m.sampled want with
    | true => rfl
    | false => exact absurd (by simp [hd]) hn

/-- **Every statement of the per-member form's run is of a member the sample holds**: where `runMembers` returns a
run, the statements' members in order (`membersHeld`) each have a unit among the evidence's `want`. With the proved
units held to `want` (`runMembers_proved`), the statements are the members the sample holds, each once. -/
theorem runMembers_sampled {slice : Member → Json} {ms : Array Member} {sts : Array StatementFiles} {want : Array Nat}
    {bs : Core.Blobs} {r : Run} (h : runMembers slice ms sts want bs = .ok r) :
    ∃ held, membersHeld ms sts = .ok held ∧ ∀ m ∈ held, m.sampled want = true := by
  unfold runMembers at h
  obtain ⟨held, hh, h⟩ := Flock.HmNets.bind_ok h
  obtain ⟨⟨⟩, hd, _⟩ := Flock.HmNets.bind_ok h
  exact ⟨held, hh, sampledHeld_ok hd⟩

/-- **Where every session of the per-member form's run accepted, the units they proved are the evidence's**: the
units the accepted sessions proved (`Run.proved`, each session's read off its member's public file, `push`), sorted,
are `want`. So a unit of `want` no accepted session proved refuses the run, as does a unit two sessions prove. -/
theorem runMembers_proved {slice : Member → Json} {ms : Array Member} {sts : Array StatementFiles} {want : Array Nat}
    {bs : Core.Blobs} {r : Run} (h : runMembers slice ms sts want bs = .ok r) (hall : r.all = true) :
    r.proved.qsort (· < ·) = want := by
  unfold runMembers at h
  obtain ⟨_, _, h⟩ := Flock.HmNets.bind_ok h
  obtain ⟨_, _, h⟩ := Flock.HmNets.bind_ok h
  obtain ⟨_, _, h⟩ := Flock.HmNets.bind_ok h
  obtain ⟨r', _, h⟩ := Flock.HmNets.bind_ok h
  obtain ⟨⟨⟩, hp, h⟩ := Flock.HmNets.bind_ok h
  have hr : r' = r := Except.ok.inj h
  subst hr
  rw [hall] at hp
  have hp' : provedHeld _ want = .ok () := hp
  unfold provedHeld at hp'
  split at hp'
  · cases hp'
  · rename_i hne
    simpa using hne

/-- **The bytes a closing round's digest is of**: the number of `parts` in an 8-byte big-endian word, then each
part's length in such a word and its bytes, as a frame holds its parts (`Flock.FV3.input`). `evidenceParts` reads the
parts back, so the digest fixes each part's bytes, not a digest of them. -/
def evidenceBytes (parts : Array ByteArray) : ByteArray :=
  parts.foldl (fun acc p => acc ++ Flock.be64 p.size.toUInt64 ++ p) (Flock.be64 parts.size.toUInt64)

/-- `n` parts of `b` from offset `o` to its end, each an 8-byte length and its bytes; none where `b` is not that. -/
def partsFrom (b : ByteArray) : Nat → Nat → Option (List ByteArray)
  | 0, o => if o == b.size then some [] else none
  | n + 1, o =>
    if o + 8 ≤ b.size then
      let stop := o + 8 + Flock.Draw.beNat b o 8
      if stop ≤ b.size then (partsFrom b n stop).map (b.extract (o + 8) stop :: ·) else none
    else none

/-- **The decoder of `evidenceBytes`**: the parts `b` holds, none where `b` is not some parts' bytes. -/
def evidenceParts (b : ByteArray) : Option (Array ByteArray) :=
  if 8 ≤ b.size then (partsFrom b (Flock.Draw.beNat b 0 8) 8).map (·.toArray) else none

/-- The refusal of a session's evidence `evidenceBytes` can't hold. -/
def EVIDENCE_FITS : String :=
  "its evidence holds a file of 2^64 bytes or more, or as many files, past the 8-byte word its closing round's " ++
    "digest holds each length in"

/-- **A session's evidence, the bytes its closing round is opened after the SHA-512 of** (`evidenceBytes`): its
record's bytes, then, for each rep file it names, in the order its `reps` holds them (their names', sorted), the file's
name in UTF-8 and its bytes, each the blob the verdict reads it from (`step`, `filesOf`). A refusal where no blob holds
one, as `step` refuses a record, or where a length is past a word (`EVIDENCE_FITS`). -/
def sessionEvidence (bs : Core.Blobs) (s : RunFiles) : V ByteArray := do
  let record ← bs.get s.record s!"session {s.name}'s record"
  let reps ← s.reps.mapM fun (f, d) => do
    let b ← bs.get d s!"session {s.name}'s files: {f}"
    pure #[f.toUTF8, b]
  let parts := #[record] ++ reps.flatten
  if decide (WORD ≤ parts.size) || parts.any (fun part => decide (WORD ≤ part.size)) then
    throw s!"session {s.name}: {EVIDENCE_FITS}"
  pure (evidenceBytes parts)

/-- The statements' sessions held to the challenger's trees `ts` and its release log's lines `log`, each statement
with its member: each session's record on the blobs, refused as `treesHeld` refuses it, at its tables with the
`root_B` lengths its statement fixes on its member's own files and lookup tables (`ownRootBTables`) and the SHA-512
of its evidence (`sessionEvidence`), then `Challenger.membersTreeReason`: each session's coins one tree's, the tree
closed after that digest, and the trees' lines, each tree's own and its rounds', one tree after another in the
sessions' order. -/
def membersTreesHeld (ts : Array Challenger.LoggedTree) (log : List String) (sts : Array (StatementFiles × Member))
    (bs : Core.Blobs) : V Unit := do
  let mut named : Array (String × Array (String × Option Nat) × Json × String) := #[]
  for (st, m) in sts do
    for s in st.sessions do
      let some record := String.fromUTF8? (← bs.get s.record s!"session {s.name}'s record")
        | throw s!"session {s.name}'s record is not UTF-8"
      let evidence ← sessionEvidence bs s
      named := named.push (s.name, ownRootBTables m.own bs record,
        (Flock.JsonParse.parse record).toOption.getD Json.null, sha512Hex evidence)
  if let some why := Challenger.membersTreeReason ts log named then throw why

/-- The refusal of members that are not the registration's population, a range each (`laidHeld`). -/
def MEMBERS_LAID : String :=
  "members: the members' units are not the registration's population, each member's one range of it: a member is " ++
    "the units [base, base + n), and the members' counts sum to the population"

/-- **The members laid over the registration's population**: each member's units are one range, within the record's
population, and the members' counts sum to it. The parameters' members share no unit (`membersOf`), so every unit of
the population is exactly one member's, and the members in the order of their first units lie one after another from
unit 0: a member's units are `[base, base + n)`, `base` the count of the members before it. -/
def laidHeld (ms : Array Member) (rec : Json) : V Unit :=
  match nat? (getD rec "population") with
  | none => throw "the record names no population"
  | some n =>
    if ms.all (fun m => m.units.size == 1 && m.units.all (·.2 ≤ n)) && ms.foldl (fun a m => a + m.count) 0 == n then
      pure ()
    else throw MEMBERS_LAID

/-- The refusal of two members' ports of one registered value that are not one tree's (`classHeld`): `diff` is the
first field they differ in (`Flock.RecVStar.portDiff?`). -/
def classTwo (first second value diff : String) : String :=
  s!"registered: members {first} and {second} read the registered value {value} as two ({diff}): a value the " ++
    "window registers is one tree, read by every member at one root"

/-- The member `name`'s registered port `q` after the ports before it: `acc` holds, for each value read so far, the
first port of it and its member's name. A port of a value `acc` holds must be that port in the value's root, leaves,
row width, domain and leaf schema (`Flock.RecVStar.portDiff?`, without the units' reads). -/
def classPort (name : String) (acc : Std.HashMap String (String × Flock.Registered.Port))
    (q : Flock.Registered.Port) : V (Std.HashMap String (String × Flock.Registered.Port)) :=
  match acc.get? q.value with
  | none => pure (acc.insert q.value (name, q))
  | some (first, q0) =>
    match Flock.RecVStar.portDiff? q0 q false with
    | none => pure acc
    | some diff => throw (classTwo first name q.value diff)

/-- One member's registered ports after the members before it (`classPort` on each); a member whose registered file
is no registered reads refuses, and one with none reads nothing. -/
def classStep (acc : Std.HashMap String (String × Flock.Registered.Port)) (m : Member) :
    V (Std.HashMap String (String × Flock.Registered.Port)) :=
  match m.own.registered with
  | none => pure acc
  | some j =>
    (Flock.Registered.ofJson j).mapError (s!"member {m.name}: {·}") >>= fun own => own.foldlM (classPort m.name) acc

end Spot.OneStageZk
