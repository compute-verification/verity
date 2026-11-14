/-
Sampled proofs' one-stage audit on the proofs themselves (`verity/sampled-proofs/one-stage/v2`), a `Core.BlobProtocol`: the
v1 audit (`OneStage`) with the verdicts computed here, from the statement's files, rather than read from the evidence.

* **params**: v1's, plus `proofs`, exactly `{"circuit", "public", "registered"}`: the SHA-512s the auditor committed for
  the circuit and public files (lowercase hex of 64 bytes) and the verifier's own registered file (`flock-verify
  --registered`'s object, or null for a program that reads no registered value). The partition must not be null: the
  sessions are verified against the verifier's own partition object; and v1's `challenger`, with its `window_root`, must
  be named (`NO_CHALLENGER`), its key at the deployed LMS parameters, `Lms.deployed` (`KEY_PARAMS`). The sampling
  design must be a `subset` one, or a rate stated as text that names one over the population (`rate:NUM/DEN[:T]`): a
  `stratified` or `work` one, written as an object or as a rate, is refused by its kind (`SUBSET_ONLY`,
  `paramsOfJson_subset`), since
  the one statement's soundness statement covers a uniform sample of a fixed count alone;
* **commitment**: v1's (the registration record);
* **evidence**: v1's without `verdicts`, its `coins` within the challenger evidence's limits (`Challenger.withinAt`
  at `Challenger.LIMITS`), plus `proofs`, exactly `{"sessions"}`: for each of v1's `sessions` in order,
  `{"name", "record", "reps"}`: its name, its `session.json`'s SHA-512 and `{file: SHA-512}` naming at least each rep
  file the statement's session reads (`<stream with / as .>.bin`). The sampled proof blocks are proved in one session, whose
  statement is theirs: the check refuses evidence naming none or several (`oneSession`) before it reads a tree;
* **blobs**: the files those digests name (`Core.Blobs`). The registered circuit names the lookup tables its session
  reads (its META's `lookups`, `lookupsOf`), so blobs whose pinned MUFU tables (`tablesOf`, each under its SHA-512 as
  `flock-verify --archive` finds one) are other than those refuse (`tablesHeld`, `NO_TABLES`) before the session is set
  up: the tables the session `run` sets up reads are the registered circuit's;
* **coins** (`coinsOf`): v1's (`OneStage.coinsOf`), what `Challenger.drawCoins` reads from the evidence's `coins` under
  the challenger's key, on the verifier's window; none read refuses. The check below runs at them (`checkBlobsAt`): v1's
  `checkAt` refuses a sample other than the one they give before v1's `check`;
* **trees** (`treesOf`, `orderHeld`, `treesHeld`): the coin trees that same session's signed release log shows the
  challenger sampled and opened (`Challenger.trustedTreesOf`). The log must release the sample, then the link, then the
  other rounds: no coin of the parameters' challenger family after the registration's digest released after a round is
  opened, and the first round opened a tree's `session` stream (`Challenger.sampleLinkFirst`, `SAMPLE_LINK_FIRST`). Each
  session's coins must be one tree's, a tree no other session's, its record's `coin_tree`, link points and every
  round's coins as the challenger logged them, its link's `root_b` naming exactly the session's tables
  (`sessionTables`), the roots the link points are opened after, each entry the length its table's statement fixes,
  `2^cap₀` digests (`rootBTables`, `Challenger.LINK_ROOT_LEN`), refused before the link's message is hashed, which is
  what makes that message parse one way, and every tree logged must be a session's (`Challenger.sessionsTreeReason`);
  else the check refuses, before any session is verified. A session whose statement sets up no table has no length to
  hold, and its own verdict rejects;
* **population** (`popHeld`): the registration's population held to the verifier's program: under its query, the
  program's proof blocks read stratum by stratum must be the record's `population` units in order (`Slots.popOk`), as
  the service's run refuses at registration (`Run.run`, `population-order`), else the check refuses, before any session
  is verified. v1's R2 holds only their count, and the session's knowledge soundness reads slot `u` as proof block `u`;
* **public** (`publicHeld`): the public file's header held to the registration (`Record.matchesPublic`): its scheme,
  roots and their domains and its instance count the record's, and its `units.indices` the population's units in
  order (`indicesHeld`, `indicesHeld_of_publicHeld`), so no header written after the sample relabels an instance as
  another unit; else the check refuses, before any session is verified;
* **check**: `flock-verify verify --zk --statement verity/flock-circuit` with `--circuit`, `--public`, `--partition`,
  `--program` and `--registered` the verifier's own and one `--session` a session, as `one_stage.proofs.verify` runs it
  (`verdicts`, the Flock library calls `verifyCmd` makes on that path, its last the files' ports, domains and roots,
  `Flock.PortRoots.checkOf`, and the form `Flock.Guarantees.SampledKSLiveV3` covers, or at a shared-row public file
  `SampledKSLiveV5`'s, `Flock.KsScope.checkOf`), and a session's verdict holds only where the registration's population
  is the circuit file's units in order (`FilePart.check`: its strata over the verifier's program and query, read stratum
  by stratum, are its units in order, and the public file holds exactly those, so slot `u` is the file's unit `u`, as
  `Flock.Bridge.partOf` reads it); then v1's `check` on those verdicts, one of
  record a session (`flock-verify/<name>`). A session accepted under another program than the partition object's, proved
  units other than the evidence's `proved_units`, or a circuit, public or record file no blob holds, rejects; a rep
  file no blob holds is that session's reject, as a file `flock-verify` can't read is. So does a length a session's
  statement can't hold, since it writes each in an 8-byte word: the circuit or public file, the lookup tables, the
  program, the partition object, the registered reads or a session's sample at `2^64` bytes or more (`fitsWhy`,
  `drawFitsWhy`, the JSON ones as `Flock.RecordText.printT` prints them);
* **per member** (`checkMembersAt`): the form for a population of several members, a statement a sampled member. It
  accepts where every statement's session accepted and v1's check at the coins accepts on their verdicts, as the one
  statement's check does (`checkMembersAt_accept`). v1's check reads the members there
  (`membersParamsOfJson_linked`), so its registration check refuses a registration under which a sample that ends
  misses a member, or a member's sampled proof blocks give two counts of link points (`link-points`). `protocol`
  reads either form (`paramsEitherOfJson`, `evidenceEitherOf`): `proofs` of the one statement's keys is read by
  `paramsOfJson` and `evidenceOf`, whose
  parameters name no members (`paramsOfJson_members`) and whose evidence names no statements
  (`evidenceOf_statements`), and `proofs` of the form's keys by the form's own parsers (`membersParamsOfJson`,
  `membersEvidenceOf`). The check at the coins (`checkBlobsAt`, which `live` runs) is the one statement's
  (`checkOneAt`) on parameters that name no members with evidence that names no statements, and `checkMembersAt` on
  parameters that name members with evidence that names statements (`checkBlobsAt_accept`); parameters of one form
  with evidence of the other reject. Its parameters' `proofs` is exactly
  `{"members"}` (`membersOf`), an object from each member's name to exactly
  `{"circuit", "public", "registered", "units", "vus_per_block"}`: its
  statement's files, as the one statement's, its units as `[lo, hi)` ranges, ascending, no unit in two members', and
  its circuit's units a proof block. Its evidence's `proofs` is exactly `{"statements"}` (`statementsOf`), a list of
  exactly `{"member", "sessions"}`, the statements' sessions in order being v1's `sessions`. The statements are held as
  public structure before any session is read (`membersHeld`): each names a member of the parameters that no earlier
  statement named, in the members' order (by first unit), with exactly one session, on live coins and never retried;
  a member the sample missed has no statement: a statement of a member the sample holds no unit of refuses
  (`sampledHeld`, at the evidence's `proved_units`), so the statements are exactly the members the sample holds, one
  session each (`membersRun_sampled`). Each session's coins are one tree's, as the one statement's are, and its tree's
  last opening is its closing round (`Challenger.memberTreeReason`): one more `session` round, of no coins, opened after
  the SHA-512 of the session's evidence (`sessionEvidence`: its record's bytes, then each rep file's name and bytes,
  each under an 8-byte length, `evidenceBytes`, which `evidenceParts` decodes), so the files the verdict reads are
  fixed before the next member's coins; and the log holds the trees one after another in the sessions' order, no line
  of a tree, the one that samples it or one opening a round of it, before the closing round of the one before it or
  after its own (`Challenger.membersOrder`). Then every
  member's files are read, sampled or not (`membersFiles`): its statement must read as a sampled member's does
  (`Member.statement`), and its files pass what a session's verdict checks of the files alone (`StepAt.filesHeld`: the
  scope, the files' ports and roots, the form the sampled audit's knowledge soundness covers without a session's
  sample, `Flock.KsScope.filesOf`, and the population check), so a member the sample missed is refused where its
  session's verdict would have refused its files (`membersRun_files`). A member's circuit file is parsed once for
  them (`StepAt.ofOwn`), and the statement keeps each check's result, never the parse. The compiled pass
  (`membersFilesKept`, equal to it) reads the blobs' lookup tables once and a circuit file once for the members that
  name it, and keeps of the file the circuit's own verdicts and the circuit without its rows (`Spot.CircuitKept`),
  so its work on the rows is by the file and not by the member; a member's public file it loads once for the three
  checks that read it, and a program it parses and hashes once for the members in a row that read it. Then each
  statement's session is verified as the one statement's is, on its member's own files as that pass read them
  (`runMembers`), after the statements before it, so no coin tree serves two statements: its circuit's units a proof
  block must be the ones the parameters name, and its verdict holds only where its strata are the registration's, each
  restricted to its units (`Member.sizes`: the parameters fix each stratum's size in a member, and nothing the prover
  sends moves a unit between strata), and its public file's units are exactly its member's, in order
  (`Member.population`), so every unit its session proves is its member's; the units the statements proved must be the
  evidence's `proved_units` (`membersRun_proved`). The registration is held to the members before any statement's
  session: the members' units are its population, each member one range of it (`laidHeld`), and each of its roots is
  exactly one member's, the member whose name, then `/`, the root's port begins with (`rootsOf`; a root that is no
  member's or two members' refuses, and so does a window's row tables' root, `ROW_TABLES`). A member's public file is
  held to its slice of the
  registration (`Member.slice`, `headerHeld`): the window's scheme, the member's roots and their domains under their
  bare ports, and the member's count of units as its instances. A member may read registered values, with its own
  registered file as the one statement's, and a value is one tree for every member (`classHeld`): two members' ports of
  one value that differ in its root, leaves, row width, domain or leaf schema refuse, whether or not either is sampled.

Without blobs (`toProtocol`) every check rejects: no blob holds the circuit file.
-/
module

public import CoreProtocol.Blobs
public import Spot.OneStage
public import Spot.ChallengerTrees
public import Spot.Slots
public import Spot.FilePart
public import Spot.CircuitKept
public import Spot.OneStageZkStatement
public import Flock
public import Flock.JsonParse
public import Std.Data.HashMap
meta import CoreProtocol.Blobs
meta import Spot.OneStage
meta import Spot.ChallengerTrees
meta import Spot.Slots
meta import Spot.FilePart
meta import Spot.OneStageZkStatement
meta import Flock
meta import Flock.JsonParse
meta import Std.Data.HashMap

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Spot.OneStageZk

open Lean (Json)
open Flock (V)

/-! ## The run over the members, the evidence's parts, the checks and the commands

The parameters, one statement's check, a member's statement and the files' pass are in `Spot/OneStageZkStatement.lean`,
in this namespace. -/

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

/-- A statement `statementOf` reads is `StepAt.ofOwn`'s, at the blobs' circuit and public files. -/
theorem statementOf_ofOwn {p : Own} {population : ByteArray → V Unit} {bs : Core.Blobs} {a : StepAt}
    (h : statementOf p population bs = .ok a) :
    ∃ tags0 circuitB pubB rd, a = StepAt.ofOwn p tags0 circuitB pubB bs rd (population pubB) := by
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
  obtain ⟨⟨⟩, hks, hp⟩ := Flock.HmNets.bind_ok hf
  exact ⟨a', hs, hsc, hk, (Member.statement_parse hs).1.symm.trans hks, hp⟩

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
`root_B` lengths its statement fixes on its member's own files (`rootBTables`) and the SHA-512 of its evidence
(`sessionEvidence`), then `Challenger.membersTreeReason`: each session's coins one tree's, the tree closed after that
digest, and the trees' lines, each tree's own and its rounds', one tree after another in the sessions' order. -/
def membersTreesHeld (ts : Array Challenger.LoggedTree) (log : List String) (sts : Array (StatementFiles × Member))
    (bs : Core.Blobs) : V Unit := do
  let mut named : Array (String × Array (String × Option Nat) × Json × String) := #[]
  for (st, m) in sts do
    for s in st.sessions do
      let some record := String.fromUTF8? (← bs.get s.record s!"session {s.name}'s record")
        | throw s!"session {s.name}'s record is not UTF-8"
      let evidence ← sessionEvidence bs s
      named := named.push (s.name, rootBTables m.own bs record,
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

/-- **A registered value is one tree for every member**: over the parameters' members, sampled or not, every port of
one value names one root, one count of leaves, one row width, one domain and one leaf schema (`classStep`), so the
values the members read are a function of the parameters, each with one root. It is one pass over the parameters'
members, each member's registered file read once, and it reads no blob. -/
def classHeld (ms : Array Member) : V Unit :=
  ms.foldlM classStep {} >>= fun _ => pure ()

/-- **The per-member form's structure**, held before its run on the parameters `p`, its members `ms`, the
registration `c`, the evidence `e` and its statements `sts`: the statements' structure (`membersHeld`), their sessions
v1's in order, the sessions' coins held to the trees the challenger logged in the order it must release them, each
tree closed after its session's evidence before the next is opened (`treesOf`, `orderHeld`, `membersTreesHeld`), the
registration's population held to the program (`popHeld`) and the members laid over it (`laidHeld`), and each
registered value the members read one tree (`classHeld`). -/
def membersStructure (p : Params) (ms : Array Member) (c : OneStage.Commitment) (e : Evidence)
    (sts : Array StatementFiles) (bs : Core.Blobs) : V Unit := do
  let held ← membersHeld ms sts
  if (sts.flatMap (·.sessions)).map (·.name) != e.base.sessions then
    throw "the statements' sessions are not the evidence's sessions, in order"
  let ts ← treesOf p.base e.base
  orderHeld p c e
  membersTreesHeld ts (← linesOf e) (sts.zip held) bs
  popHeld p.base.verifier c.record
  laidHeld ms c.record
  classHeld ms

/-- **The per-member form's sessions' verdicts** on the parameters `p`, its members `ms`, the registration `c`, the
evidence `e` and its statements `sts`: the form's structure (`membersStructure`), the registration's roots each one
member's (`rootsOf`), then every member's files and each statement's session on the blobs, at its member's slice of
the registration (`Member.slice`, `runMembers`). -/
def membersRun (p : Params) (ms : Array Member) (c : OneStage.Commitment) (e : Evidence) (sts : Array StatementFiles)
    (bs : Core.Blobs) : V Run :=
  membersStructure p ms c e sts bs >>= fun _ =>
  rootsOf ms c.record >>= fun roots =>
  runMembers (fun m => m.slice c.record roots) ms sts e.base.provedUnits bs

/-- **Where the per-member form's run returns, every member's files are held, sampled or not**: at the roots
`rootsOf` gives each member, each member of the parameters has its statement read and its files pass what a session's
verdict checks of them (`Member.FilesHeld`), at its slice of the registration, whatever the evidence's statements
name and whatever the sample holds. -/
theorem membersRun_files {p : Params} {ms : Array Member} {c : OneStage.Commitment} {e : Evidence}
    {sts : Array StatementFiles} {bs : Core.Blobs} {r : Run} (h : membersRun p ms c e sts bs = .ok r) :
    ∃ roots, rootsOf ms c.record = .ok roots ∧ ∀ m ∈ ms, m.FilesHeld (m.slice c.record roots) bs := by
  unfold membersRun at h
  obtain ⟨_, _, h⟩ := Flock.HmNets.bind_ok h
  obtain ⟨roots, hr, h⟩ := Flock.HmNets.bind_ok h
  exact ⟨roots, hr, runMembers_files h⟩

/-- **Where the per-member form's run returns with every session accepted, the units the sessions proved are the
evidence's `proved_units`** (`runMembers_proved`), which the record's checks hold to the sample (`Record.checks`'
`proved`): a sampled proof block whose member has no accepted session of this run refuses it. -/
theorem membersRun_proved {p : Params} {ms : Array Member} {c : OneStage.Commitment} {e : Evidence}
    {sts : Array StatementFiles} {bs : Core.Blobs} {r : Run} (h : membersRun p ms c e sts bs = .ok r)
    (hall : r.all = true) : r.proved.qsort (· < ·) = e.base.provedUnits := by
  unfold membersRun at h
  obtain ⟨_, _, h⟩ := Flock.HmNets.bind_ok h
  obtain ⟨_, _, h⟩ := Flock.HmNets.bind_ok h
  exact runMembers_proved h hall

/-- **Where the per-member form's run returns, every statement is of a member the sample holds**
(`runMembers_sampled`): the statements' members in order each have a unit among the evidence's `proved_units`, so no
session of the run is of a member the sample missed. -/
theorem membersRun_sampled {p : Params} {ms : Array Member} {c : OneStage.Commitment} {e : Evidence}
    {sts : Array StatementFiles} {bs : Core.Blobs} {r : Run} (h : membersRun p ms c e sts bs = .ok r) :
    ∃ held, membersHeld ms sts = .ok held ∧ ∀ m ∈ held, m.sampled e.base.provedUnits = true := by
  unfold membersRun at h
  obtain ⟨_, _, h⟩ := Flock.HmNets.bind_ok h
  obtain ⟨_, _, h⟩ := Flock.HmNets.bind_ok h
  exact runMembers_sampled h

/-- **Where the per-member form's run returns, its order refusal and its population refusal passed** (`orderHeld`,
`popHeld`): the two the one statement's check makes too, with the same arguments. -/
theorem membersRun_held {p : Params} {ms : Array Member} {c : OneStage.Commitment} {e : Evidence}
    {sts : Array StatementFiles} {bs : Core.Blobs} {r : Run} (h : membersRun p ms c e sts bs = .ok r) :
    orderHeld p c e = .ok () ∧ popHeld p.base.verifier c.record = .ok () := by
  unfold membersRun at h
  obtain ⟨⟨⟩, h, -⟩ := Flock.HmNets.bind_ok h
  unfold membersStructure at h
  obtain ⟨held, -, h⟩ := Flock.HmNets.bind_ok h
  dsimp only at h
  split at h
  · obtain ⟨_, ht, -⟩ := Flock.HmNets.bind_ok h
    cases ht
  obtain ⟨ts, -, h⟩ := Flock.HmNets.bind_ok h
  obtain ⟨⟨⟩, hord, h⟩ := Flock.HmNets.bind_ok h
  obtain ⟨lines, -, h⟩ := Flock.HmNets.bind_ok h
  obtain ⟨⟨⟩, -, h⟩ := Flock.HmNets.bind_ok h
  obtain ⟨⟨⟩, hpop, -⟩ := Flock.HmNets.bind_ok h
  exact ⟨hord, hpop⟩

/-- **The per-member form's check at the coins `ξ`** (module docstring, **per member**): the statements' sessions'
verdicts (`membersRun`), then v1's `checkAt` at `ξ` on them, as the one statement's check: v1 accepts only verdicts
that all accepted, and the run's own `all` says it again, so an acceptance reads the run. -/
def checkMembersAt (p : Params) (ms : Array Member) (c : OneStage.Commitment) (ξ : ByteArray) (e : Evidence)
    (sts : Array StatementFiles) (bs : Core.Blobs) : Core.Outcome OneStage.Public Core.Integrity :=
  match membersRun p ms c e sts bs with
  | .error err => .reject s!"proofs: {err}"
  | .ok r =>
    match OneStage.checkAt p.base c (some ξ) { e.base with verdicts := r.out } with
    | .accept pub prof => if r.all then .accept pub prof else .reject "proofs: a session's verifier did not accept"
    | .reject why => .reject why
    | .undecided why => .undecided why

/-- **What the per-member form's check accepts**: its run returned (`membersRun`) with every session accepted, and
v1's check at the coins accepts on the run's verdicts, with the check's public output and profile. -/
theorem checkMembersAt_accept {p : Params} {ms : Array Member} {c : OneStage.Commitment} {ξ : ByteArray}
    {e : Evidence} {sts : Array StatementFiles} {bs : Core.Blobs} {pub : OneStage.Public} {prof : Core.Integrity}
    (h : checkMembersAt p ms c ξ e sts bs = .accept pub prof) :
    ∃ r, membersRun p ms c e sts bs = .ok r ∧ r.all = true ∧
      OneStage.checkAt p.base c (some ξ) { e.base with verdicts := r.out } = .accept pub prof := by
  unfold checkMembersAt at h
  split at h
  · cases h
  · rename_i r hr
    split at h
    · rename_i pub' prof' hc
      split at h
      · rename_i hall
        cases h
        exact ⟨r, hr, hall, hc⟩
      · cases h
    · cases h
    · cases h

/-- The refusal of parameters and evidence of two forms (`checkBlobsAt`). -/
def FORMS : String :=
  "proofs: the parameters' proofs and the evidence's are not of one form: one statement's files and its sessions, " ++
    "or the members and their statements"

/-- **The check at the coins `ξ`**, by the form of the parameters' `proofs` and the evidence's: the one statement's
(`checkOneAt`), or the per-member form's (`checkMembersAt`); parameters of one form with evidence of the other
reject (`FORMS`). -/
def checkBlobsAt (p : Params) (c : OneStage.Commitment) (ξ : ByteArray) (e : Evidence) (bs : Core.Blobs) :
    Core.Outcome OneStage.Public Core.Integrity :=
  match p.members, e.statements with
  | none, none => checkOneAt p c ξ e bs
  | some ms, some sts => checkMembersAt p ms c ξ e sts bs
  | _, _ => .reject FORMS

/-- **What `checkBlobsAt` accepts is one form's accept**: the one statement's check accepts, on parameters that name
no members and evidence that names no statements; or the per-member form's check accepts, on parameters that name
members and evidence that names statements. -/
theorem checkBlobsAt_accept {p : Params} {c : OneStage.Commitment} {ξ : ByteArray} {e : Evidence} {bs : Core.Blobs}
    {pub : OneStage.Public} {prof : Core.Integrity} (h : checkBlobsAt p c ξ e bs = .accept pub prof) :
    (p.members = none ∧ e.statements = none ∧ checkOneAt p c ξ e bs = .accept pub prof) ∨
      ∃ ms sts, p.members = some ms ∧ e.statements = some sts ∧
        checkMembersAt p ms c ξ e sts bs = .accept pub prof := by
  unfold checkBlobsAt at h
  split at h
  · rename_i hm hs
    exact .inl ⟨hm, hs, h⟩
  · rename_i ms sts hm hs
    exact .inr ⟨ms, sts, hm, hs, h⟩
  · cases h

/-- **On parameters that name no members, what `checkBlobsAt` accepts the one statement's check accepts**, and the
evidence names no statements (`checkBlobsAt_accept`). -/
theorem checkBlobsAt_accept_one {p : Params} {c : OneStage.Commitment} {ξ : ByteArray} {e : Evidence}
    {bs : Core.Blobs} {pub : OneStage.Public} {prof : Core.Integrity} (hm : p.members = none)
    (h : checkBlobsAt p c ξ e bs = .accept pub prof) :
    checkOneAt p c ξ e bs = .accept pub prof ∧ e.statements = none := by
  rcases checkBlobsAt_accept h with ⟨-, hs, h1⟩ | ⟨ms, -, hms, -, -⟩
  · exact ⟨h1, hs⟩
  · cases hm.symm.trans hms

/-- **On parameters that name the members `ms`, what `checkBlobsAt` accepts the per-member form's check accepts**, at
the statements the evidence names (`checkBlobsAt_accept`). -/
theorem checkBlobsAt_accept_members {p : Params} {c : OneStage.Commitment} {ξ : ByteArray} {e : Evidence}
    {bs : Core.Blobs} {pub : OneStage.Public} {prof : Core.Integrity} {ms : Array Member} (hm : p.members = some ms)
    (h : checkBlobsAt p c ξ e bs = .accept pub prof) :
    ∃ sts, e.statements = some sts ∧ checkMembersAt p ms c ξ e sts bs = .accept pub prof := by
  rcases checkBlobsAt_accept h with ⟨hn, -, -⟩ | ⟨ms', sts, hms, hs, h1⟩
  · cases hm.symm.trans hn
  · obtain rfl : ms = ms' := Option.some.inj (hm.symm.trans hms)
    exact ⟨sts, hs, h1⟩

/-- On parameters that name no members and evidence that names no statements, the check is the one statement's. -/
theorem checkBlobsAt_one {p : Params} {c : OneStage.Commitment} {ξ : ByteArray} {e : Evidence} {bs : Core.Blobs}
    (hm : p.members = none) (hs : e.statements = none) : checkBlobsAt p c ξ e bs = checkOneAt p c ξ e bs := by
  unfold checkBlobsAt
  simp only [hm, hs]

/-- The audit's check on the blobs: at the coins `coinsOf` reads (`checkBlobsAt`), refused with its reason when none
read. -/
def checkBlobs (p : Params) (c : OneStage.Commitment) (e : Evidence) (bs : Core.Blobs) :
    Core.Outcome OneStage.Public Core.Integrity :=
  Core.atCoins (coinsOf p c e) (fun ξ => checkBlobsAt p c ξ e bs)

def protocol : Core.BlobProtocol where
  id := PROTOCOL
  Params := Params
  Commitment := OneStage.Commitment
  Evidence := Evidence
  Public := OneStage.Public
  Profile := Core.Integrity
  params := paramsEitherOfJson
  commitment := OneStage.commitmentOf
  evidence := evidenceEitherOf
  digest c := c.digest
  check p c e := checkBlobs p c e Core.Blobs.empty
  publicJson := OneStage.publicJson
  profileJson := Core.Integrity.toJson
  checkBlobs := checkBlobs

/-- The audit on the blobs `bs` reads its sample's coins (`coinsOf`) and decides at them (`checkBlobsAt`). -/
def live (bs : Array ByteArray) : Core.Live (protocol.on bs) where
  Coins := ByteArray
  coins := coinsOf
  checkAt p c ξ e := checkBlobsAt p c ξ e (Core.Blobs.ofArray bs)
  factors _ _ _ := rfl

/-- **What `paramsOfJson` returns, `paramsParsed` returned, under a `subset` sampling design**: its one refusal of its
own is `SUBSET_ONLY`'s. -/
theorem paramsOfJson_ok {j : Json} {p : Params} (h : paramsOfJson j = .ok p) :
    paramsParsed j = .ok p ∧ getD p.base.verifier.law "law" = .str "subset" := by
  unfold paramsOfJson at h
  cases hp : paramsParsed j with
  | error why => simp [hp, bind, Except.bind] at h
  | ok q =>
    simp only [hp, bind, Except.bind] at h
    cases hl : getD q.base.verifier.law "law" with
    | str s =>
      simp only [hl] at h
      by_cases hs : s = "subset"
      · subst hs
        simp [pure, Except.pure] at h
        subst h
        exact ⟨rfl, hl⟩
      · simp [hs, throw, throwThe, MonadExceptOf.throw] at h
    | _ => simp [hl, throw, throwThe, MonadExceptOf.throw] at h

/-- **What `paramsOfJson` returns, `paramsParsed` returned** (`paramsOfJson_ok`). -/
theorem paramsOfJson_parsed {j : Json} {p : Params} (h : paramsOfJson j = .ok p) : paramsParsed j = .ok p :=
  (paramsOfJson_ok h).1

/-- **Parsed parameters' sampling design is a `subset` one**: whatever `paramsOfJson` returns names a sampling design of
kind `subset`, the kind the sample reads (`Draw.deriveLaw`, `Draw.flockLaw`). -/
theorem paramsOfJson_subset {j : Json} {p : Params} (h : paramsOfJson j = .ok p) :
    getD p.base.verifier.law "law" = .str "subset" :=
  (paramsOfJson_ok h).2

/-- **Parsed parameters name no members**: whatever `paramsOfJson` returns is of the one statement's form. -/
theorem paramsOfJson_members {j : Json} {p : Params} (h : paramsOfJson j = .ok p) : p.members = none := by
  have h := paramsOfJson_parsed h
  unfold paramsParsed at h
  simp only [bind, Except.bind] at h
  repeat' split at h
  all_goals try contradiction
  all_goals try (rename_i ht _ _ _ _; simp [throw, throwThe, MonadExceptOf.throw] at ht; done)
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  rfl

/-- **Parsed evidence names no statements**: whatever `evidenceOf` returns is of the one statement's form. -/
theorem evidenceOf_statements {b : ByteArray} {e : Evidence} (h : evidenceOf b = .ok e) : e.statements = none := by
  unfold evidenceOf at h
  simp only [bind, Except.bind] at h
  repeat' split at h
  all_goals try contradiction
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  rfl

/-- **The per-member form's parameters name members and hand them to the registration's check**: whatever
`membersParamsOfJson` returns names some, and the members its v1 part's verifier holds
(`Registration.Verifier.members`, which `link-points` reads) are those, each its units and its circuit's units a proof
block. -/
theorem membersParamsOfJson_linked {j : Json} {p : Params} (h : membersParamsOfJson j = .ok p) :
    ∃ ms : Array Member, p.members = some ms ∧ p.base.verifier.members = ms.map fun m => (m.units, m.g) := by
  unfold membersParamsOfJson at h
  split at h
  · obtain ⟨⟨⟩, -, h⟩ := Flock.HmNets.bind_ok h
    try dsimp only at h
    obtain ⟨p0, -, h⟩ := Flock.HmNets.bind_ok h
    try dsimp only at h
    obtain ⟨ms, -, h⟩ := Flock.HmNets.bind_ok h
    cases h
    exact ⟨ms, rfl, rfl⟩
  · cases h

/-- **The per-member form's parameters name members**: whatever `membersParamsOfJson` returns names some. -/
theorem membersParamsOfJson_members {j : Json} {p : Params} (h : membersParamsOfJson j = .ok p) :
    ∃ ms, p.members = some ms :=
  let ⟨ms, hm, _⟩ := membersParamsOfJson_linked h
  ⟨ms, hm⟩

/-- **Parameters of either form are parsed parameters but for the files and the members**: their v1 part is that of
parameters `paramsParsed` returned, the one statement's own (`paramsOfJson_parsed`) or the per-member form's, read at
`proofs` of no file, but for the members its registration's check reads (`Registration.Verifier.members`), which the
per-member form's parse hands it (`membersParamsOfJson_linked`). -/
theorem paramsEitherOfJson_base {j : Json} {p : Params} (h : paramsEitherOfJson j = .ok p) :
    ∃ j' p' ms, paramsParsed j' = .ok p' ∧
      p.base = { p'.base with verifier := { p'.base.verifier with members := ms } } := by
  unfold paramsEitherOfJson at h
  split at h
  · unfold membersParamsOfJson at h
    split at h
    · obtain ⟨⟨⟩, -, h⟩ := Flock.HmNets.bind_ok h
      try dsimp only at h
      obtain ⟨p0, hp0, h⟩ := Flock.HmNets.bind_ok h
      try dsimp only at h
      obtain ⟨ms, -, h⟩ := Flock.HmNets.bind_ok h
      cases h
      exact ⟨_, p0, _, hp0, rfl⟩
    · cases h
  · exact ⟨j, p, p.base.verifier.members, paramsOfJson_parsed h, rfl⟩

/-- **Parameters of either form that name no members are the one statement's**: `paramsOfJson` returned them. -/
theorem paramsEitherOfJson_one {j : Json} {p : Params} (h : paramsEitherOfJson j = .ok p) (hm : p.members = none) :
    paramsOfJson j = .ok p := by
  unfold paramsEitherOfJson at h
  split at h
  · obtain ⟨ms, hms⟩ := membersParamsOfJson_members h
    cases hm.symm.trans hms
  · exact h

/-- **The per-member form's evidence names its statements, and its sessions are theirs in order**: whatever
`membersEvidenceOf` returns names statements whose sessions, one statement's after another's, are its sessions. -/
theorem membersEvidenceOf_sessions {b : ByteArray} {e : Evidence} (h : membersEvidenceOf b = .ok e) :
    ∃ sts, e.statements = some sts ∧ e.sessions = sts.flatMap (·.sessions) := by
  unfold membersEvidenceOf at h
  obtain ⟨j, -, h⟩ := Flock.HmNets.bind_ok h
  try dsimp only at h
  split at h
  · obtain ⟨sts, -, h⟩ := Flock.HmNets.bind_ok h
    try dsimp only at h
    obtain ⟨one, -, h⟩ := Flock.HmNets.bind_ok h
    try dsimp only at h
    obtain ⟨e0, -, h⟩ := Flock.HmNets.bind_ok h
    cases h
    exact ⟨sts, rfl, rfl⟩
  · cases h

/-- **Evidence of either form that names the statements `sts` has their sessions, in order**: the per-member form's
parser returned it (`membersEvidenceOf_sessions`), since the one statement's names none (`evidenceOf_statements`). -/
theorem evidenceEitherOf_sessions {b : ByteArray} {e : Evidence} {sts : Array StatementFiles}
    (h : evidenceEitherOf b = .ok e) (hs : e.statements = some sts) : e.sessions = sts.flatMap (·.sessions) := by
  unfold evidenceEitherOf at h
  split at h
  · split at h
    · obtain ⟨sts', hs', hse⟩ := membersEvidenceOf_sessions h
      obtain rfl : sts = sts' := Option.some.inj (hs.symm.trans hs')
      exact hse
    · cases hs.symm.trans (evidenceOf_statements h)
  · cases h

/-- **Evidence of either form has one statement's evidence's v1 part**: the one statement's own, or the per-member
form's, which `evidenceOf` read at `proofs` of the statements' sessions. -/
theorem evidenceEitherOf_base {b : ByteArray} {e : Evidence} (h : evidenceEitherOf b = .ok e) :
    ∃ b' e', evidenceOf b' = .ok e' ∧ e.base = e'.base := by
  unfold evidenceEitherOf at h
  split at h
  · split at h
    · unfold membersEvidenceOf at h
      obtain ⟨j, -, h⟩ := Flock.HmNets.bind_ok h
      try dsimp only at h
      split at h
      · obtain ⟨sts, -, h⟩ := Flock.HmNets.bind_ok h
        try dsimp only at h
        obtain ⟨one, -, h⟩ := Flock.HmNets.bind_ok h
        try dsimp only at h
        obtain ⟨e0, he0, h⟩ := Flock.HmNets.bind_ok h
        cases h
        exact ⟨_, e0, he0, rfl⟩
      · cases h
    · exact ⟨b, e, h, rfl⟩
  · cases h

/-- **Evidence of either form that names no statements is the one statement's**: `evidenceOf` returned it. -/
theorem evidenceEitherOf_one {b : ByteArray} {e : Evidence} (h : evidenceEitherOf b = .ok e)
    (hs : e.statements = none) : evidenceOf b = .ok e := by
  unfold evidenceEitherOf at h
  split at h
  · split at h
    · obtain ⟨sts, hs', -⟩ := membersEvidenceOf_sessions h
      cases hs.symm.trans hs'
    · exact h
  · cases h

/-- `protocol.main`'s commands, and `verdicts OWN PROOFS [BLOB ...]`: the sessions' verdicts alone (`verdicts`, with no
proved units to bind), from the verifier's own inputs (`ownOfJson`) and the evidence's `proofs` (`proofsOf`), each file
canonical JSON; one line, `{"verdicts": [verdict, ...]}` in session order or `{"refused": why}`, exit 0 or 1. The
difftest holds it to `flock-verify verify --zk` (`one_stage.proofs.verify`). -/
def main (args : List String) : IO UInt32 := do
  match args with
  | "verdicts" :: own :: proofs :: blobs =>
    let bs ← blobs.toArray.mapM fun (f : String) => IO.FS.readBinFile f
    let ownB ← IO.FS.readBinFile own
    let proofsB ← IO.FS.readBinFile proofs
    let r : V (Array (String × Json)) := do
      let sessions ← proofsOf (← Core.canonicalJson proofsB)
      verdicts (← ownOfJson (← Core.canonicalJson ownB)) sessions none (Core.Blobs.ofArray bs)
    let line := match r with
      | .ok vs => Json.mkObj [("verdicts", Json.arr (vs.map (·.2)))]
      | .error e => Json.mkObj [("refused", Json.str e)]
    IO.println ((Flock.canon line).toOption.getD line.compress)
    return if r.toBool then 0 else 1
  | _ => protocol.main args

#guard registeredInputName "RegisteredInput96_v1"
#guard registeredInputName "RegisteredInput1_v1"
#guard !registeredInputName "RegisteredInput0_v1"
#guard !registeredInputName "RegisteredInput01_v1"
#guard !registeredInputName "RegisteredInput_v1"
#guard !registeredInputName "RegisteredInput96_v2"
#guard !registeredInputName "XRegisteredInput96_v1"
#guard readsRegistered (Json.mkObj [])
#guard !readsRegistered (Json.mkObj [("definitions", Json.mkObj [("Rope_v1", Json.null)])])
#guard readsRegistered (Json.mkObj [("definitions", Json.mkObj [("RegisteredInput96_v1", Json.null)])])
#guard (provedUnits "{\"units\":{\"indices\":[7,3,9]}}\nrest".toUTF8 "{\"unit_draw\":{\"units\":[2,0]}}").toOption
  == some #[7, 9]
#guard (provedUnits "{\"units\":{\"indices\":[7]}}".toUTF8 "{\"unit_draw\":{\"units\":[1]}}").toOption.isNone
#guard (provedHeld (#[9, 3] ++ #[7]) #[3, 7, 9]).toBool
#guard !(provedHeld (#[3, 7] ++ #[7, 9]) #[3, 7, 9]).toBool
#guard !(provedHeld (#[3] ++ #[9]) #[3, 7, 9]).toBool
#guard !((provedUnits "{\"units\":{\"indices\":[7,3,9]}}".toUTF8 "{\"unit_draw\":{\"units\":[2,2]}}") >>=
  (provedHeld · #[9])).toBool
#guard (oneSession #[⟨"s", "", #[]⟩]).toBool
#guard !(oneSession #[]).toBool
#guard !(oneSession #[⟨"s", "", #[]⟩, ⟨"t", "", #[]⟩]).toBool

private def heldRecord : Json :=
  (Flock.JsonParse.parse
    "{\"population\":2,\"roots\":[{\"domain\":\"dd\",\"port\":\"x\",\"root\":\"aa\"}],\"scheme\":\"s\"}").toOption.getD
    .null

private def refusal : V Unit → String
  | .error e => e
  | .ok () => ""

#guard refusal (publicHeld heldRecord
  ("{\"frame_v3\":{\"domain_ids\":{\"x\":\"dd\"},\"roots\":{\"x\":\"aa\"}},\"instances\":2,\"scheme\":\"s\"," ++
    "\"units\":{\"indices\":[0,1]}}\nrest").toUTF8) == ""
#guard [("[1,0]" : String), "[0,2]", "[0]", "[0,1,2]", "[\"0\",1]"].all fun is => refusal (publicHeld heldRecord
  ("{\"frame_v3\":{\"domain_ids\":{\"x\":\"dd\"},\"roots\":{\"x\":\"aa\"}},\"instances\":2,\"scheme\":\"s\"," ++
    "\"units\":{\"indices\":" ++ is ++ "}}").toUTF8) == INDICES_WHY
#guard refusal (publicHeld heldRecord
  "{\"frame_v3\":{\"domain_ids\":{\"x\":\"dd\"},\"roots\":{\"x\":\"aa\"}},\"instances\":2,\"scheme\":\"s\"}".toUTF8) ==
  INDICES_WHY
#guard refusal (publicHeld heldRecord
  "{\"frame_v3\":{\"domain_ids\":{\"x\":\"dd\"},\"roots\":{\"x\":\"ab\"}},\"instances\":2,\"scheme\":\"s\"}".toUTF8) ==
  "public file: not the registration's: roots"
#guard refusal (publicHeld heldRecord
  "{\"frame_v3\":{\"domain_ids\":{\"x\":\"de\"},\"roots\":{\"x\":\"aa\",\"y\":\"bb\"}},\"instances\":3,\"scheme\":\"s\"}".toUTF8) ==
  "public file: not the registration's: roots, domains, population"
#guard (refusal (publicHeld heldRecord "not a public file".toUTF8)).startsWith "the public file's header is not JSON: "

/-- A circuit file whose META looks up the MUFU tables `names`. -/
private def lookingUp (names : List String) : ByteArray :=
  s!"circuit\nMETA {(Json.mkObj [("lookups", Json.mkObj (names.map (·, Json.mkObj [])))]).compress}\nrest".toUTF8

/-- The verifier's own files naming the circuit file `c` and no public file. -/
private def ownOf (c : ByteArray) : Own :=
  { programJ := Json.mkObj [("definitions", Json.mkObj [])], program := .empty, partition := Json.null,
    circuit := Flock.hexOf (Flock.Sha512.hash c), pub := "", registered := none }

/-- Blobs holding the circuit file `c` and, under their pinned SHA-512s whatever their bytes, the MUFU tables `names`. -/
private def withTables (c : ByteArray) (names : List String) : Core.Blobs :=
  ⟨#[(Flock.Sha512.hash c, c)] ++ (Flock.MUFU_TABLES.filter fun t =>
    names.contains t.name).toArray.map fun t => ((Flock.unhex t.sha512).toOption.getD .empty, .empty)⟩

private def held (lookups tables : List String) : String :=
  refusal (tablesHeld (ownOf (lookingUp lookups)) (withTables (lookingUp lookups) tables))

#guard held [] [] == ""
#guard held ["ex2", "rcp"] ["rcp", "ex2"] == ""
#guard held [] ["rcp"] == s!"{NO_TABLES}: the circuit looks up none, the blobs hold rcp"
#guard held ["ex2"] ["ex2", "rsq", "sqrt", "rcp"] == s!"{NO_TABLES}: the circuit looks up ex2, the blobs hold rsq, sqrt, rcp, ex2"
#guard held ["ex2", "rcp"] ["ex2"] == s!"{NO_TABLES}: the circuit looks up rcp, ex2, the blobs hold ex2"
#guard (lookupsOf "circuit\nMETA {\"lookups\":[]}\n".toUTF8).toOption == none
#guard (lookupsOf "circuit\nMETA {\"lookups\":null}\n".toUTF8).toOption == some []
#guard (lookupsOf "circuit\nMETA {\"lookups\":{\"foo\":{},\"ex2\":{}}}\n".toUTF8).toOption == some ["ex2"]
#guard (refusal (tablesHeld (ownOf (lookingUp [])) ⟨#[]⟩)).startsWith "circuit: "
#guard refusal ((fun _ => ()) <$> run (ownOf (lookingUp [])) #[] none (withTables (lookingUp []) ["rcp"])) ==
  s!"{NO_TABLES}: the circuit looks up none, the blobs hold rcp"
#guard (refusal ((fun _ => ()) <$> run (ownOf (lookingUp ["rcp"])) #[] none (withTables (lookingUp ["rcp"]) ["rcp"]))).startsWith
  "public file: "
#guard (tablesPiece (Json.mkObj [("lookups", Json.mkObj [("rcp", Json.mkObj [])]), ("digests", Json.arr #[Json.str
  "8a317adbcc078bf70ae01959b679d49f97cd9c8a93d6f39b7325c93d0fd550fa4396a4c3a0568d2387c2ff2f33eb76453b33a7b8890d97752d86cc1d971aeacc",
  Json.str "00"])])).toOption == some (Json.mkObj [("why", Json.null)])
#guard (tablesPiece (Json.mkObj [("digests", Json.arr #[Json.str
  "8a317adbcc078bf70ae01959b679d49f97cd9c8a93d6f39b7325c93d0fd550fa4396a4c3a0568d2387c2ff2f33eb76453b33a7b8890d97752d86cc1d971aeacc"])])).toOption ==
  some (Json.mkObj [("why", Json.str s!"{NO_TABLES}: the circuit looks up none, the blobs hold rcp")])
#guard !(tablesPiece (Json.mkObj [("digests", Json.arr #[Json.str "0"])])).toBool

#guard (rootBLen { (default : Flock.Setup) with m := 25, merkle := Flock.MerkleScheme.hm96Sha512 }).toOption == some 16384
#guard (rootBLen { (default : Flock.Setup) with m := 26, merkle := Flock.MerkleScheme.hm96Sha512 }).toOption == some 8192

/-- The program and the partition the members' statements share, naming no file. -/
private def sharedT : Own :=
  { programJ := Json.null, program := .empty, partition := Json.null, circuit := "", pub := "", registered := none }

private def memberJ (units : List (Nat × Nat)) : Json :=
  Json.mkObj [("circuit", Json.str ("".pushn 'c' 128)), ("public", Json.str ("".pushn 'd' 128)),
    ("registered", Json.null),
    ("units", Json.arr (units.toArray.map fun (lo, hi) => Json.arr #[Lean.toJson lo, Lean.toJson hi])),
    ("vus_per_block", Lean.toJson (1 : Nat))]

#guard ((membersOf sharedT (Json.mkObj [("a", memberJ [(0, 4)]), ("b", memberJ [(4, 6), (8, 9)])])).toOption.map
  (·.map fun m => (m.name, m.units, m.g, m.own.circuit.length, m.own.pub.length))) ==
  some #[("a", #[(0, 4)], 1, 128, 128), ("b", #[(4, 6), (8, 9)], 1, 128, 128)]
#guard (membersOf sharedT (Json.mkObj [("a", memberJ [(0, 4)]), ("b", memberJ [(3, 6)])])).toOption.isNone
#guard (membersOf sharedT (Json.mkObj [("a", memberJ [(0, 4), (2, 3)])])).toOption.isNone
#guard (membersOf sharedT (Json.mkObj [("a", memberJ [(5, 6), (0, 1)])])).toOption.isNone
#guard (membersOf sharedT (Json.mkObj [("a", memberJ [])])).toOption.isNone
#guard (membersOf sharedT (Json.mkObj [("", memberJ [(0, 4)])])).toOption.isNone
#guard (membersOf sharedT (Json.mkObj [])).toOption.isNone
#guard (membersOf sharedT (Json.mkObj [("a", (memberJ [(0, 4)]).setObjVal! "vus_per_block"
  (Lean.toJson (0 : Nat)))])).toOption.isNone
#guard (membersOf sharedT (Json.mkObj [("a", (memberJ [(0, 4)]).setObjVal! "interface" Json.null)])).toOption.isNone
#guard (statementsOf (Json.mkObj [("statements", Json.arr #[Json.mkObj [("member", Json.str "a"),
  ("sessions", Json.arr #[])]])])).toOption.map (·.map (·.member)) == some #[some "a"]
#guard (statementsOf (Json.mkObj [("statements", Json.arr #[Json.mkObj [("member", Json.str ""),
  ("sessions", Json.arr #[])]])])).toOption.isNone
#guard (statementsOf (Json.mkObj [("sessions", Json.arr #[])])).toOption.isNone

private def paramsWhy (r : V Params) : String :=
  match r with
  | .error e => e
  | .ok _ => ""

private def evidenceWhy (r : V Evidence) : String :=
  match r with
  | .error e => e
  | .ok _ => ""

/-- Parameters of nothing but `proofs`: each parser refuses them, the one statement's past its `proofs`. -/
private def proofsOnly (pr : Json) : Json := Json.mkObj [("proofs", pr)]

private def oneProofs : Json :=
  let zeros := "".pushn '0' 128
  Json.mkObj [("circuit", Json.str zeros), ("public", Json.str zeros), ("registered", Json.null)]

#guard paramsWhy (paramsOfJson (proofsOnly (Json.mkObj [("members", Json.mkObj [])]))) ==
  "the keys of proofs are not circuit, public, registered"
#guard paramsWhy (membersParamsOfJson (proofsOnly oneProofs)) == "the keys of proofs are not members"
#guard paramsWhy (paramsOfJson (proofsOnly oneProofs)) != ""
#guard paramsWhy (paramsOfJson (proofsOnly oneProofs)) != "the keys of proofs are not circuit, public, registered"
#guard paramsWhy (membersParamsOfJson (proofsOnly (Json.mkObj [("members", Json.mkObj [])]))) ==
  paramsWhy (paramsOfJson (proofsOnly oneProofs))
#guard paramsWhy (paramsEitherOfJson (proofsOnly (Json.mkObj [("members", Json.mkObj [])]))) ==
  paramsWhy (paramsOfJson (proofsOnly oneProofs))
#guard paramsWhy (paramsEitherOfJson (proofsOnly oneProofs)) == paramsWhy (paramsOfJson (proofsOnly oneProofs))
#guard paramsWhy (paramsEitherOfJson (proofsOnly (Json.mkObj [("statements", Json.arr #[])]))) ==
  "the keys of proofs are not circuit, public, registered"
#guard evidenceWhy (membersEvidenceOf "{\"proofs\":{\"sessions\":[]}}".toUTF8) ==
  "the keys of the evidence's proofs are not statements"
#guard evidenceWhy (evidenceOf "{\"proofs\":{\"sessions\":[]}}".toUTF8) != ""
#guard evidenceWhy (evidenceOf "{\"proofs\":{\"statements\":[]}}".toUTF8) != ""
#guard evidenceWhy (membersEvidenceOf "{\"proofs\":{\"statements\":[]}}".toUTF8) ==
  evidenceWhy (evidenceOf "{\"proofs\":{\"sessions\":[]}}".toUTF8)
#guard evidenceWhy (evidenceEitherOf "{\"proofs\":{\"statements\":[]}}".toUTF8) ==
  evidenceWhy (evidenceOf "{\"proofs\":{\"sessions\":[]}}".toUTF8)
#guard evidenceWhy (evidenceEitherOf "{\"proofs\":{\"sessions\":[]}}".toUTF8) ==
  evidenceWhy (evidenceOf "{\"proofs\":{\"sessions\":[]}}".toUTF8)
#guard evidenceWhy (evidenceEitherOf "not JSON".toUTF8) == evidenceWhy (evidenceOf "not JSON".toUTF8)

private def memberT (name : String) (units : Array (Nat × Nat)) : Member := { name, units, g := 1, own := sharedT }

private def statementT (member : String) (sessions : Nat := 1) : StatementFiles :=
  { member := some member, sessions := (List.replicate sessions { name := "s", record := "r", reps := #[] }).toArray }

private def membersT : Array Member :=
  #[memberT "m2" #[(2, 3)], memberT "m10" #[(10, 11)], memberT "m0" #[(0, 1), (5, 6)]]

private def whyT {α : Type} : V α → String
  | .error e => e
  | .ok _ => ""

#guard (memberT "m0" #[(0, 1), (5, 6)]).first == 0
#guard (memberT "m" #[(2, 4), (7, 8)]).indices == #[2, 3, 7]
#guard ascending 0 [(0, 1), (1, 3), (7, 8)]
#guard !ascending 0 [(0, 2), (1, 3)]
#guard !ascending 0 [(2, 2)]
#guard ((membersHeld membersT #[statementT "m0", statementT "m2", statementT "m10"]).toOption.map (·.map (·.name))) ==
  some #["m0", "m2", "m10"]
#guard ((membersHeld membersT #[statementT "m2", statementT "m10"]).toOption.map (·.map (·.name))) ==
  some #["m2", "m10"]
#guard ((membersHeld membersT #[]).toOption.map (·.size)) == some 0
#guard (whyT (membersHeld membersT #[statementT "m2" 0, statementT "m10"])).startsWith
  "member m2: its statement has 0 sessions"
#guard (whyT (membersHeld membersT #[statementT "m2", statementT "m0"])).startsWith
  "member m0: its statement follows a member's whose units start after its own"
#guard (whyT (membersHeld membersT #[statementT "m0", statementT "m10", statementT "m2"])).startsWith
  "member m2: its statement follows a member's"
#guard whyT (membersHeld membersT #[statementT "m0", statementT "m0"]) ==
  "member m0: a second statement: a member's units are one statement's"
#guard (whyT (membersHeld membersT #[statementT "m0" 2])).startsWith "member m0: its statement has 2 sessions"
#guard whyT (membersHeld membersT #[statementT "m3"]) == "member m3: the parameters name no such member"
#guard (whyT (membersHeld membersT #[{ member := none, sessions := #[] }])).startsWith "a statement of the proofs names"
#guard (memberT "m0" #[(0, 1), (5, 6)]).sampled #[5]
#guard !(memberT "m0" #[(0, 1), (5, 6)]).sampled #[1, 2, 6]
#guard !(memberT "m0" #[(0, 1)]).sampled #[]
#guard (sampledHeld #[memberT "m0" #[(0, 1), (5, 6)], memberT "m2" #[(2, 3)]] #[2, 5]).toBool
#guard whyT (sampledHeld #[memberT "m0" #[(0, 1), (5, 6)], memberT "m2" #[(2, 3)]] #[0, 3]) ==
  "member m2: the sample holds none of its units, and a member the sample missed has no statement"
#guard (sampledHeld #[] #[]).toBool

private def unitsJ (us : Array Nat) : Json := Json.mkObj [("units", Json.mkObj [("indices", natsJson us)])]

#guard unitsHeld #[2, 3, 7] (unitsJ #[2, 3, 7])
#guard !unitsHeld #[2, 3, 7] (unitsJ #[2, 3])
#guard !unitsHeld #[2, 3] (unitsJ #[2, 3, 7])
#guard !unitsHeld #[2, 3] (unitsJ #[3, 2])
#guard !unitsHeld #[] Json.null
#guard unitsIn #[(2, 4), (7, 8)] 3 8 == 2
#guard unitsIn #[(2, 4), (7, 8)] 4 7 == 0
#guard (memberT "m" #[(2, 4), (7, 8)]).sizes #[("a", #[(0, 3)]), ("b", #[(3, 8)])] == some #[1, 2]
#guard (memberT "m" #[(2, 4), (7, 8)]).sizes #[("a", #[(0, 2), (2, 5)]), ("b", #[(5, 9)])] == some #[2, 1]
#guard (memberT "m" #[(2, 4), (7, 8)]).sizes #[("a", #[(0, 2)]), ("b", #[(2, 8)])] == some #[0, 3]
#guard (memberT "m" #[(2, 4), (7, 8)]).sizes #[("a", #[(0, 3)]), ("b", #[(3, 7)])] == none
#guard (memberT "m" #[(2, 4), (7, 8)]).sizes #[("a", #[(0, 1), (2, 8)]), ("b", #[(1, 2)])] == none
#guard refusal ((memberT "m" #[(2, 4), (7, 8)]).heldAt #[("a", #[(0, 3)]), ("b", #[(3, 8)])] (unitsJ #[2, 3, 7])) == ""
#guard refusal ((memberT "m" #[(2, 4), (7, 8)]).heldAt #[("a", #[(0, 3)]), ("b", #[(3, 8)])] (unitsJ #[0, 1, 2])) ==
  MEMBER_UNITS
#guard refusal ((memberT "m" #[(2, 4), (7, 8)]).heldAt #[("a", #[(0, 3)]), ("b", #[(3, 7)])] (unitsJ #[2, 3, 7])) ==
  MEMBER_STRATA
#guard refusal ((memberT "m" #[(2, 4), (7, 8)]).population Json.null
  "{\"units\":{\"indices\":[2,3,7]}}\nrest".toUTF8) != ""
#guard (runMembers (fun _ => Json.null) #[] #[] #[] ⟨#[]⟩).toOption.map (·.all) == some true
#guard whyT (runMembers (fun _ => Json.null) #[] #[] #[1] ⟨#[]⟩) ==
  "the units the sessions proved are not the evidence's proved_units"
#guard (whyT (runMembers (fun _ => Json.null) membersT #[statementT "m2"] #[2] ⟨#[]⟩)).startsWith
  "member m2: circuit: "
#guard whyT (runMembers (fun _ => Json.null) membersT #[statementT "m2"] #[] ⟨#[]⟩) == unsampled "m2"
#guard whyT (runMembers (fun _ => Json.null) membersT #[statementT "m2", statementT "m10"] #[10] ⟨#[]⟩) ==
  unsampled "m2"
#guard (whyT (runMembers (fun _ => Json.null) membersT #[] #[] ⟨#[]⟩)).startsWith "member m2: circuit: "
#guard (whyT (runMembers (fun _ => Json.null) #[memberT "m0" #[(0, 1)], memberT "m2" #[(2, 3)]] #[statementT "m2"]
  #[2] ⟨#[]⟩)).startsWith "member m0: circuit: "
#guard (whyT (membersFiles (fun _ => Json.null) ⟨#[]⟩ membersT.toList)).startsWith "member m2: circuit: "
#guard ((membersFiles (fun _ => Json.null) ⟨#[]⟩ []).toOption.map (·.length)) == some 0
#guard whyT (statementAt [] (memberT "m2" #[(2, 3)])) == "the parameters name no such member"

/-- A statement of no file whose scope, ports, form and population checks are the ones given. -/
private def stepT (scope ksFiles ksForm population : V Unit) : Option StepAt :=
  (Flock.Tags.find? STATEMENT).map fun tags =>
    { I := inputsOf tags sharedT .empty .empty ⟨#[]⟩, rd := { registered := none, known := #[], params := #[] }, scope,
      ksFiles, ksForm, g := none, population, programSha := none, want := Json.null }

#guard ((stepT (.error "scope") (.error "ports") (.error "form") (.error "population")).map fun a =>
  refusal a.filesHeld) == some "scope"
#guard ((stepT (.ok ()) (.error "ports") (.error "form") (.error "population")).map fun a => refusal a.filesHeld) ==
  some "ports"
#guard ((stepT (.ok ()) (.ok ()) (.error "form") (.error "population")).map fun a => refusal a.filesHeld) ==
  some "form"
#guard ((stepT (.ok ()) (.ok ()) (.ok ()) (.error "population")).map fun a => refusal a.filesHeld) ==
  some "population"
#guard ((stepT (.ok ()) (.ok ()) (.ok ()) (.ok ())).map fun a => a.filesHeld.toOption.isSome) == some true

/-- The statement `StepAt.ofOwn` builds of no file: its form check is `Flock.KsScope.filesOf`'s refusal of a circuit
that does not parse, and it names no units a proof block. -/
private def ownT : Option StepAt :=
  (Flock.Tags.find? STATEMENT).map fun tags =>
    StepAt.ofOwn sharedT tags .empty .empty ⟨#[]⟩ { registered := none, known := #[], params := #[] } (.ok ())

#guard (ownT.map fun a => (refusal a.ksForm).startsWith "refused: ") == some true
#guard (ownT.map fun a => a.g) == some none
#guard whyT ((memberT "m0" #[(0, 1)]).blockHeld (some 2)) ==
  "vus_per_block: the member's circuit has 2 units a proof block, not the 1 the parameters name"
#guard ((memberT "m0" #[(0, 1)]).blockHeld (some 1)).toOption.isSome && ((memberT "m0" #[(0, 1)]).blockHeld none).toOption.isSome

private def rootJ (port root : String) : Json :=
  Json.mkObj [("domain", Json.str ("d" ++ root)), ("leaves", Lean.toJson (2 : Nat)), ("port", Json.str port),
    ("root", Json.str root)]

/-- A registration of the population `n` and the roots given, each a port and a root. -/
private def recordT (n : Nat) (roots : List (String × String)) : Json :=
  Json.mkObj [("population", Lean.toJson n), ("roots", Json.arr (roots.toArray.map fun (p, r) => rootJ p r)),
    ("scheme", Json.str "s")]

/-- Two members whose names hold a `/`. -/
private def keyedT : Array Member := #[memberT "t/c0" #[(0, 2)], memberT "t/c1" #[(2, 5)]]

private def strT : Json → String
  | .str s => s
  | _ => ""

/-- The roots `rootsOf` gives the member `name` of `keyedT`, each its bare port and its root. -/
private def rootsT (roots : List (String × String)) (name : String) : Option (Array (String × String)) :=
  (rootsOf keyedT (recordT 5 roots)).toOption.map fun rs =>
    (rs.getD name #[]).map fun r => (strT (getD r "port"), strT (getD r "root"))

#guard splitsOf "t/c0/x" == [("t", "c0/x"), ("t/c0", "x")]
#guard splitsOf "x" == []
#guard splitsOf "/x/" == [("", "x/"), ("/x", "")]
#guard rootsT [("t/c0/x", "aa"), ("t/c1/y", "bb"), ("t/c0/y", "cc")] "t/c0" == some #[("x", "aa"), ("y", "cc")]
#guard rootsT [("t/c0/x", "aa"), ("t/c1/y", "bb"), ("t/c0/y", "cc")] "t/c1" == some #[("y", "bb")]
#guard rootsT [] "t/c0" == some #[]
#guard (whyT (rootsOf #[memberT "t/c0" #[(0, 2)], memberT "t/c0/u" #[(2, 5)]]
  (recordT 5 [("t/c0/u/y", "bb")]))).startsWith "roots: the registration's root t/c0/u/y is two members'"
#guard (whyT (rootsOf keyedT (recordT 5 [("t/c2/x", "aa")]))).startsWith
  "roots: the registration's root t/c2/x is no member's"
#guard (whyT (rootsOf keyedT (recordT 5 [("t/c0", "aa")]))).startsWith
  "roots: the registration's root t/c0 is no member's"
#guard whyT (rootsOf keyedT (recordT 5 [("t/c0/x", "aa"), ("row-table", "bb")])) == ROW_TABLES
#guard whyT (rootsOf keyedT (Json.mkObj [("roots", Json.arr #[Json.mkObj [("root", Json.str "aa")]])])) ==
  "roots: a root of the registration names no port"
#guard whyT (rootsOf keyedT Json.null) == "roots: the registration's roots are not a list"
#guard (memberT "m" #[(2, 4), (7, 8)]).count == 3
#guard refusal (laidHeld keyedT (recordT 5 [])) == ""
#guard refusal (laidHeld keyedT (recordT 6 [])) == MEMBERS_LAID
#guard refusal (laidHeld keyedT (recordT 4 [])) == MEMBERS_LAID
#guard refusal (laidHeld #[memberT "a" #[(0, 2), (3, 5)], memberT "b" #[(2, 3)]] (recordT 5 [])) == MEMBERS_LAID
#guard refusal (laidHeld #[memberT "a" #[(0, 2)], memberT "b" #[(3, 6)]] (recordT 5 [])) == MEMBERS_LAID
#guard refusal (laidHeld keyedT Json.null) == "the record names no population"

/-- A registered file of the ports given, each its name, its value, the hex digit its root repeats and its leaves. -/
private def regT (ports : List (String × String × Char × Nat)) : Json :=
  Json.mkObj [("format", Json.str Flock.Registered.FORMAT),
    ("ports", Json.mkObj (ports.map fun (name, value, c, leaves) => (name, Json.mkObj [
      ("domain", Json.str ("".pushn 'd' 128)), ("leaves", Lean.toJson leaves), ("reads", Json.arr #[]),
      ("root", Json.str ("".pushn c 128)), ("value", Json.str value)])))]

/-- The member `name` with the registered file `reg`. -/
private def readerT (name : String) (reg : Json) : Member :=
  { memberT name #[(0, 1)] with own := { sharedT with registered := some reg } }

#guard refusal (classHeld keyedT) == ""
#guard refusal (classHeld #[readerT "a" (regT [("p", "w", 'a', 2)]), memberT "b" #[(1, 2)],
  readerT "c" (regT [("q", "w", 'a', 2), ("r", "v", 'b', 2)])]) == ""
#guard whyT (classHeld #[readerT "a" (regT [("p", "w", 'a', 2)]), memberT "b" #[(1, 2)],
  readerT "c" (regT [("q", "w", 'b', 2)])]) ==
  classTwo "a" "c" "w" ("root " ++ "".pushn 'a' 128 ++ ", against " ++ "".pushn 'b' 128)
#guard whyT (classHeld #[readerT "a" (regT [("p", "w", 'a', 2)]), readerT "c" (regT [("q", "w", 'a', 3)])]) ==
  classTwo "a" "c" "w" "leaves 2, against 3"
#guard whyT (classHeld #[readerT "a" (regT [("p", "w", 'a', 2), ("q", "w", 'b', 2)])]) ==
  classTwo "a" "a" "w" ("root " ++ "".pushn 'a' 128 ++ ", against " ++ "".pushn 'b' 128)
#guard (whyT (classHeld #[memberT "a" #[(0, 1)], readerT "c" Json.null])).startsWith "member c: "

/-- The slice `rootsOf` and `Member.slice` give the member `name` of `keyedT` on the roots given. -/
private def sliceT (roots : List (String × String)) (name : String) : Json :=
  match rootsOf keyedT (recordT 5 roots), keyedT.find? (·.name == name) with
  | .ok rs, some m => m.slice (recordT 5 roots) rs
  | _, _ => Json.null

/-- A public file's header of the scheme `s`, `n` instances and the roots given, each a bare port and a root. -/
private def headerT (n : Nat) (roots : List (String × String)) : Json :=
  Json.mkObj [("frame_v3", Json.mkObj [("domain_ids", Json.mkObj (roots.map fun (p, r) => (p, Json.str ("d" ++ r)))),
    ("roots", Json.mkObj (roots.map fun (p, r) => (p, Json.str r)))]), ("instances", Lean.toJson n),
    ("scheme", Json.str "s")]

#guard refusal (headerHeld (sliceT [("t/c0/x", "aa"), ("t/c1/y", "bb"), ("t/c0/y", "cc")] "t/c0")
  (headerT 2 [("x", "aa"), ("y", "cc")])) == ""
#guard refusal (headerHeld (sliceT [("t/c0/x", "aa"), ("t/c1/y", "bb"), ("t/c0/y", "cc")] "t/c1")
  (headerT 3 [("y", "bb")])) == ""
#guard refusal (headerHeld (sliceT [("t/c0/x", "aa"), ("t/c1/y", "bb"), ("t/c0/y", "cc")] "t/c1")
  (headerT 3 [("y", "cc")])) == "public file: not the registration's: roots, domains"
#guard refusal (headerHeld (sliceT [("t/c0/x", "aa"), ("t/c1/y", "bb"), ("t/c0/y", "cc")] "t/c0")
  (headerT 2 [("x", "aa")])) == "public file: not the registration's: roots, domains"
#guard refusal (headerHeld (sliceT [("t/c0/x", "aa"), ("t/c1/y", "bb")] "t/c0")
  (headerT 2 [("x", "aa"), ("y", "cc")])) == "public file: not the registration's: roots"
#guard refusal (headerHeld (sliceT [("t/c0/x", "aa"), ("t/c1/y", "bb")] "t/c0") (headerT 5 [("x", "aa")])) ==
  "public file: not the registration's: population"
#guard refusal (headerHeld (sliceT [("t/c0/x", "aa")] "t/c1") (headerT 3 [])) == ""

#guard Flock.hexOf (evidenceBytes #["ab".toUTF8]) == "0000000000000001" ++ "0000000000000002" ++ "6162"
#guard (evidenceParts (evidenceBytes #["ab".toUTF8, .empty, "c".toUTF8])).map (·.map Flock.hexOf) ==
  some #["6162", "", "63"]
#guard (evidenceParts (evidenceBytes #[])).map (·.size) == some 0
#guard Flock.hexOf (evidenceBytes #["a".toUTF8, "b".toUTF8]) != Flock.hexOf (evidenceBytes #["ab".toUTF8])
#guard (evidenceParts ((evidenceBytes #["ab".toUTF8]).push 0)).isNone
#guard (evidenceParts ((evidenceBytes #["ab".toUTF8]).extract 0 17)).isNone
#guard (evidenceParts .empty).isNone
#guard (whyT (sessionEvidence ⟨#[]⟩ { name := "s", record := "".pushn 'a' 128, reps := #[] })).startsWith
  "session s's record: "

end Spot.OneStageZk
