module

public import Spot.OneStageZkStatement
import all Init.Util
meta import Flock
meta import Spot.Restrict
meta import Spot.Edges
meta import Circuits.DefinitionDigest
meta import Spot.OneStageZkStatement

/-!
# One-stage v2: the files' pass with each circuit file read once

The files' pass over the members (`membersFiles`, in `Spot/OneStageZkStatement.lean`) with each circuit file parsed once
and the program's strata evaluated once (`ProgramKept`, `keptFiles`, `membersFilesKept`), which is compiled in its place
(`membersFiles_eq_kept`).
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Spot.OneStageZk

open Lean (Json)
open Flock (V)

/-- The tags a statement's sessions run under (`Flock.Witness.zkTags`): of the statement's inputs they read only its
own tags `tags0` and its circuit file. -/
def tagsAt (tags0 : Flock.Tags) (circuitB : ByteArray) : Flock.Tags :=
  Flock.Zk.tags tags0 (Flock.Zk.maskLogOf circuitB) 1

/-- **What the files' pass keeps of a circuit file**, each a function of the file, the statement's tags and the
blobs' lookup tables, so the members that name one file share one: the blob under its digest (`circuit`), the pinned
tables its META looks up (`lookups`), those of them the blobs hold, the lookup tables of a member that names the file
(`tables`, as `ownTables` reads them), the tags its statement's sessions run under (`tags`), and what the statement's
checks read of its parse at those tables (`kept`: the circuit's own verdicts and the circuit without its rows,
`Spot.CircuitKept.Kept`). It does not hold the parse. -/
structure FileKept where
  circuit : V ByteArray
  lookups : V (List String)
  tables : V (List (String × ByteArray))
  tags : Flock.Tags
  kept : V CircuitKept.Kept

/-- What the files' pass keeps of the circuit file `cf`, at the statement's tags `tags0` and the blobs' lookup tables
`pinned`: its META read and its rows parsed, at the tables of `pinned` it looks up, here and nowhere else in the pass. -/
def fileOf (tags0 : Flock.Tags) (pinned : List (String × ByteArray)) (cf : V ByteArray) : FileKept :=
  match cf with
  | .error e => { circuit := .error e, lookups := .error e, tables := .error e, tags := tags0, kept := .error e }
  | .ok circuitB =>
    match lookupsOf circuitB with
    | .error e =>
      { circuit := .ok circuitB, lookups := .error e, tables := .error e, tags := tagsAt tags0 circuitB
        kept := .error e }
    | .ok names =>
      { circuit := .ok circuitB, lookups := .ok names, tables := .ok (pinned.filter fun t => names.contains t.1)
        tags := tagsAt tags0 circuitB
        kept := (Flock.ProvedScope.circuitOf (tagsAt tags0 circuitB) circuitB
            (pinned.filter fun t => names.contains t.1)).map
          (CircuitKept.keptOf (tagsAt tags0 circuitB) true 1 templateOk) }

/-- What `fileOf` keeps under a digest whose blob is `circuitB`, a circuit file that looks up `names`. -/
theorem fileOf_ok {tags0 : Flock.Tags} {pinned : List (String × ByteArray)} {cf : V ByteArray}
    {circuitB : ByteArray} {names : List String} (hc : cf = .ok circuitB)
    (hl : lookupsOf circuitB = .ok names) :
    fileOf tags0 pinned cf =
      { circuit := .ok circuitB, lookups := .ok names, tables := .ok (pinned.filter fun t => names.contains t.1)
        tags := tagsAt tags0 circuitB
        kept := (Flock.ProvedScope.circuitOf (tagsAt tags0 circuitB) circuitB
            (pinned.filter fun t => names.contains t.1)).map
          (CircuitKept.keptOf (tagsAt tags0 circuitB) true 1 templateOk) } := by
  unfold fileOf
  rw [hc]
  dsimp only
  rw [hl]

/-- What `fileOf` keeps under a digest whose blob's META's `lookups` does not read: the refusal. -/
theorem fileOf_unread {tags0 : Flock.Tags} {pinned : List (String × ByteArray)} {cf : V ByteArray} {e : String}
    {circuitB : ByteArray} (hc : cf = .ok circuitB) (hl : lookupsOf circuitB = .error e) :
    fileOf tags0 pinned cf =
      { circuit := .ok circuitB, lookups := .error e, tables := .error e, tags := tagsAt tags0 circuitB
        kept := .error e } := by
  unfold fileOf
  rw [hc]
  dsimp only
  rw [hl]

/-- What `fileOf` keeps under a digest no blob has: the refusal. -/
theorem fileOf_error {tags0 : Flock.Tags} {pinned : List (String × ByteArray)} {cf : V ByteArray} {e : String}
    (hc : cf = .error e) :
    fileOf tags0 pinned cf =
      { circuit := .error e, lookups := .error e, tables := .error e, tags := tags0, kept := .error e } := by
  unfold fileOf
  rw [hc]

/-- A bind on a value returned is its continuation at the value. -/
theorem ok_bind {α β : Type} (a : α) (f : α → V β) : (Except.ok a : V α) >>= f = f a :=
  rfl

/-- A bind on a refusal is the refusal. -/
theorem error_bind {α β : Type} (e : String) (f : α → V β) : (Except.error e : V α) >>= f = Except.error e :=
  rfl

/-- The strata of the partition `partition`'s query over a program as parsed, `parsed`: what a member's population
check reads of its program and its partition (`Member.population`). -/
def strataOn (parsed : V Program) (partition : Json) : V (Array (String × Array (Nat × Nat))) :=
  parsed >>= fun p => Partition.strata p (getD partition "query")

/-- Whether two readings of a query's strata are one: the same strata, or the same refusal. -/
def sameStrata : V (Array (String × Array (Nat × Nat))) → V (Array (String × Array (Nat × Nat))) → Bool
  | .ok a, .ok b => decide (a = b)
  | .error a, .error b => decide (a = b)
  | _, _ => false

/-- Two readings `sameStrata` takes as one are equal. -/
theorem sameStrata_eq : ∀ {a b : V (Array (String × Array (Nat × Nat)))}, sameStrata a b = true → a = b
  | .ok _, .ok _, h => congrArg Except.ok (of_decide_eq_true h)
  | .error _, .error _, h => congrArg Except.error (of_decide_eq_true h)
  | .ok _, .error _, h => nomatch h
  | .error _, .ok _, h => nomatch h

/-- `sameStrata` takes a reading as itself. -/
theorem sameStrata_refl : ∀ (a : V (Array (String × Array (Nat × Nat)))), sameStrata a a = true
  | .ok _ => decide_eq_true rfl
  | .error _ => decide_eq_true rfl

/-- **What the files' pass keeps of a program**, for the members that read it: the bytes, their parse
(`Program.parse`) and the digest a session's record must name (`programDigest`), each a function of the bytes; and
the partition the member before read with it, with the strata of that partition's query over the parse (`strata`,
which `held` says they are), so the members in a row that read one program under one partition evaluate its query
once. -/
structure ProgramKept where
  bytes : ByteArray
  parsed : V Program
  sha : Option String
  partition : Json
  strata : V (Array (String × Array (Nat × Nat)))
  held : strata = strataOn parsed partition

/-- What is kept of the program `bytes`, parsed as `parsed` and of the digest `sha`, under the partition
`partition`: the strata of its query evaluated here. -/
def programRead (bytes : ByteArray) (parsed : V Program) (sha : Option String) (partition : Json) : ProgramKept :=
  { bytes, parsed, sha, partition, strata := strataOn parsed partition, held := rfl }

/-- What the files' pass keeps of the program `b` under the partition `partition`: `b` parsed and hashed here, and
the partition's query evaluated on the parse. -/
def programOf (b : ByteArray) (partition : Json) : ProgramKept :=
  programRead b (Program.parse b) (programDigest b) partition

/-- A program kept with any strata that are its partition's is the program kept with them evaluated. -/
theorem programRead_of {bytes : ByteArray} {parsed : V Program} {sha : Option String} {partition : Json}
    {strata : V (Array (String × Array (Nat × Nat)))} (h : strata = strataOn parsed partition) :
    ({ bytes, parsed, sha, partition, strata, held := h } : ProgramKept) = programRead bytes parsed sha partition := by
  subst h
  rfl

/-- The strata kept are those of the kept partition's query, so `sameStrata` takes the two as one under a partition
equal to the kept one. -/
theorem ProgramKept.same (pk : ProgramKept) {partition : Json} (e : partition = pk.partition) :
    sameStrata (strataOn pk.parsed partition) pk.strata = true := by
  subst e
  rw [← pk.held]
  exact sameStrata_refl pk.strata

/-- What `withPtrEq` answers is what its continuation answers. -/
theorem withPtrEq_answer {α : Type} (a b : α) (k : Unit → Bool) (h : a = b → k () = true) :
    withPtrEq a b k h = k () := by
  simp only [withPtrEq]

/-- **What is kept of `pk`'s program for a member whose partition is `partition`**: `pk`'s strata where `partition`
is the partition `pk` evaluated them under, else the strata of `partition`'s query on `pk`'s parse. Compiled, the
partition the members share is one object, which `withPtrEq` sees without reading it, so no query is evaluated; a
partition that is another object has its query evaluated and the strata compared (`sameStrata`), and evaluated once
more where they differ. Either way the result is `programRead` at `partition` (`ProgramKept.under_eq`). -/
def ProgramKept.under (pk : ProgramKept) (partition : Json) : ProgramKept :=
  if h : withPtrEq partition pk.partition (fun _ => sameStrata (strataOn pk.parsed partition) pk.strata)
      (fun e => pk.same e) = true then
    { bytes := pk.bytes, parsed := pk.parsed, sha := pk.sha, partition, strata := pk.strata
      held := (sameStrata_eq (a := strataOn pk.parsed partition) (b := pk.strata)
        ((withPtrEq_answer _ _ _ _).symm.trans h)).symm }
  else programRead pk.bytes pk.parsed pk.sha partition

/-- What `ProgramKept.under` keeps is `pk`'s program with the strata of `partition`'s query on its parse. -/
theorem ProgramKept.under_eq (pk : ProgramKept) (partition : Json) :
    pk.under partition = programRead pk.bytes pk.parsed pk.sha partition := by
  unfold ProgramKept.under
  split
  · exact programRead_of _
  · rfl

/-- What is kept of the program `b` under the partition `partition` after the member before read `prog`: `prog`
under `partition` where its bytes are `b` (`ProgramKept.under`), else `b` parsed and hashed anew, so members in a
row that read one program parse and hash it once, and evaluate their partition's query on it once. -/
def ProgramKept.next (prog : Option ProgramKept) (b : ByteArray) (partition : Json) : ProgramKept :=
  match prog with
  | some pk => if pk.bytes = b then pk.under partition else programOf b partition
  | none => programOf b partition

/-- What is kept of a program after a member is what `programOf` reads of its bytes under the member's partition. -/
theorem ProgramKept.next_eq {prog : Option ProgramKept}
    (h : ∀ pk, prog = some pk → pk = programOf pk.bytes pk.partition) (b : ByteArray) (partition : Json) :
    ProgramKept.next prog b partition = programOf b partition := by
  unfold ProgramKept.next
  cases prog with
  | none => rfl
  | some pk =>
    dsimp only
    split
    · rename_i hb
      have hp : pk.parsed = Program.parse pk.bytes := congrArg ProgramKept.parsed (h pk rfl)
      have hs : pk.sha = programDigest pk.bytes := congrArg ProgramKept.sha (h pk rfl)
      subst hb
      rw [ProgramKept.under_eq, hp, hs] <;> rfl
    · rfl

/-- `Member.population` on the strata of its partition's query over its program, `strata`, read before it. -/
def Member.populationOn (m : Member) (strata : V (Array (String × Array (Nat × Nat)))) (slice : Json)
    (pubB : ByteArray) : V Unit := do
  let strata ← strata
  let h ← FilePart.header pubB
  m.heldAt strata h
  headerHeldAt (← m.reads) (← m.held) slice h

/-- `Member.population` is `Member.populationOn` at the strata of the member's partition's query over its program's
parse. -/
theorem Member.population_on (m : Member) (slice : Json) :
    m.population slice = m.populationOn (strataOn (Program.parse m.own.program) m.own.partition) slice := by
  funext pubB
  exact (bind_assoc (Program.parse m.own.program)
    (fun p => Partition.strata p (getD m.own.partition "query")) _).symm

/-- `StepAt.ofOwn` on what is kept of the circuit file and the program: the lookup tables `tables` it is handed, the
tags its sessions run under `tags`, what its checks read of the parse `k`, and the program's digest `sha`. Every
check that read the parse reads `k`, and the four that load the public file read one load of it
(`Spot.CircuitKept.loadOf`). -/
def StepAt.ofKept (p : Own) (tags0 : Flock.Tags) (circuitB pubB : ByteArray) (tables : List (String × ByteArray))
    (rd : Reads) (population : V Unit) (tags : Flock.Tags) (k : V CircuitKept.Kept) (sha : Option String) : StepAt :=
  let pub := CircuitKept.loadOf k tags pubB
  { I := inputsAt tags0 p circuitB pubB tables, rd
    scope := CircuitKept.checkOf k pub tags true pubB 1 false rd.registered.isSome >>= fun _ => k >>= fun k => k.own
    ksFiles := CircuitKept.portsOf k pub
    ksForm := CircuitKept.filesOf k pub tags pubB
    header := CircuitKept.headerOf k pub
    g := k.toOption.map (·.c.g)
    population
    programSha := sha
    want := getD p.partition "program" }

/-- The scope check on the parse is the scope check on what is kept of it and the public file's one load, then the
kept verdict of `templateOk`. -/
theorem scopeAt_kept (parsed : V Flock.Circuit) (I : Flock.Witness.Inputs) (rd : Reads) :
    scopeAt parsed I rd =
      (CircuitKept.checkOf (parsed.map (CircuitKept.keptOf (Flock.Witness.zkTags I) true I.count templateOk))
          (CircuitKept.loadOf (parsed.map (CircuitKept.keptOf (Flock.Witness.zkTags I) true I.count templateOk))
            (Flock.Witness.zkTags I) I.publicFile)
          (Flock.Witness.zkTags I) true I.publicFile I.count false rd.registered.isSome >>= fun _ =>
        parsed.map (CircuitKept.keptOf (Flock.Witness.zkTags I) true I.count templateOk) >>= fun k => k.own) := by
  unfold scopeAt
  rw [CircuitKept.checkOf_kept parsed _ true _ _ false _ templateOk]
  cases parsed <;> rfl

/-- **`StepAt.ofOwn` is `StepAt.ofKept` at what `fileOf` keeps of its circuit file**: the statement on the parse is
the statement on the circuit's own verdicts and the circuit without its rows. -/
theorem StepAt.ofOwn_kept (p : Own) (tags0 : Flock.Tags) (circuitB pubB : ByteArray)
    (tables : List (String × ByteArray)) (rd : Reads) (population : V Unit) :
    StepAt.ofOwn p tags0 circuitB pubB tables rd population =
      StepAt.ofKept p tags0 circuitB pubB tables rd population (tagsAt tags0 circuitB)
        ((Flock.ProvedScope.circuitOf (tagsAt tags0 circuitB) circuitB tables).map
          (CircuitKept.keptOf (tagsAt tags0 circuitB) true 1 templateOk)) (programDigest p.program) := by
  show StepAt.mk _ _ _ _ _ _ _ _ _ _ = StepAt.mk _ _ _ _ _ _ _ _ _ _
  congr 1
  · exact scopeAt_kept _ _ _
  · exact CircuitKept.portsOf_kept _ _ _ true 1 templateOk
  · exact CircuitKept.filesOf_kept _ _ _ true 1 templateOk
  · exact CircuitKept.headerOf_kept _ _ _ true 1 templateOk
  · exact CircuitKept.keptOf_g _ _ true 1 templateOk

/-- `statementOf` at a member's own lookup tables, on what the files' pass keeps of its circuit file (`fk`) and its
program's digest (`sha`), at the statement's tags `tags0`: the tables kept of the file, then `statementOf`'s checks
in its order, each read of the circuit file taken from `fk`. -/
def statementOn (tags0 : Flock.Tags) (fk : FileKept) (sha : Option String) (p : Own)
    (population : ByteArray → V Unit) (bs : Core.Blobs) : V StepAt :=
  fk.tables >>= fun tables =>
  (fk.lookups >>= tablesMatch tables) >>= fun _ =>
  match tags0.plainLeaves with
  | true => throw s!"{tags0.name}: a statement with plain SHA-512 leaves: --zk's hiding needs hm96 leaves"
  | false =>
    match p.registered.isNone && readsRegistered p.programJ with
    | true =>
      throw ("registered: the program reads registered values (a RegisteredInput among its definitions) and the " ++
        "verifier holds no registered reads: verify with its own registered file")
    | false =>
      fk.circuit >>= fun circuitB =>
      bs.get p.pub "public file" >>= fun pubB =>
      match fitsWhy WORD circuitB pubB tables (some p.program) (some p.partition) p.registered with
      | some what => throw s!"{what}: {FITS}"
      | none =>
        readsOf p.registered >>= fun rd =>
          pure (StepAt.ofKept p tags0 circuitB pubB tables { rd with linked := p.linked } (population pubB) fk.tags fk.kept sha)

/-- **`statementOf` at the inputs' own lookup tables (`ownTables`) is `statementOn` at what `fileOf` keeps of its
circuit file, read at the blobs' lookup tables, and its program's digest**, where the statement's tags are known. -/
theorem statementOf_on {tags0 : Flock.Tags} (ht : Flock.Tags.byName? STATEMENT = some tags0) (p : Own)
    (population : ByteArray → V Unit) (bs : Core.Blobs) :
    (ownTables p bs >>= fun tables => statementOf p population tables bs) =
      statementOn tags0 (fileOf tags0 (tablesOf bs) p.circuitFile) (programDigest p.program) p population bs := by
  unfold ownTables statementOf statementOn
  simp only [StepAt.ofOwn_kept, ht]
  cases hc : p.circuitFile with
  | error e =>
    rw [fileOf_error rfl]
    simp only [error_bind] <;> rfl
  | ok circuitB =>
    simp only [ok_bind]
    cases hl : lookupsOf circuitB with
    | error e =>
      rw [fileOf_unread rfl hl]
      simp only [error_bind] <;> rfl
    | ok names =>
      rw [fileOf_ok rfl hl]
      simp only [ok_bind, pure_bind] <;> rfl

/-- `Member.files` on what the files' pass keeps of the member's circuit file (`fk`) and of its program (`pk`):
`statementOn` in place of `statementOf`, and the population check on the strata kept of the program. -/
def Member.filesOn (m : Member) (tags0 : Flock.Tags) (fk : FileKept) (pk : ProgramKept) (slice : Json)
    (bs : Core.Blobs) : V StepAt :=
  ((statementOn tags0 fk pk.sha m.own (m.populationOn pk.strata slice) bs >>= fun a =>
      m.blockHeld a.g >>= fun _ => pure a) >>=
    fun a => a.filesHeld >>= fun _ => pure a).mapError (s!"member {m.name}: {·}")

/-- `Member.files` is `Member.filesOn` at what `fileOf` keeps of its circuit file, read at the blobs' lookup tables,
and what `programOf` keeps of its program under its partition. -/
theorem Member.files_on {tags0 : Flock.Tags} (ht : Flock.Tags.byName? STATEMENT = some tags0) (m : Member)
    (slice : Json) (bs : Core.Blobs) :
    m.files slice bs =
      m.filesOn tags0 (fileOf tags0 (tablesOf bs) m.own.circuitFile) (programOf m.own.program m.own.partition)
        slice bs := by
  unfold Member.files Member.statement Member.filesOn
  rw [statementOf_on ht, Member.population_on] <;> rfl

/-- `membersFiles` on the program `prog` the member before read: each member's files held on what `fileOf` keeps of
its circuit file, derived from C and the partition (`Own.circuitFile`), and on what is kept of its program under its
partition (`ProgramKept.next`). A program is parsed and hashed, and its partition's query evaluated on it, once for
the members in a row that read it under the partition the parameters give them all. -/
def keptFiles (slice : Member → Json) (bs : Core.Blobs) (tags0 : Flock.Tags) (tables : List (String × ByteArray)) :
    Option ProgramKept → List Member → V (List (String × StepAt))
  | _, [] => pure []
  | prog, m :: ms =>
    let pk := ProgramKept.next prog m.own.program m.own.partition
    m.filesOn tags0 (fileOf tags0 tables m.own.circuitFile) pk (slice m) bs >>= fun a =>
      keptFiles slice bs tags0 tables (some pk) ms >>= fun as => pure ((m.name, a) :: as)

/-- `keptFiles` on a program that is what `programOf` reads of its bytes under its partition is `membersFiles`. -/
theorem keptFiles_eq {tags0 : Flock.Tags} (ht : Flock.Tags.byName? STATEMENT = some tags0) (slice : Member → Json)
    (bs : Core.Blobs) : ∀ (ms : List Member) (prog : Option ProgramKept),
      (∀ pk, prog = some pk → pk = programOf pk.bytes pk.partition) →
      keptFiles slice bs tags0 (tablesOf bs) prog ms = membersFiles slice bs ms
  | [], _, _ => rfl
  | m :: ms, prog, hp => by
    have hnext : ∀ pk, some (programOf m.own.program m.own.partition) = some pk →
        pk = programOf pk.bytes pk.partition := by
      intro pk h
      cases h
      rfl
    rw [keptFiles, membersFiles, Member.files_on ht, ProgramKept.next_eq hp]
    rw [keptFiles_eq ht slice bs ms _ hnext]

/-- **Every member's files, a circuit file read and parsed once for the members that name it**, and the program's
strata evaluated once for the members that read it: `keptFiles` from no file and no program kept, at the statement's
tags and the blobs' lookup tables, each read once. It returns what `membersFiles` returns, a refusal included
(`membersFiles_eq_kept`), and is compiled in its place. -/
def membersFilesKept (slice : Member → Json) (bs : Core.Blobs) (ms : List Member) : V (List (String × StepAt)) :=
  match Flock.Tags.byName? STATEMENT with
  | some tags0 => keptFiles slice bs tags0 (tablesOf bs) none ms
  | none => membersFiles slice bs ms

/-- **The files' pass with each circuit file read and parsed once is the files' pass**: `membersFilesKept` returns
exactly what `membersFiles` returns, a statement for each member or the first refusal, so every statement about
`membersFiles` is one about the compiled pass. -/
@[csimp] theorem membersFiles_eq_kept : @membersFiles = @membersFilesKept := by
  funext slice bs ms
  unfold membersFilesKept
  cases ht : Flock.Tags.byName? STATEMENT with
  | none => rfl
  | some tags0 => exact (keptFiles_eq ht slice bs ms none (fun _ h => nomatch h)).symm

end Spot.OneStageZk
