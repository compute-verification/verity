/-
Sampled proofs' recursive audit (`verity/sampled-proofs/recursive/v1`), a `Core.BlobProtocol`: one-stage v2's audit
(`SampledProofs.OneStageZk`) with its one verdict of record computed by V*'s compiled check, the recursive verifier the
auditor runs (`flock-verify verify --zk` once per V* part with that part's recursive inputs), on the blobs.

* **params**: v1's, plus `recursive`, exactly `{"circuit", "draw", "parts", "public", "registration"}`: the SHA-512s of
  the inner statement's circuit file, its draw (`--rec-inner-draw`), its public file and the circuit registration
  (`--rec-inner-registration`, or null), and for each of V*'s statements at the inner shape, in `RecVStar.statements`'
  order, exactly `{"bindings", "circuit", "partition", "program", "registered"}`: the part's `--rec-bindings` object,
  its circuit's SHA-512, its partition object, its program's descriptor and its registered file (or null). The inner
  statement's program and partition are v1's `program` and `partition`, which must not be null. Every one of
  them is the verifier's own and is fixed before any coin, `RecDeploy.RecX`'s statement;
* **commitment**: v1's (the registration record);
* **evidence**: v1's without `verdicts`, with exactly one session, plus `recursive`, exactly `{"parts", "record", "rows",
  "transcript"}`: the SHA-512s of the inner session's record, of V*'s shared rows (`--rec-rows`, one file every part
  reads: they register the prover's running sums, so the prover stages them, `RecDeploy.Stage.rows`) and of the coin
  server's transcript of the inner session, and for each part `{"public", "record", "reps"}`: its staged public file's
  SHA-512, its `session.json`'s and `{file: SHA-512}` naming at least each rep file its session reads;
* **blobs**: the files those digests name (`Core.Blobs`);
* **check**: the verifier's inner shape from the inner statement's files (`RecVStar.innerOf`, as `flock-verify
  rec-inner --program` writes it), and Flock's plain verifier's own checks on the inner statement at its draw
  (`innerOk`, `RecDeploy.innerOk`); then, for each part `j`, `flock-verify verify --zk --statement verity/flock-circuit`
  with the part's `--circuit`, `--public`, `--partition`, `--program`, `--registered`, `--rec-inner`, `--rec-part j`,
  `--rec-bindings` and `--rec-rows`, the inner statement's files (`--rec-inner-circuit`, `--rec-inner-public`,
  `--rec-inner-partition`, `--rec-inner-statement verity/flock-circuit+plain-leaves --rec-inner-statement-coins os
  --rec-inner-draw`), `--rec-inner-session`, `--rec-inner-transcript` and `--rec-inner-registration`, and one `--session`
  (`partVerdict`, the C-Flock library calls `verifyCmd` makes on that path, as `recursion_e2e`'s lean step runs it); then
  v1's `check` on one verdict of record (`flock-verify/<session>`), accepted exactly when every part's is, whose units
  are the inner public file's units at the inner record's draw.

**Form.** Only the dense form, the one `Flock.Guarantees.RecursiveKSAt`'s statements (`RecDeploy.DenseAt`) cover: the
inner circuits are public (no `--hide`), there is no RecSparse phase (no `--sparse`), and an inner circuit whose META
names R's shape is refused, as `rec-inner` refuses it without `--sparse`. One recursive session, with one part per V*
statement at the inner shape, in order. Any other form is refused, as are a part statement with plain SHA-512 leaves
(`verify --zk`'s refusal) and a staged public file, inner record or transcript of `2^56` bytes or more (`fits`).

Without blobs (`toProtocol`) every check rejects: no blob holds the inner circuit file.
-/
import CoreProtocol.Blobs
import SampledProofs.OneStageZk
import Flock

namespace VStar.Recursive

open Lean (Json)
open Flock (V)
open SampledProofs

def PROTOCOL : String := "verity/sampled-proofs/recursive/v1"
/-- The statement every V* part's session proves. -/
def STATEMENT : String := "verity/flock-circuit"
/-- The inner statement's tags, as each V* part's setup is told to set it up (`--rec-inner-statement`). -/
def INNER_STATEMENT : String := Flock.Tags.circuitPlainLeaves.name

/-- A V* part, as the verifier holds it before any coin. -/
structure PartOwn where
  bindings : Json
  circuit : String
  partition : Json
  programJ : Json
  program : ByteArray
  registered : Option Json

/-- The verifier's own inputs: the inner statement's circuit, public file, draw and registration digests, its program
and partition, and the parts. -/
structure Own where
  circuit : String
  pub : String
  draw : String
  registration : Option String
  programJ : Json
  program : ByteArray
  partition : Json
  parts : Array PartOwn

structure Params where
  base : OneStage.Params
  own : Own

/-- A V* part's session, as the prover sends it: its staged public file, its record and its rep files. -/
structure PartProof where
  pub : String
  record : String
  reps : Array (String × String)

/-- The recursive session's files, by SHA-512. -/
structure Proofs where
  record : String
  rows : String
  transcript : String
  parts : Array PartProof

structure Evidence where
  base : OneStage.Evidence
  proofs : Proofs

private def exactly (j : Json) (ks : Array String) (what : String) : V Unit :=
  if keys? j != some (sortStrings ks) then throw s!"the keys of {what} are not {", ".intercalate (sortStrings ks).toList}"
  else pure ()

private def digest? (s : String) : Bool := s.length == 128 && isHex s

private def digestOf (j : Json) (what : String) : V String := do
  let .str d := j | throw s!"{what} is not a SHA-512 digest (128 lowercase hex)"
  if !digest? d then throw s!"{what} is not a SHA-512 digest (128 lowercase hex)"
  pure d

private def objOrNull (j : Json) (what : String) : V (Option Json) :=
  match j with
  | .null => pure none
  | o@(.obj _) => pure (some o)
  | _ => throw s!"{what} is not an object or null"

private def without (j : Json) (k : String) : List (String × Json) :=
  match j with
  | .obj kvs => (kvs.toArray.filter (·.1 != k)).toList
  | _ => []

private def partOwnOf (j : Json) : V PartOwn := do
  exactly j #["bindings", "circuit", "partition", "program", "registered"] "a part"
  let circuit ← digestOf (getD j "circuit") "a part's circuit"
  let (bindings@(.obj _), partition@(.obj _), programJ@(.obj _)) := (getD j "bindings", getD j "partition", getD j "program")
    | throw "a part's bindings, partition and program are not objects"
  pure { bindings, circuit, partition, programJ, program := (← Flock.canon programJ).toUTF8,
         registered := ← objOrNull (getD j "registered") "a part's registered" }

/-- The params' `recursive`, with v1's program and partition the inner statement's. -/
def ownOf (rec_ programJ : Json) (partition : Option Json) : V Own := do
  exactly rec_ #["circuit", "draw", "parts", "public", "registration"] "recursive"
  let circuit ← digestOf (getD rec_ "circuit") "recursive: circuit"
  let pub ← digestOf (getD rec_ "public") "recursive: public"
  let draw ← digestOf (getD rec_ "draw") "recursive: draw"
  let registration ← match getD rec_ "registration" with
    | .null => pure none
    | j => some <$> digestOf j "recursive: registration"
  let some partition := partition
    | throw "partition is null: the inner statement is set up against the verifier's own partition object"
  let .arr ps := getD rec_ "parts" | throw "recursive: parts is not a list"
  pure { circuit, pub, draw, registration, programJ, program := (← Flock.canon programJ).toUTF8, partition,
         parts := ← ps.mapM partOwnOf }

def paramsOfJson (j : Json) : V Params := do
  let some r := field? j "recursive"
    | throw "the parameters name no recursive (the inner circuit's digest and V*'s parts)"
  let rest := Json.mkObj (without j "recursive")
  let base ← OneStage.paramsOfJson rest
  pure { base, own := ← (ownOf r (getD rest "program") base.verifier.partition).mapError (s!"recursive: " ++ ·) }

/-- The `verdicts` command's own inputs: exactly `{"partition", "program", "recursive"}`. -/
def ownOfJson (j : Json) : V Own := do
  exactly j #["partition", "program", "recursive"] "the verifier's own inputs"
  let (programJ@(.obj _), partition@(.obj _)) := (getD j "program", getD j "partition")
    | throw "the program and the partition are not objects"
  ownOf (getD j "recursive") programJ (some partition)

private def partProofOf (j : Json) : V PartProof := do
  exactly j #["public", "record", "reps"] "a part's session"
  let (.str pub, .str record, .obj _) := (getD j "public", getD j "record", getD j "reps")
    | throw "a part's session is not its public file's, its record's and its rep files' digests"
  let reps ← (entries (getD j "reps")).mapM fun (f, d) => match d with
    | .str d => pure (f, d)
    | _ => throw s!"a part's rep file {f}'s digest is not a string"
  pure { pub, record, reps }

/-- The evidence's `recursive`. -/
def proofsOf (j : Json) : V Proofs := do
  exactly j #["parts", "record", "rows", "transcript"] "the evidence's recursive"
  let (.str record, .str rows, .str transcript) := (getD j "record", getD j "rows", getD j "transcript")
    | throw "recursive: record, rows and transcript are not digests"
  let .arr ps := getD j "parts" | throw "recursive: parts is not a list"
  pure { record, rows, transcript, parts := ← ps.mapM partProofOf }

def evidenceOf (b : ByteArray) : V Evidence := do
  let j ← Core.canonicalJson b
  if !(j matches .obj _) then throw "the evidence is not an object"
  if (field? j "verdicts").isSome then
    throw "the evidence names verdicts: this audit's verdict is its own run of V*'s compiled check on the proofs"
  let some r := field? j "recursive" | throw "the evidence names no recursive (the recursive session's files)"
  let base ← OneStage.evidenceOf (← Flock.canon (Json.mkObj (without j "recursive" ++ [("verdicts", Json.mkObj [])]))).toUTF8
  if base.sessions.size != 1 then
    throw s!"the evidence names {base.sessions.size} sessions: a recursive audit is one recursive session"
  pure { base, proofs := ← proofsOf r }

/-- The end of the line that starts at `i` (the next newline's index, or the end). -/
def lineEnd (b : ByteArray) (i : Nat) : Nat :=
  if h : i < b.size then (if b[i] == 10 then i else lineEnd b (i + 1)) else i
termination_by b.size - i

/-- `flock-verify`'s `innerMeta` on the inner circuit's bytes: its `META` line among its first nine, as JSON. -/
def metaOf (b : ByteArray) : V Json := do
  let mut at_ := 0
  for _ in [0:9] do
    if at_ < b.size then
      let e := lineEnd b at_
      let some line := String.fromUTF8? (b.extract at_ e) | throw "rec inner circuit: not UTF-8"
      if line.startsWith "META " then
        match Json.parse (line.drop 5).toString with
        | .ok m => return m
        | .error err => throw s!"rec inner circuit: META is not JSON: {err}"
      at_ := e + 1
  throw "rec inner circuit: no META line"

/-- **The verifier's inner shape**, dense: `flock-verify rec-inner --circuit --public --partition --program` without
`--hide` or `--sparse`, which refuses an inner circuit whose META names R's shape. -/
def shapeOf (circuitB pubB : ByteArray) (partition : Json) (program : ByteArray) : V Flock.RecVStar.Inner := do
  let stacked ← (do Flock.RecVStar.stackedShapeOf (← metaOf circuitB)).mapError (s!"rec-inner: " ++ ·)
  if let some r := stacked then
    throw (s!"rec-inner: the inner circuit's META names R's shape (m {r.m}, claims {r.claims}), and the verifier names " ++
      "no RecSparse shape (--sparse NU,ELL,S_LOG,T_LOG): R opens the sparse phase's claims, so V*'s list would miss RecSparse")
  let i ← (Flock.RecVStar.innerOf circuitB pubB (some partition) (some program) false).mapError (s!"rec-inner: " ++ ·)
  pure { i with holo := false, sparse := none, stacked := none }

/-- `flock-verify`'s `registeredFile` on a canonical object: the ports, `true` (`flock-verify` runs `checkComplete` on
every registered file), the known rows and the template parameters the ports read. -/
def registeredOf (j : Json) :
    V (Array Flock.Registered.Port × Bool × Array (String × Array (Array Nat × ByteArray)) × Array String) := do
  let own ← Flock.Registered.ofJson j
  let known ← Flock.Registered.knownOf j
  Flock.Registered.checkKnownAll own known
  pure (own, true, known ++ Flock.KnownRead.fullRows own (← Flock.KnownRead.readOf j),
    ← Flock.Registered.paramsOf j)

/-- `pure ()` when `b`, else the refusal `why`. -/
def need (b : Bool) (why : String) : V Unit := if b then pure () else throw why

/-- The bound on a file every part's session encodes as one field (the staged public file, the inner record and the coin
server's transcript), and on the whole statement the inputs name (`statementFits`): each field's length is one 8-byte
word, so the verifier refuses a file or statement of `2^56` bytes or more, under which every encoded field stays under
`2^64`. -/
def FIELD_MAX : Nat := 2 ^ 56

/-- `need` on a file's size: `what` is under `FIELD_MAX` bytes. -/
def fits (b : ByteArray) (what : String) : V Unit :=
  need (b.size < FIELD_MAX) s!"{what} is 2^56 bytes or more"

/-- **Flock's own checks on the inner statement** (`RecDeploy.innerOk`): Flock's plain verifier sets the inner statement
up as each part's setup is told to (`--statement verity/flock-circuit+plain-leaves --coins os`, the verifier's own
partition and program, the inner statement's draw, `--stratified` at the population the verifier's partition object and
program give (`85-rec-reprice.sh`'s inner `verify`: the inner session's full draw, `rec_live.full_draw`), no lookup
tables, registered reads or identifier: `verifyCmd`'s plain path), its scope check passes (`ProvedScope.check`), and it
carries a draw. The population is `HmRow.stratifiedLaw`'s at `K = 1`, which every nonempty population admits. -/
def innerOk (o : Own) (circuitB pubB drawB : ByteArray) : V Unit := do
  let some tags := Flock.Tags.find? INNER_STATEMENT | throw s!"the statement {INNER_STATEMENT} is not known"
  let coins := Flock.Coins.os
  let rtags := tags.withCoins coins
  let some t := String.fromUTF8? drawB | throw "draw: not UTF-8"
  let draw := some (Flock.norm (← (Json.parse t).mapError fun e => s!"draw: not JSON: {e}"))
  let partition := some o.partition
  let program := some o.program
  let (_, n) ← Flock.HmRow.stratifiedLaw o.partition o.program 1
  if !rtags.hm96Rows then
    throw s!"{rtags.name}: refused: a statement without hm96-sha512 rows has no draw, and every claim is about a drawn session"
  Flock.RecSetup.checkInner none none (pure (none, none))
  let (_, sts) ← Flock.Stmt.setupTables rtags circuitB pubB coins 1 [] draw partition program none (some n) none none
  for st in sts do
    Flock.checkRegistered st none
    OneStageZk.checkIdentifier st
  Flock.Registered.checkFull none draw
  Flock.RecSetup.checks none none tags [] sts circuitB partition program none none none none
  sts.forM (Flock.RecCoins.check · none (Flock.RecCoins.Source.atC3 none none none) (Flock.RecSetup.c3 none none))
  sts.forM (Flock.RecVStar.checkPubRows · none none none)
  Flock.ProvedScope.drawLaw false rtags false draw
  Flock.ProvedScope.check rtags false circuitB pubB [] 1 false false

/-- What every part's check reads of the inner session: the inner shape, the inner statement's files (check 5's
source), what the inner circuit's META binds, the coin server's transcript as `verify` holds it and its link points. -/
structure InnerHeld where
  shape : Flock.RecVStar.Inner
  src : Flock.RecCoins.Source
  bound : V (Option String × Option Flock.RecVStar.StackedShape)
  commits : Flock.RecVStar.InnerCommits
  link : Flock.RecVStar.InnerLink
  rows : Array Flock.Registered.Port

/-- The inner session's inputs on the blobs, as `verifyCmd` reads them before any session (`recInnerInput`,
`RecCoins.filesInput`, `innerMetaRegistration`, `recTranscriptInput`, `recLinkInput`), and V*'s shared rows the prover
staged (`--rec-rows`), the one set every part reads. -/
def innerHeld (o : Own) (pr : Proofs) (bs : Core.Blobs) : V InnerHeld := do
  let circuitB ← bs.get o.circuit "the inner circuit"
  let pubB ← bs.get o.pub "the inner public file"
  let shape ← shapeOf circuitB pubB o.partition o.program
  let recordB ← bs.get pr.record "the inner session's record"
  fits recordB "rec inner session: the record"
  let some record := String.fromUTF8? recordB
    | throw "rec inner session: the record is not UTF-8"
  let drawB ← bs.get o.draw "the inner statement's draw"
  (innerOk o circuitB pubB drawB).mapError (s!"rec inner statement: Flock's plain verifier refuses it: " ++ ·)
  let src : Flock.RecCoins.Source := .files
    { circuit := circuitB, pub := pubB, partition := o.partition, record,
      form := some { statement := INNER_STATEMENT, os := true, tables := [], draw := some drawB } }
  let bound : V (Option String × Option Flock.RecVStar.StackedShape) := do Flock.RecSetup.metaBound (← metaOf circuitB)
  let transcriptB ← bs.get pr.transcript "rec inner transcript"
  fits transcriptB "rec inner transcript"
  let some t := String.fromUTF8? transcriptB
    | throw "rec inner transcript: not UTF-8"
  let reg ← o.registration.mapM fun d => do
    let some s := String.fromUTF8? (← bs.get d "rec inner registration") | throw "rec inner registration: not UTF-8"
    pure s
  let commits ← Flock.RecSetup.commits (some shape) (pure t) (pure reg) bound record
  let link ← match Json.parse t with
    | .ok j => Flock.RecVStar.innerLink j record
    | .error e => throw s!"rec inner transcript: not JSON: {e}"
  let rowsB ← bs.get pr.rows "the staged rows"
  fits rowsB "the staged rows"
  let rows ← (do
      let some t := String.fromUTF8? rowsB | throw "not UTF-8"
      pure (← registeredOf (← (Json.parse t).mapError (s!"not JSON: " ++ ·))).1).mapError (s!"rec rows: " ++ ·)
  pure { shape, src, bound, commits, link, rows }

/-- One part's verdict: `verifyCmd`'s `--zk` path at part `j`'s recursive inputs and the inner session held, on its one
session, `.ok ()` or the session's refusal. An input `verify` would refuse before any session (its exit 2) is the
whole check's refusal. -/
def partVerdict (h : InnerHeld) (j : Nat) (p : PartOwn) (s : PartProof) (bs : Core.Blobs) : V (V Unit) := do
  let some tags0 := Flock.Tags.find? STATEMENT | throw s!"the statement {STATEMENT} is not known"
  need (!tags0.plainLeaves) (s!"verify --zk: refused: {tags0.name}, a statement with plain SHA-512 leaves " ++
    "(flock-leaf/sha512-unsalted): --zk's hiding needs hm96-sha512/v1 leaves")
  let count := 1
  let coins := Flock.Coins.tree (Flock.Zk.coinSpec (tags0.tableNames count))
  let circuitB ← bs.get p.circuit s!"part {j}'s circuit"
  let pubB ← bs.get s.pub s!"part {j}'s public file"
  fits pubB s!"part {j}'s public file"
  let tags := Flock.Zk.tags (tags0.withCoins coins) (Flock.Zk.maskLogOf circuitB) count
  let tables : List (String × ByteArray) := []
  let (registered, complete, known, regParams) ← match p.registered with
    | none => pure (none, false, #[], #[])
    | some r => (fun (own, c, k, ps) => (some own, c, k, ps)) <$> registeredOf r
  let rows := some h.rows
  let bindings ← Flock.RecVStar.ownBindingsOfJson p.bindings
  let partition := some p.partition
  let program := some p.program
  let inner := some h.shape
  let recCoins := some h.src
  let recCommits := some h.commits
  let parsed := Flock.ProvedScope.circuitOf tags circuitB tables
  let scope := Flock.ProvedScope.checkOf parsed tags true pubB count false registered.isSome
  let setupOf := fun (draw : Option Json) =>
    match (do
        Flock.Registered.checkProgram program registered.isSome
        Flock.RecSetup.checkInner inner recCoins h.bound
        let r ← OneStageZk.buildSessionOf parsed tags circuitB pubB coins tables count draw partition program registered
        if complete then r.2.forM fun (st : Flock.Stmt) => Flock.Registered.checkComplete st.c (registered.getD #[])
        r.2.forM fun (st : Flock.Stmt) => Flock.Registered.checkCover st.c program partition registered regParams
        r.2.forM fun (st : Flock.Stmt) => Flock.KnownRead.check st.c known
        Flock.RecSetup.checks inner recCommits tags0 tables r.2 circuitB partition program registered (some j) (some bindings)
          rows
        Flock.RecSetup.oneFull inner r.2
        r.2.forM Flock.Zk.shapeOk
        r.2.forM (Flock.RecCoins.check · registered (Flock.RecCoins.Source.atC3 recCoins inner recCommits)
          (Flock.RecSetup.c3 inner recCommits))
        r.2.forM (Flock.RecVStar.checkPubRows · registered recCommits (inner.bind (·.sparse)))
        Flock.RecVStar.checkLinkRounds h.shape h.link
        r.2.forM (Flock.RecVStar.checkTopRows · registered (some h.link))
        r.2.forM (Flock.RecCoins.checkLink · registered (Flock.RecCoins.Source.atC3 recCoins inner recCommits) (some h.link))
        Flock.RecVStar.checkLabels draw count r.2
        pure r) with
    | .ok (spec, sts) => (if draw.isSome then Flock.ProvedScope.drawLaw true tags false draw else .ok ()).map fun () =>
        (spec, sts.map Flock.Setup.ofCircuit)
    | .error e => Except.error s!"setup: {e}"
  let some record := String.fromUTF8? (← bs.get s.record s!"part {j}'s record")
    | throw s!"part {j}'s record is not UTF-8"
  let draw := if tags.hm96Rows then Flock.Witness.recordDraw record else none
  let st := setupOf draw
  let files : V (Array ByteArray) := do
    let (spec, _) ← st
    spec.repStreams.mapM fun (_, _, w) => do
      let n := s!"{w.replace "/" "."}.bin"
      let some (_, d) := s.reps.find? (·.1 == n) | throw s!"session files: the session names no rep file {n}"
      (bs.get d s!"session files: {n}")
  let verdict : V Unit := do
    let proofs ← files
    let (spec, tabs) ← st
    Flock.Zk.verify spec tabs record proofs
    scope
    Flock.ProvedScope.sizesOk true (tabs.map (·.m))
    if draw.isNone then Flock.ProvedScope.drawLaw true tags false draw
    Flock.RecSetup.fullDraw inner true draw record
    Flock.RecSetup.innerDrawn recCoins
    Flock.ProvedScope.programHeld program
    Flock.RecordText.check record
  pure verdict

/-- The one inner form the recursive audit's soundness covers, `RecDeploy`'s `Form.Dense`: public inner circuits
(`holo = false`), no RecSparse phase and no R. -/
def dense (i : Flock.RecVStar.Inner) : Bool := !i.holo && i.sparse.isNone && i.stacked.isNone

/-- `j`'s canonical bytes (`Flock.canon`): the form the statement the component's inputs name carries a parsed field
in. -/
def canonBytes (j : Json) : V ByteArray := do
  pure (← Flock.canon j).toUTF8

/-- What the statement the inputs name encodes beside one Flock statement's or the whole's files: its length words,
flags and format strings. -/
def STMT_OVER : Nat := 1024

/-- **Part `j`'s bytes in the statement the inputs name**: its statement's name, circuit, staged public file and program,
and the canonical bytes of its partition, registered reads, draw (the one `partVerdict` reads off its record) and
bindings. -/
def partSize (j : Nat) (p : PartOwn) (s : PartProof) (bs : Core.Blobs) : V Nat := do
  let some tags0 := Flock.Tags.find? STATEMENT | throw s!"the statement {STATEMENT} is not known"
  let circuitB ← bs.get p.circuit s!"part {j}'s circuit"
  let pubB ← bs.get s.pub s!"part {j}'s public file"
  let tags := Flock.Zk.tags (tags0.withCoins (Flock.Coins.tree (Flock.Zk.coinSpec (tags0.tableNames 1))))
    (Flock.Zk.maskLogOf circuitB) 1
  let some record := String.fromUTF8? (← bs.get s.record s!"part {j}'s record")
    | throw s!"part {j}'s record is not UTF-8"
  let drawB ← (if tags.hm96Rows then Flock.Witness.recordDraw record else none).mapM canonBytes
  let regB ← p.registered.mapM canonBytes
  let partitionB ← canonBytes p.partition
  let bindingsB ← canonBytes p.bindings
  pure (STMT_OVER + STATEMENT.toUTF8.size + circuitB.size + pubB.size + p.program.size + partitionB.size +
    (regB.map (·.size)).getD 0 + (drawB.map (·.size)).getD 0 + bindingsB.size)

/-- The parts' bytes in the statement, part `j` on. -/
def partsSize (bs : Core.Blobs) : Nat → List PartOwn → List PartProof → V Nat
  | j, p :: ps, s :: ss => do
    let a ← partSize j p s bs
    let b ← partsSize bs (j + 1) ps ss
    pure (a + b)
  | _, _, _ => pure 0

/-- **A bound on the bytes of the statement the inputs name** (`Names`): the inner statement's name, circuit, public file,
draw and program and its partition's canonical bytes, the inner shape's canonical bytes, the registration and each
part's (`partsSize`). -/
def statementSize (h : InnerHeld) (o : Own) (pr : Proofs) (bs : Core.Blobs) : V Nat := do
  let circuitB ← bs.get o.circuit "the inner circuit"
  let pubB ← bs.get o.pub "the inner public file"
  let drawB ← bs.get o.draw "the inner statement's draw"
  let regB ← o.registration.mapM fun d => bs.get d "rec inner registration"
  let partitionB ← canonBytes o.partition
  let shapeB ← canonBytes h.shape.toJson
  let partsB ← partsSize bs 0 o.parts.toList pr.parts.toList
  pure (STMT_OVER + (STMT_OVER + INNER_STATEMENT.toUTF8.size + circuitB.size + pubB.size + drawB.size + o.program.size +
    partitionB.size) + shapeB.size + (regB.map (·.size)).getD 0 + partsB)

/-- `need` on the statement's size: it is under `limit` bytes (`FIELD_MAX` in `partVerdicts`), so every part's
statement, which encodes it, keeps its length words under `2^64`. -/
def statementFits (limit : Nat) (h : InnerHeld) (o : Own) (pr : Proofs) (bs : Core.Blobs) : V Unit := do
  let n ← statementSize h o pr bs
  need (n < limit) s!"the statement the inputs name is {n} bytes, {limit} or more"

/-- Whether each parsed field's canonical bytes parse back to it (the draw up to `Flock.norm`, the inner shape through
`Inner.ofJson`): that `Names`' canonical pins agree with its parsed ones at these inputs. The difftest's, at the honest
session. -/
def roundTrips (h : InnerHeld) (o : Own) (pr : Proofs) (bs : Core.Blobs) : V Bool := do
  let back (j : Json) : V Bool := do pure ((Json.parse (← Flock.canon j)).toOption == some j)
  let shapeOk := (do Flock.RecVStar.Inner.ofJson (← Json.parse (← Flock.canon h.shape.toJson))).toOption == some h.shape
  let mut ok := shapeOk && (← back o.partition)
  let some tags0 := Flock.Tags.find? STATEMENT | throw s!"the statement {STATEMENT} is not known"
  for (p, s) in o.parts.zip pr.parts do
    let circuitB ← bs.get p.circuit "a part's circuit"
    let tags := Flock.Zk.tags (tags0.withCoins (Flock.Coins.tree (Flock.Zk.coinSpec (tags0.tableNames 1))))
      (Flock.Zk.maskLogOf circuitB) 1
    let record := (String.fromUTF8? (← bs.get s.record "a part's record")).getD ""
    let draw := if tags.hm96Rows then Flock.Witness.recordDraw record else none
    let drawOk ← match draw with
      | none => pure true
      | some d => do pure ((Json.parse (← Flock.canon d)).toOption.map Flock.norm == some d)
    let regOk ← match p.registered with
      | none => pure true
      | some r => back r
    ok := ok && drawOk && regOk && (← back p.partition) && (← back p.bindings)
  pure ok

/-- **The recursive session's verdict**: the inner session held (`innerHeld`), one part per V* statement at its shape
(`RecVStar.statements`), in order, and each part's verdict (`partVerdict`), and the statement the inputs name under
`FIELD_MAX` bytes (`statementFits`). Each part's verdict, by its statement's name, and the session's: accepted exactly when
every part's is. -/
def partVerdicts (o : Own) (pr : Proofs) (bs : Core.Blobs) : V (InnerHeld × Array (String × V Unit)) := do
  let h ← innerHeld o pr bs
  let ss ← Flock.RecVStar.statements h.shape
  need (dense h.shape) "recursive: the inner shape is not dense (private inner circuits, RecSparse or R)"
  need (o.parts.size == ss.size)
    s!"recursive: the verifier holds {o.parts.size} parts, and V* has {ss.size} statements at the inner shape"
  need (pr.parts.size == ss.size)
    s!"recursive: the evidence has {pr.parts.size} part sessions, and V* has {ss.size} statements at the inner shape"
  let vs ← partsAt h ss o pr bs (List.range ss.size)
  statementFits FIELD_MAX h o pr bs
  pure (h, vs.toArray)
where
  /-- The parts `js`' verdicts, in order: part `j`'s at V*'s statement `j`. -/
  partsAt (h : InnerHeld) (ss : Array VStar.Stmt) (o : Own) (pr : Proofs) (bs : Core.Blobs) :
      List Nat → V (List (String × V Unit))
    | [] => pure []
    | j :: js => match ss[j]?, o.parts[j]?, pr.parts[j]? with
      | some sj, some p, some s => do
        let v ← partVerdict h j p s bs
        pure ((sj.name, v) :: (← partsAt h ss o pr bs js))
      | _, _, _ => throw "recursive: a part out of range"

/-- The units the recursive session proved: the inner public file's unit indices (`units.indices`) of the instances the
inner record drew (`unit_draw.units`), ascending (`OneStageZk.provedUnits`). -/
def provedUnits (pubB : ByteArray) (record : String) : V (Array Nat) := OneStageZk.provedUnits pubB record

/-- The units an accepted recursive session proved (`provedUnits` at the inner public file and record), and none for a
refused one. -/
def unitsOf (ok : Bool) (o : Own) (pr : Proofs) (bs : Core.Blobs) : V (Array Nat) :=
  if ok then do
    provedUnits (← bs.get o.pub "the inner public file")
      ((String.fromUTF8? (← bs.get pr.record "the inner session's record")).getD "")
  else pure #[]

/-- A part's verdict as the audit's verdict reports it. -/
def partJson (x : String × V Unit) : Json :=
  Json.mkObj [("statement", Json.str x.1), ("accepted", Json.bool x.2.toBool),
    ("why", Json.str (match x.2 with | .ok () => "" | .error e => e))]

/-- The audit's verdict of record: accepted or not, the session, the first part's refusal, the units it proved, each
part's verdict and the inner program's digest. -/
def verdictJson (ok : Bool) (session why : String) (units : Array Nat) (parts : Array Json) (programSha : Option String) :
    Json :=
  Json.mkObj ([("accepted", Json.bool ok), ("of_record", Json.bool true), ("session", Json.str session),
    ("why", Json.str why), ("units", natsJson units), ("parts", Json.arr parts)] ++
    (match programSha with | some h => [("program_sha512", Json.str h)] | none => []))

theorem verdictJson_accepted (ok : Bool) (session why : String) (units : Array Nat) (parts : Array Json)
    (programSha : Option String) : getD (verdictJson ok session why units parts programSha) "accepted" = .bool ok := by
  cases programSha <;> rfl

/-- The audit's one verdict of record, `flock-verify/<session>`, with the units an accepted one proved and each part's
verdict. A refusal where the inputs refuse before any session, or where an accepted session's inner program is not the
partition object's, or its proved units are not the evidence's. -/
def verdicts (o : Own) (session : String) (pr : Proofs) (provedUnits? : Option (Array Nat)) (bs : Core.Blobs) :
    V (Array (String × Json)) := do
  let parts := (← partVerdicts o pr bs).2
  let ok := parts.all (·.2.toBool)
  let why := match parts.findIdx? (!·.2.toBool) with
    | some j => match parts[j]? with
      | some (nm, .error e) => s!"part {j} ({nm}): {e}"
      | _ => ""
    | none => ""
  let programSha := OneStageZk.programDigest o.program
  let want := getD o.partition "program"
  need (!(ok && (programSha.map Json.str) != some want))
    s!"session {session}'s inner program is {programSha.getD "none"}, not the partition object's {want.compress}"
  let units ← unitsOf ok o pr bs
  need (!(ok && provedUnits?.any (units.qsort (· < ·) != ·)))
    "the units the recursive session proved are not the evidence's proved_units"
  pure #[(s!"flock-verify/{session}", verdictJson ok session why units (parts.map partJson) programSha)]

def checkBlobs (p : Params) (c : OneStage.Commitment) (e : Evidence) (bs : Core.Blobs) :
    Core.Outcome OneStage.Public Core.Integrity :=
  match e.base.sessions[0]? with
  | none => .reject "proofs: a recursive audit is one recursive session"
  | some session =>
  match verdicts p.own session e.proofs (some e.base.provedUnits) bs with
  | .error err => .reject s!"proofs: {err}"
  | .ok vs => OneStage.check p.base c { e.base with verdicts := vs }

def protocol : Core.BlobProtocol where
  id := PROTOCOL
  Params := Params
  Commitment := OneStage.Commitment
  Evidence := Evidence
  Public := OneStage.Public
  Profile := Core.Integrity
  params := paramsOfJson
  commitment := OneStage.commitmentOf
  evidence := evidenceOf
  digest c := c.digest
  check p c e := checkBlobs p c e Core.Blobs.empty
  publicJson := OneStage.publicJson
  profileJson := Core.Integrity.toJson
  checkBlobs := checkBlobs

/-- `protocol.main`'s commands, and `verdicts OWN PROOFS [BLOB ...]`: the recursive session's part verdicts alone
(`partVerdicts`), from the verifier's own inputs (`ownOfJson`) and the evidence's `recursive` (`proofsOf`), each file
canonical JSON; one line, `{"parts": [{"statement", "accepted", "why"}, ...]}` in V*'s order or `{"refused": why}`, exit 0
when every part accepted, else 1. The difftest holds each part's to `flock-verify verify --zk` at that part's flags. -/
def main (args : List String) : IO UInt32 := do
  match args with
  | "verdicts" :: own :: proofs :: blobs =>
    let bs ← blobs.toArray.mapM fun (f : String) => IO.FS.readBinFile f
    let ownB ← IO.FS.readBinFile own
    let proofsB ← IO.FS.readBinFile proofs
    let r : V (Array (String × V Unit)) := do
      let pr ← proofsOf (← Core.canonicalJson proofsB)
      pure (← partVerdicts (← ownOfJson (← Core.canonicalJson ownB)) pr (Core.Blobs.ofArray bs)).2
    let line := match r with
      | .ok vs => Json.mkObj [("parts", Json.arr (vs.map fun (nm, v) => Json.mkObj [("statement", Json.str nm),
          ("accepted", Json.bool v.toBool), ("why", Json.str (match v with | .ok () => "" | .error e => e))]))]
      | .error e => Json.mkObj [("refused", Json.str e)]
    IO.println ((Flock.canon line).toOption.getD line.compress)
    return if (r.toOption.map fun vs => vs.all (·.2.toBool)).getD false then 0 else 1
  | "statement" :: limit :: own :: proofs :: blobs =>
    let bs := Core.Blobs.ofArray (← blobs.toArray.mapM fun (f : String) => IO.FS.readBinFile f)
    let ownB ← IO.FS.readBinFile own
    let proofsB ← IO.FS.readBinFile proofs
    let r : V Json := do
      let pr ← proofsOf (← Core.canonicalJson proofsB)
      let o ← ownOfJson (← Core.canonicalJson ownB)
      let h ← innerHeld o pr bs
      let fitsWhy := match statementFits limit.toNat! h o pr bs with
        | .ok () => ""
        | .error e => e
      pure (Json.mkObj [("size", Lean.toJson (← statementSize h o pr bs)), ("round_trips", Json.bool (← roundTrips h o pr bs)),
        ("fits", Json.str fitsWhy)])
    let line := match r with
      | .ok j => j
      | .error e => Json.mkObj [("refused", Json.str e)]
    IO.println ((Flock.canon line).toOption.getD line.compress)
    return if r.toOption.isSome then 0 else 1
  | _ => protocol.main args

#guard lineEnd "ab\ncd".toUTF8 0 == 2
#guard lineEnd "ab\ncd".toUTF8 3 == 5
#guard ((metaOf "c x\nMETA {\"a\":1}\nrest".toUTF8).toOption.map (·.compress)) == some "{\"a\":1}"
#guard (metaOf "1\n2\n3\n4\n5\n6\n7\n8\n9\nMETA {}".toUTF8).toOption.isNone
#guard ((metaOf "1\n2\n3\n4\n5\n6\n7\n8\nMETA {}".toUTF8).toOption.map (·.compress)) == some "{}"

end VStar.Recursive
