import Proofs.Flock.Claims.Recursion.DeployOuter
import Proofs.Flock.Claims.Recursion.Scope
import Proofs.Flock.Claims.Complete.Zk.Honest
import Proofs.Flock.Claims.Complete.Zk.Coins
import Definitions.Claims.Recursion

/-!
# The compiled verifier in public-coin form (`RecDeploy.V`) and its record (`deployRec`)

The shipped recursive verifier is the inner session, then `flock-verify verify --zk` once per V* part with that part's
recursive inputs (`--rec-inner`, `--rec-part J`, `--rec-bindings`, the inner statement's files, the inner session's
record and the coin server's transcript). `RecDeploy.V` is that as one `Claims.Verifier` over Flock's live coins
(`FlockDeploy.Coin`):

* **Statement.** `x` is the canonical bytes (`RecX.encode`) of what the verifier holds before the run: the inner
  statement's bytes (a `FlockDeploy.Raw` encoding: its circuit, public and partition files), the verifier's own inner
  shape (`--rec-inner`), the circuit registration, and per V* part its Flock statement and the verifier's own bindings
  of it (`--rec-bindings`, which the auditor computes from that part's partition object and program, so parts differ).
  V*'s shared rows (`--rec-rows`) are no part of `x`: they are staged (below). `parse` and `scopeCheck` are the
  deployed parser's (`Flock.RecScope.parseOf`, `scopeCheckOf`) at
  `decodeR`: what `readX` reads (it refuses unless the inner statement's dimensions read, `RecVStar.innerOf`, Flock's
  plain verifier sets the inner statement up, its statement check passes and it carries a draw, which at one table is
  the public file's header's, `innerOk`, and there is
  one part per V*
  statement of the verifier's inner shape, `RecVStar.statements`, each part's circuit V*'s statement's of its index,
  `partsVStar`, as each part's session holds it, `RecVStar.checkCircuit`), as the verifier's inner
  shape and the inner statement's files with no records (`readOf`), where the setup's checks accept that shape
  (`RecScope.accepts`). The V* records are run messages, so the statement carries no draws, and `scopeCheck` reads
  only the inner scope check: the compiled verifier refuses the draw and shape conditions at each session's decision
  (`DeployScope`). `parse` reads no circuit commitment: the shipped verifier has no private-circuit form of the
  compiled statement.
* **Rounds.** The inner session's `k_I = IV.k (x_I)` rounds, then one staging round (`kS`), each message answered by
  its live coin, then each V* session in order, part `s` at its V* statement (`vstar`), over `VO.k` rounds, each step
  the outer verifier's (`VO`, `verify --zk` at the recursive setup). A session that accepts at its last round is
  answered `.msg []`, the last one `.acc`; the compiled verifier refuses a session that accepts before its last round,
  and a session whose last round ended on a coin rather than a decision (`decided`).
* **Staging.** `vstar stage` (`rec_vstage.stage`) is the prover's: each part's public file holds the prover's
  registered entries (its running sums' chain, committed under its own salts), the inner session's record is the
  prover's, and so are V*'s shared rows (`--rec-rows`: `acc0`, `rec-top-L<l>` and `rec-c<k>`, registrations of the
  prover's values), one set for every part. So the prover sends them, after the inner rounds and before any V*
  session, as the staging round's message (`encStage`); the coin server's transcript of the inner rounds is the
  verifier's own (`Brings.tr`). The deployed composer (`flock-verify rec-compose`) accepts only where every part was
  checked against one staged rows file and one set of every other input the parts share, each part at its own index
  (`ComposeParts.composed_parts`: each part is then `partRaw`'s form, up to a SHA-512 collision). That the parts'
  sessions are one run of `V` is the open condition C10 `OneRun` (`ComposeParts`), which needs the service's
  interactive-run record.
* **V\* statements.** Part `s`'s statement is what the prover staged (`Stage.ofMsg`: the part's public file, the
  inner session's record and the shared rows off the staging message, the transcript the coin server's), in the part's
  recursive flags (`staged`). Its numbers (`numsOf`: `VO`'s rounds and `deployZkRec`'s errors, `N₀` and weight) are to be the ones `x`
  fixes (`nums`, those of a staged statement that sets up, when one does): the compiled verifier refuses a part whose
  staged statement has others, and `vstar` reads such a part as that statement (`vstarAt`), so every V* statement
  `vstar` gives has `nums`' numbers.

`next_rej`, `next_coin` and `next_sess` are the steps; `next_decide_sha512` is the acceptance theorem: at a session's
decision, the compiled verifier accepts the session exactly when it is the session's last round and the deployed
`Zk.verify` at the part's recursive setup accepts the record and proof files with the custody checks
(`FlockDeploy.checks`) and the verdict's last refusals (`fullDrawO`: `RecSetup.fullDraw`, `RecSetup.innerDrawn`), the part's statement
being the one the prover staged (`ract_sess`). `next_acc`: the whole run
accepts only at the last session's last round, where the outer verifier accepts.

`deployRec IV ivScope b` is the record the recursion's claims read, at the inner verifier `IV`, its scope check and
what the deployment brings (`Brings`): the coin server's transcript, the stager `stage` reads, the outer witnesses'
readers (`Readers`: the inner messages and the SHA-512 evaluations they record), each outer session's `--zk` leaf counts
and the reference prover's pieces. `innerView` runs `IV` at the claim's oracle on those messages and the run's first
`k_I` coins. The hm96 key is the pinned one (`Hm96.Default512`). The record's `stage` lays out the staging message
from the inner rounds and its commit strings (`Brings.hs`, `Brings.nC`), as the spec has it. The reference prover
(`refP`) is a composition: the native inner prover; that stager at the commit strings of its staged values' digests
under the staging salts on its tape (`commitThen`), so the record's `stage` is its own; then Flock's masked honest
prover `honestPZk` on each V* session, its witness read off the inner prover's tape and transcript. `OuterRef` names
what a faithful one needs beyond it. The record's completeness error (`δcRec`) is a union bound over the sessions:
the inner session's plain `δc` (`FlockComplete.δc`) plus each V* session's masked one at the recursive setup (`δcO`,
`FlockComplete.δcZk`'s bad coins at `setupO`), at the V* statements `nums` reads, and `1` wherever one of them cannot
be accepted (`readyRec`).
-/

namespace RecDeploy

open Flock
open FlockDeploy (Raw Coin Final Ins encNat encBytes encStr encOpt encList Rd rdTake rdNat rdBytes rdStr rdOpt rdList
  jsonOf checks tabsOf sha512)
open Lean (Json)

/-! ## The compiled verifier's statement -/

/-- A V* part as the compiled statement holds it: its Flock statement (its public file the prover's, staged) and the
verifier's own bindings of it (`--rec-bindings`, `verity_flock.rec_bindings` of the part's partition object and
program), which each part's session reads. Its shared rows (`--rec-rows`) are no part of it: they register the
prover's running sums, so the prover stages them, one set for every part (`Stage.rows`). -/
structure PartX where
  base : Raw
  bindings : ByteArray

/-- **What the compiled verifier holds before the run**: the inner statement's bytes, the verifier's own inner shape
(`--rec-inner`), the circuit registration (`--rec-inner-registration`), and the V* parts, each with its own bindings. -/
structure RecX where
  inner : Claims.Msg
  recInner : ByteArray
  registration : Option ByteArray
  parts : List PartX

/-- The compiled statement's format string, its first field. -/
def XFORMAT : String := "flock-verify-recursion/v3"

def PartX.encode (p : PartX) : Claims.Msg := encBytes ⟨p.base.encode.toArray⟩ ++ encBytes p.bindings

/-- **The compiled statement's canonical bytes.** -/
def RecX.encode (w : RecX) : Claims.Msg :=
  encStr XFORMAT ++ encNat w.inner.length ++ w.inner ++ encBytes w.recInner ++
    encOpt encBytes w.registration ++ encList PartX.encode w.parts

def PartX.read : Rd PartX := do
  let b ← rdBytes
  let some base := FlockDeploy.decode b.data.toList | failure
  let bindings ← rdBytes
  pure ⟨base, bindings⟩

def RecX.read : Rd RecX := do
  if (← rdStr) ≠ XFORMAT then failure
  let n ← rdNat
  let inner ← rdTake n
  let recInner ← rdBytes
  let registration ← rdOpt rdBytes
  let parts ← rdList PartX.read
  pure ⟨inner, recInner, registration, parts⟩

/-- **The compiled statement's decoder**: `RecX.read` on all of `x`, kept only when its encoding is `x`. -/
def decodeX (x : Claims.Msg) : Option RecX :=
  match RecX.read.run x with
  | some (w, []) => if w.encode = x then some w else none
  | _ => none

theorem decodeX_encode {x : Claims.Msg} {w : RecX} (h : decodeX x = some w) : w.encode = x := by
  unfold decodeX at h
  split at h
  · split_ifs at h with he
    cases h
    exact he
  · cases h

theorem decodeX_injective {x y : Claims.Msg} {w : RecX} (hx : decodeX x = some w) (hy : decodeX y = some w) : x = y :=
  (decodeX_encode hx).symm.trans (decodeX_encode hy)

/-! ## Its parser -/

/-- The compiled statement as read: its fields, the inner statement's, the verifier's own inner shape and the inner
statement's shape as `RecVStar.innerOf` reads it. -/
structure XR where
  w : RecX
  innerRaw : Raw
  partition : Option Json
  shape : RecVStar.Inner
  dims : RecVStar.Inner

/-- The draw `FlockDeploy.drawOfJ` reads off a record whose `unit_draw` is `u`, rebuilt (`Flock.norm`), or none where it
is absent or `null`. -/
def drawOfU : Option Json → Option Json
  | some .null => none
  | some d => some (Flock.norm d)
  | none => none

/-- **The session's draw is the statement's at one table**: at one table the setup reads the session's `unitDraw` back
from the public file's header, so this is where the two can differ; at several tables the setup sets it to the draw. -/
def drawTiedB (s : FlockDeploy.Ins) (spec : SessionSpec) (sts : Array Stmt) : Bool :=
  sts.size != 1 || FlockDeploy.drawEq (drawOfU spec.unitDraw) s.read.draw

theorem drawTiedB_iff {s : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} :
    drawTiedB s spec sts = true ↔ (sts.size = 1 → FlockDeploy.drawEq (drawOfU spec.unitDraw) s.read.draw = true) := by
  unfold drawTiedB
  by_cases h : sts.size = 1 <;> simp [h]

/-- **Flock's own checks on the inner statement**: Flock's plain verifier sets it up (`FlockDeploy.setupOf false`), its
statement check passes (`Readable.Read.check`, `ProvedScope.check`), it carries a draw, the inner session's record's
(`Raw.draw`): the recursion's inner session is drawn, and the compiled verifier refuses a session over an inner
session whose record holds none (`RecSetup.innerDrawn`, `noInnerDraw`), and at one table the public file's header
carries that draw (`drawTiedB`): the deployed verifier refuses a record whose draw is not the session's (`Draw.check`,
U1, in `Record.check`), and at one table the session's is the header's. -/
def innerOk (inner : Claims.Msg) : Except String Unit :=
  match FlockDeploy.setupOf false inner with
  | some (s, spec, sts) => do
    s.read.check
    if s.read.draw.isNone then throw RecSetup.noInnerDraw
    if !drawTiedB s spec sts then throw "U1: the record's unit draw is not the statement's"
  | none => .error "rec: Flock's verifier does not set the inner statement up"

/-- A part's circuit, as the setup parses it (`ProvedScope.circuitOf`) at its session's tags. -/
def partCircuit (p : PartX) : Option Circuit := do
  let ins ← p.base.ins.toOption
  (ProvedScope.circuitOf ins.read.tags p.base.circuit p.base.tables).toOption

/-- **Each part's circuit names its statement's message-round ports**: wherever part `s`'s V* statement registers a
port under a `rec-c<k>` (`RecVStar.ownPorts`, `RecVStar.cIndex`), the part's circuit, as the setup parses it
(`partCircuit`), has a port of that name. Each session refuses a part without: `checkOwn` holds the reads file's
message-round ports to the statement's, and `checkPubRows` finds each port registering a `rec-c<k>` among the
circuit's rows. -/
def portsNamed (parts : List PartX) (vs : Array VStar.Stmt) : Bool :=
  (List.range vs.size).all fun s =>
    let need := (RecVStar.ownPorts vs[s]!).filter fun q => (RecVStar.cIndex q.2.1).isSome
    match parts[s]?.bind partCircuit with
    | some c => need.all fun q => c.ports.any (·.name == q.1)
    | none => need.isEmpty

/-- **RecCap's part's circuit has its port `top`**: the last part's circuit, as the setup parses it (`partCircuit`), has
a port named `top`, RecCap's one input (`VStar.capSub`). RecCap's session refuses a part without: `checkFile` holds the
circuit's bytes to V*'s, and `checkParsed` its row ports to V*'s (`RecVStar.vstarRows`, `top` among them). -/
def capPorted (parts : List PartX) : Bool :=
  match parts[parts.length - 1]?.bind partCircuit with
  | some c => c.ports.any (·.name == "top")
  | none => false

/-- The verifier's own bindings of a part (`--rec-bindings`), as its session reads them (`RecRaw.ins`). -/
def ownBindings (b : ByteArray) : Option VStar.Bindings :=
  (do RecVStar.ownBindingsOfJson (← jsonOf "rec-bindings" b)).toOption

/-- **A part's circuit is V\*'s statement `sv`'s**, as its session's `RecVStar.checkCircuit` holds it: the circuit file
is V\*'s for `sv` under the verifier's own bindings `f` at the shape's holo flag (`checkFile`, byte for byte), and the
circuit the setup parses off it (`partCircuit`) carries V\*'s META and row ports (`checkParsed`), V\*'s part
(`checkPart`), `sv`'s name as its unit's (`RecVStar.check`) and V\*'s file's META `holo` (`checkHolo`). -/
def partVStar (shape : RecVStar.Inner) (f : VStar.Bindings) (sv : VStar.Stmt) (p : PartX) : Bool :=
  match partCircuit p with
  | some c => (RecVStar.checkFile sv f p.base.circuit shape.holo).isOk && (RecVStar.checkParsed sv c).isOk &&
      (RecVStar.checkPart sv c).isOk && c.unit.name == sv.name && (RecVStar.checkHolo sv c shape.holo).isOk
  | none => false

/-- **Each part's circuit is V\*'s statement's of its index** (`partVStar`), under the verifier's own bindings of that
part. -/
def partsVStar (shape : RecVStar.Inner) (parts : List PartX) (vs : Array VStar.Stmt) : Bool :=
  (List.range vs.size).all fun s =>
    match parts[s]?, vs[s]? with
    | some p, some sv => match ownBindings p.bindings with
      | some f => partVStar shape f sv p
      | none => false
    | _, _ => false

/-- The compiled statement's reading before the parts' circuits are held to V\*'s: its bytes decode, the inner
statement's decode and set up (`RecVStar.innerOf`) and pass Flock's own setup and statement check with a draw
(`innerOk`), the verifier's own inner shape reads (`RecVStar.Inner.ofJson`), and there is one part per V* statement of
that shape, whose circuit names the statement's message-round ports (`portsNamed`), the last part's RecCap's port
`top` (`capPorted`). -/
def readXPre (x : Claims.Msg) : Option XR := do
  let w ← decodeX x
  let ri ← FlockDeploy.decode w.inner
  let shape ← (do RecVStar.Inner.ofJson (← jsonOf "rec-inner" w.recInner)).toOption
  let partition ← (ri.partition.mapM (jsonOf "partition")).toOption
  let dims ← (do
    let dims ← RecVStar.innerOf ri.circuit ri.publicFile partition ri.program
    innerOk w.inner
    pure dims).toOption
  let vs ← (RecVStar.statements shape).toOption
  if w.parts.length ≠ vs.size ∨ portsNamed w.parts vs = false ∨ capPorted w.parts = false then none
  pure ⟨w, ri, partition, shape, dims⟩

/-- **The compiled statement's reader**: `readXPre`, where each part's circuit is V\*'s statement's of its index
(`partsVStar`). Each part's session refuses a part without (`RecSetup.checks` runs `RecVStar.checkStmt`, which runs
`checkCircuit`), so a statement every part's session sets up at passes (`readX_of_sessions`). -/
def readX (x : Claims.Msg) : Option XR := do
  let r ← readXPre x
  let vs ← (RecVStar.statements r.shape).toOption
  if partsVStar r.shape r.w.parts vs = false then none
  pure r

theorem readXPre_of_readX {x : Claims.Msg} {r : XR} (h : readX x = some r) : readXPre x = some r := by
  unfold readX at h
  cases hr : readXPre x with
  | none => simp [hr] at h
  | some r' =>
    cases hv : (RecVStar.statements r'.shape).toOption with
    | none => simp [hr, hv] at h
    | some vs =>
      by_cases hp : partsVStar r'.shape r'.w.parts vs = false
      · simp [hr, hv, hp] at h
      · simp [hr, hv, hp] at h
        rw [h]

theorem readX_parts {x : Claims.Msg} {r : XR} (h : readX x = some r) :
    ∃ vs, RecVStar.statements r.shape = .ok vs ∧ partsVStar r.shape r.w.parts vs = true := by
  unfold readX at h
  cases hr : readXPre x with
  | none => simp [hr] at h
  | some r' =>
    cases hv : RecVStar.statements r'.shape with
    | error e => simp [hr, hv, Except.toOption] at h
    | ok vs =>
      by_cases hp : partsVStar r'.shape r'.w.parts vs = false
      · simp [hr, hv, hp, Except.toOption] at h
      · simp [hr, hv, hp, Except.toOption] at h
        subst h
        exact ⟨vs, hv, by simpa using hp⟩

/-- **`readX` refuses only what `partsVStar` refuses**: where `readXPre` reads and every part's circuit is V\*'s
statement's of its index, `readX` reads the same. -/
theorem readX_of_pre {x : Claims.Msg} {r : XR} (h : readXPre x = some r) {vs : Array VStar.Stmt}
    (hvs : RecVStar.statements r.shape = .ok vs) (hp : partsVStar r.shape r.w.parts vs = true) :
    readX x = some r := by
  unfold readX
  simp [h, hvs, hp, Except.toOption]

/-- How Flock's plain verifier is told to set the inner statement `w` up, as each staged V* statement hands it to the
verifier (`--rec-inner-statement`, `--rec-inner-statement-coins`, `--rec-inner-tables`, `--rec-inner-draw`): its
`--statement`, `--coins`, lookup tables and draw. -/
def innerFormOf (w : Raw) : RecCoins.InnerForm := ⟨w.statement, w.osCoins, w.tables, w.draw⟩

/-- The inner statement's files as each staged V* statement hands them to check 5 (`--rec-inner-circuit`,
`--rec-inner-public`, `--rec-inner-partition`), with how the inner statement is set up (`innerFormOf`) and no record
(`RecCoins.Files.stmt` reads none); none without a partition, where no staged statement sets up. -/
def filesOf (r : XR) : Option RecCoins.Source :=
  r.partition.map fun p => .files ⟨r.innerRaw.circuit, r.innerRaw.publicFile, p, "", none, some (innerFormOf r.innerRaw)⟩

/-- **What the compiled verifier reads of the statement**, as `RecScope` reads it: the inner statement's bytes and
files, its own inner shape, and no V* session's record, each being a run message. -/
def readOf (r : XR) : RecScope.Read := ⟨r.w.inner, filesOf r, r.shape, []⟩

/-- The statement's reading, where `readX` reads it. -/
def decodeR (x : Claims.Msg) : Option RecScope.Read := (readX x).map readOf

/-- **The claims' parser**: the deployed parser (`RecScope.parseOf`) at `decodeR`: where the setup's checks accept the
inner shape (`RecScope.accepts`), the inner statement's bytes, its dimensions as check 5 sets it up, no circuit
commitment, the form off the verifier's own inner shape, and no draw (the V* records are run messages; their draws are
refused at the run, `DeployScope`). -/
def parse (x : Claims.Msg) : Option Claims.Recursion.Stmt := RecScope.parseOf decodeR x

/-- **The claims' scope check**: the deployed one (`RecScope.scopeCheckOf`) at `decodeR`, which reads only what the
statement carries: the setup's checks accept its inner shape and the inner scope check takes its inner statement. -/
def scopeCheck (ivScope : Claims.Msg → Bool) (x : Claims.Msg) : Bool := RecScope.scopeCheckOf ivScope decodeR x

/-- The inner statement's bytes. -/
def xIOf (x : Claims.Msg) : Claims.Msg := ((parse x).map Claims.Recursion.Stmt.inner).getD []

/-- The V* parts' count. -/
def nS (x : Claims.Msg) : ℕ :=
  match readX x with
  | some r => r.w.parts.length
  | none => 0

/-! ## The V* statements -/

/-- **The V* parts' staged inputs**, read off the compiled statement and the rounds before the sessions as the run
shows them: each part's public file, the inner session's record, the coin server's transcript, and V*'s shared rows
(`--rec-rows`), one set every part reads. -/
structure Stage where
  pub : Claims.Msg → List (Claims.Msg × Claims.Out) → ℕ → ByteArray
  record : Claims.Msg → List (Claims.Msg × Claims.Out) → ByteArray
  transcript : Claims.Msg → List (Claims.Msg × Claims.Out) → ByteArray
  rows : Claims.Msg → List (Claims.Msg × Claims.Out) → ByteArray

/-- The staging rounds: one, whose message holds every part's public file, the inner session's record and V*'s shared
rows. -/
def kS (_ : Claims.Msg) : ℕ := 1

/-- **The staging message**: each part's public file, then the inner session's record, then V*'s shared rows. -/
def encStage (pubs : List ByteArray) (record rows : ByteArray) : Claims.Msg :=
  encList encBytes pubs ++ encBytes record ++ encBytes rows

def readStage : Rd (List ByteArray × ByteArray × ByteArray) := do
  let pubs ← rdList rdBytes
  let record ← rdBytes
  let rows ← rdBytes
  pure (pubs, record, rows)

/-- The staging message read back, on all of it. -/
def decodeStage (m : Claims.Msg) : Option (List ByteArray × ByteArray × ByteArray) :=
  match readStage.run m with
  | some (p, []) => some p
  | _ => none

/-- The staging message of rounds `ms`: their last message. -/
def stageOf (ms : List (Claims.Msg × Claims.Out)) : Option (List ByteArray × ByteArray × ByteArray) :=
  decodeStage ((ms.getLast?.map Prod.fst).getD [])

/-- **The staged inputs as the compiled verifier reads them** off the rounds `ms` before the sessions: each part's
public file, the inner session's record and V*'s shared rows off the staging message, `ms`' last, and the coin
server's transcript `tr` of the inner rounds before it; empty where the staging message doesn't read. -/
def Stage.ofMsg (tr : Claims.Msg → List (Claims.Msg × Claims.Out) → ByteArray) : Stage where
  pub _ ms s := ((stageOf ms).bind fun p => p.1[s]?).getD ByteArray.empty
  record _ ms := ((stageOf ms).map (·.2.1)).getD ByteArray.empty
  transcript x ms := tr x ms.dropLast
  rows _ ms := ((stageOf ms).map (·.2.2)).getD ByteArray.empty

/-- Part `s`'s recursive statement: its Flock statement at the staged public file, the verifier's own inner shape,
`--rec-part s`, the part's bindings, the staged shared rows, the inner statement's files, the staged record and
transcript, the registration, and how the inner statement is set up (`innerFormOf`). -/
def partRaw (r : XR) (st : Stage) (x : Claims.Msg) (ms : List (Claims.Msg × Claims.Out)) (s : ℕ) (p : PartX) :
    RecRaw where
  base := { p.base with publicFile := st.pub x ms s }
  recInner := some r.w.recInner
  recPart := some s
  recBindings := some p.bindings
  recRows := some (st.rows x ms)
  innerCircuit := some r.innerRaw.circuit
  innerPublic := some r.innerRaw.publicFile
  innerPartition := r.innerRaw.partition
  innerCoins := none
  innerSession := some (st.record x ms)
  transcript := some (st.transcript x ms)
  registration := r.w.registration
  innerStatement := some r.innerRaw.statement
  innerStatementCoins := some r.innerRaw.osCoins
  innerTables := some r.innerRaw.tables
  innerDraw := r.innerRaw.draw

open Classical in
/-- **Part `s`'s statement as staged** at the rounds `ms` before the sessions: its encoding, where that reads back as
it (`decodeRec`), as every statement short of `2^64`-byte fields does; else none. -/
noncomputable def staged (st : Stage) (x : Claims.Msg) (ms : List (Claims.Msg × Claims.Out)) (s : ℕ) : Claims.Msg :=
  match readX x with
  | some r =>
    match r.w.parts[s]? with
    | some p =>
      if decodeRec (partRaw r st x ms s p).encode = some (partRaw r st x ms s p) then (partRaw r st x ms s p).encode
      else []
    | none => []
  | none => []

/-- A V* statement's numbers as the compiled verifier checks them: `VO`'s rounds, and `deployZkRec`'s table error,
`N₀`, weight, waits and truncation. No check reads the `--zk` leaf counts, left `0` here; the record's are
`numsZk`'s. -/
noncomputable def numsOf (xO : Claims.Msg) : Claims.Recursion.Nums :=
  ⟨εtableO xO, N₀O xO, weightO xO, VO.k xO, Claims.Flock.Numbers.waits, Claims.Flock.Numbers.cap, 0, 0⟩

open Classical in
/-- Inner rounds at which part `s`'s staged statement sets up, when there are any. -/
noncomputable def pick (st : Stage) (x : Claims.Msg) (s : ℕ) : List (Claims.Msg × Claims.Out) :=
  if h : ∃ ms, (setupO (staged st x ms s)).isSome then Classical.choose h else []

/-- **Part `s`'s numbers at `x`**: those of its staged statement at inner rounds where it sets up, when there are
any. -/
noncomputable def nums (st : Stage) (x : Claims.Msg) (s : ℕ) : Claims.Recursion.Nums :=
  numsOf (staged st x (pick st x s) s)

open Classical in
/-- **Part `s`'s V* statement**: the staged one where its numbers are `nums`', else the one at `pick`. -/
noncomputable def vstarAt (st : Stage) (x : Claims.Msg) (ms : List (Claims.Msg × Claims.Out)) (s : ℕ) : Claims.Msg :=
  if numsOf (staged st x ms s) = nums st x s then staged st x ms s else staged st x (pick st x s) s

/-- **The V* statements of a run**, read off its inner rounds. -/
noncomputable def vstar (st : Stage) (x : Claims.Msg) (ms : List (Claims.Msg × Claims.Out)) : List Claims.Msg :=
  (List.range (nS x)).map (vstarAt st x ms)

/-- **Part `s`'s numbers in the record**: `nums`', with the outer session's `--zk` leaf counts at its V* statement
(`zl`: Flock's `n_L`, the leaves before the coins, and `n_B`, those of one draw's answers). -/
noncomputable def numsZk (zl : Claims.Msg → ℕ × ℕ) (st : Stage) (x : Claims.Msg) (s : ℕ) : Claims.Recursion.Nums :=
  { nums st x s with
    leaves := (zl (staged st x (pick st x s) s)).1
    coinLeaves := (zl (staged st x (pick st x s) s)).2 }

theorem numsOf_vstarAt (st : Stage) (x : Claims.Msg) (ms : List (Claims.Msg × Claims.Out)) (s : ℕ) :
    numsOf (vstarAt st x ms s) = nums st x s := by
  unfold vstarAt
  split_ifs with h
  · exact h
  · rfl

theorem vstar_getD (st : Stage) (x : Claims.Msg) (ms : List (Claims.Msg × Claims.Out)) {s : ℕ} (hs : s < nS x) :
    (vstar st x ms).getD s [] = vstarAt st x ms s := by
  simp [vstar, List.getD_eq_getElem?_getD, hs]

theorem vstar_length (st : Stage) (x : Claims.Msg) (ms : List (Claims.Msg × Claims.Out)) :
    (vstar st x ms).length = nS x := by
  simp [vstar]

/-- **Every V* statement's rounds are its numbers'.** -/
theorem vstar_map_k (st : Stage) (x : Claims.Msg) (ms : List (Claims.Msg × Claims.Out)) :
    (vstar st x ms).map VO.k = (List.range (nS x)).map fun s => (nums st x s).rounds := by
  unfold vstar
  rw [List.map_map]
  congr 1
  funext s
  exact congrArg Claims.Recursion.Nums.rounds (numsOf_vstarAt st x ms s)

/-! ## The verifier -/

/-- The inner rounds as a run shows them: each message with its coin's bytes. -/
def innerPairs (kI : ℕ) (as : List Claims.Msg) (cs : List Coin) : List (Claims.Msg × Claims.Out) :=
  (as.take kI).zip ((cs.take kI).map fun c => .msg c.bytes)

/-- The session of round `i` past the inner rounds, at the sessions' rounds `ks`: its index and its round in it. -/
def locate : List ℕ → ℕ → Option (ℕ × ℕ)
  | [], _ => none
  | k :: ks, i => if i < k then some (0, i) else (locate ks (i - k)).map fun p => (p.1 + 1, p.2)

theorem locate_lt : ∀ {ks : List ℕ} {i s j : ℕ}, locate ks i = some (s, j) → s < ks.length ∧ j < ks.getD s 0
  | [], _, _, _, h => by simp [locate] at h
  | k :: ks, i, s, j, h => by
    unfold locate at h
    split_ifs at h with hi
    · cases h
      simpa using hi
    · obtain ⟨⟨s', j'⟩, hl, he⟩ := Option.map_eq_some_iff.1 h
      cases he
      obtain ⟨h1, h2⟩ := locate_lt hl
      exact ⟨by simpa using h1, by simpa using h2⟩

/-- Whether a session's rounds `as`, `cs` at `xO` ended on a decision rather than a coin. -/
def decided (xO : Claims.Msg) (as : List Claims.Msg) (cs : List Coin) : Bool :=
  match actO xO as cs with
  | .decide .. => true
  | _ => false

/-- What the compiled verifier does at the prover's messages: refuse, answer an inner round's coin, or take session
`s`'s round `j` at its V* statement `xO`, on the session's messages and coins. -/
inductive RAct where
  | rej
  | coin (c : Coin)
  | sess (s j : ℕ) (xO : Claims.Msg) (as : List Claims.Msg) (cs : List Coin)

open Classical in
/-- **The compiled verifier's step** at the prover's messages `as` and the coins `cs`: a statement that doesn't parse,
refuse; an inner or staging round, answer its coin; past them, the round's V* session, refused where the part's staged
numbers are not `x`'s, or, at its first round, where the session before it ended on a coin. -/
noncomputable def ract (IV : Claims.Verifier Coin) (st : Stage) (x : Claims.Msg) (as : List Claims.Msg)
    (cs : List Coin) : RAct :=
  match readX x with
  | none => .rej
  | some _ =>
    if as.length ≤ IV.k (xIOf x) + kS x then
      match cs.getLast? with
      | some c => .coin c
      | none => .rej
    else
      let kI := IV.k (xIOf x) + kS x
      let ms := innerPairs kI as cs
      let xs := vstar st x ms
      let ks := xs.map VO.k
      match locate ks (as.length - 1 - kI) with
      | none => .rej
      | some (s, j) =>
        let off := as.length - 1 - j
        let off' := off - ks.getD (s - 1) 0
        if numsOf (staged st x ms s) ≠ nums st x s then .rej
        else if j = 0 ∧ 0 < s ∧
            decided (xs.getD (s - 1) []) ((as.take off).drop off') ((cs.take off).drop off') = false then .rej
        else .sess s j (xs.getD s []) (as.drop off) (cs.drop off)

/-- A session's answer as the compiled verifier gives it: a message as is; an acceptance at the session's last round
`.acc` at the last session, else `.msg []`, and before its last round a refusal; a refusal as is. -/
def sessOut (last atEnd : Bool) : Claims.Out → Claims.Out
  | .msg m => .msg m
  | .acc => if atEnd then (if last then .acc else .msg []) else .rej
  | .rej => .rej

/-- The compiled verifier's next message. -/
noncomputable def next (IV : Claims.Verifier Coin) (st : Stage) (x : Claims.Msg) (as : List Claims.Msg)
    (cs : List Coin) : Claims.OC Claims.Out :=
  match ract IV st x as cs with
  | .rej => .ret .rej
  | .coin c => .ret (.msg c.bytes)
  | .sess s j xO as' cs' =>
    (nextO xO as' cs').bind fun o => .ret (sessOut (s + 1 == nS x) (j + 1 == VO.k xO) o)

/-- **The compiled verifier in public-coin form**: `k(x)` is the inner rounds, the staging round and every V*
session's rounds at `x`'s numbers. -/
noncomputable def V (IV : Claims.Verifier Coin) (st : Stage) : Claims.Verifier Coin where
  k x := IV.k (xIOf x) + kS x + ((List.range (nS x)).map fun s => (nums st x s).rounds).sum
  next := next IV st

/-! ## What the record reads off a run -/

/-- An accepted session's answer, `.msg []` before the last session, as its verifier's `.acc`. -/
def fixLast (last : Bool) (k : ℕ) (seg : List (Claims.Msg × Claims.Out)) : List (Claims.Msg × Claims.Out) :=
  if seg.length == k && !last then
    match seg.getLast? with
    | some (a, .msg []) => seg.dropLast ++ [(a, .acc)]
    | _ => seg
  else seg

/-- A session's decision, off its last answer. -/
def segOut (seg : List (Claims.Msg × Claims.Out)) : Claims.Out :=
  match seg.getLast? with
  | some (_, .acc) => .acc
  | _ => .rej

/-- **Session `s`'s view in a run's view**: its rounds' messages, after the inner and staging rounds and the earlier
sessions' rounds at the V* statements read off the view's inner and staging rounds, its acceptance shown as its
verifier's; accepted where the run is, else its last answer's decision. -/
noncomputable def session (IV : Claims.Verifier Coin) (st : Stage) (x : Claims.Msg) (s : ℕ) (v : Claims.Seen) :
    Claims.Seen :=
  let kI := IV.k (xIOf x) + kS x
  let xs := vstar st x (v.msgs.take kI)
  let off := kI + ((xs.take s).map VO.k).sum
  let k := VO.k (xs.getD s [])
  let seg := fixLast (s + 1 == nS x) k ((v.msgs.drop off).take k)
  ⟨seg, if v.out = .acc then .acc else segOut seg⟩

/-- What the outer witnesses give: the inner session's messages, and the SHA-512 evaluations session `s`'s witness
records. -/
structure Readers where
  msgs : Claims.Msg → List Claims.Flock.Wit → List Claims.Msg
  calls : Claims.Msg → ℕ → Claims.Flock.Wit → List (Claims.Msg × Claims.Msg)

/-- **The inner session's view**: `IV`'s run at `H` of the inner messages the outer witnesses give, at the run's first
`k_I` coins. -/
def innerView (H : Claims.Msg → Claims.Msg) (IV : Claims.Verifier Coin) (rd : Readers) (x : Claims.Msg) {k : ℕ}
    (r : Fin k → Coin) (ws : List Claims.Flock.Wit) : Claims.Seen :=
  (Claims.run H (Claims.replay (rd.msgs x ws)) IV (xIOf x) () fun i => Claims.coinSeq r i.val).seen

/-- **The inputs of the SHA-512 evaluations the inner verifier makes on the inner messages the outer witnesses give**:
its run's calls at `H`, at the coins `c` (a draw's, whose first are the inner session's). -/
def innerCallsOf (H : Claims.Msg → Claims.Msg) (IV : Claims.Verifier Coin) (rd : Readers) (x : Claims.Msg)
    (c : ℕ → Coin) (ws : List Claims.Flock.Wit) : List Claims.Msg :=
  (Claims.run H (Claims.replay (rd.msgs x ws)) IV (xIOf x) () fun i => c i.val).vLog.map Prod.fst

/-! ## The reference prover -/

/-- **The reference prover's pieces the inner protocol brings**: the native inner prover on its tapes; the digests of
the values its staging message commits (`vstar stage`'s registered entries: each part's running sums' chain), read
off the inner witness, the inner prover's tape and the inner rounds' history; and the outer witnesses it gives, session
`s`'s read off the same. -/
structure RefParts where
  TI : Type
  [ne : Nonempty TI]
  PI : Claims.Flock.Wit → Claims.Prover TI
  digests : Claims.Flock.Wit → TI → List (Claims.Msg × Claims.Msg) → List (Fin 64 → UInt8)
  wit : Claims.Flock.Wit → TI → List (Claims.Msg × Claims.Msg) → ℕ → Claims.Flock.Wit

attribute [instance] RefParts.ne

/-- **A stager**, `vstar stage`'s layout: each part's public file, the inner session's record and V*'s shared rows, at
the statement, the inner rounds and the hm96 commit strings the staging message holds. -/
structure Stager where
  pub : Claims.Msg → List (Claims.Msg × Claims.Out) → List Claims.Msg → ℕ → ByteArray
  record : Claims.Msg → List (Claims.Msg × Claims.Out) → List Claims.Msg → ByteArray
  rows : Claims.Msg → List (Claims.Msg × Claims.Out) → List Claims.Msg → ByteArray

/-- **What the deployment brings** besides the inner verifier: the coin server's transcript of the inner rounds (`tr`),
the stager the record's `stage` reads (`hs`) and the number of commit strings it lays out (`nC`), the outer witnesses'
readers (`rd`), each V* statement's `--zk` leaf counts (`zl`: Flock's `n_L` and `n_B`, which
`Flock.zk_deployZk_of_models` takes as its parameters), the reference prover's pieces (`ref`), and the inputs of the
SHA-512 evaluations the V* setup makes outside the oracle at each V* statement (`setup`, which `Br` reads). -/
structure Brings where
  tr : Claims.Msg → List (Claims.Msg × Claims.Out) → ByteArray
  hs : Stager
  nC : Claims.Msg → ℕ
  rd : Readers
  zl : Claims.Msg → ℕ × ℕ
  ref : RefParts
  setup : Claims.Msg → List Claims.Msg := fun _ => []

/-- The staged inputs the compiled verifier reads: the staging message's, at the coin server's transcript. -/
def Brings.st (b : Brings) : Stage := Stage.ofMsg b.tr

/-- **The staging message of a stager** at the inner rounds `ms` and the commit strings `cs`: its public file of each
part, its inner session record, and its shared rows. -/
def stageMsg (hs : Stager) (x : Claims.Msg) (ms : List (Claims.Msg × Claims.Out)) (cs : List Claims.Msg) :
    Claims.Msg :=
  encStage ((List.range (nS x)).map (hs.pub x ms cs)) (hs.record x ms cs) (hs.rows x ms cs)

/-- A salt at the pinned hm96 key. -/
abbrev Salt : Type := Fin Hm96.sha512.saltBytes → UInt8

/-- **The pinned key's salt mask** `Φy`, as 64 bytes. -/
def maskOf (y : Salt) (i : Fin 64) : UInt8 := (Hm96.Default512.mask ⟨Array.ofFn y⟩).get! i.val

/-- **Commit strings, then `k` at them**: for each digest `d` and salt `y` of `ps`, in order, hm96's `b ‖ c` at the
pinned key, `b = d ⊕ Φy` and `c = H(p_salt ‖ y)` the oracle's answer, appended to `cs`. -/
def commitThen (k : List Claims.Msg → Claims.OC Claims.Msg) :
    List ((Fin 64 → UInt8) × Salt) → List Claims.Msg → Claims.OC Claims.Msg
  | [], cs => k cs
  | (d, y) :: ps, cs => .query (Claims.toMsg Hm96.Default512.saltPrefix ++ List.ofFn y) fun c =>
      commitThen k ps (cs ++ [List.ofFn (fun i => d i ^^^ maskOf y i) ++ c])

/-- A prover's history as the run shows its rounds, each answer a message. -/
def asRounds (hist : List (Claims.Msg × Claims.Msg)) : List (Claims.Msg × Claims.Out) :=
  hist.map fun p => (p.1, .msg p.2)

/-- The reference prover's tapes: the inner prover's, the staging salts, and the masked prover's of each V* session. -/
abbrev Tape (b : Brings) : Type := b.ref.TI × (ℕ → Salt) × (ℕ → FlockComplete.TapeZk)

/-- **The reference prover, a composition**: the native inner prover over the inner rounds at the inner statement;
the record's stager (`Brings.hs`) at the commit strings of its staged values' digests (`RefParts.digests`) under the
staging salts on its tape; then on each V* session Flock's masked honest prover `honestPZk` at the session's V*
statement, its witness the one the inner prover's output gives (`RefParts.wit`), its history the session's rounds. -/
noncomputable def refP (IV : Claims.Verifier Coin) (b : Brings) (w : Claims.Flock.Wit) : Claims.Prover (Tape b) :=
  ⟨fun x hist ω =>
    let kI := IV.k (xIOf x)
    if hist.length < kI then (b.ref.PI w).next (xIOf x) hist ω.1
    else if hist.length < kI + kS x then
      let ds := b.ref.digests w ω.1 (hist.take kI)
      commitThen (fun cs => .ret (stageMsg b.hs x (asRounds (hist.take kI)) cs))
        ((List.range (b.nC x)).map fun j => (ds.getD j 0, ω.2.1 j)) []
    else
      let xs := vstar b.st x (asRounds (hist.take (kI + kS x)))
      match locate (xs.map VO.k) (hist.length - (kI + kS x)) with
      | some (s, j) =>
        (FlockComplete.honestPZk (b.ref.wit w ω.1 (hist.take kI) s)).next (xs.getD s [])
          (hist.drop (hist.length - j)) (ω.2.2 s)
      | none => .ret []⟩

/-- **What a faithful reference prover needs beyond the composition**: `honestPZk` is the deployed
`flock-circuit --zk` prover of a plain statement (`FlockDeploy.V true`'s, where `Assembled` takes its completeness as
`hComplete`); on a V* session it runs at the recursive outer record `deployZkRec`, whose statements set up at the
recursive setup (`setupO`), and is to be complete there, with error `δ`. -/
def OuterRef (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) (δ : Claims.Msg → ℝ) : Prop :=
  FlockComplete.CompleteWith H (deployZkRec rel) FlockComplete.honestPZk δ

/-! ## The completeness error -/

/-- **A V\* session's zerocheck outer coins**: the masked verifier's (`FlockComplete.outerSlotsZk`) at the recursive
setup. -/
def outerSlotsO (x : Claims.Msg) : List (ℕ × ℕ) :=
  match setupO x with
  | some (_, spec, sts) => (List.range (2 * sts.size)).flatMap fun q =>
      (List.range (sts[q / 2]!.m - 13)).map fun i => (FlockComplete.repStartZk spec sts q + 1, i)
  | none => []

/-- **A V\* session's level-0 pair coins** (`FlockComplete.rhoSlotsZk` at the recursive setup). -/
def rhoSlotsO (x : Claims.Msg) : List ℕ :=
  match setupO x with
  | some (_, spec, sts) => (List.range (2 * sts.size)).flatMap fun q =>
      match Zk.schedule sts[q / 2]!.m with
      | .ok (sch, _) =>
        [FlockComplete.repStartZk spec sts q + FlockComplete.piopRoundsZk spec sts[q / 2]! + (sch.k0 + 2)]
      | .error _ => []
  | none => []

/-- **A V\* session's inner `β`s** (`FlockComplete.betaSlotsZk` at the recursive setup). -/
def betaSlotsO (x : Claims.Msg) : List ℕ :=
  match setupO x with
  | some (_, spec, sts) => (List.range (2 * sts.size)).flatMap fun q =>
      match Zk.schedule sts[q / 2]!.m with
      | .ok (sch, _) =>
        [FlockComplete.repStartZk spec sts q + FlockComplete.piopRoundsZk spec sts[q / 2]! +
          (sch.k0 + 7 + 11 * (sch.levels.size - 1)) + 2]
      | .error _ => []
  | none => []

/-- `εO`: a V\* session's outer coins, `β`s and twice its pairs, over `2^128` (`FlockComplete.εZk`). -/
noncomputable def εO (x : Claims.Msg) : ℝ :=
  ((outerSlotsO x).length + (betaSlotsO x).length + 2 * (rhoSlotsO x).length) / 2 ^ 128

open Classical in
/-- **The V\* statements the masked verifier can accept** against `H` (`FlockComplete.readyZk` at the recursive setup):
one table, its setup against `H` the one at SHA-512, a drawn session, under `2^56` bytes, and the keyed coin tree. -/
noncomputable def readyO (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) : Bool :=
  match setupO x with
  | some (s, spec, sts) => sts.size == 1 && s.base.read.draw.isSome && decide (x.length < 2 ^ 56) &&
      decide (FlockDeploy.setupWith (Claims.hOf H) s.base.read = .ok (spec, sts)) && spec.coinTree.isSome
  | none => false

/-- **A V\* session's masked `δc`** against `H`: `εO` where the masked verifier can accept at the recursive setup, else
`1` (`FlockComplete.δcZk`). -/
noncomputable def δcO (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) : ℝ := if readyO H x then εO x else 1

open Classical in
/-- **The statements the compiled verifier can accept** against `H`: the inner session's plain verifier can accept the
inner statement (`FlockComplete.ready`), and the masked verifier each part's V\* statement as `nums` reads it, the
staged one at `pick`'s inner rounds (`readyO`). -/
noncomputable def readyRec (st : Stage) (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) : Bool :=
  FlockComplete.ready H (xIOf x) && decide (∀ s < nS x, readyO H (staged st x (pick st x s) s) = true)

/-- **The compiled verifier's `δc`** against `H`: where it can accept (`readyRec`), the inner session's plain `δc`
(`FlockComplete.δc`) plus each V\* session's masked one (`δcO`), a union bound over the sessions; else `1`. -/
noncomputable def δcRec (st : Stage) (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) : ℝ :=
  if readyRec st H x then
    FlockComplete.δc H (xIOf x) + ∑ s ∈ Finset.range (nS x), δcO H (staged st x (pick st x s) s)
  else 1

/-! ## The record -/

/-- **The inputs of the SHA-512 evaluations the plain inner verifier's setup makes at the inner statement `xI`**: the
calls its run logs at SHA-512, at the statement's own setup (`FlockDeploy.setupOf`), and none where it sets up none.
A function of the statement alone, which no deployment choice moves. -/
noncomputable def innerSetupIns (xI : Claims.Msg) : List Claims.Msg :=
  match FlockDeploy.setupOf false xI with
  | some (s, _, _) => ((FlockDeploy.setupM Claims.Hashes.oc s.read).run.eval FlockDeploy.sha512).2.map Prod.fst
  | none => []

/-- **The compiled verifier**, as the record the recursion's claims read: the inner verifier `IV` and its scope
check, and what the deployment brings (`b`). The inner view runs `IV` at the claim's oracle; the hm96 key is the pinned
one (`Hm96.Default512`: its salt prefix, and its salt mask `Φ` as 64 bytes). Its completeness error is `δcRec`, which
reads the inner session as the deployed plain verifier's (`FlockComplete.δc`), the one the claims instantiate `IV`
with; its inner setup's calls are the plain setup's at SHA-512 (`innerSetupIns`), and its inner verifier's calls on the
inner messages `IV`'s run's at the coins it is handed (`innerCallsOf`). -/
noncomputable def deployRec (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (b : Brings) :
    Claims.Recursion.Deploy Coin where
  V := V IV b.st
  parse := parse
  scopeCheck := scopeCheck ivScope
  kS := kS
  nC := b.nC
  stage x ms cs := .ret (stageMsg b.hs x ms cs)
  vstar := vstar b.st
  nS := nS
  session := session IV b.st
  innerView H x r ws := innerView H IV b.rd x r ws
  calls x ws := (ws.mapIdx fun s w => b.rd.calls x s w).flatten
  nums := numsZk b.zl b.st
  saltLen := Hm96.sha512.saltBytes
  saltPrefix := Claims.toMsg Hm96.Default512.saltPrefix
  mask y i := (Hm96.Default512.mask ⟨Array.ofFn y⟩).get! i.val
  Tape := Tape b
  P := refP IV b
  δc := δcRec b.st
  setup := b.setup
  innerSetup := innerSetupIns
  innerCalls H := innerCallsOf H IV b.rd

theorem xI_deployRec (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (b : Brings) (x : Claims.Msg) :
    Claims.Recursion.xI (deployRec IV ivScope b) x = xIOf x := rfl

/-! ## The verifier's steps against `H` -/

section Steps

variable {IV : Claims.Verifier Coin} {st : Stage} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}

theorem next_rej (H : Claims.Msg → Claims.Msg) (h : ract IV st x as cs = .rej) :
    (next IV st x as cs).eval H = (.rej, []) := by
  simp only [next, h, Claims.OC.eval]

/-- **An inner or staging round is answered by its coin, with no query.** -/
theorem next_coin (H : Claims.Msg → Claims.Msg) {c : Coin} (h : ract IV st x as cs = .coin c) :
    (next IV st x as cs).eval H = (.msg c.bytes, []) := by
  simp only [next, h, Claims.OC.eval]

/-- **A session's round is the outer verifier's**, its answer as `sessOut` gives it and its oracle log the outer
verifier's. -/
theorem next_sess (H : Claims.Msg → Claims.Msg) {s j : ℕ} {xO : Claims.Msg} {as' : List Claims.Msg} {cs' : List Coin}
    (h : ract IV st x as cs = .sess s j xO as' cs') :
    (next IV st x as cs).eval H =
      (sessOut (s + 1 == nS x) (j + 1 == VO.k xO) ((nextO xO as' cs').eval H).1, ((nextO xO as' cs').eval H).2) := by
  simp only [next, h, Claims.OC.eval_bind, Claims.OC.eval, List.append_nil]

/-- **A session's round is part `s`'s, at its staged statement**: `s` is a part, `j` a round of its outer verifier,
and the V* statement is the one the prover staged, read off the inner and staging rounds. -/
theorem ract_sess {s j : ℕ} {xO : Claims.Msg} {as' : List Claims.Msg} {cs' : List Coin}
    (h : ract IV st x as cs = .sess s j xO as' cs') :
    s < nS x ∧ j < VO.k xO ∧ xO = staged st x (innerPairs (IV.k (xIOf x) + kS x) as cs) s := by
  unfold ract at h
  split at h
  · cases h
  split_ifs at h with hk
  · split at h <;> cases h
  dsimp only at h
  split at h
  · cases h
  rename_i s' j' hl
  split_ifs at h with hn hd
  obtain ⟨rfl, rfl, rfl, rfl, rfl⟩ := RAct.sess.inj h
  obtain ⟨hs, hj⟩ := locate_lt hl
  rw [List.length_map, vstar_length] at hs
  have he : (vstar st x (innerPairs (IV.k (xIOf x) + kS x) as cs))[s']? =
      some (vstarAt st x (innerPairs (IV.k (xIOf x) + kS x) as cs) s') := by
    simp [vstar, hs]
  refine ⟨hs, ?_, ?_⟩
  · rw [List.getD_eq_getElem?_getD, List.getElem?_map, he] at hj
    rw [List.getD_eq_getElem?_getD, he]
    simpa using hj
  · rw [vstar_getD st x _ hs]
    simp [vstarAt, not_not.1 hn]

/-- **The compiled verifier accepts a run only at the last session's last round**, where the outer verifier accepts. -/
theorem next_acc (H : Claims.Msg → Claims.Msg) (h : ((next IV st x as cs).eval H).1 = .acc) :
    ∃ s j xO as' cs', ract IV st x as cs = .sess s j xO as' cs' ∧ s + 1 = nS x ∧ j + 1 = VO.k xO ∧
      ((nextO xO as' cs').eval H).1 = .acc := by
  cases hr : ract IV st x as cs with
  | rej => rw [next_rej H hr] at h; cases h
  | coin c => rw [next_coin H hr] at h; cases h
  | sess s j xO as' cs' =>
    rw [next_sess H hr] at h
    refine ⟨s, j, xO, as', cs', rfl, ?_⟩
    simp only at h
    cases ho : ((nextO xO as' cs').eval H).1 with
    | msg m => rw [ho] at h; cases h
    | rej => rw [ho] at h; cases h
    | acc =>
      rw [ho] at h
      simp only [sessOut] at h
      split_ifs at h with h1 h2
      exact ⟨by simpa using h2, by simpa using h1, rfl⟩

/-- **The acceptance theorem: at SHA-512 a session's decision is the shipped recursive check's.** Where session `s`'s
outer verifier decides, the compiled verifier accepts the session (`.acc` at the last session, `.msg []` before it)
exactly when this is the session's last round and the deployed `flock-verify verify --zk` with part `s`'s recursive
inputs, `Zk.verify` at the statement's recursive setup (`actO_decide`, `RecIns.setup`), accepts the record and proof
files with the custody checks and the verdict's last refusals; else it refuses. The statement is the part's as the prover
staged it (`ract_sess`). -/
theorem next_decide_sha512 {s j : ℕ} {xO : Claims.Msg} {as' : List Claims.Msg} {cs' : List Coin} {i : Ins}
    {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : Final} (h : ract IV st x as cs = .sess s j xO as' cs')
    (ha : actO xO as' cs' = .decide i spec sts link c₁ mids f) :
    ((next IV st x as cs).eval sha512).1 = (if s + 1 = nS x then .acc else .msg []) ↔
      j + 1 = VO.k xO ∧ Zk.verify spec (tabsOf sts) f.record f.proofs = .ok () ∧
        checks true i spec xO link c₁ mids f = .ok () ∧ fullDrawO xO f = true := by
  rw [next_sess sha512 h]
  simp only
  rcases nextO_decide_out sha512 ha with ho | ho
  · have hv := (nextO_decide_sha512 ha).1 ho
    rw [ho]
    by_cases hj : j + 1 = VO.k xO <;> by_cases hl : s + 1 = nS x <;> simp [sessOut, hj, hl, hv]
  · have hv : ¬ (Zk.verify spec (tabsOf sts) f.record f.proofs = .ok () ∧
        checks true i spec xO link c₁ mids f = .ok () ∧ fullDrawO xO f = true) := fun hv => by
      have := (nextO_decide_sha512 ha).2 hv
      rw [ho] at this
      cases this
    rw [ho]
    constructor
    · intro h'
      by_cases hl : s + 1 = nS x <;> simp [sessOut, hl] at h'
    · rintro ⟨-, hv'⟩
      exact absurd hv' hv

end Steps

end RecDeploy
