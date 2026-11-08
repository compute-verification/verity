module

public import Proofs.Flock.Claims.DeployInputs
public import Flock.RecLinkPart
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init
import VCVio.Prelude.Core

/-!
# A V* part's statement: its bytes, its parser and its recursive setup

A statement of the outer verifier the compiled verifier runs per V* part, `flock-verify verify --zk` with its recursive
flags, is the canonical byte encoding (`RecRaw.encode`) of exactly what that command reads besides the session
directory: Flock's statement (`FlockDeploy.Raw`, `--zk` on) and the bytes and numbers of `--rec-inner`, `--rec-part`,
`--rec-bindings`, `--rec-rows`, the inner statement's files (`--rec-inner-circuit`, `--rec-inner-public`,
`--rec-inner-partition`; `--rec-inner-coins`, which the verifier refuses), the inner session's record
(`--rec-inner-session`'s `session.json`), the challenger's live-coin part's transcript (`--rec-inner-transcript`), the circuit
registration (`--rec-inner-registration`) and how the inner statement is set up (`--rec-inner-statement`,
`--rec-inner-tables`' tables, `--rec-inner-draw`), the inner population's link table
(`--rec-links`: its root and row count) and the verifier's program of the inner statement (`--rec-inner-program`).
`decodeRec` keeps only a reading whose encoding is `x`, so it is injective (`decodeRec_injective`), and refuses every
other format, `flock-verify-rec-statement/v3` (with the inner statement's coins) among them.

`RecRaw.ins` runs the verifier's readers on those bytes as `Main.verifyCmd` runs them: Flock's (`Raw.ins`), then
`RecLinkTree.linksArg`'s checks of the link table's root and row count (`linksOf`), `RecVStar.Inner.ofJson`,
`RecVStar.ownBindingsOfJson`, `RecCoins.filesInput`'s source, the inner circuit's META binding
(`innerMetaRegistration`, from the header's first nine lines, `metaOf`), `recTranscriptInput`'s commits
(`RecSetup.commits`), `recLinkInput`'s tops and link points (`RecVStar.innerLink`, `linkInputOf`) and `Registered.ofJson`
on `--rec-rows`. `RecIns.setup` is `verifyCmd`'s `setupOf` at the statement's sample with those inputs and no
`--identifier`, in its order: `RecSetup.checkInner`, `RecLinkPart.checkShape` (the link table the inner shape's),
`RecBudget.regs`, `RecBudget.statements`, the session's statement (`Stmt.setupTables`) with each table's registered
reads and the identifier's absence, `Registered.checkFull`, `RecSetup.checks`, `RecSetup.oneFull`, `RecBudget.table`,
`Zk.shapeOk`,
`RecCoins.check`, `RecVStar.checkPubRows`, `RecVStar.checkLinkRounds`, `RecVStar.checkTopRows`, `RecCoins.checkLink`
(through `checkOpenTops` and `checkLevelTopReads`), `RecLinkPart.check` (the link table's part against the table and
the inner statement), `RecVStar.checkLabels` and the sample's law.
Like `FlockDeploy.Ins.setup` it refuses a statement without hm96-sha512 rows first, and its SHA-512 calls are the
shipped ones, outside the oracle.
-/

@[expose] public section

namespace RecDeploy

open Flock
open FlockDeploy (Raw encNat encBytes encStr encBool encOpt encList Rd rdTake rdNat rdBytes rdStr rdBool rdOpt rdList
  jsonOf Ins noIdentifier)
open Lean (Json)

/-! ## The statement's bytes -/

/-- **What `flock-verify verify --zk` reads of a V* part's statement** besides the session directory: Flock's statement,
then `--rec-inner`, `--rec-part`, `--rec-bindings`, `--rec-rows`, `--rec-inner-circuit`, `--rec-inner-public`,
`--rec-inner-partition`, `--rec-inner-coins`, `--rec-inner-session`'s `session.json`, `--rec-inner-transcript`,
`--rec-inner-registration`, `--rec-inner-statement`, the tables in `--rec-inner-tables`,
`--rec-inner-draw`, `--rec-links`' root and row count, and `--rec-inner-program`. -/
structure RecRaw where
  base : Raw
  recInner : Option ByteArray
  recPart : Option ℕ
  recBindings : Option ByteArray
  recRows : Option ByteArray
  innerCircuit : Option ByteArray
  innerPublic : Option ByteArray
  innerPartition : Option ByteArray
  innerCoins : Option ByteArray
  innerSession : Option ByteArray
  transcript : Option ByteArray
  registration : Option ByteArray
  innerStatement : Option String
  innerTables : Option (List (String × ByteArray))
  innerDraw : Option ByteArray
  recLinks : Option (ByteArray × ℕ)
  innerProgram : Option ByteArray

/-- The encoding's format string, its first field. -/
def RFORMAT : String := "flock-verify-rec-statement/v4"

/-- **The statement's canonical bytes**: Flock's statement's (`Raw.encode`) as one field, then the recursive flags'. -/
def RecRaw.encode (w : RecRaw) : Claims.Msg :=
  encStr RFORMAT ++ encBytes ⟨w.base.encode.toArray⟩ ++ encOpt encBytes w.recInner ++ encOpt encNat w.recPart ++
    encOpt encBytes w.recBindings ++ encOpt encBytes w.recRows ++ encOpt encBytes w.innerCircuit ++
    encOpt encBytes w.innerPublic ++ encOpt encBytes w.innerPartition ++ encOpt encBytes w.innerCoins ++
    encOpt encBytes w.innerSession ++ encOpt encBytes w.transcript ++ encOpt encBytes w.registration ++
    encOpt encStr w.innerStatement ++
    encOpt (encList fun p => encStr p.1 ++ encBytes p.2) w.innerTables ++ encOpt encBytes w.innerDraw ++
    encOpt (fun l => encBytes l.1 ++ encNat l.2) w.recLinks ++ encOpt encBytes w.innerProgram

/-- `RecRaw.encode`'s layout, read back. -/
def RecRaw.read : Rd RecRaw := do
  if (← rdStr) ≠ RFORMAT then failure
  let b ← rdBytes
  let some base := FlockDeploy.decode b.data.toList | failure
  let recInner ← rdOpt rdBytes
  let recPart ← rdOpt rdNat
  let recBindings ← rdOpt rdBytes
  let recRows ← rdOpt rdBytes
  let innerCircuit ← rdOpt rdBytes
  let innerPublic ← rdOpt rdBytes
  let innerPartition ← rdOpt rdBytes
  let innerCoins ← rdOpt rdBytes
  let innerSession ← rdOpt rdBytes
  let transcript ← rdOpt rdBytes
  let registration ← rdOpt rdBytes
  let innerStatement ← rdOpt rdStr
  let innerTables ← rdOpt (rdList do
    let n ← rdStr
    let b ← rdBytes
    pure (n, b))
  let innerDraw ← rdOpt rdBytes
  let recLinks ← rdOpt (do
    let t ← rdBytes
    let n ← rdNat
    pure (t, n))
  let innerProgram ← rdOpt rdBytes
  pure ⟨base, recInner, recPart, recBindings, recRows, innerCircuit, innerPublic, innerPartition, innerCoins,
    innerSession, transcript, registration, innerStatement, innerTables, innerDraw, recLinks,
    innerProgram⟩

/-- **The statement's decoder**: `RecRaw.read` on all of `x`, kept only when its encoding is `x`. -/
def decodeRec (x : Claims.Msg) : Option RecRaw :=
  match RecRaw.read.run x with
  | some (w, []) => if w.encode = x then some w else none
  | _ => none

theorem decodeRec_encode {x : Claims.Msg} {w : RecRaw} (h : decodeRec x = some w) : w.encode = x := by
  unfold decodeRec at h
  split at h
  · split_ifs at h with he
    cases h
    exact he
  · cases h

/-- **The decoder is injective.** -/
theorem decodeRec_injective {x y : Claims.Msg} {w : RecRaw} (hx : decodeRec x = some w) (hy : decodeRec y = some w) :
    x = y :=
  (decodeRec_encode hx).symm.trans (decodeRec_encode hy)

/-! ## The verifier's readers -/

/-- A file read as text, as `IO.FS.readFile` reads one: UTF-8. -/
def textOf (what : String) (b : ByteArray) : Except String String :=
  match String.fromUTF8? b with
  | some t => .ok t
  | none => .error s!"{what}: not UTF-8"

/-- A file's lines: its bytes split at each newline. -/
def linesAux : List UInt8 → List UInt8 → List (List UInt8)
  | [], cur => [cur.reverse]
  | b :: bs, cur => if b = 10 then cur.reverse :: linesAux bs [] else linesAux bs (b :: cur)

/-- The first of at most `n` lines that starts with `META `, each read as UTF-8 as `innerMeta`'s `getLine` reads it. -/
def metaLine : ℕ → List (List UInt8) → Except String String
  | 0, _ => .error "rec inner circuit: no META line"
  | _ + 1, [] => .error "rec inner circuit: no META line"
  | n + 1, l :: ls =>
    match String.fromUTF8? ⟨l.toArray⟩ with
    | none => .error "rec inner circuit: not UTF-8"
    | some t => if t.startsWith "META " then .ok t else metaLine n ls

/-- **The inner circuit's META** (`innerMeta`): its header's first nine lines, up to its `META` line. -/
def metaOf (c : ByteArray) : Except String Json := do
  let t ← metaLine 9 (linesAux c.data.toList [])
  match Flock.JsonParse.parse (t.drop 5).toString with
  | .error e => throw s!"rec inner circuit: META is not JSON: {e}"
  | .ok m => pure m

/-- **The inner circuit's binding** (`innerMetaRegistration`): META's registration digest and R's shape; none without
`--rec-inner-circuit`. -/
def boundOf : Option ByteArray → Flock.V (Option String × Option RecVStar.StackedShape)
  | none => .ok (none, none)
  | some c => do RecSetup.metaBound (← metaOf c)

/-- **How the inner statement is set up** (`RecCoins.formInput`): `--rec-inner-statement` with its tables and sample, or
none. -/
def formOf (w : RecRaw) : Except String (Option RecCoins.InnerForm) :=
  match w.innerStatement with
  | none =>
    if w.innerTables.isSome || w.innerDraw.isSome then
      .error "--rec-inner-tables and --rec-inner-draw go with --rec-inner-statement"
    else .ok none
  | some s => .ok (some ⟨s, w.innerTables.getD [], w.innerDraw⟩)

/-- **Check 5's source** (`RecCoins.filesInput`): the inner statement's three files with the inner session's record,
how it is set up (`formOf`) and the verifier's program of it, or none; `--rec-inner-coins` is refused
(`RecCoins.refuseCoins`). -/
def sourceOf (w : RecRaw) : Except String (Option RecCoins.Source) :=
  match w.innerCoins with
  | some _ => .error RecCoins.NO_INNER_COINS
  | none =>
    match w.innerCircuit, w.innerPublic, w.innerPartition, w.innerSession with
    | none, none, none, none =>
      if w.innerProgram.isSome then
        .error "--rec-inner-program goes with the inner statement's files (--rec-inner-circuit and its three)"
      else .ok none
    | some c, some p, some q, some s => do
      let form ← formOf w
      let partition ← jsonOf "--rec-inner-partition" q
      let r ← textOf "rec inner session" s
      pure (some (.files ⟨c, p, partition, r, none, form, w.innerProgram⟩))
    | _, _, _, _ =>
      .error "--rec-inner-circuit, --rec-inner-public, --rec-inner-partition and --rec-inner-session go together"

/-- **The inner session's commits** (`recTranscriptInput`): with `--rec-inner-transcript`, `RecSetup.commits` on the
transcript, the registration, the binding and the source's record; none without it. -/
def commitsOf (w : RecRaw) (src : Option RecCoins.Source) (bound : Flock.V (Option String × Option RecVStar.StackedShape))
    (inner : Option RecVStar.Inner) : Except String (Option RecVStar.InnerCommits) :=
  match w.transcript with
  | none =>
    if w.registration.isSome then .error "--rec-inner-registration goes with --rec-inner-transcript" else .ok none
  | some t =>
    match src.map (·.record) with
    | none =>
      .error "--rec-inner-transcript goes with the inner session (--rec-inner-session and its three files)"
    | some record =>
      if w.registration.isSome && w.innerCircuit.isNone then
        .error "--rec-inner-registration goes with --rec-inner-circuit, whose META binds it"
      else
        some <$> RecSetup.commits inner (textOf "rec inner transcript" t)
          (w.registration.mapM (textOf "rec inner registration")) bound record

/-- **The circuit table's tops and the link points, as the challenger's live-coin part recorded them** (`recLinkInput`): with
`--rec-inner-transcript` and the source's record, `RecVStar.innerLink` on the transcript, which holds the record's link
points to the challenger's live-coin part's; none without either. -/
def linkInputOf (w : RecRaw) (src : Option RecCoins.Source) : Except String (Option RecVStar.InnerLink) :=
  match w.transcript, src.map (·.record) with
  | some t, some record => do
    match Flock.JsonParse.parse (← textOf "rec inner transcript" t) with
    | .ok j => some <$> RecVStar.innerLink j record
    | .error e => .error s!"rec inner transcript: not JSON: {e}"
  | _, _ => .ok none

/-- **The link table** (`RecLinkTree.linksArg`): its root and row count where `RecLinkTree.checkLinks` takes them,
none without one. -/
def linksOf (w : RecRaw) : Except String (Option (ByteArray × ℕ)) :=
  w.recLinks.mapM fun l => do RecLinkTree.checkLinks l.1 l.2; pure l

/-- What the recursive setup reads of a parsed statement: Flock's (`Ins`), and the recursive inputs as `verifyCmd`
holds them. -/
structure RecIns where
  base : Ins
  inner : Option RecVStar.Inner
  part : Option ℕ
  bindings : Option VStar.Bindings
  src : Option RecCoins.Source
  bound : Flock.V (Option String × Option RecVStar.StackedShape)
  commits : Option RecVStar.InnerCommits
  link : Option RecVStar.InnerLink
  rows : Option (Array Registered.Port)
  links : Option (ByteArray × ℕ)

/-- **The verifier's readers on the statement's bytes**, as `verifyCmd` runs them. -/
def RecRaw.ins (w : RecRaw) : Except String RecIns := do
  let base ← w.base.ins
  let links ← linksOf w
  let inner ← w.recInner.mapM fun b => do RecVStar.Inner.ofJson (← jsonOf "rec-inner" b)
  let bindings ← w.recBindings.mapM fun b => do RecVStar.ownBindingsOfJson (← jsonOf "rec-bindings" b)
  let src ← sourceOf w
  let bound := boundOf w.innerCircuit
  let commits ← commitsOf w src bound inner
  let link ← linkInputOf w src
  let rows ← w.recRows.mapM fun b => do
    (Registered.ofJson (← jsonOf "rec rows" b)).mapError fun e => s!"rec rows: {e}"
  pure { base, inner, part := w.recPart, bindings, src, bound, commits, link, rows, links }

/-- **`verifyCmd`'s setup with the recursive inputs** at the statement's sample, with no `--identifier`. -/
def RecIns.setup (s : RecIns) : Flock.V (SessionSpec × Array Stmt) := do
  let r := s.base.read
  if !r.tags.hm96Rows then
    throw s!"{r.tags.name}: refused: a statement without hm96-sha512 rows has no draw, and every claim is about a drawn session"
  RecSetup.checkInner s.inner s.src s.bound
  RecLinkPart.checkShape s.inner s.links
  RecBudget.regs s.inner s.base.registered
  RecBudget.statements s.inner
  let (spec, sts) ← r.setup
  for st in sts do
    checkRegistered st s.base.registered
    noIdentifier st
  Registered.checkFull s.base.registered r.draw
  RecSetup.checks s.inner s.commits s.base.stmtTags r.I.tables sts r.I.circuitFile r.I.partition r.I.program
    s.base.registered s.part s.bindings s.rows
  RecSetup.oneFull s.inner sts
  RecBudget.table s.inner sts
  if r.zk then sts.forM Zk.shapeOk
  sts.forM (RecCoins.check · s.base.registered (RecCoins.Source.atC3 s.src s.inner s.commits)
    (RecSetup.c3 s.inner s.commits))
  sts.forM (RecVStar.checkPubRows · s.base.registered s.commits (s.inner.bind (·.sparse)))
  if let (some i, some lk) := (s.inner, s.link) then RecVStar.checkLinkRounds i lk
  sts.forM (RecVStar.checkTopRows · s.base.registered s.link)
  sts.forM (RecCoins.checkLink · s.base.registered (RecCoins.Source.atC3 s.src s.inner s.commits) s.link)
  sts.forM (RecLinkPart.check · s.base.registered s.src s.links)
  if s.inner.isSome then RecVStar.checkLabels r.draw r.I.count sts
  if r.draw.isSome then ProvedScope.drawLaw r.zk r.tags false r.draw
  pure (spec, sts)

/-- The statement's inputs: a statement of the masked verifier only. -/
def insO (x : Claims.Msg) : Option RecIns := do
  let w ← decodeRec x
  if w.base.zk = false then none
  w.ins.toOption

/-- The statement's inputs and its recursive setup. -/
def setupO (x : Claims.Msg) : Option (RecIns × SessionSpec × Array Stmt) := do
  let s ← insO x
  let (spec, sts) ← s.setup.toOption
  pure (s, spec, sts)

/-- The recursive setup's session is the verifier's statement's (`Read.setup`, `Stmt.setupTables`). -/
theorem RecIns.setup_read {s : RecIns} {spec : SessionSpec} {sts : Array Stmt} (h : s.setup = .ok (spec, sts)) :
    s.base.read.setup = .ok (spec, sts) := by
  unfold RecIns.setup at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | (cases h; assumption)

theorem setupO_some {x : Claims.Msg} {s : RecIns} {spec : SessionSpec} {sts : Array Stmt}
    (h : setupO x = some (s, spec, sts)) : insO x = some s ∧ s.setup = .ok (spec, sts) := by
  unfold setupO at h
  cases hi : insO x with
  | none => simp [hi] at h
  | some s' =>
    cases hsu : s'.setup with
    | error e => simp [hi, hsu, Except.toOption] at h
    | ok p =>
      simp [hi, hsu, Except.toOption] at h
      obtain ⟨rfl, rfl, rfl⟩ := h
      exact ⟨rfl, hsu⟩

/-- What the verifier reads of the statement's bytes as Flock's (`Readable.Read`). -/
def readOfO (x : Claims.Msg) : Option Readable.Read := (insO x).map fun s => s.base.read

theorem RecRaw.ins_base {w : RecRaw} {s : RecIns} (h : w.ins = .ok s) : w.base.ins = .ok s.base := by
  unfold RecRaw.ins at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | (cases h; assumption)

/-- A statement of the outer verifier is read as masked. -/
theorem readOfO_zk {x : Claims.Msg} {r : Readable.Read} (h : readOfO x = some r) : r.zk = true := by
  unfold readOfO insO at h
  obtain ⟨s, hs, rfl⟩ := Option.map_eq_some_iff.1 h
  obtain ⟨w, -, hw⟩ := Option.bind_eq_some_iff.1 hs
  split_ifs at hw with hz
  · cases hw
  change w.ins.toOption = some s at hw
  cases hi : w.ins with
  | error e => rw [hi] at hw; cases hw
  | ok s' =>
    rw [hi] at hw
    cases hw
    rw [FlockDeploy.Raw.ins_zk (RecRaw.ins_base hi)]
    simpa using hz

/-! ## Flock's statements and the V* parts' -/

/-- Flock's statement reader reads Flock's format string first. -/
theorem read_format {x : Claims.Msg} {p : Raw × Claims.Msg} (h : FlockDeploy.Raw.read.run x = some p) :
    ∃ r, rdStr.run x = some (FlockDeploy.FORMAT, r) := by
  unfold FlockDeploy.Raw.read at h
  simp only [StateT.run_bind] at h
  cases hs : rdStr.run x with
  | none => simp [hs] at h
  | some q =>
    obtain ⟨s, r⟩ := q
    by_cases e : s = FlockDeploy.FORMAT
    · exact ⟨r, by rw [e]⟩
    · (simp [hs, e] at h) <;> cases h

/-- A V* part's statement reader reads its format string first. -/
theorem RecRaw.read_format {x : Claims.Msg} {p : RecRaw × Claims.Msg} (h : RecRaw.read.run x = some p) :
    ∃ r, rdStr.run x = some (RFORMAT, r) := by
  unfold RecRaw.read at h
  simp only [StateT.run_bind] at h
  cases hs : rdStr.run x with
  | none => simp [hs] at h
  | some q =>
    obtain ⟨s, r⟩ := q
    by_cases e : s = RFORMAT
    · exact ⟨r, by rw [e]⟩
    · (simp [hs, e] at h) <;> cases h

theorem formats_ne : FlockDeploy.FORMAT ≠ RFORMAT := by decide

/-- **No statement is both Flock's and a V* part's**: the two encodings' first fields, their format strings, differ. -/
theorem decodeRec_none {x : Claims.Msg} {w : Raw} (h : FlockDeploy.decode x = some w) : decodeRec x = none := by
  unfold FlockDeploy.decode at h
  split at h
  · rename_i w' hr
    obtain ⟨r, hr'⟩ := read_format hr
    unfold decodeRec
    split
    · rename_i v hv
      obtain ⟨r', hv'⟩ := RecRaw.read_format hv
      rw [hr'] at hv'
      exact absurd (Prod.mk.inj (Option.some.inj hv')).1 formats_ne
    · rfl
  · cases h

theorem decode_none {x : Claims.Msg} {w : RecRaw} (h : decodeRec x = some w) : FlockDeploy.decode x = none := by
  cases hd : FlockDeploy.decode x with
  | none => rfl
  | some v => rw [decodeRec_none hd] at h; cases h

/-- A statement that decodes as no V* part's has no recursive setup. -/
theorem setupO_none {x : Claims.Msg} (h : decodeRec x = none) : setupO x = none := by
  unfold setupO insO
  simp [h]

/-- A statement that decodes as no Flock statement sets up no session of Flock's. -/
theorem setupOf_none {zk : Bool} {x : Claims.Msg} (h : FlockDeploy.decode x = none) : FlockDeploy.setupOf zk x = none := by
  unfold FlockDeploy.setupOf FlockDeploy.insOf
  simp [h]

theorem decodeRec_of_setupO {x : Claims.Msg} {p : RecIns × SessionSpec × Array Stmt} (h : setupO x = some p) :
    ∃ w, decodeRec x = some w := by
  cases hd : decodeRec x with
  | none => rw [setupO_none hd] at h; cases h
  | some w => exact ⟨w, rfl⟩

theorem decodeRec_of_readOfO {x : Claims.Msg} {r : Readable.Read} (h : readOfO x = some r) :
    ∃ w, decodeRec x = some w := by
  unfold readOfO insO at h
  cases hd : decodeRec x with
  | none => simp [hd] at h
  | some w => exact ⟨w, rfl⟩

theorem decode_of_readOf {zk : Bool} {x : Claims.Msg} {r : Readable.Read} (h : FlockDeploy.readOf zk x = some r) :
    ∃ w, FlockDeploy.decode x = some w := by
  unfold FlockDeploy.readOf FlockDeploy.insOf at h
  cases hd : FlockDeploy.decode x with
  | none => simp [hd] at h
  | some w => exact ⟨w, rfl⟩

/-- What a deployed parser parses, its reader reads. -/
theorem read_of_parseOf {rel : Readable.Parsed → Readable.Relation} {rd : Claims.Msg → Option Readable.Read}
    {x : Claims.Msg} {s : Claims.Flock.Stmt} (h : Readable.parseOf rel rd x = some s) : ∃ r, rd x = some r := by
  unfold Readable.parseOf Readable.parseBytes at h
  cases hr : rd x with
  | none => simp [hr] at h
  | some r => exact ⟨r, rfl⟩

end RecDeploy
