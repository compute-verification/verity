import Proofs.Flock.Claims.DeployInputs
import Flock.RecLinks

/-!
# A V* part's statement: its bytes, its parser and its recursive setup

A statement of the outer verifier the compiled verifier runs per V* part, `flock-verify verify --zk` with its recursive
flags, is the canonical byte encoding (`RecRaw.encode`) of exactly what that command reads besides the session
directory: Flock's statement (`FlockDeploy.Raw`, `--zk` on) and the bytes and numbers of `--rec-inner`, `--rec-part`,
`--rec-bindings`, `--rec-rows`, the inner statement's files (`--rec-inner-circuit`, `--rec-inner-public`,
`--rec-inner-partition`; `--rec-inner-coins`, which the verifier refuses), the inner session's record
(`--rec-inner-session`'s `session.json`), the coin server's transcript (`--rec-inner-transcript`), the circuit
registration (`--rec-inner-registration`) and how the inner statement is set up (`--rec-inner-statement`,
`--rec-inner-statement-coins`, `--rec-inner-tables`' tables, `--rec-inner-draw`) and the inner population's link table
(`--rec-links`: its root and row count). `decodeRec` keeps only a reading whose encoding is `x`, so it is injective
(`decodeRec_injective`), and refuses every other format, `flock-verify-rec-statement/v1` (no link table) among them.

`RecRaw.ins` runs the verifier's readers on those bytes as `Main.verifyCmd` runs them: Flock's (`Raw.ins`), then
`RecLinks.refuseLinks` (`linksOf`: a statement carrying a link table is refused), `RecVStar.Inner.ofJson`,
`RecVStar.ownBindingsOfJson`, `RecCoins.filesInput`'s source, the inner circuit's META binding
(`innerMetaRegistration`, from the header's first nine lines, `metaOf`), `recTranscriptInput`'s commits
(`RecSetup.commits`), `recLinkInput`'s tops and link points (`RecVStar.innerLink`, `linkInputOf`) and `Registered.ofJson`
on `--rec-rows`. `RecIns.setup` is `verifyCmd`'s `setupOf` at the statement's draw with those inputs and no
`--identifier`, in its order: `RecSetup.checkInner`, the session's statement (`Stmt.setupTables`) with each table's
registered reads and the identifier's absence, `Registered.checkFull`, `RecSetup.checks`, `RecSetup.oneFull`,
`Zk.shapeOk`,
`RecCoins.check`, `RecVStar.checkPubRows`, `RecVStar.checkLinkRounds`, `RecVStar.checkTopRows`, `RecCoins.checkLink`
(through `checkOpenTops` and `checkLevelTopReads`), `RecVStar.checkLabels` and the draw's law.
Like `FlockDeploy.Ins.setup` it refuses a statement without hm96-sha512 rows first, and its SHA-512 calls are the
shipped ones, outside the oracle.
-/

namespace RecDeploy

open Flock
open FlockDeploy (Raw encNat encBytes encStr encBool encOpt encList Rd rdTake rdNat rdBytes rdStr rdBool rdOpt rdList
  jsonOf Ins noIdentifier)
open Lean (Json)

/-! ## The statement's bytes -/

/-- **What `flock-verify verify --zk` reads of a V* part's statement** besides the session directory: Flock's statement,
then `--rec-inner`, `--rec-part`, `--rec-bindings`, `--rec-rows`, `--rec-inner-circuit`, `--rec-inner-public`,
`--rec-inner-partition`, `--rec-inner-coins`, `--rec-inner-session`'s `session.json`, `--rec-inner-transcript`,
`--rec-inner-registration`, `--rec-inner-statement`, `--rec-inner-statement-coins`, the tables in `--rec-inner-tables`,
`--rec-inner-draw`, and `--rec-links`' root and row count. -/
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
  innerStatementCoins : Option Bool
  innerTables : Option (List (String × ByteArray))
  innerDraw : Option ByteArray
  recLinks : Option (ByteArray × ℕ)

/-- The encoding's format string, its first field. -/
def RFORMAT : String := "flock-verify-rec-statement/v2"

/-- **The statement's canonical bytes**: Flock's statement's (`Raw.encode`) as one field, then the recursive flags'. -/
def RecRaw.encode (w : RecRaw) : Claims.Msg :=
  encStr RFORMAT ++ encBytes ⟨w.base.encode.toArray⟩ ++ encOpt encBytes w.recInner ++ encOpt encNat w.recPart ++
    encOpt encBytes w.recBindings ++ encOpt encBytes w.recRows ++ encOpt encBytes w.innerCircuit ++
    encOpt encBytes w.innerPublic ++ encOpt encBytes w.innerPartition ++ encOpt encBytes w.innerCoins ++
    encOpt encBytes w.innerSession ++ encOpt encBytes w.transcript ++ encOpt encBytes w.registration ++
    encOpt encStr w.innerStatement ++ encOpt encBool w.innerStatementCoins ++
    encOpt (encList fun p => encStr p.1 ++ encBytes p.2) w.innerTables ++ encOpt encBytes w.innerDraw ++
    encOpt (fun l => encBytes l.1 ++ encNat l.2) w.recLinks

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
  let innerStatementCoins ← rdOpt rdBool
  let innerTables ← rdOpt (rdList do
    let n ← rdStr
    let b ← rdBytes
    pure (n, b))
  let innerDraw ← rdOpt rdBytes
  let recLinks ← rdOpt (do
    let t ← rdBytes
    let n ← rdNat
    pure (t, n))
  pure ⟨base, recInner, recPart, recBindings, recRows, innerCircuit, innerPublic, innerPartition, innerCoins,
    innerSession, transcript, registration, innerStatement, innerStatementCoins, innerTables, innerDraw, recLinks⟩

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
  match Json.parse (t.drop 5).toString with
  | .error e => throw s!"rec inner circuit: META is not JSON: {e}"
  | .ok m => pure m

/-- **The inner circuit's binding** (`innerMetaRegistration`): META's registration digest and R's shape; none without
`--rec-inner-circuit`. -/
def boundOf : Option ByteArray → Flock.V (Option String × Option RecVStar.StackedShape)
  | none => .ok (none, none)
  | some c => do RecSetup.metaBound (← metaOf c)

/-- **How the inner statement is set up** (`RecCoins.formInput`): `--rec-inner-statement` with its coins (the coin
server's where `--rec-inner-statement-coins` is `os`, `true` here; `seed` unless it says), tables and draw, or none. -/
def formOf (w : RecRaw) : Except String (Option RecCoins.InnerForm) :=
  match w.innerStatement with
  | none =>
    if w.innerStatementCoins.isSome || w.innerTables.isSome || w.innerDraw.isSome then
      .error "--rec-inner-statement-coins, --rec-inner-tables and --rec-inner-draw go with --rec-inner-statement"
    else .ok none
  | some s => .ok (some ⟨s, w.innerStatementCoins.getD false, w.innerTables.getD [], w.innerDraw⟩)

/-- **Check 5's source** (`RecCoins.filesInput`): the inner statement's three files with the inner session's record and
how it is set up (`formOf`), or none; `--rec-inner-coins` is refused (`RecCoins.refuseCoins`). -/
def sourceOf (w : RecRaw) : Except String (Option RecCoins.Source) :=
  match w.innerCoins with
  | some _ => .error RecCoins.NO_INNER_COINS
  | none =>
    match w.innerCircuit, w.innerPublic, w.innerPartition, w.innerSession with
    | none, none, none, none => .ok none
    | some c, some p, some q, some s => do
      let form ← formOf w
      let partition ← jsonOf "--rec-inner-partition" q
      let r ← textOf "rec inner session" s
      pure (some (.files ⟨c, p, partition, r, none, form⟩))
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

/-- **The circuit table's tops and the link points, as the coin server recorded them** (`recLinkInput`): with
`--rec-inner-transcript` and the source's record, `RecVStar.innerLink` on the transcript, which holds the record's link
points to the coin server's; none without either. -/
def linkInputOf (w : RecRaw) (src : Option RecCoins.Source) : Except String (Option RecVStar.InnerLink) :=
  match w.transcript, src.map (·.record) with
  | some t, some record => do
    match Json.parse (← textOf "rec inner transcript" t) with
    | .ok j => some <$> RecVStar.innerLink j record
    | .error e => .error s!"rec inner transcript: not JSON: {e}"
  | _, _ => .ok none

/-- **The link table is refused** (`RecLinks.refuseLinks`): a statement carrying one, whatever it holds. -/
def linksOf (w : RecRaw) : Except String Unit :=
  match w.recLinks with
  | some _ => .error RecLinks.NO_LINKS
  | none => .ok ()

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

/-- **The verifier's readers on the statement's bytes**, as `verifyCmd` runs them. -/
def RecRaw.ins (w : RecRaw) : Except String RecIns := do
  let base ← w.base.ins
  linksOf w
  let inner ← w.recInner.mapM fun b => do RecVStar.Inner.ofJson (← jsonOf "rec-inner" b)
  let bindings ← w.recBindings.mapM fun b => do RecVStar.ownBindingsOfJson (← jsonOf "rec-bindings" b)
  let src ← sourceOf w
  let bound := boundOf w.innerCircuit
  let commits ← commitsOf w src bound inner
  let link ← linkInputOf w src
  let rows ← w.recRows.mapM fun b => do
    (Registered.ofJson (← jsonOf "rec rows" b)).mapError fun e => s!"rec rows: {e}"
  pure { base, inner, part := w.recPart, bindings, src, bound, commits, link, rows }

/-- **`verifyCmd`'s setup with the recursive inputs** at the statement's draw, with no `--identifier`. -/
def RecIns.setup (s : RecIns) : Flock.V (SessionSpec × Array Stmt) := do
  let r := s.base.read
  if !r.tags.hm96Rows then
    throw s!"{r.tags.name}: refused: a statement without hm96-sha512 rows has no draw, and every claim is about a drawn session"
  RecSetup.checkInner s.inner s.src s.bound
  let (spec, sts) ← r.setup
  for st in sts do
    checkRegistered st s.base.registered
    noIdentifier st
  Registered.checkFull s.base.registered r.draw
  RecSetup.checks s.inner s.commits s.base.stmtTags r.I.tables sts r.I.circuitFile r.I.partition r.I.program
    s.base.registered s.part s.bindings s.rows
  RecSetup.oneFull s.inner sts
  if r.zk then sts.forM Zk.shapeOk
  sts.forM (RecCoins.check · s.base.registered (RecCoins.Source.atC3 s.src s.inner s.commits)
    (RecSetup.c3 s.inner s.commits))
  sts.forM (RecVStar.checkPubRows · s.base.registered s.commits (s.inner.bind (·.sparse)))
  if let (some i, some lk) := (s.inner, s.link) then RecVStar.checkLinkRounds i lk
  sts.forM (RecVStar.checkTopRows · s.base.registered s.link)
  sts.forM (RecCoins.checkLink · s.base.registered (RecCoins.Source.atC3 s.src s.inner s.commits) s.link)
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

/-- **A statement whose readers pass carries no link table** (`linksOf`). -/
theorem RecRaw.ins_links {w : RecRaw} {s : RecIns} (h : w.ins = .ok s) : w.recLinks = none := by
  cases hl : w.recLinks with
  | none => rfl
  | some l =>
    exfalso
    have hL : linksOf w = .error RecLinks.NO_LINKS := by simp [linksOf, hl]
    unfold RecRaw.ins at h
    cases hb : w.base.ins with
    | error e =>
      rw [hb] at h
      first | (cases h; done) | (simp [bind, Except.bind] at h; done)
    | ok b =>
      rw [hb, hL] at h
      first | (cases h; done) | (simp [bind, Except.bind] at h; done)

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

end RecDeploy
