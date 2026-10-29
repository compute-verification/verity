import Proofs.Flock.Claims.Recursion.PartsVStar
import Proofs.Flock.Claims.Recursion.OpensReads
import Proofs.Properties.Verity.Recursive.Names
import VStarRecursive

/-!
# The compiled part's setup is the model's recursive setup (piece D2)

Each part `j` of the compiled recursive component (`VStar.Recursive.partVerdict`) sets its session up as
`flock-verify verify --zk` does with that part's recursive flags: Flock's session (`OneStageZk.buildSessionOf`) and the
verifier's own checks on it (piece D1), then V\*'s checks block, `RecSetup.checks`, `RecSetup.oneFull`, `Zk.shapeOk`,
`RecCoins.check`, `RecVStar.checkPubRows`, `RecVStar.checkLinkRounds`, `RecVStar.checkTopRows`, `RecCoins.checkLink`,
`RecVStar.checkLabels` and the draw's law. The model's recursive setup (`RecDeploy.RecIns.setup`, which `setupO` runs on
a V\* statement) makes the same calls in the same order. This module matches them.

* `recIns_setup_of_compiled`: where the staged statement's inputs `i` are the compiled part's (`SetupMatch`, piece C),
  the model's Flock setup and its registered-reads checks pass with the compiled session (piece D1), and the compiled
  checks block passes (`RecChecks`), the model's recursive setup passes with the same session.
* `setupO_staged_of_compiled`: so `setupO (staged st x ms s) = some (i, spec, sts)`.
* `readX_of_compiled`: where every part's setup passes that way and the compiled scope check passes, `readX x` reads
  what `readXPre x` read (`readX_of_sessions`), and each part's circuit is V\*'s statement's of its index
  (`partsVStar`).
* `partVerdict_setup`: what an accepted compiled part's verdict shows of its setup, in the terms the lemmas above and
  D1's take.

The inner session's META binding (`InnerHeld.bound`) is read by the setup only at a RecSparse shape (`checkMeta`), so
at the dense shape the compiled component accepts (`shapeOf`: no sparse shape) the two readings of the inner circuit's
META need not be compared.
-/

namespace Verity.Recursive

open Flock
open RecDeploy
open FlockDeploy (noIdentifier)

/-- **The staged statement's inputs are the compiled part's** (piece C's conclusion, at the fields the recursive setup
and the scope check read): the model's reading `i` of part `j`'s staged statement holds, field by field, the values the
compiled part `j`'s check passes to the same calls: `--zk`, the statement's tags (`tags0`) and the session's
(`tags`), the part's circuit and staged public file, no lookup table, one table, the part's partition and program, the
verifier's registered reads, the record's draw, the inner shape, check 5's source, the inner commits and link the
compiled component holds (`h`), `--rec-part j`, the bindings and the rows. -/
structure SetupMatch (i : RecIns) (h : VStar.Recursive.InnerHeld) (j : ℕ) (tags0 tags : Tags)
    (circuitB pubB : ByteArray) (partition : Lean.Json) (program : ByteArray)
    (registered : Option (Array Registered.Port)) (bindings : VStar.Bindings)
    (rows : Option (Array Registered.Port)) (draw : Option Lean.Json) : Prop where
  zk : i.base.read.zk = true
  stmtTags : i.base.stmtTags = tags0
  tags : i.base.read.tags = tags
  circuitFile : i.base.read.I.circuitFile = circuitB
  publicFile : i.base.read.I.publicFile = pubB
  tables : i.base.read.I.tables = []
  count : i.base.read.I.count = 1
  partition : i.base.read.I.partition = some partition
  program : i.base.read.I.program = some program
  publicInputs : i.base.read.publicInputs = none
  readRegistered : i.base.read.registered = registered.isSome
  registered : i.base.registered = registered
  draw : i.base.read.draw = draw
  inner : i.inner = some h.shape
  src : i.src = some h.src
  commits : i.commits = some h.commits
  link : i.link = some h.link
  part : i.part = some j
  bindings : i.bindings = some bindings
  rows : i.rows = rows

/-- **The compiled part's checks block** after Flock's session and the checks piece D1 matches, each as the compiled
part `j`'s check calls it on the session's tables `sts`: the inner checks, the registration's and V\*'s list's caps,
V\*'s checks, one full table, the table's cap, `Zk.shapeOk`, check 5, the public rows, the link rounds, the top rows,
the link, the labels, and the draw's law where there is a draw. -/
structure RecChecks (h : VStar.Recursive.InnerHeld) (j : ℕ) (tags0 tags : Tags) (circuitB : ByteArray)
    (partition : Lean.Json) (program : ByteArray) (registered : Option (Array Registered.Port))
    (bindings : VStar.Bindings) (rows : Option (Array Registered.Port)) (draw : Option Lean.Json)
    (sts : Array Stmt) : Prop where
  inner : RecSetup.checkInner (some h.shape) (some h.src) h.bound = .ok ()
  regs : RecBudget.regs (some h.shape) registered = .ok ()
  statements : RecBudget.statements (some h.shape) = .ok ()
  checks : RecSetup.checks (some h.shape) (some h.commits) tags0 [] sts circuitB (some partition) (some program)
    registered (some j) (some bindings) rows = .ok ()
  oneFull : RecSetup.oneFull (some h.shape) sts = .ok ()
  table : RecBudget.table (some h.shape) sts = .ok ()
  shapeOk : sts.forM Zk.shapeOk = .ok ()
  coins : sts.forM (RecCoins.check · registered (RecCoins.Source.atC3 (some h.src) (some h.shape) (some h.commits))
    (RecSetup.c3 (some h.shape) (some h.commits))) = .ok ()
  pubRows : sts.forM (RecVStar.checkPubRows · registered (some h.commits) ((some h.shape).bind (·.sparse))) = .ok ()
  linkRounds : RecVStar.checkLinkRounds h.shape h.link = .ok ()
  topRows : sts.forM (RecVStar.checkTopRows · registered (some h.link)) = .ok ()
  link : sts.forM (RecCoins.checkLink · registered (RecCoins.Source.atC3 (some h.src) (some h.shape) (some h.commits))
    (some h.link)) = .ok ()
  labels : RecVStar.checkLabels draw 1 sts = .ok ()
  law : draw.isSome = true → ProvedScope.drawLaw true tags false draw = .ok ()

/-- At a shape without a RecSparse phase, the inner checks don't read the inner circuit's META binding. -/
theorem checkInner_dense {sh : RecVStar.Inner} {src : RecCoins.Source}
    {b b' : V (Option String × Option RecVStar.StackedShape)} (hs : sh.sparse = none) :
    RecSetup.checkInner (some sh) (some src) b = RecSetup.checkInner (some sh) (some src) b' := by
  cases src <;> simp [RecSetup.checkInner, hs]

/-- A loop each of whose steps passes and goes on passes. -/
theorem forIn_ok_of_all {α : Type} {xs : Array α} {b : α → PUnit → V (ForInStep PUnit)}
    (hb : ∀ a ∈ xs, ∀ u, b a u = .ok (.yield PUnit.unit)) : forIn xs PUnit.unit b = .ok PUnit.unit := by
  rcases xs with ⟨l⟩
  simp only [List.forIn_toArray]
  induction l with
  | nil => rfl
  | cons c l ih =>
    simp only [List.forIn_cons, bind, Except.bind]
    rw [hb c (by simp) PUnit.unit]
    exact ih fun a ha u => hb a (by simp_all) u

/-- **The model's recursive setup passes where the compiled part's does.** Suppose the model's reading `i` of part
`j`'s staged statement holds the compiled part's inputs (`SetupMatch`), the inner shape has no RecSparse phase and the
session's tags hm96-sha512 rows (as the compiled part's do wherever it has a draw). Suppose the model's Flock setup
gives the compiled session `(spec, sts)`, every table passes the registered-reads and identifier checks and the
registration is full (piece D1's conclusion), and the compiled checks block passes on `sts` (`RecChecks`). Then the
model's recursive setup (`RecIns.setup`) passes with `(spec, sts)`. -/
theorem recIns_setup_of_compiled {i : RecIns} {h : VStar.Recursive.InnerHeld} {j : ℕ} {tags0 tags : Tags}
    {circuitB pubB : ByteArray} {partition : Lean.Json} {program : ByteArray}
    {registered : Option (Array Registered.Port)} {bindings : VStar.Bindings}
    {rows : Option (Array Registered.Port)} {draw : Option Lean.Json} {spec : SessionSpec} {sts : Array Stmt}
    (hm : SetupMatch i h j tags0 tags circuitB pubB partition program registered bindings rows draw)
    (hsparse : h.shape.sparse = none) (hhm : tags.hm96Rows = true)
    (hset : i.base.read.setup = .ok (spec, sts))
    (hregs : ∀ st ∈ sts, checkRegistered st i.base.registered = .ok () ∧ noIdentifier st = .ok ())
    (hfull : Registered.checkFull i.base.registered i.base.read.draw = .ok ())
    (hc : RecChecks h j tags0 tags circuitB partition program registered bindings rows draw sts) :
    i.setup = .ok (spec, sts) := by
  have hsh : i.base.read.tags.hm96Rows = true := by rw [hm.tags]; exact hhm
  have hin : RecSetup.checkInner i.inner i.src i.bound = .ok () := by
    rw [hm.inner, hm.src, checkInner_dense hsparse]
    exact hc.inner
  unfold RecIns.setup
  simp only [hsh, Bool.not_true, Bool.false_eq_true, ↓reduceIte, hin, hset, bind, Except.bind, pure, Except.pure]
  rw [forIn_ok_of_all fun a ha u => by simp only [(hregs a ha).1, (hregs a ha).2]]
  simp only [hfull]
  simp only [hm.zk, hm.stmtTags, hm.tags, hm.tables, hm.circuitFile, hm.partition, hm.program, hm.registered,
    hm.part, hm.bindings, hm.rows, hm.inner, hm.commits, hm.link, hm.src, hm.draw, hm.count, ↓reduceIte,
    Option.isSome_some, hc.regs, hc.statements, hc.checks, hc.oneFull, hc.table, hc.shapeOk, hc.coins, hc.pubRows,
    hc.linkRounds, hc.topRows, hc.link, hc.labels]
  by_cases hd : draw.isSome = true
  · simp only [hd, ↓reduceIte, hc.law hd]
  · simp only [hd, Bool.false_eq_true, ↓reduceIte]

/-- **The staged statement's setup is the compiled part's.** Where `readX x` reads `r`, part `s` is `p`, its staged
statement reads back as itself (`decodeRec`, piece C) and the model reads it as `i`, the recursive setup that
`setupO` runs on the staged statement passes with the compiled session, under `recIns_setup_of_compiled`'s
hypotheses. -/
theorem setupO_staged_of_compiled {x : Claims.Msg} {r : XR} {st : Stage} {ms : List (Claims.Msg × Claims.Out)}
    {s : ℕ} {p : PartX} {i : RecIns} {h : VStar.Recursive.InnerHeld} {j : ℕ} {tags0 tags : Tags}
    {circuitB pubB : ByteArray} {partition : Lean.Json} {program : ByteArray}
    {registered : Option (Array Registered.Port)} {bindings : VStar.Bindings}
    {rows : Option (Array Registered.Port)} {draw : Option Lean.Json} {spec : SessionSpec} {sts : Array Stmt}
    (hr : readX x = some r) (hp : r.w.parts[s]? = some p)
    (hdec : decodeRec (partRaw r st x ms s p).encode = some (partRaw r st x ms s p))
    (hi : (partRaw r st x ms s p).ins = .ok i)
    (hm : SetupMatch i h j tags0 tags circuitB pubB partition program registered bindings rows draw)
    (hsparse : h.shape.sparse = none) (hhm : tags.hm96Rows = true)
    (hset : i.base.read.setup = .ok (spec, sts))
    (hregs : ∀ st ∈ sts, checkRegistered st i.base.registered = .ok () ∧ noIdentifier st = .ok ())
    (hfull : Registered.checkFull i.base.registered i.base.read.draw = .ok ())
    (hc : RecChecks h j tags0 tags circuitB partition program registered bindings rows draw sts) :
    setupO (staged st x ms s) = some (i, spec, sts) := by
  have hsu := recIns_setup_of_compiled hm hsparse hhm hset hregs hfull hc
  have hzk : (partRaw r st x ms s p).base.zk = true := by
    rw [← FlockDeploy.Raw.ins_zk (RecRaw.ins_base hi)]
    exact hm.zk
  have hstg : staged st x ms s = (partRaw r st x ms s p).encode := by
    unfold staged
    simp only [hr, hp]
    rw [if_pos hdec]
  have hins : insO (staged st x ms s) = some i := by
    rw [hstg]
    unfold insO
    simp [hdec, hzk, hi, Except.toOption]
  unfold setupO
  simp [hins, hsu, Except.toOption]

/-- **The compiled part's scope check is the model's statement check** at the staged statement's inputs. -/
theorem read_check_of_scope {i : RecIns} {h : VStar.Recursive.InnerHeld} {j : ℕ} {tags0 tags : Tags}
    {circuitB pubB : ByteArray} {partition : Lean.Json} {program : ByteArray}
    {registered : Option (Array Registered.Port)} {bindings : VStar.Bindings}
    {rows : Option (Array Registered.Port)} {draw : Option Lean.Json}
    (hm : SetupMatch i h j tags0 tags circuitB pubB partition program registered bindings rows draw)
    (hscope : ProvedScope.checkOf (ProvedScope.circuitOf tags circuitB []) tags true pubB 1 false registered.isSome =
      .ok ()) :
    i.base.read.check = .ok () := by
  unfold Readable.Read.check
  rw [hm.tags, hm.zk, hm.circuitFile, hm.publicFile, hm.tables, hm.count, hm.publicInputs, hm.readRegistered]
  exact hscope

/-- **What a compiled part shows of its setup**, at the staging `st`, `ms`: the model reads part `s`'s statement
(`partRaw`) as `i` (piece C), `i` holds the compiled part's inputs (`SetupMatch`, piece C), the model's Flock setup and
its registered-reads checks pass with the compiled session (piece D1), the compiled checks block passes (`RecChecks`),
and so does the compiled scope check. -/
def PartSetup (r : XR) (st : Stage) (x : Claims.Msg) (ms : List (Claims.Msg × Claims.Out)) (s : ℕ) (p : PartX) :
    Prop :=
  ∃ (i : RecIns) (h : VStar.Recursive.InnerHeld) (j : ℕ) (tags0 tags : Tags) (circuitB pubB : ByteArray)
    (partition : Lean.Json) (program : ByteArray) (registered : Option (Array Registered.Port))
    (bindings : VStar.Bindings) (rows : Option (Array Registered.Port)) (draw : Option Lean.Json)
    (spec : SessionSpec) (sts : Array Stmt),
    (partRaw r st x ms s p).ins = .ok i ∧
    SetupMatch i h j tags0 tags circuitB pubB partition program registered bindings rows draw ∧
    h.shape.sparse = none ∧ tags.hm96Rows = true ∧
    i.base.read.setup = .ok (spec, sts) ∧
    (∀ st ∈ sts, checkRegistered st i.base.registered = .ok () ∧ noIdentifier st = .ok ()) ∧
    Registered.checkFull i.base.registered i.base.read.draw = .ok () ∧
    RecChecks h j tags0 tags circuitB partition program registered bindings rows draw sts ∧
    ProvedScope.checkOf (ProvedScope.circuitOf tags circuitB []) tags true pubB 1 false registered.isSome = .ok ()

/-- **The compiled statement reads where every compiled part sets up**: where `readXPre x` reads `r` (piece B) and
every part's compiled setup and scope check pass as the model's (`PartSetup`) at one staging, `readX x` reads `r`, and
each part's circuit is V\*'s statement's of its index (`partsVStar`). -/
theorem readX_of_compiled {x : Claims.Msg} {r : XR} {st : Stage} {ms : List (Claims.Msg × Claims.Out)}
    (hr : readXPre x = some r) (hparts : ∀ s p, r.w.parts[s]? = some p → PartSetup r st x ms s p) :
    readX x = some r ∧
      ∃ vs, RecVStar.statements r.shape = .ok vs ∧ partsVStar r.shape r.w.parts vs = true := by
  have hrx : readX x = some r := readX_of_sessions hr fun s p hp => by
    obtain ⟨i, h, j, tags0, tags, circuitB, pubB, partition, program, registered, bindings, rows, draw, spec, sts,
      hi, hm, hsp, hhm, hset, hregs, hfull, hc, hscope⟩ := hparts s p hp
    exact ⟨st, ms, i, spec, sts, hi, recIns_setup_of_compiled hm hsp hhm hset hregs hfull hc,
      read_check_of_scope hm hscope⟩
  obtain ⟨vs, hvs, hpv⟩ := readX_parts hrx
  exact ⟨hrx, vs, hvs, hpv⟩

private theorem bind_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : (x >>= f) = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error e => simp [bind, Except.bind] at h
  | ok a => exact ⟨a, rfl, h⟩

private theorem map_ok {ε α β : Type} {x : Except ε α} {f : α → β} {b : β} (h : (f <$> x) = .ok b) :
    ∃ a, x = .ok a ∧ f a = b := by
  cases x with
  | error e => simp [Functor.map, Except.map] at h
  | ok a => exact ⟨a, rfl, Except.ok.inj h⟩

private theorem need_ok {b : Bool} {why : String} (h : VStar.Recursive.need b why = .ok ()) : b = true := by
  unfold VStar.Recursive.need at h
  split at h
  · assumption
  · first | (simp [throw, throwThe, MonadExceptOf.throw] at h; done) | cases h

private theorem fits_ok {b : ByteArray} {what : String} (h : VStar.Recursive.fits b what = .ok ()) :
    b.size < VStar.Recursive.FIELD_MAX :=
  of_decide_eq_true (need_ok h)

/-- The verifier's registered reads as the compiled part reads them off its `registered` object: none, not complete,
no known rows and no parameters without one; else `registeredOf`'s, with its ports the verifier's own. -/
def PartRegistered (pr : Option Lean.Json) (registered : Option (Array Registered.Port)) (complete : Bool)
    (known : Array (String × Array (Array ℕ × ByteArray))) (regParams : Array String) : Prop :=
  match pr with
  | none => registered = none ∧ complete = false ∧ known = #[] ∧ regParams = #[]
  | some r => ∃ own, VStar.Recursive.registeredOf r = .ok (own, complete, known, regParams) ∧ registered = some own

/-- **What an accepted compiled part shows of its setup.** Where part `j`'s verdict (`VStar.Recursive.partVerdict`)
is `.ok ()`, its readers read (the statement's tags, without plain leaves, the part's circuit and staged public file,
under `FIELD_MAX` bytes, its registered reads and bindings, its record and the record's draw, under hm96-sha512 rows;
its rows are the inner session's, `h.rows`), Flock's session and the verifier's
own checks on it pass (piece D1's inputs: `checkProgram`, `buildSessionOf`, `checkComplete`, `checkCover`,
`KnownRead.check`), the checks block passes (`RecChecks`), and so does the scope check. -/
theorem partVerdict_setup {h : VStar.Recursive.InnerHeld} {j : ℕ} {p : VStar.Recursive.PartOwn}
    {s : VStar.Recursive.PartProof} {bs : Core.Blobs} {v : V Unit}
    (hv : VStar.Recursive.partVerdict h j p s bs = .ok v) (hok : v = .ok ()) :
    ∃ (tags0 : Tags) (circuitB pubB : ByteArray) (registered : Option (Array Registered.Port)) (complete : Bool)
      (known : Array (String × Array (Array ℕ × ByteArray))) (regParams : Array String)
      (bindings : VStar.Bindings) (recordB : ByteArray) (record : String)
      (draw : Lean.Json) (spec : SessionSpec) (sts : Array Stmt),
      Tags.find? VStar.Recursive.STATEMENT = some tags0 ∧ tags0.plainLeaves = false ∧
      bs.get p.circuit s!"part {j}'s circuit" = .ok circuitB ∧
      bs.get s.pub s!"part {j}'s public file" = .ok pubB ∧ pubB.size < VStar.Recursive.FIELD_MAX ∧
      PartRegistered p.registered registered complete known regParams ∧
      RecVStar.ownBindingsOfJson p.bindings = .ok bindings ∧
      bs.get s.record s!"part {j}'s record" = .ok recordB ∧ String.fromUTF8? recordB = some record ∧
      (partTags tags0 circuitB).hm96Rows = true ∧ Witness.recordDraw record = some draw ∧
      Registered.checkProgram (some p.program) registered.isSome = .ok () ∧
      SampledProofs.OneStageZk.buildSessionOf (ProvedScope.circuitOf (partTags tags0 circuitB) circuitB [])
        (partTags tags0 circuitB) circuitB pubB (partCoins tags0) [] 1 (some draw) (some p.partition)
        (some p.program) registered = .ok (spec, sts) ∧
      (complete = true → sts.forM (fun st => Registered.checkComplete st.c (registered.getD #[])) = .ok ()) ∧
      sts.forM (fun st => Registered.checkCover st.c (some p.program) (some p.partition) registered regParams) =
        .ok () ∧
      sts.forM (fun st => KnownRead.check st.c known) = .ok () ∧
      RecChecks h j tags0 (partTags tags0 circuitB) circuitB p.partition p.program registered bindings (some h.rows)
        (some draw) sts ∧
      ProvedScope.checkOf (ProvedScope.circuitOf (partTags tags0 circuitB) circuitB []) (partTags tags0 circuitB) true
        pubB 1 false registered.isSome = .ok () := by
  subst hok
  unfold VStar.Recursive.partVerdict at hv
  rcases ht : Tags.find? VStar.Recursive.STATEMENT with _ | tags0
  · rw [ht] at hv
    first | (simp [throw, throwThe, MonadExceptOf.throw] at hv; done) | cases hv
  simp only [ht] at hv
  obtain ⟨⟨⟩, hpl, hv⟩ := bind_ok hv
  have hpl : tags0.plainLeaves = false := by simpa using need_ok hpl
  obtain ⟨circuitB, hcB, hv⟩ := bind_ok hv
  obtain ⟨pubB, hpB, hv⟩ := bind_ok hv
  obtain ⟨⟨⟩, hfp, hv⟩ := bind_ok hv
  have hfp := fits_ok hfp
  rcases hpr : p.registered with _ | rj <;> simp only [hpr] at hv <;>
    obtain ⟨⟨registered, complete, known, regParams⟩, hreg, hv⟩ := bind_ok hv
  on_goal 1 =>
    have hPR : PartRegistered none registered complete known regParams := by
      simp only [PartRegistered]
      have hreg := Except.ok.inj hreg
      simp only [Prod.mk.injEq] at hreg
      obtain ⟨rfl, rfl, rfl, rfl⟩ := hreg
      exact ⟨rfl, rfl, rfl, rfl⟩
    clear hreg
  on_goal 2 =>
    have hPR : PartRegistered (some rj) registered complete known regParams := by
      simp only [PartRegistered]
      obtain ⟨⟨own, c, k, ps⟩, hro, hreg⟩ := map_ok hreg
      simp only [Prod.mk.injEq] at hreg
      obtain ⟨rfl, rfl, rfl, rfl⟩ := hreg
      exact ⟨own, hro, rfl⟩
    clear hreg
  all_goals
    try dsimp only at hv
    obtain ⟨bindings, hbind, hv⟩ := bind_ok hv
    obtain ⟨recordB, hrecB, hv⟩ := bind_ok hv
    rcases hrec : String.fromUTF8? recordB with _ | record
    · simp only [hrec] at hv
      first | (simp [throw, throwThe, MonadExceptOf.throw] at hv; done) | cases hv
    simp only [hrec] at hv
    generalize hT : Zk.tags (tags0.withCoins (Coins.tree (Zk.coinSpec (tags0.tableNames 1))))
      (Zk.maskLogOf circuitB) 1 = T at hv
    generalize hD : (if T.hm96Rows = true then Witness.recordDraw record else none) = D at hv
    have hv := Except.ok.inj hv
    obtain ⟨proofs, -, hv⟩ := bind_ok hv
    obtain ⟨⟨spec', tabs⟩, hst, hv⟩ := bind_ok hv
    try dsimp only at hv
    obtain ⟨⟨⟩, -, hv⟩ := bind_ok hv
    obtain ⟨⟨⟩, hscope, hv⟩ := bind_ok hv
    obtain ⟨⟨⟩, -, hv⟩ := bind_ok hv
    rcases D with _ | d
    · first
        | (simp [ProvedScope.drawLaw, bind, Except.bind] at hv; done)
        | cases hv
    have hdraw : T.hm96Rows = true ∧ Witness.recordDraw record = some d := by
      cases hh : T.hm96Rows
      · simp [hh] at hD
      · simp only [hh, eq_self_iff_true, ↓reduceIte] at hD
        exact ⟨rfl, hD⟩
    obtain ⟨hhm, hrd⟩ := hdraw
    try dsimp only at hst
    split at hst
    · rename_i spec sts hb0
      have hlaw : ProvedScope.drawLaw true T false (some d) = .ok () := by
        cases hl : ProvedScope.drawLaw true T false (some d) with
        | error e => simp [hl, Except.map] at hst
        | ok u => rfl
      obtain ⟨⟨⟩, hprog, hb1⟩ := bind_ok hb0
      obtain ⟨⟨⟩, hinn, hb2⟩ := bind_ok hb1
      obtain ⟨⟨⟩, hbreg, hb3⟩ := bind_ok hb2
      obtain ⟨⟨⟩, hbstm, hb4⟩ := bind_ok hb3
      obtain ⟨⟨spec0, sts0⟩, hbuild, hb⟩ := bind_ok hb4
      clear hb0 hb1 hb2 hb3 hb4 hst hv
      subst hT
      cases complete <;> simp only [Bool.false_eq_true, eq_self_iff_true, ↓reduceIte] at hb
      on_goal 2 => obtain ⟨⟨⟩, hcomp, hb⟩ := bind_ok hb
      all_goals
        obtain ⟨⟨⟩, hcov, hb⟩ := bind_ok hb
        obtain ⟨⟨⟩, hkn, hb⟩ := bind_ok hb
        obtain ⟨⟨⟩, hchk, hb⟩ := bind_ok hb
        obtain ⟨⟨⟩, hone, hb⟩ := bind_ok hb
        obtain ⟨⟨⟩, htab, hb⟩ := bind_ok hb
        obtain ⟨⟨⟩, hshape, hb⟩ := bind_ok hb
        obtain ⟨⟨⟩, hcoins, hb⟩ := bind_ok hb
        obtain ⟨⟨⟩, hpub, hb⟩ := bind_ok hb
        obtain ⟨⟨⟩, hlr, hb⟩ := bind_ok hb
        obtain ⟨⟨⟩, htop, hb⟩ := bind_ok hb
        obtain ⟨⟨⟩, hlink, hb⟩ := bind_ok hb
        obtain ⟨⟨⟩, hlab, hb⟩ := bind_ok hb
        refine ⟨tags0, circuitB, pubB, registered, _, known, regParams, bindings, recordB, record, d, spec0,
          sts0, rfl, hpl, hcB, hpB, hfp, hPR, hbind, hrecB, hrec, hhm, hrd, hprog, hbuild, ?_, hcov, hkn,
          ⟨hinn, hbreg, hbstm, hchk, hone, htab, hshape, hcoins, hpub, hlr, htop, hlink, hlab, fun _ => hlaw⟩, hscope⟩
      · intro hc
        cases hc
      · intro _
        exact hcomp
    · cases hst

end Verity.Recursive
