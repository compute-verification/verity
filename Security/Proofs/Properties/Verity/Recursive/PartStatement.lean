import Proofs.Flock.Claims.Complete.Encode
import Proofs.Flock.Claims.Recursion.Deploy
import VStarRecursive

/-!
# Piece C of `acc_of_accepts` (PRF-202): V\* part `s`'s statement is the compiled part's inputs

The model's V\* statement of part `s` (`RecDeploy.staged st x ms s`, the encoding of `RecDeploy.partRaw`) is read by the
outer verifier's readers (`RecDeploy.insO`, `RecRaw.ins`) as exactly the recursive inputs compiled part `s`'s
`VStar.Recursive.partVerdict` sets its session up on (`partIns`): the part's circuit and staged public file, its
partition, program, registered reads, bindings and rows, the verifier's inner shape, check 5's source, the inner
circuit's binding, the inner session's commits and link points, and the part's draw. Piece D2 holds `partIns`' setup to
the compiled `setupOf` and so concludes `setupO (staged st x ms s) = some _`.

* **Round trip** (`decode_encode_fits`, `decodeRec_encode_fits`): a statement whose lengths fit their 8-byte words reads
  back as itself, so `staged` is `partRaw`'s encoding (`staged_eq`) and `insO` reads it as `partRaw`'s inputs
  (`insO_staged`). Every field `partRaw` takes from `x` fits, since `x` decodes (`partRaw_fits`); the staged files and
  the transcript, which come from the run, fit by hypothesis.
* **The readers agree** where they are two pieces of code: the two `metaOf`s (`metaOf_eq`), `jsonOf` whatever it names
  (`jsonOf_what`), `Blobs.get` whatever it names (`blobs_get_what`), and the compiled `registeredOf`'s ports
  (`registeredOf_own`). `innerHeld_ok` reads `innerHeld`'s fields off the blobs.
* **The correspondence** (`insO_staged_partIns`) takes `Names`' clauses (PRF-202, piece B) and `Shows`' as hypotheses by
  name, at `readX`'s reading `r` of `x` (`readXPre_spec` relates it to `Names`' `w` and `ri`), and `PartForm`, the
  form compiled part `s` sets up (statement `verity/flock-circuit`, `--zk`, one table, no work, stratified, closure or
  public input words), which `Names` is to carry.
* `vstarAt_eq_staged`: where part `s`'s staged numbers are `nums`' (piece C2), its V\* statement is the staged one.
-/

namespace Verity.Recursive.Part

open Flock
open RecDeploy
open FlockDeploy (Raw encNat encBytes encStr encBool encOpt encList Rd rdTake rdNat rdBytes rdStr rdBool rdOpt rdList
  jsonOf)
open Lean (Json)

/-! ## The round trip -/

/-- **A statement's lengths and numbers fit their 8-byte words** (`encNat`). -/
def RawFits (w : Raw) : Prop :=
  w.statement.toUTF8.size < 2 ^ 64 ∧ w.circuit.size < 2 ^ 64 ∧ w.publicFile.size < 2 ^ 64 ∧
    w.tables.length < 2 ^ 64 ∧ (∀ p ∈ w.tables, p.1.toUTF8.size < 2 ^ 64 ∧ p.2.size < 2 ^ 64) ∧ w.count < 2 ^ 64 ∧
    (∀ b ∈ w.draw, b.size < 2 ^ 64) ∧ (∀ b ∈ w.partition, b.size < 2 ^ 64) ∧ (∀ b ∈ w.program, b.size < 2 ^ 64) ∧
    (∀ n ∈ w.work, n < 2 ^ 64) ∧ (∀ b ∈ w.workTable, b.size < 2 ^ 64) ∧ (∀ n ∈ w.stratified, n < 2 ^ 64) ∧
    (∀ b ∈ w.closure, b.size < 2 ^ 64) ∧ (∀ b ∈ w.registered, b.size < 2 ^ 64) ∧ (∀ b ∈ w.publicInputs, b.size < 2 ^ 64)

/-- **A V\* statement's lengths and numbers fit their 8-byte words**: Flock's statement's, its encoding's length (it is
one field), and each recursive flag's. -/
def RecRawFits (w : RecRaw) : Prop :=
  RawFits w.base ∧ w.base.encode.length < 2 ^ 64 ∧ (∀ b ∈ w.recInner, b.size < 2 ^ 64) ∧ (∀ n ∈ w.recPart, n < 2 ^ 64) ∧
    (∀ b ∈ w.recBindings, b.size < 2 ^ 64) ∧ (∀ b ∈ w.recRows, b.size < 2 ^ 64) ∧
    (∀ b ∈ w.innerCircuit, b.size < 2 ^ 64) ∧ (∀ b ∈ w.innerPublic, b.size < 2 ^ 64) ∧
    (∀ b ∈ w.innerPartition, b.size < 2 ^ 64) ∧ (∀ b ∈ w.innerCoins, b.size < 2 ^ 64) ∧
    (∀ b ∈ w.innerSession, b.size < 2 ^ 64) ∧ (∀ b ∈ w.transcript, b.size < 2 ^ 64) ∧
    (∀ b ∈ w.registration, b.size < 2 ^ 64) ∧ (∀ t ∈ w.innerStatement, t.toUTF8.size < 2 ^ 64) ∧
    (∀ ts ∈ w.innerTables, ts.length < 2 ^ 64 ∧ ∀ p ∈ ts, p.1.toUTF8.size < 2 ^ 64 ∧ p.2.size < 2 ^ 64) ∧
    (∀ b ∈ w.innerDraw, b.size < 2 ^ 64)

/-- **Flock's statement reads back as itself** where it fits. -/
theorem decode_encode_fits {w : Raw} (h : RawFits w) : FlockDeploy.decode w.encode = some w := by
  sorry

/-- **A V\* statement reads back as itself** where it fits. -/
theorem decodeRec_encode_fits {w : RecRaw} (h : RecRawFits w) : decodeRec w.encode = some w := by
  sorry

/-- **What `readXPre` reads**: `x`'s fields (`decodeX`), the inner statement's (`FlockDeploy.decode`), the verifier's
own inner shape and the inner statement's partition, each as `readXPre` reads it. -/
theorem readXPre_spec {x : Claims.Msg} {r : XR} (h : readXPre x = some r) :
    decodeX x = some r.w ∧ FlockDeploy.decode r.w.inner = some r.innerRaw ∧
      (do RecVStar.Inner.ofJson (← jsonOf "rec-inner" r.w.recInner)) = .ok r.shape ∧
      r.innerRaw.partition.mapM (jsonOf "partition") = .ok r.partition := by
  sorry

/-- **Part `s`'s statement fits** where the files the run stages for it do: every field it takes from `x` fits, since
`x` decodes. -/
theorem partRaw_fits {x : Claims.Msg} {r : XR} (hr : readX x = some r) {st : Stage}
    {ms : List (Claims.Msg × Claims.Out)} {s : ℕ} {q : PartX} (hq : r.w.parts[s]? = some q)
    (hbase : ({ q.base with publicFile := st.pub x ms s } : Raw).encode.length < 2 ^ 64)
    (hrec : (st.record x ms).size < 2 ^ 64) (htr : (st.transcript x ms).size < 2 ^ 64) :
    RecRawFits (partRaw r st x ms s q) := by
  sorry

/-- **Part `s`'s staged statement is `partRaw`'s encoding** where it fits. -/
theorem staged_eq {x : Claims.Msg} {r : XR} (hr : readX x = some r) {st : Stage} {ms : List (Claims.Msg × Claims.Out)}
    {s : ℕ} {q : PartX} (hq : r.w.parts[s]? = some q) (hfit : RecRawFits (partRaw r st x ms s q)) :
    staged st x ms s = (partRaw r st x ms s q).encode := by
  unfold staged
  rw [hr]
  simp only [hq, decodeRec_encode_fits hfit, if_true]

/-- **The outer verifier reads part `s`'s staged statement as `partRaw`'s inputs** (`RecRaw.ins`), where it fits and is
a `--zk` statement. -/
theorem insO_staged {x : Claims.Msg} {r : XR} (hr : readX x = some r) {st : Stage}
    {ms : List (Claims.Msg × Claims.Out)} {s : ℕ} {q : PartX} (hq : r.w.parts[s]? = some q)
    (hfit : RecRawFits (partRaw r st x ms s q)) (hzk : q.base.zk = true) :
    insO (staged st x ms s) = (partRaw r st x ms s q).ins.toOption := by
  rw [staged_eq hr hq hfit]
  unfold insO
  simp [decodeRec_encode_fits hfit, partRaw, hzk]

/-- **Where part `s`'s staged numbers are `nums`', its V\* statement is the staged one.** -/
theorem vstarAt_eq_staged {st : Stage} {x : Claims.Msg} {ms : List (Claims.Msg × Claims.Out)} {s : ℕ}
    (hn : numsOf (staged st x ms s) = nums st x s) : vstarAt st x ms s = staged st x ms s := by
  unfold vstarAt
  rw [if_pos hn]

/-! ## The readers agree -/

/-- **The compiled `metaOf` is the model's**: the first `META` line among a circuit file's first nine, as JSON, with the
same refusals. -/
theorem metaOf_eq (b : ByteArray) : VStar.Recursive.metaOf b = RecDeploy.metaOf b := by
  sorry

/-- `jsonOf`'s reading doesn't depend on what it names. -/
theorem jsonOf_what {a a' : String} {b : ByteArray} {j : Json} (h : jsonOf a b = .ok j) : jsonOf a' b = .ok j := by
  unfold jsonOf at h ⊢
  cases hu : String.fromUTF8? b with
  | none => rw [hu] at h; cases h
  | some t =>
    rw [hu] at h ⊢
    cases hp : Json.parse t with
    | error e => rw [hp] at h; cases h
    | ok j' => rw [hp] at h ⊢; exact h

/-- A blob's lookup doesn't depend on what it names. -/
theorem blobs_get_what {bs : Core.Blobs} {d a a' : String} {b : ByteArray} (h : bs.get d a = .ok b) :
    bs.get d a' = .ok b := by
  sorry

/-- **The compiled `registeredOf`'s ports are `Registered.ofJson`'s**, the model's reading of a registered file. -/
theorem registeredOf_own {j : Json}
    {t : Array Registered.Port × Bool × Array (String × Array (Array Nat × ByteArray)) × Array String}
    (h : VStar.Recursive.registeredOf j = .ok t) : Registered.ofJson j = .ok t.1 := by
  sorry

/-- **What `innerHeld` holds**, read off the blobs: the inner circuit, the inner public file, the inner session's
record, the inner statement's draw, the coin server's transcript and the registration, each as text where `innerHeld`
reads text; its shape (`shapeOf`), check 5's source, the inner circuit's binding, the commits and the link points. -/
theorem innerHeld_ok {o : VStar.Recursive.Own} {pr : VStar.Recursive.Proofs} {bs : Core.Blobs}
    {h : VStar.Recursive.InnerHeld} (hh : VStar.Recursive.innerHeld o pr bs = .ok h) :
    ∃ (circuitB pubB recB drawB trB : ByteArray) (record t : String) (reg : Option String),
      bs.get o.circuit "the inner circuit" = .ok circuitB ∧ bs.get pr.pub "the inner public file" = .ok pubB ∧
      VStar.Recursive.shapeOf circuitB pubB o.partition o.program = .ok h.shape ∧
      bs.get pr.record "the inner session's record" = .ok recB ∧ String.fromUTF8? recB = some record ∧
      bs.get pr.draw "the inner statement's draw" = .ok drawB ∧
      h.src = .files { circuit := circuitB, pub := pubB, partition := o.partition, record,
        form := some { statement := VStar.Recursive.INNER_STATEMENT, os := true, tables := [], draw := some drawB } } ∧
      h.bound = (do RecSetup.metaBound (← VStar.Recursive.metaOf circuitB)) ∧
      bs.get pr.transcript "rec inner transcript" = .ok trB ∧ String.fromUTF8? trB = some t ∧
      (pr.registration.mapM fun d => do
          let some s := String.fromUTF8? (← bs.get d "rec inner registration")
            | throw "rec inner registration: not UTF-8"
          pure s : Flock.V (Option String)) = .ok reg ∧
      RecSetup.commits (some h.shape) (pure t) (pure reg) h.bound record = .ok h.commits ∧
      (match Json.parse t with
        | .ok j => RecVStar.innerLink j record
        | .error e => throw s!"rec inner transcript: not JSON: {e}") = .ok h.link := by
  sorry

/-! ## The correspondence -/

/-- **The form compiled part `s` sets up**: the statement `verity/flock-circuit` (`VStar.Recursive.STATEMENT`) under
`--zk`, its coins the keyed coin tree, one table and no lookup tables, and no `--work`, `--work-table`, `--stratified`,
`--closure` or `--public-inputs`. `Names` is to carry it for each part. -/
def PartForm (b : Raw) : Prop :=
  b.statement = VStar.Recursive.STATEMENT ∧ b.zk = true ∧ b.osCoins = false ∧ b.count = 1 ∧ b.tables = [] ∧
    b.work = none ∧ b.workTable = none ∧ b.stratified = none ∧ b.closure = none ∧ b.publicInputs = none

/-- The coins compiled part `s`'s session is set up on: the keyed coin tree of its one table. -/
def partCoins (tags0 : Tags) : Coins := .tree (Zk.coinSpec (tags0.tableNames 1))

/-- **Compiled part `j`'s recursive inputs**, as `partVerdict` hands them to its setup: the statement's tags `tags0`,
the part's circuit `circuitB` and public file `pubB`, its partition and program (`pj`'s), its registered reads, its draw,
the inner session held (`h`: the verifier's inner shape, check 5's source, the inner circuit's binding, the commits and
the link points), `--rec-part j`, the bindings and the rows. -/
def partIns (tags0 : Tags) (circuitB pubB : ByteArray) (pj : VStar.Recursive.PartOwn)
    (registered : Option (Array Registered.Port)) (draw : Option Json) (h : VStar.Recursive.InnerHeld) (j : ℕ)
    (bindings : VStar.Bindings) (rows : Option (Array Registered.Port)) : RecIns where
  base :=
    { stmtTags := tags0
      registered
      read :=
        { I := { tags := tags0.withCoins (partCoins tags0), circuitFile := circuitB, publicFile := pubB,
                 coins := partCoins tags0, count := 1, tables := [], partition := some pj.partition,
                 program := some pj.program, work := none, stratified := none, closure := none }
          zk := true
          draw
          publicInputs := none
          registered := registered.isSome } }
  inner := some h.shape
  part := some j
  bindings := some bindings
  src := some h.src
  bound := h.bound
  commits := some h.commits
  link := some h.link
  rows

/-- **Piece C: part `s`'s staged statement is read as compiled part `s`'s inputs.** Take `readX`'s reading `r` of `x`
and its part `s`, `q`; the compiled check's inner session held (`innerHeld`), and part `s`'s own inputs `pj` with what
the compiled `partVerdict` reads of them (`registered`, `rows`, `bindings`, the statement's tags `tags0`, the part's
circuit and public file). Suppose:
* `Names`' clauses: the inner statement's circuit, public file, partition and draw are the blobs' and `p.own`'s, its
  statement and form those `innerHeld` sets up (`INNER_STATEMENT`, the coin server's coins, no tables), its own inner
  shape `innerHeld`'s, the registration the blobs'; part `s`'s circuit, partition, program, registered reads and rows
  `pj`'s, the bindings `pj`'s, its form `PartForm`, and its draw, read as the verifier reads one, `draw`;
* `Shows`' clauses: part `s`'s staged public file is `pubB`, the inner session's record and the coin server's
  transcript the blobs';
* and part `s`'s statement fits its length words (`partRaw_fits` gives it from the staged files').

Then the outer verifier reads part `s`'s staged statement (`insO`) as `partIns`: compiled part `s`'s inputs. -/
theorem insO_staged_partIns {x : Claims.Msg} {r : XR} (hr : readX x = some r) {st : Stage}
    {ms : List (Claims.Msg × Claims.Out)} {s : ℕ} {q : PartX} (hq : r.w.parts[s]? = some q)
    {o : VStar.Recursive.Own} {pr : VStar.Recursive.Proofs} {bs : Core.Blobs} {h : VStar.Recursive.InnerHeld}
    (hh : VStar.Recursive.innerHeld o pr bs = .ok h)
    {pj : VStar.Recursive.PartOwn} {tags0 : Tags} (htags : Tags.find? VStar.Recursive.STATEMENT = some tags0)
    {circuitB pubB : ByteArray} {registered rows : Option (Array Registered.Port)} {bindings : VStar.Bindings}
    {draw : Option Json}
    (hregC : (pj.registered.mapM VStar.Recursive.registeredOf).map (Option.map (·.1)) = .ok registered)
    (hrowsC : pj.rows.mapM (fun j => ((do pure (← VStar.Recursive.registeredOf j).1 : Flock.V _)).mapError
      (s!"rec rows: " ++ ·)) = .ok rows)
    (hbindC : RecVStar.ownBindingsOfJson pj.bindings = .ok bindings)
    (hIc : bs.get o.circuit "the inner circuit" = .ok r.innerRaw.circuit)
    (hIp : bs.get pr.pub "the inner public file" = .ok r.innerRaw.publicFile)
    (hIstmt : r.innerRaw.statement = VStar.Recursive.INNER_STATEMENT) (hIos : r.innerRaw.osCoins = true)
    (hItab : r.innerRaw.tables = [])
    (hIpart : r.innerRaw.partition.map (jsonOf "partition") = some (.ok o.partition))
    (hIdraw : r.innerRaw.draw = (bs.get pr.draw "the inner statement's draw").toOption)
    (hIshape : (do RecVStar.Inner.ofJson (← jsonOf "rec-inner" r.w.recInner)) = .ok h.shape)
    (hIreg : (pr.registration.mapM fun d => (bs.get d "rec inner registration").toOption) = some r.w.registration)
    (hPform : PartForm q.base) (hPc : q.base.circuit = circuitB)
    (hPpart : q.base.partition.map (jsonOf "partition") = some (.ok pj.partition))
    (hPprog : q.base.program = some pj.program)
    (hPreg : q.base.registered.map (jsonOf "registered") = pj.registered.map Except.ok)
    (hProws : q.rows.map (jsonOf "rec-rows") = pj.rows.map Except.ok)
    (hPbind : jsonOf "rec-bindings" r.w.bindings = .ok pj.bindings)
    (hPdraw : (q.base.draw.mapM (jsonOf "draw")).map (Option.map Flock.norm) = .ok draw)
    (hSp : st.pub x ms s = pubB)
    (hSr : bs.get pr.record "the inner session's record" = .ok (st.record x ms))
    (hSt : bs.get pr.transcript "rec inner transcript" = .ok (st.transcript x ms))
    (hfit : RecRawFits (partRaw r st x ms s q)) :
    insO (staged st x ms s) = some (partIns tags0 circuitB pubB pj registered draw h s bindings rows) := by
  sorry

end Verity.Recursive.Part
