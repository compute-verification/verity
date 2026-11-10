import Proofs.Flock.SpotFlock.Code
import Proofs.Flock.SamplingGeneric.TwoAt

/-!
# SPOT[Flock]'s two-stage model on a run of the two-stage toy

`lake env lean --threads=1 --tstack=KB --run TwoGuard.lean DIR MODE [M2]` in `Security/Proofs`, with Flock's and
SPOT's code loaded native (`--load-dynlib`; `guard_node.py` links it and writes the command): the model's
round-2 check (`SpotFlock.stmtOk`) on the run `guard.export` wrote to `DIR`, at the auditor's copies (`Held`) built from
`DIR/held.json`, the reference statements `DIR/ref-<t>.bin` and their templates' reference reads
`DIR/ref-<t>.reads.json` alone, never from the run's messages or coins. `M2` replaces round 2's message `DIR/m2.bin` (a
negative's).

**What it evaluates.** `stmtOk` is its conjuncts on the registration alone and its conjuncts on the run (`stmtOk_split`):
`regOk`, which is `partition` and `two` (both ignore the statement they are given), and `runOkAt`. This evaluates
`runOkAt`, and of `regOk` only `two`'s conjuncts on the members' files (`twoFilesOk`). It never evaluates `partition`'s
`membersPart` or `two`'s `scopeOk` over `bitsProgs`: both read only `h`, so `regOk h = true` is a fact about the
registration, to be proved from the members' facts rather than decided on a run, and each unrolls the members' circuits
(`Spot.Bits.prog`, `Spot.Bits.blockOf`). The output prints each member's sizes and the time of the unrolling's pieces at
one gate: `Spot.Bits.off` past its last gate, `Spot.Bits.consts`, and `Spot.Bits.owner` at an input gate.

* `check`: `runOkAt` and `twoFilesOk` at the run, with their times and the members' sizes; exit 0 when both accept, 1
  when either refuses, 2 when the reference statements are not in Flock's masked scope (`Held.xS₀_ok`, decided before
  any `Held` exists).
* `diag`: each conjunct of `runOkAt` and `twoFilesOk` on its own, in the order they read them, with its time, then
  `check`.

Every line starts `GUARD `; the last is `GUARD-RESULT {json}`. The model's statement parsers recurse deeper than a
default stack holds: give the interpreter a large `--tstack`.

**The coins.** `stmtOk` reads each coin only through its bytes (`SpotFlock.src`), so `runOkAt` takes the bytes. `src`
itself reads its `2^20` bytes one list lookup at a time, which no evaluation finishes. The bytes here are the ones each
stage's sample read of the challenger's coins (`c0.bin`, `c1.bin`), and zeros past them: the stream's Lean `Select` read
exactly those bytes, and reads its source in order, so it reads none of the zeros.
-/

open Lean (Json)

namespace TwoGuard

open SpotFlock

/-- `SpotFlock.secondDraw` at the stages' bytes `s₀` and `s₁`. -/
def secondDrawAt (h : Held) (s₀ s₁ : ByteArray) : Option Json :=
  match Spot.Select.selectSecond (h.reg.sample s₀) h.reg.ns h.reg.ks s₁ with
  | .ok (draw, _) => some draw
  | .error _ => none

/-- `SpotFlock.ownOf` at the first stage's bytes `s₀`. -/
def ownOfAt (h : Held) (st : Roots) (rec draw : Json) (s₀ : ByteArray) : Spot.SecondStage.Own :=
  { sample := h.reg.sample s₀, parts := some (partsAt h (h.reg.sample s₀).size), interior := rec,
    boundary := st.record, drawn := (Spot.Draw.proved draw).map Int.toNat, links := h.links, wires := h.wires,
    rows := h.rows, refs := refsOf h, refReads := refRegisteredOf h }

/-- `SpotFlock.readsOk` at the stages' bytes. -/
def readsOkAt (h : Held) (t : ℕ) (x m₁ xS : Claims.Msg) (s₀ s₁ : ByteArray) : Bool :=
  match decode x, jsonMsg m₁, secondDrawAt h s₀ s₁, FlockDeploy.decode xS with
  | some st, some rec, some draw, some w =>
    let own := ownOfAt h st rec draw s₀
    Spot.SecondStage.interiorOrderOk own && ((stmtJsonOf h t w).map (codesOk h own t)).getD false && pinOk h t w
  | _, _, _, _ => false

theorem readsOk_eq (h : Held) (t : ℕ) (x m₁ xS : Claims.Msg) (c₀ c₁ : FlockDeploy.Coin) :
    readsOk h t x m₁ xS c₀ c₁ = readsOkAt h t x m₁ xS (src c₀) (src c₁) := rfl

/-- **`stmtOk`'s conjuncts on the run**, at the stages' bytes: the run's statement decodes, `m₂` states one session
statement per template, and template `t`'s has `k₀ t` rounds, is in scope (`sessionOk`) and reads the run's
(`readsOk`). -/
def runOkAt (h : Held) (x m₁ m₂ : Claims.Msg) (s₀ s₁ : ByteArray) : Bool :=
  match decode x, stmtsOf m₂ with
  | some _, some xs =>
    xs.length == h.ms.length &&
      (List.range h.ms.length).all fun t =>
        (FlockDeploy.V true).k (xs.getD t (h.xS₀ t)) == k₀ h t && sessionOk h t (xs.getD t (h.xS₀ t)) &&
          readsOkAt h t x m₁ (xs.getD t (h.xS₀ t)) s₀ s₁
  | _, _ => false

/-- **`stmtOk`'s conjuncts on the registration alone**: the partition and the two-stage design evaluate, neither
reading the statement `s`. -/
def regOk (h : Held) (s : Claims.Sampling.AuditStatement) : Bool := (partition h s).isSome && (two h s).isSome

/-- **`stmtOk` is its registration's conjuncts and its run's**, the run's read through the coins' bytes. -/
theorem stmtOk_split (h : Held) (x m₁ m₂ : Claims.Msg) (c₀ c₁ : FlockDeploy.Coin) :
    stmtOk h x m₁ m₂ c₀ c₁ = ((parse h x).any (regOk h) && runOkAt h x m₁ m₂ (src c₀) (src c₁)) := by
  unfold stmtOk runOkAt parse
  cases decode x <;> cases stmtsOf m₂ <;> simp [regOk, readsOk_eq, Bool.and_assoc]

/-- **`two`'s conjuncts on the members' files**, at the members `ms` (`h.ms`): the replay layout and the first stage's
design exist, the layout fits the members' files (`filesOk`), and each template's reference statement reads as a
member's (`templateOk`). Not `scopeOk`, whose program is the members' circuits unrolled (`bitsProgs`). -/
def twoFilesOk (h : Held) (ms : Spot.Bits.Members) : Bool :=
  match (Spot.Slots.layout h.reg.first.program h.reg.first.query).toOption, replayDesign h with
  | some l, some _ => filesOk h.reg ms l h.k && (List.range ms.length).all (templateOk h)
  | _, _ => false

/-! ## The auditor's copies -/

/-- Template `t`'s reference statement among `refs`, the last template's past them. -/
def refAt (refs : Array Claims.Msg) (t : ℕ) : Claims.Msg := refs.getD (min t (refs.size - 1)) []

theorem refAt_ok (refs : Array Claims.Msg)
    (hz : 0 < refs.size ∧ (List.range refs.size).all (fun t => zkOk (refs.getD t [])) = true) :
    ∀ t, zkOk (refAt refs t) = true := fun t =>
  List.all_eq_true.mp hz.2 (min t (refs.size - 1)) (List.mem_range.mpr (by have := hz.1; omega))

def ok {α : Type} (what : String) : Except String α → IO α
  | .ok a => pure a
  | .error e => throw (IO.userError s!"{what}: {e}")

def field (j : Json) (k : String) : Json := (j.getObjVal? k).toOption.getD .null

def natsOf (what : String) (j : Json) : IO (List ℕ) := do
  (← ok what j.getArr?).toList.mapM fun n => ok what n.getNat?

def hexOf (what : String) (j : Json) : IO Claims.Msg := do
  return (← ok what (_root_.Flock.unhex (← ok what j.getStr?))).data.toList

/-- **The auditor's copies**, from `held.json` (`held`), the reference statements `refs` and their templates' reference
reads `reads` (`Held.refReads`, null past them), behind the decision that each reference statement is in Flock's masked
scope (`refAt_ok`); none where one isn't. The first stage's registration
is the boundary registration's program, query and work table with the stream's rule over them (`Select.resolve`, as
the `select` piece takes it), the fields of `Registration.Verifier` the model doesn't read empty; `blockOf` is
`filesBlockOf` at the members (`Held.ms`), and no proof block is checked whatever the stages sample (`always`): every
slot of the toy's boundary layout is a request's. -/
def heldOf (held : Json) (refs : Array Claims.Msg) (reads : Array Json) : IO (Option Held) := do
  let first := field held "first"
  let queryJ := field first "query"
  let P ← ok "program" do Spot.Program.parse (← _root_.Flock.canon (field first "program")).toUTF8
  let rule ← ok "rule" do Spot.Select.resolve (← Spot.Select.Rule.ofJson (field first "rule")) P queryJ
  let table : Option Json := match field first "work_table" with
    | .null => none
    | t => some t
  let ns ← natsOf "ns" (field held "ns")
  let ks ← natsOf "ks" (field held "ks")
  let vus ← (← ok "vus" (field held "vus").getArr?).toList.mapM fun v => return (← natsOf "vus" v).toArray
  let names ← (← ok "names" (field held "names").getArr?).toList.mapM fun n => ok "names" n.getStr?
  let q := field held "query"
  let query : Claims.Sampling.Query :=
    ⟨← ok "query" (field q "name").getStr?, ← ok "query" (field q "version").getNat?, ← hexOf "query" (field q "params")⟩
  let desc ← hexOf "desc" (field held "desc")
  let k ← natsOf "k" (field held "k")
  let links := Spot.SecondStage.linksOf? (field held "links")
  let wires := Spot.SecondStage.wiresOf? (field held "wires")
  let rows ← ok "rows" ((Spot.SecondStage.rowsOf? (field held "rows")).elim (.error "not a list of rows") .ok)
  if hz : 0 < refs.size ∧ (List.range refs.size).all (fun t => zkOk (refs.getD t [])) = true then
    let verifier : Spot.Registration.Verifier :=
      { program := P, query := queryJ, law := .null, workTable := table, scheme := "", ports := #[], leafLayer := none,
        anchors := #[], log := #[], partition := none }
    let h₀ : Held :=
      { desc, vus, query, k, names, links, wires, rows, xS₀ := refAt refs, xS₀_ok := refAt_ok refs hz,
        refReads := fun t => reads.getD t .null, frame := SamplingDeploy.toyCode,
        reg := { first := { verifier, rule, closure := none }, ns := ns.toArray, ks := ks.toArray,
                 blockOf := fun _ _ _ => 0, always := fun _ => false } }
    let ms := h₀.ms
    return some { h₀ with reg := { h₀.reg with blockOf := Properties.Spot.filesBlockOf ms h₀.reg.ns } }
  else
    return none

/-! ## Running it -/

def say (line : String) : IO Unit := do
  IO.println s!"GUARD {line}"
  (← IO.getStdout).flush

/-- `f ()`, said with `fmt` and its time in ms, which it returns beside it. -/
def timed {α : Type} (label : String) (f : Unit → α) (fmt : α → String) : IO (α × ℕ) := do
  let t0 ← IO.monoMsNow
  let a ← IO.lazyPure f
  say s!"{label} = {fmt a}"
  let ms := (← IO.monoMsNow) - t0
  say s!"{label}: {ms} ms"
  return (a, ms)

def b (x : Bool) : String := toString x

def readMsg (p : System.FilePath) : IO Claims.Msg := return (← IO.FS.readBinFile p).data.toList

/-- **Why a reference statement is out of Flock's masked scope**: the first of `zkOk`'s steps it fails (`setupOf`:
decode, the verifier's parsers `Raw.ins`, `Ins.setup`; then `zkOk`'s bounds), with its error; none where `zkOk` holds. -/
def refWhy (xS : Claims.Msg) : Option String :=
  match FlockDeploy.decode xS with
  | none => some "decode: the bytes don't decode as a statement"
  | some w =>
    if !w.zk then some "the statement is not --zk" else
    match w.ins with
    | .error e => some s!"parse: {e}"
    | .ok s =>
      match s.setup with
      | .error e => some s!"setup: {e}"
      | .ok _ => if zkOk xS then none else some "zkOk: it sets up, outside the masked scope's bounds"

/-- A stage's bytes, zeros past `n` of them up to `L`. -/
def coinBytes (p : System.FilePath) : IO ByteArray := do
  let r ← IO.FS.readBinFile p
  return r ++ ⟨Array.replicate (L - r.size) 0⟩

/-- **Each member's key program** for its units (`Spot.Bits.keyProg`): its inputs and rows, its lowered gates (one an
input, three a row: `Spot.Bits.off` past its last gate), the walks of every gate its lowering makes (`Spot.Bits.lower`
calls `off` once per gate a row reads and twice more a row), and the time of one walk; and the time of the constants'
answer at one input gate (`Spot.Bits.consts`, which the lowering asks at each input gate) and of that gate's owner
(`Spot.Bits.owner`). -/
def members (ms : Spot.Bits.Members) : IO (List Json) :=
  (List.range ms.length).mapM fun t => do
    let (c, vs) := ms.getD t default
    let (kp, _) ← timed s!"m{t} keyProg" (fun _ => Spot.Bits.keyProg c vs.size) fun kp =>
      s!"{kp.inputs} inputs, {kp.rows.size} rows"
    let (lowered, walkMs) ← timed s!"m{t} off past its last gate" (fun _ => Spot.Bits.off kp kp.size) toString
    let walks := kp.rows.foldl (fun s r => s + r.1.length + r.2.length + 2) 0
    let κ := Spot.Bits.consts c vs.size
    let (_, constsMs) ← timed s!"m{t} consts at input gate 2" (fun _ => κ 2) toString
    let (_, ownerMs) ← timed s!"m{t} owner of input gate 2" (fun _ => Spot.Bits.owner kp 2) toString
    return Json.mkObj [("units", Lean.toJson vs.size), ("inputs", Lean.toJson kp.inputs),
      ("rows", Lean.toJson kp.rows.size), ("lowered", Lean.toJson lowered), ("lowering_walks", Lean.toJson walks),
      ("walk_ms", Lean.toJson walkMs), ("consts_ms", Lean.toJson constsMs), ("owner_ms", Lean.toJson ownerMs)]

/-- Each conjunct of `runOkAt` and `twoFilesOk` on its own, in their order. -/
def diag (h : Held) (ms : Spot.Bits.Members) (x m₁ m₂ : Claims.Msg) (s₀ s₁ : ByteArray) : IO Unit := do
  let (st?, _) ← timed "decode x" (fun _ => decode x) (fun s => b s.isSome)
  let (xs?, _) ← timed "stmtsOf" (fun _ => stmtsOf m₂) fun xs => toString (xs.map (·.map List.length))
  let xs := xs?.getD []
  let (rec?, _) ← timed "jsonMsg m₁" (fun _ => jsonMsg m₁) (fun r => b r.isSome)
  let (sd, _) ← timed "secondDraw" (fun _ => secondDrawAt h s₀ s₁) fun d => (d.map (·.compress)).getD "none"
  let _ ← timed "replay sample" (fun _ => h.reg.sample s₀) toString
  let (refs, _) ← timed "refsOf" (fun _ => refsOf h) (fun r => toString r.size)
  let (refReads, _) ← timed "refRegisteredOf" (fun _ => refRegisteredOf h) fun r =>
    toString (r.toList.map fun e => (e.1, e.2 != Json.null))
  for t in List.range ms.length do
    let xS := xs.getD t (h.xS₀ t)
    let _ ← timed s!"t{t} rounds = k₀" (fun _ => ((FlockDeploy.V true).k xS, k₀ h t)) toString
    let _ ← timed s!"t{t} zkOk" (fun _ => zkOk xS) b
    let _ ← timed s!"t{t} shapes" (fun _ => (shapeOf xS, shapeOf (h.xS₀ t))) toString
    let _ ← timed s!"t{t} scopeCheck" (fun _ => FlockDeploy.scopeCheck true xS) b
    let _ ← timed s!"t{t} hiddenOf" (fun _ => hiddenOf xS) b
    let (w?, _) ← timed s!"t{t} decode" (fun _ => FlockDeploy.decode xS) (fun w => b w.isSome)
    match st?, rec?, sd, w? with
    | some st, some rec, some draw, some w =>
      let own : Spot.SecondStage.Own :=
        { sample := h.reg.sample s₀, parts := some (partsAt h (h.reg.sample s₀).size), interior := rec,
          boundary := st.record, drawn := (Spot.Draw.proved draw).map Int.toNat, links := h.links, wires := h.wires,
          rows := h.rows, refs, refReads }
      let _ ← timed s!"t{t} interiorOrderOk" (fun _ => Spot.SecondStage.interiorOrderOk own) b
      let (S?, _) ← timed s!"t{t} stmtJsonOf" (fun _ => stmtJsonOf h t w) (fun S => b S.isSome)
      if let some S := S? then
        let _ ← timed s!"t{t} codes, as codesOk reads them" (fun _ =>
          Spot.SecondStage.stmtCodes { own with verified := S.checked.toArray }
            (Spot.SecondStage.byTemplate own.interior)
            (Spot.SecondStage.posOf (partsAt h own.sample.size) (h.names.getD t ""))
            (partAt h own.sample.size t) S) toString
        let _ ← timed s!"t{t} codesOk" (fun _ => codesOk h own t S) b
      let _ ← timed s!"t{t} pinOk" (fun _ => pinOk h t w) b
    | _, _, _, _ => say s!"t{t}: readsOk has no input (decode x, jsonMsg m₁, secondDraw or decode xS is none)"
  match (Spot.Slots.layout h.reg.first.program h.reg.first.query).toOption, replayDesign h with
  | some l, some _ =>
    let _ ← timed "filesOk" (fun _ => filesOk h.reg ms l h.k) b
  | _, _ => say "layout or replayDesign: none"
  for t in List.range ms.length do
    let _ ← timed s!"t{t} templateOk" (fun _ => templateOk h t) b

def main (args : List String) : IO UInt32 := do
  let (dir :: mode :: rest) := args
    | do
      IO.eprintln "usage: TwoGuard.lean DIR check|diag [M2]"
      return 2
  let dir : System.FilePath := dir
  let t0 ← IO.monoMsNow
  let x ← readMsg (dir / "x.bin")
  let m₁ ← readMsg (dir / "m1.bin")
  let m₂ ← readMsg (rest.head?.map (System.FilePath.mk ·) |>.getD (dir / "m2.bin"))
  let s₀ ← coinBytes (dir / "c0.bin")
  let s₁ ← coinBytes (dir / "c1.bin")
  let held ← ok "held.json" (Json.parse (← IO.FS.readFile (dir / "held.json")))
  let mut refs : Array Claims.Msg := #[]
  let mut reads : Array Json := #[]
  for t in [0:(← ok "names" (field held "names").getArr?).size] do
    refs := refs.push (← readMsg (dir / s!"ref-{t}.bin"))
    let r := dir / s!"ref-{t}.reads.json"
    unless (← r.pathExists) do
      throw (IO.userError s!"{r}: no reference reads; an export from before they left the reference statements \
        (python3 benchmarks/two_stage/guard.py refs {dir})")
    reads := reads.push (← ok r.toString (Json.parse (← IO.FS.readFile r)))
  say s!"read: x {x.length} B, m1 {m₁.length} B, m2 {m₂.length} B, {refs.size} references, {(← IO.monoMsNow) - t0} ms"
  let t1 ← IO.monoMsNow
  let some h ← heldOf held refs reads
    | do
      let whys := (List.range refs.size).filterMap fun t => (refWhy (refs.getD t [])).map (s!"ref-{t}: " ++ ·)
      for w in whys do say w
      let res := Json.mkObj [("held", Lean.toJson false),
        ("why", Lean.toJson "a reference statement is not in Flock's masked scope"), ("refs", Lean.toJson whys)]
      IO.println s!"GUARD-RESULT {res.compress}"
      return 2
  say s!"held: {(← IO.monoMsNow) - t1} ms"
  let (ms, _) ← timed "ms (each reference statement read)" (fun _ => h.ms) fun ms => toString (ms.map (·.2.size))
  let sizes ← members ms
  if mode == "diag" then diag h ms x m₁ m₂ s₀ s₁
  let (run, runMs) ← timed "runOk" (fun _ => runOkAt h x m₁ m₂ s₀ s₁) b
  let (files, filesMs) ← timed "twoFilesOk" (fun _ => twoFilesOk h ms) b
  let res := Json.mkObj [("held", Lean.toJson true), ("runOk", Lean.toJson run), ("runOk_ms", Lean.toJson runMs),
    ("twoFilesOk", Lean.toJson files), ("twoFilesOk_ms", Lean.toJson filesMs), ("members", Json.arr sizes.toArray),
    ("not_evaluated", Lean.toJson "partition's membersPart and two's scopeOk, over bitsProgs")]
  IO.println s!"GUARD-RESULT {res.compress}"
  return if run && files then 0 else 1

end TwoGuard

def main (args : List String) : IO UInt32 := TwoGuard.main args
