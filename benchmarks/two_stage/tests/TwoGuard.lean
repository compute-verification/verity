import Proofs.Flock.SpotFlock.Code
import Proofs.Flock.SamplingGeneric.TwoAt

/-!
# SPOT[Flock]'s two-stage model on a run of the two-stage toy

`lake env lean --run TwoGuard.lean DIR MODE [M2]`, in `Security/Proofs` with `Proofs.Flock.SpotFlock.Code` built: the
model's round-2 check (`SpotFlock.stmtOk`) on the run `guard.export` wrote to `DIR`, at the auditor's copies (`Held`)
built from `DIR/held.json` and the reference statements `DIR/ref-<t>.bin` alone, never from the run's messages or coins.
`M2` replaces round 2's message `DIR/m2.bin` (a negative's).

* `check`: `stmtOkAt` at the run, and its time; exit 0 when it accepts, 1 when it refuses, 2 when the reference
  statements are not in Flock's masked scope (`Held.xS₀_ok`, decided before any `Held` exists).
* `diag`: each conjunct of `stmtOk` on its own, in the order `stmtOk` reads them, with its time, then `stmtOkAt`.

Every line starts `GUARD `; the last is `GUARD-RESULT {json}`.

**The coins.** `stmtOk` reads each coin only through its bytes (`SpotFlock.src`), so `stmtOkAt` takes the bytes, and
`stmtOk_eq` says it is `stmtOk` at every pair of coins with those bytes. `src` itself reads its `2^20` bytes one list
lookup at a time, which no evaluation finishes. The bytes here are the ones each stage's sample read of the challenger's
coins (`c0.bin`, `c1.bin`), and zeros past them: the stream's Lean `Select` read exactly those bytes, and reads its source
in order, so it reads none of the zeros.
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
  { sample := h.reg.sample s₀, parts := none, interior := rec, boundary := st.record,
    drawn := (Spot.Draw.proved draw).map Int.toNat, links := h.links, refs := refsOf h }

/-- `SpotFlock.readsOk` at the stages' bytes. -/
def readsOkAt (h : Held) (t : ℕ) (x m₁ xS : Claims.Msg) (s₀ s₁ : ByteArray) : Bool :=
  match decode x, jsonMsg m₁, secondDrawAt h s₀ s₁, FlockDeploy.decode xS with
  | some st, some rec, some draw, some w =>
    let own := ownOfAt h st rec draw s₀
    Spot.SecondStage.interiorOrderOk own && ((stmtJsonOf h t w).map (codesOk h own t)).getD false && pinOk h t w
  | _, _, _, _ => false

/-- `SpotFlock.stmtOk` at the stages' bytes. -/
def stmtOkAt (h : Held) (x m₁ m₂ : Claims.Msg) (s₀ s₁ : ByteArray) : Bool :=
  match parse h x, stmtsOf m₂ with
  | some s, some xs =>
    (partition h s).isSome && (two h s).isSome && xs.length == h.ms.length &&
      (List.range h.ms.length).all fun t =>
        (FlockDeploy.V true).k (xs.getD t (h.xS₀ t)) == k₀ h t && sessionOk h t (xs.getD t (h.xS₀ t)) &&
          readsOkAt h t x m₁ (xs.getD t (h.xS₀ t)) s₀ s₁
  | _, _ => false

/-- **`stmtOk` reads its coins only through their bytes.** -/
theorem stmtOk_eq (h : Held) (x m₁ m₂ : Claims.Msg) (c₀ c₁ : FlockDeploy.Coin) :
    stmtOk h x m₁ m₂ c₀ c₁ = stmtOkAt h x m₁ m₂ (src c₀) (src c₁) := rfl

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

/-- **The auditor's copies**, from `held.json` (`held`) and the reference statements `refs`, behind the decision that
each reference statement is in Flock's masked scope (`refAt_ok`); none where one isn't. The first stage's registration
is the boundary registration's program, query and work table with the stream's rule over them (`Select.resolve`, as
the `select` piece takes it), the fields of `Registration.Verifier` the model doesn't read empty; `blockOf` is
`filesBlockOf` at the members (`Held.ms`), and no proof block is checked whatever the stages sample (`always`): every
slot of the toy's boundary layout is a request's. -/
def heldOf (held : Json) (refs : Array Claims.Msg) : IO (Option Held) := do
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
  if hz : 0 < refs.size ∧ (List.range refs.size).all (fun t => zkOk (refs.getD t [])) = true then
    let verifier : Spot.Registration.Verifier :=
      { program := P, query := queryJ, law := .null, workTable := table, scheme := "", ports := #[], leafLayer := none,
        anchors := #[], log := #[], partition := none }
    let h₀ : Held :=
      { desc, vus, query, k, names, links, xS₀ := refAt refs, xS₀_ok := refAt_ok refs hz,
        frame := SamplingDeploy.toyCode,
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
  let a := f ()
  say s!"{label} = {fmt a}"
  let ms := (← IO.monoMsNow) - t0
  say s!"{label}: {ms} ms"
  return (a, ms)

def b (x : Bool) : String := toString x

def readMsg (p : System.FilePath) : IO Claims.Msg := return (← IO.FS.readBinFile p).data.toList

/-- A stage's bytes, zeros past `n` of them up to `L`. -/
def coinBytes (p : System.FilePath) : IO ByteArray := do
  let r ← IO.FS.readBinFile p
  return r ++ ⟨Array.replicate (L - r.size) 0⟩

/-- Each conjunct of `stmtOk` on its own, in `stmtOk`'s order. -/
def diag (h : Held) (x m₁ m₂ : Claims.Msg) (s₀ s₁ : ByteArray) : IO Unit := do
  let (ms, _) ← timed "ms (the members, each reference statement read)" (fun _ => h.ms) fun ms =>
    toString (ms.map (·.2.size))
  let (some s, _) ← timed "parse" (fun _ => parse h x) (fun s => b s.isSome) | return
  let _ ← timed "partition" (fun _ => (partition h s).isSome) b
  let (xs?, _) ← timed "stmtsOf" (fun _ => stmtsOf m₂) fun xs => toString (xs.map (·.length))
  let xs := xs?.getD []
  let (sd, _) ← timed "secondDraw" (fun _ => secondDrawAt h s₀ s₁) fun d => (d.map (·.compress)).getD "none"
  let (sample, _) ← timed "replay sample" (fun _ => h.reg.sample s₀) toString
  for t in List.range ms.length do
    let xS := xs.getD t (h.xS₀ t)
    let _ ← timed s!"t{t} rounds = k₀" (fun _ => ((FlockDeploy.V true).k xS, k₀ h t)) toString
    let _ ← timed s!"t{t} zkOk" (fun _ => zkOk xS) b
    let _ ← timed s!"t{t} shapes" (fun _ => (shapeOf xS, shapeOf (h.xS₀ t))) toString
    let _ ← timed s!"t{t} scopeCheck" (fun _ => FlockDeploy.scopeCheck true xS) b
    let _ ← timed s!"t{t} hiddenOf" (fun _ => hiddenOf xS) b
    let (w?, _) ← timed s!"t{t} decode" (fun _ => FlockDeploy.decode xS) (fun w => b w.isSome)
    match decode x, jsonMsg m₁, sd, w? with
    | some st, some rec, some draw, some w =>
      let own := ownOfAt h st rec draw s₀
      let _ ← timed s!"t{t} interiorOrderOk" (fun _ => Spot.SecondStage.interiorOrderOk own) b
      let (S?, _) ← timed s!"t{t} stmtJsonOf" (fun _ => stmtJsonOf h t w) (fun S => b S.isSome)
      if let some S := S? then
        let _ ← timed s!"t{t} codes" (fun _ =>
          Spot.SecondStage.stmtCodes own (Spot.SecondStage.byTemplate own.interior)
            (Spot.SecondStage.posOf (partsAt h own.sample.size) (h.names.getD t ""))
            (partAt h own.sample.size t) S) toString
      let _ ← timed s!"t{t} pinOk" (fun _ => pinOk h t w) b
    | _, _, _, _ => say s!"t{t}: readsOk has no input (decode x, jsonMsg m₁, secondDraw or decode xS is none)"
  let _ ← timed "sample size" (fun _ => sample.size) toString
  for t in List.range ms.length do
    let _ ← timed s!"t{t} templateOk" (fun _ => templateOk h t) b
  match (Spot.Slots.layout h.reg.first.program h.reg.first.query).toOption, replayDesign h with
  | some l, some d =>
    let _ ← timed "filesOk" (fun _ => filesOk h.reg ms l h.k) b
    let (C, _) ← timed "bitsProgs (gates)" (fun _ => Properties.Spot.bitsProgs ms) fun C => toString C.gates.length
    let _ ← timed "topoOk" (fun _ => topoOk C) b
    let _ ← timed "partitionsOk" (fun _ => partitionsOk (Properties.Spot.membersPart ms) C) b
    let _ ← timed "scopeOk" (fun _ => scopeOk C (Properties.Spot.membersPart ms)
      ⟨Properties.Spot.filesReplay ms h.reg.ns l, d, h.k⟩) b
  | _, _ => say "layout or replayDesign: none"
  let _ ← timed "two" (fun _ => (two h s).isSome) b

def main (args : List String) : IO UInt32 := do
  let (dir :: mode :: rest) := args
    | do
      IO.eprintln "usage: lake env lean --run TwoGuard.lean DIR check|diag [M2]"
      return 2
  let dir : System.FilePath := dir
  let t0 ← IO.monoMsNow
  let x ← readMsg (dir / "x.bin")
  let m₁ ← readMsg (dir / "m1.bin")
  let m₂ ← readMsg (rest.head?.map (System.FilePath.mk ·) |>.getD (dir / "m2.bin"))
  let s₀ ← coinBytes (dir / "c0.bin")
  let s₁ ← coinBytes (dir / "c1.bin")
  let mut refs : Array Claims.Msg := #[]
  for t in [0:4] do
    if (← (dir / s!"ref-{t}.bin").pathExists) then refs := refs.push (← readMsg (dir / s!"ref-{t}.bin"))
  let held ← ok "held.json" (Json.parse (← IO.FS.readFile (dir / "held.json")))
  say s!"read: x {x.length} B, m1 {m₁.length} B, m2 {m₂.length} B, {refs.size} references, {(← IO.monoMsNow) - t0} ms"
  let t1 ← IO.monoMsNow
  let some h ← heldOf held refs
    | do
      IO.println "GUARD-RESULT {\"held\": false, \"why\": \"a reference statement is not in Flock's masked scope\"}"
      return 2
  say s!"held: {(← IO.monoMsNow) - t1} ms"
  if mode == "diag" then diag h x m₁ m₂ s₀ s₁
  let (acc, ms) ← timed "stmtOk" (fun _ => stmtOkAt h x m₁ m₂ s₀ s₁) b
  let res := Json.mkObj [("held", Lean.toJson true), ("stmtOk", Lean.toJson acc), ("stmtOk_ms", Lean.toJson ms)]
  IO.println s!"GUARD-RESULT {res.compress}"
  return if acc then 0 else 1

end TwoGuard

def main (args : List String) : IO UInt32 := TwoGuard.main args
