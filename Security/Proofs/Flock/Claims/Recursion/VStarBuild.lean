import Proofs.Flock.Claims.Recursion.VStarWit
import Proofs.Flock.Claims.Recursion.SlotsCommit
import VStar.In

/-!
# V\*'s outer witness builder, the deployed one (`vstarWit`), and the reference prover's pieces it gives (`refPartsAt`)

The reference prover hands V\* part `s`'s session the outer witness `RefParts.wit` builds. Here it is a definition, the
deployed builder `verity_flock.rec_vstage.stage` followed by `flock-circuit --zk`'s trace:

* **What the inner prover records** (`InnerRec`): the rows it commits at the round after each history (`rows`: a
  round's tops, then the row it absorbs), and after the inner rounds the rest of what the stager reads (`rest`, a
  `VStar.In.Src`: the openings, the running sums' salts, `κ`, `eq(α)`, the `v`s and the fresh salts).
* **The inner prover** (`InnerRec.P`): at each round, hm96's commit string of each row's `sha512/row/v1` digest
  (`digestOf`) under that round's and row's salt on its tape, in order (`commitThen`).
* **The stager's inputs** (`srcOf`): `rest`, its message rounds and tops the rows the inner prover committed
  (`roundRows`): each absorbing round's last row at the rounds the rep streams commit them (`msgsOf`, `msgRound`), level
  0's tops at round 0 and each later level's at the round committing it (`topsOf`), each under its salt.
* **Each part's port inputs** (`VStar.In.partIn`, difftested against the stager's `inst.bin`), at the part's
  V\* statement and its circuit's program (`partAt`, `progOf`).
* **The trace** (`vstarWit`): an assigner `A` on the part's circuit at those inputs. What the obligations read of the
  trace is where it places each port's rows and salts (`Places`: `rowHidden`, `hmSalt`).

`refPartsAt x` fixes the compiled statement: `RefParts.wit` reads no statement, and the builder reads part `s`'s
statement and the rep streams' schedule off it. A deployed record over `refPartsAt x` is the deployed one at runs on
`x`. The deployed `Brings` is `withReg` of it (`RegSetup`); `withReg` changes only `setup`, so `ref` is this.
-/

namespace RecDeploy.VStarBuild

open _root_.Flock
open Claims (Msg)
open Claims.Recursion.VStarB (formBit readForms)

/-- **Part `s` at the compiled statement `x`**: its V\* statement and its circuit. -/
def partAt (x : Msg) (s : ℕ) : Option (_root_.VStar.Stmt × Circuit) := do
  let r ← readX x
  let st ← (RecVStar.statements r.shape).toOption.bind (·[s]?)
  let c ← r.w.parts[s]?.bind partCircuit
  pure (st, c)

/-- A circuit's statement program's SHA-512 hex, as its META names it (`RecVStar.bindingsOf`); empty where it names
none. -/
def progOf (c : Circuit) : String := (c.metaJ.getObjValAs? String "program_sha512").toOption.getD ""

/-- **A row's message**: `sha512/row/v1`'s prefix at its word count, then its words' bits as bytes. -/
def rowMsg (ws : Array ℕ) : Msg :=
  (HmRow.rowPrefix ws.size).data.toList ++ Claims.Recursion.VStar.bytesOf (_root_.VStar.In.wordBits ws)

/-- **A row's digest**, `SHA-512` of its message. -/
def digestOf (ws : Array ℕ) : Fin 64 → UInt8 := fun i => (FlockDeploy.sha512 (rowMsg ws)).getD i.val 0

/-- A salt's bytes. -/
def saltBytes (y : Salt) : ByteArray := ⟨Array.ofFn y⟩

/-- **What the inner prover records**: the rows it commits at the round after each history (`rows`, u16 words), in the
order its message lays their commit strings (a round's tops, then the row it absorbs), and after the inner rounds the
rest of what the stager reads (`rest`), whose message rounds and tops `srcOf` replaces by the rows committed. -/
structure InnerRec where
  R : Type
  [ne : Nonempty R]
  rows : Claims.Flock.Wit → R → List (Msg × Msg) → List (Array ℕ)
  rest : Claims.Flock.Wit → R → List (Msg × Msg) → _root_.VStar.In.Src

attribute [instance] InnerRec.ne

/-- The inner prover's tape: `R`, and a salt for each round and each row of it. -/
abbrev InnerRec.Tape (I : InnerRec) : Type := I.R × (ℕ → ℕ → Salt)

/-- **The inner prover**: at each round, hm96's commit string of each row it commits there (`digestOf`), under that
round's and row's salt, in order (`commitThen`). -/
def InnerRec.P (I : InnerRec) (w : Claims.Flock.Wit) : Claims.Prover I.Tape :=
  ⟨fun _ h ρ => commitThen (fun cs => .ret cs.flatten)
    ((I.rows w ρ.1 h).mapIdx fun j ws => (digestOf ws, ρ.2 h.length j)) []⟩

/-- **Inner round `i`'s rows** off the inner rounds `hist`: the rows committed at the history before it, each under its
salt. -/
def roundRows (I : InnerRec) (w : Claims.Flock.Wit) (ρ : I.Tape) (hist : List (Msg × Msg)) (i : ℕ) :
    List _root_.VStar.In.Row :=
  (I.rows w ρ.1 (hist.take i)).mapIdx fun j ws => ⟨ws, saltBytes (ρ.2 i j)⟩

/-- **The message rounds' rows** (`VStar.In.Src.msgs`): round `k`'s, rep `r`'s, the last row of inner round
`msgRound rs r k + 1` (the round's absorbed row, after its tops); an empty row where there is none. -/
def msgsOf (rs : Array RecVStar.SchedRound) (rr : ℕ → List _root_.VStar.In.Row) :
    Array (Array _root_.VStar.In.Row) :=
  ((List.range (RecVStar.commitRounds rs).length).map fun k =>
    ((List.range 2).map fun r =>
      ((msgRound rs r k).bind fun j => (rr (j + 1)).getLast?).getD ⟨#[], ByteArray.empty⟩).toArray).toArray

/-- A row's words as bytes, each word little-endian (`VStar.In.wordsOf`'s inverse). -/
def wordsBytes (ws : Array ℕ) : ByteArray :=
  ⟨(Array.range (2 * ws.size)).map fun i =>
    if i % 2 = 0 then (ws[i / 2]! % 256).toUInt8 else (ws[i / 2]! / 256 % 256).toUInt8⟩

/-- **The tops** (`VStar.In.Src.tops`), by rep and level, each node (a top row's first 32 words, `Firewall.topRows`'
inverse) with its salt: level 0's, round `0`'s rows, the one tree both reps share; each later level's, the schedule's
next top (`topSizes`' order), at rep `rep`'s round committing it, the rows after the round's earlier tops. -/
def topsOf (rs : Array RecVStar.SchedRound) (rr : ℕ → List _root_.VStar.In.Row) :
    Array (Array (Array (ByteArray × ByteArray))) :=
  let node := fun (rw : _root_.VStar.In.Row) => (wordsBytes (rw.words.extract 0 32), rw.salt)
  ((List.range 2).map fun rep =>
    let later := (List.range rs.size).flatMap fun k =>
      let sizes := (rs.getD k default).2.1.toList
      (List.range sizes.length).map fun e =>
        ((rr (rep * rs.size + k + 1)).drop (sizes.take e).sum).take (sizes.getD e 0)
    ((rr 0 :: later).map fun l => (l.map node).toArray).toArray).toArray

/-- **The stager's inputs** at the compiled statement `x`, off the inner witness `w`, the inner prover's tape `ρ` and the
inner rounds `hist`: `rest`, with the message rounds and the tops the rows the inner prover committed (`roundRows`).
None where `x` reads no schedule. -/
def srcOf (I : InnerRec) (x : Msg) (w : Claims.Flock.Wit) (ρ : I.Tape) (hist : List (Msg × Msg)) :
    Option _root_.VStar.In.Src := do
  let r ← readX x
  let rs ← repSched r.shape
  pure { I.rest w ρ.1 hist with msgs := msgsOf rs (roundRows I w ρ hist), tops := topsOf rs (roundRows I w ρ hist) }

/-- **The deployed outer witness builder**: part `s`'s port inputs (`VStar.In.partIn`, at its statement and its
circuit's program) at the stager's inputs, assigned on its circuit by `A`; the empty witness where the statement has no
such part or the stager refuses. -/
def vstarWit (A : Circuit → _root_.VStar.In.PartIn → Claims.Flock.Wit) (I : InnerRec) (x : Msg)
    (w : Claims.Flock.Wit) (ρ : I.Tape) (hist : List (Msg × Msg)) (s : ℕ) : Claims.Flock.Wit :=
  match partAt x s, srcOf I x w ρ hist with
  | some (st, c), some src =>
    match _root_.VStar.In.partIn src (progOf c) st with
    | .ok pin => A c pin
    | .error _ => ⟨[]⟩
  | _, _ => ⟨[]⟩

/-- **The deployed reference prover's pieces at the compiled statement `x`**: the inner prover `I.P`, the running sums'
digests as the staged values (`VStar.In.chainRows`), and `vstarWit` at `x`. -/
def refPartsAt (A : Circuit → _root_.VStar.In.PartIn → Claims.Flock.Wit) (I : InnerRec) (x : Msg) : RefParts where
  TI := I.Tape
  PI := I.P
  digests w ρ hist := (_root_.VStar.In.chainRows (I.rest w ρ.1 hist)).toList.map digestOf
  wit := vstarWit A I x

/-! ## What the trace places -/

/-- Port `nm`'s row at instance `n`: row `refs[n]` of its table. -/
def portRow (pin : _root_.VStar.In.PartIn) (nm : String) (n : ℕ) : Option _root_.VStar.In.Row := do
  let p ← (pin.find? (·.1 == nm)).map (·.2)
  let i ← p.refs[n]?
  p.rows[i]?

/-- A byte string's bits, each byte's low bit first (`VStar.bytesOf`'s order). -/
def saltBits (b : ByteArray) : List Bool := b.data.toList.flatMap fun u => (List.range 8).map u.toNat.testBit

/-- **Where a trace places the port inputs** (the port lemma's statement): at each instance `n` of each port `nm`, the
row's bits at its unit input columns (`rowHidden`) and its salt at the port's hm96 slot (`hmSalt`). -/
structure Places (c : Circuit) (pin : _root_.VStar.In.PartIn) (w : Claims.Flock.Wit) : Prop where
  row : ∀ nm n rw ps, portRow pin nm n = some rw → rowHidden c nm n = some ps →
    ps.map (w.hid.getD · false) = (_root_.VStar.In.wordBits rw.words).take ps.length
  salt : ∀ nm q n rw, c.ports.findIdx? (·.name == nm) = some q → portRow pin nm n = some rw →
    (hmSalt c q n).map (formBit w) = saltBits rw.salt

/-- **`vstarWit` is the assigner's trace at the part's port inputs**, where the part and the stager's inputs exist and
the stager accepts. -/
theorem vstarWit_eq {A : Circuit → _root_.VStar.In.PartIn → Claims.Flock.Wit} {I : InnerRec} {x : Msg}
    {w : Claims.Flock.Wit} {ρ : I.Tape} {hist : List (Msg × Msg)} {s : ℕ} {st : _root_.VStar.Stmt}
    {c : Circuit} {src : _root_.VStar.In.Src} {pin : _root_.VStar.In.PartIn} (hp : partAt x s = some (st, c))
    (hs : srcOf I x w ρ hist = some src) (hi : _root_.VStar.In.partIn src (progOf c) st = .ok pin) :
    vstarWit A I x w ρ hist s = A c pin := by
  simp [vstarWit, hp, hs, hi]

end RecDeploy.VStarBuild
