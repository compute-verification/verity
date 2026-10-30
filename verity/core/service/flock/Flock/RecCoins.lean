/-
V*'s verifier-registered values against the verifier's own coins (lane rec-v0). V* registers three values itself, under
public salts (`rec_residuals.verifier_value`, rec_residuals.py:393): each Ligerito level's opening coefficients `rec-coef`
and position bits `rec-dirs` (`rec_outer.stage`, rec_outer.py:216–217), and each algebra part's coefficients `rec-v`
(`rec_residuals.write`, rec_residuals.py:450). A statement whose unit is `rec-open/…` or `rec-residuals/…` is accepted only
when those values are the ones the verifier computes here, from the inner session alone:

* the inner statement (its circuit, public file and partition, under `verity/flock-circuit+plain-leaves` on OS coins) and
  the inner session's record, decoded as `verify` decodes it (`Record.decode`): each rep's rounds' coins (stream
  `<table>/rep<r>`) and the link points, which give the extra claims (`Setup.extra`) as `verifyRep` takes them
  (Verify.lean:41);
* each rep's v, its points and κ (`RecAlgebra.verifierRows` at the rep's coins), and the comb at the lincheck's fold points
  (`CircuitFold.folded`), computed only for a part whose v holds it;
* opening `j` of level `ℓ`'s coefficients `κ[ℓ][r][l] · eq(α_ℓ)[j]` (`coefAt`), its position's low `H` bits (`dirsRow`),
  and part `p`'s v elements (`RecResiduals.groups`), each row committed as `registered.Registered.commit` commits it
  (registered.py:139) under `verifierSalt`.

Every statement table row of `coef`, `dirs` or `v` must be the verifier's row at its position, and the registered root the
root of the verifier's rows, which every instance reads at its own index (`checkValue`). A part's `out` port is the
value V* registers as its unit's outputs, all-zero rows under `zeroSalt` (`rec_residuals.zero_salts`), one per instance,
checked the same way (`checkZero`: `rec-residuals-zero` at `checkResiduals` and `checkOpening`, `rec-sparse-zero` at
`checkSparse`).

Candidate 3's statements are checked the same way at coins the prover's record does not hold, the coin server's record of
the session (`C3`, `--rec-inner-transcript` with the verifier's inner shape `rec-inner --sparse`): RecSparse's `v`
(`rec-sparse-v`, `checkSparse`) at each rep's inner points and `sparse/rep<r>`'s coins; R's levels' `coef` and `dirs`
(`checkStackedLevel`) and R's opening's `v` (`checkOpening`) at `stacked/rep0`'s coins and the two reps' claim points; an R
level's `out` reads R's tops `rec-top-stacked-L<l>` at each opening's node above its `H` siblings (`checkTopReads`).
RecSparse's refusals that read only the points' sizes are checked on the inner statement itself at every candidate-3
session's setup, whatever its unit (`checkAtC3`, `checkInnerRange`).

RecCap (`rec-cap/n<N>` at level 0, `rec-cap/n<N>-l<l>` at a later level `l`) registers no value from the coins: its `top`
reads the level's tops `rec-top-L<l>` at row `i` in instance `i`, every node of them, and its `out` is V*'s zero rows
`rec-cap-zero` (`checkCap`); its rows are held to the coin server's `root_B`, or to both reps' level-`l` tops
(`checkLink`, `RecVStar.checkTopRows`).
-/
import Flock.Keccak
import Flock.Registered
import Flock.Firewall
import Flock.RecOpening
import Flock.RecSparse
import Flock.RecVStar
import Flock.Verify
import Flock.Zk
import Flock.Norm

namespace Flock
namespace RecCoins

open Lean (Json)

def COEF_VALUE : String := "rec-coef"
def DIRS_VALUE : String := "rec-dirs"
def V_VALUE : String := "rec-v"
def VERIFIER_SALT : String := "verity/rec-v0/verifier-salt/v0"
def SALT_BYTES : Nat := 192
def DIRS_WORDS : Nat := 64

/-- Row `i`'s salt of a value the verifier registers itself (`rec_residuals.verifier_salts`, rec_residuals.py:385):
SHAKE256 of the tag, the program's SHA-512 hex, the value's name and `i` (u32 little-endian), NUL between them. -/
def verifierSalt (program value : String) (i : Nat) : ByteArray :=
  Keccak.shake256 ((((VERIFIER_SALT.toUTF8.push 0 ++ program.toUTF8).push 0 ++ value.toUTF8).push 0) ++ le32 i.toUInt32)
    SALT_BYTES

def ZERO_SALT : String := "verity/rec-v0/zero-residual-salt/v0"
def ZERO_VALUE : String := "rec-residuals-zero"
def OUT_WORDS : Nat := 128

/-- Row `i`'s salt of a value's zero rows, the outputs V* registers itself (`rec_residuals.zero_salts`,
rec_residuals.py:402): SHAKE256 of the tag, the value's name and `i` (u32 little-endian), NUL only before `i`. -/
def zeroSalt (value : String) (i : Nat) : ByteArray :=
  Keccak.shake256 ((ZERO_SALT.toUTF8 ++ value.toUTF8).push 0 ++ le32 i.toUInt32) SALT_BYTES

/-- `n` rows of `w` zero words. -/
def zeroRows (n w : Nat) : Array (Array Nat) := Array.replicate n (Array.replicate w 0)

/-- F elements as u16 words, 8 each, `lo` then `hi` little-endian (`rec_open.words`, rec_open.py:120). -/
def words (xs : Array F128) : Array Nat :=
  let u16s := fun (x : UInt64) => (Array.range 4).map fun k => ((x >>> (16 * k).toUInt64) &&& 0xFFFF).toNat
  xs.flatMap fun x => u16s x.lo ++ u16s x.hi

def padTo (ws : Array Nat) (n : Nat) : Array Nat := ws ++ Array.replicate (n - ws.size) 0

/-- Opening `j`'s coefficients at `kap = κ[ℓ]` and `e = eq(α_ℓ)[j]`: `κ[ℓ][0]`'s `LANES`, then `κ[ℓ][1]`'s, each times `e`
(`rec_outer.coefficients`, rec_outer.py:159; the proofs' `coefAt`, VBridge/Inputs.lean:41). -/
def coefAt (kap : Array (Array F128)) (LANES : Nat) (e : F128) : Array F128 :=
  (Array.range LANES).map (fun l => (kap[0]!)[l]! * e) ++ (Array.range LANES).map (fun l => (kap[1]!)[l]! * e)

/-- `dirs`: the position's low `H` bits, 16 a word, then zero words (`rec_open.dirs_words`, rec_open.py:161). -/
def dirsRow (pos H : Nat) : Array Nat :=
  #[pos % 2 ^ H % 2 ^ 16, pos % 2 ^ H / 2 ^ 16] ++ Array.replicate (DIRS_WORDS - 2) 0

/-- A registered row's `b ‖ c` under `salt` (`registered.Registered.commit`, registered.py:139: `sha512/row/v1` of its
u16 words, zero-padded to whole blocks, under hm96-sha512's default key). -/
def commitRow (row : Array Nat) (salt : ByteArray) : ByteArray :=
  Hm96.Default512.commitString (Firewall.rowDigest (padTo row (Registered.rowWords row.size 16))) salt

/-- How Flock's plain verifier is told to set the inner statement up, as the verifier holds it
(`--rec-inner-statement`, `--rec-inner-statement-coins`, `--rec-inner-tables`, `--rec-inner-draw`): the statement id,
whether its coins are the coin server's (`os`), its lookup tables, and its draw's bytes. `RecSetup.checkForm` holds
them to the setup the inner statement's files give it. -/
structure InnerForm where
  statement : String
  os : Bool
  tables : List (String × ByteArray)
  draw : Option ByteArray

/-- The inner statement and the inner session's record, as the verifier holds them, at candidate 3's session the
verifier's inner shape (`inner`), which the inner statement must be (`RecVStar.checkInnerStmt`, at `Inner.load`), and
how the inner statement is set up (`form`). -/
structure Files where
  circuit : ByteArray
  pub : ByteArray
  partition : Json
  record : String
  inner : Option RecVStar.Inner := none
  form : Option InnerForm := none

/-- What check 5 reads of the inner session of V*'s unit `unit`: its statement's `m` and `k_log`, its extra claims at the
record's link points, each rep's coins (an error without the unit), and the comb at a rep's fold points (`Inner.comb`). Set
up from the inner statement (`Inner.load`), or read from a record of them (`Inner.ofJson`, which no reader of a statement
reaches: `refuseCoins`). -/
structure Inner where
  unit : String
  m : Nat
  kLog : Nat
  extras : V (Array Claim)
  roundsOf : Nat → V (Array (Array F128))
  combOf : Nat → RecAlgebra.Info RecAlgebra.Lin → V (Array F128)
  /-- the inner circuit's unit range (META's `ranges`, the unit net's), whose slots the sparse phase's layout factors weigh -/
  unitRange : V Range := .error "the verifier's inner coins hold no unit range"

/-- A recursive session's inner statement (PROTOCOL.md §16.16), on the coins the coin server draws. -/
def innerTags : Tags := Tags.circuitPlainLeaves.withCoins Coins.os

/-- The inner circuit's unit range: META's one range of its unit net, whose slots the sparse phase's layout factors weigh. -/
def unitRangeOf (st : Stmt) : V Range :=
  match st.c.ranges.filter (·.kind == Kind.net st.c.unitNet) with
  | #[r] => .ok r
  | rs => .error s!"the inner circuit has {rs.size} ranges of its unit net, not one"

/-- The inner statement, set up from its circuit and public files on the coin server's coins, and its session's spec. -/
def Files.stmt (unit : String) (f : Files) : V (SessionSpec × Stmt) := do
  let (spec, sts) ← match Stmt.setupTables innerTags f.circuit f.pub Coins.os 1 (partition := some f.partition) with
    | .ok x => pure x
    | .error e => throw s!"rec-coins: {unit}: the inner statement: {e}"
  let some st := sts[0]? | throw s!"rec-coins: {unit}: the inner statement has no table"
  return (spec, st)

/-- A session record's `unit_draw`, when it is a draw (not absent, not null), rebuilt (`norm`), as `verify` reads it
(`Main.recordDraw`). -/
def recordDraw (record : String) : Option Json :=
  match Json.parse record with
  | .ok j => match j.getObjVal? "unit_draw" with
    | .ok Json.null => none
    | .ok d => some (Flock.norm d)
    | .error _ => none
  | .error _ => none

/-- The spec a recursive inner session's record decodes against, `spec` and `st` the inner statement's without a draw and
`setupAt d` the statement at draw `d`: `spec` for a record without a draw (whose session `verify` refuses last in its
verdict, `RecSetup.innerDrawn`), and for a record whose draw proves every one of
`st`'s units, in order, the drawn statement's spec (the coin server's full draw, `rec_live.full_draw`, which `verify`
holds to its own law with `--stratified`, `--program` and `--partition`). The drawn statement has `st`'s units, `m` and
regions; its header carries the draw, so its Σ and Hello are its own. A draw that leaves a unit out is refused: V*
recurses on an inner session that proves every unit. -/
def drawnSpec (setupAt : Json → V Stmt) (spec : SessionSpec) (st : Stmt) (record : String) : V SessionSpec := do
  let some d := recordDraw record | return spec
  let D ← Draw.UnitDraw.ofJson d
  if D.proved != Array.range st.n then
    throw s!"the session's draw proves {D.proved.size} units, not every one of the statement's {st.n} in order"
  return (← setupAt d).spec

/-- `drawnSpec` at the inner files: the inner statement set up at the record's draw as `Files.stmt` sets it up without one. -/
def Files.drawnSpec (f : Files) (spec : SessionSpec) (st : Stmt) : V SessionSpec :=
  RecCoins.drawnSpec (fun d => Stmt.setupHidden innerTags f.circuit f.pub Coins.os [] (some d) (some f.partition) none)
    spec st f.record

/-- The inner statement as the recursive audit's event reads it (`AcceptsStmt`): set up from its circuit and public files
on the coin server's coins, with an `m` that `Zk.schedule` accepts, and its session's record decoded (at the record's full
draw when it has one, `Files.drawnSpec`). -/
def Inner.load (unit : String) (f : Files) : V Inner := do
  let (spec, st) ← f.stmt unit
  if let some i := f.inner then
    if let .error e := RecVStar.checkInnerStmt i st then throw s!"rec-coins: {unit}: {e}"
  if let .error e := Zk.schedule st.m then throw s!"rec-coins: {unit}: the inner statement's schedule: {e}"
  let spec ← match f.drawnSpec spec st with
    | .ok s => pure s
    | .error e => throw s!"rec-coins: {unit}: the inner session's draw: {e}"
  let sess ← match Record.decode spec f.record with
    | .ok s => pure s
    | .error e => throw s!"rec-coins: {unit}: the inner session's record: {e}"
  let setup := Setup.ofCircuit st
  let table := spec.tables[0]!
  return { unit, m := setup.m, kLog := setup.kLog, extras := setup.extra sess.points
           roundsOf := fun r => do
             let name := s!"{table}/rep{r}"
             let some s := sess.streams.find? (·.name == name) | throw s!"the inner record has no stream {name}"
             return s.rounds.map (·.coins)
           combOf := fun _ info => do
             let f := info.lc.fold
             setup.fold.folded f.alpha f.beta f.lw f.rhoIn f.ts
           unitRange := unitRangeOf st }

/-- Rep `r`'s coins, round by round: its stream's, as `verifyRep` replays them (Verify.lean:40). -/
def Inner.rounds (inn : Inner) (r : Nat) : V (Array (Array F128)) :=
  (inn.roundsOf r).mapError fun e => s!"rec-coins: {inn.unit}: {e}"

/-- The verifier's shape and extra claims: the inner statement's `m` and `k_log`, its extra claims at the record's link
points (Verify.lean:41), two claims besides them. -/
def Inner.algebra (inn : Inner) : V (Shape × Array Claim) := do
  let extras ← inn.extras
  let some sh := RecResiduals.shape? inn.m inn.kLog (2 + extras.size)
    | throw s!"rec-coins: {inn.unit}: the inner statement's shape (m {inn.m}, k_log {inn.kLog}) has no algebra"
  return (sh, extras)

def ZERO_COMB : Array F128 := Array.replicate 64 0

/-- A rep's values: its coins, and `verifierRows` at them (v with the comb zero, which neither its points nor κ read). -/
structure Rep where
  rounds : Array (Array F128)
  v : Array F128
  info : RecAlgebra.Info RecAlgebra.Lin
  kappa : Array (Array (Array F128))

def Inner.rep (inn : Inner) (sh : Shape) (extras : Array Claim) (r : Nat) : V Rep := do
  let rounds ← inn.rounds r
  match RecAlgebra.verifierRows sh rounds extras ZERO_COMB with
  | .ok (v, info, kappa) => return ⟨rounds, v, info, kappa⟩
  | .error e => throw s!"rec-coins: {inn.unit}: rep {r}: {e}"

/-- The comb at rep `r`'s fold points (`CircuitFold.folded`, as V*'s `InnerFold.lean` computes it). -/
def Inner.comb (inn : Inner) (r : Nat) (rep : Rep) : V (Array F128) := do
  let comb ← inn.combOf r rep.info
  if comb.size != 64 then throw s!"rec-coins: {inn.unit}: a comb of {comb.size}"
  return comb

/-- Port `port` of the statement is the value `value` the verifier registers itself, of `rows`: the verifier's registered
reads hold it as `value` (`own`), of `rows.size` rows of their width, unit `i` reading row `i`; every table row's `b ‖ c`
is the verifier's row at its position under `salt` of META's `program_sha512`; and the registered root is the
frame-v3-sha512 root of those rows (`Registered.rowLeaf`). `what i` names row `i`. -/
def checkRows (salt : String → Nat → ByteArray) (c : Circuit) (pub : Public) (own : Array Registered.Port)
    (port value : String) (rows : Array (Array Nat)) (what : Nat → String) : V Unit := do
  let unit := c.unit.name
  let some reg := own.find? (·.name == port)
    | throw s!"rec-coins: {unit}: the verifier's registered reads hold no port {port}: its {value} would be the prover's"
  if reg.value != value then throw s!"rec-coins: {unit}: port {port} reads {reg.value}, not the verifier's {value}"
  let n := rows.size
  let w := (rows[0]?.map (·.size)).getD 0
  if reg.leaves != n || reg.words != some w || reg.wordBits != 16 || reg.schema != HmRow.ROW_SCHEMA then
    throw s!"rec-coins: {unit}: {value}: {reg.leaves} {reg.schema} rows of {reg.words.getD 0} words registered, the verifier computes {n} {HmRow.ROW_SCHEMA} rows of {w}"
  if reg.reads != (Array.range n).map some then
    throw s!"rec-coins: {unit}: {value}: the reads file does not read row i of the verifier's {n} rows in unit i"
  let .ok prog := c.metaJ.getObjValAs? String "program_sha512" | throw s!"rec-coins: {unit}: META names no program_sha512"
  let bcs := (Array.range n).map fun i => commitRow rows[i]! (salt prog i)
  let some t := pub.tables | throw s!"rec-coins: {unit}: {port} is not a table of a shared-row file"
  let some p := (List.range c.ports.size).find? (fun p => c.ports[p]!.name == port)
    | throw s!"rec-coins: {unit}: {port} is not a row of the circuit"
  let (positions, _) ← Registered.entry pub reg
  for r in [0:positions.size] do
    let i := positions[r]!
    if Registered.tableRow pub t p r != bcs.getD i ByteArray.empty then
      throw s!"rec-coins: {unit}: {what i}: the statement's {port} row is not the verifier's {value} row {i}"
  let leaves := (Array.range n).map fun i => Registered.rowLeaf reg.domain i (Hm96.Default512.treeLeaf bcs[i]!)
  if FV3.root reg.domain leaves Sha512.hash != reg.root then
    throw s!"rec-coins: {unit}: {value}: the registered root is not the root of the verifier's {n} rows"

/-- `checkRows` under `verifierSalt`: a value the verifier computes. -/
def checkValue (c : Circuit) (pub : Public) (own : Array Registered.Port) (port value : String) (rows : Array (Array Nat))
    (what : Nat → String) : V Unit :=
  checkRows (verifierSalt · value) c pub own port value rows what

/-- Port `out` of the statement is `value`'s `n` zero rows of `w` words under `zeroSalt`, as V* registers its unit's
outputs (`rec_residuals.write_rows`, rec_residuals.py:498). -/
def checkZero (c : Circuit) (pub : Public) (own : Array Registered.Port) (value : String) (n w : Nat) : V Unit :=
  checkRows (fun _ => zeroSalt value) c pub own "out" value (zeroRows n w) fun r => s!"rep {r}'s zero outputs"

/-- A `rec-open/l<LANES>-h<H>` statement: the level of the inner schedule with `LANES` lanes and `H` siblings below its
cap, `2q` instances (rep `r`'s opening `j` instance `r·q + j`, `rec_outer.levels`), each instance's `coef` the verifier's
`coefAt (κ[ℓ]) LANES eq(α_ℓ)[j]` and its `dirs` the low `H` bits of its query position, at its rep's coins
(`rec_vstage.coefficients`, rec_vstage.py:171). -/
def checkOpen (inn : Inner) (c : Circuit) (pub : Public) (own : Array Registered.Port) (LANES H : Nat) : V Unit := do
  let unit := c.unit.name
  let (sh, extras) ← inn.algebra
  let lv := sh.sched.levels
  let ls := (List.range lv.size).filter fun l => 2 ^ lv[l]!.lanes == LANES && lv[l]!.d - lv[l]!.cap == H
  let [l] := ls
    | throw s!"rec-coins: {unit}: no one level of the inner schedule (m {sh.m}) has {LANES} lanes and {H} siblings below its cap"
  let q := lv[l]!.q
  let r0 ← inn.rep sh extras 0
  let r1 ← inn.rep sh extras 1
  for (r, rep) in [(0, r0), (1, r1)] do
    let lg := rep.info.lg
    if lg.conss.size != lv.size || (lg.conss[l]!).pos.size != q then
      throw s!"rec-coins: {unit}: rep {r}: the verifier's coins give no {q} queries at level {l}"
  let at_ := fun (i : Nat) => (if i < q then r0 else r1, i % q)
  let what := fun (i : Nat) => s!"level {l} rep {i / q} opening {i % q}"
  let coef := (Array.range (2 * q)).map fun i =>
    let (rep, j) := at_ i
    words (coefAt (rep.kappa[l]!) LANES (eqTable (rep.info.lg.conss[l]!).alpha)[j]!)
  let dirs := (Array.range (2 * q)).map fun i =>
    let (rep, j) := at_ i
    dirsRow (rep.info.lg.conss[l]!).pos[j]! H
  checkValue c pub own "coef" COEF_VALUE coef what
  checkValue c pub own "dirs" DIRS_VALUE dirs what

/-- A `rec-residuals/m<m>-k<k_log>-c<claims>-p<j>` statement: the inner session's shape, two instances (rep `r` instance
`r`), each instance's `v` part `j`'s elements (`RecResiduals.groups`) of the verifier's v at its rep's coins, the comb last
(`rec_vstage.stage`, rec_vstage.py:404–406). -/
def checkResiduals (inn : Inner) (c : Circuit) (pub : Public) (own : Array Registered.Port) (m k cl j : Nat) : V Unit := do
  let unit := c.unit.name
  let (sh, extras) ← inn.algebra
  if (m, k, cl) != (sh.m, sh.kLog, sh.claims) then
    throw s!"rec-coins: {unit}: the inner session's shape is m{sh.m}-k{sh.kLog}-c{sh.claims}"
  let gs ← match (do RecResiduals.groups (← RecAlgebra.structureOf sh) (RecResiduals.portSizes sh)) with
    | .ok gs => pure gs
    | .error e => throw s!"rec-coins: {unit}: {e}"
  let some (_, _, vs) := gs[j]? | throw s!"rec-coins: {unit}: part {j} of {gs.size}"
  let rows ← #[0, 1].mapM fun r => do
    let rep ← inn.rep sh extras r
    let base := rep.v.extract 0 (rep.v.size - 64)
    let comb ← if vs.any (· ≥ base.size) then inn.comb r rep else pure ZERO_COMB
    let full := base ++ comb
    pure (padTo (words (vs.map (full[·]!))) (RecResiduals.portWordsOf vs.size))
  checkValue c pub own "v" V_VALUE rows fun r => s!"rep {r}: part {j}'s {vs.size} elements of v"
  checkZero c pub own ZERO_VALUE 2 OUT_WORDS

/-! ### Candidate 3's statements: RecSparse, R's levels and R's opening -/

def SPARSE_V_VALUE : String := "rec-sparse-v"
def SPARSE_ZERO_VALUE : String := "rec-sparse-zero"

/-- What check 5 reads of candidate 3's session beside the inner statement: the verifier's own inner shape (`rec-inner
--sparse`: the bucket's RecSparse shape, and R's, the inner circuit's META's) and the coin server's record of the session
(`--rec-inner-transcript`, `RecVStar.innerCommits`), whose sparse phases' and R's coins the prover's record does not hold. -/
structure C3 where
  sparse : RecSparse.Shape
  stacked : Option RecVStar.StackedShape
  commits : RecVStar.InnerCommits

/-- Stream `s`'s coins by round, as the coin server drew them. -/
def C3.coins (c3 : C3) (unit s : String) : V (Array (Array F128)) := do
  let some st := c3.commits.stream? s | throw s!"rec-coins: {unit}: the inner transcript has no stream {s}"
  return st.coins

/-- A rep's sparse-phase coins (`rec_vstage.sparse_reps`): `ζ_c`, H2's `ζ_r` (`pre`), H3's `r_k` then `r_s` (`rounds`),
`ζ_p`, and at T > 1 `ρ_t`. -/
structure SparseCoins where
  zetaC : Array F128
  pre : Array F128
  rounds : Array F128
  zetaP : Array F128
  rho : Array F128

/-- Rep `r`'s sparse-phase coins: of `sparse/rep<r>`'s rounds that draw any (`rec_live.sparse_rounds`), 6 for `ζ_c`, one a
round for H2's 6 and H3's `NU + S_LOG`, `P_LOG` for `ζ_p` (u's round) and at T > 1 `T_LOG` for `ρ_t` (vt's); refused at any
other count. -/
def sparseCoins (c3 : C3) (unit : String) (r : Nat) : V SparseCoins := do
  let sh := c3.sparse
  let s := s!"{RecVStar.SPARSE}/rep{r}"
  let cs := (← c3.coins unit s).filter (·.size != 0)
  let nsc := sh.degrees.length
  let pre := RecSparse.PRE_ROUNDS
  let want := #[pre] ++ Array.replicate (pre + nsc) 1 ++ #[sh.pLog] ++ (if sh.stacked then #[sh.tLog] else #[])
  if cs.map (·.size) != want then
    throw s!"rec-coins: {unit}: {s} draws {cs.map (·.size)} coins by round, not {RecSparse.unitName sh}'s sparse phase's {want}"
  let one := fun (i : Nat) => (cs[i]!)[0]!
  return { zetaC := cs[0]!, pre := (Array.range pre).map (one ∘ (· + 1)),
           rounds := (Array.range nsc).map (one ∘ (· + pre + 1)), zetaP := cs[pre + 1 + nsc]!,
           rho := if sh.stacked then cs[pre + 2 + nsc]! else #[] }

def dotF (a b : Array F128) : F128 := (a.zip b).foldl (fun s xy => s + xy.1 * xy.2) 0

/-- `rec_sparse.layout_factors`: slot σ's `eq(x_hi, q) eq(y_hi, q)` at `q = pos0 + σ` below the unit range's count, zero past
it, `x_hi` and `y_hi` the points' coordinates past `ELL - 6`; the range's slots are of `2^ELL` rows, at most S of them,
inside both tables. -/
def layoutFactors (sh : RecSparse.Shape) (xRest yRest : Array F128) (rng : Range) : Except String (Array F128) := do
  let k := sh.ell - RecSparse.PRE_ROUNDS
  let ex := eqTable (xRest.extract k xRest.size)
  let ey := eqTable (yRest.extract k yRest.size)
  if rng.slotLog != sh.ell || rng.count > sh.S || rng.pos0 + rng.count > min ex.size ey.size then
    throw (s!"the inner circuit's unit range (slot_log {rng.slotLog}, pos0 {rng.pos0}, count {rng.count}) is not at most " ++
      s!"{sh.S} slots of 2^{sh.ell} rows inside the block")
  return (Array.range rng.count).map (fun i => ex[rng.pos0 + i]! * ey[rng.pos0 + i]!) ++ Array.replicate (sh.S - rng.count) 0

/-- `rec_sparse.verifier_row`, RecSparse's `v` (`RecSparse.vParts`) at a rep's sparse coins `c`: eq(ζ_c, ·); per H2 round
`ρ`, `ρ` and `ρ + ρ²`; ŵ(ζ_r) = `Σ_s eq(ζ_r, s) L_s(z)`; per H3 round `ρ` of degree `d`, `ρ` then `ρ + ρ^i` for i = 2 .. d;
α; `1 + x_j` and `1 + y_j` for j < ELL (`x = (ζ_r, x_rest)`, `y = (ζ_c, y_rest)`); Ũ(r_s); eq(ζ_p, ·); at T > 1 eq(r_s, ·)
and eq(ρ_t, ·). `α`, `z`, `x_rest` (ρ_in) and `y_rest` (the lincheck's coins reversed) are the rep's inner points
(`rec_sparse.inner_points`), `U` the slots' layout factors (`layoutFactors`). -/
def sparseV (sh : RecSparse.Shape) (c : SparseCoins) (alpha z : F128) (xRest yRest U : Array F128) :
    Except String (Array F128) := do
  let k := sh.ell - RecSparse.PRE_ROUNDS
  if xRest.size < k || yRest.size < k || U.size != sh.S then throw s!"the inner points are short of {RecSparse.unitName sh}'s"
  let pows := fun (r : F128) (deg : Nat) => Id.run do
    let mut out := #[r]
    let mut p := r
    for _ in [2:deg + 1] do
      p := p * r
      out := out.push (p + r)
    return out
  let rS := c.rounds.extract sh.nu c.rounds.size
  let xLo := c.pre ++ xRest.extract 0 k
  let yLo := c.zetaC ++ yRest.extract 0 k
  let v := eqTable c.zetaC ++ c.pre.flatMap (pows · 2) ++ #[dotF (eqTable c.pre) (skipWeights z)] ++
    (c.rounds.zip sh.degrees.toArray).flatMap (fun rd => pows rd.1 rd.2) ++ #[alpha] ++ xLo.map (· + 1) ++
    yLo.map (· + 1) ++ #[dotF (eqTable rS) U] ++ eqTable c.zetaP ++ (if sh.stacked then eqTable rS ++ eqTable c.rho else #[])
  if v.size != RecSparse.nv sh then throw s!"v of {v.size} elements, {RecSparse.unitName sh}'s {RecSparse.nv sh}"
  return v

/-- A `rec-sparse/n<NU>-e<ELL>-s<S_LOG>-t<T_LOG>` statement at the verifier's RecSparse shape: two instances (rep `r`
instance `r`), each instance's `v` the verifier's (`sparseV`) at rep `r`'s inner points (its circuit stream's coins) and its
sparse stream's coins (the coin server's), and the inner circuit's unit range (`rec_vstage.stage_sparse`). -/
def checkSparse (inn : Inner) (c3 : C3) (c : Circuit) (pub : Public) (own : Array Registered.Port) (sh : RecSparse.Shape) :
    V Unit := do
  let unit := c.unit.name
  if sh != c3.sparse then throw s!"rec-coins: {unit}: the verifier's RecSparse shape is {RecSparse.unitName c3.sparse}"
  let (ash, extras) ← inn.algebra
  let rng ← inn.unitRange.mapError (s!"rec-coins: {unit}: " ++ ·)
  let rows ← #[0, 1].mapM fun r => do
    let rep ← inn.rep ash extras r
    let sc ← sparseCoins c3 unit r
    let i := rep.info
    let v ← (do sparseV sh sc i.lc.fold.alpha i.zc.z i.lc.fold.rhoIn i.lc.rIn (← layoutFactors sh i.lc.fold.rhoIn i.lc.rIn rng)
      ).mapError (s!"rec-coins: {unit}: rep {r}: " ++ ·)
    pure (padTo (words v) (RecSparse.portWords (RecSparse.nv sh)))
  checkValue c pub own "v" SPARSE_V_VALUE rows fun r => s!"rep {r}: RecSparse's v"
  checkZero c pub own SPARSE_ZERO_VALUE 2 (RecSparse.outWords sh)

/-- **Check 5's sparse refusals on the inner statement itself**: the verifier's inner shape `i` is candidate 3's at `st`
(`RecVStar.checkInnerStmt`), and at its RecSparse shape, which is valid, the inner points' `k_log − 6` coordinates are not
short of its `ELL − 6` (`sparseV`'s check) and the layout factors at the inner circuit's unit range are computed
(`layoutFactors`). Both read only the points' sizes, so at zero points this is their verdict at every rep's points. -/
def checkInnerRange (i : RecVStar.Inner) (st : Stmt) : V Unit := do
  RecVStar.checkInnerStmt i st
  let some sp := i.sparse | return
  if !sp.valid then
    throw s!"rec-inner: {RecSparse.unitName sp}: rec-sparse's NU is at least 1, ELL at least 6 and T_LOG at most 16"
  let xs := Array.replicate (st.c.kLog - 6) (0 : F128)
  if xs.size < sp.ell - RecSparse.PRE_ROUNDS then
    throw s!"rec-inner: the inner points are short of {RecSparse.unitName sp}'s"
  let rng ← (unitRangeOf st).mapError (s!"rec-inner: " ++ ·)
  if let .error e := layoutFactors sp xs xs rng then throw s!"rec-inner: {e}"

/-- R's registered opening as the verifier computes it: its opening-only shape, `v` (the coefficient slots; no comb), each
Ligerito level's query positions and eq(α), and the final check's κ. -/
structure RegisteredOpening where
  sh : Shape
  v : Array F128
  lg : RecAlgebra.LgPts
  kappa : Array (Array (Array F128))

/-- `rec_algebra.verifier_rows` at R's opening-only shape `Shape(m, 6, claims)` (`rec_vstage.stacked_reads`), on
`stacked/rep0`'s coins: the claims' ring switches, then Ligerito, built as `RecOpening.build` builds S (which must be
`RecOpening.structureOf`'s), at both reps' registered claims (`rec_sparse.registered_claim`): rep `k`'s point
`(r_k, ζ_p, ρ_t)` from its sparse coins, zero-padded to R's `m`, its skip weights eq(point[:6], ·) and `x` the rest; a
claim's value is a row of `rec-claim`, w's, so it is none of v's. Refused unless some part of that S holds each running
sum (`RecOpening.accHeld`). -/
def opening (c3 : C3) (unit : String) : V RegisteredOpening := do
  let some r := c3.stacked | throw s!"rec-coins: {unit}: the verifier's inner shape names no R (the inner circuit's META's stacked)"
  let some sh := RecOpening.shape? r.m r.claims
    | throw s!"rec-coins: {unit}: R's shape (m {r.m}, claims {r.claims}) has no fast100 schedule"
  if r.claims != 2 then throw s!"rec-coins: {unit}: R's claims {r.claims}, not the two reps' sparse claims"
  let claims ← #[0, 1].mapM fun k => do
    let sc ← sparseCoins c3 unit k
    let pt := sc.rounds.extract 0 c3.sparse.nu ++ sc.zetaP ++ sc.rho
    if pt.size < RecSparse.PRE_ROUNDS || pt.size > sh.m then
      throw s!"rec-coins: {unit}: rep {k}'s claim point has {pt.size} coordinates, R's m is {sh.m}"
    let pt := pt ++ Array.replicate (sh.m - pt.size) 0
    pure ({ value := RecAlgebra.instValLin.zero, skip := eqTable (pt.extract 0 RecSparse.PRE_ROUNDS),
            x := pt.extract RecSparse.PRE_ROUNDS pt.size } : RecAlgebra.AClaim RecAlgebra.Lin)
  let rounds ← c3.coins unit s!"{RecVStar.STACKED}/rep0"
  let M := RecAlgebra.symbolic sh
  let run : RecAlgebra.CM (Array RecAlgebra.Lin × RecAlgebra.LgPts) := do
    let (rsr, rs) ← RecAlgebra.ringSwitch M claims
    let tIn := M.tAtom.getD rs.T
    let (lgr, lg) ← RecAlgebra.ligerito sh.sched M tIn rs.sws
    return (rsr ++ lgr ++ #[RecAlgebra.instValLin.add tIn rs.T], lg)
  let built : Except String RegisteredOpening := do
    let (lins, lg) ← (run.run' 0).run (RecAlgebra.CoinSource.ofRounds rounds)
    let kappa := sh.sched.levels.map fun lv => Array.replicate 2 (Array.replicate (2 ^ lv.lanes) (0 : F128))
    let (res, st) ← (RecOpening.buildAll sh lins.toList 0).run { kappa }
    if some (⟨RecOpening.nw sh, st.vNames.size, st.ops, res⟩ : RecAlgebra.GfS) != (RecOpening.structureOf sh).toOption then
      throw "the structure moved with the coins"
    if !RecOpening.accHeld sh ⟨RecOpening.nw sh, st.vNames.size, st.ops, res⟩ then
      throw "no part of its S holds each of Ligerito's running sums"
    pure { sh, v := st.vVals, lg, kappa := st.kappa }
  built.mapError (s!"rec-coins: {unit}: R's opening at {RecVStar.STACKED}/rep0's coins: " ++ ·)

/-- R's level `l`'s `out` port in the reads file: R's level-`l` tops `rec-top-stacked-L<l>` (`rec_vstage.stage_stacked`), opening
`j` of `q` reading node `pos[j] >>> H` of them, the node `H` levels above its query position (`rec_outer.TopValue.position`:
R's one rep's tops are the value's only run). -/
def checkTopReads (unit : String) (own : Array Registered.Port) (l q H : Nat) (pos : Array Nat) : V Unit := do
  let some reg := own.find? (·.name == "out")
    | throw s!"rec-coins: {unit}: the verifier's registered reads hold no port out: R's level {l} tops would be the prover's"
  if reg.value != s!"rec-top-stacked-L{l}" then
    throw s!"rec-coins: {unit}: port out reads {reg.value}, not R's level {l} tops rec-top-stacked-L{l}"
  if reg.reads != (Array.range q).map fun j => some (pos[j]! >>> H) then
    throw s!"rec-coins: {unit}: rec-top-stacked-L{l}: the reads file does not read opening j's top at node pos[j] >>> {H}"

/-- A level of R's openings, `rec-open-stacked/l<LANES>-h<H>`: the level of R's schedule with `LANES` lanes and `H` siblings
below its cap, its `q` openings instance by instance (R's one rep, `rec_vstage.stage_stacked`), each instance's `coef`
`coefAt (κ[ℓ]) LANES eq(α_ℓ)[j]`, its `dirs` the low `H` bits of its query position, and its `out` R's tops at the node
above them (`checkTopReads`), at R's coins (`opening`). -/
def checkStackedLevel (c3 : C3) (c : Circuit) (pub : Public) (own : Array Registered.Port) (LANES H : Nat) : V Unit := do
  let unit := c.unit.name
  let o ← opening c3 unit
  let lv := o.sh.sched.levels
  let ls := (List.range lv.size).filter fun l => 2 ^ lv[l]!.lanes == LANES && lv[l]!.d - lv[l]!.cap == H
  let [l] := ls
    | throw s!"rec-coins: {unit}: no one level of R's schedule (m {o.sh.m}) has {LANES} lanes and {H} siblings below its cap"
  let q := lv[l]!.q
  let cons := o.lg.conss
  if cons.size != lv.size || (cons[l]!).pos.size != q then
    throw s!"rec-coins: {unit}: R's coins give no {q} queries at level {l}"
  let ea := eqTable (cons[l]!).alpha
  let what := fun (j : Nat) => s!"R's level {l} opening {j}"
  checkValue c pub own "coef" COEF_VALUE ((Array.range q).map fun j => words (coefAt (o.kappa[l]!) LANES ea[j]!)) what
  checkValue c pub own "dirs" DIRS_VALUE ((Array.range q).map fun j => dirsRow (cons[l]!).pos[j]! H) what
  checkTopReads unit own l q H (cons[l]!).pos

/-- A part of R's opening, `rec-opening/m<m>-c<claims>-p<j>`, at R's shape: one instance, its `v` part `j`'s elements
(`RecResiduals.groups` of `RecOpening.structureOf`) of the verifier's v at R's coins (`opening`), its `out` one zero row
(`rec_residuals.stage_opening`'s `write`). -/
def checkOpening (c3 : C3) (c : Circuit) (pub : Public) (own : Array Registered.Port) (m cl j : Nat) : V Unit := do
  let unit := c.unit.name
  let o ← opening c3 unit
  if (m, cl) != (o.sh.m, o.sh.claims) then throw s!"rec-coins: {unit}: R's shape is m{o.sh.m}-c{o.sh.claims}"
  let gs ← match (do RecResiduals.groups (← RecOpening.structureOf o.sh) (RecOpening.portSizes o.sh)) with
    | .ok gs => pure gs
    | .error e => throw s!"rec-coins: {unit}: {e}"
  let some (_, _, vs) := gs[j]? | throw s!"rec-coins: {unit}: part {j} of {gs.size}"
  if vs.any (· ≥ o.v.size) then throw s!"rec-coins: {unit}: part {j} reads past R's opening's {o.v.size} v elements"
  checkValue c pub own "v" V_VALUE #[padTo (words (vs.map (o.v[·]!))) (RecResiduals.portWordsOf vs.size)]
    fun _ => s!"part {j}'s {vs.size} elements of R's opening's v"
  checkZero c pub own ZERO_VALUE 1 OUT_WORDS

/-- `LANES` and `H` of a level of R's openings, `rec-open-stacked/l<LANES>-h<H>` in that exact form. -/
def stackedParams (name : String) : Option (Nat × Nat) :=
  if !name.startsWith "rec-open-stacked/" then none else
  match RecOpen.params s!"rec-open/{(name.drop 17).toString}" with
  | some (L, H) => if s!"rec-open-stacked/l{L}-h{H}" == name then some (L, H) else none
  | none => none

def COINS_FORMAT : String := "verity/rec-coins/v1"

def f128Of (j : Json) : Except String F128 := do
  let #[lo, hi] ← j.getArr? | throw "an F element is not [lo, hi]"
  let (lo, hi) := (← lo.getNat?, ← hi.getNat?)
  if lo ≥ 2 ^ 64 || hi ≥ 2 ^ 64 then throw "an F element's halves are 64 bits"
  return ⟨lo.toUInt64, hi.toUInt64⟩
def f128sOf (j : Json) : Except String (Array F128) := do (← j.getArr?).mapM f128Of

def exceptOf {α : Type} (f : Json → Except String α) (j : Json) : Except String (V α) :=
  match j.getObjValAs? String "error" with
  | .ok e => pure (.error e)
  | .error _ => do pure (.ok (← f (← j.getObjVal? "ok")))

/-- A record of what check 5 reads of the inner session (`verity/rec-coins/v1`: `m`, `k_log`, the extra claims, and for
reps 0 and 1 the coins and the comb at the rep's fold points, each value or the error computing it gives), for V*'s unit
`unit`, made from the record `record`, which `record_sha512` binds. -/
def Inner.ofJson (unit : String) (j : Json) (record : String) : Except String Inner := do
  if (j.getObjValAs? String "format").toOption != some COINS_FORMAT then throw s!"rec inner coins: not a {COINS_FORMAT} object"
  if (j.getObjValAs? String "record_sha512").toOption != some (hexOf (Sha512.hash record.toUTF8)) then
    throw "rec inner coins: made from another inner session record than --rec-inner-session's"
  let claim := fun (c : Json) => do
    pure ({ value := ← f128Of (← c.getObjVal? "value"), skip := ← f128sOf (← c.getObjVal? "skip"),
            x := ← f128sOf (← c.getObjVal? "x") } : Claim)
  let extras ← exceptOf (fun x => do (← x.getArr?).mapM claim) (← j.getObjVal? "extras")
  let reps ← j.getObjValAs? (Array Json) "reps"
  if reps.size != 2 then throw "rec inner coins: not two reps"
  let rounds ← reps.mapM fun rp => do exceptOf (fun x => do (← x.getArr?).mapM f128sOf) (← rp.getObjVal? "rounds")
  let combs ← reps.mapM fun rp => do
    match rp.getObjVal? "comb" with
    | .ok .null | .error _ => pure none
    | .ok c => some <$> exceptOf f128sOf c
  let range := fun (x : Json) => do
    pure ({ kind := .net 0, slotLog := ← x.getObjValAs? Nat "slot_log", pos0 := ← x.getObjValAs? Nat "pos0",
            count := ← x.getObjValAs? Nat "count", role := .none } : Range)
  let unitRange ← match j.getObjVal? "unit_range" with
    | .ok x => exceptOf range x
    | .error _ => pure (.error "the verifier's inner coins hold no unit range")
  return { unit, m := ← j.getObjValAs? Nat "m", kLog := ← j.getObjValAs? Nat "k_log", extras
           roundsOf := fun r => rounds.getD r (.error s!"the verifier's inner coins hold no rep {r}")
           combOf := fun r _ => (combs.getD r none).getD (.error s!"rec-coins: {unit}: the verifier's inner coins hold no comb of rep {r}")
           unitRange }

/-- Where check 5 reads the inner session: its statement and record (`Files`), or a record of what it reads of them
(`Inner.ofJson`) and the record. No reader of a statement builds `coins`: `filesInput` refuses `--rec-inner-coins`
(`refuseCoins`). -/
inductive Source where
  | files (f : Files)
  | coins (j : Json) (record : String)

def Source.record : Source → String
  | .files f => f.record
  | .coins _ r => r

def Source.load (unit : String) : Source → V Inner
  | .files f => Inner.load unit f
  | .coins j r => Inner.ofJson unit j r

/-- Check 5's source at a session whose inner transcript holds candidate 3's streams (`cs`): its inner statement, with
the verifier's inner shape `i`, which `Inner.load` checks it against. -/
def Source.atC3 : Option Source → Option RecVStar.Inner → Option RecVStar.InnerCommits → Option Source
  | some (.files f), some i, some cs => some (.files (if cs.streams.isEmpty then f else { f with inner := some i }))
  | s, _, _ => s

/-- **Candidate 3's inner statement at a session's setup**: where the verifier's inner shape names a RecSparse shape, the
inner statement set up from its files (`Files.stmt`) passes `checkInnerRange`, whatever the session's unit; such a session
without the inner statement's files is refused. -/
def checkAtC3 (inner : Option RecVStar.Inner) (src : Option Source) : V Unit := do
  let some i := inner | return
  if i.sparse.isNone then return
  let some (.files f) := src
    | throw ("rec-inner: candidate 3's session needs the inner statement (--rec-inner-circuit, --rec-inner-public, " ++
        "--rec-inner-partition, --rec-inner-session): its sparse range is checked on it")
  checkInnerRange i (← f.stmt "the inner session").2

/-! ### Each level's committed tops: RecCap -/

def CAP_ZERO_VALUE : String := "rec-cap-zero"

/-- **RecCap holds every node of level `l`'s committed tops at a fixed row** (`VStar.capName l N`, `VStar.capUnit`): the
reads file's `top` reads `rec-top-L<l>`, a registration of `N` rows of `VStar.CAP_TOP_WORDS` words, instance `i` reading
row `i` whatever the coins draw, its table labels its instances `0` to `n - 1` in order (`Registered.unitIndices`), as
V*'s stage writes it, so instance `i` reads node `i`, and its `out` is V*'s `N` zero rows of `VStar.CAP_OUT_WORDS` words
under `zeroSalt` (`checkRows` at `rec-cap-zero`), so every held row's words past its node are zero.
`RecVStar.checkTopRows` holds each table row of `top` to its node of `root_B` or of the reps' level-`l` tops
(`RecVStar.topRow`), and `RecVStar.checkShared` holds `top` and level `l`'s `out` to one registration. -/
def checkCap (c : Circuit) (pub : Public) (own : Array Registered.Port) (l N : Nat) : V Unit := do
  let unit := c.unit.name
  let some reg := own.find? (·.name == "top")
    | throw s!"rec-coins: {unit}: the verifier's registered reads hold no port top: the cap's rows would be the prover's"
  if reg.value != s!"rec-top-L{l}" then
    throw s!"rec-coins: {unit}: port top reads {reg.value}, not level {l}'s tops rec-top-L{l}"
  if reg.leaves != N || reg.words != some VStar.CAP_TOP_WORDS || reg.wordBits != 16 then
    throw (s!"rec-coins: {unit}: {reg.value}: {reg.leaves} rows of {reg.words.getD 0} words registered, the cap's " ++
      s!"{N} of {VStar.CAP_TOP_WORDS}")
  if reg.reads != (Array.range N).map some then
    throw s!"rec-coins: {unit}: {reg.value}: the reads file does not read node i of the cap in instance i"
  let idx ← Registered.unitIndices pub
  if idx != Array.range pub.n then
    throw (s!"rec-coins: {unit}: its instances are labelled {idx}, not 0 to {pub.n} - 1 in order, so instance i " ++
      "would not read node i")
  checkRows (fun _ => zeroSalt CAP_ZERO_VALUE) c pub own "out" CAP_ZERO_VALUE (zeroRows N VStar.CAP_OUT_WORDS)
    fun i => s!"node {i}'s zero outputs"

/-- V*'s statement `st` against the verifier's coins: a `rec-open/…`, `rec-residuals/…`, `rec-sparse/…`,
`rec-open-stacked/…` or `rec-opening/…` unit is refused unless the verifier holds its registered reads and the inner session
(`files`), candidate 3's besides the last three (`c3`), and its verifier-registered values are the ones the coins give
(`checkOpen`, `checkResiduals`, `checkSparse`, `checkStackedLevel`, `checkOpening`); a `rec-cap/…` unit, which reads no
coins, unless the verifier holds its registered reads and the inner session, as every other V* unit, and its reads hold
the cap at fixed rows (`checkCap`, last, so no other unit's branch is behind it); a `rec-cap/…` unit whose name is
not `VStar.capName l N`'s (`VStar.capParams`) is refused; any other unit passes. -/
def check (st : Stmt) (own : Option (Array Registered.Port)) (files : Option Source) (c3 : Option C3 := none) :
    V Unit := do
  let unit := st.c.unit.name
  let o := RecOpen.params unit
  let res := RecResiduals.params unit
  let sp := RecSparse.params unit
  let so := stackedParams unit
  let op := RecOpening.params unit
  let cap := VStar.capParams unit
  if o.isNone && res.isNone && sp.isNone && so.isNone && op.isNone && cap.isNone && !unit.startsWith "rec-cap/" then
    return
  let some own := own
    | throw s!"rec-coins: {unit}: the verifier holds no registered reads (--registered): its coefficients would be the prover's"
  let needC3 : V C3 := match c3 with
    | some c => pure c
    | none => throw (s!"rec-coins: {unit}: the verifier holds no inner transcript and inner shape of candidate 3's session " ++
        "(--rec-inner-transcript, --rec-inner with --sparse), so no coins to check its coefficients against")
  if let some (L, H) := so then return ← checkStackedLevel (← needC3) st.c st.pub own L H
  if let some (m, cl, j) := op then return ← checkOpening (← needC3) st.c st.pub own m cl j
  let some f := files
    | throw s!"rec-coins: {unit}: the verifier holds no inner session (--rec-inner-circuit, --rec-inner-public, --rec-inner-partition, --rec-inner-session), so no coins to check its coefficients against"
  let inn ← f.load unit
  if let some sh := sp then return ← checkSparse inn (← needC3) st.c st.pub own sh
  match o, res with
  | some (LANES, H), _ => checkOpen inn st.c st.pub own LANES H
  | _, some (m, k, cl, j) => checkResiduals inn st.c st.pub own m k cl j
  | none, none => match cap with
    | some (l, N) => checkCap st.c st.pub own l N
    | none => throw s!"rec-coins: {unit}: a rec-cap/ unit not named rec-cap/n<N> or rec-cap/n<N>-l<l>"

/-- A circuit-table level `l`'s `out` port in the reads file: level `l`'s tops `rec-top-L<l>` (`rec_outer.TopValue`),
instance `i` (rep `i / q`'s opening `i % q`, `checkOpen`'s order) reading node `first + pos >>> H` of them, `pos` the
opening's query position and `H` its siblings below the cap: `first` is 0 for rep 0, and for rep 1 0 at level 0 (both
reps' one `root_B`) and above it the node count of rep 0's top, `top0` (`RecVStar.topRow`'s rows,
`TopValue.position`). -/
def checkLevelTopReads (unit : String) (own : Array Registered.Port) (l q H top0 : Nat) (pos0 pos1 : Array Nat) :
    V Unit := do
  let some reg := own.find? (·.name == "out")
    | throw s!"rec-coins: {unit}: the verifier's registered reads hold no port out: level {l}'s tops would be the prover's"
  if reg.value != s!"rec-top-L{l}" then
    throw s!"rec-coins: {unit}: port out reads {reg.value}, not level {l}'s tops rec-top-L{l}"
  let first := if l == 0 then 0 else top0
  let want := (Array.range (2 * q)).map fun i =>
    some (if i < q then pos0[i]! >>> H else first + (pos1[i - q]! >>> H))
  if reg.reads != want then
    throw (s!"rec-coins: {unit}: rec-top-L{l}: the reads file does not read rep r's opening j at node pos[j] >>> {H} " ++
      s!"of its rep's top, rep 1's after rep 0's {first} rows")

/-- **A circuit-table level's openings climb to the coin server's tops**: at a `rec-open/l<LANES>-h<H>` statement, the
level `l` of the inner schedule `checkOpen` finds, its `out` reads `rec-top-L<l>` at each opening's node above its `H`
siblings at its rep's coins (`checkLevelTopReads`), rep 1's rows after rep 0's level-`l` top as the coin server recorded
it (`lk`). -/
def checkOpenTops (inn : Inner) (c : Circuit) (own : Array Registered.Port) (lk : RecVStar.InnerLink) (LANES H : Nat) :
    V Unit := do
  let unit := c.unit.name
  let (sh, extras) ← inn.algebra
  let lv := sh.sched.levels
  let [l] := (List.range lv.size).filter fun l => 2 ^ lv[l]!.lanes == LANES && lv[l]!.d - lv[l]!.cap == H
    | throw s!"rec-coins: {unit}: no one level of the inner schedule (m {sh.m}) has {LANES} lanes and {H} siblings below its cap"
  let q := lv[l]!.q
  let pos ← #[0, 1].mapM fun (r : Nat) => do
    let rep ← inn.rep sh extras r
    match rep.info.lg.conss[l]? with
    | some cs => if cs.pos.size == q then pure cs.pos else
        throw s!"rec-coins: {unit}: rep {r}: the verifier's coins give no {q} queries at level {l}"
    | none => throw s!"rec-coins: {unit}: rep {r}: the verifier's coins give no {q} queries at level {l}"
  let top0 := ((lk.tops[0]?.bind (·[l - 1]?)).map (·.2.size)).getD 0
  checkLevelTopReads unit own l q H top0 pos[0]! pos[1]!

/-- **Check 5's link points and its levels' tops are the coin server's**: a `rec-open/…`, `rec-residuals/…` or
`rec-sparse/…` statement, whose check 5 takes the extra claims at the inner session's record's link points, is refused
unless the verifier holds the inner transcript (`link`, `RecVStar.innerLink`, which holds those points to the ones the
coin server drew); and at a circuit-table level its `out` reads the tops the coin server recorded (`checkOpenTops`). A
RecCap statement is refused without the inner transcript, whose `root_B` (level 0) or both reps' level-`l` tops (a later
level) its rows are held to (`RecVStar.checkTopRows`), and unless those have its `N` nodes. Any other unit passes. -/
def checkLink (st : Stmt) (own : Option (Array Registered.Port)) (files : Option Source)
    (link : Option RecVStar.InnerLink) : V Unit := do
  let unit := st.c.unit.name
  let o := RecOpen.params unit
  let cap := VStar.capParams unit
  if o.isNone && (RecResiduals.params unit).isNone && (RecSparse.params unit).isNone && cap.isNone then return
  let some lk := link
    | throw (s!"rec-coins: {unit}: the verifier holds no inner transcript (--rec-inner-transcript with " ++
        "--rec-inner-session): check 5's link points and the tops its levels climb to would be the prover's")
  if let some (l, N) := cap then
    if l == 0 then
      if lk.root.size != N then
        throw s!"rec-coins: {unit}: holds {N} nodes of the cap, and the coin server's {lk.table} root_B has {lk.root.size}"
    else
      match lk.tops[0]?.bind (·[l - 1]?), lk.tops[1]?.bind (·[l - 1]?) with
      | some (_, t0), some (_, t1) =>
        if t0.size + t1.size != N then
          throw s!"rec-coins: {unit}: holds {N} nodes of level {l}'s tops, and the coin server's reps' level {l} tops have {t0.size + t1.size}"
      | _, _ => throw s!"rec-coins: {unit}: the coin server's record has no level {l} top of both reps"
    return
  let some (LANES, H) := o | return
  let some own := own
    | throw s!"rec-coins: {unit}: the verifier holds no registered reads (--registered): its tops would be the prover's"
  let some f := files
    | throw s!"rec-coins: {unit}: the verifier holds no inner session (--rec-inner-circuit, --rec-inner-public, --rec-inner-partition, --rec-inner-session), so no coins to place its openings' tops at"
  checkOpenTops (← f.load unit) st.c own lk LANES H

/-- The value after `k` in `args`. -/
def argOf (args : List String) (k : String) : Option String :=
  match args.dropWhile (· != k) with
  | _ :: v :: _ => some v
  | _ => none

/-- The lookup tables in `dir`, as `verify --tables` reads them: each `MUFU_TABLES` table whose `<name>.u32` is there. -/
def tablesIn (dir : String) : IO (List (String × ByteArray)) := do
  let mut out := []
  for t in MUFU_TABLES do
    let p : System.FilePath := dir ++ s!"/{t.name}.u32"
    if (← p.pathExists) then out := out ++ [(t.name, (← IO.FS.readBinFile p))]
  return out

/-- `--rec-inner-statement ID [--rec-inner-statement-coins os|seed] [--rec-inner-tables DIR] [--rec-inner-draw F]`:
how Flock's plain verifier is told to set the inner statement up (`InnerForm`; coins `seed` unless given), or none. -/
def formInput (args : List String) : IO (Except String (Option InnerForm)) := do
  let some s := argOf args "--rec-inner-statement"
    | return if (argOf args "--rec-inner-statement-coins").isSome || (argOf args "--rec-inner-tables").isSome ||
          (argOf args "--rec-inner-draw").isSome then
        .error "--rec-inner-statement-coins, --rec-inner-tables and --rec-inner-draw go with --rec-inner-statement"
      else .ok none
  let os ← match argOf args "--rec-inner-statement-coins" with
    | none | some "seed" => pure false
    | some "os" => pure true
    | some c => return .error s!"--rec-inner-statement-coins {c}: not os or seed"
  try
    let tables ← match argOf args "--rec-inner-tables" with
      | some d => tablesIn d
      | none => pure []
    let draw ← (argOf args "--rec-inner-draw").mapM IO.FS.readBinFile
    return .ok (some ⟨s, os, tables, draw⟩)
  catch e => return .error s!"rec inner statement: {e}"

/-- `--rec-inner-circuit F --rec-inner-public F --rec-inner-partition F --rec-inner-session DIR`: the inner statement and
the inner session (its `session.json`), all four or none, with how the inner statement is set up (`formInput`). -/
def filesOnly (args : List String) : IO (Except String (Option Files)) := do
  match argOf args "--rec-inner-circuit", argOf args "--rec-inner-public", argOf args "--rec-inner-partition",
      argOf args "--rec-inner-session" with
  | none, none, none, none => return .ok none
  | some c, some p, some q, some s =>
    let form ← match ← formInput args with
      | .ok f => pure f
      | .error e => return .error e
    try
      let partition ← match Json.parse (← IO.FS.readFile q) with
        | .ok j => pure j
        | .error e => return .error s!"--rec-inner-partition {q}: not JSON: {e}"
      return .ok (some ⟨← IO.FS.readBinFile c, ← IO.FS.readBinFile p, partition, ← IO.FS.readFile (s ++ "/session.json"),
        none, form⟩)
    catch e => return .error s!"rec inner session: {e}"
  | _, _, _, _ =>
    return .error "--rec-inner-circuit, --rec-inner-public, --rec-inner-partition and --rec-inner-session go together"

def NO_INNER_COINS : String :=
  "--rec-inner-coins is refused: check 5 reads the inner statement's own files (--rec-inner-circuit, " ++
    "--rec-inner-public, --rec-inner-partition, --rec-inner-session)"

/-- **`--rec-inner-coins` is refused**, given a value or not: a record of what check 5 reads (`Source.coins`) is bound
only to the inner session's record (`record_sha512`), not to the coins that session drew. -/
def refuseCoins (args : List String) : Except String Unit :=
  if args.contains "--rec-inner-coins" then .error NO_INNER_COINS else .ok ()

/-- `filesOnly`, after `refuseCoins`: check 5's source is the inner statement's files or none. -/
def filesInput (args : List String) : IO (Except String (Option Source)) := do
  if let .error e := refuseCoins args then return .error e
  return (← filesOnly args).map (·.map .files)

end RecCoins
end Flock
