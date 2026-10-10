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
root of the verifier's rows, which every instance reads at its own index (`checkValue`).
-/
import Flock.Keccak
import Flock.Registered
import Flock.Firewall
import Flock.Verify
import Flock.Zk

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

/-- F elements as u16 words, 8 each, `lo` then `hi` little-endian (`rec_open.words`, rec_open.py:120). -/
def words (xs : Array F128) : Array Nat :=
  let u16s := fun (x : UInt64) => (Array.range 4).map fun k => ((x >>> (16 * k).toUInt64) &&& 0xFFFF).toNat
  xs.flatMap fun x => u16s x.lo ++ u16s x.hi

def padTo (ws : Array Nat) (n : Nat) : Array Nat := ws ++ Array.replicate (n - ws.size) 0

/-- Opening `j`'s coefficients at `kap = κ[ℓ]` and `e = eq(α_ℓ)[j]`: `κ[ℓ][0]`'s `LANES`, then `κ[ℓ][1]`'s, each times `e`
(`rec_outer.coefficients`, rec_outer.py:159; the proofs' `coefAt`, VBridge/Inputs.lean:41). -/
def coefAt (kap : Array (Array F128)) (LANES : Nat) (e : F128) : Array F128 :=
  (Array.range LANES).map (fun l => (kap[0]!)[l]! * e) ++ (Array.range LANES).map (fun l => (kap[1]!)[l]! * e)

/-- `dirs`: the position's low `H` bits, then zero words (`rec_open.dirs_words`, rec_open.py:155). -/
def dirsRow (pos H : Nat) : Array Nat := #[pos % 2 ^ H] ++ Array.replicate (DIRS_WORDS - 1) 0

/-- A registered row's `b ‖ c` under `salt` (`registered.Registered.commit`, registered.py:139: `sha512/row/v1` of its
u16 words, zero-padded to whole blocks, under hm96-sha512's default key). -/
def commitRow (row : Array Nat) (salt : ByteArray) : ByteArray :=
  Hm96.Default512.commitString (Firewall.rowDigest (padTo row (Registered.rowWords row.size 16))) salt

/-- The inner statement and the inner session's record, as the verifier holds them. -/
structure Files where
  circuit : ByteArray
  pub : ByteArray
  partition : Json
  record : String

/-- The inner session of V*'s unit `unit`: its statement's setup, its record and its table. -/
structure Inner where
  unit : String
  setup : Setup
  sess : Session
  table : String

/-- A recursive session's inner statement (PROTOCOL.md §16.16), on the coins the coin server draws. -/
def innerTags : Tags := Tags.circuitPlainLeaves.withCoins Coins.os

/-- The inner statement as the recursive audit's event reads it (`AcceptsStmt`): set up from its circuit and public files
on the coin server's coins, with an `m` that `Zk.schedule` accepts, and its session's record decoded. -/
def Inner.load (unit : String) (f : Files) : V Inner := do
  let (spec, sts) ← match Stmt.setupTables innerTags f.circuit f.pub Coins.os 1 (partition := some f.partition) with
    | .ok x => pure x
    | .error e => throw s!"rec-coins: {unit}: the inner statement: {e}"
  let some st := sts[0]? | throw s!"rec-coins: {unit}: the inner statement has no table"
  if let .error e := Zk.schedule st.m then throw s!"rec-coins: {unit}: the inner statement's schedule: {e}"
  let sess ← match Record.decode spec f.record with
    | .ok s => pure s
    | .error e => throw s!"rec-coins: {unit}: the inner session's record: {e}"
  return ⟨unit, Setup.ofCircuit st, sess, spec.tables[0]!⟩

/-- Rep `r`'s coins, round by round: its stream's, as `verifyRep` replays them (Verify.lean:40). -/
def Inner.rounds (inn : Inner) (r : Nat) : V (Array (Array F128)) := do
  let name := s!"{inn.table}/rep{r}"
  let some s := inn.sess.streams.find? (·.name == name) | throw s!"rec-coins: {inn.unit}: the inner record has no stream {name}"
  return s.rounds.map (·.coins)

/-- The verifier's shape and extra claims: the inner statement's `m` and `k_log`, its extra claims at the record's link
points (Verify.lean:41), two claims besides them. -/
def Inner.algebra (inn : Inner) : V (Shape × Array Claim) := do
  let extras ← inn.setup.extra inn.sess.points
  let some sh := RecResiduals.shape? inn.setup.m inn.setup.kLog (2 + extras.size)
    | throw s!"rec-coins: {inn.unit}: the inner statement's shape (m {inn.setup.m}, k_log {inn.setup.kLog}) has no algebra"
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

/-- The comb at the rep's fold points (`CircuitFold.folded`, as V*'s `InnerFold.lean` computes it). -/
def Inner.comb (inn : Inner) (rep : Rep) : V (Array F128) := do
  let f := rep.info.lc.fold
  let comb ← inn.setup.fold.folded f.alpha f.beta f.lw f.rhoIn f.ts
  if comb.size != 64 then throw s!"rec-coins: {inn.unit}: a comb of {comb.size}"
  return comb

/-- Port `port` of the statement is the value `value` the verifier registers itself, of `rows`: the verifier's registered
reads hold it as `value` (`own`), of `rows.size` rows of their width, unit `i` reading row `i`; every table row's `b ‖ c`
is the verifier's row at its position under `verifierSalt` of META's `program_sha512`; and the registered root is the
frame-v3-sha512 root of those rows (`Registered.rowLeaf`). `what i` names row `i`. -/
def checkValue (c : Circuit) (pub : Public) (own : Array Registered.Port) (port value : String) (rows : Array (Array Nat))
    (what : Nat → String) : V Unit := do
  let unit := c.unit.name
  let some reg := own.find? (·.name == port)
    | throw s!"rec-coins: {unit}: the verifier's registered reads hold no port {port}: its {value} would be the prover's"
  if reg.value != value then throw s!"rec-coins: {unit}: port {port} reads {reg.value}, not the verifier's {value}"
  let n := rows.size
  let w := (rows[0]?.map (·.size)).getD 0
  if reg.leaves != n || reg.words != some w || reg.wordBits != 16 then
    throw s!"rec-coins: {unit}: {value}: {reg.leaves} rows of {reg.words.getD 0} words registered, the verifier computes {n} of {w}"
  if reg.reads != (Array.range n).map some then
    throw s!"rec-coins: {unit}: {value}: the reads file does not read row i of the verifier's {n} rows in unit i"
  let .ok prog := c.metaJ.getObjValAs? String "program_sha512" | throw s!"rec-coins: {unit}: META names no program_sha512"
  let bcs := (Array.range n).map fun i => commitRow rows[i]! (verifierSalt prog value i)
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
    let comb ← if vs.any (· ≥ base.size) then inn.comb rep else pure ZERO_COMB
    let full := base ++ comb
    pure (padTo (words (vs.map (full[·]!))) (RecResiduals.portWordsOf vs.size))
  checkValue c pub own "v" V_VALUE rows fun r => s!"rep {r}: part {j}'s {vs.size} elements of v"

/-- V*'s statement `st` against the verifier's coins: a `rec-open/…` or `rec-residuals/…` unit is refused unless the
verifier holds its registered reads and the inner session (`files`), and its verifier-registered values are the ones the
coins give (`checkOpen`, `checkResiduals`); any other unit passes. -/
def check (st : Stmt) (own : Option (Array Registered.Port)) (files : Option Files) : V Unit := do
  let unit := st.c.unit.name
  let o := RecOpen.params unit
  let res := RecResiduals.params unit
  if o.isNone && res.isNone then return
  let some own := own
    | throw s!"rec-coins: {unit}: the verifier holds no registered reads (--registered): its coefficients would be the prover's"
  let some f := files
    | throw s!"rec-coins: {unit}: the verifier holds no inner session (--rec-inner-circuit, --rec-inner-public, --rec-inner-partition, --rec-inner-session), so no coins to check its coefficients against"
  let inn ← Inner.load unit f
  match o, res with
  | some (LANES, H), _ => checkOpen inn st.c st.pub own LANES H
  | _, some (m, k, cl, j) => checkResiduals inn st.c st.pub own m k cl j
  | none, none => pure ()

/-- The value after `k` in `args`. -/
def argOf (args : List String) (k : String) : Option String :=
  match args.dropWhile (· != k) with
  | _ :: v :: _ => some v
  | _ => none

/-- `--rec-inner-circuit F --rec-inner-public F --rec-inner-partition F --rec-inner-session DIR`: the inner statement and
the inner session (its `session.json`), all four or none. -/
def filesInput (args : List String) : IO (Except String (Option Files)) := do
  match argOf args "--rec-inner-circuit", argOf args "--rec-inner-public", argOf args "--rec-inner-partition",
      argOf args "--rec-inner-session" with
  | none, none, none, none => return .ok none
  | some c, some p, some q, some s =>
    try
      let partition ← match Json.parse (← IO.FS.readFile q) with
        | .ok j => pure j
        | .error e => return .error s!"--rec-inner-partition {q}: not JSON: {e}"
      return .ok (some ⟨← IO.FS.readBinFile c, ← IO.FS.readBinFile p, partition, ← IO.FS.readFile (s ++ "/session.json")⟩)
    catch e => return .error s!"rec inner session: {e}"
  | _, _, _, _ =>
    return .error "--rec-inner-circuit, --rec-inner-public, --rec-inner-partition and --rec-inner-session go together"

end RecCoins
end Flock
