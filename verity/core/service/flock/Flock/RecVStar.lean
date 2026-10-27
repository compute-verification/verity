/-
V*'s statements over one inner session, checked by the verifier against V*'s own executable Lean (`VStar`, in this
package so that the verifier can read it): `VStar.statements` at the inner session's shape (`VStar/Stage.lean:48-58`: one
statement per Ligerito level of `fast100(m)`, `rec-open/l<2^lanes>-h<d − cap>` over that level's `2 q` openings, then one
per part of the rep's algebra, `rec-residuals/m<m>-k<k_log>-c<claims>-p<j>` over the two reps; over a private inner
session, `VStar.statementsHolo`, the parts `specHolo`'s and each part's file saying `holo` in META), and for candidate 3's
session its RecSparse (`rec-sparse/n<NU>-e<ELL>-s<S_LOG>-t<T_LOG>` over the two reps), R's levels under names of their own
(`rec-open-stacked/l<2^lanes>-h<d − cap>`, never the circuit table's) and the parts of R's opening
(`rec-opening/m<m>-c<claims>-p<j>`, at META's `stacked`; `candidate3Statements`), then RecCap at each later level
(`rec-cap/n<N>-l<l>`, `VStar.levelCapStatements`, the reps' schedule's tops: `checkTopsListed`) and RecCap at level 0
(`rec-cap/n<N>`, last; `checkCapListed`). A circuit whose unit is one
of V*'s is refused unless the verifier names which of these statements the session is (`--rec-part J`) and its unit is that
statement's, its population is that statement's instance count, the verifier's own bindings (`--rec-bindings F`) are its
partition object's and program's, META binds exactly those, and the circuit file is byte for byte the one `vstar stage`
writes for that statement under the verifier's bindings (`circuitFile`).

The inner shape is the verifier's own (`rec-inner`): `Stmt.setupTables` on the inner statement, as `InnerFold.lean` sets it
up for V*'s native side, read once, since the inner circuit can be tens of GB; `verify --rec-inner F` reads the line it
printed.
-/
import Flock.HmRow
import Flock.ProvedScope
import Flock.RecOpening
import Flock.Registered
import VStar.Cli
import VStar.Compose
import VStar.Stage

namespace Flock
namespace RecVStar

open Lean

/-- R's opening-only shape, `Shape(m, 6, claims, opening_only=True)`, as staging committed R (`rec_session.stage_stacked`):
`m` R's variables (`rec_sparse.table_m` of the bucket) and its claims, the two reps' sparse claims. The registered top is a
cap of `2^cap` nodes, which fixes neither: three `m` share a cap depth. -/
structure StackedShape where
  m : Nat
  claims : Nat
  deriving BEq, Repr

/-- META's key for R's shape (`rec_session.STACKED_SHAPE_KEY`), `{"m", "claims"}`. -/
def STACKED_SHAPE_KEY : String := "stacked"
/-- R is opened at the two reps' sparse claims (`rec_live.REPS`, `rec_vstage.stacked_reads`). -/
def STACKED_CLAIMS : Nat := 2

/-- RecSparse shapes compared and shown field by field, here rather than derived in `Flock.RecSparse`, which the circuit
guarantees read. -/
instance instBEqShape : BEq RecSparse.Shape := ⟨fun a b => a.nu == b.nu && a.ell == b.ell && a.sLog == b.sLog && a.tLog == b.tLog⟩
instance instReprShape : Repr RecSparse.Shape := ⟨fun s _ => repr (s.nu, s.ell, s.sLog, s.tLog)⟩

/-- The inner session's shape: its table's `m` and `k_log` (`Setup.ofCircuit`), its claims, two and one per region and
link point (`Stmt.extraClaims`, `rec_vstage.verifier`'s `2 + len(extras)`), its table's regions (`Stmt.regions`), and
whether its circuits are private (`holo`, `rec_algebra.Shape.holo`): the verifier's own `--hide`, never the prover's.
Candidate 3's session (`rec_live.schedule_for` with `sparse`) also has its sparse phase's RecSparse shape (`sparse`, the
bucket's, the verifier's own `--sparse`) and R's shape (`stacked`, the inner circuit's META's), which its V* list needs
beside it (`candidate3Statements`). -/
structure Inner where
  m : Nat
  kLog : Nat
  claims : Nat
  regions : Nat
  holo : Bool := false
  sparse : Option RecSparse.Shape := none
  stacked : Option StackedShape := none
  deriving Repr, BEq, Inhabited

/-- A recursive session's inner statement (§16.16) on the verifier's live coins, as `87-rec-private.sh` runs `inner-fold`. -/
def innerTags : Tags := Tags.circuitPlainLeaves.withCoins Coins.os

/-- The inner statement's shape, from its circuit, public file and the verifier's own partition object and program. -/
def innerOf (circuitB pubB : ByteArray) (partition : Option Json) (program : Option ByteArray) (holo : Bool := false) :
    V Inner := do
  let (spec, sts) ← Stmt.setupTables innerTags circuitB pubB Coins.os 1 (partition := partition) (program := program)
  let some st := sts[0]? | throw "rec-inner: the inner statement has no table"
  pure { m := st.m, kLog := st.c.kLog, claims := 2 + spec.points * st.regions.size, regions := st.regions.size, holo }

def FORMAT : String := "verity/rec-inner/v1"

/-- `verity/rec-inner/v1`'s key for `Inner.holo`, written only when true; absent means false. -/
def HOLO_KEY : String := "holo"

/-- `verity/rec-inner/v1`'s key for `Inner.sparse`, `[NU, ELL, S_LOG, T_LOG]`, written only when present; `Inner.stacked`
is under META's own key (`STACKED_SHAPE_KEY`), `{"m", "claims"}`, written only when present. -/
def SPARSE_KEY : String := "sparse"

def Inner.toJson (i : Inner) : Json :=
  Json.mkObj ([("format", .str FORMAT), ("m", Lean.toJson i.m), ("k_log", Lean.toJson i.kLog),
    ("claims", Lean.toJson i.claims), ("regions", Lean.toJson i.regions)] ++ (if i.holo then [(HOLO_KEY, .bool true)] else []) ++
    (i.sparse.map fun sh => (SPARSE_KEY, Lean.toJson [sh.nu, sh.ell, sh.sLog, sh.tLog])).toList ++
    (i.stacked.map fun r => (STACKED_SHAPE_KEY, Json.mkObj [("m", Lean.toJson r.m), ("claims", Lean.toJson r.claims)])).toList)

def Inner.ofJson (j : Json) : Except String Inner := do
  if (j.getObjValAs? String "format").toOption != some FORMAT then throw s!"rec-inner: not a {FORMAT} object"
  let holo ← match j.getObjVal? HOLO_KEY with
    | .error _ => pure false
    | .ok (.bool b) => pure b
    | .ok _ => throw s!"rec-inner: {HOLO_KEY} is not a boolean"
  let sparse ← match j.getObjVal? SPARSE_KEY with
    | .error _ => pure none
    | .ok v => match Lean.fromJson? (α := List Nat) v with
      | .ok [nu, ell, sLog, tLog] => pure (some ({ nu, ell, sLog, tLog } : RecSparse.Shape))
      | _ => throw s!"rec-inner: {SPARSE_KEY} is not a RecSparse shape [NU, ELL, S_LOG, T_LOG]"
  let stacked ← match j.getObjVal? STACKED_SHAPE_KEY with
    | .error _ => pure none
    | .ok v => match v.getObjValAs? Nat "m", v.getObjValAs? Nat "claims" with
      | .ok m, .ok claims => pure (some ({ m, claims } : StackedShape))
      | _, _ => throw s!"rec-inner: {STACKED_SHAPE_KEY} is not R's shape (m, claims)"
  pure { m := ← j.getObjValAs? Nat "m", kLog := ← j.getObjValAs? Nat "k_log", claims := ← j.getObjValAs? Nat "claims",
         regions := ← j.getObjValAs? Nat "regions", holo, sparse, stacked }

/-- R's variables at the bucket's RecSparse shape (`rec_sparse.table_m`): its cells' `NU + P_LOG + T_LOG`, at least
`PAD_TO` (`registered_claim`'s padded point). -/
def PAD_TO : Nat := 22
def tableM (sp : RecSparse.Shape) : Nat := max PAD_TO (sp.nu + sp.pLog + sp.tLog)

/-- The table's round whose message is `h` (`rec_algebra.h_round`, at the holo shape `RecAlgebra.rounds` lists it in): after
the zerocheck's first round, its `m - 6` rounds and `fa, fb`, the lincheck's `k_log - 6` and `z_partial`'s. RecSparse reads
it as its port `h` (`rec_vstage.stage_sparse`). -/
def hRound (m kLog : Nat) : Nat := m + kLog - 9

/-- The row port `p` of R's opening's w (`rec_residuals.spec` at an opening-only shape): a port `m<i>` per round of its `nr`,
the running sums' `acc0` and `acc1`, then a port `claim<k>` per claim. -/
def openingPortName (nr p : Nat) : String :=
  if p < nr then s!"m{p}" else if p == nr then "acc0" else if p == nr + 1 then "acc1" else s!"claim{p - nr - 2}"

/-- The parts of R's registered opening at R's shape `r` (`rec_residuals.stage_opening`): `InnerRepCheck{S}` at
`Shape(m, 6, claims, opening_only=True)` (`RecOpening.structureOf`), cut by `RecResiduals.groups` as `RecOpening.part`
cuts it, each part's row ports under their names (`openingPortName`). -/
def openingParts (r : StackedShape) : Except String (Array VStar.Stmt) := do
  let some sh := RecOpening.shape? r.m r.claims
    | throw s!"R's shape (m {r.m}, claims {r.claims}): rec-opening's m has a fast100 schedule, claims at least 1"
  let S ← RecOpening.structureOf sh
  let sizes := RecOpening.portSizes sh
  if sizes.foldl (· + ·) 0 != S.nw then throw s!"R's opening: row ports of {sizes.foldl (· + ·) 0} elements for a w of {S.nw}"
  let nr := (RecOpening.rounds sh).size
  let gs ← RecResiduals.groups S sizes
  let mut out : Array VStar.Stmt := #[]
  for j in [0:gs.size] do
    let (R, P, V) := gs[j]!
    let S' ← RecResiduals.partS S sizes R P V
    out := out.push (.opening r.m r.claims j S' (P.toList.map fun p => (openingPortName nr p, sizes[p]!)) V.size)
  pure out

/-- **Candidate 3's statements** at the inner shape `i` (`rec_vstage.stage --sparse`): the sparse phase's RecSparse over
both reps at the bucket's shape (`i.sparse`, its `h` the table's `hRound`); then R's (`i.stacked`, META's): its levels at
`fast100(m)` over its one stream's `q` openings each (`rec-open-stacked/l<2^lanes>-h<d − cap>`), then the parts of its
opening (`openingParts`). R is opened at the two reps' sparse claims, at the bucket's
`table_m`, so a shape of R otherwise is refused; so is R without the sparse phase, whose claims it opens, and the sparse
phase without R, which opens them. -/
def candidate3Statements (i : Inner) : Except String (Array VStar.Stmt) := do
  match i.sparse, i.stacked with
  | none, none => pure #[]
  | none, some r =>
    throw (s!"candidate 3's V* list: the inner shape names R's shape (m {r.m}, claims {r.claims}) and no RecSparse shape " ++
      "(rec-inner --sparse): R opens the sparse phase's claims, so RecSparse is missing")
  | some sp, r? =>
    if !sp.valid then
      throw s!"candidate 3's V* list: {RecSparse.unitName sp}: rec-sparse's NU is at least 1, ELL at least 6 and T_LOG at most 16"
    let sparse := VStar.Stmt.sparse sp (hRound i.m i.kLog)
    let some r := r? | throw (s!"candidate 3's V* list: the inner shape names RecSparse ({RecSparse.unitName sp}) and no R's " ++
      s!"shape ({STACKED_SHAPE_KEY}): R opens the sparse phases' claims, so R is missing")
    if r.claims != STACKED_CLAIMS then
      throw s!"candidate 3's V* list: R's claims {r.claims}, not the two reps' sparse claims ({STACKED_CLAIMS})"
    if r.m != tableM sp then
      throw s!"candidate 3's V* list: R's m {r.m} is not {RecSparse.unitName sp}'s table_m {tableM sp}"
    let some sch := fast100 r.m | throw s!"candidate 3's V* list: R's m {r.m} has no fast100 schedule"
    let levels := (List.range sch.levels.size).toArray.map fun l =>
      let lv := sch.levels[l]!
      VStar.Stmt.recOpenStacked l (2 ^ lv.lanes) (lv.d - lv.cap) lv.q
    pure (#[sparse] ++ levels ++ (← openingParts r))

/-- **The sparse phase is a holo session's**: RecSparse reads `h`, the table's round `hRound`, which a rep sends only
where its circuits are private (`holoRepRounds`), so an inner shape naming a RecSparse shape (`rec-inner --sparse`) at
public circuits is refused. -/
def checkSparseHolo (i : Inner) : Except String Unit :=
  match i.sparse with
  | some sp => if i.holo then .ok () else
      .error (s!"the inner shape names a RecSparse shape ({RecSparse.unitName sp}) at public circuits: the sparse " ++
        "phase reads h, which only a holo inner session (rec-inner --hide) sends")
  | none => .ok ()

/-- Candidate 3's V* list against its inner shape (`checkCandidate3`'s list checks). -/
def checkCandidate3Lists (i : Inner) (ss : Array VStar.Stmt) : Except String Unit := do
  if let some sp := i.sparse then
    if !ss.any (fun | .sparse sh _ => sh == sp | _ => false) then
      throw s!"candidate 3's V* list has no RecSparse statement ({RecSparse.unitName sp}), and the inner shape names its sparse phase"
  if let some r := i.stacked then
    let some sch := fast100 r.m | throw s!"candidate 3's V* list: R's m {r.m} has no fast100 schedule"
    for l in [0:sch.levels.size] do
      if !ss.any (fun | .recOpenStacked l' .. => l' == l | _ => false) then
        throw s!"candidate 3's V* list has no level {l} of R's openings (rec-open-stacked/…), and the inner shape names R (m {r.m}, claims {r.claims})"
    if !ss.any (fun | .opening .. => true | _ => false) then
      throw s!"candidate 3's V* list has no part of R's opening ({RecOpening.unitName r.m r.claims 0}), and the inner shape names R (m {r.m}, claims {r.claims})"
    for s in ss do
      if let .opening m c .. := s then
        if m != r.m || c != r.claims then
          throw s!"candidate 3's V* list: {s.name} opens R at (m {m}, claims {c}), not at R's shape (m {r.m}, claims {r.claims}, META's {STACKED_SHAPE_KEY})"

/-- **Candidate 3's V* list holds every statement its session needs**: a sparse phase (`i.sparse`) only at private
circuits (`checkSparseHolo`), and with it a RecSparse statement at its shape; with R (`i.stacked`), each of R's levels at
`fast100(m)` and R's opening, every part of it at R's shape. A list missing any of them is refused by name. -/
def checkCandidate3 (i : Inner) (ss : Array VStar.Stmt) : Except String Unit := do
  checkSparseHolo i
  checkCandidate3Lists i ss

/-- One round of a stream's schedule (`rec_live.rep_rounds`' tuples): its coin count, the node counts of the tops it
commits, its copies of level 0's top, and whether it absorbs a message (one row when it absorbs any elements). -/
abbrev SchedRound := Nat × Array Nat × Nat × Bool

/-- Python's `int.bit_length`. -/
def bitLength (n : Nat) : Nat := if n == 0 then 0 else Nat.log2 n + 1

/-- `rec_live.rep_rounds`: one fast100 rep's rounds for a table of `2^m` packed elements, `2^k_log` a block, with `claims`
claims, in `rec_algebra.run`'s order of draws: the zerocheck, the lincheck, each claim's ring switch, then Ligerito. -/
def repRounds (m kLog claims : Nat) : Option (Array SchedRound) := do
  let sch ← fast100 m
  let lv := sch.levels
  let r := lv.size - 1
  let cap := fun (l : Nat) => #[2 ^ (lv.getD l default).cap]
  let one := fun (n : Nat) (absorbs : Bool) => ((n, #[], 0, absorbs) : SchedRound)
  let level := fun (i split : Nat) =>
    let q := (lv.getD i default).q
    #[((split, cap (i + 1), 0, true) : SchedRound), one 1 true, one split false, one 1 true, one q false,
      one (bitLength (q - 1)) false, one 1 true]
  let mut out : Array SchedRound := #[(6, #[], 1, false), one (m - 13) false, one 1 true] ++
    Array.replicate (m - 5) (one 1 true) ++ #[one 1 false] ++ Array.replicate (kLog - 6) (one 1 true) ++ #[one 1 true] ++
    Array.replicate claims (one 7 true) ++ #[one claims false]
  let mut split := sch.n - sch.k0 + 1
  out := out ++ #[(sch.n, #[], 1, true), one 1 true] ++ Array.replicate sch.k0 (one 2 true) ++ level 0 split
  for i in [0:r] do
    if split < 4 then none
    out := out ++ #[one 2 false] ++ Array.replicate 3 (one 2 true)
    let ext := split - 4
    if i + 1 < r then
      split := ext + 1
      out := out ++ level (i + 1) split
    else
      let q := (lv.getD r default).q
      out := out ++ #[one q true, one (bitLength (q - 1)) false, one 1 false]
  return out

/-- The rounds of `rs` that absorb a message, in order: the `i`-th is a stream's `i`-th message round, whose `c_k` the
inner transcript records. -/
def commitRounds (rs : Array SchedRound) : List Nat :=
  (List.range rs.size).filter fun k => (rs.getD k default).2.2.2

/-- `rec_live.holo_rounds`: `repRounds` with hidden unit circuits (`Inner.holo`). z_partial's round (`rec_live.zp_round`,
`m + k_log - 7`) squeezes nothing, and a round of its own absorbs h and squeezes r′. None where that round is not
z_partial's. -/
def holoRepRounds (m kLog claims : Nat) : Option (Array SchedRound) := do
  let out ← repRounds m kLog claims
  let k := m + kLog - 7
  if out[k]? != some ((1, #[], 0, true) : SchedRound) then none
  return out.extract 0 k ++ #[((0, #[], 0, true) : SchedRound), (1, #[], 0, true)] ++ out.extract (k + 1) out.size

/-- **V*'s list holds every level's committed tops**: exactly one statement is RecCap at level 0, over the `2^cap₀` nodes
of `root_B`'s level-0 cap (`cap₀` of `fast100(m)`'s level 0), and exactly one is RecCap at each later level `l`, over both
reps' `2 · 2^cap_l` nodes of its tops, and none is at a level past the last, so every node of every level's tops is held
at a fixed row whatever the coins draw (`RecCoins.checkCap`). A list without one, or with one at another node count, is
refused. -/
def checkCapListed (i : Inner) (ss : Array VStar.Stmt) : Except String Unit := do
  let some sch := fast100 i.m | throw s!"V*'s list: the inner session's m {i.m} has no fast100 schedule"
  let some lv := sch.levels[0]? | throw s!"V*'s list: the inner session's schedule (m {i.m}) has no level 0"
  let caps := ss.toList.filterMap fun | .cap l n => some (l, n) | _ => none
  match caps.filterMap fun (l, n) => if l == 0 then some n else none with
  | [] => throw s!"V*'s list has no RecCap statement ({VStar.capName 0 (2 ^ lv.cap)}): the table's level-0 cap is held by none of its statements"
  | [n] => if n != 2 ^ lv.cap then
      throw s!"V*'s list: {VStar.capName 0 n} holds {n} nodes, and the table's level-0 cap has {2 ^ lv.cap} (2^cap₀, cap₀ {lv.cap})"
  | ns => throw s!"V*'s list has {ns.length} RecCap statements, and holds the table's level-0 cap once"
  for l in [1:sch.levels.size] do
    let want := 2 * 2 ^ (sch.levels[l]!).cap
    match caps.filterMap fun (l', n) => if l' == l then some n else none with
    | [] => throw s!"V*'s list has no RecCap statement ({VStar.capName l want}): level {l}'s tops are held by none of its statements"
    | [n] => if n != want then
        throw s!"V*'s list: {VStar.capName l n} holds {n} nodes, and level {l}'s tops have {want} (both reps' 2^cap_{l}, cap_{l} {(sch.levels[l]!).cap})"
    | ns => throw s!"V*'s list has {ns.length} RecCap statements at level {l}, and holds its tops once"
  if let some (l, n) := caps.find? fun (l, _) => sch.levels.size ≤ l then
    throw s!"V*'s list: {VStar.capName l n} holds level {l}'s tops, and the schedule (m {i.m}) has {sch.levels.size} levels"

/-- The node counts of the tops a rep's schedule commits past `root_B`, in round order: rep `x`'s `g`-th top past `root_B`
is level `g + 1`'s (`topRow`). -/
def topSizes (rs : Array SchedRound) : List Nat := rs.toList.flatMap (·.2.1.toList)

/-- **The later levels' caps hold the tops the reps' schedule commits**: the inner shape's rep schedule (`repRounds`,
`holoRepRounds` when its circuits are private, the one `checkLinkRounds` holds the coin server's tops to) commits, past
`root_B`, one top a later level of `caps` (`VStar.levelCapStatements`), in order, each of half its cap's nodes (a rep's of
both reps'). So every round that commits tops has each of them held by its level's cap. -/
def checkTopsListed (i : Inner) (caps : Array VStar.Stmt) : Except String Unit := do
  let some rs := (if i.holo then holoRepRounds else repRounds) i.m i.kLog i.claims
    | throw s!"V*'s list: the inner shape (m {i.m}, k_log {i.kLog}, claims {i.claims}) has no rep schedule"
  let want := (topSizes rs).map (2 * ·)
  let got := caps.toList.map (·.instances)
  if got != want then
    throw s!"V*'s list: the later levels' caps hold {got} nodes, and the reps' schedule commits tops of {topSizes rs} nodes a rep ({want} both reps')"

/-- `m<k>`'s `k`. -/
def mIndex (name : String) : Option Nat := if name.startsWith "m" then (name.drop 1).toString.toNat? else none

/-- `rec-c<k>`'s `k`. -/
def cIndex (value : String) : Option Nat := if value.startsWith "rec-c" then (value.drop 5).toString.toNat? else none

/-- `claim<k>`'s `k`, the name exactly as `openingPortName` writes it (no `claim01`). -/
def claimIndex (name : String) : Option Nat :=
  if name.startsWith "claim" then ((name.drop 5).toString.toNat?).filter (name == s!"claim{·}") else none

/-- RecSparse's registered claim (`rec_outer.CLAIM_VALUE`): row `r` rep `r`'s, which RecSparse's `claim` and R's opening's
`claim<k>` read. -/
def CLAIM_VALUE : String := "rec-claim"

/-- A reads file's message-round ports, those named `m<k>`, registering a `rec-c<k>` or reading `rec-claim`: each one's
name, value and reads. -/
def msgPorts (ports : Array Registered.Port) : List (String × String × Array (Option Nat)) :=
  (ports.toList.filter fun p => (mIndex p.name).isSome || (cIndex p.value).isSome || p.value == CLAIM_VALUE).map
    fun p => (p.name, p.value, p.reads)

/-- A statement's message-round ports: a part's ports `m<k>` (`VStar/Algebra.lean:385-387`), each under `rec-c<k>` read
at rows `0` and `1`, the two reps' (`rec_residuals.write`, `rec_residuals.py:452-454`); a level has none, R's included.
RecSparse's `h` is the table's h round, `rec-c<hRound>`, and its `claim` reads `rec-claim`, both at both reps' rows, rep
`r`'s at instance `r` (`rec_vstage.stage_sparse`, `rec_sparse.reads`); a part of R's opening reads each `m<i>` under
`rec-opening-c<i>` at row 0, R's one stream's (`rec_vstage.opening_messages`), and each `claim<k>` under `rec-claim` at
row `k`, rep `k`'s claim (`rec_residuals.stage_opening`). So no claim port reads another row than its own. -/
def ownPorts : VStar.Stmt → List (String × String × Array (Option Nat))
  | .inner _ _ p => p.S.ports.filterMap fun (nm, _) => (mIndex nm).map fun k => (nm, s!"rec-c{k}", #[some 0, some 1])
  | .recOpen .. => []
  | .recOpenStacked .. => []
  | .cap .. => []
  | .sparse _ hr => [("h", s!"rec-c{hr}", #[some 0, some 1]), ("claim", CLAIM_VALUE, #[some 0, some 1])]
  | .opening _ _ _ _ ports _ => ports.filterMap fun (nm, _) => match mIndex nm, claimIndex nm with
    | some k, _ => some (nm, s!"rec-opening-c{k}", #[some 0])
    | none, some k => some (nm, CLAIM_VALUE, #[some k])
    | none, none => none

/-- Whether one of `ss` registers `rec-c<k>` as a message-round port reading both reps' rows, rep `r`'s at row `r`
(`ownPorts`). -/
def roundListed (ss : Array VStar.Stmt) (k : Nat) : Bool :=
  ss.any fun s => (ownPorts s).any fun q => q.2.1 == s!"rec-c{k}" && q.2.2 == #[some 0, some 1]

/-- The words of statement `s`'s message-round port `nm`, off its description: the input row V* writes for it
(`vstarRows` at `s.unit`), an algebra part's `m<k>` its elements in whole 64-word rows (`VStar.innerUnit`'s
`portWordsOf`), RecSparse's its `RecSparse.ports` words (`VStar.sparseUnit`). None at any other statement. -/
def ownWords (s : VStar.Stmt) (nm : String) : Option Nat :=
  match s with
  | .inner _ _ p => (p.S.ports.find? (·.1 == nm)).map (VStar.portWordsOf ·.2)
  | .sparse sh _ => ((RecSparse.ports sh).find? (·.1 == nm)).map (·.2)
  | _ => none

/-- Each of `ss`'s message-round ports under a `rec-c<k>` (`ownPorts`), as `(k, words, statement, port)`. -/
def roundPorts (ss : Array VStar.Stmt) : List (Nat × Option Nat × String × String) :=
  ss.toList.flatMap fun s => (ownPorts s).filterMap fun q =>
    (cIndex q.2.1).map fun k => (k, ownWords s q.1, s.name, q.1)

/-- Why `checkRoundWords` refuses port `p` of `ps`: its round's first port has other words. -/
def roundWordsWhy (ps : List (Nat × Option Nat × String × String)) (p : Nat × Option Nat × String × String) : String :=
  let words := fun (w : Option Nat) => (w.map toString).getD "no"
  let (w0, s0, nm0) := ((ps.find? (·.1 == p.1)).map (·.2)).getD (none, "", "")
  s!"V*'s list: a rep's message round {p.1} (rec-c{p.1}) is held at {s0}'s port {nm0} as {words w0} words and at " ++
    s!"{p.2.2.1}'s port {p.2.2.2} as {words p.2.1}"

/-- **The parts that hold one message round's row agree on its words**: every port of V*'s list under one `rec-c<k>`
(`roundPorts`) has the words of the round's first such port. The deployed layout sizes the round's one row by the part
that holds it first, and every holder reads that row, so a list whose holders of one round differ is refused. One
comparison a port, over the list's description. -/
def checkRoundWords (ss : Array VStar.Stmt) : Except String Unit :=
  let ps := roundPorts ss
  match ps.find? fun p => (ps.find? (·.1 == p.1)).any (·.2.1 != p.2.1) with
  | some p => throw (roundWordsWhy ps p)
  | none => pure ()

/-- **V*'s list holds every message round a rep absorbs, at one width**: each `k`-th round of the inner shape's rep
schedule that absorbs a message (`commitRounds` of `repRounds`, `holoRepRounds` when its circuits are private), whose
`c_k` the inner transcript fixes as `rec-c<k>` (`fixedRow`), is a message-round port of one of its statements at both
reps' rows (`roundListed`), so some part holds each rep's every absorbed message whatever the coins draw, and every
holder of a round reads its row at the same words (`checkRoundWords`). A list without one, or with two holders of a
round at different words, is refused. -/
def checkRoundsListed (i : Inner) (ss : Array VStar.Stmt) : Except String Unit := do
  let some rs := (if i.holo then holoRepRounds else repRounds) i.m i.kLog i.claims
    | throw s!"V*'s list: the inner shape (m {i.m}, k_log {i.kLog}, claims {i.claims}) has no rep schedule"
  match (List.range (commitRounds rs).length).find? (!roundListed ss ·) with
  | some k => throw s!"V*'s list: a rep's message round {k} (rec-c{k}) is a message-round port of none of its statements"
  | none => checkRoundWords ss

/-- V*'s statements at the inner shape (`VStar.statementsOf`: `VStar.statements`, or `VStar.statementsHolo` when its
circuits are private, `i.holo`), at a shape `RecResiduals.check` accepts a part's name at, whose claims are two and two
per region of the inner statement (an extra claim per region at each of two link points), and where V*'s parts
(`VStar.parts (specOf vsh i.holo)`, `VStar/Parts.lean`) are as many as the verifier's own packing of its algebra
(`RecResiduals.groups (structureOf sh) (portSizes sh)`, `RecResiduals.lean:119-139`) at the same holo flag, at a holo
shape one of whose groups holds h's round (`hRound`) as a row port. A part's
circuit file, compared byte for byte (`checkFile`), carries `i.holo` as its META holo (`RecResiduals.HOLO_KEY`,
`circuitFile`). Candidate 3's statements follow the table's (`candidate3Statements`), and the list must hold every one
its session needs (`checkCandidate3`). It holds every level's committed tops (`checkCapListed`) and every message
round a rep absorbs, each round's holders at one width (`checkRoundsListed`). -/
def statements (i : Inner) : Except String (Array VStar.Stmt) := do
  let some sh0 := RecResiduals.shape? i.m i.kLog i.claims
    | throw (s!"the inner session's shape (m {i.m}, k_log {i.kLog}, claims {i.claims}) is not V*'s: a fast100 schedule, " ++
        "k_log at least 6, claims at least 2")
  let sh := { sh0 with holo := i.holo }
  if i.claims != 2 + 2 * i.regions then
    throw s!"the inner session's claims {i.claims}, not two and two per region of its {i.regions} ({2 + 2 * i.regions})"
  let vsh : VStar.Algebra.Shape := { m := i.m, kLog := i.kLog, claims := i.claims }
  let ps ← (do VStar.parts (← VStar.Algebra.specOf vsh i.holo)).mapError (s!"V*'s parts: " ++ ·)
  let gs ← (do RecResiduals.groups (← RecAlgebra.structureOf sh) (RecResiduals.portSizes sh)).mapError
    (s!"the verifier's groups: " ++ ·)
  if ps.size != gs.size then
    throw s!"at the inner session's shape (m {i.m}, k_log {i.kLog}, claims {i.claims}): V* has {ps.size} parts, the verifier's algebra {gs.size} groups"
  if i.holo && !(gs.any fun g => g.2.1.contains (hRound i.m i.kLog)) then
    throw (s!"at the holo inner session's shape (m {i.m}, k_log {i.kLog}, claims {i.claims}): no group of the " ++
      s!"verifier's algebra holds h's round m{hRound i.m i.kLog} as a row port")
  let caps ← VStar.levelCapStatements vsh
  checkTopsListed i caps
  let ss := (← VStar.statementsOf vsh i.holo) ++ (← candidate3Statements i) ++ caps ++ #[← VStar.capStatement vsh]
  checkCandidate3 i ss
  checkCapListed i ss
  checkRoundsListed i ss
  pure ss

def isVStar (name : String) : Bool :=
  name.startsWith "rec-open/" || name.startsWith "rec-residuals/" || name.startsWith "rec-open-stacked/" ||
    name.startsWith "rec-sparse/" || name.startsWith "rec-opening/" || name.startsWith "rec-cap/"

/-- When `name` is the circuit table's name (`rec-open/l<LANES>-h<H>`) of one of R's levels in `ss`, a sentence saying so: R's
levels are named `rec-open-stacked/…`, never as the table's. -/
def rLevelUnder (name : String) (ss : Array VStar.Stmt) : String :=
  match ss.toList.find? fun s => match s with
    | .recOpenStacked _ L H _ => name == s!"rec-open/l{L}-h{H}"
    | _ => false with
  | some s => s!"{name}: R's level {s.dir} is {s.name}, never under the circuit table's name rec-open/…; "
  | none => ""

/-- Why a unit named `name` is none of V*'s statements `ss` at the inner shape `i`: an R opening part at a shape other than
R's (META's), or with no R named; a RecSparse unit at a shape other than the bucket's, or with none named; one of R's levels
under the circuit table's name (`rec-open/…`, which R's never are: `rec-open-stacked/…`); else that it is none of them. -/
def unlisted (i : Inner) (name : String) (ss : Array VStar.Stmt) : String :=
  let none_ := s!"{name}: not one of V*'s statements at the inner session's shape (m {i.m}, k_log {i.kLog}, claims {i.claims}): {ss.map (·.name)}"
  match RecOpening.params name, RecSparse.params name with
  | some (m, c, _), _ => match i.stacked with
    | some r =>
      if m != r.m || c != r.claims then
        s!"{name}: opens R at (m {m}, claims {c}), not at R's shape (m {r.m}, claims {r.claims}, the inner circuit's META's {STACKED_SHAPE_KEY})"
      else none_
    | none => s!"{name}: one of R's opening parts, and the verifier's inner shape names no R (rec-inner: the inner circuit's META's {STACKED_SHAPE_KEY})"
  | none, some sh => match i.sparse with
    | some sp => if sh == sp then none_ else s!"{name}: a RecSparse unit, and the bucket's is {RecSparse.unitName sp} (rec-inner --sparse)"
    | none => s!"{name}: a RecSparse unit, and the verifier's inner shape names no sparse phase (rec-inner --sparse)"
  | none, none => rLevelUnder name ss ++ none_

/-- A unit named `name` over a population of `n` instances is one of V*'s statements at the inner shape `i`: a part's name
carries `i`'s `m`, `k_log` and claims, exactly one statement has the name, and `n` is its instance count. With `part = some
j`, it is statement `j` of `statements i` (`--rec-part J`: levels first, in level order, then the algebra's parts, in part
order, as `rec_vstage.stage` stages `L<level>/` and `alg/p<j>/`), so a session the verifier holds for statement `j` is
refused another statement's file. -/
def check (i : Inner) (name : String) (n : Nat) (part : Option Nat := none) : V VStar.Stmt := do
  if let some (m, k, c, _) := RecResiduals.params name then
    if m != i.m || k != i.kLog || c != i.claims then
      throw s!"{name}: shape (m {m}, k_log {k}, claims {c}), the inner session's (m {i.m}, k_log {i.kLog}, claims {i.claims})"
  let ss ← statements i
  match ss.toList.filter (·.name == name) with
  | [] => throw (unlisted i name ss)
  | [s] =>
    if n != s.instances then throw (rLevelUnder name ss ++ s!"{name}: instances {n}, statement says {s.instances}")
    if let some j := part then
      match ss[j]? with
      | none => throw s!"{name}: the session is V*'s statement {j} (--rec-part), but V* has {ss.size} statements at the inner session's shape"
      | some t => if t.name != name then throw s!"{name}: the session is V*'s statement {j} (--rec-part), {t.name}"
    pure s
  | ss => throw s!"{name}: {ss.length} of V*'s statements have this name"

/-- **V*'s circuit file of a statement**: the bytes `vstar stage` writes to `<dir>/circuit.txt` (`VStar.Cli.write`,
`VStar/Cli.lean:43-52`): the statement's unit, laid out, emitted with its subcircuit and the statement's bindings, a
part's META saying `holo` over a holo inner session (`VStar.emitCircuitOf`, as `vstar stage --holo` writes it). -/
def circuitFile (s : VStar.Stmt) (b : VStar.Bindings) (holo : Bool := false) : Except String ByteArray := do
  let u ← s.unit
  let l ← u.laid
  let e ← VStar.emitCircuitOf (m := StateM String) (holo && s matches .inner ..) (fun x => modify (· ++ x)) u l s.sub b
  pure (Id.run (e.run "")).2.toUTF8

/-- The first position at which two byte strings differ, if they do: the byte and its line (1-based). -/
def firstDiff? (a b : ByteArray) : Option (Nat × Nat) := Id.run do
  let mut line := 1
  for k in [0:min a.size b.size] do
    if a[k]! != b[k]! then return some (k, line)
    if a[k]! == 10 then line := line + 1
  if a.size == b.size then none else some (min a.size b.size, line)

/-- The circuit file is V*'s for `s` under the bindings `b`, the verifier's own, over an inner session holo or not
(`circuitFile`). -/
def checkFile (s : VStar.Stmt) (b : VStar.Bindings) (circuitB : ByteArray) (holo : Bool := false) : V Unit := do
  let f ← match circuitFile s b holo with
    | .ok f => pure f
    | .error e => throw s!"{s.name}: V*'s circuit file: {e}"
  if let some (k, line) := firstDiff? circuitB f then
    throw s!"{s.name}: the circuit file ({circuitB.size} bytes) is not V*'s ({f.size} bytes): they differ at byte {k}, line {line}"

/-- V*'s row ports of the unit `u` under the subcircuit `sub` (`VStar.metaHead`'s `ports`), as (name, words, bits): an
input row of its port's words, an output row of its port's rounded up to whole 64-word blocks, each 16 bits a word. -/
def vstarRows (sub : VStar.Sub) (u : VStar.UnitC) : List (String × Nat × Nat) :=
  (sub.inputs.zip u.inPorts).map (fun (nm, w) => (nm, w, 16 * w)) ++
    (sub.outputs.zip u.outPorts).map fun (nm, w) => (nm, (w + 63) / 64 * 64, 16 * ((w + 63) / 64 * 64))

/-- **The circuit parsed from V*'s file carries V*'s META** (`VStar.metaHead`): one unit per VU, no leaf cuts, the
unit's input leaves its input words in order, V*'s row ports (`vstarRows`), each a `sha512/row/v1` row as V*'s META writes
them (`row_schema`, no `row_schemas`), and the unit's output leaves its output words in order (`UnitC.leaves`' `out`),
with no template (V*'s circuits are flat). `checkFile` holds the bytes to V*'s;
this holds what the verifier parsed from them, which no proof reads through `Lean.Json.parse`, to what V* wrote. -/
def checkParsed (s : VStar.Stmt) (c : Circuit) : V Unit := do
  if c.tmpl.isSome then
    throw s!"{s.name}: the parsed circuit is templated, and V*'s circuits are flat"
  let u ← match s.unit with
    | .ok u => pure u
    | .error e => throw s!"{s.name}: V*'s unit: {e}"
  if c.upv != 1 || !c.leafCuts.isEmpty then
    throw s!"{s.name}: the parsed circuit has {c.upv} units per VU and {c.leafCuts.size} leaf cuts, not V*'s 1, 0"
  if c.leavesIn.getD 0 #[] != (Array.range u.inWords).map Int.ofNat then
    throw s!"{s.name}: the parsed circuit's input leaves are not V*'s, its {u.inWords} input words in order"
  let ports := c.ports.toList.map fun p => (p.name, p.words, p.bits)
  if ports != vstarRows s.sub u || c.ports.toList.any (·.v2) then
    throw s!"{s.name}: the parsed circuit's row ports {ports} are not V*'s {vstarRows s.sub u}, each a sha512/row/v1 row"
  if c.leavesOut.getD 0 #[] != Array.range u.outWords then
    throw s!"{s.name}: the parsed circuit's output leaves are not V*'s, its {u.outWords} output words in order"

/-- The whole S's row ports' names (`VStar.Algebra.spec`'s): round `r`'s `m<r>`, then `acc0` and `acc1`. -/
def portNames (sh : Shape) : Array String :=
  (Array.range (RecAlgebra.rounds sh).size).map (s!"m{·}") ++ #["acc0", "acc1"]

/-- Part `j`'s row ports, as indices of the whole S's (`RecResiduals.groups`' `P`). -/
def partPorts (sh : Shape) (j : Nat) : Except String (Array Nat) := do
  let gs ← RecResiduals.groups (← RecAlgebra.structureOf sh) (RecResiduals.portSizes sh)
  let some (_, P, _) := gs[j]? | throw s!"part {j} of {gs.size}"
  pure P

/-- **V*'s part is the verifier's**: at a part's statement, V*'s row ports' elements and v elements are those of the
verifier's part (`RecResiduals.part`) at the shape, holo flag and part the unit's name and META give, as
`RecResiduals.check` derives it; the words `checkParsed` compares round them up (`portWordsOf`), so the counts are
compared themselves. Its row ports' names are the whole S's (`portNames`) at the part's ports (`partPorts`). Any other
statement passes. -/
def checkPart (s : VStar.Stmt) (c : Circuit) : V Unit :=
  match s with
  | .inner _ _ pt =>
    match RecResiduals.params c.unit.name with
    | none => throw s!"{s.name}: unit {c.unit.name} is not rec-residuals/m<m>-k<k_log>-c<claims>-p<j>"
    | some (m, k, cl, j) =>
      match RecResiduals.shape? m k cl with
      | none => throw s!"{s.name}: no rec-residuals shape at m {m}, k_log {k}, claims {cl}"
      | some sh =>
        match RecResiduals.holoOf c with
        | .error e => throw e
        | .ok holo =>
          match RecResiduals.part { sh with holo } j with
          | .error e => throw s!"{s.name}: {e}"
          | .ok (_, ns, nv) =>
            if pt.S.ports.map (·.2) != ns || pt.S.gf.nv != nv then
              throw (s!"{s.name}: V*'s part has row ports of {pt.S.ports.map (·.2)} elements and {pt.S.gf.nv} v " ++
                s!"elements, the verifier's {ns} and {nv}")
            else
              match partPorts { sh with holo } j with
              | .error e => throw s!"{s.name}: {e}"
              | .ok P =>
                let names := P.toList.map ((portNames { sh with holo })[·]!)
                if pt.S.ports.map (·.1) != names then
                  throw s!"{s.name}: V*'s part's row ports are named {pt.S.ports.map (·.1)}, the verifier's {names}"
                else pure ()
  | _ => pure ()

/-- A statement's population: the registered population's instances for a drawn session, else its own. -/
def population (st : Stmt) : Nat := (st.pub.population.map (·.1)).getD st.n

/-- META's bindings (`VStar.Bindings`, `VStar/Compose.lean:52-55`, written at `Compose.lean:228-229`): the
`verity/partition/v1` digest, `program_sha512` and `program_digests`. -/
def bindingsOf (c : Circuit) : V VStar.Bindings := do
  let name := c.unit.name
  let part := (c.metaJ.getObjVal? "partition").toOption.getD Json.null
  let (some "verity/partition/v1", some digest) := ((part.getObjValAs? String "rule").toOption,
      (part.getObjValAs? String "digest").toOption)
    | throw s!"{name}: META partition is not a verity/partition/v1 digest"
  let some program := (c.metaJ.getObjValAs? String "program_sha512").toOption
    | throw s!"{name}: META names no program_sha512"
  let some pd := (c.metaJ.getObjVal? "program_digests").toOption | throw s!"{name}: META names no program_digests"
  pure ⟨digest, program, pd⟩

/-- The verifier's own bindings of a statement of V*'s (`--rec-bindings F`): `vstar`'s `BINDINGS` object, `{"partition",
"program", "program_digests"}` (`VStar.Cli.bindingsOf`), which the auditor computes from the verifier's partition object and
program (`verity_flock.rec_bindings`), never from the prover's circuit file. -/
def ownBindingsOfJson (j : Json) : Except String VStar.Bindings :=
  (VStar.Cli.bindingsOf j).mapError (s!"rec-bindings: " ++ ·)

/-- The verifier's own bindings `f` are its own partition object's and program's: `f`'s partition digest is the object's
(`HmRow.partitionDigest`), `f`'s program and the object's `program` are the program's digest (the SHA-512 of its canonical
descriptor without `annotations`, `TemplateQuery.Desc.parse`), and `f`'s `program_digests` names one Definition, the
template the object's query names. Its digest is not recomputed: that is core's codec's digest of the one-Definition
program, which this verifier does not encode; the auditor's `verity_flock.rec_bindings` computes it from the same files. -/
def checkOwnBindings (name : String) (f : VStar.Bindings) (obj : Json) (prog : ByteArray) : V Unit := do
  let own := hexOf (← HmRow.partitionDigest obj)
  if f.partition != own then throw s!"{name}: the verifier's bindings' partition digest {f.partition}, its partition object's {own}"
  let ps := hexOf (← TemplateQuery.Desc.parse prog).sha512
  if f.program != ps then throw s!"{name}: the verifier's bindings' program {f.program}, its program's {ps}"
  let op := (obj.getObjValAs? String "program").toOption.getD ""
  if op != ps then throw s!"{name}: the partition object's program {op}, the verifier's program's {ps}"
  let q := (obj.getObjVal? "query").toOption.getD Json.null
  let t := ((q.getObjVal? "params").toOption.getD Json.null |>.getObjValAs? String "template").toOption.getD ""
  match f.programDigests with
  | .obj kvs =>
    match kvs.toArray.map (·.1) with
    | #[k] => if k != t then throw s!"{name}: the verifier's bindings' program_digests names {k}, the partition's template {t}"
    | ks => throw s!"{name}: the verifier's bindings' program_digests names {ks.size} Definitions, not the partition's template {t} alone"
  | _ => throw s!"{name}: the verifier's bindings' program_digests is not an object"

/-- **META binds the verifier's own bindings**: META's partition digest and `program_sha512` are `f`'s, and META's
`program_digests` is `f`'s as a value, compared as RFC 8785 canonical JSON (`canon`), byte for byte. -/
def checkMeta (c : Circuit) (f : VStar.Bindings) : V Unit := do
  let name := c.unit.name
  let b ← bindingsOf c
  if b.partition != f.partition then throw s!"{name}: META partition digest {b.partition}, the verifier's bindings' {f.partition}"
  if b.program != f.program then throw s!"{name}: META program_sha512 {b.program}, the verifier's bindings' {f.program}"
  let (mc, fc) := (← canon b.programDigests, ← canon f.programDigests)
  if mc != fc then throw s!"{name}: META program_digests {mc}, the verifier's bindings' {fc}"

/-- Whether V*'s file of statement `s` over an inner session of holo flag `holo` carries META `holo` (`circuitFile`):
at a part's statement over a holo session. -/
def partHolo (s : VStar.Stmt) (holo : Bool) : Bool := holo && s matches .inner ..

/-- **The parsed circuit's META `holo` is V*'s file's**: `RecResiduals.holoOf c` is `true` exactly at a part's statement
over a holo inner session, where `circuitFile` writes META `holo` (`VStar.emitCircuitHolo`), and `false` elsewhere, where
it writes none. `checkFile` holds the bytes to V*'s; this holds what `checkPart` reads of the parsed circuit, which no
proof reads through `Lean.Json.parse`, to what V* wrote. -/
def checkHolo (s : VStar.Stmt) (c : Circuit) (holo : Bool) : V Unit := do
  if (← RecResiduals.holoOf c) != partHolo s holo then
    throw s!"{s.name}: the circuit's META holo is not V*'s file's ({partHolo s holo}) at the inner session's holo flag {holo}"

/-- `checkMeta` of the circuit `c`, parsed from the circuit file `circuitB`, against the verifier's own bindings `f`, then
`check` on its unit over `n` instances at the inner shape `i` as V*'s statement `part`, `checkFile` on `circuitB` under
`f` at `i`'s holo flag, `checkParsed` and `checkPart` on `c`, and `checkHolo` on `c` at `i`'s holo flag. -/
def checkCircuit (i : Inner) (part : Nat) (c : Circuit) (n : Nat) (circuitB : ByteArray) (f : VStar.Bindings) :
    V VStar.Stmt := do
  checkMeta c f
  let s ← check i c.unit.name n (some part)
  checkFile s f circuitB i.holo
  checkParsed s c
  checkPart s c
  checkHolo s c i.holo
  pure s

/-- A statement's message ports that `ownPorts` leaves out: RecSparse's messages but `h` and `claim`, each `<p>` under
`rec-sparse-<p>` at both reps' rows (`rec_vstage.sparse_messages`, `rec_sparse.reads`); no other statement has any. -/
def sparseMsgs : VStar.Stmt → List (String × String × Array (Option Nat))
  | .sparse sh _ => ((RecSparse.messagePorts sh).drop 1).dropLast.map fun (nm, _) =>
    (nm, s!"rec-sparse-{nm}", #[some 0, some 1])
  | _ => []

def showPort (q : String × String × Array (Option Nat)) : String :=
  s!"{q.1} under {q.2.1} reading {q.2.2.toList.map fun r => (r.map toString).getD "null"}"

/-- **The verifier's reads file has the statement's message-round ports**: `msgPorts` of the `--registered` file and
`ownPorts` of the statement hold the same ports, compared by membership (`Registered.ofJson` lists a file's ports in its
object's key order), and the file has each of the statement's `sparseMsgs`. -/
def checkOwn (s : VStar.Stmt) (registered : Array Registered.Port) : V Unit := do
  let file := msgPorts registered
  let own := ownPorts s
  if let some q := file.find? (!own.contains ·) then
    let theirs := match own.find? (·.1 == q.1) with
      | some o => s!"the statement's is {showPort o}"
      | none => s!"the statement has no port {q.1} ({own.length} message-round ports)"
    throw s!"{s.name}: the registered reads' port {showPort q} is not the statement's: {theirs}"
  if let some q := own.find? (!file.contains ·) then
    throw s!"{s.name}: the statement's message-round port {showPort q} is not in the registered reads ({file.length} message-round ports)"
  let all := registered.toList.map fun p => (p.name, p.value, p.reads)
  if let some q := (sparseMsgs s).find? (!all.contains ·) then
    throw s!"{s.name}: the statement's message port {showPort q} is not in the registered reads"

/-- A statement's row ports: a part's ports (`VStar/Algebra.lean:385-387`: its `m<k>`, `acc0`, `acc1`), not its `v` and
`out`, which are its own (`rec_residuals.write`, rec_residuals.py:451-461); a level has none, and neither has candidate 3's
statements, which share no row with another session of the family. -/
def rowPorts : VStar.Stmt → List String
  | .inner _ _ p => p.S.ports.map (·.1)
  | .recOpen .. | .recOpenStacked .. | .sparse .. | .opening .. | .cap .. => []

/-- The first port of a circuit named as a message round's (`m<k>`) that is none of a statement's message-round ports
(`ownPorts`). -/
def msgUnnamed (s : VStar.Stmt) (c : Circuit) : Option String :=
  (c.ports.toList.map (·.name)).find? fun nm => (mIndex nm).isSome && !((ownPorts s).any (·.1 == nm))

/-- The message rounds a part may name: its rep's (`commitRounds (repRounds …)`, `holoRepRounds` over a holo inner
session, whose h round is one more), Ligerito's included, whose `k`-th is `rec-c<k>`'s and V*'s port `m<k>`
(`VStar.Algebra.rounds`). At a shape with no fast100 schedule, none: every `m<k>` is refused. -/
def innerMsgRounds (s : VStar.Stmt) (holo : Bool := false) : Option Nat :=
  match s with
  | .inner _ name _ => do
    let (m, kLog, claims, _) ← RecResiduals.params name
    pure ((((if holo then holoRepRounds else repRounds) m kLog claims).map fun rs => (commitRounds rs).length).getD 0)
  | _ => none

/-- The first circuit port naming a message round past the part's rep's message rounds. -/
def msgRoundOutOfRange (rounds : Nat) (c : Circuit) : Option String :=
  (c.ports.toList.map (·.name)).find? fun nm =>
    match mIndex nm with
    | some k => decide (rounds ≤ k)
    | none => false

/-- **A statement's circuit names each port once, and a part's circuit, and a part of R's opening's, names only its
statement's message-round ports**: the verifier finds a registered port by its name (`Registered.checkPort`, the first
port of the name), so each name is one port's; and each port `m<k>` of a part's circuit is one of the statement's
(`ownPorts`: a part's `m<k>` under `rec-c<k>`, R's opening's under `rec-opening-c<k>`), so the verifier's reads file
reads it under the statement's value (`checkOwn`). The circuit file is V*'s for the statement (`checkFile`), whose ports
are the statement's, each name once (`HmRow.checkRows` refuses a name twice), so an honest session passes. A part also
refuses an `m<k>` at or past its rep's message rounds (`innerMsgRounds`, at the inner session's `holo`): its inner
transcript has no round `rec-c<k>`. -/
def checkMsgNames (s : VStar.Stmt) (c : Circuit) (holo : Bool := false) : V Unit :=
  if !decide (c.ports.toList.map (·.name)).Nodup then
    throw s!"{s.name}: the circuit names a port twice"
  else do
    match s with
    | .inner .. | .opening .. =>
      match msgUnnamed s c with
      | some nm => throw s!"{s.name}: the circuit's port {nm} is named as a message round's, and the statement has no such port"
      | none => pure ()
    | _ => pure ()
    if let some rounds := innerMsgRounds s holo then
      if let some nm := msgRoundOutOfRange rounds c then
        throw s!"{s.name}: the circuit's port {nm} names message round {(mIndex nm).getD 0}, not below its rep's {rounds} message rounds"

def showReads (r : Array (Option Nat)) : String := toString (r.toList.map fun x => (x.map toString).getD "null")

/-- The first field in which two registered ports differ, with both values: the registration (value, root, leaf count,
row width, domain, leaf schema) and, with `reads`, the read positions; none if they agree. -/
def portDiff? (a b : Registered.Port) (reads : Bool := true) : Option String :=
  let w := fun (x : Option Nat) => (x.map toString).getD "none"
  if a.value != b.value then some s!"value {a.value}, against {b.value}"
  else if a.root != b.root then some s!"root {hexOf a.root}, against {hexOf b.root}"
  else if a.leaves != b.leaves then some s!"leaves {a.leaves}, against {b.leaves}"
  else if a.words != b.words then some s!"words {w a.words}, against {w b.words}"
  else if a.wordBits != b.wordBits then some s!"word_bits {a.wordBits}, against {b.wordBits}"
  else if a.domain != b.domain then some s!"domain {hexOf a.domain}, against {hexOf b.domain}"
  else if a.schema != b.schema then some s!"row_schema {a.schema}, against {b.schema}"
  else if reads && a.reads != b.reads then some s!"reads {showReads a.reads}, against {showReads b.reads}"
  else none

/-- **The verifier's reads file names a part's row ports, each the family's one entry of that name** (`OwnNamesRows`,
`PartPortsOne`): every row port of the statement (`rowPorts`) is a port of `--registered`, and agrees in every field
(`portDiff?`: value, root, leaf count, row width, domain and reads) with the port of that name in the verifier's record of
V*'s shared rows (`--rec-rows`), one file for every session of the family. -/
def checkRows (s : VStar.Stmt) (registered : Array Registered.Port) (rows : Option (Array Registered.Port)) : V Unit := do
  let names := rowPorts s
  if let some nm := names.find? fun nm => !registered.any (·.name == nm) then
    throw s!"{s.name}: its row port {nm} is not a port of the registered reads ({registered.toList.map (·.name)})"
  if names.isEmpty then return
  let some rows := rows
    | throw s!"{s.name}: one of V*'s parts; pass the verifier's record of V*'s shared rows (--rec-rows F), which each of its row ports must be"
  for nm in names do
    let some reg := registered.find? (·.name == nm) | throw s!"{s.name}: its row port {nm} is not a port of the registered reads"
    let some want := rows.find? (·.name == nm)
      | throw s!"{s.name}: its row port {nm} is not a port of the verifier's shared rows (--rec-rows: {rows.toList.map (·.name)})"
    if let some d := portDiff? reg want then
      throw s!"{s.name}: its row port {nm} is not the verifier's shared row {nm} (--rec-rows): {d}"

def ACC_VALUE : String := "rec-acc"

/-- The first name two of the verifier's registered ports share. -/
def dupName? (registered : Array Registered.Port) : Option String :=
  let names := registered.toList.map (·.name)
  names.find? fun nm => names.count nm != 1

/-- **The verifier's reads file names each port once** (`dupName?`): `checkAcc` and `checkAccReads` read a name's first
port (`find?`), so a second port of that name would be one neither checks. A file `Registered.ofJson` parses names each
once (a JSON object's keys); the refusal holds whatever list of ports the verifier is given. -/
def checkNames (s : VStar.Stmt) (registered : Array Registered.Port) : V Unit :=
  match dupName? registered with
  | some nm => throw s!"{s.name}: the registered reads name two ports {nm}, and each check reads a name's first"
  | none => pure ()

/-- A statement's ports that read `rec-acc`: a level's `acc` and `acc_out` (`rec_outer.stage`, rec_outer.py:236), R's
levels' too (R's openings follow the reps' in the chain, `rec_vstage.stage_stacked`), a part's `acc0` and `acc1` among its
ports (rec_vstage.py:357), a part of R's opening's too (`rec_residuals.stage_opening`); RecSparse reads none. -/
def accPorts : VStar.Stmt → List String
  | .inner _ _ p => (p.S.ports.map (·.1)).filter fun nm => nm == "acc0" || nm == "acc1"
  | .recOpen .. | .recOpenStacked .. => ["acc", "acc_out"]
  | .sparse .. | .cap .. => []
  | .opening _ _ _ _ ports _ => (ports.map (·.1)).filter fun nm => nm == "acc0" || nm == "acc1"

/-- **A session's `rec-acc` reads are one registration, the verifier's** (the `site` of `Family/AccRows.lean`: value,
root, leaf count, row width and domain; not name or reads): `--registered` names each port once (`checkNames`), each of
the statement's `accPorts` is a port of it reading `rec-acc`, every port there reading `rec-acc` has the registration of
the first, and that is the registration of `acc0` in the verifier's shared rows (`--rec-rows`), one file for every session
of the family, whose `rec-acc` ports all have it. -/
def checkAcc (s : VStar.Stmt) (registered : Array Registered.Port) (rows : Option (Array Registered.Port)) : V Unit := do
  checkNames s registered
  for nm in accPorts s do
    match registered.find? (·.name == nm) with
    | some p => if p.value != ACC_VALUE then throw s!"{s.name}: its port {nm} reads {p.value}, not {ACC_VALUE}"
    | none => throw s!"{s.name}: its port {nm} is not a port of the registered reads ({registered.toList.map (·.name)})"
  let accs := registered.filter (·.value == ACC_VALUE)
  let some a := accs[0]? | return
  for b in accs do
    if let some d := portDiff? a b false then
      throw s!"{s.name}: its {ACC_VALUE} ports {a.name} and {b.name} are two registrations: {d}"
  let some rows := rows
    | throw s!"{s.name}: it reads {ACC_VALUE}; pass the verifier's record of V*'s shared rows (--rec-rows F), whose {ACC_VALUE} registration it must be"
  let some w := rows.find? (·.name == "acc0")
    | throw s!"{s.name}: the verifier's shared rows (--rec-rows) have no acc0, whose {ACC_VALUE} registration it must be"
  if w.value != ACC_VALUE then throw s!"the verifier's shared rows (--rec-rows): acc0 reads {w.value}, not {ACC_VALUE}"
  for b in rows.filter (·.value == ACC_VALUE) do
    if let some d := portDiff? w b false then
      throw s!"the verifier's shared rows (--rec-rows): its {ACC_VALUE} ports acc0 and {b.name} are two registrations: {d}"
  if let some d := portDiff? a w false then
    throw s!"{s.name}: its {ACC_VALUE} port {a.name} is not the verifier's {ACC_VALUE} registration (--rec-rows acc0): {d}"

/-- A rep's openings at schedule `sch`'s levels below `l` (every level's at `l = sch.levels.size`). -/
def opensBelow (sch : Schedule) (l : Nat) : Nat := ((List.range l).map fun k => (sch.levels[k]!).q).sum

/-- **The rows of `rec-acc` a level's units read as `acc`** (`rec_outer.chain_order`: every opening's (rep, level, j),
rep-major, then by level and query; `acc_out` reads the next row): at the table's level `l` (`fast100` at the inner `m`,
`n` openings a rep), unit `r · q + j`, rep `r`'s opening `j`, reads row `r · n + opensBelow l + j`; at R's level `l`
(`fast100` at R's `m`; R's openings follow both reps' in the chain, `rec_vstage.stage_stacked`), unit `j` reads row
`2 · n + opensBelow' l + j`. None at a statement that is no level. -/
def accReads (i : Inner) : VStar.Stmt → V (Option (Array (Option Nat)))
  | .recOpen l .. => do
    let some sch := fast100 i.m | throw s!"the inner session's m {i.m} has no fast100 schedule"
    let q := (sch.levels[l]!).q
    let n := opensBelow sch sch.levels.size
    pure (some ((Array.range (2 * q)).map fun u => some (u / q * n + opensBelow sch l + u % q)))
  | .recOpenStacked l .. => do
    let some sch := fast100 i.m | throw s!"the inner session's m {i.m} has no fast100 schedule"
    let some r := i.stacked
      | throw s!"R's level {l}: the inner shape names no R (rec-inner: the inner circuit's META's {STACKED_SHAPE_KEY})"
    let some sr := fast100 r.m | throw s!"R's m {r.m} has no fast100 schedule"
    let n := opensBelow sch sch.levels.size
    pure (some ((Array.range (sr.levels[l]!).q).map fun j => some (2 * n + opensBelow sr l + j)))
  | _ => pure none

private def showRead : Option (Option Nat) → String
  | some (some p) => toString p
  | some none => "null"
  | none => "nothing"

/-- **The rows of `rec-acc` a part reads as `acc0` and `acc1`**: each rep's running sums before its first opening and
after its last (`rec_outer.Chain.rep_ends`, `accReads`' layout). A part of the table's algebra (`.inner`) holds both
reps, rep `r` at instance `r`: `acc0` reads rows `0` and `n`, `acc1` rows `n` and `2 · n` (`n` openings a rep,
`rec_vstage.py`'s `reads["acc0"], reads["acc1"]`). A part of R's opening (`.opening`) holds R's one run, after both
reps': `acc0` reads row `2 · n`, `acc1` row `2 · n + n'` (`n'` R's openings, `rec_residuals.stage_opening`). None at a
statement that is neither. -/
def endReads (i : Inner) : VStar.Stmt → V (Option (Array (Option Nat) × Array (Option Nat)))
  | .inner .. => do
    let some sch := fast100 i.m | throw s!"the inner session's m {i.m} has no fast100 schedule"
    let n := opensBelow sch sch.levels.size
    pure (some (#[some 0, some n], #[some n, some (2 * n)]))
  | .opening .. => do
    let some sch := fast100 i.m | throw s!"the inner session's m {i.m} has no fast100 schedule"
    let some r := i.stacked
      | throw s!"R's opening: the inner shape names no R (rec-inner: the inner circuit's META's {STACKED_SHAPE_KEY})"
    let some sr := fast100 r.m | throw s!"R's m {r.m} has no fast100 schedule"
    let n := opensBelow sch sch.levels.size
    pure (some (#[some (2 * n)], #[some (2 * n + opensBelow sr sr.levels.size)]))
  | _ => pure none

/-- The rows port `nm` reads at its runs' ends (`endReads`' pair): `acc0`'s, else `acc1`'s. -/
def endWant (w : Array (Option Nat) × Array (Option Nat)) (nm : String) : Array (Option Nat) :=
  if nm == "acc0" then w.1 else w.2

/-- **A part's `acc0` and `acc1` read `rec-acc` at its runs' ends** (`endReads`), which neither `checkAcc` (one
registration, reads aside) nor `checkRows` (the verifier's shared rows, which the prover stages) fixes. -/
def checkEndReads (i : Inner) (s : VStar.Stmt) (registered : Array Registered.Port) : V Unit := do
  if (accPorts s).isEmpty then return
  let some w ← endReads i s | return
  for nm in accPorts s do
    let some p := registered.find? (·.name == nm)
      | throw s!"{s.name}: its port {nm} is not a port of the registered reads ({registered.toList.map (·.name)})"
    if p.reads != endWant w nm then
      throw s!"{s.name}: its port {nm} reads {ACC_VALUE} at {showReads p.reads}, not at its runs' ends {showReads (endWant w nm)}"

/-- **A level's `acc` and `acc_out` read `rec-acc` at its openings' rows of the chain** (`accReads`; `OpenKeysS`'
positions, which `checkAcc`'s one registration leaves open): unit `u`'s `acc` reads its opening's row, its `acc_out` the
next. At a statement that is no level, its `acc0` and `acc1` read its runs' ends (`checkEndReads`). -/
def checkAccReads (i : Inner) (s : VStar.Stmt) (registered : Array Registered.Port) : V Unit := do
  let some want ← accReads i s | checkEndReads i s registered
  for (nm, d) in [("acc", 0), ("acc_out", 1)] do
    let some p := registered.find? (·.name == nm)
      | throw s!"{s.name}: its port {nm} is not a port of the registered reads ({registered.toList.map (·.name)})"
    let w := want.map (·.map (· + d))
    if p.reads != w then
      let u := ((List.range (max p.reads.size w.size)).find? fun u => p.reads[u]? != w[u]?).getD 0
      throw (s!"{s.name}: its port {nm} reads {ACC_VALUE} at {showRead p.reads[u]?} in unit {u}, not at " ++
        s!"{showRead w[u]?} (the chain runs rep-major, then by level and query)")

def TRANSCRIPT_FORMAT : String := "rec/inner-transcript/v1"

/-- A round's coins, as hex strings. -/
def coinsOf (r : Json) : Array String := (r.getObjValAs? (Array String) "coins").toOption.getD #[]

/-- Candidate 3's streams beside the circuit table's reps (`rec_live.SPARSE`, `rec_live.STACKED`): each rep's sparse phase,
`sparse/rep<x>`, and the stacked table R's one stream, `stacked/rep0`. -/
def SPARSE : String := "sparse"
def STACKED : String := "stacked"

/-- One stream of the inner session as the coin server recorded it: its message rounds' `c_k` (`b ‖ c`), each with its
round number, in round order; and its table's committed tops by Ligerito level (`rec_vstar.tops`), each node's `b ‖ c`:
level 0 the table's `root_B` from `Commit` (the record's `link.roots`; R, which has none there, its binding round's salted
top), then each later salted top in the order the stream's rounds absorbed them. A sparse stream belongs to no table and has
none. `coins`: each round's coins, in round order, as the coin server drew them (check 5's coins of the sparse phases and
of R, which the prover's record does not hold). -/
structure InnerStream where
  name : String
  commits : Array (Nat × ByteArray)
  tops : Array (Array ByteArray)
  coins : Array (Array F128) := #[]

/-- What the verifier holds of the inner session's transcript: the circuit table's reps' `c_k` (`reps`, rep `x` at `x`), and
with candidate 3's streams every stream of the session (`streams`; empty for a session of one table's two reps) and, with
R's stream, the shape R's stream was held to (`stacked`, `StackedShape`, the inner circuit's META's). -/
structure InnerCommits where
  reps : Array (Array (Nat × ByteArray))
  streams : Array InnerStream := #[]
  stacked : Option StackedShape := none

def InnerCommits.stream? (ic : InnerCommits) (name : String) : Option InnerStream := ic.streams.find? (·.name == name)

/-- Nodes `b ‖ c` in hex, each refused unless 128 bytes. -/
def hexList (ctx : String) (xs : Array String) : V (Array ByteArray) :=
  xs.mapM fun x => do
    let b ← (unhex x).mapError (s!"rec inner transcript: {ctx}: " ++ ·)
    if b.size != 128 then throw s!"rec inner transcript: {ctx}: a b ‖ c of {b.size} bytes, not 128"
    pure b

/-- **A rep's coins, by the transcript's rounds**: what the inner session's record (the prover's, which check 5 reads)
holds of rep stream `s` at each of the transcript's rounds `0 … n-1` of it (`rs`, in round order): that round's coins, or
`none` at the round the record has no counterpart of. A holo session's firewall splits z_partial's round in two (`rec_session`:
a round of no coins, then h's round), and the transcript records both where the prover's record has the one round. So with
one round of no coins at `j` in the transcript's stream, the record has `n - 1` rounds, its round `k` the transcript's `k`
below `j` and `k + 1` from `j` on (h's round taking z_partial's coins); with none, the two have the same rounds. -/
def repCoins (s : String) (rs : Array Json) (srec : Json) : V (Array (Option (Array String))) := do
  let n := rs.size
  if rs.toList.map (fun r => (r.getObjValAs? Nat "k").toOption) != (List.range n).map some then
    throw s!"rec inner transcript: {s}'s rounds are not its rounds 0 to {n - 1}"
  let rr := (srec.getObjValAs? (Array Json) "rounds").toOption.getD #[]
  match (rs.zipIdx.filter fun (r, _) => (coinsOf r).isEmpty).map (·.2) with
  | #[] =>
    if rr.size != n then throw s!"rec inner transcript: {s} records {n} rounds and the inner session's record {rr.size}"
    return rr.map (some ∘ coinsOf)
  | #[j] =>
    if j + 1 == n then throw s!"rec inner transcript: {s}'s round {j} of no coins is its last; h's round follows z_partial's"
    if rr.size + 1 != n then
      throw s!"rec inner transcript: {s} splits z_partial's round {j} in two and records {n} rounds; the inner session's record has {rr.size}, not {n - 1} (it is the prover's record, check 5's)"
    return (Array.range n).map fun k => if k == j then none else rr[if k < j then k else k - 1]?.map coinsOf
  | js => throw s!"rec inner transcript: {s} has rounds of no coins at {js}, and only z_partial's round is split"

def REGISTRATION_FORMAT : String := "verity/registration/v2"
/-- A registered top's row: a 64-byte node as 32 u16 words, zero-padded to one SHA-512 block (`rec_live.ROW_WORDS`). -/
def TOP_ROW_WORDS : Nat := 64

def refused (what : String) : String := s!"FIREWALL-REFUSED: {what}"

/-- One-stage `registration.digest` (`SampledProofs.Registration.digest`): `SHA-512(format ‖ 0 ‖ canon(record))`, the
record's own format (v1 or v2) as the tag; hex, as META's `registration` names it. -/
def registrationDigest (rec : Json) : V String := do
  let tag := if (rec.getObjValAs? String "format").toOption == some REGISTRATION_FORMAT then REGISTRATION_FORMAT
    else "verity/registration/v1"
  return hexOf (Sha512.hash (tag.toUTF8.push 0 ++ (← canon rec).toUTF8))

/-- The registered entry of R whose rows' `b ‖ c` are `bcs` (`rec_live.top_entry`, `rec_residuals.commits_entry`): one row
of `TOP_ROW_WORDS` 16-bit words a node, its domain `Registered.domain` under the registration's `program`, and the
frame-v3-sha512 root over the rows' `hm96-sha512/row/v1` leaves (`Registered.rowLeaf`, as `Registered.opens` reads a row). -/
def topEntry (program : String) (bcs : Array ByteArray) : V Json := do
  let dom ← Registered.domain program STACKED TOP_ROW_WORDS bcs.size Registered.WORD_BITS
  let root := FV3.root dom (bcs.mapIdx fun i bc => Registered.rowLeaf dom i (Hm96.Default512.treeLeaf bc)) Sha512.hash
  return Json.mkObj [("value", Json.str STACKED), ("root", Json.str (hexOf root)), ("leaves", toJson bcs.size),
    ("words", toJson TOP_ROW_WORDS), ("domain", Json.str (hexOf dom))]

/-- The five fields that bind a `stacked` registration entry to the binding round's concrete top. This is deliberately
checked structurally beside the canonical-JSON comparison: proofs use these field equalities, while keeping the old
comparison makes the verifier accept no new entry. -/
def topEntryFields (program : String) (bcs : Array ByteArray) (e : Json) : V Bool := do
  let dom ← Registered.domain program STACKED TOP_ROW_WORDS bcs.size Registered.WORD_BITS
  let leaves := bcs.mapIdx fun i bc => Registered.rowLeaf dom i (Hm96.Default512.treeLeaf bc)
  let root := FV3.root dom leaves Sha512.hash
  return (e.getObjValAs? String "value").toOption == some STACKED &&
    (e.getObjValAs? String "root").toOption == some (hexOf root) &&
    (e.getObjValAs? Nat "leaves").toOption == some bcs.size &&
    (e.getObjValAs? Nat "words").toOption == some TOP_ROW_WORDS &&
    (e.getObjValAs? String "domain").toOption == some (hexOf dom)

/-- **R's registration**: the circuit registration (`--rec-inner-registration`, one-stage `registration.record`'s
`verity/registration/v2` record, written at staging, `inner_registered.REGISTRATION`) whose digest the inner circuit's META
binds (`bound`, META's `registration`), and its entry named `stacked` (R's registered top, `rec_live.register_top`) with the
record's `program`; none when neither the record nor META names one. A record META does not bind, or a META binding a record
the verifier does not hold, is refused; so is a v1 record, or one naming `stacked` twice. -/
def registration (reg : Option Json) (bound : Option String) : V (Option (String × Option Json)) := do
  match reg, bound with
  | none, none => return none
  | none, some d =>
    throw (refused s!"registration: META binds the circuit registration {d}, and the verifier holds none (--rec-inner-registration)")
  | some _, none => throw (refused "registration: the record is not the circuit registration META binds (META binds none)")
  | some j, some d =>
    if (← registrationDigest j) != d then throw (refused s!"registration: the record is not the circuit registration META binds ({d})")
    let none_ := refused s!"registration: {STACKED} has no root at Commit and no registered top"
    let (.ok program, .ok es) := (j.getObjValAs? String "program", j.getObjValAs? (Array Json) "registered") | throw none_
    if (j.getObjValAs? String "format").toOption != some REGISTRATION_FORMAT then throw none_
    match es.filter fun e => (e.getObjValAs? String "value").toOption == some STACKED with
    | #[] => return some (program, none)
    | #[e] => return some (program, some e)
    | _ => throw none_

/-- META's R shape (`STACKED_SHAPE_KEY`) of the inner circuit, the shape staging committed R at; none when META names none. -/
def stackedShapeOf (hdr : Json) : V (Option StackedShape) := do
  let .ok s := hdr.getObjVal? STACKED_SHAPE_KEY | return none
  let (.ok m, .ok claims) := (s.getObjValAs? Nat "m", s.getObjValAs? Nat "claims")
    | throw s!"rec inner circuit: META's {STACKED_SHAPE_KEY} is not R's shape (m, claims)"
  return some { m, claims }

/-- `rec_live.stacked_rounds` at `claims` claims: R's one stream, the opening-only prover's rep at `Shape(m, 6, claims)`. Its
binding round commits R's level-0 top (R has no root at `Commit`) and absorbs the first claim's s_hat_v; then the other
claims' ring switches, the gammas and Ligerito at `fast100 m` (`repRounds m 6 claims` past its first ring switch). -/
def stackedRounds (sh : StackedShape) : Option (Array SchedRound) := do
  let sch ← fast100 sh.m
  if sh.claims < 1 then none
  let rep ← repRounds sh.m 6 sh.claims
  return #[(7, #[2 ^ (sch.levels.getD 0 default).cap], 0, true)] ++ rep.extract (sh.m + 1) rep.size

/-- `rec_live.sparse_rounds`: a rep's sparse phase at `sh`, in `rec_sparse.prove`'s draws: ζ_c, H2's rounds, v2's (no
coins), H3's, u's (ζ_p), at T > 1 vt's (ρ_t), and last the registered claim's (no coins). Each absorbs one message but
ζ_c's. -/
def sparsePhaseRounds (sh : RecSparse.Shape) : Array SchedRound :=
  let one := fun (n : Nat) (absorbs : Bool) => ((n, #[], 0, absorbs) : SchedRound)
  #[one RecSparse.PRE_ROUNDS false] ++ Array.replicate RecSparse.PRE_ROUNDS (one 1 true) ++ #[one 0 true] ++
    (sh.degrees.map fun _ => one 1 true).toArray ++ #[one sh.pLog true] ++
    (if sh.stacked then #[one sh.tLog true] else #[]) ++ #[one 0 true]

/-- The node counts of the tops `rs` commits, in round order: the `l`-th is a stream's `l`-th top's. -/
def topCounts (rs : Array SchedRound) : List Nat :=
  (List.range rs.size).flatMap fun k => (rs.getD k default).2.1.toList

/-- A transcript round's schedule round: its coins' count, its tops' node counts, its copies of level 0's top, and whether
it commits a message. -/
def schedOf (r : Json) : SchedRound :=
  ((coinsOf r).size, ((r.getObjValAs? (Array (Array String)) "tops").toOption.getD #[]).map (·.size),
    (r.getObjValAs? Nat "level0").toOption.getD 0, !((r.getObjValAs? (Array String) "commits").toOption.getD #[]).isEmpty)

def showSched (s : SchedRound) : String := s!"(coins {s.1}, tops {s.2.1.toList}, level0 {s.2.2.1}, commits {s.2.2.2})"

/-- **R's stream is R's at the shape staging committed R at** (`sh`, META's): R is opened at the two reps' sparse claims,
and `stacked/rep0`'s rounds `rs` (in round order) are `stackedRounds sh` exactly, round for round. Neither the stream's
declared count nor its proof's header fixes R's shape, and the registered top fixes only its cap, so a stream opened at
another `m` (or claim count) over the registered top is refused here. -/
def checkStackedShape (s : String) (sh : StackedShape) (rs : Array Json) : V Unit := do
  let at_ := s!"{s} is not R's stream at the shape staging committed R at (m {sh.m}, claims {sh.claims}, META's {STACKED_SHAPE_KEY})"
  if sh.claims != STACKED_CLAIMS then
    throw (refused s!"registration: {at_}: R is opened at the two reps' sparse claims, not {sh.claims}")
  let some want := stackedRounds sh | throw (refused s!"registration: {at_}: no fast100 schedule at m {sh.m}")
  if rs.size != want.size then throw (refused s!"registration: {at_}: {rs.size} rounds, not {want.size}")
  for (r, w) in rs.zip want, k in [0:rs.size] do
    let got := schedOf r
    if got != w then throw (refused s!"registration: {at_}: round {k} is {showSched got}, not {showSched w}")

/-- **Candidate 3's session** (`rec_live.schedule_for` with `sparse`, and `stacked`): the circuit table `t`'s two reps, both
reps' sparse phases and, if it committed R, R's stream. The transcript's streams (`streams`) and the streams its message
rounds name are those, each declared where `schedule_for` puts it (a rep's sparse phase at the end of the rep's stream, the
reps and R at round 0 of no parent), each with every round from 0 to its count; the sparse streams and R's are the coin
server's alone, since the prover's record (`innerCommits`) holds only the reps; Hello's tables and the committed roots are
`t`'s alone (the inner statement's Hello), R's root being the top its stream's binding round commits; each round's tops are
lists of `b ‖ c`, none in a sparse stream; R's binding round's first top makes R's registered entry `regR` (the
registration's `program` and its `stacked` entry, `topEntry`, as `rec_vstar.registered` holds it); R's rounds are those of
the shape staging committed R at (`shape`, META's, `checkStackedShape`); and every `b ‖ c` the session holds (roots, tops,
`c_k`) is a distinct one, since the firewall commits each under a fresh salt. -/
def candidate3 (tr : Json) (rounds : Array Json) (t : String) (names : List String)
    (msgs : Array (String × Nat × ByteArray)) (regR : Option (String × Json) := none)
    (shape : Option StackedShape := none) : V (Array InnerStream) := do
  let stacked := names.contains s!"{STACKED}/rep0"
  let want := [s!"{t}rep0", s!"{t}rep1", s!"{SPARSE}/rep0", s!"{SPARSE}/rep1"] ++ (if stacked then [s!"{STACKED}/rep0"] else [])
  let tbl := (t.dropEnd 1).toString
  let same := fun (a b : List String) => a.all b.contains && b.all a.contains && a.length == b.length
  if !same names want then
    throw s!"rec inner transcript: its message rounds' streams {names} are not one table's rep0 and rep1, both reps' sparse phases and at most R's stream ({want})"
  let declared ← match tr.getObjVal? "streams" with
    | .ok (.obj kvs) => pure kvs.toArray
    | _ => throw "rec inner transcript: streams is not an object"
  if !same (declared.toList.map (·.1)) want then
    throw s!"rec inner transcript: it declares the streams {declared.toList.map (·.1)}, not {want}"
  let decl := fun (s : String) => (declared.find? (·.1 == s)).map (·.2) |>.getD Json.null
  let count := fun (s : String) => ((decl s).getObjValAs? Nat "rounds").toOption.getD 0
  for s in want do
    let parent := if s.startsWith s!"{SPARSE}/" then s!"{t}{(s.drop (SPARSE.length + 1)).toString}" else ""
    let at_ := if parent.isEmpty then 0 else count parent
    if ((decl s).getObjValAs? String "parent").toOption != some parent || ((decl s).getObjValAs? Nat "at").toOption != some at_ then
      throw s!"rec inner transcript: {s} is not declared at {parent}'s round {at_}"
    let mine := (rounds.filter fun r => (r.getObjValAs? String "stream").toOption == some s).map fun r =>
      ((r.getObjValAs? Nat "k").toOption, r)
    if mine.size != count s then
      throw s!"rec inner transcript: {s} declares {count s} rounds and records {mine.size}"
    for k in [0:count s] do
      if !(mine.any (·.1 == some k)) then throw s!"rec inner transcript: {s} has no round {k}"
  let tables := (tr.getObjVal? "hello" |>.toOption.getD Json.null).getObjValAs? (Array String) "tables" |>.toOption.getD #[]
  let wantTables := #[tbl]
  if tables != wantTables then throw s!"rec inner transcript: Hello's tables {tables}, not {wantTables}"
  let roots ← match (tr.getObjVal? "link" |>.toOption.getD Json.null).getObjVal? "roots" with
    | .ok (.obj kvs) => pure kvs.toArray
    | _ => throw "rec inner transcript: link.roots is not an object"
  if !same (roots.toList.map (·.1)) wantTables.toList then
    throw s!"rec inner transcript: the committed roots are {roots.toList.map (·.1)}'s, not {wantTables}'s"
  let mut rootOf : Array (String × Array ByteArray) := #[]
  for (name, j) in roots do
    let .ok xs := Json.getArr? j | throw s!"rec inner transcript: {name}'s root is not a list"
    let bcs ← hexList s!"{name}'s root" (xs.map fun x => (x.getStr?).toOption.getD "")
    if bcs.isEmpty then throw s!"rec inner transcript: {name}'s root has no nodes"
    rootOf := rootOf.push (name, bcs)
  let mut out : Array InnerStream := #[]
  let mut all : Array ByteArray := rootOf.foldl (· ++ ·.2) #[]
  for s in want do
    let table := if s.startsWith s!"{SPARSE}/" then none else some (if s.startsWith s!"{STACKED}/" then STACKED else tbl)
    let committed := table == some tbl
    let mut tops : Array (Array ByteArray) := if committed then #[((rootOf.find? (·.1 == tbl)).map (·.2)).getD #[]] else #[]
    let rs := (rounds.filter fun r => (r.getObjValAs? String "stream").toOption == some s).qsort fun a b =>
      (a.getObjValAs? Nat "k").toOption.getD 0 < (b.getObjValAs? Nat "k").toOption.getD 0
    for r in rs do
      let k := (r.getObjValAs? Nat "k").toOption.getD 0
      let .ok ts := r.getObjValAs? (Array (Array String)) "tops" | throw s!"rec inner transcript: {s} round {k}'s tops are not lists of b ‖ c"
      if table.isNone && !ts.isEmpty then throw s!"rec inner transcript: {s} round {k} commits a top, and a sparse stream has no table"
      if table == some STACKED && k == 0 && ts.isEmpty then
        throw s!"rec inner transcript: {s}'s binding round commits no top: R's root is committed nowhere"
      for (top, i) in ts.zipIdx do
        if top.isEmpty then throw s!"rec inner transcript: {s} round {k}'s top {i} has no nodes"
        let bcs ← hexList s!"{s} round {k}'s top {i}" top
        if table == some STACKED && k == 0 && i == 0 then
          let made ← match regR with
            | some (program, e) => do
                let structural ← topEntryFields program bcs e
                pure (structural && (← canon (← topEntry program bcs)) == (← canon e))
            | none => pure false
          if !made then throw (refused s!"registration: {s}'s binding round's top is not {STACKED}'s registered top")
        tops := tops.push bcs
    if table == some STACKED then
      let some sh := shape
        | throw (refused s!"registration: {s} binds R, and the inner circuit's META names no shape of R ({STACKED_SHAPE_KEY}): R's m and claims are bound nowhere")
      checkStackedShape s sh rs
    let cs := ((msgs.filter (·.1 == s)).qsort (·.2.1 < ·.2.1)).map (·.2)
    all := all ++ (tops.extract (if committed then 1 else 0)).foldl (· ++ ·) #[] ++ cs.map (·.2)
    let coins ← rs.mapM fun r => (coinsOf r).mapM fun h =>
      (f128OfHex h).mapError (s!"rec inner transcript: {s} round {(r.getObjValAs? Nat "k").toOption.getD 0}'s coins: " ++ ·)
    out := out.push { name := s, commits := cs, tops := if table.isSome then tops else #[], coins }
  let mut seen : Std.HashSet String := {}
  for b in all do
    let h := hexOf b
    if seen.contains h then
      throw s!"rec inner transcript: the b ‖ c {(h.take 16).toString}… is committed twice; the firewall commits each under a fresh salt"
    seen := seen.insert h
  return out

/-- **The inner session finished**: the coin server's record ends `finish` (`rec_live.Proxy`'s `Finish`). A stopped
session's record (`end`: `stop`, a refusal or a connection that ended without `Finish`) is no inner proof, and neither is
one that names no end. This refuses only what the record says: the proxy that holds a session to its schedule runs on
the prover's side, so a record ending `finish` is not thereby on the schedule, which `checkSchedule` holds it to. -/
def checkFinished (tr : Json) : V Unit := do
  match tr.getObjValAs? String "end" with
  | .ok "finish" => pure ()
  | .ok e =>
    throw s!"rec inner transcript: the inner session ended {e}, not finish: a stopped session is no inner proof"
  | .error _ => throw "rec inner transcript: the coin server's record names no end; only a finished session is read"

/-- `rec_live.schedule_for`'s streams at the inner shape `i` over Hello's one table `t`, in its order: each stream's
name, its parent and the parent's round it is opened at, and its rounds. The circuit table's two reps at `repRounds`
(`holoRepRounds` when its circuits are private), with `i`'s RecSparse shape each rep's sparse phase at
`sparsePhaseRounds`, opened at its rep's end, and with `i`'s R shape R's stream at `stackedRounds`; none where a shape
has no schedule. -/
def scheduleOf (i : Inner) (t : String) : Option (Array (String × String × Nat × Array SchedRound)) := do
  let rep ← (if i.holo then holoRepRounds else repRounds) i.m i.kLog i.claims
  let sp := match i.sparse with
    | some sh => #[0, 1].map fun x => (s!"{SPARSE}/rep{x}", s!"{t}/rep{x}", rep.size, sparsePhaseRounds sh)
    | none => #[]
  let r : Array (String × String × Nat × Array SchedRound) ← match i.stacked with
    | some sh => do pure #[(s!"{STACKED}/rep0", "", 0, ← stackedRounds sh)]
    | none => pure #[]
  return #[0, 1].map (fun x => (s!"{t}/rep{x}", "", 0, rep)) ++ sp ++ r

/-- **The coin server's record is on the verifier's schedule** (`rec_live.on_schedule`): at the verifier's inner shape
`i`, the record `tr` declares exactly `scheduleOf i`'s streams, each at its parent's round with its round count, and its
rounds are the schedule's in its order, each its stream's round `k`, with the schedule's coin count (`n` and the coins
drawn), its tops' node counts and copies of level 0's top (`schedOf`), and a `c_k` exactly where the round absorbs a
message (one: `innerCommits`, which runs first). `rec_live.Proxy` holds a session to the same schedule round by round,
and it runs on the prover's side: a session that bypasses it reaches the coin server with any rounds, and its record
still ends `finish`. This holds that record to the schedule here, by counts alone, one comparison a round of the
schedule. It reads the schedule right only at `k_log ≥ 6`: below it `SchedRound`'s absorb flag can't tell z_partial's
round from a 2-element one, and `statements`, which every V* session's setup runs, refuses such a shape. -/
def checkSchedule (i : Inner) (tr : Json) : V Unit := do
  let at_ := "the coin server's record is off the schedule of the verifier's inner shape " ++
    s!"(m {i.m}, k_log {i.kLog}, claims {i.claims})"
  let tables := ((tr.getObjVal? "hello").toOption.getD Json.null).getObjValAs? (Array String) "tables"
  let .ok #[t] := tables | throw s!"rec inner transcript: {at_}: Hello names no one table"
  let some want := scheduleOf i t | throw s!"rec inner transcript: {at_}: the shape has no schedule"
  let declared ← match tr.getObjVal? "streams" with
    | .ok (.obj kvs) => pure kvs.toArray
    | _ => throw s!"rec inner transcript: {at_}: streams is not an object"
  let names := want.toList.map (·.1)
  let got := declared.toList.map (·.1)
  if !(got.all names.contains && names.all got.contains && got.length == names.length) then
    throw s!"rec inner transcript: {at_}: it declares the streams {got}, and the schedule {names}"
  for (s, parent, pos, rs) in want do
    let d := ((declared.find? (·.1 == s)).map (·.2)).getD Json.null
    if (d.getObjValAs? String "parent").toOption != some parent || (d.getObjValAs? Nat "at").toOption != some pos ||
        (d.getObjValAs? Nat "rounds").toOption != some rs.size then
      let place := if parent.isEmpty then "at no parent" else s!"at {parent}'s round {pos}"
      throw s!"rec inner transcript: {at_}: {s} is not declared {place} with {rs.size} rounds"
  let .ok rounds := tr.getObjValAs? (Array Json) "rounds" | throw s!"rec inner transcript: {at_}: rounds is not a list"
  let flat := want.flatMap fun (s, _, _, rs) => rs.mapIdx fun k w => (s, k, w)
  if rounds.size != flat.size then
    throw s!"rec inner transcript: {at_}: {rounds.size} rounds, and the schedule's {flat.size}"
  for (r, (s, k, w)) in rounds.zip flat, j in [0:rounds.size] do
    let (rs, rk, n) := ((r.getObjValAs? String "stream").toOption, (r.getObjValAs? Nat "k").toOption,
      (r.getObjValAs? Nat "n").toOption)
    if rs != some s || rk != some k || n != some w.1 || schedOf r != w then
      throw (s!"rec inner transcript: {at_}: round {j} is {rs.getD "no stream"} round {rk.getD 0} (n {n.getD 0}) " ++
        s!"{showSched (schedOf r)}, and the schedule's {s} round {k} (n {w.1}) {showSched w}")

/-- **Each rep's `c_k`**: the message rounds of the coin server's record of V*'s inner session (`--rec-inner-transcript`,
`rec/inner-transcript/v1`, of a session that finished: `checkFinished`),
those holding commitments, of stream `<table>/rep<x>` for `x` 0 and 1, in round order, each
its round number and its one `c_k` (`b ‖ c`), as `rec_vstage.message_rounds` reads them (rec_vstage.py:140-154). The inner
session's record (`--rec-inner-session`) is the prover's, the one check 5 reads, and holds the two reps alone: each rep's
rounds' coins must be the record's at the same round, z_partial's split aside (`repCoins`). With candidate 3's streams beside
the reps (a stream named `sparse/…` or `stacked/…`), every stream of the session, as `candidate3` reads it from the coin
server's record alone. R has no root at `Commit`, so the circuit registration binds it before any coin: the record `reg`
(`--rec-inner-registration`) must be the one the inner circuit's META binds (`bound`, its `registration`; `registration`), a
session with R's stream needs the record's `stacked` entry, which its binding round's top must make (`candidate3`); and a
session whose registration names R needs R's stream, whose opening the statement then reads (a session without it is a
downgrade). R's `m` and claims are `shape`, META's `stacked` (`stackedShapeOf`), the shape staging committed R at, which R's
stream must be opened at (`checkStackedShape`) and R's opening units are then held to (`checkPubRows`); never the stream's
own count or its proof's header. -/
def innerCommits (tr : Json) (record : String) (reg : Option Json := none) (bound : Option String := none)
    (shape : Option StackedShape := none) : V InnerCommits := do
  if (tr.getObjValAs? String "kind").toOption != some TRANSCRIPT_FORMAT then throw s!"rec inner transcript: not a {TRANSCRIPT_FORMAT} record"
  checkFinished tr
  let held ← registration reg bound
  let .ok rounds := tr.getObjValAs? (Array Json) "rounds" | throw "rec inner transcript: rounds is not a list"
  let rec_ ← match Json.parse record with
    | .ok j => pure j
    | .error e => throw s!"rec inner transcript: the inner session's record is not JSON: {e}"
  let .ok streams := rec_.getObjValAs? (Array Json) "streams" | throw "rec inner transcript: the inner session's record has no streams"
  let mut msgs : Array (String × Nat × ByteArray) := #[]
  for r in rounds do
    let cs := (r.getObjValAs? (Array String) "commits").toOption.getD #[]
    if cs.isEmpty then continue
    let (.ok s, .ok k) := (r.getObjValAs? String "stream", r.getObjValAs? Nat "k")
      | throw "rec inner transcript: a message round names no stream or round"
    if cs.size != 1 then throw s!"rec inner transcript: {s} round {k} holds {cs.size} commitments, not one"
    msgs := msgs.push (s, k, ← unhex cs[0]!)
  let names := (msgs.map (·.1)).toList.eraseDups
  let some t := names.head?.map fun n => (n.dropEndWhile (· != '/')).toString | throw "rec inner transcript: no message round holds a commitment"
  let extra := fun (n : String) => n.startsWith s!"{SPARSE}/" || n.startsWith s!"{STACKED}/"
  let c3 := names.any extra
  let t ← if c3 then
      match (names.find? (!extra ·)).map fun n => (n.dropEndWhile (· != '/')).toString with
      | some t => pure t
      | none => throw s!"rec inner transcript: its message rounds' streams {names} hold no table's reps"
    else if names.length != 2 || !names.contains s!"{t}rep0" || !names.contains s!"{t}rep1" then
      throw s!"rec inner transcript: its message rounds' streams {names} are not one table's rep0 and rep1"
    else pure t
  let repNames := [s!"{t}rep0", s!"{t}rep1"]
  let recNames := streams.toList.filterMap fun x => (x.getObjValAs? String "stream").toOption
  if recNames != repNames then
    throw s!"rec inner transcript: the inner session's record has the streams {recNames}, not {repNames} (the prover's record, which check 5 reads)"
  for s in repNames do
    let rs := (rounds.filter fun r => (r.getObjValAs? String "stream").toOption == some s).qsort fun a b =>
      (a.getObjValAs? Nat "k").toOption.getD 0 < (b.getObjValAs? Nat "k").toOption.getD 0
    let srec := (streams.find? fun x => (x.getObjValAs? String "stream").toOption == some s).getD Json.null
    for (r, c) in rs.zip (← repCoins s rs srec), k in [0:rs.size] do
      if c.any (· != coinsOf r) then throw s!"rec inner transcript: {s} round {k}'s coins are not the inner session record's"
  let stackedS := s!"{STACKED}/rep0"
  let declaredR := (tr.getObjVal? "streams" >>= (·.getObjVal? stackedS)).toOption.isSome
  let regR := held.bind fun (program, e) => e.map (program, ·)
  match names.contains stackedS, held, regR with
  | true, none, _ =>
    throw (refused s!"registration: {stackedS} binds R, and the inner circuit's META binds no circuit registration (--rec-inner-circuit, --rec-inner-registration)")
  | true, some _, none => throw (refused s!"registration: {STACKED} has no root at Commit and no registered top")
  | false, _, some _ =>
    -- declared with no rounds in candidate 3's session: `candidate3`'s layout check names it
    if !(declaredR && c3) then
      throw s!"rec inner transcript: the registration names R's top ({STACKED}), and the session has no {stackedS}: R's stream is missing (a downgrade)"
  | _, _, _ => pure ()
  if shape.isSome && regR.isNone then
    throw s!"rec inner transcript: the inner circuit's META names R's shape ({STACKED_SHAPE_KEY}), and its registration names no top of R's"
  let reps := #[0, 1].map fun x => ((msgs.filter (·.1 == s!"{t}rep{x}")).qsort (·.2.1 < ·.2.1)).map (·.2)
  if c3 then return { reps, streams := ← candidate3 tr rounds t names msgs regR shape, stacked := shape }
  return { reps }

/-- **Each sparse phase draws its schedule's coins round for round**: under the verifier's RecSparse shape, every held
stream named `sparse/…` holds `sparsePhaseRounds`' coin count at each of its rounds, as `checkCommitRounds` holds the
reps' streams to theirs (`sparseCoins` counts only the rounds that draw coins). -/
def checkSparseCoins (i : Inner) (cs : InnerCommits) : V Unit := do
  if let some sh := i.sparse then
    for st in cs.streams do
      if st.name.startsWith s!"{SPARSE}/" && st.coins.map (·.size) != (sparsePhaseRounds sh).map (·.1) then
        throw (s!"rec inner transcript: {st.name} draws {st.coins.map (·.size)} coins by round, and its schedule " ++
          s!"({RecSparse.unitName sh}) {(sparsePhaseRounds sh).map (·.1)}")

/-- **Each stream's `c_k` sit at its schedule's message rounds** (`commitRounds`): the circuit table's two reps'
(`InnerCommits.reps`) at `repRounds` of the inner shape `i`, `holoRepRounds` when its circuits are private (`i.holo`,
`rec_live.schedule_for`'s `holo`), with the verifier's RecSparse shape `sparse/rep<x>`'s at `sparsePhaseRounds` of it,
and R's (`stacked/rep0`) at `stackedRounds` of the verifier's R shape, R's held tops being those rounds' (`topCounts`).
`innerCommits` holds the reps' rounds only to the record's coins and `sparseRow` only counts a sparse stream's `c_k`, so
this check is what puts the `j`-th `c_k` the verifier holds at the schedule's `j`-th message round. `checkStackedShape`
holds R's transcript rounds to its schedule; this holds the `c_k` and tops the verifier keeps of them, and refuses R's
stream under an inner shape naming no schedule of R's. It reads `sparse/rep0` and `sparse/rep1` only, so it refuses
any other held stream named `sparse/…`. Every held stream draws its schedule's coins round for round: the reps' at
their schedule, the sparse phases' at `sparsePhaseRounds` (`checkSparseCoins`). -/
def checkCommitRounds (i : Inner) (cs : InnerCommits) : V Unit := do
  let some rep := (if i.holo then holoRepRounds else repRounds) i.m i.kLog i.claims
    | throw s!"rec-inner: the inner shape (m {i.m}, k_log {i.kLog}, claims {i.claims}) has no fast100 schedule"
  for st in cs.streams do
    if !st.name.startsWith s!"{SPARSE}/" && !st.name.startsWith s!"{STACKED}/" &&
        st.coins.map (·.size) != rep.map (·.1) then
      throw (s!"rec inner transcript: {st.name} holds {st.coins.size} rounds, and its schedule " ++
        s!"(m {i.m}, k_log {i.kLog}, claims {i.claims}) {rep.size}")
  if cs.reps.size != 2 then throw s!"rec inner transcript: {cs.reps.size} reps, not the circuit table's two"
  if let some st := cs.streams.find? fun st => st.name.startsWith s!"{SPARSE}/" &&
      st.name != s!"{SPARSE}/rep0" && st.name != s!"{SPARSE}/rep1" then
    throw (s!"rec inner transcript: it holds {st.name}, and a sparse phase is one of the two reps' " ++
      s!"({SPARSE}/rep0, {SPARSE}/rep1)")
  checkSparseCoins i cs
  let sp := match i.sparse with
    | some sh => [0, 1].filterMap fun x =>
      (cs.stream? s!"{SPARSE}/rep{x}").map fun st => (st.name, st.commits, sparsePhaseRounds sh)
    | none => []
  let r ← match cs.stream? s!"{STACKED}/rep0", i.stacked.bind stackedRounds with
    | none, _ => pure []
    | some st, none =>
      throw (s!"rec-inner: the inner transcript holds R's stream ({st.name}), and the verifier's inner shape names " ++
        "no schedule of R's")
    | some st, some rs =>
      let (got, want) := (st.tops.toList.map (·.size), topCounts rs)
      if got != want then
        throw s!"rec inner transcript: {st.name} holds tops of {got} nodes, and its schedule commits tops of {want}"
      pure [(st.name, st.commits, rs)]
  for (s, cm, rs) in (cs.reps.toList.zipIdx.map fun (cm, x) => (s!"rep {x}", cm, rep)) ++ sp ++ r do
    let (got, want) := (cm.toList.map (·.1), commitRounds rs)
    if got != want then
      let j := ((got.zip want).findIdx? fun (a, b) => a != b).getD (min got.length want.length)
      let at_ := fun (ks : List Nat) => (ks[j]?.map (s!"round {·}")).getD "no round"
      throw (s!"rec inner transcript: {s}'s message round {j} is at {at_ got}, and its schedule's at {at_ want} " ++
        s!"({got.length} message rounds, the schedule's {want.length})")

/-- **The verifier's inner shape covers the inner session**: its V* list (`statements i`) has candidate 3's statements
exactly when the transcript (`innerCommits`) has candidate 3's streams. A transcript with the sparse phases needs the
bucket's RecSparse shape (`rec-inner --sparse`), one with R's stream needs R's shape, and R's shape is the one the inner
circuit's META names (`InnerCommits.stacked`); an inner shape naming either over a session without it is refused too.
So no V* list leaves out RecSparse, R's levels or R's opening of a session that has them. Then each stream's `c_k` sit
at its schedule's message rounds, and R's tops at its schedule's (`checkCommitRounds`). -/
def checkCovers (i : Inner) (cs : InnerCommits) : V Unit := do
  let sparse := cs.streams.any (·.name.startsWith s!"{SPARSE}/")
  let stacked := (cs.stream? s!"{STACKED}/rep0").isSome
  let showR := fun (r : Option StackedShape) => (r.map fun r => s!"(m {r.m}, claims {r.claims})").getD "none"
  if sparse && i.sparse.isNone then
    throw ("rec-inner: the inner transcript holds candidate 3's sparse phases, and the verifier's inner shape names no " ++
      "RecSparse shape (rec-inner --sparse): its V* list would have no RecSparse statement")
  if !sparse && i.sparse.isSome then
    throw "rec-inner: the verifier's inner shape names a RecSparse shape (--sparse), and the inner transcript holds no sparse phase"
  if stacked && i.stacked.isNone then
    throw (s!"rec-inner: the inner transcript holds R's stream ({STACKED}/rep0), and the verifier's inner shape names no shape of " ++
      s!"R ({STACKED_SHAPE_KEY}): its V* list would have no level of R's or R's opening")
  if i.stacked != cs.stacked then
    throw s!"rec-inner: the verifier's inner shape names R's shape {showR i.stacked}, and the inner circuit's META {showR cs.stacked} ({STACKED_SHAPE_KEY})"
  checkCommitRounds i cs

/-- **Candidate 3's inner shape is its inner statement's**: it names the bucket's RecSparse shape (`rec-inner --sparse`),
whose one type makes every private `τ` entry uniquely zero, and its `m`, `k_log` and region count are those of the inner
statement check 5 sets up (`m`, `kLog`, `regions`), as `innerOf` reads them, so V*'s list, the streams it holds and the
class the sparse phase is weighed at are one table's. -/
def checkInnerShape (i : Inner) (m kLog regions : Nat) : Except String Unit := do
  let some sp := i.sparse | throw ("rec-inner: candidate 3's inner shape names no RecSparse shape (rec-inner --sparse): " ++
    "its V* list would have no RecSparse statement, level of R's or part of R's opening")
  if sp.tLog != 0 then
    throw s!"rec-inner: candidate 3 supports one sparse type (T_LOG 0), not T_LOG {sp.tLog}"
  if i.m != m || i.kLog != kLog || i.regions != regions then
    throw (s!"rec-inner: the verifier's inner shape (m {i.m}, k_log {i.kLog}, regions {i.regions}) is not the inner " ++
      s!"statement's (m {m}, k_log {kLog}, regions {regions})")

/-- `checkInnerShape` at the inner statement `st`. -/
def checkInnerStmt (i : Inner) (st : Stmt) : Except String Unit :=
  checkInnerShape i st.m st.c.kLog st.regions.size

/-- `rec-opening-c<i>`'s `i`. -/
def openingIndex (value : String) : Option Nat :=
  if value.startsWith "rec-opening-c" then (value.drop 13).toString.toNat? else none

/-- `rec-top-stacked-L<l>`'s `l`. -/
def stackedTopLevel (value : String) : Option Nat :=
  if value.startsWith "rec-top-stacked-L" then (value.drop 17).toString.toNat? else none

/-- `rec-sparse-<p>`'s `p` when `p` is one of RecSparse's message ports at some shape (`pre<j>`, `v2`, `sc<j>`, `u`, `vt`),
and `claim` for `rec-claim`: not `h` (registered as its `rec-c<i>`), nor `v`, `zero` or `tau`, which no round commits. -/
def sparsePort (value : String) : Option String :=
  if value == CLAIM_VALUE then some "claim" else
  if !value.startsWith "rec-sparse-" then none else
  let p := (value.drop 11).toString
  let idx := fun (pre : String) => p.startsWith pre && ((p.drop pre.length).toString.toNat?).isSome
  if p == "v2" || p == "u" || p == "vt" || idx "pre" || idx "sc" then some p else none

/-- RecSparse's message ports at `sh` in the order its sparse phase commits them (`message_ports` without `h`): `pre<j>`,
`v2`, `sc<j>`, `u`, `vt` at T > 1, and last `claim`, the claim round's (`rec_live.sparse_rounds`). -/
def sparseMessages (sh : RecSparse.Shape) : List String :=
  ((RecSparse.messagePorts sh).map (·.1)).filter fun p => p != "h"

/-- Row `x` of RecSparse's message port `p` at `sh`: `sparse/rep<x>`'s `i`-th message round's `c_i`, `i` `p`'s place in
`sparseMessages sh`, the stream holding exactly those message rounds. -/
def sparseRow (cs : InnerCommits) (sh : RecSparse.Shape) (p : String) (x : Nat) : Except String (String × ByteArray) := do
  let names := sparseMessages sh
  let some i := names.idxOf? p | .error s!"{p} is not one of RecSparse's message ports {names} at {RecSparse.unitName sh}"
  let s := s!"{SPARSE}/rep{x}"
  let some st := cs.stream? s | .error s!"the inner transcript has no stream {s}"
  if st.commits.size != names.length then
    .error s!"{s} holds {st.commits.size} message rounds, not {RecSparse.unitName sh}'s {names.length} ({names})"
  let (k, c) := st.commits[i]!
  .ok (s!"{s}'s c_{i} ({p}, inner round {k})", c)

/-- **`rec-claim`'s row `x`, the verifier's own**: rep `x`'s registered claim is its sparse phase's last message, the claim
round (`rec_live.sparse_rounds`, after u's, and vt's at T > 1), whose `c_k` the coin server records; so row `x` of
`rec-claim` is `sparse/rep<x>`'s claim round's `c_k` at the verifier's RecSparse shape `sh` (`rec-inner --sparse`), and a claim
row the firewall did not commit, however consistently registered, opens no statement that reads it (RecSparse's `claim`, R's
opening's `claim<k>`). -/
def claimRow (cs : InnerCommits) (sh : RecSparse.Shape) (x : Nat) : Except String (String × ByteArray) :=
  sparseRow cs sh "claim" x

/-- A registered value the inner transcript fixes, and what its row `x` is there: `rec-c<i>`, rep `x`'s `i`-th message
round's `c_i`; `rec-opening-c<i>` (R's registered opening, `rec_vstage.opening_messages`), R's one stream's `i`-th message
round's `c_i`, at row 0; `rec-top-stacked-L<l>` (R's levels, `rec_vstage.stage_stacked`), node `x` of R's level-`l` top;
`rec-sparse-<p>` (RecSparse's message ports, `rec_vstage.sparse_messages`), `sparseRow`; `rec-claim`, `claimRow`; both at
`sparse`, the verifier's RecSparse shape.
`none` for any other value; else the row's description and its `b ‖ c`, or why the transcript has no such row. -/
def fixedRow (cs : InnerCommits) (value : String) (x : Nat) (sparse : Option RecSparse.Shape := none) :
    Option (Except String (String × ByteArray)) :=
  match cIndex value, openingIndex value, stackedTopLevel value, sparsePort value with
  | some i, _, _, _ => some <| match (cs.reps[x]?).bind (·[i]?) with
    | some (k, c) => .ok (s!"rep {x}'s c_{i} (inner round {k})", c)
    | none => .error s!"the inner transcript has no rep {x} message round {i}"
  | _, some i, _, _ => some <| match (if x == 0 then (cs.stream? s!"{STACKED}/rep0").bind (·.commits[i]?) else none) with
    | some (k, c) => .ok (s!"R's c_{i} ({STACKED}/rep0 round {k})", c)
    | none => .error s!"the inner transcript has no row {x} of R's message round {i}, which is R's stream's at row 0"
  | _, _, some l, _ => some <| match ((cs.stream? s!"{STACKED}/rep0").bind (·.tops[l]?)).bind (·[x]?) with
    | some c => .ok (s!"node {x} of R's level {l} top", c)
    | none => .error s!"the inner transcript has no node {x} of R's level {l} top"
  | _, _, _, some p => some <| do
    let some sh := sparse
      | .error s!"{value} is fixed by the sparse phase, and the verifier names no RecSparse shape (rec-inner --sparse)"
    if p == "claim" then claimRow cs sh x else sparseRow cs sh p x
  | none, none, none, none => none

/-- An R opening unit (`rec-opening/m<m>-c<claims>-p<j>`, `RecOpening.params`) opens R at the shape R's stream was held to
(`InnerCommits.stacked`, META's): its unit name is the prover's, and R's `m` and claims are the inner statement's. Any other
unit passes. -/
def checkOpeningShape (unit : String) (cs : InnerCommits) : V Unit := do
  let some (m, c, _) := RecOpening.params unit | return
  if cs.stacked != some { m, claims := c } then
    let held := (cs.stacked.map fun sh => s!"m {sh.m}, claims {sh.claims}").getD "no shape"
    throw (refused s!"registration: {unit} opens R at (m {m}, claims {c}), not at the shape staging committed R at ({held}, META's {STACKED_SHAPE_KEY})")

/-- **`pub.bin`'s rows at its reads of what the inner transcript fixes are the `b ‖ c` the verifier holds**: for each port of
the verifier's reads file registering a value `fixedRow` names (`rec-c<i>`, and with R `rec-opening-c<i>` and
`rec-top-stacked-L<l>`, and with the sparse phase `rec-sparse-<p>` and `rec-claim`), every row of the port's table, at the
registered position `x` it opens, is that value's row `x` in `innerCommits`: for `rec-c<i>`, rep `x`'s `i`-th message round's
`c_i`; for `rec-claim`, `claimRow`. `sparse` is the verifier's RecSparse shape (`rec-inner --sparse`), a rec-sparse unit's
own. A rec-sparse unit's `claim` port and an R opening unit's `claim<k>` ports read `rec-claim` in the verifier's reads.
`Registered.check` puts each instance's row at the position its unit reads. An R opening unit
(`rec-opening/m<m>-c<claims>-p<j>`) is R's at the shape R's stream was held to (`InnerCommits.stacked`, META's). A table
with a port reading the reps' `c_k` (`rec-c<i>`) labels its instances `0` to `n - 1` in order
(`Registered.unitIndices`), as a V* session's full draw of its two reps does, so instance `x` reads rep `x`'s row. -/
def checkPubRows (st : Stmt) (registered : Option (Array Registered.Port))
    (commits : Option InnerCommits) (sparse : Option RecSparse.Shape := none) : V Unit := do
  let unit := st.c.unit.name
  let sparse := (RecSparse.params unit).or sparse
  let claimPorts := (st.c.ports.toList.map (·.name)).filter fun nm =>
    ((RecSparse.params unit).isSome && nm == "claim") ||
      ((RecOpening.params unit).isSome && nm.startsWith "claim" && ((nm.drop 5).toString.toNat?).isSome)
  for nm in claimPorts do
    match (registered.getD #[]).find? (·.name == nm) with
    | some reg =>
      if reg.value != CLAIM_VALUE then
        throw s!"rec-pub: {unit}: port {nm} reads {reg.value}, not {CLAIM_VALUE}, the claim the sparse phase's claim round commits"
    | none => throw s!"rec-pub: {unit}: the verifier's registered reads hold no port {nm}: its {CLAIM_VALUE} would be the prover's"
  let regs := (registered.getD #[]).filter fun reg => (fixedRow {reps := #[]} reg.value 0).isSome
  let some reg0 := regs[0]? | return
  let some cs := commits
    | throw s!"rec-pub: {unit}: the verifier holds no inner transcript (--rec-inner-transcript with --rec-inner-session), so no c_k to check port {reg0.name}'s rows of {reg0.value} against"
  checkOpeningShape unit cs
  let idx ← Registered.unitIndices st.pub
  if regs.any (fun reg => (cIndex reg.value).isSome) && idx != Array.range st.pub.n then
    throw (s!"rec-pub: {unit}: it reads the reps' c_k, and its instances are labelled {idx}, not the units below " ++
      s!"{st.pub.n} in order as a V* session's full draw labels them, so an instance would read another rep's row")
  let some t := st.pub.tables | throw s!"rec-pub: {unit}: {reg0.name} is not a table of a shared-row file"
  for reg in regs do
    let some p := (List.range st.c.ports.size).find? (fun p => st.c.ports[p]!.name == reg.name)
      | throw s!"rec-pub: {unit}: {reg.name} is not a row of the circuit"
    let (positions, _) ← Registered.entry st.pub reg
    for r in [0:positions.size] do
      let x := positions[r]!
      match fixedRow cs reg.value x sparse with
      | some (.ok (what, c)) =>
        if Registered.tableRow st.pub t p r != c then
          throw s!"rec-pub: {unit}: port {reg.name}'s table row {r} is not {what} of the inner transcript"
      | some (.error why) =>
        throw s!"rec-pub: {unit}: port {reg.name}'s table row {r} reads row {x} of {reg.value}, and {why}"
      | none => throw s!"rec-pub: {unit}: port {reg.name}: the inner transcript fixes no {reg.value}"

/-! ### The circuit table's tops and the link points, at every shape -/

/-- `rec-top-L<l>`'s `l` (`rec_outer.top_value_name`): the circuit table's level-`l` tops, which that level's openings
climb to. -/
def topLevel (value : String) : Option Nat :=
  if value.startsWith "rec-top-L" then (value.drop 9).toString.toNat? else none

/-- What the coin server's record of the inner session fixes of the circuit table at every shape, dense, holo or
candidate 3's: Hello's one table (`table`), its level-0 top (`root`, `root_B` from `Commit`, the record's `link.roots`,
one `b ‖ c` a node), each rep's later salted tops (`tops`, rep `x` at `x`: each top's round and its nodes' `b ‖ c`, in
round order, `rec_vstar.tops` past level 0) and the link points it drew after `root_B` (`points`, one array of `m` a
point). -/
structure InnerLink where
  table : String
  root : Array ByteArray
  tops : Array (Array (Nat × Array ByteArray))
  points : Array (Array F128)
  deriving Inhabited

private def f128sOfHex (ctx : String) (xs : Array String) : V (Array F128) :=
  xs.mapM fun h => (f128OfHex h).mapError (s!"rec inner transcript: {ctx}: " ++ ·)

/-- **The circuit table's tops and the link points, as the coin server recorded them** (`--rec-inner-transcript`,
`rec/inner-transcript/v1`, of a session that finished: `checkFinished`):
Hello's one table `t`, its `root_B` (`link.roots`, `t`'s alone), every salted top of
`t/rep0`'s and `t/rep1`'s rounds in round order (a round's copies of a level-0 top, `level0`, are not tops of their own),
and the link points (`link.points`, `hello.link`'s `points × m` coins). The inner session's record `record` (the
prover's, which check 5 reads, `RecCoins.Inner.load`) must hold those very points: check 5 takes the extra claims at the
record's points, and only the coin server's were drawn after `root_B`. -/
def innerLink (tr : Json) (record : String) : V InnerLink := do
  if (tr.getObjValAs? String "kind").toOption != some TRANSCRIPT_FORMAT then
    throw s!"rec inner transcript: not a {TRANSCRIPT_FORMAT} record"
  checkFinished tr
  let hello := (tr.getObjVal? "hello").toOption.getD Json.null
  let tables := (hello.getObjValAs? (Array String) "tables").toOption.getD #[]
  let some t := (if tables.size == 1 then tables[0]? else none)
    | throw s!"rec inner transcript: Hello's tables {tables}, not one table"
  let link := (tr.getObjVal? "link").toOption.getD Json.null
  let roots ← match link.getObjVal? "roots" with
    | .ok (.obj kvs) => pure kvs.toArray
    | _ => throw "rec inner transcript: link.roots is not an object"
  let some xs := (if roots.size == 1 && roots[0]!.1 == t then (roots[0]!.2.getArr?).toOption else none)
    | throw s!"rec inner transcript: the committed roots are {roots.toList.map (·.1)}'s, not {t}'s alone, as a list"
  let root ← hexList s!"{t}'s root_B" (xs.map fun x => (x.getStr?).toOption.getD "")
  if root.isEmpty then throw s!"rec inner transcript: {t}'s root_B has no nodes"
  let .ok rounds := tr.getObjValAs? (Array Json) "rounds" | throw "rec inner transcript: rounds is not a list"
  let tops ← #[0, 1].mapM fun (x : Nat) => do
    let s := s!"{t}/rep{x}"
    let rs := (rounds.filter fun r => (r.getObjValAs? String "stream").toOption == some s).qsort fun a b =>
      (a.getObjValAs? Nat "k").toOption.getD 0 < (b.getObjValAs? Nat "k").toOption.getD 0
    if rs.isEmpty then throw s!"rec inner transcript: {s} has no round"
    let mut out : Array (Nat × Array ByteArray) := #[]
    for r in rs do
      let k := (r.getObjValAs? Nat "k").toOption.getD 0
      let .ok ts := r.getObjValAs? (Array (Array String)) "tops"
        | throw s!"rec inner transcript: {s} round {k}'s tops are not lists of b ‖ c"
      for (top, i) in ts.zipIdx do
        if top.isEmpty then throw s!"rec inner transcript: {s} round {k}'s top {i} has no nodes"
        out := out.push (k, ← hexList s!"{s} round {k}'s top {i}" top)
    pure out
  let spec := (hello.getObjVal? "link").toOption.getD Json.null
  let (.ok np, .ok m) := (spec.getObjValAs? Nat "points", spec.getObjValAs? Nat "m")
    | throw "rec inner transcript: Hello's link names no points and m"
  let .ok flat := link.getObjValAs? (Array String) "points"
    | throw "rec inner transcript: link.points is not a list of coins"
  if flat.size != np * m then throw s!"rec inner transcript: {flat.size} link coins for {np} points of {m}"
  let coins ← f128sOfHex "link.points" flat
  let points := (Array.range np).map fun i => coins.extract (i * m) (i * m + m)
  let rec_ ← match Json.parse record with
    | .ok j => pure j
    | .error e => throw s!"rec inner transcript: the inner session's record is not JSON: {e}"
  let .ok rp := (rec_.getObjVal? "link" >>= (·.getObjValAs? (Array (Array String)) "points"))
    | throw "rec inner transcript: the inner session's record holds no link points (link.points)"
  let held ← rp.mapM (f128sOfHex "the inner session's record's link points")
  if held != points then
    throw ("rec inner transcript: the inner session's record's link points are not the ones the coin server drew " ++
      "(link.points): check 5 would take the extra claims at points the prover chose")
  return { table := t, root, tops, points }

/-- Row `x` of `rec-top-L<l>` (`rec_outer.TopValue`): at level 0 node `x` of `root_B`, the one top both reps' openings
climb to; at a later level `l` rep 0's level-`l` top, then rep 1's (each rep's `l`-th top, `l − 1` past `root_B`). -/
def topRow (lk : InnerLink) (l x : Nat) : Except String (String × ByteArray) :=
  if l == 0 then
    match lk.root[x]? with
    | some c => .ok (s!"node {x} of {lk.table}'s root_B", c)
    | none => .error s!"{lk.table}'s root_B has {lk.root.size} nodes, no node {x}"
  else
    match (lk.tops[0]?.bind (·[l - 1]?)), (lk.tops[1]?.bind (·[l - 1]?)) with
    | some (_, t0), some (_, t1) =>
      match t0[x]?, t1[x - t0.size]? with
      | some c, _ => .ok (s!"node {x} of rep 0's level {l} top", c)
      | none, some c => .ok (s!"node {x - t0.size} of rep 1's level {l} top", c)
      | none, none => .error s!"the reps' level {l} tops have {t0.size + t1.size} nodes, no row {x}"
    | _, _ => .error s!"the inner transcript has no level {l} top of both reps"

/-- Port `name`'s table rows `row r` of `rec-top-L<l>` (`value`), row `r` at the registered position `positions[r]`, are
`topRow lk l`'s there. -/
def checkTopRowsAt (unit name value : String) (lk : InnerLink) (l : Nat) (positions : Array Nat)
    (row : Nat → ByteArray) : V Unit := do
  for r in [0:positions.size] do
    match topRow lk l positions[r]! with
    | .ok (what, c) =>
      if row r != c then
        throw s!"rec-tops: {unit}: port {name}'s table row {r} is not {what}, which the coin server committed"
    | .error why =>
      throw s!"rec-tops: {unit}: port {name}'s table row {r} reads row {positions[r]!} of {value}, and {why}"

/-- **`pub.bin`'s rows at its reads of the circuit table's tops are the `b ‖ c` the coin server recorded**: for each port
of the verifier's reads file registering `rec-top-L<l>` (`topLevel`), every row of the port's table, at the registered
position `x` it opens, is row `x` of `topRow lk l` (`checkTopRowsAt`). So an opening's `out` row opens a node the firewall
committed before the coins after it, never a top the prover chose after them. -/
def checkTopRows (st : Stmt) (registered : Option (Array Registered.Port)) (link : Option InnerLink) : V Unit := do
  let unit := st.c.unit.name
  let regs := (registered.getD #[]).filter fun reg => (topLevel reg.value).isSome
  let some reg0 := regs[0]? | return
  let some lk := link
    | throw s!"rec-tops: {unit}: the verifier holds no inner transcript (--rec-inner-transcript with --rec-inner-session), so no committed tops to check port {reg0.name}'s rows of {reg0.value} against"
  let some t := st.pub.tables | throw s!"rec-tops: {unit}: {reg0.name} is not a table of a shared-row file"
  for reg in regs do
    let some l := topLevel reg.value | continue
    let some p := (List.range st.c.ports.size).find? (fun p => st.c.ports[p]!.name == reg.name)
      | throw s!"rec-tops: {unit}: {reg.name} is not a row of the circuit"
    let (positions, _) ← Registered.entry st.pub reg
    checkTopRowsAt unit reg.name reg.value lk l positions (Registered.tableRow st.pub t p)

/-- **The circuit table's tops are its schedule's** (`rec_live.schedule_for`): `root_B` has level 0's `2^cap` nodes, and
each rep's later tops sit at its schedule's rounds with its node counts (`repRounds` of the inner shape `i`,
`holoRepRounds` when its circuits are private), so a rep's `l`-th top past `root_B` is level `l`'s, committed in the round
that absorbs it, before that level's query coins. -/
def checkLinkRounds (i : Inner) (lk : InnerLink) : V Unit := do
  let (some sch, some rep) := (fast100 i.m, (if i.holo then holoRepRounds else repRounds) i.m i.kLog i.claims)
    | throw s!"rec-inner: the inner shape (m {i.m}, k_log {i.kLog}, claims {i.claims}) has no fast100 schedule"
  let cap0 := 2 ^ (sch.levels.getD 0 default).cap
  if lk.root.size != cap0 then
    throw s!"rec inner transcript: {lk.table}'s root_B has {lk.root.size} nodes, and its schedule's level 0 top {cap0}"
  let want := (List.range rep.size).flatMap fun k => (rep.getD k default).2.1.toList.map (k, ·)
  for (tops, x) in lk.tops.toList.zipIdx do
    let got := tops.toList.map fun (k, t) => (k, t.size)
    if got != want then
      throw s!"rec inner transcript: rep {x}'s tops (round, nodes) are {got}, and its schedule's {want}"

/-- Table `j`'s draw among `count` tables of a session drawn by `d`, as `Stmt.setupTables` builds it: `d` at one table,
else part `j` of `d`'s units, a subset draw of the population. -/
def tableDraw (d : Json) (count j : Nat) : Json :=
  if count ≤ 1 then d else
    match Draw.UnitDraw.ofJson d with
    | .ok u => Json.mkObj [("law", Json.str "subset"), ("population", Lean.toJson u.population),
        ("k", Lean.toJson (u.units.size / count)),
        ("units", Json.arr ((u.units.extract (j * (u.units.size / count))
          (j * (u.units.size / count) + u.units.size / count)).map Lean.toJson))]
    | .error _ => d

/-- Table `j`'s statement labels its instances with the units its draw `dj` proves. -/
def checkLabel (dj : Json) (pub : Public) (j : Nat) : V Unit := do
  let D ← Draw.UnitDraw.ofJson dj
  let idx ← Registered.unitIndices pub
  if idx != D.proved then
    throw s!"rec-labels: table {j} labels its instances {idx}, not the units its draw proves {D.proved}"

/-- **A drawn session of V*'s labels each instance with the unit it is**: every table's statement names, in its header's
`units.indices`, exactly the units its draw proves (`tableDraw`). A registered read's position is read at an instance's
label (`Registered.checkPort`), so the label must be the unit. V*'s populations are their units `0..n-1`, so an honest
statement's labels (the population's labels of the drawn units) are those units; a population labelled from another
start would not pass, and only a session with an inner shape (`--rec-inner`) is checked. A session without a draw
passes. -/
def checkLabels (draw : Option Json) (count : Nat) (sts : Array Stmt) : V Unit :=
  match draw with
  | none => pure ()
  | some d => (Array.range sts.size).forM fun j => checkLabel (tableDraw d count j) sts[j]!.pub j

/-- **A value the verifier's shared rows register is one registration in every session** (`SharedOwnRoot`): each port of
`--registered` reading a value some entry of `--rec-rows` reads agrees with the first such entry in every field but reads
(`portDiff?`: value, root, leaf count, row width, domain), and a port reading R's tops `rec-top-stacked-L<l>`
(`stackedTopLevel`) or the table's tops `rec-top-L<l>` (`topLevel`: level `l`'s openings' `out` and its RecCap's `top`)
needs such an entry, so each level's query reads open the one registration its RecCap holds every node of. -/
def checkShared (s : VStar.Stmt) (registered : Array Registered.Port) (rows : Option (Array Registered.Port)) :
    V Unit := do
  for p in registered do
    match rows.bind (·.find? (·.value == p.value)) with
    | some w =>
      if let some d := portDiff? p w false then
        throw s!"{s.name}: its port {p.name} reads {p.value}, which the verifier's shared rows (--rec-rows) register as {w.name}: {d}"
    | none =>
      if (stackedTopLevel p.value).isSome then
        throw s!"{s.name}: its port {p.name} reads R's tops {p.value}; pass the verifier's shared rows (--rec-rows F), which register it"
      if (topLevel p.value).isSome then
        throw (s!"{s.name}: its port {p.name} reads the table's tops {p.value}, which its level's RecCap holds; pass the " ++
          "verifier's shared rows (--rec-rows F), which register it")

/-- With an inner shape, the statement must be V*'s: `tags`, the statement the verifier set it up as (`--statement`), is
V*'s own `Tags.circuit`, compared by name (`Tags.find?` gives each name one statement); it holds no lookup table
(`tables`, the verifier's `--tables`/`--archive` MUFU tables, is empty: V*'s circuits read none); it needs the verifier's own
partition object and program, which of V*'s statements the session is (`part`, `--rec-part J`) and the verifier's own
bindings (`bindings`, `--rec-bindings F`); its circuit must be in `ProvedScope.scopeOk`, every row a `sha512/row/v1` row
(`v1Ports`), never `scopeOkH`'s `bitRows`, which `ProvedScope.check` admits under hidden outputs and V*'s statement
(`VStmt.hscope`) does not cover; then `checkOwnBindings` of those bindings against the partition object and
program, `checkCircuit` on the statement's circuit and population as statement `part` under those bindings,
`checkMsgNames` on its circuit's `m<k>` ports, and `checkOwn`, `checkRows` and `checkAcc` (against the verifier's shared rows `rows`, `--rec-rows`), `checkAccReads` and
`checkShared` (against `rows`) on the verifier's own registered reads. -/
def checkStmt (inner : Option Inner) (tags : Tags) (tables : List (String × ByteArray)) (st : Stmt) (circuitB : ByteArray)
    (partition : Option Json := none)
    (program : Option ByteArray := none) (registered : Option (Array Registered.Port) := none) (part : Option Nat := none)
    (bindings : Option VStar.Bindings := none) (rows : Option (Array Registered.Port) := none) : V Unit :=
  let name := st.c.unit.name
  match inner with
  | some i => do
    if tags.name != Tags.circuit.name then
      throw s!"{name}: one of V*'s statements, which are {Tags.circuit.name} statements (--statement {Tags.circuit.name}), not {tags.name}"
    if !tables.isEmpty then
      throw s!"{name}: one of V*'s statements, which read no lookup table; the verifier holds {tables.map (·.1)} (--tables, --archive)"
    let (some obj, some prog) := (partition, program)
      | throw s!"{name}: one of V*'s statements; pass the verifier's own partition object and program (--partition F, --program F)"
    let some j := part
      | throw s!"{name}: one of V*'s statements; pass which of V*'s statements this session is (--rec-part J, its index in rec-statements' list)"
    let some f := bindings
      | throw s!"{name}: one of V*'s statements; pass the verifier's own bindings (--rec-bindings F, from verity_flock.rec_bindings)"
    if !ProvedScope.scopeOk st.c then
      throw s!"{name}: one of V*'s statements, whose circuits are in ProvedScope.scopeOk (every row sha512/row/v1, VStmt.hscope), not scopeOkH's bit rows: {ProvedScope.scopeWhy false st.c}"
    checkOwnBindings name f obj prog
    let s ← checkCircuit i j st.c (population st) circuitB f
    checkMsgNames s st.c i.holo
    if let some r := registered then
      checkOwn s r
      checkRows s r rows
      checkAcc s r rows
      checkAccReads i s r
      checkShared s r rows
  | none =>
    if isVStar name then
      throw s!"{name}: one of V*'s statements; pass the verifier's own inner shape (--rec-inner F, from rec-inner)"
    else pure ()

end RecVStar
end Flock
