import Proofs.Flock.Recursive.SaltFinder
import Proofs.Flock.VBridge.Decode
import Flock.RecCoins

/-!
# The value tie's extractor and its finder, as the guarantee reads them

`valueColl` (the value tie's extractor: `FlockC3.RegLevels.value_tie`, `C3/RegLevels/ValueTie.lean`) and the finder
that plays the recursive audit once and outputs its pair (`valueFinder`, `Recursive/ValueFinder.lean`), here so that
`RecursiveAuditC3At` can read `valueFinder` without importing the value tie's proofs, which import it:

* `up`, `lev`, `sibIdx`, `sib`, `sibsFrom`, `sibsV`: `FV3.root`'s tree over the verifier's leaves and a leaf's siblings
  in it; `openPairV`: the opening's extractor against it; `leavesV`: the leaves `checkValue` builds over its rows;
  `valueColl`: the extractor;
* `VRead`, `ValueSrc`: a read of V*'s sessions, and the verifier's rows and salts at each; `valueAt`, `ValueColl`,
  `valuePick`, `valueFinder`: `valueColl` at a read, some registered read where it collides, and the finder;
* `stackedCoefRows`, `stackedDirsRows`, `stackedLevels`, `openingVRows`, `valueRows`, `valueSrcC3`: the source
  `RecursiveAuditC3At` reads, the rows check 5 registers itself (their ties: `Recursive/ValueSource.lean`).
-/

namespace FlockC3.RegLevels

open Flock
open FlockSoundness FlockSoundness.Audit FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open FlockSoundness.Refine (H512)
open FlockSoundness.Binding (hm96Input hm96Pair)

/-- One level of `FV3.root`'s tree over the one below. -/
def up (dom : ByteArray) (d : ℕ) (a : Array ByteArray) : Array ByteArray :=
  (Array.range (a.size / 2)).map fun i => Registered.node dom d i a[2 * i]! a[2 * i + 1]!

/-- Level `d` of `FV3.root`'s tree: the leaves padded to `nextPow2`, then `up`. -/
def lev (dom : ByteArray) (leaves : Array ByteArray) : ℕ → Array ByteArray
  | 0 => leaves ++ (Array.range (nextPow2 leaves.size - leaves.size)).map fun k =>
      FV3.hash "pad" [dom, FV3.uint (leaves.size + k)] Sha512.hash
  | d + 1 => up dom d (lev dom leaves d)

/-- The index of node `j`'s sibling. -/
def sibIdx (j : ℕ) : ℕ := if j % 2 = 0 then j + 1 else j - 1

/-- The sibling at level `d` on leaf `i`'s path. -/
def sib (dom : ByteArray) (leaves : Array ByteArray) (i d : ℕ) : ByteArray := (lev dom leaves d)[sibIdx (i / 2 ^ d)]!

/-- Leaf `i`'s siblings from level `d`, whose nodes are `a`, up `k` levels, each level computed once. -/
def sibsFrom (dom : ByteArray) (i : ℕ) : ℕ → ℕ → Array ByteArray → List ByteArray
  | 0, _, _ => []
  | k + 1, d, a => a[sibIdx (i / 2 ^ d)]! :: sibsFrom dom i k (d + 1) (up dom d a)

/-- **Leaf `i`'s siblings in `FV3.root`'s tree**, lowest first. -/
def sibsV (dom : ByteArray) (leaves : Array ByteArray) (i : ℕ) : List ByteArray :=
  sibsFrom dom i (log2ceil leaves.size) 0 (lev dom leaves 0)

/-- **The opening's extractor against the verifier's tree**: for payloads `x` and `y` climbing at `pos` through the
sibling lists `s` and `t`, their frame-v3 leaves' frames if those leaves are one, else the climb's pair. -/
def openPairV (dom : ByteArray) (pos : ℕ) (x : ByteArray) (s : List ByteArray) (y : ByteArray) (t : List ByteArray) :
    ByteArray × ByteArray :=
  let lx := Registered.leafIn dom pos x
  let ly := Registered.leafIn dom pos y
  if Sha512.hash lx = Sha512.hash ly then (lx, ly)
  else Registered.climbPair dom (Sha512.hash lx) (Sha512.hash ly) pos 0 s t

/-- The leaves `checkValue` builds over `rows` under `salt`. -/
def leavesV (dom : ByteArray) (rows : Array (Array ℕ)) (salt : ℕ → ByteArray) : Array ByteArray :=
  (Array.range rows.size).map fun i =>
    Registered.rowLeaf dom i (Hm96.Default512.treeLeaf (RecCoins.commitRow rows[i]! (salt i)))

/-- **The value tie's extractor** at row `i` of the verifier's `rows`, registered under `reg` with the salts `salt`,
against a registrant's opening of the row bytes `r` under the salt `y` through `path`: nothing where `r` is the
verifier's row's bytes (`rowInV`); else hm96's pair (`hm96Pair`: the leaf preimage, `commitRow`'s salt digest or its
row digest) where the two hm96 leaves are one; else the opening's pair (`openPairV`) up the registrant's path and the
verifier's tree. Each pair holds both preimages. -/
def valueColl (reg : Registered.Port) (rows : Array (Array ℕ)) (salt : ℕ → ByteArray) (i : ℕ) (r y : List UInt8)
    (path : ByteArray) : Option (List UInt8 × List UInt8) :=
  if r = rowInV rows[i]! then none
  else if H512 (hm96Input r y) = H512 (hm96Input (rowInV rows[i]!) (salt i).data.toList) then
    some (hm96Pair r y (rowInV rows[i]!) (salt i).data.toList)
  else
    let x := openPairV reg.domain i (H512 (hm96Input r y)).1 (Registered.sibs path (log2ceil reg.leaves))
      (H512 (hm96Input (rowInV rows[i]!) (salt i).data.toList)).1 (sibsV reg.domain (leavesV reg.domain rows salt) i)
    some (x.1.data.toList, x.2.data.toList)

end FlockC3.RegLevels

namespace FlockSoundness.Discharge.Recursion

open Game Audit Refine StrictCR

variable {n : ℕ} {L : Law n} {Reg Cm Reg₂ Reg' : Type} {I : Inner} {S : ℕ} {Vs : Fin S → VStmt}

/-- **A read of V*'s sessions**: the session `j`, the unit `u` and the row port `q`, read at `.inl (u, q)`. -/
abbrev VRead (Vs : Fin S → VStmt) : Type := (j : Fin S) × Fin (Vs j).nv × Fin (Vs j).c.ports.size

/-- **The verifier's rows of the values it registers itself**: at each history `(R, ω, t)`, registration `R₂` and read
`(j, u, q)`, the rows and their salts `RecCoins.checkValue` holds the port to, or none. -/
abbrev ValueSrc (L : Law n) (Reg Cm Reg₂ : Type) (I : Inner) (Vs : Fin S → VStmt) : Type :=
  Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → VRead Vs → Option (Array (Array ℕ) × (ℕ → ByteArray))

/-- **`valueColl` at a read**: at the read `x = (j, u, q)`, against the verifier's rows and salts there, the registrant's
row bytes (`encPortH` of its value), salt and path; nothing where the source has no rows. -/
noncomputable def valueAt (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (vsrc : ValueSrc L Reg Cm Reg₂ I Vs) (R : Reg) (ω : L.Ω) (t : List (Cm × I.Coin)) (R₂ : Reg₂) (x : VRead Vs) :
    Pair :=
  match vsrc R ω t R₂ x with
  | none => none
  | some (rows, salt) =>
    FlockC3.RegLevels.valueColl ((zs R ω t R₂ x.1).rd.port (.inl x.2)) rows salt x.2.1.val
      (FlockSoundness.Discharge.HiddenExec.encPortH (Vs x.1).c.ports[x.2.2.val]!
        (FlockVBridge.Decode.regVal (zs R ω t R₂ x.1) (.inl x.2)))
      ((zs R ω t R₂ x.1).rd.ws (.inl x.2)) ((zs R ω t R₂ x.1).rd.wpath (.inl x.2))

/-- **A registrant's row is off a value the verifier registers itself, by a collision**: at the history `(R, ω, t)`
and registration `R₂`, some registered read of some session at which `valueAt` is a collision of SHA-512. -/
def ValueColl (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (vsrc : ValueSrc L Reg Cm Reg₂ I Vs) (R : Reg) (ω : L.Ω) (t : List (Cm × I.Coin)) (R₂ : Reg₂) : Prop :=
  ∃ x : VRead Vs, (zs R ω t R₂ x.1).rd.isRead (.inl x.2) ∧ IsColl Refine.H512 (valueAt zs vsrc R ω t R₂ x)

open Classical in
/-- **`valueFinder`'s output at an outcome**: `valueAt` at a read where `ValueColl`'s test passes, if there is one. -/
noncomputable def valuePick (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (vsrc : ValueSrc L Reg Cm Reg₂ I Vs) {β : Type} (o : Reg × L.Ω × List (Cm × I.Coin) × (Reg₂ × β)) : Pair :=
  if h : ValueColl zs vsrc o.1 o.2.1 o.2.2.1 o.2.2.2.1 then valueAt zs vsrc o.1 o.2.1 o.2.2.1 o.2.2.2.1 h.choose
  else none

/-- **The value finder**: one run of the recursive audit against the prover `σ`, output `valueColl`'s pair at a
registered read of the outcome's history and registration where it collides (`ValueColl`), else nothing. -/
noncomputable def valueFinder (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (vsrc : ValueSrc L Reg Cm Reg₂ I Vs) (dg : ByteArray → ByteArray) (N : ℕ)
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N)) : Finder Pair :=
  oneRun _ σ (valuePick zs vsrc)

/-- Level `l`'s `coef` rows at R's opening `o`: opening `j`'s `coefAt (κ[l]) LANES eq(α_l)[j]` in words. -/
def stackedCoefRows (o : Flock.RecCoins.Opening) (l LANES : ℕ) : Array (Array ℕ) :=
  (Array.range (o.sh.sched.levels[l]!).q).map fun j =>
    Flock.RecCoins.words (Flock.RecCoins.coefAt (o.kappa[l]!) LANES (Flock.eqTable (o.lg.conss[l]!).alpha)[j]!)

/-- Level `l`'s `dirs` rows at R's opening `o`: opening `j`'s position's low `H` bits. -/
def stackedDirsRows (o : Flock.RecCoins.Opening) (l H : ℕ) : Array (Array ℕ) :=
  (Array.range (o.sh.sched.levels[l]!).q).map fun j => Flock.RecCoins.dirsRow (o.lg.conss[l]!).pos[j]! H

/-- The levels of R's schedule with `LANES` lanes and `H` siblings below their cap. -/
def stackedLevels (o : Flock.RecCoins.Opening) (LANES H : ℕ) : List ℕ :=
  (List.range o.sh.sched.levels.size).filter fun l =>
    2 ^ (o.sh.sched.levels[l]!).lanes == LANES && (o.sh.sched.levels[l]!).d - (o.sh.sched.levels[l]!).cap == H

/-- Part `j`'s `v` row of R's opening `o`. -/
def openingVRows (o : Flock.RecCoins.Opening) (j : ℕ) : Option (Array (Array ℕ)) :=
  match (do Flock.RecResiduals.groups (← Flock.RecOpening.structureOf o.sh) (Flock.RecOpening.portSizes o.sh)) with
  | .ok gs => match gs[j]? with
    | some (_, _, vs) => some #[Flock.RecCoins.padTo (Flock.RecCoins.words (vs.map (o.v[·]!))) (Flock.RecResiduals.portWordsOf vs.size)]
    | none => none
  | .error _ => none

/-- **The rows and value check 5 registers at port `port` of circuit `c`**, at its candidate-3 inputs `c3`. -/
def valueRows (c3 : Option Flock.RecCoins.C3) (c : Flock.Circuit) (port : String) : Option (Array (Array ℕ) × String) :=
  match c3 with
  | none => none
  | some c3 => match Flock.RecCoins.opening c3 c.unit.name with
    | .error _ => none
    | .ok o => match Flock.RecCoins.stackedParams c.unit.name, Flock.RecOpening.params c.unit.name with
      | some (LANES, H), _ => match stackedLevels o LANES H with
        | [l] => if port = "coef" then some (stackedCoefRows o l LANES, Flock.RecCoins.COEF_VALUE)
          else if port = "dirs" then some (stackedDirsRows o l H, Flock.RecCoins.DIRS_VALUE) else none
        | _ => none
      | none, some (_, _, j) => if port = "v" then (openingVRows o j).map (·, Flock.RecCoins.V_VALUE) else none
      | none, none => none

/-- **The verifier's rows as a value source**, at its candidate-3 inputs `c3 t` at each history. -/
def valueSrcC3 (c3 : List (Cm × I.Coin) → Option Flock.RecCoins.C3) : ValueSrc L Reg Cm Reg₂ I Vs :=
  fun _ _ t _ x =>
    match valueRows (c3 t) (Vs x.1).c ((Vs x.1).c.ports[x.2.2.val]!).name,
      (Vs x.1).c.metaJ.getObjValAs? String "program_sha512" with
    | some (rows, value), .ok prog => some (rows, Flock.RecCoins.verifierSalt prog value)
    | _, _ => none

end FlockSoundness.Discharge.Recursion
