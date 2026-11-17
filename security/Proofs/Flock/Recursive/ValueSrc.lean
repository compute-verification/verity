module

public import Proofs.Flock.Recursive.SaltFinder
public import Proofs.Flock.VBridge.Decode
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt

/-! The value tie's extractor and its finder, apart from the value tie's proofs so that `RecursiveAuditC3At` can read them. -/

@[expose] public section

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

def sib (dom : ByteArray) (leaves : Array ByteArray) (i d : ℕ) : ByteArray := (lev dom leaves d)[sibIdx (i / 2 ^ d)]!

/-- Leaf `i`'s siblings from level `d`, whose nodes are `a`, up `k` levels, each level computed once. -/
def sibsFrom (dom : ByteArray) (i : ℕ) : ℕ → ℕ → Array ByteArray → List ByteArray
  | 0, _, _ => []
  | k + 1, d, a => a[sibIdx (i / 2 ^ d)]! :: sibsFrom dom i k (d + 1) (up dom d a)

/-- **Leaf `i`'s siblings in `FV3.root`'s tree**, lowest first. -/
def sibsV (dom : ByteArray) (leaves : Array ByteArray) (i : ℕ) : List ByteArray :=
  sibsFrom dom i (log2ceil leaves.size) 0 (lev dom leaves 0)

def openPairV (dom : ByteArray) (pos : ℕ) (x : ByteArray) (s : List ByteArray) (y : ByteArray) (t : List ByteArray) :
    ByteArray × ByteArray :=
  let lx := Registered.leafIn dom pos x
  let ly := Registered.leafIn dom pos y
  if Sha512.hash lx = Sha512.hash ly then (lx, ly)
  else Registered.climbPair dom (Sha512.hash lx) (Sha512.hash ly) pos 0 s t

def leavesV (dom : ByteArray) (rows : Array (Array ℕ)) (salt : ℕ → ByteArray) : Array ByteArray :=
  (Array.range rows.size).map fun i =>
    Registered.rowLeaf dom i (Hm96.Default512.treeLeaf (RecCoins.commitRow rows[i]! (salt i)))

/-- Each pair it outputs holds both preimages; nothing where `r` is the verifier's row's bytes. -/
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

variable {n : ℕ} {L : SamplingDesign n} {Reg Cm Reg₂ Reg' : Type} {I : Inner} {S : ℕ} {Vs : Fin S → VStmt}

abbrev VRead (Vs : Fin S → VStmt) : Type := (j : Fin S) × Fin (Vs j).nv × Fin (Vs j).c.ports.size

abbrev ValueSrc (L : SamplingDesign n) (Reg Cm Reg₂ : Type) (I : Inner) (Vs : Fin S → VStmt) : Type :=
  Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → VRead Vs → Option (Array (Array ℕ) × (ℕ → ByteArray))

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

def ValueColl (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (vsrc : ValueSrc L Reg Cm Reg₂ I Vs) (R : Reg) (ω : L.Ω) (t : List (Cm × I.Coin)) (R₂ : Reg₂) : Prop :=
  ∃ x : VRead Vs, (zs R ω t R₂ x.1).rd.isRead (.inl x.2) ∧ IsColl Refine.H512 (valueAt zs vsrc R ω t R₂ x)

open Classical in
noncomputable def valuePick (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (vsrc : ValueSrc L Reg Cm Reg₂ I Vs) {β : Type} (o : Reg × L.Ω × List (Cm × I.Coin) × (Reg₂ × β)) : Pair :=
  if h : ValueColl zs vsrc o.1 o.2.1 o.2.2.1 o.2.2.2.1 then valueAt zs vsrc o.1 o.2.1 o.2.2.1 o.2.2.2.1 h.choose
  else none

noncomputable def valueFinder (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (vsrc : ValueSrc L Reg Cm Reg₂ I Vs) (dg : ByteArray → ByteArray) (N : ℕ)
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N)) : Finder Pair :=
  oneRun _ σ (valuePick zs vsrc)

def stackedCoefRows (o : Flock.RecCoins.RegisteredOpening) (l LANES : ℕ) : Array (Array ℕ) :=
  (Array.range (o.sh.sched.levels[l]!).q).map fun j =>
    Flock.RecCoins.words (Flock.RecCoins.coefAt (o.kappa[l]!) LANES (Flock.eqTable (o.lg.conss[l]!).alpha)[j]!)

def stackedDirsRows (o : Flock.RecCoins.RegisteredOpening) (l H : ℕ) : Array (Array ℕ) :=
  (Array.range (o.sh.sched.levels[l]!).q).map fun j => Flock.RecCoins.dirsRow (o.lg.conss[l]!).pos[j]! H

def stackedLevels (o : Flock.RecCoins.RegisteredOpening) (LANES H : ℕ) : List ℕ :=
  (List.range o.sh.sched.levels.size).filter fun l =>
    2 ^ (o.sh.sched.levels[l]!).lanes == LANES && (o.sh.sched.levels[l]!).d - (o.sh.sched.levels[l]!).cap == H

def openingVRows (o : Flock.RecCoins.RegisteredOpening) (j : ℕ) : Option (Array (Array ℕ)) :=
  match (do Flock.RecResiduals.groups (← Flock.RecOpening.structureOf o.sh) (Flock.RecOpening.portSizes o.sh)) with
  | .ok gs => match gs[j]? with
    | some (_, _, vs) => some #[Flock.RecCoins.padTo (Flock.RecCoins.words (vs.map (o.v[·]!))) (Flock.RecResiduals.portWordsOf vs.size)]
    | none => none
  | .error _ => none

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
      | none, some (_, _, j) => if port = "v" then (openingVRows o j).map (·, Flock.RecCoins.V_VALUE)
        else if port = "out" then some (Flock.RecCoins.zeroRows 1 Flock.RecCoins.OUT_WORDS, Flock.RecCoins.ZERO_VALUE)
        else none
      | none, none => none

def valueSrcC3 (c3 : List (Cm × I.Coin) → Option Flock.RecCoins.C3) : ValueSrc L Reg Cm Reg₂ I Vs :=
  fun _ _ t _ x =>
    match valueRows (c3 t) (Vs x.1).c ((Vs x.1).c.ports[x.2.2.val]!).name,
      (Vs x.1).c.metaJ.getObjValAs? String "program_sha512" with
    | some (rows, value), .ok prog => some (rows, Flock.RecCoins.verifierSalt prog value)
    | _, _ => none

/-- The levels of the inner schedule with `LANES` lanes and `H` siblings below their cap. -/
def innerLevels (sh : Flock.Shape) (LANES H : ℕ) : List ℕ :=
  (List.range sh.sched.levels.size).filter fun l =>
    2 ^ sh.sched.levels[l]!.lanes == LANES && sh.sched.levels[l]!.d - sh.sched.levels[l]!.cap == H

/-- Level `l`'s `coef` rows: rep `i / q`'s opening `i % q`'s `coefAt (κ[l]) LANES eq(α_l)[j]` in words. -/
def openCoef (sh : Flock.Shape) (l LANES : ℕ) (r0 r1 : Flock.RecCoins.Rep) : Array (Array ℕ) :=
  (Array.range (2 * sh.sched.levels[l]!.q)).map fun i =>
    let (rep, j) := (if i < sh.sched.levels[l]!.q then r0 else r1, i % sh.sched.levels[l]!.q)
    Flock.RecCoins.words (Flock.RecCoins.coefAt (rep.kappa[l]!) LANES (Flock.eqTable (rep.info.lg.conss[l]!).alpha)[j]!)

/-- Level `l`'s `dirs` rows: rep `i / q`'s opening `i % q`'s position's low `H` bits. -/
def openDirs (sh : Flock.Shape) (l H : ℕ) (r0 r1 : Flock.RecCoins.Rep) : Array (Array ℕ) :=
  (Array.range (2 * sh.sched.levels[l]!.q)).map fun i =>
    let (rep, j) := (if i < sh.sched.levels[l]!.q then r0 else r1, i % sh.sched.levels[l]!.q)
    Flock.RecCoins.dirsRow (rep.info.lg.conss[l]!).pos[j]! H

/-- `rec-open`'s `coef` and `dirs` rows at the inner session `inn`. -/
def openRows (inn : Flock.RecCoins.Inner) (LANES H : ℕ) : Option (Array (Array ℕ) × Array (Array ℕ)) :=
  match inn.algebra with
  | .error _ => none
  | .ok (sh, extras) => match innerLevels sh LANES H with
    | [l] => match inn.rep sh extras 0, inn.rep sh extras 1 with
      | .ok r0, .ok r1 => some (openCoef sh l LANES r0 r1, openDirs sh l H r0 r1)
      | _, _ => none
    | _ => none

/-- The parts of the inner algebra's v. -/
def residualGroups (sh : Flock.Shape) : Except String (Array (Array ℕ × Array ℕ × Array ℕ)) :=
  do Flock.RecResiduals.groups (← Flock.RecAlgebra.structureOf sh) (Flock.RecResiduals.portSizes sh)

/-- Each rep's `v` elements `vs` (the comb last), padded to the port's words. -/
def residualRowsOf (inn : Flock.RecCoins.Inner) (sh : Flock.Shape) (extras : Array Flock.Claim) (vs : Array ℕ) :
    Flock.V (Array (Array ℕ)) :=
  #[0, 1].mapM fun r => do
    let rep ← inn.rep sh extras r
    let base := rep.v.extract 0 (rep.v.size - 64)
    let comb ← if vs.any (· ≥ base.size) then inn.comb r rep else pure Flock.RecCoins.ZERO_COMB
    let full := base ++ comb
    pure (Flock.RecCoins.padTo (Flock.RecCoins.words (vs.map (full[·]!))) (Flock.RecResiduals.portWordsOf vs.size))

/-- `rec-residuals`' `v` rows of part `j` at the inner session `inn`. -/
def residualRows (inn : Flock.RecCoins.Inner) (j : ℕ) : Option (Array (Array ℕ)) :=
  match inn.algebra with
  | .error _ => none
  | .ok (sh, extras) => match residualGroups sh with
    | .error _ => none
    | .ok gs => match gs[j]? with
      | none => none
      | some (_, _, vs) => match residualRowsOf inn sh extras vs with
        | .ok rows => some rows
        | .error _ => none

def sparseRange (inn : Flock.RecCoins.Inner) (unit : String) : Flock.V Flock.Range :=
  inn.unitRange.mapError (s!"rec-coins: {unit}: " ++ ·)

def sparseRowsOf (inn : Flock.RecCoins.Inner) (c3 : Flock.RecCoins.C3) (unit : String) (sh : Flock.RecSparse.Shape)
    (ash : Flock.Shape) (extras : Array Flock.Claim) (rng : Flock.Range) : Flock.V (Array (Array ℕ)) :=
  #[0, 1].mapM fun r => do
    let rep ← inn.rep ash extras r
    let sc ← Flock.RecCoins.sparseCoins c3 unit r
    let i := rep.info
    let v ← (do
        Flock.RecCoins.sparseV sh sc i.lc.fold.alpha i.zc.z i.lc.fold.rhoIn i.lc.rIn
          (← Flock.RecCoins.layoutFactors sh i.lc.fold.rhoIn i.lc.rIn rng)).mapError (s!"rec-coins: {unit}: rep {r}: " ++ ·)
    pure (Flock.RecCoins.padTo (Flock.RecCoins.words v) (Flock.RecSparse.portWords (Flock.RecSparse.nv sh)))

def sparseRows (inn : Flock.RecCoins.Inner) (c3 : Flock.RecCoins.C3) (unit : String) (sh : Flock.RecSparse.Shape) :
    Option (Array (Array ℕ)) :=
  match inn.algebra, sparseRange inn unit with
  | .ok (ash, extras), .ok rng => match sparseRowsOf inn c3 unit sh ash extras rng with
    | .ok rows => some rows
    | .error _ => none
  | _, _ => none

def innerRows (inn : Flock.RecCoins.Inner) (c3 : Option Flock.RecCoins.C3) (c : Flock.Circuit) (port : String) :
    Option (Array (Array ℕ) × String) :=
  match Flock.RecSparse.params c.unit.name, Flock.RecOpen.params c.unit.name, Flock.RecResiduals.params c.unit.name with
  | some sh, _, _ => match c3 with
    | some c3 => if port = "v" then (sparseRows inn c3 c.unit.name sh).map (·, Flock.RecCoins.SPARSE_V_VALUE)
      else if port = "out" then some (Flock.RecCoins.zeroRows 2 (Flock.RecSparse.outWords sh),
        Flock.RecCoins.SPARSE_ZERO_VALUE) else none
    | none => none
  | none, some (LANES, H), _ => match openRows inn LANES H with
    | some (cr, dr) => if port = "coef" then some (cr, Flock.RecCoins.COEF_VALUE)
      else if port = "dirs" then some (dr, Flock.RecCoins.DIRS_VALUE) else none
    | none => none
  | none, none, some (_, _, _, j) => if port = "v" then (residualRows inn j).map (·, Flock.RecCoins.V_VALUE)
    else if port = "out" then some (Flock.RecCoins.zeroRows 2 Flock.RecCoins.OUT_WORDS, Flock.RecCoins.ZERO_VALUE)
    else none
  | none, none, none => none

def loadRows (src : Option Flock.RecCoins.Source) (c3 : Option Flock.RecCoins.C3) (c : Flock.Circuit) (port : String) :
    Option (Array (Array ℕ) × String) :=
  match src with
  | none => none
  | some f => match f.load c.unit.name with
    | .error _ => none
    | .ok inn => innerRows inn c3 c port

def valueRowsAll (src : Option Flock.RecCoins.Source) (c3 : Option Flock.RecCoins.C3) (c : Flock.Circuit)
    (port : String) : Option (Array (Array ℕ) × String) :=
  match Flock.RecCoins.stackedParams c.unit.name, Flock.RecOpening.params c.unit.name with
  | none, none => loadRows src c3 c port
  | _, _ => valueRows c3 c port

def valueSrcAll (src : List (Cm × I.Coin) → Option Flock.RecCoins.Source)
    (c3 : List (Cm × I.Coin) → Option Flock.RecCoins.C3) : ValueSrc L Reg Cm Reg₂ I Vs :=
  fun _ _ t _ x =>
    match valueRowsAll (src t) (c3 t) (Vs x.1).c ((Vs x.1).c.ports[x.2.2.val]!).name,
      (Vs x.1).c.metaJ.getObjValAs? String "program_sha512" with
    | some (rows, value), .ok prog => some (rows,
        if value = Flock.RecCoins.ZERO_VALUE ∨ value = Flock.RecCoins.SPARSE_ZERO_VALUE then Flock.RecCoins.zeroSalt value
        else Flock.RecCoins.verifierSalt prog value)
    | _, _ => none

end FlockSoundness.Discharge.Recursion
