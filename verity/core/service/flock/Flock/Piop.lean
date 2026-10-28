/-
PROTOCOL.md §9–§11: statement binding, zerocheck, lincheck, and the claims handed to the opening.
-/
import Flock.Transcript
import Flock.Proof

namespace Flock

open Transcript

/-- One coordinate of `eqTable`: the table so far for bit 0, then for bit 1 of the new index bit. -/
def eqStep (t : Array F128) (rj : F128) : Array F128 := t.map (· * (rj + 1)) ++ t.map (· * rj)

/-- `t.map (· * r)`, from `r`'s table (`F128.mulBy_mulTable`) once `t` has four times the 512 products the table costs. -/
def mulMap (t : Array F128) (r : F128) : Array F128 :=
  if t.size < 2048 then t.map (· * r) else let tb := F128.mulTable r; t.map (F128.mulBy tb ·)

theorem mulMap_eq (t : Array F128) (r : F128) : mulMap t r = t.map (· * r) := by
  unfold mulMap
  split
  · rfl
  · exact Array.map_congr_left fun v _ => F128.mulBy_mulTable r v

/-- `eqStep` with one product an entry: bit 0's half is bit 1's plus the table (`F128.mul_add_one`). -/
def eqStepFast (t : Array F128) (rj : F128) : Array F128 :=
  let hi := mulMap t rj
  t.mapIdx (fun i x => hi[i]! + x) ++ hi

theorem eqStepFast_eq (t : Array F128) (rj : F128) : eqStepFast t rj = eqStep t rj := by
  have h : t.mapIdx (fun i x => (t.map (· * rj))[i]! + x) = t.map (· * (rj + 1)) := by
    apply Array.ext
    · simp
    · intro i h1 _
      rw [Array.size_mapIdx] at h1
      rw [Array.getElem_mapIdx, Array.getElem_map, getElem!_pos (t.map (· * rj)) i (by simpa using h1),
        Array.getElem_map, F128.mul_add_one]
  show t.mapIdx (fun i x => (mulMap t rj)[i]! + x) ++ mulMap t rj = _
  rw [mulMap_eq, h, eqStep]

/-- Compiled code runs `eqStepFast`; the kernel and every proof see `eqStep`. `@[csimp]` rewrites only code compiled
after it, so it stays above `eqTable`. -/
@[csimp] theorem eqStep_eq_eqStepFast : @eqStep = @eqStepFast :=
  funext fun t => funext fun rj => (eqStepFast_eq t rj).symm

/-- `eq(r, u)` for every `u ∈ {0,1}^|r|`, index bit `j` ↔ `r[j]` (LSB first). -/
def eqTable (r : Array F128) : Array F128 := r.foldl eqStep #[1]

def nodesS : Array F128 := (Array.range 64).map phi8
def nodesL : Array F128 := (Array.range 64).map (fun i => phi8 (64 + i))

/-- `Π_{i=1}^{2^d − 1} φ8(i)`. -/
def den (d : Nat) : F128 := (Array.range (2 ^ d - 1)).foldl (fun acc i => acc * phi8 (i + 1)) 1

def den6Inv : F128 := (den 6).inv
def den7Inv : F128 := (den 7).inv

/-- Lagrange weights on a node set of dimension 6 (a subspace or a coset), with the indicator rule. -/
def lagrange (nodes : Array F128) (denInv : F128) (z : F128) : Array F128 :=
  match nodes.findIdx? (fun n => decide (n = z)) with
  | some i => (Array.range nodes.size).map fun j => if j == i then 1 else 0
  | none =>
    let zn := nodes.foldl (fun acc n => acc * (z + n)) 1
    let c := zn * denInv
    nodes.map fun n => c * (z + n).inv

/-- `L_s(ζ)` over `S = φ8({0…63})`. -/
def skipWeights (z : F128) : Array F128 := lagrange nodesS den6Inv z

/-- The weights of `Pcomb(z)` (§9.2): zero on `S`; the indicator at a node of `Λ`; else
`Z_{S∪Λ}(z) · den_7⁻¹ · (z + φ8(64+i))⁻¹`. -/
def combWeights (z : F128) : Array F128 :=
  match nodesS.findIdx? (fun n => decide (n = z)), nodesL.findIdx? (fun n => decide (n = z)) with
  | some _, _ => (Array.range 64).map fun _ => 0
  | none, some i => (Array.range 64).map fun j => if j == i then 1 else 0
  | none, none =>
    let zall := (nodesS ++ nodesL).foldl (fun acc n => acc * (z + n)) 1
    let c := zall * den7Inv
    (Array.range 64).map fun i => c * (z + nodesL[i]!).inv

def dot (a b : Array F128) : F128 := Id.run do
  let mut acc : F128 := 0
  for i in [0:min a.size b.size] do acc := acc + a[i]! * b[i]!
  return acc

/-- The seven pinned zerocheck coordinates (address bits 6–12), §9.2. -/
def pinned : Array F128 :=
  let frac := fun (y : F128) => y * (y + 1).inv
  let x := F128.monomial 1
  #[phi8 0xF7, phi8 0x53, phi8 0xB5, frac x, frac (x * x), frac (x * x * x * x), frac (x * x * x * x * x * x * x * x)]

structure ZcOut where
  z : F128
  rho : Array F128
  rRest : Array F128
  vA : F128
  vB : F128
  vC : F128
  deriving Inhabited

def check (b : Bool) (why : String) : TM Unit := if b then pure () else throw why

/-- §9.1 then §9.2. -/
def bindAndZerocheck (m : Nat) (digest : ByteArray) (cap : ByteArray) (p : ZerocheckProof) : TM ZcOut := do
  label "flock-r1cs-v0"
  absorbBytes digest
  absorbBytes cap
  label "flock-zerocheck-v0"
  let _rSkip ← sampleVec 6
  let rOuter ← sampleVec (m - 13)
  let r := pinned ++ rOuter
  absorbSlice p.round1ab
  absorbSlice p.round1c
  let z ← sample
  let pc := dot (lagrange nodesL den6Inv z) p.round1c
  check (pc == p.finalC) "zerocheck: CEvalMismatch"
  let comb := (Array.range 64).map fun i => p.round1ab[i]! + p.round1c[i]!
  let pcomb := dot (combWeights z) comb
  let mut claim := pcomb + pc
  let mut rho : Array F128 := #[]
  for i in [0:m - 6] do
    let (g1, ginf) := p.rounds[i]!
    let req := r[i]!
    let g0 := (claim + req * g1) * (1 + req).inv
    absorbF g1
    absorbF ginf
    let rr ← sample
    rho := rho.push rr
    claim := g0 * (1 + rr) + g1 * rr + ginf * rr * (1 + rr)
  check (claim == p.finalA * p.finalB) "zerocheck: SumcheckFinalFailed"
  absorbF p.finalA
  absorbF p.finalB
  return { z, rho, rRest := r, vA := p.finalA, vB := p.finalB, vC := p.finalC }

/-- One lincheck round's column fold at challenge `t` (L3): the top half onto the bottom half. -/
def halve (t : F128) (comb : Array F128) : Array F128 :=
  (Array.range (comb.size / 2)).map fun i => comb[i]! + t * (comb[i + comb.size / 2]! + comb[i]!)

/-- What the lincheck needs from a statement's circuit (§16.1). -/
structure CircuitFold where
  kLog : Nat
  pin : Nat
  /-- `fold α lw ρ_in = α·A_0ᵀe + B_0ᵀe` over the `2^kLog` columns, for `e[s + 64·u] = lw[s] · eq(ρ_in, u)` -/
  fold : F128 → Array F128 → Array F128 → V (Array F128)
  /-- `folded α β lw ρ_in ts`: `fold α lw ρ_in` with `β` added at `pin`, halved at `ts` in turn (`halve`), on the entries
  left, `Σ_u eq(ts reversed, u) · comb[s + 64·u]` at `s`; a statement computes it from its structure where it can -/
  folded : F128 → F128 → Array F128 → Array F128 → Array F128 → V (Array F128)
  deriving Inhabited

structure LcOut where
  w : F128
  rPrime : F128
  rIn : Array F128
  deriving Inhabited

/-- §10. -/
def lincheck (c : CircuitFold) (zc : ZcOut) (p : LincheckProof) : TM LcOut := do
  let kLog := c.kLog
  let rhoIn := zc.rho.extract 0 (kLog - 6)
  label "flock-lincheck-v0"
  let alpha ← sample
  let beta ← sample
  let mut claim := alpha * zc.vA + zc.vB + beta
  let mut ts : Array F128 := #[]
  for j in [0:kLog - 6] do
    let (q1, qinf) := p.rounds[j]!
    absorbF q1
    absorbF qinf
    let t ← sample
    ts := ts.push t
    let e0 := claim + q1
    let c1 := e0 + q1 + qinf
    claim := qinf * t * t + c1 * t + e0
  let comb ← match c.folded alpha beta (skipWeights zc.z) rhoIn ts with
    | .ok v => pure v
    | .error err => throw err
  check (comb.size == 64) "lincheck: the circuit fold has the wrong width"
  absorbSlice p.zPartial
  check (dot comb p.zPartial == claim) "lincheck: ConsistencyFailed (sumcheck-final)"
  let rPrime ← sample
  let w := dot (skipWeights rPrime) p.zPartial
  return { w, rPrime, rIn := ts.reverse }

/-- A claim handed to the opening (§11, §12): value, 64 skip weights, `m − 6` coordinates. -/
structure Claim where
  value : F128
  skip : Array F128
  x : Array F128
  deriving Inhabited

def abcClaims (m kLog : Nat) (zc : ZcOut) (lc : LcOut) : Array Claim :=
  #[{ value := lc.w, skip := skipWeights lc.rPrime, x := lc.rIn ++ zc.rho.extract (kLog - 6) (m - 6) },
    { value := zc.vC, skip := skipWeights zc.z, x := zc.rRest }]

end Flock
