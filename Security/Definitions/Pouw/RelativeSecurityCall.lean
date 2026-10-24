import Definitions.Pouw.RelativeSecurity

/-!
# Relative security at J10-R's call scope: the vocabulary its statements read (trusted)

J10-R (note:compute-accounting/20261019T0845Z-draft-rt-algo-r3, RT-ALGO-11, -16 and -17) draws H1's noise once a call:
one SHA-512 stream per call row (its dither bits `b` and its digit coins) and one per call column (its coin bits `c`
and `U`), and a unit reads its rows' and its columns' slices. A call's units share their noise, so a segment's
coin × coin part is one ⟨M, 32, N⟩ product, not `MN/1024` independent ⟨32, 32, 32⟩ products, and the run a charge
counts is the call's, not each unit's.

Theorem K's design (`DBits`, `dPrefix`) is general in its shape, and at a call's shape it is J10-R's coins. For a call
of `mu × nu` units of `ur` rows and `uc` columns at `B` segments of `L`, `DBits (nu * uc) L np B (mu * ur)` holds call
column `i`'s `U` planes, which every row of the call reads (every row tile), and call row `l`'s dither bits
`b(t, j, l)`, which every column of the call reads (every column tile): both sides are shared. As in a unit's design,
the column coins `c` and the digit coins are not bits of the design; they enter through the play's constants (`pc`,
`vq`, functions of the play), so the count rests on `U` and `b`.

* the call's coins, cell-blocks and a round's coins (`CallBits`, `CallCells`, `CallStage`; overlapped, `OCallSpot`);
* the call's cell-blocks by unit (`UCells`; over a round's calls, `OUCells`) and J10-R's slices (`unitCell`,
  `oUnitCell`, `unitBit`, `unitPc`, `unitVq`): unit `(I, J)`'s column `i` is call column `uc J + i`, its row `l` call
  row `ur I + l`, and it reads those columns' `U` and those rows' `b`;
* an adversary's window run when each stage is one call (`CallRunMap`, `OCallRunMap`): per round and stage, one
  straight-line program over the call's coins, whose word slots are every unit's.
-/

namespace Pouw.RelativeSecurity

section Call

variable {mu nu ur uc L np B : ℕ}

/-- **J10-R's coins of a call** of `mu × nu` units of `ur` rows and `uc` columns, at `B` segments of `L`: Theorem K's
design at the call's shape. Call column `i`'s `U` planes `β_p(i, j)` (the weight's side) are read by every row of the
call, and call row `l`'s dither bits `b(t, j, l)` (the activation's side) by every column. -/
abbrev CallBits (mu nu ur uc L np B : ℕ) : Type := DBits (nu * uc) L np B (mu * ur)

/-- A call's cell-blocks: call column `i`, call row `l` and segment `t`. -/
abbrev CallCells (mu nu ur uc B : ℕ) : Type := DCells (nu * uc) (mu * ur) B

/-- A round's coins with `St` stages, each one call: each stage's call coins, then the check's stage and cell-block. -/
abbrev CallStage (St mu nu ur uc L np B : ℕ) : Type := DStage St (nu * uc) L np B (mu * ur)

/-- A call's cell-blocks by unit: unit `(I, J)` (row tile `I`, column tile `J`) and a cell-block of its own (its
column `i < uc`, its row `l < ur`, segment `t`). -/
abbrev UCells (mu nu ur uc B : ℕ) : Type := (Fin mu × Fin nu) × DCells uc ur B

/-- **A unit's cell-block as the call's** (J10-R's slices): unit `(I, J)`'s column `i` is call column `uc J + i`, its
row `l` call row `ur I + l`, and its segment the call's. -/
def unitCell (u : UCells mu nu ur uc B) : CallCells mu nu ur uc B :=
  (finProdFinEquiv (u.1.2, u.2.1), finProdFinEquiv (u.1.1, u.2.2.1), u.2.2.2)

/-- **Unit `(I, J)`'s coins in the call's** (J10-R's slices): its column `i`'s `U` planes are call column `uc J + i`'s,
and its row `l`'s dither bits are call row `ur I + l`'s. -/
def unitBit (I : Fin mu) (J : Fin nu) : DBits uc L np B ur → CallBits mu nu ur uc L np B :=
  Sum.map (fun a => (finProdFinEquiv (J, a.1), a.2)) fun b => (b.1, b.2.1, finProdFinEquiv (I, b.2.2))

/-- Column tile `J`'s committed weight constants: call columns `uc J + i`'s. -/
def unitPc {Ar : Type} (pc : Fin (nu * uc) → Fin B → Fin L → Ar) (J : Fin nu) : Fin uc → Fin B → Fin L → Ar :=
  fun i => pc (finProdFinEquiv (J, i))

/-- Row tile `I`'s committed activation constants: call rows `ur I + l`'s. -/
def unitVq {Ar : Type} (vq : Fin B → Fin L → Fin (mu * ur) → Ar) (I : Fin mu) : Fin B → Fin L → Fin ur → Ar :=
  fun t j l => vq t j (finProdFinEquiv (I, l))

/-- **The analysis's map from adversaries to their window runs when each stage is one J10-R call**: `RunMap` at the
call's shape. Per round and stage, the run is one straight-line program over the call's coins (`prog x w s`), on the
call's committed constants, and unit `u`'s cell-block's committed word is its slot `word x w s (unitCell u)`. A product
the program forms once is one slot, whichever units' words read it. -/
abbrev CallRunMap (Mc : Machine) (S : Service) (R St mu nu ur uc L np B : ℕ) (Ar : Type) [CommRing Ar] : Type 1 :=
  RunMap Mc S R St (nu * uc) L np B (mu * ur) Ar

end Call

section OverlapCall

variable {St : ℕ} {mu nu B : Fin St → ℕ} {ur uc : ℕ}

/-- A round's coins with overlapping stages, stage `t` a call of `mu t × nu t` units of `ur × uc` at `B t` segments:
the check's cell-block, among every call's, and every call's draw. -/
abbrev OCallSpot (St : ℕ) (mu nu : Fin St → ℕ) (ur uc L np : ℕ) (B : Fin St → ℕ) : Type :=
  OSpot St (fun t => nu t * uc) L np B fun t => mu t * ur

/-- A round's cell-blocks over its calls, by unit: stage `t`, a unit of its call and a cell-block of the unit. -/
abbrev OUCells (St : ℕ) (mu nu : Fin St → ℕ) (ur uc : ℕ) (B : Fin St → ℕ) : Type :=
  Σ t, UCells (mu t) (nu t) ur uc (B t)

/-- A unit's cell-block as its call's, among the round's (`unitCell` at its stage). -/
def oUnitCell (u : OUCells St mu nu ur uc B) : OCells St (fun t => nu t * uc) (fun t => mu t * ur) B :=
  ⟨u.1, unitCell u.2⟩

/-- **The analysis's map from adversaries to their window runs with overlapping stages, each one J10-R call**:
`ORunMap` at the calls' shapes. Per round, the run is one straight-line program over every call's coins
(`prog x w`), and unit `u`'s cell-block's committed word is its slot `word x w (oUnitCell u)`. A product the program
forms once is one slot, whichever units' and calls' words read it. -/
abbrev OCallRunMap (Mc : Machine) (S : Service) (R St : ℕ) (mu nu : Fin St → ℕ) (ur uc L np : ℕ) (B : Fin St → ℕ)
    (Ar : Type) [CommRing Ar] : Type 1 :=
  ORunMap Mc S R St (fun t => nu t * uc) L np B (fun t => mu t * ur) Ar

end OverlapCall

end Pouw.RelativeSecurity
