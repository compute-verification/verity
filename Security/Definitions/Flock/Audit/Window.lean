import Definitions.Flock.Audit.Law

/-!
# A window's draw and its members' sessions

A sampling window draws its units once, by one law over the window's `N` units, after the prover registers. Each member
of the window owns the units `[base, base + size)` and runs its own session, over its own units `[0, size)`, on the
member's share of the window's draw.

* `Member.range`: the member's units in the window.
* `Member.share D`: the member's share of the window's draw `D`, in its own indices (`v` for each drawn `base + v`).
* `Member.unshift S`: the member's own units `S`, as window units.
* `Member.localUnits`: the share as the service computes it from the draw file's unit list,
  `sorted(u - base for u in units if base <= u < base + n)` (`benchmarks/one_stage/a5.py`).
* `Member.sessionFile W`: the member's session draw file for the window draw file `W`, as the service writes it: none
  when its share is empty, else a `subset` draw of its `|local|` units over its `size`.
* `audit L Reg body`: the window's audit, the prover registers, the window draws by `L`, and `body` runs the window's
  sessions on the draw (the same game as the session audit `auditLive`, over any output).
* `DrawFile`, `StratumFile`: a draw file (`draw.json`) as `json.loads` reads it; `DrawFile.stratumOf` the stratum its
  strata put a unit in.
-/

namespace FlockSoundness.Audit.Window

open Game

/-- A member of a window: the window's units `[base, base + size)`, its own `[0, size)`. -/
structure Member where
  base : ℕ
  size : ℕ
deriving DecidableEq

namespace Member

variable {N : ℕ} (mb : Member)

/-- The member's units in the window. -/
def range : Finset (Fin N) := Finset.univ.filter fun u => mb.base ≤ u.val ∧ u.val < mb.base + mb.size

/-- The member's share of the window's draw `D`, in its own indices: `v` for each drawn unit `base + v`. -/
def share (D : Finset (Fin N)) : Finset (Fin mb.size) :=
  Finset.univ.filter fun v => ∃ u ∈ D, u.val = mb.base + v.val

/-- The member's own units `S`, as window units: `base + v` for each `v ∈ S`. -/
def unshift (S : Finset (Fin mb.size)) : Finset (Fin N) :=
  Finset.univ.filter fun u => ∃ v ∈ S, u.val = mb.base + v.val

/-- The member's local units of a window draw's unit list, as the service computes them:
`sorted(u - base for u in units if base <= u < base + n)`. -/
def localUnits (units : List ℕ) : List ℕ :=
  ((units.filter fun u => decide (mb.base ≤ u) && decide (u < mb.base + mb.size)).map (· - mb.base)).insertionSort
    (· ≤ ·)

end Member

/-- **A window's audit**: the prover registers `R`, the window draws `D` by `L` (a live coin, after the registration),
and `body D R` runs the window's sessions on the draw. Output: the registration, the draw and the sessions' output. It
is the session audit `auditLive`'s game, over any output. -/
def audit {N : ℕ} {α : Type} (L : Law N) (Reg : Type) (body : Finset (Fin N) → Reg → Game α) :
    Game (Reg × Finset (Fin N) × α) :=
  .send Reg fun R => .coin L.Ω fun ω => (body (L.draw ω) R).map fun a => (R, L.draw ω, a)

/-- One stratum of a draw file: its template, its unit ranges `[a, b)`, and how many of its units the law draws. -/
structure StratumFile where
  template : String
  units : List (ℕ × ℕ)
  k : ℕ
deriving DecidableEq

/-- A draw file (`draw.json`) as `json.loads` reads it: its law, population and `k`, its strata (none for a `subset`
draw) and its drawn units. -/
structure DrawFile where
  law : String
  population : ℕ
  k : ℕ
  strata : List StratumFile
  units : List ℕ
deriving DecidableEq

/-- The stratum the draw file's strata put unit `u` in: the index of the first whose ranges `[a, b)` hold it. -/
def DrawFile.stratumOf (W : DrawFile) (u : ℕ) : ℕ :=
  W.strata.findIdx fun st => st.units.any fun r => decide (r.1 ≤ u) && decide (u < r.2)

/-- Member `mb`'s session draw file for the window draw file `W`, as the service writes it: none when its share is
empty (no session runs), else `{"law": "subset", "population": size, "k": |local|, "units": local}`. -/
def Member.sessionFile (mb : Member) (W : DrawFile) : Option DrawFile :=
  if mb.localUnits W.units = [] then none
  else some ⟨"subset", mb.size, (mb.localUnits W.units).length, [], mb.localUnits W.units⟩

end FlockSoundness.Audit.Window
