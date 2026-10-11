import Proofs.Flock.Soundness.Model.Sparse
import Proofs.Flock.Soundness.Model.Session

/-!
# The holographic table: two reps with hidden units, then the registered opening

F4 candidate 3's session (note:proofs/20261010T0250Z-draft-f4-bitsumcheck §1.3–§1.4, row M1 of its §4), on the pattern
of `Session.lean`. A rep with hidden unit circuits (`holoRep`) runs the zerocheck, the lincheck on the public part with
the prover's `h` (`holoLincheck`, L4′), the sparse phase on `h` (`sparsePhase`, H1–H4), and then the opening and
Ligerito on the committed level-0 table, as `rep` does; it hands on the sparse phase's registered claim. The session
(`holoTable`) runs two such reps in any interleaving, then opens both registered claims together, once, on the registered
word `Creg`: level 0 of the registered schedule, committed at registration rather than in the session (§1.4).

The registered table stacks the types' tables `R(t, p, k)` as bit `k + 2^ν·(p + 2^pLog·t)`, so the registered claim's
point is `(r_k, ζ_p, ρ_t)` in that order (`RegClaim.toClaim`): skip weights `eq(point[0…6], ·)`, `x = point[6 …]`.
-/

namespace FlockSoundness.Model

open FlockSoundness.Game FlockSoundness.Accounting

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-- The registered table's variables: the entries', the planes' and the types'. -/
def SparseShape.mReg (sh : SparseShape) : ℕ := sh.nu + sh.pLog + sh.tLog

/-- Coordinate `j` of the registered claim's point `(r_k, ζ_p, ρ_t)`. -/
def RegClaim.coord {sh : SparseShape} (c : RegClaim F sh) (j : ℕ) : F :=
  if j < sh.nu then vget c.rk j
  else if j < sh.nu + sh.pLog then vget c.zetaP (j - sh.nu)
  else vget c.rho (j - sh.nu - sh.pLog)

/-- The registered claim as a claim on the registered table (`rec_sparse.registered_claim`): skip weights
`eq(point[0…6], ·)`, `x = point[6 …]`. -/
def RegClaim.toClaim {sh : SparseShape} (c : RegClaim F sh) : Claim F sh.mReg :=
  ⟨c.value, fun s => eqAt (fun j : Fin 6 => c.coord j.val) s.val, fun i => c.coord (i.val + 6)⟩

/-- The sparse phase's public values in a rep (`rec_sparse.Coins`): the zerocheck's skip coin, the lincheck's `α`, the
row point past the skip (`ρ_in`) and the column point past the skip in bit order (the column challenges reversed), and
slot `σ`'s layout factor `U_σ = Σ_{u ∈ units σ} eq(x_hi, u)·eq(y_hi, u)`, where `x_hi`, `y_hi` are the points'
coordinates past the template's `ℓ`. -/
def holoPub (S : Statement) (sh : SparseShape) (units : Fin (2 ^ sh.sLog) → Finset ℕ) (zc : ZcOut F S.m)
    (lc : HoloLcOut F S.m (S.kLog - 6)) : SparsePub F sh :=
  let x : ℕ → F := fun j => if h : j < S.kLog - 6 then zc.ρ ⟨j, by have := S.hk; omega⟩ else 0
  let y : ℕ → F := fun j => if h : j < S.kLog - 6 then lc.ts ⟨S.kLog - 6 - 1 - j, by omega⟩ else 0
  ⟨zc.z, lc.alpha, fun j => x j.val, fun j => y j.val, fun σ => ∑ u ∈ units σ,
    eqAt (fun i : Fin (S.kLog - sh.ell) => x (sh.ell - 6 + i.val)) u *
      eqAt (fun i : Fin (S.kLog - sh.ell) => y (sh.ell - 6 + i.val)) u⟩

/-- **One rep with hidden units** on the committed level-0 table `C₀`: the zerocheck, the lincheck on the public part
with L4′'s `h`, the sparse phase on `h` at the slots' types `τ`, then the opening of `ab`, `c` and `extra` and Ligerito,
as in `rep`. Returns the sparse phase's registered claim when every check passes. -/
def holoRep (A : Arith F K) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (C₀ : Oracle F)
    (extra : List (Claim F S.m)) (sh : SparseShape) (units : Fin (2 ^ sh.sLog) → Finset ℕ)
    (τ : Fin (2 ^ sh.sLog) → ℕ) : Game (Option (RegClaim F sh)) := do
  match ← zerocheck A S.m with
  | none => pure none
  | some zc =>
    match ← holoLincheck A S zc with
    | none => pure none
    | some lc =>
      match ← sparsePhase A sh (holoPub S sh units zc lc) τ lc.h with
      | none => pure none
      | some reg =>
        match ← opening A (by omega) (lc.ab :: cClaim A zc :: extra) with
        | none => pure none
        | some op => do
          let ok ← ligerito A sch op.T op.b C₀
          pure (if ok then some reg else none)

/-- **One holographic table's session** (§1.4): the prover commits the level-0 table `C₀`, the verifier draws the link
points, and the prover runs two reps with hidden units in any interleaving. If both hand on a registered claim, the two
are opened together, once, and Ligerito at the registered schedule `rsch` checks them against the registered word
`Creg`. A registered table of fewer than the opening's 7 variables is refused. Returns whether all of it accepted, and
the committed table. -/
def holoTable (A : Arith F K) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (sh : SparseShape) (units : Fin (2 ^ sh.sLog) → Finset ℕ) (τ : Fin (2 ^ sh.sLog) → ℕ) (rsch : Schedule)
    (Creg : Oracle F) : Game (Bool × Oracle F) := do
  let C₀ ← recv (Oracle F)                               -- Commit: the level-0 table (root_B)
  let pts ← draw (Fin 2 → Fin mPts → F)                  -- the link points
  let extra := extraClaims S ptLocal mPts pts
  let regs ← interleave (holoRep A S sch hm C₀ extra sh units τ) (holoRep A S sch hm C₀ extra sh units τ)
  match regs with
  | (some c₁, some c₂) =>
    if hreg : 7 ≤ sh.mReg then
      match ← opening A hreg [c₁.toClaim, c₂.toClaim] with
      | none => pure (false, C₀)
      | some op => do
        let ok ← ligerito A rsch op.T op.b Creg
        pure (ok, C₀)
    else pure (false, C₀)
  | _ => pure (false, C₀)

end FlockSoundness.Model
