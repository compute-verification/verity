import Proofs.Flock.Soundness.Model.Sparse
import Proofs.Flock.Soundness.Model.Session

/-!
# The holographic table: two reps with hidden units, then the registered opening

F4 candidate 3's session (note:proofs/20261010T0250Z-draft-f4-bitsumcheck §1.3–§1.4, row M1 of its §4), on the pattern
of `Session.lean`, in the order the live session records it (`rec_live.schedule_for`, `rec_session.Relay`). A rep with
hidden unit circuits (`holoRep`) runs the zerocheck, the lincheck on the public part with the prover's `h`
(`holoLincheck`, L4′), and then the opening and Ligerito on the committed level-0 table, as `rep` does; it hands on its
zerocheck's and lincheck's outputs, which its sparse phase reads. The session (`holoTable`) runs two such reps in any
interleaving; then each rep's sparse phase on its `h` (`sparsePhase`, H1–H4), rep 0's and then rep 1's, each a child
stream opened at its rep's end; then it opens both registered claims together, once, on the registered word `Creg`:
level 0 of the registered schedule, committed at registration rather than in the session (§1.4).

The sparse phase reads only its rep's zerocheck and lincheck outputs and `h`, all fixed before any of its coins, so the
order is the record's rather than a reordering of the draft's (sparse phase before the rep's opening).

The registered table stacks the types' tables `R(t, p, k)` as bit `k + 2^ν·(p + 2^pLog·t)`, so the registered claim's
point is `(r_k, ζ_p, ρ_t)` in that order (`RegClaim.toClaim`): skip weights `eq(point[0…6], ·)`, `x = point[6 …]`.
-/

namespace FlockSoundness.Model

open FlockSoundness.Game FlockSoundness.Accounting

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-- The fewest variables the registered opening opens a registered table at (`RecVStar.PAD_TO`). -/
def SparseShape.padTo : ℕ := 22

/-- The registered table's variables: the entries', the planes' and the types', padded to `padTo` as the verifier
opens it (`RecVStar.tableM`); the padding's cells are zero, and a claim's point is zero at its coordinates
(`RegClaim.coord`), so it reads only the cells. -/
def SparseShape.mReg (sh : SparseShape) : ℕ := max SparseShape.padTo (sh.nu + sh.pLog + sh.tLog)

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
with L4′'s `h`, then the opening of `ab`, `c` and `extra` and Ligerito, as in `rep`. Returns the zerocheck's and the
lincheck's outputs, which the rep's sparse phase reads, when every check passes. -/
def holoRep (A : Arith F K) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (C₀ : Oracle F)
    (extra : List (Claim F S.m)) : Game (Option (ZcOut F S.m × HoloLcOut F S.m (S.kLog - 6))) := do
  match ← zerocheck A S.m with
  | none => pure none
  | some zc =>
    match ← holoLincheck A S zc with
    | none => pure none
    | some lc =>
      match ← opening A (by omega) (lc.ab :: cClaim A zc :: extra) with
      | none => pure none
      | some op => do
        let ok ← ligerito A sch op.T op.b C₀
        pure (if ok then some (zc, lc) else none)

/-- **The reps' sparse phases and the registered opening**, after both reps: rep 0's sparse phase on its `h`, then rep
1's, at the slots' types `τ`; if both hand on a registered claim, the two are opened together, once, and `fin` runs on
the opening's output (Ligerito on the registered word). A registered table of fewer than the opening's 7 variables is
refused. -/
def regPhase {α : Type} (A : Arith F K) (S : Statement) (sh : SparseShape) (units : Fin (2 ^ sh.sLog) → Finset ℕ)
    (τ : Fin (2 ^ sh.sLog) → ℕ) (no : α) (fin : OpenOut F → Game α)
    (reps : Option (ZcOut F S.m × HoloLcOut F S.m (S.kLog - 6)) ×
      Option (ZcOut F S.m × HoloLcOut F S.m (S.kLog - 6))) : Game α :=
  match reps with
  | (some (zc₁, lc₁), some (zc₂, lc₂)) => do
    match ← sparsePhase A sh (holoPub S sh units zc₁ lc₁) τ lc₁.h with
    | none => pure no
    | some c₁ =>
      match ← sparsePhase A sh (holoPub S sh units zc₂ lc₂) τ lc₂.h with
      | none => pure no
      | some c₂ =>
        if hreg : 7 ≤ sh.mReg then
          match ← opening A hreg [c₁.toClaim, c₂.toClaim] with
          | none => pure no
          | some op => fin op
        else pure no
  | _ => pure no

/-- **One holographic table's session** (§1.4), in the recorded order: the prover commits the level-0 table `C₀`, the
verifier draws the link points, and the prover runs two reps with hidden units in any interleaving; then the reps'
sparse phases, rep 0's and then rep 1's, and the registered opening (`regPhase`), which Ligerito at the registered
schedule `rsch` checks against the registered word `Creg`. Returns whether all of it accepted, and the committed
table. -/
def holoTable (A : Arith F K) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (sh : SparseShape) (units : Fin (2 ^ sh.sLog) → Finset ℕ) (τ : Fin (2 ^ sh.sLog) → ℕ) (rsch : Schedule)
    (Creg : Oracle F) : Game (Bool × Oracle F) := do
  let C₀ ← recv (Oracle F)                               -- Commit: the level-0 table (root_B)
  let pts ← draw (Fin 2 → Fin mPts → F)                  -- the link points
  let extra := extraClaims S ptLocal mPts pts
  let reps ← interleave (holoRep A S sch hm C₀ extra) (holoRep A S sch hm C₀ extra)
  let ok ← regPhase A S sh units τ false (fun op => ligerito A rsch op.T op.b Creg) reps
  pure (ok, C₀)

end FlockSoundness.Model
