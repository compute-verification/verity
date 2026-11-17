module

public import Definitions.Flock.Piop
import Mathlib.Algebra.Order.Module.Field
import Mathlib.Basic.ENNReal.Real

/-! Hidden unit circuits as `Game` programs: the lincheck with `h` (L4′) and the sparse phase H1–H4, which returns
the registered claim. -/

@[expose] public section

namespace FlockSoundness.Model

open FlockSoundness.Game

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K]

structure HoloLcOut (F : Type) (m n : ℕ) where
  ab : Claim F m
  alpha : F
  ts : Fin n → F
  h : Fin 64 → F

/-- **The lincheck with hidden units** (`Flock.Piop` with L4′): `lincheck` on the public part `S`, with the prover's `h` after
`z_partial` and the check `Σ_s (folded[s] + h[s])·z_partial[s] = claim`. -/
def holoLincheck (A : Arith F K) (S : Statement) (zc : ZcOut F S.m) :
    Game (Option (HoloLcOut F S.m (S.kLog - 6))) := do
  let α ← draw F
  let β ← draw F
  let n := S.kLog - 6
  let (ts, claim) ← lincheckRounds n (α * zc.va + zc.vb + β)
  let zPartial ← recv (Fin 64 → F)
  let h ← recv (Fin 64 → F)
  let r' ← draw F
  let ρIn : Fin n → F := fun j => zc.ρ ⟨j.val, by have := S.hk; omega⟩
  let folded := foldAll (List.ofFn ts) (2 ^ S.kLog) (comb A S zc.z α β ρIn)
  let w := ∑ s : Fin 64, A.lagS r' s * zPartial s
  let x : Fin (S.m - 6) → F := fun i =>
    if hi : i.val < n then ts ⟨n - 1 - i.val, by omega⟩ else zc.ρ i
  pure (if (∑ s : Fin 64, (folded s + h s) * zPartial s) = claim
    then some ⟨⟨w, A.lagS r', x⟩, α, ts, h⟩ else none)

def sampleFin : (n : ℕ) → Game (Fin n → F)
  | 0 => pure Fin.elim0
  | n + 1 => do
      let c ← draw F
      let cs ← sampleFin n
      pure (Fin.cons c cs)

/-- A round's claim after its coin `r` (C1): `g(r)` for `g = c₀ + c₁X + Σ_i c_{i+2}X^{i+2}`, with
`c₁ = claim + Σ_i c_{i+2}`. -/
def degNext {k : ℕ} (claim : F) (c : F × (Fin k → F)) (r : F) : F :=
  c.1 + (claim + ∑ i, c.2 i) * r + ∑ i, c.2 i * r ^ (i.val + 2)

/-- `n` rounds of degree `k + 1`: per round the prover sends `(c₀, c₂ … c_{k+1})` and the verifier flips the round's
coin. Returns the coins in round order and the last claim. -/
def degRounds (k : ℕ) : (n : ℕ) → F → Game ((Fin n → F) × F)
  | 0, claim => pure (Fin.elim0, claim)
  | n + 1, claim => do
      let c ← recv (F × (Fin k → F))
      let r ← draw F
      let (rs, out) ← degRounds k n (degNext claim c r)
      pure (Fin.cons r rs, out)

def degFold {k : ℕ} (msg : ℕ → F × (Fin k → F)) (r : ℕ → F) (a n : ℕ) (claim : F) : F :=
  (List.range' a n).foldl (fun c j => degNext c (msg j) (r j)) claim

/-- `RecSparse`'s public parameters: the entries' bits `ν`, the template's row and column bits `ℓ` (at least the
skip's 6), the slots' bits and the stacked types' bits. -/
structure SparseShape where
  nu : ℕ
  ell : ℕ
  sLog : ℕ
  tLog : ℕ

/-- The planes' bits, `⌈log₂ (2ℓ + 2)⌉`; `P' = 2^pLog` planes. -/
def SparseShape.pLog (sh : SparseShape) : ℕ := Nat.log2 (2 * sh.ell + 1) + 1

structure SparsePub (F : Type) (sh : SparseShape) where
  z : F
  alpha : F
  xRest : Fin (sh.ell - 6) → F
  yRest : Fin (sh.ell - 6) → F
  U : Fin (2 ^ sh.sLog) → F

structure RegClaim (F : Type) (sh : SparseShape) where
  value : F
  rk : Fin sh.nu → F
  zetaP : Fin sh.pLog → F
  rho : Fin sh.tLog → F

def vget {n : ℕ} (u : Fin n → F) (i : ℕ) : F := if h : i < n then u ⟨i, h⟩ else 0

def cat6 {m : ℕ} (a : Fin 6 → F) (b : Fin m → F) (j : ℕ) : F := if j < 6 then vget a j else vget b (j - 6)

def sparsePhi (ell : ℕ) (α : F) (x y q : ℕ → F) : F :=
  (α * q (2 * ell) + q (2 * ell + 1)) * ∏ j ∈ Finset.range ell, ((1 + x j + q j) * (1 + y j + q (ell + j)))

/-- The phase's end checks on its messages and coins, and the registered claim: H2's `c₂ = ŵ(ζ_r)·v₂`, H3's
`c₄ = Ũ(r_s)·Φ(u)`, and with `T > 1` H4's `Σ_p eq(ζ_p, p)·u_p = Σ_t w_t·vt_t`. -/
def sparseVerdict (A : Arith F K) (sh : SparseShape) (pub : SparsePub F sh) (τ : Fin (2 ^ sh.sLog) → ℕ)
    (ζc ζr : Fin 6 → F) (c₂ v₂ : F) (rk : Fin sh.nu → F) (rs : Fin sh.sLog → F) (c₄ : F)
    (u : Fin (2 ^ sh.pLog) → F) (ζp : Fin sh.pLog → F) (vt : Fin (2 ^ sh.tLog) → F) (ρ : Fin sh.tLog → F) :
    Option (RegClaim F sh) :=
  let what := ∑ s : Fin 64, eqAt ζr s.val * A.lagS pub.z s
  let ur := ∑ σ : Fin (2 ^ sh.sLog), eqAt rs σ.val * pub.U σ
  let φ := sparsePhi sh.ell pub.alpha (cat6 ζr pub.xRest) (cat6 ζc pub.yRest) (vget u)
  let batch := ∑ q : Fin (2 ^ sh.pLog), eqAt ζp q.val * u q
  let wt : Fin (2 ^ sh.tLog) → F := fun t => ∑ σ : Fin (2 ^ sh.sLog), if τ σ = t.val then eqAt rs σ.val else 0
  if sh.tLog = 0 then
    if c₂ = what * v₂ ∧ c₄ = ur * φ then some ⟨batch, rk, ζp, ρ⟩ else none
  else if c₂ = what * v₂ ∧ c₄ = ur * φ ∧ batch = ∑ t, wt t * vt t then
    some ⟨∑ t, eqAt ρ t.val * vt t, rk, ζp, ρ⟩
  else none

def sparsePhase (A : Arith F K) (sh : SparseShape) (pub : SparsePub F sh) (τ : Fin (2 ^ sh.sLog) → ℕ)
    (h : Fin 64 → F) : Game (Option (RegClaim F sh)) := do
  let ζc ← sampleFin (F := F) 6
  let (ζr, c₂) ← degRounds 1 6 (∑ s : Fin 64, eqAt ζc s.val * h s)
  let v₂ ← recv F
  let (rk, c₃) ← degRounds (2 * sh.ell) sh.nu v₂
  let (rs, c₄) ← degRounds (2 * sh.ell + 1) sh.sLog c₃
  let u ← recv (Fin (2 ^ sh.pLog) → F)
  let ζp ← sampleFin (F := F) sh.pLog
  let vt : Fin (2 ^ sh.tLog) → F ← if sh.tLog = 0 then pure fun _ => 0 else recv (Fin (2 ^ sh.tLog) → F)
  let ρ ← sampleFin (F := F) sh.tLog
  pure (sparseVerdict A sh pub τ ζc ζr c₂ v₂ rk rs c₄ u ζp vt ρ)

end FlockSoundness.Model
