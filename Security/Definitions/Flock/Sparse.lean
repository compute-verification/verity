import Definitions.Flock.Piop

/-!
# Hidden unit circuits: L4′ with h, and the sparse phase H1–H4

Hidden unit circuits as `Game` programs, on the
pattern of `Piop.lean`. With hidden unit circuits the lincheck's comb is the public part's fold, and the prover sends
the hidden part's fold `h` after `z_partial` and before `r'` (`holoLincheck`, L4′). The sparse phase (`sparsePhase`)
then checks `h` against the registered templates at the rep's coins:

* H1: the claim `Σ_s eq(ζ_c, s)·h[s]`;
* H2: six rounds of degree 2 over the skip's row bits, then `v₂`, and `claim = ŵ(ζ_r)·v₂`;
* H3: from `v₂`, `ν` rounds of degree `d = 2ℓ + 1` over the entries and `S_LOG` of degree `d + 1` over the slots,
  then the planes `u`, and `claim = Ũ(r_s)·Φ(u)`;
* H4: the planes batched at `ζ_p`; with `T > 1` stacked types, the types' values `vt`, the check that the batch is
  `Σ_t w_t·vt_t` with `w_t = Σ_{σ : τ_σ = t} eq(r_s, σ)`, and the registered claim `Σ_t eq(ρ_t, t)·vt_t`.

It returns the registered claim: the value the registered opening reads, at `(r_k, ζ_p, ρ_t)`. A round sends its
coefficients without `c₁ = claim + Σ_{i≥2} c_i` (C1, the sum rule `g(0) + g(1) = claim` in characteristic 2), and each
coordinate of a point is a coin of its own.
-/

namespace FlockSoundness.Model

open FlockSoundness.Game

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K]

/-! ## L4′ -/

/-- What `holoLincheck` hands on: the `ab` claim, and the lincheck's `α`, its column challenges `ts` and the hidden
part's fold `h`, which the sparse phase reads. -/
structure HoloLcOut (F : Type) (m n : ℕ) where
  ab : Claim F m
  alpha : F
  ts : Fin n → F
  h : Fin 64 → F

/-- **The lincheck with hidden units** (§10 with L4′): `lincheck` on the public part `S`, with the prover's `h` after
`z_partial` and the check `Σ_s (folded[s] + h[s])·z_partial[s] = claim`. -/
def holoLincheck (A : Arith F K) (S : Statement) (zc : ZcOut F S.m) :
    Game (Option (HoloLcOut F S.m (S.kLog - 6))) := do
  let α ← draw F                                         -- L1
  let β ← draw F                                         -- L2
  let n := S.kLog - 6
  let (ts, claim) ← lincheckRounds n (α * zc.va + zc.vb + β)   -- L3
  let zPartial ← recv (Fin 64 → F)                       -- L4
  let h ← recv (Fin 64 → F)                              -- L4′: the hidden part's fold
  let r' ← draw F                                        -- L5
  let ρIn : Fin n → F := fun j => zc.ρ ⟨j.val, by have := S.hk; omega⟩
  let folded := foldAll (List.ofFn ts) (2 ^ S.kLog) (comb A S zc.z α β ρIn)
  let w := ∑ s : Fin 64, A.lagS r' s * zPartial s
  let x : Fin (S.m - 6) → F := fun i =>
    if hi : i.val < n then ts ⟨n - 1 - i.val, by omega⟩ else zc.ρ i
  pure (if (∑ s : Fin 64, (folded s + h s) * zPartial s) = claim
    then some ⟨⟨w, A.lagS r', x⟩, α, ts, h⟩ else none)

/-! ## Rounds in coefficient form -/

/-- `n` coins, one draw each. -/
def drawFin : (n : ℕ) → Game (Fin n → F)
  | 0 => pure Fin.elim0
  | n + 1 => do
      let c ← draw F
      let cs ← drawFin n
      pure (Fin.cons c cs)

/-- A round's claim after its coin `r` (C1): `g(r)` for `g = c₀ + c₁X + Σ_i c_{i+2}X^{i+2}`, with
`c₁ = claim + Σ_i c_{i+2}`. -/
def degNext {k : ℕ} (claim : F) (c : F × (Fin k → F)) (r : F) : F :=
  c.1 + (claim + ∑ i, c.2 i) * r + ∑ i, c.2 i * r ^ (i.val + 2)

/-- `n` rounds of degree `k + 1`: per round the prover sends `(c₀, c₂ … c_{k+1})` and the verifier draws the round's
coin. Returns the coins in round order and the last claim. -/
def degRounds (k : ℕ) : (n : ℕ) → F → Game ((Fin n → F) × F)
  | 0, claim => pure (Fin.elim0, claim)
  | n + 1, claim => do
      let c ← recv (F × (Fin k → F))
      let r ← draw F
      let (rs, out) ← degRounds k n (degNext claim c r)
      pure (Fin.cons r rs, out)

/-- The claim after rounds `a … a + n − 1`, on round `j`'s message `msg j` and coin `r j`. -/
def degFold {k : ℕ} (msg : ℕ → F × (Fin k → F)) (r : ℕ → F) (a n : ℕ) (claim : F) : F :=
  (List.range' a n).foldl (fun c j => degNext c (msg j) (r j)) claim

/-! ## The sparse phase -/

/-- `RecSparse`'s public parameters: the entries' bits `ν`, the template's row and column bits `ℓ` (at least the
skip's 6), the slots' bits and the stacked types' bits. -/
structure SparseShape where
  nu : ℕ
  ell : ℕ
  sLog : ℕ
  tLog : ℕ

/-- The planes' bits, `⌈log₂ (2ℓ + 2)⌉`; `P' = 2^pLog` planes. -/
def SparseShape.pLog (sh : SparseShape) : ℕ := Nat.log2 (2 * sh.ell + 1) + 1

/-- The public values the phase reads: the zerocheck's skip coin `z`, the lincheck's `α`, the row and column points
past the skip (`ρ_in`, and the column challenges in bit order) and the slots' layout factors `U_σ`. -/
structure SparsePub (F : Type) (sh : SparseShape) where
  z : F
  alpha : F
  xRest : Fin (sh.ell - 6) → F
  yRest : Fin (sh.ell - 6) → F
  U : Fin (2 ^ sh.sLog) → F

/-- The registered claim: the value the registered opening reads, at the point `(r_k, ζ_p, ρ_t)`. -/
structure RegClaim (F : Type) (sh : SparseShape) where
  value : F
  rk : Fin sh.nu → F
  zetaP : Fin sh.pLog → F
  rho : Fin sh.tLog → F

/-- A vector's entry `i`, zero past its end. -/
def vget {n : ℕ} (u : Fin n → F) (i : ℕ) : F := if h : i < n then u ⟨i, h⟩ else 0

/-- A point's coordinate `j`: the skip's six, then the rest. -/
def cat6 {m : ℕ} (a : Fin 6 → F) (b : Fin m → F) (j : ℕ) : F := if j < 6 then vget a j else vget b (j - 6)

/-- `Φ(q) = (α·q_{2ℓ} + q_{2ℓ+1}) · ∏_{j<ℓ} (1 + x_j + q_j)·(1 + y_j + q_{ℓ+j})`. -/
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

/-- **The sparse phase** (H1–H4) of one rep, on the slots' hidden types `τ` and L4′'s `h`. -/
def sparsePhase (A : Arith F K) (sh : SparseShape) (pub : SparsePub F sh) (τ : Fin (2 ^ sh.sLog) → ℕ)
    (h : Fin 64 → F) : Game (Option (RegClaim F sh)) := do
  let ζc ← drawFin (F := F) 6                                                  -- H1
  let (ζr, c₂) ← degRounds 1 6 (∑ s : Fin 64, eqAt ζc s.val * h s)             -- H2
  let v₂ ← recv F
  let (rk, c₃) ← degRounds (2 * sh.ell) sh.nu v₂                                -- H3: the entries
  let (rs, c₄) ← degRounds (2 * sh.ell + 1) sh.sLog c₃                          -- H3: the slots
  let u ← recv (Fin (2 ^ sh.pLog) → F)
  let ζp ← drawFin (F := F) sh.pLog                                             -- H4
  let vt : Fin (2 ^ sh.tLog) → F ← if sh.tLog = 0 then pure fun _ => 0 else recv (Fin (2 ^ sh.tLog) → F)
  let ρ ← drawFin (F := F) sh.tLog
  pure (sparseVerdict A sh pub τ ζc ζr c₂ v₂ rk rs c₄ u ζp vt ρ)

end FlockSoundness.Model
