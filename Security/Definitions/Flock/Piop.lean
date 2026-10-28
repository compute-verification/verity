import Definitions.Flock.Statement
import Definitions.Flock.Game

/-!
# One rep before Ligerito: zerocheck, lincheck, the claims and the opening

PROTOCOL.md §9.2 (Z1–Z7), §10 (L1–L5), §11 and §12 (O1–O6), as `Game` programs: a `recv` is a
prover message the spec absorbs, a `draw` is a round's live coins. Messages that the spec reads
from the proof but that are forced (`final_c_eval` must equal `Pc(z)`, A4; `matrix_evals` is
empty, A7) are not messages here. A program returns `none` where the spec rejects.

The Merkle cap, the statement digest and the labels that round 0 absorbs (§9.1) fix no
prover choice in this model: the level-0 table is committed before the rep (the session's
`Commit`, §16), and a rep's transcript binds it.
-/

namespace FlockSoundness.Model

open FlockSoundness.Game

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K]

/-! ## Zerocheck (§9.2) -/

/-- The zerocheck's outputs (§9.2): the skip point `z`, the residual challenges `ρ` (one per
address bit 6 … m−1), the zerocheck vector `r` over the same bits, and the claimed values. -/
structure ZcOut (F : Type) (m : ℕ) where
  z : F
  ρ : Fin (m - 6) → F
  r : Fin (m - 6) → F
  va : F
  vb : F
  vc : F

/-- One residual round (Z6): the prover sends `(G(1), G(∞))` of the eq-stripped quadratic `G`;
`G(0)` is derived with `inv(0) = 0` (A5), and the running claim becomes `G(ρ)`. -/
def residualStep (req claim g₁ gInf ρ : F) : F :=
  let g₀ := (claim + req * g₁) * (1 + req)⁻¹
  g₀ * (1 + ρ) + g₁ * ρ + gInf * ρ * (1 + ρ)

/-- The residual rounds (Z6), one per entry of `req`. -/
def residualRounds : (n : ℕ) → (Fin n → F) → F → Game ((Fin n → F) × F)
  | 0, _, claim => pure (Fin.elim0, claim)
  | n + 1, req, claim => do
      let g ← recv (F × F)
      let ρ ← draw F
      let (ρs, c) ← residualRounds n (Fin.tail req) (residualStep (req 0) claim g.1 g.2 ρ)
      pure (Fin.cons ρ ρs, c)

/-- The zerocheck vector over address bits 6 … m−1: the seven pinned coordinates, then `r[13…]`. -/
def zcVector (A : Arith F K) {m : ℕ} (rs : Fin (m - 13) → F) : Fin (m - 6) → F :=
  fun i => if h : i.val < 7 then A.pinned ⟨i.val, h⟩ else rs ⟨i.val - 7, by omega⟩

/-- **Zerocheck** (§9.2, Z1–Z7), for `m ≥ 13`. -/
def zerocheck (A : Arith F K) (m : ℕ) : Game (Option (ZcOut F m)) := do
  let _rSkip ← draw (Fin 6 → F)                          -- Z2: squeezed, never read (A1)
  let rs ← draw (Fin (m - 13) → F)                       -- Z3
  let r := zcVector A rs
  let msg ← recv ((Fin 64 → F) × (Fin 64 → F))           -- Z4: round1_ab, round1_c
  let z ← draw F                                         -- Z5
  let pc := ∑ i, A.lagΛ z i * msg.2 i                    -- Pc(z) = final_c_eval (A4)
  let claim₀ := ∑ i, A.combW z i * (msg.1 i + msg.2 i) + pc   -- Pab(z), zero on S by construction (A3)
  let (ρ, claim) ← residualRounds (m - 6) r claim₀        -- Z6
  let v ← recv (F × F)                                   -- Z7: final_a_eval, final_b_eval
  pure (if claim = v.1 * v.2 then some ⟨z, ρ, r, v.1, v.2, pc⟩ else none)

/-! ## Lincheck (§10) -/

/-- The folded column vector: fold the top bit of a vector of length `2h` at `t`. -/
def foldTop (t : F) (h : ℕ) (v : ℕ → F) : ℕ → F := fun i => v i + t * (v (i + h) + v i)

/-- The lincheck rounds (L3): `(q(1), q(∞))` per round, top column bit first; returns the
challenges in round order and the final claim. -/
def lincheckRounds : (n : ℕ) → F → Game ((Fin n → F) × F)
  | 0, claim => pure (Fin.elim0, claim)
  | n + 1, claim => do
      let q ← recv (F × F)
      let t ← draw F
      let e₀ := claim + q.1
      let c₁ := e₀ + q.1 + q.2
      let (ts, c) ← lincheckRounds n (q.2 * t ^ 2 + c₁ * t + e₀)
      pure (Fin.cons t ts, c)

/-- `comb[c] = Σ_i e[i]·(α·A₀[i,c] + B₀[i,c])` with `e[s + 64u] = L_s(z)·eq(ρ_in, u)`, and
`comb[pin] += β` (§10). -/
def comb (A : Arith F K) (S : Statement) (z α β : F) (ρIn : Fin (S.kLog - 6) → F) : ℕ → F :=
  fun c =>
    if hc : c < 2 ^ S.kLog then
      (∑ i : Fin (2 ^ S.kLog),
          A.lagS z ⟨i.val % 64, Nat.mod_lt _ (by norm_num)⟩ * eqAt ρIn (i.val / 64)
            * (α * ((S.A₀ i ⟨c, hc⟩).val : F) + ((S.B₀ i ⟨c, hc⟩).val : F)))
        + if c = S.pin.val then β else 0
    else 0

/-- Fold `v` by the round challenges, top bit first (round `j` halves the length). -/
def foldAll (t : List F) (len : ℕ) (v : ℕ → F) : ℕ → F :=
  match t with
  | [] => v
  | tj :: rest => foldAll rest (len / 2) (foldTop tj (len / 2) v)

/-- **Lincheck** (§10, L1–L5): reduces `v_a`, `v_b` to the `ab` claim on the witness. -/
def lincheck (A : Arith F K) (S : Statement) (zc : ZcOut F S.m) : Game (Option (Claim F S.m)) := do
  let α ← draw F                                         -- L1
  let β ← draw F                                         -- L2
  let n := S.kLog - 6
  let (ts, claim) ← lincheckRounds n (α * zc.va + zc.vb + β)   -- L3
  let zPartial ← recv (Fin 64 → F)                       -- L4
  let r' ← draw F                                        -- L5
  let ρIn : Fin n → F := fun j => zc.ρ ⟨j.val, by have := S.hk; omega⟩
  let folded := foldAll (List.ofFn ts) (2 ^ S.kLog) (comb A S zc.z α β ρIn)
  let w := ∑ s : Fin 64, A.lagS r' s * zPartial s
  -- the ab claim: skip `L_s(r')`, x = r_in' ‖ ρ_out, r_in' = the challenges reversed
  let x : Fin (S.m - 6) → F := fun i =>
    if h : i.val < n then ts ⟨n - 1 - i.val, by omega⟩ else zc.ρ i
  pure (if (∑ s : Fin 64, folded s * zPartial s) = claim
    then some ⟨w, A.lagS r', x⟩ else none)

/-- The `c` claim (§11): value `Pc(z)`, skip `L_s(z)`, x = `r[6 … m−1]`. -/
def cClaim (A : Arith F K) {m : ℕ} (zc : ZcOut F m) : Claim F m := ⟨zc.vc, A.lagS zc.z, zc.r⟩

/-! ## Opening: per-claim ring switching and batching (§12) -/

/-- The weight of a claim on the 128 packed positions: `w[v] = skip[v mod 64]·eq(x[0], ⌊v/64⌋)`
(§12.1, paper B.3). -/
def packedWeight {m : ℕ} (c : Claim F m) (hm : 7 ≤ m) (v : Fin 128) : F :=
  c.skip ⟨v.val % 64, Nat.mod_lt _ (by norm_num)⟩ *
    (if v.val / 64 = 1 then c.x ⟨0, by omega⟩ else 1 + c.x ⟨0, by omega⟩)

/-- The recombined value `β_k = Σ_u eq(r'', u)·t[u]`, `t[u] = Σ_v bit_u(s[v])·x^v` (§12.3). -/
def recombine (A : Arith F K) (s : Fin 128 → F) (r'' : Fin 7 → F) : F :=
  ∑ u : Fin 128, eqAt r'' u.val * A.pack fun v => A.unpack (s v) u

/-- `Φ(a) = Σ_u bit_u(a)·eq(r'', u)`: the F₂-linear map with `Φ(x^u) = eq(r'', u)` (§12.4). -/
def phi (A : Arith F K) (r'' : Fin 7 → F) (a : F) : F :=
  ∑ u : Fin 128, ((A.unpack a u).val : F) * eqAt r'' u.val

/-- The opening's output: the batched value `T` and the batched basis `b` on the packed
hypercube (`y < 2^(m−7)`), which Ligerito must match: `Σ_y f(y)·b(y) = T` (§12.5). -/
structure OpenOut (F : Type) where
  T : F
  b : ℕ → F

/-- **Opening** (§12, O1–O6): each claim is ring-switched with its own seven coins, then all are
batched with one vector of independent coins. -/
def opening (A : Arith F K) {m : ℕ} (hm : 7 ≤ m) (claims : List (Claim F m)) :
    Game (Option (OpenOut F)) := do
  -- O2–O5 for every claim: the prover sends `s_k`, the verifier checks it and draws `r''_k`
  let switched ← claims.mapM fun c => do
    let s ← recv (Fin 128 → F)                           -- O3
    let r'' ← draw (Fin 7 → F)                           -- O5
    pure (c, s, r'')
  let γ ← draw (Fin claims.length → F)                   -- O6
  let ok := switched.all fun (c, s, _) => decide ((∑ v, packedWeight c hm v * s v) = c.value)
  let T := ∑ k : Fin claims.length, γ k *
    recombine A (switched.getD k.val (claims.getD k.val ⟨0, 0, 0⟩, 0, 0)).2.1
      (switched.getD k.val (claims.getD k.val ⟨0, 0, 0⟩, 0, 0)).2.2
  let b : ℕ → F := fun y => ∑ k : Fin claims.length, γ k *
    let e := switched.getD k.val (claims.getD k.val ⟨0, 0, 0⟩, 0, 0)
    phi A e.2.2 (eqAt (fun j : Fin (m - 7) => e.1.x ⟨j.val + 1, by omega⟩) y)
  pure (if ok then some ⟨T, b⟩ else none)

end FlockSoundness.Model
