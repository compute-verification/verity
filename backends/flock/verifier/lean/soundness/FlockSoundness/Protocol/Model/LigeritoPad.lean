import FlockSoundness.Model.Session

/-!
# Ligerito with M1's padded level 0, in the clear

M1 (lane `flock-zk`, `backends/flock/live/PROTOCOL.md` §7) changes level 0 of each rep `r`:

* **the committed lanes** are `2^k₀` witness lanes and `e` extra lanes per rep, rep `r`'s at
  `2^k₀ + e·r + i`. Every lane is a codeword of `RS[2^d, 2^cols + t]`: its message plus `c₀` times
  its `t` padding coefficients at `[0, t)`, and the padding at `[2^cols, 2^cols + t)`, in the novel
  basis at `d` bits;
* **after the lane rounds**, which send no message after their last coin (`foldRoundsNL`), the
  prover sends `e^r_i`, one value per extra lane, the verifier draws `ρ^r_1 … ρ^r_e`, and the prover
  sends `ybar^r`, `t` values over `K`. The claim restarts at `s + Σ_i ρ_i·e_i`, and the code-switch
  message and level 1 follow as before. An honest `e_i` is extra lane `i`'s value against the running
  claim's weight: `b(r̄, ·)` plus level 0's out-of-domain claim `β₀·eq(z₀, (r̄, ·))`;
* **the level-0 row check** at rep `r`'s queries adds the rep's extra lanes, weighted by `ρ`, and
  the padding correction `Σ_j ybar_j·(c₀·X̂_j + X̂_{2^cols + j})` to the folded witness lanes
  (`padColumn`). The induced basis is unchanged, so the final check is.

**The model is M1 in the clear.** M1 masks the lane rounds' messages and `e^r`, and its inner proof
enforces the restart `T' = s + Σ_i ρ_i·e_i`. Here the verifier computes the restart from the
values it is sent. That the masked verifier accepts only if these relations hold is the inner
proof's soundness, a separate statement.
-/

namespace FlockSoundness.Model

open FlockSoundness.Game FlockSoundness.Accounting

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-- `e` coins over `K`, one after the other. -/
def drawKs (A : Arith F K) : (e : ℕ) → Game (Fin e → K)
  | 0 => pure Fin.elim0
  | e + 1 => do
      let r ← drawK A
      let rs ← drawKs A e
      pure (Fin.cons r rs)

/-- The padding correction at position `p`: `Σ_j ybar_j·(c₀·X̂_j + X̂_{2^cols + j})(ω_p)`. -/
def padCorr (A : Arith F K) (d cols t : ℕ) (c₀ : F) (ybar : Fin t → K) (p : ℕ) : K :=
  ∑ j, ybar j * A.embed (c₀ * A.Xhat j.val d (A.omega p) + A.Xhat (2 ^ cols + j.val) d (A.omega p))

/-- **The padded level-0 column** at position `p`, for the rep whose extra lanes are `xl i`: the
witness lanes folded by `ch`, the extra lanes weighted by `ρ`, and the padding correction. -/
def padColumn (A : Arith F K) (d cols t : ℕ) (c₀ : F) (ch : List K) {e : ℕ} (ρ : Fin e → K)
    (xl : Fin e → ℕ) (ybar : Fin t → K) (C : Oracle F) (p : ℕ) : K :=
  foldColumn A ch (fun ℓ => C ℓ p) + ∑ i, ρ i * A.embed (C (xl i) p) + padCorr A d cols t c₀ ybar p

/-- The batched consistency value `Σ_j eq(α)[j]·v(p_j)` of a column function `v`. -/
def consValueV (A : Arith F K) (α : List F) (v : ℕ → K) (pos : List ℕ) : K :=
  ((pos.zipIdx).map fun (p, j) => A.embed (eqList α j) * v p).sum

/-- Queries and the consistency claim (as `queryAndBatch`, followed by a message), on the column
function `v`. -/
def queryAndBatchV (A : Arith F K) (lv : Level) (v : ℕ → K) (start : ℕ) (st : LigState F K) :
    Game (LigState F K) := do
  let w ← draw (Fin lv.queries → F)
  let α ← draw (Fin (ceilLog2 lv.queries) → F)
  let pos := positions A lv.queries lv.logLen w
  let e := consValueV A (List.ofFn α) v pos
  let msg ← recv (K × K)
  let β ← draw F
  let ctx : ConsCtx F := ⟨lv.logCols, pos, List.ofFn α, β, start⟩
  pure { st with
    s := st.s + A.embed β * e
    q := st.q.addScaled (fromMsg msg e) (A.embed β)
    cons := st.cons ++ [ctx] }

/-- Rep `r`'s extra lanes: `2^k₀ + e·r + i`. -/
def extraLane (k0 e r : ℕ) (i : Fin e) : ℕ := 2 ^ k0 + e * r + i.val

end FlockSoundness.Model
