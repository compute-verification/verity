import Proofs.Flock.Soundness.Model.Session

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

/-- `k` folding rounds with no message after the last challenge (M1's lane rounds). Returns the
challenges and the claim. -/
def foldRoundsNL (A : Arith F K) : ℕ → K → Quad K → Game (List K × K)
  | 0, s, _ => pure ([], s)
  | 1, _, q => do
      let r ← drawK A
      pure ([r], q.eval r)
  | k + 2, _, q => do
      let r ← drawK A
      let s' := q.eval r
      let msg ← recv (K × K)
      let (rs, s'') ← foldRoundsNL A (k + 1) s' (fromMsg msg s')
      pure (r :: rs, s'')

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

/-- **Ligerito with the padded level 0** (§13.3 with M1's §7), for rep `r` of `e` extra lanes
each, padding `t` and constant `c₀`: decides `Σ_y f(y)·b(y) = T` for the committed table `C₀`. -/
def ligeritoPad (A : Arith F K) (sch : Schedule) (t : ℕ) (c₀ : F) (e r : ℕ) (T : F) (b : ℕ → F)
    (C₀ : Oracle F) : Game Bool := do
  let n := sch.m - 7
  match sch.levels with
  | [] => pure false
  | lv₀ :: rest => do
      -- the level-0 out-of-domain sample
      let z₀ ← draw (Fin n → F)
      let y₀ ← recv F
      let β₀ ← draw F
      let s := A.embed T + A.embed β₀ * A.embed y₀
      -- the lane rounds, with no message after the last coin
      let msg ← recv (K × K)
      let (initial, s) ← foldRoundsNL A sch.k0 s (fromMsg msg s)
      -- the extra lanes: `e^r` before its coins, then `ybar^r`
      let ev ← recv (Fin e → K)
      let ρ ← drawKs A e
      let ybar ← recv (Fin t → K)
      -- the restarted claim and the code-switch message
      let s := s + ∑ i, ρ i * ev i
      let msg' ← recv (K × K)
      let q := fromMsg msg' s
      match rest with
      | [] => pure false
      | lv₁ :: _ => do
          let C₁ ← recv (Oracle F)
          let dim := n - sch.k0 + 1
          let (s, q, oods) ← twoOod A dim 0 s q
          let st : LigState F K := ⟨s, q, dim, [], ⟨List.ofFn z₀, β₀, none⟩ :: oods, [], C₀, lv₀⟩
          let st ← queryAndBatchV A lv₀
            (padColumn A lv₀.logLen lv₀.logCols t c₀ initial ρ (extraLane sch.k0 e r) ybar C₀) 0 st
          recursiveRounds A n b initial sch.ext rest { st with prev := C₁, prevLevel := lv₁ }

/-- **One rep with the padded level 0.** -/
def repPad (A : Arith F K) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (t : ℕ) (c₀ : F)
    (e r : ℕ) (C₀ : Oracle F) (extra : List (Claim F S.m)) : Game Bool := do
  match ← zerocheck A S.m with
  | none => pure false
  | some zc =>
    match ← lincheck A S zc with
    | none => pure false
    | some ab =>
      match ← opening A (by omega) (ab :: cClaim A zc :: extra) with
      | none => pure false
      | some op => ligeritoPad A sch t c₀ e r op.T op.b C₀

/-- **One table's session with the padded level 0**: rep `0` and rep `1`, each with its own `e`
extra lanes. -/
def tablePad (A : Arith F K) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (t : ℕ) (c₀ : F)
    (e : ℕ) (ptLocal mPts : ℕ) : Game (Bool × Oracle F) := do
  let C₀ ← recv (Oracle F)
  let pts ← draw (Fin 2 → Fin mPts → F)
  let extra := extraClaims S ptLocal mPts pts
  let accepts ← interleave (repPad A S sch hm t c₀ e 0 C₀ extra) (repPad A S sch hm t c₀ e 1 C₀ extra)
  pure (accepts.1 && accepts.2, C₀)

end FlockSoundness.Model
