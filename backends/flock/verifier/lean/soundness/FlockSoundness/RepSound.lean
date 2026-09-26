import FlockSoundness.Defs
import FlockSoundness.Rounds

/-!
# One rep is sound

`rep_sound`: from a doomed start (no codeword near the committed table packs a witness that
satisfies the R1CS and the pin and makes every region claim true), one rep accepts with
probability at most `repError`. It is proved here from four phase lemmas, one per phase of
PROTOCOL.md §9–§13, each bounding the probability that its phase ends outside a doomed state:

| phase | doomed at its end, for every codeword near the committed table | error |
|---|---|---|
| zerocheck (§9) | the `a`, `b`, `c` claims, the pin at `ρ_out` and the region claims do not all hold | `εZc` |
| lincheck (§10) | the `ab`, `c` and region claims do not all hold | `εLc` |
| opening (§12) | the batched inner product `Σ_y f(y)·b(y)` is not `T` | `εOp` |
| Ligerito (§13) | (accepts) | `εLig` |

The errors add up to `repError` exactly (`repError_eq`), and `Game.value_bind_le` chains the
phases. Each phase lemma is a chain of per-coin (round-by-round) steps: `Rounds` for the sumcheck
rounds, `Game.prCoin_mvEval_eq_zero_le` for the vector coins.
-/

namespace FlockSoundness

open Game Model Accounting

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-! ## The witness's A-side and B-side, and the pin's extension -/

theorem Model.Statement.block_index_lt (S : Statement) (a : Fin (2 ^ S.m)) :
    a.val / 2 ^ S.kLog < 2 ^ S.nbl := by
  have h : 2 ^ S.m = 2 ^ S.nbl * 2 ^ S.kLog := by rw [← pow_add, Nat.sub_add_cancel S.hk]
  rw [Nat.div_lt_iff_lt_mul (Nat.two_pow_pos _), ← h]
  exact a.isLt

/-- The A-side bits: `(A·z)` at address `a = o·2^k_log + i` is `(A₀·z_o)_i`. -/
def aVec (S : Statement) (z : Witness S.m) : Witness S.m := fun a =>
  (S.A₀.mulVec (S.block z ⟨a.val / 2 ^ S.kLog, S.block_index_lt a⟩))
    ⟨a.val % 2 ^ S.kLog, Nat.mod_lt _ (Nat.two_pow_pos _)⟩

/-- The B-side bits. -/
def bVec (S : Statement) (z : Witness S.m) : Witness S.m := fun a =>
  (S.B₀.mulVec (S.block z ⟨a.val / 2 ^ S.kLog, S.block_index_lt a⟩))
    ⟨a.val % 2 ^ S.kLog, Nat.mod_lt _ (Nat.two_pow_pos _)⟩

/-- The pin column's multilinear extension over the blocks. The lincheck's constant-wire coin
`β` checks that it is 1 at `ρ_out` (§10, A8). -/
def pinMLE (S : Statement) (z : Witness S.m) (ρOut : Fin S.nbl → F) : F :=
  ∑ o : Fin (2 ^ S.nbl), eqAt ρOut o.val * ((S.block z o S.pin).val : F)

/-- `ρ_out`: the zerocheck's challenges for the block bits (address bits `k_log … m−1`). -/
def Model.ZcOut.ρOut (S : Statement) (zc : ZcOut F S.m) : Fin S.nbl → F := fun j =>
  if h : S.kLog - 6 + j.val < S.m - 6 then zc.ρ ⟨S.kLog - 6 + j.val, h⟩ else 0

/-! ## The codewords near the committed table -/

/-- Messages whose encodings are within the level-0 radius of the committed table. -/
def closeMsgs (A : Arith F K) (sch : Schedule) (C : Oracle F) :
    Set (Message F sch.k0 sch.level₀.logCols) :=
  {M | Close sch.level₀.radius (asTable sch C) (encode A sch.level₀.logLen M)}

/-- The witness a message packs. -/
def wit (A : Arith F K) (S : Statement) (sch : Schedule) (M : Message F sch.k0 sch.level₀.logCols) :
    Witness S.m :=
  witnessOf A S.m sch.k0 sch.level₀.logCols M

/-- The packed table entry `f(y)` of a message (§12.1, §13.1: `t[c·2^k₀ + ℓ] = M[ℓ][c]`). -/
def packedAt (sch : Schedule) (M : Message F sch.k0 sch.level₀.logCols) (y : ℕ) : F :=
  if h : y / 2 ^ sch.k0 < 2 ^ sch.level₀.logCols then
    M ⟨y % 2 ^ sch.k0, Nat.mod_lt _ (Nat.two_pow_pos _)⟩ ⟨y / 2 ^ sch.k0, h⟩
  else 0

/-! ## Doomed states between the phases -/

/-- Doomed after the zerocheck. -/
def DZc (A : Arith F K) (S : Statement) (sch : Schedule) (C : Oracle F)
    (extra : List (Claim F S.m)) : Option (ZcOut F S.m) → Prop
  | none => True
  | some zc => ∀ M ∈ closeMsgs A sch C,
      ¬ ((⟨zc.va, A.lagS zc.z, zc.ρ⟩ : Claim F S.m).Holds (aVec S (wit A S sch M))
          ∧ (⟨zc.vb, A.lagS zc.z, zc.ρ⟩ : Claim F S.m).Holds (bVec S (wit A S sch M))
          ∧ (cClaim A zc).Holds (wit A S sch M)
          ∧ pinMLE S (wit A S sch M) (zc.ρOut S) = 1
          ∧ ∀ e ∈ extra, e.Holds (wit A S sch M))

/-- Doomed after the lincheck. -/
def DLc (A : Arith F K) (S : Statement) (sch : Schedule) (C : Oracle F)
    (extra : List (Claim F S.m)) (zc : ZcOut F S.m) : Option (Claim F S.m) → Prop
  | none => True
  | some ab => ∀ M ∈ closeMsgs A sch C,
      ¬ (ab.Holds (wit A S sch M) ∧ (cClaim A zc).Holds (wit A S sch M)
          ∧ ∀ e ∈ extra, e.Holds (wit A S sch M))

/-- Doomed after the opening: no codeword near the committed table has the batched value. -/
def DOp (A : Arith F K) (S : Statement) (sch : Schedule) (C : Oracle F) :
    Option (OpenOut F) → Prop
  | none => True
  | some op => ∀ M ∈ closeMsgs A sch C,
      (∑ y ∈ Finset.range (2 ^ (S.m - 7)), packedAt sch M y * op.b y) ≠ op.T

/-! ## The error of each phase -/

/-- Zerocheck and the pin (Z3, Z5, Z6 and `ρ_out`). -/
noncomputable def εZc (s : Schedule) (st : Stmt) : ℝ :=
  (((s.listBound₀ + 1) * (s.m - 13) + s.listBound₀ * 127 + s.listBound₀ * (2 * (s.m - 6))
      + s.listBound₀ * (s.m - st.kLog) : ℕ) : ℝ) / qF

/-- Lincheck (L1, L2, L3, L5). -/
noncomputable def εLc (s : Schedule) (st : Stmt) : ℝ :=
  ((s.listBound₀ * (2 + 2 * (st.kLog - 6) + 63) : ℕ) : ℝ) / qF

/-- Opening (O5 per claim, O6). -/
noncomputable def εOp (s : Schedule) (st : Stmt) : ℝ :=
  ((s.listBound₀ * (7 * st.nClaims + 1) : ℕ) : ℝ) / qF

/-- Ligerito: the level-0 out-of-domain claim's batching coin, then every level. -/
noncomputable def εLig (s : Schedule) : ℝ := (s.listBound₀ : ℝ) / qF + ligeritoError s.levels

theorem repError_eq (s : Schedule) (st : Stmt) :
    repError s st = εZc s st + εLc s st + εOp s st + εLig s := by
  unfold repError piopError εZc εLc εOp εLig
  push_cast
  ring

theorem εZc_nonneg (s : Schedule) (st : Stmt) : 0 ≤ εZc s st := by unfold εZc qF; positivity
theorem εLc_nonneg (s : Schedule) (st : Stmt) : 0 ≤ εLc s st := by unfold εLc qF; positivity
theorem εOp_nonneg (s : Schedule) (st : Stmt) : 0 ≤ εOp s st := by unfold εOp qF; positivity
theorem εLig_nonneg (s : Schedule) (hs : ∀ l ∈ s.levels, l.Ok) : 0 ≤ εLig s := by
  unfold εLig qF
  have := ligeritoError_nonneg s.levels hs
  positivity

/-! ## The four phases (to prove) -/

/-- **Zerocheck.** From a doomed start, the zerocheck ends doomed except with probability `εZc`.
*Status: to prove.* The eq-reduction of each nonzero Boolean row (degree `m − 13`, the pinned
coordinates independent over GF(2)), the skip point (`127/|F|`), the `m − 6` residual rounds
(`Rounds.residualRound_sound`, which needs every `r_eq ≠ 1`, the exceptional event charged once),
the pin's extension at `ρ_out` (degree `m − k_log`); every term times the list bound. -/
theorem zerocheck_phase (A : Arith F K) (hA : A.Correct) (S : Statement) (sch : Schedule)
    (hsch : fast100 S.m = some sch) (hm : 13 ≤ S.m) (mPts : ℕ) (C : Oracle F)
    (extra : List (Claim F S.m)) (hdoom : RepDoomed A S sch C extra) :
    value (fun o => ¬ DZc A S sch C extra o) (zerocheck A S.m) ≤
      ENNReal.ofReal (εZc sch (S.shape mPts)) := by
  sorry

/-- **Lincheck.** *Status: to prove.* `α` and `β` (degree 1 each; `β` catches a pin whose
extension is not 1), the `k_log − 6` rounds (`Rounds.lincheckRound_sound`), the skip coin `r'`
(degree 63); every term times the list bound. -/
theorem lincheck_phase (A : Arith F K) (hA : A.Correct) (S : Statement) (sch : Schedule)
    (hsch : fast100 S.m = some sch) (mPts : ℕ) (C : Oracle F) (extra : List (Claim F S.m))
    (zc : ZcOut F S.m) (hzc : DZc A S sch C extra (some zc)) :
    value (fun o => ¬ DLc A S sch C extra zc o) (lincheck A S zc) ≤
      ENNReal.ofReal (εLc sch (S.shape mPts)) := by
  sorry

/-- **Opening.** *Status: to prove.* Per claim, a wrong `s_k` survives `r''_k` with probability
at most `7/|F|` (the injective recombination over GF(2) of paper Appendix B), and the batch
`γ` keeps a false claim false except with probability `1/|F|`; times the list bound. -/
theorem opening_phase (A : Arith F K) (hA : A.Correct) (S : Statement) (sch : Schedule)
    (hsch : fast100 S.m = some sch) (h7 : 7 ≤ S.m) (mPts : ℕ) (C : Oracle F)
    (extra : List (Claim F S.m)) (hextra : extra.length = 2 * S.regions.length)
    (zc : ZcOut F S.m) (ab : Claim F S.m) (hab : DLc A S sch C extra zc (some ab)) :
    value (fun o => ¬ DOp A S sch C o) (opening A h7 (ab :: cClaim A zc :: extra)) ≤
      ENNReal.ofReal (εOp sch (S.shape mPts)) := by
  sorry

/-- **Ligerito.** *Status: to prove.* Level by level: the folds (sumcheck `2L/|K|` per round,
`Rounds.sumRound_sound` over `K`; mutual correlated agreement from BCHKS25 Theorem 4.6, with the
row union of paper Lemma 9), the binding of the next level by its two out-of-domain samples, the
stratified queries (`(1 − γ)^Q`), and the batching coins. -/
theorem ligerito_phase (hMCA : Assumptions.BCHKS25Thm46) (A : Arith F K) (hA : A.Correct)
    (S : Statement) (sch : Schedule) (hsch : fast100 S.m = some sch) (C : Oracle F)
    (op : OpenOut F) (hop : DOp A S sch C (some op)) :
    value (fun b => b = true) (ligerito A sch op.T op.b C) ≤ ENNReal.ofReal (εLig sch) := by
  sorry

/-! ## Composition -/

/-- **One rep is sound** (PROTOCOL.md §9–§13): from a doomed start, whatever the prover does,
the rep accepts with probability at most `repError`. Proved from the four phase lemmas. -/
theorem rep_sound (hMCA : Assumptions.BCHKS25Thm46) (A : Arith F K) (hA : A.Correct)
    (S : Statement) (sch : Schedule) (hsch : fast100 S.m = some sch) (hm : 13 ≤ S.m)
    (mPts : ℕ) (C : Oracle F) (extra : List (Claim F S.m))
    (hextra : extra.length = 2 * S.regions.length) (hdoom : RepDoomed A S sch C extra) :
    value (fun b => b = true) (rep A S sch hm C extra) ≤
      ENNReal.ofReal (repError sch (S.shape mPts)) := by
  have hok := fast100_levels_ok S.m sch hsch
  have hZc := εZc_nonneg sch (S.shape mPts)
  have hLc := εLc_nonneg sch (S.shape mPts)
  have hOp := εOp_nonneg sch (S.shape mPts)
  have hLig := εLig_nonneg sch hok
  rw [repError_eq, ENNReal.ofReal_add (by positivity) hLig, ENNReal.ofReal_add (by positivity) hOp,
    ENNReal.ofReal_add hZc hLc]
  unfold rep
  simp only [bind_eq]
  -- zerocheck
  refine (value_bind_le _ _ _ (DZc A S sch C extra)
    (ENNReal.ofReal (εLc sch (S.shape mPts)) + ENNReal.ofReal (εOp sch (S.shape mPts))
      + ENNReal.ofReal (εLig sch)) (fun o ho => ?_)).trans ?_
  · cases o with
    | none => simp
    | some zc =>
      simp only
      -- lincheck
      refine (value_bind_le _ _ _ (DLc A S sch C extra zc)
        (ENNReal.ofReal (εOp sch (S.shape mPts)) + ENNReal.ofReal (εLig sch))
        (fun o₂ ho₂ => ?_)).trans ?_
      · cases o₂ with
        | none => simp
        | some ab =>
          simp only
          -- opening
          refine (value_bind_le _ _ _ (DOp A S sch C) (ENNReal.ofReal (εLig sch))
            (fun o₃ ho₃ => ?_)).trans ?_
          · cases o₃ with
            | none => simp
            | some op => exact ligerito_phase hMCA A hA S sch hsch C op ho₃
          · exact add_le_add (opening_phase A hA S sch hsch _ mPts C extra hextra zc ab ho₂) le_rfl
      · rw [← add_assoc]
        exact add_le_add (add_le_add (lincheck_phase A hA S sch hsch mPts C extra zc ho) le_rfl)
          le_rfl
  · calc value (fun o => ¬ DZc A S sch C extra o) (zerocheck A S.m)
          + (ENNReal.ofReal (εLc sch (S.shape mPts)) + ENNReal.ofReal (εOp sch (S.shape mPts))
            + ENNReal.ofReal (εLig sch))
        ≤ ENNReal.ofReal (εZc sch (S.shape mPts))
          + (ENNReal.ofReal (εLc sch (S.shape mPts)) + ENNReal.ofReal (εOp sch (S.shape mPts))
            + ENNReal.ofReal (εLig sch)) :=
          add_le_add (zerocheck_phase A hA S sch hsch hm mPts C extra hdoom) le_rfl
      _ = _ := by ring

end FlockSoundness
