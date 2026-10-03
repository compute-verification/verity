import FlockSoundness.Discharge.ZkExec.PlaysReplay
import FlockSoundness.Discharge.Composed.Inner.Lincheck

/-!
# `Zk.replay`'s lincheck terms are `Lincheck`'s at the transcript's coins

`replay_lincheck`: on a run of `Zk.replay` from `t`, the "lincheck final" constraint (`cons[2]`), the `ab` claim's value
`abValue` and the `ab` claim `claims[0]` are `lcFinal`, `lcAbValue` and `lcAbClaim` at the coins of the rounds the run
consumed, and the circuit fold the replay called returned the entries `lcFinal` reads. Round `t.k + i`'s first coin is
`coinAt t i` (`(t.rounds[t.k + i]!).coins.getD 0 0`, what `sample` reads; `zk_replay_plays`'s numbering): the zerocheck's
`rSkip`, `rOuter`, `z` at `0, 1, 2`, its residual coins `ρ_j` at `3 + j` (`rhoOf`), then the lincheck's `α`, `β` at
`3 + (m − 6)`, `4 + (m − 6)`, its round coins `t_j` at `5 + (m − 6) + j` (`tsOf`) and `r′` after them.

`foldAll_lcV`: `lincheck_complete`'s honest `z_partial` (`hzp`), the witness's block-folded columns folded at the round
coins, is entry `i` of the witness evaluated at the `ab` claim's point beyond the skip bits, `Σ_u eq(x, u)·w[64u + i]`
(the ring switch's `zPartialOf (claimOf m (lcAbClaim …)) w i`, written out).
-/

namespace FlockSoundness.Discharge.Composed.Lincheck

open Flock FlockLevel3 Model Refine

local notation "𝔽" => FlockLevel3.GF128

/-- Round `t.k + i`'s first coin, as `sample` reads it. -/
abbrev coinAt (t : Flock.Transcript) (i : ℕ) : F128 := (t.rounds[t.k + i]!).coins.getD 0 0

/-- The zerocheck's `m − 6` residual coins `ρ_j` (rounds `t.k + 3 + j`). -/
def rhoOf (m : ℕ) (t : Flock.Transcript) : Array F128 := Array.ofFn fun j : Fin (m - 6) => coinAt t (3 + j.val)

/-- The lincheck's `k_log − 6` round coins `t_j` (rounds `t.k + 5 + (m − 6) + j`). -/
def tsOf (m kLog : ℕ) (t : Flock.Transcript) : Array F128 :=
  Array.ofFn fun j : Fin (kLog - 6) => coinAt t (5 + (m - 6) + j.val)

/-- **`Zk.replay`'s lincheck terms at the transcript's coins.** On a run of `Zk.replay` from `t` (at `m ≥ 14`, where the
outer coins are one round), the circuit fold at the lincheck's `α`, `β`, the skip weights at `z`, `ρ[.. k_log − 6]` and
the round coins returned some `combv`, and `cons[2]`, `abValue` and `claims[0]` are `lcFinal`, `lcAbValue` and
`lcAbClaim` on the proof's masked messages at those coins. -/
theorem replay_lincheck (st : Setup) (lay : Zk.PadLayout) (p : Proof) (root : ByteArray) (hm : 14 ≤ st.m)
    {t t' : Flock.Transcript} {rp : Zk.Replayed} (h : Zk.replay st lay p root t = .ok (rp, t')) :
    ∃ combv : Array F128,
      st.fold.folded (coinAt t (3 + (st.m - 6))) (coinAt t (4 + (st.m - 6))) (skipWeights (coinAt t 2))
          ((rhoOf st.m t).extract 0 (st.kLog - 6)) (tsOf st.m st.kLog t) = .ok combv ∧
        rp.cons[2]! = ("lincheck final", lcFinal lay p.zerocheck.finalA p.zerocheck.finalB
          (coinAt t (3 + (st.m - 6))) (coinAt t (4 + (st.m - 6))) p.lincheck.rounds p.lincheck.zPartial
          (tsOf st.m st.kLog t) (st.kLog - 6) combv) ∧
        rp.abValue = lcAbValue lay p.lincheck.zPartial (coinAt t (5 + (st.m - 6) + (st.kLog - 6))) ∧
        rp.claims[0]! = lcAbClaim st.m st.kLog (rhoOf st.m t) (tsOf st.m st.kLog t)
          (coinAt t (5 + (st.m - 6) + (st.kLog - 6))) := by
  sorry

/-- **The honest `z_partial` is the witness at the `ab` claim's point** (§10): the block-folded columns `lcV` folded at
the round coins are, at skip position `i`, `Σ_u eq(x, u)·w[64u + i]` for the `ab` claim's point `x` (the ring switch's
`zPartialOf`). -/
theorem foldAll_lcV (S : Statement) (w : Witness S.m) (rho ts : Array F128) (hrho : rho.size = S.m - 6)
    (hts : ts.size = S.kLog - 6) (rPrime : F128) (i : Fin 64) :
    foldAll (ts.toList.map fOf) (2 ^ S.kLog) (lcV S rho w) i.val =
      ∑ u : Fin (2 ^ (S.m - 6)), Model.eqAt (vecOf (S.m - 6) (lcAbClaim S.m S.kLog rho ts rPrime).x) u.val *
        ((w ⟨64 * u.val + i.val, by
          have hu := u.isLt; have hi := i.isLt; have h6 := le_trans S.hk6 S.hk
          rw [show 2 ^ S.m = 2 ^ (S.m - 6) * 64 by
            rw [show (64 : ℕ) = 2 ^ 6 by norm_num, ← pow_add, Nat.sub_add_cancel h6]]
          omega⟩).val : 𝔽) := by
  sorry

end FlockSoundness.Discharge.Composed.Lincheck
