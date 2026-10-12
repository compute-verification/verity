import Proofs.Flock.Soundness.Discharge.ZkExec.VerdictLigModel
import Proofs.Flock.Soundness.Discharge.ZkExec.VerdictLigPad
import Proofs.Flock.Soundness.Discharge.ZkExec.VerdictLigFinal

/-!
# The hiding Ligerito's run, with its values

`zk_ligerito_plays` with the values: the run of the compiled masked phase's Ligerito that the executable's coins give has,
at the openings, the executable's claim, polynomial, out-of-domain and consistency contexts from level 0's queries on,
so its verdict is the executable's final check (`final_refinesZ`). The lane phase is masked, so only its coins, `ρ`
and the blinded claim `T'` carry over (`lane_vals`); level 0's enforced sum is `consE0` (`queryLevel0_val`), and with
the extra lanes' terms it is the model's padded column (`consPad_eq`). Levels ≥ 1 are the non-ZK session's
(`LigeritoRun.lean`).
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL Transcript

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

variable {β : Type}

/-- A round's two coins as an element of K, as `sampleK` reads them. -/
def kOf (c : Array F128) : F256 := ⟨c.getD 0 0, c.getD 1 0⟩

/-- **The lane rounds' values**: each pushes its K coin and reads the next message; the last also takes the two `ρ`
coins and sets the claim to `T'` before its message. -/
theorem lane_vals (p : LigeritoProof) (tP : F256)
    (body : ℕ → Array F256 × F256 × F256 → LM (ForInStep (Array F256 × F256 × F256))) (K : ℕ)
    (hbody : ∀ i s st t b st' t', i < K → body i s st t = .ok ((b, st'), t') →
      ∃ s', b = .yield s' ∧ t'.rounds = t.rounds ∧ t'.k = t.k + 1 + (if i + 1 = K then 2 else 0) ∧
        (t.rounds[t.k]!).n = 2 ∧ s'.1 = s.1.push (kOf (t.rounds[t.k]!).coins) ∧ st'.msg = st.msg + 1 ∧
        st'.ood = st.ood ∧ st'.oods = st.oods ∧ st'.conss = st.conss ∧
        (i + 1 = K → (t.rounds[t.k + 1]!).n = 2 ∧ (t.rounds[t.k + 2]!).n = 2 ∧
          s'.2 = (kOf (t.rounds[t.k + 1]!).coins, kOf (t.rounds[t.k + 2]!).coins) ∧ st'.s = tP ∧
          st'.q = Poly.ofMsg p.messages[st.msg]! tP)) :
    ∀ n a s st t s' st' t', a + n = K → 0 < n → forIn (List.range' a n) s body st t = .ok ((s', st'), t') →
      t'.rounds = t.rounds ∧ t'.k = t.k + n + 2 ∧ (∀ j < n + 2, (t.rounds[t.k + j]!).n = 2) ∧
      s'.1.toList = s.1.toList ++ (consumed t.rounds t.k (t.k + n)).map kOf ∧
      s'.2 = (kOf (t.rounds[t.k + n]!).coins, kOf (t.rounds[t.k + n + 1]!).coins) ∧
      st'.s = tP ∧ st'.q = Poly.ofMsg p.messages[st.msg + n - 1]! tP ∧ st'.msg = st.msg + n ∧ st'.ood = st.ood ∧
      st'.oods = st.oods ∧ st'.conss = st.conss
  | 0, _, _, _, _, _, _, _, _, h0, _ => absurd h0 (lt_irrefl 0)
  | 1, a, s, st, t, s', st', t', hK, _, h => by
      rw [List.range'_succ, List.forIn_cons] at h
      obtain ⟨b, st₁, t₁, h₁, h₂⟩ := lbind_ok h
      obtain ⟨s₁, rfl, hr, hk, hn, hs, hmsg, hood, hoods, hconss, hlast⟩ := hbody a s st t b st₁ t₁ (by omega) h₁
      simp only [List.range'_zero, List.forIn_nil] at h₂
      obtain ⟨rfl, rfl, rfl⟩ := lpure_ok h₂
      obtain ⟨hn1, hn2, hρ, hs', hq'⟩ := hlast (by omega)
      rw [ite_eq_left (show a + 1 = K by omega)] at hk
      refine ⟨hr, by omega, fun j hj => ?_, ?_, hρ, hs', by rw [hq', Nat.add_sub_cancel], hmsg, hood, hoods, hconss⟩
      · rcases (show j = 0 ∨ j = 1 ∨ j = 2 by omega) with rfl | rfl | rfl
        · simpa using hn
        · exact hn1
        · exact hn2
      rw [hs, consumed_add]
      simp
  | n + 2, a, s, st, t, s', st', t', hK, _, h => by
      rw [List.range'_succ, List.forIn_cons] at h
      obtain ⟨b, st₁, t₁, h₁, h₂⟩ := lbind_ok h
      obtain ⟨s₁, rfl, hr, hk, hn, hs, hmsg, hood, hoods, hconss, -⟩ := hbody a s st t b st₁ t₁ (by omega) h₁
      rw [ite_eq_right (show ¬ a + 1 = K by omega)] at hk
      simp only at h₂
      obtain ⟨hr', hk', hn', hs', hρ', hsS, hqS, hmsg', hood', hoods', hconss'⟩ :=
        lane_vals p tP body K hbody (n + 1) (a + 1) s₁ st₁ t₁ s' st' t' (by omega) (by omega) h₂
      refine ⟨hr'.trans hr, by omega, fun j hj => ?_, ?_, ?_, hsS, ?_, by omega, hood'.trans hood,
        hoods'.trans hoods, hconss'.trans hconss⟩
      · rcases j with _ | j
        · simpa using hn
        · have := hn' j (by omega)
          rw [show t.k + (j + 1) = t₁.k + j by omega, ← hr]
          exact this
      · rw [hs', hs, hr, hk, consumed_cons _ (show t.k < t.k + (n + 2) by omega)]
        simp only [Array.toList_push, List.map_cons, List.append_assoc, List.singleton_append]
        rw [show t.k + (n + 2) = t.k + 0 + 1 + (n + 1) by omega]
      · rw [hρ', hr, hk, show t.k + 0 + 1 + (n + 1) = t.k + (n + 2) by omega]
      · rw [hqS, hmsg, show st.msg + 1 + (n + 1) - 1 = st.msg + (n + 2) - 1 by omega]

/-- **Level 0's queries with their values**: `queryLevel0_bind` with the Merkle check on the `2^lanes + EXTRA_LANES`
rows and the enforced sum, `consE0` over the first `2^lanes` entries of each row. -/
theorem queryLevel0_val {ms : MerkleScheme} {p : LigeritoProof} {lv : Flock.Level} {cap : Array ByteArray}
    {fold : Array F256} (hq : lv.q ≠ 0)
    {f : Array ℕ × Array F128 × F256 → LM β} {st st' : LState} {t t' : Flock.Transcript} {b : β}
    (h : (Zk.queryLevel0 ms p lv cap fold >>= f) st t = .ok ((b, st'), t')) :
    (t.rounds[t.k]!).n = lv.q ∧ (t.rounds[t.k + 1]!).n = (lv.q - 1).log2 + 1 ∧
      merkleCheck ms cap lv.cap lv.d (positions lv (t.rounds[t.k]!).coins) p.initialProof.rows
        p.initialProof.siblings p.initialProof.salts (2 ^ lv.lanes + Zk.EXTRA_LANES) = .ok () ∧
      ∃ t₁, t₁.rounds = t.rounds ∧ t₁.k = t.k + 2 ∧
        f (positions lv (t.rounds[t.k]!).coins, (t.rounds[t.k + 1]!).coins,
            consE0 p.initialProof.rows lv.q (2 ^ lv.lanes) (t.rounds[t.k + 1]!).coins fold)
          { st with qn := st.qn + 1, cn := st.cn + 1 } t₁ = .ok ((b, st'), t') := by
  obtain ⟨r, st₁, t₁, h₁, h₂⟩ := lbind_ok h
  unfold Zk.queryLevel0 at h₁
  have h₁ := lget_bind h₁
  obtain ⟨qn, _, h₁⟩ := nonceFrom_bind h₁
  obtain ⟨cn, _, h₁⟩ := nonceFrom_bind h₁
  have h₁ := lset_bind h₁
  obtain ⟨_, t₃, h₃, h₁⟩ := lift_bind h₁
  obtain ⟨t₃', k₃, h₃'⟩ := pow_bind (f := fun _ => pure ()) (by rw [bind_pure]; exact h₃)
  obtain ⟨rfl, rfl⟩ := pure_ok h₃'
  obtain ⟨coins, t₄, h₄, h₁⟩ := lift_bind h₁
  obtain ⟨hn₄, hc₄, hr₄, hk₄⟩ := sampleVec_ok hq h₄
  obtain ⟨_, t₅, h₅, h₁⟩ := lift_bind h₁
  obtain ⟨t₅', k₅, h₅'⟩ := pow_bind (f := fun _ => pure ()) (by rw [bind_pure]; exact h₅)
  obtain ⟨rfl, rfl⟩ := pure_ok h₅'
  obtain ⟨alpha, t₆, h₆, h₁⟩ := lift_bind h₁
  obtain ⟨hn₆, hc₆, hr₆, hk₆⟩ := sampleVec_ok (by omega) h₆
  have er₄ : t₅.rounds = t.rounds := by rw [k₅.2, hr₄, k₃.2]
  have ek₄ : t₅.k = t.k + 1 := by rw [k₅.1, hk₄, k₃.1]
  have er₃ : t₃.rounds = t.rounds := k₃.2
  have ek₃ : t₃.k = t.k := k₃.1
  subst hc₄ hc₆
  cases hmc : merkleCheck ms cap lv.cap lv.d (positions lv (t₃.rounds[t₃.k]!).coins) p.initialProof.rows
      p.initialProof.siblings p.initialProof.salts (2 ^ lv.lanes + Zk.EXTRA_LANES) with
  | error e => simp only [hmc] at h₁; obtain ⟨_, _, _, h₀, _⟩ := lbind_ok h₁; cases h₀
  | ok u =>
    simp only [hmc] at h₁
    simp only [List.forIn_pure_yield_eq_foldl, Std.Legacy.Range.forIn_eq_forIn_range', pure_bind] at h₁
    obtain ⟨rfl, rfl, rfl⟩ := lpure_ok h₁
    simp only [range_size, ← List.range_eq_range'] at h₂
    refine ⟨by rw [← er₃, ← ek₃]; exact hn₄, by rw [← er₄, ← ek₄]; exact hn₆,
      by rw [← er₃, ← ek₃]; exact hmc, _, hr₆.trans er₄, by rw [hk₆, ek₄], ?_⟩
    rw [er₃, ek₃, er₄, ek₄] at h₂
    exact h₂

end FlockSoundness.Discharge.ZkExec
