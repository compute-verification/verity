import Proofs.Flock.Claims.KS.RoundsN
import Proofs.Flock.Soundness.Discharge.ZkExec.RunsReplay

/-!
# The coins each recorded masked round issues

`RoundsN` for `--zk`. Where a masked rep accepts, every round of its stream issued the number of coins the statement
fixes, whatever the hashes (`ReadsN`): the masked replay's zerocheck and lincheck at `m` and `kLog` (`zcNs`, `lcNs`, as
plain; `replayWith_readsN`), the ring switch's at its claims (`rsNs`), masked Ligerito's at the `--zk` schedule
(`zkLigNs`: the plain count with the lane phase's `ρ` pair after the last lane challenge; `zkLigeritoWith_readsN`) and
the inner proof's `ε`, its ten batching coins, `β` and its `Q` column queries (`innerNs`; `verifyInnerWith_readsN`), in
that order (`zkRepNs`, `zkVerifyRepWith_readsN`). So each rep stream of an accepted masked record issued its table's
`zkRepNsOf` (`streams_ns_zk`), its rounds in `spec.repStreams` order the session's `zkNs` (`rounds_ns_zk`), and custody
reads each round message's coin at that count (`custody_vals_zk`).
-/

namespace FlockDeploy

open Flock Transcript FlockSoundness.Refine

/-! ## The masked replay and the inner proof -/

theorem replayWith_readsN (hs : Hashes Flock.V) {st : Setup} {lay : Zk.PadLayout} {p : Proof} {root : ByteArray}
    (hm : 14 ≤ st.m) {t t' : Transcript} {rp : Zk.Replayed} (h : Zk.replayWith hs st lay p root t = .ok (rp, t')) :
    t'.rounds = t.rounds ∧ t'.k = t.k + (zcNs st.m ++ lcNs st.kLog).length ∧
      ReadsN t.rounds t.k (zcNs st.m ++ lcNs st.kLog) ∧ rp.cons.size = 3 := by
  unfold Zk.replayWith at h
  simp only [liftTM_V] at h
  obtain ⟨t₁, k₁, h⟩ := label_bind h
  obtain ⟨t₂, k₂, h⟩ := absorbBytes_bind h
  obtain ⟨t₃, k₃, h⟩ := absorbBytes_bind h
  obtain ⟨t₄, k₄, h⟩ := label_bind h
  obtain ⟨t₅, k₅, h⟩ := label_bind h
  obtain ⟨t₆, k₆, h⟩ := absorbBytes_bind h
  obtain ⟨-, n₇, t₇, r₇, kk₇, h⟩ := sampleVecWith_bind hs (by omega) h
  obtain ⟨-, n₈, t₈, r₈, kk₈, h⟩ := sampleVecWith_bind hs (by omega) h
  obtain ⟨t₉, k₉, h⟩ := absorbSlice_bind h
  obtain ⟨t₁₀, k₁₀, h⟩ := absorbSlice_bind h
  obtain ⟨-, n₁₁, t₁₁, r₁₁, kk₁₁, h⟩ := sampleWith_bind hs h
  obtain ⟨s, t₁₂, hloop, h⟩ := bind_ok h
  rw [forIn_range_eq] at hloop
  obtain ⟨hr₁₂, hk₁₂, R₁₂⟩ := forIn_readsN _ 1 (by
      intro i st' tt b tt' hb
      obtain ⟨tt₁, kk₁, hb⟩ := absorbF_bind hb
      obtain ⟨tt₂, kk₂, hb⟩ := absorbF_bind hb
      obtain ⟨-, nn₃, tt₃, rr₃, kk₃, hb⟩ := sampleWith_bind hs hb
      obtain ⟨rfl, rfl⟩ := pure_ok hb
      rw [kk₂.2, kk₁.2, kk₂.1, kk₁.1] at nn₃
      exact ⟨nn₃, by rw [rr₃, kk₂.2, kk₁.2], by rw [kk₃, kk₂.1, kk₁.1], _, rfl⟩) (st.m - 6) 0 _ t₁₁ t₁₂ s hloop
  obtain ⟨t₁₃, k₁₃, h⟩ := absorbF_bind h
  obtain ⟨t₁₄, k₁₄, h⟩ := absorbF_bind h
  obtain ⟨t₁₅, k₁₅, h⟩ := label_bind h
  obtain ⟨-, n₁₆, t₁₆, r₁₆, kk₁₆, h⟩ := sampleWith_bind hs h
  obtain ⟨-, n₁₇, t₁₇, r₁₇, kk₁₇, h⟩ := sampleWith_bind hs h
  obtain ⟨s', t₁₈, hloop', h⟩ := bind_ok h
  rw [forIn_range_eq] at hloop'
  obtain ⟨hr₁₈, hk₁₈, R₁₈⟩ := forIn_readsN _ 1 (by
      intro i st' tt b tt' hb
      obtain ⟨tt₁, kk₁, hb⟩ := absorbF_bind hb
      obtain ⟨tt₂, kk₂, hb⟩ := absorbF_bind hb
      obtain ⟨-, nn₃, tt₃, rr₃, kk₃, hb⟩ := sampleWith_bind hs hb
      obtain ⟨rfl, rfl⟩ := pure_ok hb
      rw [kk₂.2, kk₁.2, kk₂.1, kk₁.1] at nn₃
      exact ⟨nn₃, by rw [rr₃, kk₂.2, kk₁.2], by rw [kk₃, kk₂.1, kk₁.1], _, rfl⟩) (st.kLog - 6) 0 _ t₁₇ t₁₈ s'
        hloop'
  try dsimp only at h
  cases hf : st.fold.folded ((t₁₅.rounds[t₁₅.k]!).coins.getD 0 0) ((t₁₆.rounds[t₁₆.k]!).coins.getD 0 0)
      (skipWeights ((t₁₀.rounds[t₁₀.k]!).coins.getD 0 0)) (s.2.extract 0 (st.kLog - 6)) s'.2 with
  | error err => rw [hf] at h; obtain ⟨_, _, h0, _⟩ := bind_ok h; cases h0
  | ok v =>
  rw [hf] at h
  simp only [pure_bind] at h
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨t₁₉, k₁₉, h⟩ := absorbSlice_bind h
  obtain ⟨-, n₂₀, t₂₀, r₂₀, kk₂₀, h⟩ := sampleWith_bind hs h
  obtain ⟨rfl, rfl⟩ := pure_ok h
  have e₆r : t₆.rounds = t.rounds := by rw [k₆.2, k₅.2, k₄.2, k₃.2, k₂.2, k₁.2]
  have e₆k : t₆.k = t.k := by rw [k₆.1, k₅.1, k₄.1, k₃.1, k₂.1, k₁.1]
  have e₁₀r : t₁₀.rounds = t.rounds := by rw [k₁₀.2, k₉.2, r₈, r₇, e₆r]
  have e₁₀k : t₁₀.k = t.k + 2 := by rw [k₁₀.1, k₉.1, kk₈, kk₇, e₆k]
  have e₁₅r : t₁₅.rounds = t.rounds := by rw [k₁₅.2, k₁₄.2, k₁₃.2, hr₁₂, r₁₁, e₁₀r]
  have e₁₅k : t₁₅.k = t.k + 3 + (st.m - 6) := by rw [k₁₅.1, k₁₄.1, k₁₃.1, hk₁₂, kk₁₁, e₁₀k]
  have e₁₉r : t₁₉.rounds = t.rounds := by rw [k₁₉.2, hr₁₈, r₁₇, r₁₆, e₁₅r]
  have e₁₉k : t₁₉.k = t.k + 3 + (st.m - 6) + 2 + (st.kLog - 6) := by rw [k₁₉.1, hk₁₈, kk₁₇, kk₁₆, e₁₅k]
  rw [e₆r, e₆k] at n₇
  rw [r₇, e₆r, kk₇, e₆k] at n₈
  rw [e₁₀r, e₁₀k] at n₁₁
  rw [r₁₁, e₁₀r, kk₁₁, e₁₀k] at R₁₂
  rw [e₁₅r, e₁₅k] at n₁₆
  rw [r₁₆, e₁₅r, kk₁₆, e₁₅k] at n₁₇
  rw [r₁₇, r₁₆, e₁₅r, kk₁₇, kk₁₆, e₁₅k] at R₁₈
  rw [e₁₉r, e₁₉k] at n₂₀
  refine ⟨by rw [r₂₀, e₁₉r], ?_, ?_, by simp⟩
  · rw [kk₂₀, e₁₉k]
    simp only [zcNs, lcNs, List.length_append, List.length_cons, List.length_nil, List.length_replicate]
    omega
  · refine readsN_append (readsN_cons n₇ rfl (readsN_cons n₈ rfl (readsN_cons n₁₁ (by omega) R₁₂))) ?_
      (readsN_append (readsN_append (readsN_cons n₁₆ rfl (readsN_cons n₁₇ rfl (readsN_nil _ _))) ?_ R₁₈) ?_
        (readsN_cons n₂₀ rfl (readsN_nil _ _))) <;>
      simp only [zcNs, List.length_append, List.length_cons, List.length_nil, List.length_replicate] <;> omega

/-- The inner proof's coins at its constraints' count: `ε`, the batching coins of those and the triple's three, `β`,
and the `Q` column queries. -/
def innerNs (n : ℕ) : List ℕ := [1, n + 3, 1, Zk.Q]

theorem verifyInnerWith_readsN (hs : Hashes Flock.V) {lay : Zk.PadLayout} {pr : Zk.InnerProof} {finalC : F128}
    {cons : Array (String × Zk.Lin)} {t t' : Transcript} {u : Unit}
    (h : Zk.verifyInnerWith hs lay pr finalC cons t = .ok (u, t')) :
    t'.rounds = t.rounds ∧ t'.k = t.k + (innerNs cons.size).length ∧ ReadsN t.rounds t.k (innerNs cons.size) := by
  unfold Zk.verifyInnerWith at h
  simp only [liftTM_V] at h
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨t₁, k₁, h⟩ := label_bind h
  obtain ⟨t₂, k₂, h⟩ := absorbF_bind h
  obtain ⟨-, n₃, t₃, r₃, kk₃, h⟩ := sampleWith_bind hs h
  obtain ⟨t₄, k₄, h⟩ := absorbSlice_bind h
  obtain ⟨-, n₅, t₅, r₅, kk₅, h⟩ := sampleVecWith_bind hs (by simp [Zk.triple]) h
  obtain ⟨t₆, k₆, h⟩ := absorbBytes_bind h
  obtain ⟨-, n₇, t₇, r₇, kk₇, h⟩ := sampleWith_bind hs h
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨t₈, k₈, h⟩ := absorbSlice_bind h
  obtain ⟨-, n₉, t₉, r₉, kk₉, h⟩ := sampleVecWith_bind hs (by decide) h
  obtain ⟨tc, -, h⟩ := vlift_bind h
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨s, t₁₀, hloop, h⟩ := bind_ok h
  obtain ⟨rfl, rfl⟩ := pure_ok h
  rw [forIn_range_eq] at hloop
  obtain ⟨hr₁₀, hk₁₀⟩ := forIn_keeps _ (by
      intro i s tt b tt' hb
      obtain ⟨-, hb⟩ := check_bind hb
      obtain ⟨-, hb⟩ := check_bind hb
      obtain ⟨_, -, hb⟩ := vlift_bind hb
      obtain ⟨_, -, hb⟩ := vlift_bind hb
      obtain ⟨_, -, hb⟩ := vlift_bind hb
      obtain ⟨-, hb⟩ := check_bind hb
      obtain ⟨rfl, rfl⟩ := pure_ok hb
      exact ⟨rfl, rfl⟩) _ _ _ _ _ hloop
  have hsz : (cons ++ Zk.triple lay ((t₂.rounds[t₂.k]!).coins.getD 0 0) pr.rho pr.sigma).size = cons.size + 3 := by
    simp [Zk.triple]
  rw [hsz] at n₅
  rw [k₂.2, k₁.2, k₂.1, k₁.1] at n₃
  rw [k₄.2, r₃, k₂.2, k₁.2, k₄.1, kk₃, k₂.1, k₁.1] at n₅
  rw [k₆.2, r₅, k₄.2, r₃, k₂.2, k₁.2, k₆.1, kk₅, k₄.1, kk₃, k₂.1, k₁.1] at n₇
  rw [k₈.2, r₇, k₆.2, r₅, k₄.2, r₃, k₂.2, k₁.2, k₈.1, kk₇, k₆.1, kk₅, k₄.1, kk₃, k₂.1, k₁.1] at n₉
  refine ⟨by rw [hr₁₀, r₉, k₈.2, r₇, k₆.2, r₅, k₄.2, r₃, k₂.2, k₁.2], ?_,
    readsN_cons n₃ rfl (readsN_cons n₅ rfl (readsN_cons n₇ rfl (readsN_cons n₉ rfl (readsN_nil _ _))))⟩
  rw [hk₁₀, kk₉, k₈.1, kk₇, k₆.1, kk₅, k₄.1, kk₃, k₂.1, k₁.1]
  rfl

/-! ## Masked Ligerito -/

open FlockSoundness.Discharge.ZkExec in
/-- `Zk.queryLevel0With lv`: level 0's query coins, then its consistency coins (`qNs`). -/
theorem queryLevel0With_readsN {β : Type} (hs : Hashes Flock.V) {ms : MerkleScheme} {p : LigeritoProof}
    {lv : Flock.Level} {cap : Array ByteArray} {fold : Array F256} (hq : lv.q ≠ 0)
    {f : Array ℕ × Array F128 × F256 → LM β} {st st' : LState} {t t' : Transcript} {b : β}
    (h : (Zk.queryLevel0With hs ms p lv cap fold >>= f) st t = .ok ((b, st'), t')) :
    ReadsN t.rounds t.k (qNs lv) ∧
      ∃ r st₁ t₁, t₁.rounds = t.rounds ∧ t₁.k = t.k + 2 ∧ f r st₁ t₁ = .ok ((b, st'), t') := by
  obtain ⟨r, st₁, t₁, h₁, h₂⟩ := lbind_ok h
  unfold Zk.queryLevel0With at h₁
  simp only [liftLM_V] at h₁
  have h₁ := lget_bind h₁
  obtain ⟨qn, _, h₁⟩ := nonceFrom_bind h₁
  obtain ⟨cn, _, h₁⟩ := nonceFrom_bind h₁
  have h₁ := lset_bind h₁
  obtain ⟨_, t₃, h₃, h₁⟩ := lift_bind h₁
  obtain ⟨t₃', k₃, h₃'⟩ := pow_bind (f := fun _ => pure ()) (by rw [bind_pure]; exact h₃)
  obtain ⟨rfl, rfl⟩ := pure_ok h₃'
  obtain ⟨hn₄, t₄, hr₄, hk₄, h₁⟩ := lsampleVecWith_bind hs hq h₁
  obtain ⟨_, t₅, h₅, h₁⟩ := lift_bind h₁
  obtain ⟨t₅', k₅, h₅'⟩ := pow_bind (f := fun _ => pure ()) (by rw [bind_pure]; exact h₅)
  obtain ⟨rfl, rfl⟩ := pure_ok h₅'
  obtain ⟨hn₆, t₆, hr₆, hk₆, h₁⟩ := lsampleVecWith_bind hs (by omega) h₁
  cases hmc : merkleCheckWith (ms.opsWith hs) ms cap lv.cap lv.d (positions lv (t₃.rounds[t₃.k]!).coins)
      p.initialProof.rows p.initialProof.siblings p.initialProof.salts (2 ^ lv.lanes + Zk.EXTRA_LANES) with
  | error e => simp only [hmc] at h₁; obtain ⟨_, _, _, h₀, _⟩ := lbind_ok h₁; cases h₀
  | ok u =>
    simp only [hmc] at h₁
    simp only [List.forIn_pure_yield_eq_foldl, Std.Legacy.Range.forIn_eq_forIn_range', pure_bind] at h₁
    obtain ⟨rfl, rfl, rfl⟩ := lpure_ok h₁
    have e₅r : t₅.rounds = t.rounds := by rw [k₅.2, hr₄, k₃.2]
    have e₅k : t₅.k = t.k + 1 := by rw [k₅.1, hk₄, k₃.1]
    rw [e₅r, e₅k] at hn₆
    rw [k₃.2, k₃.1] at hn₄
    exact ⟨readsN_cons hn₄ rfl (readsN_cons hn₆ rfl (readsN_nil _ _)), _, _, _, by rw [hr₆, e₅r],
      by rw [hk₆, e₅k], h₂⟩

/-- The masked lane loop: each of its `k0` steps reads one `K` coin, and the last two more (the lane phase's `ρ`
pair), so `n ≥ 1` steps read `n + 2` coins of two field elements. -/
theorem lforIn_lanes_readsN {σ : Type} (body : ℕ → σ → LM (ForInStep σ)) (k0 : ℕ)
    (hbody : ∀ i s st t b st' t', i < k0 → body i s st t = .ok ((b, st'), t') →
      t'.rounds = t.rounds ∧ t'.k = t.k + (if i + 1 = k0 then 3 else 1) ∧
        ReadsN t.rounds t.k (List.replicate (if i + 1 = k0 then 3 else 1) 2) ∧ ∃ s', b = .yield s') :
    ∀ (n a : ℕ) (s : σ) (st : LState) (t : Transcript) (out : σ) (st' : LState) (t' : Transcript),
      a + n = k0 → forIn (List.range' a n) s body st t = .ok ((out, st'), t') →
      t'.rounds = t.rounds ∧ t'.k = t.k + (n + if n = 0 then 0 else 2) ∧
        ReadsN t.rounds t.k (List.replicate (n + if n = 0 then 0 else 2) 2) := by
  intro n
  induction n with
  | zero =>
    intro a s st t out st' t' _ h
    simp only [List.range'_zero, List.forIn_nil] at h
    obtain ⟨rfl, rfl, rfl⟩ := lpure_ok h
    exact ⟨rfl, by simp, readsN_nil _ _⟩
  | succ n ih =>
    intro a s st t out st' t' ha h
    rw [List.range'_succ, List.forIn_cons] at h
    obtain ⟨b, st₁, t₁, h₁, h₂⟩ := lbind_ok h
    obtain ⟨hr₁, hk₁, R₁, s', rfl⟩ := hbody a s st t b st₁ t₁ (by omega) h₁
    obtain ⟨hr, hk, R⟩ := ih (a + 1) s' st₁ t₁ out st' t' (by omega) h₂
    rw [hr₁] at R
    have hc : (if a + 1 = k0 then 3 else 1) + (n + if n = 0 then 0 else 2) = n + 1 + 2 := by
      by_cases hn : n = 0
      · rw [ite_eq_left (show a + 1 = k0 by omega), ite_eq_left hn]
        omega
      · rw [ite_eq_right (show ¬ a + 1 = k0 by omega), ite_eq_right hn]
        omega
    have e : List.replicate (n + 1 + if n + 1 = 0 then 0 else 2) 2 =
        List.replicate (if a + 1 = k0 then 3 else 1) 2 ++ List.replicate (n + if n = 0 then 0 else 2) 2 := by
      rw [← List.replicate_add, hc, ite_eq_right (Nat.succ_ne_zero n)]
    refine ⟨hr.trans hr₁, ?_, e ▸ readsN_append R₁ (by rw [hk₁, List.length_replicate]) R⟩
    rw [hk, hk₁, ite_eq_right (Nat.succ_ne_zero n), Nat.add_assoc, hc]

/-- Masked Ligerito's coins at `sch`: the plain count's (`ligNs`), with the lane phase's `ρ` pair, two `K` coins, after
the `k0` lane challenges. -/
def zkLigNs (sch : Schedule) : List ℕ :=
  [sch.n, 1] ++ List.replicate (sch.k0 + 2) 2 ++ [sch.n - sch.k0 + 1, 1] ++ [sch.n - sch.k0 + 1, 1] ++
    qNs (sch.levels.getD 0 default) ++ [1] ++
    loopNs (levelNs sch) (levelDim sch) 0 (sch.levels.size - 1) (sch.n - sch.k0 + 1)

open FlockSoundness.Discharge.ZkExec in
theorem zkLigeritoWith_readsN (hs : Hashes Flock.V) (ms : MerkleScheme) (sch : Schedule) (T : F128)
    (cap : Array ByteArray) (p : LigeritoProof) (basisAt : Array F256 → ℕ → Array F256) (zl : Zk.Level0)
    (hn : sch.n ≠ 0) (hk0 : 1 ≤ sch.k0) (h2 : 2 ≤ sch.levels.size)
    (hq : ∀ i < sch.levels.size, (sch.levels.getD i default).q ≠ 0) {t t' : Transcript} {rc : Zk.L0Rec}
    (h : Zk.ligeritoWith hs ms sch T cap p basisAt zl t = .ok (rc, t')) :
    t'.rounds = t.rounds ∧ t'.k = t.k + (zkLigNs sch).length ∧ ReadsN t.rounds t.k (zkLigNs sch) := by
  unfold Zk.ligeritoWith at h
  simp only [liftTM_V, liftLM_V] at h
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨t₁, k₁, h⟩ := label_bind h
  obtain ⟨t₂, k₂, h⟩ := absorbF_bind h
  obtain ⟨t₃, k₃, h⟩ := absorbBytes_bind h
  obtain ⟨t₃', k₃', h⟩ := label_bind h
  obtain ⟨⟨rc', stF⟩, t₄, h₄, h⟩ := bind_ok h
  obtain ⟨-, rfl⟩ := pure_ok h
  rw [StateT.run] at h₄
  obtain ⟨n₅, t₅, hr₅, hk₅, h₄⟩ := lsampleVecWith_bind hs hn h₄
  obtain ⟨y₀, -, h₄⟩ := nextOod_bind h₄
  obtain ⟨_, t₆, h₆, h₄⟩ := lift_bind h₄
  have k₆ := absorbF_ok h₆
  obtain ⟨t₇, k₇, h₄⟩ := claimPow_bind h₄
  obtain ⟨n₈, bc, t₈, hr₈, hk₈, h₄⟩ := lsampleWith_bind hs h₄
  have h₄ := lmodify_bind h₄
  obtain ⟨m₀, t₉, -, k₉, h₄⟩ := nextMsg_bind h₄
  have h₄ := lmodify_bind h₄
  obtain ⟨lane, st₁₀, t₁₀, h₁₀, h₄⟩ := lbind_ok h₄
  rw [lforIn_range_eq] at h₁₀
  obtain ⟨hr₁₀, hk₁₀, R₁₀⟩ := lforIn_lanes_readsN _ sch.k0 (by
      intro i v st tt b st' tt' _ hb
      obtain ⟨nn₁, r, tt₁, hr₁, hk₁, hb⟩ := lsampleKWith_bind hs hb
      have hb := lmodify_bind hb
      try dsimp only at hb
      by_cases hj : i + 1 = sch.k0
      · rw [ite_eq_left (show (i + 1 == sch.k0) = true by simp [hj])] at hb
        obtain ⟨_, tt₂, h₂, hb⟩ := lift_bind hb
        have k₂ := absorbK_ok h₂
        obtain ⟨_, tt₃, h₃, hb⟩ := lift_bind hb
        have k₃ := absorbK_ok h₃
        obtain ⟨nn₄, ρ₀, tt₄, hr₄, hk₄, hb⟩ := lsampleKWith_bind hs hb
        obtain ⟨nn₅, ρ₁, tt₅, hr₅, hk₅, hb⟩ := lsampleKWith_bind hs hb
        try dsimp only at hb
        by_cases hρ : (!Zk.rhoIndependent (ρ₀, ρ₁)) = true
        · rw [ite_eq_left hρ] at hb
          obtain ⟨_, _, _, h0, _⟩ := lbind_ok hb
          cases h0
        rw [ite_eq_right hρ] at hb
        try simp only [pure_bind] at hb
        obtain ⟨_, tt₆, h₆, hb⟩ := lift_bind hb
        have k₆ := label_ok h₆
        obtain ⟨_, tt₇, h₇, hb⟩ := lift_bind hb
        have k₇ := absorbK_ok h₇
        obtain ⟨_, tt₈, h₈, hb⟩ := lift_bind hb
        have k₈ := absorbSlice_ok h₈
        have hb := lmodify_bind hb
        obtain ⟨m, tt₉, -, k₉, hb⟩ := nextMsg_bind hb
        have hb := lmodify_bind hb
        obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb
        rw [k₃.2, k₂.2, hr₁, k₃.1, k₂.1, hk₁] at nn₄
        rw [hr₄, k₃.2, k₂.2, hr₁, hk₄, k₃.1, k₂.1, hk₁] at nn₅
        rw [ite_eq_left hj]
        refine ⟨by rw [k₉.2, k₈.2, k₇.2, k₆.2, hr₅, hr₄, k₃.2, k₂.2, hr₁], ?_,
          readsN_cons nn₁ rfl (readsN_cons nn₄ rfl (readsN_cons nn₅ rfl (readsN_nil _ _))), _, rfl⟩
        rw [k₉.1, k₈.1, k₇.1, k₆.1, hk₅, hk₄, k₃.1, k₂.1, hk₁]
      · rw [ite_eq_right (show ¬ (i + 1 == sch.k0) = true by simp [hj])] at hb
        try simp only [pure_bind] at hb
        obtain ⟨m, tt₉, -, k₉, hb⟩ := nextMsg_bind hb
        have hb := lmodify_bind hb
        obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb
        rw [ite_eq_right hj]
        exact ⟨by rw [k₉.2, hr₁], by rw [k₉.1, hk₁], readsN_cons nn₁ rfl (readsN_nil _ _), _, rfl⟩)
    sch.k0 0 _ _ t₉ lane st₁₀ t₁₀ (by omega) h₁₀
  try dsimp only at h₄
  obtain ⟨_, t₁₁, h₁₁, h₄⟩ := lift_bind h₄
  have k₁₁ := absorbBytes_ok h₁₁
  obtain ⟨R₁₂, st₁₂, t₁₂, hr₁₂, hk₁₂, h₄⟩ := oodIntroWith_readsN hs (by omega) h₄
  obtain ⟨R₁₃, st₁₃, t₁₃, hr₁₃, hk₁₃, h₄⟩ := oodIntroWith_readsN hs (by omega) h₄
  obtain ⟨R₁₄, ⟨pos0, al0, e0⟩, st₁₄, t₁₄, hr₁₄, hk₁₄, h₄⟩ := queryLevel0With_readsN hs (hq 0 (by omega)) h₄
  dsimp only at h₄
  obtain ⟨extra, stX, tX, hX, h₄'⟩ := lbind_ok h₄
  clear h₄
  rw [lforIn_range_eq, List.forIn_pure_yield_eq_foldl] at hX
  obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hX
  obtain ⟨mq, t₁₅, -, k₁₅, h₄⟩ := nextMsg_bind h₄'
  obtain ⟨t₁₆, k₁₆, h₄⟩ := claimPow_bind h₄
  obtain ⟨n₁₇, bq, t₁₇, hr₁₇, hk₁₇, h₄⟩ := lsampleWith_bind hs h₄
  have h₄ := lmodify_bind h₄
  obtain ⟨out, st₁₈, t₁₈, hloop, hcont⟩ := lbind_ok h₄
  rw [lforIn_range_eq] at hloop
  have hL := lforIn_readsN_loop _ Prod.fst (levelNs sch) (levelDim sch) ?hbody (sch.levels.size - 1) 0 _ _ _ out
    st₁₈ t₁₈ hloop
  case hbody =>
    intro i s st tt b st' tt' hb
    by_cases c4 : s.1 < 4
    · rw [ite_eq_left c4] at hb; obtain ⟨_, _, _, h0, _⟩ := lbind_ok hb; cases h0
    rw [ite_eq_right c4] at hb
    obtain ⟨chs, st₁, tt₁, hf, hb⟩ := lbind_ok hb
    rw [lforIn_range_eq] at hf
    obtain ⟨hr₁, hk₁, R₁⟩ := lforIn_readsN _ 2 (by
        intro j s' st' tt' b' st'' tt'' hb'
        obtain ⟨hc, c, st₁, tt₁, hr₁, hk₁, hb'⟩ := challengeWith_readsN hs hb'
        obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb'
        exact ⟨hc, hr₁, hk₁, _, rfl⟩) 4 0 _ _ _ chs st₁ tt₁ hf
    by_cases hi : i + 1 < sch.levels.size - 1
    · rw [ite_eq_left hi] at hb
      obtain ⟨_, tt₂, h₂, hb⟩ := lift_bind hb
      have k₂ := absorbBytes_ok h₂
      obtain ⟨R₃, st₃, tt₃, hr₃, hk₃, hb⟩ := oodIntroWith_readsN hs (by omega) hb
      obtain ⟨R₄, st₄, tt₄, hr₄, hk₄, hb⟩ := oodIntroWith_readsN hs (by omega) hb
      obtain ⟨R₅, r₅, st₅, tt₅, hr₅, hk₅, hb⟩ := queryLevelWith_readsN hs (hq (i + 1) (by omega)) hb
      obtain ⟨mm, tt₆, -, k₆, hb⟩ := nextMsg_bind hb
      obtain ⟨tt₇, k₇, hb⟩ := claimPow_bind hb
      obtain ⟨n₈, bb, tt₈, hr₈, hk₈, hb⟩ := lsampleWith_bind hs hb
      have hb := lmodify_bind hb
      by_cases c4' : s.1 - 4 + 1 < 4
      · rw [ite_eq_left c4'] at hb; obtain ⟨_, _, _, h0, _⟩ := lbind_ok hb; cases h0
      rw [ite_eq_right c4'] at hb
      obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb
      have e₂r : tt₂.rounds = tt.rounds := by rw [k₂.2, hr₁]
      have e₂k : tt₂.k = tt.k + 4 := by rw [k₂.1, hk₁]
      rw [e₂r, e₂k] at R₃
      rw [hr₃, e₂r, hk₃, e₂k] at R₄
      rw [hr₄, hr₃, e₂r, hk₄, hk₃, e₂k] at R₅
      have e₇r : tt₇.rounds = tt.rounds := by rw [k₇.2, k₆.2, hr₅, hr₄, hr₃, e₂r]
      have e₇k : tt₇.k = tt.k + 4 + 2 + 2 + 2 := by rw [k₇.1, k₆.1, hk₅, hk₄, hk₃, e₂k]
      rw [e₇r, e₇k] at n₈
      rw [levelNs, ite_eq_left hi]
      refine ⟨by rw [hr₈, e₇r], ?_, ?_, _, rfl, by rw [levelDim, ite_eq_left hi]⟩
      · rw [hk₈, e₇k]
        simp only [qNs, List.length_append, List.length_replicate, List.length_cons, List.length_nil]
      · refine readsN_append (readsN_append (readsN_append (readsN_append R₁ ?_ R₃) ?_ R₄) ?_ R₅) ?_
          (readsN_cons n₈ rfl (readsN_nil _ _)) <;>
        simp only [qNs, List.length_append, List.length_replicate, List.length_cons, List.length_nil]
    · rw [ite_eq_right hi] at hb
      have hb := lget_bind hb
      by_cases hmc₀ : (st₁.msg != p.messages.size) = true
      · rw [ite_eq_left hmc₀] at hb; obtain ⟨_, _, _, h0, _⟩ := lbind_ok hb; cases h0
      rw [ite_eq_right hmc₀] at hb
      by_cases hyr : (p.finalProof.yr.size != 2 ^ (s.1 - 4 + 1)) = true
      · rw [ite_eq_left hyr] at hb; obtain ⟨_, _, _, h0, _⟩ := lbind_ok hb; cases h0
      rw [ite_eq_right hyr] at hb
      obtain ⟨_, st₂, tt₂, h₂, hb⟩ := lbind_ok hb
      rw [← Array.forIn_toList] at h₂
      obtain ⟨-, k₂⟩ := forIn_absorbF_keeps _ h₂
      obtain ⟨R₃, r₃, st₃, tt₃, hr₃, hk₃, hb⟩ :=
        queryLevelWith_readsN hs (hq (sch.levels.size - 1) (by omega)) hb
      obtain ⟨tt₄, k₄, hb⟩ := claimPow_bind hb
      obtain ⟨n₅, bb, tt₅, hr₅, hk₅, hb⟩ := lsampleWith_bind hs hb
      have hb := lmodify_bind hb
      obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb
      rw [k₂.2, hr₁, k₂.1, hk₁] at R₃
      have e₄r : tt₄.rounds = tt.rounds := by rw [k₄.2, hr₃, k₂.2, hr₁]
      have e₄k : tt₄.k = tt.k + 4 + 2 := by rw [k₄.1, hk₃, k₂.1, hk₁]
      rw [e₄r, e₄k] at n₅
      rw [levelNs, ite_eq_right hi]
      refine ⟨by rw [hr₅, e₄r], ?_, ?_, _, rfl, by rw [levelDim, ite_eq_right hi]⟩
      · rw [hk₅, e₄k]
        simp only [qNs, List.length_append, List.length_replicate, List.length_cons, List.length_nil]
      · refine readsN_append (readsN_append R₁ ?_ R₃) ?_ (readsN_cons n₅ rfl (readsN_nil _ _)) <;>
        simp only [qNs, List.length_append, List.length_replicate, List.length_cons, List.length_nil]
  obtain ⟨hrL, hkL, RL⟩ := hL
  obtain rfl := finalProgZ_ok hcont
  have hk0' : ¬ sch.k0 = 0 := by omega
  rw [ite_eq_right hk0'] at hk₁₀ R₁₀
  have e₃r : t₃'.rounds = t.rounds := by rw [k₃'.2, k₃.2, k₂.2, k₁.2]
  have e₃k : t₃'.k = t.k := by rw [k₃'.1, k₃.1, k₂.1, k₁.1]
  have e₇r : t₇.rounds = t.rounds := by rw [k₇.2, k₆.2, hr₅, e₃r]
  have e₇k : t₇.k = t.k + 1 := by rw [k₇.1, k₆.1, hk₅, e₃k]
  have e₉r : t₉.rounds = t.rounds := by rw [k₉.2, hr₈, e₇r]
  have e₉k : t₉.k = t.k + 2 := by rw [k₉.1, hk₈, e₇k]
  have e₁₁r : t₁₁.rounds = t.rounds := by rw [k₁₁.2, hr₁₀, e₉r]
  have e₁₁k : t₁₁.k = t.k + 2 + (sch.k0 + 2) := by rw [k₁₁.1, hk₁₀, e₉k]
  have e₁₆r : t₁₆.rounds = t.rounds := by rw [k₁₆.2, k₁₅.2, hr₁₄, hr₁₃, hr₁₂, e₁₁r]
  have e₁₆k : t₁₆.k = t.k + 2 + (sch.k0 + 2) + 2 + 2 + 2 := by rw [k₁₆.1, k₁₅.1, hk₁₄, hk₁₃, hk₁₂, e₁₁k]
  rw [e₃r, e₃k] at n₅
  rw [e₇r, e₇k] at n₈
  rw [e₉r, e₉k] at R₁₀
  rw [e₁₁r, e₁₁k] at R₁₂
  rw [hr₁₂, e₁₁r, hk₁₂, e₁₁k] at R₁₃
  rw [hr₁₃, hr₁₂, e₁₁r, hk₁₃, hk₁₂, e₁₁k] at R₁₄
  rw [e₁₆r, e₁₆k] at n₁₇
  rw [hr₁₇, e₁₆r, hk₁₇, e₁₆k] at RL
  refine ⟨by rw [hrL, hr₁₇, e₁₆r], ?_, ?_⟩
  · rw [hkL, hk₁₇, e₁₆k]
    simp only [zkLigNs, qNs, List.length_append, List.length_replicate, List.length_cons, List.length_nil]
    omega
  · refine readsN_append (readsN_append (readsN_append (readsN_append (readsN_append (readsN_append
      (readsN_cons n₅ rfl (readsN_cons n₈ rfl (readsN_nil _ _))) ?_ R₁₀) ?_ R₁₂) ?_ R₁₃) ?_ R₁₄) ?_
        (readsN_cons n₁₇ rfl (readsN_nil _ _))) ?_ RL <;>
      simp only [qNs, List.length_append, List.length_replicate, List.length_cons, List.length_nil] <;> omega

/-! ## One masked rep -/

/-- A masked rep's coins at `m`, `kLog`, `claims` claims and the `--zk` schedule `sch`: the replay's, the ring
switch's, masked Ligerito's and the inner proof's at its ten constraints (the replay's three, the ring switch's two,
the lane phase's two and the triple's three). -/
def zkRepNs (m kLog claims : ℕ) (sch : Schedule) : List ℕ :=
  zcNs m ++ lcNs kLog ++ rsNs claims ++ zkLigNs sch ++ innerNs 7

open FlockSoundness.Discharge.ZkExec in
/-- **A masked rep's rounds issued its statement's coins**: when `Zk.verifyRepWith hs` accepts rep `r` of table `i` at
the table's `--zk` schedule, whatever the hashes `hs` are, its stream's rounds issued `zkRepNs` at `m`, `kLog` and
`2 + |extras|` claims, one count per round. -/
theorem zkVerifyRepWith_readsN (hs : Hashes Flock.V) {spec : SessionSpec} {st : Setup} {sch : Schedule} {tPad : ℕ}
    {sess : RunRecord} {i r : ℕ} {bytes c : ByteArray} (hsch : Zk.schedule st.m = .ok (sch, tPad))
    (h : Zk.verifyRepWith hs spec st sch tPad sess i r bytes = .ok c) :
    ∃ (stream : StreamRec) (extras : Array Claim),
      sess.streams.find? (fun x => x.name == toString spec.tables[i]! ++ toString "/rep" ++ toString r) = some stream ∧
      st.extra sess.points = .ok extras ∧
      stream.rounds.toList.map Round.n = zkRepNs st.m st.kLog (2 + extras.size) sch := by
  obtain ⟨h25, -, hsm, hk0, -, hdim, hlv, -, -⟩ := zk_schedule_seams _ _ _ hsch
  obtain ⟨hn7, h2, -, -, -, -, -⟩ := hdim
  unfold Zk.verifyRepWith at h
  dsimp only at h
  cases hs' : Array.find? (fun x => x.name == toString spec.tables[i]! ++ toString "/rep" ++ toString r)
      sess.streams with
  | none => rw [hs'] at h; cases h
  | some stream =>
  rw [hs'] at h
  obtain ⟨extras, hex, h⟩ := ebind_ok h
  obtain ⟨zp, -, h⟩ := ebind_ok h
  by_cases c1 : (zp.proof.params != pcsParams st.merkle sch) = true
  · rw [ite_eq_left c1] at h; obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  rw [ite_eq_right c1] at h
  by_cases c1' : (zp.inner.root.size != 64) = true
  · rw [ite_eq_left c1'] at h; obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  rw [ite_eq_right c1'] at h
  obtain ⟨⟨u, tr⟩, hrun, h⟩ := ebind_ok h
  change (if (tr.k != stream.rounds.size) = true then _ else _) = _ at h
  by_cases c2 : (tr.k != stream.rounds.size) = true
  · rw [ite_eq_left c2] at h; obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  simp only [bne_iff_ne, ne_eq, not_not] at c2
  refine ⟨stream, extras, rfl, hex, ?_⟩
  rw [StateT.run] at hrun
  obtain ⟨rp, t₁, h₁, hrun⟩ := bind_ok hrun
  obtain ⟨hr₁, hk₁, R₁, hc₁⟩ := replayWith_readsN hs (by omega) h₁
  try dsimp only at hrun
  obtain ⟨⟨T, sws⟩, t₂, h₂, hrun⟩ := bind_ok hrun
  obtain ⟨hr₂, hk₂, R₂⟩ := ringSwitchWith_readsN hs (by simp) h₂
  obtain ⟨rc, t₃, h₃, hrun⟩ := bind_ok hrun
  obtain ⟨hr₃, hk₃, R₃⟩ := zkLigeritoWith_readsN hs _ sch _ _ _ _ _ (by omega) hk0 h2
    (fun i hi => (hlv i hi).q_ne_zero) h₃
  cases hv : Zk.ligeritoConstraints (Zk.PadLayout.new st.m st.kLog sch.k0) rc zp.proof.ligerito zp.level0 with
  | error e =>
    rw [hv] at hrun
    try dsimp only at hrun
    obtain ⟨_, _, h₄, _⟩ := bind_ok hrun
    cases h₄
  | ok v =>
  rw [hv] at hrun
  try dsimp only at hrun
  obtain ⟨lg, t₄, h₄, hrun'⟩ := bind_ok hrun
  clear hrun
  obtain ⟨rfl, rfl⟩ := pure_ok h₄
  obtain ⟨-, hk₅, R₅⟩ := verifyInnerWith_readsN hs hrun'
  have hsz : (#[({ (rp.claims[0]!) with value := Zk.claimCheck rp.claims[0]! zp.proof.ringSwitches[0]! } : Claim),
      { (rp.claims[1]!) with value := Zk.claimCheck rp.claims[1]! zp.proof.ringSwitches[1]! }] ++ extras).size =
      2 + extras.size := by simp [Nat.add_comm]
  rw [hsz] at R₂ hk₂
  have hcons : (rp.cons ++ #[("ring switch ab", rp.abValue.addConst (Zk.claimCheck rp.claims[0]!
      zp.proof.ringSwitches[0]!)), ("ring switch c", rp.cValue.addConst (Zk.claimCheck rp.claims[1]!
      zp.proof.ringSwitches[1]!))] ++ lg).size = 7 := by
    rw [Array.size_append, Array.size_append, hc₁, ligeritoConstraints_size hv]
    rfl
  rw [hcons] at R₅ hk₅
  rw [hr₁] at R₂
  rw [hr₂, hr₁] at R₃
  rw [hr₃, hr₂, hr₁] at R₅
  have R := readsN_append (readsN_append (readsN_append R₁ hk₁ R₂)
    (by rw [hk₂, hk₁]; simp only [List.length_append]; omega) R₃)
    (by rw [hk₃, hk₂, hk₁]; simp only [List.length_append]; omega) R₅
  refine map_n_of_readsN R ?_
  rw [← c2, hk₅, hk₃, hk₂, hk₁]
  simp only [Transcript.init, zkRepNs, List.length_append]
  omega

/-! ## The masked session -/

/-- A masked rep's coins at its table's statement: `zkRepNs` at the table's `m`, `kLog`, claims and `--zk` schedule,
none where the schedule refuses `m`. -/
def zkRepNsOf (spec : SessionSpec) (st : Stmt) : List ℕ :=
  match Zk.schedule st.m with
  | .ok (sch, _) => zkRepNs st.m st.c.kLog (2 + spec.points * st.regions.size) sch
  | .error _ => []

/-- **Each masked rep stream's rounds issued its table's coins**: when the masked verdict accepts the record at any
hashes and `liveOf` reads it, there are two rep streams per table, and the rounds of stream `k` (table `k / 2`'s rep
`k % 2`) issued `zkRepNsOf spec sts[k / 2]`, one count per round. -/
theorem streams_ns_zk (hs : Hashes Flock.V) {spec : SessionSpec} {sts : Array Stmt} {record : String}
    {proofs : Array ByteArray} {lv : Live} (hv : Zk.verifyWithM hs spec (tabsOf sts) record proofs = .ok ())
    (hl : liveOf spec record = .ok lv) :
    lv.streams.size = 2 * sts.size ∧
      ∀ (k : ℕ) (hk : k < lv.streams.size), lv.streams[k].toList.map Round.n = zkRepNsOf spec sts[k / 2]! := by
  obtain ⟨hsz, hall⟩ := rep_zk hs hv hl
  refine ⟨hsz, fun k hk => ?_⟩
  obtain ⟨sess, sch, tPad, stream, c, -, hpn, hf, hrep, hfind, hfa⟩ := hall k hk
  obtain ⟨stream', extras, hfind', hex, hns⟩ := zkVerifyRepWith_readsN hs hf hrep
  rw [hfind] at hfind'
  cases hfind'
  have hx : extras.size = sts[k / 2]!.regions.size * sess.points.size := extraClaims_size hex
  rw [map_n_of_forall₂ hfa, hns, hx, hpn, zkRepNsOf]
  simp only [hf]
  rw [Nat.mul_comm]
  rfl

/-- The masked session's coins, round by round in `spec.repStreams` order: each table's two reps' `zkRepNsOf`. -/
def zkNs (spec : SessionSpec) (sts : Array Stmt) : List ℕ :=
  (List.range (2 * sts.size)).flatMap fun k => zkRepNsOf spec sts[k / 2]!

/-- **An accepted masked record's rounds issued the session's coins**: when the masked verdict accepts the record at
any hashes and `liveOf` reads it, its rounds, in `spec.repStreams` order, issued `zkNs spec sts`, one count per
round. -/
theorem rounds_ns_zk (hs : Hashes Flock.V) {spec : SessionSpec} {sts : Array Stmt} {record : String}
    {proofs : Array ByteArray} {lv : Live} (hv : Zk.verifyWithM hs spec (tabsOf sts) record proofs = .ok ())
    (hl : liveOf spec record = .ok lv) :
    (lv.streams.toList.flatMap (·.toList)).map Round.n = zkNs spec sts := by
  obtain ⟨hsz, hall⟩ := streams_ns_zk hs hv hl
  have e : lv.streams.toList = (List.range lv.streams.size).map fun k => lv.streams[k]! := by
    refine List.ext_getElem (by simp) fun i h₁ h₂ => ?_
    simp only [Array.length_toList] at h₁
    simp [getElem!_pos lv.streams i h₁]
  rw [List.map_flatMap, e, List.flatMap_map, zkNs, ← hsz, List.flatMap_def, List.flatMap_def]
  refine congrArg List.flatten (List.map_congr_left fun k hk => ?_)
  have hk' : k < lv.streams.size := List.mem_range.1 hk
  rw [getElem!_pos lv.streams k hk']
  exact hall k hk'

/-- **Custody reads each masked round's coin at the statement's count**: where the masked verdict accepts the record at
any hashes and its custody passes on the round messages `mids`, each message's coin is read at its round's count in
`zkNs spec sts`, by a recorded round of that count carrying the message. -/
theorem custody_vals_zk (hs : Hashes Flock.V) {spec : SessionSpec} {sts : Array Stmt} {record : String}
    {proofs : Array ByteArray} {lv : Live} (hv : Zk.verifyWithM hs spec (tabsOf sts) record proofs = .ok ())
    (hl : liveOf spec record = .ok lv) {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} (hc : custody spec lv link c₁ mids = .ok ()) :
    List.Forall₂ (fun (n : ℕ) (am : Claims.Msg × Coin) =>
      ∃ rd : Round, rd.n = n ∧ rd.bytes = some ⟨am.1.toArray⟩ ∧ rd.coins = am.2.vals n) (zkNs spec sts) mids := by
  rw [← rounds_ns_zk hs hv hl, List.forall₂_map_left_iff]
  exact (Claims.Bridge.Custody.custody_ok hc).2.2.2.imp fun rd am h => ⟨rd, rfl, h⟩

end FlockDeploy
