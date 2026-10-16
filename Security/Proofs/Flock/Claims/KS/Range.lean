import Proofs.Flock.Claims.KS.Deploy
import Proofs.Flock.Soundness.Discharge.Exec.Tables
import Proofs.Flock.Soundness.Discharge.ZkExec.Verify

/-!
# `k ≥ 8` wherever the deployed verifier accepts, at SHA-512

At `H = SHA-512` the deployed verifier accepts only by a decision (`next_decide_sha512`), where the shipped `verify` (or
`Zk.verify`) accepts the session's tables. Then there is a table, and every table's `m` is one the verifier serves:
`fast100`'s `22 ≤ m ≤ 35`, or `Zk.schedule`'s `25 ≤ m ≤ 35` (`served_plain`, `served_zk`, from `verify_ok_tables` and
`zk_verify_ok`). There table 0's level-0 codeword has `2^d ≥ 2^10` positions (`plainShape_table`, `zkShape_table`), so
`N₀ ≥ 2^10` and `k = ⌈4N₀/e⌉ ≥ 1507` (`Numbers.k_ge`). So at every statement `k ≥ 8` or the verifier accepts no run
(`k_range_plain`, `k_range_zk`), and KS at the deployed instances asks only one wait's bound
(`plain_of_oneWait_sha512`, `zk_of_oneWait_sha512`).
-/

namespace FlockDeploy.KS

open Flock

/-- **A run accepts only at an accepting reply**: some prover messages on which the verifier's step returns `.acc`. -/
theorem stepWith_acc {Ω : Type} (H : Claims.Msg → Claims.Msg) (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω)
    (vnext : List Claims.Msg → Claims.OC Claims.Out) :
    ∀ (n : ℕ) (hist : List (Claims.Msg × Claims.Msg)) (as : List Claims.Msg) (t : Claims.Trace),
      (Claims.stepWith H P x ω vnext n hist as t).out = .acc → ∃ as', ((vnext as').eval H).1 = .acc := by
  intro n
  induction n with
  | zero => intro hist as t h; simp [Claims.stepWith] at h
  | succ n ih =>
    intro hist as t h
    simp only [Claims.stepWith] at h
    split at h
    · exact ih _ _ _ h
    · rename_i heq
      exact ⟨_, heq⟩
    · simp at h

/-- **An accepting step at SHA-512 is an accepting verdict** at the statement's setup. -/
theorem next_acc_sha512 {zk : Bool} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    (h : ((next zk x as cs).eval sha512).1 = .acc) :
    ∃ s spec sts record proofs, setupOf zk x = some (s, spec, sts) ∧
      (if zk then Zk.verify spec (tabsOf sts) record proofs else verify spec (tabsOf sts) record proofs) = .ok () := by
  cases ha : act zk x as cs with
  | rej => simp [next_rej sha512 ha] at h
  | coin c => simp [next_coin sha512 ha] at h
  | decide s spec sts link c₁ mids f =>
    exact ⟨s, spec, sts, f.record, f.proofs, act_decide ha, ((next_decide_sha512 ha).1 h).1⟩

theorem fast100_range {m : ℕ} {s : Flock.Schedule} (h : Flock.fast100 m = some s) : 22 ≤ m ∧ m ≤ 35 :=
  ⟨FlockSoundness.Refine.exec_fast100_ge h, by
    by_contra hm
    have hb : (m < 22 || m > 35) = true := by
      simp only [Bool.or_eq_true, decide_eq_true_eq]
      omega
    unfold Flock.fast100 at h
    simp [hb] at h⟩

theorem queries0_range {m q : ℕ} (h : Zk.queries0 m = .ok q) : 25 ≤ m ∧ m ≤ 35 := by
  unfold Zk.queries0 at h
  split_ifs at h <;> first | (cases h; done) | (simp only [beq_iff_eq] at *; omega)

theorem schedule_range {m : ℕ} {p : Flock.Schedule × ℕ} (h : Zk.schedule m = .ok p) : 25 ≤ m ∧ m ≤ 35 := by
  unfold Zk.schedule at h
  obtain ⟨q, hq, -⟩ := FlockSoundness.Refine.ebind_ok h
  exact queries0_range hq

/-- **The plain verifier accepts only served tables**: the session sets up a table, and `fast100` serves every table's
`m`. -/
theorem served_plain {spec : SessionSpec} {sts : Array Stmt} {record : String} {proofs : Array ByteArray}
    (h : verify spec (tabsOf sts) record proofs = .ok ()) :
    0 < sts.size ∧ ∀ (i : ℕ) (hi : i < sts.size), 22 ≤ sts[i].m ∧ sts[i].m ≤ 35 := by
  obtain ⟨_, schs, -, -, hpos, -, hall⟩ := FlockSoundness.Discharge.Exec.verify_ok_tables h
  have hsz : (tabsOf sts).size = sts.size := Array.size_map ..
  refine ⟨hsz ▸ hpos, fun i hi => ?_⟩
  obtain ⟨hs, -, hf, -⟩ := hall i (hsz ▸ hi)
  have hm : (tabsOf sts)[i].m = sts[i].m := by simp [tabsOf, Setup.ofCircuit]
  exact hm ▸ fast100_range hf

/-- **The masked verifier accepts only served tables**: the session sets up a table, and `Zk.schedule` serves every
table's `m`. -/
theorem served_zk {spec : SessionSpec} {sts : Array Stmt} {record : String} {proofs : Array ByteArray}
    (h : Zk.verify spec (tabsOf sts) record proofs = .ok ()) :
    0 < sts.size ∧ ∀ (i : ℕ) (hi : i < sts.size), 25 ≤ sts[i].m ∧ sts[i].m ≤ 35 := by
  obtain ⟨-, _, schs, -, -, hpos, -, -, hall⟩ := FlockSoundness.Discharge.ZkExec.zk_verify_ok _ _ _ _ h
  have hsz : (tabsOf sts).size = sts.size := Array.size_map ..
  refine ⟨hsz ▸ hpos, fun i hi => ?_⟩
  obtain ⟨hs, -, hf, -⟩ := hall i (hsz ▸ hi)
  have hm : (tabsOf sts)[i].m = sts[i].m := by simp [tabsOf, Setup.ofCircuit]
  exact hm ▸ schedule_range hf

theorem zkLen_ge : ∀ i : Claims.Flock.ZkIdx, 10 ≤ Claims.Flock.zkLen i := by
  decide +kernel

/-- `N₀` at a statement that sets up a table: table 0's level-0 codeword length. -/
theorem N₀_of_setup {zk : Bool} {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (h : setupOf zk x = some (s, spec, sts)) (h0 : 0 < sts.size) :
    N₀ zk x = 2 ^ (if zk then Claims.Flock.zkLevel0 sts[0].m else Claims.Flock.plainLevel0 sts[0].m).d := by
  unfold N₀ tables
  rw [h]
  rcases sts with ⟨_ | ⟨a, l⟩⟩
  · simp at h0
  · rfl

/-- `k ≥ 8` once `N₀ ≥ 2^10`. -/
theorem eight_le_k {Col T : Type} {D : Claims.Flock.Deploy Coin Col T} {x : Claims.Msg} (hN : 2 ^ 10 ≤ D.N₀ x) :
    8 ≤ D.k x := by
  have := Claims.Flock.Numbers.k_ge hN
  unfold Claims.Flock.Deploy.k
  omega

theorem two_pow_le {d : ℕ} (hd : 10 ≤ d) : (2 : ℝ) ^ 10 ≤ 2 ^ d := pow_le_pow_right₀ (by norm_num) hd

/-- An accepting run at SHA-512 of the deployed verifier, as an accepting verdict at the statement's setup. -/
theorem acc_verdict {zk : Bool} {x : Claims.Msg} (hacc : ¬ Claims.Flock.NoAcc sha512 (V zk) x) :
    ∃ s spec sts record proofs, setupOf zk x = some (s, spec, sts) ∧
      (if zk then Zk.verify spec (tabsOf sts) record proofs else verify spec (tabsOf sts) record proofs) = .ok () := by
  unfold Claims.Flock.NoAcc at hacc
  push Not at hacc
  obtain ⟨Ω, P, ω, r, hr⟩ := hacc
  obtain ⟨as, has⟩ := stepWith_acc _ _ _ _ _ _ _ _ _ hr
  exact next_acc_sha512 has

/-- **The plain verifier's `k` range**: at every statement, `k ≥ 8` or the verifier accepts no run at SHA-512. -/
theorem k_range_plain (rel : Readable.Parsed → Readable.Relation) (x : Claims.Msg) :
    8 ≤ (deployPlain rel).k x ∨ Claims.Flock.NoAcc sha512 (deployPlain rel).V x := by
  by_cases hacc : Claims.Flock.NoAcc sha512 (deployPlain rel).V x
  · exact .inr hacc
  obtain ⟨s, spec, sts, record, proofs, hs, hv⟩ := acc_verdict (zk := false) hacc
  obtain ⟨h0, hm⟩ := served_plain (by simpa using hv)
  obtain ⟨h22, h35⟩ := hm 0 h0
  have hd := (Claims.Flock.plainShape_table ⟨sts[0].m - 22, by omega⟩)
  have hi : Claims.Flock.PlainIdx.m ⟨sts[0].m - 22, by omega⟩ = sts[0].m := by
    simp only [Claims.Flock.PlainIdx.m]
    omega
  have hlen : 10 ≤ (Claims.Flock.plainLevel0 sts[0].m).d := by
    unfold Claims.Flock.plainLen Claims.Flock.plainCols at hd
    rw [hi] at hd
    omega
  refine .inl (eight_le_k ?_)
  show 2 ^ 10 ≤ N₀ false x
  rw [N₀_of_setup hs h0]
  exact two_pow_le hlen

/-- **The masked verifier's `k` range**: at every statement, `k ≥ 8` or the verifier accepts no run at SHA-512. -/
theorem k_range_zk (rel : Readable.Parsed → Readable.Relation) (x : Claims.Msg) :
    8 ≤ (deployZk rel).k x ∨ Claims.Flock.NoAcc sha512 (deployZk rel).V x := by
  by_cases hacc : Claims.Flock.NoAcc sha512 (deployZk rel).V x
  · exact .inr hacc
  obtain ⟨s, spec, sts, record, proofs, hs, hv⟩ := acc_verdict (zk := true) hacc
  obtain ⟨h0, hm⟩ := served_zk (by simpa using hv)
  obtain ⟨h25, h35⟩ := hm 0 h0
  have hd := zkLen_ge ⟨sts[0].m - 25, by omega⟩
  have hi : Claims.Flock.ZkIdx.m ⟨sts[0].m - 25, by omega⟩ = sts[0].m := by
    simp only [Claims.Flock.ZkIdx.m]
    omega
  have hlen : 10 ≤ (Claims.Flock.zkLevel0 sts[0].m).d := by
    unfold Claims.Flock.zkLen at hd
    rw [hi] at hd
    exact hd
  refine .inl (eight_le_k ?_)
  show 2 ^ 10 ≤ N₀ true x
  rw [N₀_of_setup hs h0]
  exact two_pow_le hlen

/-- **The deployed plain verifier at SHA-512: KS from one wait's bound**, with the `k` range discharged. -/
theorem plain_of_oneWait_sha512 (rel : Readable.Parsed → Readable.Relation)
    (h : Claims.Flock.OneWait sha512 (deployPlain rel)) :
    Claims.KS sha512 (deployPlain rel).V (Claims.Flock.R sha512 (deployPlain rel).parse)
      (Claims.Flock.E sha512 (deployPlain rel)
        (Claims.Flock.plainDecodeD Claims.Flock.Numbers.agree₀ ∘ Claims.Flock.dedupPos))
      (Claims.Flock.κ (deployPlain rel)) (Claims.Flock.γ (deployPlain rel)) :=
  plain_of_oneWait sha512 rel (k_range_plain rel) h

/-- **The deployed masked (`--zk`) verifier at SHA-512: KS from one wait's bound**, with the `k` range discharged. -/
theorem zk_of_oneWait_sha512 (rel : Readable.Parsed → Readable.Relation)
    (h : Claims.Flock.OneWait sha512 (deployZk rel)) :
    Claims.KS sha512 (deployZk rel).V (Claims.Flock.R sha512 (deployZk rel).parse)
      (Claims.Flock.E sha512 (deployZk rel) (Claims.Flock.zkDecodeD Claims.Flock.zkAgree ∘ Claims.Flock.dedupPos))
      (Claims.Flock.κ (deployZk rel)) (Claims.Flock.γ (deployZk rel)) :=
  zk_of_oneWait sha512 rel (k_range_zk rel) h

end FlockDeploy.KS
