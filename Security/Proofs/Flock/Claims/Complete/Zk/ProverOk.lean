import Proofs.Flock.Claims.Complete.Zk.Json.Files
import Proofs.Flock.Claims.Complete.Replay.ProverOk

/-!
# The masked prover's pieces at every good case (`ProverOkZk`)

`proverOk` at the masked record: at an oracle of 64-byte outputs (`Out64`), `ProverOkZk` from its rep half alone,
`RepRoundsZk` (each masked rep on the table's extra claims records a round, each of at most `65536` coins).

* The masked level-0 commitment's cap is non-empty and under `2^64` bytes (`commitPZk_cap`): the masked table commits
  under `hm96-sha512/v1` (`CaseZk.hm96`), whose cap is `2^cap` digests of 64 bytes (`commitRowsP_capSize`), and the
  `--zk` schedule's level 0 caps at most its depth and at most `32` (`zkSchedule_cap0`).
-/

namespace FlockComplete.JsonRT

open Flock Transcript FlockComplete FlockDeploy

/-- The `--zk` schedule's level 0 caps at most its depth, and at most `32`. -/
theorem zkSchedule_cap0 {m : ℕ} (h1 : 25 ≤ m) (h2 : m ≤ 35) {v : Schedule × ℕ} (h : Zk.schedule m = .ok v) :
    (v.1.levels.getD 0 default).cap ≤ (v.1.levels.getD 0 default).d ∧ (v.1.levels.getD 0 default).cap ≤ 32 := by
  have key : ∀ k : Fin 11, (match Zk.schedule (k.val + 25) with
      | .ok v => decide ((v.1.levels.getD 0 default).cap ≤ (v.1.levels.getD 0 default).d ∧
          (v.1.levels.getD 0 default).cap ≤ 32)
      | .error _ => false) = true := by
    decide +kernel
  have := key ⟨m - 25, by omega⟩
  simp only [show m - 25 + 25 = m by omega, h, decide_eq_true_eq] at this
  exact this

/-- **A level's cap under `hm96-sha512/v1` has `2^cap` digests**, for a SHA-512 of 64-byte outputs, at a level that
caps at most its depth. -/
theorem commitRowsP_capSize {h : ByteArray → ByteArray} (h64 : ∀ b, (h b).size = 64) (lv : Level)
    (hcap : lv.cap ≤ lv.d) (rows : Array (Array F128)) (salt : ℕ → ByteArray) {c : Commit}
    (hc : (commitRowsP (Hashes.ofSha512 h) MerkleScheme.hm96Sha512 lv rows salt : V Commit) = .ok c) :
    c.cap.size = 2 ^ lv.cap ∧ ∀ b ∈ c.cap, b.size = 64 := by
  unfold commitRowsP at hc
  rw [hm96_ops] at hc
  obtain ⟨leaves, hl, hc⟩ := except_ok_bind hc
  obtain ⟨s, hs, hc⟩ := except_ok_bind hc
  cases hc
  have hL := mapM_sized (fun _ _ hr => hm96_leaf h64 _ _ hr) hl
  rw [Array.size_range] at hL
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hs
  have hfin := except_forIn_count (fun n (b : Array (Array ByteArray) × Array ByteArray) =>
    b.2.size = 2 ^ (lv.cap + n) ∧ ∀ x ∈ b.2, x.size = 64) ?_ _ _ _ ?_ hs
  · simpa using hfin
  · rintro - ⟨layers, cur⟩ r n ⟨h1, -⟩ hr
    obtain ⟨cur', hc', hr⟩ := except_ok_bind hr
    cases hr
    have hN := mapM_sized (fun _ _ hr => hm96_node h64 _ _ hr) hc'
    rw [Array.size_range, h1, ← Nat.add_assoc, Nat.pow_succ, Nat.mul_div_cancel _ (by norm_num)] at hN
    exact ⟨_, rfl, hN⟩
  · refine ⟨?_, hL.2⟩
    rw [hL.1]
    simp only [List.length_range', Std.Legacy.Range.size]
    congr 1
    omega

/-- **Level 0's masked commitment's cap is non-empty and under `2^64` bytes**, for a SHA-512 of 64-byte outputs, at
the `--zk` schedule's level 0. -/
theorem commitPZk_cap {h : ByteArray → ByteArray} (h64 : ∀ b, (h b).size = 64) {m : ℕ} (h1 : 25 ≤ m) (h2 : m ≤ 35)
    {v : Schedule × ℕ} (hv : Zk.schedule m = .ok v) {g : Array F128} {extra lanePad : ℕ → ℕ → F128}
    {salt : ℕ → ByteArray} {cm : Commit}
    (hc : commitPZk (Hashes.ofSha512 h : Hashes V) MerkleScheme.hm96Sha512 (v.1.levels.getD 0 default) v.2 g extra
      lanePad salt = .ok cm) :
    0 < (flat cm.cap).size ∧ (flat cm.cap).size < 2 ^ 64 := by
  obtain ⟨hcd, h32⟩ := zkSchedule_cap0 h1 h2 hv
  obtain ⟨hsz, hw⟩ := commitRowsP_capSize h64 _ hcd _ _ hc
  rw [FlockComplete.Replay.flat_size_of hw, hsz]
  refine ⟨Nat.mul_pos (by norm_num) (Nat.two_pow_pos _), ?_⟩
  calc 64 * 2 ^ (v.1.levels.getD 0 default).cap ≤ 64 * 2 ^ 32 :=
        Nat.mul_le_mul_left _ (Nat.pow_le_pow_right (by norm_num) h32)
    _ < 2 ^ 64 := by norm_num

variable {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}

/-- **Each masked rep records a round, each of at most `65536` coins**: `ProverOkZk`'s rep half, at every good masked
case's table, on the table's extra claims at the link's points. -/
def RepRoundsZk (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : Prop :=
  ∀ c : CaseZk H rel, ∀ st ∈ c.sts,
    ∀ v, Zk.schedule (Setup.ofCircuit st).m = .ok v →
      ∀ rep dom tr tb cm₀ extras pads ic extra lanePad salt nonce prior ws,
      (Setup.ofCircuit st).extra (ptsOf c.spec (coinOf (c.resp 0))) = .ok extras →
      Ret ((repPZk (asksS c.resp) (Hashes.ofSha512 (Claims.hOf H)) (Setup.ofCircuit st) v.1 v.2 rep dom tr tb cm₀
        extras pads ic extra lanePad salt nonce prior ws).run (Transcript.init #[] dom)) fun pt => RepOk pt.2.rounds

/-- **What the record needs of the masked prover's pieces** (`ProverOkZk`), at an oracle of 64-byte outputs, from its
rep half (`RepRoundsZk`). -/
theorem proverOkZk (h64 : Claims.Out64 H) (hR : RepRoundsZk H rel) : ProverOkZk H rel := by
  intro c st hst
  obtain ⟨st', hst', -, hm1, hm2⟩ := c.table
  obtain ⟨st'', hst'', hms⟩ := c.hm96
  rw [hst'] at hst
  obtain rfl : st = st' := by simpa using hst
  rw [hst'] at hst''
  obtain rfl : st = st'' := by simpa using hst''
  refine ⟨fun v hv g extra lanePad salt cm hcm => ?_, hR c st (by rw [hst']; simp)⟩
  have hm : (Setup.ofCircuit st).merkle = MerkleScheme.hm96Sha512 := hms
  rw [hm] at hcm
  exact commitPZk_cap (hOf_size h64) hm1 hm2 hv hcm

end FlockComplete.JsonRT
