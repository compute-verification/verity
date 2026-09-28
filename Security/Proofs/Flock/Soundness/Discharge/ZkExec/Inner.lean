import Proofs.Flock.Soundness.Discharge.ZkExec.Statements
import Proofs.Flock.Soundness.Refine.Exec
import Proofs.Flock.Soundness.Refine.Table

/-!
# flock-verify `--zk`'s inner proof accepts (live/PROTOCOL.md §5)

`verifyInner_ok : ZkInnerOk`: `Zk.verifyInner` accepts only with the pinned sizes, `β ≠ 0`, `τ` opening its
commitment, the batched constraint, and every queried column consistent with `Y` and climbing to the pads' root under
its `hm96` leaf. The run is peeled step by step (`Refine.Exec`); the column loop's checks come from `forIn_each`.
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Transcript Refine

/-- A loop whose every step yields and establishes `P i` establishes `P i` at every index it ran. -/
theorem forIn_each {σ : Type} (P : ℕ → Prop) (body : ℕ → σ → TM (ForInStep σ))
    (hbody : ∀ i s t b t', body i s t = .ok (b, t') → P i ∧ ∃ s', b = .yield s') :
    ∀ (n a : ℕ) (s : σ) (t t' : Transcript) (out : σ),
      forIn (List.range' a n) s body t = .ok (out, t') → ∀ i ∈ List.range' a n, P i := by
  intro n
  induction n with
  | zero => intro a s t t' out _ i hi; simp at hi
  | succ n ih =>
      intro a s t t' out h i hi
      rw [forIn_range'_cons] at h
      obtain ⟨b, t₁, h₁, h₂⟩ := bind_ok h
      obtain ⟨hP, s', rfl⟩ := hbody a s t b t₁ h₁
      rw [List.range'_succ, List.mem_cons] at hi
      rcases hi with rfl | hi
      · exact hP
      · exact ih (a + 1) s' t₁ t' out h₂ i hi

theorem map_lo_getElem! (ws : Array F128) (N k : ℕ) :
    (ws.map fun w => w.lo.toNat % N)[k]! = (ws[k]!).lo.toNat % N := by
  by_cases hk : k < ws.size
  · rw [getElem!_pos _ k (by simpa using hk), getElem!_pos ws k hk, Array.getElem_map]
  · rw [getElem!_neg _ k (by simpa using hk), getElem!_neg ws k hk]
    rfl

theorem verifyInner_ok : ZkInnerOk := by
  intro lay pr finalC cons t t' h
  unfold Flock.Zk.verifyInner at h
  dsimp only at h
  obtain ⟨hsz, h⟩ := check_bind h
  simp only [Bool.and_eq_true, beq_iff_eq] at hsz
  obtain ⟨⟨⟨⟨⟨⟨hy, ho⟩, hs⟩, hp⟩, hr⟩, htc⟩, hts⟩ := hsz
  obtain ⟨t1, -, h⟩ := label_bind h
  obtain ⟨t2, -, h⟩ := absorbF_bind h
  obtain ⟨-, -, t3, -, -, h⟩ := sample_bind h
  obtain ⟨t4, -, h⟩ := absorbSlice_bind h
  obtain ⟨-, -, t5, -, -, h⟩ := sampleVec_bind (by simp [Flock.Zk.triple]) h
  obtain ⟨t6, -, h⟩ := absorbBytes_bind h
  obtain ⟨-, -, t7, -, -, h⟩ := sample_bind h
  obtain ⟨hb, h⟩ := check_bind h
  obtain ⟨t8, -, h⟩ := absorbSlice_bind h
  obtain ⟨-, -, t9, -, -, h⟩ := sampleVec_bind (by decide) h
  obtain ⟨htau, h⟩ := check_bind h
  obtain ⟨hdot, h⟩ := check_bind h
  rw [forIn_range_eq] at h
  obtain ⟨_, _, hloop, -⟩ := bind_ok h
  generalize (t2.rounds[t2.k]!).coins.getD 0 0 = eps at hdot
  generalize (t4.rounds[t4.k]!).coins = gammas at hdot
  generalize (t6.rounds[t6.k]!).coins.getD 0 0 = beta at hb hdot hloop
  generalize (t8.rounds[t8.k]!).coins = ws at hloop
  have hk := forIn_each (fun k =>
      Flock.Zk.codeAt pr.y ((ws[k]!).lo.toNat % innerN lay) = (pr.opened[k]!).1 + beta * (pr.opened[k]!).2 ∧
      (pr.paths[k]!).size = Nat.log2 (innerN lay) ∧
      Flock.climb MerkleScheme.hm96Sha512
        (Hm96.Default512.leaf (Sha512.hash ((pr.opened[k]!).1.toBytes ++ (pr.opened[k]!).2.toBytes)) pr.salts[k]!)
        ((ws[k]!).lo.toNat % innerN lay) (pr.paths[k]!).toList = pr.root) _
    (fun i s t b t' hb => by
      obtain ⟨c1, hb⟩ := check_bind hb
      obtain ⟨c2, hb⟩ := check_bind hb
      obtain ⟨c3, hb⟩ := check_bind hb
      obtain ⟨rfl, -⟩ := pure_ok hb
      rw [map_lo_getElem!] at c1 c3
      exact ⟨⟨eq_of_beq c1, eq_of_beq c2, bytes_eq_of_beq c3⟩, _, rfl⟩) Flock.Zk.Q 0 _ _ _ _ hloop
  refine ⟨hy, ho, hs, hp, hr, htc, hts, eps, beta, gammas, ws, by simpa using hb, bytes_eq_of_beq htau,
    eq_of_beq hdot, fun k hk' => hk k (List.mem_range'_1.2 ⟨Nat.zero_le _, by simpa using hk'⟩)⟩

end FlockSoundness.Discharge.ZkExec
