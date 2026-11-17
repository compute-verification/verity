module

public import Properties.Pous.WipeCount
public import Proofs.Properties.Pous.WipeSigned
public import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-! `wipeCountTied` reads its antecedents and nothing else: an accepted count gives the two marks of the device the span
is of, or the break. -/

@[expose] public section

namespace MemoryChallenger.Properties.SecurityProofs

open MemoryChallenger Flock

theorem bits_le_of_units_le {b m : Int} (h : StreamedWipe.egressUnits b ≤ m)
    (hm : m < StreamedWipe.EGRESS_MAX_UNITS) : b ≤ m * StreamedWipe.EGRESS_UNIT_BITS := by
  have hU : (0 : Int) < StreamedWipe.EGRESS_UNIT_BITS := by decide
  have hx : -(Int.fdiv (-b) StreamedWipe.EGRESS_UNIT_BITS) ≤ m := by
    unfold StreamedWipe.egressUnits at h
    omega
  rw [Int.fdiv_eq_ediv_of_nonneg _ (Int.le_of_lt hU)] at hx
  have h' := (Int.le_ediv_iff_mul_le hU).mp (show -m ≤ -b / StreamedWipe.EGRESS_UNIT_BITS by omega)
  rw [Int.neg_mul] at h'
  omega

theorem sentBits_le_egressBits {κ : Type} {d : MarkDevice κ} {n : SpanCount} (hcap : SentWithinCapacity d)
    (hn : n.capacityBits = d.capacityBits) : n.sentBits d ≤ n.egressBits := by
  have h := Finset.sum_le_card_nsmul (Finset.range (n.endPhase + n.tail + 1 - n.startPhase))
    (fun i => d.sentBits (n.startPhase + i)) n.capacityBits (fun i _ => by rw [hn]; exact hcap _)
  rw [Finset.card_range, smul_eq_mul] at h
  exact h

/-- **`WipeCountTied`**: an accepted wipe's egress is the network attester's count over the wipe's span, from what an
accepted count under the device's parameters means, the memory challenger's start mark and the device model's lines
for a span, its capacity a phase and its marking a message once. -/
theorem wipeCountTied : WipeCountTied := by
  intro κ P C Ev Pr M count params broken d p c e out prof _ hdev s cn en n profn _ ⟨a, ha, ham, hac⟩ hn _ hst hend
    hunits hran hcov hcap honce
  rcases hran cn en n profn hn with ⟨a', b, ha', hb, ha'm, hbm, hsp, hep, htail, hcapn⟩ | hbr
  · have hph : d.stamp a.bucket = d.stamp a'.bucket := (honce a ha a' ha' (ham.trans (ha'm.trans hst).symm)).2
    refine .inl ⟨a, ha, b, hb, ⟨ham, hac, hend.trans (by rw [hbm])⟩, ?_, sentBits_le_egressBits hcap hcapn,
      bits_le_of_units_le hunits⟩
    intro k j hE hj hadm hbkt
    rw [hsp, ← hph, hep, htail, hdev a.bucket, hdev b.bucket]
    exact hcov d.schedule d.cellCap d.phaseBuckets a.bucket b.bucket k j hE hj hadm hbkt
  · exact .inr hbr

theorem countRan_marked : CountRan markedCount () (fun _ _ => False) markedDevice := by
  intro cn en n profn h
  obtain ⟨rfl, -⟩ := Core.Outcome.accept.inj h
  exact .inl ⟨emptyMark, emptyMark, List.mem_singleton.mpr rfl, List.mem_singleton.mpr rfl, rfl, rfl, rfl, rfl, rfl,
    rfl⟩

theorem sentWithinCapacity_marked : SentWithinCapacity markedDevice := fun _ => Nat.le_refl 0

theorem marksOnce_marked : MarksOnce markedDevice := by
  intro a ha b hb _
  rw [List.mem_singleton.mp ha, List.mem_singleton.mp hb]
  exact ⟨rfl, rfl⟩

/-- **`WipeCountTied`'s honest lemma**: its hypothesis holds at the honest deployed instance, the wipe of record's
session that `wipeSigned_honest` signs and the wipe's compiled check accepts; what an accepted count means holds of a
count that accepts a span, at the device that made the span's mark; and that device sends within its capacity and
marks a message once. -/
theorem wipeCountTied_honest :
    ∃ (seed K : ByteArray) (c : Nat → Nat) (k : Lms.PrivateKey) (sig : ByteArray) (out : WipeProtocol.Public)
      (prof : StreamedWipe.Params × Int),
      Lms.keygen Lms.deployed seed = .ok k ∧
        WipeProtocol.protocol.check (measuredWipeParams k.publicKey) measuredCommitment
          ⟨some (measuredWipe (measuredDigest K c) k.publicKey sig), none⟩ = .accept out prof ∧
        CountRan markedCount () (fun _ _ => False) markedDevice ∧ SentWithinCapacity markedDevice ∧
          MarksOnce markedDevice := by
  obtain ⟨seed, K, c, k, sig, out, prof, hk, h⟩ := wipeSigned_honest
  exact ⟨seed, K, c, k, sig, out, prof, hk, h, countRan_marked, sentWithinCapacity_marked,
    marksOnce_marked⟩

/-- **`WipeCountTied`'s nontrivial lemma**: its conclusion fails at a network attester that made no mark, for the
unencoded session and a break that never holds. No mark of the device is the span's start mark. -/
theorem wipeCountTied_nontrivial :
    ¬ (SpanCounted emptyMarks idleDevice measuredCommitment unencoded markedSpan 0 ∨ False) := by
  rintro (⟨a, ha, -⟩ | h)
  · exact absurd ha (by simp [idleDevice])
  · exact h

end MemoryChallenger.Properties.SecurityProofs
