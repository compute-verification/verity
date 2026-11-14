module

public import Properties.Pous.WipeCount
public import Proofs.Properties.Pous.WipeSigned
public import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# Proof of `WipeCountTied` (`Properties.Pous.WipeCount`)

`wipeCountTied` reads its antecedents and nothing else. What an accepted count under the device's parameters means
(`CountRan`) gives the two marks of the device the span is of, or the break. The marks' messages are the wipe's by the
verifier's two message equalities. The memory challenger's antecedent gives a mark over the start message whose
certificate the session signs; the device marks a message once (`MarksOnce`), so it has the count's start mark's
phase, and it is the span's start mark. The span's phases are the phases the device stamped, which are its code's
(`NetTiming.DeviceIntact`: `NetTiming.markPhase` of each mark's bucket), and its tail is the device's, so the device
model's lines for a span (`NetTiming.Guarantees.SpanCovers`) put every cell sent between the marks in the span's
phases. The span's capacity a
phase is the link's registered one, which bounds the bits the device sends a phase (`SentWithinCapacity`), so the
count's egress bounds the bits it sent in the span's phases. The egress the wipe publishes is at least the count's in
the wipe's unit, whole MiB rounded up and saturating (`StreamedWipe.egressUnits`), so below the saturated count it is
at least the count's in bits (`bits_le_of_units_le`).

`wipeCountTied_honest` is its honest lemma: the wipe's compiled check accepts the honest session of
`wipeSigned_honest`; `CountRan` holds of a count that accepts a span of a mark its device made (`countRan_marked`);
and that device sends within its capacity and marks a message once. `wipeCountTied_nontrivial` is its nontrivial
lemma: a device that made no mark gives no span its marks, so the conclusion fails there with a break that never
holds.
-/

@[expose] public section

namespace MemoryChallenger.Properties.SecurityProofs

open MemoryChallenger Flock

/-- An egress of `b` bits that is at most `m` in the wipe's unit, with `m` below the saturated count, is at most `m`
units in bits: `egressUnits` rounds up and saturates, so a count below the saturated one is `⌈b / 2^23⌉`. -/
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

/-- The bits a device that sends within its capacity sent in a span's phases are at most the span's egress, when the
span's capacity a phase is the device's. -/
theorem sentBits_le_egressBits {κ : Type} {d : MarkDevice κ} {n : SpanCount} (hcap : SentWithinCapacity d)
    (hn : n.capacityBits = d.capacityBits) : n.sentBits d ≤ n.egressBits := by
  have h := Finset.sum_le_card_nsmul (Finset.range (n.endPhase + n.tail + 1 - n.startPhase))
    (fun i => d.sentBits (n.startPhase + i)) n.capacityBits (fun i _ => by rw [hn]; exact hcap _)
  rw [Finset.card_range, smul_eq_mul] at h
  exact h

/-- **`WipeCountTied`**: an accepted wipe's egress is the network certifier's count over the wipe's span, from what an
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

/-- `CountRan` holds of the count that accepts `markedSpan`, at the device that made its mark: the span is of that
mark, as its start and its end. -/
theorem countRan_marked : CountRan markedCount () (fun _ _ => False) markedDevice := by
  intro cn en n profn h
  obtain ⟨rfl, -⟩ := Core.Outcome.accept.inj h
  exact .inl ⟨emptyMark, emptyMark, List.mem_singleton.mpr rfl, List.mem_singleton.mpr rfl, rfl, rfl, rfl, rfl, rfl,
    rfl⟩

/-- `markedDevice` sends nothing, within its capacity. -/
theorem sentWithinCapacity_marked : SentWithinCapacity markedDevice := fun _ => Nat.le_refl 0

/-- `markedDevice` made one mark, so it marks a message once. -/
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
          ⟨some (measuredSession (measuredDigest K c) k.publicKey sig), none⟩ = .accept out prof ∧
        CountRan markedCount () (fun _ _ => False) markedDevice ∧ SentWithinCapacity markedDevice ∧
          MarksOnce markedDevice := by
  obtain ⟨seed, K, c, k, sig, out, prof, hk, h⟩ := wipeSigned_honest
  exact ⟨seed, K, c, k, sig, out, prof, hk, h, countRan_marked, sentWithinCapacity_marked,
    marksOnce_marked⟩

/-- **`WipeCountTied`'s nontrivial lemma**: its conclusion fails at a network certifier that made no mark, for the
unencoded session and a break that never holds. No mark of the device is the span's start mark. -/
theorem wipeCountTied_nontrivial :
    ¬ (SpanCounted emptyMarks idleDevice measuredCommitment unencoded markedSpan 0 ∨ False) := by
  rintro (⟨a, ha, -⟩ | h)
  · exact absurd ha (by simp [idleDevice])
  · exact h

end MemoryChallenger.Properties.SecurityProofs
