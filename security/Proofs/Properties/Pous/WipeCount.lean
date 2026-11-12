module

public import Properties.Pous.WipeCount
public import Proofs.Properties.Pous.WipeSigned

/-!
# Proof of `WipeCountTied` (`Properties.Pous.WipeCount`)

`wipeCountTied` reads its antecedents and nothing else. What an accepted count means (`CountRan`) gives the two marks
of the device the span is of, or the break. The marks' messages and the start certificate's SHA-512 are the wipe's by
the verifier's three equalities. The span's phases are the phases the device stamped, which are its code's
(`NetTiming.DeviceIntact`: `NetTiming.markPhase` of each mark's bucket), and its tail is the device's, so the device
model's lines for a span (`NetTiming.Guarantees.SpanCovers`) put every cell sent between the marks in the span's phases.
The egress the wipe publishes is at least the count's in the wipe's unit, whole MiB rounded up and saturating
(`StreamedWipe.egressUnits`), so below the saturated count it is at least the count's in bits (`bits_le_of_units_le`).

`wipeCountTied_honest` is its honest lemma: the wipe's compiled check accepts the honest session of
`wipeSigned_honest`, and `CountRan` holds of a count that accepts a span of a mark its device made
(`countRan_marked`). `wipeCountTied_nontrivial` is its nontrivial lemma: a device that made no mark gives no span
its marks, so the conclusion fails there with a break that never holds.
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

/-- **`WipeCountTied`**: an accepted wipe's egress is the network certifier's count over the wipe's span, from what an
accepted count means and the device model's lines for a span. -/
theorem wipeCountTied : WipeCountTied := by
  intro κ P C Ev Pr M count broken d p c e out prof _ hdev s pn cn en n profn _ hn hsm hst hend hunits hran hcov
  rcases hran pn cn en n profn hn with ⟨a, b, ha, hb, ham, hac, hbm, hsp, hep, htail⟩ | hbr
  · refine .inl ⟨a, ha, b, hb, ⟨ham.trans hst, hac.trans hsm, hend.trans (by rw [hbm])⟩, ?_,
      bits_le_of_units_le hunits⟩
    intro k j hE hj hadm hbkt
    rw [hsp, hep, htail, hdev a.bucket, hdev b.bucket]
    exact hcov d.schedule d.cellCap d.phaseBuckets a.bucket b.bucket k j hE hj hadm hbkt
  · exact .inr hbr

/-- `CountRan` holds of the count that accepts `markedSpan`, at the device that made its mark: the span is of that
mark, as its start and its end. -/
theorem countRan_marked : CountRan markedCount (fun _ _ _ => False) markedDevice := by
  intro pn cn en n profn h
  obtain ⟨rfl, -⟩ := Core.Outcome.accept.inj h
  exact .inl ⟨emptyMark, emptyMark, List.mem_singleton.mpr rfl, List.mem_singleton.mpr rfl, rfl, rfl, rfl, rfl, rfl,
    rfl⟩

/-- **`WipeCountTied`'s honest lemma**: its hypothesis holds at the honest deployed instance, the wipe of record's
session that `wipeSigned_honest` signs and the wipe's compiled check accepts; and what an accepted count means holds of
a count that accepts a span, at the device that made the span's mark. -/
theorem wipeCountTied_honest :
    ∃ (seed K : ByteArray) (c : Nat → Nat) (k : Lms.PrivateKey) (sig : ByteArray) (out : WipeProtocol.Public)
      (prof : StreamedWipe.Params × Int),
      Lms.keygen Lms.deployed seed = .ok k ∧
        WipeProtocol.protocol.check (measuredWipeParams k.publicKey) measuredCommitment
          ⟨some (measuredSession (measuredDigest K c) k.publicKey sig)⟩ = .accept out prof ∧
        CountRan markedCount (fun _ _ _ => False) markedDevice := by
  obtain ⟨seed, K, c, k, sig, out, prof, hk, h⟩ := wipeSigned_honest
  exact ⟨seed, K, c, k, sig, out, prof, hk, h, countRan_marked⟩

/-- **`WipeCountTied`'s nontrivial lemma**: its conclusion fails at a network certifier that made no mark, for the
unencoded session and a break that never holds. No mark of the device is the span's start mark. -/
theorem wipeCountTied_nontrivial :
    ¬ (SpanCounted emptyMarks idleDevice measuredCommitment unencoded markedSpan 0 ∨ False) := by
  rintro (⟨a, ha, -⟩ | h)
  · exact absurd ha (by simp [idleDevice])
  · exact h

end MemoryChallenger.Properties.SecurityProofs
