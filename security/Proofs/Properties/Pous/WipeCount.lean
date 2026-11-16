module

public import Properties.Pous.WipeCount
public import Proofs.Properties.Pous.WipeSigned
public import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# Proof of `WipeCountTied` (`Properties.Pous.WipeCount`)

`wipeCountTied` reads the two checks' acceptances and its antecedents. The counted check's acceptance gives the case's
`WipeProtocol.named` and publishes its count (`checkCounted_accepts`), and the span's check's gives the span's egress
bits as the count's (`span_accepts`). What an accepted span means (`CountRan`) gives the two marks of the device the
span is of, or the break. The tie (`SpanTied`) makes the marks' messages the wipe's. The memory challenger's antecedent
gives a mark over the start message whose certificate the session signs; the device marks a message once
(`MarksOnce`), so it has the span's start mark's phase. The span's phases are the phases the device stamped, which are
its code's (`NetTiming.DeviceIntact`), and its tail is the device's, so the device model's lines for a span
(`NetTiming.Guarantees.SpanCovers`) put every cell sent between the marks in the span's phases. The span's capacity a
phase is the device's, which bounds the bits it sends a phase (`SentWithinCapacity`). The tie puts the count, in the
wipe's unit, at most the published egress, so below the saturated one it is at most that egress in bits
(`bits_le_of_units_le`).

`counted_accepts` is the statement's non-vacuity: the counted check accepts the measured honest session that names a
64-byte start mark beside any count in range, and publishes that count, so a published egress below the saturated one
is reached.

`wipeCountTied_honest` is its honest lemma, at the network certifier's default schedule (`Period.defaultSchedule`) and
S1's deployed LMS: the counted check accepts the honest marked session and publishes 36 MiB, below the saturated
count; the device that made its two marks is its stamp's code, sends within its capacity and marks a message once; the
tie holds for the span of those marks; and `CountRan` holds there of that span, every other span being the run's break.
`wipeCountTied_nontrivial` is its nontrivial lemma: at a device that made no mark the conclusion fails for a count of
the default schedule, with a break that never holds.
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

/-- The bits a device that sends within its capacity sent in the phases a count counts are at most the count's egress
bits, when the count's capacity a phase is the device's. -/
theorem sentOver_le_countBits {κ : Type} {d : MarkDevice κ} {n : NetworkCheck.Span.Count}
    (hcap : SentWithinCapacity d) (hn : n.capacityBits = d.capacityBits) : sentOver n d ≤ countBits n := by
  have h := Finset.sum_le_card_nsmul (Finset.range n.phases) (fun i => d.sentBits (n.startPhase + i)) n.capacityBits
    (fun i _ => by rw [hn]; exact hcap _)
  rw [Finset.card_range, smul_eq_mul] at h
  exact h

/-- What the counted check's acceptance establishes: the case names a count (`WipeProtocol.named`), and the egress it
publishes is that count. -/
theorem checkCounted_accepts {schemes : Schemes} {p : WipeProtocol.Params} {c : StreamedWipe.Commitment}
    {e : WipeProtocol.Evidence} {out : WipeProtocol.Public} {prof : StreamedWipe.Params × Int}
    (h : WipeProtocol.checkCounted schemes p c e = .accept out prof) :
    ∃ w, WipeProtocol.named c e = .ok (some w) ∧ out.egressMib = w.egressMib := by
  unfold WipeProtocol.checkCounted at h
  split at h
  · split at h
    · cases h
    · cases h
    · rename_i w hw
      split at h
      · cases h
      · obtain ⟨rfl, -⟩ := Core.Outcome.accept.inj h
        refine ⟨w, hw, ?_⟩
        unfold WipeProtocol.certify
        split <;> rfl
  · rename_i hne
    exact absurd h (hne _ _)

/-- `WipeProtocol.named` of a signed case that names a count: its start message is the commitment's, and its end
message is the signed session's. -/
theorem named_some {c : StreamedWipe.Commitment} {e : WipeProtocol.Evidence} {s : StreamedWipe.Signed}
    {w : WipeProtocol.Named} (hs : e.signed = some s) (h : WipeProtocol.named c e = .ok (some w)) :
    w.startMessage = StreamedWipe.startMessage c ∧ StreamedWipe.endMessage s = .ok w.endMessage := by
  unfold WipeProtocol.named at h
  cases he : e.egressMib with
  | none => simp [hs, he, pure, Except.pure] at h
  | some n =>
    simp only [hs, he] at h
    cases hm : StreamedWipe.endMessage s with
    | error r => simp [hm, bind, Except.bind] at h
    | ok m =>
      simp only [hm, bind, Except.bind, pure, Except.pure, Except.ok.injEq, Option.some.injEq] at h
      subst h
      exact ⟨rfl, rfl⟩

/-- What the span's check's acceptance establishes: the count's phases are its marks' and its tail and capacity a
phase are the schedule's, and the egress bits `NetworkCheck.Span.named` reads off the span are the count's. -/
theorem span_accepts {p : NetworkCheck.Span.Params} {cs : NetworkCheck.Span.Commitment}
    {ev : NetworkCheck.Span.Evidence} {n : NetworkCheck.Span.Count} {pr : NetworkCheck.Span.Profile}
    (h : NetworkCheck.Span.check p cs ev = .accept n pr) :
    n.startPhase = cs.startMark.certificate.window ∧ n.endPhase = cs.endMark.certificate.window ∧
      n.tail = NetworkCheck.Span.tail p.schedule ∧ n.capacityBits = NetworkCheck.Span.capacityBits p.schedule ∧
      (NetworkCheck.Span.named p cs).egressBits = some (countBits n) := by
  unfold NetworkCheck.Span.check at h
  split at h
  · cases h
  · cases h
  · simp only at h
    split at h
    · cases h
    · rename_i hlt
      split at h
      · cases h
      · obtain ⟨rfl, -⟩ := Core.Outcome.accept.inj h
        refine ⟨rfl, rfl, rfl, rfl, ?_⟩
        simp only [NetworkCheck.Span.named, countBits]
        rw [ite_cond_eq_true _ _ (eq_true (by omega))]

/-- **`WipeCountTied`**: a counted wipe's egress is the network certifier's count over the wipe's span, from the two
checks' acceptances, the tie, what an accepted span means, the memory challenger's start mark and the device model's
lines for a span, its capacity a phase and its marking a message once. -/
theorem wipeCountTied : WipeCountTied := by
  intro κ d p c e out prof hacc sp cs ev n pr hspan s hs ⟨a, ha, ham, hac⟩ broken htie hdev hran hcov hcap honce
  obtain ⟨w, hw, hout⟩ := checkCounted_accepts hacc
  obtain ⟨w', hw', hsm, hem, hunits⟩ := htie
  rw [hw] at hw'
  cases hw'
  obtain ⟨hwsm, hwem⟩ := named_some hs hw
  obtain ⟨-, -, -, -, hbits⟩ := span_accepts hspan
  rcases hran cs ev n pr hspan with ⟨a', b, ha', hb, ha'm, hbm, hsp, hep, htail, hcapn⟩ | hbr
  · have hmsg : a.message = a'.message := ham.trans (hwsm.symm.trans (hsm.trans ha'm.symm))
    have hph : d.stamp a.bucket = d.stamp a'.bucket := (honce a ha a' ha' hmsg).2
    refine .inl ⟨a, ha, b, hb, ⟨ham, hac, hwem.trans (congrArg _ (hem.trans hbm.symm))⟩, ?_,
      sentOver_le_countBits hcap hcapn, ?_⟩
    · intro k j hE hj hadm hbkt
      rw [hsp, ← hph, hep, htail, hdev a.bucket, hdev b.bucket]
      exact hcov d.schedule d.cellCap d.phaseBuckets a.bucket b.bucket k j hE hj hadm hbkt
    · intro hlt
      rw [hout] at hlt ⊢
      exact bits_le_of_units_le (hunits _ hbits) hlt
  · exact .inr hbr

/-! ## Non-vacuity -/

/-- **The counted check accepts and publishes a count below the saturated one**: for every key `Lms.key` derives at
S1's deployed LMS parameters, every digest of the batches' answers, every 64-byte start mark `sm` and every count `n`
in range, the measured honest session naming `sm` and signed at leaf 0 is accepted by the counted check beside `n`,
which publishes `n`; its end message is defined. -/
theorem counted_accepts (digest : Nat → ByteArray) (sm : ByteArray) (hsm : sm.size = StreamedWipe.MARK_BYTES)
    {n : Int} (hn0 : 0 ≤ n) (hn1 : n ≤ StreamedWipe.EGRESS_MAX_UNITS) {I seed : ByteArray} {k : Lms.PrivateKey}
    (hk : Lms.key Lms.deployed I seed = .ok k) :
    ∃ (sig m : ByteArray) (out : WipeProtocol.Public) (prof : StreamedWipe.Params × Int),
      StreamedWipe.endMessage { measuredWipe digest k.publicKey sig with startMark := sm } = .ok m ∧
      WipeProtocol.checkCounted SCHEMES (measuredWipeParams k.publicKey) measuredCommitment
        ⟨some { measuredWipe digest k.publicKey sig with startMark := sm }, some n⟩ = .accept out prof ∧
      out.egressMib = n := by
  obtain ⟨sig, m, out, prof, hm, h⟩ := Wipe.marked_accepts digest sm hn0 hn1 hk
  have h' : WipeProtocol.check SCHEMES (measuredWipeParams k.publicKey) measuredCommitment
      ⟨some { measuredWipe digest k.publicKey sig with startMark := sm }, some n⟩ = .accept out prof := h
  have hend : StreamedWipe.endMessage { measuredWipe digest k.publicKey sig with startMark := sm } =
      .ok (Sha512.hash (StreamedWipe.END_TAG ++ be64 m.size.toUInt64 ++ m ++
        be64 (measuredWipe digest k.publicKey sig).signature.size.toUInt64 ++
        (measuredWipe digest k.publicKey sig).signature)) := by
    unfold StreamedWipe.endMessage
    rw [hm]
    rfl
  refine ⟨sig, _, WipeProtocol.certify measuredCommitment.params n, (measuredCommitment.params, n), hend, ?_, ?_⟩
  · unfold WipeProtocol.checkCounted WipeProtocol.named
    simp only [h', hend, hsm, bind, Except.bind, pure, Except.pure, bne_self_eq_false, Bool.false_eq_true,
      ↓reduceIte]
  · unfold WipeProtocol.certify
    split <;> rfl

/-! ## The witnesses -/

theorem witness_units :
    StreamedWipe.egressUnits ((3 * NetworkCheck.Span.capacityBits NetworkCheck.Period.defaultSchedule : ℕ) : Int) ≤
      WITNESS_EGRESS_MIB := by
  decide

theorem witness_tail :
    NetworkCheck.Span.tail NetworkCheck.Period.defaultSchedule = NetTiming.spanTail 10 4096 4 600 := by
  decide

theorem witness_tail_two : NetworkCheck.Span.tail NetworkCheck.Period.defaultSchedule = 2 := by
  decide

/-- `CountRan` holds at the witnesses' device of its span: a span the count accepts with those marks is of the device's
two marks, every other span being the break. -/
theorem countRan_witness (key bytes sm em : ByteArray) :
    CountRan (witnessSpanParams key) (fun cs _ => cs ≠ witnessSpan key bytes sm em)
      (witnessDevice key (witnessMarks bytes sm em)) := by
  intro cs ev n pr h
  by_cases hc : cs = witnessSpan key bytes sm em
  · subst hc
    obtain ⟨hsp, hep, htail, hcap, -⟩ := span_accepts h
    exact .inl ⟨⟨sm, 0, bytes⟩, ⟨em, 0, bytes⟩, List.mem_cons_self, List.mem_cons_of_mem _ List.mem_cons_self,
      rfl, rfl, hsp.trans (by simp [witnessSpan, markCertificate, witnessDevice, NetTiming.markPhase]),
      hep.trans (by simp [witnessSpan, markCertificate, witnessDevice, NetTiming.markPhase]),
      htail.trans witness_tail, hcap⟩
  · exact .inr hc

theorem marksOnce_witness (key bytes sm em : ByteArray) :
    MarksOnce (witnessDevice key (witnessMarks bytes sm em)) := by
  intro a ha b hb _
  simp only [witnessDevice, witnessMarks, List.mem_cons, List.not_mem_nil, or_false] at ha hb
  rcases ha with rfl | rfl <;> rcases hb with rfl | rfl <;> exact ⟨rfl, rfl⟩

theorem sentWithinCapacity_witness (key : ByteArray) (marks : List Mark) :
    SentWithinCapacity (witnessDevice key marks) := fun _ => Nat.zero_le _

/-- **`WipeCountTied`'s honest lemma**, at S1's deployed LMS and the network certifier's default schedule: for the key
`honest_keygen` derives, the counted check accepts the measured honest session that names the SHA-512 of the start
certificate's bytes, beside a count of 36 MiB, and publishes it, below the saturated count; the device that made the
session's two marks, over the commitment's start message `sm` and the session's end message `em`, holds the memory
challenger's start mark, its stamp is its code's, the tie holds for the span of those marks under the default schedule,
`CountRan` holds there of that span (every other span being the break, which that span's is not), and the device sends
within its capacity and marks a message once. -/
theorem wipeCountTied_honest :
    ∃ (k : Lms.PrivateKey) (seed : ByteArray), Lms.keygen Lms.deployed seed = .ok k ∧
      ∃ (digest : Nat → ByteArray) (bytes sig sm em : ByteArray) (out : WipeProtocol.Public)
        (prof : StreamedWipe.Params × Int),
        sm = StreamedWipe.startMessage measuredCommitment ∧
        WipeProtocol.checkCounted SCHEMES (measuredWipeParams k.publicKey) measuredCommitment
          ⟨some { measuredWipe digest k.publicKey sig with startMark := Sha512.hash bytes },
            some WITNESS_EGRESS_MIB⟩ = .accept out prof ∧
        out.egressMib < StreamedWipe.EGRESS_MAX_UNITS ∧
        (∃ a ∈ (witnessDevice k.publicKey (witnessMarks bytes sm em)).marks,
          a.message = StreamedWipe.startMessage measuredCommitment ∧
            Sha512.hash a.certificate =
              { measuredWipe digest k.publicKey sig with startMark := Sha512.hash bytes }.startMark) ∧
        SpanTied (witnessSpanParams k.publicKey) (witnessSpan k.publicKey bytes sm em) measuredCommitment
          ⟨some { measuredWipe digest k.publicKey sig with startMark := Sha512.hash bytes },
            some WITNESS_EGRESS_MIB⟩ ∧
        NetTiming.DeviceIntact (witnessDevice k.publicKey (witnessMarks bytes sm em)).stamp
          (NetTiming.markPhase (witnessDevice k.publicKey (witnessMarks bytes sm em)).phaseBuckets) ∧
        CountRan (witnessSpanParams k.publicKey) (fun cs _ => cs ≠ witnessSpan k.publicKey bytes sm em)
          (witnessDevice k.publicKey (witnessMarks bytes sm em)) ∧
        SentWithinCapacity (witnessDevice k.publicKey (witnessMarks bytes sm em)) ∧
        MarksOnce (witnessDevice k.publicKey (witnessMarks bytes sm em)) := by
  obtain ⟨k, hkg, I, hk⟩ := Wipe.honest_keygen
  let digest : Nat → ByteArray := fun _ => .empty
  let bytes : ByteArray := .empty
  obtain ⟨sig, m, out, prof, hend, hacc, hout⟩ := counted_accepts digest (Sha512.hash bytes)
    (Properties.Certifier.Bytes.hash_size _) (n := WITNESS_EGRESS_MIB) (by decide) (by decide) hk
  refine ⟨k, _, hkg, digest, bytes, sig, _, m, out, prof, rfl, hacc, by rw [hout]; decide,
    ⟨_, List.mem_cons_self, rfl, rfl⟩, ?_, fun _ => rfl, countRan_witness _ _ _ _, sentWithinCapacity_witness _ _,
    marksOnce_witness _ _ _ _⟩
  obtain ⟨w, hw, hwout⟩ := checkCounted_accepts hacc
  obtain ⟨hwsm, hwem⟩ := named_some rfl hw
  refine ⟨w, hw, hwsm, ?_, ?_⟩
  · rw [hend] at hwem
    exact (Except.ok.inj hwem).symm
  · intro b hb
    have hb' : b = 3 * NetworkCheck.Span.capacityBits NetworkCheck.Period.defaultSchedule := by
      simp only [NetworkCheck.Span.named, witnessSpan, markCertificate, witnessSpanParams, witness_tail_two,
        le_refl, ↓reduceIte, Option.some.injEq] at hb
      omega
    rw [← hwout, hout, hb']
    exact witness_units

/-- **`WipeCountTied`'s nontrivial lemma**: its conclusion fails at a network certifier that made no mark, for the
unencoded session, a count of the default schedule and a break that never holds. No mark of the device is the span's
start mark. -/
theorem wipeCountTied_nontrivial :
    ¬ (SpanCounted (witnessDevice .empty []) measuredCommitment unencoded
        ⟨0, 0, 2, 3, NetworkCheck.Span.capacityBits NetworkCheck.Period.defaultSchedule⟩ 0 ∨ False) := by
  rintro (⟨a, ha, -⟩ | h)
  · exact absurd ha (by simp [witnessDevice])
  · exact h

end MemoryChallenger.Properties.SecurityProofs
