import Properties.Pous.Component
import Proofs.Properties.Pous.RunAt
import Proofs.Properties.Pous.MemoryChallenger
import Mathlib.Tactic.SplitIfs

/-!
# Proofs of PoUS's and the wipe's component statements (`Properties.Pous.Component`)

* `runAccepts`: `RunProtocol.protocol_accepts` and `accepts_iff` read the acceptance as `AcceptedAt`; the published
  outputs and profile are `check`'s on `accepts`' `.ok`.
* `runDigest`, `wipeDigest`: a commitment that decodes is canonical JSON (`Core.canonicalJson`) in the form its encoder
  writes (`exactly`), so the encoder's canonical text is the bytes it decoded from (`canonicalJson_ok`, `exactly_ok`).
* `runBinds`: setup's acceptance publishes the commitment and salt, so two acceptances of one `pub` share them; the two
  `C*`s and the two tables have one root each, so by `mth_binds` their leaves agree or the trees collide, and a leaf
  that agrees has one row (`leafInput`), whose block is one (`blockInput`), or the runs' data collide.
* `wipeAccepts`: `StreamedWipe.verdict` is its checks before the batches (`wpre`) followed by the batches (`wtail`),
  as `MemoryChallenger.verdict` is in `Proofs.Properties.Pous.MemoryChallenger`; an accepted verdict has no reason, so
  each check passed.
-/

namespace Pous.Component.SecurityProofs

open Flock MemoryChallenger MemoryChallenger.Properties PousCheck PousCheck.Properties Pous.EndToEnd
open Pous.EndToEnd.SecurityProofs
open PousCheck.Properties.SecurityProofs (append_cancel mth_binds leaves_entry root_mth)
open MemoryChallenger.Properties.SecurityProofs (forIn'_measure_eq)

local instance instLawfulBEqByteArray_component : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := show (a.data == a.data) = true from beq_iff_eq.2 rfl

deriving instance LawfulBEq for StreamedWipe.Params
deriving instance LawfulBEq for StreamedWipe.Commitment

/-! ## PoUS's acceptance -/

theorem runAccepts : RunAccepts := by
  intro p pub e out prof h
  have hacc := (RunProtocol.protocol_accepts p pub e).mp (Exists.intro out (Exists.intro prof h))
  refine ⟨(RunProtocol.accepts_iff _ _ _ _ _ _ _ _ _).mp hacc, ?_⟩
  have h' : RunProtocol.check SCHEMES p pub e = .accept out prof := h
  unfold RunProtocol.check at h'
  rw [hacc] at h'
  simp only [Core.Outcome.accept.injEq] at h'
  exact ⟨h'.1.symm, h'.2.symm⟩

/-! ## The digests -/

private theorem toUTF8_of_fromUTF8? {b : ByteArray} {s : String} (h : String.fromUTF8? b = some s) :
    s.toUTF8 = b := by
  unfold String.fromUTF8? at h
  split at h
  · cases h; rfl
  · cases h

/-- What `Core.canonicalJson` accepts is UTF-8 text that is the canonical JSON of what it parsed to. -/
private theorem canonicalJson_ok {b : ByteArray} {j : Lean.Json} (h : Core.canonicalJson b = .ok j) :
    ∃ s, String.fromUTF8? b = some s ∧ Flock.canon j = .ok s := by
  unfold Core.canonicalJson at h
  cases hs : String.fromUTF8? b with
  | none => simp [hs, throw, throwThe, MonadExceptOf.throw] at h
  | some s =>
    refine ⟨s, rfl, ?_⟩
    cases hp : Lean.Json.parse s with
    | error e => simp [hs, hp, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h
    | ok j0 =>
      cases hc : Flock.canon j0 with
      | error e => simp [hs, hp, hc, bind, Except.bind, pure, Except.pure] at h
      | ok s' =>
        by_cases hne : s' = s
        · subst hne
          simp [hs, hp, hc, bind, Except.bind, pure, Except.pure] at h
          subst h
          exact hc
        · simp [hs, hp, hc, hne, bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h

/-- What `exactly` accepts is in the form its encoder writes: the two have one canonical text. -/
private theorem exactly_ok {α : Type} {f : Lean.Json → Flock.V α} {enc : α → Lean.Json} {j : Lean.Json} {a : α}
    (h : MemoryChallenger.exactly f enc j = .ok a) : ∃ s, Flock.canon (enc a) = .ok s ∧ Flock.canon j = .ok s := by
  unfold MemoryChallenger.exactly at h
  cases hf : f j with
  | error e => simp [hf, bind, Except.bind] at h
  | ok a0 =>
    cases hc1 : Flock.canon (enc a0) with
    | error e => simp [hf, hc1, bind, Except.bind] at h
    | ok s1 =>
      cases hc2 : Flock.canon j with
      | error e => simp [hf, hc1, hc2, bind, Except.bind] at h
      | ok s2 =>
        by_cases hne : s1 = s2
        · subst hne
          simp [hf, hc1, hc2, bind, Except.bind, pure, Except.pure] at h
          subst h
          exact ⟨s1, hc1, hc2⟩
        · simp [hf, hc1, hc2, hne, bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h

theorem runDigest : RunDigest := by
  intro b pub h
  unfold RunProtocol.decodeCommitment at h
  cases hj : Core.canonicalJson b with
  | error e => simp [hj, bind, Except.bind] at h
  | ok j =>
    simp only [hj, bind, Except.bind] at h
    obtain ⟨s, hs, hcj⟩ := canonicalJson_ok hj
    obtain ⟨s', hpub, hcj'⟩ := exactly_ok (f := SetupProtocol.Public.ofJson) (enc := SetupProtocol.Public.toJson) h
    obtain rfl : s = s' := Except.ok.inj (hcj.symm.trans hcj')
    simp only [RunProtocol.digest, hpub, toUTF8_of_fromUTF8? hs]

theorem wipeDigest : WipeDigest := by
  intro b c h
  unfold WipeProtocol.decodeCommitment at h
  cases hj : Core.canonicalJson b with
  | error e => simp [hj, bind, Except.bind] at h
  | ok j =>
    simp only [hj, bind, Except.bind] at h
    obtain ⟨s, hs, hcj⟩ := canonicalJson_ok hj
    obtain ⟨s', hc, hcj'⟩ := exactly_ok h
    obtain rfl : s = s' := Except.ok.inj (hcj.symm.trans hcj')
    simp only [WipeProtocol.digest, hc, toUTF8_of_fromUTF8? hs]

/-! ## Two runs on one setup output -/

private theorem fork_T1 {q : P2.Params} {pub : SetupProtocol.Public} {e e' : RunProtocol.Evidence} {x : ByteArray}
    (hx : x ∈ tableStringsAt q pub e.digests e.setup.store) : x ∈ forkStrings q pub e e' :=
  List.mem_append_left _ (List.mem_append_left _ hx)

private theorem fork_T2 {q : P2.Params} {pub : SetupProtocol.Public} {e e' : RunProtocol.Evidence} {x : ByteArray}
    (hx : x ∈ tableStringsAt q pub e'.digests e'.setup.store) : x ∈ forkStrings q pub e e' :=
  List.mem_append_left _ (List.mem_append_right _ hx)

private theorem fork_B1 {q : P2.Params} {pub : SetupProtocol.Public} {e e' : RunProtocol.Evidence} {b : ByteArray}
    (hb : b ∈ flat e.setup.store) : blockInput b ∈ forkStrings q pub e e' :=
  List.mem_append_right _ (List.mem_map.2 ⟨b, List.mem_append_left _ (Array.mem_toList_iff.2 hb), rfl⟩)

private theorem fork_B2 {q : P2.Params} {pub : SetupProtocol.Public} {e e' : RunProtocol.Evidence} {b : ByteArray}
    (hb : b ∈ flat e'.setup.store) : blockInput b ∈ forkStrings q pub e e' :=
  List.mem_append_right _ (List.mem_map.2 ⟨b, List.mem_append_right _ (Array.mem_toList_iff.2 hb), rfl⟩)

private theorem idx_facts {bps segs s v : Nat} (hs : s < segs) (hv : v < bps) :
    s * bps + v < segs * bps ∧ (s * bps + v) / bps = s ∧ (s * bps + v) % bps = v := by
  have hpos : 0 < bps := Nat.lt_of_le_of_lt (Nat.zero_le v) hv
  refine ⟨?_, ?_, ?_⟩
  · calc s * bps + v < s * bps + bps := Nat.add_lt_add_left hv _
      _ = (s + 1) * bps := (Nat.succ_mul s bps).symm
      _ ≤ segs * bps := Nat.mul_le_mul_right _ hs
  · rw [Nat.add_comm, Nat.add_mul_div_right _ _ hpos, Nat.div_eq_of_lt hv, Nat.zero_add]
  · rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hv]

/-- Block `v` of segment `s` of a shaped `C*`: its leaf's input is among the run's strings, the block is one of `C*`'s,
its leaf is the tree's at `s · blocksPerSegment + v`, and that index is a block's. -/
private theorem block_mem {q : P2.Params} {pub : SetupProtocol.Public} {D : ByteArray} {C : Array (Array ByteArray)}
    (hS : ShapedAt q C) {s v : Nat} (hs : s < C.size) (hv : v < C[s].size) :
    leafInput pub.salt s v (Merkle.rowDigest C[s][v]) ∈ tableStringsAt q pub D C ∧ C[s][v] ∈ flat C ∧
      (leaves pub.salt C)[s * q.blocksPerSegment + v]! = Merkle.packageLeaf pub.salt s v C[s][v] ∧
      s * q.blocksPerSegment + v < q.blocks := by
  have hbps : v < q.blocksPerSegment := (hS.2 s hs).1 ▸ hv
  obtain ⟨hi0, hdiv, hmod⟩ := idx_facts (segs := q.segments) (hS.1 ▸ hs) hbps
  have hi : s * q.blocksPerSegment + v < q.blocks := hi0
  have hflat : (flat C)[s * q.blocksPerSegment + v]? = some C[s][v] := by
    rw [flat_getElem?At hS s v hbps, Array.getElem?_eq_getElem hs]
    simp [Array.getElem?_eq_getElem hv]
  have hleaf : (leaves pub.salt C)[s * q.blocksPerSegment + v]! = Merkle.packageLeaf pub.salt s v C[s][v] := by
    apply getElem!_of_some
    rw [leaves_getElem?At hS pub.salt s v hbps, Array.getElem?_eq_getElem hs]
    simp [Array.getElem?_eq_getElem hv]
  have hm := (mem_table_leafAt (ps := q) (pub := pub) (digests := D) (C := C) hi).2
  rw [hdiv, hmod, getElem!_of_some hflat] at hm
  exact ⟨hm, Array.mem_of_getElem? hflat, hleaf, hi⟩

private theorem stores_eq {q : P2.Params} {pub : SetupProtocol.Public} {e e' : RunProtocol.Evidence}
    (hS : ShapedAt q e.setup.store) (hS' : ShapedAt q e'.setup.store)
    (hl : ∀ i, 0 ≤ i → i < 0 + q.blocks →
      (leaves pub.salt e.setup.store)[i]! = (leaves pub.salt e'.setup.store)[i]!)
    (hX : ¬ ∃ x ∈ forkStrings q pub e e', ∃ y ∈ forkStrings q pub e e', x ≠ y ∧ Sha512.hash x = Sha512.hash y) :
    e.setup.store = e'.setup.store := by
  apply Array.ext (hS.1.trans hS'.1.symm)
  intro s hs hs'
  apply Array.ext ((hS.2 s hs).1.trans (hS'.2 s hs').1.symm)
  intro v hv hv'
  obtain ⟨hm, hb, hleaf, hi⟩ := block_mem (pub := pub) (D := e.digests) hS hs hv
  obtain ⟨hm', hb', hleaf', -⟩ := block_mem (pub := pub) (D := e'.digests) hS' hs' hv'
  have hlv := hl _ (Nat.zero_le _) (by simpa using hi)
  rw [hleaf, hleaf'] at hlv
  have hrow : Merkle.rowDigest e.setup.store[s][v] = Merkle.rowDigest e'.setup.store[s][v] := by
    by_cases hli : leafInput pub.salt s v (Merkle.rowDigest e.setup.store[s][v]) =
        leafInput pub.salt s v (Merkle.rowDigest e'.setup.store[s][v])
    · exact (append_cancel hli rfl).2
    · exact absurd ⟨_, fork_T1 hm, _, fork_T2 hm', hli, hlv⟩ hX
  by_cases hbi : blockInput e.setup.store[s][v] = blockInput e'.setup.store[s][v]
  · have hsz : (HmRow.rowPrefixV2 (8 * e.setup.store[s][v].size)).size =
        (HmRow.rowPrefixV2 (8 * e'.setup.store[s][v].size)).size := by
      rw [(hS.2 s hs).2 v hv, (hS'.2 s hs').2 v hv']
    exact (append_cancel hbi hsz).2
  · exact absurd ⟨_, fork_B1 hb, _, fork_B2 hb', hbi, hrow⟩ hX

private theorem rows_eq {d d' : ByteArray} {n : Nat} (hd : d.size = n * Merkle.DIGEST)
    (hd' : d'.size = n * Merkle.DIGEST) (h : ∀ i < n, Run.row d i = Run.row d' i) : d = d' := by
  simp only [Merkle.DIGEST] at hd hd'
  apply ByteArray.ext
  apply Array.ext_getElem?
  intro k
  by_cases hk : k < n * 64
  · have e := congrArg (fun x : ByteArray => x.data[k % 64]?) (h (k / 64) (by omega))
    simp only [Run.row, Merkle.DIGEST, ByteArray.data_extract, Array.getElem?_extract, ByteArray.size_data, hd,
      hd'] at e
    have hc : k % 64 < min (64 * (k / 64 + 1)) (n * 64) - 64 * (k / 64) := by omega
    rw [if_pos hc, if_pos hc, Nat.div_add_mod] at e
    exact e
  · rw [Array.getElem?_eq_none (by simp only [ByteArray.size_data, hd]; omega),
      Array.getElem?_eq_none (by simp only [ByteArray.size_data, hd']; omega)]

private theorem digests_eq {q : P2.Params} {pub : SetupProtocol.Public} {e e' : RunProtocol.Evidence}
    (hd : e.digests.size = q.blocks * Merkle.DIGEST) (hd' : e'.digests.size = q.blocks * Merkle.DIGEST)
    (hl : ∀ i, 0 ≤ i → i < 0 + q.blocks →
      (Run.rowLeaves q pub.salt e.digests)[i]! = (Run.rowLeaves q pub.salt e'.digests)[i]!)
    (hX : ¬ ∃ x ∈ forkStrings q pub e e', ∃ y ∈ forkStrings q pub e e', x ≠ y ∧ Sha512.hash x = Sha512.hash y) :
    e.digests = e'.digests := by
  refine rows_eq hd hd' fun i hi => ?_
  have hlv := hl i (Nat.zero_le _) (by simpa using hi)
  rw [rowLeaves_getElem!At hi, rowLeaves_getElem!At hi] at hlv
  have hm := (mem_table_leafAt (ps := q) (pub := pub) (digests := e.digests) (C := e.setup.store) hi).1
  have hm' := (mem_table_leafAt (ps := q) (pub := pub) (digests := e'.digests) (C := e'.setup.store) hi).1
  by_cases hli : leafInput pub.salt (i / q.blocksPerSegment) (i % q.blocksPerSegment) (Run.row e.digests i) =
      leafInput pub.salt (i / q.blocksPerSegment) (i % q.blocksPerSegment) (Run.row e'.digests i)
  · exact (append_cancel hli rfl).2
  · exact absurd ⟨_, fork_T1 hm, _, fork_T2 hm', hli, hlv⟩ hX

private theorem setupCommitment_ext {a b : SetupProtocol.Commitment} (hw : a.w = b.w) (hs : a.salt = b.salt) :
    a = b := by
  cases a; cases b
  simp only at hw hs
  subst hw hs
  rfl

theorem runBinds : RunBinds := by
  intro p pub e e' hA hA'
  obtain ⟨hs, hc⟩ := (RunProtocol.accepts_iff _ _ _ _ _ _ _ _ _).mp ((RunProtocol.protocol_accepts p pub e).mp hA)
  obtain ⟨hs', hc'⟩ := (RunProtocol.accepts_iff _ _ _ _ _ _ _ _ _).mp ((RunProtocol.protocol_accepts p pub e').mp hA')
  by_cases hX : ∃ x ∈ forkStrings p.point.params pub e e', ∃ y ∈ forkStrings p.point.params pub e e',
      x ≠ y ∧ Sha512.hash x = Sha512.hash y
  · exact .inr hX
  have hP := PousCheck.Properties.SecurityProofs.publishesRoot _ _ _ pub () hs
  have hP' := PousCheck.Properties.SecurityProofs.publishesRoot _ _ _ pub () hs'
  refine .inl ⟨setupCommitment_ext (hP.2.1.symm.trans hP'.2.1) (hP.2.2.1.symm.trans hP'.2.2.1), ?_, ?_⟩
  · have hS := shaped_ofAt (PousCheck.Properties.SecurityProofs.acceptsDecodes p.point.params e.setup.commitment
      e.setup.store ⟨pub, (), hs⟩)
    have hS' := shaped_ofAt (PousCheck.Properties.SecurityProofs.acceptsDecodes p.point.params e'.setup.commitment
      e'.setup.store ⟨pub, (), hs'⟩)
    have hr : Merkle.root (leaves pub.salt e.setup.store) = some pub.root := by rw [hP.2.2.1]; exact hP.1
    have hr' : Merkle.root (leaves pub.salt e'.setup.store) = some pub.root := by rw [hP'.2.2.1]; exact hP'.1
    have e1 := root_mth hr
    rw [leaves_sizeAt hS] at e1
    have e2 := root_mth hr'
    rw [leaves_sizeAt hS'] at e2
    rcases mth_binds _ _ (leaves_entry pub.salt _) (leaves_entry pub.salt _) p.point.params.blocks 0
        (by rw [leaves_sizeAt hS]; omega) (by rw [leaves_sizeAt hS']; omega) (e1.trans e2.symm) with
      hl | ⟨x, hx, y, hy, hxy⟩
    · exact stores_eq hS hS' hl hX
    · exact absurd ⟨x, fork_T1 (mem_table_LAt hS hx), y, fork_T2 (mem_table_LAt hS' hy), hxy⟩ hX
  · obtain ⟨-, -, hd, hroot, -⟩ := check_okAt hc
    obtain ⟨-, -, hd', hroot', -⟩ := check_okAt hc'
    have e1 := root_mth hroot
    rw [rowLeaves_sizeAt] at e1
    have e2 := root_mth hroot'
    rw [rowLeaves_sizeAt] at e2
    rcases mth_binds _ _ (rowLeaves_entryAt pub.salt e.digests) (rowLeaves_entryAt pub.salt e'.digests)
        p.point.params.blocks 0 (by rw [rowLeaves_sizeAt]; omega) (by rw [rowLeaves_sizeAt]; omega)
        (e1.trans e2.symm) with
      hl | ⟨x, hx, y, hy, hxy⟩
    · exact digests_eq hd hd' hl hX
    · exact absurd ⟨x, fork_T1 (mem_table_TAt hx), y, fork_T2 (mem_table_TAt hy), hxy⟩ hX

/-! ## The wipe's acceptance -/

/-- `StreamedWipe.verdict`'s reasons before the batches. -/
private def wpre (schemes : Schemes) (trusted : List ByteArray) (s : StreamedWipe.Signed)
    (c : StreamedWipe.Commitment) : Array String := Id.run do
  let mut reasons : Array String := #[]
  match schemes.lookup s.scheme with
  | none => reasons := reasons.push s!"signature scheme {pyRepr s.scheme} is not accepted"
  | some p =>
    if !trusted.contains s.publicKey then
      reasons := reasons.push "the session is signed by a key that is not a trusted memory challenger's"
    else
      let ok := match s.message >>= fun m => lmsVerify p s.publicKey s.commitment.session m s.signature with
        | .ok b => b
        | .error _ => false
      if !ok then reasons := reasons.push "the signature does not verify"
  if !(s.commitment == c) then reasons := reasons.push "the signed header is not the session's commitment"
  let p := c.params
  if !p.covered then
    reasons := reasons.push s!"batch size {p.batch} is uncovered: the guarantee covers the sequential reveal, batch size 1"
  if s.ended != StreamedWipe.COMPLETE || s.batches.size != p.batches then
    reasons := reasons.push s!"the session ended {s.ended} after {s.batches.size} of {p.batches} batches"
  return reasons

/-- `StreamedWipe.verdict` from the batches on, after the reasons `r0`. -/
private def wtail (s : StreamedWipe.Signed) (c : StreamedWipe.Commitment) (r0 : Array String) :
    StreamedWipe.Verdict := Id.run do
  let mut reasons := r0
  let p := c.params
  for h : b in [0:s.batches.size] do
    let r := s.batches[b]
    if r.b != (b : Int) || r.first != ((b * p.batch : Nat) : Int) || r.count != min (p.batch : Int) ((p.k : Int) - r.first) then
      reasons := reasons.push s!"batch {b} is not the session's batch {b}"
    if r.wrong != 0 then reasons := reasons.push s!"batch {b} has {r.wrong} wrong answers"
    if !(0 ≤ r.elapsedNs && r.elapsedNs ≤ (p.limitNs : Int)) then
      reasons := reasons.push s!"batch {b} is past its deadline of {p.limitNs} ns"
    if b == 0 && r.sendNs < s.fillEndNs then reasons := reasons.push "the first batch was sent before the fill completed"
  return ⟨reasons.isEmpty, reasons, StreamedWipe.egressUnits s.egressBits⟩

private theorem wverdict_eq (schemes : Schemes) (trusted : List ByteArray) (s : StreamedWipe.Signed)
    (c : StreamedWipe.Commitment) :
    StreamedWipe.verdict schemes trusted s c = wtail s c (wpre schemes trusted s c) := by
  unfold StreamedWipe.verdict wtail wpre
  cases schemes.lookup s.scheme with
  | none =>
    by_cases h1 : (!(s.commitment == c)) = true <;> by_cases h2 : (!c.params.covered) = true <;>
    by_cases h3 : (s.ended != StreamedWipe.COMPLETE || s.batches.size != c.params.batches) = true <;>
      simp only [h1, h2, h3, ite_true, ite_false, Bool.false_eq_true] <;> rfl
  | some p =>
    cases hm : (s.message >>= fun m => lmsVerify p s.publicKey s.commitment.session m s.signature) with
    | error e =>
      by_cases h0 : (!trusted.contains s.publicKey) = true <;>
      by_cases h1 : (!(s.commitment == c)) = true <;> by_cases h2 : (!c.params.covered) = true <;>
      by_cases h3 : (s.ended != StreamedWipe.COMPLETE || s.batches.size != c.params.batches) = true <;>
        simp only [hm, h0, h1, h2, h3, ite_true, ite_false, Bool.false_eq_true, Bool.not_false] <;> rfl
    | ok b =>
      by_cases h0 : (!trusted.contains s.publicKey) = true <;> cases b <;>
      by_cases h1 : (!(s.commitment == c)) = true <;> by_cases h2 : (!c.params.covered) = true <;>
      by_cases h3 : (s.ended != StreamedWipe.COMPLETE || s.batches.size != c.params.batches) = true <;>
        simp only [hm, h0, h1, h2, h3, ite_true, ite_false, Bool.false_eq_true, Bool.not_false, Bool.not_true] <;>
        rfl

private theorem wtail_egress (s : StreamedWipe.Signed) (c : StreamedWipe.Commitment) (r0 : Array String) :
    (wtail s c r0).egressMib = StreamedWipe.egressUnits s.egressBits := rfl

/-- Batch `b` passes every check of the loop. -/
private def WGood (s : StreamedWipe.Signed) (c : StreamedWipe.Commitment) (b : Nat) : Prop :=
  ∀ hb : b < s.batches.size,
    (s.batches[b].b != (b : Int) || s.batches[b].first != ((b * c.params.batch : Nat) : Int) ||
        s.batches[b].count != min (c.params.batch : Int) ((c.params.k : Int) - s.batches[b].first)) = false ∧
      (s.batches[b].wrong != 0) = false ∧
      (!(decide (0 ≤ s.batches[b].elapsedNs) && decide (s.batches[b].elapsedNs ≤ (c.params.limitNs : Int)))) =
        false ∧
      (b == 0 && decide (s.batches[b].sendNs < s.fillEndNs)) = false

private theorem wtail_accepted (s : StreamedWipe.Signed) (c : StreamedWipe.Commitment) (r0 : Array String)
    (h : (wtail s c r0).accepted = true) : r0 = #[] ∧ ∀ b < s.batches.size, WGood s c b := by
  unfold wtail at h
  simp only [Id.run] at h
  rw [Std.Legacy.Range.forIn'_eq_forIn'_range'] at h
  generalize hL : (forIn' (m := Id) (List.range' _ _ _) (_ : Array String) _ : Id (Array String)) = L at h
  obtain ⟨m1, m2⟩ := forIn'_measure_eq hL (fun b => b.size) (WGood s c) (by
    intro a ha b
    clear h hL
    split_ifs with h1 h2 h3 h4 <;> refine ⟨_, rfl, ?_, fun he => ?_⟩ <;> (try simp at he) <;> (try simp) <;>
      (try omega)
    intro hb
    exact ⟨by simpa using h1, by simpa using h2, by simpa using h3, by simpa using h4⟩)
  rw [show ∀ (x : Id (Array String)) (f : Array String → Id StreamedWipe.Verdict), (x >>= f) = f x.run from
    fun _ _ => rfl] at h
  simp only [Id.run] at m1 m2 h
  have h0 : L.size = 0 := by simpa using h
  exact ⟨Array.eq_empty_of_size_eq_zero (by omega), fun b hb => m2 (by omega) b (by simp; omega)⟩

private theorem wpre_empty (schemes : Schemes) (trusted : List ByteArray) (s : StreamedWipe.Signed)
    (c : StreamedWipe.Commitment) (h : wpre schemes trusted s c = #[]) :
    (∃ p, schemes.lookup s.scheme = some p ∧ s.publicKey ∈ trusted ∧
      ∃ m, s.message = .ok m ∧ lmsVerify p s.publicKey s.commitment.session m s.signature = .ok true) ∧
    (s.commitment == c) = true ∧ c.params.covered = true ∧ s.ended = StreamedWipe.COMPLETE ∧
      s.batches.size = c.params.batches := by
  unfold wpre at h
  cases hp : schemes.lookup s.scheme with
  | none =>
    simp only [hp, Id.run] at h
    split_ifs at h <;> simp [pure] at h
  | some p =>
    cases hm : (s.message >>= fun m => lmsVerify p s.publicKey s.commitment.session m s.signature) with
    | error e =>
      simp only [hp, hm, Id.run] at h
      split_ifs at h <;> simp [pure] at h <;> simp_all
    | ok b =>
      cases b
      · simp only [hp, hm, Id.run] at h
        split_ifs at h <;> simp [pure] at h <;> simp_all
      simp only [hp, hm, Id.run] at h
      split_ifs at h <;> simp [pure] at h
      refine ⟨⟨p, rfl, by simp_all, ?_⟩, by simp_all, by simp_all, by simp_all, by simp_all⟩
      cases hs : s.message with
      | error e => simp [hs, bind, Except.bind] at hm
      | ok m => exact ⟨m, rfl, by simpa [hs, bind, Except.bind] using hm⟩

theorem wipeAccepts : WipeAccepts := by
  intro p c e out prof h
  have h' : WipeProtocol.check SCHEMES p c e = .accept out prof := h
  unfold WipeProtocol.check at h'
  split at h'
  · cases h'
  next hsv =>
    have hserv : c.server = p.server ∧ c.session = p.session := by
      simpa only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, not_not] using hsv
    cases hse : e.signed with
    | none => simp [hse] at h'
    | some s =>
      simp only [hse] at h'
      split at h'
      next hv =>
        simp only [Core.Outcome.accept.injEq] at h'
        obtain ⟨hout, hprof⟩ := h'
        rw [wverdict_eq] at hv hout hprof
        rw [wtail_egress] at hout hprof
        obtain ⟨h0, hg⟩ := wtail_accepted s c _ hv
        obtain ⟨hk, hcm, hcov, hend, hsize⟩ := wpre_empty _ _ s c h0
        refine ⟨hserv.1, hserv.2, s, rfl,
          ⟨hk, eq_of_beq hcm, by simpa [StreamedWipe.Params.covered] using hcov, hend, hsize, fun b hb => ?_⟩,
          hout.symm, hprof.symm⟩
        obtain ⟨g1, g2, g3, g4⟩ := hg b hb hb
        simp only [Bool.or_eq_false_iff, bne_eq_false_iff_eq] at g1 g2
        simp only [Bool.not_eq_false', Bool.and_eq_true, decide_eq_true_eq] at g3
        simp only [Bool.and_eq_false_imp, beq_iff_eq, decide_eq_false_iff_not, Int.not_lt] at g4
        exact ⟨g1.1.1, g1.1.2, g1.2, g2, g3.1, g3.2, g4⟩
      next => cases h'

end Pous.Component.SecurityProofs
