import Properties.Pous.Component
import Proofs.Properties.Pous.RunAt
import Proofs.Properties.Pous.MemoryChallenger
import Proofs.Properties.CertifierDevice.Lms
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

Each acceptance statement's two lemmas:

* nontrivial: `runAccepts_fails`, `runBinds_fails`, `wipeAccepts_fails`, each `¬ …By check` for every check that
  accepts everything, and `runBinds_break_fails`, that `RunBinds`' collision is not always there; `runAccepts_by`,
  `runBinds_by` and `wipeAccepts_by` read each guarantee as its `…By` at the compiled check;
* honest: `runAccepts_complete` (of `RunAccepts` and `RunBinds`) and `wipeAccepts_complete`, the converses, and
  `wipeAccepts_meets`, an honest wipe signed by any key `Lms.key` derives at S1's deployed parameters, which the
  wipe's check accepts. A run's instance is open, as setup's completeness is (`runAccepts_complete`).
-/

namespace Pous.Component.SecurityProofs

open Flock MemoryChallenger MemoryChallenger.Properties PousCheck PousCheck.Properties Pous.EndToEnd
open Pous.EndToEnd.SecurityProofs
open PousCheck.Properties.SecurityProofs (append_cancel mth_binds leaves_entry root_mth)
open MemoryChallenger.Properties.SecurityProofs (forIn'_measure_eq)

local instance instLawfulBEqByteArray_component : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := show (a.data == a.data) = true from beq_iff_eq.2 rfl

deriving instance ReflBEq, LawfulBEq for StreamedWipe.Params
deriving instance ReflBEq, LawfulBEq for StreamedWipe.Commitment

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
          exact ⟨s1, hc1, rfl⟩
        · simp [hf, hc1, hc2, hne, bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at h

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
    (hX : ¬ ∃ x ∈ forkStrings q pub e e', ∃ y ∈ forkStrings q pub e e',
      x ≠ y ∧ Sha512.hash x = Sha512.hash y) :
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
    rw [ite_eq_left hc, ite_eq_left hc, Nat.div_add_mod] at e
    exact e
  · rw [Array.getElem?_eq_none (by simp only [ByteArray.size_data, hd]; omega),
      Array.getElem?_eq_none (by simp only [ByteArray.size_data, hd']; omega)]

private theorem digests_eq {q : P2.Params} {pub : SetupProtocol.Public} {e e' : RunProtocol.Evidence}
    (hd : e.digests.size = q.blocks * Merkle.DIGEST) (hd' : e'.digests.size = q.blocks * Merkle.DIGEST)
    (hl : ∀ i, 0 ≤ i → i < 0 + q.blocks →
      (Run.rowLeaves q pub.salt e.digests)[i]! = (Run.rowLeaves q pub.salt e'.digests)[i]!)
    (hX : ¬ ∃ x ∈ forkStrings q pub e e', ∃ y ∈ forkStrings q pub e e',
      x ≠ y ∧ Sha512.hash x = Sha512.hash y) :
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
  obtain ⟨hs, hc⟩ :=
    (RunProtocol.accepts_iff _ _ _ _ _ _ _ _ _).mp ((RunProtocol.protocol_accepts p pub e).mp hA)
  obtain ⟨hs', hc'⟩ :=
    (RunProtocol.accepts_iff _ _ _ _ _ _ _ _ _).mp ((RunProtocol.protocol_accepts p pub e').mp hA')
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
    rename_i h2 h3 h4
    intro hb
    exact ⟨by simpa using h1, by simpa using h2, by simpa using h3, by simpa using h4⟩)
  rw [show ∀ (x : Id (Array String)) (f : Array String → Id StreamedWipe.Verdict), (x >>= f) = f x.run from
    fun _ _ => rfl] at h
  simp only [Id.run] at m1 m2 h
  have h0 : L.size = 0 := by simpa using (show Array.isEmpty L = true from h)
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

/-! ## Each guarantee's honest and nontrivial lemmas -/

theorem runAccepts_by : RunAccepts ↔ RunAcceptsBy RunProtocol.protocol.check := Iff.rfl

theorem runBinds_by : RunBinds ↔ RunBindsBy RunProtocol.protocol.check := Iff.rfl

theorem wipeAccepts_by : WipeAccepts ↔ WipeAcceptsBy WipeProtocol.protocol.check := Iff.rfl

/-- **`RunAccepts` is false of a check that accepts everything** (its nontrivial lemma). Such a check accepts a run
whose setup outputs carry a salt (one zero byte) its setup's commitment doesn't (none), which no setup publishes
(`PublishesRoot`), so the run is not `AcceptedAt`. -/
theorem runAccepts_fails : ∀ check, AcceptsAll check → ¬ RunAcceptsBy check := by
  intro check hall h
  obtain ⟨out, prof, hc⟩ := hall ⟨Run.DEPLOYED, .empty, #[]⟩ ⟨0, .empty, .empty, ⟨#[0]⟩⟩
    ⟨.empty, ⟨default, default, #[]⟩, ⟨⟨.empty, .empty⟩, #[]⟩⟩
  have hs := (h _ _ _ _ _ hc).1.1
  have hsalt := congrArg ByteArray.size (PousCheck.Properties.SecurityProofs.publishesRoot _ _ _ _ () hs).2.2.1
  exact absurd hsalt (by decide)

private theorem nodeInputs_zero (L : Array ByteArray) (lo : Nat) : nodeInputs L lo 0 = [] := by
  unfold nodeInputs
  simp

/-- At a point of no blocks, two runs with no `C*` hash nothing. -/
private theorem fork_nil {q : P2.Params} (hq : q.blocks = 0) (pub : SetupProtocol.Public) {e e' : RunProtocol.Evidence}
    (h : e.setup.store = #[]) (h' : e'.setup.store = #[]) : forkStrings q pub e e' = [] := by
  simp [forkStrings, tableStringsAt, Run.rowLeaves, leaves, flat, hq, h, h', nodeInputs_zero]

/-- **`RunBinds`' collision is not always there**: at a point of no blocks, two runs with no `C*` hash no strings, so
none of them collide. -/
theorem runBinds_break_fails : ¬ ∀ (q : P2.Params) (pub : SetupProtocol.Public) (e e' : RunProtocol.Evidence),
    (let xs := forkStrings q pub e e'
     ∃ x ∈ xs, ∃ y ∈ xs, x ≠ y ∧ Flock.Sha512.hash x = Flock.Sha512.hash y) := by
  intro h
  obtain ⟨x, hx, -⟩ := h ⟨0, 0, 0, 0⟩ ⟨0, .empty, .empty, .empty⟩
    ⟨.empty, ⟨default, default, #[]⟩, ⟨⟨.empty, .empty⟩, #[]⟩⟩ ⟨.empty, ⟨default, default, #[]⟩, ⟨⟨.empty, .empty⟩, #[]⟩⟩
  rw [fork_nil rfl _ rfl rfl] at hx
  simp at hx

/-- **`RunBinds` is false of a check that accepts everything** (its nontrivial lemma). Such a check accepts two runs on
one setup output at a point of no blocks, with no `C*` and setup commitments to different `W`s: they carry two setup
commitments, and their strings hold no collision (`runBinds_break_fails`'s pair). -/
theorem runBinds_fails : ∀ check, AcceptsAll check → ¬ RunBindsBy check := by
  intro check hall h
  rcases h ⟨⟨⟨0, 0, 0, 0⟩, 0, 0⟩, .empty, #[]⟩ ⟨0, .empty, .empty, .empty⟩
      ⟨.empty, ⟨default, default, #[]⟩, ⟨⟨.empty, .empty⟩, #[]⟩⟩
      ⟨.empty, ⟨default, default, #[]⟩, ⟨⟨⟨#[0]⟩, .empty⟩, #[]⟩⟩ (hall _ _ _) (hall _ _ _) with
    ⟨hw, -⟩ | ⟨x, hx, -⟩
  · exact absurd (congrArg (fun c => c.w.size) hw) (by decide)
  · rw [fork_nil rfl _ rfl rfl] at hx
    simp at hx

/-- **`WipeAccepts` is false of a check that accepts everything** (its nontrivial lemma). Such a check accepts a
session signed under a key the verifier doesn't trust (its list empty), which is not `WipeChecked`. -/
theorem wipeAccepts_fails : ∀ check, AcceptsAll check → ¬ WipeAcceptsBy check := by
  intro check hall h
  obtain ⟨out, prof, hc⟩ := hall ⟨.empty, .empty, #[]⟩ default ⟨some default⟩
  obtain ⟨-, -, s, -, ⟨⟨-, -, hk, -⟩, -⟩, -⟩ := h _ _ _ _ _ hc
  simp at hk

/-- **The compiled check accepts every run `AcceptedAt` accepts** (the honest lemma of `RunAccepts` and of `RunBinds`),
publishing `certify` of the point: `RunAccepts`' hypothesis holds exactly where the decision of record does, and two
runs it accepts are two accepted runs. That an honest run is `AcceptedAt` is setup's completeness
(`PousCheck.Properties.AcceptsComplete`, open) and the composed check's on an honest session: its instance waits on
the kernel evaluating `Commit.commit`, which reads the manifest through `Flock.canon` (well-founded recursion), and on
SHA-512 Merkle roots and coin openings; the difftest's honest scenario runs it on the compiled code. -/
theorem runAccepts_complete : ∀ (p : RunProtocol.Params) (pub : SetupProtocol.Public) (e : RunProtocol.Evidence),
    AcceptedAt p.point p.trusted.toList e.setup.commitment e.setup.store pub e.digests e.session p.sessionId →
      RunProtocol.protocol.check p pub e = .accept (RunProtocol.certify p.point) p.point := by
  intro p pub e h
  have ha := (RunProtocol.accepts_iff _ _ _ _ _ _ _ _ _).mpr h
  show RunProtocol.check SCHEMES p pub e = _
  unfold RunProtocol.check
  rw [ha]

private theorem forIn'_keep {α σ : Type} (l : List α) (f : (a : α) → a ∈ l → σ → Id (ForInStep σ))
    (hf : ∀ a h b, f a h b = pure (.yield b)) (init : σ) : forIn' l init f = pure init := by
  induction l generalizing init with
  | nil => rfl
  | cons a as ih =>
    rw [List.forIn'_cons, hf a List.mem_cons_self init]
    exact ih (fun a' m b => f a' (List.mem_cons_of_mem _ m) b) (fun a' m b => hf a' _ b) init

/-- `wtail` keeps its reasons when every batch passes. -/
private theorem wtail_keep (s : StreamedWipe.Signed) (c : StreamedWipe.Commitment) (r0 : Array String)
    (hg : ∀ b < s.batches.size, WGood s c b) :
    wtail s c r0 = ⟨r0.isEmpty, r0, StreamedWipe.egressUnits s.egressBits⟩ := by
  unfold wtail
  simp only [Id.run]
  rw [Std.Legacy.Range.forIn'_eq_forIn'_range', forIn'_keep]
  · rfl
  · intro b hb r
    have hb' : b < s.batches.size := by simp at hb; omega
    obtain ⟨g1, g2, g3, g4⟩ := hg b hb' hb'
    simp only [g1, g2, g3, g4, Bool.false_eq_true, ↓reduceIte] <;> rfl

private theorem wgood_of (s : StreamedWipe.Signed) (c : StreamedWipe.Commitment) (b : Nat)
    (h : ∀ hb : b < s.batches.size,
      s.batches[b].b = (b : Int) ∧ s.batches[b].first = ((b * c.params.batch : Nat) : Int) ∧
        s.batches[b].count = min (c.params.batch : Int) ((c.params.k : Int) - s.batches[b].first) ∧
        s.batches[b].wrong = 0 ∧ 0 ≤ s.batches[b].elapsedNs ∧ s.batches[b].elapsedNs ≤ (c.params.limitNs : Int) ∧
        (b = 0 → s.fillEndNs ≤ s.batches[b].sendNs)) : WGood s c b := by
  intro hb
  obtain ⟨h1, h2, h3, h4, h5, h6, h7⟩ := h hb
  refine ⟨by simp [h1, h2, h3], by simp [h4], by simp [h5, h6], ?_⟩
  by_cases h0 : b = 0
  · subst h0
    simp [Int.not_lt.2 (h7 rfl)]
  · simp [h0]

private theorem wpre_nil (schemes : Schemes) (trusted : List ByteArray) (s : StreamedWipe.Signed)
    (c : StreamedWipe.Commitment)
    (hk : ∃ p, schemes.lookup s.scheme = some p ∧ s.publicKey ∈ trusted ∧
      ∃ m, s.message = .ok m ∧ lmsVerify p s.publicKey s.commitment.session m s.signature = .ok true)
    (hcm : s.commitment = c) (hcov : c.params.batch = 1) (hend : s.ended = StreamedWipe.COMPLETE)
    (hsize : s.batches.size = c.params.batches) : wpre schemes trusted s c = #[] := by
  obtain ⟨p, hp, ht, m, hm, hv⟩ := hk
  have hok : (s.message >>= fun m => lmsVerify p s.publicKey s.commitment.session m s.signature) = .ok true := by
    rw [hm]
    exact hv
  have hc : trusted.contains s.publicKey = true := List.elem_eq_true_of_mem ht
  subst hcm
  unfold wpre
  simp [Id.run, hp, hok, hc, StreamedWipe.Params.covered, hcov, hend, hsize]

/-- **The wipe's compiled check accepts every signed session `WipeChecked` describes** (the honest lemma of
`WipeAccepts`, with `wipeAccepts_meets` its instance), on the period's server and session, publishing `certify` of
the commitment's parameters and the session's egress: `WipeAccepts`' hypothesis holds exactly where its conclusion
does. -/
theorem wipeAccepts_complete : ∀ (p : WipeProtocol.Params) (c : StreamedWipe.Commitment) (e : WipeProtocol.Evidence)
    (s : StreamedWipe.Signed), c.server = p.server → c.session = p.session → e.signed = some s →
      WipeChecked SCHEMES p.trusted.toList s c →
      WipeProtocol.protocol.check p c e =
        .accept (WipeProtocol.certify c.params (StreamedWipe.egressUnits s.egressBits))
          (c.params, StreamedWipe.egressUnits s.egressBits) := by
  intro p c e s hsv hss hse ⟨hk, hcm, hcov, hend, hsize, hb⟩
  have hv : StreamedWipe.verdict SCHEMES p.trusted.toList s c =
      ⟨true, #[], StreamedWipe.egressUnits s.egressBits⟩ := by
    rw [wverdict_eq, wpre_nil _ _ s c hk hcm hcov hend hsize,
      wtail_keep s c _ fun b _ => wgood_of s c b (hb b)] <;> rfl
  show WipeProtocol.check SCHEMES p c e = _
  unfold WipeProtocol.check
  simp [hsv, hss, hse, hv]

/-- The honest wipe's commitment: one 64-byte label, `k = 1` read revealed one index a batch, a 1 ns deadline, and one
tier the fill covers exactly (valid: `StreamedWipe.Params.check`). -/
private def honestCommitment : StreamedWipe.Commitment := ⟨.empty, .empty, ⟨1, 1, 1, 1, #[("gpu", 64)]⟩⟩

/-- Its one batch: batch 0, of index 0, sent when the fill ended and answered at once with no wrong answer. -/
private def honestBatch : StreamedWipe.Batch := ⟨0, 0, 1, 0, 0, 0, .empty⟩

/-- The session, ended complete under S1's scheme, the key `pk` and the signature `sig`. -/
private def honestSession (pk sig : ByteArray) : StreamedWipe.Signed :=
  ⟨honestCommitment, StreamedWipe.COMPLETE, 0, 0, #[honestBatch], 0, LMS, pk, sig⟩

private theorem honest_message :
    ∃ m, StreamedWipe.message honestCommitment StreamedWipe.COMPLETE 0 0 #[honestBatch] 0 = .ok m := by
  unfold StreamedWipe.message StreamedWipe.recordsRoot
  rw [Array.mapM_eq_mapM_toList]
  exact ⟨_, rfl⟩

/-- **An honest wipe is accepted** (the instance of `WipeAccepts`' honest lemma): for every key `Lms.key` derives at
S1's deployed LMS parameters (`Lms.deployed`, the one scheme `SCHEMES` accepts), the session of one batch at `k = 1`,
answered with no wrong answer and on time after the fill, ended complete and signed at leaf 0 with `Lms.signAt` on its
message, is accepted by the wipe's compiled check of a verifier that trusts that key, on the session's server and
session. The signature verifies by LMS's correctness (`Properties.CertifierDevice.Proofs.Lms.lmsCorrect`). -/
theorem wipeAccepts_meets : ∀ (I seed : ByteArray) (k : Lms.PrivateKey), Lms.key Lms.deployed I seed = .ok k →
    ∃ (c : StreamedWipe.Commitment) (e : WipeProtocol.Evidence) (out : WipeProtocol.Public)
      (prof : StreamedWipe.Params × Int),
      WipeProtocol.protocol.check ⟨c.server, c.session, #[k.publicKey]⟩ c e = .accept out prof := by
  intro I seed k hk
  obtain ⟨-, hIk, -, hIs, -, -, -⟩ := Properties.CertifierDevice.Proofs.Lms.key_spec hk
  subst hIk
  have hroot : (k.tree[1]!).size = Lms.deployed.N :=
    Properties.CertifierDevice.Proofs.Lms.node_size hk (by decide) 1 le_rfl (by decide)
  have hpub : k.publicKey.size = 16 + Lms.deployed.N := by
    simp only [Lms.PrivateKey.publicKey, ByteArray.size_append, hIs, hroot]
  have hI : k.publicKey.extract 0 16 = k.I := ByteArray.extract_append_eq_left hIs.symm
  obtain ⟨m, hm⟩ := honest_message
  obtain ⟨M, hM⟩ : ∃ M, memoryChallengerMessage k.I honestCommitment.session m = .ok M := by
    unfold memoryChallengerMessage
    simp [hIs]
  obtain ⟨sig, hsig⟩ :=
    Properties.CertifierDevice.Proofs.Lms.lmsSigns 64 1 20 false Lms.deployed k.I seed k 0 M rfl hk (by decide)
  obtain ⟨hver, hdec⟩ :=
    Properties.CertifierDevice.Proofs.Lms.lmsCorrect 64 1 20 false Lms.deployed k.I seed k 0 M sig rfl hk hsig
  have hlms : lmsVerify Lms.deployed k.publicKey honestCommitment.session m sig.encode = .ok true := by
    unfold lmsVerify
    simp only [hpub, ne_eq, not_true_eq_false, ↓reduceIte, hdec, hI, hM, hver, bind, Except.bind] <;> rfl
  refine ⟨honestCommitment, ⟨some (honestSession k.publicKey sig.encode)⟩, _, _,
    wipeAccepts_complete ⟨honestCommitment.server, honestCommitment.session, #[k.publicKey]⟩ honestCommitment
      ⟨some (honestSession k.publicKey sig.encode)⟩ (honestSession k.publicKey sig.encode)
      rfl rfl rfl ⟨⟨Lms.deployed, rfl, List.mem_singleton.2 rfl, m, hm, hlms⟩, rfl, rfl, rfl, rfl, ?_⟩⟩
  intro b hb
  obtain rfl : b = 0 := by simp [honestSession] at hb; omega
  simp [honestSession, honestBatch, honestCommitment, StreamedWipe.Batch.elapsedNs]

end Pous.Component.SecurityProofs
