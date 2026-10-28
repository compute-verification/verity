import Properties.Pous.Component
import Proofs.Properties.Pous.RunAt
import Proofs.Properties.Pous.MemoryChallenger
import Proofs.Properties.CertifierDevice.Lms
import Mathlib.Tactic.SplitIfs
import Batteries.Tactic.OpenPrivate

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
* honest: `runAccepts_complete` (of `RunAccepts` and `RunBinds`) and `wipeAccepts_complete`, the converses, and their
  instances, each signed by any key `Lms.key` derives at S1's deployed parameters and accepted by its compiled check:
  `runAccepts_meets`, an honest run of setup's own accepted store (two 8-byte blocks) and one round, which the kernel
  evaluates (`decide +kernel`) once `Flock.canon`'s sort is shown to leave the manifest's sorted keys as they are
  (`qsort_sorted`); `runBinds_meets`, that run twice; and `wipeAccepts_meets`, an honest wipe.
-/

open private Array.qsort.sort Array.qpartition.loop from Init.Data.Array.QSort.Basic

namespace Pous.Component.SecurityProofs

open Flock MemoryChallenger MemoryChallenger.Properties PousCheck PousCheck.Properties Pous.EndToEnd
open Pous.EndToEnd.SecurityProofs
open PousCheck.Properties.SecurityProofs (append_cancel mth_binds leaves_entry root_mth)
open MemoryChallenger.Properties.SecurityProofs (forIn'_measure_eq)
open Lean (Json)

local instance instLawfulBEqByteArray_component : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := show (a.data == a.data) = true from beq_iff_eq.2 rfl

deriving instance ReflBEq, LawfulBEq for StreamedWipe.Params
deriving instance ReflBEq, LawfulBEq for StreamedWipe.Commitment
deriving instance ReflBEq, LawfulBEq for SessionCommitment

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
runs it accepts are two accepted runs. That every honest run is `AcceptedAt` is setup's completeness
(`PousCheck.Properties.AcceptsComplete`, open) and the composed check's on an honest session; `runAccepts_meets` is
one that is. -/
theorem runAccepts_complete : ∀ (p : RunProtocol.Params) (pub : SetupProtocol.Public) (e : RunProtocol.Evidence),
    AcceptedAt p.point p.trusted.toList e.setup.commitment e.setup.store pub e.digests e.session p.sessionId →
      RunProtocol.protocol.check p pub e = .accept (RunProtocol.certify p.point) p.point := by
  intro p pub e h
  have ha := (RunProtocol.accepts_iff _ _ _ _ _ _ _ _ _).mpr h
  show RunProtocol.check SCHEMES p pub e = _
  unfold RunProtocol.check
  rw [ha]
  rfl

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
    simp only [g1, g2, g3, g4, Bool.false_eq_true, ↓reduceIte]

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
  subst hcm
  unfold wpre
  simp [Id.run, hp, hok, StreamedWipe.Params.covered, hcov, hend, hsize]
  exact ite_eq_left ht

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
  rfl

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
    exact ⟨_, rfl⟩
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

/-! ## An honest run

The run of `runAccepts_meets`: setup's own accepted store (`SetupProtocol.HONEST`: `w = 64`, `c0 = 189`, one segment of
two 8-byte blocks, its `COMMITMENT` and `ROOT`) read once, under a one-coin tree with an all-zero key and salt, the round
answered at once with block 0. The kernel evaluates every check on it (`decide +kernel`), SHA-512 trees and the coin's
HM96 opening included, but `Flock.canon`'s `Array.qsort`, which it does not unfold (core's sort is well-founded and
private): the manifest is a flat object whose keys `Json.mkObj` already keeps in order, which `qsort_sorted` leaves as
they are. The session's signature is `Lms.signAt`'s, verified by `lmsCorrect`, as `wipeAccepts_meets`' is. -/

section QSort

variable {α : Type} {n : Nat}

private theorem swap_self (xs : Vector α n) (i : Nat) (h : i < n) : xs.swap i i h h = xs := by
  ext k hk
  rw [Vector.getElem_swap]
  split
  · subst_vars; rfl
  · rfl

section Loop

variable (lt : α → α → Bool) {lo hi : Nat} (hhi : hi < n) (pivot : α) (as : Vector α n)

private theorem loop_skip (i : Nat) (ilo : lo ≤ i) :
    ∀ (d k : Nat) (ik : i ≤ k) (w : k ≤ hi), hi - k = d →
      (∀ j (hj : j < n), k ≤ j → j < hi → lt as[j] pivot = false) →
      Array.qpartition.loop lt lo hi hhi pivot as i k ilo ik w =
        (⟨i, ilo, by omega⟩, as.swap i hi (by omega) hhi) := by
  intro d
  induction d with
  | zero =>
    intro k ik w hd _
    rw [Array.qpartition.loop.eq_def]
    have : ¬ k < hi := by omega
    simp only [this, dite_false]
  | succ d ih =>
    intro k ik w hd hb
    rw [Array.qpartition.loop.eq_def]
    have hk : k < hi := by omega
    simp only [hk, dite_true, hb k (by omega) (Nat.le_refl _) hk, Bool.false_eq_true, ite_false]
    exact ih (k + 1) (by omega) (by omega) (by omega) (fun j hj h1 h2 => hb j hj (by omega) h2)

private theorem loop_take (m : Nat) (hm : m ≤ hi) :
    ∀ (d k : Nat) (ilo : lo ≤ k) (w : k ≤ m), m - k = d →
      (∀ j (hj : j < n), k ≤ j → j < m → lt as[j] pivot = true) →
      (∀ j (hj : j < n), m ≤ j → j < hi → lt as[j] pivot = false) →
      Array.qpartition.loop lt lo hi hhi pivot as k k ilo (Nat.le_refl _) (by omega) =
        (⟨m, by omega, hm⟩, as.swap m hi (by omega) hhi) := by
  intro d
  induction d with
  | zero =>
    intro k ilo w hd _ hb
    obtain rfl : k = m := by omega
    exact loop_skip lt hhi pivot as k ilo (hi - k) k (Nat.le_refl _) _ rfl hb
  | succ d ih =>
    intro k ilo w hd ht hb
    rw [Array.qpartition.loop.eq_def]
    have hk : k < hi := by omega
    simp only [hk, dite_true, ht k (by omega) (Nat.le_refl _) (by omega), ite_true, swap_self]
    exact ih (k + 1) (by omega) (by omega) (by omega) (fun j hj h1 h2 => ht j hj (by omega) h2) hb

end Loop

/-- `as` is strictly sorted by `lt` on `[lo, hi]`: `lt` holds of two of its elements there exactly in their order. -/
private def SortedOn (lt : α → α → Bool) (lo hi : Nat) (as : Vector α n) : Prop :=
  ∀ i j (hi' : i < n) (hj : j < n), lo ≤ i → i ≤ hi → lo ≤ j → j ≤ hi → lt as[i] as[j] = decide (i < j)

private theorem qpartition_sorted (lt : α → α → Bool) (as : Vector α n) {lo hi : Nat} (w : lo ≤ hi)
    (hlo : lo < n) (hhi : hi < n) (hlt : lo < hi) (hs : SortedOn lt lo hi as) :
    Array.qpartition as lt lo hi w hlo hhi = (⟨(lo + hi) / 2, by omega, by omega⟩, as) := by
  have hm1 : lo ≤ (lo + hi) / 2 := by omega
  have hm2 : (lo + hi) / 2 < hi := by omega
  unfold Array.qpartition
  have h1 : lt (as[(lo + hi) / 2]'(by omega)) as[lo] = false := by
    rw [hs ((lo + hi) / 2) lo _ _ hm1 (by omega) (Nat.le_refl _) w]; simp; omega
  have h2 : lt as[hi] as[lo] = false := by
    rw [hs hi lo _ _ w (Nat.le_refl _) (Nat.le_refl _) w]; simp; omega
  have h3 : lt (as[(lo + hi) / 2]'(by omega)) as[hi] = true := by
    rw [hs ((lo + hi) / 2) hi _ _ hm1 (by omega) w (Nat.le_refl _)]; simp; omega
  simp only [h1, h2, h3, Bool.false_eq_true, ite_false, ite_true, Vector.getElem_swap_right]
  rw [loop_take lt hhi _ _ ((lo + hi) / 2) (Nat.le_of_lt hm2) _ lo (Nat.le_refl _) hm1 rfl]
  · simp only [Vector.swap_swap]
  · intro j hj h4 h5
    rw [Vector.getElem_swap_of_ne (by omega) (by omega), hs j ((lo + hi) / 2) _ _ h4 (by omega) hm1 (by omega)]
    simp; omega
  · intro j hj h4 h5
    rw [Vector.getElem_swap]
    split
    · rename_i e; subst e
      rw [hs hi ((lo + hi) / 2) _ _ w (Nat.le_refl _) hm1 (by omega)]; simp; omega
    · split
      · omega
      · rw [hs j ((lo + hi) / 2) _ _ (by omega) (by omega) hm1 (by omega)]; simp; omega

private theorem sort_sorted (lt : α → α → Bool) :
    ∀ (d : Nat) (as : Vector α n) (lo hi : Nat) (w : lo ≤ hi) (hlo : lo < n) (hhi : hi < n), hi - lo < d →
      SortedOn lt lo hi as → Array.qsort.sort lt as lo hi w hlo hhi = as := by
  intro d
  induction d with
  | zero => intro _ _ _ _ _ _ hd; omega
  | succ d ih =>
    intro as lo hi w hlo hhi hd hs
    rw [Array.qsort.sort.eq_def]
    split
    · rename_i hlt
      rw [qpartition_sorted lt as w hlo hhi hlt hs]
      simp only
      split
      · omega
      · rw [ih as lo ((lo + hi) / 2) _ hlo _ (by omega)
          (fun i j _ _ h1 h2 h3 h4 => hs i j _ _ h1 (by omega) h3 (by omega))]
        exact ih as ((lo + hi) / 2 + 1) hi _ _ hhi (by omega)
          (fun i j _ _ h1 h2 h3 h4 => hs i j _ _ (by omega) h2 (by omega) h4)
    · rfl

/-- `Array.qsort` leaves a strictly sorted array as it is: its pivot, the middle element, splits it in place. -/
private theorem qsort_sorted (as : Array α) (lt : α → α → Bool)
    (hs : ∀ i (hi : i < as.size) j (hj : j < as.size), lt as[i] as[j] = decide (i < j)) : as.qsort lt = as := by
  unfold Array.qsort
  split
  · rfl
  · simp only
    rw [sort_sorted lt _ as.toVector _ _ _ _ _ (Nat.lt_succ_self _)
      (fun i j hi hj _ _ _ _ => by simpa using hs i hi j hj)]

end QSort

private def jsonLeaf : Json → Bool
  | .null | .bool _ | .num _ | .str _ => true
  | .arr _ | .obj _ => false

/-- `Flock.canon` of a leaf. -/
private def canonLeaf : Json → V String
  | .null => pure "null"
  | .bool b => pure (if b then "true" else "false")
  | .num n => if n.exponent == 0 then pure (toString n.mantissa) else throw "canonical JSON: a non-integer number"
  | .str s => pure (canonStr s)
  | .arr _ | .obj _ => throw "canonical JSON: not a leaf"

/-- `Flock.canon` of an object of leaves. -/
private def canonObj (kvs : Std.TreeMap.Raw String Json) : V String := do
  let vals ← kvs.toArray.attach.mapM fun p => do pure (p.1.1, ← canonLeaf p.1.2)
  let items := vals.qsort (fun a b => a.1 < b.1)
  pure ("{" ++ ",".intercalate (items.toList.map fun kv => canonStr kv.1 ++ ":" ++ kv.2) ++ "}")

private def KeysSorted (as : Array (String × String)) : Prop :=
  ∀ i (hi : i < as.size) j (hj : j < as.size), decide (as[i].1 < as[j].1) = decide (i < j)

private instance instDecidableKeysSorted (as : Array (String × String)) : Decidable (KeysSorted as) := by
  unfold KeysSorted; infer_instance

/-- `canonObj`, sorting only items whose keys are out of order, so that the kernel evaluates it on sorted keys. -/
private def canonObjInOrder (kvs : Std.TreeMap.Raw String Json) : V String := do
  let vals ← kvs.toArray.attach.mapM fun p => do pure (p.1.1, ← canonLeaf p.1.2)
  let items := if KeysSorted vals then vals else vals.qsort (fun a b => a.1 < b.1)
  pure ("{" ++ ",".intercalate (items.toList.map fun kv => canonStr kv.1 ++ ":" ++ kv.2) ++ "}")

private theorem canonObj_inOrder (kvs : Std.TreeMap.Raw String Json) : canonObj kvs = canonObjInOrder kvs := by
  unfold canonObj canonObjInOrder
  congr 1
  funext vals
  split
  · rename_i h; rw [qsort_sorted vals (fun a b => decide (a.1 < b.1)) h]
  · rfl

private def jsonFlat (j : Json) : Bool := match j with
  | .obj kvs => kvs.toArray.all fun p => jsonLeaf p.2
  | _ => false

private def canonFlat (j : Json) : V String := match j with
  | .obj kvs => canonObjInOrder kvs
  | _ => throw "not an object"

private theorem canon_leaf (j : Json) (h : jsonLeaf j = true) : Flock.canon j = canonLeaf j := by
  cases j <;> first
    | (simp [jsonLeaf] at h; done)
    | simp only [Flock.canon, canonLeaf]

private theorem canon_obj (kvs : Std.TreeMap.Raw String Json) (h : (kvs.toArray.all fun p => jsonLeaf p.2) = true) :
    Flock.canon (.obj kvs) = canonObj kvs := by
  have hc : ∀ p : {x // x ∈ kvs.toArray}, Flock.canon p.1.2 = canonLeaf p.1.2 :=
    fun p => canon_leaf _ (Array.all_eq_true_iff_forall_mem.1 h p.1 p.2)
  rw [Flock.canon, canonObj]
  simp only [hc]

private theorem canon_flat (j : Json) (h : jsonFlat j = true) : Flock.canon j = canonFlat j := by
  cases j
  case obj kvs => exact (canon_obj kvs h).trans (canonObj_inOrder kvs)
  all_goals simp [jsonFlat] at h

private theorem eq_ok_of {α : Type} [DecidableEq α] {x : Except String α} {a : α}
    (h : (match x with | .ok b => decide (b = a) | .error _ => false) = true) : x = .ok a := by
  cases x with
  | error e => simp at h
  | ok b =>
    simp only [decide_eq_true_eq] at h
    rw [h]

private theorem ok_of_matches {x : Except String Unit} (h : (x matches .ok ()) = true) : x = .ok () := by
  cases x with
  | ok u => rfl
  | error e => simp at h

private def toyParams : P2.Params := ⟨64, 189, 2, 1⟩
private def toyPoint : Run.Point := ⟨toyParams, 1, 1000⟩
private def toyBytes : ByteArray := (unhex SetupProtocol.COMMITMENT).toOption.getD .empty
private def toyCommitment : SetupProtocol.Commitment := ⟨toyBytes.extract 0 64, toyBytes.extract 64 88⟩
private def toyBlocks : Array ByteArray := (SetupProtocol.HONEST.mapM unhex).toOption.getD #[]
private def toyStore : Array (Array ByteArray) := #[toyBlocks]
private def toyPublic : SetupProtocol.Public :=
  ⟨2, toyCommitment.w, (unhex SetupProtocol.ROOT).toOption.getD .empty, toyCommitment.salt⟩
private def toyManifest : String :=
  "{\"blocks_per_segment\":2,\"c0\":189,\"expander\":\"sha512-ctr-v4\",\"profile\":\"p2-6B\",\"scheme\":\"p2\"," ++
  "\"segments\":1,\"w\":64}"

private theorem canon_toy : Flock.canon (Commit.manifestJson toyParams) = .ok toyManifest := by
  rw [canon_flat (Commit.manifestJson toyParams) (by decide +kernel)]
  exact eq_ok_of (by decide +kernel)

private theorem manifest_toy : Commit.manifest toyParams = .ok toyManifest.toUTF8 := by
  unfold Commit.manifest
  rw [canon_toy]
  rfl

private theorem tie_matches : (RunProtocol.tie toyPoint toyCommitment toyStore toyPublic matches .ok ()) = true := by
  unfold RunProtocol.tie SetupProtocol.check Setup.verifierKeyFromEncoding Commit.commitState
  simp only [show toyPoint.params = toyParams from rfl, manifest_toy]
  decide +kernel

private theorem tie_toy : RunProtocol.tie toyPoint toyCommitment toyStore toyPublic = .ok () :=
  ok_of_matches tie_matches

private def toyDigests : ByteArray := blockDigest toyBlocks[0]! ++ blockDigest toyBlocks[1]!
private def toySid : ByteArray := ByteArray.mk (Array.replicate 32 7)
private def toyCoinKey : ByteArray := ByteArray.mk (Array.replicate HM96.keyBytes 0)
private def toyCoinSalt : ByteArray := ByteArray.mk (Array.replicate HM96.saltBytes 0)
private def toyCoins : CoinTree :=
  match CoinTree.make toySid 2 #[0] toyCoinKey #[toyCoinSalt] with
  | .ok t => t
  | .error _ => ⟨toySid, 2, #[0], toyCoinKey, #[toyCoinSalt], #[], .empty⟩
private def toySessionCommitment : SessionCommitment :=
  ⟨toySid, toyCoins.root, toyCoinKey, 2, 1, 1000, 8, toyPublic.root⟩
private def toyRound : MemoryChallenger.Round := ⟨0, 0, 0, 0, blockDigest toyBlocks[0]!⟩
private def toyOpenings : Array CoinOpening := #[toyCoins.opening 0]
private def toySigned (pk sig : ByteArray) : SignedSession :=
  ⟨toySessionCommitment, COMPLETE, #[toyRound], LMS, pk, sig⟩
private def toySession (pk sig : ByteArray) : Run.Session := ⟨toySigned pk sig, toySessionCommitment, toyOpenings⟩

/-- `verdict`'s reasons before the rounds. -/
private def rpre (schemes : Schemes) (trusted : List ByteArray) (s : SignedSession) (c : SessionCommitment) :
    Array String := Id.run do
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
  if s.ended != COMPLETE || s.rounds.size != c.k then
    reasons := reasons.push s!"the session ended {s.ended} after {s.rounds.size} of {c.k} rounds"
  return reasons

/-- `verdict` from the rounds `rounds` on, after the reasons `r0`. -/
private def rtail (expected : Nat → Option ByteArray) (rounds : Array MemoryChallenger.Round) (c : SessionCommitment)
    (openings : Array CoinOpening) (r0 : Array String) : Verdict := Id.run do
  let mut reasons := r0
  let byJ := fun (j : Nat) => openings.findRev? (·.j == j)
  let mut late := 0
  let mut wrong := 0
  for h : j in [0:rounds.size] do
    let r := rounds[j]
    let committed := match byJ j with
      | some o => r.j == j && o.index == r.index && verifyCoin c o
      | none => false
    if !committed then reasons := reasons.push s!"round {j}'s index is not its committed coin"
    if !(r.index < c.blocks && expected r.index == some r.digest) then wrong := wrong + 1
    if !(0 ≤ r.elapsedNs && r.elapsedNs ≤ c.limitNs) then late := late + 1
  if wrong > 0 then reasons := reasons.push s!"{wrong} answers are not their blocks of C"
  if late > 0 then reasons := reasons.push s!"{late} answers are past Δ + RTT = {c.limitNs} ns"
  return ⟨reasons.isEmpty, reasons, late, wrong⟩

private theorem rverdict_eq (schemes : Schemes) (trusted : List ByteArray) (expected : Nat → Option ByteArray)
    (s : SignedSession) (c : SessionCommitment) (openings : Array CoinOpening) :
    MemoryChallenger.verdict schemes trusted expected s c openings =
      rtail expected s.rounds c openings (rpre schemes trusted s c) := by
  unfold MemoryChallenger.verdict rtail rpre
  cases schemes.lookup s.scheme with
  | none =>
    by_cases h1 : (!s.commitment == c) = true <;> by_cases h2 : (s.ended != COMPLETE || s.rounds.size != c.k) = true <;>
      simp only [h1, h2, ite_true, ite_false, Bool.false_eq_true] <;> rfl
  | some p =>
    cases hm : (s.message >>= fun m => lmsVerify p s.publicKey s.commitment.session m s.signature) with
    | error e =>
      by_cases h0 : (!trusted.contains s.publicKey) = true <;>
      by_cases h1 : (!s.commitment == c) = true <;>
      by_cases h2 : (s.ended != COMPLETE || s.rounds.size != c.k) = true <;>
        simp only [hm, h0, h1, h2, ite_true, ite_false, Bool.false_eq_true, Bool.not_false] <;> rfl
    | ok b =>
      by_cases h0 : (!trusted.contains s.publicKey) = true <;> cases b <;>
      by_cases h1 : (!s.commitment == c) = true <;>
      by_cases h2 : (s.ended != COMPLETE || s.rounds.size != c.k) = true <;>
        simp only [hm, h0, h1, h2, ite_true, ite_false, Bool.false_eq_true, Bool.not_false, Bool.not_true] <;> rfl

private theorem rpre_nil (schemes : Schemes) (trusted : List ByteArray) (s : SignedSession) (c : SessionCommitment)
    (hk : ∃ p, schemes.lookup s.scheme = some p ∧ s.publicKey ∈ trusted ∧
      ∃ m, s.message = .ok m ∧ lmsVerify p s.publicKey s.commitment.session m s.signature = .ok true)
    (hcm : (s.commitment == c) = true) (hend : s.ended = COMPLETE) (hsize : s.rounds.size = c.k) :
    rpre schemes trusted s c = #[] := by
  obtain ⟨p, hp, ht, m, hm, hv⟩ := hk
  have hok : (s.message >>= fun m => lmsVerify p s.publicKey s.commitment.session m s.signature) = .ok true := by
    rw [hm]
    exact hv
  unfold rpre
  simp [Id.run, hp, hok, hcm, hend, hsize]
  exact ite_eq_left ht

private theorem rtail_toy :
    (rtail (Run.expected toyParams toyDigests) #[toyRound] toySessionCommitment toyOpenings #[]).accepted = true := by
  decide +kernel

private theorem verdict_toy (pk sig m : ByteArray)
    (hm : MemoryChallenger.message toySessionCommitment COMPLETE #[toyRound] = .ok m)
    (hlms : lmsVerify Lms.deployed pk toySessionCommitment.session m sig = .ok true) :
    (MemoryChallenger.verdict SCHEMES [pk] (Run.expected toyParams toyDigests) (toySigned pk sig) toySessionCommitment
      toyOpenings).accepted = true := by
  rw [rverdict_eq, rpre_nil SCHEMES [pk] (toySigned pk sig) toySessionCommitment
    ⟨Lms.deployed, rfl, List.mem_singleton.2 rfl, m, hm, hlms⟩ (beq_self_eq_true _) rfl rfl]
  exact rtail_toy

private theorem checkAt_of (schemes : Schemes) (pt : Run.Point) (trusted : List ByteArray)
    (pub : SetupProtocol.Public) (digests : ByteArray) (r : Run.Session) (sid : ByteArray)
    (h1 : (pub.blocks == pt.params.blocks) = true) (h2 : (pub.salt.size == Run.SALT) = true)
    (h3 : (digests.size == pt.params.blocks * Merkle.DIGEST) = true)
    (h4 : (Merkle.root (Run.rowLeaves pt.params pub.salt digests) == some pub.root) = true)
    (h5 : (r.commitment.blocks == pt.params.blocks) = true) (h6 : (r.commitment.k == pt.k) = true)
    (h7 : decide (r.commitment.limitNs ≤ pt.limitNs) = true)
    (h8 : (r.commitment.blockBytes == pt.params.blockBytes) = true) (h9 : (r.commitment.binding == pub.root) = true)
    (h10 : (r.commitment.session == sid) = true)
    (hv : (MemoryChallenger.verdict schemes trusted (Run.expected pt.params digests) r.signed r.commitment
      r.openings).accepted = true) :
    Run.checkAt schemes pt trusted pub digests r sid = .ok () := by
  unfold Run.checkAt
  simp only [h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, hv]
  rfl

private theorem salt_toy : (toyPublic.salt.size == Run.SALT) = true := by decide +kernel
private theorem digests_toy : (toyDigests.size == toyPoint.params.blocks * Merkle.DIGEST) = true := by decide +kernel
private theorem table_toy :
    (Merkle.root (Run.rowLeaves toyPoint.params toyPublic.salt toyDigests) == some toyPublic.root) = true := by
  decide +kernel

-- elaborating `checkAt_of`'s `rfl` arguments at the toy unfolds past the default depth (r20261027-032332-2478)
set_option maxRecDepth 100000 in
private theorem checkAt_toy (pk sig : ByteArray)
    (hv : (MemoryChallenger.verdict SCHEMES [pk] (Run.expected toyParams toyDigests) (toySigned pk sig)
      toySessionCommitment toyOpenings).accepted = true) :
    Run.checkAt SCHEMES toyPoint [pk] toyPublic toyDigests (toySession pk sig) toySid = .ok () :=
  checkAt_of SCHEMES toyPoint [pk] toyPublic toyDigests (toySession pk sig) toySid rfl salt_toy digests_toy
    table_toy rfl rfl rfl rfl (beq_self_eq_true _) (beq_self_eq_true _) hv

private theorem message_toy : ∃ m, MemoryChallenger.message toySessionCommitment COMPLETE #[toyRound] = .ok m := by
  unfold MemoryChallenger.message MemoryChallenger.recordsRoot
  rw [Array.mapM_eq_mapM_toList]
  exact ⟨_, rfl⟩

/-- **An honest run is accepted** (the instance of `RunAccepts`' honest lemma): for every key `Lms.key` derives at
S1's deployed LMS parameters, the run of setup's own accepted store (two 8-byte blocks, `w = 64`), one round at
`k = 1` answered with block 0 within its 1000 ns limit, ended complete and signed at leaf 0 with `Lms.signAt`, is
accepted by PoUS's compiled run check of a verifier that trusts that key, publishing `certify` of its point. -/
theorem runAccepts_meets : ∀ (I seed : ByteArray) (k : Lms.PrivateKey), Lms.key Lms.deployed I seed = .ok k →
    ∃ (pt : Run.Point) (sid : ByteArray) (pub : SetupProtocol.Public) (e : RunProtocol.Evidence),
      RunProtocol.protocol.check ⟨pt, sid, #[k.publicKey]⟩ pub e = .accept (RunProtocol.certify pt) pt := by
  intro I seed k hk
  obtain ⟨-, hIk, -, hIs, -, -, -⟩ := Properties.CertifierDevice.Proofs.Lms.key_spec hk
  subst hIk
  have hroot : (k.tree[1]!).size = Lms.deployed.N :=
    Properties.CertifierDevice.Proofs.Lms.node_size hk (by decide) 1 le_rfl (by decide)
  have hpub : k.publicKey.size = 16 + Lms.deployed.N := by
    simp only [Lms.PrivateKey.publicKey, ByteArray.size_append, hIs, hroot]
  have hI : k.publicKey.extract 0 16 = k.I := ByteArray.extract_append_eq_left hIs.symm
  obtain ⟨m, hm⟩ := message_toy
  obtain ⟨M, hM⟩ : ∃ M, memoryChallengerMessage k.I toySessionCommitment.session m = .ok M := by
    unfold memoryChallengerMessage
    simp [hIs]
    exact ⟨_, rfl⟩
  obtain ⟨sig, hsig⟩ :=
    Properties.CertifierDevice.Proofs.Lms.lmsSigns 64 1 20 false Lms.deployed k.I seed k 0 M rfl hk (by decide)
  obtain ⟨hver, hdec⟩ :=
    Properties.CertifierDevice.Proofs.Lms.lmsCorrect 64 1 20 false Lms.deployed k.I seed k 0 M sig rfl hk hsig
  have hlms : lmsVerify Lms.deployed k.publicKey toySessionCommitment.session m sig.encode = .ok true := by
    unfold lmsVerify
    simp only [hpub, ne_eq, not_true_eq_false, ↓reduceIte, hdec, hI, hM, hver, bind, Except.bind]
  have ha : RunProtocol.accepts SCHEMES toyPoint [k.publicKey] toyCommitment toyStore toyPublic toyDigests
      (toySession k.publicKey sig.encode) toySid = .ok () := by
    unfold RunProtocol.accepts
    rw [tie_toy, checkAt_toy k.publicKey sig.encode (verdict_toy k.publicKey sig.encode m hm hlms)]
    rfl
  exact ⟨toyPoint, toySid, toyPublic, ⟨toyDigests, toySession k.publicKey sig.encode, ⟨toyCommitment, toyStore⟩⟩,
    runAccepts_complete ⟨toyPoint, toySid, #[k.publicKey]⟩ toyPublic
      ⟨toyDigests, toySession k.publicKey sig.encode, ⟨toyCommitment, toyStore⟩⟩
      ((RunProtocol.accepts_iff _ _ _ _ _ _ _ _ _).mp ha)⟩

/-- **Two honest runs of one setup are accepted** (the instance of `RunBinds`' honest lemma): `runAccepts_meets`' run,
twice. -/
theorem runBinds_meets : ∀ (I seed : ByteArray) (k : Lms.PrivateKey), Lms.key Lms.deployed I seed = .ok k →
    ∃ (pt : Run.Point) (sid : ByteArray) (pub : SetupProtocol.Public) (e e' : RunProtocol.Evidence),
      RunProtocol.protocol.Accepts ⟨pt, sid, #[k.publicKey]⟩ pub e ∧
        RunProtocol.protocol.Accepts ⟨pt, sid, #[k.publicKey]⟩ pub e' := by
  intro I seed k hk
  obtain ⟨pt, sid, pub, e, h⟩ := runAccepts_meets I seed k hk
  exact ⟨pt, sid, pub, e, e, ⟨_, _, h⟩, ⟨_, _, h⟩⟩

end Pous.Component.SecurityProofs
