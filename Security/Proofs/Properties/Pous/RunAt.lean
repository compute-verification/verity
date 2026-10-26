import Properties.Pous.EndToEndAt
import Proofs.Properties.Pous.Run

/-!
# The composed check's property at every point, proved

`runCheckAcceptsAt : RunCheckAcceptsAt` and `runCertifiesAt : RunCertifiesAt` (`Properties.Pous.EndToEndAt`):
`runCheckAccepts`'s and `runCertifies`'s proofs (`Proofs.Properties.Pous.Run`) with the point's parameters `ps` in
place of the deployed ones. The check's steps (`checkAt_ok`), the device's record (`inv_recorded`) and the list lemmas
are already point-free and are reused; the lemmas on `C*`'s blocks, leaves and the table's strings are restated at
`ps`. The one fact the deployed proof read off the point's numbers, `0 < blocksPerSegment`, follows from a challenged
index being below `ps.blocks`.
-/

namespace Pous.EndToEnd.SecurityProofs

open Flock MemoryChallenger MemoryChallenger.Properties PousCheck PousCheck.Properties
open PousCheck.Properties.SecurityProofs (hash_size append_cancel mth_binds leaves_entry root_mth)

local instance instLawfulBEqByteArray_runAt : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := show (a.data == a.data) = true from beq_iff_eq.2 rfl

variable {ps : P2.Params}

/-! ## `C*`'s blocks and leaves at `ps` -/

theorem segment_sizeAt {C : Array (Array ByteArray)} (hS : ShapedAt ps C) :
    ∀ seg ∈ C.toList, seg.size = ps.blocksPerSegment := by
  intro seg hseg
  obtain ⟨s, hs, rfl⟩ := Array.mem_iff_getElem.1 (Array.mem_toList_iff.1 hseg)
  exact (hS.2 s hs).1

theorem flat_getElem?At {C : Array (Array ByteArray)} (hS : ShapedAt ps C) (s v : Nat)
    (hv : v < ps.blocksPerSegment) :
    (flat C)[s * ps.blocksPerSegment + v]? = C[s]?.bind (·[v]?) := by
  rw [flat, ← Array.getElem?_toList, Array.toList_flatten,
    flatten_getElem? ps.blocksPerSegment _ (fun l hl => ?_) s v hv]
  · simp only [List.getElem?_map, Array.getElem?_toList]
    cases C[s]? <;> simp
  · obtain ⟨seg, hseg, rfl⟩ := List.mem_map.1 hl
    simpa using segment_sizeAt hS seg hseg

theorem leaves_listsAt {C : Array (Array ByteArray)} (hS : ShapedAt ps C) (salt : ByteArray) :
    ∀ l ∈ C.toList.mapIdx (fun s seg => seg.toList.mapIdx fun v block => Merkle.packageLeaf salt s v block),
      l.length = ps.blocksPerSegment := by
  intro l hl
  obtain ⟨s, hs, rfl⟩ := List.mem_iff_getElem.1 hl
  simp only [List.getElem_mapIdx, List.length_mapIdx, Array.length_toList]
  simp only [List.length_mapIdx, Array.length_toList] at hs
  exact (hS.2 s hs).1

theorem leaves_sizeAt {C : Array (Array ByteArray)} (hS : ShapedAt ps C) (salt : ByteArray) :
    (leaves salt C).size = ps.blocks := by
  rw [leaves, List.size_toArray, flatten_length ps.blocksPerSegment _ (leaves_listsAt hS salt)]
  simp [hS.1, P2.Params.blocks]

theorem leaves_getElem?At {C : Array (Array ByteArray)} (hS : ShapedAt ps C) (salt : ByteArray) (s v : Nat)
    (hv : v < ps.blocksPerSegment) :
    (leaves salt C)[s * ps.blocksPerSegment + v]? =
      C[s]?.bind fun seg => seg[v]?.map (Merkle.packageLeaf salt s v) := by
  rw [leaves, List.getElem?_toArray, flatten_getElem? ps.blocksPerSegment _ (leaves_listsAt hS salt) s v hv]
  simp only [List.getElem?_mapIdx, Array.getElem?_toList]
  cases C[s]? <;> simp [List.getElem?_mapIdx]

/-! ## The table's strings at `ps` -/

theorem rowLeaves_sizeAt (salt digests : ByteArray) : (Run.rowLeaves ps salt digests).size = ps.blocks := by
  simp [Run.rowLeaves]

theorem rowLeaves_entryAt (salt digests : ByteArray) :
    ∀ i (hi : i < (Run.rowLeaves ps salt digests).size), (Run.rowLeaves ps salt digests)[i].size = 64 := by
  intro i hi
  simp [Run.rowLeaves, Run.rowLeaf, hash_size]

theorem rowLeaves_getElem!At {salt digests : ByteArray} {i : Nat} (hi : i < ps.blocks) :
    (Run.rowLeaves ps salt digests)[i]! =
      Sha512.hash (leafInput salt (i / ps.blocksPerSegment) (i % ps.blocksPerSegment) (Run.row digests i)) := by
  rw [getElem!_pos _ i (by rw [rowLeaves_sizeAt]; exact hi)]
  simp [Run.rowLeaves, Run.rowLeaf, leafInput]

theorem mem_table_TAt {pub : SetupProtocol.Public} {digests : ByteArray} {C : Array (Array ByteArray)}
    {x : ByteArray} (hx : x ∈ nodeInputs (Run.rowLeaves ps pub.salt digests) 0 ps.blocks) :
    x ∈ tableStringsAt ps pub digests C := by
  unfold tableStringsAt
  dsimp only
  rw [rowLeaves_sizeAt]
  exact List.mem_append_left _ (List.mem_append_left _ hx)

theorem mem_table_LAt {pub : SetupProtocol.Public} {digests : ByteArray} {C : Array (Array ByteArray)}
    {x : ByteArray} (hS : ShapedAt ps C) (hx : x ∈ nodeInputs (leaves pub.salt C) 0 ps.blocks) :
    x ∈ tableStringsAt ps pub digests C := by
  unfold tableStringsAt
  dsimp only
  rw [leaves_sizeAt hS]
  exact List.mem_append_left _ (List.mem_append_right _ hx)

theorem mem_table_leafAt {pub : SetupProtocol.Public} {digests : ByteArray} {C : Array (Array ByteArray)} {i : Nat}
    (hi : i < ps.blocks) :
    leafInput pub.salt (i / ps.blocksPerSegment) (i % ps.blocksPerSegment) (Run.row digests i) ∈
        tableStringsAt ps pub digests C ∧
      leafInput pub.salt (i / ps.blocksPerSegment) (i % ps.blocksPerSegment) (Merkle.rowDigest ((flat C)[i]!)) ∈
        tableStringsAt ps pub digests C := by
  unfold tableStringsAt
  constructor <;>
  · refine List.mem_append_right _ (List.mem_flatMap.2 ⟨i, List.mem_range.2 hi, ?_⟩)
    simp

/-! ## The theorem -/

/-- What an accepted `checkAt` checked, at the point `pt`. -/
theorem check_okAt {pt : Run.Point} {trusted : List ByteArray} {pub : SetupProtocol.Public} {digests : ByteArray}
    {r : Run.Session} {sid : ByteArray} (h : Run.checkAt SCHEMES pt trusted pub digests r sid = .ok ()) :
    AuditsSetupAt pt pub r.commitment ∧ r.commitment.session = sid ∧
      digests.size = pt.params.blocks * Merkle.DIGEST ∧
      Merkle.root (Run.rowLeaves pt.params pub.salt digests) = some pub.root ∧
      (verdict SCHEMES trusted (Run.expected pt.params digests) r.signed r.commitment r.openings).accepted = true := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := checkAt_ok h
  exact ⟨⟨h1, h2, h5, h6, h7, h8, h9⟩, h10, h3, h4, h11⟩

theorem runCheckAcceptsAt : RunCheckAcceptsAt := by
  intro pt trusted pub digests r sid hcheck dev C
  obtain ⟨hau, hsid, -, hroot, hv⟩ := check_okAt hcheck
  obtain ⟨⟨p, hp, htr, m, hm, hlms⟩, hcom, hend, hsize, hrounds⟩ :=
    MemoryChallenger.Properties.SecurityProofs.verdictAccepts _ _ _ _ _ _ hv
  refine ⟨hau, hsid, htr, fun hS hCroot => ?_⟩
  by_cases hT : TableBreakAt pt.params pub digests C
  · exact .inr (.inr hT)
  by_cases hR : RunBreak trusted r.signed dev C
  · exact .inr (.inl hR)
  rcases MemoryChallenger.Properties.SecurityProofs.verdictSigned _ _ _ _ _ _ dev hv with
    ⟨run, hrun, hrc, hre, hrr⟩ | ⟨x, hx, y, hy, hxy⟩ | hforge
  rotate_left
  · exact absurd (.inl ⟨x, List.mem_append_left _ hx, y, List.mem_append_left _ hy, hxy⟩) hR
  · exact absurd (.inr hforge) hR
  left
  have hrsc : run.commitment = r.commitment := hrc.trans hcom
  have hk : run.coins.k = r.commitment.k := congrArg SessionCommitment.k hrsc
  have hlim : run.limitNs = r.commitment.limitNs := congrArg SessionCommitment.limitNs hrsc
  have hbb : run.blockBytes = r.commitment.blockBytes := congrArg SessionCommitment.blockBytes hrsc
  refine ⟨run, hrun, hrsc, ?_⟩
  obtain ⟨hle, -, hent⟩ := inv_recorded run
  have hrs : run.recorded.rounds.size = run.coins.k := by rw [hrr, hsize, hk]
  have hlen : (answers run).length = run.coins.k := by
    unfold answers; rw [List.length_take]; omega
  refine ⟨hlen, fun j hj => ?_⟩
  have hjk : j < run.coins.k := hlen ▸ hj
  obtain ⟨o, hko, hro⟩ := hent j (by omega)
  have hao : (answers run)[j] = o := by
    have : (answers run)[j]? = some o := by
      unfold answers; rw [List.getElem?_take_of_lt hjk]; exact hko
    rw [List.getElem?_eq_getElem hj] at this
    exact Option.some.inj this
  rw [hao]
  have hjs : j < r.signed.rounds.size := by rw [hsize, ← hk]; exact hjk
  obtain ⟨-, -, -, hexp, hel0, hel1⟩ := hrounds j hjs
  have hround : r.signed.rounds[j] = roundOf run.coins j o := by
    have := hro
    rw [hrr, Array.getElem?_eq_getElem hjs] at this
    exact Option.some.inj this
  rw [hround] at hexp hel0 hel1
  simp only [roundOf, MemoryChallenger.Round.elapsedNs] at hexp hel0 hel1
  have htime : o.sendNs ≤ o.receiptNs ∧ o.receiptNs ≤ o.sendNs + run.limitNs := by rw [hlim]; omega
  refine ⟨?_, htime⟩
  generalize hidx : run.coins.indices[j]! = idx at hexp
  have hidxB : idx < pt.params.blocks := by
    unfold Run.expected at hexp; split_ifs at hexp with hb; exact hb
  have hbps : 0 < pt.params.blocksPerSegment :=
    Nat.pos_of_ne_zero fun h0 => by simp [P2.Params.blocks, h0] at hidxB
  have hrow : Run.row digests idx = blockDigest o.block := by
    unfold Run.expected at hexp; simp only [hidxB, ite_true, Option.some.injEq] at hexp; exact hexp
  obtain ⟨hobs, hosz⟩ := List.mem_filter.1 (List.mem_of_getElem? hko)
  have hosz' : o.block.size = pt.params.blockBytes := by
    simp only [beq_iff_eq] at hosz; rw [hosz, hbb]; exact hau.2.2.2.2.2.1
  have hsC : idx / pt.params.blocksPerSegment < C.size := by
    rw [hS.1]; exact Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm]; exact hidxB)
  have hvC : idx % pt.params.blocksPerSegment < C[idx / pt.params.blocksPerSegment].size := by
    rw [(hS.2 _ hsC).1]; exact Nat.mod_lt _ hbps
  have hsplit : idx = idx / pt.params.blocksPerSegment * pt.params.blocksPerSegment +
      idx % pt.params.blocksPerSegment :=
    (Nat.div_add_mod' idx _).symm
  generalize hs : idx / pt.params.blocksPerSegment = s at hsC hvC hsplit
  generalize hv : idx % pt.params.blocksPerSegment = v at hvC hsplit
  have hv' : v < pt.params.blocksPerSegment := by rw [← hv]; exact Nat.mod_lt _ hbps
  have hflat : (flat C)[idx]? = some C[s][v] := by
    rw [hsplit, flat_getElem?At hS s v hv', Array.getElem?_eq_getElem hsC]
    simp [Array.getElem?_eq_getElem hvC]
  have hleaf : (leaves pub.salt C)[idx]! = Merkle.packageLeaf pub.salt s v C[s][v] := by
    apply getElem!_of_some
    rw [hsplit, leaves_getElem?At hS pub.salt s v hv', Array.getElem?_eq_getElem hsC]
    simp [Array.getElem?_eq_getElem hvC]
  have e1 := root_mth hroot
  rw [rowLeaves_sizeAt] at e1
  have e2 := root_mth hCroot
  rw [leaves_sizeAt hS] at e2
  rcases mth_binds _ _ (rowLeaves_entryAt pub.salt digests) (leaves_entry pub.salt C) pt.params.blocks 0
      (by rw [rowLeaves_sizeAt]; omega) (by rw [leaves_sizeAt hS]; omega) (e1.trans e2.symm) with
    hl | ⟨x, hx, y, hy, hxy⟩
  swap
  · exact absurd ⟨x, mem_table_TAt hx, y, mem_table_LAt hS hy, hxy⟩ hT
  have hlv := hl idx (Nat.zero_le _) (by omega)
  rw [rowLeaves_getElem!At hidxB, hleaf, hs, hv] at hlv
  obtain ⟨hm1, hm2⟩ := mem_table_leafAt (pub := pub) (digests := digests) (C := C) hidxB
  rw [hs, hv] at hm1
  rw [hs, hv, getElem!_of_some hflat] at hm2
  have hrow2 : Run.row digests idx = Merkle.rowDigest C[s][v] := by
    by_cases hli : leafInput pub.salt s v (Run.row digests idx) = leafInput pub.salt s v (Merkle.rowDigest C[s][v])
    · exact (append_cancel hli rfl).2
    · exact absurd ⟨_, hm1, _, hm2, hli, hlv⟩ hT
  have hbd : Sha512.hash (blockInput o.block) = Sha512.hash (blockInput C[s][v]) := hrow.symm.trans hrow2
  have hblk : o.block = C[s][v] := by
    by_cases hbi : blockInput o.block = blockInput C[s][v]
    · have hsz : (HmRow.rowPrefixV2 (8 * o.block.size)).size = (HmRow.rowPrefixV2 (8 * C[s][v].size)).size := by
        rw [hosz', ((hS.2 s hsC).2 v hvC)]
      exact (append_cancel hbi hsz).2
    · refine absurd (.inl ⟨blockInput o.block, List.mem_append_right _ (List.mem_append_left _ ?_),
        blockInput C[s][v], List.mem_append_right _ (List.mem_append_right _ ?_), hbi, hbd⟩) hR
      · exact List.mem_flatMap.2 ⟨run, hrun, List.mem_map.2 ⟨o, hobs, rfl⟩⟩
      · exact List.mem_map.2 ⟨C[s][v], Array.mem_toList_iff.2 (Array.mem_of_getElem? hflat), rfl⟩
  rw [hflat, hblk]

/-! ## With the setup check -/

theorem shaped_ofAt {c : SetupProtocol.Commitment} {C : Array (Array ByteArray)} (h : DecodesToCommitment ps c C) :
    ShapedAt ps C := by
  obtain ⟨W, ⟨hCs, -, hseg⟩, -⟩ := h
  refine ⟨hCs, fun s hs => ⟨(hseg s hs).1, fun v hv => ?_⟩⟩
  obtain ⟨-, -, -, -, -, -, hb, -⟩ := (hseg s hs).2 v hv
  exact hb

/-- `RunCertifiesAt` from the setup check's two properties, which quantify over every parameter set. -/
theorem runCertifiesAt : RunCertifiesAt := by
  intro pt trusted c C pub digests r sid dev ⟨hs, hc⟩
  have hD := PousCheck.Properties.SecurityProofs.acceptsDecodes pt.params c C ⟨pub, (), hs⟩
  have hP := PousCheck.Properties.SecurityProofs.publishesRoot pt.params c C pub () hs
  obtain ⟨ha, hsid, htr, hrest⟩ := runCheckAcceptsAt pt trusted pub digests r sid hc dev C
  have hroot : Merkle.root (leaves pub.salt C) = some pub.root := by rw [hP.2.2.1]; exact hP.1
  exact ⟨hD, hP, ha, hsid, htr, hrest (shaped_ofAt hD) hroot⟩

end Pous.EndToEnd.SecurityProofs
