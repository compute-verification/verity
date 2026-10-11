import Properties.Pous.EndToEnd
import Properties.Proofs.Pous.TimedAudit
import Properties.Proofs.Pous.MemoryChallenger

/-!
# The composed check's property, proved

`runCheckAccepts : RunCheckAccepts` (`Properties.Pous.EndToEnd`). An accepted `Run.check` passed every `need` in turn
(`checkAt_ok`), so the session audits the setup at the deployed point and the verdict accepted it: by `verdictAccepts`
each signed round is on time with the table's row for its index, and by `verdictSigned` the signed rounds are the
device's own record or the run carries a break. The device's record is its kept answers in order (`inv_recorded`), so
each round's digest is `blockDigest` of the answer it received. The table's leaves and `C*`'s have one root, so leaf
by leaf they agree or two inner nodes collide (`mth_binds`); a leaf's input then gives the row as `C*`'s block's row
digest, and the block's input the answer as the block, or two of the run's strings collide.

`runCertifies_of` composes it with the setup check's `AcceptsDecodes` and `PublishesRoot`, which are open.
-/

namespace Pous.EndToEnd.SecurityProofs

open Flock MemoryChallenger MemoryChallenger.Properties PousCheck PousCheck.Properties
open PousCheck.Properties.SecurityProofs (hash_size append_cancel mth_binds leaves_entry root_mth)

local instance instLawfulBEqByteArray_run : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := show (a.data == a.data) = true from beq_iff_eq.2 rfl

/-! ## The check -/

theorem need_bind {b : Bool} {reason : String} {f : Unit → Except String Unit}
    (h : (Run.need b reason >>= f) = .ok ()) : b = true ∧ f () = .ok () := by
  unfold Run.need at h
  cases b <;> simp_all [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw]

theorem need_ok {b : Bool} {reason : String} (h : Run.need b reason = .ok ()) : b = true := by
  unfold Run.need at h
  cases b <;> simp_all [throw, throwThe, MonadExceptOf.throw]

/-- What an accepted `checkAt` checked, in its order. -/
theorem checkAt_ok {schemes : Schemes} {pt : Run.Point} {pub : SetupProtocol.Public} {digests : ByteArray}
    {r : Run.Session} (h : Run.checkAt schemes pt pub digests r = .ok ()) :
    pub.blocks = pt.params.blocks ∧ pub.salt.size = Run.SALT ∧ digests.size = pt.params.blocks * Merkle.DIGEST ∧
      Merkle.root (Run.rowLeaves pt.params pub.salt digests) = some pub.root ∧
      r.commitment.blocks = pt.params.blocks ∧ r.commitment.k = pt.k ∧ r.commitment.limitNs ≤ pt.limitNs ∧
      r.commitment.blockBytes = pt.params.blockBytes ∧ r.commitment.binding = pub.root ∧
      (verdict schemes r.trusted.toList (Run.expected pt.params digests) r.signed r.commitment r.openings).accepted =
        true := by
  unfold Run.checkAt at h
  obtain ⟨h1, h⟩ := need_bind h
  obtain ⟨h2, h⟩ := need_bind h
  obtain ⟨h3, h⟩ := need_bind h
  obtain ⟨h4, h⟩ := need_bind h
  obtain ⟨h5, h⟩ := need_bind h
  obtain ⟨h6, h⟩ := need_bind h
  obtain ⟨h7, h⟩ := need_bind h
  obtain ⟨h8, h⟩ := need_bind h
  obtain ⟨h9, h⟩ := need_bind h
  have h10 := need_ok h
  simp only [beq_iff_eq, decide_eq_true_eq] at h1 h2 h3 h4 h5 h6 h7 h8 h9
  exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10⟩

/-! ## The device's record -/

/-- The round the device records for the `j`-th answer it keeps. -/
def roundOf (coins : CoinTree) (j : Nat) (o : Observed) : MemoryChallenger.Round :=
  ⟨j, coins.indices[j]!, o.sendNs, o.receiptNs, blockDigest o.block⟩

/-- A record of the answers `kept` so far: one round for each of the first ones, all of them while it hasn't ended. -/
def Inv (coins : CoinTree) (st : Recorded) (kept : List Observed) : Prop :=
  st.rounds.size ≤ kept.length ∧ (st.ended = none → st.rounds.size = kept.length) ∧
    ∀ j, j < st.rounds.size → ∃ o, kept[j]? = some o ∧ st.rounds[j]? = some (roundOf coins j o)

theorem round_ended {coins : CoinTree} {limitNs bb : Nat} {st : Recorded} {o : Observed} (h : st.ended.isSome) :
    Recorded.round coins limitNs bb st o = st := by
  simp [Recorded.round, h]

theorem round_skip {coins : CoinTree} {limitNs bb : Nat} {st : Recorded} {o : Observed}
    (h : ¬ (o.block.size == bb) = true) : Recorded.round coins limitNs bb st o = st := by
  simp only [beq_iff_eq] at h
  simp [Recorded.round, h]

theorem round_kept {coins : CoinTree} {limitNs bb : Nat} {st : Recorded} {o : Observed} (he : st.ended = none)
    (h : (o.block.size == bb) = true) :
    (Recorded.round coins limitNs bb st o).rounds = st.rounds.push (roundOf coins st.rounds.size o) := by
  simp only [beq_iff_eq] at h
  simp [Recorded.round, he, h, roundOf]

theorem inv_keep {coins : CoinTree} {limitNs bb : Nat} {st : Recorded} {kept : List Observed} {o : Observed}
    (h : (o.block.size == bb) = true) (hi : Inv coins st kept) :
    Inv coins (Recorded.round coins limitNs bb st o) (kept ++ [o]) := by
  obtain ⟨hle, hnone, hent⟩ := hi
  cases he : st.ended with
  | some e =>
    rw [round_ended (by simp [he])]
    refine ⟨by simp; omega, fun hn => by simp [he] at hn, fun j hj => ?_⟩
    obtain ⟨x, hx, hr⟩ := hent j hj
    exact ⟨x, by rw [List.getElem?_append_left (by omega)]; exact hx, hr⟩
  | none =>
    have hsz := hnone he
    have hr := round_kept (coins := coins) (limitNs := limitNs) he h
    refine ⟨by rw [hr]; simp; omega, fun _ => by rw [hr]; simp; omega, fun j hj => ?_⟩
    rw [hr] at hj ⊢
    simp only [Array.size_push] at hj
    by_cases hjs : j < st.rounds.size
    · obtain ⟨x, hx, hr⟩ := hent j hjs
      refine ⟨x, by rw [List.getElem?_append_left (by omega)]; exact hx, ?_⟩
      rw [Array.getElem?_push_lt hjs, ← Array.getElem?_eq_getElem]; exact hr
    · have hj' : j = st.rounds.size := by omega
      subst hj'
      refine ⟨o, by rw [List.getElem?_append_right (by omega), hsz]; simp, ?_⟩
      simp

theorem inv_skip {coins : CoinTree} {limitNs bb : Nat} {st : Recorded} {kept : List Observed} {o : Observed}
    (h : ¬ (o.block.size == bb) = true) (hi : Inv coins st kept) :
    Inv coins (Recorded.round coins limitNs bb st o) kept := by
  rw [round_skip h]; exact hi

theorem inv_foldl (coins : CoinTree) (limitNs bb : Nat) :
    ∀ (L : List Observed) (st : Recorded) (kept : List Observed), Inv coins st kept →
      Inv coins (L.foldl (Recorded.round coins limitNs bb) st) (kept ++ L.filter fun o => o.block.size == bb) := by
  intro L
  induction L with
  | nil => intro st kept h; simpa using h
  | cons o L ih =>
    intro st kept h
    rw [List.foldl_cons, List.filter_cons]
    by_cases hp : (o.block.size == bb) = true
    · rw [ite_eq_left_of_eq_true _ _ (by simpa using hp), show kept ++ o :: L.filter (fun o => o.block.size == bb) =
          (kept ++ [o]) ++ L.filter (fun o => o.block.size == bb) by simp]
      exact ih _ _ (inv_keep hp h)
    · rw [ite_eq_right_of_eq_false _ _ (by simpa using hp)]
      exact ih _ _ (inv_skip hp h)

/-- The device's record of a session is one round for each of its first kept answers, in order. -/
theorem inv_recorded (run : Run) :
    Inv run.coins run.recorded (run.observed.filter fun o => o.block.size == run.blockBytes) := by
  have := inv_foldl run.coins run.limitNs run.blockBytes run.observed ⟨#[], none⟩ [] ⟨by simp, by simp, by simp⟩
  unfold Run.recorded
  simpa using this

/-! ## `C*`'s blocks and leaves in block order -/

theorem flatten_length {α : Type} (m : Nat) :
    ∀ (Ls : List (List α)), (∀ l ∈ Ls, l.length = m) → Ls.flatten.length = Ls.length * m := by
  intro Ls
  induction Ls with
  | nil => simp
  | cons l Ls ih =>
    intro h
    simp only [List.flatten_cons, List.length_append, List.length_cons]
    rw [h l List.mem_cons_self, ih fun l' hl' => h l' (List.mem_cons_of_mem _ hl')]
    ring

theorem flatten_getElem? {α : Type} (m : Nat) :
    ∀ (Ls : List (List α)), (∀ l ∈ Ls, l.length = m) → ∀ s v, v < m →
      Ls.flatten[s * m + v]? = Ls[s]?.bind (·[v]?) := by
  intro Ls
  induction Ls with
  | nil => intro _ s v _; simp
  | cons l Ls ih =>
    intro h s v hv
    have hl := h l List.mem_cons_self
    simp only [List.flatten_cons]
    cases s with
    | zero =>
      simp only [Nat.zero_mul, Nat.zero_add, List.getElem?_cons_zero, Option.bind_some]
      exact List.getElem?_append_left (by omega)
    | succ s =>
      rw [List.getElem?_append_right (by rw [hl, Nat.succ_mul]; omega), hl,
        show (s + 1) * m + v - m = s * m + v by rw [Nat.succ_mul]; omega]
      simpa using ih (fun l' hl' => h l' (List.mem_cons_of_mem _ hl')) s v hv

theorem bps_pos : 0 < q.blocksPerSegment := Nat.two_pow_pos 16

theorem segment_size {C : Array (Array ByteArray)} (hS : Shaped C) :
    ∀ seg ∈ C.toList, seg.size = q.blocksPerSegment := by
  intro seg hseg
  obtain ⟨s, hs, rfl⟩ := Array.mem_iff_getElem.1 (Array.mem_toList_iff.1 hseg)
  exact (hS.2 s hs).1

theorem flat_getElem? {C : Array (Array ByteArray)} (hS : Shaped C) (s v : Nat) (hv : v < q.blocksPerSegment) :
    (flat C)[s * q.blocksPerSegment + v]? = C[s]?.bind (·[v]?) := by
  rw [flat, ← Array.getElem?_toList, Array.toList_flatten,
    flatten_getElem? q.blocksPerSegment _ (fun l hl => ?_) s v hv]
  · simp only [List.getElem?_map, Array.getElem?_toList]
    cases C[s]? <;> simp
  · obtain ⟨seg, hseg, rfl⟩ := List.mem_map.1 hl
    simpa using segment_size hS seg hseg

theorem leaves_lists {C : Array (Array ByteArray)} (hS : Shaped C) (salt : ByteArray) :
    ∀ l ∈ C.toList.mapIdx (fun s seg => seg.toList.mapIdx fun v block => Merkle.packageLeaf salt s v block),
      l.length = q.blocksPerSegment := by
  intro l hl
  obtain ⟨s, hs, rfl⟩ := List.mem_iff_getElem.1 hl
  simp only [List.getElem_mapIdx, List.length_mapIdx, Array.length_toList]
  simp only [List.length_mapIdx, Array.length_toList] at hs
  exact (hS.2 s hs).1

theorem leaves_size {C : Array (Array ByteArray)} (hS : Shaped C) (salt : ByteArray) :
    (leaves salt C).size = q.blocks := by
  rw [leaves, List.size_toArray, flatten_length q.blocksPerSegment _ (leaves_lists hS salt)]
  simp [hS.1, P2.Params.blocks]

theorem leaves_getElem? {C : Array (Array ByteArray)} (hS : Shaped C) (salt : ByteArray) (s v : Nat)
    (hv : v < q.blocksPerSegment) :
    (leaves salt C)[s * q.blocksPerSegment + v]? = C[s]?.bind fun seg => seg[v]?.map (Merkle.packageLeaf salt s v) := by
  rw [leaves, List.getElem?_toArray, flatten_getElem? q.blocksPerSegment _ (leaves_lists hS salt) s v hv]
  simp only [List.getElem?_mapIdx, Array.getElem?_toList]
  cases C[s]? <;> simp [List.getElem?_mapIdx]

theorem getElem!_of_some {L : Array ByteArray} {i : Nat} {x : ByteArray} (h : L[i]? = some x) : L[i]! = x := by
  have hi : i < L.size := by
    by_contra hn; rw [Array.getElem?_eq_none (by omega)] at h; cases h
  rw [getElem!_pos L i hi]
  rw [Array.getElem?_eq_getElem hi] at h
  exact Option.some.inj h

/-! ## The table's strings -/

theorem rowLeaves_size (salt digests : ByteArray) : (Run.rowLeaves q salt digests).size = q.blocks := by
  simp [Run.rowLeaves]

theorem rowLeaves_entry (salt digests : ByteArray) :
    ∀ i (hi : i < (Run.rowLeaves q salt digests).size), (Run.rowLeaves q salt digests)[i].size = 64 := by
  intro i hi
  simp [Run.rowLeaves, Run.rowLeaf, hash_size]

theorem rowLeaves_getElem! {salt digests : ByteArray} {i : Nat} (hi : i < q.blocks) :
    (Run.rowLeaves q salt digests)[i]! =
      Sha512.hash (leafInput salt (i / q.blocksPerSegment) (i % q.blocksPerSegment) (Run.row digests i)) := by
  rw [getElem!_pos _ i (by rw [rowLeaves_size]; exact hi)]
  simp [Run.rowLeaves, Run.rowLeaf, leafInput]

theorem mem_table_T {pub : SetupProtocol.Public} {digests : ByteArray} {C : Array (Array ByteArray)} {x : ByteArray}
    (hx : x ∈ nodeInputs (Run.rowLeaves q pub.salt digests) 0 q.blocks) : x ∈ tableStrings pub digests C := by
  unfold tableStrings
  dsimp only
  rw [rowLeaves_size]
  exact List.mem_append_left _ (List.mem_append_left _ hx)

theorem mem_table_L {pub : SetupProtocol.Public} {digests : ByteArray} {C : Array (Array ByteArray)} {x : ByteArray}
    (hS : Shaped C) (hx : x ∈ nodeInputs (leaves pub.salt C) 0 q.blocks) : x ∈ tableStrings pub digests C := by
  unfold tableStrings
  dsimp only
  rw [leaves_size hS]
  exact List.mem_append_left _ (List.mem_append_right _ hx)

theorem mem_table_leaf {pub : SetupProtocol.Public} {digests : ByteArray} {C : Array (Array ByteArray)} {i : Nat}
    (hi : i < q.blocks) :
    leafInput pub.salt (i / q.blocksPerSegment) (i % q.blocksPerSegment) (Run.row digests i) ∈
        tableStrings pub digests C ∧
      leafInput pub.salt (i / q.blocksPerSegment) (i % q.blocksPerSegment) (Merkle.rowDigest ((flat C)[i]!)) ∈
        tableStrings pub digests C := by
  unfold tableStrings
  constructor <;>
  · refine List.mem_append_right _ (List.mem_flatMap.2 ⟨i, List.mem_range.2 hi, ?_⟩)
    simp

/-! ## The theorem -/

/-- What an accepted `check` checked, at the deployed point. -/
theorem check_ok {pub : SetupProtocol.Public} {digests : ByteArray} {r : Run.Session}
    (h : Run.check pub digests r = .ok ()) :
    AuditsSetup pub r.commitment ∧ digests.size = q.blocks * Merkle.DIGEST ∧
      Merkle.root (Run.rowLeaves q pub.salt digests) = some pub.root ∧
      (verdict SCHEMES r.trusted.toList (Run.expected q digests) r.signed r.commitment r.openings).accepted = true := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10⟩ := checkAt_ok (schemes := SCHEMES) (pt := Run.DEPLOYED) h
  exact ⟨⟨h1, h2, h5, h6, h7, h8, h9⟩, h3, h4, h10⟩

theorem runCheckAccepts : RunCheckAccepts := by
  intro pub digests r hcheck dev C
  obtain ⟨hau, -, hroot, hv⟩ := check_ok hcheck
  obtain ⟨⟨p, hp, htr, m, hm, hlms⟩, hcom, hend, hsize, hrounds⟩ :=
    MemoryChallenger.Properties.SecurityProofs.verdictAccepts _ _ _ _ _ _ hv
  refine ⟨hau, htr, fun hS hCroot => ?_⟩
  by_cases hT : TableBreak pub digests C
  · exact .inr (.inr hT)
  by_cases hR : RunBreak r.trusted.toList r.signed dev C
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
  have hidxB : idx < q.blocks := by
    unfold Run.expected at hexp; split_ifs at hexp with hb; exact hb
  have hrow : Run.row digests idx = blockDigest o.block := by
    unfold Run.expected at hexp; simp only [hidxB, ite_true, Option.some.injEq] at hexp; exact hexp
  obtain ⟨hobs, hosz⟩ := List.mem_filter.1 (List.mem_of_getElem? hko)
  have hosz' : o.block.size = q.blockBytes := by
    simp only [beq_iff_eq] at hosz; rw [hosz, hbb]; exact hau.2.2.2.2.2.1
  have hsC : idx / q.blocksPerSegment < C.size := by
    rw [hS.1]; exact Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm]; exact hidxB)
  have hvC : idx % q.blocksPerSegment < C[idx / q.blocksPerSegment].size := by
    rw [(hS.2 _ hsC).1]; exact Nat.mod_lt _ bps_pos
  have hsplit : idx = idx / q.blocksPerSegment * q.blocksPerSegment + idx % q.blocksPerSegment :=
    (Nat.div_add_mod' idx _).symm
  generalize hs : idx / q.blocksPerSegment = s at hsC hvC hsplit
  generalize hv : idx % q.blocksPerSegment = v at hvC hsplit
  have hv' : v < q.blocksPerSegment := by rw [← hv]; exact Nat.mod_lt _ bps_pos
  have hflat : (flat C)[idx]? = some C[s][v] := by
    rw [hsplit, flat_getElem? hS s v hv', Array.getElem?_eq_getElem hsC]
    simp [Array.getElem?_eq_getElem hvC]
  have hleaf : (leaves pub.salt C)[idx]! = Merkle.packageLeaf pub.salt s v C[s][v] := by
    apply getElem!_of_some
    rw [hsplit, leaves_getElem? hS pub.salt s v hv', Array.getElem?_eq_getElem hsC]
    simp [Array.getElem?_eq_getElem hvC]
  have e1 := root_mth hroot
  rw [rowLeaves_size] at e1
  have e2 := root_mth hCroot
  rw [leaves_size hS] at e2
  rcases mth_binds _ _ (rowLeaves_entry pub.salt digests) (leaves_entry pub.salt C) q.blocks 0
      (by rw [rowLeaves_size]; omega) (by rw [leaves_size hS]; omega) (e1.trans e2.symm) with
    hl | ⟨x, hx, y, hy, hxy⟩
  swap
  · exact absurd ⟨x, mem_table_T hx, y, mem_table_L hS hy, hxy⟩ hT
  have hlv := hl idx (Nat.zero_le _) hidxB
  rw [rowLeaves_getElem! hidxB, hleaf, hs, hv] at hlv
  obtain ⟨hm1, hm2⟩ := mem_table_leaf (pub := pub) (digests := digests) (C := C) hidxB
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

theorem shaped_of {c : SetupProtocol.Commitment} {C : Array (Array ByteArray)} (h : DecodesToCommitment q c C) :
    Shaped C := by
  obtain ⟨W, ⟨hCs, -, hseg⟩, -⟩ := h
  refine ⟨hCs, fun s hs => ⟨(hseg s hs).1, fun v hv => ?_⟩⟩
  obtain ⟨-, -, -, -, -, -, hb, -⟩ := (hseg s hs).2 v hv
  exact hb

/-- `RunCertifies` from the setup check's two open properties: an accepted `C*` is of the point's shape and has the
published root, which is all `runCheckAccepts` asks of it. -/
theorem runCertifies_of (hd : AcceptsDecodes) (hp : PublishesRoot) : RunCertifies := by
  intro c C pub digests r dev ⟨hs, hc⟩
  have hD := hd q c C ⟨pub, (), hs⟩
  have hP := hp q c C pub () hs
  obtain ⟨ha, -, hrest⟩ := runCheckAccepts pub digests r hc dev C
  have hroot : Merkle.root (leaves pub.salt C) = some pub.root := by rw [hP.2.2.1]; exact hP.1
  exact ⟨hD, hP, ha, hrest (shaped_of hD) hroot⟩

end Pous.EndToEnd.SecurityProofs
