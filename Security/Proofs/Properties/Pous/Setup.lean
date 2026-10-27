import Properties.Pous
import Proofs.Properties.Identity

/-!
# Proofs of the setup check's two properties (`Properties.Pous`): `acceptsDecodes` and `publishesRoot`

An accepted `SetupProtocol.protocol.check` validated the parameters and ran `Setup.verifierKeyFromEncoding` to the end.
Its loop over `C*`'s segments keeps `SetupSoFar` (`verifierKeyFromEncoding_ok`): every segment so far decoded
(`decodeSegment`), the commitment's hash state is `commitState` updated with their payloads in order, and the leaves are
theirs (`leavesUpTo`). The decoder's own loop keeps `SegmentSoFar` (`decodeSegment_ok`): each block decodes canonically
(`BlockDecodes`, from `decodeBlock_ok`) to a payload of `payloadBytes` bytes (`beOfNat_size`), and the segment is their
concatenation. Both loops are read by one lemma on `forIn'` over a range in `Except` (`forIn'_range'_inv`).

`acceptsDecodes`: `W` is the segments' payloads concatenated. Block `j` of segment `s` is its bytes at
`(s · blocksPerSegment + j) · payloadBytes` (`extract_chunk`), and `Commit.commit q W` is the digest the check compared:
a state represents the message it absorbed (`Properties.Identity.Rep`), so absorbing `W` whole and absorbing it segment by
segment finalize alike. `publishesRoot`: the published root is `Merkle.root` over the loop's leaves, which are
`leaves salt C`, and valid parameters give `C*` a block (`validate_ok`), so the root exists.
-/

namespace PousCheck.Properties.SecurityProofs

open Flock PousCheck PousCheck.Properties
open Properties.Identity (L L_length L_append Rep rep_init rep_update rep_foldl)

local instance instLawfulBEqByteArray_setup : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := show (a.data == a.data) = true from beq_iff_eq.2 rfl

/-! ## Loops -/

/-- A loop in `Except` over the indices `a, a + 1, …` that only yields: an invariant of the index each step keeps holds
at the end. -/
theorem forIn'_range'_inv {σ ε : Type} (I : Nat → σ → Prop) :
    ∀ (l : List Nat) (a : Nat) (init out : σ) (f : (s : Nat) → s ∈ l → σ → Except ε (ForInStep σ)),
      l = List.range' a l.length → I a init →
      (∀ s (hs : s ∈ l) r r', I s r → f s hs r = .ok r' → ∃ r'', r' = .yield r'' ∧ I (s + 1) r'') →
      forIn' l init f = .ok out → I (a + l.length) out := by
  intro l
  induction l with
  | nil =>
    intro a init out f _ h0 _ h
    rw [List.forIn'_nil] at h
    cases h
    exact h0
  | cons x l ih =>
    intro a init out f hl h0 hstep h
    have hl' : x = a ∧ l = List.range' (a + 1) l.length := by
      rw [List.length_cons, List.range'_succ] at hl
      exact List.cons.inj hl
    obtain ⟨rfl, hl⟩ := hl'
    rw [List.forIn'_cons] at h
    cases hfa : f x List.mem_cons_self init with
    | error e => rw [hfa] at h; cases h
    | ok r' =>
      obtain ⟨r'', rfl, hI⟩ := hstep x List.mem_cons_self init r' h0 hfa
      rw [hfa] at h
      have := ih (x + 1) r'' out _ hl hI (fun s hs r r2 hr hf => hstep s _ r r2 hr hf) h
      rwa [List.length_cons, Nat.add_comm l.length 1, ← Nat.add_assoc]

/-! ## Bytes -/

theorem L_inj {a b : ByteArray} (h : L a = L b) : a = b := ByteArray.ext (Array.toList_inj.1 h)

theorem L_extract' (b : ByteArray) (s e : Nat) : L (b.extract s e) = ((L b).drop s).take (e - s) := by
  simp only [L, ByteArray.data_extract, Array.toList_extract, List.extract_eq_take_drop]

theorem L_foldl_append : ∀ (xs : List ByteArray) (b : ByteArray),
    L (xs.foldl (· ++ ·) b) = L b ++ (xs.map L).flatten
  | [], b => by simp
  | x :: xs, b => by
    rw [List.foldl_cons, L_foldl_append xs, L_append, List.map_cons, List.flatten_cons, List.append_assoc]

theorem foldl_pushByte_size (g : Nat → UInt8) : ∀ (l : List Nat) (b : ByteArray),
    (l.foldl (fun b i => b.push (g i)) b).size = b.size + l.length
  | [], b => by simp
  | i :: l, b => by
    rw [List.foldl_cons, foldl_pushByte_size g l, ByteArray.size_push, List.length_cons]; omega

theorem beOfNat_size (n w : Nat) : (Prim.beOfNat n w).size = w := by
  rw [Prim.beOfNat]
  split
  · simp only [Id.run, Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl]
    change (List.foldl _ (ByteArray.emptyWithCapacity w) (List.range' 0 [:w].size)).size = w
    rw [foldl_pushByte_size, show (ByteArray.emptyWithCapacity w).size = 0 from rfl, List.length_range']
    simp [Std.Legacy.Range.size]
  · rw [ByteArray.size_append, beOfNat_size, beOfNat_size]; omega
termination_by w
decreasing_by all_goals omega

/-! ## Equal chunks -/

theorem length_flatten_chunks {α : Type} (n : Nat) :
    ∀ (xs : List (List α)), (∀ x ∈ xs, x.length = n) → xs.flatten.length = xs.length * n
  | [], _ => by simp
  | x :: xs, h => by
    rw [List.flatten_cons, List.length_append, h x List.mem_cons_self,
      length_flatten_chunks n xs fun y hy => h y (List.mem_cons_of_mem _ hy), List.length_cons, Nat.succ_mul]
    omega

theorem drop_flatten_chunks {α : Type} (n : Nat) :
    ∀ (xs : List (List α)) (j : Nat), (∀ x ∈ xs, x.length = n) → xs.flatten.drop (j * n) = (xs.drop j).flatten
  | [], j, _ => by simp
  | x :: xs, 0, _ => by simp
  | x :: xs, j + 1, h => by
    have hx := h x List.mem_cons_self
    rw [List.flatten_cons, List.drop_append, List.drop_eq_nil_of_le (by rw [hx, Nat.succ_mul]; omega), List.nil_append,
      hx, show (j + 1) * n - n = j * n by rw [Nat.succ_mul]; omega,
      drop_flatten_chunks n xs j fun y hy => h y (List.mem_cons_of_mem _ hy), List.drop_succ_cons]

/-- Chunk `j` of a flattening of `n`-element chunks is its `n` elements from `j · n`. -/
theorem take_drop_flatten_chunks {α : Type} (n : Nat) (xs : List (List α)) (j : Nat) (hj : j < xs.length)
    (h : ∀ x ∈ xs, x.length = n) : (xs.flatten.drop (j * n)).take n = xs[j] := by
  rw [drop_flatten_chunks n xs j h, List.drop_eq_getElem_cons hj, List.flatten_cons,
    List.take_append_of_le_length (Nat.le_of_eq (h _ (List.getElem_mem hj)).symm),
    List.take_of_length_le (Nat.le_of_eq (h _ (List.getElem_mem hj)))]

/-! ## The decoder -/

theorem decodeBlock_ok {q : P2.Params} {j k t : Nat} {block m : ByteArray} (h : P2.decodeBlock q j k t block = .ok m) :
    block.size = q.blockBytes ∧ Prim.natOfBE block < q.p ∧
      P2.decodeValue q.p k t (Prim.natOfBE block) < 2 ^ (8 * q.payloadBytes) ∧
      m = Prim.beOfNat (P2.decodeValue q.p k t (Prim.natOfBE block)) q.payloadBytes := by
  unfold P2.decodeBlock at h
  dsimp only at h
  split at h
  · cases h
  rename_i h1
  split at h
  · cases h
  rename_i h2
  split at h
  · cases h
  rename_i h3
  cases h
  simp only [bne_iff_ne, ne_eq, Decidable.not_not, Nat.shiftRight_eq_div_pow] at h1 h3
  exact ⟨h1, Nat.lt_of_not_le h2, (Nat.div_eq_zero_iff_lt (Nat.two_pow_pos _)).1 h3, rfl⟩

/-- What the decoder's loop has decoded after `n` blocks of segment `s`: one payload per block, each the block's
canonical decoding, and its output their concatenation. -/
def SegmentSoFar (q : P2.Params) (salt : ByteArray) (s : Nat) (blocks : Array ByteArray) (n : Nat) (out : ByteArray) :
    Prop :=
  ∃ ps : List ByteArray, ps.length = n ∧ (∀ p ∈ ps, (L p).length = q.payloadBytes) ∧
    (∀ j (hj : j < blocks.size) (hj' : j < ps.length), BlockDecodes q salt s j blocks[j] ps[j]) ∧
    L out = (ps.map L).flatten

theorem decodeSegment_ok {q : P2.Params} {salt : ByteArray} {s : Nat} {blocks : Array ByteArray} {seg : ByteArray}
    (h : Setup.decodeSegment q salt s blocks = .ok seg) :
    blocks.size = q.blocksPerSegment ∧ SegmentSoFar q salt s blocks blocks.size seg := by
  unfold Setup.decodeSegment at h
  dsimp only at h
  split at h
  · cases h
  split at h
  · cases h
  rename_i h1
  simp only [bne_iff_ne, ne_eq, Decidable.not_not] at h1
  refine ⟨h1, ?_⟩
  cases hpre : Keys.packagePrefix q salt with
  | error e => rw [hpre] at h; cases h
  | ok pre =>
    rw [hpre] at h
    simp only [Except.mapError, bind, Except.bind, pure, Except.pure] at h
    split at h
    · cases h
    rename_i out hloop
    cases h
    rw [Std.Legacy.Range.forIn'_eq_forIn'_range'] at hloop
    have := forIn'_range'_inv (SegmentSoFar q salt s blocks) _ 0 _ _ _ (by rw [List.length_range'])
      ⟨[], rfl, by simp, by simp, rfl⟩ ?_ hloop
    · simpa [Std.Legacy.Range.size] using this
    intro j hj r r' hI hf
    have hjlt : j < blocks.size := by
      simp only [List.mem_range'] at hj
      obtain ⟨i, hi, rfl⟩ := hj
      simp only [Std.Legacy.Range.size] at hi ⊢
      omega
    obtain ⟨ps, hlen, hpl, hbd, hout⟩ := hI
    split at hf
    · cases hf
    split at hf
    · cases hf
    rename_i hc
    cases hk : (Keys.Expander.ofPrefix pre).keys q s j with
    | error e => simp only [hk] at hf; cases hf
    | ok kta =>
      obtain ⟨k, t, a⟩ := kta
      cases hd : P2.decodeBlock q j k t blocks[j] with
      | error e => simp only [hk, hd] at hf; cases hf
      | ok m =>
        simp only [hk, hd, Except.ok.injEq] at hf
        subst hf
        obtain ⟨hbw, hbp, hbv, hm⟩ := decodeBlock_ok hd
        refine ⟨r ++ m, rfl, ps ++ [m], by rw [List.length_append, hlen]; rfl, ?_, ?_, ?_⟩
        · intro p hp
          rcases List.mem_append.1 hp with hp | hp
          · exact hpl p hp
          · rw [List.mem_singleton.1 hp, L_length, hm, beOfNat_size]
        · intro i hi hi'
          by_cases hij : i < ps.length
          · rw [List.getElem_append_left hij]
            exact hbd i hi hij
          · have hij' : i = j := by rw [List.length_append, hlen] at hi'; simp at hi'; omega
            subst hij'
            rw [List.getElem_append_right (Nat.le_of_not_lt hij)]
            simp only [hlen, Nat.sub_self, List.getElem_cons_zero]
            exact ⟨pre, k, t, a, hpre, hk, hbw, hbp, hbv, hm⟩
        · rw [L_append, hout, List.map_append, List.flatten_append, List.map_singleton, List.flatten_singleton]

/-! ## Setup's loop -/

/-- The leaves of the first `n` segments of `C*`, in block order (`leaves` at `n = C.size`). -/
def leavesUpTo (salt : ByteArray) (C : Array (Array ByteArray)) (n : Nat) : List ByteArray :=
  ((C.toList.take n).mapIdx fun s seg => seg.toList.mapIdx fun v block => Merkle.packageLeaf salt s v block).flatten

theorem leavesUpTo_succ (salt : ByteArray) (C : Array (Array ByteArray)) (n : Nat) (hn : n < C.size) :
    leavesUpTo salt C (n + 1) = leavesUpTo salt C n ++ (Setup.segmentLeaves salt n C[n]).toList := by
  unfold leavesUpTo
  rw [List.take_add_one, Array.getElem?_toList, Array.getElem?_eq_getElem hn, Option.toList_some, List.mapIdx_concat,
    List.flatten_append, List.flatten_singleton, List.length_take, Array.length_toList, Nat.min_eq_left (Nat.le_of_lt hn),
    Setup.segmentLeaves, Array.toList_mapIdx]

theorem leavesUpTo_size (salt : ByteArray) (C : Array (Array ByteArray)) :
    (leavesUpTo salt C C.size).toArray = leaves salt C := by
  unfold leavesUpTo leaves
  rw [List.take_of_length_le (Nat.le_of_eq Array.length_toList)]

/-- What setup's loop has checked after `n` segments: each decoded, the hash state absorbed their payloads in order from
`st0`, and the leaves are theirs. -/
def SetupSoFar (q : P2.Params) (salt : ByteArray) (C : Array (Array ByteArray)) (st0 : Identity.Sha512.State) (n : Nat)
    (r : Identity.Sha512.State × Array ByteArray) : Prop :=
  ∃ segs : List ByteArray, segs.length = n ∧
    (∀ i (hi : i < C.size) (hi' : i < segs.length), Setup.decodeSegment q salt i C[i] = .ok segs[i]) ∧
    r.1 = segs.foldl Identity.Sha512.State.update st0 ∧ r.2.toList = leavesUpTo salt C n

theorem verifierKeyFromEncoding_ok {q : P2.Params} {commitment salt : ByteArray} {C : Array (Array ByteArray)}
    {vk : Setup.VerifierKey} (h : Setup.verifierKeyFromEncoding q commitment salt C = .ok vk) :
    ∃ st0, Commit.commitState q = .ok st0 ∧ C.size = q.segments ∧
      ∃ segs : List ByteArray, segs.length = C.size ∧
        (∀ i (hi : i < C.size) (hi' : i < segs.length), Setup.decodeSegment q salt i C[i] = .ok segs[i]) ∧
        (segs.foldl Identity.Sha512.State.update st0).finalize = commitment ∧
        vk.root = (Merkle.root (leaves salt C)).getD .empty := by
  unfold Setup.verifierKeyFromEncoding at h
  dsimp only at h
  cases hst : Commit.commitState q with
  | error e => rw [hst] at h; cases h
  | ok st0 =>
    rw [hst] at h
    simp only [Except.mapError, bind, Except.bind, pure, Except.pure] at h
    split at h
    · cases h
    rename_i res hloop
    split at h
    · cases h
    rename_i hcount
    split at h
    · cases h
    rename_i hfin
    cases h
    simp only [bne_iff_ne, ne_eq, Decidable.not_not] at hcount hfin
    rw [Std.Legacy.Range.forIn'_eq_forIn'_range'] at hloop
    have := forIn'_range'_inv (SetupSoFar q salt C st0) _ 0 _ _ _ (by rw [List.length_range'])
      ⟨[], rfl, by simp, rfl, by simp [leavesUpTo]⟩ ?_ hloop
    · simp only [List.length_range', Std.Legacy.Range.size, Nat.zero_add, Nat.sub_zero, Nat.add_sub_cancel,
        Nat.div_one] at this
      obtain ⟨segs, hlen, hdec, hh, hlv⟩ := this
      refine ⟨st0, rfl, hcount, segs, hlen, hdec, by rw [← hh]; exact hfin, ?_⟩
      show (Merkle.root res.2).getD .empty = _
      rw [← leavesUpTo_size, ← hlv, Array.toArray_toList]
    intro i hi r r' hI hf
    have hilt : i < C.size := by
      simp only [List.mem_range'] at hi
      obtain ⟨k, hk, rfl⟩ := hi
      simp only [Std.Legacy.Range.size] at hk ⊢
      omega
    obtain ⟨segs, hlen, hdec, hh, hlv⟩ := hI
    split at hf
    · cases hf
    split at hf
    · cases hf
    cases hd : Setup.decodeSegment q salt i C[i] with
    | error e => simp only [hd] at hf; cases hf
    | ok seg =>
      simp only [hd, Except.ok.injEq] at hf
      subst hf
      refine ⟨_, rfl, segs ++ [seg], by rw [List.length_append, hlen]; rfl, ?_, ?_, ?_⟩
      · intro i' hi' hi''
        by_cases h' : i' < segs.length
        · rw [List.getElem_append_left h']
          exact hdec i' hi' h'
        · have : i' = i := by rw [List.length_append, hlen] at hi''; simp at hi''; omega
          subst this
          rw [List.getElem_append_right (Nat.le_of_not_lt h')]
          simp only [hlen, Nat.sub_self, List.getElem_cons_zero]
          exact hd
      · rw [List.foldl_append, List.foldl_cons, List.foldl_nil, ← hh]
      · rw [Array.toList_append, hlv, leavesUpTo_succ salt C i hilt]

/-! ## The commitment, the parameters and the check -/

theorem commitState_rep {q : P2.Params} {st0 : Identity.Sha512.State} (h : Commit.commitState q = .ok st0) :
    ∃ M, Rep st0 M := by
  unfold Commit.commitState Identity.taggedState at h
  cases hm : Commit.manifest q with
  | error e => rw [hm] at h; cases h
  | ok m =>
    rw [hm] at h
    cases hh : Identity.taggedHeader Commit.COMMIT_TAG (8 + m.size + q.wBytes) with
    | error e => simp only [hh, bind, Except.bind] at h; cases h
    | ok hd =>
      simp only [hh, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
      exact ⟨_, h ▸ rep_update (rep_update rep_init hd) _⟩

theorem validate_ok {q q' : P2.Params} (h : q.validate = .ok q') : 1 ≤ q.blocksPerSegment ∧ 1 ≤ q.segments := by
  unfold P2.Params.validate at h
  simp only [List.forIn_cons, List.forIn_nil, bind, Except.bind, pure, Except.pure, throw, throwThe,
    MonadExceptOf.throw] at h
  split at h
  · cases h
  rename_i h1
  refine ⟨Nat.pos_of_ne_zero fun hb => ?_, Nat.pos_of_ne_zero fun hs => ?_⟩
  · by_cases hw : q.w < 1 <;> by_cases hc : q.c0 < 1 <;> simp [hw, hc, hb] at h1
  · by_cases hw : q.w < 1 <;> by_cases hc : q.c0 < 1 <;> by_cases hb : q.blocksPerSegment < 1 <;>
      simp [hw, hc, hb, hs] at h1

theorem setupCheck_accept {q : P2.Params} {c : SetupProtocol.Commitment} {C : Array (Array ByteArray)}
    {pub : SetupProtocol.Public} {prof : Unit} (h : SetupProtocol.protocol.check q c C = .accept pub prof) :
    (∃ q', q.validate = .ok q') ∧ ∃ vk, Setup.verifierKeyFromEncoding q c.w c.salt C = .ok vk ∧
      pub = ⟨q.blocks, c.w, vk.root, c.salt⟩ := by
  change SetupProtocol.check q c C = .accept pub prof at h
  unfold SetupProtocol.check at h
  split at h
  · cases h
  rename_i q' hv
  split at h
  · rename_i vk hvk
    cases h
    exact ⟨⟨q', hv⟩, vk, hvk, rfl⟩
  · cases h

/-! ## The properties -/

/-- `C*` with `n` blocks in each segment has `n` leaves per segment. -/
theorem setupLeaves_size {salt : ByteArray} {C : Array (Array ByteArray)} {n : Nat}
    (h : ∀ s (hs : s < C.size), C[s].size = n) : (leaves salt C).size = C.size * n := by
  unfold leaves
  rw [List.size_toArray, length_flatten_chunks n _ ?_, List.length_mapIdx, Array.length_toList]
  intro x hx
  obtain ⟨i, hi, rfl⟩ := List.mem_mapIdx.1 hx
  rw [List.length_mapIdx, Array.length_toList, Array.getElem_toList]
  exact h i (by rwa [Array.length_toList] at hi)

theorem publishesRoot : PublishesRoot := by
  intro q c C pub prof h
  obtain ⟨⟨q', hv⟩, vk, hvk, rfl⟩ := setupCheck_accept h
  obtain ⟨-, -, hCs, segs, hlen, hdec, -, hroot⟩ := verifierKeyFromEncoding_ok hvk
  obtain ⟨hb, hs⟩ := validate_ok hv
  have hshape : ∀ s (hs : s < C.size), C[s].size = q.blocksPerSegment := fun s hs' =>
    (decodeSegment_ok (hdec s hs' (by omega))).1
  have hpos : (leaves c.salt C).size ≠ 0 := by
    rw [setupLeaves_size hshape, hCs]
    exact Nat.mul_ne_zero (by omega) (by omega)
  refine ⟨?_, rfl, rfl, rfl⟩
  show Merkle.root (leaves c.salt C) = some vk.root
  rw [hroot]
  unfold Merkle.root
  split
  · rename_i he
    exact absurd (Array.isEmpty_iff_size_eq_zero.1 he) hpos
  · rfl

/-- In `W`, the concatenation of segments of `n` chunks of `m` bytes, chunk `j` of segment `s` is the `m` bytes at
`(s · n + j) · m`. -/
theorem extract_chunk {W : ByteArray} {segs ps : List ByteArray} {n m s j : Nat} (hW : L W = (segs.map L).flatten)
    (hsegs : ∀ x ∈ segs.map L, x.length = n * m) (hs : s < segs.length) (hseg : L segs[s] = (ps.map L).flatten)
    (hps : ∀ x ∈ ps.map L, x.length = m) (hj : j < ps.length) :
    W.extract ((s * n + j) * m) ((s * n + j) * m + m) = ps[j] := by
  have hpl : ((ps.map L).flatten).length = ps.length * m := by
    rw [length_flatten_chunks m _ hps, List.length_map]
  have hjm : j * m ≤ ((ps.map L).flatten).length := by
    rw [hpl]; exact Nat.mul_le_mul_right m (Nat.le_of_lt hj)
  have hLj : (L ps[j]).length = m := hps _ (List.mem_map_of_mem (List.getElem_mem hj))
  apply L_inj
  rw [L_extract', Nat.add_sub_cancel_left, hW, Nat.add_mul, Nat.mul_assoc, ← List.drop_drop,
    drop_flatten_chunks _ _ _ hsegs, List.drop_eq_getElem_cons (l := segs.map L) (by rw [List.length_map]; exact hs),
    List.flatten_cons, List.getElem_map, hseg, List.drop_append_of_le_length hjm, drop_flatten_chunks _ _ _ hps,
    List.drop_eq_getElem_cons (l := ps.map L) (by rw [List.length_map]; exact hj), List.flatten_cons, List.getElem_map,
    List.append_assoc, List.take_append_of_le_length (Nat.le_of_eq hLj.symm),
    List.take_of_length_le (Nat.le_of_eq hLj)]

theorem acceptsDecodes : AcceptsDecodes := by
  rintro q c C ⟨pub, prof, h⟩
  obtain ⟨-, vk, hvk, -⟩ := setupCheck_accept h
  obtain ⟨st0, hst0, hCs, segs, hlen, hdec, hfin, -⟩ := verifierKeyFromEncoding_ok hvk
  have hseg : ∀ s (hs : s < segs.length),
      C[s].size = q.blocksPerSegment ∧ SegmentSoFar q c.salt s C[s] C[s].size segs[s] :=
    fun s hs => decodeSegment_ok (hdec s (by omega) hs)
  have hsegs : ∀ x ∈ segs.map L, x.length = q.blocksPerSegment * q.payloadBytes := by
    intro x hx
    obtain ⟨y, hy, rfl⟩ := List.mem_map.1 hx
    obtain ⟨s, hs, rfl⟩ := List.getElem_of_mem hy
    obtain ⟨hsz, ps, hps, hpl, -, hout⟩ := hseg s hs
    rw [hout, length_flatten_chunks q.payloadBytes _ ?_, List.length_map, hps, hsz]
    intro z hz
    obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hz
    exact hpl p hp
  have hLW : L (segs.foldl (· ++ ·) ByteArray.empty) = (segs.map L).flatten := by
    rw [L_foldl_append]; rfl
  have hWsize : (segs.foldl (· ++ ·) ByteArray.empty).size = q.wBytes := by
    rw [← L_length, hLW, length_flatten_chunks _ _ hsegs, List.length_map, hlen, hCs]
    unfold P2.Params.wBytes P2.Params.segmentBytes
    rfl
  refine ⟨segs.foldl (· ++ ·) ByteArray.empty, ⟨hCs, hWsize, fun s hs => ⟨(hseg s (by omega)).1, fun j hj => ?_⟩⟩, ?_⟩
  · obtain ⟨hsz, ps, hps, hpl, hbd, hout⟩ := hseg s (by omega)
    dsimp only
    rw [extract_chunk hLW hsegs (by omega) hout (fun x hx => ?_) (by omega)]
    · exact hbd j hj (by omega)
    · obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hx
      exact hpl p hp
  · obtain ⟨M, hM⟩ := commitState_rep hst0
    unfold Commit.commit
    simp only [hWsize, bne_self_eq_false, hst0, bind, Except.bind, pure, Except.pure, Bool.false_eq_true,
      ↓reduceIte, Except.ok.injEq]
    rw [(rep_update hM _).finalize, ← hfin, (rep_foldl segs hM).finalize]
    exact congrArg Flock.Sha512.hash (L_inj (by rw [L_append, hLW, L_foldl_append]))

end PousCheck.Properties.SecurityProofs
