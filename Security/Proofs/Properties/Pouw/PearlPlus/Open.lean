module

public import Properties.Pouw.PearlPlus
public import Proofs.Properties.Pouw.Commit
import all Init.Data.String.Defs

/-!
# An accepted opening binds its position's value (`Properties.Pouw.PearlPlus.OpeningBinds`)

Unfolding `RowsDomain.verify` gives an accepted opening's shape: a path of `depth count` 64-byte siblings, a value of at
least its 192-byte salt (in a tile tree, a digest and its salt), and the climb (`vfold`) from its frame leaf to the root.
Two such climbs at one position meet at the root, so, one hash at a time from the root down, each pair of hashed messages
is equal or is the collision: the node frames (their parts have fixed sizes, so equal frames have equal children), the
leaf frames, the hm96 leaf's inputs (`Commit.leafValue_binding`'s steps), and in a row tree `sha512/row-seg/v1`'s outer
message, whose first block fixes the row's length, then a segment where two different rows differ.

It is on Lean core, as the properties are, so the row-seg steps are stated here on byte arrays rather than read from
C-Flock's binding proof (`FlockSoundness.Binding.RowSeg`), which is on Mathlib.
-/

@[expose] public section

namespace Properties.Pouw.PearlPlus

open PouwCommit PouwCheck.PearlPlus

theorem break_mono {L L' M M' : List ByteArray} (hL : ∀ x ∈ L, x ∈ M) (hL' : ∀ x ∈ L', x ∈ M') :
    Commit.Break L L' → Commit.Break M M'
  | ⟨x, hx, y, hy, hne, he⟩ => ⟨x, hL x hx, y, hL' y hy, hne, he⟩

/-! ## The climb -/

/-- One step of `Domain.verify`'s climb: the digest and its index, then the sibling and its level. -/
def vstep (d : Domain) (acc : ByteArray × Nat) (x : ByteArray × Nat) : ByteArray × Nat :=
  (if acc.2 % 2 == 0 then d.node x.2 (acc.2 / 2) acc.1 x.1 else d.node x.2 (acc.2 / 2) x.1 acc.1, acc.2 / 2)

/-- `Domain.verify`'s climb from the leaf digest `leaf` at `rank` along `path`: the digest it reaches and its index. -/
def vfold (d : Domain) (path : List ByteArray) (leaf : ByteArray) (rank : Nat) : ByteArray × Nat :=
  path.zipIdx.foldl (vstep d) (leaf, rank)

/-- The node frame hashed at level `lv` over the digest `dg` at index `cur` and its sibling `sib`. -/
def nodeMsg (d : Domain) (lv cur : Nat) (dg sib : ByteArray) : ByteArray :=
  if cur % 2 == 0 then frameBytes "node" [d.id, Flock.FV3.uint lv, Flock.FV3.uint (cur / 2), dg, sib]
  else frameBytes "node" [d.id, Flock.FV3.uint lv, Flock.FV3.uint (cur / 2), sib, dg]

/-- `Commit.pathFrames`'s step. -/
def pstep (d : Domain) (acc : List ByteArray × ByteArray × Nat) (x : ByteArray × Nat) :
    List ByteArray × ByteArray × Nat :=
  (acc.1 ++ [nodeMsg d x.2 acc.2.2 acc.2.1 x.1], d.hash.run (nodeMsg d x.2 acc.2.2 acc.2.1 x.1), acc.2.2 / 2)

def pfold (d : Domain) (path : List ByteArray) (leaf : ByteArray) (rank : Nat) : List ByteArray × ByteArray × Nat :=
  path.zipIdx.foldl (pstep d) ([], leaf, rank)

theorem pathFrames_eq (d : Domain) (rank : Nat) (leaf : ByteArray) (path : List ByteArray) :
    Commit.pathFrames d rank leaf path = (pfold d path leaf rank).1 := rfl

theorem node_size (d : Domain) (lv i : Nat) (l r : ByteArray) : (d.node lv i l r).size = d.hash.bytes :=
  Commit.run_size d.hash (frameBytes "node" [d.id, Flock.FV3.uint lv, Flock.FV3.uint i, l, r])

theorem leaf_run (f : Domain) (r : Nat) (s : String) (w : ByteArray) :
    f.leaf r s w = f.hash.run (frameBytes "leaf" [f.id, Flock.FV3.uint r, Flock.FV3.uint r, s.toUTF8, w]) := rfl

theorem run_sha512 {h : PouwCommit.Hash} (hd : h = .sha512) (m : ByteArray) : h.run m = Flock.Sha512.hash m := by
  subst hd; rfl

theorem run_nodeMsg (d : Domain) (lv cur : Nat) (dg sib : ByteArray) :
    d.hash.run (nodeMsg d lv cur dg sib) = (vstep d (dg, cur) (sib, lv)).1 := by
  unfold nodeMsg vstep
  cases cur % 2 == 0 <;> rfl

theorem foldl_pstep_snd (d : Domain) : ∀ (L : List (ByteArray × Nat)) (ms : List ByteArray) (acc : ByteArray × Nat),
    (L.foldl (pstep d) (ms, acc)).2 = L.foldl (vstep d) acc
  | [], _, _ => rfl
  | (sib, lv) :: L, ms, (dg, cur) => by
    have e : pstep d (ms, dg, cur) (sib, lv) = (ms ++ [nodeMsg d lv cur dg sib], vstep d (dg, cur) (sib, lv)) :=
      Prod.ext rfl (Prod.ext (run_nodeMsg d lv cur dg sib) rfl)
    show (L.foldl (pstep d) (pstep d (ms, dg, cur) (sib, lv))).2 = L.foldl (vstep d) (vstep d (dg, cur) (sib, lv))
    rw [e]
    exact foldl_pstep_snd d L _ _

theorem pfold_snd (d : Domain) (p : List ByteArray) (leaf : ByteArray) (rank : Nat) :
    (pfold d p leaf rank).2 = vfold d p leaf rank :=
  foldl_pstep_snd d _ [] (leaf, rank)

theorem foldl_vstep_snd (d : Domain) : ∀ (L : List (ByteArray × Nat)) (acc : ByteArray × Nat),
    (L.foldl (vstep d) acc).2 = acc.2 / 2 ^ L.length
  | [], acc => by simp
  | x :: L, acc => by
    rw [List.foldl_cons, foldl_vstep_snd d L, List.length_cons, Nat.pow_succ', ← Nat.div_div_eq_div_mul]
    rfl

theorem vfold_snd (d : Domain) (p : List ByteArray) (leaf : ByteArray) (rank : Nat) :
    (vfold d p leaf rank).2 = rank / 2 ^ p.length :=
  (foldl_vstep_snd d _ _).trans (by rw [List.length_zipIdx])

theorem foldl_vstep_size (d : Domain) (hb : d.hash.bytes = 64) :
    ∀ (L : List (ByteArray × Nat)) (acc : ByteArray × Nat), acc.1.size = 64 → (L.foldl (vstep d) acc).1.size = 64
  | [], _, h => h
  | x :: L, acc, _ => foldl_vstep_size d hb L _ (by
      show (if acc.2 % 2 == 0 then _ else _ : ByteArray).size = 64
      split <;> exact (node_size ..).trans hb)

theorem vfold_size (d : Domain) (hb : d.hash.bytes = 64) (p : List ByteArray) (leaf : ByteArray) (rank : Nat)
    (hl : leaf.size = 64) : (vfold d p leaf rank).1.size = 64 :=
  foldl_vstep_size d hb _ _ hl

theorem concat_of_length {α : Type} {p : List α} {n : Nat} (h : p.length = n + 1) :
    ∃ q s, p = q ++ [s] ∧ q.length = n := by
  rcases List.eq_nil_or_concat p with rfl | ⟨q, s, rfl⟩
  · simp at h
  · exact ⟨q, s, List.concat_eq_append, by simpa using h⟩

theorem vfold_concat (d : Domain) (q : List ByteArray) (s leaf : ByteArray) (rank : Nat) :
    vfold d (q ++ [s]) leaf rank = vstep d (vfold d q leaf rank) (s, q.length) := by
  simp [vfold, List.zipIdx_append]

theorem pfold_concat (d : Domain) (q : List ByteArray) (s leaf : ByteArray) (rank : Nat) :
    pfold d (q ++ [s]) leaf rank = pstep d (pfold d q leaf rank) (s, q.length) := by
  simp [pfold, List.zipIdx_append]

theorem pstep_fst (d : Domain) (acc : List ByteArray × ByteArray × Nat) (x : ByteArray × Nat) :
    (pstep d acc x).1 = acc.1 ++ [nodeMsg d x.2 acc.2.2 acc.2.1 x.1] := rfl

/-- **Two climbs at one rank that reach one digest** start from one leaf digest, or two different node frames on their
paths have one SHA-512 digest. -/
theorem climb (d : Domain) (hd : d.hash = .sha512) (rank : Nat) :
    ∀ (n : Nat) (p p' : List ByteArray) (leaf leaf' : ByteArray), p.length = n → p'.length = n →
      (∀ s ∈ p, s.size = 64) → (∀ s ∈ p', s.size = 64) → leaf.size = 64 → leaf'.size = 64 →
      (vfold d p leaf rank).1 = (vfold d p' leaf' rank).1 →
        leaf = leaf' ∨ Commit.Break (pfold d p leaf rank).1 (pfold d p' leaf' rank).1
  | 0, p, p', leaf, leaf', hp, hp', _, _, _, _, h => by
    obtain rfl := List.eq_nil_of_length_eq_zero hp
    obtain rfl := List.eq_nil_of_length_eq_zero hp'
    exact .inl h
  | n + 1, p, p', leaf, leaf', hp, hp', hs, hs', hl, hl', h => by
    obtain ⟨q, s, rfl, hq⟩ := concat_of_length hp
    obtain ⟨q', s', rfl, hq'⟩ := concat_of_length hp'
    have hb : d.hash.bytes = 64 := by rw [hd]; rfl
    rw [vfold_concat, vfold_concat] at h
    rw [pfold_concat, pfold_concat, pstep_fst, pstep_fst, pfold_snd, pfold_snd]
    have hc : (vfold d q leaf rank).2 = (vfold d q' leaf' rank).2 := by rw [vfold_snd, vfold_snd, hq, hq']
    have e1 : Flock.Sha512.hash (nodeMsg d q.length (vfold d q leaf rank).2 (vfold d q leaf rank).1 s) =
        Flock.Sha512.hash (nodeMsg d q'.length (vfold d q' leaf' rank).2 (vfold d q' leaf' rank).1 s') := by
      rw [← run_sha512 hd, ← run_sha512 hd, run_nodeMsg, run_nodeMsg]
      exact h
    rcases Commit.step (List.mem_append_right _ (List.mem_singleton_self _))
      (List.mem_append_right _ (List.mem_singleton_self _)) e1 with e | b
    case inr => exact .inr b
    have ha := vfold_size d hb q leaf rank hl
    have ha' := vfold_size d hb q' leaf' rank hl'
    have hsz := hs s (List.mem_append_right _ (List.mem_singleton_self _))
    have hsz' := hs' s' (List.mem_append_right _ (List.mem_singleton_self _))
    rw [hq, hq', ← hc] at e
    have hdg : (vfold d q leaf rank).1 = (vfold d q' leaf' rank).1 := by
      unfold nodeMsg at e
      revert e
      cases (vfold d q leaf rank).2 % 2 == 0 <;> intro e
      · have e := Commit.frameBytes_sizes _ _ _ (by simp only [List.map_cons, List.map_nil, ha, ha', hsz, hsz']) e
        simp only [List.cons.injEq] at e
        exact e.2.2.2.2.1
      · have e := Commit.frameBytes_sizes _ _ _ (by simp only [List.map_cons, List.map_nil, ha, ha', hsz, hsz']) e
        simp only [List.cons.injEq] at e
        exact e.2.2.2.1
    rcases climb d hd rank n q q' leaf leaf' hq hq' (fun x hx => hs x (List.mem_append_left _ hx))
      (fun x hx => hs' x (List.mem_append_left _ hx)) hl hl' hdg with e | b
    · exact .inl e
    · exact .inr (break_mono (fun x hx => List.mem_append_left _ hx) (fun x hx => List.mem_append_left _ hx) b)

/-! ## An accepted opening -/

/-- An hm96 value's salt. -/
def saltPart (v : ByteArray) : ByteArray := v.extract (v.size - HM96_SALT) v.size

/-- The digest an hm96 leaf of `d` masks for the value `v`: the preimage itself in a tile tree, else its row digest. -/
def xOf (d : RowsDomain) (v : ByteArray) : ByteArray :=
  if d.digests then openedValue v else Flock.HmRow.segDigest (8 * (openedValue v).size) (openedValue v)

theorem split_value (v : ByteArray) : openedValue v ++ saltPart v = v := by
  simp only [openedValue, saltPart, ByteArray.extract_append_extract]
  rw [Nat.min_eq_left (Nat.zero_le _), Nat.max_eq_right (Nat.sub_le _ _), ByteArray.extract_zero_size]

theorem openedValue_size (v : ByteArray) : (openedValue v).size = v.size - HM96_SALT := by
  simp only [openedValue, ByteArray.size_extract]
  omega

theorem xOf_size (d : RowsDomain) (v : ByteArray) (h : d.digests = true → v.size = 256) : (xOf d v).size = 64 := by
  unfold xOf
  split
  · rename_i hd; rw [openedValue_size, h hd]; rfl
  · exact Commit.sha512_size _

theorem domain_hash {h : PouwCommit.Hash} {binding : ByteArray} {owner : Int} {count : Nat} {f : Domain}
    (hd : Domain.mk? h binding owner count = .ok f) : f.hash = h := by
  unfold Domain.mk? at hd
  by_cases h32 : binding.size = 32 <;> by_cases h64 : binding.size = 64 <;> by_cases ho : owner < -2 <;>
    simp [h32, h64, ho, pure, Except.pure, bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at hd <;>
    first | (cases hd; rfl) | (cases hd)

theorem rowsDomain_hm96 {binding : ByteArray} {owner : Int} {count : Nat} {digests : Bool} {d : RowsDomain}
    (h : rowsDomain binding owner count HM96 digests = .ok d) :
    d.hm96 = true ∧ d.digests = digests ∧ d.frame.hash = .sha512 := by
  have e : rowsDomain binding owner count HM96 digests =
      (do let f ← Domain.mk? .sha512 binding owner count; pure ⟨f, true, digests⟩) := rfl
  rw [e] at h
  cases hf : Domain.mk? .sha512 binding owner count with
  | error err => rw [hf] at h; cases h
  | ok f =>
    rw [hf] at h
    cases h
    exact ⟨rfl, rfl, domain_hash hf⟩

theorem opens_ok {d : RowsDomain} {root : ByteArray} {schema : String} {o : LeafOpening}
    (h : opens (.ok d) root schema o = true) : d.verify root o.position schema o.value o.path = .ok true := by
  simp only [opens] at h
  split at h
  · rename_i b hb; subst h; exact hb
  · cases h

/-- **What `Domain.verify` accepts**: a path of `depth count` siblings of the frame's digest size, an opening within
`max_proof_bytes`, a value the leaf function takes, and the climb from its leaf to the root. -/
theorem verify_true {d : Domain} {root : ByteArray} {rank : Nat} {value : ByteArray} {path : List ByteArray}
    {leafOf : ByteArray → Except String ByteArray} (h : d.verify root rank value path leafOf = .ok true) :
    path.length = depth d.count ∧ (∀ s ∈ path, s.size = d.hash.bytes) ∧
      value.size + d.hash.bytes * path.length ≤ 64 <<< 20 ∧
      ∃ lf, leafOf value = .ok lf ∧ (vfold d path lf rank).1 = root := by
  unfold Domain.verify at h
  split at h
  · simp [pure, Except.pure] at h
  split at h
  · simp [pure, Except.pure] at h
  rename_i _ h2
  split at h
  · simp [pure, Except.pure] at h
  rename_i h3
  split at h
  · simp [pure, Except.pure] at h
  rename_i h4
  dsimp only at h
  split at h
  · simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h
  rename_i h5
  cases hl : leafOf value with
  | error e => simp [hl, bind, Except.bind] at h
  | ok lf =>
    rw [hl] at h
    change (pure ((vfold d path lf rank).1 == root) : Except String Bool) = .ok true at h
    have h := Except.ok.inj h
    refine ⟨by simpa using h3, fun s hs => ?_, by omega, lf, rfl, beq_iff_eq.1 h⟩
    exact Decidable.byContradiction fun hc => h4 (List.any_eq_true.2 ⟨s, hs, by simpa using hc⟩)

/-- **What an hm96 tree's leaf function takes**: a value at least its salt long (in a tile tree, a digest and its salt),
whose frame leaf is the one over the hm96 leaf of its digest and salt. -/
theorem leaf_ok {d : RowsDomain} {rank : Nat} {schema : String} {value lf : ByteArray} (hm : d.hm96 = true)
    (h : d.leaf rank schema value = .ok lf) : 192 ≤ value.size ∧ (d.digests = true → value.size = 256) ∧
      lf = d.frame.leaf rank Flock.HmRow.ROW_SCHEMA_SEG (Flock.Hm96.Default512.leaf (xOf d value) (saltPart value)) := by
  rcases d with ⟨frame, hm', dg⟩
  dsimp only at hm
  subst hm
  unfold RowsDomain.leaf at h
  split at h
  · rename_i hc; simp at hc
  dsimp only at h
  split at h
  · simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h
  rename_i hc
  have h := Except.ok.inj h
  simp only [Bool.or_eq_true, decide_eq_true_eq, Bool.and_eq_true, bne_iff_ne, ne_eq, not_or, not_and,
    Decidable.not_not] at hc
  obtain ⟨h1, h2⟩ := hc
  simp only [HM96_SALT] at h1 h2
  exact ⟨by omega, fun hd => by have := h2 hd; omega, h.symm⟩

/-! ## The hm96 leaf and the row digest -/

/-- **An hm96 leaf binds its digest and salt**: equal leaves have equal digests and salts, or a collision among the
salt-digest and leaf inputs. -/
theorem hm96_binds (L L' : List ByteArray) (x y x' y' : ByteArray) (hx : x.size = 64) (hx' : x'.size = 64)
    (h2 : Flock.Hm96.Default512.saltPrefix ++ y ∈ L) (h2' : Flock.Hm96.Default512.saltPrefix ++ y' ∈ L')
    (h3 : Flock.Hm96.Default512.leafPrefix ++ Flock.Hm96.Default512.commitString x y ∈ L)
    (h3' : Flock.Hm96.Default512.leafPrefix ++ Flock.Hm96.Default512.commitString x' y' ∈ L')
    (hv : Flock.Hm96.Default512.leaf x y = Flock.Hm96.Default512.leaf x' y') : (x = x' ∧ y = y') ∨ Commit.Break L L' := by
  rcases Commit.step h3 h3' hv with e | b
  case inr => exact .inr b
  have e := (ByteArray.append_right_inj _).1 e
  simp only [Flock.Hm96.Default512.commitString] at e
  have hb := ByteArray.append_inj_left e (by rw [Commit.commitString_size, Commit.commitString_size])
  rw [hb, ByteArray.append_right_inj] at e
  rcases Commit.step h2 h2' e with e | b
  case inr => exact .inr b
  have hs := (ByteArray.append_right_inj _).1 e
  rw [hs] at hb
  exact .inl ⟨Commit.masked_inj _ _ _ hx hx' hb, hs⟩

theorem be64_length (x : Nat) : (Flock.HmRow.be64 x).data.toList.length = 8 := by simp [Flock.HmRow.be64]

theorem be64_inj (x y : Nat) (hx : x < 2 ^ 64) (hy : y < 2 ^ 64)
    (h : (Flock.HmRow.be64 x).data.toList = (Flock.HmRow.be64 y).data.toList) : x = y := by
  simp [Flock.HmRow.be64, List.range_succ] at h
  obtain ⟨h0, h1, h2, h3, h4, h5, h6, h7⟩ := h
  simp only [← UInt8.toNat_inj] at h0 h1 h2 h3 h4 h5 h6 h7
  simp at h0 h1 h2 h3 h4 h5 h6 h7
  omega

/-- `sha512/row-seg/v1`'s first block fixes the row's length. -/
theorem rowPrefixSeg_inj {l l' : Nat} (hl : l < 2 ^ 64) (hl' : l' < 2 ^ 64)
    (h : (Flock.HmRow.rowPrefixSeg l).data.toList = (Flock.HmRow.rowPrefixSeg l').data.toList) : l = l' := by
  simp only [Flock.HmRow.rowPrefixSeg, Flock.Hm96.pad, ByteArray.data_append, Array.toList_append,
    List.append_assoc] at h
  have h := List.append_cancel_left h
  exact be64_inj l l' hl hl' (List.append_inj h (by rw [be64_length, be64_length])).1

theorem rowPrefixSeg_length (l : Nat) : (Flock.HmRow.rowPrefixSeg l).data.toList.length = 128 := by
  have ht : ("verity/sha512-row-seg/v1".toUTF8.push 0).size = 25 := by decide
  have h4 := Commit.be32_length 1024
  simp only [Array.length_toList, ByteArray.size_data] at h4 ⊢
  simp only [Flock.HmRow.rowPrefixSeg, Flock.Hm96.pad, ByteArray.size_append, ht, h4]
  simp [ByteArray.size, Flock.HmRow.be64]

/-- Segment `j`'s message of a row of `l` bits whose bytes are `row`: its first block, then its bytes. -/
def segIn (l : Nat) (row : ByteArray) (j : Nat) : ByteArray :=
  Flock.HmRow.segmentPrefix j (Flock.HmRow.segShape l).1.size (Flock.HmRow.segShape l).1[j]!.1 ++
    row.extract (1024 * j) (1024 * j + ((Flock.HmRow.segShape l).1[j]!.1 + 7) / 8)

/-- The outer message: the first block, then each segment's digest. -/
def segOut (l : Nat) (row : ByteArray) : ByteArray :=
  (List.range (Flock.HmRow.segShape l).1.size).foldl (fun acc j => acc ++ Flock.Sha512.hash (segIn l row j))
    (Flock.HmRow.rowPrefixSeg l)

theorem segHashed_eq (row : ByteArray) : segHashed row =
    (List.range (Flock.HmRow.segShape (8 * row.size)).1.size).map (segIn (8 * row.size) row) ++
      [segOut (8 * row.size) row] := by
  simp only [segHashed, segOut, List.foldl_map]
  rfl

theorem segDigest_out (l : Nat) (row : ByteArray) : Flock.HmRow.segDigest l row = Flock.Sha512.hash (segOut l row) :=
  rfl

theorem foldl_append_data (f : Nat → ByteArray) (a : ByteArray) (n : Nat) :
    ((List.range n).foldl (fun acc j => acc ++ f j) a).data.toList =
      a.data.toList ++ (List.range n).flatMap fun j => (f j).data.toList := by
  induction n generalizing a with
  | zero => simp
  | succ n ih =>
    rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil, ByteArray.data_append,
      Array.toList_append, ih, List.flatMap_append, List.flatMap_singleton, List.append_assoc]

theorem segOut_data (l : Nat) (row : ByteArray) : (segOut l row).data.toList =
    (Flock.HmRow.rowPrefixSeg l).data.toList ++
      (List.range (Flock.HmRow.segShape l).1.size).flatMap fun j => (Flock.Sha512.hash (segIn l row j)).data.toList :=
  foldl_append_data _ _ _

theorem length_flatMap_range {α : Type} {k : Nat} {f : Nat → List α} (hf : ∀ j, (f j).length = k) :
    ∀ n, ((List.range n).flatMap f).length = k * n
  | 0 => by simp
  | n + 1 => by
    rw [List.range_succ, List.flatMap_append, List.length_append, length_flatMap_range hf n, List.flatMap_singleton,
      hf, Nat.mul_succ]

theorem flatMap_range_inj {α : Type} {k : Nat} {f g : Nat → List α} (hf : ∀ j, (f j).length = k)
    (hg : ∀ j, (g j).length = k) : ∀ n, (List.range n).flatMap f = (List.range n).flatMap g → ∀ j < n, f j = g j
  | 0, _, j, hj => absurd hj (Nat.not_lt_zero _)
  | n + 1, h, j, hj => by
    rw [List.range_succ, List.flatMap_append, List.flatMap_append, List.flatMap_singleton,
      List.flatMap_singleton] at h
    obtain ⟨h1, h2⟩ := List.append_inj h (by rw [length_flatMap_range hf, length_flatMap_range hg])
    by_cases hjn : j < n
    · exact flatMap_range_inj hf hg n h1 j hjn
    · obtain rfl : j = n := by omega
      exact h2

theorem segShape_size (l : Nat) : (Flock.HmRow.segShape l).1.size = (l + 8191) / 8192 := by
  simp [Flock.HmRow.segShape, Flock.HmRow.ROW_SEG_BITS]

theorem segShape_bits {l j : Nat} (hj : j < (l + 8191) / 8192) :
    (Flock.HmRow.segShape l).1[j]!.1 = min (l - 8192 * j) 8192 := by
  have hj' : j < (Flock.HmRow.segShape l).1.size := by rw [segShape_size]; exact hj
  rw [getElem!_pos _ j hj']
  simp [Flock.HmRow.segShape, Flock.HmRow.ROW_SEG_BITS]

theorem segIn_data (l : Nat) (row : ByteArray) (j : Nat) : (segIn l row j).data.toList =
    (Flock.HmRow.segmentPrefix j (Flock.HmRow.segShape l).1.size (Flock.HmRow.segShape l).1[j]!.1).data.toList ++
      (row.data.toList.drop (1024 * j)).take (((Flock.HmRow.segShape l).1[j]!.1 + 7) / 8) := by
  simp only [segIn, ByteArray.data_append, Array.toList_append, ByteArray.data_extract, Array.toList_extract,
    List.extract_eq_take_drop, Nat.add_sub_cancel_left]

/-- Two rows of one length whose segments' messages agree are one row. -/
theorem row_eq_of_segIn {row row' : ByteArray} (hN : row.size = row'.size)
    (hall : ∀ j < (Flock.HmRow.segShape (8 * row.size)).1.size,
      segIn (8 * row.size) row j = segIn (8 * row.size) row' j) : row = row' := by
  apply ByteArray.ext
  apply Array.toList_inj.1
  apply List.ext_getElem?
  intro i
  by_cases hi : i < row.size
  · have hj : i / 1024 < (Flock.HmRow.segShape (8 * row.size)).1.size := by rw [segShape_size]; omega
    have e := congrArg (fun b : ByteArray => b.data.toList) (hall _ hj)
    simp only [segIn_data] at e
    have e := congrArg (fun L => L[i % 1024]?) (List.append_cancel_left e)
    have hm : i % 1024 < ((Flock.HmRow.segShape (8 * row.size)).1[i / 1024]!.1 + 7) / 8 := by
      rw [segShape_bits (by rw [← segShape_size]; exact hj)]
      omega
    simp only [List.getElem?_take_of_lt hm, List.getElem?_drop, Nat.div_add_mod] at e
    exact e
  · rw [List.getElem?_eq_none (by simp; omega), List.getElem?_eq_none (by simp; omega)]

/-- **`sha512/row-seg/v1` binds its row**: two rows with one digest are one row, or two different messages hashed in
computing the two digests have one SHA-512 digest. -/
theorem segDigest_binds (row row' : ByteArray) (hs : row.size < 2 ^ 32) (hs' : row'.size < 2 ^ 32)
    (h : Flock.HmRow.segDigest (8 * row.size) row = Flock.HmRow.segDigest (8 * row'.size) row') :
    row = row' ∨ Commit.Break (segHashed row) (segHashed row') := by
  rw [segHashed_eq, segHashed_eq]
  rw [segDigest_out, segDigest_out] at h
  rcases Commit.step (List.mem_append_right _ (List.mem_singleton_self _))
    (List.mem_append_right _ (List.mem_singleton_self _)) h with e | b
  case inr => exact .inr b
  have e := congrArg (fun b : ByteArray => b.data.toList) e
  simp only [segOut_data] at e
  obtain ⟨hp, hd⟩ := List.append_inj e (by rw [rowPrefixSeg_length, rowPrefixSeg_length])
  have hN : row.size = row'.size := by
    have := rowPrefixSeg_inj (by omega) (by omega) hp
    omega
  rw [← hN] at hd ⊢
  have h64 : ∀ m, (Flock.Sha512.hash m).data.toList.length = 64 := fun m => by
    rw [Array.length_toList]; exact Commit.sha512_size m
  have hblk := flatMap_range_inj (fun j => h64 _) (fun j => h64 _) _ hd
  by_cases hall : ∀ j < (Flock.HmRow.segShape (8 * row.size)).1.size,
      segIn (8 * row.size) row j = segIn (8 * row.size) row' j
  · exact .inl (row_eq_of_segIn hN hall)
  · obtain ⟨j, hj, hne⟩ : ∃ j, j < (Flock.HmRow.segShape (8 * row.size)).1.size ∧
        segIn (8 * row.size) row j ≠ segIn (8 * row.size) row' j :=
      Classical.byContradiction fun hc => hall fun j hj => Classical.byContradiction fun hne => hc ⟨j, hj, hne⟩
    exact .inr ⟨_, List.mem_append_left _ (List.mem_map.2 ⟨j, List.mem_range.2 hj, rfl⟩),
      _, List.mem_append_left _ (List.mem_map.2 ⟨j, List.mem_range.2 hj, rfl⟩), hne,
      ByteArray.ext (Array.toList_inj.1 (hblk j hj))⟩

/-! ## Binding -/

theorem rowsHashed_eq (d : RowsDomain) (o : LeafOpening) : rowsHashed d o =
    ((if d.digests then [] else segHashed (openedValue o.value)) ++
      [Flock.Hm96.Default512.saltPrefix ++ saltPart o.value,
       Flock.Hm96.Default512.leafPrefix ++ Flock.Hm96.Default512.commitString (xOf d o.value) (saltPart o.value),
       frameBytes "leaf" [d.frame.id, Flock.FV3.uint o.position, Flock.FV3.uint o.position,
         Flock.HmRow.ROW_SCHEMA_SEG.toUTF8, Flock.Hm96.Default512.leaf (xOf d o.value) (saltPart o.value)]]) ++
      (pfold d.frame o.path (d.frame.leaf o.position Flock.HmRow.ROW_SCHEMA_SEG
        (Flock.Hm96.Default512.leaf (xOf d o.value) (saltPart o.value))) o.position).1 := rfl

theorem openingBinds : OpeningBinds := by
  intro binding owner count digests d root schema o o' hd hpos ho ho'
  rcases o with ⟨pos, v, p⟩
  rcases o' with ⟨pos', v', p'⟩
  obtain rfl : pos = pos' := hpos
  obtain ⟨hm, -, hf⟩ := rowsDomain_hm96 hd
  have hb : d.frame.hash.bytes = 64 := by rw [hf]; rfl
  obtain ⟨hlen, hsib, hmax, lf, hlf, hroot⟩ := verify_true (opens_ok ho)
  obtain ⟨hlen', hsib', hmax', lf', hlf', hroot'⟩ := verify_true (opens_ok ho')
  obtain ⟨h192, h256, rfl⟩ := leaf_ok hm hlf
  obtain ⟨h192', h256', rfl⟩ := leaf_ok hm hlf'
  dsimp only at hlen hsib hmax hroot hlen' hsib' hmax' hroot' h192 h256 h192' h256'
  rw [hb] at hsib hsib' hmax hmax'
  have mem3 : ∀ (w : ByteArray) (q : List ByteArray) x, x ∈ [Flock.Hm96.Default512.saltPrefix ++ saltPart w,
      Flock.Hm96.Default512.leafPrefix ++ Flock.Hm96.Default512.commitString (xOf d w) (saltPart w),
      frameBytes "leaf" [d.frame.id, Flock.FV3.uint pos, Flock.FV3.uint pos, Flock.HmRow.ROW_SCHEMA_SEG.toUTF8,
        Flock.Hm96.Default512.leaf (xOf d w) (saltPart w)]] → x ∈ rowsHashed d ⟨pos, w, q⟩ := fun w q x hx => by
    rw [rowsHashed_eq]; exact List.mem_append_left _ (List.mem_append_right _ hx)
  have memP : ∀ (w : ByteArray) (q : List ByteArray) x, x ∈ (pfold d.frame q (d.frame.leaf pos
      Flock.HmRow.ROW_SCHEMA_SEG (Flock.Hm96.Default512.leaf (xOf d w) (saltPart w))) pos).1 →
      x ∈ rowsHashed d ⟨pos, w, q⟩ := fun w q x hx => by
    rw [rowsHashed_eq]; exact List.mem_append_right _ hx
  have memS : d.digests = false → ∀ (w : ByteArray) (q : List ByteArray) x, x ∈ segHashed (openedValue w) →
      x ∈ rowsHashed d ⟨pos, w, q⟩ := fun hdg w q x hx => by
    rw [rowsHashed_eq, hdg]; exact List.mem_append_left _ (List.mem_append_left _ hx)
  have hlsz : ∀ w, (d.frame.leaf pos Flock.HmRow.ROW_SCHEMA_SEG w).size = 64 := fun w => by
    rw [leaf_run, Commit.run_size, hb]
  rcases climb d.frame hf pos p.length p p'
    (d.frame.leaf pos Flock.HmRow.ROW_SCHEMA_SEG (Flock.Hm96.Default512.leaf (xOf d v) (saltPart v)))
    (d.frame.leaf pos Flock.HmRow.ROW_SCHEMA_SEG (Flock.Hm96.Default512.leaf (xOf d v') (saltPart v')))
    rfl (hlen'.trans hlen.symm) hsib hsib' (hlsz _) (hlsz _) (hroot.trans hroot'.symm) with e | b
  case inr => exact .inr (break_mono (memP v p) (memP v' p') b)
  rw [leaf_run, leaf_run, run_sha512 hf, run_sha512 hf] at e
  rcases Commit.step (mem3 v p _ (.tail _ (.tail _ (.head _)))) (mem3 v' p' _ (.tail _ (.tail _ (.head _)))) e
    with e | b
  case inr => exact .inr b
  have e := Commit.frameBytes_sizes _ _ _ (by simp only [List.map_cons, List.map_nil, Commit.leaf_size]) e
  simp only [List.cons.injEq, true_and, and_true] at e
  rcases hm96_binds (rowsHashed d ⟨pos, v, p⟩) (rowsHashed d ⟨pos, v', p'⟩) (xOf d v) (saltPart v) (xOf d v')
    (saltPart v') (xOf_size d v h256) (xOf_size d v' h256') (mem3 v p _ (.head _)) (mem3 v' p' _ (.head _))
    (mem3 v p _ (.tail _ (.head _))) (mem3 v' p' _ (.tail _ (.head _))) e with ⟨ex, ey⟩ | b
  case inr => exact .inr b
  have hv : openedValue v = openedValue v' → v = v' := fun e => by
    rw [← split_value v, ← split_value v', e, ey]
  cases hdg : d.digests
  · simp only [xOf, hdg, Bool.false_eq_true, ite_false] at ex
    have hmx : (64 <<< 20 : Nat) = 67108864 := rfl
    rcases segDigest_binds _ _ (by rw [openedValue_size]; omega) (by rw [openedValue_size]; omega) ex with e | b
    · exact .inl (hv e)
    · exact .inr (break_mono (memS hdg v p) (memS hdg v' p') b)
  · simp only [xOf, hdg, ite_true] at ex
    exact .inl (hv ex)

end Properties.Pouw.PearlPlus
