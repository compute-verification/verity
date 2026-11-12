module

public import Properties.Spot.HistoryTable
public import Proofs.Flock.Recursive.TopRoot
public import Proofs.Accounting.ReadUnit.Leaf
public import Proofs.Core.CanonUtf8
public import Proofs.Flock.Bytes
import all Init.Data.String.Defs
import all Lean.Data.Json.Basic
import all Lean.Data.Json.FromToJson.Basic
import all Init.Data.Ord.String
import all Init.Data.Array.Basic
import all Init.Data.Array.DecidableEq
import all Init.Data.List.Control
import all Init.Data.Range.Basic
import all Init.Data.Int.Repr
import all Init.Data.Repr
import all Init.Data.Nat.Power2.Basic

/-!
# An accepted history table's registered root is the model's (`Properties.Spot.HistoryRootIsModel`)

* bytes: `be64`, `be32`, `FV3.uint` and a frame's input are the hash area's `be 8`, `be 4`, `uint` and `Tree.frame`
  (`be64_data`, `be32_data`, `uint_data`, `frame_data`), and the code's SHA-512 is `Verity.sha512` (`hash_data`);
* a capacity `checkAt` accepts is a power of two (`capOk_pow`), so `FV3.root` pads nothing and its level `l` is the
  model's `Tree.nodeAt` at `l` (`lev_bytes`), its root `Hash.HistoryTable.root` (`rootAt_bytes`);
* `historyRootIsModel_honest`: `checkAt` accepts one written row at `CAPACITY` against its own root
  (`checkAt_complete`), the kernel evaluating only the domain's canonical JSON (`Verity.CanonUtf8.canonUtf8_flat`);
* `historyRootIsModel_nontrivial`: an empty registered root is no domain's model root.
-/

@[expose] public section

namespace Properties.Spot.SecurityProofs

namespace HistoryRoot

open Flock Flock.SecurityProofs
open _root_.Spot.HistoryTable

local instance lawfulBEqByteArray_history : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := show (a.data == a.data) = true from beq_iff_eq.2 rfl

/-! ## Bytes -/

theorem be64_data (n : ℕ) : (Flock.be64 n.toUInt64).data.toList = Hash.be 8 n := by
  have h := Properties.NetworkCertifier.Bytes.be64_bs n.toUInt64
  simp only [Properties.NetworkCertifier.Bytes.bs] at h
  rw [h]
  simp only [Hash.be, List.range_succ, List.range_zero, List.nil_append, List.map_cons, List.map_nil,
    List.cons_append, List.cons.injEq, and_true]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> apply UInt8.toNat_inj.1 <;>
    simp [UInt64.toNat_shiftRight, Nat.shiftRight_eq_div_pow] <;> omega

theorem be32_bytes (x : UInt32) :
    Flock.be32 x = ⟨#[(x >>> 24).toUInt8, (x >>> 16).toUInt8, (x >>> 8).toUInt8, x.toUInt8]⟩ := by
  simp only [Flock.be32, Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl,
    Std.Legacy.Range.size]
  rfl

theorem be32_data (n : ℕ) : (Flock.be32 n.toUInt32).data.toList = Hash.be 4 n := by
  rw [be32_bytes]
  simp only [Hash.be, List.range_succ, List.range_zero, List.nil_append, List.map_cons, List.map_nil,
    List.cons_append, List.cons.injEq, and_true]
  refine ⟨?_, ?_, ?_, ?_⟩ <;> apply UInt8.toNat_inj.1 <;>
    simp [UInt32.toNat_shiftRight, Nat.shiftRight_eq_div_pow] <;> omega

theorem size_eq_log2 {x : ℕ} (hx : x ≠ 0) : Nat.size x = Nat.log2 x + 1 :=
  le_antisymm (Nat.size_le.2 Nat.lt_log2_self) (Nat.lt_size.2 (Nat.log2_self_le hx))

theorem uint_data (x : ℕ) : (FV3.uint x).data.toList = Hash.uint x := by
  have hn : (Nat.log2 x + 8) / 8 = max 1 ((Nat.size x + 7) / 8) := by
    rcases Nat.eq_zero_or_pos x with rfl | hx
    · simp [Nat.size_zero, Nat.log2_zero]
    · rw [size_eq_log2 (Nat.pos_iff_ne_zero.1 hx)]
      omega
  unfold FV3.uint Hash.uint Hash.be
  simp only [Array.toList_map, Array.toList_range, hn]
  refine List.map_congr_left fun i _ => ?_
  apply UInt8.toNat_inj.1
  simp [Nat.shiftRight_eq_div_pow, pow_mul]

theorem hash_data (b : ByteArray) : (Sha512.hash b).data.toList = (Verity.sha512 b.data.toList).bytes := by
  have hb : (⟨b.data.toList.toArray⟩ : ByteArray) = b := ByteArray.ext (by simp)
  rw [Accounting.ReadUnit.Gates.isSha512_sha512 b.data.toList, hb]

theorem frame0_data : FV3.FRAME.data.toList = Hash.Tree.FRAME :=
  Accounting.ReadUnit.Gates.push_zero_data _

theorem foldl_step_data (ps : List ByteArray) (acc : ByteArray) :
    (ps.foldl Registered.step acc).data.toList =
      acc.data.toList ++ ps.flatMap fun p => Hash.be 8 p.size ++ p.data.toList := by
  induction ps generalizing acc with
  | nil => simp
  | cons p ps ih =>
    rw [List.foldl_cons, ih, Registered.step, ByteArray.toList_data_append, ByteArray.toList_data_append,
      be64_data, List.flatMap_cons]
    simp only [List.append_assoc]

theorem frame_data (tag : String) (ps : List ByteArray) :
    (Registered.frame tag ps).data.toList = Hash.Tree.frame tag (ps.map fun p => p.data.toList) := by
  rw [Registered.frame, foldl_step_data, ByteArray.toList_data_append, ByteArray.toList_data_append, be32_data,
    frame0_data]
  simp only [Hash.Tree.frame, Hash.utf8, List.flatMap_map, Array.length_toList, ByteArray.size_data]

/-! ## The domain -/

theorem id_bytes {cap : ℕ} {dom : ByteArray} (h : dom.size = 64) :
    (historyDomain cap dom).id.bytes = dom.data.toList := by
  refine List.ext_getElem (by simp [Verity.Digest.bytes, h]) fun i h1 h2 => ?_
  have hi : i < dom.data.size := by simpa using h2
  simp only [Verity.Digest.bytes, List.getElem_ofFn, historyDomain, Array.getD_eq_getD_getElem?,
    Array.getElem?_eq_getElem hi, Option.getD_some, Array.getElem_toList]

theorem position_eq (cap : ℕ) (dom : ByteArray) (r : ℕ) : (historyDomain cap dom).position r = r := rfl

theorem bind_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : (x >>= f) = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error e => cases h
  | ok a => exact ⟨a, rfl, h⟩

theorem make_id_size (b : ByteArray) (owner : Int) (start count : ℕ) :
    (FV3.Sha512.Domain.make b owner start count).id.size = 64 :=
  Sha512.hash_size _

theorem domainAt_size {cap : ℕ} {program table : String} {words : ℕ} {dom : ByteArray}
    (h : domainAt cap program table words = .ok dom) : dom.size = 64 := by
  unfold domainAt at h
  obtain ⟨b, -, hb⟩ := bind_ok h
  rw [← Except.ok.inj hb]
  exact make_id_size _ _ _ _

/-! ## Leaves and levels -/

/-- The model's leaves over the verifier's table: its domain, the default key, `bcs`' strings and `ph`'s past them. -/
def leafF (cap : ℕ) (dom : ByteArray) (bcs : Array ByteArray) (ph : ByteArray) : ℕ → Hash.Digest :=
  Hash.HistoryTable.leaf Verity.sha512 (historyDomain cap dom) Flock.Hm96.Default512.key.data.toList
    (Hash.HistoryTable.string bcs.size (fun r => bcs[r]!.data.toList) ph.data.toList)

theorem treeLeaf_data (bc : ByteArray) :
    (Flock.Hm96.Default512.treeLeaf bc).data.toList =
      (Verity.sha512 (Hash.Hm96.leafPrefix Verity.sha512 Flock.Hm96.Default512.key.data.toList ++
        bc.data.toList)).bytes := by
  rw [Flock.Hm96.Default512.treeLeaf, hash_data, ByteArray.toList_data_append,
    Accounting.ReadUnit.Gates.leafPrefix_data]
  rfl

theorem leafAt_data {cap : ℕ} {dom : ByteArray} (hd : dom.size = 64) (bcs : Array ByteArray) (ph : ByteArray)
    (r : ℕ) :
    (leafAt dom bcs (Flock.Hm96.Default512.treeLeaf ph) r).data.toList = (leafF cap dom bcs ph r).bytes := by
  have hp : (if h : r < bcs.size then Flock.Hm96.Default512.treeLeaf bcs[r]
      else Flock.Hm96.Default512.treeLeaf ph).data.toList =
      (Verity.sha512 (Hash.Hm96.leafPrefix Verity.sha512 Flock.Hm96.Default512.key.data.toList ++
        Hash.HistoryTable.string bcs.size (fun r => bcs[r]!.data.toList) ph.data.toList r)).bytes := by
    unfold Hash.HistoryTable.string
    split
    · rename_i h
      rw [treeLeaf_data]
      simp only [getElem!_pos bcs r h]
    · rw [treeLeaf_data]
  rw [leafAt, Registered.leafAs_eq, hash_data, Registered.leafInAs, frame_data, leafF, Hash.HistoryTable.leaf,
    Hash.Tree.leafMsg, id_bytes hd, position_eq]
  simp only [List.map_cons, List.map_nil, uint_data, hp]
  rfl

theorem leaves_size (cap : ℕ) (dom : ByteArray) (bcs : Array ByteArray) (ph : ByteArray) :
    (leavesAt cap dom bcs ph).size = cap := by
  simp [leavesAt]

theorem log2ceil_leaves {cap k : ℕ} (hk : cap = 2 ^ k) (dom : ByteArray) (bcs : Array ByteArray) (ph : ByteArray) :
    log2ceil (leavesAt cap dom bcs ph).size = k := by
  rw [leaves_size, hk, TopRoot.log2ceil_two_pow]

theorem lev_bytes {cap k : ℕ} {dom : ByteArray} {bcs : Array ByteArray} {ph : ByteArray} (hk : cap = 2 ^ k)
    (hd : dom.size = 64) : ∀ l ≤ k, ∀ i < 2 ^ (k - l),
      ((TopRoot.lev dom (leavesAt cap dom bcs ph) l)[i]!).data.toList =
        (Hash.Tree.nodeAt Verity.sha512 (historyDomain cap dom) (leafF cap dom bcs ph) l i).bytes
  | 0, _, i, hi => by
    have hpad : nextPow2 (leavesAt cap dom bcs ph).size - (leavesAt cap dom bcs ph).size = 0 := by
      rw [leaves_size, nextPow2, hk, TopRoot.log2ceil_two_pow, Nat.sub_self]
    have hi' : i < (leavesAt cap dom bcs ph).size := by
      rw [leaves_size, hk]
      simpa using hi
    rw [TopRoot.lev, TopRoot.base, hpad]
    simp only [show Array.range 0 = #[] from rfl, Array.map_empty, Array.append_empty]
    rw [getElem!_pos _ i hi']
    simp only [leavesAt, Array.getElem_map, Array.getElem_range]
    exact leafAt_data hd bcs ph i
  | l + 1, hl, i, hi => by
    have hsz : (TopRoot.lev dom (leavesAt cap dom bcs ph) l).size = 2 ^ (k - l) := by
      have h := TopRoot.lev_size dom (leavesAt cap dom bcs ph) l (by rw [log2ceil_leaves hk]; omega)
      rwa [log2ceil_leaves hk] at h
    have hpow : 2 ^ (k - l) = 2 ^ (k - (l + 1)) * 2 := by
      rw [show k - l = (k - (l + 1)) + 1 by omega, Nat.pow_succ]
    have hlt : i < (TopRoot.lev dom (leavesAt cap dom bcs ph) l).size / 2 := by
      rw [hsz, hpow, Nat.mul_div_cancel _ (by decide)]
      exact hi
    have h2i : 2 * i < 2 ^ (k - l) := by rw [hpow]; omega
    have h2i1 : 2 * i + 1 < 2 ^ (k - l) := by rw [hpow]; omega
    rw [TopRoot.lev, TopRoot.next_get _ _ _ hlt, hash_data, Registered.nodeIn, frame_data, Hash.Tree.nodeAt,
      Hash.Tree.nodeMsg, id_bytes hd]
    simp only [List.map_cons, List.map_nil, uint_data, lev_bytes hk hd l (by omega) (2 * i) h2i,
      lev_bytes hk hd l (by omega) (2 * i + 1) h2i1]

/-! ## The root -/

theorem capOk_pow {cap : ℕ} (h : capOk cap = true) : ∃ k, cap = 2 ^ k := by
  simp only [capOk, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at h
  exact (Nat.and_sub_one_eq_zero_iff_isPowerOfTwo (by omega)).1 h.2

theorem rootAt_bytes {cap : ℕ} (hc : capOk cap = true) {dom : ByteArray} (hd : dom.size = 64)
    (bcs : Array ByteArray) (ph : ByteArray) :
    (rootAt cap dom bcs ph).data.toList =
      (Hash.HistoryTable.root Verity.sha512 (historyDomain cap dom) Flock.Hm96.Default512.key.data.toList
        (Hash.HistoryTable.string bcs.size (fun r => bcs[r]!.data.toList) ph.data.toList)).bytes := by
  obtain ⟨k, hk⟩ := capOk_pow hc
  have hlog := log2ceil_leaves hk dom bcs ph
  have hdep : Hash.Tree.depth (historyDomain cap dom).count = k := by
    show log2ceil cap = k
    rw [hk, TopRoot.log2ceil_two_pow]
  rw [rootAt, TopRoot.root_eq, hlog, lev_bytes hk hd k le_rfl 0 (by simp), Hash.HistoryTable.root,
    Hash.Tree.rootOf, hdep]
  rfl

theorem rootAt_size {cap : ℕ} (hc : capOk cap = true) {dom : ByteArray} (hd : dom.size = 64)
    (bcs : Array ByteArray) (ph : ByteArray) : (rootAt cap dom bcs ph).size = 64 := by
  have h := congrArg List.length (rootAt_bytes hc hd bcs ph)
  simpa [Verity.Digest.bytes] using h

/-! ## `checkAt` -/

theorem checkAt_ok {cap : ℕ} {program table : String} {words : ℕ} {bcs : Array ByteArray} {registered : ByteArray}
    (h : checkAt cap program table words bcs registered = .ok ()) :
    capOk cap = true ∧ ∃ dom, domainAt cap program table words = .ok dom ∧
      rootAt cap dom bcs (placeholder words) = registered := by
  unfold checkAt at h
  simp only at h
  split at h
  · exact absurd (Registered.except_bind_ok h) fun ⟨_, h1, _⟩ => Registered.throw_ne_ok _ _ h1
  rename_i hc
  refine ⟨by simpa using hc, ?_⟩
  split at h
  · exact absurd (Registered.except_bind_ok h) fun ⟨_, h1, _⟩ => Registered.throw_ne_ok _ _ h1
  split at h
  · exact absurd (Registered.except_bind_ok h) fun ⟨_, h1, _⟩ => Registered.throw_ne_ok _ _ h1
  split at h
  · exact absurd (Registered.except_bind_ok h) fun ⟨_, h1, _⟩ => Registered.throw_ne_ok _ _ h1
  split at h
  · exact absurd (Registered.except_bind_ok h) fun ⟨_, h1, _⟩ => Registered.throw_ne_ok _ _ h1
  obtain ⟨dom, hd, h⟩ := Registered.except_bind_ok h
  refine ⟨dom, hd, ?_⟩
  split at h
  · exact absurd h (Registered.throw_ne_ok _ _)
  rename_i hr
  simpa using hr

theorem checkAt_complete {cap : ℕ} {program table : String} {words : ℕ} {bcs : Array ByteArray} {dom : ByteArray}
    (hc : capOk cap = true) (hw : wordsOk words = true) (hn : bcs.size ≤ cap) (hb : ∀ b ∈ bcs, b.size = BC_BYTES)
    (hd : domainAt cap program table words = .ok dom) :
    checkAt cap program table words bcs (rootAt cap dom bcs (placeholder words)) = .ok () := by
  have hany : bcs.any (fun b => b.size != BC_BYTES) = false := by
    rw [Bool.eq_false_iff]
    intro h
    obtain ⟨i, hi, h⟩ := Array.any_eq_true.1 h
    simp [hb _ (Array.getElem_mem hi)] at h
  have hr : (rootAt cap dom bcs (placeholder words)).size = DIGEST_BYTES :=
    rootAt_size hc (domainAt_size hd) bcs (placeholder words)
  unfold checkAt
  simp [hc, hw, Nat.not_lt.2 hn, hany, hr, hd, bind, Except.bind, pure, Except.pure]

end HistoryRoot

open HistoryRoot

/-- **An accepted history table's registered root is the model's root.** -/
theorem historyRootIsModel : HistoryRootIsModel := by
  intro cap program table words bcs registered h
  obtain ⟨hc, dom, hd, hroot⟩ := checkAt_ok h
  refine ⟨dom, hd, ?_⟩
  rw [← hroot]
  exact rootAt_bytes hc (domainAt_size hd) bcs (_root_.Spot.HistoryTable.placeholder words)

private theorem domainAt_honest : ∃ dom, _root_.Spot.HistoryTable.domainAt _root_.Spot.HistoryTable.CAPACITY "p" "t" 1 =
    .ok dom := by
  have hc : (Flock.canonUtf8 (Lean.Json.mkObj [("capacity", Lean.toJson _root_.Spot.HistoryTable.CAPACITY),
      ("program", Lean.Json.str "p"), ("schema", Lean.Json.str _root_.Spot.HistoryTable.SCHEMA),
      ("table", Lean.Json.str "t"), ("words", Lean.toJson (1 : ℕ))])).isOk = true := by
    rw [Verity.CanonUtf8.canonUtf8_flat _ (by decide +kernel)]
    decide +kernel
  revert hc
  cases hj : Flock.canonUtf8 (Lean.Json.mkObj [("capacity", Lean.toJson _root_.Spot.HistoryTable.CAPACITY),
      ("program", Lean.Json.str "p"), ("schema", Lean.Json.str _root_.Spot.HistoryTable.SCHEMA),
      ("table", Lean.Json.str "t"), ("words", Lean.toJson (1 : ℕ))]) with
  | error e => intro hc; cases hc
  | ok s =>
    intro _
    simp only [_root_.Spot.HistoryTable.domainAt, _root_.Spot.HistoryTable.binding, hj]
    exact ⟨_, rfl⟩

/-- **The deployed check accepts a written table against its own root**: one row of 128 zero bytes at `CAPACITY`. -/
theorem historyRootIsModel_honest :
    _root_.Spot.HistoryTable.checkAt _root_.Spot.HistoryTable.CAPACITY "p" "t" 1 #[⟨Array.replicate 128 0⟩]
      (_root_.Spot.HistoryTable.rootAt _root_.Spot.HistoryTable.CAPACITY
        ((_root_.Spot.HistoryTable.domainAt _root_.Spot.HistoryTable.CAPACITY "p" "t" 1).toOption.getD .empty)
        #[⟨Array.replicate 128 0⟩] (_root_.Spot.HistoryTable.placeholder 1)) = .ok () := by
  obtain ⟨dom, hd⟩ := domainAt_honest
  rw [hd]
  exact checkAt_complete (by decide +kernel) (by decide) (by simp [_root_.Spot.HistoryTable.CAPACITY])
    (by intro b hb; simp at hb; subst hb; rfl) hd

/-- **An empty registered root is no model root**: the conclusion fails at a check that accepts everything. -/
theorem historyRootIsModel_nontrivial : ∀ dom : ByteArray,
    ¬ (_root_.Spot.HistoryTable.domainAt 2 "p" "t" 1 = .ok dom ∧
      ByteArray.empty.data.toList =
        (Hash.HistoryTable.root Verity.sha512 (historyDomain 2 dom) Flock.Hm96.Default512.key.data.toList
          (Hash.HistoryTable.string (#[] : Array ByteArray).size (fun r => (#[] : Array ByteArray)[r]!.data.toList)
            (_root_.Spot.HistoryTable.placeholder 1).data.toList)).bytes) := by
  rintro dom ⟨-, h⟩
  rw [show ByteArray.empty.data.toList = [] from rfl] at h
  have := congrArg List.length h
  simp [Verity.Digest.bytes] at this

end Properties.Spot.SecurityProofs
