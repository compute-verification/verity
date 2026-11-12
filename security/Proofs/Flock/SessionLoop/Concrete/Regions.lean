module

public import Proofs.Flock.SessionLoop.Concrete.Trace
public import Proofs.Flock.Soundness.Discharge.Hm.Bc
public import Proofs.Flock.Soundness.Discharge.Hm.Mask
public import Proofs.Flock.Soundness.Discharge.Layout.OutRegion
import Flock.HmRow
import Flock.HmNets
import Flock.Canon
import Flock.JsonParse
import Flock.Net
import Flock.Firewall
import Flock.Norm
import Flock.Public
import Flock.Statement
import Flock.Comp
import Flock.Bytes
import Flock.FV3
import Flock.Hm96
import Flock.Hash
import Flock.RowLeaf
import Flock.Circuit
import Flock.Tags
import Flock.Zk
import Flock.CoinTree
import Flock.Merkle
import all Lean.Data.Json.Basic
import all Lean.Data.Json.FromToJson.Basic
import all Std.Data.TreeMap.Raw.Basic
import all Std.Data.DTreeMap.Raw.Basic
import all Std.Data.DTreeMap.Internal.Operations
import all Std.Data.DTreeMap.Internal.Queries
import all Std.Data.DTreeMap.Internal.Balancing
import all Std.Data.DTreeMap.Internal.Def
import all Init.Data.Ord.String
import all Init.Data.Ord.Basic
import all Init.Data.String.Defs
import all Init.Data.String.Basic
import all Init.Data.String.Extra
import all Init.Data.Array.QSort.Basic
import all Init.Data.Array.Basic
import all Init.Data.Array.DecidableEq
import all Init.Data.List.Control
import all Init.Data.Range.Basic
import all Init.Data.Int.Repr
import all Init.Data.Repr
import all Init.Data.Nat.Fold
import all Init.Data.String.TakeDrop
import all Init.Data.String.Slice
import all Init.Data.String.Pattern.Basic
import all Init.Data.String.Pattern.String
import all Init.Data.String.Pattern.Char
import all Init.Data.String.Search
import all Init.Data.String.Iterate
import all Init.Data.String.Decode
import all Init.Data.String.Termination
import all Init.Data.Array.Attach
import all Init.WF

/-! # The concrete table's public regions hold at the trace

`HmRow.regions` gives the concrete table two regions (`regions_mem`): "digest 0", the `hm96` slot's output `b ‖ c`, and
"out", the unit slot's returned output. The public file's one row digest and the circuit's pinned dummy are both `BC`
(`bcChunk_BC`), and its one output and the pinned dummy output are 0 (`outVal_zero`).

At "digest 0" the trace holds `BC`: the row's `b ‖ c` in a witness that satisfies the rows (`Hm.row_bc_setupH`, read at
no regions) is the commit string of the row prefix and message its `sha512x3` slot reads, under the salt its `hm96` slot
reads; in the trace both are 0 (`zIn_sha`, `zIn_hm`), the mask of a zero salt is 0 (`mask_zeros`), and the commit string
of the zero row under the zero salt is `BC` (`commitString_zero`). At "out" the trace is 0 off the unit's constant
(`zIn_unit`), and so are the packed outputs (`outBytes_false`). So the trace satisfies the table at its regions
(`satisfies_regions_stW`).
-/

@[expose] public section

open Flock FlockLevel3 FlockSoundness FlockSoundness.Discharge.Hm
open Flock.HmNets hiding and

set_option maxRecDepth 200000

namespace FlockDeploy.Concrete.Trace

/-! ## Bytes and bits -/

theorem log2_one : Nat.log2 1 = 0 := Nat.log2_two_pow (n := 0)

theorem lt22 {y : ℕ} (h : y < 4194304) : y < 2 ^ 22 := lt_of_lt_of_le h (by norm_num)

theorem bit_ite (b : Bool) : FlockSoundness.bit (if b then (1 : ZMod 2) else 0) = b := by cases b <;> rfl

theorem forall₂_mem_right {α β : Type} {R : α → β → Prop} : ∀ {L : List α} {es : List β}, List.Forall₂ R L es →
    ∀ e ∈ es, ∃ a ∈ L, R a e
  | _, _, .nil, _, h => absurd h List.not_mem_nil
  | _, _, .cons hab hrest, e, h => by
      rcases List.mem_cons.1 h with rfl | h
      · exact ⟨_, List.mem_cons_self, hab⟩
      · obtain ⟨a, ha, hr⟩ := forall₂_mem_right hrest e h
        exact ⟨a, List.mem_cons_of_mem _ ha, hr⟩

/-- A byte string whose bits are all 0 is zeros. -/
theorem eq_zeros_of_bitL {l : List UInt8} {n : ℕ} (hl : l.length = n)
    (h : ∀ t < 8 * n, Discharge.Hm.bitL l t = false) : l = List.replicate n 0 := by
  refine List.ext_getElem (by rw [List.length_replicate, hl]) fun i h1 h2 => ?_
  rw [List.getElem_replicate]
  apply UInt8.toNat_inj.1
  refine Nat.eq_of_testBit_eq fun k => ?_
  rw [UInt8.toNat_zero, Nat.zero_testBit]
  by_cases hk : k < 8
  · have e := h (8 * i + k) (by omega)
    rwa [Discharge.Hm.bitL, show (8 * i + k) / 8 = i by omega, show (8 * i + k) % 8 = k by omega,
      List.getD_eq_getElem _ _ h1] at e
  · exact Nat.testBit_lt_two_pow (lt_of_lt_of_le (UInt8.toNat_lt _) (Nat.pow_le_pow_right (by norm_num) (by omega)))

theorem bytesOf_false {w : ℕ → Bool} {N : ℕ} {a : ℕ → ℕ → ℕ} (h : ∀ j < N, ∀ t < 8, w (a j t) = false) :
    Discharge.Hm.bytesOf w N a = List.replicate N 0 :=
  eq_zeros_of_bitL (Discharge.Hm.length_bytesOf w N a) fun q hq => by
    rw [Discharge.Hm.bitL_bytesOf w a hq]
    exact h _ (by omega) _ (Nat.mod_lt _ (by norm_num))

theorem bitL_zeros (n t : ℕ) : Discharge.Hm.bitL (List.replicate n (0 : UInt8)) t = false := by
  rw [Discharge.Hm.bitL, List.getD_eq_getElem?_getD, List.getElem?_replicate]
  split <;> simp

theorem foldl_xor_and_false (f : ℕ → Bool) : ∀ (l : List ℕ) (b : Bool),
    l.foldl (fun acc j => acc ^^ (f j && false)) b = b
  | [], _ => rfl
  | j :: l, b => by
    rw [List.foldl_cons, Bool.and_false, Bool.xor_false]
    exact foldl_xor_and_false f l b

/-- The mask of a zero salt is 0, bit by bit. -/
theorem maskV_false (key : ByteArray) (i : ℕ) : Discharge.Hm.maskV key (fun _ => false) i = false :=
  foldl_xor_and_false (fun j => Flock.Hm96.bit key (i + j)) _ false

/-- **The mask of the zero salt is zero.** -/
theorem mask_zeros : Refine.K512.mask (List.replicate 192 0) = List.replicate 64 0 := by
  refine eq_zeros_of_bitL (Refine.K512.mask_len _) fun t ht => ?_
  rw [Refine.K512_mask, Discharge.Hm.bitL_mask _ (show t < 512 by omega),
    show Discharge.Hm.bitL (List.replicate 192 (0 : UInt8)) = fun _ => false from funext (bitL_zeros 192)]
  exact maskV_false _ _

theorem xorBytes_zeros : ∀ l : List UInt8, Merkle.xorBytes l (List.replicate l.length 0) = l
  | [] => rfl
  | x :: l => by
    have ih := xorBytes_zeros l
    unfold Merkle.xorBytes at ih ⊢
    simp only [List.length_cons, List.replicate_succ, List.zipWith_cons_cons, UInt8.xor_zero, ih]

/-- **The commit string of the zero row under the zero salt is `BC`**, the row digest of the public file and the
circuit's pinned dummy. -/
theorem commitString_zero :
    Binding.commitString ((HmRow.rowPrefix 64).data.toList ++ List.replicate 128 0) (List.replicate 192 0) =
      BC.data.toList := by
  unfold Binding.commitString
  rw [mask_zeros]
  have h64 : (Binding.digBytes (Refine.H512 ((HmRow.rowPrefix 64).data.toList ++ List.replicate 128 0))).length = 64 :=
    Refine.hL512 _
  have hx := xorBytes_zeros (Binding.digBytes (Refine.H512 ((HmRow.rowPrefix 64).data.toList ++ List.replicate 128 0)))
  rw [h64] at hx
  rw [hx]
  decide +kernel

theorem BC_size : BC.size = 128 := rfl

theorem getD_zero (x : ℕ) : (#[0] : Array ℕ).getD x 0 = 0 := by
  rw [show (#[0] : Array ℕ) = Array.replicate 1 0 from rfl, Array.getD_eq_getD_getElem?, Array.getElem?_replicate]
  split <;> rfl

/-! ## The concrete table's regions -/

set_option maxHeartbeats 2000000 in
/-- **`HmRow.regions` makes only "digest p" and "out" regions** in an untyped circuit whose unit returns the outputs. -/
theorem regions_mem {c : Circuit} {pub : Public} {regs : Array Flock.Region} (h : HmRow.regions c pub = .ok regs)
    (ho : c.outNet = c.unitNet) (ht : c.typed = none) : ∀ r ∈ regs.toList,
      (∃ p < c.ports.size, ∃ name, mkRegion c.kLog name (Discharge.Layout.hmFree c) (Discharge.Layout.hmFixed c p)
        (Discharge.Layout.hmBytes c pub p) = .ok r) ∨ c.outRegion pub = .ok r := by
  unfold HmRow.regions at h
  dsimp only at h
  try simp only [Refine.throw_bind', pure_bind] at h
  obtain ⟨out, hloop, h⟩ := Refine.ebind_ok h
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hloop
  obtain ⟨es, hes, heq⟩ := Refine.list_forIn_push _ (fun _ v => v) _ _ _ hloop
  have hI : ∀ x ∈ out.toList, ∃ p < c.ports.size, ∃ name, mkRegion c.kLog name (Discharge.Layout.hmFree c)
      (Discharge.Layout.hmFixed c p) (Discharge.Layout.hmBytes c pub p) = .ok x := by
    intro x hx
    rw [heq] at hx
    have hx' : x ∈ es := by simpa using hx
    obtain ⟨a, ha, v, hv, rfl⟩ := forall₂_mem_right hes x hx'
    simp only [Std.Legacy.Range.size, List.mem_range'_1] at ha
    exact ⟨a, by omega, _, hv⟩
  split at h
  · rename_i hc
    simp [ho, ht] at hc
  · obtain ⟨r, hr, h⟩ := Refine.ebind_ok h
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    intro x hx
    rw [Array.toList_push, List.mem_append, List.mem_singleton] at hx
    rcases hx with hx | rfl
    · exact Or.inl (hI x hx)
    · exact Or.inr hr

/-- What the regions' proof reads of the concrete circuit and public file. -/
structure RegionFacts (c : Circuit) (pub : Public) : Prop where
  kLog : c.kLog = 22
  g : c.g = 1
  nports : c.ports.size = 1
  words : c.ports[0]!.words = 64
  segs : c.ports[0]!.segs = none
  v2 : c.ports[0]!.v2 = false
  bits : c.ports[0]!.bits = 1024
  chunks : c.ports[0]!.chunks = 1
  outNet : c.outNet = c.unitNet
  typed : c.typed = none
  perVu0 : c.perVu.getD c.hmNet 0 = 1
  perVu1 : c.perVu.getD c.hmNet 1 = 1
  slots : ((HmRow.rowSlots c.ports).2[0]!).size = 1
  msg : Discharge.Hm.shaBase c 0 0 0 + Discharge.Hm.shaIn c (2 * Discharge.Hm.shaSub c 0 0 + 1) = 262656
  salt : c.slotCol c.hmNet 0 0 + Discharge.Hm.hmIn c 2 = 1536
  dummy : c.dummyDigests.getD 0 ByteArray.empty = BC
  n : pub.n = 1
  digest : pub.digest 0 0 1 = BC
  count : c.unitRange.count = 1
  owords : (c.unit.outGroups.getD 0 default).words = 1
  obase : c.unitRange.start + (c.unit.outGroups.getD 0 default).col * WORD = 524416
  leaves : (c.leavesOut.getD 0 #[]).size = 1
  obits : Discharge.Layout.outBits c 0 = 16
  outputs : pub.outputs = #[0]
  dout : c.dummyOutputs = #[0]

section facts

variable {c : Circuit} {pub : Public} (F : RegionFacts c pub)
include F

theorem ports_v1 : ∀ p ∈ c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks := by
  intro p hp
  obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem hp
  have hi0 : i = 0 := by rw [Array.length_toList, F.nports] at hi; omega
  subst hi0
  rw [Array.getElem_toList, ← getElem!_pos c.ports 0 (by rw [F.nports]; omega), F.v2, F.bits, F.words, F.chunks]
  decide

/-- Region "digest 0"'s `b ‖ c` at every block: the public file's one row digest, or the pinned dummy, both `BC`. -/
theorem bcChunk_BC (b : ℕ) : Discharge.Layout.bcChunk c pub b 0 0 = BC := by
  unfold Discharge.Layout.bcChunk
  cases hi : c.instOf pub.n b 0 with
  | none => exact F.dummy
  | some i =>
    have hi0 : i = 0 := by
      unfold Circuit.instOf at hi
      simp only [F.g, F.n] at hi
      simp at hi
      omega
    subst hi0
    show pub.digest 0 0 c.ports.size = BC
    rw [F.nports]
    exact F.digest

/-- Every output region "out" packs is 0: the public file's one output and the pinned dummy output. -/
theorem outVal_zero (b k j : ℕ) : Discharge.Layout.outVal c pub b k j = 0 := by
  unfold Discharge.Layout.outVal
  cases c.instOf pub.n b (k / c.upv) with
  | none => show c.dummyOutputs.getD _ 0 = 0; rw [F.dout]; exact getD_zero _
  | some i => show pub.outputs.getD _ 0 = 0; rw [F.outputs]; exact getD_zero _

/-- **Region "out"'s bytes are 0**: its one slot's one word packs the zero output. -/
theorem outBytes_false {b i : ℕ} (hi : i < 128) :
    ((Discharge.Layout.outBytes c pub b).getD (i / 8) default).toNat.testBit (i % 8) = false := by
  have hslot : ((List.range c.unitRange.count).map (Discharge.Layout.slotBytes c pub b)).getD 0 [] =
      Discharge.Layout.slotBytes c pub b 0 := by
    rw [List.getD_eq_getElem _ _ (by simp [F.count])]
    simp
  have hsz : 0 < (Discharge.Layout.outPack c pub b 0).size := by
    rw [Discharge.Layout.outPack_size, F.owords]; omega
  have hword : ((Discharge.Layout.outPack c pub b 0).toList.map fun x => x.toBytes.data.toList).getD 0 [] =
      (Discharge.Layout.outPack c pub b 0)[0]!.toBytes.data.toList := by
    rw [List.getD_eq_getElem _ _ (by simpa using hsz), getElem!_pos (Discharge.Layout.outPack c pub b 0) 0 hsz]
    simp
  rw [Discharge.Layout.outBytes, show i / 8 = 0 * (16 * 1) + i / 8 by omega,
    Discharge.Layout.getD_flatten_const default 0 _ (fun l hl => by
      obtain ⟨k', -, rfl⟩ := List.mem_map.1 hl
      rw [Discharge.Layout.slotBytes_length, F.owords]) (by simp [F.count]) (by omega),
    hslot, Discharge.Layout.slotBytes, show i / 8 = 0 * 16 + i / 8 by omega,
    Discharge.Layout.getD_flatten_const default 0 _ (fun l hl => by
      obtain ⟨x, -, rfl⟩ := List.mem_map.1 hl
      simp [F128.toBytes]) (by simpa using hsz) (by omega), hword,
    Discharge.Layout.toBytes_bit _ (by omega) (Nat.mod_lt _ (by norm_num)), show 8 * (i / 8) + i % 8 = i by omega]
  obtain ⟨-, -, hP⟩ := Discharge.Layout.pack_spec (vals := Discharge.Layout.outVal c pub b 0)
    (bits := Discharge.Layout.outBits c) (N := (c.leavesOut.getD (0 % c.upv) #[]).size)
    (words := (c.unit.outGroups.getD 0 default).words)
    (fun j hj => by
      rw [Nat.zero_mod, F.leaves] at hj
      obtain rfl : j = 0 := by omega
      exact F.obits)
    (fun j _ => by rw [outVal_zero F]; exact Nat.two_pow_pos 16)
    (by rw [Nat.zero_mod, F.leaves, F.owords]; norm_num) _ le_rfl
  refine (hP 0 hsz i hi).trans ?_
  rw [outVal_zero F, Nat.zero_testBit, Bool.and_false]

/-- **The trace's `b ‖ c`** at the `hm96` slot's output, given the row's commit string there: the row's message and salt
are 0 in the trace, so it is `BC`. -/
theorem zIn_bcCol {w : ℕ → Bool} (hw : ∀ y < 2 ^ 22, w y = zIn y)
    (hbc : Discharge.Hm.bytesOf w 128 (Discharge.Hm.bcCol c 0 0) = Binding.commitString
      ((HmRow.rowPrefix c.ports[0]!.words).data.toList ++
        Discharge.Hm.bytesOf w (2 * c.ports[0]!.words) (Typed.msgCol c 0 0))
      (Discharge.Hm.bytesOf w 192 (Discharge.Hm.saltCol c 0 0)))
    {q : ℕ} (hq : q < 1024) (h22 : Discharge.Hm.bcCol c 0 0 (q / 8) (q % 8) < 2 ^ 22) :
    zIn (Discharge.Hm.bcCol c 0 0 (q / 8) (q % 8)) = Discharge.Hm.bitL BC.data.toList q := by
  have hmsg : Discharge.Hm.bytesOf w (2 * c.ports[0]!.words) (Typed.msgCol c 0 0) = List.replicate 128 0 := by
    rw [F.words]
    refine bytesOf_false fun j hj t ht => ?_
    have e := Discharge.Hm.msgCol_eq c 0 0 0 F.segs (by rw [F.slots]; omega) (show j < 128 by omega) ht
    rw [show 128 * 0 + j = j by omega, F.msg] at e
    have hwb : HmNets.wordBit (8 * j + t) < 1024 := by unfold HmNets.wordBit; omega
    rw [e, hw _ (lt22 (show 262656 + HmNets.wordBit (8 * j + t) < 4194304 by omega)),
      zIn_sha (show 262144 ≤ 262656 + HmNets.wordBit (8 * j + t) by omega)
        (show 262656 + HmNets.wordBit (8 * j + t) < 270336 by omega)]
    unfold xS
    exact ite_eq_right (by omega)
  have hsalt : Discharge.Hm.bytesOf w 192 (Discharge.Hm.saltCol c 0 0) = List.replicate 192 0 := by
    refine bytesOf_false fun j hj t ht => ?_
    have e : Discharge.Hm.saltCol c 0 0 j t = 1536 + 8 * j + t := by
      unfold Discharge.Hm.saltCol
      rw [F.salt]
    rw [e, hw _ (lt22 (show 1536 + 8 * j + t < 4194304 by omega)), zIn_hm (show 1536 + 8 * j + t < 4096 by omega)]
    unfold xH
    rw [ite_eq_right (show ¬ (1536 + 8 * j + t < 512) by omega),
      ite_eq_right (show ¬ (1536 + 8 * j + t < 1536) by omega)]
  rw [hmsg, hsalt, F.words, commitString_zero] at hbc
  have hb := Discharge.Hm.bitL_bytesOf w (Discharge.Hm.bcCol c 0 0) (N := 128) (show q < 8 * 128 by omega)
  rw [hbc, hw _ h22] at hb
  exact hb.symm

end facts

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}

/-- The trace's `b ‖ c` at the concrete table's `hm96` slot, in a statement setupH accepted whose rows the trace
satisfies. -/
theorem zIn_bc (ht : tags.typed = false) (hset : Stmt.setupH tags cf pf coins tables draw partition program = .ok st)
    (hz : ((setupH_layout ht hset).model []).Satisfies (zC ((setupH_layout ht hset).model []).m))
    (F : RegionFacts st.c st.pub) {q : ℕ} (hq : q < 1024)
    (h22 : Discharge.Hm.bcCol st.c 0 0 (q / 8) (q % 8) < 2 ^ 22) :
    zIn (Discharge.Hm.bcCol st.c 0 0 (q / 8) (q % 8)) = Discharge.Hm.bitL BC.data.toList q := by
  have o : Fin (2 ^ ((setupH_layout ht hset).model []).nbl) := ⟨0, Nat.two_pow_pos _⟩
  exact zIn_bcCol F (fun y hy => blockBit_zC (S := (setupH_layout ht hset).model []) F.kLog o hy)
    (Discharge.Hm.row_bc_setupH ht hset F.outNet (ports_v1 F) hz o (by rw [F.g]; omega) (by rw [F.nports]; omega))
    hq h22

/-- **The trace satisfies the concrete table at its regions**, where it satisfies its rows and pin. -/
theorem satisfies_regions (L : FlockSoundness.Layout st) (ht : tags.typed = false)
    (hset : Stmt.setupH tags cf pf coins tables draw partition program = .ok st)
    (hz : ((setupH_layout ht hset).model []).Satisfies (zC ((setupH_layout ht hset).model []).m))
    (F : RegionFacts st.c st.pub) :
    (L.model (Refine.regionsOf st)).Satisfies (zC (L.model (Refine.regionsOf st)).m) := by
  have hmem := regions_mem (Discharge.Layout.setupH_regions hset) F.outNet F.typed
  refine ⟨hz.1, hz.2.1, ?_⟩
  intro R hR' o i hi h
  have hR'' : R ∈ st.regions.toList.map (Refine.regionOf st.c.kLog (st.m - st.c.kLog)) := hR'
  obtain ⟨reg, hreg, rfl⟩ := List.mem_map.1 hR''
  suffices key : zIn ((Refine.regionOf st.c.kLog (st.m - st.c.kLog) reg).inBlock i) =
      ((reg.bytes o.val).get! (i / 8)).toNat.testBit (i % 8) by
    apply bit_inj'
    exact (congrArg FlockSoundness.bit (block_zC (S := L.model (Refine.regionsOf st)) F.kLog o ⟨_, h⟩)).trans
      ((bit_ite _).trans (key.trans (Discharge.Layout.regionOf_bit st.c.kLog (st.m - st.c.kLog) reg o i).symm))
  rcases hmem reg hreg with ⟨p, hp, name, hmk⟩ | hmk
  · rw [F.nports] at hp
    obtain rfl : p = 0 := by omega
    obtain ⟨hrf, -, -, -, -, hlt⟩ := Discharge.Layout.mkRegion_spec hmk
    have hfree : reg.free = Discharge.Layout.hmFree st.c := by rw [hrf]
    have hbytes : reg.bytes = Discharge.Layout.hmBytes st.c st.pub 0 := by rw [hrf]
    have hlen : (Refine.regionOf st.c.kLog (st.m - st.c.kLog) reg).free.length = 10 := by
      rw [← List.length_map (f := Fin.val), Discharge.Layout.regionOf_free_map _ _ _ (fun b hb => hlt b (by
        rw [hfree] at hb; exact Array.mem_toList_iff.1 hb)), hfree, Discharge.Layout.hmFree_toList, F.g, log2_one]
      simp
    have hi0 : i < 2 ^ (Refine.regionOf st.c.kLog (st.m - st.c.kLog) reg).free.length := hi
    rw [hlen] at hi0
    have hi' : i < 1024 := lt_of_lt_of_eq hi0 (by norm_num)
    have hpv : st.c.perVu.getD st.c.hmNet 0 = 2 ^ (st.c.perVu.getD st.c.hmNet 1).log2 := by
      rw [F.perVu0, F.perVu1, log2_one, pow_zero]
    have hib := Discharge.Layout.inBlock_digest (st.m - st.c.kLog) hmk (g := 0) (j := i / 8) (t := i % 8)
      (Nat.two_pow_pos _) (by omega) (Nat.mod_lt _ (by norm_num))
    rw [show 0 * 1024 + (8 * (i / 8) + i % 8) = i by omega] at hib
    have hcol : (Refine.regionOf st.c.kLog (st.m - st.c.kLog) reg).inBlock i =
        Discharge.Hm.bcCol st.c 0 0 (i / 8) (i % 8) :=
      hib.trans (Discharge.Layout.bcAddr_eq_slotCol _ _ _ _ _ hpv)
    have h22 : Discharge.Hm.bcCol st.c 0 0 (i / 8) (i % 8) < 2 ^ 22 := by
      rw [← hcol]
      exact lt_of_lt_of_eq h (by show 2 ^ st.c.kLog = 2 ^ 22; rw [F.kLog])
    have hsz : ∀ g' ≤ 0, (Discharge.Layout.bcChunk st.c st.pub o.val g' 0).size = 128 := by
      intro g' hg'
      obtain rfl : g' = 0 := by omega
      exact (congrArg ByteArray.size (bcChunk_BC F o.val)).trans BC_size
    have hbyte : (reg.bytes o.val).get! (i / 8) = BC.data.toList.getD (i / 8) default := by
      rw [hbytes]
      have := Discharge.Layout.hmBytes_get (c := st.c) (pub := st.pub) (p := 0) (b := o.val) (g := 0) (j := i / 8)
        (by rw [F.g]; omega) (by omega) hsz
      rw [Nat.zero_mul, Nat.zero_add, bcChunk_BC F, Discharge.Layout.get!_eq_getD BC] at this
      exact this
    rw [hcol, zIn_bc ht hset hz F hi' h22, hbyte]
    rfl
  · have hmk' := hmk
    unfold Circuit.outRegion at hmk'
    dsimp only at hmk'
    obtain ⟨hrf, -, -, -, -, hlt⟩ := Discharge.Layout.mkRegion_spec hmk'
    have hfreeA := congrArg Flock.Region.free hrf
    simp only at hfreeA
    have hlen : (Refine.regionOf st.c.kLog (st.m - st.c.kLog) reg).free.length = 7 := by
      rw [← List.length_map (f := Fin.val), Discharge.Layout.regionOf_free_map _ _ _ (fun b hb => hlt b (by
        rw [hfreeA] at hb; exact Array.mem_toList_iff.1 hb)), hfreeA, F.owords, F.count, log2_one]
      simp
    have hi0 : i < 2 ^ (Refine.regionOf st.c.kLog (st.m - st.c.kLog) reg).free.length := hi
    rw [hlen] at hi0
    have hi' : i < 128 := lt_of_lt_of_eq hi0 (by norm_num)
    have hm : i < 2 ^ Discharge.Layout.outLw st.c := by
      unfold Discharge.Layout.outLw
      rw [F.owords, log2_one]
      exact lt_of_lt_of_le hi' (by norm_num)
    have hib := Discharge.Layout.inBlock_out (st.m - st.c.kLog) hmk (k := 0) (m := i) (Nat.two_pow_pos _) hm
    simp only [Nat.zero_mul, Nat.zero_add, Nat.add_zero] at hib
    rw [F.obase] at hib
    rw [hib, zIn_unit (show 524288 ≤ 524416 + i by omega) (show 524416 + i ≠ 524544 by omega),
      Discharge.Layout.get!_eq_getD, Discharge.Layout.outRegion_bytes hmk o.val]
    exact (outBytes_false F hi').symm

/-! ## The concrete circuit's and public file's facts -/

section cW

set_option maxHeartbeats 2000000 -- the kernel evaluates the parsed circuit's and public file's fields

variable (ls lh : HmNets.Laid) (H : ByteArray) (A B A' B' : Sparse) (s1 s3 s4 s6 : String)
  (hello : ByteArray → Nat → String)

theorem pub_stW : (stW ls lh H A B A' B' s1 s3 s4 s6 hello).pub = pubV ls lh H A B A' B' s1 s3 s4 := by kernel_rfl'

theorem g_cW : (cW ls lh H A B A' B' s1).g = 1 := by kernel_rfl'
theorem nports_cW : (cW ls lh H A B A' B' s1).ports.size = 1 := by kernel_rfl'
theorem words_cW : (cW ls lh H A B A' B' s1).ports[0]!.words = 64 := by kernel_rfl'
theorem segs_cW : (cW ls lh H A B A' B' s1).ports[0]!.segs = none := by kernel_rfl'
theorem v2_cW : (cW ls lh H A B A' B' s1).ports[0]!.v2 = false := by kernel_rfl'
theorem bits_cW : (cW ls lh H A B A' B' s1).ports[0]!.bits = 1024 := by kernel_rfl'
theorem chunks_cW : (cW ls lh H A B A' B' s1).ports[0]!.chunks = 1 := by kernel_rfl'
theorem outNet_cW : (cW ls lh H A B A' B' s1).outNet = (cW ls lh H A B A' B' s1).unitNet := by kernel_rfl'
theorem typed_cW : (cW ls lh H A B A' B' s1).typed = none := by kernel_rfl'
theorem perVu0_cW : (cW ls lh H A B A' B' s1).perVu.getD (cW ls lh H A B A' B' s1).hmNet 0 = 1 := by kernel_rfl'
theorem perVu1_cW : (cW ls lh H A B A' B' s1).perVu.getD (cW ls lh H A B A' B' s1).hmNet 1 = 1 := by kernel_rfl'
theorem slots_cW : ((HmRow.rowSlots (cW ls lh H A B A' B' s1).ports).2[0]!).size = 1 := by kernel_rfl'
theorem msg_cW : Discharge.Hm.shaBase (cW ls lh H A B A' B' s1) 0 0 0 +
    Discharge.Hm.shaIn (cW ls lh H A B A' B' s1) (2 * Discharge.Hm.shaSub (cW ls lh H A B A' B' s1) 0 0 + 1) = 262656 := by
  kernel_rfl'
theorem salt_cW : (cW ls lh H A B A' B' s1).slotCol (cW ls lh H A B A' B' s1).hmNet 0 0 +
    Discharge.Hm.hmIn (cW ls lh H A B A' B' s1) 2 = 1536 := by kernel_rfl'
theorem dummy_cW : (cW ls lh H A B A' B' s1).dummyDigests.getD 0 ByteArray.empty = BC := by kernel_rfl'
theorem owords_cW : ((cW ls lh H A B A' B' s1).unit.outGroups.getD 0 default).words = 1 := by kernel_rfl'
theorem obase_cW : (cW ls lh H A B A' B' s1).unitRange.start +
    ((cW ls lh H A B A' B' s1).unit.outGroups.getD 0 default).col * WORD = 524416 := by kernel_rfl'
theorem leaves_cW : ((cW ls lh H A B A' B' s1).leavesOut.getD 0 #[]).size = 1 := by kernel_rfl'
theorem obits_cW : Discharge.Layout.outBits (cW ls lh H A B A' B' s1) 0 = 16 := by kernel_rfl'
theorem dout_cW : (cW ls lh H A B A' B' s1).dummyOutputs = #[0] := by kernel_rfl'

theorem n_pubV : (pubV ls lh H A B A' B' s1 s3 s4).n = 1 := by kernel_rfl'
theorem digest_pubV : (pubV ls lh H A B A' B' s1 s3 s4).digest 0 0 1 = BC := by
  unfold pubV
  try simp only [bodyB, bc_eq]
  kernel_rfl'
theorem outputs_pubV : (pubV ls lh H A B A' B' s1 s3 s4).outputs = #[0] := by
  unfold pubV
  try simp only [bodyB, bc_eq]
  kernel_rfl'

theorem regionFacts_cW : RegionFacts (cW ls lh H A B A' B' s1) (pubV ls lh H A B A' B' s1 s3 s4) where
  kLog := kLog_cW ls lh H A B A' B' s1
  g := g_cW ls lh H A B A' B' s1
  nports := nports_cW ls lh H A B A' B' s1
  words := words_cW ls lh H A B A' B' s1
  segs := segs_cW ls lh H A B A' B' s1
  v2 := v2_cW ls lh H A B A' B' s1
  bits := bits_cW ls lh H A B A' B' s1
  chunks := chunks_cW ls lh H A B A' B' s1
  outNet := outNet_cW ls lh H A B A' B' s1
  typed := typed_cW ls lh H A B A' B' s1
  perVu0 := perVu0_cW ls lh H A B A' B' s1
  perVu1 := perVu1_cW ls lh H A B A' B' s1
  slots := slots_cW ls lh H A B A' B' s1
  msg := msg_cW ls lh H A B A' B' s1
  salt := salt_cW ls lh H A B A' B' s1
  dummy := dummy_cW ls lh H A B A' B' s1
  n := n_pubV ls lh H A B A' B' s1 s3 s4
  digest := digest_pubV ls lh H A B A' B' s1 s3 s4
  count := (unitRange_cW ls lh H A B A' B' s1).1
  owords := owords_cW ls lh H A B A' B' s1
  obase := obase_cW ls lh H A B A' B' s1
  leaves := leaves_cW ls lh H A B A' B' s1
  obits := obits_cW ls lh H A B A' B' s1
  outputs := outputs_pubV ls lh H A B A' B' s1 s3 s4
  dout := dout_cW ls lh H A B A' B' s1

theorem regionFacts_stW :
    RegionFacts (stW ls lh H A B A' B' s1 s3 s4 s6 hello).c (stW ls lh H A B A' B' s1 s3 s4 s6 hello).pub := by
  rw [c_stW, pub_stW]
  exact regionFacts_cW ls lh H A B A' B' s1 s3 s4

end cW

/-- **The trace satisfies the concrete table**, its public regions included: its rows and pin (`satisfies_stW`), and at
"digest 0" and "out" the public values (`satisfies_regions`). -/
theorem satisfies_regions_stW (ls lh : HmNets.Laid) (H : ByteArray) (A B A' B' : Sparse) (s1 s3 s4 s6 : String)
    (hello : ByteArray → Nat → String) (ht : tags.typed = false)
    (hset : Stmt.setupH tags cf pf coins tables draw partition program = .ok (stW ls lh H A B A' B' s1 s3 s4 s6 hello))
    (L : FlockSoundness.Layout (stW ls lh H A B A' B' s1 s3 s4 s6 hello)) (hn : Properties.Flock.Nets.FixedNetsBuild) :
    (L.model (Refine.regionsOf (stW ls lh H A B A' B' s1 s3 s4 s6 hello))).Satisfies
      (zC (L.model (Refine.regionsOf (stW ls lh H A B A' B' s1 s3 s4 s6 hello))).m) :=
  satisfies_regions L ht hset
    (satisfies_stW ls lh H A B A' B' s1 s3 s4 s6 hello (setupH_layout ht hset)
      (Discharge.Aliased.setupH_unitFacts ht hset) hn)
    (regionFacts_stW ls lh H A B A' B' s1 s3 s4 s6 hello)

end FlockDeploy.Concrete.Trace
