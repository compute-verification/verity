module

public import Proofs.Flock.SessionLoop.Concrete.Setup
public import Proofs.Flock.Claims.DeployInputs
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
import Flock.Registered
import Flock.RecSetup
import Flock.RecCoins
import Flock.RecVStar
import Flock.ProvedScope
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
import all Init.Data.String.TakeDrop
import all Init.Data.String.Slice
import all Init.Data.String.Pattern.Basic
import all Init.Data.String.Pattern.String
import all Init.Data.String.Pattern.Char
import all Init.Data.String.Search
import all Init.Data.String.Iterate
import all Init.Data.String.Decode
import all Init.Data.String.Termination
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
import Flock.CanonForm
import all Init.Data.Array.Attach
import all Init.WF

/-! # The concrete statement's inputs: `Raw.ins` and `Ins.setup`

`wC` is the statement's `Raw`: the `verity/flock-circuit@eb90718f` statement, the circuit file `bytesOf`, the public
file `pubOf` at the circuit's SHA-512, no lookup table, `--zk`, one table and nothing else. `ins_W` walks `Raw.ins` on
it to `insC`, and `setup_W` walks `Ins.setup` to `setW`, at the session's tags (`tags_rC`) and setup (`setup_rC`). The
canonical forms the setup compares (`canon`, `canonUtf8`) exist because each value holds only integers (`IntJ`), and
are named (`s1C`, `s3C`, `s4Of`, `s6C`, `helloC`) so that no walk evaluates them.
-/

@[expose] public section
open Flock RecDeploy.VStarNets FlockVBridge.Names
open Flock.Readable (Read)
open Lean (Json)
open Std.DTreeMap.Internal (Impl)

set_option maxHeartbeats 0 -- the walks below step through `Raw.ins` and `Ins.setup` and leave closed checks to the kernel
set_option maxRecDepth 200000

namespace FlockDeploy.Concrete

theorem fromUTF8?_toByteArray (s : String) : String.fromUTF8? s.toByteArray = some s := by
  unfold String.fromUTF8?
  rw [dif_pos s.isValidUTF8]
  rfl

theorem nl_data : "\n".toByteArray.data.toList = [10] := by decide +kernel

theorem nl_size : "\n".toByteArray.size = 1 := by decide +kernel

/-- The first newline at or after `f` in `B ++ "\n" ++ R` ends `B` when `B` has none past `f`. -/
theorem findByte_nl (B R : ByteArray) (f : Nat) (hf : f ≤ B.size) (hB : (10 : UInt8) ∉ B.data.toList.drop f) :
    findByte (B ++ ("\n".toByteArray ++ R)) 10 f = some B.size := by
  have hn : List.findIdx? (· == (10 : UInt8)) (B.data.toList.drop f) = none :=
    List.findIdx?_eq_none_iff.2 fun x hx => by
      cases hx10 : (x == 10)
      · rfl
      · exact absurd ((beq_iff_eq.1 hx10) ▸ hx) hB
  have hl : f ≤ B.data.toList.length := by rw [Array.length_toList]; exact hf
  rw [findByte_eq, ByteArray.data_append, ByteArray.data_append, Array.toList_append, Array.toList_append, nl_data,
    List.drop_append_of_le_length hl, List.findIdx?_append, hn, Option.none_or, List.cons_append,
    List.findIdx?_cons, beq_self_eq_true, if_pos rfl, Option.map_some, Option.map_some, Nat.zero_add]
  congr 1
  rw [List.length_drop, Array.length_toList]
  have : B.data.size = B.size := rfl
  omega

/-- `maskLogOf` of a file whose first two lines are `A` and `M` reads `M`. -/
theorem maskLog_split (A M R : ByteArray) (hA : (10 : UInt8) ∉ A.data.toList) (hM : (10 : UInt8) ∉ M.data.toList) :
    Zk.maskLogOf (A ++ ("\n".toByteArray ++ (M ++ ("\n".toByteArray ++ R)))) =
      match String.fromUTF8? M with
      | none => 14
      | some line =>
        match JsonParse.parse (line.drop 5).toString with
        | .ok j => (j.getObjValAs? Nat "mask_log").toOption.getD 14
        | .error _ => 14 := by
  have h1 : findByte (A ++ ("\n".toByteArray ++ (M ++ ("\n".toByteArray ++ R)))) 10 = some A.size :=
    findByte_nl A _ 0 (Nat.zero_le _) (by rw [List.drop_zero]; exact hA)
  have hsz : (A ++ "\n".toByteArray ++ M).size = A.size + 1 + M.size := by
    rw [ByteArray.size_append, ByteArray.size_append, nl_size]
  have hc : A ++ ("\n".toByteArray ++ (M ++ ("\n".toByteArray ++ R))) = (A ++ "\n".toByteArray ++ M) ++ ("\n".toByteArray ++ R) := by
    simp only [ByteArray.append_assoc]
  have h2 : findByte (A ++ ("\n".toByteArray ++ (M ++ ("\n".toByteArray ++ R)))) 10 (A.size + 1) =
      some (A.size + 1 + M.size) := by
    rw [hc, findByte_nl (A ++ "\n".toByteArray ++ M) R (A.size + 1) (by rw [hsz]; omega), hsz]
    rw [ByteArray.data_append, ByteArray.data_append, Array.toList_append, Array.toList_append, nl_data,
      List.drop_left' (by rw [List.length_append, Array.length_toList]; rfl)]
    exact hM
  have hc' : A ++ ("\n".toByteArray ++ (M ++ ("\n".toByteArray ++ R))) = (A ++ "\n".toByteArray) ++ (M ++ ("\n".toByteArray ++ R)) := by
    simp only [ByteArray.append_assoc]
  have h3 : (A ++ ("\n".toByteArray ++ (M ++ ("\n".toByteArray ++ R)))).extract (A.size + 1) (A.size + 1 + M.size) = M := by
    have h := ByteArray.extract_append_size_add' (a := A ++ "\n".toByteArray) (b := M ++ ("\n".toByteArray ++ R))
      (i := 0) (j := M.size) (k := A.size + 1) (by rw [ByteArray.size_append, nl_size])
    rw [Nat.add_zero, ByteArray.extract_append_eq_left rfl] at h
    rw [hc', h]
  unfold Zk.maskLogOf
  simp only [h1, h2, Option.getD_some, h3]
  rfl

def restLs (ls lh : HmNets.Laid) : List String :=
  secLs "unit" "copy16" unitLaid ++ secLs VStar.SHA_NET VStar.SHA_NET ls ++ secLs VStar.HM_NET VStar.HM_NET lh

theorem bytes_split (ls lh : HmNets.Laid) : bytesOf ls lh =
    ("flock-circuit " ++ "copy16").toByteArray ++ ("\n".toByteArray ++ (("META " ++ metaText).toByteArray ++
      ("\n".toByteArray ++ (joinNl (restLs ls lh)).toByteArray))) := by
  simp only [bytesOf, lines, fileLs, restLs, joinNl, List.cons_append, List.nil_append, List.append_assoc,
    List.map_cons, String.join_cons, String.toUTF8_eq_toByteArray, String.toByteArray_append, ByteArray.append_assoc]

theorem first_noTen : (10 : UInt8) ∉ ("flock-circuit " ++ "copy16").toByteArray.data.toList := by decide +kernel

theorem meta_noTen : (10 : UInt8) ∉ ("META " ++ metaText).toByteArray.data.toList := by
  rw [String.toByteArray_append, ByteArray.data_append, Array.toList_append, List.mem_append, not_or]
  refine ⟨by decide +kernel, ?_⟩
  have h := utf8_noTen metaChars (by have := noNl_meta; unfold NoNl metaText at this; rwa [String.toList_ofList] at this)
  rwa [String.toUTF8_eq_toByteArray] at h

theorem mask_mjC : (mjC.getObjValAs? Nat "mask_log").toOption.getD 14 = 14 := by decide +kernel

/-- `maskLogOf` of a file whose second line `s` parses, past its first five characters, as `j`. -/
theorem maskLog_of (A R : ByteArray) (s : String) (hA : (10 : UInt8) ∉ A.data.toList)
    (hM : (10 : UInt8) ∉ s.toByteArray.data.toList) {j : Json} (hj : JsonParse.parse (s.drop 5).toString = .ok j) :
    Zk.maskLogOf (A ++ ("\n".toByteArray ++ (s.toByteArray ++ ("\n".toByteArray ++ R)))) =
      (j.getObjValAs? Nat "mask_log").toOption.getD 14 := by
  rw [maskLog_split _ _ _ hA hM, fromUTF8?_toByteArray]
  dsimp only
  rw [hj]

theorem maskLog_bytes (ls lh : HmNets.Laid) : Zk.maskLogOf (bytesOf ls lh) = 14 := by
  rw [bytes_split, maskLog_of _ _ _ first_noTen meta_noTen (by rw [drop_meta]; exact parse_meta)]
  exact mask_mjC

def NAME : String := "verity/flock-circuit@eb90718f"

def wC (ls lh : HmNets.Laid) (H : ByteArray) : Raw :=
  { statement := NAME, circuit := bytesOf ls lh, publicFile := pubOf (hexOf H), tables := [], zk := true,
    osCoins := false, count := 1, draw := none, partition := none, program := none, work := none, workTable := none,
    stratified := none, closure := none, registered := none, publicInputs := none }

def rC (ls lh : HmNets.Laid) (H : ByteArray) : Read :=
  { I := { tags := t0, circuitFile := bytesOf ls lh, publicFile := pubOf (hexOf H), coins := coins0, count := 1,
           tables := [], partition := none, program := none, work := none, stratified := none, closure := none },
    zk := true, draw := none, publicInputs := none, registered := false }

def insC (ls lh : HmNets.Laid) (H : ByteArray) : Ins := { stmtTags := t0, registered := none, read := rC ls lh H }

theorem byName_t0 : Tags.byName? NAME = some t0 := by kernel_rfl

theorem plain_t0 : t0.plainLeaves = false := by decide +kernel

theorem ins_W (ls lh : HmNets.Laid) (H : ByteArray) : (wC ls lh H).ins = .ok (insC ls lh H) := by
  fwd_walk [byName_t0, plain_t0] block [HmNets.check, HmNets.shaLaid, HmNets.hmLaid, canon, canonUtf8, Net.parse]
    shallow [metaText]

theorem forIn_one {α β ε : Type} (a : α) (b : β) (f : α → β → Except ε (ForInStep β)) :
    forIn #[a] b f = (f a b >>= fun s => match s with | .done b' => pure b' | .yield b' => pure b') := by
  rw [← Array.forIn_toList]
  simp only [List.toList_toArray, List.forIn_cons, List.forIn_nil]
  rfl

theorem forM_one {α ε : Type} (a : α) (f : α → Except ε PUnit) : (#[a] : Array α).forM f = f a := by
  cases h : f a <;> simp [Array.forM, Array.foldlM, Array.foldlM.loop, h, bind, Except.bind, pure, Except.pure]

theorem cw_name (ls lh : HmNets.Laid) (H : ByteArray) (A B A' B' : Sparse) (s1 : String) :
    (cW ls lh H A B A' B' s1).unit.name = "copy16" := by kernel_rfl

fwd_def setW setup_W (ls lh : HmNets.Laid) (H : ByteArray) (A B A' B' : Sparse) (s1 s3 s4 s6 : String)
    (hello : ByteArray → Nat → String) (s : Ins)
    (hr : s.read = rC ls lh H) (hreg : s.registered = none) (hst : s.stmtTags = t0)
    (hT : Read.tags (rC ls lh H) = tz)
    (hS : Read.setup (rC ls lh H) = .ok (tabW ls lh H A B A' B' s1 s3 s4 s6 hello)) :
    Ins.setup s by
  [hr, hreg, hst, hT, hS, forIn_one, forM_one, shapeOk_stW ls lh H A B A' B' s1 s3 s4 s6 hello,
    cw_name ls lh H A B A' B' s1, open_none, res_none, sparse_none, opening_none]
  block [HmNets.check, HmNets.shaLaid, HmNets.hmLaid, canon, canonUtf8, Net.parse, HmRow.parse, HmRow.loadPublic,
    HmRow.checkPublic, Stmt.setupH, Stmt.setupTables, Zk.maskLogOf, Zk.shapeOk, HmIn.delta]
  shallow [metaText]

theorem tags_rC (ls lh : HmNets.Laid) (H : ByteArray) : Read.tags (rC ls lh H) = tz := by
  show Zk.tags t0 (Zk.maskLogOf (bytesOf ls lh)) 1 = tz
  rw [maskLog_bytes]
  rfl

theorem setup_rC (ls lh : HmNets.Laid) (H : ByteArray) (A B A' B' : Sparse) (s1 s3 s4 s6 : String)
    (hello : ByteArray → Nat → String)
    (hH : Sha512.hash (bytesOf ls lh) = H)
    (hls : VStar.netLines ls = 256002) (hlh : VStar.netLines lh = 246786)
    (hps : Net.parse (netLs "sha512x3" ls).toArray = .ok (shaB A B))
    (hph : Net.parse (netLs "hm96" lh).toArray = .ok (hmB A' B'))
    (hc1 : canon lsJ = .ok s1) (hc2 : canon pinJ = .ok s1)
    (hck : ∀ c : Circuit, (c.nets.getD c.shaNet default).2 = shaB A B → (c.nets.getD c.hmNet default).2 = hmB A' B' →
      HmNets.check c = .ok ())
    (hu1 : canon unitsJ = .ok s3) (hu2 : canon (HmRow.unitsEntry (cW ls lh H A B A' B' s1) #[0]) = .ok s3)
    (hd : canonUtf8 (hdrJ (hexOf H)) = .ok s4)
    (hid : canonUtf8 tz.identity = .ok s6) (hhello : ∀ σ m, helloOf tz σ m coins0 = .ok (hello σ m)) :
    Read.setup (rC ls lh H) = .ok (tabW ls lh H A B A' B' s1 s3 s4 s6 hello) := by
  unfold Read.setup
  rw [tags_rC]
  exact setupTables_W ls lh H A B A' B' s1 s3 s4 s6 hello hH hls hlh hps hph hc1 hc2 hck hu1 hu2 hd hid hhello

mutual
/-- Every number in the value is an integer (`canon`'s one refusal). -/
def IntJ : Json → Bool
  | .null => true
  | .bool _ => true
  | .num n => n.exponent == 0
  | .str _ => true
  | .arr ⟨a⟩ => IntJs a
  | .obj ⟨⟨t⟩⟩ => IntJo t
def IntJs : List Json → Bool
  | [] => true
  | j :: js => IntJ j && IntJs js
def IntJo : Impl String (fun _ => Json) → Bool
  | .leaf => true
  | .inner _ _ v l r => IntJo l && IntJ v && IntJo r
end

theorem intJs_mem : ∀ {l : List Json}, IntJs l = true → ∀ x ∈ l, IntJ x = true
  | [], _, _, hx => by simp at hx
  | j :: js, h, x, hx => by
    simp only [IntJs, Bool.and_eq_true] at h
    rcases List.mem_cons.1 hx with rfl | hx
    · exact h.1
    · exact intJs_mem h.2 x hx

theorem intJo_mem : ∀ {t : Impl String (fun _ => Json)}, IntJo t = true → ∀ q ∈ t.toListModel, IntJ q.2 = true
  | .leaf, _, _, hq => by simp at hq
  | .inner _ k v l r, h, q, hq => by
    simp only [IntJo, Bool.and_eq_true] at h
    simp only [Std.DTreeMap.Internal.Impl.toListModel_inner, List.mem_append, List.mem_cons] at hq
    rcases hq with hq | rfl | hq
    · exact intJo_mem h.1.1 q hq
    · exact h.1.2
    · exact intJo_mem h.2 q hq

theorem isOk_bind {ε α β : Type} {x : Except ε α} {f : α → Except ε β} (hx : x.isOk = true)
    (hf : ∀ a, (f a).isOk = true) : (x >>= f).isOk = true := by
  cases x with
  | error e => cases hx
  | ok a => exact hf a

theorem mapM_list_ok {α β ε : Type} (f : α → Except ε β) :
    ∀ l : List α, (∀ x ∈ l, (f x).isOk = true) → (l.mapM f).isOk = true
  | [], _ => rfl
  | x :: xs, h => by
    rw [List.mapM_cons]
    exact isOk_bind (h x List.mem_cons_self) fun _ =>
      isOk_bind (mapM_list_ok f xs fun y hy => h y (List.mem_cons_of_mem _ hy)) fun _ => rfl

theorem mapM_arr_ok {α β ε : Type} (f : α → Except ε β) (a : Array α) (h : ∀ x ∈ a, (f x).isOk = true) :
    (a.mapM f).isOk = true := by
  rw [Array.mapM_eq_mapM_toList]
  have := mapM_list_ok f a.toList fun x hx => h x (Array.mem_toList_iff.1 hx)
  revert this
  cases a.toList.mapM f <;> intro h <;> first | rfl | cases h

theorem mem_toArray_model {kvs : Std.TreeMap.Raw String Json} {p : String × Json} (h : p ∈ kvs.toArray) :
    ∃ q ∈ kvs.inner.inner.toListModel, q.2 = p.2 := by
  rw [← Array.mem_toList_iff, Std.TreeMap.Raw.toList_toArray] at h
  obtain ⟨⟨t⟩⟩ := kvs
  simp only [Std.TreeMap.Raw.toList, Std.DTreeMap.Raw.Const.toList,
    Std.DTreeMap.Internal.Impl.Const.toList_eq_toListModel_map, List.mem_map] at h
  obtain ⟨q, hm, he⟩ := h
  exact ⟨q, hm, congrArg Prod.snd he⟩

theorem intJ_obj {kvs : Std.TreeMap.Raw String Json} (h : IntJ (.obj kvs) = true) {p : String × Json}
    (hp : p ∈ kvs.toArray) : IntJ p.2 = true := by
  obtain ⟨⟨t⟩⟩ := kvs
  obtain ⟨q, hq, he⟩ := mem_toArray_model hp
  rw [← he]
  exact intJo_mem h q hq

theorem intJ_arr {a : Array Json} (h : IntJ (.arr a) = true) {x : Json} (hx : x ∈ a) : IntJ x = true := by
  obtain ⟨l⟩ := a
  exact intJs_mem h x (Array.mem_toList_iff.2 hx)

/-- **`canon` refuses only a non-integer number.** -/
theorem canon_ok : ∀ (n : Nat) (j : Json), sizeOf j < n → IntJ j = true → (canon j).isOk = true
  | 0, _, h, _ => absurd h (Nat.not_lt_zero _)
  | n + 1, j, hs, hi => by
    match j, hi, hs with
    | .null, _, _ => rw [canon]; rfl
    | .bool b, _, _ => rw [canon]; rfl
    | .num x, hi, _ =>
      rw [canon]
      have : (x.exponent == 0) = true := hi
      rw [if_pos this]; rfl
    | .str s, _, _ => rw [canon]; rfl
    | .arr a, hi, hs =>
      rw [canon]
      refine isOk_bind (mapM_arr_ok _ _ fun x _ => canon_ok n x.1 ?_ (intJ_arr hi x.2)) fun _ => rfl
      have := Array.sizeOf_lt_of_mem x.2
      simp only [Json.arr.sizeOf_spec] at hs
      omega
    | .obj kvs, hi, hs =>
      rw [canon]
      refine isOk_bind (mapM_arr_ok _ _ fun p _ =>
        isOk_bind (canon_ok n p.1.2 ?_ (intJ_obj hi p.2)) fun _ => rfl) fun _ => rfl
      have := sizeOf_lt_of_mem_toArray p.2
      omega

/-- **`canonUtf8` refuses only a non-integer number.** -/
theorem canonUtf8_ok : ∀ (n : Nat) (j : Json), sizeOf j < n → IntJ j = true → (canonUtf8 j).isOk = true
  | 0, _, h, _ => absurd h (Nat.not_lt_zero _)
  | n + 1, j, hs, hi => by
    match j, hi, hs with
    | .null, _, _ => rw [canonUtf8]; rfl
    | .bool b, _, _ => rw [canonUtf8]; rfl
    | .num x, hi, _ =>
      rw [canonUtf8]
      have : (x.exponent == 0) = true := hi
      rw [if_pos this]; rfl
    | .str s, _, _ => rw [canonUtf8]; rfl
    | .arr a, hi, hs =>
      rw [canonUtf8]
      refine isOk_bind (mapM_arr_ok _ _ fun x _ => canonUtf8_ok n x.1 ?_ (intJ_arr hi x.2)) fun _ => rfl
      have := Array.sizeOf_lt_of_mem x.2
      simp only [Json.arr.sizeOf_spec] at hs
      omega
    | .obj kvs, hi, hs =>
      rw [canonUtf8]
      refine isOk_bind (mapM_arr_ok _ _ fun p _ =>
        isOk_bind (canonUtf8_ok n p.1.2 ?_ (intJ_obj hi p.2)) fun _ => rfl) fun _ => rfl
      have := sizeOf_lt_of_mem_toArray p.2
      omega

theorem ok_of_isOk {ε α : Type} {e : Except ε α} (d : α) (h : e.isOk = true) : e = .ok (e.toOption.getD d) := by
  cases e with
  | error _ => cases h
  | ok a => rfl

theorem canon_eq {j : Json} (h : IntJ j = true) : canon j = .ok ((canon j).toOption.getD "") :=
  ok_of_isOk _ (canon_ok _ j (Nat.lt_succ_self _) h)

theorem canonUtf8_eq {j : Json} (h : IntJ j = true) : canonUtf8 j = .ok ((canonUtf8 j).toOption.getD "") :=
  ok_of_isOk _ (canonUtf8_ok _ j (Nat.lt_succ_self _) h)

/-! ### `HmNets.check` at nets equal to the verifier's -/

theorem checkNet_of {nets : Array (String × Net)} {n : Nat} {name what : String} {gen : Thunk (Flock.V Net)} {g : Net}
    (hg : gen.get = .ok g) (h : (nets.getD n default).2 = g) : HmNets.checkNet nets n name what gen = .ok () := by
  unfold HmNets.checkNet
  rw [hg]
  simp only [h, if_true]
  rfl

theorem check_of {N1 N2 : Net} (hsha : HmNets.netOf "sha512x3" (HmNets.shaLaid ()) = .ok N1)
    (hhm : HmNets.netOf "hm96" (HmNets.hmLaid ()) = .ok N2) (c : Circuit)
    (h1 : (c.nets.getD c.shaNet default).2 = N1) (h2 : (c.nets.getD c.hmNet default).2 = N2) :
    HmNets.check c = .ok () := by
  unfold HmNets.check
  rw [HmNets.checkNets, checkNet_of (HmNets.sha512x3Net_get.trans hsha) h1,
    checkNet_of (HmNets.hm96Net_get.trans hhm) h2]
  rfl

/-! ### Canonical forms that exist (`canon` refuses only a non-integer number) -/

def s4Of (H : ByteArray) : String := (canonUtf8 (hdrJ (hexOf H))).toOption.getD ""

theorem intJ_hdr (s : String) : IntJ (hdrJ s) = true := by kernel_rfl

theorem hd_of (H : ByteArray) : canonUtf8 (hdrJ (hexOf H)) = .ok (s4Of H) := canonUtf8_eq (intJ_hdr _)

def s6C : String := (canonUtf8 tz.identity).toOption.getD ""

theorem intJ_id : IntJ tz.identity = true := by decide +kernel

theorem hid_of : canonUtf8 tz.identity = .ok s6C := canonUtf8_eq intJ_id

def helloC (σ : ByteArray) (m : Nat) : String := (helloOf tz σ m coins0).toOption.getD ""

theorem hello_isOk (σ : ByteArray) (m : Nat) : (helloOf tz σ m coins0).isOk = true :=
  canon_ok _ _ (Nat.lt_succ_self _) (by kernel_rfl)

theorem hhello_of (σ : ByteArray) (m : Nat) : helloOf tz σ m coins0 = .ok (helloC σ m) := ok_of_isOk _ (hello_isOk σ m)

/-! ### The units entry -/

def UEJ : Json := Json.mkObj [("vus_per_block", Lean.toJson (1 : Nat)), ("units_per_vu", Lean.toJson (1 : Nat)),
  ("indices", Json.arr #[Lean.toJson (0 : Nat)]), ("blocks", Json.arr #[Json.arr #[Lean.toJson (0 : Nat)]])]

theorem ue_eq (ls lh : HmNets.Laid) (H : ByteArray) (A B A' B' : Sparse) (s1 : String) :
    HmRow.unitsEntry (cW ls lh H A B A' B' s1) #[0] = UEJ := by kernel_rfl

def s3C : String := (canon unitsJ).toOption.getD ""

theorem hu1_of : canon unitsJ = .ok s3C := canon_eq (j := unitsJ) (by decide +kernel)

theorem canon_obj_congr {m1 m2 : Std.TreeMap.Raw String Json} (h : m1.toArray = m2.toArray) :
    canon (.obj m1) = canon (.obj m2) := by
  rw [canon, canon,
    Array.mapM_subtype (g := fun p => do pure (p.1, ← canon p.2)) (fun _ _ => rfl),
    Array.mapM_subtype (g := fun p => do pure (p.1, ← canon p.2)) (fun _ _ => rfl),
    Array.unattach_attach, Array.unattach_attach, h]

def objM (j : Json) : Std.TreeMap.Raw String Json := match j with
  | .obj m => m
  | _ => ∅

theorem units_objM : unitsJ = .obj (objM unitsJ) := by kernel_rfl
theorem ue_objM : UEJ = .obj (objM UEJ) := by kernel_rfl
theorem ue_arr : (objM UEJ).toArray = (objM unitsJ).toArray := by kernel_rfl

theorem hu2_of (ls lh : HmNets.Laid) (H : ByteArray) (A B A' B' : Sparse) (s1 : String) :
    canon (HmRow.unitsEntry (cW ls lh H A B A' B' s1) #[0]) = .ok s3C := by
  rw [ue_eq, ue_objM, canon_obj_congr ue_arr, ← units_objM]
  exact hu1_of

/-- The pinned key's SHA-512, as every META `row_key_sha512` and leaf scheme states it. The kernel derives the key
(`keyFromLabel`) and hashes it. -/
theorem keyHex_lit : hexOf (Sha512.hash Hm96.Default512.key) =
    "afc5a60c85d8aa75cb4c7f726588d299a1712e1c6a621f09ca64809a5fedd744a0f6b0201d05e82b29cd5b274e6c29670cd4e4ab10831911fb1f9e0c7019df22" := by
  decide +kernel

def LS : Json :=
  Json.mkObj [("id", .str "hm96-sha512/v1"), ("inner", .str "sha512 column digest"), ("salt_bytes", Lean.toJson (192 : Nat)),
    ("salt_source", .str "chacha20 expansion of a fresh 256-bit os key per proof, never reused or stored"),
    ("key", .str "default_key, a statement constant for every tree"),
    ("key_sha512", .str "afc5a60c85d8aa75cb4c7f726588d299a1712e1c6a621f09ca64809a5fedd744a0f6b0201d05e82b29cd5b274e6c29670cd4e4ab10831911fb1f9e0c7019df22")]

theorem pin_eq : pinJ = Tags.leafSchemeHm96 := by kernel_rfl

theorem scheme_eq : Tags.leafSchemeHm96 = LS := by
  unfold Tags.leafSchemeHm96
  rw [keyHex_lit]
  rfl

theorem ls_objM : lsJ = .obj (objM lsJ) := by kernel_rfl
theorem LS_objM : LS = .obj (objM LS) := by kernel_rfl
theorem ls_arr : (objM lsJ).toArray = (objM LS).toArray := by kernel_rfl

def s1C : String := (canon pinJ).toOption.getD ""

theorem hc2_of : canon pinJ = .ok s1C := canon_eq (j := pinJ) (by decide +kernel)

theorem hc1_of : canon lsJ = .ok s1C := by
  rw [ls_objM, canon_obj_congr ls_arr, ← LS_objM, ← scheme_eq, ← pin_eq]
  exact hc2_of

end FlockDeploy.Concrete
