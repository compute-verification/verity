module

public import Proofs.Properties.Verity.Sampled.HonestSession.CircuitFile
public import Proofs.Properties.Verity.Sampled.HonestSession.CircuitFileWalk
public import Proofs.Flock.SessionLoop.Concrete.Stmt
public import Proofs.Properties.Flock.Nets
public import Proofs.Flock.Claims.CanonEq
public import Proofs.Flock.Soundness.SparseRows
import Flock.HmRow
import Flock.Canon
import all Init.Data.Array.QSort.Basic
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

/-!
# The circuit file's facts

`HmRow.parse` reads the circuit file to `cG` at its digests (`parse_cG`), whatever their bytes: here they are the file's,
the fixed nets are `Properties.Flock.Nets.fixedNetsBuild`'s, the leaf scheme's canonical text is the pinned one's, and the
class hash is `classSha` (`classOf_eq`): the parser's reading of META prints, past its binding keys, as `metaPairs`'
binding-free members do (`canonUtf8_jsonEquiv`, `normS_jsonEquiv`), and the bytes it hashes after META are the file's
remaining lines (`extract_tail`). `HmOut.parse` then checks the rows (`HmOut.checkRows`), and `circuitFileFacts` reads the
fields off the parsed circuit; the unit net's reads come from its rows (`Net.ofRows_ok`, `SparseRows.ofRows_row`), of
which only rows 128 to 143 hold an entry below the input words, each a port bit. The reads a consumer takes past the
fields are stated as equations on `cR` (`cR_nets` and the others), so that no consumer unfolds `cR`.
-/

@[expose] public section

open Flock RecDeploy.VStarNets FlockVBridge.Names
open Properties.Flock.Nets (built sparseOf fixedNetsBuild)
open Lean (Json)
open FlockSoundness.Discharge.Recursion
open Std.DTreeMap.Internal (Impl)

set_option maxHeartbeats 100000000
set_option maxRecDepth 200000

namespace Verity.Sampled.Instance.HonestSession.CircuitFileProof

open Flock.Firewall SignedJson

theorem tagsZ_eq : tagsZ = tZ := by
  show Zk.tags Flock.Tags.circuit (Zk.maskLogOf circuitFile) 1 = tZ
  rw [maskLog_eq]
  rfl

/-! ## Integers only -/

mutual
theorem intJ_of_plainS : ∀ j : Json, plainS j = true → FlockDeploy.Concrete.IntJ j = true
  | .null, _ => rfl
  | .bool _, _ => rfl
  | .num ⟨_, _⟩, h => by
    simp only [plainS] at h
    exact h
  | .str _, _ => rfl
  | .arr ⟨a⟩, h => intJs_of_plainSList a h
  | .obj ⟨⟨t⟩⟩, h => by
    simp only [plainS, Bool.and_eq_true] at h
    exact intJo_of_plainSObj t h.1
theorem intJs_of_plainSList : ∀ a : List Json, plainSList a = true → FlockDeploy.Concrete.IntJs a = true
  | [], _ => rfl
  | j :: js, h => by
    simp only [plainSList, Bool.and_eq_true] at h
    show (FlockDeploy.Concrete.IntJ j && FlockDeploy.Concrete.IntJs js) = true
    simp [intJ_of_plainS j h.1, intJs_of_plainSList js h.2]
theorem intJo_of_plainSObj : ∀ t : Impl String (fun _ => Json), plainSObj t = true → FlockDeploy.Concrete.IntJo t = true
  | .leaf, _ => rfl
  | .inner _ _ v l r, h => by
    simp only [plainSObj, Bool.and_eq_true] at h
    show (FlockDeploy.Concrete.IntJo l && FlockDeploy.Concrete.IntJ v && FlockDeploy.Concrete.IntJo r) = true
    simp [intJo_of_plainSObj l h.1.1.1, intJ_of_plainS v h.1.2, intJo_of_plainSObj r h.2]
end

/-! ## `canonUtf8` prints equivalent values alike -/

theorem canonUtf8_jsonEquiv_lt : ∀ (k : Nat) (a b : Json), sizeOf a < k → FlockDeploy.JsonEquiv a b →
    canonUtf8 a = canonUtf8 b
  | 0, _, _, h, _ => absurd h (Nat.not_lt_zero _)
  | k + 1, a, b, hk, h => by
    match a, h with
    | .null, h => rw [FlockDeploy.jsonEquiv_null h]
    | .bool x, h => rw [FlockDeploy.jsonEquiv_bool h]
    | .num n, h => rw [FlockDeploy.jsonEquiv_num h]
    | .str s, h => rw [FlockDeploy.jsonEquiv_str h]
    | .arr a, h =>
      obtain ⟨b, rfl⟩ := FlockDeploy.jsonEquiv_arr h
      obtain ⟨hs, he⟩ := (FlockDeploy.jsonEquiv_arr_iff a b).1 h
      rw [canonUtf8, canonUtf8]
      congr 1
      rw [Array.mapM_eq_mapM_toList, Array.mapM_eq_mapM_toList]
      congr 1
      refine FlockDeploy.mapM_congr_getElem (by simpa using hs) fun i h₁ h₂ => ?_
      simp only [Array.toList_attach, List.getElem_attachWith]
      have hi₁ : i < a.size := by simpa using h₁
      have hi₂ : i < b.size := by simpa using h₂
      have hlt := Array.sizeOf_lt_of_mem (Array.getElem_mem hi₁)
      simp only [Json.arr.sizeOf_spec] at hk
      exact canonUtf8_jsonEquiv_lt k a[i] b[i] (by omega) (he i hi₁ hi₂)
    | .obj m, h =>
      obtain ⟨n, rfl⟩ := FlockDeploy.jsonEquiv_obj h
      obtain ⟨hl, he⟩ := FlockDeploy.jsonEquiv_obj_toList h
      rw [canonUtf8, canonUtf8]
      congr 1
      rw [Array.mapM_eq_mapM_toList, Array.mapM_eq_mapM_toList]
      congr 1
      refine FlockDeploy.mapM_congr_getElem (by simpa [Std.TreeMap.Raw.toList_toArray] using hl) fun i h₁ h₂ => ?_
      simp only [Array.toList_attach, List.getElem_attachWith]
      have hi₁ : i < m.toList.length := by simpa [Std.TreeMap.Raw.toList_toArray] using h₁
      have hi₂ : i < n.toList.length := by simpa [Std.TreeMap.Raw.toList_toArray] using h₂
      have hi₀ : i < m.toArray.toList.length := by simpa using h₁
      have hmem : m.toArray.toList[i] ∈ m.toArray := Array.mem_toList_iff.1 (List.getElem_mem hi₀)
      have hlt := sizeOf_lt_of_mem_toArray hmem
      obtain ⟨hk', hv⟩ := he i hi₁ hi₂
      simp only [Std.TreeMap.Raw.toList_toArray] at hlt ⊢
      rw [hk', canonUtf8_jsonEquiv_lt k _ _ (by omega) hv]

/-- **`canonUtf8` prints equivalent values alike.** -/
theorem canonUtf8_jsonEquiv {a b : Json} (h : FlockDeploy.JsonEquiv a b) : canonUtf8 a = canonUtf8 b :=
  canonUtf8_jsonEquiv_lt (sizeOf a + 1) a b (Nat.lt_succ_self _) h

/-! ## The parser's reading of a `plainS` value is equivalent to it -/

theorem fieldForms_toListModel (t : Impl String (fun _ => Json)) :
    FlockDeploy.fieldForms t = t.toListModel.map (fun p => (p.1, FlockDeploy.jsonForm p.2)) := by
  rw [FlockDeploy.fieldForms_eq, Impl.Const.toList_eq_toListModel_map, List.map_map]
  rfl

mutual
theorem normS_jsonEquiv : ∀ {j : Json}, plainS j = true →
    FlockDeploy.jsonForm (FirewallJson.norm j) = FlockDeploy.jsonForm j
  | .null, _ => rfl
  | .bool _, _ => rfl
  | .num _, _ => rfl
  | .str _, _ => rfl
  | .arr ⟨a⟩, h => by
    simp only [plainS] at h
    rw [FirewallJson.norm]
    simp only [FlockDeploy.jsonForm, FlockDeploy.jsonForms_eq]
    rw [normS_jsonEquivList a h]
  | .obj ⟨⟨t⟩⟩, h => by
    simp only [plainS, Bool.and_eq_true] at h
    have hasc : t.toListModel.Pairwise (fun a b => compare a.1 b.1 = .lt) := of_decide_eq_true h.2
    have hf : t.toListModel.foldl (fun a e => a.insert e.1 (FirewallJson.norm e.2)) (∅ : Std.TreeMap.Raw String Json) =
        (t.toListModel.map fun e => (⟨e.1, FirewallJson.norm e.2⟩ : (_ : String) × Json)).foldl
          (fun a e => a.insert e.1 e.2) ∅ := by
      rw [List.foldl_map]
    have h0 : (∅ : Std.TreeMap.Raw String Json).inner.inner.toListModel = [] := rfl
    rw [FirewallJson.norm]
    show FlockDeploy.jsonForm (.obj (FirewallJson.normObj t ∅)) = _
    rw [FlockDeploy.jsonForm, FlockDeploy.jsonForm, fieldForms_toListModel, fieldForms_toListModel,
      FirewallJson.normObj_eq, hf,
      FirewallJson.toListModel_foldl _ _ Impl.WF.empty (by rw [h0, List.nil_append, List.pairwise_map]; exact hasc), h0,
      List.nil_append, List.map_map]
    congr 1
    apply FlockDeploy.FieldForms.ofList_inj.2
    apply List.map_congr_left
    intro e he
    simp only [Function.comp]
    rw [normS_jsonEquivObj t h.1 e he]
theorem normS_jsonEquivList : ∀ (a : List Json), plainSList a = true →
    (FirewallJson.normList a).map FlockDeploy.jsonForm = a.map FlockDeploy.jsonForm
  | [], _ => rfl
  | j :: js, h => by
    simp only [plainSList, Bool.and_eq_true] at h
    simp only [FirewallJson.normList, List.map_cons, normS_jsonEquiv h.1, normS_jsonEquivList js h.2]
theorem normS_jsonEquivObj : ∀ (t : Impl String (fun _ => Json)), plainSObj t = true →
    ∀ e ∈ t.toListModel, FlockDeploy.jsonForm (FirewallJson.norm e.2) = FlockDeploy.jsonForm e.2
  | .leaf, _ => by simp [Impl.toListModel]
  | .inner _ k v l r, h => by
    simp only [plainSObj, Bool.and_eq_true] at h
    intro e he
    rw [Impl.toListModel_inner, List.mem_append, List.mem_cons] at he
    rcases he with he | rfl | he
    · exact normS_jsonEquivObj l h.1.1.1 e he
    · exact normS_jsonEquiv h.1.2
    · exact normS_jsonEquivObj r h.2 e he
end

/-- **Two objects built from members with distinct keys and one form, up to order,** have one form. -/
theorem jsonForm_mkObj_perm {L L' : List (String × Json)} (h : L.Pairwise fun a b => a.1 ≠ b.1)
    (h' : L'.Pairwise fun a b => a.1 ≠ b.1)
    (hp : (L.map fun e => (e.1, FlockDeploy.jsonForm e.2)).Perm (L'.map fun e => (e.1, FlockDeploy.jsonForm e.2))) :
    FlockDeploy.jsonForm (Json.mkObj L) = FlockDeploy.jsonForm (Json.mkObj L') := by
  rw [FirewallObj.mkObj_eq, FirewallObj.mkObj_eq]
  show FlockDeploy.JsonForm.obj (.ofList (FlockDeploy.fieldForms _)) =
    FlockDeploy.JsonForm.obj (.ofList (FlockDeploy.fieldForms _))
  rw [fieldForms_toListModel, fieldForms_toListModel]
  congr 2
  refine FirewallObj.eq_of_perm Prod.fst ?_ ?_ ?_
  · have e1 := (FirewallObj.perm_ofList h).map fun e : (_ : String) × Json => (e.1, FlockDeploy.jsonForm e.2)
    have e2 := (FirewallObj.perm_ofList h').map fun e : (_ : String) × Json => (e.1, FlockDeploy.jsonForm e.2)
    rw [List.map_map] at e1 e2
    exact e1.trans (hp.trans e2.symm)
  · exact FirewallObj.Asc.map (key := Prod.fst) _ (FirewallObj.asc_ordered (FirewallObj.ordered_ofList L))
  · exact FirewallObj.Asc.map (key := Prod.fst) _ (FirewallObj.asc_ordered (FirewallObj.ordered_ofList L'))

/-! ## The class hash -/

/-- META's members past its binding keys, as `classSha` prints them. -/
def metaX : Json := Json.mkObj (metaPairs.filter fun kv => !HmRow.BINDING_KEYS.contains kv.1)

def sX : String := (canonUtf8 metaX).toOption.getD ""

theorem metaPairs_ne : metaPairs.Pairwise fun a b => a.1 ≠ b.1 := by
  unfold metaPairs
  first | decide +kernel | simp

theorem plainSObj_meta : plainSObj (Impl.Const.ofList metaPairs : FirewallObj.Tree) = true := by
  have h := plainS_metaJ
  unfold metaJ at h
  rw [FirewallObj.mkObj_eq] at h
  simp only [plainS, Bool.and_eq_true] at h
  exact h.1

theorem plain_members : ∀ e ∈ metaPairs, plainStr e.1 = true ∧ plainS e.2 = true := fun e he =>
  (plainSObj_iff _).1 plainSObj_meta ⟨e.1, e.2⟩
    ((FirewallObj.perm_ofList metaPairs_ne).mem_iff.2 (List.mem_map.2 ⟨e, he, rfl⟩))

theorem canon_metaX : canonUtf8 metaX = .ok sX :=
  FlockDeploy.Concrete.canonUtf8_eq (intJ_of_plainS _ (plainS_mkObj (metaPairs_ne.sublist List.filter_sublist)
    fun e he => plain_members e (List.mem_of_mem_filter he)))

theorem classOf_eq_of {bytes : ByteArray} {head s : String} {mj : Json} {n1 n2 : Nat}
    {kvs : Std.TreeMap.Raw String Json} (h1 : findByte bytes 10 = some n1) (h2 : findByte bytes 10 (n1 + 1) = some n2)
    (hm : mj = .obj kvs)
    (hc : canonUtf8 (Json.mkObj (kvs.toArray.filter (fun kv => !HmRow.BINDING_KEYS.contains kv.1)).toList) = .ok s) :
    HmRow.classOf bytes head mj = .ok (Sha512.hash (head.toUTF8 ++ "\nMETA ".toUTF8 ++ s.toUTF8 ++ "\n".toUTF8 ++
      bytes.extract (n2 + 1) bytes.size)) := by
  subst hm
  unfold HmRow.classOf
  rw [h1]
  dsimp only
  rw [h2]
  dsimp only
  rw [hc]
  rfl

/-- The bytes after the META line's newline are the rest of the file. -/
theorem extract_tail (a b R : String) :
    (a ++ "\n" ++ (b ++ "\n" ++ R)).toUTF8.extract (a.toUTF8.size + 1 + b.toUTF8.size + 1)
      (a ++ "\n" ++ (b ++ "\n" ++ R)).toUTF8.size = R.toUTF8 := by
  have e : (a ++ "\n" ++ (b ++ "\n" ++ R)).toUTF8 = (a ++ "\n" ++ (b ++ "\n")).toUTF8 ++ R.toUTF8 := by
    simp only [toUTF8_app, ByteArray.append_assoc]
  have hn : "\n".toUTF8.size = 1 := by decide +kernel
  have hs : (a ++ "\n" ++ (b ++ "\n")).toUTF8.size = a.toUTF8.size + 1 + b.toUTF8.size + 1 := by
    simp only [toUTF8_app, ByteArray.size_append, hn] <;> omega
  rw [e, ← hs]
  exact ByteArray.extract_append_eq_right rfl ByteArray.size_append

theorem circuitFile_tail : circuitFile = (headLine ++ "\n" ++ (("META " ++ metaText) ++ "\n" ++
    joinNl ((circuitLines (HmNets.shaLaid ()) (HmNets.hmLaid ())).drop 2))).toUTF8 := by
  show (joinNl (["flock-circuit " ++ unitName, "META " ++ metaText] ++
    (circuitLines (HmNets.shaLaid ()) (HmNets.hmLaid ())).drop 2)).toUTF8 = _
  rw [List.cons_append, List.cons_append, List.nil_append, joinNl_cons, joinNl_cons]
  rfl

/-- **The class hash `HmRow.parse` computes for the circuit file is `classSha`.** -/
theorem classOf_eq : HmRow.classOf circuitFile ("flock-circuit " ++ unitName) (FirewallJson.norm metaJ) = .ok classSha := by
  obtain ⟨t', ht', hl'⟩ := FirewallObj.norm_obj (FirewallObj.ordered_ofList metaPairs)
  have hm : FirewallJson.norm metaJ = .obj ⟨⟨t'⟩⟩ := by
    show FirewallJson.norm (Json.mkObj metaPairs) = _
    rw [FirewallObj.mkObj_eq]
    exact ht'
  have hT := FirewallObj.perm_ofList metaPairs_ne
  have hmem := (plainSObj_iff _).1 plainSObj_meta
  have hL1 : ((⟨⟨t'⟩⟩ : Std.TreeMap.Raw String Json).toArray.filter
      (fun kv => !HmRow.BINDING_KEYS.contains kv.1)).toList =
      ((Impl.Const.ofList metaPairs : FirewallObj.Tree).toListModel.map
        fun e => (e.1, FirewallJson.norm e.2)).filter (fun kv => !HmRow.BINDING_KEYS.contains kv.1) := by
    rw [Array.toList_filter, FirewallObj.toArray_toList, hl', List.map_map]
    rfl
  have hc : canonUtf8 (Json.mkObj ((⟨⟨t'⟩⟩ : Std.TreeMap.Raw String Json).toArray.filter
      (fun kv => !HmRow.BINDING_KEYS.contains kv.1)).toList) = .ok sX := by
    rw [← canon_metaX, hL1]
    apply canonUtf8_jsonEquiv
    unfold metaX
    apply jsonForm_mkObj_perm
    · rw [List.filter_map, List.pairwise_map]
      exact (FirewallObj.Asc.ne (FirewallObj.asc_ordered (FirewallObj.ordered_ofList metaPairs))).sublist
        List.filter_sublist
    · exact metaPairs_ne.sublist List.filter_sublist
    · rw [List.filter_map, List.map_map]
      have e1 : ((List.filter ((fun kv : String × Json => !HmRow.BINDING_KEYS.contains kv.1) ∘
          fun e : (_ : String) × Json => (e.1, FirewallJson.norm e.2))
          (Impl.Const.ofList metaPairs : FirewallObj.Tree).toListModel).map
          ((fun e : String × Json => (e.1, FlockDeploy.jsonForm e.2)) ∘
            fun e : (_ : String) × Json => (e.1, FirewallJson.norm e.2))) =
          (List.filter ((fun kv : String × Json => !HmRow.BINDING_KEYS.contains kv.1) ∘
          fun e : (_ : String) × Json => (e.1, FirewallJson.norm e.2))
          (Impl.Const.ofList metaPairs : FirewallObj.Tree).toListModel).map
          (fun e : (_ : String) × Json => (e.1, FlockDeploy.jsonForm e.2)) := by
        apply List.map_congr_left
        intro e he
        show (e.1, FlockDeploy.jsonForm (FirewallJson.norm e.2)) = (e.1, FlockDeploy.jsonForm e.2)
        rw [normS_jsonEquiv (hmem e (List.mem_of_mem_filter he)).2]
      refine (List.Perm.of_eq e1).trans (((hT.filter _).map _).trans ?_)
      rw [List.filter_map, List.map_map]
      exact List.Perm.refl _
  rw [circuitFile_tail, classOf_eq_of (findByte_head _ _ _ headLine_noNl)
    (findByte_meta _ _ _ headLine_noNl metaLine_noNl) hm hc, extract_tail]
  rfl

/-! ## The leaf scheme's canonical text is the pinned one's -/

section
variable (bD bP bT bS : ByteArray)

def s1Z : String := (canon pinJZ).toOption.getD ""

theorem hc2Z : canon pinJZ = .ok s1Z := FlockDeploy.Concrete.canon_eq (j := pinJZ) (by decide +kernel)

theorem lsG_objM : lsJG bD bP bT bS = .obj (FlockDeploy.Concrete.objM (lsJG bD bP bT bS)) := by kernel_rfl'
theorem pinZ_objM : pinJZ = .obj (FlockDeploy.Concrete.objM pinJZ) := by kernel_rfl'
theorem lsG_arr : (FlockDeploy.Concrete.objM (lsJG bD bP bT bS)).toArray =
    (FlockDeploy.Concrete.objM pinJZ).toArray := by kernel_rfl'

theorem hc1Z : canon (lsJG bD bP bT bS) = .ok s1Z := by
  rw [lsG_objM, FlockDeploy.Concrete.canon_obj_congr (lsG_arr bD bP bT bS), ← pinZ_objM]
  exact hc2Z

end

/-! ## The parsed circuit -/

/-- **The circuit `HmOut.parse` reads the circuit file to**: `cG` at the file's digests, the verifier's fixed nets'
matrices and the pinned leaf scheme's canonical text. -/
noncomputable def cR : Flock.Circuit :=
  cG (HmNets.shaLaid ()) (HmNets.hmLaid ()) dummyBc partitionDigestB templateDigest programSha circuitSha classSha
    (sparseOf (HmNets.shaLaid ()).ra) (sparseOf (HmNets.shaLaid ()).rb) (sparseOf (HmNets.hmLaid ()).ra)
    (sparseOf (HmNets.hmLaid ()).rb) s1Z

theorem dummyBc_size : dummyBc.size = 128 := by
  have hs : yPort.segs.isSome = false := rfl
  unfold dummyBc HmOut.rowBcZeroSalt
  simp only [hs, Bool.false_eq_true, if_false, ↓reduceIte, ByteArray.size_append, Hex.size_hash]

/-- **`HmOut.parse` reads the circuit file to `cR`.** -/
theorem parse_eq : HmOut.parse tagsZ circuitFile [] = .ok cR := by
  have hp := FlockDeploy.fixedNets_parse fixedNetsBuild
  rw [tagsZ_eq, circuitFile_eqG]
  exact parse_cG (HmNets.shaLaid ()) (HmNets.hmLaid ()) dummyBc partitionDigestB templateDigest programSha circuitSha
    classSha (sparseOf (HmNets.shaLaid ()).ra) (sparseOf (HmNets.shaLaid ()).rb) (sparseOf (HmNets.hmLaid ()).ra)
    (sparseOf (HmNets.hmLaid ()).rb) s1Z rfl classOf_eq dummyBc_size (Hex.size_hash _) (Hex.size_hash _)
    (FlockDeploy.sha_lines fixedNetsBuild) (FlockDeploy.hm_lines fixedNetsBuild) hp.1.2 hp.2.2
    (hc1Z dummyBc partitionDigestB templateDigest programSha) hc2Z
    (fun c h1 h2 => FlockDeploy.Concrete.check_of hp.1.1 hp.2.1 c h1 h2)

theorem cH_eq : cH = cR := by
  unfold cH
  (rw [parse_eq]) <;> rfl

theorem tmpl_false : tagsZ.isTmpl = false := by rw [tagsZ_eq]; kernel_rfl'

/-- **`HmOut.parseCommit` reads the circuit file to `cR` too.** -/
theorem parseCommit_eq : HmOut.parseCommit tagsZ circuitFile [] = .ok cR := by
  unfold HmOut.parseCommit
  rw [tmpl_false]
  exact parse_eq

/-! ## The fields -/

theorem parse_ok : HmOut.parse tagsZ circuitFile [] = .ok cH := by rw [cH_eq]; exact parse_eq
theorem parseCommit_ok : HmOut.parseCommit tagsZ circuitFile [] = .ok cH := by rw [cH_eq]; exact parseCommit_eq
theorem hmIn_parse : HmIn.parse cH = .ok {} := by rw [cH_eq]; kernel_rfl'
theorem cH_sha : cH.sha = circuitSha := by rw [cH_eq]; kernel_rfl'
theorem cH_classSha : cH.classSha = classSha := by rw [cH_eq]; kernel_rfl'
theorem cH_ports : cH.ports = #[yPort] := by rw [cH_eq]; kernel_rfl'
theorem cH_outRowPorts : cH.outRowPorts = #[1] := by rw [cH_eq]; kernel_rfl'
theorem cH_g : cH.g = 1 := by rw [cH_eq]; kernel_rfl'
theorem cH_kLog : cH.kLog = 22 := by rw [cH_eq]; kernel_rfl'
theorem cH_metaJ : cH.metaJ = FirewallJson.norm metaJ := by rw [cH_eq]; kernel_rfl'
theorem cH_upv : cH.upv = 1 := by rw [cH_eq]; kernel_rfl'

theorem cR_dummyDigests : cR.dummyDigests = #[dummyBc] := rfl
theorem cH_dummyDigests : cH.dummyDigests = #[dummyBc] := by rw [cH_eq]; exact cR_dummyDigests
theorem cH_dummyOutputs : cH.dummyOutputs = #[0] := by rw [cH_eq]; kernel_rfl'
theorem cR_unit : cR.unit = UNH := by kernel_rfl'

theorem unit_inGroups : cH.unit.inGroups.size = 1 := by rw [cH_eq, cR_unit]; rfl
theorem unit_outNet : cH.outNet = cH.unitNet := by rw [cH_eq]; kernel_rfl'
theorem unit_unitLog : cH.unit.unitLog ≤ 32 := by rw [cH_eq, cR_unit]; decide

/-! ## The unit net reads only port bits -/

theorem portBits16 : ∀ y < 16, y ∈ FlockSoundness.portBits #[{ col := 0, words := 1, bits := #[16] }] := by
  decide

theorem raRow (i : Nat) (h : i < 257) :
    FlockDeploy.Concrete.unitLaid.ra.getD i #[] = FlockDeploy.Concrete.unitRow true i := by
  simp [FlockDeploy.Concrete.unitLaid, Array.getD, h]

theorem rbRow (i : Nat) (h : i < 257) :
    FlockDeploy.Concrete.unitLaid.rb.getD i #[] = FlockDeploy.Concrete.unitRow false i := by
  simp [FlockDeploy.Concrete.unitLaid, Array.getD, h]

theorem rowT (i : Nat) (hi : i < 16) : FlockDeploy.Concrete.unitRow true (128 + i) = #[i] := by
  unfold FlockDeploy.Concrete.unitRow
  rw [if_neg (by omega), if_pos (by simp only [Bool.and_eq_true, decide_eq_true_eq]; omega), if_pos rfl]
  rw [Nat.add_sub_cancel_left]

theorem rowF (i : Nat) (hi : i < 16) : FlockDeploy.Concrete.unitRow false (128 + i) = #[256] := by
  unfold FlockDeploy.Concrete.unitRow
  rw [if_neg (by omega), if_pos (by simp only [Bool.and_eq_true, decide_eq_true_eq]; omega)]
  rfl

theorem rowE (b : Bool) (i : Nat) (h1 : 16 ≤ i) (h2 : i < 128) : FlockDeploy.Concrete.unitRow b (128 + i) = #[] := by
  unfold FlockDeploy.Concrete.unitRow
  rw [if_neg (by omega), if_neg (by simp only [Bool.and_eq_true, decide_eq_true_eq]; omega),
    if_neg (by simp only [beq_iff_eq]; omega)]

theorem unitRows_portReads : ((List.range (256 - min (1 * Flock.WORD) 256)).all fun i =>
      ((FlockDeploy.Concrete.unitLaid.ra.getD (min (1 * Flock.WORD) 256 + i) #[]).toList ++
            (FlockDeploy.Concrete.unitLaid.rb.getD (min (1 * Flock.WORD) 256 + i) #[]).toList).all
        fun x => decide (x < min (1 * Flock.WORD) 256 → x ∈ FlockSoundness.portBits #[{ col := 0, words := 1, bits := #[16] }])) =
    true := by
  have hW : min (1 * Flock.WORD) 256 = 128 := by decide
  simp only [hW, List.all_eq_true, List.mem_range, decide_eq_true_eq]
  intro i hi x hx hx128
  rw [raRow _ (by omega), rbRow _ (by omega)] at hx
  by_cases h : i < 16
  · rw [rowT i h, rowF i h] at hx
    simp only [List.mem_append, List.toList_toArray, List.mem_singleton] at hx
    apply portBits16
    omega
  · rw [rowE true i (by omega) hi, rowE false i (by omega) hi] at hx
    simp at hx

theorem portReads_UNH : FlockSoundness.Discharge.Layout.portReadsB UNH = true := by
  have hn := Net.ofRows_ok unit_netOfH
  have hw : 2 ^ UNH.unitLog ≤ 2 ^ 32 := by show 2 ^ 9 ≤ 2 ^ 32; decide
  have ha := FlockSoundness.SparseRows.ofRows_row hn.2.1 hw
  have hb := FlockSoundness.SparseRows.ofRows_row hn.2.2.1 hw
  unfold FlockSoundness.Discharge.Layout.portReadsB
  simp only [ha, hb]
  have e1 : UNH.constPos = 256 := rfl
  have e2 : UNH.inWords = 1 := rfl
  have e3 : UNH.inGroups = #[⟨0, 1, #[16]⟩] := rfl
  simp only [e1, e2, e3]
  exact unitRows_portReads

theorem unit_portReads : FlockSoundness.Discharge.Layout.portReadsB cH.unit = true := by
  rw [cH_eq, cR_unit]
  exact portReads_UNH

/-- **The circuit file's facts.** -/
theorem circuitFileFacts : CircuitFileFacts :=
  ⟨maskLog_eq, parse_ok, parseCommit_ok, hmIn_parse, cH_sha, cH_classSha, cH_ports, cH_outRowPorts, cH_g, cH_kLog,
    cH_metaJ, cH_upv, cH_dummyDigests, cH_dummyOutputs, unit_inGroups, unit_outNet, unit_unitLog, unit_portReads⟩

/-! ## Reads of `cR` past the fields

Each by the kernel on `cR`'s own terms: the nets stay `sparseOf` of the verifier's laid nets, never evaluated. -/

theorem cR_nets : cR.nets = #[("unit", UNH),
    ("sha512x3", shaB (sparseOf (HmNets.shaLaid ()).ra) (sparseOf (HmNets.shaLaid ()).rb)),
    ("hm96", hmB (sparseOf (HmNets.hmLaid ()).ra) (sparseOf (HmNets.hmLaid ()).rb))] := by kernel_rfl'
theorem cR_unitNet : cR.unitNet = 0 := by kernel_rfl'
theorem cR_shaNet : cR.shaNet = 1 := by kernel_rfl'
theorem cR_hmNet : cR.hmNet = 2 := by kernel_rfl'
theorem cR_outNet : cR.outNet = 0 := by kernel_rfl'
theorem cR_leafScheme : cR.leafScheme = "hm96-sha512/v1" := by kernel_rfl'
theorem cR_outNetPorts : cR.outNetPorts = #[] := by kernel_rfl'
theorem cR_outNetSlots : cR.outNetSlots = none := by kernel_rfl'
theorem cR_leafCuts : cR.leafCuts = #[] := by kernel_rfl'
theorem cR_typed : cR.typed = none := by kernel_rfl'
theorem cR_cls : cR.cls = none := by kernel_rfl'
theorem cR_tmpl : cR.tmpl = none := by kernel_rfl'
theorem cR_key : cR.key = ByteArray.empty := by kernel_rfl'
theorem cR_outPorts : cR.outPorts = #[("y", 1)] := by kernel_rfl'
theorem cR_leavesIn : cR.leavesIn = #[#[-1]] := by kernel_rfl'
theorem cR_leavesOut : cR.leavesOut = #[#[0]] := by kernel_rfl'
theorem cR_perVu : cR.perVu = #[1, 1, 1] := by kernel_rfl'
theorem cR_wires : cR.wires = #[⟨(2, 0, 0), (1, 0, 0)⟩] := by kernel_rfl'
/-- META's ranges in its order: hm96, sha512x3, the unit, the mask. -/
theorem cR_ranges : cR.ranges = #[
    { kind := .net 2, slotLog := 18, pos0 := 0, count := 1, role := .none, packed := false },
    { kind := .net 1, slotLog := 18, pos0 := 1, count := 1, role := .none, packed := false },
    { kind := .net 0, slotLog := 9, pos0 := 1024, count := 1, role := .none, packed := false },
    { kind := .mask, slotLog := 14, pos0 := 64, count := 1, role := .none, packed := false }] := by kernel_rfl'

end Verity.Sampled.Instance.HonestSession.CircuitFileProof
