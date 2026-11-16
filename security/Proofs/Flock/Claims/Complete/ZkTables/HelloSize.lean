module

public import Proofs.Flock.Claims.Complete.ZkTables.RecordText
public import Proofs.Flock.Claims.Complete.Json.SizeSame
import Proofs.Flock.Claims.Complete.ZkTables.Stmt
import Proofs.Flock.Claims.Complete.Json.CanonSize
import all Lean.Data.Json.Basic

/-!
# The prover's `Hello` is short where the verifier's is

`hello_hz_le`: the prover's `Hello` (`canon (helloJ own)`, `own` the verifier's parsed) has at most `jsz` of the
verifier's object (the list it prints), of its `coins`, and `1344` characters (`jsz_same`, `jsz_insert_le`,
`canon_size_le`). `helloList_sum_le`: that list weighs a coin tree's, the table names' and at most `6000`, at a
`64`-byte `Σ` and `m_pts ≤ 64`.
-/

@[expose] public section

namespace FlockComplete.ZkTables

open Flock Transcript FlockComplete FlockDeploy JsonRT
open Lean (Json)

/-- **The prover's `Hello` is the verifier's object, parsed, with a nonce put in**: its length is at most the weight of
the list the verifier's `Hello` prints, of its `coins`, and `1344`. -/
theorem hello_hz_le {ct : CoinTree.Spec} {L : List (String × Json)} {hello : String}
    (hh : canon (Json.mkObj L) = .ok hello) (hd : Distinct L) (hg : ∀ p ∈ L, Good p.2)
    (hc : ("coins", ct.coins "") ∈ L) {own : Json} (hown : Flock.JsonParse.parse hello = .ok own) {hz : String}
    (hzc : canon (helloJ own) = .ok hz) :
    hz.length ≤ (L.map fun p => 18 * p.1.length + 4 + jsz p.2).sum + jsz (ct.coins "") + 1344 := by
  have hgj := good_mkObj hd hg
  obtain ⟨j', hp, hs⟩ := FlockComplete.JsonRT.parse_canon _ _ hgj hh
  rw [hp] at hown
  have hjo : j' = own := Except.ok.inj hown
  subst hjo
  have hjs := jsz_same hs
  have hmk := jsz_mkObj_le hd
  obtain ⟨m, rfl, hm, -⟩ := same_obj_wf hs
  obtain ⟨cv, hcv, hcs⟩ := hs.getObjVal (getObjVal?_mkObj hd hc)
  have hco : ∃ M : Std.TreeMap.Raw String Json compare, ct.coins "" = .obj M := ⟨_, rfl⟩
  obtain ⟨M, hM⟩ := hco
  rw [hM] at hcs
  obtain ⟨kvs, rfl, hkv, -⟩ := same_obj_wf hcs
  have hmg : m.get? "coins" = some (.obj kvs) := by
    simp only [Json.getObjVal?] at hcv
    split at hcv
    · cases hcv
      assumption
    · cases hcv
  have hck : jsz (.obj kvs) = jsz (ct.coins "") := by
    rw [hM]
    exact (jsz_same hcs).symm
  have e : helloJ (.obj m) = .obj (m.insert "coins" (.obj (kvs.insert "nonce" (Json.str (zeros 64))))) := by
    simp only [helloJ, Json.getObjVal?, hmg, Json.setObjVal!]
    rfl
  have h1 := jsz_insert_le hm "coins" (.obj (kvs.insert "nonce" (Json.str (zeros 64))))
  have h2 := jsz_insert_le hkv "nonce" (Json.str (zeros 64))
  have h3 := jsz_zeros 64
  have h4 := canon_size_le _ hzc
  have h5 := length_le_size hz
  rw [e] at h4
  have hc5 : ("coins" : String).length = 5 := by decide +kernel
  have hn5 : ("nonce" : String).length = 5 := by decide +kernel
  omega

/-- The list a `Hello` at a keyed coin tree prints (`Flock.helloOf`). -/
def helloList (tags : Tags) (sigma : ByteArray) (mPts : ℕ) (ct : CoinTree.Spec) (names : Array String) :
    List (String × Json) :=
  let profile := if tags.wideLeaf then Tags.WIDE_LEAF_PROFILE else "fast100"
  let base := [("profile", Json.str profile), ("reps", Lean.toJson (2 : Nat)), ("flavor", Json.str "rs"),
    ("tables", Json.arr (names.map Json.str)),
    ("link", Json.mkObj [("sigma", Json.str (hexOf sigma)), ("m", Lean.toJson mPts), ("points", Lean.toJson (2 : Nat))])]
  let base := base ++ [("coins", ct.coins "")]
  let base := if tags.hm96Rows then base ++ [("digests", Json.str "sha512")] else base
  if tags.retainRounds then base ++ [("rounds", Json.str "bytes retained")] else base

theorem helloOf_tree (tags : Tags) (sigma : ByteArray) (mPts : ℕ) (ct : CoinTree.Spec) (names : Array String) :
    Flock.helloOf tags sigma mPts (.tree ct) names = canon (Json.mkObj (helloList tags sigma mPts ct names)) := rfl

theorem link_jsz_le (sigma : ByteArray) (hs : sigma.size = 64) (mPts : ℕ) (hm : mPts ≤ 64) :
    jsz (Json.mkObj [("sigma", Json.str (hexOf sigma)), ("m", Lean.toJson mPts), ("points", Lean.toJson (2 : Nat))]) ≤
      2960 := by
  have h := jsz_mkObj_le (l := [("sigma", Json.str (hexOf sigma)), ("m", Lean.toJson mPts),
    ("points", Lean.toJson (2 : Nat))]) (by first | decide +kernel | simp)
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil] at h
  have h1 := jsz_toJson_gen mPts
  have h2 := jsz_toJson_gen 2
  have h3 : (hexOf sigma).length = 128 := by rw [hexOf_len, hs]
  rw [jsz_str] at h
  have e1 : ("sigma" : String).length = 5 := by decide +kernel
  have e2 : ("m" : String).length = 1 := by decide +kernel
  have e3 : ("points" : String).length = 6 := by decide +kernel
  omega

theorem helloList_sum_le (tags : Tags) (sigma : ByteArray) (hs : sigma.size = 64) (mPts : ℕ) (hm : mPts ≤ 64)
    (ct : CoinTree.Spec) (names : Array String) :
    ((helloList tags sigma mPts ct names).map fun p => 18 * p.1.length + 4 + jsz p.2).sum ≤
      jsz (Json.arr (names.map Json.str)) + jsz (ct.coins "") + 6000 := by
  have hl := link_jsz_le sigma hs mPts hm
  have hp : (if tags.wideLeaf then Tags.WIDE_LEAF_PROFILE else "fast100").length ≤ 64 := by
    split
    · decide +kernel
    · decide +kernel
  have hpj := jsz_str (if tags.wideLeaf then Tags.WIDE_LEAF_PROFILE else "fast100")
  have h2 := jsz_toJson_gen 2
  have e0 : ("profile" : String).length = 7 := by decide +kernel
  have e1 : ("reps" : String).length = 4 := by decide +kernel
  have e2 : ("flavor" : String).length = 6 := by decide +kernel
  have e3 : ("tables" : String).length = 6 := by decide +kernel
  have e4 : ("link" : String).length = 4 := by decide +kernel
  have e5 : ("coins" : String).length = 5 := by decide +kernel
  have e6 : ("digests" : String).length = 7 := by decide +kernel
  have e7 : ("rounds" : String).length = 6 := by decide +kernel
  have j1 : jsz (Json.str "rs") = 38 := by decide +kernel
  have j2 : jsz (Json.str "sha512") = 110 := by decide +kernel
  have j3 : jsz (Json.str "bytes retained") = 254 := by decide +kernel
  unfold helloList
  by_cases h9 : tags.hm96Rows = true <;> by_cases hr : tags.retainRounds = true <;>
    simp only [h9, hr, ↓reduceIte, List.map_cons, List.map_nil, List.map_append, List.sum_cons, List.sum_nil,
      List.sum_append, List.cons_append, List.nil_append] <;>
    omega

theorem helloList_distinct (tags : Tags) (sigma : ByteArray) (mPts : ℕ) (ct : CoinTree.Spec) (names : Array String) :
    Distinct (helloList tags sigma mPts ct names) := by
  unfold helloList
  by_cases h9 : tags.hm96Rows = true <;> by_cases hr : tags.retainRounds = true <;>
    simp only [h9, hr, ↓reduceIte, List.cons_append, List.nil_append] <;>
    first | decide +kernel | simp

theorem coins_mem_helloList (tags : Tags) (sigma : ByteArray) (mPts : ℕ) (ct : CoinTree.Spec)
    (names : Array String) : ("coins", ct.coins "") ∈ helloList tags sigma mPts ct names := by
  unfold helloList
  by_cases h9 : tags.hm96Rows = true <;> by_cases hr : tags.retainRounds = true <;>
    simp [h9, hr]

theorem helloList_good (tags : Tags) (sigma : ByteArray) (mPts : ℕ) (ct : CoinTree.Spec) (names : Array String) :
    ∀ p ∈ helloList tags sigma mPts ct names, Good p.2 := by
  have hlink : Good (Json.mkObj [("sigma", Json.str (hexOf sigma)), ("m", Lean.toJson mPts),
      ("points", Lean.toJson (2 : Nat))]) := by
    refine good_mkObj (by first | decide +kernel | simp) ?_
    simp only [List.mem_cons, List.mem_nil_iff, or_false, forall_eq_or_imp, forall_eq]
    exact ⟨.str _, good_toJson _, good_toJson _⟩
  have harr : Good (Json.arr (names.map Json.str)) := good_arr_of fun x hx => by
    obtain ⟨s, -, rfl⟩ := Array.mem_map.1 hx
    exact .str _
  unfold helloList
  by_cases h9 : tags.hm96Rows = true <;> by_cases hr : tags.retainRounds = true <;>
    simp only [h9, hr, ↓reduceIte, List.cons_append, List.nil_append, List.mem_cons, List.mem_nil_iff, or_false,
      forall_eq_or_imp, forall_eq] <;>
    first
      | exact ⟨.str _, good_toJson _, .str _, harr, hlink, good_coins ct "", .str _, .str _⟩
      | exact ⟨.str _, good_toJson _, .str _, harr, hlink, good_coins ct "", .str _⟩
      | exact ⟨.str _, good_toJson _, .str _, harr, hlink, good_coins ct ""⟩

/-- **The prover's `Hello` is short** where the verifier's names a keyed coin tree, a `64`-byte `Σ` and `m_pts ≤ 64`: at
most the weights of the table names and, twice, of the coin tree's `coins`, and `7346`. -/
theorem helloLen_le {tags : Tags} {sigma : ByteArray} {mPts : ℕ} {ct : CoinTree.Spec} {names : Array String}
    {hello : String} (hh : Flock.helloOf tags sigma mPts (.tree ct) names = .ok hello) (hs : sigma.size = 64)
    (hm : mPts ≤ 64) {spec : SessionSpec} (hspec : spec.hello = hello) {ct' : CoinTree.Spec}
    (hct : spec.coinTree = some ct') :
    helloLen spec ≤ jsz (Json.arr (names.map Json.str)) + 2 * jsz (ct.coins "") + 7346 := by
  obtain ⟨m, kvs, hown, -, -, -, hg⟩ := hello_ok hh
  obtain ⟨hz, hc, -⟩ := FlockComplete.JsonRT.canon_ok _ hg
  have hown' : Flock.JsonParse.parse spec.hello = .ok (.obj m) := by rw [hspec]; exact hown
  have e : helloLen spec = hz.length := by simp only [helloLen, hct, hown', Except.toOption, hc]
  rw [helloOf_tree] at hh
  have h1 := hello_hz_le (ct := ct) hh (helloList_distinct tags sigma mPts ct names)
    (helloList_good tags sigma mPts ct names) (coins_mem_helloList tags sigma mPts ct names) hown hc
  have h2 := helloList_sum_le tags sigma hs mPts hm ct names
  omega

end FlockComplete.ZkTables
