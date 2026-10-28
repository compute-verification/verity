import Proofs.Flock.Claims.Complete.Json
import Proofs.Flock.Claims.Complete.Zk.Json.QSort
import Proofs.Flock.Claims.Complete.Json.Canon

/-!
# The honest masked record as a JSON value, and its S2

The masked session's record is the plain one's with the keyed coin tree's S2: its `hello` is the verifier's `Hello` with
the honest prover's nonce (`64` zeros) in `coins` (`helloJ`), and it carries the `coin_tree` of a zero key, nonce and
root and the statement's spec (`coinTreeJ`). So it is `recordList` at the spec whose `Hello` is the prover's and which
names no coin tree (`specZ`), with `coin_tree` added (`recordObjZk`): `recordJson` through `Hashes.ofSha512 h` returns
its `printJ` (`recordJson_eqZk`).

* **R7 with the prover's nonce** (`checkHello_ok`): the prover's `Hello` reads back with its nonce, which is `64`
  lowercase hex digits, and is the verifier's `Hello` with that nonce put in.
* **The record's `coin_tree`** (`checkRecord_ok`): its keys sort to `key, nonce, root, spec` (`qsort_coinTree`, the
  sort run on the four names), its nonce `Hello`'s, its key and root hex of their lengths with bit 2047 clear, and its
  spec the statement's.
* **S2 passed, the masked record decodes as the plain one** at `specZ` (`decodeWith_specZ`).
-/

namespace FlockComplete.JsonRT

open Flock Transcript FlockComplete
open Lean (Json JsonNumber)

/-! ## The masked record -/

/-- The verifier's `Hello` with the honest prover's nonce, `64` zeros, in `coins` (`helloOf`). -/
def helloJ (own : Json) : Json :=
  own.setObjVal! "coins" (((own.getObjVal? "coins").toOption.getD Json.null).setObjVal! "nonce" (Json.str (zeros 64)))

/-- The honest record's `coin_tree`: a zero key, nonce and root, and the statement's spec. -/
def coinTreeJ (ct : CoinTree.Spec) : Json :=
  Json.mkObj [("key", Json.str (zeros 512)), ("nonce", Json.str (zeros 64)), ("root", Json.str (zeros 128)),
    ("spec", ct.toJson)]

/-- The session spec the masked record reads as once S2 passes: its `Hello` the prover's `hz`, no keyed coin tree. -/
def specZ (spec : SessionSpec) (hz : String) : SessionSpec := { spec with hello := hz, coinTree := none }

/-- **The honest masked session record**: the plain record at `specZ`, with the keyed coin tree's `coin_tree`. -/
def recordListZk (spec : SessionSpec) (h : ByteArray → ByteArray) (hz : String) (ct : CoinTree.Spec)
    (rootB : Array ByteArray) (points : Array (Array F128)) (streams : Array (Array Round))
    (proofs : Array ByteArray) : List (String × Json) :=
  recordList (specZ spec hz) h rootB points streams proofs ++ [("coin_tree", coinTreeJ ct)]

def recordObjZk (spec : SessionSpec) (h : ByteArray → ByteArray) (hz : String) (ct : CoinTree.Spec)
    (rootB : Array ByteArray) (points : Array (Array F128)) (streams : Array (Array Round))
    (proofs : Array ByteArray) : Json :=
  Json.mkObj (recordListZk spec h hz ct rootB points streams proofs)

theorem helloOf_some {spec : SessionSpec} {ct : CoinTree.Spec} (hct : spec.coinTree = some ct) {own : Json}
    (hown : Flock.JsonParse.parse spec.hello = .ok own) {hz : String} (hc : canon (helloJ own) = .ok hz) :
    helloOf spec = .ok (hz, [("coin_tree", coinTreeJ ct)]) := by
  unfold helloOf
  unfold helloJ at hc
  simp only [hct, hown, Except.toOption] at hc ⊢
  rw [hc]
  rfl

/-- **The masked session's record JSON is `printJ` of `recordObjZk`**, the verifier's `Hello` parsing and its
prover's version printing. -/
theorem recordJson_eqZk (h : ByteArray → ByteArray) {spec : SessionSpec} {ct : CoinTree.Spec}
    (hct : spec.coinTree = some ct) {own : Json} (hown : Flock.JsonParse.parse spec.hello = .ok own) {hz : String}
    (hc : canon (helloJ own) = .ok hz) (rootB : Array ByteArray) (points : Array (Array F128))
    (streams : Array (Array Round)) (proofs : Array ByteArray) :
    recordJson (Hashes.ofSha512 h : Hashes PS) spec rootB points streams proofs =
      pure (printJ (recordObjZk spec h hz ct rootB points streams proofs)) := by
  unfold recordJson roundJ
  have hl : (monadLift (helloOf spec) : PS _) = pure (hz, [("coin_tree", coinTreeJ ct)]) := by
    rw [helloOf_some hct hown hc]
    rfl
  simp only [hl, dgWith_pure, pure_bind, List.mapM_pure, Array.mapM_pure, Array.forIn_pure_yield_eq_foldl,
    foldl_push_map]
  unfold recordObjZk recordListZk recordList linkObj linkList rootBList streamObj streamList roundObj proofObj
    proofList lbOf yOf specZ
  rfl

/-! ## The masked record is printable -/

theorem coinTreeJ_list_distinct (ct : CoinTree.Spec) :
    Distinct [("key", Json.str (zeros 512)), ("nonce", Json.str (zeros 64)), ("root", Json.str (zeros 128)),
      ("spec", ct.toJson)] := by
  simp

theorem good_specJson (ct : CoinTree.Spec) : Good ct.toJson := by
  refine good_mkObj (by simp) ?_
  simp only [List.mem_cons, List.mem_nil_iff, or_false, forall_eq_or_imp, forall_eq]
  refine ⟨good_toJson _, good_toJson _, .str _, good_arr_of fun x hx => ?_⟩
  obtain ⟨s, -, rfl⟩ := Array.mem_map.1 hx
  exact .str _

theorem good_coinTreeJ (ct : CoinTree.Spec) : Good (coinTreeJ ct) := by
  refine good_mkObj (coinTreeJ_list_distinct ct) ?_
  simp only [List.mem_cons, List.mem_nil_iff, or_false, forall_eq_or_imp, forall_eq]
  exact ⟨.str _, .str _, .str _, good_specJson ct⟩

/-- **The coin tree's `coins` entry is ASCII only if its spec and its `nonce` key are.** -/
theorem ascii_coins {ct : CoinTree.Spec} (h : Ascii (ct.coins "")) :
    Ascii ct.toJson ∧ ∀ c ∈ "nonce".toList, c.toNat < 0x7f := by
  have hg := good_specJson ct
  unfold CoinTree.Spec.coins at h
  unfold CoinTree.Spec.toJson at hg h ⊢
  cases hg with
  | obj hm _ =>
  refine ⟨ascii_of_insert (k := "nonce") (v := Json.str "") hm
    (get?_mkObj_none (by first | (simp only [List.map_cons, List.map_nil]; decide) | simp)) h, ?_⟩
  exact ascii_key (k := "nonce") (v := Json.str "") hm.insert h
    (by rw [Std.TreeMap.Raw.get?_eq_getElem?, Std.TreeMap.Raw.getElem?_insert_self hm])

theorem ascii_zeros (n : Nat) : ∀ c ∈ (zeros n).toList, c.toNat < 0x7f := by
  unfold zeros
  rw [String.toList_ofList]
  intro c hc
  rw [List.eq_of_mem_replicate hc]
  decide

theorem recordListZk_distinct (spec : SessionSpec) (h : ByteArray → ByteArray) (hz : String) (ct : CoinTree.Spec)
    (rootB : Array ByteArray) (points : Array (Array F128)) (streams : Array (Array Round))
    (proofs : Array ByteArray) : Distinct (recordListZk spec h hz ct rootB points streams proofs) := by
  rcases dk_cases (specZ spec hz) with e | e <;> simp [recordListZk, recordList, e, toString]

/-- **The masked record is printable** when the statement's draw is. -/
theorem good_recordObjZk {spec : SessionSpec} {t : String} (ht : spec.tables = #[t]) (h : ByteArray → ByteArray)
    (hz : String) (ct : CoinTree.Spec) (rb : ByteArray) (points : Array (Array F128)) (streams : Array (Array Round))
    (proofs : Array ByteArray) (hd : Good (spec.unitDraw.getD Json.null)) :
    Good (recordObjZk spec h hz ct #[rb] points streams proofs) := by
  have ht' : (specZ spec hz).tables = #[t] := ht
  refine good_mkObj (recordListZk_distinct spec h hz ct _ points streams proofs) ?_
  simp only [recordListZk, recordList, List.cons_append, List.nil_append, List.mem_cons, List.mem_nil_iff, or_false,
    forall_eq_or_imp, forall_eq]
  refine ⟨.str _, .null, .str _, good_mkObj (by simp) (by simp; exact .bool true), .str _,
    good_linkObj ht' h rb points, hd, good_arr_of fun x hx => ?_, good_hexJ _, good_arr_of fun x hx => ?_,
    good_coinTreeJ ct⟩
  · obtain ⟨q, -, rfl⟩ := Array.mem_map.1 hx
    exact good_streamObj _ _ _ _
  · obtain ⟨q, -, rfl⟩ := Array.mem_map.1 hx
    exact good_proofObj _ _ _

/-! ## S2's checks on the masked record -/

theorem isHex_zeros (n : Nat) : CoinTree.isHex n (zeros n) = true := by
  unfold CoinTree.isHex zeros
  rw [String.length_ofList, String.all_bool_eq, String.toList_ofList]
  simp only [List.length_replicate, beq_self_eq_true, List.all_replicate, Bool.true_and]
  split <;> decide

theorem zeros_bit (n : Nat) (hn : 510 < n) : "01234567".contains ((zeros n).toList.getD 510 'f') = true := by
  unfold zeros
  rw [String.toList_ofList, String.contains_char_eq, List.getD_eq_getElem?_getD, List.getElem?_replicate]
  simp only [hn, ite_true, Option.getD_some]
  decide

theorem isSome_get?_mkObj {l : List (String × Json)} (hd : Distinct l) (k : String) :
    ((Std.TreeMap.Raw.ofList l compare).get? k).isSome = (l.map Prod.fst).contains k := by
  cases hc : (l.map Prod.fst).contains k with
  | false => rw [get?_mkObj_none hc]; rfl
  | true =>
    obtain ⟨p, hp, hpk⟩ := List.mem_map.1 (List.contains_iff_mem.1 hc)
    rw [get?_mkObj hd (k := k) (v := p.2) (by rw [← hpk]; exact hp)]
    rfl

/-- A well-formed object whose keys are those of a sorted list has that list as its keys. -/
theorem keys_of {n : Std.TreeMap.Raw String Json compare} (hn : n.WF) {L : List String}
    (hL : L.Pairwise fun a b => compare a b = .lt) (hk : ∀ k, (n.get? k).isSome = L.contains k) : n.keys = L := by
  have : Std.Irrefl (fun a b : String => compare a b = .lt) := ⟨fun a h => by simp at h⟩
  have : Std.Antisymm (fun a b : String => compare a b = .lt) := ⟨fun a b h1 h2 => by
    rw [Std.OrientedCmp.eq_swap (cmp := compare)] at h2
    rw [h1] at h2
    cases h2⟩
  refine List.Pairwise.eq_of_mem_iff (Std.TreeMap.Raw.ordered_keys hn) hL fun a => ?_
  rw [Std.TreeMap.Raw.mem_keys hn, ← Std.TreeMap.Raw.contains_iff_mem,
    Std.TreeMap.Raw.contains_eq_isSome_getElem? hn, ← Std.TreeMap.Raw.get?_eq_getElem?, hk, List.contains_iff_mem]

theorem same_obj_wf {m : Std.TreeMap.Raw String Json compare} {j : Json} (h : Same (.obj m) j) :
    ∃ n, j = .obj n ∧ n.WF ∧ ∀ k, (m.get? k).isSome = (n.get? k).isSome := by
  cases h with
  | obj _ hn hk _ => exact ⟨_, rfl, hn, hk⟩

/-- **R7 with the prover's nonce**: the prover's `Hello`, the verifier's own with `64` zeros put in `coins.nonce`,
passes, its nonce those zeros. -/
theorem checkHello_ok {m kvs : Std.TreeMap.Raw String Json compare}
    (hm : m.WF) (hkv : kvs.WF) (hco : m.get? "coins" = some (.obj kvs)) (hg : Good (helloJ (.obj m))) {hz : String}
    (hc : canon (helloJ (.obj m)) = .ok hz) : CoinTree.checkHello (.obj m) hz = .ok (zeros 64) := by
  obtain ⟨j, hp, hs⟩ := FlockComplete.JsonRT.parse_canon _ _ hg hc
  have hcoins : (helloJ (.obj m)).getObjVal? "coins" =
      .ok (Json.obj (kvs.insert "nonce" (Json.str (zeros 64)))) := by
    simp only [helloJ, Json.getObjVal?, hco, Json.setObjVal!, Std.TreeMap.Raw.get?_eq_getElem?,
      Std.TreeMap.Raw.getElem?_insert_self hm]
    rfl
  obtain ⟨cv, hcv, hcvs⟩ := hs.getObjVal hcoins
  have hn : cv.getObjValAs? String "nonce" = .ok (zeros 64) := strAs_same hcvs (by
    simp only [Json.getObjVal?, Std.TreeMap.Raw.get?_eq_getElem?, Std.TreeMap.Raw.getElem?_insert_self hkv]
    rfl)
  have hh : ((Json.obj m).setObjVal! "coins" ((((Json.obj m).getObjVal? "coins").toOption.getD Json.null).setObjVal!
      "nonce" (Json.str (zeros 64)))) = helloJ (.obj m) := rfl
  unfold CoinTree.checkHello
  simp only [hp, Except.toOption, hcv, Option.bind, hn, Option.getD_some, isHex_zeros, Bool.not_true,
    Bool.false_eq_true, ite_false]
  simp only [Except.toOption] at hh
  rw [hh, hc]
  simp only [ok_bind, bne_self_eq_false, Bool.false_eq_true, ite_false]
  rfl

/-- **The record's `coin_tree`** passes: its keys are `key, nonce, root, spec`, its nonce `Hello`'s, its key and root
hex of their lengths with bit 2047 clear, and its spec the statement's, which `canon` prints where it is ASCII. -/
theorem checkRecord_ok (ct : CoinTree.Spec) (ha : Ascii ct.toJson) {cj : Json}
    (hs : Same (coinTreeJ ct) cj) : ∃ key, CoinTree.checkRecord ct (zeros 64) cj = .ok key := by
  have hd := coinTreeJ_list_distinct ct
  have gC : ∀ {k v}, (k, v) ∈ [("key", Json.str (zeros 512)), ("nonce", Json.str (zeros 64)),
      ("root", Json.str (zeros 128)), ("spec", ct.toJson)] → (coinTreeJ ct).getObjVal? k = .ok v :=
    fun hm => getObjVal?_mkObj hd hm
  have hkey := strAs_same hs (gC (k := "key") (v := Json.str (zeros 512)) (by simp))
  have hnon := strAs_same hs (gC (k := "nonce") (v := Json.str (zeros 64)) (by simp))
  have hroot := strAs_same hs (gC (k := "root") (v := Json.str (zeros 128)) (by simp))
  obtain ⟨sv, hsv, hsvs⟩ := hs.getObjVal (gC (k := "spec") (v := ct.toJson) (by simp))
  obtain ⟨cs, hcs⟩ := FlockComplete.JsonRT.canon_ok _ (good_specJson ct) ha
  have hcs' : canon sv = .ok cs := by rw [← FlockComplete.JsonRT.canon_same _ _ hsvs, hcs]
  obtain ⟨n, rfl, hn, hk⟩ := same_obj_wf hs
  have hkeys : n.keys = ["key", "nonce", "root", "spec"] := keys_of hn (by decide) fun k => by
    rw [← hk, isSome_get?_mkObj hd]
    rfl
  have hnames : (n.toArray.map fun (kv : String × Json) => kv.1) = #["key", "nonce", "root", "spec"] := by
    rw [show (fun (kv : String × Json) => kv.1) = Prod.fst from rfl, Std.TreeMap.Raw.map_fst_toArray_eq_keysArray,
      ← Std.TreeMap.Raw.toArray_keys, hkeys]
  have hq : (n.toArray.map fun (kv : String × Json) => kv.1).qsort (· < ·) = #["key", "nonce", "root", "spec"] := by
    rw [hnames]
    exact qsort_coinTree
  have ok' : ∀ {k s : String}, (Json.obj n).getObjValAs? String k = .ok s →
      ((Json.obj n).getObjValAs? String k).toOption = some s := fun h => by rw [h]; rfl
  have hsv' : ((Json.obj n).getObjVal? "spec").toOption = some sv := by rw [hsv]; rfl
  unfold CoinTree.checkRecord
  simp only [hq, bne_self_eq_false, Bool.false_eq_true, ite_false, ok' hkey, ok' hnon, ok' hroot, Option.getD_some,
    isHex_zeros, Bool.not_true, hsv', hcs, hcs', ok_bind]
  simp only [String.contains_char_eq]
  rw [show ((Json.obj n).getObjValAs? String "key").toOption.getD "" = zeros 512 by rw [hkey]; rfl]
  have hb := zeros_bit 512 (by decide)
  rw [String.contains_char_eq] at hb
  simp only [hb, Bool.not_true, Bool.false_eq_true, ite_false]
  exact ⟨_, rfl⟩

/-- **S2 passed, the masked record decodes as the plain one** at `specZ`: the prover's `Hello` and the record's
`coin_tree` pass, and the rest reads no `Hello` and no coin tree. -/
theorem decodeWith_specZ (hs : Hashes V) {spec : SessionSpec} {ct : CoinTree.Spec} (hct : spec.coinTree = some ct)
    {text : String} {j : Json} (hp : Flock.JsonParse.parse text = .ok j) {own : Json} (hown : Flock.JsonParse.parse spec.hello = .ok own)
    {hz : String} (hh : Record.str j "hello" = .ok hz) {nonce : String} (hn : CoinTree.checkHello own hz = .ok nonce)
    {cj : Json} (hc : Record.fld j "coin_tree" = .ok cj) {key : String}
    (hk : CoinTree.checkRecord ct nonce cj = .ok key) :
    Record.decodeWith hs spec text = Record.decodeWith hs (specZ spec hz) text := by
  unfold Record.decodeWith Record.decodeJWith
  simp only [hp, hct, specZ, pure_bind, liftM, monadLift_self, hown, hh, hc, hn, hk, Except.toOption, ok_bind,
    beq_self_eq_true, check_true]
  rfl

end FlockComplete.JsonRT
