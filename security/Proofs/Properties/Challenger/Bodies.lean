module

public import Proofs.Properties.Challenger.Built
import all Init.Data.Ord.String
import all Lean.Data.Json.Basic

/-!
# The bodies of a window's open and its end, read back

A ledger line's reader checks the body of the entry its text parses to, a value alike the body the auditor wrote
(`FlockComplete.JsonRT.parse_canon`). So each check on a body is stated on a value alike it:

* a read of a key whose value is a string, a number or a Boolean gives that value (`read_str`, `read_num`,
  `read_bool`), and a value alike an object of digests is an object of digests (`digests_alike`);
* a value alike the body of the open of a window withheld (`openBody`) is an open's body under the ledger's schema
  (`openBody_alike`), and a value alike the body of the end of a window closed (`endBody`) is an end's
  (`endBody_alike`).
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

open private isDigest isBodyOf isStr isV1Run from Spot.ChallengerSession

namespace Properties.Challenger.Bodies

open Lean (Json JsonNumber)
open _root_.Spot _root_.Spot.Challenger
open Claims.Json (Same)
open FlockComplete.JsonRT (same_str same_num same_bool)

theorem read_str {a b : Json} (h : Same a b) {k s : String} (ha : getD a k = .str s) : getD b k = .str s :=
  same_str (ha ▸ Entry.same_getD h k)

theorem read_num {a b : Json} (h : Same a b) {k : String} {n : JsonNumber} (ha : getD a k = .num n) :
    getD b k = .num n :=
  same_num (ha ▸ Entry.same_getD h k)

theorem read_bool {a b : Json} (h : Same a b) {k : String} {x : Bool} (ha : getD a k = .bool x) : getD b k = .bool x :=
  same_bool (ha ▸ Entry.same_getD h k)

theorem nat?_num (n : Nat) : nat? (.num ⟨n, 0⟩) = some n := by
  simp [nat?, int?]

/-- **A value alike an object of digests is an object of digests.** -/
theorem digests_alike {l : List (String × Json)} (hs : (l.map (·.1)).Pairwise (· < ·))
    (hv : ∀ p ∈ l, ∃ b : ByteArray, 2 * b.size = 128 ∧ p.2 = .str (Flock.hexOf b)) {j : Json}
    (h : Same (Json.mkObj l) j) : ∃ n, j = .obj n ∧ (entries j).all (isDigest ·.2) = true := by
  obtain ⟨n, rfl⟩ := Entry.same_obj_eq h
  refine ⟨n, rfl, ?_⟩
  cases h with
  | obj hm hn hk hvs =>
    show n.toArray.all (isDigest ·.2) = true
    rw [Array.all_eq_true']
    rintro ⟨k, v⟩ hmem
    have hg := (Entry.mem_toArray hn).1 hmem
    have hsome : ((Std.TreeMap.Raw.ofList l compare).get? k).isSome = true := by rw [hk k, hg]; rfl
    obtain ⟨a, ha⟩ := Option.isSome_iff_exists.1 hsome
    obtain ⟨b, hb, hab⟩ := hv _ (FlockComplete.JsonRT.mem_of_get?_mkObj (CanonRead.distinct_of_keys hs) ha)
    have hsame := hvs k a v ha hg
    rw [show a = Json.str (Flock.hexOf b) from hab] at hsame
    rw [show v = Json.str (Flock.hexOf b) from same_str hsame]
    exact Built.isDigest_hex hb

/-- The body of the open of a window withheld. -/
def openBody (auditor host pins : String) (keyHash pin window : ByteArray) : Json :=
  Json.mkObj [("by", .str auditor), ("host", .str host), ("key_sha512", .str (Flock.hexOf keyHash)),
    ("pinned", .str (Flock.hexOf pin)), ("pins", Json.mkObj [("lines", Lean.toJson (1 : Nat)), ("sha512", .str pins)]),
    ("window", Json.mkObj [("root", .str (Flock.hexOf window))]), ("withheld", .bool true)]

/-- **A value alike the open's body is an open's body.** -/
theorem openBody_alike {auditor host pins : String} {keyHash pin window : ByteArray} {b : Json}
    (h : Same (openBody auditor host pins keyHash pin window) b) (hh : isHost host = true)
    (hk : keyHash.size = 64) (hp : pin.size = 20) (hw : window.size = 64) : isBodyOf "open" true b = true := by
  have hwin : isRun (getD b "window") = true := by
    have hsw := Entry.same_getD h "window"
    have hkw : keys? (getD b "window") = some #["root"] :=
      (Entry.same_keys? hsw).symm.trans (Built.keys?_mkObj (l := [("root", .str (Flock.hexOf window))])
        (by simp only [List.map_cons, List.map_nil]; decide))
    have hr : getD (getD b "window") "root" = .str (Flock.hexOf window) := read_str hsw rfl
    unfold isRun
    rw [hkw, hr, Built.isDigest_hex (by omega), beq_self_eq_true]
    rfl
  have hhost : getD b "host" = .str host := read_str h rfl
  have hpin : getD b "pinned" = .str (Flock.hexOf pin) := read_str h rfl
  have hkey : getD b "key_sha512" = .str (Flock.hexOf keyHash) := read_str h rfl
  have hby : getD b "by" = .str auditor := read_str h rfl
  have hfor : field? b "for" = none := Entry.same_field?_none h rfl
  show ((if true = true then isRun (getD b "window") else isV1Run (getD b "window")) &&
      (match getD b "host" with | .str h => isHost h | _ => false) && isDigest (getD b "pinned") 40 &&
      isDigest (getD b "key_sha512") &&
      (!true || isStr (getD b "by") && (match field? b "for" with | none => true | some j => isStr j))) = true
  rw [hwin, hhost, hpin, hkey, hby, hfor, Built.isDigest_hex (by omega), Built.isDigest_hex (by omega)]
  show (true && isHost host && true && true && true) = true
  rw [hh]
  rfl

/-- The body of the end of a window closed. -/
def endBody (commitment : ByteArray) (files : Json) (recordLeaf : Nat) : Json :=
  Json.mkObj [("closed", .bool true), ("commitment", .str (Flock.hexOf commitment)), ("files", files),
    ("record_leaf", Lean.toJson recordLeaf), ("refused_hellos", Lean.toJson (0 : Nat))]

/-- **A value alike the end's body is an end's body.** -/
theorem endBody_alike {commitment : ByteArray} {l : List (String × Json)} {recordLeaf : Nat} {b : Json}
    (h : Same (endBody commitment (Json.mkObj l) recordLeaf) b) (hs : (l.map (·.1)).Pairwise (· < ·))
    (hv : ∀ p ∈ l, ∃ d : ByteArray, 2 * d.size = 128 ∧ p.2 = .str (Flock.hexOf d)) :
    isBodyOf "end" true b = true := by
  have hc : getD b "closed" = .bool true := read_bool h rfl
  obtain ⟨n, hn, hall⟩ := digests_alike hs hv (Entry.same_getD h "files")
  have hall' : (entries (.obj n)).all (isDigest ·.2) = true := hn ▸ hall
  have hr : getD b "record_leaf" = .num ⟨recordLeaf, 0⟩ := read_num h rfl
  show ((match getD b "closed" with | .bool _ => true | _ => false) &&
      (match getD b "files" with | .obj _ => (entries (getD b "files")).all (isDigest ·.2) | _ => false) &&
      (!true || (getD b "record_leaf").isNull || (nat? (getD b "record_leaf")).isSome)) = true
  rw [hc, hn, hr, nat?_num]
  show (true && (entries (.obj n)).all (isDigest ·.2) && true) = true
  rw [hall']
  rfl

end Properties.Challenger.Bodies
