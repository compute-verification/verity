module

public import Proofs.Properties.Challenger.Bodies
import all Init.Data.Ord.String
import all Lean.Data.Json.Basic
import all Init.Data.Repr
import all Init.Data.Int.Repr

/-!
# A ledger's entry as text: plain, and no longer than a line

`Entry.ledgerEntry_of_record` reads a record entry back from its text where the entry is plain and its text is no
longer than a ledger line. These say so of the entries an auditor writes for a window withheld:

* a value plain at one depth is plain at every greater one (`plainAt_mono`), and an object built from a list whose
  keys are plain and increase is plain where its values are (`plainAt_built`);
* the first entry's `prev`, 128 zeros, is the hex of 64 zero bytes (`zero_hex`), plain and of 128 characters;
* the files an end seals (`filesOf`), the open's body and the end's body are plain, each with its text and a bound on
  the text's length (`canon_files`, `canon_openBody`, `canon_endBody`);
* a record entry is plain where its body is and its strings are (`plainAt_record`), and its text is no longer than
  its parts' (`canon_record`).
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Properties.Challenger.Ledger

open Lean (Json)
open _root_.Spot _root_.Spot.Challenger
open CanonRead (distinct_of_keys)
open Plain (PlainStr PlainAt)
open Built (plain_hexOf hexOf_len canon_mkObj_le canon_str_le canon_num)
open Bodies (openBody endBody)

/-- A value plain at one depth is plain at every greater one. -/
theorem plainAt_mono {d : Nat} {j : Json} (h : PlainAt d j) : ∀ {e : Nat}, d ≤ e → PlainAt e j := by
  induction h with
  | null d => intro e he; cases e with | zero => omega | succ e => exact .null e
  | bool d b => intro e he; cases e with | zero => omega | succ e => exact .bool e b
  | num d m => intro e he; cases e with | zero => omega | succ e => exact .num e m
  | str d h => intro e he; cases e with | zero => omega | succ e => exact .str e h
  | arr d h ih => intro e he; cases e with | zero => omega | succ e => exact .arr e fun x hx => ih x hx (by omega)
  | obj d hm hk h ih =>
    intro e he
    cases e with
    | zero => omega
    | succ e => exact .obj e hm hk fun p hp => ih p hp (by omega)

/-- An object built from a list whose keys are plain and increase is plain where its values are. -/
theorem plainAt_built {d : Nat} {l : List (String × Json)} {ks : List String} (hks : l.map (·.1) = ks)
    (hs : ks.Pairwise (· < ·)) (hk : ∀ k ∈ ks, PlainStr k) (hv : ∀ p ∈ l, PlainAt d p.2) :
    PlainAt (d + 1) (Json.mkObj l) :=
  Entry.plainAt_mkObj (distinct_of_keys (hks ▸ hs)) (fun _ hp => hk _ (hks ▸ List.mem_map_of_mem hp)) hv

/-! ## The first entry's `prev` -/

theorem zero_toList : ZERO.toList = List.replicate 128 '0' := String.toList_ofList

set_option maxRecDepth 8192 in
/-- The first entry's `prev`, 128 zeros, is the hex of 64 zero bytes. -/
theorem zero_hex : Flock.hexOf ⟨Array.replicate 64 0⟩ = ZERO := by
  apply String.toList_injective
  rw [Properties.NetworkRecords.Bytes.hexOf_toList, zero_toList]
  decide +kernel

theorem zero_len : ZERO.length = 128 := by
  rw [← String.length_toList, zero_toList, List.length_replicate]

theorem zero_plain : PlainStr ZERO := zero_hex ▸ plain_hexOf _

theorem zero_unhex : Flock.unhex ZERO = .ok ⟨Array.replicate 64 0⟩ := zero_hex ▸ Flock.Hex.unhex_hexOf _

theorem canon_true : Flock.canon (.bool true) = .ok "true" := by
  rw [Flock.canon]
  rfl

/-- The files an end seals: the digests of the close, the refused hellos, the record and the release log. -/
def filesOf (close hellos record log : ByteArray) : Json :=
  Json.mkObj [("challenger.json", .str (Flock.hexOf close)), ("hellos.jsonl", .str (Flock.hexOf hellos)),
    ("record.json", .str (Flock.hexOf record)), ("releases.jsonl", .str (Flock.hexOf log))]

theorem plainAt_files (close hellos record log : ByteArray) : PlainAt 2 (filesOf close hellos record log) :=
  plainAt_built (ks := ["challenger.json", "hellos.jsonl", "record.json", "releases.jsonl"]) rfl (by decide)
    (by decide) (by
      intro p hp
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hp
      rcases hp with rfl | rfl | rfl | rfl <;> exact .str 0 (plain_hexOf _))

theorem canon_files (close hellos record log : ByteArray) :
    ∃ t, Flock.canon (filesOf close hellos record log) = .ok t ∧
      t.length ≤ 78 + (Flock.hexOf close).length + (Flock.hexOf hellos).length + (Flock.hexOf record).length +
        (Flock.hexOf log).length := by
  obtain ⟨t, ht, hlen⟩ := canon_mkObj_le (l := [("challenger.json", .str (Flock.hexOf close)),
      ("hellos.jsonl", .str (Flock.hexOf hellos)), ("record.json", .str (Flock.hexOf record)),
      ("releases.jsonl", .str (Flock.hexOf log))])
    (ks := ["challenger.json", "hellos.jsonl", "record.json", "releases.jsonl"])
    (bs := [(Flock.hexOf close).length + 2, (Flock.hexOf hellos).length + 2, (Flock.hexOf record).length + 2,
      (Flock.hexOf log).length + 2]) rfl (by decide) (by decide)
    (.cons (canon_str_le (plain_hexOf _)) (.cons (canon_str_le (plain_hexOf _)) (.cons (canon_str_le (plain_hexOf _))
      (.cons (canon_str_le (plain_hexOf _)) .nil))))
  have e : (["challenger.json", "hellos.jsonl", "record.json", "releases.jsonl"].map (·.length + 4)).sum = 68 := by
    decide
  rw [e] at hlen
  simp only [List.sum_cons, List.sum_nil] at hlen
  exact ⟨t, ht, by omega⟩

theorem plainAt_openBody {auditor host pins : String} (keyHash pin window : ByteArray) (ha : PlainStr auditor)
    (hh : PlainStr host) (hp : PlainStr pins) : PlainAt 3 (openBody auditor host pins keyHash pin window) :=
  plainAt_built (ks := ["by", "host", "key_sha512", "pinned", "pins", "window", "withheld"]) rfl (by decide)
    (by decide) (by
      intro p hm
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hm
      rcases hm with rfl | rfl | rfl | rfl | rfl | rfl | rfl
      · exact .str 1 ha
      · exact .str 1 hh
      · exact .str 1 (plain_hexOf _)
      · exact .str 1 (plain_hexOf _)
      · exact plainAt_built (ks := ["lines", "sha512"]) rfl (by decide) (by decide) (by
          intro q hq
          simp only [List.mem_cons, List.not_mem_nil, or_false] at hq
          rcases hq with rfl | rfl
          · exact .num 0 _
          · exact .str 0 hp)
      · exact plainAt_built (ks := ["root"]) rfl (by decide) (by decide) (by
          intro q hq
          simp only [List.mem_cons, List.not_mem_nil, or_false] at hq
          subst hq
          exact .str 0 (plain_hexOf _))
      · exact .bool 1 true)

theorem canon_openBody {auditor host pins : String} (keyHash pin window : ByteArray) (ha : PlainStr auditor)
    (hh : PlainStr host) (hp : PlainStr pins) :
    ∃ t, Flock.canon (openBody auditor host pins keyHash pin window) = .ok t ∧
      t.length ≤ 118 + auditor.length + host.length + pins.length + (Flock.hexOf keyHash).length +
        (Flock.hexOf pin).length + (Flock.hexOf window).length := by
  obtain ⟨tp, htp, hlp⟩ := canon_mkObj_le (l := [("lines", Lean.toJson (1 : Nat)), ("sha512", .str pins)])
    (ks := ["lines", "sha512"]) (bs := [1, pins.length + 2]) rfl (by decide) (by decide)
    (.cons ⟨_, canon_num 1, by decide⟩ (.cons (canon_str_le hp) .nil))
  have ep : (["lines", "sha512"].map (·.length + 4)).sum = 19 := by decide
  rw [ep] at hlp
  simp only [List.sum_cons, List.sum_nil] at hlp
  obtain ⟨tw, htw, hlw⟩ := canon_mkObj_le (l := [("root", .str (Flock.hexOf window))])
    (ks := ["root"]) (bs := [(Flock.hexOf window).length + 2]) rfl (by decide) (by decide)
    (.cons (canon_str_le (plain_hexOf _)) .nil)
  have ew : (["root"].map (·.length + 4)).sum = 8 := by decide
  rw [ew] at hlw
  simp only [List.sum_cons, List.sum_nil] at hlw
  obtain ⟨t, ht, hlen⟩ := canon_mkObj_le (l := [("by", .str auditor), ("host", .str host),
      ("key_sha512", .str (Flock.hexOf keyHash)), ("pinned", .str (Flock.hexOf pin)),
      ("pins", Json.mkObj [("lines", Lean.toJson (1 : Nat)), ("sha512", .str pins)]),
      ("window", Json.mkObj [("root", .str (Flock.hexOf window))]), ("withheld", .bool true)])
    (ks := ["by", "host", "key_sha512", "pinned", "pins", "window", "withheld"])
    (bs := [auditor.length + 2, host.length + 2, (Flock.hexOf keyHash).length + 2, (Flock.hexOf pin).length + 2,
      24 + pins.length, 12 + (Flock.hexOf window).length, 4]) rfl (by decide) (by decide)
    (.cons (canon_str_le ha) (.cons (canon_str_le hh) (.cons (canon_str_le (plain_hexOf _))
      (.cons (canon_str_le (plain_hexOf _)) (.cons ⟨tp, htp, by omega⟩ (.cons ⟨tw, htw, by omega⟩
        (.cons ⟨_, canon_true, by decide⟩ .nil)))))))
  have e : (["by", "host", "key_sha512", "pinned", "pins", "window", "withheld"].map (·.length + 4)).sum = 68 := by
    decide
  rw [e] at hlen
  simp only [List.sum_cons, List.sum_nil] at hlen
  exact ⟨t, ht, by omega⟩

theorem plainAt_endBody {files : Json} (commitment : ByteArray) (recordLeaf : Nat) (hf : PlainAt 2 files) :
    PlainAt 3 (endBody commitment files recordLeaf) :=
  plainAt_built (ks := ["closed", "commitment", "files", "record_leaf", "refused_hellos"]) rfl (by decide)
    (by decide) (by
      intro p hm
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hm
      rcases hm with rfl | rfl | rfl | rfl | rfl
      · exact .bool 1 true
      · exact .str 1 (plain_hexOf _)
      · exact hf
      · exact .num 1 _
      · exact .num 1 _)

theorem canon_endBody {files : Json} {F : Nat} (commitment : ByteArray) {recordLeaf : Nat}
    (hf : ∃ t, Flock.canon files = .ok t ∧ t.length ≤ F)
    (hr : ∃ t, Flock.canon (Lean.toJson recordLeaf) = .ok t ∧ t.length ≤ 1) :
    ∃ t, Flock.canon (endBody commitment files recordLeaf) = .ok t ∧
      t.length ≤ 76 + (Flock.hexOf commitment).length + F := by
  obtain ⟨t, ht, hlen⟩ := canon_mkObj_le (l := [("closed", .bool true), ("commitment", .str (Flock.hexOf commitment)),
      ("files", files), ("record_leaf", Lean.toJson recordLeaf), ("refused_hellos", Lean.toJson (0 : Nat))])
    (ks := ["closed", "commitment", "files", "record_leaf", "refused_hellos"])
    (bs := [4, (Flock.hexOf commitment).length + 2, F, 1, 1]) rfl (by decide) (by decide)
    (.cons ⟨_, canon_true, by decide⟩ (.cons (canon_str_le (plain_hexOf _)) (.cons hf (.cons hr
      (.cons ⟨_, canon_num 0, by decide⟩ .nil)))))
  have e : (["closed", "commitment", "files", "record_leaf", "refused_hellos"].map (·.length + 4)).sum = 66 := by
    decide
  rw [e] at hlen
  simp only [List.sum_cons, List.sum_nil] at hlen
  exact ⟨t, ht, by omega⟩

/-- **A record entry is plain** where its body is, its kind, prev, schema and session are plain strings. -/
theorem plainAt_record {r : Core.Record} {s : String} (hb : PlainAt 3 r.body) (hk : PlainStr r.kind)
    (hp : PlainStr r.prev) (hsch : PlainStr r.schema) (hse : r.session = some s) (hs : PlainStr s) :
    PlainAt PyJson.DEPTH r.toJson := by
  refine plainAt_mono (d := 4) ?_ (by decide)
  unfold Core.Record.toJson
  rw [hse]
  exact plainAt_built (ks := ["body", "kind", "prev", "schema", "seq", "session", "sig"]) rfl (by decide)
    (by decide) (by
      intro p hm
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hm
      rcases hm with rfl | rfl | rfl | rfl | rfl | rfl | rfl
      · exact hb
      · exact .str 2 hk
      · exact .str 2 hp
      · exact .str 2 hsch
      · exact .num 2 _
      · exact .str 2 hs
      · exact .str 2 (plain_hexOf _))

/-- **A record entry's text and its length.** -/
theorem canon_record {r : Core.Record} {s : String} {B : Nat} (hb : ∃ t, Flock.canon r.body = .ok t ∧ t.length ≤ B)
    (hk : PlainStr r.kind) (hp : PlainStr r.prev) (hsch : PlainStr r.schema)
    (hq : ∃ t, Flock.canon (Lean.toJson r.seq) = .ok t ∧ t.length ≤ 1) (hse : r.session = some s) (hs : PlainStr s) :
    ∃ t, Flock.canon r.toJson = .ok t ∧
      t.length ≤ 72 + B + r.kind.length + r.prev.length + r.schema.length + s.length + (Flock.hexOf r.sig).length := by
  unfold Core.Record.toJson
  rw [hse]
  obtain ⟨t, ht, hlen⟩ := canon_mkObj_le (l := [("body", r.body), ("kind", Json.str r.kind), ("prev", Json.str r.prev),
      ("schema", Json.str r.schema), ("seq", Lean.toJson r.seq), ("session", ((some s).map Json.str).getD Json.null),
      ("sig", Json.str (Flock.hexOf r.sig))])
    (ks := ["body", "kind", "prev", "schema", "seq", "session", "sig"])
    (bs := [B, r.kind.length + 2, r.prev.length + 2, r.schema.length + 2, 1, s.length + 2,
      (Flock.hexOf r.sig).length + 2]) rfl (by decide) (by decide)
    (.cons hb (.cons (canon_str_le hk) (.cons (canon_str_le hp) (.cons (canon_str_le hsch) (.cons hq
      (.cons (canon_str_le hs) (.cons (canon_str_le (plain_hexOf _)) .nil)))))))
  have e : (["body", "kind", "prev", "schema", "seq", "session", "sig"].map (·.length + 4)).sum = 59 := by decide
  rw [e] at hlen
  simp only [List.sum_cons, List.sum_nil] at hlen
  exact ⟨t, ht, by omega⟩

end Properties.Challenger.Ledger
