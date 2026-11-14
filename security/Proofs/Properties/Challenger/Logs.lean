module

public import Proofs.Properties.Challenger.Bodies
import all Init.Data.Ord.String
import all Lean.Data.Json.Basic
import all Init.Data.Repr
import all Init.Data.Int.Repr

/-!
# A pins log's line and a release log's lines, read back

Two logs a session's reader reads line by line, each line the canonical text of a value:

* the pins log, whose line names the commit the auditor's device ran: the text of a plain value of the line's form
  reads as its commit (`pinOf_of_canon`);
* the release log of a session that released one challenge of one family: under the head the commitment writes, the
  root's line (`rootLine`) and the release's line (`releaseLine`) read as the family's one root (`released_one`).
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

open private isDigest strOf isTime from Spot.ChallengerSession

namespace Properties.Challenger.Logs

open Lean (Json JsonNumber)
open _root_.Spot _root_.Spot.Challenger
open Claims.Json (Same Good)
open FlockComplete.JsonRT (good_mkObj)
open CanonRead (distinct_of_keys)
open Plain (PlainStr PlainAt canonical_canon)
open Built (isDigest_hex plain_hexOf hexOf_len canon_mkObj_le canon_str_le canon_num keys?_mkObj wf_of_good)
open Bodies (read_str read_num)

/-- **A pins log's line reads as its commit**: the text of a plain value of the pins log's form. -/
theorem pinOf_of_canon {d : Json} {text t : String} {pin : ByteArray} (hp : PlainAt PyJson.DEPTH d)
    (hc : Flock.canon d = .ok text) (hk : keys? d = some #["commit", "main", "pinned_utc"])
    (hcm : getD d "commit" = .str (Flock.hexOf pin)) (hm : getD d "main" = .str (Flock.hexOf pin))
    (hsz : pin.size = 20) (ht : getD d "pinned_utc" = .str t) (hl : later t 0 = some t) :
    pinOf? (.str text) = some (Flock.hexOf pin) := by
  obtain ⟨e, hpe, hs⟩ := FlockComplete.JsonRT.parse_canon d text (Entry.good_of_plain hp) hc
  have hpy : PyJson.canonical e = .ok text := by
    rw [canonical_canon (Entry.plainAt_same hp hs), ← FlockComplete.JsonRT.canon_same d e hs, hc]
  have hke : keys? e = some #["commit", "main", "pinned_utc"] := (Entry.same_keys? hs).symm.trans hk
  have h1 : getD e "commit" = .str (Flock.hexOf pin) := read_str hs hcm
  have h2 : getD e "main" = .str (Flock.hexOf pin) := read_str hs hm
  have h3 : getD e "pinned_utc" = .str t := read_str hs ht
  have hd : isDigest (.str (Flock.hexOf pin)) 40 = true := isDigest_hex (by omega)
  unfold pinOf?
  simp [hpe, hpy, hke, h1, h2, h3, hd, hl, strOf, Except.toOption, guard, bind, Option.bind, pure, failure]

/-- The log's line of the one root the device received: the family `label`'s root `after`, at index 0. -/
def rootLine (label : String) (after : ByteArray) : Json :=
  Json.mkObj [("event", .str "root"), ("index", Lean.toJson (0 : Nat)), ("label", .str label),
    ("received_at", Lean.toJson (0 : Nat)), ("root", .str (Flock.hexOf after))]

/-- The log's line of the one release: the family `label`'s challenge at index 0, released after `after`. -/
def releaseLine (label : String) (after : ByteArray) : Json :=
  Json.mkObj [("after", .str (Flock.hexOf after)), ("event", .str "release"), ("index", Lean.toJson (0 : Nat)),
    ("label", .str label), ("released_at", Lean.toJson (0 : Nat))]

theorem jsonStrIs_str (s : String) : jsonStrIs (.str s) s = true := by simp [jsonStrIs]

theorem isTime_nat (n : Nat) : isTime (.num ⟨n, 0⟩) = true := by simp [isTime, nat?, int?]

theorem canon_zero : ∃ t, Flock.canon (Lean.toJson (0 : Nat)) = .ok t ∧ t.length ≤ 1 :=
  ⟨_, canon_num 0, by decide⟩

/-- The text of the root's line, and its length. -/
theorem canon_rootLine {label : String} (after : ByteArray) (hl : PlainStr label) :
    ∃ t, Flock.canon (rootLine label after) = .ok t ∧ t.length ≤ 64 + label.length + (Flock.hexOf after).length := by
  obtain ⟨t, ht, hlen⟩ := canon_mkObj_le (l := [("event", .str "root"), ("index", Lean.toJson (0 : Nat)),
      ("label", .str label), ("received_at", Lean.toJson (0 : Nat)), ("root", .str (Flock.hexOf after))])
    (ks := ["event", "index", "label", "received_at", "root"])
    (bs := [6, 1, label.length + 2, 1, (Flock.hexOf after).length + 2]) rfl (by decide) (by decide)
    (.cons (canon_str_le (s := "root") (by decide)) (.cons canon_zero (.cons (canon_str_le hl) (.cons canon_zero
      (.cons (canon_str_le (plain_hexOf after)) .nil)))))
  have e : (["event", "index", "label", "received_at", "root"].map (·.length + 4)).sum = 50 := by decide
  rw [e] at hlen
  simp only [List.sum_cons, List.sum_nil] at hlen
  exact ⟨t, ht, by omega⟩

/-- The text of the release's line, and its length. -/
theorem canon_releaseLine {label : String} (after : ByteArray) (hl : PlainStr label) :
    ∃ t, Flock.canon (releaseLine label after) = .ok t ∧ t.length ≤ 68 + label.length + (Flock.hexOf after).length := by
  obtain ⟨t, ht, hlen⟩ := canon_mkObj_le (l := [("after", .str (Flock.hexOf after)), ("event", .str "release"),
      ("index", Lean.toJson (0 : Nat)), ("label", .str label), ("released_at", Lean.toJson (0 : Nat))])
    (ks := ["after", "event", "index", "label", "released_at"])
    (bs := [(Flock.hexOf after).length + 2, 9, 1, label.length + 2, 1]) rfl (by decide) (by decide)
    (.cons (canon_str_le (plain_hexOf after)) (.cons (canon_str_le (s := "release") (by decide)) (.cons canon_zero
      (.cons (canon_str_le hl) (.cons canon_zero .nil)))))
  have e : (["after", "event", "index", "label", "released_at"].map (·.length + 4)).sum = 51 := by decide
  rw [e] at hlen
  simp only [List.sum_cons, List.sum_nil] at hlen
  exact ⟨t, ht, by omega⟩

theorem good_rootLine (label : String) (after : ByteArray) : Good (rootLine label after) :=
  good_mkObj (distinct_of_keys (by simp only [List.map_cons, List.map_nil]; decide)) (by
    intro p hp
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hp
    rcases hp with rfl | rfl | rfl | rfl | rfl
    all_goals first | exact Good.str _ | exact Good.num _)

theorem good_releaseLine (label : String) (after : ByteArray) : Good (releaseLine label after) :=
  good_mkObj (distinct_of_keys (by simp only [List.map_cons, List.map_nil]; decide)) (by
    intro p hp
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hp
    rcases hp with rfl | rfl | rfl | rfl | rfl
    all_goals first | exact Good.str _ | exact Good.num _)

/-- **The log of one root and one release gives the family its root**: under the head a commitment of the one family
`label` writes, the root's line and the release's line, each the text of its value, read as the family's one root. -/
theorem released_one {c : LiveCommitment} {label : String} {after : ByteArray} {n : Nat}
    (hf : c.families = [(label, n)]) (hl : PlainStr label) (hll : label.length ≤ 128) (ha : after.size = 64) :
    ∃ a b, Flock.canon (rootLine label after) = .ok a ∧ Flock.canon (releaseLine label after) = .ok b ∧
      releasedRootsOf c ["{\"event\":\"commitment\",\"root\":\"" ++ Flock.hexOf c.root ++ "\"}", a, b] =
        .ok [(label, [after])] := by
  have hx : (Flock.hexOf after).length = 128 := by rw [hexOf_len, ha]
  obtain ⟨a, hca, hla⟩ := canon_rootLine after hl
  obtain ⟨b, hcb, hlb⟩ := canon_releaseLine after hl
  obtain ⟨r, hr, hsr⟩ := Entry.event_of_canon (good_rootLine label after) hca (by unfold MAX_LOG_LINE; omega)
  obtain ⟨s, hs, hss⟩ := Entry.event_of_canon (good_releaseLine label after) hcb (by unfold MAX_LOG_LINE; omega)
  have hta : treeLine a = false :=
    Verity.Sampled.Lines.treeLine_plain
      ⟨Verity.Sampled.sortedKeys_of_wf _ (wf_of_good (good_rootLine label after)), hca⟩
      ⟨"root", rfl, by decide, by decide⟩
  have htb : treeLine b = false :=
    Verity.Sampled.Lines.treeLine_plain
      ⟨Verity.Sampled.sortedKeys_of_wf _ (wf_of_good (good_releaseLine label after)), hcb⟩
      ⟨"release", rfl, by decide, by decide⟩
  have hkr : keys? r = some #["event", "index", "label", "received_at", "root"] :=
    (Entry.same_keys? hsr).symm.trans (keys?_mkObj (by simp only [List.map_cons, List.map_nil]; decide))
  have hks : keys? s = some #["after", "event", "index", "label", "released_at"] :=
    (Entry.same_keys? hss).symm.trans (keys?_mkObj (by simp only [List.map_cons, List.map_nil]; decide))
  have h1 : getD r "event" = .str "root" := read_str hsr rfl
  have h2 : getD s "event" = .str "release" := read_str hss rfl
  have h3 : getD r "received_at" = .num ⟨(0 : Nat), 0⟩ := read_num hsr rfl
  have h4 : getD s "released_at" = .num ⟨(0 : Nat), 0⟩ := read_num hss rfl
  have h5 : getD r "label" = .str label := read_str hsr rfl
  have h6 : getD r "root" = .str (Flock.hexOf after) := read_str hsr rfl
  have h7 : getD r "index" = .num ⟨(0 : Nat), 0⟩ := read_num hsr rfl
  have h8 : getD s "index" = .num ⟨(0 : Nat), 0⟩ := read_num hss rfl
  have h9 : getD s "label" = .str label := read_str hss rfl
  have h10 : getD s "after" = .str (Flock.hexOf after) := read_str hss rfl
  exact ⟨a, b, hca, hcb, Steps.releasedRootsOf_one hf hta htb hr hs hkr hks (by rw [h1]; exact jsonStrIs_str _)
    (by rw [h2]; exact jsonStrIs_str _) (by rw [h3]; exact isTime_nat 0) (by rw [h4]; exact isTime_nat 0) h5 h6
    (Flock.Hex.unhex_hexOf after) (by rw [h7]; rfl) (by rw [h8]; rfl) (by rw [h9]; exact jsonStrIs_str _)
    (by rw [h10]; exact jsonStrIs_str _) hx⟩

end Properties.Challenger.Logs
