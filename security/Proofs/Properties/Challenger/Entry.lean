module

public import Proofs.Properties.Challenger.Plain
public import Proofs.Properties.Challenger.Steps
public import Proofs.Properties.Challenger.CanonLine
public import Proofs.Properties.Verity.Sampled.RecordDraw
public import Proofs.Flock.Claims.Complete.Json.Canon
public import Proofs.Flock.Claims.Complete.Zk.Json.Record
public import Proofs.Core.QSort
public import CoreProtocol.Record
import all Init.Data.Ord.String
import all Lean.Data.Json.Basic

/-!
# A record's line reads back as the record

A ledger line's reader (`Spot.Challenger.ledgerEntry?`) parses the line and reads the parse as a record entry
(`Core.Record.ofJson`); a log line's reader (`Spot.Challenger.event`) parses the line. The parse of a value's
`Flock.canon` text is not the value: it is a value alike it (`Claims.Json.Same`, `FlockComplete.JsonRT.parse_canon`),
another tree with the same value at each key. So:

* alike values have alike values at each key, the same keys and the same missing keys (`same_getD`, `same_keys?`,
  `same_field?_none`), and a value alike a plain value is plain (`plainAt_same`);
* `Record.ofJson` reads an object as the record of its seven values (`ofJson_obj_ok`), which a record's entry
  (`Core.Record.toJson`) holds (`toJson_vals`), so a value alike a record's entry reads as that record, its body alike
  the record's (`ofJson_alike`);
* a printable value's canonical line reads as a value alike it (`event_of_canon`), and the canonical text of a plain
  record entry of the ledger's form reads as an entry alike it (`ledgerEntry_of_record`);
* alike entries sign one message (`ledgerMessage_same`): the message reads an entry's schema and the entry without its
  `sig` (`same_unsigned`, `canonUtf8_same`), so it does not read the `sig` (`ledgerMessage_sig`).
-/

@[expose] public section

set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

open private escapesSurrogate isBodyOf from Spot.ChallengerSession

namespace Properties.Challenger.Entry

open Lean (Json)
open _root_.Spot _root_.Spot.Challenger
open Claims.Json (Same Good)
open FlockComplete.JsonRT (same_str same_null same_num keys_of isSome_get?_mkObj)
open Plain (PlainStr PlainAt canonical_canon noEscape_canon escapesSurrogate_false)

/-- `qsort` leaves an array of strings that strictly increase where it is. -/
theorem qsort_strs {as : Array String} (hp : as.toList.Pairwise (· < ·)) : as.qsort (· < ·) = as := by
  rw [List.pairwise_iff_getElem] at hp
  apply Verity.QSort.qsort_sorted
  intro i hi j hj
  rcases Nat.lt_trichotomy i j with hij | rfl | hij
  · have := hp i j (by simpa using hi) (by simpa using hj) hij
    simp only [Array.getElem_toList] at this
    simp [hij, this]
  · simp
  · have := hp j i (by simpa using hj) (by simpa using hi) hij
    simp only [Array.getElem_toList] at this
    simp [String.lt_asymm this, Nat.lt_asymm hij]

/-- Alike values have alike values at each key, `null` where they have none. -/
theorem same_getD {j j' : Json} (h : Same j j') (k : String) : Same (getD j k) (getD j' k) := by
  cases h with
  | obj hm hn hk hv =>
    rename_i m n
    rw [CanonRead.getD_obj, CanonRead.getD_obj]
    have hs := hk k
    cases ha : m.get? k with
    | none =>
      rw [ha] at hs
      cases hb : n.get? k with
      | none => exact .null
      | some b => rw [hb] at hs; cases hs
    | some a =>
      rw [ha] at hs
      cases hb : n.get? k with
      | none => rw [hb] at hs; cases hs
      | some b => exact hv k a b ha hb
  | _ => exact .null

theorem same_obj_eq {m : Std.TreeMap.Raw String Json} {j : Json} (h : Same (.obj m) j) : ∃ n, j = .obj n := by
  cases h
  exact ⟨_, rfl⟩

/-- A well-formed tree's pairs are its lookups. -/
theorem mem_toArray {m : Std.TreeMap.Raw String Json} (hm : m.WF) {k : String} {v : Json} :
    (k, v) ∈ m.toArray ↔ m.get? k = some v := by
  rw [← Array.mem_toList_iff, Std.TreeMap.Raw.toList_toArray, Std.TreeMap.Raw.mem_toList_iff_getElem?_eq_some hm,
    Std.TreeMap.Raw.get?_eq_getElem?]

/-- Alike objects have the same keys. -/
theorem same_keys? {j j' : Json} (h : Same j j') : keys? j = keys? j' := by
  cases h with
  | obj hm hn hk hv =>
    rename_i m n
    have hkeys : n.keys = m.keys := keys_of hn (Std.TreeMap.Raw.ordered_keys hm) fun k => by
      rw [← hk k, Std.TreeMap.Raw.contains_keys hm, Std.TreeMap.Raw.contains_eq_isSome_getElem? hm,
        Std.TreeMap.Raw.get?_eq_getElem?]
    simp only [keys?]
    rw [Std.TreeMap.Raw.map_fst_toArray_eq_keysArray, Std.TreeMap.Raw.map_fst_toArray_eq_keysArray,
      ← Std.TreeMap.Raw.toArray_keys, ← Std.TreeMap.Raw.toArray_keys, hkeys]
  | _ => rfl

/-- Where one of two alike values has no value at a key, neither has the other. -/
theorem same_field?_none {j j' : Json} (h : Same j j') {k : String} (hf : field? j k = none) : field? j' k = none := by
  cases h with
  | obj hm hn hk hv =>
    rename_i m n
    unfold field? Json.getObjVal? at hf ⊢
    dsimp only at hf ⊢
    have hs := hk k
    cases ha : m.get? k with
    | some a => rw [ha] at hf; cases hf
    | none =>
      rw [ha] at hs
      cases hb : n.get? k with
      | none => rfl
      | some b => rw [hb] at hs; cases hs
  | _ => rfl

/-- A value alike a plain value is plain. -/
theorem plainAt_same {d : Nat} {j : Json} (h : PlainAt d j) : ∀ {j' : Json}, Same j j' → PlainAt d j' := by
  induction h with
  | null d => intro j' hs; cases hs; exact .null d
  | bool d b => intro j' hs; cases hs; exact .bool d b
  | num d m => intro j' hs; cases hs; exact .num d m
  | str d h => intro j' hs; cases hs; exact .str d h
  | arr d h ih =>
    intro j' hs
    cases hs with
    | arr hsz he =>
      refine .arr d fun x hx => ?_
      obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hx
      exact ih _ (Array.getElem_mem (hsz ▸ hi)) (he i (hsz ▸ hi) hi)
  | @obj d m hm hk h ih =>
    intro j' hs
    cases hs with
    | @obj _ n _ hn hkk hv =>
      have key : ∀ p ∈ n.toArray, ∃ a, (p.1, a) ∈ m.toArray ∧ Same a p.2 := fun p hp => by
        have hp' : n.get? p.1 = some p.2 := (mem_toArray hn).1 hp
        have hs := hkk p.1
        rw [hp'] at hs
        obtain ⟨a, ha⟩ := Option.isSome_iff_exists.1 hs
        exact ⟨a, (mem_toArray hm).2 ha, hv p.1 a p.2 ha hp'⟩
      refine .obj d hn (fun p hp => ?_) fun p hp => ?_
      · obtain ⟨a, ha, _⟩ := key p hp
        exact hk (p.1, a) ha
      · obtain ⟨a, ha, hsame⟩ := key p hp
        exact ih _ ha hsame

/-- A plain value is printable. -/
theorem good_of_plain {d : Nat} {j : Json} (h : PlainAt d j) : Good j := by
  induction h with
  | null d => exact .null
  | bool d b => exact .bool b
  | num d m => exact .num m
  | str d h => exact .str _
  | arr d h ih => exact .arr ih
  | obj d hm hk h ih => exact .obj hm fun k v hv => ih (k, v) ((mem_toArray hm).2 hv)

/-- An object of distinct plain keys and plain values is plain. -/
theorem plainAt_mkObj {d : Nat} {l : List (String × Json)} (hd : FlockComplete.JsonRT.Distinct l)
    (hk : ∀ p ∈ l, PlainStr p.1) (hv : ∀ p ∈ l, PlainAt d p.2) : PlainAt (d + 1) (Json.mkObj l) :=
  .obj d Std.TreeMap.Raw.WF.ofList
    (fun _ hp => hk _ (FlockComplete.JsonRT.mem_of_get?_mkObj hd ((mem_toArray Std.TreeMap.Raw.WF.ofList).1 hp)))
    fun _ hp => hv _ (FlockComplete.JsonRT.mem_of_get?_mkObj hd ((mem_toArray Std.TreeMap.Raw.WF.ofList).1 hp))

/-- **A printable value's canonical line reads as a value alike it**, when it is no longer than a log line. -/
theorem event_of_canon {d : Json} {line : String} (hg : Good d) (hc : Flock.canon d = .ok line)
    (hlen : line.length ≤ MAX_LOG_LINE) : ∃ j, event line = .ok j ∧ Same d j := by
  obtain ⟨j, hp, hs⟩ := FlockComplete.JsonRT.parse_canon d line hg hc
  refine ⟨j, ?_, hs⟩
  have hcj : Flock.canon j = .ok line := FlockComplete.JsonRT.canon_same d j hs ▸ hc
  unfold event
  simp [hlen, hp, hcj, Verity.Sampled.sortedKeys_of_wf j (FlockDeploy.same_wf_right hs), pure, Except.pure]

/-- **`Record.ofJson` reads an object as the record of its seven values**: its keys are the seven, its body is an
object, its kind, prev, schema and sig strings, prev a SHA-512 in hex and sig hex, its seq a natural number and its
session a string or null. -/
theorem ofJson_obj_ok {n nb : Std.TreeMap.Raw String Json} {kind prev schema sig : String} {pb sigb : ByteArray}
    {seq : Nat} {session : Option String}
    (hkeys : (n.toArray.map (·.1)).qsort (· < ·) = Core.Record.FIELDS)
    (hbody : getD (.obj n) "body" = .obj nb) (hkind : getD (.obj n) "kind" = .str kind)
    (hprev : getD (.obj n) "prev" = .str prev) (hlen : prev.length = 128) (hhex : Flock.unhex prev = .ok pb)
    (hschema : getD (.obj n) "schema" = .str schema) (hseq : getD (.obj n) "seq" = .num ⟨seq, 0⟩)
    (hsession : getD (.obj n) "session" = (session.map Json.str).getD .null)
    (hsig : getD (.obj n) "sig" = .str sig) (hsigb : Flock.unhex sig = .ok sigb) :
    Core.Record.ofJson (.obj n) = .ok ⟨.obj nb, kind, prev, schema, seq, session, sigb⟩ := by
  unfold getD field? at hbody hkind hprev hschema hseq hsession hsig
  have ht : (Flock.unhex prev).toOption.isSome = true := by rw [hhex]; rfl
  unfold Core.Record.ofJson
  cases session with
  | none =>
    simp only [Option.map_none, Option.getD_none] at hsession
    simp only [hkeys, hbody, hkind, hprev, hlen, ht, hschema, hseq, hsession, hsig, hsigb, beq_self_eq_true,
      Bool.and_self, ↓reduceIte, Int.natCast_nonneg, decide_true, Int.toNat_natCast, bind, Except.bind, pure,
      Except.pure]
  | some s =>
    simp only [Option.map_some, Option.getD_some] at hsession
    simp only [hkeys, hbody, hkind, hprev, hlen, ht, hschema, hseq, hsession, hsig, hsigb, beq_self_eq_true,
      Bool.and_self, ↓reduceIte, Int.natCast_nonneg, decide_true, Int.toNat_natCast, bind, Except.bind, pure,
      Except.pure]

/-- A record's entry holds the record's seven values. -/
theorem toJson_vals (r : Core.Record) : getD r.toJson "body" = r.body ∧ getD r.toJson "kind" = .str r.kind ∧
    getD r.toJson "prev" = .str r.prev ∧ getD r.toJson "schema" = .str r.schema ∧
    getD r.toJson "seq" = .num ⟨r.seq, 0⟩ ∧ getD r.toJson "session" = (r.session.map Json.str).getD .null ∧
    getD r.toJson "sig" = .str (Flock.hexOf r.sig) := ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- **A value alike a record's entry reads as that record**, its body alike the record's: so reads the parse of the
entry's text. -/
theorem ofJson_alike {r : Core.Record} {e' : Json} {mb : Std.TreeMap.Raw String Json} {pb : ByteArray}
    (h : Same r.toJson e') (hb : r.body = .obj mb) (hlen : r.prev.length = 128) (hhex : Flock.unhex r.prev = .ok pb) :
    ∃ nb, Core.Record.ofJson e' = .ok { r with body := .obj nb } ∧ Same r.body (.obj nb) := by
  have hval := same_getD h
  obtain ⟨hvb, hvk, hvp, hvs, hvq, hvn, hvg⟩ := toJson_vals r
  have hbody := hval "body"
  have hkind := same_str (hvk ▸ hval "kind")
  have hprev := same_str (hvp ▸ hval "prev")
  have hschema := same_str (hvs ▸ hval "schema")
  have hseq := same_num (hvq ▸ hval "seq")
  have hsig := same_str (hvg ▸ hval "sig")
  have hsession : getD e' "session" = (r.session.map Json.str).getD .null := by
    have hs := hval "session"
    rw [hvn] at hs
    cases hr : r.session with
    | none => rw [hr] at hs; exact same_null hs
    | some s => rw [hr] at hs; exact same_str hs
  rw [hvb, hb] at hbody
  unfold Core.Record.toJson Json.mkObj at h
  cases h with
  | obj hm hn hk hv =>
    rename_i n
    obtain ⟨nb, hnb⟩ := same_obj_eq hbody
    have hkeys : (n.toArray.map (·.1)).qsort (· < ·) = Core.Record.FIELDS := by
      have hL : n.keys = Core.Record.FIELDS.toList := keys_of hn (by decide) fun k => by
        rw [← hk k, isSome_get?_mkObj
          (CanonRead.distinct_of_keys (by simp only [List.map_cons, List.map_nil]; decide))]
        rfl
      rw [Std.TreeMap.Raw.map_fst_toArray_eq_keysArray, ← Std.TreeMap.Raw.toArray_keys, hL]
      exact qsort_strs (by decide)
    refine ⟨nb, ?_, ?_⟩
    · exact ofJson_obj_ok hkeys hnb hkind hprev hlen hhex hschema hseq hsession hsig (Flock.Hex.unhex_hexOf r.sig)
    · rw [hb, ← hnb]
      exact hbody

/-- **The reader takes a record's line**: the canonical text of a plain record entry of the ledger's form, an open's or
an end's, reads as an entry alike it, with a body alike the record's. -/
theorem ledgerEntry_of_record {r : Core.Record} {text session tag : String} {mb : Std.TreeMap.Raw String Json}
    {pb : ByteArray} (hp : PlainAt PyJson.DEPTH r.toJson) (hc : Flock.canon r.toJson = .ok text)
    (hlen : text.length ≤ MAX_LEDGER_LINE) (hb : r.body = .obj mb) (hprev : r.prev.length = 128)
    (hhex : Flock.unhex r.prev = .ok pb) (ht : ledgerTag? r.schema = some tag) (h4 : 4 ≤ r.sig.size)
    (hk : r.kind = "open" ∨ r.kind = "end") (hse : r.session = some session) (hid : isChallengePhaseId session = true)
    (hbody : ∀ b', Same r.body b' → isBodyOf r.kind (r.schema == LEDGER_SCHEMA) b' = true) :
    ∃ e nb, Same r.toJson e ∧ Same r.body (.obj nb) ∧
      ledgerEntry? (.str text) = some ⟨e, text, r.schema, r.seq, r.prev, r.kind, session, r.sig, .obj nb⟩ := by
  obtain ⟨e, hpe, hs⟩ := FlockComplete.JsonRT.parse_canon r.toJson text (good_of_plain hp) hc
  obtain ⟨nb, hof, hsb⟩ := ofJson_alike hs hb hprev hhex
  have hce : Flock.canon e = .ok text := FlockComplete.JsonRT.canon_same _ _ hs ▸ hc
  have hpy : PyJson.canonical e = .ok text := (canonical_canon (plainAt_same hp hs)).trans hce
  have hsk : sortedKeys e = true := Verity.Sampled.sortedKeys_of_wf e (FlockDeploy.same_wf_right hs)
  exact ⟨e, nb, hs, hsb, Steps.ledgerEntry_some_of_steps (r := { r with body := .obj nb }) hlen hpe hpy
    (escapesSurrogate_false (noEscape_canon hp hc)) hce hsk hof ht h4 hk hse hid (hbody _ hsb)⟩

/-- A list without its `sig` pairs has none at `sig`. -/
theorem lookup_sig (l : List (String × Json)) : (l.filter (·.1 != "sig")).lookup "sig" = none := by
  induction l with
  | nil => rfl
  | cons p l ih =>
    obtain ⟨k, v⟩ := p
    by_cases hk : k = "sig"
    · subst hk
      simpa [List.filter] using ih
    · have hb : (k != "sig") = true := bne_iff_ne.2 hk
      have hne : ("sig" == k) = false := beq_eq_false_iff_ne.2 (Ne.symm hk)
      simp only [List.filter, hb, List.lookup, hne, ih]

/-- A well-formed tree's keys increase along its list. -/
theorem sorted_of_wf {m : Std.TreeMap.Raw String Json} (hm : m.WF) : (m.toList.map (·.1)).Pairwise (· < ·) := by
  rw [List.pairwise_map]
  exact (Std.TreeMap.Raw.ordered_keys_toList hm).imp fun hab => compareOfLessAndEq_eq_lt.1 hab

/-- **An entry without its `sig` holds the entry's other values**, and none at `sig`. -/
theorem get?_unsigned {m : Std.TreeMap.Raw String Json} (hm : m.WF) (k : String) :
    (Std.TreeMap.Raw.ofList (m.toList.filter (·.1 != "sig")) compare).get? k =
      if k = "sig" then none else m.get? k := by
  rw [CanonRead.get?_ofList (CanonRead.filter_keys (sorted_of_wf hm))]
  split
  · rename_i hk
    rw [hk, lookup_sig]
  · rename_i hk
    rw [CanonRead.lookup_filter hk, CanonRead.get?_lookup (sorted_of_wf hm)]

/-- Alike entries are alike without their `sig`. -/
theorem same_unsigned {m : Std.TreeMap.Raw String Json} {b : Json} (h : Same (.obj m) b) :
    Same (Core.Record.unsigned (.obj m)) (Core.Record.unsigned b) := by
  cases h with
  | @obj _ n hm hn hk hv =>
    rw [CanonRead.unsigned_obj, CanonRead.unsigned_obj]
    refine .obj Std.TreeMap.Raw.WF.ofList Std.TreeMap.Raw.WF.ofList (fun k => ?_) fun k x y hx hy => ?_
    · rw [get?_unsigned hm, get?_unsigned hn]
      split
      · rfl
      · exact hk k
    · rw [get?_unsigned hm] at hx
      rw [get?_unsigned hn] at hy
      split at hx
      · cases hx
      · rename_i hne
        simp only [hne, ↓reduceIte] at hy
        exact hv k x y hx hy

/-- A printable entry is printable without its `sig`. -/
theorem good_unsigned {m : Std.TreeMap.Raw String Json} (h : Good (.obj m)) : Good (Core.Record.unsigned (.obj m)) := by
  cases h with
  | obj hm hg =>
    rw [CanonRead.unsigned_obj]
    refine .obj Std.TreeMap.Raw.WF.ofList fun k v hv => ?_
    rw [get?_unsigned hm] at hv
    split at hv
    · cases hv
    · exact hg k v hv

/-- **Alike sorted values have one identity text**: `canonUtf8` writes them alike, where `canon` writes the first. -/
theorem canonUtf8_same {a b : Json} (h : Same a b) (ha : sortedKeys a = true) (hb : sortedKeys b = true) {s : String}
    (hc : Flock.canon a = .ok s) : Flock.canonUtf8 b = Flock.canonUtf8 a := by
  obtain ⟨t, ht, e⟩ := CanonLine.canon_utf8 ha hc
  obtain ⟨t', ht', e'⟩ := CanonLine.canon_utf8 hb (FlockComplete.JsonRT.canon_same a b h ▸ hc)
  rw [ht, ht', String.toList_inj.1 (e'.symm.trans e)]

/-- **Alike entries sign one message**: the message reads an entry's schema and the entry without its `sig`, which
alike entries have alike. -/
theorem ledgerMessage_same {I : ByteArray} {m : Std.TreeMap.Raw String Json} {b : Json} {schema : String}
    (h : Same (.obj m) b) (hg : Good (.obj m)) (ha : sortedKeys (.obj m) = true)
    (hs : getD (.obj m) "schema" = .str schema) : ledgerMessage I b = ledgerMessage I (.obj m) := by
  have hsb : getD b "schema" = .str schema := same_str (hs ▸ same_getD h "schema")
  obtain ⟨s, hc⟩ := FlockComplete.JsonRT.canon_ok_some _ (good_unsigned hg)
  have hu : Flock.canonUtf8 (Core.Record.unsigned b) = Flock.canonUtf8 (Core.Record.unsigned (.obj m)) := by
    have hb : sortedKeys b = true := Verity.Sampled.sortedKeys_of_wf b (FlockDeploy.same_wf_right h)
    cases h with
    | @obj _ n hm hn hk hv =>
      exact canonUtf8_same (same_unsigned (.obj hm hn hk hv)) (CanonRead.sorted_unsigned ha)
        (CanonRead.sorted_unsigned hb) hc
  unfold ledgerMessage taggedSha512
  rw [hs, hsb, hu]

/-- A record's entry without its `sig` is the record's six other values. -/
theorem unsigned_toJson (r : Core.Record) : Core.Record.unsigned r.toJson =
    Json.mkObj [("body", r.body), ("kind", .str r.kind), ("prev", .str r.prev), ("schema", .str r.schema),
      ("seq", Lean.toJson r.seq), ("session", (r.session.map Json.str).getD .null)] := by rfl

/-- An entry's message does not read its `sig`. -/
theorem ledgerMessage_sig (I : ByteArray) (r : Core.Record) (sig : ByteArray) :
    ledgerMessage I ({ r with sig := sig } : Core.Record).toJson = ledgerMessage I r.toJson := by
  unfold ledgerMessage
  rw [unsigned_toJson, unsigned_toJson]
  rfl

end Properties.Challenger.Entry
