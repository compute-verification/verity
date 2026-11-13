module

public import Proofs.Properties.Challenger.Plain
public import Proofs.Flock.Claims.Complete.Zk.Json.Record
public import Proofs.Core.QSort
public import CoreProtocol.Record

/-!
# A record's entry reads back from its text

A ledger line's reader (`Spot.Challenger.ledgerEntry?`) parses the line and reads the parse as a record entry
(`Core.Record.ofJson`). The parse of an entry's `Flock.canon` text is not the entry: it is a value alike it
(`Claims.Json.Same`, `FlockComplete.JsonRT.parse_canon`), another tree with the same value at each key. So:

* alike values have alike values at each key (`same_getD`);
* `Record.ofJson` reads an object as the record of its seven values (`ofJson_obj_ok`), which a record's entry
  (`Core.Record.toJson`) holds (`toJson_vals`);
* a value alike a record's entry reads as that record, its body alike the record's (`ofJson_alike`).
-/

@[expose] public section

namespace Properties.Challenger.Entry

open Lean (Json)
open Spot (getD field?)
open Claims.Json (Same)
open FlockComplete.JsonRT (same_str same_null same_num keys_of isSome_get?_mkObj)

/-- `qsort` leaves an array of strings that strictly increase where it is. -/
theorem qsort_strs {as : Array String} (hp : as.toList.Pairwise (· < ·)) : as.qsort (· < ·) = as := by
  rw [List.pairwise_iff_getElem] at hp
  apply Verity.QSort.qsort_sorted
  intro i hi j hj
  rcases Nat.lt_trichotomy i j with hij | rfl | hij
  · have := hp i j (by simpa using hi) (by simpa using hj) hij
    simp only [Array.getElem_toList] at this
    simp [hij, this]
  · simp [String.lt_irrefl]
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
    · exact ofJson_obj_ok hkeys hnb hkind hprev hlen hhex hschema hseq hsession hsig (Flock.unhex_hexOf r.sig)
    · rw [hb, ← hnb]
      exact hbody

end Properties.Challenger.Entry
