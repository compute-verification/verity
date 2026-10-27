import Properties.Verity.Game
import Proofs.Flock.Claims.Complete.Json.Record
import Proofs.Flock.Claims.Complete.Json.Canon
import Std.Data.TreeMap.Raw.Lemmas
import Std.Data.TreeMap.Raw.WF

/-!
# The play's device log reads back

The challenger's reader parses each line of a session's release log with the verifier's parser and checks it is the
line `Flock.canon` prints of what it parsed (`event`); then `releasedRootsOf` reads each family's roots in order and
`receivedOrder` the families of the roots received. The play's device writes its log as `Flock.canon` prints the root
and release lines `challenger.py` writes (`PlayDevice.lines`), so its log reads back exactly:

* **Code**: `releasedRootsOf` on a head line and pairs of lines, each pair reading as `LineFacts` at the roots read so
  far, returns the roots pushed in order (`ReadBack.releasedRootsOf_flat`); `receivedOrder` on such a log returns the
  pairs' families (`ReadBack.receivedOrder_flat`).
* **JSON**: a printable value's canonical line within the cap reads back through `event` to an alike value
  (`ReadBack.event_canon`, from the verifier's parser reading back `canon`, `FlockComplete.JsonRT.parse_canon`), whose
  keys are the printed object's in the same order (`ReadBack.keysIn_same`) and whose string and number leaves are equal
  (`ReadBack.getD_same`).
* **The play's device**: each served release's root and release lines read as `LineFacts` (`PlayDevice.lineFacts`), so
  the session's log reads back to the served roots by family, in the order served (`PlayDevice.rootsOf_session`,
  `PlayDevice.releasedRootsOf_session`), and receives the served families in order (`PlayDevice.receivedOrder_session`).
-/

open Lean (Json JsonNumber)

namespace SampledProofs.Challenger.ReadBack

open Claims.Json (Same Good)
open FlockComplete.JsonRT

/-- Two lists, each strictly ascending under an irreflexive, asymmetric `lt`, with the same members are equal. -/
theorem eq_of_ascending {α : Type} {lt : α → α → Prop} (irr : ∀ a, ¬ lt a a) (asy : ∀ a b, lt a b → ¬ lt b a) :
    ∀ {l₁ l₂ : List α}, l₁.Pairwise lt → l₂.Pairwise lt → (∀ x, x ∈ l₁ ↔ x ∈ l₂) → l₁ = l₂
  | [], [], _, _, _ => rfl
  | [], b :: _, _, _, h => absurd ((h b).2 List.mem_cons_self) (by simp)
  | a :: _, [], _, _, h => absurd ((h a).1 List.mem_cons_self) (by simp)
  | a :: t₁, b :: t₂, h₁, h₂, h => by
    rw [List.pairwise_cons] at h₁ h₂
    have hab : a = b := by
      rcases List.mem_cons.1 ((h a).1 List.mem_cons_self) with e | ha
      · exact e
      rcases List.mem_cons.1 ((h b).2 List.mem_cons_self) with e | hb
      · exact e.symm
      exact absurd (h₁.1 b hb) (asy _ _ (h₂.1 a ha))
    subst hab
    refine congrArg _ (eq_of_ascending irr asy h₁.2 h₂.2 fun x => ⟨fun hx => ?_, fun hx => ?_⟩)
    · rcases List.mem_cons.1 ((h x).1 (List.mem_cons_of_mem _ hx)) with e | e
      · subst e; exact absurd (h₁.1 _ hx) (irr _)
      · exact e
    · rcases List.mem_cons.1 ((h x).2 (List.mem_cons_of_mem _ hx)) with e | e
      · subst e; exact absurd (h₂.1 _ hx) (irr _)
      · exact e

/-- Alike well-formed objects list one key sequence: each tree lists its keys in ascending order. -/
theorem keys_same {m : Std.TreeMap.Raw String Json} (hm : m.WF) {j : Json} (h : Same (.obj m) j) :
    ∃ n : Std.TreeMap.Raw String Json, j = .obj n ∧ n.toList.map (·.1) = m.toList.map (·.1) := by
  cases h with
  | @obj _ n _ hn hk _ =>
    refine ⟨n, rfl, eq_of_ascending (lt := fun a b => compare a b = .lt) (fun a e => by simp at e)
      (fun a b e e' => by rw [Std.OrientedCmp.gt_of_lt e] at e'; cases e') ?_ ?_ fun x => ?_⟩
    · exact List.pairwise_map.2 (Std.TreeMap.Raw.ordered_keys_toList hn)
    · exact List.pairwise_map.2 (Std.TreeMap.Raw.ordered_keys_toList hm)
    · rw [FlockComplete.JsonRT.ReadBack.mem_keys hn, FlockComplete.JsonRT.ReadBack.mem_keys hm, hk]

theorem keysIn_same {m : Std.TreeMap.Raw String Json} (hm : m.WF) {j : Json} (h : Same (.obj m) j) :
    keysIn? j = keysIn? (.obj m) := by
  obtain ⟨n, rfl, hk⟩ := keys_same hm h
  simp only [keysIn?, ← Std.TreeMap.Raw.toArray_toList, List.map_toArray, hk]

/-- A leaf: a string or a number. -/
def Leaf : Json → Prop
  | .str _ | .num _ => True
  | _ => False

theorem getD_same {L : List (String × Json)} (hd : FlockComplete.JsonRT.Distinct L) {j : Json} (h : Same (Json.mkObj L) j) {k : String}
    {v : Json} (hk : (k, v) ∈ L) (hv : Leaf v) : getD j k = v := by
  obtain ⟨v', e, hs⟩ := h.getObjVal (FlockComplete.JsonRT.getObjVal?_mkObj hd hk)
  have : v' = v := by
    cases v with
    | str s => exact FlockComplete.JsonRT.same_str hs
    | num n => exact FlockComplete.JsonRT.same_num hs
    | _ => cases hv
  simp [getD, field?, e, this, Except.toOption]

/-- **A printable value's canonical line reads back**: the verifier's `event` parses the line `canon` printed of a
printable `j` to a value alike `j`, when the line fits the reader's cap. -/
theorem event_canon {j : Json} (hg : Good j) {s : String} (hc : Flock.canon j = .ok s) (hl : s.length ≤ MAX_LOG_LINE) :
    ∃ j', event s = .ok j' ∧ Same j j' := by
  obtain ⟨j', hp, hs⟩ := FlockComplete.JsonRT.parse_canon j s hg hc
  refine ⟨j', ?_, hs⟩
  have hc' : Flock.canon j' = .ok s := (FlockComplete.JsonRT.canon_same j j' hs) ▸ hc
  unfold event
  simp [hl, hp, hc', pure, Except.pure]


/-- One root of the family `l`, pushed where `releasedRootsOf` pushes it: the first family labelled `l`. -/
def push (roots : Array (String × Array ByteArray)) (x : String × ByteArray) : Array (String × Array ByteArray) :=
  match roots.findIdx? (·.1 == x.1) with
  | some k => roots.modify k fun y => (y.1, y.2.push x.2)
  | none => roots

/-- What `releasedRootsOf` reads of one pair of lines, root line `a` and release line `b`, for a root `root` of family
`l` written in hex `h`, whose index is `n`. -/
def LineFacts (a b l h : String) (root : ByteArray) (n : Int) : Prop :=
  ∃ ra rb, event a = .ok ra ∧ event b = .ok rb ∧
    keysIn? ra = some #["event", "index", "label", "received_at", "root"] ∧
    keysIn? rb = some #["after", "event", "index", "label", "released_at"] ∧
    strOf? (getD ra "event") = some "root" ∧ strOf? (getD rb "event") = some "release" ∧
    isTime (getD ra "received_at") = true ∧ isTime (getD rb "released_at") = true ∧
    getD ra "label" = .str l ∧ getD ra "root" = .str h ∧ Flock.unhex h = .ok root ∧
    int? (getD ra "index") = some n ∧ int? (getD rb "index") = some n ∧
    strOf? (getD rb "label") = some l ∧ strOf? (getD rb "after") = some h ∧ h.length = 128

/-- Every pair of lines of `ps`, from the roots `roots` on, reads as `LineFacts` at the roots read so far. -/
def Fits : Array (String × Array ByteArray) → List (String × String × String × String × ByteArray) → Prop
  | _, [] => True
  | roots, (a, b, l, h, root) :: ps =>
    (∃ k, roots.findIdx? (·.1 == l) = some k ∧ LineFacts a b l h root ((roots[k]?.map fun x => (x.2.size : Int)).getD 0)) ∧
      Fits (push roots (l, root)) ps

def flat (ps : List (String × String × String × String × ByteArray)) : List String :=
  ps.flatMap fun (a, b, _, _, _) => [a, b]

theorem pairs_flat : ∀ ps : List (String × String × String × String × ByteArray),
    pairs (flat ps) = ps.map fun (a, b, _, _, _) => (a, b)
  | [] => by simp [flat, pairs]
  | (a, b, l, h, r) :: ps => by
    simp only [flat, List.flatMap_cons, List.cons_append, List.nil_append, List.map_cons]
    rw [pairs]
    exact congrArg _ (pairs_flat ps)

theorem length_flat : ∀ ps : List (String × String × String × String × ByteArray), (flat ps).length = 2 * ps.length
  | [] => rfl
  | _ :: ps => by
    simp only [flat, List.flatMap_cons, List.length_append, List.length_cons, List.length_nil] at *
    have := length_flat ps
    simp only [flat] at this
    omega

theorem releasedRootsOf_flat (c : LiveCommitment) (ps : List (String × String × String × String × ByteArray))
    (hf : Fits (c.families.map fun x => (x.1, (#[] : Array ByteArray))).toArray ps) :
    releasedRootsOf c (("{\"event\":\"commitment\",\"root\":\"" ++ Flock.hexOf c.root ++ "\"}") :: flat ps) =
      .ok ((ps.foldl (fun roots x => push roots (x.2.2.1, x.2.2.2.2))
        (c.families.map fun x => (x.1, (#[] : Array ByteArray))).toArray).toList.map fun x => (x.1, x.2.toList)) := by
  unfold releasedRootsOf
  have hlen : ((("{\"event\":\"commitment\",\"root\":\"" ++ Flock.hexOf c.root ++ "\"}") :: flat ps).length % 2 == 0) = false := by
    simp only [List.length_cons, length_flat]; simp
  simp only [List.head?_cons, beq_self_eq_true, ↓reduceIte, hlen, Bool.false_eq_true, List.tail_cons, pairs_flat]
  clear hlen
  generalize (c.families.map fun x => (x.1, (#[] : Array ByteArray))).toArray = R at hf ⊢
  induction ps generalizing R with
  | nil => simp [bind, Except.bind, pure, Except.pure]
  | cons x ps ih =>
    obtain ⟨a, b, l, h, root⟩ := x
    obtain ⟨⟨k, hk, ra, rb, ha, hb, k1, k2, e1, e2, t1, t2, hl, hr, hu, i1, i2, l2, a2, hh⟩, hf⟩ := hf
    simp only [List.map_cons, List.forIn_cons]
    simp only [bind, Except.bind, pure, Except.pure] at ih ⊢
    simp only [ha, hb, hl, hk, hr, hu, Except.toOption, k1, k2, e1, e2, t1, t2, i1, i2, l2, a2, hh, beq_self_eq_true,
      Bool.and_self, ↓reduceIte]
    have := ih (push R (l, root)) hf
    simp only [push, hk, Except.toOption] at this
    simpa only [List.foldl_cons, push, hk, Except.toOption] using this

theorem event_parse {a : String} {r : Json} (h : event a = .ok r) : Flock.JsonParse.parse a = .ok r := by
  unfold event at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; assumption) | cases h | (rename_i hc; cases hc)

theorem zipIdx_flat (h : String) : ∀ (ps : List (String × String × String × String × ByteArray)) (n : Nat),
    (∀ x ∈ ps, ∃ r, Flock.JsonParse.parse x.1 = .ok r ∧ getD r "label" = .str x.2.2.1) →
    ((flat ps).zipIdx (2 * n)).filterMap receivedOrder.labelAt = ps.map (·.2.2.1)
  | [], _, _ => by simp [flat]
  | (a, b, l, hx, root) :: ps, n, hp => by
    obtain ⟨r, hr, hl⟩ := hp _ List.mem_cons_self
    have ih := zipIdx_flat h ps (n + 1) fun x hx => hp x (List.mem_cons_of_mem _ hx)
    simp only [flat, List.flatMap_cons, List.cons_append, List.nil_append, List.zipIdx_cons, List.filterMap_cons,
      List.map_cons, receivedOrder.labelAt] at ih ⊢
    have e1 : 2 * n % 2 = 0 := by omega
    have e2 : (2 * n + 1) % 2 = 1 := by omega
    have e3 : 2 * n + 1 + 1 = 2 * (n + 1) := by omega
    simpa [flat, e1, e2, e3, hr, hl] using ih

theorem receivedOrder_flat (h : String) (ps : List (String × String × String × String × ByteArray))
    (hp : ∀ x ∈ ps, ∃ r, Flock.JsonParse.parse x.1 = .ok r ∧ getD r "label" = .str x.2.2.1) :
    receivedOrder (h :: flat ps) = ps.map (·.2.2.1) := by
  have := zipIdx_flat h ps 0 hp
  simpa [receivedOrder] using this

end SampledProofs.Challenger.ReadBack

namespace Verity.PlayDevice

open SampledProofs SampledProofs.Challenger SampledProofs.Challenger.ReadBack FlockComplete.JsonRT
open Claims.Json (Same Good)

abbrev Entry := String × Nat × ByteArray × Release

theorem hexOf_length (b : ByteArray) : (Flock.hexOf b).length = 2 * b.size := by
  rw [← String.length_toList, FlockComplete.JsonRT.hexOf_toList, FlockComplete.JsonRT.length_flatMap2]
  rfl

theorem requestOf_size {b : ByteArray} {l : String} {root : ByteArray} (h : requestOf b = some (l, root)) :
    root.size = 64 := by
  unfold requestOf at h
  obtain ⟨⟨l', x⟩, -, e⟩ := Option.map_eq_some_iff.1 h
  cases e
  unfold Flock.Sha512.hash
  exact Array.size_ofFn

theorem isTime_toJson (t : Nat) : isTime (Lean.toJson t) = true := by
  simp [isTime, nat?, int?, Lean.toJson, JsonNumber.fromNat]

theorem int_toJson (i : Nat) : int? (Lean.toJson i) = some (i : Int) := by
  simp [int?, Lean.toJson, JsonNumber.fromNat]

/-- **The root line reads back**: `event` parses the device's root line to an object of the root line's keys, in order,
with its event, index, label, time and root. -/
theorem rootJ_read {l : String} {i t : Nat} {root : ByteArray} (hc : (line (rootJ l i t root)).length ≤ MAX_LOG_LINE) :
    ∃ ra, event (line (rootJ l i t root)) = .ok ra ∧
      keysIn? ra = some #["event", "index", "label", "received_at", "root"] ∧
      getD ra "event" = .str "root" ∧ getD ra "index" = Lean.toJson i ∧ getD ra "label" = .str l ∧
      getD ra "received_at" = Lean.toJson t ∧ getD ra "root" = .str (Flock.hexOf root) := by
  have hd : FlockComplete.JsonRT.Distinct [("event", Json.str "root"), ("index", Lean.toJson i), ("label", Json.str l),
      ("received_at", Lean.toJson t), ("root", Json.str (Flock.hexOf root))] := by simp [FlockComplete.JsonRT.Distinct] <;> decide
  have hg : Good (rootJ l i t root) := FlockComplete.JsonRT.good_mkObj hd (by
    simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq]
    exact ⟨.str _, FlockComplete.JsonRT.good_toJson _, .str _, FlockComplete.JsonRT.good_toJson _, .str _⟩)
  obtain ⟨s, hs⟩ := FlockComplete.JsonRT.canon_ok _ hg
  have hl : line (rootJ l i t root) = s := by simp [line, hs, Except.toOption]
  rw [hl] at hc ⊢
  obtain ⟨ra, he, hsame⟩ := event_canon hg hs hc
  refine ⟨ra, he, ?_, getD_same hd hsame (by simp) trivial, getD_same hd hsame (by simp) trivial,
    getD_same hd hsame (by simp) trivial, getD_same hd hsame (by simp) trivial, getD_same hd hsame (by simp) trivial⟩
  rw [keysIn_same Std.TreeMap.Raw.WF.ofList hsame]
  with_unfolding_all rfl

/-- **The release line reads back**, as `rootJ_read`. -/
theorem releaseJ_read {l : String} {i t : Nat} {root : ByteArray}
    (hc : (line (releaseJ l i t root)).length ≤ MAX_LOG_LINE) :
    ∃ rb, event (line (releaseJ l i t root)) = .ok rb ∧
      keysIn? rb = some #["after", "event", "index", "label", "released_at"] ∧
      getD rb "after" = .str (Flock.hexOf root) ∧ getD rb "event" = .str "release" ∧ getD rb "index" = Lean.toJson i ∧
      getD rb "label" = .str l ∧ getD rb "released_at" = Lean.toJson t := by
  have hd : FlockComplete.JsonRT.Distinct [("after", Json.str (Flock.hexOf root)), ("event", Json.str "release"), ("index", Lean.toJson i),
      ("label", Json.str l), ("released_at", Lean.toJson t)] := by simp [FlockComplete.JsonRT.Distinct] <;> decide
  have hg : Good (releaseJ l i t root) := FlockComplete.JsonRT.good_mkObj hd (by
    simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq]
    exact ⟨.str _, .str _, FlockComplete.JsonRT.good_toJson _, .str _, FlockComplete.JsonRT.good_toJson _⟩)
  obtain ⟨s, hs⟩ := FlockComplete.JsonRT.canon_ok _ hg
  have hl : line (releaseJ l i t root) = s := by simp [line, hs, Except.toOption]
  rw [hl] at hc ⊢
  obtain ⟨rb, he, hsame⟩ := event_canon hg hs hc
  refine ⟨rb, he, ?_, getD_same hd hsame (by simp) trivial, getD_same hd hsame (by simp) trivial,
    getD_same hd hsame (by simp) trivial, getD_same hd hsame (by simp) trivial, getD_same hd hsame (by simp) trivial⟩
  rw [keysIn_same Std.TreeMap.Raw.WF.ofList hsame]
  with_unfolding_all rfl

/-- What the reader checks of one served release's two lines holds. -/
theorem lineFacts {l : String} {i t t' : Nat} {root : ByteArray} (h64 : root.size = 64)
    (hc : (line (rootJ l i t root)).length ≤ MAX_LOG_LINE) (hc' : (line (releaseJ l i t' root)).length ≤ MAX_LOG_LINE) :
    ReadBack.LineFacts (line (rootJ l i t root)) (line (releaseJ l i t' root)) l (Flock.hexOf root) root i := by
  obtain ⟨ra, ha, k1, e1, i1, l1, t1, r1⟩ := rootJ_read hc
  obtain ⟨rb, hb, k2, a2, e2, i2, l2, t2⟩ := releaseJ_read hc'
  refine ⟨ra, rb, ha, hb, k1, k2, by rw [e1]; rfl, by rw [e2]; rfl, by rw [t1]; exact isTime_toJson _,
    by rw [t2]; exact isTime_toJson _, l1, r1, FlockComplete.JsonRT.unhex_hexOf _, by rw [i1]; exact int_toJson _,
    by rw [i2]; exact int_toJson _, by rw [l2]; rfl, by rw [a2]; rfl, by rw [hexOf_length, h64]⟩


/-- The labels of the reader's roots array stay the layout's: `push` changes only a family's roots. -/
theorem map_fst_push (R : Array (String × Array ByteArray)) (x : String × ByteArray) :
    (ReadBack.push R x).map (·.1) = R.map (·.1) := by
  unfold ReadBack.push
  split
  · apply Array.ext (by simp)
    intro j h₁ h₂
    simp only [Array.getElem_map, Array.getElem_modify]
    split <;> rfl
  · rfl

theorem findIdx_fst (R : Array (String × Array ByteArray)) (l : String) :
    R.findIdx? (·.1 == l) = (R.map (·.1)).findIdx? (· == l) := by
  rw [Array.findIdx?_map]; rfl

/-- The reader's roots array after the served releases `pre`: the layout's labels, and at each label's first place the
roots served for it, in order. -/
def RootsInv (fams : Families) (R : Array (String × Array ByteArray)) (pre : List Entry) : Prop :=
  R.map (·.1) = (fams.map (·.1)).toArray ∧
    ∀ l k, R.findIdx? (·.1 == l) = some k → R[k]?.map (·.2.toList) = some ((pre.filter (·.1 == l)).map (·.2.2.1))

theorem inv_init (fams : Families) : RootsInv fams (fams.map fun x => (x.1, (#[] : Array ByteArray))).toArray [] := by
  refine ⟨by simp, fun l k hk => ?_⟩
  obtain ⟨h, -, -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hk
  simp [Array.getElem?_eq_getElem h]

theorem findIdx_isSome {fams : Families} {R : Array (String × Array ByteArray)} (hR : R.map (·.1) = (fams.map (·.1)).toArray)
    {l : String} (hl : (fams.find? (·.1 == l)).isSome) : ∃ k, R.findIdx? (·.1 == l) = some k := by
  rw [findIdx_fst, hR]
  rw [Option.isSome_iff_exists] at hl
  obtain ⟨x, hx⟩ := hl
  have hm := List.mem_of_find?_eq_some hx
  have hp := List.find?_some hx
  cases e : (fams.map (·.1)).toArray.findIdx? (· == l) with
  | some k => exact ⟨k, rfl⟩
  | none =>
    rw [Array.findIdx?_eq_none_iff] at e
    exact absurd (e x.1 (by simpa using List.mem_map_of_mem (f := (·.1)) hm)) (by simpa using hp)

theorem inv_push {fams : Families} {R : Array (String × Array ByteArray)} {pre : List Entry} (hI : RootsInv fams R pre)
    (e : Entry) (hl : (fams.find? (·.1 == e.1)).isSome) : RootsInv fams (ReadBack.push R (e.1, e.2.2.1)) (pre ++ [e]) := by
  obtain ⟨hR, hv⟩ := hI
  obtain ⟨k', hk'⟩ := findIdx_isSome hR hl
  refine ⟨by rw [map_fst_push, hR], fun l k hk => ?_⟩
  have hk0 : R.findIdx? (·.1 == l) = some k := by rw [findIdx_fst, ← map_fst_push R (e.1, e.2.2.1), ← findIdx_fst]; exact hk
  have hold := hv l k hk0
  obtain ⟨hlt', hp', -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hk'
  obtain ⟨hlt, hp, -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hk0
  unfold ReadBack.push
  simp only [hk']
  rw [Array.getElem?_modify]
  by_cases hkk : k' = k
  · subst hkk
    have hle : e.1 = l := by
      have := (beq_iff_eq.1 hp').symm.trans (beq_iff_eq.1 hp)
      exact this
    subst hle
    simp only [↓reduceIte, Option.map_map]
    rw [Array.getElem?_eq_getElem hlt] at hold ⊢
    simp only [Option.map_some, Option.some.injEq] at hold ⊢
    simp [Function.comp, Array.toList_push, hold, List.filter_append]
  · have hne : (e.1 == l) = false := by
      rw [beq_eq_false_iff_ne]
      rintro rfl
      exact hkk (Option.some.inj (hk'.symm.trans hk0))
    simp [hkk, hold, List.filter_append, hne]

theorem inv_foldl {fams : Families} : ∀ (S pre : List Entry) (R : Array (String × Array ByteArray)), RootsInv fams R pre →
    (∀ e ∈ S, (fams.find? (·.1 == e.1)).isSome) →
    RootsInv fams (S.foldl (fun R e => push R (e.1, e.2.2.1)) R) (pre ++ S)
  | [], pre, R, hI, _ => by simpa using hI
  | e :: S, pre, R, hI, hS => by
    have := inv_foldl S (pre ++ [e]) _ (inv_push hI e (hS e List.mem_cons_self))
      fun x hx => hS x (List.mem_cons_of_mem _ hx)
    simpa using this

/-- What `served` establishes of its entry `e` after the entries `pre`: a family of the layout, its index the count of
that family's entries before it, a 64-byte root, and both lines within the reader's cap at their times. -/
def ServedAt (D : PlayDevice) (pre : List Entry) (e : Entry) : Prop :=
  (D.families.find? (·.1 == e.1)).isSome ∧ e.2.1 = (pre.filter (·.1 == e.1)).length ∧ e.2.2.1.size = 64 ∧
    (line (rootJ e.1 e.2.1 (D.time (2 * pre.length + 1)) e.2.2.1)).length ≤ MAX_LOG_LINE ∧
    (line (releaseJ e.1 e.2.1 (D.time (2 * pre.length + 2)) e.2.2.1)).length ≤ MAX_LOG_LINE

theorem foldl_inv {α β : Type} (f : β → α → β) (P : β → Prop) (h : ∀ b a, P b → P (f b a)) :
    ∀ (l : List α) (b : β), P b → P (l.foldl f b)
  | [], _, hb => hb
  | a :: l, b, hb => foldl_inv f P h l (f b a) (h b a hb)

theorem served_cond (D : PlayDevice) (rs : List Release) :
    ∀ k e, (D.served rs)[k]? = some e → ServedAt D ((D.served rs).take k) e := by
  unfold served
  refine foldl_inv _ (fun acc => ∀ k e, acc[k]? = some e → ServedAt D (acc.take k) e) ?_ rs [] (by simp)
  intro acc r hP
  dsimp only
  split
  · exact hP
  · rename_i l root heq
    split
    · rename_i hc
      simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
      obtain ⟨⟨hf, hc1⟩, hc2⟩ := hc
      intro k e hk
      rcases Nat.lt_or_ge k acc.length with hlt | hge
      · rw [List.getElem?_append_left hlt] at hk
        rw [List.take_append_of_le_length (Nat.le_of_lt hlt)]
        exact hP k e hk
      · rcases Nat.eq_or_lt_of_le hge with heq' | hgt
        · subst heq'
          rw [List.getElem?_append_right (Nat.le_refl _)] at hk
          simp only [Nat.sub_self, List.getElem?_cons_zero, Option.some.injEq] at hk
          subst hk
          rw [List.take_left' rfl]
          exact ⟨Option.isSome_of_any hf, rfl, requestOf_size heq, hc1, hc2⟩
        · rw [List.getElem?_eq_none (by simp; omega)] at hk
          cases hk
    · exact hP

/-- The reader's view of the served entries `S` from place `n` on: each entry's two lines, its family, its root in hex
and its root. -/
def entries (D : PlayDevice) (S : List Entry) (n : Nat) : List (String × String × String × String × ByteArray) :=
  (S.zipIdx n).map fun ((l, i, root, _), k) =>
    (line (rootJ l i (D.time (2 * k + 1)) root), line (releaseJ l i (D.time (2 * k + 2)) root), l, Flock.hexOf root, root)

theorem entries_cons (D : PlayDevice) (e : Entry) (S : List Entry) (n : Nat) :
    entries D (e :: S) n = (line (rootJ e.1 e.2.1 (D.time (2 * n + 1)) e.2.2.1),
      line (releaseJ e.1 e.2.1 (D.time (2 * n + 2)) e.2.2.1), e.1, Flock.hexOf e.2.2.1, e.2.2.1) :: entries D S (n + 1) := by
  simp [entries, List.zipIdx_cons]

theorem flat_entries (D : PlayDevice) : ∀ (S : List Entry) (n : Nat), ReadBack.flat (entries D S n) =
    ((S.zipIdx n).map fun ((l, i, root, _), k) =>
      [line (rootJ l i (D.time (2 * k + 1)) root), line (releaseJ l i (D.time (2 * k + 2)) root)]).flatten
  | [], _ => rfl
  | e :: S, n => by
    rw [entries_cons]
    simp only [ReadBack.flat, List.flatMap_cons, List.zipIdx_cons, List.map_cons, List.flatten_cons] at *
    rw [← flat_entries D S (n + 1)]
    rfl

theorem foldl_entries (D : PlayDevice) : ∀ (S : List Entry) (n : Nat) (R : Array (String × Array ByteArray)),
    (entries D S n).foldl (fun roots x => push roots (x.2.2.1, x.2.2.2.2)) R =
      S.foldl (fun R e => push R (e.1, e.2.2.1)) R
  | [], _, _ => rfl
  | e :: S, n, R => by
    rw [entries_cons, List.foldl_cons, List.foldl_cons]
    exact foldl_entries D S (n + 1) _

theorem labels_entries (D : PlayDevice) : ∀ (S : List Entry) (n : Nat), (entries D S n).map (·.2.2.1) = S.map (·.1)
  | [], _ => rfl
  | e :: S, n => by rw [entries_cons, List.map_cons, List.map_cons, labels_entries D S (n + 1)]

theorem fits_entries (D : PlayDevice) : ∀ (S pre : List Entry) (R : Array (String × Array ByteArray)),
    RootsInv D.families R pre → (∀ k e, S[k]? = some e → ServedAt D (pre ++ S.take k) e) → ReadBack.Fits R (entries D S pre.length)
  | [], _, _, _, _ => by simp [entries, ReadBack.Fits]
  | e :: S, pre, R, hI, hS => by
    rw [entries_cons]
    have hc := hS 0 e rfl
    simp only [List.take_zero, List.append_nil] at hc
    obtain ⟨hf, hi, h64, hc1, hc2⟩ := hc
    obtain ⟨k, hk⟩ := findIdx_isSome hI.1 hf
    refine ⟨⟨k, hk, ?_⟩, ?_⟩
    · obtain ⟨hlt, -, -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hk
      have hv := hI.2 _ _ hk
      rw [Array.getElem?_eq_getElem hlt] at hv ⊢
      simp only [Option.map_some, Option.some.injEq, Option.getD_some] at hv ⊢
      have hn : ((R[k].2.size : Nat) : Int) = (e.2.1 : Int) := by
        rw [hi, ← Array.length_toList, hv, List.length_map]
      rw [hn]
      exact lineFacts h64 hc1 hc2
    · have := fits_entries D S (pre ++ [e]) _ (inv_push hI e hf) fun k x hx => by
        have := hS (k + 1) x (by simpa using hx)
        simpa using this
      simpa using this

theorem find_roots {fams : Families} {R : Array (String × Array ByteArray)} {S : List Entry} (hI : RootsInv fams R S)
    (hS : ∀ e ∈ S, (fams.find? (·.1 == e.1)).isSome) (l : String) :
    (((R.toList.map fun x => (x.1, x.2.toList)).find? (·.1 == l)).map (·.2)).getD [] =
      (S.filter (·.1 == l)).map (·.2.2.1) := by
  rw [List.find?_map]
  cases hk : R.findIdx? (·.1 == l) with
  | none =>
    rw [Array.findIdx?_eq_none_iff] at hk
    have h1 : R.toList.find? ((fun x => x.1 == l) ∘ fun x => (x.1, x.2.toList)) = none := by
      rw [List.find?_eq_none]
      intro x hx
      simpa using hk x (Array.mem_toList_iff.1 hx)
    have h2 : S.filter (·.1 == l) = [] := by
      rw [List.filter_eq_nil_iff]
      intro e he hl
      obtain ⟨k, hk'⟩ := findIdx_isSome hI.1 (hS e he)
      rw [beq_iff_eq.1 hl] at hk'
      rw [← Array.findIdx?_eq_none_iff] at hk
      rw [hk] at hk'
      cases hk'
    rw [h1, h2]
    rfl
  | some k =>
    obtain ⟨hlt, hp, hpre⟩ := Array.findIdx?_eq_some_iff_getElem.1 hk
    have hv := hI.2 _ _ hk
    rw [Array.getElem?_eq_getElem hlt] at hv
    simp only [Option.map_some, Option.some.injEq] at hv
    have h1 : R.toList.find? ((fun x => x.1 == l) ∘ fun x => (x.1, x.2.toList)) = some R[k] := by
      rw [List.find?_eq_some_iff_getElem]
      refine ⟨by simpa using hp, k, by simpa using hlt, by simp, fun j hj => by simpa using hpre j hj⟩
    rw [h1]
    simpa using hv

/-- **The device's release log reads back**: under a commitment of the layout's families, `releasedRootsOf` reads the
log the device writes, its head line then `lines`, to the served roots: each family's, in the order served. -/
theorem releasedRootsOf_play (D : PlayDevice) (rs : List Release) (c : LiveCommitment) (hc : c.families = D.families) :
    ∃ roots, releasedRootsOf c (headLine c.root :: D.lines rs) = .ok roots ∧
      ∀ l, ((roots.find? (·.1 == l)).map (·.2)).getD [] = ((D.served rs).filter (·.1 == l)).map (·.2.2.1) := by
  have hcond := served_cond D rs
  have hS : ∀ e ∈ D.served rs, (D.families.find? (·.1 == e.1)).isSome := fun e he => by
    obtain ⟨k, hk⟩ := List.getElem?_of_mem he
    exact (hcond k e hk).1
  have hinit : RootsInv D.families (c.families.map fun x => (x.1, (#[] : Array ByteArray))).toArray [] := hc ▸ inv_init _
  have hf : ReadBack.Fits (c.families.map fun x => (x.1, (#[] : Array ByteArray))).toArray (entries D (D.served rs) 0) :=
    fits_entries D (D.served rs) [] _ hinit (by simpa using hcond)
  have hr := releasedRootsOf_flat c _ hf
  rw [foldl_entries, flat_entries] at hr
  refine ⟨_, hr, find_roots (by simpa using inv_foldl _ [] _ hinit hS) hS⟩

/-- **The device's log receives the served families in order**: `receivedOrder` reads the served releases' families. -/
theorem receivedOrder_play (D : PlayDevice) (rs : List Release) (h : String) :
    receivedOrder (h :: D.lines rs) = (D.served rs).map (·.1) := by
  have hcond := served_cond D rs
  have := receivedOrder_flat h (entries D (D.served rs) 0) fun x hx => by
    obtain ⟨⟨e, k⟩, hm, rfl⟩ := List.mem_map.1 hx
    have hk : (D.served rs)[k]? = some e := List.mem_zipIdx_iff_getElem?.1 hm
    obtain ⟨-, -, -, hc1, -⟩ := hcond k e hk
    have hlt : k < (D.served rs).length := (List.getElem?_eq_some_iff.1 hk).1
    simp only [List.length_take, Nat.min_eq_left (Nat.le_of_lt hlt)] at hc1
    obtain ⟨ra, ha, -, -, -, hl, -⟩ := rootJ_read hc1
    obtain ⟨l, i, root, r⟩ := e
    exact ⟨ra, event_parse ha, hl⟩
  rw [flat_entries, labels_entries] at this
  exact this


theorem session_commitment (D : PlayDevice) (rs : List Release) : (D.session rs).commitment = (D.base rs).commitment :=
  rfl

theorem commitment_families {D : PlayDevice} {rs : List Release} {c : LiveCommitment}
    (h : (D.base rs).commitment = some c) : c.families = D.families := by
  unfold Properties.Challenger.Session.commitment at h
  split at h
  · cases h; rfl
  · cases h

/-- **The device's session log reads back**: under its commitment `c`, `releasedRootsOf` reads the session's log to the
served roots: each family's, in the order served. -/
theorem releasedRootsOf_session {D : PlayDevice} {rs : List Release} {c : LiveCommitment}
    (hc : (D.session rs).commitment = some c) :
    ∃ roots, releasedRootsOf c (D.session rs).log = .ok roots ∧
      ∀ l, ((roots.find? (·.1 == l)).map (·.2)).getD [] = ((D.served rs).filter (·.1 == l)).map (·.2.2.1) := by
  have hb : (D.base rs).commitment = some c := hc
  have hlog : (D.session rs).log = headLine c.root :: D.lines rs := by
    show headLine (((D.base rs).commitment.map (·.root)).getD .empty) :: D.lines rs = _
    rw [hb]
    rfl
  rw [hlog]
  exact releasedRootsOf_play D rs c (commitment_families hb)

/-- **The device's session shows the served roots**: family `l`'s roots on its log are the roots served for `l`, in
order. -/
theorem rootsOf_session {D : PlayDevice} {rs : List Release} {c : LiveCommitment}
    (hc : (D.session rs).commitment = some c) (l : String) :
    (D.session rs).rootsOf l = ((D.served rs).filter (·.1 == l)).map (·.2.2.1) := by
  obtain ⟨roots, hr, hf⟩ := releasedRootsOf_session hc
  unfold Properties.Challenger.Session.rootsOf
  rw [hc]
  simp only [hr]
  exact hf l

/-- **The device's session receives the served families in order** (`receivedOrder`). -/
theorem receivedOrder_session (D : PlayDevice) (rs : List Release) :
    receivedOrder (D.session rs).log = (D.served rs).map (·.1) :=
  receivedOrder_play D rs _

end Verity.PlayDevice
