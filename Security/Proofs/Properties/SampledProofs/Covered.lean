import Properties.SampledProofs.Covered
import Proofs.Properties.SampledProofs
import Proofs.Properties.Composition
import Std.Data.TreeMap.Raw.WF

/-! # Proofs of R7's coverage at one-stage v2 (`Properties.SampledProofs.Covered`)

An acceptance of `OneStageZk.checkBlobsAt` is a run whose verdicts all report the verifier's registered reads
(`Covered.run_out`: each verdict's `registered` is `Flock.Registered.finding` of them) and v1's acceptance on those
verdicts (`Covered.checkBlobsAt_accept`), which gives R7 and `UnitsProved` (`acceptMeans`). R7 holds the record's
`registered` entries to the verifier's values and roots; `UnitsProved` holds the roots the verdicts report
(`Record.sessionRegistered`, a fold whose invariant is `Covered.RegInv`) to the record's (`Record.registeredRoots`). A
value R7 holds is then reported, by a verdict, under a root that canonical JSON equality (`jeq`, injective on strings by
`Properties.Composition.Esc.canonStr_split`) makes the held one. `GatesHeld` reads `HeldCovered` at the port
`Bits.rowBit` names (`Covered.rowBit_port`), which `Flock.Registered.checkComplete` finds among the registered reads.
-/

open Lean (Json)
open SampledProofs

namespace Properties.SampledProofs

open OneStage (Commitment Public)

namespace Covered

/-! ## Canonical JSON equality on strings -/

theorem canon_str (s : String) : Flock.canon (.str s) = .ok (Flock.canonStr s) := by
  simp only [Flock.canon]
  try rfl

theorem canon_null : Flock.canon .null = .ok "null" := by
  simp only [Flock.canon]
  try rfl

theorem canonStr_inj {a b : String} (h : Flock.canonStr a = Flock.canonStr b) : a = b :=
  (Properties.Composition.Esc.canonStr_split (r := []) (r' := []) (by rw [h])).1

theorem jeq_str_iff {x : Json} {b : String} : jeq x (.str b) = true ↔ Flock.canon x = .ok (Flock.canonStr b) := by
  unfold jeq
  rw [canon_str]
  cases Flock.canon x <;> simp

theorem jeq_str {x : Json} {b c : String} (hb : jeq x (.str b) = true) (hc : jeq x (.str c) = true) : b = c := by
  rw [jeq_str_iff] at hb hc
  rw [hb] at hc
  exact canonStr_inj (Except.ok.inj hc)

theorem jeq_str_str {a b : String} : jeq (.str a) (.str b) = true ↔ a = b := by
  rw [jeq_str_iff, canon_str]
  exact ⟨fun h => canonStr_inj (Except.ok.inj h), fun h => by rw [h]⟩

theorem null_ne_canonStr (s : String) : "null" ≠ Flock.canonStr s := by
  intro h
  have h2 := congrArg String.toList h
  rw [Properties.Composition.Esc.canonStr_toList] at h2
  have h3 : "null".toList = ['n', 'u', 'l', 'l'] := by first | rfl | decide | simp
  rw [h3] at h2
  exact absurd (List.cons.inj h2).1 (by decide)

theorem jeq_null_str (s : String) : jeq .null (.str s) = false := by
  cases h : jeq .null (.str s)
  · rfl
  · rw [jeq_str_iff, canon_null] at h
    exact absurd (Except.ok.inj h) (null_ne_canonStr s)

/-! ## Objects `Json.mkObj` builds: a key reads its last pair -/

theorem findSome_eq (k : String) (f : String × Json → Option Json)
    (hf : ∀ x, f x = if compare x.1 k = .eq then some x.2 else none) :
    ∀ m : List (String × Json), m.findSome? f = (m.find? (·.1 == k)).map (·.2)
  | [] => rfl
  | x :: m => by
    rw [List.findSome?_cons, List.find?_cons, hf x]
    by_cases h : x.1 = k
    · have h1 : compare x.1 k = .eq := Std.LawfulEqOrd.compare_eq_iff_eq.2 h
      have h2 : (x.1 == k) = true := beq_iff_eq.2 h
      simp only [h1, h2, ↓reduceIte, Option.map_some]
    · have h1 : ¬ compare x.1 k = .eq := fun h' => h (Std.LawfulEqOrd.compare_eq_iff_eq.1 h')
      have h2 : (x.1 == k) = false := by simpa using h
      simp only [h1, h2, ↓reduceIte]
      exact findSome_eq k f hf m

theorem getElem?_ofList (l : List (String × Json)) (k : String) :
    (Std.TreeMap.Raw.ofList l compare)[k]? = (l.reverse.find? (·.1 == k)).map (·.2) := by
  rw [Std.TreeMap.Raw.ofList_eq_insertMany_empty, Std.TreeMap.Raw.getElem?_insertMany_list Std.TreeMap.Raw.WF.emptyc,
    Std.TreeMap.Raw.getElem?_emptyc, Option.or_none, List.findSomeRev?_eq_findSome?_reverse]
  exact findSome_eq k _ (fun ⟨_, _⟩ => rfl) l.reverse

theorem field_mkObj (l : List (String × Json)) (k : String) :
    field? (Json.mkObj l) k = (l.reverse.find? (·.1 == k)).map (·.2) := by
  unfold field?
  simp only [Json.mkObj, Json.getObjVal?, Std.TreeMap.Raw.get?_eq_getElem?]
  rw [getElem?_ofList]
  cases (l.reverse.find? (·.1 == k)).map (·.2) <;> rfl

theorem getD_mkObj (l : List (String × Json)) (k : String) :
    getD (Json.mkObj l) k = ((l.reverse.find? (·.1 == k)).map (·.2)).getD .null := by
  unfold getD
  rw [field_mkObj]

theorem find_reverse_of_unique {l : List (String × Json)} {k : String} {v : Json} (hm : (k, v) ∈ l)
    (hu : ∀ x ∈ l, x.1 = k → x.2 = v) : (l.reverse.find? (·.1 == k)).map (·.2) = some v := by
  cases hf : l.reverse.find? (·.1 == k) with
  | none =>
    have := List.find?_eq_none.1 hf (k, v) (List.mem_reverse.2 hm)
    simp at this
  | some x =>
    have hx := List.mem_reverse.1 (List.mem_of_find?_eq_some hf)
    have hk : x.1 = k := by simpa using List.find?_some hf
    simp only [Option.map_some, hu x hx hk]

theorem getD_mkObj_of_unique {l : List (String × Json)} {k : String} {v : Json} (hm : (k, v) ∈ l)
    (hu : ∀ x ∈ l, x.1 = k → x.2 = v) : getD (Json.mkObj l) k = v := by
  rw [getD_mkObj, find_reverse_of_unique hm hu, Option.getD_some]

theorem mem_entries_mkObj {l : List (String × Json)} {k : String} {v : Json} :
    (k, v) ∈ entries (Json.mkObj l) ↔ (l.reverse.find? (·.1 == k)).map (·.2) = some v := by
  show (k, v) ∈ (Std.TreeMap.Raw.ofList l compare).toArray ↔ _
  rw [Std.TreeMap.Raw.mem_toArray_iff_getElem?_eq_some Std.TreeMap.Raw.WF.ofList, getElem?_ofList]

/-! ## A verdict's `registered`: `Flock.Registered.finding` -/

/-- A port's entry in `Flock.Registered.finding`. -/
def portJ (p : Flock.Registered.Port) : Json :=
  Json.mkObj [("value", Json.str p.value), ("root", Json.str (Flock.hexOf p.root))]

theorem finding_eq (own : Array Flock.Registered.Port) :
    Flock.Registered.finding own = Json.mkObj (own.toList.map fun p => (p.name, portJ p)) := rfl

theorem portJ_value (p : Flock.Registered.Port) : getD (portJ p) "value" = .str p.value := by
  unfold portJ
  exact getD_mkObj_of_unique (by simp) (by simp)

theorem portJ_root (p : Flock.Registered.Port) : getD (portJ p) "root" = .str (Flock.hexOf p.root) := by
  unfold portJ
  exact getD_mkObj_of_unique (by simp) (by simp)

theorem mem_entries_finding {own : Array Flock.Registered.Port} {k : String} {e : Json}
    (h : (k, e) ∈ entries (Flock.Registered.finding own)) : ∃ p ∈ own, k = p.name ∧ e = portJ p := by
  rw [finding_eq, mem_entries_mkObj, Option.map_eq_some_iff] at h
  obtain ⟨x, hx, rfl⟩ := h
  have hk : x.1 = k := by simpa using List.find?_some hx
  have hm := List.mem_reverse.1 (List.mem_of_find?_eq_some hx)
  obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hm
  exact ⟨p, Array.mem_toList_iff.1 hp, hk.symm, rfl⟩

theorem entries_finding_of_unique {own : Array Flock.Registered.Port} {q : Flock.Registered.Port} (hq : q ∈ own)
    (hu : ∀ q' ∈ own, q'.name = q.name → q' = q) : (q.name, portJ q) ∈ entries (Flock.Registered.finding own) := by
  rw [finding_eq, mem_entries_mkObj]
  apply find_reverse_of_unique
  · exact List.mem_map.2 ⟨q, Array.mem_toList_iff.2 hq, rfl⟩
  · intro x hx hx1
    obtain ⟨q', hq', rfl⟩ := List.mem_map.1 hx
    exact congrArg portJ (hu q' (Array.mem_toList_iff.1 hq') hx1)

/-- The `(value, root)` pair `Record.sessionRegistered` reads off one entry of a verdict's `registered`. -/
def entryPair (x : String × Json) : Option (String × Json) :=
  match getD x.2 "value" with
  | .str v => some (v, getD x.2 "root")
  | _ => none

theorem entryPair_portJ (k : String) (q : Flock.Registered.Port) :
    entryPair (k, portJ q) = some (q.value, .str (Flock.hexOf q.root)) := by
  simp only [entryPair, portJ_value, portJ_root]

theorem entryPair_portJ_inv {k : String} {q : Flock.Registered.Port} {z : String × Json}
    (h : entryPair (k, portJ q) = some z) : z = (q.value, .str (Flock.hexOf q.root)) := by
  rw [entryPair_portJ] at h
  exact (Option.some.inj h).symm

/-! ## Lists of distinct keys -/

theorem mem_of_nodup_length {α : Type} {l m : List α} (hl : l.Nodup) (hsub : ∀ a ∈ l, a ∈ m)
    (hlen : m.length ≤ l.length) {a : α} (ha : a ∈ m) : a ∈ l := by
  refine Classical.byContradiction fun hn => ?_
  have h1 : (a :: l).Nodup := List.nodup_cons.2 ⟨hn, hl⟩
  have h2 := h1.length_le_of_subset (l₂ := m) (by
    intro x hx
    rcases List.mem_cons.1 hx with rfl | hx
    · exact ha
    · exact hsub x hx)
  simp only [List.length_cons] at h2
  omega

theorem nodup_eraseDups_le : ∀ (n : Nat) (l : List String), l.length ≤ n → l.eraseDups.Nodup
  | _, [], _ => by rw [List.eraseDups_nil]; exact List.nodup_nil
  | 0, _ :: _, h => by simp at h
  | n + 1, a :: l, h => by
    rw [List.eraseDups_cons]
    refine List.nodup_cons.2 ⟨?_, nodup_eraseDups_le n _ (Nat.le_trans (List.length_filter_le _ l) ?_)⟩
    · rw [List.mem_eraseDups, List.mem_filter]
      simp
    · simp only [List.length_cons] at h
      omega

theorem nodup_eraseDups (l : List String) : l.eraseDups.Nodup := nodup_eraseDups_le l.length l (Nat.le_refl _)

theorem eq_of_nodup_map {α β : Type} (f : α → β) : ∀ (l : List α), (l.map f).Nodup →
    ∀ x ∈ l, ∀ y ∈ l, f x = f y → x = y
  | [], _, _, hx, _, _, _ => absurd hx List.not_mem_nil
  | a :: l, h, x, hx, y, hy, hxy => by
    rw [List.map_cons, List.nodup_cons] at h
    obtain ⟨hn, hl⟩ := h
    rcases List.mem_cons.1 hx with hxa | hxl <;> rcases List.mem_cons.1 hy with hya | hyl
    · rw [hxa, hya]
    · rw [hxa] at hxy
      exact absurd (List.mem_map.2 ⟨y, hyl, hxy.symm⟩) hn
    · rw [hya] at hxy
      exact absurd (List.mem_map.2 ⟨x, hxl, hxy⟩) hn
    · exact eq_of_nodup_map f l hl x hxl y hyl hxy

theorem nodup_filter_keys {acc : Array (String × Json)} {P : String × Json → Bool}
    (h : (acc.toList.map (·.1)).Nodup) : ((acc.filter P).toList.map (·.1)).Nodup := by
  rw [Array.toList_filter]
  exact List.Nodup.sublist (List.filter_sublist.map _) h

theorem sameRoots_mem {a b : Array (String × Json)} (h : SameRoots a b) (hn : (a.toList.map (·.1)).Nodup)
    {y : String × Json} (hy : y ∈ b) : ∃ x ∈ a, x.1 = y.1 := by
  obtain ⟨hsz, hsm⟩ := h
  have hk := mem_of_nodup_length (m := b.toList.map (·.1)) hn
    (fun k hk => by
      obtain ⟨x, hx, rfl⟩ := List.mem_map.1 hk
      obtain ⟨y', hy', hxy, -⟩ := hsm x (Array.mem_toList_iff.1 hx)
      exact List.mem_map.2 ⟨y', Array.mem_toList_iff.2 hy', hxy.symm⟩)
    (by simp only [List.length_map, Array.length_toList]; exact Nat.le_of_eq hsz.symm)
    (List.mem_map.2 ⟨y, Array.mem_toList_iff.2 hy, rfl⟩)
  obtain ⟨x, hx, hx1⟩ := List.mem_map.1 hk
  exact ⟨x, Array.mem_toList_iff.1 hx, hx1⟩

/-! ## The acceptance: the run and v1's check -/

theorem bind_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : (x >>= f) = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error _ => cases h
  | ok a => exact ⟨a, rfl, h⟩

theorem checkAt_check {p : OneStage.Params} {c : Commitment} {ξ : Option ByteArray} {e : OneStage.Evidence}
    {pub : Public} {prof : Core.Integrity} (h : OneStage.checkAt p c ξ e = .accept pub prof) :
    OneStage.check p c e = .accept pub prof := by
  unfold OneStage.checkAt at h
  split at h
  · cases h
  · exact h

theorem checkBlobsAt_accept {p : OneStageZk.Params} {c : Commitment} {ξ : ByteArray} {e : OneStageZk.Evidence}
    {bs : Core.Blobs} {pub : Public} {prof : Core.Integrity}
    (h : OneStageZk.checkBlobsAt p c ξ e bs = .accept pub prof) :
    ∃ r, OneStageZk.run p.own e.pub e.sessions (some e.base.provedUnits) bs = .ok r ∧ r.all = true ∧
      OneStage.check p.base c { e.base with verdicts := r.out } = .accept pub prof := by
  unfold OneStageZk.checkBlobsAt at h
  split at h
  · cases h
  · rename_i r hr
    obtain ⟨⟨⟩, -, hrun⟩ := bind_ok hr
    cases hc : OneStage.checkAt p.base c (some ξ) { e.base with verdicts := r.out } with
    | accept pub' prof' =>
      rw [hc] at h
      dsimp only at h
      split at h
      · cases h
        exact ⟨r, hrun, by assumption, checkAt_check hc⟩
      · cases h
    | reject why => rw [hc] at h; first | cases h | simp at h
    | undecided why => rw [hc] at h; first | cases h | simp at h

/-- A verdict reports the verifier's registered reads: its `registered` is their `Flock.Registered.finding`, null when
the verifier holds none. -/
def RegOf (rd : OneStageZk.Reads) (w : String × Json) : Prop :=
  getD w.2 "registered" = (rd.registered.map Flock.Registered.finding).getD .null

theorem verdictJson_registered (rd : OneStageZk.Reads) (programSha : Option String) (s : OneStageZk.RunFiles)
    (st : OneStageZk.SetUp) (ok : Bool) (why : String) (units : Array Nat) :
    RegOf rd (OneStageZk.verdictJson rd programSha s st ok why units) := by
  unfold RegOf OneStageZk.verdictJson
  rcases st with _ | ⟨_, _, part, opened⟩
  · dsimp only
    rw [getD_mkObj]
    rcases rd with ⟨_ | own, known, params⟩ <;> cases programSha <;> simp [Json.isNull, List.find?]
  · dsimp only
    rw [getD_mkObj]
    rcases rd with ⟨_ | own, known, params⟩ <;> cases programSha <;>
      by_cases hp : part.isNull = true <;> by_cases ho : opened.isNull = true <;> simp [hp, ho, List.find?]

theorem push_out {I : Flock.Witness.Inputs} {rd : OneStageZk.Reads} {programSha : Option String} {want : Json}
    {r r' : OneStageZk.Run} {s : OneStageZk.RunFiles} {record : String} {st : OneStageZk.SetUp}
    {coinKey : Option String} {verdict : Flock.V Unit}
    (h : OneStageZk.push I rd programSha want r s record st coinKey verdict = .ok r') :
    ∃ w, r'.out = r.out.push w ∧ RegOf rd w := by
  unfold OneStageZk.push at h
  try dsimp only at h
  split at h
  · cases h
    exact ⟨_, rfl, verdictJson_registered ..⟩
  · split at h
    · cases h
    · obtain ⟨_, -, h⟩ := bind_ok h
      cases h
      exact ⟨_, rfl, verdictJson_registered ..⟩

theorem step_out {I : Flock.Witness.Inputs} {rd : OneStageZk.Reads} {bs : Core.Blobs} {scope ksFiles : Flock.V Unit}
    {programSha : Option String} {want : Json} {r r' : OneStageZk.Run} {s : OneStageZk.RunFiles}
    (h : OneStageZk.step I rd bs scope ksFiles programSha want r s = .ok r') :
    ∃ w, r'.out = r.out.push w ∧ RegOf rd w := by
  unfold OneStageZk.step at h
  obtain ⟨_, -, h⟩ := bind_ok h
  try dsimp only at h
  split at h
  · exact push_out h
  · cases h

theorem foldlM_out {I : Flock.Witness.Inputs} {rd : OneStageZk.Reads} {bs : Core.Blobs} {scope ksFiles : Flock.V Unit}
    {programSha : Option String} {want : Json} :
    ∀ (l : List OneStageZk.RunFiles) (r r' : OneStageZk.Run), (∀ w ∈ r.out, RegOf rd w) →
      l.foldlM (OneStageZk.step I rd bs scope ksFiles programSha want) r = .ok r' → ∀ w ∈ r'.out, RegOf rd w
  | [], r, r', hr, h => by
    cases h
    exact hr
  | s :: l, r, r', hr, h => by
    rw [List.foldlM_cons] at h
    obtain ⟨r1, h1, h⟩ := bind_ok h
    obtain ⟨w, hw, hreg⟩ := step_out h1
    refine foldlM_out l r1 r' (fun w' hw' => ?_) h
    rw [hw] at hw'
    rcases Array.mem_push.1 hw' with hw' | hw'
    · exact hr w' hw'
    · rw [hw']
      exact hreg

theorem run_out {p : OneStageZk.Own} {pub : String} {sessions : Array OneStageZk.RunFiles} {pu : Option (Array Nat)}
    {bs : Core.Blobs} {r : OneStageZk.Run} (h : OneStageZk.run p pub sessions pu bs = .ok r) :
    ∃ rd, OneStageZk.readsOf p.registered = .ok rd ∧ ∀ w ∈ r.out, RegOf rd w := by
  unfold OneStageZk.run at h
  split at h
  · obtain ⟨⟨⟩, -, h⟩ := bind_ok h
    obtain ⟨⟨⟩, -, h⟩ := bind_ok h
    obtain ⟨_, -, h⟩ := bind_ok h
    obtain ⟨_, -, h⟩ := bind_ok h
    obtain ⟨rd, hrd, h⟩ := bind_ok h
    obtain ⟨r0, hr0, h⟩ := bind_ok h
    rw [← Array.foldlM_toList] at hr0
    have hout := foldlM_out _ _ r0 (fun w hw => by simp at hw) hr0
    refine ⟨rd, hrd, ?_⟩
    try dsimp only at h
    split at h
    · split at h
      · cases h
      · cases h
        exact hout
    · cases h
      exact hout
  · cases h

/-! ## R7's values and roots -/

theorem applyOutput_size (log : Array Json) (c : Array String) (mine : Array Registered.Entry)
    (held : Array (String × String)) (o : String × String × Registration.Earlier) :
    (Registration.applyOutput log (c, mine, held) o).2.1.size = mine.size := by
  obtain ⟨v, d, q⟩ := o
  simp only [Registration.applyOutput]
  split
  · rfl
  · split
    · simp
    · rfl

theorem applyOutputs_size (c : Array String) (mine : Array Registered.Entry) (held : Array (String × String))
    (outputs : Array (String × String × Registration.Earlier)) (log : Array Json) :
    (Registration.applyOutputs c mine held outputs log).2.1.size = mine.size := by
  unfold Registration.applyOutputs
  rw [← Array.foldl_toList]
  generalize outputs.toList = l
  induction l generalizing c mine held with
  | nil => rfl
  | cons o l ih =>
    rw [List.foldl_cons]
    have h1 := applyOutput_size log c mine held o
    generalize Registration.applyOutput log (c, mine, held) o = x at h1 ⊢
    obtain ⟨c', mine', held'⟩ := x
    rw [ih]
    exact h1

theorem expects_of_held {v : Registration.Verifier} {value r : String} (h : HeldRoot v value r) : ExpectsV2 v := by
  unfold ExpectsV2
  cases he : v.registered.isEmpty
  · rfl
  · exfalso
    obtain ⟨mine, mine', held', hown, happ, ⟨want, hwant, -⟩, -⟩ := h
    have h0 := Array.isEmpty_iff.1 he
    have hm : mine = #[] := by
      simp only [Registration.ownEntries, Registered.entries, h0, Array.mapM_empty] at hown
      cases hown
      rfl
    have hsz := applyOutputs_size #[] mine (v.pinned ++ Registration.heldRoots v.log (Registration.ownEntries v))
      v.outputs v.log
    rw [happ, hm] at hsz
    have : mine' = #[] := Array.size_eq_zero_iff.1 hsz
    rw [this] at hwant
    simp at hwant

theorem filterMap_all {α β : Type} {f : α → Option β} {xs : Array α} (h : (xs.filterMap f).size = xs.size) :
    ∀ x ∈ xs, (f x).isSome := by
  have h' : (xs.toList.filterMap f).length = xs.toList.length := by
    rw [← Array.toList_filterMap]
    exact h
  intro x hx
  exact List.filterMap_length_eq_length.1 h' x (Array.mem_toList_iff.2 hx)

theorem strs_of_names {es : Array Json} (h : (namesOf es).size = es.size) :
    ∀ e ∈ es, ∃ s, getD e "value" = .str s := by
  intro e he
  have := filterMap_all h e he
  try dsimp only at this
  revert this
  generalize getD e "value" = j
  cases j <;> simp

theorem mem_namesOf {es : Array Json} {s : String} (h : s ∈ (namesOf es).toList) :
    ∃ e ∈ es, getD e "value" = .str s := by
  rw [Array.mem_toList_iff] at h
  unfold namesOf at h
  obtain ⟨e, he, hs⟩ := Array.mem_filterMap.1 h
  refine ⟨e, he, ?_⟩
  try dsimp only at hs
  revert hs
  generalize getD e "value" = j
  cases j <;> simp

theorem field_isSome {j : Json} {k : String} {es : Array Json} (h : getD j k = .arr es) :
    (field? j k).isSome = true := by
  unfold getD at h
  cases hf : field? j k with
  | none => rw [hf] at h; cases h
  | some _ => rfl

/-- One entry of `Record.registeredRoots`' fold. -/
def rrStep (acc : Array (String × Json)) (e : Json) : Array (String × Json) :=
  match getD e "value" with
  | .str v => (acc.filter fun (q : String × Json) => q.1 != v).push (v, getD e "root")
  | _ => acc

theorem registeredRoots_eq {rec : Json} {es : Array Json} (h : getD rec "registered" = .arr es) :
    Record.registeredRoots rec = es.toList.reverse.foldr (fun e acc => rrStep acc e) #[] := by
  unfold Record.registeredRoots
  rw [h, List.foldr_reverse, Array.foldl_toList]
  rfl

theorem rr_foldr (value : String) : ∀ m : List Json, (∀ e ∈ m, ∃ s, getD e "value" = .str s) →
    (∀ y ∈ m.foldr (fun e acc => rrStep acc e) #[], y.1 = value →
      ∃ e, m.find? (fun e => jeq (getD e "value") (.str value)) = some e ∧ y.2 = getD e "root") ∧
    (∀ e ∈ m, getD e "value" = .str value → ∃ y ∈ m.foldr (fun e acc => rrStep acc e) #[], y.1 = value)
  | [], _ => ⟨fun y hy => by simp at hy, fun e he => by simp at he⟩
  | e :: m, hs => by
    obtain ⟨ih1, ih2⟩ := rr_foldr value m fun e' he' => hs e' (by simp [he'])
    obtain ⟨v, hv⟩ := hs e (by simp)
    have hstep : ∀ acc : Array (String × Json),
        rrStep acc e = (acc.filter fun (q : String × Json) => q.1 != v).push (v, getD e "root") :=
      fun acc => by simp only [rrStep, hv]
    simp only [List.foldr_cons]
    rw [hstep]
    by_cases hve : v = value
    · have hv' : getD e "value" = .str value := by rw [hv, hve]
      have hfe : (e :: m).find? (fun e => jeq (getD e "value") (.str value)) = some e :=
        List.find?_cons_of_pos (by simp only [hv', jeq_str_str])
      refine ⟨fun y hy hy1 => ?_, fun _ _ _ => ⟨(v, getD e "root"), Array.mem_push.2 (.inr rfl), hve⟩⟩
      rcases Array.mem_push.1 hy with hy | rfl
      · have := (Array.mem_filter.1 hy).2
        simp [hy1, hve] at this
      · exact ⟨e, hfe, rfl⟩
    · have hfe : (e :: m).find? (fun e => jeq (getD e "value") (.str value)) =
          m.find? (fun e => jeq (getD e "value") (.str value)) :=
        List.find?_cons_of_neg (by simp only [hv, jeq_str_str]; exact hve)
      refine ⟨fun y hy hy1 => ?_, fun e' he' hv' => ?_⟩
      · rcases Array.mem_push.1 hy with hy | rfl
        · obtain ⟨e'', h1, h2⟩ := ih1 y (Array.mem_filter.1 hy).1 hy1
          exact ⟨e'', by rw [hfe]; exact h1, h2⟩
        · exact absurd hy1 hve
      · rcases List.mem_cons.1 he' with rfl | he'
        · rw [hv] at hv'
          exact absurd (Json.str.inj hv') hve
        · obtain ⟨y, hy, hy1⟩ := ih2 e' he' hv'
          refine ⟨y, Array.mem_push.2 (.inl (Array.mem_filter.2 ⟨hy, ?_⟩)), hy1⟩
          simp only [hy1, bne_iff_ne, ne_eq]
          exact fun h => hve h.symm

/-! ## The roots the verdicts report (`Record.sessionRegistered`) -/

/-- `Record.sessionRegistered`'s update of a value already reported. -/
def updF (value : String) (r root : Json) : String × Json → String × Json :=
  fun (k, x) => if k == value then (k, if jeq r root then r else .null) else (k, x)

/-- `Record.sessionRegistered`'s step on one reported `(value, root)`. -/
def updAcc (acc : Array (String × Json)) (value : String) (root : Json) : Array (String × Json) :=
  match acc.find? (·.1 == value) with
  | some (_, r) => acc.map (updF value r root)
  | none => acc.push (value, root)

/-- `Record.sessionRegistered`'s inner loop body, on one entry of a verdict's `registered`. -/
def inStep (acc : Array (String × Json)) (x : String × Json) : Array (String × Json) :=
  match getD x.2 "value" with
  | .str value => updAcc acc value (getD x.2 "root")
  | _ => acc

/-- The verdicts `Record.sessionRegistered` reads: of record, accepted, with an object `registered`. -/
def regCond (w : String × Json) : Bool :=
  truthy (getD w.2 "of_record") && jeq (getD w.2 "accepted") (.bool true) && (getD w.2 "registered" matches .obj _)

/-- `Record.sessionRegistered`'s outer loop body, on one verdict. -/
def outStep (out : Option (Array (String × Json))) (w : String × Json) : Option (Array (String × Json)) :=
  if regCond w then some ((entries (getD w.2 "registered")).foldl inStep (out.getD #[])) else out

theorem forIn_foldl {α β : Type} (xs : Array α) (init : β) (f : α → β → Id (ForInStep β)) (g : β → α → β)
    (h : ∀ a b, f a b = pure (ForInStep.yield (g b a))) : forIn xs init f = (pure (xs.foldl g init) : Id β) := by
  have hf : f = fun a b => pure (ForInStep.yield (g b a)) := funext fun a => funext fun b => h a b
  subst hf
  exact Array.forIn_pure_yield_eq_foldl (m := Id) (fun a b => g b a) init

theorem sessionRegistered_eq (vs : Array (String × Json)) : Record.sessionRegistered vs = vs.foldl outStep none := by
  unfold Record.sessionRegistered
  dsimp only
  rw [forIn_foldl (g := outStep)]
  all_goals first
    | rfl
    | (intro w out
       rcases w with ⟨k, v⟩
       unfold outStep
       try dsimp only
       split <;> split
       all_goals first
         | (exfalso; rename_i h1 h2; first | exact h2 h1 | exact h1 h2)
         | (rw [forIn_foldl (g := inStep)]
            all_goals first
              | rfl
              | (intro x acc
                 rcases x with ⟨k', e⟩
                 unfold inStep
                 try dsimp only
                 generalize getD e "value" = j
                 cases j <;> rfl))
         | rfl)

/-- What the verdicts' roots `acc` hold after reading the pairs `S`: one entry per value, each value read and each read
value there, and an entry's root, unless two reads disagreed (null), the one every read of its value gave. -/
structure RegInv (acc : Array (String × Json)) (S : String × Json → Prop) : Prop where
  nodup : (acc.toList.map (·.1)).Nodup
  seen : ∀ x ∈ acc, ∃ y, S y ∧ y.1 = x.1
  covers : ∀ y, S y → ∃ x ∈ acc, x.1 = y.1
  agree : ∀ x ∈ acc, x.2 = .null ∨ ∀ y, S y → y.1 = x.1 → y.2 = x.2 ∨ jeq x.2 y.2 = true

theorem RegInv.congr {acc : Array (String × Json)} {S S' : String × Json → Prop} (h : RegInv acc S)
    (hS : ∀ y, S y ↔ S' y) : RegInv acc S' := by
  refine ⟨h.nodup, fun x hx => ?_, fun y hy => h.covers y ((hS y).2 hy), fun x hx => ?_⟩
  · obtain ⟨y, hy, h1⟩ := h.seen x hx
    exact ⟨y, (hS y).1 hy, h1⟩
  · rcases h.agree x hx with hn | ha
    · exact .inl hn
    · exact .inr fun y hy => ha y ((hS y).2 hy)

theorem regInv_empty {S : String × Json → Prop} (hS : ∀ y, ¬ S y) : RegInv #[] S :=
  ⟨List.nodup_nil, fun x hx => absurd hx (Array.not_mem_empty x), fun y hy => absurd hy (hS y),
    fun x hx => absurd hx (Array.not_mem_empty x)⟩

theorem updF_fst (value : String) (r root : Json) (kx : String × Json) : (updF value r root kx).1 = kx.1 := by
  obtain ⟨k, x⟩ := kx
  unfold updF
  dsimp only
  split <;> rfl

theorem updF_ne {value : String} {r root : Json} {kx : String × Json} (h : kx.1 ≠ value) :
    updF value r root kx = kx := by
  obtain ⟨k, x⟩ := kx
  unfold updF
  dsimp only at h ⊢
  rw [ite_eq_right (by simpa using h)]

theorem updF_eq {value : String} {r root : Json} {kx : String × Json} (h : kx.1 = value) :
    updF value r root kx = (kx.1, if jeq r root then r else .null) := by
  obtain ⟨k, x⟩ := kx
  unfold updF
  dsimp only at h ⊢
  rw [ite_eq_left (by simp [h])]

theorem updAcc_inv {acc : Array (String × Json)} {S : String × Json → Prop} {v : String} {root : Json}
    (h : RegInv acc S) : RegInv (updAcc acc v root) (fun y => y = (v, root) ∨ S y) := by
  unfold updAcc
  split
  · rename_i k0 r0 hf
    have hm0 : (k0, r0) ∈ acc := Array.mem_of_find?_eq_some hf
    have hk0 : k0 = v := by simpa using Array.find?_some hf
    have hkeys : (acc.map (updF v r0 root)).toList.map (·.1) = acc.toList.map (·.1) := by
      rw [Array.toList_map, List.map_map]
      exact List.map_congr_left fun kx _ => updF_fst v r0 root kx
    refine ⟨by rw [hkeys]; exact h.nodup, ?_, ?_, ?_⟩
    · intro x hx
      obtain ⟨kx, hkx, rfl⟩ := Array.mem_map.1 hx
      obtain ⟨y, hy, hy1⟩ := h.seen kx hkx
      exact ⟨y, .inr hy, by rw [updF_fst]; exact hy1⟩
    · rintro y (rfl | hy)
      · exact ⟨updF v r0 root (k0, r0), Array.mem_map_of_mem hm0, by rw [updF_fst]; exact hk0⟩
      · obtain ⟨x, hx, hx1⟩ := h.covers y hy
        exact ⟨updF v r0 root x, Array.mem_map_of_mem hx, by rw [updF_fst]; exact hx1⟩
    · intro x hx
      obtain ⟨kx, hkx, rfl⟩ := Array.mem_map.1 hx
      by_cases hkv : kx.1 = v
      · have hkx0 : kx = (k0, r0) := eq_of_nodup_map (·.1) acc.toList h.nodup kx (Array.mem_toList_iff.2 hkx)
          _ (Array.mem_toList_iff.2 hm0) (hkv.trans hk0.symm)
        subst hkx0
        rw [updF_eq hkv]
        by_cases hj : jeq r0 root = true
        · rw [ite_eq_left hj]
          rcases h.agree _ hm0 with hn | ha
          · exact .inl hn
          · refine .inr fun y hy hy1 => ?_
            rcases hy with rfl | hy
            · exact .inr hj
            · exact ha y hy hy1
        · rw [ite_eq_right hj]
          exact .inl rfl
      · rw [updF_ne hkv]
        rcases h.agree kx hkx with hn | ha
        · exact .inl hn
        · refine .inr fun y hy hy1 => ?_
          rcases hy with rfl | hy
          · exact absurd hy1.symm hkv
          · exact ha y hy hy1
  · rename_i hf
    have hnone : ∀ x ∈ acc, x.1 ≠ v := fun x hx hxv => by
      have := Array.find?_eq_none.1 hf x hx
      simp [hxv] at this
    refine ⟨?_, ?_, ?_, ?_⟩
    · rw [Array.toList_push, List.map_append, List.nodup_append]
      refine ⟨h.nodup, List.nodup_cons.2 ⟨List.not_mem_nil, List.nodup_nil⟩, fun a ha b hb => ?_⟩
      obtain ⟨x, hx, rfl⟩ := List.mem_map.1 ha
      simp only [List.map_cons, List.map_nil, List.mem_singleton] at hb
      rw [hb]
      exact hnone x (Array.mem_toList_iff.1 hx)
    · intro x hx
      rcases Array.mem_push.1 hx with hx | rfl
      · obtain ⟨y, hy, hy1⟩ := h.seen x hx
        exact ⟨y, .inr hy, hy1⟩
      · exact ⟨(v, root), .inl rfl, rfl⟩
    · rintro y (rfl | hy)
      · exact ⟨(v, root), Array.mem_push.2 (.inr rfl), rfl⟩
      · obtain ⟨x, hx, hx1⟩ := h.covers y hy
        exact ⟨x, Array.mem_push.2 (.inl hx), hx1⟩
    · intro x hx
      rcases Array.mem_push.1 hx with hx | rfl
      · rcases h.agree x hx with hn | ha
        · exact .inl hn
        · refine .inr fun y hy hy1 => ?_
          rcases hy with rfl | hy
          · exact absurd hy1.symm (hnone x hx)
          · exact ha y hy hy1
      · refine .inr fun y hy hy1 => ?_
        rcases hy with rfl | hy
        · exact .inl rfl
        · obtain ⟨x, hx, hx1⟩ := h.covers y hy
          exact absurd (hx1.trans hy1) (hnone x hx)

theorem inStep_inv {acc : Array (String × Json)} {S : String × Json → Prop} (x : String × Json) (h : RegInv acc S) :
    RegInv (inStep acc x) (fun y => entryPair x = some y ∨ S y) := by
  unfold inStep entryPair
  rcases hv : getD x.2 "value" with _ | _ | _ | v | _ | _ <;> try dsimp only
  all_goals first
    | exact RegInv.congr (updAcc_inv h) fun y =>
        ⟨fun | .inl h => .inl (by rw [h]) | .inr h => .inr h,
         fun | .inl h => .inl (Option.some.inj h).symm | .inr h => .inr h⟩
    | exact RegInv.congr h fun y => ⟨.inr, fun | .inl h => nomatch h | .inr h => h⟩

theorem foldl_inStep_inv : ∀ (l : List (String × Json)) (acc : Array (String × Json)) (S : String × Json → Prop),
    RegInv acc S → RegInv (l.foldl inStep acc) (fun y => (∃ x ∈ l, entryPair x = some y) ∨ S y)
  | [], acc, S, h => RegInv.congr h fun y => ⟨.inr, by
      rintro (⟨_, hx, _⟩ | hy)
      · exact absurd hx List.not_mem_nil
      · exact hy⟩
  | x :: l, acc, S, h => by
    refine RegInv.congr (foldl_inStep_inv l (inStep acc x) _ (inStep_inv x h)) fun y => ⟨?_, ?_⟩
    · rintro (⟨x', hx', hxy⟩ | hxy | hy)
      · exact .inl ⟨x', by simp [hx'], hxy⟩
      · exact .inl ⟨x, by simp, hxy⟩
      · exact .inr hy
    · rintro (⟨x', hx', hxy⟩ | hy)
      · rcases List.mem_cons.1 hx' with rfl | hx'
        · exact .inr (.inl hxy)
        · exact .inl ⟨x', hx', hxy⟩
      · exact .inr (.inr hy)

/-- `RegInv` of the verdicts' roots so far, none until a verdict reports any. -/
structure ORegInv (out : Option (Array (String × Json))) (S : String × Json → Prop) : Prop where
  ofSome : ∀ acc, out = some acc → RegInv acc S
  ofNone : out = none → ∀ y, ¬ S y

theorem ORegInv.congr {out : Option (Array (String × Json))} {S S' : String × Json → Prop} (h : ORegInv out S)
    (hS : ∀ y, S y ↔ S' y) : ORegInv out S' :=
  ⟨fun acc ha => (h.ofSome acc ha).congr hS, fun hn y hy => h.ofNone hn y ((hS y).2 hy)⟩

theorem outStep_inv {out : Option (Array (String × Json))} {S : String × Json → Prop} (w : String × Json)
    (h : ORegInv out S) :
    ORegInv (outStep out w)
      (fun y => (regCond w = true ∧ ∃ x ∈ entries (getD w.2 "registered"), entryPair x = some y) ∨ S y) := by
  unfold outStep
  split
  · rename_i hc
    have hacc : RegInv (out.getD #[]) S := by
      cases hout : out with
      | none => exact regInv_empty (h.ofNone hout)
      | some acc => exact h.ofSome acc hout
    have := foldl_inStep_inv (entries (getD w.2 "registered")).toList _ _ hacc
    rw [Array.foldl_toList] at this
    refine ⟨fun acc' hacc' => ?_, fun hn => by cases hn⟩
    cases hacc'
    refine RegInv.congr this fun y => ⟨?_, ?_⟩
    · rintro (⟨x, hx, hxy⟩ | hy)
      · exact .inl ⟨hc, x, Array.mem_toList_iff.1 hx, hxy⟩
      · exact .inr hy
    · rintro (⟨-, x, hx, hxy⟩ | hy)
      · exact .inl ⟨x, Array.mem_toList_iff.2 hx, hxy⟩
      · exact .inr hy
  · rename_i hc
    refine ORegInv.congr h fun y => ⟨.inr, ?_⟩
    rintro (⟨hcw, -⟩ | hy)
    · exact absurd hcw hc
    · exact hy

/-- A pair some verdict of `vs` that `Record.sessionRegistered` reads reports. -/
def SeenIn (vs : List (String × Json)) (y : String × Json) : Prop :=
  ∃ w ∈ vs, regCond w = true ∧ ∃ x ∈ entries (getD w.2 "registered"), entryPair x = some y

theorem foldl_outStep_inv : ∀ (l : List (String × Json)) (out : Option (Array (String × Json)))
    (S : String × Json → Prop), ORegInv out S → ORegInv (l.foldl outStep out) (fun y => SeenIn l y ∨ S y)
  | [], out, S, h => ORegInv.congr h fun y => ⟨.inr, by
      rintro (⟨_, hw, _⟩ | hy)
      · exact absurd hw List.not_mem_nil
      · exact hy⟩
  | w :: l, out, S, h => by
    refine ORegInv.congr (foldl_outStep_inv l (outStep out w) _ (outStep_inv w h)) fun y => ⟨?_, ?_⟩
    · rintro (⟨w', hw', hc, hx⟩ | ⟨hc, hx⟩ | hy)
      · exact .inl ⟨w', by simp [hw'], hc, hx⟩
      · exact .inl ⟨w, by simp, hc, hx⟩
      · exact .inr hy
    · rintro (⟨w', hw', hc, hx⟩ | hy)
      · rcases List.mem_cons.1 hw' with rfl | hw'
        · exact .inr (.inl ⟨hc, hx⟩)
        · exact .inl ⟨w', hw', hc, hx⟩
      · exact .inr (.inr hy)

theorem sessionRegistered_inv {vs acc : Array (String × Json)} (h : Record.sessionRegistered vs = some acc) :
    RegInv acc (SeenIn vs.toList) := by
  rw [sessionRegistered_eq, ← Array.foldl_toList] at h
  have := foldl_outStep_inv vs.toList none (fun _ => False) ⟨fun _ h => (nomatch h), fun _ _ h => h⟩
  exact RegInv.congr (this.ofSome acc h) fun y => ⟨fun | .inl h => h | .inr h => h.elim, .inl⟩

theorem root_of_inv {acc : Array (String × Json)} {S : String × Json → Prop} (h : RegInv acc S)
    {x : String × Json} (hx : x ∈ acc) {r : String} (hj : jeq x.2 (.str r) = true) {z : String × Json} (hz : S z)
    (hz1 : z.1 = x.1) {s : String} (hzs : z.2 = .str s) : s = r := by
  rcases h.agree x hx with hn | ha
  · rw [hn, jeq_null_str] at hj
    cases hj
  · rcases ha z hz hz1 with he | he
    · rw [hzs] at he
      rw [← he, jeq_str_str] at hj
      exact hj
    · rw [hzs] at he
      exact jeq_str he hj

/-! ## `Bits.rowBit`'s rows are the circuit's ports -/

theorem default_bits : (default : Flock.Port).bits = 0 := rfl

theorem msgAt_lt {c : Flock.Circuit} {ix : Bits.Index} {x t : Nat} {wide : Bool} {p byte bit : Nat}
    (h : Bits.msgAt c ix x t wide = .msg p byte bit) : p < c.ports.size := by
  unfold Bits.msgAt at h
  dsimp only at h
  split at h
  · rename_i hc
    injection h with hp
    subst hp
    refine Classical.byContradiction fun hn => ?_
    have hn' := Nat.not_lt.1 hn
    simp [Array.getElem?_eq_none hn', default_bits] at hc
  · cases h

theorem srcAt_lt {c : Flock.Circuit} {ix : Bits.Index} {k p0 t : Nat} {p byte bit : Nat}
    (h : Bits.srcAt c ix k p0 t = .msg p byte bit) : p < c.ports.size := by
  unfold Bits.srcAt at h
  dsimp only at h
  repeat' split at h
  all_goals first | exact msgAt_lt h | cases h

theorem srcKey_lt {c : Flock.Circuit} {ix : Bits.Index} {x : Nat} {p byte bit : Nat}
    (h : Bits.srcKey c ix x = .msg p byte bit) : p < c.ports.size := by
  unfold Bits.srcKey at h
  dsimp only at h
  split at h
  · exact srcAt_lt h
  · cases h

theorem srcPos_port {c : Flock.Circuit} {u : Nat} {k : Bits.Src} {rp : Bits.RowPos}
    (hk : ∀ q byte bit, k = .msg q byte bit → q < c.ports.size) (h : Bits.srcPos c u k = some rp) :
    ∃ p ∈ c.ports, p.name = rp.row := by
  rcases k with _ | ⟨q, byte, bit⟩ | ⟨_, _, _⟩ | _ | ⟨_, _⟩
  all_goals first
    | (have hq := hk q byte bit rfl
       simp only [Bits.srcPos, Option.some.injEq] at h
       subst h
       exact ⟨c.ports[q], Array.getElem_mem hq, by simp [Array.getElem?_eq_getElem hq]⟩)
    | cases h
    | simp [Bits.srcPos] at h

theorem rowBit_port {c : Flock.Circuit} {n g : Nat} {rp : Bits.RowPos} (h : Bits.rowBit c n g = some rp) :
    ∃ p ∈ c.ports, p.name = rp.row := by
  unfold Bits.rowBit at h
  dsimp only at h
  split at h
  · cases h
  · exact srcPos_port (fun _ _ _ hk => srcKey_lt hk) h

end Covered

open Covered

theorem heldCovered : HeldCovered := by
  intro p c ξ e bs pub prof h value r hh
  obtain ⟨R, hrun, -, hcheck⟩ := checkBlobsAt_accept h
  obtain ⟨rd, hrd, hreg⟩ := run_out hrun
  obtain ⟨law, d, n, hR, -, hunits, -, -, -, -, -⟩ := acceptMeans _ _ _ _ _ hcheck
  have hexp := expects_of_held hh
  obtain ⟨mine, mine', held', hown, happ, ⟨want, hwant, hwv⟩, hfind⟩ := hh
  obtain ⟨-, -, -, -, -, -, h7⟩ := hR
  obtain ⟨mine2, es, mine2', held2', hown2, hes, -, happ2, -, hva, hentry, -⟩ := h7 hexp
  rw [hown] at hown2
  obtain rfl := Except.ok.inj hown2
  rw [happ] at happ2
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj happ2
  obtain ⟨hva1, -, hva3, hva4⟩ := hva
  have hvL : want.value ∈ (namesOf es).toList.eraseDups :=
    mem_of_nodup_length (m := mine'.toList.map (·.value)) (nodup_eraseDups _)
      (fun a ha => by
        obtain ⟨m, hm, hmv⟩ := hva4 a ha
        exact List.mem_map.2 ⟨m, Array.mem_toList_iff.2 hm, hmv⟩)
      (Nat.le_of_eq (by rw [List.length_map, Array.length_toList, hva3]))
      (List.mem_map.2 ⟨want, Array.mem_toList_iff.2 hwant, rfl⟩)
  obtain ⟨e1, he1, he1v⟩ := mem_namesOf (List.mem_eraseDups.1 hvL)
  have hsome : (es.reverse.find? (fun e => jeq (getD e "value") (.str want.value))).isSome = true :=
    Array.find?_isSome.2 ⟨e1, Array.mem_reverse.2 he1, by simp only [he1v, jeq_str_str]⟩
  obtain ⟨e0, he0⟩ := Option.isSome_iff_exists.1 hsome
  obtain ⟨⟨s0, hs0, -, -⟩, -, -, -, -, hroot⟩ := hentry want hwant e0 he0
  have hj0 : jeq (getD e0 "root") (.str r) = true := by
    rw [hwv] at hroot
    exact hroot (value, r) hfind
  have hr0 : getD e0 "root" = .str r := by
    rw [hs0] at hj0 ⊢
    rw [jeq_str_str.1 hj0]
  have hstrs := strs_of_names hva1
  obtain ⟨hrr1, hrr2⟩ := rr_foldr want.value es.toList.reverse
    (fun e he => hstrs e (Array.mem_toList_iff.1 (List.mem_reverse.1 he)))
  rw [← registeredRoots_eq hes] at hrr1 hrr2
  have hfind0 : es.toList.reverse.find? (fun e => jeq (getD e "value") (.str want.value)) = some e0 := by
    rw [← Array.toList_reverse, Array.find?_toList]
    exact he0
  obtain ⟨y0, hy0, hy01⟩ := hrr2 e1 (List.mem_reverse.2 (Array.mem_toList_iff.2 he1)) he1v
  have hrrv : ∀ y ∈ Record.registeredRoots c.record, y.1 = want.value → y.2 = .str r := by
    intro y hy hy1
    obtain ⟨e', he', hye⟩ := hrr1 y hy hy1
    rw [hfind0] at he'
    obtain rfl := Option.some.inj he'
    rw [hye, hr0]
  obtain ⟨-, -, -, -, -, hsame⟩ := hunits
  have hsr := hsame (.inl (field_isSome hes))
  try dsimp only at hsr
  cases hs : Record.sessionRegistered R.out with
  | none =>
    simp only [openedRoots, Record.ownRegistered, hs, Option.map_none, Option.getD_none] at hsr
    obtain ⟨hsz, -⟩ := hsr
    have h0 : Record.registeredRoots c.record = #[] :=
      Array.size_eq_zero_iff.1 (hsz.symm.trans Array.size_empty)
    rw [h0] at hy0
    simp at hy0
  | some acc =>
    have hinv := sessionRegistered_inv hs
    simp only [openedRoots, Record.ownRegistered, hs, Option.map_some, Option.getD_some] at hsr
    obtain ⟨x, hx, hx1⟩ := sameRoots_mem hsr (nodup_filter_keys hinv.nodup) hy0
    obtain ⟨-, hsm⟩ := hsr
    obtain ⟨y, hy, hxy, hjxy⟩ := hsm x hx
    have hxa := (Array.mem_filter.1 hx).1
    have hxk : x.1 = want.value := hx1.trans hy01
    have hjx : jeq x.2 (.str r) = true := by
      rw [hrrv y hy (hxy.symm.trans hxk)] at hjxy
      exact hjxy
    obtain ⟨z, ⟨w, hw, hcw, ⟨tk, tv⟩, ht, hzt⟩, hz1⟩ := hinv.seen x hxa
    have hregw := hreg w (Array.mem_toList_iff.1 hw)
    cases hro : rd.registered with
    | none =>
      simp only [RegOf, hro, Option.map_none, Option.getD_none] at hregw
      simp [regCond, hregw] at hcw
    | some own =>
      have hfo : getD w.2 "registered" = Flock.Registered.finding own := by
        unfold RegOf at hregw
        rw [hro] at hregw
        exact hregw
      have ht2 : (tk, tv) ∈ entries (Flock.Registered.finding own) := by
        rw [← hfo]
        exact ht
      obtain ⟨q, hq, rfl, rfl⟩ := mem_entries_finding ht2
      have hzq := entryPair_portJ_inv hzt
      subst hzq
      refine ⟨rd, own, hrd, hro, ⟨q, hq, (hz1.trans hxk).trans hwv,
        root_of_inv hinv hxa hjx ⟨w, hw, hcw, _, ht, hzt⟩ hz1 rfl⟩, fun q' hq' hq'v hu => ?_⟩
      have ht' : (q'.name, portJ q') ∈ entries (getD w.2 "registered") := by
        rw [hfo]
        exact entries_finding_of_unique hq' hu
      exact root_of_inv hinv hxa hjx ⟨w, hw, hcw, _, ht', entryPair_portJ q'.name q'⟩
        (hq'v.trans (hwv.symm.trans hxk.symm)) rfl

theorem gatesHeld : GatesHeld := by
  intro p cm ξ e bs pub prof h rd own hrd hown c n g rp hcc hrb
  obtain ⟨pt, hpt, hpn⟩ := rowBit_port hrb
  obtain ⟨q0, hq0, hq0n⟩ := Flock.Registered.checkComplete_ok hcc pt hpt
  refine ⟨⟨q0, hq0, hq0n.trans hpn⟩, fun q hq _ hu r hr => ?_⟩
  obtain ⟨rd', own', hrd', hown', -, hall⟩ := heldCovered p cm ξ e bs pub prof h q.value r hr
  rw [hrd] at hrd'
  obtain rfl := Except.ok.inj hrd'
  rw [hown] at hown'
  obtain rfl := Option.some.inj hown'
  exact hall q hq rfl hu

end Properties.SampledProofs
