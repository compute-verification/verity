module

public import Properties.Flock.DecodeBoolean
public import Proofs.Properties.InferenceOnly.DecodeKeys
public import Std.Tactic.Do
import Batteries.Tactic.OpenPrivate
import Std.Data.HashMap.Lemmas
import Std.Data.HashSet.Lemmas

/-!
# `decode_boolean`: the Boolean map of a decoded Program

`decode_boolean : Flock.Program.DecodeBoolean`, read off `decode`'s steps (`BodiesOk.decode_steps`):

* every primitive Definition the decoder pushes is keyed by an entry of `definitions` of form `primitive`
  (`DecodeKeys.Src`, part of what `DecodeKeys.kept_fuel` keeps), so every word primitive was decoded by `decodeAll`, and
  `booleanKeys` (no `E_BODY_MISSING`) gives its key a pair of the map;
* each pair `decodeBodies` takes is checked by `checkBody` against the Definitions decoded so far (`Checked`), with
  `reach` meaning a Boolean Definition (`ReachOk`, `extendReach_ok`); the Definitions only grow (`Pre`) and the index
  keeps its entries (`idx_keep`), so the check holds of the Program;
* every key of the map is a word primitive's (`booleanKeys`, no second `E_BODY_KEY`).
-/

@[expose] public section


namespace FlockSoundness.DecodeBoolean
open Flock.Program
open Std.Do
open InferenceOnlyVerifier.Properties.SecurityProofs
open BodiesOk (DM st_bind st_pure st_get st_liftV st_discard Inv Pre pre_refl pre_trans specs go_inv decode_steps)
open DecodeKeys (Kept Keys of_triple kept_fuel getDef_idx go_kept formOf)
open private getDef composite staticFns decodeAll decodeBodies extendReach primEntries DecState DecState.defs
  DecState.index DecState.busy Desc Desc.ddefs liftV from Circuits.Program

/-! ## Maps and sets built by folds -/

theorem hm_foldl_some {α β γ : Type} [BEq α] [Hashable α] [LawfulBEq α] (f : Std.HashMap α β → γ → Std.HashMap α β)
    (g : γ → α × β) (hf : ∀ m x, f m x = m.insert (g x).1 (g x).2) (k : α) (v : β) :
    ∀ (l : List γ) (m : Std.HashMap α β), (l.foldl f m)[k]? = some v → m[k]? = some v ∨ ∃ x ∈ l, g x = (k, v)
  | [], _, h => Or.inl h
  | x :: l, m, h => by
    rcases hm_foldl_some f g hf k v l _ h with h | ⟨y, hy, he⟩
    · rw [hf, Std.HashMap.getElem?_insert] at h
      split at h
      · rename_i he
        refine Or.inr ⟨x, List.mem_cons_self, ?_⟩
        rw [← eq_of_beq he, ← Option.some.inj h]
      · exact Or.inl h
    · exact Or.inr ⟨y, List.mem_cons_of_mem _ hy, he⟩

theorem hm_foldl_isSome {α β γ : Type} [BEq α] [Hashable α] [LawfulBEq α] (f : Std.HashMap α β → γ → Std.HashMap α β)
    (g : γ → α × β) (hf : ∀ m x, f m x = m.insert (g x).1 (g x).2) (k : α) :
    ∀ (l : List γ) (m : Std.HashMap α β), (m[k]?.isSome = true ∨ ∃ x ∈ l, (g x).1 = k) → (l.foldl f m)[k]?.isSome = true
  | [], _, h => h.resolve_right (by simp)
  | x :: l, m, h => by
    refine hm_foldl_isSome f g hf k l _ ?_
    rw [hf, Std.HashMap.getElem?_insert]
    rcases h with h | ⟨y, hy, he⟩
    · left; split <;> simp_all
    · rcases List.mem_cons.1 hy with rfl | hy
      · left; simp [he]
      · right; exact ⟨y, hy, he⟩

theorem hs_foldl {γ : Type} (f : Std.HashSet String → γ → Std.HashSet String) (g : γ → String)
    (hf : ∀ s x, f s x = s.insert (g x)) (k : String) :
    ∀ (l : List γ) (s : Std.HashSet String), (l.foldl f s).contains k = true → s.contains k = true ∨ ∃ x ∈ l, g x = k
  | [], _, h => Or.inl h
  | x :: l, s, h => by
    rcases hs_foldl f g hf k l _ h with h | ⟨y, hy, he⟩
    · rw [hf, Std.HashSet.contains_insert] at h
      rcases Bool.or_eq_true_iff.mp h with h | h
      · exact Or.inr ⟨x, List.mem_cons_self, eq_of_beq h⟩
      · exact Or.inl h
    · exact Or.inr ⟨y, List.mem_cons_of_mem _ hy, he⟩

/-! ## Boolean Definitions over an array -/

/-- `Prog.BoolDef` over the Definitions `a`, with every node's callee placed before its caller. -/
inductive BoolDefA (a : Array Defn) : Nat → Prop
  | prim {d : Nat} : d < a.size → (a[d]!).prim = true → isBooleanPrim (a[d]!).key = true → BoolDefA a d
  | comp {d : Nat} : d < a.size → (a[d]!).prim = false → (∀ nd ∈ (a[d]!).body.nodes, nd.fn < d) →
      (∀ nd ∈ (a[d]!).body.nodes, (a[nd.fn]!).prim = true → isBooleanPrim (a[nd.fn]!).key = true) →
      (∀ nd ∈ (a[d]!).body.nodes, (a[nd.fn]!).prim = false → BoolDefA a nd.fn) → BoolDefA a d

theorem boolDefA_mono {a b : Array Defn} (hp : Pre a b) {d : Nat} (h : BoolDefA a d) : BoolDefA b d := by
  induction h with
  | prim hd hpr hb =>
    have e := hp.2 _ hd
    exact .prim (Nat.lt_of_lt_of_le hd hp.1) (by rw [e]; exact hpr) (by rw [e]; exact hb)
  | @comp d hd hpr hlt hpb _ ih =>
    have e := hp.2 d hd
    refine .comp (Nat.lt_of_lt_of_le hd hp.1) (by rw [e]; exact hpr) (by rw [e]; exact hlt) ?_ ?_
    · rw [e]
      intro nd hn
      rw [hp.2 _ (Nat.lt_trans (hlt nd hn) hd)]
      exact hpb nd hn
    · rw [e]
      intro nd hn h
      rw [hp.2 _ (Nat.lt_trans (hlt nd hn) hd)] at h
      exact ih nd hn h

theorem boolDef_of {P : Prog} {d : Nat} (h : BoolDefA P.defs d) : P.BoolDef d := by
  induction h with
  | prim _ hpr hb => exact .prim hpr hb
  | comp _ hpr hlt hpb _ ih => exact .comp hpr hpb (fun nd hn _ => hlt nd hn) ih

/-- `checkBody`'s verdict on word `w`'s body at place `b`, over the Definitions `a`: `Prog.BodyOf`. -/
def BodyOfA (a : Array Defn) (w b : Nat) : Prop :=
  BoolDefA a b ∧ ((a[b]!).prim = false → (a[b]!).params.all (·.2.bits) = true ∧ (a[b]!).ret.bits = true) ∧
    (a[b]!).params.map (·.2) = (a[w]!).params.map (·.2.bitType) ∧ (a[b]!).ret = (a[w]!).ret.bitType

theorem bodyOfA_mono {a b : Array Defn} (hp : Pre a b) {w bi : Nat} (hb : bi < a.size) (hw : w < a.size)
    (h : BodyOfA a w bi) : BodyOfA b w bi := by
  obtain ⟨h1, h2, h3, h4⟩ := h
  refine ⟨boolDefA_mono hp h1, ?_, ?_, ?_⟩ <;> rw [hp.2 _ hb] <;> (try rw [hp.2 _ hw]) <;> assumption

theorem bodyOf_of {P : Prog} {w b : Nat} (h : BodyOfA P.defs w b) : P.BodyOf w b :=
  ⟨boolDef_of h.1, h.2.1, h.2.2.1, h.2.2.2⟩

/-! ## `reach` -/

/-- Each entry of `reach` that is set names a Boolean Definition. -/
def ReachOk (a : Array Defn) (r : Array Bool) : Prop := r.size ≤ a.size ∧ ∀ i < r.size, r[i]! = true → BoolDefA a i

/-- Every node calls a Definition placed before its caller (`BodyFacts`). -/
def NodesBefore (a : Array Defn) : Prop := ∀ i < a.size, ∀ nd ∈ (a[i]!).body.nodes, nd.fn < i

theorem nodesBefore_of {s : DecState} (h : Inv s) : NodesBefore (DecState.defs s) := by
  intro i hi nd hn
  obtain ⟨k, hk, rfl⟩ := Array.mem_iff_getElem.1 hn
  have := ((h.1 i hi).2.1 k hk).1
  simpa [getElem!_pos, hk] using this

theorem reachOk_mono {a b : Array Defn} (hp : Pre a b) {r : Array Bool} (h : ReachOk a r) : ReachOk b r :=
  ⟨Nat.le_trans h.1 hp.1, fun i hi hv => boolDefA_mono hp (h.2 i hi hv)⟩

/-- `extendReach`'s value for place `i`, from the entries `r` before it. -/
def reachStep (a : Array Defn) (r : Array Bool) (i : Nat) : Bool :=
  if (a[i]!).prim then isBooleanPrim (a[i]!).key else (a[i]!).body.nodes.all fun x =>
    if (a[x.fn]!).prim then isBooleanPrim (a[x.fn]!).key else r[x.fn]?.getD false

theorem extendReach_eq (a : Array Defn) (r : Array Bool) :
    extendReach a r = (List.range' r.size (a.size - r.size)).foldl (fun r i => r.push (reachStep a r i)) r := by
  simp only [extendReach, Id.run, Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl,
    Std.Legacy.Range.size, Nat.add_sub_cancel, Nat.div_one, bind_pure_comp, map_pure, pure_bind, bind_pure]
  rfl

theorem push_ok {a : Array Defn} (ha : NodesBefore a) {r : Array Bool} (h : ReachOk a r) (hs : r.size < a.size) :
    ReachOk a (r.push (reachStep a r r.size)) := by
  refine ⟨by simp only [Array.size_push]; omega, fun i hi hv => ?_⟩
  simp only [Array.size_push] at hi
  rcases Nat.lt_succ_iff_lt_or_eq.1 hi with hi | rfl
  · rw [BodiesOk.getElem!_push_lt _ _ hi] at hv
    exact h.2 i hi hv
  · rw [BodiesOk.getElem!_push_eq] at hv
    unfold reachStep at hv
    split at hv
    · exact .prim hs (by assumption) hv
    · rename_i hp
      have hp' : (a[r.size]!).prim = false := by simpa using hp
      have hv' := Array.all_eq_true'.1 hv
      refine .comp hs hp' (ha _ hs) (fun nd hn hpn => ?_) (fun nd hn hpn => ?_)
      · simpa [hpn] using hv' nd hn
      · have hlt : nd.fn < r.size := ha _ hs nd hn
        have h1 := hv' nd hn
        simp only [hpn, Bool.false_eq_true, ↓reduceIte] at h1
        exact h.2 _ hlt (by simpa [getElem!_pos, hlt] using h1)

theorem reach_foldl (a : Array Defn) (ha : NodesBefore a) :
    ∀ (n : Nat) (r : Array Bool), ReachOk a r → r.size + n ≤ a.size →
      ReachOk a ((List.range' r.size n).foldl (fun r i => r.push (reachStep a r i)) r) ∧
        ((List.range' r.size n).foldl (fun r i => r.push (reachStep a r i)) r).size = r.size + n
  | 0, r, h, _ => ⟨h, rfl⟩
  | n + 1, r, h, hn => by
    rw [List.range'_succ, List.foldl_cons]
    have := reach_foldl a ha n (r.push (reachStep a r r.size)) (push_ok ha h (by omega))
      (by simp only [Array.size_push]; omega)
    simp only [Array.size_push] at this
    exact ⟨this.1, by rw [this.2]; omega⟩

theorem extendReach_ok {a : Array Defn} (ha : NodesBefore a) {r : Array Bool} (h : ReachOk a r) :
    ReachOk a (extendReach a r) ∧ (extendReach a r).size = a.size := by
  rw [extendReach_eq]
  have h1 := h.1
  have := reach_foldl a ha (a.size - r.size) r h (by omega)
  exact ⟨this.1, by rw [this.2]; omega⟩

/-! ## `checkBody` -/

theorem arr_ty_beq {a b : Array Ty} (h : (a == b) = true) : a = b := by
  obtain ⟨hs, hr⟩ := Array.isEqv_iff_rel.1 h
  exact Array.ext hs (fun i h1 _ => BodiesOk.ty_beq (hr i h1))

theorem checkBody_ok {a : Array Defn} {index : Std.HashMap String Nat} {rootKey : String} {lo : Nat}
    {reach : Array Bool} {w b : String} {bi wi : Nat} {u : Unit} (hr : ReachOk a reach) (hrs : reach.size = a.size)
    (hbi : bi < a.size) (hw : index[w]? = some wi) (h : checkBody a index rootKey lo reach w b bi = .ok u) :
    BodyOfA a wi bi := by
  unfold checkBody at h
  simp only [hw, Option.getD_some] at h
  split at h
  · exact absurd h (by intro h; injection h)
  cases hp : (a[bi]!).prim <;> simp only [hp, Bool.false_eq_true, ↓reduceIte] at h <;> split at h <;>
    (try (injection h; done)) <;> rename_i hc <;>
    simp only [Bool.not_eq_true, Bool.not_eq_true', Bool.not_eq_false, Bool.not_eq_false'] at hc <;>
    rw [Bool.and_eq_true, Bool.and_eq_true] at hc <;> obtain ⟨⟨hbool, hps⟩, hret⟩ := hc
  · rw [Bool.and_eq_true, Bool.and_eq_true] at hbool
    exact ⟨hr.2 bi (hrs ▸ hbi) hbool.2, fun _ => hbool.1, arr_ty_beq hps, BodiesOk.ty_beq hret⟩
  · exact ⟨.prim hbi hp (by simpa using hbool), fun h => absurd (hp ▸ h) (by simp), arr_ty_beq hps, BodiesOk.ty_beq hret⟩

/-! ## What the walk keeps besides `Kept` -/

/-! ## The decoder's runs -/

/-- What `decode_boolean` reads of a decoder state: `Kept` with no name busy, and `Inv`. -/
def Base (desc : Desc) (s : DecState) : Prop := Kept desc (fun _ => False) s ∧ Inv s

theorem fuel_succ : ∃ m, Flock.Steps.FUEL = m + 1 :=
  ⟨_, (Nat.sub_add_cancel (show 1 ≤ Flock.Steps.FUEL by unfold Flock.Steps.FUEL; exact Nat.one_le_two_pow)).symm⟩

theorem getDef_run (desc : Desc) (fid : String) {s s' : DecState} {r : Nat}
    (h : (getDef Flock.Steps.FUEL desc fid : DM Nat) s = .ok (r, s')) (hb : Base desc s) :
    Base desc s' ∧ Pre (DecState.defs s) (DecState.defs s') ∧ (DecState.index s')[fid]? = some r ∧
      r < (DecState.defs s').size := by
  obtain ⟨i1, p1, r1⟩ := (specs _).1 _ _ _ _ _ h hb.2
  have hk := of_triple ((kept_fuel Flock.Steps.FUEL).1 desc fid (fun _ => False)) h hb.1
  obtain ⟨m, hm⟩ := fuel_succ
  rw [hm] at h
  have hi := of_triple (getDef_idx m (kept_fuel m).2.2 desc fid _) h hb.1
  exact ⟨⟨hk, i1⟩, p1, hi, r1⟩

/-- An index entry outlives a step that keeps `Keys` and only appends Definitions. -/
theorem idx_keep {s s' : DecState} (hk : Keys s) (hk' : Keys s') (hp : Pre (DecState.defs s) (DecState.defs s'))
    {k : String} {v : Nat} (h : (DecState.index s)[k]? = some v) : (DecState.index s')[k]? = some v := by
  obtain ⟨hv, hkey⟩ := hk.1 k v h
  have := hk'.2.1 v (Nat.lt_of_lt_of_le hv hp.1)
  rwa [hp.2 v hv, hkey] at this

theorem decodeAll_run (desc : Desc) (fs : Array String) {s s' : DecState} {u : Unit}
    (h : StateT.run (decodeAll desc fs : DM Unit) s = .ok (u, s')) (hb : Base desc s) :
    Base desc s' ∧ Pre (DecState.defs s) (DecState.defs s') ∧
      ∀ f ∈ fs, (Desc.ddefs desc).contains f = true → ∃ i, (DecState.index s')[f]? = some i := by
  unfold decodeAll at h
  obtain ⟨r, s1, h1, h⟩ := st_bind h
  obtain ⟨-, rfl⟩ := st_pure h
  rw [← Array.forIn_toList] at h1
  have := BodiesOk.forIn_stateT_inv _ (fun _ _ => False) fs.toList
    (fun j _ s1 => Base desc s1 ∧ Pre (DecState.defs s) (DecState.defs s1) ∧
      ∀ j' (hj : j' < fs.toList.length), j' < j → (Desc.ddefs desc).contains fs.toList[j'] = true →
        ∃ i, (DecState.index s1)[fs.toList[j']]? = some i)
    _ _ _ _ ⟨hb, pre_refl _, fun _ _ hj => absurd hj (Nat.not_lt_zero _)⟩ ?_ h1
  · rcases this with ⟨hb', hp, hf⟩ | hf
    · refine ⟨hb', hp, fun f hf' hc => ?_⟩
      obtain ⟨j, hj, rfl⟩ := List.mem_iff_getElem.1 (by simpa using hf' : f ∈ fs.toList)
      exact hf j hj hj hc
    · exact hf.elim
  · intro i hi b s1 t s2 hI hf
    obtain ⟨hb1, hp1, hf1⟩ := hI
    split at hf
    · obtain ⟨u, s3, h3, h4⟩ := st_bind hf
      obtain ⟨rfl, rfl⟩ := st_pure h4
      obtain ⟨a, ha⟩ := st_discard h3
      obtain ⟨hb2, hp2, hi2, -⟩ := getDef_run desc _ ha hb1
      refine ⟨hb2, pre_trans hp1 hp2, fun j' hj hlt hc => ?_⟩
      rcases Nat.lt_succ_iff_lt_or_eq.1 hlt with hlt | rfl
      · obtain ⟨i0, h0⟩ := hf1 j' hj hlt hc
        exact ⟨i0, idx_keep hb1.1.1 hb2.1.1 hp2 h0⟩
      · exact ⟨a, hi2⟩
    · obtain ⟨rfl, rfl⟩ := st_pure hf
      refine ⟨hb1, hp1, fun j' hj hlt hc => ?_⟩
      rcases Nat.lt_succ_iff_lt_or_eq.1 hlt with hlt | rfl
      · exact hf1 j' hj hlt hc
      · exact absurd hc (by assumption)

/-- Pair `p` of the map was checked: its body's and its word's places in the index, and `checkBody`'s verdict. -/
def Checked (s : DecState) (p : String × String) : Prop :=
  ∃ bi wi, (DecState.index s)[p.2]? = some bi ∧ (DecState.index s)[p.1]? = some wi ∧
    BodyOfA (DecState.defs s) wi bi

theorem checked_mono {s s' : DecState} (hk : Keys s) (hk' : Keys s') (hp : Pre (DecState.defs s) (DecState.defs s'))
    {p : String × String} (h : Checked s p) : Checked s' p := by
  obtain ⟨bi, wi, h1, h2, h3⟩ := h
  exact ⟨bi, wi, idx_keep hk hk' hp h1, idx_keep hk hk' hp h2,
    bodyOfA_mono hp (hk.1 _ _ h1).1 (hk.1 _ _ h2).1 h3⟩

theorem decodeBodies_run (desc : Desc) (rootKey : String) (pairs : Array (String × String)) {s s' : DecState}
    {u : Unit} (h : StateT.run (decodeBodies desc rootKey pairs : DM Unit) s = .ok (u, s')) (hb : Base desc s)
    (hw : ∀ p ∈ pairs, ∃ wi, (DecState.index s)[p.1]? = some wi) :
    Base desc s' ∧ Pre (DecState.defs s) (DecState.defs s') ∧ ∀ p ∈ pairs, Checked s' p := by
  unfold decodeBodies at h
  obtain ⟨r, s1, h1, h⟩ := st_bind h
  obtain ⟨-, rfl⟩ := st_pure h
  rw [← Array.forIn_toList] at h1
  have := BodiesOk.forIn_stateT_inv _ (fun _ _ => False) pairs.toList
    (fun j reach s1 => Base desc s1 ∧ Pre (DecState.defs s) (DecState.defs s1) ∧ ReachOk (DecState.defs s1) reach ∧
      ∀ j' (hj : j' < pairs.toList.length), j' < j → Checked s1 pairs.toList[j'])
    _ _ _ _ ⟨hb, pre_refl _, ⟨by simp, fun i hi => absurd hi (by simp)⟩, fun _ _ hj => absurd hj (Nat.not_lt_zero _)⟩
    ?_ h1
  · rcases this with ⟨hb', hp, -, hc⟩ | hf
    · refine ⟨hb', hp, fun p hp' => ?_⟩
      obtain ⟨j, hj, rfl⟩ := List.mem_iff_getElem.1 (by simpa using hp' : p ∈ pairs.toList)
      exact hc j hj hj
    · exact hf.elim
  · intro i hi reach s1 t s2 hI hf
    obtain ⟨hb1, hp1, hr1, hc1⟩ := hI
    have hmem : pairs.toList[i] ∈ pairs := by simpa using List.getElem_mem hi
    obtain ⟨wi, hwi⟩ := hw _ hmem
    rcases hx : pairs.toList[i] with ⟨w, bb⟩
    rw [hx] at hf hwi
    dsimp only at hf hwi
    split at hf
    · exact absurd hf (by intro h; injection h)
    obtain ⟨st0, s3, hg, hf⟩ := st_bind hf
    obtain ⟨rfl, rfl⟩ := st_get hg
    obtain ⟨bi, s4, hd, hf⟩ := st_bind hf
    obtain ⟨st2, s5, hg, hf⟩ := st_bind hf
    obtain ⟨rfl, rfl⟩ := st_get hg
    obtain ⟨u, s6, hl, hf⟩ := st_bind hf
    obtain ⟨hck, rfl⟩ := st_liftV hl
    obtain ⟨rfl, rfl⟩ := st_pure hf
    obtain ⟨hb2, hp2, hi2, hlt2⟩ := getDef_run desc _ hd hb1
    obtain ⟨hr3, hrs3⟩ := extendReach_ok (nodesBefore_of hb2.2) (reachOk_mono hp2 hr1)
    have hwi1 := idx_keep hb.1.1 hb2.1.1 (pre_trans hp1 hp2) hwi
    refine ⟨hb2, pre_trans hp1 hp2, hr3, fun j' hj hlt => ?_⟩
    rcases Nat.lt_succ_iff_lt_or_eq.1 hlt with hlt | rfl
    · exact checked_mono hb1.1.1 hb2.1.1 hp2 (hc1 j' hj hlt)
    · rw [hx]
      exact ⟨bi, wi, hi2, hwi1, checkBody_ok hr3 hrs3 hlt2 hwi1 hck⟩

/-! ## `booleanKeys` -/

/-- The word primitives among `prims`, as `booleanKeys` reads them. -/
def IsWord (defs : Array Defn) (index : Std.HashMap String Nat) (f : String) : Bool :=
  (index[f]?.map fun i => isWordPrim defs[i]!.prim f).getD false

theorem booleanKeys_ok {defs : Array Defn} {index : Std.HashMap String Nat} {prims : Array String} {bm : Lean.Json}
    {pairs : Array (String × String)} (h : booleanKeys defs index prims bm = .ok pairs) :
    (∀ f ∈ prims, IsWord defs index f = true → ∃ v, (f, v) ∈ pairs) ∧
      ∀ p ∈ pairs, p.1 ∈ prims ∧ IsWord defs index p.1 = true := by
  unfold booleanKeys at h
  obtain ⟨u, -, h⟩ := Walk.except_bind_ok h
  split at h
  rotate_left; · exact absurd h (by intro h; injection h)
  split at h
  rotate_left; · exact absurd h (by intro h; injection h)
  rename_i pairs0 _
  extract_lets word keys missing words extra jpIn jpOut at h
  split at h
  · exact (BodiesOk.ex_throw_bind h).elim
  rename_i hmiss
  try intro h
  simp (config := { zeta := false }) only [jpOut] at h
  split at h
  · exact (BodiesOk.ex_throw_bind h).elim
  rename_i hextra
  try intro h
  simp (config := { zeta := false }) only [jpIn] at h
  injection h with h
  subst h
  refine ⟨fun f hf hw => ?_, fun p hp => ?_⟩
  · have hfw : f ∈ word := Array.mem_filter.2 ⟨hf, hw⟩
    have hk : keys.contains f = true := by
      cases hc : keys.contains f
      · have hm : f ∈ missing := Array.mem_filter.2 ⟨hfw, by simp [hc]⟩
        have he : missing.isEmpty = true := by simpa using hmiss
        rw [Array.isEmpty_iff] at he
        rw [he] at hm
        simp at hm
      · rfl
    rcases hs_foldl (fun s p => s.insert p.1) (·.1) (fun _ _ => rfl) f pairs0.toList ∅
      (by rw [Array.foldl_toList]; exact hk) with h0 | ⟨x, hx, rfl⟩
    · simp at h0
    · exact ⟨x.2, by simpa using hx⟩
  · have hpw : words.contains p.1 = true := by
      cases hc : words.contains p.1
      · have hm : p.1 ∈ extra := Array.mem_map.2 ⟨p, Array.mem_filter.2 ⟨hp, by simp [hc]⟩, rfl⟩
        have he : extra.isEmpty = true := by simpa using hextra
        rw [Array.isEmpty_iff] at he
        rw [he] at hm
        simp at hm
      · rfl
    rcases hs_foldl (fun s f => s.insert f) id (fun _ _ => rfl) p.1 word.toList ∅
      (by rw [Array.foldl_toList]; exact hpw) with h0 | ⟨x, hx, he⟩
    · simp at h0
    · have := Array.mem_filter.1 (by simpa using hx : x ∈ word)
      subst he
      exact this

/-! ## `decode_boolean` -/

theorem mem_primEntries {kv : Array (String × Lean.Json)} {k : String} {d : Lean.Json} (h : (k, d) ∈ kv)
    (hf : formOf d = "primitive") : k ∈ primEntries kv := by
  unfold primEntries
  refine Array.mem_filterMap.2 ⟨(k, d), h, ?_⟩
  unfold formOf at hf
  cases ho : (d.getObjValAs? String "form").toOption with
  | none => rw [ho] at hf; simp at hf
  | some s =>
    rw [ho] at hf
    simp only [Option.getD_some] at hf
    subst hf
    simp [ho]

/-- **`decode` guarantees the Boolean map (`DecodeBoolean`).** -/
theorem decode_boolean : DecodeBoolean := by
  intro bytes P h
  obtain ⟨kv, desc, rootId, programFn, bm, root, s0, st, st1, st2, pairs, hdd, hgo, hd0, hi0', -, -, -, hall, hbk, hbod,
    hd, hi, -, hbo⟩ := decode_steps bytes P h
  have hk0 := of_triple (go_kept _ _ _) hgo (DecodeKeys.kept_empty desc hd0 hi0')
  obtain ⟨hi0, -⟩ := go_inv _ _ _ _ _ _ hgo
    ⟨fun i hi => absurd hi (by rw [hd0]; exact Nat.not_lt_zero i), fun k v hk => absurd hk (by rw [hi0']; simp)⟩
  obtain ⟨hb1, hp1, hall'⟩ := decodeAll_run desc _ hall ⟨hk0, hi0⟩
  obtain ⟨hmiss, hextra⟩ := booleanKeys_ok hbk
  have hw : ∀ p ∈ pairs, ∃ wi, (DecState.index st1)[p.1]? = some wi := fun p hp => by
    have hg := (hextra p hp).2
    unfold IsWord at hg
    cases hix : (DecState.index st1)[p.1]? with
    | none => rw [hix] at hg; simp at hg
    | some wi => exact ⟨wi, rfl⟩
  obtain ⟨hb2, hp2, hch⟩ := decodeBodies_run desc _ _ hbod hb1 hw
  have hK1 := hb1.1.1
  have hK2 := hb2.1.1
  have hfold := fun k => hm_foldl_some (α := String) (β := String) (γ := String × String)
    (fun m p => m.insert p.1 p.2) id (fun _ _ => rfl) k
  refine ⟨fun w hw' hword => ?_, fun k v hkv => ?_⟩
  · rw [hd] at hw' hword ⊢
    have hprim : ((DecState.defs st2)[w]!).prim = true := by
      simp only [isWordPrim, Bool.and_eq_true] at hword
      exact hword.1.1
    obtain ⟨d, hdd', hform⟩ := hb2.1.2.2.2 w hw' hprim
    rw [hdd, ← Array.foldl_toList] at hdd'
    have hmem : ((DecState.defs st2)[w]!.key, d) ∈ kv := by
      rcases hm_foldl_some (fun m x => m.insert x.1 x.2) id (fun _ _ => rfl) _ _ kv.toList ∅ hdd' with h0 | ⟨x, hx, he⟩
      · simp at h0
      · simp only [id] at he
        subst he
        simpa using hx
    have hkp := mem_primEntries hmem hform
    have hcont : (Desc.ddefs desc).contains ((DecState.defs st2)[w]!.key) = true := by
      rw [hdd, ← Array.foldl_toList, Std.HashMap.contains_eq_isSome_getElem?, hdd']
      rfl
    obtain ⟨j, hj⟩ := hall' _ hkp hcont
    have hj2 := idx_keep hK1 hK2 hp2 hj
    have hw2 := hK2.2.1 w hw'
    have hjw : j = w := by rw [hj2] at hw2; exact Option.some.inj hw2
    subst hjw
    have hjlt := (hK1.1 _ _ hj).1
    have hword1 : IsWord (DecState.defs st1) (DecState.index st1) ((DecState.defs st2)[j]!.key) = true := by
      unfold IsWord
      rw [hj]
      simp only [Option.map_some, Option.getD_some]
      rw [← hp2.2 _ hjlt]
      exact hword
    obtain ⟨v, hv⟩ := hmiss _ hkp hword1
    have hsome := hm_foldl_isSome (α := String) (β := String) (γ := String × String) (fun m p => m.insert p.1 p.2) id (fun _ _ => rfl) ((DecState.defs st2)[j]!.key)
      pairs.toList ∅ (Or.inr ⟨_, by simpa using hv, rfl⟩)
    rw [Array.foldl_toList] at hsome
    obtain ⟨v', hv'⟩ := Option.isSome_iff_exists.1 hsome
    have hv'' := hv'
    rw [← Array.foldl_toList] at hv''
    have hmem' : ((DecState.defs st2)[j]!.key, v') ∈ pairs := by
      rcases hfold _ v' pairs.toList ∅ hv'' with h0 | ⟨x, hx, he⟩
      · simp at h0
      · simp only [id] at he
        subst he
        simpa using hx
    obtain ⟨bi, wi, h1, h2, h3⟩ := hch _ hmem'
    have hwi : wi = j := by rw [hw2] at h2; exact (Option.some.inj h2).symm
    subst hwi
    refine ⟨bi, (hK2.1 _ _ h1).1, ?_, ?_⟩
    · show (P.boolean[_]? >>= (P.index[·]?)) = some bi
      rw [hbo, hv', hi]
      exact h1
    · rw [← hd] at h3
      exact bodyOf_of h3
  · rw [hbo, ← Array.foldl_toList] at hkv
    have hmem : (k, v) ∈ pairs := by
      rcases hfold k v pairs.toList ∅ hkv with h0 | ⟨x, hx, he⟩
      · simp at h0
      · simp only [id] at he
        subst he
        simpa using hx
    obtain ⟨-, hg⟩ := hextra _ hmem
    unfold IsWord at hg
    cases hix : (DecState.index st1)[k]? with
    | none => rw [hix] at hg; simp at hg
    | some i =>
      rw [hix] at hg
      simp only [Option.map_some, Option.getD_some] at hg
      have hilt := (hK1.1 _ _ hix).1
      have hi2 := idx_keep hK1 hK2 hp2 hix
      obtain ⟨hilt2, hkey⟩ := hK2.1 _ _ hi2
      refine ⟨i, by rw [hd]; exact hilt2, by rw [hi]; exact hi2, by rw [hd]; exact hkey, ?_⟩
      rw [hd, hp2.2 _ hilt]
      exact hg

end FlockSoundness.DecodeBoolean
