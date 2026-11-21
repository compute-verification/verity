module

public import Properties.Flock.DecodeBoolean
public import Proofs.Properties.Flock.Decoder.DecodeKeys
public import Proofs.Properties.Flock.EvalBool.Fuel
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
  DecState.index DecState.busy Desc Desc.ddefs liftV from Circuits.Descriptor

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
  have hk := of_triple ((kept_fuel Flock.Steps.FUEL false).1 desc fid (fun _ => False)) h hb.1
  obtain ⟨m, hm⟩ := fuel_succ
  rw [hm] at h
  have hi := of_triple (getDef_idx m (kept_fuel m false).2.2 desc fid _) h hb.1
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

theorem params_beq {a b : Array (String × Ty)} (h : (a == b) = true) : a = b := by
  obtain ⟨hs, hr⟩ := Array.isEqv_iff_rel.1 h
  refine Array.ext hs (fun i h1 h2 => ?_)
  have hi := hr i h1
  generalize a[i] = p at hi ⊢
  generalize b[i] = q at hi ⊢
  obtain ⟨p1, p2⟩ := p
  obtain ⟨q1, q2⟩ := q
  change (p1 == q1 && p2 == q2) = true at hi
  rw [Bool.and_eq_true, beq_iff_eq] at hi
  rw [hi.1, BodiesOk.ty_beq hi.2]

theorem forIn_ok_all {α ε : Type} {f : α → PUnit → Except ε (ForInStep PUnit)} {Q : α → Prop}
    (hf : ∀ x r, f x PUnit.unit = .ok r → Q x ∧ r = .yield PUnit.unit) :
    ∀ (l : List α) {u : PUnit}, forIn l PUnit.unit f = .ok u → ∀ x ∈ l, Q x
  | [], _, _, x, hx => absurd hx List.not_mem_nil
  | y :: l, u, h, x, hx => by
    rw [List.forIn_cons] at h
    cases hy : f y PUnit.unit with
    | error e => rw [hy] at h; cases h
    | ok r =>
      obtain ⟨hq, rfl⟩ := hf y r hy
      rw [hy] at h
      rcases List.mem_cons.1 hx with rfl | hx
      · exact hq
      · exact forIn_ok_all hf l h x hx

/-- `booleanKeys` accepts only Definitions whose primitives are spelt as their family's ids and whose gates have
§9's signature. -/
theorem booleanKeys_sig {defs : Array Defn} {index : Std.HashMap String Nat} {prims : Array String} {bm : Lean.Json}
    {pairs : Array (String × String)} (h : booleanKeys defs index prims bm = .ok pairs) :
    ∀ d ∈ defs, d.prim = true → misspelt d.key = false ∧
      ∀ g, gateKind? d.key = some g → d.params = g.signature.1 ∧ d.ret = g.signature.2 := by
  unfold booleanKeys at h
  obtain ⟨u, h1, -⟩ := Walk.except_bind_ok h
  rw [← Array.forIn_toList] at h1
  intro d hd
  refine forIn_ok_all (Q := fun d => d.prim = true → misspelt d.key = false ∧
      ∀ g, gateKind? d.key = some g → d.params = g.signature.1 ∧ d.ret = g.signature.2) ?_ _ h1 d
    (Array.mem_toList_iff.2 hd)
  intro x r hx
  dsimp only at hx
  split at hx
  · cases hx
  rename_i hms
  split at hx
  · rename_i g hg
    generalize hsig : g.signature = sg at hx
    obtain ⟨ps, rt⟩ := sg
    dsimp only at hx
    split at hx
    · cases hx
    rename_i hbeq
    cases hx
    refine ⟨fun hp => ⟨by simpa [hp] using hms, fun g' hg' => ?_⟩, rfl⟩
    simp only [hp, ↓reduceIte, hg'] at hg
    cases hg
    simp only [Bool.not_eq_true', Bool.not_eq_false, Bool.and_eq_true] at hbeq
    rw [hsig]
    exact ⟨params_beq hbeq.1, BodiesOk.ty_beq hbeq.2⟩
  · rename_i hg
    cases hx
    refine ⟨fun hp => ⟨by simpa [hp] using hms, fun g' hg' => ?_⟩, rfl⟩
    simp [hp, hg'] at hg

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

/-- **What `decode` guarantees of its primitives**: the Boolean map (`Prog.BoolMapOk`), and every primitive keyed by
its id, no gate family member spelt otherwise (`misspelt`), and each gate of §9's signature. -/
theorem decode_facts (bytes : ByteArray) (P : Prog) (h : decode bytes = .ok P)
    (hr : (P.defs[P.root]!).body.recut = #[] := by assumption) :
    P.BoolMapOk ∧ ∀ w < P.defs.size, (P.defs[w]!).prim = true →
      (P.defs[w]!).id = (P.defs[w]!).key ∧ misspelt (P.defs[w]!).key = false ∧
        ∀ g, gateKind? (P.defs[w]!).key = some g →
          (P.defs[w]!).params = g.signature.1 ∧ (P.defs[w]!).ret = g.signature.2 := by
  obtain ⟨kv, desc, rootId, programFn, bm, root, s0, st, st1, st2, pairs, hdd, hgo, hd0, hi0', -, -, -, hall, hbk, hbod,
    hd, hi, -, hbo⟩ := decode_steps bytes P h hr
  have hk0 := of_triple (go_kept _ _ _ false) hgo (DecodeKeys.kept_empty desc hd0 hi0')
  obtain ⟨hi0, -⟩ := go_inv _ _ _ _ _ _ hgo
    ⟨fun i hi => absurd hi (by rw [hd0]; exact Nat.not_lt_zero i), fun k v hk => absurd hk (by rw [hi0']; simp),
      fun i hi => absurd hi (by rw [hd0]; exact Nat.not_lt_zero i),
      fun i hi => absurd hi (by rw [hd0]; exact Nat.not_lt_zero i)⟩
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
  have hearly : ∀ w < (DecState.defs st2).size, ((DecState.defs st2)[w]!).prim = true →
      w < (DecState.defs st1).size ∧ (DecState.index st1)[(DecState.defs st2)[w]!.key]? = some w ∧
      ((DecState.defs st2)[w]!).id = ((DecState.defs st2)[w]!).key ∧ (DecState.defs st2)[w]!.key ∈ primEntries kv := by
    intro w hw' hprim
    obtain ⟨⟨d, hdd', hform⟩, hid⟩ := hb2.1.2.2.2 w hw' hprim
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
    exact ⟨(hK1.1 _ _ hj).1, hj, hid, hkp⟩
  refine ⟨⟨fun w hw' hword => ?_, fun k v hkv => ?_⟩, fun w hw' hprim => ?_⟩
  · rw [hd] at hw' hword ⊢
    have hprim : ((DecState.defs st2)[w]!).prim = true := by
      simp only [isWordPrim, Bool.and_eq_true] at hword
      exact hword.1.1
    obtain ⟨hjlt, hj, -, hkp⟩ := hearly w hw' hprim
    have hw2 := hK2.2.1 w hw'
    have hword1 : IsWord (DecState.defs st1) (DecState.index st1) ((DecState.defs st2)[w]!.key) = true := by
      unfold IsWord
      rw [hj]
      simp only [Option.map_some, Option.getD_some]
      rw [← hp2.2 _ hjlt]
      exact hword
    obtain ⟨v, hv⟩ := hmiss _ hkp hword1
    have hsome := hm_foldl_isSome (α := String) (β := String) (γ := String × String) (fun m p => m.insert p.1 p.2) id
      (fun _ _ => rfl) ((DecState.defs st2)[w]!.key) pairs.toList ∅ (Or.inr ⟨_, by simpa using hv, rfl⟩)
    rw [Array.foldl_toList] at hsome
    obtain ⟨v', hv'⟩ := Option.isSome_iff_exists.1 hsome
    have hv'' := hv'
    rw [← Array.foldl_toList] at hv''
    have hmem' : ((DecState.defs st2)[w]!.key, v') ∈ pairs := by
      rcases hfold _ v' pairs.toList ∅ hv'' with h0 | ⟨x, hx, he⟩
      · simp at h0
      · simp only [id] at he
        subst he
        simpa using hx
    obtain ⟨bi, wi, h1, h2, h3⟩ := hch _ hmem'
    have hwi : wi = w := by rw [hw2] at h2; exact (Option.some.inj h2).symm
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
  · rw [hd] at hw' hprim ⊢
    obtain ⟨hjlt, -, hid, -⟩ := hearly w hw' hprim
    refine ⟨hid, ?_⟩
    rw [hp2.2 _ hjlt] at hprim ⊢
    exact booleanKeys_sig hbk _ (by rw [getElem!_pos (DecState.defs st1) w hjlt]; exact Array.getElem_mem hjlt) hprim

/-- **`decode` guarantees the Boolean map (`DecodeBoolean`).** -/
theorem decode_boolean : DecodeBoolean := fun bytes P h hr => (decode_facts bytes P h hr).1

/-! ## `decode_body_agrees` -/

/-- A Slice whose first digits, before its last three characters, are digits starts with a digit. -/
theorem front_digit (S : String.Slice) (hl : (S.dropEnd 3).toString.length > 0)
    (ha : (S.dropEnd 3).toString.all Char.isDigit = true) : S.toString.front.isDigit = true := by
  simp only [String.Slice.toString_eq] at hl ha ⊢
  rw [← String.length_toList, String.Slice.toList_copy_dropEnd] at hl
  rw [String.all_bool_eq, String.Slice.toList_copy_dropEnd] at ha
  rw [String.front_eq, String.front?_eq]
  generalize S.copy.toList = L at *
  cases L with
  | nil => simp at hl
  | cons c L =>
    obtain ⟨m, hm⟩ : ∃ m, (c :: L).length - 3 = m + 1 := ⟨(c :: L).length - 3 - 1, by simp at hl ⊢ <;> omega⟩
    rw [hm, List.take_succ_cons, List.all_cons, Bool.and_eq_true] at ha
    simpa using ha.1

/-- A member of `Flock.Lower`'s gate family `pre` is spelt `pre`, then a digit. -/
theorem family_front {pre k : String} (h : Flock.Lower.family? pre k = true) :
    k.startsWith pre = true ∧ (k.drop pre.length).toString.front.isDigit = true := by
  unfold Flock.Lower.family? at h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  exact ⟨h.1.1, front_digit _ h.2.1.1 h.2.1.2⟩

theorem word_ids {prim : Bool} {k : String} (h : isWordPrim prim k = true) :
    isInputId k = false ∧ isBooleanPrim k = false := by
  unfold isWordPrim at h
  cases h1 : isInputId k <;> cases h2 : isBooleanPrim k <;> simp_all

/-- **A word primitive's id that is no misspelt family member is no gate of `Flock.Lower`.** `gateOf?` and `gateKind?`
differ only on `And1_v1` and `Xor1_v1`, which `misspelt` holds of. -/
theorem gateOf_none {k : String} (hb : isBooleanPrim k = false) (hm : misspelt k = false)
    (hi : isInputId k = false) : Flock.Lower.gateOf? k = none := by
  have hg : gateKind? k = none := by
    cases h : gateKind? k with
    | none => rfl
    | some g => simp [isBooleanPrim, h] at hb
  have hfam : ∀ pre ∈ ["Input", "RegisteredInput", "SuppliedInput", "And", "Xor"], Flock.Lower.family? pre k = true → False := by
    intro pre hpre hf
    obtain ⟨hs, hd⟩ := family_front hf
    have hany : ["Input", "RegisteredInput", "SuppliedInput", "And", "Xor"].any (fun pre =>
        k.startsWith pre && (k.drop pre.length).toString.front.isDigit) = true :=
      List.any_eq_true.2 ⟨pre, hpre, by simp only [hs, hd, Bool.and_self]⟩
    have : misspelt k = true := by
      unfold misspelt
      rw [hany, hi, hg]
      rfl
    rw [hm] at this
    cases this
  unfold Flock.Lower.gateOf?
  split
  · rename_i h; obtain rfl := eq_of_beq h; simp [gateKind?] at hg
  split
  · rename_i h; obtain rfl := eq_of_beq h; simp [gateKind?] at hg
  split
  · rename_i h; obtain rfl := eq_of_beq h; simp [gateKind?] at hg
  split
  · exact (hfam "And" (by simp) ‹_›).elim
  split
  · exact (hfam "Xor" (by simp) ‹_›).elim
  rfl

open EvalBool (Refines evalDef?_succ evalStep_mono_at evalDef?_mono_succ evalDef?_body)

/-- A key the Boolean map leads from is no Boolean primitive's. -/
theorem bool_none {P : Prog} (h2 : ∀ k v, P.boolean[k]? = some v →
      ∃ w < P.defs.size, P.index[k]? = some w ∧ (P.defs[w]!).key = k ∧ isWordPrim (P.defs[w]!).prim k = true)
    (k : String) (hk : isBooleanPrim k = true) : P.bodyOf? k = none := by
  unfold Prog.bodyOf?
  cases hv : P.boolean[k]? with
  | none => rfl
  | some v =>
    obtain ⟨w, -, -, -, hw⟩ := h2 k v hv
    simp [isWordPrim, hk] at hw

theorem get!_of {P : Prog} {d : Nat} {defn : Defn} (h : P.defs[d]? = some defn) : P.defs[d]! = defn := by
  simp [getElem!_def, h]

theorem prim_down (P : Prog) (hB : ∀ k, isBooleanPrim k = true → P.bodyOf? k = none) {e : Nat}
    (hp : (P.defs[e]!).prim = true) (hbp : isBooleanPrim (P.defs[e]!).key = true) (k : Nat) (x : List Bool) :
    Refines (evalDef? P (k + 1 + 1) e x) (evalDef? P (k + 1) e x) := by
  rw [evalDef?_succ P (k + 1), evalDef?_succ P k]
  refine evalStep_mono_at P (fun defn hd _ b hb => ?_) (fun defn hd hpf => ?_) x
  · rw [← get!_of hd, hB _ hbp] at hb
    cases hb
  · rw [← get!_of hd, hp] at hpf
    cases hpf

/-- **A Boolean Definition's value needs no fuel beyond its place:** a value at fuel `m + 1 > d + 1` is the value at
fuel `m`. -/
theorem fuel_down (P : Prog) (hB : ∀ k, isBooleanPrim k = true → P.bodyOf? k = none) {d : Nat} (hd : P.BoolDef d) :
    ∀ m, d < m → ∀ x, Refines (evalDef? P (m + 1) d x) (evalDef? P m d x) := by
  induction hd with
  | prim hp hbp =>
    intro m hm x
    obtain ⟨k, rfl⟩ : ∃ k, m = k + 1 := ⟨m - 1, by omega⟩
    exact prim_down P hB hp hbp k x
  | @comp d hpf hprimc _ _ ih =>
    intro m hm x
    obtain ⟨k, rfl⟩ : ∃ k, m = k + 1 := ⟨m - 1, by omega⟩
    rw [evalDef?_succ P (k + 1), evalDef?_succ P k]
    refine evalStep_mono_at P (fun defn hd hpt => ?_) (fun defn hd _ nd hnd hfn ins => ?_) x
    · rw [← get!_of hd, hpf] at hpt
      cases hpt
    · rw [← get!_of hd] at hnd
      have hmem : nd ∈ (P.defs[d]!).body.nodes := Array.mem_def.2 hnd
      cases hc : (P.defs[nd.fn]!).prim
      · exact ih nd hmem hc k (by omega) ins
      · obtain ⟨j, rfl⟩ : ∃ j, k = j + 1 := ⟨k - 1, by omega⟩
        exact prim_down P hB hc (hprimc nd hmem hc) j ins

theorem fuel_stable (P : Prog) (hB : ∀ k, isBooleanPrim k = true → P.bodyOf? k = none) {b m : Nat}
    (hb : P.BoolDef b) (hm : b < m) (x : List Bool) : evalDef? P (m + 1) b x = evalDef? P m b x := by
  cases h : evalDef? P m b x with
  | some v => exact evalDef?_mono_succ P m b x v h
  | none =>
    cases h' : evalDef? P (m + 1) b x with
    | none => rfl
    | some v =>
      rw [fuel_down P hB hb m hm x v h'] at h
      cases h

/-- **`decode` guarantees `BodyAgrees` (`DecodeBodyAgrees`).** -/
theorem decode_body_agrees : DecodeBodyAgrees := by
  intro bytes P h hr p hp b hprim hb
  obtain ⟨⟨hmap, hkeys⟩, hids⟩ := decode_facts bytes P h hr
  have hidx := (DecodeKeys.decode_keys bytes P h).2 p hp
  have hb0 := hb
  unfold Prog.bodyOf? at hb0
  cases hv : P.boolean[(P.defs[p]!).key]? with
  | none => rw [hv] at hb0; simp at hb0
  | some v =>
    obtain ⟨w, -, hiw, -, hword⟩ := hkeys _ _ hv
    rw [hidx] at hiw
    obtain rfl := Option.some.inj hiw
    obtain ⟨b', hb', hbo, hbody⟩ := hmap p hp hword
    obtain rfl : b = b' := (Option.some.inj (hbo.symm.trans hb)).symm
    obtain ⟨hid, hmis, -⟩ := hids p hp hprim
    obtain ⟨hin, hbp⟩ := word_ids hword
    have hg : Flock.Lower.gateOf? (P.defs[p]!).id = none := by
      rw [hid]
      exact gateOf_none hbp hmis hin
    have hd : P.defs[p]? = some (P.defs[p]!) := by rw [getElem!_pos P.defs p hp]; exact getElem?_pos P.defs p hp
    funext x
    unfold Prog.evalBool
    rw [show 2 * P.defs.size + 2 = (2 * P.defs.size + 1) + 1 from rfl, evalDef?_body P hd hprim hg hb,
      fuel_stable P (bool_none hkeys) hbody.1 (by omega : b < 2 * P.defs.size + 1) x]

end FlockSoundness.DecodeBoolean
