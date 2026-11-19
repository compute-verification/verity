module

public import Proofs.Properties.Flock.Decoder.Decoded
public import Std.Tactic.Do
import Batteries.Tactic.OpenPrivate
import Std.Data.HashMap.Lemmas
import Std.Data.HashSet.Lemmas

/-!
# The keys of a decoded Program

`decode_keys`: in a Program `Flock.Program.decode` returns, the index sends a key to the place of the Definition that
has it, and each Definition's key to its own place, so no two Definitions share a key. `root_first` is the case the
record's units read: no Definition before the root has the root's key. `decode_ret`: a composite Definition returns its
body's return type.

The decoder (`getDef`, `composite`, `staticFns`) is walked once, as Hoare triples proved by `mvcgen` and read back at an
`.ok` result (`of_triple`), for either walk. `composite_step` and `staticFns_step` (beside `BodiesOk.decode_walk`) hold of
any property of the decoder's state that `getDef` and `staticFns` keep one fuel lower, for one descriptor. `decode_kept`
reads `decode`'s steps from `BodiesOk.decode_walk`, through `go_kept`, `decodeAll_kept` and `decodeBodies_kept`. `kept_fuel` is the induction on
the fuel for `Kept`: the keys (`Keys`), the return types (`Rets`), the names that stay busy, and each primitive's entry
of `definitions` (`Src`, which `FlockSoundness.DecodeBoolean` reads). A name is busy from `composite`'s mark to
`getDef`'s push and a busy name has no index entry, so the push finds no earlier Definition with its key.
-/

@[expose] public section

set_option mvcgen.warning false

namespace InferenceOnlyVerifier.Properties.SecurityProofs.DecodeKeys
open Flock.Program
open Std.Do
open BodiesOk (DM ex_throw_bind decode_root)
open private getDef composite staticFns decodeAll decodeBodies goOf decodeDefs_walk DecState DecState.defs DecState.index DecState.busy Desc Desc.ddefs liftV
  from Circuits.Descriptor

/-! ## What the decoder keeps -/

/-- **The decoder's keys.** The index sends a key to the place of the Definition that has it, each Definition's key to
its own place, and a busy name nowhere. -/
def Keys (s : DecState) : Prop :=
  (∀ k v, (DecState.index s)[k]? = some v → v < (DecState.defs s).size ∧ ((DecState.defs s)[v]!).key = k) ∧
    (∀ i, i < (DecState.defs s).size → (DecState.index s)[((DecState.defs s)[i]!).key]? = some i) ∧
      ∀ k, (DecState.busy s).contains k = true → (DecState.index s)[k]? = none

/-- A composite Definition returns its body's return type. -/
def Rets (s : DecState) : Prop :=
  ∀ i, i < (DecState.defs s).size → ((DecState.defs s)[i]!).prim = false →
    ((DecState.defs s)[i]!).body.retType = ((DecState.defs s)[i]!).ret

/-- The form a definition entry declares (`getDef`'s `form`). -/
def formOf (d : Lean.Json) : String := (d.getObjValAs? String "form").toOption.getD ""

/-- Each primitive Definition decoded so far is keyed by an entry of `definitions` of form `primitive`, and by its id. -/
def Src (desc : Desc) (s : DecState) : Prop :=
  ∀ i < (DecState.defs s).size, ((DecState.defs s)[i]!).prim = true →
    (∃ d, (Desc.ddefs desc)[((DecState.defs s)[i]!).key]? = some d ∧ formOf d = "primitive") ∧
      ((DecState.defs s)[i]!).id = ((DecState.defs s)[i]!).key

theorem src_push {desc : Desc} {s s' : DecState} {defn : Defn} (h : Src desc s)
    (hd : DecState.defs s' = (DecState.defs s).push defn)
    (hn : defn.prim = true →
      (∃ d, (Desc.ddefs desc)[defn.key]? = some d ∧ formOf d = "primitive") ∧ defn.id = defn.key) :
    Src desc s' := by
  intro i hi hp
  rw [hd] at hi hp ⊢
  simp only [Array.size_push] at hi
  rcases Nat.lt_succ_iff_lt_or_eq.1 hi with hi | rfl
  · rw [BodiesOk.getElem!_push_lt _ _ hi] at hp ⊢
    exact h i hi hp
  · rw [BodiesOk.getElem!_push_eq] at hp ⊢
    exact hn hp

/-- What the decoder keeps while the names of `B` are being decoded: `Keys`, `Rets`, each name of `B` busy, and
`Src`. -/
def Kept (desc : Desc) (B : String → Prop) (s : DecState) : Prop :=
  Keys s ∧ Rets s ∧ (∀ k, B k → (DecState.busy s).contains k = true) ∧ Src desc s

private theorem kept_congr (desc : Desc) (B : String → Prop) (s s' : DecState)
    (hd : DecState.defs s' = DecState.defs s) (hi : DecState.index s' = DecState.index s)
    (hb : DecState.busy s' = DecState.busy s) (h : Kept desc B s) : Kept desc B s' := by
  unfold Kept Keys Rets Src
  rw [hd, hi, hb]
  exact h

/-- The decoder's first state, with no Definition and no index entry, has `Kept` with no name busy. -/
theorem kept_empty (desc : Desc) {s : DecState} (hd : DecState.defs s = #[]) (hi : DecState.index s = ∅) :
    Kept desc (fun _ => False) s := by
  have hn : ∀ i, ¬ i < (DecState.defs s).size := fun i => by rw [hd]; exact Nat.not_lt_zero i
  refine ⟨⟨fun k v h => ?_, fun i h => absurd h (hn i), fun k _ => ?_⟩, fun i h => absurd h (hn i),
    fun _ h => h.elim, fun i h => absurd h (hn i)⟩
  · rw [hi] at h; exact absurd h (by simp)
  · rw [hi]; simp

/-- Marking a name without an index entry busy keeps `Kept`, with that name among the busy ones. -/
private theorem kept_busy {desc : Desc} {B : String → Prop} {s s1 : DecState} {fid : String} (h : Kept desc B s)
    (hnone : (DecState.index s)[fid]? = none)
    (hd : DecState.defs s1 = DecState.defs s) (hi : DecState.index s1 = DecState.index s)
    (hb : DecState.busy s1 = (DecState.busy s).insert fid) : Kept desc (fun k => B k ∨ k = fid) s1 := by
  obtain ⟨⟨k1, k2, k3⟩, hr, hB, hs⟩ := h
  refine ⟨⟨?_, ?_, ?_⟩, ?_, ?_, by unfold Src; rw [hd]; exact hs⟩
  · rw [hd, hi]; exact k1
  · rw [hd, hi]; exact k2
  · intro k hk
    rw [hb, Std.HashSet.contains_insert] at hk
    rw [hi]
    rcases Bool.or_eq_true_iff.mp hk with h1 | h1
    · rw [← eq_of_beq h1]; exact hnone
    · exact k3 k h1
  · unfold Rets
    rw [hd]; exact hr
  · intro k hk
    rw [hb, Std.HashSet.contains_insert]
    rcases hk with h1 | h1
    · simp [hB k h1]
    · simp [h1]

/-- `getDef`'s push of a Definition keyed `fid`, a name without an index entry, keeps `Keys` and `Rets`, and the names
of `B` busy when none of them is `fid`. -/
private theorem kept_push {desc : Desc} {B : String → Prop} {s s' : DecState} {defn : Defn} {fid : String}
    (hk : Keys s) (hr : Rets s) (hs : Src desc s)
    (hn : defn.prim = true →
      (∃ d, (Desc.ddefs desc)[defn.key]? = some d ∧ formOf d = "primitive") ∧ defn.id = defn.key)
    (hnone : (DecState.index s)[fid]? = none)
    (hsub : ∀ k, B k → (DecState.busy s).contains k = true ∧ k ≠ fid) (hkey : defn.key = fid)
    (hret : defn.prim = false → defn.body.retType = defn.ret)
    (hd : DecState.defs s' = (DecState.defs s).push defn)
    (hi : DecState.index s' = (DecState.index s).insert fid (DecState.defs s).size)
    (hb : DecState.busy s' = (DecState.busy s).erase fid) : Kept desc B s' := by
  obtain ⟨k1, k2, k3⟩ := hk
  refine ⟨⟨?_, ?_, ?_⟩, ?_, ?_, src_push hs hd hn⟩
  · intro k v hv
    rw [hi, Std.HashMap.getElem?_insert] at hv
    rw [hd, Array.size_push]
    by_cases hf : (fid == k) = true
    · simp only [hf, ↓reduceIte] at hv
      have hv' : (DecState.defs s).size = v := Option.some.inj hv
      subst hv'
      exact ⟨Nat.lt_succ_self _, by rw [BodiesOk.getElem!_push_eq, hkey, eq_of_beq hf]⟩
    · simp only [hf, Bool.false_eq_true, ↓reduceIte] at hv
      obtain ⟨h1, h2⟩ := k1 k v hv
      exact ⟨Nat.lt_succ_of_lt h1, by rw [BodiesOk.getElem!_push_lt _ _ h1, h2]⟩
  · intro i hlt
    rw [hd, Array.size_push] at hlt
    rw [hi, hd, Std.HashMap.getElem?_insert]
    by_cases he : i = (DecState.defs s).size
    · subst he
      simp [hkey]
    · have hlt' : i < (DecState.defs s).size := Nat.lt_of_le_of_ne (Nat.le_of_lt_succ hlt) he
      rw [BodiesOk.getElem!_push_lt _ _ hlt']
      have h2 := k2 i hlt'
      have hne : ¬ (fid == ((DecState.defs s)[i]!).key) = true := by
        intro hf
        rw [← eq_of_beq hf, hnone] at h2
        exact absurd h2 (by simp)
      simp only [hne, Bool.false_eq_true, ↓reduceIte]
      exact h2
  · intro k hc
    rw [hb, Std.HashSet.contains_erase] at hc
    rw [hi, Std.HashMap.getElem?_insert]
    have h1 := Bool.and_eq_true_iff.mp hc
    have hne : ¬ (fid == k) = true := by
      intro hf
      simp [hf] at h1
    simp only [hne, Bool.false_eq_true, ↓reduceIte]
    exact k3 k h1.2
  · intro i hlt
    rw [hd, Array.size_push] at hlt
    rw [hd]
    by_cases he : i = (DecState.defs s).size
    · subst he
      rw [BodiesOk.getElem!_push_eq]
      exact hret
    · have hlt' : i < (DecState.defs s).size := Nat.lt_of_le_of_ne (Nat.le_of_lt_succ hlt) he
      rw [BodiesOk.getElem!_push_lt _ _ hlt']
      exact hr i hlt'
  · intro k hBk
    obtain ⟨h1, h2⟩ := hsub k hBk
    rw [hb, Std.HashSet.contains_erase, h1]
    have : ¬ (fid == k) = true := fun hf => h2 (eq_of_beq hf).symm
    simp [this]

private theorem none_of {o : Option Nat} (h : ∀ i, o = some i → False) : o = none := by
  cases o with
  | none => rfl
  | some i => exact (h i rfl).elim

/-- `getDef`'s push of a primitive, in the words of `getDef`'s own conditions. -/
private theorem push_prim {desc : Desc} {B : String → Prop} {s s' : DecState} {fid : String} {o : Option Nat}
    {defn : Defn} {d : Lean.Json}
    (h : Kept desc B s) (ho : (DecState.index s)[fid]? = o) (hno : ∀ i, o = some i → False)
    (hb : ¬ (DecState.busy s).contains fid = true) (hdd : (Desc.ddefs desc)[fid]? = some d)
    (hf : ((d.getObjValAs? String "form").toOption.getD "" == "primitive") = true)
    (hid : ¬ (defn.id != fid) = true) (hkey : defn.key = fid)
    (hprim : defn.prim = true) (hd : DecState.defs s' = (DecState.defs s).push defn)
    (hi : DecState.index s' = (DecState.index s).insert fid (DecState.defs s).size)
    (hb' : DecState.busy s' = (DecState.busy s).erase fid) : Kept desc B s' :=
  kept_push h.1 h.2.1 h.2.2.2
    (fun _ => ⟨⟨d, by rw [hkey]; exact hdd, eq_of_beq hf⟩, by rw [hkey]; simpa using hid⟩) (ho.trans (none_of hno))
    (fun k hk => ⟨h.2.2.1 k hk, fun e => hb (e ▸ h.2.2.1 k hk)⟩) hkey
    (fun hp => absurd (hprim.symm.trans hp) (by decide)) hd hi hb'

/-- `composite`'s mark, in the words of `getDef`'s own conditions. -/
private theorem busy_pre {desc : Desc} {B : String → Prop} {s : DecState} {fid : String} {o : Option Nat}
    (h : Kept desc B s) (ho : (DecState.index s)[fid]? = o) (hno : ∀ i, o = some i → False) (s1 : DecState)
    (hd : DecState.defs s1 = DecState.defs s) (hi : DecState.index s1 = DecState.index s)
    (hb : DecState.busy s1 = (DecState.busy s).insert fid) : Kept desc (fun k => B k ∨ k = fid) s1 :=
  kept_busy h (ho.trans (none_of hno)) hd hi hb

/-- `getDef`'s push of what `composite` returned: `fid` is still busy there, so it has no index entry. -/
private theorem push_comp {desc : Desc} {B : String → Prop} {s0 s s' : DecState} {fid : String} {defn : Defn}
    (h0 : Kept desc B s0) (hb : ¬ (DecState.busy s0).contains fid = true)
    (h : Kept desc (fun k => B k ∨ k = fid) s ∧ defn.key = fid ∧ defn.prim = false ∧
      defn.body.retType = defn.ret)
    (hd : DecState.defs s' = (DecState.defs s).push defn)
    (hi : DecState.index s' = (DecState.index s).insert fid (DecState.defs s).size)
    (hb' : DecState.busy s' = (DecState.busy s).erase fid) : Kept desc B s' :=
  kept_push h.1.1 h.1.2.1 h.1.2.2.2 (fun hp => absurd (h.2.2.1.symm.trans hp) (by decide))
    (h.1.1.2.2 fid (h.1.2.2.1 fid (Or.inr rfl)))
    (fun k hk => ⟨h.1.2.2.1 k (Or.inl hk), fun e => hb (e ▸ h0.2.2.1 k hk)⟩) h.2.1 (fun _ => h.2.2.2) hd hi hb'

/-- `composite` at fuel `n` keeps `Kept`: from a state that has it once `fid` is busy, it ends in a state that has it,
and what it returns is a composite keyed `fid` that returns its body's return type. -/
def CompositeKept (n : Nat) (recut : Bool) : Prop :=
  ∀ (desc : Desc) (fid : String) (d : Lean.Json) (form id : String) (B : String → Prop),
    ⦃fun s => ⌜∀ s1, DecState.defs s1 = DecState.defs s → DecState.index s1 = DecState.index s →
        DecState.busy s1 = (DecState.busy s).insert fid → Kept desc B s1⌝⦄
      (composite n desc fid d form id recut : DM Defn)
    ⦃⇓? defn s => ⌜Kept desc B s ∧ defn.key = fid ∧ defn.prim = false ∧ defn.body.retType = defn.ret⌝⦄

/-- **`getDef` keeps `Kept`** when `composite` does one fuel lower. -/
theorem getDef_step (n : Nat) (recut : Bool) (hc : CompositeKept n recut) (desc : Desc) (fid : String)
    (B : String → Prop) :
    ⦃fun s => ⌜Kept desc B s⌝⦄ (getDef (n + 1) desc fid recut : DM Nat) ⦃⇓? _ s => ⌜Kept desc B s⌝⦄ := by
  have hc' := fun desc d form id => hc desc fid d form id (fun k => B k ∨ k = fid)
  unfold getDef
  mvcgen [hc', liftV_spec, throw_spec, -Spec.throw_Except]
  all_goals first
    | exact push_prim (by assumption) (by assumption) (by assumption) (by assumption) (by assumption)
        (by assumption) (by assumption) rfl rfl rfl rfl rfl
    | exact busy_pre (by assumption) (by assumption) (by assumption)
    | exact push_comp (by assumption) (by assumption) (by assumption) rfl rfl rfl

/-- **The decoder keeps `Kept`**, at every fuel: `getDef` and `staticFns` from a state that has it, and `composite` as
`CompositeKept` says. -/
theorem kept_fuel (n : Nat) (recut : Bool) :
    (∀ (desc : Desc) (fid : String) (B : String → Prop),
      ⦃fun s => ⌜Kept desc B s⌝⦄ (getDef n desc fid recut : DM Nat) ⦃⇓? _ s => ⌜Kept desc B s⌝⦄) ∧
    (∀ (desc : Desc) (v : Lean.Json) (B : String → Prop),
      ⦃fun s => ⌜Kept desc B s⌝⦄ (staticFns n desc v recut : DM Unit) ⦃⇓? _ s => ⌜Kept desc B s⌝⦄) ∧
    CompositeKept n recut := by
  induction n with
  | zero =>
    refine ⟨fun desc fid B => ?_, fun desc v B => ?_, fun desc fid d form id B => ?_⟩
    · unfold getDef
      mvcgen [throw_spec, -Spec.throw_Except]
    · unfold staticFns
      mvcgen [throw_spec, -Spec.throw_Except]
    · unfold composite
      mvcgen [throw_spec, -Spec.throw_Except]
  | succ n ih =>
    obtain ⟨hg, hs, hc⟩ := ih
    exact ⟨fun desc fid B => getDef_step n recut hc desc fid B,
      fun desc v B => staticFns_step n (Kept desc B) desc recut (fun g => hg desc g B) (fun v => hs desc v B) v,
      fun desc fid d form id B => composite_step n (Kept desc B) desc recut (kept_congr desc B) (fun g => hg desc g B)
        (fun v => hs desc v B) fid d form id⟩

/-- **`getDef` returns the place the index gives its key.** -/
theorem getDef_idx (n : Nat) (hc : CompositeKept n false) (desc : Desc) (fid : String) (B : String → Prop) :
    ⦃fun s => ⌜Kept desc B s⌝⦄ (getDef (n + 1) desc fid : DM Nat)
    ⦃⇓? r s => ⌜(DecState.index s)[fid]? = some r⌝⦄ := by
  have hc' := fun desc d form id => hc desc fid d form id (fun k => B k ∨ k = fid)
  unfold getDef
  mvcgen [hc', liftV_spec, throw_spec, -Spec.throw_Except]
  all_goals first
    | simp
    | exact busy_pre (by assumption) (by assumption) (by assumption)
    | assumption

/-! ## What `decode` returns -/

/-- `decode`'s `go` keeps `Kept`, with no name busy: the root, then the functions the program record names, each by
`getDef`. -/
theorem go_kept (desc : Desc) (rootId programFn : String) (recut : Bool) :
    ⦃fun s => ⌜Kept desc (fun _ => False) s⌝⦄
      (do
        let r ← getDef Flock.Steps.FUEL desc rootId recut
        forIn (recordFunctionIds programFn) PUnit.unit fun f __s =>
            if (Desc.ddefs desc).contains f = true then do
              discard (getDef Flock.Steps.FUEL desc f recut)
              pure (ForInStep.yield PUnit.unit)
            else pure (ForInStep.yield PUnit.unit)
        pure r : DM Nat)
    ⦃⇓? _ s => ⌜Kept desc (fun _ => False) s⌝⦄ := by
  have hg := fun f => (kept_fuel Flock.Steps.FUEL recut).1 desc f (fun _ => False)
  mvcgen [hg, discard_eq] invariants
    · ⇓? _ s => ⌜Kept desc (fun _ => False) s⌝

/-- `decodeAll`, the decoding of the primitives and the Boolean map's bodies, keeps `Kept` with no name busy. -/
theorem decodeAll_kept (desc : Desc) (fs : Array String) :
    ⦃fun s => ⌜Kept desc (fun _ => False) s⌝⦄ (decodeAll desc fs : DM Unit) ⦃⇓? _ s => ⌜Kept desc (fun _ => False) s⌝⦄ := by
  have hg := fun f => (kept_fuel Flock.Steps.FUEL false).1 desc f (fun _ => False)
  unfold decodeAll
  mvcgen [hg, discard_eq] invariants
    · ⇓? _ s => ⌜Kept desc (fun _ => False) s⌝

/-- `decodeBodies`, the decoding and checking of the Boolean map's bodies, keeps `Kept` with no name busy. -/
theorem decodeBodies_kept (desc : Desc) (rootKey : String) (pairs : Array (String × String)) :
    ⦃fun s => ⌜Kept desc (fun _ => False) s⌝⦄ (decodeBodies desc rootKey pairs : DM Unit)
    ⦃⇓? _ s => ⌜Kept desc (fun _ => False) s⌝⦄ := by
  have hg := fun f => (kept_fuel Flock.Steps.FUEL false).1 desc f (fun _ => False)
  unfold decodeBodies
  mvcgen [hg, liftV_spec, throw_spec, -Spec.throw_Except] invariants
    · ⇓? _ s => ⌜Kept desc (fun _ => False) s⌝

/-- **What `decode` returns**: the Definitions and the index of a decoder state that has `Kept`. -/
theorem decode_kept (bytes : ByteArray) (P : Prog) (h : decode bytes = .ok P) :
    ∃ desc st, P.defs = DecState.defs st ∧ P.index = DecState.index st ∧ Kept desc (fun _ => False) st := by
  obtain ⟨_, desc, _, _, _, _, _, _, st2, _, -, hgo, -, -, hall, -, hbod, hd, hi, -, -⟩ := BodiesOk.decode_walk bytes P h
  obtain ⟨recut, hgo'⟩ := decodeDefs_walk hgo
  unfold goOf at hgo'
  exact ⟨desc, st2, hd, hi, of_triple (decodeBodies_kept _ _ _) hbod
    (of_triple (decodeAll_kept _ _) hall (of_triple (go_kept _ _ _ recut) hgo' (kept_empty desc rfl rfl)))⟩

/-- **The keys of a decoded Program**: its index sends a key to the place of the Definition that has it, and each
Definition's key to its own place. -/
theorem decode_keys (bytes : ByteArray) (P : Prog) (h : decode bytes = .ok P) :
    (∀ k v, P.index[k]? = some v → v < P.defs.size ∧ (P.defs[v]!).key = k) ∧
      ∀ i < P.defs.size, P.index[(P.defs[i]!).key]? = some i := by
  obtain ⟨_, st, hd, hi, hk, -⟩ := decode_kept bytes P h
  rw [hd, hi]
  exact ⟨hk.1, hk.2.1⟩

/-- **A composite Definition of a decoded Program returns its body's return type.** -/
theorem decode_ret (bytes : ByteArray) (P : Prog) (h : decode bytes = .ok P) :
    ∀ d < P.defs.size, (P.defs[d]!).prim = false → (P.defs[d]!).body.retType = (P.defs[d]!).ret := by
  obtain ⟨_, st, hd, -, -, hr, -⟩ := decode_kept bytes P h
  rw [hd]
  exact hr

/-- **No Definition before the root of a decoded Program has the root's key.** -/
theorem root_first (bytes : ByteArray) (P : Prog) (h : decode bytes = .ok P)
    (hr : (P.defs[P.root]!).body.recut = #[] := by assumption) :
    ∀ i < P.root, (P.defs[i]!).key ≠ (P.defs[P.root]!).key := by
  intro i hi he
  have hl := (decode_root bytes P h).1
  have hk := (decode_keys bytes P h).2
  have h1 := hk i (Nat.lt_trans hi hl)
  rw [he, hk P.root hl] at h1
  exact absurd (Option.some.inj h1) (Nat.ne_of_gt hi)

end InferenceOnlyVerifier.Properties.SecurityProofs.DecodeKeys
