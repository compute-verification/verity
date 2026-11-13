module

public import Proofs.Properties.Nci.BodiesOk.Decoded
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
`.ok` result (`of_triple`). `composite_step` and `staticFns_step` hold of any property of the decoder's state that
`getDef` and `staticFns` keep one fuel lower, so another fact about the state needs no second walk. `kept_fuel` is the
induction on the fuel for `Kept`: the keys (`Keys`), the return types (`Rets`) and the names that stay busy. A name is
busy from `composite`'s mark to `getDef`'s push and a busy name has no index entry, so the push finds no earlier
Definition with its key.
-/

@[expose] public section

set_option mvcgen.warning false

namespace NciCheck.Properties.SecurityProofs.DecodeKeys
open Flock.Program
open Std.Do
open BodiesOk (DM ex_throw_bind decode_root)
open private getDef composite staticFns DecState DecState.defs DecState.index DecState.busy Desc Desc.ddefs liftV
  from Circuits.Program

/-! ## Reading the decoder's functions -/

/-- A triple of the decoder's monad, read at an `.ok` run. -/
theorem of_triple {α : Type} {x : DM α} {P : DecState → Prop} {Q : α → DecState → Prop}
    (h : ⦃fun s => ⌜P s⌝⦄ x ⦃⇓? a s => ⌜Q a s⌝⦄) {s : DecState} {a : α} {s' : DecState}
    (hx : x s = .ok (a, s')) (hp : P s) : Q a s' := by
  have h1 := h s hp
  exact Except.of_wp_eq (prog := x.run s) (x := .ok (a, s')) hx
    (fun r => match r with | .ok p => Q p.1 p.2 | .error _ => True) (fun _ => h1)

private theorem liftV_eq {α : Type} (x : Flock.V α) :
    (liftV x : DM α) = match x with
      | .ok a => pure a
      | .error e => throw e := by
  funext s
  cases x <;> rfl

/-- `liftV x` is `x`'s value or its refusal, and leaves the state alone. -/
theorem liftV_spec {α : Type} (x : Flock.V α) (Q : PostCond α (.arg DecState (.except String .pure))) :
    Triple (liftV x : DM α)
      (wp⟦(match x with
            | .ok a => pure a
            | .error e => throw e : DM α)⟧ Q) Q := by
  rw [liftV_eq]
  exact Triple.iff.mpr .rfl

/-- A refusal, in `Except`: the library's own spec leaves a universe open, which `mvcgen` refuses. -/
theorem throw_spec {α : Type} (e : String) (Q : PostCond α (.except String .pure)) :
    Triple (m := Except String) (ps := .except String .pure) (MonadExceptOf.throw e : Except String α)
      (spred(Q.2.1 e)) Q := by
  simp [Triple.iff]

private theorem discard_eq {α : Type} (x : DM α) : (discard x : DM PUnit) = (fun _ => PUnit.unit) <$> x := rfl

/-- `Array.mapM` keeps what each of its calls keeps. -/
theorem mapM_spec {α β : Type} (K : DecState → Prop) (f : α → DM β) (xs : Array α)
    (hf : ∀ x, ⦃fun s => ⌜K s⌝⦄ f x ⦃⇓? _ s => ⌜K s⌝⦄) :
    ⦃fun s => ⌜K s⌝⦄ xs.mapM f ⦃⇓? _ s => ⌜K s⌝⦄ := by
  rw [Array.mapM_eq_foldlM]
  mvcgen [hf] invariants
    · ⇓? _ s => ⌜K s⌝
  all_goals exact fun h => h

/-! ## The walk, for any property of the state -/

/-- **`staticFns` keeps what `getDef` and `staticFns` keep one fuel lower.** -/
theorem staticFns_step (n : Nat) (K : DecState → Prop)
    (hg : ∀ desc g, ⦃fun s => ⌜K s⌝⦄ (getDef n desc g : DM Nat) ⦃⇓? _ s => ⌜K s⌝⦄)
    (hs : ∀ desc v, ⦃fun s => ⌜K s⌝⦄ (staticFns n desc v : DM Unit) ⦃⇓? _ s => ⌜K s⌝⦄)
    (desc : Desc) (v : Lean.Json) :
    ⦃fun s => ⌜K s⌝⦄ (staticFns (n + 1) desc v : DM Unit) ⦃⇓? _ s => ⌜K s⌝⦄ := by
  unfold staticFns
  mvcgen [hg, hs, discard_eq, liftV_spec, throw_spec, -Spec.throw_Except] invariants
    · ⇓? _ s => ⌜K s⌝
    · ⇓? _ s => ⌜K s⌝

-- `mvcgen` follows every path of `composite`, whose body the two orders of a binding record's keys double.
set_option maxHeartbeats 2000000 in
/-- **`composite` keeps what `getDef` and `staticFns` keep one fuel lower**, for a property `K` of the state that does
not read the reinterpretations walked (`hw`): from a state that has `K` once `fid` is busy, it ends in a state with `K`,
and what it returns is a composite keyed `fid` that returns its body's return type. -/
theorem composite_step (n : Nat) (K : DecState → Prop)
    (hw : ∀ s s', DecState.defs s' = DecState.defs s → DecState.index s' = DecState.index s →
      DecState.busy s' = DecState.busy s → K s → K s')
    (hg : ∀ desc g, ⦃fun s => ⌜K s⌝⦄ (getDef n desc g : DM Nat) ⦃⇓? _ s => ⌜K s⌝⦄)
    (hs : ∀ desc v, ⦃fun s => ⌜K s⌝⦄ (staticFns n desc v : DM Unit) ⦃⇓? _ s => ⌜K s⌝⦄)
    (desc : Desc) (fid : String) (d : Lean.Json) (form id : String) :
    ⦃fun s => ⌜∀ s1, DecState.defs s1 = DecState.defs s → DecState.index s1 = DecState.index s →
        DecState.busy s1 = (DecState.busy s).insert fid → K s1⌝⦄
      (composite (n + 1) desc fid d form id : DM Defn)
    ⦃⇓? defn s => ⌜K s ∧ defn.key = fid ∧ defn.prim = false ∧ defn.body.retType = defn.ret⌝⦄ := by
  have hm := fun (f : Lean.Json → DM (Option Nat)) (xs : Array Lean.Json) hf => mapM_spec K f xs hf
  unfold composite
  mvcgen [hg, hs, hm, liftV_spec, throw_spec, -Spec.throw_Except] invariants
    · ⇓? _ s => ⌜∀ s1, DecState.defs s1 = DecState.defs s → DecState.index s1 = DecState.index s →
        DecState.busy s1 = (DecState.busy s).insert fid → K s1⌝
    · ⇓? _ s => ⌜K s⌝
    · ⇓? _ s => ⌜K s⌝
    · ⇓? _ s => ⌜∀ s1, DecState.defs s1 = DecState.defs s → DecState.index s1 = DecState.index s →
        DecState.busy s1 = (DecState.busy s).insert fid → K s1⌝
    · ⇓? _ s => ⌜K s⌝
    · ⇓? _ s => ⌜K s⌝
  all_goals first
    | (refine ‹∀ s1, DecState.defs s1 = DecState.defs _ → DecState.index s1 = DecState.index _ →
        DecState.busy s1 = (DecState.busy _).insert fid → K s1› _ ?_ ?_ ?_ <;> rfl)
    | (refine ⟨hw _ _ ?_ ?_ ?_ (by assumption), trivial, trivial, rfl⟩ <;> rfl)

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

/-- What the decoder keeps while the names of `B` are being decoded: `Keys`, `Rets`, and each name of `B` busy. -/
def Kept (B : String → Prop) (s : DecState) : Prop :=
  Keys s ∧ Rets s ∧ ∀ k, B k → (DecState.busy s).contains k = true

private theorem kept_congr (B : String → Prop) (s s' : DecState) (hd : DecState.defs s' = DecState.defs s)
    (hi : DecState.index s' = DecState.index s) (hb : DecState.busy s' = DecState.busy s) (h : Kept B s) :
    Kept B s' := by
  unfold Kept Keys Rets
  rw [hd, hi, hb]
  exact h

/-- Marking a name without an index entry busy keeps `Kept`, with that name among the busy ones. -/
private theorem kept_busy {B : String → Prop} {s s1 : DecState} {fid : String} (h : Kept B s)
    (hnone : (DecState.index s)[fid]? = none)
    (hd : DecState.defs s1 = DecState.defs s) (hi : DecState.index s1 = DecState.index s)
    (hb : DecState.busy s1 = (DecState.busy s).insert fid) : Kept (fun k => B k ∨ k = fid) s1 := by
  obtain ⟨⟨k1, k2, k3⟩, hr, hB⟩ := h
  refine ⟨⟨?_, ?_, ?_⟩, ?_, ?_⟩
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

private theorem getElem!_push_lt {xs : Array Defn} {d : Defn} {i : Nat} (h : i < xs.size) :
    (xs.push d)[i]! = xs[i]! := by
  simp [Array.getElem!_eq_getD, Array.getD_eq_getD_getElem?, Array.getElem?_push, Nat.ne_of_lt h]

private theorem getElem!_push_size (xs : Array Defn) (d : Defn) : (xs.push d)[xs.size]! = d := by
  simp

/-- `getDef`'s push of a Definition keyed `fid`, a name without an index entry, keeps `Keys` and `Rets`, and the names
of `B` busy when none of them is `fid`. -/
private theorem kept_push {B : String → Prop} {s s' : DecState} {defn : Defn} {fid : String} (hk : Keys s)
    (hr : Rets s) (hnone : (DecState.index s)[fid]? = none)
    (hsub : ∀ k, B k → (DecState.busy s).contains k = true ∧ k ≠ fid) (hkey : defn.key = fid)
    (hret : defn.prim = false → defn.body.retType = defn.ret)
    (hd : DecState.defs s' = (DecState.defs s).push defn)
    (hi : DecState.index s' = (DecState.index s).insert fid (DecState.defs s).size)
    (hb : DecState.busy s' = (DecState.busy s).erase fid) : Kept B s' := by
  obtain ⟨k1, k2, k3⟩ := hk
  refine ⟨⟨?_, ?_, ?_⟩, ?_, ?_⟩
  · intro k v hv
    rw [hi, Std.HashMap.getElem?_insert] at hv
    rw [hd, Array.size_push]
    by_cases hf : (fid == k) = true
    · simp only [hf, ↓reduceIte] at hv
      have hv' : (DecState.defs s).size = v := Option.some.inj hv
      subst hv'
      exact ⟨Nat.lt_succ_self _, by rw [getElem!_push_size, hkey, eq_of_beq hf]⟩
    · simp only [hf, Bool.false_eq_true, ↓reduceIte] at hv
      obtain ⟨h1, h2⟩ := k1 k v hv
      exact ⟨Nat.lt_succ_of_lt h1, by rw [getElem!_push_lt h1, h2]⟩
  · intro i hlt
    rw [hd, Array.size_push] at hlt
    rw [hi, hd, Std.HashMap.getElem?_insert]
    by_cases he : i = (DecState.defs s).size
    · subst he
      simp [hkey]
    · have hlt' : i < (DecState.defs s).size := Nat.lt_of_le_of_ne (Nat.le_of_lt_succ hlt) he
      rw [getElem!_push_lt hlt']
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
      rw [getElem!_push_size]
      exact hret
    · have hlt' : i < (DecState.defs s).size := Nat.lt_of_le_of_ne (Nat.le_of_lt_succ hlt) he
      rw [getElem!_push_lt hlt']
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
private theorem push_prim {B : String → Prop} {s s' : DecState} {fid : String} {o : Option Nat} {defn : Defn}
    (h : Kept B s) (ho : (DecState.index s)[fid]? = o) (hno : ∀ i, o = some i → False)
    (hb : ¬ (DecState.busy s).contains fid = true) (hkey : defn.key = fid) (hprim : defn.prim = true)
    (hd : DecState.defs s' = (DecState.defs s).push defn)
    (hi : DecState.index s' = (DecState.index s).insert fid (DecState.defs s).size)
    (hb' : DecState.busy s' = (DecState.busy s).erase fid) : Kept B s' :=
  kept_push h.1 h.2.1 (ho.trans (none_of hno)) (fun k hk => ⟨h.2.2 k hk, fun e => hb (e ▸ h.2.2 k hk)⟩) hkey
    (fun hp => absurd (hprim.symm.trans hp) (by decide)) hd hi hb'

/-- `composite`'s mark, in the words of `getDef`'s own conditions. -/
private theorem busy_pre {B : String → Prop} {s : DecState} {fid : String} {o : Option Nat}
    (h : Kept B s) (ho : (DecState.index s)[fid]? = o) (hno : ∀ i, o = some i → False) (s1 : DecState)
    (hd : DecState.defs s1 = DecState.defs s) (hi : DecState.index s1 = DecState.index s)
    (hb : DecState.busy s1 = (DecState.busy s).insert fid) : Kept (fun k => B k ∨ k = fid) s1 :=
  kept_busy h (ho.trans (none_of hno)) hd hi hb

/-- `getDef`'s push of what `composite` returned: `fid` is still busy there, so it has no index entry. -/
private theorem push_comp {B : String → Prop} {s0 s s' : DecState} {fid : String} {defn : Defn}
    (h0 : Kept B s0) (hb : ¬ (DecState.busy s0).contains fid = true)
    (h : Kept (fun k => B k ∨ k = fid) s ∧ defn.key = fid ∧ defn.prim = false ∧ defn.body.retType = defn.ret)
    (hd : DecState.defs s' = (DecState.defs s).push defn)
    (hi : DecState.index s' = (DecState.index s).insert fid (DecState.defs s).size)
    (hb' : DecState.busy s' = (DecState.busy s).erase fid) : Kept B s' :=
  kept_push h.1.1 h.1.2.1 (h.1.1.2.2 fid (h.1.2.2 fid (Or.inr rfl)))
    (fun k hk => ⟨h.1.2.2 k (Or.inl hk), fun e => hb (e ▸ h0.2.2 k hk)⟩) h.2.1 (fun _ => h.2.2.2) hd hi hb'

/-- `composite` at fuel `n` keeps `Kept`: from a state that has it once `fid` is busy, it ends in a state that has it,
and what it returns is a composite keyed `fid` that returns its body's return type. -/
def CompositeKept (n : Nat) : Prop :=
  ∀ (desc : Desc) (fid : String) (d : Lean.Json) (form id : String) (B : String → Prop),
    ⦃fun s => ⌜∀ s1, DecState.defs s1 = DecState.defs s → DecState.index s1 = DecState.index s →
        DecState.busy s1 = (DecState.busy s).insert fid → Kept B s1⌝⦄
      (composite n desc fid d form id : DM Defn)
    ⦃⇓? defn s => ⌜Kept B s ∧ defn.key = fid ∧ defn.prim = false ∧ defn.body.retType = defn.ret⌝⦄

/-- **`getDef` keeps `Kept`** when `composite` does one fuel lower. -/
theorem getDef_step (n : Nat) (hc : CompositeKept n) (desc : Desc) (fid : String) (B : String → Prop) :
    ⦃fun s => ⌜Kept B s⌝⦄ (getDef (n + 1) desc fid : DM Nat) ⦃⇓? _ s => ⌜Kept B s⌝⦄ := by
  have hc' := fun desc d form id => hc desc fid d form id (fun k => B k ∨ k = fid)
  unfold getDef
  mvcgen [hc', liftV_spec, throw_spec, -Spec.throw_Except]
  all_goals first
    | exact push_prim (by assumption) (by assumption) (by assumption) (by assumption) rfl rfl rfl rfl rfl
    | exact busy_pre (by assumption) (by assumption) (by assumption)
    | exact push_comp (by assumption) (by assumption) (by assumption) rfl rfl rfl

/-- **The decoder keeps `Kept`**, at every fuel: `getDef` and `staticFns` from a state that has it, and `composite` as
`CompositeKept` says. -/
theorem kept_fuel (n : Nat) :
    (∀ (desc : Desc) (fid : String) (B : String → Prop),
      ⦃fun s => ⌜Kept B s⌝⦄ (getDef n desc fid : DM Nat) ⦃⇓? _ s => ⌜Kept B s⌝⦄) ∧
    (∀ (desc : Desc) (v : Lean.Json) (B : String → Prop),
      ⦃fun s => ⌜Kept B s⌝⦄ (staticFns n desc v : DM Unit) ⦃⇓? _ s => ⌜Kept B s⌝⦄) ∧
    CompositeKept n := by
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
    exact ⟨fun desc fid B => getDef_step n hc desc fid B,
      fun desc v B => staticFns_step n (Kept B) (fun desc g => hg desc g B) (fun desc v => hs desc v B) desc v,
      fun desc fid d form id B => composite_step n (Kept B) (kept_congr B) (fun desc g => hg desc g B)
        (fun desc v => hs desc v B) desc fid d form id⟩

/-! ## What `decode` returns -/

/-- `decode`'s `go` keeps `Kept`, with no name busy: the root, then the functions the program record names, each by
`getDef`. -/
theorem go_kept (desc : Desc) (rootId programFn : String) :
    ⦃fun s => ⌜Kept (fun _ => False) s⌝⦄
      (do
        let r ← getDef Flock.Steps.FUEL desc rootId
        forIn (recordFunctionIds programFn) PUnit.unit fun f __s =>
            if (Desc.ddefs desc).contains f = true then do
              discard (getDef Flock.Steps.FUEL desc f)
              pure (ForInStep.yield PUnit.unit)
            else pure (ForInStep.yield PUnit.unit)
        pure r : DM Nat)
    ⦃⇓? _ s => ⌜Kept (fun _ => False) s⌝⦄ := by
  have hg := fun f => (kept_fuel Flock.Steps.FUEL).1 desc f (fun _ => False)
  mvcgen [hg, discard_eq] invariants
    · ⇓? _ s => ⌜Kept (fun _ => False) s⌝

/-- **What `decode` returns**: the Definitions and the index of a decoder state that has `Kept`. -/
theorem decode_kept (bytes : ByteArray) (P : Prog) (h : decode bytes = .ok P) :
    ∃ st, P.defs = DecState.defs st ∧ P.index = DecState.index st ∧ Kept (fun _ => False) st := by
  unfold decode at h
  extract_lets k1 at h
  split at h
  rotate_left; · exact absurd h (by intro h; injection h)
  split at h
  rotate_left; · exact (ex_throw_bind h).elim
  obtain ⟨j, h1, h⟩ := Walk.except_bind_ok h
  injection h1 with h1; subst h1
  simp (config := { zeta := false }) only [k1] at h
  split at h
  rotate_left; · exact absurd h (by intro h; injection h)
  extract_lets bare jpS at h
  split at h
  rotate_left; · exact (ex_throw_bind h).elim
  split at h
  · exact (ex_throw_bind h).elim
  simp (config := { zeta := false }) only [jpS] at h
  split at h
  rotate_left; · exact absurd h (by intro h; injection h)
  obtain ⟨u, h1, h⟩ := Walk.except_bind_ok h
  clear h1
  extract_lets freeDefs free jpD at h
  split at h
  · exact (ex_throw_bind h).elim
  simp (config := { zeta := false }) only [jpD] at h
  split at h
  rotate_left; · exact absurd h (by intro h; injection h)
  split at h
  rotate_left; · exact absurd h (by intro h; injection h)
  obtain ⟨types, h1, h⟩ := Walk.except_bind_ok h
  clear h1
  extract_lets desc at h
  split at h
  rotate_left; · exact absurd h (by intro h; injection h)
  split at h
  rotate_left; · exact absurd h (by intro h; injection h)
  extract_lets go at h
  obtain ⟨⟨root, st⟩, hgo, h⟩ := Walk.except_bind_ok h
  extract_lets rd jpR at h
  split at h
  · exact (ex_throw_bind h).elim
  simp (config := { zeta := false }) only [jpR] at h
  obtain ⟨u, -, h⟩ := Walk.except_bind_ok h
  obtain ⟨c, -, h⟩ := Walk.except_bind_ok h
  injection h with h
  subst h
  refine ⟨st, rfl, rfl, of_triple (go_kept _ _ _) hgo ⟨⟨?_, ?_, ?_⟩, ?_, ?_⟩⟩
  · intro k v h
    exact absurd h (by simp)
  · intro i h
    exact absurd h (Nat.not_lt_zero i)
  · intro k h
    simp
  · intro i h
    exact absurd h (Nat.not_lt_zero i)
  · intro k h
    exact h.elim

/-- **The keys of a decoded Program**: its index sends a key to the place of the Definition that has it, and each
Definition's key to its own place. -/
theorem decode_keys (bytes : ByteArray) (P : Prog) (h : decode bytes = .ok P) :
    (∀ k v, P.index[k]? = some v → v < P.defs.size ∧ (P.defs[v]!).key = k) ∧
      ∀ i < P.defs.size, P.index[(P.defs[i]!).key]? = some i := by
  obtain ⟨st, hd, hi, hk, -, -⟩ := decode_kept bytes P h
  rw [hd, hi]
  exact ⟨hk.1, hk.2.1⟩

/-- **A composite Definition of a decoded Program returns its body's return type.** -/
theorem decode_ret (bytes : ByteArray) (P : Prog) (h : decode bytes = .ok P) :
    ∀ d < P.defs.size, (P.defs[d]!).prim = false → (P.defs[d]!).body.retType = (P.defs[d]!).ret := by
  obtain ⟨st, hd, -, -, hr, -⟩ := decode_kept bytes P h
  rw [hd]
  exact hr

/-- **No Definition before the root of a decoded Program has the root's key.** -/
theorem root_first (bytes : ByteArray) (P : Prog) (h : decode bytes = .ok P) :
    ∀ i < P.root, (P.defs[i]!).key ≠ (P.defs[P.root]!).key := by
  intro i hi he
  have hr := (decode_root bytes P h).1
  have hk := (decode_keys bytes P h).2
  have h1 := hk i (Nat.lt_trans hi hr)
  rw [he, hk P.root hr] at h1
  exact absurd (Option.some.inj h1) (Nat.ne_of_gt hi)

end NciCheck.Properties.SecurityProofs.DecodeKeys
