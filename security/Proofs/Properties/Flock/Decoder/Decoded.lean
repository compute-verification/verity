module

public import Proofs.Properties.Flock.Decoder.Composite
public import Proofs.Properties.Flock.Decoder.Decode
public import Std.Tactic.Do
import Batteries.Tactic.OpenPrivate
import Mathlib.Analysis.SpecialFunctions.Pow.NNReal
import Mathlib.Data.NNRat.Floor
import Mathlib.Tactic.ENatToNat
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.Measurability.Init
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Std.Data.HashMap.Lemmas

/-!
# What `decode` returns

`decode_walk` walks `decode` once: on an accepted descriptor it names `decodeDefs`' walk, the root's check, `decodeAll` of
the primitive entries, `booleanKeys`, `decodeBodies`, and the Program's fields from their last state; every fact about
what `decode` returns reads it. Where the root reads no input node at another width (`Body.recut`), `decode_steps` names
the plain walk `go` from the empty state: the later passes only push (`Framed`), so the root's recut is the walk's. `decode_spec`: a Program `Flock.Program.decode` returns holds
the Definitions of a decoder state that keeps `Inv` (`specs`, from the empty state through `go`, `decodeAll` and
`decodeBodies`), and its root is a composite (`decode`'s own check). `decode_root`: the root is one of them
(`getDef`'s index), a composite without parameters (`decode`'s check, `CalleeOk`).
-/

@[expose] public section

set_option mvcgen.warning false

namespace InferenceOnlyVerifier.Properties.SecurityProofs.BodiesOk
open Flock.Program
open private getDef composite staticFns goOf decodeDefs decodeDefs_plain DecState DecState.defs DecState.index DecState.busy
  Desc Desc.types Desc.callees Desc.ddefs liftV at_ paramsOf buildNode natJ? decodeAll decodeBodies primEntries
  from Circuits.Descriptor

theorem ex_throw_bind {α β : Type} {e : String} {f : α → Flock.V β} {b : β}
    (h : (throw e >>= f : Flock.V β) = .ok b) : False := by
  obtain ⟨a, h1, -⟩ := Walk.except_bind_ok h
  exact absurd h1 (by intro h; injection h)

/-- `decode`'s `go`: the root, then the functions the program record names, each by `getDef`; the root is one of the
Definitions. -/
theorem go_inv (desc : Desc) (rootId programFn : String) (root : ℕ) (s0 st : DecState)
    (h : StateT.run (do
          let r ← getDef Flock.Steps.FUEL desc rootId
          forIn (recordFunctionIds programFn) PUnit.unit fun f __s =>
              if (Desc.ddefs desc).contains f = true then do
                discard (getDef Flock.Steps.FUEL desc f)
                pure (ForInStep.yield PUnit.unit)
              else pure (ForInStep.yield PUnit.unit)
          pure r : DM ℕ) s0 = .ok (root, st)) (hs0 : Inv s0) : Inv st ∧ root < (DecState.defs st).size := by
  obtain ⟨r, s1, h1, h⟩ := st_bind h
  obtain ⟨hs1, -, hr1⟩ := (specs _).1 _ _ _ _ _ h1 hs0
  obtain ⟨u, s2, h2, h⟩ := st_bind h
  obtain ⟨hr, rfl⟩ := st_pure h
  try rw [← Array.forIn_toList] at h2
  have hg := forIn_grows _ _ (fun x b s t s' hf => ?_) _ _ _ _ h2 hs1
  · exact ⟨hg.1, by rw [← hr]; exact Nat.lt_of_lt_of_le hr1 hg.2.1⟩
  split at hf
  · obtain ⟨u, s3, h3, h4⟩ := st_bind hf
    obtain ⟨-, rfl⟩ := st_pure h4
    obtain ⟨a, ha⟩ := st_discard h3
    intro hs
    obtain ⟨i1, p1, -⟩ := (specs _).1 _ _ _ _ _ ha hs
    exact ⟨i1, p1⟩
  · obtain ⟨-, rfl⟩ := st_pure hf
    exact grows_refl _

end InferenceOnlyVerifier.Properties.SecurityProofs.BodiesOk

namespace InferenceOnlyVerifier.Properties.SecurityProofs.DecodeKeys
open Flock.Program
open Std.Do
open BodiesOk (DM)
open private getDef composite staticFns DecState DecState.defs DecState.index DecState.busy Desc Desc.ddefs liftV
  from Circuits.Descriptor

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

theorem discard_eq {α : Type} (x : DM α) : (discard x : DM PUnit) = (fun _ => PUnit.unit) <$> x := rfl

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
theorem staticFns_step (n : Nat) (K : DecState → Prop) (desc : Desc) (recut : Bool)
    (hg : ∀ g, ⦃fun s => ⌜K s⌝⦄ (getDef n desc g recut : DM Nat) ⦃⇓? _ s => ⌜K s⌝⦄)
    (hs : ∀ v, ⦃fun s => ⌜K s⌝⦄ (staticFns n desc v recut : DM Unit) ⦃⇓? _ s => ⌜K s⌝⦄)
    (v : Lean.Json) :
    ⦃fun s => ⌜K s⌝⦄ (staticFns (n + 1) desc v recut : DM Unit) ⦃⇓? _ s => ⌜K s⌝⦄ := by
  unfold staticFns
  mvcgen [hg, hs, discard_eq, liftV_spec, throw_spec, -Spec.throw_Except] invariants
    · ⇓? _ s => ⌜K s⌝
    · ⇓? _ s => ⌜K s⌝

-- `mvcgen` follows every path of `composite`, whose body the two orders of a binding record's keys double.
set_option maxHeartbeats 2000000 in
/-- **`composite` keeps what `getDef` and `staticFns` keep one fuel lower**, for a property `K` of the state that does
not read the reinterpretations walked (`hw`): from a state that has `K` once `fid` is busy, it ends in a state with `K`,
and what it returns is a composite keyed `fid` that returns its body's return type. -/
theorem composite_step (n : Nat) (K : DecState → Prop) (desc : Desc) (recut : Bool)
    (hw : ∀ s s', DecState.defs s' = DecState.defs s → DecState.index s' = DecState.index s →
      DecState.busy s' = DecState.busy s → K s → K s')
    (hg : ∀ g, ⦃fun s => ⌜K s⌝⦄ (getDef n desc g recut : DM Nat) ⦃⇓? _ s => ⌜K s⌝⦄)
    (hs : ∀ v, ⦃fun s => ⌜K s⌝⦄ (staticFns n desc v recut : DM Unit) ⦃⇓? _ s => ⌜K s⌝⦄)
    (fid : String) (d : Lean.Json) (form id : String) :
    ⦃fun s => ⌜∀ s1, DecState.defs s1 = DecState.defs s → DecState.index s1 = DecState.index s →
        DecState.busy s1 = (DecState.busy s).insert fid → K s1⌝⦄
      (composite (n + 1) desc fid d form id recut : DM Defn)
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

end InferenceOnlyVerifier.Properties.SecurityProofs.DecodeKeys

namespace InferenceOnlyVerifier.Properties.SecurityProofs.BodiesOk
open Flock.Program
open Std.Do
open DecodeKeys (of_triple liftV_spec throw_spec discard_eq staticFns_step composite_step)
open private getDef composite staticFns goOf decodeDefs decodeDefs_plain DecState DecState.defs DecState.index DecState.busy
  Desc Desc.types Desc.callees Desc.ddefs liftV at_ paramsOf buildNode natJ? decodeAll decodeBodies primEntries
  from Circuits.Descriptor

/-- `decodeAll`: each primitive entry by `getDef`. -/
theorem decodeAll_grows (desc : Desc) (fs : Array String) (s s' : DecState) (u : Unit)
    (h : StateT.run (decodeAll desc fs : DM Unit) s = .ok (u, s')) : Grows s s' := by
  unfold decodeAll at h
  obtain ⟨r, s1, h1, h⟩ := st_bind h
  obtain ⟨-, rfl⟩ := st_pure h
  try rw [← Array.forIn_toList] at h1
  refine forIn_grows _ _ (fun x b s t s' hf => ?_) _ _ _ _ h1
  split at hf
  · obtain ⟨u, s3, h3, h4⟩ := st_bind hf
    obtain ⟨-, rfl⟩ := st_pure h4
    obtain ⟨a, ha⟩ := st_discard h3
    intro hs
    obtain ⟨i1, p1, -⟩ := (specs _).1 _ _ _ _ _ ha hs
    exact ⟨i1, p1⟩
  · obtain ⟨-, rfl⟩ := st_pure hf
    exact grows_refl _

/-- `decodeBodies`: each body by `getDef`, then `checkBody`, which leaves the state alone. -/
theorem decodeBodies_grows (desc : Desc) (rootKey : String) (pairs : Array (String × String)) (s s' : DecState)
    (u : Unit) (h : StateT.run (decodeBodies desc rootKey pairs : DM Unit) s = .ok (u, s')) : Grows s s' := by
  unfold decodeBodies at h
  obtain ⟨r, s1, h1, h⟩ := st_bind h
  obtain ⟨-, rfl⟩ := st_pure h
  try rw [← Array.forIn_toList] at h1
  refine forIn_grows _ _ (fun x b s t s' hf => ?_) _ _ _ _ h1
  obtain ⟨w, bb⟩ := x
  dsimp only at hf
  split at hf
  · exact absurd hf (by intro h; injection h)
  obtain ⟨st0, s2, hg, hf⟩ := st_bind hf
  obtain ⟨rfl, rfl⟩ := st_get hg
  obtain ⟨bi, s3, hd, hf⟩ := st_bind hf
  obtain ⟨st2, s4, hg, hf⟩ := st_bind hf
  obtain ⟨rfl, rfl⟩ := st_get hg
  obtain ⟨u, s5, hl, hf⟩ := st_bind hf
  obtain ⟨-, rfl⟩ := st_liftV hl
  obtain ⟨-, rfl⟩ := st_pure hf
  intro hs
  obtain ⟨i1, p1, -⟩ := (specs _).1 _ _ _ _ _ hd hs
  exact ⟨i1, p1⟩

/-! ## What every walk keeps -/

/-- The Definitions `D0`, a prefix of the decoder state's: what `getDef` keeps on either walk, with no other fact about the
state. -/
def Framed (D0 : Array Defn) (s : DecState) : Prop := Pre D0 (DecState.defs s)

private theorem framed_defs {D0 : Array Defn} {s s1 : DecState} (h : Framed D0 s)
    (hd : DecState.defs s1 = DecState.defs s) : Framed D0 s1 := by
  unfold Framed at *
  rw [hd]
  exact h

private theorem framed_push {D0 : Array Defn} {s s' : DecState} {defn : Defn} (h : Framed D0 s)
    (hd : DecState.defs s' = (DecState.defs s).push defn) : Framed D0 s' := by
  unfold Framed Pre at *
  rw [hd]
  refine ⟨Nat.le_trans h.1 (by rw [Array.size_push]; exact Nat.le_succ _), fun i hi => ?_⟩
  rw [getElem!_push_lt _ _ (Nat.lt_of_lt_of_le hi h.1)]
  exact h.2 i hi

private theorem framed_push_comp {D0 : Array Defn} {s s' : DecState} {defn : Defn} {X : Prop} (h : Framed D0 s ∧ X)
    (hd : DecState.defs s' = (DecState.defs s).push defn) : Framed D0 s' :=
  framed_push h.1 hd

/-- `composite` at fuel `n` keeps `Framed D0`, from a state that has it once `fid` is busy. -/
def CompositeFramed (D0 : Array Defn) (n : Nat) (recut : Bool) : Prop :=
  ∀ (desc : Desc) (fid : String) (d : Lean.Json) (form id : String),
    ⦃fun s => ⌜∀ s1, DecState.defs s1 = DecState.defs s → DecState.index s1 = DecState.index s →
        DecState.busy s1 = (DecState.busy s).insert fid → Framed D0 s1⌝⦄
      (composite n desc fid d form id recut : DM Defn)
    ⦃⇓? defn s => ⌜Framed D0 s ∧ defn.key = fid ∧ defn.prim = false ∧ defn.body.retType = defn.ret⌝⦄

/-- **The decoder keeps `Framed D0`** on either walk, at every fuel: `getDef` and `staticFns`, and `composite` as
`CompositeFramed` says. -/
theorem framed_fuel (D0 : Array Defn) (n : Nat) (recut : Bool) :
    (∀ (desc : Desc) (fid : String),
      ⦃fun s => ⌜Framed D0 s⌝⦄ (getDef n desc fid recut : DM Nat) ⦃⇓? _ s => ⌜Framed D0 s⌝⦄) ∧
    (∀ (desc : Desc) (v : Lean.Json),
      ⦃fun s => ⌜Framed D0 s⌝⦄ (staticFns n desc v recut : DM Unit) ⦃⇓? _ s => ⌜Framed D0 s⌝⦄) ∧
    CompositeFramed D0 n recut := by
  induction n with
  | zero =>
    refine ⟨fun desc fid => ?_, fun desc v => ?_, fun desc fid d form id => ?_⟩
    · unfold getDef
      mvcgen [throw_spec, -Spec.throw_Except]
    · unfold staticFns
      mvcgen [throw_spec, -Spec.throw_Except]
    · unfold composite
      mvcgen [throw_spec, -Spec.throw_Except]
  | succ n ih =>
    obtain ⟨hg, hs, hc⟩ := ih
    refine ⟨fun desc fid => ?_, fun desc v => staticFns_step n (Framed D0) desc recut (hg desc) (hs desc) v,
      fun desc fid d form id => composite_step n (Framed D0) desc recut (fun _ _ hd _ _ h => framed_defs h hd) (hg desc)
        (hs desc) fid d form id⟩
    have hc' := fun desc d form id => hc desc fid d form id
    unfold getDef
    mvcgen [hc', liftV_spec, throw_spec, -Spec.throw_Except]
    all_goals first
      | exact framed_push (by assumption) rfl
      | exact fun _ hd _ _ => framed_defs (by assumption) hd
      | exact framed_push_comp (by assumption) rfl
      | assumption

/-- `decodeAll` keeps `Framed D0`. -/
theorem decodeAll_framed (D0 : Array Defn) (desc : Desc) (fs : Array String) :
    ⦃fun s => ⌜Framed D0 s⌝⦄ (decodeAll desc fs : DM Unit) ⦃⇓? _ s => ⌜Framed D0 s⌝⦄ := by
  have hg := fun f => (framed_fuel D0 Flock.Steps.FUEL false).1 desc f
  unfold decodeAll
  mvcgen [hg, discard_eq] invariants
    · ⇓? _ s => ⌜Framed D0 s⌝

/-- `decodeBodies` keeps `Framed D0`. -/
theorem decodeBodies_framed (D0 : Array Defn) (desc : Desc) (rootKey : String) (pairs : Array (String × String)) :
    ⦃fun s => ⌜Framed D0 s⌝⦄ (decodeBodies desc rootKey pairs : DM Unit) ⦃⇓? _ s => ⌜Framed D0 s⌝⦄ := by
  have hg := fun f => (framed_fuel D0 Flock.Steps.FUEL false).1 desc f
  unfold decodeBodies
  mvcgen [hg, liftV_spec, throw_spec, -Spec.throw_Except] invariants
    · ⇓? _ s => ⌜Framed D0 s⌝
  all_goals try assumption

/-- **`decode`'s steps** on a descriptor it accepts: `decodeDefs`, the root's check, `decodeAll` of the primitive
entries, `booleanKeys` and `decodeBodies`, and the Program built from their last state. -/
theorem decode_walk (bytes : ByteArray) (P : Prog) (h : decode bytes = .ok P) :
    ∃ (kv : Array (String × Lean.Json)) (desc : Desc) (rootId programFn : String) (bm : Lean.Json) (root : ℕ)
      (st st1 st2 : DecState) (pairs : Array (String × String)),
      Desc.ddefs desc = kv.foldl (fun m x => m.insert x.1 x.2) ∅ ∧
      decodeDefs desc rootId programFn = .ok (root, st) ∧
      ((DecState.defs st)[root]!).prim = false ∧ ((DecState.defs st)[root]!).params.size = 0 ∧
      StateT.run (decodeAll desc (primEntries kv) : DM Unit) st = .ok ((), st1) ∧
      booleanKeys (DecState.defs st1) (DecState.index st1) (primEntries kv) bm = .ok pairs ∧
      StateT.run (decodeBodies desc rootId pairs : DM Unit) st1 = .ok ((), st2) ∧
      P.defs = DecState.defs st2 ∧ P.index = DecState.index st2 ∧ P.root = root ∧
      P.boolean = pairs.foldl (fun m p => m.insert p.1 p.2) ∅ := by
  unfold decode at h
  extract_lets k1 at h
  split at h
  swap; · exact absurd h (by intro h; injection h)
  split at h
  swap; · exact (ex_throw_bind h).elim
  obtain ⟨j, h1, h⟩ := Walk.except_bind_ok h
  injection h1 with h1; subst h1
  simp (config := { zeta := false }) only [k1] at h
  split at h
  swap; · exact absurd h (by intro h; injection h)
  extract_lets bm bareJ jpS at h
  split at h
  swap; · exact (ex_throw_bind h).elim
  split at h
  · exact (ex_throw_bind h).elim
  simp (config := { zeta := false }) only [jpS] at h
  split at h
  swap; · exact absurd h (by intro h; injection h)
  obtain ⟨u, h1, h⟩ := Walk.except_bind_ok h
  clear h1
  extract_lets freeDefs free prims jpD at h
  split at h
  · exact (ex_throw_bind h).elim
  simp (config := { zeta := false }) only [jpD] at h
  split at h
  swap; · exact absurd h (by intro h; injection h)
  split at h
  swap; · exact absurd h (by intro h; injection h)
  obtain ⟨types, h1, h⟩ := Walk.except_bind_ok h
  clear h1
  extract_lets desc at h
  split at h
  swap; · exact absurd h (by intro h; injection h)
  split at h
  swap; · exact absurd h (by intro h; injection h)
  obtain ⟨⟨root, st⟩, hgo, h⟩ := Walk.except_bind_ok h
  extract_lets rd jpR at h
  split at h
  · exact (ex_throw_bind h).elim
  rename_i hrd
  simp (config := { zeta := false }) only [jpR] at h
  obtain ⟨u, -, h⟩ := Walk.except_bind_ok h
  obtain ⟨⟨⟨⟩, st1⟩, hall, h⟩ := Walk.except_bind_ok h
  obtain ⟨pairs, hbk, h⟩ := Walk.except_bind_ok h
  obtain ⟨⟨⟨⟩, st2⟩, hbod, h⟩ := Walk.except_bind_ok h
  try extract_lets boolean at h
  obtain ⟨c, -, h⟩ := Walk.except_bind_ok h
  injection h with h
  subst h
  simp only [rd, Bool.or_eq_true, not_or] at hrd
  refine ⟨_, _, _, _, _, root, st, st1, st2, pairs, ?_, hgo, by simpa using hrd.1, by simpa using hrd.2, hall, hbk,
    hbod, rfl, rfl, rfl, rfl⟩
  rfl

/-- **`decode`'s steps** on a descriptor it accepts whose root reads no input node at another width: `decode_walk`'s,
with `decodeDefs` the plain walk: `go` from the empty state, the root's check, `decodeAll` of the
primitive entries, `booleanKeys` and `decodeBodies`, and the Program built from their last state. -/
theorem decode_steps (bytes : ByteArray) (P : Prog) (h : decode bytes = .ok P)
    (hr : (P.defs[P.root]!).body.recut = #[] := by assumption) :
    ∃ (kv : Array (String × Lean.Json)) (desc : Desc) (rootId programFn : String) (bm : Lean.Json) (root : ℕ)
      (s0 st st1 st2 : DecState) (pairs : Array (String × String)),
      Desc.ddefs desc = kv.foldl (fun m x => m.insert x.1 x.2) ∅ ∧
      StateT.run (do
          let r ← getDef Flock.Steps.FUEL desc rootId
          forIn (recordFunctionIds programFn) PUnit.unit fun f __s =>
              if (Desc.ddefs desc).contains f = true then do
                discard (getDef Flock.Steps.FUEL desc f)
                pure (ForInStep.yield PUnit.unit)
              else pure (ForInStep.yield PUnit.unit)
          pure r : DM ℕ) s0 = .ok (root, st) ∧
      DecState.defs s0 = #[] ∧ DecState.index s0 = ∅ ∧ DecState.busy s0 = ∅ ∧
      ((DecState.defs st)[root]!).prim = false ∧ ((DecState.defs st)[root]!).params.size = 0 ∧
      StateT.run (decodeAll desc (primEntries kv) : DM Unit) st = .ok ((), st1) ∧
      booleanKeys (DecState.defs st1) (DecState.index st1) (primEntries kv) bm = .ok pairs ∧
      StateT.run (decodeBodies desc rootId pairs : DM Unit) st1 = .ok ((), st2) ∧
      P.defs = DecState.defs st2 ∧ P.index = DecState.index st2 ∧ P.root = root ∧
      P.boolean = pairs.foldl (fun m p => m.insert p.1 p.2) ∅ := by
  obtain ⟨kv, desc, rootId, programFn, bm, root, st, st1, st2, pairs, hkv, hgo, hp, hpar, hall, hbk, hbod, hd, hi, hroot,
    hb⟩ := decode_walk bytes P h
  have hpre : Pre (DecState.defs st) (DecState.defs st2) :=
    of_triple (decodeBodies_framed (DecState.defs st) _ _ _) hbod
      (of_triple (decodeAll_framed (DecState.defs st) _ _) hall (pre_refl _))
  have hr0 : ((DecState.defs st)[root]!).body.recut = #[] := by
    rw [hd, hroot] at hr
    by_cases hlt : root < (DecState.defs st).size
    · rw [← hpre.2 root hlt]
      exact hr
    · first
        | (rw [getElem!_neg (DecState.defs st) root hlt]; rfl)
        | simp [getElem!_neg, hlt]
  have hgo' := decodeDefs_plain hgo hr0
  unfold goOf at hgo'
  exact ⟨kv, desc, rootId, programFn, bm, root, _, st, st1, st2, pairs, hkv, hgo', rfl, rfl, rfl, hp, hpar, hall, hbk,
    hbod, hd, hi, hroot, hb⟩

/-- **What `decode` returns**, where its root reads no input node at another width (`Body.recut`; every reader of a gate
refuses one that does): the Definitions of a decoder state that keeps `Inv`, its root one of them, a composite without
parameters. -/
theorem decode_full (bytes : ByteArray) (P : Prog) (h : decode bytes = .ok P)
    (hr : (P.defs[P.root]!).body.recut = #[] := by assumption) :
    ∃ st, P.defs = DecState.defs st ∧ Inv st ∧ (P.defs[P.root]!).prim = false ∧ P.root < P.defs.size ∧
      (P.defs[P.root]!).params.size = 0 := by
  obtain ⟨_, _, _, _, _, root, s0, st, st1, st2, _, -, hgo, hd0, hi0, -, hp0, hpar, hall, -, hbod, hd, -, hroot, -⟩ :=
    decode_steps bytes P h hr
  obtain ⟨hinv, hlt⟩ := go_inv _ _ _ _ _ _ hgo
    ⟨fun i hi => absurd hi (by rw [hd0]; exact Nat.not_lt_zero i), fun k v hk => absurd hk (by rw [hi0]; simp),
      fun i hi => absurd hi (by rw [hd0]; exact Nat.not_lt_zero i),
      fun i hi => absurd hi (by rw [hd0]; exact Nat.not_lt_zero i)⟩
  obtain ⟨hinv1, hp1⟩ := decodeAll_grows _ _ _ _ _ hall hinv
  obtain ⟨hinv2, hp2⟩ := decodeBodies_grows _ _ _ _ _ _ hbod hinv1
  have hp := pre_trans hp1 hp2
  refine ⟨st2, hd, hinv2, ?_, ?_, ?_⟩
  · rw [hd, hroot, hp.2 root hlt]; exact hp0
  · rw [hd, hroot]; exact Nat.lt_of_lt_of_le hlt hp.1
  · rw [hd, hroot, hp.2 root hlt]; exact hpar

/-- **What `decode` returns**, where its root reads no input node at another width: the Definitions of a decoder state
that keeps `Inv`, its root a composite. -/
theorem decode_spec (bytes : ByteArray) (P : Prog) (h : decode bytes = .ok P)
    (hr : (P.defs[P.root]!).body.recut = #[] := by assumption) :
    ∃ st, P.defs = DecState.defs st ∧ Inv st ∧ (P.defs[P.root]!).prim = false := by
  obtain ⟨st, h1, h2, h3, -⟩ := decode_full bytes P h hr
  exact ⟨st, h1, h2, h3⟩

/-- **The root of a decoded Program** that reads no input node at another width is one of its Definitions, a composite
whose body has no parameters. -/
theorem decode_root (bytes : ByteArray) (P : Prog) (h : decode bytes = .ok P)
    (hr : (P.defs[P.root]!).body.recut = #[] := by assumption) :
    P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false ∧ (P.defs[P.root]!).body.params.size = 0 := by
  obtain ⟨st, hP, hinv, hp, hlt, hp0⟩ := decode_full bytes P h hr
  have hok : ∀ i < P.defs.size, DefOk P.defs i P.defs[i]! := by rw [hP]; exact hinv.1
  refine ⟨hlt, hp, ?_⟩
  rw [((hok P.root hlt).1.2 hp).1]
  exact hp0

/-- **A decoded Program's nodes read only earlier nodes**: the decoder refuses an argument naming its own node or a
later one (`forwardRefusal`). -/
theorem decode_fwd (bytes : ByteArray) (P : Prog) (h : decode bytes = .ok P)
    (hr : (P.defs[P.root]!).body.recut = #[] := by assumption) : ∀ i < P.defs.size, BodyFwd (P.defs[i]!).body := by
  obtain ⟨st, hP, hinv, -⟩ := decode_spec bytes P h hr
  rw [hP]; exact hinv.2.2.2

end InferenceOnlyVerifier.Properties.SecurityProofs.BodiesOk
