module

public import Proofs.Properties.Flock.Decoder.Composite
public import Proofs.Properties.Flock.Decoder.Decode
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

`decode_steps` walks `decode` once: on an accepted descriptor it names the run of `go` from the empty state, the root's
check, `decodeAll` of the primitive entries, `booleanKeys`, `decodeBodies`, and the Program's fields from their last
state; every fact about what `decode` returns reads it. `decode_spec`: a Program `Flock.Program.decode` returns holds
the Definitions of a decoder state that keeps `Inv` (`specs`, from the empty state through `go`, `decodeAll` and
`decodeBodies`), and its root is a composite (`decode`'s own check). `decode_root`: the root is one of them
(`getDef`'s index), a composite without parameters (`decode`'s check, `CalleeOk`).
-/

@[expose] public section

namespace InferenceOnlyVerifier.Properties.SecurityProofs.BodiesOk
open Flock.Program
open private getDef composite staticFns DecState DecState.defs DecState.index DecState.busy Desc Desc.types Desc.callees
  Desc.ddefs liftV at_ paramsOf buildNode natJ? decodeAll decodeBodies primEntries from Circuits.Program

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

/-- **`decode`'s steps** on a descriptor it accepts: `go` from the empty state, the root's check, `decodeAll` of the
primitive entries, `booleanKeys` and `decodeBodies`, and the Program built from their last state. -/
theorem decode_steps (bytes : ByteArray) (P : Prog) (h : decode bytes = .ok P) :
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
  extract_lets go at h
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
  refine ⟨_, _, _, _, _, root, _, st, st1, st2, pairs, ?_, hgo, rfl, rfl, rfl, by simpa using hrd.1,
    by simpa using hrd.2, hall, hbk, hbod, rfl, rfl, rfl, rfl⟩
  rfl

/-- **What `decode` returns**: the Definitions of a decoder state that keeps `Inv`, its root one of them, a composite
without parameters. -/
theorem decode_full (bytes : ByteArray) (P : Prog) (h : decode bytes = .ok P) :
    ∃ st, P.defs = DecState.defs st ∧ Inv st ∧ (P.defs[P.root]!).prim = false ∧ P.root < P.defs.size ∧
      (P.defs[P.root]!).params.size = 0 := by
  obtain ⟨_, _, _, _, _, root, s0, st, st1, st2, _, -, hgo, hd0, hi0, -, hp0, hpar, hall, -, hbod, hd, -, hr, -⟩ :=
    decode_steps bytes P h
  obtain ⟨hinv, hlt⟩ := go_inv _ _ _ _ _ _ hgo
    ⟨fun i hi => absurd hi (by rw [hd0]; exact Nat.not_lt_zero i), fun k v hk => absurd hk (by rw [hi0]; simp)⟩
  obtain ⟨hinv1, hp1⟩ := decodeAll_grows _ _ _ _ _ hall hinv
  obtain ⟨hinv2, hp2⟩ := decodeBodies_grows _ _ _ _ _ _ hbod hinv1
  have hp := pre_trans hp1 hp2
  refine ⟨st2, hd, hinv2, ?_, ?_, ?_⟩
  · rw [hd, hr, hp.2 root hlt]; exact hp0
  · rw [hd, hr]; exact Nat.lt_of_lt_of_le hlt hp.1
  · rw [hd, hr, hp.2 root hlt]; exact hpar

/-- **What `decode` returns**: the Definitions of a decoder state that keeps `Inv`, its root a composite. -/
theorem decode_spec (bytes : ByteArray) (P : Prog) (h : decode bytes = .ok P) :
    ∃ st, P.defs = DecState.defs st ∧ Inv st ∧ (P.defs[P.root]!).prim = false := by
  obtain ⟨st, h1, h2, h3, -⟩ := decode_full bytes P h
  exact ⟨st, h1, h2, h3⟩

/-- **The root of a decoded Program** is one of its Definitions, a composite whose body has no parameters. -/
theorem decode_root (bytes : ByteArray) (P : Prog) (h : decode bytes = .ok P) :
    P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false ∧ (P.defs[P.root]!).body.params.size = 0 := by
  obtain ⟨st, hP, hinv, hp, hr, hp0⟩ := decode_full bytes P h
  have hok : ∀ i < P.defs.size, DefOk P.defs i P.defs[i]! := by rw [hP]; exact hinv.1
  refine ⟨hr, hp, ?_⟩
  rw [((hok P.root hr).1.2 hp).1]
  exact hp0

end InferenceOnlyVerifier.Properties.SecurityProofs.BodiesOk
