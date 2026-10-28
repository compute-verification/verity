import Proofs.Properties.Nci.BodiesOk.Composite

/-!
# What `decode` returns

`decode_spec`: a Program `Flock.Program.decode` returns holds the Definitions of a decoder state that keeps `Inv`
(`specs`, from the empty state through `go`), and its root is a composite (`decode`'s own check). `decode_root`: the
root is one of them (`getDef`'s index), a composite without parameters (`decode`'s check, `CalleeOk`).
-/

namespace NciCheck.Properties.SecurityProofs.BodiesOk
open Flock.Program
open private getDef composite staticFns DecState DecState.defs DecState.index DecState.busy Desc Desc.types Desc.callees
  Desc.ddefs liftV at_ paramsOf buildNode natJ? from Flock.Program

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

/-- **What `decode` returns**: the Definitions of a decoder state that keeps `Inv`, its root one of them, a composite
without parameters. -/
theorem decode_full (bytes : ByteArray) (P : Prog) (h : decode bytes = .ok P) :
    ∃ st, P.defs = DecState.defs st ∧ Inv st ∧ (P.defs[P.root]!).prim = false ∧ P.root < P.defs.size ∧
      (P.defs[P.root]!).params.size = 0 := by
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
  extract_lets bare jpS at h
  split at h
  swap; · exact (ex_throw_bind h).elim
  split at h
  · exact (ex_throw_bind h).elim
  simp (config := { zeta := false }) only [jpS] at h
  split at h
  swap; · exact absurd h (by intro h; injection h)
  obtain ⟨u, h1, h⟩ := Walk.except_bind_ok h
  clear h1
  extract_lets freeDefs free jpD at h
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
  obtain ⟨c, -, h⟩ := Walk.except_bind_ok h
  injection h with h
  subst h
  obtain ⟨hinv, hlt⟩ := go_inv _ _ _ _ _ _ hgo
    ⟨fun i hi => absurd hi (Nat.not_lt_zero i), fun k v hk => absurd hk (by simp)⟩
  simp only [rd, Bool.or_eq_true, not_or] at hrd
  exact ⟨st, rfl, hinv, by simpa using hrd.1, hlt, by simpa using hrd.2⟩

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

end NciCheck.Properties.SecurityProofs.BodiesOk
