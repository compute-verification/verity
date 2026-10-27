import Proofs.Properties.Nci.BodiesOk.Nonneg
import Proofs.Properties.Nci.BodiesOk.Node

/-!
# The decoder's invariant

`Flock.Program.decode` builds its Definitions with `getDef`, `composite` and `staticFns`, mutually recursive on a fuel.
Every Definition it appends is `DefOk`: a primitive one gate with an empty body; a composite's nodes each call an
earlier Definition as `NodeOk` says (`buildNode_spec`), and every reference of its body names a leaf of a parameter or
of a node's declared output (`Walk.parsePartsIn_valid`). `decode_spec` proves it by induction on the fuel.
-/

namespace NciCheck.Properties.SecurityProofs.BodiesOk
open Flock.Program
open private getDef composite staticFns DecState DecState.defs DecState.index DecState.busy Desc liftV at_ paramsOf
  from Flock.Program

abbrev DM := StateT DecState (Except String)

theorem st_bind {α β : Type} {x : DM α} {f : α → DM β} {s : DecState} {b : β} {s' : DecState}
    (h : (x >>= f) s = .ok (b, s')) : ∃ a s1, x s = .ok (a, s1) ∧ f a s1 = .ok (b, s') := by
  simp only [bind, StateT.bind] at h
  rcases hx : x s with e | ⟨a, s1⟩
  · rw [hx] at h; exact absurd h (by intro h; injection h)
  · rw [hx] at h; exact ⟨a, s1, rfl, h⟩

theorem st_pure {α : Type} {a b : α} {s s' : DecState} (h : (pure a : DM α) s = .ok (b, s')) : a = b ∧ s = s' := by
  injection h with h; injection h with h1 h2; exact ⟨h1, h2⟩

theorem st_get {st s s1 : DecState} (h : (get : DM DecState) s = .ok (st, s1)) : s = st ∧ s = s1 := by
  injection h with h; injection h with h1 h2; exact ⟨h1, h2⟩

theorem st_liftV {α : Type} {x : Flock.V α} {s : DecState} {a : α} {s' : DecState}
    (h : liftV x s = .ok (a, s')) : x = .ok a ∧ s = s' := by
  cases x with
  | ok a' => injection h with h; injection h with h1 h2; subst h1 h2; exact ⟨rfl, rfl⟩
  | error e => exact absurd h (by intro h; injection h)

/-- A reference of body `b` names a leaf of one of its parameters or of one of its nodes' outputs. -/
def RefIn (b : Body) (r : Ref) : Prop :=
  (r.p = true → r.i < b.params.size ∧ r.j < (b.params[r.i]!).2.leaves) ∧
  (r.p = false → r.i < b.nodes.size ∧ r.j < (b.nodes[r.i]!).out.leaves)

/-- What the decoder establishes of the body of Definition `i`: each node calls an earlier Definition as `NodeOk`
says, its arguments and its return are sequences of parts, none naming a negative position, and every reference names a
leaf. -/
def BodyFacts (defs : Array Defn) (i : ℕ) (b : Body) : Prop :=
  (∀ k < b.nodes.size, (b.nodes[k]!).fn < i ∧ NodeOk defs[(b.nodes[k]!).fn]! (b.nodes[k]!) ∧
    ∀ a < (b.nodes[k]!).args.size,
      (∃ ps : Array Part, ((b.nodes[k]!).args[a]!).seq = Seq.ofParts ps ∧ ∀ q ∈ ps, PartNonneg q) ∧
      ∀ t < ((b.nodes[k]!).args[a]!).seq.count, RefIn b (((b.nodes[k]!).args[a]!).seq.get t)) ∧
  (∀ t < b.ret.count, RefIn b (b.ret.get t)) ∧
  ∃ ps : Array Part, b.ret = Seq.ofParts ps ∧ ∀ q ∈ ps, PartNonneg q

def DefOk (defs : Array Defn) (i : ℕ) (c : Defn) : Prop := CalleeOk c ∧ BodyFacts defs i c.body

/-- `b` extends `a`. -/
def Pre (a b : Array Defn) : Prop := a.size ≤ b.size ∧ ∀ i < a.size, b[i]! = a[i]!

/-- The decoder's invariant: every Definition decoded so far is `DefOk`, and the index names decoded ones. -/
def Inv (s : DecState) : Prop :=
  (∀ i < (DecState.defs s).size, DefOk (DecState.defs s) i (DecState.defs s)[i]!) ∧
  ∀ (k : String) (v : ℕ), (DecState.index s)[k]? = some v → v < (DecState.defs s).size

/-- A step of the decoder: it keeps the invariant and only appends Definitions. -/
def Grows (s s' : DecState) : Prop := Inv s → Inv s' ∧ Pre (DecState.defs s) (DecState.defs s')

theorem pre_refl (a : Array Defn) : Pre a a := ⟨le_refl _, fun _ _ => rfl⟩

theorem pre_trans {a b c : Array Defn} (h1 : Pre a b) (h2 : Pre b c) : Pre a c :=
  ⟨le_trans h1.1 h2.1, fun i hi => by rw [h2.2 i (by have := h1.1; omega), h1.2 i hi]⟩

theorem grows_refl (s : DecState) : Grows s s := fun h => ⟨h, pre_refl _⟩

theorem grows_trans {s1 s2 s3 : DecState} (h1 : Grows s1 s2) (h2 : Grows s2 s3) : Grows s1 s3 := fun h =>
  let ⟨i2, p2⟩ := h1 h
  let ⟨i3, p3⟩ := h2 i2
  ⟨i3, pre_trans p2 p3⟩

theorem bodyFacts_mono {a b : Array Defn} {i : ℕ} {body : Body} (hp : Pre a b) (hi : i ≤ a.size)
    (h : BodyFacts a i body) : BodyFacts b i body := by
  refine ⟨fun k hk => ?_, h.2⟩
  obtain ⟨h1, h2, h3⟩ := h.1 k hk
  exact ⟨h1, by rw [hp.2 _ (by omega)]; exact h2, h3⟩

theorem defOk_mono {a b : Array Defn} {i : ℕ} {c : Defn} (hp : Pre a b) (hi : i ≤ a.size)
    (h : DefOk a i c) : DefOk b i c := ⟨h.1, bodyFacts_mono hp hi h.2⟩

/-- `getDef`'s last step: the Definition appended and indexed. -/
theorem push_spec (s s' : DecState) (fid : String) (defn : Defn) (hs : Inv s)
    (hd : DefOk (DecState.defs s) (DecState.defs s).size defn)
    (e1 : DecState.defs s' = (DecState.defs s).push defn)
    (e2 : DecState.index s' = (DecState.index s).insert fid (DecState.defs s).size) :
    Inv s' ∧ Pre (DecState.defs s) (DecState.defs s') := by
  rw [e1]
  have hp : Pre (DecState.defs s) ((DecState.defs s).push defn) :=
    ⟨by simp, fun i hi => getElem!_push_lt _ _ hi⟩
  refine ⟨⟨fun i hi => ?_, fun k v hk => ?_⟩, hp⟩
  · rw [e1] at hi ⊢
    simp only [Array.size_push] at hi
    rcases Nat.lt_succ_iff_lt_or_eq.1 hi with hi | rfl
    · rw [getElem!_push_lt _ _ hi]
      exact defOk_mono hp (le_of_lt hi) (hs.1 i hi)
    · rw [getElem!_push_eq]
      exact defOk_mono hp (le_refl _) hd
  · rw [e2, Std.HashMap.getElem?_insert] at hk
    rw [e1]
    simp only [Array.size_push]
    split at hk
    · injection hk with hk; omega
    · have := hs.2 k v hk; omega

def GetSpec (fuel : ℕ) : Prop := ∀ (desc : Desc) (fid : String) (s : DecState) (i : ℕ) (s' : DecState),
  getDef fuel desc fid s = .ok (i, s') → Inv s → Inv s' ∧ Pre (DecState.defs s) (DecState.defs s') ∧
    i < (DecState.defs s').size

def CompSpec (fuel : ℕ) : Prop := ∀ (desc : Desc) (fid : String) (d : Lean.Json) (form id : String) (s : DecState)
  (defn : Defn) (s' : DecState), composite fuel desc fid d form id s = .ok (defn, s') → Inv s →
    Inv s' ∧ Pre (DecState.defs s) (DecState.defs s') ∧ DefOk (DecState.defs s') (DecState.defs s').size defn

def StatSpec (fuel : ℕ) : Prop := ∀ (desc : Desc) (v : Lean.Json) (s : DecState) (u : Unit) (s' : DecState),
  staticFns fuel desc v s = .ok (u, s') → Grows s s'

theorem default_bodyFacts (defs : Array Defn) (i : ℕ) : BodyFacts defs i default :=
  ⟨fun _ hk => absurd hk (Nat.not_lt_zero _), fun _ ht => absurd ht (Nat.not_lt_zero _), #[], rfl,
    fun _ h => absurd h (by simp)⟩

theorem getDef_succ (n : ℕ) (hc : CompSpec n) : GetSpec (n + 1) := by
  intro desc fid s i s' h hs
  unfold getDef at h
  extract_lets push inp main at h
  have hpush : ∀ defn s1 i s', push defn s1 = .ok (i, s') → Inv s1 →
      DefOk (DecState.defs s1) (DecState.defs s1).size defn →
      Inv s' ∧ Pre (DecState.defs s1) (DecState.defs s') ∧ i < (DecState.defs s').size := by
    intro defn s1 i s' hp hs hd
    simp only [push] at hp
    obtain ⟨st, s2, hg, hp⟩ := st_bind hp
    obtain ⟨rfl, rfl⟩ := st_get hg
    obtain ⟨u, s3, hset, hp⟩ := st_bind hp
    obtain ⟨rfl, rfl⟩ := st_pure hp
    injection hset with hset; injection hset with h1 h2; subst h2
    refine ⟨?_, ?_, by show _ < ((DecState.defs s1).push defn).size; simp⟩
    · exact (push_spec s1 _ fid defn hs hd rfl rfl).1
    · exact (push_spec s1 _ fid defn hs hd rfl rfl).2
  obtain ⟨st, s1, hg, h1⟩ := st_bind h
  obtain ⟨rfl, rfl⟩ := st_get hg
  clear h hg
  rcases hidx : (DecState.index s)[fid]? with _ | i0
  swap
  · rw [hidx] at h1
    obtain ⟨e1, e2⟩ := st_pure h1
    subst e1 e2
    exact ⟨hs, pre_refl _, hs.2 fid _ hidx⟩
  rw [hidx] at h1
  dsimp only at h1
  obtain ⟨st, s1, hg, h2⟩ := st_bind h1
  obtain ⟨rfl, rfl⟩ := st_get hg
  clear h1 hg
  split at h2
  · exact absurd h2 (by intro h; injection h)
  simp only [main] at h2
  rcases hdd : (Desc.ddefs desc)[fid]? with _ | d
  · rw [hdd] at h2; exact absurd h2 (by intro h; injection h)
  rw [hdd] at h2
  dsimp only at h2
  split at h2
  · split at h2
    · exact absurd h2 (by intro h; injection h)
    obtain ⟨params, s2, hp, h3⟩ := st_bind h2
    obtain ⟨-, rfl⟩ := st_liftV hp
    obtain ⟨ret, s3, hr, h4⟩ := st_bind h3
    obtain ⟨-, rfl⟩ := st_liftV hr
    clear h2 h3 hp hr
    rcases ret with w | _ | _ | _
    rotate_left
    all_goals try (dsimp only at h4; exact absurd h4 (by intro h; injection h))
    dsimp only at h4
    have hd : ∀ (key id : String) (binding : Option String) (pl : ℕ),
        DefOk (DecState.defs s) (DecState.defs s).size
          { key, id, prim := true, params, ret := .value w, binding, gates := 1, paramLeaves := pl, outLeaves := 1 } :=
      fun _ _ _ _ => ⟨⟨fun _ => ⟨rfl, rfl, rfl, w, rfl⟩, fun h => absurd h (by simp)⟩, default_bodyFacts _ _⟩
    rcases hinp : inp with _ | ⟨fst, iw⟩
    all_goals rw [hinp] at h4; dsimp only at h4
    · obtain ⟨defn, s4, hdf, h5⟩ := st_bind h4
      obtain ⟨rfl, rfl⟩ := st_pure hdf
      exact hpush _ _ _ _ h5 hs (hd _ _ _ _)
    · split at h4
      · exact absurd h4 (by intro h; injection h)
      obtain ⟨defn, s4, hdf, h5⟩ := st_bind h4
      obtain ⟨rfl, rfl⟩ := st_pure hdf
      exact hpush _ _ _ _ h5 hs (hd _ _ _ _)
  · obtain ⟨defn, s2, hdf, h3⟩ := st_bind h2
    obtain ⟨i2, p2, d2⟩ := hc _ _ _ _ _ _ _ _ hdf hs
    obtain ⟨i3, p3, h4⟩ := hpush _ _ _ _ h3 i2 d2
    exact ⟨i3, pre_trans p2 p3, h4⟩

theorem forIn_grows {α β : Type} (l : List α) (f : α → β → DM (ForInStep β))
    (hf : ∀ x b s t s', f x b s = .ok (t, s') → Grows s s') :
    ∀ (b : β) (s : DecState) (r : β) (s' : DecState), forIn l b f s = .ok (r, s') → Grows s s' := by
  intro b s r s' h
  have := forIn_stateT_inv f (fun _ s1 => Grows s s1) l (fun _ _ s1 => Grows s s1) b r s s' (grows_refl s)
    (fun i hi b s1 t s2 hg hf' => by
      have := grows_trans hg (hf _ _ _ _ _ hf')
      rcases t <;> exact this) h
  rcases this with h | h <;> exact h

theorem st_discard {α : Type} {x : DM α} {s : DecState} {u : PUnit} {s' : DecState}
    (h : (discard x : DM PUnit) s = .ok (u, s')) : ∃ a, x s = .ok (a, s') := by
  rcases hx : x s with e | ⟨a, s1⟩
  · have : (discard x : DM PUnit) s = .error e := by
      simp only [discard, Functor.mapConst, Function.comp, StateT.map, hx]; rfl
    rw [this] at h; exact absurd h (by intro h; injection h)
  · have : (discard x : DM PUnit) s = .ok (PUnit.unit, s1) := by
      simp only [discard, Functor.mapConst, Function.comp, StateT.map, hx]; rfl
    rw [this] at h; injection h with h; injection h with h1 h2; subst h2; exact ⟨a, rfl⟩

theorem statFns_succ (n : ℕ) (hg : GetSpec n) (hst : StatSpec n) : StatSpec (n + 1) := by
  intro desc v s u s' h
  unfold staticFns at h
  rcases v with _ | _ | _ | _ | a | kvs
  all_goals dsimp only at h
  case null | bool | num | str => obtain ⟨-, rfl⟩ := st_pure h; exact grows_refl _
  case arr =>
    obtain ⟨u1, s1, hl, h2⟩ := st_bind h
    obtain ⟨-, rfl⟩ := st_pure h2
    rw [← Array.forIn_toList] at hl
    exact forIn_grows _ _ (fun x b s t s' hf => by
      obtain ⟨u2, s2, h1, h2⟩ := st_bind hf
      obtain ⟨-, rfl⟩ := st_pure h2
      exact hst _ _ _ _ _ h1) _ _ _ _ hl
  case obj =>
    rcases hfn : (Lean.Json.obj kvs).getObjVal? "fn" with e | f
    · rw [hfn] at h
      dsimp only at h
      rcases hf64 : (Lean.Json.obj kvs).getObjVal? "f64" with e | f
      · rw [hf64] at h
        dsimp only at h
        rcases hdict : (Lean.Json.obj kvs).getObjVal? "dict" with e | dd
        · rw [hdict] at h
          dsimp only at h
          obtain ⟨-, rfl⟩ := st_pure h; exact grows_refl _
        · rw [hdict] at h
          dsimp only at h
          rcases dd with _ | _ | _ | _ | _ | kvs2
          all_goals dsimp only at h
          all_goals try (exact absurd h (by intro h; injection h))
          obtain ⟨u1, s1, hl, h2⟩ := st_bind h
          obtain ⟨-, rfl⟩ := st_pure h2
          rw [← Array.forIn_toList] at hl
          exact forIn_grows _ _ (fun x b s t s' hf => by
            obtain ⟨u2, s2, h1, h2⟩ := st_bind hf
            obtain ⟨-, rfl⟩ := st_pure h2
            exact hst _ _ _ _ _ h1) _ _ _ _ hl
      · rw [hf64] at h
        dsimp only at h
        rcases f with _ | _ | _ | _ | _ | _
        all_goals dsimp only at h
        rotate_left 3
        · obtain ⟨-, rfl⟩ := st_pure h; exact grows_refl _
        all_goals exact absurd h (by intro h; injection h)
    · rw [hfn] at h
      dsimp only at h
      rcases f with _ | _ | _ | f | _ | _
      all_goals dsimp only at h
      all_goals try (exact absurd h (by intro h; injection h))
      obtain ⟨a, ha⟩ := st_discard h
      intro hs
      obtain ⟨h1, h2, -⟩ := hg _ _ _ _ _ ha hs
      exact ⟨h1, h2⟩

end NciCheck.Properties.SecurityProofs.BodiesOk
