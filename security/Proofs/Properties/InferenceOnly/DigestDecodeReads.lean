module

public import Proofs.Properties.InferenceOnly.DigestReads
public import Proofs.Properties.InferenceOnly.BodiesOk.Decode
import Proofs.Properties.InferenceOnly.BodiesOk.Composite
import Proofs.Properties.InferenceOnly.BodiesOk.Basic
import Proofs.Properties.InferenceOnly.Walk
import Proofs.Properties.InferenceOnly.DigestReadsPin
import Proofs.Properties.InferenceOnly.DecodeKeys
import Proofs.Flock.Claims.Complete.Json.Parse
import Proofs.Flock.Claims.Complete.Encode
import Batteries.Tactic.OpenPrivate
import Circuits.Program
import all Lean.Data.Json.Basic
import all Lean.Data.Json.FromToJson.Basic
import Std.Data.HashMap.Lemmas
import Std.Data.TreeMap.Raw.Lemmas

/-!
# The decoder builds each Definition from its entry (`DecodeReads`)

`decodeReads` carries an invariant through the decoder's fueled walk, as `BodiesOk` carries `Inv`: the index names each
Definition decoded so far by its key, and each is what `PrimReads` or `CompReads` says of its entry, in any Program
whose Definitions extend the decoder's (`RInv`). `getDef`, `composite` and `staticFns` keep it at each fuel
(`specsR`); a composite's node loop builds its nodes against a stand-in Definition holding the nodes and offsets so far
(`stand`), which `NodeReads` reads alike up to the node it reads (`nodeReads_congr`). `decode`'s descriptor is the
parsed text's (`DescOf`): its Definitions are the text's entries, since the parsed object is well formed.
-/

public section

namespace InferenceOnlyVerifier.Properties.Record.SecurityProofs

namespace DecodeReadsProof

open Lean (Json)
open Flock.Program
open Claims.Json (Wf)
open InferenceOnlyVerifier.Properties.SecurityProofs.BodiesOk (DM st_bind st_pure st_get st_liftV forIn_stateT_inv Pre
  pre_refl pre_trans getElem!_push_lt getElem!_push_eq st_modify forIn_same array_mapM_same st_ite st_throw_bind
  st_get_bind st_discard array_mapM_size array_mapM_get)
open InferenceOnlyVerifier.Properties.SecurityProofs.Walk (except_bind_ok)
open ReadsPinProof (getObjValD_of getObjValD_eq str_of_field at_ok getNat_of_intJ push_get_lt push_get_eq)
open private getDef composite staticFns DecState DecState.defs DecState.index DecState.busy DecState.walked Desc
  Desc.ddefs Desc.types Desc.callees liftV at_ paramsOf buildNode natJ? intJ? from Circuits.Program

/-- The descriptor `desc` is `j`'s: its callees and types are `j`'s tables, and its Definitions `j`'s entries. -/
def DescOf (j : Json) (desc : Desc) : Prop :=
  Desc.callees desc = tab j "callees" ∧ (tab j "types").mapM Ty.ofJson = .ok (Desc.types desc) ∧
    ∀ fid d, (Desc.ddefs desc)[fid]? = some d → d = entry j fid

/-- What the decoder keeps of its state: the index names each Definition by its key, and each Definition decoded so
far is what `PrimReads` or `CompReads` says, in any Program whose Definitions extend these. -/
def RInv (j : Json) (s : DecState) : Prop :=
  (∀ k v, (DecState.index s)[k]? = some v → v < (DecState.defs s).size ∧ ((DecState.defs s)[v]!).key = k) ∧
  ∀ (P : Prog), Pre (DecState.defs s) P.defs → ∀ i < (DecState.defs s).size,
    PrimReads j (DecState.defs s)[i]! ∨ CompReads j P i (DecState.defs s)[i]!

def GetR (j : Json) (fuel : Nat) : Prop := ∀ (desc : Desc) (fid : String) (s : DecState) (i : Nat) (s' : DecState),
  DescOf j desc → getDef fuel desc fid s = .ok (i, s') → RInv j s →
    RInv j s' ∧ Pre (DecState.defs s) (DecState.defs s') ∧ i < (DecState.defs s').size ∧
      ((DecState.defs s')[i]!).key = fid

def CompR (j : Json) (fuel : Nat) : Prop := ∀ (desc : Desc) (fid : String) (d : Json) (form id : String) (s : DecState)
  (defn : Defn) (s' : DecState), DescOf j desc → d = entry j fid → form = (d.getObjValAs? String "form").toOption.getD "" →
    id = (d.getObjValAs? String "id").toOption.getD "" →
    composite fuel desc fid d form id s = .ok (defn, s') → RInv j s →
    RInv j s' ∧ Pre (DecState.defs s) (DecState.defs s') ∧ defn.key = fid ∧
      ∀ (P : Prog), Pre (DecState.defs s') P.defs → CompReads j P (DecState.defs s').size defn

/-- A step of the decoder keeps `RInv` and only appends Definitions. -/
def RGrows (j : Json) (s s' : DecState) : Prop := RInv j s → RInv j s' ∧ Pre (DecState.defs s) (DecState.defs s')

def StatR (j : Json) (fuel : Nat) : Prop := ∀ (desc : Desc) (v : Json) (s : DecState) (u : Unit) (s' : DecState),
  DescOf j desc → staticFns fuel desc v s = .ok (u, s') → RGrows j s s'

theorem rgrows_refl (j : Json) (s : DecState) : RGrows j s s := fun h => ⟨h, pre_refl _⟩

theorem rgrows_trans {j : Json} {s1 s2 s3 : DecState} (h1 : RGrows j s1 s2) (h2 : RGrows j s2 s3) : RGrows j s1 s3 :=
  fun h => let ⟨a, b⟩ := h1 h; let ⟨c, d⟩ := h2 a; ⟨c, pre_trans b d⟩

theorem forIn_rgrows {j : Json} {α β : Type} (l : List α) (f : α → β → DM (ForInStep β))
    (hf : ∀ x b s t s', f x b s = .ok (t, s') → RGrows j s s') :
    ∀ (b : β) (s : DecState) (r : β) (s' : DecState), forIn l b f s = .ok (r, s') → RGrows j s s' := by
  intro b s r s' h
  have := forIn_stateT_inv f (fun _ s1 => RGrows j s s1) l (fun _ _ s1 => RGrows j s s1) b r s s' (rgrows_refl j s)
    (fun i hi b s1 t s2 hg hf' => by
      have := rgrows_trans hg (hf _ _ _ _ _ hf')
      rcases t <;> exact this) h
  rcases this with h | h <;> exact h

theorem pre_push (a : Array Defn) (x : Defn) : Pre a (a.push x) :=
  ⟨by simp, fun _ hi => getElem!_push_lt _ _ hi⟩

/-- `getDef`'s push keeps `RInv`. -/
theorem push_rinv {j : Json} {s s' : DecState} {fid : String} {defn : Defn} (hs : RInv j s) (hkey : defn.key = fid)
    (hd : ∀ (P : Prog), Pre ((DecState.defs s).push defn) P.defs →
      PrimReads j defn ∨ CompReads j P (DecState.defs s).size defn)
    (e1 : DecState.defs s' = (DecState.defs s).push defn)
    (e2 : DecState.index s' = (DecState.index s).insert fid (DecState.defs s).size) :
    RInv j s' ∧ Pre (DecState.defs s) (DecState.defs s') ∧ (DecState.defs s).size < (DecState.defs s').size ∧
      ((DecState.defs s')[(DecState.defs s).size]!).key = fid := by
  rw [e1]
  refine ⟨⟨fun k v hk => ?_, fun P hP i hi => ?_⟩, pre_push _ _, by simp, by rw [getElem!_push_eq, hkey]⟩
  · rw [e2, Std.HashMap.getElem?_insert] at hk
    rw [e1]
    split at hk
    · rename_i hf
      injection hk with hk
      subst hk
      exact ⟨by simp, by rw [getElem!_push_eq, hkey, eq_of_beq hf]⟩
    · obtain ⟨h1, h2⟩ := hs.1 k v hk
      exact ⟨by simp; omega, by rw [getElem!_push_lt _ _ h1, h2]⟩
  · rw [e1] at hP hi ⊢
    have hi : i < (DecState.defs s).size + 1 := by simpa using hi
    rcases Nat.lt_succ_iff_lt_or_eq.1 hi with hi | rfl
    · rw [getElem!_push_lt _ _ hi]
      exact hs.2 P (pre_trans (pre_push _ _) hP) i hi
    · rw [getElem!_push_eq]
      exact hd P hP

/-- The Definition `getDef` builds for a primitive. -/
def primDefn (fid id : String) (params : Array (String × Ty)) (w : Nat) (binding : Option String) : Defn :=
  { key := fid, id, prim := true, params, ret := .value w, binding, gates := 1,
    paramLeaves := params.foldl (· + ·.2.leaves) 0, outLeaves := 1 }

theorem getDef_succR (j : Json) (n : Nat) (hc : CompR j n) : GetR j (n + 1) := by
  intro desc fid s i s' hdesc h hs
  unfold getDef at h
  extract_lets push inp main at h
  have hpush : ∀ defn s1 i s', push defn s1 = .ok (i, s') → RInv j s1 → defn.key = fid →
      (∀ (P : Prog), Pre ((DecState.defs s1).push defn) P.defs →
        PrimReads j defn ∨ CompReads j P (DecState.defs s1).size defn) →
      RInv j s' ∧ Pre (DecState.defs s1) (DecState.defs s') ∧ i < (DecState.defs s').size ∧
        ((DecState.defs s')[i]!).key = fid := by
    intro defn s1 i s' hp hs hk hd
    simp only [push] at hp
    obtain ⟨st, s2, hg, hp⟩ := st_bind hp
    obtain ⟨rfl, rfl⟩ := st_get hg
    obtain ⟨u, s3, hset, hp⟩ := st_bind hp
    obtain ⟨rfl, rfl⟩ := st_pure hp
    injection hset with hset; injection hset with h1 h2; subst h2
    exact push_rinv hs hk hd rfl rfl
  obtain ⟨st, s1, hg, h1⟩ := st_bind h
  obtain ⟨rfl, rfl⟩ := st_get hg
  clear h hg
  rcases hidx : (DecState.index s)[fid]? with _ | i0
  rotate_left
  · rw [hidx] at h1
    obtain ⟨e1, e2⟩ := st_pure h1
    subst e1 e2
    exact ⟨hs, pre_refl _, hs.1 fid _ hidx⟩
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
  have hde := hdesc.2.2 fid d hdd
  subst hde
  dsimp only at h2
  split at h2 <;> rename_i hform
  · split at h2
    · exact absurd h2 (by intro h; injection h)
    obtain ⟨params, s2, hp, h3⟩ := st_bind h2
    obtain ⟨hpar, rfl⟩ := st_liftV hp
    obtain ⟨ret, s3, hr, h4⟩ := st_bind h3
    obtain ⟨hret, rfl⟩ := st_liftV hr
    clear h2 h3 hp hr
    rcases ret with w | _ | _ | _
    rotate_left
    all_goals try (dsimp only at h4; exact absurd h4 (by intro h; injection h))
    dsimp only at h4
    have hd : ∀ (binding : Option String), binding = (inputId? fid).map (·.1) → ∀ (P : Prog),
        Pre ((DecState.defs s).push (primDefn fid (((entry j fid).getObjValAs? String "id").toOption.getD "") params w
          binding)) P.defs →
        PrimReads j (primDefn fid (((entry j fid).getObjValAs? String "id").toOption.getD "") params w binding) ∨
        CompReads j P (DecState.defs s).size
          (primDefn fid (((entry j fid).getObjValAs? String "id").toOption.getD "") params w binding) := by
      intro binding hb P _
      exact Or.inl ⟨rfl, rfl, eq_of_beq hform, rfl, hpar, hret, hb, rfl, rfl, rfl⟩
    rcases hinp : inp with _ | ⟨fst, iw⟩
    all_goals rw [hinp] at h4; dsimp only at h4
    · obtain ⟨defn, s4, hdf, h5⟩ := st_bind h4
      obtain ⟨rfl, rfl⟩ := st_pure hdf
      exact hpush _ _ _ _ h5 hs rfl (hd _ (by show _ = inp.map _; rw [hinp]))
    · split at h4
      · exact absurd h4 (by intro h; injection h)
      obtain ⟨defn, s4, hdf, h5⟩ := st_bind h4
      obtain ⟨rfl, rfl⟩ := st_pure hdf
      exact hpush _ _ _ _ h5 hs rfl (hd _ (by show _ = inp.map _; rw [hinp]))
  · obtain ⟨defn, s2, hdf, h3⟩ := st_bind h2
    obtain ⟨i2, p2, k2, d2⟩ := hc _ _ _ _ _ _ _ _ hdesc rfl rfl rfl hdf hs
    obtain ⟨i3, p3, h4⟩ := hpush _ _ _ _ h3 i2 k2 (fun P hP => Or.inr (d2 P (pre_trans (pre_push _ _) hP)))
    exact ⟨i3, pre_trans p2 p3, h4⟩

theorem statFns_succR (j : Json) (n : Nat) (hg : GetR j n) (hst : StatR j n) : StatR j (n + 1) := by
  intro desc v s u s' hdesc h
  unfold staticFns at h
  rcases v with _ | _ | _ | _ | a | kvs
  all_goals dsimp only at h
  case null | bool | num | str => obtain ⟨-, rfl⟩ := st_pure h; exact rgrows_refl _ _
  case arr =>
    obtain ⟨u1, s1, hl, h2⟩ := st_bind h
    obtain ⟨-, rfl⟩ := st_pure h2
    rw [← Array.forIn_toList] at hl
    exact forIn_rgrows _ _ (fun x b s t s' hf => by
      obtain ⟨u2, s2, h1, h2⟩ := st_bind hf
      obtain ⟨-, rfl⟩ := st_pure h2
      exact hst _ _ _ _ _ hdesc h1) _ _ _ _ hl
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
          obtain ⟨-, rfl⟩ := st_pure h; exact rgrows_refl _ _
        · rw [hdict] at h
          dsimp only at h
          rcases dd with _ | _ | _ | _ | _ | kvs2
          all_goals dsimp only at h
          all_goals try (exact absurd h (by intro h; injection h))
          obtain ⟨u1, s1, hl, h2⟩ := st_bind h
          obtain ⟨-, rfl⟩ := st_pure h2
          rw [← Array.forIn_toList] at hl
          exact forIn_rgrows _ _ (fun x b s t s' hf => by
            obtain ⟨u2, s2, h1, h2⟩ := st_bind hf
            obtain ⟨-, rfl⟩ := st_pure h2
            exact hst _ _ _ _ _ hdesc h1) _ _ _ _ hl
      · rw [hf64] at h
        dsimp only at h
        rcases f with _ | _ | _ | _ | _ | _
        all_goals dsimp only at h
        rotate_left 3
        · obtain ⟨-, rfl⟩ := st_pure h; exact rgrows_refl _ _
        all_goals exact absurd h (by intro h; injection h)
    · rw [hfn] at h
      dsimp only at h
      rcases f with _ | _ | _ | f | _ | _
      all_goals dsimp only at h
      all_goals try (exact absurd h (by intro h; injection h))
      obtain ⟨a, ha⟩ := st_discard h
      intro hs
      obtain ⟨h1, h2, -⟩ := hg _ _ _ _ _ hdesc ha hs
      exact ⟨h1, h2⟩

theorem rinv_congr {j : Json} {s s' : DecState} (e1 : DecState.defs s' = DecState.defs s)
    (e2 : DecState.index s' = DecState.index s) (h : RInv j s) : RInv j s' := by
  unfold RInv; rw [e1, e2]; exact h

/-- The Definition a composite's node loop has built so far: what `NodeReads` reads of it. -/
def stand (fid : String) (params : Array (String × Ty)) (nodes : Array Node) (offsets : Array Nat) : Defn :=
  { key := fid, id := "", prim := false, params, ret := default,
    body := { params, retType := default, nodes, ret := default, offsets }, gates := 0, paramLeaves := 0,
    outLeaves := 0 }

/-- `NodeReads` of node `k` reads only the key, the parameters, the nodes up to `k` and offsets `k` and `k + 1`. -/
theorem nodeReads_congr {P : Prog} {types outs : Array Ty} {d d' : Defn} {k : Nat} {nd : Json}
    (hk : d'.key = d.key) (hp : d'.params = d.params)
    (hn : ∀ k2 ≤ k, d'.body.nodes[k2]! = d.body.nodes[k2]!)
    (ho0 : d'.body.offsets[k]! = d.body.offsets[k]!) (ho1 : d'.body.offsets[k + 1]! = d.body.offsets[k + 1]!)
    (h : NodeReads P types d outs k nd) : NodeReads P types d' outs k nd := by
  have hl : leavesOf d' outs = leavesOf d outs := by unfold leavesOf; rw [hp]
  have he : (fun (k2 x : Nat) => if k2 < k then (d'.body.nodes[k2]!).args[x]?.map (fun (a : Arg) => a.seq) else none) =
      (fun (k2 x : Nat) => if k2 < k then (d.body.nodes[k2]!).args[x]?.map (fun (a : Arg) => a.seq) else none) := by
    funext k2 x
    split
    · rename_i h2; rw [hn k2 (Nat.le_of_lt h2)]
    · rfl
  unfold NodeReads at h ⊢
  rw [hl, he, hk, ho0, ho1, hn k (Nat.le_refl k)]
  exact h

/-- The callee `at_` reads from a node's `fn` is the one `calleeAt` reads. -/
theorem calleeAt_of_at {cs : Array Json} {nd c : Json} {what : String}
    (h : at_ cs ((nd.getObjVal? "fn").toOption.getD Json.null) what = .ok c) : calleeAt cs nd = some c := by
  obtain ⟨i, hi, h0, hlt, rfl⟩ := at_ok h
  rw [getObjValD_eq] at hi
  unfold calleeAt
  rw [getNat_of_intJ hi h0]
  simp only [Except.toOption, Option.bind_some]
  have : i.toNat < cs.size := by omega
  simp [this]

/-- A `mapM` in `D` of a step that reads no state is the `Option` `mapM` of what each step returns. -/
theorem list_mapM_dm {α β : Type} (g : α → DM β) (f : α → Option β)
    (hg : ∀ x s a s', g x s = .ok (a, s') → f x = some a ∧ s' = s) :
    ∀ (xs : List α) (s : DecState) (as : List β) (s' : DecState), xs.mapM g s = .ok (as, s') → xs.mapM f = some as
  | [], s, as, s', h => by
    simp only [List.mapM_nil] at h ⊢
    obtain ⟨rfl, -⟩ := st_pure h; rfl
  | x :: xs, s, as, s', h => by
    rw [List.mapM_cons] at h
    obtain ⟨a, s1, h1, h⟩ := st_bind h
    obtain ⟨as', s2, h2, h⟩ := st_bind h
    obtain ⟨rfl, -⟩ := st_pure h
    rw [List.mapM_cons, list_mapM_dm g f hg xs s1 as' s2 h2, (hg x s a s1 h1).1]
    rfl

theorem array_mapM_dm {α β : Type} (g : α → DM β) (f : α → Option β)
    (hg : ∀ x s a s', g x s = .ok (a, s') → f x = some a ∧ s' = s) (xs : Array α) (s : DecState) (as : Array β)
    (s' : DecState) (h : xs.mapM g s = .ok (as, s')) : xs.mapM f = some as := by
  rw [Array.mapM_eq_mapM_toList] at h ⊢
  rcases hl : xs.toList.mapM g s with e | ⟨l, s1⟩
  · have : (List.toArray <$> xs.toList.mapM g) s = .error e := by
      simp only [Functor.map, StateT.map, hl]; rfl
    rw [this] at h; exact absurd h (by intro h; injection h)
  · have : (List.toArray <$> xs.toList.mapM g) s = .ok (l.toArray, s1) := by
      simp only [Functor.map, StateT.map, hl]; rfl
    rw [this] at h
    injection h with h; injection h with h1 h2; subst h1
    rw [list_mapM_dm g f hg _ _ _ _ hl]; rfl

theorem composite_succR (j : Json) (n : Nat) (hg : GetR j n) (hst : StatR j n) : CompR j (n + 1) := by
  intro desc fid d form id s defn s' hdesc hd hform hid h hs
  subst hd
  unfold composite at h
  extract_lets parts name ver bindings rec0 b nodes0 offsets0 order main at h
  split at h
  · exact absurd h (by intro h; injection h)
  rename_i hck
  have hfc : form = "composite" := by
    cases hf : (form == "composite")
    · rw [hf] at hck; exact absurd rfl hck
    · exact eq_of_beq hf
  clear hck
  simp (config := { zeta := false }) only [main] at h
  rcases hbk : bindings with _ | _ | _ | _ | _ | bkv
  all_goals rw [hbk] at h
  all_goals dsimp (config := { zeta := false }) only at h
  all_goals try (exact absurd h (by intro h; injection h))
  extract_lets names jp1 jpS jpO at h
  have hS : ∃ X, jpS X s = .ok (defn, s') := by
    split at h
    · replace h := st_bind h; obtain ⟨a, s1, h1, h2⟩ := h
      obtain ⟨rfl, rfl⟩ := st_pure h1
      exact ⟨_, h2⟩
    · split at h
      · replace h := st_bind h; obtain ⟨a, s1, h1, -⟩ := h
        exact absurd h1 (by intro h; injection h)
      · simp (config := { zeta := false }) only [jpO] at h
        replace h := st_bind h; obtain ⟨a, s1, h1, h2⟩ := h
        obtain ⟨rfl, rfl⟩ := st_pure h1
        exact ⟨_, h2⟩
  clear h
  obtain ⟨X, h⟩ := hS
  simp (config := { zeta := false }) only [jpS] at h
  replace h := st_bind h; obtain ⟨rr, s1, hf, h⟩ := h
  rw [← Array.forIn_toList] at hf
  have e1 := forIn_same _ _ (fun x b s t s' hf => by
    obtain ⟨a, s2, h1, h2⟩ := st_bind hf
    obtain ⟨-, rfl⟩ := st_liftV h1
    exact (st_pure h2).2.symm) _ _ _ _ hf
  subst s1
  clear hf
  replace h := st_bind h; obtain ⟨u, s2, hm, h⟩ := h
  have e2 := st_modify hm
  have hs2 : RInv j s2 := rinv_congr (by rw [e2]) (by rw [e2]) hs
  have hp2 : DecState.defs s2 = DecState.defs s := by rw [e2]
  clear hm e2
  generalize (id ++ if X.isEmpty = true then "" else "{" ++ ",".intercalate rr.toList ++ "}") = key at h
  split at h
  · replace h := st_bind h; obtain ⟨a, s3, h1, -⟩ := h
    exact absurd h1 (by intro h; injection h)
  simp (config := { zeta := false }) only [jp1] at h
  replace h := st_bind h; obtain ⟨u, s3, hf, h⟩ := h
  rw [← Array.forIn_toList] at hf
  have g3 := forIn_rgrows _ _ (fun x b s t s' hf => by
    obtain ⟨u2, s2, h1, h2⟩ := st_bind hf
    obtain ⟨-, rfl⟩ := st_pure h2
    exact hst _ _ _ _ _ hdesc h1) _ _ _ _ hf
  obtain ⟨hs3, hp3⟩ := g3 hs2
  rw [hp2] at hp3
  clear hf g3 hs2 hp2 hs
  replace h := st_bind h; obtain ⟨params, s4, hpa, h⟩ := h
  replace hpa := st_liftV hpa; obtain ⟨hpa, rfl⟩ := hpa
  replace h := st_bind h; obtain ⟨retType, s4, hrt, h⟩ := h
  replace hrt := st_liftV hrt; obtain ⟨hrt, rfl⟩ := hrt
  rcases hnd : b.getObjVal? "nodes" with e | (_ | _ | _ | _ | nds | _)
  all_goals rw [hnd] at h
  all_goals dsimp (config := { zeta := false }) only at h
  all_goals try (exact absurd h (by intro h; injection h))
  replace h := st_bind h; obtain ⟨outs, s4, hou, h⟩ := h
  replace hou := st_liftV hou; obtain ⟨hou, rfl⟩ := hou
  replace h := st_bind h; obtain ⟨⟨nodes, offsets⟩, s5, hloop, h⟩ := h
  have hosz : outs.size = nds.size := array_mapM_size _ _ _ hou
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hloop
  have hL := forIn_stateT_inv _ (fun _ _ => False) _
    (fun i (st : Array Node × Array Nat) s1 => RInv j s1 ∧ Pre (DecState.defs s3) (DecState.defs s1) ∧
      st.1.size = i ∧ st.2.size = i + 1 ∧ st.2[0]! = 0 ∧
      ∀ (P : Prog), Pre (DecState.defs s1) P.defs → ∀ k < i, (st.1[k]!).fn < (DecState.defs s1).size ∧
        calleeAt (tab j "callees") nds[k]! = some (.str (P.defs[(st.1[k]!).fn]!).key) ∧
        NodeReads P (Desc.types desc) (stand fid params st.1 st.2) outs k nds[k]!) (nodes0, offsets0) _ _ _
    ⟨hs3, pre_refl _, rfl, rfl, rfl, fun _ _ k hk => absurd hk (Nat.not_lt_zero k)⟩ ?step hloop
  case step =>
    rintro i hi ⟨nodes1, offsets1⟩ s1 t s2 ⟨hs1, hp1, hsz1, hosz1, ho01, hn1⟩ h
    simp only [List.length_range', Std.Legacy.Range.size] at hi
    simp (config := { zeta := false }) only [List.getElem_range', Nat.one_mul, Nat.zero_add] at h
    dsimp only at hsz1 hosz1 ho01 hn1
    subst hsz1
    replace h := st_bind h; obtain ⟨fname, s3', h1, h⟩ := h
    replace h1 := st_liftV h1; obtain ⟨hfn, rfl⟩ := h1
    rcases fname with _ | _ | _ | fname | _ | _
    all_goals dsimp (config := { zeta := false }) only at h
    all_goals try (replace h := st_bind h; obtain ⟨a, s7, h1, -⟩ := h; exact absurd h1 (by intro h; injection h))
    replace h := st_bind h; obtain ⟨ci, s4, hgd, h⟩ := h
    obtain ⟨hs4, hp4, hci, hkey⟩ := hg _ _ _ _ _ hdesc hgd hs1
    replace h := st_get_bind h; obtain ⟨g, hgd', h⟩ := h
    extract_lets callee fstr keys earlier JP at hgd'
    subst hgd'
    simp (config := { zeta := false }) only [fstr] at h
    have hJ : ∃ F, formOf ((nds[nodes1.size]!.getObjValAs? String "form").toOption.getD "") = some F ∧
        JP F s4 = .ok (t, s2) := by
      generalize hfs : (nds[nodes1.size]!.getObjValAs? String "form").toOption.getD "" = fs at h
      split at h
      all_goals replace h := st_bind h; obtain ⟨F, s5, h1, h⟩ := h
      rotate_left 4
      · exact absurd h1 (by intro h; injection h)
      all_goals obtain ⟨rfl, rfl⟩ := st_pure h1; exact ⟨_, by first | rfl | (rw [hfs]; rfl), h⟩
    clear h
    obtain ⟨F, hF, h⟩ := hJ
    simp (config := { zeta := false }) only [JP] at h
    lift_lets at h
    extract_lets jpC jpP jpK at h
    replace h := st_ite h; rcases h with ⟨-, h⟩ | ⟨hkeys, h⟩
    · exact (st_throw_bind h).elim
    simp (config := { zeta := false }) only [jpK] at h
    replace h := st_ite h; rcases h with ⟨-, h⟩ | ⟨hprim, h⟩
    · exact (st_throw_bind h).elim
    simp (config := { zeta := false }) only [jpP] at h
    replace h := st_ite h; rcases h with ⟨-, h⟩ | ⟨hcall, h⟩
    · exact (st_throw_bind h).elim
    simp (config := { zeta := false }) only [jpC] at h
    rcases hargs : nds[nodes1.size]!.getObjVal? "args" with e | ⟨_ | _ | _ | _ | ajs | _⟩
    all_goals rw [hargs] at h
    all_goals dsimp (config := { zeta := false }) only at h
    all_goals try (exact (st_throw_bind h).elim)
    lift_lets at h
    extract_lets jpAx at h
    have hAx : ∃ axes, (if F == .batch then ∃ xs, nds[nodes1.size]!.getObjVal? "axes" = .ok (.arr xs) ∧
        xs.mapM axisOf = some axes else axes = #[]) ∧ jpAx axes s4 = .ok (t, s2) := by
      replace h := st_ite h; rcases h with ⟨hb, h⟩ | ⟨hb, h⟩
      · rcases haxes : nds[nodes1.size]!.getObjVal? "axes" with e | ⟨_ | _ | _ | _ | xs | _⟩
        all_goals rw [haxes] at h
        all_goals dsimp (config := { zeta := false }) only at h
        all_goals try (exact (st_throw_bind h).elim)
        replace h := st_bind h; obtain ⟨axes, s5, h1, h⟩ := h
        have hax := array_mapM_dm _ axisOf (fun x s y s' hf => by
          unfold axisOf
          split at hf
          · obtain ⟨rfl, rfl⟩ := st_pure hf; exact ⟨rfl, rfl⟩
          · rename_i hx
            rcases ho : natJ? x with _ | a
            all_goals rw [ho] at hf; dsimp only at hf
            · exact absurd hf (by intro h; injection h)
            · obtain ⟨rfl, rfl⟩ := st_pure hf
              split
              · exact absurd rfl hx
              · rw [ho]; exact ⟨rfl, rfl⟩) _ _ _ _ h1
        have := array_mapM_same _ (fun x s y s' hf => by
          split at hf
          · exact (st_pure hf).2.symm
          · rcases ho : natJ? x with _ | a
            all_goals rw [ho] at hf; dsimp only at hf
            · exact absurd hf (by intro h; injection h)
            · exact (st_pure hf).2.symm) _ _ _ _ h1
        subst this
        exact ⟨_, by simp only [hb, ↓reduceIte] <;> exact ⟨xs, rfl, hax⟩, h⟩
      · replace h := st_bind h; obtain ⟨axes, s5, h1, h⟩ := h
        obtain ⟨rfl, rfl⟩ := st_pure h1
        exact ⟨_, by simp only [hb, ↓reduceIte, Bool.false_eq_true], h⟩
    clear h
    obtain ⟨axes, hax, h⟩ := hAx
    simp (config := { zeta := false }) only [jpAx] at h
    lift_lets at h
    extract_lets jpN at h
    have hN : ∃ nn, (if F == .batch || F == .scan then
        (nds[nodes1.size]!.getObjVal? "n").toOption.bind natJ? = some nn else nn = 1) ∧
        jpN nn s4 = .ok (t, s2) := by
      replace h := st_ite h; rcases h with ⟨hb, h⟩ | ⟨hb, h⟩
      · rcases hn : (nds[nodes1.size]!.getObjVal? "n").toOption.bind (fun a => natJ? a) with _ | nn
        all_goals rw [hn] at h; dsimp (config := { zeta := false }) only at h
        · exact (st_throw_bind h).elim
        · replace h := st_bind h; obtain ⟨a, s5, h1, h⟩ := h
          obtain ⟨rfl, rfl⟩ := st_pure h1
          exact ⟨_, by simp only [hb, ↓reduceIte] <;> exact hn, h⟩
      · replace h := st_bind h; obtain ⟨a, s5, h1, h⟩ := h
        obtain ⟨rfl, rfl⟩ := st_pure h1
        exact ⟨_, by simp only [hb, ↓reduceIte, Bool.false_eq_true], h⟩
    clear h
    obtain ⟨nn, hnn, h⟩ := hN
    simp (config := { zeta := false }) only [jpN] at h
    lift_lets at h
    extract_lets jpX at h
    have hX : ∃ nxs, (F = .scan → nxs < ajs.size) ∧ (if F == .scan then
        (nds[nodes1.size]!.getObjVal? "n_xs").toOption.bind natJ? = some nxs else nxs = 0) ∧
        jpX nxs s4 = .ok (t, s2) := by
      replace h := st_ite h; rcases h with ⟨hsc, h⟩ | ⟨hsc, h⟩
      · rcases hx : (nds[nodes1.size]!.getObjVal? "n_xs").toOption.bind (fun a => natJ? a) with _ | x
        all_goals rw [hx] at h; dsimp (config := { zeta := false }) only at h
        · exact (st_throw_bind h).elim
        · replace h := st_ite h; rcases h with ⟨hlt, h⟩ | ⟨-, h⟩
          · replace h := st_bind h; obtain ⟨a, s5, h1, h⟩ := h
            obtain ⟨rfl, rfl⟩ := st_pure h1
            exact ⟨_, fun _ => hlt, by simp only [hsc, ↓reduceIte] <;> exact hx, h⟩
          · exact (st_throw_bind h).elim
      · replace h := st_bind h; obtain ⟨a, s5, h1, h⟩ := h
        obtain ⟨rfl, rfl⟩ := st_pure h1
        exact ⟨_, fun e => absurd (by rw [e]; rfl) hsc, by simp only [hsc, ↓reduceIte, Bool.false_eq_true], h⟩
    clear h
    obtain ⟨nxs, hnxs, hnx, h⟩ := hX
    simp (config := { zeta := false }) only [jpX] at h
    replace h := st_bind h; obtain ⟨args, s5, h1, h⟩ := h
    replace h1 := st_liftV h1; obtain ⟨hmap, rfl⟩ := h1
    replace h := st_bind h; obtain ⟨⟨out, args'⟩, s5, h1, h⟩ := h
    replace h1 := st_liftV h1; obtain ⟨hbn, rfl⟩ := h1
    replace h := st_ite h; rcases h with ⟨-, h⟩ | ⟨hout, h⟩
    · exact (st_throw_bind h).elim
    obtain ⟨rfl, rfl⟩ := st_pure h
    clear jpK jpP jpC jpAx jpN jpX
    have hasz : args.size = ajs.size := array_mapM_size _ _ _ hmap
    have hel : ∀ a < ajs.size, (∃ t : Int, intJ? (((ajs[a]!).getObjVal? "type").toOption.getD Json.null) = some t ∧
        0 ≤ t ∧ t < (Desc.types desc).size ∧ (args[a]!).ty = (Desc.types desc)[t.toNat]!) ∧
        ∃ qs, parseParts (((ajs[a]!).getObjVal? "refs").toOption.getD Json.null)
          (leavesOf (stand fid params nodes1 offsets1) outs) earlier = .ok qs ∧
          (args[a]!).seq = Seq.ofParts qs ∧ (args[a]!).dims = none := by
      intro a ha
      have h1 := array_mapM_get _ _ _ hmap a ha (by omega)
      dsimp only at h1
      obtain ⟨ty, hty, h2⟩ := except_bind_ok h1
      obtain ⟨t, ht, h0, hlt, rfl⟩ := at_ok hty
      try dsimp only at h2
      split at h2
      · rename_i ps hps
        obtain ⟨ps', h3, h2⟩ := except_bind_ok h2
        injection h3 with h3; subst h3
        split at h2
        · obtain ⟨u, h3, -⟩ := except_bind_ok h2
          exact absurd h3 (by intro h; injection h)
        injection h2 with h2
        rw [getElem!_pos ajs a ha, getElem!_pos args a (by omega), ← h2]
        exact ⟨⟨t, ht, h0, hlt, rfl⟩, ps, hps, rfl, rfl⟩
      · obtain ⟨u, h3, -⟩ := except_bind_ok h2
        exact absurd h3 (by intro h; injection h)
    dsimp only
    generalize hND : ({ form := F, fn := ci, args := args', out := out, axes := axes, n := nn, nxs := nxs } : Node)
      = node
    generalize ho : offsets1.back! + (if (F == Flock.Program.Form.prim) = true then 1 else node.members * callee.gates) = o
    refine ⟨hs4, pre_trans hp1 hp4, by simp, by simp [hosz1], ?_, fun P hP k hk => ?_⟩
    · rw [push_get_lt _ _ (by omega)]; exact ho01
    have hP1 : Pre (DecState.defs s1) P.defs := pre_trans hp4 hP
    rcases Nat.lt_succ_iff_lt_or_eq.1 hk with hk | rfl
    · obtain ⟨c1, c2, c3⟩ := hn1 P hP1 k hk
      rw [push_get_lt _ _ hk]
      refine ⟨by have := hp4.1; omega, c2, nodeReads_congr (d := stand fid params nodes1 offsets1) rfl rfl (fun k2 hk2 => ?_) ?_ ?_ c3⟩
      · show (nodes1.push node)[k2]! = nodes1[k2]!; exact push_get_lt _ _ (by omega)
      · show (offsets1.push o)[k]! = offsets1[k]!; exact push_get_lt _ _ (by omega)
      · show (offsets1.push o)[k + 1]! = offsets1[k + 1]!; exact push_get_lt _ _ (by omega)
    have hcal : P.defs[ci]! = (DecState.defs s4)[ci]! := hP.2 ci hci
    have hnf : node.fn = ci := by rw [← hND]
    rw [push_get_eq, hnf]
    rw [hdesc.1] at hfn
    refine ⟨hci, by rw [hcal, hkey]; exact calleeAt_of_at hfn, ?_⟩
    have e1 : (stand fid params (nodes1.push node) (offsets1.push o)).body.nodes[nodes1.size]! = node :=
      push_get_eq _ _
    have hE : (fun (k2 x : Nat) => if k2 < nodes1.size then
        ((stand fid params (nodes1.push node) (offsets1.push o)).body.nodes[k2]!).args[x]?.map (fun (a : Arg) => a.seq)
        else none) = earlier := by
      funext k2 x
      show _ = (if k2 < nodes1.size then _ else none)
      split
      · rename_i hk2; show ((nodes1.push node)[k2]!).args[x]?.map _ = _; rw [push_get_lt _ _ hk2]
      · rfl
    unfold NodeReads
    dsimp only
    rw [e1, hE]
    subst hND
    dsimp only
    refine ⟨hF, by simpa using hout, hax, hnn, hnx, ⟨ajs, args, hargs, hasz, fun e => by have := hnxs e; omega,
      fun a ha => hel a ha, by rw [hcal]; exact hbn⟩, ?_⟩
    have hb : offsets1.back! = offsets1[nodes1.size]! := by
      unfold Array.back!; rw [hosz1, Nat.add_sub_cancel]
    show (offsets1.push o)[nodes1.size + 1]! = (offsets1.push o)[nodes1.size]! + _
    rw [← hosz1, push_get_eq, push_get_lt _ _ (by omega), hcal, ← ho, hb]
  simp only [List.length_range', Std.Legacy.Range.size] at hL
  rcases hL with ⟨hs5, hp5, hsz5, hosz5, ho05, hn5⟩ | hL
  rotate_left; · exact hL.elim
  dsimp only at hsz5 hosz5 ho05 hn5 h
  rcases hret : parseParts ((b.getObjVal? "ret").toOption.getD Lean.Json.null) (leavesOf (stand fid params nodes offsets) outs)
    (fun k2 a => nodes[k2]?.bind fun x => Option.map (fun x => x.seq) x.args[a]?) with e | ps
  all_goals rw [show (fun p i => if p = true then Option.map (fun (x : String × Ty) => x.2.leaves) params[i]?
      else Option.map (fun (x : Ty) => x.leaves) outs[i]?) = leavesOf (stand fid params nodes offsets) outs from rfl] at h
  all_goals rw [hret] at h
  all_goals dsimp only at h
  · replace h := st_bind h; obtain ⟨a, s6, h1, -⟩ := h
    exact absurd h1 (by intro h; injection h)
  replace h := st_bind h; obtain ⟨ret, s6, h1, h⟩ := h
  obtain ⟨rfl, rfl⟩ := st_pure h1
  split at h
  · replace h := st_bind h; obtain ⟨a, s7, h1, -⟩ := h
    exact absurd h1 (by intro h; injection h)
  replace h := st_bind h; obtain ⟨st, s7, h1, h⟩ := h
  obtain ⟨rfl, rfl⟩ := st_get h1
  generalize hwalk : EStateM.run _ _ = r at h
  rcases r with ⟨_ | m, w⟩ | ⟨e, w⟩
  all_goals dsimp only at h
  rotate_left
  · replace h := st_bind h; obtain ⟨a, s8, h1, -⟩ := h
    exact absurd h1 (by intro h; injection h)
  · replace h := st_bind h; obtain ⟨a, s8, h1, -⟩ := h
    exact absurd h1 (by intro h; injection h)
  replace h := st_bind h; obtain ⟨u, s8, h1, h⟩ := h
  injection h1 with h1; injection h1 with hu h1; subst h1
  obtain ⟨rfl, rfl⟩ := st_pure h
  simp only [Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one] at hsz5 hosz5 hn5
  refine ⟨rinv_congr rfl rfl hs5, pre_trans hp3 hp5, rfl, fun P hP => ?_⟩
  have hP5 : Pre (DecState.defs s5) P.defs := hP
  refine ⟨rfl, rfl, str_of_field (hform.symm.trans hfc) (by decide), hid, hpa, rfl, hrt, rfl, rfl, rfl, rfl,
    nds, Desc.types desc, outs, hnd, ?_, hdesc.2.1, hou, hsz5, hosz5, ho05, fun k hk => ?_, ps, hret, rfl⟩
  · have hb' : ((entry j fid).getObjValD "body").getObjVal? "nodes" = .ok (.arr nds) := by
      rw [← getObjValD_eq]; exact hnd
    unfold nodesOf
    dsimp only
    rw [getObjValD_of hb']
  · obtain ⟨c1, c2, c3⟩ := hn5 P hP5 k hk
    exact ⟨c1, c2, nodeReads_congr rfl rfl (fun _ _ => rfl) rfl rfl c3⟩

theorem getR_zero (j : Json) : GetR j 0 := by
  intro desc fid s i s' _ h
  unfold getDef at h
  exact absurd h (by intro h; injection h)

theorem compR_zero (j : Json) : CompR j 0 := by
  intro desc fid d form id s defn s' _ _ _ _ h
  unfold composite at h
  exact absurd h (by intro h; injection h)

theorem statR_zero (j : Json) : StatR j 0 := by
  intro desc v s u s' _ h
  unfold staticFns at h
  exact absurd h (by intro h; injection h)

theorem specsR (j : Json) : ∀ n, GetR j n ∧ CompR j n ∧ StatR j n
  | 0 => ⟨getR_zero j, compR_zero j, statR_zero j⟩
  | n + 1 =>
    let ⟨g, c, st⟩ := specsR j n
    ⟨getDef_succR j n c, composite_succR j n g st, statFns_succR j n g st⟩

/-- `decode`'s `go`, the root and then the functions the program record names, keeps `RInv`. -/
theorem go_reads (j : Json) (desc : Desc) (hdesc : DescOf j desc) (rootId programFn : String) (root : Nat)
    (s0 st : DecState)
    (h : StateT.run (do
          let r ← getDef Flock.Steps.FUEL desc rootId
          forIn (recordFunctionIds programFn) PUnit.unit fun f __s =>
              if (Desc.ddefs desc).contains f = true then do
                discard (getDef Flock.Steps.FUEL desc f)
                pure (ForInStep.yield PUnit.unit)
              else pure (ForInStep.yield PUnit.unit)
          pure r : DM Nat) s0 = .ok (root, st)) (hs0 : RInv j s0) : RInv j st := by
  obtain ⟨r, s1, h1, h⟩ := st_bind h
  obtain ⟨hs1, -⟩ := (specsR j _).1 _ _ _ _ _ hdesc h1 hs0
  obtain ⟨u, s2, h2, h⟩ := st_bind h
  obtain ⟨hr, rfl⟩ := st_pure h
  try rw [← Array.forIn_toList] at h2
  have hg := forIn_rgrows _ _ (fun x b s t s' hf => ?_) _ _ _ _ h2 hs1
  · exact hg.1
  split at hf
  · obtain ⟨u, s3, h3, h4⟩ := st_bind hf
    obtain ⟨-, rfl⟩ := st_pure h4
    obtain ⟨a, ha⟩ := st_discard h3
    intro hs
    obtain ⟨i1, p1, -⟩ := (specsR j _).1 _ _ _ _ _ hdesc ha hs
    exact ⟨i1, p1⟩
  · obtain ⟨-, rfl⟩ := st_pure hf
    exact rgrows_refl _ _

/-- A key the folded map holds is one of the list's, with its value. -/
theorem fold_insert_get : ∀ (l : List (String × Json)) (m : Std.HashMap String Json) (k : String) (v : Json),
    (l.foldl (fun m (p : String × Json) => m.insert p.1 p.2) m)[k]? = some v → (k, v) ∈ l ∨ m[k]? = some v
  | [], m, k, v, h => Or.inr h
  | p :: l, m, k, v, h => by
    rcases fold_insert_get l _ k v h with h | h
    · exact Or.inl (List.mem_cons_of_mem _ h)
    · rw [Std.HashMap.getElem?_insert] at h
      split at h
      · rename_i he
        injection h with h
        subst h
        exact Or.inl (by rw [← eq_of_beq he]; exact List.mem_cons_self)
      · exact Or.inr h

theorem get_of_getObjVal {m : Std.TreeMap.Raw String Json compare} {k : String} {v : Json}
    (h : (Json.obj m).getObjVal? k = .ok v) : m.get? k = some v := by
  simp only [Json.getObjVal?] at h
  split at h
  · rename_i v' hv; injection h with h; subst h; exact hv
  · cases h

theorem getObjVal_of_get {m : Std.TreeMap.Raw String Json compare} {k : String} {v : Json}
    (h : m.get? k = some v) : (Json.obj m).getObjVal? k = .ok v := by
  simp only [Json.getObjVal?, h]; rfl

theorem ex_throw_bind {α β : Type} {e : String} {f : α → Flock.V β} {b : β}
    (h : (throw e >>= f : Flock.V β) = .ok b) : False := by
  obtain ⟨a, h1, -⟩ := except_bind_ok h
  exact absurd h1 (by intro h; injection h)

/-- **What `decode` returns, read from the descriptor's text**: the Definitions and the index of a decoder state that
keeps `RInv` for the JSON the text parses to. -/
theorem decode_reads (text : String) (j0 : Json) (P : Prog) (h : decode text.toUTF8 = .ok P)
    (hp : Flock.JsonParse.parse text = .ok j0) :
    ∃ st, P.defs = DecState.defs st ∧ P.index = DecState.index st ∧ RInv j0 st := by
  have hwf := FlockComplete.JsonRT.parse_wf text j0 hp
  unfold decode at h
  extract_lets k1 at h
  split at h
  rotate_left; · exact absurd h (by intro h; injection h)
  rename_i text' htext
  rw [FlockComplete.fromUTF8?_toUTF8] at htext
  injection htext with htext
  subst htext
  split at h
  rotate_left; · exact (ex_throw_bind h).elim
  rename_i j' hj'
  rw [hp] at hj'
  injection hj' with hj'
  subst hj'
  obtain ⟨j, h1, h⟩ := except_bind_ok h
  injection h1 with h1; subst h1
  simp (config := { zeta := false }) only [k1] at h
  split at h
  rotate_left; · exact absurd h (by intro h; injection h)
  rename_i kvPairs _
  extract_lets bare jpS at h
  split at h
  rotate_left; · exact (ex_throw_bind h).elim
  split at h
  · exact (ex_throw_bind h).elim
  simp (config := { zeta := false }) only [jpS] at h
  split at h
  rotate_left; · exact absurd h (by intro h; injection h)
  rename_i dkv hdkv
  obtain ⟨u, h1, h⟩ := except_bind_ok h
  clear h1
  extract_lets freeDefs free jpD at h
  split at h
  · exact (ex_throw_bind h).elim
  simp (config := { zeta := false }) only [jpD] at h
  split at h
  rotate_left; · exact absurd h (by intro h; injection h)
  rename_i callees hcal
  split at h
  rotate_left; · exact absurd h (by intro h; injection h)
  rename_i tys htys
  obtain ⟨types, htypes, h⟩ := except_bind_ok h
  extract_lets desc at h
  split at h
  rotate_left; · exact absurd h (by intro h; injection h)
  split at h
  rotate_left; · exact absurd h (by intro h; injection h)
  extract_lets go at h
  obtain ⟨⟨root, st⟩, hgo, h⟩ := except_bind_ok h
  extract_lets rd jpR at h
  split at h
  · exact (ex_throw_bind h).elim
  simp (config := { zeta := false }) only [jpR] at h
  obtain ⟨u, -, h⟩ := except_bind_ok h
  obtain ⟨c, -, h⟩ := except_bind_ok h
  injection h with h
  subst h
  have hdesc : DescOf (Json.obj kvPairs) desc := by
    refine ⟨?_, ?_, fun fid d hd => ?_⟩
    · show callees = tab _ "callees"
      unfold tab; rw [getObjValD_of hcal]
    · show _ = Except.ok types
      unfold tab; rw [getObjValD_of htys]; exact htypes
    · have hd' : (dkv.toArray.toList.foldl (fun (m : Std.HashMap String Json) (p : String × Json) => m.insert p.1 p.2) ∅)[fid]? = some d := by
        rw [Array.foldl_toList]; exact hd
      rcases fold_insert_get _ _ _ _ hd' with hm | hm
      · have hw : dkv.WF := by
          cases hwf with
          | obj hm0 hv => cases hv _ _ (get_of_getObjVal hdkv) with
            | obj hm1 _ => exact hm1
        have hg : dkv.get? fid = some d := by
          rw [Std.TreeMap.Raw.get?_eq_getElem?]
          exact (Std.TreeMap.Raw.mem_toArray_iff_getElem?_eq_some hw).1 (Array.mem_def.2 hm)
        unfold entry
        rw [getObjValD_of hdkv, getObjValD_of (getObjVal_of_get hg)]
      · exact absurd hm (by simp)
  exact ⟨st, rfl, rfl, go_reads _ desc hdesc _ _ _ _ _ hgo
    ⟨fun k v hk => absurd hk (by simp), fun _ _ i hi => absurd hi (Nat.not_lt_zero i)⟩⟩

end DecodeReadsProof

open Lean (Json)
open Flock.Program
open InferenceOnlyVerifier.Properties.SecurityProofs.BodiesOk (Pre pre_refl)
open private DecState DecState.defs from Circuits.Program

open DecodeReadsProof in
/-- **The decoder builds each Definition from its entry and its callees** (`DecodeReads`): the index names each
Definition by its key (`decode_keys`), and each Definition is what `PrimReads` or `CompReads` says, the invariant the
decoder keeps (`decode_reads`). -/
theorem decodeReads : DecodeReads := by
  intro text j P hdec hparse
  obtain ⟨hk1, hk2⟩ := InferenceOnlyVerifier.Properties.SecurityProofs.DecodeKeys.decode_keys _ P hdec
  obtain ⟨st, hd, -, hr⟩ := decode_reads text j P hdec hparse
  refine ⟨hk1, fun i hi => ⟨hk2 i hi, ?_⟩⟩
  have hP : Pre (DecState.defs st) P.defs := by rw [hd]; exact pre_refl _
  have := hr.2 P hP i (by rw [← hd]; exact hi)
  rw [← hd] at this
  exact this

/-- **Equivalent canonical entries pin the decoded Definition** (`EntriesPin`): from what the decoder reads of each entry
(`decodeReads`) and that equivalent entries read so give one Definition (`readsPin`). -/
theorem entriesPin : EntriesPin := entriesPin_of_reads decodeReads readsPin

/-- **`DigestPins`**, from `entriesPin`. -/
theorem digestPins : DigestPins := digestPins_of_entriesPin entriesPin

end InferenceOnlyVerifier.Properties.Record.SecurityProofs

end
