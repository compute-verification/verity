import Definitions.Claims
import Flock.With.Log

/-!
# The verifier over `OC`: every SHA-512 call an oracle query

The claims' oracle programs `OC` form a lawful monad (`OC.bind`), and `OCV := ExceptT String OC` is a verdict monad the
hash-parametric verifier runs in: `Hashes.oc` answers each SHA-512 call with `OC.query`, and SHA-256 and BLAKE3 as
shipped (`Hashes.ofSha512`'s). Evaluated against `H : Msg → Msg`, a run of `verifyWithM Hashes.oc` has
`verifyWith (hOf H)`'s verdict, and its oracle log is the SHA-512 inputs `verifyLogged (hOf H)` logs, `shaCalls (hOf H)`
of them (`verifyOC_eval`, `verifyOC_log`, `verifyOC_log_length`, and their `Zk` copies). All of it is one simulation of
`Flock.With.Sim` (`ocLog`) between `OCV` and the logging monad `LogV`, at #2105's `verifyWithM_sim`.

`hOf H` is `H` on byte arrays: `ByteArray` and `Msg = List UInt8` are the same bytes (`ofMsg_toMsg`).
-/

namespace Claims

namespace OC

/-- `OC`'s bind: run `x`, then `f` on its result, every query in order. -/
def bind {α β : Type} : OC α → (α → OC β) → OC β
  | .ret a, f => f a
  | .query q k, f => .query q fun r => bind (k r) f

instance : Monad OC where
  pure := .ret
  bind := bind

@[simp] theorem pure_eq {α : Type} (a : α) : (pure a : OC α) = .ret a := rfl

@[simp] theorem bind_eq {α β : Type} (x : OC α) (f : α → OC β) : x >>= f = x.bind f := rfl

@[simp] theorem bind_ret {α β : Type} (a : α) (f : α → OC β) : (OC.ret a).bind f = f a := rfl

@[simp] theorem bind_query {α β : Type} (q : Msg) (k : Msg → OC α) (f : α → OC β) :
    (OC.query q k).bind f = .query q fun r => (k r).bind f := rfl

theorem bind_ret_right {α : Type} (x : OC α) : x.bind OC.ret = x := by
  induction x with
  | ret a => rfl
  | query q k ih => simp only [bind_query, ih]

theorem bind_assoc' {α β γ : Type} (x : OC α) (f : α → OC β) (g : β → OC γ) :
    (x.bind f).bind g = x.bind fun a => (f a).bind g := by
  induction x with
  | ret a => rfl
  | query q k ih => simp only [bind_query, ih]

instance : LawfulMonad OC := LawfulMonad.mk' OC
  (id_map := fun x => bind_ret_right x)
  (pure_bind := fun _ _ => rfl)
  (bind_assoc := fun x f g => bind_assoc' x f g)

/-- A bind's run against `H`: the second program's result, the two logs in order. -/
theorem eval_bind (H : Msg → Msg) {α β : Type} (x : OC α) (f : α → OC β) :
    (x.bind f).eval H = (((f (x.eval H).1).eval H).1, (x.eval H).2 ++ ((f (x.eval H).1).eval H).2) := by
  induction x with
  | ret a => simp [bind, OC.eval]
  | query q k ih => simp [bind, OC.eval, ih]

end OC

/-- The verdict monad over oracle programs: an error is a rejection, and a rejecting run keeps its queries. -/
abbrev OCV := ExceptT String OC

/-- A byte array as a message: its bytes. -/
def toMsg (b : ByteArray) : Msg := b.data.toList

/-- A message as a byte array. -/
def ofMsg (a : Msg) : ByteArray := ⟨a.toArray⟩

@[simp] theorem ofMsg_toMsg (b : ByteArray) : ofMsg (toMsg b) = b := by
  cases b; simp [ofMsg, toMsg]

/-- `H` on byte arrays. -/
def hOf (H : Msg → Msg) (b : ByteArray) : ByteArray := ofMsg (H (toMsg b))

/-- SHA-512 as the oracle `OC.query`; SHA-256 and BLAKE3 the deployed functions, as in `Hashes.ofSha512`. -/
def Hashes.oc : _root_.Flock.Hashes OCV :=
  { _root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash with
    sha512 := fun b => ExceptT.lift (OC.query (toMsg b) fun a => OC.ret (ofMsg a)) }

open _root_.Flock

/-- An `OCV` run against `H` and a `LogV` run from the log `l`: the same verdict, and the log is `l` with the run's
queries appended. -/
def ocLog (H : Msg → Msg) : With.Sim OCV LogV where
  R x y := ∀ l : Array ByteArray, (y.go l).1 = ((x.run).eval H).1 ∧
    (y.go l).2.toList = l.toList ++ ((x.run).eval H).2.map fun p => ofMsg p.1
  pure _ _ := ⟨rfl, by simp [OC.eval]; rfl⟩
  bind {α β x y f g} hxy hfg l := by
    obtain ⟨h1, h2⟩ := hxy l
    simp only [StV.go, Id.run, ExceptT.run_bind, StateT.run_bind] at h1 h2 ⊢
    rw [OC.bind_eq, OC.eval_bind]
    rcases hr : y.run.run l with ⟨r, l'⟩
    rcases hxr : x.run.eval H with ⟨rx, lx⟩
    rw [hr, hxr] at h1 h2
    simp only at h1 h2
    subst h1
    cases r with
    | error e => exact ⟨rfl, h2.trans (congrArg (l.toList ++ List.map _ ·) (List.append_nil lx).symm)⟩
    | ok a =>
      obtain ⟨h1', h2'⟩ := hfg a l'
      simp only [StV.go, Id.run] at h1' h2'
      refine ⟨h1', h2'.trans ?_⟩
      rw [h2]
      show _ = l.toList ++ List.map (fun p => ofMsg p.1) (lx ++ (OC.eval H (ExceptT.run (f a))).2)
      simp
  throw _ _ := ⟨rfl, by simp [OC.eval]; rfl⟩
  lift _ l := ⟨rfl, by show l.toList = l.toList ++ List.map _ (OC.eval H (OC.ret _)).2; simp [OC.eval]⟩

theorem ocLog_hashes (H : Msg → Msg) : (ocLog H).Hashes Hashes.oc (Hashes.logged (hOf H)) where
  sha512 b l := ⟨rfl, by
    show (l.push b).toList =
      l.toList ++ List.map _ (OC.eval H (OC.query (toMsg b) fun a => OC.ret (Except.ok (ofMsg a)))).2
    simp [OC.eval]⟩
  sha256 b l := ⟨rfl, by show l.toList = l.toList ++ List.map _ (OC.eval H (OC.ret _)).2; simp [OC.eval]⟩
  blake3 b l := ⟨rfl, by show l.toList = l.toList ++ List.map _ (OC.eval H (OC.ret _)).2; simp [OC.eval]⟩
  blake3Keyed k b l := ⟨rfl, by show l.toList = l.toList ++ List.map _ (OC.eval H (OC.ret _)).2; simp [OC.eval]⟩

/-- **The verifier over `OC`, against `H`, is `verifyWith (hOf H)`.** -/
theorem verifyOC_eval (H : Msg → Msg) (spec : SessionSpec) (tabs : Array Setup) (record : String)
    (proofs : Array ByteArray) :
    ((verifyWithM Hashes.oc spec tabs record proofs).run.eval H).1 = verifyWith (hOf H) spec tabs record proofs := by
  rw [← (With.verifyWithM_sim (ocLog H) (ocLog_hashes H) spec tabs record proofs #[]).1]
  exact verifyLogged_fst _ _ _ _ _ _

/-- **Its oracle log is `verifyLogged`'s log**: the SHA-512 inputs, in call order. -/
theorem verifyOC_log (H : Msg → Msg) (spec : SessionSpec) (tabs : Array Setup) (record : String)
    (proofs : Array ByteArray) :
    ((verifyWithM Hashes.oc spec tabs record proofs).run.eval H).2.map (fun p => ofMsg p.1) =
      (verifyLogged (hOf H) spec tabs record proofs).2.toList := by
  have := (With.verifyWithM_sim (ocLog H) (ocLog_hashes H) spec tabs record proofs #[]).2
  simp only [verifyLogged] at this ⊢
  rw [this]
  simp

/-- **It makes `shaCalls (hOf H)` queries.** -/
theorem verifyOC_log_length (H : Msg → Msg) (spec : SessionSpec) (tabs : Array Setup) (record : String)
    (proofs : Array ByteArray) :
    ((verifyWithM Hashes.oc spec tabs record proofs).run.eval H).2.length =
      shaCalls (hOf H) spec tabs record proofs := by
  rw [← verifyLogged_size, ← Array.length_toList, ← verifyOC_log, List.length_map]

namespace Zk

/-- **The `--zk` verifier over `OC`, against `H`, is `Zk.verifyWith (hOf H)`.** -/
theorem verifyOC_eval (H : Msg → Msg) (spec : SessionSpec) (tabs : Array Setup) (record : String)
    (proofs : Array ByteArray) :
    ((_root_.Flock.Zk.verifyWithM Hashes.oc spec tabs record proofs).run.eval H).1 =
      _root_.Flock.Zk.verifyWith (hOf H) spec tabs record proofs := by
  rw [← (With.zk_verifyWithM_sim (ocLog H) (ocLog_hashes H) spec tabs record proofs #[]).1]
  exact _root_.Flock.Zk.verifyLogged_fst _ _ _ _ _ _

theorem verifyOC_log (H : Msg → Msg) (spec : SessionSpec) (tabs : Array Setup) (record : String)
    (proofs : Array ByteArray) :
    ((_root_.Flock.Zk.verifyWithM Hashes.oc spec tabs record proofs).run.eval H).2.map (fun p => ofMsg p.1) =
      (_root_.Flock.Zk.verifyLogged (hOf H) spec tabs record proofs).2.toList := by
  have := (With.zk_verifyWithM_sim (ocLog H) (ocLog_hashes H) spec tabs record proofs #[]).2
  simp only [_root_.Flock.Zk.verifyLogged] at this ⊢
  rw [this]
  simp

/-- **It makes `Zk.shaCalls (hOf H)` queries.** -/
theorem verifyOC_log_length (H : Msg → Msg) (spec : SessionSpec) (tabs : Array Setup) (record : String)
    (proofs : Array ByteArray) :
    ((_root_.Flock.Zk.verifyWithM Hashes.oc spec tabs record proofs).run.eval H).2.length =
      _root_.Flock.Zk.shaCalls (hOf H) spec tabs record proofs := by
  rw [← _root_.Flock.Zk.verifyLogged_size, ← Array.length_toList, ← verifyOC_log, List.length_map]

end Zk

end Claims
