module

public import Batteries.Data.ByteArray
public import Definitions.Core.Interaction
public import Flock.With.SimSetup
public import Mathlib.Control.Lawful
public import Mathlib.NumberTheory.LucasLehmer
import Flock.With.SimVerify
import Flock.With.SimZk
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
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
import Mathlib.Topology.Sheaves.Init

/-! The verifier over `OC`, every SHA-512 call an oracle query, has the shipped verdict. -/

@[expose] public section

namespace Claims

namespace OC

instance instMonad : Monad OC where
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

instance instLawfulMonad : LawfulMonad OC := LawfulMonad.mk' OC
  (id_map := fun x => bind_ret_right x)
  (pure_bind := fun _ _ => rfl)
  (bind_assoc := fun x f g => bind_assoc' x f g)

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

theorem verifyOC_eval (H : Msg → Msg) (spec : SessionSpec) (tabs : Array Setup) (record : String)
    (proofs : Array ByteArray) :
    ((verifyWithM Hashes.oc spec tabs record proofs).run.eval H).1 = verifyWith (hOf H) spec tabs record proofs := by
  rw [← (With.verifyWithM_sim (ocLog H) (ocLog_hashes H) spec tabs record proofs #[]).1]
  exact verifyLogged_fst _ _ _ _ _ _

theorem verifyOC_log (H : Msg → Msg) (spec : SessionSpec) (tabs : Array Setup) (record : String)
    (proofs : Array ByteArray) :
    ((verifyWithM Hashes.oc spec tabs record proofs).run.eval H).2.map (fun p => ofMsg p.1) =
      (verifyLogged (hOf H) spec tabs record proofs).2.toList := by
  have := (With.verifyWithM_sim (ocLog H) (ocLog_hashes H) spec tabs record proofs #[]).2
  simp only [verifyLogged] at this ⊢
  rw [this]
  simp

theorem verifyOC_log_length (H : Msg → Msg) (spec : SessionSpec) (tabs : Array Setup) (record : String)
    (proofs : Array ByteArray) :
    ((verifyWithM Hashes.oc spec tabs record proofs).run.eval H).2.length =
      shaCalls (hOf H) spec tabs record proofs := by
  rw [← verifyLogged_size, ← Array.length_toList, ← verifyOC_log, List.length_map]

namespace Zk

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

theorem verifyOC_log_length (H : Msg → Msg) (spec : SessionSpec) (tabs : Array Setup) (record : String)
    (proofs : Array ByteArray) :
    ((_root_.Flock.Zk.verifyWithM Hashes.oc spec tabs record proofs).run.eval H).2.length =
      _root_.Flock.Zk.shaCalls (hOf H) spec tabs record proofs := by
  rw [← _root_.Flock.Zk.verifyLogged_size, ← Array.length_toList, ← verifyOC_log, List.length_map]

end Zk

theorem setupOC_eval (H : Msg → Msg) (tags : Tags) (circuitFile publicFile : ByteArray) (coins : Coins) (count : ℕ)
    (tables : List (String × ByteArray)) (draw : Option Lean.Json) (partition : Option Lean.Json)
    (program : Option ByteArray) (work : Option (ℕ × Array (String × ℕ × ℕ))) (stratified : Option ℕ)
    (closure : Option (Std.HashMap ℕ (Array ℕ))) (publicInputs : Option ByteArray) :
    ((Stmt.setupTablesWith Hashes.oc tags circuitFile publicFile coins count tables draw partition program work stratified
        closure publicInputs).run.eval H).1 =
      Stmt.setupTablesWith (M := _root_.Flock.V) (_root_.Flock.Hashes.ofSha512 (hOf H)) tags circuitFile publicFile coins
        count tables draw partition program work stratified closure publicInputs := by
  rw [← (With.setupTablesWith_sim (ocLog H) With.Throws.exceptT With.Throws.exceptT (ocLog_hashes H) tags circuitFile
    publicFile coins count tables draw partition program work stratified closure publicInputs #[]).1]
  exact Stmt.setupLogged_fst _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

theorem setupOC_log (H : Msg → Msg) (tags : Tags) (circuitFile publicFile : ByteArray) (coins : Coins) (count : ℕ)
    (tables : List (String × ByteArray)) (draw : Option Lean.Json) (partition : Option Lean.Json)
    (program : Option ByteArray) (work : Option (ℕ × Array (String × ℕ × ℕ))) (stratified : Option ℕ)
    (closure : Option (Std.HashMap ℕ (Array ℕ))) (publicInputs : Option ByteArray) :
    ((Stmt.setupTablesWith Hashes.oc tags circuitFile publicFile coins count tables draw partition program work stratified
        closure publicInputs).run.eval H).2.map (fun p => ofMsg p.1) =
      (Stmt.setupLogged (hOf H) tags circuitFile publicFile coins count tables draw partition program work stratified
        closure publicInputs).2.toList := by
  have := (With.setupTablesWith_sim (ocLog H) With.Throws.exceptT With.Throws.exceptT (ocLog_hashes H) tags circuitFile
    publicFile coins count tables draw partition program work stratified closure publicInputs #[]).2
  simp only [Stmt.setupLogged] at this ⊢
  rw [this]
  simp

theorem setupOC_log_length (H : Msg → Msg) (tags : Tags) (circuitFile publicFile : ByteArray) (coins : Coins)
    (count : ℕ) (tables : List (String × ByteArray)) (draw : Option Lean.Json) (partition : Option Lean.Json)
    (program : Option ByteArray) (work : Option (ℕ × Array (String × ℕ × ℕ))) (stratified : Option ℕ)
    (closure : Option (Std.HashMap ℕ (Array ℕ))) (publicInputs : Option ByteArray) :
    ((Stmt.setupTablesWith Hashes.oc tags circuitFile publicFile coins count tables draw partition program work stratified
        closure publicInputs).run.eval H).2.length =
      Stmt.setupCalls (hOf H) tags circuitFile publicFile coins count tables draw partition program work stratified
        closure publicInputs := by
  rw [← Stmt.setupLogged_size, ← Array.length_toList, ← setupOC_log, List.length_map]

end Claims
