module

public import Proofs.Flock.Claims.Complete.Encode
public import Proofs.Flock.Claims.KS.PlainStep
public import Proofs.Flock.Claims.Recursion.BoundDeploy
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
import Mathlib.NumberTheory.LucasLehmer
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
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init

/-! (P2) and (P4) of `KSBoundRouteCols` at the intended choice; `JsonTrailingSpace` is a hypothesis because
`Json.parse`'s `anyCore` is `partial`. -/

@[expose] public section

namespace RecDeploy

open Claims (Msg Out OC)
open Lean (Json)

/-- **The deployed rows' digest input** (`hm96-sha512/row/v1`): the first block for the row's u16 words
(`Flock.HmRow.rowPrefix`), then the row. -/
def rowInDeployed (row : Msg) : Msg := (_root_.Flock.HmRow.rowPrefix (row.length / 2)).data.toList ++ row

theorem rowInDeployed_injective : Function.Injective rowInDeployed := by
  intro r r' h
  unfold rowInDeployed at h
  have hl : ((_root_.Flock.HmRow.rowPrefix (r.length / 2)).data.toList).length =
      ((_root_.Flock.HmRow.rowPrefix (r'.length / 2)).data.toList).length := by
    rw [FlockSoundness.Discharge.Hm.length_rowPrefix, FlockSoundness.Discharge.Hm.length_rowPrefix]
  exact (List.append_inj h hl).2

theorem binds_deployed (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (nextE : Msg → List Msg → List FlockDeploy.Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) :
    (bdDeployed rel nextE colsE wE join rowInDeployed final w).Binds :=
  rowInDeployed_injective

/-- Lean's JSON parser accepts trailing whitespace. A hypothesis because `anyCore` is `partial`, so the kernel doesn't
see its body. -/
def JsonTrailingSpace : Prop :=
  ∀ (r : String) (j : Json), Flock.JsonParse.parse r = .ok j → Flock.JsonParse.parse (r ++ " ") = .ok j

end RecDeploy

namespace Claims.Recursion.LastGap

open Claims

theorem eval_bind (H : Msg → Msg) {α β : Type} (p : OC α) (f : α → OC β) :
    (bindOC p f).eval H = (((f (p.eval H).1).eval H).1, (p.eval H).2 ++ ((f (p.eval H).1).eval H).2) := by
  induction p with
  | ret a => simp [bindOC, OC.eval]
  | query q k ih => simp [bindOC, OC.eval, ih]

theorem eval_allOC_cons (H : Msg → Msg) (p : OC Bool) (ps : List (OC Bool)) (hp : (p.eval H).1 = true) :
    (allOC (p :: ps)).eval H = (((allOC ps).eval H).1, (p.eval H).2 ++ ((allOC ps).eval H).2) := by
  show (bindOC p fun b => if b then allOC ps else OC.ret false).eval H = _
  rw [eval_bind, hp]
  rfl

theorem allOC_true (H : Msg → Msg) : ∀ ps : List (OC Bool), ((allOC ps).eval H).1 = true →
    ∀ p ∈ ps, (p.eval H).1 = true ∧ ∀ e ∈ (p.eval H).2, e ∈ ((allOC ps).eval H).2
  | [], _, _, hp => absurd hp List.not_mem_nil
  | q :: qs, h, p, hp => by
    cases hq : (q.eval H).1 with
    | false =>
      have hf : ((allOC (q :: qs)).eval H).1 = false := by
        show ((bindOC q fun b => if b then allOC qs else OC.ret false).eval H).1 = false
        rw [eval_bind, hq]
        rfl
      rw [hf] at h
      cases h
    | true =>
      rw [eval_allOC_cons H q qs hq] at h ⊢
      rcases List.mem_cons.1 hp with rfl | hp
      · exact ⟨hq, fun e he => List.mem_append_left _ he⟩
      · obtain ⟨h1, h2⟩ := allOC_true H qs h p hp
        exact ⟨h1, fun e he => List.mem_append_right _ (h2 e he)⟩

variable {Coin Col T : Type} [Inhabited Coin]

theorem ivstar_acc (Bd : Bound Coin Col T) (H : Msg → Msg) {x : Msg} {as : List Msg} {cs : List Coin}
    (hk : as.length = Bd.DE.V.k x) (h : ((Bd.IVStar.next x as cs).eval H).1 = .acc) :
    ∃ os last, Bd.final (as.getLastD []) = some (os, last) ∧ os.length + 1 = as.length ∧
      ((Bd.agrees x (Bd.opened os last) cs).eval H).1 = true := by
  have hlt : ¬ as.length < Bd.DE.V.k x := by omega
  simp only [Bound.IVStar, hlt, ↓reduceIte] at h
  split at h
  · simp [OC.eval] at h
  · rename_i os last hfin
    split at h
    · rename_i hc
      rw [eval_bind] at h
      cases hok : ((allOC ((as.dropLast.zip os).map fun p => Bd.opensRound p.1 p.2)).eval H).1
      · rw [hok] at h
        simp [OC.eval] at h
      · rw [hok] at h
        simp only [↓reduceIte] at h
        rw [eval_bind] at h
        cases hacc : ((Bd.agrees x (Bd.opened os last) cs).eval H).1
        · rw [hacc] at h
          simp [OC.eval] at h
        · exact ⟨os, last, hfin, hc.1, hacc⟩
    · simp [OC.eval] at h

theorem agrees_last (Bd : Bound Coin Col T) (H : Msg → Msg) {x : Msg} {ms₀ : List Msg} {l : Msg}
    {cs : List Coin} (h : ((Bd.agrees x (ms₀ ++ [l]) cs).eval H).1 = true) :
    ((Bd.DE.V.next x (ms₀ ++ [l]) (cs.take (ms₀.length + 1))).eval H).1 = .acc := by
  unfold Bound.agrees at h
  have hm := (allOC_true H _ h _ (List.mem_map.2 ⟨ms₀.length, List.mem_range.2 (by simp), rfl⟩)).1
  rw [eval_bind] at hm
  have e : (ms₀ ++ [l]).take (ms₀.length + 1) = ms₀ ++ [l] := List.take_of_length_le (by simp)
  rw [e] at hm
  simpa [OC.eval] using hm

theorem agrees_snoc (Bd : Bound Coin Col T) {x : Msg} {ms₀ : List Msg} {l l' : Msg} {cs : List Coin}
    (h : Bd.DE.V.next x (ms₀ ++ [l']) (cs.take (ms₀.length + 1)) =
      Bd.DE.V.next x (ms₀ ++ [l]) (cs.take (ms₀.length + 1))) :
    Bd.agrees x (ms₀ ++ [l']) cs = Bd.agrees x (ms₀ ++ [l]) cs := by
  unfold Bound.agrees
  simp only [List.length_append, List.length_singleton]
  congr 1
  apply List.map_congr_left
  intro j hj
  rw [List.mem_range] at hj
  by_cases hjl : j < ms₀.length
  · have e1 : (ms₀ ++ [l']).take (j + 1) = ms₀.take (j + 1) := List.take_append_of_le_length (by omega)
    have e2 : (ms₀ ++ [l]).take (j + 1) = ms₀.take (j + 1) := List.take_append_of_le_length (by omega)
    rw [e1, e2]
  · have hj' : j = ms₀.length := by omega
    subst hj'
    have e1 : (ms₀ ++ [l']).take (ms₀.length + 1) = ms₀ ++ [l'] := List.take_of_length_le (by simp)
    have e2 : (ms₀ ++ [l]).take (ms₀.length + 1) = ms₀ ++ [l] := List.take_of_length_le (by simp)
    rw [e1, e2, h]

theorem ivstar_congr_snoc (Bd : Bound Coin Col T) {x : Msg} {as as' : List Msg} {cs : List Coin}
    {os : List (List (Msg × (Fin Bd.saltLen → UInt8)))} {last last' : Msg}
    (hlen : as'.length = as.length) (hdrop : as'.dropLast = as.dropLast)
    (hfin : Bd.final (as.getLastD []) = some (os, last)) (hfin' : Bd.final (as'.getLastD []) = some (os, last'))
    (hag : Bd.agrees x (Bd.opened os last') cs = Bd.agrees x (Bd.opened os last) cs) :
    Bd.IVStar.next x as' cs = Bd.IVStar.next x as cs := by
  simp only [Bound.IVStar, hlen, hfin, hfin', hdrop, hag]

end Claims.Recursion.LastGap

namespace RecDeploy.LastGap

open Claims (Msg Out OC)
open Lean (Json)

theorem recordSample_congr {r r' : String} (h : Flock.JsonParse.parse r' = Flock.JsonParse.parse r) :
    FlockSoundness.Discharge.Exec.recordDraw r' = FlockSoundness.Discharge.Exec.recordDraw r := by
  unfold FlockSoundness.Discharge.Exec.recordDraw
  rw [h]

theorem liveOf_congr (spec : _root_.Flock.SessionSpec) {r r' : String} (h : Flock.JsonParse.parse r' = Flock.JsonParse.parse r) :
    FlockDeploy.liveOf spec r' = FlockDeploy.liveOf spec r := by
  unfold FlockDeploy.liveOf
  rw [h]

theorem decodeWith_congr {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT _root_.Flock.V M]
    (hs : _root_.Flock.Hashes M) (spec : _root_.Flock.SessionSpec) {r r' : String}
    (h : Flock.JsonParse.parse r' = Flock.JsonParse.parse r) :
    _root_.Flock.Record.decodeWith hs spec r' = _root_.Flock.Record.decodeWith hs spec r := by
  unfold _root_.Flock.Record.decodeWith
  rw [h]

theorem verifyWithM_congr {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT _root_.Flock.V M]
    (hs : _root_.Flock.Hashes M) (spec : _root_.Flock.SessionSpec) (tabs : Array _root_.Flock.Setup)
    (ps : Array ByteArray) {r r' : String} (h : Flock.JsonParse.parse r' = Flock.JsonParse.parse r) :
    _root_.Flock.verifyWithM hs spec tabs r' ps = _root_.Flock.verifyWithM hs spec tabs r ps := by
  unfold _root_.Flock.verifyWithM
  rw [decodeWith_congr hs spec h]

theorem append_space_ne (r : String) : r ++ " " ≠ r := by
  intro h
  have hl := congrArg String.length h
  rw [String.length_append] at hl
  have h1 : (" " : String).length = 1 := by decide
  omega

theorem checks_record_eq {zk : Bool} {s : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec} {x : Msg}
    {link : List ByteArray × List ByteArray} {c₁ : FlockDeploy.Coin} {mids : List (Msg × FlockDeploy.Coin)}
    {st : Msg} {ps : Array ByteArray} {r r' : String} (h : Flock.JsonParse.parse r' = Flock.JsonParse.parse r)
    (hc : FlockDeploy.checks zk s spec x link c₁ mids ⟨st, r, ps⟩ = .ok ())
    (hc' : FlockDeploy.checks zk s spec x link c₁ mids ⟨st, r', ps⟩ = .ok ()) : r' = r :=
  _root_.Flock.RecordText.check_inj (Claims.Bridge.Custody.checks_canonical hc')
    (Claims.Bridge.Custody.checks_canonical hc) h

theorem check_space (hws : JsonTrailingSpace) {r : String} (hc : _root_.Flock.RecordText.check r = .ok ()) :
    _root_.Flock.RecordText.check (r ++ " ") ≠ .ok () := by
  intro h'
  obtain ⟨j, hj, -⟩ := _root_.Flock.RecordText.check_ok hc
  exact append_space_ne r (_root_.Flock.RecordText.check_inj h' hc (by rw [hws r j hj, hj]))

end RecDeploy.LastGap

