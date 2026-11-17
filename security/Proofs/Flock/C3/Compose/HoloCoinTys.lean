module

public import Proofs.Flock.Recursive.HoloMap
public import Proofs.Flock.VBridge.Family.Transcript
import all Init.Data.Array.Basic
import all Init.Data.Range.Basic
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
import Proofs.Flock.Soundness.Discharge.ZkHidden.Sched

/-! `CoinTys` at candidate 3, whose coins keep their form: a vector of one coordinate is a vector, where
`coinTy 1` is a scalar. -/

@[expose] public section

namespace FlockC3.Compose

open Flock Flock.RecVStar Flock.Guarantees FlockSoundness.Discharge.Exec FlockSoundness.Discharge.PlaysCoin
open FlockSoundness.Discharge.Recursion FlockSoundness.Refine FlockLevel3
open FlockVBridge.Family (coinTy exists_coinArr)

def holoSizes (L : List (HoloStream × Array SchedRound)) : List ℕ :=
  L.flatMap fun e => e.2.toList.flatMap fun q => match e.1 with
    | .sparse _ => List.replicate q.1 1
    | _ => if q.1 = 0 then [] else [q.1]

def formTy : Option ℕ → Type
  | none => GF128
  | some n => Fin n → GF128

def formSize : Option ℕ → ℕ
  | none => 1
  | some n => n

theorem formTy_coinTy : ∀ s : Option ℕ, s ≠ some 1 → formTy s = coinTy (formSize s)
  | none, _ => (if_pos rfl : coinTy 1 = GF128).symm
  | some n, h => (if_neg fun e => h (congrArg some e) : coinTy n = (Fin n → GF128)).symm

def zcForms (m : ℕ) : List (Option ℕ) := some 6 :: some (m - 13) :: none :: List.replicate (m - 6) none

def lcForms (kLog : ℕ) : List (Option ℕ) := none :: none :: (List.replicate (kLog - 6) none ++ [none])

def opForms (n : ℕ) : List (Option ℕ) := List.replicate n (some 7) ++ [some n]

def oodForms (dim : ℕ) : List (Option ℕ) := [some dim, none, some dim, none]

def qbForms (lv : FlockSoundness.Accounting.Level) : List (Option ℕ) :=
  [some lv.queries, some (FlockSoundness.Accounting.ceilLog2 lv.queries), none]

def recForms : List FlockSoundness.Accounting.Level → List (Option ℕ)
  | [] => []
  | [lv] => List.replicate lv.lanes (some 2) ++ qbForms lv
  | lv :: lv' :: rest =>
      List.replicate lv.lanes (some 2) ++ (oodForms (lv'.logCols + lv'.lanes) ++ (qbForms lv ++ recForms (lv' :: rest)))

def ligForms (sch : FlockSoundness.Accounting.Schedule) : List (Option ℕ) :=
  match sch.levels with
  | [] => []
  | lv₀ :: rest => some (sch.m - 7) :: none :: (List.replicate sch.k0 (some 2) ++
      match rest with
      | [] => []
      | _ :: _ => oodForms (sch.m - 7 - sch.k0 + 1) ++ (qbForms lv₀ ++ recForms rest))

def sparseForms (sh : FlockSoundness.Model.SparseShape) : List (Option ℕ) :=
  List.replicate (12 + sh.nu + sh.sLog + sh.pLog + sh.tLog) none

noncomputable def holoForms (e : ExecHolo) : List (Option ℕ) :=
  (zcForms (e.cls.pubStatement e.regs).m ++ (lcForms (e.cls.pubStatement e.regs).kLog ++
      (opForms (2 * (e.cls.pubStatement e.regs).regions.length + 2) ++ ligForms e.sch))) ++
    ((zcForms (e.cls.pubStatement e.regs).m ++ (lcForms (e.cls.pubStatement e.regs).kLog ++
      (opForms (2 * (e.cls.pubStatement e.regs).regions.length + 2) ++ ligForms e.sch))) ++
    (sparseForms e.cls.shape ++ (sparseForms e.cls.shape ++ (opForms 2 ++ ligForms e.rsch.sch))))

theorem zcForms_ne_one {m : ℕ} (h : m ≠ 14) : some 1 ∉ zcForms m := by
  have h' : 1 ≠ m - 13 := by omega
  simp [zcForms, h']

theorem lcForms_ne_one (kLog : ℕ) : some 1 ∉ lcForms kLog := by simp [lcForms]

theorem opForms_ne_one {n : ℕ} (h : n ≠ 1) : some 1 ∉ opForms n := by simp [opForms, Ne.symm h]

theorem sparseForms_ne_one (sh : FlockSoundness.Model.SparseShape) : some 1 ∉ sparseForms sh := by
  simp [sparseForms]

theorem ligForms_default : some 1 ∉ ligForms (default : FlockSoundness.Accounting.Schedule) := by decide

theorem ligForms_schedAt (m : ℕ) : some 1 ∉ ligForms (schedAt m) := by
  by_cases hm : m < 36
  · have key : ∀ m' < 36, some 1 ∉ ligForms (schedAt m') := by decide +kernel
    exact key m hm
  · have hb : (m < 22 || m > 35) = true := by
      simp only [Bool.or_eq_true, decide_eq_true_eq]; omega
    have h : Flock.fast100 m = none := by
      unfold Flock.fast100
      simp [hb]
    unfold schedAt
    rw [h]
    exact ligForms_default

theorem holoForms_ne_one_of (e : ExecHolo) (hm : (e.cls.pubStatement e.regs).m ≠ 14)
    (hs : some 1 ∉ ligForms e.sch) (hr : some 1 ∉ ligForms e.rsch.sch) : some 1 ∉ holoForms e := by
  simp only [holoForms, List.mem_append, zcForms_ne_one hm, lcForms_ne_one,
    opForms_ne_one (by omega : 2 * (e.cls.pubStatement e.regs).regions.length + 2 ≠ 1),
    opForms_ne_one (by decide : (2 : ℕ) ≠ 1), sparseForms_ne_one, hs, hr, or_self, not_false_eq_true]

theorem holoForms_ne_one (st : Flock.Stmt) (hp : HoloParams) : some 1 ∉ holoForms (holoAt st hp) := by
  refine holoForms_ne_one_of _ ?_ ?_ ?_
  · show (execClassAt st).cls.m ≠ 14
    rw [execClassAt_cls]
    by_cases h : ClassOk (shapeOf st)
    · obtain ⟨_, hs⟩ := h.2.2
      have h25 := (FlockSoundness.Discharge.ZkHidden.schedule_range hs).1
      rw [clsOf, dif_pos h]
      show (shapeOf st).m ≠ 14
      omega
    · rw [clsOf, dif_neg h]
      show (13 : ℕ) ≠ 14
      decide
  · show some 1 ∉ ligForms (execClassAt st).sch
    rw [execClassAt_sch]
    by_cases h : ClassOk (shapeOf st)
    · rw [if_pos h]
      exact ligForms_schedAt _
    · rw [if_neg h]
      exact ligForms_default
  · have e : (holoAt st hp).rsch.sch =
        if hp.EllOk (execClassAt st).cls.kLog then schedAt (holoAt st hp).cls.shape.mReg else default := rfl
    by_cases hl : hp.EllOk (execClassAt st).cls.kLog
    · rw [e, if_pos hl]
      exact ligForms_schedAt _
    · rw [e, if_neg hl]
      exact ligForms_default

theorem getD_ne_one {l : List (Option ℕ)} (hl : some 1 ∉ l) (j : ℕ) : (l[j]?).getD none ≠ some 1 := by
  intro h
  cases e : l[j]? with
  | none =>
    rw [e, Option.getD_none] at h
    cases h
  | some x =>
    rw [e, Option.getD_some] at h
    exact hl (h ▸ List.mem_of_getElem? e)

theorem holoForms_getD_ne_one (st : Flock.Stmt) (hp : HoloParams) (j : ℕ) :
    ((holoForms (holoAt st hp))[j]?).getD none ≠ some 1 :=
  getD_ne_one (holoForms_ne_one st hp) j

theorem kindCoinTys_append :
    ∀ σ τ : List FlockSoundness.Discharge.Exec.Kind, kindCoinTys (σ ++ τ) = kindCoinTys σ ++ kindCoinTys τ
  | [], _ => rfl
  | .msg _ :: σ, τ => kindCoinTys_append σ τ
  | .coin C :: σ, τ => congrArg (C :: ·) (kindCoinTys_append σ τ)

theorem kindCoinTys_rounds (C : Type) (f : ℕ → List FlockSoundness.Discharge.Exec.Kind) (hf0 : f 0 = [])
    (hf : ∀ n, kindCoinTys (f (n + 1)) = C :: kindCoinTys (f n)) (s : Option ℕ) (hs : formTy s = C) :
    ∀ n, kindCoinTys (f n) = (List.replicate n s).map formTy
  | 0 => by rw [hf0]; rfl
  | n + 1 => by rw [hf, kindCoinTys_rounds C f hf0 hf s hs n, ← hs]; rfl

theorem kindCoinTys_qbShape (lv : FlockSoundness.Accounting.Level) (b : Bool) :
    kindCoinTys (qbShape GF128 GF256 lv b) = (qbForms lv).map formTy := by
  cases b <;> rfl

theorem kindCoinTys_recShape (ext : ℕ) : ∀ lvs : List FlockSoundness.Accounting.Level,
    kindCoinTys (recShape GF128 GF256 FlockSoundness.Refine.Dig ext lvs) = (recForms lvs).map formTy
  | [] => rfl
  | [lv] => by
      rw [recShape, recForms, kindCoinTys_append, List.map_append,
        kindCoinTys_rounds _ (foldShape GF128 GF256) rfl (fun _ => rfl) (some 2) rfl]
      exact congrArg _ (kindCoinTys_qbShape lv false)
  | lv :: lv' :: rest => by
      rw [recShape, recForms, kindCoinTys_append, List.map_append,
        kindCoinTys_rounds _ (foldShape GF128 GF256) rfl (fun _ => rfl) (some 2) rfl]
      refine congrArg _ ?_
      show kindCoinTys (oodShape GF128 GF256 _ ++ _) = _
      rw [kindCoinTys_append, List.map_append, kindCoinTys_append, List.map_append, kindCoinTys_qbShape,
        kindCoinTys_recShape ext (lv' :: rest)]
      rfl

theorem kindCoinTys_ligShape (sch : FlockSoundness.Accounting.Schedule) :
    kindCoinTys (ligShape GF128 GF256 FlockSoundness.Refine.Dig sch) = (ligForms sch).map formTy := by
  have hf := kindCoinTys_rounds _ (foldShape GF128 GF256) rfl (fun _ => rfl) (some 2) rfl
  unfold ligShape ligForms
  rcases sch.levels with _ | ⟨lv₀, _ | ⟨lv₁, rest⟩⟩
  · rfl
  · simp only [kindCoinTys, hf, List.map_cons, List.append_nil]
    rfl
  · simp only [kindCoinTys, kindCoinTys_append, hf, kindCoinTys_qbShape, kindCoinTys_recShape, oodShape, oodForms,
      List.map_cons, List.map_append]
    rfl

theorem kindCoinTys_holoRepShape (m kLog nExtra : ℕ) (sch : FlockSoundness.Accounting.Schedule) :
    kindCoinTys (FlockC3.HoloValues.holoRepShape GF128 GF256 FlockSoundness.Refine.Dig m kLog nExtra sch) =
      (zcForms m ++ (lcForms kLog ++ (opForms (nExtra + 2) ++ ligForms sch))).map formTy := by
  have hp := kindCoinTys_rounds GF128 (pairRounds GF128) rfl (fun _ => rfl) none rfl
  have hs := kindCoinTys_rounds _ (switches GF128) rfl (fun _ => rfl) (some 7) rfl
  simp only [FlockC3.HoloValues.holoRepShape, zcShape, FlockC3.HoloValues.holoLcShape, opShape, zcForms, lcForms,
    opForms, kindCoinTys, kindCoinTys_append, kindCoinTys_ligShape, hp, hs, List.map_cons, List.map_append,
    List.map_nil, List.cons_append, List.append_assoc]
  rfl

theorem kindCoinTys_sparseKinds (sh : FlockSoundness.Model.SparseShape) :
    kindCoinTys (FlockC3.Sparse.sparseKinds GF128 sh) = (sparseForms sh).map formTy := by
  have hd := fun k => kindCoinTys_rounds GF128 (FlockC3.Sparse.degKinds GF128 k) rfl (fun _ => rfl) none rfl
  have hr := kindCoinTys_rounds GF128 (fun n => List.replicate n (Kind.coin GF128)) rfl (fun _ => rfl) none rfl
  unfold FlockC3.Sparse.sparseKinds sparseForms
  rw [kindCoinTys_append, kindCoinTys_append, kindCoinTys_append, kindCoinTys_append, kindCoinTys_append,
    kindCoinTys_append, kindCoinTys_append, kindCoinTys_append, hr, hd, hd, hd, hr, hr]
  have ht : kindCoinTys (if sh.tLog = 0 then [] else [Kind.msg (Fin (2 ^ sh.tLog) → GF128)]) = [] := by
    split <;> rfl
  rw [ht]
  simp only [kindCoinTys, List.nil_append, ← List.map_append, List.replicate_append_replicate]
  congr 2
  omega

theorem kindCoinTys_holoKinds (e : ExecHolo) : kindCoinTys (holoKinds e) = (holoForms e).map formTy := by
  rw [holoKinds, holoForms, kindCoinTys_append, kindCoinTys_append, kindCoinTys_append, kindCoinTys_append,
    kindCoinTys_append, kindCoinTys_holoRepShape, kindCoinTys_sparseKinds, kindCoinTys_ligShape, opShape,
    kindCoinTys_append, kindCoinTys_rounds _ (switches GF128) rfl (fun _ => rfl) (some 7) rfl]
  simp only [List.map_append, opForms]
  rfl

theorem length_kindCoinTys_holoKinds (e : ExecHolo) :
    (kindCoinTys (holoKinds e)).length = (holoForms e).length := by
  rw [kindCoinTys_holoKinds, List.length_map]

theorem nthTy_map {α : Type} (f : α → Type) :
    ∀ (l : List α) (j : ℕ) (_ : j < l.length), nthTy (l.map f) j = f l[j]
  | [], _, hj => absurd hj (Nat.not_lt_zero _)
  | _ :: _, 0, _ => rfl
  | _ :: l, j + 1, hj => nthTy_map f l j (Nat.lt_of_succ_lt_succ hj)

theorem holoCTs_succ (st : Flock.Stmt) (hp : HoloParams) (j : ℕ)
    (h : 1 + j < 1 + (kindCoinTys (holoKinds (holoAt st hp))).length) :
    holoCTs st hp ⟨1 + j, h⟩ = formTy (((holoForms (holoAt st hp))[j]?).getD none) := by
  have hj : j < (holoForms (holoAt st hp)).length := by
    rw [← length_kindCoinTys_holoKinds]
    omega
  show nthTy (_ :: kindCoinTys (holoKinds (holoAt st hp))) (1 + j) = _
  rw [Nat.add_comm 1 j]
  show nthTy (kindCoinTys (holoKinds (holoAt st hp))) j = _
  rw [kindCoinTys_holoKinds, nthTy_map formTy _ j hj, List.getElem?_eq_getElem hj, Option.getD_some]

theorem coinArr_coinWords {C : Type} {c : C} (h : ∃ a : Array F128, coinArr fOf a = ⟨C, c⟩) :
    coinArr fOf (coinWords C c) = ⟨C, c⟩ := by
  unfold coinWords
  rw [dif_pos h]
  exact h.choose_spec

theorem coinWords_vec {C : Type} {n : ℕ} (hC : C = (Fin n → GF128)) (c : C) :
    coinArr fOf (coinWords C c) = ⟨C, c⟩ ∨
      (⟨Fin (coinWords C c).size → GF128, fun i => fOf (coinWords C c)[i.val]!⟩ : FlockSoundness.Game.Msg) =
        ⟨C, c⟩ := by
  subst hC
  by_cases h : ∃ a : Array F128, coinArr fOf a = ⟨Fin n → GF128, c⟩
  · exact .inl (coinArr_coinWords h)
  · right
    have h1 : n = 1 := by
      refine Classical.byContradiction fun hn => h ?_
      have e : coinTy n = (Fin n → GF128) := if_neg hn
      obtain ⟨a, _, ha⟩ := exists_coinArr n (cast e.symm c)
      exact ⟨a, ha.symm.trans (Sigma.ext e (cast_heq e.symm c))⟩
    subst h1
    have hv : ∃ a : Array F128,
        (⟨Fin a.size → GF128, fun i => fOf a[i.val]!⟩ : FlockSoundness.Game.Msg) = ⟨Fin 1 → GF128, c⟩ :=
      ⟨#[c ⟨0, Nat.zero_lt_one⟩], Sigma.ext rfl (heq_of_eq (funext fun ⟨k, hk⟩ => by
        cases k with
        | zero => rfl
        | succ k =>
          have hk' : k + 1 < 1 := hk
          omega))⟩
    unfold coinWords
    rw [dif_neg h, dif_pos hv]
    exact hv.choose_spec

theorem coinArr_roundWords {st : Flock.Stmt} {hp : HoloParams}
    (cs : List ((k : Fin (1 + (kindCoinTys (holoKinds (holoAt st hp))).length)) → holoCTs st hp k)) (j : ℕ)
    (h : 1 + j < 1 + (kindCoinTys (holoKinds (holoAt st hp))).length) (hc : 1 + j < cs.length)
    (h1 : ((holoForms (holoAt st hp))[j]?).getD none ≠ some 1) :
    coinArr fOf (roundWords cs (1 + j)) = ⟨holoCTs st hp ⟨1 + j, h⟩, cs[1 + j] ⟨1 + j, h⟩⟩ := by
  have e := (holoCTs_succ st hp j h).trans (formTy_coinTy _ h1)
  obtain ⟨a, _, ha⟩ := exists_coinArr _ (cast e (cs[1 + j] ⟨1 + j, h⟩))
  unfold roundWords
  rw [List.getElem?_eq_getElem hc]
  dsimp only
  rw [dif_pos h]
  exact coinArr_coinWords ⟨a, ha.symm.trans (Sigma.ext e (cast_heq e _).symm).symm⟩

theorem roundWords_one {st : Flock.Stmt} {hp : HoloParams}
    (cs : List ((k : Fin (1 + (kindCoinTys (holoKinds (holoAt st hp))).length)) → holoCTs st hp k)) (j : ℕ)
    (h : 1 + j < 1 + (kindCoinTys (holoKinds (holoAt st hp))).length) (hc : 1 + j < cs.length)
    (h1 : ((holoForms (holoAt st hp))[j]?).getD none = some 1) :
    coinArr fOf (roundWords cs (1 + j)) = ⟨holoCTs st hp ⟨1 + j, h⟩, cs[1 + j] ⟨1 + j, h⟩⟩ ∨
      (⟨Fin (roundWords cs (1 + j)).size → GF128, fun i => fOf (roundWords cs (1 + j))[i.val]!⟩ :
        FlockSoundness.Game.Msg) = ⟨holoCTs st hp ⟨1 + j, h⟩, cs[1 + j] ⟨1 + j, h⟩⟩ := by
  have e : holoCTs st hp ⟨1 + j, h⟩ = (Fin 1 → GF128) := (holoCTs_succ st hp j h).trans (congrArg formTy h1)
  unfold roundWords
  rw [List.getElem?_eq_getElem hc]
  dsimp only
  rw [dif_pos h]
  exact coinWords_vec e _

end FlockC3.Compose
