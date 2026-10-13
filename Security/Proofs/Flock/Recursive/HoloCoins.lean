import Proofs.Flock.Recursive.HoloExec
import Proofs.Flock.C3.HoloValues.Shape
import Proofs.Flock.Recursive.StageKinds

/-!
# Candidate 3's coin types

`Flock.Guarantees.holoCTs st hp`: the inner protocol's coin types at candidate 3, a coin's index naming its type: the
link points, then the coins of the stages' shapes in the run's order (both reps, both sparse phases, R's opening,
Ligerito on R: `holoKinds`), as `StageCoins` lays the history's coins out. Every coin of those shapes is finite and has
an element (`coinsFin_holoKinds`), so each coin type is a `Fintype` with an element.
-/

namespace FlockSoundness.Discharge.Exec

open FlockSoundness.Accounting FlockSoundness.Model

/-- **Every coin of `σ` is finite and has an element.** -/
def CoinsFin (σ : List Kind) : Prop := ∀ C, Kind.coin C ∈ σ → Finite C ∧ Nonempty C

theorem coinsFin_nil : CoinsFin [] := fun _ h => by simp at h

theorem coinsFin_append {a b : List Kind} (ha : CoinsFin a) (hb : CoinsFin b) : CoinsFin (a ++ b) :=
  fun C h => (List.mem_append.1 h).elim (ha C) (hb C)

theorem coinsFin_msg {M : Type} {σ : List Kind} (h : CoinsFin σ) : CoinsFin (.msg M :: σ) := fun C hC => by
  rcases List.mem_cons.1 hC with e | e
  · cases e
  · exact h C e

theorem coinsFin_coin {C : Type} [Finite C] [Nonempty C] {σ : List Kind} (h : CoinsFin σ) :
    CoinsFin (.coin C :: σ) := fun C' hC => by
  rcases List.mem_cons.1 hC with e | e
  · cases e
    exact ⟨inferInstance, inferInstance⟩
  · exact h C' e

theorem coinsFin_ite {p : Prop} [Decidable p] {a b : List Kind} (ha : CoinsFin a) (hb : CoinsFin b) :
    CoinsFin (if p then a else b) := by
  split
  · exact ha
  · exact hb

theorem coinsFin_replicate {C : Type} [Finite C] [Nonempty C] : ∀ n, CoinsFin (List.replicate n (Kind.coin C))
  | 0 => coinsFin_nil
  | n + 1 => by
      rw [List.replicate_succ]
      exact coinsFin_coin (coinsFin_replicate n)

section shapes

variable {F K : Type} [Finite F] [Nonempty F]

theorem coinsFin_pairRounds : ∀ n, CoinsFin (pairRounds F n)
  | 0 => coinsFin_nil
  | n + 1 => by
      unfold pairRounds
      exact coinsFin_msg (coinsFin_coin (coinsFin_pairRounds n))

theorem coinsFin_switches : ∀ n, CoinsFin (switches F n)
  | 0 => coinsFin_nil
  | n + 1 => by
      unfold switches
      exact coinsFin_msg (coinsFin_coin (coinsFin_switches n))

theorem coinsFin_foldShape : ∀ k, CoinsFin (foldShape F K k)
  | 0 => coinsFin_nil
  | k + 1 => by
      unfold foldShape
      exact coinsFin_coin (coinsFin_msg (coinsFin_foldShape k))

theorem coinsFin_degKinds (k : ℕ) : ∀ n, CoinsFin (FlockC3.Sparse.degKinds F k n)
  | 0 => coinsFin_nil
  | n + 1 => by
      unfold FlockC3.Sparse.degKinds
      exact coinsFin_msg (coinsFin_coin (coinsFin_degKinds k n))

theorem coinsFin_zcShape (m : ℕ) : CoinsFin (zcShape F m) := by
  unfold zcShape
  exact coinsFin_coin (coinsFin_coin (coinsFin_msg (coinsFin_coin
    (coinsFin_append (coinsFin_pairRounds _) (coinsFin_msg coinsFin_nil)))))

theorem coinsFin_holoLcShape (kLog : ℕ) : CoinsFin (FlockC3.HoloValues.holoLcShape F kLog) := by
  unfold FlockC3.HoloValues.holoLcShape
  exact coinsFin_coin (coinsFin_coin (coinsFin_append (coinsFin_pairRounds _)
    (coinsFin_msg (coinsFin_msg (coinsFin_coin coinsFin_nil)))))

theorem coinsFin_opShape (n : ℕ) : CoinsFin (opShape F n) := by
  unfold opShape
  exact coinsFin_append (coinsFin_switches n) (coinsFin_coin coinsFin_nil)

theorem coinsFin_oodShape (dim : ℕ) : CoinsFin (oodShape F K dim) := by
  unfold oodShape
  exact coinsFin_coin (coinsFin_msg (coinsFin_msg (coinsFin_coin (coinsFin_coin (coinsFin_msg (coinsFin_msg
    (coinsFin_coin coinsFin_nil)))))))

theorem coinsFin_qbShape (lv : Level) (b : Bool) : CoinsFin (qbShape F K lv b) := by
  unfold qbShape
  exact coinsFin_coin (coinsFin_coin (coinsFin_append (coinsFin_ite (coinsFin_msg coinsFin_nil) coinsFin_nil)
    (coinsFin_coin coinsFin_nil)))

theorem coinsFin_recShape (D : Type) (ext : ℕ) : ∀ lvs : List Level, CoinsFin (recShape F K D ext lvs)
  | [] => by
      unfold recShape
      exact coinsFin_nil
  | [lv] => by
      unfold recShape
      exact coinsFin_append (coinsFin_foldShape _) (coinsFin_msg (coinsFin_qbShape _ _))
  | lv :: lv' :: rest => by
      rw [recShape]
      exact coinsFin_append (coinsFin_foldShape _) (coinsFin_msg (coinsFin_append (coinsFin_oodShape _)
        (coinsFin_append (coinsFin_qbShape _ _) (coinsFin_recShape D ext (lv' :: rest)))))

theorem coinsFin_ligShape (D : Type) (sch : Schedule) : CoinsFin (ligShape F K D sch) := by
  unfold ligShape
  split
  · exact coinsFin_nil
  · refine coinsFin_coin (coinsFin_msg (coinsFin_coin (coinsFin_msg (coinsFin_append (coinsFin_foldShape _) ?_))))
    split
    · exact coinsFin_nil
    · exact coinsFin_msg (coinsFin_append (coinsFin_oodShape _) (coinsFin_append (coinsFin_qbShape _ _)
        (coinsFin_recShape D _ _)))

theorem coinsFin_holoRepShape (D : Type) (m kLog nExtra : ℕ) (sch : Schedule) :
    CoinsFin (FlockC3.HoloValues.holoRepShape F K D m kLog nExtra sch) := by
  unfold FlockC3.HoloValues.holoRepShape
  exact coinsFin_append (coinsFin_zcShape m) (coinsFin_append (coinsFin_holoLcShape kLog)
    (coinsFin_append (coinsFin_opShape _) (coinsFin_ligShape D sch)))

theorem coinsFin_sparseKinds (sh : SparseShape) : CoinsFin (FlockC3.Sparse.sparseKinds F sh) := by
  unfold FlockC3.Sparse.sparseKinds
  exact coinsFin_append (coinsFin_replicate 6) (coinsFin_append (coinsFin_degKinds 1 6)
    (coinsFin_append (coinsFin_msg coinsFin_nil) (coinsFin_append (coinsFin_degKinds _ _)
      (coinsFin_append (coinsFin_degKinds _ _) (coinsFin_append (coinsFin_msg coinsFin_nil)
        (coinsFin_append (coinsFin_replicate _) (coinsFin_append (coinsFin_ite coinsFin_nil (coinsFin_msg coinsFin_nil))
          (coinsFin_replicate _))))))))

end shapes

/-- The coin types of `σ`'s kinds, in order. -/
def kindCoinTys : List Kind → List Type
  | [] => []
  | .msg _ :: σ => kindCoinTys σ
  | .coin C :: σ => C :: kindCoinTys σ

theorem mem_kindCoinTys {C : Type} : ∀ σ : List Kind, C ∈ kindCoinTys σ → Kind.coin C ∈ σ
  | [], h => by simp [kindCoinTys] at h
  | .msg _ :: σ, h => List.mem_cons.2 (Or.inr (mem_kindCoinTys σ h))
  | .coin C' :: σ, h => by
      have h' : C ∈ C' :: kindCoinTys σ := h
      rcases List.mem_cons.1 h' with e | e
      · subst e
        exact List.mem_cons.2 (Or.inl rfl)
      · exact List.mem_cons.2 (Or.inr (mem_kindCoinTys σ e))

/-- Type `k` of a list of types, `Unit` past its end. -/
def nthTy : List Type → ℕ → Type
  | [], _ => Unit
  | C :: _, 0 => C
  | _ :: l, k + 1 => nthTy l k

theorem nthTy_of {P : Type → Prop} (hU : P Unit) : ∀ (l : List Type) (k : ℕ), (∀ C ∈ l, P C) → P (nthTy l k)
  | [], _, _ => hU
  | C :: _, 0, h => h C (List.mem_cons.2 (Or.inl rfl))
  | _ :: l, k + 1, h => nthTy_of hU l k fun C hC => h C (List.mem_cons.2 (Or.inr hC))

end FlockSoundness.Discharge.Exec

namespace Flock.Guarantees

open FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Recursion FlockLevel3

/-- **The coin kinds of candidate 3's stages**, in the run's order: both reps' (`holoRepShape` at the class's statement,
two extra claims per region, and the reps' schedule), both sparse phases' at the class's shape, R's opening's (two
claims) and Ligerito's on R at R's registered schedule. -/
noncomputable def holoKinds (e : ExecHolo) : List FlockSoundness.Discharge.Exec.Kind :=
  FlockC3.HoloValues.holoRepShape GF128 GF256 FlockSoundness.Refine.Dig (e.cls.pubStatement e.regs).m
      (e.cls.pubStatement e.regs).kLog (2 * (e.cls.pubStatement e.regs).regions.length) e.sch ++
    (FlockC3.HoloValues.holoRepShape GF128 GF256 FlockSoundness.Refine.Dig (e.cls.pubStatement e.regs).m
      (e.cls.pubStatement e.regs).kLog (2 * (e.cls.pubStatement e.regs).regions.length) e.sch ++
    (FlockC3.Sparse.sparseKinds GF128 e.cls.shape ++ (FlockC3.Sparse.sparseKinds GF128 e.cls.shape ++
    (opShape GF128 2 ++ ligShape GF128 GF256 FlockSoundness.Refine.Dig e.rsch.sch))))

theorem coinsFin_holoKinds (e : ExecHolo) : CoinsFin (holoKinds e) := by
  unfold holoKinds
  exact coinsFin_append (coinsFin_holoRepShape _ _ _ _ _) (coinsFin_append (coinsFin_holoRepShape _ _ _ _ _)
    (coinsFin_append (coinsFin_sparseKinds _) (coinsFin_append (coinsFin_sparseKinds _)
      (coinsFin_append (coinsFin_opShape _) (coinsFin_ligShape _ _)))))

/-- **The inner protocol's coin types at candidate 3**, a coin's index naming its type: the link points (`mPts`
coordinates each, the table's first coin), then the coins of the stages' shapes (`holoKinds`) in order. -/
noncomputable def holoCTs (st : Flock.Stmt) (hp : HoloParams)
    (k : Fin (1 + (kindCoinTys (holoKinds (holoAt st hp))).length)) : Type :=
  nthTy ((Fin 2 → Fin (holoAt st hp).cls.mPts → GF128) :: kindCoinTys (holoKinds (holoAt st hp))) k.val

theorem holoCTs_fin (st : Flock.Stmt) (hp : HoloParams)
    (k : Fin (1 + (kindCoinTys (holoKinds (holoAt st hp))).length)) :
    Finite (holoCTs st hp k) ∧ Nonempty (holoCTs st hp k) := by
  unfold holoCTs
  refine nthTy_of (P := fun C => Finite C ∧ Nonempty C) ⟨inferInstance, inferInstance⟩ _ _ fun C hC => ?_
  rcases List.mem_cons.1 hC with e | e
  · subst e
    exact ⟨inferInstance, inferInstance⟩
  · exact coinsFin_holoKinds _ C (mem_kindCoinTys _ e)

noncomputable instance instFintypeHoloCTs (st : Flock.Stmt) (hp : HoloParams)
    (k : Fin (1 + (kindCoinTys (holoKinds (holoAt st hp))).length)) : Fintype (holoCTs st hp k) :=
  @Fintype.ofFinite _ (holoCTs_fin st hp k).1

noncomputable instance instInhabitedHoloCTs (st : Flock.Stmt) (hp : HoloParams)
    (k : Fin (1 + (kindCoinTys (holoKinds (holoAt st hp))).length)) : Inhabited (holoCTs st hp k) :=
  Classical.inhabited_of_nonempty (holoCTs_fin st hp k).2

end Flock.Guarantees
