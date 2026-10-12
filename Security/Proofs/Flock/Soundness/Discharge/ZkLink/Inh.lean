import Proofs.Flock.Soundness.Discharge.ZkLink.Compiled
import Proofs.Flock.Soundness.Refine.LiveCompiled

/-!
# Every message a compiled `--zk` rep receives has an inhabited type

`Refine.live_le` needs `AllInh` of the reps it simulates, so that where no accepted live run goes on, the compiled table
can still play. `allInh_repZCAt` gives it for `repZCAt`, `Refine.modelTable`'s `G` for the compiled `--zk` table, from
`AllInh` of each game a masked rep is built from.
-/

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting ZkSession

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
variable {D Sl : Type}

omit [DecidableEq F] in
theorem allInh_residualRoundsZK {np : ℕ} (P : PadsA np) :
    ∀ (n j : ℕ) (req : Fin n → F) (claim : LinF F np), AllInh (residualRoundsZK P n j req claim)
  | 0, _, _, _ => .ret _
  | n + 1, j, req, claim => by
      simp only [residualRoundsZK]
      repeat' (first
          | exact AllInh.ret _
          | exact AllInh.pure' _
          | exact AllInh.recv' _
          | exact AllInh.draw' _
          | refine AllInh.bind' ?_ fun _ => ?_
          | intro _
          | split
          | dsimp only)
      all_goals exact allInh_residualRoundsZK P n _ _ _

omit [DecidableEq F] [Fintype K] [DecidableEq K] in
theorem allInh_zerocheckZK (A : Arith F K) (m : ℕ) {np : ℕ} (P : PadsA np) (tr : Triple np) :
    AllInh (zerocheckZK A m P tr) := by
  unfold zerocheckZK
  repeat' (first
      | exact AllInh.ret _
      | exact AllInh.pure' _
      | exact AllInh.recv' _
      | exact AllInh.draw' _
      | refine AllInh.bind' ?_ fun _ => ?_
      | intro _
      | split
      | dsimp only)
  all_goals exact allInh_residualRoundsZK P _ _ _ _

omit [DecidableEq F] in
theorem allInh_lincheckRoundsZK {np : ℕ} (P : PadsA np) :
    ∀ (n j : ℕ) (claim : LinF F np), AllInh (lincheckRoundsZK P n j claim)
  | 0, _, _ => .ret _
  | n + 1, j, claim => by
      simp only [lincheckRoundsZK]
      repeat' (first
          | exact AllInh.ret _
          | exact AllInh.pure' _
          | exact AllInh.recv' _
          | exact AllInh.draw' _
          | refine AllInh.bind' ?_ fun _ => ?_
          | intro _
          | split
          | dsimp only)
      all_goals exact allInh_lincheckRoundsZK P n _ _

omit [DecidableEq F] [Fintype K] [DecidableEq K] in
theorem allInh_lincheckZK (A : Arith F K) (S : Statement) {np : ℕ} (P : PadsA np) (zc : ZcZK F np S.m) :
    AllInh (lincheckZK A S P zc) := by
  unfold lincheckZK
  repeat' (first
      | exact AllInh.ret _
      | exact AllInh.pure' _
      | exact AllInh.recv' _
      | exact AllInh.draw' _
      | refine AllInh.bind' ?_ fun _ => ?_
      | intro _
      | split
      | dsimp only)
  all_goals exact allInh_lincheckRoundsZK P _ _ _

omit [Fintype K] [DecidableEq K] in
theorem allInh_openingZK (A : Arith F K) {m : ℕ} (hm : 7 ≤ m) (a c : Claim F m) (extra : List (Claim F m)) :
    AllInh (openingZK A hm a c extra) := by
  have hs : ∀ c' : Claim F m, AllInh (opStep c') := fun _ => by unfold opStep; repeat' (first | exact AllInh.ret _ | exact AllInh.pure' _ | exact AllInh.recv' _ | exact AllInh.draw' _ | refine AllInh.bind' ?_ fun _ => ?_)
  unfold openingZK
  refine AllInh.bind' (hs a) fun _ => AllInh.bind' (hs c) fun _ =>
    AllInh.bind' (AllInh.mapM _ hs _) fun _ => AllInh.bind' (AllInh.draw' _) fun _ => .ret _

omit [DecidableEq F] [Fintype K] [DecidableEq K] in
theorem allInh_drawKs (A : Arith F K) : ∀ e : ℕ, AllInh (drawKs A e)
  | 0 => .ret _
  | e + 1 => by
      simp only [drawKs]
      exact AllInh.bind' (allInh_drawK A) fun _ => AllInh.bind' (allInh_drawKs A e) fun _ => .ret _

omit [DecidableEq F] [Fintype K] [DecidableEq K] in
theorem allInh_foldRoundsZK (A : Arith F K) {np e : ℕ} (P : Pads0 np e) :
    ∀ (k j : ℕ) (s : LinK K np) (q : QuadL K np), AllInh (foldRoundsZK A P k j s q)
  | 0, _, _, _ => .ret _
  | 1, _, _, _ => by
      simp only [foldRoundsZK]
      exact AllInh.bind' (allInh_drawK A) fun _ => .ret _
  | k + 2, j, _, q => by
      simp only [foldRoundsZK]
      refine AllInh.bind' (allInh_drawK A) fun _ => ?_
      repeat' (first
          | exact AllInh.ret _
          | exact AllInh.pure' _
          | exact AllInh.recv' _
          | exact AllInh.draw' _
          | refine AllInh.bind' ?_ fun _ => ?_
          | intro _
          | split
          | dsimp only)
      all_goals exact allInh_foldRoundsZK A P (k + 1) (j + 1) _ _

variable [Nonempty D]

theorem allInh_ligeritoPadZC (A : Arith F K) {np e : ℕ} (P : Pads0 np e) (sch : Schedule) (t : ℕ) (c₀ : F) (r : ℕ)
    (T : F) (b : ℕ → F) : AllInh (ligeritoPadZC (D := D) A P sch t c₀ r T b) := by
  unfold ligeritoPadZC ligTailC queryAndBatchPadC
  repeat' (first
      | exact AllInh.ret _
      | exact AllInh.pure' _
      | exact AllInh.recv' _
      | exact AllInh.draw' _
      | refine AllInh.bind' ?_ fun _ => ?_
      | intro _
      | split
      | dsimp only)
  all_goals first
    | exact allInh_foldRoundsZK A P _ _ _ _
    | exact allInh_drawKs A _
    | exact allInh_twoOodC A _ _ _ _
    | exact allInh_recursiveRoundsC A _ _ _ _ _ _ _

theorem allInh_repMaskedZC (A : Arith F K) {np e : ℕ} (P₀ : Pads0 np e) (PA : PadsA np) (tr : Triple np)
    (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (t : ℕ) (c₀ : F) (r : ℕ) (extra : List (Claim F S.m)) :
    AllInh (repMaskedZC (D := D) A P₀ PA tr S sch hm t c₀ r extra) := by
  unfold repMaskedZC
  refine AllInh.map _ (AllInh.bind (fun _ => AllInh.map _ (AllInh.bind (fun _ => AllInh.map _ ?_)
    (allInh_lincheckZK A S PA _))) (allInh_zerocheckZK A S.m PA tr))
  unfold restZC
  refine AllInh.bind' (allInh_openingZK A _ _ _ _) fun _ => ?_
  repeat' (first
      | exact AllInh.ret _
      | exact AllInh.pure' _
      | exact AllInh.recv' _
      | exact AllInh.draw' _
      | refine AllInh.bind' ?_ fun _ => ?_
      | intro _
      | split
      | dsimp only)
  all_goals exact allInh_ligeritoPadZC A _ _ _ _ _ _ _

omit [DecidableEq F] in
theorem allInh_innerGameC (H : List UInt8 → D) (Ep : Merkle.Enc (F × F) D) (Lp : Merkle.Leaf (F × F) Sl D H)
    (Lτ : Merkle.Leaf F Sl D H) {N : ℕ} (dom : Fin N ↪ F) (s nP q : ℕ) (pos : F → Fin N) {nA : ℕ}
    (rows : Fin nA → Lin F s) (tr : Triple s) (root : D) :
    AllInh (innerGameC H Ep Lp Lτ dom s nP q pos rows tr root) := by
  unfold innerGameC
  repeat' (first
      | exact AllInh.ret _
      | exact AllInh.pure' _
      | exact AllInh.recv' _
      | exact AllInh.draw' _
      | refine AllInh.bind' ?_ fun _ => ?_)

/-- **Every message a compiled `--zk` rep receives has an inhabited type** (`Refine.live_le`'s `hinh` at
`G := repZCAt A Hs T`). -/
theorem allInh_repZCAt (A : Arith F K) (Hs : Hashes F D Sl) (T : TabZK F mPts) (pts : Fin 2 → Fin mPts → F)
    (r : Fin 2) : AllInh (repZCAt A Hs T pts r) := by
  unfold repZCAt repZC
  exact AllInh.bind' (AllInh.recv' D) fun _ => AllInh.bind' (allInh_repMaskedZC A _ _ _ _ _ _ _ _ _ _) fun _ =>
    AllInh.bind' (allInh_innerGameC _ _ _ _ _ _ _ _ _ _ _ _) fun _ => .ret _

end FlockSoundness.Discharge.ZkLink
