module

public import Proofs.Flock.C3.Compose.LastRead
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
import Proofs.Flock.Soundness.Refine.Seam

/-! R's Ligerito ends in a coin (H's `LigWhole`), from the source's tie. -/

@[expose] public section

namespace FlockC3.RegLevels

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockC3.Compose

def LigEndsInCoin : Prop :=
  ∀ (m : ℕ) (sch : Flock.Schedule), Flock.fast100 m = some sch → ∀ (T₀ : GF128) (b : ℕ → GF128) (ts : List Step)
    (pend : PendL GF128 Dig), PlaysT (ligeritoC (D := Dig) execArith 2 (schedOf sch) T₀ b) ts pend →
      ∃ (l : List Step) (c : Game.Msg), ts = l ++ [.inr c]

theorem ligeritoC_exact (r : ℕ) (sch : FlockSoundness.Accounting.Schedule) (T₀ : GF128) (b : ℕ → GF128) :
    Exact (ligeritoC (D := Dig) execArith r sch T₀ b) (ligShape GF128 GF256 Dig sch) := by
  unfold ligeritoC ligShape
  rcases sch.levels with _ | ⟨lv₀, rest⟩
  · exact .ret _
  · refine Exact.draw_bind fun _ => Exact.recv_bind fun _ => Exact.draw_bind fun _ => Exact.recv_bind fun _ => ?_
    refine Exact.bind (foldRoundsC_exact execArith _ _ _) fun _ => ?_
    rcases rest with _ | ⟨lv₁, rest'⟩
    · exact .ret _
    · refine Exact.recv_bind fun _ => Exact.bind (twoOodC_exact execArith _ _ _ _) fun _ => ?_
      exact Exact.bind (queryAndBatchC_exact execArith r lv₀ _ _ _ true) fun _ =>
        ZkExec.recursiveRoundsC_exact execArith r _ b _ sch.ext (lv₁ :: rest') _

theorem recShape_coin (F K D : Type) (ext : ℕ) :
    ∀ (lv : FlockSoundness.Accounting.Level) (rest : List FlockSoundness.Accounting.Level),
      ∃ σ, recShape F K D ext (lv :: rest) = σ ++ [.coin F]
  | lv, [] => ⟨foldShape F K lv.lanes ++ [.msg (Fin (2 ^ (ext + 1)) → F), .coin (Fin lv.queries → F),
      .coin (Fin (ceilLog2 lv.queries) → F)], by simp [recShape, qbShape]⟩
  | lv, lv' :: rest => by
      obtain ⟨σ, hσ⟩ := recShape_coin F K D ext lv' rest
      exact ⟨foldShape F K lv.lanes ++ .msg (Cap D) :: (oodShape F K (lv'.logCols + lv'.lanes) ++
        (qbShape F K lv true ++ σ)), by rw [recShape, hσ]; simp⟩

theorem ligShape_coin (F K D : Type) {sch : FlockSoundness.Accounting.Schedule}
    {lv₀ lv₁ : FlockSoundness.Accounting.Level} {rest : List FlockSoundness.Accounting.Level}
    (hl : sch.levels = lv₀ :: lv₁ :: rest) : ∃ σ, ligShape F K D sch = σ ++ [.coin F] := by
  obtain ⟨σ, hσ⟩ := recShape_coin F K D sch.ext lv₁ rest
  refine ⟨.coin (Fin (sch.m - 7) → F) :: .msg F :: .coin F :: .msg (K × K) :: (foldShape F K sch.k0 ++
    .msg (Cap D) :: (oodShape F K (sch.m - 7 - sch.k0 + 1) ++ (qbShape F K lv₀ true ++ σ))), ?_⟩
  simp [ligShape, hl, hσ]

private theorem two_levels {α : Type} : ∀ (l : List α), 2 ≤ l.length → ∃ a b r, l = a :: b :: r
  | a :: b :: r, _ => ⟨a, b, r, rfl⟩

theorem ligEndsInCoin : LigEndsInCoin := by
  intro m sch hf T₀ b ts pend hp
  obtain ⟨-, h2, -⟩ := (fast100_dims m sch hf).1
  replace h2 : 2 ≤ (schedOf sch).levels.length := by simpa [schedOf] using h2
  obtain ⟨lv₀, lv₁, rest, hl⟩ := two_levels _ h2
  obtain ⟨σ, hσ⟩ := ligShape_coin GF128 GF256 Dig hl
  have hk := Exact.kinds hp (ligeritoC_exact 2 (schedOf sch) T₀ b)
  rw [hσ] at hk
  rcases List.eq_nil_or_concat ts with rfl | ⟨l, s, rfl⟩
  · simp at hk
  · rw [List.concat_eq_append, List.map_append, List.map_singleton] at hk
    have hs := (List.cons.inj (List.append_inj_right' hk rfl)).1
    cases s with
    | inl x => simp [kindOf] at hs
    | inr c => exact ⟨l, c, List.concat_eq_append⟩

theorem ligWhole_of_levelTie (hend : LigEndsInCoin) {cap₀ : Cap Dig} {e : ExecHolo} {u : C3Run e}
    (h : LevelTie cap₀ e u) : ∃ (l : List Step) (c : Game.Msg), u.lg = l ++ [.inr c] := by
  obtain ⟨msId, m, sch, Tw, sws, _nc, cap, p, _rt, _rt', hf, hsch, -, -, hwf, hlig, hk, hl, hr, -⟩ := h
  have hlv : ∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default) := fun i hi =>
    fast100_level m sch hf _ (by unfold Array.getD; rw [dite_eq_left hi]; exact Array.getElem_mem hi)
  have hα1 : ∀ i < sch.levels.size, ceilLog2 (sch.levels.getD i default).q ≠ 1 := fun i hi =>
    fast100_alpha m sch hf _ (by unfold Array.getD; rw [dite_eq_left hi]; exact Array.getElem_mem hi)
  obtain ⟨pend, _initCh, _levels, _stF, _stF', tF, -, -, hpl, -, -, -, -, -, -, -, hfin, -⟩ :=
    ligerito_refines (Flock.MerkleScheme.sha512 msId) sch Tw cap p (Flock.basis sws) 2 u.opened.b capDig
      (fun _ ℓ j => (((openAt p (sch.levels.size - 1) ℓ).1)[j]!.toList.map fOf, []))
      (fast100_dims m sch hf).1 hlv hα1 (fun _ _ _ _ => rfl) hwf hlig
  rw [← (finalProg_ok hfin).2.1] at hpl
  rw [← hsch] at hk
  exact hend m sch hf _ _ _ _ (FlockC3.RegLevels.Read.playsT_lig_of_tape hpl hk hl hr)

noncomputable section

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : SamplingDesign n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}

local notation "ℐ" => holoInnerAt T CT st hp

theorem ligWhole_of_srcLevelTie (hend : LigEndsInCoin)
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {cap₀ : Cap Dig} (h : SrcLevelTie zs dg N k Rw src cap₀) :
    FlockC3.Compose.LigWhole zs dg N k Rw src :=
  fun R ω t R₂ τ o hA => ligWhole_of_levelTie hend (h R ω t R₂ τ o hA)

theorem ligWhole_of_tie
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {cap₀ : Cap Dig} (h : SrcLevelTie zs dg N k Rw src cap₀) : FlockC3.Compose.LigWhole zs dg N k Rw src :=
  ligWhole_of_srcLevelTie ligEndsInCoin h

end

end FlockC3.RegLevels
