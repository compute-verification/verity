import Proofs.Flock.C3.RegLevels.LevelsChecked
import Proofs.Flock.C3.RegLevels.OpensRegs

/-!
# R's levels' tops reads (`TopRead`), from check 5

An R level's unit outputs each opening's climb at its `out` port, which the session reads from the reads file's `out`
entry. `RecCoins.checkStackedLevel` holds that entry to the verifier's (`checkTopReads`): R's level-`l` tops
`rec-top-stacked-L<l>`, opening `j` read at node `pos[j] >>> H`, `pos` the verifier's positions at R's coins.

* `TopRead`: the reads file's first `out` port is that entry; `topRead_of_check` from `checkTopReads`;
* `topLevel_of_check`: an accepted `rec-open-stacked/l<LANES>-h<H>` unit passed `checkTopReads` at the level `l` its
  filter picks out, as `dirsLevel_of_check` has its `dirs`;
* `topRead_of_accepted`: at an accepted session whose unit is one of R's levels (`stackedLevel_of_accepted`), its own
  reads file `(F.zs …).own` (the session's list, not `ownAt`: `ownPorts` names no port of R's levels) is `TopRead`;
  `topReads_of_accepted`: at each of R's levels of a map `M` (`StackedParams`).
-/

namespace FlockC3.RegLevels

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled FlockC3.Compose

/-- **R's level `l`'s `out` reads its tops at each opening's node**: the reads file's first `out` port is R's level-`l`
tops `rec-top-stacked-L<l>`, opening `j` of `q` reading node `pos[j] >>> H`. -/
def TopRead (own : Array Flock.Registered.Port) (l q H : ℕ) (pos : Array ℕ) : Prop :=
  ∃ pt, own.find? (·.name == "out") = some pt ∧ pt.value = s!"rec-top-stacked-L{l}" ∧
    pt.reads = (Array.range q).map fun j => some (pos[j]! >>> H)

theorem topRead_of_check {unit : String} {own : Array Flock.Registered.Port} {l q H : ℕ} {pos : Array ℕ}
    (h : Flock.RecCoins.checkTopReads unit own l q H pos = .ok ()) : TopRead own l q H pos := by
  unfold Flock.RecCoins.checkTopReads at h
  cases hf : own.find? (·.name == "out") with
  | none => rw [hf] at h; cases h
  | some pt =>
    rw [hf] at h
    dsimp only at h
    refine ⟨pt, hf, ?_, ?_⟩
    · cases hv : pt.value != s!"rec-top-stacked-L{l}"
      · exact bne_eq_false_iff_eq.mp hv
      · rw [hv] at h; cases h
    · cases hr : pt.reads != (Array.range q).map fun j => some (pos[j]! >>> H)
      · exact bne_eq_false_iff_eq.mp hr
      · have hv : (pt.value != s!"rec-top-stacked-L{l}") = false := by
          cases hv : pt.value != s!"rec-top-stacked-L{l}"
          · rfl
          · rw [hv] at h; cases h
        rw [hv, hr] at h; cases h

/-- **What an accepted `rec-open-stacked` unit checked of its `out`**: as `dirsLevel_of_check`, with `checkTopReads`
passing at the filter's level `l`, its `q` openings and the verifier's positions there. -/
theorem topLevel_of_check {c3 : Flock.RecCoins.C3} {c : Flock.Circuit} {pub : Flock.Public}
    {own : Array Flock.Registered.Port} {LANES H : ℕ}
    (h : Flock.RecCoins.checkStackedLevel c3 c pub own LANES H = .ok ()) :
    ∃ (o : Flock.RecCoins.Opening) (l : ℕ), Flock.RecCoins.opening c3 c.unit.name = .ok o ∧
      (∀ ℓ < o.sh.sched.levels.size, 2 ^ (o.sh.sched.levels[ℓ]!).lanes = LANES →
        (o.sh.sched.levels[ℓ]!).d - (o.sh.sched.levels[ℓ]!).cap = H → ℓ = l) ∧
      Flock.RecCoins.checkTopReads c.unit.name own l (o.sh.sched.levels[l]!).q H (o.lg.conss[l]!).pos = .ok () := by
  unfold Flock.RecCoins.checkStackedLevel at h
  dsimp only at h
  obtain ⟨o, hop, h⟩ := FlockVBridge.PartS.bind_ok h
  split at h
  · rename_i l hls
    split at h
    · cases h
    · obtain ⟨_u, -, h⟩ := FlockVBridge.PartS.bind_ok h
      obtain ⟨_v, -, htop⟩ := FlockVBridge.PartS.bind_ok h
      refine ⟨o, l, hop, fun ℓ hℓ hL hH => ?_, htop⟩
      have hm : ℓ ∈ [l] := by
        rw [← hls]
        exact List.mem_filter.mpr ⟨List.mem_range.mpr hℓ, by simp only [Bool.and_eq_true, beq_iff_eq]; exact ⟨hL, hH⟩⟩
      exact List.mem_singleton.mp hm
  · cases h

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => holoInnerAt T CT st hp

/-- **An accepted R level's session reads R's tops at each opening's node** (`stackedLevel_of_accepted`,
`topLevel_of_check`): at R's opening `op` at the verifier's coins and the level `l` the unit names, the session's own
reads file is `TopRead` at `l`'s openings and positions. -/
theorem topRead_of_accepted {F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂} (hco : CoinsAccepted F)
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {R : Reg} {ω : L.Ω} {t : List (Cm × (ℐ).Coin)} {R₂ : Reg₂}
    {τ o}
    (h : Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o) (j : Fin (vstarCount F.ish)) {LANES H : ℕ}
    (hn : Flock.RecCoins.stackedParams (F.Vs j).c.unit.name = some (LANES, H)) :
    ∃ (c3 : Flock.RecCoins.C3) (op : Flock.RecCoins.Opening) (l : ℕ),
      Flock.RecCoins.opening c3 (F.Vs j).c.unit.name = .ok op ∧
      (∀ ℓ < op.sh.sched.levels.size, 2 ^ (op.sh.sched.levels[ℓ]!).lanes = LANES →
        (op.sh.sched.levels[ℓ]!).d - (op.sh.sched.levels[ℓ]!).cap = H → ℓ = l) ∧
      TopRead (F.zs R ω t R₂ j).own l (op.sh.sched.levels[l]!).q H (op.lg.conss[l]!).pos := by
  obtain ⟨c3, _, hlv⟩ := stackedLevel_of_accepted hco h j hn
  obtain ⟨op, l, hop, hu, htop⟩ := topLevel_of_check hlv
  exact ⟨c3, op, l, hop, hu, topRead_of_check htop⟩

/-- **`TopRead` at each of R's levels** (`topRead_of_accepted` at `StackedParams`): at an accepted outcome, level `ℓ`'s
session of the map `M` reads R's tops at `levelLanes` and `d - cap`. -/
theorem topReads_of_accepted {F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂} (hco : CoinsAccepted F)
    {stmt : Fin (vstarCount F.ish) → VStar.Stmt} (M : FlockC3.Reps.StackedMaps stmt F.Vs) {sh : Flock.Shape}
    (hP : StackedParams M sh) {dg : ByteArray → ByteArray} {N k Rw : ℕ} {R : Reg} {ω : L.Ω}
    {t : List (Cm × (ℐ).Coin)} {R₂ : Reg₂} {τ o}
    (h : Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o) :
    ∀ ℓ < sh.sched.levels.size, ∃ (c3 : Flock.RecCoins.C3) (op : Flock.RecCoins.Opening) (l : ℕ),
      Flock.RecCoins.opening c3 (F.Vs (M.sess ℓ)).c.unit.name = .ok op ∧
      (∀ ℓ' < op.sh.sched.levels.size, 2 ^ (op.sh.sched.levels[ℓ']!).lanes = FlockVBridge.AccChain.levelLanes sh ℓ →
        (op.sh.sched.levels[ℓ']!).d - (op.sh.sched.levels[ℓ']!).cap =
          (sh.sched.levels[ℓ]!).d - (sh.sched.levels[ℓ]!).cap → ℓ' = l) ∧
      TopRead (F.zs R ω t R₂ (M.sess ℓ)).own l (op.sh.sched.levels[l]!).q
        ((sh.sched.levels[ℓ]!).d - (sh.sched.levels[ℓ]!).cap) (op.lg.conss[l]!).pos :=
  fun ℓ hℓ => topRead_of_accepted hco h (M.sess ℓ) (hP ℓ hℓ)

end FlockC3.RegLevels

#print axioms FlockC3.RegLevels.topRead_of_check
#print axioms FlockC3.RegLevels.topLevel_of_check
#print axioms FlockC3.RegLevels.topRead_of_accepted
#print axioms FlockC3.RegLevels.topReads_of_accepted
#print axioms FlockC3.RegLevels.dirsLevel_of_check
