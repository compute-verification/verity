module

public import Proofs.Flock.Recursive.ValueFinder
public import Proofs.Flock.Recursive.ValueSrc
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
import Proofs.Flock.VBridge.PartS

/-! The ties of the verifier's own rows (`valueSrcC3`) to what check 5 accepted. -/

@[expose] public section

namespace FlockSoundness.Discharge.Recursion

open Game Audit

variable {n : ℕ} {L : SamplingDesign n} {Reg Cm Reg₂ : Type} {I : Inner} {S : ℕ} {Vs : Fin S → VStmt}

theorem valueSrcC3_eq {c3 : List (Cm × I.Coin) → Option Flock.RecCoins.C3} {R : Reg} {ω : L.Ω} {t : List (Cm × I.Coin)}
    {R₂ : Reg₂} {j : Fin S} {u : Fin (Vs j).nv} {q : Fin (Vs j).c.ports.size} {rows : Array (Array ℕ)}
    {value prog : String} (hv : valueRows (c3 t) (Vs j).c ((Vs j).c.ports[q.val]!).name = some (rows, value))
    (hprog : (Vs j).c.metaJ.getObjValAs? String "program_sha512" = .ok prog) :
    valueSrcC3 (L := L) (Reg := Reg) (Reg₂ := Reg₂) (Vs := Vs) c3 R ω t R₂ ⟨j, u, q⟩ =
      some (rows, Flock.RecCoins.verifierSalt prog value) := by
  simp only [valueSrcC3, hv, hprog]

theorem stacked_of_check {c3 : Flock.RecCoins.C3} {c : Flock.Circuit} {pub : Flock.Public} {own : Array Flock.Registered.Port}
    {LANES H : ℕ} (h : Flock.RecCoins.checkStackedLevel c3 c pub own LANES H = .ok ()) :
    ∃ o l, Flock.RecCoins.opening c3 c.unit.name = .ok o ∧ stackedLevels o LANES H = [l] ∧
      (∃ what, Flock.RecCoins.checkValue c pub own "coef" Flock.RecCoins.COEF_VALUE (stackedCoefRows o l LANES) what = .ok ()) ∧
      ∃ what, Flock.RecCoins.checkValue c pub own "dirs" Flock.RecCoins.DIRS_VALUE (stackedDirsRows o l H) what = .ok () := by
  unfold Flock.RecCoins.checkStackedLevel at h
  dsimp only at h
  obtain ⟨o, hop, h⟩ := FlockVBridge.PartS.bind_ok h
  split at h
  · rename_i l hls
    split at h
    · cases h
    · obtain ⟨_u, hcv, hdv⟩ := FlockVBridge.PartS.bind_ok h
      obtain ⟨_v, hdv, -⟩ := FlockVBridge.PartS.bind_ok hdv
      exact ⟨o, l, hop, hls, ⟨_, hcv⟩, _, hdv⟩
  · cases h

theorem opening_of_check {c3 : Flock.RecCoins.C3} {c : Flock.Circuit} {pub : Flock.Public} {own : Array Flock.Registered.Port}
    {m cl j : ℕ} (h : Flock.RecCoins.checkOpening c3 c pub own m cl j = .ok ()) :
    ∃ o rows what, Flock.RecCoins.opening c3 c.unit.name = .ok o ∧ openingVRows o j = some rows ∧
      Flock.RecCoins.checkValue c pub own "v" Flock.RecCoins.V_VALUE rows what = .ok () ∧
      Flock.RecCoins.checkZero c pub own Flock.RecCoins.ZERO_VALUE 1 Flock.RecCoins.OUT_WORDS = .ok () := by
  unfold Flock.RecCoins.checkOpening at h
  dsimp only at h
  obtain ⟨o, hop, h⟩ := FlockVBridge.PartS.bind_ok h
  split at h
  · cases h
  · split at h
    · rename_i gs hX
      obtain ⟨gs', hgs, h⟩ := FlockVBridge.PartS.bind_ok h
      cases hgs
      split at h
      · rename_i hj
        split at h
        · cases h
        · obtain ⟨⟨⟩, hcv, hz⟩ := FlockVBridge.PartS.bind_ok h
          exact ⟨o, _, _, hop, by simp only [openingVRows, hX, hj], hcv, hz⟩
      · cases h
    · cases h

theorem valueRows_coef {c3 : Flock.RecCoins.C3} {c : Flock.Circuit} {LANES H l : ℕ} {o : Flock.RecCoins.RegisteredOpening}
    (hso : Flock.RecCoins.stackedParams c.unit.name = some (LANES, H)) (hop : Flock.RecCoins.opening c3 c.unit.name = .ok o)
    (hls : stackedLevels o LANES H = [l]) :
    valueRows (some c3) c "coef" = some (stackedCoefRows o l LANES, Flock.RecCoins.COEF_VALUE) := by
  simp only [valueRows, hop, hso, hls]
  rfl

theorem valueRows_dirs {c3 : Flock.RecCoins.C3} {c : Flock.Circuit} {LANES H l : ℕ} {o : Flock.RecCoins.RegisteredOpening}
    (hso : Flock.RecCoins.stackedParams c.unit.name = some (LANES, H)) (hop : Flock.RecCoins.opening c3 c.unit.name = .ok o)
    (hls : stackedLevels o LANES H = [l]) :
    valueRows (some c3) c "dirs" = some (stackedDirsRows o l H, Flock.RecCoins.DIRS_VALUE) := by
  simp only [valueRows, hop, hso, hls]
  rfl

theorem valueRows_v {c3 : Flock.RecCoins.C3} {c : Flock.Circuit} {m cl j : ℕ} {o : Flock.RecCoins.RegisteredOpening}
    {rows : Array (Array ℕ)} (hso : Flock.RecCoins.stackedParams c.unit.name = none)
    (hp : Flock.RecOpening.params c.unit.name = some (m, cl, j)) (hop : Flock.RecCoins.opening c3 c.unit.name = .ok o)
    (hr : openingVRows o j = some rows) :
    valueRows (some c3) c "v" = some (rows, Flock.RecCoins.V_VALUE) := by
  simp only [valueRows, hop, hso, hp, hr]
  rfl

theorem valueRows_out {c3 : Flock.RecCoins.C3} {c : Flock.Circuit} {m cl j : ℕ} {o : Flock.RecCoins.RegisteredOpening}
    (hso : Flock.RecCoins.stackedParams c.unit.name = none)
    (hp : Flock.RecOpening.params c.unit.name = some (m, cl, j)) (hop : Flock.RecCoins.opening c3 c.unit.name = .ok o) :
    valueRows (some c3) c "out" = some (Flock.RecCoins.zeroRows 1 Flock.RecCoins.OUT_WORDS, Flock.RecCoins.ZERO_VALUE) := by
  simp only [valueRows, hop, hso, hp] <;> rfl

theorem stackedLevel_of_check {st : Flock.Stmt} {own : Array Flock.Registered.Port} {files : Option Flock.RecCoins.Source}
    {c3 : Flock.RecCoins.C3} {LANES H : ℕ} (h : Flock.RecCoins.check st (some own) files (some c3) = .ok ())
    (hso : Flock.RecCoins.stackedParams st.c.unit.name = some (LANES, H)) :
    Flock.RecCoins.checkStackedLevel c3 st.c st.pub own LANES H = .ok () := by
  revert h
  unfold Flock.RecCoins.check
  simp [hso]

theorem openingCheck_of_check {st : Flock.Stmt} {own : Array Flock.Registered.Port} {files : Option Flock.RecCoins.Source}
    {c3 : Flock.RecCoins.C3} {m cl j : ℕ} (h : Flock.RecCoins.check st (some own) files (some c3) = .ok ())
    (hso : Flock.RecCoins.stackedParams st.c.unit.name = none) (hp : Flock.RecOpening.params st.c.unit.name = some (m, cl, j)) :
    Flock.RecCoins.checkOpening c3 st.c st.pub own m cl j = .ok () := by
  revert h
  unfold Flock.RecCoins.check
  simp [hso, hp]

end FlockSoundness.Discharge.Recursion
