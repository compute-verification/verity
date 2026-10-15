import Proofs.Flock.C3.RegLevels.ROpensPR
import Proofs.Flock.Recursive.Raw

/-!
# n11: `regReads_of_check`'s `hlv` at R's levels, from check 5

`regReads_of_check` and `regCols_of_check` (`OpensRegs.lean`) read an R level's unit where `checkStackedLevel` passes at
its reads file (`hlv`). `ChecksC3`'s check 5 (`RecCoins.check` on each statement a session sets up) gives it:

* `stackedLevel_of_check`: at a unit named `rec-open-stacked/l<L>-h<H>` (`RecCoins.stackedParams`), `RecCoins.check`
  passes only with candidate 3's `C3` held, and is `checkStackedLevel` at it;
* `CoinsAccepted`: where a session accepts, a transcript it accepts passes check 5 at some `C3` and source
  (`CoinsChecked`); `coinsAccepted_checked`: at the checked family, as `checksAccepted_checked`;
* `stackedLevel_of_accepted`: so at an accepted session whose unit is one of R's levels, `checkStackedLevel` passes at
  its reads file (its first table's statement has its circuit, `setup_zero_c`); `levels_of_accepted`: at each of R's
  levels of a map `M` (`StackedParams`).
-/

namespace FlockC3.RegLevels

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled FlockC3.Compose

/-- **`RecCoins.check` at one of R's levels is `checkStackedLevel`**: at a unit named `rec-open-stacked/l<L>-h<H>`
(`stackedParams`), check 5 passes only where the verifier holds candidate 3's transcript and shape (`c3`), and then
`checkStackedLevel` passes at them. -/
theorem stackedLevel_of_check {sj : Flock.Stmt} {own : Array Flock.Registered.Port}
    {coins : Option Flock.RecCoins.Source} {c3 : Option Flock.RecCoins.C3}
    (h : Flock.RecCoins.check sj (some own) coins c3 = .ok ()) {LANES H : ℕ}
    (hn : Flock.RecCoins.stackedParams sj.c.unit.name = some (LANES, H)) :
    ∃ c3', c3 = some c3' ∧ Flock.RecCoins.checkStackedLevel c3' sj.c sj.pub own LANES H = .ok () := by
  unfold Flock.RecCoins.check at h
  simp only [hn, Option.isNone_some, Bool.and_false, Bool.false_and, Bool.false_eq_true, ↓reduceIte] at h
  obtain ⟨c, hc, h⟩ := FlockSoundness.Refine.ebind_ok h
  cases c3 with
  | none => cases hc
  | some c3' =>
    obtain rfl : c3' = c := Except.ok.inj hc
    exact ⟨c3', rfl, h⟩

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => holoInnerAt T CT st hp

/-- **Check 5 at a session's statements** (`ChecksC3`'s third conjunct): `RecCoins.check` passes on each statement the
session's record sets up, at the reads `own`, the source `coins` and candidate 3's `c3`. -/
def CoinsChecked (c3 : Option Flock.RecCoins.C3) (coins : Option Flock.RecCoins.Source) (I : Inputs)
    (own : Array Flock.Registered.Port) (t : Exec.Transcript) : Prop :=
  ∃ spec sts, setupOf (zkInputs I) t.record = .ok (spec, sts) ∧
    sts.forM (Flock.RecCoins.check · (some own) coins c3) = .ok ()

/-- **Where a session accepts, a transcript it accepts passes check 5** at some `C3` and source (`CoinsChecked`). -/
def CoinsAccepted (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) : Prop :=
  ∀ R ω t R₂ j, (∃ tr', AcceptsZKR (F.zs R ω t R₂ j).I (F.zs R ω t R₂ j).own tr') →
    ∃ tr' c3 coins, CoinsChecked c3 coins (F.zs R ω t R₂ j).I (F.zs R ω t R₂ j).own tr'

/-- **The checked family passes check 5 where a session accepts**: a checked session that accepts kept its own list, so
a transcript it accepts passes `ChecksC3`, whose third conjunct check 5 is (`checks_of_keepOwn`, as
`checksAccepted_checked`). -/
theorem coinsAccepted_checked {ish : RecVStar.Inner} {ss : Array VStar.Stmt}
    {hss : RecVStar.statements ish = .ok ss} {bind : Fin (vstarCount ish) → VStar.Bindings}
    {Vs : Fin (vstarCount ish) → VStmt} {hob : VStarParsesScoped Flock.Tags.circuit ish bind Vs}
    {zr : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin (vstarCount ish)) → (Vs j).OuterR Reg'}
    {hCust : Flock.Premises.RecordCustody zr} {h : Typed zr} {coins : String → Option Flock.RecCoins.Source}
    {rows : Option (Array Flock.Registered.Port)} {own : Fin (vstarCount ish) → Array Flock.Registered.Port}
    {sel : ℕ → String → ℕ → Prop}
    {bc : (j : Fin (vstarCount ish)) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → List UInt8}
    {dec : ℕ → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → (ℐ).Msg}
    {wr : List (Cm × (ℐ).Coin) → Lean.Json × String} {reg : Lean.Json} :
    CoinsAccepted (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg) := by
  intro R ω t R₂ j hex
  obtain ⟨-, tr', -, -, hc⟩ := FlockC3.PubRows.checks_of_keepOwn (z := Flock.Guarantees.custodied zr hCust h R ω t R₂ j)
    (P := fun J o tr' => (ℐ).r ≤ t.length ∧
      Flock.Guarantees.ChecksC3 ish (innerCommitsAt (wr t).1 (wr t).2 st reg).toOption (coins (wr t).2) j (bind j)
        rows J o tr')
    (Pc := fun J o tr' => ∃ c3 cn, CoinsChecked c3 cn J o tr')
    (fun J o tr' hJ => by
      obtain ⟨-, spec, sts, hs, -, hco, -⟩ := hJ
      exact ⟨_, _, spec, sts, hs, hco⟩) hex
  exact ⟨tr', hc⟩

/-- **`regReads_of_check`'s `hlv` at an accepted session whose unit is one of R's levels** (`stackedLevel_of_check`):
check 5 passes on the first table's statement, which has the session's circuit (`setup_zero_c`), so
`checkStackedLevel` passes at the session's reads file, at some `C3` and public file. -/
theorem stackedLevel_of_accepted {F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂} (hco : CoinsAccepted F)
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {R : Reg} {ω : L.Ω} {t : List (Cm × (ℐ).Coin)} {R₂ : Reg₂}
    {τ o}
    (h : Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o) (j : Fin (vstarCount F.ish)) {LANES H : ℕ}
    (hn : Flock.RecCoins.stackedParams (F.Vs j).c.unit.name = some (LANES, H)) :
    ∃ (c3 : Flock.RecCoins.C3) (pub : Flock.Public),
      Flock.RecCoins.checkStackedLevel c3 (F.Vs j).c pub (F.zs R ω t R₂ j).own LANES H = .ok () := by
  obtain ⟨_, _, _, _, _, hs, hfm⟩ := hco R ω t R₂ j ⟨_, (h.1 j).1⟩
  obtain ⟨h0, hc0⟩ := setup_zero_c (F.zs R ω t R₂ j).ht (F.zs R ω t R₂ j).hTags.1 (F.zs R ω t R₂ j).hcH hs
  obtain ⟨c3, -, hlv⟩ := stackedLevel_of_check (LANES := LANES) (H := H) (array_forM_ok hfm 0 h0)
    (by rw [hc0]; exact hn)
  rw [hc0] at hlv
  exact ⟨c3, _, hlv⟩

/-- **`hlv` at each of R's levels** (`stackedLevel_of_accepted` at `StackedParams`): at an accepted outcome, level
`ℓ`'s session of the map `M` passes `checkStackedLevel` at `levelLanes` and `d - cap`. -/
theorem levels_of_accepted {F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂} (hco : CoinsAccepted F)
    {stmt : Fin (vstarCount F.ish) → VStar.Stmt} (M : FlockC3.Reps.StackedMaps stmt F.Vs) {sh : Flock.Shape}
    (hP : StackedParams M sh) {dg : ByteArray → ByteArray} {N k Rw : ℕ} {R : Reg} {ω : L.Ω}
    {t : List (Cm × (ℐ).Coin)} {R₂ : Reg₂} {τ o}
    (h : Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o) :
    ∀ ℓ < sh.sched.levels.size, ∃ (c3 : Flock.RecCoins.C3) (pub : Flock.Public),
      Flock.RecCoins.checkStackedLevel c3 (F.Vs (M.sess ℓ)).c pub (F.zs R ω t R₂ (M.sess ℓ)).own
        (FlockVBridge.AccChain.levelLanes sh ℓ) ((sh.sched.levels[ℓ]!).d - (sh.sched.levels[ℓ]!).cap) = .ok () :=
  fun ℓ hℓ => stackedLevel_of_accepted hco h (M.sess ℓ) (hP ℓ hℓ)

end FlockC3.RegLevels
