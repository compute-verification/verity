import FlockSoundness.Discharge.ZkReg
import FlockBridge.RegisteredReads

/-!
# Registered reads of a `--zk --registered` session

`RegisteredReads.regReadSound_of_checked` at `flock-verify verify --zk --registered F` sessions
(`ZkReg.LiveAcceptsZKR`), on the public plan's layout and rows (`ZkReg.laySh`, `ZkReg.rowsSh`), where the registered
reads are read. The registered check at every accepted run is `ZkReg.checked_of_accepts`, for the registrant's reads at
the commit strings the run's drawn units read, when those are the verifier's own there (`OwnAtDraws`, named). What
remains is `cr/sha-512` for the registered-path finder and the registrant's values (`RegisteredReads.RegValsSound`).

`OwnAtDraws` holds the executable's restriction: the verifier reads only rows of the drawn instances, at `F`'s ports
named as the circuit's, at the positions `F` gives for the units the drawn statement names. The rest is stated for any
committed wire the registrant's reads cover.
-/

namespace FlockBridge

namespace ZkReg

open scoped ENNReal

open FlockSoundness.Discharge.Exec (Inputs)
open FlockSoundness.Discharge.ZkReg (laySh rowsSh LiveAcceptsZKR OwnReads regPaths)
open RegisteredReads (restrict drawnReads regVals regGates RegValsSound)

variable {n : ℕ} {I : Inputs} (ht : I.tags.typed = false) {mPts : ℕ} (dj : Finset (Fin n) → Lean.Json)
  {S₀ : Finset (Fin n)} (y₀ : FlockSoundness.Discharge.ZkLink.DrawSetupZK I mPts S₀ (dj S₀)) {Reg : Type}
  {c : Flock.Circuit} (hc : Flock.HmRow.parse (FlockSoundness.Discharge.ZkExec.zkTags I) I.circuitFile I.tables = .ok c)
  (hU : FlockSoundness.Built c.unit) {pub : Flock.Public}
  (hpub : Flock.HmRow.loadPublic (FlockSoundness.Discharge.ZkExec.zkTags I) c I.publicFile = .ok pub)
  (hscope : FlockSoundness.Discharge.Layout.scopeOk c = true)

/-- **The registrant's reads are the verifier's own at each accepted run's draw** (named hypothesis): at every run that
`flock-verify verify --zk --registered F` accepts (`own` F's ports), the registrant's reads at the commit strings the
run's drawn units read are the verifier's own there (`OwnReads`): rows of drawn instances, at ports of `own` named as
the circuit's, at the positions `own` gives for the units the drawn statement names. -/
def OwnAtDraws (own : Array Flock.Registered.Port)
    (rd : FlockSoundness.Registered.Reads (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub))
      (rowsSh ht dj y₀ hc hU hpub hscope)) : Prop :=
  ∀ o : Reg × Finset (Fin n) × FlockSoundness.Discharge.Exec.Transcript, LiveAcceptsZKR I own o →
    OwnReads own c pub o.2.2
      (restrict rd (drawnReads (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)) o.2.1)).isRead rd.port rd.readAt

/-- **Registered reads of a `--zk --registered` session are sound.** In any game `G` on the live audit's outcomes (the
registration, the draw and the transcript), played by `σ`: if the registrant's reads `rd` are the verifier's own at each
accepted run's draw (`OwnAtDraws`), the registered-path finder meets `cr/sha-512` at `qR`, and the registrant's values
hold for `Xc` up to `δ` (`RegValsSound`), then registered reads are sound for `Xc` against `regVals rd` on
`regGates rd`, up to `qR²/2^513 + δ`. -/
theorem regReadSound_zk (own : Array Flock.Registered.Port)
    (hTags : FlockSoundness.Discharge.DecodeZL.TagsOk I)
    (G : FlockSoundness.Game (Reg × Finset (Fin n) × FlockSoundness.Discharge.Exec.Transcript))
    (σ : FlockSoundness.Game.Strategy G)
    (rd : FlockSoundness.Registered.Reads (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub))
      (rowsSh ht dj y₀ hc hU hpub hscope))
    (hown : OwnAtDraws ht dj y₀ hc hU hpub hscope own rd)
    (Xc : Fin ((FlockSoundness.Discharge.Placed.keyProg c hU n).circuit
      (FlockSoundness.Discharge.Layout.outsOf c hU n)).N → Bool) {qR : ℝ}
    (hR : (FlockSoundness.StrictCR.oneRun G σ fun o =>
      (restrict rd (drawnReads (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)) o.2.1)).pick o.1
        (regPaths c pub o.2.2 rd.port)).CR FlockSoundness.Refine.H512 qR)
    {δ : ℝ≥0∞} (hvals : RegValsSound G σ (LiveAcceptsZKR I own) (fun o => o.2.1) (fun o => o.1) rd Xc δ) :
    Registered.RegReadSound ((FlockSoundness.Discharge.Placed.keyProg c hU n).partition
        (FlockSoundness.Discharge.Layout.outsOf c hU n)) G σ (LiveAcceptsZKR I own) (fun o => o.2.1) Xc
      (regVals rd) (regGates rd) (ENNReal.ofReal (qR ^ 2 / 2 ^ (513 : ℝ)) + δ) :=
  RegisteredReads.regReadSound_of_checked G σ (LiveAcceptsZKR I own) (fun o => o.2.1) (fun o => o.1)
    (fun o => regPaths c pub o.2.2 rd.port) rd Xc
    (fun o hacc => FlockSoundness.Discharge.ZkReg.checked_of_accepts I ht own dj y₀ hc hU hpub hscope hTags o hacc _
      (hown o hacc)) hR hvals

end ZkReg

end FlockBridge
