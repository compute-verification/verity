import FlockSoundness.Discharge.ZkReg
import FlockBridge.RegisteredReads

/-!
# Registered reads of a `--zk --registered` session

`RegisteredReads.regReadSound_of_checked` at `flock-verify verify --zk --registered F` sessions
(`ZkReg.LiveAcceptsZKR`), on the public plan's layout and rows (`ZkReg.laySh`, `ZkReg.rowsSh`), where the registered
reads are read. The registered check at every accepted outcome of the live audit is `ZkReg.checked_of_accepts`, for the
registrant's reads at the commit strings the outcome's drawn units read, when those are the verifier's own there
(`OwnAtDraws`, named). Under any law and live session, what remains is `cr/sha-512` for the registered-path finder and
the registrant's values (`RegisteredReads.RegValsSound`): `regReadSound_zk`.

In the live audit at `zk_session_soundR`'s committed values (`ZkLink.XplurZC` at `Composed.simZ … τ`), the registrant's
values are `ZkReg.zk_session_regValsR`, under `cr/sha-512` for the registered-value finders (`ZkReg.RegCRZC`, named),
and the drawn wrong units are `ZkReg.zk_session_drawnR`: `regReadSound_zk_session` is the registered reads there, and
`registered_zk_session` is `Registered.registered_core` at that session, in the core game.

`OwnAtDraws` holds the executable's restriction: the verifier reads only rows of the drawn instances, at `F`'s ports
named as the circuit's, at the positions `F` gives for the units the drawn statement names. It is asked at the live
audit's outcomes `(R, L.draw ω, wr (L.draw ω) R lo)` only, where the transcript's recorded draw is the law's: at
an arbitrary triple the draw need not be the transcript's, and `OwnReads` needs each read's unit in the transcript's
draw. That is the shape of C-Flock's `ZkReg.ownAtDraws_of_check` (not merged here yet), which proves it at
`rd := ZkReg.ownReads … own …`, whose restriction to a draw `S` is `ZkReg.drawnOwn … S`, from `hRec` and
`(hDraw ω).1`. The rest is stated for any committed wire the registrant's reads cover.

Every theorem here takes `hscope : Layout.scopeOk c = true`, whose `v1Ports` asks every port to be a `sha512/row/v1`
row; every port of a `row_schemas` manifest is `v2`, so these theorems are vacuous for PoUW's circuits until C-Flock's
scope widens to them.
-/

namespace FlockBridge

namespace ZkReg

open scoped ENNReal

open FlockSoundness.Discharge.Exec (Inputs auditLive liveOf)
open FlockSoundness.Discharge.ZkReg (laySh rowsSh LiveAcceptsZKR OwnReads regPaths)
open RegisteredReads (restrict drawnReads regVals regGates RegValsSound)

variable {n : ℕ} {I : Inputs} (ht : I.tags.typed = false) {mPts : ℕ} (dj : Finset (Fin n) → Lean.Json)
  {S₀ : Finset (Fin n)} (y₀ : FlockSoundness.Discharge.ZkLink.DrawSetupZK I mPts S₀ (dj S₀)) {Reg : Type}
  {c : Flock.Circuit} (hc : Flock.HmRow.parse (FlockSoundness.Discharge.ZkExec.zkTags I) I.circuitFile I.tables = .ok c)
  (hU : FlockSoundness.Built c.unit) {pub : Flock.Public}
  (hpub : Flock.HmRow.loadPublic (FlockSoundness.Discharge.ZkExec.zkTags I) c I.publicFile = .ok pub)
  (hscope : FlockSoundness.Discharge.Layout.scopeOk c = true)

/-- **The registrant's reads are the verifier's own at each accepted live outcome's draw** (named hypothesis): at every
outcome of the live audit under the law `L` and the live session `wr` (a registration `R`, the law's draw at `ω`, and
the transcript `wr` writes there at a live outcome `lo`) that `flock-verify verify --zk --registered F` accepts (`own`
F's ports), the registrant's reads at the commit strings the drawn units read are the verifier's own there
(`OwnReads`): rows of drawn instances, at ports of `own` named as the circuit's, at the positions `own` gives for the
units the drawn statement names. -/
def OwnAtDraws (own : Array Flock.Registered.Port) (L : FlockSoundness.Audit.Law n)
    (wr : Finset (Fin n) → Reg → FlockSoundness.Refine.LiveOut mPts → FlockSoundness.Discharge.Exec.Transcript)
    (rd : FlockSoundness.Registered.Reads (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub))
      (rowsSh ht dj y₀ hc hU hpub hscope)) : Prop :=
  ∀ (R : Reg) (ω : L.Ω) (lo : FlockSoundness.Refine.LiveOut mPts),
    LiveAcceptsZKR I own (R, L.draw ω, wr (L.draw ω) R lo) →
    OwnReads own c pub (wr (L.draw ω) R lo)
      (restrict rd (drawnReads (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)) (L.draw ω))).isRead rd.port rd.readAt

/-- **Registered reads of a `--zk --registered` session are sound.** In the live audit under any law `L` and live
session `wr` (outcomes: the registration, the draw and the transcript), played by `σ`: if the registrant's reads `rd`
are the verifier's own at each accepted outcome's draw (`OwnAtDraws`), the registered-path finder meets `cr/sha-512` at
`qR`, and the registrant's values hold for `Xc` up to `δ` (`RegValsSound`), then registered reads are sound for `Xc`
against `regVals rd` on `regGates rd`, up to `qR²/2^513 + δ`. -/
theorem regReadSound_zk (own : Array Flock.Registered.Port)
    (hTags : FlockSoundness.Discharge.DecodeZL.TagsOk I) (L : FlockSoundness.Audit.Law n)
    (wr : Finset (Fin n) → Reg → FlockSoundness.Refine.LiveOut mPts → FlockSoundness.Discharge.Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    (σ : FlockSoundness.Game.Strategy (auditLive L Reg (liveOf dg mPts N wr)))
    (rd : FlockSoundness.Registered.Reads (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub))
      (rowsSh ht dj y₀ hc hU hpub hscope))
    (hown : OwnAtDraws ht dj y₀ hc hU hpub hscope own L wr rd)
    (Xc : Fin ((FlockSoundness.Discharge.Placed.keyProg c hU n).circuit
      (FlockSoundness.Discharge.Layout.outsOf c hU n)).N → Bool) {qR : ℝ}
    (hR : (FlockSoundness.StrictCR.oneRun (auditLive L Reg (liveOf dg mPts N wr)) σ fun o =>
      (restrict rd (drawnReads (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)) o.2.1)).pick o.1
        (regPaths c pub o.2.2 rd.port)).CR FlockSoundness.Refine.H512 qR)
    {δ : ℝ≥0∞} (hvals : RegValsSound (auditLive L Reg (liveOf dg mPts N wr)) σ (LiveAcceptsZKR I own)
      (fun o => o.2.1) (fun o => o.1) rd Xc δ) :
    Registered.RegReadSound ((FlockSoundness.Discharge.Placed.keyProg c hU n).partition
        (FlockSoundness.Discharge.Layout.outsOf c hU n)) (auditLive L Reg (liveOf dg mPts N wr)) σ
      (LiveAcceptsZKR I own) (fun o => o.2.1) Xc
      (regVals rd) (regGates rd) (ENNReal.ofReal (qR ^ 2 / 2 ^ (513 : ℝ)) + δ) := by
  have hchk : ∀ o : Reg × Finset (Fin n) × FlockSoundness.Discharge.Exec.Transcript,
      FlockSoundness.Game.Out (auditLive L Reg (liveOf dg mPts N wr)) o → LiveAcceptsZKR I own o →
      (restrict rd (drawnReads (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)) o.2.1)).Checked o.1
        (regPaths c pub o.2.2 rd.port) := by
    intro o hout hacc
    obtain ⟨R, ω, lo, rfl⟩ := RegisteredReads.out_auditLive hout
    exact FlockSoundness.Discharge.ZkReg.checked_of_accepts I ht own dj y₀ hc hU hpub hscope hTags _ hacc
      (restrict rd (drawnReads (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)) (L.draw ω))) (hown R ω lo hacc)
  exact RegisteredReads.regReadSound_of_checked (lay := laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub))
    (rs := rowsSh ht dj y₀ hc hU hpub hscope) (auditLive L Reg (liveOf dg mPts N wr)) σ (LiveAcceptsZKR I own)
    (fun o => o.2.1) (fun o => o.1) (fun o => regPaths c pub o.2.2 rd.port) rd Xc hchk hR hvals

section Session

open FlockSoundness (execArith)
open FlockSoundness.Game (Strategy prob prob_mono)
open FlockSoundness.Audit (reg cont)
open FlockSoundness.Refine (LiveOut H512)
open FlockSoundness.StrictCR (oneRun)
open FlockSoundness.Discharge.Exec (auditLive liveOf)
open FlockSoundness.Discharge.ZkExec (hs512 runsZC_live)
open FlockSoundness.Discharge.ZkLink (DrawOkZK RecordsLiveZK rateZ TableCRZ tablesAtZK_live zcStrat XplurZC ksAvgStrictZ
  linkBoundZC)
open FlockSoundness.Discharge.ZkBind (planZAt tabZAt)
open FlockSoundness.Discharge.Composed (simZ decodesOnZK_of_runsOnZC)
open FlockSoundness.Discharge.Placed (keyProg)
open FlockSoundness.Discharge.Layout (outsOf)
open FlockSoundness.Discharge.DecodeZL (TagsOk)
open FlockSoundness.Discharge.ZkReg (rsZAtS rsZAtS_computes RegCRZC regOpenZ regTermZC zk_session_regValsR
  zk_session_drawnR)

/-- **Registered reads of a `--zk --registered` session are sound, at the committed values its soundness reads.**
`regReadSound_zk` in the live audit at `zk_session_soundR`'s committed values (`XplurZC` at `simZ … τ`), with the
registrant's values from `zk_session_regValsR`. On that theorem's assumptions, if the registrant's reads are the
verifier's own at each accepted live outcome's draw (`OwnAtDraws`) and `cr/sha-512` holds for the registered-value
finders (`hW`) and the registered-path finder (`hR`), registered reads are sound up to
`qR²/2^513 + ε_ks + δ_link + δ_reg`. -/
theorem regReadSound_zk_session (own : Array Flock.Registered.Port)
    (wr : Finset (Fin n) → Reg → LiveOut mPts → FlockSoundness.Discharge.Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    {m : ℕ} (St : Fin m → Flock.Draw.Stratum) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hDraw : ∀ ω, DrawOkZK I dj mPts
      ((FlockSoundness.Audit.Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω))
    (hTags : TagsOk I) (hRec : RecordsLiveZK I dj wr)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZAt (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t')
    (rd : FlockSoundness.Registered.Reads (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub))
      (rowsSh ht dj y₀ hc hU hpub hscope))
    (hown : OwnAtDraws ht dj y₀ hc hU hpub hscope own
      (FlockSoundness.Audit.Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) wr rd)
    (τ : Strategy (auditLive (FlockSoundness.Audit.Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (liveOf dg mPts N wr)))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ execArith hs512 (planZAt dj y₀
        ((FlockSoundness.Audit.Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _
          (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ))) j
      (zcStrat execArith hs512 (planZAt dj y₀)
        (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _
          (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ) ω)) qF qS)
    {qW : ℝ}
    (hW : RegCRZC execArith hs512 ((keyProg c hU n).partition (outsOf c hU n)) (planZAt dj y₀) (tabZAt dj y₀)
      ((rsZAtS ht dj y₀ hc hU hpub hscope).bindingZ (rsZAtS_computes ht dj y₀ hc hU hpub hscope))
      (regOpenZ ht dj y₀ hc hU hpub hscope rd)
      (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _
        (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ))
      (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _
        (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ)) k Rw qW)
    {qR : ℝ}
    (hR : (oneRun (auditLive (FlockSoundness.Audit.Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
        (liveOf dg mPts N wr)) τ fun o =>
      (restrict rd (drawnReads (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)) o.2.1)).pick o.1
        (regPaths c pub o.2.2 rd.port)).CR H512 qR) :
    Registered.RegReadSound ((keyProg c hU n).partition (outsOf c hU n))
      (auditLive (FlockSoundness.Audit.Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
        (liveOf dg mPts N wr)) τ (LiveAcceptsZKR I own) (fun o => o.2.1)
      (XplurZC execArith hs512 ((keyProg c hU n).partition (outsOf c hU n)) (planZAt dj y₀) (tabZAt dj y₀)
        ((rsZAtS ht dj y₀ hc hU hpub hscope).bindingZ (rsZAtS_computes ht dj y₀ hc hU hpub hscope))
        (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _
          (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ))
        (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _
          (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ)) k Rw)
      (regVals rd) (regGates rd)
      (ENNReal.ofReal (qR ^ 2 / 2 ^ (513 : ℝ)) +
        (ksAvgStrictZ (planZAt dj y₀)
            (FlockSoundness.Audit.Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
            (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _
              (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ)) +
          linkBoundZC execArith hs512 ((keyProg c hU n).partition (outsOf c hU n)) (planZAt dj y₀) (tabZAt dj y₀)
            ((rsZAtS ht dj y₀ hc hU hpub hscope).bindingZ (rsZAtS_computes ht dj y₀ hc hU hpub hscope))
            k Rw M ρ t' (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr
              _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ))
            (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _
              (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ)) +
          regTermZC execArith ((keyProg c hU n).partition (outsOf c hU n)) (planZAt (Reg := Reg) dj y₀)
            (tabZAt dj y₀) ((rsZAtS ht dj y₀ hc hU hpub hscope).bindingZ (rsZAtS_computes ht dj y₀ hc hU hpub hscope))
            (FlockSoundness.Audit.Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k Rw ρ qW)) := by
  have hvals := zk_session_regValsR I ht own dj y₀ wr dg N hc hU hpub hscope St Kd pop Lb src hDraw hTags hRec hk hRw
    hM hρ hr ht' rd τ hCR hW
  exact regReadSound_zk ht dj y₀ hc hU hpub hscope own hTags _ wr dg N τ rd hown _ hR hvals

/-- **Soundness against registered values of a `--zk --registered` session, in the core game.** For every strategy `τ`
of the live audit, at its core strategy: on `zk_session_soundR`'s assumptions without A3 and the count curve, if the
registrant's reads are the verifier's own at each accepted live outcome's draw (`OwnAtDraws`) and `cr/sha-512` holds for
the knowledge (`hCR`), registered-value (`hW`) and registered-path (`hR`) finders, some core transcript makes an
accepting run draw a wrong unit or be off the registered values `regVals rd` on the registered gates `regGates rd` with
probability at most `(ε_ks + δ_link) + (qR²/2^513 + ε_ks + δ_link + δ_reg)`. `Registered.registered_core`, with the
drawn units' bound from `zk_session_drawnR` and the registered reads from `regReadSound_zk_session`. -/
theorem registered_zk_session (own : Array Flock.Registered.Port)
    (wr : Finset (Fin n) → Reg → LiveOut mPts → FlockSoundness.Discharge.Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    {m : ℕ} (St : Fin m → Flock.Draw.Stratum) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hDraw : ∀ ω, DrawOkZK I dj mPts
      ((FlockSoundness.Audit.Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω))
    (hTags : TagsOk I) (hRec : RecordsLiveZK I dj wr)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZAt (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t')
    (rd : FlockSoundness.Registered.Reads (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub))
      (rowsSh ht dj y₀ hc hU hpub hscope))
    (hown : OwnAtDraws ht dj y₀ hc hU hpub hscope own
      (FlockSoundness.Audit.Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) wr rd)
    (τ : Strategy (auditLive (FlockSoundness.Audit.Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (liveOf dg mPts N wr)))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ execArith hs512 (planZAt dj y₀
        ((FlockSoundness.Audit.Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _
          (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ))) j
      (zcStrat execArith hs512 (planZAt dj y₀)
        (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _
          (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ) ω)) qF qS)
    {qW : ℝ}
    (hW : RegCRZC execArith hs512 ((keyProg c hU n).partition (outsOf c hU n)) (planZAt dj y₀) (tabZAt dj y₀)
      ((rsZAtS ht dj y₀ hc hU hpub hscope).bindingZ (rsZAtS_computes ht dj y₀ hc hU hpub hscope))
      (regOpenZ ht dj y₀ hc hU hpub hscope rd)
      (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _
        (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ))
      (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _
        (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ)) k Rw qW)
    {qR : ℝ}
    (hR : (oneRun (auditLive (FlockSoundness.Audit.Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
        (liveOf dg mPts N wr)) τ fun o =>
      (restrict rd (drawnReads (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)) o.2.1)).pick o.1
        (regPaths c pub o.2.2 rd.port)).CR H512 qR) :
    ∃ X' : (Circuit.toCore ((keyProg c hU n).circuit (outsOf c hU n))).Transcript,
      Verity.Game.prob (fun o => LiveAcceptsZKR I own o ∧
          (¬ Disjoint ((Partition.toCore ((keyProg c hU n).partition (outsOf c hU n))).wrong X') o.2.1 ∨
            ∃ g ∈ regGates rd, X' g ≠ Circuit.Transcript.toCore (regVals rd) g))
        (Game.toCore (auditLive (FlockSoundness.Audit.Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
          (liveOf dg mPts N wr))) (Game.Strategy.toCore _ τ) ≤
      (ksAvgStrictZ (planZAt dj y₀)
          (FlockSoundness.Audit.Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _
            (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ)) +
        linkBoundZC execArith hs512 ((keyProg c hU n).partition (outsOf c hU n)) (planZAt dj y₀) (tabZAt dj y₀)
          ((rsZAtS ht dj y₀ hc hU hpub hscope).bindingZ (rsZAtS_computes ht dj y₀ hc hU hpub hscope))
          k Rw M ρ t' (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr
            _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ))
          (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _
            (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ))) +
      (ENNReal.ofReal (qR ^ 2 / 2 ^ (513 : ℝ)) +
        (ksAvgStrictZ (planZAt dj y₀)
            (FlockSoundness.Audit.Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
            (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _
              (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ)) +
          linkBoundZC execArith hs512 ((keyProg c hU n).partition (outsOf c hU n)) (planZAt dj y₀) (tabZAt dj y₀)
            ((rsZAtS ht dj y₀ hc hU hpub hscope).bindingZ (rsZAtS_computes ht dj y₀ hc hU hpub hscope))
            k Rw M ρ t' (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr
              _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ))
            (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _
              (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ)) +
          regTermZC execArith ((keyProg c hU n).partition (outsOf c hU n)) (planZAt (Reg := Reg) dj y₀)
            (tabZAt dj y₀) ((rsZAtS ht dj y₀ hc hU hpub hscope).bindingZ (rsZAtS_computes ht dj y₀ hc hU hpub hscope))
            (FlockSoundness.Audit.Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k Rw ρ qW)) := by
  obtain ⟨σ, rfl⟩ : ∃ σ, τ = Game.Strategy.ofCore _ σ :=
    ⟨Game.Strategy.toCore _ τ, (Game.Strategy.ofCore_toCore _ τ).symm⟩
  rw [Game.Strategy.toCore_ofCore]
  have hv := regReadSound_zk_session ht dj y₀ hc hU hpub hscope own wr dg N St Kd pop Lb src hDraw hTags hRec hk hRw
    hM hρ hr ht' rd hown (Game.Strategy.ofCore _ σ) hCR hW hR
  rw [← Circuit.Transcript.ofCore_toCore (regVals rd)] at hv
  refine Registered.registered_core _ _ _ _ _ _ σ _ False ?_ hv
  exact le_trans (prob_mono (fun o h => ⟨h.1, h.2.resolve_right id⟩) _ _)
    (zk_session_drawnR I ht own dj y₀ wr dg N hc hU hpub hscope St Kd pop Lb src hDraw hTags hRec hk hRw hM hρ hr ht'
      (Game.Strategy.ofCore _ σ) hCR)

end Session

end ZkReg

end FlockBridge
