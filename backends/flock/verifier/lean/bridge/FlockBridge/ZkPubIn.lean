import FlockSoundness.Discharge.ZkPubIn.SoundCustody
import FlockBridge.RegisteredReads

/-!
# Registered reads of a `--zk --registered R --public-inputs F` session

`FlockBridge.ZkReg` with the verifier's public inputs, at C-Flock's hidden-output sessions with public input ports
(`ZkPubIn.LiveAcceptsZKRP`, `rports` R's ports, `own` F's bytes), on the public plan's layout and rows
(`ZkPubIn.layShHP`, `ZkPubIn.rsShHP`). The registered check at every accepted outcome of the live audit is
`ZkPubIn.checked_of_acceptsHP` (`zk_session_soundHRP`'s second conjunct), for the registrant's reads at the commit
strings the outcome's drawn units read, when those are the verifier's own there (`OwnAtDrawsHP`, named):
`regReadSound_zkHP`, under any law and live session.

In the live audit, the committed values are `zk_session_soundHRP`'s public transcript (`Xpub (pubOnes …) (pubZeros …)
XplurZC` at `ZkPubIn.simZA … τ`, on every public-input gate the verifier's own bit):

* `regReadSound_zk_sessionHP`: the registered reads there. The registrant's values at that transcript
  (`RegisteredReads.RegValsSound`) stay named: C-Flock has no theorem for them at `Xpub` yet.
* `registered_zk_sessionHP`: `Registered.registered_core` there, in the core game, with the drawn wrong units from
  `zk_session_soundHRP`'s count conjunct at `K₀ = 1` (a drawn wrong unit is at least one wrong unit), so
  `η = (miss 1 + ε_ks + δ_link) + (qR²/2^513 + δ)`. `miss 1` is the largest probability that the draw misses one given
  unit, close to 1 under a sampled law: the bound holds but is not a useful number. C-Flock states the hidden-output
  public-input session only as a count curve, and no drawn form yet.
* `registered_zk_sessionHP_custody`: the same on `zk_session_soundHRP_custody`'s one record hypothesis,
  `RecordCustodyZKP` (`live-verifier`), at the link points flock-verify sets up at the law's draws.

**The joint seam.** When C-Flock's joint registered form lands at this session (the drawn wrong units of the public
transcript and the registrant's values there, in one event at `ε_ks + δ_link + δ_reg`), `registered_zk_sessionHP`
becomes `RegisteredReads.registered_of_checked_joint` at that bound: `hchk` from `checked_of_acceptsHP`, `hjoint`
(`RegisteredReads.RegJointSound`) from the joint form, and `η = qR²/2^513 + ε_ks + δ_link + δ_reg`. `hvals`, the count
conjunct and `miss 1` drop out, and `regReadSound_zk_sessionHP`'s `hvals` follows from the joint form by `prob_mono`.

**Vacuous for PoUW's circuits today.** Everything here is stated for any circuit the verifier parses (`hc`, `hpi`,
`hscope`), and nothing here claims it is non-vacuous for PoUW's circuits. Every theorem takes
`hscope : Layout.scopeOk c = true`, whose `v1Ports` asks every port, input or output, to be a `sha512/row/v1` row, and
`HmRow.rowPorts` makes every port of a `row_schemas` manifest `v2`, row-seg included: no PoUW circuit meets `hscope`,
so these theorems say nothing about PoUW's circuits until C-Flock's scope widens to v2 and row-seg ports. They are
generic in the circuit and apply once it does. Even then, C-Flock's hidden-output soundness for v2 output rows is
proved for one output port per output row and whole-SHA-512-block ports, and PoUW's ok rows (one word), tile digests
(16 words) and the v2 data row of `row_b` (a run of 2 or 3 ports) are neither; if the widened `zk_session_soundHRP`
does not reach exact v2 output rows, covering PoUW's circuits depends on per-output-row soundness for exact runs with
a partial last block (C-Flock's option B).

**`OwnAtDrawsHP` and reads defined from the registered file.** `OwnAtDrawsHP` is asked at the live audit's outcomes
`(R, L.draw ω, wr (L.draw ω) R lo)` only, as `ZkReg.OwnAtDraws` is: `OwnReadsHP` needs each read's unit in the
transcript's recorded draw, which is the law's draw there and need not be at an arbitrary triple. That is the shape of
C-Flock's `ZkReg.ownAtDraws_of_check` (`cursor/own-at-draws-95d4`, not merged here yet), which proves the form
without public inputs at `rd := ZkReg.ownReads … own …` (restricted to a draw `S`, `ZkReg.drawnOwn … S`) from
`hRec` and `(hDraw ω).1`. A public-input analog at reads defined from R's ports (not stated yet) would discharge
`hown` here as `fun R ω lo hacc => … hRec (hDraw ω).1 R lo hacc …`, with no change to these statements: each holds
for every `rd`.
-/

namespace FlockBridge

namespace ZkPubIn

open scoped ENNReal

open FlockSoundness (execArith)
open FlockSoundness.Game (Strategy prob prob_mono)
open FlockSoundness.Audit (reg cont Xpub)
open FlockSoundness.Refine (LiveOut H512)
open FlockSoundness.StrictCR (oneRun)
open FlockSoundness.Discharge.Exec (Inputs auditLive liveOf mPtsOf)
open FlockSoundness.Discharge.ZkExec (hs512 zkTags)
open FlockSoundness.Discharge.ZkLink (rateZ TableCRZ zcStrat XplurZC ksAvgStrictZ linkBoundZC)
open FlockSoundness.Discharge.Layout (scopeOk scopeOk_spec outsOf)
open FlockSoundness.Discharge.HiddenDec (TagsOkH)
open FlockSoundness.Discharge.ZkPubIn (DrawSetupZKHP DrawOkZKHP RecordsLiveZKP planAtZKHP tabAtZKHP simZA AcceptsZKP
  decodesOnZKA_of_runsOnZCA runsZC_liveHP tablesAtZKHP_live keyProgP pubOnes pubZeros rsZHP rsZHP_computes layShHP
  rsShHP LiveAcceptsZKRP OwnReadsHP regPathsHP checked_of_acceptsHP zk_session_soundHRP drawOkZKHP_of_exec)
open RegisteredReads (restrict drawnReads regVals regGates RegValsSound)

section

variable {n : ℕ} {I : Inputs} (ht : I.tags.typed = false) {own : ByteArray} {mPts : ℕ}
  (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZKHP I own mPts S₀ (dj S₀)) {Reg : Type}
  {c : Flock.Circuit} (hc : Flock.HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c)
  (hU : FlockSoundness.Built c.unit) {pi : Flock.HmIn.Ports} (hpi : Flock.HmIn.parse c = .ok pi) {pub : Flock.Public}
  (hpub : Flock.HmIn.loadPublic (zkTags I) c pi I.publicFile = .ok pub) (hscope : scopeOk c = true)

/-- **The registrant's reads are the verifier's own at each accepted live outcome's draw, with public inputs** (named
hypothesis): at every outcome of the live audit under the law `L` and the live session `wr` (a registration `R`, the
law's draw at `ω`, and the transcript `wr` writes there at a live outcome `lo`) that
`flock-verify verify --zk --registered R --public-inputs F` accepts (`rports` R's ports, `own` F's bytes), the
registrant's reads at the commit strings the drawn units read are the verifier's own there (`OwnReadsHP`): rows of drawn
instances, at ports of `rports` named as the circuit's, at the positions `rports` gives for the units the drawn
statement `HmIn.drawn c pi pub` names. -/
def OwnAtDrawsHP (rports : Array Flock.Registered.Port) (L : FlockSoundness.Audit.Law n)
    (wr : Finset (Fin n) → Reg → LiveOut mPts → FlockSoundness.Discharge.Exec.Transcript)
    (rd : FlockSoundness.Registered.Reads (layShHP (Reg := Reg) dj y₀ ht hc hscope hU hpi (pub := pub))
      (rsShHP dj y₀ ht hc hscope hU hpi hpub)) : Prop :=
  ∀ (R : Reg) (ω : L.Ω) (lo : LiveOut mPts),
    LiveAcceptsZKRP I rports own (R, L.draw ω, wr (L.draw ω) R lo) →
    OwnReadsHP rports c pi pub (wr (L.draw ω) R lo)
      (restrict rd (drawnReads (layShHP (Reg := Reg) dj y₀ ht hc hscope hU hpi (pub := pub)) (L.draw ω))).isRead
      rd.port rd.readAt

/-- **Registered reads of a `--zk --registered R --public-inputs F` session are sound.** In the live audit under any
law `L` and live session `wr` (outcomes: the registration, the draw and the transcript), played by `σ`: if the
registrant's reads `rd` are the verifier's own at each accepted outcome's draw (`OwnAtDrawsHP`), the registered-path
finder meets `cr/sha-512` at `qR`, and the registrant's values hold for `Xc` up to `δ` (`RegValsSound`), then
registered reads are sound for `Xc` against `regVals rd` on `regGates rd`, up to `qR²/2^513 + δ`. -/
theorem regReadSound_zkHP (rports : Array Flock.Registered.Port) (hTags : TagsOkH I) (L : FlockSoundness.Audit.Law n)
    (wr : Finset (Fin n) → Reg → LiveOut mPts → FlockSoundness.Discharge.Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ) (σ : Strategy (auditLive L Reg (liveOf dg mPts N wr)))
    (rd : FlockSoundness.Registered.Reads (layShHP (Reg := Reg) dj y₀ ht hc hscope hU hpi (pub := pub))
      (rsShHP dj y₀ ht hc hscope hU hpi hpub))
    (hown : OwnAtDrawsHP ht dj y₀ hc hU hpi hpub hscope rports L wr rd)
    (Xc : Fin ((keyProgP c hU n pi).circuit (outsOf c hU n)).N → Bool) {qR : ℝ}
    (hR : (oneRun (auditLive L Reg (liveOf dg mPts N wr)) σ fun o =>
      (restrict rd (drawnReads (layShHP (Reg := Reg) dj y₀ ht hc hscope hU hpi (pub := pub)) o.2.1)).pick o.1
        (regPathsHP c pi pub o.2.2 rd.port)).CR H512 qR)
    {δ : ℝ≥0∞} (hvals : RegValsSound (auditLive L Reg (liveOf dg mPts N wr)) σ (LiveAcceptsZKRP I rports own)
      (fun o => o.2.1) (fun o => o.1) rd Xc δ) :
    Registered.RegReadSound ((keyProgP c hU n pi).partition (outsOf c hU n)) (auditLive L Reg (liveOf dg mPts N wr)) σ
      (LiveAcceptsZKRP I rports own) (fun o => o.2.1) Xc (regVals rd) (regGates rd)
      (ENNReal.ofReal (qR ^ 2 / 2 ^ (513 : ℝ)) + δ) := by
  have hchk : ∀ o : Reg × Finset (Fin n) × FlockSoundness.Discharge.Exec.Transcript,
      FlockSoundness.Game.Out (auditLive L Reg (liveOf dg mPts N wr)) o → LiveAcceptsZKRP I rports own o →
      (restrict rd (drawnReads (layShHP (Reg := Reg) dj y₀ ht hc hscope hU hpi (pub := pub)) o.2.1)).Checked o.1
        (regPathsHP c pi pub o.2.2 rd.port) := by
    intro o hout hacc
    obtain ⟨R, ω, lo, rfl⟩ := RegisteredReads.out_auditLive hout
    exact checked_of_acceptsHP I ht rports own dj y₀ hc hU hpi hpub hscope hTags _ hacc
      (restrict rd (drawnReads (layShHP (Reg := Reg) dj y₀ ht hc hscope hU hpi (pub := pub)) (L.draw ω)))
      (hown R ω lo hacc)
  exact RegisteredReads.regReadSound_of_checked (lay := layShHP (Reg := Reg) dj y₀ ht hc hscope hU hpi (pub := pub))
    (rs := rsShHP dj y₀ ht hc hscope hU hpi hpub) (auditLive L Reg (liveOf dg mPts N wr)) σ
    (LiveAcceptsZKRP I rports own) (fun o => o.2.1) (fun o => o.1) (fun o => regPathsHP c pi pub o.2.2 rd.port) rd Xc
    hchk hR hvals

/-- **Registered reads of a `--zk --registered R --public-inputs F` session are sound, at the public transcript its
soundness reads.** `regReadSound_zkHP` in the live audit at `zk_session_soundHRP`'s public transcript (`Xpub` of the
verifier's own bits over `XplurZC` at `simZA … τ`): if the registrant's reads are the verifier's own at each accepted
live outcome's draw (`OwnAtDrawsHP`), `cr/sha-512` holds for the registered-path finder (`hR`), and the registrant's
values hold there up to `δ` (`hvals`, named until C-Flock's joint form at this session), registered reads are sound up
to `qR²/2^513 + δ`. -/
theorem regReadSound_zk_sessionHP (rports : Array Flock.Registered.Port)
    (wr : Finset (Fin n) → Reg → LiveOut mPts → FlockSoundness.Discharge.Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    {m : ℕ} (St : Fin m → Flock.Draw.Stratum) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hDraw : ∀ ω, DrawOkZKHP I own dj mPts
      ((FlockSoundness.Audit.Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω))
    (hTags : TagsOkH I) (hRec : RecordsLiveZKP I own dj wr) {k Rw : ℕ}
    (rd : FlockSoundness.Registered.Reads (layShHP (Reg := Reg) dj y₀ ht hc hscope hU hpi (pub := pub))
      (rsShHP dj y₀ ht hc hscope hU hpi hpub))
    (hown : OwnAtDrawsHP ht dj y₀ hc hU hpi hpub hscope rports
      (FlockSoundness.Audit.Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) wr rd)
    (τ : Strategy (auditLive (FlockSoundness.Audit.Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (liveOf dg mPts N wr)))
    {qR : ℝ}
    (hR : (oneRun (auditLive (FlockSoundness.Audit.Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
        (liveOf dg mPts N wr)) τ fun o =>
      (restrict rd (drawnReads (layShHP (Reg := Reg) dj y₀ ht hc hscope hU hpi (pub := pub)) o.2.1)).pick o.1
        (regPathsHP c pi pub o.2.2 rd.port)).CR H512 qR)
    {δ : ℝ≥0∞}
    (hvals : RegValsSound (auditLive (FlockSoundness.Audit.Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop)
        Reg (liveOf dg mPts N wr)) τ (LiveAcceptsZKRP I rports own) (fun o => o.2.1) (fun o => o.1) rd
      (Xpub (pubOnes c hU n pi own) (pubZeros c hU n pi own)
        (XplurZC execArith hs512 ((keyProgP c hU n pi).partition (outsOf c hU n)) (planAtZKHP dj y₀)
          (tabAtZKHP dj y₀)
          ((rsZHP dj y₀ ht hc hscope hU hpi hpub).bindingZ (rsZHP_computes dj y₀ ht hc hscope hU hpi hpub))
          (reg (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _
            (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (scopeOk_spec hU hscope).2.1 dj y₀ wr _
              tablesAtZKHP_live hTags hRec hDraw)) τ))
          (cont (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith
            hs512 _ (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (scopeOk_spec hU hscope).2.1 dj y₀ wr _
              tablesAtZKHP_live hTags hRec hDraw)) τ)) k Rw)) δ) :
    Registered.RegReadSound ((keyProgP c hU n pi).partition (outsOf c hU n))
      (auditLive (FlockSoundness.Audit.Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
        (liveOf dg mPts N wr)) τ (LiveAcceptsZKRP I rports own) (fun o => o.2.1)
      (Xpub (pubOnes c hU n pi own) (pubZeros c hU n pi own)
        (XplurZC execArith hs512 ((keyProgP c hU n pi).partition (outsOf c hU n)) (planAtZKHP dj y₀)
          (tabAtZKHP dj y₀)
          ((rsZHP dj y₀ ht hc hscope hU hpi hpub).bindingZ (rsZHP_computes dj y₀ ht hc hscope hU hpi hpub))
          (reg (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _
            (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (scopeOk_spec hU hscope).2.1 dj y₀ wr _
              tablesAtZKHP_live hTags hRec hDraw)) τ))
          (cont (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith
            hs512 _ (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (scopeOk_spec hU hscope).2.1 dj y₀ wr _
              tablesAtZKHP_live hTags hRec hDraw)) τ)) k Rw))
      (regVals rd) (regGates rd) (ENNReal.ofReal (qR ^ 2 / 2 ^ (513 : ℝ)) + δ) :=
  regReadSound_zkHP ht dj y₀ hc hU hpi hpub hscope rports hTags _ wr dg N τ rd hown _ hR hvals

/-- **Soundness against registered values of a `--zk --registered R --public-inputs F` session, in the core game.** For
every strategy `τ` of the live audit, at its core strategy: on `zk_session_soundHRP`'s assumptions at `K₀ = 1`, if the
registrant's reads are the verifier's own at each accepted live outcome's draw (`OwnAtDrawsHP`), `cr/sha-512` holds for
the registered-path finder (`hR`), and the registrant's values hold at the public transcript up to `δ` (`hvals`, named),
some core transcript makes an accepting run draw a wrong unit or be off the registered values `regVals rd` on the
registered gates `regGates rd` with probability at most `(miss 1 + ε_ks + δ_link) + (qR²/2^513 + δ)`, where `miss 1` is
close to 1 under a sampled law. `Registered.registered_core`, with the drawn wrong units from `zk_session_soundHRP`'s
count conjunct at `K₀ = 1` and the registered reads from `regReadSound_zk_sessionHP`. -/
theorem registered_zk_sessionHP (rports : Array Flock.Registered.Port)
    (wr : Finset (Fin n) → Reg → LiveOut mPts → FlockSoundness.Discharge.Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (FlockSoundness.Audit.Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : FlockSoundness.Audit.Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : FlockSoundness.Assumptions.UniformRandomBytes Lb src)
    (hDraw : ∀ ω, DrawOkZKHP I own dj mPts
      ((FlockSoundness.Audit.Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω))
    (hTags : TagsOkH I) (hRec : RecordsLiveZKP I own dj wr)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planAtZKHP (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t')
    (rd : FlockSoundness.Registered.Reads (layShHP (Reg := Reg) dj y₀ ht hc hscope hU hpi (pub := pub))
      (rsShHP dj y₀ ht hc hscope hU hpi hpub))
    (hown : OwnAtDrawsHP ht dj y₀ hc hU hpi hpub hscope rports
      (FlockSoundness.Audit.Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) wr rd)
    (τ : Strategy (auditLive (FlockSoundness.Audit.Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (liveOf dg mPts N wr)))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ execArith hs512 (planAtZKHP dj y₀
        ((FlockSoundness.Audit.Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _
          (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (scopeOk_spec hU hscope).2.1 dj y₀ wr _
            tablesAtZKHP_live hTags hRec hDraw)) τ))) j
      (zcStrat execArith hs512 (planAtZKHP dj y₀)
        (cont (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _
          (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (scopeOk_spec hU hscope).2.1 dj y₀ wr _
            tablesAtZKHP_live hTags hRec hDraw)) τ) ω)) qF qS)
    {qR : ℝ}
    (hR : (oneRun (auditLive (FlockSoundness.Audit.Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
        (liveOf dg mPts N wr)) τ fun o =>
      (restrict rd (drawnReads (layShHP (Reg := Reg) dj y₀ ht hc hscope hU hpi (pub := pub)) o.2.1)).pick o.1
        (regPathsHP c pi pub o.2.2 rd.port)).CR H512 qR)
    {δ : ℝ≥0∞}
    (hvals : RegValsSound (auditLive (FlockSoundness.Audit.Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop)
        Reg (liveOf dg mPts N wr)) τ (LiveAcceptsZKRP I rports own) (fun o => o.2.1) (fun o => o.1) rd
      (Xpub (pubOnes c hU n pi own) (pubZeros c hU n pi own)
        (XplurZC execArith hs512 ((keyProgP c hU n pi).partition (outsOf c hU n)) (planAtZKHP dj y₀)
          (tabAtZKHP dj y₀)
          ((rsZHP dj y₀ ht hc hscope hU hpi hpub).bindingZ (rsZHP_computes dj y₀ ht hc hscope hU hpi hpub))
          (reg (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _
            (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (scopeOk_spec hU hscope).2.1 dj y₀ wr _
              tablesAtZKHP_live hTags hRec hDraw)) τ))
          (cont (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith
            hs512 _ (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (scopeOk_spec hU hscope).2.1 dj y₀ wr _
              tablesAtZKHP_live hTags hRec hDraw)) τ)) k Rw)) δ) :
    ∃ X' : (Circuit.toCore ((keyProgP c hU n pi).circuit (outsOf c hU n))).Transcript,
      Verity.Game.prob (fun o => LiveAcceptsZKRP I rports own o ∧
          (¬ Disjoint ((Partition.toCore ((keyProgP c hU n pi).partition (outsOf c hU n))).wrong X') o.2.1 ∨
            ∃ g ∈ regGates rd, X' g ≠ Circuit.Transcript.toCore (regVals rd) g))
        (Game.toCore (auditLive (FlockSoundness.Audit.Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
          (liveOf dg mPts N wr))) (Game.Strategy.toCore _ τ) ≤
      (FlockSoundness.Audit.Law.stratified σs ks hks).miss 1 +
        ksAvgStrictZ (planAtZKHP dj y₀) (FlockSoundness.Audit.Law.execOS (n := n) src (.stratified Kd (Array.ofFn St))
          pop) k qF qS
          (reg (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _
            (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (scopeOk_spec hU hscope).2.1 dj y₀ wr _
              tablesAtZKHP_live hTags hRec hDraw)) τ)) +
        linkBoundZC execArith hs512 ((keyProgP c hU n pi).partition (outsOf c hU n)) (planAtZKHP dj y₀)
          (tabAtZKHP dj y₀)
          ((rsZHP dj y₀ ht hc hscope hU hpi hpub).bindingZ (rsZHP_computes dj y₀ ht hc hscope hU hpi hpub))
          k Rw M ρ t'
          (reg (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _
            (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (scopeOk_spec hU hscope).2.1 dj y₀ wr _
              tablesAtZKHP_live hTags hRec hDraw)) τ))
          (cont (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith
            hs512 _ (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (scopeOk_spec hU hscope).2.1 dj y₀ wr _
              tablesAtZKHP_live hTags hRec hDraw)) τ)) +
      (ENNReal.ofReal (qR ^ 2 / 2 ^ (513 : ℝ)) + δ) := by
  obtain ⟨σ, rfl⟩ : ∃ σ, τ = Game.Strategy.ofCore _ σ :=
    ⟨Game.Strategy.toCore _ τ, (Game.Strategy.ofCore_toCore _ τ).symm⟩
  rw [Game.Strategy.toCore_ofCore]
  have hv := regReadSound_zk_sessionHP ht dj y₀ hc hU hpi hpub hscope rports wr dg N St Kd pop Lb src hDraw hTags hRec
    rd hown (Game.Strategy.ofCore _ σ) hR hvals
  rw [← Circuit.Transcript.ofCore_toCore (regVals rd)] at hv
  refine Registered.registered_core _ _ _ _ _ _ σ _ False ?_ hv
  refine le_trans (prob_mono (fun o h => ⟨h.1, ?_⟩) _ _)
    (zk_session_soundHRP I ht rports own dj y₀ wr dg N hc hU hpi hpub hscope σs ks hks St hS Kd pop Lb src hA3 hDraw
      hTags hRec hk hRw hM hρ hr ht' 1 (Game.Strategy.ofCore _ σ) hCR).1
  obtain ⟨u, hu, -⟩ := Finset.not_disjoint_iff.1 (h.2.resolve_right id)
  exact Finset.one_le_card.2 ⟨u, hu⟩

end

section Custody

variable {n : ℕ} {Reg : Type}

/-- **Soundness against registered values of a `--zk --registered R --public-inputs F` session, in the core game,
under `live-verifier`.** `registered_zk_sessionHP` with its two premises about the record, `hRec` (`RecordsLiveZKP`)
and `hDraw` (`DrawOkZKHP` at each of the law's draws), replaced by `zk_session_soundHRP_custody`'s one named
assumption, `RecordCustodyZKP`, at the link points flock-verify sets up at the law's draws
(`mPts := mPtsOf c.g (∑ s, ks s)`, `drawOkZKHP_of_exec`). -/
theorem registered_zk_sessionHP_custody {I : Inputs} (ht : I.tags.typed = false)
    (rports : Array Flock.Registered.Port) {own : ByteArray}
    {c : Flock.Circuit} (hc : Flock.HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c)
    (hU : FlockSoundness.Built c.unit) {pi : Flock.HmIn.Ports} (hpi : Flock.HmIn.parse c = .ok pi) {pub : Flock.Public}
    (hpub : Flock.HmIn.loadPublic (zkTags I) c pi I.publicFile = .ok pub) (hscope : scopeOk c = true)
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (FlockSoundness.Audit.Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : FlockSoundness.Audit.Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : FlockSoundness.Assumptions.UniformRandomBytes Lb src)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZKHP I own (mPtsOf c.g (∑ s, ks s)) S₀ (dj S₀))
    (wr : Finset (Fin n) → Reg → LiveOut (mPtsOf c.g (∑ s, ks s)) → FlockSoundness.Discharge.Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    (hTags : TagsOkH I)
    (hCust : FlockSoundness.Assumptions.ZkPubIn.RecordCustodyZKP I own dj wr (.stratified Kd (Array.ofFn St)) pop)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planAtZKHP (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t')
    (rd : FlockSoundness.Registered.Reads (layShHP (Reg := Reg) dj y₀ ht hc hscope hU hpi (pub := pub))
      (rsShHP dj y₀ ht hc hscope hU hpi hpub))
    (hown : OwnAtDrawsHP ht dj y₀ hc hU hpi hpub hscope rports
      (FlockSoundness.Audit.Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) wr rd)
    (τ : Strategy (auditLive (FlockSoundness.Audit.Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (liveOf dg (mPtsOf c.g (∑ s, ks s)) N wr)))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ execArith hs512 (planAtZKHP dj y₀
        ((FlockSoundness.Audit.Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _
          (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (scopeOk_spec hU hscope).2.1 dj y₀ wr _
            tablesAtZKHP_live hTags hCust.1
            (fun ω => drawOkZKHP_of_exec I own hc σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ))) j
      (zcStrat execArith hs512 (planAtZKHP dj y₀)
        (cont (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _
          (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (scopeOk_spec hU hscope).2.1 dj y₀ wr _
            tablesAtZKHP_live hTags hCust.1
            (fun ω => drawOkZKHP_of_exec I own hc σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ) ω)) qF qS)
    {qR : ℝ}
    (hR : (oneRun (auditLive (FlockSoundness.Audit.Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
        (liveOf dg (mPtsOf c.g (∑ s, ks s)) N wr)) τ fun o =>
      (restrict rd (drawnReads (layShHP (Reg := Reg) dj y₀ ht hc hscope hU hpi (pub := pub)) o.2.1)).pick o.1
        (regPathsHP c pi pub o.2.2 rd.port)).CR H512 qR)
    {δ : ℝ≥0∞}
    (hvals : RegValsSound (auditLive (FlockSoundness.Audit.Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop)
        Reg (liveOf dg (mPtsOf c.g (∑ s, ks s)) N wr)) τ (LiveAcceptsZKRP I rports own) (fun o => o.2.1) (fun o => o.1)
      rd
      (Xpub (pubOnes c hU n pi own) (pubZeros c hU n pi own)
        (XplurZC execArith hs512 ((keyProgP c hU n pi).partition (outsOf c hU n)) (planAtZKHP dj y₀)
          (tabAtZKHP dj y₀)
          ((rsZHP dj y₀ ht hc hscope hU hpi hpub).bindingZ (rsZHP_computes dj y₀ ht hc hscope hU hpi hpub))
          (reg (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _
            (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (scopeOk_spec hU hscope).2.1 dj y₀ wr _
              tablesAtZKHP_live hTags hCust.1
              (fun ω => drawOkZKHP_of_exec I own hc σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ))
          (cont (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith
            hs512 _ (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (scopeOk_spec hU hscope).2.1 dj y₀ wr _
              tablesAtZKHP_live hTags hCust.1
              (fun ω => drawOkZKHP_of_exec I own hc σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ)) k Rw)) δ) :
    ∃ X' : (Circuit.toCore ((keyProgP c hU n pi).circuit (outsOf c hU n))).Transcript,
      Verity.Game.prob (fun o => LiveAcceptsZKRP I rports own o ∧
          (¬ Disjoint ((Partition.toCore ((keyProgP c hU n pi).partition (outsOf c hU n))).wrong X') o.2.1 ∨
            ∃ g ∈ regGates rd, X' g ≠ Circuit.Transcript.toCore (regVals rd) g))
        (Game.toCore (auditLive (FlockSoundness.Audit.Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
          (liveOf dg (mPtsOf c.g (∑ s, ks s)) N wr))) (Game.Strategy.toCore _ τ) ≤
      (FlockSoundness.Audit.Law.stratified σs ks hks).miss 1 +
        ksAvgStrictZ (planAtZKHP dj y₀) (FlockSoundness.Audit.Law.execOS (n := n) src (.stratified Kd (Array.ofFn St))
          pop) k qF qS
          (reg (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _
            (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (scopeOk_spec hU hscope).2.1 dj y₀ wr _
              tablesAtZKHP_live hTags hCust.1
              (fun ω => drawOkZKHP_of_exec I own hc σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ)) +
        linkBoundZC execArith hs512 ((keyProgP c hU n pi).partition (outsOf c hU n)) (planAtZKHP dj y₀)
          (tabAtZKHP dj y₀)
          ((rsZHP dj y₀ ht hc hscope hU hpi hpub).bindingZ (rsZHP_computes dj y₀ ht hc hscope hU hpi hpub))
          k Rw M ρ t'
          (reg (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith hs512 _
            (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (scopeOk_spec hU hscope).2.1 dj y₀ wr _
              tablesAtZKHP_live hTags hCust.1
              (fun ω => drawOkZKHP_of_exec I own hc σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ))
          (cont (simZA execArith hs512 _ (AcceptsZKP I own) wr _ dg _ _ _ _ N (decodesOnZKA_of_runsOnZCA execArith
            hs512 _ (AcceptsZKP I own) wr _ _ _ _ (runsZC_liveHP I own ht hc (scopeOk_spec hU hscope).2.1 dj y₀ wr _
              tablesAtZKHP_live hTags hCust.1
              (fun ω => drawOkZKHP_of_exec I own hc σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ)) +
      (ENNReal.ofReal (qR ^ 2 / 2 ^ (513 : ℝ)) + δ) :=
  registered_zk_sessionHP ht dj y₀ hc hU hpi hpub hscope rports wr dg N σs ks hks St hS Kd pop Lb src hA3
    (fun ω => drawOkZKHP_of_exec I own hc σs ks hks St hS Kd pop Lb src dj hCust.2 ω) hTags hCust.1 hk hRw hM hρ hr
    ht' rd hown τ hCR hR hvals

end Custody

end ZkPubIn

end FlockBridge
