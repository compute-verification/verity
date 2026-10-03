import FlockSoundness.Discharge.PrivateCircuit.InstanceReal
import FlockSoundness.Discharge.PrivateCircuit.InstanceWired

/-!
# Protocol 2 at the universal unit's hidden-output `--zk --registered` session

Objective 2's session hides the outputs (`Stmt.setupHidden`, the default `--statement`) and reads the program as
registered values (`--registered F`). `realSessionHR` is `realSessionR` at hidden outputs: `zk_session_soundHR`'s game,
event (`LiveAcceptsZKR I own`) and rows (`rsZHS`), its A6 the hidden setup's tags, record and draws (`A6ZH`). Its
soundness is `zk_session_soundHR`'s first conjunct (`zkSessionSound_realHR`), and its registered check at an accepted
outcome is the second (`ZkReg.checked_of_acceptsH`), at the reads that are the verifier's own at the outcome's draw
(`ownAtH`, `OwnReadsH`) through the hidden drawn statement's paths (`regPathsH`).

* `private_circuit_sound_realH`: `private_circuit_sound_drawn` at that session (`private_circuit_sound_real` at hidden
  outputs).
* `private_circuit_sound_wiredH`, `private_circuit_sound_coneH`: `private_circuit_sound_wired` and
  `private_circuit_sound_cone` at that session, `C`'s owed outputs end to end along the public topology `T` under
  `hwire`.
-/

namespace FlockSoundness.Discharge.PrivateCircuit

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind ZkHidden HiddenExec
  HiddenDec Integrate Placed StrictCR DecodeZL Composed ZkReg Binding
open scoped ENNReal

variable {mPts n : ℕ} {Reg : Type}

section Session

variable (I : Inputs) (ht : I.tags.typed = false) (own : Array Flock.Registered.Port)
  (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZKH I mPts S₀ (dj S₀))
  (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
  (dg : ByteArray → ByteArray) (N : ℕ)
  {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
  {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)
  (hscope : Layout.scopeOk c = true)
  {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
  (St : Fin m → Flock.Draw.Stratum) (Kd pop : ℕ)
  {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
  (k Rw M : ℕ) (ρ t' qF qS : ℝ)

/-- A6 at a hidden-output `--zk` session: the verify command's tags, the coin server's record of the live session, and
its draw at each of the law's draws, as `zk_session_soundHR` takes them. -/
def A6ZH : Prop :=
  TagsOkH I ∧ RecordsLiveZK I dj wr ∧
    ∀ ω, DrawOkZKH I dj mPts ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)

/-- The simulated prover of `τ` in the compiled batched hidden-output session, under A6. -/
noncomputable def simOfH (h : A6ZH I dj wr St Kd pop Lb src (mPts := mPts))
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr))) :=
  simZ execArith hs512 _ I wr _ dg _ _ _ _ N
    (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _
      (runsZC_liveH I ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live h.1 h.2.1 h.2.2)) τ

open Classical in
/-- **The universal unit's hidden-output `--zk --registered` session, as `zk_session_soundHR` reads it.** -/
noncomputable def realSessionHR :
    UnitSession ((keyProg c hU n).partition (Layout.outsOf c hU n)) Reg where
  Out := Reg × Finset (Fin n) × Exec.Transcript
  G := auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)
  Accepts := LiveAcceptsZKR I own
  regOf o := o.1
  Lm := Law.stratified σs ks hks
  vals τ _ := if h : A6ZH I dj wr St Kd pop Lb src (mPts := mPts) then
      XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtZKH dj y₀) (tabAtZKH dj y₀)
        ((rsZHS dj y₀ ht hc hscope hU hpub).bindingZ (rsZHS_computes dj y₀ ht hc hscope hU hpub))
        (reg (simOfH I ht dj y₀ wr dg N hc hU hscope St Kd pop Lb src h τ))
        (cont (simOfH I ht dj y₀ wr dg N hc hU hscope St Kd pop Lb src h τ)) k Rw
    else fun _ => false
  εks τ := if h : A6ZH I dj wr St Kd pop Lb src (mPts := mPts) then
      ksAvgStrictZ (planAtZKH dj y₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
        (reg (simOfH I ht dj y₀ wr dg N hc hU hscope St Kd pop Lb src h τ))
    else ⊤
  δlink τ := if h : A6ZH I dj wr St Kd pop Lb src (mPts := mPts) then
      linkBoundZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtZKH dj y₀) (tabAtZKH dj y₀)
        ((rsZHS dj y₀ ht hc hscope hU hpub).bindingZ (rsZHS_computes dj y₀ ht hc hscope hU hpub))
        k Rw M ρ t' (reg (simOfH I ht dj y₀ wr dg N hc hU hscope St Kd pop Lb src h τ))
        (cont (simOfH I ht dj y₀ wr dg N hc hU hscope St Kd pop Lb src h τ))
    else ⊤
  CR τ := if h : A6ZH I dj wr St Kd pop Lb src (mPts := mPts) then
      ∀ ω j, TableCRZ execArith hs512 (planAtZKH dj y₀
          ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
          (reg (simOfH I ht dj y₀ wr dg N hc hU hscope St Kd pop Lb src h τ))) j
        (zcStrat execArith hs512 (planAtZKH dj y₀) (cont (simOfH I ht dj y₀ wr dg N hc hU hscope St Kd pop Lb src h τ) ω))
        qF qS
    else False
  m := m
  Ω := Ω
  Lb := Lb
  src := src
  A6 := A6ZH I dj wr St Kd pop Lb src (mPts := mPts)

/-- **`ZkSessionSound` at the universal unit's hidden-output `--zk --registered` session, from `zk_session_soundHR`**,
under the count curve's side conditions. -/
theorem zkSessionSound_realHR (hS : Law.ExecStrata σs ks St) (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planAtZKH (Reg := Reg) dj y₀ S R) j k ≤ ρ) (ht' : 0 ≤ t') :
    ZkSessionSound (realSessionHR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS) := by
  intro hA3 hA6 τ hCR K₀
  have hA6' : A6ZH I dj wr St Kd pop Lb src (mPts := mPts) := hA6
  simp only [realSessionHR, hA6', ↓reduceDIte] at hCR ⊢
  exact (zk_session_soundHR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St hS Kd pop Lb src hA3 hA6'.2.2
    hA6'.1 hA6'.2.1 hk hRw hM hρ hr ht' K₀ τ hCR).1

end Session

/-- **The registered reads the verifier checks at the transcript `t`, under hidden outputs**: `p` is one of the
verifier's own at the record's draw (`OwnReadsH` at `p` alone). -/
def ownAtH {Gate : Type} (own : Array Flock.Registered.Port) (c : Flock.Circuit) (pub : Public) (t : Exec.Transcript)
    (port : (Fin n × Fin c.ports.size) ⊕ Gate → Flock.Registered.Port) (readAt : (Fin n × Fin c.ports.size) ⊕ Gate → ℕ)
    (p : (Fin n × Fin c.ports.size) ⊕ Gate) : Prop :=
  OwnReadsH own c pub t (· = p) port readAt

section Instance

variable {cls : Universal.Cls}
  (I : Inputs) (ht : I.tags.typed = false) (own : Array Flock.Registered.Port)
  (dj : Finset (Fin (Universal.units cls)) → Lean.Json) {S₀ : Finset (Fin (Universal.units cls))}
  (y₀ : DrawSetupZKH I mPts S₀ (dj S₀))
  (wr : Finset (Fin (Universal.units cls)) → Reg → LiveOut mPts → Exec.Transcript)
  (dg : ByteArray → ByteArray) (N : ℕ)
  {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
  {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub) (hscope : Layout.scopeOk c = true)
  {m : ℕ} (σs : Fin (Universal.units cls) → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
  (St : Fin m → Flock.Draw.Stratum) (Kd pop : ℕ)
  {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
  {k Rw M : ℕ} {ρ : ℝ} {t' : ℝ} {qF qS : ℝ}

/-- **Protocol 2 is sound at the universal unit's hidden-output `--zk --registered` session.**
`private_circuit_sound_real` at hidden outputs: under `zk_session_soundHR`'s hypotheses (A3, A6 as `A6ZH`, the count
curve's side conditions, `cr/sha-512` for `τ`'s knowledge finders), the step to the program at the drawn units where
the verifier's own reads hold the registrant's leaves (`LeafBindsOn`), and `SHA512CRStrict` for the finder that outputs
`progPairOn` (`hBind`): `private_circuit_sound_drawn`'s two bounds, the checked reads at `o` the verifier's own at `o`'s
draw (`ownAtH`), through the hidden drawn statement's paths (`regPathsH`), and the drawn units the audit's draw `o.2.1`. -/
theorem private_circuit_sound_realH
    (hS : Law.ExecStrata σs ks St)
    (hA3 : Assumptions.UniformRandomBytes Lb src) (hA6 : A6ZH I dj wr St Kd pop Lb src (mPts := mPts))
    (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planAtZKH (Reg := Reg) dj y₀ S R) j k ≤ ρ) (ht' : 0 ≤ t')
    (C : Universal.Circ) (hC : Universal.fits C cls)
    (progOf : (Fin ((keyProg c hU (Universal.units cls)).circuit (Layout.outsOf c hU (Universal.units cls))).N → Bool) →
      cls.Prog)
    (xOf : (Fin ((keyProg c hU (Universal.units cls)).circuit (Layout.outsOf c hU (Universal.units cls))).N → Bool) →
      ℕ → ℕ → Bool)
    (yOf : (Fin ((keyProg c hU (Universal.units cls)).circuit (Layout.outsOf c hU (Universal.units cls))).N → Bool) →
      ℕ → List Bool)
    (hunit : UnitWrong ((keyProg c hU (Universal.units cls)).partition (Layout.outsOf c hU (Universal.units cls)))
      progOf xOf yOf)
    (rd : Registered.Reads (layShH (Reg := Reg) dj y₀ ht hc hscope hU (pub := pub)) (rsShHS dj y₀ ht hc hscope hU hpub))
    {rt : Flock.Registered.Port} {L : ℕ} (ps : ProgSites rd rt L)
    (row : cls.Prog → Fin L → (layShH (Reg := Reg) dj y₀ ht hc hscope hU (pub := pub)).Val)
    (hreg : ∀ i, rd.wv (ps.site i) = row (Universal.pad C cls) i)
    (τ : Strategy (realSessionHR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS).G)
    (hCR : (realSessionHR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS).CR τ)
    (valPair : Reg × Finset (Fin (Universal.units cls)) × Exec.Transcript → List UInt8 × List UInt8)
    (hstep : LeafBindsOn ps
      (realSessionHR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS) τ progOf
      (Universal.pad C cls) (fun o => ownAtH own c pub o.2.2 rd.port rd.readAt) (fun o => o.2.1) valPair)
    (K₀ : ℕ) (cost : Reg × Finset (Fin (Universal.units cls)) × Exec.Transcript → ℝ) (q : ℝ) (hcost : ∀ o, cost o ≤ q)
    (hBind : Assumptions.SHA512CRStrict H512
      (realSessionHR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS).G τ
      (fun o => some (progPairOn ps
        (realSessionHR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS)
        (fun o => ownAtH own c pub o.2.2 rd.port rd.readAt) (fun o => regPathsH c pub o.2.2 rd.port) valPair o)) cost q) :
    let s := realSessionHR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS
    prob (fun o => LiveAcceptsZKR I own o ∧
        (K₀ ≤ wrongAgree C cls (progOf (s.vals τ o)) (xOf (s.vals τ o)) (yOf (s.vals τ o)) ∨
          (∃ u ∈ o.2.1, progOf (s.vals τ o) u ≠ Universal.pad C cls u) ∨
          ∃ i, ownAtH own c pub o.2.2 rd.port rd.readAt (ps.site i) ∧
            (layShH (Reg := Reg) dj y₀ ht hc hscope hU (pub := pub)).regLeaf o.1 (ps.site i) ≠
              (ExplicitLeaf.hm96 (rsShHS dj y₀ ht hc hscope hU hpub).enc (rsShHS dj y₀ ht hc hscope hU hpub).enc_inj).leaf
                (row (Universal.pad C cls) i) (rd.ws (ps.site i)))) s.G τ
      ≤ s.Lm.miss K₀ + s.εks τ + s.δlink τ + ENNReal.ofReal (q ^ 2 / 2 ^ (513 : ℝ)) ∧
    ∀ K₁, ForgedMiss s τ (fun o => o.2.1) (fun o => forged C cls (progOf (s.vals τ o))) K₁ →
      prob (fun o => LiveAcceptsZKR I own o ∧
          K₀ + K₁ ≤ wrong (Universal.eval C) (Universal.size C) (xOf (s.vals τ o)) (yOf (s.vals τ o))) s.G τ
        ≤ s.Lm.miss K₀ + s.Lm.miss K₁ + s.εks τ + s.δlink τ + ENNReal.ofReal (q ^ 2 / 2 ^ (513 : ℝ)) := by
  intro s
  exact private_circuit_sound_drawn _ C hC progOf xOf yOf hunit rd ps row hreg s
    (zkSessionSound_realHR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS hS hk
      hRw hM hρ hr ht')
    hA3 hA6 (fun o => ownAtH own c pub o.2.2 rd.port rd.readAt) (fun o => regPathsH c pub o.2.2 rd.port)
    (fun o ho => checked_of_acceptsH I ht own dj y₀ hc hU hpub hscope hA6.1 o ho
      (readsOn rd (ownAtH own c pub o.2.2 rd.port rd.readAt)) fun p hp => hp.2 p rfl)
    τ hCR (fun o => o.2.1) valPair hstep K₀ cost q hcost hBind

/-- **Protocol 2 is sound end to end at the universal unit's hidden-output `--zk --registered` session.**
`private_circuit_sound_wired` at `realSessionHR`, under `private_circuit_sound_realH`'s hypotheses and the claimed values
agreeing along the public topology `T` at the circuit's inputs `xIn v` (`hwire`; discharged by the layout that stages a
reader's input rows under the writer's salt and checks each cut wire's leaves equal): accepting with `C`'s owed outputs
wrong and the committed program `pad C cls`'s, or with a drawn unit's program forged, has probability at most
`miss(1) + ε_ks + δ_link + q²/2^513`; where the law misses the forged units at `1`, accepting with `C`'s owed outputs
wrong has probability at most `miss(1) + miss(1) + ε_ks + δ_link + q²/2^513`. -/
theorem private_circuit_sound_wiredH
    (hS : Law.ExecStrata σs ks St)
    (hA3 : Assumptions.UniformRandomBytes Lb src) (hA6 : A6ZH I dj wr St Kd pop Lb src (mPts := mPts))
    (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planAtZKH (Reg := Reg) dj y₀ S R) j k ≤ ρ) (ht' : 0 ≤ t')
    (C : Universal.Circ) (hC : Universal.fits C cls)
    (progOf : (Fin ((keyProg c hU (Universal.units cls)).circuit (Layout.outsOf c hU (Universal.units cls))).N → Bool) →
      cls.Prog)
    (xOf : (Fin ((keyProg c hU (Universal.units cls)).circuit (Layout.outsOf c hU (Universal.units cls))).N → Bool) →
      ℕ → ℕ → Bool)
    (yOf : (Fin ((keyProg c hU (Universal.units cls)).circuit (Layout.outsOf c hU (Universal.units cls))).N → Bool) →
      ℕ → List Bool)
    (hunit : UnitWrong ((keyProg c hU (Universal.units cls)).partition (Layout.outsOf c hU (Universal.units cls)))
      progOf xOf yOf)
    (T : Universal.Topo)
    (xIn : (Fin ((keyProg c hU (Universal.units cls)).circuit (Layout.outsOf c hU (Universal.units cls))).N → Bool) →
      ℕ → Bool)
    (hwire : ∀ v, Universal.Wired T cls (xOf v) (yOf v) (xIn v))
    (rd : Registered.Reads (layShH (Reg := Reg) dj y₀ ht hc hscope hU (pub := pub)) (rsShHS dj y₀ ht hc hscope hU hpub))
    {rt : Flock.Registered.Port} {L : ℕ} (ps : ProgSites rd rt L)
    (τ : Strategy (realSessionHR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS).G)
    (hCR : (realSessionHR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS).CR τ)
    (valPair : Reg × Finset (Fin (Universal.units cls)) × Exec.Transcript → List UInt8 × List UInt8)
    (hstep : LeafBindsOn ps
      (realSessionHR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS) τ progOf
      (Universal.pad C cls) (fun o => ownAtH own c pub o.2.2 rd.port rd.readAt) (fun o => o.2.1) valPair)
    (cost : Reg × Finset (Fin (Universal.units cls)) × Exec.Transcript → ℝ) (q : ℝ) (hcost : ∀ o, cost o ≤ q)
    (hBind : Assumptions.SHA512CRStrict H512
      (realSessionHR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS).G τ
      (fun o => some (progPairOn ps
        (realSessionHR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS)
        (fun o => ownAtH own c pub o.2.2 rd.port rd.readAt) (fun o => regPathsH c pub o.2.2 rd.port) valPair o)) cost q) :
    let s := realSessionHR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS
    prob (fun o => LiveAcceptsZKR I own o ∧
        ((OwedWrong T C (xIn (s.vals τ o)) (yOf (s.vals τ o)) ∧ progOf (s.vals τ o) = Universal.pad C cls) ∨
          ∃ u ∈ o.2.1, progOf (s.vals τ o) u ≠ Universal.pad C cls u)) s.G τ
      ≤ s.Lm.miss 1 + s.εks τ + s.δlink τ + ENNReal.ofReal (q ^ 2 / 2 ^ (513 : ℝ)) ∧
    (ForgedMiss s τ (fun o => o.2.1) (fun o => forged C cls (progOf (s.vals τ o))) 1 →
      prob (fun o => LiveAcceptsZKR I own o ∧ OwedWrong T C (xIn (s.vals τ o)) (yOf (s.vals τ o))) s.G τ
        ≤ s.Lm.miss 1 + s.Lm.miss 1 + s.εks τ + s.δlink τ + ENNReal.ofReal (q ^ 2 / 2 ^ (513 : ℝ))) := by
  intro s
  exact private_circuit_sound_wired _ C hC progOf xOf yOf hunit T xIn hwire rd ps s
    (zkSessionSound_realHR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS hS hk
      hRw hM hρ hr ht')
    hA3 hA6 (fun o => ownAtH own c pub o.2.2 rd.port rd.readAt) (fun o => regPathsH c pub o.2.2 rd.port)
    (fun o ho => checked_of_acceptsH I ht own dj y₀ hc hU hpub hscope hA6.1 o ho
      (readsOn rd (ownAtH own c pub o.2.2 rd.port rd.readAt)) fun p hp => hp.2 p rfl)
    τ hCR (fun o => o.2.1) valPair hstep cost q hcost hBind

/-- **Protocol 2's cone bound at the universal unit's hidden-output `--zk --registered` session.**
`private_circuit_sound_cone` at `realSessionHR`: for every unit `i` of `C`, accepting with unit `i`'s owed outputs other
than `evalT`'s has probability at most that of accepting with a wrong unit in `i`'s cone, plus that of accepting with a
forged unit in `i`'s cone and none at the drawn units, plus `q²/2^513`. -/
theorem private_circuit_sound_coneH
    (hA6 : A6ZH I dj wr St Kd pop Lb src (mPts := mPts))
    (C : Universal.Circ) (hC : Universal.fits C cls)
    (progOf : (Fin ((keyProg c hU (Universal.units cls)).circuit (Layout.outsOf c hU (Universal.units cls))).N → Bool) →
      cls.Prog)
    (xOf : (Fin ((keyProg c hU (Universal.units cls)).circuit (Layout.outsOf c hU (Universal.units cls))).N → Bool) →
      ℕ → ℕ → Bool)
    (yOf : (Fin ((keyProg c hU (Universal.units cls)).circuit (Layout.outsOf c hU (Universal.units cls))).N → Bool) →
      ℕ → List Bool)
    (hunit : UnitWrong ((keyProg c hU (Universal.units cls)).partition (Layout.outsOf c hU (Universal.units cls)))
      progOf xOf yOf)
    (T : Universal.Topo)
    (xIn : (Fin ((keyProg c hU (Universal.units cls)).circuit (Layout.outsOf c hU (Universal.units cls))).N → Bool) →
      ℕ → Bool)
    (hwire : ∀ v, Universal.Wired T cls (xOf v) (yOf v) (xIn v))
    (rd : Registered.Reads (layShH (Reg := Reg) dj y₀ ht hc hscope hU (pub := pub)) (rsShHS dj y₀ ht hc hscope hU hpub))
    {rt : Flock.Registered.Port} {L : ℕ} (ps : ProgSites rd rt L)
    (τ : Strategy (realSessionHR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS).G)
    (valPair : Reg × Finset (Fin (Universal.units cls)) × Exec.Transcript → List UInt8 × List UInt8)
    (hstep : LeafBindsOn ps
      (realSessionHR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS) τ progOf
      (Universal.pad C cls) (fun o => ownAtH own c pub o.2.2 rd.port rd.readAt) (fun o => o.2.1) valPair)
    (cost : Reg × Finset (Fin (Universal.units cls)) × Exec.Transcript → ℝ) (q : ℝ) (hcost : ∀ o, cost o ≤ q)
    (hBind : Assumptions.SHA512CRStrict H512
      (realSessionHR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS).G τ
      (fun o => some (progPairOn ps
        (realSessionHR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS)
        (fun o => ownAtH own c pub o.2.2 rd.port rd.readAt) (fun o => regPathsH c pub o.2.2 rd.port) valPair o)) cost q)
    (i : ℕ) (hi : i < Universal.size C) :
    let s := realSessionHR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS
    prob (fun o => LiveAcceptsZKR I own o ∧ yOf (s.vals τ o) i ≠ Universal.evalT T C (xIn (s.vals τ o)) i) s.G τ
      ≤ prob (fun o => LiveAcceptsZKR I own o ∧ ∃ j ∈ coneOf T cls i,
            j ∈ ((keyProg c hU (Universal.units cls)).partition (Layout.outsOf c hU (Universal.units cls))).wrong
              (s.vals τ o)) s.G τ +
        prob (fun o => LiveAcceptsZKR I own o ∧ (∀ u ∈ o.2.1, progOf (s.vals τ o) u = Universal.pad C cls u) ∧
          ∃ j ∈ coneOf T cls i, progOf (s.vals τ o) j ≠ Universal.pad C cls j) s.G τ +
        ENNReal.ofReal (q ^ 2 / 2 ^ (513 : ℝ)) := by
  intro s
  exact private_circuit_sound_cone _ C hC progOf xOf yOf hunit T xIn hwire rd ps s
    (fun o => ownAtH own c pub o.2.2 rd.port rd.readAt) (fun o => regPathsH c pub o.2.2 rd.port)
    (fun o ho => checked_of_acceptsH I ht own dj y₀ hc hU hpub hscope hA6.1 o ho
      (readsOn rd (ownAtH own c pub o.2.2 rd.port rd.readAt)) fun p hp => hp.2 p rfl)
    τ (fun o => o.2.1) valPair hstep cost q hcost hBind i hi

end Instance

end FlockSoundness.Discharge.PrivateCircuit
