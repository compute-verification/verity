import FlockSoundness.Discharge.PrivateCircuit.InstanceDrawn
import FlockSoundness.Discharge.PrivateCircuit.InstanceBridge
import FlockSoundness.Discharge.ZkReg

/-!
# Protocol 2 at the universal unit's `--zk --registered` session

`private_circuit_sound_real`: `private_circuit_sound_drawn` at the session `verify --zk --registered F` checks
(`realSessionR`), whose soundness is `zk_session_soundR`'s first conjunct (`zkSessionSound_realR`) and whose registered
check at an accepted outcome is `zk_session_soundR`'s second (`ZkReg.checked_of_accepts`): the layout's registered reads
that are the verifier's own at the outcome's draw (`OwnReads`), through the drawn statement's paths (`regPaths`). The
checked reads at `o` are those (`ownAt`), and the drawn units are the audit's draw (`o.2.1`).

The session's pieces enter through `realSessionR`, `zkSessionSound_realR` and `ZkReg.checked_of_accepts` only, so the
hidden-outputs session (`zk_session_soundHR`) replaces them one for one.
-/

namespace FlockSoundness.Discharge.PrivateCircuit

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR
  DecodeZL Composed ZkReg Binding
open scoped ENNReal

variable {mPts n : ℕ} {Reg : Type}

section Session

variable (I : Inputs) (ht : I.tags.typed = false) (own : Array Flock.Registered.Port)
  (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZK I mPts S₀ (dj S₀))
  (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
  (dg : ByteArray → ByteArray) (N : ℕ)
  {c : Flock.Circuit} (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
  {pub : Public} (hpub : HmRow.loadPublic (zkTags I) c I.publicFile = .ok pub)
  (hscope : Layout.scopeOk c = true)
  {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
  (St : Fin m → Flock.Draw.Stratum) (Kd pop : ℕ)
  {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
  (k Rw M : ℕ) (ρ t' qF qS : ℝ)

open Classical in
/-- **The universal unit's `--zk --registered` session, as `zk_session_soundR` reads it**: `realSession` with
`verify --zk --registered F`'s acceptance (`LiveAcceptsZKR I own`) and the rows with or without shared rows
(`rsZAtS`). -/
noncomputable def realSessionR :
    UnitSession ((keyProg c hU n).partition (Layout.outsOf c hU n)) Reg where
  Out := Reg × Finset (Fin n) × Exec.Transcript
  G := auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)
  Accepts := LiveAcceptsZKR I own
  regOf o := o.1
  Lm := Law.stratified σs ks hks
  vals τ _ := if h : A6Z I dj wr St Kd pop Lb src then
      XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZAt dj y₀) (tabZAt dj y₀)
        ((rsZAtS ht dj y₀ hc hU hpub hscope).bindingZ (rsZAtS_computes ht dj y₀ hc hU hpub hscope))
        (reg (simOf I ht dj y₀ wr dg N St Kd pop Lb src h τ)) (cont (simOf I ht dj y₀ wr dg N St Kd pop Lb src h τ)) k Rw
    else fun _ => false
  εks τ := if h : A6Z I dj wr St Kd pop Lb src then
      ksAvgStrictZ (planZAt dj y₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
        (reg (simOf I ht dj y₀ wr dg N St Kd pop Lb src h τ))
    else ⊤
  δlink τ := if h : A6Z I dj wr St Kd pop Lb src then
      linkBoundZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZAt dj y₀) (tabZAt dj y₀)
        ((rsZAtS ht dj y₀ hc hU hpub hscope).bindingZ (rsZAtS_computes ht dj y₀ hc hU hpub hscope))
        k Rw M ρ t' (reg (simOf I ht dj y₀ wr dg N St Kd pop Lb src h τ)) (cont (simOf I ht dj y₀ wr dg N St Kd pop Lb src h τ))
    else ⊤
  CR τ := if h : A6Z I dj wr St Kd pop Lb src then
      ∀ ω j, TableCRZ execArith hs512 (planZAt dj y₀
          ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
          (reg (simOf I ht dj y₀ wr dg N St Kd pop Lb src h τ))) j
        (zcStrat execArith hs512 (planZAt dj y₀) (cont (simOf I ht dj y₀ wr dg N St Kd pop Lb src h τ) ω)) qF qS
    else False
  m := m
  Ω := Ω
  Lb := Lb
  src := src
  A6 := A6Z I dj wr St Kd pop Lb src

/-- **`ZkSessionSound` at the universal unit's `--zk --registered` session, from `zk_session_soundR`**, under the count
curve's side conditions. -/
theorem zkSessionSound_realR (hS : Law.ExecStrata σs ks St) (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZAt (Reg := Reg) dj y₀ S R) j k ≤ ρ) (ht' : 0 ≤ t') :
    ZkSessionSound (realSessionR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS) := by
  intro hA3 hA6 τ hCR K₀
  have hA6' : A6Z I dj wr St Kd pop Lb src := hA6
  simp only [realSessionR, hA6', ↓reduceDIte] at hCR ⊢
  exact (zk_session_soundR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St hS Kd pop Lb src hA3 hA6'.2.2
    hA6'.1 hA6'.2.1 hk hRw hM hρ hr ht' K₀ τ hCR).1

end Session

/-- **The registered reads the verifier checks at the transcript `t`**: `p` is one of the verifier's own at the record's
draw (`OwnReads` at `p` alone). -/
def ownAt {Gate : Type} (own : Array Flock.Registered.Port) (c : Flock.Circuit) (pub : Public) (t : Exec.Transcript)
    (port : (Fin n × Fin c.ports.size) ⊕ Gate → Flock.Registered.Port) (readAt : (Fin n × Fin c.ports.size) ⊕ Gate → ℕ)
    (p : (Fin n × Fin c.ports.size) ⊕ Gate) : Prop :=
  OwnReads own c pub t (· = p) port readAt

/-- **Protocol 2 is sound at the universal unit's `--zk --registered` session.** The class `cls`'s `units cls` units are
the session's instances; the circuit `C` fits `cls`; the universal unit's partition is the interpreter's (`UnitWrong`);
`pad C c` is registered at the program's sites (`ps`, `hreg`) of the shadow layout's registered reads `rd`. Under
`zk_session_soundR`'s hypotheses (A3, A6, the count curve's side conditions, `cr/sha-512` for `τ`'s knowledge finders),
the step to the program at the drawn units where the verifier's own reads hold the registrant's leaves (`LeafBindsOn`),
and `SHA512CRStrict` for the finder that outputs `progPairOn` (`hBind`): `private_circuit_sound_drawn`'s two bounds,
the checked reads at `o` the verifier's own at `o`'s draw (`ownAt`), through the drawn statement's paths (`regPaths`),
and the drawn units the audit's draw `o.2.1`. -/
theorem private_circuit_sound_real {cls : Universal.Cls}
    (I : Inputs) (ht : I.tags.typed = false) (own : Array Flock.Registered.Port)
    (dj : Finset (Fin (Universal.units cls)) → Lean.Json) {S₀ : Finset (Fin (Universal.units cls))}
    (y₀ : DrawSetupZK I mPts S₀ (dj S₀))
    (wr : Finset (Fin (Universal.units cls)) → Reg → LiveOut mPts → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    {c : Flock.Circuit} (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmRow.loadPublic (zkTags I) c I.publicFile = .ok pub) (hscope : Layout.scopeOk c = true)
    {m : ℕ} (σs : Fin (Universal.units cls) → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src) (hA6 : A6Z I dj wr St Kd pop Lb src)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZAt (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') {qF qS : ℝ}
    (C : Universal.Circ) (hC : Universal.fits C cls)
    (progOf : (Fin ((keyProg c hU (Universal.units cls)).circuit (Layout.outsOf c hU (Universal.units cls))).N → Bool) →
      cls.Prog)
    (xOf : (Fin ((keyProg c hU (Universal.units cls)).circuit (Layout.outsOf c hU (Universal.units cls))).N → Bool) →
      ℕ → ℕ → Bool)
    (yOf : (Fin ((keyProg c hU (Universal.units cls)).circuit (Layout.outsOf c hU (Universal.units cls))).N → Bool) →
      ℕ → List Bool)
    (hunit : UnitWrong ((keyProg c hU (Universal.units cls)).partition (Layout.outsOf c hU (Universal.units cls)))
      progOf xOf yOf)
    (rd : Registered.Reads (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)) (rowsSh ht dj y₀ hc hU hpub hscope))
    {rt : Flock.Registered.Port} {L : ℕ} (ps : ProgSites rd rt L)
    (row : cls.Prog → Fin L → (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)).Val)
    (hreg : ∀ i, rd.wv (ps.site i) = row (Universal.pad C cls) i)
    (τ : Strategy (realSessionR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS).G)
    (hCR : (realSessionR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS).CR τ)
    (valPair : Reg × Finset (Fin (Universal.units cls)) × Exec.Transcript → List UInt8 × List UInt8)
    (hstep : LeafBindsOn ps
      (realSessionR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS) τ progOf
      (Universal.pad C cls) (fun o => ownAt own c pub o.2.2 rd.port rd.readAt) (fun o => o.2.1) valPair)
    (K₀ : ℕ) (cost : Reg × Finset (Fin (Universal.units cls)) × Exec.Transcript → ℝ) (q : ℝ) (hcost : ∀ o, cost o ≤ q)
    (hBind : Assumptions.SHA512CRStrict H512
      (realSessionR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS).G τ
      (fun o => some (progPairOn ps
        (realSessionR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS)
        (fun o => ownAt own c pub o.2.2 rd.port rd.readAt) (fun o => regPaths c pub o.2.2 rd.port) valPair o)) cost q) :
    let s := realSessionR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS
    prob (fun o => LiveAcceptsZKR I own o ∧
        (K₀ ≤ wrongAgree C cls (progOf (s.vals τ o)) (xOf (s.vals τ o)) (yOf (s.vals τ o)) ∨
          (∃ u ∈ o.2.1, progOf (s.vals τ o) u ≠ Universal.pad C cls u) ∨
          ∃ i, ownAt own c pub o.2.2 rd.port rd.readAt (ps.site i) ∧
            (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)).regLeaf o.1 (ps.site i) ≠
              (ExplicitLeaf.hm96 (rowsSh ht dj y₀ hc hU hpub hscope).enc (rowsSh ht dj y₀ hc hU hpub hscope).enc_inj).leaf
                (row (Universal.pad C cls) i) (rd.ws (ps.site i)))) s.G τ
      ≤ s.Lm.miss K₀ + s.εks τ + s.δlink τ + ENNReal.ofReal (q ^ 2 / 2 ^ (513 : ℝ)) ∧
    ∀ K₁, ForgedMiss s τ (fun o => o.2.1) (fun o => forged C cls (progOf (s.vals τ o))) K₁ →
      prob (fun o => LiveAcceptsZKR I own o ∧
          K₀ + K₁ ≤ wrong (Universal.eval C) (Universal.size C) (xOf (s.vals τ o)) (yOf (s.vals τ o))) s.G τ
        ≤ s.Lm.miss K₀ + s.Lm.miss K₁ + s.εks τ + s.δlink τ + ENNReal.ofReal (q ^ 2 / 2 ^ (513 : ℝ)) := by
  intro s
  exact private_circuit_sound_drawn _ C hC progOf xOf yOf hunit rd ps row hreg s
    (zkSessionSound_realR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS hS hk hRw
      hM hρ hr ht')
    hA3 hA6 (fun o => ownAt own c pub o.2.2 rd.port rd.readAt) (fun o => regPaths c pub o.2.2 rd.port)
    (fun o ho => checked_of_accepts I ht own dj y₀ hc hU hpub hscope hA6.1 o ho
      (readsOn rd (ownAt own c pub o.2.2 rd.port rd.readAt)) fun p hp => hp.2 p rfl)
    τ hCR (fun o => o.2.1) valPair hstep K₀ cost q hcost hBind

end FlockSoundness.Discharge.PrivateCircuit
