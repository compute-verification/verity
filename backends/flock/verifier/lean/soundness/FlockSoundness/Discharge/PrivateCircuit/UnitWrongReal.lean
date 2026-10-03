import FlockSoundness.Discharge.PrivateCircuit.UnitDefs
import FlockSoundness.Discharge.PrivateCircuit.UnitRows
import FlockSoundness.Discharge.PrivateCircuit.InstanceReal

/-!
# `UnitWrong` at the universal unit's session

* `unitWrong_trivial`: `UnitWrong P progOf xOf yOf` holds of every partition for some `yOf`, so as a hypothesis with
  free reads it carries nothing about the circuit.
* `unitWrong_universal`: at the session's partition of a parsed circuit of one copy per VU (`c.upv = 1`) whose net
  computes `UniversalUnit_v1` (`UniversalComputes`), `UnitWrong` holds at the unit's own reads (`uProg`, `uX`, `uY`) at
  every assignment that holds 1 at the constant's gate (`UnitWrongOn`).
* `private_circuit_sound_universal`: `private_circuit_sound_real` at those reads, with `UniversalComputes` and the
  constant's gate holding 1 in the committed values of accepted outcomes (`hone`) for `UnitWrong`.
-/

namespace FlockSoundness.Discharge.PrivateCircuit

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR
  DecodeZL Composed ZkReg Binding
open scoped ENNReal

/-- **`UnitWrong` with free reads holds of every partition**: given the program and data reads, the claimed outputs that
miss the run exactly at the partition's wrong units make it true. -/
theorem unitWrong_trivial {C : Audit.Circuit} {c : Universal.Cls} (P : Audit.Partition C (Universal.units c))
    (progOf : (Fin C.N → Bool) → c.Prog) (xOf : (Fin C.N → Bool) → ℕ → ℕ → Bool) :
    ∃ yOf, UnitWrong P progOf xOf yOf := by
  classical
  refine ⟨fun v i => if h : i < Universal.units c then
    (if (⟨i, h⟩ : Fin (Universal.units c)) ∈ P.wrong v then true :: Universal.run c (progOf v) i (xOf v i)
      else Universal.run c (progOf v) i (xOf v i)) else [], fun v i => ?_⟩
  simp only [i.isLt, ↓reduceDIte, Fin.eta]
  split_ifs with hw
  · simp only [hw, ne_eq, true_iff]
    exact fun h => List.cons_ne_self _ _ h.symm
  · simp [hw]

/-- **The session's partition of the universal unit is the interpreter's, at its own reads.** A circuit `c` of one copy
per VU (`hupv`) whose net computes `UniversalUnit_v1{G, nIn, nOut}` at the arguments' columns `L` (`hcomp`): at every
assignment `v` of the session's program's gates with `v oneG = true`, unit `i` of `units cls` is wrong exactly when
the universal unit running the program `v` holds at the units' input gates (`uProg L v`), on unit `i`'s data
(`uX L v i`), misses its claimed outputs (`uY cls v i`). -/
theorem unitWrong_universal {c : Flock.Circuit} (hU : Built c.unit) (cls : Universal.Cls) (hupv : c.upv = 1)
    (L : UIns cls (FlockSoundness.Rows.ofNet hU).nIn) (hcomp : UniversalComputes c hU cls L) :
    UnitWrongOn ((keyProg c hU (Universal.units cls)).partition (Layout.outsOf c hU (Universal.units cls)))
      (fun v => v Layout.oneG = true) (uProg L) (uX L) (uY cls) := by
  sorry

variable {mPts : ℕ} {Reg : Type}

/-- **Protocol 2 is sound at the universal unit's `--zk --registered` session, at the unit's own reads.**
`private_circuit_sound_real` with the program, data and claimed outputs the committed values hold at each unit's own
gates (`uProg`, `uX`, `uY`), for a circuit of one copy per VU (`hupv`) whose net computes `UniversalUnit_v1` at the
arguments' columns `L` (`hcomp`), and whose committed values at accepted outcomes hold 1 at the constant's gate
(`hone`), in place of `UnitWrong`. -/
theorem private_circuit_sound_universal {cls : Universal.Cls}
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
    (hupv : c.upv = 1) (L : UIns cls (FlockSoundness.Rows.ofNet hU).nIn) (hcomp : UniversalComputes c hU cls L)
    (rd : Registered.Reads (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)) (rowsSh ht dj y₀ hc hU hpub hscope))
    {rt : Flock.Registered.Port} {L' : ℕ} (ps : ProgSites rd rt L')
    (row : cls.Prog → Fin L' → (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)).Val)
    (hreg : ∀ i, rd.wv (ps.site i) = row (Universal.pad C cls) i)
    (τ : Strategy (realSessionR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS).G)
    (hCR : (realSessionR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS).CR τ)
    (hone : ∀ o, LiveAcceptsZKR I own o →
      (realSessionR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS).vals τ o
        Layout.oneG = true)
    (valPair : Reg × Finset (Fin (Universal.units cls)) × Exec.Transcript → List UInt8 × List UInt8)
    (hstep : LeafBindsOn ps
      (realSessionR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS) τ (uProg L)
      (Universal.pad C cls) (fun o => ownAt own c pub o.2.2 rd.port rd.readAt) (fun o => o.2.1) valPair)
    (K₀ : ℕ) (cost : Reg × Finset (Fin (Universal.units cls)) × Exec.Transcript → ℝ) (q : ℝ) (hcost : ∀ o, cost o ≤ q)
    (hBind : Assumptions.SHA512CRStrict H512
      (realSessionR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS).G τ
      (fun o => some (progPairOn ps
        (realSessionR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS)
        (fun o => ownAt own c pub o.2.2 rd.port rd.readAt) (fun o => regPaths c pub o.2.2 rd.port) valPair o)) cost q) :
    let s := realSessionR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St Kd pop Lb src k Rw M ρ t' qF qS
    prob (fun o => LiveAcceptsZKR I own o ∧
        (K₀ ≤ wrongAgree C cls (uProg L (s.vals τ o)) (uX L (s.vals τ o)) (uY cls (s.vals τ o)) ∨
          (∃ u ∈ o.2.1, uProg L (s.vals τ o) u ≠ Universal.pad C cls u) ∨
          ∃ i, ownAt own c pub o.2.2 rd.port rd.readAt (ps.site i) ∧
            (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)).regLeaf o.1 (ps.site i) ≠
              (ExplicitLeaf.hm96 (rowsSh ht dj y₀ hc hU hpub hscope).enc (rowsSh ht dj y₀ hc hU hpub hscope).enc_inj).leaf
                (row (Universal.pad C cls) i) (rd.ws (ps.site i)))) s.G τ
      ≤ s.Lm.miss K₀ + s.εks τ + s.δlink τ + ENNReal.ofReal (q ^ 2 / 2 ^ (513 : ℝ)) ∧
    ∀ K₁, ForgedMiss s τ (fun o => o.2.1) (fun o => forged C cls (uProg L (s.vals τ o))) K₁ →
      prob (fun o => LiveAcceptsZKR I own o ∧
          K₀ + K₁ ≤ wrong (Universal.eval C) (Universal.size C) (uX L (s.vals τ o)) (uY cls (s.vals τ o))) s.G τ
        ≤ s.Lm.miss K₀ + s.Lm.miss K₁ + s.εks τ + s.δlink τ + ENNReal.ofReal (q ^ 2 / 2 ^ (513 : ℝ)) := by
  sorry

end FlockSoundness.Discharge.PrivateCircuit
