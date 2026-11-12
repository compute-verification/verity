module

public import Proofs.Flock.Claims.Witnesses.SamplingTwoRegX
public import Proofs.Flock.SamplingGeneric.TwoFirstHitKS
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.ZK.NonVacuous

/-!
# The two-stage committed form at the first-hit error says something

`CompiledTwo.KSFirstHitTwoX` fails at `sksLTwo` (`Witnesses.SamplingTwo`: `sksL`, which accepts after one round,
samples nothing and opens nothing, with the two-stage design `toyTwo`), its silent prover and any hash, at the costs
`c_R = c_F = 0` (`compiledTwoKSFirstHitX_fails`): the three bounds it takes hold there, and what it concludes,
`CompiledTwo.FirstHitTwoXAt`, fails there (`compiledTwoFirstHitXAt_fails`). At coins `Bool` and SHA-512
(`FlockDeploy.sha512`) the second is the nontrivial lemma of `Flock.SecurityProofs.SpotFirstHitTwo`
(`compilerFirstHitTwoXAt_nontrivial`).

* The form's three bounds hold at the costs `0`: `sksLTwo` has no sessions, so `B_conf,X` finds nothing (`bconfX_pr`);
  its extractions open nothing under the statement's roots (`sksL_opensAt`) or under a replay block's interior roots
  (`sksLTwo_iopensAt`), so no run is off the boundary trace or off a committed interior (`sksLTwo_offRegX`,
  `sksLTwo_offIreg`).
* Its error at the costs `0` is `0` (`sksLTwo_εfirstTwoX`): `κ_s = 0` and `Pr[B] = 0`.
* It accepts always, while `P̄_a` at the footprint of its boundary trace read at the statement is `0` at every
  interior, so the headline at the bound `0` would bound `1` by `0`.
-/

@[expose] public section

namespace Claims.Witness.SamplingGeneric

open Claims.SamplingGeneric
open Claims.Sampling (footprintAt)
open Claims.Witness.Sampling (toyStmt toyPart rejCm)

/-- `sksLTwo`'s extractions open nothing under a replay block's interior roots: its commitment opens nothing. -/
theorem sksLTwo_iopensAt {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) {Ω : Type}
    (P : Prover Ω) (x : Msg) (ω : Ω) (r : Fin ((sksLTwo Coin).V.k x) → Coin) (c : (sksLTwo Coin).SlotEC x)
    (ρ g : ℕ) : (sksLTwo Coin).iopensAt H P x ω r c ρ g = none := by
  have hp : (sksLTwo Coin).parse x = some toyStmt := rfl
  have hc : (sksLTwo Coin).commit = rejCm := rfl
  simp only [CompiledTwo.iopensAt, hp]
  simp [Claims.Sampling.CommitmentScheme.Opens, hc, rejCm, OC.eval]

/-- No run of `sksLTwo` is off the boundary trace read at the statement: its extractions open nothing. -/
theorem sksLTwo_offRegX {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) {Ω : Type}
    [Fintype Ω] (P : Prover Ω) (x : Msg) :
    Pr (fun q : Ω × ((Fin ((sksLTwo Coin).V.k x) → Coin) × (sksLTwo Coin).SlotEC x) =>
      (run H P (sksLTwo Coin).V x q.1 q.2.1).out = .acc ∧ (sksLTwo Coin).OffRegX H P x q.1 q.2.1 q.2.2) = 0 :=
  Eq.trans (Pr_congr fun q : Ω × ((Fin ((sksLTwo Coin).V.k x) → Coin) × (sksLTwo Coin).SlotEC x) =>
    iff_false_intro fun hq => by
      obtain ⟨_, g, _, b, hb, _⟩ := hq
      exact Option.some_ne_none b (hb.symm.trans (sksL_opensAt (Coin := Coin) H P x q.1 q.2.1 q.2.2 g))) Pr_false

/-- No run of `sksLTwo` is off a committed interior: its extractions open nothing under a replay block's interior
roots. -/
theorem sksLTwo_offIreg {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) {Ω : Type}
    [Fintype Ω] (P : Prover Ω) (x : Msg) (Pρ : Claims.Sampling.Partition) :
    Pr (fun q : Ω × ((Fin ((sksLTwo Coin).V.k x) → Coin) × (sksLTwo Coin).SlotEC x) =>
      (run H P (sksLTwo Coin).V x q.1 q.2.1).out = .acc ∧ (sksLTwo Coin).OffIreg H P x q.1 Pρ q.2.1 q.2.2) = 0 :=
  Eq.trans (Pr_congr fun q : Ω × ((Fin ((sksLTwo Coin).V.k x) → Coin) × (sksLTwo Coin).SlotEC x) =>
    iff_false_intro fun hq => by
      obtain ⟨_, g, _, ρ, _, b, hb, _⟩ := hq
      exact Option.some_ne_none b (hb.symm.trans (sksLTwo_iopensAt H P x q.1 q.2.1 q.2.2 ρ g))) Pr_false

/-- `sksLTwo`'s first-hit error at the costs `0` is `0`. -/
theorem sksLTwo_εfirstTwoX {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) (x : Msg) :
    (sksLTwo Coin).εfirstTwoX H x silentP 0 0 = 0 := by
  have hκ : (sksLTwo Coin).κs x = 0 := sksL_κs (Coin := Coin) x
  have hb : PrB H (sksLTwo Coin).V silentP x = 0 := prB_accV H silentP x
  unfold CompiledTwo.εfirstTwoX
  rw [hκ, hb]
  simp

/-- **`CompiledTwo.FirstHitTwoXAt` fails** at `sksLTwo`'s silent prover, at the costs `0`: it accepts always, while
`P̄_a` at the footprint of its boundary trace read at the statement is `0` at every interior, at an error of `0`. -/
theorem compiledTwoFirstHitXAt_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ¬ (sksLTwo Coin).FirstHitTwoXAt H [] toyStmt toyPart toyTwo silentP 0 0 := by
  intro h
  unfold CompiledTwo.FirstHitTwoXAt at h
  have h1 := h.2.2 0 fun ι => (toyTwo_Pa
    (toy_footprintAt (val := (sksLTwo Coin).regValX H silentP []) (sksL_regValX H silentP [] 0) ι)).le
  have hp : Pr (fun p : Unit × (Fin ((sksLTwo Coin).V.k []) → Coin) =>
      (run H silentP (sksLTwo Coin).V [] p.1 p.2).out = .acc) = 1 :=
    Eq.trans (Pr_congr fun p : Unit × (Fin ((sksLTwo Coin).V.k []) → Coin) =>
      iff_true_intro (run_accV_out H silentP [] p.1 p.2)) Claims.ZKMal.Pr_true
  have h2 : (1 : ℝ) ≤ 0 + (sksLTwo Coin).εfirstTwoX H [] silentP 0 0 := hp.symm.trans_le h1
  rw [sksLTwo_εfirstTwoX] at h2
  norm_num at h2

/-- **`CompiledTwo.KSFirstHitTwoX` fails** at `sksLTwo`'s silent prover, at the costs `0`: the three bounds it takes
hold there (`BconfXLe`, `OffRegXLe`, `OffIregLe`), and what it concludes fails there
(`compiledTwoFirstHitXAt_fails`). -/
theorem compiledTwoKSFirstHitX_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ¬ (sksLTwo Coin).KSFirstHitTwoX H := by
  intro h
  have hz : (0 : ℝ) ≤ 2 * 0 / 2 ^ (256 : ℝ) := by simp
  have hR : (sksLTwo Coin).BconfXLe H [] silentP 0 :=
    (bconfX_pr H (sksLTwo Coin).toCompiled (fun _ _ _ => rfl) Unit silentP []).trans_le hz
  have hB : (sksLTwo Coin).OffRegXLe H [] silentP 0 := (sksLTwo_offRegX H silentP []).trans_le hz
  have hN : (sksLTwo Coin).OffIregLe H [] silentP toyTwo.part 0 :=
    (sksLTwo_offIreg H silentP [] toyTwo.part).trans_le hz
  exact compiledTwoFirstHitXAt_fails H
    (h [] toyStmt toyPart toyTwo (sksLTwo_scope []) rfl rfl rfl Unit silentP 0 0 hR hB hN)

end Claims.Witness.SamplingGeneric

/-! ## Stated over the spec -/

namespace Claims.SamplingGeneric

open Claims.Witness (silentP)
open Claims.Witness.SamplingGeneric
open Claims.Witness.Sampling (toyStmt toyPart)

/-- **What `SpotFirstHitTwo` concludes fails** (`CompiledTwo.FirstHitTwoXAt`) at `sksLTwo`, at its silent prover and
the costs `0`: it accepts with `P̄_a` at the footprint of its boundary trace read at the statement `0` at every
interior. The three bounds the form takes, which the guarantee proves at its code, hold there at those costs
(`compiledTwoKSFirstHitX_fails`). -/
theorem compilerFirstHitTwoXAt_nontrivial :
    ¬ (sksLTwo Bool).FirstHitTwoXAt FlockDeploy.sha512 [] toyStmt toyPart toyTwo silentP 0 0 :=
  compiledTwoFirstHitXAt_fails _

end Claims.SamplingGeneric
