import Proofs.Flock.Bridge.V5.Statement
import Proofs.Flock.Bridge.Assembly

/-!
# V5's assembly (`Bridge.AssemblyTargetV5`, proved)

`Assembly.lean` at the verifier's public inputs: at `kd = 0` the miss term is `1`; otherwise `EndToEndHiddenV5` at
`K₁ = min K₀ (n − kd + 1)` bounds the event, since outside `RelS` at least `K₁` slots are faulty population slots
(`CountTarget`), and each is a unit wrong on the registration with canonical openings, as a unit right in that sense is
not faulty (`FaultyRightTargetC`, with its transport from `TransportsLiveV5` at the wires `constsOfP` doesn't fix).

`constsOfP_isInput`: `constsOfP` fixes only input gates, the constant, the zero and the public gates, each below
`2 + n·B` (`Layout.keyPubOf_spec`), the program inputs of `keyProgP` (`Layout.unit_uniform_input`).
-/

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

/-- **The constants at public inputs are input gates**: `constsOfP` fixes `oneG`, `zeroG` and the public gates, all
below `2 + n·B`, which are program inputs of `keyProgP` (`Layout.unit_uniform_input`), so no unit's. -/
theorem _root_.Flock.Bridge.constsOfP_isInput {c : Flock.Circuit} {hU : Built c.unit} {n : ℕ} {pi : HmIn.Ports}
    {pub : Public} (outs : Finset (Fin (Discharge.Layout.keyN c hU n))) {g : Fin (Discharge.Layout.keyN c hU n)}
    {b : Bool} (h : Bridge.constsOfP c hU n pi pub g = some b) :
    (((keyProgP c pi hU n).circuit outs).op g).isInput = true := by
  have hg : g.val < 2 + n * Discharge.Layout.keyB c hU := by
    unfold Bridge.constsOfP at h
    split at h
    · rename_i b' hb'
      simp only [Bridge.constsOf] at hb'
      split_ifs at hb' with h1 h2
      · subst h1; show 0 < 2 + n * Discharge.Layout.keyB c hU; omega
      · subst h2; show 1 < 2 + n * Discharge.Layout.keyB c hU; omega
    · obtain ⟨x, hx, -⟩ := Option.map_eq_some_iff.1 h
      exact (Discharge.Layout.keyPubOf_spec hx).2.1
  exact (((keyProgP c pi hU n).partition outs).input_iff g).1
    (Discharge.Layout.unit_uniform_input (k := Discharge.Placed.inGates n (Discharge.Layout.keyB c hU))
      (R := Discharge.Placed.circuitRows c hU)
      (ins := Discharge.Placed.insOf n (Discharge.Layout.keyB c hU)
        (Discharge.Aliased.keyIdxP c pi (Discharge.Layout.keyB c hU)))
      (one := Discharge.Placed.oneGate n (Discharge.Layout.keyB c hU))
      (hone := Discharge.Placed.insOf_ne_one n (Discharge.Layout.keyB c hU)
        (Discharge.Aliased.keyIdxP c pi (Discharge.Layout.keyB c hU)))
      n g hg)

/-- **V5's assembly** (`Bridge.AssemblyTargetV5`). -/
theorem _root_.Flock.Bridge.assemblyTargetV5 : Bridge.AssemblyTargetV5 := by
  intro hT hF _ hC hE n Reg I own ht D hD hU pi hpi pub hpub hscope hv2 hdup hroots desc _ _ kd hkd hdiv dj S₀ d₀ y₀ wr
    dg N Ω _ _ Lb src hA3 hTags hCust k Rw M hk hRw hM ρ hρ hr t' ht' K₀ hK₀ τ qF qS q L P vb σ Xr δ Cc Lm rows roots Cd
    κ hTab hLink
  by_cases hk0 : kd = 0
  · refine (FlockSoundness.Game.prob_le_one _ _ τ).trans (le_trans (le_of_eq ?_) (le_self_add.trans le_self_add))
    rw [hk0]
    simp
  have hkd1 : 1 ≤ kd := Nat.pos_of_ne_zero hk0
  have hTu := hT n Reg I own ht D hD hU pi hpi pub hpub hscope hv2 hdup hroots kd hkd hdiv dj S₀ d₀ y₀ wr dg N Ω Lb src
    hTags hCust τ
  have hκin : ∀ g b, κ g = some b → (Cc.op g).isInput = true := fun g b h =>
    Bridge.constsOfP_isInput (Discharge.Layout.outsOfP D.unroll hU n pi) h
  have hsub : (Claims.Sampling.faulty Claims.Sampling.sha512Msg Cd (Bridge.stmtOf P κ kd desc q roots)
      (Bridge.partOf P) (Bridge.witDep Cc Cd Claims.Sampling.sha512Msg roots Xr)).filter (· < n) ⊆
      (wrongRegZc execArith P (planZHJP dj y₀) (tabZHJP dj y₀) vb
        (fun g o => Bridge.canonAt rows κ g (vb.bit (vb.val o) g) (vb.val o) o.2.2) (reg σ) Xr).map
        Fin.valEmbedding := by
    classical
    intro v hv
    obtain ⟨hv, hlt⟩ := Finset.mem_filter.1 hv
    refine Finset.mem_map.2 ⟨⟨v, hlt⟩, ?_, rfl⟩
    by_contra hu
    obtain ⟨-, op, hop⟩ := not_not.1 (mt (mem_wrongRegZc _ _ _ _ _ _).2 hu)
    refine hF P vb.pos (fun o g => vb.bit (vb.val o) g) id vb.com (vb.cs (reg σ))
      (fun g o => Bridge.canonAt rows κ g (vb.bit (vb.val o) g) (vb.val o) o.2.2)
      Claims.Sampling.sha512Msg Cd roots Xr κ ⟨v, hlt⟩ hκin ⟨op, hop⟩
      (fun g hg hn o ho hc => hTu ⟨v, hlt⟩ g hg hn o ho (hc.2 hn)) ?_ kd desc q hv
    intro g hg b hb
    obtain ⟨-, hbit, hcan⟩ := hop g hg
    exact hbit.symm.trans (hcan.1 b hb)
  have hcard : ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg Cd δ (Bridge.stmtOf P κ kd desc q roots)
      (Bridge.partOf P) (Bridge.witDep Cc Cd Claims.Sampling.sha512Msg roots Xr) →
      min K₀ (n - kd + 1) ≤ (wrongRegZc execArith P (planZHJP dj y₀) (tabZHJP dj y₀) vb
        (fun g o => Bridge.canonAt rows κ g (vb.bit (vb.val o) g) (vb.val o) o.2.2) (reg σ) Xr).card := fun hrel =>
    (hC P kd K₀ hkd1 hkd _ _ κ desc q roots _ hrel).trans
      ((Finset.card_le_card hsub).trans (Finset.card_map _).le)
  have hE' := hE n Reg I own ht D hD hU pi hpi pub hpub hscope kd hkd hdiv dj S₀ d₀ y₀ wr dg N Ω Lb src hA3 hTags hCust
    k Rw M hk hRw hM ρ hρ hr t' ht' (min K₀ (n - kd + 1)) ((min_le_left _ _).trans hK₀) τ qF qS hTab hLink
  rw [Bridge.choose_min_count hkd1 hkd] at hE'
  refine le_trans (FlockSoundness.Game.prob_mono ?_ _ τ) hE'
  exact fun o ho => ⟨ho.1, hcard ho.2⟩

end Flock.Guarantees
