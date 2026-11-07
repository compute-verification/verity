module

public import Proofs.Flock.Bridge.V5.Statement
public import Proofs.Flock.Bridge.Assembly
public import Proofs.Flock.Bridge.Count

/-!
# V5's assembly (`Bridge.AssemblyTargetV5`, proved)

`Assembly.lean` at the verifier's public inputs, under the sampling design the verifier samples under from the registration, which
`PopMatches` puts on the registration's strata (`Bridge.lawOn_drawsUnder`): outside `RelS` at `δ` the faulty population
slots' `P_a` is at most `δ` (`pa_partOf_window`), and `EndToEndHiddenV5` at those proof blocks bounds the event, since
each is a proof block faulty on the registration with canonical openings, so wrong, as a proof block right in that sense
is not faulty (`FaultyRightTargetC`, with its transport from `TransportsLiveV5` at the wires `constsOfP` doesn't fix).
Inside `RelS` the event is empty. No count of faulty slots enters, so the bound holds at every sampling design `Bridge.LawOn`
admits.

`constsOfP_isInput`: `constsOfP` fixes only input gates, the constant, the zero and the public gates, each below
`2 + n·B` (`Layout.keyPubOf_spec`), the program inputs of `keyProgP` (`Layout.unit_uniform_input`).
-/

@[expose] public section

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

/-- **The constants at public inputs are input gates**: `constsOfP` fixes `oneG`, `zeroG` and the public gates, all
below `2 + n·B`, which are program inputs of `keyProgP` (`Layout.unit_uniform_input`), so no proof block's. -/
theorem _root_.Flock.Bridge.constsOfP_isInput {c : Flock.Circuit} {hU : Built c.unit} {n : ℕ} {pi : HmIn.Ports}
    {pub : Public} (outs : Finset (Fin (Discharge.Layout.keyN c hU n))) {g : Fin (Discharge.Layout.keyN c hU n)}
    {b : Bool} (h : Bridge.constsOfP c hU n pi pub g = some b) :
    (((keyProgP c pi hU n).circuit outs).op g).isInput = true := by
  have hg : g.val < 2 + n * Discharge.Layout.keyB c hU := by
    unfold Bridge.constsOfP at h
    split at h
    · rename_i b' hb'
      simp only [Bridge.constWiresOf] at hb'
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
  intro hT hF hE n Reg I own ht D hD hU pi hpi pub hpub hscope hv2 hdup hroots desc qr hpop lj tb law kd hlaw hb hdiv dj
    S₀ d₀ y₀ wr dg N Ω _ _ Lb src hA3 hTags hCust k Rw M hk hRw hM ρ hρ hr t' ht' B τ qF qS q L P vb σ Xr Cc Lm rows
    roots Cd κ r ns δ hTab hLink
  classical
  have hns : ns.sum = n := Bridge.nsOf_sum hpop
  have hon : Bridge.LawOn law ns kd := Bridge.lawOn_drawsUnder hpop hlaw hb
  by_cases hrel : Claims.Sampling.RelS Claims.Sampling.sha512Msg Cd δ (Bridge.stmtOf P κ r desc q roots)
      (Bridge.partOf P κ ns) (Bridge.witDep Cc Cd Claims.Sampling.sha512Msg roots Xr)
  · exact (FlockSoundness.Game.prob_mono (q := fun _ => False) (fun o ho => ho.2 hrel) _ τ).trans
      ((FlockSoundness.Game.prob_false _ τ).le.trans zero_le)
  obtain ⟨F, hFe⟩ : ∃ F, Claims.Sampling.faulty Claims.Sampling.sha512Msg Cd (Bridge.stmtOf P κ r desc q roots)
      (Bridge.partOf P κ ns) (Bridge.witDep Cc Cd Claims.Sampling.sha512Msg roots Xr) = F := ⟨_, rfl⟩
  have hle : (Bridge.designOf r).Pa (Bridge.partOf P κ ns) F ≤ δ := by
    rw [← hFe]
    exact not_lt.1 hrel
  have hW : (Finset.univ.filter fun u : Fin n => u.val ∈ F).map Fin.valEmbedding = F.filter (· < n) := by
    ext v
    simp only [Finset.mem_map, Finset.mem_filter, Finset.mem_univ, true_and, Fin.valEmbedding_apply]
    constructor
    · rintro ⟨u, hu, rfl⟩; exact ⟨hu, u.isLt⟩
    · rintro ⟨hv, hlt⟩; exact ⟨⟨v, hlt⟩, hv, rfl⟩
  rw [Bridge.pa_partOf_window P hns, Bridge.window_zero_filter, ← hW] at hle
  have hTu := hT n Reg I own ht D hD hU pi hpi pub hpub hscope hv2 hdup hroots law kd hdiv dj S₀ d₀ y₀ wr dg N L hTags
    hCust (Bridge.lawSized_size (Bridge.lawSized_drawsUnder hlaw hb) hCust.2) τ
  have hκin : ∀ g b, κ g = some b → (Cc.op g).isInput = true := fun g b h =>
    Bridge.constsOfP_isInput (Discharge.Layout.outsOfP D.unroll hU n pi) h
  have hsub : (Finset.univ.filter fun u : Fin n => u.val ∈ F) ⊆ wrongRegZc execArith P (planZHJP dj y₀)
      (tabZHJP dj y₀) vb (fun g o => Bridge.canonAt rows κ g (vb.bit (vb.val o) g) (vb.val o) o.2.2) (reg σ) Xr := by
    intro u hu
    have hv : u.val ∈ Claims.Sampling.faulty Claims.Sampling.sha512Msg Cd (Bridge.stmtOf P κ r desc q roots)
        (Bridge.partOf P κ ns) (Bridge.witDep Cc Cd Claims.Sampling.sha512Msg roots Xr) := by
      rw [hFe]
      exact (Finset.mem_filter.1 hu).2
    by_contra hu'
    obtain ⟨-, op, hop⟩ := not_not.1 (mt (mem_wrongRegZc _ _ _ _ _ _).2 hu')
    refine hF P vb.pos (fun o g => vb.bit (vb.val o) g) id vb.com (vb.cs (reg σ))
      (fun g o => Bridge.canonAt rows κ g (vb.bit (vb.val o) g) (vb.val o) o.2.2)
      Claims.Sampling.sha512Msg Cd roots Xr κ u hκin ⟨op, hop⟩
      (fun g hg hn o ho hc => hTu u g hg hn o ho (hc.2 hn)) ?_ ns r desc q hns hv
    intro g hg b hb
    obtain ⟨-, hbit, hcan⟩ := hop g hg
    exact hbit.symm.trans (hcan.1 b hb)
  have hE' := hE n Reg I own ht D hD hU pi hpi pub hpub hscope law kd ns hns hon hdiv dj S₀ d₀ y₀ wr dg N Ω Lb src hA3
    hTags hCust k Rw M hk hRw hM ρ hρ hr t' ht' (Finset.univ.filter fun u : Fin n => u.val ∈ F) τ qF qS hTab hLink
  refine le_trans (FlockSoundness.Game.prob_mono ?_ _ τ) (hE'.trans ?_)
  · exact fun o ho => ⟨ho.1, hsub⟩
  · exact add_le_add (add_le_add (ENNReal.ofReal_le_ofReal hle) le_rfl) le_rfl

end Flock.Guarantees
