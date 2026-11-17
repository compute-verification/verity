module

public import Proofs.Flock.SamplingGeneric.Masked
public import Proofs.Flock.SamplingGeneric.OpensTwo
import Proofs.Flock.Masked
import Proofs.Flock.SamplingGeneric
import Proofs.Flock.SamplingGeneric.Two

/-! The two-stage compiler at the deployed sampled record, its waits' counts the two stages' own. -/

@[expose] public section

namespace SamplingDeploy

open Claims
open Claims.Sampling (AuditStatement InScopeTwo forkOn cell escCell footBits sampleBlocks)

/-- A fact about the sampled verifier's code that no Lean verifier discharges. -/
def Code.ReplaySamples (c : Code) (two : AuditStatement → Option Claims.Sampling.Design.Two)
    (rs : (x : Msg) → (Fin (c.V.k x) → FlockDeploy.Coin) → List ℕ) : Prop :=
  ∀ x s Pt d, c.parse x = some s → c.partition s = some Pt → two s = some d → InScopeTwo Pt d s →
    ∀ T ⊆ Finset.range d.part.blocks.length,
      Pr (fun r : Fin (c.V.k x) → FlockDeploy.Coin => ∀ ρ ∈ rs x r, ρ ∉ T) ≤ d.replay.Pa d.part T

/-- A fact about the sampled verifier's code that no Lean verifier discharges. -/
def Code.SecondSamples (c : Code) (two : AuditStatement → Option Claims.Sampling.Design.Two)
    (rs : (x : Msg) → (Fin (c.V.k x) → FlockDeploy.Coin) → List ℕ) : Prop :=
  ∀ x s Pt d, c.parse x = some s → c.partition s = some Pt → two s = some d → InScopeTwo Pt d s →
    ∀ (r : Fin (c.V.k x) → FlockDeploy.Coin) (B : Finset ℕ),
      Pr (fun e : Fin (c.V.k x) → FlockDeploy.Coin =>
          ∀ u ∈ (c.checked x (forkOn (c.secondStage x) r e)).take (units c x), u ∉ B) ≤
        ∏ ρ ∈ (Finset.range d.part.blocks.length).filter (· ∈ rs x r),
          ∏ t ∈ Finset.range Pt.strata.length,
            escCell (cell Pt d.part ρ t).card (d.k.getD t 0) (B ∩ cell Pt d.part ρ t).card

/-- **Replay caps**: a fact about the sampled verifier's code, for a Lean verifier to discharge. -/
def Code.ReplayCap (H : Msg → Msg) (c : Code) (two : AuditStatement → Option Claims.Sampling.Design.Two)
    (rs : (x : Msg) → (Fin (c.V.k x) → FlockDeploy.Coin) → List ℕ) (cap : Msg → ℕ) : Prop :=
  ∀ x s Pt d, c.parse x = some s → c.partition s = some Pt → two s = some d → InScopeTwo Pt d s →
    ∀ (Ω : Type) (P : Prover Ω) ω (r e : Fin (c.V.k x) → FlockDeploy.Coin),
      (run H P c.V x ω (forkOn (c.secondStage x) r e)).out = .acc → (sampleBlocks Pt d.part (rs x r)).card ≤ cap x

/-- **Refuses at two stages**: a fact about the sampled verifier's code, for a Lean verifier to discharge. -/
def Code.RefusesTwo (H : Msg → Msg) (c : Code) (two : AuditStatement → Option Claims.Sampling.Design.Two) : Prop :=
  ∀ x (Ω : Type) (P : Prover Ω) ω r, (run H P c.V x ω r).out = .acc →
    ∃ s Pt d, c.parse x = some s ∧ c.partition s = some Pt ∧ two s = some d ∧ InScopeTwo Pt d s

/-- **A statement's footprint bits**: `footBits` at its partition and two-stage design, `0` where either doesn't
evaluate. -/
noncomputable def Code.footOf (c : Code) (two : AuditStatement → Option Claims.Sampling.Design.Two) (x : Msg) : ℕ :=
  match c.parse x with
  | some s =>
    match c.partition s, two s with
    | some Pt, some d => footBits Pt d
    | _, _ => 0
  | none => 0

theorem Code.footOf_eq {c : Code} {two : AuditStatement → Option Claims.Sampling.Design.Two} {x : Msg}
    {s : AuditStatement} {Pt : Claims.Sampling.Partition} {d : Claims.Sampling.Design.Two} (hs : c.parse x = some s)
    (hPt : c.partition s = some Pt) (h2 : two s = some d) : c.footOf two x = footBits Pt d := by
  simp only [Code.footOf, hs, hPt, h2]

section CompiledTwo

variable {T : Type} (F : Claims.Flock.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) T) (c : Code)
  (iroots : Msg → Seen → ℕ → List Msg) (two : AuditStatement → Option Claims.Sampling.Design.Two)
  (rs : (x : Msg) → (Fin (c.V.k x) → FlockDeploy.Coin) → List ℕ) (cap : Msg → ℕ)

/-- **The deployed sampled verifier as a compiled two-stage verifier over Flock**: `compiledAt F c iroots` with the
statement's two-stage design `two`, the replay sample `rs`, and its waits' counts the statement's footprint bits
(`n_out`, `Code.footOf`) and the replay sample's cap (`n_in`). -/
noncomputable abbrev compiledTwo :
    Claims.SamplingGeneric.CompiledTwo FlockDeploy.Coin (Claims.SamplingGeneric.Arg.ofFlock F) :=
  { toCompiled := { compiledAt F c iroots with nOut := c.footOf two, nIn := cap }, two := two, replaySample := rs }

theorem sessionsTwo_compiled (H : Msg → Msg) (hrd : c.ReplaySamples two rs) (hsd : c.SecondSamples two rs)
    (hcap : c.ReplayCap H two rs cap) (hr : c.RefusesTwo H two) (hi : c.InteriorFixed H iroots) (hf : c.SampleFixed)
    (hruns : Claims.Sampling.SessionRunsTo H (deployOf F c))
    (hnum : Claims.Sampling.InnerNumbersAt (deployOf F c) c.unitN₀)
    (hopens : Claims.SamplingGeneric.SessionOpensTwo H (deployOf F c) two iroots) :
    (compiledTwo F c iroots two rs cap).Sessions H where
  replaySamples := hrd
  secondSamples := hsd
  slots _ _ := List.length_take_le _ _
  shared _ _ _ _ _ i j u u' _ hu hu' hk := by
    obtain rfl : i = j := hk
    obtain rfl : u = u' := Option.some.inj (hu.symm.trans hu')
    rfl
  unions x s Pt d hs hPt h2 hin := ⟨(Code.footOf_eq hs hPt h2).ge, hcap x s Pt d hs hPt h2 hin⟩
  refuses := hr
  interior := hi
  fixed := Claims.SamplingGeneric.fixed_ofDeployAt (deployOf F c) iroots c.unitN₀ hf
  runs := Claims.SamplingGeneric.runs_ofDeployAt (deployOf F c) iroots c.unitN₀ H hruns hnum
  opens x s Pt d Ω Pv ω r u xS vS w hs hP h2 hin hacc hu hsess hR :=
    hopens x s Pt d Ω Pv ω r u xS vS w hs hP h2 hin hacc
      ((Claims.SamplingGeneric.mem_ofDeployAt_checked (deployOf F c) iroots c.unitN₀).1 hu) hsess
      (Claims.SamplingGeneric.flock_check_of_R hR)
  keys x _ := Finset.card_image_le.trans ((Finset.card_range _).trans_le
    ((List.length_take_le _ _).trans_eq
      (Claims.SamplingGeneric.ofDeployAt_nSess (deployOf F c) iroots c.unitN₀ x).symm))

theorem ksTwo_compiled (H : Msg → Msg) (hKS : Claims.Flock.KS H F)
    (hlist : Claims.Flock.ListWithin F.len F.enc F.θ F.decode) (hB : (commit c).Binds H)
    (hrd : c.ReplaySamples two rs) (hsd : c.SecondSamples two rs) (hcap : c.ReplayCap H two rs cap)
    (hr : c.RefusesTwo H two) (hi : c.InteriorFixed H iroots) (hf : c.SampleFixed)
    (hruns : Claims.Sampling.SessionRunsTo H (deployOf F c))
    (hnum : Claims.Sampling.InnerNumbersAt (deployOf F c) c.unitN₀)
    (hopens : Claims.SamplingGeneric.SessionOpensTwo H (deployOf F c) two iroots) :
    (compiledTwo F c iroots two rs cap).KS H :=
  Claims.SamplingGeneric.compilerKSTwo FlockDeploy.Coin _ _ H (Claims.SamplingGeneric.flockArgKS H F hKS hlist) hB
    (sessionsTwo_compiled F c iroots two rs cap H hrd hsd hcap hr hi hf hruns hnum hopens)

noncomputable abbrev compiledTwoOne :
    Claims.SamplingGeneric.CompiledTwo FlockDeploy.Coin (Claims.SamplingGeneric.Arg.ofFlock F) :=
  { compiledTwo F c iroots two rs cap with skey := fun _ _ _ => 0, nSess := fun _ => 1 }

theorem sessionsTwo_one (H : Msg → Msg) (hS : (compiledTwo F c iroots two rs cap).Sessions H)
    (hone : ∀ x u u' v, c.session x u v = c.session x u' v) : (compiledTwoOne F c iroots two rs cap).Sessions H where
  replaySamples := hS.replaySamples
  secondSamples := hS.secondSamples
  slots := hS.slots
  shared x _ _ _ _ _ _ u u' _ _ _ _ := hone x u u' _
  unions := hS.unions
  refuses := hS.refuses
  interior := hS.interior
  fixed := hS.fixed
  runs := hS.runs
  opens := hS.opens
  keys _ _ := (Finset.card_le_card (Finset.image_subset_iff.2 fun _ _ => Finset.mem_singleton_self 0)).trans
    (Finset.card_singleton 0).le

end CompiledTwo

theorem ksTwo_masked (c : Code) (iroots : Msg → Seen → ℕ → List Msg)
    (two : AuditStatement → Option Claims.Sampling.Design.Two)
    (rs : (x : Msg) → (Fin (c.V.k x) → FlockDeploy.Coin) → List ℕ) (cap : Msg → ℕ)
    (hrd : c.ReplaySamples two rs) (hsd : c.SecondSamples two rs) (hcap : c.ReplayCap FlockDeploy.sha512 two rs cap)
    (hr : c.RefusesTwo FlockDeploy.sha512 two) (hi : c.InteriorFixed FlockDeploy.sha512 iroots) (hf : c.SampleFixed)
    (hruns : Claims.Sampling.SessionRunsTo FlockDeploy.sha512 (deploy _root_.Flock.Readable.relation c))
    (hnum : SessionsIn c)
    (hopens : Claims.SamplingGeneric.SessionOpensTwo FlockDeploy.sha512 (deploy _root_.Flock.Readable.relation c)
      two iroots) :
    (compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) c iroots two rs cap).KS FlockDeploy.sha512 :=
  ksTwo_compiled (FlockDeploy.deployZk _root_.Flock.Readable.relation) c iroots two rs cap FlockDeploy.sha512
    _root_.Flock.SecurityProofs.MaskedKS (Claims.Sampling.listWithin_deployZk _) (commit_binds c _) hrd hsd hcap hr hi
    hf hruns (innerNumbersAt_deploy _ c hnum) hopens

theorem ksTwo_one (c : Code) (iroots : Msg → Seen → ℕ → List Msg)
    (two : AuditStatement → Option Claims.Sampling.Design.Two)
    (rs : (x : Msg) → (Fin (c.V.k x) → FlockDeploy.Coin) → List ℕ) (cap : Msg → ℕ)
    (hS : (compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) c iroots two rs cap).Sessions
      FlockDeploy.sha512)
    (hone : ∀ x u u' v, c.session x u v = c.session x u' v) :
    (compiledTwoOne (FlockDeploy.deployZk _root_.Flock.Readable.relation) c iroots two rs cap).KS FlockDeploy.sha512 :=
  Claims.SamplingGeneric.compilerKSTwo FlockDeploy.Coin _ _ FlockDeploy.sha512
    (Claims.SamplingGeneric.flockArgKS FlockDeploy.sha512 _ _root_.Flock.SecurityProofs.MaskedKS
      (Claims.Sampling.listWithin_deployZk _)) (commit_binds c _) (sessionsTwo_one _ c iroots two rs cap _ hS hone)

end SamplingDeploy
