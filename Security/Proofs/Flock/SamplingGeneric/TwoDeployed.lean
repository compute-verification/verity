module

public import Proofs.Flock.SamplingGeneric.Masked
public import Proofs.Flock.SamplingGeneric.OpensTwo
import Proofs.Flock.Masked
import Proofs.Flock.SamplingGeneric
import Proofs.Flock.SamplingGeneric.Two

/-!
# The two-stage compiler at the deployed sampled record

`SamplingDeploy.compiledTwo F c iroots two rs`: the deployed sampled verifier over Flock (`SamplingDeploy.compiled`)
with the two-stage design it evaluates on a statement (`two`) and the replay blocks its first stage samples at a run's
coins (`rs`), as a compiled two-stage verifier (`Claims.SamplingGeneric.CompiledTwo`). Its sample, second stage,
sessions, reveal, interior roots and commitment are `compiled`'s.

The contract's clauses that read only the sampled verifier's code are named Props of `Code`'s functions:

* `Code.ReplayDraws` (`replayDraws`): on a statement in scope, the replay sample misses a set `T` of replay blocks
  with probability at most the first stage's `P_a(T)`;
* `Code.SecondDraws` (`secondDraws`): on a statement in scope, at a run's coins, fresh second-stage coins check no
  proof block of a set `B` among the units' first `n_PU` with probability at most the sampled replay blocks' escapes;
* `Code.RefusesTwo` (`refuses`): an accepted run's statement parses, and its partition and two-stage design evaluate
  and are in the two-stage scope;
* `Code.InteriorFixed` and `Code.DrawFixed` (`interior`, `fixed`), as at one stage.

`SessionOpensTwo` (`opens`, `Claims.Sampling.OpensTwo`) is `SessionOpens` read at a replay block: a checked unit's
Flock witness makes its proof block correct on its replay block's values, under the interior roots the run shows for
that replay block.

* `sessionsTwo_compiled`: the contract from those, with the sessions' runs and numbers (`fixed_ofDeploy`,
  `runs_ofDeploy`).
* `ksTwo_compiled`: `CompiledTwo.KS` from Flock's KS at `F` and its decoder's list bound, the commitment's binding and
  the contract (`compilerKSTwo` over `flockArgKS`).
* `ksTwo_masked`: at Flock's masked record and SHA-512, with Flock's KS, the list bound and the binding proved
  (`MaskedKS`, `listWithin_deployZk`, `commit_binds`), as `ks_masked`.
* `compiledTwoOne`, `sessionsTwo_one`, `ksTwo_one`: one session a run. At a code whose session doesn't read the unit,
  every slot of a sample has one key (`skey = 0`, `n_sess = 1`), so the extractor runs Flock's once a run, and
  `compiledTwo`'s contract gives the contract and KS there.
-/

@[expose] public section

namespace SamplingDeploy

open Claims
open Claims.Sampling (AuditStatement InScopeTwo forkOn cell escCell)

/-! ## The sampled verifier's two-stage code facts -/

/-- **Replay samples**: a fact about the sampled verifier's code, for a Lean verifier to discharge. -/
def Code.ReplayDraws (c : Code) (two : AuditStatement → Option Claims.Sampling.Design.Two)
    (rs : (x : Msg) → (Fin (c.V.k x) → FlockDeploy.Coin) → List ℕ) : Prop :=
  ∀ x s Pt d, c.parse x = some s → c.partition s = some Pt → two s = some d → InScopeTwo Pt d s →
    ∀ T ⊆ Finset.range d.part.blocks.length,
      Pr (fun r : Fin (c.V.k x) → FlockDeploy.Coin => ∀ ρ ∈ rs x r, ρ ∉ T) ≤ d.replay.Pa d.part T

/-- **Second samples**: a fact about the sampled verifier's code, for a Lean verifier to discharge. -/
def Code.SecondDraws (c : Code) (two : AuditStatement → Option Claims.Sampling.Design.Two)
    (rs : (x : Msg) → (Fin (c.V.k x) → FlockDeploy.Coin) → List ℕ) : Prop :=
  ∀ x s Pt d, c.parse x = some s → c.partition s = some Pt → two s = some d → InScopeTwo Pt d s →
    ∀ (r : Fin (c.V.k x) → FlockDeploy.Coin) (B : Finset ℕ),
      Pr (fun e : Fin (c.V.k x) → FlockDeploy.Coin =>
          ∀ u ∈ (c.checked x (forkOn (c.stage2 x) r e)).take (units c x), u ∉ B) ≤
        ∏ ρ ∈ (Finset.range d.part.blocks.length).filter (· ∈ rs x r),
          ∏ t ∈ Finset.range Pt.strata.length,
            escCell (cell Pt d.part ρ t).card (d.k.getD t 0) (B ∩ cell Pt d.part ρ t).card

/-- **Refuses at two stages**: a fact about the sampled verifier's code, for a Lean verifier to discharge. -/
def Code.RefusesTwo (H : Msg → Msg) (c : Code) (two : AuditStatement → Option Claims.Sampling.Design.Two) : Prop :=
  ∀ x (Ω : Type) (P : Prover Ω) ω r, (run H P c.V x ω r).out = .acc →
    ∃ s Pt d, c.parse x = some s ∧ c.partition s = some Pt ∧ two s = some d ∧ InScopeTwo Pt d s

section CompiledTwo

variable {T : Type} (F : Claims.Flock.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) T) (c : Code)
  (iroots : Msg → Seen → ℕ → List Msg) (two : AuditStatement → Option Claims.Sampling.Design.Two)
  (rs : (x : Msg) → (Fin (c.V.k x) → FlockDeploy.Coin) → List ℕ)

/-- **The deployed sampled verifier as a compiled two-stage verifier over Flock**: `compiled F c iroots` with the
statement's two-stage design `two` and the replay sample `rs`. -/
noncomputable abbrev compiledTwo :
    Claims.SamplingGeneric.CompiledTwo FlockDeploy.Coin (Claims.SamplingGeneric.Arg.ofFlock F) :=
  { toCompiled := compiled F c iroots, two := two, replaySample := rs }

/-- **The two-stage contract at the deployed record**: the code's facts, with the sessions' runs, numbers and
openings at a replay block. -/
theorem sessionsTwo_compiled (H : Msg → Msg) (hrd : c.ReplayDraws two rs) (hsd : c.SecondDraws two rs)
    (hr : c.RefusesTwo H two) (hi : c.InteriorFixed H iroots) (hf : c.DrawFixed)
    (hruns : Claims.Sampling.SessionRuns H (deployOf F c)) (hnum : Claims.Sampling.InnerNumbers (deployOf F c))
    (hopens : Claims.SamplingGeneric.SessionOpensTwo H (deployOf F c) two iroots) :
    (compiledTwo F c iroots two rs).Sessions H where
  replayDraws := hrd
  secondDraws := hsd
  slots _ _ := List.length_take_le _ _
  shared _ _ _ _ _ i j u u' _ hu hu' hk := by
    obtain rfl : i = j := hk
    obtain rfl : u = u' := Option.some.inj (hu.symm.trans hu')
    rfl
  unions x := ⟨(Claims.SamplingGeneric.ofDeploy_nPU (deployOf F c) iroots x).le,
    (Claims.SamplingGeneric.ofDeploy_nPU (deployOf F c) iroots x).le⟩
  refuses := hr
  interior := hi
  fixed := Claims.SamplingGeneric.fixed_ofDeploy (deployOf F c) iroots hf
  runs := Claims.SamplingGeneric.runs_ofDeploy (deployOf F c) iroots H hruns hnum
  opens x s Pt d Ω Pv ω r u xS vS w hs hP h2 hin hacc hu hsess hR :=
    hopens x s Pt d Ω Pv ω r u xS vS w hs hP h2 hin hacc
      ((Claims.SamplingGeneric.mem_ofDeploy_checked (deployOf F c) iroots).1 hu) hsess
      (Claims.SamplingGeneric.flock_check_of_R hR)
  keys x _ := Finset.card_image_le.trans ((Finset.card_range _).trans_le
    ((List.length_take_le _ _).trans_eq (Claims.SamplingGeneric.ofDeploy_nSess (deployOf F c) iroots x).symm))

/-- **The two-stage compiler at the deployed record**: `CompiledTwo.KS` from Flock's KS at `F` and its deployed
decoder's list bound, the commitment's binding and the two-stage contract (`compilerKSTwo`, `flockArgKS`). -/
theorem ksTwo_compiled (H : Msg → Msg) (hKS : Claims.Flock.KS H F)
    (hlist : Claims.Flock.ListWithin F.len F.enc F.θ F.decode) (hB : (commit c).Binds H)
    (hrd : c.ReplayDraws two rs) (hsd : c.SecondDraws two rs) (hr : c.RefusesTwo H two)
    (hi : c.InteriorFixed H iroots) (hf : c.DrawFixed) (hruns : Claims.Sampling.SessionRuns H (deployOf F c))
    (hnum : Claims.Sampling.InnerNumbers (deployOf F c))
    (hopens : Claims.SamplingGeneric.SessionOpensTwo H (deployOf F c) two iroots) :
    (compiledTwo F c iroots two rs).KS H :=
  Claims.SamplingGeneric.compilerKSTwo FlockDeploy.Coin _ _ H (Claims.SamplingGeneric.flockArgKS H F hKS hlist) hB
    (sessionsTwo_compiled F c iroots two rs H hrd hsd hr hi hf hruns hnum hopens)

/-- **The deployed two-stage verifier with one session a run**: `compiledTwo` with one key for every slot of a sample,
so its extractor runs Flock's once a run (`n_sess = 1`). -/
noncomputable abbrev compiledTwoOne :
    Claims.SamplingGeneric.CompiledTwo FlockDeploy.Coin (Claims.SamplingGeneric.Arg.ofFlock F) :=
  { compiledTwo F c iroots two rs with skey := fun _ _ _ => 0, nSess := fun _ => 1 }

/-- **The contract with one session a run**: at a code whose session doesn't read the unit, `compiledTwo`'s contract
is `compiledTwoOne`'s, every slot showing the run's one session (`shared`) and a sample holding one key (`keys`). -/
theorem sessionsTwo_one (H : Msg → Msg) (hS : (compiledTwo F c iroots two rs).Sessions H)
    (hone : ∀ x u u' v, c.session x u v = c.session x u' v) : (compiledTwoOne F c iroots two rs).Sessions H where
  replayDraws := hS.replayDraws
  secondDraws := hS.secondDraws
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

/-- **The deployed two-stage sampled verifier over Flock's masked record is knowledge sound at SHA-512**:
`ksTwo_compiled` at `deployZk`, with Flock's KS by `MaskedKS`, the list bound by `listWithin_deployZk`, binding by
`commit_binds` and the inner numbers by `innerNumbers_deploy`. It takes the sampled verifier's two-stage code facts and
its sessions' runs, scope (`SessionsIn`) and openings at a replay block. -/
theorem ksTwo_masked (c : Code) (iroots : Msg → Seen → ℕ → List Msg)
    (two : AuditStatement → Option Claims.Sampling.Design.Two)
    (rs : (x : Msg) → (Fin (c.V.k x) → FlockDeploy.Coin) → List ℕ)
    (hrd : c.ReplayDraws two rs) (hsd : c.SecondDraws two rs) (hr : c.RefusesTwo FlockDeploy.sha512 two)
    (hi : c.InteriorFixed FlockDeploy.sha512 iroots) (hf : c.DrawFixed)
    (hruns : Claims.Sampling.SessionRuns FlockDeploy.sha512 (deploy _root_.Flock.Readable.relation c))
    (hnum : SessionsIn c)
    (hopens : Claims.SamplingGeneric.SessionOpensTwo FlockDeploy.sha512 (deploy _root_.Flock.Readable.relation c)
      two iroots) :
    (compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) c iroots two rs).KS FlockDeploy.sha512 :=
  ksTwo_compiled (FlockDeploy.deployZk _root_.Flock.Readable.relation) c iroots two rs FlockDeploy.sha512
    _root_.Flock.SecurityProofs.MaskedKS (Claims.Sampling.listWithin_deployZk _) (commit_binds c _) hrd hsd hr hi hf
    hruns (innerNumbers_deploy _ c hnum) hopens

/-- **The deployed two-stage verifier with one session a run is knowledge sound at SHA-512**, at Flock's masked record
and a code whose session doesn't read the unit: `compilerKSTwo` at `sessionsTwo_one`, with Flock's KS, the list bound
and binding as in `ksTwo_masked`. -/
theorem ksTwo_one (c : Code) (iroots : Msg → Seen → ℕ → List Msg)
    (two : AuditStatement → Option Claims.Sampling.Design.Two)
    (rs : (x : Msg) → (Fin (c.V.k x) → FlockDeploy.Coin) → List ℕ)
    (hS : (compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) c iroots two rs).Sessions
      FlockDeploy.sha512)
    (hone : ∀ x u u' v, c.session x u v = c.session x u' v) :
    (compiledTwoOne (FlockDeploy.deployZk _root_.Flock.Readable.relation) c iroots two rs).KS FlockDeploy.sha512 :=
  Claims.SamplingGeneric.compilerKSTwo FlockDeploy.Coin _ _ FlockDeploy.sha512
    (Claims.SamplingGeneric.flockArgKS FlockDeploy.sha512 _ _root_.Flock.SecurityProofs.MaskedKS
      (Claims.Sampling.listWithin_deployZk _)) (commit_binds c _) (sessionsTwo_one _ c iroots two rs _ hS hone)

end SamplingDeploy
