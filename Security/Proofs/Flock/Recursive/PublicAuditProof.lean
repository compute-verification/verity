import Proofs.Flock.Recursive.PublicAudit
import Proofs.Flock.Recursive.FirewallZK
import Proofs.Flock.Recursive.Loads

/-!
# The recursive audit at public units: its proof

`Flock.SecurityProofs.PublicAudit : Flock.Guarantees.PublicAudit`, `RecursiveAudit`'s proof
(`Proofs.Flock.Recursive.Audit`) at a public inner unit, and `RecursiveFirewallPrivate`.

In (i), from the statement and the premises, as in `RecursiveAudit`'s:
* where some index has no session that types, the event is empty: a session that verifies as V*'s statement of its
  index loads it (`loads_of_verifiesV`, at the obligations `VStarParses` and `TagsFound`) and so types
  (`typable_of_accepts`), so at that index none verifies (`prob_sessionsR_untypable`);
* at a statement Lean's verifier doesn't build from its partition object and program, the event is empty
  (`AcceptsPublic` is its first conjunct);
* at one it builds, the statement is accepted (`AcceptsPublic.acceptsStmt`), so the class checks pass
  (`classOk_of_accepts`) and the compile premise holds at `execClassAt st` (`compiledForkSound_execClassAt`);
* the hidden statement is the class's one (`publicHidden st`), a constant of the run, so no description and no
  registered rows enter, and the raw event is bounded by the typed sessions' at the transported prover
  (`prob_recT_le`, `verifies_zsOf`).

(ii) is `RecursiveFirewallPrivate`, as in `RecursiveAudit`'s proof.

The inner model is the executable's (`Recursion.innerAt_leaf`): `innerAt` runs the compiled table at `Refine.leafPlain`,
and an accepted statement commits its rows with the scheme `RecVStar.innerTags` names, whose leaf that is.
`Recursion.vbridge_innerAt` gives the VBridge premise at `innerAt` from VBridge at the class and `fast100`'s schedule,
the model VBridge G is about (`FlockVBridge.Family.vbridge_of_family`).
-/

namespace FlockSoundness.Discharge.Recursion

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR

/-- **The model's leaf is the executable's**: `innerAt` runs the compiled table at `Refine.leafPlain`, and at a statement
Lean's verifier accepts, set up at `RecVStar.innerTags` (`acceptsStmt_merkle`), that leaf is its Merkle scheme's
(`Refine.leafPlain_leaf`). A change to the model's leaf, or to the scheme the inner tags name, fails here. -/
theorem innerAt_leaf {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
    {st : Flock.Stmt} (h : AcceptsStmt st) :
    Flock.Guarantees.innerAt T CT st = flockInnerCL T CT execArith Refine.H512 Refine.enc512 Refine.leafPlain
      (execClassAt st).cls (execClassAt st).regs (execClassAt st).sch ∧
    ∀ (row : Array F128) (salt : ByteArray) (y : List UInt8),
      (Refine.leafPlain.leaf row.toList y).1 = st.merkle.leaf row salt := by
  refine ⟨rfl, fun row salt y => ?_⟩
  rw [acceptsStmt_merkle h]
  exact Refine.leafPlain_leaf _ row salt y

/-- **VBridge at the inner statement's model, from VBridge at its class and `fast100`'s schedule**: at a statement whose
checks pass, `innerAt` is the compiled table at `Refine.leafPlain` and `Refine.schedOf s₀`
(`execClassAt_sch_of_fast100`), the model VBridge G proves VBridge at. The coins, messages and statements are
`schedInner`'s, which no schedule changes. -/
theorem vbridge_innerAt {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)]
    [∀ k, Nonempty (CT k)] {st : Flock.Stmt} (h : ClassOk (shapeOf st)) {s₀ : Flock.Schedule}
    (hs : Flock.fast100 st.m = some s₀) {n : ℕ} {L : Law n} {Reg Cm : Type} {S : ℕ} {Vs : Fin S → VStmt}
    {Reg' Reg₂ : Type}
    {zs : Reg → L.Ω → List (Cm × ((k : Fin r) → CT k)) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {x : Reg → L.Ω → Hidden (execClassAt st).cls}
    {wmsg : ((j : Fin S) → (Vs j).Bits) → List (List (SMsg T))}
    {Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv)}
    {cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg}
    {msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → List (SMsg T)}
    (hb : Flock.Assumptions.VBridge (I := flockInnerCL T CT execArith Refine.H512 Refine.enc512 Refine.leafPlain
      (execClassAt st).cls (execClassAt st).regs (Refine.schedOf s₀)) zs dg N k Rw x wmsg Sk cmt msg) :
    Flock.Premises.VBridge (I := Flock.Guarantees.innerAt T CT st) zs dg N k Rw x wmsg Sk cmt msg := by
  show Flock.Assumptions.VBridge (I := flockInnerCL T CT execArith Refine.H512 Refine.enc512 Refine.leafPlain
    (execClassAt st).cls (execClassAt st).regs (execClassAt st).sch) zs dg N k Rw x wmsg Sk cmt msg
  rw [execClassAt_sch_of_fast100 h hs]
  exact hb

end FlockSoundness.Discharge.Recursion

namespace Flock.SecurityProofs

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **The recursive audit at public units, as the verifier of record runs it.** -/
theorem PublicAudit : Flock.Guarantees.PublicAudit := by
  refine And.intro ?_ ?_
  swap
  · exact RecursiveFirewallPrivate
  intro nT r T CT _ _ n Ω _ _ draw Reg Cm Reg' Reg₂ obj prog st Vs zr hCust ish bind rows hob hF dg N σ wmsg Sk cmt
    msg hA3 k Rw M hk hRw hM ρ hρ hr hbr
  replace hob := vstarParsesScoped_of hob
  by_cases hT : Typed zr
  swap
  · left
    simp only [Typed, not_forall, not_exists] at hT
    obtain ⟨j₀, hj₀⟩ := hT
    refine prob_recGame_zero _ σ fun ω t s => ?_
    rw [prob_outerVR]
    exact prob_sessionsR_untypable hob (zr σ.1 ω t s.1) (fun j => (hCust σ.1 ω t s.1 j).1) (hF σ.1 ω t s.1) j₀
      (hj₀ (σ.1, ω, t, s.1)) _ (fun b hb => hb.2.1) _
  by_cases hacc : Flock.Guarantees.AcceptsPublic obj prog st
  swap
  · exact Or.inl (le_antisymm ((prob_mono_outcome _ σ fun o _ h => absurd h.1 hacc).trans (prob_false _ σ).le)
      (zero_le ..))
  refine Or.inr ⟨hT, ?_⟩
  have hin := compiledForkSound_execClassAt T CT (classOk_of_accepts hacc.acceptsStmt)
  have hr' : ∀ R ω t R₂ j S' R' i, rateZ (ZkHidden.planZHJ (Reg := Reg')
      (Flock.Guarantees.custodied zr hCust hT R ω t R₂ j).dj
      (Flock.Guarantees.custodied zr hCust hT R ω t R₂ j).y₀ S' R') i k ≤ ρ := by
    intro R ω t R₂ j S' R'
    dsimp only [Flock.Guarantees.custodied]
    by_cases hx' : (zr R ω t R₂ j).Typable
    · rw [zsOf_pos zr _ hT R ω t R₂ j hx']
      exact hr R ω t R₂ j _ _ hx'.y₀ S' R'
    · rw [zsOf_neg zr _ hT R ω t R₂ j hx']
      exact hr _ _ _ _ j _ _ _ S' R'
  rcases zk_sessions_recursive_fork_inner hA3 (Flock.Guarantees.custodied zr hCust hT) dg N hk hRw hM hρ hr'
    (Flock.Guarantees.transported zr hCust hT σ) (fun _ _ => Flock.Guarantees.publicHidden st) wmsg Sk cmt msg
    (hbr hT) Refine.H512 (clashPickCL T CT execArith Refine.H512 Refine.enc512 Refine.leafPlain (execClassAt st).cls
      (execClassAt st).regs (execClassAt st).sch) hin with hf | hf | hf | hf | hb
  · exact Or.inl hf
  · exact Or.inr (Or.inl hf)
  · exact Or.inr (Or.inr (Or.inl hf))
  · exact Or.inr (Or.inr (Or.inr (Or.inl hf)))
  · refine Or.inr (Or.inr (Or.inr (Or.inr (le_trans (prob_recT_le hob zr _ hT hF σ _
      (fun o => Flock.Guarantees.AcceptsPublic obj prog st ∧
        Flock.Guarantees.VerifiesV (Flock.Guarantees.custodied zr hCust hT o.1 o.2.1 o.2.2.1) o.2.2.2 ∧
        ¬ (Flock.Guarantees.innerAt T CT st).holds (Flock.Guarantees.publicHidden st))
      (fun o ho => ho.2.1) (fun o _ hall ho =>
        ⟨ho.1, verifies_zsOf zr _ hT o.1 o.2.1 o.2.2.1 o.2.2.2 hall (acceptsVR_of_V ho.2.1.1) ho.2.1.2, ho.2.2⟩))
      (le_trans (prob_mono_outcome _ (Flock.Guarantees.transported zr hCust hT σ)
        fun o _ h => ⟨h.2.1.1, h.2.2⟩) hb)))))

end Flock.SecurityProofs
