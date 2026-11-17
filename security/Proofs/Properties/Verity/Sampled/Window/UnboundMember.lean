module

public import Proofs.Properties.Verity.Sampled.Window.Unbound
public import Proofs.Properties.Verity.Sampled.Unbound
public import Proofs.Flock.Bridge.Deployed
public import Proofs.Flock.Bridge.FaultyRightC
public import Proofs.Flock.Bridge.V5.Transports
public import Proofs.Flock.Bridge.TransportsRefs

/-! The claim at one member of a window. -/

@[expose] public section

namespace Flock.Guarantees.MemberV5

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open scoped ENNReal

open Classical

variable {Reg : Type} {nJ : ℕ} (m : MemberV5 Reg nJ)

theorem Cc_eq (h : m.pi = ({} : HmIn.Ports)) : Flock.Bridge.keyCircuit m.D m.hU m.n = m.Cc := by
  have key : ∀ pi : HmIn.Ports, pi = {} → Flock.Bridge.keyCircuit m.D m.hU m.n =
      (FlockSoundness.Discharge.Placed.keyProgP m.D.unroll pi m.hU m.n).circuit
        (FlockSoundness.Discharge.Layout.outsOfP m.D.unroll m.hU m.n pi) := by
    rintro _ rfl
    show (FlockSoundness.Discharge.Placed.keyProg m.D.unroll m.hU m.n).circuit
        (FlockSoundness.Discharge.Layout.outsOf m.D.unroll m.hU m.n) =
      (FlockSoundness.Discharge.Placed.keyProgP m.D.unroll {} m.hU m.n).circuit
        (FlockSoundness.Discharge.Layout.outsOfP m.D.unroll m.hU m.n {})
    rw [FlockSoundness.Discharge.Layout.outsOfP_empty,
      FlockSoundness.Discharge.Placed.keyProgP_empty (c := m.D.unroll) (pi := {}) rfl m.hU m.n]
  exact key m.pi h

theorem P_heq (h : m.pi = ({} : HmIn.Ports)) : HEq (Flock.Bridge.keyPartition m.D m.hU m.n) m.P := by
  have key : ∀ pi : HmIn.Ports, pi = {} → HEq (Flock.Bridge.keyPartition m.D m.hU m.n)
      ((FlockSoundness.Discharge.Placed.keyProgP m.D.unroll pi m.hU m.n).partition
        (FlockSoundness.Discharge.Layout.outsOfP m.D.unroll m.hU m.n pi)) := by
    rintro _ rfl
    show HEq ((FlockSoundness.Discharge.Placed.keyProg m.D.unroll m.hU m.n).partition
        (FlockSoundness.Discharge.Layout.outsOf m.D.unroll m.hU m.n))
      ((FlockSoundness.Discharge.Placed.keyProgP m.D.unroll {} m.hU m.n).partition
        (FlockSoundness.Discharge.Layout.outsOfP m.D.unroll m.hU m.n {}))
    rw [FlockSoundness.Discharge.Layout.outsOfP_empty,
      FlockSoundness.Discharge.Placed.keyProgP_empty (c := m.D.unroll) (pi := {}) rfl m.hU m.n]
  exact key m.pi h

theorem consts_eq (h : m.pi = ({} : HmIn.Ports)) (g : Fin m.Cc.N) :
    Verity.parsedConsts (some ⟨m.D, m.hU, m.n⟩) g = m.κ g := by
  have key : ∀ pi : HmIn.Ports, pi = {} →
      ∀ g : Fin ((FlockSoundness.Discharge.Placed.keyProgP m.D.unroll pi m.hU m.n).circuit
        (FlockSoundness.Discharge.Layout.outsOfP m.D.unroll m.hU m.n pi)).N,
        Verity.parsedConsts (some ⟨m.D, m.hU, m.n⟩) g = Flock.Bridge.constsOfP m.D.unroll m.hU m.n pi m.pub g := by
    rintro _ rfl g
    exact (Verity.parsedConsts_some ⟨m.D, m.hU, m.n⟩ g).trans
      ((Flock.Bridge.constWiresOf_eq_ite m.D.unroll m.hU m.n g).symm.trans
        (Flock.Bridge.constsOfP_empty m.D.unroll m.hU m.n m.pub g).symm)
  exact key m.pi h g

theorem κ_isInput (g : Fin m.Cc.N) (b : Bool) (h : m.κ g = some b) : (m.Cc.op g).isInput = true :=
  Flock.Bridge.constsOfP_isInput (FlockSoundness.Discharge.Layout.outsOfP m.D.unroll m.hU m.n m.pi) h

theorem unboundFaulty : m.UnboundFaulty := by
  intro r op tr Xv h
  obtain ⟨hc, hr, hop, hcon, htr⟩ := h
  subst hr
  refine Verity.Sampled.unbound_subset_of_parsed hc.parsed (m.Cc_eq hc.noPublic) (m.P_heq hc.noPublic) (κ₀ := m.κ)
    (κ := m.κ) HEq.rfl (m.consts_eq hc.noPublic) (H := Claims.Sampling.sha512Msg) (Cm := m.Cd) (roots := m.roots)
    (val := (m.wit Xv).val Claims.Sampling.sha512Msg m.Cd m.stmt)
    (fun _ _ h => Flock.Bridge.opens_of_val h) ?_ op hop hcon _ htr
  intro opens hK hO
  exact Flock.Bridge.unbound_subset_faulty_partOf m.P m.κ (show [m.n].sum = m.n by simp) Claims.Sampling.sha512Msg
    m.Cd m.stmt rfl (m.wit Xv) opens m.κ_isInput hK hO

theorem faultyIncorrect : m.FaultyIncorrect := by
  intro hc R Xv u hu
  by_contra hu'
  obtain ⟨-, op, hop⟩ := not_not.1
    (mt (mem_incorrectRegZc execArith m.P (planZHJP m.dj m.y₀) (tabZHJP m.dj m.y₀) m.vb m.canon).2 hu')
  refine Flock.Bridge.faultyRightTargetC m.P m.vb.pos (fun o g => m.vb.bit (m.vb.val o) g) id m.vb.com (m.vb.cs R)
    m.canon Claims.Sampling.sha512Msg m.Cd m.roots Xv m.κ u m.κ_isInput ⟨op, hop⟩ ?_ ?_ [m.n] .all []
    ⟨"", 0, []⟩ (by simp) hu
  · intro g _ hn o ho hcan
    obtain ⟨⟨r, hr, hv, hvw⟩, hy⟩ := hcan.2 hn
    obtain ⟨p, v, y⟩ := o
    exact Flock.Bridge.transports_shapedP (dj := m.dj) (y₀ := m.y₀) (ht := m.ht) (hc := m.hcS) (hscope := m.hscope)
      (hU := m.hU) (hpi := m.hpi) (hpub := m.hpub) R (Flock.Bridge.rowsOk_of_scopeOkH m.hscope)
      (Flock.Bridge.rootsDistinctL_of_fixes (Flock.Bridge.LeafMap.ofPublic_dom _ _ _) hc.distinct
        (show Flock.Bridge.RootFixesDomainL m.D.unroll (Flock.Bridge.LeafMap.ofPublic m.D.unroll m.pub m.n) from
          hc.fixes))
      (fun u q => Flock.Bridge.ref_lt_cntP m.hpub u.isLt q.isLt) hr p v y hv hvw hy ho
  · intro g hg b hb
    obtain ⟨-, hbit, hcan⟩ := hop g hg
    exact hbit.symm.trans (hcan.1 b hb)

end Flock.Guarantees.MemberV5
