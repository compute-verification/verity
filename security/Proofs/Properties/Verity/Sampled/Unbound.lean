module

public import Proofs.Properties.Verity.Sampled
public import Proofs.Properties.Verity.Sampled.UnboundBridge
public import Proofs.Flock.Bridge.PopMatches
public import Proofs.Flock.Bridge.UnitRows
public import Proofs.Flock.Bridge.V5.Assembly
public import Proofs.Flock.Soundness.Discharge.Placed.KeyedP
public import Proofs.Flock.Soundness.Discharge.Aliased.KeysP
public import Proofs.Flock.Soundness.Discharge.Layout.KeyProgP
public import Proofs.Flock.Soundness.Discharge.Layout.OutsP

/-! The unbound units are faulty at the session. -/

@[expose] public section

namespace FlockSoundness.Discharge.Layout

theorem keyPubOf_empty {c : Flock.Circuit} {hU : FlockSoundness.Built c.unit} {n : ℕ} (g : Fin (keyN c hU n)) :
    keyPubOf c hU n {} g = none := by
  cases h : keyPubOf c hU n {} g with
  | none => rfl
  | some x =>
    obtain ⟨u, q, t⟩ := x
    obtain ⟨-, -, -, hk⟩ := keyPubOf_spec h
    have hneg : Aliased.srcKeyP c {} ((g.val - 2) % keyB c hU) = .src _ :=
      Aliased.srcKeyP_of_neg (by rw [Aliased.pubLeaf_empty (pi := {}) rfl]; omega)
    rw [hneg] at hk
    cases hk

theorem outsOfP_empty (c : Flock.Circuit) (hU : FlockSoundness.Built c.unit) (n : ℕ) :
    outsOfP c hU n {} = outsOf c hU n := by
  unfold outsOfP outsOf
  rw [Placed.keyProgP_empty (c := c) (pi := {}) rfl hU n]

end FlockSoundness.Discharge.Layout

namespace Flock.Bridge

theorem constsOfP_empty (c : Flock.Circuit) (hU : FlockSoundness.Built c.unit) (n : ℕ) (pub : Flock.Public)
    (g : Fin (FlockSoundness.Discharge.Layout.keyN c hU n)) : constsOfP c hU n {} pub g = constWiresOf c hU n g := by
  unfold constsOfP
  rw [FlockSoundness.Discharge.Layout.keyPubOf_empty]
  cases constWiresOf c hU n g <;> rfl

theorem constWiresOf_eq_ite (c : Flock.Circuit) (hU : FlockSoundness.Built c.unit) (n : ℕ)
    (g : Fin (FlockSoundness.Discharge.Layout.keyN c hU n)) :
    constWiresOf c hU n g = if g.val = 0 then some true else if g.val = 1 then some false else none := by
  have e1 : (FlockSoundness.Discharge.Layout.oneG (c := c) (hU := hU) (n := n)).val = 0 := rfl
  have e0 : (FlockSoundness.Discharge.Layout.zeroG (c := c) (hU := hU) (n := n)).val = 1 := rfl
  unfold constWiresOf
  simp only [Fin.ext_iff, e1, e0]

theorem constsOfP_constGate (D : Flock.Desc) (hU : FlockSoundness.Built D.unroll.unit) (n : ℕ)
    (pi : Flock.HmIn.Ports) (pub : Flock.Public) (u : Fin n) :
    constsOfP D.unroll hU n pi pub (constGate D hU n u) = some true := by
  unfold constsOfP
  rw [constWiresOf_constGate]

end Flock.Bridge

namespace Verity

theorem parsedConsts_some (r : Flock.Bridge.Parsed) (g : Fin (Flock.Bridge.Parsed.circuit (some r)).N) :
    parsedConsts (some r) g = if g.val = 0 then some true else if g.val = 1 then some false else none :=
  rfl

end Verity

namespace Verity.Sampled

open Claims.Sampling

theorem unbound_subset_of_parsed {o : Option Flock.Bridge.Parsed} {r : Flock.Bridge.Parsed} (ho : o = some r)
    {C : Verity.Boolean.Circuit} (hC : Flock.Bridge.keyCircuit r.D r.hU r.n = C)
    {P : FlockSoundness.Audit.Partition C r.n} (hP : HEq (Flock.Bridge.keyPartition r.D r.hU r.n) P)
    {κ₀ : Fin (Flock.Bridge.keyCircuit r.D r.hU r.n).N → Option Bool} {κ : Fin C.N → Option Bool} (hκ : HEq κ₀ κ)
    (hcon : ∀ g, Verity.parsedConsts (some r) g = κ₀ g)
    {H : Claims.Msg → Claims.Msg} {Cm : CommitmentScheme Bool} {roots : List Claims.Msg} {val : ℕ → Option Bool}
    (hval : ∀ p b, val p = some b → ∃ d : Claims.Msg, Cm.Opens H roots ⟨p, b, d⟩ = true) {F : Finset ℕ}
    (hgame : ∀ opens : (g : Fin (Verity.Circuit.ofAudit C).N) → BitVec ((Verity.Circuit.ofAudit C).w g) → Prop,
      (∀ g b, κ g = some b → opens g (BitVec.ofBool b)) →
      (∀ g b, κ g = none → val (Flock.Bridge.posOf C g) = some b → opens g (BitVec.ofBool b)) →
      (ResourceAccounting.unbound (Verity.Partition.ofAudit P) opens
        (Verity.Circuit.ofBits (Flock.Bridge.openedOf C κ val))).image Fin.val ⊆ F)
    (op : (g : Fin (Flock.Bridge.Parsed.circuit o).N) → BitVec ((Flock.Bridge.Parsed.circuit o).w g) → Prop)
    (hopen : ∀ (h : (Flock.Bridge.Parsed.circuit o).N = C.N) (g : Fin (Flock.Bridge.Parsed.circuit o).N) (b : Bool)
      (d : Claims.Msg), Verity.parsedConsts o g = none →
        Cm.Opens H roots ⟨Flock.Bridge.posOf C (Fin.cast h g), b, d⟩ = true →
          ∀ v : BitVec ((Flock.Bridge.Parsed.circuit o).w g), v.getLsbD 0 = b → op g v)
    (hconst : ∀ g b, Verity.parsedConsts o g = some b → ∀ v : BitVec ((Flock.Bridge.Parsed.circuit o).w g),
      (∀ i : Fin ((Flock.Bridge.Parsed.circuit o).w g), v.getLsbD i.val = b) → op g v)
    (tr : (Flock.Bridge.Parsed.circuit o).Trace) (htr : HEq tr (Flock.Bridge.traceOf r.D r.hU r.n κ₀ val)) :
    (ResourceAccounting.unbound (Flock.Bridge.Parsed.parts o) op tr).image Fin.val ⊆ F := by
  subst ho hC
  obtain rfl := eq_of_heq hP
  obtain rfl := eq_of_heq hκ
  obtain rfl := eq_of_heq htr
  refine hgame op (fun g b hk => hconst g b ((hcon g).trans hk) (BitVec.ofBool b) fun i => ?_)
    fun g b hk hv => ?_
  · obtain ⟨i, hi⟩ := i
    obtain rfl : i = 0 := by change i < 1 at hi; omega
    exact (by simp : (BitVec.ofBool b).getLsbD 0 = b)
  · obtain ⟨d, hd⟩ := hval _ b hv
    exact hopen rfl g b d ((hcon g).trans hk) hd (BitVec.ofBool b) (by simp : (BitVec.ofBool b).getLsbD 0 = b)

namespace Live

variable (A : Live)

theorem parsed_eq :
    Flock.Bridge.parsedOf (FlockSoundness.Discharge.ZkExec.zkTags A.I) A.I.circuitFile A.I.tables A.I.publicFile =
      some ⟨A.D, A.hU, A.n⟩ := by
  have h := Flock.Bridge.parsedOf_eq A.hD A.hpubO A.hU
  rw [A.hnO] at h
  exact h

theorem Cc_eq : Flock.Bridge.keyCircuit A.D A.hU A.n = A.Cc := by
  show (FlockSoundness.Discharge.Placed.keyProg A.D.unroll A.hU A.n).circuit
      (FlockSoundness.Discharge.Layout.outsOf A.D.unroll A.hU A.n) =
    (FlockSoundness.Discharge.Placed.keyProgP A.D.unroll {} A.hU A.n).circuit
      (FlockSoundness.Discharge.Layout.outsOfP A.D.unroll A.hU A.n {})
  rw [FlockSoundness.Discharge.Layout.outsOfP_empty,
    FlockSoundness.Discharge.Placed.keyProgP_empty (c := A.D.unroll) (pi := {}) rfl A.hU A.n]

theorem P_heq : HEq (Flock.Bridge.keyPartition A.D A.hU A.n) A.P := by
  show HEq ((FlockSoundness.Discharge.Placed.keyProg A.D.unroll A.hU A.n).partition
      (FlockSoundness.Discharge.Layout.outsOf A.D.unroll A.hU A.n))
    ((FlockSoundness.Discharge.Placed.keyProgP A.D.unroll {} A.hU A.n).partition
      (FlockSoundness.Discharge.Layout.outsOfP A.D.unroll A.hU A.n {}))
  rw [FlockSoundness.Discharge.Layout.outsOfP_empty,
    FlockSoundness.Discharge.Placed.keyProgP_empty (c := A.D.unroll) (pi := {}) rfl A.hU A.n]

theorem consts_eq (g : Fin A.Cc.N) : Verity.parsedConsts (some ⟨A.D, A.hU, A.n⟩) g = A.κ g := by
  show _ = Flock.Bridge.constsOfP A.D.unroll A.hU A.n {} A.pub g
  exact (Verity.parsedConsts_some ⟨A.D, A.hU, A.n⟩ g).trans
    ((Flock.Bridge.constWiresOf_eq_ite A.D.unroll A.hU A.n g).symm.trans
      (Flock.Bridge.constsOfP_empty A.D.unroll A.hU A.n A.pub g).symm)

theorem κ_isInput (g : Fin A.Cc.N) (b : Bool) (h : A.κ g = some b) : (A.Cc.op g).isInput = true :=
  Flock.Bridge.constsOfP_isInput (FlockSoundness.Discharge.Layout.outsOfP A.D.unroll A.hU A.n {}) h

theorem unbound_subset_faulty_part
    (op : (g : Fin A.fileRead.circuit.N) → BitVec (A.fileRead.circuit.w g) → Prop)
    (hop : A.OpensUnder (.one A.fileRead) op)
    (hc : ∀ g b, A.fileRead.consts g = some b → ∀ v : BitVec (A.fileRead.circuit.w g),
      (∀ i : Fin (A.fileRead.circuit.w g), v.getLsbD i.val = b) → op g v)
    (w : Wit) :
    (ResourceAccounting.unbound A.fileRead.parts op
        (cast (congrArg Verity.Circuit.Trace A.circuit_eq.symm)
          (Flock.Bridge.traceOf A.D A.hU A.n A.κ (w.val sha512Msg A.Cd
            (Flock.Bridge.stmtOf A.P A.κ (_root_.Properties.Spot.oneStageOf A.design) A.desc ⟨"", 0, []⟩
              A.roots))))).image Fin.val ⊆
      faulty sha512Msg A.Cd
        (Flock.Bridge.stmtOf A.P A.κ (_root_.Properties.Spot.oneStageOf A.design) A.desc ⟨"", 0, []⟩ A.roots)
        (Flock.Bridge.partOf A.P A.κ (Flock.Bridge.nsOf A.desc A.qr)) w := by
  refine unbound_subset_of_parsed A.parsed_eq A.Cc_eq A.P_heq (κ₀ := A.κ) (κ := A.κ) HEq.rfl A.consts_eq
    (H := sha512Msg) (Cm := A.Cd) (roots := A.roots)
    (val := w.val sha512Msg A.Cd
      (Flock.Bridge.stmtOf A.P A.κ (_root_.Properties.Spot.oneStageOf A.design) A.desc ⟨"", 0, []⟩ A.roots))
    (fun _ _ h => Flock.Bridge.opens_of_val h) ?_ op hop hc _ (cast_heq _ _)
  intro opens hK hO
  exact Flock.Bridge.unbound_subset_faulty_partOf A.P A.κ (Flock.Bridge.nsOf_sum A.hpop) sha512Msg A.Cd
    (Flock.Bridge.stmtOf A.P A.κ (_root_.Properties.Spot.oneStageOf A.design) A.desc ⟨"", 0, []⟩ A.roots) rfl w
    opens A.κ_isInput hK hO

end Live

end Verity.Sampled
