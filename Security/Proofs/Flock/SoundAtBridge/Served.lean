import Proofs.Flock.SoundAtBridge.Statement

/-!
# The rows link at served roots (trusted)

A window check's relation reads its operand leaves and its receipt `rc`, which the window commits under served roots:
the public file's `roots` entries, one per input port, each a hiding commitment over the session's own hm96 leaves
(`HmRow.checkPublic`). `EndToEndRegisteredRows` links the value layer `XplurZC` to a committed row only at a registered
read (`RowsRightZC`): a row of a port of `own`, the roots F registered before the session's statement and coin, opened
by the registrant (`rd`). At a hiding leaf nothing else is bound, so at a served root that is not a port of `own` the
value layer is the extractor's and not the committed tree's, and `RelOfRows` fails for a relation that reads it.

A served root is covered when the service registers its tree as a port of `own` before the session's statement and coin,
with the prover's commit-time opening as the registrant's (`RowsSession.ServedRegistered`: unit `u`'s opening reads
exactly `u`'s rows at the ports `own` names, through `own`'s first port of that name, at the position it gives `u`).
`Flock.Guarantees.ServedRowsLinked` says that then, at every accepted outcome, a drawn unit whose rows are right at its
owned reads (`RowsAt`) has the committed tree's row at each served read.

`RowsSession.xv_eq_of_rowsAt` gives it from `OwnReadsHJ` at the outcome, and `OwnReadsHJ` at `ServedRegistered` is
C3's `FlockC3.Compose.ownReads_cover` in this setting (`Proofs.Flock.SoundAtBridge.ServedProof`, whose
`servedRowsLinked` is the proof). `ServedRegistered` is the one premise: the service's registration of its served trees.
-/

namespace Flock.SoundAtBridge

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden FlockSoundness.Discharge.ZkReg
open FlockSoundness.Discharge.Tables (LiveOutJ)
open Classical

noncomputable section

namespace RowsSession

variable (s : RowsSession)

/-- `own`'s first port of the name of `c`'s port `q`, the one the registered check reads. -/
def ownPort (q : Fin s.c.ports.size) : _root_.Flock.Registered.Port :=
  (s.own.toList.find? (·.name == s.c.ports[q].name)).getD default

/-- **Unit `u`'s served roots are registered** (owner: the proof system, which registers each served tree a relation
reads as a port of `own` before the session's statement and coin): `rdU u` reads exactly `u`'s rows at the ports `own`
names, each through `own`'s first port of that name (`ownPort`), at the position that port gives `u` in every table
draw that proves it. -/
def ServedRegistered (u : Fin s.n) : Prop :=
  (∀ p, (s.rdU u).isRead p ↔
      ∃ q : Fin s.c.ports.size, p = .inl (u, q) ∧ ∃ pt ∈ s.own, pt.name = s.c.ports[q].name) ∧
    ∀ q : Fin s.c.ports.size, (∃ pt ∈ s.own, pt.name = s.c.ports[q].name) →
      (s.rdU u).port (.inl (u, q)) = s.ownPort q ∧
        ∀ d D dp idx, _root_.Flock.Draw.UnitDraw.ofJson (drawJ s.I d (partOf s.I d u.val)) = .ok D →
          HmOut.drawn s.c s.pub (drawJ s.I d (partOf s.I d u.val)) = .ok dp →
          _root_.Flock.Registered.unitIndices dp = .ok idx → u.val ∈ D.proved → ∀ pos,
            (s.ownPort q).reads[idx.getD (D.proved.idxOf u.val) 0]? = some (some pos) →
              (s.rdU u).readAt (.inl (u, q)) = pos

/-- **At an owned read, the value layer is the registrant's row**: where every read of `rdU u` is the verifier's own
at the outcome (`OwnReadsHJ`), a unit whose rows are right there (`RowsAt`) has, on each of its wires at a registered
read, the registrant's row's bit. -/
theorem xv_eq_of_rowsAt {o : s.Out} {u : Fin s.n}
    (hown : OwnReadsHJ s.I s.own s.c s.pub o.2.2 (s.rdU u).isRead (s.rdU u).port (s.rdU u).readAt)
    (h : s.RowsAt o u) {g} (hg : g ∈ s.P.io u) (hr : (s.rdU u).isRead (s.vb.pos g)) :
    s.Xv g = s.vb.bit (s.vb.val (s.wo u (s.vb.pos g))) g :=
  (h g hg).1 ⟨hown, hr⟩

end RowsSession

end

end Flock.SoundAtBridge

namespace Flock.Guarantees

open Flock.SoundAtBridge
open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden FlockSoundness.Discharge.ZkReg
open FlockSoundness.Discharge.Tables (LiveOutJ)
open Classical

/-- **C-Flock's rows link at served roots.** At a C-Flock session (`RowsSession`), at every outcome of its live audit
(draw `S`, registration `R`, the record `wr` writes) that flock-verify `--zk --registered` accepts, a drawn unit `u`
whose served roots are registered (`ServedRegistered`: its opening `rdU u` reads exactly its rows at the ports `own`
names, as `own` gives them) and whose rows are right at its owned reads (`RowsAt`) has, on each of its wires at a served
read, the committed tree's row: the value layer's bit (`Xv`) is the registrant's row's. Acceptance names each port of
the circuit once (`Registered.namesOnce`), so nothing is asked of the circuit's names. With `RelOfRows`, a relation that
reads its operand leaves and receipt at served reads then reads the committed trees'. -/
def ServedRowsLinked : Prop :=
  ∀ (s : RowsSession) (S : Finset (Fin s.n)) (R : s.Reg)
      (o : LiveOutJ (planZHJ (Reg := s.Reg) s.dj s.y₀ S R).length (mPtsOf s.c.g (s.kd / nTab s.I))) (u : Fin s.n),
      LiveAcceptsZKR s.I s.own (R, S, s.wr S R o) → u ∈ S → s.ServedRegistered u →
      s.RowsAt (R, S, s.wr S R o) u →
      ∀ g ∈ s.P.io u, (s.rdU u).isRead (s.vb.pos g) → s.Xv g = s.vb.bit (s.vb.val (s.wo u (s.vb.pos g))) g

end Flock.Guarantees
