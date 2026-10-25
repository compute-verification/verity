import Proofs.Flock.SoundAtBridge.Served
import Proofs.Flock.C3.Compose.RowsReg

/-!
# `ServedRowsLinked`'s proof

`RowsSession.ownReadsHJ_of_served`: at an accepted outcome, a drawn unit whose registrant's reads are the rows of its
served roots (`ServedRegistered`) has every one of them owned (`OwnReadsHJ`), as C3's `FlockC3.Compose.ownReads_cover`
has it for the honest registrant: the record's draw is the law's draw file (`RecordCustodyZKJ`), the unit's table has a
drawn statement (`drawn_of_acceptsZKR`) whose draw proves the unit (`drawOkZKHJ_of_file`), and acceptance gives each
`own` port's position at the unit's label (`reads_of_acceptsZKR`). With it, `RowsAt` is the rows link at those reads
(`RowsSession.xv_eq_of_rowsAt`).
-/

namespace Flock.SoundAtBridge

open FlockSoundness FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
  FlockSoundness.Discharge.ZkReg
open FlockSoundness.Discharge.Tables (LiveOutJ)

namespace RowsSession

variable (s : RowsSession)

/-- **A drawn unit's served reads are owned** at an accepted outcome, when its registrant's reads are its served roots'
rows (`ServedRegistered`); acceptance names each port of the circuit once (`names_of_acceptsZKR`). -/
theorem ownReadsHJ_of_served {S : Finset (Fin s.n)} {R : s.Reg}
    {o : LiveOutJ (planZHJ (Reg := s.Reg) s.dj s.y₀ S R).length (mPtsOf s.c.g (s.kd / nTab s.I))}
    (hacc : LiveAcceptsZKR s.I s.own (R, S, s.wr S R o)) {u : Fin s.n} (hu : u ∈ S) (hS : s.ServedRegistered u) :
    OwnReadsHJ s.I s.own s.c s.pub (s.wr S R o) (s.rdU u).isRead (s.rdU u).port (s.rdU u).readAt := by
  intro p hp
  obtain ⟨q, rfl, hn⟩ := (hS.1 p).1 hp
  obtain ⟨hport, hat⟩ := hS.2 q hn
  obtain ⟨hmem', hname'⟩ := FlockC3.Compose.portOf_spec hn
  have hmem : s.ownPort q ∈ s.own := hmem'
  have hname : (s.ownPort q).name = s.c.ports[q].name := hname'
  have hd : recordDraw (s.wr S R o).record = some (s.dj S) :=
    (s.hCust.1 S R o (acceptsZK_of_R (t := s.wr S R o) hacc)).1
  have hN := names_of_acceptsZKR s.I s.ht s.hc s.hTags (tr := s.wr S R o) hacc hd
  obtain ⟨D, dp, idx, hD, hdp, hidx⟩ :=
    drawn_of_acceptsZKR s.I s.ht s.hc s.hpub s.hTags (tr := s.wr S R o) hacc hd u.val
  have hmD : u.val ∈ D.proved :=
    ((FlockSoundness.Discharge.ZkReg.drawOkZKHJ_of_file s.I s.ht s.hc s.kd s.n s.hdiv s.dj s.hCust.2 S)
      (partOf s.I (s.dj S) u.val)).1 D hD u (mem_partS hu)
  obtain ⟨pos, hpos⟩ :=
    reads_of_acceptsZKR s.I s.ht s.hc s.hpub s.hTags (tr := s.wr S R o) hacc hmem hd hD hdp hidx hmD
  refine ⟨u, q, s.dj S, D, dp, idx, rfl, ?_, ?_, hd, hD, hdp, hidx, hmD, ?_⟩
  · rw [hport]; exact hmem
  · rw [hport, hname]; exact FlockC3.Compose.find_of_names hN q
  · rw [hport, hpos, hat _ D dp idx hD hdp hidx hmD pos hpos]

end RowsSession

/-- **C-Flock's rows link at served roots** (`Flock.Guarantees.ServedRowsLinked`). -/
theorem servedRowsLinked : Flock.Guarantees.ServedRowsLinked := by
  intro s S R o u hacc hu hS hrows g hg hr
  exact s.xv_eq_of_rowsAt (s.ownReadsHJ_of_served hacc hu hS) hrows hg hr

end Flock.SoundAtBridge
