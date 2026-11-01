import Proofs.Accounting.TS.Session.Group.Guarantees
import Proofs.Accounting.TS.Session.Lanes.KnownRead
import Proofs.Accounting.TS.Session.Lanes.Partials
import Proofs.Accounting.TS.Session.Lanes.Committed
import Proofs.Flock.C3.RegLevels.TopLeaf
import Proofs.Accounting.TS.CheckedSound
import Proofs.Accounting.TS.HeldSound
import Proofs.Pouw.RelativeSecurity
import Proofs.Flock.VBridge.Family.Registrants
import PouwWindow.OwnPorts
import PouwWindow.FullDraw

/-!
# A leaf's group of linked sessions realizes one unit, any shape (the proof)

`Accounting.TS.Group.group_soundAt`, the argument every leaf proved as a group of one-unit C-Flock statements shares,
whatever the group (any finite type `ι` of statements: `Ix L n`'s `L` lanes of `n` chunks and a combine, or option A's
33): a proof layer accepted only where each of the group's sessions passes the leaf verifier's per-statement checks
(`Group.SessionChecks`), whose relation the group's circuits realize at the rows the sessions open
(`Accounting.Assumptions.TS.GroupRealizes`) and admits a clash of two of the sessions' registrants' records
(`LeafColl`) and a collision of a session's known-content tie (`LeafKnownColl`), is sound at `|ι| δ`. An unsound proof
is, at one of its sessions, an unsound proof of the checked layer where its ports are named once and include `ok`
(`Accounting.TS.Checked.checkedOn_soundAt`'s `SoundAt δ`, `CheckedSessionSound`'s argument with those asked only where
they hold, which the known-read check's names give at known content naming `ok`: `names_of_check`, with full checking
asked there too) or an accepted outcome that isn't the session's write of a live run (`RowsSession.Written`), which
has probability `0` in every session's game (`RowsSession.prob_unwritten`) and so on the window (`WindowLaw`). At a
written outcome the leaf verifier's draw check (`PouwWindow.Lanes.checkFull`) gives full checking: the record's draw is
`subset` of every unit of a non-empty population, and the session's custody makes it the verifier's draw `subset kd`
over `n` units, so the outcome draws every unit, `kd = n` and `0 < n` (`RowsSession.full_of_check`), and every unit
reads its one-row ports at their rows (`RowsSession.readsOwnRow_of`). Every unit's served roots are registered
(`RowsSession.ServedRegistered`) wherever the one-read check (`PouwWindow.Lanes.checkOneRead`) holds, which the
sessions' reads built from `own` give (`RowsSession.servedRegistered_canon`). The sessions' registrants' rows agree
unless two records clash (`rowsAgree_or_leafColl`, from `FlockSoundness.Registered.enc_eq_of_not_clash`), and the
known rows hold unless a known-content tie collides (`knownHeld_or_coll`, from flock-verify's known-content check).
Under the complete check the registered reads the checked layer opens are the port rows
(`RowsSession.rowsReg_eq_portRead`), so each session's committed rows are the port rows of a wire assignment its
circuit's evaluation agrees with, which `GroupRealizes` reads. The union bound over the group gives `|ι| δ`.

`Accounting.SecurityProofs.TS.LeafGroupSound` is it at `groupLayer` and `Ix T.lanes n` (`T.lanes · n + 1` statements,
`card_ix`); `Accounting.SecurityProofs.TS.LeafLanesSound` is it at option A's `leafLayer`
(`Proofs.Accounting.TS.LanesSound`). `Accounting.TS.Group.groupLayer_not_soundAt` is the nontrivial lemma: at a proof
accepted on every path with an opening that fails its relation and no collision in hand, the guarantee's conclusion
fails at every `δ` with `(T.lanes · n + 1) δ < 1`, so its premises are what make it hold.

This module imports `Proofs.Flock.SoundAtBridge` (through `Proofs.Accounting.TS.CheckedSound`), whose imports make `X`
a token, so `Proofs/Accounting.lean` imports it.
-/

namespace Flock.SoundAtBridge.RowsSession

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden FlockSoundness.Discharge.ZkReg
open FlockSoundness.Discharge.Tables (LiveOutJ)

variable (s : RowsSession)

/-- **An outcome is written**: its transcript is the session's write (`wr`) of a live run's output at its registration
and draw, as every outcome of the session's game is. -/
def Written (o : s.Out) : Prop :=
  ∃ (R : s.Reg) (S : Finset (Fin s.n))
    (o' : LiveOutJ (planZHJ (Reg := s.Reg) s.dj s.y₀ S R).length (mPtsOf s.c.g (s.kd / nTab s.I))),
    o = (R, S, s.wr S R o')

/-- **No outcome of the session's game is unwritten**: the live session's outcomes are `wr` of its runs'. -/
theorem prob_unwritten : prob (fun o : s.Out => ¬ s.Written o) s.G s.τ = 0 := by
  refine prob_audit_eq_zero s.L s.Reg (liveOfZJ s.dg (planZHJ s.dj s.y₀) s.N s.wr) (fun o => ¬ s.Written o) ?_ s.τ
  intro R ω s'
  refine (prob_map _ (s.wr (s.L.draw ω) R) _ s').trans ?_
  exact nonpos_iff_eq_zero.1
    ((prob_mono (fun a h => h ⟨R, s.L.draw ω, a, rfl⟩) _ _).trans (prob_false _ _).le)

/-- **The leaf verifier's draw policy gives full checking**: at an accepted written outcome whose record
`PouwWindow.Lanes.checkFull` accepts, the record's draw is the verifier's draw of the outcome's units
(`RecordCustodyZKJ`: the record holds the draw file `dj S`, which decodes to `drawOf (.subset kd) n S`), and the check
makes it `subset n` of `n > 0` units, every one: the outcome draws every unit, `kd = n` and `0 < n`. -/
theorem full_of_check {o : s.Out} (hacc : LiveAcceptsZKR s.I s.own o) (hw : s.Written o)
    (hf : PouwWindow.Lanes.checkFull o.2.2.record = .ok ()) :
    o.2.1 = Finset.univ ∧ s.kd = s.n ∧ 0 < s.n := by
  obtain ⟨R, S, o', rfl⟩ := hw
  obtain ⟨d, D, k, hr, hD, hl, hp, hk, hu⟩ := PouwWindow.Lanes.checkFull_ok hf
  have hd : recordDraw (s.wr S R o').record = some (s.dj S) :=
    (s.hCust.1 S R o' (acceptsZK_of_R (t := s.wr S R o') hacc)).1
  have hr' : recordDraw (s.wr S R o').record = some d := hr
  rw [hd] at hr'
  cases Option.some.inj hr'
  have e := s.hCust.2 S D hD
  subst e
  have hkd : s.kd = k := by
    have h : Flock.Draw.Law.subset s.kd = .subset k := hl
    injection h
  have hn : k = s.n := hp
  refine ⟨?_, hkd.trans hn, by rw [← hn]; exact hk⟩
  apply Finset.eq_univ_of_forall
  intro u
  have hm : u.val ∈ (FlockSoundness.Discharge.ZkLink.drawOf (.subset s.kd) s.n S).units := by
    rw [hu, hn]
    exact Array.mem_range.2 u.isLt
  obtain ⟨u', hu', he⟩ := (FlockSoundness.Discharge.ZkLink.drawOf_mem _ _ S u.val).1 hm
  exact (Fin.ext he) ▸ hu'

/-- **At a written accepted outcome that draws every unit, every unit reads a one-row port at its row**
(`Accounting.TS.Lanes.ReadsOwnRow`): the unit is drawn and proved (`drawn_of_acceptsZKR`, `drawOkZKHJ_of_file`), so
acceptance gives `own`'s port a position at the unit's index (`reads_of_acceptsZKR`), which is the unit's read
(`ServedRegistered`); a port of one read has it at index 0. -/
theorem readsOwnRow_of {o : s.Out} (hacc : LiveAcceptsZKR s.I s.own o) (hw : s.Written o)
    (hall : o.2.1 = Finset.univ) (hS : ∀ u, s.ServedRegistered u) : Accounting.TS.Lanes.ReadsOwnRow s := by
  intro u q hn r hr
  obtain ⟨R, S, o', rfl⟩ := hw
  dsimp only at hall
  have hu : u ∈ S := by rw [hall]; exact Finset.mem_univ u
  obtain ⟨-, hat⟩ := (hS u).2 q hn
  obtain ⟨hmem', -⟩ := FlockC3.Compose.portOf_spec hn
  have hmem : s.ownPort q ∈ s.own := hmem'
  have hd : recordDraw (s.wr S R o').record = some (s.dj S) :=
    (s.hCust.1 S R o' (acceptsZK_of_R (t := s.wr S R o') hacc)).1
  obtain ⟨D, dp, idx, hD, hdp, hidx⟩ :=
    drawn_of_acceptsZKR s.I s.ht s.hc s.hpub s.hTags (tr := s.wr S R o') hacc hd u.val
  have hmD : u.val ∈ D.proved :=
    ((FlockSoundness.Discharge.ZkReg.drawOkZKHJ_of_file s.I s.ht s.hc s.kd s.n s.hdiv s.dj s.hCust.2 S)
      (partOf s.I (s.dj S) u.val)).1 D hD u (mem_partS hu)
  obtain ⟨pos, hpos⟩ :=
    reads_of_acceptsZKR s.I s.ht s.hc s.hpub s.hTags (tr := s.wr S R o') hacc hmem hd hD hdp hidx hmD
  rw [hat _ D dp idx hD hdp hidx hmD pos hpos]
  rw [hr] at hpos
  have key : ∀ j : ℕ, (#[some r] : Array (Option ℕ))[j]? = some (some pos) → pos = r := by
    intro j hj
    cases j with
    | zero => simp at hj; omega
    | succ j => simp at hj
  exact key _ hpos

/-- **Under the complete check a unit's registered reads are its port rows**: where unit `u`'s registrant reads its
served roots' rows (`ServedRegistered`) and every port of the circuit is a port of `own`
(`Flock.Registered.checkComplete`), a wire's commit string is a registered read of `rdU u` exactly when it is one of
`u`'s port rows. -/
theorem isRead_iff_portRow {u : Fin s.n} (hS : s.ServedRegistered u)
    (hcomp : _root_.Flock.Registered.checkComplete s.c s.own = .ok ()) (g : Fin s.Circ.N) :
    (s.rdU u).isRead (s.vb.pos g) ↔ Accounting.TS.Lanes.PortRow s u g :=
  (hS.1 _).trans ⟨fun ⟨q, hq, _⟩ => ⟨q, hq⟩,
    fun ⟨q, hq⟩ => ⟨q, hq, _root_.Flock.Registered.checkComplete_ok hcomp _ (Array.getElem_mem q.isLt)⟩⟩

/-- **Under the complete check the registered reads are the port rows**: `rowsReg` reads every port row of every unit
(`Accounting.TS.Lanes.portRead`). -/
theorem rowsReg_eq_portRead (hS : ∀ u, s.ServedRegistered u)
    (hcomp : _root_.Flock.Registered.checkComplete s.c s.own = .ok ()) (vals : Fin s.Circ.N → Bool) :
    Accounting.TS.Checked.rowsReg s vals = Accounting.TS.Lanes.portRead s vals := by
  funext a b
  unfold Accounting.TS.Checked.rowsReg Accounting.TS.Checked.regRead Accounting.TS.Lanes.portRead
  by_cases h : a < s.n ∧ b < s.Circ.N
  · rw [dite_eq_left h, dite_eq_left h]
    have hiff := s.isRead_iff_portRow (hS ⟨a, h.1⟩) hcomp ⟨b, h.2⟩
    by_cases hr : (⟨b, h.2⟩ : Fin s.Circ.N) ∈ s.P.io ⟨a, h.1⟩ ∧ Accounting.TS.Lanes.PortRow s ⟨a, h.1⟩ ⟨b, h.2⟩
    · rw [ite_eq_left hr, ite_eq_left ⟨hr.1, hiff.2 hr.2⟩]
    · rw [ite_eq_right hr, ite_eq_right fun hp => hr ⟨hp.1, hiff.1 hp.2⟩]
  · rw [dite_eq_right h, dite_eq_right h]

/-- **Under the complete check every port row of a drawn unit is registered and opens at its registered root**: at
an outcome flock-verify `--zk --registered` accepts, a drawn unit `u` whose registrant reads its served roots' rows
(`ServedRegistered`), in a circuit whose every port is a port of `own` (`Flock.Registered.checkComplete`; acceptance
has the circuit name each port once), reads `u`'s row of every port `q` (each input port and output row, so whatever an
output's cone reads, the lane index included) as a registered read through `own`'s port of `q`'s name (`ownPort`), and
every registered read of `u` opens the root `own` holds there (`ReadsX.Checked`, through `regPathsHJ`). -/
theorem portRow_opens (hcomp : _root_.Flock.Registered.checkComplete s.c s.own = .ok ()) {S : Finset (Fin s.n)} {R : s.Reg}
    {o : LiveOutJ (planZHJ (Reg := s.Reg) s.dj s.y₀ S R).length (mPtsOf s.c.g (s.kd / nTab s.I))}
    (hacc : LiveAcceptsZKR s.I s.own (R, S, s.wr S R o)) {u : Fin s.n} (hu : u ∈ S) (hS : s.ServedRegistered u)
    (q : Fin s.c.ports.size) :
    (s.rdU u).isRead (.inl (u, q)) ∧ s.ownPort q ∈ s.own ∧ (s.ownPort q).name = s.c.ports[q].name ∧
      (s.rdU u).port (.inl (u, q)) = s.ownPort q ∧
      (s.rdU u).Checked R (regPathsHJ s.I s.c s.pub (s.wr S R o) (s.rdU u).port) := by
  have hn : ∃ pt ∈ s.own, pt.name = s.c.ports[q].name :=
    _root_.Flock.Registered.checkComplete_ok hcomp _ (Array.getElem_mem q.isLt)
  obtain ⟨hmem, hname⟩ := FlockC3.Compose.portOf_spec hn
  exact ⟨(hS.1 _).2 ⟨q, rfl, hn⟩, hmem, hname, (hS.2 q hn).1,
    checked_of_acceptsHJ s.I s.ht s.own s.dj s.y₀ s.hc s.hU s.hpub s.hscope s.hTags _ hacc (s.rdU u)
      (s.ownReadsHJ_of_served hacc hu hS)⟩

end Flock.SoundAtBridge.RowsSession

namespace Accounting.TS.Lanes

open Accounting.TS.Checked Flock.SoundAtBridge

/-- **A group's sessions' registrants' rows agree, or two of their records clash**: two registered reads at one tree
(root, leaf count, domain and schema) and position carry one row's bytes, or their openings or hm96's leaves collide
(`FlockSoundness.Registered.enc_eq_of_not_clash`), which is `LeafColl`. -/
theorem rowsAgree_or_leafColl {ι : Type} (sess : ι → RowsSession) : RowsAgree sess ∨ LeafColl sess := by
  by_cases h : LeafColl sess
  · exact Or.inr h
  · exact Or.inl fun k k' u u' p p' hp hp' ht hat =>
      FlockSoundness.Registered.enc_eq_of_not_clash (fun hc => h ⟨k, k', u, u', hc⟩) hp hp' ht.1 ht.2.1 ht.2.2.1
        ht.2.2.2 hat

/-- **The known-content tie at any known port** (`Accounting.TS.Held.tie_ok` at a port `nm` and its known rows `ks`,
any row): where flock-verify's known-content check accepted `own` at `known`, which names `nm` with rows `ks`, and every
unit's served roots are registered, a registrant's row at a registered read of a port named `nm` is at a position of
the known rows (`checkKnown`'s one known row a leaf, and `opensLeaf`'s position below the leaves) and is the bytes of the
known row there, unless the tie collides (`OkColl` at `nm`). -/
theorem tie_known (s : RowsSession) {known : Known} {nm : String} {ks : Array (Array ℕ × ByteArray)}
    (hk : Flock.Registered.checkKnownAll s.own known = .ok ()) (hm : (nm, ks) ∈ known)
    (hS : ∀ u, s.ServedRegistered u) (hcol : ¬ Accounting.TS.Held.OkColl s nm ks) {u : Fin s.n} {p : s.vb.Pos}
    (hr : (s.rdU u).isRead p) (hn : ((s.rdU u).port p).name = nm) :
    (s.rdU u).readAt p < ks.size ∧
      (rsOf s).enc p ((s.rdU u).wv p) = (Flock.Registered.rowBytesV2 ks[(s.rdU u).readAt p]!.1).data.toList := by
  obtain ⟨reg, hf, hck⟩ := FlockC3.RegLevels.checkKnownAll_ok hk hm
  obtain ⟨q, hpq, hex⟩ := ((hS u).1 _).1 hr
  have hport : (s.rdU u).port p = s.ownPort q := by
    rw [hpq]; exact ((hS u).2 q hex).1
  have hqn : s.c.ports[q].name = nm := by
    obtain ⟨pt, hpt, hpn⟩ := hex
    have e := Accounting.TS.Held.find_name_eq (l := s.own.toList) (Array.mem_toList_iff.2 hpt) hpn
    rw [← hn, hport]
    exact e.symm
  have hreg : s.ownPort q = reg := by
    show (s.own.toList.find? (·.name == s.c.ports[q].name)).getD default = reg
    rw [hqn, hf]
    rfl
  have ho := (s.rdU u).committed _ hr
  rw [hport, hreg] at ho
  obtain ⟨hi, hb | ⟨a, b, hc, hab, hh⟩⟩ := FlockC3.RegLevels.knownColl_spec
    (r := (rsOf s).enc p ((s.rdU u).wv p)) (y := (s.rdU u).ws p) hck ho
  · exact ⟨hi, hb⟩
  · exact (hcol ⟨u, p, hr, hn, a, b, by rw [hport, hreg]; exact hc, hab, hh⟩).elim

/-- **A group's sessions' known rows hold, or a known-content tie collides**: at sessions whose known-content checks
passed and whose served roots are registered (`tie_known` at each). -/
theorem knownHeld_or_coll {ι : Type} (sess : ι → RowsSession) (known : ι → Known)
    (hk : ∀ k, Flock.Registered.checkKnownAll (sess k).own (known k) = .ok ())
    (hS : ∀ k u, (sess k).ServedRegistered u) :
    (∀ k, KnownHeld (sess k) (known k)) ∨ LeafKnownColl sess known := by
  by_cases h : LeafKnownColl sess known
  · exact Or.inr h
  · exact Or.inl fun k nm ks hm u p hr hn =>
      tie_known (sess k) (hk k) hm (hS k) (fun hc => h ⟨k, nm, ks, hm, hc⟩) hr hn

end Accounting.TS.Lanes

namespace Accounting.TS.Group

open Accounting.TS Accounting.TS.Checked Accounting.TS.Lanes Flock.SoundAtBridge FlockSoundness.Discharge.ZkReg
open Pouw.RelativeSecurity (Paths)
open scoped ENNReal

/-- **A leaf's group of linked C-Flock sessions realizes one unit, any group**: a proof layer `pf` accepted only where
the leaf verifier's group check `gate` passes and each of the group's sessions passes `SessionChecks` at its known
content `known` (naming `ok`), the ports `full` it reads in full and its file's ports `want`, whose relation the group's
circuits realize at the rows the sessions open (`GroupRealizes`) and admits a clash of two of the sessions' registrants'
records (`LeafColl`) and a collision of a session's known-content tie (`LeafKnownColl`), at sessions whose units' served
roots are registered wherever their files have one read at each port, is sound at `|ι| δ` under
`CheckedSessionSound`'s collision bounds and law at each session. -/
theorem group_soundAt {Ω : Type} [Fintype Ω] {S nπ : ℕ} {ι V : Type} [Fintype ι] (pf : ProofLayer Ω S nπ V)
    (sess : Paths Ω S → Fin S → Fin nπ → ι → RowsSession)
    (out : ∀ x w i k, (sess x w i k).Out) (known : Paths Ω S → Fin S → Fin nπ → ι → Known)
    (full : Paths Ω S → Fin S → Fin nπ → ι → Array String)
    (want : Paths Ω S → Fin S → Fin nπ → ι → List _root_.Flock.Registered.Port)
    (gate : Paths Ω S → Fin S → Fin nπ → Prop) (ok : String) (δ : ℝ)
    (hCR : Accounting.Premises.TS.SessionsCR sess) (hLaw : Accounting.Premises.TS.WindowLaws sess out)
    (hS : ∀ x w i k, PouwWindow.Lanes.checkOneRead (sess x w i k).own = .ok () →
      ∀ u, (sess x w i k).ServedRegistered u)
    (hO : ∀ x w i k, ∃ ks, (ok, ks) ∈ known x w i k)
    (hδ : ∀ x w i k, ((sess x w i k).n : ℝ≥0∞) * (sess x w i k).bound ≤ ENNReal.ofReal δ) (hδ0 : 0 ≤ δ)
    (hAcc : ∀ x w i, pf.accepted x w i →
      gate x w i ∧ ∀ k, SessionChecks (sess x w i k) (out x w i k) (known x w i k) (full x w i k) (want x w i k))
    (hReal : ∀ x w i, Accounting.Assumptions.TS.GroupRealizes (sess x w i) (known x w i) (full x w i) (want x w i)
      (gate x w i) (pf.opened x w i) (pf.rel x w i))
    (hColl : ∀ x w i, LeafColl (sess x w i) → pf.rel x w i (pf.opened x w i))
    (hKColl : ∀ x w i, LeafKnownColl (sess x w i) (known x w i) → pf.rel x w i (pf.opened x w i)) :
    pf.SoundAt (Fintype.card ι * δ) := by
  have hSound : ∀ k, (checkedOnLayer (fun x w i => sess x w i k) (fun x w i => out x w i k) ok
      (fun s => PortsOk ok s ∧ s.kd = s.n ∧ ∀ u, s.ServedRegistered u)).SoundAt δ := fun k =>
    checkedOn_soundAt (fun x w i => sess x w i k) (fun x w i => out x w i k) ok
      (fun s => PortsOk ok s ∧ s.kd = s.n ∧ ∀ u, s.ServedRegistered u) δ
      (fun x w i => hCR x w i k) (hLaw k) (fun _ h => h) (fun x w i => hδ x w i k) hδ0
  have hUnwritten : ∀ k (w : Fin S) (i : Fin nπ),
      Pouw.RelativeSecurity.prob (fun x => ¬ (sess x w i k).Written (out x w i k)) ≤ 0 := fun k w i =>
    (hLaw k w i fun s o => ¬ s.Written o).trans (avg_le _ le_rfl fun x => le_of_eq
      ((congrArg ENNReal.toReal (RowsSession.prob_unwritten (sess x w i k))).trans ENNReal.toReal_zero))
  intro w i
  have hIn : ∀ x, pf.Unsound x w i →
      ∃ k ∈ (Finset.univ : Finset ι),
        (checkedOnLayer (fun x w i => sess x w i k) (fun x w i => out x w i k) ok
          (fun s => PortsOk ok s ∧ s.kd = s.n ∧ ∀ u, s.ServedRegistered u)).Unsound x w i ∨
          ¬ (sess x w i k).Written (out x w i k) := by
    intro x ⟨hacc₀, hrel⟩
    obtain ⟨hg, hacc⟩ := hAcc x w i hacc₀
    by_contra hno
    push Not at hno
    apply hrel
    have hfull : ∀ k, (out x w i k).2.1 = Finset.univ ∧ (sess x w i k).kd = (sess x w i k).n ∧
        0 < (sess x w i k).n := fun k =>
      RowsSession.full_of_check (sess x w i k) (hacc k).1 (hno k (Finset.mem_univ _)).2 (hacc k).2.2.2.2.2.1
    have hSr : ∀ k u, (sess x w i k).ServedRegistered u := fun k => hS x w i k (hacc k).2.2.2.2.2.2
    rcases rowsAgree_or_leafColl (sess x w i) with hA | hc
    · rcases knownHeld_or_coll (sess x w i) (known x w i) (fun k => (hacc k).2.1) hSr with hK | hK
      swap
      · exact hKColl x w i hK
      have hRo : ∀ k, ReadsOwnRow (sess x w i k) := fun k =>
        RowsSession.readsOwnRow_of (sess x w i k) (hacc k).1 (hno k (Finset.mem_univ _)).2 (hfull k).1 (hSr k)
      refine hReal x w i hg (fun k => ⟨(hacc k).2.2.1, (hacc k).2.2.2.1, (hacc k).2.2.2.2.1⟩) hSr
        (fun k => (hfull k).2.2) hRo hA hK fun k => ?_
      have hk1 := (hno k (Finset.mem_univ _)).1
      have hP : PortsOk ok (sess x w i k) := by
        obtain ⟨ks, hm⟩ := hO x w i k
        have hm' : (ok, ks) ∈ known x w i k ++ Flock.KnownRead.fullRows (sess x w i k).own (full x w i k) :=
          Array.mem_append.2 (Or.inl hm)
        obtain ⟨-, hex⟩ := names_of_check (hacc k).2.2.2.1 hm'
        obtain ⟨reg, hf, -⟩ := FlockC3.RegLevels.checkKnownAll_ok (hacc k).2.1 hm
        exact ⟨hex _ _ hm', reg, Array.mem_toList_iff.1 (List.mem_of_find?_eq_some hf),
          by simpa using List.find?_some hf⟩
      have hR : CheckedRel (sess x w i k) (out x w i k) ok (openedReg (sess x w i k)) := by
        by_contra hR
        exact hk1 ⟨(hacc k).1, fun h => hR (h ⟨hP, (hfull k).2.1, hSr k⟩)⟩
      obtain ⟨-, vals, hv, he⟩ := hR
      exact ⟨vals, hv.trans (RowsSession.rowsReg_eq_portRead (sess x w i k) (hSr k) (hacc k).2.2.1 vals),
        fun u => he u (by rw [(hfull k).1]; exact Finset.mem_univ u)⟩
    · exact hColl x w i hc
  calc Pouw.RelativeSecurity.prob (fun x => pf.Unsound x w i)
      ≤ Pouw.RelativeSecurity.prob (fun x => ∃ k ∈ (Finset.univ : Finset ι),
          (checkedOnLayer (fun x w i => sess x w i k) (fun x w i => out x w i k) ok
            (fun s => PortsOk ok s ∧ s.kd = s.n ∧ ∀ u, s.ServedRegistered u)).Unsound x w i ∨
            ¬ (sess x w i k).Written (out x w i k)) :=
        Pouw.RelativeSecurity.prob_mono hIn
    _ ≤ ∑ k ∈ (Finset.univ : Finset ι), Pouw.RelativeSecurity.prob (fun x =>
          (checkedOnLayer (fun x w i => sess x w i k) (fun x w i => out x w i k) ok
            (fun s => PortsOk ok s ∧ s.kd = s.n ∧ ∀ u, s.ServedRegistered u)).Unsound x w i ∨
            ¬ (sess x w i k).Written (out x w i k)) :=
        Pouw.RelativeSecurity.prob_exists_le _ _
    _ ≤ ∑ _k ∈ (Finset.univ : Finset ι), δ := Finset.sum_le_sum fun k _ =>
          (Pouw.RelativeSecurity.prob_or_le _ _).trans
            ((add_le_add (hSound k w i) (hUnwritten k w i)).trans (add_zero δ).le)
    _ = Fintype.card ι * δ := by rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]

/-- **A proof accepted on every path with an opening that fails its relation is unsound with probability 1**, so no
layer with one is sound at a bound below 1. -/
theorem not_soundAt {Ω : Type} [Fintype Ω] {S nπ : ℕ} {V : Type} [Nonempty (Paths Ω S)] (pf : ProofLayer Ω S nπ V)
    (w : Fin S) (i : Fin nπ) (h : ∀ x, pf.Unsound x w i) {ε : ℝ} (hε : ε < 1) : ¬ pf.SoundAt ε := by
  intro hs
  have z : Pouw.RelativeSecurity.prob (fun x => ¬ pf.Unsound x w i) = 0 := by
    unfold Pouw.RelativeSecurity.prob
    simp [h]
  have e := Pouw.RelativeSecurity.prob_add_not (fun x => pf.Unsound x w i)
  linarith [hs w i]

/-- **`LeafGroupSound`'s nontrivial lemma**: at a group layer whose proof `i` of round `w` is accepted on every path
with an opening that fails its relation, and no clash of its sessions' registrants' records or of their known-content
ties, the guarantee's conclusion fails at every `δ` with `(T.lanes · n + 1) δ < 1`: the bound it states is about an
event that can happen, and the premises are what rule it out. -/
theorem groupLayer_not_soundAt {Ω : Type} [Fintype Ω] {S nπ n : ℕ} {V : Type} [Nonempty (Paths Ω S)]
    (T : PouwWindow.Lanes.Tile) (sess : Paths Ω S → Fin S → Fin nπ → Ix T.lanes n → RowsSession)
    (out : ∀ x w i k, (sess x w i k).Out) (known : Paths Ω S → Fin S → Fin nπ → Ix T.lanes n → Known)
    (okSalt : Paths Ω S → Fin S → Fin nπ → Ix T.lanes n → ByteArray)
    (full : Paths Ω S → Fin S → Fin nπ → Ix T.lanes n → Array String)
    (want : Paths Ω S → Fin S → Fin nπ → Ix T.lanes n → List _root_.Flock.Registered.Port)
    (opened : Paths Ω S → Fin S → Fin nπ → V) (rel : Paths Ω S → Fin S → Fin nπ → V → Prop)
    (right : Paths Ω S → (Fin S → Fin nπ → V) → Prop) (w : Fin S) (i : Fin nπ)
    (hacc : ∀ x, (groupLayer T n sess out known okSalt full want opened rel right).accepted x w i)
    (hrel : ∀ x, ¬ rel x w i (opened x w i)) (hcoll : ∀ x, ¬ LeafColl (sess x w i))
    (hkcoll : ∀ x, ¬ LeafKnownColl (sess x w i) fun k => okKnown (known x w i k) (okSalt x w i k))
    {δ : ℝ} (hδ : (T.lanes * n + 1) * δ < 1) :
    ¬ (groupLayer T n sess out known okSalt full want opened rel right).SoundAt ((T.lanes * n + 1) * δ) :=
  not_soundAt _ w i (fun x => ⟨hacc x, fun h => Or.elim h (hrel x) fun h => Or.elim h (hcoll x) (hkcoll x)⟩) hδ

end Accounting.TS.Group

open Accounting.TS Accounting.TS.Group in
/-- **A leaf's group of linked C-Flock sessions realizes one unit, any shape** (`Accounting.Guarantees.TS.LeafGroupSound`):
`group_soundAt` at the group layer, its `T.lanes · n + 1` statements (`card_ix`), the known content each session's
verifier builds ending in its `o0` row (`okKnown_names`), the served roots registered by the reads built from `own`
and the one-read check (`RowsSession.servedRegistered_canon`), and `GroupRealizes` at its relation's first disjunct. -/
theorem Accounting.SecurityProofs.TS.LeafGroupSound : Accounting.Guarantees.TS.LeafGroupSound := by
  intro Ω _ S nπ n V T sess₀ op sess out known okSalt full want opened rel right δ hCR hLaw hR hδ hδ0
  have h := group_soundAt (groupLayer T n sess out known okSalt full want opened rel right) sess out
    (fun x w i k => okKnown (known x w i k) (okSalt x w i k)) full want
    (fun _ _ _ => PouwWindow.Lanes.checkChunks T n = .ok ()) "o0" δ hCR hLaw
    (fun x w i k h => (sess₀ x w i k).servedRegistered_canon (op x w i k) (PouwWindow.Lanes.checkOneRead_ok h))
    (fun x w i k => okKnown_names _ _) hδ hδ0 (fun _ _ _ h => h)
    (fun x w i hg hc hS hn hRo hA hK hV => Or.inl (hR x w i hg hc hS hn hRo hA hK hV))
    (fun _ _ _ hc => Or.inr (Or.inl hc)) (fun _ _ _ hc => Or.inr (Or.inr hc))
  rw [card_ix, Nat.cast_add, Nat.cast_mul, Nat.cast_one] at h
  exact h
