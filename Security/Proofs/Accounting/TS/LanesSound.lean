import Proofs.Accounting.TS.Session.Lanes.Guarantees
import Proofs.Accounting.TS.Session.Lanes.KnownRead
import Proofs.Accounting.TS.Session.Lanes.Partials
import Proofs.Accounting.TS.Session.Lanes.Committed
import Proofs.Flock.C3.RegLevels.TopLeaf
import Proofs.Accounting.TS.CheckedSound
import Proofs.Accounting.TS.HeldSound
import Proofs.Pouw.RelativeSecurity
import Proofs.Flock.VBridge.Family.Registrants

/-!
# A leaf's 33 linked sessions realize one unit (the proof)

`Accounting.SecurityProofs.TS.LeafLanesSound`. The lanes cover the leaf (`Accounting.TS.Lanes.cover`: cell `(i, j)` is
lane `4 (i mod 8) + ⌊(j mod 8)/2⌋`'s at row slot `⌊i/8⌋` and column slot `2⌊j/8⌋ + (j mod 2)`), so a leaf whose 33
statements each hold at their committed rows is served right (`unitServed_of`): `LeafChecks`, which the circuits
computing their Definitions give (`leafChecks_of_computes`), gives each lane's `LaneOk` at a `K` the lanes bind at,
which the leaf verifier's tile check gives (`k_of_check`), and the combine's `CombineOk`, and each lane's view agrees with the leaf on what `LaneOk` reads
(`LaneAgrees`). The leaf is the one the lanes' committed rows decode to and the receipt the combine's
(`Accounting.TS.Lanes.leafOf`, `receiptOf`): each A row as lane `4 (r mod 8)` decodes it, each B row as lane
`⌊(j mod 8)/2⌋` does, each cell of the record as the one lane whose cell it is. The four lanes that read one A row (or
one B row) read one registered tree at one row in full, so wherever the sessions' registrants' rows agree they decode
it alike (`read_bits`, `decodes_of_checks`). They agree unless two registrants' records clash
(`rowsAgree_or_leafColl`, from `FlockSoundness.Registered.enc_eq_of_not_clash`: two openings of one tree at one
position are one row's bytes or a SHA-512 collision), and a clash meets the relation (`LeafColl`). The known rows hold
unless a session's known-content tie collides (`knownHeld_or_coll`, from flock-verify's known-content check), which
meets the relation too (`LeafKnownColl`); the known ports' fields then follow from the known-read check
(`laneKnownHolds_of_check`, `combKnownOk_of_check`). The partials record is the combine's decoded one: lane `l`'s
`partials` and the combine's `p<l>` are `ownOf`'s ports of one table at the leaf's row `l`, placed from one base row
for the unit (`place`), read in full (`partials_bits`), where
every unit reads its one-row ports at their rows (`RowsSession.readsOwnRow_of`, at a written outcome that draws every
unit). `LeafChecks` reads every port row; under flock-verify's complete check
(`Flock.Registered.checkComplete`, part of the leaf layer's acceptance) the registered reads the checked layer opens
are exactly those (`RowsSession.rowsReg_eq_portRead`), and each of them opens at its registered root
(`RowsSession.portRow_opens`; `Flock.Registered.complete_opens` at the verifier). An unsound proof of the leaf layer is
then, at one of its 33 sessions, an unsound proof of the checked layer where its ports are named once and include `ok`
(`Accounting.TS.Checked.checkedOn_soundAt`'s `SoundAt δ`, `CheckedSessionSound`'s argument with those asked only where
they hold, which the known-read check's names give at known content naming `ok`: `names_of_check`, with full checking
asked there too) or an accepted outcome that isn't the session's write of a live run (`RowsSession.Written`), which
has probability `0` in every session's game (`RowsSession.prob_unwritten`) and so on the window (`WindowLaw`). At a
written outcome the leaf verifier's draw check (`PouwWindow.Lanes.checkFull`, part of the leaf layer's acceptance)
gives full checking: the record's draw is `subset` of every unit of a non-empty population, and the session's custody
makes it the verifier's draw `subset kd` over `n` units, so the outcome draws every unit, `kd = n` and `0 < n`
(`RowsSession.full_of_check`). Every unit's served roots are registered at an accepted proof
(`RowsSession.ServedRegistered`): the leaf verifier's one-read check (`PouwWindow.Lanes.checkOneRead`, part of the
leaf layer's acceptance) holds there, and `leafLanes_soundAt` takes the served roots registered wherever it does,
which the guarantee's sessions, their reads built from `own` (`RowsSession.canon`), give
(`RowsSession.servedRegistered_canon`); the checked layer asks them only where its setup predicate holds, as it does
full checking. The union bound over the 33 sessions gives `33 δ`.

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

/-- **The 32 lanes cover the leaf**: cell `(i, j)` is lane `4 (i mod 8) + ⌊(j mod 8)/2⌋`'s at row slot `⌊i/8⌋` and
column slot `2⌊j/8⌋ + (j mod 2)`. -/
theorem cover (i j : Fin 32) : ∃ l s t, rowOf l s = i ∧ colOf l t = j := by
  have hi := i.isLt
  have hj := j.isLt
  refine ⟨⟨4 * (i.val % 8) + (j.val % 8) / 2, by omega⟩, ⟨i.val / 8, by omega⟩, ⟨2 * (j.val / 8) + j.val % 2, by omega⟩,
    Fin.ext ?_, Fin.ext ?_⟩
  · show (4 * (i.val % 8) + (j.val % 8) / 2) / 4 + 8 * (i.val / 8) = i.val
    omega
  · show 2 * ((4 * (i.val % 8) + (j.val % 8) / 2) % 4) + 8 * ((2 * (j.val / 8) + j.val % 2) / 2) +
      (2 * (j.val / 8) + j.val % 2) % 2 = j.val
    omega

/-- **A leaf's sessions' registrants' rows agree, or two of their records clash**: two registered reads at one tree
(root, leaf count, domain and schema) and position carry one row's bytes, or their openings or hm96's leaves collide
(`FlockSoundness.Registered.enc_eq_of_not_clash`), which is `LeafColl`. -/
theorem rowsAgree_or_leafColl (sess : Option (Fin 32) → RowsSession) : RowsAgree sess ∨ LeafColl sess := by
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

/-- **A leaf's sessions' known rows hold, or a known-content tie collides**: at sessions whose known-content checks
passed and whose served roots are registered (`tie_known` at each). -/
theorem knownHeld_or_coll (sess : Option (Fin 32) → RowsSession) (known : Option (Fin 32) → Known)
    (hk : ∀ k, Flock.Registered.checkKnownAll (sess k).own (known k) = .ok ())
    (hS : ∀ k u, (sess k).ServedRegistered u) :
    (∀ k, KnownHeld (sess k) (known k)) ∨ LeafKnownColl sess known := by
  by_cases h : LeafKnownColl sess known
  · exact Or.inr h
  · exact Or.inl fun k nm ks hm u p hr hn =>
      tie_known (sess k) (hk k) hm (hS k) (fun hc => h ⟨k, nm, ks, hm, hc⟩) hr hn

/-- **The leaf verifier's tile check gives a `K` the lanes bind at**: where `PouwWindow.Lanes.Tile.check` accepts a
tile, its `K` is whole 32-wide steps and below `2^17`, where the lanes' s8 accumulator is exact. -/
theorem k_of_check {T : PouwWindow.Lanes.Tile} (h : T.check = .ok ()) : 32 ∣ T.k ∧ T.k < 2 ^ 17 := by
  unfold PouwWindow.Lanes.Tile.check at h
  by_cases h1 : T.k % 32 = 0
  · refine ⟨Nat.dvd_of_mod_eq_zero h1, Nat.lt_of_not_le fun h2 => ?_⟩
    have h2' : 131072 ≤ T.k := h2
    have h3 : ¬T.k < 64 := by omega
    simp [h1, h2', h3] at h
    cases h
  · simp [h1] at h
    cases h

/-- **A leaf whose 33 statements hold at their committed rows is served right**: at a `K` the lanes bind at, where
each session's committed rows are the port rows of a wire assignment its evaluation agrees with at every unit, each
lane's decode agrees with the leaf `L` on what `LaneOk` reads (`LaneAgrees`) and the combine's decode is the partials
record `parts`, the receipt and `ok = 1`, `LeafChecks` gives every lane's `LaneOk` and the combine's `CombineOk`, and
the lanes cover the leaf. -/
theorem unitServed_of {K nw : ℕ} {Pub : Type} {dq : List ℤ → ℤ}
    {part : Pub → Fin 32 → (Fin 4 → Row K) → (Fin 8 → Row K) → Fin nw → BitVec 32}
    {sess : Option (Fin 32) → RowsSession} {decL : RowsSession → (ℕ → ℕ → Option Bool) → LaneView K nw Pub}
    {decC : RowsSession → (ℕ → ℕ → Option Bool) → CombineView nw} {pub : Pub}
    {parts : Fin 32 → Fin nw → BitVec 32} {receipt : Fin nw → BitVec 32} {L : Leaf K}
    (hC : Accounting.Assumptions.TS.LeafChecks dq part sess decL decC) (hK32 : 32 ∣ K) (hK : K < 2 ^ 17)
    (hL : ∀ l, LaneAgrees (decL (sess (some l)) (openedReg (sess (some l)))) L pub (parts l) l)
    (hCb : decC (sess none) (openedReg (sess none)) = ⟨parts, receipt, true⟩)
    (hV : ∀ k, ∃ vals : Fin (sess k).Circ.N → Bool,
      openedReg (sess k) = portRead (sess k) vals ∧ ∀ u, ∀ g ∈ (sess k).P.io u, (sess k).Circ.eval vals g = vals g) :
    UnitServed dq part pub receipt L := by
  have hlane : ∀ l, LaneOk dq part (laneView L pub (parts l) l) := by
    intro l
    obtain ⟨vals, hv, he⟩ := hV (some l)
    have hd : LaneAgrees (decL (sess (some l)) (portRead (sess (some l)) vals)) L pub (parts l) l := by
      rw [← hv]; exact hL l
    exact hd.laneOk (hC.1 hK32 hK l vals he hd.ok)
  have hcomb : CombineOk (⟨parts, receipt, true⟩ : CombineView nw) := by
    obtain ⟨vals, hv, he⟩ := hV none
    have hd : decC (sess none) (portRead (sess none) vals) = ⟨parts, receipt, true⟩ := by
      rw [← hv]; exact hCb
    have h := hC.2 vals he ((congrArg CombineView.ok hd).trans rfl)
    rw [hd] at h
    exact h
  refine ⟨fun i j => ?_, ?_⟩
  · obtain ⟨l, s, t, rfl, rfl⟩ := cover i j
    exact (hlane l).2 s t
  · have hp : parts = fun l => part pub l (laneA L l) (laneB L l) := funext fun l => (hlane l).1
    exact hcomb.symm.trans (congrArg xorAll hp)

end Accounting.TS.Lanes

open Accounting.TS Accounting.TS.Checked Accounting.TS.Lanes Flock.SoundAtBridge FlockSoundness.Discharge.ZkReg
open Pouw.RelativeSecurity (Paths) in
open scoped ENNReal in
/-- **A leaf's 33 linked C-Flock sessions realize one unit, at any public inputs and known content**: the leaf layer is
sound at `33 δ` under `Accounting.Guarantees.TS.LeafLanesSound`'s hypotheses with any type `Pub` of public inputs and
any known content `known` naming `ok` (the guarantee's is `knownOf` of the path's public inputs, `ok` its `o0`), at
any sessions whose units' served roots are registered wherever their files have one read at each port (the
guarantee's, built from `own`, by `RowsSession.servedRegistered_canon`). -/
theorem Accounting.TS.Lanes.leafLanes_soundAt {Ω : Type} [Fintype Ω] {S nπ K nw : ℕ} {Pub : Type}
    (sess : Paths Ω S → Fin S → Fin nπ → Option (Fin 32) → RowsSession)
    (out : ∀ x w i k, (sess x w i k).Out) (known : Paths Ω S → Fin S → Fin nπ → Option (Fin 32) → Known)
    (full : Paths Ω S → Fin S → Fin nπ → Option (Fin 32) → Array String)
    (want : Paths Ω S → Fin S → Fin nπ → Option (Fin 32) → List _root_.Flock.Registered.Port)
    (ok : String) (dq : List ℤ → ℤ)
    (part : Pub → Fin 32 → (Fin 4 → Row K) → (Fin 8 → Row K) → Fin nw → BitVec 32)
    (decL : RowsSession → (ℕ → ℕ → Option Bool) → LaneView K nw Pub)
    (decC : RowsSession → (ℕ → ℕ → Option Bool) → CombineView nw)
    (pub : Paths Ω S → Fin S → Fin nπ → Pub)
    (receipt : Paths Ω S → Fin S → Fin nπ → Fin nw → BitVec 32) (leaf : Paths Ω S → Fin S → Fin nπ → Leaf K)
    (right : Paths Ω S → (Fin S → Fin nπ → Leaf K) → Prop) (δ : ℝ)
    (hCR : ∀ x w i k, (sess x w i k).CR)
    (hLaw : ∀ k, Accounting.Assumptions.TS.WindowLaw (fun x w i => sess x w i k) (fun x w i => out x w i k))
    (hS : ∀ x w i k, PouwWindow.Lanes.checkOneRead (sess x w i k).own = .ok () →
      ∀ u, (sess x w i k).ServedRegistered u)
    (hO : ∀ x w i k, ∃ ks, (ok, ks) ∈ known x w i k)
    (hδ : ∀ x w i k, ((sess x w i k).n : ℝ≥0∞) * (sess x w i k).bound ≤ ENNReal.ofReal δ) (hδ0 : 0 ≤ δ)
    (hC : ∀ x w i, Accounting.Assumptions.TS.LeafChecks dq part (sess x w i) decL decC)
    (hSh : ∀ x w i, (∀ k, Flock.Registered.checkComplete (sess x w i k).c (sess x w i k).own = .ok () ∧
        Flock.KnownRead.check (sess x w i k).c
          (known x w i k ++ Flock.KnownRead.fullRows (sess x w i k).own (full x w i k)) = .ok () ∧
        PouwWindow.Lanes.checkOwn (want x w i k) (sess x w i k).own = .ok ()) →
      (∀ k u, (sess x w i k).ServedRegistered u) →
      (∀ k, 0 < (sess x w i k).n) → (∀ k, ReadsOwnRow (sess x w i k)) →
      RowsAgree (sess x w i) → (∀ k, KnownHeld (sess x w i k) (known x w i k)) →
      ∃ parts : Fin 32 → Fin nw → BitVec 32,
        (∀ l, LaneAgrees (decL (sess x w i (some l)) (openedReg (sess x w i (some l)))) (leaf x w i) (pub x w i)
          (parts l) l) ∧
        decC (sess x w i none) (openedReg (sess x w i none)) = ⟨parts, receipt x w i, true⟩) :
    (leafLayer sess out known full want dq part pub receipt leaf right).SoundAt (33 * δ) := by
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
  have hIn : ∀ x, (leafLayer sess out known full want dq part pub receipt leaf right).Unsound x w i →
      ∃ k ∈ (Finset.univ : Finset (Option (Fin 32))),
        (checkedOnLayer (fun x w i => sess x w i k) (fun x w i => out x w i k) ok
          (fun s => PortsOk ok s ∧ s.kd = s.n ∧ ∀ u, s.ServedRegistered u)).Unsound x w i ∨
          ¬ (sess x w i k).Written (out x w i k) := by
    intro x ⟨hacc, hrel⟩
    by_contra hno
    push Not at hno
    apply hrel
    have hfull : ∀ k, (out x w i k).2.1 = Finset.univ ∧ (sess x w i k).kd = (sess x w i k).n ∧
        0 < (sess x w i k).n := fun k =>
      RowsSession.full_of_check (sess x w i k) (hacc k).1 (hno k (Finset.mem_univ _)).2 (hacc k).2.2.2.2.2.1
    have hSr : ∀ k u, (sess x w i k).ServedRegistered u := fun k => hS x w i k (hacc k).2.2.2.2.2.2.1
    have hKr := k_of_check (hacc none).2.2.2.2.2.2.2
    rcases rowsAgree_or_leafColl (sess x w i) with hA | hc
    · rcases knownHeld_or_coll (sess x w i) (known x w i) (fun k => (hacc k).2.1) hSr with hK | hK
      swap
      · exact Or.inr (Or.inr hK)
      have hRo : ∀ k, ReadsOwnRow (sess x w i k) := fun k =>
        RowsSession.readsOwnRow_of (sess x w i k) (hacc k).1 (hno k (Finset.mem_univ _)).2 (hfull k).1 (hSr k)
      obtain ⟨parts, hL, hCb⟩ := hSh x w i (fun k => ⟨(hacc k).2.2.1, (hacc k).2.2.2.1, (hacc k).2.2.2.2.1⟩)
        hSr (fun k => (hfull k).2.2) hRo hA hK
      refine Or.inl (unitServed_of (hC x w i) hKr.1 hKr.2 hL hCb fun k => ?_)
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
    · exact Or.inr (Or.inl hc)
  calc Pouw.RelativeSecurity.prob (fun x => (leafLayer sess out known full want dq part pub receipt leaf right).Unsound x w i)
      ≤ Pouw.RelativeSecurity.prob (fun x => ∃ k ∈ (Finset.univ : Finset (Option (Fin 32))),
          (checkedOnLayer (fun x w i => sess x w i k) (fun x w i => out x w i k) ok
            (fun s => PortsOk ok s ∧ s.kd = s.n ∧ ∀ u, s.ServedRegistered u)).Unsound x w i ∨
            ¬ (sess x w i k).Written (out x w i k)) :=
        Pouw.RelativeSecurity.prob_mono hIn
    _ ≤ ∑ k ∈ (Finset.univ : Finset (Option (Fin 32))), Pouw.RelativeSecurity.prob (fun x =>
          (checkedOnLayer (fun x w i => sess x w i k) (fun x w i => out x w i k) ok
            (fun s => PortsOk ok s ∧ s.kd = s.n ∧ ∀ u, s.ServedRegistered u)).Unsound x w i ∨
            ¬ (sess x w i k).Written (out x w i k)) :=
        Pouw.RelativeSecurity.prob_exists_le _ _
    _ ≤ ∑ _k ∈ (Finset.univ : Finset (Option (Fin 32))), δ := Finset.sum_le_sum fun k _ =>
          (Pouw.RelativeSecurity.prob_or_le _ _).trans
            ((add_le_add (hSound k w i) (hUnwritten k w i)).trans (add_zero δ).le)
    _ = 33 * δ := by simp

open Accounting.TS Accounting.TS.Lanes Flock.SoundAtBridge PouwWindow.Lanes
open Pouw.RelativeSecurity (Paths) in
/-- **The decodes read the committed leaf, from acceptance**: where each session has a unit, its served roots
registered, whose units read their one-row ports at their rows (`ReadsOwnRow`), whose registrants' rows agree at one
tree and position (`RowsAgree`) and whose known rows hold, at every proof whose sessions pass flock-verify's complete
and known-read checks at the known content of the path's public inputs (`knownOf`) and the ports read in full
(`readFull`), and the leaf verifier's check of their files at `ownOf` (`roots`, the unit's placement `place`,
`checkOwn`), each lane's decode agrees
with the leaf the lanes' committed rows decode to (`leafOf`) at the combine's decoded partials, and the combine's
decode is those partials, its decoded receipt (`receiptOf`) and `ok = 1`. Lane `l`'s A row at slot `s` is lane
`4⌊l/4⌋`'s at that slot and its B row at slot `t` lane `l mod 4`'s, one row of one tree each lane reads in full
(`read_bits`); its cells are its own (`cellLane_rowOf_colOf`, `laneAgrees_of`); its partials row is the combine's
(`partials_bits`, `part_eq`); its known fields are the checks' (`laneKnownHolds_of_check`, `combKnownOk_of_check`). -/
theorem Accounting.TS.Lanes.decodes_of_checks {Ω : Type} {S nπ K nw : ℕ}
    {sess : Paths Ω S → Fin S → Fin nπ → Option (Fin 32) → RowsSession}
    {salts : Paths Ω S → Fin S → Fin nπ → Option (Fin 32) → String → ByteArray} {pub : Paths Ω S → Fin S → Fin nπ → LanePub}
    {roots : Paths Ω S → Fin S → Fin nπ → String → _root_.Flock.Registered.Port}
    {place : Paths Ω S → Fin S → Fin nπ → String → String × ℕ}
    (x : Paths Ω S) (w : Fin S) (i : Fin nπ)
    (hc : ∀ k, Flock.Registered.checkComplete (sess x w i k).c (sess x w i k).own = .ok () ∧
      Flock.KnownRead.check (sess x w i k).c (knownOf (pub x w i) (salts x w i k) k ++
        Flock.KnownRead.fullRows (sess x w i k).own (readFull (servedTile K) (k.map Fin.val))) = .ok () ∧
      checkOwn (ownOf (roots x w i) (place x w i) (servedTile K) (k.map Fin.val)) (sess x w i k).own = .ok ())
    (hS : ∀ k u, (sess x w i k).ServedRegistered u)
    (hn : ∀ k, 0 < (sess x w i k).n) (hRo : ∀ k, ReadsOwnRow (sess x w i k))
    (hA : RowsAgree (sess x w i)) (hK : ∀ k, KnownHeld (sess x w i k) (knownOf (pub x w i) (salts x w i k) k)) :
    ∃ parts : Fin 32 → Fin nw → BitVec 32,
      (∀ l, LaneAgrees (decL K nw (servedTile K) (sess x w i (some l)) (openedReg (sess x w i (some l))))
        (leafOf K (servedTile K) (sess x w i)) (pub x w i) (parts l) l) ∧
      decC nw (servedTile K) (sess x w i none) (openedReg (sess x w i none)) =
        ⟨parts, receiptOf nw (servedTile K) (sess x w i), true⟩ := by
  have hP : ∀ l : Fin 32, (decL K nw (servedTile K) (sess x w i (some l)) (openedReg (sess x w i (some l)))).part =
      (decC nw (servedTile K) (sess x w i none) (openedReg (sess x w i none))).parts l := fun l =>
    part_eq (partials_bits (T := servedTile K) (roots := roots x w i) (place := place x w i) (l := l.val)
      (by rw [servedTile_lanes]; exact l.isLt) (checkOwn_ok (hc (some l)).2.2) (checkOwn_ok (hc none).2.2)
      (hc (some l)).1 (hc none).1 (hc (some l)).2.1 (hc none).2.1 (uL := ⟨0, hn (some l)⟩) rfl (hS (some l) _)
      (uC := ⟨0, hn none⟩) rfl (hS none _) (hRo (some l)) (hRo none)
      (hA (some l) none ⟨0, hn (some l)⟩ ⟨0, hn none⟩))
  have hbits : ∀ (l l' : Fin 32) (nm nm' v : String) (row : ℕ), (nm, v, row) ∈ regReads (servedTile K) (some l.val) →
      (nm', v, row) ∈ regReads (servedTile K) (some l'.val) → fullValue v = true →
      portBit (sess x w i (some l)) (openedReg (sess x w i (some l))) nm =
        portBit (sess x w i (some l')) (openedReg (sess x w i (some l'))) nm' :=
    fun l l' _ _ _ _ hm hm' hv => funext fun b =>
      read_bits (roots := roots x w i) (place := place x w i) hm hm' (full_of hm hv) (full_of hm' hv) (checkOwn_ok (hc (some l)).2.2)
        (checkOwn_ok (hc (some l')).2.2) (hc (some l)).1 (hc (some l')).1 (hc (some l)).2.1 (hc (some l')).2.1
        (uL := ⟨0, hn (some l)⟩) rfl (hS (some l) _) (uC := ⟨0, hn (some l')⟩) rfl
        (hS (some l') _) (hRo (some l)) (hRo (some l'))
        (hA (some l) (some l') ⟨0, hn (some l)⟩ ⟨0, hn (some l')⟩) b
  refine ⟨(decC nw (servedTile K) (sess x w i none) (openedReg (sess x w i none))).parts, fun l => ?_, ?_⟩
  · refine laneAgrees_of (sess x w i) l (fun s => ?_) (fun t => ?_) (hP l)
      (laneKnownHolds_of_check (s := sess x w i (some l)) (servedTile K) (hK (some l))
        (u := ⟨0, hn (some l)⟩) rfl (hS (some l) _) (hc (some l)).1 (hc (some l)).2.1)
    · have hm' := a_read K (aHome l) s
      rw [rowOf_aHome] at hm'
      exact hbits l (aHome l) _ _ _ _ (a_read K l s) hm' fullValue_a
    · have hm' := b_read K (bHome l) t
      rw [colOf_bHome] at hm'
      exact hbits l (bHome l) _ _ _ _ (b_read K l t) hm' fullValue_b
  · have ho := combKnownOk_of_check (s := sess x w i none) (hK none) (u := ⟨0, hn none⟩) rfl
      (hS none _) (hc none).1 (hc none).2.1
    refine CombineView.eta_ok _ ?_
    unfold decC
    rw [decComb_ok, ho]
    rfl

open Accounting.TS.Lanes in
/-- **A leaf's 33 linked C-Flock sessions realize one unit** (`Accounting.Guarantees.TS.LeafLanesSound`):
`leafLanes_soundAt` at the known content of the path's public inputs (`knownOf`), the ports read in full at the layout
(`PouwWindow.Lanes.readFull`), the leaf verifier's check of each file at `ownOf` (`roots`, `place`), the layout's decodes at the
served tile, `LeafChecks` from the 33 circuits computing their pieces' compositions (`leafChecks_of_computes`,
`laneComputes_of_lowers`, `combineComputes_of_lowers`), and the leaf and
receipt the committed rows decode to (`leafOf`, `receiptOf`), whose agreement with the decodes `decodes_of_checks`
gives from acceptance. -/
theorem Accounting.SecurityProofs.TS.LeafLanesSound : Accounting.Guarantees.TS.LeafLanesSound := by
  intro Ω _ S nπ K nw sess₀ op sess out salts pub roots place right δ hCR hLaw hδ hδ0 hL hC
  exact leafLanes_soundAt sess out (fun x w i k => knownOf (pub x w i) (salts x w i k) k)
    (fun _ _ _ k => PouwWindow.Lanes.readFull (servedTile K) (k.map Fin.val))
    (fun x w i k => PouwWindow.Lanes.ownOf (roots x w i) (place x w i) (servedTile K) (k.map Fin.val)) "o0"
    (dqOf Pouw.H1Zy.zyDequant) (partOf (laneChain K nw))
    (decL K nw (servedTile K)) (decC nw (servedTile K)) pub (fun x w i => receiptOf nw (servedTile K) (sess x w i))
    (fun x w i => leafOf K (servedTile K) (sess x w i)) right δ hCR hLaw
    (fun x w i k h => (sess₀ x w i k).servedRegistered_canon (op x w i k) (PouwWindow.Lanes.checkOneRead_ok h))
    (fun x w i k => ⟨#[(PouwWindow.Lanes.okRow, salts x w i k "o0")], by
      cases k <;> simp [knownOf, PouwWindow.Lanes.knownRows]⟩) hδ hδ0
    (fun x w i => leafChecks_of_computes (servedTile K) (laneChain K nw) Pouw.H1Zy.zyDequant (sess x w i) (laneComputes_of_lowers (hL x w i))
      (combineComputes_of_lowers (hC x w i)))
    (decodes_of_checks (roots := roots) (place := place))
