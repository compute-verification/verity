import Proofs.Accounting.TS.Session.Lanes.Guarantees
import Proofs.Accounting.TS.CheckedSound
import Proofs.Accounting.TS.HeldSound
import Proofs.Pouw.RelativeSecurity
import Proofs.Flock.VBridge.Family.Registrants

/-!
# A leaf's 33 linked sessions realize one unit (the proof)

`Accounting.SecurityProofs.TS.LeafLanesSound`. The lanes cover the leaf (`Accounting.TS.Lanes.cover`: cell `(i, j)` is
lane `4 (i mod 8) + ⌊(j mod 8)/2⌋`'s at row slot `⌊i/8⌋` and column slot `2⌊j/8⌋ + (j mod 2)`), so a leaf whose 33
statements each hold at their committed rows is served right (`unitServed_of`): `LeafChecks` and `LeafReads` give each
lane's `LaneOk` and the combine's `CombineOk` at the one leaf and partials record, wherever the sessions' registrants'
rows agree. They agree unless two registrants' records clash (`rowsAgree_or_leafColl`, from
`FlockVBridge.Family.registrants_eq_or_collides`: two openings of one root at one position are one row's bytes or a
SHA-512 collision), and a clash meets the relation (`LeafColl`). An unsound proof of the leaf layer is then, at one of
its 33 sessions, an unsound proof of the checked layer (`CheckedSessionSound`'s `SoundAt δ`) or an
accepted outcome that doesn't draw every unit, which under `kd = n` has probability `0` in every session's game
(`RowsSession.prob_undrawn`) and so on the window (`WindowLaw`). The union bound over the 33 sessions gives `33 δ`.

This module imports `Proofs.Flock.SoundAtBridge` (through `Proofs.Accounting.TS.CheckedSound`), whose imports make `X`
a token, so `Proofs/Accounting.lean` imports it.
-/

namespace Flock.SoundAtBridge.RowsSession

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden FlockSoundness.Discharge.ZkReg

variable (s : RowsSession)

/-- **Under full checking no outcome of the session's game misses a unit.** -/
theorem prob_undrawn (hk : s.kd = s.n) : prob (fun o : s.Out => o.2.1 ≠ Finset.univ) s.G s.τ = 0 :=
  prob_audit_eq_zero s.L s.Reg (liveOfZJ s.dg (planZHJ s.dj s.y₀) s.N s.wr) (fun o => o.2.1 ≠ Finset.univ)
    (fun _ ω _ => nonpos_iff_eq_zero.1
      ((prob_mono (fun _ ht => ht (s.draw_full hk ω)) _ _).trans (prob_false _ _).le)) s.τ

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

/-- **A leaf's sessions' registrants' rows agree, or two of their records clash**: two registered reads at one root and
position carry one row's bytes, or their openings or hm96's leaves collide
(`FlockVBridge.Family.registrants_eq_or_collides`), which is `LeafColl`. -/
theorem rowsAgree_or_leafColl (sess : Option (Fin 32) → RowsSession) : RowsAgree sess ∨ LeafColl sess := by
  by_cases h : LeafColl sess
  · exact Or.inr h
  · refine Or.inl fun k k' u u' p p' hp hp' hport hat => ?_
    rcases FlockVBridge.Family.registrants_eq_or_collides ((sess k).rdU u) ((sess k').rdU u') hp hp' hport hat with
      he | hc | hc
    · exact he
    · exact absurd ⟨k, k', u, u', p, p', hp, hp', Or.inl hc⟩ h
    · exact absurd ⟨k, k', u, u', p, p', hp, hp', Or.inr hc⟩ h

/-- **The known-content tie at any known port** (`Accounting.TS.Held.tie_ok` at a port `nm` and its known rows `ks`,
any row): where flock-verify's known-content check accepted `own` at `known`, which names `nm` with rows `ks`, and every
unit's served roots are registered, a registrant's row at a registered read of a port named `nm` is the bytes of the
known row at its position, unless the tie collides (`OkColl` at `nm`). -/
theorem tie_known (s : RowsSession) {known : Known} {nm : String} {ks : Array (Array ℕ × ByteArray)}
    (hk : Flock.Registered.checkKnownAll s.own known = .ok ()) (hm : (nm, ks) ∈ known)
    (hS : ∀ u, s.ServedRegistered u) (hcol : ¬ Accounting.TS.Held.OkColl s nm ks) {u : Fin s.n} {p : s.vb.Pos}
    (hr : (s.rdU u).isRead p) (hn : ((s.rdU u).port p).name = nm) :
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
  obtain ⟨-, hb | ⟨a, b, hc, hab, hh⟩⟩ := FlockC3.RegLevels.knownColl_spec
    (r := (rsOf s).enc p ((s.rdU u).wv p)) (y := (s.rdU u).ws p) hck ho
  · exact hb
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

/-- **A leaf whose 33 statements hold at their committed rows is served right**: where each session's committed rows
are the registered reads of a wire assignment its evaluation agrees with at every unit, `LeafChecks` and the sessions'
decodes (`LeafReads` at one path where the rows agree) give every lane's `LaneOk` and the combine's `CombineOk` at the one leaf `L` and
partials record `parts`, and the lanes cover the leaf. -/
theorem unitServed_of {K nw : ℕ} {Pub : Type} {dq : List ℤ → ℤ}
    {part : Pub → Fin 32 → (Fin 4 → Row K) → (Fin 8 → Row K) → Fin nw → BitVec 32}
    {sess : Option (Fin 32) → RowsSession} {decL : RowsSession → (ℕ → ℕ → Option Bool) → LaneView K nw Pub}
    {decC : RowsSession → (ℕ → ℕ → Option Bool) → CombineView nw} {pub : Pub}
    {parts : Fin 32 → Fin nw → BitVec 32} {receipt : Fin nw → BitVec 32} {L : Leaf K}
    (hC : Accounting.Assumptions.TS.LeafChecks dq part sess decL decC)
    (hL : ∀ l, decL (sess (some l)) (openedReg (sess (some l))) = laneView L pub (parts l) l)
    (hCb : decC (sess none) (openedReg (sess none)) = ⟨parts, receipt, true⟩)
    (hV : ∀ k, ∃ vals : Fin (sess k).Circ.N → Bool,
      openedReg (sess k) = rowsReg (sess k) vals ∧ ∀ u, ∀ g ∈ (sess k).P.io u, (sess k).Circ.eval vals g = vals g) :
    UnitServed dq part pub receipt L := by
  have hlane : ∀ l, LaneOk dq part (laneView L pub (parts l) l) := by
    intro l
    obtain ⟨vals, hv, he⟩ := hV (some l)
    have hd : decL (sess (some l)) (rowsReg (sess (some l)) vals) = laneView L pub (parts l) l := by
      rw [← hv]; exact hL l
    have h := hC.1 l vals he ((congrArg LaneView.ok hd).trans rfl)
    rw [hd] at h
    exact h
  have hcomb : CombineOk (⟨parts, receipt, true⟩ : CombineView nw) := by
    obtain ⟨vals, hv, he⟩ := hV none
    have hd : decC (sess none) (rowsReg (sess none) vals) = ⟨parts, receipt, true⟩ := by
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
any known content `known` (the guarantee's is `knownOf` of the path's public inputs). -/
theorem Accounting.TS.Lanes.leafLanes_soundAt {Ω : Type} [Fintype Ω] {S nπ K nw : ℕ} {Pub : Type}
    (sess : Paths Ω S → Fin S → Fin nπ → Option (Fin 32) → RowsSession)
    (out : ∀ x w i k, (sess x w i k).Out) (known : Paths Ω S → Fin S → Fin nπ → Option (Fin 32) → Known)
    (ok : String) (dq : List ℤ → ℤ)
    (part : Pub → Fin 32 → (Fin 4 → Row K) → (Fin 8 → Row K) → Fin nw → BitVec 32)
    (decL : RowsSession → (ℕ → ℕ → Option Bool) → LaneView K nw Pub)
    (decC : RowsSession → (ℕ → ℕ → Option Bool) → CombineView nw)
    (pub : Paths Ω S → Fin S → Fin nπ → Pub) (parts : Paths Ω S → Fin S → Fin nπ → Fin 32 → Fin nw → BitVec 32)
    (receipt : Paths Ω S → Fin S → Fin nπ → Fin nw → BitVec 32) (leaf : Paths Ω S → Fin S → Fin nπ → Leaf K)
    (right : Paths Ω S → (Fin S → Fin nπ → Leaf K) → Prop) (δ : ℝ)
    (hCR : ∀ x w i k, (sess x w i k).CR)
    (hLaw : ∀ k, Accounting.Assumptions.TS.WindowLaw (fun x w i => sess x w i k) (fun x w i => out x w i k))
    (hk : ∀ x w i k, (sess x w i k).kd = (sess x w i k).n)
    (hN : ∀ x w i k (q q' : Fin (sess x w i k).c.ports.size),
      (sess x w i k).c.ports[q].name = (sess x w i k).c.ports[q'].name → q = q')
    (hS : ∀ x w i k u, (sess x w i k).ServedRegistered u)
    (hok : ∀ x w i k, (∃ q : Fin (sess x w i k).c.ports.size, (sess x w i k).c.ports[q].name = ok) ∧
      ∃ pt ∈ (sess x w i k).own, pt.name = ok)
    (hδ : ∀ x w i k, ((sess x w i k).n : ℝ≥0∞) * (sess x w i k).bound ≤ ENNReal.ofReal δ) (hδ0 : 0 ≤ δ)
    (hC : ∀ x w i, Accounting.Assumptions.TS.LeafChecks dq part (sess x w i) decL decC)
    (hSh : Accounting.Assumptions.TS.LeafReads sess known decL decC pub parts receipt leaf) :
    (leafLayer sess out known dq part pub receipt leaf right).SoundAt (33 * δ) := by
  have hSound : ∀ k, (checkedLayer (fun x w i => sess x w i k) (fun x w i => out x w i k) ok
      (fun _ _ => True)).SoundAt δ := fun k =>
    (Accounting.SecurityProofs.TS.CheckedSessionSound (fun x w i => sess x w i k) (fun x w i => out x w i k) ok
      (fun _ _ => True) δ (fun x w i => hCR x w i k) (hLaw k) (fun x w i => hk x w i k) (fun x w i => hN x w i k)
      (fun x w i => hS x w i k) (fun x w i => hok x w i k) (fun x w i => hδ x w i k) hδ0).2.2.2
  have hUndrawn : ∀ k (w : Fin S) (i : Fin nπ),
      Pouw.RelativeSecurity.prob (fun x => (out x w i k).2.1 ≠ Finset.univ) ≤ 0 := fun k w i =>
    (hLaw k w i fun s o => o.2.1 ≠ Finset.univ).trans (avg_le _ le_rfl fun x => le_of_eq
      ((congrArg ENNReal.toReal (RowsSession.prob_undrawn (sess x w i k) (hk x w i k))).trans ENNReal.toReal_zero))
  intro w i
  have hIn : ∀ x, (leafLayer sess out known dq part pub receipt leaf right).Unsound x w i →
      ∃ k ∈ (Finset.univ : Finset (Option (Fin 32))),
        (checkedLayer (fun x w i => sess x w i k) (fun x w i => out x w i k) ok (fun _ _ => True)).Unsound x w i ∨
        (out x w i k).2.1 ≠ Finset.univ := by
    intro x ⟨hacc, hrel⟩
    by_contra hno
    push Not at hno
    apply hrel
    rcases rowsAgree_or_leafColl (sess x w i) with hA | hc
    · rcases knownHeld_or_coll (sess x w i) (known x w i) (fun k => (hacc k).2) (hS x w i) with hK | hK
      swap
      · exact Or.inr (Or.inr hK)
      obtain ⟨hL, hCb⟩ := hSh x w i hA hK
      refine Or.inl (unitServed_of (hC x w i) hL hCb fun k => ?_)
      obtain ⟨hk1, hk2⟩ := hno k (Finset.mem_univ _)
      have hR : CheckedRel (sess x w i k) (out x w i k) ok (openedReg (sess x w i k)) := by
        by_contra hR
        exact hk1 ⟨(hacc k).1, hR⟩
      obtain ⟨-, vals, hv, he⟩ := hR
      exact ⟨vals, hv, fun u => he u (by rw [hk2]; exact Finset.mem_univ u)⟩
    · exact Or.inr (Or.inl hc)
  calc Pouw.RelativeSecurity.prob (fun x => (leafLayer sess out known dq part pub receipt leaf right).Unsound x w i)
      ≤ Pouw.RelativeSecurity.prob (fun x => ∃ k ∈ (Finset.univ : Finset (Option (Fin 32))),
          (checkedLayer (fun x w i => sess x w i k) (fun x w i => out x w i k) ok (fun _ _ => True)).Unsound x w i ∨
          (out x w i k).2.1 ≠ Finset.univ) := Pouw.RelativeSecurity.prob_mono hIn
    _ ≤ ∑ k ∈ (Finset.univ : Finset (Option (Fin 32))), Pouw.RelativeSecurity.prob (fun x =>
          (checkedLayer (fun x w i => sess x w i k) (fun x w i => out x w i k) ok (fun _ _ => True)).Unsound x w i ∨
          (out x w i k).2.1 ≠ Finset.univ) := Pouw.RelativeSecurity.prob_exists_le _ _
    _ ≤ ∑ _k ∈ (Finset.univ : Finset (Option (Fin 32))), δ := Finset.sum_le_sum fun k _ =>
          (Pouw.RelativeSecurity.prob_or_le _ _).trans
            ((add_le_add (hSound k w i) (hUndrawn k w i)).trans (add_zero δ).le)
    _ = 33 * δ := by simp

open Accounting.TS.Lanes in
/-- **A leaf's 33 linked C-Flock sessions realize one unit** (`Accounting.Guarantees.TS.LeafLanesSound`):
`leafLanes_soundAt` at the known content of the path's public inputs (`knownOf`). -/
theorem Accounting.SecurityProofs.TS.LeafLanesSound : Accounting.Guarantees.TS.LeafLanesSound := by
  intro Ω _ S nπ K nw sess out salts ok dq part decL decC pub parts receipt leaf right δ
  exact leafLanes_soundAt sess out (fun x w i k => knownOf (pub x w i) (salts x w i k) k) ok dq part decL decC pub parts
    receipt leaf right δ
