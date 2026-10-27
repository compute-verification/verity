import Proofs.Accounting.TS.Session.Held.Guarantees
import Proofs.Accounting.TS.CheckedSound
import Proofs.Pouw.RelativeSecurity

/-!
# A checked unit's proof layer with its served checks is sound through its C-Flock sessions (the proof)

`Accounting.SecurityProofs.TS.HeldSessionSound`. The held layer accepts where the checked layer does and flock-verify's
known-content check passes, and opens the same rows, so `AcceptedLive` is its acceptance's first half. Its unsound event
is inside the checked layer's (`Pouw.RelativeSecurity.prob_mono`), because on every path an accepted proof whose rows
meet `CheckedRel` meets `HeldRel` (`held_of_checked`):

* (a) at each `ok` wire of a unit, the registrant's opening was registered at `own`'s port of that name
  (`RowsSession.ServedRegistered`), which `checkKnownAll` accepted at the known rows; so its row bytes are the word
  `1`'s, or `knownColl` exhibits a SHA-512 collision (`FlockC3.RegLevels.knownColl_spec`, `tie_ok`);
* (b) the checked relation's assignment is the opened rows at each such wire, so its `ok` wires carry the `ok` row,
  and `Accounting.Assumptions.TS.OkRealizes` gives each drawn unit's checks of `Cd`.
-/

namespace Accounting.TS.Held

open Accounting.TS Accounting.TS.Checked Flock.SoundAtBridge FlockSoundness.Discharge.ZkReg
open FlockSoundness.Refine (H512)

theorem regRead_at (s : RowsSession) (f : Fin s.n → Fin s.Circ.N → Bool) {u : Fin s.n} {g : Fin s.Circ.N}
    (hg : g ∈ s.P.io u) (hr : (s.rdU u).isRead (s.vb.pos g)) : regRead s f u.val g.val = some (f u g) := by
  dsimp only [regRead]
  split_ifs with h1 h2
  · rfl
  · exact (h2 ⟨hg, hr⟩).elim
  · exact (h1 ⟨u.isLt, g.isLt⟩).elim

theorem find_name_eq {l : List Flock.Registered.Port} {n : String} {pt : Flock.Registered.Port} (hpt : pt ∈ l)
    (hn : pt.name = n) : ((l.find? (·.name == n)).getD default).name = n := by
  cases h : l.find? (·.name == n) with
  | none =>
    rw [List.find?_eq_none] at h
    exact absurd (by simp [hn]) (h pt hpt)
  | some a =>
    have := List.find?_some h
    simpa using this

/-- **The `ok` tie**: where flock-verify's known-content check accepted `own` at `known`, which names the port `ok` with
rows each the word `1`, every `ok` wire of unit `u` is at a registrant's row that is the `ok` row, unless the `ok` tie
collides. -/
theorem tie_ok (s : RowsSession) (ok : String) {known : Array (String × Array (Array ℕ × ByteArray))}
    {ks : Array (Array ℕ × ByteArray)} (hk : Flock.Registered.checkKnownAll s.own known = .ok ())
    (hm : (ok, ks) ∈ known) (hks : ∀ r ∈ ks, r.1 = okRow) (hS : ∀ u, s.ServedRegistered u)
    (hcol : ¬ OkColl s ok ks) {u : Fin s.n} {g : Fin s.Circ.N} (hw : OkWire s ok u g) :
    OkVal s (s.vb.pos g) ((s.rdU u).wv (s.vb.pos g)) := by
  obtain ⟨-, hr, hn⟩ := hw
  obtain ⟨reg, hf, hck⟩ := FlockC3.RegLevels.checkKnownAll_ok hk hm
  obtain ⟨q, hpq, hex⟩ := ((hS u).1 _).1 hr
  have hport : (s.rdU u).port (s.vb.pos g) = s.ownPort q := by
    rw [hpq]; exact ((hS u).2 q hex).1
  have hqn : s.c.ports[q].name = ok := by
    obtain ⟨pt, hpt, hpn⟩ := hex
    have e := find_name_eq (l := s.own.toList) (Array.mem_toList_iff.2 hpt) hpn
    rw [← hn, hport]
    exact e.symm
  have hreg : s.ownPort q = reg := by
    show (s.own.toList.find? (·.name == s.c.ports[q].name)).getD default = reg
    rw [hqn, hf]
    rfl
  have ho := (s.rdU u).committed _ hr
  rw [hport, hreg] at ho
  obtain ⟨hi, hb | ⟨a, b, hc, hab, hh⟩⟩ := FlockC3.RegLevels.knownColl_spec
    (r := (rsOf s).enc (s.vb.pos g) ((s.rdU u).wv (s.vb.pos g))) (y := (s.rdU u).ws (s.vb.pos g)) hck ho
  · have hmem : ks[(s.rdU u).readAt (s.vb.pos g)]! ∈ ks := by
      rw [getElem!_pos ks _ hi]
      exact Array.getElem_mem hi
    unfold OkVal
    rw [hb, hks _ hmem]
  · exact (hcol ⟨u, _, hr, hn, a, b, by rw [hport, hreg]; exact hc, hab, hh⟩).elim

/-- **An accepted proof's rows that meet `CheckedRel` meet `HeldRel`**: at a session whose known-content check passed,
naming `ok` with rows each the word `1`, whose served roots are registered and whose key program realizes `Cd`. -/
theorem held_of_checked {ι : Type} (s : RowsSession) (o : s.Out) (ok : String)
    (known : Array (String × Array (Array ℕ × ByteArray))) {ks : Array (Array ℕ × ByteArray)}
    {Cd : Accounting.Compose.Declared ι} {cls : ℕ → Accounting.Served.Cls} {unitOf : Fin s.n → ℕ}
    {dec : (Fin s.Circ.N → Bool) → ι → ℤ}
    (hk : Flock.Registered.checkKnownAll s.own known = .ok ()) (hm : (ok, ks) ∈ known)
    (hks : ∀ r ∈ ks, r.1 = okRow) (hS : ∀ u, s.ServedRegistered u)
    (hR : Accounting.Assumptions.TS.OkRealizes s ok Cd cls unitOf dec)
    (hc : CheckedRel s o ok (openedReg s)) : HeldRel s o ok Cd cls unitOf dec known (openedReg s) := by
  refine ⟨hc, ?_⟩
  by_cases hcol : OkColl s ok ks
  · exact Or.inr ⟨ks, hm, hcol⟩
  obtain ⟨-, vals, hv, hagree⟩ := hc
  refine Or.inl ⟨vals, hv, hagree, fun u hu => hR vals u (hagree u hu) fun g hw => ?_⟩
  refine ⟨_, tie_ok s ok hk hm hks hS hcol hw, ?_⟩
  obtain ⟨hg, hr, -⟩ := hw
  have e := congrFun (congrFun hv u.val) g.val
  simp only [openedReg, rowsReg] at e
  rw [regRead_at s _ hg hr, regRead_at s _ hg hr] at e
  exact (Option.some.inj e).symm

end Accounting.TS.Held

open Accounting.TS Accounting.TS.Checked Accounting.TS.Held in
theorem Accounting.SecurityProofs.TS.HeldSessionSound : Accounting.Guarantees.TS.HeldSessionSound := by
  intro ι Ω _ S nπ sess out ok known Cd cls unitOf dec right δ hCR hLaw hk hS hok hKn hR hδ hδ0
  refine ⟨fun _ _ _ h => h.1, fun w i => ?_⟩
  have hC := (Accounting.SecurityProofs.TS.CheckedSessionSound sess out ok right δ hCR hLaw hk hS hok hδ
    hδ0).2.2.2
  have hsub : ∀ x, (heldLayer sess out ok known Cd cls unitOf dec right).Unsound x w i →
      (checkedLayer sess out ok right).Unsound x w i := by
    intro x hx
    obtain ⟨⟨hacc, hkn⟩, hnot⟩ := hx
    refine ⟨hacc, fun hc => hnot ?_⟩
    obtain ⟨ks, hm, hks⟩ := hKn x w i
    exact held_of_checked (sess x w i) (out x w i) ok (known x w i) hkn hm hks (hS x w i) (hR x w i) hc
  exact le_trans (Pouw.RelativeSecurity.prob_mono hsub) (hC w i)
