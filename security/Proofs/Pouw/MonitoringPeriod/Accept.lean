module

public import Proofs.Pouw.MonitoringPeriod.Sort
public import Proofs.Properties.Pouw.Check
public import Proofs.Properties.Pouw.Sample
public import Properties.Pouw.Check
public import Properties.Pouw.MonitoringPeriod.Witnesses
import all Init.Data.Array.Basic
import all Init.Data.Array.DecidableEq
import all Init.Data.Ord.String
import all Lean.Data.Json.Basic

/-! PoUW's deployed check accepts H1's honest window at the subprotocol's parameters. -/

@[expose] public section

namespace Properties.Pouw.MonitoringPeriod.SecurityProofs.Accept

open PouwCheck
open Lean (Json)
open PouwWindow (Work Sampling)
open Properties.Pouw.MonitoringPeriod.SecurityProofs.Sort

theorem keys_of {kvs : Std.TreeMap.Raw String Json compare} {l : List String} (hs : l.Pairwise (· < ·))
    (h : kvs.toArray.map (·.1) = l.toArray) : keys (.obj kvs) = some l := by
  show some _ = _
  rw [h]
  exact congrArg some (qsort_of_sorted (as := l.toArray) hs)

theorem sortedKeys_of {kvs : Std.TreeMap.Raw String Json compare} {l : List String} (hs : l.Pairwise (· < ·))
    (h : kvs.toArray.map (·.1) = l.toArray) : PouwWindow.sortedKeys (.obj kvs) = some l := by
  show some _ = _
  rw [h]
  exact congrArg some (qsort_of_sorted (as := l.toArray) hs)

theorem registeredSorted_eq : registeredSorted = ["check", "length", "root_W", "sampling", "work"] :=
  (qsort_eq_of_perm (as := PouwWindow.registered.toArray) (bs := #["check", "length", "root_W", "sampling", "work"])
    (by decide)).trans (qsort_of_sorted (as := #["check", "length", "root_W", "sampling", "work"]) (by decide))

theorem keys_h1Items : keys h1Items = some registeredSorted := by
  rw [registeredSorted_eq]
  exact keys_of (by decide) (by decide +kernel)

theorem length_h1 : PouwWindow.lengthProblem (get h1Items "length") [2, 2, 2, 2] = none := by
  have hg : get h1Items "length" = PouwWindow.lengthItem [2, 2, 2, 2] := rfl
  have hs : PouwWindow.sortedKeys (PouwWindow.lengthItem [2, 2, 2, 2]) = some PouwWindow.lengthKeys :=
    sortedKeys_of (by decide) (by decide +kernel)
  rw [hg]
  unfold PouwWindow.lengthProblem
  rw [hs]
  decide +kernel

theorem cs_check_h1 : cs (get h1Items "check") = cs h1Check.toJson := by
  first
  | rw [show get h1Items "check" = h1Check.toJson from rfl]
  | decide +kernel

theorem cs_work_h1 : cs (get h1Items "work") = cs (h1Fixed.work.itemJson h1Check.tileWork) := by
  first
  | rw [show get h1Items "work" = h1Fixed.work.itemJson h1Check.tileWork from rfl]
  | decide +kernel

theorem level_h1 : levelOf h1Items = some 39 := by decide

theorem grid_h1 : 39 + 1 ≤ PouwWindow.level PouwWindow.gridBits (PouwWindow.floorPoint PouwWindow.gridBits 65536) ∧
    1 ≤ PouwWindow.point PouwWindow.gridBits 39 := by decide

theorem work_h1 : Work.at 3 39 = ⟨39, 128, 144⟩ := by decide

theorem kPrime_h1 (E D : Rat) (C N W : Int) (hE : E = 1 / 2) (hD : D = 1 / 1000000) (hC : C = 128 * 2048)
    (hN : N = 144) (hW : W = 2048) : PouwWindow.kPrime E D C N W = .ok 24 := by
  subst hE hD hC hN hW
  have hd : PouwWindow.KPrimeDomain (1 / 2) (1 / 1000000) (128 * 2048) 144 2048 := by
    unfold PouwWindow.KPrimeDomain; decide +kernel
  have h := Properties.Pouw.Window.kPrime_least _ _ _ _ _ hd
  dsimp only at h
  obtain ⟨k, hk, h1, -, hlt, hmin⟩ := h
  have hbig : ∀ j, j < 24 → 1 ≤ j →
      ¬ (1 - PouwWindow.missShare (1 / 2) (128 * 2048) 144 2048) ^ j ≤ (1 / 1000000 : Rat) := by
    decide +kernel
  rcases Nat.lt_trichotomy k 24 with h | rfl | h
  · exact absurd (hlt (Nat.lt_of_lt_of_le h (by decide))) (hbig k h h1)
  · exact hk
  · exact absurd (hmin 24 (by decide) h) (by decide +kernel)

theorem sampling_h1 : PouwWindow.sampling (toRat ⟨1, 2⟩) (toRat ⟨1, 1000000⟩) (Work.at 3 39) (h1Check.tileWork : Int)
    PouwWindow.gridBits = .ok h1Fixed.sampling := by
  unfold PouwWindow.sampling
  rw [kPrime_h1] <;> first | decide +kernel | rfl

theorem keeps_h1 (context format program partition : String) (bytes : ByteArray) :
    Properties.Pouw.Check.RecordKeeps (h1Params context) (h1Reg context format program partition bytes)
      h1Fixed := by
  refine ⟨⟨rfl, rfl⟩, congrArg (fun o => o.getD []) keys_h1Items, length_h1, level_h1, grid_h1.1, grid_h1.2, rfl,
    sampling_h1, rfl, ?_, rfl, rfl, rfl, rfl⟩
  intro kv hkv
  simp only [expected, List.mem_cons] at hkv
  rcases hkv with rfl | rfl | rfl | rfl | h
  · exact cs_check_h1
  · rfl
  · rfl
  · exact cs_work_h1
  · cases h

theorem saltKept_h1 (context : String) (r c : ExecutionPhaseCommitmentRecord) (units : Array Nat) :
    Properties.Pouw.Check.SaltKept (h1Params context) r (h1Evidence c units) := rfl

theorem check_h1 (context format program partition : String) (bytes : ByteArray) (units : Array Nat)
    (hu : UnitsOk units 24 144) :
    protocol.check (h1Params context) (h1Reg context format program partition bytes)
        (h1Evidence (h1Reg context format program partition bytes) units) =
      .accept ⟨h1Check, Work.at 3 39, h1Fixed.sampling, [2, 2, 2, 2], none, units⟩ (h1Profile program partition) :=
  Properties.Pouw.Check.acceptMeans_complete _ _ _ h1Fixed _ { samplingDesign := .subset 24, population := 144, units }
    (keeps_h1 context format program partition bytes)
    ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, hu.1, hu.2.1, hu.2.2⟩ (saltKept_h1 _ _ _ _)

theorem numbers_h1 (context : String) (c : ExecutionPhaseCommitmentRecord) (e : Evidence) (pub : Public) (prof : Core.Integrity)
    (hc : protocol.check (h1Params context) c e = .accept pub prof) (hl : levelOf c.window.items = some 39) :
    pub.work.n = 144 ∧ pub.sampling.k = 24 := by
  obtain ⟨f, draw, hk, -, -, rfl⟩ := Properties.Pouw.Check.acceptMeans _ _ _ _ _ hc
  obtain ⟨-, -, -, hlv, -, -, hw, hs, -⟩ := hk
  have h39 : f.work.level = 39 := Option.some.inj (hlv.symm.trans hl)
  have hw' : f.work = Work.at 3 39 := by rw [hw, h39]; rfl
  have h2 : PouwWindow.sampling (toRat (h1Params context).eps) (toRat (h1Params context).delta) f.work
      (h1Params context).check.tileWork (h1Params context).gridBits = .ok h1Fixed.sampling := by
    rw [hw']; exact sampling_h1
  have hs' : f.sampling = h1Fixed.sampling := Except.ok.inj (hs.symm.trans h2)
  exact ⟨by rw [hw']; rfl, by rw [hs']; rfl⟩

end Properties.Pouw.MonitoringPeriod.SecurityProofs.Accept
