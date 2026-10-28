import Proofs.Pouw.Period.Sort
import Proofs.Properties.Pouw.Check
import Proofs.Properties.Pouw.Draw

/-!
# PoUW's deployed check accepts H1's honest window at the leg's parameters

The all-protocols PoUW leg (`benchmarks/all_protocols/legs/pouw.py`) checks its H1 window at `h1_params`: the
pinned H1 tile's check item with the entries bound 1024 and no coin mode, ε = 1/2, δ = 10⁻⁶, the 3-bit grid, the range
`[1, 2^16]` and the loop's fixed length, 4 requests of 2 decode steps (`h1Params`). A window registering the loop's
128 real tiles is at level 39, which pads to `N = 144`, and its draw is `K′ = 24` (`kPrime_h1`:
`q = 1 − (1/2)·128/144 = 5/9`, `q^23 > 10⁻⁶ ≥ q^24`).

`check_h1`: `protocol.check` accepts the honest registration of that window (`h1Reg`: its items the verifier's own,
`N = 144` drawn `subset:24`) with any service answer citing it that accepted 24 ascending distinct units below 144 at
the profile the rule fixes (`h1Evidence`), and publishes level 39's work, `K′ = 24` and those units. `numbers_h1`: every
window the check accepts at those parameters, registered at level 39, publishes `N = 144` and `K′ = 24`, the population
and draw size `DrawIsCoin` reads at H1's instance.
-/

namespace Properties.Pouw.Period.SecurityProofs.Accept

open PouwCheck
open Lean (Json)
open PouwWindow (Work Sampling)
open Properties.Pouw.Period.SecurityProofs.Sort

/-- The pinned H1 tile (`core_check.TILES`' salted `PouwH1Tile_v1`, `K = 64`, `R = 4`, `C = 8`). -/
def h1Tile : Tile := tiles[4]

/-- The leg's check item (`h1_check`): the pinned tile's template and work per tile (`64·32`), the scheme and the
entries bound 1024, no coin mode. -/
def h1Check : CheckItem :=
  { entries := 1024, scheme := PouwCheck.scheme, template := h1Tile.template, tileWork := h1Tile.tileWork }

/-- The leg's check parameters (`h1_params`) in the period's context `context`. -/
def h1Params (context : String) : Params where
  check := h1Check
  context := context
  delta := ⟨1, 1000000⟩
  eps := ⟨1, 2⟩
  gridBits := PouwWindow.gridBits
  query := h1Tile.query
  lo := 1
  hi := 65536
  maxTokens := [2, 2, 2, 2]

/-- What the loop's honest registration fixes: level 39 (128 credited tiles, padded to 144), the sampling item at
`K′ = 24`, and no `W` root. -/
def h1Fixed : Fixed := ⟨Work.at 3 39, ⟨toRat ⟨1, 2⟩, toRat ⟨1, 1000000⟩, 3, 24⟩, none⟩

/-- Its registered items: the verifier's own at level 39 and the fixed length's. -/
def h1ItemList : List (String × Json) :=
  [("check", h1Check.toJson), ("length", PouwWindow.lengthItem [2, 2, 2, 2]), ("root_W", rootWJson none),
   ("sampling", h1Fixed.sampling.toJson), ("work", h1Fixed.work.itemJson h1Check.tileWork)]

def h1Items : Json := Json.mkObj h1ItemList

/-- The loop's honest registration in context `context`, for any record bytes, format, Program and partition: a served
window registering `h1Items`, the population 144 drawn `subset:24`, the check's query and scheme, and no roots. -/
def h1Reg (context format program partition : String) (bytes : ByteArray) : Registration where
  bytes := bytes
  format := format
  program := program
  query := h1Tile.query
  partition := partition
  scheme := PouwCheck.scheme
  population := 144
  law := subsetLaw 24
  roots := #[]
  window := ⟨"served", context, h1Items, none, none⟩

/-- The profile the rule fixes for it: the registered Program over one level, the partition's 144 units drawn
`subset:24`, at δ = 10⁻⁶, 24 drawn. -/
def h1Profile (program partition : String) : Core.Integrity where
  programs := #[program]
  prescribed := #[]
  linked := true
  covered := #[]
  levels := #[{ name := unitLevel, query := some partition, size := 144, draw := some (.subset 24) }]
  delta := ⟨1, 1000000⟩
  drawn := #[(unitLevel, 24)]

/-- One-stage's answer accepting registration `c` with the rule's profile and a `subset:24` draw of `units` over 144. -/
def h1Evidence (c : Registration) (units : Array Nat) : Evidence where
  protocol := oneStage
  commitment := some (Flock.hexOf c.digest)
  said := .accept (h1Profile c.program c.partition) { law := .subset 24, population := 144, units }

/-! ## The registration's keys -/

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

/-! ## The draw: `K′ = 24` -/

/-- **`K′ = 24` at the leg's ε, δ and level**: `p = (1/2)·(128·2048)/(144·2048) = 4/9`, so `q = 5/9`, and
`q^23 ≈ 1.34·10⁻⁶ > 10⁻⁶ ≥ q^24 ≈ 7.45·10⁻⁷`. -/
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

/-! ## The check accepts -/

/-- **The loop's honest registration keeps PoUW's rule** at the leg's parameters, fixing level 39 and `K′ = 24`. -/
theorem keeps_h1 (context format program partition : String) (bytes : ByteArray) :
    Properties.Pouw.Check.RegistrationKeeps (h1Params context) (h1Reg context format program partition bytes)
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

/-- **The window's salt is kept**: the leg's check item names no coin mode, so the window is seeded, and the honest
answer carries no salt. -/
theorem saltKept_h1 (context : String) (r c : Registration) (units : Array Nat) :
    Properties.Pouw.Check.SaltKept (h1Params context) r (h1Evidence c units) := rfl

/-- **The deployed check accepts H1's honest window at the leg's parameters**: on the loop's honest registration and
any one-stage answer citing it that accepted 24 ascending distinct units below 144 at the rule's profile, with no salt
(the window is seeded), it publishes level 39's work (`N = 144`), `K′ = 24` and those units. -/
theorem check_h1 (context format program partition : String) (bytes : ByteArray) (units : Array Nat)
    (hu : UnitsOk units 24 144) :
    protocol.check (h1Params context) (h1Reg context format program partition bytes)
        (h1Evidence (h1Reg context format program partition bytes) units) =
      .accept ⟨h1Check, Work.at 3 39, h1Fixed.sampling, [2, 2, 2, 2], none, units⟩ (h1Profile program partition) :=
  Properties.Pouw.Check.acceptMeans_complete _ _ _ h1Fixed _ { law := .subset 24, population := 144, units }
    (keeps_h1 context format program partition bytes)
    ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, hu.1, hu.2.1, hu.2.2⟩ (saltKept_h1 _ _ _ _)

/-- **`DrawIsCoin`'s numbers at the leg's parameters**: every window the check accepts at them, registered at level 39
(the loop's 128 real tiles), publishes the population `N = 144` and the draw size `K′ = 24`. -/
theorem numbers_h1 (context : String) (c : Registration) (e : Evidence) (pub : Public) (prof : Core.Integrity)
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

end Properties.Pouw.Period.SecurityProofs.Accept
