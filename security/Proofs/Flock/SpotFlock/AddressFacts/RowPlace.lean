module

public import Proofs.Flock.SpotFlock.AddressFacts.RowTable
public import Proofs.Flock.SpotFlock.AddressFacts.RowLeaves
public import Proofs.Flock.SpotFlock.AddressFacts.RowInstance
public import Proofs.Flock.SpotFlock.AddressFacts.RowOf
public import Proofs.Flock.SpotFlock.EntryRoots
public import Proofs.Flock.SpotFlock.FrameAt
public import Proofs.Flock.SpotFlock.UnitEmbeds
public import Proofs.Flock.SpotFlock.Placement
public import Proofs.Flock.SpotFlock.Facts
public import Proofs.Flock.SamplingGeneric.TwoCode
public import Proofs.Flock.Claims.Sampling.Select.SecondStrata
public import Proofs.Flock.Bridge.Members

/-! Where the second stage puts a checked inner block (`sits_of_slots`), given no inner block always checked and
`SampleSlots`, a premise because the design doesn't ask that the first stage sample only slots that name
themselves. -/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy
open Properties.Spot (cellOff unitBase filesBlockOf)
open Lean (Json)

namespace AddressRow

theorem sum_map_eq_zero (f : ℕ → ℕ) : ∀ l : List ℕ, (∀ x ∈ l, f x = 0) → (l.map f).sum = 0
  | [], _ => rfl
  | a :: l, hl => by
    rw [List.map_cons, List.sum_cons, hl a (List.mem_cons.2 (Or.inl rfl)),
      sum_map_eq_zero f l fun x hx => hl x (List.mem_cons.2 (Or.inr hx)), Nat.zero_add]

theorem eq_zero_of_sum_eq_zero : ∀ l : List ℕ, l.sum = 0 → ∀ x ∈ l, x = 0
  | [], _, _, hx => (List.not_mem_nil hx).elim
  | a :: l, hs, x, hx => by
    rw [List.sum_cons] at hs
    rcases List.mem_cons.1 hx with rfl | hx
    · omega
    · exact eq_zero_of_sum_eq_zero l (by omega) x hx

theorem cell_div_mod {n i p : ℕ} (hp : p < n) : (i * n + p) / n = i ∧ (i * n + p) % n = p := by
  have hn : 0 < n := by omega
  constructor
  · rw [Nat.add_comm, Nat.add_mul_div_right _ _ hn, Nat.div_eq_of_lt hp, Nat.zero_add]
  · rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hp]

theorem cell_split {n a p a' p' : ℕ} (hp : p < n) (hp' : p' < n) (he : a * n + p = a' * n + p') :
    a = a' ∧ p = p' := by
  have h1 := (cell_div_mod (i := a) hp).1
  have h2 := (cell_div_mod (i := a) hp).2
  rw [he] at h1 h2
  exact ⟨h1.symm.trans (cell_div_mod hp').1, h2.symm.trans (cell_div_mod hp').2⟩

theorem arr_getD_eq {a : Array ℕ} {k : ℕ} (hk : k < a.size) : a.getD k 0 = a[k] := by simp [hk]

theorem arr_getD_mem {a : Array ℕ} {k : ℕ} (hk : k < a.size) : a.getD k 0 ∈ a := by
  rw [arr_getD_eq hk]
  exact Array.getElem_mem hk

theorem arr_getElem?_getD {a : Array ℕ} {k : ℕ} (hk : k < a.size) : a[k]? = some (a.getD k 0) := by
  rw [arr_getD_eq hk]
  exact Array.getElem?_eq_getElem hk

theorem arr_findIdx {a : Array ℕ} {f : ℕ → Bool} (hex : ∃ x ∈ a.toList, f x = true) :
    a.toList.findIdx f < a.size ∧ a.getD (a.toList.findIdx f) 0 ∈ a ∧
      f (a.getD (a.toList.findIdx f) 0) = true := by
  have hw : a.toList.findIdx f < a.toList.length := List.findIdx_lt_length_of_exists hex
  have hk : a.toList.findIdx f < a.size := Nat.lt_of_lt_of_eq hw Array.length_toList
  have hg : a.getD (a.toList.findIdx f) 0 = a.toList[a.toList.findIdx f]'hw := (arr_getD_eq hk).trans rfl
  refine ⟨hk, arr_getD_mem hk, ?_⟩
  rw [hg]
  exact List.findIdx_getElem

theorem arr_idxOf_getD {a : Array ℕ} (hnd : a.toList.Nodup) {i : ℕ} (hi : i < a.size) :
    a.toList.idxOf (a.getD i 0) = i := by
  have hl : i < a.toList.length := Nat.lt_of_lt_of_eq hi Array.length_toList.symm
  have hg : a.getD i 0 = a.toList[i]'hl := (arr_getD_eq hi).trans rfl
  rw [hg]
  exact hnd.idxOf_getElem i hl

theorem mem_drawn {proved : Array ℤ} {P : Spot.SecondStage.Part} {q : ℕ} {A : Array ℕ}
    (hA : A = ((proved.map Int.toNat).filter (P.holds q)).map (· - P.lo)) {x : ℕ} :
    x ∈ A ↔ ∃ j ∈ proved, P.lo ≤ j.toNat ∧ j.toNat < P.lo + q * P.n ∧ x = j.toNat - P.lo := by
  subst hA
  constructor
  · intro hx
    obtain ⟨y, hy, rfl⟩ := Array.mem_map.1 hx
    obtain ⟨hy1, hy2⟩ := Array.mem_filter.1 hy
    obtain ⟨j, hj, rfl⟩ := Array.mem_map.1 hy1
    simp only [Spot.SecondStage.Part.holds, Bool.and_eq_true, decide_eq_true_eq] at hy2
    exact ⟨j, hj, hy2.1, hy2.2, rfl⟩
  · rintro ⟨j, hj, h1, h2, rfl⟩
    refine Array.mem_map.2 ⟨j.toNat, Array.mem_filter.2 ⟨Array.mem_map.2 ⟨j, hj, rfl⟩, ?_⟩, rfl⟩
    simp only [Spot.SecondStage.Part.holds, Bool.and_eq_true, decide_eq_true_eq]
    exact ⟨h1, h2⟩

open Classical in
noncomputable def offCells (q : ℕ) (ns : Array ℕ) (x : ℕ) : ℕ :=
  if ∃ t i p, t < ns.size ∧ i < q ∧ p < ns.getD t 0 ∧ x = cellOff q ns t + i * ns.getD t 0 + p then 0 else 1

theorem proved_in_cell {sample ns ks : Array ℕ} {src : ByteArray} {draw : Json} {read : ℕ}
    (hd : Spot.Select.selectSecond sample ns ks src = .ok (draw, read)) {j : ℤ}
    (hj : j ∈ Spot.Draw.proved draw) :
    ∃ t i p, t < ns.size ∧ i < sample.size ∧ p < ns.getD t 0 ∧
      j.toNat = cellOff sample.size ns t + i * ns.getD t 0 + p := by
  have hs : ((Spot.Draw.proved draw).toList.map fun j => offCells sample.size ns j.toNat).sum ≤
      sample.size * ((List.range ns.size).map fun t => ks.getD t 0 * 0).sum :=
    Properties.Spot.SecurityProofs.sum_proved_selectSecond_le hd (offCells sample.size ns) (fun _ => 0)
      (fun t i p ht hi hp => by
        have h0 : offCells sample.size ns (cellOff sample.size ns t + i * ns.getD t 0 + p) = 0 := by
          unfold offCells
          exact if_pos ⟨t, i, p, ht, hi, hp, rfl⟩
        exact Nat.le_of_eq h0)
  have hz : ((List.range ns.size).map fun t => ks.getD t 0 * 0).sum = 0 :=
    sum_map_eq_zero (fun t => ks.getD t 0 * 0) _ fun x _ => Nat.mul_zero _
  rw [hz, Nat.mul_zero] at hs
  have hw : offCells sample.size ns j.toNat = 0 :=
    eq_zero_of_sum_eq_zero _ (Nat.eq_zero_of_le_zero hs) _
      (List.mem_map.2 ⟨j, Array.mem_toList_iff.2 hj, rfl⟩)
  by_contra hc
  have h1 : offCells sample.size ns j.toNat = 1 := by
    unfold offCells
    exact if_neg hc
  omega

def SampleSlots (h : Held) : Prop :=
  ∀ l, Spot.Slots.layout h.reg.first.program h.reg.first.query = .ok l →
    ∀ c : ByteArray, ∀ ρ ∈ h.reg.sample c, ∃ b, l.blocks[ρ]? = some b ∧ b.unit = some ρ

variable {h : Held}

theorem blockAt_slot {s : AuditStatement} {d : Design.Two} (h2 : two h s = some d) (hSS : SampleSlots h)
    (c : ByteArray) {t i p : ℕ} (ht : t < h.ms.length) (hi : i < (h.reg.sample c).size)
    (hp : p < h.reg.ns.getD t 0) :
    h.reg.blockAt (h.reg.sample c) (cellOff (h.reg.sample c).size h.reg.ns t + i * h.reg.ns.getD t 0 + p) =
        unitBase h.ms t + ((h.reg.sample c).getD i 0 * h.reg.ns.getD t 0 + p) ∧
      (h.reg.sample c).getD i 0 * h.reg.ns.getD t 0 + p < h.ms[t].2.size := by
  obtain ⟨l, hl, -, -, -, hfo⟩ := two_eq h h2
  obtain ⟨hT, -, hfit, hblk⟩ := filesOk_sound hfo
  obtain ⟨b, hb, hbu⟩ := hSS l hl c _ (arr_getD_mem hi)
  have htn : t < h.reg.ns.size := by omega
  refine ⟨?_, ?_⟩
  · exact ((RegTwo.blockAt_cell h.reg (h.reg.sample c) htn hi hp).trans (hblk _ b hb hbu t ht p hp)).trans rfl
  · exact lt_of_lt_of_le (_root_.Flock.Bridge.Members.rank_mul_add_lt (Nat.lt_add_one _) hp)
      ((hfit _ b hb _ hbu).2 t ht)

theorem unit_inj {t t' a a' p p' : ℕ} (ht : t < h.ms.length) (ht' : t' < h.ms.length)
    (hp : p < h.reg.ns.getD t 0) (hp' : p' < h.reg.ns.getD t' 0)
    (hk : a * h.reg.ns.getD t 0 + p < h.ms[t].2.size) (hk' : a' * h.reg.ns.getD t' 0 + p' < h.ms[t'].2.size)
    (he : unitBase h.ms t + (a * h.reg.ns.getD t 0 + p) = unitBase h.ms t' + (a' * h.reg.ns.getD t' 0 + p')) :
    t = t' ∧ a = a' ∧ p = p' := by
  have e : t = t' := by
    have h1 := templateOf_unitBase ht hk
    rw [he, templateOf_unitBase ht' hk'] at h1
    exact h1.symm
  subst e
  exact ⟨rfl, cell_split hp hp' (Nat.add_left_cancel he)⟩

theorem cell_of_unit {t a p : ℕ} (ht : t < h.ms.length) (hp : p < h.reg.ns.getD t 0)
    (hk : a * h.reg.ns.getD t 0 + p < h.ms[t].2.size) :
    templateOf h.ms (unitBase h.ms t + (a * h.reg.ns.getD t 0 + p)) = t ∧
      cellBlock h (unitBase h.ms t + (a * h.reg.ns.getD t 0 + p)) = a ∧
      cellPos h (unitBase h.ms t + (a * h.reg.ns.getD t 0 + p)) = p := by
  have e := templateOf_unitBase ht hk
  refine ⟨e, ?_, ?_⟩
  · unfold cellBlock
    rw [e, Nat.add_sub_cancel_left]
    exact (cell_div_mod hp).1
  · unfold cellPos
    rw [e, Nat.add_sub_cancel_left]
    exact (cell_div_mod hp).2

theorem rankAt_eq {m₁ xS : Msg} {u : ℕ} {S : Array ℕ} {l : List ℕ}
    (hS : (((jsonMsg m₁).map blocksOf).getD []).toArray = S)
    (hl : drawnOf ((FlockDeploy.readOf true xS).bind (·.draw)) = l) :
    rankAt h [] m₁ xS u =
      l.findIdx fun p => h.reg.blockAt S (cellOff S.size h.reg.ns (templateOf h.ms u) + p) == u := by
  subst hS
  subst hl
  rfl

end AddressRow

namespace CheckedRun

open AddressRow

variable {h : Held} {r : CheckedRun h.stage}

theorem n_eq_findIdx (hr : r.Ok h) {dj : Json} {D : _root_.Flock.Draw.UnitSample} (hdj : r.ins.read.draw = some dj)
    (hD : _root_.Flock.Draw.UnitSample.ofJson dj = .ok D) :
    r.n h = D.proved.toList.findIdx fun p => h.reg.blockAt (sampleOf h r) ((partOf h r).lo + p) == r.u := by
  obtain ⟨st, rec, draw, hR⟩ := reads_of_ok hr
  have hbl := blocks_of_ok hr hR
  obtain ⟨Pt, d, spec, hx, hP, h2, hin, hok, hu, hck, hb, hρ, hrb, hset⟩ := hr
  obtain ⟨hins, -⟩ := FlockDeploy.setupOf_some hset
  have hread : FlockDeploy.readOf true (unitStmt h r.m₂ r.u) = some r.ins.read := by
    simp [FlockDeploy.readOf, hins]
  have hS : (((jsonMsg r.m₁).map blocksOf).getD []).toArray = sampleOf h r := by
    simp only [hR.msg, Option.map_some, Option.getD_some, hbl, Array.toArray_toList]
  have hl : drawnOf ((FlockDeploy.readOf true (unitStmt h r.m₂ r.u)).bind (·.draw)) = D.proved.toList := by
    simp only [hread, Option.bind_some, hdj, drawnOf, hD]
  exact rankAt_eq hS hl

end CheckedRun

theorem AddressRow.sits_of_slots {h : Held} (hna : ∀ v, h.reg.always v = false) (hSS : AddressRow.SampleSlots h) :
    AddressRow.Sits h := by
  intro r hr
  have hlab := CheckedRun.labelled_of_ok hr
  obtain ⟨dj, D, draw, hdj, hD, hdraw, hn, hprov, -⟩ := CheckedRun.sampled_of_ok hr
  obtain ⟨st, rec, draw', hR⟩ := CheckedRun.reads_of_ok hr
  have hbl := CheckedRun.blocks_of_ok hr hR
  have hrank := CheckedRun.n_eq_findIdx hr hdj hD
  have hunit : unitOf r.ins.read.draw (r.n h) = D.proved.getD (r.n h) 0 := by
    simp only [unitOf, hdj, hD]
  obtain ⟨Pt, d, spec, hx, hP, h2, hin, hok, hu, hck, hb, hρ, hrb, hset⟩ := hr
  obtain ⟨read, hsel0⟩ := AddressRow.select_of_secondSample h hdraw
  have hsel : Spot.Select.selectSecond (sampleOf h r) h.reg.ns h.reg.ks (stageSrc r.c₁) = .ok (draw, read) :=
    hsel0
  obtain ⟨-, hnd⟩ := selectSecond_sample hsel
  obtain ⟨l, -, -, -, -, hfo⟩ := two_eq h h2
  obtain ⟨hT, -, -, -⟩ := filesOk_sound hfo
  obtain ⟨t, ht, p, hp, hk, hue⟩ := member_of_replayOf h hP h2 hin (List.getElem?_eq_some_iff.1 hb).1 hρ
  obtain ⟨ht₀, hcb, hcp⟩ := AddressRow.cell_of_unit (h := h) ht hp hk
  rw [← hue] at ht₀ hcb hcp
  have hlo : (partOf h r).lo = cellOff (sampleOf h r).size h.reg.ns t := by
    show cellOff (sampleOf h r).size h.reg.ns (templateOf h.ms r.u) = _
    rw [ht₀]
  have hPn : (partOf h r).n = h.reg.ns.getD t 0 := by
    show h.reg.ns.getD (templateOf h.ms r.u) 0 = _
    rw [ht₀]
  have hcell : ∀ i, i < (sampleOf h r).size →
      i * h.reg.ns.getD t 0 + p < (sampleOf h r).size * (partOf h r).n := by
    intro i hi
    rw [hPn]
    exact _root_.Flock.Bridge.Members.rank_mul_add_lt hi hp
  have key : ∀ j : ℤ, j ∈ Spot.Draw.proved draw → h.reg.blockAt (sampleOf h r) j.toNat = r.u →
      ∃ i, i < (sampleOf h r).size ∧ (sampleOf h r).getD i 0 = r.ρ ∧
        j.toNat = (partOf h r).lo + (i * h.reg.ns.getD t 0 + p) := by
    intro j hj hju
    obtain ⟨t', i, p', ht', hi, hp', hje⟩ := AddressRow.proved_in_cell hsel hj
    have ht'' : t' < h.ms.length := by omega
    have hA : h.reg.blockAt (sampleOf h r)
          (cellOff (sampleOf h r).size h.reg.ns t' + i * h.reg.ns.getD t' 0 + p') =
          unitBase h.ms t' + ((sampleOf h r).getD i 0 * h.reg.ns.getD t' 0 + p') ∧
        (sampleOf h r).getD i 0 * h.reg.ns.getD t' 0 + p' < h.ms[t'].2.size :=
      AddressRow.blockAt_slot h2 hSS (src r.c₀) ht'' hi hp'
    have he : unitBase h.ms t + (r.ρ * h.reg.ns.getD t 0 + p) =
        unitBase h.ms t' + ((sampleOf h r).getD i 0 * h.reg.ns.getD t' 0 + p') :=
      hue.symm.trans (hju.symm.trans ((congrArg (h.reg.blockAt (sampleOf h r)) hje).trans hA.1))
    obtain ⟨e1, e2, e3⟩ := AddressRow.unit_inj (h := h) ht ht'' hp hp' hk hA.2 he
    rw [← e1, ← e3] at hje
    refine ⟨i, hi, e2.symm, ?_⟩
    rw [hlo]
    omega
  obtain ⟨j, hj, hju⟩ : ∃ j ∈ Spot.Draw.proved draw, h.reg.blockAt (sampleOf h r) j.toNat = r.u :=
    AddressRow.proved_of_checked hna hu hsel0
  obtain ⟨i, hi, -, hje⟩ := key j hj hju
  have hfind : ∃ x ∈ D.proved.toList,
      (h.reg.blockAt (sampleOf h r) ((partOf h r).lo + x) == r.u) = true := by
    have hci := hcell i hi
    refine ⟨j.toNat - (partOf h r).lo, Array.mem_toList_iff.2 ?_, ?_⟩
    · exact (AddressRow.mem_drawn hprov).2 ⟨j, hj, by omega, by omega, rfl⟩
    · have e : (partOf h r).lo + (j.toNat - (partOf h r).lo) = j.toNat := by omega
      rw [e]
      exact beq_iff_eq.2 hju
  obtain ⟨hlt, hmem, hpred⟩ : r.n h < D.proved.size ∧ D.proved.getD (r.n h) 0 ∈ D.proved ∧
      (h.reg.blockAt (sampleOf h r) ((partOf h r).lo + D.proved.getD (r.n h) 0) == r.u) = true := by
    rw [hrank]
    exact AddressRow.arr_findIdx
      (f := fun x => h.reg.blockAt (sampleOf h r) ((partOf h r).lo + x) == r.u) hfind
  -- the sampled proof block at the instance
  obtain ⟨j', hj', hge', hlt', hx'⟩ := (AddressRow.mem_drawn hprov).1 hmem
  have hju' : h.reg.blockAt (sampleOf h r) j'.toNat = r.u := by
    have e : (partOf h r).lo + D.proved.getD (r.n h) 0 = j'.toNat := by omega
    rw [← e]
    exact beq_iff_eq.1 hpred
  obtain ⟨i', hi', hρi', hje'⟩ := key j' hj' hju'
  have hU : unitOf r.ins.read.draw (r.n h) = i' * h.reg.ns.getD t 0 + p := by
    rw [hunit]
    omega
  have hplace : placeOf (iroots r.x r.seen r.ρ) = i' := by
    have hir : iroots r.x r.seen r.ρ = irootsOf r.m₁ r.ρ := rfl
    rw [hir, placeOf_irootsOf hR.msg r.ρ, hbl, ← hρi']
    exact AddressRow.arr_idxOf_getD hnd hi'
  refine ⟨?_, hlab, ?_, ?_, ?_, ?_⟩
  · show r.n h < r.sts[0]!.pub.n
    rw [hn]
    exact hlt
  · rw [hU]
    exact hcell i' hi'
  · rw [hU, hPn, (AddressRow.cell_div_mod hp).1]
    exact hplace.symm
  · rw [hU, hPn, (AddressRow.cell_div_mod hp).1, hcb, ← hρi']
    exact AddressRow.arr_getElem?_getD hi'
  · rw [hU, hPn, (AddressRow.cell_div_mod hp).2, hcp]

theorem rowFact_of_slots {h : Held} (hW : ∀ r : CheckedRun h.stage, r.Ok h → Wiring h)
    (hna : ∀ v, h.reg.always v = false) (hSS : AddressRow.SampleSlots h)
    (hwc : AddressRow.wiresChecked h = true) (hPub : AddressRow.PublishedRows h) :
    RowFact h (wireMap h) (returned h) (addressOf h) :=
  rowFact_of_wires hW (AddressRow.sits_of_slots hna hSS) hwc hPub

end SpotFlock
