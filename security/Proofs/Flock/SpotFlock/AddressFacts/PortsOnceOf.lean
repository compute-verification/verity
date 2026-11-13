module

public import Proofs.Flock.SpotFlock.AddressFacts.CopyOut
public import Proofs.Pouw.Period.Sort

/-!
# Each member names its ports once

`portsOnce_of_rows`: `PortsOnce` where the second stage's design evaluates, so the width and the copy at the address
map rest on the design and the wiring alone (`widthFact_of_design`, `copyFact_of_design`).

A member's circuit is a committed circuit unrolled (`member_commit`), whose parse ends in `HmOut.checkRows`, flat or
templated. `checkRows` sorts the input rows' names with the output ports' names and refuses two equal neighbours, so
those names are distinct (`nodup_of_sort`, on `Array.qsort`'s sort and permutation of strings). Its rows are the input
rows, then one row per run of output ports named after the run's first port; with each run one port (`OutRowsS`,
from `scopeOkH`), the rows' names are those names in order (`ports_names`). Unrolling keeps the rows.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy FlockSoundness FlockSoundness.Discharge

/-! ## Distinct names from a sort -/

/-- A list sorted by a reflexive, antisymmetric relation whose neighbours differ has no repeats. -/
theorem nodup_of_sorted {α : Type} {R : α → α → Prop} {l : List α} (hs : l.Pairwise R) (hrefl : ∀ a, R a a)
    (hanti : ∀ a b, R a b → R b a → a = b) (hn : ∀ i (hi : i + 1 < l.length), l[i] ≠ l[i + 1]) : l.Nodup := by
  rw [List.Nodup, List.pairwise_iff_getElem]
  intro i j hi hj hij e
  have hs' := List.pairwise_iff_getElem.1 hs
  have h1 : R l[i] l[i + 1] := hs' i (i + 1) hi (by omega) (by omega)
  have h2 : R l[i + 1] l[i] := by
    rw [e]
    rcases Nat.lt_or_ge (i + 1) j with h | h
    · exact hs' (i + 1) j (by omega) hj h
    · have e' : j = i + 1 := by omega
      subst e'
      exact hrefl _
  exact hn i (by omega) (hanti _ _ h1 h2)

/-- **Strings whose sort has no two equal neighbours are distinct**: `S` the sort of `A`, a permutation of it sorted
by a reflexive, antisymmetric relation. -/
theorem nodup_of_sort {R : String → String → Prop} {A S : Array String} (hp : S.toList.Perm A.toList)
    (hs : S.toList.Pairwise R) (hrefl : ∀ a, R a a) (hanti : ∀ a b, R a b → R b a → a = b)
    (h : ∀ x, x < S.size - 1 → ¬S[x]! = S[x + 1]!) : A.toList.Nodup := by
  refine hp.nodup_iff.1 (nodup_of_sorted hs hrefl hanti fun i hi => ?_)
  have hi' : i + 1 < S.size := by simpa using hi
  have hne := h i (by omega)
  rw [getElem!_pos S i (by omega), getElem!_pos S (i + 1) hi'] at hne
  simpa only [Array.getElem_toList] using hne

/-! ## What `HmOut.checkRows` checks of the names -/

/-- **What `HmOut.checkRows` checks of the rows' names**: the input rows and the output rows are all the rows, output
row `j` is named after its run's first output port, and the input rows' names with the output ports' are distinct. -/
theorem checkRows_names {c : _root_.Flock.Circuit} (h : _root_.Flock.HmOut.checkRows c = .ok ()) :
    _root_.Flock.HmOut.nIn c + c.outRowPorts.size = c.ports.size ∧
      (∀ j, j < c.outRowPorts.size → (c.outPorts[(_root_.Flock.HmOut.outGroup c j).1]?.map (·.1)) =
        some (c.ports.getD (_root_.Flock.HmOut.nIn c + j) default).name) ∧
      ((c.ports.extract 0 (_root_.Flock.HmOut.nIn c)).map (·.name) ++ c.outPorts.map (·.1)).toList.Nodup := by
  unfold _root_.Flock.HmOut.checkRows at h
  obtain ⟨k, -, h⟩ := FlockSoundness.except_bind_ok h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) names jp at h
  obtain ⟨hbad, -⟩ := FlockSoundness.ite_throw_ok' h
  simp only [names, Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, List.any_eq_true, List.mem_range, not_exists,
    not_and, beq_iff_eq, Classical.not_not] at hbad
  obtain ⟨⟨hk, hnm⟩, hrow⟩ := hbad
  have hk' : _root_.Flock.HmOut.nIn c = k := by
    unfold _root_.Flock.HmOut.nIn
    omega
  subst hk'
  exact ⟨by omega, fun j hj => (hrow j hj).1,
    nodup_of_sort (Properties.RemoteAudit.qsort_toList_perm _ _)
      (Properties.Pouw.Period.SecurityProofs.Sort.qsort_sorted _) String.le_refl (fun _ _ => String.le_antisymm) hnm⟩

/-- **The rows' names are the input rows' then the output ports'**, where each output row holds one output port. -/
theorem ports_names {c : _root_.Flock.Circuit}
    (hsz : _root_.Flock.HmOut.nIn c + c.outRowPorts.size = c.ports.size)
    (hrep : c.outRowPorts = Array.replicate c.outPorts.size 1)
    (hname : ∀ j, j < c.outRowPorts.size → (c.outPorts[(_root_.Flock.HmOut.outGroup c j).1]?.map (·.1)) =
      some (c.ports.getD (_root_.Flock.HmOut.nIn c + j) default).name) :
    c.ports.toList.map (·.name) =
      ((c.ports.extract 0 (_root_.Flock.HmOut.nIn c)).map (·.name) ++ c.outPorts.map (·.1)).toList := by
  have hs : c.outRowPorts.size = c.outPorts.size := by rw [hrep, Array.size_replicate]
  rw [Array.toList_append, Array.toList_map, Array.toList_map, Array.toList_extract, List.extract_eq_take_drop,
    List.drop_zero, Nat.sub_zero]
  conv => lhs; rw [← List.take_append_drop (_root_.Flock.HmOut.nIn c) c.ports.toList]
  rw [List.map_append]
  refine (List.append_right_inj _).2 (List.ext_getElem ?_ fun j h1 h2 => ?_)
  · simp only [List.length_map, List.length_drop, Array.length_toList]
    omega
  · simp only [List.getElem_map, List.getElem_drop, Array.getElem_toList]
    have hj : j < c.outPorts.size := by simpa using h2
    have hn := hname j (by omega)
    rw [FlockSoundness.Discharge.Hidden.outGroup_single hrep hj] at hn
    simp only [Array.getElem?_eq_getElem hj, Option.map_some, Option.some.injEq] at hn
    rw [hn]
    have hq : _root_.Flock.HmOut.nIn c + j < c.ports.size := by omega
    simp [Array.getD, hq]

/-- **A circuit `HmOut.checkRows` passes names each row once**, where each output row holds one output port. -/
theorem nodup_of_checkRows {c : _root_.Flock.Circuit} (hc : _root_.Flock.HmOut.checkRows c = .ok ())
    (hrep : c.outRowPorts = Array.replicate c.outPorts.size 1) : (c.ports.toList.map (·.name)).Nodup := by
  obtain ⟨hsz, hname, hnd⟩ := checkRows_names hc
  rw [ports_names hsz hrep hname]
  exact hnd

theorem unrolled_rows (c : _root_.Flock.Circuit) :
    (_root_.Flock.HmOut.unrolled c).ports = c.ports ∧ (_root_.Flock.HmOut.unrolled c).outPorts = c.outPorts ∧
      (_root_.Flock.HmOut.unrolled c).outRowPorts = c.outRowPorts := by
  unfold _root_.Flock.HmOut.unrolled
  split
  · split
    · exact ⟨rfl, rfl, rfl⟩
    · exact ⟨rfl, rfl, rfl⟩
  · exact ⟨rfl, rfl, rfl⟩

/-! ## The members -/

variable {h : Held}

/-- **`PortsOnce` where the design evaluates**, from what each member's parse checks of its rows and its scope. -/
theorem portsOnce_of_rows {s : AuditStatement} {d : Design.Two} (hd : two h s = some d) : PortsOnce h := by
  intro t ht
  obtain ⟨r, c, hty, hc, hm⟩ := member_commit hd ht
  have hfit := rowsFit_member hd ht
  rw [hm] at hfit ⊢
  obtain ⟨e1, e2, e3⟩ := unrolled_rows c
  obtain ⟨⟨hrep, -⟩, -⟩ := hfit
  rw [e2, e3] at hrep
  rw [e1]
  rcases FlockSoundness.Discharge.Hidden.committed_of_commit hc with hC | hC
  · exact nodup_of_checkRows (FlockSoundness.Discharge.Hidden.parse_untyped hty hC).2 hrep
  · obtain ⟨tu, c₀, -, hck, -, rfl⟩ := FlockSoundness.TmplValue.parseTmpl_cases hC
    exact nodup_of_checkRows hck hrep

/-- **The width at the address map where the design evaluates and the wiring holds.** -/
theorem widthFact_of_design {s : AuditStatement} {d : Design.Two} (hd : two h s = some d) (W : Wiring h) :
    WidthFact h (addressOf h) :=
  widthFact_of_two hd (portsOnce_of_rows hd) W

/-- **The copy at the address map where the design evaluates, the wiring holds and each counted gate's row is the
address map's.** -/
theorem copyFact_of_design {s : AuditStatement} {d : Design.Two} {ω : ℕ → ℕ} {keep : ℕ → Bool}
    (hd : two h s = some d) (W : Wiring h) (hrow : RowFact h ω keep (addressOf h)) : CopyFact h (addressOf h) :=
  copyFact_of_two hd (portsOnce_of_rows hd) W hrow

end SpotFlock
