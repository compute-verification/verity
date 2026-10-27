import Proofs.Flock.Claims.Recursion.SlotsCalls
import Proofs.Flock.Claims.Recursion.SlotsCheck
import Proofs.Flock.Claims.Recursion.PartsVStar
import Proofs.Flock.Recursive.VStarCheck

/-!
# The later caps' ports and the slots' row prefixes, off `readX` (`SaltRowsJ` at `layR`)

`readX` checks every part against its V\* statement (`readX_partVStar`), V\*'s parse check among them
(`checkParsed_ok`): the part's circuit's ports are V\*'s rows (`vstarRows`), and unit `0`'s input leaves are its input
words in order. That settles both facts `slotsRecorded_of` takes:

* `topsPorted_of`: the statement at a later cap's part is `.cap (g + 1) (2n)` (`capStmt_at`), whose one input is
  `top` (`capSub`), so the part's circuit has port `top`.
* `slotsFit_of`: every slot reads a port named outside its statement's outputs (a cap's `top`; a held row's own port,
  `ownPorts_held`), so the port's row is one of the input block's (`in_block`), its hidden values one per bit
  (`rowHidden_eq`), and every row V\* writes is `16` bits a word (`bits_words`): the row's byte length halves to the
  port's words, the prefix `rowPfx` lays.

`saltRowsJ_layR`: `SaltRowsJ` at `layR elem` and `finalR`, with no hypothesis; `saltRowRest_layR` with it.
-/

namespace RecDeploy.OpenSlots

open Flock
open Claims (Msg)
open Claims.Recursion.VStarB
open RecDeploy.DeployedR
open RecDeploy.RestRead

/-- **A compiled statement's part passed V\*'s parse check**: its circuit, its unit's input leaves in order, and its
ports V\*'s rows. -/
theorem part_parsed {x : Msg} {r : XR} (hr : readX x = some r) {s : ℕ} {sv : _root_.VStar.Stmt}
    (hs : (RecVStar.statements r.shape).toOption.bind (·[s]?) = some sv) :
    ∃ c, r.w.parts[s]?.bind partCircuit = some c ∧ ∃ u, sv.unit = .ok u ∧
      c.leavesIn.getD 0 #[] = (Array.range u.inWords).map Int.ofNat ∧
      c.ports.toList.map (fun p => (p.name, p.words, p.bits)) = RecVStar.vstarRows sv.sub u := by
  obtain ⟨vs, hvs, -, hall⟩ := readX_partVStar hr
  rw [hvs] at hs
  simp only [Except.toOption, Option.bind_some] at hs
  obtain ⟨hsz, rfl⟩ := Array.getElem?_eq_some_iff.1 hs
  obtain ⟨-, p, c, -, hp, hc, -, hcp, -⟩ := hall s hsz
  obtain ⟨u, hu, -, -, hl, hrows⟩ := FlockSoundness.Discharge.Recursion.checkParsed_ok hcp
  exact ⟨c, by rw [hp, Option.bind_some, hc], u, hu, hl, hrows⟩

/-- **Every row V\* writes is `16` bits a word**: its inputs' and its outputs' alike. -/
theorem bits_words {c : Circuit} {sub : _root_.VStar.Sub} {u : _root_.VStar.UnitC}
    (hrows : c.ports.toList.map (fun p => (p.name, p.words, p.bits)) = RecVStar.vstarRows sub u) :
    ∀ p ∈ c.ports.toList, p.bits = 16 * p.words := by
  intro p hp
  have h : (p.name, p.words, p.bits) ∈ RecVStar.vstarRows sub u := by
    rw [← hrows]
    exact List.mem_map.2 ⟨p, hp, rfl⟩
  unfold RecVStar.vstarRows at h
  rcases List.mem_append.1 h with h | h
  · obtain ⟨⟨a, b⟩, -, he⟩ := List.mem_map.1 h
    simp only [Prod.mk.injEq] at he
    obtain ⟨-, h2, h3⟩ := he
    omega
  · obtain ⟨⟨a, b⟩, -, he⟩ := List.mem_map.1 h
    simp only [Prod.mk.injEq] at he
    obtain ⟨-, h2, h3⟩ := he
    omega

/-- **A cap's part has port `top`**: its statement's one input. -/
theorem capPort_isSome {c : Circuit} {lv N : ℕ} {u : _root_.VStar.UnitC}
    (hu : (_root_.VStar.Stmt.cap lv N).unit = .ok u)
    (hrows : c.ports.toList.map (fun p => (p.name, p.words, p.bits)) =
      RecVStar.vstarRows (_root_.VStar.Stmt.cap lv N).sub u) :
    (capPort c).isSome := by
  have hu' : (Except.ok (_root_.VStar.capUnit lv N) : Except String _root_.VStar.UnitC) = .ok u := hu
  cases hu'
  have hin : (_root_.VStar.capUnit lv N).inPorts = [_root_.VStar.CAP_TOP_WORDS] := rfl
  have hsub : (_root_.VStar.Stmt.cap lv N).sub.inputs = ["top"] := rfl
  have hm : ("top", _root_.VStar.CAP_TOP_WORDS, 16 * _root_.VStar.CAP_TOP_WORDS) ∈
      c.ports.toList.map (fun p => (p.name, p.words, p.bits)) := by
    rw [hrows, RecVStar.vstarRows, hsub, hin]
    exact List.mem_append_left _ (by simp)
  obtain ⟨p, hp, he⟩ := List.mem_map.1 hm
  have hn : p.name = "top" := congrArg Prod.fst he
  unfold capPort
  rw [Array.findIdx?_isSome, Array.any_eq_true']
  exact ⟨p, Array.mem_toList_iff.1 hp, by simp [hn]⟩

/-- **`TopsPorted`** at every record `readX` reads. -/
theorem topsPorted_of {x : Msg} {r : XR} (hr : readX x = some r) : TopsPorted r := by
  intro rs hrs g hg
  obtain ⟨c, hc, u, hu, -, hrows⟩ := part_parsed hr (capStmt_at hr hrs (List.getElem?_eq_getElem hg))
  exact ⟨c, hc, capPort_isSome hu hrows⟩

/-- **A slot's row prefix fits its row** where its port is one of V\*'s input rows: the row has the port's bits, `16`
a word, so its byte length halves to the port's words. -/
theorem slot_fit {σ : Slot} {sub : _root_.VStar.Sub} {u : _root_.VStar.UnitC}
    (hl : σ.c.leavesIn.getD 0 #[] = (Array.range u.inWords).map Int.ofNat)
    (hrows : σ.c.ports.toList.map (fun p => (p.name, p.words, p.bits)) = RecVStar.vstarRows sub u)
    (hnm : σ.nm ∉ sub.outputs) (hq : σ.c.ports.findIdx? (·.name == σ.nm) = some σ.q) :
    rowPfx σ.c σ.q = (HmRow.rowPrefix (((σ.row.length + 7) / 8) / 2)).data.toList := by
  have hqs : σ.q < σ.c.ports.size := (Array.findIdx?_eq_some_iff_getElem.1 hq).1
  have hN := in_block hrows hnm hq
  rw [getElem!_pos σ.c.ports σ.q hqs] at hN
  have hbw : σ.c.ports[σ.q].bits = 16 * σ.c.ports[σ.q].words :=
    bits_words hrows _ (Array.getElem_mem_toList hqs)
  have hrow : σ.row.length = 16 * σ.c.ports[σ.q].words := by
    unfold Slot.row
    rw [rowHidden_eq σ.n hl hq hqs hN, Option.getD_some, List.length_map, List.length_map, List.length_range, hbw]
  have e : (16 * σ.c.ports[σ.q].words + 7) / 8 / 2 = σ.c.ports[σ.q].words := by omega
  rw [hrow, e]
  unfold rowPfx
  rw [Array.getElem?_eq_getElem hqs]
  rfl

/-- **A cap's slots fit**: port `top`, its statement's input. -/
theorem capSlot_fit {x : Msg} {r : XR} (hr : readX x = some r) {s lv N : ℕ}
    (hst : (RecVStar.statements r.shape).toOption.bind (·[s]?) = some (_root_.VStar.Stmt.cap lv N))
    {ns : List ℕ} {σ : Slot} (hσ : σ ∈ capSlots r s ns) :
    rowPfx σ.c σ.q = (HmRow.rowPrefix (((σ.row.length + 7) / 8) / 2)).data.toList := by
  obtain ⟨c, hc, u, hu, hl, hrows⟩ := part_parsed hr hst
  obtain ⟨q, hq⟩ := Option.isSome_iff_exists.1 (capPort_isSome hu hrows)
  have hcO : cOf r s = c := by simp [cOf, hc]
  have hnm : "top" ∉ (_root_.VStar.Stmt.cap lv N).sub.outputs := by
    show "top" ∉ ["out"]
    decide
  obtain ⟨n, -, rfl⟩ := List.mem_map.1 hσ
  rw [hcO, hq, Option.getD_some]
  exact slot_fit (σ := ⟨s, c, "top", q, n⟩) hl hrows hnm hq

/-- **A held row's slot fits**: its holder's own port, named outside its statement's outputs (`ownPorts_held`). -/
theorem rowSlot_fit {x : Msg} {r : XR} (hr : readX x = some r) {j : ℕ} {σ : Slot} (h : rowSlot r j = some σ) :
    rowPfx σ.c σ.q = (HmRow.rowPrefix (((σ.row.length + 7) / 8) / 2)).data.toList := by
  obtain ⟨s, -, hs⟩ := List.exists_of_findSome?_eq_some h
  unfold rowSlotAt at hs
  obtain ⟨hh, hf, hs⟩ := Option.bind_eq_some_iff.1 hs
  obtain ⟨c, hc, rfl⟩ := Option.map_eq_some_iff.1 hs
  obtain ⟨vs, hvs, hlen⟩ := readX_statements hr
  have hsl : s < r.w.parts.length := by
    refine Nat.lt_of_not_le fun hge => ?_
    rw [List.getElem?_eq_none hge] at hc
    cases hc
  have hst : (RecVStar.statements r.shape).toOption.bind (·[s]?) = some vs[s] := by
    rw [hvs]
    simp only [Except.toOption, Option.bind_some]
    exact Array.getElem?_eq_getElem (by omega)
  obtain ⟨q, hq, -, -, -, hname, hfi⟩ := mem_heldRows hst hc (List.mem_of_find?_eq_some hf)
  obtain ⟨c', hc', u, -, hl, hrows⟩ := part_parsed hr hst
  rw [hc] at hc'
  cases hc'
  have hnm : hh.2.1 ∉ vs[s].sub.outputs := by
    rw [hname]
    exact (ownPorts_held hq).1
  have hfi' : c.ports.findIdx? (·.name == hh.2.1) = some hh.2.2.1 := by
    rw [hname]
    exact hfi
  exact slot_fit (σ := ⟨s, c, hh.2.1, hh.2.2.1, hh.2.2.2⟩) hl hrows hnm hfi'

/-- **`SlotsFit`** at every record `readX` reads. -/
theorem slotsFit_of {x : Msg} {r : XR} (hr : readX x = some r) : SlotsFit r := by
  intro l hl σ hσ
  obtain ⟨N, hc, -⟩ := capAt_readX hr
  unfold slotRounds at hl
  rw [hc] at hl
  rcases List.mem_append.1 hl with hl | hl
  · simp only [Option.toList_some, List.map_cons, List.map_nil, List.mem_singleton] at hl
    subst hl
    exact capSlot_fit hr (capAt_stmt hr hc) hσ
  · cases hrs : repSched r.shape with
    | none => simp [hrs] at hl
    | some rs =>
      simp only [hrs, Option.map_some, Option.getD_some, List.mem_map, List.mem_range] at hl
      obtain ⟨j, hj, rfl⟩ := hl
      unfold roundSlots at hσ
      rcases List.mem_append.1 hσ with hσ | hσ
      · obtain ⟨e, he, hσ⟩ := List.mem_flatMap.1 hσ
        have hpos : 0 < rs.size := by omega
        have hst := capStmt_at hr hrs (topSizes_at (Nat.mod_lt j hpos) (List.mem_range.1 he))
        rw [← Nat.add_assoc] at hst
        unfold capSlotsAt at hσ
        exact capSlot_fit hr hst hσ
      · exact rowSlot_fit hr (Option.mem_toList.1 hσ)

end RecDeploy.OpenSlots

namespace RecDeploy.RestLow

open Flock
open Claims (Msg Out)
open FlockDeploy (FinalP)
open RecDeploy.DecompBR

/-- **`SaltRowsJ` at the layout of record and `finalR`**: `saltRowsJ_of_slots` through `opensSlots_layR_of`, whose two
facts `readX` settles (`topsPorted_of`, `slotsFit_of`). -/
theorem saltRowsJ_layR (rd : Msg → Option FinalP) (wE w : Msg → ℝ)
    (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) :
    SaltRowsJ rd wE DeployedR.finalR w (DeployedR.layR elem) :=
  saltRowsJ_of_slots rd wE w _ (OpenSlots.opensSlots_layR_of elem fun _ _ _ hr =>
    ⟨OpenSlots.topsPorted_of hr, OpenSlots.slotsFit_of hr⟩)

/-- **`SaltRowRest` at the layout of record and `finalR`**, from `saltRowsJ_layR`. -/
theorem saltRowRest_layR (rd : Msg → Option FinalP) (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round)
    (nOf : Msg → ℕ → ℕ) (hd : Msg → Flock.RecRecord.Record) (rest : Msg → List (Msg × Out) → List UInt8)
    (wE w : Msg → ℝ) (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) :
    SaltRowRest rd b aux nOf hd rest wE DeployedR.finalR w (DeployedR.layR elem) :=
  saltRowRest_of rd b aux nOf hd rest wE _ w _ (saltRowsJ_layR rd wE w elem)

end RecDeploy.RestLow
