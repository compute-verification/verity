import Proofs.Flock.Claims.Recursion.OpenSlots
import Proofs.Flock.Claims.Recursion.TopsOpens

/-!
# The slots' hm96 calls are recorded at `layR` (`SlotsRecorded` off `TopsPorted` and `SlotsFit`)

`SlotsRecorded elem` asks two things of each slot of a round before the last (`slotRounds`): its two hm96 calls are among
its part's recorded calls at `layR elem`, and its port's row prefix fits its row. The calls hold at every slot whose part
has a circuit with port `top`, where a cap's slot sits:

* **RecCap's slots** (round 0): V\*'s last statement is `.cap 0 N` (`capAt_stmt`), RecCap's part has port `top`
  (`capPorts`), and `capsCalls` records each instance's two calls there.
* **A held row's slot**: `layoutBDeployed` records each held row's calls at its holder (`rowSlot_calls`).
* **A later cap's slots**: V\*'s statement at `capsFrom + g` is `.cap (g + 1) (2n)` (`capStmt_at`) and the slot's
  instance `rep · n + t` is below `2n`, so `capsCalls` records its calls once that part's circuit has port `top`.

`readX` checks port `top` only at RecCap's part (`capPorted`); at the later caps' parts it is `TopsPorted`. So
`slotsRecorded_of` gives `SlotsRecorded elem` from `TopsPorted` and `SlotsFit` (the row prefixes) at each dense compiled
statement.
-/

namespace RecDeploy.OpenSlots

open Flock
open Claims (Msg)
open Claims.Recursion.VStarB
open RecDeploy.DeployedR
open RecDeploy.RestRead

/-- **Every later cap's part has port `top`**: for each top of the rep schedule, the part at its cap
(`capsFrom + g`) has a circuit with port `top`. -/
def TopsPorted (r : XR) : Prop :=
  ∀ rs, repSched r.shape = some rs → ∀ g < (RecVStar.topSizes rs).length,
    ∃ c, r.w.parts[capsFrom r rs + g]?.bind partCircuit = some c ∧ (capPort c).isSome

/-- **Each slot's port's row prefix fits its row**: the prefix `rowInDeployed` lays at the row's byte length. -/
def SlotsFit (r : XR) : Prop :=
  ∀ l ∈ slotRounds r, ∀ σ ∈ l, rowPfx σ.c σ.q = (HmRow.rowPrefix (((σ.row.length + 7) / 8) / 2)).data.toList

/-! ## Where `layR` records calls -/

theorem mem_layR_of_dep (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) {x : Msg}
    {s : ℕ} {p : Msg × List HmNets.Lin × List HmNets.Lin} (h : p ∈ layoutBDeployed.shaAt x s) :
    p ∈ (DeployedR.layR elem).shaAt x s := by
  show p ∈ (layoutBDeployed.shaAt x s ++ ((readX x).map fun r => capsCalls r s).getD []) ++ _
  exact List.mem_append_left _ (List.mem_append_left _ h)

theorem mem_layR_of_caps (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) {x : Msg}
    {r : XR} (hr : readX x = some r) {s : ℕ} {p : Msg × List HmNets.Lin × List HmNets.Lin} (h : p ∈ capsCalls r s) :
    p ∈ (DeployedR.layR elem).shaAt x s := by
  show p ∈ (layoutBDeployed.shaAt x s ++ ((readX x).map fun r => capsCalls r s).getD []) ++ _
  rw [hr, Option.map_some, Option.getD_some]
  exact List.mem_append_left _ (List.mem_append_right _ h)

/-! ## A cap's slots -/

theorem capSlots_calls {r : XR} {s lv N : ℕ} {c : Circuit} {q : ℕ}
    (hst : (RecVStar.statements r.shape).toOption.bind (·[s]?) = some (VStar.Stmt.cap lv N))
    (hc : r.w.parts[s]?.bind partCircuit = some c) (hq : capPort c = some q) {ns : List ℕ}
    (hns : ∀ n ∈ ns, n < N) {σ : Slot} (hσ : σ ∈ capSlots r s ns) :
    ∀ p ∈ hmCalls σ.c σ.nm σ.q σ.n, p ∈ capsCalls r σ.s := by
  obtain ⟨n, hn, rfl⟩ := List.mem_map.1 hσ
  have hcO : cOf r s = c := by simp [cOf, hc]
  intro p hp
  simp only [hcO, hq, Option.getD_some] at hp
  unfold capsCalls
  rw [hst]
  simp only [Option.bind_some, hc, hq, Option.map_some, Option.getD_some]
  exact List.mem_flatMap.2 ⟨n, List.mem_range.2 (hns n hn), hp⟩

/-- **V\*'s last statement is RecCap's `.cap 0 N`**, at the part `capAt` names. -/
theorem capAt_stmt {x : Msg} {r : XR} (hr : readX x = some r) {N : ℕ} (hc : capAt r = some (nS x - 1, N)) :
    (RecVStar.statements r.shape).toOption.bind (·[nS x - 1]?) = some (VStar.Stmt.cap 0 N) := by
  obtain ⟨vs, hvs, hlen⟩ := readX_statements hr
  have hb := statements_cap hvs
  have hn : nS x = r.w.parts.length := by simp [nS, hr]
  unfold capAt at hc
  simp only [hvs, Except.toOption, Option.bind_some, hb, Option.some.injEq, Prod.mk.injEq] at hc
  rw [hvs, hn, hlen]
  simp only [Except.toOption, Option.bind_some, ← Array.back?_eq_getElem?, hb, hc.2]

/-- **A held row's slot's calls are `layoutBDeployed`'s** at its holder. -/
theorem rowSlot_calls {x : Msg} {r : XR} (hr : readX x = some r) {j : ℕ} {σ : Slot} (h : rowSlot r j = some σ) :
    ∀ p ∈ hmCalls σ.c σ.nm σ.q σ.n, p ∈ layoutBDeployed.shaAt x σ.s := by
  obtain ⟨s, -, hs⟩ := List.exists_of_findSome?_eq_some h
  unfold rowSlotAt at hs
  obtain ⟨hh, hf, hs⟩ := Option.bind_eq_some_iff.1 hs
  obtain ⟨c, hc, rfl⟩ := Option.map_eq_some_iff.1 hs
  intro p hp
  simp only [layoutBDeployed, hr, Option.bind_some, hc, Option.map_some, Option.getD_some]
  exact List.mem_flatMap.2 ⟨hh, List.mem_of_find?_eq_some hf, hp⟩

/-- **A later cap's slots' calls are `capsCalls`'**, where its part has port `top`. -/
theorem capSlotsAt_calls {x : Msg} {r : XR} (hr : readX x = some r) {rs : Array RecVStar.SchedRound}
    (hrs : repSched r.shape = some rs) (hT : TopsPorted r) {j e : ℕ} (hj : j < 2 * rs.size)
    (he : e < ((rs.getD (j % rs.size) default).2.1).size) {σ : Slot} (hσ : σ ∈ capSlotsAt r rs j e) :
    ∀ p ∈ hmCalls σ.c σ.nm σ.q σ.n, p ∈ capsCalls r σ.s := by
  have hpos : 0 < rs.size := by omega
  have hts := topSizes_at (Nat.mod_lt j hpos) he
  have hst := capStmt_at hr hrs hts
  have hg := (List.getElem?_eq_some_iff.1 hts).1
  obtain ⟨c, hc, hq⟩ := hT rs hrs _ hg
  obtain ⟨q, hq⟩ := Option.isSome_iff_exists.1 hq
  rw [← Nat.add_assoc] at hst hc
  have hn' : ((rs.getD (j % rs.size) default).2.1).getD e 0 = (rs.getD (j % rs.size) default).2.1[e] := by
    generalize (rs.getD (j % rs.size) default).2.1 = a at he ⊢
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem he, Option.getD_some]
  unfold capSlotsAt at hσ
  refine capSlots_calls hst hc hq (fun n hn => ?_) hσ
  obtain ⟨t, ht, rfl⟩ := List.mem_map.1 hn
  rw [hn'] at ht ⊢
  have ht' := List.mem_range.1 ht
  have h2 : j / rs.size < 2 := Nat.div_lt_of_lt_mul (by omega)
  obtain h0 | h1 : j / rs.size = 0 ∨ j / rs.size = 1 := by
    generalize j / rs.size = m at h2 ⊢
    omega
  · rw [h0]
    omega
  · rw [h1]
    omega

/-! ## `SlotsRecorded` -/

/-- **Each slot's calls are recorded at `layR elem`**, at a dense compiled statement whose later caps' parts have port
`top`. -/
theorem slots_calls (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) {x : Msg}
    (hx : DenseX x) {r : XR} (hr : readX x = some r) (hT : TopsPorted r) :
    ∀ l ∈ slotRounds r, ∀ σ ∈ l, ∀ p ∈ hmCalls σ.c σ.nm σ.q σ.n, p ∈ (DeployedR.layR elem).shaAt x σ.s := by
  intro l hl σ hσ p hp
  obtain ⟨N, hc, -⟩ := capAt_readX hr
  unfold slotRounds at hl
  rw [hc] at hl
  rcases List.mem_append.1 hl with hl | hl
  · simp only [Option.toList_some, List.map_cons, List.map_nil, List.mem_singleton] at hl
    subst hl
    obtain ⟨st, hps, hd⟩ := hx
    obtain ⟨c, hc', hq⟩ := capPorts Claims.Recursion.Form.Dense x st hps hd r hr
    obtain ⟨q, hq⟩ := Option.isSome_iff_exists.1 hq
    exact mem_layR_of_caps elem hr
      (capSlots_calls (capAt_stmt hr hc) hc' hq (fun n hn => List.mem_range.1 hn) hσ p hp)
  · cases hrs : repSched r.shape with
    | none => simp [hrs] at hl
    | some rs =>
      simp only [hrs, Option.map_some, Option.getD_some, List.mem_map, List.mem_range] at hl
      obtain ⟨j, hj, rfl⟩ := hl
      unfold roundSlots at hσ
      rcases List.mem_append.1 hσ with hσ | hσ
      · obtain ⟨e, he, hσ⟩ := List.mem_flatMap.1 hσ
        exact mem_layR_of_caps elem hr (capSlotsAt_calls hr hrs hT hj (List.mem_range.1 he) hσ p hp)
      · exact mem_layR_of_dep elem (rowSlot_calls hr (Option.mem_toList.1 hσ) p hp)

/-- **`SlotsRecorded elem`** from the later caps' ports and the slots' row prefixes, at each dense compiled
statement. -/
theorem slotsRecorded_of (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (h : ∀ x, DenseX x → ∀ r, readX x = some r → TopsPorted r ∧ SlotsFit r) : SlotsRecorded elem :=
  fun x hx r hr l hl σ hσ =>
    ⟨slots_calls elem hx hr (h x hx r hr).1 l hl σ hσ, (h x hx r hr).2 l hl σ hσ⟩

/-- **`OpensSlots` at `layR elem`** from the later caps' ports and the slots' row prefixes. -/
theorem opensSlots_layR_of (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (h : ∀ x, DenseX x → ∀ r, readX x = some r → TopsPorted r ∧ SlotsFit r) :
    RestLow.OpensSlots (DeployedR.layR elem) :=
  opensSlots_layR elem (slotsRecorded_of elem h)

end RecDeploy.OpenSlots
