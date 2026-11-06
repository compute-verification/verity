module

public import Proofs.Flock.RecGeneric.AtFlockCounts
public import Proofs.Flock.Claims.Recursion.AcceptedScope
public import Proofs.Flock.Claims.Recursion.DeployedR

/-!
# The calls V\*'s parts record at Flock's deployment, within the verifier's cap

At V\*'s deployed bound layout (`RecDeploy.DeployedR.layR`), part `s` records the hm96 calls `shaAt x s` lists, and
`w`, the calls `E_O`'s witnesses record, is `W(x) = ∑_{s < nS(x)} |shaAt x s|` (`calls_length_recBoundAt`). Each
part's are its statement's at the verifier's inner shape, two an hm96 slot, at most `RecBudget.recordedAt` of them
(`shaAt_layR_le`):

* the rows its message-round ports read (`heldRows`, at most their registered reads, `heldRows_length_le`);
* RecCap's `top` at each of its `N` nodes (`capsCalls`), the `out` rows of a level's openings and of RecCap's nodes
  (`outsCalls`), check 5's `v` and `out` at an algebra part's two instances and `coef` and `dirs` at each of a level's
  openings (`valCalls`);
* the `rec-acc` chain's rows (`accCalls`): `acc0` and `acc1` at an algebra part's two instances, and at each row a
  level's units read (`RecVStar.accReads`), the unit's `acc` and its `acc_out`.

So `W(x)` is at most `RecBudget.recordedCalls` at the verifier's inner shape and V\*'s list there (`layR_W_le`), which
the recursive setup refuses past `2^32` (`RecBudget.recorded`, `budget_recorded_ok`): wherever one staged part sets up,
`W(x) ≤ 2^32` (`layR_W_le_staged`), and so on every statement a run of the compiled verifier accepts
(`layR_W_le_of_acc`). There `w ≤ v` holds at every `v ≥ 2^32` (`calls_le_layR_of_acc`), the form `ksLinkOn_tight`
takes, and at `v' = max(v, W(x))` at every statement (`calls_le_layR_max`). The bound is an upper bound, not an
equality: a part whose circuit lacks a port its statement names records fewer.
-/

@[expose] public section

namespace RecDeploy.DeployedR

open Flock
open Claims (Msg)
open FlockDeploy (Coin)

/-! ## Lengths -/

theorem sum_map_le_sum_map {α : Type} (l : List α) (f g : α → ℕ) (h : ∀ a ∈ l, f a ≤ g a) :
    (l.map f).sum ≤ (l.map g).sum := by
  induction l with
  | nil => simp
  | cons a l ih =>
    simp only [List.map_cons, List.sum_cons]
    exact Nat.add_le_add (h a (by simp)) (ih fun b hb => h b (by simp [hb]))

theorem sum_map_le_mul {α : Type} (l : List α) (f : α → ℕ) (c : ℕ) (h : ∀ a ∈ l, f a ≤ c) :
    (l.map f).sum ≤ c * l.length := by
  induction l with
  | nil => simp
  | cons a l ih =>
    simp only [List.map_cons, List.sum_cons, List.length_cons, Nat.mul_add, Nat.mul_one]
    have h1 := h a (by simp)
    have h2 := ih fun b hb => h b (by simp [hb])
    omega

theorem flatMap_length_le_mul {α β : Type} (l : List α) (f : α → List β) (c : ℕ)
    (h : ∀ a ∈ l, (f a).length ≤ c) : (l.flatMap f).length ≤ c * l.length := by
  rw [List.length_flatMap]
  exact sum_map_le_mul l _ c h

theorem getD_length_le {α : Type} {o : Option (List α)} {B : ℕ} (h : ∀ l, o = some l → l.length ≤ B) :
    (o.getD []).length ≤ B := by
  cases o with
  | none => exact Nat.zero_le _
  | some l => exact h l rfl

/-- An indexed sum over a list's positions, termwise at most a map over its elements, is at most the map's sum. -/
theorem sum_range_le_sum_map {α : Type} (f : α → ℕ) :
    ∀ (l : List α) (g : ℕ → ℕ), (∀ s (h : s < l.length), g s ≤ f l[s]) →
      ((List.range l.length).map g).sum ≤ (l.map f).sum
  | [], _, _ => by simp
  | a :: t, g, h => by
    simp only [List.length_cons, List.range_succ_eq_map, List.map_cons, List.map_map, List.sum_cons]
    exact Nat.add_le_add (h 0 (by simp))
      (sum_range_le_sum_map f t (g ∘ Nat.succ) fun s hs => h (s + 1) (by simpa using hs))

/-! ## Each part's calls -/

/-- **Part `s`'s calls at `layR`**: `layoutBDeployed`'s (its held rows'), every cap's, and the output rows', check 5's
and the chain's (`moreCalls`). -/
theorem shaAt_layR_length (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    {x : Msg} {r : XR} (hr : readX x = some r) (s : ℕ) :
    ((layR elem).shaAt x s).length = (layoutBDeployed.shaAt x s).length + (capsCalls r s).length +
      ((outsCalls r s).length + (valCalls r s).length + (accCalls r s).length) := by
  show ((layoutBDeployed.shaAt x s ++ ((readX x).map fun r => capsCalls r s).getD []) ++
    ((readX x).map fun r => moreCalls r s).getD []).length = _
  simp only [hr, Option.map_some, Option.getD_some, moreCalls, List.length_append]

theorem deployedSha_length_le (x : Msg) {r : XR} (hr : readX x = some r) (s : ℕ) :
    (layoutBDeployed.shaAt x s).length ≤ 2 * (heldRows r s).length := by
  show ((((readX x).bind fun r => (r.w.parts[s]?.bind partCircuit).map fun c =>
    (heldRows r s).flatMap fun h => hmCalls c h.2.1 h.2.2.1 h.2.2.2).getD [])).length ≤ _
  rw [hr, Option.bind_some]
  cases r.w.parts[s]?.bind partCircuit with
  | none => exact Nat.zero_le _
  | some c => exact flatMap_length_le_mul _ _ 2 fun h _ => by simp [hmCalls]

theorem heldRows_length_le {r : XR} {s : ℕ} {st : VStar.Stmt}
    (hst : (RecVStar.statements r.shape).toOption.bind (·[s]?) = some st) :
    (heldRows r s).length ≤ ((RecVStar.ownPorts st).map (·.2.2.size)).sum := by
  unfold heldRows
  rw [hst]
  cases r.w.parts[s]?.bind partCircuit with
  | none => exact Nat.zero_le _
  | some c =>
    cases repSched r.shape with
    | none => exact Nat.zero_le _
    | some rs =>
      dsimp only
      rw [List.length_flatMap]
      refine sum_map_le_sum_map _ _ _ fun q _ => ?_
      split
      · exact (List.length_filterMap_le _ _).trans (by simp)
      · exact Nat.zero_le _

theorem capsCalls_length_le {r : XR} {s : ℕ} {st : VStar.Stmt}
    (hst : (RecVStar.statements r.shape).toOption.bind (·[s]?) = some st) :
    (capsCalls r s).length ≤ (match (generalizing := false) st with | .cap _ N => 2 * N | _ => 0) := by
  unfold capsCalls
  rw [hst, Option.bind_some]
  cases st
  case cap l N =>
    try dsimp only
    cases r.w.parts[s]?.bind partCircuit with
    | none => exact Nat.zero_le _
    | some c =>
      rw [Option.bind_some]
      cases capPort c with
      | none => exact Nat.zero_le _
      | some q => exact (flatMap_length_le_mul _ _ 2 fun n _ => by simp [hmCalls]).trans (by simp)
  all_goals exact Nat.zero_le _

theorem outPortCalls_length_le (c : Circuit) (k : ℕ) : (outPortCalls c k).length ≤ 2 * k := by
  unfold outPortCalls
  split
  · exact (flatMap_length_le_mul _ _ 2 fun n _ => by simp [outCalls]).trans (by simp)
  · exact Nat.zero_le _

theorem outsCalls_length_le {r : XR} {s : ℕ} {st : VStar.Stmt}
    (hst : (RecVStar.statements r.shape).toOption.bind (·[s]?) = some st) :
    (outsCalls r s).length ≤ (match (generalizing := false) st with
      | .recOpen _ _ _ n => 2 * n
      | .recOpenStacked _ _ _ n => 2 * n
      | .cap _ N => 2 * N
      | _ => 0) := by
  unfold outsCalls
  rw [hst]
  cases r.w.parts[s]?.bind partCircuit with
  | none => cases st <;> exact Nat.zero_le _
  | some c => cases st <;> first | exact outPortCalls_length_le c _ | exact Nat.zero_le _

theorem valPorts_length_le {r : XR} {s : ℕ} {st : VStar.Stmt}
    (hst : (RecVStar.statements r.shape).toOption.bind (·[s]?) = some st) :
    (valPorts r s).length ≤
      (match (generalizing := false) st with | .inner .. => 4 | .recOpen _ _ _ n => 2 * n | _ => 0) := by
  unfold valPorts
  rw [hst]
  cases r.w.parts[s]?.bind partCircuit with
  | none => cases st <;> exact Nat.zero_le _
  | some c =>
    cases st
    case inner j nm p =>
      try dsimp only
      exact (flatMap_length_le_mul _ _ 2 fun nm _ => by split <;> simp).trans (by simp [resPorts])
    case recOpen l a h n =>
      try dsimp only
      exact (flatMap_length_le_mul _ _ n fun nm _ => by split <;> simp).trans (by simp [openPorts] <;> omega)
    all_goals exact Nat.zero_le _

theorem valCalls_length_le {r : XR} {s : ℕ} {st : VStar.Stmt}
    (hst : (RecVStar.statements r.shape).toOption.bind (·[s]?) = some st) :
    (valCalls r s).length ≤
      (match (generalizing := false) st with | .inner .. => 8 | .recOpen _ _ _ n => 4 * n | _ => 0) := by
  have hv := valPorts_length_le hst
  unfold valCalls
  cases r.w.parts[s]?.bind partCircuit with
  | none => exact Nat.zero_le _
  | some c =>
    refine (flatMap_length_le_mul _ _ 2 fun h _ => by simp [portCalls]).trans ?_
    cases st <;> (try simp only at hv ⊢) <;> omega

theorem accPortCalls_length_le (c : Circuit) (h : String × ℕ) :
    (accPortCalls c h).length ≤ 2 + (if h.1 = "acc" then 2 else 0) := by
  unfold accPortCalls
  split_ifs <;> (repeat' split) <;> simp [hmCalls, accOutCalls]

theorem accCalls_length_le_sum (r : XR) (s : ℕ) :
    (accCalls r s).length ≤ ((accHeld r s).map fun h => 2 + if h.1 = "acc" then 2 else 0).sum := by
  unfold accCalls
  cases r.w.parts[s]?.bind partCircuit with
  | none => exact Nat.zero_le _
  | some c =>
    show ((accHeld r s).flatMap (accPortCalls c)).length ≤ _
    rw [List.length_flatMap]
    exact sum_map_le_sum_map _ _ _ fun h _ => accPortCalls_length_le c h

/-- Off an algebra part, the chain is read only at port `acc`. -/
theorem accRead_ne_acc {r : XR} {st : VStar.Stmt} (hni : ∀ j nm p, st ≠ .inner j nm p) {nm : String}
    (hnm : nm ≠ "acc") (u : ℕ) : accRead r st nm u = none := by
  cases st
  case inner j nm' p => exact absurd rfl (hni j nm' p)
  all_goals (unfold accRead; simp [hnm])

theorem accCost_inner {r : XR} {s j : ℕ} {nm : String} {p}
    (hst : (RecVStar.statements r.shape).toOption.bind (·[s]?) = some (.inner j nm p)) :
    ((accHeld r s).map fun h => 2 + if h.1 = "acc" then 2 else 0).sum ≤ 8 := by
  have hC : accCands r (.inner j nm p) = [("acc0", 0), ("acc0", 1), ("acc1", 0), ("acc1", 1)] := by
    first | rfl | simp [accCands, RecVStar.accReads, Except.toOption]
  have hH : accHeld r s = ([("acc0", 0), ("acc0", 1), ("acc1", 0), ("acc1", 1)] : List (String × ℕ)).filter
      fun h => (accRead r (.inner j nm p) h.1 h.2).isSome := by
    unfold accHeld
    rw [hst]
    try dsimp only
    rw [hC]
  rw [hH]
  refine (sum_map_le_mul _ _ 2 fun h hh => ?_).trans ?_
  · have h4 := (List.mem_filter.1 hh).1
    simp only [List.mem_cons, List.not_mem_nil, or_false] at h4
    rcases h4 with rfl | rfl | rfl | rfl <;> simp
  · exact (Nat.mul_le_mul_left 2 (List.length_filter_le _ _)).trans (by simp)

theorem accCost_other {r : XR} {s : ℕ} {st : VStar.Stmt}
    (hst : (RecVStar.statements r.shape).toOption.bind (·[s]?) = some st)
    (hni : ∀ j nm p, st ≠ .inner j nm p) :
    ((accHeld r s).map fun h => 2 + if h.1 = "acc" then 2 else 0).sum ≤
      4 * ((((RecVStar.accReads r.shape st).toOption.bind id).map Array.size).getD 0) := by
  have h0 : ∀ nm : String, nm ≠ "acc" → ∀ u, (accRead r st nm u).isSome = false := fun nm hnm u => by
    rw [accRead_ne_acc hni hnm u]; rfl
  have hH : accHeld r s =
      ((List.range ((((RecVStar.accReads r.shape st).toOption.bind id).map Array.size).getD 0)).map
        fun u => ("acc", u)).filter fun h => (accRead r st h.1 h.2).isSome := by
    unfold accHeld accCands
    rw [hst]
    try dsimp only
    rw [List.filter_append]
    simp [h0]
  rw [hH]
  refine (sum_map_le_mul _ _ 4 fun h _ => by split_ifs <;> omega).trans ?_
  exact Nat.mul_le_mul_left 4 ((List.length_filter_le _ _).trans (by simp))

theorem accCalls_length_le {r : XR} {s : ℕ} {st : VStar.Stmt}
    (hst : (RecVStar.statements r.shape).toOption.bind (·[s]?) = some st) :
    (accCalls r s).length ≤ (match (generalizing := false) st with | .inner .. => 8 | _ => 0) +
      4 * ((((RecVStar.accReads r.shape st).toOption.bind id).map Array.size).getD 0) := by
  refine (accCalls_length_le_sum r s).trans ?_
  cases st
  case inner j nm p => exact (accCost_inner hst).trans (Nat.le_add_right _ _)
  all_goals exact (accCost_other hst fun _ _ _ h => by cases h).trans (Nat.le_add_left _ _)

/-- **Part `s` records at most `RecBudget.recordedAt` hm96 calls at `layR`**, at its statement in V\*'s list at the
verifier's inner shape. -/
theorem shaAt_layR_le (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    {x : Msg} {r : XR} (hr : readX x = some r) {s : ℕ} {st : VStar.Stmt}
    (hst : (RecVStar.statements r.shape).toOption.bind (·[s]?) = some st) :
    ((layR elem).shaAt x s).length ≤ RecBudget.recordedAt r.shape st := by
  rw [shaAt_layR_length elem hr]
  have h1 := deployedSha_length_le x hr s
  have h2 := heldRows_length_le hst
  have h3 := capsCalls_length_le hst
  have h4 := outsCalls_length_le hst
  have h5 := valCalls_length_le hst
  have h6 := accCalls_length_le hst
  unfold RecBudget.recordedAt
  cases st <;> (try simp only at h3 h4 h5 h6 ⊢) <;> omega

/-! ## The statement's calls -/

/-- **`W(x)` at `layR` is at most `RecBudget.recordedCalls`** at the verifier's inner shape and V\*'s list there. -/
theorem layR_W_le (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) {x : Msg} {r : XR}
    (hr : readX x = some r) {vs : Array VStar.Stmt} (hvs : RecVStar.statements r.shape = .ok vs)
    (hlen : r.w.parts.length = vs.size) :
    ((List.range (nS x)).map fun s => ((layR elem).shaAt x s).length).sum ≤ RecBudget.recordedCalls r.shape vs := by
  have hn : nS x = vs.toList.length := by simp [nS, hr, hlen]
  rw [hn]
  unfold RecBudget.recordedCalls
  refine sum_range_le_sum_map _ vs.toList _ fun s hs => ?_
  have hs' : s < vs.size := by simpa using hs
  have hst : (RecVStar.statements r.shape).toOption.bind (·[s]?) = some vs[s] := by
    rw [hvs]
    simp [Except.toOption, hs']
  exact shaAt_layR_le elem hr hst

/-- **`W(x) ≤ 2^32` wherever one staged part sets up**: V\*'s list at the setup's inner shape, `x`'s (`insO_staged`),
records at most `2^32` calls (`RecBudget.recorded`, `budget_recorded_ok`). -/
theorem layR_W_le_staged (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    {st : Stage} {x : Msg} {ms : List (Msg × Claims.Out)} {j : ℕ} {i : RecIns} {spec : SessionSpec}
    {sts : Array Stmt} (hs : setupO (staged st x ms j) = some (i, spec, sts)) :
    ((List.range (nS x)).map fun s => ((layR elem).shaAt x s).length).sum ≤ 2 ^ 32 := by
  obtain ⟨hins, hset⟩ := setupO_some hs
  obtain ⟨r, hr, hinner, -⟩ := insO_staged hins
  obtain ⟨vs, hvs, hlen, -⟩ := readX_listed hr
  have hB := RecIns.setup_statements hset
  rw [hinner] at hB
  exact (layR_W_le elem hr hvs hlen).trans (budget_recorded_ok hB hvs)

/-- **`W(x) ≤ 2^32` where every staged part is in scope** (`PartIn`), as at an accepted statement. -/
theorem layR_W_le_of_partIn (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    {st : Stage} {x : Msg}
    (hO : ∀ j < nS x, ∃ i spec' sts', PartIn (staged st x (pick st x j) j) i spec' sts') :
    ((List.range (nS x)).map fun s => ((layR elem).shaAt x s).length).sum ≤ 2 ^ 32 := by
  rcases Nat.eq_zero_or_pos (nS x) with h0 | hpos
  · rw [h0]; simp
  · obtain ⟨i, spec', sts', hp⟩ := hO 0 hpos
    exact layR_W_le_staged elem hp.setup

/-- **`W(x) ≤ 2^32` on every statement a run of the compiled verifier accepts** (`partIn_of_run_acc`). -/
theorem layR_W_le_of_acc (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    {H : Msg → Msg} {IV : Claims.Verifier Coin} {st : Stage} {x : Msg} {Ω : Type} {P : Claims.Prover Ω} {ω : Ω}
    {r : Fin ((V IV st).k x) → Coin} (hacc : (Claims.run H P (V IV st) x ω r).out = .acc) :
    ((List.range (nS x)).map fun s => ((layR elem).shaAt x s).length).sum ≤ 2 ^ 32 :=
  layR_W_le_of_partIn elem (partIn_of_run_acc hacc)

end RecDeploy.DeployedR

namespace Claims.RecGeneric

open Claims Claims.Recursion

section Deployed

open FlockDeploy (Coin)

variable {Col T Col' T' : Type} [DecidableEq Col] [DecidableEq Col']

/-- **`w ≤ v` at the deployment's layout, at every `v ≥ 2^32`, wherever a run of the composite's verifier accepts**:
there `W(x) ≤ 2^32` (`layR_W_le_of_acc`, the composite's verifier being the compiled record's), and `E_O`'s
witnesses record `W(x)` calls (`calls_le_recBoundAt`). This is `ksLinkOn_tight`'s `hW`. -/
theorem calls_le_layR_of_acc (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (b : RecDeploy.Brings)
    (elem : (r : RecDeploy.XR) → List (Fin r.w.parts.length × List (Msg × List _root_.Flock.HmNets.Lin)))
    (Bd : Bound Coin Col T) (DO : Claims.Flock.Deploy Coin Col' T') (H : Msg → Msg) {Ω : Type} (P : Prover Ω)
    (x : Msg) {v : ℕ} (hv : 2 ^ 32 ≤ v)
    (hex : ∃ ω r, (run H P (compDep (RecDeploy.recBoundAt rel b (RecDeploy.DeployedR.layR elem)) Bd DO).V x ω r).out
      = .acc) :
    ∀ (ω : Ω)
      (u : (Fin ((compDep (RecDeploy.recBoundAt rel b (RecDeploy.DeployedR.layR elem)) Bd DO).V.k x) → Coin) ×
        (compDep (RecDeploy.recBoundAt rel b (RecDeploy.DeployedR.layR elem)) Bd DO).EO.C x) ws,
      (compDep (RecDeploy.recBoundAt rel b (RecDeploy.DeployedR.layR elem)) Bd DO).wits H P x ω u = some ws →
        (((RecDeploy.recBoundAt rel b (RecDeploy.DeployedR.layR elem)).record Bd.IVStar).calls x ws).length ≤ v := by
  obtain ⟨ω₀, r₀, hacc⟩ := hex
  have hacc' : (run H P (RecDeploy.V Bd.IVStar b.st) x ω₀ r₀).out = .acc := hacc
  have hW := RecDeploy.DeployedR.layR_W_le_of_acc elem hacc'
  exact fun ω u ws h => calls_le_recBoundAt rel b _ Bd DO H P x v (hW.trans hv) ω u ws h

/-- **`w ≤ v'` at the deployment's layout at `v' = max(v, W(x))`**, at every statement. -/
theorem calls_le_layR_max (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (b : RecDeploy.Brings)
    (elem : (r : RecDeploy.XR) → List (Fin r.w.parts.length × List (Msg × List _root_.Flock.HmNets.Lin)))
    (Bd : Bound Coin Col T) (DO : Claims.Flock.Deploy Coin Col' T') (H : Msg → Msg) {Ω : Type} (P : Prover Ω)
    (x : Msg) (v : ℕ) :
    ∀ (ω : Ω)
      (u : (Fin ((compDep (RecDeploy.recBoundAt rel b (RecDeploy.DeployedR.layR elem)) Bd DO).V.k x) → Coin) ×
        (compDep (RecDeploy.recBoundAt rel b (RecDeploy.DeployedR.layR elem)) Bd DO).EO.C x) ws,
      (compDep (RecDeploy.recBoundAt rel b (RecDeploy.DeployedR.layR elem)) Bd DO).wits H P x ω u = some ws →
        (((RecDeploy.recBoundAt rel b (RecDeploy.DeployedR.layR elem)).record Bd.IVStar).calls x ws).length ≤
          max v ((List.range (RecDeploy.nS x)).map fun s =>
            ((RecDeploy.DeployedR.layR elem).shaAt x s).length).sum :=
  fun ω u ws h => calls_le_recBoundAt rel b _ Bd DO H P x _ (le_max_right _ _) ω u ws h

/-- **`max(v, W(x)) = v` on every statement a run accepts, at `v ≥ 2^32`**: so the bound at `v'` is the bound at `v`
there. -/
theorem max_layR_W_of_acc
    (elem : (r : RecDeploy.XR) → List (Fin r.w.parts.length × List (Msg × List _root_.Flock.HmNets.Lin)))
    {H : Msg → Msg} {IV : Verifier Coin} {st : RecDeploy.Stage} {x : Msg} {Ω : Type} {P : Prover Ω} {ω : Ω}
    {r : Fin ((RecDeploy.V IV st).k x) → Coin} (hacc : (run H P (RecDeploy.V IV st) x ω r).out = .acc) {v : ℕ}
    (hv : 2 ^ 32 ≤ v) :
    max v ((List.range (RecDeploy.nS x)).map fun s => ((RecDeploy.DeployedR.layR elem).shaAt x s).length).sum = v :=
  max_eq_left ((RecDeploy.DeployedR.layR_W_le_of_acc elem hacc).trans hv)

end Deployed

end Claims.RecGeneric
