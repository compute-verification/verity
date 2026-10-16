import Definitions.Claims.Sampling
import Proofs.Flock.Claims.RedTeam.ZK
import Proofs.Flock.Claims.RedTeam.Complete

/-!
# Red team: the sampling draft's claims

* `samplingZK_collSim`: the sampling `ZK` holds at `ε_zk = 0`, at any masked record and any hash with a collision, for
  every honest prover and every assignment of a leaf layer. `SimLeak.given` keeps the simulator's own calls in
  `ZKmalR`'s collision term, so `collSimLeak`, which hashes a colliding pair whatever the leaf layer, meets it
  (`RedTeam.zkmalR_of_findColl`), as Flock's `ZK` is met (`RedTeam.flockZK_collSim`).
* `samplingComplete_empty`: the sampling `Complete` at `δ_c = 0` for every record, at a prover with an empty tape:
  `Claims.Complete` quantifies over the tape and nothing makes it nonempty (`RedTeam.complete_empty`).
* `r_toy`, `not_r_toy_empty`: the sampling relation has content. On a one-input program drawn `all` (in scope,
  `inScope_toy`), the opening of its value satisfies `R δ` at every `δ < 1`, and the empty witness fails it at every
  `δ ≥ 0`: its one unit is faulty and the design catches it surely.
-/

namespace Claims.Sampling.RedTeam

open Claims.RedTeam

/-! ## Zero knowledge -/

/-- **The colliding simulator given the leakage**: it hashes `u`, then `v`, and outputs the empty transcript, whatever
the leaf layer. -/
def collSimLeak (u v : Msg) : SimLeak where
  C _ := Unit
  prog _ _ _ := .hash u fun _ => .hash v fun _ => .ret []

theorem findColl_collSimLeak {H : Msg → Msg} {u v : Msg} (h : Collision H u v) (l : Msg → List Msg) {Θ : Type}
    (Vs : VStar Θ) (θ : Θ) (x : Msg) (c : Unit) :
    findColl (simLog H ((collSimLeak u v).given l) Vs θ x c) = some (u, v) := by
  have hvu : v ≠ u := fun e => h.1 e.symm
  simp [simLog, spLog, SimLeak.given, collSimLeak, findColl, hvu, h.2]

/-- **The sampling `ZK` holds at `ε_zk = 0`** at any masked record and any hash with a collision, for every honest
prover. -/
theorem samplingZK_collSim {Coin Col T Ω : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    [Zero Col] [Fintype Ω] (H : Msg → Msg) (D : Sampling.Deploy Coin Col T) (hzk : D.flock.zk = true)
    (P : Wit → Prover Ω) {u v : Msg} (h : Collision H u v) : Sampling.ZK H D P (collSimLeak u v) 0 :=
  ⟨hzk, fun l => zkmalR_of_findColl H D.V.k P _ _ fun x _ Vs θ c => by
    simp [findColl_collSimLeak h l Vs θ x c]⟩

/-! ## Completeness -/

/-- **The sampling `Complete` holds at `δ_c = 0`** for every record, at a prover with an empty tape. -/
theorem samplingComplete_empty {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    [Zero Col] (H : Msg → Msg) (D : Sampling.Deploy Coin Col T) : Sampling.Complete H D noTape 0 :=
  complete_empty H D.V noTape _ _

/-! ## The relation has content -/

/-- A commitment whose every opening passes. -/
def toyCm : Commitment Bool := ⟨fun _ _ => .ret true, fun _ _ _ => []⟩

/-- One input gate, returned. -/
def toyProg : Prog := ⟨[.input], [0]⟩

/-- The program, drawn `all`. -/
def toyStmt : Stmt := ⟨toyProg, ⟨"", 0, []⟩, ⟨[⟨.all, none⟩]⟩, []⟩

/-- One class of one stratum of one replay unit, the proof unit `[0, 1)`. -/
def toyPart : Partition := ⟨[[[[(0, 1)]]]]⟩

/-- The opening of the input's value. -/
def toyWit : Wit := ⟨[⟨0, true, []⟩]⟩

theorem toyPart_rus : toyPart.rus = [[(0, 1)]] := rfl

theorem toyStmt_prog : toyStmt.prog = toyProg := rfl

theorem toy_boundary : toyProg.Boundary 0 1 0 = true := by decide

/-- The unit's reference values are the witness's committed ones: the unit's one gate is on its boundary. -/
theorem toy_ref (H : Msg → Msg) (w : Wit) (ι : ℕ → Bool) :
    RU.ref toyProg [(0, 1)] (w.val H toyCm toyStmt) ι 0 = w.val H toyCm toyStmt 0 := by
  simp [RU.ref, toy_boundary, RU.lo, RU.hi]

theorem toy_runAt (ref : ℕ → Option Bool) : toyProg.runAt ref 0 0 = ref 0 := by
  rw [Prog.runAt]
  rfl

/-- The unit is correct exactly when the input's value is opened. -/
theorem toy_correct_iff (H : Msg → Msg) (w : Wit) (ι : ℕ → Bool) :
    toyProg.Correct (RU.ref toyProg [(0, 1)] (w.val H toyCm toyStmt) ι) 0 1 ↔
      ∃ b, w.val H toyCm toyStmt 0 = some b := by
  constructor
  · intro h
    obtain ⟨b, hb, _⟩ := (h 0 le_rfl Nat.one_pos).2 toy_boundary
    exact ⟨b, (toy_ref H w ι).symm.trans hb⟩
  · rintro ⟨b, hb⟩ g _ hg
    obtain rfl : g = 0 := by omega
    refine ⟨fun _ _ i _ hi => absurd hi (Nat.not_lt_zero i), fun _ => ⟨b, ?_, ?_⟩⟩
    · rw [toy_ref, hb]
    · rw [toy_runAt, toy_ref, hb]

theorem toy_faulty_iff (H : Msg → Msg) (w : Wit) :
    Faulty H toyCm toyStmt toyPart w 0 ↔ w.val H toyCm toyStmt 0 = none := by
  unfold Faulty
  rw [toyPart_rus]
  simp only [List.getElem?_cons_zero, Option.some.injEq, exists_eq_left', List.mem_singleton, forall_eq,
    toyStmt_prog, toy_correct_iff, exists_const]
  generalize w.val H toyCm toyStmt 0 = o
  cases o <;> simp

theorem toy_faulty (H : Msg → Msg) (w : Wit) :
    faulty H toyCm toyStmt toyPart w = if w.val H toyCm toyStmt 0 = none then {0} else ∅ := by
  ext v
  simp only [faulty, Finset.mem_filter, Finset.mem_range, toyPart_rus, List.length_singleton]
  constructor
  · rintro ⟨hv, hf⟩
    obtain rfl : v = 0 := by omega
    simp [(toy_faulty_iff H w).1 hf]
  · intro hv
    split_ifs at hv with h
    · obtain rfl := Finset.mem_singleton.1 hv
      exact ⟨Nat.one_pos, (toy_faulty_iff H w).2 h⟩
    · simp at hv

theorem toy_Pa (B : Finset ℕ) : toyStmt.design.Pa toyPart B = if window 0 1 B = ∅ then 1 else 0 := by
  simp [Design.Pa, toyStmt, toyPart, classesPa, Rule.Pa, OneStage.Pa]

theorem toyWit_val (H : Msg → Msg) : toyWit.val H toyCm toyStmt 0 = some true := by
  simp [Wit.val, toyWit, Commitment.Opens, toyCm, OC.eval]

theorem empty_val (H : Msg → Msg) : (⟨[]⟩ : Wit).val H toyCm toyStmt 0 = none := rfl

/-- **The opening of the input satisfies the sampling relation** at every `δ < 1`. -/
theorem r_toy (H : Msg → Msg) (x : Msg) {δ : ℝ} (hδ : δ < 1) :
    R H toyCm (fun _ => some toyStmt) (fun _ _ => some toyPart) δ x toyWit := by
  refine ⟨toyStmt, toyPart, rfl, rfl, ?_⟩
  show δ < toyStmt.design.Pa toyPart (faulty H toyCm toyStmt toyPart toyWit)
  rw [toy_faulty, toyWit_val, toy_Pa]
  simpa [window] using hδ

/-- **The empty witness fails the sampling relation** at every `δ ≥ 0`. -/
theorem not_r_toy_empty (H : Msg → Msg) (x : Msg) {δ : ℝ} (hδ : 0 ≤ δ) :
    ¬ R H toyCm (fun _ => some toyStmt) (fun _ _ => some toyPart) δ x ⟨[]⟩ := by
  rintro ⟨s, P, hs, hP, h⟩
  cases hs
  cases hP
  change δ < toyStmt.design.Pa toyPart (faulty H toyCm toyStmt toyPart ⟨[]⟩) at h
  rw [toy_faulty, empty_val, toy_Pa] at h
  have hw : window 0 1 ({0} : Finset ℕ) ≠ ∅ := by decide
  simp only [ite_true, hw, ite_false] at h
  linarith

/-- The toy is in the sampling scope. -/
theorem inScope_toy : InScope toyPart toyStmt := by
  refine ⟨⟨fun g x hx i hi => ?_, by decide⟩, ⟨?_, ?_, ?_⟩, ?_⟩
  · rcases g with _ | g
    · simp [toyStmt, toyProg] at hx
      subst hx
      simp [BGate.reads] at hi
    · simp [toyStmt, toyProg] at hx
  · rw [toyPart_rus]
    decide
  · rw [toyPart_rus]
    intro r hr
    obtain rfl := List.mem_singleton.1 hr
    exact ⟨List.cons_ne_nil _ _, by simp, fun i h => absurd h (by simp)⟩
  · rw [toyPart_rus]
    decide
  · refine ⟨rfl, ?_⟩
    simp [toyStmt, toyPart, Rule.WellFormed, OneStage.WellFormed]

end Claims.Sampling.RedTeam
