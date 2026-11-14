module

public import Proofs.Properties.Verity.SampledTwo.InputMap
public import Proofs.Flock.Claims.Sampling.Select.Code
public import Proofs.Flock.Claims.Complete.Json.Record
public import Proofs.Flock.SpotFlock.Jeq
public import Spot.Run
import all Init.Data.Ord.String
import all Lean.Data.Json.Basic

/-!
# The second stage's law is computed from the first stage's accepted sample

The two-stage form's last step holds the interior audit's sample to a sampling design it computes (`secondDesignOf`), from the `draw`
the run of record returned. That `draw` is the first stage's sample of record:

* `run_draw`: a run of record whose symbol is `accept` ran the sample of record (`Select.select`, under
  `Select.resolve`'s rule and the work table the rule expands) on the bytes it was handed, held that sample to the
  registration (`Draw.checkDraw`, under the sampling design the rule gives the population), and its `draw` is that
  sample. A refused run names no symbol, and the entry a second stage adds to the answer is named `interior`.
* `accepted_secondLaw`: where the two-stage check accepts, the sampling design its last step holds the interior audit's sample to is
  `Select.secondLaw` of the proved units of the first stage's sample, the one `Select.select` drew on the first
  stage's bytes `ξ` under the boundary's parameters and the run of record held to the boundary registration.

Nothing imports this module.
-/

@[expose] public section

namespace VStar.Recursive.SampledTwo

open Lean (Json)
open Spot

/-- `getD` of `Json.mkObj l` at a key of `l`'s, its keys distinct, is its value. -/
private theorem getD_mkObj {l : List (String × Json)} (hd : FlockComplete.JsonRT.Distinct l) {k : String} {v : Json}
    (hm : (k, v) ∈ l) : Spot.getD (Json.mkObj l) k = v := by
  simp only [Spot.getD, Spot.field?, FlockComplete.JsonRT.getObjVal?_mkObj hd hm, Except.toOption, Option.getD_some]

/-- `getD` of `Json.mkObj l` at a key no entry of `l` has is `null`. -/
private theorem getD_mkObj_none {l : List (String × Json)} {k : String}
    (hk : (l.map Prod.fst).contains k = false) : Spot.getD (Json.mkObj l) k = Json.null := by
  first
    | (simp only [Spot.getD, Spot.field?, Json.mkObj, Json.getObjVal?, FlockComplete.JsonRT.get?_mkObj_none hk]
       all_goals rfl)
    | (unfold Spot.getD Spot.field? Json.mkObj Json.getObjVal?
       dsimp only
       rw [FlockComplete.JsonRT.get?_mkObj_none hk]
       all_goals rfl)

/-- **The entry a second stage adds to a run's answer** is none, or one named `interior`. -/
private theorem interiorAudit_staged {form : Bool} {i : Spot.Run.Input} {sample : Array Nat}
    {inner : Option (Spot.Run.Interior × Json × Json × Json × Nat)} {acc : Bool} {staged : List (String × Json)}
    (h : Spot.Run.interiorAudit form i sample inner = .ok (acc, staged)) :
    staged = [] ∨ ∃ e, staged = [("interior", e)] := by
  cases inner with
  | none =>
    first
      | (have h' : (Except.ok (true, []) : Except String (Bool × List (String × Json))) = .ok (acc, staged) := h
         cases h'
         exact Or.inl rfl)
      | (simp only [Spot.Run.interiorAudit, pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
         exact Or.inl h.2.symm)
      | (unfold Spot.Run.interiorAudit at h
         simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
         exact Or.inl h.2.symm)
  | some x =>
    obtain ⟨it, receipt, draw, law, read⟩ := x
    unfold Spot.Run.interiorAudit at h
    simp only [bind, Except.bind, pure, Except.pure] at h
    repeat'
      first
      | split at h
      | focus (cases h; done)
      | (simp only [Except.ok.injEq, Prod.mk.injEq] at h
         exact Or.inr ⟨_, h.2.symm⟩)

-- `run` unfolds to about 50 decided leaves under one `match` or `if` per step, hence the heartbeat budget.
set_option maxHeartbeats 8000000 in
/-- **A run's answer, at either switch** (`Spot.Run.run_decided`'s, with the second stage's audit): refused at a step,
or past the population, R1–R7, the sample of record on the bytes it was handed and that sample's own check, its answer
listing that sample as its `draw` before whatever the second stage's audit adds. -/
private theorem run_path {i : Spot.Run.Input} {form : Bool} {j : Json} (h : Spot.Run.run i form = .ok j) :
    (∃ (step : String) (codes : Array String),
      j = Json.mkObj [("step", Json.str step), ("codes", Json.arr (codes.map Json.str))]) ∨
    ∃ n rule law workTable, nat? (getD i.record "population") = some n ∧ ¬(n == 0) = true ∧ i.rule.over n
        (Select.strataSizes i.params.verifier.program i.params.verifier.query) = some rule ∧
      rule.lawOver n = .ok law ∧
      i.rule.tableOver i.params.verifier.program i.params.verifier.query i.params.verifier.workTable = .ok workTable ∧
      ¬(!(Registration.check i.record { i.params.verifier with law, workTable }).isEmpty) = true ∧
      ∃ derived, Draw.derive i.params.verifier.program i.params.verifier.query law workTable = .ok derived ∧
        ∃ resolved, Select.resolve i.rule i.params.verifier.program i.params.verifier.query = .ok resolved ∧
        ∃ draw rest read, Select.select resolved i.params.verifier.program i.params.verifier.query
            workTable i.closure i.source = .ok (draw, rest, read) ∧
          ¬(!(Draw.checkDraw draw (getD i.record "population") derived).isEmpty) = true ∧
          ∃ inner acc₂ staged receipt symbol pub audit,
            Spot.Run.interiorAudit form i ((Draw.proved draw).map Int.toNat) inner = .ok (acc₂, staged) ∧
            j = Json.mkObj ([("receipt", receipt), ("draw", draw), ("read", Lean.toJson read), ("symbol", symbol),
              ("public", pub), ("audit", audit)] ++ staged) := by
  unfold Spot.Run.run at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat'
    first
    | split at h
    | focus (cases h; done)
    | (obtain ⟨_, h⟩ | ⟨_, h⟩ := Spot.Run.ite_eq_ok h; exact Or.inl ⟨_, _, h.symm⟩)
    | exact Or.inl ⟨_, _, (Except.ok.inj h).symm⟩
    | exact Or.inr ⟨_, _, _, _, by assumption, by assumption, by assumption, by assumption, by assumption,
        by assumption, _, by assumption, _, by assumption, _, _, _, by assumption, by assumption, _, _, _, _, _, _, _,
        by assumption, (Except.ok.inj h).symm⟩

/-- **An accepted run of record's `draw` is its first stage's sample**: a run of `Spot.Run.run`, at either switch,
whose symbol is `accept` drew the sample of record (`Select.select` under `Select.resolve`'s rule and the work table
`Rule.tableOver` expands) on the bytes it was handed, held it to the registration (`Draw.checkDraw` under the
sampling design `Rule.over` gives the population), and lists it as its `draw`. -/
theorem run_draw {i : Spot.Run.Input} {form : Bool} {run : Json} (h : Spot.Run.run i form = .ok run)
    (ha : Spot.jeq (Spot.getD run "symbol") (Json.str "accept") = true) :
    ∃ n rule law workTable derived resolved draw rest read,
      nat? (getD i.record "population") = some n ∧
      i.rule.over n (Select.strataSizes i.params.verifier.program i.params.verifier.query) = some rule ∧
      rule.lawOver n = .ok law ∧
      i.rule.tableOver i.params.verifier.program i.params.verifier.query i.params.verifier.workTable = .ok workTable ∧
      Draw.derive i.params.verifier.program i.params.verifier.query law workTable = .ok derived ∧
      Select.resolve i.rule i.params.verifier.program i.params.verifier.query = .ok resolved ∧
      Select.select resolved i.params.verifier.program i.params.verifier.query workTable i.closure i.source =
        .ok (draw, rest, read) ∧
      (Draw.checkDraw draw (getD i.record "population") derived).isEmpty = true ∧
      Spot.getD run "draw" = draw := by
  rcases run_path h with ⟨step, codes, hj⟩ | ⟨n, rule, law, workTable, hn, -, hrule, hlaw, htab, -, derived, hder,
      resolved, hres, draw, rest, read, hsel, hchk, inner, acc₂, staged, receipt, symbol, pub, audit, hia, hj⟩
  · exfalso
    have hs : Spot.getD run "symbol" = Json.str "accept" := SpotFlock.str_of_jeq (SpotFlock.jeq_symm ha)
    have hn : Spot.getD (Json.mkObj [("step", Json.str step), ("codes", Json.arr (codes.map Json.str))]) "symbol" =
        Json.null := getD_mkObj_none (by simp)
    rw [hj, hn] at hs
    cases hs
  · refine ⟨n, rule, law, workTable, derived, resolved, draw, rest, read, hn, hrule, hlaw, htab, hder, hres, hsel,
      ?_, ?_⟩
    · have hb : ∀ b : Bool, ¬(!b) = true → b = true := by
        intro b hb
        cases b
        · exact absurd (rfl : (!false) = true) hb
        · rfl
      exact hb _ hchk
    · rw [hj]
      rcases interiorAudit_staged hia with rfl | ⟨e, rfl⟩
      · exact getD_mkObj (by simp [FlockComplete.JsonRT.Distinct, List.pairwise_cons, Std.compare_eq_iff_eq])
          (by simp)
      · exact getD_mkObj (by simp [FlockComplete.JsonRT.Distinct, List.pairwise_cons, Std.compare_eq_iff_eq])
          (by simp)

/-- **The second stage's law is computed from the first stage's accepted sample.** Where the two-stage check accepts,
the run of record's first stage drew `draw` on the first stage's bytes `ξ` (`Select.select`, under the rule
`Select.resolve` gives the parameters' rule at the boundary's program and query, with no closure), held it to the
boundary registration's record `j₁` as the check read it (`readBack`; `Draw.checkDraw`, under the sampling design the
rule gives its population), and listed it as
its `draw`; and the sampling design the last step held the interior audit's sample to (`secondCheck` at `law`) is
`Select.secondLaw` of `draw`'s proved units, over the replay block's cells under the interior's query at the rule's
counts. -/
theorem accepted_secondLaw {form : Bool} {p : Params} {c : OneStage.Commitment} {ξ : ByteArray} {e : Evidence}
    {bs : Core.Blobs} {pub : OneStage.Public} {prof : Core.Integrity}
    (h : checkBlobsAtWith form p c ξ e bs = .accept pub prof) :
    ∃ c₂ ξ₂ r i run law, inputAt p c ξ e bs = .ok (c₂, ξ₂, r, i) ∧ Spot.Run.run i true = .ok run ∧
      secondCheck p.audit.base.verifier p.audit.base.delta c₂ law ξ₂ { e.interior.base with verdicts := r.out } =
        .accept pub prof ∧
      ∃ j₁ n rule law₁ workTable derived resolved draw rest read, readBack c.record = .ok j₁ ∧
        nat? (getD j₁ "population") = some n ∧
        p.rule.over n (Select.strataSizes p.boundary.verifier.program p.boundary.verifier.query) = some rule ∧
        rule.lawOver n = .ok law₁ ∧
        p.rule.tableOver p.boundary.verifier.program p.boundary.verifier.query p.boundary.verifier.workTable =
          .ok workTable ∧
        Draw.derive p.boundary.verifier.program p.boundary.verifier.query law₁ workTable = .ok derived ∧
        Select.resolve p.rule p.boundary.verifier.program p.boundary.verifier.query = .ok resolved ∧
        Select.select resolved p.boundary.verifier.program p.boundary.verifier.query workTable none ξ =
          .ok (draw, rest, read) ∧
        (Draw.checkDraw draw (getD j₁ "population") derived).isEmpty = true ∧
        Spot.getD run "draw" = draw ∧
        ∃ grain replay k ns ids ks, p.rule = .twoStage grain replay k ∧
          Select.replayCells p.replay p.interior.verifier.query = .ok ns ∧
          Select.templateIds p.interior.verifier.query = .ok ids ∧ k.over ids ns = .ok ks ∧
          law = Select.secondLaw ((Draw.proved draw).map Int.toNat) ns ks := by
  obtain ⟨-, c₂, ξ₂, r, i, run, hi, hr, -, -, hj, law, hl, htwo⟩ := accepted_steps h
  obtain ⟨-, -, -, -, vs, received₂, j₁, j₂, -, hio, hj₁, -⟩ := inputAt_ok hi
  obtain ⟨n, rule, law₁, workTable, derived, resolved, draw, rest, read, hn, hrule, hlaw, htab, hder, hres, hsel,
    hchk, hd⟩ := run_draw hr hj
  obtain ⟨grain, replay, k, ns, ids, ks, hk, hns, hids, hks, hlaw₂⟩ := secondLawOf_ok hl
  rw [hd] at hlaw₂
  obtain ⟨h1, h2, h3, h4, h5⟩ := inputOf_first p { c with record := j₁ } ξ ξ₂ { e with record := j₂ } received₂ vs
  rw [hio] at hn hrule htab hder hres hsel hchk
  rw [h3] at hn hchk
  rw [h1, h2] at hrule htab hres
  rw [h2] at hder
  rw [h2, h4, h5] at hsel
  exact ⟨c₂, ξ₂, r, i, run, law, hi, hr, htwo, j₁, n, rule, law₁, workTable, derived, resolved, draw, rest, read, hj₁,
    hn, hrule, hlaw, htab, hder, hres, hsel, hchk, hd, grain, replay, k, ns, ids, ks, hk, hns, hids, hks, hlaw₂⟩

end VStar.Recursive.SampledTwo
