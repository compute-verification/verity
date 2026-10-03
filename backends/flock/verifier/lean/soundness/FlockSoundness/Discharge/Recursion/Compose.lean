import FlockSoundness.Discharge.Recursion.Defs

/-!
# Recursion: commit-and-prove composition

`commit_and_prove` bounds the recursive audit's soundness error, `Pr[the outer session accepts ∧ P's statement for the
drawn units is false]`, by four terms, given **reference openings** `ref ω p c`, one message for each commitment `c`
sent after the draw `ω` and the transcript `p`:

* `εo`, outer soundness: the outer session accepts but the witness decoded from the prover at its start (`dec`) does not
  make the inner verifier accept its own statement and messages on the transcript's public coins;
* `εP`, the circuit's binding: the decoded statement is not `x R ω`, the statement of the circuit registered behind
  `c_P` for the draw;
* `εb k`, stage `k`'s binding: the decoded round-`k` message is not the reference opening of `c_k`;
* `εin`, inner soundness (`InnerSound`) at the simulated inner prover (`refStrat`), which sends `ref p c_k` at the node
  where the recursive prover sends `c_k`, before the coin `ρ_k`.

Off those events the decoded messages are the reference openings, so the inner verifier accepts `x R ω` on the
relabelled transcript (`zip_eq_relabel`), which is the simulated inner prover's (`leafAvg_relabel`). The reference
openings are anything the hypotheses hold for: `Recursion.Fork` builds them, as the plurality opening at each node, and
bounds `εb k` under `cr/sha-512`.
-/

namespace FlockSoundness.Discharge.Recursion

open Game Audit
open scoped ENNReal

variable {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type} {β : Type}
  {outer : Reg → L.Ω → List (Cm × I.Coin) → Game β}

/-- **Commit-and-prove composition.** `Acc R ω t b`: the outer session accepts with output `b` after the registration
`R`, the draw `ω` and the round transcript `t`; `dec` decodes a witness from the prover's strategy for the outer session,
with its statement `wstmt` and round messages `wmsg`. -/
theorem commit_and_prove (σ : Strategy (recGame L Reg I Cm outer))
    (Acc : Reg → L.Ω → List (Cm × I.Coin) → β → Prop) (x : Reg → L.Ω → I.Stmt) {W : Type}
    (dec : ∀ R ω t, Strategy (outer R ω t) → W) (wstmt : W → I.Stmt) (wmsg : W → List I.Msg)
    (ref : L.Ω → List (Cm × I.Coin) → Cm → I.Msg) {εo εP εin : ℝ≥0∞} (εb : ℕ → ℝ≥0∞)
    (hout : probAt σ (fun ω t s b => Acc σ.1 ω t b ∧
      ¬ I.accepts (wstmt (dec σ.1 ω t s)) ((wmsg (dec σ.1 ω t s)).zip (t.map Prod.snd))) ≤ εo)
    (hP : probAt σ (fun ω t s b => Acc σ.1 ω t b ∧ wstmt (dec σ.1 ω t s) ≠ x σ.1 ω) ≤ εP)
    (hb : ∀ k < I.r, probAt σ (fun ω t s b => Acc σ.1 ω t b ∧
      (wmsg (dec σ.1 ω t s))[k]? ≠ (t[k]?).map fun e => ref ω (t.take k) e.1) ≤ εb k)
    (hin : InnerSound I εin) :
    prob (fun o => Acc o.1 o.2.1 o.2.2.1 o.2.2.2 ∧ ¬ I.holds (x o.1 o.2.1)) (recGame L Reg I Cm outer) σ ≤
      εo + εP + ∑ k ∈ Finset.range I.r, εb k + εin := by
  sorry

end FlockSoundness.Discharge.Recursion
