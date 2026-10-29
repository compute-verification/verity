import Proofs.Flock.Claims.Prob
import Definitions.Claims.Sampling

/-!
# `Complete` can fail

`Complete` is at a nonempty tape type, and Flock's and the sampling claim's are of the deployed record's reference
prover (`D.P`). So none holds of every verifier: at a hash whose outputs are 64 bytes, a verifier that rejects every
run fails it wherever a statement in scope has a witness in the relation, at every `δ_c < 1` there.

* `rejV`: the verifier of one round that rejects; `not_complete_rejV`: it fails `Claims.Complete` at every honest
  prover on a nonempty tape type.
* `zeroHash`: the hash whose every output is 64 zero bytes (`out64_zeroHash`).
* `Claims.Complete_fails`: `Claims.Complete` fails at `rejV` and `zeroHash`, at every such prover, relation, scope and
  `δ_c`.
* `Claims.Flock.Complete_fails`: Flock's fails at `rejD`, a plain record of `rejV` whose parser reads every message as
  `inStmt`, a statement in Flock's scope (`inScope_inStmt`) whose relation every witness meets (`relS_inStmt`).
* `Claims.Sampling.Complete_fails`: the sampling claim fails at `rejSD`, a record of `rejV` whose parser reads every
  message as `nilStmt`, the empty program, and whose partition is the empty one: in scope (`inScope_nilStmt`), with no
  unit, so no unit is faulty at any witness (`rhon_rejSD`).

The three are the lock's `claims` witnesses, stated over the spec: each quantifies (`∃`) over the hash and the toy it
fails at.
-/

namespace Claims.CompleteFails

open Claims

/-- The verifier of one round that rejects every run. -/
def rejV (Coin : Type) : Verifier Coin := ⟨fun _ => 1, fun _ _ _ => .ret .rej⟩

theorem run_rejV {Ω Coin : Type} [Inhabited Coin] (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω)
    (r : Fin ((rejV Coin).k x) → Coin) : (run H P (rejV Coin) x ω r).out = .rej := by
  simp [run, stepWith, rejV, OC.eval]

/-- **`rejV` is not complete**, at a 64-byte hash, for any honest prover on a nonempty tape type, wherever a statement
in scope has a witness in the relation and `δ_c < 1`. -/
theorem not_complete_rejV {Coin W Ω : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [Nonempty Ω]
    {H : Msg → Msg} (hH : Out64 H) (P : W → Prover Ω) {R : Rel W} {Scope : Msg → Prop} {δc : Msg → ℝ} {x : Msg}
    {w : W} (hs : Scope x) (hr : R x w) (hδ : δc x < 1) : ¬ Claims.Complete H (rejV Coin) P R Scope δc := by
  intro h
  have h1 := h hH x w hs hr (Classical.arbitrary Ω)
  have h0 : Pr (fun r : Fin ((rejV Coin).k x) → Coin =>
      (run H (P w) (rejV Coin) x (Classical.arbitrary Ω) r).out = .acc) = 0 := by
    rw [Pr_congr (q := fun _ => False) fun r => by simp [run_rejV]]
    exact Pr_false
  linarith

/-- The hash whose every output is 64 zero bytes. -/
def zeroHash : Msg → Msg := fun _ => List.replicate 64 0

theorem out64_zeroHash : Out64 zeroHash := fun _ => List.length_replicate

/-- The prover that sends the empty message every round, on the one tape. -/
def silent (W : Type) : W → Prover Unit := fun _ => ⟨fun _ _ _ => .ret []⟩

/-! ## Flock -/

/-- **A statement in Flock's plain scope**: `m = 22`, `kLog = 6`, the circuit with no gates and no outputs, and a form
the scope admits. -/
def inStmt : Flock.Stmt where
  m := 22
  kLog := 6
  C := ⟨0, [], []⟩
  inp _ := []
  out _ := []
  pub := []
  form := { hm96Rows := true, typed := false, retainedRounds := true, leaf := "hm96-sha512/v1", tables := 1,
            publicInputs := false, sharedRows := false, registeredReads := false, hiddenOutputs := false,
            unitInputGroups := 1, tail := false, unitLog := 0, readsOnlyPortBits := true, rowsV1 := true,
            rowsV2OrSeg := false, wiresAcyclic := false }

theorem inScope_inStmt : Flock.InScope false inStmt := by
  unfold Flock.InScope
  decide

theorem relS_inStmt (w : Flock.Wit) : Flock.RelS inStmt w := fun _ _ => rfl

/-- **A plain record of `rejV`**: its parser reads every message as `inStmt`, its reference prover is `silent`, and its
error `0`. -/
def rejD : Flock.Deploy Unit Unit Unit where
  zk := false
  V := rejV Unit
  parse _ := some inStmt
  scopeCheck _ := true
  table _ _ := some inStmt
  cols _ _ := []
  enc _ _ := ()
  len _ := 0
  θ _ := 0
  dec _ := ([], 0)
  read _ := ⟨[]⟩
  εtable _ := 0
  N₀ _ := 0
  weight _ := 0
  waits _ := 0
  cap _ := 0
  Tape := Unit
  P := silent Flock.Wit
  δc _ _ := 0

/-! ## Sampling -/

/-- **The empty sampling statement**: no gates, no outputs, no rules, no roots. -/
def nilStmt : Sampling.AuditStatement := ⟨[], ⟨[], []⟩, ⟨"", 0, []⟩, ⟨[]⟩, []⟩

theorem inScope_nilStmt : Sampling.InScope ⟨[]⟩ nilStmt := by
  refine ⟨⟨fun _ _ h => by simp [nilStmt] at h, fun _ h => by simp [nilStmt] at h⟩, ⟨?_, ?_, ?_, ?_, ?_, ?_⟩,
    ⟨rfl, ?_⟩⟩ <;> simp [Sampling.Partition.rus, Sampling.Partition.gates, nilStmt]

/-- **A sampling record of `rejV`**: its parser reads every message as `nilStmt`, its partition is the empty one, its
sessions are `rejD`'s, its reference prover is `silent`, and its error `0`. -/
def rejSD : Sampling.Deploy Unit Unit Unit where
  V := rejV Unit
  parse _ := some nilStmt
  scopeCheck _ := true
  partition _ := some ⟨[]⟩
  commit := ⟨fun _ _ => .ret true, fun _ _ _ => []⟩
  universal _ := ⟨[], []⟩
  flock := rejD
  stage2 _ := ∅
  checked _ _ := []
  rounds _ _ := ∅
  session _ _ _ := none
  reveal _ _ _ _ := []
  innerWaits _ := 0
  innerN₀ _ := 0
  innerCap _ := 0
  εtable _ := 0
  weight _ := 0
  openCalls _ := 0
  Tape := Unit
  P := silent Sampling.Wit
  δc _ _ := 0

theorem rhon_rejSD (H : Msg → Msg) (x : Msg) (w : Sampling.Wit) :
    Sampling.Rhon H rejSD.commit rejSD.parse rejSD.partition x w :=
  ⟨nilStmt, ⟨[]⟩, rfl, rfl, by simp [Sampling.faulty, Sampling.Partition.rus]⟩

end Claims.CompleteFails

open Claims.CompleteFails

/-- **`Claims.Complete` can fail**: at every prover with a tape and every `δ_c`, wherever a statement in scope has a
witness and `δ_c < 1`, the verifier that rejects every run fails it at the 64-byte `zeroHash`. -/
theorem Claims.Complete_fails {Coin W Ω : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [Nonempty Ω]
    (P : W → Prover Ω) (R : Rel W) (Scope : Msg → Prop) (δc : Msg → ℝ) {x : Msg} {w : W} (hs : Scope x)
    (hr : R x w) (hδ : δc x < 1) : ∃ (H : Msg → Msg) (V : Verifier Coin), ¬ Claims.Complete H V P R Scope δc :=
  ⟨zeroHash, rejV Coin, not_complete_rejV out64_zeroHash P hs hr hδ⟩

/-- **Flock's `Complete` can fail**: at `zeroHash`, the record `rejD`'s reference prover is not accepted at the empty
message, a statement in its scope whose relation every witness meets, at its error `δ_c = 0`. -/
theorem Claims.Flock.Complete_fails :
    ∃ (H : Msg → Msg) (D : Claims.Flock.Deploy Unit Unit Unit), ¬ Claims.Flock.Complete H D :=
  ⟨zeroHash, rejD, not_complete_rejV out64_zeroHash rejD.P (x := []) (w := ⟨[]⟩) ⟨inStmt, rfl, inScope_inStmt⟩
    ⟨inStmt, rfl, relS_inStmt _⟩ (by show (0 : ℝ) < 1; norm_num)⟩

/-- **The sampling `Complete` can fail**: at `zeroHash`, the record `rejSD`'s reference prover is not accepted at the
empty message, in scope with no faulty unit, at its error `δ_c = 0`. -/
theorem Claims.Sampling.Complete_fails :
    ∃ (H : Msg → Msg) (D : Claims.Sampling.Deploy Unit Unit Unit), ¬ Claims.Sampling.Complete H D :=
  ⟨zeroHash, rejSD, not_complete_rejV out64_zeroHash rejSD.P (x := []) (w := ⟨[]⟩)
    ⟨nilStmt, ⟨[]⟩, rfl, rfl, inScope_nilStmt⟩ (rhon_rejSD _ _ _) (by show (0 : ℝ) < 1; norm_num)⟩
