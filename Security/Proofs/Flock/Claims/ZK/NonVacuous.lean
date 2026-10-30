import Proofs.Flock.Claims.ZK.Congr
import Proofs.Flock.Claims.DeployOC

/-!
# `ZKmalR` is not vacuous

The form before charged the simulator's own SHA-512 calls (`simLog`): a simulator that hashes a colliding pair made the
bound's last term `1`, so the claim held at `ε_zk = 0` for every prover. The term is now `γ` times `Fork`'s success,
which reads only the honest prover's calls in two real runs. A prover that makes no calls has `Fork.Pr = 0` against
every verifier, so for it the claim is the plain bound, at every `γ`:

* `quietP m`: the prover that sends `m` every round and calls nothing; `fork_quietP`: `Fork`'s success at it is `0`.
* `accNow k`: the verifier of `k` rounds that accepts the first message, so the real transcript is `[(m, acc)]`.
* `not_zkmalR_of_out`: `ZKmalR` fails at `quietP m`, at every `γ` and every `ε_zk < 1` at a statement in the relation,
  for any simulator that never outputs that transcript; `not_zkmalR_collSim`, at `collSim u v`, the simulator that
  hashes `u`, then `v`, and outputs the empty transcript, whatever `u` and `v` are.
* `not_zkmalR_leak`: at a statement with two witnesses, the prover that sends its witness has no simulator at
  `ε_zk < 1/2`, at every `γ`.
* `toyD`: a masked record whose relation holds at every statement and witness; there Flock's `ZK` fails at `collSim`
  (`not_flockZK_collSim_toy`), and no simulator meets it for the witness-sending prover (`not_flockZK_leak_toy`).
  `not_flockZK_collSim` and `not_flockZK_leak` are the same at any record whose relation holds where they need it.
-/

namespace Claims.ZKMal

open Claims

/-- **The quiet prover**: it sends `m` every round and calls nothing. -/
def quietP (N Ω : Type) (m : Msg) : Prover (N × Ω) := ⟨fun _ _ _ => .ret m⟩

/-- The verifier of `k` rounds that accepts the first message. -/
def accNow (k : Msg → ℕ) : VStar Unit := ⟨k, fun _ _ _ => .ret .acc⟩

/-- **The colliding simulator**: it hashes `u`, then `v`, asks the verifier nothing and outputs the empty transcript. -/
def collSim (u v : Msg) : Simulator where
  C _ := Unit
  prog _ _ := .hash u fun _ => .hash v fun _ => .ret []

theorem Pr_true {α : Type} [Fintype α] [Nonempty α] : Pr (fun _ : α => True) = 1 := by
  unfold Pr; simp

theorem stepWith_quietP_pLog {N Ω : Type} (H : Msg → Msg) (m x : Msg) (ω : N × Ω) (vnext : List Msg → OC Out) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace),
      (stepWith H (quietP N Ω m) x ω vnext n hist as t).pLog = t.pLog
  | 0, _, _, _ => rfl
  | n + 1, hist, as, t => by
      simp only [stepWith]
      split
      · rw [stepWith_quietP_pLog]
        simp [quietP, OC.eval]
      · simp [quietP, OC.eval]
      · simp [quietP, OC.eval]

/-- **`Fork` finds nothing at the quiet prover**, against every verifier. -/
theorem fork_quietP {N Ω Θ : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] (H : Msg → Msg) (m : Msg)
    (Vs : VStar Θ) (θ : Θ) (x : Msg) : (Fork H (quietP N Ω m) Vs θ x).Pr H = 0 := by
  unfold Finder.Pr
  have h : ∀ s : (Fork H (quietP N Ω m) Vs θ x).S,
      ¬ ∃ u v, (Fork H (quietP N Ω m) Vs θ x).out s = some (u, v) ∧ Collision H u v := by
    rintro s ⟨u, v, hs, -⟩
    simp only [Fork, runStar, stepWith_quietP_pLog, List.append_nil, findColl] at hs
    cases hs
  rw [Pr_congr (q := fun _ => False) fun s => iff_false_intro (h s)]
  exact Pr_false

theorem runStar_quietP_accNow {N Ω : Type} (H : Msg → Msg) (m : Msg) (k : Msg → ℕ) (x : Msg) (hk : 0 < k x)
    (ω : N × Ω) : (runStar H (quietP N Ω m) (accNow k) x ω ()).msgs = [(m, .acc)] := by
  obtain ⟨n, hn⟩ : ∃ n, k x = n + 1 := ⟨k x - 1, by omega⟩
  simp [runStar, accNow, hn, stepWith, quietP, OC.eval]

/-- **The non-vacuity witness**: at the quiet prover against `accNow`, `ZKmalR` fails at every `γ`, for any simulator
that never outputs the real transcript, once `ε_zk < 1` at a statement in the relation. -/
theorem not_zkmalR_of_out {W N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] (H : Msg → Msg)
    (k : Msg → ℕ) (R : Rel W) (m : Msg) (S : Simulator) (εzk γ : Msg → ℝ) (x : Msg) (w : W) (hR : R x w)
    (hk : 0 < k x) (hε : εzk x < 1) (hS : ∀ c : S.C x, S.out H (accNow k) () x c ≠ [(m, .acc)]) :
    ¬ ZKmalR H k (fun _ => quietP N Ω m) R S εzk γ := by
  intro h
  have a := h x w hR Unit (accNow k) () rfl fun τ => τ = [(m, .acc)]
  simp only [fork_quietP, mul_zero, add_zero] at a
  have e₁ : Pr (fun ω : N × Ω => (runStar H (quietP N Ω m) (accNow k) x ω ()).msgs = [(m, .acc)]) = 1 := by
    rw [Pr_congr (q := fun _ => True) fun ω => iff_true_intro (runStar_quietP_accNow H m k x hk ω)]
    exact Pr_true
  have e₂ : Pr (fun c : S.C x => S.out H (accNow k) () x c = [(m, .acc)]) = 0 := by
    rw [Pr_congr (q := fun _ => False) fun c => iff_false_intro (hS c)]
    exact Pr_false
  rw [e₁, e₂, sub_zero, abs_one] at a
  linarith

/-- **The colliding simulator fails the fixed form**, whatever it hashes: its collision is no longer charged. -/
theorem not_zkmalR_collSim {W N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] (H : Msg → Msg)
    (k : Msg → ℕ) (R : Rel W) (u v : Msg) (εzk γ : Msg → ℝ) (x : Msg) (w : W) (hR : R x w) (hk : 0 < k x)
    (hε : εzk x < 1) : ¬ ZKmalR H k (fun _ => quietP N Ω []) R (collSim u v) εzk γ :=
  not_zkmalR_of_out H k R [] (collSim u v) εzk γ x w hR hk hε fun _ => by
    simp [Simulator.out, collSim, SP.eval]

/-- **A prover that sends its witness has no simulator** at `ε_zk < 1/2`, at every `γ`, at a statement with two
witnesses it sends apart. -/
theorem not_zkmalR_leak {W N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] (H : Msg → Msg)
    (k : Msg → ℕ) (R : Rel W) (f : W → Msg) (S : Simulator) (εzk γ : Msg → ℝ) (x : Msg) (w₁ w₂ : W) (h₁ : R x w₁)
    (h₂ : R x w₂) (hf : f w₁ ≠ f w₂) (hk : 0 < k x) (hε : εzk x < 1 / 2) :
    ¬ ZKmalR H k (fun w => quietP N Ω (f w)) R S εzk γ := by
  intro h
  have a₁ := h x w₁ h₁ Unit (accNow k) () rfl fun τ => τ = [(f w₁, .acc)]
  have a₂ := h x w₂ h₂ Unit (accNow k) () rfl fun τ => τ = [(f w₁, .acc)]
  simp only [fork_quietP, mul_zero, add_zero] at a₁ a₂
  have e₁ : Pr (fun ω : N × Ω => (runStar H (quietP N Ω (f w₁)) (accNow k) x ω ()).msgs = [(f w₁, .acc)]) = 1 := by
    rw [Pr_congr (q := fun _ => True) fun ω => iff_true_intro (runStar_quietP_accNow H (f w₁) k x hk ω)]
    exact Pr_true
  have e₂ : Pr (fun ω : N × Ω => (runStar H (quietP N Ω (f w₂)) (accNow k) x ω ()).msgs = [(f w₁, .acc)]) = 0 := by
    rw [Pr_congr (q := fun _ => False) fun ω => iff_false_intro (by
      rw [runStar_quietP_accNow H (f w₂) k x hk ω]
      simpa using fun e => hf e.symm)]
    exact Pr_false
  rw [e₁] at a₁
  rw [e₂] at a₂
  rw [abs_le] at a₁ a₂
  linarith [a₁.1, a₂.2]

/-! ## Flock's `ZK` -/

/-- **Flock's `ZK` fails at the colliding simulator**, at every `γ`, at any record whose relation holds somewhere with
`ε_zk < 1` there. -/
theorem not_flockZK_collSim {Coin Col T N Ω : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (u v : Msg)
    (εzk γ : Msg → ℝ) (x : Msg) (w : Flock.Wit) (hR : Flock.R D.parse x w) (hε : εzk x < 1) :
    ¬ Flock.ZKAt H D (fun _ => quietP N Ω []) (collSim u v) εzk γ :=
  fun h => not_zkmalR_collSim H _ _ u v εzk γ x w hR (Nat.succ_pos _) hε h.2

/-- A witness's hidden values as bytes. -/
def witMsg (w : Flock.Wit) : Msg := w.hid.map fun b => if b then 1 else 0

theorem witMsg_ne {w₁ w₂ : Flock.Wit} (h : w₁.hid ≠ w₂.hid) : witMsg w₁ ≠ witMsg w₂ := by
  have hinj : Function.Injective fun b : Bool => if b then (1 : UInt8) else 0 := by
    intro a b e
    cases a <;> cases b <;> first | rfl | exact absurd e (by decide)
  exact fun e => h (List.map_injective_iff.mpr hinj e)

/-- **The prover that sends its witness has no simulator for Flock's `ZK`** at `ε_zk < 1/2`, at every `γ`, at any record
whose relation has a statement with two witnesses of different hidden values. -/
theorem not_flockZK_leak {Coin Col T N Ω : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (εzk γ : Msg → ℝ)
    (x : Msg) (w₁ w₂ : Flock.Wit) (h₁ : Flock.R D.parse x w₁) (h₂ : Flock.R D.parse x w₂) (hw : w₁.hid ≠ w₂.hid)
    (hε : εzk x < 1 / 2) : ¬ ∃ S, Flock.ZKAt H D (fun w => quietP N Ω (witMsg w)) S εzk γ :=
  fun ⟨S, h⟩ => not_zkmalR_leak H _ _ witMsg S εzk γ x w₁ w₂ h₁ h₂ (witMsg_ne hw) (Nat.succ_pos _) hε h.2

/-- A statement whose relation holds at every witness: one block, no gates, no inputs or outputs. -/
def toyStmt : Flock.Stmt where
  m := 0
  kLog := 0
  C := ⟨0, fun g => g.elim0, fun g => g.elim0, ∅⟩
  inp _ _ := .pub 0
  out _ _ := .pub 0
  pub := []
  form := ⟨false, false, false, "", 0, false, false, false, false, 0, false, 0, false, false, false, false⟩

theorem toy_relS (w : Flock.Wit) : Flock.RelS toyStmt w := fun _ _ o => o.elim0

/-- **A masked record whose relation holds at every statement and witness.** -/
def toyD : Flock.Deploy Unit Unit Unit where
  zk := true
  V := ⟨fun _ => 0, fun _ _ _ => .ret .acc⟩
  parse _ := some toyStmt
  scopeCheck _ := false
  table _ _ := none
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
  P _ := ⟨fun _ _ _ => .ret []⟩
  δc _ _ := 0

theorem toyD_R (x : Msg) (w : Flock.Wit) : Flock.R toyD.parse x w := ⟨toyStmt, rfl, toy_relS w⟩

/-- **At `toyD`, Flock's `ZK` fails at the colliding simulator** at every `ε_zk < 1` and every `γ`. -/
theorem not_flockZK_collSim_toy {N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] (H : Msg → Msg)
    (u v : Msg) (εzk γ : Msg → ℝ) (x : Msg) (hε : εzk x < 1) :
    ¬ Flock.ZKAt H toyD (fun _ => quietP N Ω []) (collSim u v) εzk γ :=
  not_flockZK_collSim H toyD u v εzk γ x ⟨[]⟩ (toyD_R x _) hε

/-- **At `toyD`, the prover that sends its witness has no simulator** at `ε_zk < 1/2` anywhere, at every `γ`. -/
theorem not_flockZK_leak_toy {N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] (H : Msg → Msg)
    (εzk γ : Msg → ℝ) (x : Msg) (hε : εzk x < 1 / 2) :
    ¬ ∃ S, Flock.ZKAt H toyD (fun w => quietP N Ω (witMsg w)) S εzk γ :=
  not_flockZK_leak H toyD εzk γ x ⟨[]⟩ ⟨[true]⟩ (toyD_R x _) (toyD_R x _) (by simp) hε

/-! ## Flock's `ZK` at a masked record -/

theorem allPass_true_eval (H : Msg → Msg) :
    ∀ (j : ℕ) (ms : List Msg), (Flock.allPass (fun _ _ => OC.ret true) j ms).eval H = (true, [])
  | _, [] => rfl
  | j, _ :: ms => allPass_true_eval H (j + 1) ms

/-- **A masked record of `toyD` whose reference prover is quiet**: `Hello`, every message and the abort message are `m`,
every opening passes, the tape is `()`, and the simulator is `S`. -/
def toyZD (m : Msg) (S : Simulator) : Flock.ZkDeploy Unit Unit Unit :=
  { toyD with
    P := fun _ => ⟨fun _ _ _ => .ret m⟩
    Nonce := Unit
    Rest := Unit
    tapeOf := fun _ => ()
    hello := fun _ _ => m
    abort := m
    chk := fun _ _ _ _ _ => .ret true
    coinOf := id
    S := S }

/-- `toyZD`'s reference prover sends `m` and calls nothing, at every history. -/
theorem toyZD_prover (H : Msg → Msg) (m : Msg) (S : Simulator) (w : Flock.Wit) (x : Msg) (hist : List (Msg × Msg))
    (ω : Unit × Unit) : (((toyZD m S).prover w).next x hist ω).eval H = (m, []) := by
  cases hist with
  | nil => rfl
  | cons p ops =>
      obtain ⟨a, ρ⟩ := p
      show ((Flock.allPass (fun _ _ => OC.ret true) 0 (ops.map Prod.snd)).bind fun ok =>
        if ok then OC.ret m else OC.ret m).eval H = (m, [])
      rw [OC.eval_bind, allPass_true_eval]
      rfl

/-- **Flock's `ZK` fails at the masked record `toyZD m (collSim u v)`**, at every `ε_zk < 1` somewhere and every `γ`:
its reference prover is quiet, so its colliding simulator is charged nothing and outputs the wrong transcript. -/
theorem not_flockZK_toyZD (H : Msg → Msg) (m u v : Msg) (εzk γ : Msg → ℝ) (x : Msg) (hε : εzk x < 1) :
    ¬ Flock.ZK H (toyZD m (collSim u v)) εzk γ := fun h =>
  not_zkmalR_of_out (N := Unit) (Ω := Unit) H _ _ m (collSim u v) εzk γ x ⟨[]⟩ (toyD_R x _) (Nat.succ_pos _) hε
    (fun _ => by simp [Simulator.out, collSim, SP.eval])
    (zkmalR_congr_eq (fun w x hist ω _ => toyZD_prover H m _ w x hist ω) h.2)

/-- **Flock's `ZK` is not met at every masked record**, at `ε_zk = 0` and `γ = 0`: `toyZD [] (collSim [] [])` fails it.
-/
theorem not_forall_flockZK (H : Msg → Msg) :
    ¬ ∀ D : Flock.ZkDeploy Unit Unit Unit, D.zk = true → Flock.ZK H D 0 0 := fun h =>
  not_flockZK_toyZD H [] [] [] 0 0 [] one_pos (h _ rfl)

end Claims.ZKMal
