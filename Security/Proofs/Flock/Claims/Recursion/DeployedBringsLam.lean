import Proofs.Flock.Claims.Recursion.DeployedBrings
import Proofs.Flock.Claims.Recursion.LamDeployed

/-!
# The deployed brings' setup in λ₂'s call bound

`BitsLin2` charges each finder `v`, a bound on the compiled verifier's own evaluations at a run's V\* statements
(`Claims.Recursion.setupCalls`). At `deployedB b` those are `deployedSetup`'s at each of the `nS x` statements, so `v`
is at most `nS x · c` when each statement's `deployedSetup` has at most `c` inputs (`setupCalls_deployedB_le`). At
`nS x ≤ 2^16` and `c ≤ 2^39`, `v ≤ 2^55` (`setupCalls_deployedB_le_pow`) meets `lamLin2_deployed_ge_dep`'s `hV`, so
`λ₂ ≥ 100` at `deployedB b` (`lamLin2_deployedB_ge`). The per-statement count `c` is a premise.
-/

namespace RecDeploy

open Claims (Msg Out)
open FlockDeploy (Coin)

theorem flatMap_length_le {α β : Type} (f : α → List β) (c : ℕ) :
    ∀ xs : List α, (∀ a ∈ xs, (f a).length ≤ c) → (xs.flatMap f).length ≤ xs.length * c
  | [], _ => by simp
  | a :: l, h => by
    rw [List.flatMap_cons, List.length_append, List.length_cons, Nat.add_mul, Nat.one_mul,
      Nat.add_comm (l.length * c)]
    exact Nat.add_le_add (h a (List.mem_cons.2 (Or.inl rfl)))
      (flatMap_length_le f c l fun b hb => h b (List.mem_cons_of_mem _ hb))

/-- **The compiled verifier's own evaluations at the deployed brings**: at most `nS x · c` at a run's V\* statements,
when each statement's `deployedSetup` has at most `c` inputs. -/
theorem setupCalls_deployedB_le (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) (IV : Claims.Verifier Coin) (H : Msg → Msg) (x : Msg)
    (ps : List (Msg × Out)) {c : ℕ} (hc : ∀ xO, (deployedSetup xO).length ≤ c) :
    (Claims.Recursion.setupCalls H (recBound rel (deployedB b) L IV)
      ((recBound rel (deployedB b) L IV).vstar x ps)).length ≤ nS x * c := by
  have h := flatMap_length_le (fun xO => (deployedSetup xO).map fun u => (u, H u)) c
    (vstar (deployedB b).st x ps) fun xO _ => by simpa only [List.length_map] using hc xO
  rwa [vstar_length] at h

/-- **`v ≤ 2^55` at the deployed brings**, at most `2^16` V\* statements of at most `2^39` setup inputs each:
`lamLin2_deployed_ge_dep`'s `hV`. -/
theorem setupCalls_deployedB_le_pow (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (b : Brings) (L : Claims.Recursion.VStarB.LayoutB) (IV : Claims.Verifier Coin) (H : Msg → Msg) (x : Msg)
    (hn : nS x ≤ 2 ^ 16) (hc : ∀ xO, (deployedSetup xO).length ≤ 2 ^ 39) (ps : List (Msg × Out)) :
    (Claims.Recursion.setupCalls H (recBound rel (deployedB b) L IV)
      ((recBound rel (deployedB b) L IV).vstar x ps)).length ≤ 2 ^ 55 :=
  (setupCalls_deployedB_le rel b L IV H x ps hc).trans
    ((Nat.mul_le_mul_right _ hn).trans (by norm_num))

/-- **Route 2C's `λ₂ ≥ 100` at the deployed brings**, at the call bound `v = 2^55` that
`setupCalls_deployedB_le_pow` gives `BitsLin2`'s `v` premise. -/
theorem lamLin2_deployedB_ge (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace)
    {x : Msg} {s : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    (hin : FlockDeploy.PlainIn (xIOf x) s spec sts)
    (hO : ∀ j < nS x, ∃ i spec' sts',
      PartIn (staged (deployedB b).st x (pick (deployedB b).st x j) j) i spec' sts')
    (hn : nS x ≤ 2 ^ 16) (hS : (innerSetupIns (xIOf x)).length ≤ 2 ^ 56) {n : ℕ} (hU : n ≤ 2 ^ 55) :
    100 ≤ Claims.Recursion.lamLin2
      ((recBoundAt rel (deployedB b) L).record (bdDeployed rel nextE colsE wE join rowIn final w).IVStar)
      (Claims.Recursion.Inner.ofFlock (bdDeployed rel nextE colsE wE join rowIn final w).deploy decI)
      FlockDeploy.KS.CpDep x n (2 ^ 55) :=
  lamLin2_deployed_ge_dep rel (deployedB b) L nextE colsE wE join rowIn final w decI hin hO hn hS hU le_rfl

end RecDeploy
