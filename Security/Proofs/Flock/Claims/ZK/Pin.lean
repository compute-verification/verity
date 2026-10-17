import Proofs.Flock.Claims.ZK.Congr
import Proofs.Flock.Claims.DeployCode

/-!
# Flock's `ZK` at the masked record's reference prover

`Claims.Flock.ZK` reads a masked record (`Claims.Flock.ZkDeploy`): its reference prover in the masked interaction
(`ZkDeploy.prover`, the record's `P` behind the `Hello` and the opening checks) and its simulator (`ZkDeploy.S`). The
proofs prove `Claims.Flock.ZKAt`, the same form at a committed-coin protocol's prover (`Proto.prover`). This module
carries one to the other, by `zkmalR_congr` (`Congr`):

* `zkOf`: the masked record at a protocol `G`, `FlockDeploy.deployZk` with `G`'s nonce, tape, `Hello`, abort and opening
  check, a reading of each opening as its coin (`coinOf`), the reference prover's tape at `G`'s (`tapeOf`), and a
  simulator.
* `Proto.Honest`: `G`'s message after the openings is the record's reference prover's (`ZkTables.honestP`) at those openings'
  coins and the tape `tapeOf`, and `G` makes no call that it doesn't. Where it holds, the record's prover sends
  `G.prover`'s messages and makes its calls (`zkOf_prover`), so `ZKAt` at `G.prover` is `Claims.Flock.ZK` at `zkOf`
  (`zk_zkOf`).
-/

namespace Claims.ZKMal

open Claims
open Claims.Flock (allPass)

local notation "𝔽" => FlockLevel3.GF128

/-! ## One check, two binds -/

theorem ocBind_eq_bind {α β : Type} (f : α → OC β) : ∀ o : OC α, ocBind o f = o.bind f
  | .ret _ => rfl
  | .query q k => by
      simp only [ocBind, OC.bind]
      exact congrArg (OC.query q) (funext fun r => ocBind_eq_bind f (k r))

theorem allOk_eq_allPass (acc : ℕ → Msg → OC Bool) : ∀ (j : ℕ) (ms : List Msg), allOk acc j ms = allPass acc j ms
  | _, [] => rfl
  | j, m :: ms => by
      simp only [allOk, allPass, ocBind_eq_bind, allOk_eq_allPass acc (j + 1) ms]

/-! ## The masked record at a committed-coin protocol -/

/-- **The masked record at a committed-coin protocol** `G`: `FlockDeploy.deployZk rel` with `G`'s nonce and tape, its
`Hello`, abort message and opening check, the coin each opening reads as (`coinOf`), the reference prover's tape at
`G`'s (`tapeOf`), and the simulator `S`. -/
noncomputable def zkOf (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (G : Proto Flock.Wit)
    (coinOf : Msg → Msg) (tapeOf : G.Nonce × G.Rand → FlockComplete.TapeZk) (S : Simulator) :
    Claims.Flock.ZkDeploy FlockDeploy.Coin (List 𝔽) Claims.Flock.ZkTrace :=
  { FlockDeploy.deployZk rel with
    Nonce := G.Nonce, Rest := G.Rand, tapeOf := tapeOf, hello := G.hello, abort := G.abort, chk := G.chk,
    coinOf := coinOf, S := S }

/-- **`G`'s messages are the reference prover's** within `G`'s rounds, at `H`: after the openings `ops` (the history
after `Hello`'s answer, at most `G.K x` of them), `G`'s message at the coins they open (`dec`) is `P w`'s at the history
with each opening read as its coin (`coinOf`), at the tape `tapeOf (ν, r)`, and its calls are among `P w`'s. -/
def Proto.Honest {Ω : Type} (G : Proto Flock.Wit) (H : Msg → Msg) (P : Flock.Wit → Prover Ω) (coinOf : Msg → Msg)
    (tapeOf : G.Nonce × G.Rand → Ω) : Prop :=
  ∀ x w ν r (ops : List (Msg × Msg)), ops.length ≤ G.K x →
    ((G.msg x w ν r (ops.map fun p => G.dec p.2)).eval H).1 =
        (((P w).next x (ops.map fun p => (p.1, coinOf p.2)) (tapeOf (ν, r))).eval H).1 ∧
      ((G.msg x w ν r (ops.map fun p => G.dec p.2)).eval H).2 ⊆
        (((P w).next x (ops.map fun p => (p.1, coinOf p.2)) (tapeOf (ν, r))).eval H).2

/-- **The masked record's reference prover sends `G.prover`'s messages and makes its calls** within `G`'s rounds, where
`G`'s messages are the reference prover's (`Proto.Honest` at `ZkTables.honestP`): both send `Hello`, check the openings with
`G.chk`, and abort alike; past the checks, `G`'s message is `ZkTables.honestP`'s. -/
theorem zkOf_prover (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (G : Proto Flock.Wit)
    (coinOf : Msg → Msg) (tapeOf : G.Nonce × G.Rand → FlockComplete.TapeZk) (S : Simulator) (H : Msg → Msg)
    (hG : G.Honest H FlockComplete.ZkTables.honestP coinOf tapeOf) (w : Flock.Wit) (x : Msg) (hist : List (Msg × Msg))
    (ω : G.Nonce × G.Rand) (hlen : hist.length < G.k x) :
    (((G.prover w).next x hist ω).eval H).1 = ((((zkOf rel G coinOf tapeOf S).prover w).next x hist ω).eval H).1 ∧
      (((G.prover w).next x hist ω).eval H).2 ⊆ ((((zkOf rel G coinOf tapeOf S).prover w).next x hist ω).eval H).2 := by
  cases hist with
  | nil => exact ⟨rfl, List.Subset.refl _⟩
  | cons p ops =>
    have hops : ops.length ≤ G.K x := by
      simp only [List.length_cons, Proto.k] at hlen
      omega
    obtain ⟨a, ρ⟩ := p
    have e₁ : (G.prover w).next x ((a, ρ) :: ops) ω = (allPass (G.chk x ω.1 ρ) 0 (ops.map Prod.snd)).bind
        fun ok => if ok then G.msg x w ω.1 ω.2 (ops.map fun p => G.dec p.2) else .ret G.abort := by
      show ocBind (allOk (G.chk x ω.1 ρ) 0 (ops.map Prod.snd)) _ = _
      rw [ocBind_eq_bind, allOk_eq_allPass]
    have e₂ : ((zkOf rel G coinOf tapeOf S).prover w).next x ((a, ρ) :: ops) ω =
        (allPass (G.chk x ω.1 ρ) 0 (ops.map Prod.snd)).bind fun ok =>
          if ok then (FlockComplete.ZkTables.honestP w).next x (ops.map fun p => (p.1, coinOf p.2)) (tapeOf ω)
          else .ret G.abort := rfl
    rw [e₁, e₂, OC.eval_bind, OC.eval_bind]
    obtain ⟨h₁, h₂⟩ := hG x w ω.1 ω.2 ops hops
    cases ((allPass (G.chk x ω.1 ρ) 0 (ops.map Prod.snd)).eval H).1 with
    | true =>
        simp only [↓reduceIte]
        exact ⟨h₁, List.append_subset.mpr
          ⟨List.subset_append_left _ _, List.Subset.trans h₂ (List.subset_append_right _ _)⟩⟩
    | false => exact ⟨rfl, List.Subset.refl _⟩

/-- **`Claims.Flock.ZK` at the masked record of a committed-coin protocol**, from `ZKAt` at its prover: where `G`'s
rounds are `Hello` and the deployed verifier's (`deployZk_proto_k`) and its messages are the reference prover's
(`Proto.Honest`), at `γ ≥ 0`. -/
theorem zk_zkOf (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (G : Proto Flock.Wit)
    (coinOf : Msg → Msg) (tapeOf : G.Nonce × G.Rand → FlockComplete.TapeZk) (S : Simulator) {H : Msg → Msg}
    {εzk γ : Msg → ℝ} (hγ : ∀ x, 0 ≤ γ x) (hk : ∀ x, G.k x = (FlockDeploy.deployZk rel).V.k x + 1)
    (hG : G.Honest H FlockComplete.ZkTables.honestP coinOf tapeOf)
    (h : Claims.Flock.ZKAt H (FlockDeploy.deployZk rel) G.prover S εzk γ) :
    Claims.Flock.ZK H (zkOf rel G coinOf tapeOf S) εzk γ :=
  ⟨h.1, zkmalR_congr hγ
    (fun w x hist ω hh => (zkOf_prover rel G coinOf tapeOf S H hG w x hist ω (lt_of_lt_of_eq hh (hk x).symm)).1)
    (fun w x hist ω hh => (zkOf_prover rel G coinOf tapeOf S H hG w x hist ω (lt_of_lt_of_eq hh (hk x).symm)).2)
    h.2⟩

end Claims.ZKMal
