import Definitions.Claims.Recursion

/-!
# Claims: the recursion's zero knowledge if the ZK server or the workers are honest (a draft, before it freezes)

The recursion's prover (`Claims.Recursion`) split in two, with the coin server beside them:

* **The workers** hold the witness and run the inner C-Flock session in its masked interaction (`--zk`, the masked
  record `Flock.ZkDeploy`), on masks, pads and salts of their own. They hear only the coin server, and send only
  through a data diode to the ZK server.
* **The ZK server** is the only exit. It draws the coin tree's nonce and sends `Hello`. It commits each of the workers'
  messages under fresh salts of its own, as many commit strings a round as the schedule fixes, a missing message as
  zero fill. It checks the inner transcript and every V* statement before the first outer session. Then it runs the
  outer `--zk` sessions, or sends one refusal at one fixed position. Nothing goes from it to the workers.
* **The coin server** sends each live coin to both, a coin only after the ZK server's commit strings of its round.

Two claims, one for each honest party; `EitherZK` is both.

1. **`ServerZK`: an honest ZK server, any workers.**
   - The workers are any program (`Workers`): their next message is any function of the release's history so far, the
     ZK server's messages and the verifier's answers, which a colluding verifier may relay to them. They never read the
     ZK server's tape.
   - Against every malicious verifier of the compiled verifier's rounds, every event of the transcript is at most twice
     as likely at the record's simulator (`SplitDeploy.S`), which reads neither the witness nor the workers.
   - The error is the outer sessions' `--zk` simulation errors and `2δ₁` per commit string (`εzk`), plus `γzk` times
     the probability that `Fork` finds a collision among the release's SHA-512 calls in two real runs.
   - The factor `2` is the release's two forms, the whole release or the one refusal, which reads the workers'
     messages: a release leaks at most one bit.
2. **`WorkersZK`: honest workers, any ZK server.**
   - The ZK server and the verifier together are a malicious verifier of the workers' masked session, so their joint
     view is that session's transcript: `ZKmalR` at the workers' reference prover (`SplitDeploy.workers`).
   - That prover is the masked record's, with the coin tree's nonce read from the coin server's answer to `Hello`
     (`SplitDeploy.helloAns`), since the ZK server draws it, where `Flock.ZK` reads the prover's own.
   - The simulator is the record's for it (`SplitDeploy.SW`).

**Scope: the witness.** Both claims hide what the statement doesn't hold: the values (weights, data, intermediates),
their openings and their salts. The statement is public: the registered roots, the public layout, the shape and the
size classes, and the circuit where it is public. The architecture is hidden only where the inner proof hides the
circuit. At the universal unit's statements (`Claims.Recursion.RPriv`), the circuit is a committed value of the inner
witness, hidden up to its size class, which the statement holds. `ServerZK` holds at every statement, `WorkersZK` at
every statement and witness of the inner relation.

**Premises.**
- `Hm96Hiding` at the recursion's pinned key, which `ServerZK` reads as `Claims.Recursion.ZK` does; `WorkersZK`'s
  errors are its argument, as `Flock.ZK`'s are.
- The coin tree's binding is the `Fork` term in both. In `ServerZK` it is at the ZK server's own nonce. In `WorkersZK`
  the nonce is one the coalition chooses, so there it is SHA-512's collision resistance at a chosen key.
- Timing is outside both. That the ZK server's release times are fixed, so the workers' latency doesn't reach the view,
  is its deployment's.

**Relations.**
- `Claims.Recursion.ZK`, `ZKPriv` and `ZKStop` are the recursion's zero knowledge with both parties honest, as one
  prover. `ServerZK` is `ZKStop`'s form at a prover split into the ZK server and any workers. Its one stop the
  simulator can't read, the ZK server's check, is paid by the factor `2`, as a one-sided bound.
- The proved cores, which neither claim is proved from yet, are the following.
  - `Flock.Guarantees.MaskedZK` is `Flock.ZK` at the deployed masked record. `WorkersZK` asks for that proof at a nonce
    from the coin server's answer.
  - `Flock.Guarantees.RecursiveZK` takes any inner messages under the ZK server's fresh salts, the outer sessions'
    simulator, and a stop of a public family, which leaks `log₂` of the family's size. `ServerZK` asks for it at the
    two release forms, over the compiled verifier's transcript.

**Status.** A draft: nothing here is frozen, no guarantee reads it, and it is in no lock.
-/

namespace Claims.Recursion.Either

open Claims
open scoped NNReal

/-- **A worker program** on the tapes `Ω`: the workers' next message from the release's history so far, the ZK server's
messages and the verifier's answers. It never reads the ZK server's tape: nothing goes from the ZK server to the
workers, and the history is everything a colluding verifier could relay to them. -/
abbrev Workers (Ω : Type) : Type := Ω → List (Msg × Msg) → Msg

/-- **The split deployment**: the recursion's record with the reference prover's public stop (`StopDeploy`), and the
following.

* `inner`: the workers' masked record, the inner C-Flock session's.
* `serve x ws`: the ZK server's program at the statement `x`, on its own nonce and rest, given the workers' next message
  at each history (`ws`). It sends `Hello`, then per inner round the round's `nAt` commit strings to the workers'
  message under its own salts. Before the first outer session it checks the inner transcript and every V* statement.
  Then come the staging rounds and the outer `--zk` sessions, or the one refusal.
* `helloAns`: the coin server's answer to `Hello`, read as the coin tree's nonce and its commitment to every coin.
* `S t`: `ServerZK`'s simulator at the attempt budget `t`. It reads the statement and the verifier, never the workers.
* `SW`: `WorkersZK`'s simulator. -/
structure SplitDeploy (Coin Col T : Type) extends StopDeploy Coin where
  inner : Flock.ZkDeploy Coin Col T
  serve : (x : Msg) → (List (Msg × Msg) → Msg) → Prover (Nonce x × Rest x)
  helloAns : Msg → inner.Nonce × Msg
  S : ℕ → Simulator
  SW : Simulator

section Split

variable {Coin Col T Col' T' : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **The release** at the statement `x` with the workers `ws`: the ZK server's program at `x`, reading the workers'
messages from `ws` at their tape. Its tape is the ZK server's nonce and rest and the workers' tape. -/
def SplitDeploy.release {Ω : Type} (D : SplitDeploy Coin Col T) (x : Msg) (ws : Workers Ω) :
    Prover (D.Nonce x × (D.Rest x × Ω)) :=
  ⟨fun x' hist ω => (D.serve x (ws ω.2.2)).next x' hist (ω.1, ω.2.1)⟩

/-- **The workers' reference prover** in their masked session at the witness `w` and the rest `r`. Their first message
is empty, since the ZK server sends `Hello`. The verifier's answer to it is read as the coin tree's nonce `ν` and its
commitment `ρ` (`helloAns`). Then, if every opening so far passes the masked record's check against `ρ` under `ν`, it
is the masked record's reference prover at the history with each opening read as its coin, at the tape `(ν, r)`; else
`abort`. Its nonce is `PUnit`: the workers draw none. -/
def SplitDeploy.workers (D : SplitDeploy Coin Col T) (w : Flock.Wit) : Prover (PUnit × D.inner.Rest) :=
  ⟨fun x hist ω => match hist with
    | [] => .ret []
    | (_, a) :: ops =>
      (Flock.allPass (D.inner.chk x (D.helloAns a).1 (D.helloAns a).2) 0 (ops.map Prod.snd)).bind fun ok =>
        if ok then
          (D.inner.P w).next x (ops.map fun p => (p.1, D.inner.coinOf p.2)) (D.inner.tapeOf ((D.helloAns a).1, ω.2))
        else .ret D.inner.abort⟩

/-- **`ServerZK`'s error** at the attempt budget `t`, where hm96 leaves hide up to `δ₁`. It is each V* session's `--zk`
simulation error (`Nums.εzk`), and `2δ₁` for each commit string the ZK server sends before the outer sessions. Those
are the inner rounds' `nAt` (the `Hello` round first) and the staging messages' `k_S(x) · n_C(x)`, as `εzkStop`
counts them. -/
noncomputable def εzk (D : SplitDeploy Coin Col T) (t : ℕ) (δ₁ : ℝ) (x : Msg) : ℝ :=
  ∑ s : Fin (D.nS x), (D.nums x s).εzk t δ₁ +
    2 * (∑ i : Fin (D.inner.V.k (xI D.toDeploy x) + 1), D.nAt x i + D.kS x * D.nC x) * δ₁

/-- **An honest ZK server, any workers.** Against every worker program and every malicious verifier of the compiled
verifier's rounds, with the outer sessions in `--zk`, at each attempt budget `t ≥ 1` and wherever hm96 leaves hide up
to `δ₁` at the deployed key:
$$\Pr[\mathrm{Ev}(\text{release})] \le 2\cdot\Pr[\mathrm{Ev}(S_t)] + \varepsilon_{zk}(t,\delta_1,x) + \gamma_{zk}(t,x)\cdot\Pr[\mathrm{Fork}].$$
It holds at every statement: the workers needn't hold a witness. -/
def ServerZK (H : Msg → Msg) (D : SplitDeploy Coin Col T) (DO : Flock.Deploy Coin Col' T') : Prop :=
  DO.zk = true ∧ ∀ t, 0 < t → ∀ δ₁ : ℝ≥0, Hm96Hiding D.mask (D.toDeploy.saltHash H) δ₁ →
    ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (ws : Workers Ω) (Θ : Type) (Vs : VStar Θ) (θ : Θ),
      Vs.k x = D.V.k x → ∀ Ev : List (Msg × Out) → Prop,
        Pr (fun ω : D.Nonce x × (D.Rest x × Ω) => Ev (runStar H (D.release x ws) Vs x ω θ).msgs) ≤
          2 * Pr (fun c : (D.S t).C x => Ev ((D.S t).out H Vs θ x c)) + εzk D t δ₁ x +
            γzk D.toDeploy t x * (Fork H (D.release x ws) Vs θ x).Pr H

/-- **Honest workers, any ZK server.** `ZKmalR` at the workers' reference prover, against every malicious verifier of
the masked session's `k + 1` rounds (the `Hello` round, then the inner verifier's), which the ZK server and the
verifier play together, at the record's simulator `SW`, within `εW` plus `γW` times the probability that `Fork` finds a
collision among the workers' SHA-512 calls in two real runs. -/
def WorkersZK (H : Msg → Msg) (D : SplitDeploy Coin Col T) (εW γW : Msg → ℝ) : Prop :=
  D.inner.zk = true ∧ ZKmalR H (fun x => D.inner.V.k x + 1) D.workers (Flock.R D.inner.parse) D.SW εW γW

/-- **Zero knowledge if the ZK server or the workers are honest**: `ServerZK` and `WorkersZK`. -/
def EitherZK (H : Msg → Msg) (D : SplitDeploy Coin Col T) (DO : Flock.Deploy Coin Col' T') (εW γW : Msg → ℝ) :
    Prop :=
  ServerZK H D DO ∧ WorkersZK H D εW γW

end Split

end Claims.Recursion.Either
