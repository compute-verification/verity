module

public import Proofs.Flock.Claims.Sampling.Session

/-!
# `SessionRuns` from the sampled verifier's step structure

`SessionSteps D`: the sampled verifier `D.V` answers its first `pre x` rounds itself, with no `H` call and never
accepting, and every later round is Flock's verifier `D.flock.V` at the session's statement, which the prefix's prover
messages name (`stmt`), on the later messages and coins. Where the prefix's last answer is a message, that statement has
the session's `kS x` rounds. Every unit's session is on the rounds after the prefix, and the session read off a run's
transcript is the statement and the transcript after the prefix where the prefix answered every round with a message,
and else a refused run: the run of `xR x`, a statement with `kS x` rounds at which Flock refuses the first message `[]`.

`sessionRuns_of_steps`: such a verifier runs its sessions as Flock runs (`SessionRuns`). At any coins, Flock's rounds are
the rounds after the prefix, in order; Flock's prover plays the sampled prover's messages after the prefix's transcript;
and the run decides what that Flock run decides, shows its transcript as the session, and makes its `H` calls.
-/

@[expose] public section

namespace Claims.Sampling

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

/-- **The sampled verifier's step structure**: a prefix of `pre x` rounds the verifier answers with no `H` call and no
acceptance, then Flock's verifier at the statement the prefix's prover messages name (`stmt`), on the later messages and
coins; the prefix's last answer is a message only where that statement has `kS x` rounds; every unit's session rounds are
the rounds after the prefix; and the session read off a transcript is the statement and the transcript after the prefix
where the prefix answered with messages, and the refused run of `xR x` (`kS x` rounds, refusing the message `[]` at once)
where the prefix refused. -/
structure SessionSteps (D : Deploy Coin Col T) where
  pre : Msg → ℕ
  kS : Msg → ℕ
  stmt : Msg → List Msg → Msg
  xR : Msg → Msg
  pre_pos : ∀ x, 0 < pre x
  k_eq : ∀ x, D.V.k x = pre x + kS x
  next_pre : ∀ x (as : List Msg) (cs : List Coin), as ≠ [] → as.length ≤ pre x → cs.length = as.length →
    ∃ o, D.V.next x as cs = .ret o ∧ o ≠ .acc
  next_post : ∀ x (as : List Msg) (cs : List Coin), pre x < as.length → cs.length = as.length →
    D.V.next x as cs = D.flock.V.next (stmt x (as.take (pre x))) (as.drop (pre x)) (cs.drop (pre x))
  k_stmt : ∀ x (as : List Msg) (cs : List Coin) m, as.length = pre x → cs.length = as.length →
    D.V.next x as cs = .ret (.msg m) → D.flock.V.k (stmt x as) = kS x
  rounds_iff : ∀ x u (t : Fin (D.V.k x)), t ∈ D.rounds x u ↔ pre x ≤ t.val
  k_xR : ∀ x, D.flock.V.k (xR x) = kS x
  kS_pos : ∀ x, 0 < kS x
  refuses : ∀ x (c : Coin), D.flock.V.next (xR x) [[]] [c] = .ret .rej
  session_pass : ∀ x u (ms : List (Msg × Out)) rest out, ms.length = pre x → (∀ e ∈ ms, ∃ m, e.2 = .msg m) →
    D.session x u ⟨ms ++ rest, out⟩ = some (stmt x (ms.map Prod.fst), ⟨rest, out⟩)
  session_refused : ∀ x u (ms : List (Msg × Out)) a, ms.length < pre x → (∀ e ∈ ms, ∃ m, e.2 = .msg m) →
    D.session x u ⟨ms ++ [(a, .rej)], .rej⟩ = some (xR x, ⟨[([], .rej)], .rej⟩)

/-- **`SessionRuns` from the step structure**: at coins that check a unit, its session is Flock's run on the rounds after
the prefix, of the prover that plays the sampled prover's messages after the prefix's transcript (or, where the prefix
refused, of a prover whose first message Flock refuses at `xR x`), and it decides the run, is the session the run shows,
and makes only `H` calls the run makes. -/
theorem sessionRuns_of_steps {D : Deploy Coin Col T} (S : SessionSteps D) (H : Msg → Msg) : SessionRuns H D := by
  sorry

end Claims.Sampling
