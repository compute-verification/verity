import Proofs.Flock.Claims.Recursion.DeployZK

/-!
# The recursion's deployed stop (`stopsRec`) and its record (`deployRecStop`)

`Claims.Recursion.ZKStop` reads a record with its reference prover's public stop, its finite tapes and its inner
commit strings (`StopDeploy`). This file names it at the deployment, `deployRec` with the inner session the deployed
plain Flock's:

* **The stop** (`stopsRec`): at the statement `x` and the transcript `ms` the verifier has seen, the reference prover
  stops where
  - `x` is out of scope (the compiled verifier's own scope check);
  - the inner verifier can't accept the inner statement (`FlockComplete.ready` at SHA-512, the hash the honest prover
    sets up at), read off the statement alone;
  - an answer before the sessions is not a coin's bytes (`uncoinedAt`): the compiled verifier answers every inner and
    staging round with `.msg c.bytes`, and the honest prover reads any reply as a coin (`FlockComplete.coinOf`), so a
    reply that isn't one could carry a bad coin past `badAt`, and recursion completeness (`HonestPre`) reads coin
    bytes only;
  - an inner zerocheck outer coin is `1` (`badAt`: `FlockComplete.Bad` at the inner statement, its coins read off the
    verifier's answers as the compiled verifier sends them, `.msg c.bytes`), where `pr_bad_le` bounds the probability
    and off which `complete_of` accepts the honest inner proof;
  - past the inner and staging rounds, some part's staged V* statement, the run's own, is one the masked verifier
    can't accept at SHA-512 (`unreadyAt`, `readyO`), or its numbers are not `nums`' (`mismatchAt`). There the
    compiled verifier refuses that part's session (`ract`), and `vstarAt` falls back to the statement at `pick`'s
    rounds, for which the prover holds no witness. Whether the numbers read the coins is left open: the stop covers
    both cases.
  It reads no `pick`: before the staging round it reads the statement and the verifier's answers only, so at a
  statement in scope whose inner statement the inner verifier can accept, the prover runs the inner rounds wherever
  every answer is a coin's bytes and none a bad coin (`stopsRec_live`, `halted_live`). The compiled verifier's own
  answers are coins' bytes, so `uncoinedAt` costs completeness nothing.
* **The reference prover** (`refPStop`): `refP`, except that from the first round at whose start the stop holds (read
  before each inner and staging round and after the last, `stopRounds`) it sends the empty message, as `SimStop` does
  (`refPStop_halted`, `refPStop_live`).
* **The record** (`deployRecStop`): `deployRec` with `refPStop` as its reference prover and `stopsRec` as its stop, so
  `Complete` at it reads the stopping prover at `δcRec`; its tape at each statement drawn from a finite nonce and rest
  by `tapeOf` (`Tape b` is not finite), the outer simulators `SO`, and the inner rounds' commit strings `nAt` in the
  layout `innerMsg`. Those four stay parameters, as `tapeOf` and `SO` do in the masked record of `ZK`: the inner
  prover (`RefParts.PI`) is a parameter of `deployRec`, and its round messages fix `nAt` and `innerMsg`. `ZKStop`
  reads the record's own prover (`prover_deployRecStop`).

The stop reads the transcript before the sessions, which holds the staging message, not only the coins: the
mismatch and the readiness read the staged statements, whose public files, record and transcript come off the staging
message and the inner rounds. A simulator reads it off its own run, whose staging message holds commit strings to the
zero digest.

`ZKStop` at `deployRecStop` is its bound, as `ZK` is at `deployRec` (`zkStop_deployRecStop_iff`).
-/

namespace RecDeploy

open Flock
open FlockDeploy (Coin)
open scoped NNReal

variable {N Ω : Claims.Msg → Type} [∀ x, Fintype (N x)] [∀ x, Nonempty (N x)] [∀ x, Fintype (Ω x)]
  [∀ x, Nonempty (Ω x)]

/-- **An inner zerocheck outer coin is `1`** in the transcript `ms`: at some outer coin slot of the inner statement
(`FlockComplete.outerSlots`), the verifier's answer is a coin's bytes whose element there is `1` (`FlockComplete.Bad`
on the answers). -/
def badAt (x : Claims.Msg) (ms : List (Claims.Msg × Claims.Out)) : Prop :=
  ∃ p ∈ FlockComplete.outerSlots (xIOf x), ∃ c : Coin,
    (ms[p.1]?).map Prod.snd = some (Claims.Out.msg c.bytes) ∧ c.at p.2 = 1

/-- The rounds whose starts the stop is read at, and the one after: the inner rounds of `IV` and the staging round. -/
def stopRounds (IV : Claims.Verifier Coin) (x : Claims.Msg) : ℕ := IV.k (xIOf x) + kS x

/-- **An answer before the sessions is not a coin's bytes**: among the inner and staging rounds of `ms`, the verifier
answered one with something other than `.msg c.bytes`. -/
def uncoinedAt (IV : Claims.Verifier Coin) (x : Claims.Msg) (ms : List (Claims.Msg × Claims.Out)) : Prop :=
  ∃ p ∈ ms.take (stopRounds IV x), ∀ c : Coin, p.2 ≠ Claims.Out.msg c.bytes

/-- **Some part's staged numbers are not `nums`'** once `ms` has passed the inner and staging rounds: the compiled
verifier refuses that part's session (`ract`). -/
def mismatchAt (IV : Claims.Verifier Coin) (st : Stage) (x : Claims.Msg) (ms : List (Claims.Msg × Claims.Out)) :
    Prop :=
  stopRounds IV x ≤ ms.length ∧ ∃ s < nS x, numsOf (staged st x (ms.take (stopRounds IV x)) s) ≠ nums st x s

/-- **Some part's staged V\* statement is one the masked verifier can't accept** at SHA-512 once `ms` has passed the
inner and staging rounds: `readyO` at the run's own staged statement, not `pick`'s. -/
def unreadyAt (IV : Claims.Verifier Coin) (st : Stage) (x : Claims.Msg) (ms : List (Claims.Msg × Claims.Out)) :
    Prop :=
  stopRounds IV x ≤ ms.length ∧
    ∃ s < nS x, readyO FlockDeploy.sha512 (staged st x (ms.take (stopRounds IV x)) s) = false

open Classical in
/-- **The deployed stop** at the statement `x` and the transcript `ms`: out of scope or an inner statement the inner
verifier can't accept at SHA-512 (read off the statement), an answer before the sessions that is not a coin's bytes,
an inner zerocheck outer coin at `1`, or a part's staged V\* statement the masked verifier can't accept or whose
numbers are not the statement's. -/
noncomputable def stopsRec (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (st : Stage) (x : Claims.Msg)
    (ms : List (Claims.Msg × Claims.Out)) : Bool :=
  !scopeCheck ivScope x || !FlockComplete.ready FlockDeploy.sha512 (xIOf x) || decide (uncoinedAt IV x ms) ||
    decide (badAt x ms) || decide (unreadyAt IV st x ms) || decide (mismatchAt IV st x ms)

/-- **The reference prover with the deployed stop**: `refP`, except that once `stopsRec` has held at the start of a
round up to `stopRounds` it sends the empty message. -/
noncomputable def refPStop (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (b : Brings)
    (w : Claims.Flock.Wit) : Claims.Prover (Tape b) :=
  ⟨fun x hist ω =>
    if (List.range (min hist.length (stopRounds IV x) + 1)).any
        (fun i => stopsRec IV ivScope b.st x ((asRounds hist).take i)) then .ret []
    else (refP IV b w).next x hist ω⟩

/-- **The compiled verifier with its reference prover's stop**: `deployRec`, its reference prover `refPStop` and its
stop `stopsRec`, which reads no oracle; the reference prover's tape at each statement `x` drawn from the finite nonce
and rest `N x × Ω x` by `tapeOf x`, the outer simulators `SO`, and the inner rounds' commit strings `nAt` in the
layout `innerMsg`. -/
noncomputable def deployRecStop (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (b : Brings)
    (tapeOf : (x : Claims.Msg) → N x × Ω x → Tape b) (SO : ℕ → Claims.Msg → ℕ → Claims.Recursion.OSim)
    (nAt : Claims.Msg → ℕ → ℕ)
    (innerMsg : Claims.Msg → List (Claims.Msg × Claims.Out) → List Claims.Msg → Claims.OC Claims.Msg) :
    Claims.Recursion.StopDeploy Coin :=
  { deployRec IV ivScope b with
    Tape := Tape b
    P := refPStop IV ivScope b
    Nonce := N
    Rest := Ω
    tapeOf := tapeOf
    SO := SO
    nAt := nAt
    innerMsg := innerMsg
    stops := fun _ => stopsRec IV ivScope b.st }

/-- **`ZKStop` reads the record's own reference prover**: at each statement `x`, `refPStop` on the tape `tapeOf x`
draws. -/
theorem prover_deployRecStop (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (b : Brings)
    (tapeOf : (x : Claims.Msg) → N x × Ω x → Tape b) (SO : ℕ → Claims.Msg → ℕ → Claims.Recursion.OSim)
    (nAt : Claims.Msg → ℕ → ℕ)
    (innerMsg : Claims.Msg → List (Claims.Msg × Claims.Out) → List Claims.Msg → Claims.OC Claims.Msg)
    (w : Claims.Flock.Wit) (x : Claims.Msg) :
    (deployRecStop IV ivScope b tapeOf SO nAt innerMsg).prover w x =
      ⟨fun x' hist ω => (refPStop IV ivScope b w).next x' hist (tapeOf x ω)⟩ :=
  rfl

/-- **Where the stop has fired, the reference prover sends the empty message**, `SimStop`'s message there. -/
theorem refPStop_halted (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (b : Brings)
    (tapeOf : (x : Claims.Msg) → N x × Ω x → Tape b) (SO : ℕ → Claims.Msg → ℕ → Claims.Recursion.OSim)
    (nAt : Claims.Msg → ℕ → ℕ)
    (innerMsg : Claims.Msg → List (Claims.Msg × Claims.Out) → List Claims.Msg → Claims.OC Claims.Msg)
    (H : Claims.Msg → Claims.Msg) (w : Claims.Flock.Wit) (x : Claims.Msg) (hist : List (Claims.Msg × Claims.Msg))
    (ω : Tape b)
    (h : (deployRecStop IV ivScope b tapeOf SO nAt innerMsg).halted H (stopRounds IV x) x (asRounds hist) = true) :
    (refPStop IV ivScope b w).next x hist ω = .ret [] := by
  have hl : (asRounds hist).length = hist.length := List.length_map _
  have h' : (List.range (min hist.length (stopRounds IV x) + 1)).any
      (fun i => stopsRec IV ivScope b.st x ((asRounds hist).take i)) = true := by
    rw [← hl]; exact h
  simp only [refPStop, h', ↓reduceIte]

/-- **Where it hasn't, the reference prover is `refP`.** -/
theorem refPStop_live (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (b : Brings)
    (tapeOf : (x : Claims.Msg) → N x × Ω x → Tape b) (SO : ℕ → Claims.Msg → ℕ → Claims.Recursion.OSim)
    (nAt : Claims.Msg → ℕ → ℕ)
    (innerMsg : Claims.Msg → List (Claims.Msg × Claims.Out) → List Claims.Msg → Claims.OC Claims.Msg)
    (H : Claims.Msg → Claims.Msg) (w : Claims.Flock.Wit) (x : Claims.Msg) (hist : List (Claims.Msg × Claims.Msg))
    (ω : Tape b)
    (h : (deployRecStop IV ivScope b tapeOf SO nAt innerMsg).halted H (stopRounds IV x) x (asRounds hist) = false) :
    (refPStop IV ivScope b w).next x hist ω = (refP IV b w).next x hist ω := by
  have hl : (asRounds hist).length = hist.length := List.length_map _
  have h' : (List.range (min hist.length (stopRounds IV x) + 1)).any
      (fun i => stopsRec IV ivScope b.st x ((asRounds hist).take i)) = false := by
    rw [← hl]; exact h
  simp only [refPStop, h', Bool.false_eq_true, ↓reduceIte]

/-- **The stop fires at the first round on a statement out of scope, or whose inner statement the inner verifier
can't accept**: it reads the statement alone. -/
theorem halted_of_statement (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (b : Brings)
    (tapeOf : (x : Claims.Msg) → N x × Ω x → Tape b) (SO : ℕ → Claims.Msg → ℕ → Claims.Recursion.OSim)
    (nAt : Claims.Msg → ℕ → ℕ)
    (innerMsg : Claims.Msg → List (Claims.Msg × Claims.Out) → List Claims.Msg → Claims.OC Claims.Msg)
    (H : Claims.Msg → Claims.Msg) (n : ℕ) (x : Claims.Msg) (ms : List (Claims.Msg × Claims.Out))
    (h : scopeCheck ivScope x = false ∨ FlockComplete.ready FlockDeploy.sha512 (xIOf x) = false) :
    (deployRecStop IV ivScope b tapeOf SO nAt innerMsg).halted H n x ms = true := by
  unfold Claims.Recursion.StopDeploy.halted
  refine List.any_eq_true.2 ⟨0, List.mem_range.2 (by omega), ?_⟩
  change stopsRec IV ivScope b.st x (ms.take 0) = true
  rcases h with h | h <;> simp [stopsRec, h]

/-! ## The stop doesn't fire everywhere in scope -/

/-- **Every answer of `ms` is a coin's bytes**, as the compiled verifier sends them. -/
def Coined (ms : List (Claims.Msg × Claims.Out)) : Prop := ∀ p ∈ ms, ∃ c : Coin, p.2 = Claims.Out.msg c.bytes

theorem not_uncoinedAt (IV : Claims.Verifier Coin) (x : Claims.Msg) {ms : List (Claims.Msg × Claims.Out)}
    (hc : Coined ms) : ¬ uncoinedAt IV x ms := fun ⟨p, hp, hn⟩ =>
  let ⟨c, hpc⟩ := hc p (List.mem_of_mem_take hp)
  hn c hpc

theorem Coined.take {ms : List (Claims.Msg × Claims.Out)} (hc : Coined ms) (i : ℕ) : Coined (ms.take i) :=
  fun p hp => hc p (List.mem_of_mem_take hp)

/-- **Before the staging round the stop fires only at an answer that is not a coin's bytes or a bad inner coin**, at a
statement in scope whose inner statement the inner verifier can accept at SHA-512: the readiness and the numbers of
the staged statements are read only past the staging round. -/
theorem stopsRec_live (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (st : Stage) (x : Claims.Msg)
    (ms : List (Claims.Msg × Claims.Out)) (hs : scopeCheck ivScope x = true)
    (hr : FlockComplete.ready FlockDeploy.sha512 (xIOf x) = true) (hl : ms.length < stopRounds IV x)
    (hc : Coined ms) (hb : ¬ badAt x ms) : stopsRec IV ivScope st x ms = false := by
  have hu : ¬ unreadyAt IV st x ms := fun h => absurd h.1 (by omega)
  have hm : ¬ mismatchAt IV st x ms := fun h => absurd h.1 (by omega)
  simp [stopsRec, hs, hr, not_uncoinedAt IV x hc, hb, hu, hm]

/-- **The prover with the stop runs the inner rounds**: at a statement in scope whose inner statement the inner
verifier can accept at SHA-512, before the staging round, where every answer is a coin's bytes and no prefix of the
transcript holds a bad inner coin, the reference prover hasn't stopped. -/
theorem halted_live (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (b : Brings)
    (tapeOf : (x : Claims.Msg) → N x × Ω x → Tape b) (SO : ℕ → Claims.Msg → ℕ → Claims.Recursion.OSim)
    (nAt : Claims.Msg → ℕ → ℕ)
    (innerMsg : Claims.Msg → List (Claims.Msg × Claims.Out) → List Claims.Msg → Claims.OC Claims.Msg)
    (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) (ms : List (Claims.Msg × Claims.Out))
    (hs : scopeCheck ivScope x = true) (hr : FlockComplete.ready FlockDeploy.sha512 (xIOf x) = true)
    (hl : ms.length < stopRounds IV x) (hc : Coined ms) (hb : ∀ i ≤ ms.length, ¬ badAt x (ms.take i)) :
    (deployRecStop IV ivScope b tapeOf SO nAt innerMsg).halted H (stopRounds IV x) x ms = false := by
  unfold Claims.Recursion.StopDeploy.halted
  refine List.any_eq_false.2 fun i hi => ?_
  have hi' : i ≤ ms.length := by have := List.mem_range.1 hi; omega
  change ¬ stopsRec IV ivScope b.st x (ms.take i) = true
  rw [stopsRec_live IV ivScope b.st x (ms.take i) hs hr (by rw [List.length_take]; omega) (hc.take i) (hb i hi')]
  exact Bool.false_ne_true

/-! ## The claim at `deployRecStop` -/

/-- **The recursion's `ZKStop` at `deployRecStop` is its bound**: the outer record is the masked one, and the rest is
`ZKmalRx` of the record's own reference prover at `SimStop` within `εzkStop + γzk · Pr[Fork]`. -/
theorem zkStop_deployRecStop_iff (H : Claims.Msg → Claims.Msg) (IV : Claims.Verifier Coin)
    (ivScope : Claims.Msg → Bool) (b : Brings) (I : Claims.Recursion.Inner Coin Claims.Flock.Wit)
    (relO : Readable.Parsed → Readable.Relation) (tapeOf : (x : Claims.Msg) → N x × Ω x → Tape b)
    (SO : ℕ → Claims.Msg → ℕ → Claims.Recursion.OSim) (nAt : Claims.Msg → ℕ → ℕ)
    (innerMsg : Claims.Msg → List (Claims.Msg × Claims.Out) → List Claims.Msg → Claims.OC Claims.Msg) :
    Claims.Recursion.ZKStop H (deployRecStop IV ivScope b tapeOf SO nAt innerMsg) I (deployZkRec relO) ↔
      ∀ t, 0 < t → ∀ δ₁ : ℝ≥0,
        Claims.Hm96Hiding (deployRecStop IV ivScope b tapeOf SO nAt innerMsg).mask
            ((deployRecStop IV ivScope b tapeOf SO nAt innerMsg).toDeploy.saltHash H) δ₁ →
          Claims.ZKmalRx N Ω H (deployRecStop IV ivScope b tapeOf SO nAt innerMsg).V.k
            (deployRecStop IV ivScope b tapeOf SO nAt innerMsg).prover
            (Claims.Recursion.R (deployRecStop IV ivScope b tapeOf SO nAt innerMsg).toDeploy I)
            (Claims.Recursion.SimStop (deployRecStop IV ivScope b tapeOf SO nAt innerMsg) I (SO t) H)
            (Claims.Recursion.εzkStop (deployRecStop IV ivScope b tapeOf SO nAt innerMsg) I t δ₁)
            (Claims.Recursion.γzk (deployRecStop IV ivScope b tapeOf SO nAt innerMsg).toDeploy t) :=
  ⟨And.right, fun h => ⟨rfl, h⟩⟩

end RecDeploy
