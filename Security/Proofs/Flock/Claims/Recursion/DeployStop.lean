import Proofs.Flock.Claims.Recursion.DeployZK

/-!
# The recursion's deployed stop (`stopsRec`) and its record (`deployRecStop`)

`Claims.Recursion.ZKStop` reads a record with its reference prover's public stop (`StopDeploy`). This file names it at
the deployment, `deployRec` with the inner session the deployed plain Flock's:

* **The stop** (`stopsRec`): at the statement `x` and the transcript `ms` the verifier has seen, the reference prover
  stops where
  - `x` is out of scope (the compiled verifier's own scope check);
  - the compiled verifier can't accept `x` (`readyRec`, at SHA-512, the hash the honest prover sets up at). It reads the
    statement and the record only (`pick` is a choice over the transcripts, which the statement fixes), never a
    witness: it is the region where `δcRec` is `1`;
  - an inner zerocheck outer coin is `1` (`badAt`: `FlockComplete.Bad` at the inner statement, its coins read off the
    verifier's answers as the compiled verifier sends them, `.msg c.bytes`), where `pr_bad_le` bounds the probability
    and off which `complete_of` accepts the honest inner proof;
  - past the inner and staging rounds, some part's staged numbers are not `nums`' (`mismatchAt`). There the compiled
    verifier refuses that part's session (`ract`), and `vstarAt` falls back to the statement at `pick`'s rounds, for
    which the prover holds no witness. Whether the numbers read the coins is left open: the stop covers both cases.
* **The reference prover** (`refPStop`): `refP`, except that from the first round at whose start the stop holds (read
  before each inner and staging round and after the last, `stopRounds`) it sends the empty message, as `SimStop` does
  (`refPStop_halted`, `refPStop_live`).
* **The record** (`deployRecStop`): `deployRec` with `refPStop` as its reference prover and `stopsRec` as its stop, so
  `Complete` at it reads the stopping prover at `δcRec`.

The stop reads the transcript before the sessions, which holds the staging message, not only the coins: the mismatch
reads the staged statements, whose public files, record and transcript come off the staging message and the inner
rounds. A simulator reads it off its own run, whose staging message holds commit strings to the zero digest.

`ZKStop` at `deployRecStop` is its bound, as `ZK` is at `deployRec` (`zkStop_deployRecStop_iff`), and `ZKPrivStop`
holds outright (`zkPrivStop_deployRecStop`), as `ZKPriv` does.
-/

namespace RecDeploy

open Flock
open FlockDeploy (Coin)
open scoped NNReal

variable {W N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω]

/-- **An inner zerocheck outer coin is `1`** in the transcript `ms`: at some outer coin slot of the inner statement
(`FlockComplete.outerSlots`), the verifier's answer is a coin's bytes whose element there is `1` (`FlockComplete.Bad`
on the answers). -/
def badAt (x : Claims.Msg) (ms : List (Claims.Msg × Claims.Out)) : Prop :=
  ∃ p ∈ FlockComplete.outerSlots (xIOf x), ∃ c : Coin,
    (ms[p.1]?).map Prod.snd = some (Claims.Out.msg c.bytes) ∧ c.at p.2 = 1

/-- The rounds whose starts the stop is read at, and the one after: the inner rounds of `IV` and the staging round. -/
def stopRounds (IV : Claims.Verifier Coin) (x : Claims.Msg) : ℕ := IV.k (xIOf x) + kS x

/-- **Some part's staged numbers are not `nums`'** once `ms` has passed the inner and staging rounds: the compiled
verifier refuses that part's session (`ract`). -/
def mismatchAt (IV : Claims.Verifier Coin) (st : Stage) (x : Claims.Msg) (ms : List (Claims.Msg × Claims.Out)) :
    Prop :=
  stopRounds IV x ≤ ms.length ∧ ∃ s < nS x, numsOf (staged st x (ms.take (stopRounds IV x)) s) ≠ nums st x s

open Classical in
/-- **The deployed stop** at the statement `x` and the transcript `ms`: out of scope, not ready at SHA-512 (read off
the statement), an inner zerocheck outer coin at `1`, or a part's staged numbers not the statement's. -/
noncomputable def stopsRec (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (st : Stage) (x : Claims.Msg)
    (ms : List (Claims.Msg × Claims.Out)) : Bool :=
  !scopeCheck ivScope x || !readyRec st FlockDeploy.sha512 x || decide (badAt x ms) || decide (mismatchAt IV st x ms)

/-- **The reference prover with the deployed stop**: `refP`, except that once `stopsRec` has held at the start of a
round up to `stopRounds` it sends the empty message. -/
noncomputable def refPStop (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (b : Brings)
    (w : Claims.Flock.Wit) : Claims.Prover (Tape b) :=
  ⟨fun x hist ω =>
    if (List.range (min hist.length (stopRounds IV x) + 1)).any
        (fun i => stopsRec IV ivScope b.st x ((asRounds hist).take i)) then .ret []
    else (refP IV b w).next x hist ω⟩

/-- **The compiled verifier with its reference prover's stop**: `deployRec`, its reference prover `refPStop` and its
stop `stopsRec`, which reads no oracle. -/
noncomputable def deployRecStop (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (b : Brings) :
    Claims.Recursion.StopDeploy Coin :=
  { deployRec IV ivScope b with
    Tape := Tape b
    P := refPStop IV ivScope b
    stops := fun _ => stopsRec IV ivScope b.st }

/-- **Where the stop has fired, the reference prover sends the empty message**, `SimStop`'s message there. -/
theorem refPStop_halted (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (b : Brings)
    (H : Claims.Msg → Claims.Msg) (w : Claims.Flock.Wit) (x : Claims.Msg) (hist : List (Claims.Msg × Claims.Msg))
    (ω : Tape b) (h : (deployRecStop IV ivScope b).halted H (stopRounds IV x) x (asRounds hist) = true) :
    (refPStop IV ivScope b w).next x hist ω = .ret [] := by
  have hl : (asRounds hist).length = hist.length := List.length_map _
  have h' : (List.range (min hist.length (stopRounds IV x) + 1)).any
      (fun i => stopsRec IV ivScope b.st x ((asRounds hist).take i)) = true := by
    rw [← hl]; exact h
  simp only [refPStop, h', ↓reduceIte]

/-- **Where it hasn't, the reference prover is `refP`.** -/
theorem refPStop_live (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (b : Brings)
    (H : Claims.Msg → Claims.Msg) (w : Claims.Flock.Wit) (x : Claims.Msg) (hist : List (Claims.Msg × Claims.Msg))
    (ω : Tape b) (h : (deployRecStop IV ivScope b).halted H (stopRounds IV x) x (asRounds hist) = false) :
    (refPStop IV ivScope b w).next x hist ω = (refP IV b w).next x hist ω := by
  have hl : (asRounds hist).length = hist.length := List.length_map _
  have h' : (List.range (min hist.length (stopRounds IV x) + 1)).any
      (fun i => stopsRec IV ivScope b.st x ((asRounds hist).take i)) = false := by
    rw [← hl]; exact h
  simp only [refPStop, h', Bool.false_eq_true, ↓reduceIte]

/-- **The stop fires at the first round on a statement out of scope or not ready**: it reads the statement alone. -/
theorem halted_of_statement (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (b : Brings)
    (H : Claims.Msg → Claims.Msg) (n : ℕ) (x : Claims.Msg) (ms : List (Claims.Msg × Claims.Out))
    (h : scopeCheck ivScope x = false ∨ readyRec b.st FlockDeploy.sha512 x = false) :
    (deployRecStop IV ivScope b).halted H n x ms = true := by
  unfold Claims.Recursion.StopDeploy.halted
  refine List.any_eq_true.2 ⟨0, List.mem_range.2 (by omega), ?_⟩
  change stopsRec IV ivScope b.st x (ms.take 0) = true
  rcases h with h | h <;> simp [stopsRec, h]

/-! ## The claims at `deployRecStop` -/

/-- **The recursion's `ZKStop` at `deployRecStop` is its bound**: the outer record is the masked one, and the rest is
`ZKmalR` of `P` at `SimStop` within `εzk + γzk · Pr[Fork]`. -/
theorem zkStop_deployRecStop_iff (H : Claims.Msg → Claims.Msg) (IV : Claims.Verifier Coin)
    (ivScope : Claims.Msg → Bool) (b : Brings) (I : Claims.Recursion.Inner Coin W)
    (relO : Readable.Parsed → Readable.Relation) (P : W → Claims.Prover (N × Ω))
    (SO : ℕ → Claims.Msg → ℕ → Claims.Recursion.OSim) :
    Claims.Recursion.ZKStop H (deployRecStop IV ivScope b) I (deployZkRec relO) P SO ↔
      ∀ t, 0 < t → ∀ δ₁ : ℝ≥0,
        Claims.Hm96Hiding (deployRecStop IV ivScope b).mask
            ((deployRecStop IV ivScope b).toDeploy.saltHash H) δ₁ →
          Claims.ZKmalR H (deployRecStop IV ivScope b).V.k P
            (Claims.Recursion.R (deployRecStop IV ivScope b).toDeploy I)
            (Claims.Recursion.SimStop (deployRecStop IV ivScope b) I (SO t) H)
            (Claims.Recursion.εzk (deployRecStop IV ivScope b).toDeploy I t δ₁)
            (Claims.Recursion.γzk (deployRecStop IV ivScope b).toDeploy t) :=
  ⟨And.right, fun h => ⟨rfl, h⟩⟩

/-- **`ZKPrivStop` at `deployRecStop` holds**: no statement is in `RPriv`, as `parse` reads no circuit commitment. -/
theorem zkPrivStop_deployRecStop (H : Claims.Msg → Claims.Msg) (IV : Claims.Verifier Coin)
    (ivScope : Claims.Msg → Bool) (b : Brings) (I : Claims.Recursion.Inner Coin W)
    (relO : Readable.Parsed → Readable.Relation) (P : W → Claims.Prover (N × Ω))
    (SO : ℕ → Claims.Msg → ℕ → Claims.Recursion.OSim) :
    Claims.Recursion.ZKPrivStop H (deployRecStop IV ivScope b) I (deployZkRec relO) P SO := by
  refine ⟨rfl, fun t _ δ₁ _ x w hR => ?_⟩
  obtain ⟨s, hs, hc, -⟩ := hR
  rw [parse_circ (x := x) hs] at hc
  cases hc

end RecDeploy
