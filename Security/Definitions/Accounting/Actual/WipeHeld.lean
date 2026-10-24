import Definitions.Accounting.Confidence

/-!
# What a server answers of a wipe's store at a read (trusted)

The physical side of the memory cap's lifetime pool. `Accounting.WipePlay` takes a wipe's un-held set `U` as a label
on its run tree; this reads it off the challenger's verdicts on the actual circuit instead, per read.

A strategy `S` maps coins `r = (ω₀, c)` to a run (`Actual.Run`). `c : Fin n → B` is the indices of the strategy's `n`
timed reads over its whole life, every wipe's reads in order, each uniform on a store `B` of blocks; `ω₀ : Ω₀` is every
other coin (each wipe's label key, the prior content, the workload's inputs). The memory challenger's verdict on read
`j` is `acc r j`.

* `answered acc j r`: the blocks `i` the challenger would accept at read `j` had its index been `i`, the other coins as
  in `r` (`Function.update`). With verdicts fixed by the coins up to their own read
  (`Accounting.Assumptions.LiveVerdicts`), it is fixed before read `j`'s index is drawn, and read `j` is accepted
  exactly when its index is in it. It is what the server can answer, not what it holds: what it costs in state is
  `Accounting.Guarantees.Actual.AnsweredOfQuiet`'s count.
* `unanswered`: read `j`'s un-answered fraction `x_j = (|B| − |answered|) / |B|`, in `[0, 1]`, the chance read `j`
  fails given the earlier coins.
* `contents S X t K`: the contents of `X`'s state at `t` over the coins `K`. `log₂` of its size is how many bits of
  `X`'s state at `t` vary over `K`: with `K` the coins that share one wipe's labels, the bits the state holds beyond
  those labels.

A block's held set at a guessing level (a label value on a `2^-λ` share of the coins with the state's content) is not
a memory measure: one label's offset, `b − log₂ |B|` bits, makes every block of the store held at once with
probability `1 − 1/e`, at every `λ ≥ log₂ |B| + 1` (note:20261019T1330Z-report-timed-round-redteam-2).
-/

namespace Accounting

open Finset

namespace Actual

open Classical in
/-- **The blocks answered at read `j`**: the blocks `i` of the store `B` for which the challenger accepts read `j` in
the run whose coins are `r`'s with read `j`'s index set to `i`. -/
noncomputable def answered {Ω₀ B : Type*} [Fintype B] {n : ℕ} (acc : Ω₀ × (Fin n → B) → Fin n → Prop) (j : Fin n)
    (r : Ω₀ × (Fin n → B)) : Finset B :=
  univ.filter fun i => acc (r.1, Function.update r.2 j i) j

/-- **Read `j`'s un-answered fraction** `x_j = (|B| − |answered|) / |B|`: the share of the store's indices on which the
challenger rejects read `j`, the other coins as in `r`. -/
noncomputable def unanswered {Ω₀ B : Type*} [Fintype B] {n : ℕ} (acc : Ω₀ × (Fin n → B) → Fin n → Prop) (j : Fin n)
    (r : Ω₀ × (Fin n → B)) : ℝ :=
  ((Fintype.card B - (answered acc j r).card : ℕ) : ℝ) / Fintype.card B

/-- **`X`'s state contents at `t` over the coins `K`**: the content of `X`'s state at `t` (`Actual.Run.content`) in the
run of each coin of `K`. -/
def contents {Ω L : Type*} [LinearOrder L] (S : Ω → Actual.Run L) (X : Finset L) (t : ℕ) (K : Set Ω) :
    Set (List Bool) :=
  (fun r => (S r).content X t) '' K

end Actual

end Accounting
