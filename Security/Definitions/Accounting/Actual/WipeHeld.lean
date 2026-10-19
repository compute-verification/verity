import Definitions.Accounting.Confidence

/-!
# What a server holds of a wipe's store at a read (trusted)

The physical side of the memory cap's lifetime pool. `Accounting.WipePlay` takes a wipe's un-held set `U` as a label
on its run tree; this reads it off the actual circuit instead, per read, from what the server's state holds.

A strategy `S` maps coins `r = (ω₀, c)` to a run (`Actual.Run`). `c : Fin n → B` is the indices of the strategy's `n`
timed reads over its whole life, every wipe's reads in order, each uniform on a store `B` of blocks; `ω₀ : Ω₀` is every
other coin (each wipe's label key, the prior content, the workload's inputs). Read `j` challenges block `c j`, whose
label is `lab ω₀ j (c j)`: the labels of read `j`'s wipe, fixed by `ω₀`. Its index reaches the server, at the location
`X`, at the tick `arrive r j`.

* `heldAt S X lab lam arrive j r`: the blocks that the server, holding its state at read `j`'s arrival, can name the
  label of with probability at least `2^-lam`. The probability is over the coins whose runs have read `j` arrive at
  the same tick with the same content of `X`'s state there (`Actual.Run.content`: every leaf of `X`, so every memory
  tier inside it, what each holds and what is in flight between them): one label value takes at least a `2^-lam`
  share of them. Held is information, not position: a label kept compressed, encoded, split across leaves, or
  recomputable from the state is held, and one the content says nothing about, among `2^b` uniform values, is not once
  `b > lam`. The index of read `j` arrives at its tick and is not yet in the content (`Actual.Run.state` is what is held
  at the start of a tick).
* `unheldAt`: read `j`'s un-held fraction `x_j = (|B| − |heldAt|) / |B|`, in `[0, 1]`.
* `contents S X t K`: the contents of `X`'s state at `t` over the coins `K`. `log₂` of its size is how many bits of
  `X`'s state at `t` vary over `K`: with `K` the coins that share one wipe's labels, the bits the state holds beyond
  those labels.
-/

namespace Accounting

open Finset

namespace Actual

open Classical in
/-- **The blocks held at read `j`'s arrival**, at guessing level `2^-lam`: the blocks `i` of the store `B` for which
some label value `v` is `i`'s label (`lab ω₀ j i`) on at least a `2^-lam` share of the coins whose runs have read `j`
arrive at the same tick as in the run of `r`, with the same content of `X`'s state there. -/
noncomputable def heldAt {Ω₀ B L : Type*} [Fintype Ω₀] [Fintype B] [LinearOrder L] {n : ℕ}
    (S : Ω₀ × (Fin n → B) → Actual.Run L) (X : Finset L) (lab : Ω₀ → Fin n → B → ℕ) (lam : ℕ)
    (arrive : Ω₀ × (Fin n → B) → Fin n → ℕ) (j : Fin n) (r : Ω₀ × (Fin n → B)) : Finset B :=
  univ.filter fun i => ∃ v : ℕ,
    (univ.filter fun r' => arrive r' j = arrive r j ∧
        (S r').content X (arrive r j) = (S r).content X (arrive r j)).card ≤
      2 ^ lam * (univ.filter fun r' => arrive r' j = arrive r j ∧
        (S r').content X (arrive r j) = (S r).content X (arrive r j) ∧ lab r'.1 j i = v).card

/-- **Read `j`'s un-held fraction** `x_j = (|B| − |heldAt|) / |B|`: the share of the store the server can't name at
read `j`'s arrival, at guessing level `2^-lam`. -/
noncomputable def unheldAt {Ω₀ B L : Type*} [Fintype Ω₀] [Fintype B] [LinearOrder L] {n : ℕ}
    (S : Ω₀ × (Fin n → B) → Actual.Run L) (X : Finset L) (lab : Ω₀ → Fin n → B → ℕ) (lam : ℕ)
    (arrive : Ω₀ × (Fin n → B) → Fin n → ℕ) (j : Fin n) (r : Ω₀ × (Fin n → B)) : ℝ :=
  ((Fintype.card B - (heldAt S X lab lam arrive j r).card : ℕ) : ℝ) / Fintype.card B

/-- **`X`'s state contents at `t` over the coins `K`**: the content of `X`'s state at `t` (`Actual.Run.content`) in the
run of each coin of `K`. -/
def contents {Ω L : Type*} [LinearOrder L] (S : Ω → Actual.Run L) (X : Finset L) (t : ℕ) (K : Set Ω) :
    Set (List Bool) :=
  (fun r => (S r).content X t) '' K

end Actual

end Accounting
