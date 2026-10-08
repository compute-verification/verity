import Definitions.Pous.Model.Audit

/-!
# The secure-erasure audit: volatile memory in full use through the spot reads

Added on 4 Oct 2026, under the model Daniel set that day: the disks are unplugged, so the prover's memory is volatile
only (device memory and host DRAM), and the audit is a proof of memory utilization. A prover declares `S` bits of volatile memory;
the audit asks it to fill them with the keyed fill `C` and then reads `C` back at random places under a short deadline.
`EraseMeets` (`Security/Definitions/Pous/Accounting/EraseMeets.lean`) states the requirement over this game.

**The game.** It is the sequential timed audit (`Security/Definitions/Pous/Model/Audit.lean`) with a fill check
before it and the prover's foreign content beside it:

1. The prover holds **foreign content** `z`: `S_φ` bits, uniform, independent of everything below. It is what a secure
   erasure must remove (data from before the audit, another tenant's model, anything not of `C`), and `S_φ` is its
   information content.
2. Setup is the timed audit's: `W` is arbitrary and public, `ω` and the coins are uniform, `(pp, C) := enc W ω r`. `C`
   is the **keyed fill** `F_K`: the scheme's salt is the key `K` the memory challenger draws fresh for the audit
   (`verity.randomness`). Every statement holds for every salt with the prover chosen after it, so in particular for
   a fresh uniform `K` drawn after `z` is fixed.
3. **Fill.** The memory challenger draws a digest key `κ` uniformly. From `z`, `(W, pp, C)`, the whole primitive `ω` and
   `κ`, the prover returns a digest value `h` and its memory `σ₀ : Bits S` at the first spot read's arrival
   (`EraseProver.fill`). The fill is unbounded: it may compute anything, so every bound here holds at every fill
   deadline `T_fill`.
4. **Spot reads.** The memory challenger draws `s` indices `I : Fin s → Fin B`, uniform and with replacement, and
   reveals them one at a time, as the sequential audit does: round `j` receives `i_j`, the round number and the memory
   `σ_j` left by round `j − 1`, and runs a `(D, Q)`-bounded program that returns block `i_j` and the next memory
   `σ_{j+1} : Bits S` (`EraseProver.read`, a `SeqResponder`). `(D, Q)` is what the prover completes within the read
   deadline `T_read`; the hardware calibration of `T_read` is separate.
5. The memory challenger **accepts** iff `h = dig κ C` and every returned block is correct.
6. The prover **keeps** its foreign content iff `keep j σ_j = z` at every round `j < s` (`EraseProver.keep`): `z` can be
   read off its memory at every spot read's arrival, by any function of the memory alone (it does not query the
   primitive), which may change from round to round.

**Memory.** `σ_j` is the prover's whole volatile memory at round `j`'s arrival, and nothing else survives from one round
to the next: no disk, no network (perfect isolation, the 4 Oct model). So the state bound is a bound on memory at the
arrivals, which is what the guarantee is about; unlike the storage audit
(`Security/Definitions/Pous/Model/Graded.lean`, "Why late rounds are bounded"), it needs no link to storage at
rest, and a prover may recompute anything during the audit within its memory.

**What the game does not model.** The bandwidth between memory tiers: device memory and host DRAM count alike, and no statement
over this game rests on a bandwidth-separation premise (the host-link assumption `TwoTierBandwidth` was dropped on
3 Oct 2026). Memory the prover has beyond the `S` bits it declares: `S` must be all of its volatile memory.

**What to check.** That `z` is independent of the primitive, the coins, `κ` and the indices; that the fill
sees `κ` (it may, as when the digest key is derived from `K`); that keeping is required at every round's arrival
(foreign content dropped during the audit is erased, which is what a secure erasure asks); and that the spot reads are
the sequential audit's, so `AuditSecure` bounds them.
-/

namespace Pous

open scoped ENNReal

variable {M : IdealModel}

/-- A prover in the secure-erasure audit, with `S` bits of volatile memory, foreign content of `S_φ` bits, a digest key
in `K` and digest values in `V`. -/
structure EraseProver (sch : Scheme M) (S S_φ D Q : ℕ) (K V : Type) : Type where
  /-- The fill: from the foreign content, `(W, pp, C)`, the whole primitive and the digest key, the digest it returns
  and its memory at the first spot read's arrival. -/
  fill : Bits S_φ → Bits sch.n → Bits sch.t → sch.Code → M.Ω → K → V × Bits S
  /-- The spot reads: round `j` answers its index from the memory `σ_j` within `(D, Q)`. -/
  read : SeqResponder sch S D Q
  /-- Where it keeps its foreign content: round `j`'s memory gives it back. -/
  keep : ℕ → Bits S → Bits S_φ

/-- The randomness of one secure-erasure audit: the foreign content, the digest key, and the timed audit's primitive,
coins and indices. -/
abbrev EraseOutcome (sch : Scheme M) (S_φ : ℕ) (K : Type) (s : ℕ) : Type := Bits S_φ × K × Outcome sch s

/-- The blocks the spot reads return for the indices `is`, from round `j` in memory `σ`, each with what `keep` reads off
the memory at that round's arrival. -/
def eraseAnswers {ℓ S S_φ B : ℕ} (f : M.Qry → M.Ans) (step : ℕ → Bits S → Fin B → Prog M (Bits ℓ × Bits S))
    (keep : ℕ → Bits S → Bits S_φ) : ℕ → List (Fin B) → Bits S → List (Bits ℓ × Bits S_φ)
  | _, [], _ => []
  | j, i :: is, σ =>
      (((step j σ i).run f).1, keep j σ) :: eraseAnswers f step keep (j + 1) is ((step j σ i).run f).2

/-- The prover wins: the memory challenger accepts (the digest is `dig κ C` and every spot read is correct) and the
foreign content is in memory at every spot read's arrival. -/
def eraseWins (sch : Scheme M) {S S_φ D Q : ℕ} {K V : Type} (dig : K → sch.Code → V) (s : ℕ) (W : Bits sch.n)
    (P : EraseProver sch S S_φ D Q K V) (x : EraseOutcome sch S_φ K s) : Prop :=
  let z := x.1
  let κ := x.2.1
  let ω := x.2.2.1
  let pp := (sch.enc W ω x.2.2.2.1).1
  let C := (sch.enc W ω x.2.2.2.1).2
  let out := P.fill z W pp C ω κ
  let ans := eraseAnswers (M.answer ω) (P.read.step W pp) P.keep 0 (List.ofFn x.2.2.2.2) out.2
  out.1 = dig κ C ∧ ans.map Prod.fst = List.ofFn (fun j => C (x.2.2.2.2 j)) ∧ ∀ a ∈ ans, a.2 = z

/-- **The secure-erasure target.** For every `W` and every prover with `S` bits of memory and `S_φ` bits of foreign
content, the `s`-read audit with digest `dig` accepts with the foreign content kept with probability at most `ε`. -/
def EraseSecure (sch : Scheme M) {K V : Type} [Fintype K] (dig : K → sch.Code → V) (S S_φ D Q s : ℕ)
    (ε : ℝ≥0∞) : Prop :=
  ∀ (W : Bits sch.n) (P : EraseProver sch S S_φ D Q K V), pr (eraseWins sch dig s W P) ≤ ε

end Pous
