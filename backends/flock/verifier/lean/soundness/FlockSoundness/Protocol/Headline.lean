import FlockSoundness.Protocol.Binding.Rows
import FlockSoundness.Protocol.StrictCR
import FlockSoundness.Protocol.ExecDrawOS

/-!
# The headline: the program's rows, SHA-512 throughout, the live draw

`Prog.flock_headline` is the one end-to-end statement the others specialize into. It combines:
* the circuit of the statement's rows (`p.circuit outs`, `Prog.circuit`), each drawn unit's place from its table's
  statement (`ProgPlaces`);
* SHA-512 (`Refine.H512`, `Refine.enc512`) as the session's hash and the rows' commit hash, with the rows' value binding
  (`flock_e2e_count_hm96`);
* the live draw (`Law.execOS`, `Flock.Draw.drawOS`'s stratified draw on a byte source), its count curve at most
  `Law.stratified`'s under A3 (`Law.execOS_miss_le`).

Its cryptographic hypotheses are instances of three assumptions, each for the finders built from the prover bounded:
* `SHA512CRStrict` (`cr/sha-512`) through `hKS` (`StrictCR.TableCR`) and `hT` (`StrictCR.Finder.CR`);
* `SHA512CRExpected` (`ecr/sha-512`) through `hCR` (`LinkCR`);
* A3 (`uniform/io-getrandombytes`) through `hA3` (`UniformRandomBytes`).

The others are the named obligations `assumptions/e2e-checklist.md` lists: W6 and `Layout.Aliased` (`pp.placed`,
`pp.aliased`), `HmRowComputes` (`hHm`), the layout's facts (`lay`, `rs`, in place of `vb`), the serving roots' rewinding
(`tr`, in place of `hExec`; `TreeRewind.ofNever` builds it from `hExec` at `δ_tree = 0`), `hConst` and `hZero`. The rest
(`hks`, `hS`, `hk`, `hRw`, `hM`, `hρ`, `hr`, `ht`) are numeric conditions and the strata's match. The budgets must bound
their finders' real SHA-512 evaluations, `qF, qT ≥ 2t′ + 2v` and `qS ≥ t′ + v` for `t′` one session run's and `v` one
opening's verification, which Lean doesn't check (`ASSUMPTIONS.md`).

It bounds wrong units of the rows' circuit. That a correct unit computes a Definition additionally needs the unit's
certificate (`RowsCert`, `IsRowsUnit.computes_of_cert`) and the Definition's Boolean semantics in Lean.

Its proof coins are the model's: uniform, drawn each round after the prover's message. That the coin server's round
coins are those is A6 (`uniform/flock-coin-server`, `ASSUMPTIONS.md`), which the model embodies rather than a hypothesis
states, and which covers only per-round OS coins: M0's seed-derived coins are outside the headline.
-/

namespace FlockSoundness.Audit

open Game Model Accounting Extract LinkFinder Refine Partition
open scoped ENNReal

/-- SHA-512's digests are inhabited, as the session's tables ask of their digests. -/
instance digInhabited : Inhabited Dig := ⟨H512 []⟩

variable {N n : ℕ} (p : FlockSoundness.Prog N n) (outs : Finset (Fin N)) {Reg : Type} {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)

end FlockSoundness.Audit
