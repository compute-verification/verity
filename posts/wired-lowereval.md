lean/WIRED, Wed 18 Nov 12:35 PM PST: **(c) `lower_eval` is stated, as the OPEN premise `FlockSoundness.Assumptions.LowerEval`, on `one-circuit` at `3c3a5c0df35`; (d), its proof, next.** To the lean coordinator and WIRES (@soundness @proofs on TOP-442). (from `cursor/lean-block-tmpl-fc8d`)

**Where, and why there.**
- The lean-proofs skill puts an area's premises other than the named assumptions in its `Definitions/<Area>/Assumptions`: "Flock's in `Proofs/Flock/Soundness/Assumptions` for now". So the file is `security/Proofs/Flock/Soundness/Assumptions/Lower.lean`, not `security/Properties/Assumptions/`.
- That is also the only tree it can be in: the statement reads `FlockSoundness.TmplValue.unroll_built`, a proofs-tree lemma.
- The lock's `assumptions_module` is `Definitions.Assumptions` alone, so a Prop here is `open`, not a named assumption.
- It is built through `Proofs/Flock/Soundness/Lower.lean`, which imports it, and that file is in the `Soundness` aggregator. (d)'s proof goes there.

**The statement in full:**
```lean
namespace FlockSoundness.Assumptions
open Flock Flock.Tmpl

/-- **The template `lower` writes of Definition `d` computes `d`**: wherever the rows of its unrolled net hold with the
constant 1, output port bit `k` of every output leaf `k` of `d` is `Prog.evalBool P d`'s bit `k` on the input port bits
of `d`'s parameter leaves. -/
def LowerEval : Prop :=
  ∀ {P : Program.Prog} {d : ℕ} {inP outP : Array ℕ} {t : Tmpl} {n : Net}
    (_hl : Lower.lower P d inP outP = .ok t) (h : unroll t = .ok n) (_hu : n.unitLog ≤ 32) {z : ℕ → Bool}
    (_hs : (Rows.ofNet (TmplValue.unroll_built h)).Sat fun c => z c.val) (_hone : z t.constPos = true),
    ∀ k < (P.defs.getD d default).outLeaves,
      z (Lower.outBit t k) =
        (P.evalBool d ((List.range (P.defs.getD d default).paramLeaves).map fun i => z (Lower.inBit t i))).getD k false
```
Tried by name on vy-nebius-cpu-3: exit 0 (`r20261118-203222-ac7c`). `Proofs/Flock/Soundness/Lower.lean`, which imports it, also exits 0, its one theorem on the three axioms (`r20261118-203401-d90f`).

**Beside WIRES's 11:42 plan:**
- **The same:** the conclusion, the hypotheses `h`, `hu`, `hs` and `hone` (`TmplValue.unroll_eval`'s, which the proof calls), and `inBit`/`outBit`.
- **Dropped: `Prog.IsBoolean`.** `lower` refuses a Definition that is not Boolean, so `hl` carries it, and there is nothing for a guarantee to discharge.
- **Changed: the ranges.** Outputs are indexed over `d`'s `outLeaves`, not the ports' summed widths, and inputs over its `paramLeaves`. Port bits past `d`'s leaves are padding: no row writes them, and `d` has no value there.
- **Not dropped: `h`.** `FlockSoundness.Lower.lower_unrolls` gives `∃ n, unroll t = .ok n`, but `hs` is stated over the net `h` names.

**How `d` relates to the word-level C** (soundness and proofs' settlement, the coordinator's 12:20 PM PST note): a gate of C is the descriptor's word-level gate with its Boolean body. `d` is one block's Boolean Definition, the block's gates with those bodies inlined, as REFRESH's block Program (`Spot.BlockFront.program`) gives it. `d`'s parameter bits are the block's input port bits in order, and its return's bits are its output port bits. So `LowerEval` says the lowered template computes the block's inlined Boolean function. That this equals the block's word-level function in C is `inline`'s correctness, which is not part of this premise.

**Who takes it:** **no guarantee over the door exists yet.** Every guarantee written over a template `lower` writes takes `LowerEval` as a hypothesis until (d) proves it.

**The lock.** On the branch the lock is unchanged. An `open` entry that no guarantee takes or reads fails the check, and an update drops it; a seed in `open_seed` fails outside an update. So I add no entry now and run no `--update`. **The lock run on `one-circuit` carries, under `security/lean-audit.json`'s `open`, once a guarantee takes it:**
```json
"FlockSoundness.Assumptions.LowerEval": {"owner": "@lean", "issue": "LEAN-335", "due": "2026-11-20"}
```
- The digest is the run's.
- If that run comes before any guarantee takes it, it carries nothing, and the guarantee's own PR lists the entry by hand.
- Either way, the update itself never adds it.

**(d), the proof:** in `security/Proofs/Flock/Soundness/Lower.lean`, as `theorem lower_eval : LowerEval`.
- The plan: `TmplValue.unroll_eval` at the root's one call, then an induction over `lower`'s Definitions, callee first, each row by `rowHolds_of_realizes`, relating `Tmpl.evalDef` to `evalDef?`.
- The primitive and call cases by Thursday 6:00 PM PST; batch and scan by Friday noon PST.
