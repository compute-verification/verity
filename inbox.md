[2026-11-12T19:04Z] a57c7cdda8ce
COUNTS to training, Thu 12 Nov 11:04 AM PST (19:04Z). TRN-86 item 2: the statement, what it reads of lean's, what is hard, my date.

Branch `cursor/hecharged-from-count-a8e2`, head 0479360c9c20, from `origin/main` with `main` at 0582899c4392 merged in (it moved at 18:10Z, after I branched). No PR yet: it opens as a draft, not marked, when the first try is back, so that its body says what ran.

## The statement

`Nci.Cut.SecurityProofs.HECharged : Nci.Cut.Guarantees.HECharged`, the statement `main` has, unchanged, with no hypothesis added. In `security/Definitions/Nci/Guarantees/Cut.lean:111`: for every registered circuit `d`, work `W`, values `val`, partition `P` into `m` blocks and set `G` of its blocks, with `β_w, β_x, b > 0`, `ValuesInjective`, `NoSharedPair`, (H) at `k` (`Assumptions.Hardness d ![W.allW, W.allX] val ![βw, βx] k`), (E) at `b`, each block of `G` reading at most `k` gates outside θ, all the blocks at most `T` between them, and the squares of what `G`'s blocks read at most `S`:

- `N ≤ ∑_{u ∉ G} |W.inside (P.gates u)| + workHEOn T S (β_w β_x) b O`, and
- where each MAC has at least `γ` gates and each block outside `G` at most `c`: `γ N ≤ |Gᶜ| c + γ workHEOn T S (β_w β_x) b O`.

`Work.inside` and `workHEOn` do not change (your rule 3), nor does any declaration's name.

## What it reads of lean's (`security/Proofs/Nci/Cut.lean`, on `main`)

- `bound_on_rows`, taken at `κ u` = what block `u` reads, one `β_x` for every block, and `Q = S / (β_w β_x)`. Not `bound_on_budgets_he`: its sum of squares is the exact one, and the statement has any `S` at least it, which `bound_on_rows`'s `Q` takes directly.
- `card_whole_eq`: the count's `∑_a |inBlock P W u a|` is `|W.inside (P.gates u)|`.
- `outOn_le`: `O_G ≤ O`, used under the root.
- Its definitions `inBlock` and `outOn`, read only inside the proof.

Of the file itself, kept: `hardness_mono` ((H) at `k` gives (H) at what each block of `G` reads) and `card_inside_mul_le` (the `γ` step, as on `main`). Deleted: `CountsOn`, `real_step_charged`, `countsOn_rows`, `HECharged_of_countsOn`. The file goes from 183 lines to 110.

## What the PR holds besides the proof

- The lock, `security/lean-audit.json`: `Nci.Cut.SecurityProofs.HECharged`, `HECharged_honest`, `HECharged_nontrivial` under `guarantees` (owner @training), a `witnesses` entry of kind `exclusion` like `CountHE`'s, and under `shape.frozen` the three "unstated:" entries for `Nci.Cut.Guarantees.HECharged*` out and one "frozen (no-function)" entry for the proof in, in `HE`'s words. Records by a run, not by hand.
- The nine docstring words (your addition 1). RAN the lint's own `sites` on my VM: `Guarantees/Cut.lean` 17 to 14, `CutCharged.lean` 9 to 3, both 26 to 17. `Guarantees/Cut.lean` stays 3 over its list entry (11): those three are older than the commit that stated `HECharged` (14 before it, RAN at c06ac4631a^).
- `rho/tests/test_min_cut.py` (your addition 2): the assertion is `== "proved"`, the test's name and one docstring sentence say so. TIE's twelve lines there are renames; line 198 meets mine, and the second to land merges.
- One sentence of `rho/README.md` that said "no lock lists a proof of it": it is not Python and not in TIE's diff, and it would be false on landing.

## What is hard

Nothing in the mathematics: it is lean's count with three rewrites. What can still bite:

1. The proof is untested. Its first try (r20261112-182328-4537, vy-nebius-cpu-14) has waited since 18:39Z for one of that node's Lean build slots, and the records run (r20261112-190245-8534, vy-nebius-cpu-7, launched 19:02Z) is queued behind 12; whichever builds first shows a Lean error if there is one. Where a second try could be needed: the cast of `S` through the sum of squares, `Work.inside` against `card_whole_eq`'s filter (one unfolding), and the closing `linarith` over the two roots.
2. The `shape.frozen` entry is a claims change by the lean-proofs skill's rule (an entry added there). It is the category `HE`, `HEShared` and `CountHE` are in, for the same reason: the statement is about any registered circuit, not about a function of the check's code, and nothing joins it to a verdict yet. Net, the frozen list is two shorter. Say if you want it another way.
3. The record's premises: I expect (H) and (E) as named assumptions and `ValuesInjective`, `NoSharedPair` as `domain`, as `HE`'s record has them. If the records run asks for a new `open` or `domain` entry, that is a finding and I bring it to you before writing it.
4. Slots, not work: this node's Lean build slots had a queue of over 25 minutes this morning, and the whole check with its kernel replay runs well over an hour.

## My date

Draft PR open, records committed, the Lean check with its kernel replay and the changed suites passed, ready for your review: Thu 12 Nov, by 6 PM PST. If the proof needs a third try or a slot waits past an hour: Fri 13 Nov, noon PST. Either way it is ready for your mark before Mon 16 Nov. I do not mark.

