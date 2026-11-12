[2026-11-12T19:45Z] 6fe5cb24e2d3
COUNTS to training, Thu 12 Nov 11:45 AM PST (19:45Z). TRN-86 item 2: the records run asks for a new `domain` entry, and for the statement's hypotheses to be named. Yours to rule before I write it. Nothing of it is written.

RAN r20261112-190245-8534 (vy-nebius-cpu-7, at 0479360c9c20, `lean_changed.py --records security/Proofs --update`): `security/Proofs` PASS (198509 declarations in 3967 modules, the proof built); `security` FAIL, one finding that is not a record difference:

"guarantees: Nci.Cut.SecurityProofs.HECharged takes (∀ u, u ∈ G → (d.unregistered (P.gates u)).card ≤ k); (∑ u, (d.unregistered (P.gates u)).card ≤ T); (∑ u ∈ G, (d.unregistered (P.gates u)).card ^ 2 ≤ S): an inline hypothesis, neither a named assumption the lock lists, another listed guarantee's statement nor a plain range … name it as a Prop and list it in the lock …"

So `HECharged` as `main` states it cannot be listed: its three hypotheses on what the blocks read are written out, where `HE` takes the named `ReadsAtMost` (`domain`). The lock's `shape.inline` list holds only what was inline when the rule came. The witnesses rule (READ `check.py`'s `exclusion_failures`) then also wants the nontrivial witness to drop a named premise with every other one holding, and the honest one to conclude each by name.

WHAT I PROPOSE: one named Prop, beside `ReadsAtMost` in `security/Definitions/Nci/Cut.lean`:

`def ReadsOn (d) (P : Partition d.D m) (G : Finset (Fin m)) (k T S : ℕ) : Prop := (∀ u ∈ G, (d.unregistered (P.gates u)).card ≤ k) ∧ ∑ u, (d.unregistered (P.gates u)).card ≤ T ∧ ∑ u ∈ G, (d.unregistered (P.gates u)).card ^ 2 ≤ S`

- `HECharged` takes `ReadsOn d P G k T S` in place of the three; its two conclusions, `workHEOn` and `Work.inside` stay word for word (the test's pinned text holds). It says the same: the Prop unfolds to the three.
- `HECharged_honest` concludes `ReadsOn Toy.d Toy.P2 {0} 2 4 4` (today's three conjuncts, named); `HECharged_nontrivial` has `¬ ReadsOn Toy.d Toy.P1 univ 0 0 0` in place of `¬ ∑ … ≤ 0`: the same instance.
- The lock: `domain` gets `Nci.Cut.ReadsOn`, owner @training, why: "The blocks the count is over: each block of G reads at most k gates outside θ, all the blocks at most T between them, and the squares of what G's blocks read sum to at most S."

WHY ONE NAME AND NOT THREE. With three names the nontrivial witness must drop one and keep the other two true, and no such instance beats the bound at the toy (N = 2, O = 2, unit costs): any T at least what is read is at least 3; with T dropped, G empty charges both MACs (2), one block in G needs S ≥ 9 and gives ½√18 ≈ 2.12, both of `P2`'s need S ≥ 8 and give ½√16 = 2, one of them gives 1 + ½√8. Three names would need a new toy. One name keeps today's witness, as `HE_nontrivial` drops `ReadsAtMost`.

WHAT IT COSTS: three statements in `Guarantees/Cut.lean` restated in their hypotheses, one definition and one `domain` entry added, the proof's `intro` takes the Prop apart (one line). No conclusion, no name and no Python changes beyond what you have.

The other ways I see: leave it unlisted (item 2 not done), or three names and a second toy (more of lean's vocabulary for the same claim).

I prepare the one-name form in my working tree and try it, and commit it only on your word. The date holds if the word comes within the hour.

