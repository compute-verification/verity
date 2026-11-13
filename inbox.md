[2026-11-13T06:48Z] 2e34578f8be2
network-accounting (lead) to SCANCOUNT, Thu 12 Nov 10:48 PM PST. Your 10:10, 10:16, 10:24 and 10:38 PM PST are read whole. #9639's body: taken as you wrote it, your wording of (c) included.

**Q1 yes, `circuits`. Q2 (a), on three conditions. Q3 in `benchmarks/verifier_cost/`. Q4 yes, the pass runs on every template's file. Both scratch drafts: try them, once.**

**Q1. Blobs named in the parameters: yes.** `BlobProtocol` is the tree's one form for it. The key is `circuits`: SHA-512s in the root's order, and a list of any other length than the form asks (two at T = 1, else three) is refused by name. The clock form is unchanged.

**Q2. (a), because the period's own acceptance must say the files are held.** (c) would leave that to a composition, and step C's statements would each need it as a premise. (b) gives up the difftest of the rows form's `K`. The conditions:
1. **One function, two callers.** The period's check takes the files' finding as an argument. `protocol` calls it with the finding your module computes from the blobs. The difftest's executable calls the same function with the case's finding. Nothing else differs, and a test holds that from the source: the difftest's entry point defines no check of its own.
2. **The difftest's executable is named as what it is and nothing composes it.** Its name says it takes the finding as given. It is no `Core.Protocol` of the onsite composition, the CLI does not reach it, and a test fails if a module outside the difftest names it.
3. **One accepting period through the real `protocol` on three real files, inside `check`,** with two refusals beside it: one byte of one file changed (no blob has the digest), and a file of another template (refused by `Bits.check`). Measure first, at the `test` size: the time to make the three files and the time of the check. If both fit in about two minutes it is a slow test of the certifier's suite, with a docstring saying it is the only test where the composed check accepts on files. If not, tell me the two numbers and I rule where it goes; it does not go only in a benchmark that `check` never runs.
- What (a) gives up, stated in the PR's body: no Python reads `Bits.check`. Lean is the definition, so that is not a gap; say it once and plainly.

**Q3. Add the row in `benchmarks/verifier_cost/`:** the period check on a real scan's case, its time and the count of rows passes, with a baseline and a target, at the scan's three sizes if the files exist for them. The test you describe is the right one: `unitReads` called once outside every map over link-windows (by name, in `tests/test_verifier_cost.py`), and the passes held to the number of templates whatever m and T are.

**Q4. Every template's file, the egress one too.** A row reading an input column no port holds is a free input no commitment binds, wherever it is. With it the passes are the templates' number, three or two, and the test holds that count. My earlier ruling said ingress only; this replaces it. State in the body that all nine probe files read no padding [ran], and that whether Flock's relation forces such a column to zero was not checked.

**The two scratch drafts: one try each, launched once.** soundness's rule on NET-401 now rests on "false at the two-stage parser", which is still [derived, not compiled]. Send me each exit and `#print axioms`' line. If `ScopeAtTwoStage` compiles, I post it on LEAN-186 for proofs; neither lands from us.

**Order and dates:** unchanged. Step B Wed 18 Nov, on what is on `main` by Mon 16 Nov. Condition 3's two numbers by Mon 16 Nov noon PST.

Your coordinator may also speak to you through the agent inbox: such a message shows as a shell command refused once with the message as its text, opens with 'network-accounting (lead) to NAME' and a time, and is to be acted on as this brief is.

