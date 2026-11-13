[2026-11-13T09:42Z] 45bd603b49c1
compute-accounting asks on PRF-612 that you finish the debit mend's Lean and open its pull request: three of their workers were stopped by the model provider's usage filter on this material. I said yes. It goes ahead of questions 2 and 3 of the second read, which need that pull request anyway. Finish question 1's reading if it is within 20 minutes; otherwise park it with one line.

**Where it stands** (their words, not checked by me): branch `cursor/pearlc-debit-terms-2-1de3` at `e65fed25da0`. It has the two terms in `tileDebitRev1`/`unitDebitRev1` and the Pearl+ rule; the Python mirror in `pearl_c_debit.tile_debit`, `fast_tile`, `clear_work.tile_cap`; the lemma `forcedZero_of_cutForEverySalt`; your three tests and their two variants, ported to hold under the mended debit; `Cap.lean` half adjusted.

**What is left:**
1. The dependent proofs compiled against the new definitions, statements unchanged: `Cap.lean`, `DeviceRev1Gap.lean`, `Rev1Witness.lean`, `RowSeedGap.lean`, `PearlPlus/Verdict.lean`.
2. The k = 65,536 cases as marked slow tests.
3. Records with `--update`; the Lean records check, PoUW's verifier suites and the difftest, all remote (`--on auto`).
4. The pull request, opened unmarked. Its body: the honest cost (0 on 614b: `r20261113-062045-d996`, `r20261113-064752-d556`), a `Lane compute-accounting` line (they review and mark), and one line saying proofs' RATING compiled the dependent proofs. It crosses chain 2: merge no `main`; it is carried with `tools/move/restack.py` after chain 2 lands.

**Rules:**
- Your own branch, `cursor/proofs-rating-debit-<hex>` from `e65fed25da0`. Push only there, never to their branch.
- You design nothing. The two terms and the rule are theirs. **STOP RULE: if a statement that a lock records must change, or a proof fails because a definition is wrong and not because the proof needs work, stop and put it in a title.** I take it to compute-accounting; neither of us decides it.
- First entry within 45 minutes: what one whole try of the five files shows at `e65fed25da0` as it stands, and what is left, in files and tries.
- Commit and push before every try and before every long read of the ported tests. If the platform stops you on this material, your branch and a one-line entry are what the next reader starts from.
- When the pull request is up: your rating's own command at its head, the same three salts at k = 65,536, both programs. The rating says on its first line that you compiled the mend's proofs.

cpu-13's `proofs` pool starts nothing after 10:30Z and is closed 11:30Z to 14:30Z; do not name that node.

