[2026-11-12T07:52Z] 22d843a71aab
CROSSTICK, job G part 2, status at 11:52 PM PST Wed 11 Nov. Your 11:11, 11:18 and 11:34 PM lines are applied to the count. No answer needed unless the profile (point 2) is wrong.

**Head:** `cursor/na-span-count-e7c1` at `57da7538637d1f21a1b59667abfb3909c2c9bbfc`, pushed. Three commits above `a18d867f37`. [ran]

1. **The tail is ⌈(D + ⌊(E − 1)/r⌋) / T⌉**, one definition a side (`tail` in `Span.lean`, `span.tail`). The edge case (D 8, r 2, E 17, T 16 → 1) is a `#guard`, with E 18 → 1 and E 19 → 2 beside it. In the difftest it is two cases under that schedule: `edge`, accepted with a tail of 1, and `edge-tail`, a second execution phase of tail, rejected as outside the span. A mirror with the earlier formula is caught on it (Lean rejects, that mirror accepts). The mirror's own test holds the tail to the buckets a cell leaves in (bucket a + D + ⌊i/r⌋ over every admission bucket a of the end mark's phase and every cell i) on 300 sampled schedules. [ran locally: 5 mirror tests pass; every change at 30 seeds, 0 wrong]
2. **Public: `{"egress_bits", "tail"}`.** The count of phases is out of the public output; nothing of mine reads it there. It is still in the **profile**, `{"link", "phases"}`, as the period check's profile gives its count of link-windows. Say if it should leave the profile too: one line a side.
3. **Wording:** the count's docstrings said "the bytes the network certifier holds at the end mark"; they now say what it reads (a server's bytes only on a connection it opened when it admitted that client's connection). No name of the watermark.

**The merge:** `git merge-tree --write-tree --name-only origin/cursor/lean-remote-merge-landing-fc8d HEAD` at `de28a691126b` names no file; `origin/main` `d6075b75d289` neither. [ran]

**Runs at the earlier head `a18d867f37`** [ran]:
- r20261112-070911-544f, the whole root suite: "1 suites: 1 passed (0 cached), 0 failed; 5.8 min of test time".
- r20261112-070941-f6bf, the Lean package's check: "AUDIT: PASS", 1709 declarations in 11 modules, axioms propext, Classical.choice, Quot.sound, 4 guarantees. It notes 77 definitions of the package read by no guarantee, 43 of them the count's, which has no guarantee yet. A note, not a failure; `unread-caps.json` holds `security/` areas only [read].
- r20261112-070835-2772, the suites the change touches: **cancelled by me** after 38 min, at 9 suites passed, 1 cached, none failed, of 79. Its head was superseded, and the three commits touch 54 of those suites again (they take `verity` whole as an input), so its passes would not have carried to the new head.

**At the new head:**
- `verity lean try` on `Span.lean` with `--siblings`, at `2d7f5c7aa4` (the head differs from it in docstring words only): r20261112-072639-8727, lean exit 0 in 5 s, no sibling name clash, no theorem in it. It waited 19 min for a slot: every check slot was taken at 11:42 PM. [ran]
- Launched once each at `57da7538637d`: r20261112-074642-69ee (the difftest module whole, running), r20261112-074710-d7db (the Lean package's check, running), r20261112-074734-8b2d (the whole root suite, waiting for a slot), r20261112-074802-d610 (the suites the change from `62bf29346f76` touches, `--quick`, waiting). The last is 79 suites one at a time in a 15 GB slot; 9 took 37 min at the earlier head, so it is hours. I do not wait on it: I report its id.

**Next:** the model lines' branch from lean's landing branch, then the count's pull request, opened unmarked, when the first three runs are back.

