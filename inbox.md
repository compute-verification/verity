[2026-11-12T08:23Z] a42407d5918b
CROSSTICK, 12:23 AM PST Thu 12 Nov. Two things you asked for, then the runs.

**1. The model lines' branch (no pull request opened).**
- Branch `cursor/na-span-model-lines-4c2e`, head `f2f78fd25cad13727a02e595f97e9737a11acc69`, one commit on lean's landing tip `de28a691126b` [ran: git push, git rev-parse].
- Three new files and three import lines, 164 lines: `security/Definitions/NetworkCertifier/Marks.lean` (`markPhase`, `spanTail`), `security/Definitions/NetworkCertifier/Guarantees/Marks.lean` (`HeldCutInTail`, `MarkOrders`, `SpanCovers`), `security/Proofs/NetworkCertifier/Marks.lean` (the proofs and `NetTiming.SecurityProofs.{HeldCutInTail, MarkOrders, SpanCovers}`); one import each in `security/Definitions.lean` (two lines) and `security/Proofs/NetworkCertifier.lean` [read].
- `verity lean try --siblings` on each, at that head [ran]:
  - definitions, `r20261112-075901-dc7d`: "lean exit 0 in 29 s"; no theorems in it.
  - guarantees, `r20261112-075921-dd2c`: "lean exit 0 in 12 s"; no theorems in it.
  - proofs, `r20261112-075940-2c72`: "lean exit 0 in 35 s"; "none of its 8 theorems uses an axiom beyond propext, Classical.choice and Quot.sound".
  - Siblings: no built module that doesn't import the file declares a name it does. Most modules had no build to compare (454 and 3,566 unchecked), so a name clash with those is not ruled out [read].
- `spanTail D N T` is ⌈(D + N − 1) / T⌉ on N, the most buckets of egress a lane holds. The count's tail is that at N = ⌊(egress_cells − 1) / r⌋ + 1, which is your formula [argued; the guards hold D 8, N 9, T 16 → 1 and N 10 → 2].
- (i)'s docstring is your sentence, ending "it has read by then"; the watermark is not named [read].
- Not done, and I ask before doing it: the three guarantees are in no lock. `security/lean-audit.json` needs their entries and `--update` writes their records. I have run no `--update`.
- Not checked: the package's whole Lean check with these files in (only the three tries).

**2. The registered read in the count.**
- Merged #9188's head `a8d956c8dd14` into `cursor/na-span-count-e7c1` (merge commit `d8559f11ab9e`, no stop), then one commit of mine. New head `57dd66f8585ddbef5764f789450a90e423517bcc`, pushed [ran].
- The call, as you wrote it: two optional keys in the parameters, `site` and `head`, both or neither; with them `paramsOf` calls `Registered.siteOf`, then `Registered.registered S head link key` with the key as its lower-case hex string; a refusal there refuses the parameters. `Params.registered : Option Registered`. The mirror calls `registered.site_of` and `registered.registered`. The count stays on `CoreProtocol.Site` [read: `tests/test_public_boundary.py` passes locally].
- Cases added: `registered` (accepted, and its answer equals the one without the site), `site-link` (no such link; the certifier's other half; host to host; a bare uplink; certifier to certifier; an end nothing registers), `site-key` (another key; upper case; a byte short; odd length; empty), `site-alone` (one of the two keys), `site-form` (23 malformed sites and heads). In the mirror, 200 seeds each: no wrong verdict [ran].
- A refusal by the registered hardware names the link, a certifier id and a key. The parameters are public, so I let those reasons hold digits; the check's own reasons still hold none. Say if you want it otherwise.
- The lock: `layers["NetworkCheck.Span"].imports` gains `NetworkCheck.Registered`, by hand. No `--update`, no record moved.
- No meeting in an `untraced` list: mine is in `network/pyproject.toml`, #9188's in `verifier/pyproject.toml` [read].
- It is its own commit above the merge, so it can be dropped if #9188 does not land first.

**3. Runs.**
- At the head before the registered read (`57da7538637d`) [ran]:
  - root suite, `r20261112-074734-8b2d`: "passed repository 311 passed in 337 s"; "1 suites: 1 passed (0 cached), 0 failed".
  - span difftest in a scratch merge into lean's tip `de28a691126b`, `r20261112-075408-1110`: "passed verity/protocols/certifier/verifier/network 9 passed in 49 s".
  - changed suites, `r20261112-074802-d610`: I cancelled it. It waited 25 minutes for a slot and its head was already superseded.
- At the new head, launched once each at 12:15 AM, waiting or running: the span difftest `r20261112-081515-ee6f`, the root suite `r20261112-081544-7ce7`, a try of `Span.lean` `r20261112-081515-43c4`. The package's Lean check, the changed suites and the scratch-merge difftest follow once the difftest shows the Lean builds.
- Locally at the new head [ran]: the mirror's 6 tests pass; the same 6 pass in a scratch merge into lean's tip; six root invariants, 163 passed; merge-tree against the landing branch and against `main` names no file; ratchet gate from `main`'s tip: retired terms 0 → 7, the same seven sites, nothing else.

The pull request is not open yet: I open it, unmarked, when the new head's difftest and Lean check have passed.

