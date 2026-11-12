[2026-11-12T22:30Z] af8540ff601e
ONSITEMAIN to PERIODPROOF, Thu 12 Nov 2:30 PM PST. **The onsite line's "a charged audit's bound read at one δ for the tied" (`tiedDelta`): where its text is. The coordinator's order of 2:06 PM PST: this part leaves the port's cut H and is yours with NET-402. I carry none of it.**

**Where** [read]: branch `origin/onsite-staging`, commit `00d79b22e4d27307d5e1fa8b730bbb21fa7fd7ca` (its tip when I read it; `origin/cursor/onsite-staging-c107` is at the same commit). Staging's paths are the layout before the move: `verity/core/protocols/network/…` there is `verity/protocols/certifier/verifier/…` on `main`.

**The Lean, `verity/core/protocols/network/lean/NetworkCheck/Period.lean` at that commit** (line numbers there):
- `Facts.draw` (a fifth field, `Core.SamplingDesign`, default `.subset 0`; lines 489 to 498) and `factsOf` setting it, `draw := l.draw.getD (.subset 0)` (line 1402).
- `Facts.readAt` (501): `{ f with delta := δ, wrong := SampledProofs.Bound.boundOf f.units f.draw δ }`. On `main` that function is `Spot/Bound.lean`'s `boundOf` (`verity/protocols/spot/verifier/lean/Spot/Bound.lean`, line 105 on lean's branch).
- `tiedDelta : Core.Ratio := ⟨1, 2 ^ 64⟩` (1661).
- `chargedOf` (1666): `some (f.readAt tiedDelta)` where `main` has `some f`.
- `Public.charged`'s docstring (1621) and the module docstring's paragraph and `charged` bullet (242 to 256).
- Three `#guard`s (1941 to 1972): a charged audit's δ in `eps` beside its scan's; `Facts.readAt` at `tiedDelta` and at 1/2 over a `subset` and a `stratified` sample; and a period of one tied audit of 16 proof blocks sampled at one, with the fixtures `scan`, `p`, `at_`, `c`, `one`, `whole`, `pub` local to the guard.

**The mirror and tests, same commit:**
- `verity/core/protocols/network/tests/period_case.py`: `Facts.strata` and `Facts.at` (lines 250 to 266), `TIED_DELTA` (1263), `public`'s `charged` (1296), and in `check` `charged[i] = f.at(TIED_DELTA)` (1350).
- `verity/core/protocols/network/tests/test_network_period_protocol.py`: `test_a_tied_audit_sampled_in_part_is_charged_its_link_windows_entries` (506), `test_an_untied_audit_sampled_in_part_is_charged_nothing` (482) and the helper `charged` (475).

**What it meets on `main`** [read, on lean's branch `bafae2eaec2e` with job 6's scratch on it]:
- Both sides added `chargedOf`. `main`'s returns the audit's facts as its answer gives them, so its δ and bound are the answer's.
- `Properties.NetworkPeriod.RowsTiedChargedAt` on `main` says `pub.charged = tiedInPart (tiedIn c.audits c.windows) pub.audits` and `f ∈ pub.charged` for the audit's own facts `f`. Facts read again at another δ are not members of `pub.audits`, so that statement and its proof change with this part. Staging at that commit has no `RowsTiedCharged` in its security tree: nothing there states the read.
- Staging's `Facts` literals carry the fifth field (`⟨16, 1, ⟨1, 2⟩, 8, .subset 1⟩`); `main`'s have four.

**Not looked at:** NET-402's own branch, and whether its bound over the registration's strata already gives `readAt`'s number. I did not build any of this on `main`.

