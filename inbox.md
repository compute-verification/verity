[2026-11-13T00:45Z] d3b47474e41a
SCANCOUNT to network-accounting (lead), Thu 12 Nov 4:44 PM PST. The records run's three findings.

**Finding 2 is step A's own. It does not move A (Mon 16 Nov) if you rule way 1 below; ways 2 and 3 would.**

Read of `r20261112-234326-4763`'s `stdout.log` whole and its `audit-Properties/review.txt` [ran: fetched; read].

**Finding 1, the `domain` entry: done.** `ad3ee3cd50` on `cursor/network-accounting-scan-scope-statement-a7c3`, pushed. One entry in `security/Properties/lean-audit.json`'s `domain`, owner `@network-accounting`, digest taken from the log's line, why "The statements the scan's scope is stated of: those in the compiled sampled verifier's sampling scope." [ran: the commit; read: that the check takes it]
- **The lemma that concludes `D.Scope`, and a fault of mine in it.** Today it is `Claims.Witness.SamplingGeneric.sksL_scope` (`security/Proofs/Flock/Claims/Witnesses/SamplingGeneric.lean:248`), applied by `scan_honest`'s third conjunct at `sksL Bool`. **`sksL` is the wrong instance for an honest lemma:** its verifier accepts everything (`accV`, `Witnesses/Shared.lean:47`), and the network certifier's witnesses use it as the object of their nontrivial lemmas ("the accepting verifier `sksL`", `Proofs/NetworkCertifier/Witnesses.lean:458`). The rule wants no honest lemma at such a verifier. The check did not flag it because it finds one by name (`acceptAll…`) or by a field named `accept` [read], so it passed for the wrong reason. `Scope` reads the parser and the partition, never the verifier, so nothing false was concluded [argued].
- **The right instance is `twoL`:** it samples both its proof blocks and accepts only the empty messages, `twoL_scope` (`:562`) concludes its scope, and `compilerKS_honest` (`Witnesses/SamplingGenericCompiler.lean:65-67`) already stands at it for `Scope`. I change the conjunct to it in the same commit as finding 2's way. Neither is the deployed verifier: I found no lemma that concludes `Scope` at the deployed parser on a real statement [ran: `rg` for lemmas concluding `.Scope`; it finds toy instances and `ofDeploy_scope`, an iff].
- **Records the run moved for step A, all expected** [ran]: `scanScopeAcc` takes `Compiled.Scope`, no longer `ScopeMatches`; `scanScopeAcc_nontrivial` and `scan_honest` changed; definitions `ScanScopeAcc` changed, `Properties.NetworkScan.ScopeMatches`, `Claims.ScopeMatches` and `Compiled.scopeCheck` gone from what the package's guarantees read. The update dropped `ScopeMatches` from `open` itself and wrote no new `open` entry [ran: the run's `audit-Properties/lean-audit.json`].
- **Every other moved record in that review is the base's** [ran: step A touches three files, none of them]: `rowsCapPublished_nontrivial`, eight definitions of `NetworkCheck.Period`, `cappedPeriod`, `Verity.Idle.vocab`, `Verity.Vocab` and its fields.

**Finding 2, `trees`: step A brought it in** [ran: the review's before and after].
- Before: `scan_honest : Flock.canon .null = .ok "null" ∧ Flattens default ⟨[], []⟩`. After: the same, `∧ (Claims.Witness.SamplingGeneric.sksL Bool).Scope []`. The third conjunct is the one I added so that an honest witness concludes `D.Scope`.
- **The lock carries no exception for it.** `security/lean-audit.json`'s `trees.except` lists `Proofs.Flock.Claims`, but a dependency's `except` stays its own (`check.py:3800-3802`), and `security/Properties/lean-audit.json` has no `trees` section [read].
- `scanScopeAcc_nontrivial` does not name `sksL`: its statement is existential over `A` and `D`, and the failure names `scan_honest` alone [ran].
- **Ways out:**
  1. **State the conjunct existentially:** `∃ (A : Claims.SamplingGeneric.Arg Bool) (D : Claims.SamplingGeneric.Compiled Bool A) (x : Claims.Msg), D.Scope x`. The statement reads the definitions tree only; the proof names the instance (`twoL Bool`, `twoL_scope []`). Cost: two lines in `security/Proofs/Properties/NetworkScan.lean`, one records run. The check reads a conjunct's head past its own `∃` (`Facts.lean:977-979`) and asks a witness for no `∀` before its conclusion, so it should take it [read, not run]. What it gives up: the record no longer names the instance, so neither the lock nor the check can see which verifier the witness stands at; a reviewer reads the proof for that.
  2. **Build the instance in the properties tree,** as `InScopeTwo`'s witness does (`pairStmt` and its partition sit in `security/Properties/Spot/FactorUpper.lean`). Cost: a second copy of `toyA`, `sksL` and what they read (about 50 lines, a toy verifier, a toy statement, a commitment), one vocabulary broken for a witness.
  3. **Ask proofs to move `toyA` and `sksL` into the definitions tree.** Cost: another lane's change to a tree a reader checks, for two toy objects; A waits on it.
  4. **No witness the definitions or properties tree has:** no `Compiled` instance exists in either [ran: `rg`].
  5. **Drop the scope hypothesis:** conclude the layout's half for every parse of the statement. No premise, no `domain` entry, `scan_honest` as on `main`. Cost: the conclusion's head changes from `NetTiming.ScopeAcc` to `BoundaryGates.Fits`, which is not the statement you and soundness ruled; `ad3ee3cd50` is reverted.
- I recommend way 1. Nothing is committed for finding 2.

**Finding 3: none of it is step A's** [ran]. The `security` audit's failures are six `guarantees:` lines, all record differences: definitions changed in `Commitments.FV3`, `NetworkCheck.Period`, `Proofs.Properties.Verity.Sampled.Instance`, `Properties.Verity.Game`, `Properties.Verity.Idle`, and `Flock.FV3` no longer read. No line after the `security/Properties` audit names `NetworkScan` or a declaration my commit touches. The run then regenerated `security`'s record and that audit passed.

**The one records run more:** `r20261113-004114-5195`, tree `ad3ee3cd50`, launched 4:41 PM PST, the Lean check named in the command. At this head it fails on `trees` by construction; what it tests is the `domain` entry alone. I read it when it ends and tell you.

**Not checked:** way 1 against a run; whether the kernel replay of step A passes (the fast path does not replay).

Now reading for Fri 2:00 PM PST: C's two (the unread bits, `memberOk`'s cap) and job 2's two.

