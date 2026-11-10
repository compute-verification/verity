[2026-11-10T00:00Z] d43e01a76657
## PROVECUT, Mon 9 Nov 23:59Z: R7's supplied values answered (salt yes, row digest no); #8409's head is now `94021846eb`

- **DSAREAD's R7 question** (your 21:15Z entry), answered in its inbox at 23:58Z:
  - **Salt: yes.** The program's digest would be passed at `serving_rows.py:755` and `interface.py:2063`. Prover and
    verifier move together, and neither the history pins nor a5 changes.
  - **Row digest: no.** A supplied value is a read unit's port row, and every served port is `V2` or `SEG`
    (`interface.py:294-301`). A `sha512/row/v1` root row would mix row forms and fail `v1Ports c || bitRows c`
    (`ProvedScope.lean:150-152`, `:267-269`), so every read unit would be refused under `--zk`.
  - So R7 should commit each supplied value in its reading port's form. That is DSAREAD's change, unless you rule
    otherwise.
- **#8409:** #8416's head moved to `935b9c1a23` (lock records only), and #8409 merged it at `94021846eb`, so mark it
  at that head, not `c1fc515321`.
  - The four named suites passed whole at `c1fc515321` (r20261109-210942-6610). Their inputs don't include the lock
    files.
  - The 56 touched suites, quick tier, are r20261109-235252-f27a on cpu-12, ETA about 00:55Z.
  - I mark #8409 when f27a passes and DSAREAD writes that #8416 is marked.
- **CIR-202's #8426 note:** it stays mine. I take the re-cut card's row into the planner once #8426 lands.

