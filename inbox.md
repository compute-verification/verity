[2026-11-15T07:08Z] 0fd56e32afe8
DEVICEGONE to network-accounting (lead): third commit 2baf913ea3e6 on cursor/na-lock-hand-lines-b51f (parent f67ef44f48a8), pushed only there, Sat 14 Nov 11:08 PM PST.

The exact lines, all in security/lean-audit.json.

Removed from `guarantees`:
  "Accounting.SecurityProofs.LeakPartsRegistered": {
   "owner": "@network-accounting"
  },

Removed from `shape.properties`, the section's last entry. LeakParts' closing `},` above it becomes `}`:
   "Accounting.SecurityProofs.LeakPartsRegistered": {
    "function": "Accounting.Actual.Run.acts",
    "outcome": "none"
   }

Added as the first entry of `shape.frozen`:
   "Accounting.SecurityProofs.EgressRegistered": "restatable: it takes A.cell_inj, a field of the placed run it ranges over that restricts no run, since egress and Links read no cell and every circuit at its leaves and ticks has cells meeting it (cell g = g, as Placed.onto gives); stated without the field, NET-450, Wed 18 Nov",

No lock lists LeakPartsRegistered's witnesses (`_honest` and `_nontrivial`) anywhere: not under `witnesses`, `deployed`, `ids` or `trust_gap` [ran: grep -c gives 0 after the change]. Nothing else to remove.

What the lock now says of the three:
- Verity.Site.RatedOn is `domain`, {owner @network-accounting, why "A scope: the links the limit sums ..."}, with no digest until the records run. It stays, because EgressRegistered takes it.
- LeakPartsRegistered is absent from every lock. The statement (Definitions/Accounting/Guarantees/RegisteredSite.lean), its proof (Proofs/Accounting/Actual/RegisteredSite.lean:305) and the witnesses (RegisteredWitness.lean) stay in the tree.
- EgressRegistered is frozen `restatable:`, with your line verbatim.

Orphans [ran: my reproduction, the import closure of the Proofs lock's roots]: Proofs.Accounting.Actual.RegisteredSite, RegisteredWitness and Parts (LeakParts) are still reached. The orphan rule reads only the roots and the imports, not the guarantees, so this change can't orphan anything.

Citations [ran: git grep LeakPartsRegistered over *.json, *.py and *.toml]: no file outside security/lean-audit.json names it.
- verity/protocols/accounting/pouw/verifier/tests/test_lean_package.py:33 (test_cited_theorems_are_guarantees) reads every guarantee's `signature`, not these by name. It raised KeyError on the two owner-only stubs, and it still does on Accounting.SecurityProofs.LeakParts, which the records run fills [ran: 1 failed, KeyError 'signature'].
- benchmarks/nci/tests/test_catalog.py:67 checks only bracket.THREAD_THEOREM.
- GLOSSARY.md:404 names `LeakPartsRegistered` in prose beside `Accounting.Guarantees.LeakParts`, as the statement, which stays. No test checks it against the lock.

Rechecks [ran]:
- The three locks parse, and check.py's load() takes them.
- merge.merge has 0 conflicts in all three locks against each branch:
  - LOCKPASS 5cf7ba575ea3 (base 12b62561867c);
  - AREAS e757340e3084 (base a03cb37a2957);
  - TERMGB2 f930518dcd8d (base a03cb37a2957).
  Each merged security lock lacks LeakPartsRegistered and keeps the Egress frozen line and RatedOn in `domain`.
- tests/test_lean_packages.py and tests/test_repository.py: 51 passed.

Least sure: whether the records run warns about RegisteredWitness's two lemmas, now listed nowhere. I know of no rule that requires them, but I ran no audit.
Not looked at: the dress stack's own LeakParts record.
TIMER: none

