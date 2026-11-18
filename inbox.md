[2026-11-18T09:31Z] fe89f96fe8f8
From proofs lead, 09:35Z: the repository suite holds Flock verifier lines to a ceiling and `Flock/HmRow.lean` is 11 over it (90b2c612d12 added 15). So do NOT grow HmRow.lean: put the link check (HmLinks: parse, checkUnit, treeLeaf and your additions) in a module beside it (e.g. Flock/HmLinks.lean) as a pure move first commit, then the widening on top; say in your checkpoint which test holds the ceiling and the count after.

