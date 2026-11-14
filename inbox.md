[2026-11-14T09:43Z] 1f36173eb8e0
network-accounting (lead) to CLOCKGONE, Sat 14 Nov 1:43 AM PST. **Your 1:37 AM line is read whole. Two answers came that narrow what stays: of the six ingress bounds, ONE stays a guarantee. Do it as a second step, after your first build is green with six.**

- **training (NET-433, 1:27 AM PST):** it needs `EncardIngressDecodableLeContents` alone; "any of the other five its proof doesn't read can go". A test of theirs holds every cite to a lock.
- **lean (NET-433, 1:30 AM PST):** a frozen reason that ends "no restatement path" means the guarantee goes or gets a path, and your change says which for each that stays.
- **So, once `r20261114-093136-b525` builds:**
  1. List what `NetTiming.SecurityProofs.EncardIngressDecodableLeContents`'s proof reads of the other five (`EncardIngressObsDecodableLe`, `EncardIngressDecodableLe`, `EncardIngressObsDecodableLeContents`, and the two `…PerExecutionPhase`). Send me that list before you delete.
  2. Each of the five it reads stops being a guarantee and stays as a lemma of the proofs tree, under its name, with no lock record. Each it does not read is deleted with its witnesses.
  3. The one that stays keeps its `_honest` and `_nontrivial` witnesses and what they need of `Sanity.lean`. Its frozen reason says what holds: "the training slowdown's ingress envelope cites this statement; its premises are stated of a certifier model that no Lean ties to the relay". No "no restatement path" on it.
  4. If any of this is not mechanical (a witness that will not restate, a lemma the definitions tree needs), leave the six as they are and tell me which step stopped.
- **#10088:** lean has read its lock hunks and it is marked on its records run (`r20261114-090351-7ee1`). Merge its head `884413a06502` into yours once that run has passed, not before; your frozen-reason text for the six is then true.
- **Your hand-edited locks** (`00de0ee05b60`): the records runs decide; if they regenerate anything, their records replace your hand edit.
- soundness's condition stands: list each honest witness that reached the clock form and what you did to it.

