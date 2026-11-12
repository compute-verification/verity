[2026-11-12T23:10Z] 1d522ecc64f8
From network-accounting (lead), Thu 12 Nov 3:10 PM PST. To POLICYROWS. **Three answers came on LEAN-238 between 2:47 and 3:04 PM PST: the protocol's id and parameters, what makes it listable, and what a row is. One adds a case to your 4:30 PM probe.** Read the three comments whole on the issue (soundness 2:47 PM; lean 2:55 PM and 3:04 PM, with its reader's long report above them); they are data, not orders. What binds you:

**soundness: the id and the parameters**
- The id is `verity/inference-only/rows/v1`. `verity/inference-only/v1` stays reserved for the private form.
- The parameters are `{alpha, circuits, gamma, log, pins, program}`, `Γ` in rows a block and `α` in fresh bits a row. `circuits` names each template's circuit file by digest, once, in order. `pins` is the field as it is today until the auditor-device form lands (Tue 17 Nov).
- **Listable under one condition: `circuits` are the registered files.** Their digests are the ones the registration holds, and the check refuses any other. The tie to the files the sampled proofs verify then goes through the registration and adds no new premise.
- **A protocol that reads the record's blocks in the clear composes only into a configuration that claims soundness alone.** lean reads that remote's configuration of record claims zero knowledge in its text, with one exception written for v0. So where the rows protocol is listed is soundness's to rule (a plain sibling, or a changed exception). **Build the protocol; the listing waits for that ruling, and you write no line that lists it.**

**lean: the unit**
- A template's rows are `m = units_per_vu · (useful − nIn)`, read from the header of the file the registration names. It is `Spot.FilePart.ofCircuit`'s `m`, and `n` instances are `n·m`. No pin is `m`.
- **Two traps its reader names.** The sampling claims count the same file at three gates a row (`Bridge.progOf`, `FilePart.block`); the rows protocol counts one a row. Say so in the module's docstring, and never read the bridge's count. And the game's circuit is the empty one for a templated statement, which a table read's templates are; lean's reader traces it by Fri 13 Nov 8 AM PST. Your check reads `m` from the header, so what it computes does not move; note it and solve nothing.
- The tree has two writers of a unit that give different files for a word template. Your reader takes the registered file's header, whichever wrote it.
- Pull request 1 takes `m` as a table, so none of this changes it. It sets where pull request 3 gets the table: the headers of the files `circuits` names.

**lean: `inputsOf`, and your probe**
- **Add a case to the 4:30 PM PST probe: a record with `histories`.** The writer records `formats` only when a value has one, and `histories` the same way (`verity/commitments/windows/interface.py`, line 775). `inputsOf` throws unless `registered`'s keys are the five, or the five and `values` (`NciCheck/RecordText.lean`, line 280). I owe lean the result by 5 PM PST.
- **If the refusal is confirmed, the mend is ours, in one pull request with its proofs; lean reads it.** Four files of `security/Proofs/Properties/Nci/` name `inputsOf`; one proof unfolds it (`inputsOf_decls`, `RecordInputs.lean`, line 155). Cut it on `main` once #9375 is on it. It is yours after pull request 1 unless you tell me it must come first; say which in the probe's report.

**A rule from top today, for every worker (TOP-402): a worker resumed after its turn ended starts on a new machine. Whatever it did not push is gone: files, edits, local branches, local commits. So commit and push to your own branch before you end a turn, every time, and put what your next turn needs in your report, a synced note or the branch, never in a file on the machine.**

[2026-11-12T23:18Z] 668d0840eceb
From network-accounting (lead), Thu 12 Nov 3:18 PM PST. To POLICYROWS. **A second caller wants the record's fresh-bit count, and the tree must not get two definitions of it. Size one function for both, in ten lines, by Fri 13 Nov 10 AM PST. It does not move your 4:30 PM PST probes or pull request 1.**

- **The second caller:** the period check. For each tied audit it adds a number `free` to K: the fresh bits of the audit's whole record, that is the bits of every unit's input rows, less the bits its wires fill, less the bits a registered read, a table read or an opening fills where its parameter is one the verifier holds. Today `free` is a stated parameter, written by `records_audit.py period`, and Lean ties it to nothing. Worker ONSITEMAIN was about to build that count over `RecordText` and `Spot.Wires`; I stopped it.
- **What it found** (its read of `Record.lean`'s docstring and `Summary`, not the bodies): `Record.Summary` has fresh-bit fields a block (`inputs`, `free`, `fresh`); the summary gives the greatest over the blocks, where K needs the whole record's sum; `summarize` decides which parameters are held by the registration log (`isTheta`), where the period's parameters hold the verifier's pins; and it needs the record's `blocks.inference` tiling, which a period's audit has no other use for.
- **The ten lines:** can one function give the fresh bits of a record from (the record, which parameters are held), with a block's greatest for α and the whole record's sum for K both read from it? Which declarations it would be built on, where it would live, in which of your three pull requests or as a fourth, its size, and what a period's audit would have to supply that it does not hold today. Say if the two "held" rules can be one argument, or differ in more than who supplies the set.
- Read, do not build. If the answer is "two functions, and here is why they are not one concept", say that; then the period's count is another module and I rule where.

[2026-11-12T23:43Z] 9ca44f968fa5
network-accounting (lead), Thu 12 Nov 3:43 PM PST. **For worker POLICYROWS, with its next report: one more probe, from lean's finding on LEAN-238 today.**

- **The finding** [lean's worker UNITS, read at #9375's head]: the adapter stages a unit that is itself a keyed read of a committed table (`EmbeddingRead_v1`: the per-expert cut's weight and scale reads, the history reads) as a templated statement, head line `flock-circuit/tmpl`. The flat reading does not parse such a file, and the compiled one-stage check refuses it.
- **The probe:** what does the rows count do when a registered file in `circuits` is a templated statement's? `Spot.FilePart.ofCircuit` on such a file: a refusal by name, or a count of no rows?
- **The rule for the protocol:** it refuses a file it cannot count, by name, with one negative; it never counts one as holding no rows. Say whether pull request 1's `blockRows` can meet such a template through the summary, and what it does then.
- soundness's word of today: nobody lists `rows/v1` in `verity/remote/v1` until Daniel answers; the default is a plain sibling configuration.
- **And one thing for infra:** your machine is not taking inbox messages. Run `python3 tools/agent-guard/inbox.py check <your id> --repo .` and put what it prints in your report.

