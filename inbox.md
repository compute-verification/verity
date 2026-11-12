[2026-11-12T21:11Z] 0242b90766b1
Coordinator to RELAX, Thu 12 Nov 21:12Z. One case from PRF-548 that soundness ruled must be a refusal before #9023 is marked (network-accounting's comment of 12:56 PM PST and soundness's stamped 2:00 PM PST: read both).
The case: a `--registered` file that names a certified port under a value the record does not list, at another root. Today the strict record check refuses it only because the value is missing; `session_registered` drops the read (`a0.py`, the line that keeps only the reads whose value the record lists); under the relaxed check nothing compares that port, so the file's names are held by nothing but the driver.
What I need from you, in one entry by 23:30Z:
1. Whether your R7 branch (`cursor/proofs-r7-certified-b6a0`, the code `R7-certified`, `CertifiedHeld`) already refuses exactly this case, in the Lean check and in the Python: file and line for each, or "no" with the line where it passes.
2. If no: the comparison that closes it (soundness calls it one line: the relaxed check refuses, by name, any `--registered` file naming a certified port under a value the record does not list), in both verifiers, and a test of exactly this case that fails without it. Build it on your branch; it is part of R7's pull request.
3. The refusal stated as a property of the compiled check beside `Properties.Spot.ExpectsV2`: say whether `CertifiedHeld` as you wrote it already says it, or which clause it needs. If it needs a clause, that is one more change to what `acceptMeans` says: write it in the body's before/after in plain words, as for the two you have, and it goes to Daniel with them.
4. Whether #9023 itself (`cursor/proofs-relaxed-check-7a3e` at `6b782d972ec4`) should then be closed in favour of R7's pull request, or lands first with this refusal added. Your recommendation and why.
The pull request stays unmarked, as before. Fri 5 PM PST stands unless you say otherwise in that entry.

