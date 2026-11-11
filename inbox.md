[2026-11-11T08:32Z] 22c9e4212369
CROSS, 12:32 AM PST (08:32Z). CROSS, 12:35 AM PST. #8606 NOT MARKED YET: one question first. #8753 is marked.

#8753: head 3e436cc09acc734d0d7f9fd5cd124894a5f3bab7 (one commit on 5fc53c6ca4b1, pushed before the run). Run r20261111-074912-8545 at it, rc 0, its source.tree the head's (166f085e3856): "4 suites: 4 passed (0 cached), 0 failed" (benchmarks/training_slowdown 185 passed); tests/test_gates.py -v: 6 passed, test_each_count_is_the_definitions PASSED (it RAN, not skipped); python -m training_slowdown.gates --check: "every Definition's counts match", exit 0. Body fixed (a section for the crossing; the old Checks line now says it is the author's run before the move). Mark, 11:59 PM PST, rc 0: "preflight: #8753 at 3e436cc09acc merged onto `main` at 8201dacff26e (`next` at 15ab284aa5d5 doesn't hold it): nothing refused (52 s; collected 4 suite(s))" / "#8753 is ready at 3e436cc09acc".

#8606: head 1433dd58511fbec77d96e1037eec2f2545c9be3b (33fbfb729375 the grep line, then 1433dd58511f the sentence in your words; two commits because the line was pushed before your wording came; nothing amended). Run r20261111-075240-ec16 at it, rc 0, its source.tree the head's (43f9ebe453fa): "9 suites: 9 passed (1 cached), 0 failed" (verity-vllm 13953, research 4022, pouw/verifier 1844, verity-flock 1425, repository 280, verity-graph 181, verity-bench 164, verity-network-e2e 66; verity/commitments cached).

THE QUESTION. Reading the PR's other added lines before marking, read by me at 1433dd58511f: the twin of rho's sentence is still in the PR, verity/protocols/inference_only/nci/README.md:55:
  "Nothing outside `benchmarks/` imports this directory (`tests/test_boundaries.py`). Three things outside it read it:"
and its first bullet, line 57: "`verity/protocols/inference_only/rho` reads data files of this directory by path ..., and its tests load `tiers.py` and `slowdown.py` by path". The directory is under verity/ now and so is the loader, so the first sentence is the wrong statement (the body's own bullet "At the layout's move" says both sentences become wrong and "the PR that fixes the moved tree restates both"). SEPARATE in tests/test_boundaries.py lists protocols/inference_only/nci beside rho.
Proposed, one commit, nothing else: line 55 becomes
  "This directory has an import root of its own (`SEPARATE` in `tests/test_boundaries.py`). Three things outside it read it:"
and line 57's bullet rewrapped inside 120 (the move's longer path made it 140).
MEND IT (then a new run at the new head, about 37 min, then the mark), or MARK 1433dd58511f AS IT IS? I hold the mark until you say. The file is @compute-accounting's by the registry; the section is this PR's own commit.

Smaller, not truth: the move's longer paths left three more added lines over 120 (rho/README.md two: 138 and 128; benchmarks/circuit_properties/README.md one: 138). The repository suite passes with them. Say if you want them rewrapped in the same commit.

Also seen, not mine: the queue shows #8490 at 8848940a322f (ci crossed) waiting, not marked.

