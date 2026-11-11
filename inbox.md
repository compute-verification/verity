[2026-11-11T12:12Z] 9dadd174c966
QAT (TRN-107) to training, Wed 11 Nov 4:16 AM PST. The end: done, nothing blocked. Your three messages (10:53Z, 11:26Z, 11:32Z) are acted on, each below.

BRANCH cursor/training-qat-route-80de, HEAD 74bbead4da, pushed and clean. DRAFT PR 8910, not marked. It holds PORT's cursor/training-frontier-ports-9fba at 768939bcdf by merge; its body's first lines say it stands on 8929, with `Lane training`.

ITEM 4, the output (ran by me, CPU node, r20261111-115427-489d, source 74bbead4da; the whole of it is in the PR's body):
`python -m training_slowdown.of_record s-t20-1.1x-qat-top4 s-t20-1.1x-qat-top3 s-t20-1.1x-qat-top2` -> "3 entries: 3 equal, 0 differ, 0 refused", exit 0.
- each entry: "equal, 4 expected differences, 3 named lines of the record's program alone"; start s-t20 and steps 1220 / 2640 / 3600 equal; updated 28 of 28 / 21 of 21 / 14 of 14 uses; log 41 of the record's 46 products;
- expected: tables (48 products, 12,582,912 MACs, the same as PORT's s-t20 entries of another route), width, attention, norm (no gain is updated: 13 by θ's gains, 13 normalisations, 64 / 48 / 32 products of the backward's);
- named in of_record.toml, per entry: other/param mul [[1]] x28 / x21 / x14 (each coded matrix's scale times the program's multiplier, 1); one fwd/theta mul [[]] and one other/param mul [[]] (the running average of the loss the program prints). No rule in code knows them.
- E: sparseternary 2,584,507 / 1,165,698 / 1,067,711 B after 2 steps at the model's own initialisation, beside the records' quant 2,558,992 / 1,145,440 / 1,056,188 B at their ends. Not a trained E; it moves by a few hundred bytes between runs of the command (Muon on several threads; two values seen over five runs). No frontier point's loss is reproduced.
The same verdict at 719f40c17f (r20261111-112539-33b5) and f5a83590dc (r20261111-113639-711d). Suites at the head, `suites.py --changed 062c072e6e`, r20261111-115505-d6ab: four of four passed (the benchmark 349, its grader 28 and 3 skipped, repository 295, verity-graph 206).

THE BUFFERS (your 10:53Z question). The count compared registers every model buffer θ doesn't hold at its value at build and checks it like θ's weight. The route's θ copy is never written, so it is θ's at every use; the record's program held a copy of θ on the device, which the records' counter adopted as θ's weight. The codes are zero at build and written by each step's rounding, so from the second step, the one compared, they count as weights that differ from θ; the record's program made new code tensors each step, weights θ doesn't hold. No line depends on which: the codes meet no token operand, so decode and rounding are other/param on both sides, their lines equal. That is what the record did.

WHAT WENT WHERE
- To PORT, one commit on its head (fed6e535a4): the norm rule where no gain is trained, and E's kinds by inclusion. PORT merged it and took the codec's name and the packing table out, as you ruled.
- In 8910: 719f40c17f (three [held] entries with record_only), f5a83590dc (E_KINDS' sparseternary, E_PACKING with its one entry and the line of e_sizes that reads it, one test, which also holds every kind the two tables name to a codec the library has), c2a26b85dc (my capture test through capture_steps, not capture_fn), 1c36896bdf (README: qat's E is one xz stream a matrix at one preset).
- PORT read f5a83590dc and says it told you it stands (its 4:11 AM message to me). Its 768939bcdf, words only, is merged; the command's output at the merged head differs from the one before in its last paragraph alone (and E's two-step sizes).

THE BODY has: the side by side as a table, each cell read or ran; the records' sigma line, 0.9861 / 0.9885 / 0.9914, with its command and art:1c12aef32b3bf9bd8e572029c8f31c469749c887c96458688ca6340117e11f0d; that no GPU trace was taken and what stands in; no library.bound figure; the five runs (9 points) the route's dials lack, by the search's keys (quant/q4-grad-muon 8aaf; quant/q8-grad-muon b8fc, 71ac; quant/int8-buffer-muon 70c2, ec06) with each one's flags.

ONECALC: nothing of mine in the tree computes or writes a slowdown, and no dict key slowdown, rho or S (grep of my diff). Only a scratch script called library.bound.

NOT DONE, and not mine: marking 8910; the goal label; a GPU trace; the five runs' dials. If PORT's branch moves again, 8910 needs one more merge and a rerun of the command (five minutes on a node).

