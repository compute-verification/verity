[2026-11-12T15:11Z] 57fb5b07e0e6
grade to training (coordinator), TRN-117 job 2, Thu 12 Nov 7:11 AM PST. Done: evidence stored, one comment on TRN-117 (86 words, "TRN-117 commented as training/grade"). No stored record changed; nothing published. Every figure is RAN unless marked.

**Evidence: art:2f75849cc13135363f3373bfe9ccbd026998695e16ac65df8cd6b7952e7166a7** (PRESERVED; 118 files: README.md, job2.md, job2.json, both runs' out/, every row before and after, the builders; refs to the staged put, its build and Part 1).

## Runs (vy-nebius-cpu-7; each read)
- r20261112-135544-758f: rc 0, peak 65.9 GB of 72 declared. Last line "== 14:55:32Z done rc 0; peak memory of this job's cgroup: 65934872576". Layer 379 of 379 and state 379 restated, as the put's. Algorithm-policy (search, overlay, page), frontier-fit, size-scaling as control and as-if, and size-scaling a third time with standard training's traced steps' attention at float32.
- r20261112-144228-005a: rc 0, peak 10.8 GB of 16. Last line "== 14:55:14Z done; memory peak 14:54:55Z 10812 MiB". Standard-workloads by the put branch's own builder at d76ae46e6f with the put's plan: control, and each of its 16 records rewritten before its ingest (16 rewritten, 0 refused).
- Two launches before the first built nothing: r20261112-131231-6122 (my fault: thirteen inputs sent as links), r20261112-133605-2c45 (its own guard at 56 GB; the preamble holds 13 GB more than Part 1's).

## The controls are the staged tables
0 rows differ in each: search 1,497, overlay 458, page 197 (also equal to the staged body's rows), frontier-fit 46, size-scaling 90, standard-workloads 18.

## Factors, as-if over control: least / median / largest (rows counted)

| table | \|C\| | H | p | rho_default | rho_served | rho_own | S |
|---|---|---|---|---|---|---|---|
| page, all | 1.526 / 2.066 / 2.553 (194) | no column | 0.782 / 1.000 / 1.311 (194) | 0.399 / 0.484 / 0.628 (163) | 0.399 / 0.484 / 0.628 (163) | 0.399 / 0.484 / 0.628 (163) | 0.432 / 0.452 / 0.667 (163) |
| page, v3 | 1.726 / 2.079 / 2.507 (168) | no column | 0.782 / 0.992 / 1.132 (168) | 0.399 / 0.475 / 0.579 (137) | 0.399 / 0.475 / 0.580 (137) | 0.399 / 0.475 / 0.579 (137) | 0.432 / 0.452 / 0.552 (137) |
| page, v2.3-pilot | 1.526 / 1.937 / 2.553 (26) | no column | 0.890 / 1.037 / 1.311 (26) | 0.489 / 0.526 / 0.628 (26) | 0.489 / 0.526 / 0.628 (26) | 0.489 / 0.526 / 0.628 (26) | 0.513 / 0.549 / 0.667 (26) |
| overlay, all | 1.192 / 1.992 / 4.403 (452) | 1.567 / 1.947 / 2.316 (452) | 0.761 / 1.002 / 2.348 (452) | 0.399 / 0.500 / 0.628 (288) | same | same | 0.432 / 0.514 / 0.684 (288) |
| overlay, v3 | 1.726 / 2.079 / 2.507 (168) | 1.812 / 2.214 / 2.316 (168) | 0.782 / 0.992 / 1.133 (168) | 0.399 / 0.475 / 0.579 (137) | same | same | 0.432 / 0.452 / 0.552 (137) |
| search, all | 1.192 / 2.022 / 4.403 (1,477) | 1.567 / 1.947 / 2.316 (1,138) | 0.758 / 1.013 / 2.348 (1,138) | 0.399 / 0.512 / 1.000 (1,290) | 0.399 / 0.514 / 1.000 (1,290) | 0.399 / 0.512 / 1.000 (1,290) | 0.432 / 0.522 / 1.380 (956) |
| search, v3 | 1.670 / 2.142 / 2.507 (708) | 1.812 / 2.214 / 2.316 (636) | 0.758 / 1.010 / 1.384 (636) | 0.399 / 0.484 / 1.000 (666) | same | same | 0.432 / 0.492 / 1.380 (594) |
| size-scaling, 12 traced attack rows | 1.811 / 1.998 / 2.316 | no column | 0.961 / 1.000 / 1.022 | 0.432 / 0.501 / 0.552 (its one rho) | | | 0.432 / 0.494 / 0.552 |
| size-scaling, 45 standard-training rows | 1 | no column | 1 | 1 | | | 1 |
| the same, their traced steps at float32 too | 1.605 / 1.915 / 2.105 | no column | 1 | 0.475 / 0.522 / 0.623 | | | 0.475 / 0.522 / 0.623 |
| standard-workloads, 16 measured rows | 1.936 / 2.291 / 4.786 (records, by their op logs) | no column | no column | 0.209 / 0.437 / 0.516 | same | same | no column |

- The search table's factors leave out 15 rows of five v3 runs the rewrite refuses: their |C| stays while H moves, so their p falls by H's factor alone. None is on the overlay or the page.
- Frontier-fit: the v3 fit, its four points and six starts do not move. The v2.3-pilot fit: E x1.027, A x9.32, gamma x1.145; its four starts and ten of its thirty points move by 1.870 to 1.884 on C; the other twenty points are other checkpoints of the same runs.
- Standard-workloads: the 14 training rows 0.403 to 0.516; inference's two 0.209 and 0.235 (their other products are FP4-coded, so attention is most of their |C|); the two modelled rows (45M) read no record and do not move. Each row's rho factor is the inverse of its record's |C| factor.

## p, and its denominator
p = C / H (READ: te_core.algorithm_policy). C is the run's |C| through its first checkpoint reaching the target, its total where it reaches none. H is the least |C| of any scored or untraced checkpoint from the start on the same data reaching the target, every checkpoint of every run pooled. **H moves in the as-if, in every panel**: 1.812 to 2.316 on v3's eight panels, 1.567 to 1.947 on v2.3-pilot's seven. So p's median factor is 1.00, and a row's p moves by its own attention share against that of H's row. In three of the page's fifteen panels the p = 1 row passes to another row: 81M at 13.9 tok/p, gain 2, v3 (hybrid/grow-depth d64c@7326 to hybrid/grow-wide 9325@5984); 95M at 4.76 tok/p, gain 1.1, pilot (shallow/topk 513a@364 to ref-data/top2muon e4f3@357); 45M at 3.42 tok/p, gain 2, pilot (fullmuon-ema 8f70@1854 to another seed, 8066@1896).

## Rows whose checkpoint or binding term changes
- v3: no checkpoint and no reach changes on any table. Binding term: three search rows of one run (36M at 10.8 tok/p, seed-mezo/mezo-norm-subspace r20261103-052351-8d97, gains 1.1, 2 and 11) pass from none to fit; none on the overlay or the page.
- v2.3-pilot: 70 search rows and 9 page rows reach the target at another checkpoint; 2 search rows change reach (hybrid/mixed ebf7 yes to no; lowrank-restart/dgrad e8d3 no to yes); the page loses 3 rows and gains 4. No binding term changes. Cause: the pilot's starts and fit points are rewritten, so its gain column moves (0.930 to 1.626 on the search table).
- size-scaling: no route, run, binding kind or basis changes. Standard-workloads: every measured row stays fresh_products.

## Panels whose first row, least-S row or least-rho row changes (the page, 15 panels)
- 81M at 13.9 tok/p, gain 2, v3: the first row passes from hybrid/grow-depth r20261104-060904-d64c to hybrid/grow-wide r20261104-000727-9325. Least S and least rho stay.
- 95M at 4.76 tok/p, gain 1.1, pilot: least S and least rho (all three) pass from wildcard/onelayer r20261103-003910-0066 to lowrank-restart/dgrad r20261103-015141-e8d3, a row that enters the page.
- 45M at 3.42 tok/p, gain 1.1, pilot: least S passes from lowrank-restart/dgrad r20261102-223833-bf6c, which leaves the page, to lowrank/dora4 r20261102-215623-111e.
- No other panel; no v3 panel's least-S or least-rho row changes.
- Search table, 30 panels: the same first-row changes (81M v3 gain 2; 95M pilot gain 1.1; 45M pilot gain 2); no least-S or least-rho_own row changes; least rho_served changes in three pilot panels between two rows tied at 1.
- size-scaling: with the records alone at float32 the lesser S passes from standard training to the attack row at s-t20, gain 11 (console and grown): an artefact of pricing one side. With standard training's traced steps at float32 too, nowhere.

## Limits
- v3's starts are not rewritten (the scorer's state holds no op rows of theirs): v3's fit, start compute and gain column stand as published, so no v3 row can change its checkpoint here. INFERRED: with them re-priced v3 would move as the pilot does.
- The rule for a record whose trace holds no backward product of the shape: every call asks for all three (a structure of 106 of 809 runs), which over-counts |C| where a step asked for fewer.
- 25 runs left as they are: 20 with no record in the scorer's state (no row on any table), 5 refused.
- standard-workloads needed one rule Part 1's rewrite lacks: a forward call whose q is of another length than its k and v (decoding against cached keys and values), rewritten by the same products and scalings (sw_asif.cached). It moves those records' |C| by 1.0008 to 1.0058 before the self-attention rewrite.
- The as-if prices what the counter counted; nothing is retraced.

## #9029
Head 4ae01884cad14cfb664f685dea4394dcd2630e8f (reported 6:36 AM PST), draft, unmarked. Nothing else of mine is running.

