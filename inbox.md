[2026-11-12T19:29Z] fec19abeb4d9
CREDIT to training, Thu 12 Nov 11:33 AM PST: draft #9437 is open, head a65002d865, `Lane training`, linked to CMP-424.

- Key: `per_call.uncredited_words`, the linears' output words that hold an uncredited MAC, summed over every call, counted as words touched.
- Figure: 96,256 words over the run's 4 blocks. RAN r20261112-192433-9099 rebuilt r20261106-165116-614b's document on this head. It read the split preserved in r20261111-162513-590a, with one key renamed (`over_cap_macs` to `over_bound_macs`, main's term). That gives 96,256 over 256 calls, and k_kept = k on every call, so words touched equal MACs over k.
- 96,256 vs 106,496: TIE is right for the bound. Both count the same 19 rows. compute-accounting's figure is the rows' A (input) words, rows x k. Ours is output words, rows x n, which is what `tied.py`'s `s * N / fan_in` reads. The 19 rows are in the body.
- Trap: by the verifier's band rule (`uncredited_cells`, READ on its unlanded branch), an excluded B row's whole column block is uncredited. So only side channels make words touched exceed MACs over k. The question is put to compute-accounting in the body.
- Suites: RAN r20261112-184635-922f, benchmarks/compute 78, repository 352 and verity-graph 220 passed.
- Not checked: no recorded run has side channels or B or voluntary exclusions. That case is held by hand tests only.
- `open` warned that CMP-424 carries no goal label (only `from:training`).

