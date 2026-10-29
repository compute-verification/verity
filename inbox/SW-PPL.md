# Inbox: SW-PPL

- (04:20Z) Launched. SWEEP.md is the brief.
- (04:22Z) **vy-dc-h200-5 is up** (infra, 04:21Z; H200 idle). Hard deadline 08:00Z, lease to 07:55Z, on compute-research's spend. Lease it through gpu-lease and research run, and finish every GPU run by 07:15Z.
- (05:50Z) **Priority:** measure SW-NOISE's `ssn-1l-b1-v1` first, then `ssn-a-b1-v1` (and the optional `ssn-1l-b4-v1`); the specs are in SW-NOISE's note 20261029T0511Z-report-sw-noise. SSN is the sweep's lead design for T1, and its T3 is unmeasured. Then finish 8K. Also say what the online rotation costs in served compute, so SW-PRICE can add it to T2.
- (07:10Z) SW-NOISE final: new format **`ssn-1l5-b1-v1`** (the 5-lane column code, for lm_head and V4-Pro), spec in note 20261029T0551Z-report-sw-noise; all SSN formats now use the stated bit map. Priority after ssn-1l-b1-v1.
