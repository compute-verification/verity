[2026-11-13T01:00Z] da6a4d7add55
LOG, Thu 12 Nov, 5:03 PM PST

**#9039 at `ca6c4c2a5483` is done on my side: all 44 pairs at that head trace with 0 refused and 0 priced apart, the in-place rule moves none of them, Shampoo's p is unchanged, and the body now carries SF4, the SF3 line and the three limits.**

- RAN (11 runs at `ca6c4c2a54`, named in the handoff note): each of the 44 entries matches its pair at `5c793f5ee59f` in its log rows (op, shapes, count, dtype), its trace's rows and circuit, and its |C| named. Each record states `{'cuda_matmul_fp32_precision': 'none'}`.
  - The runs ending `pairs rc 0`: `r20261112-215357-bd48`, `-215438-c32a`, `-215452-7c5a`, `-215507-4885`, `-215534-fe4c`, `-215312-0450`, `-215241-9af3` (xl), `r20261112-231825-0d63` and `r20261113-000716-8c93`.
  - `r20261112-215426-30ef` and `-215327-c868` hit their caps (rc 143) after 2 entries each. I relaunched their other entries as `0d63` and `8c93`.
- RAN: Shampoo's p against `s-t20-1.1x-ref-optim`, at three counted steps, matches what I sent at 2:14:
  - `s-t20-1.1x-shampoo`: 1.06799 → 1.19277.
  - `s-t20-1.1x-shampoo-whole`: 1.21798 → 1.34547.
- RAN: named over assumed is 1.0022 to 1.0068 across the 42, so no p among them moves by 0.5% or more. Against `s-t20-1.1x-ref-optim`, s-t20's other 26 move by ×0.9991 to ×1.0026.
- READ (`gh pr view 9039`): the head is `ca6c4c2a5483` and the body is set (10542 bytes). SF5's dated line is at the top of `20261112T1904Z-progress-trn126-priced-apart-refusal`.
- The handoff note is `20261113T0059Z-handoff-LOG` (synced, 1de46f93dc).
- Not marked and not merged by me. No run in flight.

