[2026-11-10T17:08Z] 01f5d193dce0
From the coordinator, Tue 10 Nov 9:08 AM PST. **One more premise for your statement's list, from proofs on PRF-456 (9:05 AM PST): the patterns' form reads `RequestsCover`, and it is an equality.** No date of yours moves.

- `TwoStageCheckPatternsSound` takes `RequestsCover`: `servedBits req r = D.S` for every request reading in a replay block (`Security/Properties/Spot/TwoStageCheck.lean:64`, 82; `servedBits` counts the program's outputs that are r's, `Claims/SamplingServed.lean:27`; read by me on `main` e7b03e4a6c). The capped form's `ServedCover` is "at most N × S together".
- The rows of record are in the patterns' form, and the design's classes are by (P, O), so requests of different O serve different widths. I asked proofs which it takes: one padded width, S a class, or "at most S" (`/tmp/top/early/prf456-requestscover.md`, on PRF-456).
- **In your next report, one line:** what the rows-laid-by-class design gives as each request's served outputs (the class's `b.egress` row, 32 × (O + 2) bits?), and whether the period check's `pool.S` test (S at least 32 × the schedule's `max_tokens_cap`) still says what it should if S is a class's.
- Proofs runs the planner on the rows of record at S = 32,832 today, and PRF-509's check (Fri 13 Nov) covers `RequestsCover` and `ServedCover`.

