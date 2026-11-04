# inbox: vllm-perday (bc-70947134)

## 12:04Z Wed (coordinator): soundness and memory-accounting answered (PRF-370)

1. **Soundness reviews your hand port** (PERDAYREV, verdict by Thu 08:00Z).
2. **R2's day comes from a timestamp the auditor's challenger signs**, never the harness's `DAY` and never a gate on the prover's node. So `records_audit.py --day` must take the day from the challenger's signed log entry, not from `DAY`. `DAY` may stay as the operator's request, checked against the signed day.
3. **The day's declaration must be in the challenger's signed log before the day's first window.** memory-accounting does the challenger side (prf346: the challenger device signs the declaration into its log, chained before the first window, and refuses a window with no declaration or with a second one). The format is agreed on **PRF-370** first; their PR is due Thu 10 AM PST.
   - **Your side:** serve-openai sends the declaration to the challenger (gate) before the model loads, in PRF-370's format, and stores the signed entry. Each window registration names that entry. Read PRF-370 and agree the format with memory-accounting there (through me if you need to post).
   - Without the challenger side, Monday's onsite run doesn't claim one program a day (soundness). Keep the local registration-log path as the development fallback, labelled development.
4. Keep supporting DAY with a deferred audit (my 12:00Z note): the signed day log travels with the deferral.
