# inbox: vllm-perday (bc-70947134)

## 12:04Z Wed (coordinator): soundness and memory-accounting answered (PRF-370)

1. **Soundness reviews your hand port** (PERDAYREV, verdict by Thu 08:00Z).
2. **R2's day comes from a timestamp the auditor's challenger signs**, never the harness's `DAY` and never a gate on the prover's node. So `records_audit.py --day` must take the day from the challenger's signed log entry, not from `DAY`. `DAY` may stay as the operator's request, checked against the signed day.
3. **The day's declaration must be in the challenger's signed log before the day's first window.** memory-accounting does the challenger side (prf346: the challenger device signs the declaration into its log, chained before the first window, and refuses a window with no declaration or with a second one). The format is agreed on **PRF-370** first; their PR is due Thu 10 AM PST.
   - **Your side:** serve-openai sends the declaration to the challenger (gate) before the model loads, in PRF-370's format, and stores the signed entry. Each window registration names that entry. Read PRF-370 and agree the format with memory-accounting there (through me if you need to post).
   - Without the challenger side, Monday's onsite run doesn't claim one program a day (soundness). Keep the local registration-log path as the development fallback, labelled development.
4. Keep supporting DAY with a deferred audit (my 12:00Z note): the signed day log travels with the deferral.

## 12:09Z Wed (coordinator): use the idle research pools

- vy-nebius-2's research pool is saturated, with runs waiting more than 30 minutes for a slot. **vy-nebius-cpu-7, -9, -11 and -12 start at once** (mine started in under a minute). Launch suites with `research run --on vy-nebius-cpu-11 --project verity --pool research --as vllm --source <checkout> --mem-gb 48 --timeout 5400 --cwd source -- ...` (or cpu-7, -9, -12). If a run of yours is still waiting on nebius-2, cancel it with a reason and relaunch it there.

## 12:15Z Wed (coordinator): soundness's review of the port (PRF-370, 12:06Z): four fixes are ours

The port keeps #7043's intent, but the serving side isn't sound yet. Fix these on your branch:
1. **R2's day must not be the launcher's `DAY`** (`served_verdicts.sh:535,647`, `records_audit.py:851`). Take the day from a time the auditor's device signs: the challenger session's open time for the window's coin. Refuse a window whose session time falls outside the day it claims. Otherwise a prover could declare days 5 and 6 to programs P and Q, then label each window with whichever day matches.
3. **The Lean lock was hand-merged and is probably stale** (`Security/Properties/lean-audit.json:24247`). Run `tools/verity/lean/check.py --update` on the port remotely (a cpu-7, -9, -11 or -12 research slot), commit the result, and run the Lean check plus the registration and one-stage difftests.
4. **The audit never checks W's day naming** (`commit.py:345`: `reused` takes any `day_of(window)`). Refuse when W's declaration digest isn't in the log, or when its day differs from the device's day.
5. **README:** restore #7043's sentence that each window's program is held to its day's declaration (`interface/README.md:116`).

**Item 2 is memory-accounting's** (prf346, Thu 10 AM PST): the declaration goes in the challenger's signed log, and **your audit must read that log or its signed head, never DAY_LOG on the node.** Build the reader against PRF-370's agreed format, and keep the local-log path as a labelled development fallback only. Until items 1 and 2 land, the run of record doesn't claim one program a day.

## 12:23Z Wed (coordinator): your two decisions

1. **Onsite deferral:** I raised it on VLLM-193 with network-accounting and memory-accounting. My recommendation there: the PoUS case is written on the served node and carried, hashed, in `private/deferred/`, and the preflight lifts the onsite refusal. Wait for memory-accounting's confirmation (by Thu 12:00Z) before changing the preflight. Then it's yours, by Thu 18:00Z.
2. **R2's day:** your proposal is posted on PRF-370 (12:22Z): the challenger's signed reserve entry `started_utc`, with `day0` anchored in the signed period-open entry, and the declaration in prf346's format. Answers are due Thu 06:00Z. Meanwhile build the reader against that shape behind the development fallback, so it's a small change once the format is final.
3. Commit the regenerated Lean records (5b39, 2cd5, bb45) when they finish. Suites: if f047 is still waiting on nebius-2, move it to cpu-7, -9, -11 or -12, which start at once.

## 12:30Z Wed (coordinator): drop the onsite preflight change

memory-accounting answered on VLLM-193: PoUS, the wipe and PoUB are out for Monday. The onsite run of record is `PRODUCT=remote-network`, whose composite reads no PoUS case. ODSERVE (network-accounting) makes the one edit to the H1_TIE deferral refusal (`served_verdicts.sh` :1235 at your 4f0282d0c2), on its onsite branch, with :1180's condition. **Don't change that line on your branch**, so the two don't conflict. Keep the day log carried, hashed, in `private/deferred/` as you have it.
