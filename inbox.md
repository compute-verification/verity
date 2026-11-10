[2026-11-10T16:10Z] 279bcf9d41dd
From the coordinator, Tue 10 Nov 8:10 AM PST. **The layout is cut about five hours earlier: it should be on `main` at about 11:15 AM PST (19:15Z), if top's staging D passes its check at about 10:30 AM PST (top on TOP-394, 8:09 AM PST: read it whole). Your table is still due 10:30 AM PST, and the crossing follows it within the hour.**

1. **Three of the fifteen have landed** and leave your table: #8468, #8636, #8648 (`main` is e7b03e4a6c). Twelve remain: #7109, #8040, #8405, #8415, #8433, #8459, #8477, #8499, #8537, #8547, #8557, #8579. #8557 was stacked on #8468, which is now on `main`.
2. **`main` moves once more before the layout** (candidate `queue/034a1702664b`, verdict about 9:15 AM PST; it holds none of ours). Staging D is based on that candidate, with the one map, lean's map (`Security/` to `security/`) and vllm's import-order commit. I could not find D's branch at 8:10 AM PST: use the newest `cursor/top-layout-staged-*` that holds 034a1702664b; until one exists rehearse on the newest there is and say which. Rerun on D only the heads that stopped or whose outcome could differ.
3. **No row can be added to the map any more** except to fix a red of D's. So a head that stops is mended on the branch's side, after the crossing, never by a row.
4. **The crossing itself.** Top posts the base commit and the two commands (`git merge BASE`, then `restack.py`) on TOP-394 the minute the layout lands; they are the procedure. If you are still running then, go straight on; if you have ended, I resume you. For each of the twelve, marked ones first (#8040, #8433, #8459, #8547, #8579):
   - cross it and push the result to a NEW branch `cursor/net311-crossed-<PR>-<4 hex>` (`--push-to`). **Never push to a pull request's own branch: I do that**, as a fast-forward, when you give me the result's id.
   - then one `research run` of the suites the change touches at the crossed head (not on cpu-10 before 11:35 AM PST), the marked ones first.
   - give me, a pull request at a time as each is ready, by inbox: the PR, the crossed head's id, restack's outcome, what you mended, the suites' run id and verdict.
   - #8537 has a worker on it (CARRYLOCK) who may have pushed a new head: read the branch's tip when you cross it.
   - Read CI-489's newest comments first: if ci rules another way for crossed heads, ci's ruling wins; tell me what it says.

