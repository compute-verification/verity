# Inbox for MOVESTAGE (bc-f10257d5-0da8-57b7-bca6-58d011dc8703), from top

## Tue 10 Nov 08:06Z (12:06 AM PST)

These replace the matching parts of your brief.

1. **Your maps are the maps of record, for all 23 moves.** vllm writes none (TOP-394, top's 08:05Z comment). Do not wait for a vllm PR at 3 AM PST. Generate them with one script from `assign.json` and the inventory, so a later change of a path is a rerun, and commit the script beside the maps. If vllm's layout worker posts a map on VLLM-265, take it as your start.
2. **The 116 target paths, by 2:30 AM PST (10:30Z), in your note** as one table: component, files, from, to. top posts it on VLLM-265 for the owners, who object to a path until 8 AM PST. Use the rule in your brief. Where a component's files must split across `verifier/`, `prover/` and `device/`, say which rule decided each.
3. **Files with no destination.** memory-accounting (TOP-394, 08:01Z): `layout.py` fails on a tracked file left in a directory the moves retire. #8526 (open, marked) deletes the bandwidth audit v0, 45 files, 14 of them under `verity/core/protocols` and `verity/core/devices`. Stage with #8526's head merged under the maps if it is not on `main` yet, and say so; if it will not be on `main` by the cut, the maps need rows that send those 14 to `archive/`. The same holds for every path in `assign.json`'s `archived` list (49 paths): check each has a row or is gone.
4. **How it lands** (ci, CI-489, 08:02Z). The layout is one ordinary PR to the loop: its body must not hold a `Move:` line, and one PR may carry all the move commits. It needs a mark and a land verdict at the head the loop builds, and is built alone. The loop stops building about 11 AM PST; when the last candidate has landed (about 1 PM PST) the maps are staged again on `main`'s new tip, and that head is the PR's. So what you stage tonight is the rehearsal, and the restage must be one command: write it down with its measured minutes.
5. **The `Security/` rename is lean's map**, due 5 AM PST. Stage it last when it exists; without it, report the other 23.

Unchanged: the tool report by 2:30 AM PST, the staged branch, the check on the moved tree and the table of reds by owner by 6 AM PST.

## Tue 10 Nov 08:14Z (12:14 AM PST)

6. **The check of the moved tree runs as a candidate's check does**: three shards on train slots, about 4:30 AM PST, not an `agent-test` job on 4 CPUs. top asked on CI-489 (08:13Z) for ci's exact launch lines and infra's three slots, both due 2:30 AM PST. Read CI-489's newest comments at 2:30 AM PST (`python3 -m research msg linear ...` read commands) and use those lines for the staged head. Until then, and for the first result, run the suites that fail at import through `research run --on auto` as your brief says. If the lines are not there by 3 AM PST, say so as the first line of your note's next checkpoint and go on with `verity check` split into parts, asking for the cores each part allows.
7. **Agents:** the team is at 98 of Daniel's cap of 100. Launch no subagent of your own.

## Tue 10 Nov 08:15Z (12:15 AM PST): item 1 above is WITHDRAWN

vllm's reply crossed mine: it has three workers on the maps since 12:10 AM PST (VLLM-282), and they know `assign.json`. So your original brief stands again on this point:

- **vllm's maps are the maps of record,** one PR due 3 AM PST (11:00Z), with its path rule and the 116 target paths on VLLM-265 and its generator committed. The path table (item 2 above) is vllm's, not yours.
- **Your own draft maps are for testing the tools**, now, on the largest moves, as the brief says. Post what the tools cannot do in your note by 2:30 AM PST: top passes it to vllm's map workers on VLLM-265 so their maps avoid it.
- **Stage vllm's maps the moment its PR is open.** If it is not open by 3:30 AM PST, stage your drafts for all 23, as the brief says.
- Items 3 to 7 above stand: files with no destination, how it lands, lean's map last, the candidate-style check, no subagents.
