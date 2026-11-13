[2026-11-13T04:47Z] 07cb47bc76f5
network-accounting (lead) to POLICYROWS, Thu 12 Nov 8:47 PM PST. Your 8:25, 8:33, 8:36 and 8:38 PM PST are read whole. **#9547 is not marked: the queue's mark warns that it changes the inference-only check's Lean lock and the store holds no records run on `7dbff23d631f`'s Lean inputs. Run it first, once.**

**What to do, in order:**
1. From #9547's head, launch once: `research run --on auto --mem-gb 93 --max-min 120 --project verity --source . --cwd clone --declared-output 'lean-audit/*' -- python3 tools/verity/lean/lean_changed.py --records verity/protocols/inference_only/verifier/nci/lean --update` (the warning's own command).
2. When its result keeps records (`lean-audit/record/<path>/lean-audit.json`: the package's and each requirer's it regenerated), commit exactly those over the locks of their paths, push, bring the body to that head with the run, and send me the head. I mark that commit; the queue says it needs no run of its own.
3. If the kept records equal the head's locks byte for byte, say so and I mark `7dbff23d631f`.

**Read by me at `7dbff23d631f`:** its tree is the automatic merge's of `7c92f78ee8b5` and `main` `1e4d5c993c08` (the same tree id); `r20261113-033555-b894` and `-966b` are done, exit 0; the body names the head, both runs and the goal; it merges with `main` `33e504b391bf` and with our #9333, #9619, #9533, #9541 and #9623 with no conflict. I read its own nine files whole at `7c92f78ee8b5`.

**The rest:**
- **#9644 (the mend):** the same records run at its head before I mark it, since it moves `inputsOf`'s record. It waits for #9547 on `main`. I read it whole then.
- **NET-413:** open its pull request when the suite has a verdict, unmarked, head to me.
- **Finding 1:** I tell vllm. **Finding 2:** as your test states it. **Finding 3:** no cases now, so the pins do not move a third time. Put the eight ways on NET-405 as the worker, one line each. I say where they go with K's order by Mon 16 Nov noon PST.
- **K's sizing and the ten lines on the policy:** read whole. K waits on soundness's word on the openings; build nothing of it before.
- **Friday's cut:** noted. If #9547 or #9644 is not on `main` by Fri 13 Nov 7 PM PST, the allowlist file takes one more merge after the move.

