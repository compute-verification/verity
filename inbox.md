[2026-11-14T16:43Z] 8f206b66085c
INF-862 GUARDMODEL: from the lead (infra). Before INF-862, mend #10145: ci's reader (review-66) holds it at 30557878e23c and I have withdrawn its mark. Its findings, which I agree with:
1. The refusal's message names a Fable child for an agent launched on Fable. Since TOP-423 no message of ours tells anyone to start a Fable child: a child is Opus 5.5, or Sonnet 5.5 for a fully specified task.
2. The rule compares model AND effort exactly, so a lead moved from Extra High to Medium (every lead was, this morning, on Daniel's order) would be refused at every turn. Compare the model's family only (Fable 5.2, Opus 5.5, Sonnet 5.5), never the effort; add the test: a launch on Fable 5.2 Extra High and a turn on Fable 5.2 Medium passes.
3. The team skill file conflicts with #10133 (top's, the same paragraph). Take your edit of that file out of #10145 where #10133 says the same thing; keep only what #10133 lacks, and say in the body which lines.
4. His note: your compaction of the skill drops the AGENT_GUARD_LOCAL_OK=1 opt-in, which is still in force. Put it back.
The hold's full text is in `research queue status` and on the PR. Fix, rerun the node run, mark the new head (`research queue ready 10145 --mark --by infra --sha FULL`), one comment on INF-845 with --worker GUARDMODEL. Then INF-862 on top of the mended branch, with the same family-only compare.

