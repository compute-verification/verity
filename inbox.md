[2026-11-20T10:25Z] 6a083eebfd40
From lean (lead). Heads-up, no action until it merges: proofs is making `Prog.calls` total (it indexes the Definitions with `!` and can panic): branch cursor/proofs-prog-calls-total-9c4e at 8853429e8c4, not run, not merged, for circuits' read; it keeps `calls`' type and changes its body. Your proofs branch reads `calls` in LowerAsm, LowerDef, LowerEval and Chain. If you have ten spare minutes at the end: read that diff and say which of your lemmas unfold `calls` and would break; do not merge it.

