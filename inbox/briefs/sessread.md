# SESSREAD: the two-stage count when a session reads several replay blocks (one Lean attempt that decides whether prefix caching by read is in Friday's audit)

**Why.** Daniel, Wed 11 Nov 5:40 PM PST, wants the audited serve to depart from stock vLLM as little as possible: prefix caching and speculative decoding on. Our worker CACHE priced it (the notes, `lanes/proofs/20261112T0240Z-report-stock-vllm-departures.md`; its entries 5 to 7 in `inbox/cache.md` on `cursor/proofs-cache-7c3e`). Recomputing every cache hit, as `main` does, costs 39 times today's modeled bound on agent traffic. Reading the rows the session committed while serving costs 1.7 to 3.1 times, and needs one thing proved: the two-stage count when a session, taken as the request, reads several replay blocks. vLLM's goal is Fri 13 Nov 9 AM PST (17:00Z). One attempt at the lemma says whether that holds.

**The statement** [CACHE's reading, not compiled]. `AdmissibleTwoRankinPatterns` without its hypothesis that each request reads one replay block (`security/Definitions/Claims/SamplingTwoRankin.lean:79-88`). `RequestsCover` then loses only its first clause (`security/Properties/Spot/TwoStageCheck.lean:58`); its other two stay (no two sessions read one proof block, lines 59-60; every session that reads a replay block serves the point's one cap, lines 61-64). The right side of the inequality does not change. The guarantee's statement, `InfluenceTwoPatterns`, does not change.

**The proof CACHE argues** (its entry 6; checked against the definitions, not compiled). The present proof, `admissibleTwo_rankin_patterns` (`security/Proofs/Flock/SamplingGeneric/TwoRankin.lean:267-372`), uses the hypothesis three times. Write T_r for the proof blocks of a set T that session r reads.
1. Take the tuple of patterns a replay block from the set, ρ ↦ the pattern of T ∩ ρ, not from the pattern.
2. The tuple fixes the pattern: if T_r is heavy in some replay block it is heavy, since `bitsOf` is a sum; if in none, the tuple records every part and T_r is their union (`biUnion_inter_blocksIn`, line 83).
3. The pattern's bits are at most the tuple's: `lightBits |r| (lightOf |r| U)` is `min(b(U), |r|)`; `b(T_r)` is the sum over replay blocks (`bitsOf_eq_sum`, line 96); and `min(Σ a, c) ≤ Σ min(a, c)` (`min_sum_le_sum_min`, line 374; `sum_min_inter_le`, line 386).
4. Rankin's step, the supremum over a pattern's sets and the sum over tuples stand as written.
The clause is used once more, as `hreq.1` in `twoStageCheckPatternsSound_form` (`security/Proofs/Properties/Spot/TwoStageCheck.lean:243-260`).

**What to do, in order.**
1. Read the three files and say whether CACHE's reading of the hypothesis and of its three uses is right. If a use is missed, that is the first thing to report.
2. State the lemma **beside** the present one, in a new file of the same directory (`TwoRankinSessions.lean` or the name the tree's convention gives): the same conclusion with the hypothesis gone. Do not edit `AdmissibleTwoRankinPatterns`, `RequestsCover` or any listed statement in this attempt; nothing of the lock moves.
3. Prove it. Compile the one file with `verity lean try` through `research run` (the lean-proofs skill, `.agents/skills/lean-proofs/SKILL.md`, says how; never build `security/Proofs` on your VM). No `sorry` in what you call proved; a step left open is named as open with the goal's text.
4. Then, only if 3 compiles: the same for `twoStageCheckPatternsSound_form` without `hreq.1`, again beside the present one.
5. A list, by read, of what the real change would touch after this: the definition, the property, each proof that names either, the lock records that would move (name the guarantees that read them), and any test or Python that mirrors `RequestsCover`'s first clause.

**Not yours:** changing a listed definition or guarantee (I take that to Daniel with your result); the count across an execution phase's edge (a separate ask to Daniel); anything of the serve or the adapter; the session cap's value.

**Done when:** an entry that says "it compiles" with the run id and the file's commit, or "it fails at step n" with the goal that does not close and what would close it. Either is a result.

**Deadlines (UTC, Thu 12 Nov):** 07:00Z first entry: step 1's answer and the lemma's statement as you will try it (compiled or not, say which); 11:00Z: where the proof stands, each of the four steps tagged compiled, written or open; 16:00Z: the verdict, with one sentence on whether the whole change (step 5's list) fits Fri 13 Nov 17:00Z given what you now know of its size, in terms of files and proofs, not of days.
