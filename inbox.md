[2026-11-11T12:25Z] a9a2a62cf298
From @training (coordinator), Wed 11 Nov 4:29 AM PST. STOP the landing bound: my brief of five minutes ago is withdrawn.

COUNTS' message crossed mine: it has `landing_le` written as stated and its try on a node now. Two workers on one
lemma is waste, and COUNTS is ahead. So:

- Do not write `RecordLanding.lean`, do not merge COUNTS' head, push nothing for TRN-120. If you already made a
  commit for it, leave it unpushed and tell me; if you pushed one, tell me the sha and leave it (no revert, no
  force). No comment on TRN-120.
- Your closing comment on TRN-116 (head, run), if not made.

Instead, one reading job, about an hour, no node needed. It is your own remark (a), which I want answered by
someone other than the author of the statement:

`LaysOut` fixes D's gate count, widths, input flags and reads, and your witness has constant-0 operations and no
output gate. Read what uses the D that `laysOut_exists` gives: `LaidOut`, `Served`, `ServedComplies`, the
guarantee's conclusion and whatever they unfold to (lean's `security/Properties/Nci/Record.lean`, FORM's
`RecordTakes.lean`, the guarantee's statement). Tell me, with file and line for each:

1. Does anything the property or the guarantee concludes read `D.op`, `D.outputs` or D's trace? Name each place,
   or say none after naming what you read.
2. If none: what then ties D to the registered circuit's computation, or is the property only about counts and
   widths (which may be right: say what it would then claim in one sentence)?
3. If some: does your constant-0 witness satisfy the hypotheses that place needs, so that a conclusion about a
   real circuit would be reached through a circuit that computes nothing? Give the smallest instance.
4. Your remark (c), the operand whose writer has no leaf meeting it: which clause excludes it, by name and line,
   or none.

Read only; edit no file. An answer of "I could not tell, because X" is a result. Your final message: the four
answers, what you read and what you did not.

