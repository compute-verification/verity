[2026-11-14T16:40Z] a8b21c84c4b6
## Coordinator, Sat 14 Nov 16:40Z: MESSAGES: the count condition

From the proofs lead, Sat 14 Nov. Two words on the fourth refusal (`StmtsRead` by position), both on TOP-415: (1) soundness says yes, on one condition: the form ALSO refuses when the count of sessions differs from the count of statements (a session past the last statement, or a statement with no session, must not pass unread). Build that into the refusal and its test. (2) vllm read its honest play: sessions come in the statements' order with equal counts on every path (`two_stage.py` 840-855; `a5.py` 1078, 1104 on), so no key is needed; but nothing has yet run `second` on real two-stage evidence, so keep a test of the honest play at the small two-stage script.

