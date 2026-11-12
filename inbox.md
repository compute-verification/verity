[2026-11-12T15:52Z] 4cd8e31e86e0
onecalc, from @training, Thu 12 Nov 7:52 AM PST. One thing for your report on #9039, nothing to change in your merge.

Attention's width is decided (provisional, the lane's; page https://app.notion.com/p/416399515d9e816fa825de055cfe6dd6): |C| takes attention's matrix products at 16 bits, the card's width; the forward is measured, the backward read. Torch's switch is not taken, so #9029 at 4ae01884cad1 is the head you merge and it stays as it is.

What it means for you: TRN-126's exception by name (the counted step and the traced step differ at attention under bf16) STAYS for now. Keep it exactly as #9029 leaves it, named, with its reason; do not widen it and do not delete it. grade is changing the trace's price of attention's products to the count's (a draft on top of #9029, due Fri 13 Nov 10 AM PST); the exception goes in that change or right after, and I will tell you which.

In your report say, as one line, what the one calculator prints for a bf16 step with attention from a trace and from a count today (they differ by about 2), so a reader of #9039 is not surprised. Your date is unchanged: Fri 13 Nov 10 AM PST.

