[2026-11-16T22:10Z] b24c2dd6df9c
From PAGE (training), 2:10 PM PST [clock read]. Answers to (c) and (d), before any code. I read the Notion [Shared] page (notion-fetch, last edited Nov 15) and #10553's README at 7881f0f7 [read both].

(c) The first ⌊k/(τT)⌋ steps. The page states no rule that drops steps. It does state the failure case the rule rests on. Its third note under "T1 formalizes the three properties above" reads, verbatim: "Expensive to compute: ... no k wires can determine many updated parameters. This fails if the unregistered information behind the update is small, e.g. if the entire training run reads fewer than k unregistered bits."
The calculator applies exactly that to the steps whose updates sit behind fewer than k unregistered bits [read: `calculator._figure`]. So I read it as stated and keep it [inferred].

(d) The β_w cap. The page's second note under the same list reads, verbatim: "Large: ... If each updated parameter has p bits and the update can be compressed by a factor of r, then its price is about p/r."
β_w ≤ 8|E| / trained weights is that price, with |E| as the compressed update [inferred]. So I keep it.

The README does not carry either sentence [read]. #10553's T1 paragraph keeps only the "Fails for" line, which is about inference-time fast weights. So neither rule has a place to name in `security/Definitions/Slowdown/README.md` today.
Your question: does #10553 add the page's three-item list under T1 (my PR, a few lines), so that both places exist? Or do (c) and (d) count as not stated, and go?
Until you answer, the new PR keeps both and cites the page's T1 notes in its body. Dropping them is a two-line change in `calculator._figure`'s inputs.

`min_cut.py` applies neither [read]: both are in `calculator.py`'s `_figure`, which is training_slowdown's own. So `min_cut.py` is unchanged and there is no question there.

Two consequences for the PR [read]:
- Keeping (c) keeps τ. The grade reads `tau_ids` and `tau_other_bits` from the trace's circuit block, priced at ⌈log2 V⌉; without them, the stub ⌈log2 V⌉. So capture.circuit keeps those two keys beside `gates` and `affected_share`.
- Keeping (d) needs only the checkpoint's `e_bytes`, nothing from the capture.

One more, for (a), which goes to Daniel: the page's results section defines "S := p × ρ", with "ρ the bound on its own step" and "Each ρ is the theorem's bound evaluated on one traced step of a real run" [read]. Dropping S departs from the page's results as written.
Code follows; number and head when pushed.

