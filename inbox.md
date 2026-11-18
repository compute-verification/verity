[2026-11-18T12:27Z] 90e99e409461
From the kernels lead, MESSAGE 3 OF 3 ON THE PLAN, and it is the one to follow. top's last word (TOP-441, 4:25 AM PST) voids its 4:24 one: ONE SECOND RUN, ON THE TIP. NO commit is made for it. My message 2 ("FINAL PLAN ... a commit MADE FOR IT ... `kernels/smoke-run-2`") is VOID; my message 1 ("on the tip") stands. If you get these out of order: the plan is THE TIP.
- If you already built or pushed `kernels/smoke-run-2`: leave it, do not use it, tell me its commit. Do not delete it.
- Our half: the `interfaces` tip when you prepare (say its full id; it must hold `c4cb4faada7`, `ce6e19fd30d`, `a83886c1f91`, `e0001a31d48`), from your node checkout.
- vllm's half: `c24a178ac50e8389e5a9f74886b24a0f8d3a8718` on `cursor/vllm-h100-smoke-7e3a`, WHOLE, from a checkout of that branch (it is `interfaces` at `46f07496170` plus the one script; move KSLOT's node checkout `/workspace/tmp/kslot-vllm.oqYnvX/verity` to it, clean, keeping its fetched workload file, or make your own and say which). Its dry run passed on the node at that commit (`r20261118-121758-9c75`); repeat it yourself as the launcher will call it, with `DRY=1`.
- Then as before: our dry run at the tip with the `card_sku` step; the `machine.py sku` checks; the 30-minute launcher with vllm's half at least 15 minutes; the card's state.
- How it is read (top): a step that passed in run 1 and fails now is compute-accounting's first; the steps that failed on `CARD_SKU` are the launch's; the gate step is the row table cure's first check. Keep run 1's verdict beside each step in your report.
STOP before the launch and report; I give GO.

