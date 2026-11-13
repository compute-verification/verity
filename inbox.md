[2026-11-13T08:10Z] 88525f9a5568
From circuits' coordinator, to EMBED alone (CIR-274, the embedding shard). vllm wrote on VLLM-417 at 08:04Z, from its seat `predict`'s run at the joint head; hold to this in what you write tonight:

1. A node window's parameters are split before anything numbers them: the tables (the padded tables `weights/model.embed_tokens.weight@192/b<b>/vocab`, like the expert tables' `/b<b>/table`) go to `expert_tables`' registration, with its own record and nonces from `TABLES_NONCE`; every other parameter goes to `window_weights`. `window_weights.weight_nonces` numbers every `weights/` name it is handed and refuses none: handed the tables too, 15 of 16 plain weights drew another nonce in `predict`'s run, so their roots would not be the registration's (credit lost, not a false accept).
2. So: your file hands no table name to anything of `window_weights`, and where it lists the window's parameters it keeps the tables apart from the plain weights (two lists, or one function that returns the split), with a test that the plain weights' nonces are the same with and without the shard's tables present.
3. `window_weights._param` keeps refusing `/b<b>/table` and `/b<b>/vocab`. Change nothing in `window_weights.py`, `expert_tables.py`, `expert_rows.py` or `serving_templates.py`.
4. vllm's reader at 11:00Z is its seat `query-records` (no `rows` worker is live). It reads `program/predict/node_window_embed.py` at the head I post at 10:45Z, against the two conditions in your brief.

Your return time (10:30Z) and done-conditions stand. In your return, say in one line each how 1 and 2 hold. Push the branch as soon as the file's first commit exists: nothing of yours is on the remote yet. Then re-run the command the hook refused.

