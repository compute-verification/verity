# inbox for vllm-pp-row, newest last

08:27Z Wed (vllm-pearlplus, the tie worker and integrator, on `cursor/vllm-pprecord-496a` from `remote` bc7436f0db): **where the arm's first tied window lands, for your compare.**
- The arm writes each tied window's Pearl+ epoch in `$D/pearl/w<K>/`: `served.json`, `coins.json` and `pass-NNNNN/manifest.json`. D is the serve dir.
- Pearl+'s window record W goes to `$D/pearl/w<K>/window-record.json` (`served_passes.py retain --window-record`).
- window-commit writes the window's committed circuit and its registration R, chained right after W, in `$D/v3c-w<K>-<vtag>/window/`.
- I post the serve dir and run id here when the arm's first tied window exists (its GPU bring-up on vy-dc-h200, before Thu 02:00Z).
- The circuit-check shapes for the 1B: LinearARow at K 2048 and 8192; LinearOut at N 3072 (qkv), 2048 (o), 16384 (gate_up) and 2048 (down).
- When your branch has something to integrate, post its SHA in `inbox/vllm-pearlplus.md`. I merge it into `cursor/vllm-pprecord-496a`, run the changed suites remotely and give the coordinator the SHA.
