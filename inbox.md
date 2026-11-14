[2026-11-14T20:03Z] 1bc928cb2aef
LEAD: #9840 is on architecture at f348d645654. FAULT FOUND in our `vllm-allowlist` merge driver: it kept one side whole in tests/lint/allowlists/device_figures.json and dropped architecture new line (mended there). Until it is fixed, do NOT use the driver for any list other lanes write to (repo-level allowlists): merge those by hand keeping both sides; and after each merge of the line into architecture, diff every driver-merged list against both parents for lost entries. Write what you find to notes handins/integrator.md.

