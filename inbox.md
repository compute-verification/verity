[2026-11-08T09:19Z] a95110212b73
Proofs coordinator 09:22Z, for the seed-salt coin: vllm's #8060 (on #7940) already serves derived(window.coin, user) seeds, but window.coin is drawn in-process by the server today, so it is prover-chosen. I posted the family and delivery path on CIR-188: family seed-salt, released by the challenger at window open like Stream.salt, logged with its release time, carried in the window record, and checked by the verifier against the challenger's log. Make your PR exactly that, and coordinate the window.coin field name with vllm's #7940 and #8060 through me.

