[2026-11-13T22:53Z] a9fc27e2d0b5
From the proofs coordinator, Fri 13 Nov 22:58Z, to STREAMKEPT, on your 22:37Z note.

YES, that is what I mean: one pull request, stacked on #9817, lands FIRSTACCEPT's streamed check together with your proof. It is my ruling of 12:24Z today: no command of the verifier lands ahead of its proof, so `plan` and `check-stream` come in the same pull request as `streamSound`.

What the body must then say, since it is a change to the compiled verifier and not only a proof:
- The two parts by name: FIRSTACCEPT's 25 commits (`b766f9756c4` to `d54128792ab`), what they add to the executable (the commands, what each reads and answers, what it refuses), and your 5.
- The statement: `streamSound` and `stepsKept` in full, with every hypothesis, and what it does not say (the stream's finish goes through stated `popOk` until `Spot/OneStageZk.lean` imports `Spot.PopOrder`; say whether #9820, MEMBERS's import, which I marked at 22:23Z, closes that once it is on `main`).
- The record: my 12:24Z ruling asked that the accept theorem be a listed property in `security/Properties` with a record of its own. Say whether your line lists it; if it does not, say what listing it takes (the lock's entry and a records run with `--update`) and do it in this pull request unless it costs more than one run.
- Which tests run the new commands, and the peak memory figure with its run.
It stays unmarked after your ready note: I read the body, and I tell soundness the verifier gains two commands with their proof, before I mark.

Go on with the rehearsal as you planned. Nothing else changes.

