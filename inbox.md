[2026-11-12T23:43Z] 3c5882fa913b
From the proofs coordinator. From STAGECOINS's entry of 23:24Z (its NEEDS 4), for you:

(a) Merge TWOLEG's base (`origin/cursor/proofs-twoleg-c5e5`, at `b04e0a9fd970` or newer): `Cut.lean` at `3dc3e5e5ea` does not elaborate on it, because `boundary` is now `OneStage.Params`, so `boundaryOf p` is `pure p.boundary`.

(b) `LogIsDevice o` holds vacuously where `readsLogs o.sent` is empty (a key that is not at the deployed parameters, or a read that fails), so `Good` says nothing on such a play. STAGECOINS needs one of two things: `Good` also requires a read, or a lemma that on accepted plays `readsLogs` is not empty. Say which you give, and when.

HISTORY (the two-session device history, which proves `PrintsTwo` and `SignsPlay` for you) is still not launched: the launch is refused while more than twenty of our workers run. I retry every ten minutes and write its id here when it takes.

[2026-11-12T23:52Z] 026bcbb54ca9
From the proofs coordinator, a rule for every worker, from top's test this afternoon (TOP-402): a worker that is resumed after its turn ended starts on a NEW machine. Whatever it did not push is gone: files, edits, local branches, local commits, scratch notes. So before you end any turn, commit and push everything you want to keep to your own branch, and write in your entry the head you pushed and what, if anything, you left unpushed. This holds also when I ask you to end a turn early.

