[2026-11-11T06:45Z] d95bc97edb7b
From TRN-97's worker (bc-831fdbe0), Tue 10 Nov 10:45 PM PST. LAST COMMAND BEFORE THE STOP: the last one my record holds is a read of the node run's output at 9:39 PM PST (05:39:09Z), then a 270-second wait with no shell command; both finished, no error seen. What I ran between about 9:44 PM and 10:12:54 PM PST is in neither my context nor the platform's transcript of me, so I cannot say whether a command was in flight. Not looked into further.

#8816 IS MARKED READY at 62b53e40ad24, at 10:44 PM PST (06:44Z). Marked is not landed. The queue reads now: "#8816 62b53e40 ready training admitted: the verified queue builds it into a batch onto main once it has a land verdict, or on its mark alone when it is low-risk".

Result line of the node run (run, r20261111-051647-2a5c, vy-nebius-cpu-9, at 62b53e40ad24, rc 0, ended 06:36:11Z): "49 suites: 49 passed (4 cached), 0 failed; 99.0 min of test time". In it: the calculator's suite 603 passed in 30 s; the root suite (repository) 280 passed in 250 s; verity/protocols/spot/verifier (it owns profile.py) 825 passed. It waited 11 min for a slot and ran 68 min.

The mark: "research queue ready 8816 --mark --by training --sha 62b53e40..." gave "preflight: #8816 at 62b53e40ad24 merged onto main at 5c8c343f19f0: nothing refused (107 s; collected 49 suite(s))" and "marked on training's word, with no quick tier". No refusal, so ci's way around was not needed and --no-preflight was not used. Made once: before it I read the queue (not marked), the branch on origin (62b53e40ad24) and my processes (a new VM, none older than a minute).

Body: names the run and its counts, says it is marked and not landed, says your ruling on the sharper lower assertion (kept), and that main has moved to 5c8c343f19f0 without touching these files. One thing to know: "research queue edit --body-file" replaces the whole body, so it dropped the "Refs TRN-97" and "Goal training/calculator-count-held-to-core" lines that "open" had written; I put both back in the file and edited again before the mark. They are there now (read back).

Still open, yours: which delta the injected term carries. Mine: nothing, unless drive rings me for a red that names this change. The note is training-20261111T0324Z-report-trn97-injected-vs-infiltration, section "Step 3, done: the test, in pull request #8816".

