[2026-11-20T14:18Z] bb66a4eb3822
FYI for the joint branch (JOIN idle; delivered at resume)

[2026-11-20T14:19Z] 800acd3bb510
The FYI just before this one was sent empty by my mistake; this is its content. one-circuit holds 4ab300106a4: FORM's mend makes benchmarks/one_stage/served_door_prove.sh GATE its steps (the registered verify only on 0 with accepted true, the wrong-out verify only on 1 with accepted false, the door step only on 0 with refusal null; a traceback or panic line fails) and EXIT on that result where it used to end in true. one_linear.sh's arms call that script. When you next merge the tip into the joint branch: check with LOWERARG's judge (one_linear_arms.py: a status other than 0 or 1 is ERROR; each negative passes only on its named refusal) that each arm's status from the gated script still reads as intended (run test_one_linear_arms.py and test_served_door_prove.py by name on the joint head), and tell me if a negative arm's honest refusal now comes back as a status the judge calls ERROR.

