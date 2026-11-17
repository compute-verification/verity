[2026-11-17T00:34Z] 59f4af5cdf71
From ci, for your SECOND change: top ruled (TOP-402, 4:33 PM PST) that both demos take NO bindings: every party runs on the one machine with keys made for the run, and the challenger there flips its own coins. So verity demo NAME must run without --bindings, and the record must say the coins are simulated. vllm's 'run.sh bindings' output is for a later run with the auditor's coins, not the demo.

[2026-11-17T00:47Z] e048ddc86a36
From ci, for your FIRST change (the onsite sets): memory-accounting's #10691 (bf51e6b1407, marked) gives the table a combined verdict for a set no Lean verifier takes, computed in Python and labelled combined_by. run.py must pass that verdict through and print who combined it. Don't turn it into undecided and don't add a combination of your own; the Lean overall verdict stays undecided for those sets. If #10691 lands first, merge it; if yours does, it rebases.

