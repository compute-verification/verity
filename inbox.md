[2026-11-17T05:05Z] 7d270a1d11aa
From vllm lead: while you are in #10557/#10688: ci's reviewer notes the retired node name vy-dc-h200 is still written in served_run/placement.toml (ssh to it times out; node A is vy-spot-glm53). Replace every use with the current name from the one place the placement names the node, with the test, in the same push; say in your report where it was and whether any run tonight read it.

