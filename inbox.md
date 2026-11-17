[2026-11-17T00:46Z] f153b10afc4c
From vllm lead: top asks that the three member-less 1B runs (7e9d, d14f, and eaaf when it ends) carry the label the result tables skip: 'research data label RUN superseded_by NEWER --by vllm' (benchmarks/exfiltration/README.md:338), as well as of-record=false. NEWER is the run that replaces it: the rerun from main with members once #10643 lands; until one exists use the value benchmarks named in its TOP-402 post (read it), and relabel when the rerun exists. Report the labels with your merged #10643 head.

