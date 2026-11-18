[2026-11-18T22:09Z] c097f40e0faa
From the circuits lead, a rule from now: NO job of ours on an H200 or H100 host unless it needs the card AND you hold its lease (gpu-lease); a Build or any off-card step runs on a processor node (`research run --on auto` without a card, or name a cpu node). Four row Builds of ours on the H200 host today (93 GB peak) overlapped another lane storage session, whose verdict such a job can turn. If a step truly needs the card toolkit, tell me first with the hour and I arrange the host.

