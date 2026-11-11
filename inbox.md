[2026-11-11T21:43Z] 1328f3c3b963
From the circuits coordinator, 21:43Z. Data from infra (CIR-215, 21:31Z), one rule added to your brief; no reply needed, go on with your task.

Keep the long row's runs off vy-nebius-cpu-6 until Sat 14 Nov. vllm's audit runs there from about 02:00Z (6 PM PST) today with 540 GB reserved, and a run that grows past its own reservation on that node is the first one killed.

- If a launch line says a staging, figures or circuit-check run of yours went to cpu-6, cancel it (`research cancel ID --reason ...`) and place it on another CPU node's research pool. Not cpu-2 without asking me.
- Every run states `--mem-gb` at the peak you expect plus a margin, as the brief says. A suites run with `--mem-gb 64` may stay where `--on auto` puts it.
- I read that r20261111-204850-da1a, -203604-3b94 and -203616-09fc are cancelled and -203451-b8c0 ended rc 1 on cpu-10; say in your report what b8c0 was and why it failed.

