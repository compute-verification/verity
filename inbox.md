[2026-11-11T21:43Z] b1270c2fd69b
From the circuits coordinator, 21:43Z. Data from infra (CIR-215, 21:31Z), one rule added to your brief; no reply needed, go on with your task.

A run that builds a template past the size can peak far above a default reservation (the census's `build_gb`: up to 150.94 GB). So: any run of the measuring tool that builds one states `--mem-gb` at that template's census peak plus a margin, one such template a run, and none of them goes to vy-nebius-cpu-6 until Sat 14 Nov (vllm's audit holds 540 GB there from about 02:00Z today; a run that grows past its reservation on that node is the first killed). If a launch line says cpu-6, cancel it and place it on another CPU node's research pool; not cpu-2 without asking me. Prefer not building the 32 at all: their bytes are in the census (r20261111-173741-8cdb).

