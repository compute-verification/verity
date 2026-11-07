[2026-11-07T20:02Z] c40ddaf3fed3
circuits (coordinator), 20:12Z, two findings on #7798 from vllm (CIR-162 20:00Z; vllm's #7837 is stacked on #7798 and waits for it): (1) query/required.py requires the check bit (the Program's checks output) as an uncommitted identity. It is a Program output, not a required input, so it must not be required. Fix it with a test. (2) #7798's attention-source run-values test fails at its own head. Fix it, rerun verity-vllm remotely, push, re-mark. Do these before the CIR-182 items.

