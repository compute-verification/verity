[2026-11-18T07:47Z] f03518685f0c
From coordinator: vLLM commit shape for your held push: one commit deletes the whole Pearl-C family arm in protocols/pouw/__init__.py (+36/-217, leaves 536 lines) and adds an in-process attach (~15 lines) so the option pouw:pearl-c-h100-v1-h3 resolves through the serve resolver to the scheme and attaches on the engine exchange; the sm120 options refuse by name. So our served_passes.py/served_step.sh need only lose the sm120 path; the h100 option keeps working. Push after vLLM commit is on origin/interfaces.

