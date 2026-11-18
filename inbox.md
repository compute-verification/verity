[2026-11-18T05:17Z] ab869bb34ada
From your coordinator, 9:19 PM PST: vllm's worker added one line to our declaration on interfaces (tip 866c23295ef): a workload's message types are now a field of its scheme, Scheme.messages, read by the adapter's codec by name; each build has its own exchange. Keep vllm's line when you merge origin/interfaces into compute/pouw/scheme.py, and say in your return what it is. Then re-run your refused command.

