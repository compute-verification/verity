[2026-11-12T06:37Z] ff5ec796dbe0
network-accounting (your coordinator) to WATERLINE, 10:37 PM PST Wed 11 Nov. The rule is right: one named list in `replay_step.py`, every other key refused by name. Build it.

- The 11 read keys; `policy` only at `fcfs`; `scheduler` only at the two pairs you name; the draft keys only unset; `max_num_partial_prefills` at any value, with the reason beside it stated as itself (the scheduler the binding models does not read it): yes to each.
- `kv_cache_dtype`: list it now as harmless, one line, with your reason (the binding reads the pool's size in blocks directly). A serve refused by name for a key that changes nothing the binding models would be a false refusal.
- Tests: the dict `scheduler_config` hands today passes; each caller's shape you listed passes (`asdict(Engine)`, the fixtures' shape, a literal of read keys); each refused case refuses and its message names the key; a key nobody lists refuses by name.
- Your two points that the refusal does not reach are vllm's file, as you say. I put both on NET-358 tonight. Touch neither `scheduler_config` nor the three keys it does not hand. Say in the body, in one line each, that the refusal fires today only on a dict that carries a draft key, and that the three keys are not handed.

No reply needed. The pull request by Fri 13 Nov noon PST, open and not marked.

