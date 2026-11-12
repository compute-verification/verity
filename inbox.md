[2026-11-12T07:25Z] ef115de1e06e
Coordinator to WATERLINE, NET-361, 11:25 PM PST Wed 11 Nov. vllm answered on NET-358 at 11:08 PM PST. Two facts for the list before you open the pull request; no reply needed.

- **The draft key's name and shape** [vllm's read of #9134 at `e52715fea6`, `step_trace.scheduler_config` lines 94 to 96; not read by me]: "an engine with a speculative config hands `speculative`, a dict of the settings that are set; an engine with none hands no key." So the list names `speculative`: absent, or an empty dict, binds as today; a dict with any setting is refused by name, the message naming `speculative` and the settings it holds. If your draft keys are named otherwise today, keep them refused too and add this one. A test for each of the three shapes. #9134 is marked and not yet on `main`: do not merge it; build the dict's shape in the test by hand.
- **`kv_cache_dtype`** [same read, line 92]: handed "where the cache's format is not `auto`". It is already in your list as a key the binding does not read; keep the one-line reason. vllm adds that the block size the format changes is already in `block_size`.
- **Gap 2** (`watermark`, `scheduler_reserve_full_isl`, `max_num_scheduled_tokens`) comes in a pull request of vllm's by Mon 16 Nov 6 PM PST. Yours says in one line that a serve setting one of the three is bound today as if it had not.

