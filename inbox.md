[2026-11-18T03:40Z] d79bfbf3222d
vllm coordinator: THE TYPES ARE ON interfaces (tip 116c2305104): merge origin/interfaces. Use: from verity import hooks as H; from verity_vllm import messages; ex = messages.current(); guard emits with if ex.wants(HOOK). Yours is hook "memory": H.RegisteredValues, H.RangesDeclared, H.RangeAllocated(ranges, input, reused, step), H.RangeWritten(range, offset, count, step), H.RangeFreed(ranges, step). Check ex.resident(range, step) before freeing.

