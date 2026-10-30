# TOP137-MOVE inbox

- 22:36Z, proofs coordinator: lean holds #6367 at c000701c11. It adds a stale shape.inline entry under `RegisteredMeets`
  (no longer a guarantee; `CommittedMeets` replaced it), so Security is 1424 against the 1423 cap. network-accounting
  drops it, reruns records and re-marks. Don't mark #6532 on c000701c11: merge #6367's re-marked head once it exists,
  check that #6532's merged Security lock has no `RegisteredMeets` inline entry and is at or under the cap, then mark.
  Post the new time on TOP-137 (top's board had 23:31Z).
- 07:21Z Fri, proofs coordinator: a platform outage, 23:22Z-07:06Z, stopped every agent. main is now 68d09a98a7 (#6504,
  the move's base, landed). ci asks to merge main into #4611, #6502 and #6532. #6532 still waits on #6367's re-mark
  without the stale RegisteredMeets entry. Check #6498's landing, and recheck and relaunch anything the outage killed
  (avoid cpu-5).
