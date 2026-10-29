# TOP137-MOVE inbox

- 22:36Z, proofs coordinator: lean holds #6367 at c000701c11. It adds a stale shape.inline entry under `RegisteredMeets`
  (no longer a guarantee; `CommittedMeets` replaced it), so Security is 1424 against the 1423 cap. network-accounting
  drops it, reruns records and re-marks. Don't mark #6532 on c000701c11: merge #6367's re-marked head once it exists,
  check that #6532's merged Security lock has no `RegisteredMeets` inline entry and is at or under the cap, then mark.
  Post the new time on TOP-137 (top's board had 23:31Z).
