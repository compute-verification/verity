[2026-11-13T05:00Z] f5f6fac3bfa9
training (coordinator), Thu 12 Nov 9:00 PM PST, to WALKER, on TRN-145. Two rulings and one cancel.

**1. Every reading carries the refusal. #9558 stays as written (`check` is called in `read` with no switch), and it goes to review.**
- Why: where a reading ends, the pass is a small part of it (your tiny run: 1.3 s beside 17.4 s). Where it doesn't end, the pass is not the cause: the memory is the reader's own word cache, on `main` before #9558. A reading without the refusal can charge products that repeat, so it is not an option to offer.
- Not known, and the body says so: the pass's own time and memory on a circuit larger than tiny.

**2. Yes to the cache change, as its own pull request off `main`, not on #9558.**
- Branch from `main` once #9578 is on it (it is in a candidate being checked now). Draft, `Refs TRN-145`, `Goal training/slowdown-from-submitted-circuit`, `Lane training`.
- The shape I want, yours to correct if the code says otherwise:
  - No list of an argument's words is kept. `Refs` already gives `refs[j]`, `refs.slice(lo, hi)` and `len`; `member_words` and the callers that take `[0]` read those.
  - "Does this range read a word twice" (`one_to_one`, `returned`'s per-field test, line 998's single-word test) is the only whole-argument question. Answer it from the parts or the leaf runs where the structure decides it. Where it can't, stream the range once and keep the answer, a bool for each (argument, range), never the words.
  - The few places that walk every word of an argument (`softmax_rows`, the constant tests) iterate `refs` and keep nothing.
- What must hold: `test_work.py` and `test_program_reading.py` pass unchanged. On tiny, and on the largest shape `main`'s reader finishes inside 48 GB, the whole `Work` is equal field for field between `main`'s reader and yours; put the tiny comparison in a test.
- What to measure, with your harness and its self-stop: tiny, s(2,64,512), then s-t20. For each: peak memory, and the time split (build, the reading before the pass, the pass alone), and the pass's own peak memory.
- **Draft with the tiny and s(2,64,512) figures by Fri 13 Nov 6 PM PST.** Launch s-t20 when s(2,64,512) ends inside its memory; send its lines when it ends.

**3. Cancel r20261113-041530-8901 now.** By your own figures it stops on memory hours before its first summary ends, and it would tell us only where. It holds 200 GB on a node other runs share.

**#9558, in order:**
1. When #9578 is on `main`: merge `main` into #9558 (never rebase), suites `--changed` that `main` commit on a node.
2. Body: every reading carries the refusal; the tiny run's split; that a language model's step at 1,024 tokens is not read yet, for the reader's memory, and which pull request takes that; RAN apart from READ.
3. Head and the run's last line **by Fri 13 Nov 9 AM PST**. I read its tests and mark it by noon.

**Your context:** you have been on this since Wednesday. If you can't take the cache change well, say so in your next message and write the handoff note (the harness, the three shapes, the readings above); I launch a new worker from it. #9558's merge stays yours either way.

