[2026-11-12T19:21Z] 522e48df5376
BOUNDTIE, from the network-accounting coordinator, 11:21 AM PST. **A read-only job beside #9378's one change, which keeps its hour: vllm asked us on NET-297 (10:23 AM PST) to name the rule that files a release, and two things with it. I rule at 6 PM PST today from your answers, so they are due by 4 PM PST. Nothing is built for this yet.**

**Why now.** vllm files the compute side by the release (VLLM-410) once two things are fixed, by Mon 16 Nov 6 PM PST: the rule, and the certifier's word of a phase reaching the serve. vllm applies "the same rule to the same inputs". Today the two sides differ: vllm has a stop class (P, T) whatever n and a no-stop class (P, O), in admission order; we have `content_rows.file`, the first class that holds the release, in release order. vllm's comment is on NET-297; read it whole first.

**1. The rule. Compare these two and recommend one.**
- **(B), the thread's:** the schedule gives each class a kind. An `END` goes to the first stop class that holds it, a `FAIL` to the first no-stop class that holds it. Read from `Released` alone (`failed`, `request_bytes`, `outputs`).
- **(S), the rule as built and nothing new in the schedule:** the first class that holds it, and the response tie reads the end row's kind a row: an `END` row is held to the stop word (`at` = n), a `FAIL` row to `at` ≤ count ≤ O. The stop block is a function and gives a word for every row; the tie uses it only on an `END`.
- **For each, with file and line where it is code:**
  1. what changes in the schedule, its check, the parameters digest and the form's value, and whether any kept digest moves;
  2. what the compute side commits a span (stop word, count) and what is charged for it: under (S) is the count's advice charged at every span of a class, or only at its `FAIL` rows, and is a class's number of `FAIL` rows public anywhere;
  3. what the per-class counts in the head show under each (a stop class hides n; size classes show its size), and whether the count's own line already charges that;
  4. an `END` the stop rule does not read (a body's stop strings, `min_tokens` not 0, a length past T): can a request the certifier took in under the id form carry any of them? Read `active.id_request` and the ingress tie's fields. If none can, say so; if one can, say what each rule does with it;
  5. the tie's size a row at O = 1,024 under each, and what else of ours changes (`response_tie.py`, the Definitions' pins, NET-381, NET-386).

**2. The certifier's word of a phase.** vllm needs, before a class commits: each release's number, bucket, end, released count, and when the list is final. At #9378's head: what the hand of a phase holds for each release, when it is final, and how a serve on the same deployment could read it today (a file, a stream, or not at all). Then what is missing against vllm's list, and the size of adding it.

**3. A release the server never received.** vllm's read: the certifier numbers a release at the request's `END` whatever the server connection's state, so it is a held row with 0 ids and counts in n, and the compute side holds no span for it.
- What does the device record for it: is it `failed` with 0 outputs, and can the device's own record tell it from a request the server took and that failed before any id?
- **My proposal to soundness, which I post now marked as argued: a `FAIL` after no id is filed in no class.** It reads only the certifier's record, no response id left on it, and both sides can apply it. Say what it changes in `file`, in the tests, and in a period's K on your toy (I expect nothing: 0 ids), and anything that breaks it.

**4. NET-381** (the no-stop form holds `at` ≤ count ≤ O) now has vllm's yes and soundness's charge. Say whether you take it after #9378's change or it wants another worker.

Mark each line [read], [ran] or [argued]. If 4 PM PST cannot hold for all four, send 1 and 3 first.

**One fact for everything you push from now on: the repository is public on GitHub today** [checked by me with no login at 11:14 AM PST: a file on `main` and one on a working branch of ours both open; top carries it to Daniel, TOP-413]. Until Daniel answers, every commit, branch file, entry, mailbox line like this one, pull request body and comment is public text. In all of them: no node, host or pool machine's name, no IP address, and of a key or token only its environment variable's name, never a value. Write "a CPU node", "the pool's node". Rewrite no history, delete no branch and change no setting for this.

TIMER: none

