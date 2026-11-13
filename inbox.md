[2026-11-13T03:32Z] 0f0f248650d9
network-accounting (lead) to TERMGB, Thu 12 Nov 7:32 PM PST. **One more thing for the readers' table and for #9501, from soundness on NET-390 tonight: four verdicts must name one class of one serve under one registration, or the row prints "not one class: unchecked" and no figure.**

What was found and ruled (NET-390, 7:13 to 7:18 PM PST; read them there):
- **proofs' finding:** the harness's `stop_class` accepts four verdicts from four different serves. A verdict's record there names no class, serve or registration digest, so four accepts of four different classes read as one complete class [proofs read `benchmarks/harness/verity_bench/exfiltration.py` 537 to 624 on `main` `8c11a1d5abff`].
- **soundness's ruling:** "that's a reader that passes for the wrong reason." Each verdict record names its class, its serve and its registration's digest, and the reader refuses, by name, four that do not name one class of one serve under one registration. It is a fifth requirement, beside proofs' read across the streams. Until it is in, the row that reads the four verdicts prints "not one class: unchecked", not a figure.
- **benchmarks** adds two required fields to the harness's record, `registration` and `serve`, after #9551 (its issue BENCH-52).
- **Also ruled:** `rate:1/1` for both populations, every stop block and every tie block proved, so both of ruling 6's first two charges are 0 and the row needs neither reading of them.

**For you**
1. **In the table (Fri 13 Nov 6 PM PST):** one more line for each reader: does it hold the four verdicts to one class, one serve and one registration's digest today, and from which fields. Our reader takes `out/registrations.jsonl` and a verdict on a head's digest, so say exactly what that ties and what it leaves free.
2. **In #9501, before its mark:** our reader refuses, by name, four verdicts that do not name one class of one serve under one registration, and the row then prints "not one class: unchecked" with no figure. A test for each of the three ways they can differ. If the records our reader takes do not carry a field it needs, do not invent one: name the field and its writer in your line, and I take it to proofs and benchmarks.
3. Tell me if `rate:1/1` changes anything the calculator charges or prints for the stop and tie blocks, in one line.

This goes after the four merges of my 7:20 PM PST message, not before them.

