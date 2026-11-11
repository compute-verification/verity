# NET-297: the three answers proofs owes network accounting (reader's draft for the proofs lead)

Read on Wed 11 Nov 2026 between 19:48Z and 20:10Z. Nothing was built or run. Every line below is a read.

**How to read the marks.**

- A `path:line` with no mark is a file I read on `main` at `61f13c2275fb`, in `/tmp/wt-main-ro`.
- "(#9028)" marks a file I read at pull request #9028's head `23baf41620ff`. That commit is not on `main`.
- "(relayed)" marks a statement from an inbox entry or a note that I did not check against code.
- "My reading" marks an inference of mine. "Not read" marks something I did not open.
- "The inbox" is `/workspace/inbox/proofs-coordinator.md`; I give its line numbers.

**One thing to know first: the stop record is not on `main`.**

- On `main` a class's boundary record (the record serving registers for a class's spans) has two ports, `b.ingress` and `b.egress` (`adapters/vllm/verity_vllm/pipeline/boundary_rows.py:67-68`).
- `b.request`, `b.prompt`, `b.stop`, the stop record and the order check appear nowhere under `adapters/` or `verity/` on `main` (search, no hit). `benchmarks/one_stage/two_stage.py` on `main` (307 lines) has no stop code.
- They exist on #9028 only. The design brief and the 19:19Z entry speak of them as if they were on `main`. Where an answer below describes a stop record or a tie, it describes #9028 or a design, and says so.

## Answer 1. Where the two tie blocks sit

**I recommend a record of their own beside the stop record: every read the ties need is one a one-stage record already makes, while as templates of the replay block the response tie would be sampled by coins that are released before the stop record exists. The price is that the request tie reads and checks `b.request` instead of writing it, and that the pooled count needs a statement no file on `main` holds.**

So the answer to network accounting's question 1 is yes on both halves: the ties go in a record beside the stop record at a registered rate, and the request tie cannot write `b.request`.

**What a block in a record of its own may read.**

- **Its own record's ports.** The registration check (the eight checks R1 to R8 a one-stage record passes) holds a record's roots to exactly the ports the verifier's own layout derives, each with the verifier's leaf count and domain (`verity/protocols/spot/verifier/one_stage/registration.py:21-22`).
- **Another record's port, by digest.** This is an "output value": a registered value that takes its root and domain from a port of an earlier record, and reuses that commitment instead of making a new one (`registration.py:86-91`). The Lean verifier does the same (`verity/protocols/spot/verifier/lean/Spot/Registration.lean:309-320`, `:352-361`, refusal code `R7-output`).
  - The earlier record must be in the stream's log, the list of the prover's records the challenger logged before the stream (`verity/protocols/spot/verifier/service.py:1547-1549`, `register`'s `outputs` at `:1879`).
  - On #9028 the stop record reads the class's boundary record this way: its two inputs are that record's ports `b.request` and `b.egress`, and `b.stop` is its one port (`adapters/vllm/verity_vllm/pipeline/boundary_stop.py:16-18`, `:191`; `benchmarks/one_stage/two_stage.py:364`, `:367`) (#9028).
- **It cannot write another record's port.** An output value's rows stay under the earlier record's commitment and the reads open against that root (`registration.py:89-91`); the record's own roots are its own layout's ports and no others (`:21-22`). So a request tie in a record of its own reads row r of `b.request` and checks it. Network accounting's own inbox entry of Wed 10:09Z says the same (inbox :47330) (relayed).
- **The certifier's registered content rows.** On `main` the ingress tie's service does this: the certifier's root is the tie record's own port under the certifier's domain (`verity/protocols/certifier/verifier/tie_service.py:381-382`), and the service's Python holds that root and domain to the certifier's signed head (`:552-558`). The Lean verifier does not hold it.
- **Flock's side.** Flock's verifier checks each port of its registered file on its own, at the root the file gives, and compares no two ports (`verity/protocols/flock/verifier/Flock/Registered.lean:120-149`, `:266-294`). Every port of the circuit must be in the file (`:364-375`).
- **Not shown anywhere:** one record that has both a port under the certifier's domain and output values from another record. The ingress tie's record does the first and #9028's stop record the second. My reading is that nothing forbids both, because the verifier gives each port its own domain (`service.py:1873`, `tie_service.py:381-382`). No run shows it.

**Whether a block is matched to a row by position, with the prover naming nothing: yes.**

- The verifier derives each block's row from its own program and query; a read at a computed position is refused as no public read (`verity/protocols/spot/verifier/one_stage/registered.py:34-37`, `:382-441`, the sentence at `:389`).
- Rows are fixed slots: the position is a function of the block's index alone (`verity/protocols/spot/verifier/one_stage/served.py:39-44`).
- Flock holds each proved unit to the row the verifier's file names for it (`Flock/Registered.lean:262-264`, `:286-294`).
- The limit the lead posted at 19:34Z still applies (relayed): the file names a row, never a slot inside a row. "Block (class, r) reads row r on both sides" works when row r is one whole row on each side.

**How a record beside the stop record is registered, and at what rate.**

- On `main` there is no such record. The parts it would use are there: a stream opened with a log (`service.py:1547-1549`), a one-stage rate as the Spec's sampling design (`Law`, `rate:NUM/DEN`, `service.py:303-305`), and one record a window for each stream (`registration.py:25-26`).
- On #9028 the stop record shows the pattern (#9028, all in `benchmarks/one_stage/two_stage.py`):
  - its Spec has `b.stop` as the served value and a rate as its design (`stop_spec`, `:334-348`);
  - its stream's log is the class's boundary record; it registers, commits the coins to the record's digest and samples once (`stop_first`, `:351-373`);
  - the rate is `--stop-design`, and defaults to the first stage's rate (`:852`); δ is the same number the class's stream gets (`:892`).
- My reading: a tie's record is one more stream for the class and window. Its log is the class's boundary record, then the stop record where the response tie reads `b.stop`. Its Spec has a rate. It is sampled once, after its own receipt.

**What charging a wrong tie in one pooled count at δ needs.**

- **In the Python: something new, because every stream stands alone today.**
  - Each stream has its own δ, bound in its own digest (`service.py:1591`, `:1638`), and its outcome reports that δ and a profile of its own record and sample (`:2488-2497`).
  - Nothing in `service.py` adds a count across streams or records (my reading; a search for a pooled count in `service.py`, `second_stage.py`, `Spot/Record.lean` and `plancheck.py` finds none).
  - Passing the same number to two streams, as #9028 does, gives two statements at δ each. It does not give one count at δ (my reading).
  - The function that returns the bits a two-stage point certifies, `checkTwoStage` (`verity/protocols/spot/verifier/lean/Spot/TwoStageCheck.lean:85-88`), is called only by the planner's check program (`Spot/PlanCheckMain.lean:21-22`, `:85`). No verifier path runs it on an audit.
- **In the Lean statement: a statement that takes more than one record.**
  - The three statements on `main` each take one audit statement, one partition and one two-stage design (`security/Properties/Spot/TwoStageCheck.lean:69-73`, `:78-85`, `:100-107`). A tie in a record of its own is a block of none of them.
  - The one statement that pools at one δ is for one-stage stratified samples over a period, not for a two-stage design (`security/Proofs/Flock/EndToEnd/Joint/Statement.lean:56-62`, `:270-292`).
  - So the pooled statement (ask 4 below) has to count the class's two-stage design and the one-stage records beside it, the stop record and the ties, at one δ.
  - A point has one first-stage rate for all its classes (`Spot/TwoStageCheck.lean:48-55`, `:59`). My reading: a tie record could be counted as one more class only if its rate is at least that rate. Nothing on `main` states this.

**What the other choice, templates of the class's replay block, would need.**

- **The order is the wrong way round.** On #9028 the class's first stage has drawn its sample before its stop record is registered (`two_stage.py:890-893`), and the order check refuses a window where a coin released after the boundary registration comes after the first coin released after the stop registration (`:412-414`) (#9028). A tie inside a replay block is chosen by those first-stage coins. Soundness's condition is that the response tie's coins come after the stop record's receipt.
  - Fixing it means registering the stop record before the first stage's coins, and making the Lean check of the first stage's coins take the stop record's digest as its "after" (`Spot/Run.lean:161-164`, `:259-260`). Network accounting's entry of Wed 10:47Z found the same and guessed 30 to 60 Lean lines (inbox :49291) (relayed).
- **No read of `b.stop` at the sampled block's row.** The second stage's read check lets a port read a boundary port of the stream's own boundary record at the sampled block's row, a wired port, its own port, or a registered value at rows fixed before the run (`Spot/SecondStage.lean:43-55`, `:61-69`). `b.stop` is a port of another record, and its row depends on the sample. My reading, from the module's docstring; I did not read the functions.
- **The certifier's rows would be ports of the class's boundary record.** The class's Spec on `main` names only `b.ingress`, `b.egress` and the interior values (`benchmarks/one_stage/two_stage.py:95-106`), and a class port's domain binds the class's program and partition (`boundary_rows.py:262-266`), which the certifier's domain does not.
- **No two-stage run passes on `main` at all.** The switch is off in Lean and in Python (`Spot/SecondStage.lean:134-139`, `verity/protocols/spot/verifier/second_stage.py:83-84`), and the replay block is still a stand-in (`benchmarks/one_stage/two_stage.py:22-24`).
- **What it would give.** The count needs no new shape for one window: the tie blocks would be proof blocks of the class's own statement, covered by the class's cells (`security/Properties/Spot/TwoStageCheck.lean:49-51`). The request tie could also stay the writer of `b.request`, as a template whose output the replay block returns (`Spot/SecondStage.lean:70-76`). Both are my reading.

**What changes, and in whose files.**

- **A record of their own.**
  - Network accounting: the request tie's Definition (it reads and checks `b.request`), and a tie service beside `verity/protocols/certifier/verifier/tie_service.py` that holds the certifier's roots to the signed head.
  - The vLLM lane, on #9028: a Spec and a first step for the tie's record beside `stop_spec` and `stop_first` in `benchmarks/one_stage/two_stage.py`, and the order check run once more.
  - Proofs: the pooled statement under `security/Properties/Spot/`, its count beside `Spot/TwoStageCheck.lean`, and the audit driver's entry.
  - My reading: no change to the Lean verifier's registration or read checks. Not run: the stop block's own accept is not run yet (the 19:19Z entry, inbox :62295) (relayed).
- **Templates of the replay block.**
  - Proofs: `Spot/Run.lean`, `Spot/SecondStage.lean`, their Python mirrors and difftests.
  - The vLLM lane: `boundary_rows.py`, `two_stage.py`'s `spec_of`, the replay block's Definition, and the order check turned round.
  - Network accounting: both ties as templates. The challenger also changes (relayed, inbox :49291).

## Answer 2. `complete`

**`complete` is computed, never written, so it turns true when the driver gives the stop record's request entry a verdict; the driver would read that from the tie's own stream, which is a Python read, so the entry is not a verdict of record and must name that as its premise.**

- **Today.**
  - `complete` is true when every entry of a record's `verdicts` has a verdict that is not null. Python computes it (`verity/protocols/spot/verifier/one_stage/audit.py:147`, `:155`) and Lean the same (`Spot/Record.lean:380`, `:431`).
  - On `main` no driver writes a stop record. The entry names `request-row`, `cross-stream-order` and `boundary-audit` appear nowhere on `main` (search, no hit).
  - The driver the 19:19Z entry plans is not built. It writes an entry `request-row` with `"accepted": null`, which alone keeps `complete` false (relayed).
- **From what the driver would read that the class is in the population.** With the ties in a record of their own, two things in the tie's stream:
  1. in its log, the tie's registration record, whose output values name this class's boundary record by digest, under the request tie's Definition as the verifier builds it;
  2. that record's outcome, from an audit of record.
  - This is the read the ingress tie's service makes of its own stream on `main` (`tie_service.py:593-626`, the log and outcomes at `:614-623`).
- **That read is Python, not Lean.**
  - Lean holds one record to another only inside one stream's log (`Spot/Registration.lean:309-320`). A log holds registration records and none of them is sampled (`service.py:1547-1549`), so another stream's outcome is never in it.
  - A verdict of record must name its own record's sessions (`service.py:691-707`). So the entry is `of_record` false.
- **The premise the record must name while it is Python.** Soundness gave the words for the like entry at 19:45Z (relayed): "boundary stream's audit: its verdict carried from that stream's audit record by the audit driver, not checked by the Lean verifier". The request entry needs the same two facts said of the tie's stream: its verdict is carried by the audit driver from that stream's audit record, and the driver, not the Lean verifier, read that the tie's record names this class's boundary record. The exact words are soundness's to give.
- **One change of meaning to say plainly (my reading).** In the 19:19Z design every row had a tie verdict. Now the ties are sampled. A verdict on the request entry then says the class's request rows are in an audited population at a registered rate, not that this window's rows were each tied. What that is worth is the pooled count, which has no statement yet (ask 4).
- **What carries over from the 19:19Z entry and still holds.**
  - `complete` true is not `accepted`. Every entry must have a verdict, the order's and the boundary stream's audit among them (relayed; not built).
  - A sample gets one outcome (`service.py:2257-2264`). So the stop record's outcome is written after the tie stream's outcome for the same window and inside the proof deadline, or `complete` stays false for that window for good.
  - `benchmarks/network_traces/leak_bounds.py` on `main` reads no `complete` and no `of_record` (search at `61f13c2275fb`, no hit). The calculator that "charges the count while it is false" is not on `main`.
- **If the ties were templates of the replay block instead.** The class's boundary record is already in the stop stream's log on #9028 (`two_stage.py:364`) (#9028). That the class is in the population could then be read from a record the Lean verifier already holds. The audit of that stream would still be carried by Python (my reading).

## Answer 3. The six asks of Mon 9 Nov

**Five of the six have an answer drafted in the inbox on Tue 10 Nov and the sixth is answered in part; I could not read Linear, so whether any was posted is not established, and none of the changes they promise is on `main`.**

The asks are in `~/.research/notes/lanes/network-accounting/20261110T0545Z-draft-net297-content-tie.md:149-155`. The inbox has no entry under the name TIEPOOL (search, no hit).

1. **The Spec** (register the certifier's head in each class's Spec).
   - Drafted Tue 10 Nov 16:15Z in ACCOUNTD's entry: "taken, as a per-window port, not a registered value" (inbox :28555). It says it could not post as proofs (inbox :28528).
   - On `main` the class's Spec names no certifier's head (`benchmarks/one_stage/two_stage.py:95-106`).
   - With Answer 1's recommendation it need not: the certifier's rows are ports of the tie's own record, as the ingress tie's are (`tie_service.py:367-372`, `:381-382`).
2. **The stratum** (is the size-class rule the rule, and when does the cap lemma cover those sets).
   - Half answered Tue 10 Nov in the proofs note `lanes/proofs/20261110T0619Z-draft-accountd-1200-report.md:106`: the rule keeps the patterns' bound tight and is not a soundness rule. "When" was not answered.
   - From `main` now: the capped form's statement and its proof are on `main` (`security/Properties/Spot/TwoStageCheck.lean:100-107`, `security/Proofs/Properties/Spot/TwoStageCheck.lean:312`). It caps every admissible set (my reading), under a premise about served bits (`ServedCover`, `:89-95`) that no run checks (relayed, the note :80).
   - Two sources disagree on the stop block's width: network accounting's ask says a stratum at least 64 bits wide, and BREAKEVEN asks whether 16 is deliberate (inbox :29276). With the ties in a record of their own they are in no stratum of the replay block.
3. **The charge** (does λ apply to a wrong tie block).
   - Drafted Tue 10 Nov 07:35Z in ACCOUNTD's entry, read then at #8400's head: no when the block is charged S a request, yes when it is charged its committed bits below S (inbox :26986-26990).
   - The names it cites are on `main` now. λ is the exponent of the assumption that faults hand out the weights at most N^(λ-1) times faster than honest use (`FaultedUseHard`, `security/Definitions/Assumptions/WeightSecurity.lean:25-40`). The guarantee's docstring says a fault a cut covers adds only its hint and needs no such assumption (`TheftFaulted`, `security/Definitions/WeightSecurity/Guarantees.lean:166-168`).
   - I did not establish that a wrong tie block is a fault of the served function in that statement's sense.
4. **The pooled statement** (who states the count over a period at one δ).
   - Drafted Tue 10 Nov 16:15Z: "ours", a new statement beside the capped form's, the statement by Thu 12 Nov 12:00Z and the proof by Mon 30 Nov, a guess (inbox :28559-28564).
   - It is not on `main` (Answer 1, the Lean statement). COUNT's entry of Wed 19:40Z found the same for a period (inbox :62778) (relayed).
   - It is a Lean statement under `security/Properties/Spot/` with its proof under `security/Proofs/Properties/Spot/`. With Answer 1's recommendation it must also count the one-stage records.
5. **The workload card** (the two tie Definitions beside the stop block).
   - Drafted Tue 10 Nov 16:45Z in BREAKEVEN's entry: proofs takes it, the stop block first and the ties once their Definitions are on `main` (inbox :29261-29272).
   - On `main` the card names neither: a search of `benchmarks/spot_figures/cards/workload.glm-5.3.json` for "tie", "stop" and "certif" finds nothing. Its strata come from a census of the served circuit (`:7`).
   - It is a card file filled from that census, so it changes after the Definitions land. Whether ties in a record of their own are strata of that card or a line of their own is open.
6. **Which form** (capped or patterns').
   - Drafted Tue 10 Nov 16:59Z in BREAKEVEN's entry: the rows of record use the patterns' form, and both forms rest on a premise no run checks (inbox :29470-29477).
   - `main` agrees for the request-grain line: its exfiltration is the patterns' form, with the capped sets' form as the comparison (`benchmarks/spot_figures/total_cost.py:18-21`).
   - The patterns' premise needs each request to serve exactly the point's S bits (`security/Properties/Spot/TwoStageCheck.lean:63-64`).

## The order check and one more reader behind the stop record

**As it stands it does not cover the response tie; it needs the same comparison run once more, not a new kind of case, as long as the tie is in a record of its own.**

- The check is on #9028, not on `main`. It compares two receipts, the class's boundary registration and the stop registration, and holds only the stop sample's coins (`benchmarks/one_stage/two_stage.py:384-390`, `:396-414`). Its docstring rests on the stop stream being the one reader of `b.stop` (`:392-393`) (#9028).
- With the response tie in a record of its own, the same function covers it when called again with the stop registration as the earlier record and the tie's registration and the tie sample's coins as the later one. Its body already takes the two digests and the coins as parameters (`:376-377`, `:396-397`) (#9028). The docstring's sentence about one reader must change. My reading.
- Inside the tie's own stream, two checks on `main` already give part of it: the tie's record is refused unless the log holds the record whose port it takes (`Spot/Registration.lean:352-361`), and its coins must be released after its own digest (`Spot/Run.lean:161-164`, `:259-260`).
- With the response tie as a template of the replay block it needs a different check, because the present one requires the first stage's coins to come first (`:412-414`) (#9028).

## Not established

1. **Whether any drafted answer to the six asks was posted** on PRF-456 or NET-297. I did not read Linear. Reading the two issues settles it.
2. **Whether one record can hold both** a port under the certifier's domain and output values from the boundary record and the stop record. An accept of such a record at the smallest class settles it. The stop block's accept, not yet run, settles the output-value half.
3. **Whether the two ties share one record.** The lead's 19:34Z fact gives two Definitions for a stop class and a class without one. Network accounting's Definitions settle it.
4. **The pooled statement itself**, for a two-stage design with one-stage records beside it, in a window and over a period. I did not re-read the period statements COUNT cites (`security/Proofs/WeightSecurity/TwoRegPeriod.lean:131`, `Two.lean:236`, `security/Proofs/Flock/EndToEnd/Joint/Prod.lean:28`); I saw only that they exist at those lines.
5. **Whether a tie record at the first stage's rate can be counted as one more class of a point.** I did not read how `rate:NUM/DEN` draws its units. Reading `Spot/Select.lean` and the rate's parser settles it.
6. **What the second stage's read check does in code** with a registered value no link names. I read its docstring only. Reading the check in `Spot/SecondStage.lean` settles it.
7. **The premise's words for the request entry.** Soundness gives them.
8. **Whether a wrong tie block is a fault** in the weight-security statement's sense (ask 3). The statement's owner settles it.
9. **The stop block's width class**, 16 or 64 bits (ask 2). #8925 and network accounting's stand-in settle it; I read neither.
10. **Where network accounting's calculator reads `complete`.** It is not `leak_bounds.py` on `main`. Its branch settles it.
11. **What #9023 changes** in the check of a record's registered values in `Spot/Registration.lean` (relayed, the 19:19Z entry). Not read. It could change what an output value takes.
