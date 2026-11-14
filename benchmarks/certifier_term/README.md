# certifier_term

`verity bench certifier_term` prints the network certifier's term for GLM-5.3: the bits its accepted transcripts can
carry over a monitoring period, at the load of a node, the census's reference datacenter and the world, in bits and
GB, beside a target and one copy of the weights. Each row counts all its scope serves by its check's own rule, as
the monitoring-period check charges it (`verity/protocols/accounting/certifier/verifier/lean/NetworkCheck/`, `rowsTerms` and `terms`).
Its code is the calculator of K, `benchmarks/network_traces/leak_bounds.py` (`table`, `document`, `render`); this
directory holds its runner, which with `--out` also writes the term of record's row as a result (The result, below).

```text
verity bench certifier_term                              the inputs, the scopes, and the terms in GB and in bits
verity bench certifier_term --json                       the same as one document: value, unit, commit, run, inputs, rows
verity bench certifier_term --scope world --period year  only those rows (each flag repeats)
verity bench certifier_term --requests census            the census's request in place of the sampling plan's
verity bench certifier_term --out DIR                    also its result, for DIR: the amounts of the term of record's row
verity bench certifier_term --out DIR --registration F   with its capacity and verified, read from the registration log F
verity bench certifier_term --response-ids 1024 --out DIR   the result of record (The result, below)
verity bench certifier_term --scan-delta D               each execution phase's scan at δ = D, in place of the default
verity bench certifier_term --help                       every parameter and its default
```

It reads files and computes, so it needs no GPU and no run. Under `research run`, give the document its run and
commit: `--run "$RESEARCH_RUN_ID" --commit "$RESEARCH_SOURCE_SHA"`. Without `--commit` it names the checkout's HEAD.

The document's `value` is the term of record: the scan's term over the world's year, charged at two release
schedules (D_j = 2), in GB. The term at D_j = 1 is what a network certifier with a set-time card, a network card
that releases at a set time, would be charged. That card is simulated: none is deployed, and the network
certifier's software pacer stands in for it. So a figure at D_j = 1 is never the term of record and is printed only
under the label "with a simulated set-time card": the document's `with_set_time_card` holds the scan's term over
the world's year at D_j = 1, its unit, that label and its mark, `simulated`. A row whose release times are taken
once (`timing`, `all three`) has a jitter of 0 at either D_j, which rests on the same card, so every figure of it is
under the label: the row's own `with_set_time_card` is the label, where every other row's is null, and the tables
print the row's name with the label.

The card is an input of the document, `set_time_card`, the one input marked `simulated`. Its source is what a
figure under the label rests on, a clause of `trusted-host`: the network certifier's card releases each bucket at
the tick its record names, whatever the host and the server send it. The clause holds only of a calibrated hardware
card; no such card is deployed, and the software pacer that stands in for it does not meet it.

## The result

With `--out DIR` the runner also writes its result, through the harness kit's writer for `DIR` (`verity_bench.kit`)
and through nothing else. The kit says where the file is, `DIR/result.json` outside a research run
(`benchmarks/harness/README.md`). The result holds the amounts every resource's benchmark reports, for the row the
term of record is read from, the `scan` row at D_j = 2, at each scope and monitoring period the command prints. What
it prints doesn't change. Every measurement is the model's and its name says so; none is computed from a run's K.

- `certifier_term/modeled/metric`: the term of record, the document's `value`, in GB, against the document's target.
  Its verdict, `within target` or `over target`, is the document's multiple of the target in words. Its row is the
  scan's over the world's year, whichever rows the command prints.
- `certifier_term/modeled/simulated/metric`: the card's figure, the same row's term at D_j = 1, which the document
  gives as `with_set_time_card`, in GB, against the same target. It is no term of record. Its name says `simulated`,
  the card's mark, and its `inputs` open with the row it is of and the card as the document states it, whose source
  is the clause the figure rests on.
- `certifier_term/modeled/<scope>/<period>/free`: the term K, in bits. The check bounds the free amount from above,
  so this is the network's amount of record. It stands against the same target, in bits.
- `certifier_term/modeled/<scope>/<period>/capacity`: the bits the scope's certified boundary links carry out over
  the period, the scope's machines × a registered uplink's `beta` × the period's ticks. `beta` is bits a tick each
  way (`verity.device`) and the term is egress, so one direction counts. The uplink is read from a
  registration log, `--registration FILE` (`verity.protocols.accounting.registration.verifier.registration`'s committed log): the boundary link of its
  one host that a network certifier splits. The log is read as given, and one that registers anything but the
  model's machine, one host behind one network certifier, is refused. Every machine of the scope is taken as
  registered like that one, which is the model's statement and no registration's. With no `--registration` the
  capacity has no value and its `inputs` say why: it is read from a registered link or not at all, never copied
  from a data sheet, a schedule or the datacenter model.
- `certifier_term/modeled/<scope>/<period>/verified`: capacity less free, in bits, where both have a value. It
  isn't clamped, so it is negative where the term is more than the links carry.
- `certifier_term/modeled/<scope>/<period>/error`: the row's ε, a probability, which the calculator's document holds
  as `eps`. It is a union bound over the period's execution phases, their number × the scan's δ (The scans' δ,
  below).

A measurement's `inputs` name what it was computed from: its row, the document's inputs with their values, marks and
sources, the registration read, and the measurements it is a difference of. Every one names the class the term is at,
the most ids a response holds, and the scan's δ its error is at, as the document states them: terms at two classes
don't compare, and neither do errors at two δ.

The command of record passes its class, whatever the calculator's default is, and no δ:

```text
verity bench certifier_term --response-ids 1024 --out DIR
```

### The scans' δ

A row's ε is its execution phases' scans' δ summed, a union bound, so a δ that suits one execution phase bounds
nothing over the world's year. With no `--scan-delta`, each execution phase's scan is at the δ at which the row
that pools the most execution phases, the largest scope's longest period, fails with at most 2^-20 over its
monitoring period: that error divided by the row's execution phases. The calculator finds that row and counts its
execution phases from its inputs each time, so neither the row nor the δ is typed in, and both follow the inputs:
with today's it is the world's year. It is one δ for every row. That row's ε is then 2^-20 and every other row's is
less, in proportion to its execution phases: a datacenter's rows are stated at a δ stricter than they need, and two
rows' errors compare.

Every row of the document states the δ and the error it is taken from (`scan_delta`, `eps_target`) beside its `eps`,
and the document's `scan` says where the δ is from: whether it is given, the error, and the row whose execution
phases it is spread over. The δ is the input `scan_delta`, marked assumed whoever chose it, since no scan has run at
it. No term depends on it: only the errors move with it.

`--scan-delta D` gives another δ, as a fraction or a decimal. No error is then set (`eps_target` is null), a row's ε
is what `D` sums to, and a row whose ε is 1 or more has no bound at its period, since a union bound of 1 bounds
nothing. In the result its `free` and `verified` are written with no value and the reason, `no bound at this period`,
in their `inputs`, and so are `metric` and the card's figure where the row is the term of record's. Its `error` is
written with its value, and its `capacity`, which no error enters. What the command prints is the calculator's
still, those rows' terms among it.

The suite beside the runner (`tests/`) holds the result to the calculator's document and to the kit's schema. What
the command prints without `--out`, and the calculator's numbers, are held where the calculator's tests are,
`benchmarks/network_traces/tests/test_certifier_term.py`.

## The rows

A scope is a node (one machine), the reference datacenter or the world, each machine behind one network certifier.
Over a week, a month and a year it serves its GPU-seconds × the serving MFU × a GPU's peak FLOP/s / (a request's
tokens × 2 × the served MACs a token) requests, each on one connection. Which request that is is the row's request
count, which every row names (Where the scale comes from, below). Each scope and monitoring period has eight rows,
and each row says where it stands:

- `scan` (on main): the rows form, the scan on the network certifier's own run, as the check charges it today. A
  connection is charged for its schedule and again for its fail cell, one connection a request, so these two terms
  are counted by request. The jitter is counted by link, for each of its execution phases. Its accept is run, not
  proved, whatever its bound and its δ: it rests on verdicts the scan's driver hands the compiled record check, and
  nothing compiled ties them to the sessions' runs. So the row carries the mark `run, not proved`, the exfiltration
  command's mark of that name, in its legend line and in each of its rows of the document (`mark`; no other row
  carries one).
- `N2` (on main): the clock form. The schedules are counted by link, over its execution phases, whatever it serves.
  Its parameters list no connection, so it has no fail cell. The jitter is the scan's.
- `lengths`, `timing`, `entries` (each a change to the scan's check) and `all three`: the scan with a response's
  length held to its Program's output count, with the release times taken once by the compute side's tick, and with
  the entries charged through the compute audit. The schedules' 0 under `lengths` and `all three` rests on a stop
  block, a proof block that checks where a response's ids stop, sampled on a served audit: it holds that output count
  to the response's ids. Whether it does is read from a served run's verdict records, as the row `design`'s is (The
  stop class's verdicts, below): with no record read, or one verdict lacking, each connection is charged its
  schedule under these two rows too, as is each connection of an execution phase no registration was read for. The
  jitter's 0 under `timing` and `all three` rests on the set-time card, which is simulated.
- `design, no stop block` (no run behind it): the design, below, before a stop block has been sampled on a served
  audit. The lengths unit holds a response's length to its Program's output count, but until a stop block holds that
  count to the ids the server picks it, so each connection is still charged its schedule, as the scan charges it.
  Its failed connections' charge, its jitter, its count and its entries are the design's.
- `design` (no run behind it): head checks, a lengths unit on the network certifier's records, and a stop block
  sampled on a served audit. How its stop class is charged is read from a served run's verdict records and from
  nothing else (The stop class's verdicts, below). Charged below its schedule, the lengths are held, the stop block
  holding a response's output count to its ids, and its term is the jitter, the failed connections' charge and the
  count. With no record read, or one verdict lacking, each connection is charged its schedule and the row is the
  one before it; so is each connection of an execution phase no registration was read for. The entries are the
  compute audit's either way.

### The stop class's verdicts

The rows that hold the lengths by a stop block, `lengths`, `all three` and `design`, are charged below their
schedules only as the network certifier's own rule says (`verity.protocols.accounting.certifier.verifier.class_charge.charge`),
from the signed head of an execution phase and the verdict records on a class of it:

- a class that holds a row owes four verdicts, the stop record, the response tie, the certified read of the request's
  three words and the compute audit's second stage, each accepted, complete and naming that head's digest;
- with one missing, refused, not complete or decided under another head, another serve's among them, the class is
  charged its whole schedule on every row; a verdict without `complete` reads as not complete;
- so it is where the class's records on that head name two serves or two registrations: the head holds the network
  certifier's rows to a serve, and these two names hold the values the verdicts judge to it;
- so it is where the rule is not handed the schedule the head names, or that schedule can't be tied (each P listed
  once, one O for every class and a response limit): with no schedule handed that word was not asked, and the
  line says the schedule was not handed;
- a class with a count of 0 owes none.

`benchmarks/network_traces/stop_verdicts.py` reads the records and hands the calculator the answer (`of_runs`, `stop`).
The records are in the evidence store, in the run files of the runs that carry
`run_of_record=exfiltration-glm53-stop-verdicts`, each file one JSON object a line:

- `out/*-verdicts.jsonl`: a verdict record as the verdict's own verifier writes it, `run`, `serve` (the served run
  its class came from), `registration` (the digest of what that serve registered), `verifier` (its `module` and
  the `commit` it ran at), `verdict` and `proved_by`. A record that names no run, serve, registration or verifier
  doesn't read, and one that doesn't read holds no execution phase, in a run that no other replaces. Of its
  `verdict` the rule reads `link`, `window`, `class`, `head`, `accepted` and `complete`. Which of the four verdicts
  it is, is its verdict's `format`; a verdict whose verifier's format the reader doesn't list reads as missing. A
  stop record's and a response tie's record name two things more, `stream_registration` and `rate` ("The rate",
  below);
- `out/registrations.jsonl`: the registration of each execution phase's connection rows that the verdicts name by
  its head's digest, from which a class's count is read;
- `out/*-controls.jsonl`: a control record, `run`, `verifier` (its `module` and `commit`), `control` (what was
  changed), `verdict`, the verifier's own, whose `format` says which of the four it is a control of, and `reason`,
  the verifier's own word for why it refused. A control without `reason`, or a refused one whose reason is no
  text, doesn't read. Of a control record the reader reads these five fields and no other;
- `out/serve-registrations.jsonl`: each registration record the verdict records name by its digest: the one a serve
  registered its class under (`registration`), and the one of a verdict's own stream (`stream_registration`),
  which follows it and registers the rate that stream's population is sampled at.

The reader is handed no schedule and asks the rule without one. So as it stands it holds no execution phase,
whatever a run's files hold: each class that holds a row is charged its whole schedule, and its line and the
answer's first line say the schedule was not handed. The rest of this section is what the reader holds of a class
the rule charges below.

The reader holds what the rule charges below to two more things, which the rule doesn't read:

- **A negative control.** An execution phase is held only where each verifier that accepted has, at the same commit,
  a control of record that it refused: a run with the one thing changed that the verifier checks. The stop record's
  owes `stop_word` (a stop word changed, the count off by one), the response tie's `id` (a released id changed), the
  certified read's `max_tokens` (a request word changed) and the compute audit's `output` (a committed output
  changed). A control counts only where it is refused for the reason its change aims at: its `reason` is held to
  the control's own word, `stop_word`, `id`, `max_tokens` or `output`, which the reader declares
  (`stop_verdicts.REASONS`). One control a verifier a commit serves every execution phase under that commit. A
  control at another commit, by another verifier or of another thing serves none, and one the verifier did not
  refuse, or refused for another reason, leaves it without, whatever other it refused. So does one whose accept
  the verifier withheld, or whose command stopped: a control counts only where the verifier refused, and a verdict
  that states a `withheld` or a `stopped` word is no refusal, whatever `reason` its record gives. The line names
  the verifier that lacks its control, with the reason given beside the control's word; the class's line prints
  the reason each verifier gave.
- **The rate.** A class is held only where every block of the stop record's and the response tie's populations is
  proved. Each of the two verdicts runs over a stream of its own, and its record names that stream's registration
  record by digest (`stream_registration`). The reader is handed that record, holds it to follow the registration
  the class's verdicts name (its `previous` is that digest), and reads the rate it registers: K of its
  `population`, under a sampling design that is a `subset` of K, as `NUM/DEN` in lowest terms. The class is held
  only where that is `1/1` for both, and each of the two records says in `rate` that it ran at the registered
  rate. The rate is the registration's and never the verdict's own word, so a verdict that states `1/1` under a
  registration that says otherwise is refused by name. So is a record that names no registration of its stream, a
  registration that was not handed, one that follows another record, and one whose sampling design is no subset.
  The stop record's `population` is held to the head's count of the class: the stop record is of every row of the
  class and of no other, which `1/1` of a population of another size doesn't say.
  A failure is final for what its verdict speaks for, one execution phase's class at `1/1`; a rate below `1/1`
  widens the failure to the whole registered population, for when the reader takes one. The rate of the compute
  audit's second stage is not held here.

The reader is handed every run that carries the record, with those another run replaces. A superseded run adds no
accept and takes away no refusal: its records that are accepted and complete fill no verdict, its controls that
were refused count for nothing, and everything else it holds is read as any run's. An accept of its is still read
on its head: the serve and the registration it names, the rate it states and its verifier are held to what holds
any record's, so it can keep a class at its whole schedule and never helps hold one. A record is a superseded
run's where that run's files hold it or it names that run as its `run` or its `serve`.

The order of runs is read from their ids, which state the date and the time of a launch; a name that is no run id
is launched after nothing, and nothing after it. Two rules read it:

- a run has no accept on a serve launched after it: a verdict record that names such a serve and is accepted and
  complete fills no verdict, and what such a record refused stands. A record's own run is the one whose files
  hold it and the one it names as its `run`;
- a record of a superseded run that doesn't read could have been a refusal, so an execution phase is held only
  where its serve was launched after every superseded run that holds one: after the newest of them, which the
  line names with how many such records there are. It is one rule for every line of the run's four kinds of file
  that doesn't read, a line that is no JSON among them. A bad file is mended by a new serve with new verifier
  runs, never by a second run over the old serve; a control record that doesn't read is mended, for the serves
  launched after its run, by a new control run.

So superseding a run raises what is held only on a serve launched after that run, and only where the run holds a
record that doesn't read. In a run that no other replaces, a record that doesn't read holds no execution phase at
all, and a line that is no JSON is named by its line, its file and its run. The ids are the tooling's, which is
enough for the runs of record here; in a deployment the order comes from the auditor's own ledger, from something
the prover cannot write.

A verdict that doesn't count, refused or not complete, counts against the execution phase it names by its serve,
its link and its number, whatever head it names: refused under one head, registered again under another and
accepted would be a second try at a verdict. It keeps every class of that phase at its whole schedule, since two
heads need not sort a phase's rows into the same classes; under the head read a verdict stays its class's. An
accept under another head counts for nothing.

A verdict states two fields more, and both names are its verifier's own: `withheld`, null or the one word for
what the verifier held its accept back for (today `two-stage-form`, which only the compute audit's second stage
states), and `codes`, a list of the verifier's own code words, empty on an accept. The charge reads neither: a
verdict counts only where `accepted` is exactly true, and a withheld one leaves its class at its whole schedule
as a refused one does, under any head. A line says "withheld" with the word of a verdict that is not accepted and
states one, and "refused" where `withheld` is null; `codes` is handed on as written, beside the six fields. Every
verdict states both, whatever its format, a control's among them: a verifier with nothing to say writes `withheld`
null and `codes` empty, so no withheld verdict reads as refused because a line was left out. A record doesn't read
where its verdict leaves either out, and the line says which field to add; where its `withheld` is neither null
nor a word; where its `codes` is no list of words; and where its verdict is accepted and states a word or a code.

A verifier's command that stopped with an error after its stream's session opened reached no verdict, and it was
a try at one. Its record states `accepted` false, `withheld` null, `codes` empty and one field more, `stopped`, the
command's own word for the error. The line says "stopped" with the word, the charge is a refusal's, and it is final
as a refusal is, under any head and in a superseded run: a second run over the same serve fills nothing. A command
that stopped before any challenge was sent writes no record. `stopped` may be absent. A record doesn't read where
its verdict's `stopped` is neither null nor a word, where an accept states one, and where a verdict states a
`withheld` word or a code too. What this does not close: a command that is killed writes nothing, and a second
run then fills its verdict unseen. Only a record of the sessions opened, which the command does not write itself,
closes that.

An execution phase is one serve's: a serve, a link and the phase's number. A registration of connection rows names
no serve, so a head's serve is the one its verdict records name, which is a run's word and no proof; the answer's
second line says so. Two heads given for one number of a link are two execution phases only where the records of
each name a serve and none is named for both. Where one is named for both, where no record names a serve for one of
them, or where a table of one name has one domain in both (a domain binds the network certifier's session, the
link, the number and the table's rows), they are one execution phase given two heads, which holds no execution
phase at all. A head whose records name two serves is charged its whole schedule.

The four verdicts count only as one class's: of one class, in one execution phase of a link, on one signed head,
and of one serve under one registration, as their records name the two. Where each of the four was read and no
class that holds a row has them so, the answer holds nothing and its first line reads `not one class: unchecked`,
with what the four differ in: the class, the execution phase, the head, the serve or the registration.

An execution phase is held where its head was read, some class of it holds a row and every class that holds one has
its four verdicts, of the one serve its head's records name, at the registered rate, each by a verifier with its
control, with no verdict of that serve refused or not complete under another head, and of a serve launched after
each superseded run that holds a record that doesn't read. The answer is how many execution
phases of a link are held (the document's `stop`'s
`held_phases`), and a row is charged by it:

- a row's execution phases are its scope's over its monitoring period, never the ones a run kept a registration for;
- the held ones' share of the row's connections is charged below its schedule, at most all of them;
- every other execution phase is charged each connection's whole schedule, one no registration was read for among
  them. So a run's files with an execution phase left out charge what they would with that execution phase refused.

The calculator takes the answer as an argument (`document`'s and `table`'s `stop`). The command has no option for it
and no input holds it, and it reads no store itself: run alone it reads no record, charges the rows their whole
schedules and says so. A caller that reaches the store names the runs and passes the answer.

The lines print how the rows are charged, one line for each class of each head read, and each verdict read under the
mark `run, not proved`: a verifier judged it on a run, and no proof of that verifier is read here, whatever the
record's `proved_by` says. A head is read as its registration lists it, held to the digest its certificate names; its
signature is not checked here. Nothing but verdicts that name a head's digest holds an execution phase, so a
registration nobody signed holds none.

Where some execution phase is charged its whole schedule, the document holds apart what each row would be with every
one held: its `stop`'s `with_verdicts`, in a row's own fields, and one line over the world's year. No record read
supports those figures, and no row, sum or measurement holds them. The `stop`'s `below` is whether every execution
phase of every scope and monitoring period is held, where nothing is held apart.

Nothing a check counts by capacity is scaled by requests: twice the requests are twice the scan's schedules and fail
cells, and nothing else. The count is by capacity too: with the lanes given it does not move with the requests, and
with none given it moves only as the least lanes that serve them do.

## The terms

- **schedules**: for the scan, ⌈log2(cells + 1)⌉ bits a connection, where cells is the most cells a response takes,
  one more than the most ids it holds; 0 where the lengths are held with a stop block (`lengths`, `all three` and
  `design`, on the execution phases whose verdict records charge the stop class below its schedule), and the scan's
  where they are held without one (`design, no stop block`, and those three rows on every other execution phase). For N2, ⌈log2 #Σ_sync^m⌉ over a link's m execution
  phases (`--syncs`).
- **cells**: the fail cells, the same bits again a connection. Where the lengths are held, with a stop block or
  without, it is the failed connections' charge instead, pooled over the period: which connections failed and where
  each one's stream ended, at most F·(log2(e·N / F) + log2(cells + 1)) bits for F failures among N connections, and
  0 with none (`--failure-rate`, the share that fail, which no run has measured).
- **jitter**: ⌈log2 D_j^T⌉ bits for each execution phase of T buckets on each link. At D_j = 2, two release
  schedules, it is a bit a bucket, and that is the jitter of record. At D_j = 1 it is 0, which is what a network
  certifier with a set-time card would be charged: it is printed as "with a simulated set-time card". Where
  the timing is taken once it is 0 at either D_j, and that 0 rests on the same card, so those rows carry the label
  in their name.
- **count**: on the design's two rows, classes · log2(R + 1) bits for each execution phase on each link, whatever
  it serves, and 0 on every other row. Under the design the network certifier signs a head over each execution
  phase's tables, and a content class's tables are the rows it holds and no other, so the head states how many of
  the execution phase's releases each class holds: one of R + 1 values a class.
  - R is the most connections the schedule releases in an execution phase of T buckets: lanes · ⌈(T + ingress
    buckets − 1) / (D + 1)⌉, D being the delay. It is a bound: at some schedules no load releases as many.
  - The classes, the delay and the ingress buckets are inputs (`--classes`, `--delay`, `--ingress-buckets`), by
    default those of the lengths rows' cost runner (`benchmarks/network_traces/connection_rows_run.py cost`): six
    classes and a delay of 30 buckets, which its ingress buckets equal.
  - The lanes are the input the entries read (`--lanes`). With none given the schedule is not known, so the count
    is at the least lanes whose schedule releases what a link serves in an execution phase: any schedule that
    serves the load releases at least as many, so it is the least the rule can charge. With lanes given the count
    is at them. Lanes whose schedule releases fewer than a link serves in an execution phase are refused, with both
    numbers and the least lanes that do: the rows would count a schedule that does not carry the connections they
    charge.
  - The rows `lengths` and `all three` are charged no count, and each says so in its own line: its tables are of
    the lengths rows' form 1, the schedule's rows whatever the load, so the head states none.
  - The count's line is the count alone. The head's time is not held to a tick: nothing holds when the head of a
    link's execution phase leaves, so the line says so, and prints no charge for a head that leaves late, of 0 or
    of anything else.
- **term**: the four summed, the network certifier's own term, at each D_j. The term of record is the one at
  D_j = 2. The one at D_j = 1 is printed beside it only as "with a simulated set-time card", never as the
  term.
- **entries**: the slot entries, which are the compute side's and not part of the term. The scan's check charges
  every slot's entry at its whole width, a count by capacity: with `--lanes N`, the connections a link holds at once,
  it is printed so. Without it the column is one entry a request, the least the check can charge. Where the compute
  audit charges the entries, its own bound stands in their place, and this command doesn't count it.

## The tables

- **Inputs**: each input's value, unit, mark and source. A mark is `measured` (a run measured it), `modeled`
  (derived from other figures), `assumed` or `simulated` (a stand-in runs in place of a device that is not deployed,
  and the figure is what that device would be charged: the set-time card, `set_time_card`, alone); the source is the
  file and key it is read from, the parameter that gave it, or for the card the clause its figures rest on. No run
  has certified a class for GLM-5.3, so the class is assumed: the most ids a response holds
  (`--response-ids`), an id's bits (`--id-bits`), an execution phase's buckets and their seconds
  (`--phase-buckets`, `--bucket-s`), and the schedule the count is at (`--lanes`, `--classes`, `--delay`,
  `--ingress-buckets`).
- **The scopes**: the compute each row covers, which is all its scope serves: machines, GPU-seconds, requests and
  execution phases, the δ each execution phase's scan is at with the error it is taken from, and ε, those δ summed
  (The scans' δ, above). ε is a union bound, so at 1 or more it bounds nothing.
- **request count**: a column of every table, and `request_count` on every row of the document and at its top:
  `plan`, `census` or `given`, the request the row's requests are counted of. One table holds one request count;
  rows of two don't compare.
- **The terms in GB**: schedules, cells, the count, the jitter at each D_j, and the term at each D_j with its
  multiple of the target (`x target`, `--target-gb`) and of one copy of the weights in FP4 (`x weights`); then the
  entries. The columns at D_j = 2, the term of record, come before those at D_j = 1. Every column at D_j = 1, the
  multiples too, has the label in its header, as `simulated set-time card`, and so has the name of a row whose
  release times are taken once, in the `row` column: `timing, simulated set-time card`.
- **The terms in bits**: the same pooled totals in bits, beside the term for one execution phase of one link
  (`a link's phase`) and for one request (`a request`), the term as a share of the compute it covers.

The lines above the tables give the rule for the term of record, that the card is simulated, and the rule for the
rows that carry the label, each row's rule, one link's execution phase (its requests, a connection's bits, N2's
schedules, the jitter at each D_j, a slot entry's bits), the count (its rows, its bits a link's execution phase, R
and the schedule R is of: its lanes, given or the least, and its classes, delay and ingress buckets, the cost
runner's or as given; and the same rule at the cost runner's own schedule, 16 lanes and an execution phase of 600
buckets, which is that runner's and not these rows'), and the target as bits a link's execution phase and a request
over the world's year. Then they say
what the design's term over the world's year is with the release times taken once by a network certifier whose
network card releases at a set time (simulated), the failed connections' charge and the count: the largest failure
rate at which it meets the target, the document's `failure_rate_at_target`, or, where the count alone is over the
target, that no failure rate meets it, and the document's rate is null. Last they give what the jitter alone is
without that card, at D_j = 2 over the world's year, in GB and as a multiple of the target.

The document's `count` holds how the count is counted: its rows, the schedule (`classes`, `lanes` and whether they
were given, `delay`, `ingress_buckets`), R as `releases`, the bits a class and a link's execution phase, its charge
over the world's year beside the target, and the cost runner's own schedule with its R and bits (`cost_runner`).
Each row holds its own `count`, in bits and GB.

## Where the scale comes from

- The world is rho's world fleet (`benchmarks/rho/data/reference.toml`, `fleet.world-h200`): the census's inference
  (`benchmarks/census/adversary.json`) on eight-GPU machines at the fleet's serving MFU. The datacenter is the census's
  reference datacenter (`benchmarks/census/deployments.json`).
- The requests are counted of one request, by the rule `benchmarks/spot_figures/figure1.py` plans with, at the
  H200's dense FP8 peak (the H100 SXM5's entry of `benchmarks/census/hardware.json`, one die and one clock):
  - `plan`, the default: the sampling plan's request, so that the rows count the requests the compute audit's plan
    prices. The world's requests a year, a request's tokens and the served MACs a token are read from the break-even
    planner (`benchmarks/spot_figures/breakeven/`, `sweep.build` on the case `leak_bounds.PLAN_CASE`), which prices
    the 128k serving plan's request: fewer tokens than the census's, at more work a token.
  - `census` (`--requests census`): the census's tokens a request at the served MACs a token of GLM-5.3's workload
    card (`benchmarks/spot_figures/cards/workload.glm-5.3.json`).
  - `given` (`--world-requests N`): N requests a world-year, of no stated request, in place of either. Every scope
    scales with it.
- The weights are the workload card's parameters at FP4's 4 bits each.
- A bucket is the network certifier's default, and N2's #Σ_sync the most a calibration chooses by default
  (`verity/protocols/accounting/certifier/verifier/calibration.py`).

The scaling by links (`leak_bounds.at_link_rate`, which `chain_record.py leak` reads on a run) is a different count:
every link at the rate one run certified, whatever its scope serves.
