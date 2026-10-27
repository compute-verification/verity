---
name: smell-check
description: >
  How to run a smell check in Verity: find work that reads like what Daniel asked for but isn't, before anyone relies on
  it. Credit for a computation nothing ties to the served bytes, a "served" or "end to end" number from a model, a
  harness or stand-in values, a rule-following path that is opt-in, something public that should be private, numbers
  that don't compare, one value done twice, a check that passes for the wrong reason, a theorem or bound that says less
  than it reads, something deferred or only marked but stated as done, a claim relayed unread, an action beyond what was
  granted. Twelve questions, a report format, and where the catalogue of past smells lives. Use when Daniel or a lead
  asks for a smell check, before you call something served, end to end, proved, exact, measured, credited or landed,
  when you relay another lane's claim, when a result looks too good, when you write or review a brief, or when you
  found a smell and want to record it.
---

# Smell check

Daniel sees the work only through what agents tell him, so he can't ask about a shortcut he doesn't know exists; a smell
check finds it first. He asks every agent for one now and then; run one too before relying on or passing on a claim
that uses a word below.

## Run one

1. **List the claims** you were told and told others (briefs, posts, reports, cards, notes, PRs, labels), marking each
   that says *served, end to end, measured, proved, guarantees, exact, bit-exact, credited, accepted, PASS, committed,
   zero-knowledge, hiding, live, enforces, landed, works* or *pinned*.
2. **Read what backs each, not its summary:** the code that runs (followed from the entry point; a docstring says what
   was meant), the run's own outputs and labels, a theorem's statement and the definitions it reads, `main` for
   "landed", the code for the names a relayed rule cites. Ask the questions below, and the one that matters: would
   Daniel, reading that rather than the summary, recognize it as what he asked for? A "but" you'd have to explain is a
   smell.
3. **Report every smell**, yours or another lane's (to its owner too), as below; or "none found" with what you read.
4. **Act:** fix your own (the friction skill), send another's to its owner, take one that changes what a claim says or
   rests on to Daniel (the asking-daniel skill), relabel what's published (`research data label`), and correct in the
   same report anything you told Daniel.
5. **Record it** under its question in the catalogue, `~/.research/notes/kb/smell-catalogue.md`, newest first, and
   publish with `research notes sync --path kb`. The catalogue holds the examples; anyone corrects it in place.

~~~text
SMELL <one line: what it looks like versus what it is>
where:  <file and line, PR, run id or art id>
quote:  "<the words that hid it, verbatim>"
bends:  <Daniel's intent or the AGENTS.md design rule it bends>
costs:  <what it changes: a claim, a number someone cites, compute, a decision already made on it>
fix:    <the fix and its owner; started or not>
~~~

## What to ask

Each question heads a catalogue section, whose entries are its examples.

1. **Is what's credited tied to what was served?** PoUW credits noised work on a served product's operands, committed
   and checked by value; not work beside it, stand-ins, a reference pass, uncommitted bytes, or a claim narrowed to fit.
2. **Does "served", "end to end" or "measured" mean the real path?** The pinned compiled build at the record's settings,
   the real workload and values, interference included, the verifier, the critical path; not a harness, model,
   label, Python standing in for Lean, argument, or number nobody measured for this case.
3. **Is the path that follows the design rules (AGENTS.md) the record, or opt-in?**
4. **Is anything public that should be a value?** A partition, count, salt or field that follows the workload.
5. **Do the numbers compare?** Same baseline, metric, model, card, build and scheme (named in the label); the best
   baseline, not a slowed one; decode and prefill apart; a model's number labeled as one; a spec that reproduces it.
6. **Is anything done or kept twice?** Two GEMMs for one linear, two untied trees over one value, two implementations of
   one format.
7. **Does each check fail when it should?** Not a negative refused for an unrelated reason, a gate that can't fail, a
   test that reads only its own fixture or never ran where it matters, a thing compared with itself.
8. **Does the theorem or bound say what it reads as?** No definition that makes it vacuous, premise known false or true
   only of an honest machine, honest speed where the fastest adversary's is needed, or doc of another version.
9. **Is anything deferred, or only marked, but stated as done?** Queued isn't landed, "will" isn't "does", wired isn't run.
10. **Did you relay it without reading it?** Another lane's summary is a quote until you've read what backs it; a
    hypothesis goes out marked as one.
11. **Does an action or a default reach beyond what was granted?** An automated action on what it wasn't given, or a
    default that turns silence into a grant.
12. **Is there code on the soundness path that nothing runs?**
