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

Daniel sees the work only through what agents tell him, so he can't ask about a shortcut he doesn't know exists. A smell
check is how agents find these first. Daniel asks every agent for one from time to time, and each one reports what it
finds and records it in the catalogue. Run one too whenever a claim you're about to rely on or pass on carries one of the
words below.

## Run one

1. **List the claims.** Write down what you were told (your brief, other lanes' posts, PR titles and bodies, labels) and
   what you told others (reports, cards, notes, PR titles, labels), and mark every claim that uses one of these words:
   *served, end to end, measured, proved, guarantees, exact, bit-exact, credited, accepted, PASS, committed,
   zero-knowledge, hiding, live, enforces, landed, works, pinned*.
2. **Read what backs each one, not its summary.** For code, the path that actually runs on the served or proved path
   (follow the call from the entry point; a docstring says what was meant). For a number, the run's own outputs and
   labels. For a theorem, its statement and the definitions it reads. For "landed", `main`. For a rule someone relays,
   grep the code for the names it cites. Ask each question below.
3. **Ask the one question that matters:** would Daniel, reading the code and the run rather than the summary, recognize
   it as what he asked for? If you'd have to explain a "but", it's a smell.
4. **Report every smell you find,** yours or another lane's, in the format below. A smell in another lane's work goes to
   that lane's owner too. Report "none found" with the list of what you read, so the asker can see what wasn't.
5. **Act on it.** Fix your own smells without asking (the friction skill). Send another owner's to that owner. One that
   changes what a claim says or rests on goes to Daniel (`.agents/skills/asking-daniel/SKILL.md`). Relabel what's
   already published (`research data label`), and correct anything you told Daniel in the same report.
6. **Record it** in the catalogue, `~/.research/notes/kb/smell-catalogue.md`, under its question, newest first, and
   publish with `research notes sync --path kb`. The catalogue is a notes doc that anyone corrects in place; this skill
   holds only the procedure.

The report:

~~~text
SMELL <one line: what it looks like versus what it is>
where:  <file and line, PR, run id or art id>
quote:  "<the words that hid it, verbatim>"
bends:  <Daniel's intent or the AGENTS.md design rule it bends>
costs:  <what it changes: a claim, a number someone cites, compute, a decision already made on it>
fix:    <the fix and its owner; started or not>
~~~

## What to ask

Each question heads a section of the catalogue. The catalogue's entries are the examples; read the section for a
question when you aren't sure whether something counts.

1. **Is what's credited tied to what was served?** PoUW credits noised work on the operands of a served product that is
   committed and checked by value (ASK-71). A computation beside the served one that nothing ties to the served bytes,
   stand-in values, a reference pass, served bytes nobody commits, or a claim narrowed until the bound holds isn't that.
2. **Does "served", "end to end" or "measured" mean the real path?** Pinned build, compiled, the record's settings, the
   real workload and values, interference included, the verifier of record, the critical path measured; not a harness,
   a model, a label, Python standing in for Lean, an architecture argument, or a number nobody measured for this case.
3. **Is the rule-following path the record, or opt-in?** One hash, hiding leaves, live coins from a challenger off the
   prover's node, everything committed at serve time, verifiers of record in Lean, Boolean circuits.
4. **Is anything public that should be a value?** A partition, a count, a salt or a field that follows the workload.
5. **Do the numbers compare?** Same baseline, metric, model, card, build and scheme; the scheme named in the label; the
   best baseline, not one slowed by a capped batch, a frozen tile, an unfused kernel or a slower backend; decode and
   prefill reported apart; a model's number labeled as one; a spec that reproduces it.
6. **Is anything done or kept twice?** Two GEMMs for one linear, two trees over one value with nothing tying them, two
   implementations of one format.
7. **Does each check fail when it should?** A negative refused for an unrelated reason, a gate that can't fail, a test
   that reads only its own fixture or never ran where it matters, a comparison of a thing with itself, a token gate
   across processes whose kernels picked different configs.
8. **Does the theorem or bound say what it reads as?** A definition that makes it vacuous, a premise known false or
   holding only for an honest machine, the honest prover's speed where the fastest adversary's is needed, a doc that
   describes another version of the code.
9. **Is anything deferred, or only marked, but stated as done?** Queued isn't landed; "will" isn't "does"; wired isn't
   run.
10. **Did you relay it without reading it?** A claim from another lane's summary is a quote, not a fact, until you've
    read what backs it, and a hypothesis goes out marked as one.
11. **Does an action or a default reach beyond what was granted?** An automated action on things it wasn't given, or a
    default that turns silence into a grant.
12. **Is there code on the soundness path that nothing runs?**
