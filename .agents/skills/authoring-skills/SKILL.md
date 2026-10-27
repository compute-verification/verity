---
name: authoring-skills
description: >
  Conventions for creating or modifying agent skills in .agents/skills/:
  SKILL.md structure, frontmatter, writing descriptions that activate
  reliably, naming, where detail beyond SKILL.md goes, the 16 KB cap and
  compacting a skill to current rules without their history, and routing from
  AGENTS.md. Use when adding a skill, editing an existing SKILL.md, a skill is
  over its cap or too long, or deciding whether guidance belongs in a skill,
  in AGENTS.md, or in code.
---

# Authoring Skills

## When and where

A skill holds guidance too detailed for `AGENTS.md` (a multi-step workflow, a diagnostic procedure, a reusable pattern),
needed by one class of work rather than every session, and coherent enough to follow alone. A short invariant every agent
should have in context, or a gotcha most agents in its scope will meet, stays in `AGENTS.md`; a requirement that can be
enforced mechanically is code, a test, a schema or a lint.

Put it at the narrowest scope it applies to: work in one package, `<package>/.agents/skills/`, routed from that package's
`AGENTS.md` (Cursor scopes a nested `.agents/skills/` to its subtree, with no configuration); work anywhere, the root
`.agents/skills/`, routed from the root `AGENTS.md` (running research jobs is repo-level; proof benchmarks belong under
`verity/ml/flock/`).

## The file

A skill is one `SKILL.md` in a directory of its name, at most 16 KB (`SKILL_SIZE_CAP` in `tests/test_repository.py`; an
`OVERSIZE_SKILLS` entry there only shrinks). That test admits tracked Markdown only under the maintained basenames
(AGENTS.md "What the tree holds"), so a `workflow.md` or `examples.md` beside it fails; a script or `*.template.md` it uses
may sit there.

Its frontmatter needs `name` (the directory's) and `description`. Runtime-specific fields (`argument-hint`,
`user-invocable`, `disable-model-invocation`, `allowed-tools`, `model`, `context`, `agent`) go in only for a concrete
reason: keep skills portable.

**The description decides when the skill activates.** Say what it covers, when to use it, the files, commands and concepts
that match the work, and the words someone asking for the task would use; never just restate the name. "Helps with proof
benchmarks" is too vague; "Runs and compares Verity proof-backend benchmarks for exact relations. Use when benchmarking
SP1 or another proof backend, changing prover performance, ..." activates.

Name it short, descriptive, scoped to the task, lowercase and hyphenated (`proof-benchmark`, `memory-diagnosis`), with no
`verity-` prefix.

## The content

Tell the agent how to do the work: a workflow, concrete defaults, commands or templates it can adapt, the edge cases and
gotchas, and a validation step at the end. An agent loads a skill whole, so every line costs context; state the procedure
and rules as they are now:

- a ruling is one present-tense line for its class of case; its date, source and story go in Daniel's decisions log in
  Notion, or a note;
- a rule `AGENTS.md` or another skill states is cited, not repeated;
- a rule code enforces is deleted, since the check's failure message says it;
- a superseded rule is rewritten in place, never kept beside its replacement.

When an edit would push a skill past its cap, compact it by these rules in the same PR.

Link detail where it becomes useful, in the place that owns it, rather than copying it: a tool's `README.md` or `--help`,
a check's failure message, or a `kb/` doc in the notes (`~/.research/notes/kb/`) for facts that change often and anyone
corrects in place. A skill whose parts serve different kinds of work is two skills. Write no document that only records
how a skill was designed.

## Routing from AGENTS.md

`AGENTS.md` is always loaded: short invariants, gotchas and pointers. A skill is loaded when needed: the procedure,
patterns and edge cases. A skill agents in some scope should know about gets a short entry in that scope's `AGENTS.md`,
which routes to it and never duplicates it. The root `AGENTS.md` is the public guide to the code: a skill about how the
team works (landing, machines, lanes, Slack, Linear, notes) is routed from the `team` skill instead.
