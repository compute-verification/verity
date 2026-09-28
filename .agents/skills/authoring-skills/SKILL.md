---
name: authoring-skills
description: >
  Conventions for creating or modifying agent skills in .agents/skills/:
  SKILL.md structure, frontmatter, writing descriptions that activate
  reliably, naming, where detail beyond SKILL.md goes, and routing from
  AGENTS.md. Use
  when adding a skill, editing an existing SKILL.md, or deciding whether
  guidance belongs in a skill, in AGENTS.md, or in code.
---

# Authoring Skills

Use this skill when creating or modifying agent skills in `.agents/skills/`.

## When to Create a Skill

Create a skill when the guidance is:

- too detailed for `AGENTS.md`, such as a multi-step workflow, diagnostic
  procedure, or reusable implementation pattern;
- relevant only to a particular class of work rather than every agent
  session;
- coherent enough to load and follow on its own.

Keep guidance in `AGENTS.md` when it is:

- a short rule or invariant that agents should always have in context;
- a broadly applicable gotcha that agents working in that scope are likely
  to encounter.

Prefer code, tests, schemas, or linting when the requirement can be enforced
mechanically.

## Where a Skill Lives

Put a skill at the narrowest scope to which it genuinely applies.

- Applies only to one package or subsystem: `<package>/.agents/skills/`,
  routed from that package's `AGENTS.md`.
- Applies to work anywhere in the repository: `.agents/skills/` at the
  root, routed from the root `AGENTS.md`.

Cursor scopes a nested `.agents/skills/` directory to the subtree it sits
in, so package-local skills need no path configuration.

For example, `research` (how an agent anywhere in Verity runs, inspects, and
preserves substantial computational work) is repo-level. `proof-benchmark`
(what counts as a valid, comparable proof experiment) belongs under
`verity/ml/flock/`, because only proof-system work needs it.

## File Structure

~~~text
.agents/skills/
└── my-skill/
    └── SKILL.md        # Required: frontmatter + workflow
~~~

A skill is one `SKILL.md`, at most 48 KB. `tests/test_repository.py` admits
tracked markdown only under the maintained basenames (`AGENTS.md`, "Where
writing goes"), so a `workflow.md` or `examples.md` beside it fails the suite.
A script or a `*.template.md` the skill uses may sit beside it.

## Frontmatter

A skill requires at least:

~~~yaml
---
name: my-skill
description: >
  What the skill covers and when it should be used.
---
~~~

Useful optional fields supported by some agent runtimes include:

~~~yaml
argument-hint: '<argument>'
user-invocable: false
disable-model-invocation: true
allowed-tools: [Bash, Read]
model: opus
context: fork
agent: Explore
~~~

Use optional runtime-specific fields only when there is a concrete reason.
Keep ordinary Verity skills portable when possible.

The skill name should match its directory name.

## Writing Descriptions

The description is the primary surface agents use to decide whether a skill
is relevant.

Include:

1. what the skill covers;
2. when it should be used;
3. important file names, commands, or concepts that help match relevant
   work;
4. terms an engineer or agent is likely to use when asking for the task.

Avoid descriptions that merely restate the skill name.

For example:

~~~yaml
# Too vague
description: Helps with proof benchmarks.

# Better
description: >
  Runs and compares Verity proof-backend benchmarks for exact relations.
  Use when benchmarking SP1 or another proof backend, changing prover
  performance, comparing authenticated versus relation-only costs, or
  editing the proof benchmark harness.
~~~

## Content Conventions

### Structure for Action

A skill should tell the agent how to perform the work, not merely provide
background.

Prefer:

- a clear workflow;
- concrete defaults;
- commands or templates that can be adapted directly;
- important edge cases and gotchas;
- explicit validation at the end.

If the skill becomes large, keep the workflow in `SKILL.md` and point at the
detail where it is maintained anyway (see Detail Beyond SKILL.md).

### Relationship to AGENTS.md

`AGENTS.md` is always-loaded guidance. Skills are task-specific guidance
loaded when needed.

| `AGENTS.md`                        | Skills                                  |
|------------------------------------|-----------------------------------------|
| Short invariants and guardrails    | Detailed procedures                     |
| Repository or package-wide gotchas | Task-specific patterns and edge cases   |
| Pointers to relevant skills        | The workflow itself                     |

When adding a skill that agents working in a particular scope should know
exists, add a short reference to it in the relevant `AGENTS.md`.

The `AGENTS.md` entry should route to the skill, not duplicate it.

## Naming

Use names that are:

- short;
- descriptive;
- scoped to the task or concept;
- lowercase and hyphenated when multiple words are needed.

Examples:

~~~text
proof-benchmark
memory-diagnosis
vllm-capture
~~~

Do not add a `verity-` prefix; the repository already provides that scope.

## Detail Beyond SKILL.md

Link to detail at the point where it becomes useful, in the place that owns
it, rather than copying it:

- a tool's `README.md` or `--help` for its commands and flags;
- a check whose failure message says how to fix the failure;
- a `kb/` doc in the notes (`~/.research/notes/kb/`) for facts that change
  often and that anyone corrects in place, such as a toolchain's gotchas.

A skill whose parts serve different kinds of work is two skills.

Do not create supporting documents merely to record how the skill was
designed.
