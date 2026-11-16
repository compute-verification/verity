---
name: writing-readmes
description: >
  How to write or rewrite a README in Verity. A README opens with what its directory is and how to think of it, gives
  one thing to try, and then holds only what a reader needs: no list of the code's files, no account of how the code
  works, no closing list of links. Covers where the rest goes (`DESIGN.md`, docstrings, `--help`), the four READMEs
  that are the examples, how much latitude you have, and the checks to run. Use when adding a README, rewriting,
  shortening or reviewing one, moving detail out of one into a `DESIGN.md`, or when a README reads like a description
  of the code.
---

# Writing READMEs

Every README in the repository is written in one style. It is a style and not a template: keep what it is for, and
change the sections wherever a directory gives you a reason to.

## The examples

Read two of these before you write. They are the style, and the rest of this file only says why they read as they do.

| README | What it shows |
| --- | --- |
| `demos/README.md` | A short one: what a demo is, one table by protocol name, one command. |
| `verity/README.md` | A library: each part in a sentence, a first example, what each protocol tells the auditor. |
| `adapters/vllm/README.md` | An adapter: what it does to a serving stack, a quickstart, usage. Its detail is in the `DESIGN.md` beside it. |
| `benchmarks/README.md` | An index: what a benchmark is, the few that matter, one to try, then a line for each. |

## What a reader gets, in this order

1. **What the thing is.** Open with how to think of it, in words a newcomer follows: "A demo is Verity run end to end
   on real machines." No file, flag or theorem comes before that. If readers tend to take it for something else, say
   what it is not.
2. **One thing to try.** A command or a short example that runs, what it needs (a GPU, a key, nothing), and what it
   prints. Leave it out where the directory has nothing a reader would run.
3. **What a reader needs after that, and no more.** Usually one table of what the directory holds (its protocols, its
   benchmarks, its parts) by name, with a clause each. Say plainly what does not work yet.

## What stays out

- **How the code works.** Which file calls which, a list of modules or functions, the steps a script takes: the
  docstrings say that. Name a file only where the reader has to open it or run it.
- **Every option.** A tool's flags are in its `--help`, and a class's options are in its docstring.
- **A closing list of links.** No "Read next" or "See also". Link a thing in the sentence that raises it.
- **A guarantee in other words.** Link to the protocol's README or the Lean statement that states it. A rewrite does
  not reword what a guarantee says or rests on.
- **Results, history and process**, as `AGENTS.md`'s "Where writing goes" says.

Detail that only the README holds, and that someone working on the code needs, moves unchanged into a `DESIGN.md`
beside it, with one sentence in the README saying what is there. Delete it only where a docstring or `--help` already
says it.

## Style

- Sentences and short paragraphs. A table holds short facts, never paragraphs.
- Code formatting is for what a reader types or imports. Use it sparingly: a protocol or a benchmark goes by its name
  (SPOT, PoUW, the compute benchmark), in `GLOSSARY.md`'s terms.
- Give few commands, each the one a reader wants first, and only ones you ran on the tree you commit.
- Write the opening fresh, from what the thing is. An opening edited down from the old README keeps the old README's
  shape.

## Latitude

The order above is the default and not a list of headings. A directory may need a section these examples lack, or
nothing beyond its opening. A test holds some headings (a core benchmark's three interface headings, the benchmarks
index's lines): keep those, and write the rest in this style around them. Where the style and the directory disagree,
do what serves the directory's reader.

## Before you push

- Run each command and example the README gives.
- Run the documentation tests: `uv run pytest tests/test_doc_paths.py tests/test_documented_commands.py
  tests/test_benchmarks_index.py tests/test_tree.py tests/test_repository.py`.
- `git grep` the headings you removed or renamed. Docstrings and tests cite a README's sections by heading, and a test
  that held some text to the README now holds it to the place the text moved to.
