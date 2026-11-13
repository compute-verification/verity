Written Fri 04:08 AM PST [12:08Z]. CATALOGHAND's return.

Both PRs carry #9364 and are pushed. Neither is marked, and neither has a label. #9659 is at `3d484ece0953`, #9660 at `1333bd53f693`.

**Carrying #9364.**
- #9364 stopped #9659 in 16 files. Every stop was an import block #9364 sorted, or a call it rewrote, on a line #9659 changes. Each now takes #9364's side with #9659's edit on it. The list is in #9659's body.
- #9659 into #9660 was clean.
- No new caller of the lazy `specialization` came in with #9364.

**Runs at the new heads** (each `--on auto --cwd source`; a peak is an upper bound on anonymous memory):

| run | head | what | result | declared | peak |
|---|---|---|---|---|---|
| `560e` | #9659 | `verity-vllm` | 17,864 passed, 565 skipped | 96 GB | 74.7 GiB |
| `ffde` | #9659 | five light suites | all passed | 32 GB | 22.4 GiB |
| `667c` | #9659 | circuit-check's suite | 448 passed, 5 skipped | 48 GB | 15.7 GiB |
| `f36b` | #9660 | `repository`, `verity/definitions` | 359 passed; 1,137 passed, 3 skipped | 32 GB | 38.9 GiB |
| `d445` | #9660 | circuit-check's suite | 460 passed, 5 skipped | 48 GB | 34.4 GiB |
| `8212` | #9660 | `--bindings` | 288 bindings, 24,872 targets | 48 GB | 33.8 GiB |
| `d564` | `main` `8854b824` | `--bindings` | the same list | 48 GB | 33.5 GiB |

`f36b`'s PSS floor is 18.7 GiB.

**The catalog.**
- What the checker finds is unchanged. Base and head list the same 24,872 targets, byte-identical to your listing at `2e99b1cf` (`ff5e`, sha `fff790b5…`).
- The checks have not run, under 64 GB. A `--all` shard's main process holds about 53 GiB before any target, at any `-j`, and targets of about 34 GiB exist that `heavy.json` doesn't list. So no shard fits under 56. I stopped there, and I'm not writing a selector.
- Options, for you: shards declared about 96 GB, or per-binding `--binding NAME --slice` runs after one `--records-only --slices` [not tried].

**After the move** [the lines read; the rewrite inferred]. These are the lines of circuit-check's `src/` that name a module chain 2's served map moves:
- without #9660: 13 imports and 2 code strings (`_ENUMERATORS`, `PREBUILD`);
- with #9660: none in code, only three comment lines and `heavy.json`'s 23 unit names.

**Not done.**
- No restack and no merge of `main` past #9364: both wait on your word that both chains are on `main`.
- The vLLM suite and the catalog at carried heads.
- Nothing marked.

The run ids are `r20261113-083621-` followed by `560e`, `ffde`, `667c`, `f36b` or `d445`, and `r20261113-092500-8212` and `r20261113-092459-d564`. `hooks.md` is on `cursor/circuits-cir269-scratch-6f04`, from `1f21de93d816` on, beside the checkpoints and the listings.
