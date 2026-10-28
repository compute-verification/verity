
## 17:31Z
Post Linear comments only with `research msg linear comment ISSUE "text" --as soundness`, never the Linear MCP connector. The connector posts as Daniel.

## 17:47Z (changes your task)
Daniel corrected the lifecycle at 17:42Z (full wording on Linear TOP-137, 17:50Z):
- The registered compiler B runs at the start of the verification phase, after the trace is committed.
- Its output is an execution block, partitioned into inference blocks, each partitioned into proof blocks.
- The developer proves it ran B correctly on committed advice of at most l_a bits (a public parameter), and that the output explains the committed trace: each proof block's inputs and outputs are committed values.
- The coins come after.
So the restated Declaration is: committed trace, then B(advice) as the execution block with its two-level partition, then a proof that B ran correctly on advice of at most l_a bits, and that the block explains the trace. The circuit and partition are B's output. Add l_a as a public parameter, and add no second description. Tell INFONLY (#6313) when the "explains the trace" link exists, since its tie reads it.

## 18:52Z
lean says the shape works for Core. Four points to fix before it builds:
(1) One object: `Verity.Compiler` must be what the registration log's builder entry (#6231) names, with θ read from the log. Tie the two here (the compiler is read from the log's builder entry), not as a second compiler definition.
(2) Registered.lean says advice comes at each verification phase, but Daniel's lifecycle puts the reported advice in the execution phase (the compiler runs on it at the start of the verification phase). Fix the docstrings and any field names.
(3) `Compiler.Output.pos`: require it injective on committed gates, or say why `Explains` doesn't need it. Injective is likely needed so two proof-block gates can't claim one committed value.
(4) Rename to `RegisteredCircuit` rather than `Registered` (comms' list), with --moved.
If your records runs already started, cancel and resend after these edits; they'd only produce records of the wrong shape.

## 19:27Z
If your branch was cut from origin/next, it carries every queued PR on next, including ones since withdrawn, such as lean's #6043. Its Properties/Core/Site.lean breaks Properties records (lean, 19:24Z). Drive holds a PR that contains a withdrawn or outdated head, and the records gate needs runs over all of it. Before you mark, rebuild your branch on origin/main plus only the PR heads you depend on, as DEDUP did for #6340 and #6341. Push it as a new branch and update the PR head. Then rerun records once. If you already depend only on main plus named heads, ignore this.

## 20:46Z
New capacity until Thu 06:00Z: vy-nebius-cpu-11 and -12, each 192 vCPU with 6 warm Lean build slots. Use `--on vy-nebius-cpu-11 --pool research` (r0-r4; there is no soundness pool there). Send NEW runs there first. Don't cancel a run that has already started elsewhere, but a queued, not-started run 8+ deep on cpu-7/8/9 may be cancelled and resent there.
